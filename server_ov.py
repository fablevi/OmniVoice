#!/usr/bin/env python3
"""OpenAI-compatible TTS server backed by OpenVINO int8.

Mirrors server.py but installs the OpenVINO driver instead of running the
LLM in PyTorch. Per-shape compiled-model cache so static-reshape pays off
across repeated requests of the same prompt length.

Usage:
    python server_ov.py [--ir openvino_ir/omnivoice-step.int8_t160.xml]
                        [--port 8311] [--threads 8] [--cache-dir /tmp/ov_cache]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import types
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import OmniVoiceModelOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ov-server")

VOICE_INSTRUCTS = {
    "alloy":   "Male, Young Adult, Low Pitch",
    "echo":    "Male, Middle-aged, Moderate Pitch, British Accent",
    "fable":   "Female, Young Adult, High Pitch",
    "onyx":    "Male, Elderly, Very Low Pitch",
    "nova":    "Female, Middle-aged, Moderate Pitch",
    "shimmer": "Female, Young Adult, Very High Pitch, American Accent",
}

FORMAT_MEDIA_TYPE = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
    "ogg": "audio/ogg", "opus": "audio/opus", "aac": "audio/aac",
}


class SpeechRequest(BaseModel):
    model: str = Field(default="omnivoice")
    input: str
    voice: str = "nova"
    instructions: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0


class State:
    model: Optional[OmniVoice] = None
    core: Optional[ov.Core] = None
    ir_path: Optional[Path] = None
    threads: int = 8
    static_reshape: bool = True
    compiled_cache: dict = {}  # (B, S) -> (compiled_model, request)
    cache_lock_count: int = 0


state = State()


def encode_audio(waveform: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "wav":
        out = BytesIO()
        sf.write(out, waveform, sr, format="WAV", subtype="PCM_16")
        return out.getvalue()
    pcm16 = (np.clip(waveform, -1, 1) * 32767).astype(np.int16).tobytes()
    seg = AudioSegment(pcm16, sample_width=2, frame_rate=sr, channels=1)
    out = BytesIO()
    seg.export(out, format=fmt)
    return out.getvalue()


def get_compiled(B: int, S: int):
    key = (B, S)
    if key in state.compiled_cache:
        return state.compiled_cache[key]
    log.info("Compiling OV model for shape B=%d S=%d ...", B, S)
    t0 = time.time()
    m = state.core.read_model(str(state.ir_path))
    m.reshape({
        "input_ids": [B, 8, S],
        "audio_mask": [B, S],
        "attention_mask": [B, 1, S, S],
        "position_ids": [B, S],
    })
    compiled = state.core.compile_model(m, "CPU", {
        "INFERENCE_NUM_THREADS": state.threads,
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": "YES",
    })
    log.info("  done in %.2fs", time.time() - t0)
    request = compiled.create_infer_request()
    state.compiled_cache[key] = (compiled, request)
    return compiled, request


def install_ov_forward(model: OmniVoice):
    """Replace model.forward with an OV-backed version that uses the per-shape cache."""

    # Pre-compile dynamic-shape fallback for first request of an unseen shape
    log.info("Reading OV IR (dynamic): %s", state.ir_path)
    m_dyn = state.core.read_model(str(state.ir_path))
    compiled_dyn = state.core.compile_model(m_dyn, "CPU", {
        "INFERENCE_NUM_THREADS": state.threads,
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": "YES",
    })
    request_dyn = compiled_dyn.create_infer_request()
    out_port_dyn = compiled_dyn.outputs[0]

    def forward(self_, input_ids, audio_mask, labels=None,
                attention_mask=None, document_ids=None, position_ids=None):
        if attention_mask is None:
            raise RuntimeError("OV server requires 4D attention_mask")
        B, _, S = input_ids.shape
        if state.static_reshape:
            try:
                compiled, request = get_compiled(B, S)
                out_port = compiled.outputs[0]
            except Exception as e:
                log.warning("Static-reshape failed (%s); falling back to dynamic.", e)
                compiled, request, out_port = compiled_dyn, request_dyn, out_port_dyn
        else:
            compiled, request, out_port = compiled_dyn, request_dyn, out_port_dyn
        feeds = {
            "input_ids": input_ids.detach().cpu().contiguous().numpy(),
            "audio_mask": audio_mask.detach().cpu().contiguous().numpy(),
            "attention_mask": attention_mask.detach().cpu().contiguous().numpy(),
        }
        if "position_ids" in {i.any_name for i in compiled.inputs}:
            if position_ids is None:
                position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
            feeds["position_ids"] = position_ids.detach().cpu().contiguous().numpy()
        out = request.infer(feeds)
        return OmniVoiceModelOutput(
            loss=None,
            logits=torch.from_numpy(np.ascontiguousarray(out[out_port])),
        )

    model.forward = types.MethodType(forward, model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ir = os.environ.get("OMNIVOICE_OV_IR", "openvino_ir/omnivoice-step.int8_t160.xml")
    cache_dir = os.environ.get("OMNIVOICE_OV_CACHE", "/tmp/ov_cache")
    threads = int(os.environ.get("OMNIVOICE_THREADS", "8"))
    static = os.environ.get("OMNIVOICE_STATIC_RESHAPE", "1") == "1"
    state.ir_path = Path(ir).resolve()
    state.threads = threads
    state.static_reshape = static
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    log.info("OV core init (threads=%d, static_reshape=%s, cache=%s)",
             threads, static, cache_dir)
    state.core = ov.Core()
    state.core.set_property({"CACHE_DIR": cache_dir})
    state.core.set_property("CPU", {"INFERENCE_NUM_THREADS": threads})

    log.info("Loading PyTorch OmniVoice (host helpers, audio_heads, Higgs) ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
        attn_implementation="sdpa",
    )
    model.to("cpu").eval()
    log.info("Loaded in %.1fs", time.time() - t0)

    install_ov_forward(model)
    state.model = model
    log.info("Server ready (OV IR: %s)", state.ir_path.name)
    yield
    state.model = None


app = FastAPI(title="OmniVoice OV TTS", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok",
            "model_loaded": state.model is not None,
            "compiled_shapes": list(state.compiled_cache.keys())}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": "omnivoice", "object": "model", "owned_by": "omnivoice"},
        {"id": "tts-1", "object": "model", "owned_by": "omnivoice"},
    ]}


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    if state.model is None:
        raise HTTPException(503, "model not loaded yet")
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(400, "`input` must not be empty")

    voice_key = req.voice.lower()
    if req.instructions:
        instruct = req.instructions.strip()
    elif voice_key in VOICE_INSTRUCTS:
        instruct = VOICE_INSTRUCTS[voice_key]
    else:
        raise HTTPException(400, f"unknown voice '{req.voice}'")

    num_step = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))
    gs = float(os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0"))
    if gs == 0.0:
        # Install CFG-skip patch on first cfg=0 request
        from cfg_skip_patch import install
        install(state.model)

    cfg = OmniVoiceGenerationConfig(
        num_step=num_step, guidance_scale=gs, denoise=True,
        preprocess_prompt=True, postprocess_output=True,
    )
    log.info("synthesize voice=%s fmt=%s chars=%d num_step=%d gs=%.1f",
             voice_key, req.response_format, len(text), num_step, gs)

    t0 = time.time()
    try:
        audios = state.model.generate(
            text=text, language=None, instruct=instruct,
            speed=float(req.speed) if req.speed and req.speed != 1.0 else None,
            generation_config=cfg,
        )
    except Exception as e:
        log.exception("generation failed")
        raise HTTPException(500, f"generation failed: {e}") from e
    if not audios:
        raise HTTPException(500, "no audio")
    waveform = np.asarray(audios[0], dtype=np.float32)
    sr = state.model.sampling_rate
    log.info("generated %.2fs audio in %.2fs (RTF=%.3f)",
             waveform.shape[-1] / sr, time.time() - t0,
             (time.time() - t0) / max(1e-9, waveform.shape[-1] / sr))

    body = encode_audio(waveform, sr, req.response_format)
    return Response(content=body, media_type=FORMAT_MEDIA_TYPE[req.response_format])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ir", default="openvino_ir/omnivoice-step.int8_t160.xml")
    p.add_argument("--port", type=int, default=8311)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--cache-dir", default="/tmp/ov_cache")
    p.add_argument("--no-static-reshape", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn
    a = parse_args()
    os.environ["OMNIVOICE_OV_IR"] = a.ir
    os.environ["OMNIVOICE_THREADS"] = str(a.threads)
    os.environ["OMNIVOICE_OV_CACHE"] = a.cache_dir
    os.environ["OMNIVOICE_STATIC_RESHAPE"] = "0" if a.no_static_reshape else "1"
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
