#!/usr/bin/env python3
"""OpenAI-compatible TTS server backed by OmniVoice (voice design mode).

Endpoint: POST /v1/audio/speech
Run:      python server.py [--model k2-fsa/OmniVoice] [--port 8311] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("omnivoice-server")

VOICE_INSTRUCTS: dict[str, str] = {
    "alloy":   "Male, Young Adult, Low Pitch",
    "echo":    "Male, Middle-aged, Moderate Pitch, British Accent",
    "fable":   "Female, Young Adult, High Pitch",
    "onyx":    "Male, Elderly, Very Low Pitch",
    "nova":    "Female, Middle-aged, Moderate Pitch",
    "shimmer": "Female, Young Adult, Very High Pitch, American Accent",
}

ResponseFormat = Literal["wav", "mp3", "opus", "aac", "flac", "pcm"]
FORMAT_MEDIA_TYPE: dict[str, str] = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "opus": "audio/ogg",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "pcm":  "application/octet-stream",
}


class SpeechRequest(BaseModel):
    model: str = Field(default="omnivoice")
    input: str
    voice: str = Field(default="alloy")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: Optional[str] = None  # OpenAI gpt-4o-mini-tts param; ignored


class AppState:
    model: Optional[OmniVoice] = None


state = AppState()


def encode_audio(samples: np.ndarray, sr: int, fmt: str) -> bytes:
    """Encode a float32 mono waveform to the requested container/codec."""
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)

    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="FLAC")
        return buf.getvalue()

    if fmt == "pcm":
        # OpenAI spec: 24kHz signed 16-bit little-endian, mono, headerless.
        # Resample if model rate differs.
        if sr != 24000:
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sr, target_sr=24000)
            except Exception as e:  # noqa: BLE001
                log.warning("PCM resample failed (%s); returning at native %dHz", e, sr)
        pcm16 = (samples * 32767.0).astype(np.int16, copy=False)
        return pcm16.tobytes()

    # mp3 / opus / aac via pydub+ffmpeg
    pcm16 = (samples * 32767.0).astype(np.int16, copy=False)
    seg = AudioSegment(
        pcm16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    out = io.BytesIO()
    if fmt == "mp3":
        seg.export(out, format="mp3", bitrate="128k")
    elif fmt == "opus":
        seg.export(out, format="opus", bitrate="64k")
    elif fmt == "aac":
        seg.export(out, format="adts", bitrate="128k")
    else:
        raise ValueError(f"unsupported format: {fmt}")
    return out.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    device = os.environ.get("OMNIVOICE_DEVICE", "cpu")
    log.info("Loading OmniVoice model %s on %s ...", checkpoint, device)
    t0 = time.time()
    model = OmniVoice.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    state.model = model
    log.info(
        "Model loaded in %.1fs (sampling_rate=%d, device=%s)",
        time.time() - t0,
        model.sampling_rate,
        device,
    )
    yield
    state.model = None


app = FastAPI(title="OmniVoice OpenAI-compatible TTS", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.model is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "omnivoice", "object": "model", "owned_by": "omnivoice"},
            {"id": "tts-1", "object": "model", "owned_by": "omnivoice"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "omnivoice"},
            {"id": "gpt-4o-mini-tts", "object": "model", "owned_by": "omnivoice"},
        ],
    }


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`input` must not be empty")

    voice_key = req.voice.lower()
    custom = (req.instructions or "").strip()
    if custom:
        # Free-form override: pass straight through to OmniVoice as `instruct`.
        instruct = custom
    elif voice_key in VOICE_INSTRUCTS:
        instruct = VOICE_INSTRUCTS[voice_key]
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown voice '{req.voice}'. "
                f"supported presets: {sorted(VOICE_INSTRUCTS)}. "
                "Or pass free-form attributes via the `instructions` field."
            ),
        )

    num_step = int(os.environ.get("OMNIVOICE_NUM_STEP", "16"))
    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    log.info(
        "synthesize voice=%s fmt=%s speed=%.2f chars=%d num_step=%d instruct=%r",
        voice_key, req.response_format, req.speed, len(text), num_step, instruct,
    )

    t0 = time.time()
    try:
        audios = state.model.generate(
            text=text,
            language=None,  # auto-detect
            instruct=instruct,
            speed=float(req.speed) if req.speed and req.speed != 1.0 else None,
            generation_config=gen_config,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"generation failed: {e}") from e

    if not audios:
        raise HTTPException(status_code=500, detail="no audio returned")
    waveform = np.asarray(audios[0], dtype=np.float32)
    sr = state.model.sampling_rate
    log.info(
        "generated %.2fs of audio in %.1fs",
        waveform.shape[-1] / sr, time.time() - t0,
    )

    try:
        body = encode_audio(waveform, sr, req.response_format)
    except Exception as e:  # noqa: BLE001
        log.exception("encoding failed")
        raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e

    return Response(
        content=body,
        media_type=FORMAT_MEDIA_TYPE[req.response_format],
        headers={"Content-Disposition": f'inline; filename="speech.{req.response_format}"'},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible TTS server (OmniVoice)")
    p.add_argument("--model", default=os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice"))
    p.add_argument("--device", default=os.environ.get("OMNIVOICE_DEVICE", "cpu"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8311)
    p.add_argument("--threads", type=int, default=None,
                   help="torch.set_num_threads (CPU). Defaults to torch's choice.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["OMNIVOICE_MODEL"] = args.model
    os.environ["OMNIVOICE_DEVICE"] = args.device

    if args.threads:
        torch.set_num_threads(args.threads)
        log.info("torch threads = %d", args.threads)

    log.info("starting server on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
