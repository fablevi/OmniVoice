#!/usr/bin/env python3
"""OpenVINO driver for OmniVoice (voice-design mode).

Mirrors onnx_driver.py: loads the full PyTorch OmniVoice for tokenizer /
mask construction / Higgs decode, then monkey-patches model.forward so the
inner LLM + audio_heads call is served by OpenVINO instead.

Usage:
    python openvino_driver.py "Hello world." [--ir openvino_ir/omnivoice-step.xml]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import OmniVoiceModelOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ov-driver")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FP32 = REPO_ROOT / "openvino_ir" / "omnivoice-step.xml"
DEFAULT_INT8 = REPO_ROOT / "openvino_ir" / "omnivoice-step.int8_t.xml"

VOICE_INSTRUCTS = {
    "alloy":   "Male, Young Adult, Low Pitch",
    "echo":    "Male, Middle-aged, Moderate Pitch, British Accent",
    "fable":   "Female, Young Adult, High Pitch",
    "onyx":    "Male, Elderly, Very Low Pitch",
    "nova":    "Female, Middle-aged, Moderate Pitch",
    "shimmer": "Female, Young Adult, Very High Pitch, American Accent",
}


@dataclass
class StepStats:
    n_calls: int = 0
    total_seconds: float = 0.0

    def add(self, dt: float) -> None:
        self.n_calls += 1
        self.total_seconds += dt

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.total_seconds / max(1, self.n_calls)


def make_request(ir_path: Path, threads: int, static_shape=None) -> "ov.InferRequest":
    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": threads})
    log.info("Reading model: %s", ir_path)
    model = core.read_model(str(ir_path))
    if static_shape is not None:
        B, S = static_shape
        log.info("Reshaping model to static B=%d S=%d", B, S)
        model.reshape({
            "input_ids": [B, 8, S],
            "audio_mask": [B, S],
            "attention_mask": [B, 1, S, S],
            "position_ids": [B, S],
        })
    compiled = core.compile_model(model, "CPU", {
        "INFERENCE_NUM_THREADS": threads,
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": "YES",
    })
    log.info("Compiled. Inputs: %s", [i.any_name for i in compiled.inputs])
    return compiled.create_infer_request(), compiled


def install_openvino_forward(model: OmniVoice, request, compiled, stats: StepStats,
                             ir_path: Path = None, threads: int = 8,
                             static_reshape: bool = True) -> None:
    in_names = {i.any_name for i in compiled.inputs}
    out_port = compiled.outputs[0]
    needs_pos_ids = "position_ids" in in_names

    state = {"request": request, "compiled": compiled, "out_port": out_port,
             "shape_seen": None}

    def forward(self_, input_ids, audio_mask, labels=None,
                attention_mask=None, document_ids=None, position_ids=None):
        if attention_mask is None:
            raise RuntimeError("Need 4D attention_mask")
        B, _, S = input_ids.shape

        # On first call: recompile model with static shape (10-13% per-fwd faster)
        if static_reshape and state["shape_seen"] is None and ir_path is not None:
            state["shape_seen"] = (B, S)
            t0 = time.perf_counter()
            core = ov.Core()
            core.set_property("CPU", {"INFERENCE_NUM_THREADS": threads})
            new_model = core.read_model(str(ir_path))
            new_model.reshape({
                "input_ids": [B, 8, S],
                "audio_mask": [B, S],
                "attention_mask": [B, 1, S, S],
                "position_ids": [B, S],
            })
            new_compiled = core.compile_model(new_model, "CPU", {
                "INFERENCE_NUM_THREADS": threads,
                "INFERENCE_PRECISION_HINT": "f32",
                "PERFORMANCE_HINT": "LATENCY",
            })
            state["request"] = new_compiled.create_infer_request()
            state["compiled"] = new_compiled
            state["out_port"] = new_compiled.outputs[0]
            log.info("Recompiled with static shape B=%d S=%d in %.2fs",
                     B, S, time.perf_counter() - t0)

        feeds = {
            "input_ids": input_ids.detach().cpu().contiguous().numpy(),
            "audio_mask": audio_mask.detach().cpu().contiguous().numpy(),
            "attention_mask": attention_mask.detach().cpu().contiguous().numpy(),
        }
        if needs_pos_ids:
            if position_ids is None:
                position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
            feeds["position_ids"] = position_ids.detach().cpu().contiguous().numpy()

        t0 = time.perf_counter()
        out = state["request"].infer(feeds)
        stats.add(time.perf_counter() - t0)
        logits = torch.from_numpy(np.ascontiguousarray(out[state["out_port"]]))
        return OmniVoiceModelOutput(loss=None, logits=logits)

    model.forward = types.MethodType(forward, model)


def synthesize(model, text, instruct, num_step, speed, guidance_scale=2.0):
    cfg = OmniVoiceGenerationConfig(
        num_step=num_step, guidance_scale=guidance_scale, denoise=True,
        preprocess_prompt=True, postprocess_output=True,
    )
    audios = model.generate(
        text=text, language=None, instruct=instruct,
        speed=speed, generation_config=cfg,
    )
    return np.asarray(audios[0], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--voice", default="nova", choices=sorted(VOICE_INSTRUCTS))
    p.add_argument("--instruct", default=None)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--ir", default=None)
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--seed", type=int, default=None,
                   help="Set torch.manual_seed before generation")
    p.add_argument("--guidance-scale", type=float, default=2.0,
                   help="CFG scale; 0.0 disables CFG (cond-only)")
    p.add_argument("--static-reshape", action="store_true",
                   help="Recompile with static shapes on first forward "
                        "(10-13%% faster per-step but pays 1.8s recompile; "
                        "useful only when model+session is reused across requests)")
    args = p.parse_args()

    ir_path = Path(args.ir).resolve() if args.ir else (
        DEFAULT_INT8 if args.int8 else DEFAULT_FP32).resolve()
    if not ir_path.exists():
        log.error("IR file not found: %s", ir_path)
        return 1

    torch.set_num_threads(args.threads)
    log.info("Loading PyTorch OmniVoice for host helpers ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    log.info("PyTorch model loaded in %.1fs", time.time() - t0)

    request, compiled = make_request(ir_path, args.threads)
    stats = StepStats()
    install_openvino_forward(model, request, compiled, stats,
                             ir_path=ir_path, threads=args.threads,
                             static_reshape=args.static_reshape)

    if args.guidance_scale == 0.0:
        import cfg_skip_patch
        cfg_skip_patch.install(model)
        log.info("CFG-skip patch installed (guidance_scale=0)")

    instruct = args.instruct or VOICE_INSTRUCTS[args.voice]
    log.info("synthesize voice=%s chars=%d num_step=%d", args.voice, len(args.text), args.num_step)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    waveform = synthesize(model, args.text, instruct, args.num_step, args.speed,
                          guidance_scale=args.guidance_scale)
    wall = time.perf_counter() - t0

    sr = model.sampling_rate
    audio_seconds = waveform.shape[-1] / sr
    rtf = wall / audio_seconds if audio_seconds > 0 else float("inf")

    sf.write(args.out, waveform, sr, subtype="PCM_16")
    log.info("generated %.2fs audio in %.2fs wall, RTF=%.3f | LLM steps=%d, mean=%.1f ms",
             audio_seconds, wall, rtf, stats.n_calls, stats.mean_ms)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
