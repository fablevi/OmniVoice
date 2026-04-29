#!/usr/bin/env python3
"""Seed-locked side-by-side audio generation across all three backends.

Each variant runs with the *same* RNG state at the start, so any audible
difference is from the runtime / quantisation, not from the diffusion
loop's Gumbel sampling diverging by chance. Writes one WAV per (variant,
prompt) into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("seed-locked")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

from onnx_driver import (  # noqa: E402
    DEFAULT_FP32, DEFAULT_INT8, StepStats, install_onnx_forward, make_session,
)

PROMPTS = {
    "fox":   "The quick brown fox jumps over the lazy dog. "
             "Pack my box with five dozen liquor jugs. "
             "How vexingly quick daft zebras jump.",
    "short": "Hello world.",
    "zh":    "你好世界，今天天气真好。",
}
INSTRUCT = "Female, Middle-aged, Moderate Pitch"  # nova preset


def lock_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def restore_pytorch_forward(model: OmniVoice) -> None:
    if "forward" in model.__dict__:
        del model.__dict__["forward"]


def gen(model: OmniVoice, text: str, num_step: int, seed: int) -> tuple[np.ndarray, float, int]:
    lock_seed(seed)
    cfg = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )
    t0 = time.perf_counter()
    audios = model.generate(
        text=text, language=None, instruct=INSTRUCT, generation_config=cfg,
    )
    wall = time.perf_counter() - t0
    waveform = np.asarray(audios[0], dtype=np.float32)
    return waveform, wall, model.sampling_rate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--out-dir", default="/tmp/omnivoice_bench_v2")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch model ...")
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()

    rows: list[tuple[str, str, float, float, float]] = []  # variant, prompt, wall, audio, rtf

    for prompt_tag, text in PROMPTS.items():
        log.info("=== prompt: %s ===", prompt_tag)

        # --- PyTorch fp32 ---
        log.info("[PyTorch fp32] seed=%d", args.seed)
        wav, wall, sr = gen(model, text, args.num_step, args.seed)
        out = out_dir / f"{prompt_tag}_pytorch_fp32_seed{args.seed}.wav"
        sf.write(str(out), wav, sr, subtype="PCM_16")
        audio_s = wav.shape[-1] / sr
        rows.append(("pytorch_fp32", prompt_tag, wall, audio_s, wall / audio_s))
        log.info("  wrote %s (%.2fs audio in %.2fs, RTF=%.3f)", out, audio_s, wall, wall / audio_s)

        # --- ONNX fp32 ---
        sess = make_session(DEFAULT_FP32, threads=args.threads)
        stats = StepStats()
        install_onnx_forward(model, sess, stats)
        log.info("[ONNX fp32] seed=%d", args.seed)
        wav, wall, sr = gen(model, text, args.num_step, args.seed)
        out = out_dir / f"{prompt_tag}_onnx_fp32_seed{args.seed}.wav"
        sf.write(str(out), wav, sr, subtype="PCM_16")
        audio_s = wav.shape[-1] / sr
        rows.append(("onnx_fp32", prompt_tag, wall, audio_s, wall / audio_s))
        log.info("  wrote %s (%.2fs audio in %.2fs, RTF=%.3f)", out, audio_s, wall, wall / audio_s)
        restore_pytorch_forward(model)
        del sess

        # --- ONNX int8 ---
        sess = make_session(DEFAULT_INT8, threads=args.threads)
        stats = StepStats()
        install_onnx_forward(model, sess, stats)
        log.info("[ONNX int8] seed=%d", args.seed)
        wav, wall, sr = gen(model, text, args.num_step, args.seed)
        out = out_dir / f"{prompt_tag}_onnx_int8_seed{args.seed}.wav"
        sf.write(str(out), wav, sr, subtype="PCM_16")
        audio_s = wav.shape[-1] / sr
        rows.append(("onnx_int8", prompt_tag, wall, audio_s, wall / audio_s))
        log.info("  wrote %s (%.2fs audio in %.2fs, RTF=%.3f)", out, audio_s, wall, wall / audio_s)
        restore_pytorch_forward(model)
        del sess

    print("\n=== Summary ===")
    print(f"{'variant':<14} {'prompt':<8} {'wall':>7} {'audio':>7} {'RTF':>7}")
    for v, prm, wall, audio_s, rtf in rows:
        print(f"{v:<14} {prm:<8} {wall:>6.2f}s {audio_s:>6.2f}s {rtf:>7.3f}")

    print(f"\nWAVs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
