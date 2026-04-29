#!/usr/bin/env python3
"""Benchmark PyTorch fp32, ONNX fp32, and ONNX int8 on the fox pangram.

Same workload as PERFORMANCE.md: 122-char English prompt, voice = nova
preset, num_step = 16, 8 threads, voice-design mode.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench-onnx")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

# Re-use the driver helpers so behaviour is identical to onnx_driver.py.
from onnx_driver import (  # noqa: E402
    DEFAULT_FP32, DEFAULT_INT8, StepStats, install_onnx_forward, make_session,
)

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)
INSTRUCT = "Female, Middle-aged, Moderate Pitch"  # nova preset


def time_one_run(
    model: OmniVoice, num_step: int, stats: StepStats | None,
    save_wav: Path | None = None,
) -> tuple[float, float, float]:
    """Run one synthesis, return (wall_seconds, audio_seconds, mean_step_ms)."""
    if stats is not None:
        stats.reset()
    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )
    t0 = time.perf_counter()
    audios = model.generate(
        text=PROMPT,
        language=None,
        instruct=INSTRUCT,
        generation_config=gen_config,
    )
    wall = time.perf_counter() - t0
    waveform = np.asarray(audios[0], dtype=np.float32)
    audio_seconds = waveform.shape[-1] / model.sampling_rate
    mean_step_ms = stats.mean_ms if stats is not None else 0.0
    if save_wav is not None:
        sf.write(str(save_wav), waveform, model.sampling_rate, subtype="PCM_16")
        log.info("    wrote %s", save_wav)
    return wall, audio_seconds, mean_step_ms


def restore_pytorch_forward(model: OmniVoice) -> None:
    """Drop the ONNX-backed monkey patch so model.forward goes back to the
    PyTorch implementation."""
    if "forward" in model.__dict__:
        del model.__dict__["forward"]


def bench(
    label: str, model: OmniVoice, num_step: int, stats: StepStats | None,
    runs: int = 3, wav_dir: Path | None = None, wav_tag: str | None = None,
) -> list[dict]:
    log.info("=== %s ===", label)
    out = []
    for i in range(1, runs + 1):
        save_wav = None
        if wav_dir is not None and wav_tag is not None:
            save_wav = wav_dir / f"bench_{wav_tag}_run{i}.wav"
        wall, audio, step_ms = time_one_run(model, num_step, stats, save_wav=save_wav)
        rtf = wall / audio if audio > 0 else float("inf")
        log.info(
            "  run %d: wall=%.2fs  audio=%.2fs  RTF=%.3f  mean_step=%.1f ms",
            i, wall, audio, rtf, step_ms,
        )
        out.append({"run": i, "wall": wall, "audio": audio, "rtf": rtf, "step_ms": step_ms})
    avg_wall = sum(r["wall"] for r in out) / runs
    avg_audio = sum(r["audio"] for r in out) / runs
    avg_step = sum(r["step_ms"] for r in out) / runs
    avg_rtf = avg_wall / avg_audio if avg_audio > 0 else float("inf")
    log.info(
        "  avg: wall=%.2fs  audio=%.2fs  RTF=%.3f  mean_step=%.1f ms",
        avg_wall, avg_audio, avg_rtf, avg_step,
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--skip-pytorch", action="store_true",
                   help="Skip the PyTorch fp32 baseline (use existing PERFORMANCE.md numbers)")
    p.add_argument("--skip-fp32", action="store_true", help="Skip ONNX fp32")
    p.add_argument("--skip-int8", action="store_true", help="Skip ONNX int8")
    p.add_argument("--wav-dir", default="/tmp",
                   help="Directory to write per-run WAVs (default: /tmp). "
                        "Set empty string to disable.")
    args = p.parse_args()
    wav_dir = Path(args.wav_dir).resolve() if args.wav_dir else None
    if wav_dir is not None:
        wav_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    log.info("torch threads = %d", torch.get_num_threads())

    log.info("Loading PyTorch OmniVoice ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    log.info("loaded in %.1fs", time.time() - t0)

    results: dict[str, list[dict]] = {}

    if not args.skip_pytorch:
        results["pytorch_fp32"] = bench(
            "PyTorch fp32 (baseline)", model, args.num_step, stats=None,
            runs=args.runs, wav_dir=wav_dir, wav_tag="pytorch_fp32",
        )

    if not args.skip_fp32:
        if not DEFAULT_FP32.exists():
            log.warning("skipping ONNX fp32: %s missing", DEFAULT_FP32)
        else:
            sess = make_session(DEFAULT_FP32, threads=args.threads)
            stats = StepStats()
            install_onnx_forward(model, sess, stats)
            results["onnx_fp32"] = bench(
                "ONNX fp32", model, args.num_step, stats=stats,
                runs=args.runs, wav_dir=wav_dir, wav_tag="onnx_fp32",
            )
            restore_pytorch_forward(model)
            del sess

    if not args.skip_int8:
        if not DEFAULT_INT8.exists():
            log.warning("skipping ONNX int8: %s missing", DEFAULT_INT8)
        else:
            sess = make_session(DEFAULT_INT8, threads=args.threads)
            stats = StepStats()
            install_onnx_forward(model, sess, stats)
            results["onnx_int8"] = bench(
                "ONNX int8", model, args.num_step, stats=stats,
                runs=args.runs, wav_dir=wav_dir, wav_tag="onnx_int8",
            )
            restore_pytorch_forward(model)
            del sess

    # Summary table.
    print("\n=== Summary ===")
    header = f"{'variant':<22} {'wall avg':>10} {'audio avg':>10} {'RTF avg':>10} {'step ms avg':>14}"
    print(header)
    print("-" * len(header))
    for label, runs in results.items():
        wall = sum(r["wall"] for r in runs) / len(runs)
        audio = sum(r["audio"] for r in runs) / len(runs)
        rtf = wall / audio if audio > 0 else float("inf")
        step_ms = sum(r["step_ms"] for r in runs) / len(runs)
        print(f"{label:<22} {wall:>9.2f}s {audio:>9.2f}s {rtf:>10.3f} {step_ms:>13.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
