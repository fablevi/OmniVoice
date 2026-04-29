#!/usr/bin/env python3
"""Side-by-side comparison of multiple ONNX quantization variants.

For each ``.onnx`` listed, runs a seed-locked synthesis on a fixed prompt
(default: fox pangram, voice = nova preset, num_step = 16, seed = 42),
writes a WAV, and prints a metrics table comparing each waveform against
the PyTorch fp32 reference.

Metrics:
- RTF (gen-only wall clock / audio length).
- ``corr`` = sample-wise Pearson correlation against the reference.
  PyTorch fp32 ↔ ONNX fp32 is corr=1.0; the lower this drops, the more
  the quantized variant has wandered.
- ``rms_diff`` normalised by reference RMS (≈ how loud the per-sample
  error is relative to the signal).
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
log = logging.getLogger("compare-quants")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

from onnx_driver import StepStats, install_onnx_forward, make_session  # noqa: E402

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)
INSTRUCT = "Female, Middle-aged, Moderate Pitch"  # nova preset


def lock_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def restore_pytorch_forward(model: OmniVoice) -> None:
    if "forward" in model.__dict__:
        del model.__dict__["forward"]


def gen(model: OmniVoice, num_step: int, seed: int) -> tuple[np.ndarray, float, int]:
    lock_seed(seed)
    cfg = OmniVoiceGenerationConfig(
        num_step=num_step, guidance_scale=2.0, denoise=True,
        preprocess_prompt=True, postprocess_output=True,
    )
    t0 = time.perf_counter()
    audios = model.generate(text=PROMPT, language=None, instruct=INSTRUCT, generation_config=cfg)
    wall = time.perf_counter() - t0
    waveform = np.asarray(audios[0], dtype=np.float32)
    return waveform, wall, model.sampling_rate


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    da = a - a.mean()
    db = b - b.mean()
    denom = float(np.linalg.norm(da) * np.linalg.norm(db))
    return float(np.dot(da, db) / denom) if denom > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--out-dir", default="/tmp/omnivoice_quant_v3")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--variants", nargs="+", default=[
        "pytorch_fp32",
        "onnx_fp32:onnx/omnivoice-step.onnx",
        "int8_baseline:onnx/omnivoice-step.int8.onnx",
        "int8_no_heads:onnx/omnivoice-step.int8.no_heads.onnx",
        "int8_perchannel:onnx/omnivoice-step.int8.perchannel.onnx",
        "uint8:onnx/omnivoice-step.uint8.onnx",
    ], help="<tag> for PyTorch, <tag>:<path> for ONNX variants")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch model ...")
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()

    rows = []
    pytorch_wave: np.ndarray | None = None
    sr = model.sampling_rate

    for spec in args.variants:
        if ":" in spec:
            tag, path = spec.split(":", 1)
            onnx_path = Path(path).resolve()
            if not onnx_path.exists():
                log.warning("missing %s — skipping", onnx_path)
                continue
            sess = make_session(onnx_path, threads=args.threads)
            stats = StepStats()
            install_onnx_forward(model, sess, stats)
        else:
            tag = spec
            onnx_path = None
            restore_pytorch_forward(model)

        log.info("[%s] generating ...", tag)
        wav, wall, sr_out = gen(model, args.num_step, args.seed)
        size_mb = 0.0
        if onnx_path is not None:
            size_mb = onnx_path.stat().st_size / 1e6
            for cand in (Path(str(onnx_path) + ".data"), Path(str(onnx_path) + "_data")):
                if cand.exists():
                    size_mb += cand.stat().st_size / 1e6
            restore_pytorch_forward(model)
            del sess

        out = out_dir / f"fox_{tag}_seed{args.seed}.wav"
        sf.write(str(out), wav, sr_out, subtype="PCM_16")
        audio_s = wav.shape[-1] / sr_out
        rtf = wall / audio_s

        if tag == "pytorch_fp32":
            pytorch_wave = wav

        rows.append({
            "tag": tag, "wall": wall, "audio_s": audio_s, "rtf": rtf,
            "wav": out, "size_mb": size_mb, "wave": wav,
        })
        log.info("  %s wall=%.2fs audio=%.2fs RTF=%.3f size=%.0f MB",
                 tag, wall, audio_s, rtf, size_mb)

    if pytorch_wave is None:
        log.warning("no pytorch_fp32 row — skipping correlation table")

    print("\n=== Comparison vs PyTorch fp32 (fox pangram, seed=%d) ===" % args.seed)
    header = f"{'variant':<18} {'size MB':>8} {'wall':>7} {'RTF':>7} {'corr':>8} {'rms_diff/rms':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if pytorch_wave is None or r["tag"] == "pytorch_fp32":
            c = 1.0 if r["tag"] == "pytorch_fp32" else float("nan")
            nrms = 0.0 if r["tag"] == "pytorch_fp32" else float("nan")
        else:
            n = min(len(pytorch_wave), len(r["wave"]))
            a = pytorch_wave[:n]
            b = r["wave"][:n]
            c = corr(a, b)
            rms_a = float(np.sqrt(np.mean(a * a)))
            rms_d = float(np.sqrt(np.mean((a - b) ** 2)))
            nrms = rms_d / (rms_a + 1e-12)
        print(f"{r['tag']:<18} {r['size_mb']:>8.0f} {r['wall']:>6.2f}s {r['rtf']:>7.3f} "
              f"{c:>8.4f} {nrms:>13.3f}")

    print(f"\nWAVs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
