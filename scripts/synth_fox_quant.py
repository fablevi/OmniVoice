#!/usr/bin/env python3
"""Generate fox-pangram synthesis WAV for one ONNX variant + report RTF.

Reproducible: seed-locks Gumbel sampling so two runs of the same backend
produce bit-identical audio. Use this to A/B variants on the same starting
point.

Usage:
    python scripts/synth_fox_quant.py onnx/omnivoice-step.<variant>.onnx \
        --label w8a16_rtn128 --out samples/quant_v2/
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
log = logging.getLogger("synth-fox-quant")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402
from onnx_driver import StepStats, install_onnx_forward, make_session  # noqa: E402

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)
INSTRUCT = "Female, Middle-aged, Moderate Pitch"


def lock_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def synth(model: OmniVoice, num_step: int, seed: int):
    lock_seed(seed)
    cfg = OmniVoiceGenerationConfig(
        num_step=num_step, guidance_scale=2.0, denoise=True,
        preprocess_prompt=True, postprocess_output=True,
    )
    t0 = time.perf_counter()
    audios = model.generate(text=PROMPT, language=None, instruct=INSTRUCT,
                            generation_config=cfg)
    wall = time.perf_counter() - t0
    return np.asarray(audios[0], dtype=np.float32), wall


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx_path")
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="samples/quant_v2")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--n-runs", type=int, default=1)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    args = p.parse_args()

    onnx_path = Path(args.onnx_path).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch (host) ...")
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()

    # ORT external-data resolution: chdir into onnx parent for session create
    # to be safe across all variants.
    saved = os.getcwd()
    try:
        os.chdir(onnx_path.parent)
        sess = make_session(Path(onnx_path.name), threads=args.threads)
    finally:
        os.chdir(saved)

    stats = StepStats()
    install_onnx_forward(model, sess, stats)

    runs = []
    for i in range(args.n_runs):
        stats.reset()
        wav, wall = synth(model, args.num_step, args.seed + i)
        sr = model.sampling_rate
        audio_s = wav.shape[-1] / sr
        rtf = wall / audio_s if audio_s > 0 else float("inf")
        out = out_dir / f"{args.label}_seed{args.seed + i}.wav"
        sf.write(out, wav, sr, subtype="PCM_16")
        log.info("[%s run %d] wall=%.2fs audio=%.2fs RTF=%.3f mean_step=%.1fms → %s",
                 args.label, i + 1, wall, audio_s, rtf, stats.mean_ms, out.name)
        runs.append({"wall": wall, "audio_s": audio_s, "rtf": rtf, "mean_step": stats.mean_ms})

    if args.n_runs > 1:
        wall_avg = sum(r["wall"] for r in runs) / len(runs)
        rtf_avg = sum(r["rtf"] for r in runs) / len(runs)
        log.info("[%s avg over %d runs] wall=%.2fs RTF=%.3f",
                 args.label, args.n_runs, wall_avg, rtf_avg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
