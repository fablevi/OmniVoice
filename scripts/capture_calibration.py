#!/usr/bin/env python3
"""Run the fp32 ONNX driver on a diverse prompt corpus and dump every
``sess.run`` input as ``.npz`` for static-QDQ calibration.

Each prompt produces ``num_step`` calibration samples (one per diffusion
step). With 10 prompts × 16 steps = 160 samples, calibration covers:

- Cond + uncond batch halves (the wrapper sees both per call).
- Early-step inputs (mostly mask tokens) and late-step inputs
  (nearly-finalised audio tokens) — the activation distribution shifts
  meaningfully across the schedule.
- Different sequence lengths (short/long English, Chinese).
- Different style/instruct token compositions.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capture-calibration")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

from onnx_driver import (  # noqa: E402
    DEFAULT_FP32, StepStats, install_onnx_forward, make_session,
)

# 10 prompts spanning short/long, EN/ZH, varying instruct strings.
CORPUS = [
    ("short_en_alloy",   "Hello world.", "Male, Young Adult, Low Pitch"),
    ("short_en_fable",   "Good morning everyone.", "Female, Young Adult, High Pitch"),
    ("short_en_onyx",    "Welcome to the show.", "Male, Elderly, Very Low Pitch"),
    ("med_en_nova",      "The weather is nice today, isn't it?", "Female, Middle-aged, Moderate Pitch"),
    ("med_en_echo",      "Please listen carefully to the following announcement.",
                          "Male, Middle-aged, Moderate Pitch, British Accent"),
    ("long_en_fox",      "The quick brown fox jumps over the lazy dog. "
                         "Pack my box with five dozen liquor jugs. "
                         "How vexingly quick daft zebras jump.",
                          "Female, Middle-aged, Moderate Pitch"),
    ("long_en_lincoln",  "Four score and seven years ago our fathers brought forth, "
                         "on this continent, a new nation, conceived in liberty, "
                         "and dedicated to the proposition that all men are created equal.",
                          "Male, Elderly, Low Pitch"),
    ("short_zh",         "你好世界，今天天气真好。", "Female, Young Adult, High Pitch"),
    ("med_zh",           "欢迎收听今天的天气预报，明天将有小雨。", "Female, Middle-aged, Moderate Pitch"),
    ("long_zh",          "在很久很久以前，有一个小村庄住着一位善良的老人，他每天都会去山上砍柴。",
                          "Male, Middle-aged, Low Pitch"),
]


def lock_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def restore_pytorch_forward(model: OmniVoice) -> None:
    if "forward" in model.__dict__:
        del model.__dict__["forward"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "calibration"))
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--clean", action="store_true",
                   help="Wipe out_dir before capturing")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_FP32.exists():
        log.error("fp32 ONNX missing: %s", DEFAULT_FP32)
        return 1

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch model ...")
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()

    sess = make_session(DEFAULT_FP32, threads=args.threads)
    stats = StepStats()

    total_t0 = time.perf_counter()
    for tag, text, instruct in CORPUS:
        log.info("[%s] generating + capturing ...", tag)
        # Re-install the forward hook per-prompt so capture_state["step"]
        # resets and files are deterministic per-prompt.
        install_onnx_forward(model, sess, stats, capture_dir=out_dir, capture_tag=tag)
        lock_seed(args.seed)
        cfg = OmniVoiceGenerationConfig(
            num_step=args.num_step, guidance_scale=2.0, denoise=True,
            preprocess_prompt=True, postprocess_output=True,
        )
        t0 = time.perf_counter()
        model.generate(text=text, language=None, instruct=instruct, generation_config=cfg)
        dt = time.perf_counter() - t0
        log.info("  done in %.1fs", dt)

    restore_pytorch_forward(model)
    total = time.perf_counter() - total_t0

    n_files = len(list(out_dir.glob("*.npz")))
    on_disk = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    log.info(
        "captured %d calibration samples (%.1f MB) in %.1fs to %s",
        n_files, on_disk / 1e6, total, out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
