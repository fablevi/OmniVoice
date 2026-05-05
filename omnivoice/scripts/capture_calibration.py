#!/usr/bin/env python3
# Copyright    2026  Scott Yeager
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture diffusion-LM step inputs for OpenVINO/NNCF calibration.

Runs OmniVoice generation on a small diverse prompt corpus and dumps every
forward-pass input tensor to ``.npz`` files. The dumps cover both the
conditional and unconditional CFG batch rows, every diffusion timestep, and a
mix of short/long, English/Chinese prompts so the activation distribution is
representative of real inference.

Used as input to ``omnivoice.scripts.quantize_openvino``.

Usage:
    python -m omnivoice.scripts.capture_calibration \
        --model k2-fsa/OmniVoice \
        --output_dir calibration/
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import OmniVoiceModelOutput

# 10 prompts spanning short/long, EN/ZH, and varying voice-design instructs.
# 10 prompts x 16 diffusion steps = 160 calibration samples per run.
CORPUS = [
    ("short_en_alloy",   "Hello world.",
                          "Male, Young Adult, Low Pitch"),
    ("short_en_fable",   "Good morning everyone.",
                          "Female, Young Adult, High Pitch"),
    ("short_en_onyx",    "Welcome to the show.",
                          "Male, Elderly, Very Low Pitch"),
    ("med_en_nova",      "The weather is nice today, isn't it?",
                          "Female, Middle-aged, Moderate Pitch"),
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
    ("short_zh",         "你好世界，今天天气真好。",
                          "Female, Young Adult, High Pitch"),
    ("med_zh",           "欢迎收听今天的天气预报，明天将有小雨。",
                          "Female, Middle-aged, Moderate Pitch"),
    ("long_zh",          "在很久很久以前，有一个小村庄住着一位善良的老人，他每天都会去山上砍柴。",
                          "Male, Middle-aged, Low Pitch"),
]


def install_capture_hook(
    model: OmniVoice,
    out_dir: Path,
    tag: str,
) -> None:
    """Replace ``model.forward`` with a wrapper that dumps inputs and then
    delegates to the original PyTorch forward."""
    orig_forward = type(model).forward
    state = {"step": 0}

    def forward(
        self_, input_ids, audio_mask, labels=None,
        attention_mask=None, document_ids=None, position_ids=None,
    ):
        if attention_mask is None:
            raise RuntimeError("Need 4D attention_mask for calibration capture")
        B, _, S = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
            )
        np.savez(
            out_dir / f"{tag}_step{state['step']:04d}.npz",
            input_ids=input_ids.detach().cpu().numpy(),
            audio_mask=audio_mask.detach().cpu().numpy(),
            attention_mask=attention_mask.detach().cpu().numpy(),
            position_ids=position_ids.detach().cpu().numpy(),
        )
        state["step"] += 1
        out = orig_forward(
            self_,
            input_ids=input_ids,
            audio_mask=audio_mask,
            labels=labels,
            attention_mask=attention_mask,
            document_ids=document_ids,
            position_ids=position_ids,
        )
        if isinstance(out, OmniVoiceModelOutput):
            return out
        return OmniVoiceModelOutput(loss=None, logits=out)

    model.forward = types.MethodType(forward, model)


def restore_forward(model: OmniVoice) -> None:
    if "forward" in model.__dict__:
        del model.__dict__["forward"]


def lock_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture OmniVoice step inputs for OV/NNCF calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="calibration",
        help="Directory to write '.npz' calibration files.",
    )
    parser.add_argument(
        "--num_step",
        type=int,
        default=16,
        help="Diffusion steps per prompt; total samples = len(corpus) * num_step.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="CFG scale for capture; matches typical inference settings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed; reset before each prompt for reproducibility.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU threads used during capture.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe output_dir before capturing.",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    out_dir = Path(args.output_dir).resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)

    logging.info(f"Loading {args.model} on cpu ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    logging.info(f"Loaded in {time.time() - t0:.1f}s")

    total_t0 = time.perf_counter()
    for tag, text, instruct in CORPUS:
        logging.info(f"[{tag}] generating + capturing ...")
        install_capture_hook(model, out_dir=out_dir, tag=tag)
        lock_seed(args.seed)
        cfg = OmniVoiceGenerationConfig(
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )
        t0 = time.perf_counter()
        model.generate(
            text=text, language=None, instruct=instruct, generation_config=cfg,
        )
        logging.info(f"  done in {time.perf_counter() - t0:.1f}s")
        restore_forward(model)

    total = time.perf_counter() - total_t0
    n_files = len(list(out_dir.glob("*.npz")))
    on_disk = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    logging.info(
        f"captured {n_files} calibration samples ({on_disk / 1e6:.1f} MB) "
        f"in {total:.1f}s to {out_dir}"
    )


if __name__ == "__main__":
    sys.exit(main())
