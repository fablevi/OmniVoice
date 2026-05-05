#!/usr/bin/env python3
"""Weight-only compression of the OmniVoice OV IR via NNCF.

Unlike scripts/quantize_openvino.py (full W8A8 PTQ), this uses
nncf.compress_weights() which only quantizes the weight tensors. Activations
stay fp32. Targets: int8_asym, int4_asym, nf4, etc.

The goal is W8A16/W4A16-tier quality with whatever kernel acceleration
OV provides for the corresponding op (which we'll measure).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import nncf

REPO = Path(__file__).resolve().parent.parent


def calib_iter(samples_dir: Path, max_samples: int):
    files = sorted(p for p in samples_dir.glob("*.npz")
                   if p.name != "activation_max.npz")
    if max_samples:
        files = files[:max_samples]
    for f in files:
        d = np.load(f)
        yield {
            "input_ids": d["input_ids"],
            "audio_mask": d["audio_mask"],
            "attention_mask": d["attention_mask"],
            "position_ids": d["position_ids"],
        }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ir", default=str(REPO / "openvino_ir/omnivoice-step.xml"))
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="int8_asym",
                   choices=["int8_sym", "int8_asym", "int4_sym",
                            "int4_asym", "nf4"])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--ratio", type=float, default=1.0,
                   help="Fraction of layers compressed; 1.0 = all")
    p.add_argument("--samples", default=str(REPO / "calibration"))
    p.add_argument("--max-samples", type=int, default=160)
    p.add_argument("--awq", action="store_true",
                   help="Apply AWQ pre-pass (needs calibration data)")
    p.add_argument("--scale-est", action="store_true",
                   help="Use scale-estimation algorithm (needs calibration)")
    args = p.parse_args()

    core = ov.Core()
    print(f"Loading model: {args.ir}")
    model = core.read_model(args.ir)

    mode_map = {
        "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
        "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
        "int4_sym": nncf.CompressWeightsMode.INT4_SYM,
        "int4_asym": nncf.CompressWeightsMode.INT4_ASYM,
        "nf4": nncf.CompressWeightsMode.NF4,
    }

    kwargs = {
        "mode": mode_map[args.mode],
        "ratio": args.ratio,
    }
    if args.mode != "int8_asym" and args.mode != "int8_sym":
        kwargs["group_size"] = args.group_size

    if args.awq or args.scale_est:
        ds = nncf.Dataset(list(calib_iter(Path(args.samples), args.max_samples)))
        kwargs["dataset"] = ds
        kwargs["awq"] = args.awq
        kwargs["scale_estimation"] = args.scale_est

    print(f"compress_weights: mode={args.mode} ratio={args.ratio} "
          f"group_size={args.group_size} awq={args.awq} scale_est={args.scale_est}")
    compressed = nncf.compress_weights(model, **kwargs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(compressed, str(out), compress_to_fp16=False)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
