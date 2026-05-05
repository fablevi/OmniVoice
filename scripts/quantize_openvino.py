#!/usr/bin/env python3
"""Quantize the OmniVoice diffusion-LM step to OpenVINO int8 via NNCF.

Uses calibration/*.npz captured by the existing ONNX driver. Saves to
openvino_ir/omnivoice-step.int8.xml (+ .bin) by default.
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
    p.add_argument("--out", default=str(REPO / "openvino_ir/omnivoice-step.int8.xml"))
    p.add_argument("--samples", default=str(REPO / "calibration"))
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--preset", default="performance",
                   choices=["performance", "mixed"])
    p.add_argument("--mode", default="int8",
                   choices=["int8", "int8_sym", "int8_mixed"])
    p.add_argument("--smooth-quant-alpha", type=float, default=None,
                   help="Override SmoothQuant alpha for MatMul (default: 0.95)")
    p.add_argument("--accurate-bias", action="store_true",
                   help="Use slower, more accurate bias correction")
    p.add_argument("--no-transformer", action="store_true",
                   help="Disable model_type=TRANSFORMER")
    p.add_argument("--exclude", nargs="*", default=["/audio_heads/MatMul"],
                   help="Node names to exclude from quantization")
    args = p.parse_args()

    core = ov.Core()
    print(f"Loading model: {args.ir}")
    model = core.read_model(args.ir)

    print(f"Building calibration dataset (max {args.max_samples} samples)...")
    dataset = nncf.Dataset(
        list(calib_iter(Path(args.samples), args.max_samples))
    )

    preset = (nncf.QuantizationPreset.PERFORMANCE if args.preset == "performance"
              else nncf.QuantizationPreset.MIXED)

    from nncf.quantization.advanced_parameters import (
        AdvancedQuantizationParameters,
        AdvancedSmoothQuantParameters,
    )
    advanced = None
    if args.smooth_quant_alpha is not None:
        advanced = AdvancedQuantizationParameters(
            smooth_quant_alphas=AdvancedSmoothQuantParameters(
                matmul=args.smooth_quant_alpha,
            ),
        )

    print(
        f"NNCF quantize: preset={args.preset} samples={args.max_samples} "
        f"transformer={'no' if args.no_transformer else 'yes'} "
        f"accurate_bias={args.accurate_bias} "
        f"sq_alpha={args.smooth_quant_alpha} "
        f"exclude={args.exclude}"
    )

    ignored = nncf.IgnoredScope(types=["Gather"], names=list(args.exclude))
    quantized = nncf.quantize(
        model,
        dataset,
        preset=preset,
        model_type=(None if args.no_transformer else nncf.ModelType.TRANSFORMER),
        ignored_scope=ignored,
        subset_size=args.max_samples,
        fast_bias_correction=not args.accurate_bias,
        advanced_parameters=advanced,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(out), compress_to_fp16=False)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
