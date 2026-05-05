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

"""Convert OmniVoice ONNX to OpenVINO IR and quantize to int8 with NNCF.

Two-step pipeline collapsed into one script:

    1. ``ov.convert_model(<onnx>)``        -> fp32 OpenVINO IR
    2. ``nncf.quantize(<fp32 ir>, ...)``   -> int8 OpenVINO IR

NNCF runs SmoothQuant with ``model_type=TRANSFORMER`` (smooths activation
outliers into adjacent weight matrices), collects activation ranges from the
calibration set, then per-channel-quantizes weights and inserts FakeQuantize
ops on activations. The resulting IR runs on OpenVINO's CPU plugin via oneDNN
AVX-VNNI int8 GEMM kernels with per-channel weight scales.

Defaults exclude embedding ``Gather`` ops and the ``audio_heads`` MatMul from
quantization — both empirically degrade quality without meaningful speedup.

Usage:
    python -m omnivoice.scripts.quantize_openvino \
        --onnx onnx/omnivoice-step.onnx \
        --calibration_dir calibration/ \
        --output openvino_ir/omnivoice-step-int8.xml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def calibration_iter(samples_dir: Path, max_samples: int):
    files = sorted(samples_dir.glob("*.npz"))
    if max_samples:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(
            f"No '.npz' calibration files found under {samples_dir}. "
            "Run `omnivoice.scripts.capture_calibration` first."
        )
    for f in files:
        d = np.load(f)
        yield {
            "input_ids": d["input_ids"],
            "audio_mask": d["audio_mask"],
            "attention_mask": d["attention_mask"],
            "position_ids": d["position_ids"],
        }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ONNX to OpenVINO IR and quantize to int8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Source ONNX model (will be converted to OV IR in-memory).",
    )
    src.add_argument(
        "--ir",
        type=str,
        default=None,
        help="Source OpenVINO IR (skip ONNX->IR conversion).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output quantized OpenVINO IR path (.xml).",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        required=True,
        help=(
            "Directory of '.npz' calibration files captured by "
            "omnivoice.scripts.capture_calibration."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=160,
        help="Cap on calibration samples to read (more = better but slower).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="performance",
        choices=["performance", "mixed"],
        help="NNCF preset; 'performance' = symmetric, 'mixed' = sym W / asym A.",
    )
    parser.add_argument(
        "--smooth_quant_alpha",
        type=float,
        default=None,
        help="Override SmoothQuant alpha for MatMul (NNCF default: 0.95).",
    )
    parser.add_argument(
        "--accurate_bias",
        action="store_true",
        help="Use slower, more accurate bias correction.",
    )
    parser.add_argument(
        "--no_transformer",
        action="store_true",
        help="Disable NNCF model_type=TRANSFORMER (turns off SmoothQuant).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["/audio_heads/MatMul"],
        help="Node names to exclude from quantization.",
    )
    parser.add_argument(
        "--exclude_types",
        nargs="*",
        default=["Gather"],
        help="Op types to exclude from quantization (embedding lookups by default).",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    import nncf
    import openvino as ov

    if args.onnx:
        logging.info(f"Converting ONNX to OpenVINO IR: {args.onnx}")
        model = ov.convert_model(args.onnx)
    else:
        logging.info(f"Reading OpenVINO IR: {args.ir}")
        model = ov.Core().read_model(args.ir)

    logging.info(
        f"Building calibration dataset (max {args.max_samples} samples) "
        f"from {args.calibration_dir} ..."
    )
    samples = list(
        calibration_iter(Path(args.calibration_dir), args.max_samples)
    )
    logging.info(f"  loaded {len(samples)} samples")
    dataset = nncf.Dataset(samples)

    preset = (
        nncf.QuantizationPreset.PERFORMANCE
        if args.preset == "performance"
        else nncf.QuantizationPreset.MIXED
    )

    advanced = None
    if args.smooth_quant_alpha is not None:
        from nncf.quantization.advanced_parameters import (
            AdvancedQuantizationParameters,
            AdvancedSmoothQuantParameters,
        )
        advanced = AdvancedQuantizationParameters(
            smooth_quant_alphas=AdvancedSmoothQuantParameters(
                matmul=args.smooth_quant_alpha,
            ),
        )

    logging.info(
        f"NNCF quantize: preset={args.preset} "
        f"transformer={'no' if args.no_transformer else 'yes'} "
        f"accurate_bias={args.accurate_bias} "
        f"sq_alpha={args.smooth_quant_alpha} "
        f"exclude_types={args.exclude_types} exclude={args.exclude}"
    )

    ignored = nncf.IgnoredScope(
        types=list(args.exclude_types), names=list(args.exclude),
    )
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

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(out), compress_to_fp16=False)
    logging.info(f"Saved {out}")


if __name__ == "__main__":
    sys.exit(main())
