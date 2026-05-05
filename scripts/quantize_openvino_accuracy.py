#!/usr/bin/env python3
"""NNCF accuracy-aware quantization for OmniVoice OV step.

Iteratively reverts the most-impactful layers from int8 to fp32 until the
target accuracy drop is satisfied. Should automate the dp/dpup exclusion
discovery — and possibly find a better mix.
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
    p.add_argument("--samples", default=str(REPO / "calibration"))
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--val-samples", type=int, default=8,
                   help="Validation samples (smaller; used for scoring)")
    p.add_argument("--max-drop", type=float, default=0.05,
                   help="Max accuracy drop allowed (absolute)")
    args = p.parse_args()

    core = ov.Core()
    print(f"Loading: {args.ir}")
    model = core.read_model(args.ir)

    files = sorted(p for p in Path(args.samples).glob("*.npz")
                   if p.name != "activation_max.npz")
    calib_files = files[:args.calib_samples]
    val_files = files[args.calib_samples:args.calib_samples + args.val_samples]
    print(f"calib={len(calib_files)} val={len(val_files)}")

    def gen(fs):
        for f in fs:
            d = np.load(f)
            yield {k: d[k] for k in
                   ["input_ids", "audio_mask", "attention_mask", "position_ids"]}

    cal_ds = nncf.Dataset(list(gen(calib_files)))
    val_ds = nncf.Dataset(list(gen(val_files)))

    # Pre-compute reference logits using fp32 model
    fp32_compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 8})
    out_port = fp32_compiled.outputs[0]
    print("Computing fp32 reference logits...")
    fp32_argmax = []
    for f in val_files:
        d = np.load(f)
        feeds = {k: d[k] for k in ["input_ids", "audio_mask", "attention_mask", "position_ids"]}
        out = fp32_compiled.create_infer_request().infer(feeds)
        fp32_argmax.append(out[out_port].argmax(axis=-1))

    def validation_fn(quantized_model, val_iter):
        # quantized_model is already a CompiledModel here
        op = quantized_model.outputs[0]
        req = quantized_model.create_infer_request()
        per_sample = []
        for i, item in enumerate(val_iter):
            out = req.infer(item)
            arg = out[op].argmax(axis=-1)
            ref = fp32_argmax[i]
            agree = float((arg == ref).mean())
            per_sample.append(agree)
        # Return metric (float) and per-sample list
        return float(np.mean(per_sample)), per_sample

    print(f"Running accuracy-aware quantize (max_drop={args.max_drop} ABS) ...")
    quantized = nncf.quantize_with_accuracy_control(
        model,
        calibration_dataset=cal_ds,
        validation_dataset=val_ds,
        validation_fn=validation_fn,
        max_drop=args.max_drop,
        drop_type=nncf.DropType.ABSOLUTE,
        preset=nncf.QuantizationPreset.MIXED,
        model_type=nncf.ModelType.TRANSFORMER,
        ignored_scope=nncf.IgnoredScope(types=["Gather"], names=["/audio_heads/MatMul"]),
        subset_size=args.calib_samples,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(out), compress_to_fp16=False)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
