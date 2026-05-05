#!/usr/bin/env python3
"""Parity check: OpenVINO int8 vs ONNX fp32 reference on real fox-pangram input.

Same harness shape as scripts/parity_quants.py — argmax disagreement on the
[B, C, S, V] codebook logits, plus mean/max abs diff.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ir", default=str(REPO / "openvino_ir/omnivoice-step.int8.xml"))
    p.add_argument("--ref", default=str(REPO / "onnx/omnivoice-step.onnx"))
    p.add_argument("--sample",
                   default=str(REPO / "calibration/long_en_fox_step0000.npz"))
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    print(f"reference: {args.ref}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ref = ort.InferenceSession(args.ref, so, providers=["CPUExecutionProvider"])

    print(f"target:    {args.ir}")
    core = ov.Core()
    model = core.read_model(args.ir)
    compiled = core.compile_model(model, "CPU",
                                  {"INFERENCE_NUM_THREADS": args.threads})

    d = np.load(args.sample)
    feeds = {k: d[k] for k in d.files}
    print(f"sample shape: input_ids={feeds['input_ids'].shape}, "
          f"S={feeds['input_ids'].shape[-1]}")

    t0 = time.perf_counter()
    ref_logits = ref.run(["logits"], feeds)[0]
    ref_t = time.perf_counter() - t0
    print(f"ref forward: {ref_t*1000:.1f} ms")

    req = compiled.create_infer_request()
    # Warmup
    req.infer(feeds)
    t0 = time.perf_counter()
    out = req.infer(feeds)
    tgt_t = time.perf_counter() - t0
    tgt_logits = out[compiled.outputs[0]]
    print(f"OV forward:  {tgt_t*1000:.1f} ms")

    diff = np.abs(ref_logits - tgt_logits)
    print(f"shape: {ref_logits.shape}")
    print(f"mean abs diff:  {diff.mean():.4f}")
    print(f"max  abs diff:  {diff.max():.4f}")

    ref_arg = ref_logits.argmax(axis=-1)
    tgt_arg = tgt_logits.argmax(axis=-1)
    disagree = (ref_arg != tgt_arg).mean() * 100
    print(f"argmax disagree: {disagree:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
