#!/usr/bin/env python3
"""Parity harness: compare any quantized ONNX file vs the fp32 baseline.

Uses one real calibration sample (B=2, S~226, fox pangram) so the input
distribution matches what _generate_iterative actually produces. Reports
mean abs diff and argmax disagreement on the codebook logits — the same
metrics used in PERFORMANCE.md.

Usage:
    python scripts/parity_quants.py onnx/omnivoice-step.int8.onnx \
        onnx/omnivoice-step.int8.foo.onnx ...
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parity-quants")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP32 = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_SAMPLE = REPO_ROOT / "calibration" / "long_en_fox_step0000.npz"


def load_feeds(path: Path, expected: set[str]) -> dict:
    d = np.load(path)
    feeds = {
        "input_ids": d["input_ids"],
        "audio_mask": d["audio_mask"],
        "attention_mask": d["attention_mask"],
    }
    if "position_ids" in expected:
        feeds["position_ids"] = d["position_ids"]
    return feeds


def run_one(onnx_path: Path, feeds: dict, threads: int) -> tuple[np.ndarray, float]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # External-data sidecars are stored with a relative `location` field
    # ("foo.onnx.data") that ORT resolves relative to the cwd. Switch into
    # the file's parent so the relative path resolves correctly across
    # sequential session creates in one process.
    import os
    saved_cwd = os.getcwd()
    try:
        os.chdir(onnx_path.parent)
        sess = ort.InferenceSession(
            onnx_path.name, sess_options=so, providers=["CPUExecutionProvider"]
        )
    finally:
        os.chdir(saved_cwd)
    expected = {i.name for i in sess.get_inputs()}
    these_feeds = {k: v for k, v in feeds.items() if k in expected}
    # Warm-up.
    sess.run(["logits"], these_feeds)
    t0 = time.perf_counter()
    out = sess.run(["logits"], these_feeds)[0]
    dt = time.perf_counter() - t0
    return out, dt


def metrics(ref: np.ndarray, x: np.ndarray) -> dict:
    diff = np.abs(ref.astype(np.float64) - x.astype(np.float64))
    a = ref.reshape(-1).astype(np.float64)
    b = x.reshape(-1).astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    a_arg = ref.argmax(axis=-1)
    b_arg = x.argmax(axis=-1)
    disagree = float((a_arg != b_arg).mean())
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cos": cos,
        "argmax_disagree": disagree,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variants", nargs="+",
                   help="ONNX file(s) to test, or label:path")
    p.add_argument("--fp32", default=str(DEFAULT_FP32),
                   help="Baseline fp32 ONNX (reference)")
    p.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    fp32 = Path(args.fp32).resolve()
    sample = Path(args.sample).resolve()
    if not fp32.exists():
        log.error("fp32 baseline missing: %s", fp32)
        return 1
    if not sample.exists():
        log.error("calibration sample missing: %s", sample)
        return 1

    # Load feeds once with the union of possible keys.
    raw = np.load(sample)
    full_feeds = {k: raw[k] for k in raw.files}
    log.info("Sample: B=%d S=%d", raw["input_ids"].shape[0], raw["input_ids"].shape[-1])

    log.info("Running fp32 baseline: %s", fp32.name)
    ref, dt_ref = run_one(fp32, full_feeds, args.threads)
    log.info("  fp32 forward: %.2fs", dt_ref)

    print()
    print(f"{'variant':<55} {'mean_abs':>10} {'argmax_disagree':>17} {'cos':>10} {'forward_s':>10}")
    print("-" * 110)

    # fp32 vs itself = sanity row.
    m = metrics(ref, ref)
    print(f"{'fp32 (self)':<55} {m['mean_abs']:>10.4f} {m['argmax_disagree']*100:>16.2f}% "
          f"{m['cos']:>10.6f} {dt_ref:>10.3f}")

    rows = []
    for v in args.variants:
        if ":" in v and not Path(v).exists():
            label, path = v.split(":", 1)
        else:
            label = Path(v).stem
            path = v
        path = Path(path).resolve()
        if not path.exists():
            log.warning("missing: %s — skip", path)
            continue
        out, dt = run_one(path, full_feeds, args.threads)
        m = metrics(ref, out)
        rows.append((label, m, dt))
        print(f"{label:<55} {m['mean_abs']:>10.4f} {m['argmax_disagree']*100:>16.2f}% "
              f"{m['cos']:>10.6f} {dt:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
