#!/usr/bin/env python3
"""Dump the runtime-level optimized graph that ORT produces at session create.

ORT's transformers optimizer (offline, Python) and ORT's runtime graph
optimizer (online, when a session is created with ORT_ENABLE_ALL) are
two different code paths. The runtime path sometimes catches fusions the
offline one misses, including some that depend on backend-specific kernel
availability (NodeWithGraphAndBackendCpu pattern matching).

This script creates a session with optimized_model_filepath set so we
get the optimized graph dumped to disk, then we can inspect / re-bench.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dump-rt-opt")

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default=str(REPO_ROOT / "onnx" / "omnivoice-step.eager.onnx"))
    p.add_argument("--out", default=str(REPO_ROOT / "onnx" / "omnivoice-step.rt-opt.onnx"))
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    inp = Path(args.inp).resolve()
    out = Path(args.out).resolve()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(out)
    log.info("Creating session: in=%s out=%s", inp, out)
    sess = ort.InferenceSession(
        str(inp),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    log.info(
        "Session created. Inputs=%s outputs=%s providers=%s",
        [i.name for i in sess.get_inputs()],
        [o.name for o in sess.get_outputs()],
        sess.get_providers(),
    )
    log.info("Optimized graph dumped to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
