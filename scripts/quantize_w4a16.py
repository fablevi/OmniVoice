#!/usr/bin/env python3
"""W4A16 quantization via ORT's MatMulNBitsQuantizer.

Quantizes MatMul weights to 4-bit, keeps activations fp32. Block-wise
scales (default block_size=128) handle outlier channels much better
than the per-tensor scales used in W8A8 dynamic.

Algorithms:
  --algo default   round-to-nearest, symmetric/asymmetric controlled by --asymmetric
  --algo rtn       same as default but explicit
  --algo hqq       Half-Quadratic Quantization (training-free post-quant search)
  --algo gptq      GPTQ — needs a calibration data reader (TODO)

Output: onnx/omnivoice-step.w4a16.<algo>.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quant-w4a16")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"

AUDIO_HEADS_NODE = "/audio_heads/MatMul"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--out", required=True)
    p.add_argument("--algo", default="default",
                   choices=["default", "rtn", "hqq"])
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--asymmetric", action="store_true",
                   help="Asymmetric (zero-point) quant; default symmetric.")
    p.add_argument("--include-heads", action="store_true",
                   help="Include audio_heads (default keeps it fp32).")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    from onnxruntime.quantization import matmul_nbits_quantizer as mnq

    nodes_to_exclude = [] if args.include_heads else [AUDIO_HEADS_NODE]

    if args.algo in ("default", "rtn"):
        algo = mnq.DefaultWeightOnlyQuantConfig(
            block_size=args.block_size,
            is_symmetric=not args.asymmetric,
            bits=args.bits,
        )
    elif args.algo == "hqq":
        algo = mnq.HQQWeightOnlyQuantConfig(
            block_size=args.block_size,
            bits=args.bits,
        )
    else:
        raise ValueError(args.algo)

    log.info("Loading model: %s", src)
    m = onnx.load(str(src), load_external_data=True)

    log.info("W4A16 quantize: bits=%d block=%d sym=%s algo=%s exclude=%s",
             args.bits, args.block_size, not args.asymmetric, args.algo,
             nodes_to_exclude or "none")

    q = mnq.MatMulNBitsQuantizer(
        model=m,
        bits=args.bits,
        block_size=args.block_size,
        is_symmetric=not args.asymmetric,
        nodes_to_exclude=nodes_to_exclude or None,
        algo_config=algo,
    )
    q.process()

    # Wipe stale outputs.
    for path in (out, Path(str(out) + ".data"), Path(str(out) + "_data")):
        if path.exists():
            path.unlink()

    sidecar = out.name + "_data"
    log.info("Saving %s (sidecar=%s) ...", out.name, sidecar)
    onnx.save_model(
        q.model.model,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar,
        size_threshold=1024,
        convert_attribute=False,
    )
    graph_size = out.stat().st_size
    data_size = 0
    for cand in (Path(str(out) + ".data"), Path(out.parent / sidecar)):
        if cand.exists():
            data_size = cand.stat().st_size
            break
    log.info("On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
             graph_size / 1e6, data_size / 1e6, (graph_size + data_size) / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
