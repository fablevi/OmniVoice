#!/usr/bin/env python3
"""Mixed-precision quantization: skip a subset of layers, keep them fp32.

Layers 25-27 + llm.norm dominate the absolute activation magnitudes
(see calibration/activation_max.npz). Quantizing them with per-tensor
activation scales is what blows up the rest of the layers' precision.
This script keeps the named layers' MatMul ops fp32 and runs the
existing dynamic int8 quantizer on the rest.

Output: ``onnx/omnivoice-step.int8.skip{lo}-{hi}.onnx``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quant-mixed")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"

AUDIO_HEADS_NODE = "/audio_heads/MatMul"


def collect_layer_nodes(model: onnx.ModelProto, layers: list[int]) -> list[str]:
    layer_set = set(layers)
    out: list[str] = []
    for n in model.graph.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        m = re.search(r"/layers\.(\d+)/", n.name)
        if m and int(m.group(1)) in layer_set:
            out.append(n.name)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--out", required=True)
    p.add_argument("--skip-layers", default="",
                   help="Comma-sep list of layer indices to skip, e.g. '23,24,25,26,27'")
    p.add_argument("--skip-final-norm", action="store_true",
                   help="Also skip the final LM norm path (audio_heads MatMul is "
                        "already excluded by default — this is a no-op since heads stays fp32).")
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--include-heads", action="store_true",
                   help="Include audio_heads in the quantization (default keeps fp32).")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    skip_layers: list[int] = []
    if args.skip_layers:
        skip_layers = [int(x) for x in args.skip_layers.split(",") if x.strip()]

    log.info("Loading model: %s", src)
    m = onnx.load(str(src), load_external_data=False)

    nodes_to_exclude: list[str] = []
    if not args.include_heads:
        nodes_to_exclude.append(AUDIO_HEADS_NODE)
    if skip_layers:
        nodes_to_exclude.extend(collect_layer_nodes(m, skip_layers))

    log.info("Excluding %d nodes from quantization (skip_layers=%s, exclude_heads=%s)",
             len(nodes_to_exclude), skip_layers, not args.include_heads)
    for n in nodes_to_exclude[:8]:
        log.info("  ex: %s", n)
    if len(nodes_to_exclude) > 8:
        log.info("  ... and %d more", len(nodes_to_exclude) - 8)

    # Wipe stale outputs.
    for path in (out, Path(str(out) + ".data"), Path(str(out) + "_data")):
        if path.exists():
            path.unlink()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=args.per_channel,
        nodes_to_exclude=nodes_to_exclude or None,
        use_external_data_format=True,
    )

    graph_size = out.stat().st_size
    data_size = 0
    for cand in (Path(str(out) + ".data"), Path(str(out) + "_data")):
        if cand.exists():
            data_size = cand.stat().st_size
            break
    log.info("On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
             graph_size / 1e6, data_size / 1e6, (graph_size + data_size) / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
