#!/usr/bin/env python3
"""Quantize the exported OmniVoice ONNX graph to int8 (weights-only, dynamic).

Reads ``onnx/omnivoice-step.onnx`` (+ ``omnivoice-step.onnx_data``), writes
``onnx/omnivoice-step.int8.onnx`` (+ ``omnivoice-step.int8.onnx_data``).

Only ``MatMul`` and ``Gemm`` are touched, leaving Embedding, LayerNorm, and
the rest of the graph at fp32.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quantize-onnx")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_OUT = REPO_ROOT / "onnx" / "omnivoice-step.int8.onnx"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log.error("input not found: %s — run scripts/export_onnx.py first", src)
        return 1

    # Wipe any stale int8 outputs (onnxruntime writes ``<out>.data``,
    # ``onnx.save_model`` writes ``<out>_data`` — clean both layouts).
    for path in (out, Path(str(out) + ".data"), Path(str(out) + "_data")):
        if path.exists():
            path.unlink()

    log.info("Quantizing %s → %s (QInt8, dynamic, MatMul+Gemm only) ...", src.name, out.name)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        use_external_data_format=True,
    )

    graph_size = out.stat().st_size
    data_size = 0
    for cand in (Path(str(out) + ".data"), Path(str(out) + "_data")):
        if cand.exists():
            data_size = cand.stat().st_size
            break
    total = graph_size + data_size
    log.info(
        "On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
        graph_size / 1e6, data_size / 1e6, total / 1e6,
    )
    # Note: gluschenko/omnivoice-onnx qint8 ships at 612 MB. Ours is ~1.1 GB
    # because we explicitly excluded Embedding ops from quantization
    # (op_types_to_quantize=["MatMul","Gemm"]). The 621 MB Qwen3 embed_tokens
    # plus a ~33 MB audio_embeddings table stay fp32. The MatMul int8 fast
    # path is what matters for RTF — embedding lookup is bandwidth-bound and
    # already cheap in fp32.
    return 0


if __name__ == "__main__":
    sys.exit(main())
