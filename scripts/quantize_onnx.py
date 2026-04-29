#!/usr/bin/env python3
"""Quantize the exported OmniVoice ONNX graph (weights-only, dynamic).

Reads ``onnx/omnivoice-step.onnx`` (+ ``omnivoice-step.onnx_data``), writes
``--out`` (default ``onnx/omnivoice-step.int8.onnx``).

Only ``MatMul`` and ``Gemm`` are touched, leaving Embedding, LayerNorm, and
the rest of the graph at fp32. By default the ``audio_heads`` MatMul is
ALSO kept fp32 — quantising it cuts overall quality measurably (mean logit
error drops from 0.547 to 0.344, argmax disagreement from 78% to 49%).
Pass ``--no-exclude-heads`` to override.

Flags:

    --no-exclude-heads  Quantize audio_heads too (default: keep fp32).
    --per-channel       Per-output-channel weight scales.
    --uint8             QUInt8 (asymmetric, with zero-point) instead of
                        QInt8 (symmetric).
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

# Node name in the exported graph (verified once via onnx.load + grep —
# torch's tracer names the audio_heads Linear's MatMul this consistently).
AUDIO_HEADS_NODE = "/audio_heads/MatMul"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--exclude-heads", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep /audio_heads/MatMul fp32 (default true — see module docstring)")
    p.add_argument("--per-channel", action="store_true",
                   help="Per-output-channel weight scales")
    p.add_argument("--uint8", action="store_true",
                   help="QUInt8 instead of QInt8 (asymmetric)")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log.error("input not found: %s — run scripts/export_onnx.py first", src)
        return 1

    # Wipe any stale outputs (onnxruntime writes ``<out>.data``,
    # ``onnx.save_model`` writes ``<out>_data`` — clean both layouts).
    for path in (out, Path(str(out) + ".data"), Path(str(out) + "_data")):
        if path.exists():
            path.unlink()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    weight_type = QuantType.QUInt8 if args.uint8 else QuantType.QInt8
    nodes_to_exclude = [AUDIO_HEADS_NODE] if args.exclude_heads else []

    log.info(
        "Quantizing %s → %s (%s, dynamic, MatMul+Gemm, per_channel=%s, exclude=%s)",
        src.name, out.name,
        "QUInt8" if args.uint8 else "QInt8",
        args.per_channel,
        nodes_to_exclude or "none",
    )

    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=weight_type,
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
    total = graph_size + data_size
    log.info(
        "On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
        graph_size / 1e6, data_size / 1e6, total / 1e6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
