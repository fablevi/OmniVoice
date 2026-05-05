#!/usr/bin/env python3
"""Run the OmniVoice fp32 ONNX export through onnxruntime.transformers.optimizer.

Fuses Qwen3-specific patterns the generic ORT optimizer leaves alone:
- RMSNorm into a single op (instead of 5+ elementwise ops)
- Multi-head attention with rotary into a fused MultiHeadAttention op
- SwiGLU MLP (gate * up via SiLU) into a single fused op
- Removes redundant casts and dead nodes

This is portable: same graph runs on any ORT build (x86, ARM, mobile),
no kernel-level vendor specificity. The fusions exist in upstream ORT
because they're worth ~1.1-1.3× across all hardware on transformer
graphs of this shape.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import onnx
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("optimize-onnx")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_OUT = REPO_ROOT / "onnx" / "omnivoice-step.opt.onnx"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--model-type", default="qwen3",
                   help="Model type for the optimizer (qwen3, gpt_neox, bert, ...)")
    p.add_argument("--num-heads", type=int, default=16,
                   help="Qwen3-0.6B has 16 attention heads")
    p.add_argument("--hidden-size", type=int, default=1024,
                   help="Qwen3-0.6B hidden_size = 1024")
    p.add_argument("--opt-level", type=int, default=2, choices=[0, 1, 2, 99],
                   help="0=disable, 1=basic, 2=extended, 99=all")
    p.add_argument("--use-gpu", action="store_true", help="(off — CPU-only)")
    args = p.parse_args()

    inp = Path(args.inp).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        log.error("Input ONNX not found: %s", inp)
        return 1

    log.info("Loading %s ...", inp)
    t0 = time.time()
    model_proto = onnx.load(str(inp))  # loads sidecar via external_data refs
    log.info("Loaded in %.1fs", time.time() - t0)

    fusion_opts = FusionOptions(args.model_type)
    # Be aggressive — we want every fusion the optimizer knows about.
    fusion_opts.enable_attention = True
    fusion_opts.enable_rotary_embeddings = True
    fusion_opts.enable_layer_norm = True
    fusion_opts.enable_skip_layer_norm = True
    fusion_opts.enable_bias_skip_layer_norm = True
    fusion_opts.enable_bias_gelu = True
    fusion_opts.enable_gelu_approximation = False  # exact GELU only
    fusion_opts.enable_skip_group_norm = True
    fusion_opts.enable_packed_qkv = True
    fusion_opts.enable_packed_kv = True
    fusion_opts.enable_skip_rms_norm = True
    fusion_opts.enable_rms_norm = True

    log.info(
        "optimize_model(model_type=%s, num_heads=%d, hidden_size=%d, opt_level=%d)",
        args.model_type, args.num_heads, args.hidden_size, args.opt_level,
    )
    t0 = time.time()
    optimized = optimize_model(
        str(inp),
        model_type=args.model_type,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        optimization_options=fusion_opts,
        opt_level=args.opt_level,
        use_gpu=False,
        only_onnxruntime=False,  # also run python-side fusions
    )
    dt = time.time() - t0
    log.info("optimize_model done in %.1fs", dt)

    # Print fusion summary.
    log.info("Fusion summary:")
    summary = optimized.get_fused_operator_statistics()
    if summary:
        for k, v in sorted(summary.items()):
            if v:
                log.info("  %-40s %d", k, v)
    else:
        log.warning("No fusion stats reported (the optimizer may have skipped this model_type)")

    # Compare op counts pre vs post.
    pre_ops = {}
    for n in model_proto.graph.node:
        pre_ops[n.op_type] = pre_ops.get(n.op_type, 0) + 1

    post_ops = {}
    post_proto = optimized.model
    for n in post_proto.graph.node:
        post_ops[n.op_type] = post_ops.get(n.op_type, 0) + 1

    log.info("Op-count delta (post - pre, top changes):")
    all_keys = set(pre_ops) | set(post_ops)
    deltas = sorted(
        ((k, post_ops.get(k, 0) - pre_ops.get(k, 0)) for k in all_keys),
        key=lambda kv: -abs(kv[1]),
    )
    for k, d in deltas[:20]:
        if d != 0:
            log.info("  %-30s %+5d  (%d -> %d)", k, d, pre_ops.get(k, 0), post_ops.get(k, 0))

    log.info("Saving to %s", out)
    optimized.save_model_to_file(
        str(out),
        use_external_data_format=True,
        all_tensors_to_one_file=True,
    )

    graph_size = out.stat().st_size
    data_paths = list(out.parent.glob(out.name + "*data*"))
    data_size = sum(p.stat().st_size for p in data_paths)
    log.info(
        "On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
        graph_size / 1e6, data_size / 1e6, (graph_size + data_size) / 1e6,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
