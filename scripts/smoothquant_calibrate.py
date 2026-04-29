#!/usr/bin/env python3
"""SmoothQuant Phase 1: capture per-channel activation max at every
LayerNorm output that feeds a Linear (MatMul).

For each of the 57 smoothable LayerNorm groups (28x input_layernorm,
28x post_attention_layernorm, 1x final norm), exposes the LN output as
an extra graph output, runs the existing ``calibration/*.npz`` corpus
through ``onnxruntime``, and accumulates per-channel ``max(|x|)`` over
the batch+seq dims.

Output: ``calibration/activation_max.npz`` with one ``[hidden]`` array
per group, keyed by the LN gamma initializer name.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sq-calibrate")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_CALIB = REPO_ROOT / "calibration"
DEFAULT_STATS = REPO_ROOT / "calibration" / "activation_max.npz"
DEFAULT_AUG = REPO_ROOT / "onnx" / "omnivoice-step.with_intermediates.onnx"

GAMMA_PATTERNS = [
    re.compile(r"^llm\.layers\.\d+\.input_layernorm\.weight$"),
    re.compile(r"^llm\.layers\.\d+\.post_attention_layernorm\.weight$"),
    re.compile(r"^llm\.norm\.weight$"),
]


def discover_groups(model: onnx.ModelProto) -> list[dict]:
    """Walk the graph and return one record per smoothable LN->MatMul group.

    Each record has:
      gamma: name of the LN gamma initializer
      ln_out: name of the LN ``Mul_1`` output tensor (per-channel activation)
      consumers: list of (matmul_node_name, weight_init_name)
    """
    nodes = list(model.graph.node)
    inits = {init.name for init in model.graph.initializer}

    gamma_names = sorted(n for n in inits if any(p.match(n) for p in GAMMA_PATTERNS))

    groups: list[dict] = []
    for gamma in gamma_names:
        # Find the Mul that multiplies (rms-normalized x) by gamma.
        muls = [n for n in nodes if n.op_type == "Mul" and gamma in n.input]
        if len(muls) != 1:
            raise RuntimeError(
                f"expected exactly 1 Mul consuming {gamma}, got {len(muls)}"
            )
        mul = muls[0]
        if len(mul.output) != 1:
            raise RuntimeError(f"Mul {mul.name} has {len(mul.output)} outputs")
        ln_out = mul.output[0]

        # Direct MatMul consumers of the LN output.
        consumers = []
        for n in nodes:
            if n.op_type == "MatMul" and ln_out in n.input:
                # The other (non-LN) input is the weight initializer.
                if n.input[0] == ln_out:
                    weight = n.input[1]
                else:
                    weight = n.input[0]
                if weight not in inits:
                    raise RuntimeError(
                        f"MatMul {n.name} weight {weight} is not an initializer"
                    )
                consumers.append((n.name, weight))
        if not consumers:
            raise RuntimeError(f"no MatMul consumers for LN output {ln_out}")
        groups.append({"gamma": gamma, "ln_out": ln_out, "consumers": consumers})
    return groups


def augment_graph(model: onnx.ModelProto, groups: list[dict]) -> onnx.ModelProto:
    """Add each LN output as an extra graph output (so we can fetch them
    from a single session.run). Reuses the existing weight sidecar."""
    existing_outputs = {o.name for o in model.graph.output}
    # Build a name->ValueInfoProto map from value_info if available, else
    # synthesize a minimal one (rank+shape unknown is fine for ORT).
    info_map = {vi.name: vi for vi in list(model.graph.value_info)
                + list(model.graph.input) + list(model.graph.output)}
    for grp in groups:
        name = grp["ln_out"]
        if name in existing_outputs:
            continue
        if name in info_map:
            model.graph.output.append(info_map[name])
        else:
            vi = onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, None  # unknown shape
            )
            model.graph.output.append(vi)
    return model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--aug-out", default=str(DEFAULT_AUG),
                   help="Where to save the augmented graph (graph-only; reuses sidecar).")
    p.add_argument("--calibration", default=str(DEFAULT_CALIB))
    p.add_argument("--stats-out", default=str(DEFAULT_STATS))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap calibration set size (debug).")
    args = p.parse_args()

    src = Path(args.src).resolve()
    aug_out = Path(args.aug_out).resolve()
    calib_dir = Path(args.calibration).resolve()
    stats_out = Path(args.stats_out).resolve()

    if not src.exists():
        log.error("input ONNX missing: %s", src)
        return 1

    log.info("Loading %s (graph only) ...", src)
    # We only need the structural graph here; the augmented copy will share
    # the original weight sidecar via a relative ``location`` reference.
    model = onnx.load(str(src), load_external_data=False)

    log.info("Discovering smoothable groups ...")
    groups = discover_groups(model)
    log.info("Found %d groups", len(groups))
    n_consumers = sum(len(g["consumers"]) for g in groups)
    log.info("  total consumer MatMuls: %d", n_consumers)
    # Spot-print one of each kind
    for tag in ("input_layernorm", "post_attention_layernorm", "norm.weight"):
        match = next(g for g in groups if tag in g["gamma"])
        log.info(
            "  e.g. %s -> %s (consumers: %s)",
            match["gamma"], match["ln_out"],
            [c[0] for c in match["consumers"]],
        )

    augment_graph(model, groups)

    # Save the augmented graph next to the original sidecar. We do NOT call
    # save_model with save_as_external_data=True (that would re-spill weights
    # into a new file) — instead, the augmented .onnx still references
    # ``omnivoice-step.onnx_data`` because we never touched the initializers.
    aug_out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving augmented graph to %s ...", aug_out)
    # Sanity: confirm the original sidecar is the one referenced.
    onnx.save_model(model, str(aug_out))

    log.info("Opening ORT session ...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(
        str(aug_out), sess_options=so, providers=["CPUExecutionProvider"],
    )
    sess_outputs = [o.name for o in sess.get_outputs()]
    missing = [g["ln_out"] for g in groups if g["ln_out"] not in sess_outputs]
    if missing:
        log.error("session is missing %d expected outputs:", len(missing))
        for m in missing[:5]:
            log.error("  %s", m)
        return 1
    log.info("session has %d outputs (expected logits + %d LN tensors)",
             len(sess_outputs), len(groups))

    # Iterate calibration samples. We accumulate per-channel max over (B, S)
    # for each LN output.
    # Skip our own output (``activation_max.npz``) and any other non-corpus
    # npz files that may sit alongside the calibration samples.
    files = sorted(f for f in calib_dir.glob("*_step*.npz"))
    if args.max_samples:
        files = files[: args.max_samples]
    if not files:
        log.error("no calibration samples in %s", calib_dir)
        return 1
    log.info("running %d calibration samples ...", len(files))

    # We only need the LN outputs, not logits — fetching just those keeps
    # the per-call transient memory low.
    fetch = [g["ln_out"] for g in groups]
    accum: dict[str, np.ndarray] = {}

    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        d = np.load(f)
        feeds = {
            "input_ids": d["input_ids"],
            "audio_mask": d["audio_mask"],
            "attention_mask": d["attention_mask"],
            "position_ids": d["position_ids"],
        }
        outs = sess.run(fetch, feeds)
        for grp, arr in zip(groups, outs):
            # arr is [B, S, hidden] fp32
            cur_max = np.max(np.abs(arr), axis=(0, 1))  # -> [hidden]
            prev = accum.get(grp["gamma"])
            if prev is None:
                accum[grp["gamma"]] = cur_max
            else:
                np.maximum(prev, cur_max, out=prev)
        if i % 10 == 0 or i == len(files):
            log.info("  %d / %d  (%.1fs elapsed)",
                     i, len(files), time.perf_counter() - t0)

    # Sanity report: top-channel-max / median ratio per group.
    log.info("per-group outlier stats (top channel max / median):")
    ratios = []
    for grp in groups:
        a = accum[grp["gamma"]]
        med = float(np.median(a))
        top = float(np.max(a))
        ratio = top / max(med, 1e-12)
        ratios.append((grp["gamma"], ratio, top, med))
    ratios.sort(key=lambda r: r[1], reverse=True)
    for name, ratio, top, med in ratios[:5]:
        log.info("  %-50s ratio=%7.2fx  max=%.3e  median=%.3e",
                 name, ratio, top, med)
    log.info("  (... %d more groups; min ratio=%.2fx)",
             max(0, len(ratios) - 5), ratios[-1][1])

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    log.info("saving %d arrays to %s", len(accum), stats_out)
    np.savez_compressed(stats_out, **accum)
    log.info("done in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
