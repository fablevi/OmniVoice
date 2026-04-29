#!/usr/bin/env python3
"""SmoothQuant Phase 2: rewrite the fp32 graph to migrate per-channel
activation magnitude into adjacent Linear weights.

For each LayerNorm -> {q,k,v / gate,up / audio_heads} group:
  s[j] = max(act_max[j], eps)^alpha / max(weight_max[j], eps)^(1-alpha)
  gamma_new = gamma / s
  W_new[j, k] = W[j, k] * s[j]      (input-channel scaling)

After the rewrite, per-tensor activation scales (which is all ORT's
static QDQ supports) become well-behaved because the channel-wise
max(|x|) becomes uniform.

Output: ``onnx/omnivoice-step.smoothed.onnx`` + ``.onnx_data`` sidecar.
Includes a self-check that runs original vs smoothed on the same input
and asserts max-abs-diff < 1e-3 (sanity for indexing / axis correctness).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sq-apply")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_STATS = REPO_ROOT / "calibration" / "activation_max.npz"
DEFAULT_OUT = REPO_ROOT / "onnx" / "omnivoice-step.smoothed.onnx"
DEFAULT_CALIB_DIR = REPO_ROOT / "calibration"

# Reuse the discover_groups helper from the calibrate script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoothquant_calibrate import discover_groups  # noqa: E402


def compute_scale(
    act_max: np.ndarray,
    weight_max: np.ndarray,
    alpha: float,
    eps: float = 1e-5,
    clip_min: float = 1e-3,
    clip_max: float = 1e3,
) -> np.ndarray:
    """Per-channel migration scale s[j], in fp64 to keep roundoff small."""
    a = np.maximum(act_max.astype(np.float64), eps)
    w = np.maximum(weight_max.astype(np.float64), eps)
    s = np.power(a, alpha) / np.power(w, 1.0 - alpha)
    return np.clip(s, clip_min, clip_max)


def apply_to_initializer(
    init: onnx.TensorProto,
    new_array: np.ndarray,
    inits_by_name: dict,
) -> None:
    """Replace an initializer's tensor data in-place, preserving its name
    and (for external-data initializers) re-spilling on the next save."""
    new_init = nh.from_array(new_array.astype(np.float32), name=init.name)
    init.CopyFrom(new_init)


def build_dummy_inputs(calib_dir: Path) -> dict:
    """Pick the smallest calibration sample as a parity-test input."""
    files = sorted(f for f in calib_dir.glob("*_step*.npz"))
    if not files:
        raise RuntimeError(f"no calibration samples in {calib_dir}")
    f = sorted(files, key=lambda p: p.stat().st_size)[0]
    d = np.load(f)
    return {
        "input_ids": d["input_ids"],
        "audio_mask": d["audio_mask"],
        "attention_mask": d["attention_mask"],
        "position_ids": d["position_ids"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--stats", default=str(DEFAULT_STATS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--calibration", default=str(DEFAULT_CALIB_DIR),
                   help="Used by the parity self-check.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Migration strength: 0.0 = no migration, 1.0 = all migration.")
    p.add_argument("--clip-min", type=float, default=1e-3)
    p.add_argument("--clip-max", type=float, default=1e3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--no-parity-check", action="store_true")
    p.add_argument("--parity-tol", type=float, default=1e-3)
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    stats_path = Path(args.stats).resolve()

    if not src.exists():
        log.error("input ONNX missing: %s", src)
        return 1
    if not stats_path.exists():
        log.error("activation stats missing: %s — run smoothquant_calibrate.py first",
                  stats_path)
        return 1

    log.info("Loading %s with external data ...", src)
    model = onnx.load(str(src), load_external_data=True)
    inits_by_name = {init.name: init for init in model.graph.initializer}

    log.info("Discovering groups ...")
    groups = discover_groups(model)
    log.info("Found %d groups", len(groups))

    log.info("Loading activation stats from %s ...", stats_path)
    stats = np.load(stats_path)
    missing = [g["gamma"] for g in groups if g["gamma"] not in stats.files]
    if missing:
        log.error("stats missing %d entries (e.g. %s)", len(missing), missing[:3])
        return 1

    log.info("Applying SmoothQuant (alpha=%.2f, clip=[%.0e, %.0e]) ...",
             args.alpha, args.clip_min, args.clip_max)

    s_summary = []  # (group_name, s_min, s_max, s_p99)
    for grp in groups:
        gamma_init = inits_by_name[grp["gamma"]]
        gamma = nh.to_array(gamma_init).astype(np.float64)  # [hidden]
        H = gamma.shape[0]
        act_max = stats[grp["gamma"]].astype(np.float64)
        if act_max.shape != (H,):
            raise RuntimeError(
                f"shape mismatch for {grp['gamma']}: gamma {gamma.shape} vs act {act_max.shape}"
            )

        # Per-channel weight max across all consumers, axis=output (axis=1
        # for [in, out]). This treats the consumers as one fused weight for
        # the purpose of scale balancing.
        weight_max = np.zeros(H, dtype=np.float64)
        consumer_arrays = []  # cached for the rewrite step
        for mm_name, w_name in grp["consumers"]:
            w = nh.to_array(inits_by_name[w_name])  # fp32 [in, out]
            if w.shape[0] != H:
                raise RuntimeError(
                    f"weight {w_name} has shape {w.shape}; expected first dim={H}"
                )
            wmax = np.max(np.abs(w), axis=1).astype(np.float64)  # [in]
            np.maximum(weight_max, wmax, out=weight_max)
            consumer_arrays.append((w_name, w))

        s = compute_scale(
            act_max, weight_max, alpha=args.alpha,
            clip_min=args.clip_min, clip_max=args.clip_max,
        )
        s_summary.append(
            (grp["gamma"], float(s.min()), float(s.max()), float(np.quantile(s, 0.99)))
        )

        # Rewrite gamma: gamma /= s
        new_gamma = (gamma / s).astype(np.float32)
        apply_to_initializer(gamma_init, new_gamma, inits_by_name)

        # Rewrite each consumer weight: W[j, :] *= s[j]
        s32 = s.astype(np.float32)
        for w_name, w in consumer_arrays:
            new_w = (w.astype(np.float64) * s[:, None]).astype(np.float32)
            apply_to_initializer(inits_by_name[w_name], new_w, inits_by_name)

    # Print a few extreme-scale rows (informational).
    log.info("scale stats (5 highest s_p99):")
    for name, smn, smx, sp99 in sorted(s_summary, key=lambda r: r[3], reverse=True)[:5]:
        log.info("  %-50s  s_min=%.3e  s_max=%.3e  s_p99=%.3e",
                 name, smn, smx, sp99)

    # Save.
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.name + "_data"
    # Wipe any stale outputs.
    for cand in (out, Path(str(out) + "_data"), Path(str(out) + ".data")):
        if cand.exists():
            cand.unlink()
    log.info("Saving smoothed graph to %s (sidecar=%s) ...", out, sidecar)
    onnx.save_model(
        model,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar,
        size_threshold=1024,
        convert_attribute=False,
    )
    graph_size = out.stat().st_size
    data_size = Path(str(out.parent / sidecar)).stat().st_size
    log.info("On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
             graph_size / 1e6, data_size / 1e6, (graph_size + data_size) / 1e6)

    if args.no_parity_check:
        return 0

    log.info("Self-check: running original vs smoothed on one calibration sample ...")
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    feeds = build_dummy_inputs(Path(args.calibration).resolve())
    log.info("  input shapes: %s", {k: v.shape for k, v in feeds.items()})

    t0 = time.perf_counter()
    sess_orig = ort.InferenceSession(
        str(src), sess_options=so, providers=["CPUExecutionProvider"],
    )
    out_orig = sess_orig.run(["logits"], feeds)[0]
    log.info("  original fp32 forward: %.2fs", time.perf_counter() - t0)
    del sess_orig  # free RAM before loading the smoothed copy

    t0 = time.perf_counter()
    sess_new = ort.InferenceSession(
        str(out), sess_options=so, providers=["CPUExecutionProvider"],
    )
    out_new = sess_new.run(["logits"], feeds)[0]
    log.info("  smoothed fp32 forward: %.2fs", time.perf_counter() - t0)

    diff = np.abs(out_orig.astype(np.float64) - out_new.astype(np.float64))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel = diff / (np.abs(out_orig.astype(np.float64)) + 1e-9)
    rel_max = float(rel.max())
    a = out_orig.astype(np.float64).reshape(-1)
    b = out_new.astype(np.float64).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    a_arg = out_orig.argmax(axis=-1)
    b_arg = out_new.argmax(axis=-1)
    disagree = float((a_arg != b_arg).mean())

    log.info("parity: max_abs=%.3e  mean_abs=%.3e  rel_max=%.3e  cos=%.10f  argmax_disagree=%.4f%%",
             max_abs, mean_abs, rel_max, cos, disagree * 100)

    if max_abs > args.parity_tol:
        log.error("parity FAILED: max_abs=%.3e > tolerance %.3e",
                  max_abs, args.parity_tol)
        return 2
    log.info("parity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
