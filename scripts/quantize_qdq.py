#!/usr/bin/env python3
"""Static QDQ quantization with calibration data.

Reads ``calibration/*.npz`` (one per ``sess.run`` call from the fp32
driver), runs onnxruntime's static quantizer with ``Percentile``
calibration, per-channel int8 weights, and ``audio_heads`` excluded from
quantisation (matches the dynamic-quant winner).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quantize-qdq")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_CALIB = REPO_ROOT / "calibration"
DEFAULT_OUT = REPO_ROOT / "onnx" / "omnivoice-step.qdq.onnx"

AUDIO_HEADS_NODE = "/audio_heads/MatMul"


def _pad_sample(d, target_S: int) -> dict:
    """Pad one calibration sample to a uniform seq_len.

    ORT's Percentile/Entropy histogram collector calls
    ``np.asarray(list_of_per_sample_tensors)`` which fails on heterogeneous
    shapes ("inhomogeneous shape" ValueError). Solution: pad every sample
    to the same seq_len. Padded positions:

    - ``input_ids[:, :, S:]`` = 0 (token 0 exists in both vocabs).
    - ``audio_mask[:, S:]`` = False (treats padded positions as text).
    - ``attention_mask[:, :, S:, :]`` and ``[:, :, :, S:]`` = False, with
      the diagonal True for padded rows so softmax doesn't see all-zero
      keys (would produce NaN). Mirrors the cond/uncond padding pattern
      in omnivoice.py:1215-1217.
    - ``position_ids[:, S:]`` extends arange from the last real position.
    """
    input_ids = d["input_ids"]
    B, C, S = input_ids.shape
    if S == target_S:
        return {k: d[k] for k in ("input_ids", "audio_mask", "attention_mask", "position_ids")}

    new_ids = np.zeros((B, C, target_S), dtype=input_ids.dtype)
    new_ids[:, :, :S] = input_ids

    audio_mask = d["audio_mask"]
    new_amask = np.zeros((B, target_S), dtype=audio_mask.dtype)
    new_amask[:, :S] = audio_mask

    attn = d["attention_mask"]
    new_attn = np.zeros((B, 1, target_S, target_S), dtype=attn.dtype)
    new_attn[:, :, :S, :S] = attn
    diag = np.arange(S, target_S)
    new_attn[:, 0, diag, diag] = True

    pos = d["position_ids"]
    new_pos = np.zeros((B, target_S), dtype=pos.dtype)
    new_pos[:, :S] = pos
    last = pos[:, -1:]
    new_pos[:, S:] = last + np.arange(1, target_S - S + 1)

    return {
        "input_ids": new_ids,
        "audio_mask": new_amask,
        "attention_mask": new_attn,
        "position_ids": new_pos,
    }


def make_reader(
    calib_dir: Path,
    max_samples: int | None = None,
    filter_seq_len: int | None = None,
):
    from onnxruntime.quantization import CalibrationDataReader

    # Only consider per-step capture files. Skip aux artifacts that may
    # share the directory (e.g. ``activation_max.npz`` from SmoothQuant).
    all_files = sorted(calib_dir.glob("*_step*.npz"))
    if not all_files:
        raise RuntimeError(f"no calibration samples in {calib_dir}")

    if filter_seq_len is not None:
        kept = []
        for f in all_files:
            d = np.load(f)
            if d["input_ids"].shape[2] == filter_seq_len:
                kept.append(f)
        if not kept:
            raise RuntimeError(f"no calibration samples at seq_len={filter_seq_len}")
        log.info(
            "filtered to %d / %d samples at seq_len=%d",
            len(kept), len(all_files), filter_seq_len,
        )
        all_files = kept

    files = all_files
    if max_samples is not None and max_samples > 0 and max_samples < len(all_files):
        # Sort by file size ascending (≈ seq_len ascending — calibration session
        # arena sizes itself to the largest activation, so prefer shorter samples
        # under memory pressure). Pick the first ``max_samples``.
        files = sorted(all_files, key=lambda p: p.stat().st_size)[:max_samples]
        files = sorted(files)  # restore name order for deterministic feeding
        log.info(
            "calibration: %d / %d samples (subsampled, shortest-first) in %s",
            len(files), len(all_files), calib_dir,
        )
    else:
        log.info("calibration: %d samples in %s", len(files), calib_dir)

    # Determine the pad-to seq_len: max across the selected samples. Required
    # because Percentile/Entropy collectors aggregate per-tensor activations
    # via np.asarray, which can't broadcast across shapes.
    max_S = 0
    for f in files:
        d = np.load(f)
        max_S = max(max_S, d["input_ids"].shape[2])
    log.info("padding all calibration samples to seq_len=%d", max_S)

    class NpzReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter: Iterator[Path] = iter(files)
            self._n = 0

        def get_next(self):
            try:
                f = next(self._iter)
            except StopIteration:
                return None
            d = np.load(f)
            self._n += 1
            if self._n % 20 == 0:
                log.info("  ... fed %d / %d calibration samples", self._n, len(files))
            return _pad_sample(d, max_S)

        def rewind(self) -> None:
            self._iter = iter(files)
            self._n = 0

    return NpzReader()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--calibration", default=str(DEFAULT_CALIB))
    p.add_argument("--method", default="percentile",
                   choices=["percentile", "entropy", "minmax"])
    p.add_argument("--percentile", type=float, default=99.999,
                   help="Percentile for clipping (used when method=percentile)")
    p.add_argument("--no-exclude-heads", action="store_true",
                   help="Quantize audio_heads too (default: exclude for quality)")
    p.add_argument("--per-tensor", action="store_true",
                   help="Per-tensor weight scales (default: per-channel)")
    p.add_argument("--activation-type", default="qint8",
                   choices=["qint8", "quint8", "qint16", "quint16"],
                   help="Activation dtype. qint8 is the fast path on AVX-VNNI; "
                        "qint16 has ~256x more representable values (closer to fp32 "
                        "quality on transformer hidden states) but kernel support "
                        "is limited and may force fp32 fallback at runtime.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap calibration set size (shortest sequences first). "
                        "Use under memory pressure: --max-samples 32 or 16.")
    p.add_argument("--filter-seq-len", type=int, default=None,
                   help="Filter calibration to samples with exactly this seq_len "
                        "(natural same-shape subset; skips padding).")
    args = p.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    calib_dir = Path(args.calibration).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        log.error("input ONNX missing: %s", src)
        return 1
    if not calib_dir.exists():
        log.error("calibration dir missing: %s — run scripts/capture_calibration.py", calib_dir)
        return 1

    # Wipe stale outputs.
    for path in (out, Path(str(out) + ".data"), Path(str(out) + "_data")):
        if path.exists():
            path.unlink()

    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )

    method_map = {
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
        "minmax": CalibrationMethod.MinMax,
    }
    activation_type = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
        "qint16": QuantType.QInt16,
        "quint16": QuantType.QUInt16,
    }[args.activation_type]

    nodes_to_exclude = [] if args.no_exclude_heads else [AUDIO_HEADS_NODE]

    reader = make_reader(
        calib_dir, max_samples=args.max_samples, filter_seq_len=args.filter_seq_len,
    )

    extra_options: dict = {}
    if args.method == "percentile":
        extra_options["CalibPercentile"] = args.percentile

    log.info(
        "quantize_static: method=%s percentile=%s per_channel=%s "
        "activation=%s exclude=%s",
        args.method, args.percentile, not args.per_tensor,
        args.activation_type, nodes_to_exclude or "none",
    )
    t0 = time.perf_counter()
    quantize_static(
        model_input=str(src),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=not args.per_tensor,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=method_map[args.method],
        nodes_to_exclude=nodes_to_exclude or None,
        use_external_data_format=True,
        extra_options=extra_options,
    )
    log.info("  done in %.1fs", time.perf_counter() - t0)

    graph_size = out.stat().st_size
    data_size = 0
    for cand in (Path(str(out) + ".data"), Path(str(out) + "_data")):
        if cand.exists():
            data_size = cand.stat().st_size
            break
    log.info(
        "On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
        graph_size / 1e6, data_size / 1e6, (graph_size + data_size) / 1e6,
    )

    import resource
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.info("Peak RSS: %.2f GB", peak_kb / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
