#!/usr/bin/env python3
"""Single-step PyTorch vs ONNX-fp32 logits parity check.

Builds one realistic batch of inputs (mimicking what _generate_iterative
constructs at step 0), runs both backends, and reports max abs diff,
mean abs diff, and cosine similarity. Bypasses the diffusion loop so any
divergence is from the export itself, not loop-amplified RNG drift.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parity")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice  # noqa: E402

DEFAULT_FP32 = REPO_ROOT / "onnx" / "omnivoice-step.onnx"


def make_inputs(model: OmniVoice, B: int, S: int, seed: int = 0):
    cfg = model.config
    C = cfg.num_audio_codebook
    V = cfg.audio_vocab_size
    text_vocab = cfg.llm_config.vocab_size
    g = torch.Generator(device="cpu").manual_seed(seed)

    text_ids = torch.randint(0, min(1000, text_vocab), (B, 1, S), generator=g, dtype=torch.long)
    audio_ids = torch.randint(0, V, (B, C - 1, S), generator=g, dtype=torch.long)
    input_ids = torch.cat([text_ids, audio_ids], dim=1).contiguous()

    audio_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_mask[:, S // 2 :] = True

    attention_mask = torch.ones(B, 1, S, S, dtype=torch.bool)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    return input_ids, audio_mask, attention_mask, position_ids


def report(label: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = a - b
    abs_diff = np.abs(diff)
    rel = abs_diff / (np.abs(b) + 1e-9)
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    cos = float(
        np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)
    )
    print(f"  {label}")
    print(f"    shape={a.shape}")
    print(f"    max abs diff   = {abs_diff.max():.6e}")
    print(f"    mean abs diff  = {abs_diff.mean():.6e}")
    print(f"    median abs diff= {np.median(abs_diff):.6e}")
    print(f"    p99 abs diff   = {np.quantile(abs_diff, 0.99):.6e}")
    print(f"    max rel diff   = {rel.max():.6e}")
    print(f"    cosine sim     = {cos:.10f}")
    # argmax disagreement at the last dim is the actually-relevant divergence
    # for sampling: how often do we pick a different codebook token?
    a_argmax = a.argmax(axis=-1)
    b_argmax = b.argmax(axis=-1)
    disagree = (a_argmax != b_argmax).mean()
    print(f"    argmax disagree= {disagree:.4%}  ({int((a_argmax != b_argmax).sum())}/{a_argmax.size})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--onnx", default=str(DEFAULT_FP32))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="attn_implementation for the PyTorch reference run")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch (attn=%s) ...", args.attn)
    model = OmniVoice.from_pretrained(args.model, attn_implementation=args.attn)
    model.to("cpu").eval()

    log.info("Building inputs (B=%d, S=%d, seed=%d) ...", args.batch, args.seq_len, args.seed)
    input_ids, audio_mask, attention_mask, position_ids = make_inputs(
        model, args.batch, args.seq_len, args.seed
    )

    log.info("PyTorch forward ...")
    with torch.inference_mode():
        # Mirror _generate_iterative call site: no position_ids passed.
        t0 = time.perf_counter()
        torch_out_default = model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
        ).logits.to(torch.float32).cpu().numpy()
        dt_torch_default = time.perf_counter() - t0

        # Same call but with explicit position_ids (matches what we feed ONNX).
        t0 = time.perf_counter()
        torch_out_explicit = model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits.to(torch.float32).cpu().numpy()
        dt_torch_explicit = time.perf_counter() - t0
    log.info("PyTorch (default pos_ids) wall=%.2fs, (explicit pos_ids) wall=%.2fs",
             dt_torch_default, dt_torch_explicit)

    log.info("ONNX session: %s", args.onnx)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    expected = {i.name for i in sess.get_inputs()}
    log.info("  expects: %s", sorted(expected))

    feeds = {
        "input_ids": input_ids.numpy(),
        "audio_mask": audio_mask.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    if "position_ids" in expected:
        feeds["position_ids"] = position_ids.numpy()

    t0 = time.perf_counter()
    onnx_out = sess.run(["logits"], feeds)[0]
    dt_onnx = time.perf_counter() - t0
    log.info("ONNX wall=%.2fs", dt_onnx)

    print("\n=== Logit parity ===")
    report("PyTorch(default pos_ids) vs ONNX", torch_out_default, onnx_out)
    print()
    report("PyTorch(explicit pos_ids) vs ONNX", torch_out_explicit, onnx_out)
    print()
    report("PyTorch(default) vs PyTorch(explicit pos_ids)", torch_out_default, torch_out_explicit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
