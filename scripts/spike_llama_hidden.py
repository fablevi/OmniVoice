#!/usr/bin/env python3
"""Spike: prove we can get [B, T, hidden_size] hidden states from llama-cpp-python.

Loads vanilla Qwen3-0.6B Q8_0 GGUF, sets pooling=NONE + causal=False, runs
several full-sequence forward passes, prints output shapes and timings.

This is a feasibility check — does the API plumbing exist for OmniVoice's
diffusion driver pattern? We don't care about the actual values yet.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

import llama_cpp
from llama_cpp import (
    LLAMA_POOLING_TYPE_NONE,
    Llama,
    llama_set_causal_attn,
)

GGUF = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/"
    "snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf"
)


def main() -> int:
    if not Path(GGUF).exists():
        print(f"GGUF not found: {GGUF}", file=sys.stderr)
        return 1

    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    n_ctx = 1024

    print(f"Loading {Path(GGUF).name} (threads={threads}, n_ctx={n_ctx}) ...")
    llm = Llama(
        model_path=GGUF,
        n_ctx=n_ctx,
        n_threads=threads,
        n_threads_batch=threads,
        embedding=True,                 # surface hidden states instead of logits
        pooling_type=LLAMA_POOLING_TYPE_NONE,  # per-token, no pooling
        logits_all=False,
        verbose=False,
    )
    llama_set_causal_attn(llm._ctx.ctx, False)

    n_embd = llm.n_embd()
    print(f"n_embd = {n_embd}")

    rng = np.random.default_rng(0)
    seq_lens = [256, 300, 512]
    for S in seq_lens:
        tokens = rng.integers(0, 32000, size=S, dtype=np.int32).tolist()
        # warm
        llm.reset()
        llm.eval(tokens)
        t0 = time.perf_counter()
        for _ in range(3):
            llm.reset()
            llm.eval(tokens)
        dt = (time.perf_counter() - t0) / 3.0

        # Try the 3 candidate retrieval paths.
        ctx = llm._ctx.ctx
        n_tokens = S

        # 1. llama_get_embeddings(ctx) → flat (n_tokens * n_embd,) float pointer
        embd_ptr = llama_cpp.llama_get_embeddings(ctx)
        if embd_ptr:
            arr = np.ctypeslib.as_array(
                ctypes.cast(embd_ptr, ctypes.POINTER(ctypes.c_float)),
                shape=(n_tokens, n_embd),
            )
            print(
                f"S={S}  forward {dt*1000:.1f} ms  "
                f"embeddings shape={arr.shape} dtype={arr.dtype} "
                f"first3=[{arr[0,0]:+.4f}, {arr[0,1]:+.4f}, {arr[0,2]:+.4f}] "
                f"finite={np.isfinite(arr).all()}"
            )
        else:
            print(f"S={S}  llama_get_embeddings returned NULL")

    # Test the per-token retrieval path llama_get_embeddings_ith / seq.
    print("\nllama_get_embeddings_ith:")
    ctx = llm._ctx.ctx
    for i in [0, 1, S - 1]:
        p = llama_cpp.llama_get_embeddings_ith(ctx, i)
        if p:
            v = np.ctypeslib.as_array(
                ctypes.cast(p, ctypes.POINTER(ctypes.c_float)), shape=(n_embd,)
            )
            print(f"  ith={i}: first3=[{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]")
        else:
            print(f"  ith={i}: NULL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
