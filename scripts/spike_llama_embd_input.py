#!/usr/bin/env python3
"""Spike #2b: feed pre-computed embeddings (not tokens) into llama.cpp.

OmniVoice computes its inputs_embeds in PyTorch (text embeddings + audio
embeddings, summed across codebooks, where-merged by audio_mask). The LLM
must receive that pre-computed [B, S, hidden] tensor, not token IDs.

This spike confirms that llama-cpp-python's llama_batch_init(embd=n_embd, ...)
and llama_batch.embd field actually work for non-causal forward passes on
Qwen3-0.6B, and that we get hidden states out the other side.
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
    llama_batch_free,
    llama_batch_init,
    llama_decode,
    llama_set_causal_attn,
)

GGUF = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/"
    "snapshots/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf"
)


def main() -> int:
    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    n_ctx = 1024

    print(f"Loading {Path(GGUF).name} (threads={threads}) ...")
    llm = Llama(
        model_path=GGUF,
        n_ctx=n_ctx,
        n_threads=threads,
        n_threads_batch=threads,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
        logits_all=False,
        verbose=False,
    )
    ctx = llm._ctx.ctx
    llama_set_causal_attn(ctx, False)

    n_embd = llm.n_embd()
    print(f"n_embd = {n_embd}")

    rng = np.random.default_rng(0)
    S = 300

    # -- Path A: token-based forward (sanity reference) --
    tokens = rng.integers(0, 32000, size=S, dtype=np.int32).tolist()
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(tokens)
    dt_tok = time.perf_counter() - t0
    embd_ptr = llama_cpp.llama_get_embeddings(ctx)
    arr_tok = np.ctypeslib.as_array(
        ctypes.cast(embd_ptr, ctypes.POINTER(ctypes.c_float)),
        shape=(S, n_embd),
    ).copy()
    print(f"token path: forward {dt_tok*1000:.1f} ms  hidden[0,:3]={arr_tok[0,:3]}")

    # -- Path B: pre-computed embedding forward --
    # We need to mimic exactly what eval_tokens does internally, but with embd.
    # Build a batch with embd populated, no token field, full-sequence,
    # logits=True on every position so all hidden states get surfaced.
    #
    # Strategy: take the SAME embeddings the token path would have produced
    # (via llama's input embedding lookup), feed THOSE as embd[], confirm we
    # get the same (or close to same) output. There's no public API to read
    # the input embedding matrix, so instead we use a synthetic embedding,
    # check shapes / non-NaN / determinism.
    embd_input = rng.standard_normal(size=(S, n_embd), dtype=np.float32) * 0.1
    embd_flat = np.ascontiguousarray(embd_input.reshape(-1))

    batch = llama_batch_init(S, n_embd, 1)
    batch.n_tokens = S
    # Copy the embd values into the batch's owned buffer.
    ctypes.memmove(
        batch.embd,
        embd_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        embd_flat.nbytes,
    )
    # Set pos, seq_id, logits per-token
    for i in range(S):
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1  # surface output for every position

    # Properly clear the KV cache before re-decoding with embd inputs.
    mem = llama_cpp.llama_get_memory(ctx)
    llama_cpp.llama_memory_clear(mem, True)
    llm.reset()

    t0 = time.perf_counter()
    rc = llama_decode(ctx, batch)
    dt_emb = time.perf_counter() - t0
    print(f"embd path: llama_decode rc={rc}  forward {dt_emb*1000:.1f} ms")
    if rc != 0:
        print("FAILED: llama_decode returned nonzero")
        llama_batch_free(batch)
        return 1

    embd_ptr2 = llama_cpp.llama_get_embeddings(ctx)
    arr_emb = np.ctypeslib.as_array(
        ctypes.cast(embd_ptr2, ctypes.POINTER(ctypes.c_float)),
        shape=(S, n_embd),
    ).copy()
    finite = np.isfinite(arr_emb).all()
    print(
        f"embd path: hidden shape={arr_emb.shape} finite={finite}  "
        f"hidden[0,:3]={arr_emb[0,:3]}  hidden[-1,:3]={arr_emb[-1,:3]}"
    )

    # Determinism check: re-run, expect identical output.
    rc2 = llama_decode(ctx, batch)
    arr_emb2 = np.ctypeslib.as_array(
        ctypes.cast(llama_cpp.llama_get_embeddings(ctx), ctypes.POINTER(ctypes.c_float)),
        shape=(S, n_embd),
    ).copy()
    same = np.allclose(arr_emb, arr_emb2, atol=1e-5)
    print(f"determinism: identical={same}  max_abs_diff={np.abs(arr_emb-arr_emb2).max():.2e}")

    llama_batch_free(batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
