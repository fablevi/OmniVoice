#!/usr/bin/env python3
"""LLM hidden-state parity check: PyTorch fp32 vs llama.cpp Q8_0/Q6_K/Q4_K_M.

Builds realistic OmniVoice diffusion-step inputs, computes inputs_embeds in
PyTorch, then runs the LLM body two ways:
  (a) PyTorch m.llm(inputs_embeds=..., attention_mask=4D-bool, position_ids=...)
  (b) llama.cpp via the embd-input batch path with causal_attn=False

Compares last_hidden_state (the input to audio_heads) directly, then pushes
both through audio_heads and reports argmax disagreement on the resulting
[B, C, S, V] logits — that's the metric that actually drives diffusion
sampling and it's the headline number to beat (vs ORT int8's 49% disagree).
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

import llama_cpp
from llama_cpp import (
    LLAMA_POOLING_TYPE_NONE,
    Llama,
    llama_batch_free,
    llama_batch_init,
    llama_decode,
    llama_set_causal_attn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parity-llama")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice  # noqa: E402


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
    audio_mask[:, S // 2:] = True
    attention_mask = torch.ones(B, 1, S, S, dtype=torch.bool)
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    return input_ids, audio_mask, attention_mask, position_ids


def compute_inputs_embeds(model: OmniVoice, input_ids: torch.Tensor,
                          audio_mask: torch.Tensor) -> torch.Tensor:
    """Mirror omnivoice.OmniVoice._prepare_embed_inputs."""
    text_embeds = model.llm.get_input_embeddings()(input_ids[:, 0, :])
    shifted = (
        input_ids * audio_mask.unsqueeze(1).to(input_ids.dtype)
    ) + model.codebook_layer_offsets.view(1, -1, 1)
    audio_embeds = model.audio_embeddings(shifted).sum(dim=1)
    return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)


def run_llamacpp(gguf: Path, inputs_embeds: np.ndarray, threads: int) -> tuple[np.ndarray, float]:
    """Forward [B, S, hidden] through llama.cpp; returns hidden states + per-call ms."""
    B, S, H = inputs_embeds.shape
    log.info("  llama.cpp: loading %s (B=%d, S=%d, H=%d) ...", gguf.name, B, S, H)

    n_ctx = max(1024, S + 64)
    llm = Llama(
        model_path=str(gguf),
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
    if n_embd != H:
        raise RuntimeError(f"GGUF n_embd={n_embd} but inputs hidden={H}")

    out = np.zeros((B, S, H), dtype=np.float32)
    times = []
    for b in range(B):
        embd_flat = np.ascontiguousarray(inputs_embeds[b].reshape(-1).astype(np.float32))
        batch = llama_batch_init(S, n_embd, 1)
        batch.n_tokens = S
        ctypes.memmove(
            batch.embd,
            embd_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            embd_flat.nbytes,
        )
        for i in range(S):
            batch.pos[i] = i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1

        mem = llama_cpp.llama_get_memory(ctx)

        # Warm.
        llama_cpp.llama_memory_clear(mem, True)
        rc_warm = llama_decode(ctx, batch)
        if rc_warm != 0:
            llama_batch_free(batch)
            raise RuntimeError(f"warm llama_decode rc={rc_warm} on batch {b}")

        # Measure (clear KV first so positions 0..S-1 don't collide).
        llama_cpp.llama_memory_clear(mem, True)
        t0 = time.perf_counter()
        rc = llama_decode(ctx, batch)
        dt = time.perf_counter() - t0
        if rc != 0:
            llama_batch_free(batch)
            raise RuntimeError(f"timed llama_decode rc={rc} on batch {b}")
        times.append(dt)

        embd_ptr = llama_cpp.llama_get_embeddings(ctx)
        arr = np.ctypeslib.as_array(
            ctypes.cast(embd_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(S, n_embd),
        ).copy()
        out[b] = arr
        llama_batch_free(batch)

    mean_ms = 1000.0 * sum(times) / len(times)
    log.info("  llama.cpp: per-fwd mean=%.1f ms (n=%d)", mean_ms, len(times))
    del llm
    return out, mean_ms


def report_hidden(label: str, a: np.ndarray, b: np.ndarray) -> None:
    abs_diff = np.abs(a - b)
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    cos = float(
        np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12)
    )
    print(f"  hidden-state {label}")
    print(f"    shape          = {a.shape}")
    print(f"    max abs diff   = {abs_diff.max():.4e}")
    print(f"    mean abs diff  = {abs_diff.mean():.4e}")
    print(f"    p99 abs diff   = {np.quantile(abs_diff, 0.99):.4e}")
    print(f"    cosine sim     = {cos:.8f}")


def report_logits(label: str, a: np.ndarray, b: np.ndarray) -> None:
    abs_diff = np.abs(a - b)
    a_arg = a.argmax(axis=-1)
    b_arg = b.argmax(axis=-1)
    disagree = (a_arg != b_arg).mean()
    print(f"  audio_heads {label}")
    print(f"    shape          = {a.shape}")
    print(f"    max abs diff   = {abs_diff.max():.4e}")
    print(f"    mean abs diff  = {abs_diff.mean():.4e}")
    print(f"    argmax disagree= {disagree:.4%}  ({int((a_arg != b_arg).sum())}/{a_arg.size})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--gguf", default=None,
                   help="Path to a single GGUF (else sweeps Q8_0/Q6_K/Q4_K_M)")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)

    log.info("Loading PyTorch OmniVoice (sdpa) ...")
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()

    log.info("Building inputs (B=%d, S=%d) ...", args.batch, args.seq_len)
    input_ids, audio_mask, attention_mask, position_ids = make_inputs(
        model, args.batch, args.seq_len, args.seed
    )

    with torch.inference_mode():
        log.info("Computing inputs_embeds (PyTorch) ...")
        inputs_embeds = compute_inputs_embeds(model, input_ids, audio_mask)

        log.info("PyTorch m.llm forward ...")
        t0 = time.perf_counter()
        torch_hidden = model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        ).last_hidden_state.float().cpu().numpy()
        dt_torch = time.perf_counter() - t0
        log.info("  PyTorch: per-fwd %.1f ms", 1000 * dt_torch)

        # Compute audio_heads on the PyTorch hidden-state too (reference logits).
        torch_logits = model.audio_heads(
            torch.from_numpy(torch_hidden)
        ).cpu().numpy()
        # [B, S, C*V] -> [B, S, C, V] -> [B, C, S, V]
        B, S = torch_logits.shape[:2]
        C = model.config.num_audio_codebook
        V = model.config.audio_vocab_size
        torch_logits = torch_logits.reshape(B, S, C, V).transpose(0, 2, 1, 3)

    if args.gguf:
        ggufs = [Path(args.gguf)]
    else:
        gdir = REPO_ROOT / "gguf"
        ggufs = [
            gdir / "omnivoice-qwen3-q8_0.gguf",
            gdir / "omnivoice-qwen3-q6_k.gguf",
            gdir / "omnivoice-qwen3-q4_k_m.gguf",
        ]

    embeds_np = inputs_embeds.float().cpu().numpy()

    rows: list[dict] = []
    for gpath in ggufs:
        if not gpath.exists():
            log.warning("missing %s, skipping", gpath)
            continue
        print()
        print(f"=== {gpath.name} ===")
        llama_hidden, mean_ms = run_llamacpp(gpath, embeds_np, args.threads)
        report_hidden("(llama vs torch)", llama_hidden, torch_hidden)

        # Push llama hidden states through PyTorch audio_heads, compare logits.
        with torch.inference_mode():
            llama_logits = model.audio_heads(
                torch.from_numpy(llama_hidden)
            ).cpu().numpy()
        llama_logits = llama_logits.reshape(B, S, C, V).transpose(0, 2, 1, 3)
        report_logits("(llama vs torch)", llama_logits, torch_logits)

        # Quick summary line for the eventual table.
        diff = np.abs(llama_hidden - torch_hidden)
        ll_arg = llama_logits.argmax(axis=-1)
        tt_arg = torch_logits.argmax(axis=-1)
        disagree = (ll_arg != tt_arg).mean()
        rows.append({
            "name": gpath.name,
            "ms": mean_ms,
            "hidden_max": float(diff.max()),
            "hidden_mean": float(diff.mean()),
            "argmax_disagree": float(disagree),
        })

    print("\n=== summary ===")
    print(f"{'variant':30s}  {'ms/fwd':>8s}  {'h_max':>9s}  {'h_mean':>9s}  {'argmax_dis':>11s}")
    for r in rows:
        print(
            f"{r['name']:30s}  {r['ms']:8.1f}  {r['hidden_max']:9.3e}  "
            f"{r['hidden_mean']:9.3e}  {r['argmax_disagree']:11.4%}"
        )
    print(f"\nReference: ORT dynamic int8 had 49% argmax disagree (PERFORMANCE.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
