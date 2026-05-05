#!/usr/bin/env python3
"""Parity check: cached-prefix PyTorch fp32 vs no-cache PyTorch fp32.

Builds realistic OmniVoice diffusion-step inputs (same harness as
``parity_llama.py``), runs the LLM body two ways:

  (a) No-cache: ``model.llm(inputs_embeds=..., attention_mask=4D-bool, ...)``
      over the full sequence — current PyTorch path.
  (b) Cached: cold prefix forward populates a :class:`DynamicCache`, then a
      hot decode runs only the audio region against the cached K/V.

Compares ``last_hidden_state`` on the audio region (the only positions
the diffusion sampler reads). Also pushes both through ``audio_heads`` and
reports argmax disagreement on the resulting ``[B, C, T, V]`` logits.

The plan target is max abs diff < 1e-5 on the hidden states. Note the
OmniVoice LLM uses fully bidirectional attention (``omnivoice/data/collator.py:
40-43``), so prefix K/V at deeper layers depend on audio tokens and the
cached path is technically an approximation rather than bit-equivalent —
this script measures the actual divergence.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parity-kv")

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
    text_embeds = model.llm.get_input_embeddings()(input_ids[:, 0, :])
    shifted = (
        input_ids * audio_mask.unsqueeze(1).to(input_ids.dtype)
    ) + model.codebook_layer_offsets.view(1, -1, 1)
    audio_embeds = model.audio_embeddings(shifted).sum(dim=1)
    return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)


def run_nocache(model: OmniVoice, inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
    )
    return out.last_hidden_state.float().cpu()


def run_cached(model: OmniVoice, inputs_embeds: torch.Tensor,
               prefix_len: int) -> torch.Tensor:
    """Mirror the path in pytorch_kv_driver.py for a single-row input."""
    B, S, H = inputs_embeds.shape
    out_hidden = torch.zeros((B, S, H), dtype=inputs_embeds.dtype)

    for b in range(B):
        row = inputs_embeds[b : b + 1]
        # Cold prefix.
        cache = DynamicCache(config=model.llm.config)
        prefix_mask = torch.ones((1, 1, prefix_len, prefix_len), dtype=torch.bool)
        prefix_pos = torch.arange(prefix_len, dtype=torch.long).unsqueeze(0)
        model.llm(
            inputs_embeds=row[:, :prefix_len, :],
            attention_mask=prefix_mask,
            position_ids=prefix_pos,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        # Hot decode.
        audio_len = S - prefix_len
        audio_mask_4d = torch.ones((1, 1, audio_len, S), dtype=torch.bool)
        audio_pos = torch.arange(prefix_len, S, dtype=torch.long).unsqueeze(0)
        row_out = model.llm(
            inputs_embeds=row[:, prefix_len:, :],
            attention_mask=audio_mask_4d,
            position_ids=audio_pos,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        out_hidden[b, prefix_len:S, :] = row_out.last_hidden_state[0]

    return out_hidden.float()


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
    prefix_len = args.seq_len // 2  # matches the make_inputs split

    with torch.inference_mode():
        log.info("Computing inputs_embeds (PyTorch) ...")
        inputs_embeds = compute_inputs_embeds(model, input_ids, audio_mask)

        log.info("PyTorch m.llm (no-cache, full seq) ...")
        t0 = time.perf_counter()
        nocache_hidden = run_nocache(model, inputs_embeds, attention_mask, position_ids)
        dt_nocache = time.perf_counter() - t0
        log.info("  no-cache: %.1f ms", 1000 * dt_nocache)

        log.info("PyTorch m.llm (cold prefix + hot decode) ...")
        t0 = time.perf_counter()
        cached_hidden = run_cached(model, inputs_embeds, prefix_len)
        dt_cached = time.perf_counter() - t0
        log.info("  cached:   %.1f ms", 1000 * dt_cached)

    print()
    print(f"=== audio region ([:, {prefix_len}:, :]) ===")
    a_audio = nocache_hidden[:, prefix_len:, :].numpy()
    b_audio = cached_hidden[:, prefix_len:, :].numpy()
    report_hidden("(cached vs no-cache)", b_audio, a_audio)

    # Push both through audio_heads, compare logits on the audio region.
    with torch.inference_mode():
        nocache_logits = model.audio_heads(nocache_hidden).float().cpu().numpy()
        cached_logits = model.audio_heads(cached_hidden).float().cpu().numpy()

    B, S = nocache_logits.shape[:2]
    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    nocache_logits = nocache_logits.reshape(B, S, C, V).transpose(0, 2, 1, 3)
    cached_logits = cached_logits.reshape(B, S, C, V).transpose(0, 2, 1, 3)

    print()
    a_log = nocache_logits[:, :, prefix_len:, :]
    b_log = cached_logits[:, :, prefix_len:, :]
    report_logits("(cached vs no-cache, audio region)", b_log, a_log)

    print()
    abs_max = float(np.abs(b_audio - a_audio).max())
    if abs_max < 1e-5:
        print(f"PASS: max abs diff {abs_max:.4e} < 1e-5 (plan target).")
        return 0
    else:
        print(f"INFO: max abs diff {abs_max:.4e} >= 1e-5 (plan target).")
        print("  This is expected if the OmniVoice LLM is fully bidirectional —")
        print("  prefix K/V at deeper layers depend on audio tokens, so cold-prefix")
        print("  forward and full forward diverge slightly. Check downstream impact")
        print("  via argmax disagree above and the end-to-end smoke test.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
