#!/usr/bin/env python3
"""PyTorch driver with persistent prefix KV cache across diffusion steps.

Sibling of ``onnx_driver.py`` and ``llama_driver.py``: load the full PyTorch
``OmniVoice`` for tokenizer / mask construction / Higgs decode, then
monkey-patch ``model.forward`` so the LLM body is split into:

  1. A cold "prefix" forward (text + style positions) executed once per row,
     with the resulting K/V projections stored in a per-row :class:`DynamicCache`.
  2. A hot "audio" forward that decodes only the audio region against the
     cached prefix K/V on every subsequent diffusion step.

The diffusion loop in ``omnivoice/models/omnivoice.py:1254-1297`` is unchanged;
it sees only ``self.forward(...)`` returning logits. Cache invalidation is
content-addressed via blake2b over prefix embedding bytes plus a hook that
clears state at the start of every ``model.generate`` call.

Usage:
    python pytorch_kv_driver.py "Hello world." [--out out.wav] [--no-prefix-cache]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from transformers import DynamicCache

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import OmniVoiceModelOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pytorch-kv-driver")

VOICE_INSTRUCTS = {
    "alloy":   "Male, Young Adult, Low Pitch",
    "echo":    "Male, Middle-aged, Moderate Pitch, British Accent",
    "fable":   "Female, Young Adult, High Pitch",
    "onyx":    "Male, Elderly, Very Low Pitch",
    "nova":    "Female, Middle-aged, Moderate Pitch",
    "shimmer": "Female, Young Adult, Very High Pitch, American Accent",
}


@dataclass
class StepStats:
    n_calls: int = 0
    total_seconds: float = 0.0
    cold_calls: int = 0
    cold_seconds: float = 0.0
    hot_calls: int = 0
    hot_seconds: float = 0.0

    def reset(self) -> None:
        self.n_calls = 0
        self.total_seconds = 0.0
        self.cold_calls = 0
        self.cold_seconds = 0.0
        self.hot_calls = 0
        self.hot_seconds = 0.0

    def add(self, dt: float) -> None:
        self.n_calls += 1
        self.total_seconds += dt

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.total_seconds / max(1, self.n_calls)

    @property
    def cold_mean_ms(self) -> float:
        return 1000.0 * self.cold_seconds / max(1, self.cold_calls)

    @property
    def hot_mean_ms(self) -> float:
        return 1000.0 * self.hot_seconds / max(1, self.hot_calls)


def _signature(prefix_embed: torch.Tensor) -> bytes:
    """blake2b digest of a prefix embedding tensor's raw fp32 bytes."""
    arr = prefix_embed.detach().to(torch.float32).cpu().numpy()
    return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()


def install_pytorch_kv_forward(
    model: OmniVoice,
    stats: StepStats,
    enabled: bool = True,
) -> None:
    """Replace ``model.forward`` with a prefix-cached equivalent.

    When ``enabled`` is False, the replacement falls back to the stock
    ``model.forward`` (used as the A/B baseline for benchmarking).
    """

    state = {
        "gen_id": 0,
        "caches": {},        # row_idx -> DynamicCache
        "prefix_lens": {},   # row_idx -> int
        "prefix_sigs": {},   # row_idx -> bytes
    }

    orig_forward = model.forward
    orig_generate = model.generate

    def _clear_cache_state() -> None:
        state["gen_id"] += 1
        state["caches"].clear()
        state["prefix_lens"].clear()
        state["prefix_sigs"].clear()

    def generate_wrapper(self, *args, **kwargs):
        _clear_cache_state()
        return orig_generate(*args, **kwargs)

    if enabled:
        model.generate = types.MethodType(generate_wrapper, model)

    def forward(
        self: OmniVoice,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        labels=None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids=None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> OmniVoiceModelOutput:
        if not enabled:
            return orig_forward(
                input_ids=input_ids,
                audio_mask=audio_mask,
                labels=labels,
                attention_mask=attention_mask,
                document_ids=document_ids,
                position_ids=position_ids,
            )

        if attention_mask is None:
            raise RuntimeError(
                "pytorch_kv driver requires an explicit 4D attention_mask "
                "(the training-time document_ids/flex_attention path is unsupported)."
            )

        B, _, S = input_ids.shape
        device = input_ids.device
        dtype_embed = self.llm.get_input_embeddings().weight.dtype

        # inputs_embeds: [B, S, H] — cheap (two embedding lookups + sum + where).
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        H = inputs_embeds.shape[-1]

        # Per-row real length: count of True in attention_mask[:, 0, :, 0].
        real_lens = attention_mask[:, 0, :, 0].to(torch.bool).sum(dim=-1).tolist()

        out_hidden = torch.zeros((B, S, H), dtype=inputs_embeds.dtype, device=device)

        t0 = time.perf_counter()
        for b in range(B):
            real_len = int(real_lens[b])
            if real_len <= 0:
                continue

            row_audio_mask = audio_mask[b, :real_len]
            prefix_len = int((~row_audio_mask).sum().item())
            audio_len = real_len - prefix_len

            row_embeds = inputs_embeds[b : b + 1, :real_len, :]  # [1, real_len, H]

            if prefix_len == 0 or audio_len == 0:
                # Nothing to cache (uncond rows have no text prefix; the
                # full-text edge case has no audio to decode). Run a plain
                # bidirectional forward and skip the cache machinery.
                row_mask = torch.ones(
                    (1, 1, real_len, real_len), dtype=torch.bool, device=device
                )
                row_pos = torch.arange(
                    real_len, dtype=torch.long, device=device
                ).unsqueeze(0)
                row_out = self.llm(
                    inputs_embeds=row_embeds,
                    attention_mask=row_mask,
                    position_ids=row_pos,
                    return_dict=True,
                )
                out_hidden[b, :real_len, :] = row_out.last_hidden_state[0]
                continue

            sig = _signature(row_embeds[0, :prefix_len, :])
            cache: Optional[DynamicCache] = state["caches"].get(b)
            cached_sig = state["prefix_sigs"].get(b)

            if cache is None or cached_sig != sig:
                # Cold prefix forward. Drop any stale cache for this row.
                state["caches"].pop(b, None)
                state["prefix_sigs"].pop(b, None)
                state["prefix_lens"].pop(b, None)

                cache = DynamicCache(config=self.llm.config)
                prefix_mask = torch.ones(
                    (1, 1, prefix_len, prefix_len), dtype=torch.bool, device=device
                )
                prefix_pos = torch.arange(
                    prefix_len, dtype=torch.long, device=device
                ).unsqueeze(0)
                t_cold = time.perf_counter()
                self.llm(
                    inputs_embeds=row_embeds[:, :prefix_len, :],
                    attention_mask=prefix_mask,
                    position_ids=prefix_pos,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                stats.cold_seconds += time.perf_counter() - t_cold
                stats.cold_calls += 1

                state["caches"][b] = cache
                state["prefix_sigs"][b] = sig
                state["prefix_lens"][b] = prefix_len
            else:
                # Cache hit. The previous step's hot forward grew the cache
                # to prefix_len + audio_len; trim it back so this step's hot
                # decode sees only the prefix K/V.
                if cache.get_seq_length() > prefix_len:
                    cache.crop(prefix_len)

            # Hot decode: audio queries attend to cached prefix K/V + their own K/V.
            audio_mask_4d = torch.ones(
                (1, 1, audio_len, real_len), dtype=torch.bool, device=device
            )
            audio_pos = torch.arange(
                prefix_len, real_len, dtype=torch.long, device=device
            ).unsqueeze(0)
            t_hot = time.perf_counter()
            row_out = self.llm(
                inputs_embeds=row_embeds[:, prefix_len:, :],
                attention_mask=audio_mask_4d,
                position_ids=audio_pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            stats.hot_seconds += time.perf_counter() - t_hot
            stats.hot_calls += 1

            out_hidden[b, prefix_len:real_len, :] = row_out.last_hidden_state[0]

            # Trim cache back to prefix_len so it's reusable next step
            # without leaking this step's audio K/V into the prefix slot.
            cache.crop(prefix_len)

        stats.add(time.perf_counter() - t0)

        # audio_heads is identical to omnivoice.forward.
        logits_flat = self.audio_heads(out_hidden)
        Bf, Sf, _ = logits_flat.shape
        audio_logits = logits_flat.view(
            Bf, Sf, self.config.num_audio_codebook, self.config.audio_vocab_size
        ).permute(0, 2, 1, 3)
        return OmniVoiceModelOutput(loss=None, logits=audio_logits)

    model.forward = types.MethodType(forward, model)


def synthesize(
    model: OmniVoice,
    text: str,
    instruct: str,
    num_step: int,
    speed: Optional[float],
) -> np.ndarray:
    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )
    audios = model.generate(
        text=text,
        language=None,
        instruct=instruct,
        speed=speed,
        generation_config=gen_config,
    )
    return np.asarray(audios[0], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--voice", default="nova", choices=sorted(VOICE_INSTRUCTS))
    p.add_argument("--instruct", default=None)
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="Disable the prefix cache (run stock PyTorch fp32 path "
                   "for A/B benchmarking)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch OmniVoice ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    log.info("PyTorch model loaded in %.1fs", time.time() - t0)

    stats = StepStats()
    install_pytorch_kv_forward(model, stats, enabled=not args.no_prefix_cache)

    instruct = args.instruct or VOICE_INSTRUCTS[args.voice]
    log.info(
        "synthesize cache=%s voice=%s chars=%d num_step=%d speed=%s",
        "off" if args.no_prefix_cache else "on",
        args.voice, len(args.text), args.num_step, args.speed,
    )

    t0 = time.perf_counter()
    waveform = synthesize(model, args.text, instruct, args.num_step, args.speed)
    wall = time.perf_counter() - t0

    sr = model.sampling_rate
    audio_seconds = waveform.shape[-1] / sr
    rtf = wall / audio_seconds if audio_seconds > 0 else float("inf")

    sf.write(args.out, waveform, sr, subtype="PCM_16")

    if not args.no_prefix_cache and stats.n_calls > 0:
        log.info(
            "generated %.2fs audio in %.2fs wall, RTF=%.3f | LLM steps=%d, "
            "mean=%.1f ms (cold=%d×%.1fms, hot=%d×%.1fms)",
            audio_seconds, wall, rtf, stats.n_calls, stats.mean_ms,
            stats.cold_calls, stats.cold_mean_ms,
            stats.hot_calls, stats.hot_mean_ms,
        )
    else:
        log.info(
            "generated %.2fs audio in %.2fs wall, RTF=%.3f",
            audio_seconds, wall, rtf,
        )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
