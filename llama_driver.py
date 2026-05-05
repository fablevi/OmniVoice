#!/usr/bin/env python3
"""llama.cpp-backed driver for OmniVoice (voice-design mode).

Mirrors onnx_driver.py: load full PyTorch ``OmniVoice`` for tokenizer / mask
construction / Higgs decode, then monkey-patch ``model.forward`` so the
LLM body runs in llama.cpp via llama-cpp-python's embd-input batch path.
``audio_heads`` and everything else stay in PyTorch.

Why: llama.cpp's Q8_0 GEMM (per-block fp16 scales every 32 weights) preserves
output quality far better than ORT's per-tensor dynamic int8 — 6.8% vs 49%
argmax disagreement on the parity check.

Usage:
    python llama_driver.py "Hello world." [--out out.wav] [--quant q8_0|q6_k|q4_k_m]
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
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

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import OmniVoiceModelOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llama-driver")

REPO_ROOT = Path(__file__).resolve().parent
GGUF_DIR = REPO_ROOT / "gguf"
QUANT_DEFAULT = "q8_0"
QUANT_PATHS = {
    "q8_0":   GGUF_DIR / "omnivoice-qwen3-q8_0.gguf",
    "q6_k":   GGUF_DIR / "omnivoice-qwen3-q6_k.gguf",
    "q5_k_m": GGUF_DIR / "omnivoice-qwen3-q5_k_m.gguf",
    "q5_k_s": GGUF_DIR / "omnivoice-qwen3-q5_k_s.gguf",
    "q5_1":   GGUF_DIR / "omnivoice-qwen3-q5_1.gguf",
    "q5_0":   GGUF_DIR / "omnivoice-qwen3-q5_0.gguf",
    "q4_k_m": GGUF_DIR / "omnivoice-qwen3-q4_k_m.gguf",
    "q4_k_s": GGUF_DIR / "omnivoice-qwen3-q4_k_s.gguf",
    "q4_0":   GGUF_DIR / "omnivoice-qwen3-q4_0.gguf",
    "q4_1":   GGUF_DIR / "omnivoice-qwen3-q4_1.gguf",
    "iq4_xs": GGUF_DIR / "omnivoice-qwen3-iq4_xs.gguf",
    "iq4_nl": GGUF_DIR / "omnivoice-qwen3-iq4_nl.gguf",
    "f16":    GGUF_DIR / "omnivoice-qwen3.gguf",
}

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

    def reset(self) -> None:
        self.n_calls = 0
        self.total_seconds = 0.0

    def add(self, dt: float) -> None:
        self.n_calls += 1
        self.total_seconds += dt

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.total_seconds / max(1, self.n_calls)


def make_llm(gguf: Path, threads: int, n_ctx: int, n_seq_max: int = 2) -> Llama:
    """Construct a Llama context with ``n_seq_max`` sequence slots.

    The default Llama wrapper hard-wires n_seq_max=1, which blocks batched
    multi-sequence decode (used here for CFG cond+uncond in a single
    ``llama_decode`` call). Monkey-patch the default-params factory so the
    underlying llama_context is created with the right seq capacity.
    """
    log.info("Loading GGUF: %s (threads=%d, n_ctx=%d, n_seq_max=%d)",
             gguf.name, threads, n_ctx, n_seq_max)
    # The Llama wrapper resolves llama_context_default_params from the C
    # bindings module (llama_cpp.llama_cpp), not the top-level package
    # alias. Patch the bindings module directly.
    import llama_cpp.llama_cpp as _bindings
    orig_defaults = _bindings.llama_context_default_params

    def _patched_defaults():
        p = orig_defaults()
        p.n_seq_max = n_seq_max
        return p

    _bindings.llama_context_default_params = _patched_defaults
    try:
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
    finally:
        _bindings.llama_context_default_params = orig_defaults
    llama_set_causal_attn(llm._ctx.ctx, False)
    return llm


def llama_forward_one(
    llm: Llama,
    inputs_embeds_row: np.ndarray,  # [S, hidden] float32
) -> np.ndarray:
    """Run a single batch=1, fixed-length forward via llama.cpp embd input."""
    S, H = inputs_embeds_row.shape
    ctx = llm._ctx.ctx
    n_embd = llm.n_embd()
    if n_embd != H:
        raise RuntimeError(f"GGUF n_embd={n_embd} but inputs hidden={H}")

    embd_flat = np.ascontiguousarray(inputs_embeds_row.reshape(-1).astype(np.float32))
    batch = llama_batch_init(S, n_embd, 1)
    try:
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
        llama_cpp.llama_memory_clear(mem, True)
        rc = llama_decode(ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode rc={rc}")

        embd_ptr = llama_cpp.llama_get_embeddings(ctx)
        arr = np.ctypeslib.as_array(
            ctypes.cast(embd_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(S, n_embd),
        ).copy()
    finally:
        llama_batch_free(batch)
    return arr


def llama_forward_batched(
    llm: Llama,
    rows: list[np.ndarray],  # list of [L_b, hidden] float32, one per sequence
) -> list[np.ndarray]:
    """Run multiple sequences in a single ``llama_decode`` call.

    Used for OmniVoice CFG: cond at seq_id=0, uncond at seq_id=1, total
    n_tokens = sum(L_b). Each sequence has its own bidirectional attention
    graph (no cross-sequence attention) thanks to llama.cpp's seq_id
    masking. This recovers most of the batch-level matmul reuse that
    PyTorch gets for free with [B, S, ...] tensors.
    """
    if not rows:
        return []
    H = rows[0].shape[1]
    ctx = llm._ctx.ctx
    n_embd = llm.n_embd()
    if n_embd != H:
        raise RuntimeError(f"GGUF n_embd={n_embd} but inputs hidden={H}")

    lens = [r.shape[0] for r in rows]
    total = sum(lens)
    n_seq = len(rows)

    embd_concat = np.ascontiguousarray(
        np.concatenate([r.astype(np.float32) for r in rows], axis=0).reshape(-1)
    )
    batch = llama_batch_init(total, n_embd, n_seq)
    try:
        batch.n_tokens = total
        ctypes.memmove(
            batch.embd,
            embd_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            embd_concat.nbytes,
        )
        offset = 0
        for s, L in enumerate(lens):
            for k in range(L):
                i = offset + k
                batch.pos[i] = k             # position resets per sequence
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = s
                batch.logits[i] = 1
            offset += L

        mem = llama_cpp.llama_get_memory(ctx)
        llama_cpp.llama_memory_clear(mem, True)
        rc = llama_decode(ctx, batch)
        if rc != 0:
            raise RuntimeError(f"batched llama_decode rc={rc}")

        embd_ptr = llama_cpp.llama_get_embeddings(ctx)
        flat = np.ctypeslib.as_array(
            ctypes.cast(embd_ptr, ctypes.POINTER(ctypes.c_float)),
            shape=(total, n_embd),
        ).copy()
    finally:
        llama_batch_free(batch)

    out = []
    o = 0
    for L in lens:
        out.append(flat[o : o + L])
        o += L
    return out


def install_llama_forward(
    model: OmniVoice, llm: Llama, stats: StepStats, batched: bool = True
) -> None:
    """Replace ``model.forward`` so the LLM body runs in llama.cpp.

    Mirrors omnivoice/models/omnivoice.py:382-433. Inputs come in batched as
    [2*B, ...] for CFG (cond + uncond). Each row may have a real length less
    than the batched seq_len because of padding (uncond is shorter, with a
    diagonal-only attention pattern past u_len). We detect each row's real
    length from ``attention_mask[r, 0, :, 0]`` and run llama.cpp at that
    length — no padding needed since each row is its own sequence.

    If ``batched`` is True, all rows go into a single ``llama_decode`` call
    with distinct seq_ids (recovers batch-level matmul reuse). If False,
    rows run sequentially (legacy behaviour, useful as a baseline).
    """

    def forward(
        self: OmniVoice,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        labels=None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids=None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> OmniVoiceModelOutput:
        if attention_mask is None:
            raise RuntimeError(
                "llama driver requires an explicit 4D attention_mask "
                "(the training-time document_ids/flex_attention path is unsupported)."
            )

        B, _, S = input_ids.shape  # B here = 2 * actual_batch (CFG)
        # Compute inputs_embeds in PyTorch (this is cheap — just two embedding lookups).
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        embeds_np = inputs_embeds.detach().float().cpu().numpy()  # [B, S, H]

        # Detect each row's real length: count of True in
        # attention_mask[r, 0, :, 0]. For cond, that's the actual c_len; for
        # uncond, u_len.
        real_lens = (
            attention_mask[:, 0, :, 0].to(torch.bool).sum(dim=-1).cpu().numpy()
        )

        H = embeds_np.shape[-1]
        out_hidden = np.zeros((B, S, H), dtype=np.float32)

        t0 = time.perf_counter()
        if batched:
            rows = []
            slot_lens = []
            for b in range(B):
                L = int(real_lens[b])
                slot_lens.append(L)
                if L > 0:
                    rows.append(embeds_np[b, :L, :])
            hidden_rows = llama_forward_batched(llm, rows)
            j = 0
            for b, L in enumerate(slot_lens):
                if L <= 0:
                    continue
                out_hidden[b, :L, :] = hidden_rows[j]
                j += 1
        else:
            for b in range(B):
                L = int(real_lens[b])
                if L <= 0:
                    continue
                hidden_b = llama_forward_one(llm, embeds_np[b, :L, :])
                out_hidden[b, :L, :] = hidden_b
        stats.add(time.perf_counter() - t0)

        hidden_t = torch.from_numpy(out_hidden).to(inputs_embeds.dtype)
        # audio_heads stays in PyTorch (kept fp32 for quality, ~33MB).
        logits_flat = self.audio_heads(hidden_t)  # [B, S, C*V]
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
    p.add_argument("--quant", default=QUANT_DEFAULT,
                   choices=sorted(QUANT_PATHS),
                   help="GGUF quantization (default q8_0)")
    p.add_argument("--gguf", default=None,
                   help="Override path to GGUF (takes precedence over --quant)")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--n-ctx", type=int, default=2048,
                   help="llama.cpp KV ctx size (must exceed any seq_len seen)")
    p.add_argument("--no-batched-cfg", action="store_true",
                   help="Run cond/uncond as sequential forwards instead of one "
                   "batched llama_decode (legacy; used for A/B benchmarking)")
    p.add_argument("--seed", type=int, default=None,
                   help="torch.manual_seed before generate")
    args = p.parse_args()

    gguf_path = Path(args.gguf).resolve() if args.gguf else QUANT_PATHS[args.quant]
    if not gguf_path.exists():
        log.error("GGUF not found: %s — run scripts/export_qwen3_for_gguf.py "
                  "and llama-quantize first", gguf_path)
        return 1

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch OmniVoice (host-side helpers, audio_heads, Higgs) ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    log.info("PyTorch model loaded in %.1fs", time.time() - t0)

    llm = make_llm(gguf_path, threads=args.threads, n_ctx=args.n_ctx)
    stats = StepStats()
    install_llama_forward(model, llm, stats, batched=not args.no_batched_cfg)

    instruct = args.instruct or VOICE_INSTRUCTS[args.voice]
    log.info(
        "synthesize quant=%s voice=%s chars=%d num_step=%d speed=%s",
        args.quant if not args.gguf else gguf_path.name,
        args.voice, len(args.text), args.num_step, args.speed,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    waveform = synthesize(model, args.text, instruct, args.num_step, args.speed)
    wall = time.perf_counter() - t0

    sr = model.sampling_rate
    audio_seconds = waveform.shape[-1] / sr
    rtf = wall / audio_seconds if audio_seconds > 0 else float("inf")

    sf.write(args.out, waveform, sr, subtype="PCM_16")

    log.info(
        "generated %.2fs audio in %.2fs wall, RTF=%.3f | LLM batch-rows=%d, mean=%.1f ms",
        audio_seconds, wall, rtf, stats.n_calls, stats.mean_ms,
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
