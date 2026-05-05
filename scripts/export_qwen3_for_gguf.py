#!/usr/bin/env python3
"""Extract OmniVoice's Qwen3 LLM body to a HF checkpoint for GGUF conversion.

OmniVoice fine-tunes a Qwen3-0.6B backbone (28 layers, hidden 1024, GQA 16/8,
vocab 151676). The body lives at ``model.llm`` as a ``Qwen3Model``. We wrap
it in ``Qwen3ForCausalLM`` (with tied LM head) so llama.cpp's
``convert_hf_to_gguf.py`` recognises the architecture. ``audio_heads`` is
NOT exported here — it stays in PyTorch in the driver.

Output layout matches what HF expects:
    qwen3_export/
        config.json
        model.safetensors
        tokenizer.json (copied from upstream Qwen3 if present)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export-qwen3")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "gguf" / "qwen3_export"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                   help="Source repo to copy tokenizer from (vocab matches)")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.model)
    src = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    src.eval()

    llm = src.llm
    cfg = llm.config
    log.info(
        "LLM: model_type=%s layers=%d hidden=%d heads=%d kv_heads=%d vocab=%d tie=%s",
        cfg.model_type,
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
        cfg.tie_word_embeddings,
    )

    # Wrap Qwen3Model as Qwen3ForCausalLM. tie_word_embeddings=True means the
    # LM head shares weights with embed_tokens — no extra parameters needed.
    log.info("Building Qwen3ForCausalLM wrapper ...")
    wrapper = Qwen3ForCausalLM(cfg)
    # Move weights from llm into wrapper.model. wrapper.lm_head will tie
    # automatically via cfg.tie_word_embeddings.
    missing, unexpected = wrapper.model.load_state_dict(llm.state_dict(), strict=True)
    if missing or unexpected:
        log.error("State dict mismatch: missing=%s unexpected=%s", missing, unexpected)
        return 1
    wrapper.tie_weights()
    wrapper.eval()

    # Sanity: confirm a forward works at a small seq_len.
    with torch.inference_mode():
        ids = torch.zeros(1, 8, dtype=torch.long)
        out_logits = wrapper(input_ids=ids).logits
        log.info("Wrapper forward ok, logits=%s", tuple(out_logits.shape))

    log.info("Saving HF checkpoint to %s", out)
    wrapper.save_pretrained(out, safe_serialization=True)

    # Copy a tokenizer in (the GGUF converter wants tokenizer.json or
    # tokenizer.model). OmniVoice's own tokenizer is the Qwen3 one — vocab
    # size 151676 vs upstream 151936; they differ in trailing reserved
    # specials, the BPE merges are the same, but to be safe we'll write a
    # placeholder generation_config and pull a tokenizer from upstream.
    log.info("Pulling tokenizer from %s", args.tokenizer)
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tok.save_pretrained(out)
    except Exception as e:  # noqa: BLE001
        log.warning("Tokenizer copy failed (%s); converter may complain", e)

    # Write a minimal generation_config.json so the converter is happy.
    gen_cfg = out / "generation_config.json"
    if not gen_cfg.exists():
        gen_cfg.write_text(json.dumps({
            "transformers_version": "5.6.2",
            "do_sample": False,
        }, indent=2))

    log.info("Done. Files in %s:", out)
    for f in sorted(out.iterdir()):
        log.info("  %s  (%.2f MB)", f.name, f.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
