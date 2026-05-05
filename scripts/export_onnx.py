#!/usr/bin/env python3
"""Export OmniVoice's diffusion-LM step to ONNX (fp32, external-data format).

Exports the matmul-heavy subgraph (LLM forward + audio_heads), keeping the
Higgs audio decoder in PyTorch. Loads with ``attn_implementation="sdpa"`` to
force off FlexAttention.

Output: ``onnx/omnivoice-step.onnx`` + ``onnx/omnivoice-step.onnx_data``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export-onnx")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "onnx" / "omnivoice-step.onnx"


class OmniVoiceStep(nn.Module):
    """Thin wrapper around the diffusion-LM step.

    Mirrors omnivoice/models/omnivoice.py:382-433 exactly, minus the loss
    branch and the OmniVoiceModelOutput container (which torch.onnx.export
    can't trace through cleanly).
    """

    def __init__(self, model: OmniVoice):
        super().__init__()
        # Re-use the inner model's submodules without doubling parameters.
        self.audio_embeddings = model.audio_embeddings
        self.llm = model.llm
        self.audio_heads = model.audio_heads
        self.register_buffer(
            "codebook_layer_offsets",
            model.codebook_layer_offsets.clone().detach(),
            persistent=False,
        )
        self.num_audio_codebook = model.config.num_audio_codebook
        self.audio_vocab_size = model.config.audio_vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # _prepare_embed_inputs (omnivoice.py:360-380), inlined.
        text_embeds = self.llm.get_input_embeddings()(input_ids[:, 0, :])
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1).to(input_ids.dtype)
        ) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
        inputs_embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

        hidden = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        ).last_hidden_state

        logits_flat = self.audio_heads(hidden)
        B = hidden.size(0)
        S = hidden.size(1)
        # [B, S, C*V] -> [B, S, C, V] -> [B, C, S, V]
        return logits_flat.view(
            B, S, self.num_audio_codebook, self.audio_vocab_size
        ).permute(0, 2, 1, 3)


def make_dummy_inputs(model: OmniVoice, B: int = 2, S: int = 512):
    """Build realistic dummy inputs at a typical seq_len.

    Matches the shapes constructed in _generate_iterative
    (omnivoice.py:1190-1217): input_ids [B, C, S] long, audio_mask [B, S] bool,
    attention_mask [B, 1, S, S] bool, position_ids [B, S] long.
    """
    cfg = model.config
    C = cfg.num_audio_codebook
    V = cfg.audio_vocab_size
    text_vocab = cfg.llm_config.vocab_size

    g = torch.Generator(device="cpu").manual_seed(0)

    # Layer 0 row: text token IDs (must be valid in text vocab).
    # Layers 1..C-1: audio token IDs (must be valid in audio vocab [0, V)).
    text_ids = torch.randint(0, min(1000, text_vocab), (B, 1, S), generator=g, dtype=torch.long)
    audio_ids = torch.randint(0, V, (B, C - 1, S), generator=g, dtype=torch.long)
    input_ids = torch.cat([text_ids, audio_ids], dim=1).contiguous()

    # audio_mask: roughly the same split the inference path uses (style+text
    # in front, audio at the back). Half-and-half is a reasonable proxy.
    audio_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_mask[:, S // 2 :] = True

    # 4D bool attention mask, all True (full attention within each batch).
    attention_mask = torch.ones(B, 1, S, S, dtype=torch.bool)

    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()

    return input_ids, audio_mask, attention_mask, position_ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                   help="attention implementation to trace through "
                   "(eager exposes the Q*K^T pattern to ORT's graph optimizer; "
                   "sdpa is faster to trace but hides attention from the fuser)")
    p.add_argument("--skip-check", action="store_true",
                   help="skip onnx.checker.check_model (slow on multi-GB graphs)")
    args = p.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s on cpu (attn_implementation=%s) ...", args.model, args.attn)
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation=args.attn)
    model.to("cpu").eval()
    log.info("Loaded in %.1fs", time.time() - t0)

    # Sanity check: confirm SDPA propagated to the LLM.
    llm_attn = getattr(model.llm.config, "_attn_implementation", "?")
    log.info("LLM attn_implementation=%s", llm_attn)

    wrapper = OmniVoiceStep(model).eval()

    log.info("Building dummy inputs (B=2, S=%d) ...", args.seq_len)
    input_ids, audio_mask, attention_mask, position_ids = make_dummy_inputs(
        model, B=2, S=args.seq_len
    )
    log.info(
        "shapes: input_ids=%s audio_mask=%s attention_mask=%s position_ids=%s",
        tuple(input_ids.shape), tuple(audio_mask.shape),
        tuple(attention_mask.shape), tuple(position_ids.shape),
    )

    # Forward sanity check before export.
    with torch.inference_mode():
        out = wrapper(input_ids, audio_mask, attention_mask, position_ids)
    log.info("forward ok, output shape=%s dtype=%s", tuple(out.shape), out.dtype)

    dynamic_axes = {
        "input_ids":      {0: "batch", 2: "seq_len"},
        "audio_mask":     {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 2: "seq_len", 3: "seq_len"},
        "position_ids":   {0: "batch", 1: "seq_len"},
        "logits":         {0: "batch", 2: "seq_len"},
    }

    log.info("Exporting to %s (opset=%d, external data) ...", out_path, args.opset)
    # torch.onnx.export auto-spills tensors >2 GB into per-tensor files in the
    # output directory. We export to a scratch directory first, then
    # re-serialize through onnx.save with all_tensors_to_one_file=True so the
    # final layout is graph + a single ``<stem>.onnx_data`` sidecar.
    import tempfile

    import onnx

    with tempfile.TemporaryDirectory(prefix="omnivoice-onnx-") as tmpdir:
        tmp_graph = Path(tmpdir) / "graph.onnx"
        t0 = time.time()
        torch.onnx.export(
            wrapper,
            (input_ids, audio_mask, attention_mask, position_ids),
            str(tmp_graph),
            input_names=["input_ids", "audio_mask", "attention_mask", "position_ids"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            export_params=True,
        )
        log.info("torch.onnx.export done in %.1fs", time.time() - t0)

        log.info("Consolidating external data into %s_data ...", out_path.name)
        model_proto = onnx.load(str(tmp_graph), load_external_data=True)

    # Wipe stale sharded files from previous runs in the destination dir.
    for sib in list(out_path.parent.iterdir()):
        if sib.name == out_path.name or sib.name == out_path.name + "_data":
            sib.unlink()
        elif sib.name.startswith("onnx__") or sib.name.endswith(".weight"):
            sib.unlink()

    onnx.save_model(
        model_proto,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out_path.name + "_data",
        size_threshold=1024,
        convert_attribute=False,
    )

    graph_size = out_path.stat().st_size
    data_path = Path(str(out_path) + "_data")
    data_size = data_path.stat().st_size if data_path.exists() else 0
    log.info(
        "On disk: graph=%.2f MB, data=%.2f MB (total %.2f MB)",
        graph_size / 1e6, data_size / 1e6, (graph_size + data_size) / 1e6,
    )

    if not args.skip_check:
        log.info("Running onnx.checker.check_model (full_check=True) ...")
        # full_check on a 2.4 GB external-data model is slow but cheap insurance.
        onnx.checker.check_model(str(out_path), full_check=True)
        log.info("checker ok")

    return 0


if __name__ == "__main__":
    sys.exit(main())
