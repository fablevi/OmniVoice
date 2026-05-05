#!/usr/bin/env python3
# Copyright    2026  Scott Yeager
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export OmniVoice's diffusion-LM step to ONNX (fp32, external-data format).

Exports the matmul-heavy subgraph (LLM forward + audio_heads). The Higgs audio
decoder remains in PyTorch and is invoked by the regular inference path.

The exported graph has dynamic batch and sequence axes, so the same file
serves both unbatched (B=1) and CFG (B=2) inference. Loads with
``attn_implementation="sdpa"`` by default; pass ``--attn eager`` if a downstream
toolchain prefers seeing the explicit Q*K^T pattern.

Usage:
    python -m omnivoice.scripts.export_onnx \
        --model k2-fsa/OmniVoice \
        --output onnx/omnivoice-step.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from omnivoice import OmniVoice


class OmniVoiceStep(nn.Module):
    """Thin wrapper around the diffusion-LM step.

    Mirrors the matmul-heavy core of ``OmniVoice.forward`` (LLM + audio heads),
    minus the loss branch and the dataclass-typed return value (which
    ``torch.onnx.export`` cannot trace cleanly).
    """

    def __init__(self, model: OmniVoice):
        super().__init__()
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
        text_embeds = self.llm.get_input_embeddings()(input_ids[:, 0, :])
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1).to(input_ids.dtype)
        ) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
        inputs_embeds = torch.where(
            audio_mask.unsqueeze(-1), audio_embeds, text_embeds
        )

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


def make_dummy_inputs(model: OmniVoice, batch_size: int, seq_len: int):
    """Build representative dummy inputs at a typical sequence length.

    Shapes match those constructed inside ``_generate_iterative``:
    ``input_ids`` [B, C, S] long, ``audio_mask`` [B, S] bool,
    ``attention_mask`` [B, 1, S, S] bool, ``position_ids`` [B, S] long.
    """
    cfg = model.config
    C = cfg.num_audio_codebook
    V = cfg.audio_vocab_size
    text_vocab = cfg.llm_config.vocab_size

    g = torch.Generator(device="cpu").manual_seed(0)

    text_ids = torch.randint(
        0, min(1000, text_vocab), (batch_size, 1, seq_len),
        generator=g, dtype=torch.long,
    )
    audio_ids = torch.randint(
        0, V, (batch_size, C - 1, seq_len), generator=g, dtype=torch.long,
    )
    input_ids = torch.cat([text_ids, audio_ids], dim=1).contiguous()

    # Roughly mirror inference layout: style+text tokens up front, audio
    # tokens at the back. Half-and-half is a reasonable proxy for tracing.
    audio_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    audio_mask[:, seq_len // 2:] = True

    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    position_ids = (
        torch.arange(seq_len, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    return input_ids, audio_mask, attention_mask, position_ids


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the OmniVoice diffusion-LM step to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="onnx/omnivoice-step.onnx",
        help="Output ONNX path. A sibling '<name>_data' file holds tensors >1KB.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Dummy sequence length used during tracing. Dynamic at runtime.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Dummy batch size used during tracing. Dynamic at runtime.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help=(
            "Attention implementation to trace through. 'eager' exposes the "
            "explicit Q*K^T pattern to downstream graph optimizers; 'sdpa' is "
            "faster to trace but hides attention from the fuser."
        ),
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Skip onnx.checker.check_model (slow on multi-GB graphs).",
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading {args.model} on cpu (attn_implementation={args.attn}) ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation=args.attn)
    model.to("cpu").eval()
    logging.info(f"Loaded in {time.time() - t0:.1f}s")

    llm_attn = getattr(model.llm.config, "_attn_implementation", "?")
    logging.info(f"LLM attn_implementation={llm_attn}")

    wrapper = OmniVoiceStep(model).eval()

    logging.info(
        f"Building dummy inputs (B={args.batch_size}, S={args.seq_len}) ..."
    )
    input_ids, audio_mask, attention_mask, position_ids = make_dummy_inputs(
        model, batch_size=args.batch_size, seq_len=args.seq_len,
    )

    with torch.inference_mode():
        out = wrapper(input_ids, audio_mask, attention_mask, position_ids)
    logging.info(f"forward ok, output shape={tuple(out.shape)} dtype={out.dtype}")

    dynamic_axes = {
        "input_ids":      {0: "batch", 2: "seq_len"},
        "audio_mask":     {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 2: "seq_len", 3: "seq_len"},
        "position_ids":   {0: "batch", 1: "seq_len"},
        "logits":         {0: "batch", 2: "seq_len"},
    }

    logging.info(f"Exporting to {out_path} (opset={args.opset}, external data) ...")

    # torch.onnx.export auto-spills tensors >2 GB into per-tensor files in the
    # destination directory. Export to a scratch dir, then re-serialize through
    # onnx.save with all_tensors_to_one_file=True so the final layout is
    # graph + a single '<stem>.onnx_data' sidecar.
    import onnx

    with tempfile.TemporaryDirectory(prefix="omnivoice-onnx-") as tmpdir:
        tmp_graph = Path(tmpdir) / "graph.onnx"
        t0 = time.time()
        torch.onnx.export(
            wrapper,
            (input_ids, audio_mask, attention_mask, position_ids),
            str(tmp_graph),
            input_names=[
                "input_ids", "audio_mask", "attention_mask", "position_ids",
            ],
            output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            export_params=True,
        )
        logging.info(f"torch.onnx.export done in {time.time() - t0:.1f}s")

        logging.info(f"Consolidating external data into {out_path.name}_data ...")
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
    logging.info(
        f"On disk: graph={graph_size / 1e6:.2f} MB, "
        f"data={data_size / 1e6:.2f} MB "
        f"(total {(graph_size + data_size) / 1e6:.2f} MB)"
    )

    if not args.skip_check:
        logging.info("Running onnx.checker.check_model (full_check=True) ...")
        onnx.checker.check_model(str(out_path), full_check=True)
        logging.info("checker ok")


if __name__ == "__main__":
    sys.exit(main())
