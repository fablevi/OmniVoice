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

"""CPU-optimized inference CLI for OmniVoice using the OpenVINO runtime.

The matmul-heavy diffusion-LM step (LLM forward + audio_heads) runs through
OpenVINO; the rest of the pipeline (tokenization, mask construction, Higgs
audio decode) stays in PyTorch. With an int8 IR produced by
``omnivoice.scripts.quantize_openvino`` this typically halves CPU inference
latency relative to the default PyTorch path on AVX-VNNI Intel CPUs.

Usage:
    # Voice cloning
    omnivoice-infer-openvino --model k2-fsa/OmniVoice \
        --ir openvino_ir/omnivoice-step-int8.xml \
        --text "Hello, this is a text for text-to-speech." \
        --ref_audio ref.wav --ref_text "Reference transcript." --output out.wav

    # Voice design
    omnivoice-infer-openvino --model k2-fsa/OmniVoice \
        --ir openvino_ir/omnivoice-step-int8.xml \
        --text "Hello, this is a text for text-to-speech." \
        --instruct "male, British accent" --output out.wav
"""

from __future__ import annotations

import argparse
import logging
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceModelOutput
from omnivoice.utils.common import str2bool


def install_openvino_forward(
    model: OmniVoice,
    ir_path: Path,
    threads: int,
    static_reshape: bool,
) -> dict:
    """Replace ``model.forward`` with an OpenVINO-backed implementation.

    Returns a small ``stats`` dict that accumulates per-step latency so the
    caller can report a forward-only RTF alongside the wall-clock RTF.
    """
    import openvino as ov

    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": threads})
    logging.info(f"Reading OpenVINO IR: {ir_path}")
    base_model = core.read_model(str(ir_path))

    compile_kwargs = {
        "INFERENCE_NUM_THREADS": threads,
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "ENABLE_CPU_PINNING": "YES",
    }

    compiled = core.compile_model(base_model, "CPU", compile_kwargs)
    request = compiled.create_infer_request()
    out_port = compiled.outputs[0]
    in_names = {i.any_name for i in compiled.inputs}
    needs_pos_ids = "position_ids" in in_names
    state = {
        "request": request,
        "compiled": compiled,
        "out_port": out_port,
        "shape_seen": None,
        "n_calls": 0,
        "total_seconds": 0.0,
    }

    def forward(
        self_, input_ids, audio_mask, labels=None,
        attention_mask=None, document_ids=None, position_ids=None,
    ):
        if attention_mask is None:
            raise RuntimeError(
                "OpenVINO inference requires a 4D attention_mask"
            )
        B, _, S = input_ids.shape

        # On the first call, recompile with static shapes. ~10% per-step
        # speedup on this graph, paid once with a ~1-2 second recompile.
        if static_reshape and state["shape_seen"] is None:
            state["shape_seen"] = (B, S)
            t0 = time.perf_counter()
            new_model = core.read_model(str(ir_path))
            new_model.reshape({
                "input_ids":      [B, 8, S],
                "audio_mask":     [B, S],
                "attention_mask": [B, 1, S, S],
                "position_ids":   [B, S],
            })
            new_compiled = core.compile_model(new_model, "CPU", compile_kwargs)
            state["request"] = new_compiled.create_infer_request()
            state["compiled"] = new_compiled
            state["out_port"] = new_compiled.outputs[0]
            logging.info(
                f"Recompiled with static shape B={B} S={S} "
                f"in {time.perf_counter() - t0:.2f}s"
            )

        feeds = {
            "input_ids": input_ids.detach().cpu().contiguous().numpy(),
            "audio_mask": audio_mask.detach().cpu().contiguous().numpy(),
            "attention_mask": attention_mask.detach().cpu().contiguous().numpy(),
        }
        if needs_pos_ids:
            if position_ids is None:
                position_ids = (
                    torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
                )
            feeds["position_ids"] = (
                position_ids.detach().cpu().contiguous().numpy()
            )

        t0 = time.perf_counter()
        out = state["request"].infer(feeds)
        state["n_calls"] += 1
        state["total_seconds"] += time.perf_counter() - t0
        logits = torch.from_numpy(np.ascontiguousarray(out[state["out_port"]]))
        return OmniVoiceModelOutput(loss=None, logits=logits)

    model.forward = types.MethodType(forward, model)
    return state


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice single-item inference (OpenVINO backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--ir",
        type=str,
        required=True,
        help=(
            "Path to the OpenVINO IR (.xml) for the diffusion-LM step. "
            "Build one with omnivoice.scripts.quantize_openvino."
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV file path.",
    )
    # Voice cloning
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio file path for voice cloning.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text describing the reference audio.",
    )
    # Voice design
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design mode.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'English') or code (e.g. 'en').",
    )
    # Generation parameters (mirror omnivoice-infer)
    parser.add_argument("--num_step", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Fixed output duration in seconds.",
    )
    parser.add_argument("--t_shift", type=float, default=0.1)
    parser.add_argument("--denoise", type=str2bool, default=True)
    parser.add_argument("--postprocess_output", type=str2bool, default=True)
    parser.add_argument("--layer_penalty_factor", type=float, default=5.0)
    parser.add_argument("--position_temperature", type=float, default=5.0)
    parser.add_argument("--class_temperature", type=float, default=0.0)
    # OpenVINO-specific
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="CPU threads for OpenVINO inference.",
    )
    parser.add_argument(
        "--static_reshape",
        type=str2bool,
        default=True,
        help=(
            "Recompile with static shapes on the first forward (~10%% faster "
            "per-step at the cost of a one-time recompile). Disable when "
            "running ad-hoc requests of varying lengths in the same process."
        ),
    )
    return parser


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    ir_path = Path(args.ir).resolve()
    if not ir_path.exists():
        raise FileNotFoundError(f"OpenVINO IR not found: {ir_path}")

    torch.set_num_threads(args.threads)

    logging.info(f"Loading model from {args.model} on cpu ...")
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()

    stats = install_openvino_forward(
        model,
        ir_path=ir_path,
        threads=args.threads,
        static_reshape=args.static_reshape,
    )

    logging.info(f"Generating audio for: {args.text[:80]}...")
    t0 = time.perf_counter()
    audios = model.generate(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        instruct=args.instruct,
        duration=args.duration,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        speed=args.speed,
        t_shift=args.t_shift,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
    )
    wall = time.perf_counter() - t0

    sf.write(args.output, audios[0], model.sampling_rate)
    audio_seconds = len(audios[0]) / model.sampling_rate
    rtf = wall / audio_seconds if audio_seconds > 0 else float("inf")
    mean_step_ms = (
        1000.0 * stats["total_seconds"] / max(1, stats["n_calls"])
    )
    logging.info(
        f"Generated {audio_seconds:.2f}s in {wall:.2f}s wall (RTF={rtf:.3f}); "
        f"OpenVINO steps={stats['n_calls']}, mean={mean_step_ms:.1f} ms"
    )
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
