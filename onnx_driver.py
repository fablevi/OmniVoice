#!/usr/bin/env python3
"""ONNX-runtime driver for OmniVoice (voice-design mode).

Loads the full PyTorch ``OmniVoice`` model so all host-side helpers
(tokenizer, duration estimator, mask construction, sampling, Higgs decode,
post-processing) stay available, then monkey-patches ``model.forward`` so the
inner LLM + audio_heads call inside ``_generate_iterative`` is served by
``onnxruntime`` instead.

Everything else — the 16-step diffusion loop, CFG combination, Gumbel
sampling, the Higgs decoder — runs unchanged in PyTorch.

Usage:
    python onnx_driver.py "Hello world." [--out out.wav] [--int8] [--voice nova]
"""

from __future__ import annotations

import argparse
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

import onnxruntime as ort

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import OmniVoiceModelOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("onnx-driver")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FP32 = REPO_ROOT / "onnx" / "omnivoice-step.onnx"
DEFAULT_INT8 = REPO_ROOT / "onnx" / "omnivoice-step.int8.onnx"

# Mirrors server.py — 6 OpenAI-style voice presets translated into voice-design
# instruct strings.
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


def make_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    log.info(
        "ONNX session: %s (intra=%d threads, providers=%s)",
        onnx_path.name, threads, sess.get_providers(),
    )
    return sess


def install_onnx_forward(
    model: OmniVoice, sess: ort.InferenceSession, stats: StepStats,
    capture_dir: Optional[Path] = None, capture_tag: Optional[str] = None,
) -> None:
    """Replace ``model.forward`` with an ONNX-backed equivalent.

    Signature must match the call site in
    ``omnivoice/models/omnivoice.py:1255``: keyword-only ``input_ids``,
    ``audio_mask``, ``attention_mask`` (no ``position_ids`` — we synthesise
    it from the seq dim).

    If ``capture_dir`` is set, every ``sess.run`` input is dumped as
    ``<capture_dir>/<capture_tag>_<step>.npz``. Use this with the fp32
    session to collect calibration data for static QDQ quantisation.
    """

    expected_inputs = {i.name for i in sess.get_inputs()}
    needs_position_ids = "position_ids" in expected_inputs

    capture_state = {"step": 0}
    if capture_dir is not None:
        capture_dir.mkdir(parents=True, exist_ok=True)

    def forward(
        self: OmniVoice,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        labels=None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids=None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> OmniVoiceModelOutput:
        # Shapes: input_ids [B, C, S], audio_mask [B, S], attention_mask [B,1,S,S].
        B, _, S = input_ids.shape

        if attention_mask is None:
            raise RuntimeError(
                "ONNX driver requires an explicit 4D attention_mask (the "
                "training-time document_ids/flex_attention path is not exported)."
            )

        feeds = {
            "input_ids": input_ids.detach().cpu().contiguous().numpy(),
            "audio_mask": audio_mask.detach().cpu().contiguous().numpy(),
            "attention_mask": attention_mask.detach().cpu().contiguous().numpy(),
        }
        if needs_position_ids:
            if position_ids is None:
                pos = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)
            else:
                pos = position_ids
            feeds["position_ids"] = pos.detach().cpu().contiguous().numpy()

        if capture_dir is not None:
            tag = capture_tag or "sample"
            step = capture_state["step"]
            np.savez(capture_dir / f"{tag}_step{step:04d}.npz", **feeds)
            capture_state["step"] = step + 1

        t0 = time.perf_counter()
        out = sess.run(["logits"], feeds)[0]  # [B, C, S, V] fp32
        stats.add(time.perf_counter() - t0)

        # The downstream code does ``.logits.to(torch.float32)`` already, but
        # returning fp32 keeps that a no-op.
        logits = torch.from_numpy(np.ascontiguousarray(out))
        return OmniVoiceModelOutput(loss=None, logits=logits)

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
        language=None,  # auto
        instruct=instruct,
        speed=speed,
        generation_config=gen_config,
    )
    return np.asarray(audios[0], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", help="Text to synthesize")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--voice", default="nova", choices=sorted(VOICE_INSTRUCTS))
    p.add_argument("--instruct", default=None,
                   help="Override the voice preset's instruct string")
    p.add_argument("--int8", action="store_true",
                   help="Use the int8 ONNX file (default: fp32)")
    p.add_argument("--onnx", default=None,
                   help="Path to .onnx file (overrides --int8)")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--speed", type=float, default=None)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    args = p.parse_args()

    if args.onnx:
        onnx_path = Path(args.onnx).resolve()
    else:
        onnx_path = (DEFAULT_INT8 if args.int8 else DEFAULT_FP32).resolve()

    if not onnx_path.exists():
        log.error("ONNX file not found: %s — run scripts/export_onnx.py "
                  "(and scripts/quantize_onnx.py for int8) first", onnx_path)
        return 1

    torch.set_num_threads(args.threads)

    log.info("Loading PyTorch OmniVoice for host-side helpers ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    log.info("PyTorch model loaded in %.1fs", time.time() - t0)

    log.info("Opening ONNX session: %s", onnx_path)
    sess = make_session(onnx_path, threads=args.threads)
    stats = StepStats()
    install_onnx_forward(model, sess, stats)

    instruct = args.instruct or VOICE_INSTRUCTS[args.voice]
    log.info(
        "synthesize voice=%s chars=%d num_step=%d instruct=%r speed=%s",
        args.voice, len(args.text), args.num_step, instruct, args.speed,
    )

    t0 = time.perf_counter()
    waveform = synthesize(model, args.text, instruct, args.num_step, args.speed)
    wall = time.perf_counter() - t0

    sr = model.sampling_rate
    audio_seconds = waveform.shape[-1] / sr
    rtf = wall / audio_seconds if audio_seconds > 0 else float("inf")

    sf.write(args.out, waveform, sr, subtype="PCM_16")

    log.info(
        "generated %.2fs audio in %.2fs wall, RTF=%.3f | LLM steps=%d, mean=%.1f ms",
        audio_seconds, wall, rtf, stats.n_calls, stats.mean_ms,
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
