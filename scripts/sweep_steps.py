#!/usr/bin/env python3
"""Sweep num_step on the fp32 PyTorch model.

Runs the same prompt (fox pangram, voice-design nova preset, fixed seed)
at each num_step in --steps. The first value is treated as the
"full-step reference"; subsequent values report:

  - argmax_disagree_pct: fraction of audio-codebook tokens that differ
    from the reference (averaged over C codebooks x T frames).
  - RTF: wall_seconds / audio_seconds.

Caveat: the diffusion loop draws position/class temperature noise via
torch.rand, and the *number and order* of those draws depends on
num_step. So even with the same seed, low num_step values experience
both real quality loss AND a baseline RNG-drift floor. To anchor the
floor, we additionally run the reference num_step with a second seed
and report that as the "rng_floor" row.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sweep-steps")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)
INSTRUCT = "Female, Middle-aged, Moderate Pitch"  # nova preset


def lock_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def install_token_capture(model: OmniVoice, sink: dict) -> None:
    """Monkey-patch _generate_iterative to stash a deep copy of the
    returned token tensors into ``sink['tokens']`` after each call."""
    orig = model._generate_iterative

    def wrapped(task, gen_config):
        result = orig(task, gen_config)
        sink["tokens"] = [t.detach().clone().cpu() for t in result]
        return result

    model._generate_iterative = wrapped  # type: ignore[assignment]


def run(
    model: OmniVoice, num_step: int, seed: int, sink: dict,
    deterministic: bool = False,
) -> tuple[np.ndarray, torch.Tensor, float, int]:
    lock_seed(seed)
    cfg = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
        # When deterministic=True we pin both temperatures to 0 so the
        # diffusion loop has no Gumbel noise — token output then depends
        # only on (model, num_step) and argmax-disagree is a clean
        # step-count quality signal instead of being saturated by RNG.
        position_temperature=0.0 if deterministic else 5.0,
        class_temperature=0.0,
    )
    sink.pop("tokens", None)
    t0 = time.perf_counter()
    audios = model.generate(
        text=PROMPT, language=None, instruct=INSTRUCT, generation_config=cfg,
    )
    wall = time.perf_counter() - t0
    waveform = np.asarray(audios[0], dtype=np.float32)
    tokens = sink["tokens"][0]  # (C, T)
    return waveform, tokens, wall, model.sampling_rate


def disagree_pct(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int, int, int]:
    """Token-level argmax disagreement.

    If T differs (model picked different chunk lengths), compares over
    the overlapping prefix and reports both the trim and the original
    sizes so we can spot when shapes drift.
    """
    Ca, Ta = a.shape
    Cb, Tb = b.shape
    assert Ca == Cb, f"codebook count mismatch: {Ca} vs {Cb}"
    T = min(Ta, Tb)
    a_t = a[:, :T]
    b_t = b[:, :T]
    diff = (a_t != b_t).float().mean().item()
    return diff, T, Ta, Tb


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument(
        "--steps", type=int, nargs="+", default=[32, 16, 12, 8, 4],
        help="First value is the reference; rest are compared against it.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rng-floor-seed", type=int, default=43,
        help="Extra run at the reference step count with this seed, to "
             "measure how much disagreement is pure RNG drift.",
    )
    p.add_argument(
        "--threads", type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "8")),
    )
    p.add_argument("--out-dir", default="/tmp/omnivoice_step_sweep")
    p.add_argument(
        "--deterministic", action="store_true",
        help="Pin position_temperature=0, class_temperature=0 so the "
             "diffusion loop is deterministic. Removes the RNG-drift "
             "saturation that otherwise dominates argmax-disagree.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    log.info("torch threads = %d", torch.get_num_threads())

    log.info("Loading PyTorch model ...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, attn_implementation="sdpa")
    model.to("cpu").eval()
    log.info("loaded in %.1fs", time.time() - t0)

    sink: dict = {}
    install_token_capture(model, sink)

    ref_step = args.steps[0]
    log.info("=== reference: num_step=%d, seed=%d ===", ref_step, args.seed)
    ref_wav, ref_tokens, ref_wall, sr = run(model, ref_step, args.seed, sink, args.deterministic)
    ref_audio_s = ref_wav.shape[-1] / sr
    ref_rtf = ref_wall / ref_audio_s
    sf.write(str(out_dir / f"step{ref_step}_seed{args.seed}.wav"), ref_wav, sr, subtype="PCM_16")
    log.info(
        "  tokens shape=%s, wall=%.2fs, audio=%.2fs, RTF=%.3f",
        tuple(ref_tokens.shape), ref_wall, ref_audio_s, ref_rtf,
    )

    rows: list[dict] = [{
        "label": f"step{ref_step} (ref, seed={args.seed})",
        "num_step": ref_step,
        "wall": ref_wall, "audio_s": ref_audio_s, "rtf": ref_rtf,
        "disagree": 0.0, "T_ref": ref_tokens.shape[1], "T_var": ref_tokens.shape[1],
    }]

    # RNG floor measurement.
    log.info("=== rng_floor: num_step=%d, seed=%d ===", ref_step, args.rng_floor_seed)
    floor_wav, floor_tokens, floor_wall, _ = run(model, ref_step, args.rng_floor_seed, sink, args.deterministic)
    floor_audio_s = floor_wav.shape[-1] / sr
    floor_diff, T_used, T_ref, T_var = disagree_pct(ref_tokens, floor_tokens)
    sf.write(str(out_dir / f"step{ref_step}_seed{args.rng_floor_seed}.wav"), floor_wav, sr, subtype="PCM_16")
    log.info(
        "  disagree=%.4f%% over T=%d (ref T=%d, this T=%d), wall=%.2fs RTF=%.3f",
        100 * floor_diff, T_used, T_ref, T_var, floor_wall, floor_wall / floor_audio_s,
    )
    rows.append({
        "label": f"step{ref_step} (rng_floor, seed={args.rng_floor_seed})",
        "num_step": ref_step,
        "wall": floor_wall, "audio_s": floor_audio_s, "rtf": floor_wall / floor_audio_s,
        "disagree": floor_diff, "T_ref": T_ref, "T_var": T_var,
    })

    for step in args.steps[1:]:
        log.info("=== num_step=%d, seed=%d ===", step, args.seed)
        wav, tokens, wall, _ = run(model, step, args.seed, sink, args.deterministic)
        audio_s = wav.shape[-1] / sr
        rtf = wall / audio_s
        diff, T_used, T_ref, T_var = disagree_pct(ref_tokens, tokens)
        sf.write(str(out_dir / f"step{step}_seed{args.seed}.wav"), wav, sr, subtype="PCM_16")
        log.info(
            "  disagree=%.4f%% over T=%d (ref T=%d, this T=%d), wall=%.2fs RTF=%.3f",
            100 * diff, T_used, T_ref, T_var, wall, rtf,
        )
        rows.append({
            "label": f"step{step}",
            "num_step": step,
            "wall": wall, "audio_s": audio_s, "rtf": rtf,
            "disagree": diff, "T_ref": T_ref, "T_var": T_var,
        })

    print("\n=== num_step sweep (fp32, fox pangram, instruct=nova) ===")
    header = (
        f"{'variant':<34} {'wall':>7} {'audio':>7} {'RTF':>7} "
        f"{'disagree%':>10} {'T_ref':>6} {'T_var':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<34} {r['wall']:>6.2f}s {r['audio_s']:>6.2f}s "
            f"{r['rtf']:>7.3f} {100*r['disagree']:>9.3f}% "
            f"{r['T_ref']:>6d} {r['T_var']:>6d}"
        )
    print(f"\nWAVs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
