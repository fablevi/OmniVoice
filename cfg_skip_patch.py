"""Monkey-patch OmniVoice._generate_iterative to skip the uncond batch row when
guidance_scale == 0. Halves the LLM forward batch size at the cost of CFG.
"""
from __future__ import annotations

import math
import types
from typing import List

import torch

import omnivoice.models.omnivoice as omod


def _generate_iterative_cfg_skip(self, task, gen_config) -> List[torch.Tensor]:
    """Drop-in replacement that skips uncond rows when guidance_scale == 0."""

    if gen_config.guidance_scale != 0:
        return _orig_generate_iterative(self, task, gen_config)

    B = task.batch_size

    inputs_list = [
        self._prepare_inference_inputs(
            task.texts[i], task.target_lens[i], task.ref_texts[i],
            task.ref_audio_tokens[i], task.langs[i], task.instructs[i],
            gen_config.denoise,
        )
        for i in range(B)
    ]

    c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
    max_c_len = max(c_lens)
    pad_id = self.config.audio_mask_id

    batch_input_ids = torch.full(
        (B, self.config.num_audio_codebook, max_c_len),
        pad_id, dtype=torch.long, device=self.device,
    )
    batch_audio_mask = torch.zeros(
        (B, max_c_len), dtype=torch.bool, device=self.device
    )
    batch_attention_mask = torch.zeros(
        (B, 1, max_c_len, max_c_len), dtype=torch.bool, device=self.device
    )

    for i, inp in enumerate(inputs_list):
        c_len = c_lens[i]
        batch_input_ids[i, :, :c_len] = inp["input_ids"]
        batch_audio_mask[i, :c_len] = inp["audio_mask"]
        batch_attention_mask[i, :, :c_len, :c_len] = True

    tokens = torch.full(
        (B, self.config.num_audio_codebook, max(task.target_lens)),
        self.config.audio_mask_id,
        dtype=torch.long, device=self.device,
    )

    timesteps = omod._get_time_steps(
        t_start=0.0, t_end=1.0,
        num_step=gen_config.num_step, t_shift=gen_config.t_shift,
    ).tolist()
    schedules = []
    for t_len in task.target_lens:
        total_mask = t_len * self.config.num_audio_codebook
        rem = total_mask
        sched = []
        for step in range(gen_config.num_step):
            num = (rem if step == gen_config.num_step - 1 else
                   min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem))
            sched.append(int(num))
            rem -= int(num)
        schedules.append(sched)

    layer_ids = torch.arange(
        self.config.num_audio_codebook, device=self.device
    ).view(1, -1, 1)

    for step in range(gen_config.num_step):
        batch_logits = self(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)

        for i in range(B):
            k = schedules[i][step]
            if k <= 0:
                continue

            c_len, t_len = c_lens[i], task.target_lens[i]
            c_logits = batch_logits[i:i + 1, :, c_len - t_len:c_len, :]

            # No CFG: scoring uses cond logits directly. Pass dummy uncond.
            pred_tokens, scores = self._predict_tokens_with_scoring(
                c_logits, c_logits, gen_config
            )

            scores = scores - (layer_ids * gen_config.layer_penalty_factor)

            if gen_config.position_temperature > 0.0:
                scores = omod._gumbel_sample(scores, gen_config.position_temperature)

            sample_tokens = tokens[i:i + 1, :, :t_len]
            scores.masked_fill_(
                sample_tokens != self.config.audio_mask_id, -float("inf")
            )

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

            tokens[i:i + 1, :, :t_len] = sample_tokens
            batch_input_ids[i:i + 1, :, c_len - t_len:c_len] = sample_tokens

    return [tokens[i, :, :task.target_lens[i]] for i in range(B)]


_orig_generate_iterative = omod.OmniVoice._generate_iterative


def install(model: "omod.OmniVoice") -> None:
    model._generate_iterative = types.MethodType(
        _generate_iterative_cfg_skip, model
    )
