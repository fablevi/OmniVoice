# OmniVoice CPU performance: official PyTorch vs. omnivoice.cpp

CPU-only comparison of the official [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
PyTorch implementation against the standalone [bluryar/omnivoice.cpp](https://github.com/bluryar/omnivoice.cpp)
GGML runtime, on identical text, identical generation settings, and the same machine.

## Test environment

| | |
|---|---|
| CPU | Intel Core i7-14700 (20 physical cores, 28 threads, P-core max 5.4 GHz) |
| Memory | 30 GB |
| GPU | None (Intel UHD 770 integrated only — not used) |
| OS | Linux 6.8 |
| Threads pinned | 8 (matches OmniVoice README "as low as 0.025 RTF" claim's likely thread count) |
| Date | 2026-04-28 |

## Implementations under test

| | Official Python | omnivoice.cpp |
|---|---|---|
| Repo | `k2-fsa/OmniVoice` @ master | `bluryar/omnivoice.cpp` @ main |
| Runtime | PyTorch 2.8.0 (CPU wheels) | GGML 0.9.11 (CPU backend) |
| Math libs | Intel MKL / oneDNN (PyTorch default) | GGML CPU kernels (`-march=native`, OpenMP) |
| Weights | fp32 safetensors (no fp16 path on CPU; gated `cuda` only in `omnivoice/models/omnivoice.py:308`) | Q8_0 GGUF (`bluryar/omnivoice-gguf`) |
| Model files on disk | 2.45 GB main + 0.81 GB audio tokenizer = **3.25 GB** | Single `omnivoice-q8_0.gguf` = **1.47 GB** |
| Invocation | `server.py` (FastAPI), model preloaded once, requests over HTTP | `./build-cpu/omnivoice-cli`, fresh process per run |

## Workload

Single English prompt, 122 characters:

> The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump.

Generation settings (held constant):

- `num_step = 16` (the README's "faster inference" setting)
- `guidance_scale = 2.0`
- `denoise = true`
- `preprocess_prompt = true`, `postprocess_output = true`
- `speed = 1.0`, no fixed `duration`
- `threads = 8`
- Voice design `instruct = "Female, Middle-aged, Moderate Pitch"` (the `nova` preset)
- `response_format = wav`

Generated audio length differs slightly between implementations because each
chooses its own duration estimate from the text (Python: 7.48 s, C++: 7.34 s).
RTF normalises for this.

## Results — 3 runs each

### Official Python (fp32, PyTorch CPU)

End-to-end HTTP timings (server-side `generate()` time also recorded):

| Run | Wall (HTTP) | Server `generate()` | Audio | RTF (wall) | RTF (gen-only) |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.54 s | 9.5 s | 7.48 s | 1.275 | 1.270 |
| 2 | 9.28 s | 9.3 s | 7.48 s | 1.241 | 1.243 |
| 3 | 9.33 s | 9.3 s | 7.48 s | 1.247 | 1.243 |
| **avg** | **9.38 s** | **9.37 s** | **7.48 s** | **1.254** | **1.252** |

Model load (one-time, server startup): 1.1 s from local HF cache (cold first
download was ~285 s).

### omnivoice.cpp (Q8_0 GGUF, GGML CPU)

The CLI loads weights every run; `internal_total` is generation-only RTF
(excludes load + WAV write).

| Run | Wall | Audio | `internal_total` | `llm` | `decode` | RTF (wall) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.06 s | 7.34 s | 2.202 | 2.032 | 0.169 | 2.324 |
| 2 | 16.93 s | 7.34 s | 2.184 | 2.022 | 0.163 | 2.307 |
| 3 | 17.08 s | 7.34 s | 2.206 | 2.042 | 0.164 | 2.327 |
| **avg** | **17.02 s** | **7.34 s** | **2.197** | **2.032** | **0.165** | **2.319** |

Per-stage breakdown (from CLI `[rtf]` output):
- LLM diffusion decode: ~92% of generation time
- Higgs audio decode: ~8%
- Reference encode: n/a (voice design, no reference audio)
- Model load: ~0.84 s (excluded from `internal_total`)

## Comparison

| Metric | Python fp32 | C++ Q8_0 | Ratio (C++ ÷ Python) |
|---|---:|---:|---:|
| RTF (gen-only) | 1.252 | 2.197 | **1.76× slower** |
| Wall time for 7.4 s of audio | 9.4 s | 17.0 s | **1.81× slower** |
| Disk footprint | 3.25 GB | 1.47 GB | 0.45× (smaller) |
| README's headline RTF | 0.025 (CUDA, fp16) | 0.194 total / 0.154 LLM (RTX 4060 Ti, CUDA, Q8_0) | — |

**On this CPU, the official PyTorch implementation is ~76% faster than
omnivoice.cpp's Q8_0 build for the same workload.** Both are well off the
README's 0.025 RTF target, which is a CUDA fp16 number.

### Why is Python fp32 beating C++ Q8_0 on CPU?

Counter-intuitive but consistent with what's running underneath:

1. **PyTorch CPU links Intel MKL/oneDNN** by default on x86. Those kernels are
   heavily tuned for AVX2/AVX-512 GEMM at fp32 and dominate matmul throughput
   on Raptor/Alder Lake. The i7-14700 exposes `avx2`, `avx_vnni`, `bmi2`,
   `f16c`, `fma`, `vaes`, `vpclmulqdq` (no AVX-512 user-visible — Intel
   disabled it on hybrid client parts), so the relevant fast path here is
   AVX2 + VNNI.
2. **GGML CPU kernels are portable, not MKL-equivalent.** They get good
   performance for llama-style decode but for OmniVoice's diffusion LLM the
   matmul shapes and per-step compute pattern aren't as well covered as
   MKL's. Built with `-march=native` + OpenMP here.
3. **Q8_0 ≠ free speedup on CPU.** Q8_0 reduces memory but each matmul
   dequantizes to fp32 before MAC. On a CPU where weights already fit in RAM
   and bandwidth isn't saturated, fp32 GEMM on MKL beats dequant→fp32 on
   GGML's kernels.
4. **fp16 path is CUDA-only** in OmniVoice (`models/omnivoice.py:308`:
   `dtype = torch.float16 if "cuda" else torch.float32`). So the Python build
   on CPU runs full fp32 — there's no fp16 left on the table for it to lose.

### Why is the C++ runtime worth having anyway?

- Memory: 1.47 GB on disk vs 3.25 GB. Useful if storage/RAM is constrained.
- No PyTorch / Python dependency. ~50 KB binary + shared lib.
- The author reports **0.194 total RTF on RTX 4060 Ti with Q8_0** — once you
  have a CUDA GPU, the C++ runtime is competitive with the Python fp16 CUDA
  path while shipping a much smaller deliverable.
- For embedded / edge deployments, the smaller binary + smaller weights is
  the point — not raw CPU throughput.

## How to reproduce

### Python server (this repo)

```bash
# in this repo:
git checkout openai-tts-server
uv venv .venv && source .venv/bin/activate
PIP_INDEX_URL=https://download.pytorch.org/whl/cpu uv pip install torch torchaudio
uv pip install -e . fastapi 'uvicorn[standard]'
python server.py --threads 8   # listens on :8311

# benchmark:
TEXT="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump."
for i in 1 2 3; do
  start=$(date +%s.%N)
  curl -s -X POST http://127.0.0.1:8311/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"omnivoice\",\"input\":\"$TEXT\",\"voice\":\"nova\",\"response_format\":\"wav\"}" \
    -o /tmp/py_$i.wav
  echo "run $i wall=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N)-$start}")s"
done
# audio durations and gen times also visible in server log lines `generated X.XXs of audio in Y.Ys`
```

### omnivoice.cpp

```bash
git clone https://github.com/bluryar/omnivoice.cpp.git ~/code/omnivoice.cpp
cd ~/code/omnivoice.cpp
git submodule update --init --recursive
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j8
hf download bluryar/omnivoice-gguf omnivoice-q8_0.gguf --local-dir models

TEXT="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump."
for i in 1 2 3; do
  ./build-cpu/omnivoice-cli \
    --model models/omnivoice-q8_0.gguf \
    --text "$TEXT" \
    --output /tmp/cpp_$i.wav \
    --language English \
    --instruct "Female, Middle-aged, Moderate Pitch" \
    --device cpu --threads 8 --num-step 16 --seed 123 2>&1 | grep '\[rtf\]'
done
```

## Other implementations / optimization efforts in the wild

Survey of OmniVoice ports and optimized runtimes as of 2026-04-28. Most are
GPU-focused; nothing in this list has demonstrated a CPU RTF below the
official PyTorch baseline.

### ONNX

**[`gluschenko/omnivoice-onnx`](https://huggingface.co/gluschenko/omnivoice-onnx)** (HuggingFace, community)

The only ONNX export found. Multiple precision variants of the diffusion LLM
(Qwen3-0.6B backbone) are uploaded; the audio decoder/tokenizer (Higgs RVQ +
DAC) is **not** in the repo, so this is a partial export — full TTS still
needs the PyTorch audio path.

| Variant | `.onnx_data` size |
|---|---:|
| fp32 (`omnivoice.onnx`) | 2.45 GB |
| qint16 per-channel | 1.06 GB |
| qint8 | 612 MB |
| qint8 per-channel | 612 MB |
| quint8 | 612 MB |
| quint8 per-channel | 612 MB |
| static QDQ u8s8 | 612 MB |

Repo README is a verbatim copy of the official model card with **no
ONNX-specific install steps, no example inference code, and no benchmarks**.
The repo has not been integrated into a working pipeline by any other project
I could find — no Sherpa-ONNX glue, no `onnxruntime` example, no published
RTF numbers. To actually use it, you'd have to write the decode loop, RVQ
post-processing, and tokenizer plumbing yourself.

Untested here. CPU performance via `onnxruntime` with the `qint8` variant
**might** beat or roughly match the PyTorch fp32 baseline thanks to int8
matmul kernels on x86 — this is the most plausible CPU-optimization candidate
in the ecosystem right now, but it's not validated.

### GGML / C++

**[`bluryar/omnivoice.cpp`](https://github.com/bluryar/omnivoice.cpp)** —
benchmarked above. CPU is ~76% slower than official PyTorch on this box.
Author's own headline number is CUDA-only (RTX 4060 Ti, RTF 0.194 total /
0.154 LLM).

### vLLM

**[`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni)** —
official-track vLLM extension. Two-stage GPU pipeline (Qwen3 LLM → Higgs
decoder), served with `vllm serve k2-fsa/OmniVoice --omni`. GPU-only in
practice; no published RTF or throughput numbers. Documented at
[docs.vllm.ai/projects/vllm-omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/omnivoice/).

### GPU attention rewrites

**[`Saganaki22/ComfyUI-OmniVoice-TTS`](https://github.com/Saganaki22/ComfyUI-OmniVoice-TTS)**
— ComfyUI nodes with a Qwen3Attention monkey-patch that targets CUDA SM80+
(Ampere and newer). Adds VRAM management with auto CPU offload. Not a
CPU-side optimization.

### Wrappers / GUIs (no runtime change)

**[`debpalash/OmniVoice-Studio`](https://github.com/debpalash/OmniVoice-Studio)**
— dubbing/cloning GUI built on the official PyTorch runtime. Auto-detects
MPS/CUDA/ROCm/CPU device but doesn't change the math.

### Apple Silicon (MLX)

[`Blaizzy/mlx-audio` issue #637](https://github.com/Blaizzy/mlx-audio/issues/637)
— feature request opened 2026-04-06 to add OmniVoice to mlx-audio. **Closed
without implementation**, no PR, no prototype. No MLX port exists yet.

### Not found

Despite searching, no public ports yet to:

- **OpenVINO** (which would be the natural fit for this Intel CPU — Intel
  GPU/NPU + AVX-512 paths)
- **CoreML** (Apple Neural Engine)
- **TensorRT-LLM** (NVIDIA-optimized inference)
- **Sherpa-ONNX** (the k2-fsa org's own ONNX serving framework — surprising
  given the shared maintainership)

### Official position

In the [HuggingFace discussion thread](https://huggingface.co/k2-fsa/OmniVoice/discussions/2)
on this exact topic (closed ~2026-04-05), maintainer `zhu-han` wrote:

> The current PyTorch version is rather slow on CPU. I'm not sure whether an
> ONNX or GGML implementation can resolve this issue, but we will look into
> it. Contributions for a CPU-optimized, faster version are very welcome.

No official ONNX/GGML release planned at the time of writing. Community work
is fragmentary and GPU-biased.

### Practical takeaway

If you need CPU-only OmniVoice today and aren't going to port it yourself,
the **official PyTorch build with `num_step=16` and 8 threads** is the
fastest off-the-shelf option on x86. Q8_0 GGUF via `omnivoice.cpp` trades
~50% throughput for a ~55% smaller disk footprint, which only makes sense in
storage-constrained deployments. The most promising untapped option is the
existing `gluschenko/omnivoice-onnx` qint8 export wired into `onnxruntime`
with int8 GEMM kernels — but someone has to write the inference scaffolding
first.

## ONNX MVP — int8 ONNX Runtime on the same CPU

Following the survey above, we exported the matmul-heavy diffusion-LM step
(LLM forward + audio_heads) to ONNX and measured `onnxruntime`'s CPU int8
GEMM path against the official PyTorch baseline. Voice-design mode only;
the Higgs audio decoder stays in PyTorch.

### Method

- **Branch / driver**: `onnx-export-mvp` in this repo. Scripts:
  `scripts/export_onnx.py`, `scripts/quantize_onnx.py`,
  `scripts/bench_onnx.py`. Inference shim: `onnx_driver.py` (loads the full
  PyTorch model for tokenisers / mask construction / Higgs decode, then
  monkey-patches `model.forward` so the inner LLM call is served by
  `ort.InferenceSession`).
- **Export**: `OmniVoice.from_pretrained(..., attn_implementation="sdpa")`
  → wrap LLM + audio_heads in a thin `nn.Module` → `torch.onnx.export`
  (legacy TorchScript path, opset 17, dynamic batch + seq_len), then
  re-serialise through `onnx.save_model(all_tensors_to_one_file=True)` for
  one consolidated `.onnx_data` sidecar.
- **Quantisation**: `onnxruntime.quantization.quantize_dynamic`,
  `weight_type=QInt8`, `op_types_to_quantize=["MatMul","Gemm"]`.
  Embeddings and LayerNorm stay fp32 by design — int8 GEMM is the
  bandwidth-and-FLOPs win we're after.
- **Workload**: identical fox pangram, num_step=16, 8 threads,
  voice = nova preset, voice-design mode. 3 runs each.
- **Tooling**: `onnx 1.21.0`, `onnxruntime 1.25.1` (CPU package, MLAS
  backend; `ort.get_available_providers() = ['AzureExecutionProvider',
  'CPUExecutionProvider']`).

### File sizes

| File | Size |
|---|---:|
| `onnx/omnivoice-step.onnx` (fp32 graph) | 1.31 MB |
| `onnx/omnivoice-step.onnx_data` (fp32 weights) | 2.45 GB |
| `onnx/omnivoice-step.int8.onnx` (int8 graph) | 2.02 MB |
| `onnx/omnivoice-step.int8.onnx.data` (int8 weights) | 1.10 GB |

The fp32 sidecar matches gluschenko/omnivoice-onnx's 2.45 GB exactly.
Our int8 build is 1.10 GB rather than gluschenko's 612 MB because we
deliberately keep the 621 MB Qwen3 `embed_tokens` and the 33 MB
`audio_embeddings` fp32 (we only quantise `MatMul`/`Gemm`). Embedding
lookup is bandwidth-bound and already cheap; the int8 win is in the
GEMMs, not the gather.

### Quantisation skipped ops

`quantize_dynamic` correctly skips the Q·Kᵀ and (attn·V) MatMuls inside
each Qwen3 attention layer — they have non-constant B and there's no
weight to pre-quantise (purely activation × activation). Reported as
`Ignore MatMul due to non constant B: …self_attn/MatMul[_1]` for all 28
layers. This is expected and not a problem: those ops are masked-fill +
softmax-bound, not GEMM-bound.

All weight-bearing matmuls (q/k/v/o projections, gate/up/down MLP
projections, audio_heads, plus the LM's tied output projection if
exposed) are quantised.

### Results — 3 runs each, fox pangram (122 chars, 7.48 s of audio)

| Variant | Run | Wall | Audio | RTF | Mean LLM step |
|---|---:|---:|---:|---:|---:|
| **PyTorch fp32** (baseline rerun) | 1 | 10.18 s | 7.48 s | 1.360 | — |
| | 2 | 9.23 s | 7.48 s | 1.233 | — |
| | 3 | 9.19 s | 7.48 s | 1.228 | — |
| | **avg** | **9.53 s** | **7.48 s** | **1.274** | — |
| **ONNX fp32** | 1 | 9.39 s | 7.48 s | 1.256 | 542.2 ms |
| | 2 | 9.39 s | 7.48 s | 1.255 | 537.7 ms |
| | 3 | 9.28 s | 7.48 s | 1.240 | 533.2 ms |
| | **avg** | **9.35 s** | **7.48 s** | **1.250** | **537.7 ms** |
| **ONNX int8** | 1 | 4.41 s | 7.48 s | 0.590 | 228.5 ms |
| | 2 | 4.37 s | 7.48 s | 0.584 | 225.3 ms |
| | 3 | 4.52 s | 7.48 s | 0.604 | 233.3 ms |
| | **avg** | **4.43 s** | **7.48 s** | **0.593** | **229.0 ms** |

Mean LLM step is the time per `sess.run` call inside the diffusion loop
(16 steps total per generation; instrumented in `onnx_driver.py`).
PyTorch step time isn't broken out — `model.forward` is opaque to the
driver in that path.

### Headline comparison

| Runtime | RTF (gen-only, 8 threads) | vs. PyTorch fp32 |
|---|---:|---:|
| Official Python fp32 (PyTorch + MKL) | 1.252 | 1.00× |
| omnivoice.cpp Q8_0 (GGML CPU) | 2.197 | 0.57× (slower) |
| **ONNX fp32** (onnxruntime + MLAS) | **1.250** | **1.00× (parity)** |
| **ONNX int8** (onnxruntime + MLAS, AVX-VNNI) | **0.593** | **2.11× faster** |

ONNX fp32 lands within noise of PyTorch fp32 — same fp32 GEMMs running
through MLAS instead of MKL with no architectural shortcut. ONNX int8
crosses real-time (RTF < 1.0) for the first time on this CPU, by a
comfortable margin.

### Audio parity (subjective)

Spot-checked three prompts via `onnx_driver.py --int8`:

- short English ("Hello world.") — 1.64 s output, intelligible.
- long English (fox pangram) — 7.48 s output, intelligible.
- Chinese ("你好世界，今天天气真好。") — 2.36 s output, intelligible
  (auto language detection still works through the host-side
  `_prepare_inference_inputs`).

All three were normalised to peak 0.5 (the post-processing default for
non-reference output) with healthy RMS in 0.07–0.13 — no obvious
quantisation artefacts. Not bit-identical to the PyTorch path
(int8 quant + Gumbel RNG draw differ), but qualitatively similar.

### Verdict

**Hypothesis confirmed.** int8 ONNX on the i7-14700's AVX-VNNI fast path
gives a 2.1× CPU speedup over the official PyTorch fp32 baseline, taking
OmniVoice from RTF 1.27 to RTF 0.59 — the first sub-1.0 RTF result for
OmniVoice on a CPU we've measured. The ONNX fp32 export is parity with
PyTorch (good sanity check that our graph isn't accidentally degenerate);
all the gain is in `MatMul`/`Gemm` weights → int8.

This is an MVP — Python driver, single batch, no streaming, no caching.
The Higgs decoder stays in PyTorch. The ~1.1 GB int8 weight file plus
the still-fp32 PyTorch model is wasteful in RAM (we duplicate the Higgs
tokenizer + Qwen3 weights), but that's a fixable engineering issue, not
a hypothesis-blocker.

### Forward look — sherpa-onnx C++ port

The 2.1× CPU win justifies the next-session investment in a sherpa-onnx
C++ port. Specifically: the same int8 graph fed by a sherpa-onnx-style
runtime would:

1. Drop the PyTorch dependency (currently mandatory for tokenisers, mask
   construction, Higgs decode) — sherpa-onnx already supports BPE
   tokenisers and could host the Higgs decoder as a second ONNX graph.
2. Eliminate the Python ↔ NumPy ↔ ONNX tensor copy hot path on every
   diffusion step (currently ~16 copies per generation across 8 threads).
3. Enable streaming + persistent sessions for server deployments
   (current driver re-encodes a copy per `sess.run`).
4. Stay within a small static binary (sherpa-onnx + ORT shared lib),
   useful for embedded / edge deployments where the omnivoice.cpp path
   was the previous best option.

Earlier research-agent notes on sherpa-onnx integration points (BPE
tokeniser plumbing, the absent Higgs decoder block, the diffusion-loop
driver shape) are preserved in the parent session's subagent transcript
at `~/.claude/projects/-home-scott-code-OmniVoice/.../subagents/agent-a9461dc1015ced385.jsonl`.

## Caveats

- **Not parity-tested for output quality.** RTF only. The C++ tokenizer/seed
  paths are still alpha (per its README) and the generated waveforms aren't
  byte-identical to the Python output.
- **Single short prompt.** Long-form text and Chinese-language workloads were
  not measured here; both runtimes chunk long input differently.
- **No GPU available.** The dramatic RTF claim (0.025) and the C++ author's
  own numbers (0.194 total) are CUDA-only and not reproducible on this box.
- **Server-side overhead included for Python**, model-load excluded for C++ —
  but the gap is wider than that overhead. A pure `model.generate()` call in
  Python would still be ~1.25 RTF.
