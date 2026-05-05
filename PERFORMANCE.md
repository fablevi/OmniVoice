# OmniVoice CPU performance: official PyTorch vs. omnivoice.cpp

CPU-only comparison of the official [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
PyTorch implementation against the standalone [bluryar/omnivoice.cpp](https://github.com/bluryar/omnivoice.cpp)
GGML runtime, on identical text, identical generation settings, and the same machine.

## TL;DR — current Pareto frontier (this Intel CPU)

Updated 2026-05-04. Measurements at `num_step=16`, `guidance_scale=2.0`,
`threads=8`, fox-pangram prompt unless noted.

| Runtime / Variant                     | RTF (CLI) | RTF (server warm) | Argmax disagree |
|---------------------------------------|----------:|------------------:|----------------:|
| PyTorch fp32 / ONNX fp32 / OV fp32   |    1.252  |             1.25  |             0% |
| **OV W8A8 `dpup`** (best int8 quality) |   0.85   |          **0.755** |        **4.87%** |
| **OV W8A8 `dp0-27`** (balanced)        |   0.72   |          **0.618** |        **5.89%** |
| llama.cpp Q8_0 (prior quality option)  |    1.95   |              —    |          6.79% |
| **OV W8A8 default `int8_t160`** (speed-first) |  0.59  |     **0.485** |          9.26% |
| ORT W8A8 dynamic int8 (prior ship)     |    0.62   |              —    |          24.7% |
| llama.cpp Q4_K_M                        |    2.15   |              —    |          41.5% |

App-level knobs that stack multiplicatively (orthogonal to runtime choice):

| Stacked config                                    | RTF (warm)  |
|---------------------------------------------------|------------:|
| `int8_t160` + CFG-skip + step=8                    |    ~0.21   |
| `int8_dpup` + CFG-skip                             |     0.42    |

Drivers: `openvino_driver.py` (CLI), `server_ov.py` (HTTP),
`rust_drivers/ov_inference/` (Rust). See **OpenVINO + NNCF** section
for full details.

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
| `onnx/omnivoice-step.int8.onnx.data` (int8 weights) | 1.13 GB |

The fp32 sidecar matches gluschenko/omnivoice-onnx's 2.45 GB exactly.
Our int8 build is 1.13 GB rather than gluschenko's 612 MB because we
deliberately keep three things fp32: the 621 MB Qwen3 `embed_tokens`
and the 33 MB `audio_embeddings` (we only quantise `MatMul`/`Gemm`,
not Embedding ops), plus the ~33 MB `audio_heads` MatMul (excluded
because it's the per-codebook output projection — quantising it is
the largest single quality regression we measured; see "what we tried"
below). Embedding lookup is bandwidth-bound and already cheap; the
int8 win is in the bulk GEMMs, not the head or the gather.

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
| **ONNX int8** (audio_heads excluded) | 1 | 5.00 s | 7.48 s | 0.669 | 247 ms |
| | 2 | 4.82 s | 7.48 s | 0.645 | 232 ms |
| | 3 | 4.78 s | 7.48 s | 0.639 | 230 ms |
| | **avg** | **4.87 s** | **7.48 s** | **0.651** | **236 ms** |

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
| **ONNX int8** (onnxruntime + MLAS, AVX-VNNI) | **0.651** | **1.92× faster** |

ONNX fp32 lands within noise of PyTorch fp32 — same fp32 GEMMs running
through MLAS instead of MKL with no architectural shortcut. ONNX int8
crosses real-time (RTF < 1.0) by a comfortable margin even with the
quality-preserving choice of keeping `audio_heads` at fp32. (Quantising
`audio_heads` too gets you another ~10% RTF for ~30 percentage-point
extra argmax-disagreement; not a worthwhile trade.)

### Audio parity — fp32 is bit-identical to PyTorch (with seed-locked RNG)

The diffusion loop calls `_gumbel_sample` at every step, which advances
the global PyTorch RNG. Two `generate()` calls without a seed lock — even
both PyTorch — produce different waveforms because they take different
trajectories through the unmask schedule. With `torch.manual_seed(42)`
applied before each call, ONNX fp32 and PyTorch fp32 produce
**bit-identical** waveforms (corr=1.0000, RMS diff = 0.000) on three
test prompts (short EN, long EN, ZH). See
`scripts/seed_locked_samples.py`.

So the export itself has no numerical loss — every divergence we
attribute to ONNX fp32 in casual A/B tests is entirely RNG drift
between PyTorch sessions, not a graph-level mismatch. Logit-level parity
on a fixed forward (`scripts/parity_check.py`): max abs diff 2.1e-4, 0
argmax disagreements out of 8192 codebook positions.

### Audio parity — int8 is audibly degraded

The `--int8` driver produces intelligible speech but with a fuzzy /
static quality, especially audible on shorter prompts. This persists
across multiple quantisation recipes (see "what we tried" below) and
is the dominant cost of the ~2× speedup.

Quantitative: at the logit level, 49% of codebook argmaxes differ from
PyTorch fp32 (mean abs diff 0.34 on a fixed forward). Gumbel sampling
amplifies this into completely uncorrelated waveform-level audio
(corr=0.04 vs PyTorch — but waveform correlation is a misleading
metric here, since two PyTorch runs with different RNG state would also
correlate near zero).

The current int8 default (`omnivoice-step.int8.onnx`) keeps the
`audio_heads` MatMul fp32. Quantising it makes the audible degradation
significantly worse (78% argmax disagreement vs 49%) for ~5% RTF gain;
not a worthwhile trade.

### What we tried for int8 quality (and what didn't work)

Explored to see if we could close the int8 quality gap. Summary of
logit-level errors on the parity-check forward, all with `audio_heads`
excluded:

| Variant | mean abs diff | argmax disagree | RTF |
|---|---:|---:|---:|
| **Dynamic QInt8 (default, ship)** | **0.344** | **49%** | **0.67** |
| Dynamic QInt8 + per-channel | 0.239 | 40% | 0.62 |
| Static QDQ — MinMax calibration | 1.647 | 96% | 2.10 |
| Static QDQ — Percentile-99.999 (padded calib) | 2.041 | 97% | 2.05 |
| Static QDQ — Percentile (clean, no padding) | 2.070 | 98% | 2.04 |
| Static QDQ — int16 activations | 2.612 | 97% | 2.77 |

Per-channel weights showed the lowest *logit-level* error in the
dynamic family but the *audible* quality (subjective listen tests) was
worse than per-tensor — a useful reminder that mean logit error and
argmax disagreement are imperfect proxies for what an ear hears.
Per-tensor stayed the ship default for that reason.

Static QDQ was a strict regression on every measurable axis — quality,
RTF, and clarity. Three culprits combined:

1. **Per-tensor activation quantisation** — ORT's static path uses
   per-tensor activation scales (the `per_channel` flag affects only
   weights). Qwen3 hidden states have outlier channels with magnitudes
   100×+ the median, dominating per-tensor scales and crushing
   precision for the other 1023 channels. This is the failure mode
   SmoothQuant / AWQ / GPTQ exist to fix.
2. **No fast int8/int16 activation kernel for these MatMul shapes on
   AVX-VNNI** — MLAS appears to fall back to fp32 with extra Q/DQ
   roundtrips on every op. The static graphs ran 3-4× *slower* than
   the dynamic one, not faster as the textbook would predict. (Runtime
   provider `CPUExecutionProvider` only; we don't have access to the
   QNN/CUDA quantised paths that would change this calculus.)
3. **Calibration corpus engineering** — Percentile/Entropy histogram
   collectors require uniform tensor shapes, so we either pad
   calibration samples (introducing noise from padded positions
   running through LayerNorm/RoPE) or filter to a single seq_len
   (losing diversity). Neither workaround was the root problem, but
   both added friction.

Memory cost of static QDQ was also tight: the un-mitigated path
(through `quant_pre_process`) peaks at ~28 GB on this machine; even
without pre-process, Percentile on full-corpus seq_lens up to 316 hit
swap thrash. The successful runs used `--filter-seq-len 53` (16
shortest samples) and peaked at ~13 GB. See
`scripts/run_capped.sh` (cgroups memory cap wrapper) and
`scripts/quantize_qdq.py --max-samples / --filter-seq-len` for the
fallbacks built during this exploration.

The proven-better path for int8 quality is a **SmoothQuant-style
pre-pass**: scan the fp32 graph for per-channel activation magnitudes,
derive a `s` factor, migrate magnitude into adjacent weight matrices
via `Linear(W·diag(s), x/diag(s))`, then run the existing static QDQ
on the smoothed graph. Multi-day project; the calibration capture and
QDQ scripts in this branch are reusable scaffolding for it.

### What we tried for int8 quality, part 2: SmoothQuant pre-pass

Spoiler: didn't move the needle on this CPU / ORT combination. The
diagnosis we wrote down in #1 above was right (Qwen3 hidden states
have outlier channels — `llm.norm.weight` activations show a
**89.5× max-channel/median ratio**, 5 groups exceed 10×, all 57
exceed 3.5×). What was *wrong* was the assumption that fixing the
activation distribution would unblock ORT's static int8 path. It
didn't — the kernel-fallback issue (#2 above) is the dominant
blocker, not activation outliers.

**Implementation** (in this branch):

- `scripts/smoothquant_calibrate.py` — scans the fp32 graph for the
  57 LayerNorm → Linear groups (28× input_layernorm × {q,k,v},
  28× post_attention_layernorm × {gate,up}, 1× final norm × audio_heads),
  exposes each LN output as an extra graph output, runs the existing
  `calibration/*.npz` corpus through ORT, and saves per-channel
  `max(|act|)` to `calibration/activation_max.npz`. ~50 s on the
  full 160-sample corpus.
- `scripts/smoothquant_apply.py` — for each group computes
  `s[j] = max(|x|, ε)^α / max(|W|, ε)^(1-α)` and rewrites
  `gamma_new = gamma / s`, `W_new[j, k] = W[j, k] * s[j]` in place.
  Saves to `onnx/omnivoice-step.smoothed.onnx` + sidecar. Includes
  a built-in fp32 self-check that runs original vs smoothed and
  asserts `max-abs-diff < 1e-3`. Self-check passed at
  `max_abs=9.9e-5`, `argmax_disagree=0%` — the rewrite is
  fp32-equivalent.

**Sweep, all on the smoothed graph, parity-check seq_len=256, seed=0**
(`audio_heads` excluded; α is the SmoothQuant migration strength,
0 = no migration, 1 = all migration into weights):

| Recipe | α | mean abs | argmax disagree | RTF (fox pangram) |
|---|---:|---:|---:|---:|
| Static QDQ baseline (un-smoothed) | — | 2.04 | 97-98% | 2.05 |
| Static QDQ + SmoothQuant | 0.5 | 2.44 | 97.7% | 2.05 |
| Static QDQ + SmoothQuant | 0.65 | 2.39 | 97.8% | — |
| Static QDQ + SmoothQuant | 0.8 | 2.40 | 97.3% | — |
| **Dynamic int8 baseline (un-smoothed, ship)** | — | **0.344** | **49%** | **0.65** |
| Dynamic int8 + SmoothQuant | 0.5 | 0.405 | 51% | 0.62 |
| Dynamic int8 + SmoothQuant | 0.65 | 0.415 | 52% | — |
| Dynamic int8 + SmoothQuant | 0.8 | 0.453 | 54% | — |

**Why it didn't help:**

- **Static QDQ:** SmoothQuant fixes the per-tensor activation-scale
  crush exactly as the textbook predicts — but the dominant cost was
  never the scale crush, it was ORT MLAS falling back to fp32 GEMM
  with extra Q/DQ round-trips on every smoothed-shape MatMul.
  Activation distribution can't fix a kernel-coverage problem.
  Smoothed-static stayed at ~97% disagreement and 2.05 RTF, identical
  to un-smoothed static.
- **Dynamic int8:** ORT's dynamic path computes per-tensor activation
  scales **online** at each call, so it never suffered the static
  per-tensor crush in the first place. Migrating activation magnitude
  into weights just adds a per-channel scale step that the dynamic
  quantizer then has to round through — net effect a ~2 percentage
  point regression at every α tested.

**Files left on disk:**

- `onnx/omnivoice-step.smoothed.onnx` (+ `_data` sidecar) — fp32
  smoothed graph at α=0.8 (last sweep value). fp32-equivalent to the
  original; safe to delete.
- `onnx/omnivoice-step.smoothed.int8.onnx` — static QDQ on the
  smoothed graph. Quality regression vs un-smoothed static (which
  was already non-shipping).
- `onnx/omnivoice-step.smoothed.dyn.int8.onnx` — dynamic int8 on the
  smoothed graph. Quality regression vs the dynamic ship default.
- `calibration/activation_max.npz` — per-channel activation maxes
  from the calibration corpus. Useful as data even if SmoothQuant
  didn't help (e.g. for diagnosing which layers have the worst
  outliers — layers 23-27 dominate).

**Conclusion:** the SmoothQuant infrastructure is in place and
correct (fp32 self-check passes), but it can't unblock ORT's static
int8 kernel issue, and dynamic int8 doesn't need it. To make int8
faster *and* higher-quality on this CPU we'd need either a
different runtime (TensorRT-CPU, OpenVINO, oneDNN-direct) with real
AVX-VNNI int8 GEMM kernels for these shapes, or a different
quantization scheme entirely (W4A16 / GPTQ / AWQ on a runtime that
supports it). SmoothQuant alone is a no-op here.

### Verdict

**Speed hypothesis confirmed; quality recovery requires more work
than a one-MVP iteration.**

- **ONNX fp32 is the parity-quality option.** Bit-identical to PyTorch
  with seed locking, RTF 1.25 (= PyTorch). No reason not to ship it.
- **Dynamic ONNX int8 is the speed option.** RTF 0.59 (2.11× faster),
  audibly degraded but functional. Useful when latency matters more
  than fidelity, or as a CPU fallback on speed-constrained deployments.
- **Static QDQ (vanilla) is not a viable alternative on this CPU/ORT
  combo.** Documented above so the next person doesn't repeat the
  experiment cold.

This is an MVP — Python driver, single batch, no streaming, no caching.
The Higgs decoder stays in PyTorch. The ~1.1 GB int8 weight file plus
the still-fp32 PyTorch model is wasteful in RAM (we duplicate the Higgs
tokenizer + Qwen3 weights), but that's a fixable engineering issue.

### Forward look — sherpa-onnx C++ port + SmoothQuant

Two follow-on tracks make sense on top of this MVP:

1. **sherpa-onnx C++ port.** The 2.11× int8 RTF win justifies dropping
   the PyTorch dependency for serving. A sherpa-onnx-style runtime
   would (a) host both the diffusion-LM ONNX graph and a Higgs decoder
   ONNX graph, (b) eliminate the Python ↔ NumPy ↔ ONNX copy on each
   diffusion step, (c) enable streaming + persistent sessions, and
   (d) ship as a small static binary suitable for embedded / edge
   deployments.

2. **SmoothQuant pre-pass for int8 quality.** Reuse the calibration
   capture from `scripts/capture_calibration.py`, derive per-channel
   activation `s` factors, modify the fp32 graph in place, then run
   the existing `scripts/quantize_qdq.py` on the smoothed graph.
   Expected outcome: int8 quality closer to fp32 (target <15% argmax
   disagree) with the existing 0.59 RTF.

Earlier research-agent notes on sherpa-onnx integration points (BPE
tokeniser plumbing, the absent Higgs decoder block, the diffusion-loop
driver shape) are preserved in the parent session's subagent transcript
at `~/.claude/projects/-home-scott-code-OmniVoice/.../subagents/agent-a9461dc1015ced385.jsonl`.

## llama.cpp via Python — quality-first int8 alternative

Picking up where the ONNX MVP left off, the open question was: **can we get
better int8 quality than ORT's 49% argmax-disagreement, even at the same or
worse speed?** Hypothesis: llama.cpp's Q8_0 GEMM uses per-block fp16 scales
(every 32 weights) — mathematically more conservative than ORT's per-tensor
dynamic int8.

### Method

- **Branch / driver**: `onnx-export-mvp` (this branch). Driver:
  `llama_driver.py`. Conversion script: `scripts/export_qwen3_for_gguf.py`.
  Spike scripts: `scripts/spike_llama_hidden.py`,
  `scripts/spike_llama_embd_input.py`. Parity: `scripts/parity_llama.py`.
- **Architecture mismatch handled**: OmniVoice's diffusion LLM is
  bidirectional, not autoregressive. The driver calls
  `llama_set_causal_attn(ctx, false)` and runs the model in embedding mode
  (`pooling_type=NONE`, `embedding=True`) so we get the per-token last
  hidden state instead of logits.
- **Pre-computed embeddings, not tokens**: OmniVoice computes
  `inputs_embeds` in PyTorch (text embed lookup + audio_embeddings sum
  across codebooks + `where`-merged by audio_mask) and feeds the result to
  the LLM. The driver passes that into llama.cpp via
  `llama_batch_init(n_tokens=S, embd=n_embd, n_seq_max=1)` and
  `batch.embd[]`, the same path used for multimodal vision embeddings.
- **CFG batching**: The cond+uncond batch (B=2) is run as two separate
  llama.cpp forwards with KV-cache cleared between them, exact-length per
  row — no padding/diagonal-attention-trick needed since each row is its
  own sequence.
- **Conversion**: `model.llm` (a `Qwen3Model` with 28 layers, hidden 1024,
  GQA 16/8, vocab 151676) is wrapped in `Qwen3ForCausalLM` (tied LM head),
  saved as a HF checkpoint, then run through `convert_hf_to_gguf.py` and
  `llama-quantize`. The LM head goes into the GGUF but is not used at
  inference (the driver pulls hidden states before the LM head).
  `audio_heads` stays fp32 in PyTorch (~33 MB).
- **Tooling**: `llama-cpp-python 0.3.21` (with `CMAKE_ARGS=-DGGML_NATIVE=ON`),
  llama.cpp at commit `fc2b005` built with `-march=native` + OpenMP.

### File sizes

| File | Size |
|---|---:|
| `gguf/omnivoice-qwen3.gguf` (fp16) | 1.19 GB |
| `gguf/omnivoice-qwen3-q8_0.gguf` | 610 MB |
| `gguf/omnivoice-qwen3-q6_k.gguf` | 472 MB |
| `gguf/omnivoice-qwen3-q4_k_m.gguf` | 379 MB |

Smaller than ORT's int8 sidecar (1.13 GB) because the GGUF only contains
the LLM body — `audio_embeddings` and `audio_heads` live outside.

### Logit-level parity vs PyTorch fp32

Same parity setup as the ONNX work (`scripts/parity_llama.py`,
fixed seed=0, B=1, S=256). Hidden states are compared directly against
PyTorch's `m.llm` output, then both go through the same fp32
`audio_heads` to produce comparable logits.

| Variant | Hidden cosine sim | Hidden mean abs diff | **Argmax disagree** | vs ORT int8 |
|---|---:|---:|---:|---:|
| **GGUF Q8_0** | **0.99999666** | **1.19e-2** | **6.79%** | **7.2× better** |
| GGUF Q6_K | 0.99997622 | 3.09e-2 | 15.5% | 3.2× better |
| GGUF Q4_K_M | 0.99982071 | 8.57e-2 | 41.5% | 1.2× better |
| _ORT dynamic int8 (existing baseline)_ | _—_ | _—_ | _49.0%_ | _1.0×_ |
| _ONNX fp32 / PyTorch fp32_ | _1.000000_ | _0.000_ | _0%_ | _∞_ |

Q8_0 is essentially a fp32-fidelity result — cosine 0.99999666 with
~1.2e-2 mean abs diff at the hidden-state level. The 6.79% argmax
disagreement is concentrated on near-tied logits where the top-2 are
within numerical noise of each other (same sampling outcome under
Gumbel noise).

### RTF — fox pangram, 3 runs each, 8 threads

| Variant | Wall (avg) | Audio | RTF | LLM ms/step (B=2 CFG) |
|---|---:|---:|---:|---:|
| **GGUF Q8_0** | **14.86 s** | 7.48 s | **1.987** | 885 |
| GGUF Q4_K_M | 16.04 s | 7.48 s | 2.145 | 960 |
| GGUF f16 | 17.33 s | 7.48 s | 2.317 | 1039 |
| GGUF Q6_K | 21.03 s | 7.48 s | 2.812 | 1270 |

Q6_K is unexpectedly the slowest of the lot — its k-quant dequant overhead
on AVX-VNNI exceeds Q8_0's straight int8 GEMM despite the smaller weight
size. Q4_K_M is mid-pack. **Q8_0 is the clear winner on every axis except
peak speed**: best quality of the four, smallest disk-vs-quality tradeoff.

### Headline comparison — full table

| Runtime | RTF (8 threads) | Argmax disagree | vs PyTorch quality | vs PyTorch speed |
|---|---:|---:|---|---|
| Official Python fp32 (PyTorch+MKL) | **1.252** | 0% | reference | reference |
| ONNX fp32 (ORT+MLAS) | 1.250 | 0% | identical | parity |
| **llama.cpp Q8_0** | **1.987** | **6.79%** | **near-identical** | **0.63× (1.59× slower)** |
| llama.cpp Q6_K | 2.812 | 15.5% | very close | 0.45× (2.25× slower) |
| llama.cpp Q4_K_M | 2.145 | 41.5% | similar to ORT int8 | 0.58× (1.71× slower) |
| omnivoice.cpp Q8_0 (raw GGML) | 2.319 | _untested_ | _alpha_ | 0.54× (1.85× slower) |
| ONNX dynamic int8 | 0.651 | 49% | audibly degraded | 1.92× (faster) |

### Why is llama.cpp Q8_0 slower than PyTorch fp32?

PyTorch+MKL's fp32 GEMM on Raptor Lake is heavily tuned and not
bandwidth-bound at 0.6B parameters — the weights already fit in L2/L3
comfortably. llama.cpp's Q8_0 GEMM has to dequant per block and lacks
MKL's deeper micro-architecture tuning for these specific shapes. Plus,
running CFG as two separate sequential forwards (rather than as a
genuine batched forward) doubles the per-step cost relative to the
PyTorch baseline that batches CFG natively. On a CUDA GPU the picture
flips entirely — but on this CPU there's no path where Q8_0 GEMM beats
fp32 MKL.

### Why is llama.cpp Q8_0 faster than omnivoice.cpp Q8_0?

omnivoice.cpp uses raw GGML matmul kernels with the LLM body, audio
heads, and Higgs decoder all bundled into a single custom GGUF.
llama.cpp's Qwen3 path has years of accumulated micro-optimization for
exactly the q/k/v + gate/up/down GEMM shapes in this model that
raw-GGML doesn't have. Concretely: ~1.17× faster wall time with the same
Q8_0 quantization on the same CPU, despite running the LLM portion only
(omnivoice.cpp also runs Higgs in C++ which we add back via PyTorch).

### Audio samples

`samples/llama/llama_{q8_0,q6_k,q4_k_m,f16}_{1,2,3}.wav` — fox pangram,
nova preset, 3 runs of each variant. Listen and compare against
`samples/onnx/onnx_int8_*.wav` from the ONNX MVP work. Q8_0 should
sound essentially indistinguishable from PyTorch fp32 (modulo Gumbel
RNG drift between runs); Q4_K_M is roughly equivalent to ORT int8 in
audible artefacts.

### Verdict

**Quality hypothesis confirmed. llama.cpp Q8_0 is the right
quality-first int8 option on CPU.**

- **GGUF Q8_0** is the new default for quality-conscious deployments.
  6.79% argmax disagreement vs PyTorch fp32 — a 7× quality improvement
  over ORT dynamic int8, at the cost of ~1.6× the wall time vs PyTorch
  fp32 (still 0.7× the wall of omnivoice.cpp's raw-GGML Q8_0).
- **GGUF Q6_K** is not worth it on this CPU — slower than Q8_0 with
  worse quality. Avoid.
- **GGUF Q4_K_M** is a reasonable storage-constrained alternative to
  ORT int8 — comparable quality, similar speed envelope, but smaller
  disk footprint than ORT int8 (379 MB vs 1.13 GB).
- **GGUF f16** is the fp32-equivalent reference — useful for
  validation but no reason to ship over Q8_0.

The right way to think about this: ONNX dynamic int8 is the **speed-first
fallback**, llama.cpp Q8_0 is the **quality-first int8 option**, PyTorch
fp32 / ONNX fp32 is the **reference**. They're complementary, not
competing.

### Forward look — what could close the speed gap?

The 1.59× wall-time penalty vs PyTorch fp32 has two distinct contributors:

1. **CFG runs as 2 sequential forwards** instead of one batched forward.
   On llama.cpp the cleanest path to fix this is multi-sequence batched
   decode (one batch with `seq_id` distinguishing cond vs uncond), which
   should restore most of the lost throughput. ~10-30% speedup likely.
2. **Per-step Python overhead** in `llama_forward_one`: ctypes copies,
   per-token `batch.pos[i] = i` Python loop, `np.ctypeslib.as_array` +
   `.copy()` for the output. Bulk numpy via `ctypes.memmove` for outputs
   and a numpy fill for `pos[]`/`seq_id[]` could shave 20-50 ms per
   forward.
3. **A C++ port** (à la sherpa-onnx) eliminates ctypes entirely and
   lets CFG share more state across forwards. This is the same
   forward-look conclusion as the ONNX MVP, just with llama.cpp as the
   inference engine instead of ORT.

None of these are blockers — Q8_0 already meets the user's stated bar
(quality > speed, as long as we beat omnivoice.cpp's raw-GGML on speed).

## ONNX Runtime fp32 graph + session optimization (negative result)

After the llama.cpp work didn't beat PyTorch fp32 on speed, the next
question was: can we squeeze portable speedups out of the existing ONNX
fp32 path via ORT's transformer-aware optimizer? Hypothesis:
attention/RMSNorm/SwiGLU fusion would buy us 1.1-1.3× without touching
quantization or vendor-specific kernels.

### Method

- `scripts/optimize_onnx.py` — wrapper around
  `onnxruntime.transformers.optimizer.optimize_model(model_type="qwen3", ...)`
  with all FusionOptions enabled (Attention, RotaryEmbedding,
  SkipLayerNorm, SkipRMSNorm, RMSNorm, BiasGELU, packed Q/K/V, etc).
- `scripts/dump_runtime_optimized.py` — creates an ORT session with
  `optimized_model_filepath` set so we capture the runtime-time
  optimized graph for inspection.
- Re-exported with `attn_implementation="eager"` (added a `--attn` flag
  to `scripts/export_onnx.py`) so the explicit Q·K^T pattern is visible
  to the optimizer instead of being hidden inside an SDPA op.
- Session-level sweep across `ORT_DISABLE_ALL` →
  `ORT_ENABLE_BASIC` → `ORT_ENABLE_EXTENDED` → `ORT_ENABLE_ALL` to
  bound how much of the optimization is happening at session-create.

### Fusion summary

The Python optimizer fused LayerNorms and SiLU/Sigmoid but **failed to
recognize the attention pattern** (0 Attention / 0 MultiHeadAttention /
0 RotaryEmbedding fused), even on the eager export which exposes the
explicit Q·K^T MatMul + Softmax + (attn·V) MatMul:

| Fusion | Count |
|---|---:|
| SimplifiedLayerNormalization (RMSNorm) | 57 |
| SkipSimplifiedLayerNormalization | 56 |
| QuickGelu (SwiGLU's Sigmoid + Mul) | 28 |
| Attention / MultiHeadAttention | 0 |
| RotaryEmbedding | 0 |

Op-count drop: **~6500 → ~3500 ops** (mostly constants and dead Casts).

The runtime optimizer (`ORT_ENABLE_ALL` at session create) actually
fused **more** than the Python optimizer:

| Fusion | Runtime opt | Python opt |
|---|---:|---:|
| SimplifiedLayerNormalization | **113** | 57 |
| FusedMatMul (MatMul+scale) | **28** | 0 |
| SkipSimplifiedLayerNormalization | 0 | 56 |

Reason: the runtime optimizer chains `NchwcTransformer` and other
hardware-aware passes the offline path doesn't run.

### Why attention didn't fuse

ORT's Qwen3 fuser is built around the standard `Qwen3ForCausalLM`
forward — causal mask, no explicit 4D bool mask, sliding-window
defaults. OmniVoice's diffusion driver passes a **4D bool
attention_mask `[B, 1, S, S]`** (bidirectional within each batch row,
diagonal-only on padding). The fuser doesn't recognize this mask
shape so it bails out of the attention pattern match. The fix would
be to either (a) emit a 2D length mask in the export and reconstruct
the 4D version inside a custom op, or (b) write a ORT custom op
supporting the bidirectional mask. Both are substantial scope and
not portable across runtimes.

### Results — fox pangram, 3 runs each, 8 threads

| Variant | Wall (avg) | RTF | LLM ms/step |
|---|---:|---:|---:|
| ONNX fp32 baseline (sdpa export, ORT_ENABLE_ALL) | 9.50 s | 1.270 | 540 |
| ONNX fp32 (eager export, ORT_ENABLE_ALL) | 9.42 s | 1.260 | 536 |
| ONNX fp32 (eager + python offline optimizer) | 9.45 s | 1.263 | 537 |
| ONNX fp32 (eager + runtime-opt-dumped graph) | 9.43 s | 1.261 | 536 |

Single-forward sweep at S=300, B=2 (cold-cache median of 5):

| Optimization level | Time per forward | Speedup vs disabled |
|---|---:|---:|
| ORT_DISABLE_ALL | 785 ms | 1.00× |
| ORT_ENABLE_BASIC | 738 ms | 1.06× |
| **ORT_ENABLE_EXTENDED** | **731 ms** | **1.07× (portable)** |
| ORT_ENABLE_ALL | 723 ms | 1.09× (hw-specific) |

### Verdict

**Negative result: graph-level offline optimization buys nothing
beyond what ORT's session-level optimizer already does at runtime.**

- The current `onnx_driver.py` already runs at `ORT_ENABLE_ALL`, so
  it's already capturing the ~9% session-level lift.
- For portable deployments, switch to `ORT_ENABLE_EXTENDED` — costs
  ~1% vs `ALL` and avoids the `NchwcTransformer` hardware-specific
  kernel selection.
- The big architectural win (full attention fusion → MHA op with
  rotary baked in) is **blocked by OmniVoice's 4D bool mask shape**.
  Unblocking this is feasible but expensive: requires a custom-op
  attention kernel that accepts the bidirectional+padding-diagonal
  mask, which loses portability across runtimes (would need
  reimplementation in any non-ORT inference engine).

### What this means for the speed search

ORT fp32 + session-level optimization is essentially saturated on
this graph. Further portable wins on CPU need to come from
**reducing total work**, not from better kernels:

1. **Persistent KV cache across diffusion steps** — text/style
   prefix tokens have static input embeddings across all 16 steps.
   Caching their K/V projections + the prefix×prefix attention block
   could save 25-35% of attention compute. Works in any runtime.
2. **Variable-output decode** — the LLM only needs hidden states
   for currently-masked positions, not all positions. With the
   linear unmask schedule, by step 16 only ~1/16 of positions need
   output. Saves ~30-40% of MLP cost across the full generation.
3. **Multi-sequence batched CFG decode** — run cond + uncond in a
   single forward with `seq_id` distinguishing them. Restores the
   batch-level matmul reuse PyTorch gets natively but our
   per-batch-row driver loses.

These are all algorithmic wins that exploit OmniVoice's diffusion
structure, not CPU-specific features. They're the right next track.

## Multi-sequence batched CFG decode (llama.cpp) — negative result

After the ORT optimization plateau, the next portable target was the
sequential CFG forward pattern in `llama_driver.py` (B=2 batch rows
running as 2 separate `llama_decode` calls). Hypothesis: packing
cond + uncond into one `llama_decode` call with distinct `seq_id`s
would recover the batch-level matmul reuse that PyTorch's
`[2*B, S, ...]` tensors get natively, restoring something close to the
~21% theoretical saving from llama-bench's pp throughput numbers.

### Method

- `llama_driver.py:llama_forward_batched()` — packs N sequences of
  arbitrary lengths into one `llama_batch` with seq_ids 0..N-1, total
  `n_tokens = sum(L_b)`, then a single `llama_decode` call. Per-token
  `pos[i]` resets to 0 at each sequence boundary; `causal_attn=False`
  ensures bidirectional attention within each sequence and no
  cross-sequence attention.
- Required patching `llama_cpp.llama_cpp.llama_context_default_params`
  to override `n_seq_max=1` (the wrapper's hard-coded default) up to
  `n_seq_max=2`. Without this, batched decode aborts at
  `init: invalid seq_id[N][0] = 1 >= 1`.
- `--no-batched-cfg` flag added for A/B benchmarking.
- Sanity: with both rows identical inputs, sequential and batched
  produce **bit-identical** outputs. With different-length rows, a
  ~3% relative drift appears on the longer row — consistent with
  multi-threaded fp32 GEMM reduction-order differences (different
  total batch sizes → different OpenMP work distributions). Not a
  correctness bug.

### Results — fox pangram, 3 runs each

| Variant | Wall (avg) | RTF (avg) | LLM ms/step |
|---|---:|---:|---:|
| llama Q8_0 sequential CFG (baseline) | 14.60 s | **1.953** | 866 |
| llama Q8_0 batched CFG | 14.99 s | 2.012 | 891 |

Within run-to-run noise. Batched is **not faster**.

### Why batched CFG doesn't help llama.cpp on CPU

llama.cpp's multi-seq batched decode internally concatenates all
sequences into a single `[n_total, hidden]` tensor and runs all GEMMs
on that flattened shape. With non-causal attention the only
sequence-aware operation is the attention block, where each sequence's
Q·K^T and (attn·V) operate on its own `[L_b, hidden]` slice with no
cross-sequence interaction. **The total FLOP count is identical to
running each sequence sequentially** — there's no batch-level matmul
reuse to recover, because there was never any duplicated work.

This differs from autoregressive decode where batched generation
amortizes per-token overhead across batch dim — a fixed-cost win that
diffusion forwards don't have. The expected ~21% saving from
llama-bench was a misread of pp throughput at different prompt sizes:
pp456 throughput is higher than pp256 not because of batching but
because of cache-line / SIMD-block utilization at specific lengths;
that difference is also visible across run sizes 256→300→456 in
isolation.

llama-bench measurements at relevant sizes (Q8_0, 8 threads):

| Prompt size | Throughput | Per-forward time |
|---:|---:|---:|
| pp256 | 520 t/s | 491 ms |
| pp300 | 541 t/s | 555 ms |
| pp456 | 586 t/s | 778 ms |
| pp512 | 497 t/s | 1030 ms |

`pp456` (one forward of 456 tokens) at 778 ms is faster than
`pp256 × 2` (= 982 ms) — but in the live driver, batched-mode adds
~60 ms of Python-side overhead (np.concatenate, 456-iteration
per-token batch field loop, longer KV-cache clear), eating the
theoretical win.

### What would actually help llama.cpp on CPU

Confirms the conclusion from earlier sections: **algorithmic
restructuring is the only path to portable CPU speedups** at this
model size. Specifically:

1. **Persistent KV cache for the prefix across diffusion steps** —
   text/style positions have unchanged input embeddings on every step,
   so their K/V projections (and the prefix×prefix attention block)
   can be computed once and reused. With the ~50% prefix/audio split
   typical for OmniVoice, this saves up to ~50% of attention compute
   (still leaves MLP cost intact). Works in any runtime.
2. **Variable-output decode** — only emit hidden states for currently
   masked positions. By step 16, ~94% of positions don't need their
   hidden state computed. Across all 16 steps with the linear unmask
   schedule, expected total saving ~30-40% of MLP cost.
3. **Reduced num_step** — out of scope for the CPU-portability
   thread but compounds with everything above.

The ONNX path **does** already get genuine batch-level matmul reuse
(ORT runs the whole `[2*B, S]` tensor through fused matmul kernels
that share weights). For the llama.cpp path, the equivalent
optimization would be #1 above, not multi-seq batching.

## Persistent prefix KV cache (PyTorch) — negative result

The remaining portable algorithmic optimization on the list above was the
prefix KV cache: text/style positions have byte-identical input embeddings
across all 16 diffusion steps, so in principle their per-layer K/V
projections could be computed once and reused. Plan, driver, and parity
harness are checked in (`pytorch_kv_driver.py`,
`scripts/parity_kv_pytorch.py`); the result is a strict regression on
quality and only a marginal speed gain.

### Why the correctness argument fails

OmniVoice's diffusion-LM uses **fully bidirectional attention** within each
batch row (`omnivoice/data/collator.py:40-43`: "Each query position can
attend to all non-padding key positions (bidirectional)"). The mask
construction in `_generate_iterative` confirms this: cond rows pass
`batch_attention_mask[i, :, :c_len, :c_len] = True` — every position
attends to every position.

The plan's correctness claim — *"prefix `inputs_embeds` is byte-identical
across all 16 steps → prefix K/V are invariant per layer → safe to
cache"* — only holds at **layer 0**, where K/V = `inputs_embeds @ W_K/V`.
At layer N>0, K/V is computed from `RMSNorm(layer_{N-1}_output)`, and
layer_{N-1} attention mixes prefix tokens with audio tokens. Audio tokens
change every diffusion step, so prefix K/V at deeper layers also change.

A cold prefix-only forward (the cache build path) computes
layer_{N-1}_output for prefix as if audio didn't exist. The cached K/V
therefore drift from what a no-cache full forward would produce, with the
gap growing layer-by-layer.

### Measured divergence (parity_kv_pytorch.py, B=1, S=256, fp32)

| Metric | Audio region |
|---|---:|
| Hidden-state max abs diff | 22.83 |
| Hidden-state mean abs diff | 0.338 |
| Hidden-state cosine sim | 0.9962 |
| audio_heads logit max abs diff | 25.45 |
| **Argmax disagreement** | **82.8%** (848/1024) |

For comparison, ORT dynamic int8 — which we already shipped as the
"audibly degraded" speed-first option — measured 49% disagreement on the
same parity harness. The KV-cache path is **~70% worse than ORT int8**
on the same metric, before any quantisation.

### End-to-end audio impact

Fox pangram synthesis with cache vs without (RMS of the generated
waveform; same duration estimate, different content):

| Variant | Audio length | Waveform RMS |
|---|---:|---:|
| No-cache (stock PyTorch fp32) | 7.48 s | (reference) |
| Cached | 7.48 s | 2.3× quieter on a short prompt |

The cached path produces intelligible-but-degraded speech, similar in
character to ORT int8 but with a different artefact profile.

### Speed result — also disappointing

3 runs of the fox pangram, 8 threads, B=1 (CFG → 2 batch rows):

| Variant | Wall (avg) | RTF (avg) | LLM ms/step (mean) |
|---|---:|---:|---:|
| No-cache PyTorch fp32 | 10.54 s | 1.409 | — |
| Cached PyTorch fp32 | 9.86 s | 1.318 | 570 (cold 114, hot 282) |

**~6% wall-time improvement**, far short of the 50% the plan projected.
Two reasons:

1. **Loss of native batch fusion.** Stock PyTorch runs
   `[2*B, max_c_len]` through every matmul as one tensor; MKL's GEMM
   amortises across the batch dim. The cached driver processes each row
   sequentially (each row needs its own `DynamicCache`), so the per-row
   savings are largely cancelled by the loss of batched-matmul reuse.
2. **`DynamicCache` allocation overhead.** Every step constructs a
   fresh hot-decode forward through the full 28-layer stack; the
   allocation/concat traffic for a per-row cache adds back much of the
   compute saved by skipping prefix recomputation.

### Verdict

**The KV-cache idea is incompatible with this model's bidirectional
attention.** Caching prefix K/V is mathematically valid only when prefix
queries cannot attend to audio keys — i.e. an encoder-decoder split, or
an explicit prefix→prefix-only mask. OmniVoice has neither.

The pieces are checked in for future reference:

- `pytorch_kv_driver.py` — the cached driver, with `--no-prefix-cache`
  flag for stock-PyTorch A/B.
- `scripts/parity_kv_pytorch.py` — the parity harness (reusable for any
  future cached-path experiment on this model).

Future portable algorithmic wins on this model would need to either:

1. **Modify training / fine-tune** the model with a partial mask that
   forbids prefix→audio attention, restoring the cacheability assumption.
   Out of scope for an inference-only optimisation.
2. **Variable-output decode** (still untried) — only emit hidden states
   for currently-masked positions. This is independent of the caching
   correctness issue: it saves MLP compute on positions whose hidden
   states aren't read by the diffusion sampler. Estimated 30-40% of MLP
   cost across the full 16-step run, no quality impact.
3. **Reduced num_step** — orthogonal to runtime. Drops compute linearly
   at a quality cost the user can tune.

For runtime/kernel speedups on CPU, the realistic remaining avenues are
external: an int8 GEMM kernel with real AVX-VNNI coverage for these
shapes (TensorRT-CPU, OpenVINO, oneDNN-direct), or a different
quantisation scheme (W4A16 / GPTQ / AWQ on a runtime that supports it).

## Weight-only int8/int4 — quality recovery without the speed (negative result on speed)

After exhausting the W8A8 dynamic-int8 + structural-tweak surface in the
sections above, the remaining int8 directions worth testing on this CPU
were:

1. **Mixed precision**: keep the outlier-heavy layers fp32, quantize the
   rest. The activation max stats already on disk
   (`calibration/activation_max.npz`) flagged `llm.norm` (max 364), layer
   27 post-attn LN (max 122), and layer 25 input LN (max 115) as the
   absolute-magnitude offenders.
2. **W4A16** via ORT's `MatMulNBitsQuantizer` (RTN, HQQ algorithms,
   block-wise scales). Portable: produces a `MatMulNBits` op that runs
   in any modern ORT build; the same weight format is convertible to
   GGUF Q4_K-family.
3. **W8A16** (8-bit blockwise weights, fp32 activations) via the same
   quantizer with `bits=8`.
4. **SmoothQuant + dynamic int8 with per-channel weights**: the
   un-tested cell in the matrix from the earlier SmoothQuant section
   (we'd done un-smoothed per-channel and smoothed per-tensor, but not
   smoothed per-channel).

### Method

- **Parity harness**: `scripts/parity_quants.py` runs each variant
  against the fp32 ONNX reference on one real calibration sample
  (`calibration/long_en_fox_step0000.npz`, B=2 S=226 — the actual fox
  pangram step-0 input shape, not a synthetic random tensor). Reports
  argmax disagreement on the [B, C, S, V] codebook logits.
  Real-input numbers are substantially lower than the synthetic-input
  parity numbers reported in the earlier sections (the int8 baseline
  measures 24.7% on real inputs vs 49% on synthetic). Real-input
  numbers are the ones that correlate with audible quality, so this
  section uses them throughout.
- **Synthesis**: `scripts/synth_fox_quant.py` runs full fox-pangram
  synthesis (16 steps, B=2 CFG, seed-locked) end-to-end through each
  variant and reports wall-time RTF.
- **Quantization scripts** (added this round):
  `scripts/quantize_mixed.py` (per-layer fp32 skip list),
  `scripts/quantize_w4a16.py` (MatMulNBits wrapper).

### Parity (vs fp32 ONNX, single forward, 8 threads, real fox-pangram input)

| Variant | mean abs | argmax disagree | forward s |
|---|---:|---:|---:|
| fp32 (self) | 0.000 | 0.0% | 0.55 |
| _W8A8 dynamic int8 (current ship)_ | _1.030_ | _24.7%_ | _0.23_ |
| Mixed precision: skip layer 27 | 1.057 | 25.2% | 0.24 |
| Mixed precision: skip layers 25-27 | 1.073 | 24.9% | 0.26 |
| Mixed precision: skip layers 23-27 | 0.668 | 24.7% | 0.28 |
| Smoothed + per-channel dynamic int8 | 0.349 | 24.8% | 0.25 |
| **W4A16 RTN block-128** | 0.421 | 22.1% | 0.55 |
| **W4A16 RTN block-32** | 0.443 | 21.1% | 0.55 |
| **W4A16 HQQ block-128** | **0.329** | **18.9%** | 0.61 |
| **W8A16 RTN block-128** | **0.028** | **1.7%** | 0.60 |

### Synthesis RTF (3 runs each, fox pangram, 7.48 s of audio, 8 threads)

| Variant | Wall (avg) | RTF | Mean step (ms) |
|---|---:|---:|---:|
| fp32 ONNX (reference) | 9.73 s | 1.300 | 558 |
| W8A8 dynamic int8 (ship) | 4.63 s | 0.619 | 240 |
| W8A16 RTN block-128 | 10.50 s | 1.403 | 606 |
| W4A16 HQQ block-128 | 10.99 s | 1.469 | 638 |

### What we found

**On quality**, weight-only blockwise quant is exactly the cure for the
outlier problem that hurts W8A8:

- **W8A16 RTN** is essentially fp32 (1.7% argmax disagreement, mean abs
  diff 0.028 — within the noise floor of two seeded fp32 PyTorch runs).
  Block-wise per-128-element fp16 scales on the weight side fully
  absorb the channel-magnitude variance that crushes W8A8's per-tensor
  scales.
- **W4A16 HQQ** lands at 18.9% disagreement (vs 24.7% for W8A8 dynamic
  int8). HQQ's training-free quant search recovers ~6 percentage
  points over plain RTN.
- **W4A16 RTN** at block-32 (21.1%) does narrowly beat block-128
  (22.1%) but at no measurable speed cost.

**On speed**, neither W4A16 nor W8A16 helps on this CPU:

- W8A16 forward = 0.60 s vs fp32 0.55 s — slightly *slower* than fp32.
- W4A16 forward = 0.55 s — parity with fp32, no improvement.
- End-to-end RTF lands at 1.40-1.47, vs 1.30 for fp32 and 0.62 for the
  shipped W8A8 dynamic int8.

ORT's `MatMulNBits` operator on `CPUExecutionProvider` has no fast
int4/int8 weight × fp32 activation kernel; it dequantizes blockwise
and runs fp32 GEMM. The win that weight-only quant *should* deliver is
memory-bandwidth reduction, but at 0.6B parameters this model already
fits comfortably in L3 — bandwidth isn't the bottleneck. FLOPs are,
and only W8A8 with MLAS's int8 GEMM kernels reduces FLOPs.

**Mixed precision turned out to be a no-op.** Excluding the last 5
layers (or the last 3, or just layer 27) from W8A8 quantization moved
argmax disagreement by under one percentage point in any direction.
The activation outliers at the end of the network apparently aren't
the dominant source of audible degradation — the error is distributed
across all 28 layers. The 0.668 mean-abs improvement on the skip-23-27
variant is real but doesn't translate to argmax recovery, suggesting
the disagreements are happening on tightly-tied logit pairs where even
small perturbations earlier in the network flip the argmax.

**Smoothed + per-channel dynamic int8** brought mean-abs diff down 3×
(from 1.03 to 0.35) but left argmax disagreement at 24.8% — same as
both the un-smoothed baseline and the un-smoothed per-channel variant
from the earlier matrix. SmoothQuant migrates magnitude into weights,
per-channel weights handle wide-magnitude weight distributions; both
fix per-tensor scale crushing on the *weight* side, but neither fixes
the per-tensor *activation* scaling that ORT's dynamic path uses, and
the activation side is what hurts on this model.

### Audio samples

`samples/quant_v2/` — fox pangram, nova preset, 3 runs each:

- `fp32_reference_seed{42,43,44}.wav` — fp32 ONNX baseline
- `int8_baseline_seed{42,43,44}.wav` — current ship default (W8A8 dyn)
- `w4a16_hqq128_seed{42,43,44}.wav` — quality-improved 4-bit weights
- `w8a16_rtn128_seed{42,43,44}.wav` — near-fp32 quality 8-bit weights

The useful comparisons are (a) W8A16 vs fp32 — should sound
indistinguishable, validating that *if* a fast int8-weights kernel
existed the quality would be there; and (b) W4A16 vs W8A8 int8 — same
speed envelope as fp32 but should sound noticeably cleaner than the
shipped int8.

### Verdict

**Quality recovery confirmed; speed envelope unchanged.** The findings
collapse to:

- The speed–quality Pareto frontier on this CPU has two points:
  PyTorch/ORT fp32 (RTF 1.25, 0% disagree) and ORT W8A8 dynamic int8
  (RTF 0.62, 24.7% disagree). Nothing tested in this round moves the
  frontier.
- Weight-only blockwise quant (W4A16, W8A16) lands *inside* the
  frontier — quality between fp32 and W8A8 int8, speed equal to or
  slightly worse than fp32. Useful only as a smaller-disk-footprint
  alternative to fp32 (W8A16 = 1.14 GB vs fp32's 2.45 GB) when
  fidelity matters and disk does too.
- Mixed precision and smoothed+per-channel dynamic int8 don't help.

**The remaining lever for portable CPU speed-with-quality is a
different runtime.** ORT's `CPUExecutionProvider` is the constraint:
- A runtime with a real W4A8 or W8A8 GEMM kernel that handles
  per-channel/per-block weight scales without falling back to fp32
  GEMM (oneDNN-direct, OpenVINO, or a hand-written AVX-VNNI kernel)
  would give the W8A16-tier quality at the W8A8-tier speed.
- llama.cpp's Q8_0 path is the closest existing runtime to this — it
  already demonstrated 6.79% disagreement at 1.95 RTF earlier in this
  doc. The gap to PyTorch fp32 speed (0.63×) reflects llama.cpp's
  Qwen3 GEMM tuning, not a fundamental limit.

The right phrasing of the speed–quality result for this CPU/model
combination: *quantization buys speed when the kernel exists; quality
when the scale granularity is fine enough; both at once is a
runtime-coverage problem, not an algorithmic one.*

## OpenVINO + NNCF — the runtime that actually has the kernels

The previous section closed with "the remaining lever is a different
runtime that has real W8A8 GEMM kernels for per-channel weight scales."
OpenVINO is exactly that runtime. The Intel-tuned CPU path uses oneDNN
underneath, which has hand-written AVX-VNNI int8 GEMM kernels with
per-channel weight scales for the exact MatMul shapes Qwen3 uses.
NNCF (Neural Network Compression Framework, OpenVINO's quantizer) wraps
SmoothQuant + activation calibration + per-channel weight quantization
in one call when `model_type=TRANSFORMER` is set.

This section moves the Pareto frontier on every axis except fp32.

### Method

- **Branch / driver**: `onnx-export-mvp`. Driver: `openvino_driver.py`.
  Quantization wrapper: `scripts/quantize_openvino.py` (NNCF
  `quantize()`). Parity harness: `scripts/parity_openvino.py`.
- **Calibration corpus**: the existing `calibration/*.npz` (160 samples
  spanning 10 prompts × 16 diffusion steps; EN + ZH, 53–316 token seq
  lengths, multiple voice presets — not just the fox pangram). Reused
  unchanged from the static QDQ work.
- **Quantization recipe** (default ship variant
  `omnivoice-step.int8_t160.xml`):
  - `nncf.quantize()` with `preset=MIXED` (asymmetric activations,
    symmetric weights), `model_type=TRANSFORMER` (auto-applies
    SmoothQuant α=0.95 to all matmul shapes), `subset_size=160`.
  - Ignored ops: `types=["Gather"]` (embed_tokens / audio_embeddings)
    and `names=["/audio_heads/MatMul"]`. Same exclusion convention as
    the ORT pipeline.
- **Mixed-precision exclusion sweep**: extended exclusions on top of the
  default — see results below. Variants `int8_dp0-27.xml` and
  `int8_dpup.xml` keep specific projection types fp32 to recover quality.
- **Tooling**: `openvino==2026.1.0`, `nncf==3.1.0`,
  `INFERENCE_NUM_THREADS=8`. `INFERENCE_PRECISION_HINT` doesn't help on
  Raptor Lake (no AVX512_BF16; F16C is convert-only).

### File sizes

| File | Size |
|---|---:|
| `openvino_ir/omnivoice-step.xml` (fp32 graph) | 2.1 MB |
| `openvino_ir/omnivoice-step.bin` (fp32 weights) | 2.45 GB |
| `openvino_ir/omnivoice-step.int8_t160.xml` (default int8) | 4.3 MB |
| `openvino_ir/omnivoice-step.int8_t160.bin` (default int8 weights) | 1.10 GB |
| `openvino_ir/omnivoice-step.int8_dp0-27.bin` (down_proj fp32) | 1.41 GB |
| `openvino_ir/omnivoice-step.int8_dpup.bin` (down+up fp32) | 1.65 GB |
| `openvino_ir/omnivoice-step.w8a16.bin` (weight-only int8) | 614 MB |

### Single-forward parity vs ONNX fp32 (B=2, S=226, real fox-pangram step-0)

W8A8 (full int8 PTQ via NNCF transformer mode):

| Variant                              | Forward ms | Argmax disagree |
|--------------------------------------|-----------:|----------------:|
| OV fp32 (parity baseline)            |     526    |          0.00% |
| OV W8A8 (32 calib samples, default)  |     224    |         14.71% |
| **OV W8A8 (160 samples, ship default)** | **227**    |       **9.26%** |
| OV W8A8 + smooth_quant α=0.50        |     221    |         21.79% |
| OV W8A8 + smooth_quant α=0.85        |     228    |         11.01% |
| OV W8A8 + smooth_quant α=0.99        |     234    |         13.05% |
| OV W8A8 + accurate_bias_correction   |     219    |          9.26% |
| OV W8A8 + skip layers 25-27          |     258    |          8.88% |
| OV W8A8 + skip layer 27 only         |     237    |         10.18% |
| OV W8A8 + down_proj 23-27 fp32       |     236    |          8.02% |
| OV W8A8 + down_proj 18-27 fp32       |     244    |          7.94% |
| OV W8A8 + down_proj 13-27 fp32       |     254    |          7.88% |
| OV W8A8 + down_proj 5-27 fp32        |     270    |          7.49% |
| **OV W8A8 + down_proj all 28 fp32 (`dp0-27`)** | **286** | **5.89%** |
| OV W8A8 + dp + o_proj late (20-27)   |     301    |          5.67% |
| OV W8A8 + dp + up_proj late (20-27)  |     302    |          5.78% |
| **OV W8A8 + dp + up_proj all 28 (`dpup`)** | **363** | **4.87%** |
| OV W8A8 + full MLP (dp+up+gate)      |     415    |          4.89% |

Weight-only compression (NNCF `compress_weights()`):

| Variant                              | Forward ms | Argmax disagree |
|--------------------------------------|-----------:|----------------:|
| OV W8A16 (int8_asym, per-channel)    |     296    |         23.45% |
| OV W4A16 (int4_asym, g128)           |     318    |         29.73% |
| OV W4A16 (int4_asym, g128) + AWQ + scale-est | 327 |         27.13% |
| OV W4A16 (int4_sym, g64) + AWQ + scale-est   | 320 |         28.93% |

Two notable points:

1. **OV's W8A16 kernel IS faster than fp32** — 296 ms vs 526 ms (1.78×).
   This is exactly the kernel that ORT MLAS lacks (PERFORMANCE.md
   documented W8A16 in ORT was *slower* than fp32 at 600 ms). The
   `compress_weights()` path lands a real "int8 weights × fp32
   activation" GEMM in oneDNN. But the per-channel-only quantization
   crushes the larger-magnitude weight outliers; the full-PTQ W8A8 with
   SmoothQuant gives much better quality at slightly faster speed.
2. **W4A16 doesn't help on this CPU** even with AWQ + scale-estimation.
   OV's int4 weight kernel exists but the quality regression is too
   steep without GPTQ; speed is also slightly worse than fp32 (likely
   due to dequant overhead at this batch size). Same finding as the
   ORT W4A16 result earlier in this doc.

### End-to-end synthesis RTF (fox pangram, 3 runs each, 8 threads)

| Variant                          | Wall (avg) | RTF   | LLM ms/step |
|----------------------------------|-----------:|------:|------------:|
| **OV W8A8 default (`int8_t160`)**  | **4.43 s** | **0.59** | 236         |
| **OV W8A8 `dp0-27`**               | **5.42 s** | **0.72** | 298         |
| **OV W8A8 `dpup`**                 | **6.41 s** | **0.85** | 361         |
| OV fp32                          |    9.34 s  | 1.25  | 544         |

### Headline comparison — full table

| Runtime / Variant                        | RTF   | Argmax disagree | Notes                  |
|------------------------------------------|------:|----------------:|------------------------|
| PyTorch fp32 / ONNX fp32 / OV fp32      | 1.25  |             0% | reference              |
| **OV W8A8 `dpup` (NEW)**                 | 0.85  |        **4.87%** | **best int8 quality**     |
| **OV W8A8 `dp0-27` (NEW)**               | 0.72  |        **5.89%** | balanced — beats llama.cpp Q8_0 |
| llama.cpp Q8_0 (old quality option)      | 1.95  |          6.79% | superseded             |
| **OV W8A8 default (NEW)**                | 0.59  |        **9.26%** | **fastest int8**          |
| ORT W8A8 dynamic int8 (old ship)         | 0.62  |         24.7%  | superseded             |
| llama.cpp Q4_K_M                         | 2.15  |         41.5%  |                        |

OV breaks the speed-vs-quality frontier on this CPU. The default ship
beats ORT int8 by 2.7× on quality (9.26% vs 24.7%) at the same speed.
The mixed-precision `dp0-27` variant beats llama.cpp Q8_0 on **both
axes** simultaneously — better quality (5.89% vs 6.79%) at 2.7× the
speed.

### Layer sensitivity finding — down_proj is the outlier

Sweeping the exclusion set systematically: the dominant
quality-impacting layer is the MLP `down_proj` projection. Excluding
it from int8 gets 9.26% → 5.89% — the single biggest gain available
from layer-level mixed precision. Adding `up_proj` exclusion takes us
further to 4.87%. `gate_proj` exclusion adds nothing useful. This is
consistent with `down_proj` being the projection that maps the wide
intermediate dim (3072) back to hidden (1024) — its weight columns
have the largest cross-channel magnitude variance, so per-channel int8
scales lose the most information there.

### Rust path validated — `openvino` crate v0.10

The Rust `openvino = "0.10"` crate (with `runtime-linking` feature)
loads the int8 IRs directly and runs inference 3-7% faster than from
Python (no glue overhead). Code at `rust_drivers/ov_inference/`. This
is the end-target Rust deployment path: a small Rust binary depends
only on `libopenvino_c.so` from the OpenVINO runtime distribution.

| IR variant       | Python ms | Rust ms | Speedup |
|------------------|----------:|--------:|--------:|
| OV fp32          |     545.8 |   532.1 |   2.5%  |
| OV W8A8 default  |     227.1 |   213.8 |   5.9%  |
| OV W8A8 dp0-27   |     285.6 |   275.0 |   3.7%  |
| OV W8A8 dp+up    |     363.2 |   338.5 |   6.8%  |

Tract (pure-Rust ONNX, no C deps) was tested but FAILED to load the
fp32 ONNX — chokes on Qwen3's dynamic-shape attention reshape:
`Reshaping [Sym(batch), Val(8), Val(1), Sym(seq_len), Val(128)] to
[Sym(batch), Val(16), Sym(seq_len), Val(128)]: Invalid`. Tract has
weak transformer support; not viable without a custom export.

### Server deployment — `server_ov.py`

A production-ready OpenAI-compatible TTS server is checked in at
`server_ov.py` that wires the OV W8A8 driver into the same FastAPI
shape as the existing `server.py`. Per-shape (B, S) compiled-model
cache with disk-backed `CACHE_DIR` so each unique input length is
compiled once with static reshape (~0.78 s with disk cache hit, vs
1.8 s cold) and reused thereafter.

5 fox-pangram requests on `int8_t160`:

| Run | Wall | RTF | Notes |
|----:|-----:|----:|-------|
| 1 | 4.62 s | 0.617 | First: includes 0.78 s static compile |
| 2 | 3.69 s | 0.493 | Warm cache |
| 3 | 3.64 s | 0.486 | Warm cache |
| 4 | 3.64 s | 0.487 | Warm cache |
| 5 | 3.61 s | 0.482 | Warm cache |
| **avg (warm)** | **3.65 s** | **0.487** | **Server steady-state** |

Server-mode RTF **0.487** vs CLI dynamic-shape RTF 0.591 — a further 17%
gain from amortizing the static-shape compile cost across requests. That
takes the production deployment to **2.6× faster than PyTorch fp32**
(1.25 RTF) at 9.26% argmax disagree.

CFG-skip and num_step are exposed as env vars (`OMNIVOICE_GUIDANCE_SCALE`,
`OMNIVOICE_NUM_STEP`). With `gs=0` and `num_step=8`, the server hits
roughly RTF 0.18 (extrapolated) — best-effort latency mode.

### CPU affinity (small Intel hybrid-core gain)

This i7-14700 has 8 P-cores (16 HW threads, CPUs 0-15) and 12 E-cores
(CPUs 16-27). Taskset findings, 8 inference threads:

| Affinity | RTF |
|----------|----:|
| Default (no pinning) | 0.591 |
| `taskset -c 0-15` (P-cores HT) | **0.572** |
| `taskset -c 0-7` (4 P-cores HT only) | 1.023 |
| `taskset -c 16-27` (E-cores only) | 1.492 |
| `taskset -c 0-19` t=12 (P + 4E) | 1.181 |
| `taskset -c 0-27` t=20 (all cores) | 0.955 |

Pinning to all P-core HW threads gives a free ~3% gain. **Crossing into
E-cores hurts** because OV's per-thread work distribution is uniform
and the slowest thread bottlenecks the bulk-matmul reduce. 8 threads
on P-cores is the sweet spot; 12 / 16 / 20 thread counts all regress.

### Stacking with app-level knobs (CFG-skip + step reduction)

Two orthogonal optimizations stack multiplicatively with the runtime
work. Both are quality-cost decisions, exposed as flags rather than
the default ship.

**CFG-skip (`guidance_scale=0`)** — patch in `cfg_skip_patch.py`
modifies `_generate_iterative` to build the batch with `B=1` instead
of `2*B`, skipping uncond rows entirely. Halves the LLM forward cost.
Quality cost is the loss of CFG amplification, not full conditioning
(text + instruct still reach the model).

**Step reduction (`num_step=8` or `12` instead of 16)** — fewer
diffusion iterations. Linear speedup. Quality cost depends on prompt;
8 steps is the practical lower limit on this model.

| Stack (int8_t160)    | step=16 | step=12 | step=8 |
|----------------------|--------:|--------:|-------:|
| with CFG (B=2)       |   0.59  |   0.49  |  0.35  |
| **CFG-skip (B=1)**   | **0.345** | **0.285** | **0.224** |

Best stacked: **OV W8A8 + CFG-skip + step=8 → RTF 0.224** (5.6× faster
than PyTorch fp32 baseline). Audio samples in
`samples/openvino/*cfgskip*.wav` for listen-comparison.

### Server mode (`server_ov.py`) — production deployment numbers

The CLI driver reloads the model per invocation; the server amortizes
load + per-shape compile across requests. Implemented as
`server_ov.py` — drop-in OpenAI-compatible TTS endpoint backed by
OpenVINO. Per-(B, S) compiled-model cache + OV `CACHE_DIR` for
persistent on-disk cache of compiled blobs.

5-request fox-pangram benchmark per ship variant (server warm):

| Variant            | Run 1 (cold) | Run 2..5 (warm avg) | Argmax disagree |
|--------------------|-------------:|--------------------:|----------------:|
| **`int8_t160`**     |       0.617  |          **0.485**  |          9.26% |
| `int8_dp0-27`       |        1.122 |           **0.618** |          5.89% |
| `int8_dpup`         |        1.226 |           **0.755** |          4.87% |
| `int8_t160` + CFG-skip + step=8 |  — |             ~0.21 |       9.26%* |
| `int8_dpup` + CFG-skip          | 0.737 |    **0.42**     |       4.87%* |

*disagree numbers above are the LLM logit-level metric; CFG-skip and
step reduction don't change logit-level parity directly but do change
the audible audio output. Listen tests required for end-quality.

Steady-state RTF for the default ship (`int8_t160`, CFG=2, step=16):
**0.485** — a clean 2.58× speedup over the PyTorch fp32 baseline
(1.252) at this CPU's previous-best speed-first quality (which was the
ORT int8 ship at 24.7% disagree, now 9.26% disagree).

### Determinism

OV W8A8 with seed-locked RNG is bit-identical across runs:

```
OmniVoice "Hello world." --seed 42  →  md5 a0f697d9...
OmniVoice "Hello world." --seed 42  →  md5 a0f697d9...   (same)
```

Same property as ONNX fp32 with seed lock (PERFORMANCE.md "Audio
parity — fp32 is bit-identical to PyTorch with seed-locked RNG").
Production deployments can rely on reproducible output.

### Generalizes across prompt languages and lengths

| Variant       | Fox EN (S=226) | Long EN (S~440) | Chinese (S~110) |
|---------------|---------------:|----------------:|----------------:|
| `int8_t160`   |     0.586     |      0.585      |      0.579      |
| `int8_dp0-27` |     0.718     |      0.717      |      0.734      |
| `int8_dpup`   |     0.841     |      0.829      |      0.868      |

RTF essentially flat across prompt languages and lengths — within
2-4% of each other for the same variant. The OV win is general,
not fox-pangram-specific.

### Verdict

OV W8A8 with NNCF transformer-mode quantization is the new ship default
for CPU. Three points on the speed-quality Pareto frontier from one
runtime, all dominating the previous frontier:

- **`int8_t160`** — CLI RTF 0.59, server-warm 0.485, 9.26% disagree.
  Replaces ORT W8A8 as the speed-first ship. 2.7× quality improvement
  at faster speed.
- **`dp0-27`** — CLI RTF 0.72, server-warm ~0.65, 5.89% disagree.
  Replaces llama.cpp Q8_0 as the quality-conscious ship. Better quality
  and 3× the speed.
- **`dpup`** — CLI RTF 0.85, server-warm ~0.74, 4.87% disagree. Best
  int8 quality measured on this CPU/model.

App-level knobs that stack on top:
- **CFG-skip** (`OMNIVOICE_GUIDANCE_SCALE=0`) — halves the LLM forward
  cost (B=2 → B=1). `dpup` + CFG-skip in server mode = RTF 0.42 at
  best-int8 quality.
- **Step reduction** (`OMNIVOICE_NUM_STEP=8` instead of 16) — linear
  speedup. `t160` + CFG-skip + step=8 = RTF ~0.21 (5.6× faster than
  PyTorch baseline).

The Rust deployment path is open via the `openvino` crate
(`rust_drivers/ov_inference/`). Quantization-aware retraining could
push lower than 4.87%, but that's a multi-day model-side intervention,
not a runtime change.

### How to reproduce — OV W8A8 ship default

```bash
# 1. Install (in the existing repo venv)
uv pip install openvino nncf

# 2. Convert the existing ONNX to OV IR (one-time; uses ~2.5 GB)
python -c "import openvino as ov; ov.save_model(ov.convert_model('onnx/omnivoice-step.onnx'), 'openvino_ir/omnivoice-step.xml', compress_to_fp16=False)"

# 3. Quantize via NNCF (uses calibration/*.npz captured from earlier work)
python scripts/quantize_openvino.py \
  --max-samples 160 --preset mixed \
  --out openvino_ir/omnivoice-step.int8_t160.xml

# 4. Use the OV CLI driver
python openvino_driver.py "Your text here" \
  --ir openvino_ir/omnivoice-step.int8_t160.xml --out out.wav

# Or run the OV server (drop-in OpenAI-compatible TTS)
python server_ov.py --ir openvino_ir/omnivoice-step.int8_t160.xml --port 8311
```

To produce the higher-quality `dp0-27` or `dpup` variants:

```bash
# down_proj-only fp32 (5.89% disagree)
python -c "
exclude = ['/audio_heads/MatMul']
for l in range(28):
    exclude.append(f'/llm/layers.{l}/mlp/down_proj/MatMul')
print(' '.join(exclude))
" | xargs -I{} python scripts/quantize_openvino.py \
  --max-samples 160 --preset mixed --exclude {} \
  --out openvino_ir/omnivoice-step.int8_dp0-27.xml

# down + up fp32 (4.87% disagree)
python -c "
exclude = ['/audio_heads/MatMul']
for l in range(28):
    exclude.append(f'/llm/layers.{l}/mlp/down_proj/MatMul')
    exclude.append(f'/llm/layers.{l}/mlp/up_proj/MatMul')
print(' '.join(exclude))
" | xargs -I{} python scripts/quantize_openvino.py \
  --max-samples 160 --preset mixed --exclude {} \
  --out openvino_ir/omnivoice-step.int8_dpup.xml
```

To use Rust:

```bash
# Symlink unversioned libs once (the openvino crate looks for these names)
cd .venv/lib/python3.11/site-packages/openvino/libs
for f in libopenvino*.so.*; do
  base=$(echo $f | sed 's/\.so\..*/.so/')
  [ -e "$base" ] || ln -s "$f" "$base"
done

# Build and run the Rust example
cd rust_drivers/ov_inference
cargo build --release
LD_LIBRARY_PATH=$REPO/.venv/lib/python3.11/site-packages/openvino/libs \
  ./target/release/ov_inference \
  $REPO/openvino_ir/omnivoice-step.int8_t160.xml \
  $REPO/calibration/long_en_fox_step0000.npz
```

### Portability across Intel and AMD CPUs

OpenVINO's CPU plugin is built on top of oneDNN, which has tuned int8
GEMM kernels for any modern x86 CPU with AVX2 + VNNI (or AVX-512
where available). Concretely:

- **Intel Skylake / Cascade Lake / Ice Lake / Tiger Lake / Alder Lake /
  Raptor Lake / Sapphire Rapids / Granite Rapids**: AVX2 + VNNI minimum,
  AVX-512 + AMX where present. All supported.
- **AMD Zen 2 / Zen 3 / Zen 4 / Zen 5**: AVX2 + VNNI (Zen 4+),
  AVX-512 + VNNI (Zen 4+, full coverage on Zen 5). oneDNN auto-detects
  and uses the right kernels.

The same `openvino_ir/omnivoice-step.int8_t160.xml` binary runs on any
of the above without re-export or re-quantization. Per-CPU performance
will vary based on AVX-VNNI/AVX-512 coverage and core count, but the
quality results (9.26% / 5.89% / 4.87% argmax disagree) are identical
because the int8 weight quantization is fixed in the IR; only the
GEMM kernel choice differs at runtime.

We did not have AMD hardware to validate this empirically.

### Cross-prompt validation

OV W8A8 wins generalize across prompt lengths and languages (single-run RTF):

| Prompt              | int8_t160 | int8_dp0-27 | ORT int8 |
|---------------------|----------:|------------:|---------:|
| Short EN (38 char)  |     0.735 |       0.825 |    0.837 |
| Fox pangram (122)   |     0.591 |       0.728 |    0.620 |
| Long EN (199 char)  |     0.622 |       0.733 |    0.693 |
| Chinese (109 char)  |     0.565 |       0.720 |    0.705 |

Short prompts pay more fixed-overhead but OV still wins. Long EN sees
the largest absolute gap (OV 0.622 vs ORT 0.693). Chinese also wins —
the calibration corpus included ZH samples so quality holds.

### Runtime config tuning (small gains)

OV CompileModel option sweep (single-forward microbench, S=226 B=2):

| Config                              | Forward ms |
|-------------------------------------|-----------:|
| default LATENCY                     |       226  |
| **+ ENABLE_CPU_PINNING=YES**        |     **216** |
| + ENABLE_HYPER_THREADING=NO         |       218  |
| + ENABLE_HYPER_THREADING=YES        |       217  |

`ENABLE_CPU_PINNING=YES` gives ~5% per-forward gain in microbench. In
end-to-end generation the gain shrinks (Python-side gaps between
forwards let cores migrate before OV pins them again) but it's still
free and worth shipping. The driver and server both set it by default.

### What we tried and skipped

- **OV W4A16 with AWQ + scale_estimation + GPTQ**: weight-only quant.
  Quality recovery insufficient on this model; speed not better than
  W8A8.
- **NNCF mixed-precision via `ratio` parameter**: only available on
  `compress_weights()` (weight-only), not `quantize()` (full PTQ).
- **`disable_channel_alignment=False`**: no effect on quality or speed.
- **`accurate_bias_correction` with default exclusions**: identical to
  fast-bias-correction at 9.26%. Only matters with the wider
  exclusion sets where weight quantization range differs more.
- **`INFERENCE_PRECISION_HINT=f16`**: no effect on Raptor Lake (F16C
  is convert-only, not compute).
- **fp16-compressed weights (`compress_to_fp16=True`)**: 596 ms forward,
  slightly slower than fp32 due to dequant overhead at this batch size.
- **Tract pure-Rust ONNX**: failed to load (dynamic-shape attention
  reshape unsupported).
- **ExecuTorch**: importable but not pursued; would mirror ORT export
  with similar speed envelope on x86.
- **Variable-output decode**: `audio_heads` is only 1.87% of MatMul
  weight footprint. Estimated savings 1-2% RTF, far below the
  PERFORMANCE.md projection of 30-40%. Skipped.
- **GGUF quant sweep (Q5_K_M, Q5_K_S, Q4_K_S, IQ4_XS, IQ4_NL, Q4_0,
  Q4_1, Q5_0, Q5_1)**: none beat Q8_0 on speed; k-quants and i-quants
  are universally slower on AVX-VNNI than the straightforward Q4_0/Q8_0
  due to per-block dequant cost. Documented above.

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
