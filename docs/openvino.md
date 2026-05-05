# CPU inference with OpenVINO

OmniVoice's diffusion-LM step (LLM forward + audio_heads) is the dominant
matmul cost on CPU. The optional OpenVINO backend exports that subgraph to
OpenVINO IR, optionally quantizes it to int8 with NNCF, and runs it through
OpenVINO's CPU plugin (oneDNN AVX-VNNI int8 GEMM) while keeping the rest of
the pipeline (tokenization, mask construction, Higgs audio decode) in
PyTorch.

On AVX-VNNI Intel CPUs (e.g. Raptor Lake, Sapphire Rapids) the int8 path
roughly halves inference latency at the cost of small, mostly inaudible
quantization artifacts. On AMD CPUs and on older Intel CPUs without VNNI
the speedup is smaller.

## Install

The OpenVINO dependencies are an optional extra (`onnx`, `openvino`, `nncf`):

```bash
pip install omnivoice[openvino]
# or
uv sync --extra openvino
```

For an editable install from a clone, use `pip install -e '.[openvino]'`.

## End-to-end pipeline

The path is three offline steps + one runtime CLI. Each step uses a separate
script under `omnivoice.scripts`; the runtime CLI is `omnivoice-infer-openvino`.

### 1. Export PyTorch to ONNX

```bash
python -m omnivoice.scripts.export_onnx \
    --model k2-fsa/OmniVoice \
    --output onnx/omnivoice-step.onnx
```

Produces `onnx/omnivoice-step.onnx` plus a sibling `omnivoice-step.onnx_data`
file holding all tensors over 1 KB (external-data format; required because
the graph is several gigabytes).

### 2. Capture calibration data

NNCF's int8 PTQ needs a few hundred samples of representative activations.
The capture script runs ten short/long, English/Chinese prompts through a
regular `model.generate()` and dumps every diffusion-step input to disk:

```bash
python -m omnivoice.scripts.capture_calibration \
    --model k2-fsa/OmniVoice \
    --output_dir calibration/
```

10 prompts × 16 diffusion steps = 160 `.npz` files in `calibration/`. About
30 MB total, regenerated in a few minutes on CPU.

### 3. Convert to OpenVINO IR and quantize to int8

```bash
python -m omnivoice.scripts.quantize_openvino \
    --onnx onnx/omnivoice-step.onnx \
    --calibration_dir calibration/ \
    --output openvino_ir/omnivoice-step-int8.xml
```

The script does both `ov.convert_model` (ONNX → fp32 IR) and
`nncf.quantize` (fp32 IR → int8 IR with SmoothQuant + per-channel weight
quantization) in one pass. The output `.xml` plus its sidecar `.bin` file
make up the OpenVINO IR.

The default ignored scope leaves embedding `Gather` ops and the
`audio_heads` MatMul in fp32; both empirically degrade quality
disproportionately when quantized. Override with `--exclude` /
`--exclude_types` for experimentation.

### 4. Run inference

The runtime CLI is a drop-in mirror of `omnivoice-infer` with the same
generation flags, plus `--ir` for the quantized OpenVINO IR:

```bash
omnivoice-infer-openvino --model k2-fsa/OmniVoice \
    --ir openvino_ir/omnivoice-step-int8.xml \
    --text "Hello, this is a text for text-to-speech." \
    --instruct "Female, Middle-aged, Moderate Pitch" \
    --output out.wav
```

For voice cloning pass `--ref_audio` / `--ref_text` instead of `--instruct`,
exactly as you would with `omnivoice-infer`.

## Skipping quantization

If you only want to experiment with the OpenVINO runtime at fp32 (no NNCF
quantization), point `quantize_openvino.py` at the ONNX with an empty
calibration set, or write the fp32 IR directly:

```python
import openvino as ov
ov.save_model(
    ov.convert_model("onnx/omnivoice-step.onnx"),
    "openvino_ir/omnivoice-step.xml",
    compress_to_fp16=False,
)
```

Note that fp32 OpenVINO is roughly at parity with PyTorch fp32 on CPU; the
speedup comes almost entirely from int8 quantization.

## Tuning

- `--threads` selects the oneDNN thread count. Match it to the number of
  physical (P-core) cores; SMT siblings rarely help.
- `--static_reshape` (default on) recompiles the model to fixed shapes on
  the first forward call. It pays a one-time recompile (~1-2 s) for ~10%
  per-step speedup and is worth it for any non-trivial generation. Disable
  it if you serve many requests of varying lengths in the same process.
- `--max_samples` in the quantize script trades calibration time for
  quality. 160 samples (the default, one full corpus pass) is the
  empirically smallest set that gets within 10% argmax disagreement on the
  reference fp32 model. Going below 64 samples noticeably regresses
  quality.
- `--smooth_quant_alpha` overrides NNCF's default SmoothQuant alpha (0.95).
  Lower alpha shifts more outlier suppression into weights; higher alpha
  leaves activations less smoothed. The default is generally fine.
