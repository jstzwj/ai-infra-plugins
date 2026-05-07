# SGLang Quantization Reference

This document provides a comprehensive reference for all quantization methods, backends, and configurations supported by SGLang. Quantization reduces model memory footprint and can improve inference throughput by storing weights and/or activations in lower-precision formats.

## Table of Contents

- [Overview](#overview)
- [Online vs Offline Quantization](#online-vs-offline-quantization)
- [All Quantization Methods](#all-quantization-methods)
- [Platform Compatibility](#platform-compatibility)
- [GEMM Backends for FP8 Quantization](#gemm-backends-for-fp8-quantization)
- [GEMM Backends for FP4 Quantization](#gemm-backends-for-fp4-quantization)
- [MoE Runner Backends](#moe-runner-backends)
- [KV Cache Quantization](#kv-cache-quantization)
- [torchao Integration](#torchao-integration)
- [Platform-Specific Quantization](#platform-specific-quantization)
- [Offline Quantization Tools](#offline-quantization-tools)
- [Online Quantization Usage](#online-quantization-usage)
- [Configuration Reference](#configuration-reference)

---

## Overview

SGLang supports a wide range of quantization methods for LLM inference, spanning NVIDIA GPUs, AMD GPUs (ROCm), and Ascend NPUs. These methods fall into two broad categories:

- **Offline quantization**: The model weights are quantized ahead of time (pre-quantized). The quantized checkpoint is loaded directly at inference time. This is the recommended approach for best performance, usability, and convenience.
- **Online quantization**: The engine dynamically computes scaling parameters at startup, converting high-precision weights to lower-precision formats. This is convenient but adds startup overhead.

**Important**: If you use a pre-quantized model, do NOT add `--quantization` to enable online quantization at the same time. The quantization method will be automatically parsed from the model's Hugging Face config.

---

## Online vs Offline Quantization

### Offline Quantization (Recommended)

Offline quantization loads pre-quantized model weights directly during inference. Methods such as GPTQ and AWQ require this approach because they collect and pre-compute statistics from the original weights using a calibration dataset.

To load an already-quantized model, simply point to the model path. The engine detects the quantization format from the model's config:

```bash
python3 -m sglang.launch_server \
    --model-path hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --port 30000 --host 0.0.0.0
```

### Online Quantization

Online quantization dynamically computes scaling parameters during runtime, similar to NVIDIA FP8 training's delayed scaling mechanism. The engine calculates appropriate scaling factors on-the-fly to convert high-precision weights into a lower-precision format.

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

**Note**: Online quantization increases startup time and may use additional VRAM during initialization.

---

## All Quantization Methods

SGLang supports the following quantization methods (passed via `--quantization`):

| Method | Type | Description |
|--------|------|-------------|
| `fp8` | Online | Dynamic FP8 quantization with delayed scaling. Computed on-the-fly from weight statistics. |
| `mxfp4` | Offline | MX (Microscaling) FP4 format for extreme compression. Requires CDNA3/CDNA4 with MXFP support on AMD. |
| `blockwise_int8` | Online/Offline | Triton-based blockwise INT8 quantization. Works on both NVIDIA and AMD. |
| `w8a8_int8` | Online/Offline | Per-channel INT8 weight with per-token dynamic INT8 activation quantization. Uses CUTLASS int8 kernel from sgl-kernel. |
| `w8a8_fp8` | Online/Offline | Per-channel FP8 weight with per-token dynamic FP8 activation quantization. Uses CUTLASS FP8 kernel from sgl-kernel. |
| `awq` | Offline | Activation-aware Weight Quantization. Uses calibration data to protect salient weights. |
| `gptq` | Offline | Generative Post-Training Quantization. Uses approximate second-order information for optimal quantization. |
| `marlin` | Offline | Marlin optimized kernel format for high-throughput 4-bit inference. |
| `gptq_marlin` | Offline | GPTQ format optimized with Marlin kernels for NVIDIA. |
| `awq_marlin` | Offline | AWQ format optimized with Marlin kernels for NVIDIA. |
| `bitsandbytes` | Offline | bitsandbytes 4-bit/8-bit quantization (NF4, FP4, INT8). |
| `gguf` | Offline | GGUF format (llama.cpp compatible). Supports various sub-formats (Q4_K_M, Q8_0, etc.). |
| `modelopt` | Offline | NVIDIA ModelOpt quantization (auto-detects FP8/FP4 from checkpoint). |
| `modelopt_fp8` | Offline | NVIDIA ModelOpt FP8 quantization. Requires Hopper (SM90) or higher. |
| `modelopt_fp4` | Offline | NVIDIA ModelOpt FP4 (NVFP4) quantization. Requires Blackwell (SM100) or higher. |
| `modelopt_mixed` | Offline | NVIDIA ModelOpt mixed-precision quantization. |
| `petit_nvfp4` | Offline | Enables NVFP4 on AMD ROCm via Petit kernel. Auto-selected when loading NVFP4 models on AMD. |
| `moe_wna16` | Offline | Weight-only quantization for MoE layers with 16-bit activation. |
| `qoq` | Offline | Quantization-Optimized Quantization for enhanced 4-bit inference. |
| `w4afp8` | Offline | 4-bit weight with FP8 activation format. |
| `auto-round` | Offline | Intel auto-round quantization. Platform-agnostic, supports various schemes (W4A16, W8A16, etc.). |
| `compressed-tensors` | Offline | NeuralMagic compressed-tensors format. Supports FP8, INT8, and mixed precision. |
| `modelslim` | Offline | MindStudio ModelSlim quantization for Ascend NPUs. Uses CANN kernels. |
| `quark` | Offline | AMD Quark quantization framework. Supports FP8, MXFP4, and INT4-FP8 formats. |
| `quark_int4fp8_moe` | Online | AMD-only online INT4-to-FP8 MoE quantization for CDNA3/CDNA4 architectures. |
| `unquant` | Special | Explicitly disable quantization (useful for draft models when target is quantized). |

---

## Platform Compatibility

| Method | NVIDIA GPUs | AMD GPUs (MI300X/MI325X/MI350X) | Ascend NPUs (A2/A3) | Notes |
|--------|-------------|----------------------------------|----------------------|-------|
| `fp8` | Yes | Yes | WIP | Aiter or Triton backend on AMD |
| `mxfp4` | Yes | Yes | WIP | Requires CDNA3/CDNA4 with MXFP support; uses Aiter |
| `blockwise_int8` | Yes | Yes | No | Triton-based, works on both platforms |
| `w8a8_int8` | Yes | Yes | No | |
| `w8a8_fp8` | Yes | Yes | No | Aiter or Triton FP8 on AMD |
| `awq` | Yes | Yes | Yes | Triton dequantize on AMD, CANN kernels on Ascend |
| `gptq` | Yes | Yes | Yes | Triton or vLLM kernels on AMD, CANN kernels on Ascend |
| `compressed-tensors` | Yes | Yes | Partial | Aiter paths for FP8/MoE on AMD; CANN on Ascend, FP8 not yet supported |
| `quark` | Yes | Yes | No | AMD Quark quantization; Aiter GEMM paths on AMD |
| `auto-round` | Yes | Yes | Partial | Platform-agnostic (Intel auto-round). CANN kernels on Ascend |
| `quark_int4fp8_moe` | No | Yes | No | AMD-only; online INT4-to-FP8 MoE quantization (CDNA3/CDNA4) |
| `awq_marlin` | Yes | No | No | Marlin kernels are CUDA-only |
| `gptq_marlin` | Yes | No | No | Marlin kernels are CUDA-only |
| `gguf` | Yes | No | Yes | CUDA kernels in sgl-kernel; Ascend uses CPU pre-dequantization |
| `modelopt` / `modelopt_fp8` | Yes (Hopper/SM90+) | No | No | NVIDIA ModelOpt; requires NVIDIA hardware |
| `modelopt_fp4` | Yes (Blackwell/SM100+) | No | No | Native FP4 on Blackwell (B200, GB200) |
| `petit_nvfp4` | No | Yes (MI250/MI300X/MI325X) | No | Enables NVFP4 on ROCm via Petit |
| `bitsandbytes` | Yes | Experimental | No | Depends on bitsandbytes ROCm support |
| `torchao` (int4wo, etc.) | Yes | Partial | No | `int4wo` not supported on AMD |
| `modelslim` | No | No | Yes | Ascend quantization; uses CANN kernels |

On AMD, several methods use [Aiter](https://github.com/ROCm/aiter) for acceleration. Set `SGLANG_USE_AITER=1` where noted. On Ascend, see the Ascend NPU quantization documentation for detailed layer-level configurations.

---

## GEMM Backends for FP8 Quantization

Backend selection is supported for **blockwise FP8** GEMM. When running FP8 quantized models, select the GEMM backend via `--fp8-gemm-backend`.

| Backend | Hardware | Description |
|---------|----------|-------------|
| `auto` | All | Auto-selects based on hardware (default) |
| `deep_gemm` | SM90, SM100 | JIT-compiled DeepGEMM; enabled when DeepGEMM is installed |
| `flashinfer_trtllm` | SM100 | FlashInfer TensorRT-LLM backend; optimal for low-latency |
| `flashinfer_cutlass` | SM100/120 | FlashInfer CUTLASS groupwise FP8 GEMM |
| `flashinfer_deepgemm` | SM90 | Uses swapAB optimization for small M dimensions in decoding |
| `cutlass` | SM90, SM100/120 | sgl-kernel CUTLASS implementation |
| `triton` | All | Fallback; widely compatible |
| `aiter` | ROCm | AMD AITER backend |

**`auto` selection order:**
1. DeepGEMM (SM90/SM100, when installed)
2. FlashInfer TRTLLM (SM100, FlashInfer available)
3. CUTLASS (SM90/SM100/120)
4. AITER (AMD)
5. Triton (fallback)

**Exception**: SM120 always resolves to Triton.

---

## GEMM Backends for FP4 Quantization

Backend selection for NVFP4 GEMM is controlled via `--fp4-gemm-backend`.

| Backend | Hardware | Description |
|---------|----------|-------------|
| `auto` | SM100/120 | Auto-selects: `flashinfer_cudnn` on SM120; `flashinfer_cutlass` on SM100 |
| `cutlass` | SM100/120 | SGLang CUTLASS kernel |
| `flashinfer_cutlass` | SM100/120 | FlashInfer CUTLASS backend |
| `flashinfer_cudnn` | SM100/120 (CUDA 13+, cuDNN 9.15+) | FlashInfer cuDNN backend; used on SM120 for performance |
| `flashinfer_trtllm` | SM100 | FlashInfer TensorRT-LLM backend |

When FlashInfer is unavailable for NVFP4, the SGLang CUTLASS kernel is used as automatic fallback.

---

## MoE Runner Backends

For Mixture-of-Experts models with quantized weights, the MoE runner backend can be selected via `--moe-runner-backend`.

| Backend | Description |
|---------|-------------|
| `auto` | Auto-selects based on model architecture and hardware |
| `triton` | Triton-based MoE runner (widely compatible) |
| `flashinfer_trtllm` | FlashInfer with TensorRT-LLM MoE backend (SM100+) |
| `flashinfer_mxfp4` | FlashInfer MXFP4 MoE backend (SM100 with MXFP4 models) |
| `deep_gemm` | DeepGEMM MoE runner |
| `triton_kernel` | Triton kernel MoE backend (SM120 with MXFP4) |
| `aiter` | AMD AITER MoE backend |

The `auto` selection logic considers:
- SM100 + MXFP4 models -> `flashinfer_mxfp4`
- SM120 + MXFP4 models -> `triton_kernel`
- ROCm + MXFP4 -> `aiter`
- SM100 + modelopt_fp4 (DeepSeek, Kimi) -> `flashinfer_trtllm`
- Default fallback -> `triton`

---

## KV Cache Quantization

SGLang supports KV cache quantization to reduce memory usage for the key-value cache during inference. This is controlled via `--kv-cache-dtype`.

### Supported KV Cache Data Types

| Data Type | Description | Notes |
|-----------|-------------|-------|
| `auto` | Auto-select based on hardware and model | Default. Uses FP8 on supported hardware for DeepSeek DSA. |
| `fp8_e5m2` | FP8 with 5 exponent bits, 2 mantissa bits | Wider dynamic range, lower precision. |
| `fp8_e4m3` | FP8 with 4 exponent bits, 3 mantissa bits | Better precision, narrower range. Preferred for most use cases. |
| `fp4_e2m1` | FP4 with 2 exponent bits, 1 mantissa bit | Extreme compression. Supported by TensorRT-LLM MLA backend. |
| `bf16` / `bfloat16` | Bfloat16 (no quantization) | Full precision KV cache. |

### KV Cache Quantization with ModelOpt

When using NVIDIA ModelOpt, KV cache quantization can be specified during the offline quantization step via `--kv_cache_qformat` (default: `fp8`).

### KV Cache Quantization with DeepSeek DSA

For DeepSeek DSA (Dynamic Sparse Attention) models, the KV cache dtype is automatically configured:
- On SM100+ (Hopper/Blackwell): defaults to `fp8_e4m3`
- On other hardware: defaults to `bfloat16`

FlashAttention3 only supports `fp8_e4m3` for FP8 KV cache (not `fp8_e5m2`).

### Usage

```bash
# Use FP8 E4M3 KV cache
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --port 30000

# Use FP8 E5M2 KV cache
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8_e5m2 \
    --port 30000
```

---

## torchao Integration

SGLang supports quantization methods from [torchao](https://github.com/pytorch/ao) via the `--torchao-config` flag. This enables online quantization at startup.

### Supported torchao Configurations

| Config | Description |
|--------|-------------|
| `int8dq` | INT8 dynamic quantization (per-token) |
| `int8wo` | INT8 weight-only quantization |
| `fp8wo` | FP8 weight-only quantization |
| `fp8dq-per_tensor` | FP8 dynamic quantization (per-tensor) |
| `fp8dq-per_row` | FP8 dynamic quantization (per-row) |
| `int4wo-32` | INT4 weight-only quantization with group size 32 |
| `int4wo-64` | INT4 weight-only quantization with group size 64 |
| `int4wo-128` | INT4 weight-only quantization with group size 128 |
| `int4wo-256` | INT4 weight-only quantization with group size 256 |

### Usage

```bash
# INT4 weight-only with group size 128
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int4wo-128 \
    --port 30000
```

**Known Issue**: `int8dq` has bugs when used with CUDA graph capture. Disable CUDA graphs when using this method:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --torchao-config int8dq \
    --disable-cuda-graph \
    --port 30000
```

### Platform Support

- `int4wo-*`: Not supported on AMD GPUs
- Other methods may work on AMD with partial support

---

## Platform-Specific Quantization

### NVIDIA GPU Quantization

NVIDIA GPUs support the widest range of quantization methods:

- **Hopper (SM90, e.g., H100)**: FP8 (all backends), GPTQ, AWQ, Marlin, bitsandbytes, GGUF, ModelOpt FP8, torchao
- **Blackwell (SM100, e.g., B200, GB200)**: All Hopper methods + ModelOpt FP4 (NVFP4), MXFP4
- **Ampere (SM80, e.g., A100)**: GPTQ, AWQ, bitsandbytes, GGUF, INT8 methods
- **Ada Lovelace (SM89, e.g., L40, RTX 4090)**: Same as Ampere

The recommended approach for NVIDIA is to use pre-quantized models from [Unsloth](https://huggingface.co/unsloth), [NVIDIA ModelOpt](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer), or [NeuralMagic](https://huggingface.co/collections/neuralmagic).

### AMD GPU (ROCm) Quantization

AMD GPUs (MI300X, MI325X, MI350X) support:

- FP8 via Triton or Aiter backends
- GPTQ, AWQ via Triton/vLLM kernels
- `quark` and `quark_int4fp8_moe` (AMD-specific)
- `petit_nvfp4` for NVFP4 on ROCm
- MXFP4 via Aiter
- `compressed-tensors` with Aiter GEMM paths
- `auto-round` (platform-agnostic)

Set `SGLANG_USE_AITER=1` to enable Aiter acceleration where applicable.

### Ascend NPU Quantization

Ascend NPUs (A2, A3) support:

- AWQ via CANN kernels
- GPTQ via CANN kernels
- `compressed-tensors` (partial, FP8 not yet supported)
- `auto-round` via CANN kernels
- GGUF via CPU pre-dequantization at load time
- `modelslim` for Ascend-specific quantization (W4A4, W8A8, W8A8_DYNAMIC)

ModelSlim available methods for Ascend:
- W4A4_DYNAMIC (linear and MoE with online activation quantization)
- W8A8 (linear with offline activation quantization)
- W8A8_DYNAMIC (linear and MoE with online activation quantization)
- W4A8_DYNAMIC (MoE with online activation quantization)

---

## Offline Quantization Tools

### Unsloth (Recommended)

Unsloth provides easy-to-use quantization and export tools. See the [Unsloth SGLang Guide](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide).

### auto-round

Intel's auto-round supports multiple quantization schemes:

```python
from auto_round import AutoRound

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-autoround-4bit"

# Supported schemes: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4", "GGUF:Q4_K_M"
autoround = AutoRound(model_id, scheme="W4A16")
autoround.quantize_and_save(quant_path, format="auto_round")
```

For VLMs, use `AutoRoundMLLM` instead.

Command-line usage:
```bash
auto-round \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --bits 4 --group_size 128 \
    --format "auto_round" \
    --output_dir ./tmp_autoround
```

**Known limitations of auto-round**:
- Mixed-bit quantization is not fully supported due to layer fusion (e.g., QKV fusion)
- Quantized MoE models may have issues with kernel limitations
- Some VLMs have format-specific issues (e.g., Qwen2.5-VL-7B with `auto_gptq` format)

### GPTQModel

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)
model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save(quant_path)
```

### LLM Compressor

```python
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = SparseAutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)

model.save_pretrained(MODEL_ID.split("/")[1] + "-FP8-Dynamic")
tokenizer.save_pretrained(MODEL_ID.split("/")[1] + "-FP8-Dynamic")
```

### NVIDIA ModelOpt

NVIDIA ModelOpt provides advanced quantization for NVIDIA hardware with FP8 and FP4 support.

**Offline Quantization** (recommended):
```bash
python3 -m sglang.launch_server \
    --model-path nvidia/Llama-3.1-8B-Instruct-FP8 \
    --quantization modelopt_fp8 \
    --port 30000
```

**Online Quantization** (convenient but slow startup):
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --quantization modelopt_fp8 \
    --port 30000
```

**Creating quantized checkpoints** using the export workflow:
```bash
# FP8 quantization
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp8 \
    --quantization-method modelopt_fp8

# FP4 quantization (requires Blackwell GPU)
python examples/usage/modelopt_quantize_and_export.py quantize \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --export-dir ./quantized_tinyllama_fp4 \
    --quantization-method modelopt_fp4
```

### ModelSlim (Ascend)

```bash
# Install
git clone https://gitcode.com/Ascend/msmodelslim.git
cd msmodelslim && bash install.sh

# Quantize
msmodelslim quant \
    --model_path ${MODEL_PATH} \
    --save_path ${SAVE_PATH} \
    --device npu:0,1 \
    --model_type Qwen3-32B \
    --quant_type w8a8 \
    --trust_remote_code True

# Serve
python3 -m sglang.launch_server \
    --model-path $PWD/Qwen3-32B-w8a8 \
    --port 30000
```

---

## Online Quantization Usage

### FP8 Online Quantization

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --port 30000 --host 0.0.0.0
```

### W8A8 INT8 Online Quantization

For per-channel INT8 quantized models with per-token dynamic quantization activation:

```bash
python3 -m sglang.launch_server \
    --model-path neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic \
    --quantization w8a8_int8 \
    --port 30000 --host 0.0.0.0
```

### quark_int4fp8_moe (AMD Only)

For AMD GPUs (CDNA3/CDNA4), this method dynamically quantizes MoE layer weights to INT4 (upcasted to FP8 during inference) and other layers to FP8:

```bash
python3 -m sglang.launch_server \
    --model-path <your-model> \
    --quantization quark_int4fp8_moe \
    --port 30000
```

---

## Configuration Reference

### Server Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--quantization` | `str` | `None` | Quantization method. Choices: fp8, mxfp4, blockwise_int8, w8a8_int8, w8a8_fp8, awq, gptq, marlin, gptq_marlin, awq_marlin, bitsandbytes, gguf, modelopt, modelopt_fp8, modelopt_fp4, modelopt_mixed, petit_nvfp4, moe_wna16, qoq, w4afp8, mxfp4, auto-round, compressed-tensors, modelslim, quark, quark_int4fp8_moe, unquant |
| `--quantization-param-path` | `str` | `None` | Path to quantization parameters file |
| `--kv-cache-dtype` | `str` | `auto` | KV cache data type: auto, fp8_e5m2, fp8_e4m3, fp4_e2m1, bf16, bfloat16 |
| `--fp8-gemm-backend` | `str` | `auto` | FP8 GEMM backend: auto, deep_gemm, flashinfer_trtllm, flashinfer_cutlass, flashinfer_deepgemm, cutlass, triton, aiter |
| `--fp4-gemm-backend` | `str` | `auto` | FP4 GEMM backend: auto, cutlass, flashinfer_cutlass, flashinfer_cudnn, flashinfer_trtllm |
| `--moe-runner-backend` | `str` | `auto` | MoE runner backend: auto, triton, flashinfer_trtllm, flashinfer_mxfp4, deep_gemm, triton_kernel, aiter |
| `--torchao-config` | `str` | `""` | torchao quantization config: int8dq, int8wo, int4wo-32, int4wo-64, int4wo-128, int4wo-256, fp8wo, fp8dq-per_tensor, fp8dq-per_row |
| `--speculative-draft-model-quantization` | `str` | Same as target | Quantization for draft model. Use "unquant" to force no quantization |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_USE_AITER` | `0` | Enable AMD Aiter acceleration for quantized GEMM |
| `TORCHINDUCTOR_CACHE_DIR` | (none) | Cache directory for torch.compile auto-tuning results (useful for deployment) |

### Overriding Quantization for Specific Layers

For per-channel quantized models (INT8 or FP8 with per-token dynamic activation quantization), you can override the Hugging Face config's quantization settings by explicitly specifying `--quantization w8a8_int8` or `--quantization w8a8_fp8`. This invokes the corresponding CUTLASS kernel from sgl-kernel instead of the default CompressedTensors path.

---

## Pre-Quantized Model Collections

For validated pre-quantized models, see:

- [Unsloth on HuggingFace](https://huggingface.co/unsloth) - Wide variety of quantized models
- [NVIDIA ModelOpt Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer) - NVIDIA-optimized FP8/FP4 checkpoints
- [NeuralMagic Collection](https://huggingface.co/collections/neuralmagic) - NeuralMagic compressed-tensors models

**Important**: Quantized models must be validated via benchmarks post-quantization to guard against abnormal quantization loss regressions.

---

## References

- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [LLM Compressor](https://github.com/vllm-project/llm-compressor/)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [Petit: NVFP4 on ROCm](https://github.com/causalflow-ai/petit-kernel)
- [Torchao: PyTorch Architecture Optimization](https://github.com/pytorch/ao)
- [auto-round](https://github.com/intel/auto-round)
- [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
- [Aiter (AMD)](https://github.com/ROCm/aiter)
- [SGLang Quantization Source](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/quantization)
