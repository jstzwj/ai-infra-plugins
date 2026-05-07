# Chapter 29: Inference Engine

## Source Files
- `sources/Megatron-LM/megatron/core/inference/` - Inference engine
- `sources/Megatron-LM/megatron/core/model_inference_wrappers/` - Model wrappers
- `sources/Megatron-LM/megatron/core/sampling/` - Sampling strategies
- `sources/Megatron-LM/tools/run_text_generation_server.py` - Generation server

## Overview

Megatron-LM provides a high-performance inference engine optimized for large transformer models, supporting both batched generation and interactive serving. The inference pipeline supports tensor parallelism, pipeline parallelism, and MoE models.

## Inference Architecture

```
┌──────────────────────────────────────┐
│         Text Generation Server       │
│  (HTTP API / gRPC)                   │
├──────────────────────────────────────┤
│       Sampling Layer                 │
│  (Top-K, Top-P, Temperature, Beam)   │
├──────────────────────────────────────┤
│    Inference Wrapper                 │
│  (Model management, batching)        │
├──────────────────────────────────────┤
│    Inference-Optimized Layers        │
│  (Fused kernels, FP8, CUDA graphs)   │
├──────────────────────────────────────┤
│    Model Parallel Communication      │
│  (TP, PP, EP, NVLS)                 │
└──────────────────────────────────────┘
```

## ModelInferenceWrapper

The `ModelInferenceWrapper` class manages model inference, including batching, KV cache management, and generation control.

### Key Features
- **Dynamic batching**: Process multiple sequences simultaneously
- **KV cache management**: Efficient key-value cache for autoregressive generation
- **Tensor parallel inference**: Distributed inference across GPUs
- **FP8 inference**: Reduced memory with FP8 parameters

### Inference-Optimized Transformer Implementation

Set `transformer_impl='inference_optimized'` for production inference:

```python
config = TransformerConfig(
    transformer_impl='inference_optimized',
    use_inference_optimized_layers=True,
    flash_decode=True,
    inference_fuse_tp_communication=True,
    ...
)
```

### Inference-Optimized Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_inference_optimized_layers` | bool | False | Use inference-optimized transformer layers |
| `flash_decode` | bool | False | Use optimized flash decoding kernel |
| `inference_fuse_tp_communication` | bool | False | Fused reduce-scatter-residual-norm-allgather kernel |
| `inference_disable_triton_nvls_kernels` | bool | False | Disable Triton NVLS kernels |
| `inference_grouped_gemm_backend` | str | 'vllm' | Grouped GEMM backend: flashinfer, torch, vllm |
| `inference_moe_disable_fused_quant_kernels` | bool | False | Disable fused quantization kernels |
| `inference_moe_token_dispatcher_type` | str | 'nvls' | MoE token dispatcher: nccl, nvls |
| `inference_rng_tracker` | bool | False | Separate RNG tracker for inference |
| `inference_sampling_seed` | int | 42 | Random seed for sampling |

## KV Cache

### Cache Management
```python
# KV cache is managed automatically by the inference wrapper
# For MLA (Multi-Latent Attention), low-dimensional latents can be cached:
config = MLATransformerConfig(
    cache_mla_latents=True,  # Cache compressed KV representations
    ...
)
```

### Cache Types
- **Standard KV Cache**: Full key-value pairs per attention head
- **MLA Latent Cache**: Compressed low-dimensional representations (requires Flash MLA)

## Sampling Strategies

### Supported Sampling Methods

| Method | Description | Parameters |
|---|---|---|
| Greedy | Select token with highest probability | None |
| Top-K | Sample from top K tokens | `top_k` |
| Top-P (Nucleus) | Sample from tokens with cumulative probability <= P | `top_p` |
| Temperature | Scale logits before sampling | `temperature` |
| Beam Search | Maintain multiple hypotheses | `num_beams` |
| Top-K + Top-P | Combined filtering | `top_k`, `top_p` |
| Repetition Penalty | Penalize repeated tokens | `repetition_penalty` |

### Sampling Configuration
```python
from megatron.core.sampling import SamplingParams

params = SamplingParams(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)
```

## Batched Generation

### Single-Request Generation
```bash
python tools/run_text_generation_server.py \
    --load /path/to/checkpoint \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --max-tokens 512 \
    --temperature 0.7 \
    --top-p 0.9
```

### Dynamic Batched Server
```bash
python tools/run_dynamic_text_generation_server.py \
    --load /path/to/checkpoint \
    --tensor-model-parallel-size 4 \
    --max-batch-size 32 \
    --max-tokens 512
```

## FP8 Inference

FP8 inference reduces memory and improves throughput on Hopper+ GPUs:

```python
config = TransformerConfig(
    transformer_impl='inference_optimized',
    fp8='e4m3',
    fp8_recipe='mxfp8',  # Best for inference on H100+
    fp8_param=True,       # Keep parameters in FP8
    ...
)
```

### MXFP8 Inference (Recommended for Hopper+)
```bash
--transformer-impl inference_optimized \
--fp8-format e4m3 \
--fp8-recipe mxfp8 \
--fp8-param-gather \
--inference-grouped-gemm-backend torch
```

## NVLS (NVLink Sharred Memory)

NVLS enables variable-count AllGather-V/ReduceScatter-V for efficient MoE inference:

```python
config = TransformerConfig(
    inference_moe_token_dispatcher_type='nvls',  # Default
    symmetric_ar_type='multimem_all_reduce',
    ...
)
```

**Requirements**: Hopper+ GPUs with NVLink and symmetric memory support.

## Performance Optimization

### CUDA Graphs for Inference
```python
config = TransformerConfig(
    cuda_graph_impl='local',
    cuda_graph_scope='full_iteration',
    cuda_graph_warmup_steps=3,
    ...
)
```

### Flash Decode
Flash decode optimizes the decode phase for autoregressive generation:
```bash
--flash-decode
```

### Communication Fusion
```bash
--inference-fuse-tp-communication  # Fused TP ops during inference
```

## Multi-Node Inference

```bash
# Launch across multiple nodes
torchrun --nproc_per_node=8 --nnodes=2 \
    --master_addr=MASTER_IP --master_port=6000 \
    tools/run_text_generation_server.py \
    --load /path/to/checkpoint \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2
```

## Inference vs Training Configuration

| Aspect | Training | Inference |
|---|---|---|
| transformer_impl | transformer_engine | inference_optimized |
| CUDA Graphs | Partial/full iteration | Full iteration recommended |
| Precision | BF16/FP8 | FP8 (MXFP8 preferred) |
| Dropout | Enabled | Disabled (0.0) |
| Activation Recompute | Optional | Disabled |
| Gradient | Required | Disabled |
| KV Cache | N/A | Required |
