# SGLang Attention Backends and KV Cache Reference

This document provides a comprehensive reference for all attention backends, KV cache management, and related configuration options in SGLang. Selecting the optimal attention backend is crucial for maximizing performance.

## Table of Contents

- [Overview](#overview)
- [MHA Backends (Standard Attention)](#mha-backends-standard-attention)
- [MLA Backends (Multi-Head Latent Attention)](#mla-backends-multi-head-latent-attention)
- [GDN Attention Backends (Linear Attention)](#gdn-attention-backends-linear-attention)
- [DSA Attention Backend (NSA)](#dsa-attention-backend-nsa)
- [Hybrid Attention](#hybrid-attention)
- [Automatic Backend Selection](#automatic-backend-selection)
- [KV Cache Data Types](#kv-cache-data-types)
- [KV Cache Architecture](#kv-cache-architecture)
- [Page Size Configuration](#page-size-configuration)
- [RadixAttention and Tree Cache](#radixattention-and-tree-cache)
- [Launch Examples](#launch-examples)
- [Performance Tuning](#performance-tuning)
- [Adding New Backends](#adding-new-backends)

---

## Overview

SGLang supports a large variety of attention backends. Each has different performance characteristics, hardware requirements, and feature support. If `--attention-backend` is not specified, SGLang automatically selects the best backend based on hardware and model architecture.

The attention system is split into two primary categories:

1. **MHA (Multi-Head Attention)**: Standard attention used by most models (Llama, Qwen, Mistral, etc.)
2. **MLA (Multi-Head Latent Attention)**: Used by DeepSeek V2/V3/R1, Kimi K2, and similar models that project KV to a compressed latent representation

Additionally, specialized backends exist for:
- **GDN (Gated Delta Network)**: Linear attention for hybrid models (Qwen 3.5, Qwen 3 Next, Jet Nemotron)
- **DSA (DeepSeek Sparse Attention)**: Native sparse attention for DeepSeek V3.2
- **Dual Chunk FlashAttention**: For ultra-long context models

---

## MHA Backends (Standard Attention)

### Feature Matrix

| Backend | Page Size > 1 (native) | FP8 KV Cache | FP4 KV Cache | Spec topk=1 | Spec topk>1 | Sliding Window | MultiModal |
|---|---|---|---|---|---|---|---|
| **FlashInfer** | Yes | Yes | No | Yes | Yes | Yes | No |
| **FA3 (FlashAttention 3)** | Yes | Yes | No | Yes | Yes | Yes | Yes |
| **FA4 (FlashAttention 4)** | 128 | No | Yes | Yes | Yes | Yes | Yes |
| **Triton** | No | Yes | Yes | Yes | Yes | Yes | Yes |
| **Torch Native (SDPA)** | No | Yes | Yes | No | No | No | Yes |
| **FlexAttention** | No | No | Yes | No | No | No | No |
| **TRTLLM MHA** | 16, 32, or 64 | Yes | Yes | Yes | No | Yes | No |
| **Dual Chunk FlashAttention** | Yes | No | No | No | No | No | No |
| **AITER (ROCm)** | Yes | Yes | No | Yes | Yes | Yes | Yes |
| **Wave (ROCm)** | Yes | No | No | No | No | No | No |
| **Ascend (NPU)** | Yes | No | No | Yes | No | Yes | Yes |
| **Intel XPU** | Yes | No | No | No | No | Yes | No |
| **Intel AMX (CPU)** | No | No | No | No | No | No | No |

### Backend Details

#### FlashInfer

- **Best for**: Non-Hopper machines (A100, A40, etc.)
- **Native page sizes**: Any (flexible)
- **Strengths**: Broad feature support, spec decoding with topk > 1, sliding window
- **Limitations**: No multimodal attention support in the backend itself
- **Implementation**: `flashinfer_backend.py`

#### FA3 (FlashAttention 3)

- **Best for**: Hopper GPUs (H100, H200, H20)
- **Requires**: CUDA 12.3+
- **Strengths**: Full feature support including multimodal and FP8 KV cache
- **Implementation**: `flashattention_backend.py`

#### FA4 (FlashAttention 4)

- **Best for**: SM90 (Hopper) and SM100 (Blackwell)
- **Native page sizes**: MHA requires page_size=128; MLA supports page_size=1
- **Strengths**: FP4 KV cache support, full spec decoding support
- **Limitations**: On Hopper, decode speed decreases as sequence length grows due to lack of SplitKV support. At batch=1 vs FA3 on H100: ~-10% at 2K tokens, ~-18% at 4K, ~-31% at 8K, ~-49% at 16K. Larger batches reduce the gap. Blackwell is not affected.
- **Note**: On SM100, page_size=128 is auto-enforced. On SM90, users must set `--page-size 128` manually for MHA.

#### Triton

- **Best for**: General compatibility, bidirectional attention, CPU/ROCm/NPU platforms
- **Strengths**: Supports bidirectional attention (for Gemma 3 multimodal), FP4 KV cache, all spec decoding modes
- **Implementation**: `triton_backend.py`

#### TRTLLM MHA

- **Best for**: Blackwell GPUs (B200), SM90 and SM120 (H20, H200, 5090)
- **Native page sizes**: 16, 32, or 64
- **XQA backend**: Optimized for SM90/SM120; works best with page_size=64
- **Strengths**: Highly optimized for NVIDIA architectures
- **Implementation**: `trtllm_mha_backend.py`

#### Dual Chunk FlashAttention

- **Best for**: Ultra-long context models (e.g., Qwen2.5-14B-1M)
- **Use case**: When context exceeds typical limits
- **Implementation**: `dual_chunk_flashattention_backend.py`

---

## MLA Backends (Multi-Head Latent Attention)

### Feature Matrix

| Backend | Native Page Sizes | FP8 KV Cache | FP4 KV Cache | Chunked Prefix Cache | Spec topk=1 | Spec topk>1 |
|---|---|---|---|---|---|---|
| **FlashInfer MLA** | 1 | No | Yes | Yes | Yes | No |
| **FlashMLA** | 64 | Yes | Yes | Yes | Yes | No |
| **Cutlass MLA** | 128 | Yes | Yes | Yes | Yes | No |
| **TRTLLM MLA (Blackwell)** | 32 or 64 | Yes | Yes | Yes | Yes | No |
| **FA3 (FlashAttention 3)** | n/a | No | No | Yes | Yes | Partial (page_size=1 only) |
| **Triton** | n/a | No | No | No | Yes | Partial (page_size=1 only) |
| **FA4** | 1 | No | Yes | Yes | No | No |
| **Ascend MLA (NPU)** | 128 | No | No | No | No | No |

### MLA Page-Size Constraints

| Backend | Required Page Size |
|---|---|
| FlashInfer MLA | 1 |
| FlashMLA | 64 |
| Cutlass MLA | 128 |
| TRTLLM MLA | 32 or 64 |

---

## GDN Attention Backends (Linear Attention)

GDN (Gated Delta Network) is a linear attention mechanism with O(n) complexity, used in hybrid models that alternate GDN linear attention layers with standard full attention layers. GDN is NOT selected via `--attention-backend`; it is automatically activated when the model architecture requires it (e.g., Qwen 3.5, Qwen 3 Next, Jet Nemotron, Jet VLM).

GDN linear attention layers have their own kernel backends selected via `--linear-attn-backend` (default: triton). You can override per phase with `--linear-attn-decode-backend` and `--linear-attn-prefill-backend`.

### GDN Backend Matrix

| Backend | Decode | Prefill/Extend | Spec Decoding (Target Verify) |
|---|---|---|---|
| **Triton (CUDA)** | Yes | Yes | Yes |
| **Triton (AMD/ROCm)** | Yes | Yes | Yes |
| **Triton (NPU)** | Yes | Yes | No |
| **Triton (CPU)** | Yes | Yes | No |
| **CuTe DSL (CUDA only)** | Yes | No | No |

### Platform Constraints for Full-Attention Backend on Hybrid GDN Models

- **Blackwell (e.g., B200)**: `triton`, `trtllm_mha`, or `fa4` only
- **NPU (Ascend)**: `ascend` only
- **AMD (ROCm)**: `triton` recommended
- **Other CUDA (Hopper, Ampere, etc.)**: Auto-selection works; no special constraints

---

## DSA Attention Backend (NSA)

DSA (DeepSeek Sparse Attention) is a native sparse attention mechanism used by DeepSeek V3.2. It is activated automatically when the model architecture requires it and is selected via `--attention-backend nsa`.

The NSA backend dispatches to different sub-backends for prefill and decode phases. Override with `--nsa-prefill-backend` and `--nsa-decode-backend`.

### NSA Sub-backends

| Sub-backend | Prefill | Decode | Notes |
|---|---|---|---|
| **flashmla_sparse** | Yes | Yes | Default prefill on Hopper and Blackwell (bf16) |
| **flashmla_kv** | Yes | Yes | Default decode for FP8 on Blackwell with DP |
| **flashmla_auto** | Yes | No | Auto-selects flashmla_sparse or flashmla_kv based on kv_cache_dtype |
| **fa3** | Yes | Yes | Default decode on Hopper (bf16) |
| **trtllm** | Yes | Yes | Default decode on Blackwell (bf16); default for both on Blackwell without DP |
| **tilelang** | Yes | Yes | Default on AMD (ROCm) |
| **aiter** | Yes | Yes | AMD-specific kernel library |

---

## Hybrid Attention

Hybrid attention is an experimental feature that allows mixing attention backends for prefill and decode phases. This is useful when one backend excels at prefill and another excels at decode.

### Configuration

```bash
# Prefill with FA4, Decode with TRTLLM MLA (Blackwell)
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --tp 8 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --quantization modelopt_fp4 \
  --prefill-attention-backend fa4
```

- If only one of `--prefill-attention-backend` or `--decode-attention-backend` is set, the unspecified phase inherits `--attention-backend`.
- If both are specified and differ, SGLang automatically enables a hybrid wrapper.

### Speculative Decoding with Hybrid Attention

The backend used for draft decoding and target verification depends on `--speculative-attention-mode`:
- `--speculative-attention-mode decode` (recommended): draft/verify use the decode backend
- `--speculative-attention-mode prefill` (default): draft/verify use the prefill backend

**Constraints**:
- If any backend is `trtllm_mha`, speculative decoding supports only `--speculative-eagle-topk 1`
- For paged MHA backends with `--page-size > 1` and `--speculative-eagle-topk > 1`, only `flashinfer` is supported
- CUDA Graph: the decode backend is always captured; the prefill backend is captured only when `--speculative-attention-mode prefill`

---

## Automatic Backend Selection

### MHA Models (e.g., Llama, Qwen)

| Hardware | Default Backend |
|---|---|
| Hopper (H100, H200) | `fa3` if CUDA 12.3+ and model config supported |
| Blackwell (B200) | `trtllm_mha` (unless using spec decoding with topk > 1) |
| Other (Ampere, Ada, etc.) | `flashinfer` if available, otherwise `triton` |

### MLA Models (e.g., DeepSeek V3)

| Hardware | Default Backend |
|---|---|
| Hopper | `fa3` (requires CUDA 12.3+) |
| Blackwell | `flashinfer` (DeepSeek V3 models auto-select `trtllm_mla`) |
| Other | `triton` |

---

## KV Cache Data Types

### Supported Formats

#### auto (default)
Uses the model's native dtype (typically BF16 or FP16).

#### FP8 Formats

| Format | Exponent Bits | Mantissa Bits | Dynamic Range | Precision |
|---|---|---|---|---|
| **fp8_e5m2** | 5 | 2 | +/-57344.0 | Lower |
| **fp8_e4m3** | 4 | 3 | +/-240.0 | Higher |

#### FP4 Format (Experimental)

- **fp8_e2m1**: MXFP4 (Microscaling FP4) with 1 sign bit, 2 exponent bits, 1 mantissa bit
- Uses block-based microscaling (16 elements per block in SGLang's implementation)
- Scaling factors computed automatically on-the-fly; no pre-quantized models or external files required

### Enabling Quantized KV Cache

```bash
# FP8 E5M2
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-0528 \
  --kv-cache-dtype fp8_e5m2

# FP8 E4M3 (recommended for better accuracy)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-0528 \
  --kv-cache-dtype fp8_e4m3

# FP4 E2M1 (maximum memory savings, experimental)
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
  --kv-cache-dtype fp4_e2m1
```

### Scaling Factors for FP8

Scaling factors can be loaded from pre-quantized checkpoints (e.g., ModelOpt) that include `k_scale` and `v_scale` parameters, or provided via JSON:

```json
{
  "kv_cache": {
    "dtype": "float8_e4m3fn",
    "scaling_factor": {
      "0": {"0": 1.0, "1": 1.0}
    }
  }
}
```

Outer keys are tensor parallel ranks, inner keys are layer indices. If not provided, defaults to 1.0 (which may cause accuracy issues).

### Memory Savings

- **BF16 to FP4**: Approximately 3.56x more tokens (accounting for scaling factor overhead)
- **FP4 to FP8**: Approximately 1.78x more tokens
- FP4/FP8 require additional memory for block-based scaling factors

### Accuracy Impact

#### FP8 Accuracy
Typically introduces minimal accuracy degradation. E4M3 has better accuracy than E5M2.

#### FP4 Accuracy
Varies by model size and dataset complexity. Key observations:
- **Simple datasets (e.g., GSM8K)**: FP4 maintains accuracy close to FP8/BF16
- **Large models (200B+)**: Generally tolerate FP4 quantization better
- **Complex reasoning (e.g., AIME25)**: More pronounced accuracy drops, especially for smaller models

| Model | Dataset | KV16 | KV8 (FP8 E4M3) | KV4 (FP4 E2M1) |
|---|---|---|---|---|
| Qwen3-235B-A22B | gsm8k | 0.9168 | 0.9181 | 0.9186 |
| Qwen3-235B-A22B | aime25 | 0.7733 | 0.7333 | 0.6000 |
| DeepSeek-R1-0528 | gsm8k | 0.9157 | 0.9154 | 0.9124 |
| DeepSeek-R1-0528 | gpqa_diamond | 0.7707 | 0.7697 | 0.7273 |

---

## KV Cache Architecture

### Memory Pool

SGLang manages KV cache memory through a paged memory pool system. Key components:

- **MemoryPool**: Base memory pool implementation (`memory_pool.py`)
- **MemoryPoolHost**: Host-side memory management (`memory_pool_host.py`)
- **Allocator**: Token-level allocation (`allocator.py`)

### Cache Hierarchies

| Cache Type | File | Description |
|---|---|---|
| RadixCache | `radix_cache.py` | Default tree-structured prefix cache |
| ChunkCache | `chunk_cache.py` | Chunk-based cache for specific workloads |
| SWARadixCache | `swa_radix_cache.py` | Sliding window attention radix cache |
| SWAMemoryPool | `swa_memory_pool.py` | Memory pool for sliding window attention |
| HiRadixCache | `hiradix_cache.py` | Hierarchical radix cache |
| HiMambaRadixCache | `hi_mamba_radix_cache.py` | Hybrid Mamba + radix cache |
| MambaRadixCache | `mamba_radix_cache.py` | Mamba model radix cache |
| UnifiedRadixCache | `unified_radix_cache.py` | Unified cache for combined workloads |

---

## Page Size Configuration

Page size controls how many tokens are grouped into a KV cache block. This is a critical performance knob:

- **Smaller page sizes** (e.g., 1): Maximum prefix reuse (token-level matching), but lower attention kernel performance
- **Larger page sizes** (e.g., 64, 128): Better attention kernel performance, but reduced prefix cache granularity

For the prefix cache to take effect, the number of tokens must fill at least one complete page. For example, with `page_size=64`, a 32-token prompt cannot be cached (pages cannot be padded).

Many backends that do not natively support `page_size > 1` can emulate it at the wrapper layer by expanding page tables to per-token indices. The "native page size" column indicates true in-kernel paging.

Some backends require fixed native page sizes:
- TRTLLM MHA: 16, 32, or 64
- TRTLLM MLA: 32 or 64
- FlashMLA: 64
- Cutlass MLA: 128
- Ascend: 128

---

## RadixAttention and Tree Cache

SGLang's RadixAttention algorithm uses a radix tree to efficiently manage and reuse KV cache across requests. The radix tree stores prefixes of token sequences as shared nodes, enabling automatic prefix caching and reuse.

### How It Works

1. **Prefix Matching**: When a new request arrives, SGLang traverses the radix tree to find the longest matching prefix
2. **Cache Reuse**: Matched prefix KV cache pages are reused without recomputation
3. **Cache Insertion**: New token sequences are inserted into the tree for future reuse
4. **Eviction**: When memory is full, least-recently-used (LRU) pages are evicted based on the eviction policy

### Benefits

- Automatic system prompt sharing across requests
- Multi-turn conversation KV cache reuse
- Shared prefix caching for batched requests
- No user intervention required

### Configuration

- `--disable-radix-cache`: Disable prefix caching (useful for benchmarking)
- `--mem-fraction-static`: Fraction of GPU memory allocated to KV cache (default: varies)

---

## Launch Examples

### FlashInfer (Non-Hopper Default)

```bash
# MHA model
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flashinfer

# MLA model
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-V3 \
  --attention-backend flashinfer --trust-remote-code
```

### FA3 (Hopper Default)

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend fa3
```

### Triton

```bash
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend triton
```

### FlashMLA (MLA Models)

```bash
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla --trust-remote-code

# With FP8 KV cache
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-R1 \
  --attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 --trust-remote-code
```

### TRTLLM MLA (Blackwell)

```bash
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla --trust-remote-code

# With FP8 KV cache
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-R1 \
  --attention-backend trtllm_mla \
  --kv-cache-dtype fp8_e4m3 --trust-remote-code
```

### TRTLLM MHA (Blackwell)

```bash
python3 -m sglang.launch_server \
  --tp 4 --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --attention-backend trtllm_mha --trust-remote-code
```

### FA4 (MHA and MLA)

```bash
# MHA with page_size=128
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --attention-backend fa4 --page-size 128 --trust-remote-code

# MLA with FA4 prefill
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-R1 \
  --prefill-attention-backend fa4 --trust-remote-code
```

### Cutlass MLA

```bash
python3 -m sglang.launch_server \
  --tp 8 --model deepseek-ai/DeepSeek-R1 \
  --attention-backend cutlass_mla --trust-remote-code
```

### Platform-Specific Backends

```bash
# Ascend NPU
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend ascend

# Intel XPU
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend intel_xpu

# Wave (ROCm)
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend wave

# FlexAttention
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend flex_attention

# Dual Chunk FlashAttention (ultra-long context)
python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-14B-Instruct-1M \
  --attention-backend dual_chunk_flash_attn

# Torch Native
python3 -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --attention-backend torch_native
```

---

## Performance Tuning

### Backend Selection Guidelines

1. **For throughput-critical workloads**: Use FA3 on Hopper, TRTLLM MHA on Blackwell
2. **For memory-constrained scenarios**: Use FP8/F4 KV cache with a compatible backend
3. **For prefix-heavy workloads**: Use page_size=1 (FlashInfer MLA) for maximum cache reuse
4. **For spec decoding with topk > 1**: Ensure backend supports it (FlashInfer, FA3, FA4, Triton)
5. **For bidirectional attention** (Gemma 3): Use Triton backend with CUDA Graph disabled

### Speculative Decoding V2 (Spec V2)

Spec V2 uses overlap scheduling (`SGLANG_ENABLE_SPEC_V2=True`) that benefits various attention backends. Requires `--speculative-eagle-topk 1` and currently applies to EAGLE and EAGLE3.

**Verified backends**: TRTLLM MLA, TRTLLM MHA, FA3, Ascend (NPU), Triton

**Limited support**: FlashInfer can run under Spec V2, but its plan stream introduces a synchronization point that limits overlap benefits.

### Memory Optimization

- Use `--kv-cache-dtype fp8_e4m3` for 2x memory savings (minimal accuracy loss)
- Use `--kv-cache-dtype fp4_e2m1` for 3.56x memory savings (experimental, evaluate accuracy first)
- Adjust `--mem-fraction-static` to control KV cache memory allocation
- Use larger page sizes for better attention kernel performance when prefix reuse is not critical

### Important Warnings

- Quantized KV cache must be dequantized before use in attention operations. Performance can be extremely slow if dequantization is not fused with the attention kernel. Always verify backend compatibility.
- FA4 on Hopper: decode speed decreases as sequence length grows due to lack of SplitKV support.
- Not all backends are supported on all platforms and model architectures.

---

## Adding New Backends

To add a new attention backend, follow these steps (reference implementations: `triton_backend.py`, `flashattention_backend.py`):

### Step 1: Implement Forward Functions (without CUDA Graph)

- **forward_extend**: Used for prefill, prefill with KV cache, and target verification. Called once per layer.
- **forward_decode**: Used for normal decode and draft decode. Called once per layer.
- **init_forward_metadata**: Initialize class and common metadata shared by all layers. Calls the plan function for optimizations like split_kv. Called once per forward.

### Step 2: Implement CUDA Graph Functions (3 phases)

- **init_cuda_graph_state**: Called once during lifetime. Creates all common shared buffers.
- **init_forward_metadata_capture_cuda_graph**: Called before capturing a CUDA graph. Similar to init_forward_metadata but writes metadata to pre-defined buffers.
- **init_forward_metadata_replay_cuda_graph**: Called before replaying a CUDA graph. This function is in the critical path and needs to be fast.

### Registration

Register the backend using the `@register_attention_backend` decorator. Linear attention kernel backends (GDN, KDA) follow a different pattern: they implement `LinearAttnKernelBase` and are dispatched by `GDNKernelDispatcher` / `KDAKernelDispatcher`.

### Implementation Files

Key implementation directories:
- `python/sglang/srt/layers/attention/` - All attention backend implementations
- `python/sglang/srt/mem_cache/` - KV cache memory management
- `python/sglang/srt/layers/attention/base_attn_backend.py` - Base class
- `python/sglang/srt/layers/attention/attention_registry.py` - Backend registry
