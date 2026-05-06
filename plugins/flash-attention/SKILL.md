# FlashAttention - Complete Reference Manual

> FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
> Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
> Repository: https://github.com/Dao-AILab/flash-attention
> Current Version: 2.8.4 (FA2) / 4.x (FA4 CuTeDSL)

## Table of Contents

1. [Overview and Architecture](#1-overview-and-architecture)
2. [Installation and Setup](#2-installation-and-setup)
3. [Quick Start Guide](#3-quick-start-guide)
4. [API Reference - FlashAttention-2 (FA2)](#4-api-reference---flashattention-2-fa2)
5. [API Reference - FlashAttention-3 (FA3/Hopper)](#5-api-reference---flashattention-3-fa3hopper)
6. [API Reference - FlashAttention-4 (FA4/CuTeDSL)](#6-api-reference---flashattention-4-fa4cutedsl)
7. [Multi-Head Attention Module (MHA)](#7-multi-head-attention-module-mha)
8. [Model Implementations](#8-model-implementations)
9. [Operations Library (ops)](#9-operations-library-ops)
10. [Triton Kernels](#10-triton-kernels)
11. [Training Framework](#11-training-framework)
12. [Benchmarking Guide](#12-benchmarking-guide)
13. [Testing Guide](#13-testing-guide)
14. [CUDA Kernel Architecture](#14-cuda-kernel-architecture)
15. [GPU Architecture Specifics](#15-gpu-architecture-specifics)
16. [Advanced Topics](#16-advanced-topics)
17. [Troubleshooting and Debugging](#17-troubleshooting-and-debugging)
18. [Performance Optimization Guide](#18-performance-optimization-guide)
19. [Migration Guide](#19-migration-guide)
20. [Appendix](#20-appendix)

---

## Detailed Reference Files

The following reference files contain in-depth documentation for each component:

| File | Description |
|------|-------------|
| [references/01-overview-architecture.md](references/01-overview-architecture.md) | Project overview, architecture, and FlashAttention algorithm |
| [references/02-installation-setup.md](references/02-installation-setup.md) | Installation, dependencies, build from source |
| [references/03-fa2-api-reference.md](references/03-fa2-api-reference.md) | Complete FA2 API: flash_attn_func, varlen, kvcache, blocksparse |
| [references/04-fa3-hopper-api.md](references/04-fa3-hopper-api.md) | FA3 Hopper interface, FP8, TMA optimizations |
| [references/05-fa4-cutedsl-api.md](references/05-fa4-cutedsl-api.md) | FA4 CuTeDSL interface, score/mask modifiers, block sparse |
| [references/05a-fa4-forward-kernels.md](references/05a-fa4-forward-kernels.md) | FA4 forward kernel architecture (SM80/SM90/SM100/SM120) |
| [references/05b-fa4-backward-kernels.md](references/05b-fa4-backward-kernels.md) | FA4 backward kernel architecture |
| [references/05c-fa4-core-abstractions.md](references/05c-fa4-core-abstractions.md) | Softmax, mask, block_info, pipeline, tile_scheduler |
| [references/05d-fa4-arch-helpers.md](references/05d-fa4-arch-helpers.md) | Ampere/Blackwell helpers, MMA descriptors |
| [references/06-mha-module.md](references/06-mha-module.md) | Multi-Head Attention module (SelfAttention, CrossAttention, etc.) |
| [references/07-models.md](references/07-models.md) | BERT, GPT, LLaMA, GPT-NeoX, OPT, Falcon, ViT, etc. |
| [references/08-ops-library.md](references/08-ops-library.md) | Fused dense, layer norm, RMS norm, activations |
| [references/09-triton-kernels.md](references/09-triton-kernels.md) | Triton implementations: cross entropy, layer norm, linear, MLP, rotary |
| [references/10-training-framework.md](references/10-training-framework.md) | PyTorch Lightning training, configs, distributed training |
| [references/11-benchmarks.md](references/11-benchmarks.md) | Benchmarking tools and performance results |
| [references/12-testing.md](references/12-testing.md) | Test suite, two-pass testing, CUDA kernel tests |
| [references/13-cuda-kernels.md](references/13-cuda-kernels.md) | CUDA/C++ kernel implementation details |
| [references/14-gpu-architecture.md](references/14-gpu-architecture.md) | Ampere, Hopper, Blackwell specifics, SM90/SM100 features |
| [references/15-advanced-topics.md](references/15-advanced-topics.md) | Paged KV cache, GQA/MQA, block sparse, softcapping |
| [references/16-troubleshooting.md](references/16-troubleshooting.md) | Debugging GPU kernels, common issues, race conditions |
| [references/17-performance-guide.md](references/17-performance-guide.md) | Performance tuning, block size selection, memory optimization |
| [references/18-migration-guide.md](references/18-migration-guide.md) | Migrating between FA versions, API changes |
| [references/19-utils-and-helpers.md](references/19-utils-and-helpers.md) | Utility functions, pretrained model loading, distributed utils |
| [references/20-appendix.md](references/20-appendix.md) | Papers, citations, glossary, environment variables |

---

## 1. Overview and Architecture

### What is FlashAttention?

FlashAttention is an IO-aware exact attention algorithm that computes attention with **O(N) memory** instead of O(N^2), while being **2-4x faster** than standard PyTorch attention. It achieves this by:

1. **Tiling**: Processing attention in blocks that fit in GPU SRAM
2. **Recomputation**: Recomputing attention in backward pass instead of storing the N^2 attention matrix
3. **Kernel Fusion**: Fusing softmax, dropout, and attention computation into a single GPU kernel

### Version History

| Version | Status | GPU Support | Implementation | Location |
|---------|--------|-------------|----------------|----------|
| FA1 | Legacy | Ampere | CUDA/CUTLASS | (deprecated) |
| FA2 | Stable | Ampere, Ada, Hopper | CUDA/CUTLASS | `csrc/`, `flash_attn/` |
| FA3 | Beta | Hopper (H100) | CUDA | `hopper/` |
| FA4 | Active | Hopper, Blackwell | CuTeDSL (Python) | `flash_attn/cute/` |

### Algorithm Overview

The FlashAttention algorithm computes:

```
O = softmax(Q @ K^T * scale) @ V
```

Using an **online softmax** approach that processes K/V in blocks:

```python
# Pseudocode for forward pass
for each Q block (m_block):
    O_block = 0, l_block = 0, m_block = -inf
    for each K/V block (n_block):
        S_block = Q_block @ K_block^T * scale
        m_new = max(m_block, max(S_block))
        P_block = exp(S_block - m_new)
        l_new = exp(m_block - m_new) * l_block + sum(P_block)
        O_block = exp(m_block - m_new) * O_block + P_block @ V_block
        m_block = m_new
        l_block = l_new
    O_block = O_block / l_block
```

### Key Innovations

- **IO-Awareness**: Minimizes HBM (GPU DRAM) reads/writes by keeping intermediate results in SRAM
- **Online Softmax**: Numerically stable softmax computation without materializing the full attention matrix
- **Work Partitioning**: Different parallelization strategies for different GPU architectures
- **Memory-Efficient Backward**: Recomputes attention in backward pass (saves O(N^2) memory)

### Repository Structure

```
flash-attention/
├── flash_attn/              # Main Python package
│   ├── cute/                # FA4 CuTeDSL kernels (active development)
│   ├── modules/             # MHA, MLP, Block, Embedding modules
│   ├── models/              # BERT, GPT, LLaMA, OPT, Falcon, ViT, etc.
│   ├── ops/                 # Fused operations (dense, norm, activations)
│   │   └── triton/          # Triton kernel implementations
│   ├── layers/              # Rotary embedding, patch embedding
│   ├── losses/              # Cross-entropy loss
│   └── utils/               # Benchmarking, distributed, generation utils
├── hopper/                  # FA3 Hopper-specific kernels
├── csrc/                    # FA2 CUDA/C++ kernels
│   ├── flash_attn/          # Core attention CUDA kernels
│   ├── layer_norm/          # Layer normalization CUDA kernels
│   └── fused_dense_lib/     # Fused dense operation CUDA kernels
├── tests/                   # Comprehensive test suite
├── benchmarks/              # Performance benchmarks
├── training/                # Training framework (PyTorch Lightning)
├── examples/                # Usage examples
└── tools/                   # CI/CD and development tools
```

---

## 2. Installation and Setup

### Requirements

- **CUDA**: 12.0+ (CUDA 12.8+ recommended for FA3/FA4)
- **PyTorch**: 2.2+
- **GPU**: Ampere (SM80+), Ada (SM89), Hopper (SM90), Blackwell (SM100+)
- **Python packages**: `packaging`, `psutil`, `ninja`
- **OS**: Linux (Windows experimental since v2.3.2)

### Installation Methods

#### FlashAttention-2 (Stable)
```bash
pip install flash-attn --no-build-isolation
```

#### FlashAttention-4 (CuTeDSL, Active Development)
```bash
pip install flash-attn-4
# With CUDA 13 optimizations:
pip install "flash-attn-4[cu13]"
# Development install:
pip install -e "flash_attn/cute[dev]"
```

#### FlashAttention-3 (Hopper Beta)
```bash
cd hopper
python setup.py install
```

#### Build from Source
```bash
python setup.py install
# Limit parallel compilation jobs (for machines with <96GB RAM):
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### AMD ROCm Installation
```bash
# With composable_kernel backend (default):
pip install flash-attn --no-build-isolation

# With Triton backend:
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.2 | Core framework |
| `ninja` | latest | Parallel compilation |
| `packaging` | latest | Version handling |
| `psutil` | latest | System utilities |
| `nvidia-cutlass-dsl` | >=4.4.1 | CuTeDSL (FA4 only) |
| `einops` | latest | Tensor operations (FA4) |
| `apache-tvm-ffi` | latest | TVM FFI (FA4) |

---

## 3. Quick Start Guide

### Basic Usage (FA2)

```python
import torch
from flash_attn import flash_attn_func

# Create input tensors: (batch, seqlen, heads, head_dim)
q = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.float16)

# Standard attention
out = flash_attn_func(q, k, v)
# out.shape = (2, 512, 8, 64)

# Causal attention (for autoregressive models)
out = flash_attn_func(q, k, v, causal=True)

# With sliding window attention
out = flash_attn_func(q, k, v, window_size=(128, 0), causal=True)

# With softcapping (Gemma-2, Grok style)
out = flash_attn_func(q, k, v, softcap=50.0)
```

### FA4 CuTeDSL Usage

```python
from flash_attn.cute import flash_attn_func

q = torch.randn(2, 512, 8, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(2, 512, 8, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(2, 512, 8, 128, device='cuda', dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
```

### Variable-Length Sequences

```python
from flash_attn import flash_attn_varlen_func

# Concatenated sequences with cumulative length indices
# cu_seqlens: [0, len1, len1+len2, ..., total_length]
cu_seqlens = torch.tensor([0, 128, 256, 512], device='cuda', dtype=torch.int32)

# q, k, v: (total_length, heads, head_dim)
q = torch.randn(512, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(512, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(512, 8, 64, device='cuda', dtype=torch.float16)

out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 128, 128)
```

### KV Cache for Inference

```python
from flash_attn import flash_attn_with_kvcache

# Pre-allocate KV cache
batch_size, max_seqlen, nheads_k, headdim = 1, 2048, 8, 64
k_cache = torch.zeros(batch_size, max_seqlen, nheads_k, headdim, device='cuda', dtype=torch.float16)
v_cache = torch.zeros(batch_size, max_seqlen, nheads_k, headdim, device='cuda', dtype=torch.float16)

# Step 1: Prefill (seqlen_q > 1)
q = torch.randn(1, 512, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 512, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 512, 8, 64, device='cuda', dtype=torch.float16)
cache_seqlens = torch.tensor([0], device='cuda', dtype=torch.int32)
out = flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens)

# Step 2+: Decode (seqlen_q = 1)
q = torch.randn(1, 1, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 1, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 1, 8, 64, device='cuda', dtype=torch.float16)
cache_seqlens = torch.tensor([512], device='cuda', dtype=torch.int32)
out = flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens)
```

### Multi-Query / Grouped-Query Attention (MQA/GQA)

```python
# GQA: Q has 8 heads, K/V have 2 heads (each K head serves 4 Q heads)
q = torch.randn(2, 512, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 512, 2, 64, device='cuda', dtype=torch.float16)  # Fewer heads
v = torch.randn(2, 512, 2, 64, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v, causal=True)
```

### With Rotary Embedding and KV Cache

```python
# Rotary embedding applied in-place during KV cache update
rotary_cos = torch.randn(max_seqlen, 32, device='cuda', dtype=torch.float16)
rotary_sin = torch.randn(max_seqlen, 32, device='cuda', dtype=torch.float16)

out = flash_attn_with_kvcache(
    q, k_cache, v_cache, k, v,
    rotary_cos=rotary_cos,
    rotary_sin=rotary_sin,
    cache_seqlens=cache_seqlens,
    causal=True,
    rotary_interleaved=False,  # GPT-NeoX style
)
```

---

## 4. API Reference - FlashAttention-2 (FA2)

### Core Functions

#### `flash_attn_func(q, k, v, ...)`

Standard FlashAttention with separate Q, K, V tensors.

```python
def flash_attn_func(
    q: torch.Tensor,           # (batch, seqlen_q, heads, head_dim)
    k: torch.Tensor,           # (batch, seqlen_k, heads_k, head_dim)
    v: torch.Tensor,           # (batch, seqlen_k, heads_k, head_dim)
    dropout_p: float = 0.0,    # Dropout probability
    softmax_scale: float = None, # Default: 1/sqrt(head_dim)
    causal: bool = False,      # Causal mask (autoregressive)
    window_size: tuple = (-1, -1), # (left, right) sliding window
    softcap: float = 0.0,      # Attention softcapping (>0 to enable)
    alibi_slopes: torch.Tensor = None, # ALiBi bias slopes
    deterministic: bool = False, # Deterministic backward
    return_attn_probs: bool = False, # Return attention probs
) -> torch.Tensor:             # (batch, seqlen_q, heads, head_dim)
```

**Tensor Layout**: `(batch, seqlen, num_heads, head_dim)` with last dimension contiguous and 16-byte aligned.

**Head Dimension Support**: Up to 256 (backward >192 requires A100/H100, or consumer GPUs with no dropout as of v2.5.5).

**Data Types**: FP16, BF16 (BF16 requires Ampere+).

#### `flash_attn_qkvpacked_func(qkv, ...)`

Optimized for stacked QKV tensor. More efficient backward pass.

```python
def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,  # (batch, seqlen, 3, heads, head_dim)
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:  # (batch, seqlen, heads, head_dim)
```

#### `flash_attn_kvpacked_func(q, kv, ...)`

Optimized for stacked KV tensor. Supports MQA/GQA.

```python
def flash_attn_kvpacked_func(
    q: torch.Tensor,   # (batch, seqlen, heads, head_dim)
    kv: torch.Tensor,  # (batch, seqlen, 2, heads_k, head_dim)
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:  # (batch, seqlen, heads, head_dim)
```

### Variable-Length Functions

#### `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`

For batches with variable sequence lengths (concatenated along dim 0).

```python
def flash_attn_varlen_func(
    q: torch.Tensor,              # (total_q, heads, head_dim)
    k: torch.Tensor,              # (total_k, heads_k, head_dim)
    v: torch.Tensor,              # (total_k, heads_k, head_dim)
    cu_seqlens_q: torch.Tensor,   # (batch+1,), int32
    cu_seqlens_k: torch.Tensor,   # (batch+1,), int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor = None,
) -> torch.Tensor:  # (total_q, heads, head_dim)
```

#### `flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, ...)`

```python
def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,           # (total, 3, heads, head_dim)
    cu_seqlens: torch.Tensor,    # (batch+1,), int32
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:  # (total, heads, head_dim)
```

#### `flash_attn_varlen_kvpacked_func(q, kv, ...)`

```python
def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,              # (total_q, heads, head_dim)
    kv: torch.Tensor,             # (total_k, 2, heads_k, head_dim)
    cu_seqlens_q: torch.Tensor,   # (batch+1,), int32
    cu_seqlens_k: torch.Tensor,   # (batch+1,), int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:  # (total_q, heads, head_dim)
```

### KV Cache Function

#### `flash_attn_with_kvcache(q, k_cache, v_cache, ...)`

Inference-optimized attention with KV cache update and optional rotary embedding.

```python
def flash_attn_with_kvcache(
    q: torch.Tensor,                   # (batch, seqlen_q, heads, head_dim)
    k_cache: torch.Tensor,             # (batch, seqlen_cache, heads_k, head_dim)
                                       # OR (num_blocks, page_block_size, heads_k, head_dim) for paged
    v_cache: torch.Tensor,             # Same shape as k_cache
    k: torch.Tensor = None,            # (batch, seqlen_new, heads_k, head_dim) - new keys
    v: torch.Tensor = None,            # (batch, seqlen_new, heads_k, head_dim) - new values
    rotary_cos: torch.Tensor = None,   # (seqlen_ro, rotary_dim/2)
    rotary_sin: torch.Tensor = None,   # (seqlen_ro, rotary_dim/2)
    cache_seqlens: Union[int, torch.Tensor] = None,  # Current cache lengths
    cache_batch_idx: torch.Tensor = None,  # (batch,), int32
    cache_leftpad: torch.Tensor = None,   # (batch,), int32
    block_table: torch.Tensor = None,     # (batch, max_blocks_per_seq), int32 - for paged KV
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,  # True=GPT-J style, False=GPT-NeoX style
    alibi_slopes: torch.Tensor = None,
    num_splits: int = 0,              # Split-KV for decode (>1 for parallel KV loading)
    return_softmax_lse: bool = False,
) -> torch.Tensor:  # (batch, seqlen_q, heads, head_dim)
```

**Note**: No backward pass support.

**Paged KV Cache**: When `block_table` is provided, `k_cache`/`v_cache` use paged layout: `(num_blocks, page_block_size, heads_k, head_dim)` where `page_block_size` must be a multiple of 256.

### Causal Mask Behavior

When `seqlen_q != seqlen_k` and `causal=True`, the mask aligns to the **bottom-right corner**:

```
seqlen_q=2, seqlen_k=5:        seqlen_q=5, seqlen_k=2:
  1 1 1 1 0                      0 0
  1 1 1 1 1                      0 0
                                 0 0
                                 1 0
                                 1 1
```

### ALiBi Support

ALiBi (Attention with Linear Bias) adds position-dependent bias:
```python
# slopes: (nheads,) or (batch_size, nheads), fp32
# Bias added: -alibi_slope * |i + seqlen_k - seqlen_q - j|
alibi_slopes = torch.tensor([0.5, 0.25, 0.125, ...], device='cuda')
out = flash_attn_func(q, k, v, alibi_slopes=alibi_slopes)
```

### Softcapping

Used in Gemma-2 and Grok models:
```python
# score = softcap * tanh(score / softcap)
out = flash_attn_func(q, k, v, softcap=50.0)
```

### Sliding Window (Local) Attention

```python
# Query at position i attends to keys in [i-window_left, i+window_right]
out = flash_attn_func(q, k, v, window_size=(128, 0), causal=True)
# Used in Mistral-7B
```

### Internal Functions

#### Block Size Selection (`_get_block_size_n`)

```python
def _get_block_size_n(device, head_dim, is_dropout, is_causal) -> int
```

Returns the KV block size based on GPU architecture and parameters:
- `head_dim <= 32`: 128
- `head_dim <= 64`: 128 (64 with dropout)
- `head_dim <= 96`: 64
- `head_dim <= 128`: 64 (32 with dropout on SM86/89)
- `head_dim <= 256`: 64

#### Custom Op Registration

FA2 registers custom PyTorch operations for `torch.compile()` support (PyTorch >= 2.4):

- `flash_attn::_flash_attn_forward`
- `flash_attn::_flash_attn_varlen_forward`
- `flash_attn::_flash_attn_backward`
- `flash_attn::_flash_attn_varlen_backward`

---

## 5. API Reference - FlashAttention-3 (FA3/Hopper)

### Installation
```bash
cd hopper
python setup.py install
```

### Usage
```python
import flash_attn_interface
out = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
```

### Supported Features
- FP16/BF16 forward and backward
- FP8 forward
- CUDA >= 12.3 required (12.8 recommended)
- H100/H800 GPUs only

### Key Optimizations
- TMA (Tensor Memory Accelerator) for async memory loads
- Warpgroup MMA (Matrix Multiply-Accumulate)
- FP8 support (E4M3/E5M2)
- Softcapping support

---

## 6. API Reference - FlashAttention-4 (FA4/CuTeDSL)

### Installation
```bash
pip install flash-attn-4
# CUDA 13 optimizations:
pip install "flash-attn-4[cu13]"
```

### Usage
```python
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

# Standard attention
out = flash_attn_func(q, k, v, causal=True)

# Variable-length attention
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k)
```

### Advanced Features

#### Score Modifiers
User-defined `@cute.jit` callables injected at compile time:
```python
from flash_attn.cute.interface import flash_attn_func

# Custom score modification
def my_score_mod(score, batch, head, q_idx, kv_idx):
    return score * 0.5  # Custom scaling

out = flash_attn_func(q, k, v, score_mod=my_score_mod)
```

#### Mask Modifiers
```python
def my_mask_mod(batch, head, q_idx, kv_idx):
    return q_idx >= kv_idx  # Custom causal-like mask

out = flash_attn_func(q, k, v, mask_mod=my_mask_mod)
```

#### Block Sparse Attention
```python
out = flash_attn_func(q, k, v, block_sparse_tensors=sparse_tensors)
```

#### Pack GQA
```python
out = flash_attn_func(q, k, v, pack_gqa=True)
```

### Key Parameters

```python
def flash_attn_func(
    q, k, v,
    causal=False,
    window_size_left=None,     # Sliding window left
    window_size_right=None,    # Sliding window right
    softmax_scale=None,        # Default: 1/sqrt(head_dim)
    softcap=0.0,               # Softcapping
    score_mod=None,            # Custom score modifier
    mask_mod=None,             # Custom mask modifier
    block_sparse_tensors=None, # Block sparse attention
    num_splits=0,              # Split-KV parallelism
    pack_gqa=False,            # Pack GQA for efficiency
    m_block_size=None,         # Override M block size
    n_block_size=None,         # Override N block size
    num_threads=None,          # Override thread count
)
```

### Compilation and Caching

Kernels are JIT-compiled with multi-level caching:
- **In-memory LRU**: Fast cache within session
- **Disk cache**: Persistent at `/tmp/${USER}/flash_attention_cute_dsl_cache/`

Cache key includes: dtype, head_dim, causal, mask/score_mod hashes, architecture, block sizes.

```bash
# Enable disk cache
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 python my_script.py

# Dump CUBIN/SASS for inspection
CUTE_CUBIN_PATH=/tmp/cubin_output python my_script.py

# Keep PTX for inspection
CUTE_DSL_KEEP_PTX=1 python my_script.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | Enable persistent disk cache |
| `FLASH_ATTENTION_FAKE_TENSOR` | Use FakeTensorMode (no GPU allocation) |
| `CUTE_CUBIN_PATH` | Dump CUBIN/SASS to directory |
| `CUTE_DSL_KEEP_PTX` | Keep intermediate PTX files |
| `CUTE_DSL_PTXAS_PATH` | Custom ptxas binary path |
| `CUTE_DSL_LINEINFO` | Add line info for sanitizer source mapping |

---

## 7. Multi-Head Attention Module (MHA)

### Module Classes

The `flash_attn/modules/mha.py` provides ready-to-use attention modules that wrap the FlashAttention kernels:

```python
from flash_attn.modules.mha import MHA, SelfAttention, CrossAttention
```

#### `MHA` Class

The base multi-head attention class that handles QKV projections and output projection:

```python
class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,           # Embedding dimension
        num_heads: int,           # Number of attention heads
        num_heads_kv: int = None, # KV heads (for GQA/MQA, default=num_heads)
        qkv_proj_bias: bool = True,
        out_proj_bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        rotary_emb_dim: int = 0,       # Rotary embedding dimension
        rotary_emb_base: float = 10000.0,
        rotary_emb_scale_base: float = None,
        rotary_emb_interleaved: bool = False,
        use_alibi: bool = False,
        window_size: tuple = (-1, -1), # Sliding window
        fused_bias_fc: bool = False,   # Use fused dense for projections
        use_flash_attn: bool = True,   # Use FlashAttention kernels
        return_softmax: bool = False,
        checkpointing: bool = False,   # Gradient checkpointing
        device=None,
        dtype=None,
    )
```

#### `SelfAttention`

```python
class SelfAttention(MHA):
    """Self-attention with same Q, K, V source"""
    def forward(self, x, seqlen=None, cu_seqlens=None, max_seqlen=None,
                kv_cache=None, cache_seqlens=None, **kwargs)
```

#### `CrossAttention`

```python
class CrossAttention(MHA):
    """Cross-attention with different Q and K/V sources"""
    def forward(self, x, context=None, cu_seqlens=None, cu_seqlens_context=None,
                max_seqlen=None, max_seqlen_context=None, **kwargs)
```

### Example: Using MHA in a Transformer Block

```python
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
from flash_attn.modules.block import Block

# Single transformer block
block = Block(
    dim=768,
    mix_cls=MHA,        # Attention class
    mlp_cls=Mlp,        # MLP class
    num_heads=12,
    causal=True,
    dropout=0.1,
)
x = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)
out = block(x)
```

---

## 8. Model Implementations

### Supported Models

| Model | File | Description |
|-------|------|-------------|
| BERT | `models/bert.py` | Bidirectional encoder with FlashAttention |
| GPT | `models/gpt.py` | GPT-2/3 style decoder with full training |
| LLaMA | `models/llama.py` | LLaMA model with RMS norm, SwiGLU, rotary |
| GPT-NeoX | `models/gpt_neox.py` | GPT-NeoX (EleutherAI) |
| OPT | `models/opt.py` | Meta OPT model |
| Falcon | `models/falcon.py` | TII Falcon model |
| GPT-J | `models/gptj.py` | GPT-J model |
| ViT | `models/vit.py` | Vision Transformer |
| Baichuan | `models/baichuan.py` | Baichuan model |
| BigCode | `models/bigcode.py` | BigCode model |
| BTLM | `models/btlm.py` | BTLM model |

### Pretrained Model Loading

All models support loading pretrained weights:
```python
from flash_attn.models.gpt import GPTLMHeadModel

model = GPTLMHeadModel.from_pretrained('gpt2')
```

---

## 9. Operations Library (ops)

### Fused Dense Operations (`ops/fused_dense.py`)

Fused linear layers that combine matmul + bias + activation:

```python
from flash_attn.ops.fused_dense import FusedDense, FusedMLP

# Fused linear projection
linear = FusedDense(in_features, out_features, bias=True)
# Equivalent to: output = x @ weight.T + bias

# Fused MLP (2-layer with activation)
mlp = FusedMLP(in_features, hidden_features, out_features, activation='gelu')
```

### Layer Normalization (`ops/layer_norm.py`)

```python
from flash_attn.ops.layer_norm import layer_norm_fn, rms_norm_fn, DropoutAddLayerNorm

# Functional API
normalized = layer_norm_fn(x, weight, bias, residual=residual, eps=1e-5)

# Module API
norm = DropoutAddLayerNorm(hidden_size, dropout=0.1, prenorm=True)
output, residual = norm(x, residual)
```

### RMS Normalization (`ops/rms_norm.py`)

```python
from flash_attn.ops.rms_norm import RMSNorm, rms_norm_fn

norm = RMSNorm(hidden_size, eps=1e-5)
output = norm(x)
```

### Activations (`ops/activations.py`)

```python
from flash_attn.ops.activations import swiglu, swish, squared_relu
```

---

## 10. Triton Kernels

Triton implementations in `flash_attn/ops/triton/`:

### Cross Entropy Loss (`triton/cross_entropy.py`)
```python
from flash_attn.ops.triton.cross_entropy import CrossEntropyLoss
loss_fn = CrossEntropyLoss()
```

### Layer Norm (`triton/layer_norm.py`)
```python
from flash_attn.ops.triton.layer_norm import layer_norm_fn
```

### Linear (`triton/linear.py`)
```python
from flash_attn.ops.triton.linear import triton_linear
```

### MLP (`triton/mlp.py`)
```python
from flash_attn.ops.triton.mlp import triton_mlp
```

### Rotary Embedding (`triton/rotary.py`)
```python
from flash_attn.ops.triton.rotary import triton_rotary
```

---

## 11. Training Framework

### Overview

The training framework uses PyTorch Lightning for training GPT models:

```bash
cd training
python train.py experiment=owt  # OpenWebText
python train.py experiment=pile # The Pile
```

### Configuration Structure

```
training/configs/
├── callbacks/     # Callback configurations
├── datamodule/    # Data module configs
├── experiment/    # Experiment configs (owt, pile)
│   ├── owt/       # OpenWebText
│   └── pile/      # The Pile
├── logger/        # Logging configs
├── metrics/       # Metric configs
├── mode/          # Train/eval mode configs
├── model/         # Model configs
│   └── gpt2model/ # GPT-2 model configs
├── optimizer/     # Optimizer configs
├── scheduler/     # LR scheduler configs
├── task/          # Task configs
└── trainer/       # Trainer configs
```

### Performance

Up to **225 TFLOPs/sec** per A100, equivalent to 72% model FLOPs utilization without activation checkpointing.

---

## 12. Benchmarking Guide

### Running Benchmarks

```bash
# FA2 benchmarks
python benchmarks/benchmark_flash_attn.py

# FA3 Hopper benchmarks
python hopper/benchmark_attn.py
python hopper/benchmark_flash_attention_fp8.py
python hopper/benchmark_mla_decode.py
python hopper/benchmark_split_kv.py

# FA4 CuTeDSL benchmarks
python flash_attn/cute/benchmark.py
python flash_attn/cute/benchmark_flash_attention_fp8.py
```

### Benchmark Utilities

```python
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward

# Benchmark forward pass
time_ms = benchmark_forward(flash_attn_func, q, k, v, causal=True)

# Benchmark forward + backward
time_ms = benchmark_backward(flash_attn_func, q, k, v, causal=True)
```

### Performance Reference (A100 80GB)

| Head Dim | Seq Len | Speedup (FWD+BWD) | Memory Savings |
|----------|---------|-------------------|----------------|
| 64 | 512 | ~2x | ~5x |
| 64 | 2048 | ~3x | ~10x |
| 64 | 4096 | ~4x | ~20x |
| 128 | 8192 | ~3x | ~40x |

---

## 13. Testing Guide

### Running Tests

```bash
# FA2 tests
pytest -q -s tests/test_flash_attn.py

# FA3 tests
cd hopper && pytest -q -s test_flash_attn.py

# FA4 CuTeDSL tests
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py

# Specific test
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x
```

### Fast Two-Pass Testing (FA4)

Separates compilation (parallel, no GPU) from execution:

```bash
# Pass 1: Compile all kernels in parallel (no GPU memory)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: Run tests using cached compiled kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  pytest -x tests/cute/test_flash_attn.py
```

### Test Parametrization

Tests cover:
- **dtypes**: FP16, BF16
- **head dimensions**: 64, 96, 128, 256
- **sequence lengths**: 128, 256, 512, 1024, 2048
- **causal/non-causal**
- **MHA/GQA/MQA** (different head ratios)
- **with/without dropout**
- **with/without ALiBi**
- **with/without softcapping**

---

## 14. CUDA Kernel Architecture

### FA2 Kernels (`csrc/`)

```
csrc/flash_attn/src/
├── flash_api.cpp           # C++/CUDA API bindings
├── flash_fwd_kernel.h      # Forward kernel template
├── flash_bwd_kernel.h      # Backward kernel template
├── flash_fwd_launch_template.h  # Forward launch wrapper
├── flash_bwd_launch_template.h  # Backward launch wrapper
├── kernel_traits.h         # Kernel configuration traits
├── utils.h                 # CUDA utility functions
└── ... (architecture-specific instantiations)
```

### FA3 Hopper Kernels (`hopper/`)

```
hopper/
├── flash.h                     # Main header
├── flash_fwd_kernel_sm80.h     # Ampere forward
├── flash_fwd_kernel_sm90.h     # Hopper forward
├── flash_bwd_kernel_sm80.h     # Ampere backward
├── flash_bwd_kernel_sm90.h     # Hopper backward
├── flash_fwd_launch_template.h # Forward launch
├── flash_bwd_launch_template.h # Backward launch
├── block.h                     # Block-level operations
├── softmax.h                   # Softmax implementation
├── mask.h                      # Masking
├── rotary.h                    # Rotary embedding
├── tile_size.h                 # Tile size heuristics
├── heuristics.h                # Auto-tuning heuristics
├── static_switch.h             # Compile-time switches
├── seqlen.h                    # Sequence length handling
├── pack_gqa.h                  # GQA packing
├── paged_kv.h                  # Paged KV cache
└── utils.h                     # Utilities
```

### Kernel Execution Flow (Forward)

```
1. Load Q tile from HBM to SRAM (TMA on Hopper/Blackwell)
2. For each K/V block:
   a. Load K block to SRAM (pipelined)
   b. Compute S = Q @ K^T (MMA/warpgroup)
   c. Apply mask (causal, sliding window, ALiBi)
   d. Online softmax update (max, sum)
   e. Load V block to SRAM (pipelined)
   f. Accumulate O += softmax(S) @ V
3. Rescale O by final softmax normalization
4. Store O to HBM
5. Store softmax LSE (log-sum-exp) to HBM
```

---

## 15. GPU Architecture Specifics

### Ampere (SM80, A100)

- **Shared Memory**: 164 KB per SM
- **Tensor Cores**: mma.sync (FP16, BF16)
- **Async Copy**: cp.async for HBM->SRAM
- **Block sizes**: Head-dim-dependent (32-128)

### Hopper (SM90, H100)

- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated async memory copies
- **Warpgroup MMA**: 4 warps cooperate on 16x16x16 MMA
- **Dynamic Shared Memory**: Up to 227 KB
- **Cluster**: Thread block clusters for cross-SM coordination
- **FP8 Support**: E4M3 and E5M2 data types

### Blackwell (SM100, B200)

- **UMMA**: Universal MMA with new descriptor format
- **2CTA Instructions**: Two CTAs cooperate for head_dim=256
- **Persistent Kernels**: Long-running kernels for split-KV
- **Enhanced TMA**: More copy atom types
- **SM120 Support**: Next-gen architecture support

### Block Size Heuristics

```
head_dim  | SM80  | SM90  | SM100
----------|-------|-------|-------
64        | 128   | 128   | 128
96        | 64    | 64    | 64
128       | 64    | 64    | 128 (2CTA)
256       | 64    | 64    | 128 (2CTA)
```

---

## 16. Advanced Topics

### Paged KV Cache (PagedAttention)

```python
# Paged KV cache layout: (num_blocks, page_block_size, heads_k, head_dim)
# page_block_size must be multiple of 256

block_table = torch.randint(0, num_blocks, (batch, max_blocks_per_seq),
                            device='cuda', dtype=torch.int32)
out = flash_attn_with_kvcache(q, k_cache, v_cache, k, v,
                              block_table=block_table,
                              cache_seqlens=cache_seqlens)
```

### Block Sparse Attention

```python
from flash_attn import flash_blocksparse_attn_func

# Block sparse attention with custom sparsity pattern
sparsity_layout = ...  # (batch, heads, num_blocks_q, num_blocks_k)
out = flash_blocksparse_attn_func(q, k, v, sparsity_layout)
```

### MQA/GQA (Multi-Query / Grouped-Query Attention)

```python
# MQA: K/V have 1 head
k = torch.randn(batch, seqlen, 1, head_dim, device='cuda', dtype=torch.float16)
v = torch.randn(batch, seqlen, 1, head_dim, device='cuda', dtype=torch.float16)
q = torch.randn(batch, seqlen, num_heads, head_dim, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)

# GQA: K/V have fewer heads (must divide Q heads)
k = torch.randn(batch, seqlen, num_heads_kv, head_dim, ...)
v = torch.randn(batch, seqlen, num_heads_kv, head_dim, ...)
```

### torch.compile() Support

FA2 supports `torch.compile()` for PyTorch >= 2.4:
```python
@torch.compile
def compiled_attn(q, k, v):
    return flash_attn_func(q, k, v, causal=True)
```

---

## 17. Troubleshooting and Debugging

### Common Issues

1. **OOM during compilation**: Set `MAX_JOBS=4`
2. **Head dimension alignment**: head_dim must be divisible by 8
3. **Contiguous last dimension**: All tensors must have `stride(-1) == 1`
4. **16-byte alignment**: Last dimension must be 16-byte aligned

### GPU Kernel Debugging

```bash
# Race condition check
compute-sanitizer --tool=racecheck python my_script.py

# Memory check
compute-sanitizer --tool=memcheck python my_script.py

# PTX inspection
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_LINEINFO=1 python my_script.py
```

### Debugging Tools

- `cute.printf` with thread guards for targeted output
- `compute-sanitizer --tool=racecheck` (beware TMA false positives)
- `CUTE_DSL_KEEP_PTX=1` for PTX inspection
- `CUTE_DSL_LINEINFO=1` for sanitizer source mapping

---

## 18. Performance Optimization Guide

### Choosing Parameters

```python
# For training (long sequences):
out = flash_attn_func(q, k, v, causal=True)
# Memory: O(N * d) instead of O(N^2)

# For inference (decode):
out = flash_attn_with_kvcache(q, k_cache, v_cache, k, v,
                              num_splits=0)  # Auto heuristic

# For very long sequences:
out = flash_attn_func(q, k, v, causal=True, window_size=(4096, 0))
```

### GPU Selection

```bash
# Select specific GPU
CUDA_VISIBLE_DEVICES=0 python my_script.py

# Check GPU availability
nvidia-smi
```

### Block Size Tuning

Override default block sizes in FA4:
```python
out = flash_attn_func(q, k, v, m_block_size=128, n_block_size=64)
```

---

## 19. Migration Guide

### FA1 -> FA2

```python
# Old (FA1):
from flash_attn import flash_attn_unpadded_func

# New (FA2):
from flash_attn import flash_attn_varlen_func
```

### FA2 -> FA3

```python
# FA2:
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)

# FA3 (requires Hopper GPU):
import flash_attn_interface
out = flash_attn_interface.flash_attn_func(q, k, v, causal=True)
```

### FA2 -> FA4

```python
# FA2:
from flash_attn import flash_attn_func

# FA4:
from flash_attn.cute import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```

---

## 20. Appendix

### Papers

1. **FlashAttention**: [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
   - NeurIPS 2022
   - IO-aware exact attention

2. **FlashAttention-2**: [tridao.me/publications/flash2/flash2.pdf](https://tridao.me/publications/flash2/flash2.pdf)
   - ICLR 2024
   - Better parallelism and work partitioning

3. **FlashAttention-3**: [tridao.me/publications/flash3/flash3.pdf](https://tridao.me/publications/flash3/flash3.pdf)
   - Hopper GPU optimizations

### Citations

```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={NeurIPS},
  year={2022}
}
@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={ICLR},
  year={2024}
}
```

### Environment Variables Summary

| Variable | Description |
|----------|-------------|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | Use Triton backend for AMD |
| `FLASH_ATTENTION_FAKE_TENSOR` | Use FakeTensorMode (FA4) |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | Enable disk cache (FA4) |
| `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE` | Autotune Triton kernels (AMD) |
| `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` | Override Triton config (AMD) |
| `CUTE_CUBIN_PATH` | Dump CUBIN/SASS |
| `CUTE_DSL_KEEP_PTX` | Keep PTX files |
| `CUTE_DSL_PTXAS_PATH` | Custom ptxas binary |
| `CUTE_DSL_LINEINFO` | Add line info |
| `MAX_JOBS` | Limit parallel compilation |

### Glossary

| Term | Definition |
|------|-----------|
| HBM | High Bandwidth Memory (GPU DRAM) |
| SRAM | Static Random-Access Memory (GPU shared memory) |
| TMA | Tensor Memory Accelerator (Hopper+ feature) |
| MMA | Matrix Multiply-Accumulate (Tensor Core operation) |
| MQA | Multi-Query Attention (1 KV head) |
| GQA | Grouped-Query Attention (fewer KV heads) |
| CuTeDSL | CUTLASS DSL - Python-based CUDA kernel authoring |
| LSE | Log-Sum-Exp (softmax normalization factor) |
| Paged KV | Paged KV cache (vLLM-style memory management) |
| Softcapping | tanh-based attention score normalization |
| ALiBi | Attention with Linear Bias |
| varlen | Variable-length sequence batching |
| Split-KV | Parallel KV loading across thread blocks |
| 2CTA | Two-CTA cooperative attention (Blackwell) |

### Supported GPU Architectures

| Architecture | SM Version | GPUs | FA2 | FA3 | FA4 |
|-------------|-----------|------|-----|-----|-----|
| Ampere | SM80 | A100 | Yes | - | - |
| Ampere | SM86 | A10, A40 | Yes | - | Backward only |
| Ada | SM89 | RTX 4090, L40 | Yes | - | - |
| Hopper | SM90 | H100 | Yes | Yes | Yes |
| Blackwell | SM100 | B200 | - | - | Yes |
| Blackwell | SM110 | B100 | - | - | Yes |

### Supported Data Types

| Type | FA2 | FA3 | FA4 |
|------|-----|-----|-----|
| FP16 | Yes | Yes | Yes |
| BF16 | Yes (Ampere+) | Yes | Yes |
| FP8 E4M3 | - | Forward only | Yes |
| FP8 E5M2 | - | Forward only | Yes |
| FP32 | - | - | - |

### Head Dimension Support

| Head Dim | FA2 Forward | FA2 Backward |
|----------|------------|--------------|
| <= 128 | All GPUs | All GPUs |
| 192 | All GPUs | A100/H100 (or consumer w/o dropout since v2.5.5) |
| 256 | All GPUs | A100/H100 (or consumer w/o dropout since v2.5.5) |
