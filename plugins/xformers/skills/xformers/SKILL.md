---
name: xformers
description: >
  Comprehensive reference documentation and skill for xFormers, Facebook Research's toolbox
  to accelerate research on Transformers. Use this skill whenever the user mentions xformers,
  memory_efficient_attention, FMHA, flash attention, SwiGLU, RMSNorm, RoPE, rope_padded,
  2:4 structured sparsity, sparsify24, sequence parallelism, fused all-gather/reduce-scatter,
  tiled matmul, block-sparse tensors, attention patterns, selective activation checkpointing,
  forward-backward overlap, tree attention, model parallel linear layers, xformers profiler,
  Triton kernels for transformers, CUTLASS attention, BlockDiagonalMask, LowerTriangularMask,
  merge_attentions, or xformers internals and build configuration.
version: 0.0.36
---

# xFormers - Toolbox to Accelerate Research on Transformers

## How to use this skill

Use this skill as a code-aware xFormers reference manual. When a user asks about xFormers usage, debugging,
performance, attention, sparsity, parallelism, or internals, first identify which layer they are working at:

1. **Memory-Efficient Attention (FMHA)**: `memory_efficient_attention`, Flash Attention (v2/v3), CUTLASS, CK (ROCm), split-K Triton, paged attention, merge_attentions, attention biases/masks.
2. **Activation Optimizations**: SwiGLU fused linear layers, RMSNorm, RoPE with padded KV-cache, tiled matmul to avoid wave quantization.
3. **2:4 Structured Sparsity**: `sparsify24`, `sparsify24_like`, `sparsify24_ste`, CUTLASS/cuSPARSELt backends, gradient modes (sparse/dense/STE).
4. **Sequence/Model Parallelism**: Fused all-gather + matmul, fused matmul + reduce-scatter, ColumnParallelLinear, RowParallelLinear, symmetric memory NVLink optimization.
5. **Checkpointing & Overlap**: Selective activation checkpointing with optimal MILP policy, forward-backward pass overlap for DeepSeek-style comms/compute overlap.
6. **Sparse & Block-Sparse**: BlockSparseTensor, attention pattern generators (local, causal, axial, Swin, dilated, ALiBi, random).
7. **Profiler**: Multi-backend profiler (PyTorch, Nsight, Memory Snapshots, DCGM), MFU/HFU analysis.
8. **Triton Kernels**: Custom kernels for RMSNorm, RoPE, tiled matmul, indexing (scaled_index_add, index_select_cat).
9. **Tree Attention**: Hierarchical/tree-structured attention for speculative decoding.
10. **Build & Deployment**: CUDA/ROCm builds, CUTLASS integration, PyTorch stable ABI, C++/CUDA extensions.

When the answer needs precise behavior, cite the source file path from `sources/xformers` and, if you read
current files, cite `path:line`. If a reference here names a symbol and the user is about to modify code,
verify the current source first.

## xFormers in one page

xFormers is a PyTorch library providing optimized building blocks for Transformer models:

1. **Memory-Efficient Exact Attention** - Up to 10x faster MHA using Flash Attention, CUTLASS, and custom kernels.
   Supports causal, local, block-diagonal, paged attention patterns. Multiple backends auto-dispatch.
2. **Fused Activation Operations** - SwiGLU (SiLU(x1)*x2), RMSNorm, RoPE with KV-cache management, all with
   Triton/CUDA kernels for maximum performance.
3. **2:4 Structured Sparsity** - Hardware-accelerated sparse tensor operations using NVIDIA's 2:4 sparsity pattern
   with CUTLASS and cuSPARSELt backends. Supports training with dynamic sparsification.
4. **Sequence & Model Parallelism** - Fused communication + computation operators that overlap all-gather/reduce-scatter
   with matrix multiplications over NVLink using symmetric memory.
5. **Selective Activation Checkpointing** - Automatically find the optimal checkpointing policy given a memory budget
   using Mixed Integer Linear Programming (MILP).
6. **Forward-Backward Overlap** - Overlap communication and computation in both forward and backward passes,
   inspired by DeepSeek's approach.

**Core module hierarchy:**
- `xformers/ops/fmha/` - Flash Memory-Efficient Attention (FMHA), multiple backends
- `xformers/ops/swiglu_op.py` - Fused SwiGLU activation
- `xformers/ops/rmsnorm.py` - RMS Normalization
- `xformers/ops/rope_padded.py` - Rotary Position Embeddings with padded KV-cache
- `xformers/ops/sp24.py` - 2:4 structured sparsity
- `xformers/ops/sequence_parallel_fused_ops.py` - Fused sequence parallel operations
- `xformers/ops/modpar_layers.py` - Column/Row parallel linear layers
- `xformers/ops/tiled_matmul.py` - Tiled matrix multiplication
- `xformers/ops/indexing.py` - Optimized indexing operations
- `xformers/ops/tree_attention.py` - Tree attention for speculative decoding
- `xformers/checkpoint.py` - Selective activation checkpointing
- `xformers/fwbw_overlap.py` - Forward-backward pass overlap
- `xformers/profiler/` - Multi-backend profiler
- `xformers/sparse/` - Block-sparse tensor support
- `xformers/components/attention/attention_patterns.py` - Attention pattern generators
- `xformers/ops/_triton/` - Triton kernel implementations
- `xformers/csrc/` - C++/CUDA extensions

**Hardware support:**
- NVIDIA GPUs: A100+ (Ampere), H100 (Hopper), B100 (Blackwell)
- AMD GPUs: ROCm 7.1+ with Composable Kernel (CK) backend
- Triton kernels require compute capability >= 8.0 (A100+)

**Dependencies:**
- PyTorch >= 2.10.0 (stable ABI)
- Triton (optional, for A100+ custom kernels)
- scipy (optional, for optimal checkpointing policy)
- CUTLASS (bundled)
- Flash Attention (external, from PyTorch indices)

## Quick reference

```python
import xformers.ops as xops
import xformers.profiler

# Memory-efficient attention
output = xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask())

# SwiGLU
output = xops.swiglu(x, w1, b1, w2, b2, w3, b3)

# RMSNorm
from xformers.ops import rms_norm, RMSNorm
normed = rms_norm(x, weight, eps=1e-6)

# RoPE with KV-cache
from xformers.ops import rope_padded
out_q = rope_padded(xq, xk, xv, cache_k, cache_v, attn_bias)

# 2:4 Sparsity
from xformers.ops import sparsify24, sparsify24_ste
sparse_tensor = sparsify24(weight, backend="cutlass")
ste_tensor = sparsify24_ste(weight, bw_mul0=0.5, bw_mul1=1.0)

# Tiled matmul
output_tiles = xops.tiled_matmul(a_tiles, b_tiles)

# Selective checkpointing
from xformers import checkpoint, get_optimal_checkpoint_policy
policy = get_optimal_checkpoint_policy(model, *args, memory_budget=0.5)
output = checkpoint(model, *args, policy_fn=policy)

# Profiler
with xformers.profiler.profile("profile_data", module=model):
    for i in range(20):
        model(inp).sum().backward()
        xformers.profiler.step()
```

## Detailed Reference Documents

Each topic below is documented in its own reference file for in-depth information:

| # | Topic | Reference |
|---|-------|-----------|
| 01 | Overview, Installation & Architecture | [01-overview-installation.md](references/01-overview-installation.md) |
| 02 | Memory-Efficient Attention (FMHA) | [02-memory-efficient-attention.md](references/02-memory-efficient-attention.md) |
| 03 | Attention Biases and Masks | [03-attention-biases-masks.md](references/03-attention-biases-masks.md) |
| 04 | SwiGLU Operation | [04-swiglu.md](references/04-swiglu.md) |
| 05 | RMSNorm | [05-rmsnorm.md](references/05-rmsnorm.md) |
| 06 | RoPE with Padded KV-Cache | [06-rope-padded.md](references/06-rope-padded.md) |
| 07 | 2:4 Structured Sparsity (SP24) | [07-structured-sparsity-sp24.md](references/07-structured-sparsity-sp24.md) |
| 08 | Sequence Parallel Fused Ops | [08-sequence-parallel-ops.md](references/08-sequence-parallel-ops.md) |
| 09 | Model Parallel Linear Layers | [09-model-parallel-layers.md](references/09-model-parallel-layers.md) |
| 10 | Tiled Matrix Multiplication | [10-tiled-matmul.md](references/10-tiled-matmul.md) |
| 11 | Indexing Operations | [11-indexing-operations.md](references/11-indexing-operations.md) |
| 12 | Block-Sparse Tensors | [12-block-sparse-tensors.md](references/12-block-sparse-tensors.md) |
| 13 | Attention Patterns | [13-attention-patterns.md](references/13-attention-patterns.md) |
| 14 | Selective Activation Checkpointing | [14-checkpointing.md](references/14-checkpointing.md) |
| 15 | Forward-Backward Overlap | [15-fw-bw-overlap.md](references/15-fw-bw-overlap.md) |
| 16 | Profiler | [16-profiler.md](references/16-profiler.md) |
| 17 | Triton Kernels | [17-triton-kernels.md](references/17-triton-kernels.md) |
| 18 | Tree Attention | [18-tree-attention.md](references/18-tree-attention.md) |
| 19 | C++/CUDA Extensions | [19-cpp-cuda-extensions.md](references/19-cpp-cuda-extensions.md) |
| 20 | Build Configuration | [20-build-configuration.md](references/20-build-configuration.md) |
| 21 | LLaMA Inference Example | [21-examples-llama.md](references/21-examples-llama.md) |
| 22 | Changelog & Version History | [22-changelog-history.md](references/22-changelog-history.md) |
