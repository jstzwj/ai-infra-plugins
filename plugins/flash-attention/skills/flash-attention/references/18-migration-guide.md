# 18 - Migration Guide

This document covers migrating between FlashAttention versions, including FA1 to FA2, FA2 to FA3, and FA2 to FA4, with detailed API changes, parameter changes, and behavioral differences.

---

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [FA1 to FA2 Migration](#fa1-to-fa2-migration)
3. [FA2 to FA3 Migration](#fa2-to-fa3-migration)
4. [FA2 to FA4 Migration](#fa2-to-fa4-migration)
5. [Function Renaming](#function-renaming)
6. [Parameter Changes](#parameter-changes)
7. [Behavior Changes](#behavior-changes)
8. [Backward Compatibility](#backward-compatibility)
9. [Common Migration Issues](#common-migration-issues)

---

## Migration Overview

| From | To | Code Location | Key Changes |
|------|----|--------------|-------------|
| FA1 | FA2 | `csrc/flash_attn/` | CuTe/CUTLASS integration, better heuristics |
| FA2 | FA3 | `hopper/` | TMA, WGMMA, SM90 optimizations |
| FA2 | FA4 | `flash_attn/cute/` | CuTeDSL, score/mask modifiers, 2CTA |

### Version Identification

```python
import flash_attn
print(flash_attn.__version__)

# FA2: version < 3.0
# FA3: version >= 3.0, uses hopper/ kernels on SM90
# FA4: version >= 4.0, uses flash_attn/cute/ kernels
```

---

## FA1 to FA2 Migration

### Major Changes

1. **Online softmax**: FA2 uses the online softmax algorithm, eliminating the need for a separate softmax kernel
2. **CuTe/CUTLASS**: FA2 is built on CuTe tensor abstractions and CUTLASS library
3. **Backward rewrite**: FA2 backward uses 5 GEMMs with improved tiling
4. **Split-KV**: FA2 introduces split-KV parallelism for long sequences

### API Changes

#### Function Signatures

```python
# FA1
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
output = flash_attn_unpadded_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal)

# FA2 (same function, improved implementation)
from flash_attn.flash_attn_interface import flash_attn_varlen_func
output = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k,
                                 max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal)
```

#### New Functions in FA2

```python
# Standard (padded) attention
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

# Variable-length (unpadded) attention
from flash_attn import flash_attn_varlen_func
output = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                 max_seqlen_q, max_seqlen_k,
                                 dropout_p=0.0, softmax_scale=None, causal=False)
```

#### Deprecated Functions

| FA1 Function | FA2 Replacement |
|-------------|----------------|
| `flash_attn_unpadded_func` | `flash_attn_varlen_func` |
| `flash_attn_unpadded_kvpacked_func` | `flash_attn_varlen_kvpacked_func` |
| `flash_attn_unpadded_qkvpacked_func` | `flash_attn_varlen_qkvpacked_func` |

### New Features in FA2

- **GQA/MQA**: Support for grouped-query and multi-query attention
- **Window attention**: Sliding window support via `window_size_left`, `window_size_right`
- **ALiBi**: Attention with Linear Biases
- **Softcapping**: `tanh`-based score capping
- **Rotary embeddings**: In-kernel rotary embedding application
- **Deterministic backward**: Optional deterministic gradient computation
- **FP8**: Forward-only FP8 support (in FA3/Hopper kernels)

---

## FA2 to FA3 Migration

### Major Changes

1. **TMA (Tensor Memory Accelerator)**: FA3 uses TMA for bulk memory transfers on SM90
2. **WGMMA (Warpgroup MMA)**: FA3 uses warpgroup-level matrix multiply on SM90
3. **PackGQA**: Automatic packing of GQA heads for efficiency
4. **Paged KV**: Full paged KV cache support
5. **FP8**: Native FP8 support with descale factors

### API Additions

```python
# KV cache with paged attention (new in FA3)
from flash_attn import flash_attn_with_kvcache
output = flash_attn_with_kvcache(
    q, cache_k, cache_v,
    k_new=k, v_new=v,
    cache_seqlens=cache_seqlens,
    block_table=block_table,
    rotary_cos=cos, rotary_sin=sin,
)

# FP8 attention (new in FA3)
output = flash_attn_func(q_fp8, k_fp8, v_fp8,
                          q_descale=q_scale, k_descale=k_scale, v_descale=v_scale)
```

### New Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_table` | Tensor | Paged KV cache page table |
| `cache_seqlens` | Tensor | Current cache lengths per batch |
| `rotary_cos` | Tensor | Cosine values for rotary embedding |
| `rotary_sin` | Tensor | Sine values for rotary embedding |
| `q_descale` | Tensor | FP8 descale factors for Q |
| `k_descale` | Tensor | FP8 descale factors for K |
| `v_descale` | Tensor | FP8 descale factors for V |

### Behavioral Differences

1. **Causal mask alignment**: FA3's causal mask handling aligns differently at tile boundaries. The difference is at most `kBlockN - 1` positions and is numerically equivalent (just different rounding at block edges).

2. **LSE format**: FA3 uses `log2`-based LSE internally (for `exp2` optimization) but converts to natural log before returning to the user.

3. **Deterministic backward**: FA3's deterministic backward uses a different accumulation strategy (separate buffers per thread block instead of atomic adds).

---

## FA2 to FA4 Migration

### Major Changes

1. **CuTeDSL**: Kernels are written in Python using CUTLASS DSL and JIT-compiled
2. **Score/mask modifiers**: User-defined JIT callables for custom attention patterns
3. **2CTA**: Cooperative attention for Blackwell GPUs
4. **Persistent kernels**: Work-pulling scheduler for better load balancing
5. **torch.compile support**: First-class `torch.compile()` integration

### API Changes

#### Standard Interface

```python
# FA2
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True, window_size_left=256)

# FA4 (same interface, different backend)
from flash_attn.cute.interface import flash_attn_func
output = flash_attn_func(q, k, v, causal=True, window_size_left=256)
```

#### New: Score Modifiers

```python
from flash_attn.cute.interface import flash_attn_func
import cute

@cute.jit
def my_score_mod(score, batch, head, q_idx, kv_idx):
    # Custom score modification
    score *= 1.0 / (1.0 + cute.math.float32(abs(q_idx - kv_idx)))

output = flash_attn_func(q, k, v, score_mod=my_score_mod)
```

#### New: Mask Modifiers

```python
@cute.jit
def my_mask_mod(batch, head, q_idx, kv_idx):
    return abs(q_idx - kv_idx) <= 512

output = flash_attn_func(q, k, v, mask_mod=my_mask_mod)
```

#### New: Block Sparse Attention

```python
# Block sparse mask tensors
sparse_mask = create_sparse_mask(batch, heads, m_blocks, n_blocks)
output = flash_attn_func(q, k, v, block_sparse_tensors=[sparse_mask])
```

### New Parameters in FA4

| Parameter | Type | Description |
|-----------|------|-------------|
| `score_mod` | JIT callable | Custom score modification function |
| `mask_mod` | JIT callable | Custom mask function |
| `block_sparse_tensors` | list[Tensor] | Block sparse mask tensors |
| `pack_gqa` | bool | Force PackGQA optimization |
| `m_block_size` | int | Custom M tile size |
| `n_block_size` | int | Custom N tile size |
| `num_threads` | int | Custom thread count per block |

### Installation Differences

```bash
# FA2/FA3: C++/CUDA compilation
pip install flash-attn  # Compiles C++/CUDA kernels

# FA4: CuTeDSL (Python-based, JIT compiled)
pip install flash-attn-4  # No C++ compilation needed
# Or development install:
pip install -e "flash_attn/cute[dev]"
```

---

## Function Renaming

### Complete Renaming Table

| FA1 | FA2 | FA3/FA4 |
|-----|-----|---------|
| `flash_attn_unpadded_func` | `flash_attn_varlen_func` | `flash_attn_varlen_func` |
| `flash_attn_unpadded_qkvpacked_func` | `flash_attn_varlen_qkvpacked_func` | -- |
| `flash_attn_unpadded_kvpacked_func` | `flash_attn_varlen_kvpacked_func` | -- |
| -- | `flash_attn_func` | `flash_attn_func` |
| -- | `flash_attn_qkvpacked_func` | -- |
| -- | `flash_attn_kvpacked_func` | -- |
| -- | -- | `flash_attn_with_kvcache` |

### Import Paths

```python
# FA2/FA3
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import flash_attn_with_kvcache

# FA4
from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func
```

---

## Parameter Changes

### Parameter History

| Parameter | FA1 | FA2 | FA3 | FA4 |
|-----------|-----|-----|-----|-----|
| `q, k, v` | Yes | Yes | Yes | Yes |
| `cu_seqlens_q, cu_seqlens_k` | Yes | Yes | Yes | Yes |
| `max_seqlen_q, max_seqlen_k` | Yes | Yes | Yes | Yes |
| `dropout_p` | Yes | Yes | Yes | Yes |
| `softmax_scale` | Yes | Yes | Yes | Yes |
| `causal` | Yes | Yes | Yes | Yes |
| `window_size_left` | No | Yes | Yes | Yes |
| `window_size_right` | No | Yes | Yes | Yes |
| `alibi_slopes` | No | Yes | Yes | Yes |
| `softcap` | No | Yes | Yes | Yes |
| `deterministic` | No | Yes | Yes | Yes |
| `return_attn_probs` | Yes | Yes | Yes | Yes |
| `block_table` | No | No | Yes | Yes |
| `cache_seqlens` | No | No | Yes | Yes |
| `rotary_cos, rotary_sin` | No | Partial | Yes | Yes |
| `q/k/v_descale` | No | No | Yes | Yes |
| `score_mod` | No | No | No | Yes |
| `mask_mod` | No | No | No | Yes |
| `block_sparse_tensors` | No | No | No | Yes |
| `pack_gqa` | No | No | Yes | Yes |
| `m/n_block_size` | No | No | No | Yes |

### Deprecated Parameters

| Parameter | Replacement | Version |
|-----------|------------|---------|
| `return_attn_probs` | Removed (use `p_ptr` internally) | FA2 |
| `max_seqlen_q, max_seqlen_k` (in varlen) | Computed from `cu_seqlens` | FA4 |

---

## Behavior Changes

### Causal Mask Alignment

**FA2**: Causal mask at tile boundaries masks based on the first row of the tile.

**FA3**: Causal mask alignment differs by up to `kBlockN - 1` positions at block boundaries.

**Impact**: Numerically equivalent for positions within the block, but the exact set of masked elements at the boundary may differ. This can cause small numerical differences (< 1e-6 in fp32) when comparing FA2 and FA3 outputs.

### LSE (Log-Sum-Exp) Format

**FA2**: Returns LSE in natural log base: `lse = log(sum(exp(scores)))`
**FA3**: Internally uses `log2` for `exp2` optimization, converts to natural log before output
**FA4**: Same as FA3

### Dropout RNG

**All versions**: Use Philox counter-based RNG for deterministic dropout patterns. The exact RNG sequence may differ between versions, so dropout patterns are NOT reproducible across versions.

### Softmax Scale

**FA2**: Applies `softmax_scale` during the QK^T computation: `scores = Q @ K^T * softmax_scale`
**FA3/FA4**: Same behavior

### Gradient Accumulation

**FA2 (non-deterministic)**: Uses `atomicAdd` for dQ accumulation across thread blocks
**FA2 (deterministic)**: Uses separate accumulation buffers per thread block, then sums
**FA3/FA4**: Same options, with SM90-optimized accumulation

---

## Backward Compatibility

### FA2 Backward Compatibility

FA2 maintains backward compatibility with FA1:
- `flash_attn_unpadded_func` is aliased to `flash_attn_varlen_func`
- All FA1 parameter combinations are supported
- Numerical results may differ slightly due to implementation improvements

### FA3 Backward Compatibility

FA3 is backward compatible with FA2:
- On SM90 (Hopper), FA3 kernels are automatically selected
- On SM80 (Ampere), FA2 kernels are used as fallback
- All FA2 parameters are accepted by FA3 functions

### FA4 Backward Compatibility

FA4 is a separate package (`flash-attn-4`):
- Can coexist with `flash-attn` (FA2/FA3)
- Import from `flash_attn.cute` instead of `flash_attn`
- New features (score_mod, mask_mod) are FA4-only

---

## Common Migration Issues

### Issue 1: Import Errors

```python
# If you get: ModuleNotFoundError: No module named 'flash_attn.cute'
# You need FA4, not FA2/FA3:
pip install flash-attn-4
```

### Issue 2: Shape Mismatches

```python
# FA2/FA3 expect: (batch, seqlen, num_heads, head_dim)
# Some code uses: (batch, seqlen, head_dim, num_heads) -- WRONG
q = q.transpose(1, 2)  # Fix: swap num_heads and head_dim
```

### Issue 3: Contiguity

```python
# FA2/FA3 require contiguous last dimension
q = q.contiguous()
k = k.contiguous()
v = v.contiguous()
```

### Issue 4: cu_seqlens Format

```python
# Correct: cumulative, starting at 0
cu_seqlens = [0, 10, 25, 42]  # Sequences of length 10, 15, 17

# Wrong: direct lengths (use seqused parameter for this)
cu_seqlens = [10, 15, 17]  # WRONG for cu_seqlens

# For direct lengths, use:
# cu_seqlens = torch.cumsum(lengths, dim=0)
# cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32), cu_seqlens])
```

### Issue 5: Head Dimension

Supported head dimensions: 16, 32, 64, 96, 128, 192, 256

```python
# If your head_dim is not supported, pad to the nearest supported value
supported = [16, 32, 64, 96, 128, 192, 256]
padded_dim = min(d for d in supported if d >= actual_head_dim)
q = F.pad(q, (0, padded_dim - actual_head_dim))
```

### Issue 6: BF16 vs FP16

```python
# FA2 kernels are specialized per dtype
# Using wrong dtype gives incorrect results or crashes
q = q.to(torch.bfloat16)  # or torch.float16
k = k.to(torch.bfloat16)
v = v.to(torch.bfloat16)
```

### Issue 7: Version-Specific Features

```python
# Check for feature availability at runtime
import flash_attn
version = tuple(int(x) for x in flash_attn.__version__.split('.')[:2])

has_window = version >= (2, 3)
has_softcap = version >= (2, 5)
has_paged_kv = version >= (2, 6)

# FA4 check
try:
    from flash_attn.cute.interface import flash_attn_func
    has_fa4 = True
except ImportError:
    has_fa4 = False
```

### Issue 8: Performance Regression After Migration

If performance degrades after migration:

1. **Check SM version**: FA3 kernels require SM90+. On SM80, FA2 kernels are used.
2. **Block size selection**: Different versions may choose different block sizes. Use `num_splits` and custom block sizes to tune.
3. **PackGQA**: FA3 automatically uses PackGQA for GQA models. If this causes issues, disable it.
4. **Compile cache**: Clear the JIT cache after upgrading: `rm -rf /tmp/${USER}/flash_attention_cute_dsl_cache/`
5. **Warm up**: Run a few iterations before benchmarking.
