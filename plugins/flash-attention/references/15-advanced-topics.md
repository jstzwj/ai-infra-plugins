# 15 - Advanced Topics

This document covers advanced features and implementation details of FlashAttention, including paged KV cache, MQA/GQA, block sparse attention, softcapping, ALiBi, sliding window attention, rotary embeddings, variable-length batching, and more.

---

## Table of Contents

1. [Paged KV Cache (PagedAttention)](#paged-kv-cache-pagedattention)
2. [MQA/GQA Implementation](#mqagqa-implementation)
3. [Block Sparse Attention](#block-sparse-attention)
4. [Softcapping (Tanh Attention)](#softcapping-tanh-attention)
5. [ALiBi Position Encoding](#alibi-position-encoding)
6. [Sliding Window Attention](#sliding-window-attention)
7. [Rotary Embedding Integration](#rotary-embedding-integration)
8. [Variable-Length Sequence Batching](#variable-length-sequence-batching)
9. [Split-KV Parallelism](#split-kv-parallelism)
10. [2CTA Cooperative Attention](#2cta-cooperative-attention)
11. [Score and Mask Modifiers](#score-and-mask-modifiers)
12. [torch.compile() Support](#torchcompile-support)
13. [FP8 Quantized Attention](#fp8-quantized-attention)
14. [PackGQA Optimization](#packgqa-optimization)
15. [Inference-Specific Optimizations](#inference-specific-optimizations)

---

## Paged KV Cache (PagedAttention)

### Overview

Paged KV Cache applies the virtual memory concept to KV cache management. Instead of requiring contiguous memory for each sequence's KV cache, the cache is divided into fixed-size pages that can be non-contiguous in physical memory.

### Page Structure

```
Physical Memory:
+--------+--------+--------+--------+--------+--------+
| Page 0 | Page 1 | Page 2 | Page 3 | Page 4 | Page 5 |
| (B,H)  | (B,H)  | (A,H)  | (B,H)  | (C,H)  | (A,H)  |
+--------+--------+--------+--------+--------+--------+

Sequence A: Pages 2, 5 (non-contiguous)
Sequence B: Pages 0, 1, 3
Sequence C: Page 4
```

### Block Table

Each sequence has a block table mapping logical page indices to physical pages:

```python
# block_table shape: (batch_size, max_num_pages_per_seq)
# block_table[batch][logical_page] = physical_page_index
```

For a KV block at position `n_block`:
```python
page_index = n_block * kBlockN // page_block_size
page_offset = n_block * kBlockN - page_index * page_block_size
physical_addr = page_table[batch][page_index] * batch_stride + page_offset * row_stride
```

### PagedKVManager (CUDA Implementation)

File: `hopper/paged_kv.h`

```cpp
template<int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads,
         typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {
    // Uses cp.async (not TMA) because pages may be non-contiguous
    using GmemCopyAtomCpAsync = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;

    // Distributed page pointer computation:
    // 8 threads per row compute page pointers, then __shfl_sync to broadcast
    static constexpr int kPageEntryPerThread = ceil_div(rows_per_thread, kGmemThreadsPerRow);

    void load_K(n_block);  // Load K tile from paged memory
    void load_V(n_block);  // Load V tile from paged memory
};
```

### Key Design Decisions

1. **cp.async instead of TMA**: TMA requires contiguous tensor descriptors, which paged memory violates. cp.async with software-managed addressing handles non-contiguous pages.

2. **Distributed pointer computation**: Computing int64 physical addresses is expensive. The work is distributed across threads loading the same row, with results broadcast via warp shuffle.

3. **Page boundary handling**: When a KV block spans two pages, the load function handles the split by computing two separate physical addresses.

4. **Zero-fill on empty pages**: Pages that haven't been written yet are zero-filled to avoid reading uninitialized memory.

### Usage in Forward Pass

```python
# Python API
block_table = torch.tensor([[0, 2, 5], [1, 3, 4]], device='cuda', dtype=torch.int32)
flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    block_table=block_table,
    page_block_size=256  # Must match kernel's kBlockN alignment
)
```

---

## MQA/GQA Implementation

### Multi-Query Attention (MQA)

In MQA, all query heads share a single K/V head:
- Q: `(batch, seqlen_q, num_heads, headdim)`
- K: `(batch, seqlen_k, 1, headdim)`
- V: `(batch, seqlen_k, 1, headdim)`

The `h_h_k_ratio = num_heads / 1 = num_heads`.

### Grouped-Query Attention (GQA)

In GQA, Q heads are divided into groups, each sharing K/V heads:
- Q: `(batch, seqlen_q, num_heads, headdim)`
- K: `(batch, seqlen_k, num_kv_heads, headdim)`
- V: `(batch, seqlen_k, num_kv_heads, headdim)`
- `h_h_k_ratio = num_heads / num_kv_heads`

### CUDA Implementation

The kernel broadcasts K/V across Q heads:

```cpp
// In compute_attn_1rowblock:
Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), ...);
Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), ...);
```

Each query head `bidh` maps to KV head `bidh / h_h_k_ratio`. Multiple Q heads read the same K/V data.

### Grid Scheduling

For the standard (non-SplitKV) forward:
```
grid = (num_m_blocks, batch_size, num_heads)
```

Multiple Q heads for the same KV head will load the same K/V data independently. This is acceptable because:
1. K/V loads are from HBM (no caching conflict at the SM level)
2. The shared memory layout is per-CTA (no sharing across thread blocks)

### PackGQA Optimization

File: `hopper/pack_gqa.py`

PackGQA packs multiple Q heads into the same CTA to share K/V loading:

```python
def pack_gqa(q, k, v, qhead_per_khead):
    # Reshape Q from (B, S, H_q, D) to (B, S, H_kv, qhead_per_khead * D)
    q_packed = q.reshape(B, S, H_kv, qhead_per_khead, D)
    # Treat as larger head dimension: D' = qhead_per_khead * D
    q_packed = q_packed.reshape(B, S, H_kv, qhead_per_khead * D)
    return q_packed, k, v
```

Benefits:
- Single K/V load per CTA, shared across multiple Q heads
- Higher arithmetic intensity (larger effective M dimension)
- Better utilization for small sequence lengths

### PackGQA Heuristic

```cpp
inline bool should_pack_gqa(bool varlen_q, int seqlen_q,
                            int qhead_per_khead, int blockM) {
    if (varlen_q) return true;  // Always pack for varlen
    float nopack_eff = float(seqlen_q) / float(round_up(seqlen_q, blockM));
    float pack_eff = float(seqlen_q * qhead_per_khead) / float(round_up(seqlen_q * qhead_per_khead, blockM));
    return nopack_eff < 0.9 * pack_eff;
}
```

---

## Block Sparse Attention

### Overview

Block sparse attention applies a sparsity mask at the block level, where entire blocks of the attention matrix are either computed or skipped entirely.

### Sparse Block Format

The sparsity pattern is specified as a set of dense tensors where each element indicates whether a corresponding block should be computed:

```python
# block_sparse_tensors: list of tensors
# Each tensor has shape (batch, num_heads, num_m_blocks, num_n_blocks)
# Value of 1 means compute, 0 means skip
```

### Implementation

In the CUDA kernel, sparsity is applied during the loop over K/V blocks:

```cpp
for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // Check if this block is in the sparse mask
    if (!block_sparse_mask[bidb][bidh][m_block][n_block]) {
        continue;  // Skip this block entirely
    }
    // ... compute attention for this block
}
```

### Benefits

- O(blocks_computed) instead of O(seqlen_q * seqlen_k)
- Compatible with causal, local, and other mask types
- Sparse pattern can be dynamic (different per batch/head)

### Limitations

- Block granularity: Sparsity is at the block level (e.g., 128x64), not individual elements
- Load balancing: Sparse patterns may create unbalanced work distribution
- Currently supported in FA4 (CuTeDSL) kernels

---

## Softcapping (Tanh Attention)

### Mathematical Formulation

Softcapping applies a tanh function to attention scores before softmax:

```
scores = Q @ K^T / sqrt(d)
capped_scores = softcap * tanh(scores / softcap)
attention = softmax(capped_scores) @ V
```

The `softcap` parameter controls the maximum absolute value of scores. When `softcap > 0`, all scores are bounded in `[-softcap, softcap]`.

### Forward Implementation

```cpp
// In compute_attn_1rowblock:
if constexpr (Is_softcap) {
    FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
}
// apply_softcap applies: tensor = softcap * tanh(tensor / softcap)
```

### Backward Implementation

The gradient of softcapping is:

```
d_capped_scores = d_output * (1 - tanh(scores/softcap)^2)
```

In the backward kernel:

```cpp
// Compute dtanh = 1 - tanh^2(scores / softcap)
Tensor dtanh = make_tensor_like(scores);
FLASH_NAMESPACE::calculate_dtanh(scores, dtanh, params.softcap);

// Apply to dS
for each element:
    scaled_ds = P * (dP - dP_sum) * dtanh
```

### Usage

```python
output = flash_attn_func(q, k, v, softcap=50.0)
```

Common values: 15.0 (Gemma), 30.0 (some Llama variants), 50.0.

---

## ALiBi Position Encoding

### Overview

Attention with Linear Biases (ALiBi) adds a position-dependent bias to attention scores instead of using positional embeddings in the input:

```
scores = Q @ K^T / sqrt(d) + alibi_bias(row, col)
```

### Bias Formula

**Causal mode**: `bias = slope * col_idx` (adds a decreasing penalty for attending to distant keys)

**Non-causal mode**: `bias = -slope * |row_idx + seqlen_k - seqlen_q - col_idx|` (symmetric distance penalty)

### Slope Values

Slopes are typically derived from a geometric sequence:
```python
slopes = 2 ** (-8 * arange(num_heads) / num_heads)
# For 8 heads: [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
```

### CUDA Implementation

```cpp
struct Alibi {
    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    void apply_alibi(Tensor &tensor, int col_idx_offset, int row_idx_offset, int warp_row_stride) {
        if (Is_causal) {
            tensor(mi, col) += alibi_slope * col_idx;
        } else {
            tensor(mi, col) -= alibi_slope * abs(row_idx + seqlen_k - seqlen_q - col_idx);
        }
    }
};
```

The slope is pre-divided by `scale_softmax` before being passed to the kernel, so the bias is applied in the scaled score space.

### Usage

```python
alibi_slopes = torch.tensor([...], device='cuda')  # (batch, heads) or (heads,)
output = flash_attn_func(q, k, v, alibi_slopes=alibi_slopes)
```

---

## Sliding Window Attention

### Overview

Sliding window attention restricts each query to attend only to keys within a local window:

```
window_size_left:  Number of tokens to the left of the diagonal to include
window_size_right: Number of tokens to the right of the diagonal to include
```

Special cases:
- `window_size_right=0, window_size_left=infinity`: Causal attention
- `window_size_right=0, window_size_left=W`: Causal with window W
- `window_size_right=W_r, window_size_left=W_l`: Bidirectional window

### Masking Logic

For each element at `(row_idx, col_idx)`:

```cpp
col_idx_limit_left = max(0, row_idx + seqlen_k - seqlen_q - window_size_left)
col_idx_limit_right = min(seqlen_k, row_idx + 1 + seqlen_k - seqlen_q + window_size_right)

if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
    tensor[row][col] = -INFINITY;
}
```

### Block Range Computation

The kernel computes `n_block_min` and `n_block_max` to skip blocks entirely outside the window:

```cpp
n_block_min = max(0, (m_block * kBlockM + seqlen_k - seqlen_q - window_size_left) / kBlockN);
n_block_max = min(ceil_div(seqlen_k, kBlockN),
                  ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q + window_size_right, kBlockN));
```

### Performance Impact

Sliding window attention is faster than full attention because:
1. Many K/V blocks are skipped entirely (no load, no compute)
2. The effective sequence length for attention is reduced
3. L2 cache is more effective with the reduced working set

### Usage

```python
# Causal with window of 512 tokens
output = flash_attn_func(q, k, v, causal=True, window_size_left=512)

# Bidirectional window
output = flash_attn_func(q, k, v, window_size_left=256, window_size_right=256)
```

---

## Rotary Embedding Integration

### Overview

Rotary Position Embedding (RoPE) encodes position information by rotating query and key vectors in a 2D plane based on their position.

### Mathematical Formulation

For position `pos` and dimension pair `(2i, 2i+1)`:

```
Q'[2i]   = Q[2i]   * cos(pos * theta_i) - Q[2i+1] * sin(pos * theta_i)
Q'[2i+1] = Q[2i]   * sin(pos * theta_i) + Q[2i+1] * cos(pos * theta_i)
```

Where `theta_i = 1 / (10000^(2i/d))`.

### Two Layout Formats

FlashAttention supports two rotary embedding layouts:

1. **Interleaved**: Pairs are consecutive `(x0, x1, x2, x3, ...)` where `(x0, x1)` form a rotation pair.

2. **Contiguous (half-rotation)**: First half and second half form pairs `(x0, x_{d/2}), (x1, x_{d/2+1}), ...)`.

### In-Kernel Application

Rotary embeddings can be applied during the kernel's memory load phase:

```cpp
// When Append_KV is true, rotary is applied to K during cache append
copy_rotary_interleaved(tKgKnew, tKgK, tRgCos, tRgSin, ...);
// Or
copy_rotary_contiguous(tKgKnew, tKgK, tRgCos, tRgSin, ...);
```

And optionally to Q during loading:
```cpp
copy_rotary_interleaved(tQgQ, tQsQ, tRgCos, tRgSin, ...);
```

### Parameters

```python
flash_attn_with_kvcache(
    q, k, v, cache_k, cache_v,
    rotary_cos=cos_tensor,    # (max_seqlen, rotary_dim/2) or (max_seqlen, rotary_dim)
    rotary_sin=sin_tensor,    # Same shape as cos
    rotary_interleaved=True,  # True for interleaved, False for contiguous
)
```

The `rotary_dim` parameter controls how many dimensions get rotary embedding. Dimensions beyond `rotary_dim` pass through unchanged.

---

## Variable-Length Sequence Batching

### Overview

Variable-length (varlen) batching packs multiple sequences of different lengths into a single batch tensor, avoiding padding waste.

### Data Layout

Sequences are packed contiguously:

```
[seq_0_tokens][seq_1_tokens][seq_2_tokens]...
```

With cumulative sequence length arrays:

```python
cu_seqlens_q = [0, len_0, len_0 + len_1, ..., total_tokens]
cu_seqlens_k = [0, len_0, len_0 + len_1, ..., total_tokens]
```

### BlockInfo for Variable Length

```cpp
struct BlockInfo {
    // For batch element bidb:
    int sum_s_q;         // Starting offset in the packed Q tensor
    int actual_seqlen_q; // Length of this sequence
    int actual_seqlen_k; // Length of the corresponding K/V sequence

    // Offset computation for packed tensors:
    index_t q_offset(batch_stride, row_stride, bidb) {
        return sum_s_q == -1 ? bidb * batch_stride : sum_s_q * row_stride;
    }
};
```

### Unpadding/Repadding

File: `flash_attn/bert_padding.py`

```python
def unpad_input(hidden_states, attention_mask):
    """
    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool
    Returns:
        output: (total_nnz, ...)
        indices: (total_nnz,)
        cu_seqlens: (batch + 1,)
        max_seqlen: int
        seqused: (batch,)
    """
    indices = torch.nonzero(attention_mask.flatten()).flatten()
    output = index_first_axis(hidden_states.reshape(-1, *rest), indices)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0), (1, 0))
    return output, indices, cu_seqlens, max_seqlen, seqused

def pad_input(hidden_states, indices, batch, seqlen):
    """Reverse of unpad_input."""
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return output.reshape(batch, seqlen, ...)
```

### Varlen-Specific Optimizations

1. **Early exit**: Thread blocks for empty sequences exit immediately
2. **Left padding**: K/V sequences can have left padding, handled by `leftpad_k` parameter
3. **Non-cumulative seqlens**: `cu_seqlens_k` can store direct lengths instead of cumulative offsets (via `is_seqlens_k_cumulative=False`)
4. **seqused arrays**: When only some sequences in a batch need attention (others unused), `seqused_q/k` specifies actual lengths per batch

### Tile Quantization Bug

A known issue with variable-length sequences is tile quantization: when `actual_seqlen_k` is not a multiple of `kBlockN`, the last block needs boundary masking. The `n_masking_steps` computation handles this:

```cpp
constexpr int n_masking_steps = (!Is_causal && !Is_local)
    ? 1
    : ((Is_even_MN && Is_causal) ? ceil_div(kBlockM, kBlockN) : ceil_div(kBlockM, kBlockN) + 1);
```

---

## Split-KV Parallelism

### Overview

Split-KV parallelism partitions the K/V sequence across multiple thread blocks, each computing partial attention results that are combined afterwards.

### When to Use

Split-KV is beneficial when:
- The K/V sequence is very long (>4K tokens)
- The number of Q blocks is small relative to available SMs
- KV doesn't fit in L2 cache (>50MB per head)

### Algorithm

**Phase 1 (Split computation)**:
Each split `s` handles K/V blocks `[s * n_blocks_per_split, (s+1) * n_blocks_per_split)`:
```
partial_O[s] = softmax_partial(Q @ K[s]^T / sqrt(d)) @ V[s]
partial_LSE[s] = log(sum(exp(Q @ K[s]^T / sqrt(d))))
```

**Phase 2 (Combine)**:
```
total_LSE = log(sum_s(exp(partial_LSE[s] - max_LSE))) + max_LSE
final_O = sum_s(exp(partial_LSE[s] - total_LSE) * partial_O[s])
```

### Heuristic

```cpp
int num_splits_heuristic(total_mblocks, num_SMs, ...) {
    if (total_mblocks >= 0.8 * num_SMs) return 1;  // Enough parallelism
    if (num_n_blocks <= 4) return 1;  // Too few K/V blocks
    // Find splits maximizing SM occupancy
    for (int s = 1; s <= max_splits; s++) {
        if (efficiency[s] >= 0.85 * max_efficiency) return s;
    }
    return 1;
}
```

### Implementation Details

The combine kernel uses shared memory for LSE transposition:
```cpp
__shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];  // +1 to reduce bank conflicts
```

The combine kernel is templated on `Log_max_splits` to minimize shared memory usage:
- `num_splits <= 2`: `Log_max_splits = 1` (2 entries)
- `num_splits <= 4`: `Log_max_splits = 2` (4 entries)
- ...up to `num_splits <= 128`: `Log_max_splits = 7`

---

## 2CTA Cooperative Attention

### Overview

2CTA (2-Cooperative-Thread-Array) is a Blackwell SM100 feature where two CTAs in a cluster cooperate on a single attention computation.

### How It Works

1. Two CTAs are launched as a cluster with `cluster_dim_x = 2`
2. Both CTAs load the **same** K/V tiles (redundantly) but **different** Q tiles
3. The MMA (UMMA) instruction spans both CTAs' shared memory
4. Each CTA gets half of the output

### Benefits

- Doubles the effective M-dimension of the MMA without increasing per-CTA shared memory
- Better utilization of the 128-row M dimension in UMMA
- Particularly beneficial for head_dim=128 where two 64-row Q tiles can be processed together

### Implementation Pitfalls

1. **tx_count doubling**: Both CTAs' TMA operations signal the same cluster-level mbarrier. Expected byte count must be `N * cta_group_size`.

2. **Tile scheduler**: `blockIdx.x` must be divided by `cluster_shape_m` to get the logical tile index.

3. **Softmax masking**: Causal mask row positions must account for the CTA's position within the cluster (`m_block * cta_group_size`).

4. **tcgen05.commit**: If there are no pending MMA operations, the commit signal only reaches the local CTA's barrier. Must use explicit `mbarrier_arrive` to both CTAs.

5. **producer_tail**: The default producer_tail deadlocks in 2CTA mode because the consumer may have already exited. Must make it a no-op.

---

## Score and Mask Modifiers

### Overview (FA4)

FA4 introduces user-defined score and mask modifiers as `@cute.jit` callables that are injected into the kernel at compile time.

### Score Modifiers

A score modifier transforms the attention scores after the QK^T computation:

```python
@cute.jit
def my_score_mod(score, batch, head, q_idx, kv_idx):
    # score: float tensor element
    # Can modify in place
    score *= some_function(q_idx, kv_idx)
```

Built-in score modifiers:
- `softcap_score_mod(cap)`: `score = cap * tanh(score / cap)`
- ALiBi: Position-dependent linear bias

### Mask Modifiers

A mask modifier determines whether each attention element should be computed:

```python
@cute.jit
def my_mask_mod(batch, head, q_idx, kv_idx):
    # Return True to compute, False to mask
    return abs(q_idx - kv_idx) <= window_size
```

Built-in mask modifiers:
- Causal: `q_idx >= kv_idx`
- Local/sliding window: `q_idx - kv_idx <= window_right and kv_idx - q_idx <= window_left`
- Block sparse: Uses pre-computed sparsity tensors

### Compile-Time Injection

Modifiers are compiled into the kernel at JIT time, so there is zero runtime overhead:

```python
kernel = compile_flash_attn(
    score_mod=my_score_mod,
    mask_mod=my_mask_mod,
    dtype=torch.bfloat16,
    head_dim=128,
)
```

---

## torch.compile() Support

### Overview

FlashAttention provides `torch.compile()` compatibility through the `torch.library` custom op mechanism.

### Library Registration

File: `flash_attn/utils/library.py`

```python
from torch.library import custom_op

@custom_op("flash_attn::flash_attn_func", mutates_args=())
def flash_attn_func(q, k, v, ...) -> torch.Tensor:
    ...

# Register fake (meta) kernel for tracing
@flash_attn_func.register_fake
def _(q, k, v, ...):
    return torch.empty_like(q)
```

### triton_op Wrapper

```python
def triton_op(name, fn, *, mutates_args, schema=None, allow_decomposition=True):
    """Wraps a function as a torch.library custom op with optional decomposition."""
    result = custom_op(name, fn, mutates_args=mutates_args, schema=schema)
    result.register_fake(fn)
    if allow_decomposition:
        # Decompose under FunctionalTensorMode for Inductor optimization
        result.register_torch_dispatch(FunctionalTensorMode, functional_decomp)
    return result
```

### Inductor Compatibility

The `allow_decomposition=True` flag allows torch.compile to decompose the flash attention operation for further optimization. When `False`, the operation is treated as opaque (like `torch.library.custom_op`).

### Custom AMP Decorators

File: `flash_attn/utils/torch.py`

```python
# Handles PyTorch version differences in AMP decorators
from torch.amp import custom_fwd, custom_bwd  # PyTorch >= 2.x
# or
from torch.cuda.amp import custom_fwd, custom_bwd  # PyTorch 1.x
```

---

## FP8 Quantized Attention

### Overview

FP8 attention uses 8-bit floating point (E4M3 or E5M2) for Q, K, and V tensors, reducing memory bandwidth by 2x compared to fp16.

### Descale Factors

FP8 values are stored as quantized integers with per-tensor descale factors:

```python
# Q stored as FP8: actual_value = stored_fp8 * q_descale
q_descale: (batch, heads) or scalar
k_descale: (batch, heads) or scalar
v_descale: (batch, heads) or scalar
```

The GEMM applies descaling during computation:
```
score = (Q * q_descale) @ (K * k_descale)^T / sqrt(d)
output = softmax(score) @ (V * v_descale)
```

### Parameters

```python
Flash_fwd_params {
    float *q_descale_ptr, *k_descale_ptr, *v_descale_ptr;
    index_t q_descale_batch_stride, q_descale_head_stride;
    index_t k_descale_batch_stride, k_descale_head_stride;
    index_t v_descale_batch_stride, v_descale_head_stride;
}
```

### Limitations

- Forward only in FA3 (backward uses fp16/bf16)
- Requires SM90+ (Hopper) or SM100+ (Blackwell) for native FP8 tensor cores
- E4M3 typically used for Q/K (forward), E5M2 for V and gradients

---

## PackGQA Optimization

### When PackGQA Helps

PackGQA is most beneficial when:
1. `qhead_per_khead` is large (e.g., 8 or 16 for GQA)
2. `seqlen_q` is small (e.g., 1 for generation)
3. Variable-length sequences (always packs for varlen)

### How It Works

1. Reshape Q from `(B, S_q, H_q, D)` to `(B, S_q, H_kv, G, D)` where `G = H_q / H_kv`
2. Treat the G dimension as part of the head dimension: `(B, S_q, H_kv, G*D)`
3. Run flash attention with the packed Q
4. Unpack the output back to `(B, S_q, H_q, D)`

### Tile Size Interaction

After packing, the effective head dimension is `G * D`. This affects tile sizes:
- `G=4, D=128` -> effective hdim=512: Uses hdim256 tile sizes with adjustments
- `G=2, D=128` -> effective hdim=256: Uses hdim256 tile sizes

---

## Inference-Specific Optimizations

### KV Cache Update (Append_KV)

During autoregressive generation, new K/V tokens are appended to the cache:

```python
flash_attn_with_kvcache(
    q,           # (B, 1, H, D) - single new query token
    cache_k,     # (B, max_seqlen, H_kv, D) - K cache
    cache_v,     # (B, max_seqlen, H_kv, D) - V cache
    k_new,       # (B, 1, H_kv, D) - new K token(s)
    v_new,       # (B, 1, H_kv, D) - new V token(s)
    cache_seqlens,  # (B,) - current cache lengths
    rotary_cos=cos, rotary_sin=sin,
)
```

The kernel:
1. Copies `k_new` and `v_new` into the cache at the positions indicated by `cache_seqlens`
2. Optionally applies rotary embeddings during the copy
3. Computes attention of the new Q against the full cached K/V

### CUDA Graph Support

The generation module supports CUDA graphs for low-latency inference:

```python
from flash_attn.utils.generation import decode, GenerationMixin

# Automatic CUDA graph capture for autoregressive decoding
output = decode(input_ids, model, max_length, cg=True)
```

### InferenceParams

```python
@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()
```

### allocate_inference_cache

```python
def allocate_inference_cache(max_batch_size, max_seqlen, nheads, headdim,
                             layers, device, dtype=torch.float16):
    kv_cache_shape = (max_batch_size, max_seqlen, 2, nheads, headdim)
    return {i: torch.empty(kv_cache_shape, device=device, dtype=dtype) for i in layers}
```
