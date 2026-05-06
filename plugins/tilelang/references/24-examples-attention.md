# 24. Attention Examples

This reference covers all attention mechanism examples in TileLang with complete source code
and detailed explanations. TileLang's attention implementations follow the Flash Attention
algorithm for memory-efficient, tiled computation.

---

## Table of Contents

1. [Multi-Head Attention Forward (BSHD)](#multi-head-attention-forward-bshd)
2. [Flash Attention with Causal Masking](#flash-attention-with-causal-masking)
3. [GQA (Grouped Query Attention)](#gqa-grouped-query-attention)
4. [Block-Sparse Attention](#block-sparse-attention)
5. [Linear Attention](#linear-attention)
6. [DeepSeek MLA Decode](#deepseek-mla-decode)
7. [DeepSeek NSA Forward](#deepseek-nsa-forward)
8. [Attention Sink Examples](#attention-sink-examples)
9. [AMD Flash Attention](#amd-flash-attention)
10. [Attention Backward Pass](#attention-backward-pass)
11. [Variable-Length Attention (Varlen)](#variable-length-attention-varlen)
12. [Attention Score Computation Patterns](#attention-score-computation-patterns)
13. [Softmax Optimization Techniques](#softmax-optimization-techniques)
14. [Output Accumulation Strategies](#output-accumulation-strategies)

---

## Multi-Head Attention Forward (BSHD)

The foundational Flash Attention implementation with BSHD (batch, sequence, heads, dim) tensor
layout. This implements the online softmax algorithm for memory-efficient attention.

### Complete Source Code

```python
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial


def get_configs():
    iter_params = dict(block_M=[64], block_N=[64], num_stages=[1], threads=[128])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn(batch, heads, seq_len, dim, is_causal,
              block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, bx * block_M:(bx+1)*block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N),
                      T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N:(k+1)*block_N, by, :], K_shared)

                # Apply causal mask
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j,
                            0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len,
                            -T.infinity(acc_s.dtype), 0)

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                # Online softmax: update running max and scale
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)

                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k * block_N:(k+1)*block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx+1)*block_M, by, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output


def main(batch=8, heads=32, seq_len=4096, dim=128, is_causal=False, tune=False):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if not tune:
        kernel = flashattn(batch, heads, seq_len, dim, is_causal,
                          block_M=128, block_N=128, num_stages=1, threads=128)
        ref_program_processed = partial(ref_program, is_causal=is_causal)
        profiler = kernel.get_profiler()
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(warmup=500)
        print(f"Tile-lang: {latency:.2f} ms")
        print(f"Tile-lang: {total_flops / latency * 1e-9:.2f} TFlops")
    else:
        best_result = flashattn(batch, heads, seq_len, dim, is_causal)
        print(f"Best latency: {best_result.latency}")
        print(f"Best config: {best_result.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.tune)
```

### Key Concepts Explained

**Online Softmax Algorithm**: Instead of computing the full attention matrix and then applying
softmax, Flash Attention maintains running statistics (max, sum) and rescales incrementally:

1. Compute `S_ij = Q_i @ K_j^T` (score block)
2. Update running max: `m_new = max(m_old, max(S_ij))`
3. Compute correction factor: `scale = exp(m_old - m_new)`
4. Rescale running sum and output: `O *= scale`
5. Apply softmax to current block: `P_ij = exp(S_ij - m_new)`
6. Update running sum: `l_new = l_old * scale + sum(P_ij)`
7. Accumulate output: `O += P_ij @ V_j`
8. Final normalization: `O /= l_final`

**Scale Factor**: `(1.0 / dim) ** 0.5 * 1.44269504` combines the standard attention scale
(`1/sqrt(d)`) with `log2(e)` to enable `exp2` instead of `exp` for faster computation.

**Warp Policy**: `GemmWarpPolicy.FullRow` distributes warps across the row dimension of the
score matrix, which is optimal for the subsequent reduction operations.

---

## Flash Attention with Causal Masking

Causal masking ensures that each query position can only attend to current and previous key
positions. The basic MHA example above already supports causal masking via the `is_causal`
parameter.

### Causal Loop Range Optimization

```python
# For causal attention, we only need to iterate over keys up to the current query position
loop_range = (
    T.min(T.ceildiv(seq_len, block_N),
          T.ceildiv((bx + 1) * block_M, block_N))
    if is_causal
    else T.ceildiv(seq_len, block_N)
)
```

This reduces computation by approximately 50% for causal attention since we skip future key
blocks entirely.

### Causal Mask Application

```python
if is_causal:
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.if_then_else(
            bx * block_M + i >= k * block_N + j,
            0,                    # Valid: query position >= key position
            -T.infinity(acc_s.dtype)  # Invalid: mask out future positions
        )
```

---

## GQA (Grouped Query Attention)

GQA shares KV heads across multiple query heads, reducing memory bandwidth requirements
during inference while maintaining model quality.

### Complete Source Code

```python
@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn(batch, heads, seq_len, dim, is_causal, groups=1,
              block_M=64, block_N=64, num_stages=0, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            # ... (same allocation as MHA) ...

            T.copy(Q[bz, bx * block_M:(bx+1)*block_M, by, :], Q_shared)
            # ... initialization ...

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Key difference: use by // groups to select KV head
                T.copy(K[bz, k * block_N:(k+1)*block_N, by // groups, :], K_shared)
                # ... causal masking and score computation ...

                # Use by // groups for V as well
                T.copy(V[bz, k * block_N:(k+1)*block_N, by // groups, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    return main


def ref_program(Q, K, V, is_causal, groups=1):
    # Q: [B, T, HQ, D], K: [B, T, HK, D], V: [B, T, HV, D]
    # HQ = HKV * groups
    assert Q.size(2) == K.size(2) * groups
    dim = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output
```

### Key Differences from MHA

1. **Separate KV shape**: `kv_shape = [batch, seq_len, head_kv, dim]` where `head_kv = heads // groups`
2. **KV head selection**: `by // groups` maps query head index to KV head index
3. **GQA with shared memory optimization**: Multiple query heads share the same KV data

---

## Block-Sparse Attention

Block-sparse attention skips computation for blocks determined to be unimportant, using
either a learned or heuristic sparsity pattern.

```python
@tilelang.jit(out_idx=[-1])
def block_sparse_attn(batch, heads, seq_len, dim, block_M, block_N,
                      block_size, grid_height, grid_width):
    # ... setup ...
    @T.prim_func
    def main(Q, K, V, BlockMask, Output):
        with T.Kernel(...) as (bx, by, bz):
            for k in T.Pipelined(loop_range, ...):
                # Only compute if block is in the sparse mask
                if BlockMask[bx, k]:
                    T.copy(K[...], K_shared)
                    # ... compute attention for this block ...
```

---

## Linear Attention

Linear attention replaces the softmax with a kernel function, reducing complexity from
O(N^2) to O(N) by changing the order of computation.

### Complete Source Code

```python
@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def tl_fused_chunk_fwd_kernel(B, S, H, DK, DV, dtype=T.float16, scale=None):
    if scale is None:
        scale = DK ** -0.5
    accum_dtype = T.float32

    chunk_size = 64
    BK = BV = 64
    NK = tilelang.cdiv(DK, BK)
    NV = tilelang.cdiv(DV, BV)
    NT = tilelang.cdiv(S, chunk_size)

    @T.prim_func
    def fused_chunk_linear_attn_fwd(
        Q: T.Tensor([B, S, H, DK], dtype),
        K: T.Tensor([B, S, H, DK], dtype),
        V: T.Tensor([B, S, H, DV], dtype),
        O: T.Tensor([B, S, H, DV], accum_dtype),
        final_state: T.Tensor([B, H, DK, DV], accum_dtype),
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            o_shared = T.alloc_shared([chunk_size, BV], accum_dtype)

            T.use_swizzle(10)
            T.clear(h)

            for i in T.Pipelined(0, NT):
                # Load Q with scaling
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h,
                                     i_k * BK + col] * scale
                T.copy(K[...], k)
                T.copy(V[...], v)

                # Intra-chunk: Q @ K^T (lower triangular for causality)
                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col], 0)

                # Intra-chunk output: S_lower @ V
                T.gemm(s_shared, v, o, clear_accum=True)

                # Inter-chunk: Q @ H (accumulated state)
                T.copy(h, h_shared)
                # Update state: H += K^T @ V
                T.gemm(k, v, h, transpose_A=True)
                # Add inter-chunk contribution
                T.gemm(q, h_shared, o)

                T.copy(o, o_shared)
                T.atomic_add(O[...], o_shared)

            # Output final state for decoding
            T.copy(h, final_state[i_b, i_h, i_k*BK:(i_k+1)*BK, i_v*BV:(i_v+1)*BV])

    return fused_chunk_linear_attn_fwd
```

### Key Concepts

**Chunk-Based Processing**: The sequence is divided into chunks of 64 tokens. Within each
chunk, attention is computed exactly (intra-chunk). Between chunks, a running state `H`
is maintained (inter-chunk).

**State Update**: `H = K^T @ V` (cumulative sum across chunks)
**Inter-chunk Output**: `O_inter = Q @ H`
**Intra-chunk Output**: `O_intra = (Q @ K^T)_lower @ V`
**Total Output**: `O = O_intra + O_inter`

**Atomic Addition**: Multiple thread blocks write to the same output tensor, requiring
`T.atomic_add` for correctness.

---

## DeepSeek MLA Decode

Multi-head Latent Attention (MLA) is DeepSeek's attention variant that compresses KV cache
into a low-rank latent representation for efficient inference.

### Complete Source Code

```python
@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim,
              block_N, block_H, num_split, softmax_scale):
    scale = float(softmax_scale * 1.44269504)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split], dtype)
        Output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)

        # Split-KV attention: divide KV sequence across splits
        with T.Kernel(batch, heads // VALID_BLOCK_H, num_split, threads=256) as (bid, hid, bz):
            # ... allocate shared/local buffers ...

            cur_kv_head = hid // (kv_group_num // block_H)
            T.use_swizzle(10)

            T.copy(Q[bid, hid * VALID_BLOCK_H:(hid+1)*VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H:(hid+1)*VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv // num_split, block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bid, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, kv_start:kv_end, cur_kv_head, :], K_pe_shared)

                # Compute scores from both content and position embeddings
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)

                # Online softmax ...
                # ...

        # Combine phase: merge split results
        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            # ... combine partial outputs using log-sum-exp ...
            for i in T.Parallel(dim):
                o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]

    if num_split > 1:
        return main_split
    else:
        return main_no_split
```

### Key Concepts

**Split-KV Processing**: For long KV sequences, the KV cache is divided across multiple
splits. Each split computes partial attention, then results are combined via log-sum-exp.

**Dual Attention Scores**: MLA computes attention from both content embeddings (Q @ KV^T)
and positional embeddings (Q_pe @ K_pe^T), combining them additively.

**Position Encoding**: Separate `pe_dim` handles RoPE (Rotary Position Embedding) separately
from the content dimension.

---

## DeepSeek NSA Forward

Native Sparse Attention (NSA) uses learned block indices to attend only to selected key-value
blocks, achieving hardware-efficient sparse attention.

### Complete Source Code (Key Parts)

```python
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def native_sparse_attention(batch, heads, seq_len, dim, is_causal, scale=None,
                            block_size=64, groups=1, selected_blocks=16):
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        scale = scale * 1.44269504

    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    @T.prim_func
    def native_sparse_attention(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, NV, batch * head_kv, threads=32) as (bx, by, bz):
            # ... allocate buffers ...

            cur_kv_head = bz % head_kv
            i_b = bz // head_kv
            i_h_q_start = (bz % head_kv) * groups
            i_h_q_end = i_h_q_start + groups

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for k in T.serial(selected_blocks):
                # Get the actual KV block index from the sparse indices
                block_idx = BlockIndices[i_b, bx, cur_kv_head, k]
                kv_start = block_idx * block_S

                # Load only the selected KV block
                T.copy(K[i_b, kv_start:kv_start+block_S, cur_kv_head, :], K_shared)
                T.copy(V[i_b, kv_start:kv_start+block_S, cur_kv_head, :], V_shared)

                # Compute attention scores
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol)

                # Online softmax update ...
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

    return native_sparse_attention
```

### Key Concepts

**Block Selection**: Instead of attending to all KV blocks, NSA uses `BlockIndices` to select
only `selected_blocks` blocks per query position.

**Gather Pattern**: `block_idx = BlockIndices[i_b, bx, cur_kv_head, k]` provides indirect
addressing into the KV cache.

**Efficiency**: For `selected_blocks=16` with `block_size=64`, each query only attends to
1024 tokens regardless of total sequence length.

---

## Attention Sink Examples

Attention sinks handle special tokens (e.g., the first token) that receive disproportionate
attention, common in streaming language models.

```python
# Attention sink kernels handle the case where the first few tokens
# must always be attended to, regardless of the KV cache eviction policy.
# See examples/attention_sink/ for full implementations:
# - example_mha_sink_fwd_bhsd.py: MHA with sink tokens
# - example_gqa_sink_fwd_varlen.py: GQA with variable-length sequences
# - example_gqa_sink_bwd_bhsd.py: Backward pass with sink attention
```

---

## AMD Flash Attention

Flash attention implementation for AMD GPUs using the ROCm/HIP backend.

### Key Adaptations

```python
def supply_tensors_gpu(params):
    """Supply function that creates tensors on GPU for ROCm/HIP."""
    tensors = []
    for param in params:
        if hasattr(param, "shape") and hasattr(param, "dtype"):
            shape = [int(s) for s in param.shape]
            torch_dtype = param.dtype.as_torch()
            tensor = torch.randn(shape, dtype=torch_dtype, device="cuda")
            tensors.append(tensor)
        else:
            tensors.append(param)
    return tensors


# Use CDNA architecture for roller hints
from tilelang.carver.arch import CDNA
arch = CDNA("hip")
```

### AMD-Specific Considerations

1. **MFMA instructions**: AMD uses Matrix Fused Multiply-Add instead of MMA
2. **Shared memory**: Different bank conflict patterns than NVIDIA
3. **Warp size**: AMD wavefront size is 64 (vs NVIDIA warp size of 32)
4. **Backend**: Use `target="hip"` for AMD compilation

---

## Attention Backward Pass

The attention backward pass computes gradients for Q, K, V given the output gradient.

```python
@tilelang.jit(out_idx=[3, 4])
def flashattn_bwd(batch, heads, seq_len, dim, is_causal, block_M, block_N):
    # Forward pass (recomputed to avoid storing full attention matrix)
    # Then backward:
    # dV = S^T @ dO  (where S = softmax(QK^T))
    # dK = dS @ Q
    # dQ = dS @ K
    # dS = softmax_backward(dO, O, S)
```

The backward pass recomputes attention scores (forward pass) to avoid materializing the
full N^2 attention matrix, following the Flash Attention 2 approach.

---

## Variable-Length Attention (Varlen)

Variable-length attention handles sequences of different lengths packed into contiguous
tensors, using cumulative sequence length arrays for indexing.

### Complete Source Code (Key Parts)

```python
@autotune(configs=get_configs())
@tilelang.jit(
    out_idx=[6],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def flashattn(batch_size, UQ, UKV, heads, dim, is_causal,
              block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    q_shape = [UQ, heads, dim]      # Unpadded total Q length
    k_shape = [UKV, heads, dim]     # Unpadded total KV length
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(k_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size,
                     threads=threads) as (bx, by, bz):
            # Get actual sequence boundaries
            q_start = cu_seqlens_q[bz]
            q_end = cu_seqlens_q[bz + 1]
            k_start = cu_seqlens_k[bz]
            k_end = cu_seqlens_k[bz + 1]
            q_len = q_end - q_start
            k_len = k_end - k_start

            # Bounds check
            if bx * block_M >= q_len:
                return

            # Load Q from unpadded position
            actual_block_M = T.min(block_M, q_len - bx * block_M)
            T.copy(Q_unpad[q_start + bx * block_M:q_start + (bx+1)*block_M, by, :], Q_shared)

            # Iterate over KV blocks
            for k in T.Pipelined(T.ceildiv(k_len, block_N), num_stages=num_stages):
                T.copy(K_unpad[k_start + k * block_N:k_start + (k+1)*block_N, by, :], K_shared)
                # ... attention computation ...
```

### Key Concepts

- **Cumulative sequence lengths**: `cu_seqlens_q[b]` gives the starting offset of sequence `b`
- **Unpadded tensors**: All sequences are packed contiguously without padding
- **Bounds checking**: Each block checks if it falls within its sequence's actual length
- **Autotune with `set_autotune_inputs`**: Variable-length inputs require custom tensor supply

---

## Attention Score Computation Patterns

### Standard Dot-Product Attention

```python
# S = Q @ K^T / sqrt(d)
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
# Apply scale in softmax computation
```

### Additive Attention (DeepSeek MLA)

```python
# S = Q_content @ K_content^T + Q_pe @ K_pe^T
T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True)
T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True)  # Additive
```

### Linear Attention (No Softmax)

```python
# Feature map: phi(x) = elu(x) + 1
# O_i = sum_j(phi(q_i) * phi(k_j) * v_j^T) / sum_j(phi(q_i) * phi(k_j))
# = phi(q_i) @ (sum_j phi(k_j) @ v_j^T) / phi(q_i) @ (sum_j phi(k_j))
# = phi(q_i) @ H / phi(q_i) @ h
```

---

## Softmax Optimization Techniques

### Log-Domain Softmax (Flash Attention)

```python
# Use log2 instead of natural log for faster exp computation
scale = (1.0 / dim) ** 0.5 * 1.44269504  # 1.44269504 = log2(e)

# Use exp2 instead of exp
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
```

### Online Softmax Rescaling

```python
# Rescale previous accumulations when max changes
for i in T.Parallel(block_M):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] *= scores_scale[i]
```

### Numerically Stable Max Tracking

```python
# Always maintain running max
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
for i in T.Parallel(block_M):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
```

---

## Output Accumulation Strategies

### Direct Accumulation (Flash Attention)

```python
# Accumulate into FP32 fragment
acc_o = T.alloc_fragment([block_M, dim], T.float32)
# ... for each KV block ...
T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
# Final normalization
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] /= logsum[i]
```

### Split-KV Accumulation (MLA Decode)

```python
# Split KV sequence, compute partial results, then combine
# Phase 1: Compute partial outputs per split
Output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)
glse = T.alloc_global([batch, heads, num_split], dtype)

# Phase 2: Combine using log-sum-exp
lse_max = max(glse[...])
for k in T.serial(num_split):
    scale = exp2(glse[k] - lse_max)
    Output += Output_partial[k] * scale
```

### Atomic Accumulation (Linear Attention)

```python
# Multiple thread blocks write to the same output
T.atomic_add(O[bz, i * chunk_size:(i+1)*chunk_size, i_h, i_v*BV:(i_v+1)*BV], o_shared)
```
