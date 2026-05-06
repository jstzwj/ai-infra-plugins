# 13 - CUDA Kernels Reference

This document provides an exhaustive reference for all CUDA kernels in FlashAttention, covering the FA2 C++/CUDA kernels in `csrc/flash_attn/src/`, the FA3 Hopper kernels in `hopper/`, layer normalization kernels, fused dense kernels, and all supporting device-side operations.

---

## Table of Contents

1. [Overview](#overview)
2. [FA2 Forward Kernel Architecture](#fa2-forward-kernel-architecture)
3. [FA2 Backward Kernel Architecture](#fa2-backward-kernel-architecture)
4. [Kernel Traits and Configurations](#kernel-traits-and-configurations)
5. [Launch Templates](#launch-templates)
6. [Softmax Operations](#softmax-operations)
7. [Mask Operations](#mask-operations)
8. [Rotary Embedding Operations](#rotary-embedding-operations)
9. [ALiBi Implementation](#alibi-implementation)
10. [Dropout Implementation](#dropout-implementation)
11. [Block Information](#block-information)
12. [Layer Normalization CUDA Kernels](#layer-normalization-cuda-kernels)
13. [Fused Dense CUDA Kernels](#fused-dense-cuda-kernels)
14. [Hopper SM80 Forward Kernels](#hopper-sm80-forward-kernels)
15. [Hopper SM90 Forward Kernels](#hopper-sm90-forward-kernels)
16. [Hopper SM80 Backward Kernels](#hopper-sm80-backward-kernels)
17. [Hopper SM90 Backward Kernels](#hopper-sm90-backward-kernels)
18. [Split-KV Parallelism](#split-kv-parallelism)
19. [Paged KV Cache Manager](#paged-kv-cache-manager)
20. [Tile Size Heuristics](#tile-size-heuristics)
21. [Utility Functions](#utility-functions)
22. [Generated Kernel Instantiations](#generated-kernel-instantiations)
23. [Static Switch Macros](#static-switch-macros)

---

## Overview

FlashAttention implements attention computation through a family of tiled CUDA kernels. The core idea is to compute attention in tiles (blocks) to keep the working set in fast GPU shared memory (SRAM), avoiding the materialization of the full N x N attention matrix in high-bandwidth memory (HBM).

### Kernel Hierarchy

| Generation | Location | Architecture | Features |
|------------|----------|-------------|----------|
| FA2 | `csrc/flash_attn/src/` | SM80+ | cp_async, WGMMA (SM80 MMA) |
| FA3 | `hopper/` | SM80/SM90 | TMA (SM90), WGMMA, warpgroup MMA |
| FA4 | `flash_attn/cute/` | SM90/SM100 | CuTeDSL, 2CTA, persistent |

### Key Abstractions

- **Tiling**: The Q sequence is divided into blocks of `kBlockM` rows, and the K/V sequences are divided into blocks of `kBlockN` rows. Each CUDA thread block processes one Q block against all K/V blocks.
- **Online Softmax**: Scores are accumulated using the online softmax algorithm to avoid materializing the full attention matrix.
- **Register Tiling**: Data flows through HBM -> shared memory -> registers -> tensor cores, with explicit management at each level.
- **Pipeline**: Asynchronous memory copies (cp_async for SM80, TMA for SM90) overlap data movement with computation.

---

## FA2 Forward Kernel Architecture

### File: `csrc/flash_attn/src/flash_fwd_kernel.h`

The FA2 forward kernel implements the standard attention computation:

```
O = softmax(Q * K^T / sqrt(d)) * V
```

### `compute_attn_1rowblock`

This is the core forward computation function, processing one row block of Q against all column blocks of K/V:

```cpp
template<typename Kernel_traits,
         bool Is_dropout, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K,
         bool Is_softcap, bool Return_softmax,
         typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params &params,
    const int bidb,      // batch index
    const int bidh,      // head index
    const int m_block    // Q block index
);
```

#### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `Kernel_traits` | Contains block sizes, element types, layout types, copy atoms |
| `Is_dropout` | Whether dropout is applied (uses Philox RNG) |
| `Is_causal` | Whether causal masking is applied |
| `Is_local` | Whether local (sliding window) masking is applied |
| `Has_alibi` | Whether ALiBi positional bias is applied |
| `Is_even_MN` | Whether sequence lengths are multiples of block sizes |
| `Is_even_K` | Whether head dimension matches the kernel's compile-time `kHeadDim` |
| `Is_softcap` | Whether softcapping (tanh) is applied to scores |
| `Return_softmax` | Whether to return the softmax probability matrix |
| `Params` | The `Flash_fwd_params` struct with all pointers and strides |

#### Algorithm

1. **Initialize**: Set up dropout RNG, load block info for variable-length sequences, compute block ranges.

2. **Early Exit**: If the Q block is beyond the actual sequence length, or if causal/local masking means no valid K/V blocks exist, write zeros and return.

3. **Prologue**: Load the Q tile from HBM to shared memory using async copy. Optionally copy Q to registers if `Is_Q_in_regs` is true.

4. **Masking Loop** (reverse iteration): For the last `n_masking_steps` blocks of K/V:
   - Wait for K to arrive in shared memory
   - Load V tile (with boundary masking for the last block)
   - Prefetch next K block if not the last iteration
   - Compute QK^T using MMA (matrix multiply-accumulate)
   - Apply softcap if enabled: `scores = softcap * tanh(scores / softcap)`
   - Apply causal/local/general masking
   - Online softmax rescale with `softmax_rescale_o`
   - Convert scores to element type (fp16/bf16)
   - Apply dropout if enabled
   - Compute P * V using register-source GEMM (`gemm_rs`)

5. **Main Loop** (no masking needed): Same as above but without masking operations.

6. **Epilogue**: Normalize softmax, convert output from fp32 to fp16/bf16, write through shared memory to HBM, write log-sum-exp (LSE) values.

### `compute_attn_1rowblock_splitkv`

Implements the forward pass with split-KV parallelism, where different thread blocks handle different ranges of K/V:

```cpp
template<typename Kernel_traits,
         bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap,
         bool Split, bool Append_KV,
         typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(
    const Params &params,
    const int bidb, const int bidh,
    const int m_block, const int n_split_idx,
    const int num_n_splits
);
```

The SplitKV path adds:
- **Paged KV Support**: Uses `block_table` for page table lookups
- **KV Cache Append**: Copies new K/V into cache, optionally applying rotary embeddings
- **Rotary Embedding on Q**: Applies rotary embeddings when loading Q if `Append_KV` and `rotary_dim > 0`
- **Partial Accumulation**: Writes partial O and LSE to accumulation buffers

### `compute_attn`

The top-level entry point that extracts block indices and delegates:

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal,
         bool Is_local, bool Has_alibi, bool Is_even_MN,
         bool Is_even_K, bool Is_softcap, bool Return_softmax,
         typename Params>
inline __device__ void compute_attn(const Params &params);
```

- `blockIdx.x` = m_block (Q block index)
- `blockIdx.y` = bidb (batch index)
- `blockIdx.z` = bidh (head index)

### `compute_attn_splitkv`

Top-level for SplitKV path:
- When `Split=true`: `blockIdx.y` = split index, `blockIdx.z` encodes batch and head
- When `Split=false`: Same mapping as `compute_attn`

### `combine_attn_seqk_parallel`

Combines partial results from SplitKV parallel processing:

```cpp
template<typename Kernel_traits, int kBlockM, int Log_max_splits,
         bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params);
```

Algorithm:
1. Load per-split LSE values into shared memory
2. Compute log-sum-exp of all splits: `lse_logsum = log(sum(exp(lse_i - lse_max))) + lse_max`
3. Scale each split's output by `exp(lse_i - lse_logsum)`
4. Accumulate scaled outputs and write final result to HBM

---

## FA2 Backward Kernel Architecture

### File: `csrc/flash_attn/src/flash_bwd_kernel.h`

The backward kernel computes gradients for Q, K, and V given the gradient of the output (dO).

### `compute_dq_dk_dv_1colblock`

Core backward computation for one column block of K/V:

```cpp
template<typename Kernel_traits,
         bool Is_dropout, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K,
         bool Is_softcap, bool Is_first, bool Is_last,
         bool Seq_parallel=false,
         typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(
    const Params &params,
    const int bidb, const int bidh, const int n_block
);
```

#### Template Parameters

| Parameter | Description |
|-----------|-------------|
| `Is_first` | True for the first (highest-indexed) n_block |
| `Is_last` | True for the last (n_block=0) block |
| `Seq_parallel` | True for split-KV parallel backward |

#### Algorithm

1. **Initialize**: Set up shared memory layouts for Q, dO, K, V, dS, P, dQ, dK, dV. Zero accumulators for dK and dV.

2. **Main Loop** (iterating m_block from max to min):
   - Load Q, dO blocks (with double-buffering if enabled)
   - Compute S = Q @ K^T (forward scores)
   - Apply softcap: `scores = softcap * tanh(scores / softcap)`
   - Compute `dtanh = 1 - tanh(scores/softcap)^2` for softcap backward
   - Apply masking (causal, local, boundary)
   - Recover forward probabilities: `P = exp(S - LSE) * scale`
   - Compute dP = dO @ V
   - Compute dS = P * (dP - dP_sum) [where dP_sum = sum(dO * O)]
   - Apply softcap gradient: `dS *= dtanh`
   - Compute dV = P^T @ dO
   - Compute dK = dS^T @ Q
   - Compute dQ = dS @ K^T
   - Store/accumulate dQ

3. **Epilogue**: Convert dK, dV from fp32 to fp16/bf16, write to HBM.

### `compute_dq_dk_dv`

Top-level backward function that iterates over all n_blocks:

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal,
         bool Has_alibi, bool Is_even_M, bool Is_even_K,
         typename Params>
inline __device__ void compute_dq_dk_dv(const Params &params);
```

- If there is only 1 n_block: single call to `compute_dq_dk_dv_1colblock` with `Is_first=true, Is_last=true`
- Otherwise: first block (`Is_first=true, Is_last=false`), middle blocks (`Is_first=false, Is_last=false`), last block (`Is_first=false, Is_last=true`)

### `compute_dq_dk_dv_seqk_parallel`

For deterministic backward with split-KV parallelism:

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal,
         bool Is_local, bool Has_alibi, bool Is_even_MN,
         bool Is_even_K, bool Is_softcap,
         typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params);
```

Each thread block handles specific n_blocks using `atomicAdd` to accumulate dQ from multiple splits.

---

## Kernel Traits and Configurations

### File: `csrc/flash_attn/src/kernel_traits.h`

### `Flash_kernel_traits` (Base)

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {
    using Element = elem_type;        // fp16 or bf16
    using ElementAccum = float;       // Accumulation in fp32
    using index_t = int64_t;
    static constexpr bool Has_cp_async = (__CUDA_ARCH__ >= 800);

    // SM80 tensor core MMA atom
    using MMA_Atom_Arch = conditional_t<
        is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;

    // SM75+ LDSM (load shared memory) copy atoms
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
};
```

### `Flash_fwd_kernel_traits`

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false,
         typename elem_type=cutlass::half_t, ...>
struct Flash_fwd_kernel_traits : public Flash_kernel_traits<...> {
```

#### Key Constants

| Constant | Description | Value |
|----------|-------------|-------|
| `kBlockM` | Q tile rows | 64, 128, or 256 |
| `kBlockN` | K/V tile rows | 32, 64, 128, or 256 |
| `kHeadDim` | Head dimension | 32, 64, 96, 128, 192, or 256 |
| `kNWarps` | Number of warps | 4 or 8 |
| `kNThreads` | Total threads | `kNWarps * 32` |
| `kBlockKSmem` | K tile in smem | 64 if `kHeadDim % 64 == 0` else 32 |
| `kBlockKGmem` | K tile in gmem | 128, 64, or 32 depending on `kHeadDim` |
| `kSwizzle` | Swizzle pattern | 3 for 64-wide, 2 for 32-wide |
| `Share_Q_K_smem` | Q and K/V share smem | false (default) |
| `Is_Q_in_regs` | Keep Q in registers | false or forced by `Share_Q_K_smem` |

#### Memory Layouts

- **SmemLayoutQ**: `(kBlockM, kHeadDim)` with swizzle for bank conflict avoidance
- **SmemLayoutKV**: `(kBlockN, kHeadDim)` same atom as Q
- **SmemLayoutVtransposed**: `(kHeadDim, kBlockN)` logical transpose of V layout
- **SmemLayoutO**: `(kBlockM, kHeadDim)` same as Q layout

#### Shared Memory Size

```cpp
kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);  // kBlockM * kHeadDim * 2
kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);  // 2 * kBlockN * kHeadDim * 2
kSmemSize = Share_Q_K_smem ? max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
```

For `kBlockM=128, kBlockN=64, kHeadDim=128, 4 warps`:
- `kSmemQSize = 128 * 128 * 2 = 32 KB`
- `kSmemKVSize = 64 * 128 * 2 * 2 = 32 KB`
- `kSmemSize = 64 KB`

#### Tiled MMA

```cpp
using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>, _1, _1>>,   // Warp layout
    Tile<Int<16 * kNWarps>, _16, _16>      // Tile shape
>;
```

Each warp computes a 16x16 tile of the output. With 4 warps, the effective M dimension is 64.

### `Flash_bwd_kernel_traits`

```cpp
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false,
         typename elem_type=cutlass::half_t, ...>
struct Flash_bwd_kernel_traits : public Flash_kernel_traits<...> {
```

#### Additional Constants

| Constant | Description |
|----------|-------------|
| `AtomLayoutMSdP` | Warp group layout for S and dP computation |
| `AtomLayoutNdKV` | Warp group layout for dK and dV computation |
| `AtomLayoutMdQ` | Warp group layout for dQ computation |
| `Is_V_in_regs` | Keep V in registers to reduce smem usage |
| `No_double_buffer` | Disable double-buffering of Q |

#### Three Separate TiledMMA Operations

1. **TiledMmaSdP**: Computes `S = Q @ K^T` and `dP = dO @ V`
2. **TiledMmadKV**: Computes `dK = dS^T @ Q` and `dV = P^T @ dO`
3. **TiledMmadQ**: Computes `dQ = dS @ K^T`

#### Shared Memory Layout for Backward

The backward kernel requires significantly more shared memory:

- `sQ`, `sdO`: Double-buffered, 2-3 copies (current, next, and potentially one more for overlap)
- `sK`, `sV`: One copy each
- `sdS`, `sP`: One copy each (can share space with `sdQ`)

---

## Launch Templates

### File: `csrc/flash_attn/src/flash_fwd_launch_template.h`

### `run_flash_fwd`

Launches the standard (non-SplitKV) forward kernel:

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream);
```

**Grid dimensions**: `(num_m_block, params.b, params.h)` where `num_m_block = ceil(seqlen_q / kBlockM)`.

**Launch process**:
1. Compute `smem_size` from `Kernel_traits::kSmemSize`
2. If `smem_size >= 48KB`, request expanded shared memory via `cudaFuncSetAttribute`
3. Dispatch through a chain of compile-time boolean switches:
   - `IsEvenMNConst`: Whether sequence lengths are even multiples
   - `IsEvenKConst`: Whether head dim is exactly `kHeadDim`
   - `Is_local`: Whether sliding window masking
   - `ReturnSoftmaxConst`: Whether to return P matrix
   - `Has_alibi`: Whether ALiBi bias
   - `Is_softcap`: Whether softcap is applied

### `run_flash_splitkv_fwd`

Launches the SplitKV forward kernel:

```cpp
template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream);
```

**Grid dimensions**:
- When `num_splits > 1`: `(num_m_block, num_splits, b * h)`
- When `num_splits == 1`: `(num_m_block, b, h)`

After the main kernel, if `num_splits > 1`, a combine kernel is launched to merge partial results.

### Head-Dimension Dispatch Functions

```cpp
// For each supported head dimension:
template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream);
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream);
```

Each function selects optimal block sizes based on the architecture:

| Head Dim | SM80 Block (M x N) | SM86/SM89 Block (M x N) | Notes |
|----------|--------------------|-----------------------|-------|
| 32 | 128 x 128 | 128 x 128 | 4 warps |
| 64 | 128 x 128 | 128 x 128 | 4 warps; dropout uses 128 x 64 |
| 96 | 128 x 64 | 64 x 64 (causal) | SM86 prefers square tiles for causal |
| 128 | 128 x 64 | 128 x 32 (non-causal), 64 x 64 (causal) | SM86 optimization for occupancy |
| 192 | 128 x 64, 8 warps | same | 8 warps needed for register pressure |
| 256 | 128 x 64 or 64 x 64 | same | Depends on available smem |

### `run_mha_fwd_splitkv_dispatch`

```cpp
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);
```

Block sizes for SplitKV:
- `kBlockM = 64` always
- `kBlockN = 256` for hdim <= 64, `128` for hdim <= 128, `64` otherwise
- When `num_splits == 1`: Uses smaller `kBlockN` to match standard kernel numerics

---

## Softmax Operations

### File: `csrc/flash_attn/src/softmax.h`

### `Softmax<kNRows>` Class

Implements online softmax with row-wise max tracking and exponential scaling:

```cpp
template <int kNRows>
struct Softmax {
    TensorT row_max, row_sum;  // Per-row max and sum trackers

    // First or subsequent iteration softmax
    template<bool Is_first, bool Check_inf, typename Tensor0, typename Tensor1>
    void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2);

    // Final normalization
    template<bool Is_dropout, bool Split, typename Tensor0>
    TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0);
};
```

### `softmax_rescale_o`

Implements the online softmax algorithm:

**First iteration** (`Is_first=true`):
1. Compute row max: `row_max = max(scores)`
2. Apply exp2 scaling: `scores = exp2(scores * scale - row_max * scale)`
3. Compute row sum: `row_sum = sum(scores)`

**Subsequent iterations** (`Is_first=false`):
1. Save previous max: `scores_max_prev = row_max`
2. Update row max: `row_max = max(row_max, max(scores))`
3. Compute rescale factor: `scores_scale = exp2((scores_max_prev - row_max) * scale)`
4. Rescale output accumulator: `acc_o *= scores_scale`
5. Rescale running sum: `row_sum *= scores_scale`
6. Apply exp2 and update sum

The `Check_inf` parameter handles cases where all scores in a row are -infinity (fully masked rows).

### `scale_apply_exp2`

Applies `exp2(scores * scale - max_scaled)` element-wise:

```cpp
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor,
                      Tensor<Engine1, Layout1> const &max, const float scale);
```

Uses `exp2f` instead of `expf` to enable FMA instruction fusion. When max is -inf (all masked), the scaled max is set to 0 to avoid NaN.

### `normalize_softmax_lse`

Final normalization:
1. All-reduce row sums across the quad (4 threads)
2. Compute LSE: `lse = row_max * scale + log(sum)` (or +/- infinity for edge cases)
3. Scale output: `acc_o *= (1 / sum) * rp_dropout` (if dropout)

### Reduction Primitives

```cpp
// Thread-level reduction across columns
template<bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op);

// Quad-level (4-thread) all-reduce
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op);

// Combined thread + quad reduce for max
template<bool zero_init, ...>
void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max);

// Thread-only reduce for sum (no cross-thread reduce needed until final)
template<bool zero_init, ...>
void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum);
```

---

## Mask Operations

### File: `csrc/flash_attn/src/mask.h`

### `Mask<Is_causal, Is_local, Has_alibi>` Struct

Combines causal, local, and ALiBi masking into a single operation:

```cpp
template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    void apply_mask(Tensor<Engine, Layout> &tensor,
                    const int col_idx_offset,
                    const int row_idx_offset,
                    const int warp_row_stride);
};
```

### Standalone Mask Functions

```cpp
// General sequence length masking
template <typename Engine, typename Layout>
void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                const int col_idx_offset = 0);

// Local (sliding window) masking
template <bool HasWSLeft=true, typename Engine, typename Layout>
void apply_mask_local(Tensor<Engine, Layout> &tensor, const int col_idx_offset,
                      const int max_seqlen_k, const int row_idx_offset,
                      const int max_seqlen_q, const int warp_row_stride,
                      const int window_size_left, const int window_size_right);

// Causal masking (delegates to apply_mask_local with window_size_left=-1, window_size_right=0)
template <typename Engine, typename Layout>
void apply_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset,
                       const int max_seqlen_k, const int row_idx_offset,
                       const int max_seqlen_q, const int warp_row_stride);

// Causal masking with precomputed row/column indices
template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
void apply_mask_causal_w_idx(Tensor<Engine0, Layout0> &tensor,
                             Tensor<Engine1, Layout1> const &idx_rowcol,
                             const int col_idx_offset, const int max_seqlen_k,
                             const int row_idx_offset);
```

### Masking Logic

For each element at `(row_idx, col_idx)` in the score matrix:

**Causal**: Mask if `col_idx >= row_idx + seqlen_k - seqlen_q + 1`

**Local (Sliding Window)**: Mask if:
- `col_idx >= row_idx + seqlen_k - seqlen_q + window_size_right + 1` (right boundary)
- `col_idx < row_idx + seqlen_k - seqlen_q - window_size_left` (left boundary)

**Sequence Length**: Mask if `col_idx >= actual_seqlen_k`

Masked elements are set to `-INFINITY` to produce zero probability after softmax.

### SM90 R2P Masking Optimization

On SM90 (Hopper), the `R2P` (Register-to-Predicate) instruction converts bitmask bytes into predicate registers efficiently:

- Each `R2P` instruction maps 7 bits of a register byte to 7 predicate registers
- For 32 accumulator elements: 4 R2P instructions cover 28 elements, with 4 handled by LOP3/ISETP
- Performance improvement: ~1% for causal, ~7-15% for local masking (where masking is a larger fraction of work)

---

## Rotary Embedding Operations

### File: `csrc/flash_attn/src/rotary.h`

Two implementations for different rotary embedding formats:

### Interleaved Format

```cpp
template<bool Is_even_K=true, bool Clear_OOB_K=true, ...>
void copy_rotary_interleaved(
    Tensor<Engine0, Layout0> const &S,   // Source (Q or K from gmem)
    Tensor<Engine1, Layout1> &D,          // Destination (smem)
    Tensor<Engine2, Layout2> const &Cos,  // Cosine values
    Tensor<Engine2, Layout2> const &Sin,  // Sine values
    Tensor<Engine3, Layout3> const &identity_MN,
    const int max_MN, const int min_MN,
    const int dim, const int rotary_dim);
```

Interleaved format pairs `(x0, x1, x2, x3, ...)` with rotation applied to consecutive pairs:
```
real = x[2i] * cos(i) - x[2i+1] * sin(i)
imag = x[2i] * sin(i) + x[2i+1] * cos(i)
```

### Contiguous (Half-Rotation) Format

```cpp
template<bool Is_even_K=true, bool Clear_OOB_K=true, ...>
void copy_rotary_contiguous(S, D, Cos, Sin, identity_MN, max_MN, min_MN, dim, rotary_dim);
```

Contiguous format has the first half and second half of dimensions rotated:
```
out[i] = x[i] * cos(i) + x[i + rotary_dim/2] * (-sin(i))   // for i < rotary_dim/2
out[i] = x[i] * cos(i) + x[i - rotary_dim/2] * sin(i)      // for i >= rotary_dim/2
```

Both functions:
- Only apply rotary to dimensions `< rotary_dim`
- Pass through dimensions `>= rotary_dim` unchanged
- Handle boundary conditions for variable sequence lengths
- Compute in fp32 for numerical precision, then convert back to fp16/bf16

---

## ALiBi Implementation

### File: `csrc/flash_attn/src/alibi.h`

### `Alibi<Is_causal>` Struct

```cpp
template <bool Is_causal>
struct Alibi {
    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    template <typename Engine, typename Layout>
    void apply_alibi(Tensor<Engine, Layout> &tensor,
                     const int col_idx_offset,
                     const int row_idx_offset,
                     const int warp_row_stride);
};
```

**Causal mode**: Adds `alibi_slope * col_idx` to all elements in each column (independent of row).

**Non-causal mode**: Subtracts `alibi_slope * |row_idx + seqlen_k - seqlen_q - col_idx|` (symmetric distance-based bias).

The `alibi_slope` is pre-divided by `scale_softmax` before being passed to the kernel.

---

## Dropout Implementation

### File: `csrc/flash_attn/src/dropout.h`

### `Dropout` Struct

```cpp
struct Dropout {
    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    void apply_dropout(Tensor<Engine, Layout> &tensor,
                       int block_row_start, int block_col_start,
                       int block_row_stride);
};
```

#### Philox RNG

Uses the Philox counter-based RNG to generate deterministic dropout patterns:
- Key: `(seed, offset + (bid * nheads + hid) * 32 + tid % 32)`
- Counter: `(block_row_start, block_col_start)` position in attention matrix

This ensures forward and backward generate identical dropout patterns regardless of thread count or traversal order.

#### Dropout Application

For fp16/bf16, uses efficient 16-bit comparison:
```cpp
asm volatile("set.le.u32.f16x2 %0, %1, %2;" : "=r"(mask) : "r"(rnd), "r"(threshold));
tensor_uint32(i) &= mask;
```

For other types or when `encode_dropout_in_sign_bit=true` (for returning P matrix):
- Compare random byte with threshold
- Keep or zero/negate the element

#### Dropout Scaling

- Forward: `output *= (1 / (1 - p_dropout))` (inverted dropout)
- Backward: `dV *= (1 - p_dropout)`, `dK *= scale_softmax * (1 - p_dropout)`, `dQ *= scale_softmax * (1 / (1 - p_dropout))`

---

## Block Information

### File: `csrc/flash_attn/src/block_info.h`

### `BlockInfo<Varlen>` Struct

```cpp
template<bool Varlen=true>
struct BlockInfo {
    const int sum_s_q;         // Cumulative offset for Q
    const int sum_s_k;         // Cumulative offset for K
    const int actual_seqlen_q; // Actual Q sequence length for this batch
    const int leftpad_k;       // Left padding for K
    const int seqlen_k_cache;  // K cache length (excluding new KV)
    const int actual_seqlen_k; // Total K length (cache + new)

    template <typename Params>
    __device__ BlockInfo(const Params &params, const int bidb);

    template <typename index_t>
    __device__ index_t q_offset(index_t batch_stride, index_t row_stride, int bidb) const;

    template <typename index_t>
    __device__ index_t k_offset(index_t batch_stride, index_t row_stride, int bidb) const;
};
```

Handles:
- Variable-length sequences via `cu_seqlens_q/k` arrays
- Left padding for K sequences
- KV cache with new key/value appending
- Non-cumulative seqlen arrays (direct length storage)

---

## Layer Normalization CUDA Kernels

### Directory: `csrc/layer_norm/`

FlashAttention includes optimized layer normalization kernels that are specialized per hidden size. This avoids runtime branching and enables optimal memory access patterns.

### Kernel Traits

File: `csrc/layer_norm/ln_kernel_traits.h`

```cpp
template<uint32_t HIDDEN_SIZE_, typename weight_t_, typename input_t_,
         typename residual_t_, typename output_t_, typename compute_t_,
         typename index_t_, uint32_t THREADS_PER_CTA_>
struct Kernel_traits_base {
    using weight_t = weight_t_;      // fp16 or bf16
    using input_t = input_t_;        // fp16 or bf16
    using residual_t = residual_t_;  // fp16 or bf16
    using output_t = output_t_;      // fp16 or bf16
    using compute_t = compute_t_;    // fp32
    using index_t = index_t_;        // int32_t
    enum { HIDDEN_SIZE = HIDDEN_SIZE_ };
    enum { THREADS_PER_CTA = THREADS_PER_CTA_ };
};
```

### Per-Hidden-Size Specializations

Each hidden size gets its own compilation unit:

| File Pattern | Hidden Sizes |
|-------------|-------------|
| `ln_fwd_*.cu` | 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192 |
| `ln_bwd_*.cu` | Same as above |
| `ln_parallel_fwd_*.cu` | Same as above (parallel residual) |
| `ln_parallel_bwd_*.cu` | Same as above |

### Forward Kernel Operations

The layer norm forward computes:
```
mean = mean(x, dim=-1)
var = var(x, dim=-1)
x_hat = (x - mean) / sqrt(var + eps)
output = gamma * x_hat + beta
```

### Parallel Residual Variant

Supports pre-norm with residual connection:
```
residual = x + residual_input
x_hat = layernorm(residual)
output = gamma * x_hat + beta
```

### Backward Kernel

Computes gradients for input, gamma, and beta:
```
dx_hat = dout * gamma
dvar = sum(dx_hat * (x - mean) * -0.5 * (var + eps)^(-3/2))
dmean = sum(dx_hat * -1/sqrt(var + eps)) + dvar * mean(-2 * (x - mean))
dx = dx_hat / sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N
dgamma = sum(dout * x_hat)
dbeta = sum(dout)
```

### API Entry Point

File: `csrc/layer_norm/ln_api.cpp`

Dispatches to the appropriate specialized kernel based on hidden size and data types.

---

## Fused Dense CUDA Kernels

### File: `csrc/fused_dense_lib/fused_dense_cuda.cu`

Implements fused linear (dense) layer operations combining GEMM with bias addition and activation.

### Operations

- **Fused Linear + Bias**: `output = input @ weight.T + bias`
- **Fused Linear + Bias + Gelu**: `output = gelu(input @ weight.T + bias)`
- **Backward**: Fused gradient computation for the above operations

### Benefits

- Eliminates the separate bias-add kernel launch
- Reduces memory traffic by fusing operations
- Better GPU utilization through kernel fusion

---

## Hopper SM80 Forward Kernels

### File: `hopper/flash_fwd_kernel_sm80.h`

The Hopper SM80 forward kernel is a reimplementation of the FA2 forward kernel using the Hopper codebase structure (CUTLASS library, CuTe tensor abstractions) but targeting SM80 tensor cores.

### Key Differences from FA2

- Uses CUTLASS 3.x abstractions throughout
- Cleaner separation of concerns with explicit pipeline management
- Support for additional features like packed GQA and paged KV via the Hopper infrastructure

---

## Hopper SM90 Forward Kernels

### File: `hopper/flash_fwd_kernel_sm90.h`

The SM90 forward kernel leverages Hopper-specific hardware features:

### TMA (Tensor Memory Accelerator)

- Uses `cp.async.bulk.tensor` for efficient bulk memory transfers
- Descriptor-based addressing eliminates pointer arithmetic in the kernel
- Hardware-managed prefetch and multi-stage pipeline

### Warpgroup MMA

- Uses `wgmma.mma_async` instructions for 64xNxK matrix multiply
- 4 warps (128 threads) cooperate as a warpgroup
- Asynchronous execution overlaps with memory operations

### Features

- **Paged KV Cache**: Full support via `PagedKVManager`
- **Pack GQA**: Packs multiple Q heads per KV head for efficiency
- **Softcap**: tanh-based attention score capping
- **FP8**: Support for FP8 quantized inputs with descaling
- **Different V dimension**: `headdim_v` can differ from `headdim`

---

## Hopper SM80 Backward Kernels

### File: `hopper/flash_bwd_kernel_sm80.h`

SM80 backward kernel using the Hopper code structure. Implements the same 5-GEMM backward algorithm as FA2 but with cleaner abstractions.

---

## Hopper SM90 Backward Kernels

### File: `hopper/flash_bwd_kernel_sm90.h`

SM90 backward kernel with TMA and warpgroup MMA.

### Warp Group Architecture

Each SM90 backward kernel has `num_wg + 1` warp groups:
- **WG0** (producer): TMA loads for Q, K, V, dO, LSE, dPsum
- **WG1** (producer): dQaccum store (TMA reduce-add to gmem)
- **WG2..WG(num_wg)** (MMA consumers): All GEMM computations

### Register-Source Optimization (mma_dkv_is_rs)

When conditions allow (`AtomLayoutMSdP == 1 && AtomLayoutNdKV == num_wg && SdP_swapAB && !dKV_swapAB`):
- P and dS matrices stay in registers
- Fed directly as A operand to dV and dK GEMMs
- Eliminates sP from shared memory
- Eliminates P register-to-shared store

---

## Split-KV Parallelism

### Heuristic Function

File: `hopper/heuristics.h`

```cpp
inline int num_splits_heuristic(
    int total_mblocks, int num_SMs, int num_n_blocks,
    int num_m_blocks, int size_one_kv_head,
    bool is_causal_or_local, int max_splits);
```

The heuristic:
1. If `total_mblocks >= 0.8 * num_SMs`, use 1 split (enough parallelism)
   - Exception: If KV doesn't fit in L2 (>50MB), split by L2 size
2. If `num_n_blocks <= 4`, use 1 split (too few blocks to split)
3. Otherwise, find the number of splits that maximizes SM occupancy, then pick the smallest `num_splits` achieving >= 85% of maximum efficiency

### PackGQA Heuristic

```cpp
inline bool should_pack_gqa(bool varlen_q, int seqlen_q,
                            int qhead_per_khead, int blockM);
```

Always packs for varlen. For fixed lengths, packs when the tiling efficiency improvement exceeds 10%.

---

## Paged KV Cache Manager

### File: `hopper/paged_kv.h`

### `PagedKVManager` Template

```cpp
template<int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads,
         typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {
```

Handles:
- Page table lookups: Given a block index, computes the physical page and offset
- Pointer computation: Distributes page pointer calculations across threads in a warp
- K/V loading with page boundaries: Handles non-contiguous memory access patterns
- Uses `cp.async` (not TMA) for paged KV since pages may be non-contiguous

### Page Table Structure

```
block_table[batch][max_num_pages_per_seq] -> physical_page_index
```

For block `n_block`:
```
page_index = n_block * kBlockN / page_block_size
page_offset = n_block * kBlockN - page_index * page_block_size
physical_address = page_table[batch][page_index] * k_batch_stride + page_offset * k_row_stride + head * k_head_stride
```

---

## Tile Size Heuristics

### File: `hopper/tile_size.h`

### SM90 Forward Tile Sizes

```cpp
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
    int headdim, int headdim_v, bool is_causal, bool is_local,
    int element_size=2, bool v_colmajor=false,
    bool paged_kv_non_TMA=false, bool softcap=false);
```

Returns `{kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}`.

| Head Dim | kBlockM | kBlockN | MmaPV_is_RS | Notes |
|----------|---------|---------|-------------|-------|
| <= 64 | 192 | 128-192 | true/false | Depends on causal/local |
| <= 96 | 192 | 128-144 | false | |
| <= 128 | 128 | 128-176 | true | Most common: 128x128 |
| <= 192 | 128 | 96-128 | true | Depends on headdim_v |
| > 192 | 128 | 64-80 | true | |

### SM8x Forward Tile Sizes

```cpp
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
    bool sm86_or_89, int headdim, int headdim_v, bool is_causal,
    bool is_local, int element_size=2, ...);
```

Returns `{kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}`.

---

## Utility Functions

### File: `csrc/flash_attn/src/utils.h`

Key utility functions used across all kernels:

#### `copy`

Generic copy function with boundary handling:
```cpp
template<bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN=true,
         bool Clear_OOB_K=true, ...>
void copy(TiledCopy tiled_copy, Tensor const &S, Tensor &D,
          Tensor const &identity_MN, Tensor &predicate_K,
          const int max_MN=0);
```

#### `gemm`

General matrix multiply using tiled MMA:
```cpp
template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0,
         typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, ...>
void gemm(Tensor0 &acc, Tensor1 const &tCrA, Tensor2 const &tCrB,
          Tensor3 const &tCsA, Tensor3 const &tCsB,
          TiledMma tiled_mma, ...);
```

#### `gemm_rs`

Register-source GEMM for P*V computation:
```cpp
template<typename Tensor0, typename Tensor1, typename Tensor2,
         typename Tensor3, typename TiledMma, ...>
void gemm_rs(Tensor0 &acc, Tensor1 const &tCrA, Tensor2 const &tCrB,
             Tensor3 const &tCsB, TiledMma tiled_mma, ...);
```

#### `convert_type`

Type conversion for tensor elements:
```cpp
template<typename To_type, typename Engine, typename Layout>
auto convert_type(Tensor<Engine, Layout> const &tensor);
```

#### Layout Conversion Functions

```cpp
convert_layout_acc_rowcol(Layout)    // (4, M, N) -> ((2, M/2), (2, N/2))
convert_layout_acc_Aregs(Layout)     // Reshape for register-source A operand
convert_layout_acc_dropout(Layout)   // Reshape for dropout application
```

---

## Generated Kernel Instantiations

### File: `csrc/flash_attn/src/generate_kernels.py`

Python script that generates the `.cu` files for all kernel instantiations.

### Forward Kernel Instantiations

For each combination of:
- **Head dimension**: 32, 64, 96, 128, 192, 256
- **Data type**: fp16, bf16
- **Causal**: true, false

Generated files: `flash_fwd_hdim{H}_{dtype}_{causal}_sm80.cu`

Each file instantiates the forward kernel with the appropriate `Flash_fwd_kernel_traits` and calls `run_flash_fwd` or `run_flash_splitkv_fwd`.

### Backward Kernel Instantiations

Same combinations, generating `flash_bwd_hdim{H}_{dtype}_{causal}_sm80.cu`.

### Split-KV Forward Instantiations

Additional files `flash_fwd_split_hdim{H}_{dtype}_{causal}_sm80.cu` for the SplitKV path.

### Total Kernel Count

With 6 head dims x 2 data types x 2 causal modes = 24 forward + 24 backward + 24 split = 72 instantiations, plus additional layer norm kernels.

---

## Static Switch Macros

### File: `csrc/flash_attn/src/static_switch.h`

Macros that convert runtime values to compile-time template parameters:

```cpp
// Boolean switch
BOOL_SWITCH(bool_val, CONST_NAME, lambda)

// Even K switch (special handling for Is_even_K)
EVENK_SWITCH(is_even_K, CONST_NAME, lambda)

// Dropout switch
DROPOUT_SWITCH(has_dropout, CONST_NAME, lambda)

// Local attention switch
LOCAL_SWITCH(is_local, CONST_NAME, lambda)

// ALiBi switch
ALIBI_SWITCH(has_alibi, CONST_NAME, lambda)

// Softcap switch
SOFTCAP_SWITCH(has_softcap, CONST_NAME, lambda)
```

These macros use C++ template lambdas to generate both true and false code paths, allowing the compiler to optimize each path independently while selecting at runtime.

### File: `hopper/static_switch.h`

Similar macros for the Hopper kernels, with additional switches for features like Split, Append_KV, and FP8.

---

## Flash API (C++ Host Side)

### File: `csrc/flash_attn/flash_api.cpp`

Host-side entry points that:
1. Parse tensor shapes and strides
2. Allocate workspace memory
3. Set up `Flash_fwd_params` or `Flash_bwd_params`
4. Dispatch to the appropriate head-dimension function
5. Launch CUDA kernels

### `hopper/flash_api.cpp`

Host-side API for the FA3 Hopper kernels, handling:
- Architecture detection and dispatch
- TMA descriptor setup
- Workspace allocation for SplitKV accumulation
- Paged KV table setup

### `hopper/flash_api_stable.cpp`

Stable API variant with additional error checking and backward compatibility.

---

## Preprocess and Postprocess Kernels

### File: `hopper/flash_bwd_preprocess_kernel.h`

Preprocesses backward input:
- Computes `dP_sum = rowsum(dO * O)` for each row
- Converts LSE to log2 space: `lse_log2 = lse * log2(e)`
- Zeros out `dQ_accum` buffer for accumulation

### File: `hopper/flash_bwd_postprocess_kernel.h`

Postprocesses backward output:
- Converts accumulated `dQ` from fp32 to the output dtype
- Handles variable-length sequence padding

### File: `hopper/flash_prepare_scheduler.cu`

Prepares the work scheduler for varlen and persistent kernels:
- Computes block ranges for each sequence in the batch
- Sets up scheduler metadata

---

## Combine Kernels

### File: `hopper/flash_fwd_combine_kernel.h`

### `flash_fwd_combine_launch_template.h`

Implements the combine step for SplitKV parallelism:
1. Each thread block handles a tile of the output
2. Loads per-split LSE and O_accum values
3. Computes log-sum-exp across splits
4. Scales and accumulates partial outputs
5. Writes final output to HBM

---

## Namespace Configuration

### File: `csrc/flash_attn/src/namespace_config.h`

Provides namespace aliasing to allow both FA2 and FA3 code to coexist:
- FA2 kernels use `FLASH_NAMESPACE` macro
- FA3 Hopper kernels use `flash` namespace

---

## Hardware Information

### File: `csrc/flash_attn/src/hardware_info.h`

Provides runtime hardware detection:
```cpp
int get_current_device();
std::pair<int, int> get_compute_capability(int device);
```

Used by launch templates to select optimal block sizes based on SM version (SM80, SM86, SM89, SM90).
