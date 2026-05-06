# FA4 Backward Kernels Reference Documentation

This document provides comprehensive reference documentation for all FlashAttention-4 backward
kernel implementations. The backward kernels compute gradients dQ, dK, dV given upstream gradients
dO from the attention output.

## Table of Contents

1. [Backward Algorithm Overview](#backward-algorithm-overview)
2. [FlashAttentionBackwardSm80](#flashattentionbackwardsm80)
3. [FlashAttentionBackwardSm90](#flashattentionbackwardsm90)
4. [FlashAttentionBackwardSm100](#flashattentionbackwardsm100)
5. [FlashAttentionBackwardSm120](#flashattentionbackwardsm120)
6. [FlashAttentionBackwardPreprocess](#flashattentionbackwardpreprocess)
7. [FlashAttentionBackwardPostprocess](#flashattentionbackwardpostprocess)
8. [Gradient Computation Details](#gradient-computation-details)
9. [Block Sparse Backward](#block-sparse-backward)
10. [GQA/MQA Backward](#gpamqa-backward)

---

## Backward Algorithm Overview

### Mathematical Formulation

Given attention output O = softmax(QK^T) V with upstream gradient dO, the backward pass computes:

```
P = softmax(QK^T)                           # Attention probabilities (from forward)
dP = dO @ V^T                               # Gradient w.r.t. P
D = rowsum(dO * O)                          # Diagonal correction term
dS = P * (dP - D)                           # Gradient w.r.t. S (pre-softmax scores)
dQ = dS @ K                                 # Gradient w.r.t. Q
dK = dS^T @ Q                               # Gradient w.r.t. K
dV = P^T @ dO                               # Gradient w.r.t. V
```

### Multi-Kernel Strategy

The backward pass is split into multiple kernels:

1. **Preprocess kernel:** Computes `D = rowsum(dO * O)` and optionally `D' = D - dLSE`
2. **Main backward kernel:** Computes dS, dK, dV, and accumulates dQ (atomically)
3. **Postprocess kernel:** Converts accumulated dQ from FP32 to the output dtype

### Tiled Backward Computation

The main backward kernel iterates over N blocks (KV dimension) rather than M blocks:

```
For each N block (n_block):
    Load K[n_block], V[n_block]
    Initialize acc_dK[n_block] = 0, acc_dV[n_block] = 0

    For each M block (m_block):
        Load Q[m_block], dO[m_block], LSE[m_block], dPsum[m_block]

        # Recompute attention scores
        S = Q[m_block] @ K[n_block]^T     # GEMM

        # Recompute probabilities from LSE
        P = exp(S * scale - LSE[m_block])  # Softmax recomputation

        # Compute dP
        dP = dO[m_block] @ V[n_block]^T   # GEMM

        # Compute dS = P * (dP - dPsum)
        dS = P * (dP - dPsum[m_block])     # Element-wise

        # Accumulate gradients
        dV += P^T @ dO[m_block]            # GEMM
        dK += dS^T @ Q[m_block]            # GEMM
        dQ[m_block] += dS @ K              # GEMM (atomic accumulation)

    Write dK[n_block], dV[n_block]
```

### Source Files

| File | Primary Class | Description |
|---|---|---|
| `flash_attn/cute/flash_bwd.py` | `FlashAttentionBackwardSm80` | Ampere backward |
| `flash_attn/cute/flash_bwd_sm90.py` | `FlashAttentionBackwardSm90` | Hopper backward |
| `flash_attn/cute/flash_bwd_sm100.py` | `FlashAttentionBackwardSm100` | Blackwell backward |
| `flash_attn/cute/flash_bwd_sm120.py` | `FlashAttentionBackwardSm120` | Blackwell GeForce backward |
| `flash_attn/cute/flash_bwd_preprocess.py` | `FlashAttentionBackwardPreprocess` | Preprocess kernel |
| `flash_attn/cute/flash_bwd_postprocess.py` | `FlashAttentionBackwardPostprocess` | Postprocess kernel |

---

## FlashAttentionBackwardSm80

**Location:** `flash_attn/cute/flash_bwd.py`

Ampere-class backward kernel using `mma.sync.aligned.m16n8k16` instructions and `cp.async`
for memory transfers.

### Constructor

```python
class FlashAttentionBackwardSm80:
    def __init__(
        self, dtype, head_dim, head_dim_v=None,
        qhead_per_kvhead=1, m_block_size=64, n_block_size=128,
        num_stages_Q=2, num_stages_dO=2, num_threads=256,
        pack_gqa=False, is_causal=False,
        SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
        AtomLayoutMSdP=1, AtomLayoutNdKV=8, AtomLayoutMdQ=1,
        V_in_regs=False, score_mod=None, score_mod_bwd=None,
    )
```

#### Parameters

- **dtype** (`Type[cutlass.Numeric]`): Data type. Must be Float16 or BFloat16.
- **head_dim** (`int`): Head dimension. Must be multiple of 8. Internally padded to multiple of 32.
- **head_dim_v** (`Optional[int]`): Value head dimension. Defaults to head_dim.
- **qhead_per_kvhead** (`int`): GQA ratio. Default 1.
- **m_block_size** (`int`): M dimension tile size. Default 64.
- **n_block_size** (`int`): N dimension tile size. Default 128.
- **num_stages_Q** (`int`): Pipeline stages for Q loading. Default 2.
- **num_stages_dO** (`int`): Pipeline stages for dO loading. Default 2.
- **num_threads** (`int`): Threads per block. Default 256.
- **pack_gqa** (`bool`): Whether to pack GQA Q heads. Default False.
- **is_causal** (`bool`): Causal masking flag. Default False.
- **SdP_swapAB** (`bool`): Swap A and B operands in S/dP MMA. Default False.
- **dKV_swapAB** (`bool`): Swap A and B operands in dK/dV MMA. Default False.
- **dQ_swapAB** (`bool`): Swap A and B operands in dQ MMA. Default False.
- **AtomLayoutMSdP** (`int`): Atom layout M dimension for S/dP MMA. Default 1.
- **AtomLayoutNdKV** (`int`): Atom layout N dimension for dK/dV MMA. Default 8.
- **AtomLayoutMdQ** (`int`): Atom layout M dimension for dQ MMA. Default 1.
- **V_in_regs** (`bool`): Keep V tile in registers. Default False.
- **score_mod** (`Optional[Constexpr]`): Score modification callable.
- **score_mod_bwd** (`Optional[Constexpr]`): Backward score modification callable.

### Shared Memory Layout

The backward kernel requires more shared memory than forward because it needs to store Q, K, V,
dO, LSE, dPsum, P, and dS simultaneously.

**Shared memory tensors:**
- `sQ`: Q tile with `num_stages_Q` pipeline stages
- `sK`: K tile (single stage, reused across M blocks)
- `sV`: V tile (single stage, shared with sQ if `V_in_regs`)
- `sdO`: dO tile with `num_stages_dO` pipeline stages
- `sP`/`sdS`: P and dS tiles (same layout, can share memory)
- `sLSE`: LSE values with pipeline stages
- `sdPsum`: dP sum values with pipeline stages

**Memory optimization:** When `V_in_regs=True`, V is loaded into registers and its shared memory
is shared with Q, reducing total SMEM usage.

### MMA Configuration

Three separate tiled MMA operations:

1. **tiled_mma_sdp:** For computing S = Q @ K^T and dP = dO @ V^T
   - Atom: `MmaF16BF16Op(dtype, Float32, (16, 8, 16))`
   - Layout: `(AtomLayoutMSdP, num_warps // AtomLayoutMSdP, 1)`

2. **tiled_mma_dkv:** For computing dV = P^T @ dO and dK = dS^T @ Q
   - Atom: `MmaF16BF16Op(dtype, Float32, (16, 8, 16))`
   - Layout: `(AtomLayoutNdKV, num_warps // AtomLayoutNdKV, 1)`

3. **tiled_mma_dq:** For computing dQ = dS @ K
   - Atom: `MmaF16BF16Op(dtype, Float32, (16, 8, 16))`
   - Layout: `(AtomLayoutMdQ, num_warps // AtomLayoutMdQ, 1)`

### `can_implement` (static)

```python
@staticmethod
def can_implement(
    dtype, head_dim, head_dim_v, m_block_size, n_block_size,
    num_stages_Q, num_stages_dO, num_threads, is_causal, V_in_regs=False,
) -> bool
```

**Validation:**
1. dtype must be Float16 or BFloat16
2. head_dim, head_dim_v must be multiples of 8
3. n_block_size must be multiple of 16
4. num_threads must be multiple of 32
5. Total SMEM must fit in 163 KB:
   ```
   smem = smem_Q(num_stages_Q) + smem_dO(num_stages_dO) + smem_K + smem_V
   ```

### `__call__`

```python
@cute.jit
def __call__(
    self, mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV,
    softmax_scale, mCuSeqlensQ=None, mCuSeqlensK=None,
    mSeqUsedQ=None, mSeqUsedK=None,
    window_size_left=None, window_size_right=None,
    mdQ_semaphore=None, mdK_semaphore=None, mdV_semaphore=None,
    aux_tensors=None, blocksparse_tensors=None,
    stream=None,
)
```

#### Parameters

- **mQ** (`cute.Tensor`): Query tensor, shape `(batch, seqlen_q, num_heads, head_dim)` or
  `(total_q, num_heads, head_dim)` for varlen.
- **mK** (`cute.Tensor`): Key tensor, shape `(batch, seqlen_k, num_kv_heads, head_dim)`.
- **mV** (`cute.Tensor`): Value tensor, shape `(batch, seqlen_k, num_kv_heads, head_dim_v)`.
- **mdO** (`cute.Tensor`): Gradient of output, same shape as O.
- **mLSE** (`cute.Tensor`): Log-sum-exp from forward pass, shape `(batch, num_heads, seqlen_q)`.
- **mdPsum** (`cute.Tensor`): Precomputed `rowsum(dO * O)`, shape `(batch, num_heads, seqlen_q)`.
- **mdQaccum** (`cute.Tensor`): Accumulator for dQ gradients (FP32), shape
  `(batch, num_heads, seqlen_q * head_dim)`.
- **mdK** (`cute.Tensor`): Output gradient for K.
- **mdV** (`cute.Tensor`): Output gradient for V.
- **softmax_scale** (`Float32`): Scale factor for attention scores.
- **mCuSeqlensQ** (`Optional[cute.Tensor]`): Cumulative sequence lengths for Q (varlen).
- **mCuSeqlensK** (`Optional[cute.Tensor]`): Cumulative sequence lengths for K (varlen).
- **mSeqUsedQ** (`Optional[cute.Tensor]`): Used sequence lengths for Q.
- **mSeqUsedK** (`Optional[cute.Tensor]`): Used sequence lengths for K.
- **window_size_left** (`Optional[Int32]`): Left window size for local attention.
- **window_size_right** (`Optional[Int32]`): Right window size for local attention.
- **mdQ_semaphore** (`Optional[cute.Tensor]`): Semaphore for deterministic dQ reduction.
- **mdK_semaphore** (`Optional[cute.Tensor]`): Semaphore for deterministic dK reduction.
- **mdV_semaphore** (`Optional[cute.Tensor]`): Semaphore for deterministic dV reduction.
- **aux_tensors** (`Optional[list]`): Auxiliary tensors for score/mask mods.
- **blocksparse_tensors** (`Optional[BlockSparseTensors]`): Block sparsity information.

### Kernel Flow

1. **Initialization:**
   - Create `SeqlenInfoQK` from tile scheduler
   - Compute m_block range: `m_block_min` to `m_block_max`
   - For causal: `m_block_min = max(0, (n_block * n_block_size + seqlen_q - seqlen_k) // m_block_size)`

2. **Prologue:**
   - Load V tile for the current n_block
   - Load K tile for the current n_block
   - Load first stages of Q, LSE, dO, dPsum

3. **Mainloop (iterating over M blocks):**
   - Call `compute_one_m_block()` for each m_block

4. **Epilogue:**
   - Scale dK by softmax_scale (if qhead_per_kvhead == 1)
   - Write dK and dV to global memory via `epilogue()`

### `compute_one_m_block`

```python
@cute.jit
def compute_one_m_block(
    self, m_block, smem_pipe_read_q, smem_pipe_read_do,
    smem_pipe_write_q, smem_pipe_write_do,
    mma_params, smem_copy_params, gmem_copy_params,
    load_Q_LSE, load_dO_dPsum, m_block_max,
    softmax_scale, softmax_scale_log2, mask_fn=None,
)
```

Processes one M block within the current N block.

**Algorithm:**
1. **Compute S (attention scores):**
   - GEMM: `S = Q[m_block] @ K^T` using `sm80_utils.gemm()`
   - Wait for Q pipeline synchronization

2. **Recompute P from S and LSE:**
   - Load LSE values from shared memory
   - Compute `P = exp2(S * scale_log2 - LSE)` for each element
   - Apply causal masking

3. **Compute dP:**
   - GEMM: `dP = dO[m_block] @ V^T`

4. **Compute dS:**
   - Load dPsum values
   - `dS = P * (dP - dPsum)` element-wise
   - Apply score_mod_bwd if provided

5. **Convert P and dS to dtype:**
   - Store P and dS in shared memory for subsequent GEMMs

6. **Compute dV:**
   - GEMM: `dV += P^T @ dO[m_block]`

7. **Compute dK:**
   - GEMM: `dK += dS^T @ Q[m_block]`

8. **Compute dQ:**
   - GEMM: `dQ_partial = dS @ K`
   - Atomic add to `dQaccum[m_block]` in global memory

9. **Prefetch next Q and dO tiles:**
   - Load next Q tile via pipeline
   - Load next dO tile via pipeline

### `epilogue`

```python
@cute.jit
def epilogue(
    self, acc_dK, acc_dV, mdK, mdV, sdK, sdV,
    gmem_tiled_copy_dK, gmem_tiled_copy_dV, tiled_mma,
    tidx, n_block, num_head, batch_size, seqlen, d_head, d_head_v,
)
```

Writes accumulated dK and dV to global memory.

**For standard MHA (qhead_per_kvhead == 1):**
1. Scale dK by softmax_scale
2. Convert dK, dV from FP32 to dtype
3. Store to shared memory via smem copy atom
4. Load from shared memory to registers for wider vectorization
5. Store to global memory with predicates for sequence length and head dimension OOB

**For GQA (qhead_per_kvhead > 1):**
1. Atomic add dK and dV to FP32 accumulation buffers in global memory
2. Actual scaling and type conversion deferred to postprocess

### `load_Q_LSE`

```python
@cute.jit
def load_Q_LSE(
    self, gmem_tiled_copy_Q, gmem_tiled_copy_LSE,
    tQgQ, tQsQ, tQcQ, t0QcQ, tQpQ,
    tLSEgLSE, tLSEsLSE, tLSEcLSE,
    block, smem_pipe_write_q, seqlen,
)
```

Loads Q tile and corresponding LSE values from global memory.

**Behavior:**
- For each row in the Q tile, checks sequence length bounds
- Applies head dimension predicates
- Loads LSE values padded to `m_block_size` elements (all initialized even for partial rows)
- Pipeline write index tracks the SMEM buffer

### `load_dO_dPsum`

```python
@cute.jit
def load_dO_dPsum(
    self, gmem_tiled_copy_dO, gmem_tiled_copy_dPsum,
    tdOgdO, tdOsdO, tdOcdO, t0dOcdO, tdOpdO,
    tdPsumgdPsum, tdPsumsdPsum, tdPsumcdPsum,
    block, smem_pipe_write_q, seqlen,
)
```

Loads dO tile and corresponding dPsum values. Similar structure to `load_Q_LSE`.

---

## FlashAttentionBackwardSm90

**Location:** `flash_attn/cute/flash_bwd_sm90.py`

Hopper-class backward kernel using WGMMA instructions and TMA for memory transfers.

### Constructor Additions

```python
class FlashAttentionBackwardSm90:
    def __init__(
        self, dtype, head_dim, head_dim_v=None,
        qhead_per_kvhead=1, is_causal=False, is_local=False,
        deterministic=False, tile_m=64, tile_n=128,
        Q_stage=2, dO_stage=2, PdS_stage=2,
        SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
        AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        num_threads=384, V_in_regs=False,
        score_mod=None, score_mod_bwd=None, mask_mod=None,
        has_aux_tensors=False, subtile_factor=1,
        dQ_single_wg=False,
    )
```

#### Additional Parameters

- **is_local** (`bool`): Local/sliding window attention. Default False.
- **deterministic** (`bool`): Enable deterministic gradient computation using semaphores.
  Default False.
- **Q_stage** (`int`): Pipeline stages for Q. Default 2.
- **dO_stage** (`int`): Pipeline stages for dO. Default 2.
- **PdS_stage** (`int`): Pipeline stages for P/dS shared memory. Default 2.
- **num_threads** (`int`): Threads per block. Default 384 (3 warp groups).
- **has_aux_tensors** (`Constexpr[bool]`): Whether aux tensors are provided.
- **subtile_factor** (`Constexpr[int]`): Subtiling factor for block sparse.
- **dQ_single_wg** (`bool`): Only WG0 computes dQ GEMM, WG1 skips. Default False.

### Warp Group Architecture

| Warp Group | Role | Threads |
|---|---|---|
| Warp 0 (producer) | TMA loads | 32 |
| Warp Groups 1-N (consumer) | MMA compute | 128 per WG |

**Thread count options:**
- 2 MMA WGs: 288 threads
- 3 MMA WGs: 416 threads

### TMA Configuration

All tensor loads use TMA descriptors:

```python
tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),
    mQ, sQ_layout, (tile_m, tile_hdim),
)
```

**TMA atoms created:**
- `tma_atom_Q`: Q tile loading
- `tma_atom_K`: K tile loading
- `tma_atom_V`: V tile loading
- `tma_atom_dO`: dO tile loading
- `tma_atom_dK`: dK tile storing
- `tma_atom_dV`: dV tile storing
- `tma_atom_dQ`: dQ tile storing

### Shared Memory Layout

Uses Hopper-optimized layouts with swizzling:
- `sQ_layout`: Q storage with `Q_stage` pipeline stages
- `sdO_layout`: dO storage with `dO_stage` pipeline stages
- `sK_layout`: K storage (single stage)
- `sV_layout`: V storage (single stage)
- `sPdS_layout`: P/dS shared memory with `PdS_stage` stages

### Register Allocation

```python
# 2 warp groups
num_mma_regs_wg0 = 240  # or 256 if dQ_single_wg
num_mma_regs_wg1 = 240  # or 224 if dQ_single_wg
num_producer_regs = 24

# 3 warp groups
num_mma_regs = 160  # per WG
num_producer_regs = 32
```

### Shuffle Optimization

When `SdP_swapAB=True` and `head_dim <= 64`:
- `shuffle_LSE = True`: LSE values are split across 8 threads and shuffled when needed
- `shuffle_dPsum = True`: Same optimization for dPsum values
- Reduces register pressure at the cost of warp-level shuffles

### Deterministic Mode

When `deterministic=True`:
- Uses semaphores for dQ, dK, dV atomic additions
- Ensures bit-identical results across runs
- Requires `mdQ_semaphore`, `mdK_semaphore`, `mdV_semaphore` tensors

### Block Sparsity Support

```python
from flash_attn.cute.block_sparse_utils import (
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
)
```

Block sparsity in backward:
- Only loads Q/dO tiles that have corresponding non-zero KV blocks
- `produce_block_sparse_q_loads_bwd_sm90`: Determines which Q blocks to load
- `consume_block_sparse_mma_bwd_sm90`: Processes only non-zero blocks

---

## FlashAttentionBackwardSm100

**Location:** `flash_attn/cute/flash_bwd_sm100.py`

Blackwell-class backward kernel using UMMA instructions, TMEM, and 2CTA support.

### Constructor

```python
class FlashAttentionBackwardSm100:
    def __init__(
        self, head_dim, head_dim_v=None,
        is_causal=False, is_local=False,
        qhead_per_kvhead=1, tile_m=128, tile_n=128,
        is_persistent=False, deterministic=False,
        spt=None, cluster_size=1, use_2cta_instrs=False,
        score_mod=None, score_mod_bwd=None, mask_mod=None,
        has_aux_tensors=False, subtile_factor=1,
    )
```

#### Parameters

- **spt** (`Optional[bool]`): Override for software preemption timeout.
- **cluster_size** (`int`): Number of CTAs per cluster (1 or 2). Default 1.
- **use_2cta_instrs** (`bool`): Enable 2CTA MMA instructions. Default False.

### Warp Architecture (16 warps = 512 threads)

| Warp IDs | Role |
|---|---|
| 0-3 | Reduce (dQaccum reduction, dK/dV epilogue) |
| 4-11 | Compute (S/dP MMA, dK/dV/dQ MMA) |
| 12 | MMA (UMMA dispatch) |
| 13 | Load (TMA Q/K/V/dO loading) |
| 14 | Relay (cluster communication) |
| 15 | Empty (register deallocation) |

### TMEM Layout

```
tmem_S_offset:     0
tmem_P_offset:     0           # overlaps with S
tmem_dV_offset:    n_block_size
tmem_dP_offset:    n_block_size + head_dim_v
tmem_dQ_offset:    varies (may overlap with dP)
tmem_dK_offset:    dP_offset + tile_m
tmem_dS_offset:    dP_offset   # overlaps with dP
```

### MMA Configuration

Five separate tiled MMA operations:

1. **tiled_mma_S:** `S.T = K @ Q.T` (recompute attention scores)
2. **tiled_mma_dP:** `dP.T = V @ dO.T` (compute dP)
3. **tiled_mma_dV:** `dV += P.T @ dO` (accumulate dV)
4. **tiled_mma_dK:** `dK += dS.T @ Q` (accumulate dK)
5. **tiled_mma_dQ:** `dQ = dS @ K` (compute dQ)

### 2CTA Support

When `use_2cta_instrs=True`:
- Cluster shape: `(2, 1)` or `(1, 2)`
- MMA tiler M dimension doubled: `cta_group_size * tile_n`
- TMEM offsets adjusted for 2CTA layout
- Special dQ reduction with `dQ_reduce_ncol` subtiling

### dQ Accumulation Strategy

dQ is accumulated via TMA atomic adds:
1. Compute dQ_partial via UMMA
2. Convert from TMEM to shared memory
3. TMA bulk add to global dQaccum buffer
4. Postprocess kernel converts accumulated FP32 to final dtype

### Register Budget

```python
num_regs_reduce = 152    # Reduce warps
num_regs_compute = 136   # Compute warps (x2 groups)
num_regs_load = 96       # Load warp
num_regs_mma = 96        # MMA warp
num_regs_empty = 24      # Empty warps

# Total budget: 512 registers per thread
assert (reduce + compute * 2 + max(load, mma)) <= 512
```

---

## FlashAttentionBackwardSm120

**Location:** `flash_attn/cute/flash_bwd_sm120.py`
**Inheritance:** `FlashAttentionBackwardSm80`

SM120 variant with reduced shared memory (99 KB).

### Key Differences

- SMEM capacity: 99 KB vs 163 KB on SM80
- Same MMA instructions as SM80
- Overrides `can_implement()` to use SM120 SMEM limit

---

## FlashAttentionBackwardPreprocess

**Location:** `flash_attn/cute/flash_bwd_preprocess.py`

Computes `D_i = rowsum(dO_i * O_i)`, the diagonal correction term needed by the main backward kernel.

### Mathematical Background

In the backward pass:
```
dS_ij = P_ij * (dP_ij - D_i)
```

When LSE is differentiable, an extra term is added:
```
dS_ij = P_ij * (dP_ij - D_i) + dLSE_i * P_ij
      = P_ij * (dP_ij - (D_i - dLSE_i))
```

So the preprocess kernel computes `D' = D - dLSE` when `dLSE` is provided.

### Constructor

```python
class FlashAttentionBackwardPreprocess:
    def __init__(
        self, dtype, head_dim, head_dim_v,
        tile_m=128, num_threads=256, use_padded_offsets=True,
    )
```

#### Parameters

- **dtype** (`Type[cutlass.Numeric]`): Data type for O and dO.
- **head_dim** (`int`): Q/K head dimension.
- **head_dim_v** (`int`): V/O head dimension.
- **tile_m** (`int`): M block size. Default 128.
- **num_threads** (`int`): Threads per block. Default 256.
- **use_padded_offsets** (`bool`): Use padded offsets for stats buffers. Default True.

### `__call__`

```python
@cute.jit
def __call__(
    self, mO, mdO, mPdPsum, mLSE=None, mLSElog2=None,
    mdQaccum=None, mCuSeqlensQ=None, mSeqUsedQ=None,
    mdLSE=None, stream=None,
)
```

#### Parameters

- **mO** (`cute.Tensor`): Forward pass output.
- **mdO** (`cute.Tensor`): Upstream gradient.
- **mPdPsum** (`cute.Tensor`): Output buffer for `D = rowsum(dO * O)`.
- **mLSE** (`Optional[cute.Tensor]`): Log-sum-exp from forward.
- **mLSElog2** (`Optional[cute.Tensor]`): Output buffer for `LSE * log2(e)`.
- **mdQaccum** (`Optional[cute.Tensor]`): dQ accumulator to clear (set to zero).
- **mCuSeqlensQ** (`Optional[cute.Tensor]`): Variable-length Q offsets.
- **mSeqUsedQ** (`Optional[cute.Tensor]`): Used sequence lengths.
- **mdLSE** (`Optional[cute.Tensor]`): Gradient of LSE (for LSE differentiation).

### Kernel Algorithm

1. **Load O and dO tiles** from global memory to shared memory
2. **Compute element-wise product:** `O * dO`
3. **Reduce along head dimension:** `pdpsum = sum(O * dO, dim=-1)`
4. **Apply dLSE correction:** `PdPsum_val -= dLSE[row]` if dLSE is provided
5. **Store PdPsum** to global memory
6. **Compute LSElog2:** `LSElog2 = LSE * log2(e)` (or 0 if LSE is -inf)
7. **Clear dQaccum:** Zero out the dQ accumulation buffer for the current tile

### PDL (Programmatic Dependent Launch)

The preprocess kernel uses PDL to overlap with the previous kernel:
```python
# Wait for upstream kernel to finish writing O and dO
cute.arch.griddepcontrol_wait()
# ... compute D ...
# Signal that dependent kernels (main backward) can start
cute.arch.griddepcontrol_launch_dependents()
```

---

## FlashAttentionBackwardPostprocess

**Location:** `flash_attn/cute/flash_bwd_postprocess.py`

Converts accumulated dQ from FP32 to the output dtype, applying the softmax scale.

### Constructor

```python
class FlashAttentionBackwardPostprocess:
    def __init__(
        self, dtype, head_dim, arch,
        tile_m=128, num_threads=256,
        AtomLayoutMdQ=1, dQ_swapAB=False,
        use_2cta_instrs=False, cluster_size=1,
    )
```

#### Parameters

- **dtype** (`Type[cutlass.Numeric]`): Output dtype (Float16 or BFloat16).
- **head_dim** (`int`): Head dimension.
- **arch** (`int`): GPU architecture (80, 90, 100, etc.).
- **tile_m** (`int`): M block size. Default 128.
- **num_threads** (`int`): Threads per block. Default 256.
- **AtomLayoutMdQ** (`int`): Atom layout for dQ MMA. Default 1.
- **dQ_swapAB** (`bool`): Whether dQ MMA swaps AB. Default False.
- **use_2cta_instrs** (`bool`): Use 2CTA instructions. Default False.
- **cluster_size** (`int`): Cluster size for varlen. Default 1.

### `__call__`

```python
@cute.jit
def __call__(
    self, mdQaccum, mdQ, scale,
    mCuSeqlensQ=None, mSeqUsedQ=None, stream=None,
)
```

#### Parameters

- **mdQaccum** (`cute.Tensor`): FP32 accumulated dQ.
- **mdQ** (`cute.Tensor`): Output dQ tensor.
- **scale** (`Float32`): Softmax scale factor.
- **mCuSeqlensQ** (`Optional[cute.Tensor]`): Variable-length offsets.
- **mSeqUsedQ** (`Optional[cute.Tensor]`): Used sequence lengths.

### Kernel Algorithm

**Architecture-specific paths:**

**SM80/SM120:**
1. Load dQaccum from global to shared memory (async copy)
2. Load from shared memory to registers
3. Convert: `dQ = (dQaccum * scale).to(dtype)`
4. Store to shared memory via smem copy atom
5. Load from shared memory to registers for coalesced write
6. Store to global memory with predicates

**SM90:**
1. Load dQaccum from global to shared memory
2. Use warp group layout for smem-to-register transfer
3. Convert and write back to smem
4. TMA or coalesced store to global memory

**SM100 (1-CTA):**
1. Load dQaccum from global to shared memory
2. Use TMEM-compatible layout for register loading
3. Convert via TMEM load/store path
4. TMA store to global memory

**SM100 (2-CTA):**
1. Load dQaccum in subtile chunks (`dQ_reduce_ncol` columns)
2. Remap dQaccum layout into TMEM view
3. Convert in stages with barrier synchronization
4. Store to global memory

---

## Gradient Computation Details

### dQ Computation

dQ requires atomic accumulation because multiple N blocks contribute to the same dQ tile:
```
dQ[m_block] += dS[m_block, n_block_0] @ K[n_block_0]
dQ[m_block] += dS[m_block, n_block_1] @ K[n_block_1]
...
```

This is implemented via `atomic_add_fp32` to the `dQaccum` buffer:
```python
for i in cutlass.range(cute.size(acc_dQ_atomic), unroll_full=True):
    utils.atomic_add_fp32(acc_dQ_atomic[i],
                          utils.elem_pointer(tdQgdQaccum_atomic, i))
```

### dK and dV Computation

dK and dV are accumulated across M blocks within a single N block's kernel launch:
```
dK[n_block] = sum_m(dS[m, n_block]^T @ Q[m])
dV[n_block] = sum_m(P[m, n_block]^T @ dO[m])
```

For MHA, the result is written directly. For GQA, atomic adds to FP32 buffers.

### Softmax Recomputation

P is recomputed from S and LSE rather than stored from the forward pass:
```python
# In compute_one_m_block:
acc_S_mn[r, None].store(
    cute.math.exp2(acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r],
                   fastmath=True)
)
```

This saves memory (no need to store P) at the cost of recomputation.

### Score Modification Backward

When `score_mod` is used in the forward pass, the backward requires a separate `score_mod_bwd`:

```python
# Forward: S = score_mod(S_raw * scale, ...)
# Backward: dS_raw = score_mod_bwd(dS, S_raw * scale, ...)
```

The `score_mod_bwd` callable receives the gradient and the original score value.

---

## Block Sparse Backward

### Block Sparse Iteration

The backward kernel supports block sparsity where only certain (Q_block, KV_block) pairs
are computed:

```python
# Get total Q blocks to process for this N block
total_q_blocks = get_total_q_block_count_bwd(
    blocksparse_tensors, batch_idx, head_idx, n_block, ...)

# Produce sparse Q loads
produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors, batch_idx, head_idx, n_block,
    q_producer_state, tma_load_Q_fn, pipeline_q, ...)

# Consume sparse MMA blocks
consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors, ..., mma_one_m_block, ...)
```

### Sparse dQ Storage

```python
dQaccum_store_block_sparse_bwd_sm90(
    blocksparse_tensors, batch_idx, head_idx, n_block,
    acc_dQ, dQaccum, ...)
```

Only writes dQ accumulation for Q blocks that have non-zero attention weights.

---

## GQA/MQA Backward

### Multi-Query Attention (MQA)

When `qhead_per_kvhead > 1`, multiple Q heads share the same K and V:
- dK and dV receive contributions from all associated Q heads
- The main backward kernel accumulates dK/dV in FP32 buffers using atomic adds
- A separate postprocess step handles the final type conversion

### Accumulation Strategy

**For dK/dV (GQA):**
```python
# Atomic add to FP32 accumulation buffer
for i in cutlass.range(cute.size(acc_dV_atomic), unroll_full=True):
    utils.atomic_add_fp32(acc_dV_atomic[i],
                          utils.elem_pointer(tdVgdVaccum, i))
```

**For dQ (always):**
```python
# dQ always uses atomic accumulation (even for MHA)
# because multiple N blocks contribute to the same Q tile
for i in cutlass.range(cute.size(acc_dQ_atomic), unroll_full=True):
    utils.atomic_add_fp32(acc_dQ_atomic[i],
                          utils.elem_pointer(tdQgdQaccum_atomic, i))
```

### Variable Length with GQA

When using varlen with GQA:
- Q, dO, LSE, dPsum use `cu_seqlens_q` for offsets
- K, V use `cu_seqlens_k` for offsets (different sequence lengths)
- dKaccum, dVaccum use padded offsets for alignment
