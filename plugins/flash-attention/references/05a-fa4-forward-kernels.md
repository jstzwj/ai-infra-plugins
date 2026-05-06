# FA4 Forward Kernels Reference Documentation

This document provides comprehensive reference documentation for all FlashAttention-4 forward kernel
implementations. The forward kernels compute the attention output `O = softmax(Q @ K^T) @ V` using
tiled, memory-efficient algorithms that scale linearly in sequence length.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [FlashAttentionForwardBase](#flashattentionforwardbase)
3. [FlashAttentionForwardSm80](#flashattentionforwardsm80)
4. [FlashAttentionForwardSm90](#flashattentionforwardsm90)
5. [FlashAttentionForwardSm100](#flashattentionforwardsm100)
6. [FlashAttentionForwardSm120](#flashattentionforwardsm120)
7. [FlashAttentionForwardCombine](#flashattentionforwardcombine)
8. [FlashAttentionForwardMLASm100](#flashattentionforwardmlasm100)
9. [BlackwellFusedMultiHeadAttentionForward (hd256 2CTA)](#blackwellfusedmultiheadattentionforward)
10. [Kernel Pipeline and Execution Flow](#kernel-pipeline-and-execution-flow)
11. [TMA Usage Patterns](#tma-usage-patterns)
12. [SplitKV Mechanism](#splitkv-mechanism)
13. [Paged KV Cache](#paged-kv-cache)
14. [Persistent Kernels](#persistent-kernels)
15. [2CTA Instructions](#2cta-instructions)

---

## Architecture Overview

FA4 provides forward kernel implementations targeting three GPU architectures:

| Class | Architecture | GPU | MMA Instructions | Key Features |
|---|---|---|---|---|
| `FlashAttentionForwardSm80` | SM80 (Ampere) | A100 | mma.sync (m16n8k16) | cp.async, LdMatrix |
| `FlashAttentionForwardSm90` | SM90 (Hopper) | H100 | WGMMA (64xN) | TMA, warpgroup |
| `FlashAttentionForwardSm100` | SM100 (Blackwell) | B100/B200 | UMMA (128xN) | TMA, TMEM, 2CTA |
| `FlashAttentionForwardSm120` | SM120 (Blackwell) | DGX Spark | mma.sync (m16n8k16) | cp.async, 99KB smem |

All forward kernels implement an online softmax algorithm that processes the KV sequence in tiles,
maintaining running maximum and sum statistics to compute attention without materializing the full
S = Q @ K^T matrix.

### Source Files

| File | Primary Class | Description |
|---|---|---|
| `flash_attn/cute/flash_fwd.py` | `FlashAttentionForwardSm80`, `FlashAttentionForwardBase` | Ampere forward with cp.async pipelines |
| `flash_attn/cute/flash_fwd_sm90.py` | `FlashAttentionForwardSm90` | Hopper forward with TMA and WGMMA |
| `flash_attn/cute/flash_fwd_sm100.py` | `FlashAttentionForwardSm100` | Blackwell forward with UMMA and TMEM |
| `flash_attn/cute/flash_fwd_sm120.py` | `FlashAttentionForwardSm120` | Blackwell GeForce (SM120) forward |
| `flash_attn/cute/flash_fwd_combine.py` | `FlashAttentionForwardCombine` | SplitKV partial result combiner |
| `flash_attn/cute/flash_fwd_mla_sm100.py` | `FlashAttentionForwardMLASm100` | MLA-specific Blackwell forward |
| `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py` | `BlackwellFusedMultiHeadAttentionForward` | Dedicated hd256 2CTA Blackwell kernel |

---

## FlashAttentionForwardBase

**Location:** `flash_attn/cute/flash_fwd.py`
**Inheritance:** None (base class)

Base class providing common configuration, type checking, and shared memory layout management
for all forward kernel implementations.

### Constructor

```python
class FlashAttentionForwardBase:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        q_subtile_factor: int | None = None,
    )
```

#### Parameters

- **dtype** (`Type[cutlass.Numeric]`): Data type for Q, K, V, O tensors. Must be `cutlass.Float16` or
  `cutlass.BFloat16`.
- **head_dim** (`int`): Head dimension (d). Must be a multiple of 8 for 16-byte alignment. Internally
  padded to a multiple of 16 as `tile_hdim`.
- **head_dim_v** (`Optional[int]`): Output head dimension (d_v). Defaults to `head_dim` if not specified.
  Allows different dimensions for the value projection (e.g., for multi-latent attention).
- **qhead_per_kvhead** (`int`): Number of query heads per key-value head. Used for GQA/MQA. Default is 1
  (standard MHA).
- **is_causal** (`bool`): Whether to apply causal masking. When True, positions can only attend to
  previous positions.
- **is_local** (`bool`): Whether to apply local/sliding window attention. Requires `window_size_left`
  and/or `window_size_right` at runtime.
- **pack_gqa** (`bool`): Whether to pack multiple Q heads into the same tile for GQA efficiency.
  Default True.
- **tile_m** (`int`): Tile size along the M (query sequence) dimension. Default 128.
- **tile_n** (`int`): Tile size along the N (key-value sequence) dimension. Default 128.
- **num_stages** (`int`): Number of pipeline stages for async memory copies. Default 1.
- **num_threads** (`int`): Number of threads per CUDA block. Default 128.
- **Q_in_regs** (`bool`): Whether to keep the Q tile in registers rather than shared memory. Saves
  shared memory at the cost of register pressure.
- **score_mod** (`Optional[cutlass.Constexpr]`): A compile-time callable that modifies attention scores.
  Signature: `score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, aux_tensors) -> Any`.
- **mask_mod** (`Optional[cutlass.Constexpr]`): A compile-time callable that returns boolean mask.
  Signature: `mask_mod(batch_idx, head_idx, q_idx, kv_idx, aux_tensors) -> Boolean`.
- **has_aux_tensors** (`bool`): Whether the kernel receives auxiliary tensors for score/mask mods.
- **q_subtile_factor** (`Optional[int]`): Subtiling factor for block-sparse attention Q tiles.

### Instance Attributes

| Attribute | Type | Description |
|---|---|---|
| `dtype` | `Type[cutlass.Numeric]` | Input/output data type |
| `tile_hdim` | `int` | Head dimension padded to multiple of 16 |
| `tile_hdimv` | `int` | Value head dimension padded to multiple of 16 |
| `same_hdim_kv` | `bool` | Whether head_dim == head_dim_v |
| `check_hdim_oob` | `bool` | Whether head_dim OOB checks needed |
| `check_hdim_v_oob` | `bool` | Whether head_dim_v OOB checks needed |
| `qhead_per_kvhead` | `int` | GQA ratio |
| `is_causal` | `bool` | Causal masking flag |
| `is_local` | `bool` | Local attention flag |
| `pack_gqa` | `bool` | GQA packing flag |
| `tile_m` | `int` | M dimension tile size |
| `tile_n` | `int` | N dimension tile size |
| `num_threads` | `int` | Threads per block |
| `num_stages` | `int` | Pipeline stages |
| `Q_in_regs` | `bool` | Q in registers flag |
| `score_mod` | `Optional[Constexpr]` | Score modification callable |
| `mask_mod` | `Optional[Constexpr]` | Mask modification callable |
| `qk_acc_dtype` | `Type` | Accumulator data type (always Float32) |
| `vec_size` | `Constexpr[int]` | Vector size for score mod processing |
| `arch` | `Arch` | Target GPU architecture |

### Methods

#### `can_implement` (static)

```python
@staticmethod
def can_implement(
    dtype, head_dim, head_dim_v, tile_m, tile_n,
    num_stages, num_threads, is_causal, Q_in_regs=False,
) -> bool
```

Checks whether the kernel can be implemented with the given configuration parameters.

**Validation rules:**
1. `dtype` must be `Float16` or `BFloat16`
2. `head_dim` and `head_dim_v` must be multiples of 8
3. `tile_n` must be a multiple of 16
4. `num_threads` must be a multiple of 32
5. Total shared memory usage must fit within the SM80 capacity (163 KB)
6. `tile_m * 2` must be divisible by `num_threads`

**Shared memory computation:**
```
smem_Q = tile_m * head_dim * 2  (bytes)
smem_K = tile_n * head_dim * num_stages * 2
smem_V = tile_n * head_dim_v * num_stages * 2
smem_QV = smem_Q + smem_V  (or max if Q_in_regs)
total = smem_QV + smem_K
```

#### `_check_type`

```python
def _check_type(
    self, mQ_type, mK_type, mV_type, mO_type, mLSE_type,
    mCuSeqlensQ_type, mCuSeqlensK_type, mSeqUsedQ_type, mSeqUsedK_type,
)
```

Validates tensor data types at kernel compile time. Enforces:
- All Q, K, V, O tensors must have the same type (Float16 or BFloat16)
- LSE must be Float32
- cu_seqlens and seqused tensors must be Int32

#### `_setup_attributes`

Sets up shared memory layouts and GMEM tiled copy configurations. Creates:
- `sQ_layout`, `sK_layout`, `sV_layout`, `sO_layout`: Shared memory layouts for Q, K, V, O
- `gmem_tiled_copy_Q`, `gmem_tiled_copy_K`, `gmem_tiled_copy_V`: Async GMEM-to-SMEM copy tilings
- `gmem_tiled_copy_O`: Universal SMEM-to-GMEM copy tiling for O store

**Memory layout strategy:**
- Uses `cpasync.CopyG2SOp` with `LoadCacheMode.GLOBAL` for async Q, K, V loads
- Uses `CopyUniversalOp` for O stores
- 128-bit per copy operation for maximum memory bandwidth utilization

#### `__call__`

```python
@cute.jit
def __call__(
    self, mQ, mK, mV, mO, mLSE, softmax_scale, stream=None,
)
```

Abstract entry point for kernel launch. Must be overridden by subclasses.

#### `epilogue`

```python
@cute.jit
def epilogue(
    self, acc_O, lse, mO, mLSE, sO, seqlen,
    gmem_tiled_copy_O, tma_atom_O, tiled_mma,
    tidx, m_block, head_idx, batch_idx,
)
```

Handles the final output writing phase after the mainloop completes.

**Algorithm:**
1. Convert accumulator `acc_O` from Float32 to `dtype`
2. Synchronize all threads via `NamedBarrierFwd.Epilogue`
3. Copy from registers to shared memory via smem copy atom
4. If `mLSE` is provided:
   - Write LSE values from registers to global memory
   - Handle pack_gqa LSE storage differently from standard
5. Write O from shared memory to global memory:
   - TMA path: Fence smem, barrier, TMA bulk store
   - cp.async path: smem-to-rmem load, then rmem-to-gmem store with predicates

**Parameters:**
- `acc_O`: Accumulator tensor containing unnormalized attention output
- `lse`: Log-sum-exp values for each row
- `mO`: Global memory O tensor
- `mLSE`: Optional global memory LSE tensor
- `sO`: Shared memory buffer for O (reuses sQ data pointer)
- `seqlen`: `SeqlenInfoQK` with sequence length information
- `gmem_tiled_copy_O`: Tiled copy for O store
- `tma_atom_O`: TMA copy atom (None for SM80)
- `tiled_mma`: MMA tiling configuration
- `tidx`: Thread index
- `m_block`: Current M block index
- `head_idx`: Attention head index
- `batch_idx`: Batch index

#### `load_Q`

```python
@cute.jit
def load_Q(self, gmem_thr_copy, gQ, sQ, block, seqlen, headdim)
```

Loads a Q tile from global memory to shared memory.

**Behavior:**
- Partitions global and shared memory tiles according to the thread copy layout
- Creates an identity tensor for coordinate tracking
- Generates predicates for head dimension OOB
- For each row in the tile, checks if the row index is within sequence length before copying
- Uses the `t0QcQ` (thread-0 coordinates, compile-time known) optimization to avoid runtime checks

#### `load_K`

```python
@cute.jit
def load_K(
    self, gmem_tiled_copy, tKgK, tKsK, tKcK, t0KcK, tKpK,
    block, smem_pipe_write, seqlen, need_predicates,
)
```

Loads a K tile from global memory to shared memory with pipeline support.

**Behavior:**
- If `need_predicates` or the tile doesn't evenly divide the copy, applies per-row predicates
- Computes sequence length limit for each row
- Commits to async copy group for pipeline synchronization

#### `load_V`

```python
@cute.jit
def load_V(
    self, gmem_tiled_copy, tVgV, tVsV, tVcV, t0VcV, tVpV,
    block, smem_pipe_write, seqlen, need_predicates,
)
```

Loads a V tile from global memory to shared memory with pipeline support.

**Behavior:**
- Similar to `load_K` but handles the `head_dim_v` dimension separately
- Combines head dimension and sequence length predicates
- Supports multi-stage pipelines via `smem_pipe_write` index

---

## FlashAttentionForwardSm80

**Location:** `flash_attn/cute/flash_fwd.py`
**Inheritance:** `FlashAttentionForwardBase`

Ampere-class forward kernel using `mma.sync.aligned.m16n8k16` MMA instructions and `cp.async`
for memory transfers.

### Shared Memory Layout

Uses Ampere-style shared memory layout atoms from `ampere_helpers.get_smem_layout_atom()`:
- `sQ_layout_atom`: Row-major layout optimized for LdMatrix loads
- `sK_layout_atom`: Same as Q layout atom
- `sV_layout_atom`: Separate layout if head_dim_v differs, or same as Q
- `sO_layout_atom`: Same as V layout atom

### MMA Configuration

```python
tiled_mma_qk = cute.make_tiled_mma(
    warp.MmaF16BF16Op(dtype, Float32, (16, 8, 16)),
    (num_threads // 32, 1, 1),
    permutation_mnk=(num_threads // 32 * 16, 16, 16),
)
```

- Uses `MmaF16BF16Op` with MMA shape (16, 8, 16)
- Replicates MMA atoms across warps: `num_threads // 32` atoms in the M dimension
- Separate tiled MMA for QK and PV operations

### Kernel Flow

The SM80 kernel (`FlashAttentionForwardSm80.kernel`) follows this flow:

1. **Initialization:**
   - Create `BlockInfo` and `SeqlenInfoQK` from tile scheduler parameters
   - Compute `n_block_min` and `n_block_max` from block info
   - Partition global and shared memory tensors

2. **Prologue:**
   - Load Q tile via `load_Q()` (async copy)
   - If `Q_in_regs`: load one stage of K, preprocess Q (smem-to-reg), barrier, then load V
   - If not `Q_in_regs`: load all stages of K and V, then preprocess Q
   - Commit async copy groups

3. **Mainloop (iterating over N blocks):**
   - First iteration with sequence length masking
   - Causal/local masking iterations (if applicable)
   - Remaining iterations without masking
   - Each iteration calls `compute_one_n_block()`

4. **Finalization:**
   - Call `softmax.finalize()` to get row_scale
   - Rescale `acc_O` by row_scale
   - Call `epilogue()` to write O and LSE to global memory

### `compute_one_n_block`

```python
@cute.jit
def compute_one_n_block(
    self, n_block, smem_pipe_read, smem_pipe_write,
    mma_params, smem_copy_params, softmax,
    load_K, load_V, score_mod, batch_idx, head_idx,
    m_block, seqlen, aux_tensors, fastdiv_mods,
    mask_fn=None, is_first_n_block=False, check_inf=True,
)
```

Processes a single N block (KV tile) in the mainloop.

**Algorithm:**
1. **Synchronize:** Wait for QK smem tiles (`cp_async_wait_group(num_stages * 2 - 2)`)
2. **Load V for next iteration:** Prefetch V tile for pipeline overlap
3. **GEMM QK:** Compute S = Q @ K^T using `sm80_utils.gemm()`
4. **Score modification:** Apply optional `score_mod` to attention scores
5. **Advance pipeline:** Increment smem write pointer
6. **Load K for next+1 iteration:** Prefetch K tile
7. **Masking:** Apply attention mask (sequence length, causal, local)
8. **Online softmax:** `softmax.online_softmax(acc_S)` returns row_scale
9. **Rescale O:** `softmax.rescale_O(acc_O, row_scale)` rescales running attention output
10. **Convert P:** Cast attention probabilities from Float32 to dtype
11. **GEMM PV:** Compute O += P @ V using `sm80_utils.gemm_rs()`

---

## FlashAttentionForwardSm90

**Location:** `flash_attn/cute/flash_fwd_sm90.py`
**Inheritance:** `FlashAttentionForwardBase`

Hopper-class forward kernel using WGMMA (Warp Group Matrix Multiply Accumulate) instructions
and TMA (Tensor Memory Access) for high-bandwidth memory transfers.

### Constructor Additions

```python
class FlashAttentionForwardSm90(FlashAttentionForwardBase):
    def __init__(
        self, *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        **kwargs,
    )
```

- **intra_wg_overlap** (`bool`): Enable overlapping QK and PV GEMMs within a warp group.
  Default True. When enabled, the QK GEMM for the next tile can overlap with the PV GEMM
  of the current tile.
- **mma_pv_is_rs** (`bool`): Whether PV MMA uses register-to-SMEM (True) or SMEM-to-SMEM (False).
  Default True. Using registers avoids SMEM bandwidth pressure.
- **paged_kv_non_tma** (`bool`): If True, use cp.async instead of TMA for KV loads. Required for
  paged KV with page_size != tile_n.

### Warp Group Architecture

The SM90 kernel uses a producer-consumer model with specialized warp roles:

| Warp Group | Role | Threads | Registers |
|---|---|---|---|
| Warp 0 (producer) | TMA loads | 32 | 24-56 |
| Warp Groups 1-N (consumer) | MMA compute | 128 per WG | 160-256 |

**Thread counts:**
- 1 MMA WG: 160 threads (32 producer + 128 consumer)
- 2 MMA WGs: 288 threads (32 producer + 256 consumer)
- 3 MMA WGs: 416 threads (32 producer + 384 consumer)

### Pipeline Configuration

The SM90 kernel uses three separate pipelines:

1. **Pipeline Q** (`PipelineTmaAsync` or `PipelineCpAsync`): Manages Q tile loads
   - TMA: single warp (warp 0) loads, warpgroup consumes
   - cp.async: full warp group loads, warpgroup consumes
   - 1 stage (no double buffering for Q)

2. **Pipeline K** (`PipelineTmaAsync` or `PipelineCpAsync`): Manages K tile loads
   - `num_stages` stages for double/triple buffering
   - TMA or cp.async depending on configuration

3. **Pipeline V** (`PipelineTmaAsync` or `PipelineCpAsync`): Manages V tile loads
   - `num_stages` stages, same as K pipeline
   - Producer commits after TMA/cp.async completion

### Shared Memory Layout

Uses Hopper WGMMA-optimized layouts from `sm90_utils_basic.get_smem_layout_atom()`:
- Supports swizzling for bank conflict avoidance
- Layouts accommodate both K-major (for QK MMA) and MN-major (for PV MMA) views
- P layout stored in SMEM if `mma_pv_is_rs` is False

### `load` Method

```python
@cute.jit
def load(self, mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K,
         tma_atom_V, pipeline_k, pipeline_v, pipeline_q, ...)
```

Persistent producer loop that loads Q, K, V tiles via TMA or cp.async.

**Features:**
- TMA path: Only warp 0 performs loads; cp.async path: full warp group loads
- Block sparsity support: Can skip loading K/V tiles that are masked out
- Paged KV support: Translates virtual KV coordinates to physical page indices
- Intra-WG overlap: Alternates K and V loads to overlap K-V pipeline stages

**Paged KV TMA path:**
When `mPageTable` is provided and `use_tma_KV` is True, the producer:
1. Looks up physical page index from page table: `mPageTable[batch_idx, n_block]`
2. Passes the physical page index as the TMA source coordinate

**Paged KV cp.async path:**
When `paged_kv_non_tma` is True, uses `PagedKVManager` to handle page table indirection
with cp.async loads for arbitrary page sizes.

### `mma` Method

```python
@cute.jit
def mma(self, tiled_mma_qk, tiled_mma_pv, mO, mLSE, sQ, sK, sVt, ...)
```

Persistent consumer loop that performs MMA computation and writes output.

**Algorithm per tile:**
1. Wait for Q tile (`pipeline_q.consumer_wait`)
2. Create `Softmax` instance for this tile
3. Create `AttentionMask` for this tile
4. Mainloop over N blocks:
   - If `intra_wg_overlap`: use `mma_one_n_block_intrawg_overlap`
   - Else: use `mma_one_n_block`
5. Apply learnable sink (if provided)
6. Finalize softmax: `softmax.finalize()`
7. Rescale O: `softmax.rescale_O()`
8. Call `epilogue()` to write O and LSE

### `mma_one_n_block_intrawg_overlap`

Overlapped version of the N-block processing that allows QK and PV GEMMs to execute concurrently
within a warp group.

**Overlap strategy:**
1. Wait for K pipeline, start QK GEMM (async, WGMMA with wg_wait=-1)
2. Wait for V pipeline, start PV GEMM (async, WGMMA with wg_wait=-1)
3. Barrier sync between warp groups
4. Wait for QK GEMM completion (`warpgroup.wait_group(1)`)
5. Apply masking and online softmax
6. Wait for PV GEMM completion (`warpgroup.wait_group(0)`)
7. Convert P, rescale O

### `mma_one_n_block`

Non-overlapped version with simpler sequencing:
1. Wait for K, complete QK GEMM
2. Apply masking and online softmax
3. Wait for V, complete PV GEMM

### Block Sparsity Support

When `blocksparse_tensors` is provided:
- Producer: calls `produce_block_sparse_loads()` to iterate only over non-zero KV blocks
- Consumer: calls `consume_block_sparse_loads()` to process only loaded blocks
- Supports `q_subtile_factor` for subtiling Q dimension

---

## FlashAttentionForwardSm100

**Location:** `flash_attn/cute/flash_fwd_sm100.py`
**Inheritance:** None (standalone class, not inheriting from ForwardBase)

Blackwell-class forward kernel using UMMA (Unified MMA) instructions, TMEM (Tensor Memory),
and advanced scheduling features.

### Constructor

```python
class FlashAttentionForwardSm100:
    def __init__(
        self, head_dim, head_dim_v=None,
        qhead_per_kvhead=1, is_causal=False, is_local=False,
        is_split_kv=False, pack_gqa=False, q_subtile_factor=None,
        m_block_size=128, n_block_size=128, q_stage=2,
        is_persistent=True, score_mod=None, mask_mod=None,
        has_aux_tensors=False, paged_kv_non_tma=False,
        is_varlen_q=False, use_2cta_instrs=False,
        use_clc_scheduler=False,
    )
```

#### Key Parameters

- **is_split_kv** (`bool`): Enable SplitKV for long sequences. Splits KV across multiple CTA blocks
  and combines results in a separate kernel.
- **q_stage** (`Constexpr[int]`): Number of Q pipeline stages (1 or 2).
- **use_2cta_instrs** (`bool`): Use 2-CTA MMA instructions for doubled compute throughput.
  Required for head_dim=192 with head_dim_v=128.
- **use_clc_scheduler** (`bool`): Use Cooperative Launch Cluster dynamic tile scheduler for
  improved load balancing.

### Warp Architecture

The SM100 kernel uses 16 warps (512 threads) with specialized roles:

| Warp IDs | Role | Count |
|---|---|---|
| 0-3 | Softmax 0 | 4 warps |
| 4-7 | Softmax 1 / Q loading | 4 warps |
| 8-11 | Correction / Epilogue | 4 warps |
| 12 | MMA | 1 warp |
| 13-14 | Load / Epilogue | 1-2 warps |
| 15 | Empty / CLC Scheduler | 1 warp |

### TMEM (Tensor Memory) Layout

TMEM is used for storing intermediate S, P, and O tensors:

```
TMEM Layout (for hd128, 2CTA):
Offset 0-127:   S stage 0 (n_block_size columns)
Offset 128-255: S stage 1 (n_block_size columns)
Offset 256-383: O stage 0 (head_dim_v columns)
Offset 384-511: O stage 1 (head_dim_v columns)
```

**TMEM offsets:**
- `tmem_s_offset`: Attention score (S) storage
- `tmem_o_offset`: Output accumulator (O) storage
- `tmem_p_offset`: Probability (P) storage (overlaps with S)

### Tuning Configuration

The `_TUNING_CONFIG` dictionary maps kernel configurations to register counts and exp2 emulation
parameters:

```python
_TUNING_CONFIG = {
    (use_2cta, is_causal, head_dim_padded, is_sm103): {
        "ex2_emu_freq": int,       # exp2 emulation frequency
        "ex2_emu_start_frg": int,  # start fragment for emulation
        "num_regs_softmax": int,   # registers for softmax warps
        "num_regs_correction": int,# registers for correction warps
    }
}
```

**exp2 emulation:** On SM100 (non-SM103), the native `ex2.approx.ftz.f32` instruction is relatively
slow. The kernel uses a polynomial emulation on a fraction of elements (`ex2_emu_freq`) to trade
SFU throughput for FMA throughput, which is faster overall.

### Pipeline Stages

```python
def _setup_attributes(self):
    self.kv_stage = computed  # KV pipeline stages (auto-computed from SMEM budget)
    self.s_stage = 2  # Score pipeline stages (always 2)
```

KV stages are automatically computed based on available shared memory:
```
kv_stage = (224 KB - smem_size_Q_O) / smem_size_kv_per_stage
```

### Execution Flow

The SM100 kernel separates computation into distinct warp roles:

1. **Load Warp:** TMA-based Q, K, V loading with persistent scheduling
2. **MMA Warp:** UMMA QK and PV matrix multiplication
3. **Softmax Warps:** Online softmax from TMEM, exp2 computation, P generation
4. **Correction Warps:** Online O rescaling, TMEM-to-GMEM output writing
5. **Epilogue Warps:** Final O store with sequence length predicates

### Block Sparsity (SM100)

SM100-specific block sparsity support:
- `produce_block_sparse_loads_sm100()`: Producer-side sparse KV block loading
- `softmax_block_sparse_sm100()`: Consumer-side sparse softmax processing
- `handle_block_sparse_empty_tile_correction_sm100()`: Handles empty tile edge case

---

## FlashAttentionForwardSm120

**Location:** `flash_attn/cute/flash_fwd_sm120.py`
**Inheritance:** `FlashAttentionForwardSm80`

SM120 (Blackwell GeForce / DGX Spark) variant that uses SM80-era MMA instructions but
has reduced shared memory capacity (99 KB vs 163 KB on SM80).

### Key Differences from SM80

- **Shared memory:** 99 KB vs 163 KB on SM80
- **MMA instructions:** Same `mma.sync.aligned.m16n8k16` as SM80
- **Arch field:** Kept at 80 to use cp.async code paths (no TMA for output)
- The GPU compilation target is determined at runtime, not by the `arch` field

### `can_implement` (override)

```python
@staticmethod
def can_implement(dtype, head_dim, head_dim_v, tile_m, tile_n,
                  num_stages, num_threads, is_causal, Q_in_regs=False) -> bool
```

Same validation logic as SM80 but uses SM120's 99 KB shared memory limit.

---

## FlashAttentionForwardCombine

**Location:** `flash_attn/cute/flash_fwd_combine.py`

Combine kernel for merging partial results from SplitKV attention. When the KV sequence is split
across multiple CTA blocks, each block produces a partial `O_partial` and `LSE_partial`. This kernel
combines them into the final attention output.

### Constructor

```python
class FlashAttentionForwardCombine:
    def __init__(
        self, dtype, dtype_partial, head_dim,
        tile_m=8, k_block_size=64, log_max_splits=4,
        num_threads=256, stages=4,
    )
```

#### Parameters

- **dtype** (`Type[cutlass.Numeric]`): Output data type. Supports Float16, BFloat16, Float32.
- **dtype_partial** (`Type[cutlass.Numeric]`): Partial accumulation data type.
- **head_dim** (`int`): Head dimension.
- **tile_m** (`int`): M block size for the combine kernel (default 8).
- **k_block_size** (`int`): K block size for partial O loads (default 64).
- **log_max_splits** (`int`): log2 of maximum number of KV splits (default 4, max 256).
- **num_threads** (`int`): Threads per block (default 256).
- **stages** (`int`): Pipeline stages for O partial loads (default 4).

### Algorithm

1. **Load LSE partial:** Load `LSE_partial[num_splits, seqlen, num_heads]` into shared memory
2. **Compute final LSE:**
   - Find max LSE across splits using warp reduction
   - Compute `scale[s] = exp(LSE[s] - max_LSE)` for each split
   - Compute `lse_sum = sum(scale[s])`
   - Compute `final_LSE = max_LSE + log(lse_sum)`
   - Normalize scales: `scale[s] /= lse_sum`
3. **Load O partial:** Pipeline O partial tiles from global memory to shared memory
4. **Accumulate final O:**
   - For each split, multiply by scale and accumulate
   - `O_final += scale[s] * O_partial[s]`
5. **Write output:** Store final O to global memory

### Grid Dimensions

```
grid = (ceil_div(seqlen * num_heads, tile_m),
        ceil_div(head_dim, k_block_size),
        batch_size)
```

### Variable Length Support

Supports variable-length sequences via `cu_seqlens` and `seqused` parameters. Uses
`SeqlenInfo` for offset computation.

### Dynamic Splits

When `num_splits_dynamic_ptr` is provided, the number of splits is read per-batch at runtime,
allowing different sequences in a batch to use different split counts.

---

## FlashAttentionForwardMLASm100

**Location:** `flash_attn/cute/flash_fwd_mla_sm100.py`

Multi-Latent Attention (MLA) specific forward kernel for Blackwell. MLA uses a low-rank
projection for the KV cache, resulting in different head_dim_q and head_dim_kv.

This kernel handles the case where the KV head dimension differs significantly from the
query head dimension, optimizing for the compressed KV representation used in MLA.

---

## BlackwellFusedMultiHeadAttentionForward

**Location:** `flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`

Specialized Blackwell kernel for head_dim=256 with 2CTA instructions. Uses a dedicated
architecture with TMEM-based softmax and correction warps.

### Constructor

```python
class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self, head_dim=256, head_dim_v=256,
        qhead_per_kvhead=1, is_causal=False, is_local=False,
        is_split_kv=False, m_block_size=128, n_block_size=128,
        q_stage=2, is_persistent=True,
        use_2cta_instrs=False, use_clc_scheduler=False,
    )
```

**Restrictions:**
- Only supports (head_dim, head_dim_v) = (256, 256)
- Only supports tile_m=128, tile_n=128
- No score_mod, mask_mod, aux_tensors, or block sparsity support

### Warp Architecture (16 warps = 512 threads)

| Warp IDs | Role |
|---|---|
| 0-3 | Softmax (S tile processing, exp2, P generation) |
| 4-7 | Correction (online rescaling, output finalize) |
| 8 | MMA (UMMA QK and PV) |
| 9 | Load (TMA Q, K, V loading) |
| 10-11 | Empty (register deallocation) |

### TMEM Layout

```
Offset 0-127:   S accumulator (128 columns for n_block_size)
Offset 256-511: O accumulator (256 columns for head_dim_v=256)
```

### Pipeline Stages

- **Q stage:** 2 (iterations_qk = head_dim / 128 = 2)
- **KV stage:** 4
- **QK accumulator stage:** 2
- **MMA-correction stage:** 1

### Execution Flow

1. **Load Warp:** Persistent TMA loading loop
   - Load Q tile across `iterations_qk` phases
   - Interleave K and V loads with page table lookups (paged KV)
   - Prefetch next K page while V TMA is in flight

2. **MMA Warp:** UMMA computation
   - QK GEMM: `S = Q @ K^T` with multi-iteration along head_dim
   - PV GEMM: `O += P @ V` with multi-iteration along head_dim
   - Accumulate flag management for online rescaling

3. **Softmax Warps:** TMEM-based softmax
   - Load S from TMEM using `tcgen05.Ld32x32bOp`
   - Apply causal/local masking
   - Compute row_max, exp2 (with optional polynomial emulation)
   - Store P to TMEM for PV GEMM
   - Compute row_sum
   - Store stats (prev_max, current_max) to TMEM for correction

4. **Correction Warps:** Online rescaling and output
   - Read stats from TMEM
   - Compute rescale factor: `exp(scale_log2 * (prev_max - current_max))`
   - Rescale O in TMEM
   - Finalize: divide by row_sum, convert to dtype, write to GMEM

### `softmax_step`

```python
@cute.jit
def softmax_step(self, mask_args, value_args, tensor_args, pipeline_args)
    -> Tuple[Float32, Float32, ...]
```

Processes one N-block of attention scores through softmax.

**Algorithm:**
1. Load S tile from TMEM to registers
2. Apply masking (causal/local/sequence length)
3. Compute row_max (with previous max tracking)
4. Store stats (prev_max, current_max_safe) to TMEM for correction warps
5. Compute `S_scaled = scale_log2 * S - scale_log2 * row_max`
6. Compute `P = exp2(S_scaled)` with optional emulation
7. Store P to TMEM
8. Compute row_sum via packed FMA reduction

### `correction_rescale`

```python
@cute.jit
def correction_rescale(self, scale_softmax_log2, stats_args, o_args, epi_tile)
```

Rescales O accumulator between N-block iterations.

**Algorithm:**
1. Read stats (prev_max, current_max) from TMEM
2. Compute `scale = exp2(scale_log2 * (prev_max - current_max))`
3. Load O from TMEM, multiply by scale, store back to TMEM

### `correction_epilog`

```python
@cute.jit
def correction_epilog(self, value_args, sum_args, o_args, epi_tile)
```

Final output normalization and writing.

**Algorithm:**
1. Read row_sum from shared memory
2. Compute `scale = scale_output / row_sum`
3. Load O from TMEM, multiply by scale
4. Convert to output dtype
5. Write to global memory with sequence length predicates

---

## Kernel Pipeline and Execution Flow

### General Forward Attention Pipeline

All forward kernels follow the same high-level algorithm:

```
For each (batch, head) tile:
    Load Q tile
    Initialize acc_O = 0, row_max = -inf, row_sum = 0

    For each KV block (n_block = n_block_max-1 down to n_block_min):
        Load K[n_block], V[n_block]

        # Compute attention scores
        S = Q @ K[n_block]^T       # GEMM

        # Apply masking
        Apply mask_mod and causal/local masks to S

        # Online softmax update
        row_max_new = max(row_max, max(S))
        row_scale = exp(row_max - row_max_new)
        acc_O *= row_scale          # Rescale existing O
        P = exp(S - row_max_new)    # Compute probabilities
        row_sum = row_sum * row_scale + sum(P)

        # Accumulate output
        acc_O += P @ V[n_block]     # GEMM

    # Finalize
    O = acc_O / row_sum
    LSE = row_max + log(row_sum)
    Write O and LSE to global memory
```

---

## TMA Usage Patterns

### TMA for Q Loading

```python
# Create TMA descriptor for Q
tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),
    mQ, sQ_layout,
    (tile_m, tile_hdim),  # TMA tile shape
)

# Load Q tile via TMA (producer warp)
pipeline_q.producer_acquire(...)
cute.copy(tma_atom_Q, gQ, sQ, tma_bar_ptr=barrier)
```

### TMA for KV Loading

```python
# TMA load function with pipeline integration
tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
    tma_atom_K, 0, layout_1, gK, sK
)
tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)

# Load K tile
tma_load_K_fn(src_idx=n_block, producer_state=state)
```

### TMA for O Store (SM90+)

```python
# TMA bulk store for output
tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(),
    mO, sO_layout,
    (tile_m, tile_hdimv),
)

# Store O tile via TMA
cute.arch.fence_view_async_shared()
barrier_arrive(...)
cute.copy(tma_atom_O, sO, gO, tma_bar_ptr=barrier)
```

### TMA with Paged KV

When `mPageTable` is provided, TMA loads use physical page indices:
```python
page_idx = mPageTable[batch_idx, n_block]
tma_load_K_fn(src_idx=page_idx, producer_state=state)
```

---

## SplitKV Mechanism

SplitKV divides the KV sequence into chunks processed by separate CTA blocks, then combines
the partial results.

### When to Use SplitKV

- Long KV sequences where a single CTA would have too many N blocks
- When KV sequence length exceeds the effective tile coverage for a single CTA

### Implementation

1. **Main kernel** (`FlashAttentionForwardSm100` with `is_split_kv=True`):
   - Each CTA processes a subset of KV blocks
   - Outputs partial `O_partial[num_splits, batch, seqlen, heads, headdim]`
   - Outputs partial `LSE_partial[num_splits, batch, seqlen, heads]`

2. **Combine kernel** (`FlashAttentionForwardCombine`):
   - Loads all partial O and LSE values
   - Computes weighted combination using LSE-based scaling
   - Writes final O and LSE

### Mathematical Foundation

For split-i producing `(O_i, LSE_i)`:
```
scale_i = exp(LSE_i - max(LSE_0, ..., LSE_{S-1}))
O_final = sum(scale_i * O_partial_i) / sum(scale_i)
LSE_final = max(LSE) + log(sum(scale_i))
```

---

## Paged KV Cache

Paged KV cache allows non-contiguous KV storage using a page table for memory efficiency
in inference serving.

### Page Table Format

```
mPageTable: shape (batch, max_pages_per_seq)
mPageTable[batch_idx, kv_block_idx] = physical_page_idx
```

### TMA Paged KV

When `use_tma_KV=True` and `page_size == tile_n`:
```python
# Direct TMA load with physical page index
page_idx = mPageTable[batch_idx, n_block]
tma_load_K_fn(src_idx=page_idx, producer_state=state)
```

### cp.async Paged KV

When `paged_kv_non_tma=True` (page_size != tile_n):
```python
paged_kv_manager = PagedKVManager.create(
    mPageTable, mK, mV,
    page_size_divmod, batch_idx, head_idx_kv,
    tidx, seqlen_k, leftpad_k,
    tile_n, tile_hdim, tile_hdimv,
    num_threads, dtype, arch=arch,
)
paged_kv_manager.load_KV(block, smem_slice, "K")
```

---

## Persistent Kernels

Persistent kernels keep CTAs active across multiple tiles, reducing launch overhead and
improving load balancing.

### Static Persistent Scheduler

```python
TileScheduler = StaticPersistentTileScheduler
```

- Pre-computes tile assignments at kernel launch
- Each CTA processes tiles in order
- Simple but may have load imbalance

### LPT (Longest Processing Time) Scheduler

```python
TileScheduler = SingleTileLPTScheduler
```

- Used for causal/local attention
- Assigns tiles in reverse order to balance work (later tiles have fewer N blocks)

### CLC (Cooperative Launch Cluster) Dynamic Scheduler

```python
use_clc_scheduler=True
```

- Uses hardware CLC feature for dynamic work distribution
- A scheduler warp prefetches next tile assignments
- Best load balancing for heterogeneous workloads

### Persistent Loop Structure

```python
tile_scheduler = TileScheduler.create(params)
work_tile = tile_scheduler.initial_work_tile_info()

while work_tile.is_valid_tile:
    m_block, head_idx, batch_idx, _ = work_tile.tile_idx
    # ... process tile ...
    tile_scheduler.prefetch_next_work()
    tile_scheduler.advance_to_next_work()
    work_tile = tile_scheduler.get_current_work()
```

---

## 2CTA Instructions

2CTA (2-CTA) instructions use two CTAs in a cluster to double the MMA throughput.

### When 2CTA is Used

- `use_2cta_instrs=True` in kernel configuration
- Cluster shape: `(2, 1)` (2 CTAs along M dimension)
- Each CTA owns `m_block_size` rows of Q
- The 2CTA MMA spans both CTAs' Q rows

### CTA Coordination

```python
cluster_shape_mn = (2, 1)  # 2 CTAs along M
cluster_layout_vmnk = tiled_divide(cluster_shape_mnk, thr_id_shape)

# TMA multicast: both CTAs receive the same K/V tile
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    tma_load_op, mK, sK_layout,
    (tile_n, tile_hdim),  # Per-CTA tile
    qk_tiled_mma,
    cluster_layout_vmnk.shape,  # Cluster layout for multicast
)
```

### MMA Tiler Adjustment

```python
# Standard: mma_tiler = (m_block_size, n_block_size, head_dim)
# 2CTA:     mma_tiler = (2 * m_block_size, n_block_size, head_dim)
mma_tiler_qk = (cta_group_size * m_block_size, n_block_size, head_dim_padded)
```

### Register Allocation

With 2CTA, the total register budget is shared across more operations:
```python
num_regs_softmax = 176      # Softmax warps
num_regs_correction = 88    # Correction warps
num_regs_other = 512 - 176*2 - 88  # MMA and load warps
```

### Head Dimension 192 Special Case

For head_dim=192, head_dim_v=128 with 2CTA:
- Uneven KV shared memory: stages have different sizes
- Special TMEM offset layout to accommodate larger S accumulator
- Extra SMEM offset adjustment for the middle KV stage
