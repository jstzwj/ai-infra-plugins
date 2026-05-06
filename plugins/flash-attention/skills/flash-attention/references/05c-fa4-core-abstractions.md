# FlashAttention-4 Core Abstractions Reference

This document provides an exhaustive reference for the core abstraction modules in FlashAttention-4 (FA4).
These modules provide the building blocks used by all forward and backward kernel classes across
architectures (SM80, SM90, SM100, SM120).

## Table of Contents

1. [Softmax (`softmax.py`)](#1-softmax)
2. [Mask (`mask.py`)](#2-mask)
3. [Block Info (`block_info.py`)](#3-block-info)
4. [Sequence Length Info (`seqlen_info.py`)](#4-sequence-length-info)
5. [Pipeline (`pipeline.py`)](#5-pipeline)
6. [Tile Scheduler (`tile_scheduler.py`)](#6-tile-scheduler)
7. [Copy Utilities (`copy_utils.py`)](#7-copy-utilities)
8. [Named Barrier (`named_barrier.py`)](#8-named-barrier)
9. [Pack GQA (`pack_gqa.py`)](#9-pack-gqa)
10. [Paged KV (`paged_kv.py`)](#10-paged-kv)
11. [Fast Math (`fast_math.py`)](#11-fast-math)
12. [Block Sparse Utils (`block_sparse_utils.py`)](#12-block-sparse-utils)
13. [Block Sparsity (`block_sparsity.py`)](#13-block-sparsity)

---

## 1. Softmax

**File**: `flash_attn/cute/softmax.py`

The softmax module implements online softmax computation for flash attention. It provides the core
numerical logic for computing attention probabilities in a memory-efficient streaming fashion.

### Class: `Softmax`

The base softmax class for SM80/SM90 architectures. Operates on register-memory tensors holding
row_max and row_sum statistics for multiple rows of the attention score matrix.

**Constructor Fields**:
- `scale_log2: Float32` - Precomputed `softmax_scale * log2(e)` for use with exp2
- `num_rows: cutlass.Constexpr[int]` - Number of rows in the tile (typically tile_m)
- `row_max: cute.Tensor` - Register tensor tracking per-row maximum values
- `row_sum: cute.Tensor` - Register tensor tracking per-row sum of exp values
- `arch: cutlass.Constexpr[int]` - Target architecture (80 or 90)
- `softmax_scale: Float32 | None` - Scale factor (None when folded into scale_log2)

**Static Method: `create`**
```python
@staticmethod
def create(scale_log2, num_rows, arch=80, softmax_scale=None)
```
Allocates register tensors for `row_max` and `row_sum` and returns a `Softmax` instance.

**Method: `reset`**
Initializes `row_max` to `-inf` and `row_sum` to `0.0`, preparing for a new tile.

**Method: `_compute_row_max`**
```python
def _compute_row_max(self, acc_S_row, init_val=None)
```
Reduces the input tensor to find the maximum value. When `init_val` is provided (non-first block),
the reduction includes the previous row_max value. Dispatches to `utils.fmax_reduce` with
architecture-specific optimization.

**Method: `_compute_row_sum`**
```python
def _compute_row_sum(self, acc_S_row_exp, init_val=None)
```
Reduces the exp-transformed row values to compute their sum. Uses `utils.fadd_reduce` with
architecture-specific optimization.

**Method: `online_softmax`**
```python
@cute.jit
def online_softmax(self, acc_S, is_first=False, check_inf=True)
```
The core online softmax algorithm, processing one n_block of attention scores. For each row:

1. Extract the row from the accumulator tensor (`acc_S_mn[r, None]`)
2. Compute new row_max by reducing the current row with previous max
3. Warp-level reduction for row_max across warp groups
4. Handle `-inf` case (all masked out) by setting safe max to 0
5. Compute `exp2(acc_S * scale_log2 - row_max_cur_scaled)` using fastmath
6. If first block: `row_scale[r] = 1.0`, row_sum = sum of exp values
7. If not first: `row_scale[r] = exp2((row_max_prev - row_max_cur) * scale_log2)`,
   row_sum = previous_sum * row_scale + new_exp_sum
8. Store the exp values back into the accumulator for later P*V multiplication

Returns `row_scale` tensor used to rescale the accumulated output O.

**Method: `finalize`**
```python
@cute.jit
def finalize(self, final_scale=1.0, sink_val=None)
```
Completes the softmax computation after all n_blocks are processed:

1. Warp-level reduction for `row_sum` (quad reduction across warp groups)
2. Optional sink value integration (for learnable sink attention)
3. Compute `row_scale = rcp_approx(row_sum) * final_scale` for output normalization
4. Compute log-sum-exp: `LSE = row_max * scale_log2 + log2(row_sum) * LN2`
5. Handle zero/NaN row_sum by setting LSE to `-inf` and scale to `1.0/rcp(1.0)`

Returns the final `row_scale` tensor.

**Method: `rescale_O`**
```python
@cute.jit
def rescale_O(self, acc_O, row_scale)
```
Scales each row of the output accumulator by the corresponding row scale value. Used to
maintain numerical stability across n_block iterations.

### Class: `SoftmaxSm100`

Extends `Softmax` for Blackwell (SM100) architecture. Key differences:
- Operates on single-row statistics (TMEM-based, `num_rows=1`)
- Adds `rescale_threshold` for skip-rescale optimization
- Adds `max_offset` for learnable sink attention with bias

**Additional Methods**:

**`compute_row_max_local`** / **`update_row_max_from_local`**
Split version of update_row_max that separates the reduction from the state update.
The `rescale_threshold` optimization: if the exp of the old-to-new max ratio is close to 1
(below threshold), skip the rescale entirely, keeping the old max value.

**`update_row_max`**
Combines row_max computation with the state update and rescale computation. Returns
`(row_max_safe, acc_scale)` where `acc_scale` is `exp2((old_max - new_max) * scale_log2)`.

**`update_row_sum`**
Updates the running row sum: `row_sum = row_sum * acc_scale + new_exp_sum`.

**`scale_subtract_rowmax`**
```python
@cute.jit
def scale_subtract_rowmax(self, acc_S_row, row_max)
```
Computes `acc_S * scale_log2 - row_max * scale_log2 + max_offset` using packed FP32 FMA
operations (processing 2 elements at a time via `fma_packed_f32x2`). This is the first step
before exp2 conversion.

**`apply_exp2_convert`**
```python
@cute.jit
def apply_exp2_convert(self, acc_S_row, acc_S_row_converted, ex2_emu_freq=0, ex2_emu_res=4, ex2_emu_start_frg=0)
```
Applies exp2 to the scaled scores and converts to the target dtype. The `ex2_emu_freq` parameter
controls exp2 emulation frequency for mixed precision: every `ex2_emu_freq` elements, some are
computed with the fast hardware exp2 and others with software emulation for better accuracy.

**`scale_apply_exp2_convert`**
Combined scale-subtract-max + exp2 + convert in one pass.

### Free Functions

**`floor_if_packed`**
```python
@cute.jit
def floor_if_packed(q_idx, qhead_per_kvhead)
```
Divides q_idx by the PackGQA factor when `qhead_per_kvhead > 1`, mapping from packed
query index space to logical query index space.

**`apply_score_mod_inner`**
```python
@cute.jit
def apply_score_mod_inner(score_tensor, index_tensor, score_mod, batch_idx, head_idx,
                           softmax_scale, vec_size, qk_acc_dtype, aux_tensors,
                           fastdiv_mods, seqlen_info, constant_q_idx,
                           qhead_per_kvhead=1, transpose_indices=False)
```
Applies user-defined score modification (e.g., softcap, custom score transforms) to attention
scores. Processes elements in vectors of `vec_size` for efficiency. Handles:
- PackGQA head index remapping (computing per-element head indices)
- Aux tensor index wrapping via FastDivmodDivisor
- SSA conversion for JIT compatibility
- Transposed indices for backward pass

**`apply_score_mod_bwd_inner`**
Backward counterpart that applies the score modification's backward (Jacobian) pass. Takes
both `grad_tensor` (dlogits) and `score_tensor` (pre-mod scores) and computes the gradient
through the score modification.

---

## 2. Mask

**File**: `flash_attn/cute/mask.py`

The mask module handles all attention masking operations: sequence length boundary masking,
causal masking, local/sliding window masking, custom mask_mod functions (FlexAttention),
and R2P (Register-to-Predicate) bit-mask optimization.

### R2P Bit-mask Functions

**`r2p_bitmask_below`**
```python
@cute.jit
def r2p_bitmask_below(limit, s) -> Uint32
```
Generates a 32-bit bitmask for R2P instructions. Bits set (1) for positions `< limit` within
chunk `s`, bits clear (0) for positions >= limit. Each chunk covers `MASK_R2P_CHUNK_SIZE=32`
consecutive columns.

**`r2p_bitmask_above`**
```python
@cute.jit
def r2p_bitmask_above(limit, s) -> Uint32
```
Inverse: bits set for positions >= limit. Used for the left boundary of sliding windows.

**`mask_r2p_lambda`**
```python
@cute.jit
def mask_r2p_lambda(X, mask_gen_fn, rank1=False)
```
Applies R2P masking using a custom bitmask generator function. Iterates over 32-element chunks
and applies the generated bitmask to each element. When a bit is 0, the corresponding element
is set to `-inf` (masked out). This function must use `range_constexpr` for the compiler to
generate R2P instructions.

**`sm90_col_to_r2p_idx`**
```python
@cute.jit
def sm90_col_to_r2p_idx(col_limit) -> Int32
```
Converts SM90 WGMMA accumulator column coordinates (non-contiguous: 0,1,8,9,16,17,...)
to element indices (contiguous: 0,1,2,3,4,5,...) for R2P bitmask generation.

**`row_to_r2p_idx`**
```python
@cute.jit
def row_to_r2p_idx(x, num_rep, num_wg) -> Int32
```
Converts row coordinates to R2P element indices for the warp-group interleaved TMEM layout
used in SM100 backward. Handles the gap between warp groups by clamping.

### Class: `AttentionMask`

The primary masking class for SM80/SM90 forward and backward passes. Handles all mask types
through a unified interface.

**Constructor Fields**:
- `tile_m: Constexpr[int]` - M tile dimension
- `tile_n: Constexpr[int]` - N tile dimension
- `seqlen_info: SeqlenInfoQK` - Sequence length information
- `window_size_left: Optional[Int32]` - Left sliding window boundary
- `window_size_right: Optional[Int32]` - Right sliding window boundary (also used for causal)
- `qhead_per_kvhead_packgqa: Constexpr[int]` - PackGQA factor
- `swap_AB: Constexpr[bool]` - Whether A/B are swapped (backward pass)

**Method: `apply_mask`**
```python
@cute.jit
def apply_mask(self, acc_S, batch_idx, head_idx, m_block, n_block, thr_mma,
               mask_seqlen, mask_causal, mask_local=False, mask_mod=None, ...)
```
Applies masking to the attention score accumulator. Three main paths:

1. **Sequence length only**: Masks columns beyond `seqlen_k`. Uses R2P optimization when
   `swap_AB=False` (forward), element-wise loop otherwise.

2. **Custom mask_mod (FlexAttention)**: For each (r, c) element, computes global
   (q_idx, kv_idx) coordinates and calls the user's mask_mod function. Handles PackGQA
   head index remapping and aux tensor index wrapping.

3. **Causal or local window**: Computes column limits based on row position, causal offset,
   and window sizes. Supports:
   - Pure causal: `col_limit = row_idx + causal_row_offset`
   - Sliding window: separate left and right column limits
   - R2P optimization for non-transposed cases
   - Element-wise loops for transposed cases

**Method: `apply_mask_sm100`**
SM100-specific masking that operates on TMEM-loaded score tensors with `thr_tmem_load` indexing.
Similar logic to `apply_mask` but adapted for the Blackwell accumulator layout.

**Method: `apply_mask_sm100_transposed`**
SM100 backward-specific masking where S = K @ Q.T (transposed). The coordinate convention
is swapped: ROW corresponds to Q (m_block), COL corresponds to KV (n_block). Supports
`is_full_block` optimization to skip mask_mod for fully valid tiles.

### SM100 Fused Mask System

**`Sm100MaskEnum`**
Enumeration of mask types for hardware-fused masking:
- `NO_MASK` - No masking applied
- `RESIDUAL_MASK` - Variable sequence length boundary
- `CAUSAL_MASK` - Autoregressive causal mask
- `WINDOW_MASK` - Sliding window with causal
- `WINDOW_MASK_INFERENCE` - End-aligned Q/K for inference

**`Sm100FusedMask`**
A collection of static methods that compute trip counts and mask boundaries for the SM100
persistent kernel scheduler. Key methods:

- `get_trip_count` - Number of KV tile iterations for given block coordinates
- `get_trip_start` - Starting KV tile index (for sliding window left boundary)
- `get_leading_mask_id` / `get_trailing_mask_id` - Tile ranges that need element-wise masking
- `get_unmasked_trip_count` - Tiles that can skip per-element masking entirely
- `apply_mask` / `apply_mask_via_causal_local` - Apply the actual mask to score elements

The fused mask system enables the SM100 kernel to skip masking entirely for interior tiles
(`unmasked_trip_count > 0`), only applying element-wise masks on boundary tiles.

---

## 3. Block Info

**File**: `flash_attn/cute/block_info.py`

### Class: `BlockInfo`

A frozen dataclass that encapsulates tile dimensions and masking parameters, providing methods
to compute valid block ranges for the attention loop.

**Fields**:
- `tile_m: Constexpr[int]` - M tile dimension (Q sequence direction)
- `tile_n: Constexpr[int]` - N tile dimension (KV sequence direction)
- `is_causal: Constexpr[bool]` - Whether causal masking is applied
- `is_local: Constexpr[bool]` - Whether local/sliding window masking is applied
- `is_split_kv: Constexpr[bool]` - Whether SplitKV is active
- `window_size_left: Optional[Int32]` - Left window boundary
- `window_size_right: Optional[Int32]` - Right window boundary
- `qhead_per_kvhead_packgqa: Constexpr[int]` - PackGQA factor

**Method: `get_n_block_min_max`**
```python
@cute.jit
def get_n_block_min_max(self, seqlen_info, m_block, split_idx=0, num_splits=1)
    -> Tuple[Int32, Int32]
```
Computes the valid range of n_blocks for a given m_block. For causal attention, the upper
bound is limited by the diagonal constraint. For local attention, both lower and upper bounds
are constrained by the window. For SplitKV, the range is further divided among splits.

**Method: `get_m_block_min_max`**
```python
@cute.jit
def get_m_block_min_max(self, seqlen_info, n_block) -> Tuple[Int32, Int32]
```
Inverse mapping: computes valid m_block range for a given n_block. Used in the backward pass
where the outer loop iterates over n_blocks.

**Method: `get_n_block_k_new_min_max`**
For append-KV operations, maps the full n_block range to the new-K index space by subtracting
`seqlen_k_og` (original K length).

**Method: `get_n_block_min_causal_local_mask`**
Computes the boundary between the "causal/local masked" region and the "fully unmasked" region
for a given m_block. Used to skip masking on interior tiles.

**Method: `get_n_block_min_before_local_mask`**
Computes where the local window's left boundary starts, separating fully unmasked tiles from
tiles that need left-boundary masking.

---

## 4. Sequence Length Info

**File**: `flash_attn/cute/seqlen_info.py`

### Class: `SeqlenInfo`

Simple sequence length info for single-tensor (Q or K) varlen tracking.

**Fields**: `offset`, `offset_padded`, `seqlen`, `has_cu_seqlens`

**`create`**: Constructs from `cu_seqlens` or `seqused` tensors, computing offset and padded
offset with alignment hints.

**`offset_batch`**: Offsets a tensor by batch index. Handles both fixed-stride (no cu_seqlens)
and ragged (with cu_seqlens) cases.

### Class: `SeqlenInfoQK`

Combined sequence length info for both Q and K sequences.

**Fields**: `offset_q`, `offset_k`, `padded_offset_q`, `padded_offset_k`, `seqlen_q`, `seqlen_k`,
and boolean flags for which info sources are available.

**`create`**: Constructs from separate Q and K sources (`mCuSeqlensQ`, `mCuSeqlensK`,
`mSeqUsedQ`, `mSeqUsedK`), with alignment hints for padded offsets.

**`offset_batch_Q` / `offset_batch_K`**: Offsets tensors by batch index. Supports:
- Fixed-stride indexing (no cu_seqlens)
- Ragged indexing with cu_seqlens
- PackGQA-aware ragged indexing
- Ragged tensor truncation via `copy_utils.offset_ragged_tensor`

### Class: `SeqlenInfoQKNewK`

Extended sequence length info for append-KV (inference) with new K token support.

**Additional Fields**:
- `leftpad_k` - Left padding for K
- `offset_k_new` - Offset into new K tensor
- `seqlen_k_og` - Original K length (before append)
- `seqlen_k_new` - Length of new K tokens
- `seqlen_rotary` - Position for rotary embedding computation

---

## 5. Pipeline

**File**: `flash_attn/cute/pipeline.py`

The pipeline module wraps CUTLASS pipeline primitives with FA4-specific extensions for
synchronization and elect_one optimization.

### Class: `PipelineStateSimple`

A lightweight pipeline state that stores index and phase in a single integer, using divmod
to extract them. For power-of-2 stages, this compiles to bit twiddling.

**Properties**:
- `index` - Current buffer index in the circular buffer
- `phase` - Current phase bit for barrier synchronization

**`advance`** - Increments to the next state (XOR for single-stage, increment for multi-stage).

### `make_pipeline_state`

Factory that creates `PipelineStateSimple` with correct initial values:
- Producer starts at `phase_index = stages` (flipped phase, empty buffer)
- Consumer starts at `phase_index = 0`

### Pipeline Wrappers

The module wraps CUTLASS pipeline classes with additional functionality:

**`NamedBarrier`** - Extended with `arrive_w_index` and `arrive_and_wait_w_index` methods that
add an index offset to the barrier ID.

**`PipelineAsync`** - Adds `elect_one_commit` and `elect_one_release` flags. When enabled,
only the elected thread in each warp signals the barrier, reducing barrier traffic.

**`PipelineCpAsync`** - Similar elect_one support for cp.async-based pipelines.

**`PipelineTmaAsync`** - Overrides `producer_acquire` to support `extra_tx_count` parameter
for TMA transactions that include additional bytes beyond the default tile size.

**`PipelineTmaUmma`** - SM100-specific TMA pipeline with leader-CTA gating on barrier arrivals.

**`PipelineUmmaAsync`** / **`PipelineAsyncUmma`** - SM100-specific pipeline variants.

All wrappers include `_w_index_phase` convenience methods that accept raw (index, phase)
instead of PipelineState objects.

---

## 6. Tile Scheduler

**File**: `flash_attn/cute/tile_scheduler.py`

### Scheduling Modes

**`SchedulingMode`** enum: `NONE`, `STATIC`, `DYNAMIC`, `CLC`

### Class: `ClcState`

Runtime state for CLC (Cluster Launch Control) hardware-based tile scheduling on Blackwell.
Wraps the hardware scheduler, pipeline, and producer/consumer states. Methods:
- `initial_work_tile_info` / `get_current_work` - Query the hardware scheduler
- `prefetch_next_work` - Issue CLC query for the next tile (producer side)
- `consumer_wait` / `consumer_release` - Synchronize on CLC responses
- `producer_tail` - Cleanup after last tile

### Class: `WorkTileInfo`

Extended work tile info with four axes: `(block, head, batch, split)`.

### `TileSchedulerProtocol`

Protocol defining the interface all schedulers must implement:
- `get_current_work() -> WorkTileInfo`
- `initial_work_tile_info() -> WorkTileInfo`
- `advance_to_next_work()`
- `prefetch_next_work()`
- `producer_tail()`

### Class: `SingleTileScheduler`

Simple scheduler where each CTA processes exactly one tile. Grid dimensions:
`(num_block, num_head * num_splits, num_batch)`.

Supports cluster indexing for 2CTA kernels. The `get_grid_shape` method rounds up the grid
to account for cluster dimensions.

### Class: `StaticPersistentTileScheduler`

Persistent kernel scheduler where CTAs loop over all tiles via grid-stride iteration.
Decodes tile index to (block, head, batch) using FastDivmodDivisor. Grid size is limited
to the number of SMs.

### Class: `SingleTileLPTScheduler`

L2-cache-optimized scheduler using Longest-Processing-Time-first (LPT) ordering and
L2 swizzling for improved cache locality.

**L2 Swizzle Logic**: Computes how many attention heads fit in L2 cache (50MB for K+V).
Tiles are grouped into L2-sized sections, and within each section, heads are processed
contiguously. The last (residual) section may have fewer heads.

**LPT Ordering**: Reverses the block order so that tiles with more work (longer sequences
in causal mode) are processed first, improving load balancing.

**CLC Support**: When `scheduling_mode == SchedulingMode.CLC`, uses hardware dynamic scheduling
with the `ClcState` wrapper. The `clc_work_to_coords` method converts CLC grid coordinates
to the scheduler's logical coordinates, applying LPT reversal.

### Class: `SingleTileLPTBwdScheduler`

Backward-specific LPT scheduler. Similar to the forward LPT scheduler but with
cluster-aware indexing. Uses SPT (Shortest-Processing-Time-first) by default for backward.

### Class: `SingleTileVarlenScheduler`

Variable-length sequence scheduler that handles packed batched sequences without padding.

**Coordinate Mapping (`_varlen_coord_map`)**:
Uses warp-level prefix sums to efficiently map a flat tile index to (block, head, batch)
coordinates:
1. Each lane loads the sequence length for its assigned batch
2. Warp prefix sum computes cumulative block counts
3. Binary search via `vote_ballot_sync` and `popc` finds the target batch
4. Within the batch, compute head and block indices

**L2 Swizzle + LPT**: When enabled, applies the same L2 swizzle pattern as `SingleTileLPTScheduler`,
but adapted for variable-length sequences where the number of blocks varies per batch.

**CLC Support**: Maps CLC hardware tile indices to varlen coordinates. Invalid CLC tiles
are mapped to `grid_dim[0]` (past the last valid index) to produce `is_valid=False`.

### SM100 FMHA Schedulers

**`Sm100FmhaStaticTileScheduler`**: Static scheduler for the SM100 FMHA kernel.
Supports both persistent and non-persistent modes. Problem shape is `(M_tiles, B, H)`.

**`Sm100FmhaClcDynamicTileScheduler`**: CLC dynamic scheduler for SM100 FMHA.
Uses hardware Cluster Launch Control for automatic load balancing. The FMHA CLC scheduler
maps hardware tile coordinates to `(m_idx, 0, (bid, hid))` format.

**`compute_sm100_fmha_grid` / `compute_sm100_fmha_grid_clc`**: Factory functions that
compute grid shapes and scheduler parameters from output tensor shapes.

---

## 7. Copy Utilities

**File**: `flash_attn/cute/copy_utils.py`

### Copy Operations

**`cvt_copy`**: Type-converting copy that handles dtype mismatch between source and destination.
If `src.element_type != dst.element_type`, creates an intermediate fragment with type conversion.

**`load_s2r`**: Shared memory to register copy using autovec.

**`get_copy_atom`**: Creates a copy atom with the specified bit width (up to 128 bits),
optionally using async copy (`cp.async`).

**`make_tmem_copy`**: Creates a tiled copy atom for TMEM (Tensor Memory) operations on SM100.
Computes the tiler and thread-value layout from TMEM copy properties.

**`copy`**: Generic copy with automatic copy atom creation.

**`tiled_copy_1d` / `tiled_copy_2d`**: Creates 1D or 2D tiled copy configurations with
specified thread layouts.

### Atomic and Bulk Operations

**`atomic_add_fp32x4`**: Performs a 128-bit atomic add of 4 FP32 values to global memory
using inline PTX `red.global.add.v4.f32`.

**`set_block_rank`**: Maps a shared memory pointer to the corresponding address at another
CTA rank in the cluster using `mapa.shared::cluster`.

**`store_shared_remote_fp32x4`**: Stores 4 FP32 values to a remote CTA's shared memory with
mbarrier completion using `st.async.shared::cluster.mbarrier::complete_tx::bytes`.

**`cpasync_bulk_s2cluster`**: Bulk copy from local shared memory to remote CTA's shared memory
within a cluster.

**`cpasync_bulk_g2s`**: Bulk copy from global memory to shared memory with TMA barrier.

**`cpasync_reduce_bulk_add_f32`**: Bulk reduction (add) from shared memory to global memory
using `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`.

### Factory Functions

**`cpasync_bulk_get_copy_fn`**: Returns a callable that performs bulk async copy for a given
source/destination tensor pair. Handles both single-stage and multi-stage (pipelined) cases.

**`tma_get_copy_fn`**: Returns a TMA copy callable with pre-partitioned tensors. Supports
zero-filtering for sparse layouts.

**`tma_producer_copy_fn`**: Wraps a TMA copy function with pipeline producer state management,
automatically acquiring barriers and passing the correct stage index.

---

## 8. Named Barrier

**File**: `flash_attn/cute/named_barrier.py`

Enumerations of named barrier IDs used for inter-warp and inter-warp-group synchronization
within kernels. Named barriers are identified by integer IDs starting from 1 (barrier 0 is
reserved for `sync_threads()`).

### `NamedBarrierFwd` (SM80/SM90 Forward)
- `Epilogue` (1) - Epilogue synchronization
- `WarpSchedulerWG1/2/3` (2-4) - Warp scheduler barriers for 3 warp groups
- `PFull` (5) - P buffer full signal
- `PEmpty` (6) - P buffer empty signal

### `NamedBarrierFwdSm100` (SM100 Forward)
- `Epilogue` (1) - Epilogue sync
- `TmemPtr` (2) - TMEM pointer management
- `SoftmaxStatsW0-W7` (3-10) - Per-warp softmax statistics barriers (8 warps)

### `NamedBarrierBwd` (SM80/SM90 Backward)
- `Epilogue` (1)
- `WarpSchedulerWG1/2/3` (2-4)
- `PdS` (5) - P to dS synchronization
- `dQFullWG0/1/2` (6-8) - dQ buffer full for each warp group
- `dQEmptyWG0/1/2` (9-11) - dQ buffer empty for each warp group

### `NamedBarrierBwdSm100` (SM100 Backward)
- `EpilogueWG1/WG2` (1-2) - Per-warp-group epilogue
- `Compute` (3) - Compute synchronization
- `dQaccReduce` (4) - dQ accumulation reduction
- `TmemPtr` (5) - TMEM pointer management

### `NamedBarrierFwdSm100_MLA2CTA` (SM100 MLA 2CTA Forward)
- `Epilogue`, `TmemPtr`, `Cpasync`, `Softmax`
- `SoftmaxStatsFull`, `SoftmaxStatsEmpty`

---

## 9. Pack GQA

**File**: `flash_attn/cute/pack_gqa.py`

Pack GQA (Grouped Query Attention packing) optimizes GQA/MQA by folding multiple Q heads into
the sequence dimension, enabling a single MMA tile to process multiple Q heads simultaneously.

### Layout Functions

**`pack_gqa_layout`**
```python
def pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx)
```
Reshapes a tensor to fold `qhead_per_kvhead` into the seqlen dimension:
- Q/O tensors (head_idx=2): `(seqlen, headdim, nheads, batch)` -> `((qhead_per_kvhead, seqlen), headdim, nheads_kv, batch)`
- LSE tensors (head_idx=1): `(seqlen, nheads, batch)` -> `((qhead_per_kvhead, seqlen), nheads_kv, batch)`

**`unpack_gqa_layout`**
Reverses the pack operation, restoring the original tensor shape.

**`make_packgqa_tiled_tma_atom`**
Creates a TMA copy atom that maintains the same TMA dimension count as the non-packed case.
Packs the head and seqlen dimensions into one TMA dimension to avoid 5D TMA descriptors.

### Class: `PackGQA`

Dataclass with PackGQA parameters and load/store methods.

**`compute_ptr`**
Computes per-thread global memory pointers for PackGQA loads. Each row in the packed layout
maps to `(m_idx, h_idx)` in the unpacked layout: `m_idx = row // qhead_per_kvhead`,
`h_idx = row % qhead_per_kvhead`. Pointers are computed per-thread and shared via
`shuffle_sync`.

**`load_Q`**
Loads a Q tile from global memory to shared memory. Handles:
- Per-thread pointer computation for the packed layout
- Warp-shuffle to distribute pointers across threads
- Boundary checking against `seqlen * qhead_per_kvhead`
- Optional head-dimension bounds checking

**`store_LSE`**
Stores log-sum-exp values from registers to global memory. Only the thread at column 0
in the MMA accumulator writes LSE values.

**`store_O`**
Stores output values from registers to global memory. Similar pointer computation and
warp-shuffle logic as `load_Q`.

---

## 10. Paged KV

**File**: `flash_attn/cute/paged_kv.py`

### Class: `PagedKVManager`

Manages paged KV cache access for inference serving, where KV pairs are stored in a
non-contiguous page table structure.

**Constructor Fields**:
- `mPageTable`, `mK_paged`, `mV_paged` - Paged KV cache tensors
- `page_size_divmod` - Fast divmod for page size
- `seqlen_k`, `leftpad_k` - Sequence length and left padding
- `n_block_size`, `head_dim_padded`, `head_dim_v_padded` - Tile dimensions
- `arch` - Target architecture
- `v_gmem_transposed` - Whether V is transposed in global memory (SM100 vs SM90)
- `gmem_tiled_copy_KV`, `gmem_thr_copy_KV` - Copy configurations
- `tPrPage`, `tPrPageOffset` - Per-thread page table entries
- `tKpK`, `tVpV` - Per-thread predicates for K/V bounds

**`create`**: Constructs a PagedKVManager with architecture-specific configurations:
- SM100: V is transposed to `(dv, page_size, num_pages)` for efficient UMMA access
- SM90: V layout matches K `(page_size, dv, num_pages)`
- Uses cp.async with 128-bit copy atoms and universal cache mode

**`load_page_table`**: For each row in the n_block, computes `(page_idx, page_offset)` from
the row index and page size, then loads the page table entry. Invalid rows use page 0.

**`compute_X_ptr`**: Computes per-thread global memory pointers into the paged K or V tensor.
Handles the transposed V layout for SM100.

**`load_KV`**: Main load function that:
1. Computes per-thread page pointers
2. Flattens shared memory layout (SM100 also transposes V in smem)
3. Creates identity tensors for bounds checking
4. Iterates over rows, using warp-shuffle to distribute page pointers
5. Issues async copies with valid-row predicates

---

## 11. Fast Math

**File**: `flash_attn/cute/fast_math.py`

### `clz`
```python
@cute.jit
def clz(x: Int32) -> Int32
```
Count leading zeros in a 32-bit integer. Used by tile schedulers for computing log2 of
L2 swizzle factors. Implements via iteration with early exit.

---

## 12. Block Sparse Utils

**File**: `flash_attn/cute/block_sparse_utils.py`

Runtime utilities for block-sparse attention kernels. These functions execute on the GPU
inside CUTE DSL kernels.

### Data Structures

**`BlockSparseTensors`**: NamedTuple containing CuTe tensors for block sparse indices:
- `mask_block_cnt` - Count of partial (masked) blocks per row
- `mask_block_idx` - Column indices of partial blocks
- `full_block_cnt` - Count of full (unmasked) blocks per row (optional)
- `full_block_idx` - Column indices of full blocks (optional)
- `dq_write_order` - Lock values for deterministic dQ writes (optional)

### Producer-Side Functions

**`load_block_list`**: Iterates over sparse block indices and issues K/V loads into the
pipeline. Supports intra-warp-group overlap where K and V loads are pipelined.

**`finish_overlap_v_load`**: Drains the final V load after overlapped K/V processing.

**`sparse_tensor_m_block`**: Maps packed m_block indices to block-sparse tensor indices,
accounting for PackGQA and subtiling factors.

**`produce_block_sparse_loads`**: Top-level producer function that handles both masked and
full block lists. Manages the overlap between the masked list's trailing V and the full
list's leading K load.

### Consumer-Side Functions

**`consume_block_sparse_loads`**: Consumer counterpart that processes loaded blocks through
MMA. Separates masked blocks (with mask_mod) from full blocks (without mask_mod) for
efficiency. Handles warp scheduler barrier synchronization.

### SM100 Functions

**`load_block_list_sm100`**: SM100-specific block list loader that also handles Q loading
alongside the first KV block.

**`produce_block_sparse_loads_sm100`**: SM100 producer with simplified pipeline management.

**`get_total_block_count`**: Returns total masked + full block count for a tile.

**`handle_block_sparse_empty_tile_correction_sm100`**: Handles the case where a tile has
zero active KV blocks. Seeds fully-masked row stats and runs correction epilogue with
scale=0 to write zeros. Manages barrier phase alignment across empty tiles.

**`softmax_block_sparse_sm100`**: SM100 softmax loop that iterates over sparse block indices,
applying mask_mod only to partial blocks.

### Backward Functions

**`get_total_q_block_count_bwd`**: Counts total Q-tile iterations for a given KV tile in backward.

**`produce_block_sparse_q_loads_bwd_sm100` / `produce_block_sparse_q_loads_bwd_sm90`**:
Backward producer functions that load Q, dO, LSE, dPsum for each sparse m_block, with
K/V loaded only on the first iteration.

**`consume_block_sparse_mma_bwd_sm90`**: Backward consumer that processes sparse m_blocks
through MMA, separating partial (with mask_mod) and full blocks.

**`dQaccum_store_block_sparse_bwd_sm90`**: Stores dQaccum to global memory for sparse
m_blocks, using TMA bulk reduction.

---

## 13. Block Sparsity

**File**: `flash_attn/cute/block_sparsity.py`

Host-side utilities for creating, validating, and normalizing block-sparse tensor metadata.

### Data Structures

**`BlockSparseTensorsTorch`**: PyTorch-based named tuple with:
- `mask_block_cnt/idx` - Partial block metadata
- `full_block_cnt/idx` - Full block metadata
- `block_size` - Sparse block size `(q_block_size, kv_block_size)`
- `dq_write_order/full` - Backward write ordering metadata
- `spt` - Shortest-processing-time-first flag

### Shape Inference

**`infer_block_sparse_expected_shapes`**: Determines expected shapes for block sparse tensors
based on problem dimensions and tile sizes. Validates:
- `sparse_block_size_kv == tile_n`
- `sparse_block_size_q` is a multiple of `q_stage * tile_m`
- Batch/head dimensions match or are broadcastable (size 1)

**`get_block_sparse_expected_shapes_bwd`**: Same for backward pass, using transposed indexing
(Q-direction).

### Normalization

**`normalize_block_sparse_tensors`**: Validates and expands block sparse tensors:
- Checks dtype (int32), device (CUDA), and shape compatibility
- Expands broadcast dimensions (size 1 dims)
- Allows compact index tensors (last dim <= expected) for memory efficiency

**`normalize_block_sparse_config`** / **`normalize_block_sparse_config_bwd`**: Complete
normalization pipeline for forward and backward, returning normalized tensors,
broadcast patterns, and subtile factors.

### dQ Write Order

**`compute_dq_write_order`**: Computes semaphore lock values for deterministic backward dQ
writes. For each (n_block, m_block) pair, computes the rank of n_block in the sorted
contributor list. Lock values are assigned in ascending or descending order based on the
CTA scheduling direction to guarantee deadlock freedom.

### Conversion

**`to_cute_block_sparse_tensors`**: Converts PyTorch tensors to CuTe tensors for kernel use,
optionally enabling TVM FFI.

**`fast_sampling`**: Decorator that marks a mask_mod as safe for 5-point fast sampling optimization.
