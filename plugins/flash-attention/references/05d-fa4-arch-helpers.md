# FlashAttention-4 Architecture Helpers Reference

This document provides an exhaustive reference for the architecture-specific helper modules
in FlashAttention-4 (FA4). These modules contain GEMM operations, hardware descriptor construction,
configuration search, utility functions, and compilation infrastructure.

## Table of Contents

1. [Blackwell Helpers (`blackwell_helpers.py`)](#1-blackwell-helpers)
2. [Ampere Helpers (`ampere_helpers.py`)](#2-ampere-helpers)
3. [MMA SM100 Descriptor (`mma_sm100_desc.py`)](#3-mma-sm100-descriptor)
4. [SM90 Config Search (`sm90_config_search.py`)](#4-sm90-config-search)
5. [Utilities (`utils.py`)](#5-utilities)
6. [Cache Utilities (`cache_utils.py`)](#6-cache-utilities)
7. [CuTe DSL Utilities (`cute_dsl_utils.py`)](#7-cute-dsl-utilities)
8. [CuTe DSL PTXAS (`cute_dsl_ptxas.py`)](#8-cute-dsl-ptxas)

---

## 1. Blackwell Helpers

**File**: `flash_attn/cute/blackwell_helpers.py`

Architecture-specific GEMM helper functions for SM100 (Blackwell). Provides PTX-level
control over the UMMA (Unified Matrix Multiply-Accumulate) instructions.

### MMA Kind Detection

**`_tcgen05_mma_kind`**
```python
def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str
```
Maps a tcgen05 MMA operation to its PTX instruction kind string:
- `MmaF16BF16Op` -> `"f16"`
- `MmaTF32Op` -> `"tf32"`
- `MmaI8Op` -> `"i8"`
- `MmaFP8Op` -> `"f8f6f4"`
- `MmaMXF8Op` -> `"mxf8f6f4"`
- `MmaMXF4Op` -> `"mxf4"`
- `MmaMXF4NVF4Op` -> `"mxf4nvf4"`

### GEMM Functions

**`gemm_w_idx`**
```python
@cute.jit
def gemm_w_idx(tiled_mma, acc, tCrA, tCrB, A_idx=None, B_idx=None,
               zero_init=False, swap_AB=False, num_unroll_groups=1)
```
UMMA GEMM with pipeline stage indexing. Supports:
- Stage-indexed A and/or B operands via `A_idx` / `B_idx`
- `zero_init` flag to zero the accumulator on the first K iteration
- `swap_AB` to transpose the operation
- `num_unroll_groups` for controlling the K-dimension unroll factor

Uses `mma_atom.set(tcgen05.Field.ACCUMULATE, ...)` to control whether the MMA accumulates
or overwrites, based on `zero_init` and iteration index.

**`gemm`**
```python
@cute.jit
def gemm(tiled_mma, acc, tCrA, tCrB, zero_init=False)
```
Simplified GEMM without stage indexing. Iterates over the K dimension using `range_constexpr`
for full unrolling.

**`gemm_ptx`**
```python
@cute.jit
def gemm_ptx(op, acc, tCrA, tCrB, sA, sB, zero_init=False)
```
Low-level PTX-based GEMM that directly emits `tcgen05.mma` instructions. Handles:
1. Determines operand source (TMEM vs shared memory) from `op.a_src`
2. Computes the 32-bit instruction descriptor via `mma_sm100_desc.mma_op_to_idesc`
3. Builds 64-bit shared memory descriptors for A and B operands
4. Emits PTX inline assembly with the correct `tcgen05.mma.cta_group::1.kind::*` instruction
5. Two paths: TMEM-source (A from TMEM, B from smem) and smem-source (both from smem)

**`gemm_ptx_loop`**
Similar to `gemm_ptx` but optimized for loop-based K iteration with precomputed offsets
and offset differences between consecutive K tiles. This avoids redundant address computation.

**`gemm_ptx_w_idx`**
Combines PTX-level GEMM with stage indexing. Constructs the appropriate smem descriptors
and delegates to `gemm_ptx_partial`.

**`gemm_ptx_partial`**
Core PTX emission function supporting:
- TMEM-source and smem-source operand modes
- `cta_group` parameter for 2CTA operations
- Precomputed smem descriptor bases and offsets

### Address Manipulation

**`i64_to_i32x2`**
```python
def i64_to_i32x2(i: int) -> Tuple[int, int]
```
Splits a 64-bit integer into two 32-bit halves for PTX register assignment.

---

## 2. Ampere Helpers

**File**: `flash_attn/cute/ampere_helpers.py`

Architecture-specific GEMM helper functions for SM80 (Ampere) and SM120.

### SMEM Layout

**`get_smem_layout_atom`**
```python
def get_smem_layout_atom(dtype, k_dim) -> cute.ComposedLayout
```
Creates a swizzled shared memory layout for MMA operands. Computes:
1. `smem_k_block_size`: The K-dimension block size, chosen as the largest power of 2
   (128/64/32/16 elements) that divides the bytes per row
2. `swizzle_bits`: XOR swizzle width (4/3/2/1) based on block size
3. `swizzle_base`: Swizzle base offset based on dtype byte width
4. Row count: 8 or 16 rows depending on whether `k_dim % 32 == 0`

Returns a composed layout with swizzle applied to an ordered (row-major K) layout.

### GEMM Functions

**`gemm`**
```python
@cute.jit
def gemm(tiled_mma, acc, tCrA, tCrB, tCsA, tCsB,
         smem_thr_copy_A, smem_thr_copy_B, hook_fn=None,
         A_in_regs=False, B_in_regs=False, swap_AB=False)
```
Full GEMM operation with smem-to-register copy and MMA. Pipeline:
1. If `swap_AB`, recursively calls with A/B swapped
2. Load first K tile of A and B from shared memory to registers
3. For each K tile:
   a. Prefetch next K tile (if not last)
   b. Issue MMA instruction for current tile
   c. Call `hook_fn` on first iteration (for overlapping with other work)
4. Supports `A_in_regs` / `B_in_regs` to skip smem loads when data is already in registers

**`gemm_rs`**
```python
@cute.jit
def gemm_rs(tiled_mma, acc, tCrA, tCrB, tCsB, smem_thr_copy_B, hook_fn=None)
```
Register-source GEMM where operand A is already in registers (e.g., from TMEM load).
Only loads B from shared memory. Pipeline:
1. Load first B tile
2. For each K tile: prefetch next B, issue MMA, call hook on first iteration

---

## 3. MMA SM100 Descriptor

**File**: `flash_attn/cute/mma_sm100_desc.py`

Low-level hardware descriptor construction for Blackwell UMMA instructions.
Ported from CUTLASS C++ (`include/cute/arch/mma_sm100_desc.hpp`).

### Enumerations

**`Major`**: Matrix layout encoding
- `K = 0` - K-major (column-major for A, row-major for B)
- `MN = 1` - MN-major (row-major for A, column-major for B)

**`ScaleIn`**: Input negation flags
- `One = 0` - No negation
- `Neg = 1` - Negate input

**`Saturate`**: Accumulator saturation
- `False_ = 0` - No saturation
- `True_ = 1` - Saturate output

**`CFormat`**: Accumulator format (2-bit field)
- `F16 = 0`, `F32 = 1`, `S32 = 2`

**`F16F32Format`**: Input element type (3-bit field)
- `F16 = 0`, `BF16 = 1`, `TF32 = 2`

**`S8Format`**: Integer 8-bit format
- `UINT8 = 0`, `INT8 = 1`

**`MXF8F6F4Format`**: Microscaling formats
- `E4M3 = 0`, `E5M2 = 1`, `E2M3 = 3`, `E3M2 = 4`, `E2M1 = 5`

**`MaxShift`**: Maximum shift for scaled integer MMA
- `NoShift = 0`, `MaxShift8/16/32`

**`LayoutType`**: Swizzle pattern encoding for smem descriptors
- `SWIZZLE_NONE = 0`, `SWIZZLE_128B_BASE32B = 1`, `SWIZZLE_128B = 2`,
  `SWIZZLE_64B = 4`, `SWIZZLE_32B = 6`

### Descriptor Construction

**`to_UMMA_format`**
Maps CUTLASS scalar types to 3-bit UMMA encoding for A/B operands.

**`to_C_format`**
Maps CUTLASS scalar types to 2-bit accumulator encoding.

**`make_instr_desc`**
```python
def make_instr_desc(a_type, b_type, c_type, M, N, a_major, b_major,
                    a_neg=ScaleIn.One, b_neg=ScaleIn.One,
                    c_sat=Saturate.False_, is_sparse=False,
                    max_shift=MaxShift.NoShift) -> int
```
Constructs the 32-bit UMMA instruction descriptor by packing bit fields:
- Bits [0:1] - sparse_id2
- Bit [2] - sparse_flag
- Bit [3] - saturate
- Bits [4:5] - c_format
- Bits [7:9] - a_format
- Bits [10:12] - b_format
- Bit [13] - a_negate
- Bit [14] - b_negate
- Bit [15] - a_major
- Bit [16] - b_major
- Bits [17:22] - n_dim (N >> 3)
- Bits [24:28] - m_dim (M >> 4)
- Bits [30:31] - max_shift

M must be 64/128/256, N must be a multiple of 8 in [8, 256].

**`mma_op_to_idesc`**
Converts a `cute.nvgpu.tcgen05.mma.MmaOp` to its instruction descriptor by extracting
the operand types, shapes, and major modes.

### Shared Memory Descriptor Construction

**`_layout_type`**
Determines the swizzle layout type from a `cute.Swizzle` object. Maps swizzle parameters
(num_bits, num_base, num_shift) to `LayoutType` values.

**`make_smem_desc_base`**
Constructs the upper 32 bits of a 64-bit shared memory descriptor, encoding the layout type,
stride, and leading dimension.

**`make_smem_desc_start_addr`**
Constructs the lower 32 bits from a shared memory pointer, encoding the base address and
offset.

---

## 4. SM90 Config Search

**File**: `flash_attn/cute/sm90_config_search.py`

Configuration space search tool for SM90 (Hopper) forward and backward kernels.
Enumerates feasible tile sizes, swap modes, atom layouts, and staging options.

### Hardware Limits

- `SMEM_LIMIT = 224 KB` - Maximum shared memory per CTA
- `REG_LIMITS = {2: 216, 3: 128}` - Per-WG register budgets (240-24 or 160-32)
- `THREADS_PER_WG = 128` - Threads per warp group

### Feasibility Checking

**`_check_mma`**
```python
def _check_mma(M, N, num_wg, atom_layout_m, swap_AB)
```
Checks WGMMA feasibility: M must be divisible by `atom_layout_m * 64`, N by `atom_layout_n * 8`.
Returns register count per WG, or None if infeasible.

**`_mma_traffic`**
```python
def _mma_traffic(M_eff, N_eff, K_red, num_wg, wg_n, is_rs=False)
```
Computes total shared memory read traffic for one MMA operation across all warp groups.

### Backward Configuration Search

**`_check_bwd_config`**
Checks a complete backward configuration with 4 MMA operations (S, dP, dK/dV, dQ):
1. Verifies MMA feasibility for all 4 operations
2. Checks register budget: `max(2*regs_SdP, regs_dQ) + regs_dK + regs_dV <= reg_limit`
3. Checks shared memory budget for Q, K, V, dO, P, dS, dQaccum tiles
4. Computes total SMEM traffic

**`find_feasible_bwd_configs`**
Enumerates all feasible backward configurations for given (head_dim, head_dim_v) over:
- `tile_m`, `tile_n` choices (64-128)
- `num_wg` (2 or 3)
- `SdP_swapAB`, `dKV_swapAB`, `dQ_swapAB`
- `AtomLayoutMSdP`, `AtomLayoutNdKV`, `AtomLayoutMdQ`

Returns a list of feasible configurations sorted by SMEM traffic (lower is better).

### Forward Configuration Search

**`find_feasible_fwd_configs`**
Enumerates forward configurations for given (head_dim, head_dim_v) with:
- WGMMA register-source optimization (A in registers when atom layout allows)
- Q staging options (1 or 2 stages)
- K/V staging options

---

## 5. Utilities

**File**: `flash_attn/cute/utils.py`

### Exp2 Polynomial Coefficients

**`POLY_EX2`**: Dictionary mapping polynomial degree (0-5) to coefficient tuples for
exp2 approximation via Sollya-generated minimax polynomials. Used for fast exp2 on GPUs.

### Environment Variables

- `_fa_clc_enabled` - `FA_CLC=1` enables CLC scheduling
- `_fa_disable_2cta_enabled` - `FA_DISABLE_2CTA=1` disables 2CTA forward
- `_fa_disable_2cta_cuda12` - Auto-disables 2CTA on CUDA 12.x due to codegen regression

### Hash Functions

**`_compute_base_hash`**
Hashes a callable based on its source code (or bytecode) and closure values. Used for
compile cache invalidation.

**`hash_callable`**
```python
def hash_callable(func, mixer_attrs=("__vec_size__",), set_cute_hash=True) -> str
```
Computes a stable hash for a callable, incorporating:
- Base hash from source/bytecode
- Mutable metadata dunders (e.g., `__vec_size__` for vectorization)
- Cached via `__cute_hash__` attribute

### Softcap Score Mods

**`create_softcap_scoremod`**
Creates a score modification function that applies softcap: `score = softcap * tanh(score / softcap)`.
Uses fastmath tanh.

**`create_softcap_scoremod_bwd`**
Creates the backward pass: `grad = grad_out * (1 - tanh^2(score / softcap))`.

### Softmax Scale Computation

**`compute_softmax_scale_log2`**
```python
def compute_softmax_scale_log2(softmax_scale, score_mod)
```
When `score_mod is None`, folds `log2(e)` into the scale for direct exp2 usage:
`(softmax_scale_log2, None)`. When score_mod is present, keeps them separate:
`(LOG2E, softmax_scale)`.

### FastDivmod

**`compute_fastdiv_mods`**
Computes `(seqlen_q_divmod, seqlen_k_divmod)` for FlexAttention aux tensor index wrapping.
Returns None when no aux tensors are present.

### Hardware Query

**`get_max_active_clusters`**: Cached query for maximum concurrent clusters.

**`get_device_capacity`**: Returns (major, minor) compute capability.

### PTX-level Operations

**`fmax_reduce`**: Architecture-aware max reduction. For SM90+, uses `warp_reduction_max`
with 4-wide warp group reduction. For SM80, uses standard reduction.

**`fadd_reduce`**: Architecture-aware sum reduction with similar dispatch.

**`warp_reduce`**: Generic warp-level reduction using `shfl_sync_down`.

**`warp_prefix_sum`**: Warp-level parallel prefix sum using Kogge-Stone algorithm.

### Register-level Helpers

**`scalar_to_ssa`** / **`ssa_to_scalar`**: Convert between scalar values and SSA form
for JIT compatibility in score_mod calls.

**`predicate_k`**: Creates per-thread predicates for the K (head) dimension to handle
cases where `head_dim_padded > actual_head_dim`.

**`elem_pointer`**: Extracts the element pointer from a tensor at given coordinates.

**`shuffle_sync`**: Wrapper for `cute.arch.shuffle_sync` with width parameter.

### Exp2 Emulation

**`ex2_emulation_2`**: Software exp2 emulation for mixed-precision accuracy, computing
two values simultaneously.

---

## 6. Cache Utilities

**File**: `flash_attn/cute/cache_utils.py`

Manages JIT compilation caching with both in-memory and persistent disk storage.

### Cache Configuration

- `CUTE_DSL_CACHE_ENABLED` - Enable/disable via `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`
- `CUTE_DSL_CACHE_DIR` - Custom cache directory (default: `/tmp/$USER/flash_attention_cute_dsl_cache`)

### Source Fingerprinting

**`_compute_source_fingerprint`**
```python
@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str
```
Computes a SHA-256 hash of:
- Python version (major.minor)
- CUTLASS version
- TVM FFI version
- All `.py` files under `flash_attn/cute/`

Changes whenever source code, Python version, or dependency versions change.

### File Locking

**`FileLock`**
Context manager for advisory file locks using `fcntl.flock`. Supports:
- Exclusive (write) and shared (read) locks
- Configurable timeout with polling
- Automatic cleanup on exit

### Cache Classes

**`JITCache`**
Simple in-memory dictionary cache for compiled functions.

**`JITPersistentCache`**
Extends `JITCache` with disk persistence:
- `__setitem__`: Stores in memory and exports to disk via `_try_export_to_storage`
- `__contains__`: Checks memory first, then loads from disk if available
- `__getitem__`: Loads from disk on cache miss
- Uses file locking for thread-safe disk access

The disk cache stores compiled kernels as shared libraries (`.so` files) alongside metadata
files containing the compile key and source fingerprint. Cache hits require matching both
the compile key and the source fingerprint.

### Runtime Library Pre-loading

On import, pre-loads CuTe DSL runtime libraries with `RTLD_GLOBAL` flag to ensure their
symbols are visible to cached `.so` modules loaded later.

---

## 7. CuTe DSL Utilities

**File**: `flash_attn/cute/cute_dsl_utils.py`

### Type Conversion

**`torch2cute_dtype_map`**: Maps PyTorch dtypes to CUTLASS types (float16, bfloat16, float32,
float8_e4m3fn, float8_e5m2).

### Tensor Conversion

**`to_cute_tensor`**
```python
def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True)
```
Converts a PyTorch tensor to a CuTe tensor:
1. Handles fp8 types via uint8 view (workaround for DLPack limitations)
2. Creates tensor via `from_dlpack` with TVM FFI
3. Marks layout dynamic on the leading dimension (or fully dynamic if specified)
4. Applies alignment assumptions

**`to_cute_aux_tensor`**
Converts FlexAttention aux tensors with custom alignment and leading dimension from
`__assumed_align__` and `__leading_dim__` attributes.

**`get_aux_tensor_metadata`**
Extracts alignment, leading dimension, and dynamic flag from aux tensors for cache keys.

### Stride Alignment

**`assume_strides_aligned`**
Assumes all strides except the last are divisible by 128 bits (16 bytes). Python integer
strides (e.g., stride=0 from broadcasting) are kept as-is.

**`assume_tensor_aligned`**
Rebuilds a tensor with aligned stride assumptions. Returns None if input is None.

### Broadcast Pattern

**`get_broadcast_dims`**
Returns a tuple of booleans indicating which dimensions have stride=0 (broadcasting).
Used in compile keys to ensure recompilation when broadcast patterns change.

### Kernel Introspection

**`dump_kernel_attributes`**
Uses CUDA driver API to query compiled kernel attributes:
- `local_size_bytes` - Local memory usage
- `num_regs` - Register count per thread

---

## 8. CuTe DSL PTXAS

**File**: `flash_attn/cute/cute_dsl_ptxas.py`

System `ptxas` replacement for CUTLASS DSL that allows using the system's CUDA toolkit
`ptxas` instead of the embedded one, which may be older or missing.

### Configuration

- `CUTE_DSL_PTXAS_PATH` - Path to system ptxas binary
- `CUTE_DSL_PTXAS_VERBOSE` - Enable verbose logging (default: "0")

### Operation

**`patch`**
```python
def patch()
```
Installs the system ptxas hook by monkey-patching `CudaDialectJitCompiledFunction._load_cuda_library`:

1. Verifies ptxas exists and is executable
2. Requires `CUTE_DSL_KEEP_PTX=1` to ensure PTX files are preserved
3. Replaces `_load_cuda_library` with `_patched_load_cuda_library`

**`_patched_load_cuda_library`**
The replacement method:
1. Finds the PTX file for the compiled function
2. Extracts the target architecture (e.g., `sm_90a`)
3. Compiles PTX to cubin using system `ptxas -O3`
4. Loads the cubin via `cudaLibraryLoadData`
5. Registers the kernel on all devices
6. Falls back to embedded ptxas on any failure

**`_get_ptx`**
Searches the dump directory for PTX files matching the function name, strips null bytes.

**`_compile_ptx`**
Runs `ptxas -arch=<arch> -O3` and returns the cubin bytes. Optionally saves cubin to disk
via `CUTE_DSL_KEEP_CUBIN=1`.
