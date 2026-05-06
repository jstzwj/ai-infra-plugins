# Built-ins and Intrinsics

This document provides comprehensive reference for all built-in operations and hardware intrinsics exposed by TileLang. These operations map directly to GPU hardware instructions and provide fine-grained control over memory access, synchronization, warp-level communication, and Tensor Core operations.

## Table of Contents

- [Memory Access](#memory-access)
- [Barrier Operations](#barrier-operations)
- [Warp Operations](#warp-operations)
- [Warp-Vote Operations](#warp-vote-operations)
- [Warp-Shuffle Operations](#warp-shuffle-operations)
- [Tensor Core Operations](#tensor-core-operations)
- [PTX Async Operations](#ptx-async-operations)
- [TMA Operations](#tma-operations)
- [Register Control](#register-control)
- [TMEM Operations](#tmem-operations)
- [LDG/STG Explicit Memory Operations](#ldgstg-explicit-memory-operations)
- [LDS Transpose Read Operations](#lds-transpose-read-operations)
- [Math Intrinsics](#math-intrinsics)
- [Syncthreads Variants](#syncthreads-variants)
- [Warp Match Intrinsics](#warp-match-intrinsics)
- [Memory Descriptors](#memory-descriptors)
- [Synchronization Primitives](#synchronization-primitives)
- [Debugging](#debugging)
- [Control Flow](#control-flow)
- [IEEE-Compliant Operations](#ieee-compliant-operations)
- [Packed x2 Element-wise Operations](#packed-x2-element-wise-operations)

---

## Memory Access

### T.__ldg

```python
T.__ldg(load_or_buf: BufferLoad | tir.Buffer, index: PrimExpr | int | None = None) -> PrimExpr
```

Explicitly loads data through the CUDA read-only data cache. On CUDA backends, this emits `__ldg(&x[i])`. On non-CUDA backends, falls back to a regular load.

**Usage patterns:**

```python
# Preferred: pass a BufferLoad expression
val = T.__ldg(A[i, j])  # emits __ldg(&A[i, j])

# Alternative: pass a Buffer with explicit index
val = T.__ldg(A, index=i * stride + j)
```

**When to use:**
- When the data is known to be read-only for the kernel's lifetime.
- To reduce pressure on the L1/L2 data cache by routing reads through the larger read-only cache.
- For data that is accessed once and not modified (streaming reads).

**Backend behavior:**

| Backend | Generated Code          |
|---------|-------------------------|
| CUDA    | `__ldg(&ptr)`           |
| HIP     | Regular load (fallback) |
| CPU     | Regular load (fallback) |

### T.access_ptr

```python
T.access_ptr(
    base: BufferLikeType,
    access_type: str | int = "r",
    *extents: PrimExpr | int | tuple | list,
    offset: PrimExpr | int = 0,
    extent: PrimExpr | int | None = None,
    ignore_last_ndim: int = 0,
) -> PrimExpr
```

Creates a TileLang `tl.access_ptr` from a buffer-like base location. This is a frontend convenience wrapper that preserves buffer metadata for downstream synchronization and safety checks.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `Buffer`, `BufferLoad`, `BufferRegion`, or `Var` | The base location |
| `access_type` | `str` or `int` | Access mask: `"r"` (1), `"w"` (2), `"rw"` (3) |
| `*extents` | `PrimExpr`, `int`, or tuples | Per-axis extents |
| `offset` | `PrimExpr` or `int` | Additional element offset |
| `extent` | `PrimExpr` or `int` | Explicit 1D extent override |
| `ignore_last_ndim` | `int` | Ignore trailing axes for offset computation |

**Examples:**

```python
# Element pointer (extent=1)
ptr = T.access_ptr(A[i], "r")

# Range pointer with explicit extent
ptr = T.access_ptr(A[i], "r", 16)       # extent=16

# Multi-dimensional extents
ptr = T.access_ptr(A[i, j], "r", m, n)  # extent=m*n

# Read-write access
ptr = T.access_ptr(A[i], "rw", 128)

# Buffer base pointer (full buffer)
ptr = T.access_ptr(A, "r")
```

The returned `tl.access_ptr` is lowered to `tir.builtin.tvm_access_ptr` during the `LowerAccessPtr` transform pass.

---

## Barrier Operations

### T.mbarrier_wait_parity

```python
T.mbarrier_wait_parity(mbarrier: BarrierType, parity: int | Var) -> tir.Call
```

Waits for a memory barrier until the specified parity condition is met. This is the fundamental synchronization primitive for pipeline-style kernels.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mbarrier` | `Buffer`, `BufferLoad` | The memory barrier to wait on |
| `parity` | `int` or `Var` | The parity value to wait for (0 or 1) |

**Pipelined kernel example:**

```python
mbar = T.alloc_barrier([2], "shared")  # [wait, arrive]

for ko in range(num_stages):
    # Producer waits for consumer to finish previous iteration
    T.mbarrier_wait_parity(mbar[1], ko ^ 1)
    # Producer copies data
    T.copy(A_global, A_shared)
    # Producer signals data ready
    T.mbarrier_arrive(mbar[0])

    # Consumer waits for producer data
    T.mbarrier_wait_parity(mbar[0], ko)
    # Consumer computes
    T.gemm(A_shared, B_shared, C_local)
    # Consumer signals completion
    T.mbarrier_arrive(mbar[1])
```

**How parity works:** mbarrier operations use a parity bit that toggles each time all expected threads have arrived. By waiting on alternating parities, the same barrier can be reused across pipeline stages without explicit reset.

### T.mbarrier_arrive

```python
T.mbarrier_arrive(mbarrier: BarrierType, cta_id: int | Var | None = None) -> tir.Call
```

Signals arrival at a memory barrier. When all expected threads (set by `mbarrier_expect_tx` or implicit count) have arrived, the barrier completes and waiting threads are released.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mbarrier` | `Buffer`, `BufferLoad` | The memory barrier to arrive at |
| `cta_id` | `int`, `Var`, or `None` | Peer CTA rank for cluster barriers. If `None`, arrives on current CTA's barrier. |

**Cluster barrier example:**

```python
cluster_mbar = T.alloc_barrier([1], "shared.cluster_barrier")
T.mbarrier_arrive(cluster_mbar, cta_id=peer_cta_rank)
```

### T.mbarrier_expect_tx

```python
T.mbarrier_expect_tx(mbarrier: BarrierType, tx: int) -> tir.Call
```

Sets the expected transaction count for a memory barrier. Used with async copy operations (TMA, cp.async) where the barrier tracks byte-level transaction completion rather than thread arrivals.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mbarrier` | `Buffer`, `BufferLoad` | The memory barrier |
| `tx` | `int` | Expected transaction count in bytes |

**Example with TMA:**

```python
mbar = T.alloc_barrier([1], "shared")
T.mbarrier_expect_tx(mbar[0], 128 * 128 * 2)  # expect 32KB of TMA data
T.copy(A_global, A_shared)  # TMA copy
T.mbarrier_wait_parity(mbar[0], 0)
```

### T.mbarrier_arrive_expect_tx

```python
T.mbarrier_arrive_expect_tx(mbarrier: BarrierType, tx: int) -> tir.Call
```

Combined arrive-and-expect-tx operation. Atomically arrives at the barrier and sets the expected transaction count. This is more efficient than separate `mbarrier_expect_tx` + `mbarrier_arrive` calls.

### T.barrier_wait

```python
T.barrier_wait(mbarrier: BarrierType, parity: int | Var) -> tir.Call
```

Sugar syntax for `mbarrier_wait_parity`. Identical behavior.

### T.barrier_arrive

```python
T.barrier_arrive(mbarrier: BarrierType) -> tir.Call
```

Sugar syntax for `mbarrier_arrive`. Arrives at a memory barrier without specifying a CTA ID.

---

## Warp Operations

### T.get_lane_idx

```python
T.get_lane_idx(warp_size: int | PrimExpr | None = None) -> PrimExpr
```

Returns the logical lane index of the calling thread within its warp.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warp_size` | `int`, `PrimExpr`, or `None` | `None` | Logical warp/wavefront size. Defaults to 32 on NVIDIA, 64 on AMD. |

**Examples:**

```python
lane = T.get_lane_idx()           # Default warp size
lane = T.get_lane_idx(64)         # Override for 64-lane wavefront
```

**Backend mapping:**

| Backend | Generated Code |
|---------|---------------|
| CUDA | `tl::get_lane_idx(warp_size)` |
| HIP | `tl::get_lane_idx(64)` |

### T.get_warp_idx

```python
T.get_warp_idx(warp_size: int | PrimExpr | None = None) -> PrimExpr
```

Returns the canonical warp index without requiring warp convergence.

```python
warp = T.get_warp_idx()       # Default warp size
warp = T.get_warp_idx(64)     # Custom warp size
```

### T.get_warp_idx_sync

```python
T.get_warp_idx_sync(warp_size: int | PrimExpr | None = None) -> PrimExpr
```

Returns the canonical warp index, assuming the warp's threads are converged. Issues an implicit warp synchronization barrier.

### T.get_warp_group_idx

```python
T.get_warp_group_idx(
    warp_size: int | PrimExpr | None = None,
    warps_per_group: int | PrimExpr | None = None,
) -> PrimExpr
```

Returns the canonical warp group index for the calling thread. A warp group consists of multiple warps that operate together (typically 4 warps = 128 threads on NVIDIA).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warp_size` | `int`, `PrimExpr`, or `None` | `None` | Logical warp size (32 on NVIDIA, 64 on AMD) |
| `warps_per_group` | `int`, `PrimExpr`, or `None` | `None` | Number of warps per group (default: 4) |

```python
group = T.get_warp_group_idx()          # Default: 4 warps of 32 threads
group = T.get_warp_group_idx(32, 6)     # 6 warps per group
```

### T.shuffle_elect

```python
T.shuffle_elect(thread_extent: int) -> PrimExpr
```

Elects exactly one lane within a logical thread group. Returns `True` for the elected lane and `False` for all others.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `thread_extent` | `int` | Size of the group. `0` elects a single lane in the entire thread block. |

**Example:**

```python
is_leader = T.shuffle_elect(64)  # Elect one leader per 64-thread group
if is_leader:
    # Only the leader executes this code
    T.barrier_arrive(mbar)
```

**Backend mapping:** Uses `cute::elect_one_sync()` or `__shfl_sync` to select one lane per group deterministically.

---

## Warp-Vote Operations

### T.any_sync

```python
T.any_sync(predicate: int | PrimExpr, mask: int | PrimExpr = 0xFFFFFFFF) -> PrimExpr
```

Returns non-zero if **any** active lane in `mask` has a non-zero `predicate`.

**CUDA:** `__any_sync(mask, predicate)`
**HIP:** `__any(predicate)` (mask ignored)

```python
has_any = T.any_sync(cond)                  # Any thread in warp has cond=True
has_any = T.any_sync(cond, mask=0x0000FFFF) # Only check first 16 lanes
```

### T.all_sync

```python
T.all_sync(predicate: int | PrimExpr, mask: int | PrimExpr = 0xFFFFFFFF) -> PrimExpr
```

Returns non-zero only if **all** active lanes in `mask` have a non-zero `predicate`.

```python
all_true = T.all_sync(cond)  # All threads in warp have cond=True
```

### T.ballot_sync

```python
T.ballot_sync(predicate: int | PrimExpr, mask: int | PrimExpr = 0xFFFFFFFF) -> PrimExpr
```

Returns a `uint64` bitmask where bit N is set if lane N's predicate is non-zero.

**CUDA:** Returns `unsigned int` (zero-extended to `uint64`).
**HIP:** Returns `uint64` natively (covering all 64 wavefront lanes).

```python
mask = T.ballot_sync(threadIdx.x < N)  # Bitmask of valid threads
popcount = T.popcount(mask)             # Count of valid threads
```

### T.ballot

```python
T.ballot(predicate: int | PrimExpr) -> PrimExpr
```

Full-warp/full-wavefront ballot. Equivalent to `ballot_sync(predicate)` with the default full mask.

### T.activemask

```python
T.activemask() -> PrimExpr
```

Returns a `uint64` bitmask of currently active (non-exited) lanes.

**CUDA:** `__activemask()` (zero-extended to `uint64`)
**HIP:** `__ballot(1)`

```python
active = T.activemask()  # Which threads are currently active
```

---

## Warp-Shuffle Operations

All shuffle operations support both CUDA and HIP backends. On HIP, the mask parameter is ignored.

### T.shfl_xor_sync / T.shfl_xor

```python
T.shfl_xor(value, delta, width=32, mask=0xFFFFFFFF) -> PrimExpr
```

XOR-swap `value` across lanes. Lane `i` exchanges data with lane `i ^ delta`.

```python
# Tree reduction using XOR shuffle
partial_sum = my_value
partial_sum += T.shfl_xor(partial_sum, 16)  # Sum with lane 16 away
partial_sum += T.shfl_xor(partial_sum, 8)   # Sum with lane 8 away
partial_sum += T.shfl_xor(partial_sum, 4)
partial_sum += T.shfl_xor(partial_sum, 2)
partial_sum += T.shfl_xor(partial_sum, 1)
# partial_sum now contains the warp-wide sum in every lane
```

### T.shfl_down_sync / T.shfl_down

```python
T.shfl_down(value, delta, width=32, mask=0xFFFFFFFF) -> PrimExpr
```

Shift `value` down by `delta` lanes. Lane `i` receives the value from lane `i + delta`.

```python
# Sequential reduction using down shuffle
val = my_value
val += T.shfl_down(val, 1)
val += T.shfl_down(val, 2)
val += T.shfl_down(val, 4)
val += T.shfl_down(val, 8)
val += T.shfl_down(val, 16)
# Lane 0 has the warp-wide sum
```

### T.shfl_up_sync / T.shfl_up

```python
T.shfl_up(value, delta, width=32, mask=0xFFFFFFFF) -> PrimExpr
```

Shift `value` up by `delta` lanes. Lane `i` receives the value from lane `i - delta`.

```python
# Prefix sum using up shuffle
val = my_value
val += T.shfl_up(val, 1)   # Add value from lane above
val += T.shfl_up(val, 2)
val += T.shfl_up(val, 4)
val += T.shfl_up(val, 8)
val += T.shfl_up(val, 16)
```

### T.shfl_sync

```python
T.shfl_sync(value, srcLane, width=32, mask=0xFFFFFFFF) -> PrimExpr
```

Broadcast `value` from `srcLane` to all lanes in the subgroup.

```python
# Broadcast the value from lane 0 to all lanes
broadcast_val = T.shfl_sync(my_value, 0)
```

---

## Tensor Core Operations

### T.warpgroup_arrive

```python
T.warpgroup_arrive() -> tir.Call
```

Signals warpgroup readiness for subsequent WGMMA operations. This must be called before issuing WGMMA instructions on Hopper (SM90+) GPUs.

```python
T.warpgroup_arrive()
# Now safe to issue WGMMA
```

### T.warpgroup_commit_batch

```python
T.warpgroup_commit_batch() -> tir.Call
```

Commits the current warpgroup batch of WGMMA operations. After calling this, the WGMMA operation begins execution asynchronously.

```python
# Issue WGMMA
T.gemm(A_shared, B_shared, C_frag)
# Commit the batch
T.warpgroup_commit_batch()
```

### T.warpgroup_wait

```python
T.warpgroup_wait(num_mma: int) -> tir.Call
```

Waits for completion of the specified number of WGMMA batches.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_mma` | `int` | Number of outstanding WGMMA batches to wait for (0 = wait for all) |

```python
T.warpgroup_commit_batch()
# ... overlap with other work ...
T.warpgroup_wait(0)  # Wait for all WGMMA to complete
```

### T.warpgroup_fence_operand

```python
T.warpgroup_fence_operand(
    buffer_or_ptr: BufferLikeType | PrimExpr,
    offset: int | PrimExpr = 0,
    num_regs: int | PrimExpr | None = None,
    dtype: DType | None = None,
)
```

Inserts a warpgroup fence for destination accumulator registers. This prevents NVCC from sinking uses of accumulator fragments past corresponding WGMMA operations by issuing an empty inline assembly barrier on every register.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `buffer_or_ptr` | `Buffer`, `BufferLoad`, `BufferRegion`, or `PrimExpr` | The accumulator buffer or pointer |
| `offset` | `int` or `PrimExpr` | Element offset from start (default: 0) |
| `num_regs` | `int`, `PrimExpr`, or `None` | Number of 32-bit registers to fence. Auto-derived from Buffer shape when possible. |
| `dtype` | `DType` or `None` | Data type of accumulator elements. Auto-inferred from buffer when possible. |

**Examples:**

```python
# Fence entire accumulator buffer
C_frag = T.alloc_fragment((128, 256), "float32")
T.warpgroup_fence_operand(C_frag)

# Fence a sub-region
T.warpgroup_fence_operand(C_frag[0:64, 0:128])

# Fence with explicit register count (for pointer expressions)
ptr = T.access_ptr(C_frag, "rw")
T.warpgroup_fence_operand(ptr, num_regs=128, dtype="float32")
```

### T.wait_wgmma

```python
T.wait_wgmma(id: int) -> tir.Call
```

Waits for a specific WGMMA operation to complete.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `int` | The identifier of the WGMMA operation to wait for |

```python
T.wait_wgmma(0)  # Wait for WGMMA operation 0
```

---

## PTX Async Operations

### T.cp_async_barrier_noinc

```python
T.cp_async_barrier_noinc(barrier: BarrierType) -> tir.Call
```

Performs a PTX async copy barrier arrival without incrementing the thread count. Maps to `cp.async.mbarrier.arrive.noinc`. Used when async copy operations (not threads) should trigger barrier completion.

```python
mbar = T.alloc_barrier([1], "shared")
T.mbarrier_expect_tx(mbar[0], bytes_expected)
# Issue async copies...
T.cp_async_barrier_noinc(mbar[0])
```

### T.fence_proxy_async

```python
T.fence_proxy_async() -> tir.Call
```

Issues a shared memory fence for asynchronous proxy operations. Ensures that prior asynchronous operations (e.g., TMA stores) are visible to subsequent memory accesses. Maps to `fence.proxy.async.shared::cta`.

```python
T.fence_proxy_async()
# Prior async writes to shared memory are now visible
```

---

## TMA Operations

### T.tma_store_arrive

```python
T.tma_store_arrive() -> tir.Call
```

Signals the arrival (commitment) of TMA store operations. Maps to `cp.async.bulk.commit_group`.

```python
# Issue TMA store
T.copy(shared_buf, global_buf)  # May use TMA for shared->global
T.tma_store_arrive()  # Commit the TMA store group
```

### T.tma_store_wait

```python
T.tma_store_wait(count: int = 0) -> tir.Call
```

Waits for completion of TMA store operations. Maps to `cp.async.bulk.wait_group.read <count>`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `count` | `int` | `0` | Maximum outstanding store groups to allow. `0` = wait for all. |

```python
T.tma_store_arrive()
# ... other work ...
T.tma_store_wait(0)  # Wait for all TMA stores to complete
T.tma_store_wait(1)  # Allow up to 1 outstanding group
```

### T.create_tma_descriptor (Internal)

```python
T.create_tma_descriptor(*args) -> tir.Call
```

Creates a TMA descriptor for tensor memory access operations. This is an internal API used by copy lowering.

**Argument format:** `create_tma_descriptor(data_type, rank, global_addr, global_shape..., global_stride..., smem_box..., smem_stride..., interleave, swizzle, l2_promotion, oob_fill)` with total arguments: `7 + 4 * rank`.

### T.tma_load (Internal)

```python
T.tma_load(*args) -> tir.Call
```

Performs a TMA load operation. Internal API used by copy lowering. Arguments: `tma_load(descriptor, mbarrier, smem_addr, coord_0, ..., coord_n, eviction_policy)`.

### T.tma_load_2sm (Internal)

```python
T.tma_load_2sm(*args) -> tir.Call
```

Performs a TMA load with 2SM (two SMs) on Blackwell. Same arguments as `tma_load` but with `use_2cta` annotation enabled for 2-CTA cooperative loading.

---

## Register Control

### T.inc_max_nreg

```python
T.inc_max_nreg(reg_count: int) -> tir.Call
```

Increments the maximum number of registers available to the current warp group. Maps to PTX `setmaxnreg.inc`.

```python
T.inc_max_nreg(232)  # Allocate 232 more registers for consumer warp group
```

### T.dec_max_nreg

```python
T.dec_max_nreg(reg_count: int) -> tir.Call
```

Decrements the maximum number of registers available to the current warp group. Maps to PTX `setmaxnreg.dec`.

```python
T.dec_max_nreg(40)  # Release 40 registers from producer warp group
```

### T.no_set_max_nreg

```python
T.no_set_max_nreg() -> tir.Call
```

Disables automatic maximum register limit management. The compiler will not emit `setmaxnreg` instructions.

### T.annotate_producer_reg_dealloc

```python
T.annotate_producer_reg_dealloc(reg_count: int = 24) -> tir.Call
```

Hints that the producer warp group will deallocate `reg_count` registers.

### T.annotate_consumer_reg_alloc

```python
T.annotate_consumer_reg_alloc(reg_count: int = 240) -> tir.Call
```

Hints that the consumer warp group needs `reg_count` registers.

---

## TMEM Operations

### T.deallocate_tmem

```python
T.deallocate_tmem(tmem: tir.Buffer) -> None
```

Explicitly deallocates a TMEM (Tensor Memory) buffer allocated by `T.alloc_tmem`. By default, TileLang inserts automatic TMEM deallocation at the end of the allocation block. Calling this suppresses the automatic deallocation and places an explicit one at the call site instead.

**Constraints:**
- Must be called by the same warp that performed the allocation.
- The buffer scope must be `shared.tmem`.
- Once called, buffer lifetime is user-managed.

```python
tmem_buf = T.alloc_tmem((128, 64), "float16")
# ... use tmem_buf ...
T.deallocate_tmem(tmem_buf)  # Explicit deallocation
```

### T.tcgen05_mma_arrive

```python
T.tcgen05_mma_arrive(mbar: tir.Buffer | BufferLoad | PrimExpr, arrive_2cta: bool = False) -> tir.Call
```

Signals UMMA (TCGEN05) barrier arrival for a shared-memory mbarrier pointer. Used on Blackwell (SM100+) GPUs.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mbar` | `Buffer`, `BufferLoad`, or `PrimExpr` | required | The mbarrier object in shared memory |
| `arrive_2cta` | `bool` | `False` | Whether to also arrive at peer CTA's barrier (2-CTA mode) |

```python
# Standard arrival
T.tcgen05_mma_arrive(mbar)

# 2-CTA cooperative arrival
T.tcgen05_mma_arrive(mbar, arrive_2cta=True)
```

### T.tcgen05_cp_warpx4

```python
T.tcgen05_cp_warpx4(smem_src, tmem_dst, tmem_col_offset=0, *, use_2cta: bool = False)
```

Copies packed scale-factor chunks from shared memory to tensor memory. Each 128-word chunk maps to 4 TMEM columns.

### T.tcgen05_sf_warp_transpose

```python
T.tcgen05_sf_warp_transpose(smem_src)
```

Warp-level transpose for packed scale-factor chunks in shared memory. Automatically handles multiple 128-word chunks.

### T.tcgen05_before_thread_sync / T.tcgen05_after_thread_sync

```python
T.tcgen05_before_thread_sync() -> tir.Call
T.tcgen05_after_thread_sync() -> tir.Call
```

Internal fences for TCGEN05 operations around thread synchronization boundaries.

---

## LDG/STG Explicit Memory Operations

These intrinsics provide explicit control over global memory load/store widths using PTX instructions. They are useful for fine-tuned memory access patterns.

### Load Operations

#### T.ldg32

```python
T.ldg32(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr
```

Loads 32 bits (4 bytes) from global memory. Returns `uint32`.

```python
val = T.ldg32(x[i])                    # Unconditional load
val = T.ldg32(x[i], pred=i < N)        # Predicated load
val = T.ldg32(x[i:i+2])               # Load 2 x fp16
```

#### T.ldg64

```python
T.ldg64(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr
```

Loads 64 bits (8 bytes). Returns `uint32x2`.

```python
val = T.ldg64(x[i])                    # Load 8 bytes
val = T.ldg64(x[i:i+4])               # Load 4 x fp16
```

#### T.ldg128

```python
T.ldg128(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr
```

Loads 128 bits (16 bytes). Returns `uint32x4`.

```python
val = T.ldg128(x[i])                   # Load 16 bytes
val = T.ldg128(x[i:i+8])              # Load 8 x fp16
```

#### T.ldg256

```python
T.ldg256(src: BufferLikeType, pred: PrimExpr = None) -> PrimExpr
```

Loads 256 bits (32 bytes). Returns `uint32x8`.

```python
val = T.ldg256(x[i])                   # Load 32 bytes
val = T.ldg256(x[i:i+16])             # Load 16 x fp16
```

### Store Operations

#### T.stg32

```python
T.stg32(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None
```

Stores 32 bits to global memory.

```python
T.stg32(y[i], val)
T.stg32(y[i], val, pred=i < N)  # Predicated store
```

#### T.stg64

```python
T.stg64(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None
```

Stores 64 bits to global memory.

```python
T.stg64(y[i:i+2], val)  # Store 2 x fp16
```

#### T.stg128

```python
T.stg128(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None
```

Stores 128 bits to global memory.

```python
T.stg128(y[i:i+4], val)  # Store 4 x fp16
```

#### T.stg256

```python
T.stg256(dst: BufferLikeType, value: PrimExpr, pred: PrimExpr = None) -> None
```

Stores 256 bits to global memory.

```python
T.stg256(y[i:i+8], val)  # Store 8 x fp16
```

### Load/Store Width Selection Guide

| Width | Bytes | Common Use Cases |
|-------|-------|-----------------|
| 32-bit | 4 | Single fp32/int32, 2x fp16/bf16 |
| 64-bit | 8 | 2x fp32, 4x fp16/bf16 |
| 128-bit | 16 | 4x fp32, 8x fp16/bf16, cache line access |
| 256-bit | 32 | 8x fp32, 16x fp16/bf16, wide vectorized |

---

## LDS Transpose Read Operations

AMD GPU (gfx950) specific intrinsics for transposed shared memory reads.

### T.ds_read_tr16_b64

```python
T.ds_read_tr16_b64(src: BufferLikeType) -> PrimExpr
```

LDS transpose read with 16-element transpose, 64-bit width. Used for FP16/BF16 MFMA matrix B-loads on MI350/MI355X (gfx950). Returns `uint32x2`.

```python
val = T.ds_read_tr16_b64(smem[i])
```

### T.ds_read_tr8_b64

```python
T.ds_read_tr8_b64(src: BufferLikeType) -> PrimExpr
```

LDS transpose read with 8-element transpose, 64-bit width. Used for FP32 MFMA matrix B-loads on MI350/MI355X (gfx950). Returns `uint32x2`.

```python
val = T.ds_read_tr8_b64(smem[i])
```

---

## Math Intrinsics

All math intrinsics use fast-math mode for improved performance at the cost of strict IEEE compliance. For IEEE-compliant alternatives, see the [IEEE-Compliant Operations](#ieee-compliant-operations) section.

### T.__log

```python
T.__log(x: PrimExpr) -> PrimExpr
```

Natural logarithm (ln). Computes `log(x)` with fast math.

```python
result = T.__log(x)
```

### T.__log2

```python
T.__log2(x: PrimExpr) -> PrimExpr
```

Base-2 logarithm. Computes `log2(x)` with fast math.

```python
result = T.__log2(x)
```

### T.__log10

```python
T.__log10(x: PrimExpr) -> PrimExpr
```

Base-10 logarithm. Computes `log10(x)` with fast math.

```python
result = T.__log10(x)
```

### T.__tan

```python
T.__tan(x: PrimExpr) -> PrimExpr
```

Tangent. Computes `tan(x)` with fast math.

```python
result = T.__tan(x)
```

### T.__cos

```python
T.__cos(x: PrimExpr) -> PrimExpr
```

Cosine. Computes `cos(x)` with fast math.

```python
result = T.__cos(x)
```

### T.__sin

```python
T.__sin(x: PrimExpr) -> PrimExpr
```

Sine. Computes `sin(x)` with fast math.

```python
result = T.__sin(x)
```

### T.__exp10

```python
T.__exp10(x: PrimExpr) -> PrimExpr
```

Power of 10. Computes `10^x` with fast math.

```python
result = T.__exp10(x)
```

### T.__exp

```python
T.__exp(x: PrimExpr) -> PrimExpr
```

Natural exponential. Computes `e^x` with fast math.

```python
result = T.__exp(x)
```

---

## Syncthreads Variants

### T.syncthreads_count

```python
T.syncthreads_count(predicate: int | PrimExpr) -> PrimExpr
```

Block barrier that synchronizes all threads and returns the number of threads whose `predicate` evaluates to non-zero. Maps to `__syncthreads_count()`.

```python
count = T.syncthreads_count(is_valid)  # How many threads have valid data
```

### T.syncthreads_and

```python
T.syncthreads_and(predicate: int | PrimExpr) -> PrimExpr
```

Block barrier that synchronizes all threads and returns non-zero only if ALL threads have a non-zero `predicate`. Maps to `__syncthreads_and()`.

```python
all_converged = T.syncthreads_and(error < threshold)
```

### T.syncthreads_or

```python
T.syncthreads_or(predicate: int | PrimExpr) -> PrimExpr
```

Block barrier that synchronizes all threads and returns non-zero if ANY thread has a non-zero `predicate`. Maps to `__syncthreads_or()`.

```python
has_error = T.syncthreads_or(error > threshold)
```

---

## Warp Match Intrinsics

These operations are CUDA-only (compute capability >= 7.0) and are not supported on HIP.

### T.match_any_sync

```python
T.match_any_sync(value: int | PrimExpr, mask: int | PrimExpr = 0xFFFFFFFF) -> PrimExpr
```

Returns a `uint32` bitmask of lanes in `mask` whose `value` equals the calling lane's value.

```python
# Find all lanes that have the same value as this lane
matching = T.match_any_sync(my_value)
# matching is a bitmask where bit N is set if lane N's value == my_value
```

### T.match_all_sync

```python
T.match_all_sync(value: int | PrimExpr, mask: int | PrimExpr = 0xFFFFFFFF) -> PrimExpr
```

Returns `mask` if all lanes in `mask` agree on `value`, else returns 0.

```python
# Check if all lanes have the same value
all_same = T.match_all_sync(my_value)
# all_same is either the full mask (all agree) or 0 (disagreement)
```

---

## Memory Descriptors

### T.initialize_wgmma_descriptor

```python
T.initialize_wgmma_descriptor(
    descriptor: tir.Buffer,
    start_address: PrimExpr,
    layout_type_: int = 0,
    leading_byte_offset: int = 0,
    stride_byte_offset: int = 0,
) -> PrimExpr
```

Initializes a WGMMA/UTCMMA shared-memory descriptor. Used on Hopper (SM90+) for tensor-memory-access-based WGMMA operations.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `descriptor` | `tir.Buffer` | 1D buffer of size 1 to hold the descriptor |
| `start_address` | `PrimExpr` | Starting address of the tensor in shared memory |
| `layout_type_` | `int` | Layout type enum value |
| `leading_byte_offset` | `int` | Byte offset between consecutive elements in the leading dimension |
| `stride_byte_offset` | `int` | Byte offset for stride |

### T.initialize_tcgen05_descriptor

```python
T.initialize_tcgen05_descriptor(
    descriptor: tir.Buffer,
    start_address: PrimExpr,
    leading_byte_offset: int,
    stride_byte_offset: int,
    base_offset: int = 0,
    leading_is_absolute: bool = False,
    swizzle_mode: int = 0,
) -> PrimExpr
```

Initializes a TCGEN05 shared-memory descriptor for Blackwell (SM100+) operations.

### T.increase_descriptor_offset

```python
T.increase_descriptor_offset(descriptor: PrimExpr, offset: PrimExpr) -> PrimExpr
```

Increases the offset of a memory descriptor. Used to advance the descriptor pointer for tiled access patterns.

---

## Synchronization Primitives

### T.sync_threads

```python
T.sync_threads(barrier_id: int = None, arrive_count: int = None) -> tir.Call
```

Synchronizes all threads in a block. Maps to `__syncthreads()` or named barrier variants.

```python
T.sync_threads()                    # Standard __syncthreads()
T.sync_threads(barrier_id=1)        # Named barrier
T.sync_threads(barrier_id=1, arrive_count=128)  # Named barrier with count
```

### T.sync_warp

```python
T.sync_warp(mask: int = None) -> tir.Call
```

Synchronizes all threads in a warp. Maps to `__syncwarp()`.

```python
T.sync_warp()               # Full warp sync
T.sync_warp(mask=0xFFFFFFFF)  # Explicit mask
```

### T.sync_global

```python
T.sync_global() -> tir.Call
```

Synchronizes all threads in the entire grid. Only available when `tir.detect_global_barrier` is enabled in the pass config.

### T.sync_grid

```python
T.sync_grid() -> tir.Call
```

Synchronizes all threads in a grid using TileLang's intrinsic grid sync.

---

## Debugging

### T.print

```python
T.print(obj: Any = None, msg: str = "", warp_group_id: int = 0, warp_id: int = 0) -> None
```

A generic print function for debugging TileLang kernels. Handles multiple input types:

- **`tir.Buffer`**: Prints buffer values. For shared/fragment buffers, only prints on the first thread (warp_group_id * 128 + warp_id * 32).
- **`tir.PrimExpr`**: Prints the expression value directly.
- **`None`**: Prints only the message string.

**Examples:**

```python
# Print a message
T.print(msg="Reached checkpoint A")

# Print a variable value
T.print(my_value, msg="my_value")

# Print a buffer
shared_buf = T.alloc_shared((128,), "float32")
T.print(shared_buf, msg="shared buffer contents")

# Print only on specific warp
T.print(my_buf, msg="warp 1 data", warp_id=1)
```

**Buffer scope handling:**

| Scope | Behavior |
|-------|----------|
| `local` | All threads print (each sees its own local data) |
| `local.fragment` | Only one thread prints (data copied to shared first) |
| `shared` / `shared.dyn` | Only one thread prints |
| `global` | All threads print (each prints its portion) |

### device_assert

```python
T.device_assert(condition: tir.PrimExpr, msg: str = "", no_stack_info: bool = False)
```

Device-side assertion. Emits a device assert call on CUDA targets. Always enabled.

```python
T.device_assert(idx >= 0 and idx < N, "Index out of bounds")
T.device_assert(result == expected, "Computation mismatch")
```

---

## Control Flow

### T.loop_break

```python
T.loop_break() -> tir.Call
```

Breaks out of the innermost loop. Maps to the `tl.loop_break` intrinsic.

```python
for i in range(N):
    if found:
        T.loop_break()
```

---

## IEEE-Compliant Operations

These operations provide strict IEEE 754 compliance with configurable rounding modes. Unlike the fast-math intrinsics, these guarantee bit-exact results.

### Rounding Modes

All IEEE operations accept a `rounding_mode` parameter:

| Mode | Value | Description |
|------|-------|-------------|
| `rn` | Round to nearest (ties to even) | Default |
| `rz` | Round toward zero | Truncation |
| `ru` | Round toward positive infinity | Ceiling |
| `rd` | Round toward negative infinity | Floor |

### T.ieee_add

```python
T.ieee_add(x: PrimExpr, y: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant addition: `x + y`.

### T.ieee_sub

```python
T.ieee_sub(x: PrimExpr, y: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant subtraction: `x - y`.

### T.ieee_mul

```python
T.ieee_mul(x: PrimExpr, y: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant multiplication: `x * y`.

### T.ieee_fmaf

```python
T.ieee_fmaf(x: PrimExpr, y: PrimExpr, z: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant fused multiply-add: `x * y + z`. Single rounding at the end.

### T.ieee_frcp

```python
T.ieee_frcp(x: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant reciprocal: `1/x`.

### T.ieee_fsqrt

```python
T.ieee_fsqrt(x: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant square root: `sqrt(x)`.

### T.ieee_frsqrt

```python
T.ieee_frsqrt(x: PrimExpr) -> PrimExpr
```

IEEE-compliant reciprocal square root: `1/sqrt(x)`. Round-to-nearest only.

### T.ieee_fdiv

```python
T.ieee_fdiv(x: PrimExpr, y: PrimExpr, rounding_mode="rn") -> PrimExpr
```

IEEE-compliant division: `x / y`.

---

## Packed x2 Element-wise Operations

These operations work on packed 2-element vectors (`float32x2`, `bfloat16x2`, `float16x2`), performing element-wise operations on both elements simultaneously.

### T.add2

```python
T.add2(x: PrimExpr, y: PrimExpr) -> PrimExpr
```

Packed element-wise addition: `x + y`.

### T.sub2

```python
T.sub2(x: PrimExpr, y: PrimExpr) -> PrimExpr
```

Packed element-wise subtraction: `x - y`.

### T.mul2

```python
T.mul2(x: PrimExpr, y: PrimExpr) -> PrimExpr
```

Packed element-wise multiplication: `x * y`.

### T.fma2

```python
T.fma2(x: PrimExpr, y: PrimExpr, z: PrimExpr) -> PrimExpr
```

Packed fused multiply-add: `x * y + z`.

### T.max2

```python
T.max2(x: PrimExpr, y: PrimExpr) -> PrimExpr
```

Packed element-wise maximum.

### T.min2

```python
T.min2(x: PrimExpr, y: PrimExpr) -> PrimExpr
```

Packed element-wise minimum.

### T.abs2

```python
T.abs2(x: PrimExpr) -> PrimExpr
```

Packed element-wise absolute value.

### Supported Data Types for Packed x2 Operations

| Type | Description |
|------|-------------|
| `float32x2` | Two 32-bit floats |
| `bfloat16x2` | Two BF16 values |
| `float16x2` | Two FP16 values |

---

## SM70 Tensor Core Operations

### T.ptx_mma_sm70

```python
T.ptx_mma_sm70(
    shape: str,           # e.g., "m16n16k4"
    A_layout: str,        # "row" or "col"
    B_layout: str,        # "row" or "col"
    A_dtype: str,         # e.g., "fp16"
    B_dtype: str,         # e.g., "fp16"
    C_dtype: str,         # "fp16" or "fp32"
    multiplicand_a: Var,
    a_index: Expr,
    multiplicand_b: Var,
    b_index: Expr,
    accumulator: Var,
    c_index: Expr,
) -> PrimExpr
```

PTX tensor core MMA instructions for SM70 (Volta). Supports `m16n16k4` shape with FP16 inputs and FP16/FP32 accumulation.

```python
T.ptx_mma_sm70(
    "m16n16k4", "row", "col",
    "fp16", "fp16", "fp32",
    A_frag.data, 0, B_frag.data, 0, C_frag.data, 0,
)
```

---

## Quick Reference: Backend Support Matrix

| Intrinsic | CUDA | HIP | CPU (C) |
|-----------|------|-----|---------|
| `T.__ldg` | Yes | Fallback | Fallback |
| `T.mbarrier_wait_parity` | SM80+ | gfx9+ | No |
| `T.mbarrier_arrive` | SM80+ | gfx9+ | No |
| `T.mbarrier_expect_tx` | SM80+ | gfx9+ | No |
| `T.get_lane_idx` | Yes | Yes | No |
| `T.get_warp_idx` | Yes | Yes | No |
| `T.shuffle_elect` | Yes | Yes | No |
| `T.any_sync` | Yes | Yes | No |
| `T.all_sync` | Yes | Yes | No |
| `T.ballot_sync` | Yes | Yes | No |
| `T.shfl_*` | Yes | Yes | No |
| `T.warpgroup_*` | SM90+ | No | No |
| `T.tma_*` | SM90+ | No | No |
| `T.tcgen05_*` | SM100+ | No | No |
| `T.ds_read_*` | No | gfx950 | No |
| `T.ldg/stg*` | Yes | No | No |
| `T.match_*_sync` | SM70+ | No | No |
| `T.__log/sin/cos/exp` | Yes | Yes | Yes |
| `T.ieee_*` | Yes | Yes | Yes |
| `T.print` | Yes | Yes | Limited |
