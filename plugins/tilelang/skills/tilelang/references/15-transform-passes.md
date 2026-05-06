# Transform Passes

This document provides comprehensive reference for all transform passes in TileLang's compilation pipeline. Each pass is documented with its purpose, configuration options, and examples showing its effect on the IR.

## Table of Contents

- [Overview](#overview)
- [Pass Ordering and Dependencies](#pass-ordering-and-dependencies)
- [High-Level Tile IR Passes](#high-level-tile-ir-passes)
- [Memory and Layout Passes](#memory-and-layout-passes)
- [Pipeline and Synchronization Passes](#pipeline-and-synchronization-passes)
- [Lowering Passes](#lowering-passes)
- [Optimization Passes](#optimization-passes)
- [Host/Device Splitting Passes](#hostdevice-splitting-passes)
- [Backend-Specific Passes](#backend-specific-passes)
- [Utility Passes](#utility-passes)
- [Pass Configuration Options](#pass-configuration-options)

---

## Overview

TileLang's transform passes are organized into three phases within the compilation pipeline:

1. **PreLowerSemanticCheck**: Validation before any modifications.
2. **LowerAndLegalize**: Transform frontend Tile IR into TVM-compatible TIR.
3. **OptimizeForTarget**: Apply target-specific optimizations.

Passes are invoked through `tilelang.transform` and are implemented as TVM transform passes (both Python and C++ via FFI). Most passes are configured through the `PassContext` mechanism.

---

## Pass Ordering and Dependencies

The passes must execute in a specific order due to data dependencies. Here is the complete ordered sequence:

### LowerAndLegalize Phase

```
1.  BindTarget(target)
2.  LetInline                     [conditional: tl.force_let_inline]
3.  AddWrapperForSingleBufStore
4.  LegalizeNegativeIndex
5.  VerifyParallelLoop            [conditional: !tl.disable_data_race_check]
6.  InjectAssumes
7.  Simplify
8.  LayoutReducer
9.  ProducerConsumerWarpSpecialized  [conditional: CUDA + TMA + !tl.disable_warp_specialized]
10. LowerBlackwell2SM
11. PipelinePlanning
12. InjectSoftwarePipeline
13. Simplify
14. LayoutInferences
15. LowerTileOp
16. LowerL2Persistent
17. DecoupleTypeCast
18. LegalizeVectorizedLoop
19. LegalizeSafeMemoryAccess
20. LowerAccessPtr
21. Simplify
22. HoistNonRestrictParams
```

### OptimizeForTarget Phase

```
23. LowerSharedTmem
24. IfStmtBinding
25. PlanAndUpdateBufferAllocationLocation
26. LowerSharedBarrier
27. FuseMBarrierArriveExpectTx   [conditional: has TMA]
28. HoistGlobalBufferAllocations
29. LowerOpaqueBlock
30. Simplify
31. NarrowDataType(32)
32. FlattenBuffer
33. ConfigIndexBitwidth
34. Simplify
35. VectorizeLoop                [conditional: !tir.disable_vectorize]
36. StorageRewrite
37. LoopUnswitching
38. UnrollLoop
39. RenormalizeSplitPattern
40. Simplify
41. RemoveNoOp
42. HoistIfThenElse
43. VerifyMemory
44. AnnotateEntryFunc
45. InferFragment
46. LowerThreadAllreduce
47. LowerLDGSTG
48. LowerHopperIntrin
49. ThreadSync("global")         [conditional: tir.detect_global_barrier]
50. AnnotateDeviceRegions
51. SplitHostDevice
52. MarkCudaSyncCalls
53. AnnotateReadOnlyParams
54. MergeSharedMemoryAllocations
55. InjectFenceProxy
56. ThreadSync("shared")
57. ThreadSync("shared.dyn")
58. InjectTcgen05Fence
59. MergeIfStmt
60. AnnotateWarpGroupRegAlloc    [conditional: warp specialized]
61. MakePackedAPI
62. Simplify
63. LowerDeviceKernelLaunch
64. PersistThreadblock
```

---

## High-Level Tile IR Passes

### ClusterPlanning

```python
tilelang.transform.ClusterPlanning()
```

Plans GPU cluster-level execution for kernels that use thread block clusters (Hopper SM90+). Determines how thread blocks are grouped into clusters and manages cross-CTA shared memory access.

**When it runs:** During `LowerAndLegalize`, before layout inference.

**Effect:** Adds cluster annotations to the IR that guide subsequent lowering passes.

### PipelinePlanning

```python
tilelang.transform.PipelinePlanning()
```

Analyzes tile operations in pipelined loops and plans multi-stage buffer allocation. Determines how many pipeline stages are needed and how shared memory buffers should be replicated.

**When it runs:** During `LowerAndLegalize`, before `InjectSoftwarePipeline`.

**Effect:** Adds pipeline stage annotations to shared memory buffers, transforming single buffers into multi-stage arrays.

**Example effect:**

```python
# Before PipelinePlanning:
A_shared = T.alloc_shared((128, 32), "float16")

# After PipelinePlanning (with num_stages=2):
A_shared = T.alloc_shared((2, 128, 32), "float16")  # 2-stage buffer
```

### InstructionAnnotation

```python
tilelang.transform.InstructionAnnotation()
```

Annotates tile operations with coarse-grained instruction kinds. Adds `tl_instruction_kind` annotation to each tile-op Call node indicating the instruction category.

**Annotation categories:**
- `"tma"`: Tensor Memory Access operations
- `"cp_async"`: Async copy operations
- `"sync"`: Synchronization operations
- `"wgmma"`: Warp Group Matrix Multiply-Accumulate operations

**When it runs:** Before `LayoutInferences` and `LowerTileOp`.

---

## Memory and Layout Passes

### LayoutInferences

```python
tilelang.transform.LayoutInferences()
```

Infers memory layouts for fragment buffers and shared memory buffers based on how they are consumed by operations (e.g., Tensor Core operations).

**When it runs:** During `LowerAndLegalize`, after `InjectSoftwarePipeline` and before `LowerTileOp`.

**Effect:** Assigns layout annotations to buffers that were not explicitly annotated by the user. The inferred layout is based on the operation that consumes the buffer:

- Buffers consumed by WGMMA: Assigned WGMMA-compatible shared memory layout.
- Buffers consumed by MMA: Assigned MMA fragment layout.
- Buffers consumed by copy operations: Assigned optimal copy layout (e.g., swizzled for bank conflict avoidance).

**Example:**

```python
# Before LayoutInferences:
A_shared = T.alloc_shared((128, 32), "float16")  # No layout annotation
C_frag = T.alloc_fragment((64, 64), "float32")    # No layout annotation
T.gemm(A_shared, B_shared, C_frag)

# After LayoutInferences:
# A_shared gets WGMMA-compatible shared memory layout
# C_frag gets WGMMA accumulator fragment layout
```

### LayoutReducer

```python
tilelang.transform.LayoutReducer()
```

Sets layouts for reduction operations. Normalizes reduction buffer layouts for efficient cross-warp reduction.

### LowerAccessPtr

```python
tilelang.transform.LowerAccessPtr()
```

Lowers TileLang's frontend `tl.access_ptr` operations to standard `tir.builtin.tvm_access_ptr`. This converts the high-level access pointer metadata (buffer, rw_mask, extent) into the low-level representation used by TVM's memory system.

**When it runs:** During `LowerAndLegalize`, after safety checks.

**Example effect:**

```python
# Before:
ptr = tl.access_ptr(A[i], "r", extent=128, rw_mask=1)

# After:
ptr = tir.builtin.tvm_access_ptr(dtype, data, offset, extent, rw_mask)
```

### FlattenBuffer

```python
tilelang.transform.FlattenBuffer()
```

Flattens multi-dimensional buffer accesses into 1D linear indexing. Converts `buf[i, j]` to `buf[i * stride + j]`.

**When it runs:** During `OptimizeForTarget`, after `NarrowDataType`.

**Effect:** All buffer accesses become 1D, which simplifies subsequent code generation.

### PlanAndUpdateBufferAllocationLocation

```python
tilelang.transform.PlanAndUpdateBufferAllocationLocation()
```

Plans optimal buffer allocation placement within PrimFuncs. Moves buffer allocations to the earliest point where they are needed and the latest point where they can be safely deallocated.

**When it runs:** During `OptimizeForTarget`, after `IfStmtBinding`.

### HoistGlobalBufferAllocations

```python
tilelang.transform.HoistGlobalBufferAllocations()
```

Hoists global buffer allocations to the top of the block (host side). Ensures global memory allocations are visible to the host launcher.

### MergeSharedMemoryAllocations

```python
tilelang.transform.MergeSharedMemoryAllocations(
    enable_aggressive_merge: bool = False,
    align_bytes: int = 16,
)
```

Merges multiple shared memory allocations into a single allocation to reduce total shared memory usage. This is particularly important for occupancy optimization.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_aggressive_merge` | `bool` | `False` | Enable aggressive merging of aliased buffers |
| `align_bytes` | `int` | `16` | Alignment requirement in bytes |

**When it runs:** During `OptimizeForTarget`, after `SplitHostDevice` (must run after because the merged allocation is at the device function start).

**Configuration:**
- `tl.enable_aggressive_shared_memory_merge`: Enable aggressive merging.
- `tl.debug_merge_shared_memory_allocations`: Debug information for merging.
- When warp specialization is enabled, aggressive merging is automatically disabled to avoid issues with pipeline buffer aliasing.

**Example effect:**

```python
# Before (total: 64KB):
A_shared = T.alloc_shared((128, 128), "float16")  # 32KB
B_shared = T.alloc_shared((128, 128), "float16")  # 32KB

# After (total: 32KB, reused):
shared_buf = T.alloc_shared((128, 128), "float16")  # 32KB
# A and B use the same memory at different times
```

### StorageRewrite

```python
tilelang.transform.StorageRewrite()
```

Rewrites storage allocation for efficiency. Analyzes buffer liveness and reuses storage where possible. This pass is particularly effective for local (register) memory optimization.

**Configuration:**
- `tl.storage_rewrite_detect_inplace`: Control inplace detection. When `False` (default), keeps distinct temporaries to avoid aliasing.

---

## Pipeline and Synchronization Passes

### InjectSoftwarePipeline

```python
tilelang.transform.InjectSoftwarePipeline()
```

Injects software pipeline structure into pipelined loops. Transforms a sequential loop into a multi-stage pipeline where data loading and computation overlap.

**When it runs:** During `LowerAndLegalize`, after `PipelinePlanning`.

**Example effect:**

```python
# Before (sequential):
for k in range(K):
    T.copy(A_global[k], A_shared)
    T.gemm(A_shared, B_shared, C_frag)

# After (pipelined, 2 stages):
# Prologue
T.copy(A_global[0], A_shared[0])
# Steady state
for k in range(K - 1):
    T.copy(A_global[k + 1], A_shared[(k + 1) % 2])  # Async, overlapped
    T.mbarrier_wait(...)  # Wait for previous stage data
    T.gemm(A_shared[k % 2], B_shared, C_frag)
# Epilogue
T.gemm(A_shared[(K - 1) % 2], B_shared, C_frag)
```

### ProducerConsumerWarpSpecialized

```python
tilelang.transform.ProducerConsumerWarpSpecialized()
```

Rewrites eligible pipelined tile-op loops into warp-specialized producer and consumer branches with explicit barrier synchronization. Producer warps handle data loading while consumer warps handle computation.

**When it runs:** During `LowerAndLegalize`, before `LayoutInferences`.

**Conditions:**
- Target must be CUDA with TMA support.
- `tl.disable_warp_specialized` must be `False`.
- The kernel must have eligible pipelined loops.

**Effect:** Splits the kernel into producer and consumer warp groups:

```python
# Before:
with T.Kernel(...):
    for k in range(K):
        T.copy(A_global, A_shared)
        T.gemm(A_shared, B_shared, C_frag)

# After warp specialization:
with T.Kernel(...):
    warp_group_id = T.get_warp_group_idx()
    if warp_group_id == 0:
        # Producer: load data
        for k in range(K):
            T.mbarrier_wait_parity(...)
            T.copy(A_global, A_shared)  # TMA load
            T.mbarrier_arrive(...)
    else:
        # Consumer: compute
        for k in range(K):
            T.mbarrier_wait_parity(...)
            T.gemm(A_shared, B_shared, C_frag)
            T.mbarrier_arrive(...)
```

### ThreadSync

```python
tilelang.transform.ThreadSync(storage_scope: str)
```

Inserts synchronization barriers for parallel read/write of shared buffers. Generates `__syncthreads()` calls (or equivalent) at points where data hazards are detected.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage_scope` | `str` | Target scope: `"shared"`, `"shared.dyn"`, or `"global"` |

**Scopes:**

| Scope | Synchronization | Condition |
|-------|----------------|-----------|
| `"shared"` | `__syncthreads()` | Always for shared memory |
| `"shared.dyn"` | `__syncthreads()` | For dynamic shared memory |
| `"global"` | Grid-level sync | Only when `tir.detect_global_barrier=True` |

**Configuration:**
- `tl.disable_thread_storage_sync`: Disable automatic sync insertion.

### LowerSharedBarrier

```python
tilelang.transform.LowerSharedBarrier()
```

Lowers shared memory barrier operations to hardware-specific implementations. Converts high-level barrier constructs to PTX `mbarrier` instructions on NVIDIA or equivalent on AMD.

### FuseMBarrierArriveExpectTx

```python
tilelang.transform.FuseMBarrierArriveExpectTx()
```

Fuses simple `expect_tx` -> TMA issue -> `arrive` patterns into a single `arrive_and_expect_tx` operation. This optimization reduces barrier synchronization overhead by combining two operations into one.

**When it runs:** During `OptimizeForTarget`, only when TMA operations are present.

**Example effect:**

```python
# Before:
T.mbarrier_expect_tx(mbar, tx_bytes)
T.copy(A_global, A_shared)  # TMA
T.mbarrier_arrive(mbar)

# After:
T.copy(A_global, A_shared)  # TMA
T.mbarrier_arrive_expect_tx(mbar, tx_bytes)  # Combined
```

### InjectFenceProxy

```python
tilelang.transform.InjectFenceProxy()
```

Injects `fence.proxy.async.shared::cta` instructions at appropriate points in the IR. These fences ensure that asynchronous proxy operations (e.g., TMA stores) are visible to subsequent shared memory accesses.

**When it runs:** During `OptimizeForTarget`, after `MergeSharedMemoryAllocations`.

**Note:** This pass is a no-op on targets that lack the TMA/async-proxy programming model.

### InjectTcgen05Fence

```python
tilelang.transform.InjectTcgen05Fence()
```

Injects `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` at conservative TCGEN05/TMEM synchronization boundaries on Blackwell (SM100+) targets.

**Behavior:**
- Wraps CTA-wide shared-memory syncs.
- Inserts fences around linear mbarrier wait/use and use/arrive handoff patterns.
- Intentionally conservative -- does not infer arbitrary barrier protocols.

**When it runs:** During `OptimizeForTarget`, after `ThreadSync`.

**Note:** No-op on non-SM100 targets or functions without TMEM operations.

---

## Lowering Passes

### LowerTileOp

```python
tilelang.transform.LowerTileOp()
```

Lowers high-level tile operations (`T.copy`, `T.gemm`, `T.fill`, `T.reduce`) into low-level TIR operations. This is the primary lowering pass that converts TileLang DSL operations to hardware-specific instructions.

**Operations lowered:**

| DSL Operation | Lowered To |
|--------------|-----------|
| `T.copy(global -> shared)` | TMA load or cp.async or synchronous copy |
| `T.copy(shared -> global)` | TMA store or synchronous store |
| `T.copy(shared -> fragment)` | Shared memory load with inferred layout |
| `T.gemm(...)` | WGMMA or MMA or MFMA tensor core instruction |
| `T.fill(buf, value)` | Loop of buffer stores |
| `T.reduce(buf, ...)` | Reduction with appropriate layout |

**When it runs:** During `LowerAndLegalize`, after `LayoutInferences`.

**Note:** Sets the `tl.has_tma` attribute on the function if TMA operations were generated.

### LowerHopperIntrin

```python
tilelang.transform.LowerHopperIntrin()
```

Lowers Hopper (SM90) specific intrinsics. Converts high-level Hopper operations to PTX instructions.

**When it runs:** During `OptimizeForTarget`.

### LowerPTXAsyncCopy

```python
tilelang.transform.LowerPTXAsyncCopy()
```

Lowers eligible global-to-shared memory copies into PTX `cp.async` instructions on CUDA. When enabled, this pass rewrites plain `BufferStore` patterns in `T.Parallel` loops into `tir.ptx_cp_async` and inserts `tir.ptx_commit_group` + `tir.ptx_wait_group(0)`.

**Key behaviors:**
- Converts Ramp-based global BufferStore to `cp.async`.
- Inserts commit/wait to preserve synchronous semantics.
- Avoids duplicating existing commit/wait patterns.
- Only enabled inside software-pipelined loops (`num_stages > 0`).
- Can be requested per-loop via `T.Parallel(..., prefer_async=True)`.

**Configuration:**
- `tl.enable_async_copy`: Enable/disable (default: `True`).

### LowerLDGSTG

```python
tilelang.transform.LowerLDGSTG()
```

Lowers Ramp-based global memory load/store to ldg/stg intrinsics. Converts vectorized global memory operations into explicit `ldg32/64/128/256` and `stg32/64/128/256` instructions for better codegen.

**Key behaviors:**
- Converts Ramp-based global BufferLoad to ldg intrinsics.
- Converts Ramp-based global BufferStore to stg intrinsics.
- Supports predicated loads (`if_then_else` with `else=0`).
- Supports predicated stores (if in then case).
- Skips loads in async scope (handled by `LowerPTXAsyncCopy`).
- Only enabled for CUDA targets.

**Configuration:**
- `tl.enable_lower_ldgstg`: Enable non-predicated lowering (default: `False`).
- `tl.enable_lower_ldgstg_predicated`: Enable predicated lowering (default: `False`).
- `tl.disable_vectorize_256`: Disable 256-bit ldg/stg (default: `False`).

### LowerL2Persistent

```python
tilelang.transform.LowerL2Persistent()
```

Lowers L2 cache persistence annotations (`T.annotate_l2_hit_ratio`) into CUDA access policy window setup code in the host launcher.

### LowerSharedTmem

```python
tilelang.transform.LowerSharedTmem()
```

Lowers shared TMEM (Tensor Memory) allocations to specific initialization slots on Blackwell (SM100+) GPUs.

### LowerOpaqueBlock

```python
tilelang.transform.LowerOpaqueBlock()
```

Lowers opaque block constructs into standard TIR blocks for consistent codegen.

### LowerThreadAllreduce

```python
tilelang.transform.LowerThreadAllreduce()
```

Lowers thread-level all-reduce operations (cross-warp reductions) into efficient warp shuffle + shared memory sequences.

### LowerIntrin

```python
tilelang.transform.LowerIntrin()
```

Lowers remaining TileLang intrinsics that have not been handled by earlier passes.

### LowerDeviceKernelLaunch

```python
tilelang.transform.LowerDeviceKernelLaunch()
```

Lowers device kernel launch constructs to target-specific IR. Transforms high-level kernel launch intrinsics into the low-level IR needed by backend code generators.

### LowerDeviceStorageAccessInfo

```python
tilelang.transform.LowerDeviceStorageAccessInfo()
```

Lowers attached storage access information on device. Must run after all storage access analysis is complete.

### LowerBlackwell2SM

```python
tilelang.transform.LowerBlackwell2SM()
```

Lowers 2SM TCGEN5MMA and related operations on Blackwell targets. Must run before `LayoutInferences` so that `use_2cta` annotations are visible during layout inference.

### MakePackedAPI

```python
tilelang.transform.MakePackedAPI()
```

Creates the packed function API for the compiled kernel. Wraps the kernel function with TVM's packed function calling convention for interoperability with the runtime system.

---

## Optimization Passes

### Simplify

```python
tilelang.transform.Simplify(simplify_arguments: bool = False)
```

Simplifies TIR expressions using TileLang's enhanced simplification pass. This is an extended version of TVM's built-in `Simplify` with additional pattern matching for TileLang-specific constructs.

**Configuration (via `tl.Simplify` dict):**

| Sub-key | Default | Description |
|---------|---------|-------------|
| `transitively_prove_inequalities` | `False` | Enable transitive inequality proving |
| `convert_boolean_to_and_of_ors` | `False` | Convert boolean to AND-of-ORs form |
| `apply_constraints_to_boolean_branches` | `False` | Apply constraints to boolean branches |
| `propagate_knowns_to_prove_conditional` | `False` | Propagate knowns to prove conditionals |
| `propagate_knowns_to_simplify_expressions` | `False` | Propagate knowns to simplify expressions |
| `enable_simplify_let_inline` | `True` | Enable let statement inlining |

### LetInline

```python
tilelang.transform.LetInline()
```

Inlines all `let` bindings, replacing each let-bound variable with its value. This can expose optimization opportunities but may increase code size.

**Configuration:**
- `tl.force_let_inline`: Force let inlining during simplification.

### VectorizeLoop

```python
tilelang.transform.VectorizeLoop(enable_vectorize: bool = True)
```

Vectorizes inner loops where possible. Converts scalar loops into vectorized load/store operations.

**Configuration:**
- `tir.disable_vectorize`: Disable vectorization.
- `tl.enable_vectorize_planner_verbose`: Debug vectorization decisions.

### UnrollLoop

```python
tilelang.transform.UnrollLoop()
```

Unrolls loops based on configuration options:

| Option | Description |
|--------|-------------|
| `auto_max_step` | Maximum steps for auto-unrolling |
| `auto_max_depth` | Maximum nesting depth for auto-unrolling |
| `auto_max_extent` | Maximum loop extent for unrolling |
| `explicit_unroll` | Whether to explicitly unroll vs. set pragma |
| `unroll_local_access` | Always unroll local access loops |

### StorageRewrite

```python
tilelang.transform.StorageRewrite()
```

Optimizes storage allocation by analyzing buffer liveness and reuse opportunities.

**Configuration:**
- `tir.disable_storage_rewrite`: Disable storage rewrite.
- `tl.storage_rewrite_detect_inplace`: Allow inplace buffer aliasing (default: `False`).

### LoopUnswitching

```python
tilelang.transform.LoopUnswitching()
```

Hoists loop-invariant if statements out of loops. This can eliminate branch divergence inside loops.

**Configuration:**
- `tl.disable_loop_unswitching`: Disable this pass.
- `tl.loop_unswitching_allow_non_trivial_else`: Allow more aggressive hoisting.

**Example effect:**

```python
# Before:
for i in range(N):
    if condition:  # condition is loop-invariant
        A[i] = B[i] + 1
    else:
        A[i] = B[i] * 2

# After:
if condition:
    for i in range(N):
        A[i] = B[i] + 1
else:
    for i in range(N):
        A[i] = B[i] * 2
```

### HoistBroadcastValues

```python
tilelang.transform.HoistBroadcastValues()
```

Hoists broadcast values (constants that are replicated across threads) out of loops and parallel regions to avoid redundant computation.

### DecoupleTypeCast

```python
tilelang.transform.DecoupleTypeCast()
```

Separates type cast operations from vectorized operations. This allows vectorization to proceed on the core computation while handling type conversions separately.

### AnnotateWarpGroupRegAlloc

```python
tilelang.transform.AnnotateWarpGroupRegAlloc()
```

Injects `set_max_nreg` calls into warp-specialized functions. Analyzes the function to collect register allocation hints and inserts appropriate register management instructions in producer and consumer branches.

**When it runs:** During `OptimizeForTarget`, only when warp specialization is enabled.

**Effect:**

```python
# Before:
if warp_group_id == 0:
    # Producer
    ...
else:
    # Consumer
    ...

# After:
if warp_group_id == 0:
    T.dec_max_nreg(24)  # Producer uses fewer registers
    # Producer
    ...
else:
    T.inc_max_nreg(232)  # Consumer uses more registers for Tensor Core
    # Consumer
    ...
```

---

## Host/Device Splitting Passes

### AnnotateDeviceRegions

```python
tilelang.transform.AnnotateDeviceRegions()
```

Annotates regions of the IR that should execute on the device. Marks functions with the appropriate calling convention.

### SplitHostDevice

```python
tilelang.transform.SplitHostDevice()
```

Splits the IR module into host and device functions. Host functions manage kernel launch and parameter marshaling; device functions contain the actual GPU kernel code.

### AnnotateReadOnlyParams

```python
tilelang.transform.AnnotateReadOnlyParams()
```

Annotates read-only handle parameters for PrimFuncs. Adds the `tl.readonly_param_indices` attribute listing parameter indices that are never written. This enables the CUDA codegen to emit `const` qualifiers for better read-only cache utilization.

### MakePackedAPI

```python
tilelang.transform.MakePackedAPI()
```

Wraps the kernel function with TVM's packed function API. Creates the host-side wrapper that marshals parameters and launches the kernel.

### PersistThreadblock

```python
tilelang.transform.PersistThreadblock()
```

Transforms thread block scheduling to persistent mode, where thread blocks remain active and pull work from a queue rather than terminating after one iteration.

---

## Backend-Specific Passes

### MarkCudaSyncCalls

```python
tilelang.transform.MarkCudaSyncCalls(have_pdl: bool = False)
```

Marks CUDA synchronization calls in the IR. When PDL (Programmatic Dependent Launch) is available (Hopper+), annotates synchronization points for PDL optimization.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `have_pdl` | `bool` | Whether PDL is available on the target |

### ConfigIndexBitwidth

```python
tilelang.transform.ConfigIndexBitwidth()
```

Configures the index bitwidth for the generated code. Must run after `FlattenBuffer` as it flattens index computation.

**Configuration:**
- `tl.config_index_bitwidth`: Bitwidth (default: 32).

### AddWrapperForSingleBufStore

```python
tilelang.transform.AddWrapperForSingleBufStore()
```

Wraps single buffer store operations for consistent handling by downstream passes. Ensures all buffer stores follow a uniform pattern.

---

## Utility Passes

### IfStmtBinding

```python
tilelang.transform.IfStmtBinding()
```

Binds if-statement conditions to named variables. This improves code readability and enables better optimization by other passes.

**Example effect:**

```python
# Before:
if (i < N and j < M):
    A[i, j] = B[i, j] + 1

# After:
let cond = i < N and j < M
if cond:
    A[i, j] = B[i, j] + 1
```

### MergeIfStmt

```python
tilelang.transform.MergeIfStmt()
```

Merges consecutive if statements with identical conditions. Reduces branching overhead.

**Example effect:**

```python
# Before:
if i < N:
    A[i] = 0
if i < N:
    B[i] = 0

# After:
if i < N:
    A[i] = 0
    B[i] = 0
```

### LegalizeNegativeIndex

```python
tilelang.transform.LegalizeNegativeIndex()
```

Converts negative indices in buffer loads/stores to canonical non-negative form. For example, `A[-1]` becomes `A[N - 1]`.

### LegalizeVectorizedLoop

```python
tilelang.transform.LegalizeVectorizedLoop()
```

Ensures that vectorized loops are valid. Checks that loop extents are divisible by the vectorization factor and inserts necessary adjustments.

### LegalizeSafeMemoryAccess

```python
tilelang.transform.LegalizeSafeMemoryAccess()
```

Inserts safety checks for out-of-bounds memory accesses. Generates boundary condition handling for edge tiles where the tensor dimensions are not perfectly divisible by the tile size.

**Configuration:**
- `tl.disable_safe_memory_legalize`: Disable this pass.
- `tl.disable_out_of_bound_warning`: Disable OOB warnings (default: `True`).

**Interaction with `T.annotate_safe_value`:** When safe values are annotated, this pass generates unconditional loads with padding rather than conditional guards.

### InjectAssumes

```python
tilelang.transform.InjectAssumes()
```

Injects assume statements for natural shape boundary conditions. These assumptions help TVM's prover verify loop bounds and eliminate redundant checks.

### VerifyParallelLoop

```python
tilelang.transform.VerifyParallelLoop()
```

Verifies parallel loop correctness. Checks for potential data races in `T.Parallel` loops where multiple threads write to the same memory location.

**Configuration:**
- `tl.disable_data_race_check`: Disable race checking.

### VerifyMemory

```python
tir.transform.VerifyMemory()
```

Validates memory access patterns after optimization. Ensures all memory accesses are within bounds and follow the memory model rules.

### RemoveNoOp

```python
tir.transform.RemoveNoOp()
```

Removes no-op statements from the IR. Eliminates dead code that has no effect on program behavior.

### HoistIfThenElse

```python
tir.transform.HoistIfThenElse()
```

Hoists if-then-else statements out of loops when the condition is loop-invariant.

### AnnotateEntryFunc

```python
tir.transform.AnnotateEntryFunc()
```

Annotates the entry function in the IR module so that the code generator knows which function to compile as the kernel entry point.

### InferFragment

```python
tir.transform.InferFragment()
```

Infers fragment information for Tensor Core operations. Determines the physical register layout for accumulator and operand fragments.

### RenormalizeSplitPattern

```python
tir.transform.RenormalizeSplitPattern()
```

Renormalizes split patterns in loop nests to ensure consistent representation.

### NarrowDataType

```python
tir.transform.NarrowDataType(bits: int)
```

Narrows data types to the specified bit width. For example, `NarrowDataType(32)` converts 64-bit indices to 32-bit where safe.

---

## Pass Configuration Options

All pass configuration is done through the `PassContext` mechanism:

```python
import tilelang

with tilelang.transform.PassContext(opt_level=3, config={
    "tl.enable_fast_math": True,
    "tl.Simplify": {
        "transitively_prove_inequalities": True,
    },
}):
    kernel = tilelang.compile(func)
```

### Complete Configuration Key Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tl.Simplify` | dict | see above | Simplification sub-config |
| `tl.disable_data_race_check` | bool | `False` | Disable data race checking |
| `tl.disable_prelower_semantic_check` | bool | `False` | Disable pre-lower checks |
| `tl.disable_warp_specialized` | bool | `False` | Disable warp specialization |
| `tl.enable_fast_math` | bool | `False` | Enable fast math in nvcc |
| `tl.ptxas_register_usage_level` | int | `None` | PTXAS register level (0-10) |
| `tl.enable_ptxas_verbose_output` | bool | `False` | Verbose PTXAS output |
| `tl.device_compile_flags` | str/list | `None` | Extra nvcc flags |
| `tl.config_index_bitwidth` | int | `32` | Index bitwidth |
| `tl.disable_tma_lower` | bool | `False` | (Deprecated) Disable TMA lowering |
| `tl.disable_safe_memory_legalize` | bool | `False` | Disable safe memory access |
| `tl.disable_vectorize_256` | bool | `False` | Disable 256-bit vectorization |
| `tl.enable_async_copy` | bool | `True` | Enable cp.async lowering |
| `tl.enable_lower_ldgstg` | bool | `False` | Enable LDG/STG lowering |
| `tl.enable_lower_ldgstg_predicated` | bool | `False` | Enable predicated LDG/STG |
| `tl.enable_vectorize_planner_verbose` | bool | `False` | Verbose vectorization |
| `tl.disable_wgmma` | bool | `False` | Disable WGMMA usage |
| `tl.debug_merge_shared_memory_allocations` | bool | `False` | Debug SMEM merge |
| `tl.enable_aggressive_shared_memory_merge` | bool | `False` | Aggressive SMEM merge |
| `tl.disable_shuffle_elect` | bool | `False` | Disable shuffle election |
| `tl.disable_loop_unswitching` | bool | `False` | Disable loop unswitching |
| `tl.loop_unswitching_allow_non_trivial_else` | bool | `False` | Allow non-trivial else in unswitching |
| `tl.disable_thread_storage_sync` | bool | `False` | Disable auto thread sync |
| `tl.force_let_inline` | bool | `False` | Force let inlining |
| `tl.ast_print_enable` | bool | `False` | Enable AST printing |
| `tl.layout_visualization_enable` | bool | `False` | Enable layout visualization |
| `tl.layout_visualization_formats` | str | `"txt"` | Visualization formats |
| `tl.storage_rewrite_detect_inplace` | bool | `False` | Allow inplace in storage rewrite |
| `tl.enable_dump_ir` | bool | `False` | Enable IR dumping |
| `tl.dump_ir_path` | str | `"./dump_ir"` | IR dump directory |
| `tl.disable_out_of_bound_warning` | bool | `True` | Disable OOB warnings |
| `tir.enable_equiv_terms_in_cse_tir` | bool | `True` | CSE equivalent terms |
| `tir.disable_cse_tir` | bool | `False` | Disable TIR CSE |
| `tir.Simplify` | bool | `True` | Enable TIR simplification |
| `tir.disable_storage_rewrite` | bool | `False` | Disable storage rewrite |
| `tir.disable_vectorize` | bool | `False` | Disable vectorization |
| `tir.use_async_copy` | bool | `True` | Enable async copy |
| `tir.enable_debug` | bool | `False` | Enable debug info |
| `tir.merge_static_smem` | bool | `True` | Merge static SMEM |
| `tir.add_lower_pass` | list | `None` | Additional lowering passes |
| `tir.noalias` | bool | `True` | Assume no aliasing |
| `tir.detect_global_barrier` | bool | `False` | Enable global barrier detection |

### Dumping IR Between Passes

To inspect the IR at every stage of compilation:

```python
pass_configs = {
    "tl.enable_dump_ir": True,
    "tl.dump_ir_path": "./my_dump",
}

with tilelang.transform.PassContext(
    opt_level=3,
    config=pass_configs,
    instruments=[tvm.ir.instrument.DumpIR(dump_dir="./my_dump")],
):
    kernel = tilelang.compile(func, pass_configs=pass_configs)
```

Each pass writes its output IR to a separate file in the dump directory, allowing step-by-step inspection of the compilation process.
