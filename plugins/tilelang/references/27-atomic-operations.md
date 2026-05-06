# TileLang Atomic Operations Reference

Atomic operations in TileLang provide thread-safe mechanisms for concurrent memory access across GPU threads. These operations are essential for reduction patterns, output accumulation, and synchronization in parallel GPU kernels.

## Table of Contents

1. [Overview](#overview)
2. [Atomic Addition](#atomic-addition)
3. [Atomic Maximum and Minimum](#atomic-maximum-and-minimum)
4. [Atomic Load and Store](#atomic-load-and-store)
5. [Wide Atomic Operations](#wide-atomic-operations)
6. [Memory Ordering](#memory-ordering)
7. [Usage Patterns](#usage-patterns)
8. [Performance Considerations](#performance-considerations)

---

## Overview

TileLang exposes atomic operations through the `tilelang.language.atomic` module. These operations map directly to hardware atomic instructions on NVIDIA and AMD GPUs.

All atomic operations in TileLang fall into two categories:

| Category | Description | Mechanism |
|----------|-------------|-----------|
| **Scalar/Element-wise** | Operate on single memory addresses | `tl.atomic_*_elem_op` intrinsics |
| **Tile/Region-based** | Operate on buffer regions (tensors) | `tl.tileop.atomic*` tile operations |

The module is imported as:

```python
from tilelang.language.atomic import (
    atomic_add,
    atomic_addx2,
    atomic_addx4,
    atomic_max,
    atomic_min,
    atomic_load,
    atomic_store,
)
```

Or through the customize module:

```python
from tilelang.language.customize import (
    atomic_add,
    atomic_max,
    atomic_min,
    atomic_addx2,
    atomic_addx4,
    atomic_load,
    atomic_store,
)
```

---

## Atomic Addition

### atomic_add

```python
def atomic_add(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False,
    use_tma: bool = False,
) -> PrimExpr
```

Atomically adds `value` into `dst`. This is the most commonly used atomic operation in GPU programming.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dst` | `Buffer` | required | Destination buffer/address |
| `value` | `PrimExpr` | required | Value to add atomically |
| `memory_order` | `str or None` | `None` | Memory ordering constraint |
| `return_prev` | `bool` | `False` | Whether to return the previous value |
| `use_tma` | `bool` | `False` | Use TMA (cp.reduce) for SM90+ |

**Dispatch Logic:**

1. If neither `dst` nor `value` exposes extents (scalar path):
   - Uses `tl.atomic_add_elem_op` intrinsic for single-element atomics
   - Uses `tl.atomic_add_ret_elem_op` when `return_prev=True`
2. If at least one argument has extents (tile path):
   - Converts arguments to buffer regions
   - Uses `tl.tileop.atomicadd` tile operation

**Scalar Path Examples:**

```python
import tilelang as tl
import tilelang.language as T

@T.prim_func
def atomic_counter(data: T.Buffer((1024,), "float32"), result: T.Buffer((1,), "float32")):
    for i in T.thread_binding(1024, "threadIdx.x"):
        # Each thread atomically adds its value to the result
        atomic_add(result, data[i])
```

**Tile/Region Path Examples:**

```python
@T.prim_func
def atomic_add_tensor(
    src: T.Buffer((128, 64), "float32"),
    dst: T.Buffer((128, 64), "float32"),
):
    # Atomically add entire tensor regions
    atomic_add(dst, src)
```

**Return Previous Value:**

```python
@T.prim_func
def atomic_add_with_prev(counter: T.Buffer((1,), "int32")):
    # Returns the value before the atomic add
    old_value = atomic_add(counter, 1, return_prev=True)
    # old_value contains the counter value before incrementing
```

**TMA-accelerated Atomic Add (SM90+):**

```python
@T.prim_func
def atomic_add_tma(
    src: T.Buffer((128, 128), "float32"),
    dst: T.Buffer((128, 128), "float32"),
):
    # Uses cp.reduce (TMA) for hardware-accelerated atomic add on Hopper
    atomic_add(dst, src, use_tma=True)
```

**Gradient Accumulation Pattern:**

```python
@T.prim_func
def accumulate_gradients(
    local_grad: T.Buffer((256,), "float32"),
    global_grad: T.Buffer((256,), "float32"),
):
    for i in T.thread_binding(256, "threadIdx.x"):
        atomic_add(global_grad, local_grad[i])
```

### Atomic Operations on Global Memory

Atomic addition to global memory is the default behavior when the destination buffer resides in global memory. This is the most common use case for reduction accumulation and output merging.

```python
@T.prim_func
def global_atomic_reduce(
    data: T.Buffer((1024, 256), "float32"),
    output: T.Buffer((256,), "float32"),
):
    with T.Kernel(256, 1024) as (bx, by):
        # Each block accumulates into shared memory first
        shared_sum = T.alloc_shared((1,), "float32")
        shared_sum[0] = 0.0
        T.sync_threads()
        atomic_add(shared_sum, data[by, bx])
        T.sync_threads()
        # Then atomically add to global output
        if by == 0:
            atomic_add(output, shared_sum[0])
```

### Atomic Operations on Shared Memory

Atomic operations on shared memory use the same API but with shared memory buffers. This is useful for intra-block reductions.

```python
@T.prim_func
def shared_mem_atomic(data: T.Buffer((128,), "float32"), result: T.Buffer((1,), "float32")):
    with T.Kernel(128) as bx:
        shared = T.alloc_shared((1,), "float32")
        shared[0] = 0.0
        T.sync_threads()
        # Atomic add in shared memory
        atomic_add(shared, data[bx])
        T.sync_threads()
        if bx == 0:
            result[0] = shared[0]
```

**Note:** Shared memory atomic performance depends on the GPU architecture. On older architectures (pre-Volta), shared memory atomics may be emulated and slower than global memory atomics.

---

## Atomic Maximum and Minimum

### atomic_max

```python
def atomic_max(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False,
) -> PrimExpr
```

Atomically updates the value at `dst` to the maximum of its current value and `value`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dst` | `Buffer` | required | Destination buffer |
| `value` | `PrimExpr` | required | Value to compare against |
| `memory_order` | `str or None` | `None` | Memory ordering constraint |
| `return_prev` | `bool` | `False` | Return previous value (scalar path only) |

**Dispatch Logic:**

1. Scalar path (no extents): Uses `tl.atomic_max_elem_op` intrinsic
2. Tile path (has extents): Uses `tl.tileop.atomicmax` tile operation

**Example -- Global Maximum:**

```python
@T.prim_func
def find_global_max(
    data: T.Buffer((4096,), "float32"),
    result: T.Buffer((1,), "float32"),
):
    result[0] = float("-inf")
    for i in T.thread_binding(4096, "threadIdx.x"):
        atomic_max(result, data[i])
```

**Example -- Per-row Maximum:**

```python
@T.prim_func
def row_max(
    matrix: T.Buffer((128, 64), "float32"),
    max_vals: T.Buffer((128,), "float32"),
):
    with T.Kernel(128) as bx:
        max_vals[bx] = float("-inf")
        for j in T.serial(64):
            atomic_max(max_vals, matrix[bx, j])
```

### atomic_min

```python
def atomic_min(
    dst: Buffer,
    value: PrimExpr,
    memory_order: str | None = None,
    return_prev: bool = False,
) -> PrimExpr
```

Atomically updates the value at `dst` to the minimum of its current value and `value`.

**Example -- Tracking Minimum Value:**

```python
@T.prim_func
def find_min_loss(
    losses: T.Buffer((1000,), "float32"),
    min_loss: T.Buffer((1,), "float32"),
):
    min_loss[0] = float("inf")
    for i in T.thread_binding(1000, "threadIdx.x"):
        atomic_min(min_loss, losses[i])
```

**Example -- Tensor-to-Tensor Atomic Min:**

```python
@T.prim_func
def tensor_atomic_min(
    src: T.Buffer((64, 64), "float32"),
    dst: T.Buffer((64, 64), "float32"),
):
    # Element-wise atomic min across entire tensors
    atomic_min(dst, src)
```

---

## Atomic Load and Store

### atomic_load

```python
def atomic_load(
    src: Buffer,
    memory_order: str = "seq_cst",
) -> PrimExpr
```

Loads a value from `src` using the specified atomic memory ordering. Always returns the loaded value.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src` | `Buffer` | required | Source buffer to load from |
| `memory_order` | `str` | `"seq_cst"` | Memory ordering for the load |

**Returns:** `PrimExpr` -- The loaded value.

**Example -- Spin-Wait Pattern:**

```python
@T.prim_func
def consumer_wait(
    flag: T.Buffer((1,), "int32"),
    data: T.Buffer((1024,), "float32"),
    result: T.Buffer((1024,), "float32"),
):
    # Spin until the producer signals completion
    while atomic_load(flag, memory_order="acquire") == 0:
        pass
    # Now safely read the data
    for i in T.serial(1024):
        result[i] = data[i]
```

**Example -- Relaxed Load for Counters:**

```python
@T.prim_func
def read_counter(
    counter: T.Buffer((1,), "int64"),
    snapshot: T.Buffer((1,), "int64"),
):
    # Relaxed load -- no ordering guarantees needed
    snapshot[0] = atomic_load(counter, memory_order="relaxed")
```

### atomic_store

```python
def atomic_store(
    dst: Buffer,
    src: PrimExpr,
    memory_order: str = "seq_cst",
) -> PrimExpr
```

Stores `src` into `dst` atomically with the specified memory ordering.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dst` | `Buffer` | required | Destination buffer |
| `src` | `PrimExpr` | required | Value to store |
| `memory_order` | `str` | `"seq_cst"` | Memory ordering for the store |

**Returns:** `PrimExpr` -- A handle to the atomic store operation.

**Example -- Producer Signal:**

```python
@T.prim_func
def producer_signal(
    data: T.Buffer((1024,), "float32"),
    flag: T.Buffer((1,), "int32"),
):
    # Write data first
    for i in T.serial(1024):
        data[i] = float(i)
    # Release store ensures all writes are visible before flag
    atomic_store(flag, 1, memory_order="release")
```

---

## Wide Atomic Operations

### atomic_addx2

```python
def atomic_addx2(
    dst: Buffer,
    value: PrimExpr,
    return_prev: bool = False,
) -> PrimExpr
```

Performs a double-width atomic addition, operating on two elements simultaneously. This maps to hardware instructions like `atomicAdd.half.2` on NVIDIA GPUs (requires compute capability >= 7.0 for FP16, >= 8.0 for BF16).

**Use Cases:**
- FP16/BF16 gradient accumulation (2 elements at once)
- Vectorized atomic operations for improved throughput

**Example:**

```python
@T.prim_func
def fp16_atomic_addx2(
    grads: T.Buffer((256,), "float16"),
    accum: T.Buffer((256,), "float16"),
):
    for i in T.thread_binding(128, "threadIdx.x"):
        # Process 2 FP16 elements per atomic operation
        atomic_addx2(accum[i * 2], grads[i * 2])
```

### atomic_addx4

```python
def atomic_addx4(
    dst: Buffer,
    value: PrimExpr,
    return_prev: bool = False,
) -> PrimExpr
```

Performs a quad-width atomic addition, operating on four float32 elements simultaneously. This maps to `atomicAdd.float4` on NVIDIA Blackwell GPUs (requires compute capability >= 9.0).

**Example:**

```python
@T.prim_func
def float4_atomic_add(
    grads: T.Buffer((1024,), "float32"),
    accum: T.Buffer((1024,), "float32"),
):
    for i in T.thread_binding(256, "threadIdx.x"):
        # Process 4 float32 elements per atomic operation
        atomic_addx4(accum[i * 4], grads[i * 4])
```

---

## Memory Ordering

TileLang supports six memory ordering levels, mapped from string names to numeric IDs:

| Name | ID | Description |
|------|-----|-------------|
| `"relaxed"` | 0 | No ordering guarantees |
| `"consume"` | 1 | Data-dependent ordering (rarely used on GPUs) |
| `"acquire"` | 2 | Subsequent reads/writes cannot be reordered before this |
| `"release"` | 3 | Previous reads/writes cannot be reordered after this |
| `"acq_rel"` | 4 | Both acquire and release semantics |
| `"seq_cst"` | 5 | Sequentially consistent (total ordering) |

**When to Use Each Ordering:**

| Pattern | Recommended Ordering |
|---------|---------------------|
| Counter increments | `"relaxed"` |
| Producer-consumer flag (producer writes) | `"release"` |
| Producer-consumer flag (consumer reads) | `"acquire"` |
| General synchronization | `"seq_cst"` |
| Performance-critical accumulations | `"relaxed"` |

**Example -- Memory Ordering in Producer-Consumer:**

```python
@T.prim_func
def producer_consumer(
    data: T.Buffer((1024,), "float32"),
    flag: T.Buffer((1,), "int32"),
):
    with T.Kernel(1) as bx:
        if bx == 0:
            # Producer: write data, then set flag
            for i in T.serial(1024):
                data[i] = float(i)
            atomic_store(flag, 1, memory_order="release")
        else:
            # Consumer: wait for flag, then read data
            while atomic_load(flag, memory_order="acquire") == 0:
                pass
            # Data is now visible
```

---

## Usage Patterns

### Pattern 1: Parallel Sum Reduction

The most common atomic pattern is parallel reduction using `atomic_add`:

```python
@T.prim_func
def parallel_sum(
    data: T.Buffer((4096,), "float32"),
    result: T.Buffer((1,), "float32"),
):
    result[0] = 0.0
    for i in T.thread_binding(4096, "threadIdx.x"):
        atomic_add(result, data[i])
```

**Performance Note:** For large reductions, a two-level approach (shared memory + global atomics) is significantly faster:

```python
@T.prim_func
def optimized_parallel_sum(
    data: T.Buffer((4096,), "float32"),
    result: T.Buffer((1,), "float32"),
):
    with T.Kernel(64) as bx:
        shared = T.alloc_shared((1,), "float32")
        shared[0] = 0.0
        T.sync_threads()
        # Each thread processes multiple elements
        for i in T.serial(64):
            atomic_add(shared, data[bx * 64 + i])
        T.sync_threads()
        # One thread per block accumulates to global
        if bx == 0:
            atomic_add(result, shared[0])
```

### Pattern 2: Atomic Operations for Output Accumulation

When multiple thread blocks contribute to the same output tensor, atomic operations ensure correctness:

```python
@T.prim_func
def scatter_add(
    src: T.Buffer((1024,), "float32"),
    indices: T.Buffer((1024,), "int32"),
    output: T.Buffer((256,), "float32"),
):
    with T.Kernel(1024) as bx:
        idx = indices[bx]
        atomic_add(output, src[bx])  # Simplified; real code would index properly
```

### Pattern 3: Atomic Operations in Multi-Head Attention

FlashAttention-style kernels use atomics for accumulation when the output tiles overlap:

```python
@T.prim_func
def attention_accumulate(
    qk_scores: T.Buffer((128, 128), "float32"),
    values: T.Buffer((128, 64), "float32"),
    output: T.Buffer((128, 64), "float32"),
):
    # Each block computes a partial output tile
    # Multiple blocks may contribute to the same output tile
    with T.Kernel(128, 128) as (bx, by):
        # Compute partial attention output
        partial = qk_scores[bx, by] * values[by, 0]  # Simplified
        # Atomically accumulate into global output
        atomic_add(output, partial)
```

### Pattern 4: Thread Safety with Atomics

When multiple threads write to the same location, atomics prevent data races:

```python
@T.prim_func
def thread_safe_histogram(
    data: T.Buffer((8192,), "int32"),
    bins: T.Buffer((256,), "int32"),
):
    # Initialize bins to zero
    for i in T.serial(256):
        bins[i] = 0
    # Each thread processes one element
    for i in T.thread_binding(8192, "threadIdx.x"):
        bin_idx = data[i] % 256
        atomic_add(bins, 1)  # Simplified; real code would index by bin_idx
```

### Pattern 5: Lock-Free Patterns Using Atomics

Atomic operations enable lock-free data structures and algorithms:

**Lock-Free Counter:**

```python
@T.prim_func
def lock_free_counter(
    increment: T.Buffer((1,), "int32"),
    counter: T.Buffer((1,), "int32"),
):
    # Atomically increment the counter without any lock
    old_value = atomic_add(counter, increment[0], return_prev=True)
    # old_value is the value before incrementing -- no data race
```

**Lock-Free Slot Allocation:**

```python
@T.prim_func
def allocate_slots(
    num_threads: int,
    slot_counter: T.Buffer((1,), "int32"),
    slots: T.Buffer((256,), "float32"),
    values: T.Buffer((256,), "float32"),
):
    for i in T.thread_binding(num_threads, "threadIdx.x"):
        # Each thread gets a unique slot index
        slot_idx = atomic_add(slot_counter, 1, return_prev=True)
        slots[slot_idx] = values[i]
```

**Compare-and-Swap (via atomic_max/atomic_min):**

```python
@T.prim_func
def find_top_k(
    data: T.Buffer((4096,), "float32"),
    threshold: T.Buffer((1,), "float32"),
):
    # Use atomic_max to track the threshold in a lock-free manner
    for i in T.thread_binding(4096, "threadIdx.x"):
        atomic_max(threshold, data[i])
```

---

## Performance Considerations

### Atomic Contention

Atomic operations serialize when multiple threads write to the same address. The performance impact depends on the number of contending threads:

| Contention Level | Threads | Impact |
|------------------|---------|--------|
| Low | 1-4 | Negligible |
| Medium | 4-32 | Moderate slowdown |
| High | 32+ | Severe serialization |

**Mitigation Strategies:**

1. **Hierarchical Reduction:** Reduce in shared memory first, then use a single atomic per block for global memory.

2. **Warp-Level Reduction:** Use warp shuffle intrinsics (`T.shuffle`) for intra-warp reduction before atomics.

3. **Split-K Pattern:** Partition the reduction dimension across blocks and accumulate partial results with atomics.

### Memory Access Patterns

| Pattern | Performance | Recommendation |
|---------|-------------|---------------|
| Same address, all threads | Poor | Use shared memory reduction first |
| Different addresses per warp | Good | Preferred pattern |
| Random addresses | Variable | Acceptable for scatter operations |
| Coalesced addresses | Best | Align with memory transaction boundaries |

### Hardware Support Matrix

| Operation | FP16 | BF16 | FP32 | INT32 | INT64 |
|-----------|------|------|------|-------|-------|
| `atomic_add` (scalar) | SM70+ | SM80+ | All | All | All |
| `atomic_addx2` | SM70+ | SM80+ | N/A | N/A | N/A |
| `atomic_addx4` | N/A | N/A | SM90+ | N/A | N/A |
| `atomic_max` | SM70+ | SM80+ | SM70+ | All | All |
| `atomic_min` | SM70+ | SM80+ | SM70+ | All | All |
| `atomic_load` | All | All | All | All | All |
| `atomic_store` | All | All | All | All | All |

### Shared Memory Atomic Performance

| Architecture | Shared Memory Atomics | Notes |
|-------------|----------------------|-------|
| Kepler (SM35) | Emulated | Very slow, avoid |
| Maxwell (SM52) | Emulated | Slow, avoid |
| Pascal (SM60) | Native | Moderate performance |
| Volta (SM70) | Native | Good performance |
| Ampere (SM80) | Native | Good performance |
| Hopper (SM90) | Native | Excellent with async |

### TMA Atomic Add (Hopper+)

On SM90+ architectures, the `use_tma=True` flag enables hardware-accelerated atomic addition using TMA (Tensor Memory Accelerator) copy-reduce operations. This provides significantly higher throughput for large tensor accumulations by offloading the atomic operations from the SM cores to the TMA unit.

**Requirements:**
- SM90+ (Hopper or later)
- Source and destination must be in global memory
- Operands must be aligned to 16-byte boundaries
- The TMA path uses `cp.reduce.async.add` PTX instructions

### Best Practices

1. **Minimize Global Atomics:** Use shared memory reductions within a block, then a single atomic per block for global accumulation.

2. **Use Appropriate Memory Ordering:** `"relaxed"` ordering is sufficient for most GPU kernels and provides the best performance. Only use stronger orderings when actual synchronization is required.

3. **Vectorize When Possible:** Use `atomic_addx2` and `atomic_addx4` for wider operations when the data type and architecture support them.

4. **Avoid Bank Conflicts:** When using shared memory atomics, ensure that concurrent atomic operations target different shared memory banks.

5. **Consider Alternative Algorithms:** For reductions, tree-based reductions in shared memory are typically faster than atomics for large thread counts. Reserve atomics for irregular access patterns where tree reductions are impractical.
