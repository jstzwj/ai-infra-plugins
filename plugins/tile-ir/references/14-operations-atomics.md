# Operations: Atomics

> **Warning:**
> Atomic operations are limited in Tile IR as of early-access and will be updated in coming releases in accordance with the incoming memory model and memory operation updates.

Tile IR provides two atomic operations for performing thread-safe modifications to global memory:

- `cuda_tile.atomic_cas_tko` -- Atomic compare-and-swap
- `cuda_tile.atomic_rmw_tko` -- Atomic read-modify-write

Both operations are token-ordered and operate element-wise on tiles of pointers, performing one atomic transaction per element.

---

## `cuda_tile.atomic_cas_tko`

Atomic compare-and-swap on global memory.

```
cuda_tile.atomic_cas_tko %memory_ordering_semantics %memory_scope %pointers %cmp %val %mask %token
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| memory_ordering_semantics | `MemoryOrderingSemantics` | The memory ordering semantics for the atomic operation. |
| memory_scope | `MemoryScope` | The memory scope for the atomic operation. |
| pointers | `tile<ptr<E>>` | The pointers to the memory locations to perform the atomic compare-and-swap operation on. Each pointer identifies a single memory location. |
| cmp | `tile<E>` | The values to compare against. Each element is compared to the corresponding memory location. |
| val | `tile<E>` | The values to swap in. Each element is conditionally written to the corresponding memory location. |
| mask | `tile<i1>` | Optional mask for the atomic operation. Controls which elements participate in the operation. |
| token | `token` | Optional token for the atomic operation. Establishes ordering with prior memory operations. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<E>` | The result of the atomic operation. Contains the original value stored at each memory location before the atomic operation. |
| result_token | `token` | The result token of the atomic operation. Can be consumed by subsequent token-ordered operations. |

**Description:**

The `atomic_cas_tko` operation performs element-wise, atomic compare-and-swaps at the specified global memory pointers. The data in memory is compared to `cmp` and the data written to memory is specified by `val`. The operation returns the original value that was stored in memory before the atomic operation was performed.

The shape (and the element type) of `pointers`, `cmp`, `val` and `result` must match. The `atomic_cas` operation performs the following steps for every `(pointer, cmp, val)` tuple in one atomic transaction (one atomic transaction per tuple):

```
atomic() {
    x = *pointer
    if x == cmp {
        *pointer = val
    }
    return x
}
```

This means:
1. The current value at `*pointer` is read into `x`.
2. If `x` equals `cmp`, then `val` is written to `*pointer`.
3. The original value `x` (before any write) is returned as the result.

If `x != cmp`, the memory location is not modified, but the current value `x` is still returned. This allows the caller to detect whether the swap succeeded and, if not, what the actual value was.

**Mask Behavior:**

An optional parameter, `mask`, allows specifying which elements participate in the atomic operation. A `false` value at position `i` masks out the corresponding element in `pointers`, excluding it from the operation. The returned value for a masked element at position `i` is `cmp[i]`. If no mask is provided, all elements are included in the computation by default. The shape of `mask` must match that of `pointers`, `cmp`, and `val`.

**Token Ordering:**

A token-ordered atomic compare-and-swap is not constrained by program order. The compiler may reorder it (i.e. place them earlier or later in program order) unless constrained by tokens.

**Supported Data Types:**

| Type Category | Supported Types |
|---|---|
| Integer | `i32`, `i64` |
| Floating-point | `f32`, `f64` |

**Floating-point Bitwise Comparison:**

For floating-point types (`f32`, `f64`), the comparison uses bitwise equality rather than IEEE-754 semantics. This means:

- Different NaN bit patterns are treated as distinct values. `NaN != NaN` in IEEE-754, but in atomic_cas, two values with the same NaN bit pattern are considered equal.
- `+0.0` and `-0.0` are considered different if their bit representations differ (they have different sign bits in IEEE-754).
- This behavior is consistent with GPU hardware atomic compare-and-swap implementations.

**Memory Ordering Semantics:**

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For more information, refer to the memory model section of the specification.

| Ordering | Description |
|----------|-------------|
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `acquire` | There may be concurrent accesses to the location. If this acquire observes a release operation, then happens-before is established. |
| `release` | There may be concurrent access to the location. If this release is observed with an acquire operation, then happens-before is established. |
| `acq_rel` | There may be concurrent accesses to the location. This has the effect of both a release and acquire operation. |

Note: The `weak` variant is not supported by this operation.

**Memory Scope:**

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

| Scope | Description |
|-------|-------------|
| `tl_blk` | There may be concurrent accesses from within the same tile block. |
| `device` | There may be concurrent accesses from within the same device (i.e., GPU). |
| `sys` | There may be concurrent accesses from anywhere within the system (i.e., all devices). |

**Constraints:**

- `cmp`, `val` and `result` must have the same shape and element type (tile).
- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

**Examples:**

Example 1 -- Atomic CAS without input token:

```
%ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
%cmp = constant <i32: [0, 1, 2, 3, 4, 5, 6, 7]> : tile<8xi32>
%val = constant <i32: [7, 6, 5, 4, 3, 2, 1, 0]> : tile<8xi32>

// Atomic CAS without input token.
// For each element: if *ptr == cmp, then *ptr = val
// Returns the original values at each memory location.
%0, %token = atomic_cas_tko relaxed device %ptrs, %cmp, %val :
  tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token
```

Example 2 -- Atomic CAS with mask:

```
%ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
%cmp = constant <i32: [0, 1, 2, 3, 4, 5, 6, 7]> : tile<8xi32>
%val = constant <i32: [7, 6, 5, 4, 3, 2, 1, 0]> : tile<8xi32>
%mask = constant <i1: [0, 1, 0, 1, 0, 1, 0, 1]> : tile<8xi1>

// Atomic CAS without input token but with mask.
// Only elements at positions 1, 3, 5, 7 participate.
// Masked-out elements return their corresponding cmp value.
%1, %token1 = atomic_cas_tko relaxed device %ptrs, %cmp, %val, %mask :
  tile<8xptr<i32>>, tile<8xi32>, tile<8xi1> -> tile<8xi32>, token
```

Example 3 -- Atomic CAS with input token:

```
%ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
%cmp = constant <i32: [0, 1, 2, 3, 4, 5, 6, 7]> : tile<8xi32>
%val = constant <i32: [7, 6, 5, 4, 3, 2, 1, 0]> : tile<8xi32>

// Atomic CAS with input token - ordered with respect to prior operations.
%token2 = make_token : token
%2, %token3 = atomic_cas_tko relaxed device %ptrs, %cmp, %val token=%token2 :
  tile<8xptr<i32>>, tile<8xi32> -> tile<8xi32>, token
```

---

## `cuda_tile.atomic_rmw_tko`

Atomic read-modify-write on global memory.

```
cuda_tile.atomic_rmw_tko %memory_ordering_semantics %memory_scope %pointers %mode %arg %mask %token
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| memory_ordering_semantics | `MemoryOrderingSemantics` | The memory ordering semantics for the atomic operation. |
| memory_scope | `MemoryScope` | The memory scope for the atomic operation. |
| pointers | `tile<ptr<E>>` | The pointer tile to perform the atomic operation on. Each pointer identifies a memory location to modify. |
| mode | `AtomicRMWMode` | The atomic operation mode (e.g., `add`, `addf`, `max`, `min`, etc.). Default value: `add`. |
| arg | `tile<E>` | The value tile to use in the atomic operation. Each element is used as the argument to the read-modify-write at the corresponding pointer. |
| mask | `tile<i1>` | Optional mask for the atomic operation. Controls which elements participate. |
| token | `token` | Optional token for the atomic operation. Establishes ordering with prior memory operations. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<E>` | The result of the atomic operation. Contains the original value stored at each location before the atomic update. |
| result_token | `token` | The result token of the atomic operation. Can be consumed by subsequent token-ordered operations. |

**Description:**

The `atomic_rmw_tko` operation performs element-wise, atomic read-modify-write operations at the global memory locations specified by pointers. The values written to memory are determined by `mode` and `arg`. The operation returns the original value stored at each location before the atomic update.

The shapes of `pointers`, `arg`, and `result` must match. The element type of the pointer type must match the element types of both `arg` and `result`. Each `(pointer, arg)` pair is processed in a single atomic transaction:

```
atomic {
    x = *pointer
    y = mode(x, arg)
    *pointer = y
    return x
}
```

This means:
1. The current value at `*pointer` is read into `x`.
2. The `mode` function is applied to `x` and `arg`, producing `y`.
3. `y` is written back to `*pointer`.
4. The original value `x` (before the write) is returned as the result.

**Mask Behavior:**

An optional parameter, `mask`, specifies which elements participate in the atomic operation. A `False` value at position `i` excludes the corresponding element in `pointers` from the operation. The value returned for a masked-out element is implementation-defined. The shape of `mask` must match the shape of `pointers`.

**Floating-point Addition Rounding:**

The `atomic_addf` operation is defined to round to the nearest even value.

> **Note:** The current implementation of the compiler flushes denormals to zero. This behavior will be fixed in a future version of the compiler and users should not rely on it.

**Token Ordering:**

Token-ordered atomic read-modify-write operations are not constrained by program order. The compiler may reorder them (i.e., move them earlier or later in the program) unless further constrained by tokens.

**RMW Modes:**

The `mode` attribute specifies the mode of the atomic read-modify-write operation. The mode attribute has a default value of `add`.

| Mode | Description | Formula |
|------|-------------|---------|
| `and` | Perform bitwise AND as the modification operation. | `y = x & arg` |
| `or` | Perform bitwise OR as the modification operation. | `y = x \| arg` |
| `xor` | Perform bitwise XOR as the modification operation. | `y = x ^ arg` |
| `add` | Perform integer addition as the modification operation. | `y = x + arg` |
| `addf` | Perform floating-point addition as the modification operation. | `y = x + arg` (floating-point) |
| `max` | Perform signed maximum as the modification operation. | `y = max(x, arg)` |
| `min` | Perform signed minimum as the modification operation. | `y = min(x, arg)` |
| `umax` | Perform unsigned maximum as the modification operation. | `y = umax(x, arg)` |
| `umin` | Perform unsigned minimum as the modification operation. | `y = umin(x, arg)` |
| `xchg` | Perform exchange as the modification operation. | `y = arg` |

The `U` prefix in `umax` and `umin` distinguishes these from their signed counterparts (`max` and `min`) by interpreting the comparison as unsigned. For example, with 32-bit integers, `max` interprets `0xFFFFFFFF` as -1, while `umax` interprets it as 4294967295.

**Supported Data Types by Mode:**

| Mode | Supported Types |
|------|----------------|
| `add`, `and`, `max`, `min`, `or`, `umax`, `umin`, `xor` | `i32`, `i64` |
| `addf` | `f16`, `f32`, `f64` |
| `xchg` | `i32`, `i64`, `f32`, `f64` |

**Memory Ordering Semantics:**

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For more information, refer to the memory model section of the specification.

| Ordering | Description |
|----------|-------------|
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `acquire` | There may be concurrent accesses to the location. If this acquire observes a release operation, then happens-before is established. |
| `release` | There may be concurrent access to the location. If this release is observed with an acquire operation, then happens-before is established. |
| `acq_rel` | There may be concurrent accesses to the location. This has the effect of both a release and acquire operation. |

Note: The `weak` variant is not supported by this operation.

**Memory Scope:**

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

| Scope | Description |
|-------|-------------|
| `tl_blk` | There may be concurrent accesses from within the same tile block. |
| `device` | There may be concurrent accesses from within the same device (i.e., GPU). |
| `sys` | There may be concurrent accesses from anywhere within the system (i.e., all devices). |

**Constraints:**

- `arg` and `result` must have the same shape and element type (tile).
- Operation must encode variadic operand segment sizes in attributes.
- Operation must infer result types from operands and attributes.

**Examples:**

Example 1 -- Atomic floating-point add without token:

```
// Reshape the input pointer tile to have a 1d shape
%ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
// Broadcast the reshaped tile to a tile with 8 rows, effectively replicating the pointer 8 times
%ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
// Create a tile of offsets [0, 1, 2, ..., 7] to index into memory
%offsets = iota : tile<8xi32>
// Add the offsets to each pointer in the vector to create 8 unique pointers
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
%vals = constant <f32: [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]> : tile<8xf32>

// Perform atomic addf operations on the memory locations pointed by %ptrs
// without requiring an input token. Returns the original values and a result token
%0, %res_token0 = atomic_rmw_tko relaxed device %ptrs, addf, %vals :
    tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
```

Example 2 -- Atomic floating-point add with token:

```
// Perform atomic add operations again, this time using the explicit input token
%token = make_token : token
%1, %res_token1 = atomic_rmw_tko relaxed device %ptrs, addf, %vals, token = %token :
    tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
```

Example 3 -- Atomic integer add with mask:

```
%ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<8xptr<i32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<i32>>, tile<8xi32> -> tile<8xptr<i32>>
%vals = constant <i32: [10, 20, 30, 40, 50, 60, 70, 80]> : tile<8xi32>
%mask = constant <i1: [1, 0, 1, 0, 1, 0, 1, 0]> : tile<8xi1>

// Only elements at even positions participate in the atomic add
// Masked-out elements return implementation-defined values
%0, %token0 = atomic_rmw_tko relaxed device %ptrs, add, %vals, %mask :
    tile<8xptr<i32>>, tile<8xi32>, tile<8xi1> -> tile<8xi32>, token
```

Example 4 -- Atomic exchange (xchg):

```
%ptr_1x = reshape %ptr : tile<ptr<i64>> -> tile<1xptr<i64>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i64>> -> tile<4xptr<i64>>
%offsets = iota : tile<4xi32>
%ptrs = offset %ptr_vec, %offsets : tile<4xptr<i64>>, tile<4xi32> -> tile<4xptr<i64>>
%vals = constant <i64: [100, 200, 300, 400]> : tile<4xi64>

// Atomically swap: write %vals to memory, return previous values
%old_vals, %token = atomic_rmw_tko relaxed device %ptrs, xchg, %vals :
    tile<4xptr<i64>>, tile<4xi64> -> tile<4xi64>, token
```

Example 5 -- Atomic unsigned minimum:

```
%ptr_1x = reshape %ptr : tile<ptr<i32>> -> tile<1xptr<i32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<i32>> -> tile<4xptr<i32>>
%offsets = iota : tile<4xi32>
%ptrs = offset %ptr_vec, %offsets : tile<4xptr<i32>>, tile<4xi32> -> tile<4xptr<i32>>
%vals = constant <i32: [5, 10, 15, 20]> : tile<4xi32>

// Atomically compute unsigned min of current memory value and %vals
// Treats values as unsigned: 0xFFFFFFFF is 4294967295, not -1
%old_vals, %token = atomic_rmw_tko relaxed device %ptrs, umin, %vals :
    tile<4xptr<i32>>, tile<4xi32> -> tile<4xi32>, token
```

---

## Comparison: atomic_cas_tko vs atomic_rmw_tko

| Property | `atomic_cas_tko` | `atomic_rmw_tko` |
|---|---|---|
| **Purpose** | Conditional swap: write only if current value matches | Unconditional read-modify-write |
| **Algorithm** | `if *ptr == cmp then *ptr = val` | `*ptr = mode(*ptr, arg)` |
| **Returns** | Original value at each location | Original value at each location |
| **Masked return value** | `cmp[i]` for masked elements | Implementation-defined for masked elements |
| **Integer types** | `i32`, `i64` | `i32`, `i64` |
| **Float types** | `f32`, `f64` | `f16`, `f32`, `f64` (mode-dependent) |
| **Float comparison** | Bitwise (not IEEE-754) | N/A (no comparison) |
| **Modes available** | None (fixed compare-and-swap) | `and`, `or`, `xor`, `add`, `addf`, `max`, `min`, `umax`, `umin`, `xchg` |
| **Memory orderings** | `relaxed`, `acquire`, `release`, `acq_rel` | `relaxed`, `acquire`, `release`, `acq_rel` |
| **Token support** | Optional input, always output | Optional input, always output |
| **Use case** | Lock-free algorithms, conditional updates | Counters, accumulators, reductions |

**When to use which:**

- Use `atomic_cas_tko` when you need to conditionally update a value only if it currently matches an expected value. This is the foundation for lock-free data structures, spinlocks, and compare-and-swap loops.
- Use `atomic_rmw_tko` when you need to unconditionally modify a value using a well-defined operation (add, min, max, etc.) and retrieve the old value. This is more efficient than CAS for common patterns like counters and accumulators.
- Both operations return the original value before the modification, enabling the caller to detect what happened and build higher-level synchronization primitives.

---

## Memory Ordering Semantics Details

Both atomic operations support the same set of memory ordering semantics, which control the visibility of the atomic operation's effects to other concurrent threads or tile blocks. These semantics follow the same conventions as the CUDA/PTX memory model.

### Relaxed Ordering

`relaxed` ordering provides no synchronization guarantees between threads. The atomic operation itself is indivisible (no torn reads or writes), but it does not establish any happens-before relationship with other threads. This is appropriate for simple counters or statistics where the order of updates does not matter.

### Acquire Ordering

`acquire` ordering guarantees that all memory operations (both atomic and non-atomic) that appear after the acquire operation in program order will not be reordered before it. If this acquire observes a release store from another thread, a happens-before relationship is established, making all memory writes from the releasing thread visible to the acquiring thread.

### Release Ordering

`release` ordering guarantees that all memory operations (both atomic and non-atomic) that appear before the release operation in program order will not be reordered after it. When another thread observes the released value with an acquire operation, a happens-before relationship is established.

### Acquire-Release Ordering

`acq_rel` ordering combines the guarantees of both `acquire` and `release`. It ensures that no prior memory operations are reordered after it and no subsequent memory operations are reordered before it. This is useful for operations that both read and write to memory and need to synchronize in both directions.

### Choosing the Right Ordering

| Scenario | Recommended Ordering |
|---|---|
| Simple counter increment, no cross-thread visibility needed | `relaxed` |
| Producer signals completion to consumer | Producer: `release`, Consumer: `acquire` |
| Bidirectional synchronization (e.g., mutex lock/unlock) | `acq_rel` |
| No concurrent access guaranteed | `relaxed` (no `weak` available for atomics) |

Note: The `weak` ordering variant is not supported by either atomic operation. The `weak` ordering is available only on `load_ptr_tko` and `store_ptr_tko`.

---

## Common Patterns

### Atomic Counter (per-element)

```
// Each tile block atomically adds its contribution to a shared counter
%ptr = ... : tile<ptr<i64>>
%one = constant <i64: 1> : tile<i64>

%old, %token = atomic_rmw_tko relaxed device %ptr, add, %one :
    tile<ptr<i64>>, tile<i64> -> tile<i64>, token
```

### Compare-and-Swap Loop (spinlock pattern)

```
// Attempt to atomically set a flag from 0 to 1
%expected = constant <i32: 0> : tile<i32>
%desired = constant <i32: 1> : tile<i32>

%old, %token = atomic_cas_tko acq_rel device %lock_ptr, %expected, %desired :
    tile<ptr<i32>>, tile<i32>, tile<i32> -> tile<i32>, token

// If %old == 0, the lock was acquired (swap succeeded)
// If %old == 1, another thread holds the lock (swap failed, retry needed)
```

### Reduction with Atomic Add

```
// Accumulate partial results from each tile block into a shared buffer
%ptrs = ... : tile<8xptr<f32>>
%partials = ... : tile<8xf32>

%old_vals, %token = atomic_rmw_tko relaxed device %ptrs, addf, %partials :
    tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
```
