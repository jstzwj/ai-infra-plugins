# Operations: Memory

Tile IR contains a set of memory operations which enable loading, storing, and manipulating memory. There are a few families of memory operations in Tile IR:

- **Tile of pointer based** memory operations such as `cuda_tile.load_ptr_tko` and `cuda_tile.store_ptr_tko` which load and store tiles from and to global memory.
- **View based** memory operations such as `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko` which load and store tiles from and to views.
- **Atomic** memory operations such as `cuda_tile.atomic_rmw_tko` and `cuda_tile.atomic_cas_tko` which perform atomic operations on global memory.

Currently all memory operations are token-ordered; the ordering between any pair of memory operations is undefined unless connected by tokens. For more discussion on token-ordered operations see Memory Model.

> **Warning:**
> Reading or writing out of bounds of any allocation is undefined behavior. Examples of out of bounds access are:
> - Pointer memory operations to tiles containing elements outside the allocation, for example offsetting past the end of the allocation.
> - Associating an invalid layout with a base pointer, that describes a striding or shape that overruns the allocation and then indexing into the view.
> - Indexing into a view with indices that are out of bounds.

> **Note:**
> The rules of what constitutes out of bounds is modified when using padded views or masking, see Type System for more details on specific types.

---

## `cuda_tile.join_tokens`

Produce a new token which depends on the input tokens.

```
cuda_tile.join_tokens %tokens
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| tokens | `Variadic<token>` | The input tokens to join. One or more tokens can be provided. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `token` | The joined token. |

**Description:**

The `join_tokens` operation produces a fresh token which depends on all input tokens. Token-ordered operations which consume the new token will then be ordered with respect to all joined tokens.

This operation is essential for implementing fork-join synchronization patterns. When multiple independent memory operations (each with their own token chain) need to be synchronized before a subsequent operation, `join_tokens` merges the dependency information from all input tokens into a single output token.

The join operation establishes a dependency relationship: the output token semantically depends on every input token. Any token-ordered operation that consumes the output token will not execute until all operations that produced the input tokens have completed their memory effects.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must infer result types from operands and attributes.

**Example:**

```
// Create two independent loads, each with their own token chain
%token0 = make_token : token
%result1, %token1 = load_ptr_tko weak %ptrs1 token=%token0 : ...
%result2, %token2 = load_ptr_tko weak %ptrs2 token=%token0 : ...

// Join the tokens so the store waits for both loads
%joined = join_tokens %token1, %token2 : token
%token3 = store_ptr_tko weak %dst, %value token=%joined : ...
```

---

## `cuda_tile.load_ptr_tko`

Load and gather data from global memory using a pointer tile without ordering guarantees.

```
cuda_tile.load_ptr_tko %memory_ordering_semantics %memory_scope %source %mask %paddingValue %token %optimization_hints
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| memory_ordering_semantics | `MemoryOrderingSemantics` | The memory ordering semantics for the load operation. |
| memory_scope | `MemoryScope` | The memory scope for the load operation. |
| source | `tile<ptr<E>>` | The source tile of pointers. Each pointer specifies the memory location from which the corresponding element is gathered. |
| mask | `tile<i1>` | Optional mask for the load operation. Controls which elements are loaded. |
| paddingValue | `tile<E>` | Optional padding value for the load operation. Specifies the value for masked elements. Must have the same shape as the source tile. Supported element types: `i1, i8, i16, i32, i64, f16, bf16, f32, f64, fp8e4m3fn, fp8e5m2, tf32`. |
| token | `token` | Optional token for the load operation. Establishes ordering with prior memory operations. |
| optimization_hints | `OptimizationHints` | Architecture-specific compiler hints. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<E>` | The result of the load operation. Contains the gathered data from global memory. |
| result_token | `token` | The result token of the load operation. Can be used to establish ordering with subsequent memory operations. |

**Description:**

This load operation performs a gather operation by loading a tile of data from global memory into a result tile based on a tile of pointers provided by the source operand.

The source operand is a tile of pointers, which specifies the memory locations from which the data is gathered. The operation loads this data and returns it as the result tile. The source tile shape determines the shape of the result tile, and the pointer element type determines the element type of the result.

**i1 Loading Behavior:**

When loading i1 values, each value is loaded from a full byte in memory. Any nonzero byte is canonicalized to 0x01, and zero bytes become 0x00. This means that the memory representation of i1 values is one byte per element, not one bit per element.

**Mask and Padding:**

Optionally, a mask operand can be provided to control the gathering of elements. If present, only the elements specified by the mask (where the mask value is `1` / `true`) are loaded. The shape of the mask must match the shape of the source.

When mask is present, a `paddingValue` can optionally be provided as well. The `paddingValue` must have the same shape as the source tile and its element type must match the result type. If `paddingValue` is not provided when a mask is present, the value of masked elements is undefined.

**Token Ordering:**

Token-ordered operations are not constrained by program order. The compiler may reorder them (i.e. place them earlier or later in program order) unless further constrained by tokens.

**Memory Ordering Semantics:**

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

| Ordering | Description |
|----------|-------------|
| `weak` | No concurrent accesses to the source/destination location. The compiler can assume no other threads are accessing the same memory. |
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `acquire` | There may be concurrent accesses to the location. If this acquire observes a release operation, then happens-before is established. |

Note: The following variants are not supported by this operation: `release`, `acq_rel`.

**Memory Scope:**

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

| Scope | Description |
|-------|-------------|
| `tl_blk` | There may be concurrent accesses from within the same tile block. |
| `device` | There may be concurrent accesses from within the same device (i.e., GPU). |
| `sys` | There may be concurrent accesses from anywhere within the system (i.e., all devices). |

**Optimization Hints:**

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

| Hint | Description |
|------|-------------|
| `num_cta_in_cga` | Suggest the number of CTAs in a CGA (which must be a power of 2 less than or equal to 16) for `cuda_tile.entry`. |
| `allow_tma` | Suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`. |
| `latency` | Latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`. |

**Constraints:**

- Operation must encode variadic operand segment sizes in attributes.
- Source type is expected to be a pointer type of result type.
- Shape of `mask` must match the shape of `source`.
- Type of `paddingValue` must match the type of `result`.

**Examples:**

Example 1 -- Load with mask and padding, without token:

```
%mask = constant <i1: 1> : tile<i1>
%padding = constant <f32: 0.0> : tile<f32>

// Load without token.
%result0, %res_token0 = load_ptr_tko weak %ptr, %mask, %padding
    : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token
```

Example 2 -- Load with mask, padding, and token:

```
%mask = constant <i1: 1> : tile<i1>
%padding = constant <f32: 0.0> : tile<f32>

// Load with token.
%token0 = make_token : token
%result1, %res_token1 = load_ptr_tko weak %ptr, %mask, %padding token=%token0
    : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token
```

Example 3 -- Load with masked elements using a non-trivial mask:

```
// Create a mask where some elements are disabled
%mask = constant <i1: [1, 1, 1, 0, 0, 0, 1, 1]> : tile<8xi1>
%padding = constant <f32: -1.0> : tile<8xf32>

// Only elements at positions 0, 1, 2, 6, 7 are loaded from memory.
// Elements at positions 3, 4, 5 receive the padding value -1.0.
%result, %token = load_ptr_tko weak %ptrs, %mask, %padding
    : tile<8xptr<f32>>, tile<8xi1>, tile<8xf32> -> tile<8xf32>, token
```

Example 4 -- Load without mask (all elements loaded):

```
// Load all elements from memory using a tile of pointers
%ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>

%result, %res_token = load_ptr_tko weak %ptrs
    : tile<8xptr<f32>> -> tile<8xf32>, token
```

---

## `cuda_tile.make_token`

Create a fresh token with no prior dependencies.

```
cuda_tile.make_token
```

**Parameters:** None.

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `token` | A fresh token with no prior dependencies. |

**Description:**

The `make_token` operation creates a fresh token with no prior dependencies. This token serves as the starting point for establishing memory operation ordering chains. By passing the token to a token-ordered memory operation, the operation's memory effects become ordered with respect to subsequent operations that consume the resulting output token.

The token produced by `make_token` has no dependency on any prior memory operation, meaning it does not constrain the operation that consumes it to wait for any previous memory effects. It is the "root" of a token dependency chain.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- Operation must infer result types from operands and attributes.

**Example:**

```
// Create a fresh token to start an ordering chain
%token = make_token : token

// The load will depend on %token (no prior ordering constraints)
%result, %out_token = load_ptr_tko weak %ptrs token=%token
    : tile<8xptr<f32>> -> tile<8xf32>, token

// The store depends on %out_token, establishing: load completes before store
%final_token = store_ptr_tko weak %dst, %value token=%out_token
    : tile<8xptr<f32>>, tile<8xf32> -> token
```

---

## `cuda_tile.store_ptr_tko`

Store and scatter data from pointer of tile to global memory without ordering guarantees.

```
cuda_tile.store_ptr_tko %memory_ordering_semantics %memory_scope %destination %value %mask %token %optimization_hints
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| memory_ordering_semantics | `MemoryOrderingSemantics` | The memory ordering semantics for the store operation. |
| memory_scope | `MemoryScope` | The optional memory scope for the store operation. |
| destination | `tile<ptr<E>>` | The destination pointer tile. Each pointer indicates a global memory location where the corresponding element will be stored. |
| value | `tile<E>` | The value tile to store. Each element is written to the corresponding pointer in the destination tile. |
| mask | `tile<i1>` | Optional mask for selective storage. Only elements where the mask is `1` / `true` are stored. |
| token | `token` | Optional token for operation ordering. Establishes ordering with prior memory operations. |
| optimization_hints | `OptimizationHints` | Architecture-specific compiler hints. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result_token | `token` | The result token for synchronization. Can be consumed by subsequent token-ordered operations to establish ordering. |

**Description:**

The store operation performs a scatter by storing a tile of data from a tile into global memory. The destination operand is a tile of pointers indicating the global memory locations where data from the value tile will be stored.

Each element of the value tile is stored to the memory location specified by the corresponding pointer in the destination tile. The shapes of the destination and value tiles must match.

**i1 Storing Behavior:**

When storing i1 values, each value occupies a full byte in memory. Any nonzero value is canonicalized to 0x01, and zero values become 0x00. This is consistent with the i1 loading behavior of `load_ptr_tko`.

**Mask:**

Additionally, the operation supports an optional mask operand, which allows selective scattering of elements. If provided, only the elements specified by the mask (where the mask value is `1` / `true`) are stored. The shape of the mask must align with the shape of the value tile.

Elements that are masked out (mask value is `0` / `false`) are not written to memory. The corresponding memory locations are left unchanged.

**Token Ordering:**

Token-ordered store operations are not constrained by program order. The compiler may reorder them (i.e. place them earlier or later in program order) unless further constrained by tokens.

**Memory Ordering Semantics:**

The `memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required. For example, `weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location. For more information, refer to the memory model section of the specification.

| Ordering | Description |
|----------|-------------|
| `weak` | No concurrent accesses to the source/destination location. The compiler can assume no other threads are accessing the same memory. |
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `release` | There may be concurrent access to the location. If this release is observed with an acquire operation, then happens-before is established. |

Note: The following variants are not supported by this operation: `acquire`, `acq_rel`.

**Memory Scope:**

The `memory_scope` attribute specifies a communication scope for memory operations. When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other threads which are participating in the communication, or data races may occur.

| Scope | Description |
|-------|-------------|
| `tl_blk` | There may be concurrent accesses from within the same tile block. |
| `device` | There may be concurrent accesses from within the same device (i.e., GPU). |
| `sys` | There may be concurrent accesses from anywhere within the system (i.e., all devices). |

**Optimization Hints:**

The `optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries. The hints are specified for each architecture (e.g., `sm_100`, `sm_120`) and for each architecture the user can specify specific hints for each operation.

| Hint | Description |
|------|-------------|
| `num_cta_in_cga` | Suggest the number of CTAs in a CGA (which must be a power of 2 less than or equal to 16) for `cuda_tile.entry`. |
| `allow_tma` | Suggest whether to use TMA for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`. |
| `latency` | Latency hint for `cuda_tile.load_view_tko` and `cuda_tile.store_view_tko`. |

**Constraints:**

- Operation must encode variadic operand segment sizes in attributes.
- Destination type is expected to be a pointer type of value type (i.e., `destination` has type `tile<ptr<E>>` where `value` has type `tile<E>`).
- Shape of `destination` must match the shape of `mask` (when mask is provided).
- Shape of `destination` must match the shape of `value`.
- Operation must infer result types from operands and attributes.

**Examples:**

Example 1 -- Store all elements without token:

```
// Store all elements from a value tile to memory locations specified by pointers
%ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>

%vals = constant <f32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tile<8xf32>

// Store without token or mask - all elements stored
%res_token = store_ptr_tko weak %ptrs, %vals
    : tile<8xptr<f32>>, tile<8xf32> -> token
```

Example 2 -- Store with mask for selective storage:

```
%ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
%ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
%offsets = iota : tile<8xi32>
%ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>

%vals = constant <f32: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tile<8xf32>
%mask = constant <i1: [1, 0, 1, 0, 1, 0, 1, 0]> : tile<8xi1>

// Only elements at positions 0, 2, 4, 6 are stored.
// Elements at positions 1, 3, 5, 7 are not modified in memory.
%res_token = store_ptr_tko weak %ptrs, %vals, %mask
    : tile<8xptr<f32>>, tile<8xf32>, tile<8xi1> -> token
```

Example 3 -- Store with token for ordering:

```
// Create a token chain: load -> compute -> store
%token0 = make_token : token
%loaded, %token1 = load_ptr_tko weak %src_ptrs token=%token0
    : tile<8xptr<f32>> -> tile<8xf32>, token

// ... compute on %loaded ...

// Store is ordered after the load via %token1
%token2 = store_ptr_tko weak %dst_ptrs, %computed_vals token=%token1
    : tile<8xptr<f32>>, tile<8xf32> -> token
```

Example 4 -- Store with mask and token:

```
%mask = constant <i1: [1, 1, 1, 1, 0, 0, 0, 0]> : tile<8xi1>
%token0 = make_token : token

// Selective store with token ordering - only first 4 elements stored
%res_token = store_ptr_tko relaxed device %ptrs, %vals, %mask token=%token0
    : tile<8xptr<f32>>, tile<8xf32>, tile<8xi1> -> token
```

---

## Token Ordering Explanation

Tile IR memory operations use a token-based ordering system rather than relying on program order. Understanding how tokens work is essential for writing correct Tile IR programs.

### How Tokens Work

Each token-ordered memory operation optionally accepts an input token and always produces an output token. The relationship between an operation's input token and its output token establishes a dependency: the output token's operation must complete its memory effects after all operations that contributed to the input token have completed their memory effects.

### Token Dependency Chains

Tokens form dependency chains through explicit threading:

```
%t0 = make_token : token                    // Fresh token, no dependencies
%r1, %t1 = load_ptr_tko ... token=%t0       // t1 depends on t0 (no prior constraint)
%r2, %t2 = load_ptr_tko ... token=%t1       // t2 depends on t1, transitively on t0
%t3 = store_ptr_tko ... token=%t2           // t3 depends on t2, transitively on t1 and t0
```

This chain establishes: load1 -> load2 -> store (ordered sequentially).

### Fork-Join Patterns with join_tokens

When multiple operations can proceed independently but must all complete before a subsequent operation, use `join_tokens`:

```
%t0 = make_token : token
%r1, %t1 = load_ptr_tko ... token=%t0       // Independent load 1
%r2, %t2 = load_ptr_tko ... token=%t0       // Independent load 2 (concurrent with load 1)
%joined = join_tokens %t1, %t2              // Merge dependencies
%t3 = store_ptr_tko ... token=%joined       // Store waits for both loads
```

### No Token = No Ordering Guarantee

When a token-ordered operation is invoked without an input token, the compiler is free to reorder it with respect to all other token-ordered operations. The only guarantee is that the operation will eventually complete.

### Memory Ordering vs Token Ordering

Token ordering controls **when** an operation's memory effects become visible relative to other operations. Memory ordering semantics (`weak`, `relaxed`, `acquire`, `release`) control **how** those effects are synchronized with other threads or tile blocks. Both mechanisms are orthogonal and serve different purposes:

- **Token ordering**: Establishes ordering between operations within the same tile block.
- **Memory ordering semantics**: Establishes visibility guarantees between different tile blocks or devices.
