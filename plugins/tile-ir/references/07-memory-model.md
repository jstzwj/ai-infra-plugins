# Chapter 7: Memory Model

## 7.1 Overview

The memory model defines the legal values that loads can return from memory. This is not as straightforward as one might expect at first glance; to enable compiler and hardware optimizations, Tile IR allows the apparent re-ordering of instructions.

This memory model is derived from the **PTX memory model**, and synchronization primitives are deliberately similar to enable seamless interoperation between Tile IR and PTX code.

### Key Principles

1. **Tile operations expand to per-element memory operations**: Each tile load/store/atomic generates one memory model operation per element.
2. **Element order within tiles is unspecified**: When expanding a tile operation, the order of individual element accesses is deliberately left unspecified for implementation flexibility.
3. **Token-based ordering**: Unlike traditional threading models where program order implies happens-before, Tile IR requires explicit token dependencies for memory ordering.
4. **Scope-aware synchronization**: Memory operations carry scope information that determines which other threads/operations they can synchronize with.

---

## 7.2 Memory Model Operations

The memory model is built of relations between individual element accesses of tile operations, and restrictions on cycles of those relations. A Tile IR memory instruction generates one or more memory model operations. Tile loads, stores, and atomic updates generate one memory operation per element in the tile.

The order in which element-level memory model operations happen within a single tile operation is deliberately left unspecified. This gives hardware and compiler freedom to optimize access patterns (e.g., coalesced memory transactions, vectorized loads).

### Memory Model Operations Table

| Tile IR Operation | Generated Memory Model Operations | Notes |
|---|---|---|
| `cuda_tile.load_ptr_tko` | One read operation per element | Token-ordered; produces a token |
| `cuda_tile.store_ptr_tko` | One write operation per element | Token-ordered; consumes and produces a token |
| `cuda_tile.load_view_tko` | One read operation per element | View-based; produces a token |
| `cuda_tile.store_view_tko` | One write operation per element | View-based; consumes and produces a token |
| `cuda_tile.atomic_rmw_tko` | One read + one write per element (RMW pair) | Paired by read-modify-write order |
| `cuda_tile.atomic_cas_tko` | One read + one write per element (RMW pair) | Compare-and-swap; paired by RMW order |
| `cuda_tile.make_token` | No memory operations | Creates an initial token value |
| `cuda_tile.join_tokens` | No memory operations | Combines multiple tokens into one |
| `cuda_tile.print_tko` | One write operation per element | Produces a token for memory ordering |

### Element-wise Expansion Example

When a tile load reads a `tile<4x4xf32>` from memory, it generates 16 individual memory model read operations (one per element). The order of these 16 operations relative to each other is unspecified:

```
%val, %tok = load_ptr_tko weak %ptrs : tile<4x4xptr<f32>> -> tile<4x4xf32>, token

// Internally generates 16 read operations:
//   read[0,0], read[0,1], read[0,2], read[0,3],
//   read[1,0], read[1,1], read[1,2], read[1,3],
//   read[2,0], read[2,1], read[2,2], read[2,3],
//   read[3,0], read[3,1], read[3,2], read[3,3]
//
// Order of these reads relative to each other: UNSPECIFIED
```

---

## 7.3 Scopes

Memory operations in Tile IR may have a **scope**. Operations without a scope are called **weak**. All memory operations specify either a scope or `weak`. Any scope other than `weak` requires a memory ordering to be set.

### Scope Definitions

| Scope | Description | Use Case |
|-------|-------------|----------|
| `tile_block` | Communication within a single tile block | Synchronizing operations within the same tile block thread, handling internal aliasing |
| `device` | Communication within the same GPU device | Synchronizing between different tile blocks on the same GPU |
| `sys` | System-wide communication | Synchronizing between GPU and CPU or other devices |

### Scope Hierarchy

```
+================================================================+
|                        sys scope                                |
|  (System-wide: CPU, GPU, other devices)                        |
|                                                                |
|  +============================================================+|
|  |                     device scope                            ||
|  |  (Single GPU device: all tile blocks)                       ||
|  |                                                            ||
|  |  +========================================================+||
|  |  |                tile_block scope                        |||
|  |  |  (Single tile block thread: within one block)          |||
|  |  +========================================================+||
|  |                                                            ||
|  +============================================================+|
|                                                                |
+================================================================+
```

Scopes are ordered by inclusiveness: `tile_block` < `device` < `sys`. A wider scope includes all narrower scopes within it. When two operations specify scopes, they are morally strong with respect to each other if each scope includes the tile block executing the other operation.

### Weak Operations

Weak operations cannot be used to communicate through memory between threads, or between fragments of the same tile block which are not ordered by token order. The compiler may assume that tiles accessed with `weak` are not concurrently accessed by any other thread.

```
// Safe: no other thread accesses this memory
%val, %tok = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token
```

### Scope Selection Guidelines

| Scenario | Recommended Scope | Rationale |
|----------|-------------------|-----------|
| Private tile access, no sharing | `weak` | Maximum optimization freedom |
| Communication within tile block thread | `tile_block` | Covers intra-block synchronization |
| Communication between tile blocks on same GPU | `device` | Covers all blocks on one GPU |
| Communication between GPU and CPU/other devices | `sys` | System-wide visibility required |
| Tile store with potential internal aliasing | `tile_block` | Prevents intra-element races |

### Example: Tile Block Scope

```
// Necessary: another operation in this tile block thread
// may access overlapping memory
store_ptr_tko %ptrs, %val : tile<128xptr<f32>>, tile<128xf32> -> token
    relaxed tile_block
```

### Example: Device Scope

```
// Necessary: other tile blocks on the same GPU may read this memory
store_ptr_tko %ptrs, %val : tile<128xptr<f32>>, tile<128xf32> -> token
    release device
```

---

## 7.4 Memory Ordering

Memory operations have a memory ordering parameter that controls how the operation can be used for synchronization. Synchronization through memory is a two-party process: it requires a **releaser** and an **acquirer** observing the same location. When a pair of memory accesses synchronize through memory, they establish a **happens-before** relationship.

Any ordering other than `weak` requires a scope to be set.

### Memory Orderings

| Ordering | Description | Scope Required | Synchronizes |
|----------|-------------|----------------|-------------|
| `weak` | No concurrent accesses to the source/destination location | No | No |
| `relaxed` | Concurrent access allowed, but no happens-before established | Yes | No |
| `release` | Write side of release-acquire pair; establishes happens-before when observed by acquire | Yes | Yes (write side) |
| `acquire` | Read side of release-acquire pair; observes release ordering | Yes | Yes (read side) |
| `acq_rel` | Combined release + acquire semantics; used for RMW operations | Yes | Yes (both) |

### Memory Ordering Strength Lattice

```
            weak
             |
           relaxed
           /    \
      release   acquire
           \    /
           acq_rel
```

Orderings are ordered by strength: `weak` < `relaxed` < `release`/`acquire` < `acq_rel`. A stronger ordering provides more synchronization guarantees but may limit hardware and compiler optimizations.

### Detailed Ordering Descriptions

**`weak`**: The least restrictive ordering. The programmer asserts that no other thread or tile block fragment will concurrently access the same memory location. This allows the compiler maximum freedom to reorder, combine, or eliminate memory operations. Use `weak` when you can guarantee exclusive access (e.g., thread-local scratchpad memory, or when token ordering already provides sufficient ordering).

**`relaxed`**: Permits concurrent accesses to the same location but does not establish any happens-before relationship. Useful for counters, statistics, or other cases where atomicity matters but ordering relative to other memory operations does not. A relaxed atomic increment will increment without tearing but will not synchronize any other memory operations.

**`release`**: The write side of a release-acquire synchronization pair. A release store ensures that all prior memory operations (reads and writes) are visible to any thread that observes this store via a matching acquire. When a release store is observed by an acquire load on the same location, a happens-before relationship is established from the release to the acquire.

**`acquire`**: The read side of a release-acquire synchronization pair. An acquire load ensures that all subsequent memory operations occur after the acquire in the observed order. When an acquire load reads a value written by a release store, a happens-before relationship is established.

**`acq_rel`**: Combines both release and acquire semantics. Primarily used for read-modify-write operations (like atomic compare-and-swap) where the operation both reads from and writes to memory. Ensures that prior operations are visible (release) and subsequent operations are ordered after (acquire).

### Example: Release-Acquire Synchronization

```
// Tile Block A (producer):
store_ptr_tko release device %ptrs, %flag_value : ... -> token
// All prior writes are now visible to any acquiring tile block

// Tile Block B (consumer):
%val, %tok = load_ptr_tko acquire device %ptrs : ... -> tile<...>, token
// If this load observes the release store above,
// happens-before is established from A to B
```

---

## 7.5 Moral Strength

Two accesses to the same location are **morally strong** if:

1. The operations are related in restricted program order (they come from the same tile block thread and are token-ordered or within the same instruction), **or**
2. Each operation specifies a scope that includes the tile block executing the other operation.

Moral strength determines whether two overlapping accesses are guaranteed to observe each other. Two operations that are morally strong will not exhibit "out of thin air" values between them.

### Example: Morally Strong Pair

```
// Operation A in Tile Block 0:
store_ptr_tko release device %ptrs, %val : ... -> token

// Operation B in Tile Block 1:
%result, %tok = load_ptr_tko acquire device %ptrs : ... -> tile<...>, token

// Both specify device scope, which includes the other's tile block.
// Therefore A and B are morally strong.
```

### Example: NOT Morally Strong Pair

```
// Operation A in Tile Block 0:
store_ptr_tko release tile_block %ptrs, %val : ... -> token

// Operation B in Tile Block 1:
%result, %tok = load_ptr_tko acquire device %ptrs : ... -> tile<...>, token

// A specifies tile_block scope, which does NOT include Tile Block 1.
// Therefore A and B are NOT morally strong.
// (Even though B has device scope, A's tile_block scope excludes B.)
```

---

## 7.6 Tokens and Token Order

Tile IR provides explicit annotation of dependencies between loads and stores via **token-ordered operations**. Tile IR produces wide loads and stores of whole tiles of data, making efficient use of various GPU resources in parallel. Token-ordered operations explicitly inform the Tile IR toolchain that two operations may happen in parallel and will not interfere with each other.

There is a family of memory operations called **token-ordered operations** (suffixed `_tko`) that produce and consume tokens. Tokens are abstract values for building dependencies between memory operations within the same tile block thread.

### Token Properties

- **Abstract**: Tokens have no runtime representation. They are compile-time constructs used to express ordering constraints.
- **Non-computable**: Tokens cannot be used in arithmetic, comparison, or any data-dependent operation.
- **Non-storable**: Tokens cannot be stored to or loaded from memory.
- **Ordering only**: The sole purpose of tokens is to establish ordering between memory operations.
- **Not optimizable**: Unlike program dependencies, the Tile IR toolchain cannot remove token dependencies.

### Critical Rule: Tokens Must Be Used Explicitly

Program dependencies (control flow, data dependency, address dependency) **do not provide ordering** between two memory operations. **Tokens must be used**, even where the token ordering appears redundant with program dependencies. Program dependencies may be optimized away by the Tile IR toolchain, whereas token dependencies are not.

```
// WRONG: Relying on program order (data dependency)
%val, %tok1 = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token
store_ptr_tko weak %out_ptrs, %val : tile<128xptr<f32>>, tile<128xf32> -> token
// The compiler MAY reorder these because there's no token chain!

// CORRECT: Using explicit token ordering
%val, %tok1 = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token
%tok2 = store_ptr_tko weak %out_ptrs, %val token=%tok1 : tile<128xptr<f32>>, tile<128xf32> -> token
```

### Token Chain Example

```
// Create an initial token
%tok0 = make_token

// Token-ordered load: consumes tok0, produces tok1
%val, %tok1 = load_ptr_tko weak %ptrs token=%tok0 : tile<128xptr<f32>> -> tile<128xf32>, token

// Token-ordered store: consumes tok1, produces tok2
// This store is guaranteed to happen AFTER the load above
%tok2 = store_ptr_tko weak %out_ptrs, %val token=%tok1 : tile<128xptr<f32>>, tile<128xf32> -> token

// Token-ordered store: consumes tok2, produces tok3
%tok3 = store_ptr_tko weak %out_ptrs2, %val2 token=%tok2 : tile<128xptr<f32>>, tile<128xf32> -> token
```

### Token Join (Parallel Operations)

When multiple independent operations can proceed in parallel, they can share the same input token. The `join_tokens` operation waits for all input tokens before producing a combined output token.

```
// Create initial token
%tok0 = make_token

// Two independent stores can share the same input token
%tok1 = store_ptr_tko weak %ptrs_a, %val_a token=%tok0 : tile<128xptr<f32>>, tile<128xf32> -> token
%tok2 = store_ptr_tko weak %ptrs_b, %val_b token=%tok0 : tile<128xptr<f32>>, tile<128xf32> -> token

// Join the tokens: tok3 is available only after both stores complete
%tok3 = join_tokens %tok1, %tok2 : token, token -> token

// This load happens after both stores
%result, %tok4 = load_ptr_tko weak %ptrs_c token=%tok3 : tile<128xptr<f32>> -> tile<128xf32>, token
```

### Token Dependency Graph (Visual)

```
     tok0 (make_token)
      /       \
     v         v
  store_a    store_b      (parallel: both consume tok0)
  (tok1)     (tok2)
     \       /
      v     v
   join (tok3)            (waits for both tok1 and tok2)
      |
      v
    load (tok4)           (after both stores)
```

---

## 7.7 Base Relations

The memory model is defined in terms of several base relations between memory model operations. These relations form the foundation upon which the memory consistency model is built.

### 7.7.1 Coherence Order

There exists a partial transitive order that relates overlapping write operations, determined at runtime, called **coherence order**. Two overlapping write operations are related in coherence order if they are morally strong or if they are related in happens-before order.

**Properties:**

- **Transitive**: If write W1 precedes W2 in coherence order, and W2 precedes W3, then W1 precedes W3.
- **Per-location**: Coherence order is defined per memory location. Writes to different locations are not related.
- **Partial**: Only morally strong writes or writes related by happens-before are in coherence order.

```
Location X:
  Tile Block A: store X = 1 (release, device)
  Tile Block B: store X = 2 (release, device)

If A and B are morally strong, then one of the following holds:
  - Coherence(X): store_1 before store_2  (A's write observed first)
  - Coherence(X): store_2 before store_1  (B's write observed first)

The order is determined at runtime.
```

### 7.7.2 Program Order

A memory operation `a` is **program order** before a memory operation `b` if the instruction that gave rise to `a` is before the instruction that gave rise to `b` in the program source.

**Important Notes:**

- Program order relates individual memory model operations, not tile-level operations.
- Within a single tile operation, the order of element-level memory model operations is unspecified.
- Program order alone does **NOT** establish happens-before in Tile IR (unlike many other memory models).
- Token order, not program order, is the primary mechanism for establishing ordering within a tile block thread.

```
// In the Tile IR source:
store_ptr_tko weak %ptrs_a, %val_a token=%tok0 : ... -> token   // Instruction A
store_ptr_tko weak %ptrs_b, %val_b token=%tok0 : ... -> token   // Instruction B

// Program order: A's memory ops are before B's memory ops
// But the compiler/hardware MAY reorder these because:
//   - Both use the same input token (tok0), NOT chained
//   - No happens-before between them via token order
```

### 7.7.3 Waits-for Order

An operation `a` **waits-for** an operation `b` if:

1. An instruction `I1` gave rise to `a`, `I1` produced a token `t`, an instruction `I2` gave rise to `b` and `I2` depends upon the token `t`; **or**
2. There is some operation `c` such that `a` waits-for `c` and `c` waits-for `b`.

Waits-for order is the transitive closure of the direct token dependency relation.

**Properties:**

- **Transitive**: If a waits-for b and b waits-for c, then a waits-for c.
- **Directed**: The relation flows from consumer to producer (the consumer waits for the producer).
- **Within tile block thread**: Token dependencies are only meaningful within a single tile block thread.

```
%tok0 = make_token
%tok1 = store_ptr_tko ... token=%tok0    // I1 produces tok1
%tok2 = load_ptr_tko ... token=%tok1     // I2 produces tok2
%tok3 = store_ptr_tko ... token=%tok2    // I3 produces tok3

// Direct waits-for:
//   I3's ops wait-for I2's ops (via tok2)
//   I2's ops wait-for I1's ops (via tok1)

// Transitive waits-for:
//   I3's ops wait-for I1's ops (via tok1 and tok2)
```

### 7.7.4 Reads From

A read operation `r` **reads-from** a write operation `w` when `r` and `w` access the same location and `r` reads the value written by `w`.

A read must read from one of:

1. The latest write in coherence order visible to the read (considering scope and ordering)
2. The initial value of the memory location (if no prior write exists)

**Properties:**

- **Per-element**: For tile operations, reads-from is defined per element, not per tile.
- **Value-determined**: The reads-from relation is determined by the actual value returned by the read.
- **Consistency-constrained**: Valid reads-from relations must satisfy all memory model axioms.

```
Location X (initially 0):
  Tile Block A: store X = 42 (release, scope=device)
  Tile Block B: load X (acquire, scope=device)

// Possible reads-from relations:
// Case 1: B reads 0  (reads-from initial value)
//   - The release store has not been made visible yet
// Case 2: B reads 42 (reads-from A's store)
//   - The release-acquire pair synchronizes
//   - happens-before is established from A to B
```

### 7.7.5 Read-Modify-Write Order

Read-modify-write atomics generate a pair of memory operations for each element location within a tile. Each pair is related by **read-modify-write order**.

The RMW order ensures that the read and write components of an atomic operation are indivisible with respect to other writes. No other write can be observed between the read and write of an RMW operation.

**Supported RMW Operations:**

| Tile IR Operation | RMW Behavior |
|---|---|
| `atomic_rmw_tko add` | Atomically adds a value |
| `atomic_rmw_tko sub` | Atomically subtracts a value |
| `atomic_rmw_tko andi` | Atomically performs bitwise AND |
| `atomic_rmw_tko ori` | Atomically performs bitwise OR |
| `atomic_rmw_tko xori` | Atomically performs bitwise XOR |
| `atomic_rmw_tko maxi` | Atomically computes maximum (signed) |
| `atomic_rmw_tko mini` | Atomically computes minimum (signed) |
| `atomic_rmw_tko maxf` | Atomically computes maximum (float) |
| `atomic_rmw_tko minf` | Atomically computes minimum (float) |
| `atomic_cas_tko` | Atomically compares and swaps |

### 7.7.6 Atomicity

When an atomic operation `a` and a write `w` overlap and are morally strong, the following two communications cannot both exist in the same execution:

- `a` reads from a write `w'` that precedes `w` in coherence order
- `a` follows `w` in coherence order

This ensures atomic operations provide an all-or-nothing guarantee: an atomic operation either observes a write completely or not at all.

```
Location X (initially 0):
  Tile Block A: atomic_rmw_tko add X += 1
  Tile Block B: store X = 100

If A and B are morally strong, the following CANNOT both happen:
  - A reads 0 (from initial value, which precedes 100 in coherence)
  - A's write follows B's write of 100 in coherence

This means A either:
  - Reads 0 and writes 1 (B's write happens after A completes), or
  - Reads 100 and writes 101 (B's write happens before A starts)
```

---

## 7.8 No-Thin-Air

It is not practical to specify a "no thin air" axiom without preventing useful compiler optimizations. Tile IR provides an informal guarantee that the implementation will not provide values out of thin air to satisfy program executions.

The "no-thin-air" guarantee means that a read should not return a value that could only arise from a circular dependency chain. While formalizing this precisely is known to be very difficult in memory model theory, Tile IR provides an informal guarantee that the implementation will not produce surprising values through speculative execution or out-of-order optimization.

```
// Initially X = 0, Y = 0
// Tile Block A:                 // Tile Block B:
  r1 = load X                     r2 = load Y
  store Y = r1                    store X = r2

// Thin-air would be: both reads return 42 (a value never stored)
// Tile IR guarantees this cannot happen.
```

---

## 7.9 Data Races

Two accesses are said to **conflict** when they access the same location and at least one of them is a write.

Two conflicting memory accesses are said to be in a **data race** if they are not related in happens-before and they are not morally strong.

**Programs with data races have undefined behaviour.**

### Data Race Detection Checklist

1. Do the two operations access the same memory location?
2. Is at least one of them a write?
3. Are the operations related by happens-before?
4. Are the operations morally strong?

If answers to (1) and (2) are **yes**, and answers to (3) and (4) are **no**, then there is a data race.

### Example: Data Race (Undefined Behavior)

```
// Tile Block A:
store_ptr_tko relaxed tile_block %ptrs, %val : ... -> token

// Tile Block B (same device, no synchronization):
%result, %tok = load_ptr_tko relaxed tile_block %ptrs : ... -> tile<...>, token

// These two operations conflict (same location, one is a write).
// They are NOT morally strong (tile_block scope does not cover different tile blocks).
// They are NOT related by happens-before (no release/acquire pair).
// Therefore: DATA RACE -> undefined behavior.
```

### Example: No Data Race (Properly Synchronized)

```
// Tile Block A (producer):
store_ptr_tko release device %ptrs, %val : ... -> token

// Tile Block B (consumer):
%result, %tok = load_ptr_tko acquire device %ptrs : ... -> tile<...>, token

// These two operations conflict (same location, one is a write).
// But: if the acquire observes the release, happens-before is established.
// Therefore: NO DATA RACE (properly synchronized).
```

### Example: No Data Race (Morally Strong)

```
// Two atomic operations on the same location with device scope:
// Tile Block A:
%old_a, %tok_a = atomic_rmw_tko add relaxed device %ptrs, %val_a : ... -> tile<...>, token

// Tile Block B:
%old_b, %tok_b = atomic_rmw_tko add relaxed device %ptrs, %val_b : ... -> tile<...>, token

// Both use device scope, which includes each other's tile blocks.
// Therefore: morally strong -> NO DATA RACE (atomicity guaranteed).
```

---

## 7.10 PTX Interoperability

The axioms and relations of the Tile IR memory model are designed to be a **strict weakening** of the PTX memory model.

The Tile IR memory model is designed to allow communication with PTX threads. Release and acquire patterns in Tile IR will match up with acquire and release patterns in PTX to build PTX causality and Tile IR happens-before.

Data races between accesses in a Tile IR program and a PTX program will result in undefined behavior as if the data race were all in Tile IR.

### PTX Interoperability Rules

| Tile IR Operation | Compatible PTX Operation | Notes |
|---|---|---|
| `release` store | `st.release` | Establishes happens-before / PTX causality |
| `acquire` load | `ld.acquire` | Observes Tile IR release / PTX release |
| `acq_rel` RMW | `atom.acq_rel` | Full bidirectional synchronization |
| `relaxed` atomic | `atom.relaxed` | No ordering guarantees |
| `weak` load/store | Standard `ld`/`st` | No concurrency expected |

### Scope Mapping

| Tile IR Scope | PTX Scope | Notes |
|---|---|---|
| `tile_block` | `.cta` | Cooperative Thread Array scope |
| `device` | `.gpu` | GPU device scope |
| `sys` | `.sys` | System-wide scope |

### Example: Tile IR to PTX Communication

```
// Tile IR (on GPU, Tile Block A):
store_ptr_tko release device %ptrs, %result : ... -> token

// PTX (on GPU, Thread Block B):
ld.acquire.gpu %r, [%addr]

// The release-acquire pair establishes synchronization.
// PTX causality and Tile IR happens-before are unified.
```

### Example: Mixed Data Race (Undefined Behavior)

```
// Tile IR (unsynchronized):
store_ptr_tko weak %ptrs, %val : ... -> token

// PTX (unsynchronized):
st.global [%addr], %val2

// Data race between Tile IR and PTX -> undefined behavior.
```

---

## 7.11 Hazards and Intuition

### 7.11.1 Token Ordered Operations Reading from the Future

Store buffering and load buffering behavior are visible when using token-ordered operations without proper token constraints. When two token-ordered operations are not connected by a token chain, the compiler and hardware are free to reorder them.

**Store Buffering Hazard:**

```
// Initially: memory[X] = 0
%tok0 = make_token
%tok1 = store_ptr_tko weak %ptrs_x, %one token=%tok0 : ... -> token
%val, %tok2 = load_ptr_tko weak %ptrs_x token=%tok0 : ... -> tile<...>, token
// ^ BUG: uses tok0, NOT tok1! Load may read 0 (before store completes)
```

**Correct Version:**

```
%tok0 = make_token
%tok1 = store_ptr_tko weak %ptrs_x, %one token=%tok0 : ... -> token
%val, %tok2 = load_ptr_tko weak %ptrs_x token=%tok1 : ... -> tile<...>, token
// ^ Now guaranteed to read the stored value
```

### 7.11.2 Race Hazards Within a Single Tile Block Thread

In Tile IR you can construct a data race when there are two accesses to the same location within a single tile block store (because of internal overlap in the destination tile), or with two accesses within a tile block thread to the same location which are not ordered by token order.

**Internal Alias Hazard:**

```
// A tile store where the tile has internal aliasing:
// Some elements map to the same memory location.
// The writes to aliased locations are not ordered -> potential data race.

// Solution: use tile_block scope
store_ptr_tko release tile_block %ptrs, %tile : ... -> token
```

**Intra-Thread Data Race:**

```
%tok0 = make_token
%tok1 = store_ptr_tko weak %ptrs, %val_42 token=%tok0 : ... -> token
%tok2 = store_ptr_tko weak %ptrs, %val_100 token=%tok0 : ... -> token
// ^ BUG: Both stores use tok0, NOT chained! Same address, unordered -> DATA RACE

// Correct: chain the tokens
%tok0 = make_token
%tok1 = store_ptr_tko weak %ptrs, %val_42 token=%tok0 : ... -> token
%tok2 = store_ptr_tko weak %ptrs, %val_100 token=%tok1 : ... -> token
```

### 7.11.3 Intuition on Token Ordering

**Guideline 1: Non-overlapping accesses can be parallel**

```
// Two tiles don't overlap in memory, so parallel execution is safe:
%tok0 = make_token
%tok1 = store_ptr_tko weak %ptrs_a, %tile_a token=%tok0 : ... -> token
%tok2 = store_ptr_tko weak %ptrs_b, %tile_b token=%tok0 : ... -> token
%tok3 = join_tokens %tok1, %tok2 : token, token -> token
```

**Guideline 2: Order all prior events before a release**

```
%tok0 = make_token
%tok1 = store_ptr_tko weak %data_ptrs, %data token=%tok0 : ... -> token
%tok2 = store_ptr_tko release device %flag_ptrs, %flag token=%tok1 : ... -> token
// The release store is token-ordered after all prior writes
```

**Guideline 3: Order all subsequent events after an acquire**

```
%tok0 = make_token
%flag, %tok1 = load_ptr_tko acquire device %flag_ptrs token=%tok0 : ... -> tile<...>, token
%data, %tok2 = load_ptr_tko weak %data_ptrs token=%tok1 : ... -> tile<...>, token
// The data load is token-ordered after the acquire
```

**Guideline 4: Tokens with conditional control flow**

```
%tok0 = make_token
%tok1 = store_ptr_tko weak %ptrs, %val token=%tok0 : ... -> token

%x, %y = if %cond -> (tile<i32>, token) {
  %val, %tok_inner = load_ptr_tko weak %ptrs token=%tok1 : ... -> tile<...>, token
  yield %val, %tok_inner : tile<...>, token
} else {
  %tok_inner = store_ptr_tko weak %ptrs2, %other token=%tok1 : ... -> token
  %zero = constant <i32: 0> : tile<i32>
  yield %zero, %tok_inner : tile<i32>, token
}
// Both branches properly order relative to the initial store
```

### Summary of Key Hazards

| Hazard | Description | Prevention |
|--------|-------------|------------|
| Store buffering | Load may read stale value if not ordered after store | Token-order the load after the store |
| Load buffering | Store may become visible before preceding loads complete | Token-order the store after the load |
| Intra-thread race | Two operations on same address without token ordering | Always use tokens for same-address operations |
| Internal alias | Tile store with overlapping elements | Use `tile_block` scope |
| Cross-block race | Unsynchronized access from different tile blocks | Use release/acquire with appropriate scope |
| Missing token chain | Gap in token chain breaks ordering | Ensure complete token chains from creation to use |

---

## 7.12 Complete Memory Operation Syntax Reference

### Load Operations

```
// Pointer-based load (weak)
%val, %token = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token

// Pointer-based load with mask and padding (weak)
%val, %token = load_ptr_tko weak %ptrs, %mask, %pad : tile<128xptr<f32>>, tile<128xi1>, tile<128xf32> -> tile<128xf32>, token

// Pointer-based load (relaxed, device scope)
%val, %token = load_ptr_tko relaxed device %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token

// Pointer-based load (acquire, device scope)
%val, %token = load_ptr_tko acquire device %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token

// View-based load (weak)
%tile, %token = load_view_tko weak %partition[%x, %y] :
    partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128,1]>>, tile<i32>
    -> tile<64x64xf32>, token

// View-based load (acquire, sys scope)
%tile, %token = load_view_tko acquire sys %partition[%x, %y] : ... -> tile<64x64xf32>, token
```

### Store Operations

```
// Pointer-based store (weak)
%token = store_ptr_tko weak %ptrs, %val token=%tok_in : tile<128xptr<f32>>, tile<128xf32> -> token

// Pointer-based store (release, device scope)
%token = store_ptr_tko release device %ptrs, %val token=%tok_in : tile<128xptr<f32>>, tile<128xf32> -> token

// View-based store (weak)
%token = store_view_tko weak %val, %partition[%x, %y] token=%tok_in :
    tile<64x64xf32>, partition_view<...>, tile<i32> -> token

// View-based store (release, device scope)
%token = store_view_tko release device %val, %partition[%x, %y] token=%tok_in :
    tile<64x64xf32>, partition_view<...>, tile<i32> -> token
```

### Atomic Operations

```
// Atomic RMW (relaxed, device scope)
%old, %token = atomic_rmw_tko add relaxed device %ptrs, %val token=%tok_in :
    tile<128xptr<f32>>, tile<128xf32> -> tile<128xf32>, token

// Atomic RMW (acq_rel, device scope)
%old, %token = atomic_rmw_tko add acq_rel device %ptrs, %val token=%tok_in :
    tile<128xptr<f32>>, tile<128xf32> -> tile<128xf32>, token

// Atomic CAS (acq_rel, sys scope)
%old, %token = atomic_cas_tko acq_rel sys %ptrs, %cmp, %val token=%tok_in :
    tile<128xptr<f32>>, tile<128xf32>, tile<128xf32> -> tile<128xf32>, token
```

### Token Operations

```
// Create initial token
%tok0 = make_token

// Join multiple tokens
%tok_joined = join_tokens %tok1, %tok2 : token, token -> token
```
