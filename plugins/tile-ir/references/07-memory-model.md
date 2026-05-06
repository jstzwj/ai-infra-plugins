# Memory Model

The memory model defines the legal values that loads can return from memory. This is not as straightforward as one might expect at first glance; to enable compiler and hardware optimizations, we allow the apparent re-ordering of instructions.

This memory model is derived from the PTX memory model, and synchronization primitives are deliberately similar.

## Memory Model Operations

The memory model is built of relations between individual element accesses of tile operations, and restrictions on cycles of those relations. Therefore, a Tile IR memory instruction generates one or more memory model operations. In particular, tile loads, stores, and atomic updates generate one memory operation per element in the tile. When expanding a tile operation into many memory model operations, the order in which the memory model operations happen is deliberately left unspecified for implementation flexibility.

## Scopes

Memory operations in Tile IR may have a scope. Operations without a scope are called **weak**. All memory operations specify a scope or weak. Any scope other than weak requires a memory ordering to be set.

| Scope | Description |
|-------|-------------|
| `tile_block` | Tile block scope, for communication within a single tile block. |
| `device` | Device scope, for communication within the same GPU. |
| `sys` | System scope, for communication anywhere in the system. |

Weak operations cannot be used to communicate through memory between threads, or between fragments of the same tile block which are not ordered by token order. The compiler may assume that tiles accessed with weak are not concurrently accessed by any other thread.

> **Note:** Tile Block scope is needed when building communicating algorithms where there is communication within a single tile block. This is necessary when communicating memory through memory between operations which are not ordered by token order, or when storing to a tile with internal aliasing.

## Memory Ordering

Memory operations have a memory ordering parameter which controls how that operation can be used for synchronization. Synchronization through memory is a two-party process, which requires a releaser and an acquirer observing the same location. When a pair of memory accesses synchronize through memory it establishes a happens before relationship.

Any ordering other than weak requires a scope to be set.

| Memory ordering | Description |
|-----------------|-------------|
| `weak` | No concurrent accesses to the source/destination location. |
| `relaxed` | There may be concurrent access to the location, but this access does not establish a happens-before relationship. |
| `release` | There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established. |
| `acquire` | There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established. |
| `acq_rel` | There may be concurrent accesses to the location. This has the effect of both a release and acquire operation. |

## Moral Strength

Two accesses to the same location are **morally strong** if the operations are related in restricted program order, or each operation specifies a scope which includes the tile block executing the other operation.

## Tokens and Token Order

In Tile IR we have explicit annotation of dependencies between loads and stores for the token-ordered operations. Tile IR produces wide loads and stores of whole tiles of data, making efficient use of various resources of the GPU in parallel. We provide token ordered operations to explicitly inform the Tile IR toolchain that two operations may happen in parallel, and will not interfere with each other.

There is a family of memory operations called **token ordered operations** which produce and consume tokens. Tokens are abstract values in the Tile IR language for building dependencies between memory operations within the same tile block thread. They have no concrete representation at runtime, cannot be compared, computed upon, or stored/loaded to/from memory.

Program dependencies (i.e. dependencies apparent from control flow, data dependency, or address dependency) do not provide ordering between two memory operations. **Tokens must be used**, even where the token ordering appears redundant with program dependencies. Program dependencies may be optimized away by the Tile IR toolchain, whereas token dependencies are not.

## Base Relations

### Coherence Order

There exists a partial transitive order that relates overlapping write operations, determined at runtime, called **coherence order**. Two overlapping write operations are related in coherence order if they are morally strong or if they are related in happens before order.

### Program Order

A memory operation `a` is **program order** before a memory operation `b` if the instruction that gave rise to `a` is before the instruction that gave rise to `b` in the program source.

### Waits-for Order

An operation `a` **waits-for** an operation `b` if:

- An instruction `I1` gave rise to `a`, `I1` produced a token `t`, an instruction `I2` gave rise to `b` and `I2` depends upon the token `t`; or
- There is some operation `c` such that `a` waits-for `c` and `c` waits-for `b`.

### Reads From

A read operation `r` **reads-from** a write operation `w` when `r` and `w` access the same location and `r` reads the value written by `w`.

### Read-Modify-Write Order

Read-modify-write atomics generate a pair of memory operations for each element location within a tile. Each pair of memory operations is related by **read-modify-write order**.

### Atomicity

When an atomic operation `a` and a write `w` overlap and are morally strong, then the following two communications cannot both exist in the same execution:

- `a` reads from a write `w'` that precedes `w` in coherence order.
- `a` follows `w` in coherence order.

## No-Thin-Air

It is not practical to specify a "no thin air" axiom without preventing useful compiler optimizations. We therefore say informally that the implementation will not provide values out of thin air to satisfy program executions.

## Data Races

Two accesses are said to **conflict** when they access the same location and at least one of them is a write.

Two conflicting memory accesses are said to be in a **data race** if they are not related in happens before and they are not morally strong.

Programs with data races have **undefined behaviour**.

## PTX Interoperability

The axioms and relations of the Tile IR memory model are intended to be a strict weakening of the PTX memory model.

The Tile IR memory model is designed to allow communication with PTX threads. Release and acquire patterns in Tile IR will match up with acquire and release patterns in PTX to build PTX causality and Tile IR happens before.

The same is true for data races: data races between accesses in a Tile IR program and a PTX program will result in undefined behavior as if the data race were all in Tile IR.

## Hazards and Intuition

### Token Ordered Operations Reading from the Future

Store buffering and load buffering behavior are visible when using token ordered operations without token constraints.

### Race Hazards Within a Single Tile Block Thread

The definition of data race relies on two operations being in happens before order. In many programming languages this is always true within a single thread, but this is not the case in Tile IR. In Tile IR you can construct a data race when there are two accesses to the same location within a single tile block store (because of internal overlap in the destination tile), or with two accesses within a tile block thread to the same location which are not ordered by token order. This motivates the need for a `tile_block` scope, and is a hazard to be aware of in the language.

### Intuition on How to Use Token Ordering

When you have memory accesses to tiles with no overlap between them, it is generally safe to not order these with respect to each other in token order. When you use a release operation, you need to token-order all memory events that must stay before the release to the release itself. Similarly, when an acquire operation is used, all memory events which must remain after the acquire need to be token ordered after the acquire operation itself.

To reiterate: a user of Tile IR **cannot rely on program dependencies** of any form other than token dependencies to enforce ordering within a Tile Block Thread. The Tile IR toolchain can remove dependencies through any possible complex reasoning, breaking dependencies which appear to be in the program syntax.
