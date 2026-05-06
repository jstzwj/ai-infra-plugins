# NVIDIA cuTile — Tile-Based GPU Programming Model

NVIDIA cuTile is a high-level, array-based programming framework for CUDA that abstracts away thread-level programming. Instead of writing SIMT (Single Instruction, Multiple Threads) code, developers write tile-based kernels that operate on multidimensional tiles of data. The cuTile compiler then maps these tiles to GPU hardware resources efficiently.

## Activation

Activate this skill when the user asks about:
- NVIDIA cuTile, cuTile Python, cuda.tile, tile-based GPU programming
- Tile kernels, tile operations, tile programming model
- TileIR compiler, tileiras, TileIR bytecode
- GPU kernel development without SIMT programming
- Array-based CUDA programming, tile-level parallelism
- cuTile data types (bfloat16, tfloat32, float8_e4m3fn, float8_e5m2)
- cuTile AOT compilation, kernel export, calling conventions
- cuTile autotuning, performance hints, ByTarget
- cuTile memory model, atomic operations, memory ordering
- cuTile interoperability with CuPy, PyTorch, NumPy
- TiledView, tile space, element space

## Overview

cuTile Python provides a Python-based tile programming model for NVIDIA GPUs. Key characteristics:

- **Array-based model** — No raw pointers; all data accessed through bounds-checked arrays
- **Tile-centric operations** — Operations on immutable multidimensional tiles rather than individual elements
- **Block-level parallelism** — Abstracts thread management; no SIMT programming required
- **Python subset** — Familiar Python syntax with a restricted, compilable subset
- **JIT and AOT compilation** — Just-in-time via `ct.launch()` or ahead-of-time via `export_kernel()`
- **Zero-copy interoperability** — Works with CuPy, PyTorch, NumPy via DLPack/CUDA Array Interface

### Supported Hardware

| Compute Capability | GPU Families |
|---|---|
| 8.x | Ampere (A100), Ada Lovelace (L40) |
| 10.x | Hopper (H100) |
| 11.x | Blackwell (B200) |
| 12.x | Next-gen architectures |

### Requirements

- Linux x86_64/aarch64 or Windows x86_64
- NVIDIA Driver r580+
- Python 3.10–3.13
- CUDA Toolkit 13.1+ (or bundled via `pip install cuda-tile[tileiras]`)

## Quick Reference

### Basic Kernel Structure

```python
import cuda.tile as ct
import cupy as cp

@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    result = a_tile + b_tile
    ct.store(c, index=(pid,), tile=result)

# Launch
vector_size = 4096
tile_size = 16
a = cp.random.random(vector_size)
b = cp.random.random(vector_size)
c = cp.zeros_like(a)
ct.launch(cp.cuda.get_current_stream(),
          (ct.cdiv(vector_size, tile_size),),
          vector_add,
          (a, b, c, tile_size))
```

### Matrix Multiplication

```python
@ct.kernel
def matmul(X, Y, Out, TM: ct.Constant[int], TN: ct.Constant[int], TK: ct.Constant[int]):
    i, j = ct.bid(0), ct.bid(1)
    x_view = X.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    y_view = Y.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(x_view.num_tiles(1)):
        tx = x_view.load((i, k))
        ty = y_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    ct.store(Out, (i, j), acc.astype(Out.dtype))
```

### Common Patterns

```python
# Reduction
total = ct.sum(tile, axis=0)

# Conditional selection
result = ct.where(condition, x, y)

# Custom reduction
def my_add(a, b): return a + b
reduced = ct.reduce(tile, axis=0, init=0, fn=my_add)

# Atomic operations
ct.atomic_add(array, indices, values,
              memory_order=ct.MemoryOrder.RELEASE,
              memory_scope=ct.MemoryScope.DEVICE)
```

## Core Architecture

```
cuda.tile
├── Kernel System
│   ├── @ct.kernel          — Kernel decorator
│   ├── ct.launch()         — JIT kernel launch
│   └── ct.Constant[T]      — Compile-time constant parameters
├── Data Model
│   ├── Array               — Global device array (host-allocated)
│   ├── Tile                — Immutable tile (compile-time shape, power-of-2 dims)
│   ├── TiledView           — Tile space view of an array
│   └── DType               — 16 data types (bool, int, uint, float, bf16, tf32, fp8)
├── Operations
│   ├── Load/Store          — load, store, gather, scatter
│   ├── Factory             — zeros, ones, arange, full
│   ├── Shape               — reshape, permute, transpose, broadcast_to, cat, expand_dims
│   ├── Reduction           — sum, max, min, prod, argmax, argmin, reduce
│   ├── Scan                — cumsum, cumprod, scan
│   ├── Matmul              — mma, matmul
│   ├── Math                — 30+ elementwise math functions
│   ├── Bitwise             — and, or, xor, shift, not
│   ├── Comparison          — 6 comparison operators
│   ├── Selection           — where, extract
│   └── Atomic              — cas, xchg, add, max, min, and, or, xor
├── Memory Model
│   ├── MemoryOrder         — WEAK, RELAXED, ACQUIRE, RELEASE, ACQ_REL
│   └── MemoryScope         — BLOCK, DEVICE, SYS
├── Performance
│   ├── ByTarget            — Architecture-specific configuration
│   ├── Hints               — latency, allow_tma on load/store
│   └── tune                — exhaustive_search, TuningResult
├── Compilation
│   ├── export_kernel()     — AOT compilation to cubin/bytecode
│   └── KernelSignature     — Parameter constraints and calling conventions
└── Metaprogramming
    ├── static_assert()     — Compile-time assertion
    ├── static_eval()       — Compile-time Python evaluation
    └── static_iter()       — Compile-time iteration
```

## Reference Chapters

See the `references/` directory for comprehensive documentation:

| # | Chapter | Description |
|---|---------|-------------|
| 01 | [Overview & Architecture](references/01-overview-and-architecture.md) | cuTile design philosophy, programming model, relationship to CUDA |
| 02 | [Getting Started](references/02-getting-started.md) | Installation, prerequisites, first kernel, testing, profiling |
| 03 | [Data Model — Arrays](references/03-data-model-arrays.md) | Global arrays, strided layout, shape, slicing, DLPack/CAI interop |
| 04 | [Data Model — Tiles & Scalars](references/04-data-model-tiles-scalars.md) | Tiles, scalars, immutability, tile space, element space |
| 05 | [Data Types Reference](references/05-data-types-reference.md) | All 16 dtypes, DType class, arithmetic promotion table |
| 06 | [Execution Model](references/06-execution-model.md) | Kernels, functions, control flow, constantness, object model |
| 07 | [Load & Store Operations](references/07-load-and-store.md) | load, store, gather, scatter, TMA hints, padding modes |
| 08 | [Tiled Views](references/08-tiled-views.md) | TiledView class, tiled_view(), tile space, num_tiles, padding |
| 09 | [Tile Factory Operations](references/09-tile-factory.md) | zeros, ones, arange, full — creating tiles from scratch |
| 10 | [Shape & DType Operations](references/10-shape-and-dtype.md) | reshape, permute, transpose, broadcast_to, cat, expand_dims, astype, bitcast |
| 11 | [Reduction Operations](references/11-reduction-operations.md) | sum, max, min, prod, argmax, argmin, custom reduce |
| 12 | [Scan Operations](references/12-scan-operations.md) | cumsum, cumprod, custom scan (inclusive prefix) |
| 13 | [Matrix Operations](references/13-matrix-operations.md) | mma (matrix multiply-accumulate), matmul |
| 14 | [Elementwise Math Operations](references/14-math-operations.md) | 30+ math functions: exp, log, sqrt, trig, comparison, selection |
| 15 | [Bitwise & Comparison Operations](references/15-bitwise-comparison.md) | Bitwise ops, comparison ops, selection (where, extract) |
| 16 | [Atomic Operations](references/16-atomic-operations.md) | All 8 atomic ops, memory_order, memory_scope semantics |
| 17 | [Memory Model](references/17-memory-model.md) | Memory ordering, scopes, synchronization between blocks |
| 18 | [Compilation & AOT Export](references/18-compilation-and-export.md) | export_kernel, KernelSignature, calling conventions, cubin/bytecode |
| 19 | [Performance Tuning & Autotuning](references/19-performance-tuning.md) | ByTarget, load/store hints, exhaustive_search, TuningResult |
| 20 | [Metaprogramming](references/20-metaprogramming.md) | static_assert, static_eval, static_iter, compile-time computation |
| 21 | [Debugging & Error Handling](references/21-debugging-and-errors.md) | Exception types, environment variables, Nsight Compute profiling |
| 22 | [Interoperability](references/22-interoperability.md) | CuPy, PyTorch, NumPy, DLPack, CUDA Array Interface, SIMT interop |
| 23 | [Utility Functions](references/23-utility-functions.md) | printf, print, assert_, cdiv, bid, num_blocks, num_tiles |
| 24 | [Release Notes](references/24-release-notes.md) | Version history 1.0.0–1.3.0, features, bug fixes, ABI changes |

## Important Notes

- cuTile uses **immutable tiles** — all operations create new tiles rather than modifying in place
- **Tile dimensions must be powers of 2** — each dimension of a tile shape must be a power of 2
- **Constant embedding** via `ct.Constant[T]` generates specialized kernels for each unique value
- **Block-level parallelism only** — no thread-level programming or explicit intra-block synchronization
- The **Python subset** restricts available features — no exceptions, coroutines, or dynamic attributes
- **Kernel arguments must not alias** — passing overlapping arrays to a kernel is undefined behavior
- `bfloat16` and `tfloat32` have limited arithmetic promotion — cannot mix with `float16` or FP8 types
