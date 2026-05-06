# 10 — TensorIR Abstraction

## Overview

TensorIR is one of the core abstractions in Apache TVM, used to represent and optimize primitive tensor functions. The TensorIR codebase consists of two modules:

- **tirx** — Core IR definitions and lowering (PrimFunc, Buffer, SBlock, expressions, statements, lowering passes)
- **s_tir** (Schedulable TIR) — Schedule primitives, MetaSchedule, DLight, and tensor intrinsics

In TVMScript, both modules are accessed via `from tvm.script import tirx as T`.

---

## PrimFunc

`tirx::PrimFunc` is a low-level program representation containing:
- Loop nests
- Multi-dimensional buffer access
- Threading and vector instructions
- Tensor intrinsics

```python
from tvm.script import tirx as T

@T.prim_func
def vector_add(
    A: T.Buffer((128,), "float32"),
    B: T.Buffer((128,), "float32"),
    C: T.Buffer((128,), "float32"),
) -> None:
    for i in range(128):
        with T.sblock("C"):
            vi = T.axis.spatial(128, i)
            C[vi] = A[vi] + B[vi]
```

---

## Buffer Type

### T.Buffer(shape, dtype)
Multi-dimensional buffer with known shape and data type:

```python
A: T.Buffer((128, 128), "float32")  # 128x128 float32 buffer
B: T.Buffer((256,), "int32")        # 256-element int32 buffer
```

### T.alloc_buffer(shape, dtype)
Allocate intermediate buffer:

```python
Y = T.alloc_buffer((128, 128), dtype="float32")
```

### Buffer Access
```python
# Read
val = A[i, j]

# Write
A[i, j] = val

# Multi-dimensional
Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
```

---

## SBlock Annotation

`T.sblock` is the fundamental scheduling unit:

```python
with T.sblock("name"):
    # Block axes declarations
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(128, j)
    # Computation
    C[vi, vj] = A[vi, vj] + B[vi, vj]
```

A block can contain:
- Single computation statement
- Multiple computation statements with loops
- Opaque intrinsics (e.g., Tensor Core instructions)

---

## Block Axes

### T.axis.spatial(range, mapped_value)
Spatial iteration — directly maps to spatial region of output. Iterations are independent.

```python
vi = T.axis.spatial(128, i)
# vi maps to loop i, range is 128, spatial (independent iterations)
```

### T.axis.reduce(range, mapped_value)
Reduction iteration — accumulates results across iterations.

```python
vk = T.axis.reduce(128, k)
# vk is a reduction axis (e.g., k in matmul)
```

### T.axis.remap(pattern, values)
Sugar for declaring multiple axes at once:

```python
# "SSR" means: Spatial, Spatial, Reduce
vi, vj, vk = T.axis.remap("SSR", [i, j, k])

# Equivalent to:
# vi = T.axis.spatial(range_i, i)
# vj = T.axis.spatial(range_j, j)
# vk = T.axis.reduce(range_k, k)
```

### Axis Properties
- **Spatial**: axis directly corresponds to spatial region of output buffer. Different values produce independent results.
- **Reduce**: axis accumulates (reduces) values. Used for sum, product, etc.

---

## T.grid

Syntactic sugar for nested loops:

```python
# With T.grid
for i, j, k in T.grid(128, 128, 128):
    ...

# Equivalent to range
for i in range(128):
    for j in range(128):
        for k in range(128):
            ...
```

---

## T.init()

Initialization block for reduction operations:

```python
with T.sblock("Y"):
    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
    with T.init():
        Y[vi, vj] = T.float32(0)
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
```

The `T.init()` block specifies the initial value for reduction. It is executed when the first value of the reduce axis is encountered.

---

## Handle-based Parameters (Dynamic Shapes)

For functions with dynamic (symbolic) shapes:

```python
@T.prim_func
def dynamic_matmul(
    x: T.handle,      # opaque handle
    w: T.handle,
    z: T.handle,
):
    M, N, K = T.int64(), T.int64(), T.int64()  # symbolic sizes
    X = T.match_buffer(x, (M, K), "float32")
    W = T.match_buffer(w, (K, N), "float32")
    Z = T.match_buffer(z, (M, N), "float32")

    Y = T.alloc_buffer((M, N), "float32")
    for i, j, k in T.grid(M, N, K):
        with T.sblock("Y"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                Y[vi, vj] = T.float32(0)
            Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vk, vj]
    for i, j in T.grid(M, N):
        with T.sblock("Z"):
            vi, vj = T.axis.remap("SS", [i, j])
            Z[vi, vj] = Y[vi, vj]
```

### T.handle
Opaque pointer to external data. Used when buffer size is not known at function definition time.

### T.match_buffer
Binds a handle to a Buffer with specified shape and dtype. Extracts shape dimensions as symbolic variables.

---

## Complete Example: Matrix Multiply + ReLU

```python
from tvm.script import tirx as T

@T.prim_func
def mm_relu(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    Y = T.alloc_buffer((128, 128), dtype="float32")
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("Y"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                Y[vi, vj] = T.float32(0)
            Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(128, 128):
        with T.sblock("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

### Comparison with NumPy

| NumPy | TensorIR |
|-------|----------|
| `np.empty((128, 128), dtype="float32")` | `Y = T.alloc_buffer((128, 128), dtype="float32")` |
| `for i in range(128):` | `for i in range(128):` |
| `Y[i, j] = Y[i, j] + A[i, k] * B[k, j]` | `Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]` |
| N/A | `with T.sblock("Y"):` |
| N/A | `vi = T.axis.spatial(128, i)` |
| N/A | `with T.init(): Y[vi, vj] = T.float32(0)` |

---

## Block Properties

### Self-Contained
A block has all iteration information independent of external loops. This enables:
- Safe reordering of loops
- Parallelization of spatial axes
- Splitting/fusing loops
- Vectorization

### Validation
Block axis range is validated against external loops:

```python
# WRONG: loop range doesn't cover block range
for i in range(127):  # 127 < 128
    with T.sblock("C"):
        vi = T.axis.spatial(128, i)  # ERROR: size mismatch
        C[vi] = A[vi] + B[vi]
```

### Why Extra Information Matters
The additional block annotation is not needed for *executing* the program, but it is essential for *transforming* it safely. It enables the compiler to:
1. Verify that transformations are correct
2. Apply aggressive optimizations (reorder, parallelize, vectorize)
3. Generate correct target-specific code

---

## BufferRegion Access

Buffers can be accessed with:
- Single indices: `A[i]`
- Multi-dimensional indices: `A[i, j, k]`
- Slice regions: accessed through loop bounds

---

## Nested Blocks

Blocks can be nested for fused operations:

```python
for i, j in T.grid(128, 128):
    with T.sblock("outer"):
        vi, vj = T.axis.remap("SS", [i, j])
        # Nested computation
        temp = A[vi, vj] * T.float32(2.0)
        C[vi, vj] = temp + B[vi, vj]
```

---

## Key Takeaways

1. **PrimFunc** = low-level tensor function with loops, buffers, and compute
2. **SBlock** = fundamental scheduling unit with self-contained iteration info
3. **Block axes** = spatial (independent) vs reduce (accumulating)
4. **Dynamic shapes** supported via handles and match_buffer
5. Block annotations enable safe program transformations
