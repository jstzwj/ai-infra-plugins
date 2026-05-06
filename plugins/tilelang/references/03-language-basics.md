# TileLang Language Basics

## Table of Contents

- [1. Function Definition: @T.prim_func](#1-function-definition--tprim_func)
- [2. T.Tensor Type](#2-ttensor-type)
- [3. T.Buffer vs T.Tensor (Deprecation)](#3-tbuffer-vs-ttensor-deprecation)
- [4. T.Kernel Context Manager](#4-tkernel-context-manager)
- [5. Variable Bindings and Thread/Block Indices](#5-variable-bindings-and-threadblock-indices)
- [6. Buffer Indexing and Slicing](#6-buffer-indexing-and-slicing)
- [7. Arithmetic Operations](#7-arithmetic-operations)
- [8. Comparison Operations](#8-comparison-operations)
- [9. Conditional Statements](#9-conditional-statements)
- [10. T.reads and T.writes Annotations](#10-treads-and-twrites-annotations)
- [11. Python Compatibility in TileLang](#11-python-compatibility-in-tilelang)
- [12. Symbolic vs Dynamic Dimensions](#12-symbolic-vs-dynamic-dimensions)
- [13. T.ceildiv and Division](#13-tceildiv-and-division)
- [14. Function Structure Patterns](#14-function-structure-patterns)
- [15. Data Types](#15-data-types)
- [16. Loop Constructs in Detail](#16-loop-constructs-in-detail)
- [17. Warp and Warpgroup Operations](#17-warp-and-warpgroup-operations)
- [18. Atomic Operations](#18-atomic-operations)
- [19. Annotations and Pragmas](#19-annotations-and-pragmas)
- [20. Custom CUDA Source Integration](#20-custom-cuda-source-integration)
- [21. Mathematical Intrinsics](#21-mathematical-intrinsics)
- [22. Logical Operations](#22-logical-operations)
- [23. Random Number Generation](#23-random-number-generation)
- [24. Debug Operations](#24-debug-operations)

---

## 1. Function Definition: @T.prim_func

### 1.1 Basic Syntax

The `@T.prim_func` decorator marks a Python function as a TileLang primitive function. This decorator is re-exported from TVM's TIR scripting framework.

```python
from tvm.script import tir as T

@T.prim_func
def my_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # Kernel body
    ...
```

### 1.2 Parameters

Parameters are declared using Python type annotations with `T.Tensor(shape, dtype)`:

```python
@T.prim_func
def kernel(
    A: T.Tensor((128, 64), "float32"),        # Static shape
    B: T.Tensor((M, N), "float16"),            # Symbolic shape
    C: T.Tensor((1, N), "float16"),            # Batch dimension
    idx: T.int32,                               # Scalar parameter
):
    ...
```

#### Parameter Types

| Type | Syntax | Description |
|------|--------|-------------|
| Tensor | `T.Tensor(shape, dtype)` | N-dimensional tensor in global memory |
| StridedTensor | `T.StridedTensor(shape, strides, dtype)` | Tensor with explicit strides |
| Integer scalar | `T.int32` | 32-bit integer scalar |
| Pointer | `T.ptr("float32")` | Raw pointer with type |

### 1.3 Return Types

TileLang functions do not use explicit return statements for output. Instead, outputs are declared as tensor parameters and the `out_idx` parameter in `@tilelang.jit` specifies which parameters are outputs.

```python
@tilelang.jit(out_idx=[-1])  # Last parameter is the output
def matmul(M, N, K, BM, BN, BK):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),  # This is the output
    ):
        ...
    return kernel
```

### 1.4 Function Attributes

You can attach attributes to a prim_func using `T.func_attr`:

```python
@T.prim_func
def kernel(...):
    T.func_attr({"tir.noalias": True})
    ...
```

Common function attributes:
- `"tir.noalias"`: Indicates that pointer parameters do not alias
- `"global_symbol"`: The external name of the kernel

---

## 2. T.Tensor Type

### 2.1 Basic Usage

`T.Tensor` is the primary way to declare tensor parameters. It creates a typed buffer in global memory with contiguous row-major strides by default.

```python
# 1D tensor
A: T.Tensor((N,), "float32")

# 2D tensor
A: T.Tensor((M, K), "float16")

# 3D tensor
A: T.Tensor((B, M, K), "float16")

# Tensor with symbolic dimensions
A: T.Tensor((M, K), "float16")  # M and K are compile-time parameters
```

### 2.2 Shape Specification

Shapes can be:
- **Concrete integers**: `(128, 64)`
- **Symbolic variables**: `(M, K)` where M and K are function parameters
- **Mixed**: `(M, 64)` -- symbolic M with concrete 64
- **Scalar**: `(1,)` or just `1`

```python
@T.prim_func
def kernel(
    A: T.Tensor((128, 64), "float16"),      # Concrete
    B: T.Tensor((M, K), "float16"),          # Symbolic
    C: T.Tensor((M, 1), "float16"),          # Mixed
    D: T.Tensor((1,), "float32"),            # Scalar-like
):
    ...
```

### 2.3 Dtype Specification

The dtype parameter accepts string representations of data types:

```python
A: T.Tensor((M, K), "float32")      # Standard float32
A: T.Tensor((M, K), "float16")      # Half precision
A: T.Tensor((M, K), "bfloat16")     # Brain float
A: T.Tensor((M, K), "int32")        # 32-bit integer
A: T.Tensor((M, K), "int8")         # 8-bit integer
A: T.Tensor((M, K), "float8_e4m3fn") # FP8 E4M3
A: T.Tensor((M, K), "float8_e5m2")  # FP8 E5M2
```

### 2.4 Strided Tensors

For tensors with non-contiguous memory layouts, use `T.StridedTensor`:

```python
A: T.StridedTensor((M, K), (K, 1), "float16")  # Row-major with explicit strides
B: T.StridedTensor((M, K), (1, M), "float16")  # Column-major
```

### 2.5 Eager-Mode Type Annotations

In eager mode (`@tilelang.jit` without `@T.prim_func`), tensor shapes are annotated differently:

```python
@tilelang.jit
def kernel(A, B, block_M=64):
    M, N, K = T.const('M N K')
    A: T.Tensor[[M, K], "float16"]  # Note: list, not tuple
    B: T.Tensor[[K, N], "float16"]
    C = T.empty([M, N], "float16")
    ...
    return C
```

Note the use of list `[[M, K], ...]` instead of tuple `((M, K), ...)` in eager mode.

### 2.6 Creating Tensors from Addresses

```python
# Create a tensor from a raw address
ptr = T.address_of(some_buffer[i, j])
new_buf = T.make_tensor(ptr, (128, 64), "float16")

# Or with explicit strides
new_buf = T.make_tensor(ptr, (128, 64), "float16", strides=(64, 1))
```

---

## 3. T.Buffer vs T.Tensor (Deprecation)

`T.Buffer` is a deprecated alias for `T.Tensor`. It still works but emits a deprecation warning.

```python
# Deprecated (works but warns)
A: T.Buffer((M, K), "float16")

# Recommended
A: T.Tensor((M, K), "float16")
```

The `T.Buffer` proxy also supports bracket syntax:

```python
# Deprecated bracket syntax
A = T.Buffer[128, 64, "float16"]

# Recommended
A: T.Tensor((128, 64), "float16")
```

### Key Differences

| Feature | T.Tensor | T.Buffer (deprecated) |
|---------|----------|----------------------|
| Scope | Global by default | Global by default |
| Strides | Auto-computed (contiguous) | Explicit or auto |
| Recommended | Yes | No (deprecated) |
| JIT integration | Full | Partial |

---

## 4. T.Kernel Context Manager

### 4.1 Basic Syntax

`T.Kernel` launches a GPU kernel grid. It is the entry point for all GPU computation in TileLang.

```python
with T.Kernel(grid_x, grid_y, grid_z, threads=128) as (bx, by, bz):
    # Kernel body
    ...
```

### 4.2 Grid Dimensions

The first 1-3 positional arguments specify the grid dimensions (number of blocks in each dimension):

```python
# 1D grid
with T.Kernel(num_blocks_x, threads=128) as bx:
    ...

# 2D grid
with T.Kernel(num_blocks_x, num_blocks_y, threads=128) as (bx, by):
    ...

# 3D grid
with T.Kernel(num_blocks_x, num_blocks_y, num_blocks_z, threads=128) as (bx, by, bz):
    ...
```

For a single dimension, the block index is returned as a single variable (not a tuple):

```python
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    # bx is blockIdx.x
    ...
```

### 4.3 Thread Specification

The `threads` parameter specifies the number of threads per block:

```python
# 1D thread block (default)
with T.Kernel(grid_x, threads=128) as bx:
    ...

# 2D thread block
with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
    tx, ty = T.get_thread_bindings()
    ...

# 3D thread block
with T.Kernel(grid_x, threads=(32, 4, 2)) as bx:
    tx, ty, tz = T.get_thread_bindings()
    ...
```

Default thread count is 128 when `threads` is not specified.

### 4.4 Cluster Dimensions (SM90+)

For Hopper cluster launches, use `cluster_dims`:

```python
# 2-CTA cluster
with T.Kernel(grid_x, grid_y, threads=128,
              cluster_dims=(2, 1, 1)) as (bx, by):
    ...

# 1D cluster
with T.Kernel(grid_x, threads=128,
              cluster_dims=2) as bx:
    ...
```

Cluster dimensions enable cooperative operations across multiple CTAs (thread blocks) within the same cluster. Available on SM90+ (Hopper and later).

### 4.5 CPU Kernels

For CPU kernels, use `is_cpu=True`:

```python
with T.Kernel(T.ceildiv(N, 128), is_cpu=True) as (i,):
    # i is a loop iteration variable, not a block index
    ...
```

When `is_cpu=True`:
- No thread bindings are created
- The kernel runs serially on the CPU
- Thread-related functions should not be used

### 4.6 CUDA Source Prelude

For injecting custom CUDA code before the kernel:

```python
with T.Kernel(grid_x, threads=128,
              prelude="#include <my_header.cuh>") as bx:
    ...
```

### 4.7 Unpacking Bindings

For a 1D kernel, the binding is a single variable:

```python
with T.Kernel(num_blocks, threads=128) as bx:
    # bx is blockIdx.x
    pass
```

For multi-dimensional kernels, use tuple unpacking:

```python
with T.Kernel(nx, ny, threads=128) as (bx, by):
    # bx is blockIdx.x, by is blockIdx.y
    pass
```

---

## 5. Variable Bindings and Thread/Block Indices

### 5.1 Block Indices

Block indices are obtained from `T.Kernel`:

```python
with T.Kernel(nx, ny, nz, threads=128) as (bx, by, bz):
    # bx = blockIdx.x
    # by = blockIdx.y
    # bz = blockIdx.z
    ...
```

### 5.2 Thread Bindings

Thread indices must be explicitly obtained:

```python
with T.Kernel(grid, threads=128) as bx:
    # Get individual thread binding
    tx = T.get_thread_binding(0)   # threadIdx.x

    # Get all thread bindings
    tx, ty, tz = T.get_thread_bindings()
```

### 5.3 Thread and Block Extents

```python
# Get thread extent (blockDim)
tx_count = T.get_thread_extent(0)   # blockDim.x
ty_count = T.get_thread_extent(1)   # blockDim.y

# Get all thread extents
thread_extents = T.get_thread_extents()  # [blockDim.x, blockDim.y, blockDim.z]

# Get block extent (gridDim)
bx_count = T.get_block_extent(0)   # gridDim.x
by_count = T.get_block_extent(1)   # gridDim.y

# Get all block extents
block_extents = T.get_block_extents()  # [gridDim.x, gridDim.y, gridDim.z]
```

### 5.4 Total Threads

```python
frame = T.KernelLaunchFrame.Current()
num_threads = frame.get_num_threads()  # blockDim.x * blockDim.y * blockDim.z
```

### 5.5 Lane and Warp Information

```python
# Get lane index within warp (0-31 on NVIDIA, 0-63 on AMD)
lane = T.get_lane_idx()

# Get warp index within block
warp = T.get_warp_idx()

# Get warp index with synchronization
warp = T.get_warp_idx_sync()

# Get warp group index (4 warps per group on NVIDIA)
wg = T.get_warp_group_idx()
```

### 5.6 Scalar Variables

Use `T.alloc_var` for thread-private scalar variables:

```python
# Allocate with initialization
count = T.alloc_var("int32", 0)    # Initialized to 0
result = T.alloc_var("float32", 1.0)  # Initialized to 1.0

# Read/write
count[0] = count[0] + 1
val = result[0]
```

Note: `alloc_var` returns a 1-element buffer. Access with `[0]`.

---

## 6. Buffer Indexing and Slicing

### 6.1 Basic Indexing

Buffers are indexed using multi-dimensional subscript notation:

```python
# 1D buffer
val = buffer[i]
buffer[i] = val

# 2D buffer
val = buffer[i, j]
buffer[i, j] = val

# 3D buffer
val = buffer[b, i, j]
buffer[b, i, j] = val
```

### 6.2 Sub-Region Indexing for T.copy

When using `T.copy`, you can specify sub-regions by providing the starting indices:

```python
# Copy a tile starting at (row, col) from global to shared
T.copy(A[by * BM, k * BK], A_shared)

# The tile size is inferred from the destination buffer shape
# A_shared has shape (BM, BK), so A[by * BM, k * BK] selects a (BM, BK) region
```

### 6.3 Range Slicing

You can use Python-style slicing for explicit regions:

```python
# Explicit range (not commonly used; T.copy with start index is preferred)
A_shared[i * 16:(i + 1) * 16, :]
```

### 6.4 Negative Indices

Negative indexing follows Python conventions (wraps around):

```python
val = buffer[-1]   # Last element
```

### 6.5 Buffer Load Expressions

Reading from a buffer produces a `tir.BufferLoad` expression:

```python
val = A[i, j]  # tir.BufferLoad node
```

This can be used directly in arithmetic expressions:

```python
C[i, j] = A[i, j] + B[i, j]
```

---

## 7. Arithmetic Operations

TileLang supports standard Python arithmetic operators on buffer elements and scalar expressions:

### 7.1 Basic Arithmetic

```python
# Addition
c = a + b

# Subtraction
c = a - b

# Multiplication
c = a * b

# Division
c = a / b       # Floating-point division
c = a // b      # Integer division (floor)
c = T.floordiv(a, b)  # Explicit floor division

# Modulo
c = a % b

# Negation
c = -a
```

### 7.2 Compound Assignment

```python
C[i, j] += A[i, k] * B[k, j]    # Add-assign
C[i, j] -= value                  # Subtract-assign
C[i, j] *= scale                  # Multiply-assign
```

Note: Compound assignment like `+=` is handled by TVM's TIR and maps to atomic or regular operations depending on context.

### 7.3 Type Casting

```python
# Cast float16 to float32
val_f32 = T.cast(val_f16, "float32")

# Cast float32 to float16
val_f16 = T.cast(val_f32, "float16")

# Cast int to float
val_float = T.cast(val_int, "float32")
```

### 7.4 Min/Max

```python
# Element-wise minimum and maximum
val = T.min(a, b)
val = T.max(a, b)

# Example: ReLU activation
C[i, j] = T.max(C[i, j], 0)
```

### 7.5 Packed x2 Operations

For SIMD operations on packed pairs (float16x2, bfloat16x2, float32x2):

```python
# Packed add (both elements simultaneously)
result = T.add2(a_packed, b_packed)

# Packed multiply
result = T.mul2(a_packed, b_packed)

# Packed fused multiply-add: a * b + c
result = T.fma2(a_packed, b_packed, c_packed)

# Packed min/max
result = T.min2(a_packed, b_packed)
result = T.max2(a_packed, b_packed)

# Packed absolute value
result = T.abs2(a_packed)
```

---

## 8. Comparison Operations

Standard comparison operators are supported:

```python
# Equality
cond = a == b

# Inequality
cond = a != b

# Less than
cond = a < b

# Greater than
cond = a > b

# Less than or equal
cond = a <= b

# Greater than or equal
cond = a >= b
```

These produce boolean (or int1) expressions that can be used in conditionals:

```python
if idx < N:
    C[idx] = A[idx] + B[idx]
```

### Logical Operators

```python
# Logical AND (use Python's `and` or T.and)
cond = a < N and b < M

# Logical OR
cond = a < N or b < M

# Logical NOT
cond = not (a < N)
```

---

## 9. Conditional Statements

### 9.1 If/Else Statements

TileLang supports Python-style if/else within kernels:

```python
# Simple if
if idx < N:
    C[idx] = A[idx] + B[idx]

# If/else
if condition:
    result = value_a
else:
    result = value_b

# Nested conditions
if idx < N:
    if idx % 2 == 0:
        C[idx] = A[idx]
    else:
        C[idx] = B[idx]
```

### 9.2 T.if_then_else

For conditional expressions (ternary-like):

```python
result = T.if_then_else(condition, true_value, false_value)
```

### 9.3 Conditional within T.Parallel

```python
for i in T.Parallel(block_M):
    for j in T.Parallel(block_N):
        if i + row_offset < M and j + col_offset < N:
            C_local[i, j] = T.max(C_local[i, j], 0)
```

---

## 10. T.reads and T.writes Annotations

### 10.1 Read/Write Annotations

`T.reads` and `T.writes` annotate which buffer regions a block reads from and writes to. These are inherited from TVM's TIR framework:

```python
with T.block("matmul"):
    T.reads(A[i, k], B[k, j])
    T.writes(C[i, j])
    C[i, j] = C[i, j] + A[i, k] * B[k, j]
```

In typical TileLang usage, these annotations are inferred automatically and rarely need to be specified manually.

---

## 11. Python Compatibility in TileLang

### 11.1 Supported Python Features

TileLang supports a subset of Python that can be mapped to TIR:

| Feature | Supported | Notes |
|---------|-----------|-------|
| Arithmetic (+, -, *, /, %) | Yes | Maps to TIR arithmetic |
| Comparison (<, >, ==, !=) | Yes | Maps to TIR comparison |
| if/else | Yes | Maps to TIR IfThenElse |
| for loops | Yes | Via T.serial, T.Parallel, etc. |
| Function calls | Yes | TIR intrinsics and builtins |
| Type annotations | Yes | Used for tensor declarations |
| Variable assignment | Yes | Maps to TIR Let/Store |
| Compound assignment (+=, etc.) | Yes | Maps to load-modify-store |
| f-strings | No | Not supported in kernel context |
| List comprehension | No | Not supported |
| Class definitions | No | Not supported |
| Exception handling | No | Not supported |
| Lambda functions | No | Not supported |
| print() | Partial | Use T.print() instead |

### 11.2 Python Features That Work Differently

#### Loops

Python `for` loops must use TileLang loop constructs:

```python
# Wrong: Standard Python for loop
for i in range(N):
    ...

# Correct: TileLang serial loop
for i in T.serial(N):
    ...

# Correct: TileLang parallel loop
for i, j in T.Parallel(M, N):
    ...
```

#### Print

```python
# Wrong: Python print
print(value)

# Correct: TileLang print
T.print(value, "label:")
```

#### Range

TileLang does not use Python's `range()`. Instead, use TileLang loop constructs that directly accept bounds.

---

## 12. Symbolic vs Dynamic Dimensions

### 12.1 Compile-Time Parameters

Dimensions passed as function parameters to the outer JIT function are compile-time constants:

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, BM, BN, BK):  # M, N, K are compile-time
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), "float16"),  # Uses compile-time M, K
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        ...
    return kernel

kernel = matmul(1024, 1024, 1024, 128, 128, 32)  # Compile for specific sizes
```

### 12.2 Dynamic Dimensions with T.dynamic

For truly dynamic (runtime) dimensions, use `T.dynamic`:

```python
# Create a dynamic symbolic variable
N = T.dynamic("N", "int32")

# Create multiple dynamic variables at once
M, N, K = T.dynamic("M N K", "int32")
```

Note: `T.symbolic` is a deprecated alias for `T.dynamic`.

### 12.3 T.const in Eager Mode

In eager mode, `T.const` declares compile-time constants from runtime tensor shapes:

```python
@tilelang.jit
def kernel(A, B):
    M, N, K = T.const('M N K')  # Extracted from input tensor shapes
    A: T.Tensor[[M, K], "float16"]
    B: T.Tensor[[K, N], "float16"]
    ...
```

---

## 13. T.ceildiv and Division

### 13.1 T.ceildiv

`T.ceildiv(a, b)` computes the ceiling of `a / b`, which is essential for computing grid dimensions:

```python
# Number of blocks needed to cover N elements with block_size elements per block
num_blocks = T.ceildiv(N, block_size)

# Example: N=1000, block_size=128 -> ceildiv(1000, 128) = 8
```

This is equivalent to `(a + b - 1) // b` but is recognized by the TileLang compiler for optimization.

### 13.2 Other Division Operations

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Ceiling div | `T.ceildiv(a, b)` | ceil(a / b) |
| Floor div | `a // b` or `T.floordiv(a, b)` | floor(a / b) |
| True div | `a / b` | Floating-point division |
| Trunc div | `T.truediv(a, b)` | Truncating division |

---

## 14. Function Structure Patterns

### 14.1 Lazy Mode Pattern

The standard pattern for writing TileLang kernels:

```python
@tilelang.jit(out_idx=[-1])
def kernel_name(param1, param2, ...):
    @T.prim_func
    def prim_func_name(
        input1: T.Tensor((shape1), dtype1),
        input2: T.Tensor((shape2), dtype2),
        output: T.Tensor((shape3), dtype3),
    ):
        with T.Kernel(grid_dims, threads=num_threads) as (bx, by):
            # Allocate buffers
            shared_buf = T.alloc_shared(...)
            local_buf = T.alloc_fragment(...)

            # Initialize
            T.clear(local_buf)

            # Main computation loop
            for k in T.serial(T.ceildiv(K, BK)):
                T.copy(input1[...], shared_buf)
                T.gemm(shared_buf, ..., local_buf)

            # Write output
            T.copy(local_buf, output[...])

    return prim_func_name
```

### 14.2 Eager Mode Pattern

```python
@tilelang.jit
def kernel_name(input1, input2, tile_param=64):
    M, N, K = T.const('M N K')
    input1: T.Tensor[[M, K], "float16"]
    input2: T.Tensor[[K, N], "float16"]
    output = T.empty([M, N], "float16")

    with T.Kernel(..., threads=128) as (bx, by):
        ...

    return output
```

### 14.3 Multiple Kernels

A single function can contain multiple kernel launches that execute sequentially:

```python
@tilelang.jit(out_idx=[-1])
def matmul_relu(M, N, K, BM, BN, BK):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float16"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        # Intermediate buffer
        D = T.alloc_global((M, N), "float16")

        # First kernel: matmul
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            ...  # Compute D = A @ B

        # Second kernel: ReLU (executes after first kernel completes)
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            ...  # Compute C = relu(D)

    return kernel
```

### 14.4 Nested Functions with @T.macro

Macros allow code reuse within TileLang kernels:

```python
from tilelang.language.eager.builder import macro

@macro
def load_tile(src, dst, offset):
    T.copy(src[offset], dst)

@T.prim_func
def kernel(...):
    with T.Kernel(...) as bx:
        A_shared = T.alloc_shared((BM, BK), "float16")
        load_tile(A, A_shared, bx * BM)
```

---

## 15. Data Types

### 15.1 Available Data Types

| TileLang dtype | Bit Width | Description |
|----------------|-----------|-------------|
| `"float32"` | 32 | IEEE 754 single precision |
| `"float16"` | 16 | IEEE 754 half precision |
| `"bfloat16"` | 16 | Brain floating point |
| `"float64"` | 64 | IEEE 754 double precision |
| `"int64"` | 64 | 64-bit signed integer |
| `"int32"` | 32 | 32-bit signed integer |
| `"int16"` | 16 | 16-bit signed integer |
| `"int8"` | 8 | 8-bit signed integer |
| `"uint8"` | 8 | 8-bit unsigned integer |
| `"uint32"` | 32 | 32-bit unsigned integer |
| `"uint64"` | 64 | 64-bit unsigned integer |
| `"bool"` | 1 | Boolean |
| `"float8_e4m3fn"` | 8 | FP8 with E4M3 format |
| `"float8_e5m2"` | 8 | FP8 with E5M2 format |
| `"int4"` | 4 | 4-bit integer (sub-byte) |
| `"int2"` | 2 | 2-bit integer (sub-byte) |

### 15.2 dtype Module

```python
from tilelang.language import dtypes

# Access dtype objects
dtypes.float32   # "float32"
dtypes.float16   # "float16"
dtypes.bfloat16  # "bfloat16"
dtypes.int32     # "int32"

# Type casting function
dtypes.dtype("float32")  # Returns "float32"
```

### 15.3 Accumulation Types

For numerical stability, accumulate in higher precision:

| Input dtype | Recommended accum_dtype | Notes |
|-------------|------------------------|-------|
| `float16` | `float32` | Standard practice |
| `bfloat16` | `float32` | Standard practice |
| `float8_e4m3fn` | `float32` | FP8 inference |
| `int8` | `int32` | Quantized inference |
| `float32` | `float32` | No upcast needed |

---

## 16. Loop Constructs in Detail

### 16.1 T.serial

Sequential loop execution:

```python
for i in T.serial(n):
    # Body executes sequentially for i = 0, 1, ..., n-1
    ...

# With start and stop
for i in T.serial(start, stop):
    ...

# With step
for i in T.serial(start, stop, step=2):
    ...
```

### 16.2 T.Parallel

Parallel loop nest. Creates parallel for loops mapped to GPU threads:

```python
# 1D parallel loop
for i in T.Parallel(block_M):
    ...

# 2D parallel loop nest
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)

# With coalesced width hint
for i, j in T.Parallel(block_M, block_N, coalesced_width=16):
    ...

# With layout hint
for i, j in T.Parallel(block_M, block_N, loop_layout=my_fragment):
    ...
```

Parameters:
- `coalesced_width`: Hint for coalesced memory access pattern.
- `loop_layout`: A `Fragment` defining the iteration layout.
- `prefer_async`: Hint to prefer cp.async for copies in this loop.
- `annotations`: Custom loop annotations.

### 16.3 T.Pipelined

Software-pipelined loop for overlapping memory and compute:

```python
for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[...], A_shared)  # Load
    T.copy(B[...], B_shared)  # Load
    T.gemm(A_shared, B_shared, C_local)  # Compute
```

Parameters:
- `start`: Start of iteration range.
- `stop`: End of iteration range (when two args given).
- `num_stages`: Number of pipeline stages (0 = disabled).
- `order`: Execution order of statements.
- `stage`: Stage assignment for each statement.
- `sync`: Synchronization points.
- `group`: Statement grouping.

### 16.4 T.unroll

Unrolled loop for small iteration counts:

```python
for i in T.unroll(4):
    # Body is replicated 4 times
    ...

# With explicit unroll
for i in T.unroll(0, 8, explicit=True):
    ...

# With unroll factor
for i in T.unroll(0, 128, unroll_factor=4):
    ...
```

### 16.5 T.vectorized

Vectorized loop for memory operations:

```python
for i in T.vectorized(0, N):
    dst[i] = src[i]
```

### 16.6 T.Persistent

Persistent kernel loop for keeping the kernel resident on the GPU:

```python
for tile_idx in T.Persistent(
    domain=[total_tiles],
    wave_size=wave_size,
    index=tile_index,
    group_size=8,
):
    # Process tile
    ...
```

### 16.7 T.Serial / T.Unroll / T.Vectorized (Uppercase Aliases)

TileLang provides uppercase aliases for use as tile-level loop constructs:

```python
for i in T.Serial(n):          # Alias for T.serial(n)
for i in T.Unroll(0, n):       # Alias for T.unroll(0, n)
for i in T.Vectorized(0, n):   # Alias for T.vectorized(0, n)
```

---

## 17. Warp and Warpgroup Operations

### 17.1 Warp Shuffle

```python
# XOR shuffle: exchange value with lane (lane_id ^ delta)
result = T.shfl_xor(value, delta, width=32)

# Down shuffle: get value from lane (lane_id + delta)
result = T.shfl_down(value, delta, width=32)

# Up shuffle: get value from lane (lane_id - delta)
result = T.shfl_up(value, delta, width=32)

# Broadcast: get value from specific lane
result = T.shfl_sync(value, srcLane, width=32)
```

### 17.2 Warp Vote

```python
# Any: non-zero if any lane's predicate is true
result = T.any_sync(predicate, mask=0xFFFFFFFF)

# All: non-zero if all lanes' predicates are true
result = T.all_sync(predicate, mask=0xFFFFFFFF)

# Ballot: bitmask of lanes with true predicate
mask = T.ballot_sync(predicate, mask=0xFFFFFFFF)

# Ballot with full warp mask
mask = T.ballot(predicate)
```

### 17.3 Warp Match

```python
# Match any: bitmask of lanes with matching value
mask = T.match_any_sync(value, mask=0xFFFFFFFF)

# Match all: mask if all lanes agree, else 0
mask = T.match_all_sync(value, mask=0xFFFFFFFF)
```

### 17.4 Active Mask

```python
# Get bitmask of currently active lanes
mask = T.activemask()
```

### 17.5 Warp-Level Reduction

```python
# Sum across all threads in the warp
total = T.warp_reduce_sum(value)

# Max across all threads in the warp
max_val = T.warp_reduce_max(value)

# Min across all threads in the warp
min_val = T.warp_reduce_min(value)

# Bitwise AND across warp
and_val = T.warp_reduce_bitand(value)

# Bitwise OR across warp
or_val = T.warp_reduce_bitor(value)
```

### 17.6 Warpgroup Operations (SM90+)

```python
# Signal warpgroup readiness
T.warpgroup_arrive()

# Commit warpgroup batch
T.warpgroup_commit_batch()

# Wait for warpgroup batch
T.warpgroup_wait(num_mma)

# Fence accumulator registers
T.warpgroup_fence_operand(buffer, offset=0, num_regs=N, dtype="float32")
```

### 17.7 Elect

```python
# Elect exactly one lane within a thread group
is_leader = T.shuffle_elect(64)  # One leader per 64 threads
```

---

## 18. Atomic Operations

### 18.1 Basic Atomics

```python
# Atomic add
T.atomic_add(dst[idx], value)

# Atomic max
T.atomic_max(dst[idx], value)

# Atomic min
T.atomic_min(dst[idx], value)

# Atomic load
val = T.atomic_load(src[idx])

# Atomic store
T.atomic_store(dst[idx], value)
```

### 18.2 Extended Atomics

```python
# 64-bit atomic add (two 32-bit values packed)
T.atomic_addx2(dst[idx], value)

# 128-bit atomic add (four 32-bit values packed)
T.atomic_addx4(dst[idx], value)
```

---

## 19. Annotations and Pragmas

### 19.1 Layout Annotation

```python
# Annotate buffer layout for manual layout control
T.annotate_layout({C_frag: my_layout})
```

### 19.2 Swizzle Pattern

```python
# Enable 2D row swizzle for L2 cache optimization
T.use_swizzle(panel_size=10)

# Column swizzle
T.use_swizzle(panel_size=10, order="column")

# Disable swizzle
T.use_swizzle(panel_size=10, enable=False)
```

### 19.3 Min Blocks Per SM

```python
# Hint that at least 2 blocks should run per SM
T.annotate_min_blocks_per_sm(2)
```

### 19.4 L2 Cache Hit Ratio

```python
# Hint L2 cache hit ratio for global buffers (0.0 - 1.0)
T.annotate_l2_hit_ratio({A_global: 0.5, B_global: 0.5})
```

### 19.5 Safe Value Annotation

```python
# Specify a safe default value for out-of-bounds access
T.annotate_safe_value({C_frag: 0})
```

### 19.6 Restrict Buffers

```python
# Mark buffers as potentially aliasing (remove __restrict__)
T.annotate_restrict_buffers(A, B)
```

### 19.7 Register Control (SM90+)

```python
# Increase register allocation
T.inc_max_nreg(232)

# Decrease register allocation
T.dec_max_nreg(232)

# Producer register deallocation hint
T.annotate_producer_reg_dealloc(24)

# Consumer register allocation hint
T.annotate_consumer_reg_alloc(240)

# Disable warp group register allocation
T.disable_warp_group_reg_alloc()
```

---

## 20. Custom CUDA Source Integration

### 20.1 T.CUDASourceCodeKernel

Embed existing CUDA kernels within TileLang:

```python
cuda_source = """
__global__ void my_custom_kernel(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx] * 2.0f;
}
"""

with T.Kernel(T.ceildiv(N, 128), threads=128):
    T.CUDASourceCodeKernel(
        source_code_or_path=cuda_source,
        entry_name="my_custom_kernel",
    )
```

### 20.2 Import Source

```python
# Import C code as prelude
T.import_source("#include <my_header.cuh>")
```

---

## 21. Mathematical Intrinsics

### 21.1 Fast Math (Approximate)

```python
result = T.__exp(x)       # Fast approximate exp
result = T.__exp10(x)     # Fast approximate 10^x
result = T.__log(x)       # Fast approximate log
result = T.__log2(x)      # Fast approximate log2
result = T.__log10(x)     # Fast approximate log10
result = T.__sin(x)       # Fast approximate sin
result = T.__cos(x)       # Fast approximate cos
result = T.__tan(x)       # Fast approximate tan
```

### 21.2 IEEE-Compliant Math

```python
result = T.ieee_add(x, y, rounding_mode="rn")     # IEEE addition
result = T.ieee_sub(x, y, rounding_mode="rn")     # IEEE subtraction
result = T.ieee_mul(x, y, rounding_mode="rn")     # IEEE multiplication
result = T.ieee_fmaf(x, y, z, rounding_mode="rn") # IEEE fused multiply-add
result = T.ieee_frcp(x, rounding_mode="rn")       # IEEE reciprocal
result = T.ieee_fsqrt(x, rounding_mode="rn")      # IEEE square root
result = T.ieee_frsqrt(x)                          # IEEE reciprocal sqrt
result = T.ieee_fdiv(x, y, rounding_mode="rn")    # IEEE division
```

Rounding modes: `"rn"` (nearest), `"rz"` (toward zero), `"ru"` (toward +inf), `"rd"` (toward -inf).

### 21.3 Standard Math

These are inherited from TVM's TIR and available through the `T` namespace:

```python
result = T.sqrt(x)
result = T.rsqrt(x)
result = T.exp(x)
result = T.log(x)
result = T.log2(x)
result = T.sin(x)
result = T.cos(x)
result = T.tanh(x)
result = T.sigmoid(x)
result = T.abs(x)
result = T.ceil(x)
result = T.floor(x)
result = T.round(x)
result = T.pow(x, y)
```

---

## 22. Logical Operations

### 22.1 Buffer-Level Logical

```python
# Check if any element in buffer is non-zero
result = T.any_of(buffer)

# Check if all elements in buffer are non-zero
result = T.all_of(buffer)
```

---

## 23. Random Number Generation

```python
# Initialize RNG state
T.rng_init(seed, offset)

# Generate random integer
val = T.rng_rand()

# Generate random float [0, 1)
val = T.rng_rand_float()
```

---

## 24. Debug Operations

### 24.1 Print

```python
# Print a value
T.print(value, "label:")

# Print buffer element
T.print(A[i, j], "A[i,j]:")
```

### 24.2 Device Assert

```python
# Assert a condition on device
T.device_assert(condition, "Error message")

# Example: bounds check
T.device_assert(idx < N, "Index out of bounds")
```

### 24.3 Block/Thread Synchronization with Count

```python
# Block sync that returns count of threads with true predicate
count = T.syncthreads_count(predicate)

# Block sync that returns AND of all predicates
result = T.syncthreads_and(predicate)

# Block sync that returns OR of all predicates
result = T.syncthreads_or(predicate)
```
