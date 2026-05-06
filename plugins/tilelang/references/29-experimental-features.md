# TileLang Experimental Features Reference

The TileLang experimental module provides access to cutting-edge and specialized features that extend the core language. These modules cover sparse computations, custom intrinsics, random number generation, programmatic dependent launch (PDL), cluster operations, warpgroup management, fast math, fill operations, parser utilities, TIR integration, symbolic computation, frame management, and buffer overrides.

## Table of Contents

1. [Experimental Module Overview](#experimental-module-overview)
2. [Sparse GEMM](#sparse-gemm)
3. [Custom Intrinsics](#custom-intrinsics)
4. [Random Number Generation](#random-number-generation)
5. [PDL: Programmatic Dependent Launch](#pdl-programmatic-dependent-launch)
6. [Cluster Operations](#cluster-operations)
7. [Warpgroup Operations](#warpgroup-operations)
8. [Fast Math](#fast-math)
9. [Fill Operations](#fill-operations)
10. [Parser Module](#parser-module)
11. [TIR Integration](#tir-integration)
12. [Symbolic Computation](#symbolic-computation)
13. [Frame Management](#frame-management)
14. [Buffer Overrides](#buffer-overrides)

---

## Experimental Module Overview

The experimental module at `tilelang.language.experimental` provides access to features that are either:
- Architecture-specific (SM90+, CDNA, etc.)
- Under active development with potential API changes
- Specialized use cases (sparse GEMM, custom intrinsics)

**Module Location:** `tilelang/language/experimental/`

**Available Submodules:**

| Module | Import Path | Description |
|--------|-------------|-------------|
| Sparse GEMM | `tilelang.language.experimental.gemm_sp` | Structured sparse matrix multiplication |
| Custom Intrinsics | `tilelang.language.customize` | Custom operations and utilities |
| Random | `tilelang.language.random` | GPU random number generation |
| PDL | `tilelang.language.pdl` | Programmatic Dependent Launch |
| Cluster | `tilelang.language.cluster` | Thread cluster operations |
| Warpgroup | `tilelang.language.warpgroup` | Warp group specialization |
| Fast Math | `tilelang.language.fastmath` | Fast math approximations |
| Fill | `tilelang.language.fill_op` | Buffer fill and clear operations |
| Parser | `tilelang.language.parser` | Source parsing utilities |
| TIR | `tilelang.language.tir` | TIR integration |
| Symbolics | `tilelang.language.symbolics` | Symbolic/dynamic variable creation |
| Frame | `tilelang.language.frame` | Frame stack management |
| Overrides | `tilelang.language.overrides` | Buffer view and reshape |

---

## Sparse GEMM

Sparse GEMM operations compute `C = A_sparse @ B` where `A_sparse` is a 2:4 structured sparse matrix. NVIDIA Ampere+ GPUs provide hardware support for this sparsity pattern through the sparse Tensor Core.

### gemm_sp

```python
from tilelang.language.experimental.gemm_sp import gemm_sp

def gemm_sp(
    A_sparse: BufferLikeType | tir.Var,
    E: BufferLikeType | tir.Var,
    B: BufferLikeType | tir.Var,
    C: BufferLikeType | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
) -> tir.Call
```

Computes a sparse GEMM where `A_sparse` contains only the non-zero elements of a 2:4 structured sparse matrix, and `E` contains the metadata that describes the sparsity pattern.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_sparse` | `BufferLikeType` | required | Dense values of the sparse matrix (non-zero elements only) |
| `E` | `BufferLikeType` | required | Sparsity metadata (element index) |
| `B` | `BufferLikeType` | required | Dense weight matrix |
| `C` | `BufferLikeType` | required | Output matrix |
| `transpose_A` | `bool` | `False` | Whether A is transposed |
| `transpose_B` | `bool` | `False` | Whether B is transposed |
| `policy` | `GemmWarpPolicy` | `Square` | Warp partition policy |
| `clear_accum` | `bool` | `False` | Clear accumulator before computation |
| `k_pack` | `int` | `1` | Number of K dimensions packed per warp |
| `wg_wait` | `int` | `0` | Warp group wait count |

**Shape Constraints:**

- For 2:4 structured sparsity, `K_A * 2 == K_B` (the sparse K dimension is half the dense K)
- `C` shape is `(M, N)`
- `A_sparse` shape is `(M, K_A)` or `(K_A, M)` depending on `transpose_A`
- `B` shape is `(K_B, N)` or `(N, K_B)` depending on `transpose_B`

**Example:**

```python
import tilelang as tl
import tilelang.language as T

@T.prim_func
def sparse_matmul(
    A_sparse: T.Buffer((128, 32), "float16"),   # 2:4 sparse, K_sparse = 32
    E: T.Buffer((128, 32), "int32"),              # Sparsity metadata
    B: T.Buffer((64, 64), "float16"),             # Dense, K = 64
    C: T.Buffer((128, 64), "float16"),
):
    with T.Kernel(128, 64) as (i, j):
        gemm_sp(A_sparse, E, B, C)
```

### gemm_sp_v2

```python
from tilelang.language.experimental.gemm_sp import gemm_sp_v2

def gemm_sp_v2(
    A_sparse: BufferLikeType | tir.Var,
    E: BufferLikeType | tir.Var,
    B: BufferLikeType | tir.Var,
    C: BufferLikeType | tir.Var,
    transpose_A: bool = False,
    transpose_B: bool = False,
    transpose_E: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
) -> tir.Call
```

An enhanced variant of `gemm_sp` with additional parameters for more flexible sparse GEMM compilation. This version uses the `tl.tileop.gemm_sp_py` intrinsic path for faster compilation.

**Additional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transpose_E` | `bool` | `False` | Whether E metadata is transposed |

**Key Differences from gemm_sp:**

| Feature | gemm_sp | gemm_sp_v2 |
|---------|---------|------------|
| Intrinsic | `tl.tileop.gemm_sp` | `tl.tileop.gemm_sp_py` |
| Metadata transpose | Not supported | Supported via `transpose_E` |
| Stride/offset handling | Implicit | Explicit stride and offset |
| Compilation speed | Standard | Faster |

### make_cutlass_metadata_layout

Located in `tilelang.layout.gemm_sp`, this function creates the metadata layout required for CUTLASS-style sparse GEMM operations. It generates the element index (E) tensor that describes the 2:4 sparsity pattern.

The metadata layout follows the CUTLASS convention where each metadata element encodes which 2 out of every 4 consecutive elements are non-zero. This is required for the sparse Tensor Core instructions on NVIDIA Ampere+ GPUs.

---

## Custom Intrinsics

The `tilelang.language.customize` module provides custom operations frequently used in tensor programming.

### dp4a: Dot Product with Accumulation

```python
def dp4a(A: Buffer, B: Buffer, C: Buffer) -> PrimExpr
```

Performs a 4-element integer dot product with accumulation: `C += A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3]`. Maps to the CUDA `dp4a` instruction.

**Requirements:** INT8 input types, INT32 accumulator.

**Example:**

```python
@T.prim_func
def int8_dp4a(
    A: T.Buffer((4,), "int8"),
    B: T.Buffer((4,), "int8"),
    C: T.Buffer((1,), "int32"),
):
    dp4a(A, B, C)
```

### clamp

```python
def clamp(dst: PrimExpr, min_val: PrimExpr, max_val: PrimExpr) -> PrimExpr
```

Clamps the input value to the range `[min_val, max_val]`.

```python
result = clamp(value, 0.0, 1.0)  # Clamp to [0, 1]
```

### reshape

```python
def reshape(src: Buffer, shape: ShapeType) -> Buffer
```

Reshapes a buffer to a new shape without copying data. The total number of elements must match.

```python
flat = reshape(matrix_2d, (1024,))       # (32, 32) -> (1024,)
matrix = reshape(flat, (32, 32))          # (1024,) -> (32, 32)
```

### view

```python
def view(src: Buffer, shape: ShapeType | None = None, dtype: DType | None = None) -> Buffer
```

Creates a view of a buffer with optional new shape and dtype. The bit count must be preserved.

```python
# View FP16 tensor as INT16
int_view = view(fp16_buffer, dtype="int16")

# View 1D tensor as 2D
matrix_view = view(flat_buffer, shape=(32, 64))
```

### loop_break

```python
def loop_break() -> PrimExpr
```

Breaks out of the current loop. Maps to the `tl.loop_break` intrinsic.

### Atomic Operations

The customize module also re-exports all atomic operations for convenience:

```python
from tilelang.language.customize import (
    atomic_add, atomic_max, atomic_min,
    atomic_addx2, atomic_addx4,
    atomic_load, atomic_store,
)
```

---

## Random Number Generation

The `tilelang.language.random` module provides GPU random number generation using CUDA's cuRAND library.

### rng_init

```python
def rng_init(
    seed,
    seq=None,
    off=0,
    generator="curandStatePhilox4_32_10_t",
) -> tir.PrimExpr
```

Initializes a cuRAND random number generator state.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `PrimExpr` | required | Random seed value |
| `seq` | `PrimExpr or None` | `None` | Sequence number (auto-generated from thread ID if None) |
| `off` | `PrimExpr` | `0` | Offset for parallel generation |
| `generator` | `str` | `"curandStatePhilox4_32_10_t"` | Generator type |

**Supported Generators:**

| Generator | Description | Quality | Performance |
|-----------|-------------|---------|-------------|
| `curandStatePhilox4_32_10_t` | Philox4x32-10 | Excellent | Good |
| `curandStateXORWOW_t` | XORWOW | Good | Excellent |
| `curandStateMRG32k3a_t` | MRG32k3a | Excellent | Moderate |

**Example:**

```python
@T.prim_func
def random_kernel(output: T.Buffer((1024,), "float32")):
    with T.Kernel(1024) as bx:
        # Initialize RNG with thread-specific sequence
        rng_init(42)  # Uses thread ID for seq

        # Generate random values
        for i in T.serial(4):
            val = rng_rand_float(32, "uniform")
            output[bx * 4 + i] = val
```

### rng_rand

```python
def rng_rand() -> tir.PrimExpr
```

Generates a 32-bit unsigned random integer. Returns type `uint32`.

### rng_rand_float

```python
def rng_rand_float(bit: int = 32, dist: str = "uniform") -> tir.PrimExpr
```

Generates a random floating-point number.

**Parameters:**

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `bit` | `int` | `32` | `32` or `64` |
| `dist` | `str` | `"uniform"` | `"uniform"` or `"normal"` |

**Return Types:**

| bit | dist | Return Type |
|-----|------|-------------|
| 32 | uniform | `float32` in [0, 1) |
| 32 | normal | `float32` (standard normal) |
| 64 | uniform | `float64` in [0, 1) |
| 64 | normal | `float64` (standard normal) |

---

## PDL: Programmatic Dependent Launch

The `tilelang.language.pdl` module provides operations for Programmatic Dependent Launch, a mechanism for overlapping kernel execution on NVIDIA Hopper+ GPUs.

### pdl_trigger

```python
def pdl_trigger() -> tir.PrimExpr
```

Issues a PDL trigger that signals the completion of a dependent launch phase. This allows the GPU scheduler to begin executing a dependent kernel before the current kernel fully completes.

### pdl_sync

```python
def pdl_sync() -> tir.PrimExpr
```

Issues a PDL synchronization point that waits for all previously triggered PDL operations to complete.

**Usage Pattern:**

```python
@T.prim_func
def pdl_producer(output: T.Buffer((1024,), "float32")):
    with T.Kernel(1024) as bx:
        output[bx] = float(bx)
        pdl_trigger()  # Signal completion to consumer
```

---

## Cluster Operations

The `tilelang.language.cluster` module provides operations for NVIDIA Thread Block Clusters, available on Hopper (SM90+) and later architectures. Clusters allow multiple thread blocks to cooperate through shared memory.

### Barrier Operations

| Function | Description |
|----------|-------------|
| `cluster_arrive_relaxed()` | Issue `barrier.cluster.arrive.relaxed.aligned` |
| `cluster_arrive()` | Issue `barrier.cluster.arrive.aligned` |
| `cluster_wait()` | Issue `barrier.cluster.wait.aligned` |
| `cluster_sync()` | Full cluster barrier (arrive + wait) |

### Query Operations

| Function | Return Type | Description |
|----------|-------------|-------------|
| `block_rank_in_cluster()` | `int32` | 1-D rank of calling CTA within its cluster |

### Cluster Launch Control (CLC)

| Function | Description |
|----------|-------------|
| `clc_try_cancel(result, mbarrier)` | Single-CTA CLC query |
| `clc_try_cancel_multicast(result, mbarrier)` | Cluster-wide multicast CLC query |
| `clc_is_canceled(result)` | Check if CLC successfully canceled |
| `clc_get_first_ctaid_x(result)` | Get X coordinate of first CTA |
| `clc_get_first_ctaid_y(result)` | Get Y coordinate of first CTA |
| `clc_get_first_ctaid_z(result)` | Get Z coordinate of first CTA |

**Example -- Cluster Shared Memory Access:**

```python
@T.prim_func
def cluster_example(data: T.Buffer((256,), "float32")):
    with T.Kernel(2, 128, cluster_dims=(2, 1, 1)) as (bx, tx):
        shared = T.alloc_shared((128,), "float32")
        shared[tx] = data[bx * 128 + tx]
        cluster_sync()  # Synchronize across cluster
        # Now can access shared memory from neighboring CTAs
```

---

## Warpgroup Operations

The `tilelang.language.warpgroup` module provides warp group specialization for Hopper (SM90+) GPUs. Warp specialization allows different warp groups within a thread block to execute different code paths.

### WarpSpecialize

```python
def WarpSpecialize(*warp_group_idx) -> WarpSpecializeFrame
```

Creates a warp group specialization frame where only the specified warp groups execute the enclosed code.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `warp_group_idx` | `int` | One or more warp group indices (0, 1, 2, 3) |

**Warp Group Layout:**
- Each warp group consists of 128 threads (4 warps)
- For a 256-thread block: warp group 0 covers threads 0-127, warp group 1 covers threads 128-255

**Usage:**

```python
@T.prim_func
def warp_specialized_kernel(
    A: T.Buffer((128, 64), "float16"),
    B: T.Buffer((64, 64), "float16"),
    C: T.Buffer((128, 64), "float16"),
):
    with T.Kernel(128, 256) as (bx, tx):
        shared_a = T.alloc_shared((128, 64), "float16")
        shared_b = T.alloc_shared((64, 64), "float16")

        with T.ws(0):
            # Warp group 0: Load A from global to shared
            for i in T.serial(128):
                shared_a[i, tx % 64] = A[i, tx % 64]

        with T.ws(1):
            # Warp group 1: Load B from global to shared
            for i in T.serial(64):
                shared_b[i, tx % 64] = B[i, tx % 64]

        T.sync_threads()

        # All warps: Compute
        # ...
```

**Shorthand:** `T.ws` is an alias for `T.WarpSpecialize`.

---

## Fast Math

The `tilelang.language.fastmath` module provides fast (but potentially less accurate) mathematical function approximations using the `--use_fast_math` CUDA compiler flag.

### Available Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `__log(x)` | `(PrimExpr) -> PrimExpr` | Natural logarithm |
| `__log2(x)` | `(PrimExpr) -> PrimExpr` | Base-2 logarithm |
| `__log10(x)` | `(PrimExpr) -> PrimExpr` | Base-10 logarithm |
| `__sin(x)` | `(PrimExpr) -> PrimExpr` | Sine |
| `__cos(x)` | `(PrimExpr) -> PrimExpr` | Cosine |
| `__tan(x)` | `(PrimExpr) -> PrimExpr` | Tangent |
| `__exp(x)` | `(PrimExpr) -> PrimExpr` | Exponential (e^x) |
| `__exp10(x)` | `(PrimExpr) -> PrimExpr` | Exponential (10^x) |

**Usage:**

```python
from tilelang.language.fastmath import __exp, __log

@T.prim_func
def softmax_approx(input: T.Buffer((128,), "float32"), output: T.Buffer((128,))):
    with T.Kernel(128) as bx:
        # Use fast math approximations
        val = input[bx]
        output[bx] = __exp(val)  # Fast exp approximation
```

**Accuracy vs Performance:**

| Function | Max ULP Error | Performance Gain |
|----------|---------------|------------------|
| `__sin` | ~2 ULP | ~2x |
| `__cos` | ~2 ULP | ~2x |
| `__exp` | ~2 ULP | ~3x |
| `__log` | ~2 ULP | ~2x |

---

## Fill Operations

The `tilelang.language.fill_op` module provides operations to fill or clear buffer regions.

### fill

```python
def fill(buffer: BufferLikeType, value: tir.PrimExpr) -> tir.PrimExpr
```

Fills a buffer or buffer region with a specified value. Maps to the `tl.tileop.fill` intrinsic.

**Supported Input Types:**

| Input Type | Behavior |
|------------|----------|
| `tir.Buffer` | Fills entire buffer |
| `tir.BufferRegion` | Fills the specified region |
| `tir.BufferLoad` | Fills the loaded region |
| `tir.Var` (with let value) | Resolves to underlying buffer |

**Example:**

```python
@T.prim_func
def fill_example(buf: T.Buffer((128, 64), "float32")):
    with T.Kernel(1) as bx:
        shared = T.alloc_shared((128, 64), "float32")
        # Fill entire buffer with a value
        fill(shared, 1.0)

        # Fill a sub-region with a different value
        fill(shared[0:32, 0:32], 0.0)
```

### clear

```python
def clear(buffer: BufferLikeType) -> tir.PrimExpr
```

Clears a buffer by filling it with zeros. Equivalent to `fill(buffer, 0)`.

```python
clear(shared)              # Zero entire buffer
clear(shared[0:32, :])    # Zero first 32 rows
```

---

## Parser Module

The parser module (`tilelang.language.parser`) provides internal utilities for parsing TileLang source code. This module is primarily used by the eager mode infrastructure.

**Key Functionality:**
- Source code tokenization
- AST construction from TileLang DSL code
- Error reporting with source location information

---

## TIR Integration

The `tilelang.language` module re-exports the TVM TIR (Tensor Intermediate Representation) builder as the foundation for kernel construction. All TileLang DSL constructs ultimately generate TIR statements.

**Key TIR Constructs Used:**

| Construct | TileLang Equivalent | Description |
|-----------|---------------------|-------------|
| `tir.prim_func` | `@T.prim_func` | Function definition |
| `tir.Buffer` | `T.Tensor` / `T.Buffer` | Buffer declaration |
| `tir.BufferStore` | `buf[i] = value` | Buffer write |
| `tir.BufferLoad` | `buf[i]` | Buffer read |
| `tir.For` | `for i in T.serial(N)` | Loop |
| `tir.If` | `if cond:` | Conditional |
| `tir.evaluate` | `T.evaluate(expr)` | Expression evaluation |
| `tir.call_intrin` | Internal use | Intrinsic function call |

---

## Symbolic Computation

The `tilelang.language.symbolics` module provides helpers for creating symbolic variables used in dynamic shape expressions.

### dynamic

```python
def dynamic(name: str, dtype: DType = "int32") -> tir.Var | tuple[tir.Var, ...]
```

Creates TIR dynamic symbolic variables for use in kernel definitions with runtime-determined shapes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Variable name(s), comma or space separated |
| `dtype` | `DType` | `"int32"` | Data type for the variable |

**Multi-variable Declaration:**

```python
# Create single variable
M = T.dynamic("M")

# Create multiple variables (comma-separated)
M, N, K = T.dynamic("M, N, K")

# Create multiple variables (space-separated)
M, N, K = T.dynamic("M N K")
```

**Usage in Dynamic Kernels:**

```python
@T.prim_func
def dynamic_matmul(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float16"),
):
    # M, N, K are symbolic and resolved at compile time
    with T.Kernel(M, N) as (i, j):
        for k in T.serial(K):
            C[i, j] += A[i, k] * B[k, j]

M, N, K = T.dynamic("M, N, K")
```

### symbolic (Deprecated)

```python
@deprecated("T.symbolic(...)", "T.dynamic(...)", "v0.1.9")
def symbolic(name: str, dtype: DType = "int32") -> tir.Var | tuple[tir.Var, ...]
```

Deprecated alias for `dynamic`. Will be removed in v0.1.9.

---

## Frame Management

The `tilelang.language.frame` module manages the TIR let-frame stack for variable binding and scope tracking.

### FrameStack

```python
class FrameStack:
    def __init__(self)
    def push(self, item)
    def pop(self)
    def get_value(self, var)
    def has_value(self, var) -> bool
    def top(self)
```

A stack-like container that tracks active let-frame bindings. Each entry maps a `tir.Var` to its bound value.

**Thread Safety:** The frame stack uses thread-local storage (`threading.local()`) to avoid cross-thread interference.

### LetFrame

```python
@register_object("script.ir_builder.tir.LetFrame")
class LetFrame(TIRFrame)
```

Extended `LetFrame` that tracks variable bindings in the global frame stack. When entered, it pushes the binding onto the stack; when exited, it pops the binding.

**Key Features:**
- Automatically converts block `BufferLoad` with vectorized indices to `BufferRegion`
- Provides class methods for querying current bindings

### Utility Functions

```python
def has_let_value(var: Var) -> bool
def get_let_value(var: Var) -> PrimExpr | None
```

Check and retrieve the value bound to a variable in the current frame stack.

---

## Buffer Overrides

Buffer override operations are provided through the `tilelang.language.customize` module:

### reshape and view

As described in [Custom Intrinsics](#custom-intrinsics), these operations create new buffer views without copying data.

**Key Operations:**

| Operation | Description | Bit Preservation |
|-----------|-------------|------------------|
| `reshape(src, shape)` | View with new shape | Yes (same dtype) |
| `view(src, shape, dtype)` | View with optional new shape and dtype | Yes (total bits preserved) |

**Use Cases:**

1. **Data Type Reinterpretation:** View FP16 data as INT16 for bit manipulation
2. **Dimensionality Change:** Reshape 1D buffers to 2D for tiled operations
3. **Memory Layout Transformation:** Create views with different logical shapes over the same data

**Example -- Bit Cast:**

```python
@T.prim_func
def bit_cast_example(data: T.Buffer((128,), "float16")):
    # View float16 as uint16 for bit manipulation
    int_view = view(data, dtype="uint16")
    with T.Kernel(128) as bx:
        # Perform bit-level operations
        bits = int_view[bx]
        int_view[bx] = bits ^ 0x8000  # Flip sign bit
```

---

## Quick Reference Table

| Module | Primary Functions | Architecture |
|--------|------------------|--------------|
| `experimental.gemm_sp` | `gemm_sp`, `gemm_sp_v2` | SM80+ (Ampere) |
| `customize` | `dp4a`, `clamp`, `reshape`, `view`, `loop_break`, atomics | All |
| `random` | `rng_init`, `rng_rand`, `rng_rand_float` | CUDA |
| `pdl` | `pdl_trigger`, `pdl_sync` | SM90+ (Hopper) |
| `cluster` | `cluster_sync`, `cluster_arrive`, `cluster_wait` | SM90+ (Hopper) |
| `warpgroup` | `WarpSpecialize`, `ws` | SM90+ (Hopper) |
| `fastmath` | `__log`, `__exp`, `__sin`, `__cos` | All (with `--use_fast_math`) |
| `fill_op` | `fill`, `clear` | All |
| `symbolics` | `dynamic`, `symbolic` | All |
| `frame` | `LetFrame`, `FrameStack`, `has_let_value`, `get_let_value` | All |
