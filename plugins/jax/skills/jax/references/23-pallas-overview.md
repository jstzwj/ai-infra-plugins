# Chapter 23: Pallas - GPU and TPU Kernel Programming Overview

## 23.1 Introduction to Pallas

Pallas is an extension to JAX that enables writing custom GPU and TPU kernels directly
within the JAX ecosystem. While JAX normally compiles high-level operations (like
`jax.numpy.matmul`) to optimized hardware code via XLA, Pallas provides a lower-level
programming model for cases where fine-grained control over memory access patterns,
synchronization, and hardware features is required.

Pallas bridges the gap between JAX's high-level functional transformations and the
low-level hardware execution model. It allows kernel authors to express computations
that operate on blocks (tiles) of data, control which memory spaces are used, and
explicitly manage data movement between memory hierarchies -- all while remaining
composable with JAX transformations like `jax.vmap`, `jax.grad`, and `jax.jit`.

### Why Pallas?

Standard JAX programs operate on whole arrays and rely on XLA to generate efficient
hardware code. However, certain workloads require explicit control that XLA cannot
provide automatically:

- **Custom memory access patterns** for bandwidth-limited kernels (e.g., fused attention,
  custom reduction patterns)
- **Explicit shared memory management** for tiling strategies that outperform the
  compiler's automatic tiling
- **Hardware-specific features** like Tensor Cores, TMA (Tensor Memory Accelerator),
  and warp-level primitives
- **Block-sparse computations** where only a subset of blocks are processed
- **Fused kernels** that combine multiple operations to avoid memory round-trips

### Pallas in the JAX Stack

```
+--------------------------------------------------------------+
|  User Code (Python)                                          |
|  jax.numpy | jax.lax | custom Pallas kernels                |
+--------------------------------------------------------------+
|  JAX Transformations                                         |
|  jit | grad | vmap | pjit | shard_map                       |
+--------------------------------------------------------------+
|  Pallas Runtime                                              |
|  pallas_call | BlockSpec | GridSpec | memory spaces         |
+--------------------------------------------------------------+
|  Backend Code Generation                                     |
|  GPU: Triton/PTX  |  TPU: Mosaic                            |
+--------------------------------------------------------------+
|  Hardware                                                    |
|  NVIDIA GPU (SM, Tensor Core)  |  Google TPU (systolic)     |
+--------------------------------------------------------------+
```

Pallas compiles kernels through different backends depending on the target hardware:
- **GPU backend**: Lowers to Triton IR, which then compiles to PTX for NVIDIA GPUs
- **TPU backend**: Lowers to the Mosaic compiler, which generates TPU microcode

---

## 23.2 Core Concepts

### 23.2.1 The Grid Programming Model

Pallas uses a **SPMD** (Single Program, Multiple Data) programming model. A Pallas
kernel function is written from the perspective of a single "program" that processes
one point in a logical iteration space called a **grid**.

A grid is an N-dimensional tuple of integers that defines how many program instances
will execute. Each program instance is identified by its **program ID**, a tuple of
indices into the grid. The hardware executes multiple program instances in parallel
when possible.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# A 1D grid of size 8: 8 program instances execute
# Each instance gets program_id = (0,), (1,), ..., (7,)
grid = (8,)

# A 2D grid: 4 x 3 = 12 program instances
# program_id = (0,0), (0,1), (0,2), (1,0), ..., (3,2)
grid = (4, 3)

# A 3D grid: batch, row, column
grid = (2, 16, 16)
```

The grid defines the iteration space, and each program instance uses its `program_id`
to determine which portion of the input/output data it should process.

### 23.2.2 Ref Types (Mutable Buffers)

Unlike standard JAX arrays which are immutable, Pallas kernels operate on **Ref**
objects. A `Ref` is a mutable buffer that represents a view into a memory space
(GMEM, SMEM, or registers). Refs support both reading and writing via array
indexing syntax.

```python
def my_kernel(x_ref: pl.Ref, y_ref: pl.Ref, o_ref: pl.Ref):
    # x_ref, y_ref are input Refs (read-only in practice)
    # o_ref is an output Ref (written to)

    # Read from input refs
    x_block = x_ref[...]          # Read entire block
    y_block = y_ref[:, 0]         # Read a column slice

    # Compute
    result = x_block + y_block

    # Write to output ref
    o_ref[...] = result           # Write entire block
```

Key properties of Refs:
- **Shape**: Each Ref has a fixed shape determined by the BlockSpec
- **Dtype**: The data type of elements in the Ref
- **Memory space**: Where the Ref physically resides (GMEM, SMEM, registers)
- **Mutable**: Can be read from and written to multiple times within a kernel

### 23.2.3 program_id

The `program_id` function returns the index of the current program instance along a
given axis of the grid. This is analogous to `threadIdx` / `blockIdx` in CUDA or
`program_id` in Triton.

```python
def kernel_fn(x_ref, o_ref):
    # For a 2D grid (M, N):
    row_idx = pl.program_id(0)    # Index along first grid axis [0, M)
    col_idx = pl.program_id(1)    # Index along second grid axis [0, N)

    # Use indices to determine which block to process
    # ...
```

`program_id(axis)` takes an integer axis index and returns a scalar integer representing
the position of the current program along that grid dimension.

### 23.2.4 Memory Spaces

Pallas defines a memory hierarchy that maps to hardware-specific memory spaces:

| Memory Space | GPU Equivalent | TPU Equivalent | Description |
|---|---|---|---|
| `DRAM` | HBM (Global Memory) | HBM | Large, high-latency, high-bandwidth |
| `SMEM` | Shared Memory | (scalar ops) | Small, low-latency, per-SM/per-core |
| `registers` | Registers | VMEM | Fastest, per-thread/per-subcore |

```python
# Specifying memory spaces in BlockSpec
pl.BlockSpec(
    block_shape=(64, 64),
    index_map=lambda i, j: (i, j),
    memory_space=pl.DRAM    # Default: data stays in global memory
)

pl.BlockSpec(
    block_shape=(64, 64),
    index_map=lambda i, j: (i, j),
    memory_space=pl.SMEM    # Data loaded into shared memory
)
```

On GPUs, the typical pattern is:
1. Load data from DRAM (HBM) into SMEM (shared memory) or registers
2. Compute on data in SMEM or registers
3. Store results from registers back to DRAM

On TPUs, data flows between HBM and VMEM (vector memory), with SEM (scalar memory)
used for scalar operations and control flow.

---

## 23.3 The pallas_call API

`pallas_call` is the primary entry point for executing Pallas kernels. It is a
higher-order function that takes a kernel function and configuration parameters,
and returns a JAX-callable function.

### 23.3.1 Function Signature

```python
def pallas_call(
    fun: Callable,                  # The kernel function
    out_shape: PyTree[ShapeDtype],  # Output shapes and dtypes
    grid: Tuple[int, ...],          # Grid dimensions
    in_specs: PyTree[BlockSpec],    # How inputs are blocked
    out_specs: PyTree[BlockSpec],   # How outputs are blocked
    interpret: bool = False,        # Emulation mode
    compiler_params: dict = {},     # Backend-specific parameters
    name: str = '',                 # Name for debugging
) -> Callable:
    """Returns a function that executes the Pallas kernel."""
```

### 23.3.2 Parameters in Detail

**`fun`**: The kernel function. It receives Ref arguments corresponding to each input
and output. The function body operates on these Refs to read inputs, compute results,
and write outputs.

**`out_shape`**: A pytree of `jax.ShapeDtypeStruct` objects describing the shape and
dtype of each output. This is required because the kernel writes to output Refs
imperatively, so the caller must declare the output shape.

```python
# Single output
out_shape = jax.ShapeDtypeStruct((1024, 1024), jnp.float32)

# Multiple outputs
out_shape = (
    jax.ShapeDtypeStruct((1024, 1024), jnp.float32),
    jax.ShapeDtypeStruct((1024,), jnp.float32),
)
```

**`grid`**: A tuple of integers defining the multi-dimensional iteration space. The
kernel function executes once for each point in this grid. The total number of
program instances is the product of all grid dimensions.

**`in_specs`**: A pytree of `BlockSpec` objects (or `None`) that describes how each
input array is partitioned into blocks and mapped to grid indices.

**`out_specs`**: A pytree of `BlockSpec` objects that describes how output Refs are
mapped to grid indices.

### 23.3.3 Complete Example: Vector Addition

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_vectors_kernel(x_ref: pl.Ref, y_ref: pl.Ref, o_ref: pl.Ref):
    """Kernel that adds two vectors element-wise."""
    # program_id(0) gives the block index along the single grid dimension
    # The BlockSpec already maps the correct block to the refs
    x = x_ref[...]          # Load block of x
    y = y_ref[...]          # Load block of y
    o_ref[...] = x + y      # Store result

BLOCK_SIZE = 256

def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]
    assert n % BLOCK_SIZE == 0, f"Size {n} must be divisible by {BLOCK_SIZE}"
    grid = (n // BLOCK_SIZE,)

    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),   # x[i*B:(i+1)*B]
            pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),   # y[i*B:(i+1)*B]
        ],
        out_specs=pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),  # o[i*B:(i+1)*B]
    )(x, y)

# Usage
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (4096,))
y = jax.random.normal(jax.random.fold_in(key, 1), (4096,))
result = add_vectors(x, y)

# Verify against JAX
expected = x + y
print(jnp.allclose(result, expected))  # True
```

---

## 23.4 BlockSpec

`BlockSpec` is the central configuration object that describes how arrays are tiled
and how tiles map to grid indices.

### 23.4.1 BlockSpec Fields

```python
class BlockSpec:
    block_shape: Tuple[Optional[int], ...]   # Size of each block dimension
    index_map: Callable[..., Tuple[int, ...]] # Maps grid indices to block indices
    memory_space: MemorySpace = DRAM          # Where the block resides
    # (GPU backend also supports padding via other fields)
```

**`block_shape`**: Defines the shape of the block that the kernel receives as a Ref.
Each dimension specifies the size of the block along that axis. Use `None` for
dimensions that are not blocked (i.e., the entire dimension is passed).

**`index_map`**: A function that takes the same arguments as grid dimensions (one
integer per grid axis) and returns a tuple of indices. Each index selects which
block along that dimension the current program instance processes.

**`memory_space`**: The target memory space for the block. Defaults to `DRAM`.

### 23.4.2 BlockSpec Examples

```python
# 1D blocking: process 256 elements per program
pl.BlockSpec(
    block_shape=(256,),
    index_map=lambda i: (i,),
)

# 2D blocking: process 64x64 tiles of a matrix
pl.BlockSpec(
    block_shape=(64, 64),
    index_map=lambda i, j: (i, j),
)

# Mixed blocking: block rows but pass full columns
pl.BlockSpec(
    block_shape=(64, None),  # None = don't block this dimension
    index_map=lambda i: (i, 0),
)

# Blocking only some dimensions of a 3D tensor
# Shape: (B, M, N), block B and M but not N
pl.BlockSpec(
    block_shape=(1, 64, None),
    index_map=lambda b, i: (b, i, 0),
)
```

### 23.4.3 How BlockSpec Maps Data to Refs

The BlockSpec works as follows for a given program instance with grid index `(g0, g1, ...)`:

1. Call `index_map(g0, g1, ...)` to get block indices `(b0, b1, ...)`
2. For each dimension `d`, the block spans indices `[b_d * block_shape[d], (b_d + 1) * block_shape[d])`
3. The kernel's Ref argument contains this block of the original array

```python
# Example: 1024x1024 matrix with 64x64 blocks
# Grid: (16, 16) = (1024//64, 1024//64)
# For grid point (2, 3):
#   index_map(2, 3) = (2, 3)
#   Block covers rows [128, 192) and columns [192, 256)

matrix = jnp.ones((1024, 1024))
block_spec = pl.BlockSpec(
    block_shape=(64, 64),
    index_map=lambda i, j: (i, j),
)
# At grid point (2, 3), the kernel's ref contains matrix[128:192, 192:256]
```

### 23.4.4 Advanced Index Mapping

The `index_map` function can express complex data access patterns, including:
- Broadcasting (multiple grid points reading the same block)
- Diagonal access
- Strided access

```python
# Broadcasting: every program reads the same bias vector
bias_spec = pl.BlockSpec(
    block_shape=(64,),
    index_map=lambda i, j: (0,),  # Always reads block 0 regardless of grid position
)

# Diagonal: process elements along the diagonal of a square matrix
diagonal_spec = pl.BlockSpec(
    block_shape=(64,),
    index_map=lambda i: (i, i),  # 2D index maps to diagonal blocks
)
# Note: index_map returns a tuple matching the array dimensionality
```

---

## 23.5 Grid Specification

### 23.5.1 Basic Grid

The grid defines the total number of program instances and the dimensionality of the
iteration space. Each dimension of the grid is an integer count.

```python
# 1D grid: N program instances
grid = (N,)

# 2D grid: M x N program instances
grid = (M, N)

# 3D grid
grid = (B, M, N)
```

### 23.5.2 Grid and Program ID Interaction

```python
def kernel(x_ref, y_ref, o_ref):
    # For grid = (4, 3):
    i = pl.program_id(0)  # 0, 1, 2, or 3
    j = pl.program_id(1)  # 0, 1, or 2

    # Use i, j to compute local offsets or control flow
    # (Usually not needed if BlockSpec handles indexing)
```

### 23.5.3 Grid and BlockSpec Consistency

The number of arguments to each `index_map` function must match the number of grid
dimensions. The `block_shape` multiplied by the grid size along each blocked dimension
should equal the full array size.

```python
# Array shape: (512, 1024)
# Block shape: (64, 128)
# Grid must be: (512 // 64, 1024 // 128) = (8, 8)

grid = (8, 8)
block_spec = pl.BlockSpec(
    block_shape=(64, 128),
    index_map=lambda i, j: (i, j),  # Two args for 2D grid
)
```

---

## 23.6 Programming Model Details

### 23.6.1 SPMD Execution

All program instances execute the same kernel function. This is the SPMD model:
each program runs the same code but operates on different data, identified by its
`program_id`.

```python
def spmd_kernel(x_ref, o_ref):
    # Every program instance executes this same code
    # But x_ref points to a different block of x for each program
    block = x_ref[...]
    o_ref[...] = jnp.square(block)

# Each of the 16 programs processes 64 elements
result = pl.pallas_call(
    spmd_kernel,
    out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
    grid=(16,),
    in_specs=[pl.BlockSpec((64,), lambda i: (i,))],
    out_specs=pl.BlockSpec((64,), lambda i: (i,)),
)(x)
```

### 23.6.2 Within-Program Parallelism

Within a single program instance, operations on blocks are vectorized. When you write
`o_ref[...] = x_ref[...] + y_ref[...]`, the addition is performed in parallel across
all elements of the block. On GPUs, this maps to CUDA threads within a thread block;
on TPUs, this maps to vector units.

### 23.6.3 Program Ordering and Independence

Program instances are logically independent -- each program writes to a distinct
portion of the output (as defined by `out_specs`). The hardware may execute programs
in any order or concurrently. There is no implicit synchronization between programs.

If programs need to coordinate (rare in basic Pallas), this requires explicit
synchronization primitives provided by the GPU or TPU backend.

---

## 23.7 Emulation Mode (Interpret Mode)

Pallas provides an emulation mode (`interpret=True`) that executes the kernel on the
host CPU using standard JAX operations. This is invaluable for debugging because:
- You can use `jax.debug.print` inside kernels
- Errors produce readable Python tracebacks
- You can step through kernel logic without GPU/TPU hardware

```python
def debug_kernel(x_ref, o_ref):
    x = x_ref[...]
    jax.debug.print("Block shape: {}", x.shape)   # Only works in interpret mode
    jax.debug.print("Block sum: {}", jnp.sum(x))
    o_ref[...] = x * 2.0

def debug_double(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    block_size = 64
    return pl.pallas_call(
        debug_kernel,
        out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
        grid=(n // block_size,),
        in_specs=[pl.BlockSpec((block_size,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block_size,), lambda i: (i,)),
        interpret=True,   # Enable emulation mode
    )(x)

# Works on CPU, no GPU required
x = jnp.arange(64.0)
result = debug_double(x)
```

### 23.7.1 Limitations of Interpret Mode

- Performance is much slower than hardware execution (intended for debugging only)
- Some hardware-specific features (SMEM, barriers, atomic operations) may not be
  faithfully emulated
- Memory space specifications are ignored
- The execution order of programs may differ from hardware execution

---

## 23.8 Composing with JAX Transformations

One of Pallas's most powerful features is that `pallas_call` is a first-class JAX
primitive, meaning it composes with JAX transformations.

### 23.8.1 Composing with jax.jit

`pallas_call` is automatically JIT-compiled when used inside a `jax.jit` context or
when called on accelerator arrays. You can also explicitly JIT the outer function.

```python
@jax.jit
def jit_add_vectors(x, y):
    return add_vectors(x, y)

result = jit_add_vectors(x, y)
```

### 23.8.2 Composing with jax.vmap

`jax.vmap` can vectorize a Pallas kernel over additional batch dimensions. This is
useful for applying the same kernel across a batch of inputs.

```python
# Batched vector addition
# x: (batch, n), y: (batch, n)
batched_add = jax.vmap(add_vectors)

x_batch = jax.random.normal(key, (8, 4096))   # 8 batches
y_batch = jax.random.normal(key2, (8, 4096))
result = batched_add(x_batch, y_batch)
# result.shape = (8, 4096)
```

When `vmap` is applied to a `pallas_call`, the batched dimension is handled by
adding an additional grid dimension or by running multiple kernel invocations.

### 23.8.3 Composing with jax.grad

Pallas kernels can be differentiated using `jax.grad`. The automatic differentiation
system generates a reverse-mode kernel that computes gradients by inverting the
data flow through the original kernel.

```python
def my_function(x):
    return jnp.sum(add_vectors(x, jnp.ones_like(x)))

# Compute gradient through the Pallas kernel
gradient = jax.grad(my_function)(x)
```

### 23.8.4 Composing with jax.vmap and jax.grad Together

```python
# Per-sample gradients through a Pallas kernel
def loss_per_sample(x):
    y = add_vectors(x, jnp.ones_like(x))
    return jnp.sum(y ** 2)

# Vectorized gradient: compute gradient for each sample in the batch
per_sample_grads = jax.vmap(jax.grad(loss_per_sample))(x_batch)
```

---

## 23.9 Complete Example: Matrix Multiplication

This example demonstrates a tiled matrix multiplication kernel with accumulation
across the shared dimension.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

def matmul_kernel(
    a_ref: pl.Ref,    # (BLOCK_M, BLOCK_K)
    b_ref: pl.Ref,    # (BLOCK_K, BLOCK_N)
    c_ref: pl.Ref,    # (BLOCK_M, BLOCK_N)
):
    """Tiled matrix multiplication kernel.

    For each output tile (i, j), accumulate partial matmuls across the K dimension.
    """
    i = pl.program_id(0)
    j = pl.program_id(1)

    M, K = a_ref.shape  # Only valid if we passed the full array specs
    # Instead, we accumulate across k grid iterations

    # Initialize accumulator
    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)

    # Accumulate over K tiles
    num_k_tiles = K_TOTAL // BLOCK_K  # Must be known at trace time
    for k in range(num_k_tiles):
        # Load tiles of A and B
        # In a real kernel with GridSpec, k would be a grid dimension
        # Here we loop explicitly
        pass

def tiled_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Tiled matrix multiplication using Pallas with explicit K-loop."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: {K} vs {K2}"

    BM, BN, BK = 64, 64, 32
    assert M % BM == 0 and N % BN == 0 and K % BK == 0

    def kernel(a_ref, b_ref, c_ref):
        # Initialize output accumulator
        acc = jnp.zeros((BM, BN), dtype=jnp.float32)

        # Loop over K dimension in tiles
        for k in range(K // BK):
            # Load a tile of A: (BM, BK)
            a_tile = a_ref[:, k * BK:(k + 1) * BK]
            # Load a tile of B: (BK, BN)
            b_tile = b_ref[k * BK:(k + 1) * BK, :]
            # Accumulate
            acc += jnp.dot(a_tile, b_tile)

        c_ref[...] = acc

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // BM, N // BN),
        in_specs=[
            pl.BlockSpec((BM, None), lambda i, j: (i, 0)),   # Full K for each A block
            pl.BlockSpec((None, BN), lambda i, j: (0, j)),   # Full K for each B block
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)

# Usage
M, K, N = 512, 256, 512
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (M, K), dtype=jnp.float32)
b = jax.random.normal(jax.random.fold_in(key, 1), (K, N), dtype=jnp.float32)
c = tiled_matmul(a, b)

# Verify
expected = jnp.dot(a, b)
print(f"Max error: {jnp.max(jnp.abs(c - expected))}")
# Should be small (floating point tolerance)
```

### 23.9.1 Three-Grid-Dimension Matmul

A more efficient approach uses three grid dimensions, with the K dimension handled
via an accumulator that is initialized in the kernel:

```python
def matmul_3d_kernel(a_ref, b_ref, c_ref):
    """Matmul kernel with 3D grid: grid = (M//BM, N//BN, 1).

    The K dimension is handled by a loop inside the kernel.
    """
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    for k_block in range(K // BK):
        # a_ref is the full row block: (BM, K)
        # b_ref is the full column block: (K, BN)
        a_tile = a_ref[:, k_block * BK:(k_block + 1) * BK]
        b_tile = b_ref[k_block * BK:(k_block + 1) * BK, :]
        acc += jnp.dot(a_tile, b_tile)
    c_ref[...] = acc
```

---

## 23.10 Complete Example: Fused Activation Functions

### 23.10.1 Fused GELU

The GELU (Gaussian Error Linear Unit) activation is commonly used in transformers.
Fusing the matmul with GELU avoids writing the intermediate result to memory.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BLOCK_SIZE = 256

def gelu_kernel(x_ref: pl.Ref, o_ref: pl.Ref):
    """Compute GELU activation: x * Phi(x) where Phi is the standard normal CDF."""
    x = x_ref[...]
    # Approximate GELU using tanh approximation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x ** 3)
    o_ref[...] = 0.5 * x * (1.0 + jnp.tanh(inner))

def fused_gelu(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    assert n % BLOCK_SIZE == 0
    grid = (n // BLOCK_SIZE,)
    return pl.pallas_call(
        gelu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=[pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,))],
        out_specs=pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,)),
    )(x)

# Usage
x = jax.random.normal(jax.random.PRNGKey(0), (4096,))
result = fused_gelu(x)
expected = jax.nn.gelu(x, approximate="tanh")
print(jnp.allclose(result, expected, atol=1e-5))
```

### 23.10.2 Fused Softmax

A fused softmax kernel that computes softmax row-by-row, avoiding materializing
intermediate exp() values in global memory.

```python
def softmax_kernel(x_ref: pl.Ref, o_ref: pl.Ref):
    """Fused softmax over the last dimension of a 2D block."""
    x = x_ref[...]                              # (BLOCK_M, N)
    row_max = jnp.max(x, axis=-1, keepdims=True)  # (BLOCK_M, 1)
    exp_x = jnp.exp(x - row_max)                  # (BLOCK_M, N)
    row_sum = jnp.sum(exp_x, axis=-1, keepdims=True)
    o_ref[...] = exp_x / row_sum

BLOCK_M = 32
N = 128

def fused_softmax(x: jax.Array) -> jax.Array:
    """Softmax over the last dimension, processed in row blocks."""
    M = x.shape[0]
    assert M % BLOCK_M == 0
    assert x.shape[1] == N
    grid = (M // BLOCK_M,)
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=grid,
        in_specs=[pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),
    )(x)

# Usage
x = jax.random.normal(jax.random.PRNGKey(0), (256, 128))
result = fused_softmax(x)
expected = jax.nn.softmax(x, axis=-1)
print(jnp.allclose(result, expected, atol=1e-5))
```

### 23.10.3 Fused Bias + GELU + Residual

A kernel that fuses multiple operations: add bias, apply GELU, and add a residual
connection in a single pass.

```python
def fused_bias_gelu_residual_kernel(
    x_ref: pl.Ref,        # Input: (BLOCK_SIZE, D)
    bias_ref: pl.Ref,     # Bias: (D,) -- broadcast
    residual_ref: pl.Ref, # Residual: (BLOCK_SIZE, D)
    o_ref: pl.Ref,        # Output: (BLOCK_SIZE, D)
):
    x = x_ref[...]
    bias = bias_ref[...]         # (D,) broadcast across block
    residual = residual_ref[...]

    # Add bias
    x_biased = x + bias[jnp.newaxis, :]

    # GELU activation (tanh approximation)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x_biased + coeff * x_biased ** 3)
    gelu_out = 0.5 * x_biased * (1.0 + jnp.tanh(inner))

    # Add residual
    o_ref[...] = gelu_out + residual

BLOCK_SIZE = 64
D = 256

def fused_bias_gelu_residual(x: jax.Array, bias: jax.Array, residual: jax.Array):
    M = x.shape[0]
    assert x.shape == (M, D)
    assert bias.shape == (D,)
    assert residual.shape == (M, D)
    assert M % BLOCK_SIZE == 0
    grid = (M // BLOCK_SIZE,)

    return pl.pallas_call(
        fused_bias_gelu_residual_kernel,
        out_shape=jax.ShapeDtypeStruct((M, D), jnp.float32),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_SIZE, D), lambda i: (i, 0)),  # x blocked on M
            pl.BlockSpec((D,), lambda i: (0,)),                # bias: always full
            pl.BlockSpec((BLOCK_SIZE, D), lambda i: (i, 0)),  # residual blocked on M
        ],
        out_specs=pl.BlockSpec((BLOCK_SIZE, D), lambda i: (i, 0)),
    )(x, bias, residual)

# Usage
key = jax.random.PRNGKey(0)
M = 512
x = jax.random.normal(key, (M, D))
bias = jax.random.normal(jax.random.fold_in(key, 1), (D,))
residual = jax.random.normal(jax.random.fold_in(key, 2), (M, D))
result = fused_bias_gelu_residual(x, bias, residual)
```

---

## 23.11 Reductions in Pallas

Pallas kernels can perform reductions within a block. Cross-block reductions require
either atomics (GPU) or a separate kernel launch.

### 23.11.1 Intra-Block Reduction

```python
def reduce_sum_kernel(x_ref: pl.Ref, o_ref: pl.Ref):
    """Sum-reduce along the second dimension within each block."""
    x = x_ref[...]                              # (BLOCK_M, N)
    o_ref[...] = jnp.sum(x, axis=-1)             # (BLOCK_M,)

BLOCK_M = 32
N = 64

def pallas_sum(x: jax.Array) -> jax.Array:
    M = x.shape[0]
    assert x.shape == (M, N)
    assert M % BLOCK_M == 0
    grid = (M // BLOCK_M,)
    return pl.pallas_call(
        reduce_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((M,), jnp.float32),
        grid=grid,
        in_specs=[pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((BLOCK_M,), lambda i: (i,)),
    )(x)
```

### 23.11.2 Full Array Reduction with Final Pass

For a full array reduction, use a two-pass approach: first reduce within blocks,
then reduce the block results.

```python
def full_sum_kernel(x_ref: pl.Ref, o_ref: pl.Ref):
    """First pass: partial sum within each block."""
    o_ref[...] = jnp.sum(x_ref[...])

BLOCK_SIZE = 256

def full_sum(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    assert n % BLOCK_SIZE == 0
    num_blocks = n // BLOCK_SIZE

    # First pass: partial sums
    partials = pl.pallas_call(
        full_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((num_blocks,), jnp.float32),
        grid=(num_blocks,),
        in_specs=[pl.BlockSpec((BLOCK_SIZE,), lambda i: (i,))],
        out_specs=pl.BlockSpec((), lambda i: (i,)),  # Scalar per block
    )(x)

    # Second pass: sum the partials (trivially small, no Pallas needed)
    return jnp.sum(partials)
```

---

## 23.12 Atomic Operations

On GPU, Pallas supports atomic operations for safe cross-block updates. These are
essential for operations like histograms, scatter-adds, and cross-block reductions.

```python
def atomic_add_kernel(x_ref: pl.Ref, o_ref: pl.Ref):
    """Atomically add each element of x to o."""
    x = x_ref[...]
    # o_ref[...] += x  # NOT atomic!
    o_ref.at[...].add(x)  # Atomic add

def histogram_kernel(indices_ref: pl.Ref, out_ref: pl.Ref):
    """Compute histogram using atomic adds."""
    indices = indices_ref[...]
    for i in range(indices.shape[0]):
        idx = indices[i]
        out_ref.at[idx].add(1)
```

---

## 23.13 Dynamic Grid Sizes and Advanced Patterns

### 23.13.1 Dynamic Shapes

Pallas supports dynamic grid sizes through `jax.ShapeDtypeStruct` and careful use of
`block_shape`:

```python
def dynamic_matmul(a: jax.Array, b: jax.Array, block_m: int, block_n: int):
    M, K = a.shape
    _, N = b.shape
    grid = (M // block_m, N // block_n)

    def kernel(a_ref, b_ref, c_ref):
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        for k in range(K // 32):
            a_tile = a_ref[:, k * 32:(k + 1) * 32]
            b_tile = b_ref[k * 32:(k + 1) * 32, :]
            acc += jnp.dot(a_tile, b_tile)
        c_ref[...] = acc

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=grid,
        in_specs=[
            pl.BlockSpec((block_m, None), lambda i, j: (i, 0)),
            pl.BlockSpec((None, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
    )(a, b)
```

### 23.13.2 Multiple Outputs

```python
def multi_output_kernel(x_ref, sum_ref, max_ref):
    x = x_ref[...]
    sum_ref[...] = jnp.sum(x)
    max_ref[...] = jnp.max(x)

def sum_and_max(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    n = x.shape[0]
    num_blocks = n // 64
    return pl.pallas_call(
        multi_output_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_blocks,), jnp.float32),
            jax.ShapeDtypeStruct((num_blocks,), jnp.float32),
        ),
        grid=(num_blocks,),
        in_specs=[pl.BlockSpec((64,), lambda i: (i,))],
        out_specs=(
            pl.BlockSpec((), lambda i: (i,)),
            pl.BlockSpec((), lambda i: (i,)),
        ),
    )(x)
```

---

## 23.14 Common Patterns and Best Practices

### 23.14.1 Choosing Block Sizes

Block size selection has a major impact on performance:

- **Too small**: Insufficient parallelism, high overhead per block
- **Too large**: Exceeds shared memory or register capacity, low occupancy
- **Guidelines for GPU**:
  - 1D kernels: 256-2048 elements per block
  - 2D kernels: 64x64 to 128x128 tiles
  - Matmul: match Tensor Core (16x16, 32x32, or multiples of 8 for MMA)

### 23.14.2 Memory Access Patterns

- Coalesced access: ensure consecutive threads access consecutive memory addresses
- Avoid bank conflicts in shared memory
- Use `None` in block_shape to avoid unnecessary blocking

### 23.14.3 Numerical Considerations

- Use `jnp.float32` accumulators even for `float16`/`bfloat16` inputs
- Be aware of reduction order differences across programs
- Use numerically stable formulations (e.g., subtract max before exp in softmax)

### 23.14.4 Debugging Strategies

1. Start with `interpret=True` for correctness
2. Compare against JAX reference implementation
3. Use `jax.debug.print` for intermediate values in emulation mode
4. Check for shape mismatches and dtype issues
5. Verify BlockSpec index_map logic independently

---

## 23.15 Integration with shard_map

Pallas kernels can be used inside `shard_map` for distributed computation:

```python
from jax.experimental import shard_map

# Define a sharded computation that uses Pallas kernels
@shard_map.check_rep(False)
def sharded_matmul(a_shard, b_shard):
    # Each device runs the Pallas kernel on its local shard
    return tiled_matmul(a_shard, b_shard)

mesh = jax.sharding.Mesh(jax.devices(), ("data",))
with mesh:
    result = sharded_matmul(a_sharded, b_sharded)
```

---

## 23.16 Backend-Specific Extensions

The GPU and TPU backends each provide additional primitives beyond the core Pallas API:

- **GPU backend** (Chapter 24): SMEM management, Tensor Core operations (WMMA, WGMMA),
  TMA (Tensor Memory Accelerator), barriers, software pipelining
- **TPU backend** (Chapter 25): VMEM operations, dot primitives, systolic array control,
  distributed communication via ICI, block-sparse support

These backend-specific features are accessed through `jax.experimental.pallas` submodules:
- `jax.experimental.pallas.gpu` for GPU-specific operations
- `jax.experimental.pallas.tpu` for TPU-specific operations

---

## 23.17 Summary

Pallas provides a structured approach to writing custom accelerator kernels within JAX:

| Concept | Description |
|---|---|
| **Grid** | Multi-dimensional iteration space defining program instances |
| **program_id** | Index of current program within the grid |
| **Ref** | Mutable buffer representing a block of data in a memory space |
| **BlockSpec** | Describes how arrays are tiled and mapped to grid indices |
| **Memory spaces** | DRAM (global), SMEM (shared), registers (fastest) |
| **pallas_call** | Entry point that connects kernel functions to JAX |
| **Emulation** | `interpret=True` for CPU debugging |
| **Composability** | Works with `jit`, `vmap`, `grad`, `shard_map` |

The key design principles are:
1. Write one program that processes one tile of data
2. Use BlockSpec to describe how data is partitioned
3. Let the runtime launch multiple program instances in parallel
4. Compose with JAX transformations for higher-level functionality
