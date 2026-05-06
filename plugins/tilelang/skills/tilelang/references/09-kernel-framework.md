# TileLang Kernel Framework Reference

This reference covers the TileLang kernel definition and execution framework, including kernel launch contexts, thread and block index accessors, external CUDA kernel integration, the JIT compilation pipeline, and kernel caching.

---

## Table of Contents

1. [Overview](#overview)
2. [T.Kernel -- Kernel Launch Context](#tkernel----kernel-launch-context)
3. [KernelLaunchFrame](#kernellaunchframe)
4. [T.CUDASourceCodeKernel -- External CUDA Integration](#tcudasourcecodekernel----external-cuda-integration)
5. [Thread Binding Accessors](#thread-binding-accessors)
6. [Block Index Computation](#block-index-computation)
7. [Thread Index Computation](#thread-index-computation)
8. [Warp and Lane Index Helpers](#warp-and-lane-index-helpers)
9. [Grid-Stride Loops](#grid-stride-loops)
10. [Dynamic Shared Memory Patterns](#dynamic-shared-memory-patterns)
11. [@tilelang.jit Decorator](#tilelangjit-decorator)
12. [Kernel Compilation Modes](#kernel-compilation-modes)
13. [Kernel Caching](#kernel-caching)
14. [Kernel Execution and Tensor Supply](#kernel-execution-and-tensor-supply)
15. [Practical Examples](#practical-examples)

---

## Overview

TileLang kernels are defined using a Python-embedded DSL that compiles to efficient GPU code. The kernel framework provides:

- **Kernel definition**: `T.Kernel` sets up the execution context (grid dimensions, thread count, cluster configuration).
- **Index accessors**: Functions to query thread, block, and warp indices within the kernel.
- **JIT compilation**: The `@tilelang.jit` decorator compiles kernel definitions to GPU executables on demand.
- **External integration**: `T.CUDASourceCodeKernel` allows embedding raw CUDA code within TileLang kernels.
- **Caching**: Compiled kernels are cached to avoid recompilation overhead.

### Kernel Lifecycle

```
1. Definition: Write kernel function in TileLang DSL
2. Decoration: Apply @tilelang.jit with configuration
3. Compilation: TileLang compiles DSL -> TVM IR -> GPU PTX/CUBIN
4. Caching: Compiled kernel is cached by parameter signature
5. Execution: Call the decorated function with PyTorch tensors
6. Return: Output tensors are returned to the caller
```

---

## T.Kernel -- Kernel Launch Context

### Signature

```python
T.Kernel(
    *blocks,                    # Grid dimensions (number of blocks in each dim)
    threads=None,               # Threads per block
    cluster_dims=None,          # Cluster dimensions (CC 9.0+)
    is_cpu=False,               # CPU kernel mode
    prelude=None,               # Prelude code to inject
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*blocks` | int(s) | required | Grid dimensions. 1, 2, or 3 integers specifying the number of thread blocks along each dimension. For example, `T.Kernel(grid_x, grid_y)` creates a 2D grid of `grid_x * grid_y` blocks. |
| `threads` | int or tuple | `None` | Number of threads per block. If a single int, creates a 1D thread block. If a tuple of 2 or 3 ints, creates a 2D or 3D thread block. If `None`, the compiler infers the thread count from the kernel body. |
| `cluster_dims` | tuple or None | `None` | Thread block cluster dimensions for compute capability 9.0+ (Hopper). A tuple of 1-3 integers specifying the cluster size along each dimension. For example, `(2, 1, 1)` creates a cluster of 2 blocks. |
| `is_cpu` | bool | `False` | If `True`, the kernel is compiled for CPU execution instead of GPU. Useful for debugging and testing. |
| `prelude` | str or None | `None` | A string of code to inject at the beginning of the compiled kernel. Can be used to define helper functions or include headers. |

### Block Dimensions Specification

The `*blocks` parameter defines the grid dimensions (number of thread blocks):

```python
# 1D grid: gridDim.x = num_blocks, gridDim.y = 1, gridDim.z = 1
with T.Kernel(num_blocks):
    # Kernel body
    pass

# 2D grid: gridDim.x = blocks_x, gridDim.y = blocks_y
with T.Kernel(blocks_x, blocks_y):
    # Kernel body
    pass

# 3D grid
with T.Kernel(blocks_x, blocks_y, blocks_z):
    # Kernel body
    pass
```

### Thread Count Specification

The `threads` parameter defines the thread block dimensions:

```python
# 1D thread block: blockDim.x = 128, blockDim.y = 1, blockDim.z = 1
with T.Kernel(num_blocks, threads=128):
    pass

# 2D thread block: blockDim.x = 16, blockDim.y = 8
with T.Kernel(num_blocks, threads=(16, 8)):
    pass

# 3D thread block: blockDim.x = 8, blockDim.y = 8, blockDim.z = 4
with T.Kernel(num_blocks, threads=(8, 8, 4)):
    pass

# Auto-inferred thread count
with T.Kernel(num_blocks):
    # Thread count determined by T.Parallel usage in the kernel body
    with T.Parallel(256) as i:
        pass  # threads=256 inferred
```

### Cluster Dimensions for CC 9.0+

On NVIDIA Hopper (SM 90) and later architectures, thread blocks can be grouped into **clusters** that cooperate via shared memory:

```python
# 2-block cluster along X dimension
with T.Kernel(
    num_blocks_x, num_blocks_y,
    threads=128,
    cluster_dims=(2, 1, 1)
):
    # Blocks within a cluster can access each other's shared memory
    # using distributed shared memory (distributed shared)
    pass

# 4-block cluster (2x2)
with T.Kernel(
    num_blocks_x, num_blocks_y,
    threads=128,
    cluster_dims=(2, 2, 1)
):
    pass
```

Cluster size restrictions:
- Maximum cluster size: 8 thread blocks (SM 90), 16 thread blocks (SM 100).
- Total threads in a cluster must not exceed 1024 (SM 90) or 2048 (SM 100).
- Cluster dimensions must divide evenly into the grid dimensions.

### CPU Kernel Mode

```python
# CPU kernel for debugging and testing
with T.Kernel(1, threads=1, is_cpu=True):
    # This kernel runs on the CPU
    # Useful for verifying correctness without GPU
    for i in range(N):
        output[i] = input[i] * 2.0
```

### Prelude Code Injection

```python
# Inject helper code at the beginning of the kernel
prelude_code = """
__device__ float my_helper(float x) {
    return x * x + 1.0f;
}
"""

with T.Kernel(num_blocks, threads=128, prelude=prelude_code):
    # my_helper() is now available in the kernel
    pass
```

### Example: Complete Kernel with T.Kernel

```python
import tilelang
import tilelang.language as T

def vector_add(M, N, block_M=64, block_N=64, dtype="float16"):
    # Define the kernel function
    @T.Kernel((M + block_M - 1) // block_M, (N + block_N - 1) // block_N,
              threads=(block_M, block_N))
    def kernel(A_ptr, B_ptr, C_ptr):
        # Get thread indices
        i = T.get_thread_binding(0)
        j = T.get_thread_binding(1)
        # Get block indices
        bi = T.get_block_binding(0)
        bj = T.get_block_binding(1)

        # Compute global indices
        gi = bi * block_M + i
        gj = bj * block_N + j

        # Bounds check
        if gi < M and gj < N:
            C_ptr[gi, gj] = A_ptr[gi, gj] + B_ptr[gi, gj]

    return kernel
```

---

## KernelLaunchFrame

The `KernelLaunchFrame` class provides runtime information about the current kernel execution context. It is accessed through `KernelLaunchFrame.Current()`.

### Class Interface

```python
class KernelLaunchFrame:
    @staticmethod
    def Current() -> KernelLaunchFrame
        # Get the current kernel launch frame

    @property
    def blocks(self) -> tuple
        # Grid dimensions as a tuple of integers

    @property
    def threads(self) -> tuple
        # Thread block dimensions as a tuple of integers

    @property
    def num_threads(self) -> int
        # Total number of threads per block (blockDim.x * blockDim.y * blockDim.z)

    def get_block_extent(self, dim: int) -> int
        # Get the grid extent along the specified dimension (gridDim)

    def get_block_binding(self, dim: int) -> int
        # Get the current block index along the specified dimension (blockIdx)

    def get_thread_extent(self, dim: int) -> int
        # Get the thread block extent along the specified dimension (blockDim)

    def get_thread_binding(self, dim: int) -> int
        # Get the current thread index within the block along the specified dimension (threadIdx)
```

### Properties and Methods

| Property/Method | Return Type | GPU Equivalent | Description |
|----------------|------------|----------------|-------------|
| `Current()` | KernelLaunchFrame | N/A | Static method to get the current frame |
| `blocks` | tuple | `gridDim` | Grid dimensions |
| `threads` | tuple | `blockDim` | Thread block dimensions |
| `num_threads` | int | `blockDim.x*y*z` | Total threads per block |
| `get_block_extent(dim)` | int | `gridDim[dim]` | Grid extent along dimension |
| `get_block_binding(dim)` | int | `blockIdx[dim]` | Block index along dimension |
| `get_thread_extent(dim)` | int | `blockDim[dim]` | Block size along dimension |
| `get_thread_binding(dim)` | int | `threadIdx[dim]` | Thread index along dimension |

### Example: Using KernelLaunchFrame

```python
import tilelang.language as T

@T.Kernel(grid_x, grid_y, threads=(tx, ty))
def my_kernel(A, B, C):
    frame = KernelLaunchFrame.Current()

    # Query grid dimensions
    grid_x = frame.blocks[0]
    grid_y = frame.blocks[1]

    # Query thread block dimensions
    block_x = frame.threads[0]
    block_y = frame.threads[1]

    # Total threads
    total = frame.num_threads  # block_x * block_y

    # Current block index
    bx = frame.get_block_binding(0)  # blockIdx.x
    by = frame.get_block_binding(1)  # blockIdx.y

    # Current thread index
    tx = frame.get_thread_binding(0)  # threadIdx.x
    ty = frame.get_thread_binding(1)  # threadIdx.y
```

---

## T.CUDASourceCodeKernel -- External CUDA Integration

### Signature

```python
T.CUDASourceCodeKernel(
    source_code_or_path,        # CUDA source code or path to .cu file
    entry_name="main_kernel",   # Name of the kernel entry function
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_code_or_path` | str | required | Either a string containing CUDA C++ source code, or a path to a `.cu` or `.cuh` file. |
| `entry_name` | str | `"main_kernel"` | The name of the `__global__` function to use as the kernel entry point. |

### Overview

`T.CUDASourceCodeKernel` allows developers to integrate existing CUDA kernels into TileLang programs. This is useful when:

1. An optimized CUDA kernel already exists and should be reused.
2. A specific operation requires hand-written CUDA that TileLang cannot express.
3. Low-level hardware intrinsics need to be accessed directly.

### Example: Inline CUDA Source

```python
import tilelang.language as T

cuda_source = """
__global__ void vector_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
"""

# Register the CUDA kernel
cuda_kernel = T.CUDASourceCodeKernel(cuda_source, entry_name="vector_add_kernel")

# Launch it from TileLang
cuda_kernel(blocks=(N + 255) // 256, threads=256)(A_tensor, B_tensor, C_tensor, N)
```

### Example: External CUDA File

```python
# Load kernel from a .cu file
cuda_kernel = T.CUDASourceCodeKernel(
    "/path/to/my_kernel.cu",
    entry_name="custom_gemm_kernel"
)

# Launch with grid/block configuration
cuda_kernel(blocks=grid_dims, threads=block_dims)(
    A_ptr, B_ptr, C_ptr, M, N, K
)
```

### Integration with TileLang Buffers

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def hybrid_kernel(M, N, K, in_dtype="float16"):
    # Use TileLang for data movement
    A_smem = T.alloc_shared([M, K], in_dtype)
    T.copy(A_global, A_smem)

    # Use custom CUDA kernel for specialized computation
    custom_cuda = T.CUDASourceCodeKernel("""
    __global__ void specialized_op(half* input, half* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // Custom hardware-intrinsic-based operation
            output[idx] = __hmul(input[idx], __float2half(2.0f));
        }
    }
    """, entry_name="specialized_op")

    # Launch custom kernel
    custom_cuda(blocks=(M*K + 127) // 128, threads=128)(A_smem, B_global, M * K)

    return B_global
```

---

## Thread Binding Accessors

Thread binding accessors provide the current thread's position within the thread block hierarchy. These are the TileLang equivalents of CUDA's built-in variables (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`).

### Accessor Functions

```python
T.get_thread_binding(dim)    # Current thread index within block along dimension dim
T.get_block_binding(dim)     # Current block index within grid along dimension dim
T.get_thread_extent(dim)     # Block size along dimension dim
T.get_block_extent(dim)      # Grid size along dimension dim
```

### Dimension Indexing

| dim | CUDA Equivalent | Description |
|-----|----------------|-------------|
| 0 | `.x` | First (fastest-varying) dimension |
| 1 | `.y` | Second dimension |
| 2 | `.z` | Third (slowest-varying) dimension |

### Example: 1D Thread Mapping

```python
import tilelang.language as T

@T.Kernel(num_blocks, threads=256)
def kernel_1d(A, B, N):
    tid = T.get_thread_binding(0)      # threadIdx.x (0..255)
    bid = T.get_block_binding(0)       # blockIdx.x
    bdim = T.get_thread_extent(0)      # blockDim.x = 256
    gdim = T.get_block_extent(0)       # gridDim.x

    global_idx = bid * bdim + tid      # Global thread index

    if global_idx < N:
        B[global_idx] = A[global_idx] * 2.0
```

### Example: 2D Thread Mapping

```python
import tilelang.language as T

@T.Kernel(blocks_x, blocks_y, threads=(16, 16))
def kernel_2d(A, B, M, N):
    tx = T.get_thread_binding(0)       # threadIdx.x (0..15)
    ty = T.get_thread_binding(1)       # threadIdx.y (0..15)
    bx = T.get_block_binding(0)        # blockIdx.x
    by = T.get_block_binding(1)        # blockIdx.y

    gi = bx * 16 + tx                  # Global row index
    gj = by * 16 + ty                  # Global column index

    if gi < M and gj < N:
        B[gi, gj] = A[gi, gj] * 2.0
```

### Example: Linearized Thread Index

```python
# Convert multi-dimensional thread index to linear index
tx = T.get_thread_binding(0)
ty = T.get_thread_binding(1)
tz = T.get_thread_binding(2)
bdx = T.get_thread_extent(0)
bdy = T.get_thread_extent(1)
linear_tid = tz * bdx * bdy + ty * bdx + tx
```

---

## Block Index Computation Patterns

Block indices determine which tile of the output data the current thread block is responsible for.

### 1D Block Tiling

```python
# Each block processes one tile along a 1D dimension
bid = T.get_block_binding(0)
tile_start = bid * tile_size
```

### 2D Block Tiling (Common for Matrix Operations)

```python
# 2D block grid for matrix tiling
bx = T.get_block_binding(0)    # Block column (along M dimension)
by = T.get_block_binding(1)    # Block row (along N dimension)

m_start = bx * block_M
n_start = by * block_N
```

### 3D Block Tiling (For Volumes/Tensors)

```python
bx = T.get_block_binding(0)
by = T.get_block_binding(1)
bz = T.get_block_binding(2)

d_start = bz * block_D
h_start = by * block_H
w_start = bx * block_W
```

### Flattened Block Index

When using a 1D grid to cover a 2D problem:

```python
# Linear block index to 2D coordinates
bid = T.get_block_binding(0)
bx = bid % num_blocks_x
by = bid // num_blocks_x
```

### Batch-aware Block Indexing

```python
# Common pattern for batched operations
# Grid: (batch_size * num_tiles_m, num_tiles_n)
bid_x = T.get_block_binding(0)
bid_y = T.get_block_binding(1)

batch_idx = bid_x // num_tiles_m
tile_m = bid_x % num_tiles_m
tile_n = bid_y

m_start = tile_m * block_M
n_start = tile_n * block_N
```

---

## Thread Index Computation Patterns

### Global Linear Index

```python
# Compute the global linear thread index (across entire grid)
tid = T.get_thread_binding(0)
bid = T.get_block_binding(0)
block_size = T.get_thread_extent(0)
global_tid = bid * block_size + tid
```

### 2D Global Index

```python
tx = T.get_thread_binding(0)
ty = T.get_thread_binding(1)
bx = T.get_block_binding(0)
by = T.get_block_binding(1)

global_i = bx * blockDim_x + tx
global_j = by * blockDim_y + ty
```

### Warp-aware Thread Index

```python
# Compute warp ID and lane ID within a thread block
linear_tid = T.get_thread_binding(0)  # Assuming 1D thread block
warp_id = linear_tid // 32
lane_id = linear_tid % 32
```

### Sub-warp Thread Grouping

```python
# Group threads into sub-warps of size 8
group_size = 8
linear_tid = T.get_thread_binding(0)
group_id = linear_tid // group_size
lane_in_group = linear_tid % group_size
```

---

## Warp and Lane Index Helpers

TileLang provides convenient helpers for warp-level programming:

### Warp Index

```python
# Get the warp ID within the current thread block
warp_id = T.get_warp_id()        # Linear warp index (0, 1, 2, ...)
num_warps = T.get_num_warps()    # Total warps in the block
```

### Lane Index

```python
# Get the lane (thread) index within the current warp
lane_id = T.get_lane_id()        # 0-31
```

### Warp Group Index (Hopper+)

```python
# On Hopper, warp groups (4 warps) cooperate on WGMMA
warp_group_id = T.get_warp_group_id()    # 0, 1, 2, ...
num_warp_groups = T.get_num_warp_groups()
```

### Common Warp Patterns

```python
import tilelang.language as T

# Pattern 1: Warp-level broadcast
# Thread 0 in each warp loads a value, others use it
warp_value = buffer[T.get_warp_id() * 32]  # Thread 0 loads
# All threads in the warp now have warp_value

# Pattern 2: Warp-level reduction
partial_sum = my_value
# Reduce across warp
for offset in [16, 8, 4, 2, 1]:
    partial_sum += T.shfl_down(partial_sum, offset)
# Thread 0 in each warp holds the sum

# Pattern 3: Warp-group cooperative GEMM (Hopper)
# All 128 threads in a warp group cooperate on WGMMA
warp_group = T.get_warp_group_id()
if warp_group == 0:
    T.wgmma_gemm(A, B, C, mbar=mbar)
```

---

## Grid-Stride Loops

Grid-stride loops allow a fixed-size grid to process an arbitrarily large dataset by having each thread stride through the data.

### Basic Pattern

```python
import tilelang.language as T

@T.Kernel(num_blocks, threads=256)
def grid_stride_kernel(A, B, N):
    tid = T.get_thread_binding(0)
    bid = T.get_block_binding(0)
    block_size = T.get_thread_extent(0)
    grid_size = T.get_block_extent(0) * block_size

    global_start = bid * block_size + tid

    # Grid-stride loop: each thread processes elements at stride = grid_size
    idx = global_start
    while idx < N:
        B[idx] = A[idx] * 2.0
        idx += grid_size
```

### When to Use Grid-Stride Loops

| Scenario | Grid-Stride? | Rationale |
|----------|-------------|-----------|
| N is known and small | No | Direct mapping is simpler |
| N is very large | Yes | Avoids launching too many blocks |
| N is variable | Yes | Handles any N with fixed launch config |
| Want maximum occupancy | Yes | Fixed grid size allows occupancy tuning |

### Example: Grid-Stride Reduction

```python
import tilelang.language as T

@T.Kernel(num_blocks, threads=256)
def grid_stride_reduce(A, out, N):
    tid = T.get_thread_binding(0)
    bid = T.get_block_binding(0)
    block_size = T.get_thread_extent(0)
    grid_size = T.get_block_extent(0) * block_size

    global_start = bid * block_size + tid

    # Partial sum across all grid-stride elements
    partial_sum = 0.0
    idx = global_start
    while idx < N:
        partial_sum += A[idx]
        idx += grid_size

    # Store partial sum to shared memory
    smem = T.alloc_shared([256], "float32")
    smem[tid] = partial_sum
    T.sync_shared_memory()

    # Tree reduction within the block
    stride = 128
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        stride //= 2
        T.sync_shared_memory()

    # Thread 0 writes the block result
    if tid == 0:
        out[bid] = smem[0]
```

---

## Dynamic Shared Memory Patterns

Dynamic shared memory allows kernels to allocate shared memory at launch time, with the size determined by a runtime parameter.

### Basic Dynamic Shared Memory

```python
import tilelang.language as T

# Allocate dynamic shared memory
# The size is determined at kernel launch time
smem_size = 48 * 1024  # 48 KB
smem = T.alloc_shared_dynamic(smem_size, "float32")
```

### Pattern: Shared Memory Double Buffering

```python
import tilelang.language as T

# Split shared memory into two buffers for double buffering
total_smem = 2 * block_M * block_K * element_size
smem = T.alloc_shared_dynamic(total_smem, "float16")

# Use the first half for buffer 0, second half for buffer 1
buffer_0 = smem[0 : block_M * block_K]
buffer_1 = smem[block_M * block_K : 2 * block_M * block_K]

# Double-buffered GEMM
for k in range(0, K, block_K):
    stage = (k // block_K) % 2
    current = buffer_0 if stage == 0 else buffer_1
    next_buf = buffer_1 if stage == 0 else buffer_0

    # Compute on current buffer
    T.gemm(current, B_smem, C_local)

    # Load into next buffer (overlaps with compute if pipelined)
    T.copy(A_global[k + block_K : k + 2 * block_K], next_buf)
    T.sync_shared_memory()
```

### Pattern: Shared Memory Workspace

```python
import tilelang.language as T

# Use shared memory as a temporary workspace for intermediate results
workspace = T.alloc_shared([block_M, block_N], "float32")

# Phase 1: Compute intermediate results into workspace
T.gemm(A_smem, B_smem, workspace)

# Phase 2: Apply activation to workspace
with T.Parallel(block_M, block_N) as i, j:
    workspace[i, j] = T.exp(workspace[i, j])

# Phase 3: Use workspace in next computation
T.gemm(workspace, D_smem, E_local)
```

### Bank Conflict Avoidance

Shared memory is organized into 32 banks, each 4 bytes wide. Concurrent accesses to the same bank by different threads cause bank conflicts:

```python
# Avoid bank conflicts by padding the shared memory allocation
# For float32: pad by 1 element per row to avoid same-bank access
padded_width = block_N + 1  # +1 to break bank conflicts
smem = T.alloc_shared([block_M, padded_width], "float32")

# Access without padding in the index (the padding is transparent)
for i in range(block_M):
    for j in range(block_N):
        smem[i, j] = data[i, j]
```

---

## @tilelang.jit Decorator

### Signature

```python
@tilelang.jit(
    out_idx=None,               # Output tensor indices
    target="cuda",              # Compilation target
    execution_backend="dlpack", # Execution backend
    verbose=False,              # Verbose output
    pass_configs=None,          # Compiler pass configurations
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out_idx` | list or None | `None` | List of indices of the output tensors in the kernel's return value. For example, `out_idx=[2]` means the third returned tensor is the primary output. If `None`, all returned tensors are considered outputs. |
| `target` | str | `"cuda"` | Compilation target. Options: `"cuda"` (NVIDIA GPU), `"hip"` (AMD GPU), `"cpu"` (CPU). |
| `execution_backend` | str | `"dlpack"` | Backend for tensor interchange. Options: `"dlpack"` (DLPack tensor protocol), `"torch"` (PyTorch tensors), `"cupy"` (CuPy arrays). |
| `verbose` | bool | `False` | If `True`, prints detailed compilation information including generated IR, PTX code, and compilation times. |
| `pass_configs` | dict or None | `None` | Configuration for TileLang compiler passes. Allows fine-grained control over optimization passes. |

### out_idx Parameter

The `out_idx` parameter specifies which of the kernel's returned tensors are the primary outputs. This affects:

1. **Memory allocation**: Output tensors are allocated automatically.
2. **Gradient computation**: Only primary outputs participate in autograd (when supported).
3. **Caching**: The cache key includes output tensor metadata.

```python
# Single output: the first returned tensor
@tilelang.jit(out_idx=[0])
def kernel_single_output(A, B):
    C = A + B
    return C

# Multiple outputs
@tilelang.jit(out_idx=[0, 1])
def kernel_multi_output(A, B):
    C = A + B
    D = A * B
    return C, D

# Output is the third returned buffer
@tilelang.jit(out_idx=[2])
def kernel_with_temp(A, B, M, N, K):
    A_smem = T.alloc_shared([M, K], "float16")  # Index 0 (not output)
    B_smem = T.alloc_shared([K, N], "float16")  # Index 1 (not output)
    C = T.alloc_shared([M, N], "float16")        # Index 2 (output)
    return C
```

### target Parameter

```python
# Compile for NVIDIA CUDA
@tilelang.jit(target="cuda")
def cuda_kernel():
    pass

# Compile for AMD HIP
@tilelang.jit(target="hip")
def hip_kernel():
    pass

# Compile for CPU
@tilelang.jit(target="cpu")
def cpu_kernel():
    pass
```

### execution_backend Parameter

```python
# Use DLPack for generic tensor interchange (works with PyTorch, JAX, etc.)
@tilelang.jit(execution_backend="dlpack")
def dlpack_kernel():
    pass

# Use PyTorch-specific backend
@tilelang.jit(execution_backend="torch")
def torch_kernel():
    pass
```

### verbose Parameter

```python
# Enable verbose output for debugging compilation issues
@tilelang.jit(verbose=True)
def verbose_kernel():
    pass
# Prints:
# [TileLang] Compiling kernel...
# [TileLang] Generated IR: ...
# [TileLang] Lowered to PTX: ...
# [TileLang] Compilation time: 1.23s
# [TileLang] Kernel cache key: ...
```

### pass_configs Parameter

```python
# Configure specific compiler passes
@tilelang.jit(pass_configs={
    "tir.disable_vectorize": False,
    "tir.unroll_max_step": 8,
    "tl.enable_warp_specialized": True,
    "tl.shared_memory_scope": "dynamic",
})
def configured_kernel():
    pass
```

### Common pass_configs Options

| Config Key | Type | Default | Description |
|-----------|------|---------|-------------|
| `"tir.disable_vectorize"` | bool | `False` | Disable auto-vectorization |
| `"tir.unroll_max_step"` | int | `8` | Maximum unroll factor |
| `"tl.enable_warp_specialized"` | bool | `False` | Enable warp-specialized codegen |
| `"tl.shared_memory_scope"` | str | `"static"` | `"static"` or `"dynamic"` shared memory |
| `"tl.use_async_copy"` | bool | `True` | Use async memory copies (TMA/CpAsync) |
| `"tl.enable_cluster"` | bool | `False` | Enable thread block clusters |
| `"tl.max_shared_memory"` | int | Auto | Maximum shared memory per block (bytes) |

---

## Kernel Compilation Modes

### Lazy Compilation (Default)

With lazy compilation, the kernel is compiled on the first call with specific tensor shapes and types. Subsequent calls with the same shapes/types reuse the cached compiled kernel.

```python
@tilelang.jit(out_idx=[2])
def my_kernel(M, N, K, block_M=64, block_N=64, block_K=32, in_dtype="float16"):
    # ... kernel body ...
    return C

# First call: compiles the kernel (may take a few seconds)
kernel_func = my_kernel(M=1024, N=1024, K=512)
result = kernel_func(A_tensor, B_tensor)

# Second call with same shapes: uses cached kernel (instant)
result2 = kernel_func(A_tensor2, B_tensor2)
```

### Eager Compilation

With eager compilation, the kernel is compiled immediately when the decorated function is created, before any data is provided:

```python
# Force compilation at definition time
@tilelang.jit(out_idx=[2], eager=True)
def eager_kernel(M=1024, N=1024, K=512, block_M=64, block_N=64, block_K=32):
    # ... kernel body ...
    return C

# Kernel is already compiled, no compilation delay at call time
result = eager_kernel(A_tensor, B_tensor)
```

### Compilation Pipeline

The TileLang compilation pipeline consists of several stages:

```
1. Python DSL Code
   |
   v
2. TileLang Frontend Parser
   |  - Converts Python functions to TVM TIR (Tensor IR)
   |  - Resolves buffer allocations, data types, shapes
   v
3. TIR Optimization Passes
   |  - Loop unrolling, vectorization
   |  - Shared memory layout optimization
   |  - Bank conflict analysis
   |  - Warp specialization
   v
4. TIR Lowering
   |  - Convert high-level operations to hardware instructions
   |  - GEMM -> MMA/WGMMA/TCGEN05 selection
   |  - Memory copy -> CpAsync/TMA selection
   |  - Reduction -> Warp shuffle / shared memory reduction
   v
5. Code Generation
   |  - Generate PTX or CUBIN for the target GPU
   |  - Register allocation
   |  - Instruction scheduling
   v
6. Compiled Kernel
   - Ready for execution with input tensors
```

---

## Kernel Caching

### How Caching Works

TileLang caches compiled kernels to avoid recompilation overhead. The cache key is determined by:

1. **Kernel function**: The Python function definition (including its source code hash).
2. **Compilation parameters**: `target`, `pass_configs`, and other JIT parameters.
3. **Runtime parameters**: Tensor shapes, data types, and any compile-time constants.

```
Cache Key = hash(kernel_source, jit_params, tensor_shapes, tensor_dtypes)

If cache hit: reuse compiled kernel (0ms overhead)
If cache miss: compile kernel (100ms - 10s depending on complexity)
```

### Cache Behavior

```python
@tilelang.jit(out_idx=[2])
def cached_kernel(M, N, K, block_M=64, in_dtype="float16"):
    # ... kernel body ...
    return C

# Call 1: Cache miss -> compile
result1 = cached_kernel(M=1024, N=1024, K=512)(A, B)

# Call 2: Cache hit (same M, N, K, block_M, in_dtype) -> reuse
result2 = cached_kernel(M=1024, N=1024, K=512)(A2, B2)

# Call 3: Cache miss (different K) -> recompile
result3 = cached_kernel(M=1024, N=1024, K=1024)(A3, B3)

# Call 4: Cache hit (same as call 3) -> reuse
result4 = cached_kernel(M=1024, N=1024, K=1024)(A4, B4)
```

### Cache Invalidation

The kernel cache is invalidated when:

1. The kernel source code changes (detected via source hash).
2. The Python process restarts (in-memory cache is lost).
3. The TileLang version changes.
4. The GPU driver is updated (PTX may need recompilation).

### Persistent Cache

TileLang can optionally persist compiled kernels to disk for cross-session reuse:

```python
# Enable persistent caching
import tilelang
tilelang.enable_cache(cache_dir="/path/to/cache")

# Compiled kernels are saved to disk and reused across sessions
@tilelang.jit(out_idx=[2])
def persistent_kernel(M, N, K):
    # ...
    return C
```

---

## Kernel Execution and Tensor Supply

### Tensor Interface

TileLang kernels accept PyTorch tensors (or other framework tensors via DLPack) as input and return output tensors:

```python
import torch
import tilelang

@tilelang.jit(out_idx=[2])
def matmul_kernel(M, N, K, block_M=64, block_N=64, block_K=32, in_dtype="float16"):
    A = T.alloc_shared([block_M, block_K], in_dtype)
    B = T.alloc_shared([block_K, block_N], in_dtype)
    C = T.alloc_local([block_M, block_N], "float32")

    T.clear(C)
    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A)
        T.copy(B_global[k:k+block_K], B)
        T.sync_shared_memory()
        T.gemm(A, B, C)

    T.copy(C, C_global)
    return C_global

# Compile the kernel
kernel = matmul_kernel(M=512, N=512, K=256)

# Create input tensors
A = torch.randn(512, 256, dtype=torch.float16, device="cuda")
B = torch.randn(256, 512, dtype=torch.float16, device="cuda")

# Execute: kernel allocates output tensor and returns it
C = kernel(A, B)
# C is a torch.Tensor of shape [512, 512], dtype float32, on CUDA
```

### Input Tensor Requirements

| Requirement | Description |
|------------|-------------|
| Device | Must be on the correct device (CUDA for GPU kernels, CPU for CPU kernels). |
| Dtype | Must match the kernel's expected data type. |
| Contiguity | Must be contiguous in memory (no strides). Use `tensor.contiguous()` if needed. |
| Shape | Must match the kernel's expected shape. |

### Output Tensor Allocation

Output tensors are automatically allocated by the TileLang runtime:

```python
# The kernel allocates output tensors based on the shapes specified in out_idx
C = kernel(A, B)
# C is automatically allocated with the correct shape, dtype, and device

# Multiple outputs are returned as a tuple
@tilelang.jit(out_idx=[0, 1])
def multi_output_kernel(M, N):
    X = T.alloc_shared([M, N], "float16")
    Y = T.alloc_shared([M, N], "float16")
    return X, Y

kernel = multi_output_kernel(M=256, N=256)
X, Y = kernel(input_tensor)
```

### Supplying Pre-allocated Output Tensors

For performance-critical applications, you can pre-allocate output tensors to avoid allocation overhead:

```python
# Pre-allocate output tensor
C = torch.empty(512, 512, dtype=torch.float32, device="cuda")

# Supply pre-allocated output (kernel writes directly into C)
kernel(A, B, out=C)
```

---

## Practical Examples

### Complete Matmul Kernel with Full Framework Usage

```python
import torch
import tilelang
import tilelang.language as T
from tilelang.language import GemmWarpPolicy

@tilelang.jit(out_idx=[2], target="cuda", verbose=False)
def matmul(
    M: int, N: int, K: int,
    block_M: int = 128, block_N: int = 128, block_K: int = 32,
    in_dtype: str = "float16", out_dtype: str = "float16",
    accum_dtype: str = "float32",
    num_stages: int = 2,
):
    # Shared memory with multi-stage buffering
    A_smem = T.alloc_shared([num_stages, block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([num_stages, block_K, block_N], in_dtype)

    # Register accumulator
    C_local = T.alloc_local([block_M, block_N], accum_dtype)

    T.clear(C_local)

    # Pipelined K loop
    for k in T.Pipelined(K // block_K, num_stages=num_stages):
        s = k % num_stages

        T.copy(A_global[k * block_K : (k+1) * block_K], A_smem[s])
        T.copy(B_global[k * block_K : (k+1) * block_K], B_smem[s])
        T.sync_shared_memory()

        T.gemm(
            A_smem[s], B_smem[s], C_local,
            policy=GemmWarpPolicy.Square,
        )

    T.copy(C_local, C_global)
    return C_global

# Compile and run
M, N, K = 2048, 2048, 1024
kernel = matmul(M=M, N=N, K=K, block_M=128, block_N=128, block_K=32)

A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(K, N, dtype=torch.float16, device="cuda")

C = kernel(A, B)
print(C.shape)  # torch.Size([2048, 2048])
print(C.dtype)  # torch.float16
```

### Kernel with Cluster Cooperation (Hopper)

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def cluster_matmul(
    M, N, K, block_M=128, block_N=128, block_K=64,
    in_dtype="float16",
):
    # 2-block cluster for cooperative GEMM
    A_smem = T.alloc_shared([block_M, block_K], in_dtype)
    B_smem = T.alloc_shared([block_K, block_N], in_dtype)
    C_local = T.alloc_local([block_M, block_N], "float32")

    mbar = T.alloc_shared([1], "uint64")
    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem, mbar=mbar)
        T.copy(B_global[k:k+block_K], B_smem, mbar=mbar)
        T.wait_memory_barrier(mbar)

        T.tcgen05_gemm(A_smem, B_smem, C_local, mbar=mbar, use_2cta=True)

    T.wait_memory_barrier(mbar)
    T.copy(C_local, C_global)
    return C_global
```

### Debugging with CPU Kernel

```python
import tilelang
import tilelang.language as T

# CPU kernel for debugging
@tilelang.jit(out_idx=[0], target="cpu")
def debug_matmul(M, N, K, in_dtype="float32"):
    A = T.alloc_shared([M, K], in_dtype)
    B = T.alloc_shared([K, N], in_dtype)
    C = T.alloc_shared([M, N], in_dtype)

    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    return C

# Run on CPU for easy debugging
import numpy as np
A = torch.randn(4, 4, dtype=torch.float32)
B = torch.randn(4, 4, dtype=torch.float32)
kernel = debug_matmul(M=4, N=4, K=4)
C = kernel(A, B)
print(C)  # Easy to inspect and verify
```

### Integration with External CUDA Kernel

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[1])
def hybrid_pipeline(M, N, K, block_M=64, block_N=64, block_K=32):
    # Phase 1: TileLang data movement and tiling
    A_smem = T.alloc_shared([block_M, block_K], "float16")
    B_smem = T.alloc_shared([block_K, block_N], "float16")
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Phase 2: Custom CUDA activation
    custom_activation = T.CUDASourceCodeKernel("""
    __global__ void custom_activation(half* data, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            half val = data[idx];
            half sq = __hmul(val, val);
            // Swish-like activation: x * sigmoid(x)
            half one = __float2half(1.0f);
            half neg_val = __hneg(val);
            // Approximate exp using __expf
            float exp_val = __expf(__half2float(neg_val));
            half sigmoid_val = __hdiv(one, __hadd(one, __float2half(exp_val)));
            data[idx] = __hmul(val, sigmoid_val);
        }
    }
    """, entry_name="custom_activation")

    T.copy(C_local, C_global)

    return C_global

# Apply custom CUDA activation after TileLang GEMM
```

---

## Summary

| Component | Purpose | Key API |
|-----------|---------|---------|
| `T.Kernel` | Define kernel launch context | `blocks`, `threads`, `cluster_dims` |
| `KernelLaunchFrame` | Query runtime info | `Current()`, `blocks`, `threads` |
| `T.CUDASourceCodeKernel` | Integrate external CUDA | `source_code_or_path`, `entry_name` |
| Thread accessors | Get thread/block indices | `get_thread_binding`, `get_block_binding` |
| `@tilelang.jit` | JIT compilation | `out_idx`, `target`, `pass_configs` |
| Cache | Avoid recompilation | Automatic by default |
| Execution | Run with PyTorch tensors | Call compiled kernel directly |

The TileLang kernel framework provides a complete pipeline from Python-embedded kernel definition through JIT compilation to GPU execution. The framework is designed to make common patterns (like tiled GEMM) easy to express while providing escape hatches (like `T.CUDASourceCodeKernel`) for specialized operations that require hand-written CUDA code.
