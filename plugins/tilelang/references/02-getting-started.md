# TileLang Getting Started Guide

## Table of Contents

- [1. Installation](#1-installation)
- [2. Environment Setup](#2-environment-setup)
- [3. Your First Kernel: Vector Addition](#3-your-first-kernel-vector-addition)
- [4. Basic GEMM: Step-by-Step Walkthrough](#4-basic-gemm-step-by-step-walkthrough)
- [5. Understanding the Compilation Flow](#5-understanding-the-compilation-flow)
- [6. Using PyTorch with TileLang](#6-using-pytorch-with-tilelang)
- [7. Profiling and Benchmarking Kernels](#7-profiling-and-benchmarking-kernels)
- [8. Advanced: Pipelined GEMM](#8-advanced-pipelined-gemm)
- [9. Advanced: Eager Mode with @tilelang.jit](#9-advanced-eager-mode-with-tilelangjit)
- [10. Common Pitfalls and Debugging Tips](#10-common-pitfalls-and-debugging-tips)
- [11. Quick Reference Table](#11-quick-reference-table)

---

## 1. Installation

### 1.1 Install with Pip (Recommended)

The quickest way to get started is to install the latest release from PyPI:

```bash
pip install tilelang
```

This installs TileLang with all Python dependencies. The C++ extension is pre-built for common platforms.

### 1.2 Install from GitHub

```bash
pip install git+https://github.com/tile-ai/tilelang
```

### 1.3 Install from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/tile-ai/tilelang.git
cd tilelang

# Install required system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev \
    build-essential cmake libedit-dev libxml2-dev

# Install in editable mode with verbose output
pip install -e . -v
```

### 1.4 Install with Docker

TileLang provides Docker images with all dependencies pre-installed:

```bash
# Build the Docker image
docker build -t tilelang -f docker/Dockerfile .

# Run interactively
docker run --gpus all -it tilelang bash
```

### 1.5 Install Nightly Builds

For access to the latest features and improvements before official releases:

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly
```

Nightly builds contain the most recent code changes but may be less stable than official releases.

### 1.6 Verify Installation

```python
import tilelang
print(tilelang.__version__)
```

---

## 2. Environment Setup

### 2.1 Python Version Requirements

| Version | Support Status |
|---------|---------------|
| Python 3.8 | Supported until v0.1.6.post2 (last compatible version) |
| Python 3.9 | Supported |
| Python 3.10 | Supported (recommended) |
| Python 3.11 | Supported (recommended) |
| Python 3.12 | Supported |

### 2.2 CUDA Toolkit

TileLang requires the CUDA Toolkit for NVIDIA GPU support:

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| CUDA Toolkit | 11.0+ | 12.x |
| cuDNN | Optional | Latest |
| NVIDIA Driver | 450+ | 535+ |

Verify CUDA installation:

```bash
nvcc --version
nvidia-smi
```

### 2.3 AMD ROCm (for AMD GPU Support)

For AMD GPU support, install the ROCm toolkit:

```bash
# Verify ROCm installation
rocm-smi
hipcc --version
```

### 2.4 PyTorch

TileLang integrates with PyTorch for tensor management:

```bash
pip install torch  # Install PyTorch with CUDA support
```

Verify PyTorch CUDA:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

### 2.5 Environment Variables

TileLang behavior can be configured via environment variables:

```bash
# Set compilation target
export TILELANG_TARGET=cuda          # or: hip, cpu, auto

# Set execution backend
export TILELANG_EXECUTION_BACKEND=nvrtc  # or: auto, dlpack, tvm_ffi, cython, torch

# Enable verbose output
export TILELANG_VERBOSE=1

# Set cache directory
export TILELANG_CACHE_DIR=~/.tilelang/cache
```

### 2.6 Import Conventions

```python
import tilelang
import tilelang.language as T
```

The `T` namespace provides all DSL operations as `T.copy`, `T.gemm`, `T.alloc_shared`, etc.

---

## 3. Your First Kernel: Vector Addition

Let's start with the simplest possible TileLang kernel: element-wise vector addition.

### 3.1 Complete Example

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def vector_add(N, block_size=128):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        # Launch a 1D grid of blocks, each with `block_size` threads
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            # Get the thread index within this block
            tx = T.get_thread_binding(0)

            # Compute the global index
            idx = bx * block_size + tx

            # Bounds check and compute
            if idx < N:
                C[idx] = A[idx] + B[idx]

    return add_kernel

# Compile the kernel
kernel = vector_add(1024, 128)

# Create input tensors
a = torch.randn(1024, device="cuda", dtype=torch.float32)
b = torch.randn(1024, device="cuda", dtype=torch.float32)

# Execute
c = kernel(a, b)

# Verify
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-5, atol=1e-5)
print("Vector addition passed!")
```

### 3.2 Line-by-Line Explanation

#### The `@tilelang.jit` Decorator

```python
@tilelang.jit(out_idx=[-1])
def vector_add(N, block_size=128):
```

- `@tilelang.jit` wraps the function as a JIT-compilable kernel.
- `out_idx=[-1]` means the last argument tensor (`C`) is the output to return.
- `N` and `block_size` are compile-time parameters that can be varied.

#### The `@T.prim_func` Decorator

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
):
```

- `@T.prim_func` marks this function as a TileLang primitive function.
- `A: T.Tensor((N,), "float32")` declares an input tensor of shape `(N,)` with float32 dtype.
- `C: T.Tensor((N,), "float32")` declares an output tensor. The `out_idx=[-1]` in the outer decorator tells TileLang to return this tensor's result.

#### The `T.Kernel` Context Manager

```python
with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
```

- `T.Kernel(grid_x, threads=block_size)` launches a 1D grid.
- The first argument `T.ceildiv(N, block_size)` computes the number of blocks needed.
- `threads=block_size` sets the number of threads per block (default: 128).
- `bx` is the block index variable (`blockIdx.x` in CUDA).

#### Thread Binding

```python
tx = T.get_thread_binding(0)
```

- `T.get_thread_binding(0)` returns `threadIdx.x`.
- For 2D thread blocks, use `T.get_thread_binding(1)` for `threadIdx.y`.

#### Computation

```python
idx = bx * block_size + tx

if idx < N:
    C[idx] = A[idx] + B[idx]
```

- Computes the global element index from block and thread indices.
- The bounds check `if idx < N:` handles cases where N is not a multiple of block_size.
- `C[idx] = A[idx] + B[idx]` is standard element-wise arithmetic.

#### Compilation and Execution

```python
kernel = vector_add(1024, 128)  # Compile for N=1024, block_size=128
c = kernel(a, b)                # Execute with PyTorch tensors
```

- `vector_add(1024, 128)` triggers JIT compilation and returns a `JITKernel`.
- `kernel(a, b)` executes the kernel on GPU, passing PyTorch tensors directly.

### 3.3 Using Parallel Copy Instead of Manual Indexing

For element-wise operations, TileLang provides a more idiomatic approach using `T.Parallel` and `T.copy`:

```python
@tilelang.jit(out_idx=[-1])
def vector_add_parallel(N, block_size=128):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            # Allocate a shared/local buffer for the tile
            A_local = T.alloc_fragment((block_size,), "float32")
            B_local = T.alloc_fragment((block_size,), "float32")
            C_local = T.alloc_fragment((block_size,), "float32")

            # Copy tile from global memory
            T.copy(A[bx * block_size], A_local)
            T.copy(B[bx * block_size], B_local)

            # Element-wise compute in parallel
            for i in T.Parallel(block_size):
                C_local[i] = A_local[i] + B_local[i]

            # Copy result back
            T.copy(C_local, C[bx * block_size])

    return add_kernel
```

---

## 4. Basic GEMM: Step-by-Step Walkthrough

This section walks through a complete matrix multiplication kernel, explaining every construct.

### 4.1 Complete GEMM Example

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),    # Input matrix A (M x K)
        B: T.Tensor((K, N), dtype),    # Input matrix B (K x N)
        C: T.Tensor((M, N), dtype),    # Output matrix C (M x N)
    ):
        # Launch a 2D grid: bx covers N dimension, by covers M dimension
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Allocate shared memory for the current tile
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Allocate register file for accumulation
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize accumulator to zero
            T.clear(C_local)

            # Loop over K dimension in tiles of block_K
            for k in T.serial(T.ceildiv(K, block_K)):
                # Load tiles from global to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Perform tile-level matrix multiplication
                # Dispatches to Tensor Core (MMA/WGMMA) automatically
                T.gemm(A_shared, B_shared, C_local)

            # Store result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm
```

### 4.2 Step-by-Step Breakdown

#### Step 1: Function Signature and Parameters

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
```

All parameters are compile-time constants. Key parameters:
- `M, N, K`: Matrix dimensions (can be symbolic or concrete).
- `block_M, block_N, block_K`: Tile sizes that determine the work per thread block.
- `dtype`: Data type for input/output matrices.
- `accum_dtype`: Data type for accumulation (typically `float32` for numerical stability).

#### Step 2: Tensor Declarations

```python
A: T.Tensor((M, K), dtype),    # Input matrix A
B: T.Tensor((K, N), dtype),    # Input matrix B
C: T.Tensor((M, N), dtype),    # Output matrix C
```

`T.Tensor(shape, dtype)` declares a tensor parameter in global memory. The shape can include symbolic dimensions.

#### Step 3: Grid Launch

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
```

- **First argument** (`T.ceildiv(N, block_N)`): Grid dimension X = number of tiles along N.
- **Second argument** (`T.ceildiv(M, block_M)`): Grid dimension Y = number of tiles along M.
- **`threads=128`**: 128 threads per block.
- **`(bx, by)`**: Block indices. `bx` corresponds to the N dimension, `by` to the M dimension.

Each thread block computes one (block_M x block_N) tile of the output matrix C.

#### Step 4: Memory Allocation

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)    # Shared memory
B_shared = T.alloc_shared((block_K, block_N), dtype)    # Shared memory
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # Register file
```

- **`T.alloc_shared`**: Allocates shared memory visible to all threads in the block. Used for staging data from global memory.
- **`T.alloc_fragment`**: Allocates register file memory (thread-private). Used for accumulation. The layout is automatically inferred for Tensor Core compatibility.

#### Step 5: Accumulator Initialization

```python
T.clear(C_local)
```

Fills `C_local` with zeros. Equivalent to `T.fill(C_local, 0)`.

#### Step 6: Tiled Loop over K

```python
for k in T.serial(T.ceildiv(K, block_K)):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[k * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

- `T.serial(n)`: A sequential loop from 0 to n-1.
- `T.copy(src, dst)`: Copies a tile from global memory to shared memory. The source is a sub-region of the global tensor.
  - `A[by * block_M, k * block_K]` selects a (block_M x block_K) tile starting at row `by * block_M`, column `k * block_K`.
- `T.gemm(A, B, C)`: Performs `C += A @ B` using Tensor Cores. On Ampere/Hopper, this dispatches to MMA/WGMMA automatically.

#### Step 7: Write Results

```python
T.copy(C_local, C[by * block_M, bx * block_N])
```

Copies the accumulated result from registers back to global memory.

### 4.3 Running the GEMM

```python
# Compile
kernel = matmul(1024, 1024, 1024, 128, 128, 32)

# Prepare inputs
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Execute
c = kernel(a, b)

# Verify
ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("GEMM check passed!")
```

---

## 5. Understanding the Compilation Flow

### 5.1 What Happens When You Call `kernel(a, b)`

1. **Tensor Preparation**: PyTorch tensors are wrapped as DLPack tensors for interop.
2. **Cache Lookup**: TileLang checks if a kernel with the same parameters is already compiled.
3. **Compilation** (if cache miss):
   - The `@T.prim_func` body is traced into a TIR `PrimFunc`.
   - The TileLang engine runs transform passes: layout inference, lowering, optimization.
   - Target-specific codegen produces CUDA C++ or HIP source.
   - NVRTC or NVCC compiles the source to a GPU binary.
4. **Execution**: The compiled kernel is launched with the provided tensors.
5. **Output**: Result tensors are returned as PyTorch tensors.

### 5.2 Inspecting Generated Code

```python
# Get the generated CUDA source
print(kernel.get_kernel_source())

# Get the TIR (Tensor IR) representation
print(kernel.get_tir(M, N, K, BM, BN, BK).script())
```

### 5.3 Saving Debug Information

```python
@tilelang.jit(out_idx=[-1], debug_root_path="./debug")
def matmul(M, N, K, BM, BN, BK, dtype, accum_dtype):
    ...
```

This saves the generated kernel source and TIR script to `./debug/` on each compilation.

### 5.4 Compilation Targets

```python
# Explicitly target CUDA
kernel = tilelang.compile(func, target="cuda")

# Explicitly target AMD HIP
kernel = tilelang.compile(func, target="hip")

# CPU target
kernel = tilelang.compile(func, target="cpu")

# Auto-detect (default)
kernel = tilelang.compile(func, target="auto")
```

---

## 6. Using PyTorch with TileLang

### 6.1 Tensor Interoperability

TileLang kernels accept and return PyTorch tensors directly:

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def scale_kernel(N, block_size=128):
    @T.prim_func
    def kernel(A: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            A_local = T.alloc_fragment((block_size,), "float32")
            C_local = T.alloc_fragment((block_size,), "float32")
            T.copy(A[bx * block_size], A_local)
            for i in T.Parallel(block_size):
                C_local[i] = A_local[i] * 2.0
            T.copy(C_local, C[bx * block_size])
    return kernel

# PyTorch tensors work directly
kernel = scale_kernel(1024)
a = torch.randn(1024, device="cuda", dtype=torch.float32)
c = kernel(a)  # Returns a PyTorch tensor
assert isinstance(c, torch.Tensor)
```

### 6.2 Supported Data Types

| TileLang dtype | PyTorch dtype | Notes |
|---------------|---------------|-------|
| `T.float32` | `torch.float32` | Standard single precision |
| `T.float16` | `torch.float16` | Half precision |
| `T.bfloat16` | `torch.bfloat16` | Brain float |
| `T.int32` | `torch.int32` | Standard integer |
| `T.int8` | `torch.int8` | Quantized |
| `T.uint8` | `torch.uint8` | Unsigned byte |
| `T.float8_e4m3fn` | `torch.float8_e4m3fn` | FP8 (E4M3) |
| `T.float8_e5m2` | `torch.float8_e5m2` | FP8 (E5M2) |
| `T.int4` | N/A | Sub-byte, internal use |
| `T.int2` | N/A | Sub-byte, internal use |

### 6.3 Gradient Integration

While TileLang kernels are not differentiable through PyTorch autograd directly, you can write separate forward and backward kernels:

```python
@tilelang.jit(out_idx=[-1])
def matmul_fwd(M, N, K, BM, BN, BK, dtype, accum_dtype):
    # Forward kernel
    ...

@tilelang.jit(out_idx=[-1, -1, -1])
def matmul_bwd(M, N, K, BM, BN, BK, dtype, accum_dtype):
    # Backward kernel computing dA, dB, dC
    ...
```

### 6.4 Custom PyTorch Function

```python
class TileLangMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return matmul_kernel(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        # Custom backward using TileLang
        ...

tilelang_matmul = TileLangMatmul.apply
```

---

## 7. Profiling and Benchmarking Kernels

### 7.1 Built-in Profiler

Every compiled kernel provides a profiler:

```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)

# Get the profiler
profiler = kernel.get_profiler()

# Benchmark with CUDA events (default)
latency_ms = profiler.do_bench()
print(f"Latency: {latency_ms:.3f} ms")

# Benchmark with CUPTI for more accurate results
latency_ms = profiler.do_bench(backend="cupti")
print(f"CUPTI Latency: {latency_ms:.3f} ms")

# Benchmark with CUDA graphs for minimal launch overhead
latency_ms = profiler.do_bench(backend="cudagraph")

# Get specific quantiles
results = profiler.do_bench(quantiles=[0.5, 0.95, 0.99])
print(f"Median: {results[0]:.3f} ms, P95: {results[1]:.3f} ms")
```

### 7.2 Profiler Backends

| Backend | Method | Accuracy | Overhead |
|---------|--------|----------|----------|
| `event` | CUDA events | Good | Low |
| `cupti` | CUPTI profiler | Best | Medium |
| `cudagraph` | CUDA graph replay | Good | Lowest launch overhead |

### 7.3 Profiler Parameters

```python
profiler.do_bench(
    warmup=25,          # Target warmup time in ms (auto-calculated)
    rep=100,            # Target benchmark time in ms (auto-calculated)
    _n_warmup=0,        # Manual warmup iterations (0 = auto)
    _n_repeat=0,        # Manual benchmark iterations (0 = auto)
    quantiles=None,     # Percentiles to compute, e.g., [0.5, 0.95]
    fast_flush=True,    # Use fast L2 cache flush (int32 vs int8)
    backend="event",    # Profiler backend
    return_mode="mean", # Aggregation: "mean", "median", "min", "max"
)
```

### 7.4 Computing TFLOPS

```python
M, N, K = 1024, 1024, 1024
kernel = matmul(M, N, K, 128, 128, 32)
profiler = kernel.get_profiler()
latency_ms = profiler.do_bench()

# FLOPS for GEMM = 2 * M * N * K
flops = 2 * M * N * K
tflops = flops / (latency_ms * 1e-3) / 1e12
print(f"Performance: {tflops:.2f} TFLOPS")
```

### 7.5 Manual Benchmarking with torch.cuda.Event

```python
import torch

kernel = matmul(1024, 1024, 1024, 128, 128, 32)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Warmup
for _ in range(10):
    kernel(a, b)

# Measure
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    kernel(a, b)
end.record()
torch.cuda.synchronize()

avg_ms = start.elapsed_time(end) / 100
print(f"Average latency: {avg_ms:.3f} ms")
```

---

## 8. Advanced: Pipelined GEMM

Software pipelining overlaps data loading with computation for better performance. TileLang provides `T.Pipelined` for this:

```python
@tilelang.jit(out_idx=[-1])
def pipelined_matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            # Pipelined loop with 3 stages
            # Stage 0: Prologue (loads first tile)
            # Stage 1-N: Steady state (overlaps load of next tile with compute of current)
            # Final: Epilogue (computes last tile while storing results)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm
```

### How Pipelining Works

With `num_stages=3`, the compiler creates a 3-buffer rotation scheme:

```
Stage 0 (Prologue):  Load A[0], B[0] into buffer[0]
Stage 1 (Steady):    Load A[1], B[1] into buffer[1] | Compute C += A[0]*B[0]
Stage 2 (Steady):    Load A[2], B[2] into buffer[2] | Compute C += A[1]*B[1]
Stage 3 (Steady):    Load A[3], B[3] into buffer[0] | Compute C += A[2]*B[2]
...
Final (Epilogue):    Store C | Compute C += A[last]*B[last]
```

The load and compute operations in each stage overlap, hiding memory latency behind computation.

---

## 9. Advanced: Eager Mode with @tilelang.jit

Eager mode allows direct execution without explicitly returning a `@T.prim_func`:

```python
@tilelang.jit
def matmul(A, B, block_M=64, block_N=64, block_K=32, dtype=T.float16, accum_dtype=T.float32):
    # Declare compile-time constants
    M, N, K = T.const('M N K')

    # Annotate input tensor shapes (type annotations)
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]

    # Declare output tensor
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C

# Usage: pass tensors directly
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = matmul(a, b)  # Compiles and executes immediately
```

### Key Differences from Lazy Mode

| Feature | Lazy Mode | Eager Mode |
|---------|-----------|------------|
| Output declaration | `out_idx=[-1]` in `@tilelang.jit` | `T.empty()` in function body |
| Shape constants | Function parameters | `T.const('M N K')` |
| Input shapes | `@T.prim_func` annotations | Type annotations |
| Return value | Compiled `JITKernel` | Execution result tensor |
| Multiple outputs | `out_idx=[0, 2]` | Return multiple `T.empty()` tensors |

---

## 10. Common Pitfalls and Debugging Tips

### 10.1 Common Errors and Solutions

#### Error: "JITNoBuilderError: T.Kernel() can only be used inside @tilelang.jit or @T.prim_func context"

**Cause**: `T.Kernel()` was called outside a JIT/prim_func context.

**Solution**: Ensure the function is decorated with `@T.prim_func` or `@tilelang.jit`.

```python
# Wrong
def my_kernel():
    with T.Kernel(10, threads=128) as bx:  # Error!
        ...

# Correct
@T.prim_func
def my_kernel(...):
    with T.Kernel(10, threads=128) as bx:  # OK
        ...
```

#### Error: Shape mismatch in T.copy or T.gemm

**Cause**: The source and destination buffers have incompatible shapes.

**Solution**: Verify that tile dimensions match between copy sources and allocated buffers.

```python
# This will fail:
T.copy(A[by * 128, k * 32], A_shared)  # Expects (128, 32) tile
A_shared = T.alloc_shared((64, 32), dtype)   # But only (64, 32)!

# Correct:
A_shared = T.alloc_shared((128, 32), dtype)  # Match the tile size
```

#### Error: Out-of-bounds access

**Cause**: The grid dimensions don't cover all elements, or the block computes out-of-bounds indices.

**Solution**: Add bounds checks when dimensions aren't multiples of tile sizes, or ensure the compiler handles edge cases through safe value annotations.

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # Add bounds check if needed
    row = by * block_M
    col = bx * block_N
    if row < M and col < N:
        ...
```

#### Error: Shared memory overflow

**Cause**: Too much shared memory allocated per block.

**Solution**: Reduce tile sizes or reduce the number of pipeline stages. Typical shared memory limits are 48KB-228KB per SM.

```python
# Each allocation uses shared memory:
A_shared = T.alloc_shared((128, 32), "float16")  # 128 * 32 * 2 = 8KB
B_shared = T.alloc_shared((32, 128), "float16")  # 128 * 32 * 2 = 8KB
# Total: 16KB. With 3-stage pipeline: 3 * 16KB = 48KB.
```

#### Error: Register pressure too high

**Cause**: Too many registers per thread, reducing occupancy.

**Solution**: Reduce accumulator tile sizes or use `T.annotate_min_blocks_per_sm`.

```python
with T.Kernel(...):
    T.annotate_min_blocks_per_sm(2)  # Hint compiler to limit register usage
    ...
```

### 10.2 Debugging Techniques

#### Print Values from Kernels

```python
with T.Kernel(1, threads=128) as bx:
    tx = T.get_thread_binding(0)
    # Print a value from thread 0
    T.print(tx, "thread_idx:")
    T.print(A[tx], "A[tx]:")
```

#### Device Assertions

```python
T.device_assert(idx < N, "Index out of bounds")
```

#### Inspect Generated Code

```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)
print(kernel.get_kernel_source())
```

#### Save Debug Artifacts

```python
@tilelang.jit(out_idx=[-1], debug_root_path="./debug")
def matmul(...):
    ...
```

This saves:
- `./debug/tilelang_jit_kernel_<name>.c` -- Generated kernel source
- `./debug/tilelang_jit_program_<name>.py` -- TIR script

#### Verbose Compilation

```python
# Enable verbose output globally
import tilelang
tilelang.set_log_level("DEBUG")

# Or per-kernel
kernel = tilelang.compile(func, verbose=True)
```

### 10.3 Performance Tips

1. **Use pipelining**: `T.Pipelined` with `num_stages=2` or `3` significantly improves performance by overlapping memory access with computation.

2. **Choose good tile sizes**: Typical tile sizes:
   - `block_M = 128`, `block_N = 128`, `block_K = 32` for FP16 GEMM on A100/H100
   - Larger tiles use more shared memory and registers but may improve compute utilization

3. **Use Tensor Core-friendly dimensions**: Tile sizes should be multiples of the Tensor Core tile size:
   - MMA (SM80): 16x16x16 or 16x8x16
   - WGMMA (SM90): varies by dtype

4. **Enable swizzle for L2 cache**: `T.use_swizzle(panel_size=10)` can improve L2 cache hit rates for 2D kernel grids.

5. **Choose appropriate accumulation dtype**: Use `float32` accumulation for `float16` inputs to avoid numerical issues.

6. **Use the profiler**: Always benchmark with `profiler.do_bench()` to measure actual performance.

---

## 11. Quick Reference Table

### Kernel Structure

| Construct | Syntax | Description |
|-----------|--------|-------------|
| JIT decorator | `@tilelang.jit(out_idx=[-1])` | Compile and wrap kernel |
| PrimFunc | `@T.prim_func` | Define a primitive function |
| Tensor param | `A: T.Tensor((M, K), "float16")` | Declare input/output tensor |
| Kernel launch | `T.Kernel(grid_x, grid_y, threads=128)` | Launch GPU kernel |
| Block index | `with T.Kernel(...) as (bx, by):` | Get block indices |

### Memory Allocation

| Function | Syntax | Scope |
|----------|--------|-------|
| Shared memory | `T.alloc_shared((M, N), "float16")` | `shared.dyn` |
| Register file | `T.alloc_fragment((M, N), "float32")` | `local.fragment` |
| Local memory | `T.alloc_local((M, N), "float32")` | `local` |
| Scalar variable | `T.alloc_var("int32", 0)` | `local.var` |
| Global workspace | `T.alloc_global((M, N), "float32")` | `global` |
| Barrier | `T.alloc_barrier(128)` | `shared.barrier` |
| Output tensor | `T.empty([M, N], "float16")` | Eager mode output |

### Data Movement

| Function | Syntax | Description |
|----------|--------|-------------|
| Copy | `T.copy(src, dst)` | Synchronous copy (auto-dispatch) |
| Async copy | `T.async_copy(src, dst)` | Asynchronous copy (cp.async) |
| TMA copy | `T.tma_copy(src, dst, barrier=bar)` | TMA-based copy (SM90+) |
| Transpose | `T.transpose(src, dst)` | Transpose in shared memory |

### Computation

| Function | Syntax | Description |
|----------|--------|-------------|
| GEMM | `T.gemm(A, B, C)` | Matrix multiply (auto-dispatch) |
| WGMMA GEMM | `T.wgmma_gemm(A, B, C)` | Explicit Hopper WGMMA |
| TCGEN05 GEMM | `T.tcgen05_gemm(A, B, C, mbar=bar)` | Explicit Blackwell TCGEN05 |
| Fill | `T.fill(buf, value)` | Fill buffer with value |
| Clear | `T.clear(buf)` | Fill buffer with zeros |

### Reductions

| Function | Syntax | Description |
|----------|--------|-------------|
| Reduce sum | `T.reduce_sum(buf, out, dim=-1)` | Sum along dimension |
| Reduce max | `T.reduce_max(buf, out, dim=-1)` | Max along dimension |
| Reduce min | `T.reduce_min(buf, out, dim=-1)` | Min along dimension |
| Warp reduce sum | `T.warp_reduce_sum(val)` | Warp-level sum |
| Cumulative sum | `T.cumsum(src, dst, dim=0)` | Prefix sum |

### Loop Constructs

| Construct | Syntax | Description |
|-----------|--------|-------------|
| Serial | `T.serial(n)` | Sequential loop |
| Parallel | `T.Parallel(m, n)` | Parallel loop nest |
| Pipelined | `T.Pipelined(n, num_stages=3)` | Software pipeline |
| Unroll | `T.unroll(0, n)` | Unrolled loop |
| Vectorized | `T.vectorized(0, n)` | Vectorized loop |
| Persistent | `T.Persistent(domain, wave_size, index)` | Persistent kernel loop |

### Synchronization

| Function | Syntax | Description |
|----------|--------|-------------|
| Block sync | `T.sync_threads()` | `__syncthreads()` |
| Warp sync | `T.sync_warp()` | `__syncwarp()` |
| Barrier arrive | `T.barrier_arrive(bar)` | Arrive at mbarrier |
| Barrier wait | `T.barrier_wait(bar, parity)` | Wait on mbarrier |
| Cluster sync | `T.cluster_sync()` | Sync across cluster |

### Thread/Block Information

| Function | Syntax | Returns |
|----------|--------|---------|
| Thread binding | `T.get_thread_binding(0)` | `threadIdx.x` |
| All thread bindings | `T.get_thread_bindings()` | `[tx, ty, tz]` |
| Block binding | `T.get_block_binding(0)` | `blockIdx.x` |
| All block bindings | `T.get_block_bindings()` | `[bx, by, bz]` |
| Thread extent | `T.get_thread_extent(0)` | `blockDim.x` |
| Block extent | `T.get_block_extent(0)` | `gridDim.x` |
| Lane index | `T.get_lane_idx()` | Lane within warp |
| Warp index | `T.get_warp_idx()` | Warp within block |

### Warp Shuffle

| Function | Syntax | Description |
|----------|--------|-------------|
| Shuffle XOR | `T.shfl_xor(val, delta)` | XOR-based lane exchange |
| Shuffle down | `T.shfl_down(val, delta)` | Shift value down |
| Shuffle up | `T.shfl_up(val, delta)` | Shift value up |
| Shuffle broadcast | `T.shfl_sync(val, srcLane)` | Broadcast from lane |

### Annotations

| Function | Syntax | Description |
|----------|--------|-------------|
| Swizzle | `T.use_swizzle(panel_size=10)` | L2 cache swizzle pattern |
| Layout | `T.annotate_layout({buf: layout})` | Set buffer layout |
| Min blocks/SM | `T.annotate_min_blocks_per_sm(2)` | Occupancy hint |
| L2 hit ratio | `T.annotate_l2_hit_ratio({buf: 0.5})` | L2 cache hint |
| Safe value | `T.annotate_safe_value({buf: 0})` | Out-of-bounds safe value |

### Math Intrinsics

| Function | Syntax | Description |
|----------|--------|-------------|
| Fast exp | `T.__exp(x)` | Fast approximate exp |
| Fast log | `T.__log(x)` | Fast approximate log |
| Fast sin | `T.__sin(x)` | Fast approximate sin |
| Fast cos | `T.__cos(x)` | Fast approximate cos |
| IEEE sqrt | `T.ieee_fsqrt(x)` | IEEE-compliant sqrt |
| IEEE div | `T.ieee_fdiv(x, y)` | IEEE-compliant division |
| Packed add | `T.add2(x, y)` | float16x2/bfloat16x2 add |
| Packed mul | `T.mul2(x, y)` | float16x2/bfloat16x2 multiply |
