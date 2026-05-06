# Chapter 2: Getting Started

## Installation

### From PyPI (Recommended)
```bash
pip install triton
```
Binary wheels available for CPython 3.10-3.14 on Linux.

### From Source

```bash
git clone https://github.com/triton-lang/triton.git
cd triton

# Install build dependencies
pip install -r python/requirements.txt

# Install in editable mode
pip install -e .
```

### With Virtual Environment
```bash
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt
pip install -e .
```

### Building with Custom LLVM

Triton normally downloads a prebuilt LLVM. To use a custom LLVM:

```bash
# Quick method
make dev-install-llvm
```

Or manually:
1. Check `cmake/llvm-hash.txt` for the required LLVM commit
2. Clone and build LLVM:
```bash
cd $HOME/llvm-project
mkdir build && cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
ninja
```
3. Build Triton with custom LLVM:
```bash
export LLVM_BUILD_DIR=$HOME/llvm-project/build
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
pip install -e .
```

## Build Tips

| Environment Variable | Purpose |
|---------------------|---------|
| `TRITON_BUILD_WITH_CLANG_LLD=true` | Use clang and lld for faster builds |
| `TRITON_BUILD_WITH_CCACHE=true` | Enable ccache |
| `TRITON_HOME=/some/path` | Change `.triton` cache directory location |
| `MAX_JOBS=N` | Limit parallel build jobs (helps with OOM) |
| `--no-build-isolation` | Faster nop builds with pip install |

## Running Tests

```bash
# One-time setup (reinstalls local Triton because torch overwrites it)
make dev-install

# Run all tests (requires GPU)
make test

# Run tests without GPU
make test-nogpu

# Run specific pytest test
pytest python/test/unit/language/test_core.py::test_add -s --tb=short

# Run lit tests (no GPU required)
cd BUILD_DIR && ninja triton-opt
lit -v test/TritonGPU/some_test.mlir
```

### Build Directory
```bash
# Get build directory
PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())'
```

## Quick Start: First Kernel

### Vector Addition
```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,        # Pointer to first input
    y_ptr,        # Pointer to second input
    output_ptr,   # Pointer to output
    n_elements,   # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    # Get program ID (which block this thread block processes)
    pid = tl.program_id(axis=0)

    # Create block starting offset
    block_start = pid * BLOCK_SIZE

    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for out-of-bounds
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch function
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# Test
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
result = add(x, y)
assert torch.allclose(result, x + y)
```

### Fused Softmax
```python
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)

    # Compute exponentials
    numerator = tl.exp(row_minus_max)

    # Compute denominator
    denominator = tl.sum(numerator, axis=0)

    # Compute softmax
    softmax_output = numerator / denominator

    # Write back
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

### Matrix Multiplication with Autotuning
```python
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # ... kernel implementation
    pass
```

## Key Concepts for Beginners

### Program Grid
Triton kernels run on a grid of programs. Each program is a thread block on the GPU. Use:
- `tl.program_id(axis)` - Get the current program's ID along an axis
- `tl.num_programs(axis)` - Get the total number of programs along an axis

### Block-Level Programming
Unlike CUDA where you program individual threads, Triton programs at the block level:
- `tl.arange(0, BLOCK_SIZE)` creates a range of offsets
- Operations on tensors are implicitly parallel across the block
- The compiler handles thread allocation and synchronization

### Memory Operations
```python
# Load with mask (essential for boundary handling)
data = tl.load(ptr + offsets, mask=offsets < n)

# Store with mask
tl.store(ptr + offsets, data, mask=offsets < n)

# Load with boundary value
data = tl.load(ptr + offsets, mask=offsets < n, other=0.0)
```

### Compile-Time Constants
Use `tl.constexpr` annotation for values known at compile time:
```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):  # Must be constexpr
    offsets = tl.arange(0, BLOCK_SIZE)  # Needs constexpr
```

### Grid Launch
```python
# Static grid
kernel[(num_blocks, 1, 1)](args)

# Dynamic grid using lambda
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
kernel[grid](args)
```

## Supported Hardware

| Platform | Architecture | Minimum Version |
|----------|-------------|-----------------|
| NVIDIA | CUDA Compute Capability 8.0+ | Ampere (A100) and newer |
| AMD | ROCm | 6.2+ |
| CPU | Under development | N/A |
