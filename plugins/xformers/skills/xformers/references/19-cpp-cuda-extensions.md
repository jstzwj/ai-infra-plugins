# 19 - C++/CUDA Extensions

## Overview

xFormers includes C++ and CUDA extensions for performance-critical operations that cannot be efficiently implemented in pure Python or Triton.

**Source**: `xformers/csrc/`

## Directory Structure

```
csrc/
├── attention/
│   ├── attention.cpp          # Main attention operators
│   ├── hip_fmha/              # ROCm FMHA kernels
│   │   ├── GENERATE_INSTANCES.md
│   │   ├── generate_instances.py
│   │   └── instances/         # Generated kernel instances
│   └── hip_decoder/           # ROCm decoder kernels
│       └── CMakeLists.txt
├── sparse24/                  # 2:4 sparsity CUDA kernels
│   ├── sparse24.cpp           # Main sparse24 operators
│   ├── sparse24_pack.cu       # Packing kernels
│   ├── sparse24_apply.cu      # Apply sparsity pattern
│   ├── sparse24_gemm.cu       # Sparse GEMM kernels
│   ├── sparse24_gemm_sm90.cu  # H100+ optimized kernels
│   ├── sparse24_largest_mask_2d.cu  # Mask computation
│   ├── meta_utils.cu          # Metadata utilities
│   ├── sparse24_metadata.h    # Metadata header
│   ├── compute_sparse_tile.h  # Tile computation header
│   ├── warp_tensor.h          # Warp-level tensor ops
│   └── static_sort.h          # Compile-time sorting
├── pt_stable_utils.cu         # PyTorch stable ABI utilities
├── pt_stable_utils.h          # Header
└── nvcc_info.cu               # NVCC compiler info
```

## Attention Extension

### `attention.cpp`

Registers attention operators for ROCm (AMD GPU) platforms using Composable Kernel (CK) instances.

**Operators:**
- Forward and backward attention kernels for FMHA
- Multiple kernel instances for different head dimensions, sequence lengths, and data types
- Generated via `generate_instances.py`

### CK (Composable Kernel) Backend

The ROCm backend uses AMD's Composable Kernel library:

```
hip_fmha/
├── generate_instances.py  # Generates kernel instances
└── instances/             # Generated C++ files
```

**Generation:**
```bash
python generate_instances.py
# Generates kernel instances for various configurations:
# - Different head dimensions (64, 128, 256)
# - Different data types (fp16, bf16, fp32)
# - Different sequence lengths
```

## 2:4 Sparsity Extension

### `sparse24.cpp`

Main operator registration for 2:4 sparsity operations:

- `sparse24_sparsify_both_ways` - Convert dense to sparse (both orientations)
- `sparse24_apply` - Apply sparsity pattern to new data
- `sparse24_apply_dense_output` - Apply with dense output
- `_sparse24_gemm` - Sparse-dense GEMM

### `sparse24_pack.cu`

Packing kernels that rearrange dense data into the 2:4 sparse format expected by the hardware.

### `sparse24_apply.cu`

Kernels that apply an existing sparsity pattern (thread masks) to new data:
- Takes dense input + sparsity mask
- Returns packed sparse data with metadata

### `sparse24_gemm.cu`

Sparse GEMM kernels using CUTLASS:
- `Sparse24TensorCutlass._mm()` uses these kernels
- Sparse (2:4) @ Dense matrix multiplication

### `sparse24_gemm_sm90.cu`

H100 (SM90) optimized sparse GEMM kernels:
- Uses Hopper-specific tensor core instructions
- Higher throughput than SM80 kernels

### `sparse24_largest_mask_2d.cu`

Computes the 2:4 sparsity mask by selecting the 2 largest values in each group of 4:

```cpp
// For each group of 4 elements, keep the 2 with largest absolute values
// This maximizes the information retained in the sparse representation
```

### Supporting Headers

#### `sparse24_metadata.h`

Metadata structures for sparse tensors:
- Block metadata
- Thread-level mask information
- Packed format descriptors

#### `compute_sparse_tile.h`

Tile-level computation for sparse operations:
- Processing 4x4 tiles
- Selecting 2:4 pattern within each tile

#### `warp_tensor.h`

Warp-level tensor operations:
- Register-level matrix operations
- Warp shuffle operations
- Tensor core integration

#### `static_sort.h`

Compile-time sorting for small arrays:
- Used for selecting top-2 values in groups of 4
- Template-based, zero overhead

## PyTorch Stable ABI

### `pt_stable_utils.cu/h`

Utilities for building against PyTorch's stable ABI:
- `xFormersWasNotBuiltException` - When C++ extensions aren't built
- `xFormersInvalidLibException` - When the library is incompatible
- Build metadata extraction from `cpp_lib.json`

Since xFormers 0.0.34, builds target the PyTorch stable ABI, meaning:
- Binary built for PyTorch 2.10 works with any later PyTorch version
- No need to rebuild for each PyTorch release

### `nvcc_info.cu`

Collects NVCC compiler information at build time for diagnostics.

## Build Integration

The C++ extensions are built as part of `setup.py`:

```python
# From setup.py
ext_modules = [
    CppExtension("xformers._C", sources=[
        "xformers/csrc/attention/attention.cpp",
        "xformers/csrc/sparse24/sparse24.cpp",
        ...
    ]),
    CUDAExtension("xformers._C_cuda", sources=[
        "xformers/csrc/sparse24/sparse24_pack.cu",
        "xformers/csrc/sparse24/sparse24_apply.cu",
        "xformers/csrc/sparse24/sparse24_gemm.cu",
        "xformers/csrc/sparse24/sparse24_gemm_sm90.cu",
        ...
    ]),
]
```

## Third-Party Dependencies

### CUTLASS

Bundled in `third_party/cutlass/`:
- Used for sparse GEMM kernels
- Used for attention GEMM kernels
- NVIDIA's CUTLASS library

### Composable Kernel (CK)

Bundled in `third_party/composable_kernel_tiled/`:
- Used for ROCm attention kernels
- AMD's CK library
