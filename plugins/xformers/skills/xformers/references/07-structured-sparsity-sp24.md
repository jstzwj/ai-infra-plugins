# 07 - 2:4 Structured Sparsity (SP24)

## Overview

xFormers provides comprehensive support for NVIDIA's 2:4 structured sparsity pattern, where in every group of 4 values, exactly 2 are zeros. This is hardware-accelerated on Ampere+ GPUs (A100, H100) and provides ~2x throughput improvement for sparse matrix multiplication.

**Source**: `xformers/ops/sp24.py`
**CUDA kernels**: `xformers/csrc/sparse24/`

## 2:4 Sparsity Pattern

In 2:4 structured sparsity, for every consecutive group of 4 elements in a row, exactly 2 elements must be zero:

```
Original:  [a, b, c, d, e, f, g, h]
Sparse:    [a, 0, c, 0, 0, f, g, 0]  # 2 non-zero per group of 4
Mask:      [1, 0, 1, 0, 0, 1, 1, 0]  # 50% sparsity
```

The hardware automatically selects which 2 elements to keep to minimize information loss.

## API Reference

### Core Types

```python
# Gradient modes
GRADIENT_SP24 = "24sparse"   # Gradient is also 2:4 sparse
GRADIENT_DENSE = "24dense"   # Gradient is dense
GRADIENT_STE = "ste"         # Straight-Through Estimator

# Backends
BACKEND_CUTLASS = "cutlass"       # CUTLASS sparse GEMM
BACKEND_CUSPARSELT = "cusparselt" # cuSPARSELt library
BACKEND_DENSE = "dense"           # Dense output (not sparse)
```

### `sparsify24`

```python
from xformers.ops import sparsify24

sparse_tensor = sparsify24(
    x: torch.Tensor,              # Dense input tensor [M, K]
    algo: str = "",               # Sparsification algorithm ("" = auto)
    gradient: str = GRADIENT_SP24,# Gradient mode
    backend: str = BACKEND_CUTLASS, # Backend to use
) -> Sparse24Tensor               # Sparse tensor subclass
```

Converts a dense tensor to 2:4 sparse format. The result is a `Sparse24Tensor` subclass that supports:
- Matrix multiplication with dense tensors
- Element-wise operations (relu, gelu, silu, add, mul)
- Transpose
- torch.compile compatibility

**Gradient modes:**
- `GRADIENT_SP24`: The gradient is also sparsified (same pattern). Most memory-efficient.
- `GRADIENT_DENSE`: The gradient is dense. More accurate but uses more memory.
- `GRADIENT_STE`: Straight-Through Estimator - gradient passes through unmodified.

### `sparsify24_ste`

```python
from xformers.ops import sparsify24_ste

sparse_tensor = sparsify24_ste(
    x: torch.Tensor,
    algo: str = "",
    backend: str = BACKEND_CUTLASS,
    bw_mul0: float = 1.0,  # Gradient multiplier for pruned values
    bw_mul1: float = 1.0,  # Gradient multiplier for kept values
) -> Sparse24Tensor
```

2:4 sparsification with Straight-Through Estimator. The forward pass is sparse, but the backward pass gradient is not sparsified. The `bw_mul0` and `bw_mul1` parameters allow rescaling gradients differently for pruned vs kept values.

**Use case:** Training with learned sparsity patterns where you want gradient flow through all values.

### `sparsify24_like`

```python
from xformers.ops import sparsify24_like

sparse_tensor = sparsify24_like(
    x: torch.Tensor,           # Dense tensor to sparsify
    pattern: Sparse24Tensor,   # Reference sparse tensor (provides sparsity pattern)
    gradient: str = GRADIENT_SP24,
    backend: str = "",         # Auto-detect from pattern
    out_dense: Optional[bool] = None,  # Deprecated
) -> Sparse24Tensor
```

Applies the sparsity pattern from one tensor to another. Essential for training where you want consistent sparsity patterns.

## Sparse24Tensor Classes

### `Sparse24Tensor`

Base tensor subclass for 2:4 sparse tensors. Cannot be instantiated directly - use `sparsify24` or one of its subclasses.

**Attributes:**
- `packed` - Packed sparse data (non-zero values)
- `meta` - Metadata describing sparsity pattern
- `packed_t` - Transposed packed data
- `meta_t` - Transposed metadata
- `threads_masks` - Thread-level sparsity masks

### `Sparse24TensorCutlass`

CUTLASS backend sparse tensor. Supports:
- `mm` (matrix multiplication with dense tensor)
- `relu`, `gelu`, `silu` (element-wise activations)
- `add`, `mul` (element-wise arithmetic)
- `gelu_backward`, `silu_backward`, `threshold_backward`
- `t` (transpose)
- `detach`

**Limitations:**
- No matmul with bias (use `Sparse24TensorCuSparseLt` for that)
- No `Sparse24Tensor @ Sparse24Tensor`

### `Sparse24TensorCuSparseLt`

cuSPARSELt backend sparse tensor. Supports:
- `mm` (with optional bias)
- `addmm` (fused bias + matmul)
- `linear` (linear layer)
- `t` (transpose)
- `detach`

**Requirements:**
- cuSPARSELt >= 0.5.0
- For SM90 (H100): cuSPARSELt >= 6.0.0

## Supported Operations

### CUTLASS Dispatch Table

| Operation | Support |
|-----------|---------|
| `aten.mm` / `aten.matmul` | Yes (sparse @ dense) |
| `aten.relu` | Yes |
| `aten.gelu` | Yes |
| `aten.silu` | Yes |
| `aten.mul` | Yes (with auto-sparsify) |
| `aten.add` | Yes |
| `aten.gelu_backward` | Yes (with auto-sparsify) |
| `aten.silu_backward` | Yes (with auto-sparsify) |
| `aten.threshold_backward` (relu BW) | Yes (with auto-sparsify) |
| `aten.t` | Yes |
| `aten.linear` | Yes |
| `aten.detach` | Yes |

### cuSPARSELt Dispatch Table

| Operation | Support |
|-----------|---------|
| `aten.mm` / `aten.matmul` | Yes |
| `aten.addmm` | Yes (with bias) |
| `aten.linear` | Yes |
| `aten.t` | Yes |
| `aten.detach` | Yes |

## Usage Examples

### Inference with Sparse Weights

```python
import torch
from xformers.ops import sparsify24

# Sparsify weight matrix
weight = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
sparse_weight = sparsify24(weight, backend="cutlass")

# Matrix multiplication is automatically sparse
x = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
output = x @ sparse_weight.t()  # Uses sparse GEMM
```

### Training with Dynamic Sparsification

```python
import torch
import torch.nn as nn
from xformers.ops import sparsify24

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Sparsify weight on every forward pass
        w_sparse = sparsify24(self.weight, gradient="24dense")
        return x @ w_sparse.t()
```

### STE Training (Gradient Flows Through All Values)

```python
from xformers.ops import sparsify24_ste

class STELinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Forward: sparse; Backward: dense gradient
        w_sparse = sparsify24_ste(
            self.weight,
            bw_mul0=0.5,  # Scale gradient for pruned values
            bw_mul1=1.0,  # Keep gradient for kept values
        )
        return x @ w_sparse.t()
```

### Applying Same Pattern to Multiple Tensors

```python
from xformers.ops import sparsify24, sparsify24_like

# Create reference pattern
w1 = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
w1_sparse = sparsify24(w1)

# Apply same pattern to another weight
w2 = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
w2_sparse = sparsify24_like(w2, w1_sparse)
```

### With cuSPARSELt Backend (Supports Bias)

```python
sparse_weight = sparsify24(weight, backend="cusparselt")
bias = torch.randn(4096, device="cuda", dtype=torch.float16)
output = torch.nn.functional.linear(x, sparse_weight, bias)
```

## Algorithm Tuning

cuSPARSELt has multiple algorithms for different GEMM dimensions. xFormers can auto-tune:

```bash
# Enable algorithm search (disabled by default due to correctness issues)
XFORMERS_CUSPARSELT_TUNE=1 python train.py
```

The tuning benchmarks all available algorithms and caches the best one per (M, N, K, format, dtype, has_bias) combination.

## Registered Operators

| Operator | Backend | Description |
|----------|---------|-------------|
| `SparsifyBothWays` | C++/CUDA | Creates sparse format in both orientations |
| `SparsifyApply` | C++/CUDA | Applies existing sparsity pattern to new data |
| `SparsifyApplyDenseOutput` | C++/CUDA | Applies pattern with dense output |
| `Sp24Gemm` | CUTLASS | Sparse-dense matrix multiplication |
| `Sp24GemmCusplt` | cuSPARSELt | Sparse-dense GEMM via cuSPARSELt |
| `Sp24GemmCuspltSearch` | cuSPARSELt | Algorithm search for best GEMM kernel |

## torch.compile Support

Both `Sparse24TensorCutlass` and `Sparse24TensorCuSparseLt` are registered with `torch._dynamo.allow_in_graph` for `torch.compile` compatibility.

## Limitations

1. **Only 2D tensors** for matmul (no batched matmul)
2. **No sparse @ sparse** multiplication
3. **cuSPARSELt**: Dense matrix N dimension must be aligned to 8
4. **Same dtype required**: Sparse and dense tensors must have the same dtype
5. **Compute capability**: Ampere+ (A100) for CUTLASS, varies for cuSPARSELt
