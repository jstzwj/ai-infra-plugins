# Chapter 24: triton_kernels Library

`triton_kernels` is a library of high-performance GPU kernels built with Triton.

## Core Modules

### matmul

High-performance matrix multiplication with advanced features.

```python
from triton_kernels import matmul

result = matmul(
    a, b,                        # Input tensors
    a_scale=None, b_scale=None,  # Optional FP8 scales
    acc_dtype=None,              # Accumulator dtype
    out_dtype=None,              # Output dtype
    fused_activation=None,       # Optional fused activation
    epilogue=None,               # Post-processing
    precision_config=None,       # Precision settings
)
```

**Features:**
- Multiple precisions: FP16, BF16, FP8, MXFP4, MXFP8, NVFP4
- Ragged (batched) matrix multiplication
- Fused activations (SwiGLU, ReLU, etc.)
- Split-K optimization
- TMA (Tensor Memory Accelerator) support
- Expert parallel for MoE models
- HBM swizzling optimizations

### PrecisionConfig

```python
from triton_kernels import PrecisionConfig

config = PrecisionConfig(
    input_precision="ieee",      # ieee, tf32, tf32x3
    mx_fp8_format="e4m3",       # e4m3 or e5m2
    output_dtype=tl.float16,
)
```

### FusedActivation

```python
from triton_kernels import FusedActivation

# SwiGLU fused into matmul
activation = FusedActivation(
    name="swiglu",
    args={"routing_data": routing},  # Optional routing info
)
result = matmul(a, b, fused_activation=activation)
```

### reduce

Reduction operations with advanced features.

```python
from triton_kernels import reduce

result = reduce(
    input,                       # Input tensor
    axis=0,                      # Reduction axis
    op="sum",                    # Operation: sum, max, min
    mask=None,                   # Optional mask
    postprocess_fn=None,         # Optional post-processing
    opt_flags=None,              # Optimization flags
)
```

**Features:**
- Masked reductions
- MX quantization support
- Custom post-processing functions
- Multi-dimensional reductions

### swiglu

SwiGLU activation function.

```python
from triton_kernels import swiglu

result = swiglu(
    x, w1, w2,                  # Input and weight matrices
    routing_data=None,          # Optional expert routing
    precision_config=None,      # Precision settings
)
```

### topk

Top-K element selection.

```python
from triton_kernels import topk

values, indices = topk(input, k=5, dim=-1, descending=True)
```

### compaction

Sparse tensor compaction based on bitmask.

```python
from triton_kernels import compaction

result = compaction(
    input_tensor,
    bitmask,                    # Selection mask
)
```

### distributed

Multi-GPU distributed computing for MoE models.

```python
from triton_kernels import distributed

# Create expert assignments
assignment = distributed.make_expt_assignment(
    num_experts=8,
    num_gpus=4,
    tokens_per_expert=[100, 200, 150, 80, 120, 300, 90, 160],
)

# Convert between data parallel and expert parallel
distributed.convert_dp_to_ep(data, assignment)
distributed.convert_ep_to_dp(data, assignment)
```

## Tensor Types

### Storage

```python
from triton_kernels import Storage

storage = Storage(
    data=torch_tensor,
    layout=layout,              # Optional layout info
)
```

### RaggedTensor

```python
from triton_kernels import RaggedTensor

ragged = RaggedTensor(
    data=flat_tensor,
    metadata=metadata,          # Shape and stride info per group
)
```

### SparseMatrix

```python
from triton_kernels import SparseMatrix

sparse = SparseMatrix(
    data=values,
    bitmatrix=mask,             # Binary sparsity mask
)
```

## Numerics

### BaseFlexData / InFlexData / OutFlexData

Flexible precision data with scaling:

```python
from triton_kernels import InFlexData, OutFlexData

# Input with MX scale
input_data = InFlexData(tensor, scale=mx_scale)

# Output with flexible precision
output_data = OutFlexData(tensor, scale=output_scale)
```

## Target Info

```python
from triton_kernels import target_info

# Check hardware capabilities
target_info.current_target()
```
