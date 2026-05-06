# FlashAttention Operations Library Reference

This document provides comprehensive reference documentation for the optimized operations library in FlashAttention. These operations provide fused CUDA and Triton kernels for common transformer building blocks.

---

## Table of Contents

1. [Overview](#overview)
2. [Fused Dense Operations (fused_dense.py)](#fused-dense-operations)
3. [Layer Norm Operations (layer_norm.py)](#layer-norm-operations)
4. [RMS Norm Operations (rms_norm.py)](#rms-norm-operations)
5. [Activation Functions (activations.py)](#activation-functions)
6. [Tensor Parallel Linear Layers](#tensor-parallel-linear-layers)
7. [Usage Examples](#usage-examples)
8. [Performance Characteristics](#performance-characteristics)

---

## Overview

The operations library provides highly optimized implementations of common transformer operations:

- **Fused Dense**: Fused linear layers with support for bias, residual connections, and tensor parallelism
- **Fused MLP**: Two-layer MLP with fused activation (GELU/ReLU/SqReLU)
- **Fused Layer Norm**: Dropout + residual addition + layer normalization in a single kernel
- **Fused RMS Norm**: Same as above but with RMS normalization
- **Activations**: Custom autograd functions for GELU, SwiGLU, squared ReLU

These operations are designed to:
1. Reduce kernel launch overhead by fusing multiple operations
2. Reduce memory bandwidth by avoiding intermediate tensor materialization
3. Support both fp16 and bf16 training
4. Integrate seamlessly with tensor parallelism

---

## Fused Dense Operations

**File:** `flash_attn/ops/fused_dense.py`

### Classes

#### `FusedDense`

Drop-in replacement for `nn.Linear` with fused CUDA kernels for forward and backward passes.

```python
class FusedDense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        return_residual: bool = False,
        device=None,
        dtype=None,
    )
```

**Parameters:**
- `in_features` (int): Size of each input sample
- `out_features` (int): Size of each output sample
- `bias` (bool): If True, adds a learnable bias (default: True)
- `return_residual` (bool): If True, returns the input alongside the output for residual connection fusion in backward

**Methods:**

##### `forward`

```python
def forward(self, x, process_group=None)
```

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape `(*, in_features)`
- `process_group` (ProcessGroup, optional): For tensor parallelism

**Returns:**
- If `return_residual=False`: Output tensor, shape `(*, out_features)`
- If `return_residual=True`: Tuple of (output, input) where input is saved for backward fusion

**Behavior:**
- When `process_group` is not None and `sequence_parallel=True`, performs all-gather of input before matmul
- Supports automatic mixed precision (AMP) with automatic dtype conversion
- Falls back to `F.linear` for CPU tensors or unsupported dtypes

#### `FusedMLP`

Two-layer MLP (linear -> activation -> linear) with fused CUDA kernels.

```python
class FusedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=True,
        bias2=True,
        activation="gelu_approx",
        return_residual=False,
        checkpoint_lvl=0,
        heuristic="auto",
        device=None,
        dtype=None,
    )
```

**Parameters:**
- `in_features` (int): Input dimension
- `hidden_features` (int, optional): Hidden dimension (default: `4 * in_features`)
- `out_features` (int, optional): Output dimension (default: `in_features`)
- `bias1` (bool): Bias for first linear layer (default: True)
- `bias2` (bool): Bias for second linear layer (default: True)
- `activation` (str): Activation function, one of `"gelu_approx"`, `"relu"`, `"sqrelu"`
- `return_residual` (bool): Return input alongside output
- `checkpoint_lvl` (int): Gradient checkpointing level:
  - `0`: No recomputation (fastest, most memory)
  - `1`: Recompute activation output in backward
  - `2`: Recompute pre-activation and activation output in backward (slowest, least memory)
- `heuristic` (int or str): CuBLAS heuristic for fused GEMM+activation:
  - `-1`: Don't fuse GEMM + activation (separate kernels)
  - `0..4`: Use specific CuBLASLt heuristic
  - `"auto"`: Automatically select based on GPU and CUDA version

**Heuristic Selection (auto mode):**
- For H100 (SM90): Always uses `-1` (unfused) as CuBLASLt fused is slower
- For CUDA >= 11.8: Uses heuristic `0`
- For CUDA < 11.8 with fp16: Uses heuristic `1`
- For CUDA < 11.8 with bf16: Uses `-1` (unfused)

**Methods:**

##### `forward`

```python
def forward(self, x, process_group=None)
```

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `process_group` (ProcessGroup, optional): For tensor parallelism

**Returns:**
- If `return_residual=False`: Output tensor
- If `return_residual=True`: Tuple of (output, input)

**Tensor Parallel Behavior:**
1. All-gather input across ranks (if `sequence_parallel=True`)
2. Forward through fused MLP
3. Reduce-scatter output across ranks

### Low-Level Functions

#### `fused_dense_func`

```python
def fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
)
```

Functional interface for fused dense operation. Falls back to `F.linear` if:
- Tensors are not on CUDA
- Input dtype is not fp16/bf16 (and AMP is not active)

#### `fused_mlp_func`

```python
def fused_mlp_func(
    x: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    bias1: Optional[Tensor] = None,
    bias2: Optional[Tensor] = None,
    activation: str = "gelu_approx",
    save_pre_act: bool = True,
    return_residual: bool = False,
    checkpoint_lvl: int = 0,
    heuristic: int = 0,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
)
```

Functional interface for fused MLP.

**Additional parameters:**
- `save_pre_act` (bool): Save pre-activation values for backward (default: True during training)
- `heuristic` (int): CuBLASLt heuristic for fused GEMM+activation

**Dimension requirements:**
- For `"relu"` activation with `save_pre_act=True`: input dim must be divisible by 128
- For `"gelu_approx"` with `save_pre_act=True`: input dim must be divisible by 8

### Autograd Functions

#### `FusedDenseFunc`

Custom autograd function for fused linear operations. Key optimizations:

**Forward:**
- Kicks off all-gather early (async) before weight dtype conversion
- Uses standard `F.linear` for the matmul

**Backward:**
- Uses `fused_dense_cuda.linear_bias_wgrad` for efficient weight gradient computation
- Async reduce-scatter for gradient synchronization
- Fuses residual gradient addition when `return_residual=True`

#### `FusedMLPFunc`

Custom autograd function for fused MLP operations.

**Forward optimizations:**
- When `heuristic >= 0`: Uses `fused_dense_cuda.linear_act_forward` for fused GEMM + activation
- When `heuristic == -1`: Separate linear + activation with JIT fusion via `torch.jit.fuser("fuser2")`

**Backward optimizations:**
- Uses `fused_dense_cuda.bias_act_linear_dgrad_bgrad` for fused dgrad + bias gradient
- Gradient checkpointing at 3 levels to trade computation for memory

---

## Layer Norm Operations

**File:** `flash_attn/ops/layer_norm.py`

Provides fused dropout + residual addition + layer normalization operations. Based on NVIDIA Apex's FastLayerNorm with significant extensions.

### Module Classes

#### `DropoutAddLayerNorm`

```python
class DropoutAddLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    )
```

**Parameters:**
- `hidden_size` (int): Hidden dimension for layer norm
- `prenorm` (bool): If True, also returns the residual output (before norm) for the next layer
- `p` (float): Dropout probability
- `eps` (float): Epsilon for numerical stability
- `residual_in_fp32` (bool): Cast residual to fp32 before adding (only applies when residual is None initially)

**Methods:**

##### `forward`

```python
def forward(self, x0, residual=None)
```

Computes: `z = LayerNorm(dropout(x0) + residual)`

**Parameters:**
- `x0` (torch.Tensor): Main branch input, shape `(*, hidden_size)`
- `residual` (torch.Tensor, optional): Residual connection, shape `(*, hidden_size)`

**Returns:**
- If `prenorm=False`: Normalized output `z`
- If `prenorm=True`: Tuple `(z, x)` where `x` is the pre-normalization residual

### Functional API

#### `dropout_add_layer_norm`

```python
def dropout_add_layer_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
)
```

**Parameters:**
- `x0` (torch.Tensor): Main input
- `residual` (torch.Tensor, optional): Residual tensor
- `weight` (torch.Tensor): Layer norm weight (gamma)
- `bias` (torch.Tensor): Layer norm bias (beta)
- `dropout_p` (float): Dropout probability
- `epsilon` (float): Layer norm epsilon
- `rowscale` (torch.Tensor, optional): Per-row scaling factor, shape `(batch * seqlen,)`
- `layerscale` (torch.Tensor, optional): Per-column scaling (depthwise), shape `(hidden_size,)`
- `prenorm` (bool): Return pre-norm residual
- `residual_in_fp32` (bool): Force residual in fp32
- `return_dropout_mask` (bool): Also return the dropout mask

#### `layer_norm`

```python
def layer_norm(x, weight, bias, epsilon)
```

Simple layer norm without dropout or residual. Equivalent to `F.layer_norm` but uses the fused kernel.

#### `dropout_add_layer_norm_subset`

```python
def dropout_add_layer_norm_subset(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    layerscale=None,
    x0_subset=None,
    out_subset=None,
    rowscale_const=1.0,
    out_numrows=0,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
)
```

Efficient variant for when only a subset of rows/tokens need to be processed (e.g., last-layer cross-attention for CLS token in ViT).

**Additional parameters:**
- `x0_subset` (torch.Tensor): Indices of rows in x0 to process
- `out_subset` (torch.Tensor): Indices of output rows
- `rowscale_const` (float): Constant scaling factor instead of per-row tensor
- `out_numrows` (int): Number of output rows

#### `dropout_add_layer_norm_parallel_residual`

```python
def dropout_add_layer_norm_parallel_residual(
    x0,
    x1,
    residual,
    weight0,
    bias0,
    weight1,
    bias1,
    dropout_p,
    epsilon,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
)
```

For parallel block architecture (GPT-J style). Applies dropout to both `x0` (attention output) and `x1` (MLP output), adds them to residual, and applies separate layer norms.

### Autograd Functions

#### `DropoutAddLayerNormFn`

Custom autograd function that wraps the `dropout_layer_norm` CUDA extension.

**Forward:**
1. Ensures 16-byte alignment for all tensors
2. Calls `dropout_layer_norm.dropout_add_ln_fwd`
3. Returns normalized output, optionally with pre-norm residual and dropout mask

**Backward:**
1. Calls `dropout_layer_norm.dropout_add_ln_bwd`
2. Computes gradients for input, residual, gamma, beta, and optional colscale

**Memory optimizations:**
- Dropout mask is stored as a bit-mask (1 bit per element instead of 1 byte)
- When `dropout_p == 0.0`, no mask is stored at all
- When `colscale` is not needed, `x0` is not saved for backward

---

## RMS Norm Operations

**File:** `flash_attn/ops/rms_norm.py`

RMSNorm (Root Mean Square Normalization) variants of the LayerNorm operations. RMSNorm computes:

```
output = x * rsqrt(mean(x^2) + eps) * weight
```

Unlike LayerNorm, RMSNorm does not subtract the mean, making it slightly faster.

### Module Classes

#### `RMSNorm`

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None)
```

Simple RMS normalization module.

**Parameters:**
- `hidden_size` (int): Hidden dimension
- `eps` (float): Epsilon for numerical stability

#### `DropoutAddRMSNorm`

```python
class DropoutAddRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    )
```

Fused dropout + residual + RMSNorm. Same interface as `DropoutAddLayerNorm` but uses RMS normalization.

### Functional API

#### `rms_norm`

```python
def rms_norm(x, weight, epsilon)
```

Simple RMS norm without dropout or residual.

#### `dropout_add_rms_norm`

```python
def dropout_add_rms_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
)
```

Same interface as `dropout_add_layer_norm` but uses RMS normalization internally.

#### `dropout_add_rms_norm_subset`

```python
def dropout_add_rms_norm_subset(...)
```

Subset variant for RMS norm. Same parameters as `dropout_add_layer_norm_subset`.

#### `dropout_add_rms_norm_parallel_residual`

```python
def dropout_add_rms_norm_parallel_residual(...)
```

Parallel residual variant for RMS norm.

---

## Activation Functions

**File:** `flash_attn/ops/activations.py`

Custom autograd functions for activation functions used in transformer models.

### GELU (Approximate / Tanh)

#### `bias_gelu_impl`

```python
bias_gelu_impl = GeLUFunction.apply
```

Computes `GELU(y + bias)` where GELU uses the tanh approximation:

```
GELU(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
```

**Parameters:**
- `input` (torch.Tensor): Input tensor
- `bias` (torch.Tensor): Bias tensor broadcasted to input shape

#### `fast_gelu_impl`

```python
fast_gelu_impl = FastGeLUFunction.apply
```

GELU with tanh approximation (without bias). Used in fused MLP backward path.

### Squared ReLU

#### `sqrelu_fwd`

```python
@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)
```

Squared ReLU activation from the Primer paper: `SqReLU(x) = ReLU(x)^2`

#### `sqrelu_bwd`

```python
@torch.jit.script
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x)).to(dtype=x.dtype)
```

Gradient: `d/dx SqReLU(x) = 2 * g * ReLU(x)`

### SwiGLU

#### `swiglu`

```python
swiglu = SwiGLUFunction.apply
```

Computes SwiGLU activation: `SwiGLU(x, y) = x * sigmoid(x) * y`

This is used in gated MLP architectures (e.g., LLaMA):

```python
# In a gated MLP:
gate = F.linear(x, w_gate)  # -> x in SwiGLU
up = F.linear(x, w_up)      # -> y in SwiGLU
hidden = swiglu(gate, up)    # = gate * sigmoid(gate) * up
output = F.linear(hidden, w_down)
```

**Backward gradients:**
- `dx = sigmoid(x) * (1 + x * (1 - sigmoid(x))) * g * y`
- `dy = x * sigmoid(x) * g`

### ReLU Backward

#### `relu_bwd`

```python
@torch.jit.script
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)
```

Efficient JIT-compiled ReLU backward.

### Reference: Activation Mapping

| Config Name | Python Function | CUDA Fused | Triton Kernel | Notes |
|------------|-----------------|-----------|---------------|-------|
| `"gelu"` | `F.gelu` | No | `gelu` | Exact GELU via erf |
| `"gelu_new"` / `"gelu_fast"` / `"gelu_approx"` / `"gelu_pytorch_tanh"` | `F.gelu(approximate="tanh")` | Yes (fused_mlp) | `gelu_approx` | Tanh approximation |
| `"relu"` | `F.relu` | Yes (fused_mlp) | `relu` | Standard ReLU |
| `"sqrelu"` | `sqrelu_fwd` | No | `squared_relu` | ReLU^2 from Primer |
| `"glu"` | `F.sigmoid` (gating) | No | - | Sigmoid-gated |
| `"swiglu"` | `swiglu` (x * sigmoid(x) * y) | No | - | SiLU-gated (LLaMA) |
| `"geglu"` | `F.gelu` (gating) | No | - | GELU-gated |

---

## Tensor Parallel Linear Layers

### `ColumnParallelLinear`

```python
class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    )
```

Splits the output features across tensor parallel ranks. Each rank computes a shard of the output.

**Behavior:**
- `out_features` is divided across `world_size` ranks
- Handles uneven division: first `mod` ranks get `div + 1` copies, rest get `div`
- When `sequence_parallel=True`: all-gathers input before matmul

**Use cases:**
- First linear layer in MLP (fc1): splits hidden features
- QKV projection: splits attention heads
- LM head: splits vocabulary

### `RowParallelLinear`

```python
class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    )
```

Splits the input features across tensor parallel ranks. Each rank computes with its shard of input, then results are reduced across ranks.

**Behavior:**
- `in_features` is divided across `world_size` ranks
- Only rank 0 has bias (to avoid double-counting)
- After matmul: reduce-scatter (if `sequence_parallel=True`) or all-reduce (if not)

**Use cases:**
- Second linear layer in MLP (fc2): reduces scattered hidden features
- Attention output projection: reduces scattered head outputs

### Parallel Communication Patterns

**Column Parallel + Sequence Parallel:**
```
Input [B, S, D] -> AllGather -> [B*W, S, D] -> MatMul -> [B*W, S, D/W] -> output
```

**Row Parallel + Sequence Parallel:**
```
Input [B*W, S, D/W] -> MatMul -> [B*W, S, D] -> ReduceScatter -> [B, S, D]
```

Where `B` = batch, `S` = sequence, `D` = hidden dim, `W` = world size.

---

## Usage Examples

### Basic Fused Dense

```python
import torch
from flash_attn.ops.fused_dense import FusedDense

# Drop-in replacement for nn.Linear
linear = FusedDense(768, 3072, bias=True).cuda()
x = torch.randn(2, 512, 768, device="cuda", dtype=torch.float16)
output = linear(x)  # Shape: (2, 512, 3072)
```

### Fused Dense with Residual Return

```python
# For fusing residual connection in backward
linear = FusedDense(768, 3072, return_residual=True).cuda()
output, x_saved = linear(x)
# x_saved is used in backward to fuse residual addition
```

### Fused MLP

```python
from flash_attn.ops.fused_dense import FusedMLP

mlp = FusedMLP(
    in_features=768,
    hidden_features=3072,
    activation="gelu_approx",
    checkpoint_lvl=1,  # Recompute activation in backward
).cuda().half()

x = torch.randn(2, 512, 768, device="cuda", dtype=torch.float16)
output = mlp(x)  # Shape: (2, 512, 768)
```

### Fused Layer Norm

```python
from flash_attn.ops.layer_norm import dropout_add_layer_norm

# Pre-norm architecture
x0 = torch.randn(2, 512, 768, device="cuda", dtype=torch.float16)
residual = torch.randn(2, 512, 768, device="cuda", dtype=torch.float16)
weight = torch.ones(768, device="cuda", dtype=torch.float16)
bias = torch.zeros(768, device="cuda", dtype=torch.float16)

# Computes: z = LayerNorm(dropout(x0) + residual)
# Also returns pre-norm residual for next block
z, x_residual = dropout_add_layer_norm(
    x0, residual, weight, bias,
    dropout_p=0.1,
    epsilon=1e-5,
    prenorm=True,
)
```

### DropoutAddLayerNorm Module

```python
from flash_attn.ops.layer_norm import DropoutAddLayerNorm

norm = DropoutAddLayerNorm(
    hidden_size=768,
    prenorm=True,
    p=0.1,
    eps=1e-5,
    residual_in_fp32=True,
).cuda()

x0 = torch.randn(2, 512, 768, device="cuda", dtype=torch.float16)
residual = torch.randn(2, 512, 768, device="cuda", dtype=torch.float32)
z, x_residual = norm(x0, residual)
```

### RMS Norm (for LLaMA-style models)

```python
from flash_attn.ops.rms_norm import DropoutAddRMSNorm

norm = DropoutAddRMSNorm(
    hidden_size=4096,
    prenorm=True,
    p=0.0,  # No dropout
    eps=1e-5,
).cuda()

x0 = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
z, x_residual = norm(x0)
```

### Tensor Parallel Setup

```python
import torch.distributed as dist
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear

process_group = dist.new_group(backend="nccl")

# Column parallel: split output dimension
fc1 = ColumnParallelLinear(
    in_features=768,
    out_features=3072,
    process_group=process_group,
    sequence_parallel=True,
).cuda()

# Row parallel: split input dimension
fc2 = RowParallelLinear(
    in_features=3072,
    out_features=768,
    process_group=process_group,
    sequence_parallel=True,
).cuda()

# Forward pass with tensor parallel
x = torch.randn(2, 512, 768, device="cuda", dtype=torch.float16)
hidden = fc1(x)   # All-gather + matmul
output = fc2(hidden)  # Matmul + reduce-scatter
```

### SwiGLU Activation (LLaMA-style MLP)

```python
from flash_attn.ops.activations import swiglu

batch, seqlen, hidden = 2, 512, 4096
intermediate = 11008

# Gated MLP with SwiGLU
x = torch.randn(batch, seqlen, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
w_gate = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
w_up = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
w_down = torch.randn(hidden, intermediate, device="cuda", dtype=torch.float16, requires_grad=True)

gate = F.linear(x, w_gate)
up = F.linear(x, w_up)
hidden_state = swiglu(gate, up)
output = F.linear(hidden_state, w_down)
```

---

## Performance Characteristics

### Fused Dense

- **Kernel launch reduction**: 1 kernel instead of separate bias add + matmul kernels
- **Memory savings**: When `return_residual=True`, avoids allocating extra tensor for residual in backward
- **Async communication**: All-gather is overlapped with weight dtype conversion in tensor parallel mode

### Fused MLP

- **Fused GEMM + activation**: Single kernel for `matmul + bias + GELU/ReLU` (when `heuristic >= 0`)
- **Checkpoint levels**: Trade 10-20% throughput for 40-50% memory savings
- **Dimension requirements**: Optimal when dimensions are multiples of 128 for ReLU, 8 for GELU

### Fused Layer Norm

- **Memory alignment**: All tensors are aligned to 16 bytes for optimal memory access
- **Bit-packed dropout mask**: 8x less memory for dropout mask compared to boolean mask
- **Supported dimensions**: Up to 6144 (limited by CUDA shared memory)

### Fused RMS Norm

- **Faster than LayerNorm**: No mean subtraction, slightly fewer operations
- **Better for large models**: Used in LLaMA, Falcon, and other modern architectures
- **Same fusion benefits**: Dropout + residual + normalization in single kernel

### Benchmark Tips

1. For best performance, use CUDA >= 11.8 (better CuBLASLt support for bf16)
2. On H100, the unfused path (`heuristic=-1`) may be faster for MLP due to optimized matmul
3. `checkpoint_lvl=1` provides the best trade-off for most cases
4. Ensure hidden dimensions are aligned to multiples of 8 for GELU, 128 for ReLU
5. When using tensor parallelism, `sequence_parallel=True` is generally recommended for better throughput
