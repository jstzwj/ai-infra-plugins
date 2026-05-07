# 04 - SwiGLU Operation

## Overview

The SwiGLU (SiLU-Gated Linear Unit) is a fused activation function commonly used in modern Transformer models (LLaMA, PaLM, etc.). It computes `SiLU(x @ w1) * (x @ w2) @ w3` with fused linear layers for maximum performance.

**Source**: `xformers/ops/swiglu_op.py`

## Mathematical Definition

```
SwiGLU(x, w1, b1, w2, b2, w3, b3):
    x1 = x @ w1^T + b1        # gate projection
    x2 = x @ w2^T + b2        # up projection
    hidden = SiLU(x1) * x2    # gated activation
    return hidden @ w3^T + b3  # down projection
```

## API Reference

### `swiglu`

```python
xformers.ops.swiglu(
    x: torch.Tensor,       # Input tensor [..., in_features]
    w1: torch.Tensor,      # Gate weight [hidden_features, in_features]
    b1: Optional[torch.Tensor],  # Gate bias [hidden_features]
    w2: torch.Tensor,      # Up weight [hidden_features, in_features]
    b2: Optional[torch.Tensor],  # Up bias [hidden_features]
    w3: torch.Tensor,      # Down weight [out_features, hidden_features]
    b3: Optional[torch.Tensor],  # Down bias [out_features]
    *,
    op: Optional[SwiGLUOp] = None,  # Force specific operator
) -> torch.Tensor          # Output [..., out_features]
```

Computes the SwiGLU block. Equivalent to:

```python
x1 = F.linear(x, w1, b1)
x2 = F.linear(x, w2, b2)
hidden = F.silu(x1) * x2
return F.linear(hidden, w3, b3)
```

### `swiglu_packed`

```python
xformers.ops.swiglu_packed(
    x: torch.Tensor,       # Input [..., in_features]
    w1w2: torch.Tensor,    # Packed gate+up weights [2, hidden_features, in_features]
    b1b2: Optional[torch.Tensor],  # Packed biases [2, hidden_features]
    w3: torch.Tensor,      # Down weight [out_features, hidden_features]
    b3: Optional[torch.Tensor],  # Down bias [out_features]
    *,
    op: SwiGLUOp,          # Required, must support PACKED_WEIGHTS
) -> torch.Tensor
```

Computes SwiGLU with packed w1/w2 weights for better performance.

### `SwiGLU` (nn.Module)

```python
from xformers.ops import SwiGLU

layer = SwiGLU(
    in_features=4096,        # Input dimension
    hidden_features=11008,   # Hidden dimension (typically ~4x input)
    out_features=None,       # Output dimension (defaults to in_features)
    bias=True,               # Whether to include biases
    _pack_weights=True,      # Pack w1/w2 for performance
)

output = layer(x)  # x: [..., in_features] -> [..., out_features]
```

The module creates:
- `self.w12` = `nn.Linear(in_features, 2 * hidden_features, bias=bias)` (packed gate+up)
- `self.w3` = `nn.Linear(hidden_features, out_features, bias=bias)` (down projection)

**Attributes:**
- `in_features` - Input feature count
- `hidden_features` - Hidden feature count
- `out_features` - Output feature count
- `op` - Can be set to force a specific operator

## Operator Classes

### `SwiGLUOp`

Base class for SwiGLU operators.

```python
class SwiGLUOp:
    NAME: str               # Operator name
    PACKED_WEIGHTS: bool    # Whether it supports packed w1/w2
```

### `SwiGLUEagerOp`

Default eager implementation. Uses standard PyTorch operations:

```python
x1 = F.linear(x, w1, b1)
x2 = F.linear(x, w2, b2)
hidden = F.silu(x1) * x2
return F.linear(hidden, w3, b3)
```

### `_SwiGLUDecomposedOp`

Decomposed implementation showing all operations explicitly. Useful for understanding the computation but slower than eager.

**Characteristics:**
- `FORCE_BW_F32 = False` - Can force float32 backward pass
- Shows timing breakdown per operation

### `SwiGLUOpDispatch`

Automatic operator selection:

```python
dispatch = SwiGLUOpDispatch.from_arguments(x, w1, b1, w2, b2, w3, b3)
op = dispatch.op  # Best available operator
```

**Dispatch considers:**
- `device` - CPU vs CUDA
- `dtype` - fp16, bf16, fp32
- `dtype_autocast_gpu` - autocast dtype
- `packed_weights` - Whether w1/w2 share storage
- `bias_enabled` - Whether biases are provided

## Weight Packing

For best performance, pack w1 and w2 into a single tensor:

```python
import xformers.ops as xops

# Create packed weights
w12 = torch.randn(2 * hidden_features, in_features, device="cuda", dtype=torch.float16)
w1, w2 = xops.unbind(w12.view(2, hidden_features, in_features), dim=0)

# Use with swiglu - w1/w2 come from same storage
result = xops.swiglu(x, w1, b1, w2, b2, w3, b3)
```

Or use the `SwiGLU` module which handles packing automatically:

```python
layer = SwiGLU(4096, 11008, _pack_weights=True)
# Access packed weight: layer.w12.weight
# Access individual: layer._ordered_params()
```

## Performance Notes

1. **A100+ Optimization**: The optimized SwiGLU kernel was removed in 0.0.34. Current implementation uses eager PyTorch operations.

2. **Weight Packing**: Using packed w1/w2 (via `stack_or_none`) enables potential kernel fusion.

3. **Autocast Support**: Works with `torch.autocast` for mixed-precision training.

4. **Memory**: The SwiGLU operation requires storing intermediate activations (x1, x2, x3, x4) for the backward pass.

## Usage in LLaMA

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = xops.SwiGLU(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=False,
        )

    def forward(self, x):
        return self.swiglu(x)
```

## Backward Pass

The backward pass computes gradients for all weights and the input:

```
dx5 = grad_output
dx4 = dx5 @ w3
dw3 = dx5^T @ x4
db3 = dx5.sum(0)
dx3 = dx4 * x2
dx2 = dx4 * x3
dx1 = silu_backward(dx3, x1)
dx = dx2 @ w2 + dx1 @ w1
dw1 = dx1^T @ x
dw2 = dx2^T @ x
```

The decomposed implementation provides timing breakdown for each backward operation.

## Shape Validation

The `swiglu` function validates:
- `w1` and `w2` must have the same shape: `[hidden_features, in_features]`
- `w3` must have shape: `[out_features, hidden_features]`
- `b1`, `b2` must have shape: `[hidden_features]`
- `b3` must have shape: `[out_features]`
- Input last dimension must equal `in_features`
