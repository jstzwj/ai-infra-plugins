# 05 - RMSNorm

## Overview

RMSNorm (Root Mean Square Layer Normalization) is a simplified alternative to LayerNorm that normalizes by the root mean square of the inputs. It is used in LLaMA, PaLM, and many modern Transformer models.

**Source**: `xformers/ops/rmsnorm.py`
**Kernel**: `xformers/ops/_triton/rmsnorm_kernels.py`

## Mathematical Definition

For each dim-length vector x:

```
RMSNorm(x) = x / sqrt(x^2.sum() + eps) * weight
```

This is similar to `torch.nn.functional.normalize` but with `eps` added (not max).

## API Reference

### `rms_norm`

```python
xformers.ops.rms_norm(
    x: torch.Tensor,                  # Input [..., dim], must be contiguous
    weight: Optional[torch.Tensor],    # Learnable scale [dim]
    eps: float = 1e-6,                 # Small constant for numerical stability
) -> torch.Tensor                      # Output [..., dim]
```

Computes RMS normalization along the last dimension.

**Constraints:**
- Requires Triton (GPU with compute capability >= 8.0)
- Input must be contiguous
- Does NOT support gradients (inference only or use `torch.nn.LayerNorm` for training)

**Raises:**
- `ValueError` if gradients are enabled and any input requires grad
- `AssertionError` if Triton is not available

### `rms_norm_add`

```python
xformers.ops.rms_norm_add(
    x: torch.Tensor,                  # Input [..., dim], MODIFIED IN PLACE
    y: torch.Tensor,                  # Addend [..., dim]
    weight: Optional[torch.Tensor],    # Learnable scale [dim]
    eps: float = 1e-6,
) -> torch.Tensor                      # Output [..., dim]
```

Fused addition + RMS normalization. Equivalent to:

```python
x += y
return rms_norm(x, weight, eps)
```

This is useful for residual connections where you want to fuse the addition with normalization.

**Constraints:**
- Same as `rms_norm` (no gradient support, requires Triton)
- `x` is modified in place before normalization

### `RMSNorm` (nn.Module)

```python
from xformers.ops import RMSNorm

layer = RMSNorm(
    dim=4096,              # Normalization dimension
    include_weight=True,   # Whether to include learnable weight
    eps=1e-6,              # Epsilon for numerical stability
)

output = layer(x)  # x: [..., dim] -> [..., dim]
```

**Module attributes:**
- `weight` - Learnable scale parameter `[dim]` (optional, `None` if `include_weight=False`)
- `eps` - Epsilon value

**Methods:**
- `forward(x)` - Standard forward pass
- `increment_and_forward_(x, y)` - Fused `x += y; return self(x)`

## Usage Examples

### Basic Usage

```python
import torch
from xformers.ops import rms_norm

x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)

with torch.no_grad():
    output = rms_norm(x, weight, eps=1e-6)
```

### As nn.Module

```python
from xformers.ops import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = Attention(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Fused Residual + Norm

```python
class EfficientTransformerBlock(nn.Module):
    def __init__(self, dim):
        self.norm = RMSNorm(dim)
        self.attn = Attention(dim)

    def forward(self, x, attn_out):
        # Fused: x += attn_out; norm(x)
        return self.norm.increment_and_forward_(x, attn_out)
```

## Comparison with LayerNorm

| Feature | RMSNorm | LayerNorm |
|---------|---------|-----------|
| Mean subtraction | No | Yes |
| Bias | No | Yes |
| Computation | `x / RMS(x)` | `(x - mean) / std` |
| Parameters | weight only | weight + bias |
| Speed | Faster (simpler) | Slower |
| Training support | No (in this impl) | Yes |

## Implementation Details

The RMSNorm implementation uses a Triton kernel that:

1. Computes the sum of squares along the last dimension
2. Computes `1 / sqrt(sum_sq / dim + eps)`
3. Multiplies each element by this factor and the weight

The fused `rms_norm_add` additionally:
1. Adds `y` to `x` first (in place)
2. Then performs RMSNorm on the result

## Performance

- On A100/fp16: ~2-3x faster than equivalent PyTorch implementation
- Avoids materializing intermediate tensors
- The Triton kernel is optimized for the common case of large hidden dimensions

## Limitations

1. **No gradient support**: This implementation is for inference only. For training, use `torch.nn.LayerNorm` or implement a differentiable version.

2. **Requires Triton**: Only works on GPUs with compute capability >= 8.0 (A100+)

3. **Last dimension only**: Normalizes only along the last dimension

4. **Contiguous input**: Requires contiguous input tensors
