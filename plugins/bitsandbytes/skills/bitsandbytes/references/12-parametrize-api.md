# Parametrize API Reference

This document provides a comprehensive reference for the bitsandbytes parametrize API, which enables 4-bit quantization of parameters in arbitrary `nn.Module` instances (not just `nn.Linear`) using PyTorch's `torch.nn.utils.parametrize` system. This is particularly useful for Mixture-of-Experts (MoE) models and other architectures where parameters live in non-standard module types.

---

## Table of Contents

1. [Overview](#overview)
2. [Bnb4bitParametrization](#bnb4bitparametrization)
3. [replace_parameter_4bit](#replace_parameter_4bit)
4. [replace_parameter_4bit_prequantized](#replace_parameter_4bit_prequantized)
5. [Hook Registration](#hook-registration)
6. [State Dict Serialization](#state-dict-serialization)
7. [Use Cases](#use-cases)

---

## Overview

The parametrize API provides an alternative to `Params4bit`-based quantization. Instead of replacing the parameter with a custom `torch.nn.Parameter` subclass, it uses PyTorch's built-in parametrization mechanism (`torch.nn.utils.parametrize`) to wrap an existing parameter with a dequantization layer.

### Architecture

```
Original Module
    weight: nn.Parameter (float16/bfloat16)
        |
        v  [replace_parameter_4bit()]
        |
Module with Parametrization
    parametrizations.weight:
        original: nn.Parameter (quantized uint8, requires_grad=False)
        0: Bnb4bitParametrization (holds QuantState)
    |
    v  [forward() access]
    |
    weight (dequantized float16/bfloat16)
```

When the parameter is accessed during the forward pass, the parametrization's `forward()` method dequantizes the stored quantized data on-the-fly, producing the original floating-point tensor.

### Comparison with Params4bit

| Aspect | Params4bit (Linear4bit) | Parametrize API |
|--------|------------------------|-----------------|
| Module types | `nn.Linear`, `nn.Embedding` only | Any `nn.Module` |
| Parameter type | Custom `Params4bit` subclass | Standard `nn.Parameter` with parametrization wrapper |
| Access pattern | Direct `.weight` access triggers custom `__torch_function__` | Parametrization `forward()` called on access |
| FSDP compatibility | Via `@property` descriptors | Via parametrization + hooks |
| State dict | Custom `_save_to_state_dict` | `register_state_dict_post_hook` |
| MoE support | Limited (expert weights need special handling) | Full support |
| `torch.compile` | Supported via `_ops.py` custom ops | Supported |

---

## Bnb4bitParametrization

`Bnb4bitParametrization` is an `nn.Module` that wraps a `QuantState` and performs dequantization when the parameter is accessed.

**Location:** `bitsandbytes/nn/parametrize.py`

### Class Definition

```python
class Bnb4bitParametrization(nn.Module):
    """
    A parametrization module that handles dequantization of a 4-bit quantized parameter.

    The parameter data is expected to be already quantized when this parametrization
    is applied. This module will dequantize the parameter data to its original
    floating-point representation when the forward method is called (i.e. when
    the parameter is accessed).

    Args:
        quant_state (F.QuantState):
            The quantization state containing the necessary information for dequantization.
    """

    def __init__(self, quant_state: F.QuantState):
        super().__init__()
        self.quant_state = quant_state

    @torch.no_grad()
    def forward(self, quantized_param: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to dequantize the parameter.

        Args:
            quantized_param (torch.Tensor): The quantized parameter tensor (from .original)

        Returns:
            torch.Tensor: The dequantized parameter tensor in the original shape and dtype.
        """
        return F.dequantize_4bit(quantized_param, self.quant_state)
```

### Key Design Decisions

1. **`@torch.no_grad()` decorator**: The `forward()` method is decorated with `@torch.no_grad()` because dequantization is a pure computation with no learnable parameters. Gradients should not flow through the dequantization step -- they flow through the downstream computation instead.

2. **`quant_state` stored as module attribute**: The `QuantState` is stored as a regular attribute (not a parameter or buffer), so it does not appear in the module's `parameters()` or `buffers()` iterators. This is intentional -- the quantization state is metadata, not a trainable parameter.

3. **Dequantization on every access**: Each time the parameter is accessed (e.g., `module.weight` in the forward pass), the full dequantization is performed. The caching hooks (described below) mitigate the performance impact.

### How Parametrization Works

When `P.register_parametrization(module, "weight", parametrization)` is called, PyTorch:

1. Replaces `module.weight` with a `ParametrizedParameter` descriptor
2. Stores the original parameter as `module.parametrizations.weight.original`
3. When `module.weight` is accessed, it calls `parametrization.forward(original)` and returns the result
4. The original parameter's `requires_grad` is set to `False` (since we can't differentiate through quantization)

---

## replace_parameter_4bit

Quantizes an existing module parameter to 4-bit and sets up parametrization for automatic dequantization.

**Location:** `bitsandbytes/nn/parametrize.py`

### Function Signature

```python
def replace_parameter_4bit(
    module: nn.Module,
    param_name: str,
    compress_statistics: bool = False,
    quant_type: Literal["nf4", "fp4"] = "nf4",
    blocksize: Optional[int] = None,
):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `nn.Module` | (required) | The module containing the parameter to quantize |
| `param_name` | `str` | (required) | Name of the parameter within the module (e.g., `"weight"`) |
| `compress_statistics` | `bool` | `False` | Whether to additionally quantize the absmax values (double quantization) |
| `quant_type` | `Literal["nf4", "fp4"]` | `"nf4"` | The 4-bit quantization format |
| `blocksize` | `Optional[int]` | `None` | Block size for quantization (default: 64) |

### Raises

| Exception | Condition |
|-----------|-----------|
| `AttributeError` | Module does not have the specified parameter |
| `TypeError` | The specified attribute is not an `nn.Parameter` |

### Step-by-Step Process

#### 1. Validation

```python
if not hasattr(module, param_name):
    raise AttributeError(f"Module does not have parameter '{param_name}'")

original_param = getattr(module, param_name)

if not isinstance(original_param, nn.Parameter):
    raise TypeError(f"Parameter '{param_name}' is not an instance of nn.Parameter")
```

#### 2. Quantization

```python
quantized_data, quant_state = F.quantize_4bit(
    original_param.data,
    blocksize=blocksize,
    compress_statistics=compress_statistics,
    quant_type=quant_type,
)
```

This calls the standard `quantize_4bit` function from `bitsandbytes.functional`, which:
- Divides the parameter into blocks of `blocksize` elements
- Computes the absolute maximum per block
- Quantizes each value to 4 bits using the specified type (NF4 or FP4)
- Returns the packed uint8 data and a `QuantState` object

#### 3. Replace Parameter

```python
setattr(module, param_name, nn.Parameter(quantized_data, requires_grad=False))
del original_param
```

The original floating-point parameter is replaced with the quantized uint8 data. The `requires_grad=False` flag ensures no gradients are computed for the quantized representation. The original parameter is explicitly deleted to free memory.

#### 4. Register Parametrization

```python
P.register_parametrization(module, param_name, Bnb4bitParametrization(quant_state), unsafe=True)
```

The `unsafe=True` flag is required because:
- We are registering a parametrization on a parameter that already has `requires_grad=False`
- The parametrization output has a different dtype/shape than the input (dequantized float vs quantized uint8)

Without `unsafe=True`, PyTorch would reject the parametrization due to these inconsistencies.

#### 5. Register Hooks

```python
_register_parametrization_hooks(module, param_name)
```

See [Hook Registration](#hook-registration) for details.

### Example Usage

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# Simple MoE-style module with a custom parameter
class ExpertLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

# Create module and quantize the weight
expert = ExpertLayer(512)
bnb.replace_parameter_4bit(expert, "weight", quant_type="nf4")

# The weight is now 4-bit quantized
# When accessed in forward(), it is automatically dequantized
print(expert.weight.shape)        # torch.Size([512, 512]) -- dequantized
print(expert.parametrizations.weight.original.dtype)  # torch.uint8 -- quantized
```

---

## replace_parameter_4bit_prequantized

Loads an already-quantized parameter into a module with parametrization. This is used when loading pre-quantized model checkpoints.

**Location:** `bitsandbytes/nn/parametrize.py`

### Function Signature

```python
def replace_parameter_4bit_prequantized(
    module: nn.Module,
    param_name: str,
    qs_dict: dict[str, Any],
    device: torch.device,
):
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `nn.Module` | The module containing the parameter |
| `param_name` | `str` | Name of the parameter within the module |
| `qs_dict` | `dict[str, Any]` | Dictionary of quantization state components (as returned by `QuantState.as_dict()`) |
| `device` | `torch.device` | Target device for the quantization state tensors |

### Step-by-Step Process

#### 1. Validation

Same validation as `replace_parameter_4bit`: checks parameter existence and type.

#### 2. Reconstruct QuantState

```python
quant_state = F.QuantState.from_dict(qs_dict, device=device)
```

The `QuantState.from_dict()` method handles:
- Unpacking the packed tensor (if `quant_state.bitsandbytes__nf4` key exists)
- Stripping prefix names
- Converting string dtypes back to `torch.dtype`
- Reconstructing nested state (for double quantization)
- Moving tensors to the specified device

#### 3. Register Parametrization

```python
P.register_parametrization(module, param_name, Bnb4bitParametrization(quant_state), unsafe=True)
```

Same as the non-prequantized version.

#### 4. Register Hooks

```python
_register_parametrization_hooks(module, param_name)
```

### Example Usage

```python
# Loading a pre-quantized checkpoint
expert = ExpertLayer(512)

# Load quantized weight data
expert.weight.data = torch.load("quantized_weight.pt")

# Load quantization state
qs_dict = torch.load("quant_state.pt")
bnb.replace_parameter_4bit_prequantized(expert, "weight", qs_dict, device=torch.device("cuda:0"))
```

---

## Hook Registration

The `_register_parametrization_hooks` function registers forward hooks and a state dict post-hook to manage caching and serialization.

### Hook Registration Code

```python
def _register_parametrization_hooks(module: nn.Module, param_name: str):
    # State dict post-hook (torch >= 2.5.0 only)
    if torch.__version__ >= (2, 5):
        module.register_state_dict_post_hook(
            partial(_parametrized_state_dict_post_hook, param_name=param_name)
        )

    # Forward hooks for cache management
    module.register_forward_pre_hook(_enable_parametrization_cache)
    module.register_forward_hook(_disable_parametrization_cache)
```

### _enable_parametrization_cache (Forward Pre-Hook)

```python
def _enable_parametrization_cache(module, inputs):
    P._cache_enabled += 1
```

Called before the module's `forward()` method. Increments the parametrization cache counter, which enables caching of the dequantized parameter. This means the parameter is dequantized only once per forward pass, even if accessed multiple times.

This accesses PyTorch's internal `P._cache_enabled` counter (where `P = torch.nn.utils.parametrize`). When `_cache_enabled > 0`, parametrization results are cached.

### _disable_parametrization_cache (Forward Hook)

```python
def _disable_parametrization_cache(module, inputs, output):
    P._cache_enabled -= 1
    if not P._cache_enabled:
        P._cache = {}
```

Called after the module's `forward()` method. Decrements the cache counter. When the counter reaches zero (no nested forward calls), the cache is cleared to free memory.

### Why Caching Matters

Without caching, every access to `module.weight` would trigger a full dequantization. In a typical module, the weight might be accessed multiple times:
1. During the matrix multiplication in the forward pass
2. During gradient computation (if applicable)
3. During any inspection or debugging

The cache ensures that within a single forward call, the dequantized weight is computed only once.

### Cache Nesting

The counter-based approach supports nested module calls correctly:

```
outer.forward()          -> _cache_enabled = 1 (cache enabled)
    inner.forward()      -> _cache_enabled = 2 (still cached)
    inner.forward() done -> _cache_enabled = 1 (cache still active)
outer.forward() done     -> _cache_enabled = 0 (cache cleared)
```

---

## State Dict Serialization

The `_parametrized_state_dict_post_hook` ensures that quantized parameters are saved correctly in the model's state dict.

### Hook Implementation

```python
def _parametrized_state_dict_post_hook(
    module, state_dict, prefix, local_metadata, *,
    param_name="weight", **kwargs
):
    original_key = f"{prefix}parametrizations.{param_name}.original"

    if original_key in state_dict:
        # Move quantized data to the clean key
        clean_key = f"{prefix}{param_name}"
        state_dict[clean_key] = state_dict.pop(original_key)

        # Find the parametrization to get the quantization state
        parametrization = next(
            filter(lambda x: isinstance(x, Bnb4bitParametrization),
                   module.parametrizations[param_name]),
            None
        )
        assert parametrization is not None

        quant_state = parametrization.quant_state

        # Save quantization state components
        if quant_state is not None:
            for k, v in quant_state.as_dict(packed=True).items():
                state_dict[f"{prefix}{param_name}.{k}"] = v
```

### What Gets Saved

For a parameter named `weight` in a module at prefix `model.layers.0.`, the state dict will contain:

```
model.layers.0.weight                           # Quantized uint8 data
model.layers.0.weight.quant_state.bitsandbytes__nf4  # Packed metadata tensor
model.layers.0.weight.absmax                    # Block absolute maxima
model.layers.0.weight.quant_map                 # Quantization code map (NF4/FP4 values)
```

### Key Transformation

1. **Before hook:** `model.layers.0.parametrizations.weight.original` (PyTorch's internal key)
2. **After hook:** `model.layers.0.weight` (clean key, compatible with standard loading)

The packed format (`quant_state.bitsandbytes__nf4`) is critical for safetensors compatibility, which only supports tensor values (not strings, ints, or other Python objects).

### Loading

When loading a state dict saved by this hook:

```python
# Load the model with quantization first
model = MyModel()
replace_parameter_4bit(model.layer, "weight", quant_type="nf4")

# Then load the state dict
state_dict = torch.load("checkpoint.pt")
model.load_state_dict(state_dict)
```

The loading process:
1. PyTorch loads `model.layer.weight` into the parametrized parameter's `original` tensor
2. The quantization state keys (e.g., `model.layer.weight.absmax`) are matched and loaded
3. For pre-quantized loading, use `replace_parameter_4bit_prequantized` instead

### Torch Version Requirement

The state dict post-hook requires PyTorch >= 2.5.0 (the `register_state_dict_post_hook` API was added in that version). For older PyTorch versions, the hook is simply not registered, and state dict saving will use the default PyTorch behavior (which includes the `parametrizations.weight.original` key).

```python
if torch.__version__ >= (2, 5):
    module.register_state_dict_post_hook(
        partial(_parametrized_state_dict_post_hook, param_name=param_name)
    )
```

---

## Use Cases

### Mixture-of-Experts (MoE) Models

MoE models have multiple expert modules, each with their own weight parameters. The parametrize API allows quantizing each expert's weights independently:

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.experts = nn.ModuleList([
            ExpertLayer(dim) for _ in range(num_experts)
        ])

    def forward(self, x, expert_indices):
        # Route to selected experts
        outputs = []
        for i in expert_indices:
            outputs.append(self.experts[i](x))
        return torch.stack(outputs)

# Quantize all experts
moe = MoELayer(num_experts=8, dim=512)
for expert in moe.experts:
    bnb.replace_parameter_4bit(expert, "weight", quant_type="nf4")
```

### Non-Linear Module Quantization

Quantize parameters in any module type, not just `nn.Linear`:

```python
class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # Custom parameter for relative position bias
        self.rel_pos_bias = nn.Parameter(torch.randn(64, dim))

    def forward(self, x):
        # ... attention computation
        pass

attn = AttentionLayer(768)
# Quantize the custom parameter
bnb.replace_parameter_4bit(attn, "rel_pos_bias", quant_type="nf4")
```

### Custom Module Quantization

For modules with non-standard forward patterns:

```python
class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Parameter(torch.randn(hidden_dim, dim))
        self.up_proj = nn.Parameter(torch.randn(hidden_dim, dim))
        self.down_proj = nn.Parameter(torch.randn(dim, hidden_dim))

    def forward(self, x):
        gate = torch.nn.functional.linear(x, self.gate_proj)
        up = torch.nn.functional.linear(x, self.up_proj)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up,
            self.down_proj,
        )

mlp = GatedMLP(512, 2048)
# Quantize all projection weights
for name in ["gate_proj", "up_proj", "down_proj"]:
    bnb.replace_parameter_4bit(mlp, name, quant_type="nf4")
```

### Integration with PEFT/LoRA

The parametrize API is compatible with PEFT LoRA adapters. Quantize the base weight first, then add LoRA:

```python
from peft import LoraConfig, get_peft_model

# 1. Create model with quantized base weights
model = MyModel()
for module in model.modules():
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        bnb.replace_parameter_4bit(module, "weight", quant_type="nf4")

# 2. Add LoRA adapters on top
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Only LoRA parameters are trainable; base weights stay quantized
```

### Caution: Memory and Performance

- **Dequantization overhead**: The weight is dequantized on every forward pass. For large models, this can be a bottleneck.
- **Memory**: The dequantized weight exists in memory during the forward pass. Ensure sufficient GPU memory for the full fp16/bf16 weight during computation.
- **Gradient flow**: Since the original parameter has `requires_grad=False`, no gradients flow through the quantized parameter. This is appropriate for QLoRA-style training where only adapter weights are updated.
