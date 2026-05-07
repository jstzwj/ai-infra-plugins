# 08 - Optimizer Variants Reference

This document provides a comprehensive reference for all optimizer variants available in bitsandbytes. Every optimizer is available in multiple configurations: 32-bit (standard precision), 8-bit (memory-efficient), and paged (for large models with unified memory support).

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Base Class Hierarchy](#base-class-hierarchy)
- [GlobalOptimManager](#globaloptimmanager)
- [Adam](#adam)
- [AdamW](#adamw)
- [SGD (Momentum)](#sgd-momentum)
- [Lion](#lion)
- [LAMB](#lamb)
- [LARS](#lars)
- [Adagrad](#adagrad)
- [RMSprop](#rmsprop)
- [AdEMAMix](#ademamix)
- [Memory Savings Comparison](#memory-savings-comparison)
- [Choosing the Right Optimizer](#choosing-the-right-optimizer)
- [GlobalOptimManager Per-Parameter Overrides](#globaloptimmanager-per-parameter-overrides)

---

## Architecture Overview

All bitsandbytes optimizers inherit from `torch.optim.Optimizer` and share a common architecture:

```
torch.optim.Optimizer
  |-- Optimizer8bit                 (bitsandbytes/optim/optimizer.py)
        |-- Optimizer1State         (1-state: momentum-based optimizers)
        |-- Optimizer2State         (2-state: Adam-family optimizers)
```

**Optimizer1State** maintains a single state tensor per parameter (e.g., momentum buffer). Used by SGD, Lion, LARS, Adagrad, and RMSprop.

**Optimizer2State** maintains two state tensors per parameter (e.g., first and second moment estimates). Used by Adam, AdamW, LAMB, and AdEMAMix.

The state type depends on the `optim_bits` setting:
- **32-bit** (`optim_bits=32`): State tensors are `torch.float32`
- **8-bit** (`optim_bits=8`): State tensors are `torch.uint8` with blockwise quantization (blocksize=256), plus per-block `absmax` scales and quantization maps (`qmap`)

For 8-bit mode, the quantization uses dynamic quantization maps created by `F.create_dynamic_map()`:
- `qmap1` (signed dynamic map) for state1
- `qmap2` (unsigned dynamic map) for state2

---

## Base Class Hierarchy

### Optimizer8bit

```python
class Optimizer8bit(torch.optim.Optimizer):
    def __init__(self, params, defaults, optim_bits=32, is_paged=False)
```

The base class for all bitsandbytes optimizers. Key responsibilities:
- Manages quantization maps (`name2qmap`)
- Handles paged memory via `GlobalPageManager`
- Provides `state_dict()` / `load_state_dict()` with FSDP compatibility
- Implements the `step()` loop with lazy initialization, state prefetching, and GPU synchronization

### Optimizer1State

```python
class Optimizer1State(Optimizer8bit):
    def __init__(
        self,
        optimizer_name: str,
        params,
        lr=1e-3,
        betas=(0.9, 0.0),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False,
    )
```

Maintains one state buffer per parameter. The update step dispatches to either `F.optimizer_update_32bit()` or `F.optimizer_update_8bit_blockwise()` depending on the state dtype.

### Optimizer2State

```python
class Optimizer2State(Optimizer8bit):
    def __init__(
        self,
        optimizer_name: str,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False,
        alpha=0.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
    )
```

Maintains two state buffers per parameter. The `alpha`, `t_alpha`, and `t_beta3` parameters are specific to AdEMAMix but are included in the base for dispatch compatibility.

---

## GlobalOptimManager

The `GlobalOptimManager` is a singleton that enables per-parameter optimizer configuration overrides. This allows mixing 8-bit and 32-bit states within the same optimizer.

```python
mng = bnb.optim.GlobalOptimManager.get_instance()

# Register parameters before moving to GPU
mng.register_parameters(model.parameters())

model = model.cuda()
adam = bnb.optim.Adam(model.parameters(), lr=0.001, optim_bits=8)

# Override: model.fc1.weight uses 32-bit states instead
mng.override_config(model.fc1.weight, 'optim_bits', 32)
```

The `get_config()` method merges the base configuration with per-parameter overrides from both `index2config` (position-based) and `pid2config` (id-based) lookups.

---

## Adam

**Class hierarchy:** `Adam -> Optimizer2State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"adam"` (2-state optimizer)

The Adam optimizer from "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015). Maintains first moment (m) and second moment (v) estimates with bias correction.

### Variants

#### Adam

```python
class Adam(Optimizer2State):
    def __init__(
        self,
        params,                          # Model parameters or param groups
        lr=1e-3,                         # Learning rate
        betas=(0.9, 0.999),             # Coefficients for running averages of gradient and its square
        eps=1e-8,                        # Term added for numerical stability
        weight_decay=0,                  # Weight decay (L2 penalty)
        amsgrad=False,                   # Whether to use AMSGrad variant
        optim_bits=32,                   # State precision: 32 or 8
        args=None,                       # Additional arguments object
        min_8bit_size=4096,              # Minimum parameter size for 8-bit quantization
        is_paged=False,                  # Enable paged memory management
    )
```

The configurable base variant. Supports both 32-bit and 8-bit states via `optim_bits`, and AMSGrad via `amsgrad`. When `optim_bits=8`, the state buffers are quantized to uint8 with blockwise scaling.

#### Adam8bit

```python
class Adam8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,                   # Must be False - raises ValueError if True
        optim_bits=32,                   # Must be 32 (default) - hardcodes 8 internally
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

Hardcoded 8-bit variant. The `optim_bits` parameter is accepted for API compatibility but must be the default value of 32; the actual state precision is always 8 bits. The `amsgrad` parameter is not supported and will raise a `ValueError` if set to `True`.

#### Adam32bit

```python
class Adam32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

Hardcoded 32-bit variant. State buffers are always `torch.float32`.

#### PagedAdam

```python
class PagedAdam(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,                  # Ignored - always True
    )
```

Paged memory variant. State tensors larger than 100,000 elements are allocated via `F.get_paged()`, which uses CUDA unified memory for automatic CPU-GPU paging. The optimizer prefetches state before each update via `prefetch_state()`.

#### PagedAdam8bit

```python
class PagedAdam8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,                   # Must be False
        optim_bits=32,                   # Must be 32 (default)
        args=None,
        min_8bit_size=4096,
        is_paged=False,                  # Ignored - always True
    )
```

Combines 8-bit quantization with paged memory. Achieves maximum memory savings for large models.

#### PagedAdam32bit

```python
class PagedAdam32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,                  # Ignored - always True
    )
```

Paged memory with 32-bit precision. Useful for parameters where 8-bit quantization degrades training quality.

### Usage Example

```python
import bitsandbytes as bnb
import torch

model = torch.nn.Linear(4096, 4096).cuda()

# 8-bit Adam - saves ~75% optimizer memory
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)

# Paged 8-bit Adam - for models that don't fit in GPU memory
optimizer = bnb.optim.PagedAdam8bit(model.parameters(), lr=1e-3)

# Standard 32-bit Adam via bitsandbytes (same as torch.optim.Adam)
optimizer = bnb.optim.Adam32bit(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

### When to Use Adam Variants

| Variant | Memory per param | Use Case |
|---------|-----------------|----------|
| `Adam32bit` | 8 bytes (2 x float32) | When full precision is needed |
| `Adam8bit` | 2 bytes (2 x uint8 + overhead) | General-purpose memory savings |
| `PagedAdam32bit` | 8 bytes (paged) | Large models with OOM risk |
| `PagedAdam8bit` | 2 bytes (paged) | Maximum memory savings for large models |

---

## AdamW

**Class hierarchy:** `AdamW -> Optimizer2State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"adam"` (2-state optimizer, identical to Adam)

AdamW implements "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019). It uses the same kernel as Adam (`optimizer_name="adam"`) but applies decoupled weight decay by default (`weight_decay=1e-2`).

### Key Difference from Adam

The only code-level difference between `Adam` and `AdamW` is the default `weight_decay`:
- `Adam`: `weight_decay=0`
- `AdamW`: `weight_decay=1e-2`

The underlying update kernel is identical (`"adam"`), because the 32-bit kernel applies weight decay as `p *= (1 - lr * weight_decay)` (decoupled form) when `weight_decay > 0`.

### Variants

#### AdamW

```python
class AdamW(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,              # Default is 1e-2 (decoupled weight decay)
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

#### AdamW8bit

```python
class AdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,                   # Must be False
        optim_bits=32,                   # Must be 32 (default)
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

#### AdamW32bit

```python
class AdamW32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

#### PagedAdamW

```python
class PagedAdamW(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    )
```

#### PagedAdamW8bit

```python
class PagedAdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,                   # Must be False
        optim_bits=32,                   # Must be 32 (default)
        args=None,
        min_8bit_size=4096,
    )
```

#### PagedAdamW32bit

```python
class PagedAdamW32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# AdamW with 8-bit states - the most common choice for LLM fine-tuning
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4, weight_decay=0.01)

# Paged AdamW 8-bit - recommended for QLoRA / LoRA fine-tuning of large models
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
```

### When to Use AdamW Variants

AdamW is the recommended default optimizer for most deep learning workloads, especially transformer fine-tuning. The decoupled weight decay provides better generalization than L2 regularization.

| Variant | Memory per param | Use Case |
|---------|-----------------|----------|
| `AdamW8bit` | ~2 bytes | Default for LLM fine-tuning with QLoRA/LoRA |
| `PagedAdamW8bit` | ~2 bytes (paged) | When GPU memory is tight with large models |
| `PagedAdamW32bit` | 8 bytes (paged) | Large models needing full precision |

---

## SGD (Momentum)

**Class hierarchy:** `SGD -> Optimizer1State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"momentum"` (1-state optimizer)

Stochastic Gradient Descent with momentum. Uses the `"momentum"` kernel which maintains a single velocity buffer per parameter.

**Important:** SGD without momentum (`momentum=0`) is NOT supported and will raise `NotImplementedError`.

### Variants

#### SGD

```python
class SGD(Optimizer1State):
    def __init__(
        self,
        params,                          # Model parameters (required)
        lr,                              # Learning rate (required, no default)
        momentum=0,                      # Momentum factor (must be > 0)
        dampening=0,                     # Dampening for momentum
        weight_decay=0,                  # Weight decay (L2 penalty)
        nesterov=False,                  # Whether to use Nesterov momentum
        optim_bits=32,                   # State precision: 32 or 8
        args=None,
        min_8bit_size=4096,
    )
```

Note: `lr` is a required positional argument with no default value (unlike Adam where `lr=1e-3`).

#### SGD8bit

```python
class SGD8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,                              # Required
        momentum=0,                      # Must be > 0
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
    )
```

#### SGD32bit

```python
class SGD32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,                              # Required
        momentum=0,                      # Must be > 0
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# 8-bit SGD with momentum
optimizer = bnb.optim.SGD8bit(model.parameters(), lr=0.1, momentum=0.9)

# 32-bit SGD with Nesterov momentum
optimizer = bnb.optim.SGD32bit(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True,
)
```

### When to Use SGD Variants

SGD with momentum is effective for computer vision tasks and scenarios where Adam-family optimizers may overfit. The 8-bit variant saves memory on the momentum buffer.

**Memory per parameter:** 1 byte (8-bit) vs 4 bytes (32-bit) for the momentum buffer.

---

## Lion

**Class hierarchy:** `Lion -> Optimizer1State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"lion"` (1-state optimizer)

Lion (EvoLved Sign Momentum) from "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023). Unlike Adam, Lion uses sign operations on the momentum update, making it more memory-efficient by nature (only one state buffer needed instead of two).

### Update Rule

```
update = sign(beta1 * m + (1 - beta1) * g)
p = p - lr * update
m = beta2 * m + (1 - beta2) * g
```

### Variants

#### Lion

```python
class Lion(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,                         # Default is 1e-4 (lower than Adam's 1e-3)
        betas=(0.9, 0.99),              # Note: (0.9, 0.99) not (0.9, 0.999)
        weight_decay=0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

#### Lion8bit

```python
class Lion8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

#### Lion32bit

```python
class Lion32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
        is_paged=False,
    )
```

#### PagedLion

```python
class PagedLion(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    )
```

#### PagedLion8bit

```python
class PagedLion8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
    )
```

#### PagedLion32bit

```python
class PagedLion32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0,
        args=None,
        min_8bit_size=4096,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# 8-bit Lion - only 1 state buffer needed (vs 2 for Adam)
optimizer = bnb.optim.Lion8bit(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
)

# Paged 8-bit Lion for large models
optimizer = bnb.optim.PagedLion8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.99),
)
```

### When to Use Lion Variants

Lion is more memory-efficient than Adam because it only needs one state buffer (momentum) instead of two (first and second moments). It also tends to be faster due to simpler updates (sign operation).

**Memory per parameter:** 1 byte (8-bit) vs 4 bytes (32-bit) for the single momentum buffer. Compared to Adam's 2 bytes (8-bit, 2 buffers) or 8 bytes (32-bit, 2 buffers), Lion uses half the optimizer state memory.

**Recommended learning rate:** Lion typically requires 3-10x lower learning rate than Adam (default `lr=1e-4` vs Adam's `1e-3`).

---

## LAMB

**Class hierarchy:** `LAMB -> Optimizer2State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"lamb"` (2-state optimizer)

Large Batch Optimization for Deep Learning (You et al., 2020). LAMB extends Adam with layer-wise adaptive scaling (`max_unorm=1.0` by default), enabling large-batch training with stable convergence.

### Key Parameters

- `adam_w_mode` (bool, default `True`): Use decoupled weight decay (AdamW-style). Note: this parameter is accepted but only affects weight decay application logic, not the kernel selection.
- `max_unorm` (float, default `1.0`): Maximum update norm relative to weight norm. This enables the trust-ratio clipping that distinguishes LAMB from Adam.
- `bias_correction` (bool, default `True`): Apply bias correction to moment estimates.

### Variants

#### LAMB

```python
class LAMB(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adam_w_mode=True,                # Use decoupled weight decay
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        max_unorm=1.0,                   # Trust ratio clipping (key LAMB feature)
    )
```

#### LAMB8bit

```python
class LAMB8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adam_w_mode=True,
        args=None,
        min_8bit_size=4096,
        max_unorm=1.0,
    )
```

#### LAMB32bit

```python
class LAMB32bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        adam_w_mode=True,
        args=None,
        min_8bit_size=4096,
        max_unorm=1.0,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# 8-bit LAMB for large-batch distributed training
optimizer = bnb.optim.LAMB8bit(
    model.parameters(),
    lr=1e-3,
    max_unorm=1.0,        # Trust ratio clipping
    weight_decay=0.01,
)

# LAMB with custom trust ratio
optimizer = bnb.optim.LAMB(
    model.parameters(),
    lr=0.001,
    max_unorm=0.5,        # More conservative clipping
    adam_w_mode=True,
)
```

### When to Use LAMB

LAMB is designed for large-batch distributed training (batch sizes > 8K). The trust-ratio clipping normalizes updates to be proportional to the weight norm, preventing any single layer from dominating the update. No paged variants are provided because LAMB is typically used in distributed settings where memory is less constrained per device.

---

## LARS

**Class hierarchy:** `LARS -> Optimizer1State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"lars"` (1-state optimizer, uses `"momentum"` kernel on CUDA)

Layer-wise Adaptive Rate Scaling (You et al., 2017). Like LAMB, LARS applies layer-wise adaptive scaling but with a momentum-based optimizer instead of Adam.

**Important:** LARS without momentum (`momentum=0`) is NOT supported and will raise `NotImplementedError`.

### Key Parameters

- `max_unorm` (float, default `0.02`): Maximum update norm relative to weight norm. The trust-ratio coefficient for LARS.
- `nesterov` (bool): Accepted but not used in the 8-bit path (used by `PytorchLARS` fallback only).

### Variants

#### LARS

```python
class LARS(Optimizer1State):
    def __init__(
        self,
        params,
        lr,                              # Required (no default)
        momentum=0,                      # Must be > 0
        dampening=0,
        weight_decay=0,
        nesterov=False,
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
        max_unorm=0.02,                  # Trust ratio clipping
    )
```

#### LARS8bit

```python
class LARS8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum=0,                      # Must be > 0
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        max_unorm=0.02,
    )
```

#### LARS32bit

```python
class LARS32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr,
        momentum=0,                      # Must be > 0
        dampening=0,
        weight_decay=0,
        nesterov=False,
        args=None,
        min_8bit_size=4096,
        max_unorm=0.02,
    )
```

### PytorchLARS (Reference Implementation)

A pure PyTorch implementation of LARS is also provided for reference:

```python
class PytorchLars(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        max_unorm=0.02,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# 8-bit LARS for large-batch training
optimizer = bnb.optim.LARS8bit(
    model.parameters(),
    lr=1.0,
    momentum=0.9,
    weight_decay=1e-4,
    max_unorm=0.02,
)
```

### When to Use LARS

LARS is effective for large-batch training of convolutional networks and is commonly used in self-supervised learning (e.g., SimCLR, BYOL). It provides similar benefits to LAMB but for SGD-based training.

---

## Adagrad

**Class hierarchy:** `Adagrad -> Optimizer1State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"adagrad"` (1-state optimizer)

Adagrad from "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" (Duchi et al., 2011). Adapts learning rates per-parameter based on historical gradient magnitudes.

### Constraints

- `initial_accumulator_value` must be 0.0 (non-zero values not supported)
- `lr_decay` must be 0.0 (learning rate decay not supported)

### Variants

#### Adagrad

```python
class Adagrad(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,                         # Note: default is 1e-2, not 1e-3
        lr_decay=0,                      # Must be 0
        weight_decay=0,
        initial_accumulator_value=0,     # Must be 0
        eps=1e-10,                       # Note: default is 1e-10, not 1e-8
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    )
```

#### Adagrad8bit

```python
class Adagrad8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,                      # Must be 0
        weight_decay=0,
        initial_accumulator_value=0,     # Must be 0
        eps=1e-10,
        optim_bits=8,                    # Hardcoded
        args=None,
        min_8bit_size=4096,
    )
```

#### Adagrad32bit

```python
class Adagrad32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        optim_bits=32,                   # Hardcoded
        args=None,
        min_8bit_size=4096,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# 8-bit Adagrad
optimizer = bnb.optim.Adagrad8bit(
    model.parameters(),
    lr=1e-2,
    eps=1e-10,
)
```

### When to Use Adagrad

Adagrad is most effective for sparse data problems (e.g., NLP with rare features, recommendation systems). It adapts per-parameter learning rates based on accumulated squared gradients, giving larger updates for infrequent parameters. However, it can be overly aggressive in reducing learning rates for frequently updated parameters.

---

## RMSprop

**Class hierarchy:** `RMSprop -> Optimizer1State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"rmsprop"` (1-state optimizer)

RMSprop from "Neural Networks for Machine Learning" (Hinton et al., 2012, lecture 6). Divides the learning rate by a running average of the magnitudes of recent gradients.

### Constraints

- `alpha` must not be 0.0 (raises `NotImplementedError`)
- `centered=True` is NOT supported (raises `NotImplementedError`)

### Variants

#### RMSprop

```python
class RMSprop(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,                      # Smoothing constant (must be > 0)
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,                  # Must be False
        optim_bits=32,
        args=None,
        min_8bit_size=4096,
    )
```

#### RMSprop8bit

```python
class RMSprop8bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,                  # Must be False
        args=None,
        min_8bit_size=4096,
    )
```

#### RMSprop32bit

```python
class RMSprop32bit(Optimizer1State):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,                  # Must be False
        args=None,
        min_8bit_size=4096,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# 8-bit RMSprop
optimizer = bnb.optim.RMSprop8bit(
    model.parameters(),
    lr=1e-3,
    alpha=0.99,
    eps=1e-8,
)
```

### When to Use RMSprop

RMSprop is effective for non-stationary or noisy objectives and is popular in reinforcement learning and recurrent neural networks. It is less commonly used for transformer training where AdamW is dominant.

---

## AdEMAMix

**Class hierarchy:** `AdEMAMix -> Optimizer2State -> Optimizer8bit -> torch.optim.Optimizer`

**Internal optimizer name:** `"ademamix"` (2-state optimizer)

AdEMAMix from "The AdEMAMix Optimizer: Better, Faster, Older" (Pagliardini et al., 2024). Extends Adam with a second, slower EMA (exponential moving average) of gradients, combining fast and slow momentum to improve convergence.

### Unique Parameters

- `alpha` (float, default `5.0`): Weight of the slow EMA term relative to the fast EMA.
- `t_alpha` (Optional[int], default `None`): Number of steps for alpha scheduling. When set, alpha is linearly ramped from 0 to `alpha` over `t_alpha` steps: `alpha_t = min(step * alpha / t_alpha, alpha)`.
- `t_beta3` (Optional[int], default `None`): Number of steps for beta3 scheduling. When set, beta3 is interpolated from beta1 to the target beta3 using a geometric mean schedule.
- `betas` (tuple of 3 floats, default `(0.9, 0.999, 0.9999)`): Three betas for (fast EMA, second moment, slow EMA).

### Update Rule

```
m1 = beta1 * m1 + (1 - beta1) * g          # fast EMA
m2 = beta3 * m2 + (1 - beta3) * g          # slow EMA
nu = beta2 * nu + (1 - beta2) * g^2        # second moment

update = (m1 / correction1 + alpha * m2) / (sqrt(nu) / correction2 + eps)
p = p - lr * update + weight_decay_term
```

### State Layout

Unlike standard 2-state optimizers, AdEMAMix stores both the fast EMA (`m1`) and slow EMA (`m2`) in `state["state1"]` as a stacked tensor of shape `(2, *param.shape)`. The second moment (`nu`) is stored in `state["state2"]` with the normal param shape.

For 8-bit mode, `absmax1` has shape `(2, num_blocks)` to account for the doubled state1 buffer.

### Variants

#### AdEMAMix

```python
class AdEMAMix(Optimizer2State):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        optim_bits: Literal[8, 32] = 32,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    )
```

#### AdEMAMix8bit

```python
class AdEMAMix8bit(AdEMAMix):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    )
```

Hardcodes `optim_bits=8`.

#### AdEMAMix32bit

```python
class AdEMAMix32bit(Optimizer2State):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    )
```

Hardcodes `optim_bits=32`.

#### PagedAdEMAMix

```python
class PagedAdEMAMix(AdEMAMix):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        optim_bits: Literal[8, 32] = 32,
        min_8bit_size: int = 4096,
    )
```

#### PagedAdEMAMix8bit

```python
class PagedAdEMAMix8bit(AdEMAMix8bit):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
    )
```

#### PagedAdEMAMix32bit

```python
class PagedAdEMAMix32bit(AdEMAMix32bit):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
    )
```

### Usage Example

```python
import bitsandbytes as bnb

# Basic AdEMAMix
optimizer = bnb.optim.AdEMAMix(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999, 0.9999),
    alpha=5.0,
    weight_decay=1e-2,
)

# With scheduling - ramp up alpha and beta3 over training
optimizer = bnb.optim.PagedAdEMAMix8bit(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999, 0.9999),
    alpha=5.0,
    t_alpha=1000,          # Ramp alpha from 0 to 5.0 over 1000 steps
    t_beta3=5000,          # Ramp beta3 from beta1 to target over 5000 steps
    weight_decay=1e-2,
)
```

### When to Use AdEMAMix

AdEMAMix can be a drop-in replacement for AdamW, often achieving better convergence with the same hyperparameters. The slow EMA captures long-term gradient trends that the fast EMA misses. The `alpha` and scheduling parameters (`t_alpha`, `t_beta3`) allow fine-tuning the balance between fast and slow momentum.

**Memory note:** AdEMAMix uses more memory than Adam because `state1` is doubled (stacked fast + slow EMA). In 8-bit mode, this means `state1` takes 2 bytes per parameter (2 x uint8) instead of 1 byte. Combined with `state2` (1 byte per param), total 8-bit state memory is 3 bytes per parameter.

---

## Memory Savings Comparison

The following table shows approximate optimizer state memory per parameter for a model with N parameters:

| Optimizer | 32-bit State | 8-bit State | Savings |
|-----------|-------------|-------------|---------|
| **Adam / AdamW** | 8 bytes (2 x float32) | ~2 bytes (2 x uint8 + scales) | 75% |
| **SGD (momentum)** | 4 bytes (1 x float32) | ~1 byte (1 x uint8 + scales) | 75% |
| **Lion** | 4 bytes (1 x float32) | ~1 byte (1 x uint8 + scales) | 75% |
| **LAMB** | 8 bytes (2 x float32) | ~2 bytes (2 x uint8 + scales) | 75% |
| **LARS** | 4 bytes (1 x float32) | ~1 byte (1 x uint8 + scales) | 75% |
| **Adagrad** | 4 bytes (1 x float32) | ~1 byte (1 x uint8 + scales) | 75% |
| **RMSprop** | 4 bytes (1 x float32) | ~1 byte (1 x uint8 + scales) | 75% |
| **AdEMAMix** | 12 bytes (3 x float32) | ~3 bytes (3 x uint8 + scales) | 75% |

### Concrete Example: 7B Parameter Model

| Optimizer | 32-bit Total | 8-bit Total | Saved |
|-----------|-------------|-------------|-------|
| AdamW | 56 GB | ~14 GB | ~42 GB |
| Lion | 28 GB | ~7 GB | ~21 GB |
| AdEMAMix | 84 GB | ~21 GB | ~63 GB |

The "8-bit" column includes overhead for absmax scales and quantization maps, which is proportional to `numel / blocksize` where `blocksize=256`.

### Quantization Overhead

For 8-bit optimizers, additional overhead per parameter includes:
- **absmax** arrays: `numel / 256` float32 values per state buffer
- **qmap** tensors: 256-element float32 lookup tables (shared across all parameters)
- For small parameters (`numel < min_8bit_size=4096`), 8-bit quantization is bypassed and float32 is used instead

---

## Choosing the Right Optimizer

### Decision Matrix

| Scenario | Recommended Optimizer | Rationale |
|----------|----------------------|-----------|
| LLM fine-tuning (QLoRA/LoRA) | `AdamW8bit` or `PagedAdamW8bit` | Standard choice, maximum compatibility |
| Very large model, OOM risk | `PagedAdamW8bit` | Paged memory prevents OOM |
| Memory-constrained training | `Lion8bit` | Only 1 state buffer vs 2 for Adam |
| Large-batch distributed training | `LAMB8bit` | Trust-ratio clipping for stability |
| Computer vision, self-supervised | `LARS8bit` | Proven for SimCLR/BYOL-style training |
| Sparse features, recommendation systems | `Adagrad8bit` | Per-parameter adaptive learning rates |
| Reinforcement learning | `RMSprop8bit` | Handles non-stationary objectives well |
| Best convergence quality | `AdEMAMix8bit` | Fast + slow EMA captures more gradient history |

### 8-bit vs 32-bit Trade-offs

**When to use 8-bit:**
- Training or fine-tuning large models (>1B parameters)
- GPU memory is the bottleneck
- Slight precision loss in optimizer states is acceptable (empirically, 8-bit states rarely affect final model quality)

**When to use 32-bit:**
- Training small models where memory savings are negligible
- Debugging convergence issues (to rule out quantization artifacts)
- When training with very low learning rates where quantization noise could dominate
- For critical layers where full precision is required (use `GlobalOptimManager` to override per-parameter)

---

## GlobalOptimManager Per-Parameter Overrides

Use `GlobalOptimManager` to mix 8-bit and 32-bit states within a single optimizer:

```python
import bitsandbytes as bnb

model = MyModel().cuda()

# Register parameters BEFORE creating the optimizer
mng = bnb.optim.GlobalOptimManager.get_instance()
mng.register_parameters(model.parameters())

# Create 8-bit optimizer for all parameters
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)

# Override specific parameters to use 32-bit states
mng.override_config(model.embedding.weight, 'optim_bits', 32)
mng.override_config(model.output.weight, 'optim_bits', 32)

# Override with multiple config changes at once
mng.override_config(
    model.attention.q_proj.weight,
    key_value_dict={'optim_bits': 32, 'lr': 5e-4}
)
```

The override system supports changing any optimizer config key: `optim_bits`, `lr`, `betas`, `eps`, `weight_decay`, `min_8bit_size`, `max_unorm`, `skip_zeros`.
