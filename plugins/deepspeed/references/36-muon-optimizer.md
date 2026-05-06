# Muon Optimizer - MomentUm Orthogonalized by Newton-Schulz

## Overview

The Muon optimizer (**Mu**mentum **Orthogonalized** by **N**ewton-Schulz) is a novel optimization algorithm that orthogonalizes momentum-based weight updates using Newton-Schulz iterations. Unlike standard optimizers (Adam, SGD) that apply updates directly, Muon constrains the update direction to lie on the Stiefel manifold (the set of orthogonal matrices), which has been shown to improve training dynamics and final model quality for large-scale neural networks.

DeepSpeed provides multiple Muon implementations:
- **Original Muon** (`original_muon.py`): Standalone optimizer for single-device and distributed training
- **ZeRO-integrated Muon** (`muon_optimizer.py`): Optimizer subclass that works with ZeRO's flat parameter tensor management

---

## Mathematical Foundation

### Newton-Schulz Iteration (Standard)

The standard Newton-Schulz iteration computes an approximate polar decomposition to orthogonalize a matrix `G`. Given a matrix `G` with SVD `G = U @ S @ V^T`, the iteration converges to `U @ V^T` (the nearest orthogonal matrix):

```
Z_0 = G / ||G||_F          (initial Frobenius normalization)
Z_{k+1} = 0.5 * Z_k * (3 * I - Z_k^T @ Z_k)
```

In the DeepSpeed implementation, a fifth-order variant is used with polynomial coefficients:

```
Z_{k+1} = a * Z_k + b * Z_k @ (Z_k^T @ Z_k) + c * Z_k @ (Z_k^T @ Z_k) @ (Z_k^T @ Z_k)
```

Where `a = 3.4445`, `b = -4.7750`, `c = 2.0315`.

These coefficients are derived from a quintic polynomial approximation to the sign function, providing faster convergence than the standard cubic iteration.

### Gram-Based Newton-Schulz Iteration

For large matrices where `M > N` (rows > columns), the Gram-based variant computes the smaller Gram matrix `R = X^T @ X` (N x N) instead of `X @ X^T` (M x M), significantly reducing computational cost:

```
X = G / ||G||_F
R = X^T @ X                              (N x N Gram matrix)
R_new = a * R + b * R^2 + c * R^3        (polynomial iteration on R)
X = X @ (R_new @ inv_sqrt_R)             (project back)
```

This reduces the FLOP count from O(M^2 * N) to O(M * N^2) when M >> N.

### Muon Update Rule

The complete Muon update for a parameter matrix `W`:

```
1. Compute gradient: G = dL/dW
2. Update momentum:  M_t = beta * M_{t-1} + G
3. Orthogonalize:    O_t = newton_schulz(M_t, steps=ns_steps)
4. Update weight:    W_t = W_{t-1} - lr * O_t
```

With Nesterov momentum (optional):
```
1. Lookahead:       M_t = beta * M_{t-1} + G
2. Nesterov:        M_nesterov = M_t + beta * (M_t - M_{t-1})
3. Orthogonalize:   O_t = newton_schulz(M_nesterov, steps=ns_steps)
4. Update weight:   W_t = W_{t-1} - lr * O_t
```

---

## Original Muon Implementation (original_muon.py)

### `zeropower_via_newtonschulz5(G, steps=5, eps=1e-7)`

Computes an approximate polar decomposition using the standard fifth-order Newton-Schulz iteration.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G` | torch.Tensor | required | Input matrix to orthogonalize. Must be a 2D tensor. |
| `steps` | int | `5` | Number of Newton-Schulz iterations. More steps give better approximation. |
| `eps` | float | `1e-7` | Small epsilon for numerical stability in Frobenius normalization. |

#### Algorithm Details

1. Normalize `G` by its Frobenius norm: `X = G / (||G||_F + eps)`
2. Compute `A = X^T @ X` (the Gram matrix)
3. For each iteration step:
   - `B = a * A + b * (A @ A) + c * (A @ A @ A)` where `(a, b, c) = (3.4445, -4.7750, 2.0315)`
   - `A = B`
4. Compute result: `X @ A` (this approximates the orthogonal component)

#### Precision

All computation is done in **bfloat16** (`torch.bfloat16`) for performance. The coefficients and intermediate values are cast to bf16 before the iteration loop.

#### Returns

A 2D tensor of the same shape as `G`, approximately orthogonalized.

---

### `zeropower_via_gram_newtonschulz(G, steps=5, eps=1e-7)`

Gram-based variant of Newton-Schulz for tall/skinny matrices. More efficient when the number of rows significantly exceeds the number of columns.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G` | torch.Tensor | required | Input matrix (2D, shape M x N where M >= N) |
| `steps` | int | `5` | Number of Newton-Schulz iterations |
| `eps` | float | `1e-7` | Numerical stability epsilon |

#### Algorithm Details

1. Normalize: `X = G / (||G||_F + eps)`
2. Compute Gram matrix: `R = X^T @ X` (shape N x N)
3. Save `R_0 = R` for later inversion
4. For each iteration step:
   - `R = 3.4445 * R - 4.7750 * (R @ R) + 2.0315 * (R @ R @ R)`
5. Compute approximate inverse square root: `R_0_inv_sqrt = R_0^{-1/2}`
6. Result: `X = X @ R @ R_0_inv_sqrt`

#### Precision

Computation is done in **float16** (`torch.float16`) for the Gram matrix operations.

#### Returns

A 2D tensor of the same shape as `G`, approximately orthogonalized using the Gram approach.

---

### `muon_update(grad, momentum, beta, ns_steps, nesterov, ns_method)`

Performs a single Muon update step: momentum update followed by orthogonalization.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grad` | torch.Tensor | required | Current gradient for the parameter |
| `momentum` | torch.Tensor | required | Momentum buffer (same shape as grad). Modified in-place. |
| `beta` | float | required | Momentum coefficient (typically 0.95) |
| `ns_steps` | int | required | Number of Newton-Schulz iterations |
| `nesterov` | bool | required | Whether to use Nesterov-style momentum lookahead |
| `ns_method` | str | required | Orthogonalization method: `"standard"` or `"gram"` |

#### Returns

The orthogonalized update direction (torch.Tensor, same shape as input).

---

### `Muon` Class

Distributed Muon optimizer for multi-GPU training. Handles gradient synchronization via all-reduce and applies orthogonalized momentum updates.

#### Constructor

```python
Muon(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, ns_method="standard")
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Model parameters or parameter groups to optimize |
| `lr` | float | `0.02` | Learning rate. Note: Muon typically uses higher learning rates than Adam (0.01-0.02 vs 1e-4). |
| `momentum` | float | `0.95` | Momentum coefficient (beta). Range [0, 1). |
| `nesterov` | bool | `True` | Use Nesterov-style momentum. Improves convergence in practice. |
| `ns_steps` | int | `5` | Number of Newton-Schulz iterations for orthogonalization. |
| `ns_method` | str | `"standard"` | Newton-Schulz variant: `"standard"` or `"gram"`. |

#### Key Behaviors

- **Distributed**: Calls `torch.distributed.all_reduce()` on gradients before applying the update. All ranks must participate.
- **2D Parameters Only**: Skips parameters with more or fewer than 2 dimensions (biases, scalars, etc.). These should be optimized by a separate optimizer.
- **Momentum Buffers**: Maintains a `momentum_buffer` dict keyed by parameter, initialized as zeros on first step.

#### `step(closure=None)` Method

Performs a single optimization step:

1. Zeroes gradients and all-reduces them across ranks
2. For each 2D parameter:
   - Retrieves or initializes momentum buffer
   - Calls `muon_update()` to compute the orthogonalized update
   - Applies the update: `param.data -= lr * update`

---

### `SingleDeviceMuon` Class

Single-device variant of Muon that skips the distributed all-reduce. Identical parameters and behavior otherwise.

#### Constructor

```python
SingleDeviceMuon(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, ns_method="standard")
```

Use this when training on a single GPU without distributed data parallel.

---

### `MuonWithAuxAdam` Class

Hybrid optimizer that applies Muon to 2D weight matrices and Adam to all other parameters (biases, norms, scalars). This is the recommended way to use Muon for full model training.

#### Constructor

```python
MuonWithAuxAdam(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                ns_method="standard", adam_lr=1e-3, adam_betas=(0.9, 0.999),
                adam_eps=1e-8, adam_weight_decay=0.0)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Model parameters. Should include `use_muon` flag in param groups. |
| `lr` | float | `0.02` | Learning rate for Muon parameters |
| `momentum` | float | `0.95` | Muon momentum coefficient |
| `nesterov` | bool | `True` | Nesterov momentum for Muon |
| `ns_steps` | int | `5` | Newton-Schulz iteration count |
| `ns_method` | str | `"standard"` | Newton-Schulz variant |
| `adam_lr` | float | `1e-3` | Learning rate for Adam (non-Muon) parameters |
| `adam_betas` | tuple | `(0.9, 0.999)` | Adam beta coefficients |
| `adam_eps` | float | `1e-8` | Adam epsilon for numerical stability |
| `adam_weight_decay` | float | `0.0` | Adam weight decay |

#### Parameter Group Configuration

Parameters must be split into groups with `use_muon` flag:

```python
import deepspeed

# Split parameters into Muon and Adam groups
muon_params = []
adam_params = []

for name, param in model.named_parameters():
    if param.ndim >= 2:
        muon_params.append(param)
    else:
        adam_params.append(param)

param_groups = [
    {"params": muon_params, "use_muon": True},
    {"params": adam_params, "use_muon": False},
]

optimizer = MuonWithAuxAdam(param_groups, lr=0.02, adam_lr=1e-3)
```

The `use_muon` flag is automatically set by `deepspeed.set_optimizer_flags()` (see below).

#### `step(closure=None)` Method

1. All-reduce gradients across ranks
2. For each parameter group:
   - If `use_muon=True` and param is 2D: apply Muon update (momentum + Newton-Schulz)
   - Otherwise: apply Adam update (first moment + second moment + bias correction)

---

### `SingleDeviceMuonWithAuxAdam` Class

Single-device variant of `MuonWithAuxAdam`. Identical behavior but skips the distributed all-reduce step.

---

### `adam_update(param, grad, exp_avg, exp_avg_sq, step, beta1, beta2, lr, eps, weight_decay)`

Helper function implementing a single Adam update step. Used internally by `MuonWithAuxAdam`.

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `param` | torch.Tensor | Parameter to update |
| `grad` | torch.Tensor | Gradient |
| `exp_avg` | torch.Tensor | First moment estimate (modified in-place) |
| `exp_avg_sq` | torch.Tensor | Second moment estimate (modified in-place) |
| `step` | int | Current step count (for bias correction) |
| `beta1` | float | First moment decay rate |
| `beta2` | float | Second moment decay rate |
| `lr` | float | Learning rate |
| `eps` | float | Numerical stability epsilon |
| `weight_decay` | float | Weight decay coefficient |

---

## ZeRO-Integrated Muon (muon_optimizer.py)

### `MuonWithAuxAdam` (ZeRO variant)

This is a subclass of the original `MuonWithAuxAdam` that overrides `step()` to handle ZeRO's flat parameter tensor management. In ZeRO stages 2 and 3, parameters are flattened into contiguous 1D tensors ("flat params"), so the optimizer must handle these differently from standalone 2D parameters.

#### How It Works

The ZeRO variant overrides `step()` to distinguish between Muon and non-Muon parameter groups:

**For Muon parameter groups (`use_muon=True`)**:
1. Gradient is already in the flat param's `.grad` attribute
2. Performs SGD-like update directly: `param.data -= lr * orthogonalized_update`
3. Does NOT use Adam momentum/variance states
4. Only processes 2D parameters (skips 1D biases/norms that may be in the group)

**For non-Muon parameter groups (`use_muon=False`)**:
1. Uses standard Adam update with bias correction
2. Maintains `exp_avg` and `exp_avg_sq` states per parameter
3. Handles weight decay

#### Constructor

```python
from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
```

Takes the same parameters as the original `MuonWithAuxAdam`.

#### Integration with ZeRO

The ZeRO-integrated Muon is automatically used when:
1. The DeepSpeed config specifies `"type": "Muon"` in the optimizer block
2. ZeRO optimization is enabled (stage 1, 2, or 3)

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": true,
            "ns_steps": 5,
            "ns_method": "standard",
            "adam_lr": 1e-3,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "adam_weight_decay": 0.0
        }
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

---

## `deepspeed.set_optimizer_flags()`

This utility function automatically marks parameters with the `use_muon` attribute based on their dimensionality, enabling proper Muon/Adam splitting.

### Function Signature

```python
deepspeed.set_optimizer_flags(model)
```

### Behavior

Iterates over all parameters in the model and sets:
- `param.use_muon = True` if `param.ndim >= 2` (weight matrices)
- `param.use_muon = False` if `param.ndim < 2` (biases, norm parameters, scalars)

### Usage

```python
import deepspeed

model = MyModel()
deepspeed.set_optimizer_flags(model)

# Now param.use_muon is set, MuonWithAuxAdam will use it automatically
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=config_dict
)
```

---

## Configuration Reference

### Muon Optimizer Configuration

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": true,
            "ns_steps": 5,
            "ns_method": "standard",
            "adam_lr": 0.0003,
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1e-8,
            "adam_weight_decay": 0.0
        }
    }
}
```

### Parameter Table

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `lr` | float | `0.02` | [0.001, 0.1] | Learning rate for Muon parameters. Typically 10-100x higher than Adam lr. |
| `momentum` | float | `0.95` | [0.8, 0.99] | Momentum coefficient (beta). Higher values smooth the update trajectory. |
| `nesterov` | bool | `true` | - | Enable Nesterov momentum lookahead. Recommended true. |
| `ns_steps` | int | `5` | [1, 25] | Newton-Schulz iteration count. 5 is a good default; more steps improve orthogonalization quality at computational cost. |
| `ns_method` | str | `"standard"` | `"standard"`, `"gram"` | Newton-Schulz variant. Use `"gram"` for tall/skinny weight matrices. |
| `adam_lr` | float | `1e-3` | [1e-5, 1e-2] | Learning rate for non-Muon (Adam) parameters |
| `adam_betas` | list[float] | `[0.9, 0.999]` | - | Adam beta coefficients for non-Muon parameters |
| `adam_eps` | float | `1e-8` | [1e-10, 1e-6] | Adam epsilon for non-Muon parameters |
| `adam_weight_decay` | float | `0.0` | [0.0, 0.1] | Weight decay for Adam parameters |

### ZeRO Integration Configuration

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 0.02,
            "momentum": 0.95,
            "ns_steps": 5
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

### Mixed Precision with Muon

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 0.02
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

BF16 is recommended for Muon training since the Newton-Schulz iteration already uses bf16/fp16 internally.

---

## Performance Considerations

### Computational Overhead

The Newton-Schulz orthogonalization adds per-step overhead:
- **Standard method**: O(M * N * ns_steps) FLOPs per 2D parameter of shape (M, N)
- **Gram method**: O((M + N) * N^2 * ns_steps) FLOPs -- more efficient when M >> N

For a typical transformer layer with weight matrices (4096 x 4096) and `ns_steps=5`:
- Standard: ~5 * 4096^3 = ~344 GFLOPs per weight matrix
- Gram: same for square matrices, but ~2x faster for (4096 x 1024) matrices

### Memory Overhead

Muon requires one additional momentum buffer per 2D parameter:
- **Momentum buffer**: Same size as the parameter (fp32 or bf16)
- **Total**: ~1x parameter size additional memory

Compared to Adam which requires 2x (first + second moment), Muon uses **50% less optimizer memory**.

### Recommended Settings by Model Size

| Model Size | lr | momentum | ns_steps | ns_method | adam_lr |
|---|---|---|---|---|---|
| < 100M | 0.02 | 0.95 | 5 | standard | 1e-3 |
| 100M - 1B | 0.02 | 0.95 | 5 | standard | 3e-4 |
| 1B - 7B | 0.01 | 0.95 | 5 | gram | 1e-4 |
| 7B - 70B | 0.01 | 0.95 | 5 | gram | 1e-4 |
| > 70B | 0.005 | 0.95 | 5 | gram | 5e-5 |

### Scaling Behavior

Muon's learning rate scales differently from Adam:
- Muon lr typically remains in the range [0.005, 0.02] regardless of model size
- Adam lr scales down with model size (1e-4 to 5e-5)
- This is because the orthogonalization normalizes the update magnitude

---

## Troubleshooting

### NaN Loss After First Step

**Cause**: Newton-Schulz iteration diverging due to large gradients.

**Solution**: Reduce `ns_steps` to 3 or use `"gram"` method. Ensure the learning rate is not too high (try 0.01 instead of 0.02).

### Out of Memory

**Cause**: Newton-Schulz iteration creates temporary matrices.

**Solution**: Use `"gram"` method which has lower peak memory for non-square matrices. Enable gradient checkpointing. Use ZeRO stage 3.

### Slow Training

**Cause**: Newton-Schulz iteration on every step for every weight matrix.

**Solution**: Reduce `ns_steps` from 5 to 3. Profile to confirm the bottleneck is in orthogonalization. Consider using a smaller `ns_steps` with Nesterov momentum enabled for similar convergence quality.

### Non-2D Parameters Not Updating

**Cause**: Muon only updates 2D parameters. 1D parameters (biases, norms) require the auxiliary Adam optimizer.

**Solution**: Ensure you are using `MuonWithAuxAdam` (the default when selecting `"type": "Muon"` in config). Call `deepspeed.set_optimizer_flags(model)` before `deepspeed.initialize()`.

---

## Source Files

| File | Location | Description |
|---|---|---|
| `original_muon.py` | `deepspeed/runtime/zero/muon/` | Standalone Muon with standard and Gram Newton-Schulz |
| `muon_optimizer.py` | `deepspeed/runtime/zero/muon/` | ZeRO-integrated MuonWithAuxAdam subclass |
| `__init__.py` | `deepspeed/` | `set_optimizer_flags()` function |
