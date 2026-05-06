# DeepSpeed Optimizers

## Overview

DeepSpeed provides a comprehensive suite of high-performance optimizers designed for large-scale model training. These include fused GPU implementations for standard optimizers (Adam, AdamW, LAMB, Lion), CPU-optimized versions for parameter offloading, communication-efficient variants (1-bit Adam, 0/1 Adam, 1-bit LAMB), and novel optimizers such as Muon (MomentUm Orthogonalized by Newton-Schulz). All optimizers are registered in the `DEEPSPEED_OPTIMIZERS` list and can be selected via the DeepSpeed configuration JSON.

---

## DEEPSPEED_OPTIMIZERS Registry

The complete list of DeepSpeed-provided optimizers:

```python
from deepspeed.utils import logger

DEEPSPEED_OPTIMIZERS = [
    "adagrad",
    "adam",
    "adamw",
    "lamb",
    "onebitadam",
    "zerooneadam",
    "onebitlamb",
    "muadam",
    "muadamw",
    "musgd",
    "lion",
    "muon",
]
```

Each optimizer is selected via the `"type"` field in the optimizer configuration block:

```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

### Optimizer Name Resolution

DeepSpeed uses case-insensitive matching and aliases:
- `"adam"` and `"adamw"` both map to `FusedAdam` (with `adam_w_mode` controlling the variant)
- `"lamb"` maps to `FusedLamb`
- `"lion"` maps to `FusedLion`
- `"onebitadam"` maps to `OneBitAdam`
- `"zerooneadam"` maps to `ZeroOneAdam`
- `"onebitlamb"` maps to `OneBitLamb`
- `"muon"` maps to `Muon`
- `"muadam"` maps to `MuAdam`
- `"muadamw"` maps to `MuAdamW`
- `"musgd"` maps to `MuSGD`
- `"adagrad"` maps to `CPUAdagrad`

---

## FusedAdam (ops/adam/fused_adam.py)

The default and most widely used optimizer in DeepSpeed. FusedAdam is a GPU-only implementation that fuses multiple Adam operations into a single CUDA kernel for maximum throughput.

### Key Characteristics

- **GPU-only**: All state and computation resides on GPU. Not suitable for CPU offloading.
- **Fused kernel**: Combines the Adam update (momentum, variance, bias correction, weight decay, and parameter update) into a single kernel launch, reducing kernel launch overhead and memory bandwidth usage.
- **Supports both Adam and AdamW**: Controlled by the `adam_w_mode` parameter.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.001` | Learning rate |
| `bias_correction` | bool | `true` | Whether to apply bias correction to first and second moment estimates |
| `betas` | tuple[float, float] | `(0.9, 0.999)` | Coefficients for computing running averages of gradient and its square |
| `eps` | float | `1e-8` | Term added to the denominator for numerical stability |
| `weight_decay` | float | `0` | Weight decay coefficient. In Adam mode, applied as L2 regularization. In AdamW mode, applied as decoupled weight decay. |
| `adam_w_mode` | bool | `true` | If `true`, uses decoupled weight decay (AdamW). If `false`, uses L2 regularization (Adam). This is controlled by the `ADAM_W_MODE` global which defaults to `True`. |
| `amsgrad` | bool | `false` | Whether to use the AMSGrad variant |
| `set_grad_none` | bool | `true` | If `true`, sets gradients to `None` instead of zeroing them, which is more memory-efficient |

### Adam vs AdamW Mode

The distinction between Adam and AdamW is critical for training stability:

**Adam (L2 regularization)**:
```
gradient = gradient + weight_decay * weight
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
weight = weight - lr * m / (sqrt(v) + eps)
```

**AdamW (decoupled weight decay)**:
```
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
weight = weight - lr * m / (sqrt(v) + eps)
weight = weight - lr * weight_decay * weight
```

### FusedAdam Implementation Details

```python
# Simplified fused kernel pseudocode
class FusedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 adam_w_mode=True, amsgrad=False, set_grad_none=True):
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Single fused kernel call
                fused_adam_kernel(
                    p.data,          # weight
                    grad,            # gradient
                    state['exp_avg'],     # first moment
                    state['exp_avg_sq'],  # second moment
                    group['lr'],
                    group['beta1'],
                    group['beta2'],
                    group['eps'],
                    group['weight_decay'],
                    state['step'],
                    group['adam_w_mode'],
                    group['bias_correction']
                )
                state['step'] += 1
        return loss
```

### Configuration Examples

#### Standard AdamW
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    }
}
```

#### Adam with L2 Regularization
```json
{
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "adam_w_mode": false
        }
    }
}
```

#### AMSGrad Variant
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": true
        }
    }
}
```

---

## CPU Adam (ops/adam/cpu_adam.py)

CPU Adam is an optimized Adam implementation that runs on the CPU. It is specifically designed for use with ZeRO offloading (ZeRO-Offload and ZeRO-Infinity), where optimizer states and computations are moved to CPU to reduce GPU memory usage.

### Key Characteristics

- **CPU-optimized**: Uses SIMD instructions (AVX, AVX2, AVX-512) for efficient CPU computation.
- **Compatible with ZeRO-Offload**: Designed to work seamlessly with DeepSpeed's parameter offloading.
- **Memory-efficient**: Processes parameters in chunks to minimize CPU memory spikes.
- **Supports both Adam and AdamW modes**: Via the `adam_w_mode` parameter.

### Parameters

Same parameters as FusedAdam, with additional considerations:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.001` | Learning rate |
| `bias_correction` | bool | `true` | Bias correction for moments |
| `betas` | tuple | `(0.9, 0.999)` | Beta values |
| `eps` | float | `1e-8` | Epsilon for numerical stability |
| `weight_decay` | float | `0` | Weight decay |
| `adam_w_mode` | bool | `true` | AdamW vs Adam mode |
| `amsgrad` | bool | `false` | AMSGrad variant |

### Usage with ZeRO-Offload

CPU Adam is automatically selected when using ZeRO-Offload:

```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

When `offload_optimizer.device` is set to `"cpu"`, DeepSpeed automatically uses the CPU Adam implementation instead of FusedAdam.

### CPU Adam Performance Considerations

- **SIMD width**: On CPUs with AVX-512, the vectorized Adam kernel processes 16 FP32 values per instruction.
- **Multi-threading**: CPU Adam uses OpenMP for parallel processing across CPU cores.
- **Memory bandwidth**: The primary bottleneck is CPU memory bandwidth. Use high-bandwidth memory (e.g., DDR5) for best performance.
- **Pin memory**: Always enable `pin_memory` for faster CPU-GPU transfers.

---

## FusedLamb (ops/lamb/)

LAMB (Layer-wise Adaptive Moments optimizer for Batch training) is designed for training with very large batch sizes. It provides layer-wise adaptive learning rates that enable stable training with batch sizes of millions.

### Key Characteristics

- **Layer-wise learning rate scaling**: Each layer gets its own trust ratio that scales the update.
- **Large batch training**: Specifically designed to scale batch sizes to millions without losing accuracy.
- **Fused GPU kernel**: Single kernel launch per parameter group for efficiency.

### LAMB Algorithm

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

# Bias correction
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

# Update ratio (per-layer)
r_t = ||w_t|| / ||m_hat / (sqrt(v_hat) + eps) + weight_decay * w_t||

# Update
w_{t+1} = w_t - lr * r_t * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w_t)
```

The trust ratio `r_t` automatically scales the learning rate for each layer based on the ratio of the weight norm to the update norm. This prevents layers with large weights from receiving disproportionately large updates.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.001` | Learning rate |
| `bias_correction` | bool | `true` | Whether to apply bias correction |
| `betas` | tuple | `(0.9, 0.999)` | Beta values |
| `eps` | float | `1e-6` | Epsilon |
| `weight_decay` | float | `0` | Weight decay |
| `max_grad_norm` | float | `1.0` | Maximum gradient norm for clipping |
| `adam_w_mode` | bool | `true` | AdamW vs Adam mode |

### Configuration Example

```json
{
    "optimizer": {
        "type": "LAMB",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-6,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        }
    }
}
```

---

## FusedLion (ops/lion/)

LION (EvoLved Sign Momentum) is a memory-efficient optimizer discovered through program search. It uses the sign of the momentum instead of the momentum value itself, making it simpler and often more effective than Adam.

### Key Characteristics

- **Memory efficient**: Only stores momentum (one state tensor per parameter vs. two for Adam).
- **Sign-based updates**: Uses `sign(momentum)` for the update direction.
- **No second moment**: Eliminates the need for the variance estimate, saving 50% optimizer state memory.
- **Fused GPU kernel**: Single kernel for maximum throughput.

### LION Algorithm

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
update = sign(m_t) * max(|g_t| + beta2 * (m_t - g_t), 0)
w_{t+1} = w_t - lr * (update + weight_decay * w_t)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.0001` | Learning rate. Note: LION typically uses 3-10x smaller lr than Adam. |
| `betas` | tuple | `(0.9, 0.99)` | Beta1 for momentum, beta2 for the sign update |
| `weight_decay` | float | `0` | Weight decay coefficient |
| `adam_w_mode` | bool | `true` | Always true for Lion (decoupled weight decay) |

### Configuration Example

```json
{
    "optimizer": {
        "type": "Lion",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.99],
            "weight_decay": 0.01
        }
    }
}
```

### LION vs AdamW

| Aspect | LION | AdamW |
|---|---|---|
| State memory per param | 1 tensor (momentum) | 2 tensors (momentum + variance) |
| Typical learning rate | 1e-4 to 1e-5 | 1e-3 to 1e-4 |
| Update computation | sign(momentum) | momentum / sqrt(variance) |
| Best for | Memory-constrained training | General-purpose training |

---

## 1-Bit Adam

1-Bit Adam is a communication-efficient variant of Adam that compresses gradient communication to 1 bit per value, significantly reducing the network bandwidth required for distributed training.

### Key Characteristics

- **1-bit gradient compression**: Reduces gradient communication volume by ~32x compared to FP32 allreduce.
- **Error compensation**: Maintains a residual error buffer to prevent accuracy loss from compression.
- **Two-phase training**: Starts with standard Adam warmup, then switches to 1-bit communication.
- **Compatible with ZeRO**: Can be combined with ZeRO stages 1 and 2.

### Architecture

The 1-bit Adam optimizer is located across multiple modules:
- `deepspeed/runtime/fp16/onebit/` - Core 1-bit communication logic
- `deepspeed/ops/` - CUDA kernels for compression

### Algorithm

```
# Phase 1: Warmup with standard AllReduce
for step in range(warmup_steps):
    standard_allreduce(gradients)

# Phase 2: 1-bit compressed communication
for step in remaining_steps:
    # Error-compensated compression
    compressed_grad = compress(grad + error_buffer, bits=1)
    allgather(compressed_grad)
    decompressed_grad = decompress(compressed_grad)
    error_buffer = (grad + error_buffer) - decompressed_grad
    grad = decompressed_grad
    # Standard Adam update with compressed gradients
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.001` | Learning rate |
| `betas` | tuple | `(0.9, 0.999)` | Beta values |
| `eps` | float | `1e-8` | Epsilon |
| `weight_decay` | float | `0` | Weight decay |
| `freeze_step` | int | - | Step at which to switch from warmup to 1-bit communication |

### Configuration Example

```json
{
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 1000
        }
    },
    "gradient_accumulation_steps": 1,
    "train_micro_batch_size_per_gpu": 16
}
```

### When to Use 1-Bit Adam

- Training on clusters with limited inter-node bandwidth (e.g., Ethernet instead of InfiniBand)
- Models where gradient communication is the bottleneck (moderate model size, large number of GPUs)
- The warmup phase is essential; too short a warmup can hurt convergence

---

## 0/1 Adam (ZeroOneAdam)

0/1 Adam is an improved version of 1-bit Adam that dynamically switches between 0-bit (no communication) and 1-bit communication based on gradient variance.

### Key Characteristics

- **Dynamic communication**: Skips communication entirely (0-bit) when gradients are sufficiently similar across workers.
- **Variance-based scheduling**: Uses gradient variance to decide when to communicate.
- **Better convergence**: The adaptive communication schedule often leads to better final accuracy than fixed 1-bit compression.
- **Reduced total communication**: Can reduce total communication by 2-5x compared to 1-bit Adam.

### Algorithm

```
for step in range(total_steps):
    if step < warmup_steps:
        # Standard AllReduce during warmup
        allreduce(gradients)
    else:
        # Check gradient variance
        variance = compute_variance(local_grad, previous_compressed_grad)
        if variance > threshold:
            # 1-bit communication
            compressed = compress(grad + error, bits=1)
            allgather(compressed)
        else:
            # 0-bit: skip communication, reuse previous gradient
            pass
    # Adam update
    adam_update(grad)
```

### Configuration Example

```json
{
    "optimizer": {
        "type": "ZeroOneAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 1000,
            "cuda_aware": false,
            "comm_backend_name": "nccl"
        }
    }
}
```

---

## 1-Bit LAMB

1-Bit LAMB combines the communication efficiency of 1-bit compression with the large-batch training capabilities of LAMB.

### Key Characteristics

- **1-bit gradient compression + LAMB's trust ratio**: Enables training with both limited bandwidth and large batch sizes.
- **Layer-wise compression**: Different layers can have different compression ratios.
- **Error compensation**: Same residual error mechanism as 1-bit Adam.

### Configuration Example

```json
{
    "optimizer": {
        "type": "OneBitLamb",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-6,
            "weight_decay": 0.01,
            "freeze_step": 1000,
            "max_grad_norm": 1.0
        }
    }
}
```

---

## Muon Optimizer (runtime/zero/muon/)

Muon (MomentUm Orthogonalized by Newton-Schulz) is a novel optimizer that orthogonalizes the momentum matrix using the Newton-Schulz iteration, providing improved training dynamics for matrix-shaped parameters.

### Key Characteristics

- **Momentum orthogonalization**: Uses Newton-Schulz iteration to orthogonalize the momentum, preventing correlated updates.
- **Designed for matrix parameters**: Most effective on 2D weight matrices (linear layers, embedding layers).
- **Works with ZeRO**: Integrated into the ZeRO runtime for scalability.
- **Two variants**: `Muon` (standalone) and `MuonWithAuxAdam` (uses Adam for non-matrix parameters).

### Newton-Schulz Iteration

The Newton-Schulz iteration computes an approximate orthogonalization of a matrix:

```
X_0 = M / ||M||_F  (normalize momentum)
X_{k+1} = 0.5 * X_k * (3 * I - X_k^T * X_k)  (Newton-Schulz step)
```

After 5-10 iterations, X converges to the closest orthogonal matrix to M (in the Frobenius norm sense). This is a memory-efficient alternative to SVD-based orthogonalization.

### Muon Algorithm

```
m_t = beta * m_{t-1} + (1 - beta) * g_t

# Orthogonalize momentum (for 2D parameters)
if momentum.ndim == 2:
    m_orth = newton_schulz(m_t, num_iterations=5)
    update = m_orth * (||m_t||_F / ||m_orth||_F)  # Preserve scale
else:
    update = m_t

w_{t+1} = w_t - lr * (update + weight_decay * w_t)
```

### Muon Class

```python
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95,
                 nesterov=True, ns_steps=5, weight_decay=0.0):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum coefficient (beta)
            nesterov: Whether to use Nesterov momentum
            ns_steps: Number of Newton-Schulz iterations
            weight_decay: Weight decay coefficient
        """
```

### MuonWithAuxAdam

Since Muon is designed for 2D (matrix) parameters, `MuonWithAuxAdam` uses standard Adam for 1D parameters (biases, layer norms) and Muon for 2D parameters:

```python
class MuonWithAuxAdam:
    """Uses Muon for 2D parameters and Adam for 1D parameters."""

    def __init__(self, muon_params, adam_params,
                 lr=0.02, momentum=0.95, ns_steps=5,
                 adam_lr=3e-4, adam_betas=(0.9, 0.95),
                 adam_eps=1e-8, weight_decay=0.0):
        self.muon_optimizer = Muon(muon_params, lr=lr,
                                    momentum=momentum, ns_steps=ns_steps)
        self.adam_optimizer = torch.optim.Adam(adam_params,
                                                lr=adam_lr,
                                                betas=adam_betas,
                                                eps=adam_eps,
                                                weight_decay=weight_decay)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.02` | Learning rate (typically larger than Adam's lr) |
| `momentum` | float | `0.95` | Momentum coefficient (beta) |
| `nesterov` | bool | `true` | Whether to use Nesterov momentum |
| `ns_steps` | int | `5` | Number of Newton-Schulz iterations for orthogonalization |
| `weight_decay` | float | `0` | Weight decay coefficient |
| `adam_lr` | float | `3e-4` | Learning rate for aux Adam (MuonWithAuxAdam only) |
| `adam_betas` | tuple | `(0.9, 0.95)` | Betas for aux Adam |

### Configuration Example

```json
{
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": true,
            "ns_steps": 5,
            "weight_decay": 0.0
        }
    }
}
```

---

## MuAdam, MuAdamW, MuSGD (Mu-Variants)

These are momentum-orthogonalized variants of Adam, AdamW, and SGD respectively. They apply the same Newton-Schulz orthogonalization technique from Muon but within the Adam/SGD update framework.

### MuAdam

```python
class MuAdam(torch.optim.Optimizer):
    """Adam with momentum orthogonalization for 2D parameters."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, ns_steps=5):
```

For 2D parameters, the first moment (momentum) is orthogonalized before computing the update. For 1D parameters, standard Adam is used.

### MuAdamW

```python
class MuAdamW(torch.optim.Optimizer):
    """AdamW with momentum orthogonalization for 2D parameters."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01, ns_steps=5):
```

Same as MuAdam but with decoupled weight decay.

### MuSGD

```python
class MuSGD(torch.optim.Optimizer):
    """SGD with momentum orthogonalization for 2D parameters."""

    def __init__(self, params, lr=0.1, momentum=0.9,
                 weight_decay=0, nesterov=True, ns_steps=5):
```

### Configuration Examples

```json
// MuAdam
{
    "optimizer": {
        "type": "MuAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "ns_steps": 5
        }
    }
}

// MuAdamW
{
    "optimizer": {
        "type": "MuAdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "ns_steps": 5
        }
    }
}

// MuSGD
{
    "optimizer": {
        "type": "MuSGD",
        "params": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 0,
            "nesterov": true,
            "ns_steps": 5
        }
    }
}
```

---

## CPU Adagrad (ops/adagrad/cpu_adagrad.py)

CPU Adagrad is an Adagrad implementation optimized for CPU execution, designed for use with parameter offloading in recommendation systems and other sparse gradient scenarios.

### Key Characteristics

- **CPU-optimized**: SIMD-accelerated for efficient CPU computation.
- **Sparse gradient support**: Efficiently handles sparse gradients common in embedding-heavy models.
- **Compatible with ZeRO-Offload**: Works with DeepSpeed's parameter offloading.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.01` | Learning rate |
| `eps` | float | `1e-10` | Epsilon for numerical stability |
| `weight_decay` | float | `0` | Weight decay |
| `initial_accumulator_value` | float | `0` | Initial value for the sum of squared gradients |

### Configuration Example

```json
{
    "optimizer": {
        "type": "Adagrad",
        "params": {
            "lr": 0.01,
            "eps": 1e-10,
            "weight_decay": 0
        }
    }
}
```

---

## torch_adam Option (TORCH_ADAM_PARAM)

DeepSpeed provides the option to use PyTorch's native `torch.optim.Adam` instead of the fused implementation. This is controlled by the `torch_adam` parameter.

### When to Use torch_adam

- **Debugging**: PyTorch's Adam is easier to debug with standard tools.
- **CPU training**: When training on CPU (no GPU available).
- **Compatibility**: When the fused kernel has issues on specific hardware.
- **Custom dtypes**: When using unusual data types not supported by the fused kernel.

### Configuration

```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "torch_adam": true
}
```

The `torch_adam` flag at the top level of the config overrides the default DeepSpeed optimizer with PyTorch's native implementation.

---

## adam_w_mode Default (ADAM_W_MODE)

By default, DeepSpeed uses AdamW mode (`adam_w_mode=True`) for all Adam-based optimizers. This means:

- **Weight decay is decoupled** from the gradient update
- **The update rule follows Loshchilov & Hutter (2019)** rather than the original Adam paper
- **This default is set by the `ADAM_W_MODE` constant**, which defaults to `True`

### Overriding adam_w_mode

To use the original Adam (L2 regularization) instead of AdamW:

```json
{
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "adam_w_mode": false
        }
    }
}
```

---

## Custom Optimizer Integration

DeepSpeed supports using any PyTorch-compatible optimizer as a custom optimizer.

### Method 1: Passing the Optimizer Directly

```python
import torch
import deepspeed

model = MyModel()

# Create any PyTorch optimizer
custom_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Pass to deepspeed.initialize
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=custom_optimizer,
    config=ds_config
)
```

### Method 2: Using a Custom Optimizer Class

```python
class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, defaults={"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(p.grad, alpha=-group['lr'])
        return loss

# Register with DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=MyOptimizer(model.parameters(), lr=0.01),
    config=ds_config
)
```

### Method 3: Using optimizer from a Library

```python
from bitsandbytes.optim import AdamW8bit

# Use 8-bit Adam from bitsandbytes
optimizer = AdamW8bit(model.parameters(), lr=1e-4)

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)
```

### Important Considerations for Custom Optimizers

1. **ZeRO compatibility**: Custom optimizers may not be compatible with ZeRO stages 2 and 3 unless they follow DeepSpeed's parameter sharding conventions.
2. **FP16/BF16 training**: Custom optimizers must handle mixed-precision gradients correctly. DeepSpeed wraps the optimizer with an FP16 optimizer wrapper.
3. **Gradient accumulation**: Custom optimizers should not modify gradients in-place during `step()` if gradient accumulation is used.
4. **State dict**: Custom optimizers should implement `state_dict()` and `load_state_dict()` for checkpoint compatibility.

---

## Optimizer Selection Guide

| Scenario | Recommended Optimizer | Reason |
|---|---|---|
| General training (GPUs available) | FusedAdam / AdamW | Fastest, most widely tested |
| ZeRO-Offload training | CPU Adam | Optimized for CPU execution |
| Limited network bandwidth | 1-Bit Adam / 0/1 Adam | Reduces communication by 10-32x |
| Large batch training | FusedLAMB | Stable with batch sizes > 64K |
| Memory-constrained training | FusedLion | 50% less optimizer state memory |
| Cutting-edge performance | Muon / MuAdamW | Momentum orthogonalization for better convergence |
| Recommendation systems | CPU Adagrad | Efficient for sparse gradients |
| Debugging | torch_adam=True | Easier to debug with standard tools |

---

## Full Configuration Examples

### Example 1: GPT Pretraining with FusedAdam

```json
{
    "train_batch_size": 4096,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### Example 2: BERT Fine-Tuning with ZeRO-Offload

```json
{
    "train_batch_size": 128,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

### Example 3: Large-Scale Training with 1-Bit Adam

```json
{
    "train_batch_size": 16384,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 2000
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    }
}
```

### Example 4: LLM Training with Muon

```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 4,
    "optimizer": {
        "type": "Muon",
        "params": {
            "lr": 0.02,
            "momentum": 0.95,
            "nesterov": true,
            "ns_steps": 5,
            "weight_decay": 0.0
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    }
}
```

### Example 5: Large Batch with LAMB

```json
{
    "train_batch_size": 65536,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "LAMB",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-6,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### Example 6: Memory-Optimized with Lion + ZeRO-3

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 2,
    "optimizer": {
        "type": "Lion",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.99],
            "weight_decay": 0.01
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```
