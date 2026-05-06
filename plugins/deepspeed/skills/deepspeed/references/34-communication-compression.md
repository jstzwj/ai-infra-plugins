# Communication Compression

## Overview

Communication compression is a family of techniques that reduce the volume of data transmitted between GPUs during distributed training. In large-scale training with thousands of GPUs, gradient and parameter communication can become a dominant bottleneck, consuming significant time and network bandwidth. DeepSpeed provides multiple compression algorithms -- 1-bit Adam, 0/1 Adam, 1-bit LAMB, and ZeRO++ quantization -- that compress gradients and parameters to a fraction of their original size while maintaining model quality.

These algorithms address different aspects of the communication bottleneck:
- **1-bit Adam** compresses gradient communication for Adam-based training to 1 bit per value (vs. 32 bits for FP32 or 16 bits for FP16), achieving up to 26x communication volume reduction.
- **0/1 Adam** extends 1-bit Adam with dynamic communication skipping, further reducing communication frequency.
- **1-bit LAMB** applies 1-bit compression to the LAMB optimizer for layer-wise adaptive scaling.
- **ZeRO++ (HPZ, QGZ, QQZ)** uses quantization-based compression within the ZeRO framework for weights and gradients.

---

## Source Code Organization

```
deepspeed/runtime/fp16/onebit/
    __init__.py
    onebit_adam.py               # 1-bit Adam optimizer implementation
    onebit_lamb.py               # 1-bit LAMB optimizer implementation

deepspeed/runtime/zero/
    zero_quantization.py         # ZeRO quantization utilities
    zero_quantized_weights.py    # Weight quantization for ZeRO++
    zero_quantized_gradients.py  # Gradient quantization for ZeRO++

deepspeed/csrc/
    onebit/adam.cpp              # C++ kernel for 1-bit Adam operations
```

---

## 1-Bit Adam

### Overview

1-bit Adam is a communication-compressed variant of the Adam optimizer that reduces gradient communication volume by compressing each gradient value to just 1 bit. The key insight is that the sign of the gradient (positive or negative) captures most of the information needed for the Adam update, especially in later stages of training when the optimizer has accumulated accurate momentum and variance estimates.

### Algorithm

The 1-bit Adam algorithm operates in two phases:

**Phase 1: Warmup**
- Standard Adam training with full-precision (FP16 or FP32) gradients.
- All-reduce communicates full gradients during warmup.
- The warmup phase allows the optimizer to build up accurate momentum and variance estimates.
- Typical warmup duration: 10-20% of total training steps.

**Phase 2: 1-bit Compression**
- Gradients are compressed to 1 bit per value (sign only).
- The compressed gradients are communicated via all-reduce (sign all-reduce).
- A residual accumulation mechanism tracks the error introduced by compression.
- Momentum correction ensures the Adam update direction remains accurate despite compression.

```
# 1-bit Adam update rule (simplified):
# After warmup:
#   compressed_grad = sign(grad + residual)
#   residual = grad + residual - compressed_grad * scale_factor
#   momentum = beta1 * momentum + (1 - beta1) * compressed_grad
#   variance = beta2 * variance + (1 - beta2) * compressed_grad^2
#   param_update = momentum / (sqrt(variance) + eps)
#   param = param - lr * param_update
```

### Configuration

```json
{
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 200
        }
    },
    "gradient_accumulation_steps": 1,
    "train_micro_batch_size_per_gpu": 16,
    "fp16": {
        "enabled": true
    }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be `"OneBitAdam"`. |
| `params.lr` | float | - | Learning rate. |
| `params.betas` | list[float] | `[0.9, 0.999]` | Adam beta1 (momentum) and beta2 (variance) coefficients. |
| `params.eps` | float | `1e-8` | Epsilon for numerical stability. |
| `params.weight_decay` | float | `0` | Weight decay (L2 regularization). |
| `params.freeze_step` | int | `100` | Number of warmup steps with full-precision gradients before switching to 1-bit compression. Must be large enough for momentum/variance to stabilize. |

### Usage

```python
import deepspeed
import torch

# Model and data setup
model = MyModel()
train_dataset = MyDataset()

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 200
        }
    },
    "fp16": {
        "enabled": true
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    training_data=train_dataset,
)

# Standard training loop
for epoch in range(num_epochs):
    for batch in model_engine:
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
```

### Performance Characteristics

| Metric | Standard Adam | 1-Bit Adam |
|--------|--------------|------------|
| Communication volume per step | 2 * M * sizeof(fp16) | 2 * M * 1 bit |
| Compression ratio | 1x | 16x (per value) |
| Effective reduction (with residual) | 1x | ~26x (end-to-end) |
| Final model quality | Baseline | Within 0.1-0.5% of baseline |
| Warmup overhead | None | 10-20% of total steps |

*M = number of parameters communicated per all-reduce.*

---

## 0/1 Adam

### Overview

0/1 Adam extends 1-bit Adam with a dynamic communication mechanism that skips gradient communication entirely on some steps (the "0" in 0/1). This is based on the observation that gradients change slowly across consecutive steps in later stages of training, making frequent communication unnecessary.

### Algorithm

0/1 Adam introduces a dynamic communication schedule:
- **0-bit steps**: No communication occurs. Each GPU uses its local gradient for the Adam update.
- **1-bit steps**: Compressed gradient communication (same as 1-bit Adam).
- The transition between 0-bit and 1-bit steps is controlled by a schedule based on gradient similarity across steps.

```
# 0/1 Adam decision rule:
# At each step:
#   similarity = cosine_similarity(current_grad, previous_grad)
#   if similarity > threshold:
#       0-bit: skip communication, use local gradient
#   else:
#       1-bit: compress and communicate gradient
```

### Configuration

```json
{
    "optimizer": {
        "type": "ZeroOneAdam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 200,
            "comm_batch_name": "cosine_similarity",
            "comm_batch_size": 1,
            "overlap_comm": true,
            "cxpb": 0.5
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### Additional Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comm_batch_name` | str | `"cosine_similarity"` | Similarity metric for communication decision. `"cosine_similarity"`: compare current and previous gradients using cosine similarity. |
| `comm_batch_size` | int | `1` | Number of steps to batch before making communication decision. |
| `overlap_comm` | bool | `true` | Overlap 1-bit communication with computation on 0-bit steps. |
| `cxpb` | float | `0.5` | Communication probability after warmup. Controls the ratio of 1-bit vs 0-bit steps. Lower values reduce communication more aggressively. |

### Performance Characteristics

| Metric | 1-Bit Adam | 0/1 Adam |
|--------|-----------|----------|
| Communication volume (typical) | 1/16 of FP16 | 1/32 of FP16 (with 50% skip) |
| Communication frequency | Every step | Dynamic (30-70% of steps) |
| Final model quality | Within 0.1-0.5% | Within 0.1-0.5% |
| Best for | Communication-bound training | Very large clusters with high latency |

---

## 1-Bit LAMB

### Overview

1-bit LAMB applies 1-bit gradient compression to the LAMB optimizer. LAMB (Layer-wise Adaptive Moments) is designed for large-batch training and provides layer-wise adaptive scaling, which enables stable training with very large batch sizes (millions of samples). 1-bit LAMB combines LAMB's large-batch capability with communication compression for large-scale distributed training.

### Algorithm

1-bit LAMB follows the same two-phase structure as 1-bit Adam (warmup with full precision, then 1-bit compression), but uses the LAMB update rule instead of Adam:

```
# LAMB update rule (per layer):
#   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t           # momentum
#   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         # variance
#   m_hat = m_t / (1 - beta1^t)                          # bias correction
#   v_hat = v_t / (1 - beta2^t)                          # bias correction
#   update = m_hat / (sqrt(v_hat) + eps)
#   trust_ratio = ||w|| / ||update||                     # layer-wise scaling
#   w_{t+1} = w_t - lr * trust_ratio * update
```

### Configuration

```json
{
    "optimizer": {
        "type": "OneBitLamb",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 200,
            "clip_grad": 1.0,
            "max_coeff": 0.3,
            "min_coeff": 0.01
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be `"OneBitLamb"`. |
| `params.lr` | float | - | Learning rate. |
| `params.betas` | list[float] | `[0.9, 0.999]` | LAMB beta1 and beta2 coefficients. |
| `params.eps` | float | `1e-8` | Epsilon for numerical stability. |
| `params.weight_decay` | float | `0` | Weight decay. |
| `params.freeze_step` | int | `100` | Warmup steps with full-precision communication. |
| `params.clip_grad` | float | `1.0` | Gradient clipping norm. |
| `params.max_coeff` | float | `0.3` | Maximum LAMB trust ratio coefficient. |
| `params.min_coeff` | float | `0.01` | Minimum LAMB trust ratio coefficient. |

### Usage

```python
import deepspeed

ds_config = {
    "train_batch_size": 65536,  # Very large batch for LAMB
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "OneBitLamb",
        "params": {
            "lr": 2e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 500,
            "clip_grad": 1.0
        }
    },
    "fp16": {
        "enabled": true
    }
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    training_data=train_dataset,
)
```

---

## ZeRO++ Quantization

### Overview

ZeRO++ is an extension to the ZeRO optimizer that adds quantization-based compression for both weights and gradients. Unlike 1-bit Adam (which compresses gradients at the optimizer level), ZeRO++ operates within the ZeRO communication framework, compressing the all-gather (weight), reduce-scatter (gradient), and all-to-all (partition) operations.

ZeRO++ introduces three complementary quantization techniques:
1. **HPZ (Hierarchical Partitioned ZeRO++)**: Quantized weight all-gather
2. **QGZ (Quantized Gradient ZeRO++)**: Quantized gradient reduce-scatter
3. **QQZ (Quantized Quantized ZeRO++)**: Combined weight and gradient quantization

### Weight Quantization (zero_quantized_weights)

Weight quantization compresses the parameter tensors exchanged during ZeRO Stage 3's all-gather operations. When a layer needs its full parameters for computation, instead of communicating FP16 weights, ZeRO++ communicates quantized weights and dequantizes on the receiving end.

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_weights": true,
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zero_quantized_weights` | bool | `false` | Enable weight quantization for ZeRO Stage 3 all-gather operations. Reduces communication volume by 4x (FP16 -> 4-bit). |
| `zero_hpz_partition_size` | int | `1` | Number of GPUs in each hierarchical partition group for HPZ quantization. Larger groups reduce inter-node communication at the cost of more intra-node communication. |

### Gradient Quantization (zero_quantized_gradients)

Gradient quantization compresses the gradient tensors exchanged during ZeRO's reduce-scatter operations. Gradients are quantized before communication and dequantized after reception.

```json
{
    "zero_optimization": {
        "stage": 2,
        "zero_quantized_gradients": {
            "enabled": true,
            "num_bits": 8
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zero_quantized_gradients` | dict/bool | `false` | Enable gradient quantization. Can be `true` for defaults or a dict with options. |
| `zero_quantized_gradients.enabled` | bool | `false` | Explicit enable/disable. |
| `zero_quantized_gradients.num_bits` | int | `8` | Number of bits for gradient quantization. Options: 4, 8. Lower values give more compression but may hurt quality. |

### Combined ZeRO++ Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": {
            "enabled": true,
            "num_bits": 8
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8
    }
}
```

---

## LoCo-ZeRO++

### Overview

LoCo-ZeRO++ (Low-Communication ZeRO++) is an enhancement over ZeRO++ that further reduces communication overhead using error-compensated compression. LoCo maintains an error feedback mechanism that accumulates quantization errors across steps, ensuring that the compression does not introduce bias into the training process.

### Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "zeropp_loco_param": {
            "err_beta": 0.8,
            "reset_T": 1024
        },
        "zero_quantized_weights": true,
        "zero_quantized_gradients": {
            "enabled": true,
            "num_bits": 8
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

### LoCo Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zeropp_loco_param` | dict | `null` | LoCo configuration. When present, enables LoCo error-compensated compression. |
| `zeropp_loco_param.err_beta` | float | `0.8` | Error feedback decay factor. Controls how much of the accumulated quantization error is carried forward to the next step. Values close to 1.0 preserve more error history (more accurate but slower convergence). Values close to 0.0 discard error history quickly (less accurate but more responsive). |
| `zeropp_loco_param.reset_T` | int | `1024` | Number of steps between error feedback resets. Periodic resets prevent error accumulation from growing unbounded. Setting to 0 disables resets. |

### How LoCo Works

```
# LoCo error-compensated quantization:
# At step t:
#   error_buffer_t = err_beta * error_buffer_{t-1} + (1 - err_beta) * quantization_error_t
#   compensated_grad = grad + error_buffer_t
#   quantized_grad = quantize(compensated_grad)
#   quantization_error_t = compensated_grad - dequantize(quantized_grad)
#
# Every reset_T steps:
#   error_buffer = 0  # Reset to prevent unbounded growth
```

The error feedback mechanism ensures that the quantization error from one step is used to correct the quantization in the next step, so errors do not accumulate systematically. The `err_beta` parameter controls the exponential moving average of the error, and `reset_T` provides periodic cleanup.

---

## Compressed Gradient Algorithm Comparison

### When to Use Each Method

| Method | Best For | Compression Ratio | Quality Impact | Communication Pattern |
|--------|----------|-------------------|----------------|----------------------|
| **1-bit Adam** | Adam-based training with moderate cluster size (8-64 GPUs) | ~26x end-to-end | Within 0.1-0.5% of baseline | All-reduce (sign) |
| **0/1 Adam** | Very large clusters (64+ GPUs) with high inter-node latency | ~26-52x (with skipping) | Within 0.1-0.5% | Dynamic (skip or sign all-reduce) |
| **1-bit LAMB** | Very large batch training (millions of samples) | ~26x | Within 0.1-0.5% | All-reduce (sign) |
| **ZeRO++ weights** | ZeRO Stage 3 with bandwidth-limited inter-node communication | 4x (FP16 to 4-bit) | Minimal | All-gather (quantized) |
| **ZeRO++ gradients** | ZeRO Stage 2/3 with gradient communication bottleneck | 2x (FP16 to 8-bit) or 4x (to 4-bit) | Minimal with 8-bit | Reduce-scatter (quantized) |
| **LoCo-ZeRO++** | ZeRO++ scenarios where quality is critical | 2-4x | Near-zero (error compensated) | Same as ZeRO++ |

### Decision Flowchart

```
Is your training communication-bound?
├── Yes
│   ├── Using ZeRO Stage 2 or 3?
│   │   ├── Yes → Use ZeRO++ (zero_quantized_weights + zero_quantized_gradients)
│   │   │        └── Quality critical? → Add LoCo-ZeRO++
│   │   └── No (Standard DDP or Stage 1)
│   │       ├── Using Adam? → Use 1-bit Adam
│   │       │   └── Very large cluster (64+ GPUs)? → Consider 0/1 Adam
│   │       └── Using LAMB? → Use 1-bit LAMB
│   └── No
│       └── Consider other optimizations (gradient accumulation, overlap, etc.)
```

---

## Configuration Examples

### Example 1: 1-Bit Adam for BERT Pre-training

```json
{
    "train_batch_size": 4096,
    "train_micro_batch_size_per_gpu": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "OneBitAdam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "freeze_step": 500
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 12
    },
    "gradient_clipping": 1.0
}
```

### Example 2: 0/1 Adam for Large-Scale GPT Training

```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "ZeroOneAdam",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
            "freeze_step": 1000,
            "overlap_comm": true,
            "cxpb": 0.3
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 0
    },
    "gradient_clipping": 1.0
}
```

### Example 3: 1-Bit LAMB for Large-Batch ViT Training

```json
{
    "train_batch_size": 32768,
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "OneBitLamb",
        "params": {
            "lr": 5e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05,
            "freeze_step": 200,
            "clip_grad": 1.0
        }
    },
    "fp16": {
        "enabled": true
    },
    "gradient_clipping": 1.0
}
```

### Example 4: ZeRO++ with Weight and Gradient Quantization

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": {
            "enabled": true,
            "num_bits": 8
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0
}
```

### Example 5: Full LoCo-ZeRO++ Configuration

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": {
            "enabled": true,
            "num_bits": 8
        },
        "zeropp_loco_param": {
            "err_beta": 0.8,
            "reset_T": 1024
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0
}
```

---

## Performance Benchmarks

### 1-Bit Adam Performance

Training BERT-Large on 64 GPUs:

| Method | Throughput (seq/s) | Communication Time (%) | Final Accuracy |
|--------|-------------------|----------------------|----------------|
| Standard Adam (FP16) | 4,200 | 45% | 81.2% |
| 1-Bit Adam | 5,800 | 12% | 80.9% |
| Speedup | 1.38x | - | -0.3% |

### ZeRO++ Performance

Training a 10B parameter model on 64 GPUs:

| Method | Throughput (samples/s) | Comm Volume (GB/step) | Final Loss |
|--------|----------------------|----------------------|------------|
| ZeRO-3 (FP16) | 1,250 | 80 | 2.34 |
| ZeRO++ (8-bit) | 1,800 | 40 | 2.36 |
| ZeRO++ + LoCo | 1,750 | 40 | 2.34 |
| Speedup | 1.44x | 2x | ~0% |

### 0/1 Adam Performance

Training GPT-3 6.7B on 128 GPUs:

| Method | Throughput (tokens/s) | Comm Steps (%) | Final Perplexity |
|--------|----------------------|----------------|-----------------|
| Standard Adam | 85,000 | 100% | 14.2 |
| 1-Bit Adam | 110,000 | 100% | 14.4 |
| 0/1 Adam (cxpb=0.5) | 135,000 | 50% | 14.3 |
| Speedup | 1.59x | - | ~0% |

---

## Implementation Details

### 1-Bit Compression Kernel

The core 1-bit compression operation:

```python
# Simplified 1-bit gradient compression
def compress_gradient_1bit(gradient):
    """Compress gradient to 1 bit per value.

    Returns:
        signs: Tensor of +1/-1 (the compressed gradient)
        scale: Scalar representing the mean absolute value
    """
    scale = gradient.abs().mean()
    signs = torch.sign(gradient)
    return signs, scale

def decompress_gradient_1bit(signs, scale):
    """Decompress 1-bit gradient.

    Returns:
        gradient: Reconstructed gradient tensor
    """
    return signs * scale
```

### Residual Error Compensation

```python
# Residual accumulation for error compensation
class ResidualAccumulator:
    def __init__(self, shape, device):
        self.residual = torch.zeros(shape, device=device, dtype=torch.float32)

    def compress(self, gradient):
        """Compress gradient with residual compensation."""
        # Add residual to gradient before compression
        compensated = gradient + self.residual

        # Compress
        scale = compensated.abs().mean()
        signs = torch.sign(compensated)
        compressed = signs * scale

        # Update residual (error introduced by compression)
        self.residual = compensated - compressed

        return signs, scale
```

### Quantization for ZeRO++

```python
# INT8 quantization for ZeRO++ gradients
def quantize_tensor_fp8(tensor, num_bits=8):
    """Quantize a tensor to fixed-point representation.

    Args:
        tensor: Input tensor (FP16 or BF16).
        num_bits: Number of bits for quantization (4 or 8).

    Returns:
        quantized: Quantized tensor (uint8 for 8-bit, uint4 for 4-bit).
        scale: Per-tensor scale factor.
        zero_point: Per-tensor zero point.
    """
    # Compute scale and zero point
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    qmin = 0
    qmax = (1 << num_bits) - 1

    scale = (tensor_max - tensor_min) / (qmax - qmin)
    zero_point = qmin - tensor_min / scale

    # Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale + zero_point),
        qmin, qmax
    ).to(torch.uint8)

    return quantized, scale, zero_point

def dequantize_tensor(quantized, scale, zero_point):
    """Dequantize a tensor back to floating point."""
    return (quantized.float() - zero_point) * scale
```

---

## Troubleshooting

### Common Issues

1. **Model quality degradation with 1-bit Adam**: Increase `freeze_step` to give the warmup phase more time to build accurate momentum/variance. Typical values: 200-2000 steps depending on model size.

2. **0/1 Adam not skipping communication**: The `cxpb` parameter may be too high. Reduce to 0.2-0.3 for more aggressive skipping. Also ensure `overlap_comm` is enabled.

3. **ZeRO++ quantization errors with Stage 3**: Ensure `overlap_comm` is enabled. Quantization errors can accumulate with overlapping operations if buffers are not managed correctly.

4. **LoCo error feedback growing too large**: Reduce `err_beta` (e.g., from 0.8 to 0.5) to decay the error faster. Also reduce `reset_T` for more frequent resets.

5. **"OneBitAdam requires fp16"**: All 1-bit compression methods require mixed-precision training. Enable `"fp16": {"enabled": true}` or `"bf16": {"enabled": true}` in the configuration.

6. **Compilation errors for 1-bit kernels**: Build with `DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext"`.

### Compatibility Matrix

| Method | ZeRO Stage 0 | ZeRO Stage 1 | ZeRO Stage 2 | ZeRO Stage 3 | FP16 | BF16 | FP32 |
|--------|-------------|-------------|-------------|-------------|------|------|------|
| 1-bit Adam | Yes | Yes | No | No | Yes | Yes | No |
| 0/1 Adam | Yes | Yes | No | No | Yes | Yes | No |
| 1-bit LAMB | Yes | Yes | No | No | Yes | Yes | No |
| ZeRO++ weights | No | No | No | Yes | Yes | Yes | No |
| ZeRO++ gradients | No | No | Yes | Yes | Yes | Yes | No |
| LoCo-ZeRO++ | No | No | No | Yes | Yes | Yes | No |

Note: 1-bit Adam/0/1 Adam/1-bit LAMB are not compatible with ZeRO Stage 2/3 because they replace the gradient all-reduce with their own compressed communication, which conflicts with ZeRO's reduce-scatter. ZeRO++ quantization operates within the ZeRO framework and is compatible with Stage 2/3.

---

## Summary

DeepSpeed's communication compression suite provides multiple complementary algorithms for reducing the communication overhead in distributed training. 1-bit Adam, 0/1 Adam, and 1-bit LAMB operate at the optimizer level, compressing gradients to 1 bit per value and achieving ~26x communication volume reduction with minimal quality impact. ZeRO++ operates within the ZeRO framework, using INT4/INT8 quantization to compress weight all-gather and gradient reduce-scatter operations by 2-4x. LoCo-ZeRO++ adds error-compensated compression to ZeRO++ for near-zero quality degradation. The choice of algorithm depends on the optimizer being used (Adam vs LAMB), the ZeRO stage, the cluster size, and the acceptable quality-communication trade-off.
