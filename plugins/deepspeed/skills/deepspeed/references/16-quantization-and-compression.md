# DeepSpeed Quantization and Compression

## Overview

DeepSpeed provides a comprehensive compression toolkit that enables training-aware model compression through quantization, pruning, and knowledge distillation. The compression framework is located in `deepspeed/compression/` and supports a wide range of techniques including weight quantization, activation quantization, sparse pruning, structured pruning (row, column, head, channel), and layer reduction via knowledge distillation.

DeepSpeed also supports inference-time quantization through specialized modules such as FP8 quantization (`ops/fp_quantizer/`), INT4/INT8 quantization, and the ZeroQuant family of quantization methods.

---

## Compression Module Architecture

### Directory Structure

```
deepspeed/compression/
    __init__.py
    config.py           # CompressionConfig class and sub-configs
    compress.py          # CompressionScheduler and compression manager
    basic_layer.py       # Compressed linear/conv layers with quantization/pruning
    scheduler.py         # Compression scheduling logic
```

### config.py -- CompressionConfig

The `CompressionConfig` class is the central configuration object. It is instantiated from the `compression_training` key in the DeepSpeed JSON configuration. It aggregates sub-configurations for each compression technique:

```python
class CompressionConfig:
    def __init__(self, compression_config_dict):
        # Sub-configs
        self.layer_reduction = LayerReductionConfig(...)
        self.weight_quantization = WeightQuantizationConfig(...)
        self.activation_quantization = ActivationQuantizationConfig(...)
        self.sparse_pruning = SparsePruningConfig(...)
        self.row_pruning = RowPruningConfig(...)
        self.head_pruning = HeadPruningConfig(...)
        self.channel_pruning = ChannelPruningConfig(...)
```

#### Sub-Configuration Classes

Each technique has its own config class with shared parameters and per-group overrides:

| Class | Key in Config | Purpose |
|---|---|---|
| `LayerReductionConfig` | `layer_reduction` | Knowledge distillation with fewer layers |
| `WeightQuantizationConfig` | `weight_quantization` | Quantize model weights |
| `ActivationQuantizationConfig` | `activation_quantization` | Quantize intermediate activations |
| `SparsePruningConfig` | `sparse_pruning` | Unstructured sparse pruning |
| `RowPruningConfig` | `row_pruning` | Structured row-level pruning |
| `HeadPruningConfig` | `head_pruning` | Attention head pruning |
| `ChannelPruningConfig` | `channel_pruning` | Channel-level pruning |

### compress.py -- CompressionScheduler

The `CompressionScheduler` manages the lifecycle of compression during training:

```python
class CompressionScheduler:
    def __init__(self, model, compression_config):
        """Initialize all compression sub-schedulers."""

    def step(self, step_index):
        """Called every training step; adjusts quantization bits, pruning ratios, etc."""

    def compression_norm(self):
        """Returns the compression ratio / norm metric."""

    def is_step_in_schedule(self, step_index):
        """Check if the current step is within the compression schedule."""
```

The scheduler reads the configuration and, at each training step, adjusts:
- Quantization bit-widths (from `start_bits` toward `target_bits`)
- Pruning ratios (increasing sparsity over time)
- Layer reduction schedules (which layers to keep)

### basic_layer.py -- Compressed Layers

DeepSpeed replaces standard `nn.Linear` (and optionally `nn.Conv`) layers with compressed variants that implement quantization and pruning inline:

```python
class CompressedLinear(nn.Module):
    """Linear layer with integrated weight quantization and pruning support."""

    def __init__(self, original_linear, weight_quantization, pruning_config, ...):
        super().__init__()
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        # Quantization and pruning state
        self.weight_quantizer = ...  # Weight quantizer kernel
        self.pruning_mask = ...      # Binary or ternary pruning mask

    def forward(self, input):
        # Optionally quantize activations
        if self.activation_quantization_enabled:
            input = self.activation_quantizer(input)
        # Optionally quantize weights
        if self.weight_quantization_enabled:
            weight = self.weight_quantizer(self.weight)
        # Apply pruning mask
        if self.pruning_enabled:
            weight = weight * self.pruning_mask
        return torch.nn.functional.linear(input, weight, self.bias)
```

### scheduler.py -- Compression Schedule Logic

The scheduler module handles time-based compression transitions. Key concepts:

- **schedule_offset**: The global step at which a compression technique begins.
- **quantization_period**: How frequently the quantization bits are reduced during the transition.
- **Transition formula**: Bit-width decreases from `start_bits` to `target_bits` over a scheduled number of steps.

---

## Weight Quantization

Weight quantization reduces the precision of model weights from FP32/FP16 to lower bit-width representations (e.g., INT8, INT4, or even binary/ternary).

### Configuration Structure

Weight quantization is configured under the `weight_quantization` key:

```json
{
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantizer_kernel": false,
                "schedule_offset": 0,
                "quantize_groups": 1,
                "quantization_type": "symmetric",
                "rounding": "nearest",
                "quantize_weight_in_forward": true,
                "fp16_mixed_quantize": false
            },
            "different_groups": {
                "wq_group_1": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 4,
                        "quantization_period": 1000
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### shared_parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for weight quantization |
| `quantizer_kernel` | bool | `false` | Whether to use a custom CUDA kernel for quantization (faster but less flexible) |
| `schedule_offset` | int | `0` | Global step at which to begin weight quantization |
| `quantize_groups` | int | `1` | Number of groups for grouped quantization. Group size = `weight_dim / quantize_groups` |
| `quantization_type` | str | `"symmetric"` | `"symmetric"` (zero-point = 0) or `"asymmetric"` (learned zero-point) |
| `rounding` | str | `"nearest"` | `"nearest"` (round-to-nearest) or `"stochastic"` (stochastic rounding) |
| `quantize_weight_in_forward` | bool | `true` | If `true`, quantize weights during the forward pass (simulates inference quantization). If `false`, quantize once and cache. |
| `fp16_mixed_quantize` | bool | `false` | Enables mixed FP16 + INT quantization for gradient stability |

### different_groups

The `different_groups` dictionary allows specifying different quantization schedules for different module groups. Each group has:

- **`params`**: Group-specific overrides for quantization parameters.
- **`modules`**: List of module name substrings to match. Layers whose names contain any of these strings will use this group's parameters.

#### Group Parameters

| Parameter | Type | Description |
|---|---|---|
| `start_bits` | int | Initial bit-width at the start of the schedule |
| `target_bits` | int | Final bit-width after the transition completes |
| `quantization_period` | int | Number of steps between bit-width reductions |

#### Bit-Width Transition

The transition from `start_bits` to `target_bits` occurs linearly:

```
current_bits = start_bits - floor((step - schedule_offset) / quantization_period)
current_bits = max(current_bits, target_bits)
```

For example, with `start_bits=8`, `target_bits=4`, `quantization_period=1000`, and `schedule_offset=0`:
- Steps 0-999: 8 bits
- Steps 1000-1999: 7 bits
- Steps 2000-2999: 6 bits
- Steps 3000-3999: 5 bits
- Steps 4000+: 4 bits

### Quantization Types

#### Symmetric Quantization

```
scale = max(|weight|) / (2^(bits-1) - 1)
quantized = round(weight / scale)
dequantized = quantized * scale
```

The range is `[-2^(bits-1) + 1, 2^(bits-1) - 1]` with zero-point fixed at 0.

#### Asymmetric Quantization

```
w_min = min(weight)
w_max = max(weight)
scale = (w_max - w_min) / (2^bits - 1)
zero_point = round(-w_min / scale)
quantized = round(weight / scale) + zero_point
dequantized = (quantized - zero_point) * scale
```

### Quantization Groups

When `quantize_groups > 1`, weights are divided into groups along the output dimension, and each group has its own scale factor. This improves accuracy for weights with non-uniform distributions:

```python
# Example: quantize_groups=4 for a [4096, 4096] weight
# Each group covers 1024 output rows
# scale shape: [4, 1] instead of [1, 1]
```

### fp16_mixed_quantize

When enabled, the quantized weight is mixed with the original FP16 weight:

```
mixed_weight = alpha * quantized_weight + (1 - alpha) * fp16_weight
```

This provides a smooth transition from FP16 training to fully quantized training and helps maintain training stability during the early phases of quantization.

---

## Activation Quantization

Activation quantization reduces the precision of intermediate activations (inputs and outputs of layers) during training. This is critical for quantization-aware training (QAT) because it ensures the model learns to be robust to quantization noise in both weights and activations.

### Configuration Structure

```json
{
    "compression_training": {
        "activation_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "symmetric",
                "range_calibration": "dynamic",
                "schedule_offset": 0
            },
            "different_groups": {
                "aq_group_1": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1000
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### shared_parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for activation quantization |
| `quantization_type` | str | `"symmetric"` | `"symmetric"` or `"asymmetric"` |
| `range_calibration` | str | `"dynamic"` | `"dynamic"` (per-tensor min/max) or `"static"` (learned range) |
| `schedule_offset` | int | `0` | Global step at which to begin activation quantization |

### Range Calibration

#### Dynamic Range Calibration

The min/max range is computed dynamically for each activation tensor during each forward pass:

```python
# Per-forward-pass
act_min = activation.min()
act_max = activation.max()
scale = (act_max - act_min) / (2^bits - 1)
```

This is the default mode and works well for most cases. It adapts to the actual distribution of activations.

#### Static Range Calibration

The range is learned during training using a running average or explicit learnable parameters:

```python
# Learnable range (pseudo-code)
self.act_scale = nn.Parameter(initial_scale)
quantized = round(activation / self.act_scale)
```

### Activation Quantization in the Forward Pass

When enabled, activation quantization inserts fake-quantization nodes at the inputs and/or outputs of designated layers:

```python
def forward(self, hidden_states):
    # Quantize input activations
    if self.aq_enabled:
        hidden_states = fake_quantize(hidden_states, bits=self.aq_bits)
    # Normal layer computation
    output = self.linear(hidden_states)
    return output
```

The `fake_quantize` operation:
1. Computes the quantization scale from the activation range
2. Quantizes to discrete levels
3. Dequantizes back to FP16/FP32 (simulating quantization noise)

This is differentiable through the Straight-Through Estimator (STE), which passes gradients through as if the quantization were the identity function.

---

## MoQ (Model Quantization) -- ZeroQuant Family

DeepSpeed includes the ZeroQuant family of quantization methods for post-training quantization (PTQ) and quantization-aware training (QAT).

### ZeroQuant

ZeroQuant is an efficient post-training quantization method that achieves INT8 weight and activation quantization with minimal accuracy loss. Key features:

- **Group-wise weight quantization**: Weights are quantized in small groups for better granularity
- **Token-wise activation quantization**: Activations are quantized per-token to handle dynamic ranges
- **Iterative knowledge distillation**: Uses the original FP32 model as a teacher to recover accuracy

```python
# ZeroQuant usage
from deepspeed.compression.compress import compression_scheduler

# Configuration
zeroquant_config = {
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "symmetric",
                "quantize_weight_in_forward": true
            },
            "different_groups": {
                "wq": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1
                    },
                    "modules": ["attention", "mlp", "embed"]
                }
            }
        },
        "activation_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "symmetric",
                "range_calibration": "dynamic"
            },
            "different_groups": {
                "aq": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### ZeroQuant-V2

ZeroQuant-V2 extends ZeroQuant with improved quantization algorithms:

- **Automatic precision assignment**: Different layers can use different bit-widths based on sensitivity analysis
- **Improved calibration**: Better min/max estimation for activation ranges
- **Group-wise quantization for both weights and activations**: More flexible grouping strategies
- **Support for 4-bit and mixed-precision quantization**

### ZeroQuant-FP

ZeroQuant-FP uses floating-point quantization (FP8, FP6, FP4) instead of integer quantization:

- **FP8 (E4M3/E5M2)**: Uses IEEE 754-style floating-point with 4 exponent bits and 3/2 mantissa bits
- **FP6**: 6-bit floating-point for better dynamic range than INT6
- **FP4**: 4-bit floating-point for maximum compression

```json
{
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "fp",
                "quantize_weight_in_forward": true
            },
            "different_groups": {
                "wq_fp": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### ZeroQuant-HERO

ZeroQuant-HERO (Highly Efficient Runtime Optimization) focuses on hardware-efficient quantization:

- **Hardware-aware quantization**: Considers specific GPU architectures (Tensor Cores) for optimal quantization strategies
- **Mixed-precision search**: Automatically finds the best bit-width combination across layers
- **Runtime-optimized kernels**: Custom CUDA kernels for fast quantized inference

### ZeroQuant(4+2)

ZeroQuant(4+2) is a specific mixed-precision scheme:

- **4-bit weights**: Most layers use 4-bit weight quantization
- **2-bit activations**: Sensitive layers use 2-bit activation quantization
- **Sensitivity-based assignment**: Layers are profiled to determine the best configuration
- **Accuracy-aware search**: Automatically searches for the Pareto-optimal compression-accuracy trade-off

### FP6-LLM

FP6-LLM provides 6-bit floating-point quantization specifically optimized for large language models:

- **FP6 format**: 1 sign bit, 3 exponent bits, 2 mantissa bits
- **LLM-optimized**: Handles the specific distribution patterns in transformer-based LLMs
- **Efficient inference**: Custom kernels for FP6 matrix multiplication
- **Kernel fusion**: Fuses dequantization with GEMM for minimal overhead

---

## Sparse Pruning

Sparse pruning removes individual weight elements (unstructured sparsity) by setting them to zero. This reduces the effective number of parameters without changing the tensor shapes.

### Configuration

```json
{
    "compression_training": {
        "sparse_pruning": {
            "shared_parameters": {
                "enabled": true,
                "schedule_offset": 0,
                "method": "l1"
            },
            "different_groups": {
                "sp_group_1": {
                    "params": {
                        "dense_ratio": 0.5
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch for sparse pruning |
| `method` | str | `"l1"` | Pruning criterion. Currently supports `"l1"` (L1-norm magnitude pruning) |
| `schedule_offset` | int | `0` | Global step at which to begin pruning |
| `dense_ratio` | float | `1.0` | Fraction of weights to retain (0.0-1.0). 0.5 means 50% sparsity. |

### L1 Magnitude Pruning

The L1 method prunes weights with the smallest absolute magnitudes:

```python
def l1_prune(weight, dense_ratio):
    """Prune smallest-magnitude weights."""
    flat = weight.abs().flatten()
    threshold_idx = int(len(flat) * (1 - dense_ratio))
    threshold = torch.kthvalue(flat, threshold_idx).values
    mask = (weight.abs() >= threshold).float()
    return weight * mask, mask
```

### Pruning Schedule

Sparse pruning can be scheduled to gradually increase sparsity:

1. **One-shot pruning**: Prune to the target `dense_ratio` at `schedule_offset` and fine-tune.
2. **Gradual pruning** (via `different_groups` with varying `dense_ratio`): Iteratively increase sparsity.

### Gradient Masking

During training, gradients for pruned (zeroed) positions are masked out to prevent those weights from being updated:

```python
# Gradient masking in backward pass
weight.grad = weight.grad * pruning_mask
```

This ensures that pruned weights remain at zero and only the surviving weights are updated.

---

## Row Pruning

Row pruning is a structured pruning technique that removes entire rows from weight matrices. This results in actual shape changes and provides computational speedups.

### Configuration

```json
{
    "compression_training": {
        "row_pruning": {
            "shared_parameters": {
                "enabled": true,
                "schedule_offset": 0,
                "method": "topk"
            },
            "different_groups": {
                "rp_group_1": {
                    "params": {
                        "dense_ratio": 0.5
                    },
                    "modules": ["mlp"]
                }
            }
        }
    }
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch |
| `method` | str | `"topk"` | Selection method (`"topk"` selects rows with largest norms) |
| `schedule_offset` | int | `0` | When to start pruning |
| `dense_ratio` | float | `1.0` | Fraction of rows to keep |
| `related_modules` | list | `[]` | Modules whose shapes must be kept consistent (e.g., attention projections that share dimensions) |

### TopK Row Selection

```python
def topk_row_prune(weight, dense_ratio):
    """Select top-k rows by L2 norm."""
    row_norms = weight.norm(dim=1)  # [out_features]
    k = int(len(row_norms) * dense_ratio)
    _, indices = torch.topk(row_norms, k)
    mask = torch.zeros_like(row_norms)
    mask[indices] = 1.0
    return mask
```

### related_modules

When two layers share a dimension (e.g., the output dimension of one layer equals the input dimension of the next), row pruning must be coordinated:

```json
{
    "related_modules": ["attention.query", "attention.key", "attention.value"]
}
```

This ensures that the same rows are pruned across all related modules, maintaining shape consistency.

---

## Column Pruning

Column pruning removes entire columns from weight matrices (i.e., the input dimension). Similar to row pruning but operates on the other dimension.

### Configuration

```json
{
    "compression_training": {
        "channel_pruning": {
            "shared_parameters": {
                "enabled": true,
                "schedule_offset": 0,
                "method": "topk"
            },
            "different_groups": {
                "cp_group_1": {
                    "params": {
                        "dense_ratio": 0.5
                    },
                    "modules": ["mlp"]
                }
            }
        }
    }
}
```

The parameters and mechanics are analogous to row pruning but applied to the column (input) dimension.

---

## Head Pruning

Head pruning removes entire attention heads from multi-head attention layers. This is the most impactful structured pruning technique for transformer models.

### Configuration

```json
{
    "compression_training": {
        "head_pruning": {
            "shared_parameters": {
                "enabled": true,
                "schedule_offset": 0,
                "method": "topk"
            },
            "different_groups": {
                "hp_group_1": {
                    "params": {
                        "dense_ratio": 0.75
                    },
                    "modules": ["attention"]
                }
            }
        }
    }
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch |
| `method` | str | `"topk"` | Head selection method |
| `schedule_offset` | int | `0` | When to start head pruning |
| `dense_ratio` | float | `1.0` | Fraction of attention heads to keep |
| `related_modules` | list | `[]` | Q/K/V projections that must be pruned together |

### Head Importance Scoring

The `topk` method scores each attention head by the L2 norm of its corresponding weight partition:

```python
def score_attention_heads(query_weight, num_heads):
    """Score each attention head by its weight norm."""
    head_dim = query_weight.shape[0] // num_heads
    scores = []
    for h in range(num_heads):
        head_weights = query_weight[h * head_dim : (h+1) * head_dim]
        scores.append(head_weights.norm())
    return torch.stack(scores)
```

---

## Layer Reduction (Knowledge Distillation)

Layer reduction compresses a model by removing entire transformer layers and using knowledge distillation to recover accuracy. The student model has fewer layers than the teacher.

### Configuration

```json
{
    "compression_training": {
        "layer_reduction": {
            "enabled": true,
            "schedule_offset": 0,
            "keep_number_layer": 12
        }
    }
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch |
| `schedule_offset` | int | `0` | When to start layer reduction |
| `keep_number_layer` | int | -1 | Number of layers to keep in the compressed model |

### How Layer Reduction Works

1. **Teacher model**: The original model with all layers is loaded as the teacher.
2. **Student model**: A copy with only `keep_number_layer` layers.
3. **Layer selection** (`teacher_layer`): Determines which layers from the teacher correspond to which layers in the student. Common strategies:
   - **Uniform sampling**: Every N-th layer is kept (e.g., for 24 layers -> 12 layers, keep every other layer)
   - **Last N layers**: Keep the final `keep_number_layer` layers
   - **Balanced sampling**: Keep layers evenly distributed across the model

4. **Knowledge distillation loss**:
```python
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL divergence distillation loss."""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)
```

### Integration with Training Loop

```python
# In the training loop with layer reduction
for batch in dataloader:
    # Forward pass through student
    student_output = model_engine(batch)
    # Forward pass through teacher (no gradients)
    with torch.no_grad():
        teacher_output = teacher_model(batch)
    # Compute distillation loss
    kd_loss = distillation_loss(student_output, teacher_output)
    # Combined loss
    total_loss = student_output.loss + kd_loss
    model_engine.backward(total_loss)
    model_engine.step()
```

---

## FP8 Quantization (ops/fp_quantizer/)

DeepSpeed provides FP8 quantization support for training and inference through the `ops/fp_quantizer/` module.

### FP8 Formats

| Format | Sign | Exponent | Mantissa | Range |
|---|---|---|---|---|
| E4M3 | 1 | 4 | 3 | +/-448, NaN |
| E5M2 | 1 | 5 | 2 | +/-57344, NaN, Inf |

- **E4M3**: Higher precision, smaller range. Used for forward-pass weights and activations.
- **E5M2**: Lower precision, larger range. Used for backward-pass gradients.

### Usage

```python
from deepspeed.ops.fp_quantizer import FP_Quantizer

quantizer = FP_Quantizer()

# Quantize tensor to FP8 E4M3
fp8_tensor = quantizer.quantize(
    tensor,
    q_bits=8,
    fp8_format="e4m3"
)

# Dequantize back
restored_tensor = quantizer.dequantize(fp8_tensor, scale)
```

### FP8 Quantization Configuration

FP8 quantization can be configured alongside other compression techniques or used standalone for training efficiency:

```json
{
    "fp8_quantization": {
        "enabled": true,
        "fp8_format": "hybrid"
    }
}
```

The `"hybrid"` format uses E4M3 for forward pass and E5M2 for backward pass automatically.

---

## INT4/INT8 Inference Quantization

DeepSpeed supports INT4 and INT8 quantization for inference, enabling efficient deployment of large models.

### INT8 Inference

```python
from deepspeed.ops.quantizer import DequantizeLinear, QuantizeLinear

# Quantize a linear layer for INT8 inference
quantized_layer = QuantizeLinear(
    in_features,
    out_features,
    q_bits=8,
    q_group_size=-1  # -1 for per-channel, positive for group-wise
)
```

### INT4 Inference

```python
# INT4 group-wise quantization
quantized_layer = QuantizeLinear(
    in_features,
    out_features,
    q_bits=4,
    q_group_size=128  # Group every 128 elements
)
```

### Inference Quantization Pipeline

1. **Calibration**: Run a few batches through the model to collect activation statistics.
2. **Quantization**: Compute scales and zero-points from calibration data.
3. **Conversion**: Replace FP32/FP16 layers with quantized equivalents.
4. **Fine-tuning** (optional): Run quantization-aware training to recover accuracy.

---

## Configuration Examples

### Example 1: INT8 Weight + Activation Quantization (ZeroQuant)

```json
{
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantizer_kernel": false,
                "schedule_offset": 0,
                "quantize_groups": 1,
                "quantization_type": "symmetric",
                "rounding": "nearest",
                "quantize_weight_in_forward": true,
                "fp16_mixed_quantize": false
            },
            "different_groups": {
                "wq_all": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1
                    },
                    "modules": ["attention", "mlp", "embed"]
                }
            }
        },
        "activation_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "symmetric",
                "range_calibration": "dynamic",
                "schedule_offset": 0
            },
            "different_groups": {
                "aq_all": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### Example 2: Progressive 8-to-4 Bit Weight Quantization

```json
{
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantizer_kernel": false,
                "schedule_offset": 1000,
                "quantize_groups": 4,
                "quantization_type": "symmetric",
                "rounding": "nearest",
                "quantize_weight_in_forward": true,
                "fp16_mixed_quantize": true
            },
            "different_groups": {
                "wq_progressive": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 4,
                        "quantization_period": 2000
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        }
    }
}
```

### Example 3: Sparse Pruning + Weight Quantization

```json
{
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "symmetric",
                "quantize_weight_in_forward": true,
                "schedule_offset": 2000
            },
            "different_groups": {
                "wq": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 4,
                        "quantization_period": 1000
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        },
        "sparse_pruning": {
            "shared_parameters": {
                "enabled": true,
                "method": "l1",
                "schedule_offset": 0
            },
            "different_groups": {
                "sp": {
                    "params": {
                        "dense_ratio": 0.5
                    },
                    "modules": ["mlp"]
                }
            }
        }
    }
}
```

### Example 4: Head Pruning + Layer Reduction

```json
{
    "compression_training": {
        "head_pruning": {
            "shared_parameters": {
                "enabled": true,
                "method": "topk",
                "schedule_offset": 500
            },
            "different_groups": {
                "hp": {
                    "params": {
                        "dense_ratio": 0.5
                    },
                    "modules": ["attention"]
                }
            }
        },
        "layer_reduction": {
            "enabled": true,
            "schedule_offset": 10000,
            "keep_number_layer": 6
        }
    }
}
```

### Example 5: Row Pruning + Channel Pruning

```json
{
    "compression_training": {
        "row_pruning": {
            "shared_parameters": {
                "enabled": true,
                "method": "topk",
                "schedule_offset": 0
            },
            "different_groups": {
                "rp_mlp": {
                    "params": {
                        "dense_ratio": 0.75
                    },
                    "modules": ["mlp.dense_4h_to_h"]
                }
            }
        },
        "channel_pruning": {
            "shared_parameters": {
                "enabled": true,
                "method": "topk",
                "schedule_offset": 0
            },
            "different_groups": {
                "cp_mlp": {
                    "params": {
                        "dense_ratio": 0.75
                    },
                    "modules": ["mlp.dense_h_to_4h"]
                }
            }
        }
    }
}
```

### Example 6: Full Compression Pipeline

```json
{
    "compression_training": {
        "weight_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantizer_kernel": false,
                "schedule_offset": 2000,
                "quantize_groups": 1,
                "quantization_type": "symmetric",
                "rounding": "nearest",
                "quantize_weight_in_forward": true,
                "fp16_mixed_quantize": false
            },
            "different_groups": {
                "wq_attention": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 4,
                        "quantization_period": 1000
                    },
                    "modules": ["attention"]
                },
                "wq_mlp": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 4,
                        "quantization_period": 500
                    },
                    "modules": ["mlp"]
                }
            }
        },
        "activation_quantization": {
            "shared_parameters": {
                "enabled": true,
                "quantization_type": "symmetric",
                "range_calibration": "dynamic",
                "schedule_offset": 3000
            },
            "different_groups": {
                "aq_all": {
                    "params": {
                        "start_bits": 8,
                        "target_bits": 8,
                        "quantization_period": 1
                    },
                    "modules": ["attention", "mlp"]
                }
            }
        },
        "sparse_pruning": {
            "shared_parameters": {
                "enabled": true,
                "method": "l1",
                "schedule_offset": 0
            },
            "different_groups": {
                "sp_mlp": {
                    "params": {
                        "dense_ratio": 0.6
                    },
                    "modules": ["mlp"]
                }
            }
        },
        "head_pruning": {
            "shared_parameters": {
                "enabled": true,
                "method": "topk",
                "schedule_offset": 1000
            },
            "different_groups": {
                "hp_attn": {
                    "params": {
                        "dense_ratio": 0.75
                    },
                    "modules": ["attention"]
                }
            }
        },
        "layer_reduction": {
            "enabled": true,
            "schedule_offset": 8000,
            "keep_number_layer": 12
        }
    }
}
```

---

## Training with Compression

### Basic Compression Training Loop

```python
import deepspeed
from deepspeed.compression.compress import compression_scheduler

# Initialize with compression config
ds_config = {
    "compression_training": {
        "weight_quantization": { ... },
        "activation_quantization": { ... }
    }
}

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# The compression scheduler is automatically integrated.
# Training loop is standard:
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model_engine(batch)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()
        # Compression schedule is updated automatically in model_engine.step()
```

### Manual Compression Schedule Control

```python
# Access the compression scheduler
scheduler = model_engine.compression_scheduler

# Check current compression state
current_bits = scheduler.get_current_weight_bits()
print(f"Current weight bits: {current_bits}")

# Force a specific schedule step (advanced)
scheduler.step(forced_step_index=5000)
```

### Compression Metrics Logging

```python
# Log compression metrics
if hasattr(model_engine, 'compression_scheduler'):
    cs = model_engine.compression_scheduler
    metrics = {
        "compression/weight_bits": cs.weight_bits,
        "compression/activation_bits": cs.activation_bits,
        "compression/sparsity": cs.sparsity_ratio,
        "compression/dense_ratio": cs.dense_ratio,
    }
    # Log to wandb/tensorboard
    for k, v in metrics.items():
        wandb.log({k: v})
```

---

## Best Practices

1. **Start with weight quantization only**: Begin with INT8 weight quantization before adding activation quantization or pruning. This provides a baseline for accuracy loss.

2. **Use gradual bit-width reduction**: Rather than jumping directly to 4-bit, use the progressive schedule (`start_bits=8`, `target_bits=4`) to allow the model to adapt.

3. **Enable fp16_mixed_quantize for aggressive quantization**: When going below 6 bits, mixed quantization helps maintain training stability.

4. **Combine techniques judiciously**: Not all compression techniques compose well. Recommended combinations:
   - Weight quantization + sparse pruning (complementary)
   - Weight quantization + activation quantization (ZeroQuant style)
   - Head pruning + layer reduction (structured compression)
   - Avoid: Row pruning + column pruning on the same layer simultaneously

5. **Calibrate after compression**: After training with compression, run a calibration pass to fine-tune the quantization parameters for inference.

6. **Use grouped quantization for better accuracy**: Set `quantize_groups` to 4 or 8 for better accuracy at low bit-widths, at the cost of slightly higher memory for scale factors.

7. **Schedule compression after warmup**: Set `schedule_offset` to be after the learning rate warmup period to avoid interference between compression and initial training dynamics.
