# Chapter 28: Post-Training Optimization

## Source Files
- `megatron/core/post_training/__init__.py` - Post-training module initialization
- `megatron/core/post_training/modelopt/__init__.py` - ModelOpt integration
- `megatron/core/post_training/modelopt/layers.py` - ModelOpt layer wrappers (Norm, quantization)
- `megatron/core/post_training/modelopt/gpt/model_specs.py` - GPT model specs for ModelOpt
- `megatron/core/post_training/modelopt/gpt/state_dict_hooks.py` - State dict conversion hooks
- `megatron/core/post_training/modelopt/hybrid/model_specs.py` - Hybrid model specs
- `megatron/core/post_training/modelopt/mamba/model_specs.py` - Mamba model specs

## Overview

Post-training optimization in Megatron-LM encompasses techniques applied after the initial pretraining phase to reduce model size, improve inference throughput, or adapt models for specific tasks. The framework integrates with NVIDIA's TensorRT ModelOpt library for production-quality quantization and distillation.

## ModelOpt Integration

The post-training module integrates with TensorRT ModelOpt (`modelopt.torch`), which provides:

- Quantization-Aware Training (QAT)
- Post-Training Quantization (PTQ)
- Knowledge Distillation
- INT8/INT4 weight-only quantization
- FP8 quantization
- Sparse pruning

### Dependencies

```bash
pip install nvidia-modelopt torch-tensorrt
```

The ModelOpt integration is optional and gracefully degrades when not installed:

```python
try:
    from modelopt.torch.distill.plugins.megatron import (
        get_tensor_shapes_adjust_fn_for_distillation,
    )
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False
```

## Quantization-Aware Training (QAT)

QAT simulates quantization during training so the model learns to be robust to quantization noise:

### FP8 Per-Tensor Quantization Config

```python
FP8_PER_TENSOR_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}
```

### FP8 2D Blockwise Quantization Config

```python
FP8_2D_BLOCKWISE_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 128, -2: 128}
        },
        "*input_quantizer": {"enable": False},
        "*output_layer*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}
```

### Quantization Configuration via YAML

The quantization recipe framework (`megatron/core/quantization/`) provides YAML-based per-layer quantization:

```yaml
configs:
  int8_weight_only:
    quant_cfg:
      "*weight_quantizer":
        num_bits: 8
        axis: 0
      "*input_quantizer":
        enable: false
      "default":
        enable: false
    algorithm: max

matchers:
  all_linear:
    config: "int8_weight_only"
    type: "glob"
    pattern: "*"
    enabled: true
```

```bash
--quantization-recipe /path/to/quant_recipe.yaml
```

### Kitchen Sink Quantization

The `kitchen_quantization_recipe_config` provides a pre-configured QAT recipe:

```python
from megatron.core.quantization.utils import kitchen_quantization_recipe_config

recipe = kitchen_quantization_recipe_config(recipe_idx=0)
```

This creates a recipe that applies `QLinearParams` quantization to all layers.

## Distillation

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model:

### ModelOpt Distillation Plugin

The ModelOpt distillation plugin for Megatron provides:

```python
from modelopt.torch.distill.plugins.megatron import (
    get_tensor_shapes_adjust_fn_for_distillation,
)
```

This function adjusts tensor shapes for the distillation forward pass, handling:
- Tensor parallel sharding differences between teacher and student
- Pipeline parallel stage boundary handling
- Hidden size mismatches between teacher and student

### Distillation Process

1. **Setup teacher and student models:**
   - Teacher: Large pretrained model (frozen)
   - Student: Smaller model being trained

2. **Forward pass through both models:**
   - Teacher forward generates soft labels (logits, hidden states)
   - Student forward generates predictions

3. **Compute distillation loss:**
   - KL divergence between teacher and student logits
   - Optional intermediate layer matching losses
   - Combined with the standard training loss

4. **Update student model only:**
   - Gradients flow only through the student model
   - Teacher parameters are frozen

### Model Specs for Distillation

Model specs define the architecture mapping for ModelOpt:

```python
# GPT model specs
from megatron.core.post_training.modelopt.gpt.model_specs import ...

# Hybrid model specs
from megatron.core.post_training.modelopt.hybrid.model_specs import ...

# Mamba model specs
from megatron.core.post_training.modelopt.mamba.model_specs import ...
```

These specs define:
- Layer type mappings (attention, MLP, MoE)
- Quantization targets per layer
- Distillation loss configuration per layer pair

### State Dict Hooks

The `state_dict_hooks.py` module handles converting between Megatron and ModelOpt state dict formats:

- ModelOpt uses different key naming conventions
- Megatron has its own sharding conventions
- Hooks automatically convert between formats during save/load

## SFT (Supervised Fine-Tuning) Workflow

### SFT Tokenizer

The SFT tokenizer formats training data with appropriate prompt templates:

```bash
--tokenizer-type SFTTokenizer
--sft-tokenizer-prompt-format chatml
```

Supported formats:
- `chatml`: ChatML with `<|im_start|>` and `<|im_end|>` tokens
- `alpaca`: Alpaca instruction format
- `raw`: No formatting

### SFT Training

SFT follows the standard Megatron pretraining workflow with modifications:

1. **Data format:** JSONL with instruction/response pairs
2. **Loss masking:** Only compute loss on response tokens (not prompt tokens)
3. **Learning rate:** Typically much smaller than pretraining (1e-5 to 5e-6)
4. **Training duration:** Much shorter (few epochs over the dataset)

### Loss Masking

For instruction tuning, the loss is masked to only compute gradients on the assistant's response:

```python
# Generation mask indicates which tokens were generated
generation_mask = [0, 0, 0, 1, 1, 1, 1]  # 0=prompt, 1=response
```

This is handled by the RL framework's `generation_mask` field in `TokenRollout`.

## ModelOpt Layer Wrappers

### Norm Layers

The `Norm` class in `layers.py` wraps Transformer Engine's LayerNorm and RMSNorm:

```python
class Norm:
    def __new__(cls, config, hidden_size, eps=1e-5):
        if config.normalization == "LayerNorm":
            instance = te.pytorch.LayerNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
            )
        elif config.normalization == "RMSNorm":
            instance = te.pytorch.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                zero_centered_gamma=config.layernorm_zero_centered_gamma,
            )
```

These wrappers handle:
- Extra state dict keys (`_extra_state`) that ModelOpt may add
- State dict hooks for proper serialization
- Quantization-aware normalization

### Quantized Linear Layers

Quantization configurations can be applied to linear layers through ModelOpt's quantizer system:

```python
# Per-tensor FP8 quantization for weights
FP8_PER_TENSOR_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        ...
    },
    "algorithm": "max",
}

# 2D blockwise FP8 quantization
FP8_2D_BLOCKWISE_REAL_QUANT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {-1: 128, -2: 128},
        },
        ...
    },
    "algorithm": "max",
}
```

## Quantization Utilities

### get_quant_config_or_none

```python
from megatron.core.quantization.utils import get_quant_config_or_none

config = get_quant_config_or_none(
    module_path="encoder.layers.5.mlp.fc1",
    recipe=quant_recipe,
)
```

Resolves the quantization configuration for a specific layer by matching against the recipe's matchers.

### load_quantization_recipe

```python
from megatron.core.quantization.utils import load_quantization_recipe

recipe = load_quantization_recipe("/path/to/recipe.yaml")
```

Loads a quantization recipe from a YAML file, creating a `RecipeConfig` with matchers and configuration dictionaries.

## Post-Training Workflow

### Typical Post-Training Pipeline

```
Pretrained Model (BF16/FP32)
    │
    ├── SFT (Supervised Fine-Tuning)
    │   ├── Instruction/response data
    │   ├── Loss masking on response tokens
    │   └── SFTTokenizer with chat templates
    │
    ├── RL Alignment (GRPO/PPO)
    │   ├── Environment rollout collection
    │   ├── Reward computation
    │   └── Policy optimization with KL constraint
    │
    ├── Quantization-Aware Training (QAT)
    │   ├── FP8/INT8 simulation during training
    │   ├── Per-layer quantization configuration
    │   └── Fine-tuning with quantization noise
    │
    ├── Distillation (optional)
    │   ├── Teacher-student forward pass
    │   ├── KL divergence + intermediate losses
    │   └── Student model training
    │
    └── Export
        ├── TensorRT-LLM export
        └── Deployment-ready quantized model
```

### Quantization Methods Comparison

| Method | Accuracy | Inference Speed | Model Size | Use Case |
|--------|----------|----------------|------------|----------|
| FP32 baseline | Best | Slowest | Largest | Reference |
| BF16 | Near FP32 | 2x faster | 50% smaller | Default training |
| FP8 (weights + compute) | ~99% of BF16 | 2x faster than BF16 | 75% smaller | Production inference |
| INT8 weight-only | ~98% of FP32 | 1.5-2x faster | 75% smaller | Memory-constrained |
| INT4 weight-only | ~95-97% | 2-3x faster | 87.5% smaller | Edge deployment |
| FP8 + distillation | ~99.5% | 2x faster | 75% smaller | Best accuracy-speed trade-off |

## Configuration Examples

### QAT with FP8
```bash
python pretrain_gpt.py \
    --fp8 hybrid \
    --fp8-recipe delayed \
    --quantization-recipe qat_recipe.yaml \
    --use-distributed-optimizer \
    --lr 5e-6 \
    ...
```

### Distillation
```bash
python pretrain_gpt.py \
    --teacher-model-path /path/to/teacher \
    --distillation \
    --quantization-recipe distill_recipe.yaml \
    --lr 1e-5 \
    ...
```

### SFT
```bash
python pretrain_gpt.py \
    --tokenizer-type SFTTokenizer \
    --sft-tokenizer-prompt-format chatml \
    --data-path /path/to/sft_data \
    --lr 2e-6 \
    --min-lr 1e-7 \
    --weight-decay 0.01 \
    ...
```

### Post-Training Quantization with ModelOpt
```bash
python pretrain_gpt.py \
    --load /path/to/pretrained_checkpoint \
    --quantization-recipe /path/to/quant_recipe.yaml \
    --qat \
    --iterations 1000 \
    --lr 1e-6 \
    ...
```

## Integration with Export Pipeline

After post-training optimization, the model can be exported to TensorRT-LLM for production deployment:

```bash
python tools/export.py \
    --model-type gpt \
    --load /path/to/quantized_checkpoint \
    --export-dir /path/to/export \
    --target-tensorrt-llm
```

The export pipeline:
1. Loads the quantized Megatron checkpoint
2. Converts quantization parameters to TensorRT-LLM format
3. Builds the TensorRT engine with optimized kernels
4. Produces a deployment-ready model artifact
