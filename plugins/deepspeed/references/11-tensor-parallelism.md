# DeepSpeed Tensor Parallelism (AutoTP) Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration Reference](#configuration-reference)
4. [AutoTP: Automatic Tensor Parallelism](#autotp-automatic-tensor-parallelism)
5. [Partition Configuration](#partition-configuration)
6. [HuggingFace tp_plan Auto-Detection](#huggingface-tp_plan-auto-detection)
7. [Manual Tensor Parallelism](#manual-tensor-parallelism)
8. [TP + ZeRO Hybrid](#tp--zero-hybrid)
9. [Communication Patterns](#communication-patterns)
10. [Random Number Synchronization](#random-number-synchronization)
11. [Model Parsing and Policy Application](#model-parsing-and-policy-application)
12. [Code Examples](#code-examples)
13. [Performance Tuning](#performance-tuning)
14. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed Tensor Parallelism (AutoTP) partitions individual tensor operations across multiple GPUs so that each device holds a slice of every large weight matrix. Unlike data parallelism (which replicates the model) or pipeline parallelism (which partitions layers), tensor parallelism splits the **weights and activations within a single layer** across devices, enabling:

- Training and inference of models too large for a single GPU's memory
- Linear memory reduction proportional to the tensor-parallel degree
- Efficient compute utilization via overlapping communication with computation
- Seamless integration with ZeRO stages 0, 1, and 2

Tensor parallelism is most effective for large dense layers (linear projections, embedding tables, output heads) where the weight matrices can be cleanly partitioned along a specific dimension.

---

## Architecture

### Core Components

```
deepspeed/
  runtime/
    tensor_parallel/
      __init__.py
      tp_training_manager.py    # TpTrainingManager class
      tp_config.py              # TPConfig and TPTrainingConfig
  module_inject/
    auto_tp.py                  # AutoTP automatic detection and injection
    tp_module_wrapper.py        # Wrapper for tensor-parallelized modules
```

### Class Hierarchy

```
TPConfig
  +-- TPTrainingConfig
        +-- TpTrainingManager
```

---

## Configuration Reference

### TPConfig

The base configuration class for tensor parallelism settings.

```python
@dataclass
class TPConfig:
    """Base configuration for tensor parallelism."""
    enabled: bool = False
    tp_size: int = 1
    tp_grain_size: int = 64
    tp_overlap_comm: bool = False
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Whether tensor parallelism is active |
| `tp_size` | `int` | `1` | Number of GPUs across which tensors are partitioned |
| `tp_grain_size` | `int` | `64` | Granularity (in elements) for partitioning; must be a power of 2 |
| `tp_overlap_comm` | `bool` | `False` | Overlap all-reduce communication with backward computation |

### TPTrainingConfig

Extends `TPConfig` with training-specific settings.

```python
@dataclass
class TPTrainingConfig:
    """Training-specific tensor parallelism configuration."""
    autotp_size: int = 1
    preset_model: Optional[str] = None
    tp_overlap_comm: bool = False
    partition_config: Optional[Dict] = None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `autotp_size` | `int` | `1` | Target tensor parallelism degree for AutoTP |
| `preset_model` | `Optional[str]` | `None` | Predefined model type for automatic policy selection |
| `tp_overlap_comm` | `bool` | `False` | Enable communication-computation overlap |
| `partition_config` | `Optional[Dict]` | `None` | Manual partition specification for custom models |

### DeepSpeed Configuration JSON

```json
{
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 4,
        "preset_model": "llama",
        "tp_overlap_comm": true,
        "tp_grain_size": 64,
        "partition_config": {
            "layer_specs": [],
            "patterns": [],
            "model_types": []
        }
    }
}
```

---

## AutoTP: Automatic Tensor Parallelism

AutoTP is DeepSpeed's mechanism for automatically applying tensor parallelism to transformer models without requiring manual specification of which layers to partition. It is implemented in `deepspeed/module_inject/auto_tp.py`.

### How AutoTP Works

1. **Model Detection**: AutoTP inspects the model's class name and configuration to determine the model family (e.g., LLaMA, BLOOM, ChatGLM).
2. **Policy Selection**: Based on the detected or specified model type, AutoTP selects a predefined partitioning policy that describes which layers to partition and how.
3. **Module Replacement**: The selected policy is applied by replacing target modules with tensor-parallelized equivalents.
4. **Weight Redistribution**: Pre-trained weights are automatically redistributed across the tensor-parallel group.

### AutoTP Entry Point

```python
from deepspeed.module_inject.auto_tp import auto_tp

# Apply AutoTP to a model
model = auto_tp(
    model,
    tp_size=4,
    mp_group=mp_group,       # torch.distributed process group for TP
    preset_model="llama",    # Optional: override auto-detection
    partition_config=None,   # Optional: custom partition config
    tp_grain_size=64,
)
```

### Supported Preset Models

AutoTP includes built-in partitioning policies for the following model families:

| Preset Name | Model Family | Key Layers Partitioned |
|-------------|-------------|----------------------|
| `llama` | LLaMA / LLaMA-2 / LLaMA-3 | QKV projections, output projection, FFN up/down/gate |
| `bloom` | BLOOM | QKV projections, output projection, FFN |
| `chatglm` | ChatGLM | QKV projections, output projection, dense FFN |
| `mixtral` | Mixtral 8x7B / 8x22B | QKV projections, output projection, expert FFN layers |
| `deepseek_v2` | DeepSeek-V2 | QKV projections, output projection, MLA layers, expert FFN |
| `qwen2` | Qwen2 / Qwen2.5 | QKV projections, output projection, FFN up/down/gate |
| `phi3` | Phi-3 | QKV projections, output projection, FFN down/up/gate |

### AutoTP Configuration via DeepSpeed JSON

```json
{
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 4,
        "preset_model": "llama",
        "tp_overlap_comm": true
    },
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5
        }
    }
}
```

### Launching AutoTP Training

```bash
# Launch with 4-way tensor parallelism
deepspeed --num_gpus=4 train.py \
    --deepspeed_config ds_config.json \
    --model_name meta-llama/Llama-2-7b-hf
```

```python
# Python API
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
ds_engine = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)
```

---

## Partition Configuration

The `partition_config` provides fine-grained control over how individual layers are partitioned. This is useful for custom architectures not covered by the preset models.

### Structure

```python
partition_config = {
    "layer_specs": [...],    # Explicit layer specifications
    "patterns": [...],       # Regex-based pattern matching
    "model_types": [...],    # Model type constraints
}
```

### Layer Specifications

Each entry in `layer_specs` describes how a specific layer should be partitioned:

```python
{
    "layer_specs": [
        {
            "module": "model.layers.0.self_attn.q_proj",
            "partition_type": "column",   # or "row" or "skip"
            "shape": [4096, 4096],         # Original weight shape
            "partition_dim": 0,            # Dimension along which to partition
        },
        {
            "module": "model.layers.0.self_attn.k_proj",
            "partition_type": "column",
            "shape": [4096, 4096],
            "partition_dim": 0,
        },
        {
            "module": "model.layers.0.self_attn.v_proj",
            "partition_type": "column",
            "shape": [4096, 4096],
            "partition_dim": 0,
        },
        {
            "module": "model.layers.0.self_attn.o_proj",
            "partition_type": "row",
            "shape": [4096, 4096],
            "partition_dim": 1,
        },
    ]
}
```

### Partition Types

| Partition Type | Description | Communication | Use Case |
|---------------|-------------|---------------|----------|
| `column` | Partition weight matrix along rows (output dimension) | All-reduce after forward | QKV projections, FFN up/gate projections |
| `row` | Partition weight matrix along columns (input dimension) | All-reduce after forward | Output projections, FFN down projections |
| `skip` | Do not partition this layer | None | LayerNorm, bias terms, small layers |

### partition_dim Values

| partition_dim | Column Parallel | Row Parallel |
|--------------|----------------|--------------|
| `0` | Split output features | Split input features |
| `1` | Split input features | Split output features |

For a weight matrix of shape `[out_features, in_features]`:
- **Column parallel** with `partition_dim=0`: Each GPU gets `[out_features/tp_size, in_features]`
- **Row parallel** with `partition_dim=1`: Each GPU gets `[out_features, in_features/tp_size]`

### Pattern-Based Partitioning

Instead of listing every layer individually, use regex patterns:

```python
{
    "patterns": [
        {
            "regex": ".*self_attn\\.(q_proj|k_proj|v_proj)\\.weight",
            "partition_type": "column",
            "partition_dim": 0
        },
        {
            "regex": ".*self_attn\\.o_proj\\.weight",
            "partition_type": "row",
            "partition_dim": 1
        },
        {
            "regex": ".*mlp\\.(gate_proj|up_proj)\\.weight",
            "partition_type": "column",
            "partition_dim": 0
        },
        {
            "regex": ".*mlp\\.down_proj\\.weight",
            "partition_type": "row",
            "partition_dim": 1
        },
    ]
}
```

### model_types Filter

Restrict partitioning rules to specific model architectures:

```python
{
    "model_types": ["LlamaForCausalLM", "LlamaModel"],
    "patterns": [...]
}
```

### Shape Constraint

Specify expected shapes to ensure correct partitioning:

```python
{
    "regex": ".*self_attn\\.q_proj\\.weight",
    "partition_type": "column",
    "partition_dim": 0,
    "shape": [4096, 4096],   # Only match layers with this exact shape
}
```

---

## HuggingFace tp_plan Auto-Detection

Modern HuggingFace transformers models include a `tp_plan` attribute in their configuration that specifies tensor parallelism plans. DeepSpeed AutoTP can automatically detect and use these plans.

### How tp_plan Detection Works

```python
# In auto_tp.py, the detection logic checks:
if hasattr(model.config, "base_model_tp_plan"):
    tp_plan = model.config.base_model_tp_plan
    # Parse the plan and apply partitioning accordingly
```

### tp_plan Format

HuggingFace `tp_plan` is a dictionary mapping layer name patterns to partitioning instructions:

```python
# Example from a HuggingFace model config
model.config.base_model_tp_plan = {
    "model.layers.*.self_attn.q_proj": "column",
    "model.layers.*.self_attn.k_proj": "column",
    "model.layers.*.self_attn.v_proj": "column",
    "model.layers.*.self_attn.o_proj": "row",
    "model.layers.*.mlp.gate_proj": "column",
    "model.layers.*.mlp.up_proj": "column",
    "model.layers.*.mlp.down_proj": "row",
}
```

### Auto-Detection Priority

When both `preset_model` and HuggingFace `tp_plan` are available:

1. If `preset_model` is explicitly specified, use the DeepSpeed built-in policy.
2. If `preset_model` is `None`, check for HuggingFace `base_model_tp_plan`.
3. If neither is available, attempt to infer the model type from the model class name.

```python
# Priority logic in auto_tp.py (simplified)
def resolve_partition_policy(model, preset_model, partition_config):
    if partition_config is not None:
        return partition_config  # Manual config takes highest priority
    if preset_model is not None:
        return get_preset_policy(preset_model)
    if hasattr(model.config, "base_model_tp_plan"):
        return parse_hf_tp_plan(model.config.base_model_tp_plan)
    return infer_policy_from_class(model)
```

---

## Manual Tensor Parallelism

For maximum control, DeepSpeed provides a manual tensor parallelism API via `tp_model_init()`.

### ds_tensor_parallel Module

```python
import deepspeed

# Initialize tensor parallelism for a model
model = deepspeed.tp_model_init(
    model,
    tp_size=4,
    mp_group=None,         # Optional: specify process group
    partition_config=None,  # Optional: custom partition config
)
```

### tp_model_init() API

```python
def tp_model_init(
    model: torch.nn.Module,
    tp_size: int = 1,
    mp_group: Optional[torch.distributed.ProcessGroup] = None,
    partition_config: Optional[Dict] = None,
    tp_grain_size: int = 64,
    preset_model: Optional[str] = None,
) -> torch.nn.Module:
    """Initialize tensor parallelism on a model.

    Args:
        model: The PyTorch model to tensor-parallelize.
        tp_size: Number of tensor-parallel partitions.
        mp_group: Process group for tensor parallel communication.
                  If None, the default TP group is used.
        partition_config: Custom partition configuration dictionary.
        tp_grain_size: Granularity for weight partitioning (default: 64).
        preset_model: Model family name for built-in policy selection.

    Returns:
        The tensor-parallelized model.
    """
```

### Manual Partitioning Example

```python
import torch
import torch.nn as nn
import deepspeed
from deepspeed.runtime.tensor_parallel import TpTrainingManager

# Define custom partition config
custom_config = {
    "patterns": [
        {
            "regex": ".*attention\\.(query|key|value)\\.weight",
            "partition_type": "column",
            "partition_dim": 0,
        },
        {
            "regex": ".*attention\\.output\\.weight",
            "partition_type": "row",
            "partition_dim": 1,
        },
    ]
}

# Initialize model
model = CustomTransformerModel()

# Apply tensor parallelism
model = deepspeed.tp_model_init(
    model,
    tp_size=2,
    partition_config=custom_config,
    tp_grain_size=64,
)

# Initialize DeepSpeed engine
ds_engine = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=list(model.parameters()),
)
```

### tp_grain_size Parameter

The `tp_grain_size` parameter controls the minimum partition granularity:

```python
# tp_grain_size=64 means partition sizes are multiples of 64
# For a weight of shape [4096, 4096] with tp_size=4:
#   Partition size = 4096 / 4 = 1024 (multiple of 64, OK)

# For a weight of shape [100, 4096] with tp_size=4:
#   Partition size = 100 / 4 = 25 (NOT a multiple of 64)
#   Effective partition = 25, rounded down to nearest multiple of 64 = 0
#   This layer would be skipped or handled differently
```

**Recommended values:**

| tp_grain_size | Use Case |
|---------------|----------|
| `1` | Maximum flexibility, any dimension |
| `32` | Good balance for medium models |
| `64` | Default, optimal for most transformer models |
| `128` | Large models with very wide layers |
| `256` | Extremely large models (70B+ parameters) |

---

## TP + ZeRO Hybrid

Tensor parallelism can be combined with ZeRO optimization stages for maximum memory savings and scalability.

### Supported Combinations

| ZeRO Stage | TP Supported | Notes |
|-----------|-------------|-------|
| Stage 0 (DDP) | Yes | TP only, no ZeRO optimization |
| Stage 1 | Yes | Optimizer state partitioning + TP |
| Stage 2 | Yes | Optimizer + gradient partitioning + TP |
| Stage 3 | **No** | Not supported; use ZeRO-3 alone or TP+ZeRO-2 |

### Configuration for TP + ZeRO-2

```json
{
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 4,
        "preset_model": "llama"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8
    },
    "gradient_accumulation_steps": 4,
    "train_batch_size": 64,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": true
    }
}
```

### Resource Calculation for TP + ZeRO-2

For a model with P parameters, TP size = T, and DP size = D (total GPUs = T * D):

| Resource | Per-GPU Memory |
|----------|---------------|
| Model parameters | `P * bytes_per_param / T` |
| Gradients | `P * bytes_per_param / T` |
| Optimizer states | `P * bytes_per_param * optimizer_bytes / (T * D)` |
| Activations | Depends on batch size and sequence length |

### Launch Command

```bash
# 8 GPUs: 4-way TP + 2-way DP
deepspeed --num_gpus=8 train.py \
    --deepspeed_config ds_config_tp_zero2.json
```

---

## Communication Patterns

Tensor parallelism introduces collective communication operations at specific points in the forward and backward passes.

### Column Parallel Communication

In column parallelism, the weight matrix is split along the output dimension. Each GPU computes a partial output, and an **all-reduce** is needed to combine results.

```
Forward Pass:
  Input X --> [GPU0: Y0 = XW0]  |
  Input X --> [GPU1: Y1 = XW1]  |  All-Gather or identity
  Input X --> [GPU2: Y2 = XW2]  |  (partial results are independent)
  Input X --> [GPU3: Y3 = XW3]  |

Backward Pass:
  dL/dY0 ------+
  dL/dY1 ------+--> All-Reduce --> dL/dX
  dL/dY2 ------+
  dL/dY3 ------+
```

For column parallel, the input is replicated, and each GPU computes a different slice of the output. No communication is needed in the forward pass.

### Row Parallel Communication

In row parallelism, the weight matrix is split along the input dimension. Each GPU operates on a different slice of the input, and an **all-reduce** combines partial sums.

```
Forward Pass:
  X0 --> [GPU0: Y0 = X0W0] ---+
  X1 --> [GPU1: Y1 = X1W1] ---+--> All-Reduce --> Y = sum(Y0..Y3)
  X2 --> [GPU2: Y2 = X2W2] ---+
  X3 --> [GPU3: Y3 = X3W3] ---+

Backward Pass:
  dL/dY --> broadcast to all GPUs
  Each GPU computes: dL/dXi = dL/dY * Wi^T
```

### Combined Pattern (Transformer Block)

A standard transformer attention block uses a combined column-row pattern:

```
                    ┌─────────────────────┐
                    │   Input (replicated) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Column Parallel     │
                    │  QKV Projections     │  (split output dim)
                    │  [out/tp, in]        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Attention Computation │
                    │  (local heads only)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Row Parallel        │
                    │  Output Projection   │  (split input dim)
                    │  [out, in/tp]        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  All-Reduce          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Output (replicated)│
                    └─────────────────────┘
```

Similarly, the FFN block:

```
                    ┌─────────────────────┐
                    │   Input (replicated) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐     │     ┌──────────▼──────────┐
    │ Column Parallel    │     │     │ Column Parallel      │
    │ Gate Projection    │     │     │ Up Projection        │
    └─────────┬─────────┘     │     └──────────┬──────────┘
              │               │               │
              └────────┬──────┘───────────────┘
                       │
              ┌────────▼────────┐
              │ SiLU Activation │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Row Parallel    │
              │ Down Projection │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ All-Reduce      │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Output          │
              └─────────────────┘
```

### Communication Overlap

When `tp_overlap_comm=True`, the all-reduce communication is overlapped with computation:

```python
# Without overlap:
output = row_parallel_forward(input)
output = all_reduce(output)       # GPU idle during communication

# With overlap:
output_chunks = []
for i, chunk in enumerate(input_chunks):
    partial = row_parallel_forward(chunk)
    if i > 0:
        prev_output = wait_for_all_reduce(prev_handle)
    prev_handle = async_all_reduce(partial)
output = finalize_all_reduce()
```

---

## Random Number Synchronization

Tensor parallelism must ensure that random operations (e.g., dropout) produce consistent results across all TP ranks when needed, while allowing independent randomness where appropriate.

### Synchronization Strategy

1. **Replicated Layers (LayerNorm, etc.)**: All TP ranks use the same random seed. These layers produce identical outputs on every rank.

2. **Column Parallel Layers (QKV projections)**: Each rank handles different output slices. Dropout within these slices can use rank-local seeds since the outputs are independent.

3. **Row Parallel Layers (Output projections)**: After the all-reduce combines partial results, any subsequent dropout must use a synchronized seed across all TP ranks.

```python
# In TpTrainingManager
def synchronize_random_state(self):
    """Ensure all TP ranks have the same random state for synchronized operations."""
    rng_state = torch.random.get_rng_state()
    if dist.get_rank() == 0:
        # Broadcast RNG state from rank 0
        dist.broadcast(rng_state, src=0, group=self.tp_group)
    else:
        dist.broadcast(rng_state, src=0, group=self.tp_group)
        torch.random.set_rng_state(rng_state)

    # Also synchronize CUDA RNG state
    cuda_rng_state = torch.cuda.get_rng_state()
    dist.broadcast(cuda_rng_state, src=0, group=self.tp_group)
    torch.cuda.set_rng_state(cuda_rng_state)
```

### CUDA RNG State Management

DeepSpeed uses CUDA RNG state manipulation to allow both synchronized and independent random operations:

```python
# Save CUDA RNG state before a region that needs independent randomness
rng_tracker = deepspeed.checkpointing.cuda_rng_tracker()

# Independent dropout per rank (within column parallel)
with rng_tracker.fork():
    x = F.dropout(x, p=0.1)  # Different per rank

# Synchronized dropout (after row parallel all-reduce)
x = F.dropout(x, p=0.1)  # Same across all ranks
```

---

## Model Parsing and Policy Application

AutoTP includes a model parsing system that traverses the model's module hierarchy and applies partitioning policies.

### Parsing Algorithm

```python
def parse_model(model, policy, tp_size, tp_grain_size):
    """Traverse model modules and apply TP policy.

    1. Walk model.named_modules() to enumerate all leaf modules.
    2. For each module, check if its name matches any policy pattern.
    3. If matched, replace the module with a tensor-parallelized version.
    4. Verify partition dimensions are compatible with tp_size and tp_grain_size.
    """
    replacements = {}
    for name, module in model.named_modules():
        partition_info = policy.match(name, module)
        if partition_info is not None:
            tp_module = create_tp_module(
                module,
                partition_info,
                tp_size=tp_size,
                grain_size=tp_grain_size,
            )
            replacements[name] = tp_module

    # Apply replacements
    for name, tp_module in replacements.items():
        set_module_by_name(model, name, tp_module)
```

### Policy Matching

```python
class PartitionPolicy:
    """Matches module names against partitioning rules."""

    def match(self, module_name: str, module: nn.Module) -> Optional[PartitionInfo]:
        """Check if a module matches any partitioning rule.

        Returns PartitionInfo if matched, None otherwise.
        """
        for rule in self.rules:
            if re.match(rule.regex, module_name):
                if isinstance(module, nn.Linear):
                    return PartitionInfo(
                        partition_type=rule.partition_type,
                        partition_dim=rule.partition_dim,
                        original_shape=tuple(module.weight.shape),
                    )
        return None
```

### Module Replacement

When a module is identified for partitioning, it is replaced with a tensor-parallel wrapper:

```python
class TPLinear(nn.Module):
    """Tensor-parallelized linear layer."""

    def __init__(self, original_linear, partition_type, tp_size, tp_rank, tp_group):
        super().__init__()
        self.partition_type = partition_type
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        # Partition the weight
        if partition_type == "column":
            # Split along output dimension
            weight_chunks = torch.chunk(original_linear.weight, tp_size, dim=0)
            self.weight = nn.Parameter(weight_chunks[tp_rank].clone())
            if original_linear.bias is not None:
                bias_chunks = torch.chunk(original_linear.bias, tp_size, dim=0)
                self.bias = nn.Parameter(bias_chunks[tp_rank].clone())
            else:
                self.bias = None
        elif partition_type == "row":
            # Split along input dimension
            weight_chunks = torch.chunk(original_linear.weight, tp_size, dim=1)
            self.weight = nn.Parameter(weight_chunks[tp_rank].clone())
            self.bias = original_linear.bias  # Bias is not partitioned in row parallel

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        if self.partition_type == "row":
            # All-reduce for row parallel
            dist.all_reduce(output, group=self.tp_group)
        return output
```

---

## Code Examples

### Example 1: LLaMA-2 7B with AutoTP and ZeRO-2

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# DeepSpeed configuration
ds_config = {
    "tensor_parallel": {
        "enabled": True,
        "autotp_size": 4,
        "preset_model": "llama",
        "tp_overlap_comm": True,
        "tp_grain_size": 64,
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": 4,
    "train_batch_size": 64,
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
        },
    },
}

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Initialize DeepSpeed
ds_engine = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters(),
)

# Training loop
for batch in dataloader:
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(ds_engine.device) for k, v in inputs.items()}
    outputs = ds_engine(**inputs, labels=inputs["input_ids"])
    ds_engine.backward(outputs.loss)
    ds_engine.step()
```

### Example 2: Mixtral 8x7B with Custom Partition Config

```json
{
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 8,
        "preset_model": "mixtral",
        "partition_config": {
            "patterns": [
                {
                    "regex": ".*block_sparse_moe\\.experts\\.\\d+\\.w1\\.weight",
                    "partition_type": "column",
                    "partition_dim": 0
                },
                {
                    "regex": ".*block_sparse_moe\\.experts\\.\\d+\\.w3\\.weight",
                    "partition_type": "column",
                    "partition_dim": 0
                },
                {
                    "regex": ".*block_sparse_moe\\.experts\\.\\d+\\.w2\\.weight",
                    "partition_type": "row",
                    "partition_dim": 1
                }
            ]
        }
    }
}
```

### Example 3: Manual TP with tp_model_init()

```python
import torch
import torch.nn as nn
import deepspeed

class CustomAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # ... attention computation ...
        return self.o_proj(context)

class CustomTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = CustomAttention(hidden_size, num_heads)
        self.ffn_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

class CustomTransformerModel(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.embed = nn.Embedding(32000, hidden_size)

# Define partition config
partition_config = {
    "patterns": [
        # Attention projections
        {"regex": ".*attention\\.(q_proj|k_proj|v_proj)\\.weight", "partition_type": "column", "partition_dim": 0},
        {"regex": ".*attention\\.o_proj\\.weight", "partition_type": "row", "partition_dim": 1},
        # FFN projections
        {"regex": ".*ffn_(gate|up)\\.weight", "partition_type": "column", "partition_dim": 0},
        {"regex": ".*ffn_down\\.weight", "partition_type": "row", "partition_dim": 1},
    ]
}

# Create and tensor-parallelize model
model = CustomTransformerModel(
    num_layers=32,
    hidden_size=4096,
    num_heads=32,
    intermediate_size=11008,
)

model = deepspeed.tp_model_init(
    model,
    tp_size=4,
    partition_config=partition_config,
    tp_grain_size=64,
)
```

### Example 4: DeepSeek-V2 with MLA Attention

```json
{
    "tensor_parallel": {
        "enabled": true,
        "autotp_size": 4,
        "preset_model": "deepseek_v2",
        "tp_overlap_comm": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "train_batch_size": 32,
    "bf16": {"enabled": true}
}
```

---

## Performance Tuning

### Communication Overlap

Enable `tp_overlap_comm` to hide communication latency behind computation:

```json
{
    "tensor_parallel": {
        "tp_overlap_comm": true
    }
}
```

This provides 10-30% throughput improvement on most workloads by pipelining all-reduce operations.

### Grain Size Tuning

| tp_grain_size | Partitioning Overhead | Memory Alignment | Recommendation |
|---------------|----------------------|------------------|----------------|
| 1 | High | Poor | Only for debugging |
| 32 | Low | Fair | Small models |
| 64 | Optimal | Good | Default for most cases |
| 128 | Low | Excellent | Large models (>30B) |
| 256 | Very Low | Excellent | Very large models (>70B) |

### TP Size Selection

| Model Size | Recommended TP Size | Rationale |
|-----------|--------------------|-----------|
| 7B | 1-2 | Fits on single GPU with ZeRO-2 |
| 13B | 2-4 | Two GPUs with ZeRO-2 or four with ZeRO-1 |
| 30B | 4 | Memory requirement exceeds 2 GPUs |
| 70B | 4-8 | 4 with ZeRO-2 + offload, 8 without offload |
| 140B+ | 8+ | Requires 8+ GPUs for model parameters alone |

### NCCL Tuning for Tensor Parallelism

```bash
# Environment variables for optimizing TP communication
export NCCL_IB_DISABLE=0               # Enable InfiniBand
export NCCL_IB_GID_INDEX=3             # Use RoCE v2
export NCCL_SOCKET_IFNAME=eth0         # Network interface
export NCCL_DEBUG=INFO                 # Debug communication
export NCCL_ALGO=Ring                  # Force ring algorithm for all-reduce
export NCCL_PROTO=Simple               # Simple protocol for small messages
```

---

## Troubleshooting

### Common Issues

**1. Shape mismatch after partitioning**

```
RuntimeError: shape '[64, 4096]' is invalid for input of size 65536
```

This typically means `tp_grain_size` is incompatible with the weight dimension. Ensure that `weight_dim % tp_size == 0` and that the result is a multiple of `tp_grain_size`.

**2. NCCL timeout during all-reduce**

```
RuntimeError: NCCL error in: /path/to/net.cc:123
NCCL error: unhandled system error
```

Check that all TP ranks can communicate, and increase the NCCL timeout:

```bash
export NCCL_COMM_BLOCKING=1
export NCCL_LAUNCH_MODE=PARALLEL
```

**3. ZeRO stage 3 incompatibility**

```
AssertionError: Tensor parallelism is not compatible with ZeRO stage 3
```

Use ZeRO stage 2 or lower with tensor parallelism. ZeRO-3 has its own parameter partitioning mechanism that conflicts with TP.

**4. Incorrect results with tp_overlap_comm**

When using communication overlap, ensure that no operations depend on the all-reduce output before it completes. The overlap mechanism handles this automatically for supported models but may not work for custom architectures.

**5. OOM during weight redistribution**

When converting a single-GPU checkpoint to a TP checkpoint, the full model must fit in CPU memory. For very large models, use `save_mp_checkpoint_path` to save directly in the TP format.
