# FlashAttention Training Framework Reference

This document provides comprehensive reference documentation for the training framework included with FlashAttention. The framework demonstrates how to train transformer models end-to-end using FlashAttention's optimized components.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Configuration System](#configuration-system)
4. [Data Modules](#data-modules)
5. [Model Configurations](#model-configurations)
6. [Optimizer and Scheduler](#optimizer-and-scheduler)
7. [Distributed Training](#distributed-training)
8. [Callbacks](#callbacks)
9. [Metrics](#metrics)
10. [Training Commands](#training-commands)
11. [Performance Benchmarks](#performance-benchmarks)
12. [Advanced Usage](#advanced-usage)

---

## Overview

The FlashAttention training framework provides end-to-end training scripts for GPT-2 and GPT-3 style models, demonstrating:

- **3-5x speedup** over baseline HuggingFace implementations
- Up to **189 TFLOPs/sec per A100** (60.6% model FLOPs utilization)
- No activation checkpointing needed due to FlashAttention's memory efficiency
- Model-agnostic and task-agnostic training code

### Key Technologies

- **Hydra**: Configuration management with composable YAML configs
- **PyTorch Lightning**: Training loop abstraction
- **Weights & Biases (Wandb)**: Experiment logging
- **FlashAttention**: All optimized components

### Design Goals

1. **Performance**: Optimize for speed and memory, especially on single-node (8x A100)
2. **Flexibility**: Provide composable building blocks that can be used independently
3. **Reproducibility**: Configuration-driven experiments with full parameter logging

---

## Project Structure

```
training/
    README.md                   # Documentation
    Dockerfile                  # Docker environment
    run.py                      # Entry point
    configs/                    # Hydra configuration files
        config.yaml             # Main config
        datamodule/             # Data configurations
            openwebtext.yaml
            thepile.yaml
        model/                  # Model configurations
            gpt2.yaml
            gpt2-hf.yaml
            gpt2model/
        optimizer/              # Optimizer configurations
            adamw.yaml
            adamw-apex.yaml
            adamw-zero.yaml
            ...
        scheduler/              # Learning rate schedules
            cosine-warmup.yaml
            invsqrt.yaml
            ...
        trainer/                # Trainer configurations
            default.yaml
            ddp.yaml
            debug.yaml
        experiment/             # Experiment presets
            owt/                # OpenWebText experiments
                gpt2s-flash.yaml
                gpt2m-flash.yaml
                ...
            pile/               # The Pile experiments
                gpt3s-flash.yaml
                gpt3xl-flash.yaml
                ...
        callbacks/              # Callback configurations
        metrics/                # Metric configurations
        logger/                 # Logging configurations
        task/                   # Task configurations
        mode/                   # Run mode presets
    src/
        train.py                # Lightning training module
        eval.py                 # Evaluation module
        datamodules/            # Data loading
            language_modeling_hf.py
            fault_tolerant_sampler.py
            datasets/
                lm_dataset.py
                detokenizer.py
        metrics/                # Custom metrics
            accuracy.py
            perplexity.py
            num_tokens.py
        optim/                  # Optimizer utilities
            timm_lr_scheduler.py
            param_grouping.py
        callbacks/              # Training callbacks
            speed_monitor.py
            gpu_affinity.py
            model_checkpoint.py
            ...
        tasks/                  # Task definitions
            seq.py
        utils/                  # Utilities
            distributed.py
            ddp_zero1.py
            ddp_zero2.py
            checkpoint.py
            ema.py
            flops.py
            utils.py
    tests/
        datamodules/
            test_language_modeling_hf.py
```

---

## Configuration System

The framework uses Hydra for hierarchical configuration management. Configs are composed from multiple YAML files.

### Main Config (`configs/config.yaml`)

```yaml
defaults:
  - _self_
  - trainer: default
  - optimizer: adamw
  - scheduler: null
  - task: sequence-model
  - model: null
  - datamodule: null
  - callbacks: default
  - metrics: null
  - logger: null
  - mode: default
  - experiment: null
  - hparams_search: null
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

work_dir: ${hydra:runtime.cwd}
data_dir: ${work_dir}/data/
print_config: True
ignore_warnings: True
test_after_training: True
resume: False
seed: null
name: null
```

### Configuration Composition

Hydra composes configurations using the `defaults` list. Each experiment config overrides specific defaults:

```yaml
# configs/experiment/owt/gpt2s-flash.yaml
defaults:
  - override /model: gpt2
  - override /datamodule: openwebtext

model:
  config:
    n_embd: 768
    n_head: 12
    n_layer: 12
    use_flash_attn: true
    fused_mlp: true
    fused_bias_fc: true
    fused_dropout_add_ln: true
```

### Override Syntax

```bash
# Override any config value from command line
python run.py experiment=owt/gpt2s-flash trainer.devices=8
python run.py optimizer.lr=6e-4 datamodule.batch_size=4
python run.py trainer.precision=bf16
```

---

## Data Modules

### LMDataModule

**File:** `src/datamodules/language_modeling_hf.py`

Handles loading, tokenization, and batching of language modeling datasets.

```python
class LMDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name,
        dataset_config_name=None,
        tokenizer_name="gpt2",
        cache_dir=None,
        max_length=1024,
        val_ratio=0.0005,
        val_split_seed=2357,
        add_eos=True,
        batch_size=8,
        batch_size_eval=None,
        num_workers=32,
        shuffle=True,
        pin_memory=True,
    )
```

**Parameters:**
- `dataset_name` (str): HuggingFace dataset name (e.g., `"openwebtext"`, `"the_pile"`)
- `tokenizer_name` (str): Tokenizer name (default: `"gpt2"`)
- `cache_dir` (str): Directory for cached tokenized datasets
- `max_length` (int): Maximum sequence length (default: 1024)
- `val_ratio` (float): Fraction of training data for validation (default: 0.0005)
- `val_split_seed` (int): Random seed for train/val split
- `add_eos` (bool): Add EOS token at end of each document
- `batch_size` (int): Per-GPU batch size
- `num_workers` (int): Number of workers for preprocessing

### OpenWebText Configuration

```yaml
# configs/datamodule/openwebtext.yaml
_target_: src.datamodules.language_modeling_hf.LMDataModule
dataset_name: openwebtext
tokenizer_name: gpt2
cache_dir: ${oc.env:DATA_DIR,${data_dir}}/openwebtext/cache
max_length: 1024
val_ratio: 0.0005
val_split_seed: 2357
add_eos: True
batch_size: 8
batch_size_eval: ${eval:${.batch_size} * 2}
num_workers: 32
shuffle: True
pin_memory: True
__train_len: ${div_up:9035582198, ${.max_length}}
```

### The Pile Configuration

```yaml
# configs/datamodule/thepile.yaml
_target_: src.datamodules.language_modeling_hf.LMDataModule
dataset_name: the_pile
tokenizer_name: gpt2
cache_dir: ${oc.env:DATA_DIR,${data_dir}}/the_pile/cache
max_length: 2048
```

### Dataset Preparation

Tokenization is done once and cached to disk:

```bash
# OpenWebText (~1 hour on 64 cores, ~17GB)
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "openwebtext"

# The Pile (~20 hours on 64 cores, ~699GB)
pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "pile"
```

### Fault-Tolerant Sampling

**File:** `src/datamodules/fault_tolerant_sampler.py`

Provides a sampler that can resume from a specific state after a failure, ensuring no data is missed or repeated.

---

## Model Configurations

### GPT-2 Variants

| Model | Parameters | `n_embd` | `n_layer` | `n_head` | Sequence Length |
|-------|-----------|---------|---------|--------|----------------|
| GPT-2 Small | 125M | 768 | 12 | 12 | 1024 |
| GPT-2 Medium | 355M | 1024 | 24 | 16 | 1024 |
| GPT-2 Large | 760M | 1280 | 36 | 20 | 1024 |
| GPT-2 XL | 1.6B | 1600 | 48 | 25 | 1024 |

### GPT-3 Variants

| Model | Parameters | `n_embd` | `n_layer` | `n_head` | Sequence Length | Batch Size (tokens) |
|-------|-----------|---------|---------|--------|----------------|-------------------|
| GPT-3 Small | 125M | 768 | 12 | 12 | 2048 | 512K |
| GPT-3 Medium | 355M | 1024 | 24 | 16 | 2048 | 512K |
| GPT-3 Large | 760M | 1536 | 24 | 16 | 2048 | 512K |
| GPT-3 XL | 1.3B | 2048 | 24 | 16 | 2048 | 1M |
| GPT-3 2.7B | 2.7B | 2560 | 32 | 32 | 2048 | 1M |

### Flash-Optimized Configuration

Example GPT-3 small config with FlashAttention optimizations:

```yaml
# configs/experiment/pile/gpt3s-flash.yaml
defaults:
  - override /model: gpt2

model:
  config:
    vocab_size: 50257
    n_positions: 2048
    n_embd: 768
    n_layer: 12
    n_head: 12
    n_inner: 3072
    activation_function: gelu_new
    resid_pdrop: 0.0
    embd_pdrop: 0.0
    attn_pdrop: 0.0
    # FlashAttention optimizations
    scale_attn_weights: true
    scale_attn_by_inverse_layer_idx: true
    use_flash_attn: true
    fused_mlp: true
    fused_bias_fc: true
    fused_dropout_add_ln: true
    pad_vocab_size_multiple: 8
    rotary_emb_fraction: 0.0
```

### Rotary Embedding Variant

```bash
# Train GPT-3 with rotary embeddings
python run.py experiment=pile/gpt3s-flash-rotary trainer.devices=8
```

Rotary variants set `rotary_emb_fraction=0.5` (or `1.0` for full rotary).

### Head Dimension 128 Variant

For better efficiency on A100, the GPT-3 2.7B model can use head dimension 128 instead of the default 80:

```bash
python run.py experiment=pile/gpt3-2.7B-flash-hdim128 trainer.devices=8
```

---

## Optimizer and Scheduler

### AdamW

```yaml
# configs/optimizer/adamw.yaml
_target_: torch.optim.AdamW
lr: 6e-4
weight_decay: 0.1
betas: [0.9, 0.95]
eps: 1e-8
```

### FusedAdamW (Apex)

```yaml
# configs/optimizer/adamw-apex.yaml
_target_: apex.optimizers.FusedAdam
lr: 6e-4
weight_decay: 0.1
betas: [0.9, 0.95]
adam_w_mode: true
```

### DeepSpeed ZeRO Optimizers

```yaml
# configs/optimizer/adamw-zero.yaml
_target_: deepspeed.ops.adam.DeepSpeedCPUAdam
lr: 6e-4
```

### Learning Rate Schedulers

#### Cosine Warmup

```yaml
# configs/scheduler/cosine-warmup.yaml
_target_: src.optim.timm_lr_scheduler.CosineLRScheduler
t_in_epochs: false
warmup_lr_init: 1e-6
warmup_t: 2000
lr_min: 6e-5
```

#### Inverse Square Root

```yaml
# configs/scheduler/invsqrt.yaml
_target_: torch.optim.lr_scheduler.LambdaLR
```

### Parameter Grouping

**File:** `src/optim/param_grouping.py`

Separates parameters into groups with different weight decay settings:

- **No weight decay**: Biases, LayerNorm parameters, embeddings
- **Weight decay**: Linear layer weights, convolution weights

```python
def get_param_groups(model, weight_decay=0.1, no_weight_decay_list=None)
```

---

## Distributed Training

### DDP Configuration

```yaml
# configs/trainer/ddp.yaml
defaults:
  - default.yaml
accelerator: gpu
devices: 4
strategy: ddp
```

### Multi-Node Training

```bash
# On each node
python run.py experiment=pile/gpt3s-flash \
    trainer.devices=8 \
    trainer.num_nodes=4 \
    trainer.strategy=ddp
```

### ZeRO Stage 1

**File:** `src/utils/ddp_zero1.py`

Wraps model parameters with DeepSpeed ZeRO Stage 1 for memory-efficient training.

### ZeRO Stage 2

**File:** `src/utils/ddp_zero2.py`

Full ZeRO Stage 2 integration with optimizer state partitioning.

### GPU Affinity

**File:** `src/callbacks/gpu_affinity.py` and `src/utils/gpu_affinity.py`

Sets CPU affinity for each GPU process to optimize NUMA locality.

### DDP Communication Hooks

**File:** `src/distributed/ddp_comm_hooks.py`

Custom communication hooks for gradient compression during distributed training.

---

## Callbacks

### Speed Monitor

**File:** `src/callbacks/speed_monitor.py`

Monitors and logs training throughput (tokens/sec, TFLOPs/sec).

```yaml
# Enable verbose speed monitoring
+callbacks.speed_monitor.verbose=True
```

### Model Checkpoint

**File:** `src/callbacks/model_checkpoint.py`

Saves model checkpoints based on validation metrics.

### EMA (Exponential Moving Average)

**File:** `src/callbacks/ema.py`

Maintains an EMA copy of model parameters for more stable evaluation.

### Norm Monitor

**File:** `src/callbacks/norm_monitor.py`

Logs gradient and weight norms during training for debugging.

### Causality Monitor

**File:** `src/callbacks/causality_monitor.py`

Verifies that causal attention masks are working correctly.

### FLOP Counter

**File:** `src/callbacks/flop_count.py`

Counts and logs model FLOPs for performance analysis.

### Parameter Logging

**File:** `src/callbacks/params_log.py`

Logs all model parameters and their gradients.

### Wandb Callbacks

**File:** `src/callbacks/wandb_callbacks.py`

Integration with Weights & Biases for experiment tracking.

---

## Metrics

### Perplexity

**File:** `src/metrics/perplexity.py`

Computes perplexity from cross-entropy loss.

### Accuracy

**File:** `src/metrics/accuracy.py`

Token-level prediction accuracy (top-1 and top-5).

### Num Tokens

**File:** `src/metrics/num_tokens.py`

Tracks total number of training tokens processed.

---

## Training Commands

### GPT-2 on OpenWebText

```bash
# GPT-2 Small (125M)
python run.py experiment=owt/gpt2s-flash trainer.devices=8

# GPT-2 Medium (355M)
python run.py experiment=owt/gpt2m-flash trainer.devices=8

# GPT-2 Large (760M)
python run.py experiment=owt/gpt2l-flash trainer.devices=8

# GPT-2 XL (1.6B)
python run.py experiment=owt/gpt2xl-flash trainer.devices=8
```

### GPT-3 on The Pile

```bash
# GPT-3 Small (125M)
python run.py experiment=pile/gpt3s-flash trainer.devices=8

# GPT-3 Medium (355M)
python run.py experiment=pile/gpt3m-flash trainer.devices=8

# GPT-3 Large (760M)
python run.py experiment=pile/gpt3l-flash trainer.devices=8

# GPT-3 XL (1.3B)
python run.py experiment=pile/gpt3xl-flash trainer.devices=8

# GPT-3 2.7B
python run.py experiment=pile/gpt3-2.7B-flash-hdim128 trainer.devices=8
```

### Training Options

**Gradient accumulation** (adjust per-device batch size):
```bash
python run.py experiment=owt/gpt2s-flash trainer.devices=8 datamodule.batch_size=4
```

**Mixed precision**:
```bash
# Use bf16 instead of fp16
python run.py experiment=owt/gpt2s-flash trainer.devices=8 trainer.precision=bf16
```

**Rotary embeddings**:
```bash
python run.py experiment=pile/gpt3s-flash-rotary trainer.devices=8
```

**Resume training**:
```bash
python run.py experiment=pile/gpt3s-flash trainer.devices=8 name=pile-gpt3s-flash resume=True
```

**Speed benchmarking**:
```bash
python run.py experiment=pile/gpt3s-flash trainer.devices=8 +callbacks.speed_monitor.verbose=True
```

**Multi-node**:
```bash
python run.py experiment=pile/gpt3-2.7B-flash trainer.devices=8 trainer.num_nodes=2
```

### Running Modes

```bash
# Debug mode (few batches, no logging)
python run.py experiment=pile/gpt3s-flash mode=debug

# Smoke test (quick validation)
python run.py experiment=pile/gpt3s-flash mode=smoke

# Profile mode
python run.py experiment=pile/gpt3s-flash mode=profile
```

---

## Performance Benchmarks

### GPT-2 Training (Sequence Length 1024)

The FlashAttention implementation achieves **3-4x speedup** over HuggingFace baseline on 8x A100 80GB.

### GPT-3 Training (Sequence Length 2048)

The FlashAttention implementation achieves **3-5x speedup** over HuggingFace baseline on 8x A100 80GB.

### Throughput on 8x A100 80GB (GPT-3 on The Pile)

| Model | Batch Size (tokens) | Throughput (tokens/sec) | Hours / 1B tokens |
|-------|-------------------|----------------------|-------------------|
| GPT-3 125M | 0.5M | 1,310,000 | 0.21 |
| GPT-3 355M | 0.5M | 503,000 | 0.55 |
| GPT-3 760M | 0.5M | 245,000 | 1.13 |
| GPT-3 1.3B | 1M | 169,000 | 1.64 |
| GPT-3 2.7B | 1M | 85,000 | 3.27 |

**Example**: Training GPT-3 1.3B on 26B tokens (Chinchilla-optimal) takes ~43 hours on 8x A100.

### FLOPs Calculation

Model FLOPs are calculated using the Megatron-LM formula (Section 5.1), scaled by 3/4 to get model FLOPs (without activation checkpointing):

```
FLOPs per token ≈ 6 * N_params (for forward + backward)
```

**Model FLOPs Utilization (MFU):**
```
MFU = (6 * N_params * tokens/sec) / (GPU_count * GPU_peak_FLOPs)
```

On A100 80GB SXM4 (400W) with NVLink: achieves up to **189 TFLOPs/sec** = **60.6% MFU**.

---

## Advanced Usage

### Custom Experiments

Create a new experiment config:

```yaml
# configs/experiment/custom/my-experiment.yaml
defaults:
  - override /model: gpt2
  - override /datamodule: openwebtext

model:
  config:
    n_embd: 1024
    n_head: 16
    n_layer: 24
    use_flash_attn: true
    fused_mlp: true
    fused_bias_fc: true
    fused_dropout_add_ln: true
    rotary_emb_fraction: 0.5

optimizer:
  lr: 3e-4
  weight_decay: 0.1

scheduler:
  _target_: src.optim.timm_lr_scheduler.CosineLRScheduler
  warmup_t: 1000
```

```bash
python run.py experiment=custom/my-experiment trainer.devices=8
```

### Using FlashAttention Models Standalone

```python
from transformers import GPT2Config
from flash_attn.models.gpt import GPTLMHeadModel

# Create model
config = GPT2Config(
    vocab_size=50257,
    n_positions=2048,
    n_embd=2048,
    n_layer=24,
    n_head=16,
    n_inner=8192,
    activation_function="gelu_new",
    scale_attn_by_inverse_layer_idx=True,
    rotary_emb_fraction=0.5,
    use_flash_attn=True,
    fused_mlp=True,
    fused_bias_fc=True,
    fused_dropout_add_ln=True,
    pad_vocab_size_multiple=8,
)

model = GPTLMHeadModel(config).cuda().half()

# Training step
input_ids = torch.randint(0, 50257, (4, 2048), device="cuda")
output = model(input_ids)
loss = cross_entropy(output.logits.view(-1, 50257), input_ids.view(-1))
loss.backward()
```

### Loading Pretrained Models

```python
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.llama import config_from_hf_checkpoint, llama_config_to_gpt2_config

# Load LLaMA from HuggingFace
llama_config = config_from_hf_checkpoint("/path/to/models", "Llama-2-7b")
gpt2_config = llama_config_to_gpt2_config(llama_config)
model = GPTLMHeadModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    gpt2_config,
    device="cuda",
    dtype=torch.float16,
)

# Generate text
input_ids = torch.tensor([[1, 2, 3]], device="cuda")
for _ in range(100):
    output = model(input_ids, num_last_tokens=1)
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

### Distributed Training Utilities

```python
from flash_attn.utils.distributed import all_gather, all_reduce, reduce_scatter

# Tensor parallel communication
process_group = dist.new_group(backend="nccl")

# All-gather for sequence parallel
full_tensor = all_gather(local_tensor, process_group)

# Reduce-scatter for sequence parallel
local_result = reduce_scatter(full_tensor, process_group)
```
