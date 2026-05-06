# Data Pipeline and Efficiency

## Overview

DeepSpeed provides a comprehensive data pipeline and efficiency framework that goes beyond standard data loading to include curriculum learning, data routing strategies, advanced data sampling methods, and data efficiency optimizations. These features are designed to accelerate training convergence, reduce computational waste, and enable intelligent data utilization patterns. The data pipeline infrastructure integrates deeply with the DeepSpeed engine, allowing fine-grained control over how training data is selected, ordered, and processed across distributed training runs.

The DeepSpeed Data Efficiency system was introduced to address the observation that not all training samples contribute equally to model learning. By dynamically adjusting data selection strategies during training -- such as starting with easier sequences and progressively increasing difficulty -- models can converge faster while using fewer total compute resources. This is formalized in the DeepSpeed Data Efficiency research, which demonstrated up to 2x training speedup on common NLP benchmarks.

---

## Source Code Organization

```
deepspeed/runtime/data_pipeline/
    __init__.py
    config.py                      # DataPipelineConfig class
    constants.py                   # Constants for data pipeline configuration
    data_routing/
        __init__.py
        data_routing_helper.py     # Data routing logic and strategies
    data_sampling/
        __init__.py
        data_sampling_helper.py    # Data sampling strategies

deepspeed/runtime/data_efficiency/
    __init__.py
    curricula_scheduler.py         # Curriculum learning scheduler
    data_efficiency.py             # Core data efficiency logic
```

---

## Data Efficiency Configuration

### Core API Functions

DeepSpeed exposes several top-level functions for checking and retrieving data efficiency configuration:

```python
# deepspeed/runtime/data_efficiency.py

def get_data_efficiency_enabled(deepspeed_config):
    """Check if data efficiency is enabled in the DeepSpeed config.

    Args:
        deepspeed_config (dict): The full DeepSpeed configuration dictionary.

    Returns:
        bool: True if data efficiency is enabled, False otherwise.
    """
    ...

def get_data_efficiency_config(deepspeed_config):
    """Extract data efficiency configuration from the DeepSpeed config.

    Args:
        deepspeed_config (dict): The full DeepSpeed configuration dictionary.

    Returns:
        dict: Data efficiency configuration sub-dictionary. Empty dict if
              data efficiency is not configured.
    """
    ...
```

### Configuration Structure

Data efficiency is configured under the `data_efficiency` top-level key in the DeepSpeed JSON configuration:

```json
{
    "data_efficiency": {
        "enabled": true,
        "data_routing": {
            "enabled": true,
            "random_seed": 42,
            "routing_algorithm": "hash_based"
        },
        "data_sampling": {
            "enabled": true,
            "num_epochs": 100,
            "num_workers": 0,
            "curriculum_learning": {
                "enabled": true,
                "curriculum_type": "seqlen",
                "curriculum_num_eligible_epochs": 1,
                "curriculum_start_epoch": 0,
                "curriculum_max_difficulty": 1.0,
                "curriculum_min_difficulty": 0.0,
                "curriculum_max_seqlen": 2048,
                "curriculum_seqlen_boundaries": [128, 256, 512, 1024, 2048]
            }
        }
    }
}
```

### DataPipelineConfig (config.py)

```python
# deepspeed/runtime/data_pipeline/config.py

class DataPipelineConfig:
    """Configuration for DeepSpeed data pipeline features.

    Attributes:
        data_efficiency_enabled (bool): Whether any data efficiency feature is active.
        data_routing_enabled (bool): Whether data routing is active.
        data_sampling_enabled (bool): Whether data sampling is active.
        curriculum_learning_enabled (bool): Whether curriculum learning is active.
        data_routing_config (dict): Configuration for data routing.
        data_sampling_config (dict): Configuration for data sampling.
        curriculum_learning_config (dict): Configuration for curriculum learning.
    """
```

### Constants (constants.py)

```python
# deepspeed/runtime/data_pipeline/constants.py

# Top-level keys
DATA_EFFICIENCY = "data_efficiency"
DATA_EFFICIENCY_ENABLED = "enabled"

# Data routing keys
DATA_ROUTING = "data_routing"
DATA_ROUTING_ENABLED = "enabled"
DATA_ROUTING_RANDOM_SEED = "random_seed"
DATA_ROUTING_ALGORITHM = "routing_algorithm"

# Data sampling keys
DATA_SAMPLING = "data_sampling"
DATA_SAMPLING_ENABLED = "enabled"

# Curriculum learning keys
CURRICULUM_LEARNING = "curriculum_learning"
CURRICULUM_ENABLED = "enabled"
CURRICULUM_TYPE = "curriculum_type"
CURRICULUM_NUM_ELIGIBLE_EPOCHS = "curriculum_num_eligible_epochs"
CURRICULUM_START_EPOCH = "curriculum_start_epoch"
CURRICULUM_MAX_DIFFICULTY = "curriculum_max_difficulty"
CURRICULUM_MIN_DIFFICULTY = "curriculum_min_difficulty"
CURRICULUM_MAX_SEQLEN = "curriculum_max_seqlen"
CURRICULUM_SEQLEN_BOUNDARIES = "curriculum_seqlen_boundaries"
```

---

## Curriculum Learning

### Overview

Curriculum learning is a training strategy inspired by human learning, where the model is trained on progressively harder examples over the course of training. In DeepSpeed, curriculum learning is implemented primarily through sequence length scheduling: the model starts training on shorter sequences and gradually increases sequence length over epochs.

The rationale is twofold:
1. **Faster early convergence**: Shorter sequences require less computation per step, enabling more parameter updates per unit time in early training when the model benefits most from frequent updates.
2. **Better final accuracy**: Starting with simpler patterns (shorter context) helps the model learn fundamental patterns before tackling complex long-range dependencies.

### Legacy API Functions

DeepSpeed provides legacy API functions for backward compatibility:

```python
# deepspeed/runtime/data_efficiency.py

def get_curriculum_enabled_legacy(deepspeed_config):
    """Check if curriculum learning is enabled (legacy API).

    Inspects the DeepSpeed config and returns whether curriculum
    learning is active.

    Args:
        deepspeed_config (dict): The full DeepSpeed configuration dictionary.

    Returns:
        bool: True if curriculum learning is enabled.
    """
    ...

def get_curriculum_params_legacy(deepspeed_config):
    """Get curriculum learning parameters (legacy API).

    Args:
        deepspeed_config (dict): The full DeepSpeed configuration dictionary.

    Returns:
        dict: Curriculum learning parameters including:
            - curriculum_enabled: bool
            - curriculum_num_eligible_epochs: int
            - curriculum_start_epoch: int
            - curriculum_max_difficulty: float
            - curriculum_type: str
            - curriculum_min_difficulty: float
            - curriculum_max_seqlen: int
    """
    ...
```

### Curriculum Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curriculum_enabled` | bool | `false` | Whether curriculum learning is enabled. |
| `curriculum_type` | str | `"seqlen"` | Type of curriculum. Currently only `"seqlen"` is supported, which adjusts sequence length. |
| `curriculum_num_eligible_epochs` | int | `1` | Number of epochs at each difficulty level before progressing to the next. Controls the pace of curriculum progression. |
| `curriculum_start_epoch` | int | `0` | The epoch at which curriculum learning begins. Epochs before this value train at full difficulty. |
| `curriculum_max_difficulty` | float | `1.0` | Maximum difficulty level, expressed as a fraction from 0.0 to 1.0. 1.0 means the curriculum will progress to the maximum sequence length. |
| `curriculum_min_difficulty` | float | `0.0` | Minimum difficulty level at the start of curriculum learning. 0.0 means start from the shortest possible sequences. |
| `curriculum_max_seqlen` | int | `2048` | Maximum sequence length at full difficulty. The curriculum will gradually increase sequence length up to this value. |
| `curriculum_seqlen_boundaries` | list[int] | None | Explicit sequence length boundaries for curriculum stages. If provided, the curriculum progresses through these exact lengths rather than interpolating linearly. |

### How Curriculum Learning Works

The curriculum learning system operates in the following sequence:

1. **Epoch 0 to `curriculum_start_epoch`**: No curriculum applied. Training uses full sequence length.

2. **From `curriculum_start_epoch`**: Curriculum begins at `curriculum_min_difficulty`.

3. **Progressive difficulty increase**: Every `curriculum_num_eligible_epochs` epochs, the difficulty level increases. The current difficulty is interpolated between `curriculum_min_difficulty` and `curriculum_max_difficulty` based on the current epoch.

4. **Sequence length mapping**: The difficulty level maps to a sequence length between the minimum possible length and `curriculum_max_seqlen`.

5. **Data truncation**: Training data sequences longer than the current curriculum length are truncated (or filtered, depending on configuration).

### Difficulty Computation

```python
def compute_curriculum_seqlen(
    current_epoch,
    curriculum_start_epoch,
    curriculum_num_eligible_epochs,
    curriculum_min_difficulty,
    curriculum_max_difficulty,
    curriculum_max_seqlen,
    curriculum_seqlen_boundaries=None,
):
    """Compute the current sequence length for curriculum learning.

    The difficulty progresses linearly from curriculum_min_difficulty to
    curriculum_max_difficulty over the curriculum period. The number of
    curriculum steps is determined by curriculum_num_eligible_epochs.

    Returns:
        int: The maximum sequence length for the current epoch.
    """
    if current_epoch < curriculum_start_epoch:
        return curriculum_max_seqlen

    # Compute elapsed curriculum epochs
    elapsed = current_epoch - curriculum_start_epoch

    if curriculum_seqlen_boundaries is not None:
        # Use explicit boundaries
        step = elapsed // curriculum_num_eligible_epochs
        step = min(step, len(curriculum_seqlen_boundaries) - 1)
        return curriculum_seqlen_boundaries[step]
    else:
        # Linear interpolation
        total_curriculum_epochs = (
            (curriculum_max_difficulty - curriculum_min_difficulty)
            * curriculum_num_eligible_epochs
        )
        difficulty = curriculum_min_difficulty + (
            (curriculum_max_difficulty - curriculum_min_difficulty)
            * min(elapsed, total_curriculum_epochs)
            / total_curriculum_epochs
        )
        seqlen = int(difficulty * curriculum_max_seqlen)
        return max(1, seqlen)
```

### Curriculum Learning with Sequence Length Boundaries

```json
{
    "data_efficiency": {
        "enabled": true,
        "data_sampling": {
            "enabled": true,
            "curriculum_learning": {
                "enabled": true,
                "curriculum_type": "seqlen",
                "curriculum_num_eligible_epochs": 2,
                "curriculum_start_epoch": 0,
                "curriculum_max_difficulty": 1.0,
                "curriculum_min_difficulty": 0.0,
                "curriculum_max_seqlen": 4096,
                "curriculum_seqlen_boundaries": [128, 256, 512, 1024, 2048, 4096]
            }
        }
    }
}
```

With the above configuration:
- Epochs 0-1: Sequence length = 128
- Epochs 2-3: Sequence length = 256
- Epochs 4-5: Sequence length = 512
- Epochs 6-7: Sequence length = 1024
- Epochs 8-9: Sequence length = 2048
- Epochs 10-11: Sequence length = 4096
- Epochs 12+: Sequence length = 4096 (max)

---

## Data Routing

### Overview

Data routing controls how training samples are assigned to different data-parallel ranks. In standard distributed training, each rank processes a different shard of the dataset. DeepSpeed's data routing provides more sophisticated strategies for sample-to-rank assignment.

### Data Routing Configuration

```json
{
    "data_efficiency": {
        "enabled": true,
        "data_routing": {
            "enabled": true,
            "random_seed": 42,
            "routing_algorithm": "hash_based"
        }
    }
}
```

### Data Routing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Whether data routing is enabled. |
| `random_seed` | int | `42` | Random seed for deterministic routing. Ensures reproducible data assignment across runs. |
| `routing_algorithm` | str | `"hash_based"` | Algorithm for routing samples to ranks. Options: `"hash_based"`, `"random"`, `"round_robin"`. |

### Routing Algorithms

#### Hash-Based Routing

Hash-based routing uses a deterministic hash function to assign each training sample to a specific rank. This ensures:
- Deterministic assignment: the same sample always goes to the same rank
- Load balancing: hash functions naturally distribute samples uniformly
- Cross-epoch consistency: samples stay on the same rank across epochs

```python
def hash_based_route(sample_index, num_ranks, seed=42):
    """Route a sample to a rank using hash-based assignment.

    Args:
        sample_index: Index of the training sample.
        num_ranks: Number of data-parallel ranks.
        seed: Random seed for the hash function.

    Returns:
        int: The rank index that should process this sample.
    """
    hash_val = hash((sample_index, seed))
    return hash_val % num_ranks
```

#### Random Routing

Random routing assigns samples to ranks using a random permutation that is regenerated each epoch. This provides:
- Randomized assignment that changes each epoch
- Load balancing through permutation
- Controlled randomness via seed

#### Round-Robin Routing

Round-robin routing assigns samples to ranks in sequential order: sample 0 to rank 0, sample 1 to rank 1, etc. This is the simplest strategy and matches standard PyTorch DistributedSampler behavior.

---

## Data Sampling

### Overview

Data sampling controls the order and selection of training samples. DeepSpeed extends PyTorch's sampling mechanisms with curriculum-aware sampling that adjusts sample selection based on the current curriculum difficulty level.

### Data Sampling Helper

```python
# deepspeed/runtime/data_pipeline/data_sampling/data_sampling_helper.py

class DeepSpeedDataSampler:
    """Custom data sampler that integrates with DeepSpeed's data efficiency.

    This sampler wraps PyTorch's DistributedSampler and adds support for:
    - Curriculum learning: filtering/truncating samples based on difficulty
    - Efficient batch construction: grouping similar-length sequences
    - Distributed consistency: ensuring all ranks see consistent curriculum state
    """

    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        seed=0,
        drop_last=False,
        curriculum_learning_enabled=False,
        curriculum_seqlen=None,
    ):
        ...

    def set_epoch(self, epoch):
        """Set the current epoch for curriculum-aware sampling.

        Updates the curriculum difficulty based on the current epoch
        and adjusts the sampling strategy accordingly.
        """
        ...

    def __iter__(self):
        """Iterate over samples, applying curriculum filtering.

        When curriculum learning is active, samples with sequence lengths
        exceeding the current curriculum threshold are skipped or truncated.
        """
        ...

    def __len__(self):
        """Return the number of samples considering curriculum filtering."""
        ...
```

---

## Integration with DeepSpeed Engine

### Automatic Data Pipeline Setup

When `data_efficiency` is enabled in the DeepSpeed configuration, the engine automatically initializes the data pipeline components:

```python
# Conceptual flow in DeepSpeed engine initialization
class DeepSpeedEngine:
    def __init__(self, ...):
        ...
        # Data efficiency initialization
        self.data_efficiency_config = get_data_efficiency_config(self.config)
        if get_data_efficiency_enabled(self.config):
            self._init_data_pipeline()

    def _init_data_pipeline(self):
        """Initialize data pipeline components based on configuration."""
        config = self.data_efficiency_config

        # Data routing
        if config.get('data_routing', {}).get('enabled', False):
            self.data_routing_helper = DataRoutingHelper(
                config['data_routing'],
                num_ranks=self.mp_world_size,
                rank=self.mp_world_rank,
            )

        # Data sampling with curriculum learning
        if config.get('data_sampling', {}).get('enabled', False):
            curriculum_config = config['data_sampling'].get('curriculum_learning', {})
            self.curriculum_scheduler = CurriculumScheduler(
                curriculum_config,
            )
```

### Training Loop Integration

```python
# Typical training loop with data efficiency
model_engine, optimizer, _, _ = deepspeed.initialize(...)

for epoch in range(num_epochs):
    # Update curriculum scheduler (adjusts difficulty for this epoch)
    if hasattr(model_engine, 'curriculum_scheduler'):
        model_engine.curriculum_scheduler.set_epoch(epoch)

    for batch in dataloader:
        # Data pipeline automatically applies curriculum filtering
        # if configured in the DeepSpeed config
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
```

---

## DeepSpeed Data Efficiency Research

### Key Findings

The DeepSpeed Data Efficiency system is based on the research paper:

**"DeepSpeed Data Efficiency: Improving Deep Learning Training Efficiency via Data Efficiency"**

The paper demonstrates that intelligent data selection and ordering can significantly reduce the computational cost of training large models:

1. **Curriculum learning (seqlen) achieves up to 2.4x speedup** on BERT pre-training. By starting with shorter sequences and progressively increasing length, the model makes more parameter updates per unit time in early training.

2. **Data routing can reduce communication overhead** by ensuring that similar samples are processed on the same rank, reducing gradient variance and enabling more aggressive gradient accumulation.

3. **Combined data efficiency techniques** (curriculum learning + data routing + data sampling) can provide additive benefits, achieving up to 3x total training speedup.

### Training Efficiency Metrics

| Technique | Task | Baseline Epochs | Efficient Epochs | Speedup |
|-----------|------|-----------------|-------------------|---------|
| Curriculum (seqlen) | BERT Pretrain | 100 | 42 | 2.4x |
| Data Routing | GPT Pretrain | 100 | 78 | 1.3x |
| Combined | BERT Pretrain | 100 | 33 | 3.0x |

---

## Configuration Examples

### Example 1: Basic Curriculum Learning for Language Model

```json
{
    "train_batch_size": 2048,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
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
        "stage": 2,
        "allgather_partitions": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "contiguous_gradients": true
    },
    "data_efficiency": {
        "enabled": true,
        "data_sampling": {
            "enabled": true,
            "curriculum_learning": {
                "enabled": true,
                "curriculum_type": "seqlen",
                "curriculum_num_eligible_epochs": 5,
                "curriculum_start_epoch": 0,
                "curriculum_max_difficulty": 1.0,
                "curriculum_min_difficulty": 0.05,
                "curriculum_max_seqlen": 2048
            }
        }
    }
}
```

### Example 2: Data Routing for Multi-Source Data

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "data_efficiency": {
        "enabled": true,
        "data_routing": {
            "enabled": true,
            "random_seed": 12345,
            "routing_algorithm": "hash_based"
        }
    }
}
```

### Example 3: Full Data Efficiency with Explicit Boundaries

```json
{
    "train_batch_size": 4096,
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
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
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e5
    },
    "data_efficiency": {
        "enabled": true,
        "data_routing": {
            "enabled": true,
            "random_seed": 42,
            "routing_algorithm": "hash_based"
        },
        "data_sampling": {
            "enabled": true,
            "curriculum_learning": {
                "enabled": true,
                "curriculum_type": "seqlen",
                "curriculum_num_eligible_epochs": 3,
                "curriculum_start_epoch": 0,
                "curriculum_max_difficulty": 1.0,
                "curriculum_min_difficulty": 0.0,
                "curriculum_max_seqlen": 8192,
                "curriculum_seqlen_boundaries": [128, 256, 512, 1024, 2048, 4096, 8192]
            }
        }
    }
}
```

### Example 4: Curriculum Learning with ZeRO-3 for Large Models

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1.5e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 2e8,
        "stage3_prefetch_bucket_size": 2e8,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "data_efficiency": {
        "enabled": true,
        "data_sampling": {
            "enabled": true,
            "curriculum_learning": {
                "enabled": true,
                "curriculum_type": "seqlen",
                "curriculum_num_eligible_epochs": 10,
                "curriculum_start_epoch": 0,
                "curriculum_max_difficulty": 1.0,
                "curriculum_min_difficulty": 0.01,
                "curriculum_max_seqlen": 4096
            }
        }
    },
    "gradient_clipping": 1.0
}
```

---

## Programmatic API

### Checking Data Efficiency at Runtime

```python
import deepspeed

def setup_training():
    ds_config = {
        "data_efficiency": {
            "enabled": True,
            "data_sampling": {
                "enabled": True,
                "curriculum_learning": {
                    "enabled": True,
                    "curriculum_type": "seqlen",
                    "curriculum_max_seqlen": 2048,
                }
            }
        },
        # ... other config ...
    }

    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # Check if data efficiency is active
    from deepspeed.runtime.data_efficiency import (
        get_data_efficiency_enabled,
        get_data_efficiency_config,
    )

    if get_data_efficiency_enabled(ds_config):
        eff_config = get_data_efficiency_config(ds_config)
        print(f"Data routing enabled: {eff_config.get('data_routing', {}).get('enabled', False)}")
        print(f"Curriculum learning: {eff_config.get('data_sampling', {}).get('curriculum_learning', {}).get('enabled', False)}")
```

### Using Legacy Curriculum API

```python
from deepspeed.runtime.data_efficiency import (
    get_curriculum_enabled_legacy,
    get_curriculum_params_legacy,
)

ds_config = {
    "data_efficiency": {
        "enabled": True,
        "data_sampling": {
            "enabled": True,
            "curriculum_learning": {
                "enabled": True,
                "curriculum_type": "seqlen",
                "curriculum_num_eligible_epochs": 5,
                "curriculum_max_seqlen": 2048,
            }
        }
    }
}

# Legacy API
if get_curriculum_enabled_legacy(ds_config):
    params = get_curriculum_params_legacy(ds_config)
    print(f"Curriculum type: {params['curriculum_type']}")
    print(f"Max sequence length: {params['curriculum_max_seqlen']}")
    print(f"Start epoch: {params['curriculum_start_epoch']}")
```

---

## Monitoring Data Pipeline

### Logging Curriculum Progress

DeepSpeed automatically logs curriculum learning progress when `wall_clock_breakdown` is enabled:

```
[2024-01-15 10:30:45,123] [INFO] [logging.py:log:63] Data Efficiency] curriculum_learning] current_epoch=5, current_seqlen=512, max_seqlen=2048, difficulty=0.25
[2024-01-15 10:35:22,456] [INFO] [logging.py:log:63] Data Efficiency] curriculum_learning] current_epoch=10, current_seqlen=1024, max_seqlen=2048, difficulty=0.50
[2024-01-15 10:45:18,789] [INFO] [logging.py:log:63] Data Efficiency] curriculum_learning] current_epoch=20, current_seqlen=2048, max_seqlen=2048, difficulty=1.00
```

### TensorBoard Integration

```python
# Log curriculum metrics to TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(num_epochs):
    if hasattr(model_engine, 'curriculum_scheduler'):
        seqlen = model_engine.curriculum_scheduler.get_current_seqlen(epoch)
        difficulty = model_engine.curriculum_scheduler.get_current_difficulty(epoch)
        writer.add_scalar('data_efficiency/curriculum_seqlen', seqlen, epoch)
        writer.add_scalar('data_efficiency/curriculum_difficulty', difficulty, epoch)
```

---

## Best Practices

### When to Use Curriculum Learning

1. **Pre-training large language models**: Curriculum learning provides the most benefit during pre-training when training for many epochs over large corpora. Start with short sequences to quickly learn basic token relationships.

2. **Long-context models**: When the target sequence length is very large (4K+ tokens), curriculum learning can significantly reduce early training cost. Without curriculum, the first few epochs waste computation processing long sequences the model cannot yet model well.

3. **Fine-tuning with limited data**: Curriculum learning is less beneficial for fine-tuning with few epochs. The overhead of curriculum progression may not be amortized.

### Choosing `curriculum_num_eligible_epochs`

- **Small values (1-3)**: Faster curriculum progression. Good for shorter training runs or when the model converges quickly on easy data.
- **Medium values (5-10)**: Balanced progression. Good for standard pre-training runs of 50-200 epochs.
- **Large values (10-20)**: Slow progression. Good for very long training runs where you want the model to thoroughly learn each difficulty level.

### Choosing `curriculum_min_difficulty`

- **0.0**: Start from the shortest possible sequences. Maximum speedup but may lose early exposure to long-range patterns.
- **0.05-0.1**: Start from 5-10% of max sequence length. Good balance between speedup and early pattern exposure.
- **0.2+**: Start from a more substantial sequence length. Less speedup but more similar to standard training.

### Data Routing Best Practices

- Use `hash_based` routing for reproducible experiments.
- Use `random_seed` to ensure cross-run consistency.
- Data routing is most beneficial when training data has heterogeneous properties (varying lengths, domains, difficulty levels).

---

## Troubleshooting

### Common Issues

1. **Curriculum learning not activating**: Ensure `data_efficiency.enabled = true` AND `data_sampling.enabled = true` AND `curriculum_learning.enabled = true`. All three flags must be set.

2. **Unexpected sequence lengths**: Check that `curriculum_max_seqlen` matches the actual maximum sequence length in your data. If set too high, the curriculum may progress to a length that exceeds your data.

3. **Slower convergence**: If curriculum learning causes the model to converge to a worse final accuracy, try increasing `curriculum_num_eligible_epochs` to give the model more time at each difficulty level, or increase `curriculum_min_difficulty` to start from a higher base.

4. **Data routing errors with custom datasets**: Ensure your dataset implements `__len__` and `__getitem__` properly. Data routing relies on consistent indexing.

5. **Legacy API deprecation warnings**: Migrate to the new `data_efficiency` configuration structure. The legacy `curriculum_learning` top-level key is deprecated in favor of the nested structure under `data_efficiency.data_sampling.curriculum_learning`.

---

## Summary

DeepSpeed's data pipeline and efficiency framework provides a comprehensive set of tools for optimizing how training data is routed, sampled, and curriculum-scheduled during distributed training. The curriculum learning system, which progressively increases sequence length difficulty, is the flagship feature, offering up to 2.4x training speedup on language model pre-training. Combined with data routing strategies (hash-based, random, round-robin) and advanced data sampling, these features form a complete data efficiency stack that reduces computational waste while maintaining or improving final model quality. All features are configured through the `data_efficiency` section of the DeepSpeed JSON configuration and integrate seamlessly with the DeepSpeed engine's training loop.
