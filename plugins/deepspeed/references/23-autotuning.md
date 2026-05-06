# Autotuning

## Overview

The DeepSpeed autotuning module (`deepspeed/autotuning/`) provides automated hyperparameter optimization for DeepSpeed training configurations. It systematically explores the configuration space to discover optimal settings for batch size, ZeRO stage, gradient accumulation, offloading strategies, and other performance-critical parameters. The autotuner runs multiple short training experiments, measures throughput and other metrics, and recommends the best-performing configuration.

## Source Code Structure

```
deepspeed/autotuning/
    __init__.py               # Module exports
    autotuner.py              # Main autotuner orchestrator class
    scheduler.py              # Experiment scheduling and execution
    config_templates/         # Configuration template files for experiments
        zero0.json            # Template for ZeRO stage 0 experiments
        zero1.json            # Template for ZeRO stage 1 experiments
        zero2.json            # Template for ZeRO stage 2 experiments
        zero2_offload.json    # Template for ZeRO-2 with CPU offloading
        zero3.json            # Template for ZeRO stage 3 experiments
        zero3_offload.json    # Template for ZeRO-3 with offloading
    tuner/                    # Tuning strategy implementations
        __init__.py
        grid_search.py        # Grid search tuner
        random_search.py      # Random search tuner
        model_based.py        # Model-based (Bayesian) tuner
```

## DeepSpeedAutotuningConfig

### Configuration Schema

The autotuning configuration is specified under the `"autotuning"` key in the DeepSpeed configuration JSON:

```json
{
    "autotuning": {
        "enabled": true,
        "start_step": 0,
        "end_step": 100,
        "metric_path": "autotuning_metrics.json",
        "metric": "throughput",
        "results_dir": "autotuning_results",
        "exps_dir": "autotuning_exps",
        "overwrite": false,
        "fast": true,
        "start_profile_step": 3,
        "end_profile_step": 8,
        "tuner_type": "model_based",
        "tuner_num_trials": 50,
        "max_train_batch_size": 4096,
        "min_train_batch_size": 1,
        "max_train_micro_batch_size_per_gpu": 32,
        "min_train_micro_batch_size_per_gpu": 1,
        "num_tuning_micro_batch_sizes": 4,
        "model_info": null,
        "model_info_path": null,
        "mp_size": 1
    }
}
```

### Configuration Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable or disable autotuning. When `true`, the autotuner runs before normal training begins. |
| `start_step` | `int` | `0` | The global training step at which to start profiling. Steps before this are warmup. |
| `end_step` | `int` | `100` | The global training step at which to stop profiling. The autotuner uses steps in `[start_step, end_step)` for measurement. |
| `metric_path` | `str` | `"autotuning_metrics.json"` | File path where per-experiment metrics are written. Each experiment appends its metrics to this file. |
| `metric` | `str` | `"throughput"` | The optimization metric. One of `"throughput"`, `"latency"`, or `"FLOPS"`. |
| `results_dir` | `str` | `"autotuning_results"` | Directory where autotuning results and recommendations are saved. |
| `exps_dir` | `str` | `"autotuning_exps"` | Directory where experiment configuration files and logs are stored. |
| `overwrite` | `bool` | `false` | If `true`, overwrite existing results and experiments from previous autotuning runs. If `false`, skip experiments that already have results. |
| `fast` | `bool` | `false` | If `true`, use a reduced search space for faster tuning. Skips some ZeRO stages or offload configurations. |
| `start_profile_step` | `int` | `3` | Step within each experiment at which to start profiling. Allows warmup steps to stabilize. |
| `end_profile_step` | `int` | `8` | Step within each experiment at which to stop profiling. The experiment runs only these steps for measurement. |
| `tuner_type` | `str` | `"model_based"` | The search strategy. One of `"grid_search"`, `"random_search"`, or `"model_based"`. |
| `tuner_num_trials` | `int` | `50` | Maximum number of experiments (trials) to run. The tuner will not exceed this limit even if the search space is larger. |
| `max_train_batch_size` | `int` | `4096` | Maximum global training batch size to explore. |
| `min_train_batch_size` | `int` | `1` | Minimum global training batch size to explore. |
| `max_train_micro_batch_size_per_gpu` | `int` | `0` | Maximum micro batch size per GPU to explore. `0` means auto-detect based on GPU memory. |
| `min_train_micro_batch_size_per_gpu` | `int` | `1` | Minimum micro batch size per GPU to explore. |
| `num_tuning_micro_batch_sizes` | `int` | `4` | Number of distinct micro batch sizes to try for each configuration. |
| `model_info` | `dict` or `None` | `None` | Model information dictionary with parameter count and other metadata. Used by model-based tuner for predictions. |
| `model_info_path` | `str` or `None` | `None` | Path to a JSON file containing model information. Alternative to inline `model_info`. |
| `mp_size` | `int` | `1` | Model parallelism size (tensor parallel). Used to compute per-GPU memory requirements. |

### model_info Structure

When `model_info` is provided, it helps the model-based tuner make better predictions:

```json
{
    "model_info": {
        "param_count": 350000000,
        "hidden_size": 1024,
        "num_layers": 24,
        "sequence_length": 2048,
        "vocab_size": 50257,
        "activation_checkpointing": true
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `param_count` | `int` | Total number of model parameters |
| `hidden_size` | `int` | Hidden dimension size |
| `num_layers` | `int` | Number of transformer layers |
| `sequence_length` | `int` | Maximum sequence length |
| `vocab_size` | `int` | Vocabulary size |
| `activation_checkpointing` | `bool` | Whether activation checkpointing is used |

## Tuning Strategies

### Grid Search

The grid search tuner exhaustively evaluates all combinations in the search space. It guarantees finding the best configuration within the defined space but can be slow for large spaces.

**How it works:**

1. Define the complete search space as a Cartesian product of all parameter values
2. Evaluate every point in the grid
3. Rank results by the target metric
4. Return the best configuration

**Search space:**

```
ZeRO stages: [0, 1, 2, 3]
Offloading: [none, optimizer_offload_cpu, param_offload_cpu]
Micro batch sizes: [min_mbs, ..., max_mbs] (num_tuning_micro_batch_sizes values)
Gradient accumulation: computed from train_batch_size and micro_batch_size
```

**Configuration:**
```json
{
    "autotuning": {
        "enabled": true,
        "tuner_type": "grid_search",
        "tuner_num_trials": 200
    }
}
```

**Grid size estimation:**

```
Total experiments = num_zero_stages * num_offload_options * num_micro_batch_sizes * num_batch_sizes
Example: 4 * 3 * 4 * 5 = 240 experiments
```

### Random Search

The random search tuner samples random configurations from the search space. It is more efficient than grid search for high-dimensional spaces and often finds good configurations with fewer trials.

**How it works:**

1. Define the search space with parameter ranges
2. Randomly sample `tuner_num_trials` configurations
3. Evaluate each sampled configuration
4. Return the best configuration found

**Advantages over grid search:**
- Better coverage of high-dimensional spaces
- Can find good configurations with fewer trials
- Naturally handles continuous parameter ranges

**Configuration:**
```json
{
    "autotuning": {
        "enabled": true,
        "tuner_type": "random_search",
        "tuner_num_trials": 30
    }
}
```

### Model-Based (Bayesian Optimization)

The model-based tuner uses Bayesian optimization to intelligently select configurations based on past results. It builds a surrogate model of the performance landscape and uses it to guide the search toward promising regions.

**How it works:**

1. Start with a few random configurations (warm-up phase)
2. Train a surrogate model (e.g., Gaussian Process) on observed results
3. Use an acquisition function (e.g., Expected Improvement) to select the next configuration
4. Evaluate the selected configuration
5. Update the surrogate model with the new result
6. Repeat steps 3-5 for `tuner_num_trials` iterations

**Advantages:**
- Most sample-efficient strategy
- Adapts search based on observed performance
- Can leverage model_info for better initial predictions

**Configuration:**
```json
{
    "autotuning": {
        "enabled": true,
        "tuner_type": "model_based",
        "tuner_num_trials": 20,
        "model_info": {
            "param_count": 350000000,
            "hidden_size": 1024,
            "num_layers": 24,
            "sequence_length": 2048
        }
    }
}
```

### Strategy Comparison

| Strategy | Experiments Needed | Search Quality | Time Cost | Best For |
|----------|--------------------|----------------|-----------|----------|
| Grid Search | All in space | Optimal (within space) | Very High | Small search spaces, thorough analysis |
| Random Search | `tuner_num_trials` | Good (probabilistic) | Medium | Large search spaces, quick exploration |
| Model-Based | `tuner_num_trials` | Very Good (adaptive) | Low-Medium | Most practical scenarios |

## Auto-Discovery Process

### Batch Size Discovery

The autotuner automatically discovers the optimal batch size through the following process:

1. **Maximum micro batch size detection**: For each ZeRO stage and offload configuration, the autotuner runs a short experiment to find the largest micro batch size that fits in GPU memory.

2. **Batch size grid construction**: Using the discovered maximum micro batch size, the autotuner constructs a grid of candidate batch sizes:

```
micro_batch_sizes = [1, 2, 4, 8, 16, max_mbs]
For each micro_batch_size:
    gradient_accumulation_steps = train_batch_size / (micro_batch_size * num_gpus)
    if gradient_accumulation_steps is valid (integer, positive):
        add to candidate list
```

3. **Throughput measurement**: Each candidate is run for `start_profile_step` to `end_profile_step` steps, measuring throughput (samples/second or TFLOPS).

### ZeRO Stage Discovery

The autotuner tests each applicable ZeRO stage:

```
ZeRO Stage 0: Standard DDP (no memory optimization)
  - Tested if model fits in GPU memory with given batch size
  - Baseline for comparison

ZeRO Stage 1: Optimizer state partitioning
  - Always applicable
  - Memory savings: ~4x for Adam optimizer

ZeRO Stage 2: + Gradient partitioning
  - Always applicable
  - Memory savings: ~4x + gradient memory / world_size

ZeRO Stage 3: + Parameter partitioning
  - Always applicable
  - Memory savings: proportional to world_size
  - Higher communication overhead
```

### Offloading Discovery

For each ZeRO stage, the autotuner tests offloading options:

```
ZeRO Stage 2:
  - No offload (GPU optimizer states)
  - CPU optimizer offload (reduced GPU memory)

ZeRO Stage 3:
  - No offload (GPU optimizer states, gradients, parameters)
  - CPU optimizer offload (CPU optimizer states)
  - CPU param offload (CPU parameters)
  - CPU optimizer + param offload (both on CPU)
```

When `fast` mode is enabled, the autotuner reduces the offload search space:
- Only tests no-offload and full-offload configurations
- Skips intermediate combinations

## Experiment Scheduling

### Scheduler Architecture

```python
class AutotuningScheduler:
    """Schedules and executes autotuning experiments."""

    def __init__(self, config, tuner):
        self.config = config
        self.tuner = tuner
        self.results = []
        self.best_config = None

    def run(self):
        """Run all scheduled experiments."""
        # Phase 1: Generate experiment configurations
        experiments = self.tuner.generate_experiments()

        # Phase 2: Execute experiments
        for exp in experiments:
            if len(self.results) >= self.config.tuner_num_trials:
                break
            result = self.execute_experiment(exp)
            self.results.append(result)
            self.tuner.update(exp, result)

        # Phase 3: Analyze and recommend
        self.best_config = self.analyze_results()
        return self.best_config
```

### Experiment Execution Flow

```
For each experiment:
    1. Generate experiment-specific ds_config.json
    2. Create experiment directory under exps_dir/
    3. Launch training with experiment config
        - Run for start_profile_step warmup steps
        - Profile from start_profile_step to end_profile_step
        - Record metrics to metric_path
    4. Collect and parse metrics
    5. Clean up experiment
    6. Record result
    7. Report progress (optional)
```

### Experiment Directory Structure

```
autotuning_exps/
    exp_001_zeRO2_mbs4/
        ds_config.json          # Experiment-specific DeepSpeed config
        train.log               # Training log output
        metrics.json            # Measured metrics
    exp_002_zeRO2_mbs8/
        ds_config.json
        train.log
        metrics.json
    exp_003_zeRO3_mbs4/
        ds_config.json
        train.log
        metrics.json
    ...
```

### Experiment Naming Convention

```
exp_{INDEX}_{ZERO_STAGE}_mbs{MICRO_BATCH_SIZE}[_offload_{OFFLOAD_TYPE}]
```

Examples:
```
exp_001_zeRO0_mbs8
exp_002_zeRO1_mbs16
exp_003_zeRO2_mbs4_offload_optimizer_cpu
exp_004_zeRO3_mbs2_offload_optimizer_cpu_param_cpu
```

## Configuration Templates

### Template Structure

Each template in `config_templates/` defines a base configuration for a specific ZeRO stage and offload strategy. The autotuner fills in the variable fields (batch size, gradient accumulation, etc.) when generating experiment configs.

### Template: zero2.json

```json
{
    "train_batch_size": "$TRAIN_BATCH_SIZE",
    "train_micro_batch_size_per_gpu": "$MICRO_BATCH_SIZE",
    "gradient_accumulation_steps": "$GRADIENT_ACCUMULATION_STEPS",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": "auto"
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": true
}
```

### Template: zero2_offload.json

```json
{
    "train_batch_size": "$TRAIN_BATCH_SIZE",
    "train_micro_batch_size_per_gpu": "$MICRO_BATCH_SIZE",
    "gradient_accumulation_steps": "$GRADIENT_ACCUMULATION_STEPS",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": "auto"
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": true
}
```

### Template: zero3.json

```json
{
    "train_batch_size": "$TRAIN_BATCH_SIZE",
    "train_micro_batch_size_per_gpu": "$MICRO_BATCH_SIZE",
    "gradient_accumulation_steps": "$GRADIENT_ACCUMULATION_STEPS",
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5
    },
    "bf16": {
        "enabled": "auto"
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": true
}
```

### Template Variable Substitution

The autotuner replaces template variables with computed values:

| Variable | Computation |
|----------|-------------|
| `$TRAIN_BATCH_SIZE` | Set by the tuner (from search space) |
| `$MICRO_BATCH_SIZE` | Set by the tuner (from search space) |
| `$GRADIENT_ACCUMULATION_STEPS` | `train_batch_size / (micro_batch_size * num_gpus * mp_size)` |

## Usage

### Basic Usage

```bash
# Run autotuning with default settings
deepspeed --autotuning run train.py --deepspeed ds_config.json

# The ds_config.json must have autotuning enabled:
# {
#     "autotuning": {
#         "enabled": true
#     }
# }
```

### Full CLI Invocation

```bash
# With explicit autotuning configuration
deepspeed --autotuning run \
    --num_gpus=8 \
    train.py --deepspeed ds_config.json --model=bert-large
```

### Programmatic Usage

```python
from deepspeed.autotuning import Autotuner

# Initialize autotuner
autotuner = Autotuner(
    user_args=["train.py", "--deepspeed", "ds_config.json"],
    autotuning_config=autotuning_config,
)

# Run autotuning
best_config = autotuner.tune()

# Print recommended configuration
print(f"Best configuration found: {best_config}")
print(f"Best throughput: {autotuner.best_metric}")
```

### Reading Tuning Results

After autotuning completes, results are written to the `results_dir`:

```
autotuning_results/
    best_config.json          # Recommended configuration
    all_results.json          # Results from all experiments
    tuning_summary.txt        # Human-readable summary
```

**best_config.json:**
```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0
}
```

**all_results.json:**
```json
{
    "experiments": [
        {
            "name": "exp_001_zeRO0_mbs4",
            "config": { "zero_stage": 0, "micro_batch_size": 4 },
            "metric": "throughput",
            "value": 1234.5,
            "status": "completed",
            "error": null
        },
        {
            "name": "exp_002_zeRO1_mbs8",
            "config": { "zero_stage": 1, "micro_batch_size": 8 },
            "metric": "throughput",
            "value": 2345.6,
            "status": "completed",
            "error": null
        },
        {
            "name": "exp_003_zeRO2_mbs4",
            "config": { "zero_stage": 2, "micro_batch_size": 4 },
            "metric": "throughput",
            "value": 3456.7,
            "status": "completed",
            "error": null
        }
    ],
    "best_experiment": "exp_003_zeRO2_mbs4",
    "best_metric_value": 3456.7
}
```

**tuning_summary.txt:**
```
DeepSpeed Autotuning Results
============================
Target metric: throughput
Number of experiments: 12
Best experiment: exp_003_zeRO2_mbs4_offload_optimizer_cpu
Best throughput: 3456.7 samples/sec

Top 5 Configurations:
  1. zeRO2_mbs4_offload_optimizer_cpu: 3456.7 samples/sec
  2. zeRO2_mbs8: 3234.5 samples/sec
  3. zeRO3_mbs2_offload_optimizer_cpu: 2987.3 samples/sec
  4. zeRO1_mbs16: 2765.4 samples/sec
  5. zeRO2_mbs4: 2543.2 samples/sec

Recommended configuration saved to: autotuning_results/best_config.json
```

## Metric Measurement

### Throughput

The primary metric for most autotuning runs. Measures the number of training samples processed per second.

```
throughput = (end_profile_step - start_profile_step) * train_micro_batch_size_per_gpu * num_gpus
             / total_elapsed_time_seconds
```

The autotuner uses `wall_clock_breakdown` to measure the elapsed time precisely:

```python
# Internal throughput measurement
def measure_throughput(engine, start_step, end_step):
    total_samples = 0
    start_time = time.time()

    for step in range(start_step, end_step):
        loss = engine.step()
        total_samples += engine.train_micro_batch_size_per_gpu() * engine.world_size

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed
    return throughput
```

### Latency

Measures the time per training step in milliseconds. Lower is better.

```
latency_ms = total_elapsed_time_ms / (end_profile_step - start_profile_step)
```

### FLOPS

Measures the floating-point operations per second achieved during training. Higher is better.

```
FLOPS = model_flops_per_sample * throughput
```

Where `model_flops_per_sample` is estimated from the model architecture:

```
model_flops_per_sample ≈ 6 * param_count * sequence_length  (for transformers)
```

## Fast Mode

When `fast` mode is enabled (`"fast": true`), the autotuner applies several optimizations to reduce tuning time:

1. **Reduced search space**: Only tests ZeRO stage 2 and 3 (skips 0 and 1)
2. **Fewer micro batch sizes**: Tests only 2-3 micro batch sizes instead of `num_tuning_micro_batch_sizes`
3. **Fewer profile steps**: Uses `start_profile_step=1` and `end_profile_step=3` (fewer measurement steps)
4. **Skip offload combos**: Tests only "no offload" and "full offload" (skips partial offload)
5. **Early termination**: Stops if a configuration exceeds 2x the best known throughput

```json
{
    "autotuning": {
        "enabled": true,
        "fast": true,
        "tuner_type": "model_based",
        "tuner_num_trials": 10
    }
}
```

## Metrics Collection

### Wall Clock Breakdown Integration

The autotuner relies on DeepSpeed's `wall_clock_breakdown` feature to measure timing accurately. This is automatically enabled during autotuning experiments:

```json
{
    "wall_clock_breakdown": true
}
```

This enables precise measurement of:
- Forward pass time
- Backward pass time
- Gradient reduction time
- Optimizer step time
- Total step time

### FLOPS Profiler Integration

When the metric is `"FLOPS"`, the autotuner enables the FLOPS profiler:

```json
{
    "autotuning": {
        "enabled": true,
        "metric": "FLOPS",
        "flops_profiler": {
            "enabled": true,
            "profile_step": 3,
            "module_depth": -1,
            "top_modules": 3,
            "detailed": true
        }
    }
}
```

## Configuration Examples

### Minimal Autotuning

```json
{
    "autotuning": {
        "enabled": true
    }
}
```

This uses all defaults:
- Grid search strategy
- 50 max trials
- Throughput metric
- All ZeRO stages tested
- Automatic batch size discovery

### Fast Autotuning with Model Info

```json
{
    "autotuning": {
        "enabled": true,
        "fast": true,
        "tuner_type": "model_based",
        "tuner_num_trials": 15,
        "model_info": {
            "param_count": 350000000,
            "hidden_size": 1024,
            "num_layers": 24,
            "sequence_length": 2048
        },
        "max_train_batch_size": 512,
        "min_train_batch_size": 32
    }
}
```

### Throughput-Optimized for Large Scale

```json
{
    "autotuning": {
        "enabled": true,
        "tuner_type": "model_based",
        "tuner_num_trials": 30,
        "metric": "throughput",
        "max_train_batch_size": 4096,
        "min_train_batch_size": 64,
        "max_train_micro_batch_size_per_gpu": 16,
        "min_train_micro_batch_size_per_gpu": 1,
        "num_tuning_micro_batch_sizes": 6,
        "start_profile_step": 5,
        "end_profile_step": 15,
        "mp_size": 1
    }
}
```

### FLOPS-Optimized

```json
{
    "autotuning": {
        "enabled": true,
        "metric": "FLOPS",
        "tuner_type": "random_search",
        "tuner_num_trials": 20,
        "start_profile_step": 3,
        "end_profile_step": 10
    }
}
```

### Latency-Optimized for Inference Benchmarking

```json
{
    "autotuning": {
        "enabled": true,
        "metric": "latency",
        "tuner_type": "grid_search",
        "tuner_num_trials": 100,
        "max_train_batch_size": 256,
        "min_train_batch_size": 1,
        "max_train_micro_batch_size_per_gpu": 8,
        "num_tuning_micro_batch_sizes": 8
    }
}
```

### Complete Production Configuration

```json
{
    "train_batch_size": -1,
    "autotuning": {
        "enabled": true,
        "fast": false,
        "start_step": 0,
        "end_step": 100,
        "metric_path": "/shared/autotuning/metrics.json",
        "metric": "throughput",
        "results_dir": "/shared/autotuning/results",
        "exps_dir": "/shared/autotuning/experiments",
        "overwrite": false,
        "start_profile_step": 5,
        "end_profile_step": 15,
        "tuner_type": "model_based",
        "tuner_num_trials": 25,
        "max_train_batch_size": 2048,
        "min_train_batch_size": 16,
        "max_train_micro_batch_size_per_gpu": 32,
        "min_train_micro_batch_size_per_gpu": 1,
        "num_tuning_micro_batch_sizes": 5,
        "model_info": {
            "param_count": 6700000000,
            "hidden_size": 4096,
            "num_layers": 32,
            "sequence_length": 4096,
            "vocab_size": 32000,
            "activation_checkpointing": true
        },
        "mp_size": 1
    },
    "bf16": {
        "enabled": true
    }
}
```

### Launching Autotuning

```bash
# Basic autotuning
deepspeed --autotuning run --num_gpus=8 train.py --deepspeed ds_config.json

# With specific GPU allocation
deepspeed --autotuning run --include="worker1:0-3,worker2:0-3" \
    train.py --deepspeed ds_config.json

# Multi-node autotuning
deepspeed --autotuning run --hostfile=myhostfile --num_nodes=2 \
    train.py --deepspeed ds_config.json
```

## Best Practices

### Choosing a Tuning Strategy

1. **Small models (< 1B params)**: Use `grid_search` for thorough exploration. The search space is manageable and experiments run quickly.

2. **Medium models (1B-10B params)**: Use `random_search` or `model_based` with 20-30 trials. Each experiment is expensive, so efficient search is important.

3. **Large models (> 10B params)**: Use `model_based` with model info. Provide accurate model information for better initial predictions. Use `fast: true` to reduce the number of experiments.

### Setting Batch Size Ranges

1. **max_train_batch_size**: Set to the largest batch size you would practically use. Consider convergence implications.

2. **max_train_micro_batch_size_per_gpu**: Set to `0` (auto-detect) to let the autotuner find the maximum that fits in memory.

3. **num_tuning_micro_batch_sizes**: Use 4-6 for thorough exploration, 2-3 for quick tuning.

### Profile Step Settings

1. **start_profile_step**: Use 3-5 to allow warmup. The first few steps often have outlier timings due to JIT compilation, caching, and memory allocation.

2. **end_profile_step**: Use 8-15 for reliable throughput measurement. More steps give more stable measurements but increase tuning time.

3. Ensure `end_profile_step - start_profile_step >= 3` for statistical reliability.

### Memory Considerations

1. Some ZeRO-3 configurations may not fit in GPU memory for very large models. The autotuner will detect OOM errors and skip those configurations.

2. Use `model_info.param_count` to help the autotuner predict memory requirements and skip infeasible configurations.

3. If autotuning runs out of memory on all configurations, try reducing `max_train_micro_batch_size_per_gpu` or increasing `mp_size`.

## Troubleshooting

### "All experiments failed with OOM"

The model is too large for the available GPU memory. Solutions:
- Reduce `max_train_micro_batch_size_per_gpu`
- Add `model_info` to help skip infeasible configurations
- Use ZeRO-3 with CPU offloading explicitly

### "Autotuning is taking too long"

- Enable `fast: true` mode
- Reduce `tuner_num_trials`
- Reduce `end_profile_step - start_profile_step`
- Use `tuner_type: "model_based"` for more efficient search

### "Results are inconsistent"

- Increase `start_profile_step` (more warmup)
- Increase `end_profile_step - start_profile_step` (more measurement steps)
- Ensure no other GPU workloads are running during autotuning
- Check that `gradient_accumulation_steps` is computing correctly
