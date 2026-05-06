# Monitoring and Profiling

## Overview

DeepSpeed provides a comprehensive suite of monitoring and profiling tools for tracking training progress, analyzing performance bottlenecks, and understanding resource utilization. The monitoring subsystem integrates with popular experiment tracking platforms (TensorBoard, Weights & Biases, Comet ML, CSV), while the profiling subsystem includes a FLOPS profiler, wall-clock timing breakdowns, memory tracking, and communication logging. Together, these tools provide deep visibility into every aspect of distributed training.

## Source Code Structure

```
deepspeed/monitor/
    __init__.py               # Module exports
    monitor.py                # DeepSpeedMonitor base class and manager
    tensorboard.py            # TensorBoard monitor
    wandb.py                  # Weights & Biases monitor
    comet.py                  # Comet ML monitor
    csv_monitor.py            # CSV file monitor

deepspeed/profiling/
    __init__.py
    flops_profiler/
        __init__.py
        flops_profiler.py     # Core FLOPS profiling implementation
```

## Monitoring

### DeepSpeedMonitorConfig

Monitoring is configured under the `"monitor"` key in the DeepSpeed configuration JSON. Each monitor type has its own sub-configuration:

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "runs/",
        "job_name": "deepspeed_training"
    },
    "wandb": {
        "enabled": true,
        "group": "deepspeed_experiment",
        "team": "my_team",
        "project": "llm-training"
    },
    "comet": {
        "enabled": true,
        "samples_log_interval": 100,
        "project": "deepspeed-project",
        "workspace": "my_workspace",
        "api_key": "",
        "experiment_name": "deepspeed_exp_1",
        "experiment_key": "",
        "online": true,
        "mode": null
    },
    "csv_monitor": {
        "enabled": true,
        "output_path": "csv_logs/",
        "job_name": "deepspeed_training"
    }
}
```

### TensorBoard Monitor

#### Configuration: TensorBoardConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable TensorBoard logging. |
| `output_path` | `str` | `""` | Directory where TensorBoard event files are written. If empty, uses the current working directory. |
| `job_name` | `str` | `"DeepSpeed"` | Name of the training job, used as the TensorBoard run name. |

#### Configuration Example

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "/logs/tensorboard",
        "job_name": "gpt3-6.7b-training"
    }
}
```

#### Logged Metrics

The TensorBoard monitor automatically logs:

| Metric | Description | Logging Frequency |
|--------|-------------|-------------------|
| `train/train_loss` | Training loss per step | Every `steps_per_print` steps |
| `train/train_accuracy` | Training accuracy (if provided) | Every `steps_per_print` steps |
| `train/learning_rate` | Current learning rate | Every `steps_per_print` steps |
| `train/mem_allocated` | GPU memory allocated (MB) | Every `steps_per_print` steps |
| `train/mem_cached` | GPU memory cached (MB) | Every `steps_per_print` steps |
| `perf/throughput` | Training throughput (samples/sec) | Every `steps_per_print` steps |
| `perf/flops` | FLOPS achieved | When FLOPS profiler is enabled |
| `time/forward` | Forward pass time (ms) | When wall_clock_breakdown is enabled |
| `time/backward` | Backward pass time (ms) | When wall_clock_breakdown is enabled |
| `time/step` | Total step time (ms) | When wall_clock_breakdown is enabled |
| `time/reduce` | Gradient reduction time (ms) | When wall_clock_breakdown is enabled |

#### Usage

```bash
# Launch training with TensorBoard monitoring
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json

# View TensorBoard in browser
tensorboard --logdir=/logs/tensorboard
```

#### Programmatic Access

```python
# Access the TensorBoard writer directly
model_engine.tensorboard.add_scalar("custom/metric", value, step)
model_engine.tensorboard.add_text("custom/text", "training started", step)
model_engine.tensorboard.add_histogram("weights/layer0", tensor, step)
```

### Weights & Biases (WandB) Monitor

#### Configuration: WandbConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable Weights & Biases logging. |
| `group` | `str` | `""` | W&B run group name. All ranks in the same training job share the same group. |
| `team` | `str` | `""` | W&B team name for organization-level tracking. |
| `project` | `str` | `"deepspeed"` | W&B project name. All runs in a project are grouped together. |

#### Configuration Example

```json
{
    "wandb": {
        "enabled": true,
        "group": "gpt3-experiment-1",
        "team": "my-research-team",
        "project": "large-language-model-training"
    }
}
```

#### Authentication

W&B requires an API key for authentication. Set it via environment variable:

```bash
export WANDB_API_KEY="your-api-key-here"
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

Or log in interactively:

```bash
wandb login
```

#### Logged Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss |
| `train/accuracy` | Training accuracy |
| `train/learning_rate` | Learning rate |
| `perf/throughput` | Throughput (samples/sec) |
| `perf/flops` | FLOPS achieved |
| `memory/allocated` | GPU memory allocated |
| `memory/cached` | GPU memory cached |
| `time/forward_ms` | Forward pass time |
| `time/backward_ms` | Backward pass time |
| `time/step_ms` | Total step time |
| `time/reduce_ms` | Gradient reduction time |

#### Programmatic Access

```python
# Access the W&B run directly
model_engine.wandb.log({"custom/metric": value, "step": step})
model_engine.wandb.summary["best_loss"] = best_loss
model_engine.wandb.config.update({"model": "gpt3-6.7b"})
```

### Comet ML Monitor

#### Configuration: CometConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable Comet ML logging. |
| `samples_log_interval` | `int` | `100` | Interval (in steps) for logging sample data (images, text, etc.). |
| `project` | `str` | `""` | Comet project name. |
| `workspace` | `str` | `""` | Comet workspace name. |
| `api_key` | `str` | `""` | Comet API key. Can also be set via `COMET_API_KEY` environment variable. |
| `experiment_name` | `str` | `""` | Custom experiment name. If empty, Comet auto-generates one. |
| `experiment_key` | `str` | `""` | Key to resume an existing experiment. If set, Comet resumes the experiment instead of creating a new one. |
| `online` | `bool` | `true` | If `true`, logs to Comet servers in real-time. If `false`, logs locally only. |
| `mode` | `str` or `None` | `None` | Comet mode: `"get"`, `"create"`, `"online"`, `"offline"`, or `None` (auto). |

#### Configuration Example

```json
{
    "comet": {
        "enabled": true,
        "samples_log_interval": 50,
        "project": "llm-pretraining",
        "workspace": "my-team",
        "api_key": "",
        "experiment_name": "deepspeed-gpt3-run-1",
        "experiment_key": "",
        "online": true,
        "mode": null
    }
}
```

#### Authentication

```bash
export COMET_API_KEY="your-comet-api-key"
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

#### Logged Metrics

Comet ML logs the same set of metrics as TensorBoard and WandB, plus:

| Metric | Description |
|--------|-------------|
| `samples/input_text` | Input text samples (for NLP models) |
| `samples/generated_text` | Generated text samples |
| `system/gpu_utilization` | GPU utilization percentage |
| `system/gpu_memory_used` | GPU memory usage |
| `system/cpu_utilization` | CPU utilization |

#### Programmatic Access

```python
# Access the Comet experiment directly
model_engine.comet.log_metric("custom/metric", value, step=step)
model_engine.comet.log_text("Generated: " + generated_text, step=step)
model_engine.comet.set_epoch(epoch)
```

### CSV Monitor

#### Configuration: CSVConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable CSV file logging. |
| `output_path` | `str` | `""` | Directory where CSV files are written. If empty, uses the current working directory. |
| `job_name` | `str` | `"DeepSpeed"` | Prefix for the CSV filename. |

#### Configuration Example

```json
{
    "csv_monitor": {
        "enabled": true,
        "output_path": "/logs/csv",
        "job_name": "gpt3-training"
    }
}
```

#### Output Format

The CSV monitor writes one row per logged step:

```
/logs/csv/gpt3-training_rank0.csv
```

**CSV columns:**

| Column | Type | Description |
|--------|------|-------------|
| `step` | `int` | Global training step |
| `loss` | `float` | Training loss |
| `learning_rate` | `float` | Learning rate |
| `mem_allocated` | `float` | GPU memory allocated (MB) |
| `mem_cached` | `float` | GPU memory cached (MB) |
| `throughput` | `float` | Throughput (samples/sec) |
| `forward_time` | `float` | Forward pass time (ms) |
| `backward_time` | `float` | Backward pass time (ms) |
| `step_time` | `float` | Total step time (ms) |
| `reduce_time` | `float` | Gradient reduction time (ms) |

**Example output:**

```csv
step,loss,learning_rate,mem_allocated,mem_cached,throughput,forward_time,backward_time,step_time,reduce_time
1,10.234,0.0001,32768.0,40960.0,1234.5,12.3,45.6,80.1,8.2
2,9.876,0.0001,32768.0,40960.0,1256.7,12.1,44.8,79.5,7.9
3,9.543,0.0001,32768.0,40960.0,1289.1,11.9,44.3,78.8,7.6
```

### Monitor Manager

The `MonitorMaster` class coordinates all enabled monitors:

```python
class MonitorMaster:
    """Manages all configured monitors."""

    def __init__(self, monitors):
        self.monitors = monitors  # List of enabled monitor instances

    def write_events(self, step, events):
        """Write events to all enabled monitors."""
        for monitor in self.monitors:
            for event_name, event_value in events.items():
                monitor.write_event(step, event_name, event_value)

    def flush(self):
        """Flush all monitor buffers."""
        for monitor in self.monitors:
            monitor.flush()

    def finish(self):
        """Finish all monitors (close files, finish runs, etc.)."""
        for monitor in self.monitors:
            monitor.finish()
```

### Multiple Monitors Simultaneously

DeepSpeed supports enabling multiple monitors at the same time:

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "/logs/tensorboard"
    },
    "wandb": {
        "enabled": true,
        "project": "llm-training",
        "group": "experiment-1"
    },
    "csv_monitor": {
        "enabled": true,
        "output_path": "/logs/csv"
    }
}
```

Each metric is logged to all enabled monitors simultaneously. Only rank 0 writes to monitors by default.

## Profiling

### FLOPS Profiler

#### Overview

The DeepSpeed FLOPS profiler measures the floating-point operations (FLOPs) per second achieved during model forward and backward passes. It provides both aggregate and per-layer FLOPs measurements, enabling detailed analysis of computational efficiency.

#### Configuration: DeepSpeedFlopsProfilerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable the FLOPS profiler. |
| `recompute_fwd_factor` | `float` | `0.0` | Factor for re-computing forward pass FLOPs during backward. Set to `1.0` if activation checkpointing is used (forward is recomputed during backward). Set to `0.0` if activation checkpointing is not used. |
| `profile_step` | `int` | `1` | The global step at which to profile. Only this step is profiled; other steps run normally without profiling overhead. |
| `module_depth` | `int` | `-1` | Depth of module hierarchy to display in the profile. `-1` means show all depths. `0` shows only the top-level model. `1` shows top-level and one level of sub-modules. |
| `top_modules` | `int` | `3` | Number of top modules to display by FLOPs. Shows the most computationally expensive modules. |
| `detailed` | `bool` | `false` | If `true`, prints detailed per-module FLOPs breakdown. If `false`, prints only summary. |
| `output_file` | `str` or `None` | `None` | File path to write profile output. If `None`, prints to stdout. |

#### Configuration Example

```json
{
    "flops_profiler": {
        "enabled": true,
        "recompute_fwd_factor": 0.0,
        "profile_step": 10,
        "module_depth": -1,
        "top_modules": 5,
        "detailed": true,
        "output_file": "flops_profile.txt"
    }
}
```

#### Programmatic Usage

```python
import deepspeed
from deepspeed.profiling.flops_profiler import get_model_profile

# Profile a model's forward pass
flops, macs, params = get_model_profile(
    model,
    input_shape=(1, 128),       # Input tensor shape (batch_size, seq_len)
    input_constructor=None,      # Custom input constructor function
    print_profile=True,          # Print detailed profile
    detailed=True,               # Include per-layer breakdown
    output_file="profile.txt",   # Output file path
    module_depth=-1,             # Depth of module hierarchy
    top_modules=3,               # Top modules to display
    as_string=False,             # Return as numbers (True for formatted strings)
)

print(f"Total FLOPs: {flops}")
print(f"Total MACs: {macs}")
print(f"Total Params: {params}")
```

#### Profile Output Format

When `detailed=True`, the profiler outputs a hierarchical breakdown:

```
Model Profile:
-----------------------------  ------------  ------------  ------------  ------------  ------------
Module                         FLOPs         MACs          Params (%)    FLOPs (%)     MACs (%)
-----------------------------  ------------  ------------  ------------  ------------  ------------
GPT3Model                      350.0 T       175.0 T       6.7 B (100%)  100.0%        100.0%
  TransformerEncoder           340.5 T       170.2 T       6.6 B (98.5%) 97.3%        97.3%
    SelfAttention              180.2 T       90.1 T        3.5 B (52.2%) 51.5%        51.5%
      QKVLinear                45.0 T        22.5 T        1.1 B (16.4%) 12.9%        12.9%
      AttentionScore           30.1 T        15.0 T        0 (0.0%)      8.6%         8.6%
      AttentionOutput          45.0 T        22.5 T        1.1 B (16.4%) 12.9%        12.9%
    MLP                        160.3 T       80.1 T        3.1 B (46.3%) 45.8%        45.8%
      Linear1                  80.1 T        40.0 T        1.6 B (23.9%) 22.9%        22.9%
      Linear2                  80.1 T        40.0 T        1.6 B (23.9%) 22.9%        22.9%
  Embedding                    9.5 T         4.75 T        0.1 B (1.5%)  2.7%         2.7%
-----------------------------  ------------  ------------  ------------  ------------  ------------
Total                          350.0 T       175.0 T       6.7 B (100%)
-----------------------------  ------------  ------------  ------------  ------------  ------------
Throughput: 123.4 TFLOPS
```

#### Measuring Throughput

```python
# During training with the engine
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config_params=ds_config,  # with flops_profiler enabled
)

# The profiler automatically measures on the configured profile_step
# After profiling, access the results:
print(f"Model FLOPs: {model_engine.flops_profiler.get_total_flops()}")
print(f"Throughput: {model_engine.flops_profiler.get_total_flops() / step_time:.2f} TFLOPS")
```

#### Supported Operations

The FLOPS profiler handles the following operation types:

| Operation | FLOPs Formula |
|-----------|---------------|
| Linear (nn.Linear) | `2 * in_features * out_features * batch_size` |
| Conv1d/Conv2d/Conv3d | `2 * kernel_prod * in_channels * out_channels * output_size` |
| MultiheadAttention | `4 * hidden_size^2 * seq_len * batch_size` |
| Embedding | `vocab_size * hidden_size * batch_size * seq_len` |
| LayerNorm | `2 * hidden_size * batch_size * seq_len` |
| ReLU/GELU/SiLU | `batch_size * seq_len * hidden_size` |
| Softmax | `2 * batch_size * seq_len * hidden_size` |
| BatchNorm | `4 * num_features * spatial_size` |
| Element-wise ops | `batch_size * num_elements` |

#### Recompute Forward Factor

When activation checkpointing is enabled, the forward pass is recomputed during the backward pass. The `recompute_fwd_factor` accounts for this additional computation:

```json
{
    "flops_profiler": {
        "enabled": true,
        "recompute_fwd_factor": 1.0
    }
}
```

| recompute_fwd_factor | Meaning |
|----------------------|---------|
| `0.0` | No activation checkpointing. Forward FLOPs counted once. |
| `1.0` | Full activation checkpointing. Forward FLOPs counted twice (once in forward, once in backward). |
| `0.5` | Partial checkpointing. Half of forward FLOPs recomputed in backward. |

### Wall Clock Breakdown

#### Overview

The wall clock breakdown feature measures the time spent in each phase of a training step: forward pass, backward pass, gradient reduction, and optimizer step. This helps identify which phase is the bottleneck.

#### Configuration

```json
{
    "wall_clock_breakdown": true,
    "wall_clock_breakdown": {
        "enabled": true
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wall_clock_breakdown` | `bool` | `false` | Enable wall clock timing of forward, backward, and step phases. Can also be specified as an object with `enabled` key. |

#### Measured Phases

| Phase | Timer Name | What is Measured |
|-------|-----------|-----------------|
| Forward | `forward_timers` | Model forward pass, including any communication (ZeRO-3 all-gather) |
| Backward | `backward_timers` | Model backward pass, including gradient computation and any communication |
| Backward-Reduce | `backward_reduce_timers` | Gradient reduction (all-reduce or reduce-scatter) |
| Optimizer Step | `step_timers` | Optimizer.step(), including parameter updates and any communication |
| Total Step | `step_time` | Wall clock time for the entire training step |

#### Timing Output

When enabled, DeepSpeed prints timing information at the configured interval:

```
[2024-01-15 10:30:45] step=100 loss=5.432 lr=0.0001 step_time=85.3ms
  forward=12.3ms backward=45.6ms reduce=8.2ms step=19.2ms
```

#### Accessing Timing Data Programmatically

```python
# Access timing data from the engine
timers = model_engine.timers

# Get individual phase times
forward_time = timers("forward").elapsed(reset=True)
backward_time = timers("backward").elapsed(reset=True)
reduce_time = timers("reduce").elapsed(reset=True)
step_time = timers("step").elapsed(reset=True)

# Print timing summary
model_engine.print_timers()
```

#### SynchronizedWallClockTimer

DeepSpeed provides a synchronized timer that accounts for inter-rank timing differences:

```python
from deepspeed.utils.timer import SynchronizedWallClockTimer

timer = SynchronizedWallClockTimer()

# Start timing
timer.start("forward")
output = model(input)
timer.stop("forward")

timer.start("backward")
loss.backward()
timer.stop("backward")

timer.start("allreduce")
model.allreduce_gradients()
timer.stop("allreduce")

# Get mean across all ranks
summary = timer.get_mean()
# {
#     "forward": {"total_time_ms": 12.3, "avg_time_ms": 12.3, "count": 1},
#     "backward": {"total_time_ms": 45.6, "avg_time_ms": 45.6, "count": 1},
#     "allreduce": {"total_time_ms": 8.2, "avg_time_ms": 8.2, "count": 1},
# }
```

### Memory Breakdown

#### Overview

Memory breakdown tracks GPU memory usage during training, providing visibility into where memory is consumed (model parameters, gradients, optimizer states, activations).

#### Configuration

```json
{
    "memory_breakdown": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory_breakdown` | `bool` | `false` | Enable memory usage tracking and logging. |

#### Tracked Memory Categories

| Category | Description |
|----------|-------------|
| `model_parameters` | Memory for model parameters (weights, biases) |
| `model_gradients` | Memory for parameter gradients |
| `optimizer_states` | Memory for optimizer state (momentum, variance, etc.) |
| `activations` | Memory for intermediate activations (forward pass) |
| `gradient_buffers` | Memory for contiguous gradient buffers (ZeRO) |
| `temporary_buffers` | Memory for temporary computation buffers |
| `total_allocated` | Total GPU memory currently allocated |
| `total_reserved` | Total GPU memory reserved by the allocator |

#### Memory Output

```
Memory Usage (GPU 0):
  Model Parameters: 8192.0 MB (25.0%)
  Model Gradients:  8192.0 MB (25.0%)
  Optimizer States: 16384.0 MB (50.0%)
  Activations:      0.0 MB (0.0%)
  Total Allocated:  32768.0 MB
  Total Reserved:   40960.0 MB
  GPU Total:        81920.0 MB
```

#### Programmatic Access

```python
# Get current memory usage
memory_stats = model_engine.memory_stats()
print(f"Parameters: {memory_stats['model_parameters'] / 1e9:.2f} GB")
print(f"Gradients: {memory_stats['model_gradients'] / 1e9:.2f} GB")
print(f"Optimizer: {memory_stats['optimizer_states'] / 1e9:.2f} GB")

# Track memory during training
import torch
for step in range(100):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Step {step}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
```

### Communication Logging

#### Overview

Communication logging provides detailed tracking of all collective communication operations (all-reduce, all-gather, reduce-scatter, etc.) during training. This is essential for diagnosing communication bottlenecks in distributed training.

#### Configuration

```json
{
    "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true,
        "debug": false,
        "log_dir": "comms_logs/"
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable communication logging. |
| `verbose` | `bool` | `false` | Print detailed info for each communication call. |
| `prof_all` | `bool` | `false` | Profile all communication operations by default. |
| `debug` | `bool` | `false` | Enable debug mode with call stack traces. |
| `log_dir` | `str` | `""` | Directory for communication log files. |

#### Logged Information

For each communication operation:

| Field | Description |
|-------|-------------|
| `operation` | Type of communication (allreduce, allgather, etc.) |
| `rank` | Rank that performed the operation |
| `tensor_shape` | Shape of the tensor(s) involved |
| `tensor_dtype` | Data type of the tensor(s) |
| `message_size` | Total message size in bytes |
| `duration_ms` | Wall-clock time for the operation |
| `group` | Process group (if not default) |
| `caller` | Function that triggered the operation |

#### Communication Log Output

When `verbose=true`:

```
[Rank 0] allreduce: shape=(4096, 4096), dtype=torch.float16, size=32.0MB, time=12.35ms
[Rank 0] allgather: shape=(1024, 1024), dtype=torch.float16, size=2.0MB, time=9.13ms
[Rank 0] reduce_scatter: shape=(2048, 2048), dtype=torch.float16, size=8.0MB, time=11.42ms
```

#### Communication Summary

At the end of training, a summary is printed:

```
Communication Summary:
Operation       Count   Total(ms)   Avg(ms)   Min(ms)   Max(ms)   Total(MB)
allreduce       500     6175.0      12.35     10.2      15.7     16000.0
allgather       200     1826.0      9.13      8.1       11.3     400.0
reduce_scatter  200     2284.0      11.42     9.8       13.5     1600.0
broadcast       10      23.4        2.34      2.1       2.8      256.0
send            100     450.0       4.50      3.2       6.1     800.0
recv            100     420.0       4.20      3.0       5.8     800.0
```

### PyTorch Profiler Integration

DeepSpeed integrates with PyTorch's native profiler for detailed operator-level profiling:

```python
import torch.profiler

# Use PyTorch profiler with DeepSpeed
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        outputs = model_engine(batch)
        loss = criterion(outputs)
        model_engine.backward(loss)
        model_engine.step()
        prof.step()
```

This generates a TensorBoard-compatible trace that shows:
- CUDA kernel execution timeline
- Memory allocation/deallocation events
- Communication operations
- CPU-side Python call stack

### steps_per_print Configuration

#### Overview

The `steps_per_print` parameter controls how often DeepSpeed prints training progress and logs metrics to monitors.

#### Configuration

```json
{
    "steps_per_print": 10
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `steps_per_print` | `int` | `10` | Number of training steps between printed status updates and metric logging. |

#### Output Format

```
[2024-01-15 10:30:45,000] [INFO] [logging.py:123:log_dist] [Rank 0]
  step=100 loss=5.432 learning_rate=0.0001
  step_time=85.3ms throughput=1234.5 samples/sec
  mem_allocated=32768.0 MB mem_cached=40960.0 MB
```

#### Interaction with Monitoring

`steps_per_print` controls when metrics are logged to all enabled monitors:

```python
# Internal logic
if global_step % self.steps_per_print == 0:
    # Print to console
    self.print_training_status(global_step, loss, learning_rate)

    # Log to all monitors
    self.monitor_manager.write_events(global_step, {
        "train/loss": loss,
        "train/learning_rate": learning_rate,
        "train/mem_allocated": mem_allocated,
        "train/mem_cached": mem_cached,
    })
```

### dump_state for Debugging

#### Overview

The `dump_state` configuration option enables periodic dumping of the complete training state for debugging purposes. This includes model parameter norms, gradient norms, optimizer state, and memory usage.

#### Configuration

```json
{
    "dump_state": true,
    "dump_state_step": 100
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dump_state` | `bool` | `false` | Enable state dumping. |
| `dump_state_step` | `int` | `0` | Step at which to dump state. `0` means dump at every `steps_per_print` interval. |

#### State Dump Contents

When enabled, the state dump includes:

```
=== Training State Dump (Step 100) ===

Model Parameter Norms:
  layer.0.weight: norm=123.456, min=-0.0123, max=0.0456, mean=0.0001
  layer.0.bias: norm=0.456, min=-0.001, max=0.002, mean=0.000
  layer.1.weight: norm=234.567, min=-0.0234, max=0.0567, mean=0.0002
  ...

Gradient Norms:
  layer.0.weight: norm=12.345, max_abs=0.123
  layer.0.bias: norm=0.567, max_abs=0.005
  layer.1.weight: norm=23.456, max_abs=0.234
  ...

Optimizer State (Adam):
  layer.0.weight:
    exp_avg: norm=1.234, max_abs=0.01
    exp_avg_sq: norm=0.567, max_abs=0.005
  ...

Memory Usage:
  Parameters: 8192.0 MB
  Gradients:  8192.0 MB
  Optimizer:  16384.0 MB
  Allocated:  32768.0 MB

ZeRO State:
  Stage: 3
  Param partitions: 8
  Gradient partitions: 8
  Optimizer partitions: 8
```

#### Programmatic Access

```python
# Trigger state dump manually
model_engine.dump_state()

# Access parameter norms
for name, param in model_engine.named_parameters():
    if param.grad is not None:
        print(f"{name}: param_norm={param.data.norm():.4f}, "
              f"grad_norm={param.grad.data.norm():.4f}")
```

## Configuration Examples

### Complete Monitoring Configuration

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "/shared/logs/tensorboard",
        "job_name": "gpt3-6.7b-pretraining"
    },
    "wandb": {
        "enabled": true,
        "group": "gpt3-experiment-1",
        "team": "research-team",
        "project": "llm-pretraining"
    },
    "csv_monitor": {
        "enabled": true,
        "output_path": "/shared/logs/csv",
        "job_name": "gpt3-6.7b"
    },
    "steps_per_print": 10
}
```

### Complete Profiling Configuration

```json
{
    "flops_profiler": {
        "enabled": true,
        "recompute_fwd_factor": 1.0,
        "profile_step": 20,
        "module_depth": 2,
        "top_modules": 5,
        "detailed": true,
        "output_file": "/shared/profiles/flops_profile_step20.txt"
    },
    "wall_clock_breakdown": true,
    "memory_breakdown": true,
    "comms_logger": {
        "enabled": true,
        "verbose": true,
        "prof_all": true,
        "debug": false,
        "log_dir": "/shared/logs/comms"
    },
    "dump_state": true,
    "dump_state_step": 100
}
```

### Monitoring + Profiling Combined

```json
{
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,

    "tensorboard": {
        "enabled": true,
        "output_path": "/logs/tensorboard",
        "job_name": "deepspeed-training"
    },
    "wandb": {
        "enabled": true,
        "project": "llm-training",
        "group": "experiment-1"
    },
    "csv_monitor": {
        "enabled": true,
        "output_path": "/logs/csv"
    },

    "flops_profiler": {
        "enabled": true,
        "profile_step": 15,
        "module_depth": -1,
        "top_modules": 5,
        "detailed": true,
        "output_file": null
    },
    "wall_clock_breakdown": true,
    "memory_breakdown": true,
    "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true,
        "debug": false
    }
}
```

### Minimal Profiling (FLOPS Only)

```json
{
    "flops_profiler": {
        "enabled": true,
        "profile_step": 5
    },
    "steps_per_print": 5
}
```

### Debug Configuration (Maximum Visibility)

```json
{
    "wall_clock_breakdown": true,
    "memory_breakdown": true,
    "flops_profiler": {
        "enabled": true,
        "profile_step": 5,
        "detailed": true,
        "module_depth": -1,
        "top_modules": 10,
        "output_file": "debug_profile.txt"
    },
    "comms_logger": {
        "enabled": true,
        "verbose": true,
        "prof_all": true,
        "debug": true,
        "log_dir": "debug_comms/"
    },
    "dump_state": true,
    "dump_state_step": 0,
    "steps_per_print": 1,
    "tensorboard": {
        "enabled": true,
        "output_path": "debug_logs/tensorboard",
        "job_name": "debug-run"
    },
    "csv_monitor": {
        "enabled": true,
        "output_path": "debug_logs/csv",
        "job_name": "debug-run"
    }
}
```

### Production Configuration (Lightweight Monitoring)

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "/shared/logs/tensorboard",
        "job_name": "production-training"
    },
    "wall_clock_breakdown": true,
    "steps_per_print": 100,
    "flops_profiler": {
        "enabled": true,
        "profile_step": 50
    }
}
```

## Best Practices

### Monitoring

1. **Enable TensorBoard for all training runs** - It provides low-overhead real-time visualization and is invaluable for debugging training issues.

2. **Use WandB for experiment comparison** - W&B excels at comparing multiple training runs, hyperparameter sweeps, and sharing results with collaborators.

3. **Use CSV logging for automated analysis** - CSV files are easy to parse with pandas or other tools for automated reporting and analysis.

4. **Set appropriate `steps_per_print`** - Too frequent logging (e.g., every step) can slow training. Use 10-100 for most runs.

5. **Only rank 0 logs by default** - This reduces overhead. For debugging multi-rank issues, enable per-rank logging.

### Profiling

1. **Profile early and profile often** - Run the FLOPS profiler at least once at the beginning of training to establish a baseline.

2. **Use `profile_step` to avoid overhead** - The FLOPS profiler adds overhead. Set `profile_step` to a step after warmup (5-20) and only profile one step.

3. **Enable `wall_clock_breakdown` for all serious training** - It has minimal overhead and provides essential timing data.

4. **Use `recompute_fwd_factor=1.0` with activation checkpointing** - Otherwise FLOPs will be underestimated.

5. **Enable `memory_breakdown` when debugging OOM** - It shows exactly where memory is consumed.

6. **Use `comms_logger` to diagnose communication bottlenecks** - If communication time exceeds 30% of step time, investigate communication optimization (overlap, data type, coalescing).

### Performance Analysis

1. **Calculate model FLOPS utilization (MFU)**:

```
MFU = achieved_FLOPS / peak_hardware_FLOPS

Target: > 40% MFU for efficient training
Good: > 50% MFU
Excellent: > 60% MFU
```

2. **Identify bottlenecks from timing breakdown**:

```
If forward > 60% of step time: Check model architecture, kernel efficiency
If backward > 60% of step time: Check gradient computation, activation checkpointing
If reduce > 30% of step time: Check communication optimization (overlap, compression)
If step (optimizer) > 20%: Check optimizer efficiency, CPU offloading
```

3. **Memory budgeting**:

```
For a model with P parameters (FP16):
  Parameters:     2P bytes
  Gradients:      2P bytes
  Adam states:    12P bytes (FP32: momentum=4P, variance=4P, master=4P)
  Total per GPU:  16P / world_size (with ZeRO-3)

Example: 7B parameter model, 8 GPUs, ZeRO-3
  Per GPU: 16 * 7B / 8 = 14 GB per GPU (fits in 16 GB GPU)
```
