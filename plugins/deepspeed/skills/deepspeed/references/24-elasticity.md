# Elasticity

## Overview

The DeepSpeed elasticity module (`deepspeed/elasticity/`) provides dynamic resource scaling capabilities that allow training jobs to adapt to changing resource availability. It enables training to continue when workers join or leave the job, automatically adjusting batch sizes and parallelism configurations to maintain training progress. This is essential for training on shared clusters, spot/preemptible instances, and environments where resource availability fluctuates.

## Source Code Structure

```
deepspeed/elasticity/
    __init__.py               # Module exports
    elasticity.py             # Core elasticity logic: DeepSpeedElasticity class
    elastic_agent.py          # Agent for managing worker processes and restarts
    config.py                 # ElasticityConfig and validation
    constants.py              # Constants: default values, version strings
```

## ElasticityConfig

### Configuration Schema

The elasticity configuration is specified under the `"elasticity"` key in the DeepSpeed configuration JSON:

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 2000,
        "micro_batches": 4,
        "min_gpus": 1,
        "max_gpus": 64,
        "model_parallel_size": 1,
        "num_gpus_per_node": 8,
        "min_time": 60,
        "version": 0.2,
        "prefer_larger_batch": true,
        "ignore_non_elastic_batch_info": false
    }
}
```

### Configuration Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable or disable elasticity. When enabled, the training job can dynamically adjust to resource changes. |
| `max_acceptable_batch_size` | `int` | `2000` | Maximum global batch size that is acceptable for training. The elastic system will not exceed this value even if more GPUs become available. This ensures convergence is not adversely affected by overly large batch sizes. |
| `micro_batches` | `int` | `4` | Number of micro-batches used for gradient accumulation. The elastic system maintains this count across different GPU configurations to keep the effective gradient update consistent. |
| `min_gpus` | `int` | `1` | Minimum number of GPUs required to continue training. If the number of available GPUs drops below this threshold, training is paused until more GPUs become available. |
| `max_gpus` | `int` | `10000` | Maximum number of GPUs to use. The elastic system will not use more GPUs than this limit, even if more are available. |
| `model_parallel_size` | `int` | `1` | Tensor model parallelism size. The total number of GPUs must be divisible by this value. Data parallelism size is computed as `total_gpus / model_parallel_size`. |
| `num_gpus_per_node` | `int` | `8` | Number of GPUs per compute node. Used for resource allocation and process placement. |
| `min_time` | `int` | `60` | Minimum training time (in seconds) between elastic reconfigurations. This prevents thrashing caused by rapid resource fluctuations. After a reconfiguration, the system waits at least `min_time` seconds before considering another reconfiguration. |
| `version` | `float` | `0.2` | Elasticity configuration version. `0.1` uses the original batch-size calculation algorithm. `0.2` uses the improved algorithm that provides better batch-size scaling. |
| `prefer_larger_batch` | `bool` | `true` | When multiple valid batch sizes exist for a given GPU count, prefer the larger one. This maximizes throughput when GPU resources increase. |
| `ignore_non_elastic_batch_info` | `bool` | `false` | If `true`, ignore batch size information from non-elastic configuration (e.g., `train_batch_size` in the main config). The elastic system will compute batch sizes solely from its own parameters. |

### Version Differences

#### Version 0.1

In version 0.1, the batch size calculation uses a simpler formula:

```
micro_batch_size_per_gpu = max_acceptable_batch_size / (micro_batches * max_gpus)
train_batch_size = micro_batch_size_per_gpu * micro_batches * current_num_gpus
```

This version can lead to small micro batch sizes when `max_gpus` is large, potentially underutilizing GPU compute capacity.

#### Version 0.2 (Default)

In version 0.2, the batch size calculation is improved to maximize GPU utilization:

```
micro_batch_size_per_gpu = compute_optimal_micro_batch_size(current_num_gpus, max_acceptable_batch_size, micro_batches)
train_batch_size = micro_batch_size_per_gpu * micro_batches * current_num_gpus
```

Version 0.2 selects the largest micro batch size per GPU that:
1. Keeps the total batch size below `max_acceptable_batch_size`
2. Results in an integer number of gradient accumulation steps
3. Maximizes GPU utilization

## DeepSpeedElasticity Class

### Class Overview

The `DeepSpeedElasticity` class is the main entry point for elasticity functionality. It computes valid elastic configurations and manages transitions between resource configurations.

```python
class DeepSpeedElasticity:
    def __init__(self, elasticity_config):
        self.config = elasticity_config
        self.elastic_dict = self._build_elastic_dictionary()

    def compute_elastic_config(self, world_size):
        """Compute the optimal configuration for the given number of GPUs."""

    def ensure_immutable_elastic_config(self, config):
        """Validate and freeze an elastic configuration."""

    def get_valid_world_sizes(self):
        """Get all valid GPU counts for this configuration."""

    def get_all_compatible_configs(self):
        """Get all valid (world_size, batch_size) pairs."""
```

### Initialization

```python
from deepspeed.elasticity import DeepSpeedElasticity
from deepspeed.elasticity.config import ElasticityConfig

# Create from config dictionary
elasticity_config = ElasticityConfig({
    "enabled": True,
    "max_acceptable_batch_size": 2000,
    "micro_batches": 4,
    "min_gpus": 2,
    "max_gpus": 64,
    "model_parallel_size": 1,
    "num_gpus_per_node": 8,
})

elasticity = DeepSpeedElasticity(elasticity_config)
```

### compute_elastic_config()

The core method that computes the optimal batch and GPU configuration for a given number of available GPUs.

```python
def compute_elastic_config(self, world_size):
    """
    Compute the optimal training configuration for the given world size.

    Args:
        world_size (int): Number of available GPUs.

    Returns:
        dict: Computed configuration with:
            - train_batch_size (int): Global batch size
            - train_micro_batch_size_per_gpu (int): Micro batch size per GPU
            - gradient_accumulation_steps (int): Number of gradient accumulation steps
            - dp_size (int): Data parallelism size
            - world_size (int): Number of GPUs to use

    Raises:
        RuntimeError: If world_size is not a valid configuration.
    """
```

**Computation Logic (Version 0.2):**

```python
def compute_elastic_config(self, world_size):
    # Step 1: Validate world_size
    if world_size < self.config.min_gpus:
        raise RuntimeError(f"world_size {world_size} < min_gpus {self.config.min_gpus}")
    if world_size > self.config.max_gpus:
        world_size = self.config.max_gpus

    # Step 2: Compute data parallelism size
    assert world_size % self.config.model_parallel_size == 0
    dp_size = world_size // self.config.model_parallel_size

    # Step 3: Find optimal micro batch size per GPU
    # Try from largest to smallest to maximize utilization
    max_mbs = self.config.max_acceptable_batch_size // (self.config.micro_batches * dp_size)

    for micro_batch_size in range(max_mbs, 0, -1):
        train_batch_size = micro_batch_size * self.config.micro_batches * dp_size

        if train_batch_size <= self.config.max_acceptable_batch_size:
            # Valid configuration found
            return {
                "train_batch_size": train_batch_size,
                "train_micro_batch_size_per_gpu": micro_batch_size,
                "gradient_accumulation_steps": self.config.micro_batches,
                "dp_size": dp_size,
                "world_size": world_size,
            }

    raise RuntimeError(f"No valid batch size found for world_size={world_size}")
```

**Example Computations:**

```
Config: max_acceptable_batch_size=2000, micro_batches=4, model_parallel_size=1

world_size=8:
  dp_size = 8
  max_mbs = 2000 / (4 * 8) = 62
  train_batch_size = 62 * 4 * 8 = 1984
  gradient_accumulation_steps = 4

world_size=16:
  dp_size = 16
  max_mbs = 2000 / (4 * 16) = 31
  train_batch_size = 31 * 4 * 16 = 1984
  gradient_accumulation_steps = 4

world_size=32:
  dp_size = 32
  max_mbs = 2000 / (4 * 32) = 15
  train_batch_size = 15 * 4 * 32 = 1920
  gradient_accumulation_steps = 4

world_size=64:
  dp_size = 64
  max_mbs = 2000 / (4 * 64) = 7
  train_batch_size = 7 * 4 * 64 = 1792
  gradient_accumulation_steps = 4
```

### ensure_immutable_elastic_config()

Validates and freezes an elastic configuration to ensure it cannot change during training:

```python
def ensure_immutable_elastic_config(self, config):
    """
    Validate that the given configuration is a valid elastic configuration
    and freeze it for the duration of training.

    Args:
        config (dict): Configuration to validate.

    Returns:
        dict: Validated and frozen configuration.

    Raises:
        RuntimeError: If the configuration is invalid.
    """
```

**Validation checks:**
1. `world_size` is within `[min_gpus, max_gpus]`
2. `world_size` is divisible by `model_parallel_size`
3. `train_batch_size` equals `train_micro_batch_size_per_gpu * gradient_accumulation_steps * dp_size`
4. `train_batch_size` does not exceed `max_acceptable_batch_size`
5. `gradient_accumulation_steps` equals `micro_batches`

### get_valid_world_sizes()

Returns all GPU counts that produce valid elastic configurations:

```python
valid_sizes = elasticity.get_valid_world_sizes()
# Returns: [1, 2, 4, 8, 16, 32, 64]  (depends on configuration)
```

**Filtering rules:**
- `min_gpus <= size <= max_gpus`
- `size % model_parallel_size == 0`
- `max_acceptable_batch_size >= micro_batches * model_parallel_size` (minimum valid batch)

### get_all_compatible_configs()

Returns all valid (world_size, configuration) pairs:

```python
configs = elasticity.get_all_compatible_configs()
# Returns:
# {
#     2: {"train_batch_size": 500, "train_micro_batch_size_per_gpu": 62, ...},
#     4: {"train_batch_size": 992, "train_micro_batch_size_per_gpu": 62, ...},
#     8: {"train_batch_size": 1984, "train_micro_batch_size_per_gpu": 62, ...},
#     16: {"train_batch_size": 1984, "train_micro_batch_size_per_gpu": 31, ...},
#     32: {"train_batch_size": 1920, "train_micro_batch_size_per_gpu": 15, ...},
#     64: {"train_batch_size": 1792, "train_micro_batch_size_per_gpu": 7, ...},
# }
```

## Dynamic Batch Size Adjustment

### How Batch Size Scaling Works

When the number of available GPUs changes, the elastic system recomputes the batch size to maintain training quality:

```
Key Principle:
  gradient_accumulation_steps (micro_batches) stays CONSTANT
  micro_batch_size_per_gpu ADJUSTS to keep total batch size acceptable

This ensures:
  - Gradient accumulation depth is consistent (stable training dynamics)
  - Total batch size remains within convergence-safe bounds
  - GPU utilization is maximized for the available resources
```

### Scaling Scenarios

**Scale Up (2 GPUs -> 8 GPUs):**

```
Before (2 GPUs):
  micro_batch_size_per_gpu = 250
  gradient_accumulation_steps = 4
  train_batch_size = 250 * 4 * 2 = 2000

After (8 GPUs):
  micro_batch_size_per_gpu = 62
  gradient_accumulation_steps = 4
  train_batch_size = 62 * 4 * 8 = 1984

Note: Total batch size is approximately maintained (~2000).
      Per-GPU micro batch size decreases, but more GPUs process in parallel.
```

**Scale Down (16 GPUs -> 4 GPUs):**

```
Before (16 GPUs):
  micro_batch_size_per_gpu = 31
  gradient_accumulation_steps = 4
  train_batch_size = 31 * 4 * 16 = 1984

After (4 GPUs):
  micro_batch_size_per_gpu = 125
  gradient_accumulation_steps = 4
  train_batch_size = 125 * 4 * 4 = 2000

Note: Total batch size is approximately maintained (~2000).
      Per-GPU micro batch size increases to compensate for fewer GPUs.
```

### Data Parallelism Adjustment

When GPUs change, the data parallelism degree changes accordingly:

```
DP size = world_size / model_parallel_size

If model_parallel_size = 2:
  8 GPUs -> DP size = 4
  16 GPUs -> DP size = 8
  32 GPUs -> DP size = 16
```

The gradient averaging (all-reduce) is performed over the data parallel group, so changing DP size affects communication overhead.

## Checkpoint and Restart

### Elastic Checkpoint Strategy

For elasticity to work, the training state must be checkpointed regularly so that it can be restored after a reconfiguration:

```python
# Checkpoint with elastic-compatible state
def save_elastic_checkpoint(model_engine, optimizer, epoch, global_step):
    checkpoint_state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_engine.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "elastic_config": model_engine.elasticity_state(),
        "rng_state": torch.cuda.get_rng_state(),
        "numpy_rng_state": numpy.random.get_state(),
        "random_rng_state": random.getstate(),
    }
    torch.save(checkpoint_state, "elastic_checkpoint.pt")
```

### Restart Flow

```
1. Elastic change detected (GPU added/removed)
    |
    v
2. Save checkpoint with current training state
    |
    v
3. Gracefully shut down current training processes
    |
    v
4. Launch new processes with updated world_size
    |
    v
5. Load checkpoint on all ranks
    |
    v
6. Compute new elastic config (batch size, etc.)
    |
    v
7. Reinitialize DeepSpeed engine with new config
    |
    v
8. Restore training state (model weights, optimizer state, RNG)
    |
    v
9. Resume training from checkpointed global_step
```

### Universal Checkpoint Compatibility

DeepSpeed's universal checkpoint format is designed to work across different parallelism configurations:

- **ZeRO checkpoints**: Can be loaded with different DP sizes because states are partitioned
- **Model state**: Saved in a rank-agnostic format that can be redistributed
- **Optimizer state**: Each rank's optimizer state is saved independently

```python
# Loading checkpoint with different world size
model_engine.load_checkpoint(
    load_dir="checkpoints/",
    tag="latest",
    load_optimizer_states=True,
    load_lr_scheduler_states=True,
)
# The engine handles redistribution automatically
```

## Fault Tolerance

### Failure Detection

The elastic agent monitors worker processes and detects failures through:

1. **Process exit code**: Non-zero exit code indicates a failure
2. **Heartbeat timeout**: Workers send periodic heartbeats; missed heartbeats indicate a failure
3. **Signal handling**: SIGTERM/SIGKILL on worker processes

```python
class ElasticAgent:
    def monitor_workers(self):
        """Monitor worker processes for failures."""
        while True:
            for rank, worker in enumerate(self.workers):
                retcode = worker.poll()
                if retcode is not None:
                    # Worker has exited
                    if retcode != 0:
                        self.handle_failure(rank, retcode)
                    else:
                        self.handle_completion(rank)

            time.sleep(self.monitor_interval)
```

### Failure Handling Strategies

**Strategy 1: Scale Down (Default)**

Remove the failed worker and continue with fewer GPUs:

```python
def handle_failure(self, failed_rank, exit_code):
    # Remove failed worker
    self.workers.pop(failed_rank)
    new_world_size = len(self.workers)

    if new_world_size >= self.config.min_gpus:
        # Recompute config with fewer GPUs
        new_config = self.elasticity.compute_elastic_config(new_world_size)
        self.restart_all_workers(new_config)
    else:
        # Not enough GPUs, wait for more
        self.wait_for_resources(self.config.min_gpus)
```

**Strategy 2: Replace**

Try to restart the failed worker on the same or a different node:

```python
def handle_failure(self, failed_rank, exit_code):
    # Try to restart on the same node
    try:
        self.restart_worker(failed_rank)
    except RuntimeError:
        # Node is unavailable, try another node
        self.replace_worker_from_pool(failed_rank)
```

**Strategy 3: Restart All**

Shut down all workers and restart from checkpoint:

```python
def handle_failure(self, failed_rank, exit_code):
    # Kill all remaining workers
    self.kill_all_workers()

    # Wait for resources
    self.wait_for_resources(self.config.min_gpus)

    # Restart from checkpoint
    self.restart_all_workers(self.current_config)
```

## Elastic Agent

### Overview

The `ElasticAgent` class manages the lifecycle of worker processes, including launching, monitoring, and restarting them.

```python
class ElasticAgent:
    """Agent that manages elastic training worker processes."""

    def __init__(self, elasticity_config, user_script, user_args):
        self.config = elasticity_config
        self.user_script = user_script
        self.user_args = user_args
        self.workers = {}
        self.current_world_size = 0
        self.current_config = None

    def start(self):
        """Start the elastic training loop."""
        initial_world_size = self.discover_resources()
        self.run(initial_world_size)

    def run(self, world_size):
        """Run training with the specified world size."""
        config = self.elasticity.compute_elastic_config(world_size)
        self.current_config = config
        self.current_world_size = world_size
        self.launch_workers(config)

        try:
            self.monitor_workers()
        except ElasticRestart as e:
            self.handle_restart(e.new_world_size)

    def launch_workers(self, config):
        """Launch worker processes with the given configuration."""
        for rank in range(config["world_size"]):
            env = self.construct_worker_env(rank, config)
            worker = subprocess.Popen(
                ["python", self.user_script] + self.user_args,
                env=env,
            )
            self.workers[rank] = worker

    def monitor_workers(self):
        """Monitor workers and detect failures or completion."""
        while True:
            for rank, worker in list(self.workers.items()):
                retcode = worker.poll()
                if retcode is not None:
                    if retcode == 0:
                        # Normal completion
                        self.workers.pop(rank)
                        if not self.workers:
                            return  # All workers completed
                    else:
                        # Failure
                        raise ElasticRestart(
                            new_world_size=len(self.workers)
                        )
            time.sleep(1)

    def construct_worker_env(self, rank, config):
        """Construct environment variables for a worker."""
        env = os.environ.copy()
        env.update({
            "RANK": str(rank),
            "WORLD_SIZE": str(config["world_size"]),
            "LOCAL_RANK": str(rank % self.config.num_gpus_per_node),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "DS_ELASTIC_CONFIG": json.dumps(config),
        })
        return env
```

### Agent Lifecycle

```
ElasticAgent.start()
    |
    v
Discover initial resources (available GPUs)
    |
    v
Compute elastic config for initial world_size
    |
    v
Launch workers with computed config
    |
    v
Monitor workers
    |
    +-- All workers completed -> Exit successfully
    |
    +-- Worker failed -> Handle failure
        |
        +-- Scale down and restart with fewer GPUs
        |   (if remaining GPUs >= min_gpus)
        |
        +-- Wait for more resources
            (if remaining GPUs < min_gpus)
            |
            v
        New resources available
            |
            v
        Compute new elastic config
            |
            v
        Save checkpoint
            |
            v
        Kill old workers
            |
            v
        Launch new workers with new config
            |
            v
        Load checkpoint and resume
            |
            v
        Continue monitoring
```

### Resource Discovery

The elastic agent discovers available resources through several mechanisms:

```python
def discover_resources(self):
    """Discover available GPU resources."""
    # Method 1: Check CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        return len(visible_devices)

    # Method 2: Check torch.cuda.device_count()
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass

    # Method 3: Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True, text=True
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        pass

    # Method 4: Use hostfile
    if self.hostfile:
        hosts = parse_hostfile(self.hostfile)
        return sum(hosts.values())

    return 1  # Default to single GPU
```

## Integration with Launcher

### CLI Integration

Elastic training can be enabled via the DeepSpeed CLI:

```bash
# Enable elastic training
deepspeed --elastic_training --hostfile=myhostfile \
    train.py --deepspeed ds_config.json

# With specific GPU range
deepspeed --elastic_training --hostfile=myhostfile \
    train.py --deepspeed ds_config.json
```

### ds_config.json Integration

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 2000,
        "micro_batches": 4,
        "min_gpus": 4,
        "max_gpus": 64
    },
    "train_batch_size": -1,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "bf16": {
        "enabled": true
    }
}
```

Note: When elasticity is enabled, `train_batch_size` should be set to `-1` or omitted. The elastic system will compute the batch size dynamically.

### Engine Integration

The DeepSpeed engine checks for elastic configuration during initialization:

```python
# In DeepSpeedEngine.__init__()
if self.elasticity_enabled():
    self._configure_elasticity()
    # Override batch size with elastic config
    elastic_config = self.elasticity.compute_elastic_config(self.world_size)
    self.train_batch_size = elastic_config["train_batch_size"]
    self.train_micro_batch_size_per_gpu = elastic_config["train_micro_batch_size_per_gpu"]
    self.gradient_accumulation_steps = elastic_config["gradient_accumulation_steps"]
```

## Configuration Examples

### Minimal Elastic Configuration

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 256,
        "micro_batches": 2,
        "min_gpus": 1,
        "max_gpus": 8
    },
    "train_batch_size": -1,
    "zero_optimization": {
        "stage": 2
    },
    "bf16": {
        "enabled": true
    }
}
```

### Multi-Node Elastic Training

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 4096,
        "micro_batches": 8,
        "min_gpus": 8,
        "max_gpus": 128,
        "model_parallel_size": 2,
        "num_gpus_per_node": 8,
        "min_time": 300,
        "version": 0.2,
        "prefer_larger_batch": true
    },
    "train_batch_size": -1,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": true
}
```

### Spot Instance Training

For training on spot/preemptible instances where GPUs may be reclaimed at any time:

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 2048,
        "micro_batches": 4,
        "min_gpus": 2,
        "max_gpus": 64,
        "min_time": 120,
        "version": 0.2,
        "prefer_larger_batch": false
    },
    "train_batch_size": -1,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    },
    "bf16": {
        "enabled": true
    },
    "checkpoint": {
        "tag_validation": "Ignore"
    }
}
```

**Key considerations for spot instances:**
- Set `min_gpus` to the minimum viable GPU count (e.g., 2)
- Set `min_time` to at least 120 seconds to avoid frequent reconfigurations
- Use ZeRO-3 with CPU offloading so fewer GPUs can hold the model
- Enable frequent checkpointing (every N steps)

### Elastic with Tensor Parallelism

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 4096,
        "micro_batches": 4,
        "min_gpus": 4,
        "max_gpus": 32,
        "model_parallel_size": 4,
        "num_gpus_per_node": 8,
        "version": 0.2
    },
    "train_batch_size": -1,
    "zero_optimization": {
        "stage": 0
    },
    "bf16": {
        "enabled": true
    }
}
```

**Valid GPU counts** (must be divisible by `model_parallel_size=4`): 4, 8, 12, 16, 20, 24, 28, 32

**Invalid GPU counts**: 5, 6, 7, 9, 10, etc. (not divisible by 4)

### Version 0.1 Configuration (Legacy)

```json
{
    "elasticity": {
        "enabled": true,
        "max_acceptable_batch_size": 1000,
        "micro_batches": 2,
        "min_gpus": 1,
        "max_gpus": 16,
        "version": 0.1
    },
    "train_batch_size": -1,
    "zero_optimization": {
        "stage": 2
    }
}
```

**Version 0.1 batch size calculation:**
```
micro_batch_size_per_gpu = 1000 / (2 * 16) = 31
For world_size=4: train_batch_size = 31 * 2 * 4 = 248
For world_size=8: train_batch_size = 31 * 2 * 8 = 496
For world_size=16: train_batch_size = 31 * 2 * 16 = 992
```

## Constants Reference

The elasticity module defines the following constants:

```python
# Default values
DEFAULT_MAX_ACCEPTABLE_BATCH_SIZE = 2000
DEFAULT_MICRO_BATCHES = 4
DEFAULT_MIN_GPUS = 1
DEFAULT_MAX_GPUS = 10000
DEFAULT_MODEL_PARALLEL_SIZE = 1
DEFAULT_NUM_GPUS_PER_NODE = 8
DEFAULT_MIN_TIME = 60
DEFAULT_VERSION = 0.2

# Version identifiers
ELASTICITY_VERSION_1 = 0.1
ELASTICITY_VERSION_2 = 0.2

# Configuration keys
ELASTICITY_ENABLED = "enabled"
ELASTICITY_MAX_BATCH_SIZE = "max_acceptable_batch_size"
ELASTICITY_MICRO_BATCHES = "micro_batches"
ELASTICITY_MIN_GPUS = "min_gpus"
ELASTICITY_MAX_GPUS = "max_gpus"
ELASTICITY_MP_SIZE = "model_parallel_size"
ELASTICITY_GPUS_PER_NODE = "num_gpus_per_node"
ELASTICITY_MIN_TIME = "min_time"
ELASTICITY_VERSION = "version"
ELASTICITY_PREFER_LARGER_BATCH = "prefer_larger_batch"
ELASTICITY_IGNORE_NON_ELASTIC = "ignore_non_elastic_batch_info"
```

## Best Practices

### Setting min_gpus and max_gpus

1. **min_gpus**: Set to the minimum number of GPUs that can reasonably train the model. Consider memory requirements and training speed.

2. **max_gpus**: Set to the maximum number of GPUs available in your cluster. Do not set this higher than the actual maximum, as it affects batch size calculation.

### Choosing max_acceptable_batch_size

1. Start with the batch size you would use without elasticity
2. Consider the maximum batch size that does not degrade model quality (beyond the "critical batch size")
3. For large models, the critical batch size is typically 10-20% of the dataset size per epoch

### micro_batches Selection

1. Higher `micro_batches` provides more stable gradient estimates but increases latency per training step
2. Use `micro_batches` = 2-4 for small models, 4-8 for large models
3. Keep `micro_batches` constant across scaling events for consistent training dynamics

### Checkpointing Strategy

1. Save checkpoints frequently enough to minimize lost progress on restart (every 100-1000 steps)
2. Use asynchronous checkpointing (Nebula) to minimize checkpoint overhead
3. Store checkpoints on shared storage accessible from all nodes
4. Include RNG state in checkpoints for exact reproducibility

### Monitoring

1. Log elastic events (scale up/down) for debugging
2. Track batch size changes and their impact on training metrics
3. Monitor time between reconfigurations to detect thrashing
4. Use `min_time` to prevent rapid reconfiguration cycles
