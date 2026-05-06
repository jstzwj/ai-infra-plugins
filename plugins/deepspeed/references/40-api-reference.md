# Complete API Reference

## Overview

This reference documents the complete public API of DeepSpeed, including initialization functions, engine classes, configuration classes, CLI commands, and utility functions. Every public method, parameter, and return value is documented.

---

## Initialization Functions

### `deepspeed.initialize(args, model, optimizer=None, model_parameters=None, training_data=None, lr_scheduler=None, mpu=None, dist_init_required=None, collate_fn=None, config=None, config_params=None)`

The primary entry point for DeepSpeed training. Initializes the DeepSpeed engine with the given model, optimizer, and configuration.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `args` | Namespace or None | required | Command-line arguments. Must contain `local_rank` (int). If `None`, `deepspeed.init_distributed()` is called automatically. |
| `model` | nn.Module | required | The PyTorch model to wrap. Must be on the correct GPU before calling. |
| `optimizer` | Optimizer or None | `None` | Existing optimizer. If `None`, DeepSpeed creates one from config. If provided, DeepSpeed wraps it. |
| `model_parameters` | iterable or None | `None` | Model parameters for optimizer creation when `optimizer` is None. Required if optimizer is None and config specifies an optimizer. |
| `training_data` | Dataset or Iterable | `None` | Training dataset. If provided, DeepSpeed creates a DataLoader. |
| `lr_scheduler` | LRScheduler or None | `None` | Learning rate scheduler. DeepSpeed wraps it to coordinate with gradient accumulation. |
| `mpu` | object or None | `None` | Model parallel utility object for Megatron-LM integration. Must provide `get_model_parallel_group()`, `get_data_parallel_group()`, etc. |
| `dist_init_required` | bool or None | `None` | Whether to call `torch.distributed.init_process_group()`. If `None`, auto-detected. |
| `collate_fn` | callable or None | `None` | Collate function for DataLoader creation. |
| `config` | str or dict or None | `None` | DeepSpeed config as file path or dict. Mutually exclusive with `config_params`. |
| `config_params` | dict or None | `None` | DeepSpeed config dict. Alias for `config` when passing a dict directly. |

#### Returns

`tuple[DeepSpeedEngine or PipelineEngine, Optimizer, DataLoader, LRScheduler]`

| Return | Type | Description |
|---|---|---|
| `engine` | `DeepSpeedEngine` or `PipelineEngine` | The wrapped DeepSpeed model engine |
| `optimizer` | `Optimizer` | The wrapped optimizer |
| `training_dataloader` | `DataLoader` or `None` | The training DataLoader (None if `training_data` was None) |
| `lr_scheduler` | `LRScheduler` or `None` | The wrapped LR scheduler |

#### Example

```python
import deepspeed

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
    config="ds_config.json"
)

for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
```

---

### `deepspeed.init_inference(model, config=None, dtype=None, mp_size=1, checkpoint=None, replace_with_kernel_inject=None, replace_method='auto', ep_group=None, ep_size=1, tp_group=None, tp_size=1, moe_config=None, quantize_config=None, injection_policy=None)`

Initialize DeepSpeed for inference. Returns an inference engine optimized for low-latency, high-throughput serving.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | nn.Module | required | The model to optimize for inference |
| `config` | str or dict or None | `None` | DeepSpeed inference config |
| `dtype` | torch.dtype or None | `None` | Model data type. If None, auto-detected. |
| `mp_size` | int | `1` | Model parallelism size (tensor parallel) |
| `checkpoint` | str or dict or None | `None` | Path to DeepSpeed checkpoint or checkpoint dict |
| `replace_with_kernel_inject` | bool or None | `None` | Replace transformer layers with optimized kernels. If None, auto-detected. |
| `replace_method` | str | `'auto'` | Kernel replacement method: `"auto"`, `"bs"` (batch size optimized) |
| `ep_group` | ProcessGroup or None | `None` | Expert parallel process group for MoE models |
| `ep_size` | int | `1` | Expert parallel size |
| `tp_group` | ProcessGroup or None | `None` | Tensor parallel process group |
| `tp_size` | int | `1` | Tensor parallel size |
| `moe_config` | dict or None | `None` | MoE configuration |
| `quantize_config` | dict or None | `None` | Quantization configuration |
| `injection_policy` | dict or None | `None` | Custom kernel injection policy. Maps module types to their attention/MLP submodules. |

#### Returns

`InferenceEngine` or `InferenceEngineV2`

#### Configuration Cases

`init_inference` handles 4 distinct configuration paths:

1. **Config dict provided**: Use all settings from the config dict
2. **Checkpoint provided, no config**: Auto-configure from checkpoint metadata
3. **kwargs only**: Build config from individual parameters
4. **No config, no checkpoint**: Minimal config with defaults

#### Example

```python
import deepspeed

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

engine = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)

outputs = engine.generate(input_ids, max_length=100)
```

---

### `deepspeed.tp_model_init(tp_size=1, mp_group=None)`

Initialize a tensor-parallel model context. Use this to wrap model initialization code so that parameters are automatically partitioned across tensor-parallel ranks.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tp_size` | int | `1` | Tensor parallel size |
| `mp_group` | ProcessGroup or None | `None` | Model parallel process group |

#### Returns

A context manager or function wrapper.

#### Example

```python
import deepspeed

# Initialize with tensor parallelism
with deepspeed.tp_model_init(tp_size=4):
    model = MyLargeModel()
```

---

### `deepspeed.set_optimizer_flags(model)`

Set `use_muon` attribute on all model parameters based on dimensionality. Used with the Muon optimizer to distinguish 2D weight matrices from 1D parameters.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | nn.Module | required | The model whose parameters should be flagged |

#### Behavior

- `param.use_muon = True` if `param.ndim >= 2`
- `param.use_muon = False` if `param.ndim < 2`

---

## Engine Classes

### `DeepSpeedEngine`

The main training engine. Wraps a PyTorch model and provides ZeRO optimization, mixed precision training, gradient accumulation, checkpointing, and distributed training.

#### Constructor (Internal)

The `DeepSpeedEngine` is constructed by `deepspeed.initialize()` and should not be instantiated directly.

```python
# Do not call directly; use deepspeed.initialize()
engine, optimizer, dataloader, scheduler = deepspeed.initialize(...)
```

#### Core Methods

##### `forward(*args, **kwargs)`

Forward pass through the wrapped model. Handles mixed precision casting and parameter gathering (ZeRO-3).

| Parameter | Type | Description |
|---|---|---|
| `*args` | any | Positional arguments forwarded to the model |
| `**kwargs` | any | Keyword arguments forwarded to the model |

**Returns**: Model output (same as calling the original model).

##### `backward(loss, retain_graph=False)`

Backward pass. Computes gradients with mixed precision handling and gradient scaling.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss` | torch.Tensor | required | Loss tensor from forward pass |
| `retain_graph` | bool | `False` | Whether to retain the computation graph |

**Returns**: None. Gradients are accumulated in model parameters.

##### `step(lr_kwargs=None)`

Optimizer step. Applies gradient accumulation, gradient clipping, optimizer update, and learning rate scheduling. Only performs the actual update at gradient accumulation boundaries.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lr_kwargs` | dict or None | `None` | Additional kwargs for the LR scheduler step (e.g., `{"metric": val_loss}` for ReduceLROnPlateau) |

**Returns**: `None`

##### `train()`

Set the engine to training mode. Affects dropout, batch norm, etc.

##### `eval()`

Set the engine to evaluation mode.

##### `zero_grad()`

Zero all parameter gradients. Equivalent to `model.zero_grad()`.

##### `allreduce_gradients()`

Manually trigger gradient all-reduce across data parallel ranks. Normally called automatically by `step()`.

#### Checkpoint Methods

##### `save_checkpoint(save_dir, tag=None, client_state=None, save_latest=True)`

Save a DeepSpeed checkpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_dir` | str | required | Directory to save checkpoint |
| `tag` | str or None | `None` | Checkpoint tag (e.g., `"step_1000"`). If None, auto-generated from step count. |
| `client_state` | dict or None | `None` | User state to save alongside the checkpoint (e.g., RNG states, epoch) |
| `save_latest` | bool | `True` | Whether to update the `latest` file |

**Returns**: `str` -- The tag of the saved checkpoint, or `None` on failure.

##### `load_checkpoint(load_dir, tag=None, load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True, load_module_only=False, custom_load_fn=None)`

Load a DeepSpeed checkpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `load_dir` | str | required | Directory containing the checkpoint |
| `tag` | str or None | `None` | Checkpoint tag. If None, loads from `latest` file. |
| `load_module_strict` | bool | `True` | Strict loading for model state dict |
| `load_optimizer_states` | bool | `True` | Whether to load optimizer states |
| `load_lr_scheduler_states` | bool | `True` | Whether to load LR scheduler states |
| `load_module_only` | bool | `False` | Only load model weights, skip optimizer and scheduler |
| `custom_load_fn` | callable or None | `None` | Custom function for loading state dict |

**Returns**: `tuple[dict, dict]` -- (load_path, client_state)

##### `save_16bit_model(save_dir, save_filename=None)`

Save model weights in 16-bit (FP16/BF16) format. Useful for deployment.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_dir` | str | required | Directory to save the model |
| `save_filename` | str or None | `None` | Filename. If None, uses `model_name.pt`. |

**Returns**: `bool` -- True if save succeeded.

##### `module_state_dict()`

Get the model's state dict. In ZeRO-3, this gathers all parameters across ranks.

**Returns**: `OrderedDict` -- Model state dictionary.

#### Compilation Methods

##### `compile(backend="inductor", mode="default", dynamic=False, fullgraph=False)`

Compile the model using `torch.compile()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | str | `"inductor"` | Compilation backend |
| `mode` | str | `"default"` | Compilation mode: `"default"`, `"reduce-overhead"`, `"max-autotune"` |
| `dynamic` | bool | `False` | Enable dynamic shapes |
| `fullgraph` | bool | `False` | Compile entire model as single graph |

**Returns**: `self` (the engine, for chaining).

#### Scaling and Gradient Methods

##### `scale(tensor)`

Scale a tensor by the current loss scale (for FP16 training).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tensor` | torch.Tensor | required | Tensor to scale |

**Returns**: Scaled tensor.

##### `unscale(tensor)`

Unscale a tensor by the inverse loss scale.

##### `get_loss_scale()`

Get the current loss scale value.

**Returns**: `float`

##### `set_gradient_accumulation_boundary(is_boundary)`

Override whether the current step is a gradient accumulation boundary.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `is_boundary` | bool | required | Whether current step is a boundary |

##### `is_gradient_accumulation_boundary()`

Check if the current step is a gradient accumulation boundary (optimizer will update).

**Returns**: `bool`

##### `is_gradient_accumulation_step()`

Check if the current step is a gradient accumulation step (not a boundary).

**Returns**: `bool`

#### Memory and State Methods

##### `offload_states(save_dir)`

Offload all model states (parameters, optimizer states, gradients) to disk.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_dir` | str | required | Directory to save states |

##### `reload_states(save_dir)`

Reload previously offloaded states from disk.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_dir` | str | required | Directory containing offloaded states |

##### `destroy()`

Clean up distributed resources. Call at the end of training.

##### `empty_partition_cache()`

Empty the ZeRO partition cache to free memory.

#### Batch and Training Info Methods

##### `get_batch_info()`

Get current batch configuration.

**Returns**: `tuple[int, int, int]` -- (train_batch_size, micro_batch_size_per_gpu, gradient_accumulation_steps)

##### `set_train_batch_size(batch_size)`

Dynamically adjust the training batch size.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch_size` | int | required | New total training batch size |

##### `get_global_grad_norm()`

Get the global gradient norm across all parameters and ranks.

**Returns**: `float` -- Global gradient norm.

#### Properties

| Property | Type | Description |
|---|---|---|
| `global_steps` | int | Total number of optimizer steps completed |
| `global_samples` | int | Total number of training samples processed |
| `micro_steps` | int | Number of forward/backward calls in current accumulation window |
| `gradient_average` | bool | Whether gradients are averaged across data parallel ranks |
| `gradient_accumulation_steps` | int | Number of gradient accumulation steps |
| `loss_scale` | float | Current loss scale value |
| ` zeRO_optimization_stage` | int | Current ZeRO optimization stage (0, 1, 2, or 3) |
| `is_first_step()` | bool | Whether this is the first optimizer step |
| `is_zero3_singleton` | bool | Whether using ZeRO-3 singleton mode |

#### Context Managers

##### `no_sync()`

Context manager to disable gradient synchronization. Useful for manual gradient accumulation:

```python
with engine.no_sync():
    for micro_batch in micro_batches[:-1]:
        loss = engine(micro_batch)
        engine.backward(loss)

# Last micro-batch triggers synchronization
loss = engine(micro_batches[-1])
engine.backward(loss)
engine.step()
```

#### DataLoader Method

##### `deepspeed_io(training_data, collate_fn=None, data_sampler=None, dataloader=None, data_sampler_seed=None)`

Create a DeepSpeed-optimized DataLoader.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training_data` | Dataset | required | Training dataset |
| `collate_fn` | callable or None | `None` | Collate function |
| `data_sampler` | Sampler or None | `None` | Custom data sampler |
| `dataloader` | DataLoader or None | `None` | Existing DataLoader to wrap |
| `data_sampler_seed` | int or None | `None` | Seed for the data sampler |

**Returns**: `DataLoader`

---

### `PipelineEngine`

The pipeline-parallel training engine. Extends `DeepSpeedEngine` with pipeline parallelism support.

#### Additional Methods

##### `forward(input_ids, attention_mask=None, labels=None)`

Forward pass through the pipeline. Handles micro-batch splitting and inter-stage communication.

##### `backward(loss)`

Backward pass through the pipeline. Handles gradient computation across stages.

##### `step()`

Optimizer step with pipeline synchronization.

#### Pipeline-Specific Properties

| Property | Type | Description |
|---|---|---|
| `num_stages` | int | Number of pipeline stages |
| `stage_id` | int | Current stage ID |
| `num_microbatches` | int | Number of micro-batches per pipeline batch |
| `pipeline_parallel_size` | int | Pipeline parallel degree |

---

### `DeepSpeedHybridEngine`

The hybrid engine combines training and inference capabilities. It uses ZeRO for training and switches to optimized inference kernels for evaluation/generation.

#### Methods

##### `generate(input_ids, **kwargs)`

Generate text using the hybrid engine's inference mode.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | torch.Tensor | required | Input token IDs |
| `**kwargs` | any | - | Additional arguments for `model.generate()` |

**Returns**: `torch.Tensor` -- Generated token IDs.

---

### `InferenceEngine`

The V1 inference engine optimized for single-query and batch inference.

#### Constructor

```python
from deepspeed.inference.engine import InferenceEngine

engine = InferenceEngine(model, config=config)
```

#### Methods

##### `forward(inputs, **kwargs)`

Run inference forward pass with kernel replacements.

##### `generate(input_ids, **kwargs)`

Generate tokens using the inference engine.

#### Properties

| Property | Type | Description |
|---|---|---|
| `model` | nn.Module | The wrapped model |
| `config` | dict | Inference configuration |
| `mp_size` | int | Model parallel size |

---

### `InferenceEngineV2`

The V2 inference engine with improved architecture, better kernel injection, and support for more model types.

#### Constructor

```python
from deepspeed.inference.v2.engine_v2 import InferenceEngineV2

engine = InferenceEngineV2(model, config=config)
```

#### Key Differences from V1

- Support for quantization (INT8, INT4)
- Improved kernel injection policy
- Better memory management for long sequences
- Native tensor parallelism support
- Continuous batching support

---

## Configuration Classes

### `DeepSpeedConfig`

The main configuration class. Parses and validates the DeepSpeed JSON configuration.

#### Constructor

```python
from deepspeed.runtime.config import DeepSpeedConfig

config = DeepSpeedConfig("ds_config.json")
# or
config = DeepSpeedConfig(config_dict)
```

#### Key Methods

##### `get_batch_info()`

Get batch configuration.

**Returns**: `tuple[int, int, int]` -- (train_batch_size, micro_batch_size_per_gpu, gradient_accumulation_steps)

#### Key Properties

| Property | Type | Description |
|---|---|---|
| `train_batch_size` | int | Total training batch size |
| `train_micro_batch_size_per_gpu` | int | Micro batch size per GPU |
| `gradient_accumulation_steps` | int | Gradient accumulation steps |
| `zero_optimization_stage` | int | ZeRO stage (0-3) |
| `fp16_enabled` | bool | Whether FP16 is enabled |
| `bf16_enabled` | bool | Whether BF16 is enabled |
| `gradient_clipping` | float | Gradient clipping value |
| `optimizer_name` | str | Optimizer type name |
| `optimizer_params` | dict | Optimizer hyperparameters |
| `scheduler_name` | str | LR scheduler type |
| `scheduler_params` | dict | LR scheduler parameters |
| `wall_clock_breakdown` | bool | Whether timing breakdown is enabled |
| `memory_breakdown` | bool | Whether memory breakdown is enabled |
| `checkpoint_config` | dict | Checkpoint configuration |
| `fp16_config` | dict | FP16 configuration |
| `bf16_config` | dict | BF16 configuration |
| `zero_config` | dict | ZeRO configuration |
| `activation_checkpointing_config` | dict | Activation checkpointing configuration |
| `pipeline_config` | dict | Pipeline parallelism configuration |
| `tensor_parallel_config` | dict | Tensor parallelism configuration |
| `hybrid_engine_config` | dict | Hybrid engine configuration |

---

### `DeepSpeedInferenceConfig`

Configuration class for inference.

#### Constructor

```python
from deepspeed.runtime.config import DeepSpeedInferenceConfig

config = DeepSpeedInferenceConfig(
    dtype=torch.float16,
    tensor_parallel={"tp_size": 2},
    replace_with_kernel_inject=True,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dtype` | torch.dtype or str | `"fp16"` | Model data type |
| `tensor_parallel` | dict | `{}` | Tensor parallel configuration |
| `replace_with_kernel_inject` | bool | `True` | Enable kernel replacement |
| `injection_policy` | dict or None | `None` | Custom injection policy |
| `mp_size` | int | `1` | Model parallel size |
| `ep_size` | int | `1` | Expert parallel size |
| `moe_config` | dict or None | `None` | MoE configuration |
| `quantize_config` | dict or None | `None` | Quantization configuration |
| `checkpoint` | str or None | `None` | Checkpoint path |

---

### `HybridEngineConfig`

Configuration for the DeepSpeed hybrid engine.

#### Constructor

```python
from deepspeed.runtime.config import HybridEngineConfig

config = HybridEngineConfig(
    enabled=True,
    max_out_tokens=512,
    inference_tp_size=2,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `False` | Enable hybrid engine |
| `max_out_tokens` | int | `512` | Maximum output tokens for inference |
| `inference_tp_size` | int | `1` | Tensor parallel size for inference |
| `pin_parameters` | bool | `True` | Pin parameters in GPU memory |
| `release_parameters` | bool | `False` | Release parameters after inference |

---

### `DeepSpeedConfigWriter`

Utility class for programmatically generating DeepSpeed config JSON files.

#### Methods

##### `add_config(key, value)`

Add a configuration entry.

```python
from deepspeed.runtime.config import DeepSpeedConfigWriter

writer = DeepSpeedConfigWriter()
writer.add_config("train_batch_size", 32)
writer.add_config("fp16", {"enabled": True})
writer.add_config("zero_optimization", {"stage": 2})

writer.write_json("ds_config.json")
```

##### `write_json(path)`

Write the configuration to a JSON file.

---

## CLI Commands

### `deepspeed`

The main DeepSpeed launcher command.

```bash
deepspeed [launcher_args] train.py [user_args]
```

#### Launcher Arguments

| Argument | Default | Description |
|---|---|---|
| `--hostfile` | `None` | Path to hostfile listing available nodes and GPU slots |
| `--include` | `None` | Specify nodes/GPUs to use: `"node1:0-3,node2:0-3"` |
| `--exclude` | `None` | Specify nodes/GPUs to exclude |
| `--num_nodes` | `-1` (all) | Number of nodes to use |
| `--num_gpus` | `-1` (all) | Number of GPUs per node |
| `--master_port` | `29500` | Port for distributed communication |
| `--master_addr` | auto | Address of the master node |
| `--launcher` | `"pdsh"` | Launcher backend: `"pdsh"`, `"openmpi"`, `"mvapich"`, `"slurm"` |
| `--launcher_args` | `""` | Additional arguments for the launcher |
| `--bind_cores_to_rank` | `False` | Bind CPU cores to each rank |
| `--bind_core_list` | `None` | List of CPU cores to bind to |
| `--force_multi` | `False` | Force multi-node mode even with 1 node |
| `--save_launch_script` | `None` | Save the launch script to this path |

#### Examples

```bash
# Single node, 8 GPUs
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json

# Multi-node via hostfile
deepspeed --hostfile=my_hostfile train.py --deepspeed ds_config.json

# Specific GPUs on specific nodes
deepspeed --include="node1:0-3,node2:0-3" train.py --deepspeed ds_config.json

# With SLURM launcher
deepspeed --launcher=slurm train.py --deepspeed ds_config.json
```

### `ds_report`

Display environment and diagnostic information.

```bash
ds_report
```

See [38-debugging-and-troubleshooting.md](38-debugging-and-troubleshooting.md) for full documentation.

### `ds_elastic`

Launch elastic training with DeepSpeed.

```bash
ds_elastic train.py [args]
```

### Hostfile Format

```
node1 slots=8
node2 slots=8
node3 slots=4
```

Each line specifies a hostname and the number of available GPU slots.

---

## Helper Functions

### `deepspeed.init_distributed(dist_backend=None, auto_mpi_discovery=True, distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True, timeout=None, init_method=None)`

Initialize the PyTorch distributed process group. Called automatically by `deepspeed.initialize()` if not already initialized.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dist_backend` | str or None | `None` | Distributed backend (`"nccl"`, `"gloo"`, `"mpi"`). Auto-detected if None. |
| `auto_mpi_discovery` | bool | `True` | Auto-discover MPI environment for multi-node setup |
| `distributed_port` | int | `29500` | Port for distributed communication |
| `verbose` | bool | `True` | Print initialization messages |
| `timeout` | timedelta or None | `None` | Process group timeout. Defaults to `DEEPSPEED_TIMEOUT` env var or 30 minutes. |
| `init_method` | str or None | `None` | URL specifying how to initialize the process group |

### `deepspeed.add_config_arguments(parser)`

Add DeepSpeed-specific arguments to an argparse parser.

```python
import argparse
import deepspeed

parser = argparse.ArgumentParser()
deepspeed.add_config_arguments(parser)
args = parser.parse_args()
```

Adds the `--deepspeed` argument (path to config file) and `--deepspeed_config` alias.

### `deepspeed.default_inference_config()`

Get the default inference configuration dictionary.

**Returns**: `dict` -- Default inference config.

---

## ZeRO Utility Functions

### `deepspeed.zero.GatheredParameters(params, modifier_rank=None, fwd_module=None, enabled=True)`

Context manager for gathering ZeRO-3 sharded parameters. Required when accessing parameters outside of forward/backward (e.g., during evaluation, checkpointing, or weight initialization).

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | required | Parameters to gather |
| `modifier_rank` | int or None | `None` | Rank that can modify gathered parameters. If None, all ranks can read. |
| `fwd_module` | nn.Module or None | `None` | Module to register forward hooks on |
| `enabled` | bool | `True` | Enable/disable gathering (for conditional use) |

#### Example

```python
import deepspeed

# Evaluate with full parameters
with deepspeed.zero.GatheredParameters(model.parameters()):
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            output = model(batch)
```

### `deepspeed.zero.Init(module=None, config_dict_or_path=None, enabled=True, mem_efficient_linear=True, mpu=None)`

Context manager for initializing models with ZeRO-3 parameter partitioning. Parameters are allocated in a sharded manner, enabling models larger than a single GPU's memory.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `module` | nn.Module or None | `None` | Module to initialize (if None, acts as context manager) |
| `config_dict_or_path` | dict or str or None | `None` | DeepSpeed config |
| `enabled` | bool | `True` | Enable ZeRO-3 initialization |
| `mem_efficient_linear` | bool | `True` | Use memory-efficient linear layers |
| `mpu` | object or None | `None` | Model parallel utility |

#### Example

```python
import deepspeed

# Initialize model with ZeRO-3 partitioning
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

# Now model's parameters are sharded across GPUs
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

### `deepspeed.zero.register_external_parameter(module, parameter)`

Register a parameter as external to ZeRO-3's partitioning. The parameter will not be sharded and will be available on all ranks.

```python
import deepspeed

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_weight = nn.Parameter(torch.randn(100, 100))

    def forward(self, x):
        # Register as external so it's not sharded in ZeRO-3
        deepspeed.zero.register_external_parameter(self, self.shared_weight)
        return x @ self.shared_weight
```

---

## Quick Reference: Complete Training Loop

```python
import deepspeed
import argparse

# 1. Parse arguments
parser = argparse.ArgumentParser()
deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

# 2. Create model
model = MyModel()

# 3. Initialize DeepSpeed
model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=args.deepspeed_config,
)

# 4. Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward
        outputs = model_engine(batch)
        loss = outputs.loss

        # Backward
        model_engine.backward(loss)

        # Optimizer step (handles gradient accumulation)
        model_engine.step()

    # Save checkpoint
    tag = f"epoch_{epoch}"
    model_engine.save_checkpoint("./checkpoints", tag=tag)

# 5. Save final model
model_engine.save_16bit_model("./output", "model.pt")

# 6. Cleanup
model_engine.destroy()
```

---

## Complete Configuration Parameter Table

### Top-Level Parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `train_batch_size` | int | `32` | Total training batch size across all GPUs |
| `train_micro_batch_size_per_gpu` | int | auto | Micro batch size per GPU |
| `gradient_accumulation_steps` | int | auto | Gradient accumulation steps |
| `optimizer` | dict | `{}` | Optimizer configuration |
| `scheduler` | dict | `{}` | LR scheduler configuration |
| `fp16` | dict | `{}` | FP16 mixed precision configuration |
| `bf16` | dict | `{}` | BF16 mixed precision configuration |
| `zero_optimization` | dict | `{}` | ZeRO optimization configuration |
| `gradient_clipping` | float | `0.0` | Gradient clipping value (0 = disabled) |
| `grad_accum_dtype` | str | `None` | Gradient accumulation dtype |
| `activation_checkpointing` | dict | `{}` | Activation checkpointing configuration |
| `checkpoint` | dict | `{}` | Checkpoint configuration |
| `pipeline` | dict | `{}` | Pipeline parallelism configuration |
| `tensor_parallel` | dict | `{}` | Tensor parallelism configuration |
| `sparse_attention` | dict | `{}` | Sparse attention configuration |
| `wall_clock_breakdown` | bool | `false` | Enable timing breakdown |
| `memory_breakdown` | bool | `false` | Enable memory breakdown |
| `validation_mode` | str | `"warn"` | Validation strictness |
| `dataloader_drop_last` | bool | `false` | Drop last incomplete batch |
| `compile` | dict | `{}` | torch.compile configuration |
| `comms_logger` | dict | `{}` | Communication logging configuration |
| `data_types` | dict | `{}` | Data type configuration |

### FP16 Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable FP16 training |
| `loss_scale` | float | `0` | Static loss scale (0 = dynamic) |
| `initial_scale_power` | int | `16` | Initial scale power (2^16 = 65536) |
| `loss_scale_window` | int | `1000` | Window for scale adjustment |
| `hysteresis` | int | `2` | Hysteresis factor |
| `min_loss_scale` | float | `1` | Minimum loss scale |
| `fp16_master_weights_and_grads` | bool | `false` | FP16 master weights |

### BF16 Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable BF16 training |
| `check_overflow` | bool | `false` | Check for gradient overflow |

### ZeRO Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `stage` | int | `0` | ZeRO stage (0, 1, 2, 3) |
| `contiguous_gradients` | bool | `true` | Use contiguous gradient buffers |
| `reduce_scatter` | bool | `true` | Use reduce-scatter instead of all-reduce |
| `reduce_bucket_size` | int | `5e8` | Bucket size for gradient reduction |
| `allgather_bucket_size` | int | `5e8` | Bucket size for all-gather |
| `overlap_comm` | bool | `false` | Overlap communication with computation |
| `offload_optimizer` | dict | `{}` | Optimizer offloading configuration |
| `offload_param` | dict | `{}` | Parameter offloading configuration |
| `sub_group_size` | int | `1e9` | Sub-group size for parameter partitioning |
| `stage3_prefetch_bucket_size` | int | `5e8` | Prefetch bucket size for ZeRO-3 |
| `stage3_param_persistence_threshold` | int | `1e5` | Threshold for keeping params unsharded |
| `stage3_max_live_parameters` | int | `1e9` | Max concurrent parameters in GPU memory |
| `stage3_max_reuse_distance` | int | `1e9` | Reuse distance for parameter caching |

### Optimizer Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | str | required | Optimizer name from `DEEPSPEED_OPTIMIZERS` |
| `params.lr` | float | `0.001` | Learning rate |
| `params.betas` | list[float] | `[0.9, 0.999]` | Adam beta coefficients |
| `params.eps` | float | `1e-8` | Numerical stability epsilon |
| `params.weight_decay` | float | `0.0` | Weight decay |
| `params.momentum` | float | `0.95` | Momentum (Muon) |
| `params.nesterov` | bool | `true` | Nesterov momentum (Muon) |
| `params.ns_steps` | int | `5` | Newton-Schulz iterations (Muon) |
| `params.ns_method` | str | `"standard"` | Newton-Schulz method (Muon) |
| `params.adam_lr` | float | `1e-3` | Adam LR for non-Muon params (Muon) |

### Activation Checkpointing Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `partition_activations` | bool | `false` | Partition activations across GPUs |
| `cpu_checkpointing` | bool | `false` | Offload activations to CPU |
| `contiguous_memory_optimization` | bool | `false` | Use contiguous memory for checkpoints |
| `number_checkpoints` | int or None | `None` | Number of checkpoint segments |
| `synchronize_checkpoint_boundary` | bool | `false` | Synchronize at checkpoint boundaries |

### Checkpoint Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `checkpoint_dir` | str | required | Checkpoint directory |
| `tag_interval` | int | `None` | Checkpoint tag interval |
| `size_limit` | int | `0` | Max checkpoint size (bytes) |
| `parallel_save` | bool | `false` | Enable parallel save |
