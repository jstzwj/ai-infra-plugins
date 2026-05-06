# DeepSpeed Activation Checkpointing

## Overview

Activation checkpointing (also known as gradient checkpointing or activation recomputation) is a memory optimization technique that trades computation for memory. Instead of storing all intermediate activations for the backward pass, only a subset of activations (checkpoints) are retained. During the backward pass, the intermediate activations between checkpoints are recomputed from the saved checkpoints.

DeepSpeed provides an advanced activation checkpointing system located in `deepspeed/runtime/activation_checkpointing/` that supports partitioning activations across GPUs (for pipeline parallelism), CPU offloading of checkpoints, and CUDA RNG state management for reproducibility.

---

## Module Architecture

```
deepspeed/runtime/activation_checkpointing/
    __init__.py
    config.py           # DeepSpeedActivationCheckpointingConfig
    checkpointing.py    # Core checkpoint function and utilities
    setup.py            # Engine integration and initialization
```

### Related Source Files

```
deepspeed/runtime/engine.py     # DeepSpeedEngine integrates checkpointing
```

---

## DeepSpeedActivationCheckpointingConfig

The `DeepSpeedActivationCheckpointingConfig` class (defined in `config.py`) parses and validates the activation checkpointing configuration from the DeepSpeed JSON config.

### Configuration Structure

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "checkpoint_in_cpu": false,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `partition_activations` | bool | `false` | When `true`, partitions (splits) activation checkpoints across data-parallel GPUs. Each GPU stores only its portion. Essential for pipeline parallelism with activation memory optimization. |
| `cpu_checkpointing` | bool | `false` | When `true`, stores activation checkpoints on CPU instead of GPU. Reduces GPU memory at the cost of CPU-GPU transfer overhead during recomputation. |
| `contiguous_memory_optimization` | bool | `false` | When `true`, stores checkpoints in contiguous memory buffers to reduce memory fragmentation. Requires pre-allocating a contiguous memory pool. |
| `number_checkpoints` | int or null | `null` | Number of checkpoint boundaries in the model. If `null`, uses the number of transformer layers. Controls the granularity of checkpointing. |
| `checkpoint_in_cpu` | bool | `false` | Alias for `cpu_checkpointing`. When `true`, checkpoints are moved to CPU. |
| `synchronize_checkpoint_boundary` | bool | `false` | When `true`, adds synchronization barriers at checkpoint boundaries. Required for pipeline parallelism to ensure correct ordering. |
| `profile` | bool | `false` | When `true`, profiles activation checkpointing overhead for debugging. |

---

## Core Checkpoint Function

The `checkpoint()` function in `checkpointing.py` is the fundamental building block. It wraps a function call and ensures its intermediate activations are recomputed during the backward pass rather than stored.

### Function Signature

```python
def checkpoint(function, *args, **kwargs):
    """Activation checkpointing wrapper.

    Args:
        function: The function to checkpoint. Must be a callable that takes
                  *args and **kwargs and returns a tensor or tuple of tensors.
        *args: Positional arguments to pass to function.
        **kwargs: Keyword arguments to pass to function.

    Returns:
        The output of function(*args, **kwargs), but with intermediate
        activations freed and scheduled for recomputation during backward.
    """
```

### How It Works

The `checkpoint()` function operates by:

1. **Forward pass (first evaluation)**: Runs the function normally but marks the inputs as "detached" (no gradient tracking). The function's intermediate activations are NOT stored.
2. **Saving inputs**: Only the function inputs (tensors that require gradients) are saved.
3. **Backward pass (recomputation)**: When gradients flow back to this function, the function is re-executed with the saved inputs to recompute the intermediate activations. Gradients are then computed through this recomputed forward pass.

```python
# Simplified checkpoint pseudocode
def checkpoint(function, *args):
    # Save only the inputs (not the intermediate activations)
    saved_inputs = detach_variable(args)

    # Run the function and get outputs (activations are not saved)
    with torch.no_grad():
        outputs = function(*args)

    # Attach a hook to recompute during backward
    def recomputation_hook(grad_outputs):
        # Recompute the forward pass
        with torch.enable_grad():
            recomputed_outputs = function(*saved_inputs)
        # Compute gradients through the recomputed forward pass
        torch.autograd.backward(recomputed_outputs, grad_outputs)
        return

    outputs.register_hook(recomputation_hook)
    return outputs
```

---

## detach_variable()

The `detach_variable()` function is a utility used by checkpointing to prepare inputs for storage:

```python
def detach_variable(inputs):
    """Detach tensors from the computation graph while preserving requires_grad info.

    This creates copies of the input tensors that are detached from the
    autograd graph but remembers which ones required gradients. During
    recomputation, these detached tensors are re-attached to the graph.
    """
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                x = inp.detach()
                x.requires_grad = inp.requires_grad
                out.append(x)
            else:
                out.append(inp)
        return tuple(out)
    else:
        if isinstance(inputs, torch.Tensor):
            out = inputs.detach()
            out.requires_grad = inputs.requires_grad
            return out
        return inputs
```

Key behaviors:
- **Detaches** tensors from the computation graph (breaks the autograd chain)
- **Preserves** `requires_grad` flags so gradients flow correctly during recomputation
- **Handles** non-tensor inputs (ints, bools, None) by passing them through unchanged
- **No copying**: `detach()` does not copy data; it creates a new tensor sharing the same storage

---

## _set_cuda_rng_state()

The `_set_cuda_rng_state()` function manages CUDA random number generator (RNG) state, which is critical for reproducibility when using activation checkpointing with dropout.

### Why RNG State Management Matters

When a layer uses dropout (or any random operation), the random values depend on the CUDA RNG state. If activations are recomputed during the backward pass, the random values must be identical to those used in the original forward pass. Otherwise, the gradients will be incorrect.

```python
def _set_cuda_rng_state(new_state, device=-1):
    """Set the CUDA RNG state for the specified device.

    Args:
        new_state: The RNG state to restore (from torch.cuda.get_rng_state()).
        device: The CUDA device index. -1 means current device.
    """
    if device == -1:
        device = torch.cuda.current_device()
    # The RNG state is stored as a ByteTensor
    if isinstance(new_state, torch.Tensor):
        # Cast to the correct device
        new_state_copy = new_state.clone()
        torch.cuda.set_rng_state(new_state_copy)
```

### RNG Checkpointing Flow

```python
# During forward pass (checkpoint):
rng_state = torch.cuda.get_rng_state()  # Save RNG state
with torch.no_grad():
    output = function(*args)
# rng_state is saved alongside the inputs

# During backward pass (recomputation):
torch.cuda.set_rng_state(saved_rng_state)  # Restore RNG state
with torch.enable_grad():
    recomputed_output = function(*saved_inputs)
# Now the random operations (dropout) produce the same values
```

### Multi-GPU RNG State

For models using tensor parallelism or pipeline parallelism, each GPU has its own CUDA RNG state. DeepSpeed manages per-device RNG states:

```python
# Save all device RNG states
def get_all_rng_states():
    states = {}
    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            states[device_id] = torch.cuda.get_rng_state()
    return states
```

---

## PARTITION_ACTIVATIONS Flag

The `PARTITION_ACTIVATIONS` flag controls whether activation checkpoints are partitioned (split) across data-parallel GPUs. This is primarily used with pipeline parallelism.

### How Partition Activations Work

Without partitioning, each GPU stores a full copy of its activation checkpoints. With partitioning, each activation checkpoint is split across the data-parallel group, and each GPU stores only its partition:

```
Without partitioning (4 GPUs, each stores full activation):
GPU 0: [a0, a1, a2, a3]  (full activation for its pipeline stage)
GPU 1: [a0, a1, a2, a3]
GPU 2: [a0, a1, a2, a3]
GPU 3: [a0, a1, a2, a3]

With partitioning (each GPU stores 1/4):
GPU 0: [a0]  (only partition 0)
GPU 1: [a1]  (only partition 1)
GPU 2: [a2]  (only partition 2)
GPU 3: [a3]  (only partition 3)
```

During recomputation, the partitions are gathered via all-gather:

```python
# Partition save
partition = activation.chunk(world_size)[rank]
saved_partitions[rank] = partition.detach()

# Recomputation restore
all_partitions = [torch.empty_like(partition) for _ in range(world_size)]
dist.all_gather(all_partitions, saved_partition)
full_activation = torch.cat(all_partitions)
```

### Memory Savings

With `partition_activations=True` and `N` data-parallel GPUs:
- Each GPU stores `1/N` of the activation checkpoint memory
- Total memory across all GPUs remains the same
- But per-GPU memory is reduced by `(N-1)/N`

### When to Use Partition Activations

- **Pipeline parallelism**: Each pipeline stage has different activation sizes; partitioning helps balance memory.
- **Large batch sizes**: When activation memory dominates GPU memory.
- **Combined with ZeRO**: Partition activations complement ZeRO's optimizer state partitioning.

---

## CPU_CHECKPOINT Flag

The `CPU_CHECKPOINT` flag (also configurable as `cpu_checkpointing` or `checkpoint_in_cpu`) controls whether activation checkpoints are offloaded to CPU memory.

### How CPU Checkpointing Works

```python
# During forward pass
activation = layer(hidden_states)  # Compute activation
if cpu_checkpointing:
    # Move activation to CPU
    cpu_activation = activation.cpu().pin_memory()
    # Free GPU memory
    activation = None
    # Store CPU copy
    checkpoint_store[layer_idx] = cpu_activation

# During backward pass (recomputation)
if cpu_checkpointing:
    # Move activation back to GPU
    activation = checkpoint_store[layer_idx].to(device, non_blocking=True)
    # Recompute layer output
    recomputed = layer(activation)
```

### Performance Considerations

- **CPU-GPU transfer overhead**: Each checkpoint requires a CPU-to-GPU transfer during recomputation. With PCIe 4.0 (32 GB/s), a 100 MB checkpoint takes ~3ms to transfer.
- **Pin memory**: Always use `pin_memory=True` for faster transfers via DMA.
- **Overlap computation with transfer**: DeepSpeed overlaps the CPU-GPU transfer with other computation where possible.
- **CPU memory requirement**: Ensure sufficient CPU RAM is available for all checkpoints.

### Memory Savings Estimate

```
GPU memory saved = number_checkpoints * avg_activation_size_per_checkpoint
CPU memory required = number_checkpoints * avg_activation_size_per_checkpoint
```

For a 7B parameter model with 32 layers:
- Each layer's activation: ~200 MB (batch_size=4, seq_len=2048, hidden_dim=4096)
- 32 checkpoints: ~6.4 GB
- GPU memory saved: ~6.4 GB
- CPU memory required: ~6.4 GB

---

## CONTIGUOUS_CHECKPOINTING Flag

The `CONTIGUOUS_CHECKPOINTING` flag (also configurable as `contiguous_memory_optimization`) controls whether checkpoints are stored in pre-allocated contiguous memory buffers.

### Why Contiguous Memory Matters

GPU memory fragmentation occurs when tensors of different sizes are allocated and freed at different times. Activation checkpointing exacerbates this because:
1. Checkpoints are allocated during the forward pass
2. Freed during the backward pass
3. Recomputed (re-allocated) during the backward pass

This pattern can fragment the CUDA memory allocator, leading to out-of-memory errors even when sufficient total memory exists.

### How Contiguous Checkpointing Works

```python
# Pre-allocate a contiguous buffer
total_checkpoint_memory = number_checkpoints * avg_checkpoint_size
contiguous_buffer = torch.empty(total_checkpoint_memory, device='cuda')

# During forward pass, allocate checkpoints from the contiguous buffer
offset = 0
for layer_idx in range(num_layers):
    size = layer_checkpoint_sizes[layer_idx]
    checkpoint = contiguous_buffer[offset:offset+size].view(checkpoint_shape)
    checkpoint.copy_(activation)
    offset += size
```

### When to Use Contiguous Checkpointing

- **Memory fragmentation issues**: When you see CUDA OOM errors but the model should fit in memory.
- **Variable-length sequences**: When sequence lengths vary across batches.
- **Combined with partition_activations**: Both flags can be used together for maximum memory savings.

---

## Number of Checkpointing Layers (num_checkpoints)

The `number_checkpoints` parameter (also called `num_checkpoints`) controls how many checkpoint boundaries are placed in the model.

### Checkpoint Granularity

- **More checkpoints = less memory, more recomputation**:
  - Each checkpoint reduces memory by the amount of activations between checkpoints
  - Each checkpoint adds recomputation cost for the layers between checkpoints
- **Fewer checkpoints = more memory, less recomputation**:
  - Fewer recomputations but higher memory usage

### Optimal Number of Checkpoints

The optimal number depends on the model architecture:

```
For a transformer with N layers:
- Every layer: num_checkpoints = N (maximum memory savings, 2x forward cost)
- Every 2 layers: num_checkpoints = N/2
- Every 4 layers: num_checkpoints = N/4
- None: num_checkpoints = 0 (no checkpointing, maximum memory usage)
```

### Setting num_checkpoints

```json
{
    "activation_checkpointing": {
        "number_checkpoints": 16
    }
}
```

If `number_checkpoints` is `null` or not specified, DeepSpeed automatically determines the number based on the model's transformer layers.

---

## Partition Activation Support

Beyond the `partition_activations` flag, DeepSpeed provides additional infrastructure for managing partitioned activations in pipeline-parallel scenarios.

### Activation Partitioning in Pipeline Parallelism

In pipeline parallelism, each pipeline stage processes a micro-batch and passes activations to the next stage. With activation checkpointing and partitioning:

1. **Stage 0** computes and partitions activations, sends partition to Stage 1
2. **Stage 1** receives its partition, stores it, and uses it for recomputation later
3. During backward, each stage gathers the full activation from all partitions

```python
# Pipeline stage with partitioned activations
class PipelineStage:
    def forward(self, input_activations):
        # Receive partitioned input from previous stage
        if self.stage_id > 0:
            input_activations = self.gather_partitioned_activations()

        # Run layers with checkpointing
        for layer in self.layers:
            if should_checkpoint(layer):
                hidden = checkpoint(layer, hidden)
            else:
                hidden = layer(hidden)

        # Partition output activations for next stage
        if self.partition_activations:
            output_partition = self.partition_activations(hidden)
            return output_partition
        return hidden
```

### Synchronization at Checkpoint Boundaries

When `synchronize_checkpoint_boundary=True`, DeepSpeed inserts `torch.cuda.synchronize()` calls at each checkpoint boundary. This ensures that all GPUs have completed their forward pass up to the checkpoint before proceeding, which is necessary for:

- Correct pipeline-parallel scheduling
- Deterministic memory usage patterns
- Accurate profiling

---

## RNG Checkpointing for Model Parallelism

When using tensor model parallelism (Megatron-style), each GPU computes a different part of the same layer. Dropout operations within these layers must produce consistent results across recomputations.

### Per-Device RNG State Management

```python
# In Megatron-style tensor parallelism, each GPU has its own CUDA RNG state
# but must be consistent between forward and recomputation

class ModelParallelRNGState:
    """Manages CUDA RNG state for tensor model parallelism."""

    def __init__(self):
        self.tensor_model_parallel_rng_states = {}

    def save(self):
        """Save the current RNG state for the model parallel group."""
        for device in self.model_parallel_devices:
            with torch.cuda.device(device):
                self.tensor_model_parallel_rng_states[device] = (
                    torch.cuda.get_rng_state()
                )

    def restore(self):
        """Restore the saved RNG state."""
        for device, state in self.tensor_model_parallel_rng_states.items():
            with torch.cuda.device(device):
                torch.cuda.set_rng_state(state)
```

### Activation Checkpointing with Model Parallelism

When combining activation checkpointing with model parallelism, the checkpoint function must:

1. Save the model-parallel RNG state alongside the inputs
2. Restore the RNG state before recomputation
3. Handle the data-parallel RNG state separately

```python
def checkpoint_with_mp_rng(function, *args):
    """Checkpoint with model-parallel RNG state management."""
    # Save inputs
    saved_inputs = detach_variable(args)

    # Save model-parallel RNG state
    mp_rng_state = get_model_parallel_rng_state()

    # Forward pass
    with torch.no_grad():
        outputs = function(*args)

    # Define recomputation hook
    def backward_hook(grad_outputs):
        # Restore model-parallel RNG state
        set_model_parallel_rng_state(mp_rng_state)
        # Recompute
        with torch.enable_grad():
            outputs = function(*saved_inputs)
        torch.autograd.backward(outputs, grad_outputs)

    return outputs
```

---

## Memory Savings Calculation

### Basic Memory Savings

For a transformer model with `N` layers and activation checkpointing on every layer:

```
Without checkpointing:
  Memory per layer = batch_size * seq_len * hidden_dim * bytes_per_element
  Total activation memory = N * memory_per_layer

With checkpointing (every layer):
  Memory per checkpoint = batch_size * seq_len * hidden_dim * bytes_per_element (inputs only)
  Total activation memory = N * memory_per_checkpoint (much smaller than full activations)

  But recomputation cost = N additional forward passes (2x forward cost total)
```

### Detailed Calculation for a Transformer Layer

For a single transformer layer with:
- `B` = batch size
- `S` = sequence length
- `H` = hidden dimension
- `A` = number of attention heads
- `dtype` = FP16 (2 bytes)

```
Activation memory per layer (without checkpointing):
  - QKV projections: 3 * B * S * H * 2 = 6BSH bytes
  - Attention scores: B * A * S * S * 2 = 2BAS^2 bytes
  - Attention output: B * S * H * 2 = 2BSH bytes
  - MLP intermediate: B * S * 4H * 2 = 8BSH bytes
  - MLP output: B * S * H * 2 = 2BSH bytes
  - Layer norms, residuals, etc.: ~4BSH bytes

  Total per layer ≈ (22BSH + 2BAS^2) bytes

Activation memory with checkpointing (only inputs saved):
  Per checkpoint = B * S * H * 2 = 2BSH bytes

  Savings ratio = (22BSH + 2BAS^2) / (2BSH)
                = 11 + AS/H
                ≈ 11 + S/A  (since H = A * head_dim, head_dim typically 64-128)

For typical values (S=2048, A=32, H=4096):
  Savings ratio ≈ 11 + 2048/32 = 11 + 64 = 75
  Memory saved ≈ 75x for activation storage
  Recomputation cost ≈ 1x additional forward pass (33% overhead total)
```

### Memory Savings with Different Strategies

| Strategy | Memory Saved | Recomputation Cost |
|---|---|---|
| No checkpointing | 0% | 0% |
| Checkpoint every 4 layers | ~65% | ~25% |
| Checkpoint every 2 layers | ~82% | ~50% |
| Checkpoint every layer | ~98% | ~100% |
| Every layer + CPU offload | ~99% | ~100% + transfer |
| Every layer + partition | ~99% per GPU | ~100% + communication |

---

## Usage in Training Loop

### Automatic Integration with DeepSpeed Engine

When activation checkpointing is configured in the DeepSpeed config, it is automatically applied to the model during `deepspeed.initialize()`:

```python
import deepspeed

ds_config = {
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": 16
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# Training loop - checkpointing is handled automatically
for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.loss
    model_engine.backward(loss)
    model_engine.step()
    # Activations are automatically checkpointed and recomputed
```

### Manual Checkpoint Application

For finer control, you can manually apply checkpointing to specific layers:

```python
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.layernorm1 = LayerNorm(config)
        self.layernorm2 = LayerNorm(config)

    def forward(self, hidden_states, attention_mask=None):
        # Option 1: Checkpoint the entire layer
        return checkpoint(self._forward, hidden_states, attention_mask)

    def _forward(self, hidden_states, attention_mask=None):
        # Self-attention
        normed = self.layernorm1(hidden_states)
        attn_out = self.attention(normed, attention_mask)
        hidden_states = hidden_states + attn_out

        # MLP
        normed = self.layernorm2(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = hidden_states + mlp_out

        return hidden_states
```

### Selective Checkpointing

You can choose to checkpoint only certain layers (e.g., only attention, not MLP):

```python
class SelectiveCheckpointLayer(nn.Module):
    def forward(self, hidden_states):
        # Checkpoint only the attention (most memory-intensive due to SxS matrix)
        attn_output = checkpoint(self.attention, hidden_states)
        hidden_states = hidden_states + attn_output

        # Don't checkpoint the MLP (less memory, less benefit)
        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states
```

---

## Integration with ZeRO

Activation checkpointing is fully compatible with all ZeRO stages and provides complementary memory savings.

### ZeRO Stage 1 + Activation Checkpointing

ZeRO-1 partitions optimizer states. Activation checkpointing reduces activation memory. Combined savings:

```
Total memory = model_params + optimizer_states/N + checkpointed_activations
```

### ZeRO Stage 2 + Activation Checkpointing

ZeRO-2 partitions optimizer states and gradients. Combined with activation checkpointing:

```
Total memory = model_params + (optimizer_states + gradients)/N + checkpointed_activations
```

### ZeRO Stage 3 + Activation Checkpointing

ZeRO-3 partitions all components (parameters, gradients, optimizer states). Activation checkpointing further reduces memory:

```
Total memory = (model_params + optimizer_states + gradients)/N + checkpointed_activations
```

With `partition_activations=True`:
```
Total memory = (model_params + optimizer_states + gradients)/N + checkpointed_activations/N
```

### Configuration for Maximum Memory Savings

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 32
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5
    }
}
```

---

## Configuration Examples

### Example 1: Basic Activation Checkpointing

```json
{
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": 16
    }
}
```

### Example 2: Activation Checkpointing with CPU Offload

```json
{
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 32
    }
}
```

### Example 3: Full Pipeline Parallelism Configuration

```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 8,
        "synchronize_checkpoint_boundary": true
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 4
    }
}
```

### Example 4: Maximum Memory Savings (ZeRO-3 + CPU Checkpointing)

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 32
    }
}
```

### Example 5: Activation Checkpointing with Profiling

```json
{
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": 16,
        "profile": true
    }
}
```

When `profile` is enabled, DeepSpeed logs detailed timing information:
- Time to save checkpoints
- Time to recompute activations
- Memory usage per checkpoint
- Total activation memory saved

---

## Best Practices

1. **Start with checkpointing every layer**: This provides maximum memory savings with a predictable ~33% throughput cost. Reduce the number of checkpoints only if the recomputation overhead is too high.

2. **Use CPU checkpointing only when necessary**: CPU checkpointing adds transfer overhead. Use it only when GPU memory is the binding constraint and CPU memory is plentiful.

3. **Enable contiguous memory optimization for long training runs**: Memory fragmentation accumulates over time. Contiguous checkpointing prevents this.

4. **Combine with ZeRO for maximum efficiency**: Activation checkpointing and ZeRO target different memory components (activations vs. parameters/gradients/optimizer states). Using both provides the most memory savings.

5. **Profile before and after**: Use the `profile` flag to measure the actual memory savings and recomputation overhead before committing to a configuration.

6. **Synchronize boundaries in pipeline parallelism**: Always enable `synchronize_checkpoint_boundary` when using pipeline parallelism to ensure correct execution ordering.

7. **Consider selective checkpointing for large models**: For very large models (70B+ parameters), checkpointing every layer may be too expensive in terms of recomputation. Consider checkpointing every other layer or only the attention modules.

8. **Ensure sufficient CPU RAM for CPU checkpointing**: Each checkpoint requires CPU memory. For a 70B model with 80 layers, this could be 50+ GB of CPU RAM.
