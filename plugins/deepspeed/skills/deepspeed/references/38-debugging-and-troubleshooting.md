# Debugging and Troubleshooting

## Overview

This reference covers all DeepSpeed debugging tools, common errors and solutions, environment variable debugging, memory profiling techniques, and per-ZeRO-stage troubleshooting guides. Use this as your first resource when encountering issues during DeepSpeed training.

---

## `ds_report` - System Environment Report

`ds_report` is the primary diagnostic tool for DeepSpeed. It collects and displays your complete hardware/software environment, library versions, and DeepSpeed configuration.

### Usage

```bash
# Basic report
ds_report

# From Python
python -c "import deepspeed; deepspeed.report()"
```

### What It Reports

| Category | Information |
|---|---|
| **DeepSpeed** | Version, install location, commit hash |
| **PyTorch** | Version, CUDA version, cuDNN version, build configuration |
| **CUDA** | CUDA home, nvcc version, CUDA driver version |
| **NCCL** | NCCL version and configuration |
| **Python** | Version, executable path |
| **OS** | Kernel version, distribution |
| **Hardware** | Number of GPUs, GPU model, total GPU memory, CPU count |
| **Libraries** | Transformers, Accelerate, aiohttp, mpi4py versions |
| **Torch Extensions** | Status of compiled CUDA kernels (op_builder) |

### Sample Output

```
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compilation fails may indicate the
      wrong version of CUDA or PyTorch is installed.
--------------------------------------------------
 [JIT] ..... async_io ..... [OK] ..... (no binary found)
 [JIT] ..... fused_adam ..... [OK] ..... (no binary found)
 [JIT] ..... fused_lamb ..... [OK] ..... (no binary found)
 [JIT] ..... cpu_adam ..... [OK] ..... (no binary found)
 [JIT] ..... cpu_adagrad ..... [OK] ..... (no binary found)
 [JIT] ..... quantizer ..... [OK] ..... (no binary found)
 [JIT] ..... random_ltd ..... [OK] ..... (no binary found)
 [JIT] ..... sparse_attention ..... [OK] ..... (no binary found)
 [JIT] ..... spatial_inference ..... [OK] ..... (no binary found)
 [JIT] ..... transformer ..... [OK] ..... (no binary found)
 [JIT] ..... stochastic_transformer ..... [OK] ..... (no binary found)
 [JIT] ..... transformer_inference ..... [OK] ..... (no binary found)

--------------------------------------------------
DeepSpeed general environment info:
--------------------------------------------------
torch install path ............... ['/opt/conda/lib/python3.10/site-packages/torch']
torch version .................... ['2.1.0']
cuda version ..................... ['11.8']
torch cuda version ............... ['11.8']
torch backend .................... ['cu118']
nvcc version ..................... ['11.8']
cuda install path ................ ['/usr/local/cuda']
deepspeed install path ........... ['/opt/conda/lib/python3.10/site-packages/deepspeed']
deepspeed info ................... ['0.12.6', 'unknown', 'unknown']
torch extension name ............. ['op_builder']
```

### Interpreting Results

- **`[OK]` next to JIT**: The op was successfully JIT-compiled during the report. This means it will work at runtime.
- **`[FAILED]`**: The op could not be compiled. Check your CUDA/PyTorch version compatibility.
- **`[BUILT]`**: The op was pre-installed (not JIT). Best for production.
- **`(no binary found)**: The op is not pre-installed and will be JIT-compiled on first use.

---

## Common Errors and Solutions

### Error: `RuntimeError: CUDA out of memory`

#### Root Causes

1. Batch size too large for available GPU memory
2. Model parameters + optimizer states exceed memory
3. Activation checkpointing not enabled
4. ZeRO offloading not configured

#### Solutions by ZeRO Stage

**No ZeRO**:
```json
{
    "zero_optimization": { "stage": 0 },
    "train_micro_batch_size_per_gpu": 4
}
```

**ZeRO Stage 1** (shard optimizer states):
```json
{
    "zero_optimization": { "stage": 1 },
    "train_micro_batch_size_per_gpu": 8
}
```

**ZeRO Stage 2** (shard optimizer states + gradients):
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

**ZeRO Stage 3** (shard all parameters + optimizer states + gradients):
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

### Error: `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one`

#### Root Cause

The backward pass did not complete fully before `engine.step()` was called, or there is a mismatch in the number of forward/backward calls per step.

#### Solutions

1. Ensure every `forward()` call is followed by exactly one `backward()` before calling `step()`
2. Check gradient accumulation: the number of `forward+backward` calls must match `gradient_accumulation_steps`
3. If using pipeline parallelism, verify `num_microbatches` matches across all stages

```python
# Correct pattern with gradient accumulation
for step, batch in enumerate(dataloader):
    outputs = engine(batch)
    loss = engine.backward(outputs.loss)

    if (step + 1) % gradient_accumulation_steps == 0:
        engine.step()
```

### Error: `ValueError: Unrecognized optimizer: <name>`

#### Root Cause

The optimizer name is not in the `DEEPSPEED_OPTIMIZERS` registry.

#### Solutions

1. Check spelling (optimizer names are case-insensitive but must match exactly)
2. Valid names: `adam`, `adamw`, `lamb`, `onebitadam`, `zerooneadam`, `onebitlamb`, `muadam`, `muadamw`, `musgd`, `lion`, `muon`, `adagrad`

```json
{
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-4 }
    }
}
```

### Error: `AssertionError: Micro batch size per GPU must be divisible by...

#### Root Cause

The micro batch size is not evenly divisible by the model parallel size or pipeline parallel degree.

#### Solutions

Ensure `train_micro_batch_size_per_gpu` is divisible by all parallelism dimensions:

```json
{
    "train_micro_batch_size_per_gpu": 4,
    "tensor_parallel": { "enabled": true, "tp_size": 2 }
    // micro_batch must be divisible by tp_size (4 / 2 = 2, OK)
}
```

### Error: `deepspeed.ops.op_builder.CUDAOpBuilderFailed`

#### Root Cause

DeepSpeed CUDA extension compilation failed, typically due to:
- Incompatible CUDA version (requires CUDA 11.6+)
- Missing `nvcc` in PATH
- Incompatible PyTorch CUDA version

#### Solutions

```bash
# Check CUDA availability
nvcc --version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Install matching versions
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Force JIT recompilation
rm -rf ~/.cache/torch_extensions/
ds_report
```

### Error: `OSError: [Errno 12] Cannot allocate memory` (CPU OOM)

#### Root Cause

CPU memory exhausted, typically with ZeRO offloading or large model checkpointing.

#### Solutions

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true,
            "max_in_cpu": 1e9
        }
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true
    }
}
```

### Error: `NaN loss detected`

#### Root Cause

Loss became NaN, typically due to:
- Loss scale too high in FP16 training
- Learning rate too high
- Data containing NaN/Inf values

#### Solutions

```json
{
    "fp16": {
        "enabled": true,
        "initial_scale_power": 12,
        "loss_scale_window": 500,
        "min_loss_scale": 1
    }
}
```

Or switch to BF16 which has a wider dynamic range:
```json
{
    "bf16": {
        "enabled": true
    }
}
```

---

## Debugging Tools

### Environment Variable Debugging

DeepSpeed supports several environment variables for debugging:

| Variable | Values | Description |
|---|---|---|
| `DS_DEBUG` | `0`, `1` | Enable DeepSpeed debug logging |
| `NCCL_DEBUG` | `INFO`, `WARN`, `TRACE` | NCCL collective communication debugging |
| `NCCL_DEBUG_SUBSYS` | `ALL`, `COLL`, `NET` | NCCL subsystem to debug |
| `CUDA_LAUNCH_BLOCKING` | `0`, `1` | Synchronize CUDA operations for precise error locations |
| `TORCH_DISTRIBUTED_DEBUG` | `OFF`, `INFO`, `DETAIL` | PyTorch distributed debugging |
| `TORCH_SHOW_CPP_STACKTRACES` | `0`, `1` | Show C++ stack traces on errors |
| `DEEPSPEED_TIMEOUT` | integer (minutes) | Override default 30-minute process group timeout |
| `DS_BUILD_OPS` | `0`, `1` | Force skip/build CUDA extensions |
| `DS_ACCELERATOR` | string | Override accelerator (e.g., `"cuda"`, `"xpu"`) |

#### Usage Examples

```bash
# Enable all debugging
DS_DEBUG=1 NCCL_DEBUG=INFO CUDA_LAUNCH_BLOCKING=1 deepspeed train.py

# Debug NCCL communication hangs
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL deepspeed train.py

# Debug distributed errors
TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=1 deepspeed train.py

# Increase timeout for slow networks
DEEPSPEED_TIMEOUT=120 deepspeed train.py
```

### Wall Clock Breakdown

Enable detailed timing of each training step phase:

```json
{
    "wall_clock_breakdown": true
}
```

This reports timing for:
- Forward pass
- Backward pass
- Gradient reduction
- Optimizer step
- Batch loading
- Total step time

Access from Python:
```python
timer = engine.tictoc
print(f"Forward: {timer.forward_time_ms()} ms")
print(f"Backward: {timer.backward_time_ms()} ms")
print(f"Step: {timer.step_time_ms()} ms")
```

### Memory Breakdown

Enable detailed GPU memory usage reporting:

```json
{
    "memory_breakdown": true
}
```

This tracks:
- Parameter memory
- Gradient memory
- Optimizer state memory
- Activation memory
- Temporary buffer memory

Access from Python:
```python
memory_stats = engine.memory_status()
print(memory_stats)
```

### Gradient Overflow Detection

```json
{
    "fp16": {
        "enabled": true
    },
    "gradient_clipping": 1.0
}
```

DeepSpeed automatically detects gradient overflow (NaN/Inf) and skips the optimizer step when detected. The loss scale is adjusted dynamically.

---

## Memory Profiling

### Understanding Memory Usage by ZeRO Stage

#### Stage 0 (No ZeRO)

Each GPU stores the full model:
```
Memory per GPU = Model Params + Optimizer States + Gradients + Activations
              ≈ 2Φ + 12Φ + 2Φ + A     (for Adam with FP32 master weights)
              ≈ 16Φ + A
```
Where Φ = number of parameters, A = activation memory.

#### Stage 1 (Optimizer State Partitioning)

Optimizer states are partitioned across N GPUs:
```
Memory per GPU = Model Params + Gradients + (Optimizer States / N) + Activations
              ≈ 4Φ + (12Φ / N) + A
```

#### Stage 2 (Optimizer + Gradient Partitioning)

Both optimizer states and gradients are partitioned:
```
Memory per GPU = Model Params + (Gradients / N) + (Optimizer States / N) + Activations
              ≈ 2Φ + (14Φ / N) + A
```

#### Stage 3 (Full Parameter Partitioning)

All states are partitioned:
```
Memory per GPU = (Model Params / N) + (Gradients / N) + (Optimizer States / N) + Activations
              ≈ (16Φ / N) + A
```

### Memory Profiling Tools

#### DeepSpeed Memory Estimator

```python
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# Stage 2 memory estimate
estimate_zero2_model_states_mem_needs_all_live(
    model_size=7e9,           # 7B parameter model
    num_gpus_per_node=8,
    num_nodes=1
)

# Stage 3 memory estimate
estimate_zero3_model_states_mem_needs_all_live(
    model_size=7e9,
    num_gpus_per_node=8,
    num_nodes=1,
    additional_buffer_factor=1.0
)
```

#### PyTorch Memory Profiler

```python
import torch

# Before training step
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

start_mem = torch.cuda.memory_allocated()
start_reserved = torch.cuda.memory_reserved()

# Training step
engine.step()

# After training step
end_mem = torch.cuda.memory_allocated()
end_reserved = torch.cuda.memory_reserved()
peak_mem = torch.cuda.max_memory_allocated()

print(f"Allocated: {(end_mem - start_mem) / 1e9:.2f} GB")
print(f"Peak: {peak_mem / 1e9:.2f} GB")
print(f"Reserved: {(end_reserved - start_reserved) / 1e9:.2f} GB")
```

### Activation Memory Estimation

Activation memory depends on batch size, sequence length, and hidden dimension:

```python
# Approximate activation memory per transformer layer (in GB)
def estimate_activation_memory(batch_size, seq_len, hidden_dim, num_layers, precision_bytes=2):
    bytes_per_element = precision_bytes
    # Each transformer layer stores ~34 * batch_size * seq_len * hidden_dim activations
    per_layer = 34 * batch_size * seq_len * hidden_dim * bytes_per_element
    total = per_layer * num_layers
    return total / 1e9  # GB

# Example: 7B model with bf16
act_mem = estimate_activation_memory(
    batch_size=4,
    seq_len=2048,
    hidden_dim=4096,
    num_layers=32,
    precision_bytes=2
)
print(f"Activation memory: {act_mem:.2f} GB")
```

---

## Per-ZeRO-Stage Troubleshooting

### ZeRO Stage 1 Issues

#### Issue: No memory savings visible

**Diagnosis**: Stage 1 only partitions optimizer states. With Adam, optimizer states use 12 bytes per parameter (FP32 master weights, momentum, variance) vs 2 bytes for the model itself. Savings are visible for large models.

**Solution**: Verify ZeRO is active by checking `engine.zero_optimization_stage()` returns `1`.

#### Issue: Gradient synchronization errors

**Diagnosis**: Stage 1 uses standard DDP for gradient synchronization.

**Solution**:
```bash
# Check NCCL
NCCL_DEBUG=INFO deepspeed train.py

# Verify NCCL socket path length (common on Kubernetes)
export NCCL_SOCKET_IFNAME=^lo,docker0
```

### ZeRO Stage 2 Issues

#### Issue: Slow training with CPU offload

**Diagnosis**: CPU-GPU data transfer is the bottleneck. PCIe bandwidth (~32 GB/s) is much lower than GPU memory bandwidth (~2 TB/s).

**Solutions**:
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": true
        }
    }
}
```

Key parameters:
- `pin_memory: true` -- Uses pinned memory for faster CPU-GPU transfers
- `buffer_count: 4` -- Number of buffers for overlapping compute and transfer
- `fast_init: true` -- Enables fast optimizer state initialization

#### Issue: `RuntimeError: NCCL error: unhandled system error`

**Diagnosis**: NCCL communication failure, often caused by GPU timeout or network issues.

**Solution**:
```bash
# Increase NCCL timeout
export NCCL_TIMEOUT=1800

# Disable NVLink if having issues
export NCCL_P2P_DISABLE=1

# Use TCP instead of shared memory
export NCCL_SHM_DISABLE=1
```

#### Issue: Gradient overflow loop (loss scale keeps decreasing)

**Diagnosis**: Model is producing consistently overflowing gradients.

**Solution**:
```json
{
    "fp16": {
        "enabled": true,
        "initial_scale_power": 8,
        "min_loss_scale": 0.0001
    },
    "gradient_clipping": 0.5
}
```

Or switch to BF16:
```json
{
    "bf16": { "enabled": true }
}
```

### ZeRO Stage 3 Issues

#### Issue: Forward pass is very slow

**Diagnosis**: Stage 3 must gather sharded parameters before each forward pass. If parameters are frequently gathered and released, this creates significant overhead.

**Solutions**:

1. Enable contiguous memory optimization:
```json
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5
    }
}
```

2. Use `stage3_param_persistence_threshold` to keep small parameters unsharded:
```json
{
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": "auto"
    }
}
```

3. Use `deepspeed.zero.GatheredParameters` for modules that need all parameters:
```python
with deepspeed.zero.GatheredParameters(model.parameters()):
    output = model(input_ids)
```

#### Issue: `RuntimeError: Trying to access a parameter that has been freed`

**Diagnosis**: In Stage 3, parameters are freed after use. Accessing them outside the forward/backward context causes this error.

**Solution**: Use `GatheredParameters` context manager or mark persistent parameters:
```python
import deepspeed

# For inference/validation
with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
    logits = model(input_ids)
    loss = criterion(logits, labels)
```

#### Issue: Checkpoint save/load failures

**Diagnosis**: Stage 3 saves sharded checkpoints. Each rank saves its own shard. Missing shards cause load failures.

**Solutions**:
```python
# Save with tag
tag = f"step_{global_step}"
engine.save_checkpoint(save_dir, tag=tag)

# Load - ensure all rank files are present
engine.load_checkpoint(save_dir, tag=tag)

# For universal checkpoint (all ranks in single file)
# Use the --save_universal checkpoint flag
```

#### Issue: Activation checkpointing + ZeRO-3 conflicts

**Diagnosis**: Activation checkpointing re-computes activations during backward pass, but ZeRO-3 may not have the parameters available.

**Solution**:
```json
{
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_contiguous_parameters": true
    }
}
```

---

## Pipeline Parallelism Troubleshooting

### Issue: Pipeline bubble overhead

**Diagnosis**: Pipeline parallelism has a "bubble" where some stages are idle waiting for data.

**Solution**: Increase micro-batches per pipeline batch:
```json
{
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "pipeline": {
        "enabled": true,
        "parallel_size": 4
    }
}
```

The pipeline bubble fraction is `(p-1) / m` where `p` = pipeline stages, `m` = micro-batches.

### Issue: `RuntimeError: loss values are not consistent across stages`

**Diagnosis**: Loss values differ between pipeline stages, typically due to inconsistent batch sizes or data parallelism mismatches.

**Solution**: Ensure all stages use the same micro-batch size and that data is consistently split.

---

## Tensor Parallelism Troubleshooting

### Issue: NCCL timeout during TP communication

**Diagnosis**: Tensor parallelism requires frequent all-reduce within each transformer layer. Slow inter-node links cause timeouts.

**Solution**: Keep tensor parallelism within a single node:
```json
{
    "tensor_parallel": {
        "enabled": true,
        "tp_size": 4
    }
}
```
Use `tp_size` <= GPUs per node (typically 4 or 8).

### Issue: Incorrect results with TP

**Diagnosis**: Column-parallel and row-parallel splits not matching, or bias handling incorrect.

**Solution**: Verify model is properly annotated for tensor parallelism. Check that linear layers use the correct parallelism mode.

---

## Inference Troubleshooting

### Issue: Kernel replacement not working

**Diagnosis**: The inference engine cannot find matching kernel replacements for the model architecture.

**Solution**: Check supported models and enable the correct mode:
```json
{
    "inference": {
        "enabled": true,
        "kernel_inject": true,
        "tensor_parallel": {
            "tp_size": 1
        },
        "dtype": "fp16"
    }
}
```

### Issue: `ValueError: <model_name> is not supported`

**Diagnosis**: The specialized inference mode does not support this model.

**Solution**: Use generic mode:
```json
{
    "inference": {
        "enabled": true,
        "kernel_inject": true,
        "replace_with_kernel_inject": true
    }
}
```

---

## Performance Debugging Checklist

1. **Run `ds_report`** first to verify your environment
2. **Enable `wall_clock_breakdown`** to identify the slowest step phase
3. **Check GPU utilization** with `nvidia-smi dmon -s pucvmet -d 1`
4. **Verify NCCL bandwidth** with NCCL tests: `./all_reduce_perf -b 8M -e 256M`
5. **Profile with PyTorch profiler**:
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    engine.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```
6. **Check for CPU bottlenecks** when using offloading: `htop` or `py-spy top --pid <pid>`
7. **Verify gradient accumulation** is working: `engine.gradient_accumulation_steps()`
8. **Enable torch.compile** for additional speedups: `engine.compile()`

---

## Diagnostic Commands Quick Reference

```bash
# Environment check
ds_report

# GPU status
nvidia-smi
nvidia-smi dmon -s pucvmet -d 1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# CUDA version check
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# NCCL debug
NCCL_DEBUG=INFO deepspeed train.py 2>&1 | grep NCCL

# Memory debug
CUDA_LAUNCH_BLOCKING=1 python -c "
import torch
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'PyTorch alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
"

# Process group timeout debug
DEEPSPEED_TIMEOUT=120 NCCL_DEBUG=TRACE deepspeed train.py

# JIT compile all ops
python -c "from deepspeed.ops.op_builder import ALL_OPS; [op().load() for op in ALL_OPS.values()]"
```
