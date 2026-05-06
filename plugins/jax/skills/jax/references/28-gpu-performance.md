# Chapter 28: GPU Performance Optimization

## Overview

JAX compiles Python functions to GPU kernels via XLA. While XLA applies many optimizations automatically, understanding GPU performance characteristics and JAX's configuration options is essential for achieving peak throughput. This chapter covers matmul precision, XLA flags, memory management, NCCL configuration, compilation caching, and common performance pitfalls.

**Key Insight:** GPU performance is typically bounded by one of: compute throughput, memory bandwidth, kernel launch overhead, or communication overhead. Profiling (Chapter 27) helps identify which bottleneck applies.

## Matmul Precision Selection

### Default Precision Behavior

JAX uses `bfloat16` for matrix multiplications on NVIDIA GPUs by default (since jaxlib 0.4.24). This provides much higher throughput than `float32` while maintaining acceptable accuracy for most deep learning workloads.

```python
import jax
import jax.numpy as jnp

# Check current default matmul precision
print(f"Default matmul precision: {jax.default_matmul_precision}")

# Available options:
# 'bfloat16'   -- Uses BF16 for matmul (fastest, ~19 TFLOPS on A100)
# 'tensorfloat32' -- Uses TF32 for matmul (good balance, ~19 TFLOPS on A100)
# 'float32'    -- Uses full FP32 for matmul (slowest, ~9.7 TFLOPS on A100)
```

### Setting Global Matmul Precision

```python
import jax
import jax.numpy as jnp

# Set matmul precision globally
jax.config.update("jax_default_matmul_precision", "bfloat16")

# Or via environment variable
# export JAX_DEFAULT_MATMUL_PRECISION=bfloat16

# All subsequent matmuls use the specified precision
@jax.jit
def my_fn(x, w):
    return jnp.dot(x, w)  # Uses bfloat16 precision

# Or set to float32 for maximum accuracy
jax.config.update("jax_default_matmul_precision", "float32")
```

### Per-Operation Precision

```python
import jax
import jax.numpy as jnp
from jax import lax

x = jnp.ones((1024, 1024))
w = jnp.ones((1024, 1024))

# Use jnp.dot with specific precision
result_bf16 = jnp.dot(x, w, precision=jax.lax.Precision.DEFAULT)
result_tf32 = jnp.dot(x, w, precision=jax.lax.Precision.HIGH)
result_f32 = jnp.dot(x, w, precision=jax.lax.Precision.HIGHEST)

# lax.dot_general with explicit precision
result = lax.dot_general(
    x, w,
    dimension_numbers=(((1,), (0,)), ((), ())),
    precision=jax.lax.Precision.HIGHEST
)
```

### Precision Comparison Table

| Precision | NVIDIA Ampere TFLOPS | Accuracy | Use Case |
|-----------|---------------------|----------|----------|
| `bfloat16` | ~19 (A100) | ~3 decimal digits | Training (most cases) |
| `tensorfloat32` | ~19 (A100) | ~7 decimal digits | Training (precision-sensitive) |
| `float32` | ~9.7 (A100) | Full FP32 | Scientific computing, verification |

### Mixed Precision Training

```python
import jax
import jax.numpy as jnp
import optax

# Use bfloat16 for compute, float32 for parameters
@jax.jit
def mixed_precision_step(params, opt_state, x, y):
    def loss_fn(p):
        # Cast inputs to bfloat16 for forward pass
        x_bf16 = x.astype(jnp.bfloat16)
        w_bf16 = p["w"].astype(jnp.bfloat16)
        b_bf16 = p["b"].astype(jnp.bfloat16)

        # Compute in bfloat16
        hidden = jnp.dot(x_bf16, w_bf16) + b_bf16
        hidden = hidden.astype(jnp.float32)  # Upcast for stability
        hidden = jax.nn.gelu(hidden)

        pred = jnp.dot(hidden, p["w2"]) + p["b2"]
        loss = jnp.mean((pred - y) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optax.adam(1e-3).update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

## XLA Performance Flags

### Common XLA Flags

XLA compiler behavior can be tuned via the `XLA_FLAGS` environment variable:

```bash
# Set multiple XLA flags
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=true \
                  --xla_gpu_enable_pipelined_all_gather=true \
                  --xla_gpu_enable_pipelined_reduce_scatter=true \
                  --xla_gpu_enable_while_loop_double_buffering=true \
                  --xla_gpu_enable_reduce_of_reduce_contraction=true \
                  --xla_gpu_all_reduce_combine_threshold_bytes=8388608 \
                  --xla_gpu_all_gather_combine_threshold_bytes=8388608"
```

### Setting XLA Flags Programmatically

```python
import os
import jax

# Must be set BEFORE importing jax or creating any arrays
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=true "
    "--xla_gpu_autotune_level=4"
)

# After setting flags, import jax
import jax.numpy as jnp
```

### Key XLA Flags Reference

| Flag | Purpose | Default |
|------|---------|---------|
| `--xla_gpu_enable_triton_gemm` | Use Triton GEMM emitter | `true` (since JAX 0.4.28) |
| `--xla_gpu_autotune_level` | Autotuning aggressiveness (0-4) | `4` |
| `--xla_gpu_enable_pipelined_all_gather` | Pipeline all-gather with compute | `true` |
| `--xla_gpu_enable_pipelined_reduce_scatter` | Pipeline reduce-scatter | `true` |
| `--xla_gpu_enable_while_loop_double_buffering` | Double buffer in while loops | `true` |
| `--xla_gpu_all_reduce_combine_threshold_bytes` | Combine small all-reduces | `8MB` |
| `--xla_gpu_all_gather_combine_threshold_bytes` | Combine small all-gathers | `8MB` |
| `--xla_gpu_enable_reduce_of_reduce_contraction` | Fuse reductions | `true` |
| `--xla_gpu_enable_latency_hiding_scheduler` | Schedule for latency hiding | `false` |
| `--xla_gpu_force_compilation_parallelism` | Parallel compilation threads | `1` |
| `--xla_dump_to` | Dump HLO/IR to directory | (none) |
| `--xla_dump_hlo_as_text` | Dump HLO as text files | `false` |

## Auto PGLE (Profile Guided Latency Estimator)

Auto PGLE uses profiling data to improve compilation decisions. It automatically collects cost models from previous compilations to guide future ones.

### Enabling Auto PGLE

```python
import os

# Enable Auto PGLE
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["JAX_PGLE_EMBEDDING_DIR"] = "/tmp/jax_pgle"
os.environ["JAX_PGMEM_ESTIMATOR"] = "default"

import jax
import jax.numpy as jnp
```

### Auto PGLE via Config

```python
import jax

# Enable PGLE through JAX config
jax.config.update("jax_pgle_autotune_level", 4)

# PGLE benefits:
# 1. Faster recompilation after first run
# 2. Better instruction scheduling
# 3. Improved fusion decisions based on actual execution data
```

### How Auto PGLE Works

```
First compilation:
  1. JAX traces function -> Jaxpr
  2. Lowers to HLO
  3. Compiles with default cost model
  4. Executes and collects profiling data
  5. Stores cost model in PGLE directory

Subsequent compilations:
  1. JAX traces function -> Jaxpr
  2. Lowers to HLO
  3. Looks up PGLE cost model
  4. Compiles with improved cost estimates
  5. Better instruction scheduling and fusion
```

## Manual PGLE Workflow

For fine-grained control over PGLE:

### Step 1: Collect a Profile

```python
import jax
import jax.numpy as jnp

# Compile and run with profiling enabled
log_dir = "/tmp/jax_pgle_profile"

with jax.profiler.trace(log_dir):
    @jax.jit
    def model_fn(params, x):
        h = jnp.dot(x, params["w1"]) + params["b1"]
        h = jax.nn.gelu(h)
        h = jnp.dot(h, params["w2"]) + params["b2"]
        return h

    params = {
        "w1": jax.random.normal(jax.random.PRNGKey(0), (784, 512)),
        "b1": jnp.zeros(512),
        "w2": jax.random.normal(jax.random.PRNGKey(1), (512, 10)),
        "b2": jnp.zeros(10),
    }
    x = jax.random.normal(jax.random.PRNGKey(2), (128, 784))

    for _ in range(10):
        result = model_fn(params, x)
        result.block_until_ready()
```

### Step 2: Use Profile for Re-compilation

```python
import os

# Point JAX to the profile for re-compilation
os.environ["XLA_FLAGS"] = (
    f"--xla_gpu_pgle_profile_file_or_directory_path={log_dir} "
    "--xla_gpu_pgle_enable=true"
)

import jax
import jax.numpy as jnp

# Now recompile -- should produce better code
```

## Pipeline Parallelism Flags

When using pipeline parallelism (e.g., with multiple GPU stages), specific flags help overlap communication with computation:

```bash
# Pipeline parallelism optimization flags
export XLA_FLAGS="\
  --xla_gpu_enable_pipelined_all_gather=true \
  --xla_gpu_enable_pipelined_reduce_scatter=true \
  --xla_gpu_enable_pipelined_all_reduce=true \
  --xla_gpu_enable_pipelined_collective_permute=true"
```

### Overlapping Communication and Computation

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Enable pipeline parallelism flags before importing JAX
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_pipelined_all_gather=true"

import jax
import jax.numpy as jnp

devices = jax.devices()
if len(devices) >= 4:
    mesh = Mesh(devices[:4].reshape(2, 2), ("data", "model"))
else:
    mesh = Mesh(devices, ("data",))

# With pipelining enabled, all-gather operations can overlap with
# subsequent computation, hiding communication latency
```

## Communication Optimization Flags

### Combining Small Collectives

When many small all-reduce or all-gather operations are issued, XLA can combine them into fewer, larger operations:

```bash
# Combine threshold: operations smaller than this will be combined
export XLA_FLAGS="\
  --xla_gpu_all_reduce_combine_threshold_bytes=8388608 \
  --xla_gpu_all_gather_combine_threshold_bytes=8388608 \
  --xla_gpu_reduce_scatter_combine_threshold_bytes=8388608"

# 8388608 bytes = 8 MB
# Increase to combine more aggressively
# Decrease to reduce latency for small operations
```

### Reduce of Reduce Contraction

```bash
# Enable fusion of consecutive reductions
export XLA_FLAGS="--xla_gpu_enable_reduce_of_reduce_contraction=true"

# This allows XLA to fuse patterns like:
# x = reduce_sum(x, axis=0)
# x = reduce_sum(x, axis=1)
# Into a single reduction operation
```

## NCCL Configuration

NCCL handles GPU-to-GPU communication. Proper configuration is critical for multi-GPU and multi-node training.

### Essential NCCL Settings

```python
import os

# NCCL algorithm selection
os.environ["NCCL_ALGO"] = "Ring"
# Options:
#   Ring         - Good for most cases, predictable
#   Tree         - Better for small messages
#   CollnetDirect - Offload to switch (if supported)

# NCCL protocol
os.environ["NCCL_PROTO"] = "Simple"
# Options:
#   Simple  - Default, good throughput
#   LL      - Low Latency, better for small messages
#   LL128   - Balance between throughput and latency

# Number of channels (more = higher bandwidth, more memory)
os.environ["NCCL_NCHANNELS"] = "4"

# Disable shared memory for multi-node
os.environ["NCCL_SHM_DISABLE"] = "0"  # Enable for single-node

# Network interface for multi-node
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Adjust to your network

# Buffer sizes
os.environ["NCCL_BUFFSIZE"] = "4194304"  # 4MB per channel
```

### NCCL Tuning for Specific Scenarios

```python
import os

# For single-node multi-GPU (NVLink connected)
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_PROTO"] = "Simple"
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable P2P (NVLink)

# For multi-node (Ethernet connected)
os.environ["NCCL_ALGO"] = "Tree"
os.environ["NCCL_PROTO"] = "LL"
os.environ["NCCL_SHM_DISABLE"] = "1"  # Disable SHM across nodes
os.environ["NCCL_SOCKET_NTHREADS"] = "4"

# For InfiniBand connected
os.environ["NCCL_NET"] = "IB"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1"  # Specify IB devices
```

### NCCL Debugging

```python
import os

# Enable NCCL debug logging
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

# Enable NCCL tracing (generates trace file)
os.environ["NCCL_TRACE"] = "enable"
os.environ["NCCL_TRACE_FILE"] = "/tmp/nccl_trace_%h_%p"

# Timeout for collective operations (ms)
os.environ["NCCL_COMM_BLOCKING"] = "1"
os.environ["NCCL_MIN_NCHANNELS"] = "1"
os.environ["NCCL_MAX_NCHANNELS"] = "4"

import jax
```

## GPU Memory Allocation

JAX offers three memory allocation strategies for GPUs: preallocation, on-demand allocation, and Virtual Memory Management (VMM).

### Memory Preallocation (Default)

By default, JAX preallocates 75% of GPU memory at startup:

```python
import jax

# Check preallocation settings
print(f"Prealloc: {jax.config.jax_gpu_memory_fraction}")

# Default: preallocates 75% of GPU memory
# Controlled by JAX_PLATFORMS and memory_fraction
```

### Configuring Preallocation

```python
import os

# Set preallocation fraction (0.0 to 1.0)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

# Or disable preallocation entirely (use on-demand)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.0"

# Preallocate a specific amount
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
```

### On-Demand Memory Allocation

On-demand allocation only allocates memory when needed and releases it when no longer required:

```python
import os

# Disable preallocation to use on-demand allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp

# Memory is allocated as needed
x = jnp.ones((10000, 10000))  # Allocates ~400MB
y = jnp.dot(x, x)              # Allocates more
# When x, y go out of scope, memory can be reused
```

### Virtual Memory Management (VMM)

VMM is the modern approach to GPU memory management, providing better memory utilization:

```python
import os

# Enable VMM (available on CUDA 11.2+)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "vmm"

import jax
```

### Allocator Comparison

| Allocator | Pros | Cons | Setting |
|-----------|------|------|---------|
| **Default (BFC)** | Predictable, no fragmentation | Preallocates memory | `XLA_PYTHON_CLIENT_ALLOCATOR=default` |
| **On-demand** | Shares GPU with other processes | Higher allocation overhead | `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| **VMM** | Best utilization, reduced fragmentation | Requires CUDA 11.2+ | `XLA_PYTHON_CLIENT_ALLOCATOR=vmm` |

### Memory Configuration for Multi-Process

```python
import os

# When running multiple processes on the same GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# When running one process per GPU with visible_devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
```

## Multi-Process GPU Usage

### Single-Process Multi-GPU

```python
import jax
import jax.numpy as jnp

# JAX automatically sees all available GPUs
print(f"Number of GPUs: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

# Use all GPUs with sharding
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

devices = jax.devices()
mesh = Mesh(devices, ("data",))
sharding = NamedSharding(mesh, P("data"))

x = jax.random.normal(jax.random.PRNGKey(0), (len(devices) * 1024, 784))
x_sharded = jax.device_put(x, sharding)
```

### Multi-Process Multi-GPU (Distributed)

```python
import os
import jax
import jax.numpy as jnp

# Configure distributed initialization
# Must be set before any JAX operations
os.environ["JAX_COORDINATOR_ADDRESS"] = "localhost:6000"
os.environ["JAX_COORDINATOR_PORT"] = "6000"
os.environ["JAX_NUM_PROCESSES"] = "4"
os.environ["JAX_PROCESS_ID"] = "0"  # Each process sets its own ID

# Or use jax.distributed.initialize()
jax.distributed.initialize(
    coordinator_address="localhost:6000",
    num_processes=4,
    process_id=0,  # Set per process
)

print(f"Global devices: {jax.devices()}")
print(f"Local devices: {jax.local_devices()}")
print(f"Process index: {jax.process_index()}")
print(f"Total processes: {jax.process_count()}")
```

### Distributed Training Setup Script

```python
# launch_distributed.py -- run on each node
import os
import sys

# Each node runs: python launch_distributed.py <process_id>
process_id = int(sys.argv[1])
num_processes = int(sys.argv[2])

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["JAX_PROCESS_ID"] = str(process_id)
os.environ["JAX_NUM_PROCESSES"] = str(num_processes)

import jax

jax.distributed.initialize(
    coordinator_address="node0:6000",
    num_processes=num_processes,
    process_id=process_id,
)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Create mesh across all devices
devices = jax.devices()
mesh = Mesh(
    devices.reshape((num_processes, len(jax.local_devices()))),
    ("nodes", "devices")
)

print(f"Process {process_id}: initialized with {len(jax.local_devices())} local GPUs")
```

## Triton GEMM Emitter

JAX can use Triton as an alternative GEMM (General Matrix Multiply) emitter, often providing better performance for certain matrix shapes and layouts.

### Enabling Triton GEMM

```python
import os

# Enable Triton GEMM emitter
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"

import jax
import jax.numpy as jnp
```

### Triton GEMM Benefits

```
Benefits of Triton GEMM emitter:
1. Better performance for non-standard matmul shapes
2. Improved autotuning for specific hardware
3. Support for custom epilogues (fused operations after matmul)
4. Better handling of batched matmuls
5. Active development with frequent improvements
```

### Autotune Level

The autotune level controls how much effort Triton spends searching for optimal kernels:

```python
import os

# Autotune levels:
# 0: No autotuning (fastest compilation, may be suboptimal)
# 1: Basic autotuning
# 2: Moderate autotuning
# 3: Extensive autotuning
# 4: Maximum autotuning (slowest compilation, best performance)

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"

import jax
```

### Checking if Triton GEMM is Active

```python
import jax
import jax.numpy as jnp
import os

# Enable Triton GEMM
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true")

@jax.jit
def matmul_fn(a, b):
    return jnp.dot(a, b)

a = jnp.ones((2048, 2048))
b = jnp.ones((2048, 2048))

# Lower and check the HLO for triton-specific operations
lowered = matmul_fn.lower(a, b)
hlo_text = lowered.as_text()

# If Triton GEMM is active, you may see custom_call operations
# with "triton" in their names
has_triton = "triton" in hlo_text.lower()
print(f"Triton GEMM active: {has_triton}")
```

## Latency Hiding Scheduler

The latency hiding scheduler reorders operations to overlap communication with computation, improving throughput in distributed training.

### Enabling the Latency Hiding Scheduler

```python
import os

# Enable latency hiding scheduler
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_latency_hiding_scheduler=true"

import jax
```

### How Latency Hiding Works

```
Without latency hiding:
  [Compute] -> [AllReduce] -> [Compute] -> [AllReduce] -> ...
               ^^^^^^^^^^    (GPU idle during communication)

With latency hiding:
  [Compute] -> [AllReduce + Compute overlap] -> [Compute] -> ...
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               Communication and computation overlap
```

### Example: Distributed Training with Latency Hiding

```python
import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_pipelined_all_gather=true "
    "--xla_gpu_enable_pipelined_reduce_scatter=true"
)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# This setup allows all-gather and reduce-scatter operations
# to overlap with forward/backward pass computation
```

## GPU Memory Profiling

### Monitoring Memory Usage

```python
import jax
import jax.numpy as jnp

def print_gpu_memory():
    """Print current GPU memory usage for all devices."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            used_gb = stats.get("bytes_in_use", 0) / 1e9
            limit_gb = stats.get("bytes_limit", 0) / 1e9
            peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
            pct = (used_gb / limit_gb * 100) if limit_gb > 0 else 0
            print(f"{device}: {used_gb:.2f} / {limit_gb:.2f} GB "
                  f"({pct:.1f}%) [peak: {peak_gb:.2f} GB]")

print("Before allocation:")
print_gpu_memory()

# Allocate large arrays
x = jnp.ones((16384, 16384))  # ~1 GB
y = jnp.dot(x, x)              # ~1 GB

print("\nAfter allocation:")
print_gpu_memory()
```

### Memory Profiling with jax.profiler

```python
import jax
import jax.numpy as jnp

# Save memory profile
jax.config.update("jax_profiler_memory_profile_path", "/tmp/gpu_memory.pb")

@jax.jit
def memory_intensive(x):
    a = jnp.dot(x, x)     # Large intermediate
    b = jnp.exp(a)         # Another large intermediate
    c = jnp.dot(b, x)     # Yet another
    return jnp.sum(c)

x = jnp.ones((8192, 8192))
result = memory_intensive(x)
result.block_until_ready()

# Analyze: go tool pprof -http=:8080 /tmp/gpu_memory.pb
```

### Estimating Memory Requirements

```python
import jax
import jax.numpy as jnp

def estimate_memory(shape, dtype=jnp.float32):
    """Estimate memory for a single array."""
    element_size = jnp.dtype(dtype).itemsize
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return num_elements * element_size

def estimate_training_memory(
    batch_size, seq_len, hidden_dim, num_layers, vocab_size
):
    """Rough estimate of memory for a transformer training step."""
    bytes_per_element = 4  # float32

    # Parameter memory
    params_per_layer = 4 * hidden_dim * hidden_dim  # QKV + output projections
    total_params = num_layers * params_per_layer + 2 * vocab_size * hidden_dim
    param_memory = total_params * bytes_per_element

    # Activation memory (forward pass)
    # Each layer produces activations of shape (batch, seq, hidden)
    activation_per_layer = batch_size * seq_len * hidden_dim * bytes_per_element
    activation_memory = num_layers * activation_per_layer

    # Gradient memory (same size as parameters)
    gradient_memory = param_memory

    # Optimizer state (2x for Adam: momentum + variance)
    optimizer_memory = 2 * param_memory

    total = param_memory + activation_memory + gradient_memory + optimizer_memory

    print(f"Parameters: {param_memory / 1e9:.2f} GB")
    print(f"Activations: {activation_memory / 1e9:.2f} GB")
    print(f"Gradients: {gradient_memory / 1e9:.2f} GB")
    print(f"Optimizer: {optimizer_memory / 1e9:.2f} GB")
    print(f"Total estimate: {total / 1e9:.2f} GB")

    return total

# Example: GPT-2 sized model
estimate_training_memory(
    batch_size=32,
    seq_len=1024,
    hidden_dim=1280,
    num_layers=36,
    vocab_size=50257,
)
```

## Common Performance Pitfalls and Fixes

### Pitfall 1: Python Loops Over Array Elements

```python
import jax
import jax.numpy as jnp

# BAD: Python loop -- extremely slow
def slow_sum(x):
    total = jnp.array(0.0)
    for i in range(x.shape[0]):
        total = total + x[i]  # Each iteration is a separate JAX operation
    return total

# GOOD: Use JAX reduction
def fast_sum(x):
    return jnp.sum(x)

x = jnp.ones(10000)

# The slow version generates 10000 separate add operations in the HLO
# The fast version generates a single efficient reduction
```

### Pitfall 2: Frequent Recompilation

```python
import jax
import jax.numpy as jnp

# BAD: Different shapes trigger recompilation
@jax.jit
def process_batch(x):
    return jnp.mean(x, axis=0)

process_batch(jnp.ones(10))    # Compiles
process_batch(jnp.ones(20))    # Recompiles!
process_batch(jnp.ones(30))    # Recompiles!

# GOOD: Pad to fixed batch size
MAX_BATCH = 64

@jax.jit
def process_padded(x, valid_length):
    # Pad to fixed size
    padded = jnp.zeros(MAX_BATCH)
    padded = padded.at[:x.shape[0]].set(x)
    # Compute with mask
    mask = jnp.arange(MAX_BATCH) < valid_length
    return jnp.sum(padded * mask) / valid_length

# Only compiles once
process_padded(jnp.ones(10), 10)
process_padded(jnp.ones(20), 20)
process_padded(jnp.ones(30), 30)
```

### Pitfall 3: Not Using block_until_ready for Timing

```python
import time
import jax.numpy as jnp

# BAD: Async dispatch makes timing inaccurate
start = time.time()
result = jnp.dot(jnp.ones((4096, 4096)), jnp.ones((4096, 4096)))
print(f"Time: {time.time() - start:.6f}s")  # Nearly 0!

# GOOD: Block to get accurate timing
start = time.time()
result = jnp.dot(jnp.ones((4096, 4096)), jnp.ones((4096, 4096)))
result.block_until_ready()
print(f"Time: {time.time() - start:.6f}s")  # Accurate
```

### Pitfall 4: Unnecessary Host-Device Transfers

```python
import jax
import jax.numpy as jnp
import numpy as np

# BAD: Repeated transfers between host and device
@jax.jit
def bad_fn(x):
    result = x
    for _ in range(10):
        # Each device_get transfers data to host
        # Each jax.device_put transfers back to device
        host_val = jax.device_get(result)
        result = jax.device_put(host_val + 1)
    return result

# GOOD: Stay on device
@jax.jit
def good_fn(x):
    result = x
    for _ in range(10):
        result = result + 1  # Entirely on device
    return result
```

### Pitfall 5: Not Fusing Operations

```python
import jax
import jax.numpy as jnp

# BAD: Materializes intermediate arrays
@jax.jit
def unfused(x):
    a = x + 1.0
    b = a * 2.0
    c = jnp.exp(b)
    d = c / jnp.sum(c)
    return d
# XLA may not fuse all of these depending on complexity

# GOOD: Write as a single expression when possible
@jax.jit
def fused(x):
    return jax.nn.softmax(jnp.exp((x + 1.0) * 2.0))
# Or use jax.checkpoint to control materialization
```

### Pitfall 6: Using Python Control Flow in Traced Code

```python
import jax
import jax.numpy as jnp

# BAD: Python if -- ConcretizationTypeError inside jit
@jax.jit
def bad_cond(x, flag):
    if flag:  # flag is traced, not concrete!
        return x * 2
    return x * 3

# GOOD: Use jax.lax.cond
@jax.jit
def good_cond(x, flag):
    return jax.lax.cond(flag, lambda x: x * 2, lambda x: x * 3, x)
```

### Pitfall 7: Excessive GradCheckpointing Overhead

```python
import jax
import jax.numpy as jnp

# BAD: Checkpointing operations that are cheap to recompute
@jax.jit
def over_checkpointed(x):
    # These are all cheap -- no need to checkpoint
    x = jax.checkpoint(lambda v: v + 1)(x)
    x = jax.checkpoint(lambda v: v * 2)(x)
    x = jax.checkpoint(lambda v: v ** 2)(x)
    return x

# GOOD: Only checkpoint expensive operations (like matmul)
@jax.jit
def smart_checkpoint(x, w):
    def expensive_block(x):
        x = jnp.dot(x, w)      # Expensive: worth checkpointing
        x = jax.nn.gelu(x)     # Cheap: gets checkpointed as part of block
        x = jnp.dot(x, w.T)    # Expensive: worth checkpointing
        return x
    return jax.checkpoint(expensive_block)(x)
```

### Pitfall 8: Ignoring Data Layout

```python
import jax
import jax.numpy as jnp

# Matrix multiplication is most efficient with (M, K) x (K, N) layout
# Transposing at matmul time can be expensive

# BAD: Transpose forces a copy or slower kernel
@jax.jit
def bad_layout(x, w):
    return jnp.dot(x, w.T)  # w.T may force layout change

# GOOD: Store weights in the layout needed for matmul
# If you always compute x @ w, store w as (input_dim, output_dim)
@jax.jit
def good_layout(x, w):
    return jnp.dot(x, w)  # No transpose needed
```

## Performance Checklist

```
1. [ ] Set appropriate matmul precision (bfloat16/TF32 for training)
2. [ ] Enable Triton GEMM emitter (XLA_FLAGS)
3. [ ] Use jax.jit for all performance-critical functions
4. [ ] Warm up JIT-compiled functions before timing
5. [ ] Use .block_until_ready() for accurate timing
6. [ ] Avoid Python control flow inside JIT (use jax.lax.cond, scan, etc.)
7. [ ] Use consistent array shapes to avoid recompilation
8. [ ] Use jax.checkpoint for memory-intensive models
9. [ ] Profile with jax.profiler.trace to identify bottlenecks
10. [ ] Check GPU memory usage (device.memory_stats())
11. [ ] Enable latency hiding scheduler for distributed training
12. [ ] Configure NCCL for your network topology
13. [ ] Use VMM allocator for better memory utilization
14. [ ] Minimize host-device data transfers
15. [ ] Profile compilation time and use PGLE for re-compilation
```
