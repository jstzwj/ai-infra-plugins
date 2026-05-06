# Chapter 39: JAX Configuration

## 39.1 Overview

JAX provides a rich set of configuration options that control compilation behavior,
numerical precision, device allocation, debugging features, and more. These can be
set via Python API, environment variables, or command-line flags.

---

## 39.2 Configuration via Python API

### 39.2.1 jax.config

The primary configuration interface is `jax.config`:

```python
import jax

# Read a configuration value
print(jax.config.jax_platforms)        # Current platform setting
print(jax.config.jax_enable_x64)       # Whether 64-bit is enabled

# Set a configuration value
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# Context manager for temporary changes
with jax.config.update("jax_enable_x64", True):
    import jax.numpy as jnp
    x = jnp.float64(1.0)  # Works inside context
    print(x.dtype)  # float64

# x = jnp.float64(1.0)  # Would fail outside context (if x64 disabled)
```

### 39.2.2 Common Configuration Options

```python
import jax

# Platform selection
jax.config.update("jax_platforms", "cpu")       # Force CPU
jax.config.update("jax_platforms", "gpu")       # Force GPU
jax.config.update("jax_platforms", "tpu")       # Force TPU
jax.config.update("jax_platforms", "cpu,gpu")   # Preference order

# 64-bit precision
jax.config.update("jax_enable_x64", True)       # Enable float64/int64

# Debugging
jax.config.update("jax_debug_nans", True)        # Detect NaN after each op
jax.config.update("jax_debug_infs", True)        # Detect Inf after each op
jax.config.update("jax_disable_jit", True)       # Disable JIT compilation
jax.config.update("jax_enable_checks", True)     # Enable shape/dtype checks

# Tracing
jax.config.update("jax_traceback_filtering", "off")     # Full tracebacks
jax.config.update("jax_traceback_filtering", "trace")   # Filtered tracebacks
jax.config.update("jax_traceback_filtering", "numba")   # Numba-style filtering

# Memory
jax.config.update("jax_pmap_no_rank_reduction", True)
jax.config.update("ax_gpu_mem_fraction", 0.8)   # GPU memory fraction

# Distributed
jax.config.update("jax_coordination_service", "cherney")  # Coordination service
jax.config.update("jax_distributed_debug", True)          # Debug distributed ops
```

### 39.2.3 Per-Function Configuration with contextmanager

```python
from jax import config as jax_config

def debug_mode(fn):
    """Decorator to run a function in debug mode."""
    def wrapper(*args, **kwargs):
        with jax_config.update("jax_debug_nans", True), \
             jax_config.update("jax_disable_jit", True):
            return fn(*args, **kwargs)
    return wrapper

@debug_mode
def problematic_fn(x):
    return jnp.sqrt(x)  # Will detect NaN immediately
```

---

## 39.3 Environment Variables

All JAX configuration options can be set via environment variables. The
environment variable name is the config option name in uppercase:

### 39.3.1 Core Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `JAX_PLATFORMS` | `""` (auto) | Comma-separated list of platforms to try |
| `JAX_ENABLE_X64` | `False` | Enable 64-bit floating point and integer types |
| `JAX_DISABLE_JIT` | `False` | Disable JIT compilation (eager execution) |
| `JAX_DEBUG_NANS` | `False` | Automatically detect NaN values after each operation |
| `JAX_DEBUG_INFS` | `False` | Automatically detect Inf values after each operation |
| `JAX_ENABLE_CHECKS` | `False` | Enable shape and dtype checks at trace time |
| `JAX_TRACEBACK_FILTERING` | `"trace"` | Control traceback filtering level |

```bash
# Example: Force CPU and enable 64-bit
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
python my_script.py
```

### 39.3.2 GPU Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | Fraction of GPU memory to preallocate |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `True` | Whether to preallocate GPU memory |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `"default"` | GPU allocator: `default`, `platform`, `bfc` |
| `XLA_PYTHON_CLIENT_GPU_MEM_LIMIT` | `""` | Absolute GPU memory limit (e.g., `"8GiB"`) |
| `XLA_PYTHON_CLIENT_GPU_ALLOCATOR_CONFIG` | `""` | Advanced GPU allocator configuration |

```bash
# Preallocate only 50% of GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

# Disable preallocation (allocate on demand)
export XLA_PYTHON_CLIENT_PREALLOCATE=False

# Set absolute memory limit
export XLA_PYTHON_CLIENT_GPU_MEM_LIMIT=4GiB

# Use BFC allocator (allows memory growth)
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
```

### 39.3.3 Distributed Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `JAX_COORDINATION_SERVICE` | `""` | Coordination service for multi-process |
| `JAX_DISTRIBUTED_DEBUG` | `False` | Enable distributed debugging |
| `JAX_USE_SHARDED_MAP` | `False` | Use sharded map for distributed |
| `NPROC_PER_NODE` | `""` | Number of processes per node |
| `MASTER_ADDR` | `""` | Address of the coordinator |
| `MASTER_PORT` | `""` | Port of the coordinator |

```bash
# Multi-process setup
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=12345
export JAX_COORDINATION_SERVICE=cherney

# Launch with 4 processes on 2 nodes
# Node 0:
python -m jax.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 train.py
# Node 1:
python -m jax.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 train.py
```

### 39.3.4 Compilation and Tracing

| Environment Variable | Default | Description |
|---|---|---|
| `JAX_TRACER_LEVEL` | `0` | Tracing verbosity level |
| `JAX_PROFILE` | `False` | Enable profiling |
| `JAX_CACHE_IR` | `True` | Cache intermediate IR |
| `JAX_EXPLICIT_MEM` | `False` | Explicit memory management |
| `JAX_COMPILATION_CACHE` | `True` | Enable compilation caching |

```bash
# Debug compilation issues
export JAX_TRACER_LEVEL=1

# Disable compilation cache (for debugging)
export JAX_COMPILATION_CACHE=False
```

---

## 39.4 XLA Flags

XLA (Accelerated Linear Algebra) is JAX's compilation backend. XLA flags
provide fine-grained control over compilation behavior.

### 39.4.1 Setting XLA Flags

```python
# Must be set BEFORE importing JAX
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=4'

import jax  # Import after setting flags
```

Or via command line:

```bash
XLA_FLAGS="--xla_gpu_force_compilation_parallelism=4" python my_script.py
```

### 39.4.2 Common XLA Flags

| Flag | Description |
|---|---|
| `--xla_gpu_force_compilation_parallelism=N` | Parallel compilation threads |
| `--xla_gpu_autotune_level=N` | Autotuning level (0-4) |
| `--xla_gpu_enable_triton_gemm` | Enable Triton GEMM kernels |
| `--xla_gpu_enable_cublaslt` | Enable cuBLASLt for matmul |
| `--xla_gpu_enable_highest_priority_async_stream` | High-priority async streams |
| `--xla_gpu_enable_command_buffer` | Command buffer optimization |
| `--xla_gpu_crash_on_internal_error` | Crash on XLA internal errors |
| `--xla_force_host_platform_device_count=N` | Simulate N CPU devices |
| `--xla_tensor_fusion_budget=N` | Fusion budget in bytes |
| `--xla_disable_hlo_passes=pass1,pass2` | Disable specific HLO passes |
| `--xla_enable_hlo_passes_only=pass1` | Enable only specific passes |
| `--xla_dump_to=/path` | Dump HLO to directory |
| `--xla_dump_hlo_as_text` | Dump HLO as text files |
| `--xla_dump_hlo_as_proto` | Dump HLO as protobuf |

```bash
# Dump HLO for debugging compilation
XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text" python train.py

# Force CPU device count for testing
XLA_FLAGS="--xla_force_host_platform_device_count=8" python test_sharding.py

# Disable specific optimizations for debugging
XLA_FLAGS="--xla_disable_hlo_passs=algebraic-simplifier,broadcast-simplifier" python debug.py
```

### 39.4.3 GPU-Specific XLA Flags

```bash
# Enable Triton-based GEMM (often faster on Ampere+)
XLA_FLAGS="--xla_gpu_enable_triton_gemm" python train.py

# Control autotuning (higher = more compile time, better runtime)
XLA_FLAGS="--xla_gpu_autotune_level=4" python train.py

# Enable command buffer for reduced launch overhead
XLA_FLAGS="--xla_gpu_enable_command_buffer" python train.py
```

### 39.4.4 TPU-Specific XLA Flags

```bash
# TPU-specific flags
XLA_FLAGS="--xla_tpu_enable_spmd_resharding" python train.py
XLA_FLAGS="--xla_tpu_enable_data_parallel_all_reduce_opt" python train.py
```

---

## 39.5 NCCL Flags

For distributed GPU training, JAX uses NCCL (NVIDIA Collective Communications
Library). NCCL flags control communication behavior.

### 39.5.1 Common NCCL Flags

| Environment Variable | Default | Description |
|---|---|---|
| `NCCL_DEBUG` | `WARN` | Debug level: `VERSION`, `WARN`, `INFO`, `TRACE` |
| `NCCL_DEBUG_SUBSYS` | `ALL` | Subsystem filter: `NET`, `COLL`, `SHM`, etc. |
| `NCCL_SOCKET_IFNAME` | auto | Network interface for communication |
| `NCCL_IB_DISABLE` | `0` | Disable InfiniBand (force TCP) |
| `NCCL_IB_HCA` | auto | InfiniBand HCA to use |
| `NCCL_NET_GDR_LEVEL` | `3` | GPUDirect RDMA level |
| `NCCL_MIN_NCHANNELS` | `1` | Minimum number of channels |
| `NCCL_MAX_NCHANNELS` | `4` | Maximum number of channels |
| `NCCL_P2P_DISABLE` | `0` | Disable P2P (peer-to-peer) transfers |
| `NCCL_SHM_DISABLE` | `0` | Disable shared memory (intra-node) |
| `NCCL_BUFFSIZE` | varies | Per-channel buffer size |
| `NCCL_NET_CHUNK_SIZE` | varies | Network chunk size |
| `NCCL_ALGO` | auto | Algorithm: `Tree`, `Ring`, `CollnetDirect` |
| `NCCL_PROTO` | auto | Protocol: `LL`, `LL128`, `Simple` |
| `NCCL_MAX_NRINGS` | varies | Maximum number of rings for collectives |
| `NCCL_ASSIST_THREAD_ENABLED` | `1` | Enable assist thread for proxy ops |
| `NCCL_TOPO_FILE` | auto | Path to topology file |
| `NCCL_TOPO_DUMP_FILE` | `""` | Dump topology to this file |
| `NCCL_NET` | auto | Network plugin: `Socket`, `IB`, `AWS` |
| `NCCL_COLLNET_ENABLE` | `0` | Enable CollNet (SHARP) |

```bash
# Debug NCCL issues
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Force specific network interface
export NCCL_SOCKET_IFNAME=eth0

# Disable InfiniBand (fallback to TCP sockets)
export NCCL_IB_DISABLE=1

# Increase channel count for higher bandwidth
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=16

# Use ring algorithm for all reduce
export NCCL_ALGO=Ring
export NCCL_MAX_NRINGS=8
```

### 39.5.2 NCCL Performance Tuning

```bash
# High-performance configuration for multi-node GPU training
export NCCL_MIN_NCHANNELS=16        # More channels = more bandwidth
export NCCL_MAX_NCHANNELS=32
export NCCL_NET_GDR_LEVEL=5         # Maximum GPUDirect RDMA
export NCCL_IB_GPU_DIRECT_RDMA=1    # Enable GPU Direct RDMA over IB
export NCCL_IB_RETRY_CNT=7          # Increase IB retry count
export NCCL_IB_TIMEOUT=22           # IB timeout (log2 scale)
export NCCL_IB_QPS_PER_CONNECTION=4 # Multiple QPs per connection

# For NVLink-only single-node
export NCCL_P2P_LEVEL=5             # Maximize NVLink usage
export NCCL_SHM_DISABLE=0           # Enable shared memory
```

### 39.5.3 NCCL Troubleshooting

```bash
# Common NCCL issues and fixes:

# 1. "NCCL error: unhandled system error"
export NCCL_DEBUG=INFO              # Get detailed logs
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback

# 2. "Network Interface not found"
export NCCL_SOCKET_IFNAME=eth0      # Specify correct interface
# Or use:
ip addr show  # Find available interfaces

# 3. "Failed to initialize NCCL"
export NCCL_IB_DISABLE=1            # Disable InfiniBand
export NCCL_P2P_DISABLE=1           # Disable P2P (for debugging)

# 4. Timeout during collectives
export NCCL_COMM_BLOCKING=1         # Blocking communication mode
export NCCL_LAUNCH_MODE=PARALLEL    # Parallel launch mode
```

---

## 39.6 Memory Configuration

### 39.6.1 GPU Memory Management

```python
import jax

# Option 1: Set memory fraction (before any JAX computation)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

# Option 2: Set absolute limit
# export XLA_PYTHON_CLIENT_GPU_MEM_LIMIT=8GiB

# Option 3: Disable preallocation (allocate as needed)
# export XLA_PYTHON_CLIENT_PREALLOCATE=False

# Option 4: BFC allocator (allows growth with fragmentation)
# export XLA_PYTHON_CLIENT_ALLOCATOR=bfc

# Check current memory usage
devices = jax.devices()
for d in devices:
    if d.device_kind == 'gpu':
        stats = d.memory_stats()
        if stats:
            print(f"Device {d.id}:")
            print(f"  Bytes used: {stats['bytes_in_use'] / 1e9:.2f} GB")
            print(f"  Peak bytes: {stats['peak_bytes_in_use'] / 1e9:.2f} GB")
            print(f"  Pool bytes: {stats['pool_bytes'] / 1e9:.2f} GB")
```

### 39.6.2 Memory Fragmentation

```bash
# Use BFC allocator to handle fragmentation
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc

# Or use platform allocator (may be more stable)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

---

## 39.7 Profiling Configuration

### 39.7.1 Profiling Flags

```bash
# Enable profiling
export JAX_PROFILE=True
export JAX_PROFILER_LOG_DIR=/tmp/jax_profile

# TensorBoard profiling
export TENSORBOARD_PROFILER_DIRECTORY=/tmp/tb_profile
```

### 39.7.2 Programmatic Profiling

```python
import jax.profiler

# Start profiling
jax.profiler.start_trace("/tmp/jax_trace")

# ... run your computation ...

# Stop profiling
jax.profiler.stop_trace()

# Or use context manager
with jax.profiler.trace("/tmp/jax_trace", create_perfetto_trace=True):
    result = jax.jit(my_fn)(x)
    result.block_until_ready()
```

---

## 39.8 Debugging Configuration

### 39.8.1 Debug Mode Combination

```bash
# Full debug mode (slow but informative)
export JAX_DISABLE_JIT=True
export JAX_DEBUG_NANS=True
export JAX_DEBUG_INFS=True
export JAX_ENABLE_CHECKS=True
export JAX_TRACEBACK_FILTERING=off
export NCCL_DEBUG=INFO
```

### 39.8.2 Selective Debugging

```bash
# Debug NaN only
export JAX_DEBUG_NANS=True

# Debug shape mismatches only
export JAX_ENABLE_CHECKS=True

# Get full Python tracebacks (not JAX-filtered)
export JAX_TRACEBACK_FILTERING=off
```

### 39.8.3 Debugging JIT Compilation

```bash
# Dump HLO during compilation
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text"

# Disable specific optimizations
export XLA_FLAGS="--xla_disable_hlo_passs=algebraic-simplifier"
```

---

## 39.9 Configuration Reference Table

### 39.9.1 All JAX Configuration Options

| Option | Type | Default | Description |
|---|---|---|---|
| `jax_platforms` | str | `""` | Platform preference order |
| `jax_enable_x64` | bool | `False` | Enable 64-bit types |
| `jax_disable_jit` | bool | `False` | Disable JIT compilation |
| `jax_debug_nans` | bool | `False` | NaN detection |
| `jax_debug_infs` | bool | `False` | Inf detection |
| `jax_enable_checks` | bool | `False` | Shape/dtype checks |
| `jax_traceback_filtering` | str | `"trace"` | Traceback filtering mode |
| `jax_pmap_no_rank_reduction` | bool | `False` | pmap rank behavior |
| `ax_gpu_mem_fraction` | float | `0.75` | GPU memory preallocation fraction |
| `jax_default_matmul_precision` | str | `"highest"` | Default matmul precision |
| `jax_default_device` | str | `""` | Default device |
| `jax_distributed_debug` | bool | `False` | Distributed debugging |
| `jax_coordination_service` | str | `""` | Coordination service type |
| `jax_numpy_rank_promotion` | str | `"allow"` | NumPy rank promotion policy |
| `jax_numpy_dtype_promotion` | str | `"standard"` | Dtype promotion policy |
| `jax_threefry_partitionable` | bool | `True` | PRNG partitionability |
| `jax_tracer_level` | int | `0` | Tracing verbosity |
| `jax_cache_ir` | bool | `True` | IR caching |
| `jax_compilation_cache` | bool | `True` | Compilation caching |
| `jax_experimental_mem` | bool | `False` | Experimental memory management |
| `jax_legacy_prng_key` | str | `"allow"` | Legacy PRNG key behavior |

---

## 39.10 Complete Setup Examples

### 39.10.1 Development Setup

```bash
# .env.dev
export JAX_ENABLE_X64=True
export JAX_DEBUG_NANS=True
export JAX_TRACEBACK_FILTERING=off
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=False
```

### 39.10.2 Multi-GPU Training Setup

```bash
# .env.gpu_training
export JAX_PLATFORMS=gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_triton_gemm --xla_gpu_autotune_level=4"
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=16
export NCCL_DEBUG=WARN
```

### 39.10.3 Multi-Node TPU Setup

```bash
# .env.tpu_distributed
export JAX_PLATFORMS=tpu
export JAX_COORDINATION_SERVICE=cherney
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=12345
export JAX_DISTRIBUTED_DEBUG=True
export XLA_FLAGS="--xla_tpu_enable_spmd_resharding"
```

### 39.10.4 Memory-Constrained Setup

```bash
# .env.low_memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export JAX_ENABLE_X64=False
export XLA_FLAGS="--xla_tensor_fusion_budget=1048576"
```
