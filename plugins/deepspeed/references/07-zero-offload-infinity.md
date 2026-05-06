# ZeRO-Offload and ZeRO-Infinity

## Overview

ZeRO-Offload and ZeRO-Infinity extend the ZeRO optimization family by enabling offloading of model training states to secondary memory devices (CPU RAM and NVMe storage). This allows training models that would otherwise exceed the aggregate GPU memory across all available devices.

- **ZeRO-Offload**: Offloads optimizer states and computation to CPU memory. Designed for models up to ~10B parameters on limited GPU resources.
- **ZeRO-Infinity**: Extends offloading to NVMe storage, enabling training of models with hundreds of billions of parameters on modest GPU clusters by leveraging the vast capacity of SSDs.

### Memory Hierarchy

```
GPU Memory (HBM)          CPU Memory (DRAM)           NVMe Storage (SSD)
  ~40-80 GB                 ~256-1024 GB                ~1-100 TB
  Bandwidth: ~2 TB/s       Bandwidth: ~50 GB/s         Bandwidth: ~3-7 GB/s
  Latency: ~ns             Latency: ~100ns             Latency: ~us-ms
  
  <-- Fastest, Smallest --  -- Medium Speed/Size --    -- Slowest, Largest -->
```

## ZeRO-Offload

ZeRO-Offload (introduced in ZeRO-Offload: Democratizing Billion-Scale Model Training, SC 2021) offloads optimizer states and optimizer computation to CPU while keeping parameters and forward/backward computation on GPU.

### Architecture

```
GPU                              CPU
+----------------------------+   +----------------------------+
| Forward Pass               |   |                            |
|   Parameters (FP16)        |   |                            |
|   Activations              |   |                            |
+----------------------------+   |                            |
| Backward Pass              |   |                            |
|   Gradients (FP16)         |   |                            |
|   Partial gradient comp    |   |                            |
+----------------------------+   +----------------------------+
| Gradient send ----------->|-->| Optimizer Step             |
|                            |   |   FP32 Optimizer States    |
|                            |   |   FP32 Master Weights      |
|                            |   |   CPU Adam Update          |
|                            |   |                            |
|<------- Updated params ----|---|   FP16 Param Update        |
+----------------------------+   +----------------------------+
```

### DeepSpeedZeRoOffload Class

The CPU offloading logic for parameters is managed by `DeepSpeedZeRoOffload` in `deepspeed/runtime/zero/parameter_offload.py`:

```python
class DeepSpeedZeRoOffload(object):
    """Manages offloading of parameters to CPU memory.
    
    Handles the transfer of parameters between GPU and CPU,
    including prefetching and pin_memory optimizations.
    """
    
    def __init__(self,
                 module,
                 timers,
                 ds_config,
                 overlap_comm=True,
                 prefetch_in_gpu_memory=False,
                 max_num_params_per_cpu_tensor=1e8):
        ...
```

### Forward/Backward Hooks for Parameter Swapping

ZeRO-Offload registers forward and backward hooks on model modules to manage parameter movement between CPU and GPU:

```python
def _register_hooks(self, module):
    """Register forward/backward hooks for parameter swapping."""
    
    def _pre_forward_module_hook(module, *args):
        """Before forward: move needed parameters to GPU."""
        for param in module.parameters():
            if param.ds_offload and not param.ds_on_gpu:
                self._move_param_to_gpu(param)
    
    def _post_forward_module_hook(module, *args):
        """After forward: optionally release GPU copy."""
        if self.offload_param:
            for param in module.parameters():
                if param.ds_offload and param.ds_on_gpu:
                    self._release_gpu_copy(param)
    
    def _pre_backward_module_hook(module, *args):
        """Before backward: move needed parameters to GPU."""
        for param in module.parameters():
            if param.ds_offload and not param.ds_on_gpu:
                self._move_param_to_gpu(param)
    
    # Register hooks on all modules
    for module in self.module.modules():
        module.register_forward_pre_hook(_pre_forward_module_hook)
        module.register_forward_hook(_post_forward_module_hook)
        module.register_full_backward_pre_hook(_pre_backward_module_hook)
```

### Prefetching Mechanisms

To hide CPU-GPU transfer latency, ZeRO-Offload supports prefetching parameters for upcoming layers:

```python
def _prefetch_params(self, current_module, all_modules):
    """Prefetch parameters for modules after current_module."""
    current_idx = all_modules.index(current_module)
    prefetch_count = min(self.prefetch_count, len(all_modules) - current_idx - 1)
    
    for i in range(1, prefetch_count + 1):
        next_module = all_modules[current_idx + i]
        for param in next_module.parameters():
            if param.ds_offload and not param.ds_on_gpu:
                self._async_move_param_to_gpu(param)  # Non-blocking transfer
```

### pin_memory Option

```python
# When pin_memory=True, CPU tensors are allocated in pinned (page-locked) memory
# This enables faster DMA transfers to GPU via PCIe
# Trade-off: pinned memory is a limited resource; too much can cause system instability

def _allocate_cpu_tensor(self, shape, dtype):
    if self.pin_memory:
        return torch.empty(shape, dtype=dtype).pin_memory()
    else:
        return torch.empty(shape, dtype=dtype)
```

### Buffer Management

ZeRO-Offload uses buffer pools to reuse memory allocations:

```python
class OffloadBufferPool:
    """Pool of reusable CPU/GPU buffers for parameter transfers."""
    
    def __init__(self, buffer_size, buffer_count, device):
        self.buffers = [
            torch.empty(buffer_size, dtype=torch.float16, device=device)
            for _ in range(buffer_count)
        ]
        self.available = list(range(buffer_count))
    
    def acquire(self):
        if not self.available:
            raise RuntimeError("No buffers available")
        idx = self.available.pop(0)
        return idx, self.buffers[idx]
    
    def release(self, idx):
        self.available.append(idx)
```

### ZeRO-Offload with Stage 1

In Stage 1 + Offload, optimizer states live on CPU:

```json
{
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Memory on GPU**: $2\Psi$ (FP16 params) + $2\Psi$ (FP16 grads) = $4\Psi$ bytes
**Memory on CPU**: $12\Psi$ (FP32 optimizer states) per rank

### ZeRO-Offload with Stage 2

Stage 2 + Offload keeps both optimizer states and gradients on CPU:

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Memory on GPU**: $2\Psi$ (FP16 params) + working buffers
**Memory on CPU**: $14\Psi$ (FP32 optimizer states + FP16 gradients)

## ZeRO-Infinity

ZeRO-Infinity (introduced in ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning, SC 2021) extends offloading to NVMe storage, enabling training models that exceed both GPU and CPU memory combined.

### Architecture

```
GPU HBM                   CPU DRAM                    NVMe SSD
+-------------------+    +-------------------+     +-------------------+
| Working params    |    | Active params     |     | All parameters    |
| (FP16, ~few GB)   |    | (FP16, staged)    |     | (FP16, full model)|
|                   |    | Optimizer states  |     | Optimizer states  |
| Activations       |    | (FP32, partial)   |     | (FP32, full)      |
+-------------------+    +-------------------+     +-------------------+
     ^     |                  ^     |                    ^     |
     |     v                  |     v                    |     v
     +-- PCIe ---------------+--- DMA -----------------+
         ~32 GB/s                  ~3-7 GB/s
```

### AsyncPartitionedParameterSwapper

The core component for NVMe offloading is `AsyncPartitionedParameterSwapper` in `deepspeed/runtime/zero/offload_config.py` and related modules:

```python
class AsyncPartitionedParameterSwapper:
    """Manages asynchronous swapping of partitioned parameters between
    NVMe, CPU, and GPU memory hierarchically.
    
    Implements double-buffering and pipelining to overlap:
    - NVMe -> CPU reads
    - CPU -> GPU transfers  
    - GPU computation
    - GPU -> CPU writes
    - CPU -> NVMe writes
    """
    
    def __init__(self, ds_config, model_dtype, nvme_swapper=None):
        self.model_dtype = model_dtype
        self.swap_config = ds_config.zero_config.offload_param
        
        # Swap buffers for pipelined transfers
        self.swap_buffer_pool = SwapBufferPool(
            buffer_size=self.swap_config.buffer_size,
            buffer_count=self.swap_config.buffer_count,
            max_in_cpu=self.swap_config.max_in_cpu
        )
        
        # NVMe aio handle for async I/O
        if self.swap_config.device == OffloadDeviceEnum.nvme:
            self.nvme_swapper = nvme_swapper or self._init_nvme_swapper()
```

### SwapBufferPool

```python
class SwapBufferPool:
    """Pool of reusable buffers for parameter swapping operations.
    
    Manages a fixed set of buffers that are reused across swap operations
    to avoid repeated allocation/deallocation overhead.
    """
    
    def __init__(self, buffer_size, buffer_count, max_in_cpu):
        self.buffer_size = buffer_size  # Size of each buffer in bytes
        self.buffer_count = buffer_count  # Total number of buffers
        self.max_in_cpu = max_in_cpu  # Max buffers resident in CPU simultaneously
        
        # Pre-allocate CPU buffers
        self.cpu_buffers = [
            torch.empty(buffer_size, dtype=torch.float16).pin_memory()
            for _ in range(buffer_count)
        ]
        
        # Track buffer states
        self.buffer_states = [BufferState.FREE] * buffer_count
```

### NVMe Path Configuration

The NVMe path specifies where parameter and optimizer state files are stored on the local SSD:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme/deepspeed_offload/"
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme/deepspeed_offload/"
        }
    }
}
```

DeepSpeed creates a directory structure under the specified path:

```
/local_nvme/deepspeed_offload/
  rank0/
    params/           # Parameter shards for rank 0
    optimizer/        # Optimizer state shards for rank 0
  rank1/
    params/
    optimizer/
  ...
```

## OffloadDeviceEnum

```python
# deepspeed/runtime/zero/offload_config.py
class OffloadDeviceEnum(str, Enum):
    none = "none"     # No offloading; all states on GPU
    cpu = "cpu"       # Offload to CPU (DRAM)
    nvme = "nvme"     # Offload to NVMe (SSD)
```

| Device | Bandwidth | Latency | Use Case |
|--------|-----------|---------|----------|
| `none` | ~2 TB/s (HBM) | ~ns | Model fits entirely in GPU memory |
| `cpu` | ~32 GB/s (PCIe) | ~100ns | Optimizer offload, partial parameter offload |
| `nvme` | ~3-7 GB/s (NVMe) | ~us | Extreme-scale models exceeding CPU memory |

## DeepSpeedZeroOffloadParamConfig

Controls parameter offloading behavior:

```python
class DeepSpeedZeroOffloadParamConfig:
    device: OffloadDeviceEnum = OffloadDeviceEnum.none
    nvme_path: str = "/local_nvme"        # NVMe mount point for parameter storage
    buffer_count: int = 5                  # Number of swap buffers for param transfer
    buffer_size: int = 1e8                 # Size of each buffer in elements
    max_in_cpu: int = 1e9                  # Max elements kept in CPU simultaneously
    pin_memory: bool = False               # Use pinned memory for CPU tensors
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | OffloadDeviceEnum | `"none"` | Target offload device: `"none"`, `"cpu"`, or `"nvme"` |
| `nvme_path` | str | `"/local_nvme"` | Path to NVMe mount point (only used when `device="nvme"`) |
| `buffer_count` | int | 5 | Number of pre-allocated buffers for swap operations. More buffers enable deeper pipelining but consume more memory |
| `buffer_size` | int | $1 \times 10^8$ | Size of each swap buffer in number of elements. Must be large enough to hold the largest parameter group |
| `max_in_cpu` | int | $1 \times 10^9$ | Maximum number of parameter elements kept in CPU DRAM simultaneously. Limits CPU memory usage |
| `pin_memory` | bool | false | Use CUDA pinned memory for CPU buffers. Improves CPU-GPU transfer speed at the cost of reduced available system memory |

### Example Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/fast_ssd/deepspeed_swap/",
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9,
            "pin_memory": true
        }
    }
}
```

## DeepSpeedZeroOffloadOptimizerConfig

Controls optimizer state offloading:

```python
class DeepSpeedZeroOffloadOptimizerConfig:
    device: OffloadDeviceEnum = OffloadDeviceEnum.none
    nvme_path: str = "/local_nvme"
    buffer_count: int = 4
    pin_memory: bool = False
    pipeline_read: bool = False
    pipeline_write: bool = False
    fast_init: bool = False
    ratio: float = 1.0
    super_offload: bool = False
    cpuadam_cores_perc: float = 1.0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | OffloadDeviceEnum | `"none"` | Target offload device for optimizer states |
| `nvme_path` | str | `"/local_nvme"` | NVMe path for optimizer state storage |
| `buffer_count` | int | 4 | Number of buffers for optimizer state swap operations |
| `pin_memory` | bool | false | Use pinned memory for optimizer state CPU buffers |
| `pipeline_read` | bool | false | Pipeline NVMe reads: overlap reading optimizer states with GPU computation |
| `pipeline_write` | bool | false | Pipeline NVMe writes: overlap writing updated optimizer states with next forward pass |
| `fast_init` | bool | false | Enable fast optimizer state initialization by skipping zero-fills. Only use when states are immediately overwritten |
| `ratio` | float | 1.0 | Fraction of optimizer states to offload (0.0-1.0). 1.0 = offload all, 0.5 = offload half |
| `super_offload` | bool | false | Enable SuperOffload mode for high-performance CPU optimization |
| `cpuadam_cores_perc` | float | 1.0 | Fraction of CPU cores to use for CPU Adam optimizer (0.0-1.0) |

### Example Configuration

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "pipeline_read": true,
            "pipeline_write": true,
            "ratio": 1.0,
            "cpuadam_cores_perc": 0.75
        }
    }
}
```

## Partial Offloading

The `ratio` parameter enables partial offloading, keeping some optimizer states on GPU while offloading the rest to CPU:

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "ratio": 0.5,
            "pin_memory": true
        }
    }
}
```

With `ratio=0.5`:
- 50% of optimizer states remain on GPU (for parameters in active computation)
- 50% are offloaded to CPU
- The selection is based on parameter access frequency during forward/backward

**Use cases for partial offloading:**
- GPU has enough memory for a fraction of optimizer states
- Reducing CPU-GPU transfer overhead by keeping hot optimizer states on GPU
- Balancing GPU memory usage with training throughput

### Ratio Selection Guide

| ratio | GPU Optimizer Memory | CPU Optimizer Memory | Throughput Impact |
|-------|---------------------|---------------------|-------------------|
| 0.0 | $12\Psi$ (full) | 0 | No impact (no offload) |
| 0.25 | $9\Psi$ | $3\Psi$ | Minimal |
| 0.5 | $6\Psi$ | $6\Psi$ | Low |
| 0.75 | $3\Psi$ | $9\Psi$ | Moderate |
| 1.0 | 0 | $12\Psi$ (full) | Significant |

## CPU Adam Optimizer (DeepSpeedCPUAdam)

When optimizer states are offloaded to CPU, DeepSpeed uses a custom CPU Adam implementation for high-performance optimization on the CPU:

```python
# deepspeed/ops/adam/cpu_adam.py
class DeepSpeedCPUAdam(torch.optim.Optimizer):
    """High-performance CPU Adam optimizer.
    
    Optimized multi-threaded Adam implementation for CPU offloading.
    Achieves near-peak CPU memory bandwidth utilization by:
    1. Parallelizing across CPU cores
    2. SIMD-vectorized operations (AVX2/AVX-512)
    3. Cache-friendly memory access patterns
    
    Performance: ~40-60 GFLOPS on modern CPUs (vs ~5-10 GFLOPS for PyTorch Adam)
    """
    
    optimizer_id = -1  # Class-level ID for C++ backend
    
    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True,
                 fp32_optimizer_states=True):
        ...
```

### CPU Adam Performance

| CPU | Cores | Throughput (GB/s) | Time per 1B params (ms) |
|-----|-------|-------------------|------------------------|
| Intel Xeon 8280 | 28 | ~120 | ~50 |
| AMD EPYC 7763 | 64 | ~180 | ~35 |
| ARM Neoverse V2 | 64 | ~100 | ~60 |

### Controlling CPU Core Usage

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "cpuadam_cores_perc": 0.75
        }
    }
}
```

With `cpuadam_cores_perc=0.75`, CPU Adam uses 75% of available CPU cores, leaving 25% for other tasks (data loading, etc.).

## AIO (Async IO) Configuration

For NVMe offloading, DeepSpeed uses a custom AIO library for high-throughput asynchronous I/O operations:

```json
{
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": false,
        "overlap_events": true
    }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `block_size` | int | 1048576 (1 MB) | Block size in bytes for each NVMe read/write operation. Larger blocks improve sequential throughput but may increase latency for small transfers |
| `queue_depth` | int | 8 | Number of outstanding I/O operations in the queue. Higher values enable more pipelining but require more pinned memory for buffers |
| `thread_count` | int | 1 | Number of threads for I/O operations. Increase for multi-queue NVMe devices |
| `single_submit` | bool | false | Submit I/O requests one at a time instead of batching. Set to `true` for debugging or with single-queue devices |
| `overlap_events` | bool | true | Overlap I/O completion checking with submission. Improves throughput for mixed read/write workloads |

### AIO Tuning Guide

| NVMe Type | block_size | queue_depth | thread_count | Expected Throughput |
|-----------|------------|-------------|--------------|---------------------|
| SATA SSD | 524288 | 4 | 1 | ~500 MB/s |
| NVMe Gen3 | 1048576 | 8 | 1 | ~3 GB/s |
| NVMe Gen4 | 1048576 | 16 | 2 | ~5-7 GB/s |
| NVMe Gen5 | 2097152 | 32 | 4 | ~10-14 GB/s |

## GDS (GPU Direct Storage) Support

DeepSpeed supports GPUDirect Storage (GDS) for direct GPU-to-NVMe transfers, bypassing CPU memory entirely:

```json
{
    "aio": {
        "block_size": 1048576,
        "queue_depth": 16,
        "thread_count": 1,
        "single_submit": false,
        "overlap_events": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/gds_mount/",
            "use_gds": true
        }
    }
}
```

GDS benefits:
- Eliminates CPU memory copy: data moves directly from NVMe to GPU memory
- Reduces PCIe round-trips (NVMe -> CPU -> GPU becomes NVMe -> GPU)
- Requires NVIDIA GPUDirect Storage driver and compatible hardware

## Memory Management Strategies

### Strategy 1: GPU-Only (No Offload)

```
GPU Memory Layout:
+-----------------------------+
| Parameters (FP16)           |
| Gradients (FP16)            |
| Optimizer States (FP32)     |
| Activations                 |
+-----------------------------+
Total: 16*Psi + activation_memory

Best for: Small models (< 7B with 80 GB GPUs)
```

### Strategy 2: Optimizer Offload to CPU

```
GPU Memory Layout:            CPU Memory Layout:
+-------------------+         +-------------------+
| Parameters (FP16) |         | Optimizer States  |
| Gradients (FP16)  |         |   Master Weights  |
| Activations       |         |   Momentum        |
+-------------------+         |   Variance        |
Total: 4*Psi + act            +-------------------+
                              Total: 12*Psi

Best for: Medium models (7B-13B) on 40 GB GPUs
```

### Strategy 3: Parameter + Optimizer Offload to CPU

```
GPU Memory Layout:            CPU Memory Layout:
+-------------------+         +-------------------+
| Working params    |         | Parameters (FP16) |
| Activations       |         | Gradients (FP16)  |
+-------------------+         | Optimizer States  |
Total: ~few GB + act          +-------------------+
                              Total: 16*Psi

Best for: Large models (13B-30B) on limited GPUs
```

### Strategy 4: Full NVMe Offload (ZeRO-Infinity)

```
GPU Memory Layout:            CPU Memory Layout:          NVMe Layout:
+-------------------+         +-------------------+       +-------------------+
| Active params     |         | Staged params     |       | All Parameters    |
| Activations       |         | Active optimizer  |       | All Optim States  |
+-------------------+         +-------------------+       +-------------------+
Total: ~few GB                Total: Limited by DRAM      Total: Model size

Best for: Very large models (30B+) with limited GPU + CPU memory
```

## Configuration Examples

### Example 1: Stage 2 with CPU Optimizer Offload

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 5e8
    }
}
```

### Example 2: Stage 3 with Full CPU Offload

```json
{
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "cpuadam_cores_perc": 0.75
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "param_persistence_threshold": 1e5
    }
}
```

### Example 3: Stage 3 with NVMe Offload (ZeRO-Infinity)

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 64,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 16,
        "thread_count": 2,
        "single_submit": false,
        "overlap_events": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvde_infinity/",
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9,
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvde_infinity/",
            "buffer_count": 4,
            "pin_memory": true,
            "pipeline_read": true,
            "pipeline_write": true,
            "fast_init": false
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 5e8,
        "param_persistence_threshold": 1e5
    }
}
```

### Example 4: Partial Offloading with Ratio

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true,
            "ratio": 0.6
        },
        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

### Example 5: Stage 3 with CPU Param Offload + NVMe Optimizer Offload

```json
{
    "aio": {
        "block_size": 2097152,
        "queue_depth": 16,
        "thread_count": 2,
        "single_submit": false,
        "overlap_events": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 5e8
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/fast_ssd/offload/",
            "pin_memory": true,
            "pipeline_read": true,
            "pipeline_write": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 5e8,
        "param_persistence_threshold": 1e5
    }
}
```

## Performance Characteristics

### Throughput Impact by Offload Strategy

Measured on 8x A100-40GB GPUs with a 7B parameter model:

| Configuration | TFLOPS/GPU | Samples/sec | Step Time (s) | Memory/GPU |
|---------------|-----------|-------------|----------------|------------|
| Stage 2 (no offload) | 156 | 320 | 0.8 | 32 GB |
| Stage 2 + CPU optim | 142 | 290 | 0.9 | 18 GB |
| Stage 3 (no offload) | 138 | 280 | 0.9 | 16 GB |
| Stage 3 + CPU offload | 98 | 200 | 1.3 | 6 GB |
| Stage 3 + NVMe offload | 45 | 92 | 2.8 | 4 GB |

### Key Performance Factors

1. **PCIe Bandwidth**: CPU offload throughput is limited by PCIe bandwidth (~32 GB/s for PCIe Gen4 x16)
2. **CPU Adam Speed**: Multi-core CPU Adam throughput determines optimizer step time
3. **NVMe Throughput**: NVMe offload is limited by SSD sequential throughput
4. **Buffer Size**: Larger buffers amortize transfer overhead but increase peak memory
5. **Pipeline Depth**: Enabling `pipeline_read` and `pipeline_write` can hide 30-50% of transfer latency

## Best Practices

1. **Start with CPU offload before NVMe**: CPU offload provides better throughput for models that fit within CPU memory.

2. **Use pin_memory**: Always enable `pin_memory=true` for CPU offloading to maximize PCIe transfer speed.

3. **Tune buffer_count**: Increase `buffer_count` (e.g., to 8-10) to enable deeper pipelining, especially for NVMe offload.

4. **Reduce max_live_parameters for NVMe**: Lower `max_live_parameters` when using NVMe offload to minimize CPU memory pressure.

5. **Pipeline reads and writes**: Enable both `pipeline_read` and `pipeline_write` for NVMe offloading to overlap I/O with computation.

6. **Allocate sufficient CPU cores**: Set `cpuadam_cores_perc` based on available cores. Leave some cores for data loading and I/O threads.

7. **Use fast NVMe SSDs**: NVMe Gen4+ SSDs with high sequential read/write speeds are critical for ZeRO-Infinity performance.

8. **Monitor CPU memory**: Ensure sufficient free CPU memory for buffers. Swap usage on the CPU will catastrophically degrade performance.

## Key Source Files

| File | Description |
|------|-------------|
| `deepspeed/runtime/zero/offload_config.py` | OffloadDeviceEnum, offload configuration classes |
| `deepspeed/runtime/zero/parameter_offload.py` | DeepSpeedZeRoOffload, parameter offloading logic |
| `deepspeed/runtime/zero/stage1and2.py` | Optimizer state offloading in Stages 1 & 2 |
| `deepspeed/runtime/zero/stage3.py` | DeepSpeedZeroOptimizer_Stage3 with offload support |
| `deepspeed/runtime/zero/utils.py` | ZeRO utility functions including buffer management |
| `deepspeed/ops/adam/cpu_adam.py` | DeepSpeedCPUAdam optimizer |
| `deepspeed/ops/op_builder/cpu_adam.py` | CPU Adam C++/CUDA extension builder |
| `deepspeed/runtime/swap_tensor/` | AsyncPartitionedParameterSwapper, SwapBufferPool |
| `deepspeed/runtime/swap_tensor/aio/utils.py` | AIO configuration and utilities |
