# DeepSpeed Accelerator Abstraction Layer

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [DeepSpeedAccelerator Abstract Base Class](#deepspeedaccelerator-abstract-base-class)
4. [CudaAccelerator (NVIDIA)](#cudaaccelerator-nvidia)
5. [CpuAccelerator](#cpuaccelerator)
6. [HpuAccelerator (Intel Gaudi)](#hpuaccelerator-intel-gaudi)
7. [NpuAccelerator (Huawei Ascend)](#npuaccelerator-huawei-ascend)
8. [XpuAccelerator (Intel GPU)](#xpuaccelerator-intel-gpu)
9. [MluAccelerator (Cambricon)](#mluaccelerator-cambricon)
10. [SdaaAccelerator (Tecorigin)](#sdaaaccelerator-tecorigin)
11. [MpsAccelerator (Apple Silicon)](#mpsaccelerator-apple-silicon)
12. [RealAccelerator: Auto-Detection and Dispatch](#realaccelerator-auto-detection-and-dispatch)
13. [Adding a Custom Accelerator](#adding-a-custom-accelerator)
14. [Environment Variables](#environment-variables)
15. [Configuration Examples](#configuration-examples)
16. [Troubleshooting](#troubleshooting)

---

## Overview

The DeepSpeed Accelerator Abstraction Layer (`deepspeed/accelerator/`) provides a unified hardware abstraction that decouples DeepSpeed's training and inference engines from specific hardware backends. This layer enables DeepSpeed to run transparently across a wide range of accelerators -- NVIDIA GPUs, Intel Gaudi HPUs, Huawei Ascend NPUs, Intel XPUs, Cambricon MLUs, Tecorigin SDAA devices, and Apple Silicon MPS -- without requiring user code changes.

The abstraction layer follows a **strategy pattern**: each accelerator implements a well-defined interface (the `DeepSpeedAccelerator` abstract base class), and a runtime dispatcher (`RealAccelerator`) selects the appropriate implementation based on hardware detection and user configuration. This design ensures that all higher-level DeepSpeed components (ZeRO optimizers, pipeline parallelism, tensor parallelism, inference engines) can operate hardware-agnosticly.

### Key Design Goals

1. **Hardware Portability**: Write training code once, run on any supported accelerator
2. **Transparent Dispatch**: Automatic hardware detection without manual configuration
3. **Backend Flexibility**: Different communication backends (NCCL, HCCL, CCL, CNCL) mapped to the appropriate hardware
4. **Custom Op Compatibility**: Accelerator-aware op building and JIT compilation
5. **Minimal Overhead**: The abstraction layer adds negligible runtime overhead

---

## Architecture

### Directory Structure

```
deepspeed/accelerator/
    __init__.py                  # Public API exports
    abstract_accelerator.py      # DeepSpeedAccelerator ABC
    cuda_accelerator.py          # CudaAccelerator (NVIDIA GPUs)
    cpu_accelerator.py           # CpuAccelerator (CPU-only training)
    hpu_accelerator.py           # HpuAccelerator (Intel Gaudi)
    npu_accelerator.py           # NpuAccelerator (Huawei Ascend)
    xpu_accelerator.py           # XpuAccelerator (Intel GPU)
    mlu_accelerator.py           # MluAccelerator (Cambricon)
    sdaa_accelerator.py          # SdaaAccelerator (Tecorigin)
    mps_accelerator.py           # MpsAccelerator (Apple Silicon)
    real_accelerator.py          # RealAccelerator (auto-detection + dispatch)
```

### Class Hierarchy

```
DeepSpeedAccelerator (ABC)          # abstract_accelerator.py
    |
    +-- CudaAccelerator             # cuda_accelerator.py
    +-- CpuAccelerator              # cpu_accelerator.py
    +-- HpuAccelerator              # hpu_accelerator.py
    +-- NpuAccelerator              # npu_accelerator.py
    +-- XpuAccelerator              # xpu_accelerator.py
    +-- MluAccelerator              # mlu_accelerator.py
    +-- SdaaAccelerator             # sdaa_accelerator.py
    +-- MpsAccelerator              # mps_accelerator.py

RealAccelerator                     # real_accelerator.py (dispatch wrapper)
```

### Accelerator Selection Flow

```
User Code
    |
    v
deepspeed.initialize() / init_inference()
    |
    v
get_accelerator()                     # Returns singleton RealAccelerator
    |
    v
RealAccelerator.__init__()
    |
    +-- Check ACCELERATOR environment variable
    |       |
    |       +-- "CUDA"  --> CudaAccelerator()
    |       +-- "CPU"   --> CpuAccelerator()
    |       +-- "HPU"   --> HpuAccelerator()
    |       +-- "NPU"   --> NpuAccelerator()
    |       +-- "XPU"   --> XpuAccelerator()
    |       +-- "MLU"   --> MluAccelerator()
    |       +-- "SDAA"  --> SdaaAccelerator()
    |       +-- "MPS"   --> MpsAccelerator()
    |
    +-- Auto-detect (no env var):
            |
            +-- torch.cuda.is_available()   --> CudaAccelerator
            +-- torch.hpu.is_available()    --> HpuAccelerator
            +-- torch.npu.is_available()    --> NpuAccelerator
            +-- hasattr(torch, 'xpu')       --> XpuAccelerator
            +-- torch.mlu.is_available()    --> MluAccelerator
            +-- hasattr(torch, 'sdaa')      --> SdaaAccelerator
            +-- torch.backends.mps.is_available() --> MpsAccelerator
            +-- fallback                    --> CpuAccelerator
```

---

## DeepSpeedAccelerator Abstract Base Class

The `DeepSpeedAccelerator` class in `deepspeed/accelerator/abstract_accelerator.py` defines the contract that all accelerator implementations must satisfy. It uses Python's `abc.ABC` and `abc.abstractmethod` to enforce implementation.

### Class Definition

```python
# deepspeed/accelerator/abstract_accelerator.py
from abc import ABC, abstractmethod

class DeepSpeedAccelerator(ABC):
    """Abstract base class for all DeepSpeed accelerator backends.
    
    Every hardware accelerator supported by DeepSpeed must implement
    this interface. The methods cover device management, random state,
    synchronization, communication, dataloader creation, and op building.
    """
    
    @abstractmethod
    def __init__(self):
        self._name = None
        self._communication_backend = None
```

### Abstract Methods Reference

| Method | Signature | Description |
|--------|-----------|-------------|
| `device_name` | `device_name() -> str` | Returns the accelerator name string (e.g., `"cuda"`, `"hpu"`) |
| `device` | `device(tensor=None) -> torch.device` | Returns the current `torch.device`; if tensor provided, returns tensor's device |
| `communication_backend_name` | `communication_backend_name() -> str` | Returns the collective communication library name (e.g., `"nccl"`, `"hccl"`) |
| `set_seed` | `set_seed(seed: int) -> None` | Sets random seed for all relevant libraries (torch, torch.cuda, numpy, random) |
| `synchronize` | `synchronize() -> None` | Waits for all operations on the accelerator to complete |
| `op_builder` | `op_builder() -> str` | Returns the name of the OpBuilder class for this accelerator |
| `default_dataloader` | `default_dataloader() -> str` | Returns the default dataloader class name |
| `is_gradient_accumulation_boundary` | `is_gradient_accumulation_boundary() -> bool` | Checks if the current micro-step is at a gradient accumulation boundary |
| `recommended_device_name` | `recommended_device_name() -> str` | Returns the human-readable device name for logging (e.g., `"GPU"`, `"HPU"`) |

### Detailed Method Specifications

#### `device_name()`

Returns the canonical lowercase string identifier for the accelerator. This string is used throughout DeepSpeed for:
- Configuration dispatch
- Op builder selection
- Device-specific code paths

```python
@abstractmethod
def device_name(self) -> str:
    """Return the canonical accelerator name.
    
    Returns:
        str: One of "cuda", "cpu", "hpu", "npu", "xpu", "mlu", "sdaa", "mps"
    """
    pass
```

**Implementation examples by accelerator:**

| Accelerator | Return Value |
|-------------|-------------|
| CudaAccelerator | `"cuda"` |
| CpuAccelerator | `"cpu"` |
| HpuAccelerator | `"hpu"` |
| NpuAccelerator | `"npu"` |
| XpuAccelerator | `"xpu"` |
| MluAccelerator | `"mlu"` |
| SdaaAccelerator | `"sdaa"` |
| MpsAccelerator | `"mps"` |

#### `device()`

Returns the `torch.device` for the accelerator. When called with a tensor argument, returns that tensor's device. When called without arguments, returns the default device for this accelerator type.

```python
@abstractmethod
def device(self, tensor=None) -> torch.device:
    """Return torch.device for this accelerator.
    
    Args:
        tensor (torch.Tensor, optional): If provided, return this tensor's device.
    
    Returns:
        torch.device: The device object for this accelerator.
    """
    pass

# Usage examples:
accelerator = get_accelerator()
device = accelerator.device()           # e.g., torch.device("cuda:0")
device = accelerator.device(tensor)     # e.g., torch.device("cuda:3")
```

#### `communication_backend_name()`

Returns the name of the preferred collective communication library backend for `torch.distributed`. This value is used during process group initialization.

```python
@abstractmethod
def communication_backend_name(self) -> str:
    """Return the communication backend name for torch.distributed.
    
    Returns:
        str: Backend name compatible with torch.distributed.init_process_group()
    """
    pass
```

**Backend mapping:**

| Accelerator | Communication Library | Backend Name |
|-------------|----------------------|-------------|
| CudaAccelerator | NVIDIA NCCL | `"nccl"` |
| CpuAccelerator | Gloo | `"gloo"` |
| HpuAccelerator | Intel HCCL | `"hccl"` |
| NpuAccelerator | Huawei HCCL | `"hccl"` |
| XpuAccelerator | Intel oneCCL | `"ccl"` |
| MluAccelerator | Cambricon CNCL | `"cncl"` |
| SdaaAccelerator | Tecorigin custom | `"cncl"` |
| MpsAccelerator | Gloo (no MPS backend) | `"gloo"` |

#### `set_seed()`

Sets the random seed across all relevant random number generators to ensure reproducibility.

```python
@abstractmethod
def set_seed(self, seed: int) -> None:
    """Set random seed for reproducibility.
    
    Sets seed for: Python random, NumPy, PyTorch CPU, PyTorch accelerator
    
    Args:
        seed (int): The random seed value.
    """
    pass

# Implementation pattern (CudaAccelerator):
def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

#### `synchronize()`

Blocks until all asynchronous operations on the accelerator complete. Critical for:
- Timing benchmarks
- Ensuring allreduce completion before next step
- Debugging race conditions

```python
@abstractmethod
def synchronize(self) -> None:
    """Wait for all pending operations to complete.
    
    Equivalent to torch.cuda.synchronize() for CUDA.
    Must not be called on unsupported backends.
    """
    pass

# Usage:
accelerator.synchronize()
start_time = time.time()
# ... training step ...
accelerator.synchronize()
end_time = time.time()
```

#### `op_builder()`

Returns the name of the `OpBuilder` subclass responsible for JIT-compiling or loading custom CUDA/accelerator kernels for this hardware.

```python
@abstractmethod
def op_builder(self) -> str:
    """Return the OpBuilder class name for custom ops.
    
    Returns:
        str: Name of the builder class (e.g., "CUDAOpBuilder", "CPUOpBuilder")
    """
    pass
```

#### `default_dataloader()`

Returns the default dataloader class to use for training. Some accelerators provide optimized data-loading pipelines.

```python
@abstractmethod
def default_dataloader(self, dataloader=None):
    """Return or wrap the dataloader for this accelerator.
    
    Args:
        dataloader: Optional existing dataloader to wrap.
    
    Returns:
        A dataloader compatible with this accelerator.
    """
    pass
```

#### `is_gradient_accumulation_boundary()`

Determines whether the current micro-step is at a gradient accumulation boundary. This is accelerator-specific because some hardware platforms handle micro-batching differently.

```python
@abstractmethod
def is_gradient_accumulation_boundary(self) -> bool:
    """Check if current step is a gradient accumulation boundary.
    
    Returns:
        bool: True if gradients should be synchronized and optimizer step taken.
    """
    pass
```

#### `recommended_device_name()`

Returns a human-readable name for display and logging purposes.

```python
@abstractmethod
def recommended_device_name(self) -> str:
    """Return human-readable device name for logging.
    
    Returns:
        str: e.g., "GPU", "CPU", "HPU", "NPU"
    """
    pass
```

### Concrete Methods (Non-Abstract)

The base class also provides some default implementations that concrete accelerators may override:

```python
def on_accelerator(self, tensor):
    """Check if tensor is on this accelerator.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
    
    Returns:
        bool: True if tensor.device matches this accelerator type.
    """
    device = self.device()
    return tensor.device.type == device.type

def resolve_data_device(self):
    """Return the device where input data should be placed.
    
    Returns:
        torch.device: Target device for input tensors.
    """
    return self.device()

def is_sanity_check_enabled(self):
    """Whether to run sanity checks during initialization.
    
    Returns:
        bool: True by default. Override to disable on platforms where
              sanity checks are expensive or unavailable.
    """
    return True
```

---

## CudaAccelerator (NVIDIA)

The `CudaAccelerator` in `deepspeed/accelerator/cuda_accelerator.py` is the most mature and feature-complete accelerator implementation, as NVIDIA CUDA is DeepSpeed's primary target platform.

### Class Definition

```python
# deepspeed/accelerator/cuda_accelerator.py
class CudaAccelerator(DeepSpeedAccelerator):
    """NVIDIA CUDA GPU accelerator backend.
    
    Supports all CUDA-capable GPUs (Volta, Turing, Ampere, Hopper, Blackwell).
    Uses NCCL for collective communication.
    """
    
    def __init__(self):
        self._name = "cuda"
        self._communication_backend = "nccl"
```

### CUDA-Specific Behavior

#### Device Management

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("cuda", torch.cuda.current_device())

def current_device(self):
    return torch.cuda.current_device()

def set_device(self, device_index):
    torch.cuda.set_device(device_index)
```

#### Seed Setting

```python
def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CuDNN operations (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
```

#### Synchronization

```python
def synchronize(self):
    torch.cuda.synchronize()
```

#### Memory Management

```python
# Additional CUDA-specific methods (not in ABC but available):
def empty_cache(self):
    """Release cached memory blocks back to the OS."""
    torch.cuda.empty_cache()

def memory_allocated(self, device=None):
    """Return current GPU memory occupied by tensors in bytes."""
    return torch.cuda.memory_allocated(device)

def max_memory_allocated(self, device=None):
    """Return maximum GPU memory occupied by tensors in bytes."""
    return torch.cuda.max_memory_allocated(device)

def memory_reserved(self, device=None):
    """Return current GPU memory managed by caching allocator in bytes."""
    return torch.cuda.memory_reserved(device)
```

#### Op Builder

```python
def op_builder(self):
    return "CUDAOpBuilder"
```

The CUDA op builder enables JIT compilation of custom CUDA kernels via `torch.utils.cpp_extension`. This includes fused optimizers, transformer kernels, sparse attention, quantization kernels, and more.

#### Default Dataloader

```python
def default_dataloader(self, dataloader=None):
    if dataloader is not None:
        return dataloader
    return torch.utils.data.DataLoader
```

#### NCCL Backend

```python
def communication_backend_name(self):
    return "nccl"
```

NCCL (NVIDIA Collective Communications Library) provides optimized implementations of:
- `all_reduce`: Gradient synchronization in data parallelism
- `all_gather`: Parameter gathering in ZeRO Stage 3
- `reduce_scatter`: Gradient partitioning in ZeRO Stage 2
- `broadcast`: Model parameter broadcasting
- `send/recv`: Pipeline parallelism point-to-point

### CUDA Architecture Support

| Architecture | Compute Capability | GPU Examples |
|-------------|-------------------|--------------|
| Volta | 7.0, 7.2 | V100, Titan V |
| Turing | 7.5 | RTX 2080, T4 |
| Ampere | 8.0, 8.6 | A100, RTX 3090 |
| Hopper | 9.0 | H100 |
| Blackwell | 10.0 | B200 |

### CUDA-Specific Optimizations

1. **Fused Adam/Lamb/Lion Kernels**: Multi-operator fusion reduces kernel launch overhead and memory traffic
2. **Flash Attention**: Memory-efficient attention using tiled computation
3. **Tensor Core Utilization**: Automatic mixed-precision (FP16/BF16) leverages tensor cores
4. **NCCL Tuning**: Auto-tuning of NCCL channel count, algorithm selection, and protocol
5. **CUDA Graphs**: Capture and replay GPU kernel sequences for reduced launch overhead

---

## CpuAccelerator

The `CpuAccelerator` in `deepspeed/accelerator/cpu_accelerator.py` enables CPU-only training, which is useful for development, testing, and environments without GPU access.

### Class Definition

```python
# deepspeed/accelerator/cpu_accelerator.py
class CpuAccelerator(DeepSpeedAccelerator):
    """CPU-only accelerator backend.
    
    Used for development, testing, and CPU-based training (e.g., with
    large-memory servers or Intel oneAPI optimizations).
    """
    
    def __init__(self):
        self._name = "cpu"
        self._communication_backend = "gloo"
```

### CPU-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("cpu")

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def synchronize(self):
    # CPU operations are synchronous by default
    pass

def op_builder(self):
    return "CPUOpBuilder"

def communication_backend_name(self):
    return "gloo"

def recommended_device_name(self):
    return "CPU"
```

### CPU Training Support

The CPU accelerator supports:
- **ZeRO Stage 1, 2, 3**: Full ZeRO optimization on CPU
- **CPU Adam/Adagrad**: Optimized CPU implementations via `cpu_adam` and `cpu_adagrad` ops
- **DeepSpeed CPU Training**: Using `cpu_adam` with `zero_optimization` for models that fit in CPU RAM
- **NVMe Offload**: Using NVMe as extension of CPU memory via ZeRO-Infinity

### CPU Optimizer Configuration

```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

---

## HpuAccelerator (Intel Gaudi)

The `HpuAccelerator` in `deepspeed/accelerator/hpu_accelerator.py` provides support for Intel Gaudi AI accelerators (Gaudi2, Gaudi3). These accelerators use HCCL (Habana Collective Communications Library) for inter-device communication.

### Class Definition

```python
# deepspeed/accelerator/hpu_accelerator.py
class HpuAccelerator(DeepSpeedAccelerator):
    """Intel Gaudi HPU accelerator backend.
    
    Supports Intel Gaudi2 and Gaudi3 AI accelerators.
    Uses HCCL for collective communication.
    Requires the habana_framework package.
    """
    
    def __init__(self):
        self._name = "hpu"
        self._communication_backend = "hccl"
```

### HPU-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    import habana_frameworks.torch.core as htcore
    return torch.device("hpu", torch.hpu.current_device())

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.hpu.manual_seed(seed)
    torch.hpu.manual_seed_all(seed)

def synchronize(self):
    torch.hpu.synchronize()

def op_builder(self):
    return "HPUOpBuilder"

def communication_backend_name(self):
    return "hccl"

def recommended_device_name(self):
    return "HPU"
```

### Gaudi-Specific Optimizations

1. **Habana HabanaOperators**: Deepspeed can leverage Habana-specific operators
2. **HCCL Tuning**: Optimized collective communication for Gaudi's interconnect
3. **Mixed Precision**: BF16 is the primary precision on Gaudi (native hardware support)
4. **Recipe-Based Compilation**: Gaudi uses "recipes" for graph compilation

### HPU Configuration Example

```json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### HPU Launch Command

```bash
# Using DeepSpeed launcher with Habana
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json

# Using Habana's torchrun wrapper
torchrun --nproc_per_node=8 train.py --deepspeed ds_config.json
```

### HPU Environment Setup

```bash
# Set Habana environment
export HABANA_LOGS=/tmp/habana_logs
export ENABLE_EXPERIMENTAL_FLAGS=true
export PT_HPU_LAZY_MODE=1  # Lazy mode for graph compilation

# Load Habana module (if using module system)
module load habana
```

---

## NpuAccelerator (Huawei Ascend)

The `NpuAccelerator` in `deepspeed/accelerator/npu_accelerator.py` provides support for Huawei Ascend NPUs (910B, 310P). These accelerators use Huawei's HCCL (Hierarchical Collective Communication Library).

### Class Definition

```python
# deepspeed/accelerator/npu_accelerator.py
class NpuAccelerator(DeepSpeedAccelerator):
    """Huawei Ascend NPU accelerator backend.
    
    Supports Huawei Ascend 910B and 310P NPUs.
    Uses HCCL for collective communication.
    Requires the torch_npu package.
    """
    
    def __init__(self):
        self._name = "npu"
        self._communication_backend = "hccl"
```

### NPU-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("npu", torch.npu.current_device())

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)

def synchronize(self):
    torch.npu.synchronize()

def op_builder(self):
    return "NPUOpBuilder"

def communication_backend_name(self):
    return "hccl"

def recommended_device_name(self):
    return "NPU"
```

### Ascend-Specific Optimizations

1. **Ascend C Kernels**: Custom operators written in Ascend C programming language
2. **HCCL Topology-Aware**: Optimized communication based on NPU topology (Ring, Mesh)
3. **Mixed Precision**: FP16 and BF16 support with Ascend Matrix Unit (Cube Unit)
4. **Graph Mode**: Support for static graph compilation via `torch_npu.npu.Graph`

### NPU Configuration Example

```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "npu"
        }
    }
}
```

### NPU Launch Command

```bash
# Set NPU environment
export ASCEND_RT_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH

# Launch with DeepSpeed
torchrun --nproc_per_node=8 train.py --deepspeed ds_config.json
```

---

## XpuAccelerator (Intel GPU)

The `XpuAccelerator` in `deepspeed/accelerator/xpu_accelerator.py` supports Intel discrete GPUs (Intel Data Center GPU Max Series, Arc GPUs) and uses Intel oneCCL for collective communication.

### Class Definition

```python
# deepspeed/accelerator/xpu_accelerator.py
class XpuAccelerator(DeepSpeedAccelerator):
    """Intel XPU GPU accelerator backend.
    
    Supports Intel Data Center GPU Max (Ponte Vecchio), Arc GPUs.
    Uses oneCCL for collective communication.
    Requires intel_extension_for_pytorch package.
    """
    
    def __init__(self):
        self._name = "xpu"
        self._communication_backend = "ccl"
```

### XPU-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("xpu", torch.xpu.current_device())

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.xpu.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)

def synchronize(self):
    torch.xpu.synchronize()

def op_builder(self):
    return "XPUOpBuilder"

def communication_backend_name(self):
    return "ccl"

def recommended_device_name(self):
    return "XPU"
```

### XPU Configuration Example

```json
{
    "train_batch_size": 32,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4
        }
    }
}
```

### XPU Environment Setup

```bash
# Load oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Set oneCCL environment
export CCL_WORKER_COUNT=4
export CCL_LOG_LEVEL=info

# Launch training
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

---

## MluAccelerator (Cambricon)

The `MluAccelerator` in `deepspeed/accelerator/mlu_accelerator.py` supports Cambricon MLU accelerators (MLU370, MLU290) using CNCL (Cambricon Neuro-Computing Communication Library).

### Class Definition

```python
# deepspeed/accelerator/mlu_accelerator.py
class MluAccelerator(DeepSpeedAccelerator):
    """Cambricon MLU accelerator backend.
    
    Supports Cambricon MLU370, MLU290 accelerators.
    Uses CNCL for collective communication.
    Requires torch_mlu package (Cambricon PyTorch extension).
    """
    
    def __init__(self):
        self._name = "mlu"
        self._communication_backend = "cncl"
```

### MLU-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("mlu", torch.mlu.current_device())

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    torch.mlu.manual_seed_all(seed)

def synchronize(self):
    torch.mlu.synchronize()

def op_builder(self):
    return "MLUOpBuilder"

def communication_backend_name(self):
    return "cncl"

def recommended_device_name(self):
    return "MLU"
```

### MLU Configuration Example

```json
{
    "train_batch_size": 16,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

---

## SdaaAccelerator (Tecorigin)

The `SdaaAccelerator` in `deepspeed/accelerator/sdaa_accelerator.py` supports Tecorigin SDAA (Software-Defined AI Accelerator) devices, commonly used in Chinese domestic AI compute deployments.

### Class Definition

```python
# deepspeed/accelerator/sdaa_accelerator.py
class SdaaAccelerator(DeepSpeedAccelerator):
    """Tecorigin SDAA accelerator backend.
    
    Supports Tecorigin SDAA devices.
    Uses custom communication backend.
    Requires sdaa framework package.
    """
    
    def __init__(self):
        self._name = "sdaa"
        self._communication_backend = "cncl"
```

### SDAA-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("sdaa", torch.sdaa.current_device())

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.sdaa.manual_seed(seed)
    torch.sdaa.manual_seed_all(seed)

def synchronize(self):
    torch.sdaa.synchronize()

def op_builder(self):
    return "SDAAOpBuilder"

def communication_backend_name(self):
    return "cncl"

def recommended_device_name(self):
    return "SDAA"
```

---

## MpsAccelerator (Apple Silicon)

The `MpsAccelerator` in `deepspeed/accelerator/mps_accelerator.py` supports Apple Silicon GPUs via Metal Performance Shaders (MPS). This enables DeepSpeed on Apple M1/M2/M3/M4 Mac hardware.

### Class Definition

```python
# deepspeed/accelerator/mps_accelerator.py
class MpsAccelerator(DeepSpeedAccelerator):
    """Apple Silicon MPS accelerator backend.
    
    Supports Apple M1/M2/M3/M4 GPUs via Metal Performance Shaders.
    Uses Gloo for communication (no MPS-specific backend).
    Limited multi-device support (single GPU per Mac).
    """
    
    def __init__(self):
        self._name = "mps"
        self._communication_backend = "gloo"
```

### MPS-Specific Behavior

```python
def device(self, tensor=None):
    if tensor is not None:
        return tensor.device
    return torch.device("mps")

def set_seed(self, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS shares the same manual_seed as CPU
    torch.mps.manual_seed(seed)

def synchronize(self):
    torch.mps.synchronize()

def op_builder(self):
    return "MPSOpBuilder"

def communication_backend_name(self):
    return "gloo"

def recommended_device_name(self):
    return "MPS"
```

### MPS Limitations

1. **Single Device**: Apple Silicon Macs have a single unified GPU; no multi-GPU data parallelism
2. **No Custom CUDA Kernels**: MPS does not support CUDA; custom ops fall back to CPU
3. **Limited Op Coverage**: Not all PyTorch operations are accelerated on MPS
4. **Memory Sharing**: Unified memory architecture means CPU and GPU share the same memory pool
5. **Communication**: Only Gloo backend available (no NCCL-equivalent for MPS)

### MPS Configuration Example

```json
{
    "train_batch_size": 8,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

### MPS Usage

```python
# Device placement for Apple Silicon
import deepspeed
import torch

# MPS auto-detection
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)
```

---

## RealAccelerator: Auto-Detection and Dispatch

The `RealAccelerator` in `deepspeed/accelerator/real_accelerator.py` is the runtime dispatcher that automatically detects available hardware and instantiates the correct accelerator. It implements the same `DeepSpeedAccelerator` interface via delegation (composition over inheritance).

### Class Definition

```python
# deepspeed/accelerator/real_accelerator.py
class RealAccelerator(DeepSpeedAccelerator):
    """Auto-detecting accelerator dispatcher.
    
    On first instantiation, detects available hardware and creates
    the appropriate concrete accelerator. All method calls are delegated
    to the underlying accelerator instance.
    """
    
    def __init__(self):
        self._accelerator = self._detect_and_create()
    
    def _detect_and_create(self):
        """Detect hardware and return the appropriate accelerator.
        
        Detection priority:
        1. ACCELERATOR environment variable (explicit override)
        2. Auto-detect via torch.xxx.is_available() checks
        3. Fallback to CPU
        """
        ...
```

### Detection Logic

```python
def _detect_and_create(self):
    import os
    
    # 1. Check explicit environment variable
    accel_env = os.environ.get("ACCELERATOR", "").upper()
    
    if accel_env:
        return self._create_by_name(accel_env)
    
    # 2. Auto-detect in priority order
    detection_order = [
        ("cuda", lambda: torch.cuda.is_available()),
        ("hpu",  lambda: hasattr(torch, 'hpu') and torch.hpu.is_available()),
        ("npu",  lambda: hasattr(torch, 'npu') and torch.npu.is_available()),
        ("xpu",  lambda: hasattr(torch, 'xpu') and torch.xpu.is_available()),
        ("mlu",  lambda: hasattr(torch, 'mlu') and torch.mlu.is_available()),
        ("sdaa", lambda: hasattr(torch, 'sdaa') and torch.sdaa.is_available()),
        ("mps",  lambda: hasattr(torch.backends, 'mps') 
                        and torch.backends.mps.is_available()),
    ]
    
    for name, check_fn in detection_order:
        try:
            if check_fn():
                return self._create_by_name(name)
        except Exception:
            continue
    
    # 3. Fallback to CPU
    return CpuAccelerator()

def _create_by_name(self, name):
    """Instantiate accelerator by canonical name."""
    creators = {
        "CUDA": CudaAccelerator,
        "CPU":  CpuAccelerator,
        "HPU":  HpuAccelerator,
        "NPU":  NpuAccelerator,
        "XPU":  XpuAccelerator,
        "MLU":  MluAccelerator,
        "SDAA": SdaaAccelerator,
        "MPS":  MpsAccelerator,
    }
    creator = creators.get(name)
    if creator is None:
        raise ValueError(
            f"Unknown accelerator '{name}'. "
            f"Supported: {list(creators.keys())}"
        )
    return creator()
```

### Method Delegation

```python
# All DeepSpeedAccelerator methods delegate to the underlying instance:
def device_name(self):
    return self._accelerator.device_name()

def device(self, tensor=None):
    return self._accelerator.device(tensor)

def communication_backend_name(self):
    return self._accelerator.communication_backend_name()

def set_seed(self, seed):
    self._accelerator.set_seed(seed)

def synchronize(self):
    self._accelerator.synchronize()

def op_builder(self):
    return self._accelerator.op_builder()

def default_dataloader(self, dataloader=None):
    return self._accelerator.default_dataloader(dataloader)

def is_gradient_accumulation_boundary(self):
    return self._accelerator.is_gradient_accumulation_boundary()

def recommended_device_name(self):
    return self._accelerator.recommended_device_name()

# Passthrough for accelerator-specific methods:
def __getattr__(self, name):
    """Delegate unknown attribute access to underlying accelerator."""
    return getattr(self._accelerator, name)
```

### Singleton Access

```python
# deepspeed/accelerator/__init__.py
_accelerator_instance = None

def get_accelerator():
    """Return the global accelerator singleton.
    
    Creates the RealAccelerator on first call, then returns the
    cached instance on subsequent calls.
    
    Returns:
        RealAccelerator: The auto-detected accelerator instance.
    """
    global _accelerator_instance
    if _accelerator_instance is None:
        _accelerator_instance = RealAccelerator()
    return _accelerator_instance

def set_accelerator(accelerator):
    """Override the global accelerator (for testing or custom hardware).
    
    Args:
        accelerator (DeepSpeedAccelerator): The accelerator to use.
    """
    global _accelerator_instance
    _accelerator_instance = accelerator
```

---

## Adding a Custom Accelerator

To add support for a new hardware accelerator, follow these steps:

### Step 1: Create the Accelerator Class

Create a new file `deepspeed/accelerator/custom_accelerator.py`:

```python
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator

class CustomAccelerator(DeepSpeedAccelerator):
    """Custom accelerator implementation.
    
    Replace all method implementations with hardware-specific logic.
    """
    
    def __init__(self):
        self._name = "custom"
        self._communication_backend = "custom_backend"
    
    def device_name(self):
        return "custom"
    
    def device(self, tensor=None):
        if tensor is not None:
            return tensor.device
        return torch.device("custom", 0)
    
    def communication_backend_name(self):
        return "custom_backend"
    
    def set_seed(self, seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Set hardware-specific seed
        # torch.custom.manual_seed(seed)
        # torch.custom.manual_seed_all(seed)
    
    def synchronize(self):
        # Hardware-specific synchronization
        # torch.custom.synchronize()
        pass
    
    def op_builder(self):
        return "CustomOpBuilder"
    
    def default_dataloader(self, dataloader=None):
        if dataloader is not None:
            return dataloader
        return torch.utils.data.DataLoader
    
    def is_gradient_accumulation_boundary(self):
        return True
    
    def recommended_device_name(self):
        return "CUSTOM"
```

### Step 2: Register in RealAccelerator

Add the new accelerator to the detection and creation logic in `real_accelerator.py`:

```python
# In _detect_and_create():
detection_order = [
    # ... existing entries ...
    ("custom", lambda: hasattr(torch, 'custom') and torch.custom.is_available()),
]

# In _create_by_name():
creators = {
    # ... existing entries ...
    "CUSTOM": CustomAccelerator,
}
```

### Step 3: Create the Op Builder

Create a custom op builder (if needed) in `op_builder/`:

```python
# deepspeed/op_builder/custom.py
from deepspeed.op_builder.builder import OpBuilder

class CustomOpBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_CUSTOM_OPS"
    NAME = "custom_ops"
    
    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)
    
    def absolute_name(self):
        return f"deepspeed.ops.custom.{self.NAME}"
    
    def sources(self):
        return []
    
    def extra_ldflags(self):
        return []
```

### Step 4: Configuration and Environment

Add environment variable support for explicit selection:

```bash
# Force custom accelerator
export ACCELERATOR=CUSTOM

# Then run DeepSpeed normally
deepspeed train.py --deepspeed ds_config.json
```

### Step 5: Testing

Create comprehensive tests:

```python
# tests/unit/test_custom_accelerator.py
import pytest
import torch
from deepspeed.accelerator.custom_accelerator import CustomAccelerator

class TestCustomAccelerator:
    def setup_method(self):
        self.accel = CustomAccelerator()
    
    def test_device_name(self):
        assert self.accel.device_name() == "custom"
    
    def test_device(self):
        device = self.accel.device()
        assert device.type == "custom"
    
    def test_communication_backend(self):
        assert self.accel.communication_backend_name() == "custom_backend"
    
    def test_set_seed(self):
        self.accel.set_seed(42)
        a = torch.randn(10)
        self.accel.set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)
    
    def test_recommended_name(self):
        assert self.accel.recommended_device_name() == "CUSTOM"
```

---

## Environment Variables

### Accelerator Selection

| Variable | Values | Description |
|----------|--------|-------------|
| `ACCELERATOR` | `CUDA`, `CPU`, `HPU`, `NPU`, `XPU`, `MLU`, `SDAA`, `MPS` | Force a specific accelerator backend. Overrides auto-detection. |
| `DS_BUILD_OPS` | `0`, `1` | Globally enable/disable custom op building (default: `1`) |

### CUDA-Specific

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Restrict visible CUDA devices (e.g., `"0,1,2,3"`) |
| `TORCH_CUDA_ARCH_LIST` | Target CUDA architectures for compilation (e.g., `"8.0;8.6;9.0"`) |
| `NCCL_DEBUG` | NCCL debug level (`INFO`, `WARN`, `TRACE`) |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL communication |
| `NCCL_IB_DISABLE` | Disable InfiniBand (`0` or `1`) |

### HPU-Specific

| Variable | Description |
|----------|-------------|
| `HABANA_LOGS` | Path for Habana log files |
| `PT_HPU_LAZY_MODE` | Enable lazy execution mode (`0` or `1`) |
| `ENABLE_EXPERIMENTAL_FLAGS` | Enable experimental Habana features |
| `HABANA_VISIBLE_MODULES` | Restrict visible Gaudi devices |

### NPU-Specific

| Variable | Description |
|----------|-------------|
| `ASCEND_RT_AICPU_PATH` | Path to Ascend toolkit |
| `ASCEND_DEVICE_ID` | Default NPU device ID |
| `HCCL_CONNECT_TIMEOUT` | HCCL connection timeout in seconds |

### XPU-Specific

| Variable | Description |
|----------|-------------|
| `ZE_AFFINITY_MASK` | Restrict visible XPU devices |
| `CCL_WORKER_COUNT` | Number of oneCCL worker threads |
| `CCL_LOG_LEVEL` | oneCCL log level |

### Build Control

| Variable | Default | Description |
|----------|---------|-------------|
| `DS_BUILD_OPS` | `1` | Build all ops |
| `DS_BUILD_FUSED_ADAM` | `1` | Build fused Adam kernel |
| `DS_BUILD_FUSED_LAMB` | `1` | Build fused Lamb kernel |
| `DS_BUILD_FUSED_LION` | `1` | Build fused Lion kernel |
| `DS_BUILD_CPU_ADAM` | `1` | Build CPU Adam kernel |
| `DS_BUILD_CPU_ADAGRAD` | `1` | Build CPU Adagrad kernel |
| `DS_BUILD_CPU_LION` | `1` | Build CPU Lion kernel |
| `DS_BUILD_TRANSFORMER` | `1` | Build transformer kernel |
| `DS_BUILD_TRANSFORMER_INFERENCE` | `1` | Build transformer inference kernel |
| `DS_BUILD_STOCHASTIC_TRANSFORMER` | `1` | Build stochastic transformer kernel |
| `DS_BUILD_SPARSE_ATTN` | `0` | Build sparse attention kernel |
| `DS_BUILD_RAGGED_OPS` | `1` | Build ragged inference ops |
| `DS_BUILD_QUANTIZER` | `1` | Build quantization ops |
| `DS_BUILD_FP_QUANTIZER` | `0` | Build FP quantizer ops |
| `DS_BUILD_AIO` | `1` | Build async I/O ops |
| `DS_BUILD_UTILS` | `1` | Build utility ops |
| `DS_BUILD_CUTLASS` | `0` | Build CUTLASS-based ops |

---

## Configuration Examples

### Multi-Accelerator Cluster Configuration

For heterogeneous clusters (e.g., mixing NVIDIA and Intel Gaudi), each node type needs its own configuration:

```json
// nvidia_node_config.json
{
    "train_batch_size": 64,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999]
        }
    }
}
```

```json
// gaudi_node_config.json
{
    "train_batch_size": 64,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999]
        }
    }
}
```

### Force CPU Accelerator

```bash
# Force CPU mode for testing
export ACCELERATOR=CPU
deepspeed --num_gpus=1 train.py --deepspeed ds_config_cpu.json
```

```json
// ds_config_cpu.json
{
    "train_batch_size": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu"
        },
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```

### Intel Gaudi Multi-Node Setup

```bash
# On each Gaudi node
export ACCELERATOR=HPU
export PT_HPU_LAZY_MODE=1
export HCCL_CONNECT_TIMEOUT=7200

# Launch with DeepSpeed
deepspeed --hostfile=hostfile --num_nodes=4 --num_gpus=8 \
    train.py --deepspeed ds_config_hpu.json
```

### Huawei Ascend NPU Setup

```bash
# On each Ascend node
export ACCELERATOR=NPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Launch with torchrun
torchrun --nproc_per_node=8 --nnodes=4 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    train.py --deepspeed ds_config_npu.json
```

### Programmatic Accelerator Override

```python
import deepspeed
from deepspeed.accelerator.cuda_accelerator import CudaAccelerator
from deepspeed.accelerator.cpu_accelerator import CpuAccelerator

# Override before calling deepspeed.initialize()
from deepspeed.accelerator import set_accelerator

# Force CPU for unit testing
set_accelerator(CpuAccelerator())

# ... rest of training code ...
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)
```

### Detecting Current Accelerator in Code

```python
from deepspeed.accelerator import get_accelerator

accel = get_accelerator()

print(f"Running on: {accel.recommended_device_name()}")
print(f"Device name: {accel.device_name()}")
print(f"Communication: {accel.communication_backend_name()}")
print(f"Op builder: {accel.op_builder()}")

# Move tensor to current accelerator device
device = accel.device()
tensor = torch.randn(10, 10).to(device)
print(f"Tensor device: {tensor.device}")
```

---

## Troubleshooting

### Accelerator Not Detected

**Symptom**: DeepSpeed falls back to CPU despite accelerator hardware being present.

**Solutions**:
1. Set `ACCELERATOR` environment variable explicitly:
   ```bash
   export ACCELERATOR=CUDA  # or HPU, NPU, etc.
   ```

2. Verify PyTorch can see the device:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

3. Check that the required framework is installed:
   - HPU: `pip install habana_frameworks`
   - NPU: `pip install torch_npu`
   - XPU: `pip install intel_extension_for_pytorch`
   - MLU: `pip install torch_mlu`

### NCCL Errors on Non-NVIDIA Hardware

**Symptom**: `NCCL error` or `Backend NCCL not available` on non-NVIDIA hardware.

**Solution**: The accelerator abstraction should handle backend selection automatically. If it doesn't:
```bash
export ACCELERATOR=HPU  # or appropriate accelerator
```

### Custom Op Build Failures

**Symptom**: `RuntimeError: CUDA Op is not available` or similar.

**Solutions**:
1. Set the correct CUDA architecture:
   ```bash
   export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
   ```

2. Force op rebuild:
   ```bash
   DS_BUILD_OPS=1 pip install -e . --global-option="build_ext" --global-option="--force"
   ```

3. Disable specific ops that fail:
   ```bash
   export DS_BUILD_SPARSE_ATTN=0
   export DS_BUILD_CUTLASS=0
   ```

### Memory Errors on Accelerators with Unified Memory

**Symptom**: OOM errors on Apple Silicon or other unified-memory systems.

**Solution**: Reduce batch size and use ZeRO Stage 2 or 3:
```json
{
    "train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 2
    }
}
```

### Communication Backend Mismatch

**Symptom**: `RuntimeError: ProcessGroupNCCL is not supported on device type: xpu`

**Solution**: Ensure the correct backend is selected for your accelerator:
```python
from deepspeed.accelerator import get_accelerator
accel = get_accelerator()
print(f"Expected backend: {accel.communication_backend_name()}")
```

If the backend is wrong, set `ACCELERATOR` explicitly or initialize `torch.distributed` with the correct backend before calling `deepspeed.initialize()`.

---

## Accelerator Comparison Matrix

| Feature | CUDA | CPU | HPU | NPU | XPU | MLU | SDAA | MPS |
|---------|------|-----|-----|-----|-----|-----|------|-----|
| **ZeRO Stage 1** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **ZeRO Stage 2** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| **ZeRO Stage 3** | Yes | Yes | Yes | Yes | Yes | Partial | Partial | Yes |
| **Pipeline Parallelism** | Yes | No | Partial | Partial | No | No | No | No |
| **Tensor Parallelism** | Yes | No | Partial | Partial | Partial | No | No | No |
| **MoE** | Yes | Partial | Partial | Partial | Partial | No | No | No |
| **Inference V1** | Yes | No | Partial | No | No | No | No | No |
| **Inference V2** | Yes | No | No | No | No | No | No | No |
| **Fused Adam** | Yes | Yes (CPU) | No | No | No | No | No | No |
| **Fused Lamb** | Yes | No | No | No | No | No | No | No |
| **Sparse Attention** | Yes | No | No | No | No | No | No | No |
| **FP16** | Yes | No | No | Yes | No | Yes | Yes | No |
| **BF16** | Yes | No | Yes | Yes | Yes | No | No | No |
| **FP8** | Yes (Hopper+) | No | No | No | No | No | No | No |
| **NVMe Offload** | Yes | Yes | No | No | No | No | No | No |
| **Multi-Node** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| **Custom Kernels** | Extensive | CPU ops | Limited | Limited | Limited | No | No | No |
| **Communication** | NCCL | Gloo | HCCL | HCCL | CCL | CNCL | CNCL | Gloo |
| **Maturity** | Production | Production | Production | Beta | Beta | Alpha | Alpha | Alpha |

---

## API Quick Reference

```python
from deepspeed.accelerator import get_accelerator, set_accelerator

# Get the current accelerator
accel = get_accelerator()

# Core properties
accel.device_name()                   # "cuda", "hpu", etc.
accel.communication_backend_name()    # "nccl", "hccl", etc.
accel.recommended_device_name()       # "GPU", "HPU", etc.
accel.op_builder()                    # "CUDAOpBuilder", etc.

# Device operations
device = accel.device()               # torch.device("cuda:0")
device = accel.device(tensor)         # tensor's device
accel.set_seed(42)                    # Set all RNG seeds
accel.synchronize()                   # Wait for all ops

# Check if tensor is on accelerator
is_on_device = accel.on_accelerator(tensor)

# Override accelerator (for testing or custom hardware)
from deepspeed.accelerator.cpu_accelerator import CpuAccelerator
set_accelerator(CpuAccelerator())
```
