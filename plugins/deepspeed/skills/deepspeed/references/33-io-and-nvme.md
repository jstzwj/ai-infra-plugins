# IO and DeepNVMe

## Overview

DeepSpeed provides a comprehensive I/O subsystem designed to handle the extreme data throughput demands of large-scale distributed training. The I/O infrastructure encompasses three major components: asynchronous I/O (AIO) for non-blocking file operations, GPU Direct Storage (GDS) for direct GPU-to-storage data paths, and DeepNVMe for affordable I/O scaling using NVMe solid-state drives. These components are critical for efficiently managing model checkpointing (which can reach terabytes for trillion-parameter models), parameter offloading (where optimizer states and parameters are swapped between GPU and NVMe), and training data streaming.

The I/O system is particularly important for ZeRO-Infinity and ZeRO-Offload, which extend GPU memory to NVMe storage by partitioning optimizer states, gradients, and parameters across the NVMe devices attached to each node. Without high-throughput I/O, the overhead of moving data between GPU and NVMe would overwhelm the computational benefits of offloading. DeepSpeed's custom AIO and NVMe stack achieves near-hardware-limit throughput by using asynchronous operations, kernel-bypass techniques, and direct memory management.

---

## Source Code Organization

```
deepspeed/io/
    __init__.py
    io_optimizer.py                # I/O-optimized parameter server

deepspeed/nvme/
    __init__.py
    nvme.py                        # NVMe device management and operations
    nvme_io.py                     # NVMe I/O operations

csrc/aio/
    py_lib/                        # Python-accessible async I/O library
        deepspeed_aio.py           # Python bindings and interface
        deepspeed_aio_threaded.py  # Threaded async I/O implementation
    py_test/                       # Python tests for AIO
    common/                        # Common utilities for AIO
    utils/                         # AIO utility functions

csrc/gds/
    py_lib/                        # Python-accessible GDS library
        deepspeed_gds.py           # GDS Python bindings
    py_test/                       # Python tests for GDS

deepspeed/runtime/swap_tensor/
    __init__.py
    async_partitioned_param_swapper.py  # Async parameter swapping for ZeRO-Infinity
    swap_buffer.py                      # Swap buffer management
    swap_buffer_pool.py                 # Buffer pool for swap operations
```

---

## Asynchronous I/O (AIO)

### Overview

DeepSpeed's asynchronous I/O (AIO) library provides high-throughput, non-blocking file operations optimized for the large sequential reads and writes that dominate model checkpointing and parameter offloading. Unlike standard POSIX I/O (which blocks the calling thread), or even `libaio` (which has kernel-level limitations), DeepSpeed AIO uses a threaded async I/O model that submits I/O requests to a pool of worker threads, enabling overlap between computation and I/O.

### AIO C++/CUDA Source

```
csrc/aio/
    aio_common.h                  # Common AIO structures and definitions
    aio_handle.h                  # AIO handle for managing async operations
    aio_handle.cpp                # Handle implementation
    libaio_context.h              # Linux libaio context wrapper
    libaio_context.cpp            # Context implementation
    py_lib/
        deepspeed_aio.py          # High-level Python interface
        deepspeed_aio_threaded.py # Threaded async I/O implementation
    common/
        aio_context.h             # Abstract I/O context
    utils/
        boolean_list.h            # Utility for boolean lists
        siling_list.h             # Utility for sizing
```

### AIO Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | int | `1048576` (1 MB) | Block size in bytes for each individual I/O operation. Larger block sizes generally yield higher throughput for sequential access patterns. Must be a power of 2 and at least 4096. |
| `queue_depth` | int | `8` | Maximum number of outstanding (in-flight) I/O requests. Higher values allow more overlap between I/O submission and completion, up to device limits. |
| `thread_count` | int | `1` | Number of I/O worker threads. More threads can increase throughput for concurrent operations, but also increase CPU overhead. |
| `single_submit` | bool | `false` | When `true`, submit I/O requests one at a time rather than in batches. Reduces latency for small operations but may reduce throughput for large ones. |
| `overlap_events` | bool | `true` | When `true`, overlap I/O event processing with I/O submission. Enables pipelining of I/O operations for higher throughput. |

### AIO Configuration in DeepSpeed JSON

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

### AIO Python Interface

```python
# deepspeed/csrc/aio/py_lib/deepspeed_aio.py

class DeepSpeedAio:
    """High-level interface to DeepSpeed asynchronous I/O.

    Provides async read and write operations with configurable
    block size, queue depth, and threading.
    """

    def __init__(
        self,
        block_size=1048576,
        queue_depth=8,
        thread_count=1,
        single_submit=False,
        overlap_events=True,
    ):
        """Initialize the AIO engine.

        Args:
            block_size: Size of each I/O operation in bytes.
            queue_depth: Maximum concurrent I/O operations.
            thread_count: Number of I/O worker threads.
            single_submit: Submit one request at a time.
            overlap_events: Overlap event processing with submission.
        """
        ...

    def async_pread(self, buffer, fd, offset, length):
        """Asynchronous pread: read from file into buffer.

        Args:
            buffer: Target memory buffer (must be pinned for GPU).
            fd: File descriptor to read from.
            offset: Byte offset in the file.
            length: Number of bytes to read.

        Returns:
            Future-like object for checking completion.
        """
        ...

    def async_pwrite(self, buffer, fd, offset, length):
        """Asynchronous pwrite: write buffer to file.

        Args:
            buffer: Source memory buffer.
            fd: File descriptor to write to.
            offset: Byte offset in the file.
            length: Number of bytes to write.

        Returns:
            Future-like object for checking completion.
        """
        ...

    def wait(self):
        """Wait for all outstanding I/O operations to complete."""
        ...

    def synchronize(self):
        """Synchronize all pending I/O operations."""
        ...
```

### AIO Threaded Implementation

```python
# deepspeed/csrc/aio/py_lib/deepspeed_aio_threaded.py

class DeepSpeedAioThreaded:
    """Threaded implementation of async I/O.

    Uses a pool of worker threads to submit and complete I/O
    operations without blocking the main training thread.

    Architecture:
    - Main thread: Submits I/O requests to a queue
    - Worker threads: Dequeue requests, submit to OS, wait for completion
    - Completion queue: Signals completed operations back to main thread

    This approach avoids the limitations of Linux libaio (which requires
    files opened with O_DIRECT and aligned buffers) while still providing
    true asynchronous behavior.
    """

    def __init__(self, block_size, queue_depth, thread_count, single_submit, overlap_events):
        ...

    def read(self, buffer, file_path, offset, length):
        """Submit an async read operation."""
        ...

    def write(self, buffer, file_path, offset, length):
        """Submit an async write operation."""
        ...

    def wait_for_completion(self):
        """Block until all submitted operations complete."""
        ...
```

### AIO Performance Tuning

| Workload | `block_size` | `queue_depth` | `thread_count` | Expected Throughput |
|----------|-------------|---------------|----------------|---------------------|
| Checkpoint save (large sequential write) | 4 MB (4194304) | 32 | 4 | 3-6 GB/s per NVMe |
| Checkpoint load (large sequential read) | 4 MB (4194304) | 32 | 4 | 3-7 GB/s per NVMe |
| Parameter swap (many small reads) | 1 MB (1048576) | 16 | 2 | 2-4 GB/s per NVMe |
| Mixed checkpoint + swap | 1 MB (1048576) | 8 | 1 | 1-3 GB/s per NVMe |

**Guidelines:**
- Use larger `block_size` (2-4 MB) for checkpoint operations where data is accessed sequentially.
- Use higher `queue_depth` (16-32) to maximize device utilization on modern NVMe SSDs.
- Increase `thread_count` (2-4) only if CPU cores are available; single thread is sufficient for most workloads.
- Keep `overlap_events = true` for pipelined I/O; set to `false` only for debugging.

---

## GPU Direct Storage (GDS)

### Overview

GPU Direct Storage (GDS) enables direct data transfer between NVMe storage and GPU memory, bypassing the CPU and system memory entirely. This eliminates the traditional data path (NVMe -> CPU memory -> GPU memory) and replaces it with a direct path (NVMe -> GPU memory), reducing latency and CPU overhead while increasing throughput.

GDS requires NVIDIA GPUDirect Storage-compatible hardware and software (NVIDIA GPUDirect Storage SDK, CUDA 11.4+, compatible filesystems such as Weka, DDN EXAScaler, or BeeGFS).

### GDS Source

```
csrc/gds/
    py_lib/
        deepspeed_gds.py          # GDS Python bindings
    py_test/
        test_gds.py               # GDS tests
    gds_common.h                  # Common GDS definitions
    gds_handle.h                  # GDS handle for managing operations
    gds_handle.cpp                # Handle implementation
```

### GDS Configuration

```json
{
    "gds": {
        "enabled": true,
        "block_size": 1048576,
        "queue_depth": 8
    }
}
```

### GDS Python Interface

```python
class DeepSpeedGds:
    """GPU Direct Storage interface for direct GPU-to-NVMe transfers.

    Requires:
        - NVIDIA GPUDirect Storage SDK
        - CUDA 11.4 or later
        - GDS-compatible filesystem
        - NVIDIA GPU with GPUDirect support
    """

    def __init__(self, block_size=1048576, queue_depth=8):
        ...

    def pread_gpu(self, gpu_buffer, file_path, offset, length):
        """Read directly from file to GPU memory.

        Data path: NVMe -> GPU memory (bypasses CPU).

        Args:
            gpu_buffer: Target GPU tensor.
            file_path: Source file path.
            offset: Byte offset in the file.
            length: Number of bytes to read.
        """
        ...

    def pwrite_gpu(self, gpu_buffer, file_path, offset, length):
        """Write directly from GPU memory to file.

        Data path: GPU memory -> NVMe (bypasses CPU).

        Args:
            gpu_buffer: Source GPU tensor.
            file_path: Target file path.
            offset: Byte offset in the file.
            length: Number of bytes to write.
        """
        ...
```

---

## DeepNVMe

### Overview

DeepNVMe is DeepSpeed's NVMe optimization layer that provides affordable I/O scaling for deep learning workloads. It enables training of models that exceed GPU memory by transparently swapping data between GPU memory and NVMe SSDs. DeepNVMe was introduced as part of ZeRO-Infinity (also called ZeRO-3 Infinity), which extends the memory hierarchy from GPU -> CPU -> NVMe, enabling trillion-parameter model training on limited GPU resources.

### NVMe Device Management

```python
# deepspeed/nvme/nvme.py

class NVMeDevice:
    """Manages an NVMe device for DeepSpeed I/O operations.

    Provides device enumeration, health checking, and
    performance tuning for NVMe SSDs used in training.
    """

    @staticmethod
    def get_nvme_device_list():
        """List available NVMe devices.

        Returns:
            list[str]: Paths to available NVMe devices (e.g., ['/dev/nvme0n1']).
        """
        ...

    def __init__(self, mount_point, device_name=None):
        """Initialize NVMe device.

        Args:
            mount_point: Filesystem mount point for the NVMe device.
            device_name: Optional device name (auto-detected if not specified).
        """
        ...

    def get_device_info(self):
        """Get NVMe device information.

        Returns:
            dict: Device info including model, serial, capacity,
                  available space, and health status.
        """
        ...
```

### NVMe Optimization for Checkpointing

When saving checkpoints for large models (e.g., 175B parameters), the checkpoint files can be hundreds of gigabytes. DeepNVMe optimizes this by:

1. **Parallel writes**: Each rank writes its partition of the checkpoint to its local NVMe device simultaneously, avoiding network bottleneck of shared filesystems.

2. **Streaming writes**: Checkpoint data is written in a streaming fashion with configurable buffer sizes, matching the NVMe device's optimal write pattern.

3. **Compression support**: Optional compression of checkpoint data before writing to reduce I/O volume.

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": false
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": false
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        }
    }
}
```

---

## Swap Tensor System

### Overview

The swap tensor system is the runtime component that manages the movement of tensors (parameters, gradients, optimizer states) between GPU memory and NVMe storage. It is the core mechanism that enables ZeRO-Infinity to train models larger than aggregate GPU memory.

### AsyncPartitionedParameterSwapper

```python
# deepspeed/runtime/swap_tensor/async_partitioned_param_swapper.py

class AsyncPartitionedParameterSwapper:
    """Manages asynchronous swapping of partitioned parameters between
    GPU memory and NVMe storage.

    This class handles the complex logistics of:
    - Partitioning parameters across data-parallel ranks
    - Swapping partitions to/from NVMe asynchronously
    - Overlapping swap operations with computation
    - Managing swap buffers and I/O scheduling
    """

    def __init__(self, ds_config, model_dtype, dtype):
        """Initialize the parameter swapper.

        Args:
            ds_config: DeepSpeed configuration dictionary.
            model_dtype: Model parameter data type (e.g., torch.float16).
            dtype: Compute data type for buffer operations.
        """
        self.aio_config = ds_config.aio_config  # AIO configuration
        self.swap_config = ds_config.zero_config.offload_param  # Offload config
        self.aio_handle = None  # Initialized on first use
        self.swap_buffer_pool = None  # Buffer pool for swap operations
        self.aio_callback = None  # Callback for swap completion

    async def swap_out_parameter(self, parameter, parameter_name):
        """Asynchronously swap a parameter from GPU to NVMe.

        Steps:
        1. Allocate a swap buffer from the pool
        2. Copy parameter data to the swap buffer
        3. Submit async write to NVMe
        4. Return immediately (write completes in background)

        Args:
            parameter: The parameter tensor to swap out.
            parameter_name: Unique name for the swap file.
        """
        ...

    async def swap_in_parameter(self, parameter_name, target_buffer):
        """Asynchronously swap a parameter from NVMe to GPU.

        Steps:
        1. Submit async read from NVMe to swap buffer
        2. Copy from swap buffer to target GPU buffer on completion
        3. Return future for synchronization

        Args:
            parameter_name: Name of the swapped parameter.
            target_buffer: GPU buffer to receive the parameter data.

        Returns:
            Future that resolves when the swap-in completes.
        """
        ...

    def synchronize(self):
        """Wait for all pending swap operations to complete."""
        ...
```

### SwapBuffer

```python
# deepspeed/runtime/swap_tensor/swap_buffer.py

class SwapBuffer:
    """A fixed-size buffer for staging data during GPU <-> NVMe transfers.

    Swap buffers are allocated in pinned (page-locked) host memory or
    GPU memory, depending on the configuration. They serve as the
    intermediate staging area between GPU tensors and NVMe files.

    Key design decisions:
    - Fixed allocation: Buffers are pre-allocated and reused to avoid
      malloc/free overhead during training.
    - Pinned memory: Uses cudaMallocHost for host buffers to enable
      faster GPU <-> Host transfers via DMA.
    - Alignment: Buffers are aligned to block boundaries for optimal
      NVMe I/O performance.
    """

    def __init__(self, buffer_size, dtype, device, numa_node=None):
        """Initialize a swap buffer.

        Args:
            buffer_size: Size of the buffer in number of elements.
            dtype: Data type of elements (e.g., torch.float16).
            device: Device for the buffer ('cpu' or 'cuda:X').
            numa_node: NUMA node for memory allocation (optimizes for
                       local NVMe device access).
        """
        ...

    def allocate(self):
        """Allocate the buffer memory.

        For CPU buffers, uses pinned memory via torch.cuda.pin_memory.
        For GPU buffers, uses standard CUDA allocation.
        """
        ...

    def get_slice(self, offset, length):
        """Get a view/slice of the buffer.

        Args:
            offset: Starting element offset.
            length: Number of elements in the slice.

        Returns:
            torch.Tensor: A view into the buffer.
        """
        ...
```

### SwapBufferPool

```python
# deepspeed/runtime/swap_tensor/swap_buffer_pool.py

class SwapBufferPool:
    """Pool of SwapBuffers for managing multiple concurrent swap operations.

    The buffer pool pre-allocates a set of swap buffers and manages
    their allocation/deallocation. This avoids repeated memory
    allocation during training and ensures that buffer memory is
    reused efficiently.

    Pool sizing is critical:
    - Too few buffers: Swap operations stall waiting for buffers
    - Too many buffers: Wastes memory that could be used for training
    """

    def __init__(self, num_buffers, buffer_size, dtype, device):
        """Initialize the buffer pool.

        Args:
            num_buffers: Number of buffers in the pool.
            buffer_size: Size of each buffer in elements.
            dtype: Data type of buffer elements.
            device: Device for buffer allocation.
        """
        self.buffers = [SwapBuffer(buffer_size, dtype, device) for _ in range(num_buffers)]
        self.available = Queue()  # Available buffer indices
        self.in_use = set()       # Indices of buffers currently in use

    def allocate_buffer(self):
        """Get an available buffer from the pool.

        Blocks if no buffers are available until one is released.

        Returns:
            SwapBuffer: An available swap buffer.
        """
        ...

    def release_buffer(self, buffer):
        """Return a buffer to the pool.

        Args:
            buffer: The SwapBuffer to release.
        """
        ...

    def get_available_count(self):
        """Return the number of currently available buffers."""
        ...
```

### AIO Config for Swap Operations

The AIO configuration for the swap tensor system is specified in the ZeRO offload configuration:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme/checkpoint_tensors",
            "pin_memory": false,
            "buffer_count": 5,
            "fast_init": false,
            "pipeline_read": false,
            "pipeline_write": false,
            "max_in_cpu": 0
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme/optimizer_states",
            "pin_memory": false,
            "pipeline_read": true,
            "pipeline_write": false,
            "pipeline_num_chunks": 2
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        }
    }
}
```

### Swap Operation Flow

The swap tensor system follows a pipelined operation flow:

1. **Swap-Out (GPU -> NVMe)**:
   ```
   Parameter on GPU
     -> Copy to SwapBuffer (pinned host memory or GPU buffer)
       -> Async write to NVMe via AIO
         -> Signal completion
   ```

2. **Swap-In (NVMe -> GPU)**:
   ```
   Async read from NVMe via AIO
     -> Read into SwapBuffer (pinned host memory)
       -> Copy to GPU tensor
         -> Signal completion (parameter ready for use)
   ```

3. **Overlapping with Computation**:
   ```
   Timeline:
   |---Compute Layer 0---|---Compute Layer 1---|---Compute Layer 2---|
   |--Swap-In L1--|                          |--Swap-Out L0--|
                      |--Swap-In L2--|
   ```

   The swap system prefetches parameters for upcoming layers while the current layer is computing, hiding I/O latency behind computation.

---

## Buffer Management

### Buffer Sizing

Proper buffer sizing is critical for NVMe swap performance:

```python
# Recommended buffer sizes based on model size

# For a 10B parameter model with FP16 parameters:
# Total parameter memory: 10B * 2 bytes = 20 GB
# Per-GPU partition (8 GPUs): 2.5 GB
# Buffer size: at least 256 MB per buffer to amortize I/O overhead
# Buffer count: 4-8 to enable pipelining

# For a 175B parameter model with FP16 parameters:
# Total parameter memory: 175B * 2 bytes = 350 GB
# Per-GPU partition (64 GPUs): ~5.5 GB
# Buffer size: at least 512 MB per buffer
# Buffer count: 8-16 for aggressive pipelining
```

### Buffer Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_count` | int | `5` | Number of swap buffers in the pool. More buffers enable more concurrent swap operations. |
| `pipeline_read` | bool | `false` | Enable pipelined reads for overlapping swap-in with computation. |
| `pipeline_write` | bool | `false` | Enable pipelined writes for overlapping swap-out with computation. |
| `pipeline_num_chunks` | int | `2` | Number of chunks for pipelining. Higher values increase overlap but require more buffers. |
| `fast_init` | bool | `false` | Use fast initialization for swap buffers (skip zero-filling). |
| `max_in_cpu` | int | `0` | Maximum number of parameters to keep in CPU memory as cache. 0 means no CPU caching. |

---

## NVMe Performance Tuning

### Hardware Recommendations

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| NVMe SSD | PCIe 3.0 x4 (3.5 GB/s) | PCIe 4.0 x4 (7 GB/s) | Sequential throughput is critical |
| NVMe Count | 1 per node | 2-4 per node | Striping across multiple NVMe increases throughput |
| CPU Cores | 1 per NVMe for I/O threads | 2-4 per NVMe | I/O threads compete with training threads for CPU |
| System Memory | 128 GB | 256-512 GB | Host-side buffer staging |
| PCIe Lanes | PCIe 3.0 x16 for GPU | PCIe 4.0 x16 for GPU | Avoid sharing lanes between GPU and NVMe |

### Filesystem Recommendations

- **ext4 with XFS**: Good default choice. Mount with `-o noatime,discard` for best performance.
- **XFS**: Often preferred for large file workloads. Better performance than ext4 for files > 100 GB.
- **Avoid NFS**: Network filesystems add latency and reduce throughput for checkpoint operations.
- **Avoid ZFS/btrfs**: Copy-on-write filesystems have poor performance for large sequential writes.

### NVMe Setup Script

```bash
#!/bin/bash
# Setup NVMe for DeepSpeed training

# 1. Identify NVMe devices
nvme_devices=$(lsblk -d -o name,rota,type | grep '0 disk' | grep 'nvme' | awk '{print $1}')

# 2. Create filesystem (if not already formatted)
for dev in $nvme_devices; do
    if ! blkid /dev/$dev >/dev/null 2>&1; then
        echo "Formatting /dev/$dev with XFS..."
        mkfs.xfs -f /dev/$dev
    fi
done

# 3. Create mount points
mkdir -p /local_nvme

# 4. Mount with optimal options
for dev in $nvme_devices; do
    mount_point="/local_nvme"
    mount -o noatime,discard /dev/$dev $mount_point
    echo "Mounted /dev/$dev at $mount_point"
done

# 5. Set permissions
chmod 777 /local_nvme

# 6. Verify NVMe performance
echo "Testing NVMe write performance..."
dd if=/dev/zero of=/local_nvme/test bs=1M count=10240 oflag=direct
echo "Testing NVMe read performance..."
dd of=/dev/null if=/local_nvme/test bs=1M count=10240 iflag=direct
rm /local_nvme/test
```

---

## Configuration Examples

### Example 1: ZeRO-3 with NVMe Parameter Offloading

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme/params",
            "pin_memory": false,
            "buffer_count": 5,
            "pipeline_read": true,
            "pipeline_write": true,
            "pipeline_num_chunks": 2
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme/optimizer",
            "pin_memory": false,
            "pipeline_read": true,
            "pipeline_write": false
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        }
    },
    "gradient_clipping": 1.0
}
```

### Example 2: High-Throughput NVMe for Checkpointing

```json
{
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-4,
            "betas": [0.9, 0.95]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme/optimizer_states"
        }
    },
    "aio": {
        "block_size": 4194304,
        "queue_depth": 32,
        "thread_count": 4,
        "single_submit": false,
        "overlap_events": true
    },
    "checkpoint": {
        "use_node_local_storage": true
    }
}
```

### Example 3: ZeRO-Infinity (Full NVMe Offloading)

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 12
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "pipeline_read": true,
            "pipeline_write": true,
            "pipeline_num_chunks": 3
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "pipeline_read": true,
            "pipeline_write": true,
            "pipeline_num_chunks": 3
        },
        "aio": {
            "block_size": 4194304,
            "queue_depth": 16,
            "thread_count": 2,
            "single_submit": false,
            "overlap_events": true
        }
    },
    "gradient_clipping": 1.0,
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true
    }
}
```

### Example 4: GDS-Enabled Configuration

```json
{
    "train_batch_size": 512,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/gds_mount/params",
            "pin_memory": false
        },
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/gds_mount/optimizer",
            "pin_memory": false
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        }
    },
    "gds": {
        "enabled": true,
        "block_size": 1048576,
        "queue_depth": 8
    }
}
```

---

## Performance Characteristics

### NVMe Throughput Bottleneck Analysis

For ZeRO-Infinity with NVMe offloading, the training throughput is bounded by:

```
Effective throughput = min(
    GPU_compute_throughput,
    NVMe_bandwidth / (bytes_per_param * swap_factor),
    PCIe_bandwidth / (bytes_per_param * transfer_factor)
)
```

Where:
- `swap_factor`: Average number of times each parameter is swapped per training step (depends on pipeline depth and model architecture).
- `transfer_factor`: Average number of times each parameter crosses the PCIe bus per step.

### Typical Throughput on NVIDIA A100 + PCIe 4.0 NVMe

| Model Size | GPUs | NVMe Offload | Throughput (TFLOPS) | vs Pure GPU |
|------------|------|-------------|---------------------|-------------|
| 10B | 8x A100 40GB | None (fits in GPU) | 148 | 100% |
| 10B | 4x A100 40GB | Params + Optim to NVMe | 82 | 55% |
| 175B | 32x A100 40GB | Params + Optim to NVMe | 126 | - |
| 175B | 16x A100 80GB | Params + Optim to NVMe | 68 | - |
| 530B | 32x A100 80GB | Params + Optim to NVMe | 45 | - |

---

## Troubleshooting

### Common Issues

1. **"NVMe path not found"**: Ensure the NVMe device is mounted at the specified path. Check `df -h` and `lsblk`.

2. **Slow swap performance**: Increase `queue_depth` and `block_size`. Ensure the NVMe device is not shared with other I/O-intensive processes. Verify NVMe health with `smartctl -a /dev/nvme0`.

3. **Out of host memory**: NVMe swapping uses pinned host memory as staging buffers. Reduce `buffer_count` or `buffer_size` if host memory is limited.

4. **GDS initialization failure**: GDS requires NVIDIA GPUDirect Storage SDK, compatible filesystem, and CUDA 11.4+. Check compatibility with `nvidia-smi -q | grep "GDS"`.

5. **Checkpoint corruption**: Use `overlap_events = true` and ensure the filesystem supports O_DIRECT. Avoid using NFS for checkpoint storage.

6. **"AIO library not found"**: Build DeepSpeed with AIO support: `DS_BUILD_AIO=1 pip install deepspeed --global-option="build_ext"`.

### Environment Variables

```bash
# Build AIO kernel
export DS_BUILD_AIO=1

# Build GDS kernel
export DS_BUILD_GDS=1

# NVMe debug logging
export DS_NVME_DEBUG=1

# Force libaio backend (instead of threaded)
export DS_AIO_LIBAIO=1

# Disable pipeline reads (for debugging)
export DS_DISABLE_PIPELINE_READ=1
```

---

## Summary

DeepSpeed's I/O and NVMe subsystem provides the high-throughput data movement infrastructure required for training models that exceed GPU memory capacity. The asynchronous I/O (AIO) library delivers near-hardware-limit throughput through configurable block sizes, queue depths, and multi-threaded I/O submission. GPU Direct Storage (GDS) further accelerates transfers by enabling direct GPU-to-NVMe paths that bypass the CPU. The swap tensor system (`AsyncPartitionedParameterSwapper`, `SwapBuffer`, `SwapBufferPool`) manages the complex logistics of asynchronously moving parameters and optimizer states between GPU memory and NVMe, enabling ZeRO-Infinity to train models with hundreds of billions or trillions of parameters. Together, these components form a tiered storage hierarchy (GPU -> CPU -> NVMe) that makes large-model training accessible on commodity hardware with local NVMe SSDs.
