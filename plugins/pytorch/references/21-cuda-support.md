# CUDA Support in PyTorch

This chapter provides a comprehensive reference for all CUDA-related functionality in PyTorch, covering device management, memory management, streams, events, graphs, NVTX profiling, multi-GPU support, and environment variables.

---

## 1. torch.cuda Module Overview

The `torch.cuda` module provides access to CUDA-specific functionality. It acts as the primary interface for GPU operations, device management, and memory handling.

```python
import torch

# Check basic CUDA availability
torch.cuda.is_available()       # True if CUDA is available
torch.cuda.is_initialized()     # True if CUDA has been initialized
```

CUDA is lazily initialized on the first CUDA operation. You can force initialization:

```python
torch.cuda.init()  # Explicitly initialize CUDA (rarely needed)
```

### 1.1 CUDA Availability Checks

```python
torch.cuda.is_available() -> bool
```

Returns `True` if CUDA is available on this system. Checks for a compatible NVIDIA GPU and CUDA driver.

```python
torch.cuda.is_initialized() -> bool
```

Returns `True` if the CUDA driver has been initialized. CUDA initialization happens lazily.

```python
torch.cuda.is_bf16_supported() -> bool
```

Returns `True` if the current CUDA device supports BFloat16 operations.

```python
# Comprehensive availability check
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_mem / 1e9:.2f} GB")
```

---

## 2. Device Management

### 2.1 torch.cuda.current_device

```python
torch.cuda.current_device() -> int
```

Returns the index of the currently selected GPU device.

**Returns:** Integer device index (0-based).

**Raises:** RuntimeError if CUDA is not available.

```python
device = torch.cuda.current_device()
print(f"Current device: cuda:{device}")
```

### 2.2 torch.cuda.set_device

```python
torch.cuda.set_device(device: Union[int, torch.device, str]) -> None
```

Sets the current GPU device. Subsequent CUDA operations will use this device by default.

**Parameters:**
- `device` (int | torch.device | str): GPU device index or device object. Can be an integer, a `torch.device` object, or a string like `"cuda:0"`.

```python
# Set by integer index
torch.cuda.set_device(0)

# Set by torch.device object
torch.cuda.set_device(torch.device('cuda:1'))

# Set by string
torch.cuda.set_device('cuda:2')

# Context manager approach (preferred)
with torch.cuda.device(1):
    # All operations here use device 1
    x = torch.randn(3, 3, device='cuda')
    # x is on cuda:1

# After context manager, device reverts to original
```

### 2.3 torch.cuda.device_count

```python
torch.cuda.device_count() -> int
```

Returns the number of available CUDA-capable GPUs.

```python
n_gpus = torch.cuda.device_count()
print(f"Found {n_gpus} GPUs")
```

### 2.4 Device Properties

```python
torch.cuda.get_device_name(device: Optional[int] = None) -> str
```

Returns the name of the GPU device.

```python
torch.cuda.get_device_capability(device: Optional[int] = None) -> Tuple[int, int]
```

Returns the compute capability of the device as a tuple `(major, minor)`.

```python
torch.cuda.get_device_properties(device: Optional[int] = None) -> _CudaDeviceProperties
```

Returns device properties including:
- `name`: Device name string
- `major`, `minor`: Compute capability version
- `total_memory`: Total memory in bytes
- `multi_processor_count`: Number of SMs
- `is_integrated`, `is_multi_gpu_board`: Hardware characteristics

```python
# Example: enumerate all device properties
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"Device {i}: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"  Multiprocessors: {props.multi_processor_count}")
    print(f"  Is Integrated: {props.is_integrated}")
    print(f"  Is Multi-GPU Board: {props.is_multi_gpu_board}")
```

### 2.5 torch.cuda.synchronize

```python
torch.cuda.synchronize(device: Optional[Union[int, torch.device]] = None) -> None
```

Waits for all kernels in all streams on the specified device to complete. If `device` is None, uses the current device.

**Parameters:**
- `device` (int | torch.device | None): Device to synchronize. Default: current device.

```python
# Synchronize current device
torch.cuda.synchronize()

# Synchronize specific device
torch.cuda.synchronize(device=0)

# Timing a CUDA operation
import time

start = time.time()
x = torch.randn(10000, 10000, device='cuda')
y = x @ x.T
torch.cuda.synchronize()  # Wait for GPU to finish
end = time.time()
print(f"Time: {end - start:.4f}s")
```

---

## 3. Memory Management

### 3.1 Memory Query Functions

#### torch.cuda.memory_allocated

```python
torch.cuda.memory_allocated(device: Optional[Union[int, torch.device]] = None) -> int
```

Returns the current GPU memory occupied by tensors in bytes for a given device. This does not include memory in the caching allocator's free pool.

```python
# Check memory before and after allocation
before = torch.cuda.memory_allocated(0)
x = torch.randn(1000, 1000, device='cuda:0')
after = torch.cuda.memory_allocated(0)
print(f"Tensor used {after - before} bytes")  # ~4,000,000 bytes (1000*1000*4)
```

#### torch.cuda.max_memory_allocated

```python
torch.cuda.max_memory_allocated(device: Optional[Union[int, torch.device]] = None) -> int
```

Returns the maximum GPU memory occupied by tensors in bytes for a given device. Track the peak memory usage.

```python
torch.cuda.reset_peak_memory_stats()  # Reset before measurement
x = torch.randn(10000, 10000, device='cuda')
del x
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1e9:.2f} GB")
```

#### torch.cuda.memory_reserved

```python
torch.cuda.memory_reserved(device: Optional[Union[int, torch.device]] = None) -> int
```

Returns the current GPU memory managed by the caching allocator in bytes (both used and free blocks). This is the memory PyTorch has requested from the CUDA driver.

#### torch.cuda.max_memory_reserved

```python
torch.cuda.max_memory_reserved(device: Optional[Union[int, torch.device]] = None) -> int
```

Returns the maximum GPU memory managed by the caching allocator in bytes.

```python
# Compare allocated vs reserved
allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
print(f"Allocated: {allocated / 1e9:.2f} GB")
print(f"Reserved:  {reserved / 1e9:.2f} GB")
print(f"Free in pool: {(reserved - allocated) / 1e9:.2f} GB")
```

#### torch.cuda.memory_stats

```python
torch.cuda.memory_stats(device: Optional[Union[int, torch.device]] = None) -> Dict[str, int]
```

Returns a dictionary of CUDA memory allocator statistics for the given device.

```python
stats = torch.cuda.memory_stats()
for key, value in sorted(stats.items()):
    print(f"{key}: {value}")
```

Key statistics include:
- `allocated_bytes.all.current`: Current allocated bytes
- `allocated_bytes.all.peak`: Peak allocated bytes
- `reserved_bytes.all.current`: Current reserved bytes
- `reserved_bytes.all.peak`: Peak reserved bytes
- `active_bytes.all.current`: Bytes in active use
- `num_alloc_retries`: Number of allocation retries
- `num_ooms`: Number of OOM events

#### torch.cuda.memory_summary

```python
torch.cuda.memory_summary(
    device: Optional[Union[int, torch.device]] = None,
    abbreviated: bool = True
) -> str
```

Returns a human-readable printout of the current memory allocator statistics.

```python
print(torch.cuda.memory_summary())
# Output includes:
# |===========================================================================|
# |                  PyTorch CUDA memory summary, device ID 0               |
# |---------------------------------------------------------------------------|
# |            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
# |===========================================================================|
# |        Metric         | Cur Usage  | Max Usage  | Num Allocs   | Max Allocs |
# |---------------------------------------------------------------------------|
# | Allocated memory      |   1024 B   |   2048 B   |       2      |       3    |
# |       from large pool|   1024 B   |   2048 B   |       2      |       3    |
# |---------------------------------------------------------------------------|
# | Reserved memory       |   2048 B   |   2048 B   |       1      |       1    |
# |       from large pool|   2048 B   |   2048 B   |       1      |       1    |
# |===========================================================================|
```

### 3.2 Memory Management Operations

#### torch.cuda.empty_cache

```python
torch.cuda.empty_cache() -> None
```

Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU applications. This does not free memory occupied by tensors -- only memory that is already free in the caching allocator pool.

```python
# Free unused cached memory
x = torch.randn(10000, 10000, device='cuda')
del x  # Tensor is freed, but memory stays in caching pool
torch.cuda.empty_cache()  # Now returns memory to CUDA driver

# Typical usage pattern during training
def train_epoch(model, dataloader, optimizer):
    for batch in dataloader:
        # ... training code ...

    # At end of epoch, optionally clear cache
    torch.cuda.empty_cache()
```

#### torch.cuda.reset_peak_memory_stats

```python
torch.cuda.reset_peak_memory_stats(device: Optional[Union[int, torch.device]] = None) -> None
```

Resets the starting point for tracking peak memory statistics.

```python
torch.cuda.reset_peak_memory_stats()
# ... run model ...
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory during run: {peak / 1e9:.2f} GB")
```

#### torch.cuda.reset_accumulated_memory_stats

```python
torch.cuda.reset_accumulated_memory_stats(device: Optional[Union[int, torch.device]] = None) -> None
```

Resets accumulated memory statistics (num_alloc_retries, num_ooms, etc.).

---

## 4. Pinned (Page-Locked) Memory

### 4.1 torch.cuda.pin_memory

```python
torch.cuda.pin_memory(
    tensor: torch.Tensor,
    device: Optional[torch.device] = None
) -> torch.Tensor
```

Pins the memory of a CPU tensor so that the data can be transferred to GPU more efficiently via DMA (Direct Memory Access). Pinned memory enables faster host-to-device transfers because the GPU can DMA directly from pinned memory.

**Parameters:**
- `tensor` (Tensor): CPU tensor to pin
- `device` (torch.device | None): Target device for the pinned memory

**Returns:** A new tensor with the same data but pinned in host memory.

```python
# Pin a tensor for faster transfer
x = torch.randn(1000, 1000)  # CPU tensor
x_pinned = torch.cuda.pin_memory(x)

# Transfer to GPU (faster from pinned memory)
x_gpu = x_pinned.to('cuda', non_blocking=True)

# Using pin_memory with DataLoader for automatic pinning
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, pin_memory=True)
```

### 4.2 torch.cuda.is_pinned

```python
torch.cuda.is_pinned(
    tensor: torch.Tensor,
    device: Optional[torch.device] = None
) -> bool
```

Returns `True` if the tensor is in pinned memory.

```python
x = torch.randn(100)
print(torch.cuda.is_pinned(x))  # False
x_pinned = torch.cuda.pin_memory(x)
print(torch.cuda.is_pinned(x_pinned))  # True
```

### 4.3 Pinned Memory with Storage

```python
# Create pinned storage directly
pinned_storage = torch.Storage(1000).pin_memory()

# Create tensor from pinned storage
tensor = torch.tensor(pinned_storage)

# Using pin_memory_tensor context
with torch.cuda.device(0):
    pinned = torch.randn(100).pin_memory()
```

---

## 5. CUDA Random Number Generation

### 5.1 torch.cuda.manual_seed

```python
torch.cuda.manual_seed(seed: int) -> None
```

Sets the seed for generating random numbers for the current GPU device. It is safe to call this function even if CUDA is not available; in that case, it is silently ignored.

**Parameters:**
- `seed` (int): The desired seed value. Must be a non-negative integer within the range of `torch.uint64` (i.e., 0 <= seed < 2^64).

```python
torch.cuda.manual_seed(42)
a = torch.randn(3, device='cuda')
torch.cuda.manual_seed(42)
b = torch.randn(3, device='cuda')
assert torch.equal(a, b)  # Same seed produces same results
```

### 5.2 torch.cuda.manual_seed_all

```python
torch.cuda.manual_seed_all(seed: int) -> None
```

Sets the seed for generating random numbers on all GPUs. Useful for multi-GPU training.

```python
# Set same seed on all GPUs
torch.cuda.manual_seed_all(42)

# Best practice: set all seeds together
def set_all_seeds(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)
```

### 5.3 torch.cuda.initial_seed

```python
torch.cuda.initial_seed(device: Optional[int] = None) -> int
```

Returns the initial seed for the current GPU device. This is the seed that was last set via `manual_seed` or the random seed generated at initialization.

```python
seed = torch.cuda.initial_seed()
print(f"Initial seed: {seed}")
```

### 5.4 torch.cuda.get_rng_state

```python
torch.cuda.get_rng_state(device: Optional[Union[int, torch.device]] = None) -> torch.Tensor
```

Returns the random number generator state of the specified GPU as a ByteTensor.

```python
# Save RNG state
state = torch.cuda.get_rng_state()

# Generate some random numbers
a = torch.randn(3, device='cuda')

# Restore RNG state
torch.cuda.set_rng_state(state)
b = torch.randn(3, device='cuda')
# b will be the same as what a was (same seed state)
```

### 5.5 torch.cuda.set_rng_state

```python
torch.cuda.set_rng_state(
    new_state: torch.Tensor,
    device: Optional[Union[int, torch.device]] = None
) -> None
```

Sets the random number generator state of the specified GPU.

### 5.6 torch.cuda.get_rng_state_all

```python
torch.cuda.get_rng_state_all() -> List[torch.Tensor]
```

Returns a list of ByteTensor representing the random number states for all GPUs.

### 5.7 torch.cuda.set_rng_state_all

```python
torch.cuda.set_rng_state_all(new_states: List[torch.Tensor]) -> None
```

Sets the random number generator state for all GPUs.

```python
# Save and restore all GPU RNG states
states = torch.cuda.get_rng_state_all()
# ... do random operations ...
torch.cuda.set_rng_state_all(states)
```

---

## 6. CUDA Streams

### 6.1 Stream Class

```python
torch.cuda.Stream(
    device: Optional[Union[int, torch.device]] = None,
    priority: int = 0
)
```

A CUDA stream is a sequence of operations that execute in order on the GPU. Operations in different streams can run concurrently.

**Parameters:**
- `device` (int | torch.device | None): The device on which to create the stream. Default: current device.
- `priority` (int): Priority of the stream. Lower numbers represent higher priorities. The default priority is 0. The range of meaningful priorities is `[-(stream_priority_range[1] - stream_priority_range[0]), 0]`.

```python
# Create a regular stream
stream = torch.cuda.Stream()

# Create a high-priority stream
high_priority_stream = torch.cuda.Stream(priority=-1)

# Create stream on specific device
stream_on_1 = torch.cuda.Stream(device=1)
```

**Stream Properties:**
- `stream.device`: The device this stream is on
- `stream.priority`: Priority value

### 6.2 Stream Context Manager

```python
with torch.cuda.stream(stream):
    # Operations in this block run on `stream`
    x = torch.randn(10, device='cuda')
    y = x * 2
```

### 6.3 Stream Methods

```python
stream.query() -> bool
```

Checks if all operations submitted to the stream have completed.

```python
stream.synchronize() -> None
```

Waits for all operations on this stream to complete.

```python
stream.wait_event(event) -> None
```

Makes all future work submitted to the stream wait for the event.

```python
stream.wait_stream(other_stream) -> None
```

Synchronizes with another stream. All future work on this stream will wait for all work currently on `other_stream` to complete.

```python
# Example: overlapping computation
import torch

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# Start work on stream s1
with torch.cuda.stream(s1):
    a = torch.randn(10000, 10000, device='cuda')
    b = a @ a.T

# Start work on stream s2 (concurrent with s1)
with torch.cuda.stream(s2):
    c = torch.randn(10000, 10000, device='cuda')
    d = c @ c.T

# Wait for both to finish
torch.cuda.synchronize()
```

### 6.4 Default and Current Streams

```python
torch.cuda.default_stream(device: Optional[Union[int, torch.device]] = None) -> torch.cuda.Stream
```

Returns the default CUDA stream for the given device.

```python
torch.cuda.current_stream(device: Optional[Union[int, torch.device]] = None) -> torch.cuda.Stream
```

Returns the currently selected CUDA stream.

```python
torch.cuda.set_stream(stream: torch.cuda.Stream) -> None
```

Sets the current stream. All subsequent CUDA operations will use this stream.

```python
# Get default and current stream
default = torch.cuda.default_stream()
current = torch.cuda.current_stream()
print(f"Default == Current: {default == current}")  # Usually True
```

### 6.5 Stream Equality and Comparison

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = s1

print(s1 == s2)  # False (different streams)
print(s1 == s3)  # True (same stream object)
```

---

## 7. CUDA Events

### 7.1 Event Class

```python
torch.cuda.Event(
    enable_timing: bool = False,
    blocking: bool = False,
    interprocess: bool = False
)
```

CUDA events are synchronization markers that can be used to monitor device progress, measure elapsed time between points, and synchronize between streams.

**Parameters:**
- `enable_timing` (bool): If True, the event will record timing data. Default: False.
- `blocking` (bool): If True, `event.wait()` will block the calling CPU thread. Default: False.
- `interprocess` (bool): If True, the event can be shared between processes. Default: False. Requires `enable_timing=False`.

```python
# Create a timing event
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
```

### 7.2 Event Methods

#### record

```python
event.record(stream: Optional[torch.cuda.Stream] = None) -> None
```

Records the event in the given stream.

```python
start_event = torch.cuda.Event(enable_timing=True)
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    start_event.record()
    x = torch.randn(10000, 10000, device='cuda')
    y = x @ x.T
```

#### wait

```python
event.wait(stream: Optional[torch.cuda.Stream] = None) -> None
```

Makes all future work submitted to the given stream wait for the event. If `stream` is None, uses the current stream.

```python
# Cross-stream synchronization
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
event = torch.cuda.Event()

with torch.cuda.stream(s1):
    a = torch.randn(100, 100, device='cuda')
    event.record()

with torch.cuda.stream(s2):
    event.wait()  # s2 waits for event from s1
    b = a @ a.T  # Safe: a is ready
```

#### synchronize

```python
event.synchronize() -> None
```

Synchronizes the CPU thread with the event. The CPU thread will block until the event is recorded and completed.

```python
event = torch.cuda.Event()
x = torch.randn(1000, 1000, device='cuda')
event.record()
event.synchronize()  # CPU blocks until event completes on GPU
```

#### elapsed_time

```python
event.elapsed_time(end_event: torch.cuda.Event) -> float
```

Returns the elapsed time (in milliseconds) between the completion of this event and the completion of `end_event`. Both events must have `enable_timing=True`.

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
x = torch.randn(5000, 5000, device='cuda')
y = x @ x.T
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
print(f"Time: {elapsed_ms:.3f} ms")
```

#### query

```python
event.query() -> bool
```

Checks if all work currently recorded by the event has completed.

```python
event = torch.cuda.Event()
x = torch.randn(100, device='cuda')
event.record()
print(event.query())  # May be True or False
```

### 7.3 Event Properties

```python
event = torch.cuda.Event(enable_timing=True)
print(event.is_timing)       # True
print(event.is_blocking)     # False
print(event.is_interprocess) # False
```

### 7.4 Inter-Process Events

```python
# Create an IPC event
ipc_event = torch.cuda.Event(interprocess=True)

# In the producing process
event.record()
ipc_handle = ipc_event.ipc_handle()

# In the consuming process (different process)
event_from_handle = torch.cuda.Event.from_ipc_handle(device=0, ipc_handle=ipc_handle)
event_from_handle.wait()
```

---

## 8. CUDA Graphs

### 8.1 CUDAGraph Class

```python
torch.cuda.CUDAGraph()
```

CUDA Graphs capture a series of GPU operations into a graph that can be replayed with reduced CPU overhead. This is especially useful for reducing kernel launch overhead for small operations.

### 8.2 Capturing a Graph

```python
torch.cuda.CUDAGraph.capture(
    graph: torch.cuda.CUDAGraph,
    pool: Optional[Tuple[int, int]] = None,
    stream: Optional[torch.cuda.Stream] = None
) -> None
```

Actually, the API is:

```python
g = torch.cuda.CUDAGraph()

# Static inputs (will be reused during replay)
static_input = torch.randn(100, 100, device='cuda')

# Capture the graph
with torch.cuda.graph(g):
    static_output = static_input * 2 + 1
    static_output = static_output.sin()
```

### 8.3 Replaying a Graph

```python
# Modify the input (must use .copy_() to keep same memory)
new_data = torch.randn(100, 100, device='cuda')
static_input.copy_(new_data)

# Replay the graph (very fast, minimal CPU overhead)
g.replay()

# Output is now in static_output with new data
```

### 8.4 Full CUDA Graphs Example

```python
import torch

# Setup
g = torch.cuda.CUDAGraph()
static_input = torch.randn(1000, 1000, device='cuda')
static_weight = torch.randn(1000, 1000, device='cuda')

# Warmup (required before capture)
with torch.no_grad():
    static_output = static_input @ static_weight

# Capture
with torch.cuda.graph(g):
    with torch.no_grad():
        static_output = static_input @ static_weight

# Replay multiple times
for _ in range(1000):
    new_input = torch.randn(1000, 1000, device='cuda')
    static_input.copy_(new_input)
    g.replay()
    # static_output now holds the result for new_input
```

### 8.5 make_graphed_callables

```python
torch.cuda.make_graphed_callables(
    callables: Union[Callable, List[Callable]],
    sample_inputs: Union[Tuple, List[Tuple]],
    *,
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[Tuple[int, int]] = None,
    stream: Optional[torch.cuda.Stream] = None
) -> Union[Callable, List[Callable]]
```

Creates graphed versions of callables (functions, modules) that automatically handle capture and replay.

```python
import torch

model = torch.nn.Linear(512, 512).cuda()
optimizer = torch.optim.Adam(model.parameters())

sample_input = torch.randn(32, 512, device='cuda')

# Create graphed training step
graphed_model = torch.cuda.make_graphed_callables(
    model,
    (sample_input,),
    num_warmup_iters=3
)

# Use like regular model
output = graphed_model(sample_input)
```

### 8.6 Graph Memory Pooling

```python
# Share memory pool between graphs
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# First graph capture sets up the pool
static_input1 = torch.randn(100, device='cuda')
with torch.cuda.graph(g1):
    static_output1 = static_input1 * 2

# Second graph reuses the pool
pool = g1.pool()
static_input2 = torch.randn(100, device='cuda')
with torch.cuda.graph(g2, pool=pool):
    static_output2 = static_input2 * 3
```

### 8.7 CUDAGraph Properties

```python
g = torch.cuda.CUDAGraph()
# After capture:
g.pool()  # Returns the memory pool used by the graph
g.device  # The device the graph was captured on
```

---

## 9. NVTX (NVIDIA Tools Extension) Profiling

### 9.1 Overview

NVTX annotations allow you to mark regions of code that appear in NVIDIA profiling tools like Nsight Systems and Nsight Compute.

### 9.2 torch.cuda.nvtx.range_push

```python
torch.cuda.nvtx.range_push(msg: str) -> None
```

Pushes a range onto a stack of nested range spans. Call `range_pop()` to end the range.

**Parameters:**
- `msg` (str): The message to associate with the range in the profiler.

```python
torch.cuda.nvtx.range_push("forward_pass")
# ... forward pass code ...
torch.cuda.nvtx.range_pop()
```

### 9.3 torch.cuda.nvtx.range_pop

```python
torch.cuda.nvtx.range_pop() -> None
```

Pops a range from the stack of nested range spans. Must be paired with a `range_push()`.

### 9.4 torch.cuda.nvtx.mark

```python
torch.cuda.nvtx.mark(msg: str) -> None
```

Describes an instantaneous event that occurs at the point of the call.

```python
torch.cuda.nvtx.mark("start_batch_42")
```

### 9.5 NVTX Example with Training Loop

```python
import torch

def train_with_nvtx(model, dataloader, optimizer):
    for batch_idx, (data, target) in enumerate(dataloader):
        torch.cuda.nvtx.range_push(f"batch_{batch_idx}")

        torch.cuda.nvtx.range_push("data_transfer")
        data, target = data.cuda(), target.cuda()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("forward")
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()  # batch
```

Run with profiling:
```bash
nsys profile -t cuda,nvtx --force-overwrite=true python train.py
```

---

## 10. Multi-GPU and Peer-to-Peer Access

### 10.1 torch.cuda.can_device_access_peer

```python
torch.cuda.can_device_access_peer(
    device: int,
    peer_device: int
) -> bool
```

Checks if `device` can access memory directly from `peer_device` via peer-to-peer (P2P) access.

```python
if torch.cuda.device_count() >= 2:
    can_p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f"P2P access from GPU 0 to GPU 1: {can_p2p}")
```

### 10.2 torch.cuda.device_set_peer_access

```python
# Enable peer-to-peer access (if supported)
# This is typically done automatically by NCCL
# But can be done manually:
import torch.cuda as cuda

# Check and enable P2P
if cuda.can_device_access_peer(0, 1):
    # P2P is already enabled or available
    pass
```

### 10.3 Cross-Device Operations

```python
# Transfer tensor between GPUs
x = torch.randn(100, 100, device='cuda:0')

# Direct transfer to another GPU
y = x.to('cuda:1')

# Or using device indices
y = x.to(device=1)
```

### 10.4 Multi-GPU Training Patterns

#### DataParallel (Simple, single-machine)

```python
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
output = model(input)
```

#### DistributedDataParallel (Recommended)

```python
import torch.distributed as dist
import torch.multiprocessing as mp

def train_worker(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = MyModel().cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ... training loop ...

    dist.destroy_process_group()

mp.spawn(train_worker, args=(4,), nprocs=4)
```

### 10.5 NCCL Backend

```python
# Check NCCL availability
torch.distributed.is_nccl_available()

# Common NCCL environment variables
# NCCL_DEBUG=INFO        - Enable debug logging
# NCCL_SOCKET_IFNAME=eth0 - Network interface
# NCCL_IB_DISABLE=1      - Disable InfiniBand
# NCCL_P2P_DISABLE=0     - Enable/disable P2P
```

---

## 11. Environment Variables

### 11.1 CUDA Memory Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `PYTORCH_CUDA_ALLOC_CONF` | Configuration for CUDA memory allocator | `max_split_size_mb:128` |
| `CUDA_VISIBLE_DEVICES` | Controls which GPUs are visible to PyTorch | `0,2` or `GPU-abcdef` |
| `CUDA_LAUNCH_BLOCKING` | Synchronize CUDA calls for debugging | `1` |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | Disable memory caching | `1` |
| `PYTORCH_CUDA_ALLOC_CONF` | Memory allocator settings | See below |

### 11.2 PYTORCH_CUDA_ALLOC_CONF Options

```bash
# Limit max split size (blocks larger than this won't be split)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Set garbage collection threshold (0.0 to 1.0)
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.5"

# Combine multiple options
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.5"

# Expandable segments (PyTorch 2.0+)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### 11.3 Debugging Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_LAUNCH_BLOCKING` | Forces synchronous CUDA execution | `1` |
| `TORCH_SHOW_CPP_STACKTRACES` | Show C++ stack traces on errors | `1` |
| `TORCH_CUDA_SANITIZER` | Enable CUDA sanitizer for detecting race conditions | `1` |
| `TORCH_USE_CUDA_DSA` | Enable Device-Side Assertions | `1` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Limit concurrent CUDA connections | `1` |

### 11.4 Performance Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TORCH_CUDNN_V8_API_ENABLED` | Enable cuDNN v8 API | `1` |
| `CUBLAS_WORKSPACE_CONFIG` | Configure cuBLAS workspace for deterministic mode | `:4096:8` |
| `TORCH_CUDNN_BENCHMARK` | Override cudnn.benchmark setting | `0` or `1` |

### 11.5 Setting Environment Variables

```python
import os

# Set before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
print(f"Visible GPUs: {torch.cuda.device_count()}")  # 2
```

```bash
# In shell script
CUDA_VISIBLE_DEVICES=0,1 python train.py

# Or in your job script
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py
```

---

## 12. CUDA Graph Memory Management

### 12.1 Graph Memory Pools

When a CUDA graph is captured, it uses a dedicated memory pool. This pool is separate from the regular PyTorch memory allocator.

```python
# Graph memory is managed separately
g = torch.cuda.CUDAGraph()

# Memory used by the graph is not counted in memory_allocated()
# but is included in memory_reserved()
```

### 12.2 Sharing Memory Pools Between Graphs

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# Capture first graph
x = torch.randn(10, device='cuda')
with torch.cuda.graph(g1):
    y = x * 2

# Share memory pool with second graph
pool = g1.pool()
with torch.cuda.graph(g2, pool=pool):
    z = x * 3
```

---

## 13. Common Patterns and Recipes

### 13.1 Device-Agnostic Code

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# Or with automatic placement
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
```

### 13.2 Multi-GPU Memory Monitoring

```python
def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"GPU {i}: {allocated:.2f}/{total:.2f} GB allocated, "
              f"{reserved:.2f}/{total:.2f} GB reserved")
```

### 13.3 Timing GPU Operations

```python
def time_gpu(func, *args, **kwargs):
    """Time a GPU function using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    # Measure
    start.record()
    for _ in range(100):
        func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / 100.0  # Average ms
```

### 13.4 Safe Tensor Transfer

```python
def safe_to_cuda(tensor, device=None):
    """Safely move tensor to CUDA with fallback to CPU."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        return tensor.to(device, non_blocking=True)
    except RuntimeError:
        return tensor.to(device)
```

---

## 14. Troubleshooting

### 14.1 Common Issues

1. **CUDA out of memory**: Use `torch.cuda.empty_cache()`, reduce batch size, or use gradient checkpointing.

2. **Incorrect device errors**: Ensure all tensors are on the same device before operations.

3. **Slow transfers**: Use `pin_memory=True` in DataLoader, use `non_blocking=True` in `.to()`.

4. **Non-deterministic results**: Set seeds properly and use `torch.use_deterministic_algorithms(True)`.

### 14.2 Debugging Tips

```python
# Enable synchronous execution for precise error locations
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check for NaN/Inf
x = torch.randn(10, device='cuda')
assert not torch.isnan(x).any(), "NaN detected"
assert not torch.isinf(x).any(), "Inf detected"

# Memory debugging
print(torch.cuda.memory_summary())
```
