# Memory Management in PyTorch

This chapter provides a comprehensive reference for memory management in PyTorch, covering the GPU caching allocator model, memory query APIs, out-of-memory handling, environment configuration, memory profiling, gradient checkpointing, and CPU memory management.

---

## 1. PyTorch Memory Allocation Model

### 1.1 Overview

PyTorch manages memory through a hierarchical allocation system:

1. **CUDA Driver Memory:** The lowest level, managed by the NVIDIA driver via `cudaMalloc`/`cudaFree`
2. **Caching Allocator:** PyTorch's memory pool that sits on top of the CUDA driver
3. **Tensor Storage:** The Python-level abstraction that holds data
4. **Tensor View:** Views (slices, reshapes) share underlying storage

```
Tensor (Python object)
  |
  v
Storage (contiguous memory block)
  |
  v
Caching Allocator (manages memory pools)
  |
  v
CUDA Driver (cudaMalloc / cudaFree)
  |
  v
GPU Physical Memory (VRAM)
```

### 1.2 The Caching Allocator

The caching allocator is the core of PyTorch's GPU memory management. It sits between the tensor allocation requests and the CUDA driver.

**How it works:**

1. When a tensor is allocated, the caching allocator first checks its free pool for a suitable block.
2. If a suitable block is found, it is reused without calling `cudaMalloc`.
3. If no suitable block exists, the caching allocator calls `cudaMalloc` to get new memory from the driver.
4. When a tensor is freed (deleted or goes out of scope), its memory is returned to the caching allocator's free pool, not to the CUDA driver.
5. The caching allocator may split large blocks into smaller ones or merge adjacent free blocks.

**Key concepts:**

- **Allocated memory**: Memory currently occupied by tensors
- **Reserved memory**: Memory held by the caching allocator (both used by tensors and free in the pool)
- **Driver memory**: Memory allocated from the CUDA driver via `cudaMalloc`

```
+-------------------+
|  GPU Total VRAM   |
|                   |
|  +-------------+  |
|  | Reserved    |  |
|  |  +--------+ |  |
|  |  |Allocated| |  |
|  |  | (tensors| |  |
|  |  |  in use)| |  |
|  |  +--------+ |  |
|  |             |  |
|  | Free pool   |  |
|  | (available  |  |
|  |  for reuse) |  |
|  +-------------+  |
|                   |
|  Not yet claimed  |
|  by PyTorch       |
+-------------------+
```

### 1.3 Block Allocation Strategy

The caching allocator uses a best-fit strategy with the following rules:

1. **Block sizes are rounded** to the nearest power of 2 (or configurable segment size).
2. **Large blocks** (>= `max_split_size_mb`) are never split into smaller blocks.
3. **Small blocks** can be split to fulfill smaller allocation requests.
4. When a block is freed, the allocator tries to **merge it with adjacent free blocks**.
5. **Garbage collection** is triggered when allocation fails and `garbage_collection_threshold` is set.

---

## 2. Memory Query APIs

### 2.1 torch.cuda.memory_allocated

```python
torch.cuda.memory_allocated(
    device: Optional[Union[int, torch.device]] = None
) -> int
```

Returns the current GPU memory occupied by tensors in bytes for the given device. This only counts memory that is actually being used by tensors, not the memory in the caching allocator's free pool.

**Parameters:**
- `device` (int | torch.device | None): Device index or object. Default: current device.

**Returns:** Integer number of bytes.

```python
import torch

# Before allocation
before = torch.cuda.memory_allocated()
print(f"Before: {before} bytes")

# Allocate tensor
x = torch.randn(1000, 1000, device='cuda')

# After allocation
after = torch.cuda.memory_allocated()
diff = after - before
print(f"Tensor used: {diff} bytes")  # ~4,000,000 (1000*1000*4 for float32)
print(f"Tensor size:  {x.nelement() * x.element_size()} bytes")
```

### 2.2 torch.cuda.max_memory_allocated

```python
torch.cuda.max_memory_allocated(
    device: Optional[Union[int, torch.device]] = None
) -> int
```

Returns the maximum GPU memory occupied by tensors in bytes for a given device. Tracks the peak memory usage since the last `reset_peak_memory_stats()` call or since CUDA initialization.

```python
torch.cuda.reset_peak_memory_stats()

# Simulate training
x = torch.randn(10000, 10000, device='cuda')  # ~400 MB
y = x @ x.T  # Temporary, plus result ~800 MB
del y
x2 = torch.randn(10000, 10000, device='cuda')  # ~400 MB

peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1e9:.2f} GB")
# Peak is the maximum at any point, not just current
```

### 2.3 torch.cuda.memory_reserved

```python
torch.cuda.memory_reserved(
    device: Optional[Union[int, torch.device]] = None
) -> int
```

Returns the current GPU memory managed by the caching allocator in bytes. This includes both memory in use by tensors and memory in the free pool.

```python
x = torch.randn(10000, 10000, device='cuda')
allocated = torch.cuda.memory_allocated()
reserved = torch.cuda.memory_reserved()
free_in_pool = reserved - allocated

print(f"Allocated:     {allocated / 1e9:.3f} GB")
print(f"Reserved:      {reserved / 1e9:.3f} GB")
print(f"Free in pool:  {free_in_pool / 1e9:.3f} GB")
```

### 2.4 torch.cuda.max_memory_reserved

```python
torch.cuda.max_memory_reserved(
    device: Optional[Union[int, torch.device]] = None
) -> int
```

Returns the maximum GPU memory managed by the caching allocator in bytes.

### 2.5 torch.cuda.memory_stats

```python
torch.cuda.memory_stats(
    device: Optional[Union[int, torch.device]] = None
) -> Dict[str, int]
```

Returns a dictionary of CUDA memory allocator statistics for the given device.

```python
stats = torch.cuda.memory_stats()

# Key statistics
print(f"Current allocated: {stats['allocated_bytes.all.current']}")
print(f"Peak allocated:    {stats['allocated_bytes.all.peak']}")
print(f"Current reserved:  {stats['reserved_bytes.all.current']}")
print(f"Peak reserved:     {stats['reserved_bytes.all.peak']}")
print(f"Active segments:   {stats['segment.all.current']}")
print(f"Alloc retries:     {stats['num_alloc_retries']}")
print(f"OOM events:        {stats['num_ooms']}")
```

**Complete list of key statistics:**

| Key | Description |
|-----|-------------|
| `allocated_bytes.all.current` | Currently allocated bytes |
| `allocated_bytes.all.peak` | Peak allocated bytes |
| `allocated_bytes.all.freed` | Total bytes freed |
| `allocated_bytes.all.alloced` | Total bytes allocated |
| `allocated_bytes.large_pool.*` | Stats for large pool allocations |
| `allocated_bytes.small_pool.*` | Stats for small pool allocations |
| `reserved_bytes.all.current` | Currently reserved bytes |
| `reserved_bytes.all.peak` | Peak reserved bytes |
| `active_bytes.all.current` | Bytes in active (non-freed) allocations |
| `segment.all.current` | Current number of segments |
| `segment.all.peak` | Peak number of segments |
| `num_alloc_retries` | Number of times allocation was retried |
| `num_ooms` | Number of OOM events |
| `num_sync_all_streams` | Number of stream synchronization events |

### 2.6 torch.cuda.memory_summary

```python
torch.cuda.memory_summary(
    device: Optional[Union[int, torch.device]] = None,
    abbreviated: bool = True
) -> str
```

Returns a human-readable printout of the current memory allocator statistics.

**Parameters:**
- `device` (int | torch.device | None): Device to query. Default: current device.
- `abbreviated` (bool): If `True`, shows an abbreviated summary. If `False`, shows all statistics. Default: `True`.

```python
print(torch.cuda.memory_summary(abbreviated=False))
```

Sample output:
```
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Max Usage  | Num Allocs   | Max Allocs |
|---------------------------------------------------------------------------|
```

---

## 3. Memory Management Operations

### 3.1 torch.cuda.empty_cache

```python
torch.cuda.empty_cache() -> None
```

Releases all unoccupied cached memory currently held by the caching allocator so that it can be used in other GPU applications and visible in `nvidia-smi`. **This does not free memory occupied by tensors** -- only memory that is already free in the caching allocator pool.

**When to use:**
- Before reporting memory to `nvidia-smi` (which shows reserved, not allocated, memory)
- Before running non-PyTorch GPU code that needs GPU memory
- Between very different phases of processing (e.g., between training and evaluation)

**When NOT to use:**
- Every training iteration (unnecessary overhead, allocator will just re-allocate)
- Expecting it to free tensor memory (it only frees already-free pooled memory)

```python
# Example: empty cache between phases
def train_then_eval(model, train_loader, val_loader):
    # Training phase
    model.train()
    for batch in train_loader:
        # ... training ...
        pass

    # Free up memory before evaluation
    torch.cuda.empty_cache()

    # Evaluation phase (may use different memory patterns)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # ... evaluation ...
            pass
```

### 3.2 torch.cuda.reset_peak_memory_stats

```python
torch.cuda.reset_peak_memory_stats(
    device: Optional[Union[int, torch.device]] = None
) -> None
```

Resets the "peak" stats (max_memory_allocated, max_memory_reserved) to the current values. Useful for tracking peak memory during specific phases.

```python
# Track peak memory for a specific operation
torch.cuda.reset_peak_memory_stats()

output = model(input)
loss = criterion(output, target)
loss.backward()

peak = torch.cuda.max_memory_allocated()
print(f"Peak for forward+backward: {peak / 1e9:.2f} GB")
```

### 3.3 torch.cuda.reset_accumulated_memory_stats

```python
torch.cuda.reset_accumulated_memory_stats(
    device: Optional[Union[int, torch.device]] = None
) -> None
```

Resets accumulated stats (num_alloc_retries, num_ooms, etc.).

```python
torch.cuda.reset_accumulated_memory_stats()
# ... run some code ...
stats = torch.cuda.memory_stats()
print(f"OOMs: {stats['num_ooms']}")
print(f"Retries: {stats['num_alloc_retries']}")
```

---

## 4. OutOfMemoryError Handling

### 4.1 Understanding OOM

A `torch.cuda.OutOfMemoryError` occurs when:
1. The caching allocator cannot find a suitable free block in its pool, AND
2. `cudaMalloc` fails because the GPU is out of memory

**Important:** `empty_cache()` does NOT help with OOM, because the memory in the free pool is already available for reallocation. The OOM means the total reserved + requested exceeds GPU memory.

### 4.2 Catching and Handling OOM

```python
import torch

def try_batch(model, batch, device='cuda'):
    """Try to process a batch, reducing batch size on OOM."""
    data, target = batch

    try:
        data = data.to(device)
        target = target.to(device)

        with torch.amp.autocast('cuda'):
            output = model(data)
            loss = criterion(output, target)

        return loss

    except torch.cuda.OutOfMemoryError:
        # Clear any partial allocations
        torch.cuda.empty_cache()
        print(f"OOM with batch size {data.shape[0]}")
        return None
```

### 4.3 Gradient Accumulation as OOM Solution

When the desired batch size doesn't fit in GPU memory, use gradient accumulation:

```python
effective_batch_size = 64
micro_batch_size = 16
accum_steps = effective_batch_size // micro_batch_size  # 4

optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    data, target = data.cuda(), target.cuda()

    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = criterion(output, target) / accum_steps

    scaler.scale(loss).backward()

    if (i + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 4.4 Memory-Efficient Techniques

```python
# 1. Use in-place operations when safe
x = torch.randn(100, 100, device='cuda')
x.relu_()  # In-place, saves allocation
x.add_(1)  # In-place addition

# 2. Delete unnecessary tensors explicitly
intermediate = compute_intermediate(data)
result = use_intermediate(intermediate)
del intermediate  # Free immediately
torch.cuda.empty_cache()  # Return to driver

# 3. Use torch.no_grad() for inference
with torch.no_grad():
    output = model(input)

# 4. Use .detach() to remove from computation graph
loss = criterion(output, target)
# If you need the loss value but not the graph
loss_value = loss.detach().item()

# 5. Use mixed precision to halve memory
with torch.amp.autocast('cuda', dtype=torch.float16):
    output = model(input)  # Uses ~half the memory
```

---

## 5. PYTORCH_CUDA_ALLOC_CONF Configuration

### 5.1 Configuration Options

The `PYTORCH_CUDA_ALLOC_CONF` environment variable controls the caching allocator behavior. It accepts a comma-separated list of key:value pairs.

```bash
export PYTORCH_CUDA_ALLOC_CONF="key1:value1,key2:value2"
```

#### max_split_size_mb

```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

Blocks larger than this size (in MB) will not be split. This prevents fragmentation for large allocations. Default: no limit (all blocks can be split).

- **When to increase:** If you see excessive splitting in memory stats (many small segments).
- **When to decrease:** If you need more fine-grained memory reuse.
- **Typical values:** 32, 64, 128, 256

#### garbage_collection_threshold

```bash
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.5"
```

Sets the threshold (0.0 to 1.0) for triggering garbage collection of unused blocks. When the ratio of free memory to total reserved memory exceeds this threshold, the allocator will attempt to free blocks.

- **Default:** Not set (no automatic GC)
- **Recommended range:** 0.5 to 0.8
- **When to use:** When you experience memory fragmentation issues

#### expandable_segments

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

(Python 2.0+) Enables expandable segments, which can reduce memory fragmentation by allowing segments to grow.

- **Default:** `False` (PyTorch 2.0-2.1), `True` (PyTorch 2.2+)
- **When to use:** Set to `True` if you experience fragmentation issues

#### round_alloc_size_to

```bash
export PYTORCH_CUDA_ALLOC_CONF="round_alloc_size_to:8"
```

Rounds allocation sizes to the nearest multiple of this value (in bytes). Default: 2 MB (2097152).

### 5.2 Combining Options

```bash
# Multiple options
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"
```

### 5.3 Programmatic Configuration (Python)

```python
# Cannot set PYTORCH_CUDA_ALLOC_CONF after CUDA is initialized
# Must set before first CUDA operation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.5'

import torch  # CUDA will use these settings when initialized
```

---

## 6. Memory Profiling Techniques

### 6.1 Basic Memory Tracking

```python
import torch

def memory_tracker(func, *args, **kwargs):
    """Track memory usage of a function."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    before = torch.cuda.memory_allocated()
    result = func(*args, **kwargs)
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    print(f"Memory before:  {before / 1e9:.3f} GB")
    print(f"Memory after:   {after / 1e9:.3f} GB")
    print(f"Memory delta:   {(after - before) / 1e9:.3f} GB")
    print(f"Peak memory:    {peak / 1e9:.3f} GB")

    return result
```

### 6.2 Per-Layer Memory Tracking

```python
def track_memory_by_layer(model, input_tensor):
    """Track memory consumption layer by layer."""
    torch.cuda.reset_peak_memory_stats()

    x = input_tensor
    activations = [x]

    for name, layer in model.named_children():
        before = torch.cuda.memory_allocated()
        x = layer(x)
        after = torch.cuda.memory_allocated()
        activations.append(x)

        print(f"{name:30s}: +{(after - before) / 1e6:.1f} MB "
              f"(total: {after / 1e9:.2f} GB)")

    total_activations = sum(a.nelement() * a.element_size()
                           for a in activations)
    print(f"\nTotal activation memory: {total_activations / 1e9:.2f} GB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### 6.3 torch.cuda.memory Snapshot

```python
# PyTorch 2.1+ supports memory snapshots for detailed debugging
torch.cuda.memory._record_memory_history(
    enabled='all',
    context='all',
    stacks='python',
)

# ... run your code ...

# Save memory snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
```

Analyze the snapshot with the PyTorch memory visualizer:
```bash
python -m torch.cuda.memory.analysis_tool memory_snapshot.pickle
```

### 6.4 Using torch.profiler for Memory

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Print memory stats
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
```

### 6.5 Estimating Model Memory

```python
def estimate_model_memory(model, input_shape, batch_size=1, dtype=torch.float32):
    """Estimate memory usage for model training."""
    element_size = torch.tensor([], dtype=dtype).element_size()

    # 1. Model parameters
    param_memory = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"Parameters: {param_memory / 1e9:.3f} GB")

    # 2. Gradients (same size as parameters)
    grad_memory = param_memory
    print(f"Gradients:  {grad_memory / 1e9:.3f} GB")

    # 3. Optimizer state (Adam: 2x params for momentum + variance)
    optimizer_memory = 2 * param_memory  # For Adam
    print(f"Optimizer:  {optimizer_memory / 1e9:.3f} GB")

    # 4. Activations (rough estimate: forward pass saves activations)
    # This varies greatly by model; use per-layer tracking for accuracy
    activation_estimate = param_memory * 2  # Very rough estimate
    print(f"Activations (est): {activation_estimate / 1e9:.3f} GB")

    total = param_memory + grad_memory + optimizer_memory + activation_estimate
    print(f"\nTotal estimate: {total / 1e9:.3f} GB")
    print(f"Per sample:     {total / batch_size / 1e6:.1f} MB")
```

---

## 7. Gradient Checkpointing for Memory

### 7.1 Overview

Gradient checkpointing (also called activation recomputation or rematerialization) trades compute for memory. Instead of saving all intermediate activations during the forward pass, it saves only selected checkpoints. During the backward pass, the intermediate activations are recomputed from the checkpoints.

**Trade-off:**
- Memory reduction: ~50-70% less activation memory
- Compute overhead: ~30% more compute (one extra forward pass for recomputed segments)

### 7.2 torch.utils.checkpoint.checkpoint

```python
torch.utils.checkpoint.checkpoint(
    function: Callable,
    *args,
    use_reentrant: bool = True,
    context_fn: Optional[Callable[[], Tuple[ContextManager, ContextManager]]] = None,
    determinism_check: Optional[str] = None,
    debug: bool = False,
    **kwargs
) -> Any
```

**Parameters:**
- `function` (Callable): The function to checkpoint. Should be a part of the model that takes tensor inputs and returns tensor outputs.
- `*args`: Input arguments to `function`. Tensors will be tracked.
- `use_reentrant` (bool): Whether to use the reentrant autograd API. Default: `True` in older PyTorch, `False` recommended in PyTorch 2.0+.
- `context_fn` (Callable | None): A function returning a pair of context managers for the forward recomputation and backward passes.
- `determinism_check` (str | None): If set to `"default"`, checks for determinism between the original forward and the recomputed forward.
- `debug` (bool): If `True`, prints debug information.

```python
import torch
import torch.utils.checkpoint as cp

class CheckpointedBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # Instead of: return self.net(x)
        return cp.checkpoint(self.net, x, use_reentrant=False)
```

### 7.3 Checkpointing a Full Transformer

```python
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 4, d_model),
            torch.nn.Dropout(dropout),
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def _attn_block(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.dropout(attn_out)

    def _ffn_block(self, x):
        return self.ffn(x)

    def forward(self, x):
        # Checkpoint the attention block
        x = x + cp.checkpoint(self._attn_block, self.norm1(x), use_reentrant=False)
        # Checkpoint the FFN block
        x = x + cp.checkpoint(self._ffn_block, self.norm2(x), use_reentrant=False)
        return x

class TransformerModel(torch.nn.Module):
    def __init__(self, n_layers, d_model, n_heads):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### 7.4 checkpoint_sequential

```python
torch.utils.checkpoint.checkpoint_sequential(
    functions: Union[torch.nn.Sequential, List[Callable]],
    segments: int,
    input: torch.Tensor,
    use_reentrant: bool = True,
    **kwargs
) -> torch.Tensor
```

Checkpointing for sequential models. Divides the model into `segments` and checkpoints each segment.

**Parameters:**
- `functions` (Sequential | List[Callable]): The sequential model or list of functions.
- `segments` (int): Number of segments to divide the model into.
- `input` (Tensor): Input tensor.
- `use_reentrant` (bool): Whether to use reentrant API.

```python
# Simple sequential checkpointing
model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
)

# Checkpoint in 2 segments (each half is recomputed during backward)
x = torch.randn(32, 512, device='cuda', requires_grad=True)
output = cp.checkpoint_sequential(model, segments=2, input=x)
```

### 7.5 Non-Reentrant Checkpointing (Recommended)

```python
# PyTorch 2.0+ recommends use_reentrant=False
# It supports:
# - Inputs that are not tensors (int, str, etc.)
# - Keyword arguments
# - TorchDynamo compatibility
# - Better error messages

def custom_block(x, scale_factor=1.0):
    return x * scale_factor + torch.relu(x)

# With use_reentrant=False, kwargs work
output = cp.checkpoint(
    custom_block,
    x,
    use_reentrant=False,
    scale_factor=2.0
)
```

---

## 8. set_per_process_memory_fraction

### 8.1 API

```python
torch.cuda.set_per_process_memory_fraction(
    fraction: float,
    device: Optional[Union[int, torch.device]] = None
) -> None
```

Sets the memory fraction that a single process can use for a given GPU. The fraction is clamped to the range [0.0, 1.0]. If the total allocated memory exceeds this fraction, an OOM error will be raised.

**Parameters:**
- `fraction` (float): Fraction of total GPU memory (0.0 to 1.0).
- `device` (int | torch.device | None): Device. Default: current device.

```python
# Limit this process to 80% of GPU memory
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

# Now the process can only use 80% of GPU 0's total memory
x = torch.randn(10000, 10000, device='cuda:0')  # OK
# If total memory exceeds 80%, OOM is raised
```

**Use case:** Multi-tenant GPU sharing, preventing one process from consuming all GPU memory.

```python
# In a multi-process setup
import torch
import torch.multiprocessing as mp

def worker(gpu_id, fraction):
    torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_id)
    # This process is limited to `fraction` of GPU memory
    model = MyModel().cuda(gpu_id)
    # ... training ...

# Each of 4 processes gets 25% of GPU 0
mp.spawn(worker, args=(0.25,), nprocs=4)
```

---

## 9. CPU Memory Management

### 9.1 CPU Memory Allocation

PyTorch tensors on CPU use standard memory allocation (malloc/free). There is no caching allocator for CPU memory by default.

```python
# CPU tensor allocation
x = torch.randn(1000, 1000)  # Uses ~4 MB of CPU RAM

# Check CPU memory (OS-level, not PyTorch-specific)
import psutil
process = psutil.Process()
print(f"RSS: {process.memory_info().rss / 1e9:.2f} GB")
```

### 9.2 torch.cuda.mem_get_info

```python
torch.cuda.mem_get_info(
    device: Optional[Union[int, torch.device]] = None
) -> Tuple[int, int]
```

Returns tuple `(free_memory, total_memory)` for the given GPU device in bytes. This queries the CUDA driver directly.

```python
free, total = torch.cuda.mem_get_info()
used = total - free
print(f"GPU Memory: {used / 1e9:.1f} / {total / 1e9:.1f} GB ({100*used/total:.1f}% used)")
```

### 9.3 Shared Memory for Multiprocessing

```python
# Move tensor to shared memory for multiprocessing
tensor = torch.randn(100, 100)
shared_tensor = tensor.share_memory_()

# Or use storage directly
storage = torch.FloatStorage(10000).share_memory_()
```

### 9.4 Memory-Mapped Tensors

```python
# Create memory-mapped tensor (data stays on disk)
large_tensor = torch.randn(100000, 100000)  # 40 GB on disk
torch.save(large_tensor, 'large_tensor.pt')

# Load as memory-mapped (doesn't load into RAM)
mapped_tensor = torch.load('large_tensor.pt', mmap_mode='r')

# Can also use numpy memory mapping
import numpy as np
np_array = np.memmap('data.npy', dtype='float32', mode='r', shape=(100000, 100000))
tensor = torch.from_numpy(np_array)
```

### 9.5 Reducing CPU Memory

```python
# 1. Use generators instead of loading all data
def data_generator(data_path):
    for sample in load_samples(data_path):
        yield process(sample)

# 2. Use appropriate dtypes
x = torch.randn(1000, 1000, dtype=torch.float32)  # 4 MB
x_half = x.to(torch.float16)                        # 2 MB
x_int8 = x.to(torch.int8)                           # 1 MB

# 3. Delete references explicitly
import gc
large_object = create_large_object()
result = process(large_object)
del large_object
gc.collect()

# 4. Use gradient accumulation instead of large batches
```

---

## 10. Memory Optimization Recipes

### 10.1 Training Large Models

```python
def train_large_model():
    model = LargeModel().cuda()

    # 1. Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # 2. Use mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # 3. Use gradient accumulation
    accum_steps = 8
    optimizer = torch.optim.AdamW(model.parameters())

    for i, batch in enumerate(dataloader):
        with torch.amp.autocast('cuda'):
            output = model(batch)
            loss = output.loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### 10.2 Memory Cleanup Between Epochs

```python
def training_loop(model, train_loader, epochs):
    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Frees gradient memory

        # End of epoch cleanup
        del loss
        torch.cuda.empty_cache()
        gc.collect()

        # Validation
        model.eval()
        with torch.no_grad():
            validate(model, val_loader)
```

### 10.3 zero_grad(set_to_none=True)

```python
# More memory-efficient than zero_grad()
# Sets gradients to None instead of zero tensors
optimizer.zero_grad(set_to_none=True)

# This is the default in PyTorch 2.0+
# It saves memory because None takes no space vs a zero tensor
```

### 10.4 Monitoring Memory During Training

```python
class MemoryMonitor:
    def __init__(self):
        self.history = []

    def record(self, phase: str):
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        self.history.append({
            'phase': phase,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
        })

    def report(self):
        peak = max(h['allocated_gb'] for h in self.history)
        print(f"Peak allocated: {peak:.2f} GB")
        for h in self.history:
            print(f"  {h['phase']:20s}: "
                  f"allocated={h['allocated_gb']:.2f} GB, "
                  f"reserved={h['reserved_gb']:.2f} GB")
```

---

## 11. Troubleshooting Memory Issues

### 11.1 Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Fragmentation | OOM despite low allocated memory | Set `max_split_size_mb`, enable `expandable_segments` |
| Memory leak | Memory grows unboundedly | Check for reference cycles, use `gc.collect()` |
| Large optimizer | OOM from optimizer state | Use 8-bit optimizers, SGD instead of Adam |
| Large activations | OOM during forward | Gradient checkpointing, mixed precision |
| DataLoader memory | High CPU memory | Use `pin_memory=True`, streaming dataset |

### 11.2 Debugging Memory Leaks

```python
import gc

def debug_memory_leak():
    # Take snapshot of tensors
    torch.cuda.empty_cache()
    gc.collect()

    before = torch.cuda.memory_allocated()

    # Run suspect code
    for _ in range(100):
        output = model(input)

    gc.collect()
    after = torch.cuda.memory_allocated()

    leaked = after - before
    if leaked > 0:
        print(f"Leaked {leaked / 1e6:.1f} MB")

        # Find tensors holding memory
        tensors = []
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((obj.shape, obj.dtype, obj.size()))

        for shape, dtype, size in sorted(tensors, key=lambda x: -x[2])[:10]:
            print(f"  {shape} {dtype}: {size / 1e6:.1f} MB")
```
