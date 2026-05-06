# Profiling

## Overview

PyTorch provides comprehensive profiling tools for analyzing model performance, identifying bottlenecks, and optimizing both CPU and GPU execution. The modern profiler is based on the Kineto profiler and integrates with TensorBoard for visualization.

**Source location**: `torch/csrc/profiler/`, `torch/profiler/`

---

## torch.profiler.profile

The main profiling context manager that records operator execution, memory usage, and timing information.

### Signature

```python
torch.profiler.profile(
    activities: Optional[Iterable[ProfilerActivity]] = None,
    schedule: Optional[Callable[[int], ProfilerAction]] = None,
    on_trace_ready: Optional[Callable[[profile], None]] = None,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    with_flops: bool = False,
    with_modules: bool = False,
    emit_nvtx: bool = False,
    experimental_config: Optional[_ExperimentalConfig] = None,
) -> torch.profiler.profile
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activities` | `Iterable[ProfilerActivity]` | All | Which activities to profile (CPU, CUDA) |
| `schedule` | `Callable` | None | Profiling schedule (wait/warmup/active cycles) |
| `on_trace_ready` | `Callable` | None | Callback when trace is ready |
| `record_shapes` | `bool` | False | Record tensor shapes for each op |
| `profile_memory` | `bool` | False | Track memory allocation/deallocation |
| `with_stack` | `bool` | False | Record Python call stack for each op |
| `with_flops` | `bool` | False | Estimate FLOPs for each op |
| `with_modules` | `bool` | False | Record module hierarchy information |
| `emit_nvtx` | `bool` | False | Emit NVTX ranges for NVIDIA Nsight |
| `experimental_config` | `_ExperimentalConfig` | None | Experimental configuration options |

### ProfilerActivity

```python
from torch.profiler import ProfilerActivity

# Available activities
ProfilerActivity.CPU    # Profile CPU operations
ProfilerActivity.CUDA   # Profile CUDA operations
ProfilerActivity.MTIA   # Profile MTIA operations
ProfilerActivity.XPU    # Profile XPU operations

# Common combinations
activities = [ProfilerActivity.CPU]                        # CPU only
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]  # CPU + CUDA
```

---

## Basic Profiling

### Simple Example

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torchvision.models.resnet18(pretrained=True)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

# Profile a single forward pass
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with record_function("model_inference"):
        model(input_tensor)

# Print results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### Output Format

```
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     # of Calls  CPU Mem      Self CPU Mem
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                 model_inference        0.01%      10.000us       100.00%     100.000ms     100.000ms             1           0 b           0 b
                    aten::conv2d        0.05%      50.000us        45.00%      45.000ms       5.625ms             8     512.00 Kb      12.00 Kb
               aten::batch_norm        0.03%      30.000us        30.00%      30.000ms       3.750ms             8     256.00 Kb       8.00 Kb
                      aten::relu        0.02%      20.000us        10.00%      10.000ms       1.250ms             8     128.00 Kb       4.00 Kb
                    aten::addmm        0.01%      10.000us         5.00%       5.000ms       5.000ms             1      64.00 Kb       2.00 Kb
                   aten::adaptive        0.01%       5.000us         3.00%       3.000ms       3.000ms             1      32.00 Kb       1.00 Kb
...
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 100.000ms
```

---

## torch.profiler.record_function

Creates a user-defined named range in the profiler output.

### Signature

```python
torch.profiler.record_function(name: str) -> contextmanager
```

### Usage

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function("data_preprocessing"):
        data = preprocess(raw_data)

    with record_function("model_forward"):
        output = model(data)

    with record_function("loss_computation"):
        loss = criterion(output, target)

    with record_function("backward_pass"):
        loss.backward()

    with record_function("optimizer_step"):
        optimizer.step()

# See timing for each custom range
print(prof.key_averages().table(sort_by="cpu_time_total"))
```

---

## torch.profiler.tensorboard_trace_handler

Exports profiling results to TensorBoard-compatible format.

### Signature

```python
torch.profiler.tensorboard_trace_handler(
    dir_name: str,
    worker_name: Optional[str] = None,
    use_gzip: bool = False,
) -> Callable
```

### Usage

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

# Profile with TensorBoard export
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./log_dir"),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    for epoch in range(5):
        for batch in dataloader:
            # Training step
            prof.step()  # notify profiler of step boundary

# View in TensorBoard:
# tensorboard --logdir=./log_dir
```

### TensorBoard View

The TensorBoard plugin displays:
- **Overview**: Summary of time spent in each category
- **Operator View**: Per-operator timing breakdown
- **GPU Kernel View**: CUDA kernel execution times
- **Trace View**: Timeline visualization of CPU and GPU activity
- **Memory View**: Memory allocation timeline
- **Python Stack**: Call stack for expensive operations

---

## Schedule

### Signature

```python
torch.profiler.schedule(
    wait: int = 1,
    warmup: int = 1,
    active: int = 1,
    repeat: int = 0,
    skip_first: int = 0,
) -> Callable[[int], ProfilerAction]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `wait` | `int` | Steps to wait before starting warmup |
| `warmup` | `int` | Warmup steps (profiling active but low overhead) |
| `active` | `int` | Active profiling steps (full overhead) |
| `repeat` | `int` | Number of wait/warmup/active cycles (0=infinite) |
| `skip_first` | `int` | Steps to skip at the beginning |

### Schedule Actions

```python
from torch.profiler import ProfilerAction

ProfilerAction.NONE       # Not profiling
ProfilerAction.WARMUP     # Warming up (reduced profiling)
ProfilerAction.RECORD     # Active profiling
ProfilerAction.RECORD_AND_SAVE  # Profile and save trace
```

### Usage

```python
# Profile steps 3-5 (skip 1, warmup 1, active 3)
my_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=tensorboard_trace_handler("./logs"),
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()  # advance schedule
```

---

## Profiler Results

### key_averages()

Returns averaged events grouped by operator name.

```python
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    model(input)

# Key averages grouped by operator name
avg = prof.key_averages()

# Table output
print(avg.table(
    sort_by="cpu_time_total",  # sort column
    row_limit=20,              # max rows to display
    header=True,               # show header
))

# Available sort_by options:
# "cpu_time", "cpu_time_total", "cuda_time", "cuda_time_total",
# "cpu_memory_usage", "cuda_memory_usage", "self_cpu_memory_usage",
# "self_cuda_memory_usage", "flops", "key"
```

### events()

Returns individual profiling events (not averaged).

```python
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    model(input)

# Iterate over all events
for event in prof.events():
    print(f"Op: {event.key}, CPU time: {event.cpu_time_total}us")
    if event.shapes:
        print(f"  Shapes: {event.shapes}")
```

### Event Attributes

```python
class FunctionEvent:
    # Identity
    key: str                    # operator name
    id: int                     # unique event ID
    node_id: int                # node ID for distributed profiling
    thread: int                 # thread ID
    start_thread: int           # start thread
    end_thread: int             # end thread
    fwd_thread: int             # forward thread
    category: int               # event category

    # Timing (microseconds)
    cpu_time: int               # self CPU time
    cpu_time_total: int         # total CPU time (including children)
    cuda_time: int              # self CUDA time
    cuda_time_total: int        # total CUDA time (including children)
    self_cpu_time: int          # CPU time excluding children
    cpu_children: List          # child events

    # Memory
    cpu_memory_usage: int       # CPU memory change
    self_cpu_memory_usage: int  # self CPU memory change
    cuda_memory_usage: int      # CUDA memory change
    self_cuda_memory_usage: int # self CUDA memory change

    # Shapes
    shapes: List[List[int]]     # input tensor shapes
    input_shapes: List[List[int]]

    # Stack
    stack: List[str]            # Python call stack
    scope: int                  # scope ID

    # FLOPs
    flops: Optional[int]        # estimated FLOPs

    # Module
    module_hierarchy: Optional[str]  # module path (e.g., "model.layer1.conv")

    # CUDA correlation
    cuda_device: int            # CUDA device index
    correlation_id: int         # correlation with CUDA events
    stream: int                 # CUDA stream

    # Count
    count: int                  # number of calls
```

### table() Method

```python
# Available table formats
avg.table(
    sort_by="cpu_time_total",
    row_limit=10,
    header=True,
    max_name_column_width=80,
    max_shapes_column_width=80,
    max_src_column_width=80,
)

# Export to different formats
print(avg.table(sort_by="cpu_time_total"))       # text table
prof.export_chrome_trace("trace.json")             # Chrome trace format
prof.export_stacks("stacks.txt", "self_cpu_time")  # stack traces
```

---

## TensorBoard Integration

### Setup

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

# Install torch-tb-profiler
# pip install torch-tb-profiler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler(
        dir_name="./tb_logs",
        worker_name="worker0",
        use_gzip=False,
    ),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_stack=True,
) as prof:
    for epoch in range(10):
        for batch in train_loader:
            train_step(batch)
            prof.step()

# Launch TensorBoard
# tensorboard --logdir=./tb_logs
```

### TensorBoard Views

1. **Overview Page**: High-level summary
   - GPU utilization percentage
   - Time breakdown by category (kernel, memcpy, etc.)
   - Top operations by time

2. **Operator View**: Per-operator analysis
   - Sortable table of all operators
   - Time, memory, FLOPs per operator
   - Input shapes and call counts

3. **GPU Kernel View**: CUDA-specific
   - Per-kernel execution time
   - Kernel registration and launch overhead
   - SM (Streaming Multiprocessor) utilization

4. **Trace View**: Timeline
   - Visual timeline of CPU and GPU operations
   - Async launch and execution gaps
   - Memory operations overlaid

5. **Memory View**: Memory analysis
   - Allocation timeline
   - Peak memory usage
   - Memory by allocator type

6. **Distributed View**: Multi-GPU/multi-node
   - Communication operations
   - Overlap of compute and communication
   - Bandwidth utilization

---

## Kineto Profiler Integration

Kineto is the underlying profiler library that PyTorch uses for GPU profiling.

### Kineto Configuration

```python
from torch.profiler import _ExperimentalConfig

experimental_config = _ExperimentalConfig(
    verbose=True,                    # verbose logging
    enable_cuda_sync_event_logging=True,  # log CUDA sync events
    record_mtia_events=False,        # record MTIA events
    record_dt_events=False,          # record DT events
    enable_cpu_execution_trace=False, # CPU execution trace
    cur_epoch=0,                     # current epoch for scheduling
    cur_epoch_steps=0,               # steps in current epoch
    max_cpu_execution_trace_records=50000,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    experimental_config=experimental_config,
) as prof:
    model(input)
```

---

## Legacy Profiler: torch.autograd.profiler

### profile

```python
# Legacy profiler (pre-Kineto)
with torch.autograd.profiler.profile(
    use_cuda=True,
    record_shapes=True,
    profile_memory=True,
    use_kineto=False,       # use legacy implementation
) as prof:
    output = model(input)
    loss.backward()

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```

### emit_nvtx

Emits NVIDIA Tools Extension (NVTX) ranges for use with NVIDIA Nsight profiler.

```python
# Use with NVIDIA Nsight Systems
# nsys profile -t cuda,osrt,nvtx -o profile_report python my_script.py

with torch.autograd.profiler.emit_nvtx():
    output = model(input)
    loss.backward()

# Or with the new profiler
with profile(activities=[ProfilerActivity.CUDA], emit_nvtx=True):
    output = model(input)
```

---

## Memory Profiling

### torch.cuda.memory_summary()

```python
import torch

# Comprehensive memory summary
print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Output:
# |===========================================================================|
# |                  PyTorch CUDA memory summary, device ID 0                |
# |---------------------------------------------------------------------------|
# |            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
# |===========================================================================|
# |        Metric         | Usage / Total |      PyTorch Limit     |  Percent |
# |---------------------------------------------------------------------------|
# |  GPU Allocated Memory |   1024.00 MB  |    1024.00 MB          |   100%   |
# |       GPU Memory Use  |   2048.00 MB  |    4096.00 MB          |    50%   |
# |===========================================================================|
```

### torch.cuda.memory_stats()

```python
# Detailed memory statistics
stats = torch.cuda.memory_stats()

# Key statistics:
stats["allocated_bytes.all.current"]     # currently allocated bytes
stats["allocated_bytes.all.peak"]        # peak allocated bytes
stats["allocated_bytes.all.freed"]       # total freed bytes
stats["reserved_bytes.all.current"]      # currently reserved bytes
stats["reserved_bytes.all.peak"]         # peak reserved bytes
stats["active_bytes.all.current"]        # active bytes (allocated - freed)
stats["num_alloc_retries"]               # number of allocation retries
stats["num_ooms"]                        # number of OOM events
```

### Memory Snapshot

```python
# Detailed memory allocation trace
torch.cuda.memory._snapshot()

# Save memory snapshot for analysis
snapshot = torch.cuda.memory._snapshot()
from pickle import dump
with open("memory_snapshot.pickle", "wb") as f:
    dump(snapshot, f)

# Visualize at https://pytorch.org/memory_viz
```

### Reset Memory Stats

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()
torch.cuda.empty_cache()  # release cached memory back to CUDA
```

### Per-Allocation Tracking

```python
# Track where allocations happen
import torch.cuda.memory as tcm

# Set memory profiler hooks
tcm._record_memory_history(
    enabled="stack+data",  # record stacks and tensor data
    stacks="python",
)

# ... run model ...

# Get memory snapshot
snapshot = tcm._snapshot()

# Stop recording
tcm._record_memory_history(enabled=None)
```

---

## torch.utils.benchmark.Timer

High-precision benchmarking utility for comparing code performance.

### Signature

```python
from torch.utils.benchmark import Timer

timer = Timer(
    stmt="model(x)",               # statement to benchmark
    setup="import torch; model = torch.nn.Linear(100, 100); x = torch.randn(32, 100)",
    globals=None,                   # global variables
    label=None,                     # human-readable label
    sub_label=None,                 # sub-label for comparison
    description=None,               # description
    env=None,                       # environment description
    num_threads=1,                  # number of threads
    language=Language.PYTHON,       # Python or C++
)
```

### Usage

```python
from torch.utils.benchmark import Timer

# Benchmark a single operation
timer = Timer(
    stmt="torch.mm(a, b)",
    setup="import torch; a = torch.randn(100, 100); b = torch.randn(100, 100)",
)
result = timer.timeit(100)
print(result)
# <torch.utils.benchmark.utils.common.Measurement object>
# Median: 123.45 us
# IQR:    2.34 us (122.78, 125.12)

# Compare multiple approaches
results = []
for size in [64, 128, 256, 512]:
    t = Timer(
        stmt="torch.mm(a, b)",
        setup=f"import torch; a = torch.randn({size}, {size}); b = torch.randn({size}, {size})",
        sub_label=f"size={size}",
    )
    results.append(t.timeit(50))

# Compare results
from torch.utils.benchmark import Compare
compare = Compare(results)
compare.print()
```

### Compare Output

```
[----------------------------- mm -----------------------------]
                   |  size=64  |  size=128  |  size=256  |  size=512
1 threads: -----------------  ----------  ----------  ----------
               |      12.3  |       98.4  |      789.5  |    6234.1
               |     (1.2)  |     (3.4)   |    (12.3)   |    (45.6)

Times are in microseconds (us).
```

### Advanced Timer Usage

```python
# Using globals
a = torch.randn(100, 100)
b = torch.randn(100, 100)

timer = Timer(
    stmt="torch.mm(a, b)",
    globals={"a": a, "b": b},
)
result = timer.blocked_autorange(min_run_time=1.0)
print(f"Median: {result.median * 1e6:.2f} us")
print(f"Mean: {result.mean * 1e6:.2f} us")
print(f"IQR: {result.iqr * 1e6:.2f} us")
```

---

## torch.profiler._memory_profiler

Low-level memory profiler for tracking tensor allocations.

```python
# Experimental memory profiling
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    model(input_tensor)

# Analyze memory
for event in prof.events():
    if event.cpu_memory_usage != 0:
        print(f"{event.key}: {event.cpu_memory_usage / 1024:.1f} KB")
        if event.stack:
            for frame in event.stack[:3]:
                print(f"  {frame}")
```

---

## Flame Graph Generation

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU],
    with_stack=True,
) as prof:
    model(input)

# Export stacks as flame graph data
prof.export_stacks(
    "/tmp/profiler_stacks.txt",
    metric="self_cpu_time",  # or "cpu_time"
)

# The output format is compatible with flamegraph.pl:
# symbol_name stack_trace; count
# model;forward;conv2d;addmm 12345
# model;forward;relu;threshold_kernel 6789

# Generate flame graph:
# git clone https://github.com/brendangregg/FlameGraph
# FlameGraph/flamegraph.pl /tmp/profiler_stacks.txt > flamegraph.svg
```

### Speedscope Integration

```python
# Export as Chrome trace (compatible with speedscope)
prof.export_chrome_trace("trace.json")

# Open in speedscope:
# https://www.speedscope.app/
# Or: npx speedscope trace.json
```

---

## Example: Profile a Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, schedule, tensorboard_trace_handler, ProfilerActivity

# Setup
model = torchvision.models.resnet18(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define schedule
prof_schedule = schedule(
    wait=2,      # skip first 2 steps
    warmup=1,    # 1 warmup step
    active=3,    # profile 3 steps
    repeat=1,    # do this once
)

# Profile training loop
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./logs/train_profile"),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_stack=True,
    with_modules=True,
) as prof:
    for step, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        with record_function("forward"):
            output = model(data)

        with record_function("loss"):
            loss = criterion(output, target)

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

        prof.step()

        if step >= 10:
            break

# Print summary
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=15,
))
```

---

## Example: Profile Inference

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torchvision.models.resnet18(pretrained=True)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

# Warmup (important for accurate timing)
with torch.no_grad():
    for _ in range(10):
        model(input_tensor)

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    with torch.no_grad():
        for _ in range(100):
            model(input_tensor)

# Analyze results
print("=" * 80)
print("TOP OPERATIONS BY CPU TIME:")
print("=" * 80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("\n" + "=" * 80)
print("TOP OPERATIONS BY CUDA TIME:")
print("=" * 80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\n" + "=" * 80)
print("TOP OPERATIONS BY FLOPS:")
print("=" * 80)
print(prof.key_averages().table(sort_by="flops", row_limit=10))

print("\n" + "=" * 80)
print("TOP OPERATIONS BY MEMORY:")
print("=" * 80)
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# Export traces
prof.export_chrome_trace("inference_trace.json")
prof.export_stacks("inference_stacks.txt", "self_cuda_time")
```

---

## Performance Analysis Best Practices

### 1. Warmup Before Profiling

```python
# Always warm up before profiling to avoid:
# - JIT compilation overhead
# - CUDA context initialization
# - Memory allocation patterns not yet stabilized
with torch.no_grad():
    for _ in range(10):
        model(dummy_input)
```

### 2. Use Appropriate Schedule

```python
# For training: use schedule to profile specific steps
schedule(wait=5, warmup=2, active=3)

# For inference: profile a batch of inferences
# (single inference may be too fast for accurate timing)
with profile(...) as prof:
    for _ in range(100):
        model(input)
```

### 3. Minimize Profiling Overhead

```python
# Only enable what you need:
# record_shapes: adds ~10% overhead
# profile_memory: adds ~20% overhead
# with_stack: adds ~30% overhead
# with_flops: adds ~5% overhead

# For quick profiling:
with profile(activities=[ProfilerActivity.CPU]) as prof:
    model(input)

# For detailed analysis:
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    model(input)
```

### 4. Check GPU Utilization

```python
# If cuda_time_total << cpu_time_total:
#   -> GPU is underutilized
#   -> Consider larger batch size or more data parallelism

# If cpu_time_total is dominated by data loading:
#   -> Use DataLoader with num_workers > 0
#   -> Use pin_memory=True for CUDA

# If cuda_time has large gaps:
#   -> CPU-GPU synchronization bottlenecks
#   -> Consider CUDA graphs or reduce sync points
```

### 5. Compare Optimizations

```python
from torch.utils.benchmark import Timer, Compare

results = []
for batch_size in [16, 32, 64, 128]:
    for precision in ["fp32", "fp16", "bf16"]:
        t = Timer(
            stmt="model(x)",
            setup=f"""
import torch
model = torchvision.models.resnet18().cuda().eval()
x = torch.randn({batch_size}, 3, 224, 224, device='cuda')
if '{precision}' == 'fp16':
    model = model.half()
    x = x.half()
elif '{precision}' == 'bf16':
    model = model.to(torch.bfloat16)
    x = x.to(torch.bfloat16)
""",
            sub_label=f"bs={batch_size}",
            description=precision,
        )
        results.append(t.blocked_autorange(min_run_time=1.0))

compare = Compare(results)
compare.print()
```

---

## Summary

PyTorch's profiling tools provide:

1. **torch.profiler.profile**: Main profiler with CPU, CUDA, memory, and FLOPs tracking
2. **record_function**: Custom named ranges for profiling sections
3. **schedule**: Wait/warmup/active cycles for controlled profiling
4. **TensorBoard integration**: Visual analysis via tensorboard_trace_handler
5. **Event analysis**: key_averages(), events(), table() for result examination
6. **Export formats**: Chrome trace, flame graph stacks
7. **Memory profiling**: cuda.memory_summary(), memory_stats(), memory snapshots
8. **Benchmarking**: torch.utils.benchmark.Timer for precise timing comparisons
9. **NVTX integration**: emit_nvtx for NVIDIA Nsight Systems
10. **Best practices**: warmup, minimal overhead, GPU utilization analysis
