# 16 - Profiler

## Overview

xFormers provides a multi-backend profiler for training deep learning models. It combines memory snapshots, Nsight profiling, PyTorch profiler, and DCGM profiling into a unified interface with automatic scheduling.

**Source**: `xformers/profiler/`

## API Reference

### `profile`

```python
import xformers.profiler

profiler = xformers.profiler.profile(
    output_dir: str,                    # Directory for profiler output
    module: Optional[nn.Module] = None, # Model to profile
    schedule: Sequence[Tuple[Any, int, int]] = DEFAULT_SCHEDULE,
)
```

Creates a profiler that runs on the first ~20 steps of training.

**Default schedule:**
```python
DEFAULT_SCHEDULE = (
    (MemSnapshotsProfiler, 0, 2),      # Steps 0-1: Memory snapshots
    (NsightProfiler, 4, 6),            # Steps 4-5: Nsight Systems
    (PyTorchProfiler, 6, 7),           # Step 6: Full PyTorch profiler
    (PyTorchProfiler_CUDAOnly, 7, 8),  # Step 7: CUDA-only profiler
)
```

**Usage as context manager:**
```python
with xformers.profiler.profile("profile_data", module=model) as prof:
    for i in range(20):
        model(inputs).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        xformers.profiler.step()
```

**Usage with start/stop:**
```python
prof = xformers.profiler.profile("profile_data", module=model)
prof.start()
for i in range(20):
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    prof.step()
prof.stop()
```

### `step`

```python
xformers.profiler.step()
```

Signals the profiler that a training step has completed. Must be called after each iteration.

## Profiler Backends

### `MemSnapshotsProfiler`

Captures memory allocation/deallocation traces for tensor memory analysis.

**Output**: HTML file with memory trace visualization (uses `torch.cuda._memory_viz.trace_plot`).

**Features:**
- Records up to 100,000 allocation/free events
- Records stack traces for each allocation
- Generates interactive HTML visualization
- No GPU overhead

### `NsightProfiler`

Triggers NVIDIA Nsight Systems profiling during specified steps.

**Requirements:**
- Script must be launched with `nsys profile`
- Must use `--capture-range=cudaProfilerApi` flag

```bash
nsys profile --capture-range=cudaProfilerApi -o profile python train.py
```

**Features:**
- Full GPU kernel tracing
- CUDA API tracing
- Memory transfer tracing
- Zero Python overhead (hardware-level profiling)

### `PyTorchProfiler`

Full PyTorch profiler with CPU and CUDA tracing.

**Activities:**
```python
ACTIVITIES = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]
```

**Output:**
- Chrome trace files (`.pt.trace.json.gz`)
- Kernel timing CSV files
- MFU/HFU analysis

**Analysis includes:**
- Step time (ms)
- TFlop/step
- TFlops (throughput)
- HFU (Hardware FLOP Utilization)
- MFU (Model FLOP Utilization)

### `PyTorchProfiler_CUDAOnly`

Low-overhead variant that only profiles CUDA kernels. No step time or MFU analysis.

### `DCGMProfiler`

DCGM (Data Center GPU Manager) profiling for power, temperature, and utilization metrics. Currently disabled in default schedule due to startup latency.

## Profile Analysis

### MFU and HFU Computation

The profiler computes:

- **MFU (Model FLOP Utilization)**: Ratio of actual model FLOPs to hardware peak FLOPs
- **HFU (Hardware FLOP Utilization)**: Ratio of all compute FLOPs (including framework overhead) to hardware peak

```python
from xformers.profiler.profile_analyzer import AnalyzedTrace

results = AnalyzedTrace.from_profile(events)
hw_flops = {torch.float16: 312e12}  # A100 fp16 peak
hfu = results.compute_hfu(hw_flops)
mfu = results.compute_mfu(hw_flops)
```

### Device Limits

`device_limits.py` provides peak FLOP rates for different GPUs:

```python
from xformers.profiler.device_limits import get_device_limits

limits = get_device_limits(torch.device("cuda"))
# Returns DeviceLimits with gemm_tflops per dtype
```

### Find Slowest

`find_slowest.py` identifies the slowest operations in a profiling trace:

```bash
python -m xformers.profiler.find_slowest
```

## Schedule Configuration

### Custom Schedule

```python
from xformers.profiler import profile, MemSnapshotsProfiler, PyTorchProfiler

custom_schedule = [
    (MemSnapshotsProfiler, 0, 2),
    (PyTorchProfiler, 2, 20),
]

with profile("profile_data", schedule=custom_schedule) as prof:
    for i in range(20):
        train_step()
        prof.step()
```

### Schedule Constraints

1. **No overlap**: Profiler periods must not overlap
2. **Non-negative start**: `begin >= 0`
3. **Positive end**: `end > 0`
4. **start < end**: Begin must be before end

### Manual Trigger

Create a file named `trigger.{step_number:09}` in the output directory to manually trigger profiling at a specific step:

```bash
# Trigger profiling at step 100
touch profile_data/trigger.000000100
```

The profiler checks for trigger files every 10 steps.

## Distributed Training

In distributed settings:
- Each rank creates its own trace files
- Files are organized into subdirectories per rank
- Worker names include rank, hostname, and PID:
  ```
  profile_data/
    memory_trace_plot/
      000001_rank00_gpu42_12345.html
      000001_rank01_gpu43_12346.html
  ```

## Output Structure

```
profile_data/
├── memory_trace_plot/
│   ├── 000001_rank00_hostname_pid.html
│   └── 000001_rank01_hostname_pid.html
├── profile_CPU_CUDA_000006/
│   ├── rank00_hostname_pid.timestamp.pt.trace.json.gz
│   ├── kernels_rank00_hostname_pid.timestamp.csv
│   ├── rank01_hostname_pid.timestamp.pt.trace.json.gz
│   └── kernels_rank01_hostname_pid.timestamp.csv
├── trigger.000000100  # Manual trigger file
└── ...
```

## Usage Example

### Full Training Profile

```python
import torch
import torch.nn as nn
import xformers.profiler

model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6).cuda()
optimizer = torch.optim.Adam(model.parameters())
inputs = torch.randn(32, 10, 512, device="cuda")

with xformers.profiler.profile(
    output_dir="profile_output",
    module=model,
    schedule=[
        (xformers.profiler.MemSnapshotsProfiler, 0, 2),
        (xformers.profiler.NsightProfiler, 4, 6),
        (xformers.profiler.PyTorchProfiler, 6, 20),
    ]
):
    for step in range(20):
        output = model(inputs, inputs)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        xformers.profiler.step()
```

## Internal Architecture

### `_Profiler`

Main profiler class that:
1. Manages the schedule
2. Creates/destroys backend profilers
3. Coordinates step counting
4. Generates summary output

```python
class _Profiler:
    _CURRENT_PROFILER = None  # Singleton

    def __init__(self, output_dir, schedule, module):
        self.done_steps = 0
        self.output_dir = Path(output_dir).absolute()
        self.profilers: List[_ProfilerState] = ...
        self.summary: List[Tuple[str, str]] = []
```

### `_ProfilerState`

```python
@dataclass
class _ProfilerState:
    cls: Any           # Profiler backend class
    iter_begin: int    # Start step
    iter_end: int      # End step
    object: Any = None # Instantiated profiler (created on first step)
```

Only one profiler can be active at a time (`_CURRENT_PROFILER` singleton).
