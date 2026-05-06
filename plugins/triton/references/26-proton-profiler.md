# Chapter 26: Proton Profiler

Proton is Triton's profiling system for measuring GPU kernel performance.

## Installation

Proton is included with Triton by default (`TRITON_BUILD_PROTON=ON`).

## Quick Start

```python
import triton.profiler as proton

# Start profiling
proton.start("profile_name", hook="triton")

# Run your code
kernel[grid](args)

# Stop profiling
proton.finalize()

# View results
proton.viewer.print_tree()
```

## Profile API

### `proton.start(name, *, hook=None, data="tree", context="shadow")`

Start a profiling session.

**Parameters:**
- `name` (str): Profile name/output file path
- `hook` (str): Hook type ("triton" for Triton kernels)
- `data` (str): Data format ("tree" or "trace")
- `context` (str): Context mode ("shadow" or "standalone")

### `proton.finalize(session=None)`

Finalize and save profiling session.

### `proton.activate(session=None)`

Activate a profiling session.

### `proton.deactivate(session=None, *, flushing=False)`

Deactivate a profiling session.

## Scope API

### `proton.scope(name)`

Context manager for profiling scopes:

```python
with proton.scope("matmul"):
    result = matmul(a, b)

with proton.scope("activation"):
    result = torch.relu(result)
```

### `proton.enter_scope(name) / proton.exit_scope()`

Manual scope management:

```python
proton.enter_scope("my_section")
# ... code ...
proton.exit_scope()
```

## Profiling Data

### `proton.get() -> dict`

Get profiling data as JSON-compatible dict.

### `proton.get_msgpack() -> bytes`

Get profiling data in MessagePack format.

### `proton.advance_phase()`

Advance to next profiling phase.

### `proton.clear()`

Clear profiling data.

## Profiling Modes

```python
from triton.profiler import mode

# Default mode - instrumentation-based profiling
proton.start("default_profile", hook="triton")

# PC Sampling mode
proton.start("pc_sampling_profile", hook="triton", mode="pc_sampling")
```

### Available Modes

| Mode | Description |
|------|-------------|
| `Default` | Standard instrumentation profiling |
| `MMA` | Matrix multiply acceleration profiling |
| `PCSampling` | Program counter sampling |

## Language API

```python
from triton.profiler import language

# Record custom profiling events
language.record("custom_event", value=42)

# Enable/disable semantic profiling
language.enable_semantic()
language.disable_semantic()
```

## Metric System

```python
from triton.profiler import metric

# Set custom metric kernels
metric.set_metric_kernels(kernels)

# Transform tensor metrics
metric.transform_tensor_metrics(metrics)
```

## Hardware Specifications

```python
from triton.profiler import specs

# Get peak FLOPS for device
flops = specs.max_flops(device)

# Get peak bandwidth
bps = specs.max_bps(device)
```

## Hooks

### Launch Hook

```python
from triton.profiler.hooks.launch import LaunchHook

# Automatically profiles kernel launches
hook = LaunchHook()
hook.configure(
    include_patterns=["my_kernel"],
    exclude_patterns=["debug_kernel"],
)
```

### Instrumentation Hook

```python
from triton.profiler.hooks.instrumentation import InstrumentationHook

# Hardware-based instrumentation profiling
hook = InstrumentationHook()
hook.activate()
# ... run kernels ...
hook.deactivate()
```

## Viewer

```bash
# Command-line viewer
proton-viewer profile.sqlite
```

### Python Viewer

```python
from triton.profiler import viewer

# Read profile file
db = viewer.read("profile.sqlite")

# Parse and display
viewer.parse(db)

# Print performance tree
viewer.print_tree(db)
```

## CLI

```bash
# Profile a Python script
proton python my_script.py

# Profile a pytest test
proton pytest test_file.py::test_name
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TRITON_PROTON_DISABLE` | Disable Proton (default: false) |
| `TRITON_CUPTI_LIB_PATH` | CUPTI library path |
| `TRITON_CUPTI_LIB_BLACKWELL_PATH` | CUPTI Blackwell path |
| `TRITON_PROFILE_BUFFER_SIZE` | Profile buffer size |
| `TRITON_PROFILE_METRIC_BUFFER_SIZE` | Metric buffer size |
| `TRITON_ENABLE_NVTX` | Enable NVTX annotations |
| `TRITON_ENABLE_HW_TRACE` | Enable hardware trace (Blackwell+) |

## Integration with Testing

```python
import triton.testing as testing

# Benchmark with Proton
ms = testing.do_bench_proton(kernel_fn, warmup=25, rep=100)

# Benchmark with CUDA graphs
ms = testing.do_bench_cudagraph_proton(kernel_fn, rep=20)
```
