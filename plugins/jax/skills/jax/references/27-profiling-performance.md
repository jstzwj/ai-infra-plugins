# Chapter 27: Profiling and Performance Analysis

## Overview

JAX provides multiple profiling tools for analyzing computation performance, memory usage, and distributed communication. Understanding profiling is essential for identifying bottlenecks, optimizing throughput, and reducing memory consumption.

This chapter covers the complete profiling workflow: from basic timing to advanced trace analysis with Perfetto and XProf, memory profiling, GPU-specific analysis, and benchmarking best practices.

**Key Concept:** JAX dispatches operations asynchronously. Naive timing with `time.time()` produces misleading results. You must use `.block_until_ready()` or JAX's profiling tools for accurate measurements.

## Block Until Ready for Accurate Timing

### The Async Dispatch Problem

JAX operations are dispatched asynchronously to the accelerator. A Python `time.time()` measurement after a JAX call may return before the computation finishes:

```python
import time
import jax
import jax.numpy as jnp

x = jnp.ones((10000, 10000))

# WRONG: inaccurate timing
start = time.time()
y = jnp.dot(x, x)
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.6f}s")
# This prints a tiny number because dispatch is async!
# The actual computation hasn't finished yet.
```

### Correct Timing with block_until_ready

```python
import time
import jax
import jax.numpy as jnp

x = jnp.ones((10000, 10000))

# CORRECT: wait for computation to complete
start = time.time()
y = jnp.dot(x, x)
y.block_until_ready()  # Block until the result is ready
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.6f}s")
# Now the timing is accurate
```

### Timing JIT-Compiled Functions

The first call to a JIT-compiled function includes compilation time. Subsequent calls are much faster. Always include a warm-up:

```python
import time
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

x = jnp.ones((10000, 10000))

# Warm up: trigger compilation
_ = my_fn(x).block_until_ready()

# Measure execution time
times = []
for _ in range(10):
    start = time.perf_counter()
    result = my_fn(x)
    result.block_until_ready()
    times.append(time.perf_counter() - start)

import statistics
print(f"Mean: {statistics.mean(times):.6f}s")
print(f"Median: {statistics.median(times):.6f}s")
print(f"Std: {statistics.stdev(times):.6f}s")
```

### Timing Utility Function

```python
import time
import functools
import jax

def timeit(fn, *args, warmup=3, num_runs=10, **kwargs):
    """Time a JAX function with proper warmup and blocking."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (list, tuple)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (list, tuple)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        times.append(time.perf_counter() - start)

    import statistics
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0,
        "runs": num_runs,
    }

# Usage
@jax.jit
def matmul(x, y):
    return jnp.dot(x, y)

x = jnp.ones((4096, 4096))
y = jnp.ones((4096, 4096))
stats = timeit(matmul, x, y)
print(f"Mean time: {stats['mean']*1000:.2f}ms")
```

## jax.profiler.trace -- Context Manager Profiling

### Basic Trace Collection

`jax.profiler.trace` is a context manager that collects a performance trace for the code within its scope. The trace can be viewed with Perfetto or TensorBoard.

```python
import jax
import jax.numpy as jnp

# Create a trace directory
log_dir = "/tmp/jax_profiler_trace"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    # All operations within this block are profiled
    x = jnp.ones((4096, 4096))
    y = jnp.dot(x, x)
    z = jnp.sum(y)
    z.block_until_ready()

print(f"Trace saved to {log_dir}")
# A Perfetto link will be printed for online viewing
```

### Trace with Step Metadata

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_profiler_steps"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    x = jnp.ones((4096, 4096))

    for step in range(5):
        # Add step annotation for clarity in the trace
        with jax.profiler.TraceAnnotation(f"train_step_{step}"):
            y = jnp.dot(x, x)
            z = jax.nn.relu(y)
            loss = jnp.mean(z)
            loss.block_until_ready()

# View the trace to see each train_step clearly labeled
```

### Trace with Custom Annotations

```python
import jax
import jax.numpy as jnp

def profiled_training_step(params, x, y):
    log_dir = "/tmp/jax_profiler_annotated"

    with jax.profiler.trace(log_dir, create_perfetto_link=True):
        with jax.profiler.TraceAnnotation("forward_pass"):
            hidden = jnp.dot(x, params["w1"]) + params["b1"]
            hidden = jax.nn.gelu(hidden)
            output = jnp.dot(hidden, params["w2"]) + params["b2"]

        with jax.profiler.TraceAnnotation("loss_computation"):
            loss = jnp.mean((output - y) ** 2)

        with jax.profiler.TraceAnnotation("backward_pass"):
            grads = jax.grad(lambda p: jnp.mean(
                (jnp.dot(jax.nn.gelu(jnp.dot(x, p["w1"]) + p["b1"]), p["w2"]) + p["b2"] - y) ** 2
            ))(params)

        loss.block_until_ready()
```

## jax.profiler.start_trace / stop_trace

For profiling code that spans multiple functions or does not fit neatly into a context manager:

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_profiler_manual"

# Start profiling
jax.profiler.start_trace(log_dir, create_perfetto_link=True)

# ... any code ...
x = jnp.ones((4096, 4096))
y = jnp.dot(x, x)
z = jnp.sum(y)
z.block_until_ready()

# Stop profiling
jax.profiler.stop_trace()

print(f"Trace saved to {log_dir}")
```

### Profiling Across Multiple Functions

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_profiler_multi_fn"

jax.profiler.start_trace(log_dir)

# Function 1
def data_preprocessing(raw_data):
    with jax.profiler.TraceAnnotation("preprocessing"):
        normalized = (raw_data - jnp.mean(raw_data)) / jnp.std(raw_data)
        return normalized

# Function 2
@jax.jit
def model_inference(params, x):
    with jax.profiler.TraceAnnotation("inference"):
        return jnp.dot(x, params)

# Function 3
def post_processing(output):
    with jax.profiler.TraceAnnotation("postprocessing"):
        return jax.nn.softmax(output)

raw = jax.random.normal(jax.random.PRNGKey(0), (1000, 100))
params = jnp.ones((100, 10))

data = data_preprocessing(raw)
output = model_inference(params, data)
result = post_processing(output)
result.block_until_ready()

jax.profiler.stop_trace()
```

## jax.profiler.start_server

Start a profiling server that can be connected to from TensorBoard or XProf. This is useful for profiling long-running training jobs.

```python
import jax
import jax.numpy as jnp

# Start the profiler server on port 9999
server = jax.profiler.start_server(9999)

# Now the server is running. Connect TensorBoard to capture traces:
# tensorboard --log_dir=/tmp/jax_logs --port 6006
#
# Then open TensorBoard and click "Profile" to capture a trace.

# In a real training loop:
@jax.jit
def train_step(params, x, y):
    def loss_fn(p):
        pred = jnp.dot(x, p) - y
        return jnp.mean(pred ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

params = jnp.zeros((100, 10))
for step in range(10000):
    key = jax.random.PRNGKey(step)
    x = jax.random.normal(key, (32, 100))
    y = jax.random.normal(key, (32, 10))
    loss, grads = train_step(params, x, y)
    params = params - 0.01 * grads
    loss.block_until_ready()

# The server runs in the background until the process exits
```

### Combining start_server with TensorBoard

```bash
# Terminal 1: Start your training script with profiler server
python train.py  # Contains jax.profiler.start_server(9999)

# Terminal 2: Start TensorBoard
tensorboard --logdir=/tmp/jax_logs --port 6006

# Browser: Open http://localhost:6006
# Navigate to the "Profile" tab
# Click "Capture Profile" to start trace collection
# Set duration (e.g., 1000ms) and click "Capture"
```

## jax.profiler.TraceAnnotation

`TraceAnnotation` adds named spans to your trace for better organization and analysis.

### Basic Annotation

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_annotations"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    x = jax.random.normal(jax.random.PRNGKey(0), (4096, 4096))

    with jax.profiler.TraceAnnotation("matrix_multiply"):
        y = jnp.dot(x, x)

    with jax.profiler.TraceAnnotation("elementwise"):
        z = jax.nn.gelu(y)

    with jax.profiler.TraceAnnotation("reduction"):
        result = jnp.sum(z)

    result.block_until_ready()
```

### Nested Annotations

```python
import jax
import jax.numpy as jnp

@jax.jit
def transformer_layer(params, x):
    with jax.profiler.TraceAnnotation("self_attention"):
        with jax.profiler.TraceAnnotation("qkv_projection"):
            q = jnp.dot(x, params["wq"])
            k = jnp.dot(x, params["wk"])
            v = jnp.dot(x, params["wv"])

        with jax.profiler.TraceAnnotation("attention_scores"):
            scores = jnp.dot(q, k.T) / jnp.sqrt(q.shape[-1])

        with jax.profiler.TraceAnnotation("attention_weights"):
            weights = jax.nn.softmax(scores, axis=-1)

        with jax.profiler.TraceAnnotation("attention_output"):
            attn_out = jnp.dot(weights, v)

    with jax.profiler.TraceAnnotation("feedforward"):
        hidden = jnp.dot(attn_out, params["w1"])
        hidden = jax.nn.gelu(hidden)
        output = jnp.dot(hidden, params["w2"])

    return output
```

### Conditional Annotations

```python
import jax
import jax.numpy as jnp

def train_with_annotations(params, x, y, step):
    log_dir = "/tmp/jax_cond_annot"

    with jax.profiler.trace(log_dir):
        # Only annotate every 100th step (to reduce trace size)
        annotation_ctx = (
            jax.profiler.TraceAnnotation(f"step_{step}")
            if step % 100 == 0
            else contextlib.nullcontext()
        )

        with annotation_ctx:
            loss = jnp.mean((jnp.dot(x, params) - y) ** 2)
            grads = jax.grad(lambda p: jnp.mean((jnp.dot(x, p) - y) ** 2))(params)
            params = params - 0.01 * grads
            loss.block_until_ready()

    return params, loss
```

## jax.profiler.annotate_function

Decorator-based annotation for functions that appear in traces:

```python
import jax
import jax.numpy as jnp

@jax.profiler.annotate_function("data_loading")
def load_batch(key, batch_size, dim):
    return jax.random.normal(key, (batch_size, dim))

@jax.profiler.annotate_function("forward_pass")
def forward(params, x):
    return jnp.dot(x, params)

@jax.profiler.annotate_function("loss_computation")
def compute_loss(pred, y):
    return jnp.mean((pred - y) ** 2)

log_dir = "/tmp/jax_annotated_fn"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    key = jax.random.PRNGKey(0)
    params = jnp.ones((100, 10))

    x = load_batch(key, 32, 100)
    pred = forward(params, x)
    y = jax.random.normal(key, (32, 10))
    loss = compute_loss(pred, y)
    loss.block_until_ready()
```

## Perfetto Trace Visualization

Perfetto is the primary trace viewer for JAX profiles. It provides an interactive web-based interface for exploring traces.

### Generating a Perfetto Trace

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_perfetto_trace"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    x = jnp.ones((8192, 8192))

    with jax.profiler.TraceAnnotation("step_0"):
        y = jnp.dot(x, x)
        y.block_until_ready()

    with jax.profiler.TraceAnnotation("step_1"):
        y = jnp.dot(x, x)
        y.block_until_ready()

# Output:
# Perfetto trace: https://ui.perfetto.dev/...
# Or open the file in log_dir with the Perfetto UI
```

### Reading a Perfetto Trace

When you open a trace in Perfetto UI (https://ui.perfetto.dev):

1. **Timeline view:** Shows when each operation executed on each device/stream
2. **Thread tracks:** Shows CPU-side activity (dispatch, compilation)
3. **Device tracks:** Shows GPU/TPU kernel execution
4. **Slice details:** Click any slice to see duration, metadata, arguments
5. **Flame graph:** View call stack and hierarchical annotations

### Key Perfetto Features

```python
import jax
import jax.numpy as jnp

# Generate a detailed trace with annotations for analysis
log_dir = "/tmp/jax_perfetto_detailed"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    key = jax.random.PRNGKey(42)

    # Trace shows separate spans for each operation
    with jax.profiler.TraceAnnotation("data_generation"):
        x = jax.random.normal(key, (4096, 4096))
        w = jax.random.normal(key, (4096, 4096))

    with jax.profiler.TraceAnnotation("matmul"):
        y = jnp.dot(x, w)

    with jax.profiler.TraceAnnotation("activation"):
        z = jax.nn.gelu(y)

    with jax.profiler.TraceAnnotation("reduction"):
        result = jnp.sum(z)

    result.block_until_ready()

# In Perfetto, you can:
# - Zoom into specific operations
# - See exact kernel execution times on GPU
# - Identify gaps between kernel launches (CPU overhead)
# - Measure memory transfer time
# - Compare annotated regions
```

## XProf (TensorBoard) Profiling

XProf (formerly TensorBoard profiler plugin) provides TensorBoard integration for JAX profiling.

### Setup for XProf

```python
import jax
import jax.numpy as jnp
import datetime

# Create a timestamped log directory
log_dir = f"/tmp/jax_xprof/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Method 1: Use jax.profiler.trace
with jax.profiler.trace(log_dir):
    x = jnp.ones((4096, 4096))
    y = jnp.dot(x, x)
    y.block_until_ready()

# Method 2: Use start_trace/stop_trace for long-running processes
jax.profiler.start_trace(log_dir)
# ... training loop ...
jax.profiler.stop_trace()

# Then start TensorBoard:
# tensorboard --logdir=/tmp/jax_xprof --port 6006
# Open http://localhost:6006/#profile
```

### XProf Trace Analysis

In the XProf/TensorBoard Profile tab, you can see:

- **Overview Page:** High-level summary of step times, device utilization
- **Trace Viewer:** Timeline of all operations on each device stream
- **Memory Viewer:** Memory usage over time
- **Op Stats:** Table of operations sorted by duration
- **Dataflow Viewer:** Shows data dependencies between operations
- **Kernel Stats:** GPU kernel-level statistics

### Profiling a Training Loop with XProf

```python
import jax
import jax.numpy as jnp
import datetime

log_dir = f"/tmp/jax_xprof_training/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

@jax.jit
def train_step(params, opt_state, x, y, optimizer):
    def loss_fn(p):
        pred = jnp.dot(x, p["w"]) + p["b"]
        return jnp.mean((pred - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

import optax
optimizer = optax.adam(1e-3)

key = jax.random.PRNGKey(0)
params = {
    "w": jax.random.normal(key, (784, 10)) * 0.01,
    "b": jnp.zeros(10),
}
opt_state = optimizer.init(params)

# Profile the training loop
with jax.profiler.trace(log_dir):
    for step in range(20):
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (32, 784))
        y = jax.random.normal(subkey, (32, 10))

        with jax.profiler.TraceAnnotation(f"step_{step}"):
            params, opt_state, loss = train_step(params, opt_state, x, y, optimizer)
            loss.block_until_ready()

# Analyze in TensorBoard:
# - Check step-to-step timing variance
# - Identify if compilation is happening mid-loop
# - Check GPU utilization percentage
# - Look for unnecessary data transfers
```

## Memory Profiling with pprof

JAX can generate memory profiles in pprof format for analyzing memory allocation patterns.

### Basic Memory Profiling

```python
import jax
import jax.numpy as jnp

# Enable memory profiling
jax.config.update("jax_profiler_memory_profile_path", "/tmp/jax_memory_profile.pb")

@jax.jit
def memory_heavy_fn(x):
    a = jnp.dot(x, x)           # (4096, 4096) -- 64MB float32
    b = jnp.dot(a, x)           # Another 64MB
    c = jnp.dot(b, a.T)         # Another 64MB
    return jnp.sum(c)

x = jnp.ones((4096, 4096))
result = memory_heavy_fn(x)
result.block_until_ready()

# Analyze with pprof:
# go tool pprof -http=:8080 /tmp/jax_memory_profile.pb
```

### Analyzing Memory with Device Memory Stats

```python
import jax
import jax.numpy as jnp

# Check device memory stats
devices = jax.devices()
for device in devices:
    stats = device.memory_stats()
    if stats:
        print(f"Device: {device}")
        print(f"  Bytes in use: {stats['bytes_in_use'] / 1e9:.2f} GB")
        print(f"  Peak bytes in use: {stats['peak_bytes_in_use'] / 1e9:.2f} GB")
        print(f"  Bytes limit: {stats['bytes_limit'] / 1e9:.2f} GB")
        print(f"  Pool bytes in use: {stats.get('pool_bytes_in_use', 0) / 1e9:.2f} GB")
        print(f"  Peak pool bytes: {stats.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")

# Example output:
# Device: gpu:0
#   Bytes in use: 0.50 GB
#   Peak bytes in use: 2.13 GB
#   Bytes limit: 15.99 GB
```

### Memory Profiling for Gradient Checkpointing Decisions

```python
import jax
import jax.numpy as jnp

def measure_memory_usage(fn, *args):
    """Measure peak memory usage of a function."""
    device = jax.devices()[0]

    # Clear cache
    jax.clear_backends()

    # Run and measure
    result = fn(*args)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()

    stats = device.memory_stats()
    return stats['peak_bytes_in_use'] if stats else 0

# Compare memory with and without checkpointing
def standard_fn(x):
    for _ in range(10):
        x = jnp.dot(x, jnp.ones_like(x))
    return x

@jax.checkpoint
def checkpointed_fn(x):
    for _ in range(10):
        x = jnp.dot(x, jnp.ones_like(x))
    return x

x = jnp.ones((4096, 4096))

standard_mem = measure_memory_usage(jax.jit(standard_fn), x)
checkpointed_mem = measure_memory_usage(jax.jit(checkpointed_fn), x)

print(f"Standard: {standard_mem / 1e9:.2f} GB")
print(f"Checkpointed: {checkpointed_mem / 1e9:.2f} GB")
```

## GPU-Specific Profiling Tips

### GPU Utilization Analysis

```python
import jax
import jax.numpy as jnp

# Profile GPU utilization
log_dir = "/tmp/jax_gpu_profile"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    # Identify whether GPU is fully utilized
    x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))

    # Good: large matmul keeps GPU busy
    with jax.profiler.TraceAnnotation("large_matmul"):
        y = jnp.dot(x, x)
        y.block_until_ready()

    # Bad: many small operations cause kernel launch overhead
    with jax.profiler.TraceAnnotation("small_ops"):
        result = x
        for i in range(1000):
            result = result + 1.0
        result.block_until_ready()
```

### Data Transfer Profiling

```python
import jax
import jax.numpy as jnp
import numpy as np

log_dir = "/tmp/jax_transfer_profile"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    # Profile host-to-device transfers
    with jax.profiler.TraceAnnotation("host_to_device"):
        large_array = np.random.randn(10000, 10000).astype(np.float32)
        x = jax.device_put(large_array)
        x.block_until_ready()

    # Profile device-to-host transfers
    with jax.profiler.TraceAnnotation("device_to_host"):
        result = jax.device_get(x)

    # Profile device-to-device (if multiple GPUs)
    if len(jax.devices()) > 1:
        with jax.profiler.TraceAnnotation("device_to_device"):
            x_gpu1 = jax.device_put(x, jax.devices()[1])
            x_gpu1.block_until_ready()
```

### Identifying Fusion Opportunities

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_fusion_profile"

# Unfused: multiple separate kernels
@jax.jit
def unfused(x):
    a = x + 1.0      # Kernel 1
    b = a * 2.0      # Kernel 2
    c = jnp.exp(b)   # Kernel 3
    d = c + a        # Kernel 4
    return d

# Fused: XLA should fuse these into fewer kernels
@jax.jit
def fused(x):
    return jnp.exp((x + 1.0) * 2.0) + (x + 1.0)

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    x = jnp.ones((4096, 4096))

    with jax.profiler.TraceAnnotation("unfused"):
        result = unfused(x)
        result.block_until_ready()

    with jax.profiler.TraceAnnotation("fused"):
        result = fused(x)
        result.block_until_ready()

# In the trace, compare kernel counts and total execution time
```

## NCCL Profiling

NCCL (NVIDIA Collective Communications Library) handles multi-GPU communication. Profiling NCCL is important for distributed training.

### Enabling NCCL Logging

```python
import os

# Enable NCCL debug logging
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

# Enable NCCL tracing
os.environ["NCCL_TRACE"] = "enable"
os.environ["NCCL_TRACE_FILE"] = "/tmp/nccl_trace"

import jax
import jax.numpy as jnp
```

### Profiling Collective Operations

```python
import jax
import jax.numpy as jnp

log_dir = "/tmp/jax_nccl_profile"

with jax.profiler.trace(log_dir, create_perfetto_link=True):
    devices = jax.devices()
    if len(devices) > 1:
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

        mesh = Mesh(devices[:4], ("data",))
        sharding = NamedSharding(mesh, P("data"))

        x = jax.random.normal(jax.random.PRNGKey(0), (16384, 16384))
        x_sharded = jax.device_put(x, sharding)

        with jax.profiler.TraceAnnotation("all_reduce"):
            result = jax.jit(lambda x: jax.lax.psum(x, "data"))(x_sharded)
            result.block_until_ready()

        with jax.profiler.TraceAnnotation("all_gather"):
            result = jax.jit(lambda x: jax.lax.all_gather(x, "data"))(x_sharded)
            result.block_until_ready()
```

### NCCL Environment Variables for Tuning

```bash
# Set NCCL algorithm
export NCCL_ALGO=Ring          # Options: Ring, Tree, CollnetDirect, CollnetChain

# Set NCCL protocol
export NCCL_PROTO=Simple       # Options: Simple, LL (Low Latency), LL128

# Number of NCCL channels
export NCCL_NCHANNELS=4

# NCCL net plugin (for InfiniBand, etc.)
export NCCL_NET=IB             # Options: Socket, IB, AWS

# Disable SHM (shared memory) for multi-node
export NCCL_SHM_DISABLE=1

# Set NCCL socket interface for multi-node
export NCCL_SOCKET_IFNAME=eth0
```

## Compilation Profiling

### Measuring Compilation Time

```python
import time
import jax
import jax.numpy as jnp

def measure_compilation(fn, *args):
    """Measure JIT compilation time (first call only)."""
    start = time.perf_counter()
    result = fn(*args)
    result.block_until_ready()
    total = time.perf_counter() - start
    return total

@jax.jit
def complex_fn(x, y):
    z = jnp.dot(x, y)
    for _ in range(10):
        z = jnp.dot(z, z.T)
        z = jax.nn.gelu(z)
    return z

x = jnp.ones((1024, 1024))
y = jnp.ones((1024, 1024))

# First call includes compilation
compile_and_run = measure_compilation(complex_fn, x, y)
print(f"Compile + run: {compile_and_run:.3f}s")

# Subsequent calls are fast
run_only = measure_compilation(complex_fn, x, y)
print(f"Run only: {run_only:.3f}s")
print(f"Compilation overhead: {compile_and_run - run_only:.3f}s")
```

### Inspecting Lowered and Compiled Code

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

# Step 1: Lower to HLO (without compiling)
lowered = my_fn.lower(jnp.ones(100))
print("=== Lowered HLO ===")
print(lowered.as_text()[:500])  # View first 500 chars

# Step 2: Compile to executable
compiled = lowered.compile()

# Step 3: Inspect cost analysis
cost = compiled.cost_analysis()
print("\n=== Cost Analysis ===")
for key, value in cost[0].items():
    print(f"  {key}: {value}")

# Step 4: Get the executable's text representation
print("\n=== Compiled HLO ===")
# compiled.as_text() provides the final HLO
```

### Using XLA_FLAGS for Compilation Debugging

```python
import os

# Dump XLA compilation artifacts
os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"

import jax
import jax.numpy as jnp

@jax.jit
def my_fn(x):
    return jnp.sum(x ** 2)

my_fn(jnp.ones(100))

# Check /tmp/xla_dump/ for:
# - before_optimizations.txt  (original HLO)
# - after_optimizations.txt   (optimized HLO)
# - *.ir.txt                  (LLVM IR for GPU)
# - *.ptx                     (PTX assembly for GPU)
```

## Performance Analysis Workflow

### Step-by-Step Profiling Workflow

```python
import jax
import jax.numpy as jnp
import time

# ===== Step 1: Establish a baseline =====
@jax.jit
def model_fn(params, x):
    h = jnp.dot(x, params["w1"]) + params["b1"]
    h = jax.nn.relu(h)
    h = jnp.dot(h, params["w2"]) + params["b2"]
    return h

params = {
    "w1": jax.random.normal(jax.random.PRNGKey(0), (784, 512)),
    "b1": jnp.zeros(512),
    "w2": jax.random.normal(jax.random.PRNGKey(1), (512, 10)),
    "b2": jnp.zeros(10),
}
x = jax.random.normal(jax.random.PRNGKey(2), (128, 784))

# Warmup
_ = model_fn(params, x).block_until_ready()

# Baseline measurement
start = time.perf_counter()
for _ in range(100):
    result = model_fn(params, x)
result.block_until_ready()
baseline_time = (time.perf_counter() - start) / 100
print(f"Baseline: {baseline_time * 1000:.2f}ms per step")

# ===== Step 2: Profile with trace =====
log_dir = "/tmp/jax_perf_workflow"
with jax.profiler.trace(log_dir, create_perfetto_link=True):
    for step in range(10):
        with jax.profiler.TraceAnnotation(f"step_{step}"):
            result = model_fn(params, x)
            result.block_until_ready()

# ===== Step 3: Analyze trace in Perfetto =====
# Open the Perfetto link and look for:
# - Long gaps between kernel launches (CPU-bound?)
# - Small kernels that should be fused
# - Unnecessary data transfers
# - Low GPU utilization percentage

# ===== Step 4: Check memory usage =====
device = jax.devices()[0]
stats = device.memory_stats()
if stats:
    print(f"Memory: {stats['bytes_in_use'] / 1e6:.1f}MB / {stats['bytes_limit'] / 1e9:.1f}GB")

# ===== Step 5: Optimize and re-measure =====
# Apply optimizations based on trace analysis...
```

## Benchmarking Best Practices

### Standard Benchmark Template

```python
import jax
import jax.numpy as jnp
import time
import statistics

def benchmark(fn, *args, warmup=5, num_runs=20, **kwargs):
    """
    Benchmark a JAX function.

    Args:
        fn: The function to benchmark (should already be JIT-compiled).
        *args: Arguments to pass to fn.
        warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.

    Returns:
        Dictionary with timing statistics.
    """
    # Ensure function is compiled
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        _block_result(result)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        _block_result(result)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "throughput": None,  # User can compute based on problem size
    }

def _block_result(result):
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, (list, tuple)):
        for r in result:
            _block_result(r)
    elif isinstance(result, dict):
        for r in result.values():
            _block_result(r)
```

### Benchmarking Matrix Multiplication

```python
import jax
import jax.numpy as jnp

# Benchmark different matrix sizes
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

sizes = [256, 512, 1024, 2048, 4096, 8192]

print(f"{'Size':>8} | {'Time (ms)':>10} | {'TFLOPS':>8}")
print("-" * 40)

for n in sizes:
    a = jnp.ones((n, n), dtype=jnp.float32)
    b = jnp.ones((n, n), dtype=jnp.float32)

    stats = benchmark(matmul, a, b)

    # Compute TFLOPS: 2*n^3 FLOPs for matmul
    flops = 2 * n ** 3
    tflops = flops / (stats['median_ms'] * 1e-3) / 1e12

    print(f"{n:>8} | {stats['median_ms']:>10.2f} | {tflops:>8.2f}")
```

### Benchmarking with Throughput

```python
import jax
import jax.numpy as jnp

@jax.jit
def train_step(params, x, y):
    def loss_fn(p):
        pred = jnp.dot(x, p) + y * 0  # Simplified
        return jnp.mean(pred ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss

batch_size = 128
dim = 784
params = jnp.zeros((dim,))
x = jnp.ones((batch_size, dim))
y = jnp.ones((batch_size,))

stats = benchmark(train_step, params, x, y)

# Compute throughput: samples per second
throughput = batch_size / (stats['median_ms'] * 1e-3)
print(f"Training throughput: {throughput:.0f} samples/sec")
print(f"Time per step: {stats['median_ms']:.2f}ms")
```

### Comparing Implementations

```python
import jax
import jax.numpy as jnp

def compare_implementations(*fns, labels, args, warmup=5, num_runs=20):
    """Compare multiple implementations of the same function."""
    results = {}
    for fn, label in zip(fns, labels):
        jitted = jax.jit(fn)

        # Warmup
        for _ in range(warmup):
            _block_result(jitted(*args))

        # Time
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            result = jitted(*args)
            _block_result(result)
            times.append(time.perf_counter() - start)

        results[label] = {
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
        }

    # Print comparison
    print(f"{'Implementation':>20} | {'Mean (ms)':>10} | {'Median (ms)':>12}")
    print("-" * 50)
    for label, r in results.items():
        print(f"{label:>20} | {r['mean_ms']:>10.2f} | {r['median_ms']:>12.2f}")

    return results

# Example usage
def impl_v1(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))

def impl_v2(x):
    return jax.nn.softmax(x)

x = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
compare_implementations(impl_v1, impl_v2, labels=["manual", "jax.nn"],
                       args=(x,))
```

## Summary: Profiling Quick Reference

| Task | Tool | Usage |
|------|------|-------|
| Accurate timing | `.block_until_ready()` | `result.block_until_ready()` |
| Capture trace | `jax.profiler.trace` | `with jax.profiler.trace(dir):` |
| Manual trace control | `start_trace/stop_trace` | `jax.profiler.start_trace(dir)` |
| Profiler server | `start_server` | `jax.profiler.start_server(9999)` |
| Annotate spans | `TraceAnnotation` | `with jax.profiler.TraceAnnotation("name"):` |
| Annotate functions | `annotate_function` | `@jax.profiler.annotate_function("name")` |
| View traces | Perfetto | https://ui.perfetto.dev |
| TensorBoard profiling | XProf | `tensorboard --logdir=dir` |
| Memory stats | `device.memory_stats()` | `jax.devices()[0].memory_stats()` |
| Memory profile | pprof | Set `jax_profiler_memory_profile_path` |
| View HLO IR | `lower().as_text()` | `jax.jit(fn).lower(*args).as_text()` |
| Cost analysis | `compile().cost_analysis()` | `lowered.compile().cost_analysis()` |
| NCCL debug | Environment | `NCCL_DEBUG=INFO` |
| Compilation debug | XLA_FLAGS | `--xla_dump_to=/tmp/xla_dump` |
| Disable JIT | Config | `jax.config.update("jax_disable_jit", True)` |
| NaN debugging | Config | `jax.config.update("jax_debug_nans", True)` |
