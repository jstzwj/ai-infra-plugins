# 45 - Benchmarking

## Overview

Proper benchmarking in JAX requires understanding asynchronous dispatch, JIT warmup, and device-specific behavior. This chapter covers benchmarking best practices and tools.

---

## 1. Basic Timing

### Wall-clock timing

```python
import time
import jax
import jax.numpy as jnp

# The WRONG way
start = time.time()
result = jnp.dot(x, y)  # Returns immediately (async)
elapsed = time.time() - start  # ~0 (doesn't wait!)

# The RIGHT way
start = time.time()
result = jnp.dot(x, y)
result.block_until_ready()  # Wait for computation
elapsed = time.time() - start  # Actual time
```

### Using `%timeit` in notebooks

```python
# Correct pattern
%timeit jnp.dot(x, y).block_until_ready()
```

---

## 2. JIT Warmup

```python
@jax.jit
def f(x):
    return jnp.dot(x, x.T)

x = jnp.ones((1000, 1000))

# Warmup: compile once
f(x).block_until_ready()

# Now benchmark
%timeit f(x).block_until_ready()
```

### Measuring compilation time separately

```python
import time

# Measure compilation
start = time.time()
f_jit = jax.jit(lambda x: jnp.dot(x, x.T))
f_jit(x).block_until_ready()
compile_time = time.time() - start
print(f"Compilation: {compile_time:.3f}s")

# Measure execution
start = time.time()
for _ in range(100):
    f_jit(x).block_until_ready()
exec_time = (time.time() - start) / 100
print(f"Execution: {exec_time*1000:.3f}ms")
```

---

## 3. Benchmarking Utilities

### Simple benchmark function

```python
def benchmark(fn, *args, warmup=5, num_runs=100, **kwargs):
    """Benchmark a JAX function."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    # Time runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        times.append(time.perf_counter() - start)

    import numpy as np
    times = np.array(times) * 1000  # Convert to ms
    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times),
        'runs': num_runs,
    }
```

### Usage

```python
results = benchmark(jnp.dot, jnp.ones((1000, 1000)), jnp.ones((1000, 1000)))
print(f"Mean: {results['mean']:.3f} ± {results['std']:.3f} ms")
```

---

## 4. Comparing Implementations

```python
def compare_implementations(*fns, names=None, args=()):
    """Compare multiple implementations."""
    results = {}
    for fn, name in zip(fns, names or [f.__name__ for f in fns]):
        results[name] = benchmark(fn, *args)
    return results

# Example
@jax.jit
def v1(x):
    return jnp.sum(x ** 2)

@jax.jit
def v2(x):
    return jnp.dot(x, x)

compare_implementations(v1, v2, names=['sum_square', 'dot'], args=(jnp.ones(10000),))
```

---

## 5. Profiling Tools

### jax.profiler.trace

```python
import jax.profiler

with jax.profiler.trace("/tmp/profile", create_perfetto_link=True):
    for i in range(10):
        result = train_step(params, x, y)
        result.block_until_ready()
        jax.profiler.step_trace(i)  # Mark step boundary

# Open in TensorBoard or Perfetto
# tensorboard --logdir=/tmp/profile
```

### Annotation

```python
with jax.profiler.TraceAnnotation("my_custom_region"):
    result = expensive_computation(x)
```

### Function annotation

```python
@jax.profiler.annotate_function("forward_pass")
def forward(params, x):
    return predict(params, x)
```

---

## 6. Memory Benchmarking

### Peak memory

```python
# Before
device = jax.devices()[0]
stats_before = device.memory_stats()

# Run computation
result = f(x)
result.block_until_ready()

# After
stats_after = device.memory_stats()
memory_used = stats_after['bytes_in_use'] - stats_before['bytes_in_use']
print(f"Memory used: {memory_used / 1e9:.2f} GB")
```

### Memory estimation

```python
def estimate_memory(params, batch_size, dtype=jnp.float32):
    """Estimate memory for forward + backward pass."""
    param_bytes = sum(p.size * p.dtype.itemsize for p in jax.tree.leaves(params))
    # Rough estimate: forward stores activations, backward stores gradients
    # 3× params is a rough rule of thumb
    return 3 * param_bytes
```

---

## 7. FLOPS Estimation

```python
def estimate_flops(matmul_m=1024, matmul_n=1024, matmul_k=1024):
    """Estimate FLOPs for a matmul."""
    return 2 * matmul_m * matmul_n * matmul_k

# TFLOPS = FLOPs / (time_seconds * 1e12)
flops = estimate_flops(1024, 1024, 1024)
time_ms = benchmark(lambda: jnp.dot(a, b))['mean']
tflops = flops / (time_ms / 1000) / 1e12
print(f"Throughput: {tflops:.2f} TFLOPS")
```

---

## 8. Device-Specific Benchmarking

### CPU vs GPU vs TPU

```python
devices = jax.devices()
for device in devices:
    x = jax.device_put(jnp.ones((1000, 1000)), device)
    result = benchmark(lambda: jnp.dot(x, x))
    print(f"{device.device_kind}: {result['mean']:.3f} ms")
```

### Multi-device

```python
# Benchmark with different numbers of devices
for n_devices in [1, 2, 4, 8]:
    devices = jax.devices()[:n_devices]
    mesh = Mesh(devices, ('x',))
    # ... run and benchmark
```

---

## 9. Common Benchmarking Mistakes

| Mistake | Problem | Fix |
|---|---|---|
| No `block_until_ready` | Measures dispatch, not compute | Always block |
| No JIT warmup | Includes compilation time | Warmup before timing |
| Too few runs | Noisy results | At least 100 runs |
| GC during timing | Outlier spikes | Force GC before timing |
| Not controlling for other load | Interference | Isolate benchmark |
| Ignoring first iteration | Always slow | Skip first N iterations |

---

## 10. Benchmarking Template

```python
import jax
import jax.numpy as jnp
import time
import numpy as np

def full_benchmark(name, fn, *args, warmup=10, runs=1000):
    """Complete benchmark with compilation and memory tracking."""
    print(f"\n=== {name} ===")

    # Compilation
    start = time.perf_counter()
    fn_jit = jax.jit(fn)
    result = fn_jit(*args)
    result.block_until_ready()
    compile_ms = (time.perf_counter() - start) * 1000
    print(f"Compile: {compile_ms:.1f} ms")

    # Warmup
    for _ in range(warmup):
        fn_jit(*args).block_until_ready()

    # Measure
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn_jit(*args).block_until_ready()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"Mean: {times.mean():.3f} ms")
    print(f"Std:  {times.std():.3f} ms")
    print(f"Min:  {times.min():.3f} ms")
    print(f"P50:  {np.percentile(times, 50):.3f} ms")
    print(f"P99:  {np.percentile(times, 99):.3f} ms")

    return {'compile_ms': compile_ms, 'mean_ms': times.mean(), 'times': times}
```
