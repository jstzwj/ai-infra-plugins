# Apache TVM - Testing and Benchmarking

This reference covers testing and benchmarking in Apache TVM, including the testing framework, target parametrization, benchmarking approaches, and CI/CD integration.

---

## 34.1 Testing Framework

### 34.1.1 pytest-Based Testing

TVM uses [pytest](https://docs.pytest.org/) as its primary testing framework. All Python tests are organized as pytest test cases and can be discovered and run using standard pytest commands.

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/python/unittest/test_ir_builder.py

# Run a specific test function
pytest tests/python/unittest/test_ir_builder.py::test_if_then_else

# Run tests matching a keyword
pytest tests/python/unittest/ -k "test_tensor"

# Run with verbose output
pytest tests/python/unittest/test_ir_builder.py -v

# Run with print output visible
pytest tests/python/unittest/test_ir_builder.py -s

# Run and stop at first failure
pytest tests/python/unittest/test_ir_builder.py -x

# Run in parallel (requires pytest-xdist)
pytest tests/python/unittest/ -n 4
```

A typical TVM test case looks like:

```python
import pytest
import tvm
import numpy as np
from tvm import relay, tir


def test_simple_add():
    """Test basic addition operation in Relay."""
    x = relay.var("x", shape=(10,), dtype="float32")
    y = relay.var("y", shape=(10,), dtype="float32")
    z = relay.add(x, y)
    mod = tvm.IRModule.from_expr(z)

    # Compile and execute
    with tvm.transform.PassContext(opt_level=0):
        lib = tvm.relay.build(mod, target="llvm")

    dev = tvm.cpu(0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Set inputs
    x_data = np.random.randn(10).astype("float32")
    y_data = np.random.randn(10).astype("float32")
    runtime.set_input("x", x_data)
    runtime.set_input("y", y_data)

    # Run and check
    runtime.run()
    output = runtime.get_output(0).numpy()
    expected = x_data + y_data
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
```

### 34.1.2 Test Organization in tests/ Directory

TVM's test suite is organized into several subdirectories under `tests/`:

```
tests/
├── python/
│   ├── unittest/              # Core unit tests
│   │   ├── test_ir_builder.py
│   │   ├── test_arith_canonical_simplify.py
│   │   ├── test_tir_transform_*.py
│   │   ├── test_runtime_*.py
│   │   └── ...
│   ├── relay/                 # Relay-specific tests
│   │   ├── test_pass_*.py     # Pass tests
│   │   ├── test_op_*.py       # Operator tests
│   │   ├── test_type_solver.py
│   │   └── ...
│   ├── contrib/               # External framework integration tests
│   │   ├── test_onnx.py
│   │   ├── test_pytorch.py
│   │   ├── test_tensorflow.py
│   │   └── ...
│   ├── frontend/              # Frontend import tests
│   ├── driver/                # Driver tests
│   └── topi/                  # TOPI (Tensor Operator Inventory) tests
├── scripts/                   # Test helper scripts
├── lint/                      # Linting configuration
└── CI/                        # CI-specific configurations
```

Key test categories:
- **`unittest/`**: Tests for core TVM infrastructure -- IR, TIR, runtime, target, arithmetic analysis.
- **`relay/`**: Tests for the Relay intermediate representation -- passes, operators, type inference.
- **`contrib/`**: Tests for integrations with external frameworks (ONNX, PyTorch, TensorFlow, cuDNN, etc.).
- **`topi/`**: Tests for the Tensor Operator Inventory -- operator implementations for various targets.

### 34.1.3 Test Categories

Tests are categorized by scope and purpose:

| Category | Directory | Purpose |
|----------|-----------|---------|
| Unit tests | `python/unittest/` | Test individual components in isolation |
| Integration tests | `python/relay/` | Test multi-component interactions |
| Operator tests | `python/relay/test_op_*.py` | Test operator correctness |
| Pass tests | `python/relay/test_pass_*.py` | Test IR transformation passes |
| Frontend tests | `python/contrib/` | Test framework import and conversion |
| TOPI tests | `python/topi/` | Test operator implementations per target |

---

## 34.2 Target Parametrization

### 34.2.1 @tvm.testing.parametrize_targets Decorator

TVM provides the `@tvm.testing.parametrize_targets` decorator to run a test across multiple hardware targets. This ensures correctness on all supported platforms.

```python
import tvm
import tvm.testing
import numpy as np
from tvm import relay

@tvm.testing.parametrize_targets
def test_relu_on_all_targets(target, dev):
    """Test ReLU operation on all available targets."""
    x = relay.var("x", shape=(10,), dtype="float32")
    y = relay.nn.relu(x)
    mod = tvm.IRModule.from_expr(y)

    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    x_data = np.array([-1, 2, -3, 4, -5, 6, -7, 8, -9, 10], dtype="float32")
    runtime.set_input("x", x_data)
    runtime.run()
    output = runtime.get_output(0).numpy()

    expected = np.maximum(x_data, 0)
    np.testing.assert_allclose(output, expected, rtol=1e-5)
```

The decorator automatically parametrizes the test for all targets that are enabled in the current TVM build (e.g., `llvm`, `cuda`, `rocm`, `vulkan`, `opencl`).

### 34.2.2 Testing on Multiple Devices

You can explicitly specify which targets to test:

```python
import tvm.testing

# Test only on CPU (LLVM) and CUDA
@tvm.testing.parametrize_targets("llvm", "cuda")
def test_dense_on_cpu_gpu(target, dev):
    """Test dense operation on CPU and GPU only."""
    # ... test implementation
    pass

# Test on specific targets with custom parameters
@tvm.testing.parametrize_targets(
    tvm.testing.parameter("llvm"),
    tvm.testing.parameter("cuda"),
    tvm.testing.parameter("vulkan"),
)
def test_conv2d_multiple_targets(target, dev):
    """Test conv2d on specific targets."""
    # ... test implementation
    pass

# Skip targets that are not available
@tvm.testing.parametrize_targets
@tvm.testing.skip_if_gpu_not_enabled
def test_cpu_only_feature(target, dev):
    """Test a CPU-only feature, skip on GPU."""
    assert "cpu" in str(target) or "llvm" in str(target)
```

### 34.2.3 pytest_target_parametrization

TVM integrates target parametrization with pytest's parametrize mechanism:

```python
import pytest
import tvm
import tvm.testing

# Method 1: Using the built-in target fixture
def test_with_target_fixture(target, dev):
    """Using target and dev fixtures from tvm.testing."""
    assert target is not None
    assert dev is not None

# Method 2: Explicit parametrize
@pytest.mark.parametrize("target,dev", [
    ("llvm", tvm.cpu(0)),
    ("cuda", tvm.cuda(0)),
])
def test_explicit_targets(target, dev):
    """Test with explicitly specified targets."""
    pass

# Method 3: Using enabled_targets fixture
def test_all_enabled_targets(enabled_targets):
    """Test across all enabled targets."""
    for target, dev in enabled_targets:
        print(f"Testing on: {target}")
```

---

## 34.3 Testing Utilities (tvm.testing)

### 34.3.1 assert_allclose

`tvm.testing.assert_allclose` is a convenience wrapper around `numpy.testing.assert_allclose` with TVM-specific defaults.

```python
import tvm.testing
import numpy as np

def test_assert_allclose():
    actual = np.array([1.0, 2.0, 3.0], dtype="float32")
    expected = np.array([1.0, 2.0, 3.0], dtype="float32")

    # Basic usage (default tolerance: rtol=1e-5, atol=1e-7)
    tvm.testing.assert_allclose(actual, expected)

    # Custom tolerances
    tvm.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)

    # With a custom error message
    tvm.testing.assert_allclose(
        actual, expected,
        rtol=1e-5,
        atol=1e-7,
        msg="Dense layer output mismatch"
    )
```

### 34.3.2 randn_dtype

`tvm.testing.randn_dtype` generates random tensors with specified dtype and shape, handling different numeric types appropriately.

```python
import tvm.testing
import numpy as np

def test_randn_dtype():
    # Generate float32 random tensor
    x_float = tvm.testing.randn(shape=(128, 128), dtype="float32")
    assert x_float.shape == (128, 128)
    assert x_float.dtype == "float32"

    # Generate int32 random tensor
    x_int = tvm.testing.randn(shape=(64,), dtype="int32")
    assert x_int.dtype == "int32"

    # Generate with specific seed for reproducibility
    x1 = tvm.testing.randn(shape=(10,), dtype="float32", seed=42)
    x2 = tvm.testing.randn(shape=(10,), dtype="float32", seed=42)
    np.testing.assert_array_equal(x1, x2)  # Same seed = same values
```

### 34.3.3 device_enabled

`tvm.testing.device_enabled` checks whether a specific device type is available on the current system.

```python
import tvm.testing

def test_conditional_device():
    # Check if CUDA is available
    if tvm.testing.device_enabled("cuda"):
        print("CUDA is available, running GPU test")
        dev = tvm.cuda(0)
        # Run GPU-specific test
    else:
        print("CUDA not available, skipping GPU test")

    # Check other devices
    if tvm.testing.device_enabled("rocm"):
        print("ROCm (AMD GPU) is available")

    if tvm.testing.device_enabled("vulkan"):
        print("Vulkan is available")

    if tvm.testing.device_enabled("opencl"):
        print("OpenCL is available")

# Skip test if device is not available
@pytest.mark.skipif(
    not tvm.testing.device_enabled("cuda"),
    reason="CUDA not available"
)
def test_cuda_specific():
    """Only runs if CUDA is available."""
    dev = tvm.cuda(0)
    # ... CUDA-specific test
```

### 34.3.4 uses_gpu Marker

The `uses_gpu` pytest marker indicates that a test requires a GPU. Tests marked with `uses_gpu` are skipped in CPU-only CI environments.

```python
import pytest
import tvm.testing

@pytest.mark.uses_gpu
def test_gpu_kernel():
    """This test requires a GPU to run."""
    dev = tvm.cuda(0)
    # ... GPU test implementation

# Multiple markers combined
@pytest.mark.uses_gpu
@pytest.mark.skipif(
    not tvm.testing.device_enabled("cuda"),
    reason="CUDA device not available"
)
def test_cuda_advanced():
    """Advanced CUDA test with proper guards."""
    pass
```

---

## 34.4 Benchmarking Approaches

### 34.4.1 tvm.runtime.time_evaluator

`time_evaluator` is TVM's primary tool for measuring the execution time of compiled functions. It handles warmup, repeated execution, and statistical measurement.

```python
import tvm
import numpy as np
from tvm import relay

# Build a simple function
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
w = relay.const(np.random.randn(32, 3, 3, 3).astype("float32"))
y = relay.nn.conv2d(x, w, padding=(1, 1), channels=32, kernel_size=(3, 3))
mod = tvm.IRModule.from_expr(y)
params = {}

with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target="llvm", params=params)

dev = tvm.cpu(0)
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

# Set input
runtime.set_input("x", np.random.randn(1, 3, 224, 224).astype("float32"))

# Create a time evaluator
# number: number of runs per measurement
# repeat: number of repeated measurements
evaluator = runtime.module.time_evaluator(
    "run",       # Function name to time
    dev,         # Target device
    number=10,   # Runs per measurement
    repeat=5,    # Number of repeated measurements
)

# Run the benchmark
prof_result = evaluator()

# Access results
print(f"Mean time: {prof_result.mean * 1000:.3f} ms")
print(f"Median time: {prof_result.median * 1000:.3f} ms")
print(f"Min time: {prof_result.min * 1000:.3f} ms")
print(f"Max time: {prof_result.max * 1000:.3f} ms")
print(f"Std dev: {prof_result.std * 1000:.3f} ms")

# Print all individual measurements
for i, result in enumerate(prof_result.results):
    print(f"  Run {i}: {result * 1000:.3f} ms")
```

### 34.4.2 Repeat and Median Timing

For robust benchmarking, use repeated measurements and report the median to reduce the impact of outliers.

```python
import tvm
import numpy as np
import statistics

def benchmark_function(runtime, dev, input_data, number=100, repeat=10):
    """Benchmark a TVM function with robust statistics."""
    runtime.set_input("x", input_data)

    # Warmup
    for _ in range(5):
        runtime.run()

    # Create evaluator with many repeats
    evaluator = runtime.module.time_evaluator(
        "run", dev, number=number, repeat=repeat
    )

    results = evaluator()

    # Compute statistics
    times_ms = [r * 1000 for r in results.results]
    median_ms = statistics.median(times_ms)
    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0

    print(f"Benchmark results ({repeat} repeats x {number} runs):")
    print(f"  Median: {median_ms:.4f} ms")
    print(f"  Mean:   {mean_ms:.4f} ms")
    print(f"  Std:    {std_ms:.4f} ms")
    print(f"  Min:    {min(times_ms):.4f} ms")
    print(f"  Max:    {max(times_ms):.4f} ms")

    return {
        "median": median_ms,
        "mean": mean_ms,
        "std": std_ms,
        "min": min(times_ms),
        "max": max(times_ms),
    }
```

### 34.4.3 Excluding Warmup

Warmup runs are essential for accurate benchmarking because the first few executions include JIT compilation, cache cold-start, and other one-time costs.

```python
import tvm
import numpy as np

def benchmark_with_warmup(runtime, dev, input_data, warmup=10, number=50, repeat=5):
    """Benchmark with explicit warmup phase."""
    runtime.set_input("x", input_data)

    # Warmup phase: run several times to warm caches and JIT
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        runtime.run()

    # Ensure warmup is complete by synchronizing the device
    dev.sync()

    # Measurement phase
    evaluator = runtime.module.time_evaluator(
        "run", dev, number=number, repeat=repeat
    )
    results = evaluator()

    return results

# For CUDA, also consider CUDA events for precise timing
def benchmark_cuda_precise(runtime, dev):
    """Benchmark using CUDA events for precise GPU timing."""
    # TVM's time_evaluator already uses device-specific timers
    # For CUDA, it uses CUDA events internally
    evaluator = runtime.module.time_evaluator(
        "run", dev, number=100, repeat=10
    )
    return evaluator()
```

### 34.4.4 Remote Benchmarking via RPC

TVM supports benchmarking on remote devices through its RPC mechanism. This is essential for testing on embedded devices, mobile phones, or remote GPUs.

```python
import tvm
from tvm import rpc
import numpy as np

# Start an RPC server on the remote device (run on the remote machine):
#   python -m tvm.exec.rpc_server --host 0.0.0.0 --port 9090
#
# Or use the tracker for multiple devices:
#   python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190
#   python -m tvm.exec.rpc_server --tracker 127.0.0.1:9190 --key cuda_device

# Connect to the remote server
remote = rpc.connect("192.168.1.100", 9090)
dev = remote.device("cuda", 0)

# Or connect via tracker
# tracker = rpc.connect_tracker("192.168.1.100", 9190)
# remote = tracker.request("cuda_device", priority=0, session_timeout=60)
# dev = remote.device("cuda", 0)

# Build locally, run remotely
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target=target)

# Upload the compiled module to the remote device
temp = rpc.Server.local()
remote.upload(lib.get_lib())
remote.upload(lib.get_graph_json())
# Also upload params if any

# Load the module on the remote device
runtime = tvm.contrib.graph_executor.GraphModule(
    remote["tvm.graph_executor.create"](lib.get_graph_json(), lib.get_lib(), dev)
)

# Set input and benchmark on the remote device
x_data = np.random.randn(1, 3, 224, 224).astype("float32")
runtime.set_input("x", tvm.nd.array(x_data, device=dev))

# Time evaluation on the remote device
evaluator = runtime.module.time_evaluator("run", dev, number=10, repeat=5)
results = evaluator()
print(f"Remote execution time: {results.median * 1000:.3f} ms")
```

---

## 34.5 MetaSchedule Benchmarking

### 34.5.1 Runner Configuration

MetaSchedule uses a `Runner` abstraction for benchmarking tuned configurations. The runner controls how measurements are performed during auto-tuning.

```python
import tvm
from tvm import meta_schedule as ms
from tvm.target import Target

# Configure the runner for MetaSchedule tuning
runner = ms.runner.Runner(
    evaluator_config=ms.runner.EvaluatorConfig(
        # Number of runs per measurement
        number=3,
        # Number of repeated measurements
        repeat=1,
        # Minimum evaluation time in seconds
        min_repeat_ms=100,
        # Enable cache flush between runs
        enable_cpu_cache_flush=True,
        # Maximum number of concurrent measurement jobs
        max_workers=1,
    ),
)

# Using the runner with a tuning task
database = ms.database.MemoryDatabase()
cost_model = ms.cost_model.RandomModel()

tune_config = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=64,
    max_trials_per_task=2000,
    max_trials_global=10000,
)

# Run tuning with the configured runner
# (See MetaSchedule documentation for full tuning workflow)
```

### 34.5.2 Measurement Callbacks

MetaSchedule supports callbacks that are invoked after each measurement, enabling custom analysis and logging.

```python
import tvm
from tvm import meta_schedule as ms
import json

class CustomMeasurementCallback:
    """Custom callback that logs measurement results."""

    def __init__(self, log_file="benchmark_log.json"):
        self.log_file = log_file
        self.results = []

    def __call__(self, tuning_record):
        """Called after each measurement."""
        result = {
            "workload": str(tuning_record.workload),
            "run_secs": [float(s) for s in tuning_record.run_secs],
            "instruction": str(tuning_record.trace),
        }
        self.results.append(result)

        # Log to file
        with open(self.log_file, "a") as f:
            json.dump(result, f)
            f.write("\n")

# Use built-in callbacks
callbacks = [
    ms.database.DefaultDatabase(),
    ms.callback.RemoveBuildArtifact(),
    ms.callback.EchoStatistics(),
]
```

### 34.5.3 Database Analysis

After tuning, analyze the database to understand performance characteristics.

```python
import tvm
from tvm import meta_schedule as ms

def analyze_tuning_results(database):
    """Analyze results from a MetaSchedule tuning database."""
    # Get all workloads
    workloads = database.commit_workload
    print(f"Total workloads: {len(workloads)}")

    # Get best tuning records for each workload
    for workload in workloads:
        best = database.query_top_k(workload, top_k=1)
        if best:
            record = best[0]
            avg_time = sum(record.run_secs) / len(record.run_secs)
            print(f"  Workload: {workload}")
            print(f"  Best time: {avg_time * 1000:.4f} ms")
            print(f"  Trace: {record.trace}")
            print()

# Compare tuning results across different strategies
def compare_strategies(mod, params, target):
    """Compare different tuning strategies."""
    strategies = ["evolutionary", "replay_func", "replay_trace"]

    results = {}
    for strategy in strategies:
        print(f"Tuning with strategy: {strategy}")
        # Run tuning (simplified)
        # ...
        # Store results
        results[strategy] = {"best_time": 0.0}  # Placeholder

    return results
```

---

## 34.6 Performance Regression Testing

### 34.6.1 Baseline Comparison

Performance regression tests compare current performance against established baselines.

```python
import tvm
import numpy as np
import json
import os

class PerformanceBaseline:
    """Manage performance baselines for regression testing."""

    def __init__(self, baseline_file="perf_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines = {}
        if os.path.exists(baseline_file):
            with open(baseline_file, "r") as f:
                self.baselines = json.load(f)

    def get_baseline(self, test_name):
        """Get the baseline time for a test."""
        return self.baselines.get(test_name, None)

    def save_baseline(self, test_name, time_ms):
        """Save a new baseline time."""
        self.baselines[test_name] = time_ms
        with open(self.baseline_file, "w") as f:
            json.dump(self.baselines, f, indent=2)

    def check_regression(self, test_name, current_time_ms, tolerance=0.2):
        """Check if there's a performance regression."""
        baseline = self.get_baseline(test_name)
        if baseline is None:
            print(f"No baseline for {test_name}, setting current as baseline")
            self.save_baseline(test_name, current_time_ms)
            return True

        ratio = current_time_ms / baseline
        if ratio > 1.0 + tolerance:
            print(f"REGRESSION: {test_name}")
            print(f"  Baseline: {baseline:.4f} ms")
            print(f"  Current:  {current_time_ms:.4f} ms")
            print(f"  Ratio:    {ratio:.2f}x slower")
            return False
        else:
            print(f"OK: {test_name} ({ratio:.2f}x)")
            return True

# Usage
def test_perf_no_regression():
    baseline = PerformanceBaseline("perf_baselines.json")

    # Build and benchmark the model
    # ... (compilation and benchmarking code)

    current_time_ms = 5.0  # Example benchmark result
    assert baseline.check_regression("conv2d_3x3", current_time_ms, tolerance=0.2)
```

### 34.6.2 Variance Analysis

Understanding variance in benchmark results is crucial for reliable performance testing.

```python
import tvm
import numpy as np
import statistics

def analyze_benchmark_variance(runtime, dev, input_data, number=100, repeat=30):
    """Analyze variance in benchmark measurements."""
    runtime.set_input("x", input_data)

    # Warmup
    for _ in range(20):
        runtime.run()

    # Collect many measurements
    evaluator = runtime.module.time_evaluator(
        "run", dev, number=number, repeat=repeat
    )
    results = evaluator()

    times_ms = sorted([r * 1000 for r in results.results])

    # Compute statistics
    mean = statistics.mean(times_ms)
    median = statistics.median(times_ms)
    stdev = statistics.stdev(times_ms)
    cv = stdev / mean * 100  # Coefficient of variation

    # Percentiles
    p50 = np.percentile(times_ms, 50)
    p90 = np.percentile(times_ms, 90)
    p95 = np.percentile(times_ms, 95)
    p99 = np.percentile(times_ms, 99)

    print("Benchmark Variance Analysis")
    print(f"  Mean:   {mean:.4f} ms")
    print(f"  Median: {median:.4f} ms")
    print(f"  StdDev: {stdev:.4f} ms")
    print(f"  CV:     {cv:.2f}%")
    print(f"  P50:    {p50:.4f} ms")
    print(f"  P90:    {p90:.4f} ms")
    print(f"  P95:    {p95:.4f} ms")
    print(f"  P99:    {p99:.4f} ms")

    # Recommendations
    if cv > 5.0:
        print("  WARNING: High variance detected. Consider:")
        print("    - Increasing number of runs per measurement")
        print("    - Pinning CPU frequency")
        print("    - Disabling turbo boost")
        print("    - Running on a dedicated machine")

    return {
        "mean": mean, "median": median, "stdev": stdev,
        "cv": cv, "p50": p50, "p99": p99,
    }
```

### 34.6.3 Statistical Significance

Determine if a performance change is statistically significant rather than due to noise.

```python
import numpy as np
from scipy import stats

def is_significant(before_times, after_times, alpha=0.05):
    """
    Perform a statistical test to determine if the performance
    difference is significant using Welch's t-test.
    """
    before = np.array(before_times)
    after = np.array(after_times)

    # Welch's t-test (does not assume equal variance)
    t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(before) + np.var(after)) / 2
    )
    cohens_d = (np.mean(before) - np.mean(after)) / pooled_std

    print(f"Before: mean={np.mean(before):.4f}, std={np.std(before):.4f}")
    print(f"After:  mean={np.mean(after):.4f}, std={np.std(after):.4f}")
    print(f"Change: {(np.mean(after)/np.mean(before)-1)*100:.2f}%")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_value:.6f}")
    print(f"Cohen's d:   {cohens_d:.4f}")

    if p_value < alpha:
        if np.mean(after) < np.mean(before):
            print("Result: SIGNIFICANT IMPROVEMENT")
        else:
            print("Result: SIGNIFICANT REGRESSION")
        return True
    else:
        print("Result: NOT SIGNIFICANT (likely noise)")
        return False
```

---

## 34.7 Writing Test Cases

### 34.7.1 TIR Program Testing

Testing TIR programs involves verifying both structural correctness and numerical accuracy.

```python
import tvm
import numpy as np
from tvm.script import tir as T
import pytest

@T.prim_func
def matmul_64x64(
    A: T.Buffer((64, 64), "float32"),
    B: T.Buffer((64, 64), "float32"),
    C: T.Buffer((64, 64), "float32"),
) -> None:
    for i, j in T.grid(64, 64):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float32(0.0)
    for i, j, k in T.grid(64, 64, 64):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def test_tir_matmul():
    """Test TIR matrix multiplication correctness."""
    mod = tvm.ir.IRModule.from_expr(matmul_64x64)

    # Compile and execute
    with tvm.transform.PassContext(opt_level=0):
        rt_mod = tvm.build(mod, target="llvm")

    a_np = np.random.randn(64, 64).astype("float32")
    b_np = np.random.randn(64, 64).astype("float32")
    c_np = np.zeros((64, 64), dtype="float32")

    dev = tvm.cpu(0)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(c_np, dev)

    rt_mod["main"](a, b, c)

    # Verify against NumPy
    expected = a_np @ b_np
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5, atol=1e-5)

def test_tir_roundtrip():
    """Test TIR script roundtrip."""
    mod = tvm.ir.IRModule.from_expr(matmul_64x64)
    script = mod.script()
    mod_rt = tvm.ir.IRModule.from_expr(
        tvm.script.from_source(script)["main"]
    )
    assert tvm.ir.structural_equal(mod["main"], mod_rt["main"])

def test_tir_pass_transform():
    """Test that a TIR pass produces correct output."""
    mod = tvm.ir.IRModule.from_expr(matmul_64x64)

    # Apply a transformation
    mod_tiled = tvm.tir.transform.UnrollLoop(64)(mod)

    # Verify the transformed module still produces correct results
    with tvm.transform.PassContext(opt_level=0):
        rt_mod = tvm.build(mod_tiled, target="llvm")

    a = tvm.nd.array(np.random.randn(64, 64).astype("float32"))
    b = tvm.nd.array(np.random.randn(64, 64).astype("float32"))
    c = tvm.nd.array(np.zeros((64, 64), dtype="float32"))

    rt_mod["main"](a, b, c)
    expected = a.numpy() @ b.numpy()
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)
```

### 34.7.2 Relax Function Testing

Testing Relax functions follows a similar pattern but works with the Relax IR.

```python
import tvm
import numpy as np
from tvm import relax
from tvm.script import relax as R

@R.function
def simple_mlp(
    x: R.Tensor((1, 784), "float32"),
    w1: R.Tensor((256, 784), "float32"),
    b1: R.Tensor((256,), "float32"),
    w2: R.Tensor((10, 256), "float32"),
    b2: R.Tensor((10,), "float32"),
) -> R.Tensor((1, 10), "float32"):
    with R.dataflow():
        lv1 = R.matmul(x, R.permute_dims(w1, [1, 0]))
        lv2 = R.add(lv1, b1)
        lv3 = R.nn.relu(lv2)
        lv4 = R.matmul(lv3, R.permute_dims(w2, [1, 0]))
        lv5 = R.add(lv4, b2)
        R.output(lv5)
    return lv5

def test_relax_mlp():
    """Test Relax MLP function correctness."""
    mod = tvm.ir.IRModule.from_expr(simple_mlp)
    mod = relax.transform.LegalizeOps()(mod)

    # Compile with Relax VM
    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu(0))

    # Prepare inputs
    x_np = np.random.randn(1, 784).astype("float32")
    w1_np = np.random.randn(256, 784).astype("float32")
    b1_np = np.random.randn(256).astype("float32")
    w2_np = np.random.randn(10, 256).astype("float32")
    b2_np = np.random.randn(10).astype("float32")

    args = [
        tvm.nd.array(x_np),
        tvm.nd.array(w1_np),
        tvm.nd.array(b1_np),
        tvm.nd.array(w2_np),
        tvm.nd.array(b2_np),
    ]

    result = vm["simple_mlp"](*args)

    # Compute reference with NumPy
    ref = x_np @ w1_np.T + b1_np
    ref = np.maximum(ref, 0)
    ref = ref @ w2_np.T + b2_np

    np.testing.assert_allclose(result.numpy(), ref, rtol=1e-5, atol=1e-5)
```

### 34.7.3 End-to-End Model Testing

End-to-end tests verify that entire models can be imported, compiled, and executed correctly.

```python
import tvm
import numpy as np
from tvm import relay
import pytest

def test_resnet18_e2e():
    """End-to-end test: import ResNet-18, compile, and verify output."""
    # Import from PyTorch
    try:
        import torch
        import torchvision
    except ImportError:
        pytest.skip("PyTorch not available")

    model = torchvision.models.resnet18(pretrained=False)
    model.eval()

    # Create sample input
    input_name = "input"
    input_shape = (1, 3, 224, 224)
    input_data = np.random.randn(*input_shape).astype("float32")

    # Convert to TorchScript
    scripted_model = torch.jit.trace(
        model, torch.from_numpy(input_data)
    )

    # Import to Relay
    mod, params = relay.frontend.from_pytorch(
        scripted_model,
        [(input_name, input_shape)],
    )

    # Compile
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target, params=params)

    # Execute
    dev = tvm.cpu(0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input(input_name, input_data)
    runtime.run()

    # Get TVM output
    tvm_output = runtime.get_output(0).numpy()

    # Get PyTorch reference output
    with torch.no_grad():
        torch_output = model(torch.from_numpy(input_data)).numpy()

    # Compare
    np.testing.assert_allclose(tvm_output, torch_output, rtol=1e-3, atol=1e-3)
```

### 34.7.4 Numerical Correctness Verification

A comprehensive numerical verification helper:

```python
import tvm
import numpy as np
from tvm import relay

def verify_relay_op(relay_op, input_shapes, dtypes, ref_func,
                    target="llvm", rtol=1e-5, atol=1e-5):
    """
    Verify a Relay operation produces numerically correct results.

    Parameters
    ----------
    relay_op : relay.Expr
        The Relay expression to test.
    input_shapes : list[tuple]
        Shapes for each input variable.
    dtypes : list[str]
        Data types for each input variable.
    ref_func : callable
        Reference implementation (NumPy-based).
    target : str
        Compilation target.
    rtol, atol : float
        Relative and absolute tolerance for comparison.
    """
    # Create variables
    vars = []
    inputs_np = []
    for shape, dtype in zip(input_shapes, dtypes):
        var = relay.var("x", shape=shape, dtype=dtype)
        vars.append(var)
        inputs_np.append(np.random.randn(*shape).astype(dtype))

    # Build expression
    expr = relay_op(*vars) if callable(relay_op) else relay_op
    mod = tvm.IRModule.from_expr(expr)

    # Compile
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target=target)

    # Execute
    dev = tvm.cpu(0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    for i, (inp, var) in enumerate(zip(inputs_np, vars)):
        runtime.set_input(var.name_hint, inp)
    runtime.run()

    # Get result
    tvm_output = runtime.get_output(0).numpy()

    # Compute reference
    ref_output = ref_func(*inputs_np)

    # Compare
    np.testing.assert_allclose(tvm_output, ref_output, rtol=rtol, atol=atol)
    print(f"Verification passed: rtol={rtol}, atol={atol}")

# Example usage
def test_relu_numerical():
    verify_relay_op(
        relay_op=lambda x: relay.nn.relu(x),
        input_shapes=[(128,)],
        dtypes=["float32"],
        ref_func=lambda x: np.maximum(x, 0),
    )

def test_matmul_numerical():
    verify_relay_op(
        relay_op=lambda a, b: relay.nn.matmul(a, b),
        input_shapes=[(32, 64), (64, 48)],
        dtypes=["float32", "float32"],
        ref_func=lambda a, b: a @ b,
    )
```

---

## 34.8 CI/CD Integration

### 34.8.1 GitHub Actions

TVM uses GitHub Actions for continuous integration. The CI configuration is in `.github/workflows/`.

```yaml
# Example: .github/workflows/ci.yml (simplified)
name: TVM CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.8", "3.9", "3.10"]
        include:
          - os: ubuntu-latest
            python-version: "3.10"
            gpu: true

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-xdist

      - name: Configure CMake
        run: |
          mkdir build
          cd build
          cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DUSE_CUDA=${{ matrix.gpu || 'OFF' }} \
            -DUSE_LLVM=ON

      - name: Build
        run: |
          cd build
          make -j$(nproc)

      - name: Run tests
        env:
          TVM_HOME: ${{ github.workspace }}
          PYTHONPATH: ${{ github.workspace }}/python
        run: |
          pytest tests/python/unittest/ -v -n 4 --timeout=300
```

### 34.8.2 Test Matrix

The CI test matrix covers multiple combinations of:
- **Operating systems**: Linux, macOS, Windows
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **GPU availability**: CPU-only, CUDA, ROCm
- **Build types**: Release, Debug, with sanitizers

```bash
# Running a subset of tests locally to match CI

# CPU-only unit tests
pytest tests/python/unittest/ -v --timeout=300

# Relay tests
pytest tests/python/relay/ -v -k "test_pass" --timeout=600

# GPU tests (requires CUDA)
pytest tests/python/unittest/test_runtime_cuda.py -v
pytest tests/python/topi/test_topi_cuda.py -v

# Specific frontend tests
pytest tests/python/contrib/test_onnx.py -v
pytest tests/python/contrib/test_pytorch.py -v

# Run with coverage
pytest tests/python/unittest/ --cov=tvm --cov-report=html
```

### 34.8.3 Nightly Builds

TVM runs nightly builds for extended testing that covers:
- Full test suite across all platforms.
- Performance regression benchmarks.
- Documentation build verification.
- Stress tests for memory leaks and long-running operations.

```bash
# Nightly test script (simplified)
#!/bin/bash
set -e

echo "Starting nightly test run: $(date)"

# Build TVM
cd /workspace/tvm
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_LLVM=ON
make -j$(nproc)

# Set up environment
export TVM_HOME=/workspace/tvm
export PYTHONPATH=/workspace/tvm/python:$PYTHONPATH

# Run full test suite
cd /workspace/tvm
pytest tests/python/ -v --timeout=600 --junitxml=nightly-results.xml

# Run performance benchmarks
python tests/scripts/nightly_benchmark.py

# Check for memory leaks
valgrind --leak-check=full pytest tests/python/unittest/test_runtime_ndarray.py

echo "Nightly test run completed: $(date)"
```

---

## 34.9 Test Best Practices

### 34.9.1 Writing Deterministic Tests

Tests should be deterministic to avoid flaky CI failures.

```python
import tvm
import numpy as np

# Good: Use a fixed seed for reproducibility
def test_deterministic():
    np.random.seed(42)  # Fixed seed
    x_np = np.random.randn(128, 128).astype("float32")
    # ... test implementation

# Better: Use tvm.testing utilities that handle seeds
def test_with_testing_utils():
    x_np = tvm.testing.randn((128, 128), dtype="float32", seed=42)
    # ... test implementation
```

### 34.9.2 Test Isolation

Each test should be independent and not rely on state from other tests.

```python
import pytest
import tvm

# Good: Each test sets up its own state
def test_pass_a():
    mod = create_test_module()
    result = tvm.relay.transform.FoldConstant()(mod)
    # Verify result

def test_pass_b():
    mod = create_test_module()  # Fresh module, not shared
    result = tvm.relay.transform.SimplifyExpr()(mod)
    # Verify result

# Bad: Shared mutable state
_shared_mod = None  # Don't do this

def test_shared_state():
    global _shared_mod
    if _shared_mod is None:
        _shared_mod = create_test_module()
    # Tests become order-dependent
```

### 34.9.3 Handling Optional Dependencies

Tests that depend on optional packages should use `pytest.importorskip`.

```python
import pytest

def test_onnx_import():
    """Test ONNX model import."""
    onnx = pytest.importorskip("onnx")
    # Now safe to use onnx
    model = onnx.load("test_model.onnx")
    # ...

def test_pytorch_frontend():
    """Test PyTorch frontend."""
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    # ...
```

### 34.9.4 Timeout Configuration

Set appropriate timeouts for tests to prevent CI hangs.

```python
import pytest

@pytest.mark.timeout(60)  # 60-second timeout
def test_long_running_compilation():
    """A test that may take a while."""
    # ...

# Or configure globally in pytest.ini / pyproject.toml
# [tool.pytest.ini_options]
# timeout = 300
```

---

## 34.10 Summary

Effective testing and benchmarking in TVM involves:
- Using pytest with TVM-specific decorators for target parametrization.
- Leveraging `tvm.testing` utilities for numerical comparison and device detection.
- Using `time_evaluator` for accurate performance measurements with warmup and repetition.
- Applying statistical methods for variance analysis and regression detection.
- Writing deterministic, isolated tests with proper handling of optional dependencies.
- Integrating with CI/CD for automated testing across platforms and configurations.
