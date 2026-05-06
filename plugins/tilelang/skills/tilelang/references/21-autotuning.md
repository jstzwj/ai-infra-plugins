# 21. Auto-Tuning

TileLang provides a comprehensive auto-tuning framework for finding optimal kernel configurations.
The auto-tuner systematically explores a configuration space, compiles each variant, benchmarks it,
and returns the best-performing kernel. This reference covers every aspect of the auto-tuning system.

---

## Table of Contents

1. [Overview](#overview)
2. [The @tilelang.autotune Decorator](#the-tilelangautotune-decorator)
3. [AutoTuner Class (Programmatic API)](#autotuner-class-programmatic-api)
4. [Configuration Space Definition](#configuration-space-definition)
5. [Input Tensor Supply](#input-tensor-supply)
6. [BitBLAS Roller for Device-Aware Recommendations](#bitblas-roller-for-device-aware-recommendations)
7. [Cartesian Product Search Space Generation](#cartesian-product-search-space-generation)
8. [Parallel Compilation with num_workers](#parallel-compilation-with-num_workers)
9. [Performance Benchmarking](#performance-benchmarking)
10. [Caching and Result Reuse](#caching-and-result-reuse)
11. [Result Recording and Database Storage](#result-recording-and-database-storage)
12. [Heuristic Configurations Based on GPU Architecture](#heuristic-configurations-based-on-gpu-architecture)
13. [Carver-Based Configuration Generation](#carver-based-configuration-generation)
14. [Template-Specific Auto-Tuning](#template-specific-auto-tuning)
15. [Performance Regression Testing](#performance-regression-testing)
16. [Complete Example: Autotuned GEMM](#complete-example-autotuned-gemm)
17. [Environment Variables](#environment-variables)

---

## Overview

Auto-tuning in TileLang addresses the fundamental challenge that optimal kernel parameters depend on
GPU architecture, problem dimensions, and data types. The auto-tuner automates the process of:

1. Defining a search space of configurable parameters (block sizes, pipeline stages, thread counts)
2. Compiling each configuration in parallel
3. Benchmarking each compiled kernel with realistic inputs
4. Optionally validating correctness against a reference implementation
5. Caching results for future reuse
6. Returning the best-performing compiled kernel

The two primary interfaces are:

- **`@tilelang.autotune` decorator**: Declarative, used with `@tilelang.jit` decorated functions
- **`AutoTuner` class**: Programmatic, for fine-grained control over the tuning process

---

## The @tilelang.autotune Decorator

The `@tilelang.autotune` decorator wraps a `@tilelang.jit` decorated function to automatically
search for the best kernel configuration. It must be applied *after* `@tilelang.jit` (i.e., it
appears above `@tilelang.jit` in the source code).

### Signature

```python
def autotune(
    func: Callable | PrimFunc | None = None,
    *,
    configs: dict | Callable,
    # Profile arguments
    warmup: int = 25,
    rep: int = 100,
    timeout: int = 100,
    # Compile arguments
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
    ref_prog: Callable = None,
    supply_prog: Callable = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    max_mismatched_ratio: float = 0.01,
    skip_check: bool = False,
    manual_check_prog: Callable = None,
    cache_input_tensors: bool = False,
):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `configs` | `dict \| Callable` | required | Configuration space to explore. Either a list of dicts or a callable that returns one. |
| `warmup` | `int` | `25` | Number of warmup iterations before timing measurements. |
| `rep` | `int` | `100` | Number of repetitions for timing measurements. |
| `timeout` | `int` | `100` | Maximum time in seconds per configuration benchmark. |
| `supply_type` | `TensorSupplyType` | `Auto` | How to generate input tensors for benchmarking. |
| `ref_prog` | `Callable` | `None` | Reference implementation for correctness validation. |
| `supply_prog` | `Callable` | `None` | Custom function to supply input tensors. |
| `rtol` | `float` | `1e-2` | Relative tolerance for correctness checks. |
| `atol` | `float` | `1e-2` | Absolute tolerance for correctness checks. |
| `max_mismatched_ratio` | `float` | `0.01` | Maximum allowed ratio of mismatched elements. |
| `skip_check` | `bool` | `False` | Whether to skip correctness validation. |
| `manual_check_prog` | `Callable` | `None` | Custom validation function. |
| `cache_input_tensors` | `bool` | `False` | Whether to reuse input tensors across configurations. |

### Basic Usage

```python
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools


def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        block_K=[32, 64],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )
    return [
        dict(zip(iter_params, values))
        for values in itertools.product(*iter_params.values())
    ]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=None, block_N=None, block_K=None,
           num_stages=None, threads=None, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


# Run with auto-tuning -- all tunable params left as None
result = matmul(4096, 4096, 4096)
print(f"Best latency: {result.latency}")
print(f"Best config: {result.config}")

# Run with explicit params -- skips auto-tuning
kernel = matmul(4096, 4096, 4096, block_M=128, block_N=256, block_K=64,
                num_stages=3, threads=256)
```

### Key Behavior Notes

1. **Tunable parameters must have default value `None`**: Parameters in the function signature
   that the auto-tuner should search must be defaulted to `None`. When called without explicit
   values, the auto-tuner fills them from the configuration space.

2. **Skipping auto-tune**: If all tunable parameters are explicitly provided, the auto-tuner
   detects this and skips the search, directly compiling with the given parameters.

3. **The decorator must wrap a `@tilelang.jit` function**: The `@autotune` decorator can only
   be applied to instances of `JITImpl`, which is what `@tilelang.jit` produces.

---

## AutoTuner Class (Programmatic API)

For advanced use cases requiring fine-grained control, the `AutoTuner` class provides a
programmatic interface.

### Creating an AutoTuner

```python
from tilelang.autotuner import AutoTuner
import tilelang as tl

# Create from a kernel function and config list
autotuner = AutoTuner.from_kernel(kernel=my_kernel_func, configs=config_list)
```

### Setting Compile Arguments

```python
autotuner.set_compile_args(
    out_idx=[-1],             # Output tensor indices
    target="auto",            # Target: "auto", "cuda", "hip"
    execution_backend="auto", # Backend: "auto", "tvm_ffi", "cython", "nvrtc", "torch"
    target_host=None,         # Host target for cross-compilation
    verbose=False,            # Verbose output
    pass_configs=None,        # Additional pass configurations
)
```

The `set_compile_args` method supports method chaining and reads defaults from environment
variables when parameters are `None`:

| Environment Variable | Default | Purpose |
|---------------------|---------|---------|
| `TILELANG_TARGET` | `"auto"` | Default compilation target |
| `TILELANG_EXECUTION_BACKEND` | `"auto"` | Default execution backend |
| `TILELANG_VERBOSE` | `"0"` | Enable verbose compilation |

### Setting Profile Arguments

```python
autotuner.set_profile_args(
    warmup=25,                                # Warmup iterations
    rep=100,                                  # Benchmark repetitions
    timeout=30,                               # Timeout per config (seconds)
    supply_type=tl.TensorSupplyType.Randn,    # Input tensor supply type
    ref_prog=ref_program,                     # Reference implementation
    supply_prog=None,                         # Custom tensor supply function
    rtol=1e-2,                                # Relative tolerance
    atol=1e-2,                                # Absolute tolerance
    max_mismatched_ratio=0.01,                # Max mismatch ratio
    skip_check=False,                         # Skip correctness check
    manual_check_prog=None,                   # Custom check program
    cache_input_tensors=False,                # Cache input tensors
    backend="event",                          # Profiler backend
)
```

### Running the Auto-Tuner

```python
# Run with default warmup/rep
result = autotuner.run()

# Run with custom timing parameters
result = autotuner.run(warmup=50, rep=200, timeout=60)
```

### ProfileArgs Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup` | `int` | `25` | Warmup iterations before timing |
| `rep` | `int` | `100` | Number of timing repetitions |
| `timeout` | `int` | `30` | Max seconds per configuration |
| `backend` | `str` | `"event"` | Profiler backend: `"event"`, `"cupti"`, `"cudagraph"` |
| `supply_type` | `TensorSupplyType` | `Auto` | Input tensor generation mode |
| `ref_prog` | `Callable` | `None` | Reference program for validation |
| `supply_prog` | `Callable` | `None` | Custom input tensor supplier |
| `rtol` | `float` | `1e-2` | Relative tolerance |
| `atol` | `float` | `1e-2` | Absolute tolerance |
| `max_mismatched_ratio` | `float` | `0.01` | Maximum element mismatch ratio |
| `skip_check` | `bool` | `False` | Skip correctness validation |
| `manual_check_prog` | `Callable` | `None` | Custom validation function |
| `cache_input_tensors` | `bool` | `False` | Reuse input tensors across configs |

### CompileArgs Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out_idx` | `list[int] \| int` | `None` | Output tensor indices |
| `execution_backend` | `str` | `"auto"` | Kernel execution backend |
| `target` | `str \| Target` | `"auto"` | Compilation target |
| `target_host` | `str \| Target` | `None` | Host target |
| `verbose` | `bool` | `False` | Verbose output |
| `pass_configs` | `dict` | `None` | Compiler pass configurations |

### AutotuneResult

The `run()` method returns an `AutotuneResult` dataclass:

```python
@dataclass(frozen=True)
class AutotuneResult:
    latency: float | None = None       # Best achieved latency (ms)
    config: dict | None = None         # Configuration producing best latency
    ref_latency: float | None = None   # Reference implementation latency
    libcode: str | None = None         # Generated library code
    func: Callable | None = None       # Optimized PrimFunc
    kernel: Callable | None = None     # Compiled JITKernel
```

The `kernel` field is a fully compiled `JITKernel` that can be called directly:

```python
result = autotuner.run()
best_kernel = result.kernel
output = best_kernel(input_a, input_b)
```

---

## Configuration Space Definition

The configuration space defines the set of parameter combinations the auto-tuner will explore.
Each configuration is a dictionary mapping parameter names (matching the kernel function's
signature) to values.

### Block Sizes

Block sizes determine the tile dimensions for shared memory operations:

```python
config = {
    "block_M": 128,   # Tile size along the M dimension
    "block_N": 256,   # Tile size along the N dimension
    "block_K": 64,    # Tile size along the K (reduction) dimension
}
```

Guidelines for block sizes:
- Must be powers of 2 or multiples of warp sizes (32)
- Larger blocks reduce loop overhead but increase shared memory usage
- `block_K` typically 32 or 64 for FP16 tensor core operations
- `block_M` and `block_N` typically range from 64 to 256

### Pipeline Stages

Pipeline stages control software pipelining for overlapping memory transfers with computation:

```python
config = {
    "num_stages": 3,   # Number of pipeline stages (0 = no pipelining)
}
```

- `0`: No pipelining (synchronous loads)
- `1`: Single-buffered
- `2-3`: Multi-stage pipelining (recommended for Hopper/Ampere)
- Higher stages increase shared memory usage but improve overlap

### Thread Counts

Thread counts control the number of threads per CUDA block:

```python
config = {
    "threads": 256,    # Total threads per block
}
```

Common values: 128, 256, 512. Must be a multiple of the warp size (32).

### Warp Policies

Warp policies control how work is distributed across warps within a thread block:

```python
# Used with T.gemm() policy parameter
T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.Square)
T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullCol)
```

| Policy | Description |
|--------|-------------|
| `Square` | Distribute warps in a square grid (default) |
| `FullRow` | All warps work on different rows |
| `FullCol` | All warps work on different columns |

### Enable Rasterization (Swizzle)

Rasterization improves L2 cache locality by reordering block scheduling:

```python
config = {
    "enable_rasteration": True,  # Enable L2 cache swizzle optimization
}
```

---

## Input Tensor Supply

The auto-tuner needs input tensors for benchmarking and correctness validation. TileLang
provides several supply modes via the `TensorSupplyType` enum.

### TensorSupplyType Options

```python
import tilelang

# Available supply types:
tilelang.TensorSupplyType.Auto      # Automatically select based on dtype
tilelang.TensorSupplyType.Rand      # Uniform random [0, 1)
tilelang.TensorSupplyType.Randn     # Normal distribution (mean=0, std=1)
tilelang.TensorSupplyType.Integer   # Random integers in a safe range
tilelang.TensorSupplyType.Ones      # All ones
tilelang.TensorSupplyType.Zeros     # All zeros
tilelang.TensorSupplyType.Normal    # Normal distribution
```

### Choosing a Supply Type

| Supply Type | Best For | Notes |
|-------------|----------|-------|
| `Auto` | General use | Avoids NaN/Inf-producing values |
| `Rand` | FP16/BF16 kernels | Uniform distribution |
| `Randn` | Attention kernels | More realistic distribution |
| `Integer` | Quantized/integer kernels | Produces safe integer ranges |
| `Ones` | Debugging | Deterministic, easy to verify |
| `Zeros` | Testing edge cases | Deterministic, easy to verify |

### Custom Supply Programs

For more control over input generation, provide a `supply_prog` callable:

```python
def custom_supply(params):
    """Generate custom input tensors.
    Args:
        params: List of KernelParam objects with shape/dtype info
    Returns:
        List of tensors matching the parameter specifications
    """
    import torch
    tensors = []
    for param in params:
        if hasattr(param, "shape") and hasattr(param, "dtype"):
            torch_dtype = param.dtype.as_torch()
            tensor = torch.randn(
                [int(s) for s in param.shape],
                dtype=torch_dtype,
                device="cuda"
            ) * 0.5  # Scale down to avoid overflow
            tensors.append(tensor)
        else:
            tensors.append(param)
    return tensors

autotuner.set_profile_args(supply_prog=custom_supply)
```

### Context-Based Input Supply

Use the `set_autotune_inputs` context manager to provide specific tensors:

```python
from tilelang.autotuner import set_autotune_inputs

a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

with set_autotune_inputs(a, b):
    result = autotuner.run()
```

The context manager uses a thread-local stack, making it safe for nested use:

```python
with set_autotune_inputs(a, b):
    # Within this context, auto-tuning uses (a, b) as inputs
    result = autotuner.run()
```

---

## BitBLAS Roller for Device-Aware Recommendations

The BitBLAS Roller (also called "roller" in the codebase) generates device-specific tile
recommendations based on GPU architecture and tensor core capabilities. This dramatically
reduces the search space by only considering configurations that are known to work well on
the target hardware.

### Using the Roller with MatmulTemplate

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA, CDNA
from tilelang.carver.roller.rasterization import NoRasterization
import torch

def get_roller_configs(M, N, K, topk=20):
    # Select architecture based on runtime
    arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")

    # Create the template with problem dimensions
    carve_template = MatmulTemplate(
        M=M,
        N=N,
        K=K,
        in_dtype=T.float16,
        out_dtype=T.float16,
        accum_dtype=T.float32,
    ).with_arch(arch)

    # Generate the equivalent function (verifies template correctness)
    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    # Get device-aware hints
    roller_hints = carve_template.recommend_hints(topk=topk)
    if roller_hints is None:
        raise ValueError("No Roller Hints Found for TensorCore Scheduling")

    # Convert hints to configuration dicts
    configs = []
    for hint in roller_hints:
        config = {}
        block_m, block_n = hint.block
        warp_m, warp_n = hint.warp
        block_rows, block_cols = block_m // warp_m, block_n // warp_n

        config["block_M"] = block_m
        config["block_N"] = block_n
        config["block_K"] = hint.rstep[0]
        config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
        config["thread_num"] = block_rows * block_cols * 32
        config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
        configs.append(config)

    return configs
```

### Roller Hint Properties

Each `hint` object from `recommend_hints()` contains:

| Property | Type | Description |
|----------|------|-------------|
| `hint.block` | `tuple(int, int)` | Block tile sizes (block_M, block_N) |
| `hint.warp` | `tuple(int, int)` | Warp tile sizes (warp_M, warp_N) |
| `hint.rstep` | `tuple(int)` | Reduction dimension step sizes |
| `hint.pipeline_stage` | `int` | Recommended pipeline stages |
| `hint.rasterization_plan` | `RasterizationPlan` | L2 cache rasterization strategy |

### Architecture Support

```python
from tilelang.carver.arch import CUDA, CDNA

# NVIDIA GPUs
cuda_arch = CUDA("cuda")

# AMD GPUs (ROCm)
cdna_arch = CDNA("hip")
```

---

## Cartesian Product Search Space Generation

The most common approach to defining a search space is using `itertools.product` to generate
all combinations of candidate values.

### Standard Approach

```python
import itertools

def get_configs():
    iter_params = dict(
        block_M=[64, 128, 256],
        block_N=[64, 128, 256],
        block_K=[32, 64],
        num_stages=[0, 1, 2, 3],
        thread_num=[128, 256],
        enable_rasterization=[True, False],
    )
    _configs = list(itertools.product(*iter_params.values()))
    return [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_stages": c[3],
            "thread_num": c[4],
            "enable_rasteration": c[5],
        }
        for c in _configs
    ]
```

### Filtering Invalid Configurations

For attention kernels, shared memory constraints limit valid configurations:

```python
def get_filtered_configs(max_shared_mem=100 * 1024, dim=128, dtype_bytes=2):
    block_sizes = (64, 128, 256)
    thread_options = (128, 256, 512)
    num_stages_range = (2, 3)
    warp_alignment = 16

    valid_configs = []
    for block_M, block_N in itertools.product(block_sizes, repeat=2):
        for threads in thread_options:
            warp_count = threads // 32
            warp_M = block_M // warp_count
            warp_N = block_N // warp_count

            # Warp alignment check
            if warp_M % warp_alignment != 0 or warp_N % warp_alignment != 0:
                continue

            # Shared memory constraint
            shared_mem = 2 * dtype_bytes * dim * (block_M + block_N)
            if shared_mem > max_shared_mem:
                continue

            for num_stages in num_stages_range:
                valid_configs.append({
                    "block_M": block_M,
                    "block_N": block_N,
                    "num_stages": num_stages,
                    "threads": threads,
                })
    return valid_configs
```

### Callable Configuration Generators

The `configs` parameter accepts a callable, which receives the kernel's arguments:

```python
def generate_configs(*kernel_args, **kernel_kwargs):
    M, N, K = kernel_args[0], kernel_args[1], kernel_args[2]
    # Generate configs based on problem size
    block_sizes = [128, 256] if max(M, N) > 2048 else [64, 128]
    configs = []
    for bm in block_sizes:
        for bn in block_sizes:
            configs.append({"block_M": bm, "block_N": bn, "block_K": 32})
    return configs

@autotune(configs=generate_configs)
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=None, block_N=None, block_K=None, ...):
    ...
```

---

## Parallel Compilation with num_workers

The auto-tuner compiles configurations in parallel using a `ThreadPoolExecutor`. The number
of worker threads is controlled by environment variables.

### Worker Thread Configuration

```python
# Environment variables controlling parallelism:
TILELANG_AUTO_TUNING_CPU_UTILITIES = "0.9"   # Use 90% of available CPUs
TILELANG_AUTO_TUNING_CPU_COUNTS = "-1"        # -1 = auto-detect
TILELANG_AUTO_TUNING_MAX_CPU_COUNT = "-1"     # -1 = no limit
```

### How Workers Are Determined

1. If `TILELANG_AUTO_TUNING_CPU_COUNTS > 0`: Use exactly that many workers
2. Otherwise: `num_workers = max(1, int(available_cpus * TILELANG_AUTO_TUNING_CPU_UTILITIES))`
3. If `TILELANG_AUTO_TUNING_MAX_CPU_COUNT > 0`: Clamp to this maximum

### Parallel Compilation Flow

```python
# Internal flow (simplified):
pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

# Submit all compilation tasks
futures = []
for config in configs:
    future = pool.submit(compile_func, **config)
    futures.append(future)

# Collect results as they complete
for future in tqdm(concurrent.futures.as_completed(futures)):
    result = future.result()
    compiled_kernels.append(result)

# Benchmark sequentially (thread safety requirements)
for kernel, config in compiled_kernels:
    latency = benchmark(kernel)
```

Key notes:
- Compilation is parallelized across configurations
- Benchmarking runs sequentially to avoid GPU contention
- Each worker thread sets the correct CUDA device

---

## Performance Benchmarking

### Benchmarking Process

Each configuration goes through the following benchmarking pipeline:

1. **Warmup phase**: Run the kernel `warmup` times to warm up the GPU and stabilize clocks
2. **Timing phase**: Run the kernel `rep` times and measure total wall time
3. **Latency calculation**: `latency = total_time / rep` (in milliseconds)

### TFLOPS Calculation

```python
# For a GEMM with dimensions M x N x K:
total_flops = 2 * M * N * K  # Multiply-add = 2 FLOPs
latency_ms = profiler.do_bench()  # In milliseconds
tflops = total_flops / latency_ms * 1e-9
```

### Profiler Backends

Three benchmarking backends are available:

| Backend | Description | Accuracy | Overhead |
|---------|-------------|----------|----------|
| `"event"` | CUDA event timing | Good | Low |
| `"cupti"` | CUPTI profiling | High | Medium |
| `"cudagraph"` | CUDA graph timing | Good | Higher setup |

```python
# Using different backends:
autotuner.set_profile_args(backend="cupti")  # Most accurate
autotuner.set_profile_args(backend="event")  # Default, good balance
autotuner.set_profile_args(backend="cudagraph")  # For graph-captured kernels
```

### Correctness Validation

When a `ref_prog` is provided, the auto-tuner validates each configuration:

```python
def ref_program(A, B):
    return A @ B.T

autotuner.set_profile_args(
    ref_prog=ref_program,
    rtol=1e-2,
    atol=1e-2,
    max_mismatched_ratio=0.01,
)
```

The validation process:
1. Generate input tensors using the supply mechanism
2. Run both the TileLang kernel and the reference program
3. Compare outputs using `torch.testing.assert_close` with the specified tolerances
4. If the mismatch ratio exceeds `max_mismatched_ratio`, the configuration is rejected

### Manual Checking

For custom validation logic:

```python
def manual_check(ref_output, kernel_output, inputs):
    # Custom validation: check only specific elements
    diff = (ref_output - kernel_output).abs()
    max_diff = diff.max().item()
    return max_diff < 0.1  # Return True if acceptable

autotuner.set_profile_args(
    ref_prog=ref_program,
    manual_check_prog=manual_check,
)
```

---

## Caching and Result Reuse

### Two-Level Cache

The auto-tuner uses a two-level cache to avoid redundant tuning:

1. **In-memory cache**: A class-level dictionary `_memory_cache` shared across instances
2. **Disk cache**: Persistent storage in `~/.tilelang/cache/autotuner/`

### Cache Key Generation

The cache key is a SHA-256 hash of:

```python
key_data = {
    "version": __version__,               # TileLang version
    "op_parameters": tuple(op_params),     # Operation parameters
    "extra_parameters": extra_params,      # Closure variables
    "func_source": func_source,            # Function source code
    "configs": configs,                    # Configuration space
    "compile_args": hash(compile_args),    # Compilation settings hash
    "profile_args": hash(profile_args),    # Profiling settings hash
}
key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
```

### Cache Behavior

```python
# Check in-memory cache first
if key in self._memory_cache:
    return cached_result

# Then check disk cache
result = self._load_result_from_disk(key)
if result is not None:
    self._memory_cache[key] = result  # Populate memory cache
    return result

# If not cached, run auto-tuning
result = self.run()

# Save to both caches
self._memory_cache[key] = result
self._save_result_to_disk(key, result)
```

### Disabling the Cache

```python
# Via environment variable
os.environ["TILELANG_AUTO_TUNING_DISABLE_CACHE"] = "1"

# Or globally
tilelang.disable_cache()
```

---

## Result Recording and Database Storage

### Disk Storage Format

Each auto-tuned result is stored as a directory under `~/.tilelang/cache/autotuner/<hash>/`:

```
<cache_hash>/
  best_config.json       # Best configuration found
  function.pkl           # Pickled PrimFunc (cloudpickle)
  out_idx.json           # Output tensor indices
  latency.json           # Best and reference latencies
  device_kernel.cu       # Device kernel source code
  host_kernel.cu         # Host kernel source code
  executable.so          # Compiled kernel library (tvm_ffi backend)
  params.pkl             # Kernel parameters (cloudpickle)
```

### Atomic Writes

Results are saved atomically to prevent corruption:

1. Files are written to a temporary staging directory
2. The staging directory is atomically renamed to the final location
3. If another process writes the same key concurrently, the first complete write wins

### Loading Cached Results

```python
# Automatic on autotuner.run()
result = autotuner.run()  # Checks cache first

# Manual loading
from pathlib import Path
from tilelang.autotuner.param import AutotuneResult, CompileArgs

cache_path = Path("~/.tilelang/cache/autotuner/<hash>")
compile_args = CompileArgs(out_idx=[-1], target="cuda")
result = AutotuneResult.load_from_disk(cache_path, compile_args)
```

---

## Heuristic Configurations Based on GPU Architecture

When auto-tuning is too expensive, heuristic configurations provide good defaults based on
the GPU architecture:

```python
import torch

def get_heuristic_config():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor

    if sm_version == 80:  # A100
        return {
            "block_M": 128, "block_N": 256, "block_K": 32,
            "num_stages": 2, "thread_num": 128,
            "enable_rasteration": True,
        }
    elif sm_version == 90:  # H100 (Hopper)
        return {
            "block_M": 128, "block_N": 256, "block_K": 64,
            "num_stages": 3, "thread_num": 256,
            "enable_rasteration": True,
        }
    else:
        return {
            "block_M": 128, "block_N": 256, "block_K": 32,
            "num_stages": 0, "thread_num": 128,
            "enable_rasteration": True,
        }
```

### Architecture-Specific Tuning Guidelines

| GPU Architecture | SM Version | Recommended block_K | Pipeline Stages | Notes |
|-----------------|------------|-------------------|-----------------|-------|
| Ampere (A100) | 8.0 | 32 | 2-3 | Good shared memory bandwidth |
| Ada Lovelace (4090) | 8.9 | 32-64 | 2-3 | High clock speeds |
| Hopper (H100) | 9.0 | 64 | 3+ | TMA, WGMMA support |
| Blackwell (B200) | 10.0 | 64+ | 3+ | TCGEN5 MMA |

---

## Carver-Based Configuration Generation

The Carver module provides template-based configuration generation with architecture-aware
recommendations.

### Available Templates

```python
from tilelang.carver.template import (
    MatmulTemplate,        # Matrix multiplication
    FlashAttentionTemplate, # Flash attention
    GEMVTemplate,          # Matrix-vector multiplication
    ConvTemplate,          # Convolution
    ElementwiseTemplate,   # Element-wise operations
    GeneralReduceTemplate, # General reductions
)
```

### MatmulTemplate Example

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA

template = MatmulTemplate(
    M=4096,
    N=4096,
    K=4096,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
).with_arch(CUDA("cuda"))

# Get the equivalent TileLang function
func = template.equivalent_function()

# Get recommended hints
hints = template.recommend_hints(topk=10)
```

### FlashAttentionTemplate

```python
from tilelang.carver.template import FlashAttentionTemplate

template = FlashAttentionTemplate(
    batch=8,
    heads=32,
    seq_len=4096,
    dim=128,
    is_causal=False,
    in_dtype="float16",
    accum_dtype="float32",
)
```

---

## Template-Specific Auto-Tuning

### MatmulTemplate with AutoTuner

```python
def get_best_config(M, N, K):
    def kernel(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
        @T.prim_func
        def main(A, B, C):
            # ... kernel definition ...
        return main

    configs = get_roller_configs(M, N, K, topk=20)

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            backend="event",
        )
    )
    return autotuner.run(warmup=3, rep=20)
```

### FlashAttentionTemplate with AutoTuner

```python
class FlashAttentionTuneSpace:
    def __init__(
        self,
        block_sizes=(64, 128, 256),
        thread_options=(128, 256, 512),
        num_stages_range=(2, 3),
        max_shared_mem=100 * 1024,
        warp_alignment=16,
        dim=128,
        dtype_bytes=2,
    ):
        self.block_sizes = block_sizes
        self.thread_options = thread_options
        self.num_stages_range = num_stages_range
        self.max_shared_mem = max_shared_mem
        self.warp_alignment = warp_alignment
        self.dim = dim
        self.dtype_bytes = dtype_bytes
```

---

## Performance Regression Testing

TileLang includes infrastructure for performance regression testing to detect performance
degradation across code changes.

### Regression Test Pattern

```python
def run_regression_perf(M=4096, N=4096, K=4096):
    config = get_heuristic_config()
    kernel = matmul(M, N, K, **config)
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")
```

### Using the Testing Infrastructure

```python
from tilelang.testing.perf_regression import PerformanceRegression

# Run regression test
regression = PerformanceRegression(
    baseline_latency=0.5,  # ms
    tolerance=0.1,         # 10% tolerance
)
latency = run_regression_perf()
assert regression.check(latency), f"Regression detected: {latency}ms vs baseline 0.5ms"
```

---

## Complete Example: Autotuned GEMM

This complete example demonstrates all auto-tuning features:

```python
import argparse
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA, CDNA
from tilelang.carver.roller.rasterization import NoRasterization
import torch


def ref_program(A, B):
    return A @ B.T


def get_configs(M, N, K, with_roller=False, topk=20):
    if with_roller:
        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        carve_template = MatmulTemplate(
            M=M, N=N, K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float32,
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]

        _configs = list(itertools.product(
            block_M, block_N, block_K,
            num_stages, thread_num, enable_rasterization,
        ))
        configs = [
            {
                "block_M": c[0], "block_N": c[1], "block_K": c[2],
                "num_stages": c[3], "thread_num": c[4],
                "enable_rasteration": c[5],
            }
            for c in _configs
        ]
    return configs


def get_best_config(M, N, K, with_roller=False, profile_backend="event"):
    def kernel(block_M, block_N, block_K, num_stages, thread_num, enable_rasteration):
        dtype = T.bfloat16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                         threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N, K, with_roller))
        .set_compile_args(out_idx=[-1], target="auto")
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
            backend=profile_backend,
        )
    )
    return autotuner.run(warmup=3, rep=20)


def main(M=4096, N=4096, K=4096, use_autotune=False, with_roller=False):
    if use_autotune:
        result = get_best_config(M, N, K, with_roller=with_roller)
        print(f"Best config: {result.config}")
        print(f"Best latency: {result.latency} ms")
        kernel = result.kernel
    else:
        config = get_heuristic_config()
        kernel = matmul(M, N, K, **config)

    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    latency = profiler.do_bench()
    ref_latency = profiler.do_bench(ref_program)
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

    print(f"TileLang latency: {latency} ms")
    print(f"Reference latency: {ref_latency} ms")
    print(f"TileLang TFlops: {2 * M * N * K / latency * 1e-9}")
    print(f"Reference TFlops: {2 * M * N * K / ref_latency * 1e-9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--use_autotune", action="store_true")
    parser.add_argument("--with_roller", action="store_true")
    args = parser.parse_args()
    main(args.m, args.n, args.k, args.use_autotune, args.with_roller)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TILELANG_CACHE_DIR` | `~/.tilelang/cache` | Root cache directory |
| `TILELANG_DISABLE_CACHE` | `"0"` | Disable all caching |
| `TILELANG_AUTO_TUNING_DISABLE_CACHE` | `"0"` | Disable auto-tuner cache only |
| `TILELANG_AUTO_TUNING_CPU_UTILITIES` | `"0.9"` | Fraction of CPUs for parallel compilation |
| `TILELANG_AUTO_TUNING_CPU_COUNTS` | `"-1"` | Exact worker count (-1 = auto) |
| `TILELANG_AUTO_TUNING_MAX_CPU_COUNT` | `"-1"` | Maximum worker count (-1 = unlimited) |
| `TILELANG_TARGET` | `"auto"` | Default compilation target |
| `TILELANG_EXECUTION_BACKEND` | `"auto"` | Default execution backend |
| `TILELANG_VERBOSE` | `"0"` | Enable verbose compilation |
| `TILELANG_PRINT_ON_COMPILATION` | `"1"` | Print kernel name on compilation |
| `TILELANG_TMP_DIR` | `<cache>/tmp` | Temporary file directory |

---

## Summary

The TileLang auto-tuning system provides:

- **Declarative API** via `@tilelang.autotune` decorator for easy integration
- **Programmatic API** via `AutoTuner` class for fine-grained control
- **Device-aware recommendations** via the BitBLAS Roller and Carver templates
- **Flexible configuration spaces** via Cartesian product or custom generators
- **Parallel compilation** with configurable worker counts
- **Robust benchmarking** with multiple profiler backends
- **Correctness validation** with configurable tolerances
- **Two-level caching** (in-memory + disk) for result reuse
- **Atomic persistence** for safe concurrent access
