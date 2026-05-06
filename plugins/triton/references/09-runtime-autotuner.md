# Triton Runtime Autotuner Reference

This document provides an exhaustive reference for the Triton autotuning system, located in `sources/triton/python/triton/runtime/autotuner.py`. The autotuner automatically searches over kernel configurations to find the one that delivers the best performance for a given set of input parameters.

---

## Table of Contents

1. [Overview](#overview)
2. [The Config Class](#the-config-class)
3. [The @triton.autotune Decorator](#the-tritonautotune-decorator)
4. [The Autotuner Class](#the-autotuner-class)
5. [The @triton.heuristics Decorator](#the-tritonheuristics-decorator)
6. [The Heuristics Class](#the-heuristics-class)
7. [Pruning Strategies](#pruning-strategies)
8. [Performance Model](#performance-model)
9. [Benchmarking](#benchmarking)
10. [Cache Management](#cache-management)
11. [CUDA Graph Integration](#cuda-graph-integration)
12. [Pre and Post Hooks](#pre-and-post-hooks)
13. [Early Config Pruning](#early-config-pruning)
14. [Warmup and Repetition](#warmup-and-repetition)
15. [Environment Variables and Knobs](#environment-variables-and-knobs)
16. [Complete Code Examples](#complete-code-examples)

---

## Overview

Triton kernels often have tunable meta-parameters (block sizes, number of warps, pipeline stages, etc.) that significantly impact performance. Manually selecting these parameters is tedious and hardware-dependent. The autotuner automates this process by:

1. Defining a search space of candidate configurations via `triton.Config` objects.
2. Benchmarking each configuration with the actual input data.
3. Caching the best configuration per unique set of key arguments.
4. Reusing cached results on subsequent calls with the same key arguments.

The autotuning system consists of these primary components:

- **`Config`** -- Represents a single kernel configuration (meta-parameters, warps, stages, etc.).
- **`@triton.autotune`** -- Decorator that wraps a `@triton.jit` kernel with autotuning behavior.
- **`Autotuner`** -- Internal class implementing the autotuning logic (benchmarking, caching, pruning).
- **`@triton.heuristics`** -- Decorator for computing meta-parameters via heuristic functions rather than exhaustive search.
- **`Heuristics`** -- Internal class implementing heuristic-based configuration.

The decorator ordering matters. `@triton.autotune` or `@triton.heuristics` must appear *above* `@triton.jit`:

```python
@triton.autotune(configs=[...], key=[...])
@triton.jit
def my_kernel(...):
    ...
```

---

## The Config Class

**File:** `autotuner.py`, lines 313-390

A `Config` object represents a single candidate configuration that the autotuner will evaluate. It bundles together tunable meta-parameters and hardware execution hints.

### Constructor

```python
class Config:
    def __init__(
        self,
        kwargs: dict,
        num_warps: int = 4,
        num_stages: int = 3,
        num_ctas: int = 1,
        maxnreg: Optional[int] = None,
        pre_hook: Optional[Callable] = None,
        ir_override: Optional[str] = None,
    )
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kwargs` | `dict[str, Any]` | Required | Dictionary of meta-parameters to pass to the kernel as keyword arguments. These become `tl.constexpr` values inside the kernel. |
| `num_warps` | `int` | `4` | Number of warps per thread block. Each warp contains 32 threads on NVIDIA GPUs. For example, `num_warps=8` means `8 * 32 = 256` threads per block. Valid values are typically powers of 2: 1, 2, 4, 8, 16, 32. |
| `num_stages` | `int` | `3` | Number of pipeline stages for software pipelining. Primarily useful for matrix multiplication workloads on SM80+ GPUs. More stages can overlap memory loads with computation but increase shared memory usage. |
| `num_ctas` | `int` | `1` | Number of thread blocks in a block cluster (SM90+ / Hopper architecture only). Block clusters allow cooperative groups across multiple CTAs. |
| `maxnreg` | `Optional[int]` | `None` | Maximum number of registers a single thread can use. Corresponds to the PTX `.maxnreg` directive. Limiting registers can increase occupancy at the cost of more register spills. |
| `pre_hook` | `Optional[Callable]` | `None` | A function called before the kernel launches, specific to this configuration. Receives a dict of all kernel arguments. |
| `ir_override` | `Optional[str]` | `None` | Filename of a user-defined IR file (`*.ttgir`, `*.llir`, `*.ptx`, or `*.amdgcn`) that overrides the compiler's generated IR for this config. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.kwargs` | `dict` | The meta-parameter dictionary passed in the constructor. |
| `self.num_warps` | `int` | Warps per CTA. |
| `self.num_ctas` | `int` | CTAs per cluster. |
| `self.num_stages` | `int` | Pipeline stages. |
| `self.maxnreg` | `Optional[int]` | Max registers per thread. |
| `self.pre_hook` | `Optional[Callable]` | Per-config pre-launch hook. |
| `self.ir_override` | `Optional[str]` | IR override filename. |

### Methods

#### `all_kwargs() -> dict`

Returns a merged dictionary of all configuration parameters, including both user-defined `kwargs` and the standard parameters (`num_warps`, `num_ctas`, `num_stages`, `maxnreg`, `ir_override`). Parameters that are `None` are excluded.

```python
config = Config({'BLOCK_SIZE': 128, 'NUM_THREADS': 64}, num_warps=8, num_stages=2)
config.all_kwargs()
# Returns: {'BLOCK_SIZE': 128, 'NUM_THREADS': 64, 'num_warps': 8, 'num_stages': 2, 'num_ctas': 1}
```

Note that `maxnreg` and `ir_override` are excluded from the result when they are `None`.

#### `__str__() -> str`

Returns a human-readable string representation listing all parameters:

```python
str(config)
# "BLOCK_SIZE: 128, NUM_THREADS: 64, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None"
```

#### `__hash__() -> int`

Computes a hash from the tuple of all kwargs items plus the `pre_hook`. This enables using `Config` objects as dictionary keys for timing lookups.

#### `__eq__(other) -> bool`

Two configs are equal if all their kwargs items and pre_hooks are identical. The comparison includes `pre_hook` but notably does **not** include `ir_override` in the equality check (only kwargs and pre_hook).

#### `__setstate__(state: dict)`

Restores a `Config` from a serialized state dictionary. Used during deserialization (e.g., loading from disk cache). Handles missing keys gracefully by providing defaults:

```python
def __setstate__(self, state):
    self.kwargs = state.get("kwargs", {})
    self.num_warps = state.get("num_warps", 4)
    self.num_stages = state.get("num_stages", 3)
    self.num_ctas = state.get("num_ctas", 1)
    self.maxnreg = state.get("maxnreg", None)
    self.pre_hook = state.get("pre_hook", None)
    self.ir_override = state.get("ir_override", None)
```

### Usage Examples

```python
import triton

# Basic config with a single meta-parameter
config1 = triton.Config({'BLOCK_SIZE': 128}, num_warps=4)

# Config with multiple meta-parameters and hardware tuning
config2 = triton.Config(
    {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
    num_warps=8,
    num_stages=4,
)

# Config with max register limit (Hopper/Ada)
config3 = triton.Config(
    {'BLOCK_SIZE': 256},
    num_warps=16,
    maxnreg=64,
)

# Config with CTA cluster (Hopper SM90+)
config4 = triton.Config(
    {'BLOCK_SIZE': 128},
    num_warps=8,
    num_ctas=2,
)

# Config with per-config pre_hook
def my_pre_hook(args):
    args['scratch_buffer'].zero_()

config5 = triton.Config(
    {'BLOCK_SIZE': 512},
    num_warps=8,
    pre_hook=my_pre_hook,
)

# Config with IR override
config6 = triton.Config(
    {'BLOCK_SIZE': 128},
    num_warps=4,
    ir_override='custom_kernel.ttgir',
)
```

---

## The @triton.autotune Decorator

**File:** `autotuner.py`, lines 393-459

The `@triton.autotune` decorator wraps a `@triton.jit` kernel function with autotuning logic. It must be placed above `@triton.jit` in the decorator stack.

### Full Signature

```python
def autotune(
    configs: list,
    key: list,
    prune_configs_by: Optional[dict] = None,
    reset_to_zero: Optional[list] = None,
    restore_value: Optional[list] = None,
    pre_hook: Optional[Callable] = None,
    post_hook: Optional[Callable] = None,
    warmup: Optional[int] = None,          # Deprecated
    rep: Optional[int] = None,             # Deprecated
    use_cuda_graph: bool = False,          # Deprecated
    do_bench: Optional[Callable] = None,
    cache_results: bool = False,
) -> Callable
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `configs` | `list[triton.Config]` | Required | A list of `triton.Config` objects defining the search space. If an empty list is provided, a single default config `Config({}, num_warps=4, num_stages=3, num_ctas=1)` is used. |
| `key` | `list[str]` | Required | A list of argument names whose values determine when autotuning should be re-evaluated. When any of these argument values change, the autotuner benchmarks all configs again. |
| `prune_configs_by` | `Optional[dict]` | `None` | A dictionary of pruning options. See [Pruning Strategies](#pruning-strategies). Fields: `'perf_model'`, `'top_k'`, `'early_config_prune'`. |
| `reset_to_zero` | `Optional[list[str]]` | `None` | A list of tensor argument names that will be zeroed before each config evaluation. Prevents accumulation of results across benchmark runs. |
| `restore_value` | `Optional[list[str]]` | `None` | A list of tensor argument names whose values will be saved before and restored after each config evaluation. More expensive but preserves exact input data. |
| `pre_hook` | `Optional[Callable]` | `None` | A function called before each kernel invocation. Overrides the default hook used for `reset_to_zero` and `restore_value`. Signature: `pre_hook(kwargs: dict, reset_only: bool)`. |
| `post_hook` | `Optional[Callable]` | `None` | A function called after each kernel invocation. Overrides the default hook for `restore_value`. Signature: `post_hook(kwargs: dict, exception: Optional[Exception])`. |
| `warmup` | `Optional[int]` | `None` | **Deprecated.** Warmup time in milliseconds for benchmarking. Use `do_bench` instead. |
| `rep` | `Optional[int]` | `None` | **Deprecated.** Repetition time in milliseconds for benchmarking. Use `do_bench` instead. |
| `use_cuda_graph` | `bool` | `False` | **Deprecated.** Whether to use CUDA graphs for benchmarking. Use `do_bench` instead. |
| `do_bench` | `Optional[Callable]` | `None` | Custom benchmark function. Signature: `do_bench(kernel_call: Callable, quantiles: list[float]) -> list[float]`. If not provided, uses `driver.active.get_benchmarker()`. |
| `cache_results` | `bool` | `False` | Whether to persist autotune timings to disk cache. Also controlled by the `TRITON_CACHE_AUTOTUNING` environment variable. |

### Return Value

Returns a decorator function that wraps the target kernel in an `Autotuner` instance.

### Basic Usage

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],  # Re-autotune when n_elements changes
)
@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### How the Key System Works

The `key` parameter specifies which kernel arguments are used to determine whether re-autotuning is needed. The autotuner builds a cache key tuple from:

1. The values of arguments listed in `key`.
2. The `.dtype` string of any argument that has a `dtype` attribute.

```python
# The cache key construction (simplified):
key = [_args[k] for k in self.keys if k in _args]
for _, arg in _args.items():
    if hasattr(arg, "dtype"):
        key.append(str(arg.dtype))
key = tuple(key)
```

When a kernel is called, the autotuner checks if the computed key exists in its cache. If it does, the cached best config is used directly. If not, all (pruned) configs are benchmarked and the best one is cached.

```python
# Example: key=['M', 'N']
# Call with M=128, N=256 -> key = (128, 256, 'torch.float32', 'torch.float32')
# Call with M=256, N=256 -> key = (256, 256, 'torch.float32', 'torch.float32') -> triggers re-autotuning
# Call with M=128, N=256 -> key = (128, 256, ...) -> uses cached config
```

### Decorator Ordering

The `@triton.autotune` decorator must be placed **above** `@triton.jit`. If `@triton.heuristics` is also used, `@triton.autotune` should be the outermost decorator:

```python
# Correct ordering:
@triton.autotune(configs=[...], key=[...])
@triton.jit
def kernel(...):
    ...

# Also valid: heuristics with autotune (autotune takes priority)
@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n'])})
@triton.autotune(configs=[...], key=[...])
@triton.jit
def kernel(...):
    ...
```

---

## The Autotuner Class

**File:** `autotuner.py`, lines 19-310

`Autotuner` is the internal class that implements autotuning logic. It extends `KernelInterface` and is created by the `@triton.autotune` decorator. Users typically do not instantiate it directly.

### Class Hierarchy

```
KernelInterface (Generic[T])
  └── Autotuner
```

### Constructor

```python
class Autotuner(KernelInterface):
    def __init__(
        self,
        fn,                          # The JIT function (or wrapped function)
        arg_names,                   # Argument names from the JIT function
        configs,                     # List of Config objects
        key,                         # List of key argument names
        reset_to_zero,               # Arguments to zero before each bench
        restore_value,               # Arguments to restore after each bench
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        cache_results=False,
    )
```

### Internal State

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.configs` | `list[Config]` | All candidate configurations. If an empty list was passed, defaults to `[Config({}, num_warps=4, num_stages=3, num_ctas=1)]`. |
| `self.keys` | `list[str]` | Argument names used for cache key construction. |
| `self.cache` | `Dict[Tuple, Config]` | In-memory cache mapping key tuples to best configs. |
| `self.arg_names` | `list[str]` | Names of the kernel's positional arguments. |
| `self.cache_results` | `bool` | Whether to persist results to disk. Determined by `cache_results` flag OR `TRITON_CACHE_AUTOTUNING` env var, but NOT in interpreter mode. |
| `self.reset_to_zero` | `list[str]` | Argument names whose tensors are zeroed before each bench run. |
| `self.restore_value` | `list[str]` | Argument names whose tensors are cloned before and restored after each bench run. |
| `self.pre_hook` | `Callable` | Hook called before kernel invocation. Default implementation handles reset_to_zero and restore_value. |
| `self.post_hook` | `Callable` | Hook called after kernel invocation. Default implementation handles restore_value. |
| `self.user_defined_pre_hook` | `bool` | Whether a user-provided pre_hook is active (vs. auto-generated one). |
| `self.user_defined_post_hook` | `bool` | Whether a user-provided post_hook is active. |
| `self.perf_model` | `Optional[Callable]` | Performance model function for pruning. |
| `self.configs_top_k` | `float or int` | Number of configs to keep after pruning (int or float <= 1.0 for ratio). |
| `self.early_config_prune` | `Optional[Callable]` | Custom early pruning function. |
| `self.fn` | The wrapped function | The JIT function being autotuned. |
| `self.base_fn` | `function` | The underlying Python function (unwrapped through any decorator layers). |
| `self._do_bench` | `Optional[Callable]` | User-provided or deprecated benchmark function. |
| `self.num_warmups` | `Optional[int]` | Deprecated warmup parameter. |
| `self.num_reps` | `Optional[int]` | Deprecated rep parameter. |
| `self.use_cuda_graph` | `bool` | Deprecated CUDA graph flag. |
| `self.nargs` | `Optional[dict]` | Dict mapping argument names to values. Set during `run()` and cleared after. |

### Methods

#### `do_bench` (cached property)

```python
@cached_property
def do_bench(self) -> Callable:
    if self._do_bench is None:
        return driver.active.get_benchmarker()
    return self._do_bench
```

Returns the benchmark function. If no custom `do_bench` was provided, delegates to the active GPU driver's `get_benchmarker()` method. The result is cached after first access via `@cached_property`.

#### `_bench(*args, config, **meta) -> list[float]`

Benchmarks a single configuration. This is the core benchmarking method.

**Parameters:**
- `*args` -- Positional arguments passed to the kernel.
- `config` -- The `Config` object to benchmark.
- `**meta` -- Additional meta-parameters (e.g., from heuristics).

**Returns:** A list of three timing values `[median, p20, p80]` from the benchmark function's quantile output, or `[inf, inf, inf]` if the config fails.

**Behavior:**

1. Checks for conflicts between `meta` kwargs and `config.kwargs`. Raises `ValueError` if any key appears in both.
2. Merges meta-parameters with config kwargs: `current = {**meta, **config.all_kwargs()}`.
3. Constructs `full_nargs = {**self.nargs, **current}` -- a dict of all kernel arguments.
4. Defines `kernel_call()` which:
   - Calls `config.pre_hook(full_nargs)` if the config has one.
   - Calls `self.pre_hook(full_nargs)`.
   - Calls `self.fn.run(*args, **current)` inside a try/except.
   - Calls `self.post_hook(full_nargs, exception=e)` if an error occurs, then re-raises.
   - Calls `self.post_hook(full_nargs, exception=None)` on success.
5. Calls `self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))` to measure timing.
6. If the kernel fails with `OutOfResources`, `CompileTimeAssertionFailure`, or `PTXASError`, returns `[float("inf"), float("inf"), float("inf")]` to mark it as invalid.
7. If `TRITON_PRINT_AUTOTUNING` is set, prints progress messages.

#### `run(*args, **kwargs) -> Any`

The main entry point invoked when the autotuned kernel is called. Orchestrates the entire autotuning process.

**Behavior flow:**

1. **Build argument mapping:** `self.nargs = dict(zip(self.arg_names, args))`
2. **Single-config shortcut:** If there is only one config, skip benchmarking entirely and use that config directly.
3. **Build cache key:** Extracts key argument values and dtypes into a tuple.
4. **Cache hit:** If the key is already in `self.cache`, use the cached config.
5. **Cache miss:**
   a. Prune configs via `self.prune_configs(kwargs)`.
   b. Benchmark all pruned configs.
   c. Select the config with the minimum timing (median).
   d. Call `self.pre_hook(full_nargs, reset_only=True)` to reset state.
   e. Optionally cache results to disk via `check_disk_cache()`.
6. **Listener notification:** If `knobs.autotuning.listener` is set, calls it with autotuning results.
7. **Print results:** If `TRITON_PRINT_AUTOTUNING` is set, prints timing and best config.
8. **Execute kernel:** Calls `config.pre_hook()` if defined, then `self.fn.run(*args, **kwargs, **config.all_kwargs())`.
9. **Cleanup:** Sets `self.nargs = None`.

**Key implementation details:**

```python
# Cache key construction
key = [_args[key] for key in self.keys if key in _args]
for _, arg in _args.items():
    if hasattr(arg, "dtype"):
        key.append(str(arg.dtype))
key = tuple(key)

# Config selection
timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
self.cache[key] = builtins.min(timings, key=timings.get)
```

The `timings` dictionary maps each config to its benchmark result list. `builtins.min` selects the config with the smallest first element (median timing).

#### `prune_configs(kwargs: dict) -> list[Config]`

Reduces the number of configurations to benchmark using pruning strategies.

**Pruning pipeline:**

1. **Early pruning:** If `self.early_config_prune` is set, calls it with `(configs, nargs, **kwargs)`. Raises `AutotunerError` if the result is empty.
2. **Performance model pruning:** If `self.perf_model` is set:
   - Resolves `top_k`: if it is a float <= 1.0, it is interpreted as a fraction of total configs; if an int, it is an absolute count.
   - If the number of pruned configs exceeds `top_k`, evaluates the performance model for each config and keeps only the top-k lowest estimated times.

```python
def prune_configs(self, kwargs: Dict) -> List[Config]:
    pruned_configs = self.configs
    # Step 1: Early pruning
    if self.early_config_prune:
        pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if not pruned_configs:
            raise AutotunerError("No valid autotuner configs after pruning.")
    # Step 2: Performance model pruning
    if self.perf_model:
        top_k = self.configs_top_k
        if isinstance(top_k, float) and top_k <= 1.0:
            top_k = int(len(self.configs) * top_k)
        elif not isinstance(top_k, int):
            raise TypeError("top_k must be either a float <= 1.0 or an int")
        if len(pruned_configs) > top_k:
            est_timing = {
                config: self.perf_model(**self.nargs, **kwargs, **config.all_kwargs())
                for config in pruned_configs
            }
            pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
    return pruned_configs
```

#### `warmup(*args, **kwargs) -> list`

Pre-compiles kernel binaries for all pruned configs without executing them. Useful for ahead-of-time compilation.

```python
def warmup(self, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))
    ret = []
    for autotune_config in self.prune_configs(kwargs):
        ret.append(self.fn.warmup(
            *args,
            **kwargs,
            **autotune_config.all_kwargs(),
        ))
    self.nargs = None
    return ret
```

#### `check_disk_cache(tuning_key, configs, bench_fn) -> bool`

Manages persistent disk caching of autotune results.

**Parameters:**
- `tuning_key` -- The cache key tuple.
- `configs` -- The pruned config list.
- `bench_fn` -- A callable that runs benchmarks and populates `self.cache`.

**Returns:** `True` if a cached result was found on disk, `False` otherwise.

**Behavior:**

1. If any config has a `pre_hook`, skip disk caching (pre_hooks are not serializable) and run benchmarks directly.
2. Computes a SHA-256 hash from: Triton version key, backend hash, JIT function cache key, environment variables, tuning key, and all config strings.
3. Uses `get_cache_manager(cache_key)` to look up or store a JSON file named `{kernel_name}.autotune.json`.
4. On cache hit: loads the JSON, reconstructs `Config` objects and timings, selects the best config.
5. On cache miss: runs `bench_fn()`, then saves timings to disk (excluding configs with pre_hooks).

**Cache key components:**

```python
cache_key = [
    triton_key(),                                              # Triton version + all source hashes
    make_backend(driver.active.get_current_target()).hash(),   # Backend hash
    fn.cache_key,                                              # JIT function source hash
    str(sorted(env_vars.items())),                             # Environment variables
    str(tuning_key),                                           # Runtime tuning key
] + [str(c) for c in configs]                                 # Config representations
```

**Cache file format:**

```json
{
    "key": <tuning_key>,
    "configs_timings": [
        [{"kwargs": {...}, "num_warps": 4, ...}, [0.123, 0.110, 0.140]],
        ...
    ]
}
```

---

## The @triton.heuristics Decorator

**File:** `autotuner.py`, lines 475-496

The `@triton.heuristics` decorator provides an alternative to autotuning when the optimal meta-parameter values can be determined analytically from the input arguments, making exhaustive benchmarking unnecessary.

### Full Signature

```python
def heuristics(
    values: dict,
) -> Callable
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `dict[str, Callable[[dict[str, Any]], Any]]` | A dictionary mapping meta-parameter names to functions that compute their values. Each function receives a single dict argument containing all kernel arguments (both positional and keyword). |

### Return Value

Returns a decorator function that wraps the target kernel in a `Heuristics` instance.

### Usage Example

```python
import triton
import triton.language as tl

@triton.heuristics(
    values={
        'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_elements']),
    }
)
@triton.jit
def copy_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)
```

### Multiple Heuristic Values

```python
@triton.heuristics(
    values={
        'BLOCK_M': lambda args: 64 if args['M'] > 128 else 32,
        'BLOCK_N': lambda args: 64 if args['N'] > 128 else 32,
        'BLOCK_K': lambda args: triton.cdiv(args['K'], 4) if args['K'] > 256 else 32,
    }
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    ...
```

### How Heuristic Functions Receive Arguments

The heuristic function receives a single dictionary containing all kernel arguments merged from positional and keyword arguments:

```python
# Inside Heuristics.run():
for v, heur in self.values.items():
    kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
```

This means the heuristic function can access any kernel argument by name:

```python
def compute_block_size(args):
    n = args['n_elements']          # positional argument
    dtype = args.get('dtype', None) # keyword argument (if any)
    return triton.next_power_of_2(n)

@triton.heuristics(values={'BLOCK_SIZE': compute_block_size})
@triton.jit
def my_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

---

## The Heuristics Class

**File:** `autotuner.py`, lines 462-472

`Heuristics` is the internal class that implements heuristic-based configuration. It extends `KernelInterface`.

### Constructor

```python
class Heuristics(KernelInterface):
    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names
```

### Methods

#### `run(*args, **kwargs) -> Any`

Computes heuristic values and delegates to the wrapped function's `run` method.

```python
def run(self, *args, **kwargs):
    for v, heur in self.values.items():
        kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
    return self.fn.run(*args, **kwargs)
```

**Behavior:**
1. For each meta-parameter name and heuristic function in `self.values`:
   - Constructs the full argument dict by merging positional args (zipped with arg_names) and keyword args.
   - Calls the heuristic function with this dict.
   - Stores the computed value in `kwargs`.
2. Calls the wrapped function's `run` method with the augmented kwargs.

---

## Pruning Strategies

The `prune_configs_by` parameter of `@triton.autotune` controls how the search space is reduced before benchmarking. This is critical when the config space is large, as benchmarking hundreds of configs is expensive.

### The prune_configs_by Dictionary

```python
@triton.autotune(
    configs=my_configs,
    key=['M', 'N'],
    prune_configs_by={
        'perf_model': my_perf_model,
        'top_k': 10,
        'early_config_prune': my_early_prune,
    },
)
```

### Fields

#### `perf_model` (Optional[Callable])

A function that estimates the execution time of a given configuration. Used to rank configs and keep only the `top_k` most promising ones.

**Signature:**

```python
def perf_model(**kwargs) -> float:
    """
    Args:
        **kwargs: All kernel arguments merged with config kwargs
                  (nargs + kwargs + config.all_kwargs())
    Returns:
        float: Estimated execution time (lower is better)
    """
    ...
```

**Example -- Simple occupancy-based model:**

```python
def matmul_perf_model(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, **kwargs):
    # Estimate based on work per thread and memory traffic
    flops_per_thread = (BLOCK_M * BLOCK_N * BLOCK_K * 2) / (num_warps * 32)
    bytes_loaded = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4  # float32
    return bytes_loaded / flops_per_thread  # Arithmetic intensity ratio
```

**Example -- Shared memory constraint model:**

```python
def shared_mem_model(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, **kwargs):
    # Estimate shared memory usage and penalize configs that exceed limits
    smem_per_stage = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2  # bytes for fp16
    total_smem = smem_per_stage * num_stages
    if total_smem > 48 * 1024:  # 48KB default shared memory limit
        return float('inf')
    return total_smem  # Lower shared memory -> higher occupancy -> better
```

#### `top_k` (int or float)

Specifies how many configs to retain after performance model pruning.

- **If `int`**: Keep exactly that many configs.
- **If `float <= 1.0`**: Keep that fraction of the total configs (e.g., `0.5` keeps half).
- **Must be either an int or a float <= 1.0**, otherwise raises `TypeError`.

```python
# Keep the top 5 configs
prune_configs_by = {'perf_model': model, 'top_k': 5}

# Keep the top 20% of configs
prune_configs_by = {'perf_model': model, 'top_k': 0.2}
```

The top_k pruning is only applied when `len(pruned_configs) > top_k`. If there are fewer configs than top_k after early pruning, no further pruning occurs.

#### `early_config_prune` (Optional[Callable])

A function that performs domain-specific pruning based on kernel arguments before any benchmarking or performance model evaluation. This is the first pruning step.

**Signature:**

```python
def early_config_prune(
    configs: list[triton.Config],
    named_args: dict[str, Any],
    **kwargs,
) -> list[triton.Config]:
    """
    Args:
        configs: The full list of candidate configs.
        named_args: Dict mapping positional argument names to values.
        **kwargs: Keyword arguments passed to the kernel.
    Returns:
        list[triton.Config]: A pruned list of configs. Must return at least one config.
    """
    ...
```

**Example -- Prune based on problem size:**

```python
def prune_small_blocks(configs, named_args, **kwargs):
    M = named_args['M']
    N = named_args['N']
    # For small matrices, skip large block configs (wasted parallelism)
    if M * N < 256 * 256:
        return [c for c in configs if c.kwargs.get('BLOCK_M', 64) <= 64
                                      and c.kwargs.get('BLOCK_N', 64) <= 64]
    return configs
```

**Example -- Prune based on shared memory constraints:**

```python
def prune_by_shared_memory(configs, named_args, **kwargs):
    max_smem = 48 * 1024  # 48KB
    valid = []
    for config in configs:
        bm = config.kwargs.get('BLOCK_M', 64)
        bn = config.kwargs.get('BLOCK_N', 64)
        bk = config.kwargs.get('BLOCK_K', 32)
        ns = config.num_stages
        # Estimate shared memory
        smem = (bm * bk + bk * bn) * 4 * ns  # float32, double-buffered
        if smem <= max_smem:
            valid.append(config)
    return valid if valid else configs  # Fallback to all if none valid
```

### Pruning Pipeline Summary

The pruning pipeline in `Autotuner.prune_configs()` executes in this order:

```
1. Start with all configs
         |
2. early_config_prune(configs, nargs, **kwargs)  [if provided]
         |
3. Remaining configs
         |
4. perf_model evaluation for each config  [if perf_model provided]
         |
5. Sort by estimated time, keep top_k  [if len(configs) > top_k]
         |
6. Final pruned configs -> benchmark each
```

---

## Performance Model

The `perf_model` parameter provides a cost function to predict kernel performance without actually running it. This is used in the pruning phase to reduce the number of configs that need to be benchmarked.

### How It Works

1. For each remaining config (after early pruning), the autotuner calls `perf_model(**nargs, **kwargs, **config.all_kwargs())`.
2. The returned float value represents an estimated execution cost (lower is better).
3. Configs are sorted by this estimated cost, and only the `top_k` cheapest ones are actually benchmarked.

### Key Constraints

- The `perf_model` function must accept **all** kernel arguments plus all config kwargs as keyword arguments. Use `**kwargs` to absorb extra parameters.
- The return value must be a comparable number (float or int). Lower values indicate better expected performance.
- The `perf_model` is only invoked when `len(pruned_configs) > top_k`.

### Example: Roofline-Based Model for Matrix Multiplication

```python
def matmul_roofline_model(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, **kwargs):
    """Estimate performance using the roofline model."""
    # Compute operations and memory traffic per block
    flops = 2.0 * BLOCK_M * BLOCK_N * BLOCK_K  # multiply-add = 2 ops
    bytes_a = BLOCK_M * BLOCK_K * 2  # fp16 = 2 bytes
    bytes_b = BLOCK_K * BLOCK_N * 2
    bytes_c = BLOCK_M * BLOCK_N * 2
    total_bytes = bytes_a + bytes_b + bytes_c

    # Arithmetic intensity (ops/byte)
    ai = flops / total_bytes

    # Thread count
    threads = num_warps * 32

    # Rough estimate: higher AI and more threads = faster
    estimated_time = total_bytes / (threads * ai)
    return estimated_time
```

---

## Benchmarking

### Default Benchmarking Behavior

When no custom `do_bench` is provided, the autotuner calls `driver.active.get_benchmarker()` to obtain the backend's default benchmark function. This is a lazily-evaluated cached property:

```python
@cached_property
def do_bench(self):
    if self._do_bench is None:
        return driver.active.get_benchmarker()
    return self._do_bench
```

The default benchmarker is `triton.testing.do_bench`, which:

1. Runs a warmup phase to estimate kernel runtime.
2. Clears the L2 cache between repetitions for consistent measurements.
3. Records timing using GPU events.
4. Returns statistics based on quantiles.

### Quantiles Used

The autotuner always requests three quantiles from the benchmark function:

```python
self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
```

This returns `[median, p20, p80]` where:
- `median` (p50) -- The 50th percentile timing.
- `p20` -- The 20th percentile timing.
- `p80` -- The 80th percentile timing.

The **median** value is used for config selection:

```python
self.cache[key] = builtins.min(timings, key=timings.get)
```

Since `timings.get` returns the full list `[median, p20, p80]` and Python compares lists element-wise, the config with the lowest median is selected. In case of a tie, p20 is used as a tiebreaker, then p80.

### Custom do_bench Function

You can provide your own benchmark function via the `do_bench` parameter:

```python
import triton
import triton.testing

@triton.autotune(
    configs=[...],
    key=['n'],
    do_bench=triton.testing.do_bench_cudagraph,
)
@triton.jit
def my_kernel(...):
    ...
```

**Custom benchmark function signature:**

```python
def my_benchmarker(kernel_call: Callable, *, quantiles: list[float]) -> list[float]:
    """
    Args:
        kernel_call: A callable that executes the kernel once.
        quantiles: A list of quantile values to compute.
    Returns:
        list[float]: Timing values at the requested quantiles.
    """
    ...
```

### Error Handling During Benchmarking

If a config causes certain compilation errors, the autotuner catches them and returns infinite timing rather than crashing:

```python
try:
    return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
    if verbose:
        print(f"Autotuning failed with {e}")
    return [float("inf"), float("inf"), float("inf")]
```

The caught exceptions are:
- **`OutOfResources`** -- The config requires more resources (shared memory, registers, etc.) than the hardware provides.
- **`CompileTimeAssertionFailure`** -- A compile-time assertion in the kernel failed for this config.
- **`PTXASError`** -- The PTX assembler rejected the generated code.

All other exceptions propagate upward and halt autotuning.

---

## Cache Management

### In-Memory Cache

The autotuner maintains an in-memory dictionary `self.cache: Dict[Tuple, Config]` that maps cache key tuples to the best config for that key. This cache persists for the lifetime of the `Autotuner` object (typically the lifetime of the program).

```python
# Cache lookup in run()
if key not in self.cache:
    # ... benchmark and populate
    self.cache[key] = builtins.min(timings, key=timings.get)
config = self.cache[key]
```

### Cache Key Construction

The cache key is built from:

1. Values of arguments specified in the `key` parameter of `@triton.autotune`.
2. String representations of `.dtype` for any argument that has a dtype attribute.

```python
key = [_args[key] for key in self.keys if key in _args]
for _, arg in _args.items():
    if hasattr(arg, "dtype"):
        key.append(str(arg.dtype))
key = tuple(key)
```

### Disk Cache

When `cache_results=True` (or `TRITON_CACHE_AUTOTUNING=1`), autotuning results are persisted to disk.

**Disk cache key:** A SHA-256 hash combining:
- `triton_key()` -- Version and source hashes of all Triton code
- Backend hash -- The target GPU architecture hash
- JIT function cache key -- The kernel source code hash
- Environment variables -- Cache-invalidating env vars
- Tuning key -- The runtime key tuple
- Config strings -- String representations of all configs

**Cache file:** `{kernel_name}.autotune.json`

**Limitations:**
- Configs with `pre_hook` functions cannot be cached to disk (functions are not serializable). If any config has a `pre_hook`, the disk cache is bypassed entirely and benchmarks run every time.
- The disk cache is invalidated automatically when any component of the cache key changes (Triton version, kernel source, GPU architecture, env vars, etc.).

### Cache Flow

```
run() called
    |
    v
Key in self.cache? --Yes--> Use cached config
    |
    No
    |
    v
cache_results enabled? --Yes--> check_disk_cache()
    |                                |
    |                    Found on disk? --Yes--> Load, populate self.cache
    |                                |
    |                    No
    |                                |
    |                    Run benchmarks, save to disk
    |
    No
    |
    v
Run benchmarks, populate self.cache
    |
    v
Execute kernel with best config
```

### Autotune Listener

The `knobs.autotuning.listener` knob allows registering a callback that is notified after autotuning completes:

```python
from triton.knobs import knobs

def my_listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
    print(f"Kernel: {fn.__name__}")
    print(f"Best config: {best_config}")
    print(f"Benchmark duration: {duration}s")
    print(f"Cache hit: {cache_hit}")

knobs.autotuning.listener = my_listener
```

**Listener signature (AutotuneListener protocol):**

```python
class AutotuneListener(Protocol):
    def __call__(
        self,
        *,
        fn: JITFunction,
        key: tuple,
        best_config: Config,
        configs_timings: dict[Config, list[float]],
        duration: Optional[float],
        cache_hit: bool,
    ) -> None:
        ...
```

The listener is called regardless of whether the result was cached or freshly benchmarked. When `cache_hit=True`, `duration` is `None`.

---

## CUDA Graph Integration

### The use_cuda_graph Parameter (Deprecated)

The `use_cuda_graph=True` parameter was previously used to benchmark kernels using CUDA graphs, which reduces CPU overhead during benchmarking for more accurate measurements on small kernels. This parameter is now **deprecated**.

When `use_cuda_graph=True` was set, the autotuner would use `triton.testing.do_bench_cudagraph` instead of `triton.testing.do_bench`:

```python
if use_cuda_graph:
    from ..testing import do_bench_cudagraph
    self._do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
        kernel_call,
        rep=rep if rep is not None else 100,
        quantiles=quantiles,
    )
```

### Modern Replacement

Instead of `use_cuda_graph=True`, provide a custom `do_bench` function:

```python
import triton.testing

@triton.autotune(
    configs=[...],
    key=['n'],
    do_bench=triton.testing.do_bench_cudagraph,
)
@triton.jit
def my_kernel(...):
    ...
```

### How CUDA Graph Benchmarking Works

`do_bench_cudagraph` (in `triton/testing.py`) uses CUDA graphs to minimize host-side overhead:

1. **Warmup:** Calls the function once.
2. **Runtime estimation:** Measures 5 iterations to estimate per-call time.
3. **Graph construction:** Creates a CUDA graph with `n_repeat` unrolled function calls.
4. **Measurement:** Replays the graph 10 times and measures total time.
5. **Result:** Returns timing per call by dividing total time by `n_repeat`.

This approach eliminates the CPU overhead of launching kernels individually, which can dominate timing for very fast kernels.

---

## Pre and Post Hooks

Hooks allow custom code to run before and after kernel execution during both benchmarking and normal runs.

### Pre-Hook

The pre-hook is called before each kernel invocation. It has two modes of operation:

#### Default Pre-Hook (auto-generated)

When `reset_to_zero` or `restore_value` is specified without a custom `pre_hook`, a default pre-hook is generated:

```python
def _pre_hook(kwargs, reset_only=False):
    # Always zero specified tensors
    for name in self.reset_to_zero:
        kwargs[name].zero_()
    # Only clone if this is a benchmark run (not a reset-only call)
    if not reset_only:
        self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}
```

- `reset_only=True` is used after benchmarking completes, to zero out tensors one final time before the actual kernel run.
- `reset_only=False` is used during benchmarking, where both zeroing and cloning occur.

#### User-Defined Pre-Hook

When the user provides a `pre_hook` function, it overrides the default behavior entirely:

```python
@triton.autotune(
    configs=[...],
    key=['n'],
    pre_hook=lambda kwargs, reset_only=False: kwargs['output'].zero_(),
)
```

Setting `user_defined_pre_hook = True` disables the auto-generated reset/restore behavior.

### Post-Hook

The post-hook is called after each kernel invocation.

#### Default Post-Hook (auto-generated)

When `restore_value` is specified without a custom `post_hook`:

```python
def _post_hook(kwargs, exception):
    for name in self.restore_value:
        kwargs[name].copy_(self.restore_copies[name])
    self.restore_copies = {}
```

This restores tensors to their original values after benchmarking.

#### User-Defined Post-Hook

```python
@triton.autotune(
    configs=[...],
    key=['n'],
    post_hook=lambda kwargs, exception: print(f"Kernel completed. Exception: {exception}"),
)
```

### Hook Call Order During Benchmarking

In `_bench()`, the hooks are called in this order:

```
1. config.pre_hook(full_nargs)    -- Per-config hook (if defined)
2. self.pre_hook(full_nargs)      -- Global pre-hook
3. self.fn.run(...)               -- Kernel execution
4. self.post_hook(full_nargs, exception=None)  -- Global post-hook (on success)
```

If the kernel raises an exception:

```
1. config.pre_hook(full_nargs)
2. self.pre_hook(full_nargs)
3. self.fn.run(...)               -- Raises exception
4. self.post_hook(full_nargs, exception=e)  -- Called with exception
5. Exception is re-raised
```

### Per-Config Pre-Hooks

Individual `Config` objects can have their own `pre_hook`:

```python
def clear_l2_cache(args):
    # Clear L2 cache before this specific config
    import torch
    torch.cuda.empty_cache()

config_with_hook = triton.Config(
    {'BLOCK_SIZE': 128},
    num_warps=4,
    pre_hook=clear_l2_cache,
)
```

Per-config pre-hooks are called before the global pre-hook:

```python
def kernel_call():
    if config.pre_hook:
        config.pre_hook(full_nargs)   # Per-config hook first
    self.pre_hook(full_nargs)         # Then global hook
    ...
```

### Hook Signature Reference

```python
# Pre-hook signature
def pre_hook(kwargs: dict, reset_only: bool = False) -> None:
    """
    Args:
        kwargs: Dict of all kernel arguments (positional + keyword + config kwargs).
        reset_only: If True, only reset values (no save/restore needed).
                    This is True during the final reset before the real kernel run.
    """

# Post-hook signature
def post_hook(kwargs: dict, exception: Optional[Exception]) -> None:
    """
    Args:
        kwargs: Dict of all kernel arguments.
        exception: The exception raised by the kernel, or None if successful.
    """
```

---

## Early Config Pruning

The `early_config_prune` parameter (provided via `prune_configs_by['early_config_prune']`) allows domain-specific pruning of configurations based on runtime argument values. This is the first pruning stage and runs before any benchmarking or performance model evaluation.

### When to Use Early Config Pruning

- When certain configs are provably invalid for given input sizes (e.g., block size larger than the problem size).
- When hardware constraints make certain configs impossible (e.g., shared memory exceeds limits).
- When domain knowledge can eliminate clearly suboptimal configs.

### Signature

```python
def early_config_prune(
    configs: list[triton.Config],
    named_args: dict[str, Any],
    **kwargs: dict[str, Any],
) -> list[triton.Config]:
    """
    Args:
        configs: List of all candidate configurations.
        named_args: Dict of kernel positional arguments (name -> value).
        **kwargs: Keyword arguments passed to the kernel.
    Returns:
        A (possibly smaller) list of configs. Must return at least one config.
    """
```

### Important Rules

- The function **must** return at least one config. Returning an empty list raises `AutotunerError`:
  ```
  Autotuner error: No valid autotuner configs after pruning.
  `early_config_prune` should return at least one config.
  ```
- The function receives `named_args` (positional arguments) and `**kwargs` (keyword arguments) separately.
- Early pruning runs **before** performance model pruning, so the performance model only evaluates already-pruned configs.

### Example: Prune by Problem Dimensions

```python
def prune_matmul_configs(configs, named_args, **kwargs):
    M = named_args.get('M', 1)
    N = named_args.get('N', 1)
    pruned = []
    for c in configs:
        bm = c.kwargs.get('BLOCK_M', 64)
        bn = c.kwargs.get('BLOCK_N', 64)
        # Skip configs where blocks are larger than the problem
        if bm > M or bn > N:
            continue
        pruned.append(c)
    return pruned if pruned else configs
```

### Example: Prune by Hardware Constraints

```python
def prune_by_register_pressure(configs, named_args, **kwargs):
    MAX_REGS_PER_THREAD = 255
    pruned = []
    for c in configs:
        if c.maxnreg is not None and c.maxnreg > MAX_REGS_PER_THREAD:
            continue
        pruned.append(c)
    return pruned if pruned else configs
```

---

## Warmup and Repetition

### The warmup and rep Parameters (Deprecated)

These parameters control the benchmarking behavior and are now **deprecated**. They generate a `DeprecationWarning` when used:

```python
import warnings
warnings.warn(
    "warmup, rep, and use_cuda_graph parameters are deprecated. See "
    "https://github.com/triton-lang/triton/pull/4496 for details.",
    DeprecationWarning,
    stacklevel=1,
)
```

### What They Did

When `warmup` and/or `rep` are specified (without `use_cuda_graph`), the autotuner creates a custom benchmark function wrapping `triton.testing.do_bench`:

```python
import triton.testing
self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
    kernel_call,
    warmup=warmup if warmup is not None else 25,
    rep=rep if rep is not None else 100,
    quantiles=quantiles,
)
```

- **`warmup`** (int, in milliseconds): Total warmup time. The number of warmup iterations is computed as `warmup_time / estimated_per_call_time`. Default: 25ms.
- **`rep`** (int, in milliseconds): Total repetition time. The number of measurement iterations is computed as `rep_time / estimated_per_call_time`. Default: 100ms.

When `use_cuda_graph=True` is combined with `warmup`/`rep`, the `do_bench_cudagraph` function is used instead:

```python
from ..testing import do_bench_cudagraph
self._do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
    kernel_call,
    rep=rep if rep is not None else 100,
    quantiles=quantiles,
)
```

### Modern Replacement

Use the `do_bench` parameter directly:

```python
# Equivalent to warmup=50, rep=200
import triton.testing

@triton.autotune(
    configs=[...],
    key=['n'],
    do_bench=lambda fn, quantiles: triton.testing.do_bench(
        fn, warmup=50, rep=200, quantiles=quantiles,
    ),
)
@triton.jit
def my_kernel(...):
    ...
```

### Warmup Method (AOT Compilation)

The `Autotuner.warmup()` method (distinct from the `warmup` parameter) is used for ahead-of-time compilation. It pre-compiles all pruned configs without executing them:

```python
def warmup(self, *args, **kwargs):
    self.nargs = dict(zip(self.arg_names, args))
    ret = []
    for autotune_config in self.prune_configs(kwargs):
        ret.append(self.fn.warmup(
            *args,
            **kwargs,
            **autotune_config.all_kwargs(),
        ))
    self.nargs = None
    return ret
```

This returns a list of compiled kernel handles (one per pruned config), allowing the kernels to be compiled before they are first needed.

---

## Environment Variables and Knobs

### TRITON_PRINT_AUTOTUNING

Set to `"1"` to print autotuning progress and results:

```bash
TRITON_PRINT_AUTOTUNING=1 python my_script.py
```

This produces output like:
```
Autotuning kernel vector_add_kernel with config BLOCK_SIZE: 128, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
Autotuning kernel vector_add_kernel with config BLOCK_SIZE: 256, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
...
Triton autotuning for function vector_add_kernel,
with key as (1024, 'torch.float32'),
finished after 0.42s,
best config selected: BLOCK_SIZE: 256, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
```

### TRITON_CACHE_AUTOTUNING

Set to `"1"` to enable disk caching of autotune results:

```bash
TRITON_CACHE_AUTOTUNING=1 python my_script.py
```

This is equivalent to setting `cache_results=True` in the decorator.

### Autotuning Knobs

The autotuning knobs are defined in `triton/knobs.py`:

```python
class autotuning_knobs(base_knobs):
    cache: env_bool = env_bool("TRITON_CACHE_AUTOTUNING")
    print: env_bool = env_bool("TRITON_PRINT_AUTOTUNING")
    listener: Union[AutotuneListener, None] = None
```

Access them via `knobs.autotuning`:

```python
from triton.knobs import knobs

# Check if printing is enabled
if knobs.autotuning.print:
    print("Autotuning print is enabled")

# Register a listener
def my_listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
    ...

knobs.autotuning.listener = my_listener
```

---

## Complete Code Examples

### Example 1: Basic Vector Addition Autotuning

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements)
    return output


# Usage
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
result = add(x, y)
```

### Example 2: Matrix Multiplication with Pruning

```python
import torch
import triton
import triton.language as tl


# Define configs for matmul
matmul_configs = [
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=4),
    triton.Config({'BM': 128, 'BN': 64,  'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4, num_stages=2),
    triton.Config({'BM': 64,  'BN': 64,  'BK': 64}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64,  'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64,  'BK': 64}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=4),
    triton.Config({'BM': 256, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=3),
]


def matmul_early_prune(configs, named_args, **kwargs):
    """Prune configs with block sizes larger than the problem."""
    M = named_args.get('M', 1)
    N = named_args.get('N', 1)
    pruned = []
    for c in configs:
        if c.kwargs['BM'] <= M and c.kwargs['BN'] <= N:
            pruned.append(c)
    return pruned if pruned else configs


def matmul_perf_model(M, N, K, BM, BN, BK, num_warps, num_stages, **kwargs):
    """Simple performance model: estimate time from memory traffic."""
    # Total memory traffic per tile (bytes)
    bytes_a = BM * BK * 2  # fp16
    bytes_b = BK * BN * 2
    bytes_c = BM * BN * 4  # fp32 output
    total_traffic = bytes_a + bytes_b + bytes_c

    # Computational work per tile
    flops = 2 * BM * BN * BK

    # Threads available
    threads = num_warps * 32

    # Estimate: time proportional to traffic / threads
    return total_traffic / threads


@triton.autotune(
    configs=matmul_configs,
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': matmul_early_prune,
        'perf_model': matmul_perf_model,
        'top_k': 5,  # Only benchmark top 5 candidates
    },
    reset_to_zero=['C'],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Kernel implementation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    # Pointers
    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(A_ptr, mask=(rk[None, :] < K - k * BK), other=0.0)
        b = tl.load(B_ptr, mask=(rk[:, None] < K - k * BK), other=0.0)
        acc += tl.dot(a, b)
        A_ptr += BK * stride_ak
        B_ptr += BK * stride_bk

    # Store result
    C_ptr = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr, acc, mask=mask)


def matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta['BM']),
        triton.cdiv(N, meta['BN']),
    )
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
```

### Example 3: Heuristics-Only Configuration

```python
import triton
import triton.language as tl


@triton.heuristics(
    values={
        'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_elements']),
    }
)
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))

    # Softmax
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp

    # Store
    tl.store(output_ptr + offsets, result, mask=mask)
```

### Example 4: Autotuning with Custom Hooks and Restore

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n'],
    restore_value=['input'],    # Save and restore input tensor
    reset_to_zero=['output'],   # Zero output before each run
)
@triton.jit
def in_place_transform_kernel(
    input, output,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input + offsets, mask=mask)
    result = x * 2.0 + 1.0
    tl.store(output + offsets, result, mask=mask)
```

### Example 5: Autotuning with Disk Caching and Listener

```python
import torch
import triton
import triton.language as tl
from triton.knobs import knobs


# Register a listener to log autotuning results
def autotune_logger(*, fn, key, best_config, configs_timings, duration, cache_hit):
    print(f"[AUTOTUNE] {fn.__name__}: key={key}")
    print(f"  Best: {best_config}")
    print(f"  Cache hit: {cache_hit}")
    if duration is not None:
        print(f"  Benchmark time: {duration:.3f}s")

knobs.autotuning.listener = autotune_logger


@triton.autotune(
    configs=[
        triton.Config({'BS': 64}, num_warps=4),
        triton.Config({'BS': 128}, num_warps=4),
        triton.Config({'BS': 256}, num_warps=8),
        triton.Config({'BS': 512}, num_warps=8),
    ],
    key=['size'],
    cache_results=True,  # Persist to disk
)
@triton.jit
def reduction_kernel(
    data_ptr, output_ptr,
    size,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BS + tl.arange(0, BS)
    mask = offsets < size
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr, partial_sum)


def reduce_sum(data: torch.Tensor):
    output = torch.zeros(1, device=data.device, dtype=data.dtype)
    size = data.numel()
    grid = lambda meta: (triton.cdiv(size, meta['BS']),)
    reduction_kernel[grid](data, output, size)
    return output.item()
```

### Example 6: Autotuning with Custom Benchmark Function

```python
import torch
import triton
import triton.language as tl
import triton.testing


# Use CUDA graph benchmarking for very small kernels
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n'],
    do_bench=triton.testing.do_bench_cudagraph,
)
@triton.jit
def elementwise_kernel(
    x_ptr, y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sigmoid(x)
    tl.store(y_ptr + offsets, y, mask=mask)


# Alternatively, a fully custom benchmark function
def my_custom_bench(kernel_call, quantiles):
    """A custom benchmark function with more warmup."""
    import time
    # Warmup
    for _ in range(100):
        kernel_call()
    torch.cuda.synchronize()
    # Measure
    times = []
    for _ in range(50):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    median = times[n // 2]
    p20 = times[int(n * 0.2)]
    p80 = times[int(n * 0.8)]
    return [median, p20, p80]
```

### Example 7: Combining Autotune and Heuristics

```python
import triton
import triton.language as tl


# Heuristics compute values that autotune doesn't need to search over
@triton.heuristics(
    values={
        'num_warps': lambda args: 4 if args['n'] < 4096 else 8,
    }
)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n'],
)
@triton.jit
def combined_kernel(
    x_ptr, y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

Note: When combining decorators, the order matters and conflicts between autotune config kwargs and heuristic values will be detected. The autotuner checks for conflicts in `_bench()`:

```python
conflicts = meta.keys() & config.kwargs.keys()
if conflicts:
    raise ValueError(
        f"Conflicting meta-parameters: {', '.join(conflicts)}. "
        "Make sure that you don't re-define auto-tuned symbols."
    )
```

### Example 8: Fractional top_k Pruning

```python
import triton


# Generate many configs programmatically
configs = []
for bm in [32, 64, 128, 256]:
    for bn in [32, 64, 128, 256]:
        for bk in [16, 32, 64]:
            for nw in [2, 4, 8]:
                configs.append(triton.Config(
                    {'BM': bm, 'BN': bn, 'BK': bk},
                    num_warps=nw,
                    num_stages=2,
                ))

# 192 total configs -- too many to benchmark exhaustively

def my_perf_model(M, N, K, BM, BN, BK, num_warps, **kwargs):
    """Crude model: prefer larger tiles with more parallelism."""
    tile_utilization = (BM * BN) / max(M * N, 1)
    thread_count = num_warps * 32
    return -tile_utilization * thread_count  # Negate because lower = better


@triton.autotune(
    configs=configs,  # 192 configs
    key=['M', 'N', 'K'],
    prune_configs_by={
        'perf_model': my_perf_model,
        'top_k': 0.1,  # Keep only top 10% (about 19 configs)
    },
)
@triton.jit
def my_matmul_kernel(...):
    ...
```

### Example 9: Per-Config Pre-Hooks

```python
import triton
import torch


def clear_scratch_for_large_blocks(args):
    """Zero out scratch buffer for large block configs."""
    args['scratch'].zero_()


configs = [
    triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(
        {'BLOCK_SIZE': 512}, num_warps=8,
        pre_hook=clear_scratch_for_large_blocks,  # Only this config gets the hook
    ),
]


@triton.autotune(configs=configs, key=['n'])
@triton.jit
def kernel_with_scratch(
    input_ptr, output_ptr, scratch_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    ...
```

Note: When using disk caching (`cache_results=True`), configs with `pre_hook` functions cannot be cached to disk because functions are not serializable. The disk cache check skips caching when any config has a `pre_hook`:

```python
if not tuning_key or any(cfg.pre_hook for cfg in configs):
    bench_fn()
    return False
```

---

## Summary of Key Implementation Details

1. **Cache key** includes argument values from `key` parameter plus dtype strings of tensor arguments.
2. **Config selection** uses the median timing (first element of the three-quantile return).
3. **Error tolerance** -- configs that fail with `OutOfResources`, `CompileTimeAssertionFailure`, or `PTXASError` get infinite timing and are automatically skipped.
4. **Pruning** is a two-stage pipeline: early_config_prune first, then perf_model with top_k.
5. **Disk cache** is keyed on Triton version, backend, kernel source, environment variables, and runtime key.
6. **Hook execution** follows a strict order: config.pre_hook -> global pre_hook -> kernel -> global post_hook.
7. **reset_to_zero and restore_value** are implemented as auto-generated hooks that can be overridden by user-provided hooks.
8. **The deprecated parameters** (warmup, rep, use_cuda_graph) generate warnings and create internal benchmark function wrappers. Use `do_bench` instead.
9. **The listener mechanism** provides a way to observe autotuning decisions programmatically for logging and analysis.
