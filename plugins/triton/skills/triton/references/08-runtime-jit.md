# Triton Runtime JIT System -- Comprehensive Reference

This document provides an exhaustive reference for the Triton runtime
just-in-time (JIT) compilation system. It covers every public class,
function, decorator, and internal mechanism used to transform Python
functions into GPU kernels at runtime.

**Source files covered:**

- `python/triton/runtime/jit.py` -- Core JIT decorator, `JITFunction`,
  `KernelParam`, specialization, caching, cache-key computation,
  dependencies finder, `MockTensor`, `TensorWrapper`, `ConstexprFunction`.
- `python/triton/runtime/_allocation.py` -- Memory allocation protocols
  (`Buffer`, `Allocator`) and the global allocator context.
- `python/triton/runtime/_async_compile.py` -- Asynchronous compilation
  (`AsyncCompileMode`, `FutureKernel`).

---

## Table of Contents

1. [@triton.jit Decorator](#1-tritonjit-decorator)
2. [JITFunction Class](#2-jitfunction-class)
3. [KernelParam Class](#3-kernelparam-class)
4. [KernelInterface Class](#4-kernelinterface-class)
5. [JITCallable Class](#5-jitcallable-class)
6. [ConstexprFunction and constexpr_function](#6-constexprfunction-and-constexpr_function)
7. [Specialization System](#7-specialization-system)
8. [Type Mangling](#8-type-mangling)
9. [Dependency Finding (DependenciesFinder)](#9-dependency-finding-dependenciesfinder)
10. [Memory Allocation](#10-memory-allocation)
11. [Async Compilation](#11-async-compilation)
12. [TensorWrapper and MockTensor](#12-tensorwrapper-and-mocktensor)
13. [Cache Key Computation](#13-cache-key-computation)
14. [Helper Utilities](#14-helper-utilities)
15. [Global Registries and Hooks](#15-global-registries-and-hooks)

---

## 1. @triton.jit Decorator

The `@triton.jit` decorator is the primary entry point for defining GPU
kernels in Triton. It transforms a regular Python function into a
`JITFunction` object that can be launched on a GPU using the grid
syntax `kernel[grid](*args, **kwargs)`.

### 1.1 Full Signature

```python
@overload
def jit(fn: T) -> JITFunction[T]: ...

@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], JITFunction[T]]: ...

def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> KernelInterface[T]: ...
```

### 1.2 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Optional[T]` | `None` | The function to JIT-compile. When provided directly (i.e., `@triton.jit` without arguments), this is the decorated function. |
| `version` | Any | `None` | Version identifier for the kernel. Not actively used in compilation but stored on the `JITFunction`. |
| `repr` | `Optional[Callable]` | `None` | Custom representation function for the kernel. Called by `JITFunction.repr()`. |
| `launch_metadata` | `Optional[Callable]` | `None` | Function to generate launch metadata for the kernel. |
| `do_not_specialize` | `Optional[Iterable[int \| str]]` | `None` | Indices or names of parameters that should **not** be specialized. Prevents the JIT compiler from creating separate kernels for different values of these arguments. |
| `do_not_specialize_on_alignment` | `Optional[Iterable[int \| str]]` | `None` | Indices or names of pointer parameters that should **not** be specialized on alignment. |
| `debug` | `Optional[bool]` | `None` | Enable debug output for this specific kernel. Overrides the global `TRITON_DEBUG` knob per-kernel. |
| `noinline` | `Optional[bool]` | `None` | Prevent the compiler from inlining this function when called from other `@triton.jit` functions. |

### 1.3 How It Works

The decorator supports two usage patterns:

**Pattern 1: Direct decoration (no arguments)**

```python
@triton.jit
def my_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # kernel body
    pass
```

When `fn` is not `None`, the decorator immediately wraps the function.

**Pattern 2: Parameterized decoration**

```python
@triton.jit(do_not_specialize=[2], debug=True)
def my_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # kernel body
    pass
```

When `fn` is `None`, the decorator returns a second decorator function
that will be called with the actual function.

### 1.4 Internal Dispatch Logic

```python
def decorator(fn: T) -> JITFunction[T]:
    assert callable(fn)
    if knobs.runtime.interpret:
        from .interpreter import InterpretedFunction
        return InterpretedFunction(fn, version=version, ...)
    else:
        return JITFunction(fn, version=version, ...)

if fn is not None:
    return decorator(fn)
else:
    return decorator
```

When the `TRITON_INTERPRET` environment variable is set, the decorator
returns an `InterpretedFunction` instead of a `JITFunction`. This runs
the kernel in a Python interpreter for debugging purposes.

### 1.5 Complete Usage Examples

**Basic kernel:**

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,        # pointer to first input
    y_ptr,        # pointer to second input
    output_ptr,   # pointer to output
    n_elements,   # number of elements
    BLOCK_SIZE: tl.constexpr,  # block size (compile-time constant)
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

**With do_not_specialize:**

```python
@triton.jit(do_not_specialize=[3])
def kernel_with_stride(
    data_ptr,
    n_elements,
    stride,  # will NOT create separate kernels per stride value
    BLOCK_SIZE: tl.constexpr,
):
    pass
```

**With debug enabled:**

```python
@triton.jit(debug=True)
def debug_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel will produce verbose debug output during compilation
    pass
```

**With noinline:**

```python
@triton.jit(noinline=True)
def helper_function(x):
    # Will not be inlined into callers
    return x + 1

@triton.jit
def main_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    val = tl.load(x_ptr)
    result = helper_function(val)  # call remains as a function call
    tl.store(x_ptr, result)
```

**With repr and launch_metadata:**

```python
@triton.jit(
    repr=lambda _: "CustomKernelName",
    launch_metadata=lambda grid, stream, *args: {"grid": grid}
)
def custom_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pass
```

### 1.6 Important Constraints

- The decorated function **must** be defined in a Python source file
  (not in an interactive REPL or dynamically generated code), because
  `inspect.getsourcelines` is used to extract the source code.
- Calling a `@triton.jit` function directly (without grid indexing) raises
  `RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")`.
- JIT functions can only access: Python primitives, Triton builtins,
  their arguments, and other `@triton.jit` functions.

---

## 2. JITFunction Class

`JITFunction` is the core class that wraps a Python function annotated
with `@triton.jit`. It manages kernel compilation, caching, and
execution.

### 2.1 Class Hierarchy

```
JITCallable
  |
  +-- JITFunction
        |
        +-- KernelInterface[T] (mixin)
```

`JITFunction` inherits from both `JITCallable` (which provides source
code management, hashing, and dependency tracking) and
`KernelInterface[T]` (which provides the grid-indexing launch syntax).

### 2.2 Constructor

```python
class JITFunction(JITCallable, KernelInterface[T]):

    def __init__(
        self,
        fn,
        version=None,
        do_not_specialize=None,
        do_not_specialize_on_alignment=None,
        debug=None,
        noinline=None,
        repr=None,
        launch_metadata=None,
    ):
```

**Constructor behavior:**

1. Calls `super().__init__(fn)` (the `JITCallable` constructor) which:
   - Extracts source code via `inspect.getsourcelines(fn)`
   - Dedents the source and strips leading decorators
   - Computes the fully qualified function name
   - Records file name, line number, and column number
   - Initializes the hash lock and used_global_vals dict

2. Stores metadata:
   - `self.module` -- `fn.__module__`
   - `self.version` -- version parameter
   - `self.do_not_specialize` -- list of parameter indices/names
   - `self.do_not_specialize_on_alignment` -- list of parameter indices/names

3. Registers in the global registry:
   ```python
   _triton_jit_function_registry[f"{self.module}:{self.fn.__qualname__}"] = self
   ```

4. Creates `KernelParam` objects for each parameter:
   ```python
   self.params = []
   for i, param in enumerate(self.signature.parameters.values()):
       dns = i in do_not_specialize or param.name in do_not_specialize
       dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
       self.params.append(KernelParam(i, param, dns, dns_oa))
   ```

5. Initializes the device-specific cache:
   ```python
   self.device_caches = defaultdict(self.create_binder)
   ```

6. Sets convenience attributes:
   ```python
   self.arg_names = [p.name for p in self.params]
   self.constexprs = [p.num for p in self.params if p.is_constexpr]
   ```

### 2.3 Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `fn` | `Callable` | The original Python function |
| `signature` | `inspect.Signature` | Function signature from `inspect.signature(fn)` |
| `src` | `str` | Dedented source code (read-only property; use `_unsafe_update_src` to modify) |
| `params` | `list[KernelParam]` | Parameter metadata list |
| `arg_names` | `list[str]` | Parameter names |
| `constexprs` | `list[int]` | Indices of constexpr parameters |
| `module` | `str` | Module name where the function is defined |
| `hash` | `Optional[str]` | Cached hash value |
| `cache_key` | `str` | Computed cache key (lazy) |
| `used_global_vals` | `Dict` | Global variables used by this function |
| `debug` | `Optional[bool]` | Per-kernel debug flag |
| `noinline` | `Optional[bool]` | Prevent inlining flag |
| `kernel` | `None` | Always None; placeholder |
| `device_caches` | `defaultdict` | Per-device compilation cache |

### 2.4 Methods

#### `run(self, *args, grid, warmup, **kwargs)`

The main kernel execution path. This method:

1. **Prepares options**: Merges debug settings from kernel-level and
   global knobs:
   ```python
   kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
   kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
   ```

2. **Acquires device and stream**:
   ```python
   device = driver.active.get_current_device()
   stream = driver.active.get_current_stream(device)
   ```

3. **Executes pre-run hooks**:
   ```python
   for hook in self.pre_run_hooks:
       hook(*args, **kwargs)
   ```

4. **Retrieves per-device cache, target, backend, and binder**:
   ```python
   kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
   ```

5. **Binds arguments and computes specialization**:
   ```python
   bound_args, specialization, options = binder(*args, **kwargs)
   ```
   The `binder` is a dynamically generated function (see
   `create_function_from_signature`) that performs argument binding
   and specialization in a single call, optimized for repeated
   invocations.

6. **Optionally adds custom pipeline stage**:
   ```python
   if knobs.runtime.add_stages_inspection_hook is not None:
       inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
       specialization.append(f'("custom_pipeline", {inspect_stages_hash})')
   ```

7. **Computes cache key**:
   ```python
   key = compute_cache_key(kernel_key_cache, specialization, options)
   ```

8. **Cache lookup**:
   ```python
   kernel = kernel_cache.get(key, None)
   ```

9. **If not cached, compiles**:
   ```python
   if kernel is None:
       options, signature, constexprs, attrs = self._pack_args(
           backend, kwargs, bound_args, specialization, options
       )
       kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
       if kernel is None:
           return None
   ```

10. **Validates global variables** have not changed:
    ```python
    for (name, _), (val, globals_dict) in self.used_global_vals.items():
        if (newVal := globals_dict.get(name, not_present)) != val:
            raise RuntimeError(
                f"Global variable {name} has changed since we compiled this kernel")
    ```

11. **Launches the kernel** (if not a warmup):
    ```python
    if not warmup:
        # canonicalize grid
        if callable(grid):
            grid = grid(bound_args)
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1
        # launch
        launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
        kernel.run(grid_0, grid_1, grid_2, stream, kernel.function,
                   kernel.packed_metadata, launch_metadata,
                   knobs.runtime.launch_enter_hook,
                   knobs.runtime.launch_exit_hook,
                   *bound_args.values())
    ```

#### `warmup(self, *args, grid, **kwargs)`

Compiles the kernel without launching it. Arguments are wrapped in
`MockTensor` if they are `torch.dtype` objects:

```python
def warmup(self, *args, grid, **kwargs):
    return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)
```

Usage:
```python
# Warmup with a mock float32 tensor
kernel.warmup(MockTensor(torch.float32), 1024, BLOCK_SIZE=1024, grid=lambda args: (1,))
```

#### `create_binder(self)`

Creates a per-device compilation environment. Called lazily by
`defaultdict` the first time a device is used:

```python
def create_binder(self):
    from ..compiler import CompiledKernel, compile, ASTSource, make_backend
    target = driver.active.get_current_target()
    backend = make_backend(target)
    self.CompiledKernel = CompiledKernel
    self.compile = compile
    self.ASTSource = ASTSource
    binder = create_function_from_signature(self.signature, self.params, backend)
    return {}, {}, target, backend, binder
```

Returns a tuple:
- `{}` -- Empty kernel cache (maps cache key to compiled kernel)
- `{}` -- Empty kernel key cache (maps (specialization, options) to cache key)
- `target` -- `GPUTarget` for the active device
- `backend` -- Backend implementation (e.g., `CUDABackend`)
- `binder` -- Dynamically generated function for argument binding

#### `_pack_args(self, backend, kwargs, bound_args, specialization, options)`

Extracts structured information from the specialization results:

```python
def _pack_args(self, backend, kwargs, bound_args, specialization, options):
    options = backend.parse_options(kwargs)
    sigkeys = [x.name for x in self.params]
    sigvals = [x[0] for x in specialization]
    signature = {k: v for (k, v) in zip(sigkeys, sigvals)}

    # constexprs
    constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
    constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}

    # attributes
    attrvals = ['' if x[0] == 'constexpr' else x[1] for x in specialization]
    attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
    attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

    return options, signature, constexprs, attrs
```

**Outputs:**
- `options` -- Parsed compilation options
- `signature` -- Dict mapping parameter names to their types
- `constexprs` -- Dict mapping paths to constexpr values
- `attrs` -- Dict mapping paths to parsed attributes (e.g., divisibility)

#### `_do_compile(self, key, signature, device, constexprs, options, attrs, warmup)`

Performs the actual compilation or submits it for async compilation:

```python
def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
    kernel_cache, _, target, backend, _ = self.device_caches[device]

    # Call pre-compile hook
    if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, target,
                       device, constexprs, options, [attrs], warmup):
        return None

    src = self.ASTSource(self, signature, constexprs, attrs)

    async_mode = _async_compile.active_mode.get()
    if async_mode is not None:
        # Async compilation path
        env_vars = get_cache_invalidating_env_vars()
        cache_key = get_cache_key(src, backend, options, env_vars)

        def async_compile():
            return self.compile(src, target=target, options=options.__dict__,
                              _env_vars=env_vars)

        def finalize_compile(kernel):
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, ...)

        kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
        kernel_cache[key] = kernel
    else:
        # Synchronous compilation path
        kernel = self.compile(src, target=target, options=options.__dict__)
        kernel_cache[key] = kernel
        self._call_hook(knobs.runtime.jit_post_compile_hook, ...)

    return kernel
```

#### `preload(self, specialization_data)`

Deserializes previously serialized specialization data to compile a
kernel without needing the original arguments:

```python
def preload(self, specialization_data):
    import json
    import triton.language as tl
    device = driver.active.get_current_device()
    deserialized_obj = json.loads(specialization_data)
    # ... validates name and target match
    # ... deserializes constants, attributes, signature, options
    return self._do_compile(key, signature, device, constexprs, options, attrs, warmup=True)
```

This enables "ahead-of-time" kernel compilation where specialization
data captured from a previous run can be replayed.

#### `_call_hook(self, hook, key, signature, target, device, constants, options, configs, is_warmup)`

Calls a JIT hook (if provided) with detailed compilation information:

```python
def _call_hook(self, hook, key, signature, target, device, constants,
               options, configs, is_warmup) -> bool | None:
    if not hook:
        return None

    name = self.fn.__qualname__
    module = self.fn.__module__
    arg_reprs = ", ".join([f"{param.name}: {ty}"
                          for param, ty in zip(self.params, key[1])])
    repr = (f"{name}[num_warps={options.num_warps}, "
            f"num_ctas={options.num_ctas}, ...]({arg_reprs})")

    specialization_data = serialize_specialization_data(...)

    return hook(key=key, repr=repr, fn=JitFunctionInfo(module, name, self),
                compile={...}, is_manual_warmup=is_warmup, already_compiled=False)
```

If the hook returns a truthy value, compilation is skipped.

#### `add_pre_run_hook(self, hook)`

Registers a hook to be executed before `run()` with the same arguments:

```python
def add_pre_run_hook(self, hook):
    assert callable(hook)
    self.pre_run_hooks.append(hook)
```

#### `__call__(self, *args, **kwargs)`

Raises `RuntimeError` because `@triton.jit` functions must be called
with the grid syntax `kernel[grid](...)`:

```python
def __call__(self, *args, **kwargs):
    raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")
```

#### `__repr__(self)`

```python
def __repr__(self):
    return f"JITFunction({self.module}:{self.fn.__qualname__})"
```

#### `__get__(self, instance, owner)`

Implements the descriptor protocol, enabling `@triton.jit` to work as
a method decorator on classes. Uses `@overload` for type checking:

```python
@overload
def __get__(self, instance: None, owner=None) -> "JITFunction[T]": ...

@overload
def __get__(self, instance: Any, owner=None) -> Callable[P, R]: ...

def __get__(self, instance, owner=None): ...
```

When accessed on an instance, the first argument (self) is
automatically bound.

#### `repr(self, _)`

Returns the custom repr string if provided, otherwise the fully
qualified function name:

```python
def repr(self, _):
    return self._fn_name if self._repr is None else self._repr(_)
```

### 2.5 Device Cache Structure

The `device_caches` attribute is a `defaultdict` keyed by device ID.
Each entry is a tuple of:

```python
(kernel_cache: dict,       # cache_key -> CompiledKernel
 kernel_key_cache: dict,   # (specialization, options) -> cache_key_str
 target: GPUTarget,        # device target info
 backend: BaseBackend,     # backend implementation
 binder: Callable)         # dynamic argument binding function
```

The `defaultdict` factory function is `create_binder`, which lazily
initializes all these components for each new device.

### 2.6 Kernel Launch Flow

The complete flow from `kernel[grid](*args)` to GPU execution:

```
kernel[grid]               # KernelInterface.__getitem__ returns lambda
  -> lambda(*args)          # calls JITFunction.run()
     -> get device/stream
     -> run pre_run_hooks
     -> binder(*args)       # compute specialization
     -> compute_cache_key
     -> cache lookup
     -> if miss: _pack_args + _do_compile
     -> validate global vars
     -> if not warmup:
          resolve grid
          compute launch_metadata
          kernel.run(...)   # actual GPU launch
```

---

## 3. KernelParam Class

`KernelParam` stores metadata about a single parameter of a
`@triton.jit` function. It is used during specialization and cache key
computation.

### 3.1 Definition

```python
class KernelParam:
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

    def __init__(self, num: int, param: inspect.Parameter,
                 do_not_specialize: bool,
                 do_not_specialize_on_alignment: bool):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
```

### 3.2 Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num` | `int` | Zero-based parameter index |
| `param` | `inspect.Parameter` | The `inspect.Parameter` object for this parameter |
| `do_not_specialize` | `bool` | Whether to skip specialization for this parameter |
| `do_not_specialize_on_alignment` | `bool` | Whether to skip alignment specialization for this parameter |

### 3.3 Properties

#### `name` (cached_property)

The parameter name, lazily computed:

```python
@cached_property
def name(self):
    return self._param.name
```

#### `annotation` (cached_property)

Normalized type annotation string. Returns empty string if no
annotation is present:

```python
@cached_property
def annotation(self) -> str:
    if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
        return ""
    return _normalize_ty(self._param.annotation)
```

Examples of normalized annotations:
- `tl.constexpr` -> `"constexpr"`
- `torch.float32` -> `"fp32"`
- `tl.pointer_type(tl.float32)` -> `"*fp32"`

#### `annotation_type` (cached_property)

The base type portion of the annotation (without const/constexpr markers).
Used to determine if the annotation provides a fixed type that overrides
runtime specialization:

```python
@cached_property
def annotation_type(self) -> str:
    a = self.annotation
    if a.startswith("*k"):
        a = a[2:]
    elif a.startswith("*"):
        a = a[1:]
    if a in set(type_canonicalisation_dict.values()):
        return self.annotation
    return ""
```

Returns the full annotation if it resolves to a known Triton type,
otherwise returns empty string (indicating the type should be inferred
from the argument at runtime).

Examples:
- Parameter annotated `x: "float32"` -> annotation_type = `"fp32"`
- Parameter annotated `x: "tl.constexpr"` -> annotation_type = `""`
- Parameter annotated `ptr: "*float32"` -> annotation_type = `"*fp32"`
- Parameter annotated `cptr: "const *float32"` -> annotation_type = `"*kfp32"`

#### `is_constexpr` (cached_property)

Whether the parameter is a `constexpr`:

```python
@cached_property
def is_constexpr(self):
    return "constexpr" in self.annotation
```

A parameter is constexpr if its annotation contains "constexpr"
(e.g., `BLOCK_SIZE: tl.constexpr`).

#### `is_const` (cached_property)

Whether the parameter is a const pointer (but not constexpr):

```python
@cached_property
def is_const(self):
    if self.is_constexpr:
        return False
    return "const" in self.annotation or self.annotation.startswith("*k")
```

A const pointer is annotated with `const` or starts with `*k`.

#### `default` (property)

The default value of the parameter, or `inspect.Parameter.empty` if
no default exists:

```python
@property
def default(self):
    return self._param.default
```

#### `has_default` (property)

Whether the parameter has a default value:

```python
@property
def has_default(self):
    return self._param.default != inspect.Parameter.empty
```

### 3.4 Usage in Specialization

During `create_function_from_signature`, each `KernelParam` controls
how the corresponding argument is specialized:

```python
for name, kp in zip(sig.parameters.keys(), kparams):
    if kp.is_constexpr:
        # Constexpr parameters are always specialized
        specialization.append(f'("constexpr", {name})')
    else:
        is_const = 'True' if kp.is_const else 'False'
        specialize = 'False' if kp.do_not_specialize else 'True'
        align = 'False' if kp.do_not_specialize_on_alignment else 'True'
        ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"

        if kp.annotation_type:
            if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                # Do not specialize non-constexpr floats and bools
                specialize = False
            if specialize:
                specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
            else:
                specialization.append(f'("{kp.annotation_type}", None)')
        else:
            specialization.append(f"{ret}")
```

### 3.5 Complete Example

```python
import triton
import triton.language as tl

@triton.jit(do_not_specialize=[2])
def example_kernel(
    input_ptr: tl.pointer_type(tl.float32),  # pointer, type known from annotation
    n_elements: tl.constexpr,                 # compile-time constant
    stride: int,                              # not specialized (index 2)
    scale: float = 1.0,                       # float, default value
    output_ptr,                               # no annotation
    BLOCK_SIZE: tl.constexpr = 1024,          # constexpr with default
):
    pass

# The resulting params:
# params[0]: num=0, annotation="*fp32", is_const=False, is_constexpr=False,
#            annotation_type="*fp32", do_not_specialize=False
# params[1]: num=1, annotation="constexpr", is_const=False, is_constexpr=True,
#            annotation_type="", do_not_specialize=False
# params[2]: num=2, annotation="i32", is_const=False, is_constexpr=False,
#            annotation_type="i32", do_not_specialize=True
# params[3]: num=3, annotation="fp32", is_const=False, is_constexpr=False,
#            annotation_type="fp32", do_not_specialize=False
# params[4]: num=4, annotation="", is_const=False, is_constexpr=False,
#            annotation_type="", do_not_specialize=False
# params[5]: num=5, annotation="constexpr", is_const=False, is_constexpr=True,
#            annotation_type="", do_not_specialize=False
```

---

## 4. KernelInterface Class

`KernelInterface` is a generic mixin class that provides the grid-based
launch syntax for JIT functions.

### 4.1 Definition

```python
class KernelInterface(Generic[T]):
    run: T

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True,
                       *map(MockTensor.wrap_dtype, args), **kwargs)

    def run(self, *args, grid, warmup, **kwargs):
        raise NotImplementedError("run not implemented")

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False,
                                                 *args, **kwargs)
```

### 4.2 Grid Syntax

The `__getitem__` method enables the `kernel[grid](args)` syntax:

```python
# Grid is a tuple of (X, Y, Z) dimensions
add_kernel[(1024 + BLOCK_SIZE - 1) // BLOCK_SIZE,)](
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE=1024
)

# Grid can be a callable that takes bound args
kernel[lambda args: (triton.cdiv(args['n'], 1024),)](
    data_ptr, n=n_elements, BLOCK_SIZE=1024
)
```

The grid parameter can be:
- A 1-tuple `(X,)` for 1D grids
- A 2-tuple `(X, Y)` for 2D grids
- A 3-tuple `(X, Y, Z)` for 3D grids
- A callable that receives the bound arguments dict and returns a tuple

### 4.3 Warmup

The `warmup` method compiles the kernel without launching. It wraps
any `torch.dtype` arguments into `MockTensor` objects:

```python
# Warmup without real tensors
kernel.warmup(
    MockTensor(torch.float32),
    1024,
    BLOCK_SIZE=1024,
    grid=lambda args: (1,)
)
```

### 4.4 Type Parameter

The generic type parameter `T` represents the type of the `run`
method. For `JITFunction`, this is the callable signature of the
kernel.

---

## 5. JITCallable Class

`JITCallable` is the base class for both `JITFunction` and
`ConstexprFunction`. It provides source code management, hash
computation, and dependency tracking.

### 5.1 Definition

```python
class JITCallable:

    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        try:
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e
        self._fn_name = get_full_name(fn)
        self._hash_lock = threading.RLock()

        raw_src_str = "".join(self.raw_src)
        self.file_name = fn.__code__.co_filename
        self.def_file_line_number = get_def_line_number(self.raw_src,
                                                         self.starting_line_number)
        self.def_file_col_number = get_def_col_number(raw_src_str)

        src = textwrap.dedent(raw_src_str)
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._src = src
        self.hash = None

        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__
```

### 5.2 Constructor Behavior

1. **Source extraction**: Uses `inspect.getsourcelines` to get the raw
   source code. Raises `ValueError` if the function is not defined in
   a file (e.g., REPL-defined functions).

2. **Source normalization**:
   - `textwrap.dedent` removes common leading whitespace
   - A regex finds the `def` keyword to strip any preceding decorator
     lines

3. **Line/column tracking**:
   - `def_file_line_number`: The line number of the `def` keyword
     (accounting for any decorators above it)
   - `def_file_col_number`: The column number of the `def` keyword
     (for indentation tracking)

4. **Metadata forwarding**: The wrapped function's `__doc__`,
   `__name__`, `__qualname__`, `__globals__`, and `__module__` are
   forwarded to the `JITCallable`.

### 5.3 Properties and Methods

#### `cache_key` (property)

The lazily-computed, thread-safe cache key. This is a SHA-256 hash
that incorporates:

- The function source code
- All transitive dependencies (other `JITCallable` objects)
- Used global variable values
- Line number
- Constexpr global variable names and values

```python
@property
def cache_key(self) -> str:
    with self._hash_lock:
        if self.hash is not None:
            return self.hash
        # Break recursion
        self.hash = f"recursion:{self._fn_name}"
        nonlocals = inspect.getclosurevars(self.fn).nonlocals
        dependencies_finder = DependenciesFinder(
            name=self._fn_name,
            globals=self.__globals__,
            nonlocals=nonlocals,
            src=self.src
        )
        dependencies_finder.visit(self.parse())
        self.hash = dependencies_finder.ret + str(self.starting_line_number)
        self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

        from triton.language.core import constexpr
        self.hash += str([(name, val)
                          for (name, _), (val, _) in self.used_global_vals.items()
                          if isinstance(val, constexpr)])
        self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
    return self.hash
```

Key implementation details:
- **Thread safety**: Uses `threading.RLock` to prevent concurrent
  hash computation
- **Recursion guard**: Sets a placeholder hash before computing to
  handle self-referential functions
- **Transitive dependencies**: `DependenciesFinder` walks the AST to
  find all referenced `JITCallable` objects and merges their
  `used_global_vals`

#### `__hash__(self)`

Returns `hash(self.cache_key)`, enabling `JITCallable` objects to be
used as dictionary keys and in sets.

#### `parse(self)`

Parses the source code into an AST and validates it contains exactly
one function definition:

```python
def parse(self):
    tree = ast.parse(self._src)
    assert isinstance(tree, ast.Module)
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.FunctionDef)
    return tree
```

#### `type` (property)

Returns a `constexpr_type` wrapping this callable, used in the Triton
type system:

```python
@property
def type(self):
    from triton.language.core import constexpr_type
    return constexpr_type(self)
```

#### `src` (property)

A read-only property that returns `self._src`. Direct assignment is
blocked:

```python
def _set_src(self):
    raise AttributeError("Cannot set attribute 'src' directly. "
                         "Use '_unsafe_update_src()' and manually clear `.hash` "
                         "of all callers instead.")

def _get_src(self):
    return self._src

src = property(fget=_get_src, fset=_set_src)
```

#### `_unsafe_update_src(self, new_src)`

The only method allowed to modify the source code:

```python
def _unsafe_update_src(self, new_src):
    self.hash = None
    self._src = new_src
```

This resets the hash to `None` so it will be recomputed on next access.
Callers are responsible for clearing hashes of any functions that call
this one.

#### `_flatten_ir(self, handles)`

A no-op method defined for interface compatibility:

```python
def _flatten_ir(self, handles: list[ir.value]) -> None:
    pass
```

#### `get_capture_scope(self)`

Returns the combined global and nonlocal variables accessible from
the function:

```python
def get_capture_scope(self):
    fn = self.fn
    if fn.__closure__ is None:
        return self.__globals__
    nonlocals = {
        name: cell.cell_contents
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)
    }
    return self.__globals__ | nonlocals
```

### 5.4 Used Global Values Tracking

The `used_global_vals` dictionary tracks global variables referenced
by the function and all its transitive dependencies:

```python
# Key: (variable_name, id(__globals__))
# Value: (deepcopy_of_value, __globals__)
self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}
```

This enables validation at kernel launch time that global variables
have not changed since compilation:

```python
for (name, _), (val, globals_dict) in self.used_global_vals.items():
    if (newVal := globals_dict.get(name, not_present)) != val:
        raise RuntimeError(
            f"Global variable {name} has changed since we compiled this kernel")
```

---

## 6. ConstexprFunction and constexpr_function

`ConstexprFunction` wraps a Python function so it can be called at
compile-time on `constexpr` arguments within a Triton kernel and
returns a `constexpr` result.

### 6.1 constexpr_function Decorator

```python
def constexpr_function(fn: T) -> ConstexprFunction[T]:
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """
    return ConstexprFunction(fn)
```

### 6.2 ConstexprFunction Class

```python
class ConstexprFunction(JITCallable, Generic[T]):

    def __init__(self, fn):
        super().__init__(fn)

    def __get__(self, obj, objclass):
        # Support constexpr_function as methods
        if obj is not None:
            return BoundConstexprFunction(obj, self)
        return self

    @overload
    def __call__(self, *args, **kwargs) -> R: ...

    def __call__(self, *args, _semantic=None, **kwargs):
        from triton.language.core import _unwrap_if_constexpr, constexpr
        # Unwrap constexpr arguments to plain Python values
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        # Call the raw Python function
        res = self.fn(*args, **kwargs)

        if _semantic is None:
            # Called from host code or another constexpr function
            return res

        # Called from Triton code generator: wrap result as constexpr
        if knobs.runtime.interpret:
            return res  # No constexpr wrapping in interpreter mode
        return constexpr(res)
```

### 6.3 How It Works

The `_semantic` parameter controls the calling context:

- **`_semantic=None`** (default): The function is called from host
  code, another `constexpr_function`, or an aggregate's `__init__`.
  The raw Python result is returned.

- **`_semantic=<non-None>`**: The function is called by the Triton
  code generator during kernel compilation. The result is wrapped in
  `constexpr()` to make it a compile-time constant.

### 6.4 BoundConstexprFunction

Supports using `constexpr_function` as a method:

```python
class BoundConstexprFunction(JITCallable):

    def __init__(self, instance, fn):
        self.__self__ = instance
        self.__func__ = fn

    @property
    def cache_key(self):
        return self.__func__.cache_key

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)
```

### 6.5 Usage Examples

**Basic constexpr function:**

```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    # compute_block_size is evaluated at compile time
    effective_size = compute_block_size(BLOCK_SIZE)
    pass

@triton.constexpr_function
def compute_block_size(base_size):
    # This is pure Python, executed at compile time
    if base_size > 256:
        return base_size // 2
    return base_size
```

**Using as a method:**

```python
class MyHelper:
    def __init__(self, factor):
        self.factor = factor

    @triton.constexpr_function
    def scale(self, value):
        return value * self.factor  # self.factor must be constexpr at call site
```

**Calling from host code:**

```python
@triton.constexpr_function
def cdiv(a, b):
    return (a + b - 1) // b

# This works in regular Python code (returns plain int)
result = cdiv(1024, 32)  # returns 32

# This works inside a @triton.jit kernel (returns constexpr)
@triton.jit
def kernel(n, BLOCK_SIZE: tl.constexpr):
    num_blocks = cdiv(n, BLOCK_SIZE)  # evaluated at compile time if n is constexpr
```

---

## 7. Specialization System

Specialization is the process by which Triton determines the types and
values of kernel arguments to decide whether a new compilation is
needed or a cached kernel can be reused.

### 7.1 What Gets Specialized

Each argument is classified into one of these categories:

| Category | Example | Specialization Result |
|----------|---------|----------------------|
| `None` | `None` | `("constexpr", None)` |
| `bool` | `True` | `("u1", None)` |
| `int` | `42` | `("i32", key)` or `("i64", key)` or `("u64", key)` |
| `float` | `3.14` | `("fp32", None)` |
| Tensor (has `data_ptr`) | `torch.Tensor` | `("*fp32", key)` or `("*kfp32", key)` |
| `JITCallable` | `@triton.jit` function | `("constexpr", cache_key)` |
| `constexpr` | `tl.constexpr(42)` | `("constexpr", value)` |
| `tuple` | `(1, 2, 3)` | Recursively specialized |
| `TensorDescriptor` | descriptor object | `("tensordesc<fp32[16, 16]>", None)` |

### 7.2 Specialization Tuples

The specialization for each argument is a 2-tuple `(type_string,
specialization_key)`:

- **`type_string`**: The Triton type (e.g., `"i32"`, `"*fp32"`,
  `"constexpr"`, `"tensordesc<fp32[16, 16]>"`)
- **`specialization_key`**: An optional key for cache differentiation.
  `None` means no per-value specialization. A non-None value creates
  separate cache entries for different argument values.

### 7.3 Integer Specialization Details

Integer arguments are specialized based on their value range:

```python
# From the reference implementation:
if isinstance(arg, int):
    key = backend.get_int_specialization(arg, align=align) if specialize_value else None
    if arg == 1 and specialize_value:
        return ("constexpr", 1)
    elif -(2**31) <= arg and arg <= 2**31 - 1:
        return ("i32", key)
    elif 2**63 <= arg and arg <= 2**64 - 1:
        return ("u64", key)
    else:
        return ("i64", key)
```

Key behaviors:
- `1` is always treated as a constexpr when specialization is enabled
- Values in `[-2^31, 2^31 - 1]` map to `i32`
- Values in `[2^63, 2^64 - 1]` map to `u64`
- All other values map to `i64`

### 7.4 Pointer Specialization

Pointer arguments are specialized based on:
- The element type (from `arg.dtype`)
- Whether the pointer is const (from annotation or wrapping)
- Alignment (from `arg.data_ptr()` value)

```python
if hasattr(arg, "data_ptr"):
    dsk = (arg.dtype, is_const)
    res = ("*k" if dsk[1] else "*") + canonicalize_dtype(dsk[0])
    key = backend.get_tensor_specialization(arg, align=align) if specialize_value else None
    return (res, key)
```

The specialization key encodes information about the pointer's
alignment, enabling the compiler to generate optimized memory access
patterns.

### 7.5 do_not_specialize

The `do_not_specialize` parameter prevents specialization for
specified arguments:

```python
@triton.jit(do_not_specialize=[2])  # by index
def kernel(ptr, n, stride, BLOCK_SIZE: tl.constexpr):
    pass

@triton.jit(do_not_specialize=["stride"])  # by name
def kernel(ptr, n, stride, BLOCK_SIZE: tl.constexpr):
    pass
```

When `do_not_specialize` is set for a parameter, the specialization
key for that argument is always `None`, meaning different values will
reuse the same compiled kernel.

### 7.6 do_not_specialize_on_alignment

Similarly, `do_not_specialize_on_alignment` prevents alignment-based
specialization for pointer arguments:

```python
@triton.jit(do_not_specialize_on_alignment=[0])
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    # ptr will be typed but not specialized on alignment
    pass
```

### 7.7 Annotation-Driven Type Override

When a parameter has a type annotation that resolves to a known Triton
type, the annotation overrides runtime type inference:

```python
@triton.jit
def kernel(
    ptr: tl.pointer_type(tl.float32),  # Always *fp32, regardless of actual dtype
    n: tl.int32,                        # Always i32
    scale: tl.float32,                  # Always fp32
):
    pass
```

However, for `u1` (bool) and `fp*`/`bf*` (float/bfloat) types,
specialization is disabled even with annotations:

```python
# This prevents creating separate kernels per float value
if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
    specialize = False
```

### 7.8 The create_function_from_signature Function

This function generates an optimized Python function that performs
argument binding and specialization in a single call:

```python
def create_function_from_signature(sig, kparams, backend):
```

It constructs a function string using `exec` that avoids the overhead
of `inspect.Signature.bind()` and `apply_defaults()` on every kernel
launch:

```python
# Example generated function body:
def dynamic_func(x, y, n, BLOCK_SIZE=1024, **options):
    params = {'x': x, 'y': y, 'n': n, 'BLOCK_SIZE': BLOCK_SIZE}
    specialization = [
        specialize_impl(backend, x, False, True, True),   # x: tensor
        specialize_impl(backend, y, False, True, True),   # y: tensor
        ("i32",) + specialize_impl(backend, n, False, True, True)[1:],  # n: annotated i32
        ("constexpr", BLOCK_SIZE),                         # BLOCK_SIZE: constexpr
    ]
    return params, specialization, options
```

This function is memoized per kernel and reused across launches,
avoiding the significant overhead of `sig.bind()` + `apply_defaults()`.

### 7.9 Custom Pipeline Stages

When `knobs.runtime.add_stages_inspection_hook` is set, additional
specialization data is appended:

```python
if knobs.runtime.add_stages_inspection_hook is not None:
    inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
    specialization.append(f'("custom_pipeline", {inspect_stages_hash})')
```

This allows external tools to inject custom compilation pipeline
stages based on runtime conditions.

---

## 8. Type Mangling

Type mangling converts Python and Triton type annotations into
canonical string representations used in cache keys and kernel
signatures.

### 8.1 _normalize_ty Function

```python
def _normalize_ty(ty) -> str:
    import triton.language.core as core
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_ty(ty[1:])
        if ty.startswith("tl."):
            return _normalize_ty(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_type):
        return f"*{_normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)
```

### 8.2 Mangling Rules

| Input | Normalized Output |
|-------|-------------------|
| `"float32"` | `"fp32"` |
| `"half"` | `"fp16"` |
| `"bfloat16"` | `"bf16"` |
| `"bool"` | `"u1"` |
| `"int32"` | `"i32"` |
| `"int64"` | `"i64"` |
| `"tl.float32"` | `"fp32"` |
| `"tl.pointer_type(tl.float32)"` | `"*fp32"` |
| `"const *float32"` | `"*kfp32"` |
| `tl.dtype.float32` | `"fp32"` |
| `tl.pointer_type(tl.float32)` | `"*fp32"` |
| `int` | `"int"` |
| `float` | `"float"` |

### 8.3 Const Pointer Encoding

Const pointers use the `*k` prefix:

```python
# "const float32*" becomes:
# 1. Strip "const" -> "float32*"
# 2. Normalize "float32*" -> "*fp32" (trailing * handling)
# 3. Since we started with "const", prepend "*k"
# Final: "*kfp32"
```

### 8.4 Pointer Dereference in Mangling

Both prefix and suffix `*` are handled:

```python
# "float32*" -> strip trailing * -> "*float32" -> recurse -> "*fp32"
# "*float32" -> strip leading * -> "float32" -> recurse -> "*fp32"
```

### 8.5 mangle_type Function

A convenience function for external use:

```python
def mangle_type(arg, specialize=False):
    is_const = False
    align = True
    return native_specialize_impl(BaseBackend, arg, is_const, specialize, align)[0]
```

This calls the native C++ specialization implementation to get the
type string for an argument without full specialization.

### 8.6 Type Canonicalization Dictionary

The `_utils.py` module provides `type_canonicalisation_dict`:

```python
type_canonicalisation_dict = {
    "bool": "u1", "int1": "u1", "uint1": "u1", "i1": "u1",
    "float8e4nv": "fp8e4nv", "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16", "float16": "fp16", "bfloat16": "bf16",
    "float": "fp32", "float32": "fp32",
    "double": "fp64", "float64": "fp64",
    "int8": "i8", "int16": "i16", "int": "i32", "int32": "i32", "int64": "i64",
    "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
    "void": "void",
}
# Self-referential: "fp32" -> "fp32", "i32" -> "i32", etc.
for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v
```

---

## 9. Dependency Finding (DependenciesFinder)

`DependenciesFinder` is an AST visitor that computes a hash of a
JIT function and all its dependencies. This hash is used as the base
for the function's cache key.

### 9.1 Class Definition

```python
class DependenciesFinder(ast.NodeVisitor):

    def __init__(self, name, globals, nonlocals, src) -> None:
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))

        self.globals = globals
        self.nonlocals = nonlocals

        self.supported_python_builtins = {
            'float', 'getattr', 'int', 'isinstance', 'len',
            'list', 'max', 'min', 'print', 'range',
        }
        self.supported_modules = {
            GLUON_MODULE,        # triton.experimental.gluon.language
            TRITON_MODULE,       # triton.language
            "copy",
            "math",
        }

        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}
        self.visiting_arg_default_value = False
```

### 9.2 Hash Computation

The hash is incrementally built from:

1. **Function source code**: `self.hasher = hashlib.sha256(src.encode("utf-8"))`

2. **Transitive dependencies**: Each referenced `JITCallable`'s cache
   key is incorporated:
   ```python
   def _update_hash(self, func):
       func_key = func.cache_key
       func_key += str(getattr(func, "noinline", False))
       self.hasher.update(func_key.encode("utf-8"))
   ```

3. **Global variable annotations**: For objects with
   `__triton_aggregate__`:
   ```python
   if getattr(val, "__triton_aggregate__", False):
       self.hasher.update(str(val.__annotations__).encode("utf-8"))
       for attr in val.hash_attrs:
           self.record_reference(attr)
   ```

### 9.3 Global Variable Tracking

The `record_reference` method handles different kinds of values:

```python
def record_reference(self, val, var_dict=None, name=None):
    # None and modules are ignored
    if val is None or type(val) is ModuleType:
        return

    # Triton aggregates: hash their annotations and attributes
    if getattr(val, "__triton_aggregate__", False):
        self.hasher.update(str(val.__annotations__).encode("utf-8"))
        for attr in val.hash_attrs:
            self.record_reference(attr)
        return

    # Triton builtins are ignored
    if getattr(val, "__triton_builtin__", False):
        return

    # libdevice stubs are ignored
    if getattr(val, "__module__", "") == "triton.language.extra.libdevice":
        return

    # JITCallable dependencies: merge global vals and update hash
    if isinstance(val, JITCallable):
        self._update_hash(val)
        return

    # Other callables raise an error (not supported in kernels)
    if callable(val) and not isinstance(val, type) and not isinstance(val, constexpr):
        raise RuntimeError(f"Unsupported function referenced: {val}")

    # Default argument values are not tracked (resolved at definition time)
    if self.visiting_arg_default_value:
        return

    # Track the value with deep copy
    if var_dict is not None:
        self.used_global_vals[(name, id(var_dict))] = (copy.deepcopy(val), var_dict)
```

### 9.4 AST Visitor Methods

#### `visit_Name(self, node)`

Handles name references in the AST:

```python
def visit_Name(self, node):
    if type(node.ctx) is ast.Store:
        return node.id  # Assignment target, not a reference

    if node.id in self.local_names:
        return None  # Hidden by local variable

    # Look up in globals, then nonlocals
    val, var_dict = name_lookup(node.id)

    # Supported builtins don't need tracking
    if node.id in self.supported_python_builtins:
        return val

    self.record_reference(val, var_dict, node.id)
    return val
```

#### `visit_Attribute(self, node)`

Handles attribute access like `tl.load` or `math.sqrt`:

```python
def visit_Attribute(self, node):
    lhs = self.visit(node.value)
    while isinstance(lhs, ast.Attribute):
        lhs = self.visit(lhs.value)
    lhs_name = getattr(lhs, "__name__", "")
    if lhs is None or lhs_name in self.supported_modules:
        return None  # Module attribute, no tracking needed
    ret = getattr(lhs, node.attr)
    self.record_reference(ret)
    return ret
```

#### `visit_FunctionDef(self, node)`

Tracks local variable names to avoid false positives from shadowed
globals:

```python
def visit_FunctionDef(self, node):
    self.local_names = {arg.arg for arg in node.args.args}
    self.generic_visit(node)
```

#### `visit_arguments(self, node)`

Visits function arguments, marking default value expressions specially:

```python
def visit_arguments(self, node):
    def visit_defaults(defaults):
        try:
            assert not self.visiting_arg_default_value
            self.visiting_arg_default_value = True
            for expr in defaults:
                if expr is not None:
                    self.visit(expr)
        finally:
            self.visiting_arg_default_value = False

    for arg in itertools.chain(node.posonlyargs, node.args, ...):
        self.visit(arg)
    visit_defaults(node.kw_defaults)
    visit_defaults(node.defaults)
```

#### `visit_Assign(self, node)`

Tracks assignment targets as local names:

```python
def visit_Assign(self, node):
    self.visitAssnTarget(node.targets[0])
    self.generic_visit(node)
```

#### `visit_AnnAssign(self, node)`

Tracks annotated assignment targets:

```python
def visit_AnnAssign(self, node):
    self.visitAssnTarget(node.target)
    self.generic_visit(node)
```

#### `visit_For(self, node)`

Tracks loop variable as a local name:

```python
def visit_For(self, node):
    self.visitAssnTarget(node.target)
    self.generic_visit(node)
```

#### `visit_Tuple(self, node)`

Returns a list of visited elements for tuple unpacking:

```python
def visit_Tuple(self, node):
    return [self.visit(elt) for elt in node.elts]
```

### 9.5 Global Value Consistency Check

When merging `used_global_vals` from a dependency, conflicting values
are detected:

```python
for k in self.used_global_vals.keys() & func.used_global_vals.keys():
    var_name, _ = k
    v1, _ = self.used_global_vals[k]
    v2, _ = func.used_global_vals[k]
    if v1 != v2:
        raise RuntimeError(
            f"Global variable {var_name} has value {v1} when compiling "
            f"{self.name}, but inner kernel {func.__name__} has conflicting "
            f"value {v2} from when it was first compiled.  This is not allowed."
        )
```

This ensures that if two kernels share a global variable, they agree
on its value.

### 9.6 Supported Builtins

The following Python builtins are allowed inside `@triton.jit`
functions without being tracked as dependencies:

```python
supported_python_builtins = {
    'float',      # type conversion
    'getattr',    # attribute access
    'int',        # type conversion
    'isinstance', # type checking
    'len',        # length
    'list',       # list construction
    'max',        # maximum
    'min',        # minimum
    'print',      # debug printing
    'range',      # range iteration
}
```

### 9.7 Supported Modules

References to these modules are not tracked as dependencies:

```python
supported_modules = {
    "triton.experimental.gluon.language",  # Gluon language module
    "triton.language",                      # Core Triton language
    "copy",                                 # Python copy module
    "math",                                 # Python math module
}
```

---

## 10. Memory Allocation

The memory allocation system provides a way for Triton kernels to
allocate global memory workspace during execution.

### 10.1 Buffer Protocol

```python
class Buffer(Protocol):
    def data_ptr(self) -> int:
        ...
```

`Buffer` is a `Protocol` (structural type) that requires only a
`data_ptr()` method returning an integer pointer. Any object with this
method satisfies the protocol (e.g., PyTorch tensors, CuPy arrays).

### 10.2 Allocator Protocol

```python
class Allocator(Protocol):
    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        ...
```

`Allocator` is a callable protocol that takes:
- `size`: Number of bytes to allocate
- `alignment`: Required alignment in bytes
- `stream`: Optional CUDA/HIP stream for the allocation

Returns a `Buffer` object.

### 10.3 NullAllocator

The default allocator that raises an error when called:

```python
class NullAllocator:
    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        raise RuntimeError(
            "Kernel requires a runtime memory allocation, but no allocator was set. "
            "Use triton.set_allocator to specify an allocator."
        )
```

### 10.4 set_allocator

```python
_allocator: ContextVar[Allocator] = ContextVar("_allocator", default=_NULL_ALLOCATOR)

def set_allocator(allocator: Allocator) -> None:
    """
    The allocator function is called during kernel launch for kernels that
    require additional global memory workspace.
    """
    _allocator.set(allocator)
```

The allocator is stored in a `ContextVar`, making it context-local.
This means different async contexts can have different allocators.

**Usage example:**

```python
import torch
import triton

def my_allocator(size: int, alignment: int, stream) -> torch.Tensor:
    """Allocate a tensor that satisfies the Buffer protocol."""
    return torch.empty(size, dtype=torch.int8, device='cuda')

triton.set_allocator(my_allocator)

# Now kernels that need workspace memory will use my_allocator
kernel[grid](...)
```

### 10.5 _AllocatorWrapper

A wrapper class that provides `ContextVar`-like `get()`/`set()` methods:

```python
class _AllocatorWrapper:
    def __init__(self, allocator: Allocator) -> None:
        self._allocator = allocator

    def get(self) -> Allocator:
        return self._allocator

    def set(self, allocator: Allocator) -> None:
        self._allocator = allocator

    def __call__(self, size: int, alignment: int, stream: Optional[int]) -> Buffer:
        return self._allocator(size, alignment, stream)
```

### 10.6 Profile Allocator

A separate allocator for profiling/instrumentation:

```python
_profile_allocator = _AllocatorWrapper(_NULL_ALLOCATOR)

def set_profile_allocator(allocator: Optional[Allocator]) -> None:
    """
    The profile allocator function is called before kernel launch for kernels
    that require additional global memory workspace.
    """
    _profile_allocator.set(allocator if allocator is not None else _NULL_ALLOCATOR)

def has_profile_allocator() -> bool:
    return not isinstance(_profile_allocator.get(), NullAllocator)
```

The profile allocator is separate from the regular allocator because
it may need to allocate memory before the kernel runs (for
instrumentation buffers).

**Usage example:**

```python
# Set up profiling allocator
def profiling_allocator(size, alignment, stream):
    print(f"Allocating {size} bytes for profiling")
    return torch.empty(size, dtype=torch.int8, device='cuda')

triton.set_profile_allocator(profiling_allocator)

# Check if profiling is active
if triton.has_profile_allocator():
    print("Profiling is enabled")
```

---

## 11. Async Compilation

The async compilation system allows kernel compilation to happen in
parallel threads, overlapping compilation with other work.

### 11.1 AsyncCompileMode

```python
class AsyncCompileMode:

    def __init__(self, executor: Executor, *, ignore_errors=False):
        self.executor = executor
        self.ignore_errors = ignore_errors
        self.raw_futures = []
        self.future_kernels = {}
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `executor` | `Executor` | A `concurrent.futures.Executor` for submitting compilation tasks |
| `ignore_errors` | `bool` | If True, compilation errors are silently ignored |

#### Context Manager

`AsyncCompileMode` is a context manager that sets itself as the active
async mode:

```python
def __enter__(self):
    if active_mode.get() is not None:
        raise RuntimeError("Another AsyncCompileMode is already active")
    active_mode.set(self)
    return self

def __exit__(self, exc_type, exc_value, traceback):
    active_mode.set(None)
    # Finalize any outstanding compiles
    for future in as_completed(self.raw_futures):
        self.future_kernels[future._key].result(self.ignore_errors)
```

Key behaviors:
- Only one `AsyncCompileMode` can be active at a time
- On exit, all pending compiles are finalized (blocking until done)
- Errors during finalization are suppressed if `ignore_errors=True`

#### submit Method

```python
def submit(self, key, compile_fn, finalize_fn):
    future = self.future_kernels.get(key)
    if future is not None:
        return future  # Deduplicate: same key returns existing future

    future = self.executor.submit(compile_fn)
    future._key = key
    self.raw_futures.append(future)
    future_kernel = FutureKernel(finalize_fn, future)
    self.future_kernels[key] = future_kernel
    return future_kernel
```

Key behaviors:
- **Deduplication**: If a compilation with the same key was already
  submitted, the existing `FutureKernel` is returned
- The `key` is stored on the `Future` object for later lookup during
  finalization
- The `compile_fn` is submitted to the executor for background execution

### 11.2 FutureKernel

```python
class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future

    def result(self, ignore_errors: bool = False):
        if self.kernel is not None:
            return self.kernel  # Already finalized

        try:
            kernel = self.future.result()
        except Exception:
            if ignore_errors:
                return
            else:
                raise
        self.finalize_compile(kernel)
        self.kernel = kernel
        return kernel

    def __getattr__(self, name):
        # Proxy attribute access to the compiled kernel
        return getattr(self.result(), name)
```

Key behaviors:
- **Lazy finalization**: The kernel is finalized only when `result()`
  is first called
- **Transparent proxy**: `__getattr__` delegates to the compiled kernel,
  so `FutureKernel` can be used like a regular `CompiledKernel`
- **Error handling**: Optional error suppression for batch compilation

### 11.3 Active Mode Context Variable

```python
active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar(
    "async_compile_active_mode", default=None
)
```

The active async mode is stored in a `ContextVar`, checked during
`_do_compile`:

```python
async_mode = _async_compile.active_mode.get()
if async_mode is not None:
    # Submit for async compilation
    ...
else:
    # Compile synchronously
    ...
```

### 11.4 Integration with JITFunction

In `_do_compile`, the async path:

1. Computes a cache key for the source
2. Defines `async_compile()` -- calls `self.compile()`
3. Defines `finalize_compile(kernel)` -- stores in cache and calls post-compile hook
4. Submits via `async_mode.submit(cache_key, async_compile, finalize_compile)`
5. Stores the `FutureKernel` in the kernel cache immediately

This means subsequent calls to `run()` may find a `FutureKernel` in
the cache. When the kernel is actually needed for launch, the
`FutureKernel.result()` call blocks until compilation completes.

### 11.5 Complete Usage Example

```python
import triton
from concurrent.futures import ThreadPoolExecutor

# Create an executor with 4 threads
executor = ThreadPoolExecutor(max_workers=4)

# Use async compilation for multiple kernels
with triton.runtime._async_compile.AsyncCompileMode(executor, ignore_errors=False):
    # These kernel launches will compile in parallel
    kernel1[grid1](...)
    kernel2[grid2](...)
    kernel3[grid3](...)
    # ...
# At context exit, all outstanding compilations are finalized

# Alternative: ignore compilation errors (useful for auto-tuning)
with triton.runtime._async_compile.AsyncCompileMode(executor, ignore_errors=True):
    for config in tuning_configs:
        kernel[grid](..., **config)
```

---

## 12. TensorWrapper and MockTensor

### 12.1 TensorWrapper

`TensorWrapper` wraps a tensor with a different dtype interpretation,
enabling type reinterpretation without copying data.

#### Definition

```python
class TensorWrapper:

    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.data = base.data
        self.device = base.device
        self.shape = self.base.shape
```

#### Methods

| Method | Description |
|--------|-------------|
| `data_ptr()` | Returns `self.base.data_ptr()` |
| `stride(*args)` | Returns `self.base.stride(*args)` |
| `element_size()` | Returns `self.base.element_size()` |
| `cpu()` | Returns `TensorWrapper(self.base.cpu(), self.dtype)` |
| `copy_(other)` | Copies from `other.base` to `self.base` |
| `clone()` | Returns `TensorWrapper(self.base.clone(), self.dtype)` |
| `to(device)` | Returns `TensorWrapper(self.base.to(device), self.dtype)` |
| `new_empty(sizes)` | Returns `TensorWrapper(self.base.new_empty(sizes), self.dtype)` |
| `__str__()` | Returns `TensorWrapper[{dtype}]({base})` |

#### reinterpret Function

```python
def reinterpret(tensor, dtype):
    if isinstance(tensor, TensorWrapper):
        if dtype == tensor.base.dtype:
            return tensor.base  # Unwrap back to original
        else:
            return TensorWrapper(tensor.base, dtype)  # Re-wrap with new dtype
    elif hasattr(tensor, "data_ptr"):
        return TensorWrapper(tensor, dtype)  # New wrapper
    else:
        raise TypeError(f"Cannot reinterpret a {type(tensor)}.")
```

#### Usage Examples

```python
import torch
import triton
from triton.runtime.jit import TensorWrapper, reinterpret

# Reinterpret a float32 tensor as int32 (bit-level reinterpretation)
x = torch.randn(100, device='cuda', dtype=torch.float32)
x_as_int = reinterpret(x, torch.int32)
# x_as_int.dtype is torch.int32, but shares memory with x

# Use in kernel
@triton.jit
def bitwise_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # input_ptr will be treated as int32 pointers
    pass

# Unwrap back to original type
original = reinterpret(x_as_int, torch.float32)
assert original is x  # True - returns the original unwrapped tensor

# Chain reinterpretations
x_as_bf16 = reinterpret(x, torch.bfloat16)
x_back = reinterpret(x_as_bf16, torch.float32)
assert x_back is x  # Unwraps fully
```

### 12.2 MockTensor

`MockTensor` is a lightweight stand-in for real tensors during kernel
warmup, avoiding the need to allocate actual GPU memory.

#### Definition

```python
class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if arg.__class__.__name__ == "dtype" and arg.__module__ == "torch":
            return MockTensor(arg)
        return arg

    def __init__(self, dtype, shape=None):
        if shape is None:
            shape = [1]
        self.dtype = dtype
        self.shape = shape

    def stride(self):
        strides = [1]
        for size in self.shape[1:]:
            strides.append(strides[-1] * size)
        return tuple(reversed(strides))

    @staticmethod
    def data_ptr():
        return 0  # optimistically assumes multiple of 16

    @staticmethod
    def ptr_range():
        return 0  # optimistically assumes 32 bit pointer range
```

#### Key Details

- **`wrap_dtype`**: Automatically wraps `torch.dtype` objects (like
  `torch.float32`) into `MockTensor` instances. This is used by
  `KernelInterface.warmup()`.

- **`data_ptr`**: Returns 0, which is aligned to 16 bytes (matching
  the compiler's optimistic alignment assumption).

- **`ptr_range`**: Returns 0, indicating a 32-bit pointer range.

- **`stride`**: Computes contiguous strides from the shape, matching
  PyTorch's default stride computation.

#### Usage Examples

```python
import torch
import triton
from triton.runtime.jit import MockTensor

@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pass

# Warmup with mock tensors (no GPU memory allocated)
my_kernel.warmup(
    MockTensor(torch.float32),
    MockTensor(torch.float32),
    1024,
    BLOCK_SIZE=1024,
    grid=lambda args: (1,),
)

# Warmup with dtype auto-wrapping
my_kernel.warmup(
    torch.float32,  # Automatically wrapped to MockTensor
    torch.float32,
    1024,
    BLOCK_SIZE=1024,
    grid=lambda args: (1,),
)

# Warmup with shaped mock tensors
mock = MockTensor(torch.float32, shape=[1024, 1024])
# mock.stride() returns (1024, 1)
```

---

## 13. Cache Key Computation

Cache keys determine whether a kernel needs recompilation or can reuse
a previously compiled version.

### 13.1 Two-Level Cache

Triton uses a two-level cache:

1. **Kernel key cache**: Maps `(specialization_tuple, options_string)`
   to a cache key string. This avoids recomputing the expensive
   string representation on every call.

2. **Kernel cache**: Maps cache key string to the compiled kernel
   (`CompiledKernel` or `FutureKernel`).

### 13.2 compute_cache_key Function

```python
def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key
```

### 13.3 Cache Key Components

The cache key is composed of:

1. **Specialization data**: The full list of specialization tuples for
   all parameters. Each tuple contains `(type_string, specialization_key)`.

2. **Compilation options**: String representation of all compilation
   options (num_warps, num_ctas, num_stages, etc.).

3. **JITCallable references**: Any `JITCallable` objects in the
   specialization are replaced with their `cache_key` property, which
   is a SHA-256 hash incorporating the function source and all
   transitive dependencies.

### 13.4 Cache Key for ASTSource (in cache.py)

When checking the persistent cache:

```python
def get_cache_key(src, backend, backend_options, env_vars):
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
    return key
```

This incorporates:
- **triton_key()**: The Triton version key
- **src.hash()**: Hash of the AST source
- **backend.hash()**: Hash of the backend implementation
- **backend_options.hash()**: Hash of the compilation options
- **env_vars**: Cache-invalidating environment variables

### 13.5 JITCallable Cache Key (JITCallable.cache_key)

The `JITCallable.cache_key` property computes a SHA-256 hash from:

1. The function's source code (via `DependenciesFinder`)
2. All transitive dependency cache keys
3. The starting line number
4. Constexpr global variable names and values

```python
@property
def cache_key(self) -> str:
    with self._hash_lock:
        if self.hash is not None:
            return self.hash
        self.hash = f"recursion:{self._fn_name}"  # recursion guard
        # ... compute dependencies ...
        self.hash = dependencies_finder.ret + str(self.starting_line_number)
        # ... add constexpr globals ...
        self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
    return self.hash
```

### 13.6 Cache Invalidation

The cache is invalidated when:

1. **Source code changes**: The hash incorporates the source code text
2. **Dependencies change**: Any transitively referenced `JITCallable`
   whose source or dependencies changed
3. **Global variables change**: At kernel launch, the values of tracked
   global variables are checked against their stored values
4. **Specialization changes**: Different argument types or values that
   produce different specialization tuples
5. **Options change**: Different compilation options (num_warps, etc.)
6. **Environment variables change**: `get_cache_invalidating_env_vars()`
   provides a set of environment variables that invalidate the cache

### 13.7 Global Variable Change Detection

At every kernel launch, the runtime checks that global variables have
not changed:

```python
not_present = object()
for (name, _), (val, globals_dict) in self.used_global_vals.items():
    if (newVal := globals_dict.get(name, not_present)) != val:
        raise RuntimeError(
            f"Global variable {name} has changed since we compiled "
            f"this kernel, from {val} to {newVal}")
```

This is a runtime check (not a cache invalidation), because the
function would need to be re-hashed if the source code references
changed global values.

### 13.8 Cache Lookup Flow

```
run() called
  -> binder(*args) returns (bound_args, specialization, options)
  -> compute_cache_key(kernel_key_cache, specialization, options)
     -> check kernel_key_cache for (specialization, options)
     -> if miss: compute string key, replace JITCallables with hashes
     -> cache for future use
  -> kernel_cache.get(key)
     -> if miss: compile and store
     -> if hit: use cached kernel
```

---

## 14. Helper Utilities

### 14.1 get_full_name

```python
def get_full_name(fn):
    return f"{fn.__module__}.{fn.__qualname__}"
```

Returns the fully qualified name of a function (module + qualname).

### 14.2 get_def_line_number

```python
def get_def_line_number(raw_src, starting_line_number):
    def_file_line_number = starting_line_number
    for idx, line in enumerate(raw_src):
        if line.strip().startswith("def "):
            def_file_line_number += idx
            break
    return def_file_line_number
```

Finds the actual `def` line, skipping decorator lines above it.

### 14.3 get_def_col_number

```python
def get_def_col_number(raw_src_str):
    indented_def = INDENT_PATTERN.search(raw_src_str)
    if not indented_def:
        raise ValueError("No function definition found for kernel")
    def_file_col_number = len(indented_def.group("indent"))
    def_file_col_number += 1  # Columns start at 1
    return def_file_col_number
```

Finds the column number of the `def` keyword, accounting for
indentation. Uses the regex pattern:

```python
INDENT_PATTERN = re.compile(r"^(?P<indent>[ \t]*)def\s+\w+\s*\(", re.MULTILINE)
```

### 14.4 serialize_specialization_data

```python
def serialize_specialization_data(name, signature, constants, attrs, options, key, target):
    constants = {
        key: str(value) if value.__class__.__name__ == "dtype" else
              {"constexpr": value.value} if value.__class__.__name__ == "constexpr" else
              {"jit_function": f"{value.module}:{value.fn.__qualname__}"} if value.__class__.__name__ == "JITFunction" else
              value
        for key, value in constants.items()
    }
    obj = {
        'name': name,
        'signature': signature,
        'constant_keys': [list(x) for x in constants.keys()],
        'constant_vals': list(constants.values()),
        'attrs_keys': [list(x) for x in attrs.keys()],
        'attrs_vals': list(attrs.values()),
        'options': options.__dict__,
        'key': key,
        'target': target.__dict__,
    }
    return json.dumps(obj)
```

Serializes specialization data to JSON for `preload()`. Handles
special types:
- `dtype` -> string representation
- `constexpr` -> `{"constexpr": value}`
- `JITFunction` -> `{"jit_function": "module:qualname"}`

### 14.5 convert_to_tuple_if_list

```python
def convert_to_tuple_if_list(item):
    if not isinstance(item, list):
        return item
    for i, nested_value in enumerate(item):
        item[i] = convert_to_tuple_if_list(nested_value)
    return tuple(item)
```

Recursively converts lists to tuples. Used during deserialization
because JSON converts tuples to lists.

### 14.6 _is_triton_builtin

```python
def _is_triton_builtin(self, node, func):
    if inspect.isbuiltin(node.func):
        return True
    module = getattr(func, "__module__", "")
    return module.startswith(TRITON_MODULE)
```

Checks if a function call is a Triton builtin (either a Python builtin
or a function from the `triton.language` module).

---

## 15. Global Registries and Hooks

### 15.1 _triton_jit_function_registry

```python
_triton_jit_function_registry = {}
```

A global dictionary mapping `"module:qualname"` strings to their
`JITFunction` objects. Used for deserialization of `JITFunction`
references in `preload()`:

```python
# Registration (in JITFunction.__init__):
_triton_jit_function_registry[f"{self.module}:{self.fn.__qualname__}"] = self

# Lookup (in preload):
jf_key = value['jit_function']
if jf_key in _triton_jit_function_registry:
    return _triton_jit_function_registry[jf_key]
```

### 15.2 JitFunctionInfo

```python
@dataclass
class JitFunctionInfo:
    module: ModuleType
    name: str
    jit_function: JITFunction
```

A dataclass that packages JIT function information for hook callbacks.

### 15.3 Pre-Run Hooks

```python
# Adding a hook:
kernel.add_pre_run_hook(my_hook)

# Hooks are called before compilation/launch:
for hook in self.pre_run_hooks:
    hook(*args, **kwargs)
```

Pre-run hooks receive the same arguments as the kernel launch and can
be used for logging, profiling, or argument validation.

### 15.4 JIT Cache Hooks

Two hook points in the compilation lifecycle:

1. **`knobs.runtime.jit_cache_hook`**: Called before compilation. If
   it returns truthy, compilation is skipped.

2. **`knobs.runtime.jit_post_compile_hook`**: Called after compilation
   succeeds.

Both hooks receive the same arguments:
```python
hook(
    key=key,
    repr=repr,
    fn=JitFunctionInfo(module, name, self),
    compile={
        'signature': signature,
        'device': device,
        'constants': constants,
        'num_warps': options.num_warps,
        'num_ctas': options.num_ctas,
        'num_stages': options.num_stages,
        'enable_fp_fusion': options.enable_fp_fusion,
        'launch_cooperative_grid': options.launch_cooperative_grid,
        'extern_libs': options.extern_libs,
        'configs': configs,
        'specialization_data': specialization_data,
        'is_warmup': is_warmup,
    },
    is_manual_warmup=is_warmup,
    already_compiled=False,
)
```

### 15.5 Launch Hooks

```python
# Configured via knobs:
knobs.runtime.launch_enter_hook  # Called before kernel launch
knobs.runtime.launch_exit_hook   # Called after kernel launch
```

These are `HookChain` objects that allow multiple hooks to be
registered in sequence.

### 15.6 Pipeline Stages Hook

```python
knobs.runtime.add_stages_inspection_hook
```

An optional hook that returns a `(key, hash)` tuple for injecting
custom compilation pipeline stages. When set, the hash is appended
to the specialization data.

---

## Appendix A: Complete Kernel Launch Example

This example traces through the entire JIT system for a simple kernel:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr,        # *fp32 (inferred from tensor dtype)
    y_ptr,        # *fp32
    output_ptr,   # *fp32
    n_elements,   # i32 (inferred from int value)
    BLOCK_SIZE: tl.constexpr,  # constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Step 1: Decoration
# @triton.jit creates a JITFunction wrapping add_kernel
# - Source code is extracted and dedented
# - KernelParam objects are created for each parameter
# - Function is registered in _triton_jit_function_registry

# Step 2: Launch
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
output = torch.empty_like(x)
BLOCK_SIZE = 256

grid = (triton.cdiv(1024, BLOCK_SIZE),)

add_kernel[grid](x, y, output, 1024, BLOCK_SIZE=BLOCK_SIZE)

# Step 2a: __getitem__(grid) returns lambda
# Step 2b: lambda(*args) calls run(grid=grid, warmup=False, *args)
# Step 2c: binder(x, y, output, 1024, BLOCK_SIZE=256) returns:
#   params = {'x_ptr': x, 'y_ptr': y, 'output_ptr': output,
#             'n_elements': 1024, 'BLOCK_SIZE': 256}
#   specialization = [
#     ('*fp32', 'aligned_16'),    # x_ptr
#     ('*fp32', 'aligned_16'),    # y_ptr
#     ('*fp32', 'aligned_16'),    # output_ptr
#     ('i32', None),              # n_elements (not specialized)
#     ('constexpr', 256),         # BLOCK_SIZE
#   ]
#   options = {}
# Step 2d: compute_cache_key hashes the specialization
# Step 2e: cache miss -> compile
# Step 2f: _pack_args extracts signature, constexprs, attrs
# Step 2g: _do_compile creates ASTSource and calls compile()
# Step 2h: Kernel is cached and launched
```

## Appendix B: Compilation Options

The `options` object (parsed by `backend.parse_options`) typically
contains:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_warps` | `int` | 4 | Number of warps per CTA |
| `num_ctas` | `int` | 1 | Number of CTAs per cluster |
| `num_stages` | `int` | 3 | Number of pipeline stages |
| `enable_fp_fusion` | `bool` | True | Enable FP fusion optimizations |
| `launch_cooperative_grid` | `bool` | False | Enable cooperative grid launch |
| `extern_libs` | `dict` | {} | External library paths |
| `debug` | `bool` | False | Enable debug output |

## Appendix C: Environment Variables

| Variable | Affects | Description |
|----------|---------|-------------|
| `TRITON_INTERPRET` | `knobs.runtime.interpret` | Run kernels in Python interpreter |
| `TRITON_DEBUG` | `knobs.runtime.debug` | Enable debug output |
| `TRITON_DEFAULT_BACKEND` | Driver selection | Force a specific backend |
| `TRITON_FRONT_END_DEBUGGING` | `knobs.compilation.front_end_debugging` | Front-end debug output |

Cache-invalidating environment variables are provided by
`get_cache_invalidating_env_vars()` from the C++ extension.

## Appendix D: Thread Safety

The JIT system is designed to be thread-safe:

- **Hash computation**: Protected by `threading.RLock` (reentrant lock)
  to handle recursive dependency resolution
- **Device caches**: Per-device `defaultdict` avoids cross-device
  contention
- **Async compilation**: Uses `concurrent.futures.Executor` with
  proper synchronization
- **Allocator**: Uses `contextvars.ContextVar` for async-context-safe
  storage

## Appendix E: Error Handling

Common errors and their causes:

| Error | Cause |
|-------|-------|
| `ValueError("@jit functions should be defined in a Python file")` | Function defined in REPL or dynamically |
| `RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")` | Direct call without grid indexing |
| `RuntimeError(f"Global variable {name} has changed...")` | Global variable value changed after first compilation |
| `RuntimeError("Another AsyncCompileMode is already active")` | Nested AsyncCompileMode context managers |
| `RuntimeError(f"Unsupported function referenced: {val}")` | Non-JIT callable referenced in kernel |
| `RuntimeError("Kernel requires a runtime memory allocation, but no allocator was set.")` | Kernel needs workspace but no allocator configured |
| `KeyError("Keyword argument %s was specified but unrecognised")` | Unknown keyword argument passed to kernel |
| `TypeError("Simultaneous multiple assignment is not supported.")` | Multiple assignment targets in kernel code |
