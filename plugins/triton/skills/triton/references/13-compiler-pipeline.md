# 13. Triton Compiler Pipeline

This reference provides an exhaustive description of the Triton compiler pipeline -- the system that transforms Python kernel functions (decorated with `@triton.jit`) into GPU-executable binaries. The pipeline spans AST parsing, MLIR generation, backend-specific lowering, caching, and runtime loading.

**Source files covered:**

- `python/triton/compiler/compiler.py` -- top-level `compile()` function, source abstractions (`ASTSource`, `IRSource`), `CompiledKernel`, cache integration
- `python/triton/compiler/code_generator.py` -- `CodeGenerator` (AST-to-MLIR visitor), `ASTFunction`, `ast_to_ttir`
- `python/triton/compiler/errors.py` -- `CompilationError`, `CompileTimeAssertionFailure`, `UnsupportedLanguageConstruct`

---

## Table of Contents

1. [Compilation Pipeline Overview](#1-compilation-pipeline-overview)
2. [compile() Function](#2-compile-function)
3. [ASTSource Class](#3-astsource-class)
4. [IRSource Class](#4-irsource-class)
5. [CompiledKernel Class](#5-compiledkernel-class)
6. [LazyDict and AsmDict](#6-lazydict-and-asmdict)
7. [CodeGenerator Class](#7-codegenerator-class)
8. [ASTFunction Class](#8-astfunction-class)
9. [Error Handling](#9-error-handling)
10. [Cache System](#10-cache-system)
11. [CompileTimer](#11-compiletimer)
12. [Backend Interface](#12-backend-interface)
13. [Code Examples](#13-code-examples)

---

## 1. Compilation Pipeline Overview

The full compilation flow from Python function to GPU binary proceeds through these stages:

```
Python @triton.jit function
        |
        v
[1] AST Parsing (ast.parse)
        |
        v
[2] CodeGenerator (AST -> TTIR MLIR)
        |   ast_to_ttir() drives CodeGenerator.visit() over the AST
        |   Produces a Triton IR (TTIR) MLIR module
        v
[3] Backend Lowering Stages (backend-specific)
        |   e.g., for CUDA:
        |     ttir  ->  ttgir  ->  llir  ->  ptx  ->  cubin
        |   Each stage is a function: (module, metadata) -> next_module
        v
[4] Cache Storage
        |   All intermediate IR and final binary are stored in a cache manager
        |   Metadata JSON is written with compilation parameters
        v
[5] CompiledKernel Creation
        |   Loads metadata, reads assembly artifacts
        |   Binary loaded lazily on first launch
        v
[6] Runtime Launch
        |   kernel[grid](args...) triggers _init_handles()
        |   Binary loaded onto GPU, launcher function created
```

### Key Design Principles

- **Two source types**: `ASTSource` (from Python functions) and `IRSource` (from pre-existing IR files on disk).
- **Backend-agnostic core**: The `compile()` function is generic; backend-specific stages are injected via `BaseBackend.add_stages()`.
- **Incremental compilation**: If the source is already IR (e.g., `.ttgir`), the pipeline starts from the appropriate stage, skipping earlier ones.
- **Cache-first**: Before any compilation, the cache is checked. A cache hit returns a `CompiledKernel` immediately.
- **Lazy binary loading**: The GPU binary is not loaded until the kernel is actually launched, avoiding unnecessary device operations.

---

## 2. compile() Function

**Location**: `python/triton/compiler/compiler.py`

### Signature

```python
def compile(src, target=None, options=None, _env_vars=None):
```

### Parameters

| Parameter   | Type                              | Description                                                                                              |
|-------------|-----------------------------------|----------------------------------------------------------------------------------------------------------|
| `src`       | `ASTSource` or `str`              | The kernel source. Either an `ASTSource` wrapping a `@triton.jit` function, or a file path string to an IR file. |
| `target`    | `GPUTarget` or `None`             | Target GPU. If `None`, uses `driver.active.get_current_target()`. Must be a `GPUTarget` instance.        |
| `options`   | `dict` or `None`                  | Compilation options (e.g., `num_warps`, `num_ctas`). Merged with `src.parse_options()`, then passed to `backend.parse_options()`. |
| `_env_vars` | `dict` or `None`                  | Cache-invalidating environment variables. If `None`, fetched via `get_cache_invalidating_env_vars()`.    |

### Return Value

Returns a `CompiledKernel` object. This object holds all compiled artifacts, metadata, and the ability to launch the kernel on the GPU.

### Execution Flow (Step by Step)

1. **Target resolution**: If `target` is `None`, obtains the current driver target via `driver.active.get_current_target()`.

2. **Backend creation**: Calls `make_backend(target)` which searches all registered backends for one that `supports_target(target)`. Exactly one backend must match.

3. **Source normalization**: If `src` is a string (file path), wraps it in `IRSource(path, context, backend)`. If it is already an `ASTSource`, it is used directly.

4. **Options parsing**: Extra options from `src.parse_options()` are merged with user-provided `options`, then passed to `backend.parse_options()` which returns a validated options object.

5. **Cache key computation**:
   - `env_vars` are obtained (or provided).
   - `get_cache_key(src, backend, options, env_vars)` produces a key incorporating: Triton version hash, source hash, backend hash, options hash, and sorted environment variables.
   - SHA-256 of this key becomes the cache hash.

6. **Cache managers**: Three cache managers are created:
   - `fn_cache_manager` (via `get_cache_manager(hash)`) -- always created.
   - `fn_override_manager` (via `get_override_manager(src.hash())`) -- only if `TRITON_KERNEL_OVERRIDE` is enabled.
   - `fn_dump_manager` (via `get_dump_manager(src.hash())`) -- only if `TRITON_DUMP_IR` is enabled.

7. **Cache lookup**: The metadata file (`{name}.json`) is looked up in the cache. If found and `TRITON_ALWAYS_COMPILE` is not set, a `CompiledKernel` is returned immediately (cache hit).

8. **IR initialization**:
   - Creates an MLIR `ir.context()`.
   - Loads Triton dialects via `ir.load_dialects(context)`.
   - Loads backend-specific dialects via `backend.load_dialects(context)`.
   - Gets codegen functions via `backend.get_codegen_implementation(options)`.
   - Gets module map via `backend.get_module_map()`.
   - Calls `src.make_ir(target, options, codegen_fns, module_map, context)` to produce the initial MLIR module.
   - Traceback filtering via `filter_traceback()` cleans up error traces from code_generator internals.

9. **Backend stages**: `backend.add_stages(stages, options, src.language)` populates an ordered dict of `{extension: compile_function}`. The pipeline finds the starting stage based on `src.ext` and runs each stage sequentially:
   ```python
   for ext, compile_ir in list(stages.items())[first_stage:]:
       next_module = compile_ir(module, metadata)
   ```
   Each stage function takes `(module, metadata)` and returns the next IR representation. It may also populate `metadata` with additional information (e.g., shared memory size, register count).

10. **Override support**: At each stage, if an override manager is active and has a file for the current stage, that file is used instead of the compiled output.

11. **Metadata finalization**: The full metadata dictionary (including hash, target, options, env vars, triton version, and any backend-populated fields) is serialized to JSON and stored.

12. **Compilation listener**: If `knobs.compilation.listener` is set, it is called with timing information and metadata.

13. **Return**: A `CompiledKernel(src, metadata_group, hash)` is created and returned.

---

## 3. ASTSource Class

**Location**: `python/triton/compiler/compiler.py`

Wraps a Python `@triton.jit` function as a compilation source.

### Constructor

```python
class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None):
```

| Parameter     | Type                  | Description                                                                 |
|---------------|-----------------------|-----------------------------------------------------------------------------|
| `fn`          | `JITFunction`         | The JIT-compiled function object (a `@triton.jit` decorated function).      |
| `signature`   | `dict[str, str]`      | Mapping from argument names to type strings (e.g., `"*fp32"`, `"i32"`).    |
| `constexprs`  | `dict[str/int, value]`| Constexpr values. Keys can be argument names (str) or index tuples.        |
| `attrs`       | `dict` or `None`      | Additional attributes for function arguments (e.g., divisibility hints).   |

### Instance Attributes

| Attribute    | Type        | Description                                                |
|-------------|-------------|------------------------------------------------------------|
| `fn`        | `JITFunction` | The wrapped JIT function.                                 |
| `language`  | `Language`  | Always `Language.TRITON`.                                  |
| `ext`       | `str`       | Always `"ttir"` -- the starting IR extension.              |
| `name`      | `str`       | `fn.__name__` -- the function name.                        |
| `signature` | `dict`      | The type signature mapping.                                |
| `constants` | `dict`      | Constexpr values, with keys normalized to tuples of ints.  |
| `attrs`     | `dict`      | Argument attributes.                                       |

### Methods

#### `hash()`

```python
def hash(self) -> str:
```

Returns a SHA-256 hex digest computed from:
- `fn.cache_key` -- the function's source/content hash.
- `str(self.attrs)` -- attributes dict as string.
- Sorted signature values.
- Sorted constexpr values (using `cache_key` attribute if available, else `str()`).

#### `make_ir(target, options, codegen_fns, module_map, context)`

```python
def make_ir(self, target, options, codegen_fns, module_map, context):
```

Delegates to `ast_to_ttir()` from `code_generator.py`:
```python
from .code_generator import ast_to_ttir
return ast_to_ttir(self.fn, self, context=context, options=options,
                   codegen_fns=codegen_fns, module_map=module_map)
```

Returns an MLIR module containing the TTIR representation of the kernel.

#### `parse_options()`

```python
def parse_options(self) -> dict:
```

Returns an empty dictionary. `ASTSource` does not contribute extra options (these come from the user and backend).

---

## 4. IRSource Class

**Location**: `python/triton/compiler/compiler.py`

Wraps a pre-existing IR file on disk as a compilation source.

### Constructor

```python
class IRSource:
    def __init__(self, path, context, backend):
```

| Parameter  | Type          | Description                                          |
|-----------|---------------|------------------------------------------------------|
| `path`    | `str`         | File path to the IR file.                            |
| `context` | `ir.context`  | MLIR context for parsing.                            |
| `backend` | `BaseBackend` | Backend instance for loading dialects.               |

### Constructor Behavior

1. Stores the file path and reads the file contents into `self.src`.
2. Sets `self.language = Language.TRITON`.
3. Determines `self.ext` from the file suffix (e.g., `"ttir"`, `"ttgir"`, `"ptx"`).
4. Loads Triton and backend dialects into the context.
5. Parses the file to extract function name and signature:
   - **For PTX files**: Uses regex to extract the function name and parameter types.
   - **For MLIR files** (`.ttir`, `.ttgir`): Uses `ir.parse_mlir_module()` to parse, then queries the module for entry function name and signature.

### Instance Attributes

| Attribute   | Type          | Description                                        |
|------------|---------------|----------------------------------------------------|
| `path`     | `str`         | Original file path.                                |
| `ext`      | `str`         | File extension without dot.                        |
| `language` | `Language`    | Always `Language.TRITON`.                           |
| `src`      | `str`         | Full file contents as text.                        |
| `name`     | `str`         | Extracted function name (with `@` prefix for MLIR).|
| `signature`| `dict`        | `{index: type_string}` mapping of argument types.  |
| `module`   | MLIR module   | Parsed MLIR module (for non-PTX files).            |

### Methods

#### `hash()`

```python
def hash(self) -> str:
```

Returns SHA-256 hex digest of `self.src` (the raw file contents).

#### `make_ir(target, options, codegen_fns, module_map, context)`

```python
def make_ir(self, target, options, codegen_fns, module_map, context):
```

Re-associates the stored module with the new context and returns it:
```python
self.module.context = context
return self.module
```

#### `parse_options()`

```python
def parse_options(self) -> dict:
```

For `.ttgir` files, extracts `num_warps` (required) and `num_ctas` (optional) from module attributes. Returns an empty dict for other formats.

---

## 5. CompiledKernel Class

**Location**: `python/triton/compiler/compiler.py`

Represents a fully compiled kernel, holding metadata, assembly artifacts, and runtime launch capability.

### Constructor

```python
class CompiledKernel:
    def __init__(self, src, metadata_group, hash):
```

| Parameter        | Type                | Description                                           |
|-----------------|---------------------|-------------------------------------------------------|
| `src`           | `ASTSource`/`IRSource` | The original source object.                          |
| `metadata_group`| `dict[str, str]`    | Mapping of `{filename: file_path}` for all cached artifacts. |
| `hash`          | `str`               | The cache key hash.                                    |

### Constructor Behavior

1. Reads the metadata JSON file from the cache.
2. Restores the `target` field from a plain dict back to a `GPUTarget` namedtuple.
3. Creates a `KernelMetadata` namedtuple from the sorted metadata keys.
4. Creates the backend via `make_backend(self.metadata.target)`.
5. Packs metadata via `backend.pack_metadata(self.metadata)` for efficient runtime use.
6. Reads all assembly files from the cache into an `AsmDict`, reading binary files as bytes and text files as strings.
7. Stores the GPU binary (`self.kernel = self.asm[binary_ext]`).
8. Initializes `self.module`, `self.function`, `self._run` to `None` (lazy initialization).

### Instance Attributes

| Attribute          | Type                  | Description                                                |
|-------------------|-----------------------|------------------------------------------------------------|
| `metadata`        | `KernelMetadata`      | Namedtuple with all compilation metadata fields.            |
| `packed_metadata`  | varies                | Backend-packed metadata for efficient launch.               |
| `src`             | `ASTSource`/`IRSource`| The original source object.                                 |
| `hash`            | `str`                 | Cache key hash.                                             |
| `name`            | `str`                 | Kernel function name (from metadata).                       |
| `asm`             | `AsmDict`             | Dictionary of `{ext: contents}` for all IR stages.          |
| `metadata_group`  | `dict`                | The raw metadata group from cache.                          |
| `kernel`          | `bytes`               | The GPU binary (cubin or hsaco).                            |
| `module`          | runtime module        | GPU driver module handle (initialized lazily).              |
| `function`        | runtime function      | GPU driver function handle (initialized lazily).            |
| `n_regs`          | `int`                 | Number of registers used (set during loading).              |
| `n_spills`        | `int`                 | Number of register spills (set during loading).             |
| `n_max_threads`   | `int`                 | Maximum threads per block (set during loading).             |
| `_run`            | callable or `None`    | The launcher function (initialized lazily).                 |

### Methods and Properties

#### `__del__()`

Destructor that unloads the GPU module via `driver.active.utils.unload_module(self.module)`. Also calls `knobs.runtime.kernel_unload_hook` if set.

#### `_init_handles()`

```python
def _init_handles(self):
```

Lazily initializes the GPU binary loading. This is called the first time the kernel is launched. Steps:

1. **Short-circuit**: If `self.module` is not `None`, the kernel is already loaded; return immediately.
2. **Error handling**: Sets up a closure-based error capture so that if initialization fails after this point, subsequent launch attempts will re-raise the same error.
3. **Device and launcher**: Gets the current device and creates the launcher via `driver.active.launcher_cls(self.src, self.metadata)`.
4. **Resource checks**:
   - **Shared memory**: Raises `OutOfResources` if `metadata.shared > max_shared_mem(device)`.
   - **Tensor memory** (Blackwell+): Raises `OutOfResources` if `metadata.tmem_size > 512`.
   - **Thread count**: Raises `OutOfResources` if `metadata.num_warps * warp_size > n_max_threads`.
5. **Binary loading**: Calls `driver.active.utils.load_binary(name, kernel, shared, device)` which returns `(module, function, n_regs, n_spills, n_max_threads)`.
6. **Hooks**: Calls `kernel_load_start_hook` and `kernel_load_end_hook` if configured.

#### `run` (property)

```python
@property
def run(self):
```

Returns the launcher function. Calls `_init_handles()` if not yet initialized.

#### `launch_metadata(grid, stream, *args)`

```python
def launch_metadata(self, grid, stream, *args):
```

Creates a `LazyDict` with base launch metadata (`name`, `function`, `stream`). If the source is an `ASTSource` with a `launch_metadata` callback, adds that as a lazy extra. Returns `None` if no launch hooks are configured.

#### `__getitem__(grid)`

```python
def __getitem__(self, grid):
```

The primary kernel launch interface. `grid` is a 3-tuple `(x, y, z)`. Returns a callable `runner(*args, stream=None)` that:

1. Ensures handles are initialized via `_init_handles()`.
2. Gets the current stream if none provided.
3. Computes launch metadata.
4. Calls `self.run(grid_x, grid_y, grid_z, stream, function, packed_metadata, launch_metadata, hooks, *args)`.

Usage:
```python
compiled_kernel[grid_x, grid_y, grid_z](arg1, arg2, stream=stream)
```

---

## 6. LazyDict and AsmDict

### LazyDict

**Location**: `python/triton/compiler/compiler.py`

A dictionary wrapper that supports lazy evaluation of additional key-value pairs.

```python
class LazyDict:
    def __init__(self, data):
        self.data = data
        self.extras = []
```

| Method     | Signature                    | Description                                                    |
|-----------|------------------------------|----------------------------------------------------------------|
| `get()`   | `() -> dict`                 | Applies all pending extras (each is a `(func, args)` pair, merged via `\|` operator) and returns the merged dict. Clears extras after applying. |
| `add()`   | `(func, args) -> None`       | Appends a lazy computation `(func, args)` to the extras list.  |

Use case: In `CompiledKernel.launch_metadata()`, the base metadata is created immediately, but additional metadata (e.g., from `fn.launch_metadata`) is computed lazily when `get()` is called.

### AsmDict

**Location**: `python/triton/compiler/compiler.py`

A `dict` subclass that lazily computes derived assembly artifacts.

```python
class AsmDict(dict):
    def __missing__(self, key):
```

Currently supports one lazy key:
- `"sass"`: Computes SASS (NVIDIA assembly) disassembly from the `"cubin"` entry using `get_sass()`.

Any other missing key raises a `KeyError`.

---

## 7. CodeGenerator Class

**Location**: `python/triton/compiler/code_generator.py`

The core AST-to-MLIR visitor. Extends `ast.NodeVisitor` to walk the Python AST of a `@triton.jit` kernel and emit Triton MLIR (TTIR) instructions.

### Constructor

```python
class CodeGenerator(ast.NodeVisitor):
    def __init__(self, context, prototype, gscope, function_name, jit_fn,
                 *, options, codegen_fns, module_map, is_gluon,
                 module=None, is_kernel=False, function_types=None,
                 noinline=False, caller_context=None,
                 file_name=None, begin_line=0, begin_col=1):
```

| Parameter         | Type                    | Description                                                     |
|------------------|-------------------------|-----------------------------------------------------------------|
| `context`        | `ir.context`            | MLIR context.                                                   |
| `prototype`      | `ASTFunction`           | Function prototype (arg types, ret types, attrs).               |
| `gscope`         | `dict`                  | Global scope (module-level variables visible to the kernel).    |
| `function_name`  | `str`                   | Name for the generated MLIR function.                           |
| `jit_fn`         | `JITFunction`           | The source JIT function (for source info, params, etc.).        |
| `options`        | object                  | Compilation options from backend.                               |
| `codegen_fns`    | dict                    | Backend codegen implementation functions.                        |
| `module_map`     | `dict`                  | Module name -> module mapping for interface overrides.          |
| `is_gluon`       | `bool`                  | Whether using the Gluon (experimental) semantic layer.          |
| `module`         | MLIR module or None     | Existing module to use, or creates a new one.                   |
| `is_kernel`      | `bool`                  | Whether this is a top-level kernel (affects naming/visibility). |
| `function_types` | `dict` or None          | Cache of `{fn_name: return_type}` for previously compiled functions. |
| `noinline`       | `bool`                  | Whether the function is marked `noinline`.                      |
| `caller_context` | object or None          | Context from the caller (for inter-function calls).             |
| `file_name`      | `str` or None           | Source file name for debug locations.                            |
| `begin_line`     | `int`                   | Starting line number (0-based adjustment).                      |
| `begin_col`      | `int`                   | Starting column number.                                         |

### Key Internal State

| Attribute              | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| `builder`             | `ir.builder` (or `gluon_ir.GluonOpBuilder` for Gluon). Constructs MLIR ops. |
| `semantic`            | `TritonSemantic` (or `GluonSemantic`). High-level operation semantics.       |
| `module`              | The MLIR module being built.                                                 |
| `fn`                  | The current MLIR function being generated. `None` until `visit_FunctionDef`. |
| `lscope`              | Local scope: `{name: value}` for current block.                              |
| `gscope`              | Global scope: `{name: value}` for module-level names.                        |
| `local_defs`          | `{name: value}` for SSA definitions in the current block.                    |
| `scf_stack`           | Stack of structured control flow nodes (for/while) currently being visited.  |
| `return_vals`         | List of return values encountered during code generation.                    |
| `return_ips`          | Corresponding insertion points for each return value.                        |
| `ret_type`            | The inferred return type (set by `handle_returns()`).                        |
| `cur_node`            | The current AST node being visited (for error reporting).                    |
| `prototype`           | The `ASTFunction` prototype being built.                                     |
| `function_ret_types`  | `{fn_name: ret_type}` cache for non-kernel function return types.            |
| `name_loc_as_prefix`  | Optional name prefix for debug locations.                                    |

### Built-in Namespace

The `CodeGenerator` provides these built-in names accessible within kernels:

```python
builtin_namespace = {
    'len': len,
    'list': list,
    'range': range,
    'float': float,
    'int': int,
    'isinstance': isinstance,
    'getattr': getattr,
    'hasattr': hasattr,
    'print': language.core.device_print,
    'min': language.core.builtin_min,
    'max': language.core.builtin_max,
}
```

### visit_* Methods -- Complete Reference

The following documents every `visit_*` method, describing the Python AST node type it handles and how it converts to MLIR.

---

#### `visit_Module(node)`

**AST Node**: `ast.Module`

Simply calls `generic_visit` to process the module body. The top-level module node is typically not visited directly; instead `visit_FunctionDef` is the entry point.

---

#### `visit_FunctionDef(node)`

**AST Node**: `ast.FunctionDef`

This is the main entry point for kernel compilation. Steps:

1. **Nesting check**: Raises `UnsupportedLanguageConstruct` if a function is already being defined (nested definitions are not allowed inside kernels).

2. **Argument processing**: Calls `visit(node.args)` to get `arg_names` and `kwarg_names`.

3. **Default values**: For each argument with a default value, creates an assignment node and visits it. The `visiting_arg_default_value` flag is set to allow access to global variables during default value evaluation.

4. **Function creation**:
   - Sets visibility to `"public"` for kernels, `"private"` for helper functions.
   - Serializes the prototype to get the MLIR function type.
   - Calls `builder.get_or_insert_function(module, name, fn_ty, visibility, noinline)`.
   - Adds the function to the module via `module.push_back(fn)`.

5. **Entry block**: Creates an entry block and deserializes arguments using `prototype.deserialize(fn)`, which maps MLIR block arguments back to frontend values (including tuple destructuring and attribute setting).

6. **Body visit**: Visits the function body statements via `visit_compound_statement()`.

7. **Return handling**: Calls `handle_returns()` to finalize return values and set the function type.

---

#### `visit_arguments(node)`

**AST Node**: `ast.arguments`

Visits each `ast.arg` to collect argument names. Returns `(arg_names, kwarg_name)`.

---

#### `visit_arg(node)`

**AST Node**: `ast.arg`

Validates that the argument is not both `constexpr` and in `do_not_specialize`. Returns the argument name string.

---

#### `visit_Assign(node)`

**AST Node**: `ast.Assign` and `ast.AnnAssign`

Handles variable assignment. Steps:

1. **Value sanitization**: The assigned value is sanitized via `_sanitize_value()`:
   - Tuples have sanitization applied recursively.
   - `constexpr` values are unwrapped.
   - Non-Triton, non-dtype, non-tuple values are converted to tensors via `semantic.to_tensor()`.

2. **Target assignment**: Calls `assignTarget(target, value)` which handles:
   - `ast.Subscript`: Calls `visit_Subscript_Store` (raises `NotImplementedError`).
   - `ast.Tuple`: Recursively assigns tuple elements to sub-targets.
   - `ast.Attribute`: Raises `NotImplementedError`.
   - `ast.Name`: Calls `set_value(name, value)`.

---

#### `visit_AnnAssign(node)`

**AST Node**: `ast.AnnAssign`

Handles annotated assignment (`x: type = value`). Special case: if the annotation is `constexpr`, wraps the value as a `constexpr` and stores it in `lscope` directly (not via `set_value`, so it is not tracked as an SSA definition). Otherwise delegates to `visit_Assign`.

---

#### `visit_AugAssign(node)`

**AST Node**: `ast.AugAssign`

Handles augmented assignment (`x += 1`). Transforms into an equivalent `Assign` node with a `BinOp` on the right-hand side, then visits the constructed assign. Also visits the LHS to return the updated value.

---

#### `visit_Name(node)`

**AST Node**: `ast.Name`

- If `node.ctx` is `ast.Store`, returns the name string `node.id`.
- Otherwise, looks up the name via `self.dereference_name(node.id)` which searches: local scope -> global scope -> builtin namespace. Raises `NameError` if not found.

Global scope lookup enforces constexpr-only access (unless the value is a module, JIT callable, Triton builtin, etc., or `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` is set).

---

#### `visit_Store(node)` / `visit_Load(node)`

**AST Node**: `ast.Store` / `ast.Load`

Context nodes. Calls `generic_visit` (no-op for these).

---

#### `visit_Tuple(node)`

**AST Node**: `ast.Tuple`

Visits all elements and returns a `language.tuple(values)`.

---

#### `visit_List(node)`

**AST Node**: `ast.List`

Visits all elements and returns a `language.tuple(values)` (Triton uses tuples internally for list-like collections).

---

#### `visit_ListComp(node)`

**AST Node**: `ast.ListComp`

Handles list comprehensions. Currently restricted to:
- Exactly one generator.
- The iterable must be a `tl_tuple`.

Iterates over the tuple, sets the target variable for each element, and visits the element expression.

---

#### `visit_Constant(node)`

**AST Node**: `ast.Constant`

Returns `constexpr(node.value)`. All Python constants become Triton constexprs.

---

#### `visit_BinOp(node)`

**AST Node**: `ast.BinOp`

Dispatches binary operations to the appropriate dunder method. The mapping from AST operators to method names:

| AST Operator    | Method Name       |
|----------------|-------------------|
| `ast.Add`      | `__add__`         |
| `ast.Sub`      | `__sub__`         |
| `ast.Mult`     | `__mul__`         |
| `ast.Div`      | `__truediv__`     |
| `ast.FloorDiv` | `__floordiv__`    |
| `ast.Mod`      | `__mod__`         |
| `ast.Pow`      | `__pow__`         |
| `ast.LShift`   | `__lshift__`      |
| `ast.RShift`   | `__rshift__`      |
| `ast.BitAnd`   | `__and__`         |
| `ast.BitOr`    | `__or__`          |
| `ast.BitXor`   | `__xor__`         |

Via `_apply_binary_method()`:
- If LHS is a Triton tensor: calls `lhs.method_name(rhs, _semantic=...)`.
- If RHS is a Triton tensor: calls `rhs.__r{method}__(lhs, _semantic=...)`.
- If both are constexpr: resolves via constexpr arithmetic.

---

#### `visit_Compare(node)`

**AST Node**: `ast.Compare`

Only single comparisons are supported (no chained comparisons like `a < b < c`). Special handling for `is` and `is not` (constexpr comparison). Otherwise dispatches to dunder methods:

| AST Comparator | Method Name |
|---------------|-------------|
| `ast.Eq`      | `__eq__`    |
| `ast.NotEq`   | `__ne__`    |
| `ast.Lt`      | `__lt__`    |
| `ast.LtE`     | `__le__`    |
| `ast.Gt`      | `__gt__`    |
| `ast.GtE`     | `__ge__`    |

---

#### `visit_UnaryOp(node)`

**AST Node**: `ast.UnaryOp`

Dispatches to the corresponding unary method:

| AST Operator  | Method Name   |
|--------------|---------------|
| `ast.USub`   | `__neg__`     |
| `ast.UAdd`   | `__pos__`     |
| `ast.Not`    | `__not__`     |
| `ast.Invert` | `__invert__`  |

For Triton tensors: calls `operand.method_name(_semantic=...)`. For constexpr: calls the method directly. Special case: `__not__` on non-tensor types falls back to `constexpr(not operand)`.

---

#### `visit_BoolOp(node)`

**AST Node**: `ast.BoolOp` (`and` / `or`)

Handles boolean `and` and `or` with short-circuit evaluation:

| AST Operator | Method Name     |
|-------------|-----------------|
| `ast.And`   | `logical_and`   |
| `ast.Or`    | `logical_or`    |

Short-circuit behavior:
- For constexpr values: evaluates immediately and short-circuits if the result is determined.
- For tensor values: accumulates non-trivial (tensor) values and combines them pairwise using `logical_and` or `logical_or`.
- Emits a deprecation warning when used with non-scalar tensors.

---

#### `visit_If(node)`

**AST Node**: `ast.If`

Two compilation paths depending on the condition type:

**Dynamic condition (Triton tensor)**:
- Condition must be a scalar tensor. Non-scalar conditions raise an error (with deprecation warning suggesting `.item()`).
- Condition is cast to `int1` (boolean).
- **With `return` in body**: Uses `visit_if_top_level()` which generates basic blocks with conditional branches (unstructured control flow).
- **Without `return`**: Uses `visit_if_scf()` which generates a structured `scf.if` operation with yield for modified variables.

**Static condition (constexpr/Python bool)**:
- The condition type must be `bool`, `int`, or `NoneType`.
- Only the active branch is compiled; the dead branch is completely omitted.

---

#### `visit_IfExp(node)`

**AST Node**: `ast.IfExp` (ternary: `a if cond else b`)

Similar dual-path logic as `visit_If`:

**Dynamic condition**: Creates an `scf.if` operation with yield for both branches. Both `then` and `else` values must have the same type. Returns a tensor wrapping the `scf.if` result.

**Static condition**: Evaluates the condition and visits only the active branch.

---

#### `visit_While(node)`

**AST Node**: `ast.While`

Generates a structured `scf.while` operation. Steps:

1. **Dry run**: Enters a sub-region and dry-visits the loop body to find loop-carried variables (variables that change value across iterations).

2. **Carried variable detection**: Compares `liveins` (values before loop) with values after the dry run. Only Triton values whose IR handles differ are considered carried.

3. **While operation creation**: Creates `scf.while_op` with carried value types and initial handles.

4. **Before (condition) region**: Sets up block arguments from carried values, visits the condition expression, creates `scf.condition_op`.

5. **After (body) region**: Sets up block arguments, visits the loop body, creates `scf.yield_op` with updated carried values.

6. **Result update**: Updates `lscope` and `local_defs` with the while operation results.

Support for `condition` objects with `disable_licm`: Sets LLVM loop annotation attribute.

---

#### `visit_For(node)`

**AST Node**: `ast.For`

Only `range` and `static_range` iterators are supported.

**`static_range`**: Fully unrolled at compile time. Each iteration is visited sequentially with constexpr loop variable.

**`range`**: Generates a structured `scf.for` operation. Steps:

1. **Iterator arguments**: Extracts `lb` (lower bound), `ub` (upper bound), `step`, plus optional `num_stages`, `loop_unroll_factor`, `disallow_acc_multi_buffer`, `flatten`, `warp_specialize`, `disable_licm`.

2. **Negative step handling**: If step is a negative constexpr, negates it and swaps `lb`/`ub`. The induction variable is computed as `ub - iv + lb` inside the loop.

3. **Type promotion**: All bounds and step are cast to a common integer type via `integer_promote_impl`.

4. **Dry run**: Same as `visit_While` -- dry-visits the body to find loop-carried variables (excluding the induction variable).

5. **ForOp creation**: Creates `scf.for` with lb, ub, step, and carried values. Sets optional attributes (`tt.num_stages`, `tt.loop_unroll_factor`, `tt.disallow_acc_multi_buffer`, `tt.flatten`, `tt.warp_specialize`, `llvm.loop_annotation`).

6. **Body generation**: Visits loop body, creates yield for carried values.

7. **Induction variable**: Creates a poison placeholder before the loop, then replaces all uses with the actual `scf.for` induction variable.

8. **Result update**: Updates carried variables with `scf.for` results.

---

#### `visit_Return(node)`

**AST Node**: `ast.Return`

Only valid in non-kernel (helper) functions. Steps:

1. Visits the return value (defaults to `constexpr(None)`).
2. Appends the value to `self.return_vals`.
3. Records the current insertion point and location in `self.return_ips`.
4. Creates a dead basic block after the return (since return terminates the block).

---

#### `visit_Call(node)`

**AST Node**: `ast.Call`

Dispatches function calls based on the function type:

1. **Static implementation check**: If the function is in `statically_implemented_functions` (e.g., `static_assert`, `static_print`, `int`, `len`), the static implementation is called directly.

2. **JITFunction**: Calls `call_JitFunction()` which:
   - Binds and normalizes arguments.
   - Mangles the function name based on argument types.
   - Creates a new `CodeGenerator` for the callee if not already compiled.
   - Calls the function via `builder.call()`.

3. **Triton builtin / method on tensor**: Calls the function with `_semantic` (and optionally `_generator`) injected as keyword arguments.

4. **ConstexprFunction**: Calls with `_semantic` injected.

5. **Builtin namespace functions**: Unwraps constexpr arguments and calls the Python function directly.

Supports `*args` unpacking via `ast.Starred` nodes.

---

#### `visit_Attribute(node)`

**AST Node**: `ast.Attribute`

Visits the LHS (value), follows module_map for module substitutions, then calls `get_Attribute()`:

- **`.T`** on a Triton tensor: Returns `semantic.permute(lhs, (1, 0))` (transpose).
- **`.value`** on constexpr: Special backward-compatible case -- accesses the constexpr directly.
- **JITFunction** on a Triton value: Returns `BoundJITMethod(lhs, attr)` for method calls.
- Otherwise: Returns `getattr(lhs, attr)`.

---

#### `visit_Subscript(node)`

**AST Node**: `ast.Subscript`

Load context only (`__getitem__`). Visits the value and slice, then:
- For Triton values: Calls `lhs.__getitem__(slices, _semantic=...)`.
- For Python values: Returns `lhs[slices]`.

---

#### `visit_Slice(node)`

**AST Node**: `ast.Slice`

Visits lower, upper, and step, returns a `language.slice(lower, upper, step)`.

---

#### `visit_Index(node)`

**AST Node**: `ast.Index`

Simply visits and returns the inner value node.

---

#### `visit_ExtSlice(node)`

**AST Node**: `ast.ExtSlice`

Visits each dimension and returns a list of dimension values.

---

#### `visit_With(node)`

**AST Node**: `ast.With`

Supports context managers in Triton kernels (e.g., `tl.range` with specialized semantics):

1. Instantiates each context manager by visiting the call expression with `_semantic` injection.
2. Calls `__enter__()` for each, binding the result to `optional_vars`.
3. Visits the body.
4. Calls `__exit__(None, None, None)` for each manager in reverse order.
5. Raises `UnsupportedLanguageConstruct` if `return` is found inside the `with` body.

---

#### `visit_Pass(node)`

**AST Node**: `ast.Pass`

No-op.

---

#### `visit_Assert(node)`

**AST Node**: `ast.Assert`

Calls `language.core.device_assert(test, msg, _semantic=...)` which generates a device-side assertion operation.

---

#### `visit_Expr(node)`

**AST Node**: `ast.Expr`

Marks the expression value as unused (`node.value._is_unused = True`) and visits it. This enables "must-use result" checking for functions annotated with `_must_use_result`.

---

#### `visit_NoneType(node)`

**AST Node**: `NoneType`

Returns `None`.

---

#### `visit_JoinedStr(node)`

**AST Node**: `ast.JoinedStr` (f-strings)

Handles f-strings at compile time. All interpolated values must be constexpr. `ast.Constant` parts are stringified directly. `ast.FormattedValue` parts are evaluated and formatted using the specified conversion code.

---

#### `visit_keyword(node)`

**AST Node**: `ast.keyword`

Returns `(node.arg, self.visit(node.value))` -- a key-value pair for keyword arguments.

---

#### `generic_visit(node)`

**Default fallback** for any unhandled AST node type. Raises `UnsupportedLanguageConstruct` with the node type name.

---

### visit() Override

The base `visit()` method is overridden to provide:

1. **Location tracking**: Before visiting each node with `lineno`/`col_offset`, sets the MLIR builder location. Line numbers are adjusted by `begin_line` and `begin_col`.

2. **Name location prefix**: If `name_loc_as_prefix` is set, creates a named location wrapping the real location.

3. **Error wrapping**: Any non-`CompilationError` exception is wrapped in a `CompilationError` with the source and current node info.

4. **Location restoration**: After visiting, the builder location is restored to the previous value.

---

### Helper Methods

#### `set_value(name, value)`

Records a local definition: `lscope[name] = value` and `local_defs[name] = value`.

#### `_find_carries(node, liveins, ignore=set())`

Dry-runs the loop body to find loop-carried variables. Returns `(names, init_handles, init_fe_tys)`.

#### `_verify_loop_carried_variable(name, loop_val, live_val)`

Asserts that a loop-carried variable maintains consistent type across iterations.

#### `visit_then_else_blocks(node, liveins, then_block, else_block)`

Shared helper for `visit_if_top_level` and `visit_if_scf`. Visits both branches, collects redefined variables, and checks type consistency.

#### `visit_if_top_level(cond, node)`

Generates unstructured control flow (basic blocks with branches) for `if` statements containing `return`.

#### `visit_if_scf(cond, node)`

Generates structured `scf.if` with yield for `if` statements without `return`.

#### `decide_return_type()`

Analyzes all `return_vals` to determine a common return type. Handles type promotion between constexpr and tensor types.

#### `handle_returns()`

Finalizes all return statements:
1. Determines common return type.
2. For each return, casts the value and emits `builder.ret()`.
3. Updates the function type with the inferred return types.
4. Emits a final poison-value return for the function's main block.

#### `call_JitFunction(fn, args, kwargs, caller_context)`

Compiles and calls a nested `@triton.jit` function:
1. Normalizes arguments (wraps non-Triton values as constexpr).
2. Mangles the function name based on argument types.
3. Compiles the callee with a new `CodeGenerator` if not cached.
4. Emits `builder.call()` with the flattened argument handles.

#### `call_Function(node, fn, args, kws)`

Central dispatch for function calls. Routes to the appropriate handler based on function type (JITFunction, builtin, constexpr function, etc.).

#### `call_Method(node, fn, fn_self, args, kws)`

Handles method calls on Triton values. Prepends `fn_self` to the argument list and delegates to `call_Function`.

#### `execute_static_assert(node)`

Implements `static_assert()`: evaluates the condition at compile time and raises `CompileTimeAssertionFailure` if false.

#### `static_executor(python_fn)` (decorator)

Creates a static implementation that evaluates a Python function at compile time with constexpr arguments.

---

### Statically Implemented Functions

```python
statically_implemented_functions = {
    language.core.static_assert: execute_static_assert,
    language.core.static_print: static_executor(print),
    ttgl.static_assert: execute_static_assert,
    ttgl.static_print: static_executor(print),
    int: static_executor(int),
    len: static_executor(len),
}
```

These are executed at compile time with constexpr-unwrapped arguments.

---

### `enter_sub_region` Context Manager

```python
class enter_sub_region:
    def __init__(self, generator):
```

Saves and restores the code generator's state when entering a sub-region (for structured control flow):

- Saves `lscope` and `local_defs`.
- Saves the current insertion block and point.
- On exit: restores insertion point and scopes.

---

### `ContainsReturnChecker` Visitor

An `ast.NodeVisitor` that checks whether an AST subtree contains a `return` statement. Used by `visit_If` and `visit_With` to decide between structured (`scf.if`) and unstructured (basic block) control flow. Traverses function calls by checking the function body only if the function is resolved in the global scope.

---

### `ast_to_ttir()` Function

**Location**: `python/triton/compiler/code_generator.py`

```python
def ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None):
```

Top-level function that drives AST-to-TTIR compilation:

1. **Type construction**: Builds `arg_types` list from the signature and constexpr values in `src.constants`.
2. **Prototype creation**: Creates `ASTFunction([], arg_types, src.attrs)`.
3. **CodeGenerator instantiation**: Creates a `CodeGenerator` with kernel settings (`is_kernel=True`).
4. **Visiting**: Calls `generator.visit(fn.parse())` to walk the AST.
5. **Module verification**: Calls `module.verify()`. If verification fails, prints the module and raises `RuntimeError`.
6. **Return**: Returns the MLIR module.

---

## 8. ASTFunction Class

**Location**: `python/triton/compiler/code_generator.py`

Represents a function prototype with argument types, return types, and attributes.

### Constructor

```python
class ASTFunction:
    def __init__(self, ret_types, arg_types, attrs):
```

| Parameter   | Type       | Description                                       |
|------------|------------|---------------------------------------------------|
| `ret_types`| `list`     | Return types (empty `[]` at start, filled by `handle_returns`). |
| `arg_types`| `list`     | Argument types (Triton type objects or `None` for constexpr). |
| `attrs`    | `dict`     | Attributes per argument path (e.g., divisibility). |

### Methods

#### `flatten_ir_types(builder, types)`

```python
def flatten_ir_types(self, builder, types) -> List[ir.type]:
```

Recursively flattens Triton types (including tuples) into a flat list of MLIR IR types, skipping `None` entries.

#### `return_types_ir(builder)`

```python
def return_types_ir(self, builder) -> List[ir.type]:
```

Returns the flattened list of IR types for the return values.

#### `serialize(builder)`

```python
def serialize(self, builder):
```

Builds an MLIR function type from the flattened argument and return IR types:
```python
arg_types_ir = self.flatten_ir_types(builder, self.arg_types)
ret_types_ir = self.return_types_ir(builder)
return builder.get_function_ty(arg_types_ir, ret_types_ir)
```

#### `deserialize(fn)`

```python
def deserialize(self, fn):
```

Reconstructs frontend values from an MLIR function's block arguments:
1. Creates a template structure matching `self.arg_types` with `constexpr(None)` placeholders.
2. Iterates over types using `apply_with_path`, which provides a path-based traversal.
3. For each type: sets argument attributes on the function, and calls `ty._unflatten_ir()` to reconstruct the frontend value from IR handles.
4. Returns the reconstructed argument values.

---

## 9. Error Handling

**Location**: `python/triton/compiler/errors.py`

All compiler errors inherit from `TritonError` (defined in `python/triton/errors.py`).

### CompilationError

```python
class CompilationError(TritonError):
    source_line_count_max_in_message = 12
```

Base class for all errors raised during compilation.

**Constructor**:
```python
def __init__(self, src: Optional[str], node: ast.AST, error_message: Optional[str] = None):
```

| Parameter       | Type             | Description                                              |
|----------------|------------------|----------------------------------------------------------|
| `src`          | `str` or `None`  | The source code of the kernel function.                  |
| `node`         | `ast.AST`        | The AST node where the error occurred.                   |
| `error_message`| `str` or `None`  | Optional additional message.                             |

**Message formatting** (`_format_message()`):
- If `src` is `None`: shows `<source unavailable>`.
- If `node` has `lineno`: extracts source lines up to the error line (up to `source_line_count_max_in_message` = 12 lines), appends a `^` caret at the column offset.
- Appends `error_message` if provided.

**Pickling support**: `__reduce__` returns `(type(self), (src, node, error_message))` to enable pickling.

### CompileTimeAssertionFailure

```python
class CompileTimeAssertionFailure(CompilationError):
```

Raised when `static_assert()` fails at compile time. Inherits all formatting from `CompilationError`.

### UnsupportedLanguageConstruct

```python
class UnsupportedLanguageConstruct(CompilationError):
```

Raised when the code generator encounters a Python construct it cannot translate to MLIR (e.g., `global` statements, `try/except`, `class` definitions, nested function definitions, etc.).

### Traceback Filtering

The `filter_traceback(e)` function (in `compiler.py`) removes frames from `code_generator.py` and `ast.py` from exception tracebacks, showing only user code. This is controlled by `knobs.compilation.front_end_debugging` (when True, filtering is disabled to aid debugging).

---

## 10. Cache System

The compilation cache avoids recompiling identical kernels. It is managed by three types of cache managers.

### Cache Key Generation

```python
def get_cache_key(src, backend, backend_options, env_vars):
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(sorted(env_vars.items()))}"
    return key
```

The key incorporates:
- **`triton_key()`**: Hash of the entire Triton source code (frontend, compiler, backends, language, C++ library).
- **`src.hash()`**: Hash of the specific source (function + signature + constants for ASTSource; file contents for IRSource).
- **`backend.hash()`**: Backend-specific hash.
- **`backend_options.hash()`**: Options hash.
- **`env_vars`**: Cache-invalidating environment variables.

The final cache hash is `SHA-256(key)` encoded as hex, then Base32-encoded for filesystem safety.

### Cache Managers

All created via factory functions in `python/triton/runtime/cache.py`:

| Function               | Purpose                                                      |
|------------------------|--------------------------------------------------------------|
| `get_cache_manager(key)` | Primary cache for storing compilation artifacts.            |
| `get_override_manager(key)` | Override cache for kernel overriding (via `TRITON_KERNEL_OVERRIDE`). |
| `get_dump_manager(key)`  | Dump cache for writing IR to disk for inspection (via `TRITON_DUMP_IR`). |

### Cache Flow in compile()

1. **Lookup**: `fn_cache_manager.get_group(metadata_filename)` checks for existing metadata.
2. **Hit**: If found and `TRITON_ALWAYS_COMPILE` is not set, creates `CompiledKernel` from cached artifacts.
3. **Miss**: Runs full compilation, stores each stage's output via `fn_cache_manager.put(module, filename)`, then stores metadata via `fn_cache_manager.put_group()`.

### Override System

When `TRITON_KERNEL_OVERRIDE` is enabled:
- At each compilation stage, the override manager is checked for a replacement file.
- If found, the replacement is loaded via `parse(full_name, ext, context)` instead of using the compiled output.
- This allows developers to hand-edit IR at any stage and re-test without modifying the source.

### Inspection Hook

`knobs.runtime.add_stages_inspection_hook` allows injecting additional data into the cache key for custom invalidation logic.

---

## 11. CompileTimer

**Location**: `python/triton/compiler/compiler.py`

Tracks timing metrics for the compilation pipeline.

### Constructor

```python
class CompileTimer:
    def __init__(self):
        self.start = time.time()
        self.ir_initialization_end = None
        self.lowering_stage_ends = []
        self.store_results_end = None
```

### Methods

| Method                        | Description                                           |
|------------------------------|-------------------------------------------------------|
| `finished_ir_initialization()` | Records end of IR initialization phase.              |
| `stage_finished(stage_name)` | Records end of a backend lowering stage.              |
| `end() -> CompileTimes`      | Finalizes timing and returns a `CompileTimes` object. |

### CompileTimes Dataclass

**Location**: `python/triton/knobs.py`

```python
@dataclass(frozen=True)
class CompileTimes:
    ir_initialization: int           # microseconds
    lowering_stages: list[tuple[str, int]]  # [(stage_name, microseconds)]
    store_results: int               # microseconds

    @property
    def total_lowering(self) -> int: ...
    @property
    def total(self) -> int: ...
```

### Compilation Listener Protocol

```python
class CompilationListener(Protocol):
    def __call__(self, *, src, metadata, metadata_group,
                 times: CompileTimes, cache_hit: bool) -> None:
```

Registered via `knobs.compilation.listener`. Called after each compilation (both cache hits and misses) with full timing and metadata information.

---

## 12. Backend Interface

The compiler delegates backend-specific work to a `BaseBackend` subclass.

### GPUTarget

```python
@dataclass(frozen=True)
class GPUTarget:
    backend: str          # e.g., "cuda", "hip"
    arch: Union[int, str] # e.g., 90, "gfx940"
    warp_size: int        # e.g., 32
```

### Language Enum

```python
class Language(Enum):
    TRITON = 0
    GLUON = 1
```

### BaseBackend Abstract Methods

| Method                       | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| `supports_target(target)`    | Returns True if this backend can compile for the given target.    |
| `hash()`                     | Returns a unique identifier for the backend version.              |
| `parse_options(options)`     | Validates and converts options dict to options object.            |
| `add_stages(stages, options, language)` | Populates `stages` dict with `{ext: compile_fn}` entries. |
| `load_dialects(context)`     | Loads backend-specific MLIR dialects.                             |
| `get_module_map()`           | Returns module name -> implementation module mapping.             |
| `get_codegen_implementation(options)` | Returns codegen function dict.                               |
| `pack_metadata(metadata)`    | Packs metadata into a runtime-efficient form.                     |
| `binary_ext` (property)     | Returns the final binary extension (e.g., `"cubin"`, `"hsaco"`).  |

### make_backend(target)

```python
def make_backend(target: GPUTarget) -> BaseBackend:
```

Searches all registered backends and returns the one (exactly one must match) that supports the given target. Raises `RuntimeError` if zero or multiple backends match.

---

## 13. Code Examples

### Example 1: Compiling a Kernel from a Python Function

```python
import triton
import triton.language as tl
from triton.compiler import compile, ASTSource

@triton.jit
def add_kernel(
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

# Create the source representation
src = ASTSource(
    fn=add_kernel,
    signature={
        "x_ptr": "*fp32",
        "y_ptr": "*fp32",
        "output_ptr": "*fp32",
        "n_elements": "i32",
    },
    constexprs={"BLOCK_SIZE": 1024},
)

# Compile (target=None uses the current device)
compiled = compile(src, target=None, options={"num_warps": 4})

# Inspect the compiled kernel
print(f"Kernel name: {compiled.name}")
print(f"Kernel hash: {compiled.hash}")
print(f"Shared memory: {compiled.metadata.shared} bytes")
print(f"Available IR stages: {list(compiled.asm.keys())}")

# Launch the kernel
import torch
n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
output = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
compiled[(n + 1023) // 1024, 1, 1](
    x, y, output, n,
    BLOCK_SIZE=1024,
)
```

### Example 2: Compiling from an IR File

```python
from triton.compiler import compile

# Compile a .ttgir file (skip earlier stages)
compiled = compile(
    src="/path/to/kernel.ttgir",
    target=None,
    options={"num_warps": 4},
)

# The pipeline starts from ttgir -> llir -> ptx -> cubin
print(f"Compiled from IR: {compiled.name}")
```

### Example 3: Inspecting Intermediate IR

```python
from triton.compiler import compile, ASTSource
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    tl.store(x_ptr + tl.arange(0, N), x * 2.0)

src = ASTSource(fn=my_kernel, signature={"x_ptr": "*fp32"}, constexprs={"N": 128})
compiled = compile(src, options={"num_warps": 4})

# Access intermediate representations
if "ttir" in compiled.asm:
    print("=== TTIR ===")
    print(compiled.asm["ttir"])

if "ttgir" in compiled.asm:
    print("=== TTGIR ===")
    print(compiled.asm["ttgir"])

if "llir" in compiled.asm:
    print("=== LLIR ===")
    print(compiled.asm["llir"])

if "ptx" in compiled.asm:
    print("=== PTX ===")
    print(compiled.asm["ptx"])

# SASS is lazily computed from cubin
if "cubin" in compiled.asm:
    print("=== SASS ===")
    print(compiled.asm["sass"])
```

### Example 4: Using Compilation Listener for Profiling

```python
import triton
from triton.compiler import compile, ASTSource

# Set up a compilation listener (via knobs)
def my_listener(*, src, metadata, metadata_group, times, cache_hit):
    print(f"Cache hit: {cache_hit}")
    print(f"IR initialization: {times.ir_initialization} us")
    for stage_name, duration in times.lowering_stages:
        print(f"  Stage {stage_name}: {duration} us")
    print(f"Store results: {times.store_results} us")
    print(f"Total: {times.total} us")

triton.knobs.compilation.listener = my_listener

# Now compile -- listener will be called
@triton.jit
def kernel(x_ptr, BLOCK: tl.constexpr):
    pass

src = ASTSource(fn=kernel, signature={"x_ptr": "*fp32"}, constexprs={"BLOCK": 256})
compiled = compile(src)
```

### Example 5: Direct AST-to-TTIR Conversion (Low-Level)

```python
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource, make_backend
from triton.runtime.driver import driver

# Get the current target and backend
target = driver.active.get_current_target()
backend = make_backend(target)
options = backend.parse_options({"num_warps": 4})

# Define the kernel
@triton.jit
def simple_add(x_ptr, y_ptr, n: tl.constexpr):
    offsets = tl.arange(0, n)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(y_ptr + offsets, x + y)

# Create source and context
src = ASTSource(fn=simple_add, signature={"x_ptr": "*fp32", "y_ptr": "*fp32"}, constexprs={"n": 256})
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)

# Convert AST to TTIR directly
codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()
module = ast_to_ttir(src.fn, src, context=context, options=options,
                     codegen_fns=codegen_fns, module_map=module_map)

# Print the TTIR
print(module)
```

### Example 6: Error Handling

```python
from triton.compiler import compile, ASTSource, CompilationError
from triton.compiler.errors import CompileTimeAssertionFailure, UnsupportedLanguageConstruct
import triton
import triton.language as tl

@triton.jit
def kernel_with_assert(x_ptr, N: tl.constexpr):
    tl.static_assert(N > 0, "N must be positive")
    pass

# This will raise CompileTimeAssertionFailure
try:
    src = ASTSource(fn=kernel_with_assert, signature={"x_ptr": "*fp32"}, constexprs={"N": 0})
    compiled = compile(src)
except CompileTimeAssertionFailure as e:
    print(f"Assertion failed: {e}")
except CompilationError as e:
    print(f"Compilation error: {e}")
except UnsupportedLanguageConstruct as e:
    print(f"Unsupported construct: {e}")
```

---

## Appendix A: Compilation Stage Extensions (Typical CUDA Backend)

| Extension | Stage Name                      | Input           | Output          |
|-----------|--------------------------------|-----------------|-----------------|
| `ttir`    | Triton IR                      | Python AST      | TTIR module     |
| `ttgir`   | Triton GPU IR                  | TTIR            | TTGIR           |
| `llir`    | LLVM IR                        | TTGIR           | LLIR text       |
| `ptx`     | PTX assembly                   | LLIR            | PTX text        |
| `cubin`   | CUDA binary                    | PTX             | Binary bytes    |

## Appendix B: Key Environment Variables

| Variable                            | Effect                                                 |
|-------------------------------------|--------------------------------------------------------|
| `TRITON_ALWAYS_COMPILE`             | Force recompilation even on cache hits.                |
| `TRITON_DUMP_IR`                    | Dump all intermediate IR to disk.                      |
| `TRITON_KERNEL_OVERRIDE`            | Enable kernel override from disk.                      |
| `TRITON_STORE_BINARY_ONLY`          | Store only cubin/hsaco and metadata in cache.          |
| `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS`| Allow access to non-constexpr global variables.        |
| `TRITON_FRONT_END_DEBUGGING`        | Disable traceback filtering for debugging.             |
| `TRITON_USE_IR_LOC`                 | Use IR-based source locations for debugging.           |

## Appendix C: Function Name Mangling

Non-kernel (helper) functions are mangled to include argument type information:

```python
def mangle_fn(name, arg_tys, caller_context):
    mangled_args = '_'.join([ty.mangle() for ty in arg_tys])
    mangled_args = mangled_args.replace("'", '_sq_')
    mangled_args = mangled_args.replace('[', '_').replace(']', '_')
    ret = f'{name}__{mangled_args}'
    if caller_context is not None:
        ret += caller_context.mangle()
    return ret
```

This ensures that the same function called with different argument types produces separate MLIR functions.

## Appendix D: Type Representation Conversion

The `convert_type_repr()` function converts MLIR type strings to a simplified signature format:

- `!tt.ptr<type>` becomes `*type`
- `tt.nv_tma_desc = 1` attribute yields `nvTmaDesc`
- Attribute annotations `{...}` are stripped
- Other types pass through unchanged
