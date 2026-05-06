# TileLang Eager Execution Mode Reference

TileLang supports two execution paradigms for JIT-compiled kernels: **lazy mode** and **eager mode**. The eager execution mode enables a more dynamic, Pythonic kernel authoring experience where tensor shapes are inferred at call time rather than specified statically at definition time.

## Table of Contents

1. [Eager Execution Mode Overview](#eager-execution-mode-overview)
2. [Eager Module Architecture](#eager-module-architecture)
3. [Eager AST: ast.py](#eager-ast-astpy)
4. [Eager Builder: builder.py](#eager-builder-builderpy)
5. [Eager Utilities: utils.py](#eager-utilities-utilspy)
6. [JIT Mode Detection](#jit-mode-detection)
7. [When Eager Mode is Triggered](#when-eager-mode-is-triggered)
8. [Differences Between Lazy and Eager Execution](#differences-between-lazy-and-eager-execution)
9. [Tensor Supply in Eager Mode](#tensor-supply-in-eager-mode)
10. [Debugging with Eager Mode](#debugging-with-eager-mode)
11. [Limitations of Eager Mode](#limitations-of-eager-mode)

---

## Eager Execution Mode Overview

Eager mode in TileLang allows kernel functions to accept actual tensor arguments at call time, with the system automatically inferring shapes and constructing the appropriate TIR (Tensor Intermediate Representation). This contrasts with lazy mode, where the kernel function explicitly constructs and returns a `PrimFunc`.

**Key Characteristics:**

- Tensor shapes are derived from runtime arguments rather than being statically encoded
- The function body uses the DSL builder pattern with type annotations
- The system performs a two-phase compilation: (1) shape inference and template creation, (2) shape substitution and final TIR generation
- Constants and symbolic dimensions are declared with `T.const()`

**Module Location:** `tilelang.language.eager`

---

## Eager Module Architecture

The eager module consists of three core files:

```
tilelang/language/eager/
    __init__.py     -- Re-exports from builder and dtypes
    ast.py          -- AST transformation and IR generation infrastructure
    builder.py      -- TIR Builder, JITFunc, TirTemplate, and related classes
    utils.py        -- Utility functions for AST manipulation and compilation
```

The eager mode pipeline involves these components:

```
User Function (Python)
       |
       v
  mutate() [ast.py]   -- AST transformation via DSLMutator
       |
       v
  IRGenerator          -- Wrapped function that accepts a BaseBuilder
       |
       v
  JITFunc              -- Manages lazy/eager dispatch and caching
       |
       v
  [Phase 1] Builder.eager_jit = "phase1"
       |                -- Create constexpr variables, infer tensor types
       v
  TirTemplate           -- Cached PrimFunc template with shape matchers
       |
       v
  [Phase 2] Builder.eager_jit = "phase2"
       |                -- Substitute constexpr vars with actual shapes
       v
  Final PrimFunc        -- Ready for compilation
```

---

## Eager AST: ast.py

The `ast.py` module implements the AST (Abstract Syntax Tree) transformation infrastructure that converts user-written Python functions into IR generators. This is the foundation of TileLang's metaprogramming system.

### BaseBuilder

```python
class BaseBuilder:
    """Base class for both eager and lazy builders."""
```

`BaseBuilder` defines the interface that the transformed AST calls into. Each method corresponds to a Python construct that the DSLMutator intercepts:

| Method | Python Construct | Description |
|--------|------------------|-------------|
| `ctx_if(cond)` | `if cond:` | Context manager for conditional execution |
| `ctx_then(val)` | `if cond: ...` | True branch handler |
| `ctx_else(val)` | `else: ...` | False branch handler |
| `eval(val)` | Expression statement | Evaluate an expression |
| `ctx_for(range)` | `for x in range:` | For loop context |
| `ctx_continue()` | `continue` | Continue statement |
| `ctx_break()` | `break` | Break statement |
| `ctx_while(cond)` | `while cond:` | While loop context |
| `bind(name, value, annot)` | Assignment | Variable binding |
| `assign_slice(lval, sl, value, annot)` | Indexed assignment | Slice assignment |
| `aug_assign(op, target, aug_value, name)` | Augmented assignment (+=, etc.) | In-place update |
| `boolop(op, left, right)` | `and`, `or`, `not` | Boolean operations |
| `ifexp(cond, then, otherwise)` | Ternary expression | Conditional expression |
| `ret(value)` | `return` | Return statement |
| `ctx_with(ctx)` | `with` statement | Context manager |
| `assert_expr(cond, msg)` | `assert` | Assertion |
| `rval(name, value)` | Name reference | Right-value resolution |
| `arg(name, value)` | Function argument | Argument binding |
| `override(name)` | Override resolution | Name override (e.g., `range`) |

### DSLMutator

```python
class DSLMutator(ast.NodeTransformer):
    def __init__(self, nonlocals: dict[str, Any], globals: dict[str, Any], filename: str)
```

The `DSLMutator` is an AST node transformer that rewrites Python functions to call `BaseBuilder` methods instead of executing Python operations directly. It handles:

**Control Flow Transformation:**

- `if/elif/else` -> `__tb.ctx_if()`, `__tb.ctx_then()`, `__tb.ctx_else()`
- `for` loops -> `__tb.ctx_for()` with proper target binding
- `while` loops -> `__tb.ctx_while(lambda: cond)`
- `continue/break` -> `__tb.ctx_continue()`, `__tb.ctx_break()`

**Assignment Transformation:**

- `x = value` -> `x = __tb.bind('x', value)`
- `x += value` -> `x = __tb.aug_assign('Add', x, value, name='x')`
- `x[i] = value` -> `__tb.assign_slice(x, i, value)`
- `x: T.Tensor[...] = value` -> `__tb.bind('x', value, annot=...)`

**Expression Transformation:**

- `a and b` -> `__tb.boolop('And', a, lambda: b)`
- `a or b` -> `__tb.boolop('Or', a, lambda: b)`
- `not a` -> `__tb.boolop('Not', a)`
- `a if cond else b` -> `__tb.ifexp(cond, lambda: a, lambda: b)`
- `a == b` / `a < b` etc. -> `__tb.boolop('And', ...)` for chained comparisons

**Function Transformation:**

Each function is wrapped in a `make_closure` that captures nonlocal variables:

```python
# Before transformation:
def kernel(A, B):
    for i in range(128):
        B[i] = A[i] + 1.0

# After transformation (conceptual):
def make_closure():
    def kernel(__tb):
        A = __tb.arg('A', A)
        B = __tb.arg('B', B)
        range = __tb.override('range')
        for __tmp in __tb.ctx_for(range(128)):
            i = __tb.bind('i', __tmp)
            __tb.assign_slice(B, i, __tb.rval('A', A)[i] + 1.0)
    return kernel
```

### IRGenerator

```python
@dataclass
class IRGenerator(Generic[_P, _T]):
    gen: Callable[[BaseBuilder], Callable[_P, _T]]
    source: str
    extra_type_hints: dict[str, Any] = field(default_factory=dict)
```

Stores the transformed function generator along with its source code. The `gen` field is a function that takes a `BaseBuilder` and returns the actual callable. The `source` field contains the unparsed transformed AST for debugging.

### mutate Function

```python
def mutate(func: Callable[_P, _T]) -> IRGenerator[_P, _T]
```

Transforms a Python function into an `IRGenerator` by:
1. Parsing the function source into an AST
2. Collecting closure variables (nonlocals)
3. Applying `DSLMutator` transformation
4. Compiling the transformed AST into a new function
5. Creating a closure that captures the original nonlocals

### Quote Functions

```python
def quote(expr: str, *, passes=None, span=None, **kws) -> list[ast.AST]
def quote1(expr: str, *, passes=None, span=None, **kws) -> ast.AST
def quote_expr(expr: str, **kws) -> ast.expr
```

Template-based AST construction utilities used by `DSLMutator` to generate replacement AST nodes. These allow embedding Python expressions with placeholder substitutions.

### SpanAttacher

```python
class SpanAttacher(ast.NodeTransformer):
    """Attaches file/line information to each statement for error reporting."""
```

Inserts `__tb.set_fileline(filename, lineno, func_name)` calls before each statement to enable accurate error location reporting.

---

## Eager Builder: builder.py

The `builder.py` module contains the TIR `Builder`, `JITFunc`, `TirTemplate`, and related classes that implement the actual TIR construction.

### Builder

```python
class Builder(BaseBuilder):
    def __init__(self):
        self.frames: list[AnyFrame] = []
        self.ir_builder = IRBuilder()
        self.eager_jit: EagerJITStage = "none"
        self.eager_jit_subs: dict[str, PrimExpr] = {}
        self.func_pass_configs: dict[str, Any] | None = None
        self.func_compile_flags: list[str] | str | None = None
```

The `Builder` extends `BaseBuilder` with TIR-specific operations. It maintains a frame stack for scoping and an `IRBuilder` for constructing TIR statements.

**Eager JIT Stages:**

| Stage | Constant | Description |
|-------|----------|-------------|
| Phase 1 | `"phase1"` | Template creation -- constexpr variables are created, tensor types are inferred |
| Phase 2 | `"phase2"` | Template instantiation -- constexpr variables are substituted with actual values |
| None | `"none"` | Not in eager JIT -- standard lazy compilation |

**Key Methods for Eager Mode:**

- `prim_func(name)` -- Context manager that initializes the Builder and IRBuilder
- `bind(name, value, annot)` -- Handles variable binding with special logic for `PrimExpr`, `Buffer`, `Var`, and `Ref` types
- `skip_kernel_ctx()` -- Returns `True` during Phase 1 to skip kernel execution context
- `constexpr(name, dtype)` -- Creates a `tir.Var` for a constexpr dimension during Phase 1

### JITFunc

```python
@dataclass
class JITFunc(Generic[_P, _T]):
    orig_func: Callable[_P, _T]
    arg_names: list[str]
    tensor_args: dict[str, Buffer | Var]
    tensor_args_defaults: dict[str, Any]
    ir_gen: IRGenerator[_P, _T]
    mode: Literal["auto", "lazy", "eager"] = "auto"
```

The `JITFunc` wraps a user function and handles both lazy and eager execution. It manages the two-phase compilation process and caches templates.

**Mode Detection (`_is_lazy_style`):**

1. If the function contains an internal `@T.prim_func` decorator, it is lazy
2. If calling the function returns a `PrimFunc`, it is lazy
3. If calling raises `JITNoBuilderError` or `EagerJITBuildError`, it is eager

**Phase 1 -- Template Creation:**

```python
def _build_tir_template(self, *args, **kwargs) -> TirTemplate[_P, _T]:
    builder = Builder()
    builder.eager_jit = "phase1"
    with builder.prim_func(self.orig_func.__name__):
        self.ir_gen.gen(builder)(**self.tensor_args, **kwargs)
    pf = builder.get()
    pf.orig_func = self.orig_func
    return TirTemplate.create(
        self.orig_func.__name__, pf,
        builder.constexpr_var, self.ir_gen,
    )
```

**Phase 2 -- Template Instantiation:**

```python
# In TirTemplate.get_tir():
builder = Builder()
builder.eager_jit = "phase2"
builder.eager_jit_subs = subs  # Actual shape values
with builder.prim_func(self.name):
    self.ir_gen.gen(builder)(**tensor_args, **kwargs)
pf = builder.get()
```

### TirTemplate

```python
@dataclass
class TirTemplate(Generic[_P, _T]):
    name: str
    prim_func: PrimFunc[_P, _T]
    matcher: dict[Var, tuple[str, str, int, str]] | None = None
    constexprs: set[Var] = None
    is_lazy_style: bool = False
    ir_gen: IRGenerator[_P, _T] | None = None
```

Represents a cached TIR template with shape variable matchers. The `matcher` maps each constexpr variable to a tuple of `(buffer_name, field, index, var_name)` indicating where the shape value should be extracted from.

**Matcher Construction:**

During `TirTemplate.create()`, the system scans the `buffer_map` of the PrimFunc to find which buffer shapes/strides reference each constexpr variable:

```python
for k, v in prim_func.buffer_map.items():
    for i, s in enumerate(v.shape):
        if s in constexpr and s not in matcher:
            matcher[s] = (k.name, "shape", i, s.name)
    for i, s in enumerate(v.strides):
        if s in constexpr and s not in matcher:
            matcher[s] = (k.name, "stride", i, s.name)
```

**Cache Key Computation:**

```python
def _parse_phase2_key(self, **kwargs):
    # For each constexpr variable, extract its value from:
    # 1. Explicit keyword argument
    # 2. Tensor argument's shape/stride at the matched position
    for k, ty, i, name in self.matcher.values():
        if name in kwargs:
            result.append(kwargs.get(name))
        elif k in kwargs:
            if ty == "shape":
                result.append(kwargs[k].shape[i])
            elif ty == "stride":
                result.append(kwargs[k].stride()[i])
```

### Macro

```python
@dataclass
class Macro(Generic[_P, _T]):
    name: str
    orig_func: Callable[_P, _T]
    ir_gen: IRGenerator[_P, _T]
    annotations: dict[str, Any]
```

Macros are reusable code fragments that can be called within `@T.prim_func` or other macros. They are transformed by `mutate()` and executed within the Builder context to generate TIR inline.

### const Function

```python
def const(name: str, dtype: str = "int32") -> Var | tuple[Var, ...]
```

Declares constexpr variables for dynamic tensor dimensions in eager mode. These variables are resolved during Phase 2 with actual shape values.

**Usage:**

```python
@tilelang.jit
def kernel(A, B):
    M, N = T.const("M, N")  # Declare symbolic dimensions
    A: T.Tensor[[M, N], T.float32]  # Shape uses the constexpr vars
    B: T.Tensor[[M, N], T.float32]
    with T.Kernel(M, N) as (i, j):
        B[i, j] = A[i, j] + 1.0
```

### annotate_compile_flags / annotate_pass_configs

```python
def annotate_compile_flags(flags: list[str] | str) -> None
def annotate_pass_configs(configs: dict[str, Any]) -> None
```

These functions allow embedding compile flags and pass configurations within the function body. They are processed during Phase 2 and attached as PrimFunc attributes.

---

## Eager Utilities: utils.py

The `utils.py` module provides utility functions for AST manipulation, source compilation, and function introspection.

### get_ast

```python
def get_ast(func: Callable) -> ast.AST
```

Parses a function's source code into an AST, preserving line numbers by prepending blank lines to account for the source offset.

### get_func_nonlocals

```python
def get_func_nonlocals(func) -> dict[str, Any]
```

A modified version of `inspect.getclosurevars` that returns a dictionary of nonlocal variable names to their values. This is used to capture closure variables for the `make_closure` wrapper.

### get_compiled_object

```python
def get_compiled_object(
    source: str | ast.AST,
    name: str,
    filename: str = None,
    globals: dict[str, Any] = None,
)
```

Compiles source code (string or AST) into a callable object. Supports two compilation methods:

| Method | When Used | Description |
|--------|-----------|-------------|
| `"direct"` | AST input | Compiles directly using `compile()` |
| `"disk"` | String input | Writes to disk cache for line number support |

### construct_strides

```python
def construct_strides(shape: tuple[Any, ...], allow_prim_expr: bool = True) -> tuple[Any, ...]
```

Computes row-major strides from a tensor shape, used when constructing buffer arguments.

---

## JIT Mode Detection

TileLang supports three JIT modes controlled by the `mode` parameter:

### "auto" Mode (Default)

The system automatically detects whether the function uses lazy or eager style:

1. **Check for internal `@T.prim_func`:** If the function body contains a nested `@T.prim_func` decorator, it is classified as lazy.
2. **Try calling the function:** If calling the function successfully returns a `PrimFunc`, it is lazy.
3. **Catch eager indicators:** If calling raises `JITNoBuilderError` (from `T.const()`, `T.Kernel()` without a builder), it is eager.

### "lazy" Mode

Forces lazy execution. The function must explicitly return a `PrimFunc`:

```python
@tilelang.jit(mode="lazy")
def kernel(M, N, K):
    @T.prim_func
    def main(A: T.Buffer((M, K), "float16"),
             B: T.Buffer((K, N), "float16"),
             C: T.Buffer((M, N), "float16")):
        with T.Kernel(M, N) as (i, j):
            # ... kernel body
    return main
```

### "eager" Mode

Forces eager execution. The function uses type annotations and `T.const()`:

```python
@tilelang.jit(mode="eager")
def kernel(A, B):
    M, N = T.const("M, N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]
    with T.Kernel(M, N) as (i, j):
        B[i, j] = A[i, j] + 1.0
```

---

## When Eager Mode is Triggered

Eager mode is triggered when any of the following conditions are met:

1. **Explicit mode selection:** `@tilelang.jit(mode="eager")`
2. **Use of `T.const()`:** The function calls `T.const()` to declare symbolic dimensions
3. **Use of `T.Kernel()` without returning PrimFunc:** The function uses `T.Kernel()` context but does not return a `PrimFunc`
4. **Type annotation pattern:** The function has type annotations like `A: T.Tensor[[M, N], T.float32]` without an inner `@T.prim_func`

**Example -- Automatic Eager Detection:**

```python
import tilelang

# This will be detected as eager because it uses T.const()
@tilelang.jit
def add_kernel(A, B):
    M, N = T.const("M, N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]
    with T.Kernel(M, N) as (i, j):
        B[i, j] = A[i, j] + 1.0

# Called with actual tensors -- shapes are inferred
import torch
a = torch.randn(128, 64, device="cuda", dtype=torch.float32)
b = torch.randn(128, 64, device="cuda", dtype=torch.float32)
add_kernel(a, b)  # M=128, N=64 inferred from a.shape
```

---

## Differences Between Lazy and Eager Execution

### Syntax Comparison

| Feature | Lazy Mode | Eager Mode |
|---------|-----------|------------|
| Dimension specification | Function parameters (`M, N, K`) | `T.const("M, N, K")` |
| Buffer declaration | `T.Buffer((M, N), dtype)` in `@T.prim_func` | `A: T.Tensor[[M, N], dtype]` |
| Kernel launch | `with T.Kernel(M, N) as ...` | Same |
| Return value | Returns `PrimFunc` | No return needed |
| Inner decorator | `@T.prim_func` required | Not used |

### Lazy Mode Example

```python
@tilelang.jit
def vector_add(M: int, N: int):
    @T.prim_func
    def main(A: T.Buffer((M, N), "float32"),
             B: T.Buffer((M, N), "float32"),
             C: T.Buffer((M, N), "float32")):
        with T.Kernel(M, N) as (i, j):
            C[i, j] = A[i, j] + B[i, j]
    return main

# Must specify dimensions explicitly
program = vector_add(128, 64)
```

### Eager Mode Example

```python
@tilelang.jit
def vector_add(A, B, C):
    M, N = T.const("M, N")
    A: T.Tensor[[M, N], "float32"]
    B: T.Tensor[[M, N], "float32"]
    C: T.Tensor[[M, N], "float32"]
    with T.Kernel(M, N) as (i, j):
        C[i, j] = A[i, j] + B[i, j]

# Dimensions inferred from tensor shapes
import torch
a = torch.randn(128, 64, device="cuda")
b = torch.randn(128, 64, device="cuda")
c = torch.randn(128, 64, device="cuda")
vector_add(a, b, c)
```

### Compilation Flow Comparison

| Aspect | Lazy Mode | Eager Mode |
|--------|-----------|------------|
| Compilation trigger | When dimensions are known | When function is called |
| Template caching | Per-dimension tuple | Per-(non-tensor args, shape tuple) |
| Shape inference | Explicit parameters | Automatic from tensor arguments |
| Code generation | Single pass | Two-phase (template + instantiation) |

---

## Tensor Supply in Eager Mode

In eager mode, tensors are supplied as actual PyTorch tensors at call time. The system extracts shape and stride information from these tensors to instantiate the TIR template.

### Shape Inference

The `TirTemplate._parse_phase2_key()` method resolves constexpr variables from the supplied tensors:

1. If a keyword argument matches the constexpr name directly, use its value
2. If a tensor argument matches the buffer name, extract the shape or stride value at the matched position

```python
# Given: matcher maps M -> ("A", "shape", 0, "M")
# When called with A = torch.randn(128, 64):
#   M is resolved to A.shape[0] = 128
```

### Caching Behavior

The JIT system caches compiled kernels using a two-level key:

1. **Phase 1 key:** Tuple of sorted non-tensor keyword arguments
2. **Phase 2 key:** Tuple of resolved constexpr values (shapes)

```python
# First call with (128, 64) shapes -- compiles and caches
kernel(a_128x64, b_128x64)

# Second call with same shapes -- cache hit
kernel(a2_128x64, b2_128x64)

# Call with different shapes -- new compilation
kernel(a_256x128, b_256x128)
```

### Stride Handling

By default, eager mode assumes dense (contiguous) tensor layout. The system can also extract stride information from the supplied tensors when constexpr variables are declared for strides.

---

## Debugging with Eager Mode

### Source Code Inspection

The `IRGenerator.source` attribute contains the transformed AST source, which shows how the DSLMutator rewrote the original function:

```python
@tilelang.jit
def kernel(A, B):
    M, N = T.const("M, N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]
    with T.Kernel(M, N) as (i, j):
        B[i, j] = A[i, j] + 1.0

# Access the transformed source
print(kernel.ir_gen.source)
```

### AST Print and Layout Visualization

Enable TIR AST printing and layout visualization via pass configs:

```python
kernel = tilelang.compile(
    program,
    pass_configs={
        PassConfigKey.TL_AST_PRINT_ENABLE: True,
        PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
    },
)
```

### IR Dumping

Enable IR dumping between passes for detailed inspection:

```python
kernel = tilelang.compile(
    program,
    pass_configs={
        PassConfigKey.TL_ENABLE_DUMP_IR: True,
        PassConfigKey.TL_DUMP_IR_DIR: "./debug_ir/",
    },
)
```

### Common Error Messages

| Error | Cause | Resolution |
|-------|-------|------------|
| `JITNoBuilderError: T.const() can only be used inside @tilelang.jit` | Called `T.const()` outside JIT context | Use within `@tilelang.jit` decorated function |
| `Only tensor allocated from T.empty can be returned` | Returning non-tensor values from prim_func | Only return tensors created with `T.empty` |
| `Not all tensor from T.empty are returned` | Missing return for allocated tensors | Ensure all `T.empty` tensors are returned |
| `Constexpr variable X is not used in any buffer shape or stride` | Declared constexpr not used in buffer shapes | Use the variable directly in a buffer shape |
| `Cannot find value for constexpr variable X` | Shape cannot be inferred from arguments | Provide the value as a keyword argument |
| `Immutable variable X is used outside its defining region` | Variable scope violation | Restructure code to use variables within proper scope |

---

## Limitations of Eager Mode

### Current Limitations

1. **Direct Shape Usage Required:** Constexpr variables must be used directly (not in expressions) in at least one buffer shape or stride. Indirect usage like `M * 2` in a buffer shape is not supported for automatic inference -- use separate constexpr variables instead.

2. **No Stride Inference by Default:** While strides can be declared as constexpr, the automatic inference currently focuses on shapes. Custom stride handling requires explicit keyword arguments.

3. **Single Function Per JIT:** Each `@tilelang.jit` decorated function produces exactly one kernel. Multiple kernel variants require separate JIT functions.

4. **Type Annotation Restrictions:** Type annotations must follow the `T.Tensor[[dims], dtype]` or `T.dtype` pattern. Complex generic types are not supported.

5. **No Dynamic Control Flow in Phase 1:** During Phase 1 (template creation), the kernel body is not executed -- only type annotations are processed. Dynamic branching based on runtime values is not available during template creation.

6. **Macro Restrictions:** Macros within eager-mode functions cannot return from inside control flow constructs (if/for/while). Use `T.alloc_var()` to create mutable variables before the control flow.

7. **Augmented Assignment on Buffers:** Augmented assignment on buffer objects (e.g., `buffer += value`) raises an error. Use slice assignment instead: `buffer[0] += value`.

### Comparison with Lazy Mode Capabilities

| Feature | Lazy Mode | Eager Mode |
|---------|-----------|------------|
| Dynamic shapes | Manual parameter passing | Automatic inference |
| Multiple kernels | Multiple `@T.prim_func` | Separate JIT functions |
| Complex shape expressions | Arbitrary PrimExpr | Direct constexpr only |
| Return type flexibility | Any PrimFunc | Tensor outputs only |
| Compile-time computation | Full Python execution | Builder-mediated only |
| External function calls | Full support | Limited to TIR operations |

### Best Practices for Eager Mode

1. **Use `T.const()` for all symbolic dimensions** -- this ensures proper shape inference
2. **Keep shape expressions simple** -- use direct constexpr variables in buffer shapes
3. **Test with multiple shapes** -- ensure the generated kernel works for different input dimensions
4. **Use `T.annotate_pass_configs()`** for kernel-specific optimizations
5. **Leverage caching** -- the JIT system caches per-shape; reuse tensors with same shapes for maximum performance
