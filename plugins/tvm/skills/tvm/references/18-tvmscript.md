# TVMScript - Python DSL for TVM IR

TVMScript is TVM's Python-based domain-specific language (DSL) that allows developers to write, inspect, and manipulate TVM IR directly in Python. It serves as both a human-readable representation and a programmable interface for constructing TVM programs.

---

## 18.1 Overview

### 18.1.1 What TVMScript Is

TVMScript is a Python DSL embedded in standard Python syntax. It allows writing TVM IR constructs (IRModule, PrimFunc, Relax functions) using familiar Python class and function definitions. The Python AST is parsed and transformed into TVM IR objects.

**Critical point**: TVMScript code is **NOT executed by the Python interpreter**. Instead, the TVMScript parser captures the Python AST at import/definition time and translates it into TVM IR. This means:
- Python `for` loops become TIR loop nests, not runtime Python loops.
- Python `if/else` becomes TIR conditional statements.
- Python function bodies are parsed into TIR or Relax IR, not executed.

### 18.1.2 Import Conventions

```python
from tvm.script import ir as I       # Module-level constructs
from tvm.script import tirx as T     # TensorIR (TIR) constructs
from tvm.script import relax as R    # Relax IR constructs
```

The aliases `I`, `T`, and `R` are standard conventions used throughout TVM documentation and codebase:
- **`I`**: IRModule-level decorators and utilities (`I.ir_module`, `I.module_attrs`).
- **`T`**: TensorIR primitives (`T.prim_func`, `T.Buffer`, `T.sblock`, `T.axis.spatial`, etc.).
- **`R`**: Relax IR constructs (`R.function`, `R.Tensor`, `R.call_tir`, `R.dataflow`, etc.).

---

## 18.2 Module-Level Constructs

### 18.2.1 @I.ir_module

The `@I.ir_module` decorator transforms a Python class into a TVM IRModule. The class body may contain methods decorated with `@T.prim_func` and `@R.function`.

```python
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class MyModule:
    """An IRModule containing both TIR and Relax functions."""

    @T.prim_func
    def elementwise(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
    ) -> None:
        """A TIR primitive function."""
        for i, j in T.grid(128, 128):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * T.float32(2.0) + T.float32(1.0)

    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        """A Relax function (graph-level)."""
        with R.dataflow():
            lv = R.call_tir(
                MyModule.elementwise,
                (x,),
                out_sinfo=R.Tensor((128, 128), "float32"),
            )
            R.output(lv)
        return lv
```

### 18.2.2 Module Attributes

```python
@I.ir_module
class MyModule:
    I.module_attrs({"attr_key": "attr_value", "target": "cuda"})

    @T.prim_func
    def my_func(A: T.Buffer((64,), "float32")) -> None:
        for i in range(64):
            with T.sblock("A"):
                vi = T.axis.spatial(64, i)
                A[vi] = A[vi] + T.float32(1.0)
```

### 18.2.3 Module Structure Rules

- An `@I.ir_module` class can contain multiple `@T.prim_func` and `@R.function` methods.
- Class-level statements other than function definitions and `I.module_attrs` are not allowed.
- The class itself is replaced by a `tvm.IRModule` object after parsing.
- `__init__` and other Python special methods are not supported.

```python
# Valid: multiple functions in one module
@I.ir_module
class MultiFuncModule:
    @T.prim_func
    def add(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
        for i in range(128):
            with T.sblock("add"):
                vi = T.axis.spatial(128, i)
                B[vi] = B[vi] + A[vi]

    @T.prim_func
    def mul(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
        for i in range(128):
            with T.sblock("mul"):
                vi = T.axis.spatial(128, i)
                B[vi] = B[vi] * A[vi]

    @R.function
    def main(
        x: R.Tensor((128,), "float32"),
        y: R.Tensor((128,), "float32"),
    ) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            lv1 = R.call_tir(MultiFuncModule.add, (x, y), out_sinfo=R.Tensor((128,), "float32"))
            lv2 = R.call_tir(MultiFuncModule.mul, (lv1, x), out_sinfo=R.Tensor((128,), "float32"))
            R.output(lv2)
        return lv2
```

---

## 18.3 TIR Syntax (@T.prim_func)

### 18.3.1 Function Definition

A TIR prim_func defines a low-level tensor computation with explicit loops, buffer accesses, and scheduling annotations.

```python
@T.prim_func
def function_name(
    param1: T.Buffer(shape, dtype),
    param2: T.Buffer(shape, dtype),
) -> T.None:  # or a specific data type for the return value
    """Function docstring."""
    # Function body
```

### 18.3.2 Function Parameters

**Buffer parameters**: The primary parameter type for tensor data.

```python
@T.prim_func
def matmul(
    A: T.Buffer((M, K), "float32"),    # 2D buffer of shape (M, K)
    B: T.Buffer((K, N), "float32"),    # 2D buffer of shape (K, N)
    C: T.Buffer((M, N), "float32"),    # 2D output buffer
) -> None:
    ...
```

**Handle-based parameters**: For generic pointers or when buffer shape is determined at runtime.

```python
@T.prim_func
def dynamic_kernel(
    A_handle: T.handle,
    B_handle: T.handle,
    n: T.int64,            # symbolic dimension
) -> None:
    A = T.match_buffer(A_handle, (n, n), "float32")
    B = T.match_buffer(B_handle, (n, n), "float32")
    for i, j in T.grid(n, n):
        with T.sblock("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + T.float32(1.0)
```

**Scalar parameters**: For passing integer or float constants.

```python
@T.prim_func
def scaled_add(
    A: T.Buffer((128,), "float32"),
    B: T.Buffer((128,), "float32"),
    alpha: T.float32,       # scalar parameter
) -> None:
    for i in range(128):
        with T.sblock("B"):
            vi = T.axis.spatial(128, i)
            B[vi] = A[vi] * alpha
```

### 18.3.3 Dynamic Shapes via Symbolic Variables

TVMScript supports symbolic dimensions using Python variable annotations in the function signature:

```python
@T.prim_func
def dynamic_matmul(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
) -> None:
    # M, K, N are symbolic variables automatically extracted from type annotations
    for i, j, k in T.grid(M, K, N):
        with T.sblock("C"):
            with T.init():
                C[i, j] = T.float32(0.0)
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] += A[vi, vk] * B[vk, vj]
```

For `T.handle`-based dynamic shapes, use `T.int64()` or `T.int32()` parameters:

```python
@T.prim_func
def dynamic_conv(
    data: T.handle,
    kernel: T.handle,
    out: T.handle,
    batch: T.int64,
    height: T.int64,
    width: T.int64,
    in_channels: T.int64,
    out_channels: T.int64,
    ksize: T.int64,
) -> None:
    D = T.match_buffer(data, (batch, height, width, in_channels), "float32")
    W = T.match_buffer(kernel, (out_channels, ksize, ksize, in_channels), "float32")
    O = T.match_buffer(out, (batch, height - ksize + 1, width - ksize + 1, out_channels), "float32")
    # ... computation ...
```

### 18.3.4 Buffer Allocation

```python
@T.prim_func
def my_kernel(
    A: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    # Allocate intermediate buffers
    # scope: "global", "shared", "local", "shared.dyn"
    B = T.alloc_buffer((128, 128), "float32", scope="global")
    shared_buf = T.alloc_buffer((128, 128), "float32", scope="shared")
    local_buf = T.alloc_buffer((16, 16), "float32", scope="local")

    # Dynamic shared memory (CUDA)
    dyn_shared = T.alloc_buffer((256,), "float32", scope="shared.dyn")

    for i, j in T.grid(128, 128):
        with T.sblock("compute"):
            vi, vj = T.axis.remap("SS", [i, j])
            shared_buf[vi, vj] = A[vi, vj]
            local_buf[vi % 16, vj % 16] = shared_buf[vi, vj]
            B[vi, vj] = local_buf[vi % 16, vj % 16]
            C[vi, vj] = B[vi, vj]
```

### 18.3.5 Loop Constructs

**T.grid**: Creates nested loops.

```python
# 3-dimensional loop nest
for i, j, k in T.grid(128, 128, 64):
    # body
    pass

# Equivalent to:
# for i in range(128):
#     for j in range(128):
#         for k in range(64):
#             body
```

**range loops**: Standard Python range is also supported.

```python
for i in range(128):
    for j in range(64):
        pass
```

**Serial, Parallel, and Vectorized loops**: Annotations via `T.serial`, `T.parallel`, `T.vectorized`.

```python
for i in T.serial(128):       # explicit serial loop
    pass

for i in T.parallel(128):     # parallelizable loop
    pass

for i in T.vectorized(16):    # vectorized loop
    pass
```

### 18.3.6 Scheduling Blocks

The `T.sblock` (scheduling block) defines a named computation region that can be targeted by scheduling primitives.

```python
with T.sblock("block_name"):
    # Block body
    # Must contain axis bindings via T.axis.*
    pass
```

### 18.3.7 Axis Annotations

Axis annotations define the iteration semantics of a scheduling block.

```python
# Spatial axis: output element depends only on this index
vi = T.axis.spatial(extent, value)

# Reduce axis: contributes to a reduction (sum, max, min, etc.)
vk = T.axis.reduce(extent, value)

# Remap: shorthand for spatial/reduce based on character codes
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
# 'S' = spatial, 'R' = reduce
```

```python
@T.prim_func
def matmul(
    A: T.Buffer((128, 64), "float32"),
    B: T.Buffer((64, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 64):
        with T.sblock("C"):
            vi = T.axis.spatial(128, i)      # spatial, extent=128, bound to i
            vj = T.axis.spatial(128, j)      # spatial, extent=128, bound to j
            vk = T.axis.reduce(64, k)         # reduce, extent=64, bound to k
            with T.init():
                C[vi, vj] = T.float32(0.0)
            C[vi, vj] += A[vi, vk] * B[vk, vj]
```

### 18.3.8 Reduction Initialization

The `T.init()` block sets the initial value for reduction operations:

```python
with T.sblock("matmul"):
    with T.init():
        C[i, j] = T.float32(0.0)      # Initialize accumulator to zero
    C[i, j] += A[i, k] * B[k, j]      # Accumulate products
```

Common reduction patterns:

```python
# Sum reduction
with T.init():
    result = T.float32(0.0)
result += value

# Max reduction
with T.init():
    result = T.min_value("float32")
result = T.max(result, value)

# Min reduction
with T.init():
    result = T.max_value("float32")
result = T.min(result, value)
```

### 18.3.9 Constants and Type Casting

```python
# Typed constants
x = T.float32(3.14)       # float32 constant
y = T.int32(42)           # int32 constant
z = T.int64(1000)         # int64 constant
b = T.bool(True)          # boolean constant

# Type casting
a_f32 = T.cast(a_i32, "float32")    # int32 -> float32
a_f16 = T.cast(a_f32, "float16")    # float32 -> float16
a_i8 = T.cast(a_f32, "int8")        # float32 -> int8 (truncation)

# Reinterpret cast (bit-level)
a_as_int = Treinterpret(a_f32, "int32")
```

### 18.3.10 Math Functions

```python
# Standard math
T.max(a, b)          # element-wise maximum
T.min(a, b)          # element-wise minimum
T.abs(a)             # absolute value
T.ceil(a)            # ceiling
T.floor(a)           # floor
T.round(a)           # rounding
T.sqrt(a)            # square root
T.rsqrt(a)           # reciprocal square root
T.exp(a)             # exponential
T.log(a)             # natural logarithm
T.log2(a)            # base-2 logarithm
T.sin(a)             # sine
T.cos(a)             # cosine
T.tanh(a)            # hyperbolic tangent
T.sigmoid(a)         # sigmoid function
T.clz(a)             # count leading zeros (integer)

# Power
T.pow(base, exp)     # power function

# Division variants
T.div(a, b)          # standard division
T.floordiv(a, b)     # floor division
T.floormod(a, b)     # floor modulo
T.truncdiv(a, b)     # truncation division
T.truncmod(a, b)     # truncation modulo
```

### 18.3.11 Expression Evaluation

```python
# Evaluate an expression (no assignment, just compute for side effects)
T.evaluate(expr)

# Example: barrier synchronization
T.evaluate(T.call_intrin("void", "tir.tvm_storage_sync", "shared"))

# Example: prefetch hint
T.evaluate(T.call_intrin("void", "tir.prefetch", addr, 0))
```

### 18.3.12 Conditional Statements

```python
# If-then-else
if condition:
    # then branch
    pass
else:
    # else branch
    pass

# Select expression (ternary)
result = T.select(condition, true_value, false_value)
```

### 18.3.13 Function Attributes

```python
@T.prim_func
def my_kernel(A: T.Buffer((128,), "float32")) -> None:
    T.func_attr({
        "global_symbol": "my_kernel",     # exported symbol name
        "tir.noalias": True,              # no pointer aliasing
        "target": T.target("cuda"),       # target hint (deprecated style)
    })
    for i in range(128):
        with T.sblock("A"):
            vi = T.axis.spatial(128, i)
            A[vi] = A[vi] + T.float32(1.0)
```

### 18.3.14 Let Bindings

```python
# Let binding in TVMScript
with T.let(var, value):
    # var is bound to value in this scope
    pass

# Example
with T.let(sum_val, A[i] + B[i]):
    C[i] = sum_val * sum_val
```

### 18.3.15 Assert Statements

```python
T.assert(condition, "Error message")
```

### 18.3.16 Complete TIR Example

```python
from tvm.script import ir as I, tirx as T

@I.ir_module
class SoftmaxModule:
    @T.prim_func
    def softmax(
        A: T.Buffer((M, N), "float32"),
        B: T.Buffer((M, N), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "softmax", "tir.noalias": True})
        # Softmax: B[i, j] = exp(A[i, j]) / sum_k(exp(A[i, k]))
        max_val = T.alloc_buffer((M,), "float32")
        exp_sum = T.alloc_buffer((M,), "float32")

        # Step 1: Find max per row
        for i, j in T.grid(M, N):
            with T.sblock("max"):
                vi, vj = T.axis.remap("SR", [i, j])
                with T.init():
                    max_val[vi] = T.min_value("float32")
                max_val[vi] = T.max(max_val[vi], A[vi, vj])

        # Step 2: Compute exp and sum
        for i, j in T.grid(M, N):
            with T.sblock("exp_sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                with T.init():
                    exp_sum[vi] = T.float32(0.0)
                exp_sum[vi] += T.exp(A[vi, vj] - max_val[vi])

        # Step 3: Normalize
        for i, j in T.grid(M, N):
            with T.sblock("normalize"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.exp(A[vi, vj] - max_val[vi]) / exp_sum[vi]
```

---

## 18.4 Relax Syntax (@R.function)

### 18.4.1 Function Definition

```python
@R.function
def function_name(
    param1: R.Tensor(shape, dtype),
    param2: R.Tensor(shape, dtype),
) -> R.Tensor(shape, dtype):
    """Relax function body."""
    ...
```

### 18.4.2 Type Annotations

```python
# Tensor type with known shape and dtype
x: R.Tensor((128, 128), "float32")

# Tensor with symbolic dimensions
x: R.Tensor(("batch", "seq_len", "hidden"), "float32")

# Tensor with unknown rank (dyn shape)
x: R.Tensor(dtype="float32")         # shape unknown
x: R.Tensor(ndim=2, dtype="float32") # rank 2, shapes unknown

# Tuple type
t: R.Tuple(R.Tensor((128,), "float32"), R.Tensor((64,), "int32"))

# Shape type
s: R.Shape((128, 64))

# Object type (generic)
obj: R.Object
```

### 18.4.3 Private and Impure Functions

```python
@R.function(private=True)
def internal_func(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
    """Private function - not exported in the final module symbol table."""
    return x

@R.function(pure=False)
def impure_func(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
    """Impure function - may have side effects (e.g., prints, logging)."""
    gv = R.call_pure_packed("my_external_func", x, sinfo_args=R.Tensor((128,), "float32"))
    return gv
```

### 18.4.4 Dataflow Blocks

Dataflow blocks mark regions of pure computation. Variables defined inside a dataflow block are local to that block unless explicitly exposed via `R.output()`.

```python
@R.function
def main(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
    with R.dataflow():
        # All variables here are "dataflow" (local) variables
        lv1 = R.nn.relu(x)
        lv2 = R.nn.relu(lv1)
        # Must output variables that are needed outside the block
        R.output(lv2)
    # lv2 is now accessible as a standard (binding) variable
    lv3 = lv2 * R.const(2.0, "float32")
    return lv3
```

### 18.4.5 R.output()

The `R.output()` call exposes dataflow block variables to the outer scope.

```python
with R.dataflow():
    lv1 = R.nn.relu(x)
    lv2 = R.nn.sigmoid(lv1)
    # Expose both variables
    R.output(lv1, lv2)
# Both lv1 and lv2 are accessible here
result = lv1 + lv2
```

### 18.4.6 R.call_tir

Calls a TIR function (prim_func) with tensor arguments. The output is allocated automatically.

```python
# Basic call_tir
lv = R.call_tir(
    MyModule.my_tir_func,           # TIR function reference
    (x, y),                          # input tuple
    out_sinfo=R.Tensor((128, 128), "float32"),  # output shape info
)

# call_tir with multiple outputs
lv1, lv2 = R.call_tir(
    MyModule.multi_output_func,
    (x,),
    out_sinfo=[
        R.Tensor((128, 128), "float32"),
        R.Tensor((128,), "float32"),
    ],
)

# call_tir with tir_vars (scalar arguments)
lv = R.call_tir(
    MyModule.dynamic_func,
    (x,),
    out_sinfo=R.Tensor((batch, seq, hidden), "float32"),
    tir_vars=PackedVars(batch, seq, hidden),
)
```

### 18.4.7 R.call_dps_packed

Calls an externally registered PackedFunc using the destination-passing style (DPS) convention.

```python
lv = R.call_dps_packed(
    "my_packed_func",                # function name (string)
    (x, y),                          # input tensors
    out_sinfo=R.Tensor((128, 128), "float32"),
)
```

### 18.4.8 R.call_pure_packed

Calls an externally registered PackedFunc (pure variant, no side effects).

```python
result = R.call_pure_packed(
    "tvm.contrib.cublas.matmul",
    x,
    y,
    sinfo_args=R.Tensor((M, N), "float32"),
)
```

### 18.4.9 Control Flow

```python
@R.function
def conditional_func(
    x: R.Tensor((128,), "float32"),
    flag: R.Tensor((), "bool"),
) -> R.Tensor((128,), "float32"):
    if flag:
        result = R.nn.relu(x)
    else:
        result = R.nn.sigmoid(x)
    return result
```

### 18.4.10 Tuple Operations

```python
@R.function
def tuple_func(
    x: R.Tensor((128,), "float32"),
) -> R.Tuple(R.Tensor((128,), "float32"), R.Tensor((128,), "float32")):
    # Create tuple
    tup = (x, x)
    # Tuple unfold
    a, b = tup
    # Tuple indexing
    first = tup[0]
    return tup
```

### 18.4.11 Shape Expressions

```python
@R.function
def shape_func(
    x: R.Tensor(("n", "m"), "float32"),
) -> R.Tensor(("n", "m"), "float32"):
    # Get shape of a tensor
    s = R.shape_of(x)  # R.Shape(("n", "m"))
    # Compute with shapes
    n, m = R.shape_to_tensor(s)
    new_shape = (m, n)
    return R.reshape(x, new_shape)
```

### 18.4.12 Complete Relax Example

```python
from tvm.script import ir as I, relax as R, tirx as T

@I.ir_module
class TransformerBlock:
    @T.prim_func
    def matmul_kernel(
        A: T.Buffer((1, 128, 64), "float32"),
        B: T.Buffer((64, 128), "float32"),
        C: T.Buffer((1, 128, 128), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "matmul_kernel", "tir.noalias": True})
        for b, i, j, k in T.grid(1, 128, 128, 64):
            with T.sblock("C"):
                with T.init():
                    C[b, i, j] = T.float32(0.0)
                vb, vi, vj, vk = T.axis.remap("SSSR", [b, i, j, k])
                C[vb, vi, vj] += A[vb, vi, vk] * B[vk, vj]

    @R.function
    def main(
        x: R.Tensor((1, 128, 64), "float32"),
        weight_q: R.Tensor((64, 128), "float32"),
        weight_k: R.Tensor((64, 128), "float32"),
        weight_v: R.Tensor((64, 128), "float32"),
    ) -> R.Tensor((1, 128, 128), "float32"):
        with R.dataflow():
            lv_q = R.call_tir(
                TransformerBlock.matmul_kernel,
                (x, weight_q),
                out_sinfo=R.Tensor((1, 128, 128), "float32"),
            )
            lv_k = R.call_tir(
                TransformerBlock.matmul_kernel,
                (x, weight_k),
                out_sinfo=R.Tensor((1, 128, 128), "float32"),
            )
            lv_v = R.call_tir(
                TransformerBlock.matmul_kernel,
                (x, weight_v),
                out_sinfo=R.Tensor((1, 128, 128), "float32"),
            )
            # Attention: Q * K^T / sqrt(d)
            lv_kt = R.permute_dims(lv_k, axes=[0, 2, 1])
            lv_attn = R.matmul(lv_q, lv_kt)
            lv_attn = lv_attn / R.const(8.0, "float32")
            lv_attn = R.nn.softmax(lv_attn, axis=-1)
            # Attention * V
            lv_out = R.matmul(lv_attn, lv_v)
            R.output(lv_out)
        return lv_out
```

### 18.4.13 R.rewriter Decorator

The `R.rewriter` decorator marks a function as a pattern-rewriting function for Relax graph optimization:

```python
@R.rewriter
def fuse_relu_matmul_pattern(
    x: R.Tensor((128, 64), "float32"),
    w: R.Tensor((64, 128), "float32"),
) -> R.Tensor((128, 128), "float32"):
    """Rewrites matmul + relu into a fused operation."""
    with R.dataflow():
        matmul_result = R.matmul(x, w)
        fused_result = R.nn.relu(matmul_result)
        R.output(fused_result)
    return fused_result
```

---

## 18.5 Parser Pipeline

The TVMScript parser transforms Python source code into TVM IR objects through a multi-stage pipeline.

### 18.5.1 Pipeline Stages

```
Python Source Code
       |
       v
[1] Python AST Extraction  (ast module)
       |
       v
[2] AST Dispatch           (node visitor pattern)
       |
       v
[3] Frame Stack            (IR builder frames)
       |
       v
[4] Variable Binding        (name resolution)
       |
       v
[5] IR Construction         (TIR/Relax IR nodes)
       |
       v
TVM IR Object (IRModule, PrimFunc, etc.)
```

### 18.5.2 Python AST Extraction

When TVMScript code is loaded, the parser uses Python's `ast` module to extract the AST:

```python
import ast

# TVMScript internally does something like:
source_code = """
@I.ir_module
class MyModule:
    @T.prim_func
    def add(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
        for i in range(128):
            with T.sblock("B"):
                vi = T.axis.spatial(128, i)
                B[vi] = B[vi] + A[vi]
"""
tree = ast.parse(source_code)
```

### 18.5.3 Dispatch Mechanism

The parser dispatches to different handlers based on the AST node type and context:

- **ClassDef** with `@I.ir_module` -> Parse as IRModule
- **FunctionDef** with `@T.prim_func` -> Parse as TIR PrimFunc
- **FunctionDef** with `@R.function` -> Parse as Relax function
- **For** loop -> Parse as TIR loop nest
- **With** block -> Parse as TIR sblock, init, etc.
- **Subscript** -> Parse as buffer access or type annotation
- **BinOp** -> Parse as TIR binary operation
- **Call** -> Dispatch based on function name (T.axis, T.Buffer, etc.)

### 18.5.4 IR Builder Frame Stack

The parser maintains a stack of "frames" that track the current context:

```
Frame Stack:
  [ModuleFrame]        -- IRModule construction
    [PrimFuncFrame]    -- TIR function construction
      [ForFrame]       -- Loop nest
        [BlockFrame]   -- Scheduling block
```

Each frame knows how to handle AST nodes within its context and how to finalize its contribution to the IR.

### 18.5.5 Variable Binding

Symbolic variables are extracted from type annotations:

```python
@T.prim_func
def func(A: T.Buffer((M, N), "float32")) -> None:
    # M and N are automatically registered as symbolic variables
    pass
```

The parser recognizes:
- Buffer shape annotations (`(M, N)`) as symbolic variables.
- Function parameters of type `T.int64`, `T.int32` as integer variables.
- `T.handle` parameters requiring `T.match_buffer` for shape binding.

---

## 18.6 Printer Pipeline

The TVMScript printer converts TVM IR objects back into TVMScript source code. This enables a roundtrip: `mod.script()` -> text -> parser -> IRModule.

### 18.6.1 Pipeline Stages

```
TVM IR Object (IRModule, PrimFunc, etc.)
       |
       v
[1] IR-to-Doc Conversion     (Python frame)
       |
       v
[2] Python Code Generation    (Doc -> string)
       |
       v
TVMScript Source Code (string)
```

### 18.6.2 Usage

```python
# Convert IRModule to TVMScript string
mod = ...  # IRModule
script_text = mod.script()
print(script_text)

# Roundtrip: parse the script back into an IRModule
from tvm.script import parse
mod_roundtrip = parse(script_text)

# Verify roundtrip
assert tvm.ir.structural_equal(mod, mod_roundtrip)
```

### 18.6.3 Selective Printing

```python
# Print only a specific function
func = mod["my_kernel"]
print(func.script())

# Print with show_meta=True (includes all metadata)
print(mod.script(show_meta=True))
```

---

## 18.7 TVMScript in Practice

### 18.7.1 Creating Modules from Strings

```python
from tvm.script import ir as I, tirx as T
import tvm

# Method 1: Direct class definition (parsed at import time)
@I.ir_module
class AddModule:
    @T.prim_func
    def add(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
        for i in range(128):
            with T.sblock("B"):
                vi = T.axis.spatial(128, i)
                B[vi] = B[vi] + A[vi]

# Method 2: Parse from string
source = """
from tvm.script import ir as I, tirx as T

@I.ir_module
class MyModule:
    @T.prim_func
    def add(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
        for i in range(128):
            with T.sblock("B"):
                vi = T.axis.spatial(128, i)
                B[vi] = B[vi] + A[vi]
"""
mod = tvm.script.parse(source)
```

### 18.7.2 Inspecting and Modifying Modules

```python
# Print the module in TVMScript format
print(mod.script())

# Access individual functions
func = mod["add"]
print(type(func))  # <class 'tvm.tir.PrimFunc'>

# Modify the module using passes
from tvm import tir
new_mod = tir.transform.Simplify()(mod)
print(new_mod.script())
```

### 18.7.3 Common Pitfalls

**Python execution vs. TVMScript parsing**: The body of `@T.prim_func` is parsed, not executed. Do not use Python runtime features:

```python
# WRONG: Python list comprehension is not supported in TVMScript
@T.prim_func
def bad_func(A: T.Buffer((128,), "float32")) -> None:
    vals = [A[i] for i in range(128)]  # ERROR: not valid TVMScript

# CORRECT: Use TIR loops
@T.prim_func
def good_func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
    for i in range(128):
        with T.sblock("B"):
            vi = T.axis.spatial(128, i)
            B[vi] = A[vi]
```

**Variable naming**: Dataflow block variables must be prefixed with `lv` or `gv`:

```python
@R.function
def my_func(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        lv1 = R.nn.relu(x)     # OK: starts with "lv"
        gv1 = lv1 * R.const(2.0)  # OK: starts with "gv"
        R.output(gv1)
    return gv1
```

**Buffer scope**: Always specify the correct buffer scope for GPU targets:

```python
# GPU kernel with shared memory
shared_buf = T.alloc_buffer((32, 32), "float32", scope="shared")
local_buf = T.alloc_buffer((16, 16), "float32", scope="local")
```

---

## 18.8 Source Code Locations

| Component | Path |
|---|---|
| Core parser infrastructure | `python/tvm/script/parser/core/` |
| TIR parser | `python/tvm/script/parser/tirx/` |
| Relax parser | `python/tvm/script/parser/relax/` |
| IR module parser | `python/tvm/script/parser/ir/` |
| IR builder (Python) | `python/tvm/script/ir_builder/` |
| TIR IR builder | `python/tvm/script/ir_builder/tir/` |
| Relax IR builder | `python/tvm/script/ir_builder/relax/` |
| C++ printer | `src/script/printer/` |
| C++ IR builder backend | `src/script/ir_builder/` |
| TVMScript `__init__.py` | `python/tvm/script/__init__.py` |
| Highlighting support | `python/tvm/script/highlight.py` |
