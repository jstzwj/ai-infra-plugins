# Relax IR -- Graph Abstraction for ML Models

Relax is TVM's graph-level intermediate representation designed specifically for expressing machine learning models. It provides a graph abstraction where nodes represent computational operations and edges represent data flow, while also supporting symbolic shapes, multi-level abstractions, and composable transformations. This document covers Relax IR in exhaustive detail, including its type system, core constructs, control flow, function definitions, and cross-level integration with TIR.

---

## 4.1 Graph Abstraction Overview

### 4.1.1 What Is Graph Abstraction

In the context of compiler IR for machine learning, graph abstraction represents a model as a directed acyclic graph (DAG) where:

- **Nodes** represent computational operations such as matrix multiplication, convolution, element-wise transforms, and reductions.
- **Edges** represent data flow: the output of one operation feeds as input to another.
- **Variables** are named bindings that capture intermediate results in the graph.

This abstraction naturally maps to how ML models are structured: a neural network is a sequence (or DAG) of layers where each layer takes tensor inputs and produces tensor outputs.

```
Input (x)
   |
   v
[Linear Layer: x @ W1 + b1]
   |
   v
[ReLU Activation]
   |
   v
[Linear Layer: h @ W2 + b2]
   |
   v
Output (logits)
```

Relax provides this graph abstraction while adding capabilities that traditional graph IRs lack:
- Symbolic shape reasoning throughout the graph
- Explicit marking of pure vs. side-effecting operations
- Direct integration with low-level TIR functions
- Support for control flow within the graph

### 4.1.2 Why Relax Exists

Prior to Relax, TVM used Relay as its graph-level IR. Relax was introduced to address several limitations:

1. **Symbolic shapes**: Relay treats shapes opaquely; Relax makes shapes first-class citizens. A Relax function can express `"batch"` as a symbolic dimension that flows through the graph without requiring concrete values during compilation.

2. **Cross-level optimization**: Relax is designed from the ground up to coexist with TIR in the same IRModule. The `R.call_tir` construct explicitly bridges the graph level and the loop-nest level, enabling passes that optimize across both.

3. **Dataflow purity**: Relax introduces dataflow blocks that explicitly mark regions of pure computation. This gives optimization passes precise boundaries for applying transformations like operator fusion without needing to prove purity through analysis.

4. **Destination-passing style**: Relax uses destination-passing style (DPS) for calling TIR functions, where the output buffer is allocated externally and passed as an argument. This avoids implicit allocations and makes memory management explicit.

### 4.1.3 Relax in the Compilation Pipeline

Relax sits at the center of the TVM compilation stack:

```
Frontend (PyTorch, ONNX, etc.)
         |
         v
    Relax IR (high-level graph)
         |
         v  (LegalizeOps, FuseOps, etc.)
    Relax IR + TIR PrimFuncs (mixed)
         |
         v  (LowerOps, etc.)
    TIR PrimFuncs (low-level loop nests)
         |
         v  (tirx.transform passes)
    Optimized TIR
         |
         v  (_codegen)
    Runtime Module (LLVM, CUDA, etc.)
```

---

## 4.2 Key Features of Relax

### 4.2.1 First-Class Symbolic Shapes

Relax treats tensor shapes as first-class symbolic expressions. Shape dimensions can be named variables rather than fixed constants. This enables:

- **Batch-size-independent compilation**: Compile once, run with any batch size.
- **Dynamic sequence lengths**: Handle variable-length sequences in NLP models.
- **Shape propagation**: Track how shapes transform through operations.

```python
from tvm.script import tirx as T, relax as R

@R.function
def symbolic_mlp(
    x: R.Tensor(("batch", "seq_len", "d_model"), "float32"),
    w1: R.Tensor(("d_model", "d_ff"), "float32"),
    b1: R.Tensor(("d_ff",), "float32"),
    w2: R.Tensor(("d_ff", "d_model"), "float32"),
    b2: R.Tensor(("d_model",), "float32"),
) -> R.Tensor(("batch", "seq_len", "d_model"), "float32"):
    batch = T.int64()
    seq_len = T.int64()
    d_model = T.int64()
    d_ff = T.int64()
    with R.dataflow():
        # Reshape to 2D for linear operation
        lv0 = R.reshape(x, R.shape((batch * seq_len, d_model)))
        # First linear layer: output shape uses symbolic d_ff
        lv1 = R.call_tir(
            "linear1",
            (lv0, w1, b1),
            out_sinfo=R.Tensor((batch * seq_len, d_ff), "float32"),
        )
        # Activation
        lv2 = R.nn.relu(lv1)
        # Second linear layer
        lv3 = R.call_tir(
            "linear2",
            (lv2, w2, b2),
            out_sinfo=R.Tensor((batch * seq_len, d_model), "float32"),
        )
        # Reshape back to 3D
        lv4 = R.reshape(lv3, R.shape((batch, seq_len, d_model)))
        R.output(lv4)
    return lv4
```

Symbolic shape variables (`batch`, `seq_len`, `d_model`, `d_ff`) are introduced as `T.int64()` variables. These names must match the symbolic names used in the function signature annotations. The shape arithmetic (e.g., `batch * seq_len`) is tracked by the compiler and used for memory planning.

### 4.2.2 Multi-Level Abstraction

Relax supports multiple levels of abstraction within the same function:

- **High-level NN operations**: `R.nn.relu`, `R.nn.softmax`, `R.nn.conv2d` -- these are abstract ops that will later be legalized to implementations.
- **Mid-level tensor operations**: `R.call_tir` with a TIR PrimFunc -- explicitly calls a low-level implementation.
- **Low-level packed calls**: `R.call_dps_packed` with an external function -- calls a library function directly.

```python
@R.function
def multi_level_example(
    x: R.Tensor((1, 3, 224, 224), "float32"),
    w_conv: R.Tensor((64, 3, 7, 7), "float32"),
    w_fc: R.Tensor((512, 512), "float32"),
) -> R.Tensor((1, 512), "float32"):
    with R.dataflow():
        # High-level: abstract conv2d op
        lv0 = R.nn.conv2d(x, w_conv, strides=[2, 2], padding=[3, 3, 3, 3])
        # High-level: abstract relu op
        lv1 = R.nn.relu(lv0)
        # High-level: abstract adaptive pool
        lv2 = R.nn.adaptive_avg_pool2d(lv1, output_size=[1, 1])
        # High-level: reshape
        lv3 = R.reshape(lv2, R.shape((1, 64)))
        # Mid-level: explicit TIR call for matmul
        lv4 = R.call_tir(
            "matmul_prim",
            (lv3, w_fc),
            out_sinfo=R.Tensor((1, 512), "float32"),
        )
        R.output(lv4)
    return lv4
```

During compilation, the `LegalizeOps` pass replaces high-level abstract ops with concrete implementations (either generated TIR or external library calls).

### 4.2.3 Composable Transformations

Relax is designed for composable, modular transformations. Each pass does one thing well, and passes can be composed into arbitrary pipelines:

```python
from tvm import transform, relax

# A typical Relax optimization pipeline
pipeline = transform.Sequential([
    # Decompose compound ops into simpler ones
    relax.transform.DecomposeOpsForTarget(),

    # Legalize remaining ops to TIR implementations
    relax.transform.LegalizeOps(),

    # Annotate TIR op patterns for fusion
    relax.transform.AnnotateTIROpPattern(),

    # Fuse operators based on patterns
    relax.transform.FuseOps(),

    # Fuse TIR functions
    relax.transform.FuseTIR(),

    # Layout transformation
    relax.transform.AlterOpImplementation(),

    # Bind constants
    relax.transform.BindParams("main", params),

    # Static shape planning
    relax.transform.StaticPlanBlockMemory(),
])

with transform.PassContext(opt_level=3):
    optimized_mod = pipeline(mod)
```

---

## 4.3 StructInfo -- Relax Type System

### 4.3.1 Overview

`StructInfo` (Structure Info) is Relax's type system. Every Relax expression has a StructInfo that describes its structure: what kind of value it produces, its shape, its dtype, and so on. StructInfo is more expressive than traditional type systems because it captures both the type and the static structure of values.

StructInfo serves multiple purposes:
- **Type checking**: Verify that operations receive valid inputs.
- **Shape inference**: Propagate symbolic shapes through the graph.
- **Optimization**: Enable passes to reason about tensor properties without runtime checks.

### 4.3.2 TensorStructInfo

`TensorStructInfo` describes a tensor value with a known (possibly symbolic) shape and dtype:

```python
from tvm import relax

# Static shape tensor
sinfo = relax.TensorStructInfo((128, 256), "float32")
print(sinfo.shape)   # ShapeExpr([128, 256])
print(sinfo.dtype)   # "float32"
print(sinfo.ndim)    # 2

# Symbolic shape tensor
from tvm.script import tirx as T
n = T.int64()
m = T.int64()
sym_sinfo = relax.TensorStructInfo((n, m), "float32")

# Unknown rank tensor (used when shape cannot be statically determined)
unknown_rank = relax.TensorStructInfo("float32", ndim=-1)

# Scalar tensor (0-dimensional)
scalar_sinfo = relax.TensorStructInfo((), "float32")

# 4D tensor (common for image data)
image_sinfo = relax.TensorStructInfo(("batch", 3, 224, 224), "float32")
```

In function annotations, `TensorStructInfo` is written as `R.Tensor`:

```python
@R.function
def my_func(
    x: R.Tensor((128, 256), "float32"),   # TensorStructInfo((128, 256), "float32")
    y: R.Tensor(("n",), "float32"),        # TensorStructInfo(("n",), "float32")
    z: R.Tensor("float32"),                # TensorStructInfo("float32", ndim=-1)
) -> R.Tensor((128, 256), "float32"):
    ...
```

### 4.3.3 TupleStructInfo

`TupleStructInfo` describes a tuple of values, where each element has its own StructInfo:

```python
from tvm import relax

# Tuple of two tensors
tuple_sinfo = relax.TupleStructInfo([
    relax.TensorStructInfo((128, 128), "float32"),
    relax.TensorStructInfo((128,), "int32"),
])

# Nested tuple
nested_sinfo = relax.TupleStructInfo([
    relax.TensorStructInfo((64, 64), "float32"),
    relax.TupleStructInfo([
        relax.TensorStructInfo((32,), "float32"),
        relax.TensorStructInfo((32,), "float32"),
    ]),
])

# Tuple with mixed types
mixed_sinfo = relax.TupleStructInfo([
    relax.TensorStructInfo((10,), "float32"),
    relax.PrimStructInfo("int32"),
    relax.ShapeStructInfo((128, 128)),
])
```

In function annotations, tuples are written as Python tuples:

```python
@R.function
def returns_tuple(
    x: R.Tensor((128, 128), "float32"),
) -> R.Tuple(R.Tensor((128, 128), "float32"), R.Tensor((128,), "float32")):
    ...
```

### 4.3.4 ShapeStructInfo

`ShapeStructInfo` describes a shape value itself (not a tensor, but the shape of a tensor). This is used when operations produce or consume shape values:

```python
from tvm import relax

# Known shape
shape_sinfo = relax.ShapeStructInfo((128, 256))

# Symbolic shape
from tvm.script import tirx as T
n = T.int64()
sym_shape_sinfo = relax.ShapeStructInfo((n, 256))

# Unknown shape
unknown_shape = relax.ShapeStructInfo(ndim=2)
```

### 4.3.5 PrimStructInfo

`PrimStructInfo` describes a scalar primitive value (not a tensor, but a single number or boolean):

```python
from tvm import relax

# Integer scalar
int_sinfo = relax.PrimStructInfo("int32")

# Float scalar
float_sinfo = relax.PrimStructInfo("float32")

# Boolean
bool_sinfo = relax.PrimStructInfo("bool")
```

PrimStructInfo is used for scalar values that appear in the graph, such as dynamic shape dimensions or configuration parameters.

### 4.3.6 ObjectStructInfo

`ObjectStructInfo` is the catch-all StructInfo for values that do not fit into the other categories. It represents an opaque object with no static structural information:

```python
from tvm import relax

# Opaque object (e.g., a runtime module, a PackedFunc handle)
obj_sinfo = relax.ObjectStructInfo()
```

This is typically used for values that are only meaningful at runtime, such as external library handles or dynamically loaded modules.

### 4.3.7 FuncStructInfo

`FuncStructInfo` describes a function value. It captures the parameter types and return type:

```python
from tvm import relax

# Function type: (Tensor(128,128,f32), Tensor(128,128,f32)) -> Tensor(128,128,f32)
func_sinfo = relax.FuncStructInfo(
    params=[
        relax.TensorStructInfo((128, 128), "float32"),
        relax.TensorStructInfo((128, 128), "float32"),
    ],
    ret=relax.TensorStructInfo((128, 128), "float32"),
)
```

FuncStructInfo is used for higher-order functions that take function arguments or return functions.

### 4.3.8 StructInfo Inference

Relax provides a mechanism for inferring StructInfo for any expression:

```python
from tvm import relax

# Get the StructInfo of an expression
expr = relax.op.add(x, y)
sinfo = expr.struct_info
print(sinfo)  # TensorStructInfo(shape=(128, 128), dtype="float32")

# For function parameters, StructInfo comes from the annotation
@R.function
def infer_example(
    x: R.Tensor(("n", "m"), "float32"),
) -> R.Tensor(("n", "m"), "float32"):
    # x.struct_info == TensorStructInfo(("n", "m"), "float32")
    y = R.nn.relu(x)
    # y.struct_info == TensorStructInfo(("n", "m"), "float32")
    return y
```

### 4.3.9 StructInfo Equality

StructInfo supports structural equality comparison, which is used extensively in type checking and optimization:

```python
from tvm.ir import structural_equal

# Two identical StructInfo values are structurally equal
sinfo_a = relax.TensorStructInfo((128, 256), "float32")
sinfo_b = relax.TensorStructInfo((128, 256), "float32")
assert structural_equal(sinfo_a, sinfo_b)

# Different shapes are not equal
sinfo_c = relax.TensorStructInfo((64, 256), "float32")
assert not structural_equal(sinfo_a, sinfo_c)

# Different dtypes are not equal
sinfo_d = relax.TensorStructInfo((128, 256), "float16")
assert not structural_equal(sinfo_a, sinfo_d)
```

---

## 4.4 R.call_tir -- Calling TIR from Relax

### 4.4.1 Overview

`R.call_tir` is the primary mechanism for calling TIR PrimFuncs from Relax functions. It implements destination-passing style (DPS), where the output buffer is allocated by the caller (the Relax runtime) and passed to the callee (the TIR function) as a parameter.

### 4.4.2 Syntax and Semantics

```python
# Basic syntax
result = R.call_tir(
    func,              # GlobalVar or string name of the TIR PrimFunc
    args,              # tuple of input tensors
    out_sinfo=...,     # StructInfo of the output (determines output allocation)
)
```

The semantics of `R.call_tir(func, args, out_sinfo)` are equivalent to:

1. Allocate an output buffer based on `out_sinfo` (shape and dtype).
2. Call `func(*args, output_buffer)`.
3. Return `output_buffer` as the result.

This explicit output allocation gives the compiler full control over memory management, enabling optimizations like in-place operations and memory planning.

### 4.4.3 Simple Example

```python
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class AddModule:
    @T.prim_func
    def add_prim(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((128,), "float32"),
    ) -> None:
        for i in T.grid(128):
            with T.sblock("C"):
                vi = T.axis.S(128, i)
                C[vi] = A[vi] + B[vi]

    @R.function
    def main(
        x: R.Tensor((128,), "float32"),
        y: R.Tensor((128,), "float32"),
    ) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            # Call the TIR add function
            # The output C buffer (128, float32) is allocated by the runtime
            lv = R.call_tir(
                AddModule.add_prim,       # reference to the TIR PrimFunc
                (x, y),                    # input tensors
                out_sinfo=R.Tensor((128,), "float32"),  # output shape/dtype
            )
            R.output(lv)
        return lv
```

### 4.4.4 Multiple Outputs

When a TIR function produces multiple output buffers, `out_sinfo` is a `TupleStructInfo`:

```python
@T.prim_func
def split_prim(
    A: T.Buffer((128,), "float32"),
    B: T.Buffer((64,), "float32"),
    C: T.Buffer((64,), "float32"),
) -> None:
    for i in T.grid(64):
        with T.sblock("split"):
            vi = T.axis.S(64, i)
            B[vi] = A[vi]
            C[vi] = A[vi + 64]

@R.function
def main(
    x: R.Tensor((128,), "float32"),
) -> R.Tuple(R.Tensor((64,), "float32"), R.Tensor((64,), "float32")):
    with R.dataflow():
        lv = R.call_tir(
            "split_prim",
            (x,),
            out_sinfo=R.Tuple(
                R.Tensor((64,), "float32"),
                R.Tensor((64,), "float32"),
            ),
        )
        R.output(lv)
    return lv
```

### 4.4.5 Symbolic Output Shapes

`R.call_tir` supports symbolic shapes in the output StructInfo:

```python
@R.function
def dynamic_linear(
    x: R.Tensor(("n", "d_in"), "float32"),
    w: R.Tensor(("d_in", "d_out"), "float32"),
    b: R.Tensor(("d_out",), "float32"),
) -> R.Tensor(("n", "d_out"), "float32"):
    n = T.int64()
    d_in = T.int64()
    d_out = T.int64()
    with R.dataflow():
        lv = R.call_tir(
            "linear_prim",
            (x, w, b),
            out_sinfo=R.Tensor((n, d_out), "float32"),
        )
        R.output(lv)
    return lv
```

### 4.4.6 Call TIR with TIR vars

When using `R.call_tir` inside a Relax function, you can pass TIR vars as additional arguments to pass scalar values:

```python
@T.prim_func
def scaled_add(
    A: T.Buffer((128,), "float32"),
    B: T.Buffer((128,), "float32"),
    C: T.Buffer((128,), "float32"),
    scale: T.float32,
) -> None:
    for i in T.grid(128):
        with T.sblock("C"):
            vi = T.axis.S(128, i)
            C[vi] = A[vi] + B[vi] * scale

@R.function
def main(
    x: R.Tensor((128,), "float32"),
    y: R.Tensor((128,), "float32"),
    scale: R.Prim("float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        lv = R.call_tir(
            "scaled_add",
            (x, y),
            scale,  # scalar argument
            out_sinfo=R.Tensor((128,), "float32"),
        )
        R.output(lv)
    return lv
```

---

## 4.5 R.call_dps_packed and R.call_pure_packed

### 4.5.1 R.call_dps_packed

`R.call_dps_packed` calls an externally registered function using destination-passing style. It is similar to `R.call_tir` but targets packed functions (typically C/C++ or library functions) rather than TIR PrimFuncs.

```python
@R.function
def call_dps_example(
    x: R.Tensor((128, 128), "float32"),
    w: R.Tensor((128, 256), "float32"),
) -> R.Tensor((128, 256), "float32"):
    with R.dataflow():
        # Call an external matmul library function via DPS
        lv = R.call_dps_packed(
            "tvm.contrib.cublas.matmul",   # packed function name
            (x, w),                          # input arguments
            out_sinfo=R.Tensor((128, 256), "float32"),
        )
        R.output(lv)
    return lv
```

The packed function must have the signature:

```c
// C signature for a DPS packed function
void packed_func(TVMArrayHandle* args, int num_args);
// The last arg is the pre-allocated output buffer
```

### 4.5.2 R.call_pure_packed

`R.call_pure_packed` calls a packed function that is known to be pure (side-effect free). It is similar to `call_dps_packed` but explicitly marks the call as pure, enabling more aggressive optimization:

```python
@R.function
def call_pure_example(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        lv = R.call_pure_packed(
            "my_pure_kernel",
            (x,),
            out_sinfo=R.Tensor((128,), "float32"),
        )
        R.output(lv)
    return lv
```

### 4.5.3 R.call_packed (with side effects)

For functions with side effects (e.g., printing, logging, random number generation), use `R.call_packed`:

```python
@R.function(pure=False)
def side_effect_example(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    # This call may have side effects, so it cannot be inside a dataflow block
    result = R.call_packed(
        "my_side_effect_kernel",
        (x,),
        sinfo_args=[relax.TensorStructInfo((128,), "float32")],
    )
    return result
```

### 4.5.4 Difference Summary

| Construct | Purity | DPS | Use Case |
|-----------|--------|-----|----------|
| `R.call_tir` | Pure | Yes | Call TIR PrimFunc from Relax |
| `R.call_dps_packed` | Pure | Yes | Call external library function (DPS) |
| `R.call_pure_packed` | Pure | No | Call pure packed function (returns value) |
| `R.call_packed` | Impure | No | Call arbitrary packed function |

---

## 4.6 Dataflow Blocks

### 4.6.1 Purpose

Dataflow blocks (`R.dataflow()`) mark regions of pure computation within a Relax function. They serve as optimization boundaries: all operations inside a dataflow block are guaranteed to be side-effect free, which enables aggressive transformations.

```python
@R.function
def with_dataflow(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    # Operations BEFORE the dataflow block may have side effects
    # (e.g., logging, random state mutation)

    with R.dataflow():
        # All operations here are pure (no side effects)
        lv0 = R.nn.relu(x)
        lv1 = R.nn.sigmoid(lv0)
        lv2 = R.multiply(lv1, R.const(2.0, "float32"))
        R.output(lv2)  # Expose lv2 to the outer scope

    # Operations AFTER the dataflow block may have side effects
    result = lv2  # Use the exposed variable
    return result
```

### 4.6.2 Rules for Dataflow Blocks

1. **All operations inside must be pure**: No side-effecting operations are allowed. This includes `R.call_packed` (impure calls), `R.print`, and any operation that modifies global state.

2. **Variables defined inside are DataflowVar**: Variables created inside a dataflow block are `DataflowVar` instances (not regular `Var`). They cannot be referenced outside the block unless explicitly exposed.

3. **R.output() exposes variables**: To use a dataflow variable outside the block, it must be passed to `R.output()`. The output creates a regular `Var` binding in the outer scope.

4. **Nested dataflow blocks are not allowed**: A dataflow block cannot contain another dataflow block.

```python
@R.function
def dataflow_rules(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        # DataflowVar: only visible inside this block
        lv0 = R.nn.relu(x)        # lv0: DataflowVar
        lv1 = R.nn.sigmoid(lv0)   # lv1: DataflowVar

        # ERROR: Cannot call impure function here
        # bad = R.call_packed("impure_func", (lv1,))

        # Expose lv1 to outer scope
        R.output(lv1)

    # Now lv1 is a regular Var and can be used
    return lv1
```

### 4.6.3 Multiple Outputs

A dataflow block can expose multiple variables:

```python
@R.function
def multi_output_dataflow(
    x: R.Tensor((128, 128), "float32"),
) -> R.Tuple(R.Tensor((128, 128), "float32"), R.Tensor((128, 128), "float32")):
    with R.dataflow():
        lv0 = R.nn.relu(x)
        lv1 = R.nn.sigmoid(x)
        R.output(lv0, lv1)
    return (lv0, lv1)
```

### 4.6.4 Why Manual Marking

Dataflow blocks are manually marked by the programmer or by the frontend converter. The reason for manual marking is **precision**:

- Automatically proving that a function call is pure requires interprocedural analysis, which is expensive and fragile.
- Manual marking provides clear, precise boundaries that optimization passes can rely on without analysis.
- The frontend (e.g., PyTorch converter) knows which operations are pure and can generate correct dataflow blocks.

This design trades some programmer convenience for compiler reliability and performance.

### 4.6.5 Optimization Within Dataflow Blocks

The dataflow block boundary is used by several optimization passes:

1. **Operator fusion**: Fuses consecutive element-wise operations within the same dataflow block into a single kernel.
2. **Dead code elimination**: Removes dataflow variables that are never used in `R.output`.
3. **Common subexpression elimination**: Deduplicates identical pure computations within a dataflow block.
4. **Memory planning**: Reuses memory buffers for dataflow variables that are not exposed.

```python
# Before fusion:
with R.dataflow():
    lv0 = R.nn.relu(x)      # separate kernel
    lv1 = R.nn.sigmoid(lv0)  # separate kernel
    R.output(lv1)

# After fusion (conceptual):
with R.dataflow():
    # relu and sigmoid fused into a single kernel
    lv1 = fused_relu_sigmoid(x)
    R.output(lv1)
```

---

## 4.7 Pure vs. Side-Effect Functions

### 4.7.1 Purity in Relax

A function is **pure** if:
- It only reads its input arguments (no mutation of global state).
- It returns its output exclusively through the return value (no side channels).
- It has no observable side effects (no I/O, no logging, no global state mutation).

A function has **side effects** if it violates any of the above:
- It modifies input tensors in-place.
- It writes to global state.
- It performs I/O (printing, network communication, etc.).
- It generates random numbers with a non-deterministic seed.

### 4.7.2 Marking Function Purity

By default, Relax functions are pure. Use `pure=False` to mark a function with side effects:

```python
@R.function
def pure_func(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
    # This function is pure by default
    with R.dataflow():
        lv = R.nn.relu(x)
        R.output(lv)
    return lv

@R.function(pure=False)
def impure_func(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
    # This function may have side effects
    result = R.call_packed(
        "side_effect_kernel",
        (x,),
        sinfo_args=[relax.TensorStructInfo((128,), "float32")],
    )
    return result
```

### 4.7.3 Implications of Purity

Purity affects what optimizations can be applied:

| Property | Pure Function | Impure Function |
|----------|--------------|-----------------|
| Can appear in dataflow block | Yes | No |
| Can be reordered | Yes | No (relative to other impure ops) |
| Can be eliminated if unused | Yes | No (side effects must execute) |
| Can be common-subexpression-eliminated | Yes | No |
| Can be fused | Yes | Limited |

### 4.7.4 In-Place Operations

Some operations conceptually modify their inputs in-place. In Relax, these are handled by creating a new output and letting the compiler decide whether to reuse memory:

```python
# In-place addition: conceptually x += y
# In Relax, this is expressed as a pure operation
@R.function
def in_place_add(
    x: R.Tensor((128,), "float32"),
    y: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        # This is pure: it returns a new tensor
        # The compiler may optimize to reuse x's memory
        lv = R.add(x, y)
        R.output(lv)
    return lv
```

---

## 4.8 Control Flow

### 4.8.1 Conditional (if/else)

Relax supports conditional execution via `if/else` statements. The condition must be a Relax variable with boolean type:

```python
@R.function
def conditional_example(
    x: R.Tensor((128,), "float32"),
    use_relu: R.Prim("bool"),
) -> R.Tensor((128,), "float32"):
    if use_relu:
        result = R.nn.relu(x)
    else:
        result = R.nn.sigmoid(x)
    return result
```

Both branches are required (no `if` without `else`). The return type of both branches must match structurally:

```python
@R.function
def conditional_with_tuple(
    x: R.Tensor((128, 128), "float32"),
    mode: R.Prim("bool"),
) -> R.Tensor((128, 128), "float32"):
    if mode:
        # Branch 1: split and recombine
        with R.dataflow():
            lv0 = R.split(x, indices_or_sections=2, axis=1)
            lv1 = R.multiply(lv0[0], R.const(2.0, "float32"))
            lv2 = R.multiply(lv0[1], R.const(3.0, "float32"))
            lv3 = R.concatenate((lv1, lv2), axis=1)
            R.output(lv3)
        result = lv3
    else:
        # Branch 2: simple scaling
        with R.dataflow():
            lv = R.multiply(x, R.const(1.5, "float32"))
            R.output(lv)
        result = lv
    return result
```

### 4.8.2 Condition Restrictions

The condition of an `if` statement must be a `R.Prim("bool")` value. Tensor conditions are not directly supported -- you must use reduction operations to produce a scalar:

```python
@R.function
def tensor_condition(
    x: R.Tensor((128,), "float32"),
    threshold: R.Prim("float32"),
) -> R.Tensor((128,), "float32"):
    # Compute a scalar condition from tensor operations
    with R.dataflow():
        max_val = R.max(x, axis=None)
        condition = R.op.greater(max_val, threshold)
        R.output(condition)

    # Use the scalar condition
    if condition:
        result = R.nn.relu(x)
    else:
        result = R.nn.sigmoid(x)
    return result
```

### 4.8.3 Nested Conditionals

Relax supports nested conditional statements:

```python
@R.function
def nested_conditional(
    x: R.Tensor((128,), "float32"),
    mode: R.Prim("int32"),
) -> R.Tensor((128,), "float32"):
    if R.op.equal(mode, R.const(0, "int32")):
        result = R.nn.relu(x)
    else:
        if R.op.equal(mode, R.const(1, "int32")):
            result = R.nn.sigmoid(x)
        else:
            result = R.nn.tanh(x)
    return result
```

---

## 4.9 Variable Types

### 4.9.1 Var

`Var` is the standard variable in Relax. It represents a named binding that can be used anywhere in the function. Variables defined outside dataflow blocks and variables exposed via `R.output()` are `Var` instances.

```python
from tvm import relax

# Create a Var
x = relax.Var("x", relax.TensorStructInfo((128,), "float32"))
print(x.name_hint)       # "x"
print(x.struct_info)     # TensorStructInfo((128,), "float32")

# Var can be used in any context
y = relax.op.add(x, x)   # valid usage
```

Characteristics of `Var`:
- Can be referenced from any point after its definition.
- Persists for the lifetime of the function.
- Can be used both inside and outside dataflow blocks.
- Is the type of function parameters.

### 4.9.2 DataflowVar

`DataflowVar` is a specialized variable used within dataflow blocks. It represents an intermediate computation that is only valid within its defining block.

```python
# Inside a dataflow block, all bindings are DataflowVar
@R.function
def dataflow_vars(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        lv0 = R.nn.relu(x)      # lv0 is a DataflowVar
        lv1 = R.nn.sigmoid(lv0)  # lv1 is a DataflowVar
        R.output(lv1)            # lv1 becomes a Var in outer scope
    return lv1                    # lv1 is now a Var
```

Characteristics of `DataflowVar`:
- Can only be referenced within the same dataflow block.
- Is automatically created for bindings inside `R.dataflow()`.
- Must be explicitly exposed via `R.output()` to be used outside.
- Enables dead code elimination and memory reuse optimizations.

```python
# Creating a DataflowVar directly (advanced)
from tvm import relax
dfv = relax.DataflowVar("dfv", relax.TensorStructInfo((64,), "float32"))
```

### 4.9.3 GlobalVar

`GlobalVar` represents a reference to a function within the IRModule. It is used when one function calls another:

```python
@I.ir_module
class ModuleWithCalls:
    @T.prim_func
    def add_func(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((128,), "float32"),
    ) -> None:
        for i in T.grid(128):
            with T.sblock("C"):
                vi = T.axis.S(128, i)
                C[vi] = A[vi] + B[vi]

    @R.function
    def main(
        x: R.Tensor((128,), "float32"),
    ) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            # ModuleWithCalls.add_func is a GlobalVar
            lv = R.call_tir(
                ModuleWithCalls.add_func,
                (x, x),
                out_sinfo=R.Tensor((128,), "float32"),
            )
            R.output(lv)
        return lv

# Access GlobalVar from module
gv = ModuleWithCalls.get_global_var("add_func")
print(type(gv))   # <class 'tvm.ir.GlobalVar'>
print(gv.name_hint)  # "add_func"
```

### 4.9.4 Variable Naming Conventions

By convention in TVM:
- Function parameters use meaningful names: `x`, `weight`, `bias`, `input`.
- Dataflow block variables use `lv` prefix: `lv0`, `lv1`, `lv2`.
- Global variables use descriptive names: `matmul`, `conv2d_relu`, `main`.

---

## 4.10 Function Definition

### 4.10.1 @R.function Decorator

The `@R.function` decorator is the primary way to define Relax functions in TVMScript:

```python
from tvm.script import relax as R, tirx as T

@R.function
def basic_function(
    x: R.Tensor((128, 784), "float32"),
    w: R.Tensor((784, 256), "float32"),
    b: R.Tensor((256,), "float32"),
) -> R.Tensor((128, 256), "float32"):
    with R.dataflow():
        lv = R.call_tir(
            "linear",
            (x, w, b),
            out_sinfo=R.Tensor((128, 256), "float32"),
        )
        R.output(lv)
    return lv
```

### 4.10.2 Private Functions

A function can be marked as private, meaning it is only used internally within the module and will not be exposed as a public API:

```python
@R.function(private=True)
def internal_helper(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        lv = R.nn.relu(x)
        R.output(lv)
    return lv
```

Private functions may be inlined or removed by optimization passes if they are not called.

### 4.10.3 Impure Functions

Functions with side effects must be marked with `pure=False`:

```python
@R.function(pure=False)
def impure_function(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    # Cannot use dataflow blocks here (would violate purity)
    result = R.call_packed(
        "my_stateful_op",
        (x,),
        sinfo_args=[relax.TensorStructInfo((128,), "float32")],
    )
    return result
```

### 4.10.4 Parameter and Return Annotations

Every parameter must have a StructInfo annotation. The return type annotation is required:

```python
@R.function
def annotated_function(
    # Tensor parameters
    x: R.Tensor(("batch", "seq_len", "d_model"), "float32"),
    w: R.Tensor(("d_model", "d_model"), "float32"),
    # Scalar parameters
    alpha: R.Prim("float32"),
    # Boolean parameters
    training: R.Prim("bool"),
    # Shape parameters
    seq_len: R.Shape(("seq_len",)),
) -> R.Tuple(
    R.Tensor(("batch", "seq_len", "d_model"), "float32"),
    R.Tensor(("batch", "seq_len", "d_model"), "float32"),
):
    batch = T.int64()
    seq_len_var = T.int64()
    d_model = T.int64()
    with R.dataflow():
        # Use the parameters
        scaled = R.multiply(x, R.prim_value(alpha))
        # ... more operations
        R.output(scaled, x)
    return (scaled, x)
```

### 4.10.5 Symbolic Variable Declaration

Symbolic shape variables used in annotations must be declared at the beginning of the function body:

```python
@R.function
def with_symbolic_shapes(
    x: R.Tensor(("n", "m"), "float32"),
) -> R.Tensor(("n", "m"), "float32"):
    # Declare symbolic variables (must match annotation names)
    n = T.int64()
    m = T.int64()
    with R.dataflow():
        lv = R.nn.relu(x)
        R.output(lv)
    return lv
```

These declarations serve as the binding point for the symbolic names. The type `T.int64()` indicates the shape dimensions are 64-bit integers.

### 4.10.6 Recursive and Mutually Recursive Functions

Relax supports recursive function calls:

```python
@R.function
def recursive_sum(
    x: R.Tensor(("n",), "float32"),
    n: R.Prim("int64"),
) -> R.Tensor((), "float32"):
    if R.op.equal(n, R.const(0, "int64")):
        result = R.const(0.0, "float32")
    else:
        partial = recursive_sum(x, R.sub(n, R.const(1, "int64")))
        result = R.add(partial, R.take(x, R.prim_value(n - 1), axis=0))
    return result
```

---

## 4.11 Shape Expressions

### 4.11.1 Symbolic Shape Variables

Shape dimensions can be symbolic variables that represent runtime-determined values:

```python
@R.function
def symbolic_shapes(
    x: R.Tensor(("batch", "seq_len", "d_model"), "float32"),
) -> R.Tensor(("batch", "seq_len", "d_model"), "float32"):
    batch = T.int64()
    seq_len = T.int64()
    d_model = T.int64()
    # batch, seq_len, d_model are now symbolic shape variables
    ...
```

### 4.11.2 Shape Arithmetic

Shapes can include arithmetic expressions:

```python
@R.function
def shape_arithmetic(
    x: R.Tensor(("n", "d"), "float32"),
) -> R.Tensor(("n", "d"), "float32"):
    n = T.int64()
    d = T.int64()
    with R.dataflow():
        # Reshape: (n, d) -> (n * d,)
        lv0 = R.reshape(x, R.shape((n * d,)))
        # Reshape back: (n * d,) -> (n, d)
        lv1 = R.reshape(lv0, R.shape((n, d)))
        # Split: (n, d) -> (n, d // 2) + (n, d - d // 2)
        lv2 = R.split(lv1, indices_or_sections=2, axis=1)
        R.output(lv2)
    return lv1
```

### 4.11.3 Shape Propagation

The Relax compiler tracks shape relationships across operations:

```python
@R.function
def shape_propagation(
    x: R.Tensor(("n", "in_features"), "float32"),
    w1: R.Tensor(("in_features", "hidden"), "float32"),
    w2: R.Tensor(("hidden", "out_features"), "float32"),
) -> R.Tensor(("n", "out_features"), "float32"):
    n = T.int64()
    in_features = T.int64()
    hidden = T.int64()
    out_features = T.int64()
    with R.dataflow():
        # n x in_features @ in_features x hidden -> n x hidden
        lv0 = R.call_tir(
            "matmul",
            (x, w1),
            out_sinfo=R.Tensor((n, hidden), "float32"),
        )
        # n x hidden @ hidden x out_features -> n x out_features
        lv1 = R.call_tir(
            "matmul",
            (lv0, w2),
            out_sinfo=R.Tensor((n, out_features), "float32"),
        )
        R.output(lv1)
    return lv1
```

### 4.11.4 R.shape()

The `R.shape()` construct creates a shape expression from a tuple of dimensions:

```python
# Static shape
s1 = R.shape((128, 256))

# Symbolic shape
s2 = R.shape((n, m))

# Mixed static and symbolic
s3 = R.shape((n, 3, 224, 224))

# Arithmetic in shape
s4 = R.shape((n * seq_len, d_model))
```

### 4.11.5 ShapeExpr

Internally, shapes are represented as `ShapeExpr` objects. A `ShapeExpr` is a tuple of `PrimExpr` values:

```python
from tvm import relax

# ShapeExpr from a tuple
shape = relax.ShapeExpr([128, 256])
print(shape)  # [128, 256]

# Symbolic ShapeExpr
n = tirx.Var("n", "int64")
sym_shape = relax.ShapeExpr([n, 256])
print(sym_shape)  # [n, 256]
```

---

## 4.12 NNModule Pattern

### 4.12.1 Overview

Relax provides a `nn.Module` system similar to PyTorch's `nn.Module` for defining neural network models. This pattern provides a familiar, object-oriented interface for model definition.

### 4.12.2 Basic Module Definition

```python
from tvm.relax.frontend import nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: nn.Tensor):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return x
```

### 4.12.3 Available NN Layers

The `nn` module provides common neural network layers:

```python
from tvm.relax.frontend import nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        # Linear layers
        self.fc = nn.Linear(128 * 8 * 8, 10)

        # Pooling layers (as methods, not modules)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x: nn.Tensor):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.max_pool2d(x, kernel_size=2)
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.max_pool2d(x, kernel_size=2)
        x = self.pool(x)
        x = nn.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
```

### 4.12.4 Exporting to IRModule

Models defined with `nn.Module` can be exported to TVM IRModules:

```python
# Create the model
model = SimpleMLP()

# Define the input specification
spec = {
    "forward": {
        "x": nn.spec.Tensor((1, 784), "float32"),
    }
}

# Export to IRModule and parameters
mod, params = model.export_tvm(spec=spec)

# mod is an IRModule with a "forward" Relax function
# params is a dict mapping parameter names to NDArray values
print(mod)

# Build the module
import tvm
from tvm import relax

target = tvm.target.Target("llvm")
exec_mod = relax.build(mod, target)

# Create a virtual machine to run
vm = relax.VirtualMachine(exec_mod, tvm.cpu())
vm["forward"](input_tensor)
```

### 4.12.5 Module with Custom Operations

```python
from tvm.relax.frontend import nn

class CustomOpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)

    def forward(self, x: nn.Tensor):
        x = self.linear(x)
        # Custom operation using nn.op
        x = nn.op.clip(x, min=0.0, max=1.0)
        x = nn.op.softmax(x, axis=-1)
        return x
```

### 4.12.6 Nested Modules

Modules can be composed hierarchically:

```python
from tvm.relax.frontend import nn

class Block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: nn.Tensor):
        x = self.linear(x)
        x = self.norm(x)
        x = nn.relu(x)
        return x

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block(784, 512)
        self.block2 = Block(512, 256)
        self.block3 = Block(256, 128)
        self.head = nn.Linear(128, 10)

    def forward(self, x: nn.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x
```

---

## 4.13 Cross-Level Integration

### 4.13.1 Relax and TIR Together

The defining feature of Relax is its tight integration with TIR through `R.call_tir`. A single IRModule can contain both Relax functions (graph-level) and TIR PrimFuncs (loop-level), and the two levels interact through well-defined interfaces.

```python
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class CrossLevelModule:
    # TIR: low-level matrix multiply
    @T.prim_func
    def matmul(
        A: T.Buffer((128, 64), "float32"),
        B: T.Buffer((64, 256), "float32"),
        C: T.Buffer((128, 256), "float32"),
    ) -> None:
        for i, j, k in T.grid(128, 256, 64):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    # TIR: low-level bias add
    @T.prim_func
    def bias_add(
        A: T.Buffer((128, 256), "float32"),
        B: T.Buffer((256,), "float32"),
        C: T.Buffer((128, 256), "float32"),
    ) -> None:
        for i, j in T.grid(128, 256):
            with T.sblock("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vj]

    # TIR: low-level ReLU
    @T.prim_func
    def relu(
        A: T.Buffer((128, 256), "float32"),
        B: T.Buffer((128, 256), "float32"),
    ) -> None:
        for i, j in T.grid(128, 256):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0.0))

    # Relax: high-level model using TIR functions
    @R.function
    def main(
        x: R.Tensor((128, 64), "float32"),
        w: R.Tensor((64, 256), "float32"),
        b: R.Tensor((256,), "float32"),
    ) -> R.Tensor((128, 256), "float32"):
        with R.dataflow():
            # Call TIR matmul
            lv0 = R.call_tir(
                CrossLevelModule.matmul,
                (x, w),
                out_sinfo=R.Tensor((128, 256), "float32"),
            )
            # Call TIR bias add
            lv1 = R.call_tir(
                CrossLevelModule.bias_add,
                (lv0, b),
                out_sinfo=R.Tensor((128, 256), "float32"),
            )
            # Call TIR ReLU
            lv2 = R.call_tir(
                CrossLevelModule.relu,
                (lv1,),
                out_sinfo=R.Tensor((128, 256), "float32"),
            )
            R.output(lv2)
        return lv2
```

### 4.13.2 The Lowering Process

During compilation, the high-level Relax operations are progressively lowered to TIR:

```python
# Step 1: Initial state (high-level ops)
@R.function
def step1(x: R.Tensor((128, 64), "float32")) -> R.Tensor((128, 256), "float32"):
    with R.dataflow():
        lv0 = R.nn.linear(x, weight, bias)      # high-level op
        lv1 = R.nn.relu(lv0)                      # high-level op
        R.output(lv1)
    return lv1

# Step 2: After DecomposeOps (decomposed into simpler ops)
@R.function
def step2(x: R.Tensor((128, 64), "float32")) -> R.Tensor((128, 256), "float32"):
    with R.dataflow():
        lv0 = R.matmul(x, weight)                  # decomposed linear
        lv1 = R.add(lv0, bias)                      # decomposed linear
        lv2 = R.nn.relu(lv1)                        # still high-level
        R.output(lv2)
    return lv2

# Step 3: After LegalizeOps (each op replaced with TIR call)
@R.function
def step3(x: R.Tensor((128, 64), "float32")) -> R.Tensor((128, 256), "float32"):
    with R.dataflow():
        lv0 = R.call_tir("matmul_prim", (x, weight),
                         out_sinfo=R.Tensor((128, 256), "float32"))
        lv1 = R.call_tir("bias_add_prim", (lv0, bias),
                         out_sinfo=R.Tensor((128, 256), "float32"))
        lv2 = R.call_tir("relu_prim", (lv1,),
                         out_sinfo=R.Tensor((128, 256), "float32"))
        R.output(lv2)
    return lv2
```

### 4.13.3 Cross-Level Optimization

Cross-level optimizations analyze both Relax and TIR to make better decisions:

1. **Operator fusion across call_tir boundaries**: If two consecutive `R.call_tir` calls operate on dataflow variables and the called TIR functions have compatible patterns, they can be fused into a single TIR function.

2. **Layout transformation propagation**: A layout change needed at the TIR level can propagate up to the Relax level to adjust the surrounding dataflow.

3. **Constant folding across levels**: If a `R.call_tir` input is a constant, the TIR function can be evaluated at compile time and replaced with the constant result.

```python
# Example: Fusion across call_tir boundaries
# Before fusion:
with R.dataflow():
    lv0 = R.call_tir("matmul", (x, w), out_sinfo=...)
    lv1 = R.call_tir("bias_add", (lv0, b), out_sinfo=...)
    lv2 = R.call_tir("relu", (lv1,), out_sinfo=...)
    R.output(lv2)

# After FuseOps + FuseTIR:
with R.dataflow():
    # Single fused TIR function: matmul_bias_relu
    lv0 = R.call_tir("fused_matmul_bias_relu", (x, w, b), out_sinfo=...)
    R.output(lv0)
```

### 4.13.4 Mixed IRModule Example

A realistic IRModule might contain many functions at different levels:

```python
@I.ir_module
class RealisticModule:
    # High-level entry point (Relax)
    @R.function
    def main(
        images: R.Tensor(("batch", 3, 224, 224), "float32"),
    ) -> R.Tensor(("batch", 1000), "float32"):
        batch = T.int64()
        with R.dataflow():
            # Backbone
            features = R.call_tir(
                "resnet_backbone",
                (images,),
                out_sinfo=R.Tensor((batch, 512, 7, 7), "float32"),
            )
            # Global pooling
            pooled = R.nn.adaptive_avg_pool2d(features, output_size=[1, 1])
            flattened = R.reshape(pooled, R.shape((batch, 512)))
            # Classification head
            logits = R.call_tir(
                "linear_head",
                (flattened,),
                out_sinfo=R.Tensor((batch, 1000), "float32"),
            )
            R.output(logits)
        return logits

    # Mid-level TIR function: backbone
    @T.prim_func
    def resnet_backbone(
        images: T.Buffer(("batch", 3, 224, 224), "float32"),
        features: T.Buffer(("batch", 512, 7, 7), "float32"),
    ) -> None:
        # Complex loop nest implementing convolution layers
        ...

    # Mid-level TIR function: classification head
    @T.prim_func
    def linear_head(
        inp: T.Buffer(("batch", 512), "float32"),
        out: T.Buffer(("batch", 1000), "float32"),
    ) -> None:
        # Loop nest for matrix multiply
        ...

    # Low-level TIR helper: single convolution
    @T.prim_func
    def conv2d_3x3(
        inp: T.Buffer(("b", "c_in", "h", "w"), "float32"),
        weight: T.Buffer(("c_out", "c_in", 3, 3), "float32"),
        out: T.Buffer(("b", "c_out", "h", "w"), "float32"),
    ) -> None:
        ...
```

### 4.13.5 Checking Function Types

When working with mixed IRModules, you may need to distinguish function types:

```python
from tvm import relax, tirx

for gv in mod.get_global_vars():
    func = mod[gv]
    if isinstance(func, relax.Function):
        print(f"{gv.name_hint}: Relax function")
        print(f"  Params: {[p.name_hint for p in func.params]}")
        print(f"  Return: {func.ret_struct_info}")
    elif isinstance(func, tirx.PrimFunc):
        print(f"{gv.name_hint}: TIR PrimFunc")
        print(f"  Params: {[p.name for p in func.params]}")
    else:
        print(f"{gv.name_hint}: Unknown function type")
```

---

## 4.14 Common Relax Operations Reference

### 4.14.1 Creation Operations

```python
# Constants
c1 = R.const(1.0, "float32")                    # scalar constant
c2 = R.const(np.array([1.0, 2.0, 3.0]))         # tensor constant

# PrimValue (scalar from TIR expression)
pv = R.prim_value(n * 2)                         # PrimValue from expression

# Tuple creation
t = (lv0, lv1, lv2)                              # tuple of Relax expressions
```

### 4.14.2 Shape Operations

```python
# Reshape
lv = R.reshape(x, R.shape((n, d)))

# Transpose
lv = R.transpose(x, axes=[1, 0, 2])

# Permute dimensions
lv = R.permute_dims(x, axes=[0, 2, 1])

# Expand dimensions
lv = R.expand_dims(x, axis=1)

# Squeeze dimensions
lv = R.squeeze(x, axis=1)

# Concatenate
lv = R.concatenate((x, y), axis=0)

# Split
lv = R.split(x, indices_or_sections=3, axis=1)

# Flatten
lv = R.flatten(x, start_dim=1)
```

### 4.14.3 Element-wise Operations

```python
# Arithmetic
lv = R.add(x, y)
lv = R.subtract(x, y)
lv = R.multiply(x, y)
lv = R.divide(x, y)
lv = R.negative(x)

# Activation functions
lv = R.nn.relu(x)
lv = R.nn.sigmoid(x)
lv = R.nn.tanh(x)
lv = R.nn.gelu(x)
lv = R.nn.silu(x)
lv = R.nn.softmax(x, axis=-1)

# Comparison
lv = R.greater(x, y)
lv = R.less(x, y)
lv = R.equal(x, y)

# Math
lv = R.exp(x)
lv = R.log(x)
lv = R.sqrt(x)
lv = R.abs(x)
lv = R.clip(x, min=0.0, max=1.0)
```

### 4.14.4 Reduction Operations

```python
# Sum
lv = R.sum(x, axis=1, keepdims=True)

# Max / Min
lv = R.max(x, axis=0)
lv = R.min(x, axis=0)

# Mean
lv = R.mean(x, axis=-1)

# Argmax / Argmin
lv = R.argmax(x, axis=1)
lv = R.argmin(x, axis=1)

# All / Any
lv = R.all(x, axis=0)
lv = R.any(x, axis=0)
```

### 4.14.5 Linear Algebra

```python
# Matrix multiply
lv = R.matmul(x, y)

# Batched matrix multiply
lv = R.matmul(x, y)  # handles batched inputs automatically

# Linear (matmul + bias)
lv = R.nn.linear(x, weight, bias)
```

### 4.14.6 NN-Specific Operations

```python
# Convolution
lv = R.nn.conv2d(x, weight, strides=[1, 1], padding=[1, 1])

# Pooling
lv = R.nn.max_pool2d(x, kernel_size=[2, 2], strides=[2, 2])
lv = R.nn.adaptive_avg_pool2d(x, output_size=[1, 1])

# Normalization
lv = R.nn.layer_norm(x, weight, bias, axes=-1)
lv = R.nn.batch_norm(x, weight, bias, mean, var, axis=1)

# Dropout (inference mode only in Relax)
lv = R.nn.dropout(x, rate=0.5)

# Embedding
lv = R.nn.embedding(indices, weight)
```

### 4.14.7 Type Casting

```python
# Cast dtype
lv = R.astype(x, "float16")

# Cast to boolean
lv = R.cast(x, "bool")
```

---

## 4.15 Relax Compilation Pipeline

### 4.15.1 Standard Pipeline

```python
from tvm import transform, relax

# Standard Relax compilation pipeline
def compile_relax_module(mod, target, params=None):
    """Compile a Relax IRModule to a runnable module."""

    # Phase 1: High-level Relax transformations
    pipeline_high = transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOpsByPattern(
            patterns=[...],  # define fusion patterns
        ),
        relax.transform.FuseTIR(),
        relax.transform.DecomposeOpsForTarget(target),
        relax.transform.AlterOpImplementation(),
    ])

    # Phase 2: Low-level TIR transformations
    pipeline_low = transform.Sequential([
        tirx.transform.Simplify(),
        tirx.transform.StorageRewrite(),
        tirx.transform.LowerAutoCopy(),
        tirx.transform.UnrollLoop(),
        tirx.transform.VectorizeLoop(),
        tirx.transform.BindTarget(target),
        tirx.transform.LowerCPU(),
        # or tirx.transform.LowerGPU() for GPU targets
    ])

    # Build
    with transform.PassContext(opt_level=3):
        mod = pipeline_high(mod)
        mod = pipeline_low(mod)
        exec_mod = relax.build(mod, target)

    return exec_mod
```

### 4.15.2 Running a Compiled Module

```python
import tvm
import numpy as np

# Build
target = tvm.target.Target("llvm")
exec_mod = compile_relax_module(mod, target)

# Create VM
dev = tvm.cpu()
vm = relax.VirtualMachine(exec_mod, dev)

# Prepare inputs
input_data = tvm.nd.array(np.random.randn(1, 784).astype("float32"), dev)

# Run
output = vm["main"](input_data)
print(output.numpy())
```

### 4.15.3 GPU Compilation

```python
# Compile for CUDA
target = tvm.target.Target("cuda")
with tvm.transform.PassContext(opt_level=3):
    exec_mod = relax.build(mod, target)

# Run on GPU
dev = tvm.cuda(0)
vm = relax.VirtualMachine(exec_mod, dev)
input_data = tvm.nd.array(np.random.randn(1, 784).astype("float32"), dev)
output = vm["main"](input_data)
```

---

## 4.16 Debugging Relax Programs

### 4.16.1 Printing Relax IR

```python
# Print the entire IRModule
print(mod)

# Print a specific function in TVMScript format
print(mod["main"])

# Show with syntax highlighting (in Jupyter/interactive)
mod.show()
```

### 4.16.2 Inspecting StructInfo

```python
# Get StructInfo of a function
func = mod["main"]
print(f"Return type: {func.ret_struct_info}")
print(f"Params: {[p.struct_info for p in func.params]}")

# Get StructInfo of an expression within a function
# This requires understanding the AST structure
body = func.body
if isinstance(body, relax.SeqExpr):
    for binding in body.blocks[0].bindings:
        print(f"  {binding.var.name_hint}: {binding.var.struct_info}")
```

### 4.16.3 Step-by-Step Compilation Debug

```python
from tvm import transform, relax

# Debug pipeline: print after each major step
debug_pipeline = transform.Sequential([
    relax.transform.LegalizeOps(),
    transform.PrintIR(name="after_legalize"),
    relax.transform.FuseOpsByPattern([...]),
    transform.PrintIR(name="after_fuse"),
    relax.transform.FuseTIR(),
    transform.PrintIR(name="after_fuse_tir"),
])

with transform.PassContext(opt_level=3):
    mod = debug_pipeline(mod)
```

### 4.16.4 Common Errors and Solutions

```python
# Error: "Cannot use DataflowVar outside of dataflow block"
# Solution: Ensure all DataflowVars are exposed via R.output()

# Error: "StructInfo mismatch in call_tir"
# Solution: Verify that out_sinfo matches the TIR function's output buffer

# Error: "Impure operation in dataflow block"
# Solution: Move impure operations outside R.dataflow() or use pure=True

# Error: "Symbolic shape variable not bound"
# Solution: Declare all symbolic variables with T.int64() at function start
n = T.int64()  # Must appear before first use of "n"
```
