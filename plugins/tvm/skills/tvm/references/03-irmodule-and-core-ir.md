# IRModule and Core IR Infrastructure

The IRModule is the central data structure that underpins the entire Apache TVM compilation stack. It provides a unified container for functions at multiple levels of abstraction, enabling whole-program optimization across the boundary between high-level graph operations and low-level loop nests. This document covers the IRModule, its type system, operator registry, pass infrastructure, and all related core IR mechanisms in exhaustive detail.

---

## 3.1 IRModule Overview

### 3.1.1 What Is an IRModule

The IRModule (Intermediate Representation Module) is TVM's primary compilation unit. It represents a complete program or a fragment of a program as a collection of named functions. Every stage of the TVM compilation pipeline -- from frontend import through optimization to code generation -- operates on IRModule instances.

An IRModule maps `GlobalVar` names to function bodies. Each function in the module can be one of several variants, and different variants can coexist within the same IRModule. This coexistence is fundamental to TVM's design: it enables cross-level optimization where a high-level graph pass can analyze and transform code that spans both the relax layer and the TIR layer.

```python
import tvm
from tvm import relax, tirx
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class MyModule:
    """An IRModule with both Relax and TIR functions."""

    @T.prim_func
    def matmul_prim(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ) -> None:
        """Low-level TIR PrimFunc implementing matrix multiply."""
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("C_update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
        w: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        """High-level Relax function implementing the model."""
        with R.dataflow():
            lv = R.call_tir(
                MyModule.matmul_prim,
                (x, w),
                out_sinfo=R.Tensor((128, 128), "float32"),
            )
            R.output(lv)
        return lv
```

### 3.1.2 Function Variants

The IRModule supports two primary function variants. These two variants together span the full range from high-level model description to low-level hardware instructions.

#### relax::Function

A `relax::Function` represents a high-level computational graph with support for control flow, symbolic shapes, and calls to lower-level TIR functions. Relax functions describe *what* to compute: they define the dataflow between operations using a graph abstraction.

Key characteristics:
- Operates on tensors with symbolic shapes (e.g., `(n, 784)`)
- Uses `StructInfo` (Structure Info) as its type system instead of raw types
- Supports dataflow blocks for marking pure computation regions
- Can call TIR PrimFuncs via `R.call_tir`
- Can call external packed functions via `R.call_dps_packed` and `R.call_pure_packed`
- Supports control flow (if/else) and recursive function calls

```python
@R.function
def relax_func(
    x: R.Tensor(("batch", "seq_len", "d_model"), "float32"),
    w: R.Tensor(("d_model", "d_ff"), "float32"),
    b: R.Tensor(("d_ff",), "float32"),
) -> R.Tensor(("batch", "seq_len", "d_ff"), "float32"):
    batch, seq_len, d_model = T.int64(), T.int64(), T.int64()
    d_ff = T.int64()
    with R.dataflow():
        # Reshape for linear layer
        lv0 = R.reshape(x, R.shape((batch * seq_len, d_model)))
        # Call TIR function for matmul
        lv1 = R.call_tir(
            "matmul",
            (lv0, w),
            out_sinfo=R.Tensor((batch * seq_len, d_ff), "float32"),
        )
        # Add bias
        lv2 = R.call_tir(
            "bias_add",
            (lv1, b),
            out_sinfo=R.Tensor((batch * seq_len, d_ff), "float32"),
        )
        # Reshape back
        lv3 = R.reshape(lv2, R.shape((batch, seq_len, d_ff)))
        R.output(lv3)
    return lv3
```

#### tirx::PrimFunc

A `tirx::PrimFunc` represents a low-level imperative program with explicit loop nests, buffer access, threading primitives, and vector/tensor instructions. TIR functions describe *how* to compute: they specify the precise loop structure, memory access patterns, and hardware-specific operations.

Key characteristics:
- Explicit loop nests with `T.grid` and `T.sblock`
- Buffer objects with defined memory layout
- Thread-level parallelism via `T.axis.parallel`
- Vectorized and tensorized operations via `T.axis.vectorized`
- Memory scope annotations (shared, local, texture, etc.)
- Direct hardware intrinsic access

```python
@T.prim_func
def vectorized_add(
    A: T.Buffer((1024,), "float32"),
    B: T.Buffer((1024,), "float32"),
    C: T.Buffer((1024,), "float32"),
) -> None:
    """Vectorized addition using TIR primitives."""
    for i in T.grid(1024):
        with T.sblock("C"):
            vi = T.axis.S(1024, i)
            C[vi] = A[vi] + B[vi]
```

### 3.1.3 Coexistence of Function Variants

A single IRModule can contain both relax::Function and tirx::PrimFunc. The high-level relax functions call into low-level TIR functions via `R.call_tir`, creating a two-layer architecture. This design enables:

1. **Cross-level optimization**: A pass can inline a simple TIR function into a Relax graph, or conversely, lift a pattern from Relax into a reusable TIR function.
2. **Incremental lowering**: The compilation pipeline gradually lowers Relax functions by replacing high-level operations with calls to progressively more optimized TIR functions.
3. **Mixed execution**: Some operations may remain at the Relax level (dispatched to external libraries) while others are fully lowered to TIR and compiled to native code.

```python
@I.ir_module
class MixedModule:
    # TIR function: low-level elementwise compute
    @T.prim_func
    def relu_prim(
        data: T.Buffer((1, 64), "float32"),
        out: T.Buffer((1, 64), "float32"),
    ) -> None:
        for i, j in T.grid(1, 64):
            with T.sblock("out"):
                vi, vj = T.axis.remap("SS", [i, j])
                out[vi, vj] = T.max(data[vi, vj], T.float32(0.0))

    # Relax function: high-level model logic
    @R.function
    def main(
        x: R.Tensor((1, 64), "float32"),
    ) -> R.Tensor((1, 64), "float32"):
        with R.dataflow():
            # Call the TIR PrimFunc from Relax
            lv = R.call_tir(
                MixedModule.relu_prim,
                (x,),
                out_sinfo=R.Tensor((1, 64), "float32"),
            )
            R.output(lv)
        return lv
```

### 3.1.4 IRModule Operations

The IRModule provides methods for querying and manipulating its contents.

```python
# Create an IRModule
mod = tvm.IRModule.from_expr(relax_func)

# Access functions by name
func = mod["main"]
print(func)  # prints the function's TVMScript representation

# Add a new function
mod["new_func"] = new_relax_func

# Update an existing function (returns a new IRModule, immutable)
mod = mod.update_func(mod.get_global_var("main"), updated_func)

# List all function names
for gv in mod.get_global_vars():
    print(gv.name_hint)

# Remove a function
mod = mod.remove("unused_func")

# Get functions of a specific type
relax_funcs = [mod[gv] for gv in mod.get_global_vars()
               if isinstance(mod[gv], relax.Function)]
tir_funcs = [mod[gv] for gv in mod.get_global_vars()
             if isinstance(mod[gv], tirx.PrimFunc)]
```

### 3.1.5 IRModule Properties

IRModules carry metadata that guides compilation:

```python
@I.ir_module
class AnnotatedModule:
    I.module_attrs({"target": "cuda", "num_input": 2})

    I.module_global_infos(
        {
            "kCompilerAttrs": [
                R.call_pure_packed(
                    "tvm.contrib.thrust.can_use_thrust",
                    R.prim_value(1),
                    sinfo_args=[R.Prim("int32")],
                )
            ]
        }
    )

    @R.function
    def main(
        x: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        with R.dataflow():
            gv = R.call_tir("identity", (x,), out_sinfo=R.Tensor((128, 128), "float32"))
            R.output(gv)
        return gv
```

---

## 3.2 Unified Type System

### 3.2.1 Overview

TVM's type system is shared across both relax::Function and tirx::PrimFunc. This unification means that type-level reasoning works consistently regardless of which function variant is being analyzed. The type system includes primitive types, tensor types, tuple types, and function types.

The unified type system is critical because it enables:
1. Cross-function-variant type checking: A Relax function's return type can be verified against a called TIR function's output buffer type.
2. Type-driven optimizations: Passes can reason about types without special-casing the function variant.
3. Consistent serialization: Types serialize and deserialize identically regardless of where they appear.

### 3.2.2 PrimType

`PrimType` represents scalar primitive types. These are the fundamental building blocks of the type system.

```python
from tvm import tirx

# Creating PrimType instances
int32_type = tvm.ir.PrimType("int32")
float32_type = tvm.ir.PrimType("float32")
bool_type = tvm.ir.PrimType("bool")
int64_type = tvm.ir.PrimType("int64")
float16_type = tvm.ir.PrimType("float16")
uint8_type = tvm.ir.PrimType("uint8")
e4m3_float8_type = tvm.ir.PrimType("e4m3_float8")
e5m2_float8_type = tvm.ir.PrimType("e5m2_float8")

# PrimType is used as the dtype field of TensorType
# and as the type of scalar variables in TIR
x = tirx.Var("x", "int32")  # type is PrimType("int32")
assert x.dtype == "int32"
```

Supported primtype dtype strings include:

| Category | Dtypes |
|----------|--------|
| Integer | `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64` |
| Float | `float16`, `float32`, `float64`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` |
| Other | `bool`, `e4m3_float8`, `e5m2_float8` |

### 3.2.3 TensorType

`TensorType` represents a multi-dimensional array (tensor) with a fixed shape and element dtype. It is the most commonly used type in ML workloads.

```python
from tvm import ir

# Static shape tensor
static_tensor = ir.TensorType((128, 128), "float32")
print(static_tensor.shape)  # [128, 128]
print(static_tensor.dtype)  # "float32"
print(static_tensor.ndim)   # 2

# Symbolic shape tensor
n = tirx.Var("n", "int64")
symbolic_tensor = ir.TensorType((n, 784), "float32")
print(symbolic_tensor.shape)  # [n, 784]

# Scalar tensor (0-dimensional)
scalar_tensor = ir.TensorType((), "float32")

# High-dimensional tensor
hd_tensor = ir.TensorType((1, 3, 224, 224), "float32")
```

The shape field of a TensorType is a list of `PrimExpr` values. These can be constants (`IntImm`), symbolic variables (`tirx.Var`), or expressions involving symbolic variables:

```python
# Shape with arithmetic expressions
batch = tirx.Var("batch", "int64")
seq = tirx.Var("seq", "int64")
d_model = tirx.Var("d_model", "int64")
reshaped_type = ir.TensorType((batch * seq, d_model), "float32")
```

### 3.2.4 TupleType

`TupleType` represents a heterogeneous collection of values, analogous to a struct or tuple in programming languages. Each field can have a different type.

```python
from tvm import ir

# Tuple of tensors
tuple_type = ir.TupleType([
    ir.TensorType((128, 128), "float32"),
    ir.TensorType((128,), "float32"),
    ir.TensorType((128, 128), "int32"),
])

# Nested tuples
nested_tuple = ir.TupleType([
    ir.TensorType((64, 64), "float32"),
    ir.TupleType([
        ir.TensorType((32, 32), "float32"),
        ir.TensorType((32, 32), "float32"),
    ]),
])

# Tuple of prim types
prim_tuple = ir.TupleType([
    ir.PrimType("int32"),
    ir.PrimType("float32"),
])
```

TupleType is used to represent:
- Multiple return values from functions
- Intermediate values in Relax that group related tensors
- Named fields (though field names are not part of the type itself)

### 3.2.5 FuncType

`FuncType` represents a function signature: a mapping from parameter types to a return type. Both Relax functions and TIR PrimFuncs use FuncType to describe their type signatures.

```python
from tvm import ir

# Function type for: (Tensor(128,128,f32), Tensor(128,128,f32)) -> Tensor(128,128,f32)
func_type = ir.FuncType(
    arg_types=[
        ir.TensorType((128, 128), "float32"),
        ir.TensorType((128, 128), "float32"),
    ],
    ret_type=ir.TensorType((128, 128), "float32"),
)

# Function type with no arguments returning a scalar
nullary_type = ir.FuncType(
    arg_types=[],
    ret_type=ir.PrimType("int32"),
)

# Function type with tuple return
multi_return_type = ir.FuncType(
    arg_types=[ir.TensorType((128, 128), "float32")],
    ret_type=ir.TupleType([
        ir.TensorType((128, 64), "float32"),
        ir.TensorType((64, 128), "float32"),
    ]),
)
```

FuncType can also carry type parameters for generic/polymorphic functions, though this is less common in typical ML workloads:

```python
# FuncType with type constraints (advanced)
generic_type = ir.FuncType(
    arg_types=[ir.TensorType((1,), "float32")],
    ret_type=ir.TensorType((1,), "float32"),
    type_params=[],
    type_constraints=[],
)
```

### 3.2.6 Type Consistency Across Function Variants

The same type system applies to both function variants. This means a Relax function and a TIR PrimFunc can share type definitions and type-level reasoning:

```python
from tvm import ir

# A shared tensor type
weight_type = ir.TensorType((768, 768), "float32")

# This type is valid for both a Relax parameter annotation
# and for TIR buffer type checking
# In Relax:
#   w: R.Tensor((768, 768), "float32")  -> TensorStructInfo wrapping TensorType
# In TIR:
#   W: T.Buffer((768, 768), "float32")  -> Buffer with matching TensorType
```

When a Relax function calls a TIR PrimFunc via `R.call_tir`, the type system ensures that the output StructInfo of the `call_tir` matches the output buffer type of the TIR function. This cross-variant type consistency is what makes whole-program optimization reliable.

---

## 3.3 Op Class — Operator Registry

### 3.3.1 What Is the Op Class

The `Op` class is a registry of system-defined primitive operators and intrinsics. Each Op instance represents a named operation (like `add`, `matmul`, `conv2d`, `relu`) and can carry additional attributes that describe its properties.

Ops serve as the bridge between high-level graph operations and their implementations. When the Relax frontend encounters a `relay.add` or a `relax.op.add`, it resolves to a registered Op. Downstream passes then match these Ops to implement them via TIR functions, external library calls, or other mechanisms.

```python
from tvm import ir

# Access a registered Op
add_op = ir.Op.get("relax.add")
print(add_op.name)           # "relax.add"
print(add_op.num_inputs)     # number of input arguments

# All Ops are singletons -- same name always returns same Op
assert ir.Op.get("relax.add") is ir.Op.get("relax.add")
```

### 3.3.2 Registered Op Attributes

Ops can carry arbitrary attributes that describe their behavior. These attributes are used by passes to make optimization decisions:

```python
# Check if an Op has a specific attribute
add_op = ir.Op.get("relax.add")
# Ops can have attributes like:
# - "FInferStructInfo" : function to infer output shape/dtype from inputs
# - "FPurity"          : whether the op is pure (side-effect free)
# - "TCallEffectKind"  : effect classification (pure, read-only, etc.)
```

### 3.3.3 Registering New Ops

Developers can register new operators with custom attributes. This is typically done in C++ using the `TVM_REGISTER_OP` macro, but the Python API also provides registration mechanisms.

C++ registration (the standard approach):

```cpp
// In C++ (simplified)
TVM_REGISTER_OP("relax.my_custom_op")
    .set_num_inputs(2)
    .set_attr<FInferStructInfo>("FInferStructInfo", MyCustomOpInferStructInfo)
    .set_attr<TCallEffectKind>("TCallEffectKind", CallEffectKind::kPure)
    .set_attr<std::string>("TOpPattern", "kElemWise");
```

Python attribute access:

```python
# After registration, attributes are accessible
my_op = ir.Op.get("relax.my_custom_op")
# The FInferStructInfo attribute would be used by the type inference pass
```

### 3.3.4 Common Op Categories

Ops are organized into categories based on their computation pattern, which guides fusion decisions:

| Pattern | Description | Example Ops |
|---------|-------------|-------------|
| `kElemWise` | Element-wise, no data dependency between elements | `add`, `relu`, `sigmoid` |
| `kBroadcast` | Element-wise with broadcasting | `broadcast_add`, `broadcast_mul` |
| `kInjective` | Injective: different inputs always produce different outputs | `reshape`, `transpose`, `clip` |
| `kCommReduce` | Communication-reducing: reduces dimensionality | `sum`, `max`, `argmax` |
| `kOutEWiseFusable` | Output is element-wise fusable with following ops | `matmul`, `conv2d` |
| `kTuple` | Produces a tuple of tensors | `split` |
| `kOpaque` | No structural assumption | `custom_library_call` |

These patterns are stored as the `"TOpPattern"` attribute on each Op and are consumed by the operator fusion pass:

```python
# Query the pattern of an op (if registered)
# In practice, this is done through internal APIs
conv2d_op = ir.Op.get("relax.nn.conv2d")
matmul_op = ir.Op.get("relax.matmul")
add_op = ir.Op.get("relax.add")
relu_op = ir.Op.get("relax.nn.relu")
```

### 3.3.5 Op Implementation Dispatch

The actual implementation of an Op is determined at lower stages. The same `relax.add` Op might be implemented differently depending on the target:

1. **Lowered to a TIR PrimFunc**: A pass generates a TIR implementation and replaces the Op call with `R.call_tir`.
2. **Dispatched to an external library**: A pass replaces the Op call with `R.call_dps_packed` pointing to a library function (e.g., cuBLAS for matmul).
3. **Kept as-is**: Some Ops are resolved only at the final runtime stage.

This dispatch mechanism means the Op class itself is target-independent; the compilation pipeline makes the implementation decision.

---

## 3.4 PassContext — Configuring Pass Behavior

### 3.4.1 Overview

`PassContext` is the configuration mechanism for TVM's pass infrastructure. It allows users to control how individual passes behave, set global optimization levels, and enable or disable specific passes. PassContext is a context manager that sets configuration for the duration of a `with` block.

### 3.4.2 Basic Usage

```python
import tvm
from tvm import transform

# Configure pass behavior with PassContext
with transform.PassContext(
    opt_level=3,
    config={
        "tirx.UnrollLoop": {"auto_max_step": 10},
        "tirx.vectorize.SkipCriterion": "skip_widen_intrin",
    },
):
    mod = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        tirx.transform.UnrollLoop(),
    ])(mod)
```

### 3.4.3 PassContext Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opt_level` | int | 2 | Global optimization level (0 = minimal, 4 = aggressive) |
| `required_pass` | List[str] | `[]` | Passes that must be included; error if not present |
| `disabled_pass` | List[str] | `[]` | Passes that must be skipped |
| `config` | Dict | `{}` | Per-pass configuration key-value pairs |
| `instruments` | List[PassInstrument] | `[]` | Instrumentation hooks for pass execution |
| `trace_stack` | List | `[]` | Trace information for debugging |
| `make_traceable` | bool | False | Whether to enable tracing |

### 3.4.4 Optimization Levels

The `opt_level` parameter controls how aggressively passes optimize:

```python
# Level 0: No optimization, fastest compilation
with transform.PassContext(opt_level=0):
    mod = pipeline(mod)

# Level 1: Basic optimizations only
with transform.PassContext(opt_level=1):
    mod = pipeline(mod)

# Level 2: Standard optimizations (default)
with transform.PassContext(opt_level=2):
    mod = pipeline(mod)

# Level 3: Aggressive optimizations
with transform.PassContext(opt_level=3):
    mod = pipeline(mod)

# Level 4: Maximum optimization, may increase compile time significantly
with transform.PassContext(opt_level=4):
    mod = pipeline(mod)
```

### 3.4.5 Required and Disabled Passes

```python
# Require specific passes to be present in the pipeline
with transform.PassContext(
    required_pass=["tirx.UnrollLoop", "tirx.Simplify"]
):
    # Error if the pipeline does not include these passes
    mod = pipeline(mod)

# Disable specific passes
with transform.PassContext(
    disabled_pass=["tirx.UnrollLoop", "tirx.VectorizeLoop"]
):
    # These passes will be skipped even if included in Sequential
    mod = pipeline(mod)
```

### 3.4.6 Per-Pass Configuration

Each pass can define its own configuration keys. These are specified in the `config` dictionary with the pass name as the key prefix:

```python
# Configure TIR UnrollLoop pass
with transform.PassContext(
    config={
        "tirx.UnrollLoop": {
            "auto_max_step": 10,      # max unrolled stmts in auto mode
            "auto_max_depth": 8,      # max loop nesting depth
            "auto_max_extent": 0,     # max loop extent (0 = no limit)
            "explicit_unroll": True,  # generate explicit unrolled code
        }
    }
):
    mod = tirx.transform.UnrollLoop()(mod)

# Configure multiple passes simultaneously
with transform.PassContext(
    config={
        "tirx.UnrollLoop": {"auto_max_step": 16},
        "tirx.VectorizeLoop": {"enable_vectorize": True},
        "relax.transform.FuseOps": {"fuse_max_depth": 5},
        "tirx.transform.BindTarget": {"target": "cuda"},
    }
):
    mod = pipeline(mod)
```

### 3.4.7 Nested PassContext

PassContext supports nesting. Inner contexts override outer contexts for the specified keys:

```python
with transform.PassContext(opt_level=2):
    # opt_level=2 applies here

    with transform.PassContext(
        opt_level=0,
        disabled_pass=["tirx.UnrollLoop"]
    ):
        # opt_level=0 and UnrollLoop disabled apply here
        mod = sub_pipeline(mod)

    # Back to opt_level=2
    mod = another_pipeline(mod)
```

### 3.4.8 PassContext.current()

You can retrieve the current PassContext from anywhere in the call stack:

```python
def my_custom_pass(mod):
    ctx = transform.PassContext.current()
    print(f"Current opt level: {ctx.opt_level}")
    print(f"Config: {ctx.config}")

    # Check for a specific configuration
    if "tirx.UnrollLoop" in ctx.config:
        unroll_config = ctx.config["tirx.UnrollLoop"]
        max_step = unroll_config.get("auto_max_step", 0)

    return mod
```

### 3.4.9 Pass Instrumentation

PassContext supports instrumentation hooks that run before and after each pass:

```python
class TimingInstrument(transform.PassInstrument):
    """Instrument that times each pass execution."""

    def __init__(self):
        super().__init__()
        self.timings = {}

    def run_before_pass(self, mod, info):
        self._start = time.time()

    def run_after_pass(self, mod, info):
        elapsed = time.time() - self._start
        self.timings[info.name] = elapsed
        print(f"Pass {info.name} took {elapsed:.3f}s")

instrument = TimingInstrument()

with transform.PassContext(instruments=[instrument]):
    mod = pipeline(mod)

print("All timings:", instrument.timings)
```

---

## 3.5 Pass Base Classes

### 3.5.1 Pass Hierarchy

All passes in TVM inherit from `tvm.transform.Pass`. The hierarchy is:

```
tvm.transform.Pass (abstract base)
  +-- tvm.transform.ModulePass
  +-- tvm.transform.FunctionPass (renamed in TIR: tirx.transform.PrimFuncPass)
  +-- tvm.transform.Sequential
```

### 3.5.2 tvm.transform.Pass

The abstract base class for all passes. It defines the common interface:

```python
class Pass:
    @property
    def info(self):
        """PassMeta: metadata about this pass."""
        ...

    def __call__(self, mod):
        """Execute the pass on the given IRModule.

        Parameters
        ----------
        mod : IRModule
            The input module.

        Returns
        -------
        IRModule
            The transformed module.
        """
        ...
```

Every pass returns a new IRModule (passes are semantically immutable transformations). The pass may read the current `PassContext` to configure its behavior.

### 3.5.3 tvm.transform.ModulePass

A `ModulePass` transforms the entire IRModule as a unit. The transformation function receives the full module and can add, remove, or modify any function within it.

```python
from tvm import transform
from tvm.ir import IRModule

def my_module_transform(mod: IRModule, ctx: transform.PassContext) -> IRModule:
    """Custom module-level transformation."""
    # Examine all functions
    for gv in mod.get_global_vars():
        func = mod[gv]
        # Perform analysis or transformation
        # ...
    # Return a new (or the same) module
    return mod

# Create and apply the module pass
my_pass = transform.module_pass(my_module_transform, opt_level=0, name="my_module_pass")
new_mod = my_pass(mod)
```

Built-in module passes include:
- `relax.transform.LegalizeOps` -- lower Relax ops to TIR
- `relax.transform.FuseOps` -- operator fusion
- `relax.transform.FuseOpsByPattern` -- pattern-based fusion
- `relax.transform.DecomposeOpsForTarget` -- target-specific op decomposition
- `tirx.transform.BindTarget` -- bind target information to TIR functions
- `tirx.transform.LowerAutoCopy` -- lower auto-copy annotations to explicit copy

### 3.5.4 tvm.transform.FunctionPass

A `FunctionPass` applies a transformation to each function in the module independently. The pass infrastructure handles iteration; the user only provides a per-function callback.

```python
from tvm import transform, relax

def my_func_transform(func, mod, ctx):
    """Transform a single function."""
    # Only transform Relax functions
    if isinstance(func, relax.Function):
        # ... transform func ...
        return transformed_func
    return func

# Create the function pass
my_pass = transform.function_pass(
    my_func_transform,
    opt_level=0,
    name="my_func_pass",
)

new_mod = my_pass(mod)
```

The function pass callback receives:
- `func`: The current function being transformed
- `mod`: The full IRModule (for context, e.g., looking up called functions)
- `ctx`: The current PassContext

Built-in function passes include:
- `tirx.transform.Simplify` -- algebraic simplification of TIR expressions
- `tirx.transform.UnrollLoop` -- loop unrolling
- `tirx.transform.VectorizeLoop` -- vectorize inner loops
- `tirx.transform.StorageRewrite` -- storage allocation optimization
- `tirx.transform.LowerThreadAllreduce` -- lower cross-thread reductions

### 3.5.5 tvm.transform.Sequential

`Sequential` composes multiple passes into a pipeline. Passes are executed in order, and each pass receives the output of the previous pass:

```python
from tvm import transform
from tvm import relax, tirx

# Define a compilation pipeline
pipeline = transform.Sequential(
    passes=[
        # Relax-level passes
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),

        # TIR-level passes
        tirx.transform.Simplify(),
        tirx.transform.StorageRewrite(),
        tirx.transform.UnrollLoop(),
        tirx.transform.VectorizeLoop(),
        tirx.transform.LowerAutoCopy(),
    ],
    opt_level=3,
    name="my_compilation_pipeline",
)

# Apply the pipeline
with transform.PassContext(opt_level=3):
    optimized_mod = pipeline(mod)
```

Sequential also supports conditional pass execution:

```python
# Passes that are disabled in PassContext will be skipped
pipeline = transform.Sequential([
    relax.transform.LegalizeOps(),
    tirx.transform.UnrollLoop(),  # can be disabled via PassContext
    tirx.transform.VectorizeLoop(),  # can be disabled via PassContext
])

with transform.PassContext(disabled_pass=["tirx.UnrollLoop"]):
    mod = pipeline(mod)  # UnrollLoop is skipped
```

### 3.5.6 Custom Pass Registration

You can register custom passes so they integrate with the pass infrastructure:

```python
from tvm import transform

@transform.module_pass(opt_level=2, name="my_custom_pass")
def my_custom_module_pass(mod, ctx):
    """A custom module-level pass."""
    # Implementation
    return new_mod

# Now it can be used in Sequential
pipeline = transform.Sequential([
    my_custom_module_pass,
    relax.transform.LegalizeOps(),
])
```

### 3.5.7 Pass Metadata

Each pass carries metadata that describes it:

```python
# Access pass metadata
pass_info = my_pass.info
print(pass_info.name)        # pass name string
print(pass_info.opt_level)   # required opt_level
print(pass_info.required)    # list of required prerequisite passes
```

---

## 3.6 Accessing IR from Python

### 3.6.1 Object Model

All IR nodes in TVM are subclasses of `runtime.Object`. This base class provides:
- Reference counting (garbage collection via `tvm.runtime.ObjectRef`)
- Field access by name
- Structural equality comparison (`tvm.ir.structural_equal`)
- Hashing (`tvm.ir.structural_hash`)
- Serialization

```python
import tvm
from tvm import tirx, relax, ir

# All IR nodes inherit from tvm.runtime.Object
x = tirx.Var("x", "int32")
assert isinstance(x, tvm.runtime.ObjectRef)

# IRModule is also an Object
mod = tvm.IRModule()
assert isinstance(mod, tvm.runtime.ObjectRef)
```

### 3.6.2 Field Access by Name

Any field of an IR node can be accessed by its attribute name. This is a fundamental capability for writing analysis and transformation passes in Python:

```python
from tvm import tirx

# Create a binary expression: x + x
x = tirx.Var("x", "int32")
y = tirx.Add(x, x)

# Access the fields of the Add node
assert y.a == x   # left operand
assert y.b == x   # right operand
assert y.a.name == "x"
assert y.a.dtype == "int32"

# Create a more complex expression: (x + 1) * 2
one = tirx.IntImm("int32", 1)
two = tirx.IntImm("int32", 2)
add_expr = tirx.Add(x, one)
mul_expr = tirx.Mul(add_expr, two)

# Traverse the expression tree
assert mul_expr.a == add_expr
assert mul_expr.b == two
assert mul_expr.a.a == x
assert mul_expr.a.b == one
```

### 3.6.3 Common IR Node Types and Their Fields

```python
from tvm import tirx, ir

# --- tirx.Var ---
var = tirx.Var("my_var", "float32")
print(var.name)    # "my_var"
print(var.dtype)   # "float32"
print(var.type_annotation)  # PrimType("float32")

# --- tirx.Buffer ---
buf = tirx.Buffer(
    shape=(128, 128),
    dtype="float32",
    name="A",
)
print(buf.shape)    # [128, 128]
print(buf.dtype)    # "float32"
print(buf.name)     # "A"
print(buf.data)     # the backing Var

# --- tirx.For ---
loop_var = tirx.Var("i", "int32")
body = tirx.Evaluate(tirx.IntImm("int32", 0))  # placeholder
for_node = tirx.For(
    loop_var=loop_var,
    min_val=0,
    extent=128,
    kind=tirx.ForKind.SERIAL,
    body=body,
)
print(for_node.loop_var)   # i
print(for_node.min_val)    # 0
print(for_node.extent)     # 128
print(for_node.kind)       # ForKind.SERIAL

# --- tirx.BufferStore ---
store = tirx.BufferStore(buf, tirx.FloatImm("float32", 1.0), [0, 0])
print(store.buffer)   # the buffer
print(store.value)    # 1.0
print(store.indices)  # [0, 0]

# --- tirx.Call ---
call = tirx.call_extern("float32", "my_func", tirx.FloatImm("float32", 3.14))
print(call.dtype)       # "float32"
print(call.op)          # Op("tirx.call_extern") or similar
print(call.args)        # [3.14]
```

### 3.6.4 Relax IR Node Access

```python
from tvm import relax

# Accessing Relax function fields
@R.function
def my_func(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
    with R.dataflow():
        lv = R.call_tir("add", (x, x), out_sinfo=R.Tensor((128, 128), "float32"))
        R.output(lv)
    return lv

# Access function parameters
params = my_func.params
print(params[0].name_hint)  # "x"

# Access function body
body = my_func.body
print(type(body))  # relax.SeqExpr (typically)

# Access StructInfo
sinfo = my_func.ret_struct_info
print(sinfo)  # TensorStructInfo(shape=(128, 128), dtype="float32")
```

### 3.6.5 Structural Equality

TVM provides `structural_equal` and `structural_hash` for comparing IR nodes by value (not by identity):

```python
from tvm.ir import structural_equal, structural_hash
from tvm import tirx

# Two independently created but structurally identical nodes
x1 = tirx.Var("x", "int32")
x2 = tirx.Var("x", "int32")

# Identity comparison: False (different objects)
assert x1 is not x2

# Structural comparison: True (same structure)
assert structural_equal(x1, x2)

# Structural hash
assert structural_hash(x1) == structural_hash(x2)

# More complex structural equality
expr1 = tirx.Add(tirx.IntImm("int32", 1), tirx.IntImm("int32", 2))
expr2 = tirx.Add(tirx.IntImm("int32", 1), tirx.IntImm("int32", 2))
assert structural_equal(expr1, expr2)

# Structural inequality
expr3 = tirx.Add(tirx.IntImm("int32", 2), tirx.IntImm("int32", 1))
assert not structural_equal(expr1, expr3)
```

### 3.6.6 IR Node Visitor Pattern

TVM provides visitor infrastructure for traversing IR nodes:

```python
from tvm import tirx

class MyVisitor(tirx.StmtVisitor):
    """Visit all For loops in a TIR PrimFunc."""

    def __init__(self):
        super().__init__()
        self.loop_count = 0

    def visit_for(self, stmt):
        self.loop_count += 1
        print(f"Found loop: {stmt.loop_var.name} in [0, {stmt.extent})")
        self.visit_stmt(stmt.body)  # recurse into body

# Usage
visitor = MyVisitor()
for gv in mod.get_global_vars():
    func = mod[gv]
    if isinstance(func, tirx.PrimFunc):
        visitor.visit(func.body)
        print(f"Total loops in {gv.name_hint}: {visitor.loop_count}")
```

### 3.6.7 IR Node Mutation

TVM provides mutator infrastructure for transforming IR nodes. Mutators create new nodes with modified fields while preserving the structure of unchanged subtrees:

```python
from tvm import tirx

class ConstantFolder(tirx.StmtMutator):
    """Fold constant expressions in TIR statements."""

    def visit_expr(self, expr):
        expr = super().visit_expr(expr)
        if isinstance(expr, tirx.Add):
            if isinstance(expr.a, tirx.IntImm) and isinstance(expr.b, tirx.IntImm):
                return tirx.IntImm(expr.dtype, expr.a.value + expr.b.value)
        elif isinstance(expr, tirx.Mul):
            if isinstance(expr.a, tirx.IntImm) and isinstance(expr.b, tirx.IntImm):
                return tirx.IntImm(expr.dtype, expr.a.value * expr.b.value)
        return expr

# Usage
folder = ConstantFolder()
new_body = folder.visit(func.body)
new_func = func.with_body(new_body)
```

---

## 3.7 Serialization

### 3.7.1 Overview

TVM's serialization system can convert any IR node to JSON and load it back. This enables:
- Saving compiled modules to disk
- Distributing pre-compiled models
- Inspecting IR state for debugging
- Caching compilation results

### 3.7.2 Serializing to JSON

```python
import tvm
import json

# Create an IRModule
@I.ir_module
class MyModule:
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

mod = MyModule

# Serialize to JSON string
json_str = tvm.ir.save_json(mod)
print(json_str[:500])  # preview the JSON

# The JSON can be saved to a file
with open("module.json", "w") as f:
    f.write(json_str)
```

### 3.7.3 Deserializing from JSON

```python
# Load from JSON string
with open("module.json", "r") as f:
    json_str = f.read()

loaded_mod = tvm.ir.load_json(json_str)
assert isinstance(loaded_mod, tvm.IRModule)

# Verify structural equality with original
assert tvm.ir.structural_equal(mod, loaded_mod)
```

### 3.7.4 Serializing Individual IR Nodes

Any IR node can be serialized, not just IRModules:

```python
from tvm import tirx

# Serialize a TIR expression
x = tirx.Var("x", "int32")
expr = tirx.Add(x, tirx.IntImm("int32", 1))
json_str = tvm.ir.save_json(expr)

# Deserialize
loaded_expr = tvm.ir.load_json(json_str)
assert tvm.ir.structural_equal(expr, loaded_expr)

# Serialize a type
from tvm import ir
tensor_type = ir.TensorType((128, 128), "float32")
json_str = tvm.ir.save_json(tensor_type)
loaded_type = tvm.ir.load_json(json_str)
assert tvm.ir.structural_equal(tensor_type, loaded_type)
```

### 3.7.5 Binary Serialization (for Deployment)

For deployment, TVM uses a more compact binary format:

```python
# Build and export a runtime module (compiled binary)
target = tvm.target.Target("llvm")
exec_mod = relax.build(mod, target)

# Export to binary
binary = exec_mod.export_library("model.so")

# Or export to a more portable format
exec_mod.export_library(
    "model.tar",
    fmt="so",  # shared object format
)
```

### 3.7.6 Serialization Considerations

When serializing IR nodes, be aware of these limitations and best practices:

1. **External references**: `GlobalVar` references are serialized by name. When deserializing, the target IRModule must contain the referenced function.
2. **Target annotations**: Target information is serialized but may not be portable across different compilation environments.
3. **Op registry**: Deserialized Ops must be registered in the running TVM instance. Custom ops require registration before deserialization.
4. **Large modules**: For very large IRModules (millions of operations), JSON serialization can be slow and produce large files. Binary serialization is preferred for deployment.

---

## 3.8 IRModule Creation Patterns

### 3.8.1 From TVMScript

The most common and readable way to create an IRModule is using TVMScript decorators:

```python
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class MyModule:
    @T.prim_func
    def matmul(
        A: T.Buffer((64, 64), "float32"),
        B: T.Buffer((64, 64), "float32"),
        C: T.Buffer((64, 64), "float32"),
    ) -> None:
        for i, j, k in T.grid(64, 64, 64):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(
        x: R.Tensor((64, 64), "float32"),
        w: R.Tensor((64, 64), "float32"),
    ) -> R.Tensor((64, 64), "float32"):
        with R.dataflow():
            lv = R.call_tir(
                MyModule.matmul,
                (x, w),
                out_sinfo=R.Tensor((64, 64), "float32"),
            )
            R.output(lv)
        return lv

mod = MyModule  # This is already an IRModule
print(type(mod))  # <class 'tvm.ir.module.IRModule'>
```

### 3.8.2 From Relax Frontend

The Relax frontend can convert models from popular frameworks directly into IRModules:

```python
import tvm
from tvm import relax
from tvm.relax.frontend import nn

# Define a model using relax.nn (similar to PyTorch)
class MLPModel(nn.Module):
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

# Export to IRModule
model = MLPModel()
mod, params = model.export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
# mod is now an IRModule with a "forward" Relax function
# params is a dict of parameter names to NDArray values
```

### 3.8.3 From PyTorch via TVM Frontend

```python
import torch
import torch.nn as nn
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Define a PyTorch model
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

torch_model = TorchMLP()
torch_model.eval()

# Export using torch.export
example_input = torch.randn(1, 784)
exported_program = torch.export.export(torch_model, (example_input,))

# Convert to TVM IRModule
mod = from_exported_program(exported_program)
# mod is now an IRModule with Relax functions
```

### 3.8.4 From ONNX

```python
import tvm
from tvm import relax

# Load an ONNX model
import onnx
onnx_model = onnx.load("model.onnx")

# Convert to IRModule using the ONNX frontend
from tvm.relax.frontend.onnx import from_onnx
mod = from_onnx(onnx_model)
# mod is now an IRModule with Relax functions representing the ONNX model
```

### 3.8.5 From Tensor Expression (TE)

Tensor Expression is TVM's older API for defining computations. TE-created schedules can be lowered into TIR PrimFuncs within an IRModule:

```python
import tvm
from tvm import te

# Define computation using Tensor Expression
n = te.var("n")
A = te.placeholder((n,), name="A", dtype="float32")
B = te.placeholder((n,), name="B", dtype="float32")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

# Create a schedule
s = te.create_schedule(C.op)

# Lower to IRModule (containing TIR PrimFuncs)
mod = tvm.lower(s, [A, B, C], name="add_func")
print(type(mod))  # <class 'tvm.ir.module.IRModule'>

# The module contains a TIR PrimFunc
print(mod["add_func"])
```

### 3.8.6 Manual Construction

For advanced use cases, IRModules can be constructed programmatically:

```python
import tvm
from tvm import tirx, relax, ir

# Build a TIR PrimFunc manually
def build_add_primfunc():
    n = tirx.Var("n", "int64")
    A = tirx.decl_buffer((n,), "float32", name="A")
    B = tirx.decl_buffer((n,), "float32", name="B")
    C = tirx.decl_buffer((n,), "float32", name="C")
    i = tirx.Var("i", "int64")

    body = tirx.For(
        loop_var=i,
        min_val=0,
        extent=n,
        kind=tirx.ForKind.SERIAL,
        body=tirx.BufferStore(C, A[i] + B[i], [i]),
    )

    func = tirx.PrimFunc(
        params=[A, B, C],
        body=body,
        ret_type=None,
        buffer_map={},
    )
    return func

# Build a Relax function manually
def build_main_relax_func():
    from tvm.relax import expr as rx_expr

    n = tirx.Var("n", "int64")
    x = relax.Var("x", relax.TensorStructInfo((n,), "float32"))
    w = relax.Var("w", relax.TensorStructInfo((n,), "float32"))

    gv = relax.GlobalVar("add_func")
    call_node = relax.call_tir(
        gv,
        [x, w],
        out_sinfo=relax.TensorStructInfo((n,), "float32"),
    )

    func = relax.Function(
        params=[x, w],
        body=call_node,
        ret_struct_info=relax.TensorStructInfo((n,), "float32"),
    )
    return func

# Assemble the IRModule
prim_func = build_add_primfunc()
relax_func = build_main_relax_func()

mod = tvm.IRModule({"add_func": prim_func, "main": relax_func})
print(mod)
```

### 3.8.7 Combining IRModules

IRModules can be merged, enabling modular compilation:

```python
# Two separate IRModules
mod_a = tvm.IRModule({"func_a": func_a})
mod_b = tvm.IRModule({"func_b": func_b})

# Merge modules
combined = tvm.IRModule.merge(mod_a, mod_b)
# combined now contains both func_a and func_b

# Alternatively, copy functions from one module to another
for gv in mod_b.get_global_vars():
    combined[gv.name_hint] = mod_b[gv]
```

### 3.8.8 IRModule from Relax Expressions

```python
from tvm import relax

# Create a simple Relax expression
x = relax.Var("x", relax.TensorStructInfo((128,), "float32"))
expr = relax.op.add(x, x)

# Wrap in a function
func = relax.Function(
    params=[x],
    body=expr,
    ret_struct_info=relax.TensorStructInfo((128,), "float32"),
    name="main",
)

# Create IRModule from function
mod = tvm.IRModule.from_expr(func)
# or equivalently:
mod = tvm.IRModule({"main": func})
```

---

## 3.9 Diagnostic and Debugging

### 3.9.1 Printing IR

The `__str__` method of IR nodes produces TVMScript-formatted output:

```python
# Print the entire IRModule
print(mod)

# Print a specific function
print(mod["main"])

# Pretty-print an expression
from tvm import tirx
x = tirx.Var("x", "int32")
expr = tirx.Add(x, tirx.IntImm("int32", 1))
print(expr)  # x + 1
```

### 3.9.2 Show with TVMScript

For a more structured view, use the `show` method:

```python
# Show full module with syntax highlighting (in interactive environments)
mod.show()

# Show a specific function
mod["main"].show()
```

### 3.9.3 Inspecting Pass Effects

To understand how a pass transforms the IRModule, print before and after:

```python
from tvm import transform

# Before
print("=== Before ===")
print(mod)

# Apply a pass
new_mod = relax.transform.LegalizeOps()(mod)

# After
print("=== After ===")
print(new_mod)
```

### 3.9.4 Pass Debug Instrumentation

For detailed pass debugging, use the `PrintIR` pass or custom instrumentation:

```python
from tvm import transform

# Insert PrintIR between passes for debugging
pipeline = transform.Sequential([
    relax.transform.LegalizeOps(),
    transform.PrintIR(name="after_legalize", show_meta_data=False),
    relax.transform.FuseOps(),
    transform.PrintIR(name="after_fuse", show_meta_data=False),
])

with transform.PassContext(opt_level=3):
    new_mod = pipeline(mod)
```

### 3.9.5 Structural Difference

To compare two IRModules structurally:

```python
from tvm.ir import structural_equal

# Check if two modules are structurally identical
are_equal = structural_equal(mod_a, mod_b)
print(f"Modules are structurally equal: {are_equal}")

# For more detailed comparison, use assert_structural_equal
from tvm.testing.utils import assert_structural_equal
try:
    assert_structural_equal(mod_a, mod_b)
except AssertionError as e:
    print(f"Difference found: {e}")
```

---

## 3.10 Advanced Topics

### 3.10.1 IRBuilder

The `IRBuilder` provides a builder pattern for constructing IR programmatically. It is particularly useful for meta-programming and code generation:

```python
from tvm.script import tirx as T, ir as I
from tvm import tirx

# Using IRBuilder for TIR
with tirx.IRBuilder() as builder:
    A = tirx.decl_buffer((128,), "float32", name="A")
    B = tirx.decl_buffer((128,), "float32", name="B")
    with tirx.grid(128) as i:
        with tirx.sblock("B"):
            B[i] = A[i] * 2.0

# Get the built statement
stmt = builder.get()
```

### 3.10.2 Span and Source Location

IR nodes can carry source location information (`Span`) for debugging:

```python
from tvm import tirx, ir

# Create a variable with source location
span = ir.Span(ir.SourceName("my_file.py"), line=42, column=10, end_line=42, end_column=15)
x = tirx.Var("x", "int32", span=span)

# Access span information
if x.span is not None:
    print(f"Defined at {x.span.source_name.name}:{x.span.line}")
```

### 3.10.3 Attributes and DictAttrs

IR nodes can carry named attributes:

```python
from tvm import ir

# Create DictAttrs
attrs = ir.DictAttrs({"pragma_unroll": True, "vectorize_length": 4})

# Attributes on PrimFunc
func = tirx.PrimFunc(
    params=[...],
    body=...,
    attrs=attrs,
)
```

### 3.10.4 GlobalVar and Function References

When one function calls another, it uses `GlobalVar` as the reference:

```python
from tvm import relax

# GlobalVar is how functions refer to each other within an IRModule
gv = relax.GlobalVar("my_function")

# In a Relax call_tir:
# R.call_tir(MyModule.my_function, ...) uses GlobalVar("my_function")

# Get GlobalVar from a module
gv_main = mod.get_global_var("main")
print(gv_main.name_hint)  # "main"
```

### 3.10.5 IRModule Immutability

IRModules are designed to be immutable in the sense that passes return new modules rather than modifying existing ones. However, the Python API provides some mutation methods for convenience during construction:

```python
# "Adding" a function creates a new module (or modifies in-place during construction)
mod["new_func"] = some_func

# For production pass writing, use update_func:
from tvm.ir import IRModule
mod = mod.update_func(mod.get_global_var("main"), new_main_func)

# The remove method returns a new module
mod = mod.remove("unused_func")
```

### 3.10.6 Memory and Performance Considerations

When working with large IRModules (common in production ML models):

1. **Use structural_equal sparingly**: Structural equality comparison traverses the entire IR tree. For identity checks, use `is`.
2. **Avoid unnecessary copying**: Passes return new modules, but unchanged functions are shared by reference.
3. **Lazy construction**: When building large modules, construct functions independently and assemble at the end.
4. **Serialization overhead**: JSON serialization of very large modules can be slow. Consider binary formats for production use.

```python
# Good: Build functions independently
func_a = build_func_a()
func_b = build_func_b()
mod = tvm.IRModule({"a": func_a, "b": func_b})

# Avoid: Repeatedly modifying a growing module
# Each addition may trigger internal bookkeeping
mod = tvm.IRModule()
for name, func in all_functions:
    mod[name] = func  # OK for small counts, but prefer batch construction
```
