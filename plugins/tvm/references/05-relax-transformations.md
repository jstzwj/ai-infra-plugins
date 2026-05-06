# Apache TVM — Chapter 5: Relax Transformations

This reference covers the `relax.transform` module in Apache TVM, which contains all graph-level transformation passes that operate on Relax IR. These passes form the backbone of the Relax compilation pipeline, converting high-level operator graphs into optimized, lowerable representations. The passes are composable, meaning they can be chained together in arbitrary sequences to build complete compilation pipelines.

---

## 5.1 Overview of Relax Transformations

### The Transformation Architecture

Relax transformations follow the pass manager infrastructure inherited from TVM's core IR framework. Each transformation pass is a function that takes an `IRModule` as input and returns a transformed `IRModule`. Passes are registered with the TVM global registry and can be invoked either individually or as part of a pipeline.

```python
import tvm
from tvm import relax

# Individual pass invocation
mod = relax.transform.LegalizeOps()(mod)

# Pipeline composition with Sequential
pipeline = relax.transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.DeadCodeElimination(),
])
mod = pipeline(mod)
```

### Pass Categories

Relax transformation passes fall into several broad categories:

| Category | Purpose | Key Passes |
|----------|---------|------------|
| Legalization | Lower high-level ops to TIR | `LegalizeOps` |
| Fusion | Combine operators for efficiency | `FuseOps`, `FuseOpsByPattern`, `FuseTIR` |
| Decomposition | Break complex ops into simpler ones | `DecomposeOpsForInference`, `DecomposeOpsForTraining` |
| Canonicalization | Standardize IR representation | `CanonicalizeBindings`, `Normalize` |
| Optimization | Simplify and optimize expressions | `FoldConstant`, `SimplifyExpr`, `DeadCodeElimination` |
| Lowering | Convert to lower-level forms | `ToNonDataflow`, `VMBuiltinLower`, `ComputePrimValue` |
| Backend Integration | Interface with external backends | `RunCodegen`, `BackendDispatch`, `AttachGlobalSymbol` |
| Parameter Management | Handle model parameters | `LiftTransformParams`, `BundleModelParams` |

### Pass Invariants

All Relax transformation passes preserve certain invariants:

1. **IRModule validity**: The output is always a valid IRModule.
2. **Semantic equivalence**: The transformation preserves the meaning of the program (except for `Gradient`, which computes a new derivative program).
3. **Composability**: Passes can be applied in any order, though some orderings produce better results than others.

---

## 5.2 LegalizeOps

### Overview

`relax.transform.LegalizeOps` is one of the most critical passes in the Relax pipeline. It translates high-level Relax operators (such as `relax.op.add`, `relax.op.matmul`, `relax.op.conv2d`) into low-level TIR `PrimFunc` implementations. Each legalized operator results in a new `tir.PrimFunc` added to the IRModule, and the original Relax operator call is replaced with a `R.call_tir` invocation that calls the generated PrimFunc.

```python
import tvm
from tvm import relax

# Before legalization: high-level Relax ops
@tvm.script.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32"),
             w: R.Tensor((784, 10), dtype="float32")):
        with R.dataflow():
            lv1 = R.matmul(x, w)
            lv2 = R.nn.relu(lv1)
            R.output(lv2)
        return lv2

# Apply legalization
mod = relax.transform.LegalizeOps()(Module)
# After legalization: R.matmul and R.nn.relu replaced with R.call_tir
# New tir.PrimFunc entries added for fused_matmul and relu computations
```

### How Legalization Works

The legalization process follows these steps for each operator in the Relax function:

1. **Operator lookup**: The pass looks up the legalization strategy for the specific operator from a registry of legalization functions.
2. **TE computation creation**: A Tensor Expression (TE) computation is created that implements the operator semantics.
3. **TE-to-TIR lowering**: The TE computation is lowered to a `tir.PrimFunc` using TVM's TE compiler.
4. **IRModule insertion**: The new PrimFunc is added to the IRModule with a unique name (e.g., `fused_matmap`).
5. **Call replacement**: The original `relax.op.*` call is replaced with `R.call_tir(new_prim_func, args, out_sinfo)`.

### Custom Legalization

You can register custom legalization functions for operators:

```python
from tvm import relax, te, tir
from tvm.relax.transform import LegalizeOps

# Register a custom legalization for a user-defined op
@relax.op.register_legalize("custom.my_op")
def legalize_my_op(attrs, inputs, types):
    """Custom legalization function.

    Args:
        attrs: Operator attributes
        inputs: Input tensors
        types: Input types

    Returns:
        A TE computation or a Relax expression
    """
    x = inputs[0]
    return te.compute(
        x.shape,
        lambda i, j: te.exp(x[i, j]) + 1.0,
        name="custom_my_op"
    )
```

### Selective Legalization

You can control which operators are legalized by specifying a custom legalizer map:

```python
# Only legalize specific operators
mod = relax.transform.LegalizeOps(
    legalize_map={"relax.matmul": my_custom_matmul_legalizer}
)(mod)
```

### LegalizeOps and Custom PrimFunc

In some cases, you may want to provide a hand-written TIR PrimFunc instead of relying on TE lowering. You can do this by having the legalization function return a `R.call_tir` with a pre-defined PrimFunc:

```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def custom_matmul(A: T.Buffer((128, 128), "float32"),
                      B: T.Buffer((128, 128), "float32"),
                      C: T.Buffer((128, 128), "float32")):
        for i, j, k in T.grid(128, 128, 128):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(x: R.Tensor((128, 128), "float32"),
             w: R.Tensor((128, 128), "float32")):
        with R.dataflow():
            lv = R.call_tir(custom_matmul, (x, w),
                            out_sinfo=R.Tensor((128, 128), "float32"))
            R.output(lv)
        return lv
```

---

## 5.3 Operator Fusion

### 5.3.1 FuseOps

`relax.transform.FuseOps` performs graph-level fusion of consecutive Relax operators. It analyzes the data dependency graph and groups operators that can be executed together in a single kernel into fused sub-graphs.

```python
import tvm
from tvm import relax

# Apply operator fusion
mod = relax.transform.FuseOps()(mod)
```

**Fusion algorithm:**

The `FuseOps` pass uses a post-dominator tree analysis to determine fusion opportunities:

1. Build a dataflow graph from the Relax function.
2. Compute the post-dominator tree.
3. Identify fusion groups based on operator patterns (element-wise, injective, reduction, opaque).
4. Group operators into fused sub-graphs.
5. Create new Relax functions for each fused group.

**Fusion rules:**

| Producer Pattern | Consumer Pattern | Can Fuse? |
|-----------------|------------------|-----------|
| Element-wise | Element-wise | Yes |
| Element-wise | Broadcast | Yes |
| Broadcast | Element-wise | Yes |
| Element-wise | Reduction | Yes |
| Reduction | Element-wise | No (by default) |
| Opaque | Any | No |

```python
# Example showing fusion before and after
@tvm.script.ir_module
class BeforeFusion:
    @R.function
    def main(x: R.Tensor((1, 128), "float32")):
        with R.dataflow():
            lv1 = R.multiply(x, R.const(2.0))
            lv2 = R.add(lv1, R.const(1.0))
            lv3 = R.nn.relu(lv2)
            R.output(lv3)
        return lv3

# After FuseOps, lv1, lv2, lv3 are fused into a single function
mod_fused = relax.transform.FuseOps()(BeforeFusion)
```

### 5.3.2 FuseTIR

`relax.transform.FuseTIR` operates at the TIR level. After legalization, multiple `R.call_tir` calls may exist that could be combined. `FuseTIR` merges these TIR PrimFunc functions into single fused PrimFuncs, eliminating the overhead of storing and loading intermediate results.

```python
# Apply TIR-level fusion after legalization
mod = relax.transform.FuseTIR()(mod)
```

**When to use FuseTIR:**

`FuseTIR` is typically applied after `LegalizeOps` and before MetaSchedule auto-tuning. It is especially effective when:

- Multiple consecutive `R.call_tir` calls operate on the same data tiles.
- Intermediate tensors between `call_tir` boundaries can be kept in registers or shared memory.
- The target hardware benefits from larger fused kernels (e.g., GPUs).

```python
# Typical pipeline ordering
pipeline = relax.transform.Sequential([
    relax.transform.LegalizeOps(),       # High-level ops -> TIR
    relax.transform.FuseTIR(),            # Fuse consecutive TIR functions
    relax.transform.DeadCodeElimination(), # Clean up
])
```

**FuseTIR internals:**

The pass works by:

1. Analyzing `R.call_tir` call chains in each Relax function.
2. Checking that callee PrimFuncs have compatible loop structures and memory access patterns.
3. Merging the loop nests of compatible PrimFuncs into a single fused PrimFunc.
4. Replacing the original `call_tir` chain with a single `call_tir` to the fused function.

```python
# Example: Two separate TIR functions become one
# Before FuseTIR:
#   lv1 = R.call_tir(fused_matmul, (x, w), ...)
#   lv2 = R.call_tir(fused_bias_add, (lv1, b), ...)
#   lv3 = R.call_tir(fused_relu, (lv2,), ...)

# After FuseTIR:
#   lv3 = R.call_tir(fused_matmul_bias_add_relu, (x, w, b), ...)
```

### 5.3.3 FuseOpsByPattern

`relax.transform.FuseOpsByPattern` provides pattern-based fusion using DPL (Domain Pattern Language) patterns. Unlike `FuseOps`, which relies on post-dominator analysis, `FuseOpsByPattern` uses user-defined patterns to match and group operators.

```python
from tvm.relax.transform import FuseOpsByPattern
from tvm.relax.dpl import PatternContext

# Define fusion patterns
# Pattern: matmul + bias_add + relu
matmul_pat = is_op("relax.matmul")(wildcard(), wildcard())
bias_pat = is_op("relax.add")(matmul_pat, wildcard())
relu_pat = is_op("relax.nn.relu")(bias_pat)

# Create FusionPattern
from tvm.relax.transform import FusionPattern
pattern = FusionPattern("fused_matmul_bias_relu", relu_pat)

# Apply pattern-based fusion
mod = FuseOpsByPattern(
    patterns=[pattern],
    bind_constants=False,
    annotate_codegen=True
)(mod)
```

**Key parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patterns` | `List[FusionPattern]` | Required | List of fusion patterns to match |
| `bind_constants` | `bool` | `False` | Whether to bind constants into fused functions |
| `annotate_codegen` | `bool` | `False` | Annotate fused functions for external codegen dispatch |
| `check` | `Callable` | `None` | Optional check function for matched patterns |

**FusionPattern construction:**

```python
from tvm.relax.transform import FusionPattern
from tvm.relax.dpl import *

# Simple pattern: single operator
relu_pattern = FusionPattern(
    name="fused_relu",
    pattern=is_op("relax.nn.relu")(wildcard())
)

# Composite pattern: conv2d + batch_norm + relu
conv_pat = is_op("relax.nn.conv2d")(wildcard(), wildcard())
bn_pat = is_op("relax.nn.batch_norm")(conv_pat, wildcard(), wildcard(),
                                       wildcard(), wildcard())
relu_pat = is_op("relax.nn.relu")(bn_pat)

conv_bn_relu_pattern = FusionPattern(
    name="fused_conv_bn_relu",
    pattern=relu_pat
)

# Pattern with attribute constraints
conv2d_pat = is_op("relax.nn.conv2d")(wildcard(), wildcard()).has_attr({
    "data_layout": "NCHW",
    "kernel_layout": "OIHW"
})
```

**External backend dispatch with annotate_codegen:**

When `annotate_codegen=True`, matched patterns are wrapped into composite functions that can be dispatched to external backends (e.g., CUTLASS, TensorRT):

```python
# Define patterns for CUTLASS dispatch
cutlass_patterns = [
    FusionPattern("cutlass.matmul", is_op("relax.matmul")(wildcard(), wildcard())),
    FusionPattern("cutlass.conv2d", is_op("relax.nn.conv2d")(wildcard(), wildcard())),
]

# Fuse and annotate for CUTLASS
mod = FuseOpsByPattern(
    patterns=cutlass_patterns,
    annotate_codegen=True
)(mod)

# Later, BackendDispatch will route annotated functions to CUTLASS
mod = relax.transform.BackendDispatch()(mod)
```

**Pattern check function:**

The `check` parameter allows additional validation of matched patterns:

```python
def check_conv2d_relu(matched):
    """Only fuse conv2d + relu when conv2d has specific properties."""
    conv2d_call = matched["root"]
    return (conv2d_call.attrs.data_layout == "NCHW" and
            conv2d_call.attrs.out_dtype == "float32")

pattern = FusionPattern(
    name="fused_conv2d_relu",
    pattern=is_op("relax.nn.relu")(is_op("relax.nn.conv2d")(wildcard(), wildcard())),
    check=check_conv2d_relu
)
```

**Extracting matched subgraphs:**

```python
from tvm.relax.dpl import PatternContext, is_op, wildcard

# Create a named pattern with annotations
pat_conv = is_op("relax.nn.conv2d")(wildcard().named("input"),
                                     wildcard().named("weight"))
pat_relu = is_op("relax.nn.relu")(pat_conv.named("conv_output"))

# After matching, access named nodes in the check function
def check_fn(matched):
    conv_output_shape = matched["conv_output"].struct_info.shape
    # Only fuse if the conv output is small enough
    return conv_output_shape[-1] <= 1024

pattern = FusionPattern("conv_relu", pat_relu, check=check_fn)
```

---

## 5.4 Decomposition

### 5.4.1 DecomposeOpsForInference

`relax.transform.DecomposeOpsForInference` decomposes complex high-level operators into sequences of simpler operators that are more amenable to optimization during inference. This pass is applied before legalization and fusion.

```python
mod = relax.transform.DecomposeOpsForInference()(mod)
```

**Decomposition examples:**

| Original Operator | Decomposed Into |
|------------------|-----------------|
| `batch_norm` | `reshape`, `multiply`, `add` |
| `layer_norm` | `mean`, `variance`, `subtract`, `multiply`, `add` |
| `softmax` | `exp`, `subtract` (max), `sum`, `divide` |
| `gelu` | `multiply`, `add`, `tanh`, `erf` |
| `group_norm` | `reshape`, `mean`, `variance`, `multiply`, `add`, `reshape` |
| `adaptive_avg_pool2d` | `avg_pool2d` with computed kernel size |

```python
# Example: batch_norm decomposition for inference
@tvm.script.ir_module
class BeforeDecompose:
    @R.function
    def main(x: R.Tensor((1, 64, 224, 224), "float32"),
             gamma: R.Tensor((64,), "float32"),
             beta: R.Tensor((64,), "float32"),
             mean: R.Tensor((64,), "float32"),
             var: R.Tensor((64,), "float32")):
        with R.dataflow():
            # Batch norm in inference mode
            lv = R.nn.batch_norm(x, gamma, beta, mean, var,
                                 axis=1, epsilon=1e-5)
            R.output(lv)
        return lv

# After decomposition, batch_norm becomes element-wise operations
mod = relax.transform.DecomposeOpsForInference()(BeforeDecompose)
# Result: gamma_reshaped * (x - mean_reshaped) / sqrt(var_reshaped + eps) + beta_reshaped
```

**Custom decomposition:**

You can register custom decomposition functions for specific operators:

```python
@relax.transform.register_decompose_op("custom.my_norm")
def decompose_my_norm(attrs, args):
    """Decompose custom normalization into simpler ops."""
    x, gamma, beta = args
    mean = R.mean(x, axis=attrs.axis, keepdims=True)
    var = R.variance(x, axis=attrs.axis, keepdims=True)
    x_norm = (x - mean) / R.sqrt(var + R.const(attrs.epsilon))
    return x_norm * R.reshape(gamma, mean.shape) + R.reshape(beta, mean.shape)

# Apply with custom decompositions included
mod = relax.transform.DecomposeOpsForInference()(mod)
```

### 5.4.2 DecomposeOpsForTraining

`relax.transform.DecomposeOpsForTraining` performs decomposition tailored for training mode. Unlike the inference variant, training decomposition must preserve the operations needed for backward gradient computation.

```python
mod = relax.transform.DecomposeOpsForTraining()(mod)
```

**Key differences from inference decomposition:**

| Aspect | Inference | Training |
|--------|-----------|----------|
| Batch norm | Folds running stats into weights | Preserves running stat updates |
| Dropout | Removed entirely | Preserved as identity with mask |
| Softmax | Can be approximated | Exact computation preserved |
| Loss functions | Not needed | Fully decomposed for gradient flow |

```python
# Training decomposition example
@tvm.script.ir_module
class TrainingModel:
    @R.function
    def main(x: R.Tensor((32, 784), "float32"),
             y: R.Tensor((32, 10), "float32"),
             w: R.Tensor((784, 10), "float32")):
        with R.dataflow():
            logits = R.matmul(x, w)
            loss = R.nn.cross_entropy_with_logits(logits, y)
            R.output(loss)
        return loss

# Decompose for training
mod = relax.transform.DecomposeOpsForTraining()(TrainingModel)
```

---

## 5.5 Canonicalization

### 5.5.1 CanonicalizeBindings

`relax.transform.CanonicalizeBindings` normalizes variable bindings in Relax functions. It ensures that each variable is bound exactly once and that bindings follow a consistent order. This pass is essential for making subsequent analyses and transformations reliable.

```python
mod = relax.transform.CanonicalizeBindings()(mod)
```

**What CanonicalizeBindings does:**

1. **Inline trivial bindings**: Replaces single-use variable bindings with their direct values.
2. **Eliminate redundant bindings**: Removes bindings that duplicate existing variables.
3. **Normalize tuple bindings**: Restructures tuple extraction patterns.
4. **Simplify nested bindings**: Flattens chains of variable assignments.

```python
# Before canonicalization
@tvm.script.ir_module
class BeforeCanonical:
    @R.function
    def main(x: R.Tensor((1, 10), "float32")):
        with R.dataflow():
            a = R.add(x, R.const(1.0))
            b = a              # Redundant binding
            c = R.nn.relu(b)   # Can be simplified to R.nn.relu(a)
            d = c              # Another redundant binding
            R.output(d)
        return d

# After canonicalization
@tvm.script.ir_module
class AfterCanonical:
    @R.function
    def main(x: R.Tensor((1, 10), "float32")):
        with R.dataflow():
            lv = R.add(x, R.const(1.0))
            gv = R.nn.relu(lv)
            R.output(gv)
        return gv

mod = relax.transform.CanonicalizeBindings()(BeforeCanonical)
```

### 5.5.2 CanonicalizeBindingsForBlockBuilder

`relax.transform.CanonicalizeBindingsForBlockBuilder` is a specialized variant of `CanonicalizeBindings` designed to work with the Relax BlockBuilder. It canonicalizes bindings that were generated during IR construction, ensuring consistent naming and structure.

```python
mod = relax.transform.CanonicalizeBindingsForBlockBuilder()(mod)
```

This pass is typically used internally by other passes that construct IR via the BlockBuilder. It handles:

- Normalization of variable names generated by the BlockBuilder.
- Elimination of intermediate bindings created during incremental IR construction.
- Consistent handling of DataflowVar vs. Var annotations.

---

## 5.6 Simplification and Optimization

### 5.6.1 FoldConstant

`relax.transform.FoldConstant` performs constant folding on Relax expressions. It evaluates operations on constant tensors at compile time and replaces them with pre-computed results. This eliminates unnecessary computation at runtime.

```python
mod = relax.transform.FoldConstant()(mod)
```

**Constant folding examples:**

```python
# Before constant folding
@tvm.script.ir_module
class BeforeFold:
    @R.function
    def main(x: R.Tensor((1, 10), "float32")):
        with R.dataflow():
            # These are all constants that can be folded
            a = R.const(2.0)
            b = R.const(3.0)
            c = R.multiply(a, b)  # Can be folded to R.const(6.0)
            d = R.multiply(x, c)  # Becomes R.multiply(x, R.const(6.0))
            R.output(d)
        return d

mod = relax.transform.FoldConstant()(BeforeFold)
# c is replaced with R.const(6.0), eliminating the multiply at runtime
```

**What gets folded:**

| Expression | Result |
|------------|--------|
| `R.add(R.const(a), R.const(b))` | `R.const(a + b)` |
| `R.multiply(R.const(a), R.const(b))` | `R.const(a * b)` |
| `R.reshape(R.const(tensor), new_shape)` | Pre-computed reshaped constant |
| `R.nn.softmax(R.const(tensor))` | Pre-computed softmax values |
| `R.strided_slice(R.const(tensor), ...)` | Pre-computed slice |

**FoldConstant with shape expressions:**

```python
# Constant folding also works with shape computations
@tvm.script.ir_module
class ShapeFold:
    @R.function
    def main(x: R.Tensor(("batch", 128), "float32")):
        batch = T.int64()
        with R.dataflow():
            # Shape operations on constants are folded
            s = R.shape_of(x)
            dim0 = R.take(s, R.const(0, "int64"))  # Evaluates to batch
            new_shape = R.concat([R.const([1, 128], "int64")], axis=0)
            result = R.reshape(x, new_shape)
            R.output(result)
        return result

mod = relax.transform.FoldConstant()(ShapeFold)
```

### 5.6.2 SimplifyExpr

`relax.transform.SimplifyExpr` performs algebraic simplification on Relax expressions. Unlike `FoldConstant`, which evaluates constant expressions, `SimplifyExpr` rewrites expressions using algebraic identities.

```python
mod = relax.transform.SimplifyExpr()(mod)
```

**Simplification rules:**

| Expression | Simplified To |
|------------|---------------|
| `x + 0` | `x` |
| `x * 1` | `x` |
| `x * 0` | `zeros_like(x)` |
| `reshape(reshape(x, s1), s2)` where original shape is preserved | `reshape(x, s2)` |
| `expand_dims(squeeze(x, axis), axis)` | `x` |
| `x + x` | `x * 2` |
| `concat(split(x, ...), ...)` | `x` |
| `broadcast_to(x, x.shape)` | `x` |

```python
# Simplification example
@tvm.script.ir_module
class BeforeSimplify:
    @R.function
    def main(x: R.Tensor((1, 128), "float32")):
        with R.dataflow():
            # Identity operations
            a = R.add(x, R.const(0.0))     # x + 0 -> x
            b = R.multiply(a, R.const(1.0)) # x * 1 -> x
            # Redundant reshapes
            c = R.reshape(b, (1, 128))
            d = R.reshape(c, (1, 128))
            R.output(d)
        return d

mod = relax.transform.SimplifyExpr()(BeforeSimplify)
# All redundant operations eliminated, result is just x
```

### 5.6.3 DeadCodeElimination

`relax.transform.DeadCodeElimination` (DCE) removes code that does not contribute to the function's output. This includes unused variable bindings, unused functions in the IRModule, and dead parameters.

```python
mod = relax.transform.DeadCodeElimination()(mod)
```

**DCE behavior:**

1. **Local DCE**: Within a function, removes bindings whose results are never used.
2. **Global DCE**: Removes entire functions from the IRModule that are never called.
3. **Parameter elimination**: May remove unused function parameters in some cases.

```python
# Before DCE
@tvm.script.ir_module
class BeforeDCE:
    @R.function
    def main(x: R.Tensor((1, 10), "float32")):
        with R.dataflow():
            a = R.nn.relu(x)      # Used
            b = R.multiply(x, x)  # Not used (dead)
            c = R.add(a, R.const(1.0))  # Used
            R.output(c)
        return c

    @R.function
    def unused_func(x: R.Tensor((1, 10), "float32")):
        return x  # Never called from main

# After DCE
mod = relax.transform.DeadCodeElimination()(BeforeDCE)
# b and unused_func are removed
```

### 5.6.4 SimplifyReshape

`relax.transform.SimplifyReshape` is a specialized pass that simplifies reshape operations. It eliminates redundant reshapes and converts reshape chains into single reshape operations.

```python
mod = relax.transform.SimplifyReshape()(mod)
```

**Reshape simplification rules:**

| Pattern | Result |
|---------|--------|
| `reshape(x, x.shape)` | `x` (identity reshape) |
| `reshape(reshape(x, s1), s2)` | `reshape(x, s2)` |
| `reshape(transpose(x, ...), ...)` | May rewrite to single operation |
| `reshape(expand_dims(x, ...), ...)` | May eliminate expand_dims |

```python
# Reshape chain simplification
@tvm.script.ir_module
class ReshapeChain:
    @R.function
    def main(x: R.Tensor((1, 3, 32, 32), "float32")):
        with R.dataflow():
            a = R.reshape(x, (1, 3, 1024))
            b = R.reshape(a, (1, 3072))
            c = R.reshape(b, (3, 1024))
            R.output(c)
        return c

mod = relax.transform.SimplifyReshape()(ReshapeChain)
# Chain of 3 reshapes collapsed to single: reshape(x, (3, 1024))
```

### 5.6.5 KnowledgeBasedSimplify

`relax.transform.KnowledgeBasedSimplify` performs simplification using knowledge about tensor shapes, dtypes, and value ranges. It uses structural information extracted from the IR to make simplification decisions.

```python
mod = relax.transform.KnowledgeBasedSimplify()(mod)
```

This pass uses the `StructInfo` attached to each expression to derive knowledge:

- **Shape knowledge**: If `x` has shape `(1, n)` and we compute `expand_dims(x, 0)`, we know the result has shape `(1, 1, n)`.
- **Value range knowledge**: If `x` is the result of `relu`, we know all values are non-negative.
- **Dtype knowledge**: Integer operations can be simplified based on bit width.

```python
# Knowledge-based simplification
@tvm.script.ir_module
class KnowledgeSimplify:
    @R.function
    def main(x: R.Tensor((1, 128), "float32")):
        with R.dataflow():
            a = R.nn.relu(x)       # All values >= 0
            b = R.maximum(a, R.const(0.0))  # max(relu(x), 0) = relu(x)
            R.output(b)
        return b

mod = relax.transform.KnowledgeBasedSimplify()(KnowledgeSimplify)
# b = maximum(relu(x), 0) is simplified to b = relu(x) since relu output is always >= 0
```

---

## 5.7 Layout and Type Transforms

### 5.7.1 AlterOpImpl

`relax.transform.AlterOpImpl` replaces operator implementations across the IRModule. This is useful for swapping between different implementations of the same operation (e.g., replacing a reference implementation with an optimized one).

```python
mod = relax.transform.AlterOpImpl(
    origin_op="relax.matmul",
    target_op="relax.matmul_nt",
    attrs_map={"transpose_a": False, "transpose_b": True}
)(mod)
```

**Use cases:**

1. **Replace generic ops with specialized variants**: e.g., `matmul` -> `matmul_nt` when one input is transposed.
2. **Swap implementation backends**: e.g., replace a TIR implementation with a CUTLASS call.
3. **Insert instrumentation**: Replace ops with instrumented variants for profiling.

```python
# Example: Replace all conv2d with depthwise_conv2d when groups == channels
from tvm.relax.transform import AlterOpImpl

def should_replace_conv2d(call_node):
    """Check if this conv2d should be replaced with depthwise variant."""
    return (hasattr(call_node.attrs, 'groups') and
            call_node.attrs.groups == call_node.attrs.channels)

# Custom replacement function
mod = relax.transform.AlterOpImpl(
    origin_op="relax.nn.conv2d",
    target_op="relax.nn.depthwise_conv2d",
    check=should_replace_conv2d
)(mod)
```

### 5.7.2 ConvertLayout

`relax.transform.ConvertLayout` converts data layout (e.g., NCHW to NHWC) throughout the model. It rewrites operators and inserts layout transformation operations as needed.

```python
mod = relax.transform.ConvertLayout(
    desired_layouts={"relax.nn.conv2d": ["NHWC", "OHWI"]}
)(mod)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `desired_layouts` | `Dict[str, List[str]]` | Map from operator name to desired data and kernel layouts |

```python
# Full layout conversion example
mod = relax.transform.ConvertLayout(
    desired_layouts={
        "relax.nn.conv2d": ["NHWC", "OHWI"],
        "relax.nn.max_pool2d": ["NHWC"],
        "relax.nn.avg_pool2d": ["NHWC"],
        "relax.nn.batch_norm": ["NHWC"],
    }
)(mod)
```

**Layout conversion process:**

1. For each operator listed in `desired_layouts`, find all calls to that operator.
2. Insert `layout_transform` operations before inputs to convert them to the desired layout.
3. Modify the operator call to use the new layout attributes.
4. Insert `layout_transform` operations after outputs to convert back to the original layout (if needed by downstream consumers).
5. Eliminate redundant back-to-back layout transforms.

```python
# Before layout conversion (NCHW)
@tvm.script.ir_module
class NCHWModel:
    @R.function
    def main(x: R.Tensor((1, 3, 224, 224), "float32"),
             w: R.Tensor((32, 3, 3, 3), "float32")):
        with R.dataflow():
            conv = R.nn.conv2d(x, w, padding=(1, 1),
                               data_layout="NCHW", kernel_layout="OIHW")
            bn = R.nn.batch_norm(conv)
            pool = R.nn.max_pool2d(bn, pool_size=(2, 2),
                                   layout="NCHW")
            R.output(pool)
        return pool

# After layout conversion (NHWC)
mod = relax.transform.ConvertLayout(
    desired_layouts={"relax.nn.conv2d": ["NHWC", "OHWI"]}
)(NCHWModel)
# x is transformed to NHWC before conv2d
# conv2d operates on NHWC data and OHWI kernel
# Output is in NHWC format
```

### 5.7.3 NarrowDataType

`relax.transform.NarrowDataType` narrows the data types used in the model. This is useful for mixed-precision inference where certain operations can use lower precision (e.g., float16 instead of float32).

```python
mod = relax.transform.NarrowDataType(
    target_dtype="float16"
)(mod)
```

**Behavior:**

1. Converts intermediate tensor dtypes to the target dtype.
2. Inserts cast operations where necessary to maintain correctness.
3. Preserves the original dtype for operations that require higher precision (e.g., reduction accumulators).

```python
# Narrow to float16 for GPU inference
mod = relax.transform.NarrowDataType("float16")(mod)
```

---

## 5.8 Lowering Passes

### 5.8.1 ToNonDataflow

`relax.transform.ToNonDataflow` converts dataflow blocks (`R.dataflow()`) into standard (non-dataflow) binding blocks. This is a necessary step before further lowering because dataflow blocks have restrictions on what operations can appear within them.

```python
mod = relax.transform.ToNonDataflow()(mod)
```

**What this pass does:**

1. Removes `R.dataflow()` block wrappers.
2. Converts `DataflowVar` to regular `Var`.
3. Ensures all bindings are in standard binding blocks.

```python
# Before ToNonDataflow
@tvm.script.ir_module
class WithDataflow:
    @R.function
    def main(x: R.Tensor((1, 10), "float32")):
        with R.dataflow():
            lv1 = R.nn.relu(x)
            lv2 = R.add(lv1, R.const(1.0))
            R.output(lv2)
        return lv2

# After ToNonDataflow
mod = relax.transform.ToNonDataflow()(WithDataflow)
# DataflowVar becomes Var, dataflow block becomes binding block
```

### 5.8.2 RemovePurityChecking

`relax.transform.RemovePurityChecking` removes purity annotations and checks from the IR. Relax functions can be annotated as pure or impure, and the type system enforces certain constraints based on purity. This pass removes those annotations.

```python
mod = relax.transform.RemovePurityChecking()(mod)
```

This is typically applied late in the compilation pipeline when purity checking is no longer needed:

```python
# Late-stage pipeline
pipeline = relax.transform.Sequential([
    relax.transform.ToNonDataflow(),
    relax.transform.RemovePurityChecking(),
    relax.transform.VMBuiltinLower(),
])
```

### 5.8.3 VMBuiltinLower

`relax.transform.VMBuiltinLower` lowers Relax built-in operations to VM (Virtual Machine) executable instructions. This pass converts high-level Relax constructs into forms that the TVM Relax VM can execute.

```python
mod = relax.transform.VMBuiltinLower()(mod)
```

**What gets lowered:**

| High-level Construct | VM Instruction |
|---------------------|----------------|
| `R.call_tir` | VM call_tir instruction |
| `R.call_dps_packed` | VM packed call instruction |
| `R.shape_of` | VM shape_of instruction |
| `R.alloc_tensor` | VM alloc_tensor instruction |
| `R.builtin.alloc_tensor` | VM allocation instruction |
| Tuple construction | VM tuple construction |
| Tuple indexing | VM tuple get element |

```python
# VM lowering is typically one of the final passes
pipeline = relax.transform.Sequential([
    relax.transform.ToNonDataflow(),
    relax.transform.RemovePurityChecking(),
    relax.transform.VMBuiltinLower(),
    relax.transform.ComputePrimValue(),
])
```

### 5.8.4 ComputePrimValue

`relax.transform.ComputePrimValue` evaluates primitive value expressions (scalars, shapes, etc.) at compile time. It replaces expressions that compute primitive values with their computed results.

```python
mod = relax.transform.ComputePrimValue()(mod)
```

**Examples of computed values:**

- Shape arithmetic: `shape[0] * shape[1]` is replaced with the computed integer.
- Scalar arithmetic: `R.const(2) + R.const(3)` is replaced with `R.const(5)`.
- Type-cast of constants: `R.cast(R.const(1.0), "int32")` is replaced with `R.const(1, "int32")`.

```python
# Before ComputePrimValue
@tvm.script.ir_module
class BeforeCompute:
    @R.function
    def main(x: R.Tensor(("n", 128), "float32")):
        n = T.int64()
        with R.dataflow():
            s = R.shape_of(x)
            dim = R.take(s, R.const(0, "int64"))
            new_dim = R.add(dim, R.const(128, "int64"))
            result = R.reshape(x, (new_dim,))
            R.output(result)
        return result

mod = relax.transform.ComputePrimValue()(BeforeCompute)
# Shape arithmetic is evaluated where possible
```

### 5.8.5 StaticPlanBlockMemory

`relax.transform.StaticPlanBlockMemory` performs static memory planning for the Relax VM. It analyzes tensor lifetimes and allocates memory with reuse, minimizing peak memory usage.

```python
mod = relax.transform.StaticPlanBlockMemory()(mod)
```

**Memory planning strategy:**

1. **Liveness analysis**: Determines when each tensor is first used and last used.
2. **Memory reuse**: Identifies tensors with non-overlapping lifetimes that can share the same memory.
3. **In-place operations**: Identifies operations that can work in-place without allocating new memory.
4. **Storage sharing**: Groups tensors of the same size into shared storage pools.

```python
# Memory planning is applied late in the pipeline
pipeline = relax.transform.Sequential([
    relax.transform.StaticPlanBlockMemory(),
    relax.transform.VMBuiltinLower(),
])
```

---

## 5.9 Backend Integration

### 5.9.1 AttachGlobalSymbol

`relax.transform.AttachGlobalSymbol` attaches global symbol names to functions in the IRModule. This is necessary for generating callable entry points in the compiled executable.

```python
mod = relax.transform.AttachGlobalSymbol()(mod)
```

**Behavior:**

- Assigns `global_symbol` attributes to all global functions.
- Ensures that each function has a unique, valid C-compatible name.
- Required before `RunCodegen` and before serialization.

```python
# Attach symbols before codegen
mod = relax.transform.AttachGlobalSymbol()(mod)
# Now each function has a "global_symbol" attribute matching its name
```

### 5.9.2 RunCodegen

`relax.transform.RunCodegen` invokes external code generators (such as CUTLASS, TensorRT, or custom backends) to compile annotated sub-graphs. Functions that have been marked for external codegen (via `annotate_codegen=True` in `FuseOpsByPattern`) are dispatched to the appropriate code generator.

```python
mod = relax.transform.RunCodegen(
    target="cuda",
    codegen_options={
        "cutlass": {"sm_version": 80}
    }
)(mod)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `str` or `Target` | None | Compilation target |
| `codegen_options` | `Dict` | `{}` | Options passed to code generators |

**Supported external codegen backends:**

| Backend | Operations | Notes |
|---------|-----------|-------|
| CUTLASS | matmul, conv2d, attention | NVIDIA GPUs with compute capability 7.0+ |
| TensorRT | Various | NVIDIA GPUs, requires TensorRT installation |
| BYOC | Custom | Bring Your Own Codegen framework |

```python
# Full backend integration pipeline
mod = relax.transform.FuseOpsByPattern(
    patterns=cutlass_patterns,
    annotate_codegen=True
)(mod)
mod = relax.transform.AttachGlobalSymbol()(mod)
mod = relax.transform.RunCodegen(target="cuda")(mod)
```

### 5.9.3 BackendDispatch

`relax.transform.BackendDispatch` dispatches operators to specific backends based on annotations and target capabilities. It resolves which backend should handle each operator and inserts the appropriate dispatch calls.

```python
mod = relax.transform.BackendDispatch()(mod)
```

**Dispatch strategy:**

1. Check operator annotations for backend preferences.
2. Match operators against registered backend patterns.
3. Dispatch to the best available backend for each operator.
4. Fall back to the default TIR backend if no specialized backend is available.

```python
# Backend dispatch with custom options
mod = relax.transform.BackendDispatch(
    fallback_backend="cutlass"
)(mod)
```

### 5.9.4 MergeCompositeFunctions

`relax.transform.MergeCompositeFunctions` merges multiple composite functions that were created by `FuseOpsByPattern` into single functions. This is useful when multiple small composite functions should be handled as a unit by the backend.

```python
mod = relax.transform.MergeCompositeFunctions()(mod)
```

**When to use:**

- After `FuseOpsByPattern` has created many small composite functions.
- When the backend can handle larger fused operations more efficiently.
- To reduce the overhead of managing many small external functions.

```python
# Merge small composite functions before codegen
mod = relax.transform.MergeCompositeFunctions()(mod)
mod = relax.transform.RunCodegen(target="cuda")(mod)
```

---

## 5.10 Parameter Management

### 5.10.1 LiftTransformParams

`relax.transform.LiftTransformParams` lifts parameter transformation operations out of the main computation function. When model parameters require preprocessing (e.g., reshaping, transposing, normalization), this pass moves those operations into a separate function that can be executed once at model loading time.

```python
mod = relax.transform.LiftTransformParams()(mod)
```

**What gets lifted:**

- Reshape/transpose operations on constant parameters.
- Batch normalization folding (merging running statistics into weights).
- Quantization/dequantization of parameters.
- Any pure computation that depends only on model parameters.

```python
# Before LiftTransformParams
@tvm.script.ir_module
class BeforeLift:
    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w: R.Tensor((10, 784), "float32")):
        with R.dataflow():
            # This transpose is always the same for given weights
            w_t = R.permute_dims(w, axes=(1, 0))  # (784, 10)
            result = R.matmul(x, w_t)
            R.output(result)
        return result

# After LiftTransformParams
mod = relax.transform.LiftTransformParams()(BeforeLift)
# Creates a new function: transform_params that does the transpose
# main now receives the pre-transformed parameters
```

**Use in quantization pipelines:**

```python
# Lift parameter transformations for quantized models
pipeline = relax.transform.Sequential([
    relax.transform.DecomposeOpsForInference(),
    relax.transform.FoldConstant(),
    relax.transform.LiftTransformParams(),  # Lift quant/dequant of weights
    relax.transform.LegalizeOps(),
])
```

### 5.10.2 BundleModelParams

`relax.transform.BundleModelParams` bundles all model parameters into a single tuple argument. This simplifies parameter management by collecting all individual parameter tensors into one container.

```python
mod = relax.transform.BundleModelParams()(mod)
```

**Before bundling:**

```python
@R.function
def main(x: R.Tensor((1, 784), "float32"),
         w1: R.Tensor((784, 256), "float32"),
         b1: R.Tensor((256,), "float32"),
         w2: R.Tensor((256, 10), "float32"),
         b2: R.Tensor((10,), "float32")):
    ...
```

**After bundling:**

```python
@R.function
def main(x: R.Tensor((1, 784), "float32"),
         params: R.Tuple(R.Tensor((784, 256), "float32"),
                         R.Tensor((256,), "float32"),
                         R.Tensor((256, 10), "float32"),
                         R.Tensor((10,), "float32"))):
    w1 = params[0]
    b1 = params[1]
    w2 = params[2]
    b2 = params[3]
    ...
```

### 5.10.3 ExpandTupleArguments

`relax.transform.ExpandTupleArguments` expands tuple arguments into individual function parameters. This is the inverse of `BundleModelParams` and is useful when interfacing with backends that do not support tuple arguments.

```python
mod = relax.transform.ExpandTupleArguments()(mod)
```

```python
# Before: function takes a tuple
@R.function
def main(params: R.Tuple(R.Tensor((128, 128), "float32"),
                          R.Tensor((128,), "float32"))):
    ...

# After: function takes individual tensors
@R.function
def main(p0: R.Tensor((128, 128), "float32"),
         p1: R.Tensor((128,), "float32")):
    ...
```

---

## 5.11 Other Passes

### 5.11.1 Normalize

`relax.transform.Normalize` normalizes the Relax IR into its canonical form. This includes normalizing struct info, ensuring all expressions have proper type annotations, and standardizing the IR representation.

```python
mod = relax.transform.Normalize()(mod)
```

**Normalization includes:**

1. **Struct info normalization**: Ensures all expressions have up-to-date struct info.
2. **Binding normalization**: Standardizes variable binding forms.
3. **Type normalization**: Normalizes type representations.

```python
# Normalize is often applied after transformations that may leave the IR
# in a non-canonical state
mod = some_custom_pass(mod)
mod = relax.transform.Normalize()(mod)  # Ensure IR is well-formed
```

### 5.11.2 Gradient

`relax.transform.Gradient` computes the gradient (reverse-mode automatic differentiation) of a Relax function. This is the foundation for training support in TVM.

```python
mod = relax.transform.Gradient(
    func_name="main",
    require_grads=["x", "w"]
)(mod)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func_name` | `str` | `"main"` | Name of the function to differentiate |
| `require_grads` | `List[str]` | All params | Parameters to compute gradients for |

```python
# Training pipeline with gradient computation
@tvm.script.ir_module
class SimpleModel:
    @R.function
    def main(x: R.Tensor((32, 784), "float32"),
             w: R.Tensor((784, 10), "float32"),
             y: R.Tensor((32, 10), "float32")) -> R.Tensor((), "float32"):
        with R.dataflow():
            logits = R.matmul(x, w)
            loss = R.nn.cross_entropy_with_logits(logits, y)
            R.output(loss)
        return loss

# Compute gradients
mod = relax.transform.Gradient(
    func_name="main",
    require_grads=["w"]
)(SimpleModel)
# Result: a new function that returns (loss, grad_w)
```

**Gradient computation process:**

1. Perform forward pass, recording all operations.
2. Construct the adjoint (reverse) computation graph.
3. Chain adjoint rules for each operation.
4. Generate a new function that returns both the original result and the gradients.

```python
# Full training pipeline
pipeline = relax.transform.Sequential([
    relax.transform.DecomposeOpsForTraining(),
    relax.transform.LegalizeOps(),
    relax.transform.Gradient(func_name="main"),
    relax.transform.DeadCodeElimination(),
])
```

---

## 5.12 Pipeline Composition

### 5.12.1 Built-in Pipelines

TVM provides several built-in compilation pipelines that compose the individual passes in the optimal order for common use cases.

#### Zero Pipeline

The `"zero"` pipeline provides a basic compilation path with minimal optimizations:

```python
import tvm
from tvm import relax

# Load a model
mod, params = relax.frontend.torch.from_exported_program(exported_program)

# Apply zero pipeline
mod = relax.get_pipeline("zero")(mod)
```

The zero pipeline applies these passes in order:

1. `LegalizeOps` — lower all operators to TIR
2. `CanonicalizeBindings` — normalize bindings
3. `FoldConstant` — fold constants
4. `FuseTIR` — fuse TIR functions
5. `DeadCodeElimination` — remove dead code

#### Static Shape Tuning Pipeline

The `"static_shape_tuning"` pipeline is designed for models with static shapes and uses MetaSchedule for auto-tuning:

```python
mod = relax.get_pipeline("static_shape_tuning")(mod)
```

This pipeline includes:

1. `DecomposeOpsForInference` — decompose for inference
2. `FuseOpsByPattern` — pattern-based fusion
3. `LegalizeOps` — legalize remaining ops
4. `FoldConstant` — fold constants
5. `FuseTIR` — fuse TIR PrimFuncs
6. MetaSchedule tuning — auto-tune fused PrimFuncs
7. `DeadCodeElimination` — clean up

#### Default Pipeline

The default pipeline (obtained via `relax.get_pipeline()` without arguments) applies a comprehensive set of optimizations:

```python
mod = relax.get_pipeline()(mod)
```

### 5.12.2 Custom Pipelines with Sequential

You can create custom pipelines using `relax.transform.Sequential`:

```python
from tvm import relax

# Define a custom pipeline
custom_pipeline = relax.transform.Sequential([
    # Phase 1: Decomposition and canonicalization
    relax.transform.DecomposeOpsForInference(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),

    # Phase 2: Layout optimization
    relax.transform.ConvertLayout(
        desired_layouts={"relax.nn.conv2d": ["NHWC", "OHWI"]}
    ),
    relax.transform.CanonicalizeBindings(),

    # Phase 3: Fusion
    relax.transform.FuseOpsByPattern(
        patterns=my_patterns,
        annotate_codegen=True
    ),
    relax.transform.LegalizeOps(),
    relax.transform.FuseTIR(),

    # Phase 4: Optimization
    relax.transform.SimplifyExpr(),
    relax.transform.FoldConstant(),
    relax.transform.DeadCodeElimination(),

    # Phase 5: Lowering
    relax.transform.ToNonDataflow(),
    relax.transform.RemovePurityChecking(),
    relax.transform.StaticPlanBlockMemory(),
    relax.transform.VMBuiltinLower(),
    relax.transform.ComputePrimValue(),
])

mod = custom_pipeline(mod)
```

### 5.12.3 Pipeline with MetaSchedule Integration

For auto-tuning with MetaSchedule, the pipeline must be split to allow tuning between TIR generation and scheduling:

```python
import tvm
from tvm import relax, meta_schedule

# Phase 1: Frontend to TIR
pipeline_phase1 = relax.transform.Sequential([
    relax.transform.DecomposeOpsForInference(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),
    relax.transform.LegalizeOps(),
    relax.transform.FuseTIR(),
])
mod = pipeline_phase1(mod)

# Phase 2: MetaSchedule tuning
with meta_schedule.database.Database("json", "tuning_records.json") as db:
    with tvm.target.Target("cuda"):
        mod = meta_schedule.tune_tir(
            mod=mod,
            target=tvm.target.Target("cuda"),
            max_trials_global=1000,
            database=db,
        )

# Phase 3: Lowering and codegen
pipeline_phase3 = relax.transform.Sequential([
    relax.transform.AttachGlobalSymbol(),
    relax.transform.ToNonDataflow(),
    relax.transform.RemovePurityChecking(),
    relax.transform.VMBuiltinLower(),
    relax.transform.ComputePrimValue(),
])
mod = pipeline_phase3(mod)
```

### 5.12.4 Pipeline with External Backend

For hybrid compilation using external backends like CUTLASS:

```python
from tvm.relax.transform import FusionPattern

# Define CUTLASS fusion patterns
cutlass_matmul = FusionPattern(
    "cutlass.matmul",
    is_op("relax.matmul")(wildcard(), wildcard())
)
cutlass_conv2d = FusionPattern(
    "cutlass.conv2d",
    is_op("relax.nn.conv2d")(wildcard(), wildcard())
)
cutlass_attention = FusionPattern(
    "cutlass.attention",
    is_op("relax.nn.attention")(wildcard(), wildcard(), wildcard())
)

cutlass_patterns = [cutlass_matmul, cutlass_conv2d, cutlass_attention]

# Pipeline with CUTLASS integration
cutlass_pipeline = relax.transform.Sequential([
    # Decompose and canonicalize
    relax.transform.DecomposeOpsForInference(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),

    # Pattern-based fusion with CUTLASS annotation
    relax.transform.FuseOpsByPattern(
        patterns=cutlass_patterns,
        annotate_codegen=True
    ),

    # Legalize remaining ops
    relax.transform.LegalizeOps(),
    relax.transform.FoldConstant(),
    relax.transform.DeadCodeElimination(),

    # Backend dispatch and codegen
    relax.transform.AttachGlobalSymbol(),
    relax.transform.RunCodegen(target="cuda"),

    # Lowering for VM
    relax.transform.ToNonDataflow(),
    relax.transform.RemovePurityChecking(),
    relax.transform.StaticPlanBlockMemory(),
    relax.transform.VMBuiltinLower(),
    relax.transform.ComputePrimValue(),
])

mod = cutlass_pipeline(mod)
```

---

## 5.13 Pass Ordering Guidelines

The order in which transformation passes are applied significantly impacts the quality of the generated code. Here are recommended orderings for common scenarios.

### Inference Pipeline (Recommended Order)

```
DecomposeOpsForInference
    -> CanonicalizeBindings
    -> FoldConstant
    -> SimplifyExpr
    -> ConvertLayout (optional)
    -> FuseOpsByPattern (or FuseOps)
    -> LegalizeOps
    -> FuseTIR
    -> FoldConstant
    -> DeadCodeElimination
    -> LiftTransformParams (optional)
    -> StaticPlanBlockMemory
    -> ToNonDataflow
    -> RemovePurityChecking
    -> VMBuiltinLower
    -> ComputePrimValue
```

### Training Pipeline (Recommended Order)

```
DecomposeOpsForTraining
    -> CanonicalizeBindings
    -> FoldConstant
    -> LegalizeOps
    -> Gradient
    -> FoldConstant
    -> DeadCodeElimination
    -> ToNonDataflow
    -> RemovePurityChecking
    -> VMBuiltinLower
    -> ComputePrimValue
```

### Debugging Tips

If a pass produces unexpected results, try inserting `CanonicalizeBindings` and `Normalize` before it:

```python
# Debug pipeline with normalization between passes
debug_pipeline = relax.transform.Sequential([
    relax.transform.DecomposeOpsForInference(),
    relax.transform.Normalize(),           # Ensure clean IR
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),
    relax.transform.Normalize(),           # Ensure clean IR
    relax.transform.LegalizeOps(),
])
```

---

## 5.14 Advanced Topics

### 5.14.1 Writing Custom Transformation Passes

You can write custom transformation passes using the Relax BlockBuilder API:

```python
import tvm
from tvm import relax
from tvm.relax import BlockBuilder

@relax.transform
class MyCustomPass:
    """Custom transformation pass example."""

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """Transform the IRModule."""
        builder = BlockBuilder(mod)

        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_func = self._transform_function(builder, func)
                builder.update_func(gv, new_func)

        return builder.get()

    def _transform_function(self, builder: BlockBuilder, func: relax.Function) -> relax.Function:
        """Transform a single Relax function."""
        # Use the BlockBuilder to construct the transformed function
        with builder.function(func.name):
            builder.name(func.params)
            with builder.dataflow():
                # Transform the function body here
                pass
            builder.emit_output(...)
        return builder.get()[func.name]

# Use the custom pass
mod = MyCustomPass()(mod)
```

### 5.14.2 Pass Context Configuration

Some passes accept configuration through the `PassContext`:

```python
with tvm.transform.PassContext(
    config={
        "relax.transform.LegalizeOps": {
            "legalize_map": custom_legalize_map
        },
        "relax.transform.FuseOpsByPattern": {
            "bind_constants": True
        }
    }
):
    mod = pipeline(mod)
```

### 5.14.3 Inspecting Pass Results

You can inspect the IRModule after each pass for debugging:

```python
# Manual step-by-step execution for debugging
passes = [
    relax.transform.LegalizeOps(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),
    relax.transform.DeadCodeElimination(),
]

for i, p in enumerate(passes):
    mod = p(mod)
    print(f"\n=== After pass {i}: {p.name} ===")
    print(mod.script())
    print(f"Functions: {list(mod.functions.keys())}")
```

### 5.14.4 Combining with TIR Passes

Relax transformations can be interleaved with TIR-level passes:

```python
from tvm import tir

# Combined Relax + TIR pipeline
combined_pipeline = relax.transform.Sequential([
    # Relax-level transformations
    relax.transform.LegalizeOps(),
    relax.transform.FuseTIR(),
    relax.transform.DeadCodeElimination(),

    # TIR-level transformations on the generated PrimFuncs
    tir.transform.Simplify(),
    tir.transform.StorageRewrite(),
    tir.transform.VectorizeLoop(),
])
```

### 5.14.5 Pass Instrumentation

You can instrument passes for timing and profiling:

```python
import time

class TimingInstrument:
    """Instrument that times each pass."""
    def __init__(self):
        self.timings = {}

    def run_before_pass(self, mod, info):
        self.timings[info.name] = {"start": time.time()}

    def run_after_pass(self, mod, info):
        self.timings[info.name]["end"] = time.time()
        elapsed = self.timings[info.name]["end"] - self.timings[info.name]["start"]
        print(f"Pass {info.name}: {elapsed:.4f}s")

# Use with PassContext
instrument = TimingInstrument()
with tvm.transform.PassContext(instruments=[instrument]):
    mod = pipeline(mod)

# Print all timings
for name, timing in instrument.timings.items():
    print(f"{name}: {timing['end'] - timing['start']:.4f}s")
```

---

## 5.15 Quick Reference: All Relax Transform Passes

| Pass Name | Category | Description |
|-----------|----------|-------------|
| `LegalizeOps` | Legalization | Lower Relax ops to TIR PrimFunc |
| `FuseOps` | Fusion | Graph-level operator fusion |
| `FuseTIR` | Fusion | TIR-level function fusion |
| `FuseOpsByPattern` | Fusion | Pattern-based fusion with DPL |
| `DecomposeOpsForInference` | Decomposition | Decompose ops for inference |
| `DecomposeOpsForTraining` | Decomposition | Decompose ops for training |
| `CanonicalizeBindings` | Canonicalization | Normalize variable bindings |
| `CanonicalizeBindingsForBlockBuilder` | Canonicalization | Normalize BB-generated bindings |
| `FoldConstant` | Optimization | Constant folding |
| `SimplifyExpr` | Optimization | Algebraic simplification |
| `DeadCodeElimination` | Optimization | Remove dead code |
| `SimplifyReshape` | Optimization | Reshape chain simplification |
| `KnowledgeBasedSimplify` | Optimization | Knowledge-based simplification |
| `AlterOpImpl` | Layout/Type | Replace operator implementations |
| `ConvertLayout` | Layout/Type | Layout conversion |
| `NarrowDataType` | Layout/Type | Narrow data types |
| `ToNonDataflow` | Lowering | Convert dataflow to normal form |
| `RemovePurityChecking` | Lowering | Remove purity annotations |
| `VMBuiltinLower` | Lowering | Lower builtins for VM |
| `ComputePrimValue` | Lowering | Compute primitive values |
| `StaticPlanBlockMemory` | Lowering | Static memory planning |
| `AttachGlobalSymbol` | Backend | Attach global symbols |
| `RunCodegen` | Backend | Run external codegen |
| `BackendDispatch` | Backend | Dispatch to backends |
| `MergeCompositeFunctions` | Backend | Merge composite functions |
| `LiftTransformParams` | Params | Lift parameter transforms |
| `BundleModelParams` | Params | Bundle parameters into tuple |
| `ExpandTupleArguments` | Params | Expand tuple arguments |
| `Normalize` | Canonicalization | Normalize IR |
| `Gradient` | Differentiation | Reverse-mode AD |
