# Apache TVM — Chapter 25: Operator Fusion

This reference covers operator fusion in TVM — the process of combining multiple operators into single kernels to reduce memory bandwidth usage and improve execution performance. Fusion is one of the most critical optimizations in the TVM compilation pipeline.

---

## 25.1 Fusion Overview

### Why Fusion Matters

In a naive execution model, each operator reads its inputs from memory, computes results, and writes outputs back to memory. For deep learning workloads dominated by element-wise and reduction operations, this creates a severe memory bandwidth bottleneck. Operator fusion addresses this by:

- **Eliminating intermediate tensor materialization**: Fused operators pass data through registers or shared memory instead of global memory.
- **Reducing kernel launch overhead**: Fewer kernels means fewer launches, reducing driver and runtime overhead.
- **Enabling joint optimization**: Fused kernels can apply tiling and scheduling that would be impossible across separate kernels.
- **Improving cache utilization**: Data produced by one operator is consumed immediately while still hot in cache.

### Fusion in the TVM Pipeline

Operator fusion occurs at multiple stages in the TVM compilation pipeline:

```
Input Model (PyTorch/ONNX/TFLite)
        |
        v
   Relax Frontend Import
        |
        v
   IRModule (relax::Function + tir::PrimFunc)
        |
        v
   [FuseOps]           -- Graph-level fusion of Relax operators
        |
        v
   [FuseOpsByPattern]  -- Pattern-based fusion for external backends
        |
        v
   [LegalizeOps]       -- Lower Relax ops to TIR PrimFunc
        |
        v
   [FuseTIR]           -- Fuse TIR PrimFunc functions
        |
        v
   [MetaSchedule / DLight]  -- Schedule fused PrimFunc
        |
        v
   Compiled Executable
```

---

## 25.2 Operator Pattern Classification

TVM classifies operators into several pattern types that determine fusion eligibility. The `FUSE_PATTERN` attribute on each operator dictates its fusion behavior.

### Pattern Types

| Pattern | Description | Fusion Behavior |
|---------|-------------|-----------------|
| `kElemWise` | Element-wise operations (add, relu, multiply) | Can fuse with any injective or reduction parent |
| `kBroadcast` | Broadcasting element-wise ops | Similar to kElemWise, allows dimension expansion |
| `kInjective` | One-to-one mapping (reshape, transpose, copy) | Can fuse with any parent; always safe to compose |
| `kCommReduce` | Reduction operations (sum, max, argmax) | Can have element-wise children fused into it |
| `kCommReduceIdx` | Reduction with index output (argmax, argmin) | Limited fusion; index computation complicates merging |
| `kOutEWiseFusable` | Complex ops whose output is element-wise fusable (matmul) | Element-wise children can fuse onto the output |
| `kComplex` | Operations with complex output patterns | Cannot be fused in general |
| `kOpaque` | Black-box operations (external calls) | Cannot be fused |

### Pattern Determination

Each Relax operator registers its fusion pattern via the `FUSE_PATTERN` attribute:

```python
# Example: Element-wise addition is kElemWise
@ir.register_op_attr("relax.add", "FUSE_PATTERN", level=10)
def add_pattern(attrs, args):
    return OpPatternKind.kElemWise

# Example: Matmul is kOutEWiseFusable
@ir.register_op_attr("relax.matmul", "FUSE_PATTERN", level=10)
def matmul_pattern(attrs, args):
    return OpPatternKind.kOutEWiseFusable

# Example: Reshape is kInjective
@ir.register_op_attr("relax.reshape", "FUSE_PATTERN", level=10)
def reshape_pattern(attrs, args):
    return OpPatternKind.kInjective
```

---

## 25.3 Fusion Rules

Fusion is governed by rules that determine which operator patterns can be combined. The rules follow a dominator-based analysis on the computation graph.

### Basic Fusion Rules

1. **Element-wise into Element-wise**: Always valid. Two element-wise ops can be merged into a single kernel where each thread computes both operations.

2. **Element-wise into Reduction (post-fusion)**: A reduction operation followed by element-wise operations can be fused. The element-wise work is appended after the reduction writes its output.

3. **Element-wise into OutEWiseFusable (post-fusion)**: Operations like matmul produce outputs that can have element-wise operations fused directly onto them. For example, `matmul + bias + relu` can become a single kernel.

4. **Injective into anything**: Injective operations (reshape, transpose, squeeze) can always be fused with their consumers or producers since they only change indexing, not data.

5. **No fusion with Opaque or Complex**: These patterns block fusion across their boundary.

### Fusion Direction

Fusion typically proceeds **upward** (consumer fuses into producer) in TVM. A consumer node checks whether it can fuse with its producer based on the pattern types:

```
Producer Pattern    Consumer Pattern    Can Fuse?
-----------------------------------------------------
kElemWise           kElemWise           YES
kElemWise           kBroadcast          YES
kElemWise           kInjective          YES
kElemWise           kCommReduce         YES (consumer reduces)
kBroadcast          kElemWise           YES
kInjective          kElemWise           YES
kCommReduce         kElemWise           YES (post-fusion)
kOutEWiseFusable    kElemWise           YES (post-fusion)
kOutEWiseFusable    kBroadcast          YES (post-fusion)
kComplex            *                   NO
kOpaque             *                   NO
*                   kComplex            NO
*                   kOpaque             NO
```

---

## 25.4 FuseOps Pass

The `FuseOps` pass is the primary graph-level fusion transformation for Relax functions. It analyzes the dataflow graph and groups operators into fused subgraphs.

### API

```python
import tvm
from tvm import relax

# Apply FuseOps with a specific fuse depth limit
mod = relax.transform.FuseOps(fuse_opt_level=0)(mod)

# The fuse_opt_level controls the maximum depth of fusion:
#   0 = disable fusion
#   1 = conservative fusion (only trivial cases)
#   2 = default fusion level
#   3+ = aggressive fusion
```

### How FuseOps Works

The pass operates in these phases:

1. **Graph construction**: Build a directed acyclic graph (DAG) from the Relax function where each node is an operator call.

2. **Pattern assignment**: Each node is assigned a fusion pattern type based on the `FUSE_PATTERN` attribute of its operator.

3. **Dominator analysis**: Compute the post-dominator tree of the DAG. A node `B` post-dominates node `A` if every path from `A` to the output passes through `B`. The post-dominator is the natural fusion point.

4. **Fusion grouping**: Walk the dominator tree and decide which nodes to fuse based on pattern compatibility rules. Each group becomes a single fused subgraph.

5. **Subgraph extraction**: Create new `relax.Call` nodes that invoke fused sub-functions containing the grouped operators.

### Example: Before Fusion

```python
@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((1, 128), "float32"),
        w1: R.Tensor((128, 64), "float32"),
        b1: R.Tensor((64,), "float32"),
    ) -> R.Tensor((1, 64), "float32"):
        with R.dataflow():
            # Each operator is a separate call
            lv0 = R.matmul(x, w1)               # kOutEWiseFusable
            lv1 = R.add(lv0, b1)                 # kElemWise
            lv2 = R.nn.relu(lv1)                 # kElemWise
            R.output(lv2)
        return lv2
```

### Example: After FuseOps

```python
@I.ir_module
class Module:
    @R.function
    def fused_matmul_add_relu(
        x: R.Tensor((1, 128), "float32"),
        w1: R.Tensor((128, 64), "float32"),
        b1: R.Tensor((64,), "float32"),
    ) -> R.Tensor((1, 64), "float32"):
        with R.dataflow():
            lv0 = R.matmul(x, w1)
            lv1 = R.add(lv0, b1)
            lv2 = R.nn.relu(lv1)
            R.output(lv2)
        return lv2

    @R.function
    def main(
        x: R.Tensor((1, 128), "float32"),
        w1: R.Tensor((128, 64), "float32"),
        b1: R.Tensor((64,), "float32"),
    ) -> R.Tensor((1, 64), "float32"):
        with R.dataflow():
            lv2 = R.call_tir(
                cls.fused_matmul_add_relu,
                (x, w1, b1),
                out_sinfo=R.Tensor((1, 64), "float32"),
            )
            R.output(lv2)
        return lv2
```

The three separate operators are grouped into a single fused subgraph function. After legalization, this becomes a single TIR kernel.

### Fusion Depth Control

The `fuse_opt_level` parameter indirectly controls how aggressively operators are fused by limiting the depth of the dominator tree traversal:

```python
# Disable fusion entirely
mod = relax.transform.FuseOps(fuse_opt_level=0)(mod)

# Default fusion
mod = relax.transform.FuseOps(fuse_opt_level=2)(mod)

# Aggressive fusion (deeper chains, larger fused kernels)
mod = relax.transform.FuseOps(fuse_opt_level=3)(mod)
```

---

## 25.5 FuseTIR Pass

The `FuseTIR` pass operates at the TIR (Tensor IR) level. After `LegalizeOps` has converted Relax operators into TIR `PrimFunc` functions, `FuseTIR` can merge multiple `PrimFunc` calls into a single combined function.

### When FuseTIR Runs

```
FuseOps        -- graph-level grouping
    |
    v
LegalizeOps    -- relax ops -> TIR PrimFunc
    |
    v
FuseTIR        -- merge TIR PrimFunc calls
```

### API

```python
from tvm import relax

# Apply FuseTIR
mod = relax.transform.FuseTIR()(mod)
```

### How FuseTIR Works

1. **Identify call_tir chains**: Find sequences of `R.call_tir` or `R.call_dps_packed` that can be merged.

2. **Check fusion eligibility**: Verify that the TIR functions are compatible for merging. This includes checking:
   - No conflicting buffer bindings
   - No side effects that require ordering
   - Compatible parallelism patterns (e.g., element-wise loops can be merged)

3. **Merge PrimFunc bodies**: Combine the loop nests of compatible functions into a single `PrimFunc`. Intermediate buffers that were outputs of one function and inputs of the next are allocated locally within the merged function.

4. **Update call sites**: Replace the chain of `call_tir` calls with a single call to the merged function.

### Example: Before FuseTIR

```python
@I.ir_module
class Module:
    @T.prim_func
    def add(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((128,), "float32"),
    ):
        for i in range(128):
            with T.sblock("add"):
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]

    @T.prim_func
    def relu(
        C: T.Buffer((128,), "float32"),
        D: T.Buffer((128,), "float32"),
    ):
        for i in range(128):
            with T.sblock("relu"):
                vi = T.axis.spatial(128, i)
                D[vi] = T.max(C[vi], T.float32(0.0))

    @R.function
    def main(
        x: R.Tensor((128,), "float32"),
        y: R.Tensor((128,), "float32"),
    ) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            lv = R.call_tir(cls.add, (x, y), out_sinfo=R.Tensor((128,), "float32"))
            lv2 = R.call_tir(cls.relu, (lv,), out_sinfo=R.Tensor((128,), "float32"))
            R.output(lv2)
        return lv2
```

### Example: After FuseTIR

```python
@I.ir_module
class Module:
    @T.prim_func
    def fused_add_relu(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        D: T.Buffer((128,), "float32"),
    ):
        # Intermediate buffer C is allocated locally
        C = T.alloc_buffer((128,), "float32")
        for i in range(128):
            with T.sblock("add"):
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]
        for i in range(128):
            with T.sblock("relu"):
                vi = T.axis.spatial(128, i)
                D[vi] = T.max(C[vi], T.float32(0.0))

    @R.function
    def main(
        x: R.Tensor((128,), "float32"),
        y: R.Tensor((128,), "float32"),
    ) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            lv2 = R.call_tir(
                cls.fused_add_relu,
                (x, y),
                out_sinfo=R.Tensor((128,), "float32"),
            )
            R.output(lv2)
        return lv2
```

The intermediate buffer `C` becomes a local allocation within the fused function, eliminating the need to write and read it from global memory.

### FuseTIR Conditions

FuseTIR will only merge functions that satisfy these conditions:

- **Same dataflow block**: All calls must be within the same `R.dataflow()` block.
- **Acyclic dependency**: The call chain must form a DAG without cycles.
- **Compatible loop structure**: The loop nests must be mergeable (e.g., both spatial over the same dimensions).
- **No external side effects**: Functions must be pure (no I/O, no global state mutation).

---

## 25.6 FuseOpsByPattern

`FuseOpsByPattern` is a pattern-based fusion pass designed primarily for external backend integration (BYOC). Instead of using dominator-based analysis, it matches registered fusion patterns against the computation graph.

### API

```python
from tvm import relax
from tvm.relax.dpl import PatternContext

# Define patterns
patterns = [
    ("cutlass.matmul_bias_relu", my_matmul_bias_relu_pattern),
    ("cublas.matmul", my_matmul_pattern),
]

# Apply pattern-based fusion
mod = relax.transform.FuseOpsByPattern(
    patterns=patterns,
    bind_constants=True,
    annotate_codegen=True,
)(mod)
```

### Pattern Definition

Patterns are defined using the Dataflow Pattern Language (DPL):

```python
from tvm.relax.dpl.pattern import (
    is_op,
    is_tuple_get_item,
    make_fused_bias_activation_pattern,
    wildcard,
)

# Simple matmul pattern
def matmul_pattern():
    x = wildcard()
    w = wildcard()
    return is_op("relax.matmul")(x, w)

# Matmul + bias + activation pattern
def matmul_bias_relu_pattern():
    x = wildcard()
    w = wildcard()
    b = wildcard()
    matmul = is_op("relax.matmul")(x, w)
    bias = is_op("relax.add")(matmul, b)
    return is_op("relax.nn.relu")(bias)

# More complex: conv2d + batch_norm + relu
def conv_bn_relu_pattern():
    x = wildcard()
    w = wildcard()
    gamma = wildcard()
    beta = wildcard()
    mean = wildcard()
    var = wildcard()

    conv = is_op("relax.nn.conv2d")(x, w)
    bn = is_op("relax.nn.batch_norm")(conv, gamma, beta, mean, var)
    bn_out = is_tuple_get_item(bn, 0)
    return is_op("relax.nn.relu")(bn_out)
```

### Pattern Registration

For external backend integration, patterns are registered as part of the backend's pattern table:

```python
# In tvm/contrib/cutlass.py
def get_patterns():
    """Return CUTLASS fusion patterns."""
    patterns = [
        ("cutlass.matmul", _matmul_pattern()),
        ("cutlass.matmul_bias", _matmul_bias_pattern()),
        ("cutlass.matmul_bias_relu", _matmul_bias_relu_pattern()),
        ("cutlass.matmul_bias_gelu", _matmul_bias_gelu_pattern()),
        ("cutlass.conv2d_bias_relu", _conv2d_bias_relu_pattern()),
    ]
    return patterns

# In tvm/contrib/cublas.py
def get_patterns():
    """Return cuBLAS fusion patterns."""
    return [
        ("cublas.matmul", _matmul_pattern()),
        ("cublas.batch_matmul", _batch_matmul_pattern()),
    ]
```

### Pattern Matching and Grouping

When `FuseOpsByPattern` runs, it:

1. **Registers patterns**: Collects all patterns from the provided list.

2. **Matches patterns against graph**: For each pattern, walks the computation graph and finds all matching subgraphs. Matches are non-overlapping — once a set of nodes is matched by one pattern, they are not available for another pattern.

3. **Prioritizes patterns**: Patterns earlier in the list take priority. This allows more specific patterns to be matched before more general ones.

4. **Groups matched nodes**: Each match creates a group of operators that will be fused into a single sub-function.

5. **Annotates for codegen**: If `annotate_codegen=True`, matched groups are annotated with the pattern name and target backend, directing subsequent code generation to the appropriate external compiler.

### Sub-Function Creation

Matched groups are extracted into new functions:

```python
# Before FuseOpsByPattern
@R.function
def main(x: R.Tensor((1, 128), "float32"),
         w: R.Tensor((128, 64), "float32"),
         b: R.Tensor((64,), "float32")) -> R.Tensor((1, 64), "float32"):
    with R.dataflow():
        lv0 = R.matmul(x, w)
        lv1 = R.add(lv0, b)
        lv2 = R.nn.relu(lv1)
        R.output(lv2)
    return lv2

# After FuseOpsByPattern with cutlass.matmul_bias_relu pattern
@R.function
def main(x: R.Tensor((1, 128), "float32"),
         w: R.Tensor((128, 64), "float32"),
         b: R.Tensor((64,), "float32")) -> R.Tensor((1, 64), "float32"):
    with R.dataflow():
        lv2 = R.call_tir(
            cls.fused_matmul_add_relu_cutlass,
            (x, w, b),
            out_sinfo=R.Tensor((1, 64), "float32"),
            attrs={"operator_name": "cutlass.matmul_bias_relu"},
        )
        R.output(lv2)
    return lv2
```

---

## 25.7 Fusion Patterns for Specific Backends

### CUTLASS Fusion Patterns

CUTLASS (CUDA Templates for Linear Algebra Subroutines) supports several fused GEMM patterns:

```python
# CUTLASS pattern table
cutlass_patterns = [
    # GEMM + bias + activation
    "cutlass.matmul_bias_relu",
    "cutlass.matmul_bias_gelu",
    "cutlass.matmul_bias_sigmoid",

    # GEMM + epilogue fusion
    "cutlass.matmul_bias",
    "cutlass.matmul",

    # Convolution patterns
    "cutlass.conv2d_bias_relu",
    "cutlass.conv2d_bias",

    # Attention patterns
    "cutlass.attention",
]
```

CUTLASS enables **epilogue fusion** where the activation function is computed in the same kernel as the GEMM without writing the intermediate result to memory:

```python
# CUTLASS generates a single kernel for:
#   C = alpha * A @ B + beta * bias
#   D = activation(C)
# Instead of:
#   C = A @ B           (kernel 1)
#   D = C + bias        (kernel 2)
#   E = relu(D)         (kernel 3)
```

### cuBLAS Fusion Patterns

cuBLAS primarily handles matrix multiplication patterns:

```python
cublas_patterns = [
    "cublas.matmul",           # FP16/FP32/FP64 matmul
    "cublas.batch_matmul",     # Batched matmul
    "cublas.matmul_bias",      # Matmul + bias
    "cublas.int8_matmul",      # INT8 quantized matmul
]
```

### cuDNN Fusion Patterns

cuDNN provides fusion patterns for convolution workloads:

```python
cudnn_patterns = [
    "cudnn.conv2d",
    "cudnn.conv2d_bias",
    "cudnn.conv2d_bias_relu",
    "cudnn.conv2d_bias_sigmoid",
    "cudnn.conv2d_bias_batch_norm",
    "cudnn.conv2d_bias_add_relu",   # ResNet skip-connection pattern
]
```

### Pattern Priority

When multiple backends can handle the same pattern, priority is determined by the order in the pattern list. More specialized patterns should come first:

```python
# Correct ordering: specific patterns first
all_patterns = [
    # CUTLASS patterns (more fusion capability)
    ("cutlass.matmul_bias_relu", cutlass_matmul_bias_relu()),
    ("cutlass.matmul_bias", cutlass_matmul_bias()),
    ("cutlass.matmul", cutlass_matmul()),

    # cuBLAS patterns (fallback for basic matmul)
    ("cublas.matmul", cublas_matmul()),
]
```

---

## 25.8 Fusion and Dataflow Blocks

### Dataflow Block Scope

In Relax IR, fusion is limited to **within a single dataflow block**. A dataflow block (`R.dataflow()`) marks a region of pure computation where operators have no side effects and can be freely reordered and fused.

```python
@R.function
def main(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
    # Outside dataflow block: no fusion here
    lv0 = R.unique(x, sorted=True)  # side-effecting op

    with R.dataflow():
        # Inside dataflow block: fusion happens here
        lv1 = R.add(lv0, lv0)       # fusable
        lv2 = R.nn.relu(lv1)        # fusable with lv1
        R.output(lv2)

    # Outside dataflow block: no fusion here
    lv3 = R.print(lv2)  # side-effecting op
    return lv2
```

### Multiple Dataflow Blocks

If a function has multiple dataflow blocks, fusion is applied independently within each block:

```python
@R.function
def multi_block(
    x: R.Tensor((128,), "float32"),
) -> R.Tensor((128,), "float32"):
    with R.dataflow():
        # Fusion group 1: add + relu
        lv1 = R.add(x, x)
        lv2 = R.nn.relu(lv1)
        R.output(lv2)

    # Side-effecting operation acts as a barrier
    lv3 = R.assert_op(lv2 > 0)

    with R.dataflow():
        # Fusion group 2: multiply + sigmoid (independent of group 1)
        lv4 = R.multiply(lv2, lv2)
        lv5 = R.sigmoid(lv4)
        R.output(lv5)

    return lv5
```

### Dataflow Block Requirements

For a region to be a valid dataflow block:

1. **Pure operations only**: All operations within the block must be pure (no side effects, no I/O).
2. **Single output**: Exactly one `R.output()` statement specifying the block's outputs.
3. **Acyclic**: The dataflow must form a directed acyclic graph.
4. **No control flow**: No `if/else` or loops within the dataflow block (use `R.call_tir` with conditionals inside TIR instead).

---

## 25.9 Cross-Level Fusion

### Relax + TIR Fusion

TVM's two-level IR (Relax and TIR) enables cross-level fusion where graph-level information guides low-level kernel fusion:

```
Relax Level                    TIR Level
-----------                    ---------
FuseOps groups ops         --> FuseTIR merges PrimFuncs
                              |
                              v
                           MetaSchedule/DLight
                           optimizes the fused kernel
```

### Fusion Pipeline Integration

The typical fusion pipeline combines multiple passes:

```python
from tvm import relax

def fusion_pipeline(mod, target):
    """Complete fusion pipeline."""

    # Step 1: Graph-level fusion
    mod = relax.transform.FuseOps(fuse_opt_level=2)(mod)

    # Step 2: Legalize to TIR
    mod = relax.transform.LegalizeOps()(mod)

    # Step 3: TIR-level fusion
    mod = relax.transform.FuseTIR()(mod)

    # Step 4: Apply DLight or MetaSchedule for scheduling
    mod = relax.transform.MetaScheduleApplyDatabase()(mod)

    return mod
```

### Built-in Pipeline Integration

The `relax.get_pipeline()` function applies fusion as part of the standard optimization pipeline:

```python
# The "zero" pipeline includes FuseOps and FuseTIR
mod = relax.get_pipeline("zero")(mod)

# The "static_shape_tensor" pipeline is optimized for
# models with static shapes
mod = relax.get_pipeline("static_shape_tensor")(mod)
```

---

## 25.10 Fusion Limitations and Constraints

### Known Limitations

1. **No cross-dataflow-block fusion**: Operators in different dataflow blocks cannot be fused. This is a fundamental design constraint enforced by the Relax IR semantics.

2. **No dynamic shape fusion**: Fusion analysis may be conservative when symbolic (dynamic) shapes are present. Some fusion opportunities are missed because the compiler cannot prove equivalence at all dynamic shape values.

3. **Reduction fusion limitations**: Only element-wise operations can be fused after a reduction. Two reductions cannot be fused into a single kernel unless they operate on the same input with the same reduction axes.

4. **Memory layout constraints**: Fusion may be inhibited if operators require incompatible memory layouts (e.g., one requires row-major and another requires column-major).

5. **Size constraints**: Extremely large fused kernels may exceed register or shared memory limits on GPU targets, requiring the compiler to split them.

6. **External function calls**: `R.call_dps_packed` and `R.call_pure_packed` calls to external functions cannot participate in TIR-level fusion because the compiler does not have access to their implementation.

### Workarounds

```python
# If fusion is too conservative, try increasing fuse_opt_level
mod = relax.transform.FuseOps(fuse_opt_level=3)(mod)

# For specific patterns, use FuseOpsByPattern with custom patterns
from tvm.relax.dpl.pattern import wildcard, is_op
x = wildcard()
w = wildcard()
my_pattern = is_op("relax.nn.conv2d")(x, w)
patterns = [("custom.conv2d", my_pattern)]
mod = relax.transform.FuseOpsByPattern(patterns)(mod)

# For TIR-level fusion issues, ensure intermediate buffers have
# compatible shapes and types
```

### Debugging Fusion

```python
# Print the IR after each fusion pass to inspect results
import tvm

# After FuseOps
mod_fused = relax.transform.FuseOps()(mod)
print(mod_fused.script())

# After LegalizeOps
mod_legal = relax.transform.LegalizeOps()(mod_fused)
print(mod_legal.script())

# After FuseTIR
mod_tir_fused = relax.transform.FuseTIR()(mod_legal)
print(mod_tir_fused.script())

# Check which operators were fused by inspecting function names
for gv in mod_tir_fused.functions:
    print(f"Function: {gv}")
```

### Fusion and Auto-Tuning Interaction

When using MetaSchedule for auto-tuning, fusion decisions interact with schedule search:

```python
# MetaSchedule can explore whether fusing two ops is beneficial
# by trying different tiling strategies
with tvm.transform.PassContext(
    config={
        "relay.backend.use_meta_schedule": True,
        "relay.backend.meta_schedule.apply_name": "fuse_ops_tune",
    }
):
    mod = relax.get_pipeline("zero")(mod)
```

The auto-tuner may discover that for certain input sizes, it is better to keep operators separate (for better parallelism) rather than fusing them. This is one of the advantages of combining MetaSchedule with fusion — the schedule search can compensate for suboptimal fusion decisions.

---

## 25.11 Complete Fusion Example

### End-to-End Fusion for a Two-Layer MLP

```python
import tvm
from tvm import relax
from tvm.script import ir as I, tir as T, relax as R

# Before optimization
@I.ir_module
class MLPBefore:
    @R.function
    def main(
        x: R.Tensor((1, 784), "float32"),
        w1: R.Tensor((784, 256), "float32"),
        b1: R.Tensor((256,), "float32"),
        w2: R.Tensor((256, 10), "float32"),
        b2: R.Tensor((10,), "float32"),
    ) -> R.Tensor((1, 10), "float32"):
        with R.dataflow():
            # Layer 1: matmul + bias + relu
            lv0 = R.matmul(x, w1, out_dtype="float32")
            lv1 = R.add(lv0, b1)
            lv2 = R.nn.relu(lv1)

            # Layer 2: matmul + bias
            lv3 = R.matmul(lv2, w2, out_dtype="float32")
            lv4 = R.add(lv3, b2)
            R.output(lv4)
        return lv4

# Apply the fusion pipeline
mod = MLPBefore
mod = relax.transform.FuseOps(fuse_opt_level=2)(mod)
mod = relax.transform.LegalizeOps()(mod)
mod = relax.transform.FuseTIR()(mod)

# After optimization, we expect:
# - Layer 1: fused_matmul_add_relu (single kernel)
# - Layer 2: fused_matmul_add (single kernel)
# The intermediate tensor lv2 is still materialized between
# the two fused kernels because it crosses a reduction boundary
# (matmul is kOutEWiseFusable, not kElemWise)

print(mod.script())
```

This example demonstrates how a two-layer MLP is transformed from 6 separate operator calls into 2 fused kernel calls, reducing the number of memory round-trips from 6 to 2 for the intermediate activations.

---

## 25.12 Summary

| Pass | Level | Input | Output | Key Mechanism |
|------|-------|-------|--------|---------------|
| `FuseOps` | Relax graph | Relax operators | Grouped subgraphs | Dominator tree analysis |
| `FuseOpsByPattern` | Relax graph | Relax operators | Pattern-matched groups | DPL pattern matching |
| `FuseTIR` | TIR | PrimFunc calls | Merged PrimFunc | Loop nest merging |
| `LegalizeOps` | Relax to TIR | Relax ops | TIR PrimFunc | Per-op legalization |

The fusion system in TVM is multi-layered and composable. By combining graph-level fusion (FuseOps, FuseOpsByPattern) with TIR-level fusion (FuseTIR), TVM can generate highly optimized kernels that minimize memory traffic and maximize computational throughput.
