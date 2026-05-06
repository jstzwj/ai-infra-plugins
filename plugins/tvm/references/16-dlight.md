# Apache TVM Reference - Chapter 16: DLight Scheduling Rules

This reference covers DLight, TVM's framework for applying pre-defined, high-performance scheduling rules to common computational patterns. Unlike MetaSchedule's automated search, DLight uses deterministic pattern matching to identify computation structures and apply hand-crafted schedules that are known to perform well. DLight provides fast, predictable compilation with near-optimal performance for supported patterns.

---

## 16.1 Overview

### 16.1.1 Motivation

MetaSchedule's evolutionary search can find excellent schedules, but it requires significant tuning time (thousands of trials per task). For production deployment scenarios where compilation speed matters, or for well-understood computational patterns where optimal strategies are already known, DLight provides:

1. **Zero-tuning compilation**: No search is required. The schedule is determined immediately from the program structure.
2. **Deterministic results**: The same input always produces the same schedule, enabling reproducible builds.
3. **Near-optimal performance**: The rules encode expert knowledge about GPU/CPU optimization, matching or exceeding auto-tuning for supported patterns.
4. **Composability**: Rules can be combined and extended for new patterns.

### 16.1.2 DLight Architecture

DLight follows a rule-based architecture where each rule:
1. Inspects the PrimFunc to detect a known pattern
2. If the pattern matches, applies a pre-defined schedule
3. If no pattern matches, falls back to a default strategy

```
PrimFunc
    |
    v
+-------------------+
| Pattern Matcher   |  Examines block structure, loop patterns, access regions
+--------+----------+
         |
   Match found?
    /         \
  Yes          No
   |            |
   v            v
+------+   +---------+
| Apply |   | Try next|  or fallback to default rule
| Rule  |   |  Rule   |
+------+   +---------+
   |
   v
Scheduled PrimFunc
```

### 16.1.3 Basic Usage

```python
import tvm
from tvm import dlight as dl

# Apply DLight GPU rules to an entire IRModule
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Decode(),
        dl.gpu.Fallback(),
    )(mod)
```

---

## 16.2 GPU Rules

### 16.2.1 GEMV Rule

Optimizes General Matrix-Vector Multiplication (GEMV) patterns: `y = A @ x` where A is a matrix and x is a vector. GEMV is memory-bandwidth-bound on GPUs, so the optimization strategy focuses on maximizing memory throughput.

**Pattern characteristics:**
- A reduction over one dimension of a 2D input, producing a 1D output
- The non-reduced dimension maps to the output
- The reduced dimension multiplies corresponding elements and accumulates

**Detection criteria:**
- The block has exactly one spatial axis and one reduce axis
- The spatial axis corresponds to the output dimension
- The reduce axis iterates over the "contraction" dimension

```python
from tvm import dlight as dl

# Apply the GEMV rule
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
```

**Schedule strategy:**

| Step | Transformation | Purpose |
|------|---------------|---------|
| 1 | Bind spatial axis to `blockIdx.x` | Distribute rows across thread blocks |
| 2 | Split spatial axis: `threadIdx.x` for inner | Map threads within a block |
| 3 | Vectorize the innermost load | Maximize memory bandwidth |
| 4 | Unroll the reduction loop | Reduce loop overhead |
| 5 | Apply cache_read if beneficial | Reduce global memory traffic |

**Example GEMV pattern in TVMScript:**

```python
from tvm.script import ir as I, tirx as T

@I.ir_module
class GEMVModule:
    @T.prim_func
    def gemv(
        A: T.Buffer((4096, 4096), "float16"),
        x: T.Buffer((4096,), "float16"),
        y: T.Buffer((4096,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for i, k in T.grid(4096, 4096):
            with T.sblock("gemv"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(A[vi, vk], x[vk])
                T.writes(y[vi])
                with T.init():
                    y[vi] = T.float32(0)
                y[vi] = y[vi] + T.cast(A[vi, vk], "float32") * T.cast(x[vk], "float32")

# Apply DLight GEMV rule
import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(GEMVModule)
```

**Performance characteristics:**
- Achieves >80% of peak memory bandwidth for large GEMV operations
- Performance is bounded by memory bandwidth, not compute throughput
- Optimal tile sizes are determined by the target GPU's memory hierarchy

---

### 16.2.2 Matmul Rule

Optimizes General Matrix-Matrix Multiplication (GEMM) patterns: `C = A @ B`. Matmul is compute-bound for large matrices, so the strategy focuses on maximizing arithmetic throughput through tiling and shared memory usage.

**Pattern characteristics:**
- Two spatial axes (output dimensions) and one reduce axis
- The computation is a sum of products along the reduce axis
- Both input matrices are 2D

**Detection criteria:**
- The block has two spatial axes and one reduce axis
- The spatial axes correspond to the M and N dimensions of the output matrix
- The reduce axis corresponds to the K dimension

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
```

**Schedule strategy:**

The Matmul rule implements a hierarchical tiling strategy inspired by CUTLASS:

| Level | Tile Size (typical) | Mapping |
|-------|---------------------|---------|
| Thread block tile | 128 x 128 | `blockIdx.y`, `blockIdx.x` |
| Warp tile | 64 x 64 | warp-level scheduling |
| Thread tile | 8 x 8 | `threadIdx.y`, `threadIdx.x` |
| Micro-kernel | 4x4 or WMMA | Per-thread computation |

**Detailed transformation steps:**

1. **Cache reads to shared memory**: Load tiles of A and B into shared memory for fast access
2. **Multi-level tiling**: Split M, N, K dimensions into block/warp/thread levels
3. **Thread binding**: Map outer tiles to GPU thread blocks and inner tiles to threads
4. **Pipeline shared memory loads**: Overlap computation with data loading (if supported)
5. **Vectorize memory access**: Use vector loads for global-to-shared transfers
6. **Storage alignment**: Pad shared memory to avoid bank conflicts
7. **Unroll inner loops**: Maximize instruction-level parallelism

```python
from tvm.script import ir as I, tirx as T

@I.ir_module
class MatmulModule:
    @T.prim_func
    def matmul(
        A: T.Buffer((1024, 1024), "float16"),
        B: T.Buffer((1024, 1024), "float16"),
        C: T.Buffer((1024, 1024), "float32"),
    ):
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")

import tvm
from tvm import dlight as dl

# Apply DLight Matmul rule
with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(MatmulModule)
```

**Performance characteristics:**
- Achieves >70% of peak FLOPS for large square matrices (M=N=K >= 512)
- Automatically selects tile sizes based on the target GPU's specifications
- Supports mixed-precision computation (float16 inputs, float32 accumulation)

---

### 16.2.3 Reduction Rule

Optimizes general reduction operations: computing a scalar or lower-dimensional tensor by combining elements along one or more axes.

**Pattern characteristics:**
- One or more spatial axes and one or more reduce axes
- The output has fewer dimensions than the input
- Common operations: sum, max, min, mean, norm

**Detection criteria:**
- The block has at least one reduce axis
- The reduce axis is not part of a matmul pattern (otherwise Matmul rule takes precedence)
- The output is a scalar or has reduced dimensionality

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(mod)
```

**Schedule strategy:**

| Step | Transformation | Purpose |
|------|---------------|---------|
| 1 | Bind spatial axes to `blockIdx` | Distribute independent reductions |
| 2 | Bind reduce axis to `threadIdx` | Enable parallel reduction |
| 3 | Use warp shuffle for reduction | Efficient cross-thread communication |
| 4 | Handle remainder elements | Support non-power-of-2 dimensions |

```python
from tvm.script import ir as I, tirx as T

@I.ir_module
class ReductionModule:
    @T.prim_func
    def row_sum(
        A: T.Buffer((1024, 4096), "float32"),
        B: T.Buffer((1024,), "float32"),
    ):
        for i, j in T.grid(1024, 4096):
            with T.sblock("sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vj]

import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(ReductionModule)
```

---

### 16.2.4 Decode Rule

Optimizes decode-phase operations commonly found in LLM inference. During the decode phase, each step processes a single token and updates the KV cache. The computation is essentially a GEMV-like operation: `output = W @ x` where x is a vector (the current token embedding).

**Pattern characteristics:**
- GEMV-like structure (matrix-vector multiply)
- The "matrix" is a weight tensor from the model
- The "vector" is a single token's representation
- Often includes dequantization for quantized weights

**Detection criteria:**
- The block has one spatial axis and one reduce axis (like GEMV)
- The spatial extent is large (hidden dimension of the model)
- The reduce extent may also be large (for MLP layers)

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Decode())(mod)
```

**Schedule strategy:**

The Decode rule applies similar optimizations to GEMV but with additional considerations for:
- **Batch size = 1**: No batch dimension to parallelize over
- **Weight pre-fetching**: Overlap weight loading with computation
- **Dequantization fusion**: Fuse quantize/dequantize operations into the GEMV kernel

```python
from tvm.script import ir as I, tirx as T

@I.ir_module
class DecodeModule:
    @T.prim_func
    def decode_linear(
        W: T.Buffer((4096, 4096), "float16"),  # Weight matrix
        x: T.Buffer((4096,), "float16"),         # Input vector (single token)
        y: T.Buffer((4096,), "float32"),          # Output vector
    ):
        for i, k in T.grid(4096, 4096):
            with T.sblock("decode_gemm"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(W[vi, vk], x[vk])
                T.writes(y[vi])
                with T.init():
                    y[vi] = T.float32(0)
                y[vi] = y[vi] + T.cast(W[vi, vk], "float32") * T.cast(x[vk], "float32")

import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.Decode())(DecodeModule)
```

---

### 16.2.5 DecodeGEMV Rule

A variant of the Decode rule specifically optimized for the GEMV structure in decode-phase operations. This rule applies when the decode operation has a pure GEMV pattern without additional fused operations.

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.DecodeGEMV())(mod)
```

**Distinction from Decode rule:**
- DecodeGEMV specifically targets the case where the weight matrix is accessed in a regular pattern
- May use different tiling strategies for weight loading
- Can apply more aggressive vectorization for the weight access

---

### 16.2.6 GeneralReduction Rule

Optimizes general reduction patterns that do not match the specific GEMV or Matmul patterns. This includes reductions over multiple dimensions, reductions with non-trivial access patterns, and reductions with epilogue operations.

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(mod)
```

**Schedule strategy:**

1. Analyze the reduction dimensions to determine the optimal thread mapping
2. Bind spatial axes to `blockIdx` for independent output elements
3. Use shared memory for intermediate reduction results if needed
4. Apply cross-thread reduction for the innermost reduce axis

**Example: 2D reduction (row + column):**

```python
from tvm.script import ir as I, tirx as T

@I.ir_module
class GeneralRedModule:
    @T.prim_func
    def norm(
        A: T.Buffer((256, 256, 256), "float32"),
        B: T.Buffer((256,), "float32"),
    ):
        for i, j, k in T.grid(256, 256, 256):
            with T.sblock("sum"):
                vi, vj, vk = T.axis.remap("SRR", [i, j, k])
                T.reads(A[vi, vj, vk])
                T.writes(B[vi])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vj, vk]

import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(GeneralRedModule)
```

---

### 16.2.7 Fallback Rule (GPU)

The GPU fallback rule applies a generic scheduling strategy when no specific rule matches. It ensures that every PrimFunc receives at least a basic GPU schedule.

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Fallback(),  # Catch-all for unmatched patterns
    )(mod)
```

**Fallback strategy:**

1. Find all blocks in the function
2. For element-wise blocks: bind to `blockIdx.x` and `threadIdx.x` with vectorization
3. For reduction blocks: apply basic cross-thread reduction
4. Inline all trivial producers/consumers
5. Bind outermost loop to `blockIdx` and inner loop to `threadIdx`

---

## 16.3 CPU Rules

### 16.3.1 Fallback Rule (CPU)

The CPU fallback rule applies a generic scheduling strategy for CPU targets. It handles all patterns that do not have CPU-specific optimizations.

```python
from tvm import dlight as dl

with tvm.target.Target("cpu"):
    mod = dl.ApplyDefaultSchedule(
        dl.cpu.Fallback(),
    )(mod)
```

**Fallback strategy:**

1. Inline all trivial (element-wise, injective) blocks
2. Parallelize the outermost loop
3. Vectorize the innermost loop with a target-appropriate width
4. Unroll small inner loops

```python
from tvm.script import ir as I, tirx as T
import tvm
from tvm import dlight as dl

@I.ir_module
class CPUModule:
    @T.prim_func
    def elementwise(
        A: T.Buffer((4096, 4096), "float32"),
        B: T.Buffer((4096, 4096), "float32"),
    ):
        for i, j in T.grid(4096, 4096):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * T.float32(2.0) + T.float32(1.0)

# Apply CPU fallback
with tvm.target.Target("cpu"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.cpu.Fallback())(CPUModule)
```

---

## 16.4 Analysis Helpers

DLight uses several internal analysis helpers to detect patterns and make scheduling decisions. These are available for users writing custom rules.

### 16.4.1 `is_broadcast_epilogue`

Checks whether a reduction block has a broadcast epilogue -- a situation where the final operation after reduction broadcasts a value across multiple output elements.

```python
from tvm.dlight import analysis as dl_analysis

# Check if a block has a broadcast epilogue
has_broadcast = dl_analysis.is_broadcast_epilogue(block, spatial_axes, reduce_axis)
```

**Usage in rule matching:**

Broadcast epilogues require special handling because the output elements may not be independent. If a reduction has a broadcast epilogue (e.g., softmax normalization, layer norm), the scheduling strategy must ensure that all threads contributing to the same output element synchronize properly.

**Common patterns with broadcast epilogues:**

| Pattern | Broadcast Dimension | Example |
|---------|---------------------|---------|
| Softmax | Over class dimension | `softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))` |
| LayerNorm | Over hidden dimension | `layernorm(x)[i] = (x[i] - mean) / sqrt(var)` |
| BatchNorm | Over spatial dimensions | `batchnorm(x)[i,j] = (x[i,j] - mean[j]) / sqrt(var[j])` |

### 16.4.2 `detect_dominated_var`

Detects whether a variable's value is uniquely determined by the surrounding loop iteration variables. Used to validate that loop transformations are safe.

```python
from tvm.dlight import analysis as dl_analysis

is_dominated = dl_analysis.detect_dominated_var(expr, var, loop_vars)
```

**Usage in DLight rules:**

Before applying a tiling transformation, the rule checks that buffer indices are dominated by the loop variables. If an index expression uses variables outside the loop nest, the transformation may produce incorrect results.

### 16.4.3 `detect_reduction`

Analyzes a block to determine if it contains a reduction pattern and extracts the reduction axes.

```python
from tvm.dlight import analysis as dl_analysis

reduction_info = dl_analysis.detect_reduction(block)
if reduction_info is not None:
    spatial_axes, reduce_axes = reduction_info
    print(f"Spatial axes: {spatial_axes}")
    print(f"Reduce axes: {reduce_axes}")
```

**Return value:** A tuple `(spatial_axes, reduce_axes)` if the block contains a reduction, or `None` if it does not.

**Details:**

This analysis is the primary pattern-matching mechanism for DLight rules. It examines the block's iteration variables and classifies them as:
- **Spatial**: Iteration variables that map directly to output elements (one-to-one)
- **Reduce**: Iteration variables that are collapsed through an associative reduction operation

The classification determines which DLight rule applies:
- 1 spatial + 1 reduce: GEMV or Decode pattern
- 2 spatial + 1 reduce: Matmul pattern
- N spatial + M reduce (M >= 1): GeneralReduction pattern
- N spatial + 0 reduce: Element-wise (handled by Fallback)

---

## 16.5 Rule Application Flow

### 16.5.1 Pattern Matching Pipeline

When `ApplyDefaultSchedule` is called with multiple rules, it processes each PrimFunc in the IRModule through the following pipeline:

```
For each PrimFunc in IRModule:
    For each rule in the provided list (in order):
        1. Analyze the PrimFunc's block structure
        2. Extract iteration variable classifications (spatial/reduce)
        3. Check access patterns and data dependencies
        4. If the pattern matches:
            a. Create a schedule: sch = tvm.s_tir.Schedule(mod)
            b. Apply the rule's transformation
            c. Return the scheduled module
    If no rule matches:
        Return the original module unchanged
```

**Important:** Rules are tried in the order they are provided. The first matching rule is applied. If a more specific rule is listed after a more general one, the general rule will match first. Always list rules from most specific to most general:

```python
# CORRECT order: specific rules first, fallback last
mod = dl.ApplyDefaultSchedule(
    dl.gpu.GEMV(),              # Most specific: 1 spatial + 1 reduce
    dl.gpu.Decode(),            # Specific: decode-phase GEMV
    dl.gpu.Matmul(),            # Specific: 2 spatial + 1 reduce
    dl.gpu.Reduction(),         # General: any reduction
    dl.gpu.GeneralReduction(),  # Most general reduction
    dl.gpu.Fallback(),          # Catch-all
)(mod)

# WRONG order: fallback would match everything
# mod = dl.ApplyDefaultSchedule(
#     dl.gpu.Fallback(),      # This would match first!
#     dl.gpu.Matmul(),        # Never reached
# )(mod)
```

### 16.5.2 Applying Rules to Specific Functions

To apply DLight rules to a specific PrimFunc rather than the entire module:

```python
import tvm
from tvm import dlight as dl

# Get a specific function from the module
func_name = "my_matmul"
func = mod[func_name]

# Apply the Matmul rule to this specific function
with tvm.target.Target("nvidia/nvidia-a100"):
    rule = dl.gpu.Matmul()
    sch = rule.apply(func, tvm.target.Target("nvidia/nvidia-a100"))
    mod = sch.mod
```

---

## 16.6 Integration with Optimization Pipelines

### 16.6.1 DLight in the Relax Pipeline

DLight can be used as the scheduling backend in the Relax optimization pipeline:

```python
from tvm import relax
from tvm import dlight as dl

# Option 1: Use DLight directly in the pipeline
target = tvm.target.Target("nvidia/nvidia-a100")

# First apply standard Relax optimizations
mod = relax.get_pipeline("zero")(mod)

# Then apply DLight scheduling to the TIR functions
with target:
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Decode(),
        dl.gpu.Fallback(),
    )(mod)

# Build the optimized module
exec = relax.build(mod, target=target)
```

### 16.6.2 Combining DLight with MetaSchedule

DLight and MetaSchedule can be combined: use DLight for well-understood patterns and MetaSchedule for custom or unusual operations.

```python
from tvm import relax, meta_schedule as ms
from tvm import dlight as dl
from tvm.meta_schedule.database import JSONDatabase

# Step 1: Extract tasks
tasks = ms.extract_tasks(mod, target=target)

# Step 2: Classify tasks
dlight_tasks = []    # Tasks that match DLight patterns
tuning_tasks = []    # Tasks that need MetaSchedule tuning

for task in tasks:
    if is_dlight_pattern(task):  # Custom classification logic
        dlight_tasks.append(task)
    else:
        tuning_tasks.append(task)

# Step 3: Apply DLight to matching tasks
with target:
    for task in dlight_tasks:
        mod[task.task_name] = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.Fallback(),
        )(task.mod)[task.task_name]

# Step 4: Tune remaining tasks with MetaSchedule
db = ms.tune_tasks(
    tasks=tuning_tasks,
    target=target,
    max_trials_global=5000,
)

# Step 5: Build the final module
exec = relax.build(mod, target=target)
```

### 16.6.3 DLight as MetaSchedule Default

When using MetaSchedule's `ApplyDefaultSchedule`, DLight rules serve as the default scheduling strategy when no tuning data is available:

```python
from tvm.meta_schedule.tune import tune_tasks
from tvm import dlight as dl

# MetaSchedule will fall back to DLight rules for un-tuned operators
mod = relax.get_pipeline("static_shape_tuning")(
    mod,
    target=target,
    database=db,  # May be empty for first run
    default_schedule=dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Fallback(),
    ),
)
```

---

## 16.7 Performance Characteristics

### 16.7.1 Expected Performance

| Pattern | DLight Performance | MetaSchedule (2000 trials) | Notes |
|---------|-------------------|---------------------------|-------|
| GEMV (M=4096) | ~85% peak BW | ~87% peak BW | Memory-bound, DLight is close to optimal |
| Matmul (1024x1024) | ~70% peak FLOPS | ~75% peak FLOPS | Compute-bound, MetaSchedule slightly better |
| Reduction | ~80% peak BW | ~82% peak BW | Memory-bound, similar performance |
| Element-wise | ~90% peak BW | ~90% peak BW | Trivially parallelizable |
| Decode (LLM) | ~80% peak BW | ~83% peak BW | Memory-bound with specific access pattern |

### 16.7.2 Compilation Time

| Approach | Compilation Time | Tuning Time |
|----------|-----------------|-------------|
| DLight | 1-5 seconds | None |
| MetaSchedule | 1-5 seconds | 30 min - 2 hours |
| Manual Schedule | Hours (development) | None |

### 16.7.3 When DLight Performance is Sufficient

DLight is the recommended approach when:
- The model uses standard operations (linear, attention, normalization, element-wise)
- Compilation speed is important (CI/CD, rapid prototyping)
- The target hardware is a well-known GPU (NVIDIA A100, H100, etc.)
- The model has been previously validated with DLight schedules

### 16.7.4 When to Use MetaSchedule Instead

MetaSchedule is recommended when:
- The model contains custom or unusual operators
- Maximum performance is critical and worth the tuning investment
- The target hardware has unusual characteristics
- The operation has dimensions that don't match standard tiling patterns
- Previous tuning data is available (reducing the effective tuning time)

---

## 16.8 Example: Applying DLight to a Custom Operator

This section demonstrates the complete workflow of applying DLight rules to a custom operator.

### 16.8.1 Custom Softmax Operator

```python
from tvm.script import ir as I, tirx as T
import tvm
from tvm import dlight as dl

@I.ir_module
class SoftmaxModule:
    @T.prim_func
    def softmax(
        A: T.Buffer((128, 1024), "float32"),
        B: T.Buffer((128, 1024), "float32"),
    ):
        # Step 1: Compute max for numerical stability
        max_val = T.alloc_buffer((128,), "float32")
        for i, j in T.grid(128, 1024):
            with T.sblock("max"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(A[vi, vj])
                T.writes(max_val[vi])
                with T.init():
                    max_val[vi] = T.float32("-inf")
                max_val[vi] = T.max(max_val[vi], A[vi, vj])

        # Step 2: Compute exp(x - max)
        exp_val = T.alloc_buffer((128, 1024), "float32")
        for i, j in T.grid(128, 1024):
            with T.sblock("exp"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj], max_val[vi])
                T.writes(exp_val[vi, vj])
                exp_val[vi, vj] = T.exp(A[vi, vj] - max_val[vi])

        # Step 3: Compute sum of exp values
        sum_val = T.alloc_buffer((128,), "float32")
        for i, j in T.grid(128, 1024):
            with T.sblock("sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(exp_val[vi, vj])
                T.writes(sum_val[vi])
                with T.init():
                    sum_val[vi] = T.float32(0)
                sum_val[vi] = sum_val[vi] + exp_val[vi, vj]

        # Step 4: Normalize
        for i, j in T.grid(128, 1024):
            with T.sblock("normalize"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(exp_val[vi, vj], sum_val[vi])
                T.writes(B[vi, vj])
                B[vi, vj] = exp_val[vi, vj] / sum_val[vi]

# Apply DLight rules
with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Reduction(),  # Matches "max" and "sum" blocks
        dl.gpu.Fallback(),   # Matches "exp" and "normalize" blocks
    )(SoftmaxModule)

print("Scheduled softmax kernel:")
print(scheduled_mod.script())
```

### 16.8.2 Custom Fused Attention Operator

```python
from tvm.script import ir as I, tirx as T
import tvm
from tvm import dlight as dl

@I.ir_module
class AttentionModule:
    @T.prim_func
    def attention(
        Q: T.Buffer((1, 8, 128, 64), "float16"),   # [batch, heads, seq, dim]
        K: T.Buffer((1, 8, 128, 64), "float16"),
        V: T.Buffer((1, 8, 128, 64), "float16"),
        O: T.Buffer((1, 8, 128, 64), "float16"),
    ):
        # QK^T: [batch, heads, seq_q, seq_k]
        QK = T.alloc_buffer((1, 8, 128, 128), "float32")
        for b, h, i, j, k in T.grid(1, 8, 128, 128, 64):
            with T.sblock("QK"):
                vb, vh, vi, vj, vk = T.axis.remap("SSSSR", [b, h, i, j, k])
                T.reads(Q[vb, vh, vi, vk], K[vb, vh, vj, vk])
                T.writes(QK[vb, vh, vi, vj])
                with T.init():
                    QK[vb, vh, vi, vj] = T.float32(0)
                QK[vb, vh, vi, vj] = QK[vb, vh, vi, vj] + \
                    T.cast(Q[vb, vh, vi, vk], "float32") * T.cast(K[vb, vh, vj, vk], "float32")

        # Softmax is omitted for brevity (see Section 16.8.1)

        # Attention * V: [batch, heads, seq_q, dim]
        for b, h, i, j, k in T.grid(1, 8, 128, 64, 128):
            with T.sblock("AttnV"):
                vb, vh, vi, vj, vk = T.axis.remap("SSSSR", [b, h, i, j, k])
                T.reads(QK[vb, vh, vi, vk], V[vb, vh, vk, vj])
                T.writes(O[vb, vh, vi, vj])
                with T.init():
                    O[vb, vh, vi, vj] = T.float32(0)
                O[vb, vh, vi, vj] = O[vb, vh, vi, vj] + \
                    QK[vb, vh, vi, vk] * T.cast(V[vb, vh, vk, vj], "float32")

# Apply DLight rules
with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),    # Matches the batched matmul patterns
        dl.gpu.Fallback(),
    )(AttentionModule)
```

---

## 16.9 Writing Custom DLight Rules

Users can write custom DLight rules for domain-specific patterns:

```python
import tvm
from tvm import dlight as dl
from tvm import tir

class MyCustomGPURule(dl.ScheduleRule):
    """A custom DLight rule for a specific computational pattern."""

    def apply(self, func: tir.PrimFunc, target: tvm.target.Target) -> tvm.s_tir.Schedule:
        """
        Apply the scheduling rule to a PrimFunc.

        Parameters
        ----------
        func : tir.PrimFunc
            The function to schedule.
        target : Target
            The compilation target.

        Returns
        -------
        sch : s_tir.Schedule
            The scheduled module, or None if the rule does not match.
        """
        # Step 1: Analyze the function structure
        sch = tvm.s_tir.Schedule(tvm.IRModule.from_expr(func))

        # Step 2: Detect the pattern
        # ... custom pattern matching logic ...

        # Step 3: Apply the schedule
        # ... schedule transformations ...

        return sch

# Use the custom rule
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(
        MyCustomGPURule(),
        dl.gpu.Fallback(),  # Fallback for unmatched patterns
    )(mod)
```

**Guidelines for writing effective DLight rules:**

1. **Be specific in pattern matching**: Return `None` early if the pattern does not match. Do not apply transformations that assume a specific structure without verification.
2. **Use analysis helpers**: Leverage `detect_reduction`, `is_broadcast_epilogue`, and other analysis functions to understand the block structure.
3. **Consider the target**: Check the target's attributes (shared memory size, max threads, etc.) before applying target-specific transformations.
4. **Validate the schedule**: After applying transformations, consider using `verify_gpu_code` to ensure correctness.
5. **Document the expected pattern**: Clearly document what pattern the rule matches and what schedule it produces.

---

## 16.10 Summary

| Component | Description | Key Rules |
|-----------|-------------|-----------|
| **GPU Rules** | Optimized schedules for GPU targets | GEMV, Matmul, Reduction, Decode, DecodeGEMV, GeneralReduction, Fallback |
| **CPU Rules** | Optimized schedules for CPU targets | Fallback |
| **Analysis Helpers** | Pattern detection utilities | `is_broadcast_epilogue`, `detect_dominated_var`, `detect_reduction` |
| **Application** | Rule composition and ordering | `ApplyDefaultSchedule` |
| **Integration** | Pipeline and MetaSchedule integration | Relax pipeline, combined tuning |
