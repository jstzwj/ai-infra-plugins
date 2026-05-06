# Apache TVM Reference - Chapter 13: TIR Analysis Passes

This reference covers the TIR (TensorIR) analysis infrastructure in Apache TVM. Analysis passes inspect TIR programs to compute properties needed by transformation and scheduling decisions. They are read-only: they never modify the IR, but extract information such as workspace requirements, computational cost, access regions, variable usage, and correctness constraints.

---

## 13.1 Overview of TIR Analysis

TIR analysis passes serve as the foundation for all optimization decisions in TVM. Before a transformation can safely rewrite a loop nest, it must understand the data access patterns, loop bounds, memory requirements, and structural properties of the program. The analysis module provides these capabilities through a collection of pure functions that accept TIR statements or IRModules and return analysis results.

The analysis infrastructure is organized into two primary modules:

| Module | Purpose |
|--------|---------|
| `tvm.tirx.analysis` | Core TIR analysis passes operating on `tirx.PrimFunc` and `tirx.IRModule` |
| `tvm.s_tir.analysis` | Analysis helpers specifically designed to support scheduling decisions and pattern detection |

---

## 13.2 `tirx.analysis` Module

The `tirx.analysis` module provides the core analysis passes. These are used extensively by the lowering pipeline, the MetaSchedule auto-tuner, and the DLight rule system.

### 13.2.1 `calculate_workspace`

Calculates the total workspace (scratchpad) memory size required by a PrimFunc. Workspace memory is temporary storage allocated during kernel execution for intermediate results that do not fit in registers.

```python
from tvm.tirx import analysis

# Calculate workspace for a PrimFunc
workspace_bytes = analysis.calculate_workspace(func)
print(f"Required workspace: {workspace_bytes} bytes")
```

**Signature:**
```python
def calculate_workspace(func: tir.PrimFunc) -> int
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `tir.PrimFunc` | The PrimFunc to analyze |

**Return value:** An integer representing the total workspace size in bytes.

**Details:**

The analysis traverses all `Allocate` nodes in the PrimFunc and sums up the sizes of buffers that are allocated with the `"global"` or workspace storage scope. It accounts for:
- The data type size (e.g., `float32` = 4 bytes, `float16` = 2 bytes)
- The total number of elements across all dimensions
- Nested allocations (the maximum over concurrent allocations, not the sum)

**Usage in the compilation pipeline:**

```python
import tvm
from tvm import tirx
from tvm.script import ir as I, tirx as T

@I.ir_module
class MyModule:
    @T.prim_func
    def matmul_relu(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "matmul_relu"})
        # Shared memory workspace for tiled computation
        A_shared = T.alloc_buffer((32, 32), "float32", scope="shared")
        B_shared = T.alloc_buffer((32, 32), "float32", scope="shared")
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# Analyze the module
func = MyModule["matmul_relu"]
workspace = tirx.analysis.calculate_workspace(func)
print(f"Workspace needed: {workspace} bytes")
```

---

### 13.2.2 `calculate_constant_bytes`

Calculates the total bytes of constant (read-only) memory referenced by a PrimFunc. Constants are buffers that are only read during execution, such as weight matrices, bias vectors, and lookup tables.

```python
constant_bytes = analysis.calculate_constant_bytes(func)
print(f"Constant memory: {constant_bytes} bytes")
```

**Signature:**
```python
def calculate_constant_bytes(func: tir.PrimFunc) -> int
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `tir.PrimFunc` | The PrimFunc to analyze |

**Return value:** An integer representing total constant memory in bytes.

**Details:**

This analysis identifies buffer parameters that are only read (never written to) within the PrimFunc. It computes the total size of these read-only buffers. This information is critical for:
- Determining whether constant data fits in specialized memory (e.g., constant cache on GPUs)
- Guiding cache read / cache write decisions during scheduling
- Estimating memory bandwidth requirements for the kernel

```python
from tvm.tirx import analysis

# For a convolution kernel with weight and bias as constants
func = module["conv2d"]
const_bytes = analysis.calculate_constant_bytes(func)
print(f"Constant data: {const_bytes} bytes")
# This might include weights + bias: e.g., 3*3*64*64*4 + 64*4 = 147712 bytes
```

---

### 13.2.3 `estimate_tir_flops`

Estimates the number of floating-point operations (FLOPs) in a PrimFunc. This is a static analysis that counts arithmetic operations by traversing the TIR AST.

```python
flops = analysis.estimate_tir_flops(func)
print(f"Estimated FLOPs: {flops}")
```

**Signature:**
```python
def estimate_tir_flops(func: tir.PrimFunc) -> int
```

**Return value:** An integer representing the estimated total number of FLOPs.

**Details:**

The analysis counts the following operations:
- **Add, Subtract, Multiply, Divide** -- counted as 1 FLOP each
- **Multiply-Add (fused)** -- counted as 2 FLOPs (multiply + add)
- **Exponential, Logarithm, Sqrt, Pow** -- implementation-defined cost (typically higher)
- **Comparison and logical operations** -- typically not counted as FLOPs
- **Type casts** -- not counted

The estimation multiplies the operation count by the loop trip counts to arrive at the total. For loops with symbolic bounds, the analysis uses the `arith.Analyzer` to simplify and, if possible, evaluate the bound expressions.

```python
from tvm.script import ir as I, tirx as T
from tvm.tirx import analysis

@I.ir_module
class MatmulModule:
    @T.prim_func
    def matmul(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

flops = analysis.estimate_tir_flops(MatmulModule["matmul"])
# 128 * 128 * 128 multiply-adds = 128 * 128 * 128 * 2 = 4,194,304 FLOPs
print(f"Estimated FLOPs: {flops}")
```

This metric is used by MetaSchedule to estimate computational density and guide search prioritization.

---

### 13.2.4 `get_block_access_region`

Returns the buffer access regions of a TIR block. Access regions describe which portions of each buffer are read from, written to, or both, within the scope of a block.

```python
regions = analysis.get_block_access_region(block, func_buffer_var_map)
```

**Signature:**
```python
def get_block_access_region(
    block: tir.Block,
    buffer_var_map: Dict[tir.Var, tir.Buffer],
) -> List[List[tir.Range]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block` | `tir.Block` | The TIR block to analyze |
| `buffer_var_map` | `Dict[tir.Var, tir.Buffer]` | Mapping from buffer variables to buffer objects |

**Return value:** A list of three lists of `tir.Range` objects:
1. **Read regions** -- buffer regions that are only read
2. **Write regions** -- buffer regions that are only written
3. **Read-write regions** -- buffer regions that are both read and written

**Details:**

Access region analysis is fundamental for:
- **Correctness of scheduling**: Transformations must preserve data dependencies. Knowing which regions are read and written allows the scheduler to verify that a transformation is safe.
- **Cache optimization**: Determines which data must be loaded into shared memory or registers for a given computation tile.
- **Fusion decisions**: Two blocks can be fused if their access regions are compatible (no write-after-write or write-after-read hazards in the fused schedule).

```python
import tvm
from tvm import tirx
from tvm.tirx import analysis

# Get access regions for a specific block in a schedule
sch = tvm.s_tir.Schedule(mod)
block_rv = sch.get_block("C")
block_sref = sch.get(block_rv)

# Build buffer var map from the function
func = mod["main"]
buffer_var_map = {}
for param in func.params:
    buf = func.buffer_map[param]
    buffer_var_map[buf.data] = buf

# Get the actual Block IR node
block_stmt = block_sref.stmt
regions = analysis.get_block_access_region(block_stmt, buffer_var_map)

reads, writes, readwrites = regions
print(f"Read regions: {reads}")
print(f"Write regions: {writes}")
print(f"Read-write regions: {readwrites}")
```

---

### 13.2.5 `get_var_touch`

Returns the set of variables that a statement "touches" -- that is, variables whose values are read or written by the statement.

```python
touched_vars = analysis.get_var_touch(stmt)
```

**Signature:**
```python
def get_var_touch(stmt: tir.Stmt) -> Set[tir.Var]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `stmt` | `tir.Stmt` | The TIR statement to analyze |

**Return value:** A `Set[tir.Var]` containing all variables referenced (read or written) by the statement.

**Details:**

This analysis recursively traverses the statement tree and collects all `tir.Var` nodes. It distinguishes from `get_block_access_region` in that it operates at the variable level rather than the buffer region level. Common use cases include:

- **Dead code elimination**: If a variable is written but never read by any subsequent statement, it is dead.
- **Dependency analysis**: Understanding which loop variables affect a computation.
- **Register pressure estimation**: Counting the number of live variables at each program point.

```python
from tvm.tirx import analysis

# Analyze which variables are touched by a loop body
stmt = func.body  # or any tir.Stmt
touched = analysis.get_var_touch(stmt)
print(f"Variables touched: {touched}")
# Example output: {Var(i, int32), Var(j, int32), Var(k, int32)}
```

---

### 13.2.6 `verify_build_simplified`

Verifies that a PrimFunc is in a "simplified" form suitable for code generation. A simplified TIR function has normalized structures that the code generator can reliably translate to target code.

```python
analysis.verify_build_simplified(func)
```

**Signature:**
```python
def verify_build_simplified(func: tir.PrimFunc) -> None
```

**Details:**

This pass raises `tvm.error.TVMError` if the function does not meet the simplification requirements. The requirements include:

1. **No complex buffer indices**: All buffer access indices must be affine expressions of loop variables (no arbitrary expressions).
2. **Normalized loop structure**: Loops must have a unit stride and start from 0 (i.e., `for i in range(N)` form).
3. **Block structure**: All computation must be inside blocks with properly defined `reads`, `writes`, and `axis` annotations.
4. **No loose statements**: All statements must be contained within blocks (no floating `Store` or `BufferStore` outside blocks).
5. **Predicate-free blocks**: Block predicates must be simplified away (no remaining `where` clauses that the code generator cannot handle).

This verification is typically called at the end of the TIR lowering pipeline, just before handing off to the code generator.

```python
from tvm.tirx import analysis

# After applying lowering transformations
mod = tirx.transform.Simplify()(mod)
mod = tirx.transform.LowerOpaqueBlock()(mod)
mod = tirx.transform.VectorizeLoop()(mod)

# Verify the result is ready for codegen
try:
    for func_name, func in mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            analysis.verify_build_simplified(func)
    print("All functions are simplified and ready for code generation")
except tvm.error.TVMError as e:
    print(f"Simplification verification failed: {e}")
```

---

### 13.2.7 `verify_gpu_code`

Verifies that a PrimFunc satisfies GPU code generation constraints. This includes thread limits, shared memory limits, register limits, and other GPU-specific constraints.

```python
analysis.verify_gpu_code(func, constraints)
```

**Signature:**
```python
def verify_gpu_code(
    func: tir.PrimFunc,
    constraints: Dict[str, int],
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `tir.PrimFunc` | The PrimFunc to verify |
| `constraints` | `Dict[str, int]` | GPU constraints to check against |

**Supported constraint keys:**

| Key | Description | Typical Value |
|-----|-------------|---------------|
| `"max_local_memory_per_block"` | Maximum local memory per thread block (bytes) | Varies by GPU |
| `"max_shared_memory_per_block"` | Maximum shared memory per thread block (bytes) | 49152 (A100) |
| `"max_threads_per_block"` | Maximum threads per thread block | 1024 |
| `"max_thread_x"` | Maximum threads in x dimension | 1024 |
| `"max_thread_y"` | Maximum threads in y dimension | 1024 |
| `"max_thread_z"` | Maximum threads in z dimension | 64 |
| `"max_vector_bytes"` | Maximum vector width in bytes | 16 |

**Details:**

This pass is essential for catching GPU constraint violations before runtime. It checks:
- The product of `threadIdx.x * threadIdx.y * threadIdx.z` does not exceed `max_threads_per_block`
- Shared memory allocations within a thread block do not exceed `max_shared_memory_per_block`
- Local (per-thread) memory allocations do not exceed `max_local_memory_per_block`
- Vector load/store widths do not exceed `max_vector_bytes`

```python
from tvm.tirx import analysis

# Define GPU constraints (NVIDIA A100)
constraints = {
    "max_shared_memory_per_block": 49152,
    "max_threads_per_block": 1024,
    "max_thread_x": 1024,
    "max_thread_y": 1024,
    "max_thread_z": 64,
}

# After scheduling with GPU thread bindings
try:
    analysis.verify_gpu_code(func, constraints)
    print("GPU code verification passed")
except tvm.error.TVMError as e:
    print(f"GPU constraint violation: {e}")
```

**Usage in MetaSchedule:**

MetaSchedule's `VerifyGPUCode` post-processor calls this analysis to reject schedules that violate GPU constraints during the search process.

---

### 13.2.8 `verify_sblock`

Verifies that a PrimFunc uses the `sblock` (structured block) pattern correctly. SBlocks are TVM's structured computation blocks that carry explicit metadata about data access patterns.

```python
analysis.verify_sblock(func)
```

**Signature:**
```python
def verify_sblock(func: tir.PrimFunc) -> None
```

**Details:**

The verification checks:
1. **Block annotations**: Each block has valid `reads`, `writes`, and `axis` annotations.
2. **Axis bindings**: All `T.axis.spatial` and `T.axis.reduce` bindings are within the block's iteration domain.
3. **Access region consistency**: The actual buffer accesses in the block body match the declared `reads` and `writes` annotations.
4. **Init body**: If the block has an `init` clause, the init body only writes to the output buffer (no reads from the output buffer).
5. **No side effects outside blocks**: All side-effecting operations (buffer stores) must be within blocks.

```python
from tvm.tirx import analysis

# After constructing or transforming TIR
mod = tirx.transform.CompleteThreshAndUnroll()(mod)

# Verify block structure integrity
for name, func in mod.functions.items():
    if isinstance(func, tir.PrimFunc):
        analysis.verify_sblock(func)
        print(f"Function '{name}' has valid sblock structure")
```

---

### 13.2.9 `OpaqueBlock`

`OpaqueBlock` is a special analysis marker that represents a block whose internal structure cannot be analyzed. Opaque blocks arise when:
- A block contains function calls whose behavior is not analyzable (e.g., external library calls)
- The block has been lowered to a form where access regions cannot be determined
- The block intentionally hides its implementation details

```python
from tvm.tirx.analysis import OpaqueBlock

# Check if a block is opaque
block = sch.get_block_rv("my_block")
block_stmt = sch.get(block).stmt

if isinstance(block_stmt, OpaqueBlock):
    print("This block is opaque and cannot be further analyzed")
```

**Implications of opaque blocks:**

- Scheduling primitives may be restricted (e.g., `compute_inline` cannot inline an opaque block)
- Fusion decisions must conservatively assume the block accesses all of its declared buffers
- Analysis passes return conservative (over-approximated) results for opaque blocks

---

## 13.3 `s_tir.analysis` Module

The `s_tir.analysis` module provides analysis functions specifically designed to support scheduling decisions. These are used internally by the scheduling API and by MetaSchedule / DLight to detect patterns and make optimization decisions.

### 13.3.1 `estimate_tir_flops` (s_tir variant)

The s_tir variant of `estimate_tir_flops` operates on an IRModule or a schedulable PrimFunc.

```python
from tvm.s_tir import analysis as s_analysis

flops = s_analysis.estimate_tir_flops(func_or_mod)
```

**Signature:**
```python
def estimate_tir_flops(func_or_mod: Union[tir.PrimFunc, ir.IRModule]) -> int
```

This variant can accept an entire IRModule, summing FLOPs across all PrimFuncs, or a single PrimFunc. It is used by MetaSchedule to rank tasks by computational intensity.

---

### 13.3.2 `detect_dominated_var`

Detects whether a specific variable is "dominated" within a loop nest -- meaning that the variable's value is uniquely determined by the surrounding loop iteration variables. This is critical for analyzing whether loop transformations are valid.

```python
from tvm.s_tir import analysis as s_analysis

is_dominated = s_analysis.detect_dominated_var(expr, var, loop_vars)
```

**Signature:**
```python
def detect_dominated_var(
    expr: tir.PrimExpr,
    var: tir.Var,
    loop_vars: List[tir.Var],
) -> bool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `tir.PrimExpr` | The expression to check |
| `var` | `tir.Var` | The variable to check for dominance |
| `loop_vars` | `List[tir.Var]` | The surrounding loop iteration variables |

**Return value:** `True` if the variable is dominated by the loop variables, `False` otherwise.

**Details:**

A variable is "dominated" if every occurrence of the variable in the expression can be expressed as a unique affine function of the loop variables. For example, in the expression `A[i * 4 + j]`, the index `i * 4 + j` is dominated by loop variables `i` and `j` because it is a unique affine mapping.

This analysis is used by:
- **Loop tiling** to verify that tile indices can be computed from loop variables
- **Loop fusion** to check that fused loops maintain valid index mappings
- **Cache read/write** to determine if data can be safely loaded into a different memory scope

```python
import tvm
from tvm.s_tir import analysis as s_analysis
from tvm.script import ir as I, tirx as T

@I.ir_module
class MyModule:
    @T.prim_func
    def example(A: T.Buffer((64, 64), "float32"), B: T.Buffer((64, 64), "float32")):
        for i, j in T.grid(64, 64):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi * 2, vj + 3]

# The index expression "vi * 2" is dominated by the loop variable vi
# The index expression "vj + 3" is dominated by the loop variable vj
```

---

### 13.3.3 `detect_linear_equation`

Detects whether an expression is a linear equation over a set of variables. This analysis decomposes an expression into a linear combination of variables plus a constant offset.

```python
from tvm.s_tir import analysis as s_analysis

result = s_analysis.detect_linear_equation(expr, vars)
```

**Signature:**
```python
def detect_linear_equation(
    expr: tir.PrimExpr,
    vars: List[tir.Var],
) -> Optional[Tuple[List[tir.PrimExpr], tir.PrimExpr]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `tir.PrimExpr` | The expression to decompose |
| `vars` | `List[tir.Var]` | The variables to express as linear combinations |

**Return value:** If the expression is linear in the given variables, returns a tuple `(coefficients, base)` where `coefficients` is a list of coefficients (one per variable) and `base` is the constant offset. Returns `None` if the expression is not linear.

**Details:**

This analysis is fundamental to index analysis and loop transformation. It determines if buffer access indices are affine functions of loop variables, which is a prerequisite for:
- **Loop tiling**: Tiled indices must be affine for the tile loop to be separable
- **Loop permutation**: Reordering loops is only valid when indices remain computable
- **Memory bank conflict avoidance**: Computing strided access patterns

```python
from tvm.s_tir import analysis as s_analysis
from tvm import tir

# Example: expression is 4*i + j + 2
i = tir.Var("i", "int32")
j = tir.Var("j", "int32")
expr = 4 * i + j + 2

result = s_analysis.detect_linear_equation(expr, [i, j])
if result is not None:
    coefficients, base = result
    print(f"Coefficients: {coefficients}")  # [4, 1]
    print(f"Base: {base}")                  # 2
```

---

### 13.3.4 `find_blocked_var`

Finds variables that are "blocked" -- variables that have been partitioned by a tiling transformation into outer tile variables and inner intra-tile variables.

```python
from tvm.s_tir import analysis as s_analysis

blocked_vars = s_analysis.find_blocked_var(block, var)
```

**Signature:**
```python
def find_blocked_var(
    block: tir.Block,
    var: tir.Var,
) -> Optional[Tuple[tir.Var, tir.Var]]
```

**Return value:** If the variable has been blocked, returns `(outer_var, inner_var)` representing the tile index and the intra-tile index. Returns `None` if the variable is not blocked.

**Details:**

This analysis is used after loop tiling to discover the relationship between the original loop variable and the tiled loop variables. For example, after tiling a loop over `i` with factor 32, the original variable `i` is decomposed into `i_outer` (tile index) and `i_inner` (intra-tile index), where `i = i_outer * 32 + i_inner`.

```python
import tvm
from tvm.s_tir import analysis as s_analysis

sch = tvm.s_tir.Schedule(mod)
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Tile the i loop with factor 32
i_outer, i_inner = sch.split(i, factors=[None, 32])

# Now find_blocked_var can detect the blocked variable
# It returns (i_outer, i_inner)
```

---

### 13.3.5 `is_broadcast_epilogue`

Checks whether the epilogue (final computation) of a block is a broadcast operation. A broadcast epilogue means the output of the block is replicated across one or more dimensions.

```python
from tvm.s_tir import analysis as s_analysis

is_bcast = s_analysis.is_broadcast_epilogue(block, loop_vars, var)
```

**Signature:**
```python
def is_broadcast_epilogue(
    block: tir.Block,
    loop_vars: List[tir.Var],
    var: tir.Var,
) -> bool
```

**Details:**

This analysis is used by DLight rules to determine whether a reduction block's output needs special handling. When a block has a broadcast epilogue, the scheduling strategy must account for the fact that multiple threads may write to the same output location, requiring either atomic operations or a separate epilogue kernel.

Common scenarios where broadcast epilogues appear:
- Reduction followed by bias addition (the bias is broadcast across the reduction dimension)
- Softmax computation where the normalization factor is broadcast
- Layer normalization where mean and variance are broadcast

```python
from tvm.s_tir import analysis as s_analysis

# Check if a softmax or layernorm block has a broadcast epilogue
sch = tvm.s_tir.Schedule(mod)
block_rv = sch.get_block("softmax_norm")
block_stmt = sch.get(block_rv).stmt

# Determine if the epilogue is broadcast
is_bcast = s_analysis.is_broadcast_epilogue(block_stmt, [vi, vj], vk)
if is_bcast:
    print("Block has a broadcast epilogue -- requires special scheduling")
```

---

### 13.3.6 `is_gpu_script`

Checks whether a PrimFunc or IRModule is designed for GPU execution. This is determined by examining thread axis bindings and other GPU-specific attributes.

```python
from tvm.s_tir import analysis as s_analysis

is_gpu = s_analysis.is_gpu_script(func)
```

**Signature:**
```python
def is_gpu_script(func: tir.PrimFunc) -> bool
```

**Return value:** `True` if the function has GPU thread bindings, `False` otherwise.

**Details:**

A function is considered a GPU script if any of the following conditions hold:
1. It contains `attr["thread_extent"]` annotations indicating thread axis bindings
2. It contains `launch_bounds` attributes
3. It has `T.thread_binding` annotations

This analysis is used by MetaSchedule to select appropriate schedule rules (GPU rules vs CPU rules) and by DLight to choose between GPU and CPU optimization strategies.

```python
from tvm.s_tir import analysis as s_analysis
from tvm.script import ir as I, tirx as T

@I.ir_module
class GPUModule:
    @T.prim_func
    def gpu_kernel(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32")):
        T.func_attr({"tir.noalias": True})
        for i in T.thread_binding(0, 1024, thread="threadIdx.x"):
            B[i] = A[i] * T.float32(2.0)

is_gpu = s_analysis.is_gpu_script(GPUModule["gpu_kernel"])
print(f"Is GPU script: {is_gpu}")  # True
```

---

### 13.3.7 `TensorizeInfo`

`TensorizeInfo` is an analysis structure that captures information about tensorization opportunities in a PrimFunc. Tensorization replaces a block of computation with a hardware-accelerated intrinsic (e.g., a WMMA instruction on NVIDIA GPUs).

```python
from tvm.s_tir import analysis as s_analysis

info = s_analysis.TensorizeInfo.analyze(func, target)
```

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `block_name` | `str` | Name of the block that can be tensorized |
| `intrinsic_name` | `str` | Name of the matching tensor intrinsic |
| `input_buffers` | `List[tir.Buffer]` | Input buffer shapes and types |
| `output_buffers` | `List[tir.Buffer]` | Output buffer shapes and types |
| `reduce_axis` | `List[tir.IterVar]` | Reduction axis information |

**Details:**

TensorizeInfo performs pattern matching on the PrimFunc's computation structure to find blocks that match known hardware intrinsics. The analysis considers:
- The arithmetic operations in the block (e.g., multiply-add)
- The dimensionality and sizes of the iteration domain
- The data types of input and output buffers
- The target hardware's supported intrinsic shapes

```python
from tvm.s_tir import analysis as s_analysis

# Analyze tensorization opportunities
info = s_analysis.TensorizeInfo.analyze(func, target="nvidia/nvidia-a100")

if info.has_tensorizable_blocks():
    for block_info in info.tensorizable_blocks:
        print(f"Block '{block_info.block_name}' can use intrinsic '{block_info.intrinsic_name}'")
        print(f"  Input shapes: {block_info.input_buffers}")
        print(f"  Output shapes: {block_info.output_buffers}")
```

---

## 13.4 How Analysis Passes Support Optimization Decisions

### 13.4.1 Analysis in the Compilation Pipeline

Analysis passes are invoked at multiple stages of the TVM compilation pipeline:

```
Source Model
    |
    v
Frontend Import
    |
    v
Relax Optimizations --> [struct_info_analysis, shape_analysis]
    |
    v
Legalize to TIR
    |
    v
TIR Optimizations --> [estimate_tir_flops, get_block_access_region]
    |
    v
Scheduling --> [detect_linear_equation, detect_dominated_var, find_blocked_var]
    |
    v
Post-schedule Verification --> [verify_gpu_code, verify_build_simplified]
    |
    v
Code Generation
```

### 13.4.2 Analysis-Driven Scheduling

MetaSchedule and DLight use analysis results to drive scheduling decisions:

1. **FLOPs estimation** determines which functions are worth tuning (high FLOPs = high optimization potential).
2. **Access region analysis** determines which blocks can be safely fused or inlined.
3. **Linear equation detection** validates that loop transformations preserve correct index mappings.
4. **Dominated variable detection** ensures that cache read/write transformations produce valid memory accesses.
5. **GPU verification** rejects schedules that exceed hardware limits.
6. **Broadcast epilogue detection** triggers special handling for reduction patterns.

### 13.4.3 Analysis-Driven Transformation Selection

```python
from tvm.tirx import analysis
from tvm.s_tir import analysis as s_analysis
import tvm

# Example: deciding whether to apply shared memory caching
func = mod["matmul"]
flops = analysis.estimate_tir_flops(func)
workspace = analysis.calculate_workspace(func)

# If the computation is memory-bound (low FLOP/byte ratio), apply caching
if flops / max(workspace, 1) < 10:
    print("Memory-bound kernel -- applying shared memory caching")
    sch = tvm.s_tir.Schedule(mod)
    block = sch.get_block("C")
    sch.cache_read(block, 0, "shared")
    sch.cache_read(block, 1, "shared")
    # ... apply tiling and binding
else:
    print("Compute-bound kernel -- focusing on instruction-level parallelism")
    # ... apply vectorization and unrolling
```

---

## 13.5 Integration with Scheduling and Transformation Passes

### 13.5.1 Pre-transformation Analysis

Before applying a transformation, the system must verify that the transformation is legal:

```python
from tvm.tirx import analysis
from tvm.s_tir import analysis as s_analysis
import tvm

sch = tvm.s_tir.Schedule(mod)

# Before fusing two blocks, check access regions for hazards
block_a = sch.get_block("A")
block_b = sch.get_block("B")
# The scheduler internally uses get_block_access_region to verify safety
```

### 13.5.2 Post-transformation Verification

After applying transformations, analysis passes verify correctness:

```python
import tvm
from tvm.tirx import analysis, transform

# Apply a series of transformations
mod = transform.Simplify()(mod)
mod = transform.LowerOpaqueBlock()(mod)
mod = transform.VectorizeLoop()(mod)

# Verify the result
for name, func in mod.functions.items():
    if isinstance(func, tir.PrimFunc):
        analysis.verify_build_simplified(func)

# If targeting GPU
target = tvm.target.Target("nvidia/nvidia-a100")
constraints = target.attrs.get("gpu_constraints", {})
for name, func in mod.functions.items():
    if isinstance(func, tir.PrimFunc):
        analysis.verify_gpu_code(func, constraints)
```

### 13.5.3 Custom Analysis Integration

Users can write custom analysis passes using the TIR visitor pattern:

```python
import tvm
from tvm import tir

class BufferUsageAnalyzer(tir.StmtVisitor):
    """Custom analysis that counts buffer accesses per statement."""

    def __init__(self):
        super().__init__()
        self.read_count = {}
        self.write_count = {}

    def visit_buffer_load(self, expr):
        buf_name = expr.buffer.name
        self.read_count[buf_name] = self.read_count.get(buf_name, 0) + 1
        super().visit_buffer_load(expr)

    def visit_buffer_store(self, stmt):
        buf_name = stmt.buffer.name
        self.write_count[buf_name] = self.write_count.get(buf_name, 0) + 1
        super().visit_buffer_store(stmt)

# Use the custom analyzer
analyzer = BufferUsageAnalyzer()
analyzer.visit(func.body)
print(f"Read counts: {analyzer.read_count}")
print(f"Write counts: {analyzer.write_count}")
```

---

## 13.6 Summary

| Analysis Pass | Module | Purpose |
|---------------|--------|---------|
| `calculate_workspace` | `tirx.analysis` | Compute scratchpad memory requirements |
| `calculate_constant_bytes` | `tirx.analysis` | Compute constant memory requirements |
| `estimate_tir_flops` | `tirx.analysis` | Estimate floating-point operation count |
| `get_block_access_region` | `tirx.analysis` | Analyze buffer read/write regions of a block |
| `get_var_touch` | `tirx.analysis` | Find variables referenced by a statement |
| `verify_build_simplified` | `tirx.analysis` | Verify function is ready for codegen |
| `verify_gpu_code` | `tirx.analysis` | Verify GPU hardware constraints |
| `verify_sblock` | `tirx.analysis` | Verify structured block correctness |
| `OpaqueBlock` | `tirx.analysis` | Marker for unanalyzable blocks |
| `estimate_tir_flops` | `s_tir.analysis` | FLOPs estimation for scheduling |
| `detect_dominated_var` | `s_tir.analysis` | Check variable dominance by loop vars |
| `detect_linear_equation` | `s_tir.analysis` | Decompose expression as linear equation |
| `find_blocked_var` | `s_tir.analysis` | Find tiled (blocked) variable decomposition |
| `is_broadcast_epilogue` | `s_tir.analysis` | Detect broadcast in block epilogue |
| `is_gpu_script` | `s_tir.analysis` | Check if function targets GPU |
| `TensorizeInfo` | `s_tir.analysis` | Analyze tensorization opportunities |
