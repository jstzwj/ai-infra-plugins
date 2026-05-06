# TileLang Carver System: Configuration Generation and Analysis

The Carver system is TileLang's automated configuration generation and analysis framework. It analyzes TIR (Tensor Intermediate Representation) programs to produce hardware-aware scheduling configurations. Carver bridges the gap between high-level kernel descriptions and low-level hardware-specific optimizations by performing static analysis on computation graphs and generating tuned parameters for various GPU architectures.

## Table of Contents

1. [Carver Overview](#carver-overview)
2. [Analysis Module](#analysis-module)
3. [Common Schedules](#common-schedules)
4. [Roller Module](#roller-module)
5. [Architecture-Specific Implementations](#architecture-specific-implementations)
6. [Templates](#templates)

---

## Carver Overview

The Carver system is located at `tilelang.carver` and provides the following capabilities:

- **Static analysis** of TIR PrimFunc programs to identify computation patterns (GEMM, GEMV, element-wise, reduction, etc.)
- **Automated configuration generation** through heuristic-driven search that considers shared memory capacity, register pressure, memory coalescing, and L2 cache locality
- **Hardware-aware optimization hints** that encode tiling strategies, warp partitioning, rasterization plans, and pipeline configurations
- **Architecture-specific parameter derivation** for NVIDIA CUDA (Volta, Ampere, Ada, Hopper), AMD CDNA, and other targets

The main entry points are:
- `get_roller_hints_from_func()` -- generates hints from a single PrimFunc
- `get_roller_hints_from_output_nodes()` -- generates hints from a graph of OutputNodes (for multi-stage kernels like FlashAttention)

### Architecture of the Carver Pipeline

```
PrimFunc / IRModule
       |
       v
  normalize_prim_func()  -- Normalize to canonical form
       |
       v
  BlockInfo / IterInfo   -- Extract block and iteration structure
       |
       v
  detect_dominant_read() -- Identify memory access patterns
       |
       v
  Policy (DefaultPolicy / TensorCorePolicy)
       |
       v
  Hint objects           -- Generated scheduling configurations
       |
       v
  Applied by code generator
```

---

## Analysis Module

The analysis module (`tilelang.carver.analysis`) provides foundational tools for inspecting TIR blocks, loops, and functions.

### IterInfo: Iteration Variable Information

`IterInfo` encapsulates information about a single loop or iteration variable within a TIR block.

```python
class IterInfo:
    kind: Literal["S", "R", "O"]  # Iteration type
    var: tir.Var                    # The TIR variable
    _dom: tir.PrimExpr             # Domain extent
    loop_rv: tir.schedule.LoopRV   # Reference to the loop in schedule
```

**Iteration Types:**

| Kind | Label | Description |
|------|-------|-------------|
| `"S"` | Spatial | Data-parallel iteration (e.g., output dimensions in matmul) |
| `"R"` | Reduction | Reduction dimension (e.g., K in matmul) |
| `"O"` | Other | Opaque or mixed iteration types |

**Construction:**

```python
from tilelang.carver.analysis import IterInfo
from tvm import tir

iter_info = IterInfo(
    kind="S",
    var=tir.Var("i", "int32"),
    dom=128,
    loop_rv=my_loop_rv,
)
```

**Properties:**

- `dom` -- Returns the iteration domain as an integer (if `IntImm`) or `PrimExpr`. This represents how many times this axis iterates.
- `__str__()` -- Returns a human-readable representation like `Iter("S", 128)`.

### BlockInfo: Block Analysis Information

`BlockInfo` provides comprehensive information about a TIR block, including its iteration structure and classification.

```python
class BlockInfo:
    name: str                          # Block name hint
    iters: list[IterInfo]              # All iteration variables
    block_rv: tir.schedule.BlockRV     # Reference to the block
    _reduction_block: bool             # Whether this is a reduction block
```

**Key Methods:**

- `dom()` -- Returns a list of domain extents for all iteration variables: `[i.dom for i in self.iters]`
- `dom_kind()` -- Returns a string encoding of iteration types, e.g., `"SSSS"`, `"SSSR"`, `"SSR"`
- `is_injective()` -- Returns `True` if all iterations are spatial (all `"S"`)
- `is_elementwise(sch)` -- Returns `True` if the block is element-wise (single read, single write, trivial index mapping)
- `is_reduction()` -- Returns `True` if the block contains at least one reduction axis
- `is_gemv()` -- Returns whether the block represents a GEMV workload (not yet implemented)
- `is_gemm()` -- Returns whether the block represents a GEMM workload (not yet implemented)

**Example:**

```python
# For a matmul block C[i, j] += A[i, k] * B[k, j]:
block_info = BlockInfo(
    name="T_dense",
    iters=[
        IterInfo("S", i_var, 128, loop_i),
        IterInfo("S", j_var, 64, loop_j),
        IterInfo("R", k_var, 32, loop_k),
    ],
    block_rv=block_rv,
    reduction_block=True,
)
block_info.dom_kind()   # "SSR"
block_info.is_injective()  # False
block_info.is_reduction()  # True
```

### collect_block_iter_vars_used_in_access_region

```python
def collect_block_iter_vars_used_in_access_region(
    block: tir.Block,
    region: list[ir.Range],
) -> set[tir.Var]
```

Collects the block iteration variables that appear in a buffer access region. This is used to determine which loop variables are actually used when accessing a particular buffer, which helps identify memory access patterns.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block` | `tir.Block` | The TIR block to analyze |
| `region` | `list[ir.Range]` | The access region (list of ranges) |

**Returns:** `set[tir.Var]` -- A set of iteration variables used in the region.

**Algorithm:** For each range in the region, if `extent == 1` (unit dimension), collect all `tir.Var` instances from the range's `min` expression. Then intersect with the block's iteration variables.

**Usage Example:**

```python
from tilelang.carver.analysis import collect_block_iter_vars_used_in_access_region

# For a block that reads A[i, k] where i is spatial and k is reduction:
used_vars = collect_block_iter_vars_used_in_access_region(block, block.reads[0].region)
# used_vars contains {i_var, k_var}
```

### collect_vars_used_in_prim_expr

```python
def collect_vars_used_in_prim_expr(expr: tir.PrimExpr) -> set[tir.Var]
```

Collects all `tir.Var` instances used within a `PrimExpr` by performing a post-order traversal.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `tir.PrimExpr` | The expression to analyze |

**Returns:** `set[tir.Var]` -- All variables referenced in the expression.

This is a utility function used by other analysis routines, such as `collect_block_iter_vars_used_in_access_region`.

### detect_dominant_read: Identify Dominant Memory Access Patterns

```python
def detect_dominant_read(block: tir.Block) -> tir.PrimExpr
```

Identifies the dominant read pattern in a TIR block. The dominant read is the buffer access that uses the most iteration variables, which typically corresponds to the primary input tensor in the computation.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block` | `tir.Block` | The TIR block to analyze |

**Returns:** `tir.PrimExpr` -- The buffer offset expression for the dominant read.

**Algorithm:**
1. Iterate through all read buffer regions of the block
2. For each region, count how many block iteration variables are used in the access indices
3. Select the buffer region that uses the most iteration variables (the "dominant" read)
4. Compute and return the buffer offset expression for that region

This function is critical for determining which input tensor drives the memory access pattern, which in turn influences tiling and vectorization decisions.

### is_broadcast_epilogue: Detect Broadcast Patterns

```python
def is_broadcast_epilogue(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
    epilogue: tir.schedule.BlockRV,
) -> bool
```

Determines whether an epilogue block exhibits a broadcast pattern relative to a main computation block. This occurs when the epilogue reads from the main block's output but uses fewer iteration variables than its total non-unit dimensions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sch` | `tir.Schedule` | The TIR schedule |
| `block` | `BlockRV` | The main computation block |
| `epilogue` | `BlockRV` | The epilogue block to check |

**Returns:** `bool` -- `True` if the epilogue is a broadcast pattern.

**When this matters:** Broadcast epilogues arise in operations like scalar bias addition, layer normalization, or fused scale operations where the epilogue has fewer active dimensions than the main block. Recognizing this pattern allows the scheduler to avoid unnecessary tiling on the broadcast dimensions.

### normalize_prim_func: Normalize PrimFunc for Analysis

```python
def normalize_prim_func(sch: tir.Schedule) -> list[BlockInfo] | None
```

Normalizes a TIR PrimFunc into a canonical form suitable for analysis and returns structured `BlockInfo` objects for each computation block.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sch` | `tir.Schedule` | The TIR schedule to normalize |

**Returns:** `list[BlockInfo] | None` -- A list of `BlockInfo` objects for each block, or `None` if normalization fails.

**Process:**
1. Calls the TVM-registered function `tir.schedule.NormalizePrimFunc`
2. Extracts block names, loop structures, iteration variables, and reduction flags
3. Maps `IterVar.DataPar` to `"S"` and `IterVar.CommReduce` to `"R"` for each iteration variable
4. Constructs `BlockInfo` objects with full iteration information

**Error Handling:** Returns `None` if the normalization process raises any exception, allowing callers to fall back to alternative analysis strategies.

### Utility Functions

#### find_var_from_func

```python
def find_var_from_func(func, var: str) -> tir.Var | None
```

Searches for a named variable in the buffer shapes of a PrimFunc's buffer map. Used primarily to locate symbolic dimension variables for specialization.

#### check_func_with_dynamic

```python
def check_func_with_dynamic(func) -> bool
```

Returns `True` if any buffer in the function has a symbolic (dynamic) dimension, indicating the function uses dynamic shapes.

#### get_max_threads_per_block

```python
def get_max_threads_per_block(target: Target) -> int
```

Returns the maximum number of threads per block for a GPU target. Falls back to 64 if the attribute cannot be found.

#### get_max_shared_memory_per_block

```python
def get_max_shared_memory_per_block(target: Target) -> int
```

Returns the maximum shared memory per block for a GPU target. Raises `ValueError` if the attribute is unavailable.

#### get_root_block

```python
def get_root_block(sch: Schedule, func_name: str = "main") -> BlockRV
```

Returns the root block of a TIR schedule, which serves as the entry point for block traversal.

#### get_reduction_blocks

```python
def get_reduction_blocks(sch: tir.Schedule, blocks: list[BlockRV]) -> list[BlockRV] | None
```

Filters and returns only the reduction blocks from a list of blocks. Returns `None` if the block types are inconsistent (not all spatial or reduction).

#### get_coalesced_veclen

```python
def get_coalesced_veclen(block_stmt: tir.Block, target_bits: int = 128) -> int
```

Calculates the vector length for coalesced memory access. GPU memory prefers 128-bit (16-byte) coalesced transactions. Returns `target_bits // max_dtype_bits`.

---

## Common Schedules

The common schedules module (`tilelang.carver.common_schedules`) provides reusable scheduling strategies for TIR programs.

### get_block

```python
def get_block(
    sch: tir.Schedule,
    blocks: list[BlockInfo],
    name: str,
) -> BlockRV | None
```

Finds a block by name within a list of `BlockInfo` objects. Returns the `BlockRV` if found, or `None` otherwise.

**Usage:**

```python
from tilelang.carver.common_schedules import get_block

target_block = get_block(sch, blocks, "T_dense")
if target_block is not None:
    # Apply scheduling to the target block
    sch.reorder(target_block, ...)
```

### get_output_blocks

```python
def get_output_blocks(
    sch: tir.Schedule,
    blocks: list[BlockInfo],
) -> list[BlockRV]
```

Identifies output blocks by checking which blocks write to buffers that appear in the function's argument buffer map. These are the blocks whose results are exposed to the caller.

**Algorithm:**
1. Retrieve the function from the schedule's IRModule
2. Collect all argument buffers from `func.buffer_map`
3. For each block, check if any write buffer is an argument buffer
4. Return all blocks that write to function arguments

### try_inline

```python
def try_inline(
    sch: tir.Schedule,
    blocks: list[BlockInfo],
) -> list[BlockInfo]
```

Attempts to inline as many blocks as possible using both `compute_inline` and `reverse_compute_inline`, returning the blocks that could not be inlined.

**Inlining Strategy:**
1. Repeatedly attempt `sch.compute_inline(block)` for each block
2. If that fails, try `sch.reverse_compute_inline(block)` for each block
3. Remove successfully inlined blocks from the list
4. Repeat until no more blocks can be inlined

**When to use:** Use this to simplify the scheduling problem by eliminating trivial intermediate computations that can be fused into their producers or consumers.

### try_inline_contiguous_spatial

```python
def try_inline_contiguous_spatial(
    sch: tir.Schedule,
    block_infos: list[BlockInfo],
) -> list[BlockInfo]
```

A more structured version of `try_inline` that only attempts to inline contiguous sequences of spatial (injective) blocks. Blocks with reduction axes are preserved and not considered for inlining.

**Algorithm:**
1. Iterate through blocks, accumulating spatial blocks
2. When a non-spatial block is encountered, attempt to inline the accumulated spatial blocks
3. Preserve non-spatial blocks (reduction blocks) in the result

---

## Roller Module

The Roller module implements the core configuration search algorithm, inspired by the Roller system for tensor program compilation. It generates hardware-aware scheduling hints by exploring the tiling space with cost-model-guided search.

### Rasterization Strategies

Rasterization strategies control how thread blocks are mapped to the GPU grid to improve L2 cache locality. Located in `tilelang.carver.roller.rasterization`.

#### NoRasterization

```python
class NoRasterization(Rasterization):
    """No rasterization -- default linear block ordering."""
```

The default strategy where blocks execute in standard grid order. This is suitable when:
- The computation is small enough to fit in L2 cache
- Only a single PrimFuncNode exists
- The architecture does not benefit from 2D rasterization

#### Rasterization2DRow

```python
class Rasterization2DRow(Rasterization):
    def __init__(self, panel_width=4) -> None
```

Row-major rasterization that groups blocks into horizontal panels of width `panel_width`. The block indices are remapped so consecutive blocks access adjacent memory regions along rows:

```
 ___________
 ___________|
|___________
____________|
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `panel_width` | `int` | `4` | Width of each rasterization panel |

#### Rasterization2DColumn

```python
class Rasterization2DColumn(Rasterization):
    def __init__(self, panel_width=4) -> None
```

Column-major rasterization that groups blocks into vertical panels. This is the preferred strategy for GEMM workloads on Ampere+ GPUs because it improves L2 cache hit rates for both A and B matrix tiles.

```
         _
      | | | |
      | | | |
      |_| |_|
```

**Device Function:** The `Rasterization2DColumn` strategy injects a CUDA device function that remaps `blockIdx` coordinates:

```cuda
__device__ __inline__ dim3 rasterization2DColumn(const int panel_width) {
    const auto baseBlockIdx = blockIdx.x + gridDim.x * blockIdx.y;
    const auto totalPanel = (gridDim.x * gridDim.y +
        panel_width * gridDim.x - 1) / (panel_width * gridDim.x);
    const auto totalBlock = gridDim.x * gridDim.y;
    const auto panelIdx = baseBlockIdx / (panel_width * gridDim.x);
    const auto strideLd = panelIdx + 1 < totalPanel ?
        panel_width : (totalBlock - panelIdx *
        (panel_width * gridDim.x)) / gridDim.x;
    const auto bx = (panelIdx & 1) ? gridDim.x -
        (baseBlockIdx - panelIdx * panel_width * gridDim.x) /
        strideLd - 1 : (baseBlockIdx - panelIdx *
        panel_width * gridDim.x) / strideLd;
    const auto by = (baseBlockIdx - panelIdx * panel_width *
        gridDim.x) % strideLd + panelIdx * panel_width;
    const auto bz = blockIdx.z;
    dim3 blockIdx(bx, by, bz);
    return blockIdx;
}
```

### Hint: Optimization Hints

The `Hint` class is the central configuration object that encodes all scheduling decisions for a single PrimFunc.

```python
class Hint:
    arch: TileDevice | None
    use_tc: bool | None              # Whether to use Tensor Core
    block: list[int]                  # Block-level tiling
    thread: list[int]                 # Thread-level tiling (for CUDA Core)
    warp: list[int]                   # Warp-level tiling (for Tensor Core)
    rstep: list[int]                  # Reduction axis step sizes
    reduce_thread: list[int]          # Thread-level reduction tiling
    rasterization_plan: Rasterization # L2 cache rasterization strategy
    cached_tensors: list[str]         # Tensors to cache in shared memory
    output_strides: dict              # Stride configurations for outputs
    schedule_stages: list | None      # Schedule stage blocks
    block_reduction_depth: int | None # Block reduction depth
    split_k_factor: int               # Split-K factor
    vectorize: dict[str, int]         # Vectorization size per tensor
    pipeline_stage: int               # Software pipeline depth
    use_async: bool                   # Use async copy (cp.async)
    intrin_info: IntrinInfo           # Tensor Core intrinsic info
    shared_scope: str                 # Shared memory scope ("shared" or "shared.dyn")
    pass_context: dict                # Additional pass configurations
    opt_shapes: dict[str, int]        # Optimized shape bindings
```

**Key Methods:**

- `to_dict()` -- Serializes the hint to a dictionary for logging/debugging
- `from_dict(dic)` -- Class method to deserialize from a dictionary
- `tensorcore_legalization()` -- Normalizes Tensor Core configurations to use only the last 2 axes
- `complete_config(node)` -- Fills in derived configurations like pass context
- `raxis_order` -- Returns the reduction axis ordering (defaults to sequential)
- `step` -- Returns the step sizes for each spatial axis

#### IntrinInfo

```python
class IntrinInfo:
    in_dtype: str        # Input data type (e.g., "float16", "int8")
    out_dtype: str       # Output/accumulator data type
    trans_a: bool        # Whether A is transposed (always False in TileLang)
    trans_b: bool        # Whether B is transposed
    input_transform_kind: int   # Input transform level (0=none, 1=inter, 2=smooth)
    weight_transform_kind: int  # Weight transform level
```

**Properties:**
- `is_input_8bit()` -- Returns `True` if input dtype is 8-bit
- `smooth_a` / `smooth_b` -- Returns `True` if smooth quantization is enabled (kind >= 2)
- `inter_transform_a` / `inter_transform_b` -- Returns `True` if intermediate transforms are enabled (kind >= 1)

#### Stride

```python
class Stride:
    def __init__(self, stride: int = 1, ax: int = -1) -> None
```

Manages stride (padding) information for shared memory buffers. Padding is used to avoid bank conflicts in shared memory when accessing Tensor Core layouts.

- `ax` -- The axis on which to apply the stride padding
- `stride` -- The stride size in elements
- `compute_strides_from_shape(shape)` -- Computes the full stride array
- `compute_elements_from_shape(shape)` -- Computes total elements including padding
- `is_valid()` -- Returns `True` if a valid axis is specified

#### TileDict

```python
class TileDict:
    output_tile: list[int]          # The output tile configuration
    tile_map: dict                  # Map from node to tile sizes
    rstep_map: dict                 # Map from node to reduction step sizes
    cached_tensors_map: dict        # Map from node to cached tensor names
    output_strides_map: dict        # Map from node to output strides
    tensor_strides_map: dict        # Map from node to tensor strides
    traffic: int                    # Estimated memory traffic
    smem_cost: int                  # Shared memory cost in bytes
    block_per_SM: int               # Estimated blocks per SM
    num_wave: int                   # Number of waves
    grid_size: int                  # Total grid size
    valid: bool                     # Whether this configuration is valid
```

### Memory Policies

#### DefaultPolicy

```python
class DefaultPolicy:
    def __init__(self, arch: TileDevice, tags: dict | None = None)
```

The default policy for CUDA Core schedules. It uses a heuristic-driven search that minimizes memory traffic and maximizes parallelism.

**Construction Methods:**

| Method | Description |
|--------|-------------|
| `from_prim_func(func, arch, tags)` | Create from a single PrimFunc |
| `from_output_nodes(nodes, arch, tags)` | Create from a list of OutputNodes |

**Configuration Generation:**

```python
policy = DefaultPolicy.from_prim_func(func, arch=arch, tags=tags)
hints = policy.emit_config(topk=10)
```

**Key Methods:**

- `emit_config(topk)` -- Generates up to `topk` scheduling configurations
- `get_base_tile()` -- Computes the minimum tile that eliminates redundant computation
- `dfs_smem_tile(init_tile, rstep_map)` -- Depth-first search over shared memory tile sizes, prioritized by `(traffic + 1) * num_wave`
- `compute_tile_dict(output_tile, rstep_map)` -- Evaluates a specific tile configuration
- `assign_block_size(td)` -- Assigns thread block sizes to a tile dictionary
- `recommend_block_size(td)` -- Recommends block sizes based on warp efficiency scoring
- `plan_rasterization(td)` -- Plans the L2 cache rasterization strategy

**Search Strategy:**

1. Compute the base tile (minimum non-redundant tile)
2. Assign reduction step sizes by optimizing for memory coalescing
3. DFS over shared memory tile configurations, prioritized by `(traffic + 1) * num_wave`
4. For each valid tile, assign block sizes by factorizing and greedily distributing threads across spatial and reduction axes
5. Score block sizes based on warp occupancy (`score_block_size`)

#### TensorCorePolicy

```python
class TensorCorePolicy(DefaultPolicy):
    wmma_k: int = 16
    pipeline_stage: int = 1
    use_async_copy: bool = False
    block_reduction_depth: int | None = None
```

Extends `DefaultPolicy` with Tensor Core-specific optimizations. It enforces that tile sizes are multiples of WMMA/MMA shapes and applies software pipelining when appropriate.

**Tensor Core Constraints:**

| Constraint | Description |
|-----------|-------------|
| WMMA/MMA shapes | Block tiles must be >= 16x16 for spatial axes |
| `wmma_k` alignment | Reduction steps must be multiples of 16 |
| Pipeline stages | Set to 2 on Ampere+ (SM80+), 1 otherwise |
| Async copy | Enabled on Ampere+ by default |

**Architecture-Specific Defaults:**

| Architecture | Pipeline | Async Copy |
|-------------|----------|------------|
| Volta (SM70) | 1 | False |
| Ampere (SM80) | 2 | True |
| Ada (SM89) | 2 | True |
| Hopper (SM90) | 2 | True |

**Key Overrides:**

- `_assign_block_size()` -- Uses warp-level tiling instead of thread-level tiling; factors block size into warps and distributes across spatial dimensions
- `_assign_reduce_step()` -- Ensures reduction steps are multiples of `wmma_k` (16 or 32)
- `get_node_reduce_step_candidates()` -- Generates candidates that are multiples of `wmma_k`
- `check_tile_shape_isvalid()` -- Validates that spatial tiles satisfy WMMA/MMA minimum size requirements
- `infer_node_smem_usage()` -- Multiplies shared memory cost by `pipeline_stage` to account for multi-buffering
- `plan_rasterization()` -- Applies `Rasterization2DColumn` when the total global memory footprint fits within L2 cache

### PrimFuncNode

```python
class PrimFuncNode(Node):
    def __init__(self, prim_func: PrimFunc, tags: dict | None = None, name: str = "PrimFuncNode")
```

Wraps a TIR PrimFunc with analysis capabilities for the Roller configuration search.

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `prim_func` | `PrimFunc` | The wrapped TIR function |
| `sch` | `tir.Schedule` | Schedule derived from the PrimFunc |
| `block_analyzer` | `BlockAnalyzer` | Block-level analysis helper |
| `schedule_stages` | `list[BlockRV]` | Blocks that need scheduling |
| `output_blocks` | `list[BlockRV]` | Blocks that write to function outputs |
| `reduction_block` | `BlockRV | None` | The primary reduction block |
| `raxis` | `list` | Reduction axis information |
| `input_buffers` | `list[Buffer]` | Function input buffers |
| `output_buffers` | `list[Buffer]` | Function output buffers |
| `ana` | `InputShapeInference` | Shape inference analyzer |

**Key Methods:**

- `get_space_dim()` -- Returns the spatial dimension sizes (cached)
- `propagate(tile, rstep, targets)` -- Propagates tile sizes through the computation graph
- `propagate_inputs(tile, rstep)` -- Propagates tile sizes to input shapes
- `propagate_outputs(tile, rstep)` -- Propagates tile sizes to output shapes
- `propagate_reduction_inputs(shape, rstep)` -- Propagates shapes specifically for the reduction block inputs
- `footprint(shape, rstep, stride_map)` -- Computes shared memory footprint for a given tile configuration
- `infer_tensorcore_axis()` -- Infers the Tensor Core axis mapping (cached)
- `get_opt_shape(name)` -- Returns the specialized shape value for a named symbolic dimension

### OutputNode

```python
class OutputNode(Node):
    def __init__(self, node, id=0)
```

Represents an output in the computation graph. Each `OutputNode` connects to a `PrimFuncNode` and exposes its output shape and dtype.

### Edge

```python
@dataclass
class Edge:
    src_node: Node    # Source node
    dst_node: Node    # Destination node
    src_id: int       # Output index in source node
    dst_id: int       # Input index in destination node
```

Represents a data dependency between nodes in the computation graph.

### get_analyzer_by_tir: Get TIR Analyzer

```python
def get_analyzer_by_tir(block_analyzer, args) -> InputShapeInference
```

Creates an `InputShapeInference` analyzer from a TIR block analyzer and a list of blocks. This analyzer propagates shape constraints through the computation graph to determine input tensor sizes for any given output tile configuration.

The `InputShapeInference` class:
1. Extracts dependent regions from each block (which input indices are functions of which loop variables)
2. Constructs a dependency graph between tensors
3. Uses `arith.detect_iter_map` and `arith.inverse_affine_iter_map` to compute reverse mappings
4. For a given output tile and reduction step, computes the required input tensor sizes using `arith.Analyzer` with `ConstIntBound` propagation

---

## Architecture-Specific Implementations

### TileDevice Base Class

```python
class TileDevice:
    reg_cap: int           # Register capacity per SM
    smem_cap: int          # Shared memory capacity per block
    compute_max_core: int  # Number of SMs
    warp_size: int         # Threads per warp
    sm_partition: int      # SM partitions for occupancy calculation
    transaction_size: list[int]  # [write, read] transaction sizes in bytes
    max_smem_usage: int    # Maximum shared memory usage (typically 2x smem_cap)
    bandwidth: list[int]   # [write, read] bandwidth in MB/s
    platform: str          # Platform name
    compute_capability: str  # Compute capability string
    l2_cache_size_bytes: int  # L2 cache size in bytes
```

### CUDA Architecture

```python
class CUDA(TileDevice):
    def __init__(self, target: Target | str)
```

NVIDIA CUDA GPU architecture configuration.

**Auto-detected Properties:**

| Property | Source | Typical Value |
|----------|--------|--------------|
| `smem_cap` | `cuda_driver.get_shared_memory_per_block()` | 49152 (48KB) on A100 |
| `compute_max_core` | `device.multi_processor_count` | 108 on A100 |
| `warp_size` | `device.warp_size` | 32 |
| `compute_capability` | `device.compute_version` | "80", "90", etc. |
| `reg_cap` | Fixed | 65536 |
| `max_smem_usage` | `2 * smem_cap` | 98304 |
| `sm_partition` | Fixed | 4 |
| `transaction_size` | Fixed | `[32, 128]` |
| `bandwidth` | Fixed | `[750, 12080]` |

**Architecture Detection Functions:**

| Function | Condition |
|----------|-----------|
| `is_volta_arch(arch)` | SM version >= 70 and < 80 |
| `is_ampere_arch(arch)` | SM version >= 80 and < 89 |
| `is_ada_arch(arch)` | SM version == 89 |
| `is_hopper_arch(arch)` | SM version == 90 |
| `has_mma_support(arch)` | SM version >= 80 |

**Tensor Core Supported Precisions:**

| Architecture | Supported (input, accumulator) Pairs |
|-------------|--------------------------------------|
| Volta (SM70) | `(fp16, fp32)`, `(fp16, fp16)` |
| Ampere (SM80) | `(bf16, fp32)`, `(fp16, fp32)`, `(fp16, fp16)`, `(int8, int32)`, `(int4, int32)`, `(int2, int32)`, `(int1, int32)` |
| Ada (SM89) | `(bf16, fp32)`, `(fp16, fp32)`, `(fp16, fp16)`, `(int8, int32)`, `(fp8_e5m2, fp32)`, `(fp8_e4m3, fp32)` |
| Hopper (SM90) | Same as Ada |

**Tensor Intrinsic Shapes:**

```python
def get_avaliable_tensorintrin_shapes(self) -> list[list[int]]
# Returns [[16, 16], [16, 16]] for MMA and WMMA respectively
```

### CDNA Architecture

```python
class CDNA(TileDevice):
    def __init__(self, target: Target | str)
```

AMD CDNA GPU architecture configuration for ROCm/HIP targets.

**Auto-detected Properties:**

| Property | Source | Typical Value |
|----------|--------|--------------|
| `smem_cap` | `device.max_shared_memory_per_block` | 65536 (64KB), 163840 (160KB) for gfx950 |
| `compute_max_core` | `device.multi_processor_count` | Varies |
| `warp_size` | `device.warp_size` | 64 |
| `compute_capability` | `device.compute_version` | e.g., "942", "950" |
| `reg_cap` | Fixed | 32768 |
| `max_smem_usage` | `2 * smem_cap` | Varies |
| `transaction_size` | Fixed | `[32, 128]` |
| `bandwidth` | Fixed | `[1300, 14000]` |

**gfx950 Special Handling:** The CDNA class overrides the reported shared memory for gfx950 (CDNA4/MI350) devices to ensure the full 160KB LDS is available, as older drivers may report a conservative 64KB default.

---

## Templates

Templates are pre-built computation patterns that integrate with the Carver system to generate hardware-aware configurations. Each template defines a TVM tensor expression computation and provides a method to retrieve optimized hints.

### BaseTemplate

```python
@dataclass
class BaseTemplate(ABC):
    _arch: TileDevice = field(default=auto_infer_current_arch())
    _func: PrimFunc = field(default=None)
    _output_nodes: list[OutputNode] = field(default=None)
```

The abstract base class for all templates. It provides:

**Abstract Methods:**

| Method | Description |
|--------|-------------|
| `get_hardware_aware_configs(arch, topk)` | Must return a list of `Hint` objects |
| `initialize_function()` | Must set `self._func` to a valid PrimFunc |

**Concrete Methods:**

| Method | Description |
|--------|-------------|
| `with_arch(arch)` | Set the target architecture |
| `has_arch()` | Check if architecture is set |
| `is_volta_arch()` | Check for Volta |
| `is_ampere_arch()` | Check for Ampere |
| `is_cdna_arch()` | Check for CDNA |
| `equivalent_function()` | Get the stored PrimFunc |
| `set_function(func)` | Set the PrimFunc |
| `set_output_nodes(nodes)` | Set output nodes (for multi-stage kernels) |
| `recommend_hints(topk)` | Get top-k hardware-aware hints |

**Lifecycle:** `__post_init__` automatically calls `initialize_function()` to build the computation.

### MatmulTemplate: GEMM Configuration Generation

```python
@dataclass
class MatmulTemplate(BaseTemplate):
    M: int = None
    N: int = None
    K: int = None
    trans_A: bool = False
    trans_B: bool = True
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False
```

Generates optimized GEMM configurations for matrix multiplication.

**Computation Definition:**
- Creates placeholders A, B, and optional Bias
- Computes `C[i, j] = sum(A[i, k] * B[k, j], axis=k)` with transpose support
- Optionally adds bias: `C[i, j] = C[i, j] + Bias[j]`
- Optionally casts output: `D[i, j] = C[i, j].astype(out_dtype)`

**Usage Example:**

```python
from tilelang.carver import MatmulTemplate, CUDA

template = MatmulTemplate(
    M=1024,
    N=1024,
    K=1024,
    trans_B=True,
    in_dtype="float16",
    accum_dtype="float32",
    out_dtype="float16",
)

# Get hardware-aware configurations for CUDA
arch = CUDA("nvidia/nvidia-a100")
hints = template.get_hardware_aware_configs(arch=arch, topk=10)

# Each hint contains block, warp, rstep, pipeline_stage, etc.
for hint in hints:
    print(f"Block: {hint.block}, Warp: {hint.warp}, RStep: {hint.rstep}")
    print(f"Pipeline: {hint.pipeline_stage}, Use TC: {hint.use_tc}")
```

**Shape Handling:**

| trans_A | A Shape | trans_B | B Shape |
|---------|---------|---------|---------|
| False | (M, K) | False | (K, N) |
| False | (M, K) | True | (N, K) |
| True | (K, M) | False | (K, N) |
| True | (K, M) | True | (N, K) |

### GEMVTemplate: Matrix-Vector Template

```python
@dataclass
class GEMVTemplate(BaseTemplate):
    N: int = None
    K: int = None
    trans_B: bool = True
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False
```

Specialized template for matrix-vector multiplication where M=1. This is a common pattern in inference workloads (e.g., single-batch matrix-vector multiply in attention decoding).

**Computation:** Fixed M=1, producing shapes:
- A: `(1, K)`
- B: `(N, K)` if trans_B, else `(K, N)`
- C: `(1, N)`

### ElementwiseTemplate: Element-wise Operations

```python
@dataclass
class ElementwiseTemplate(BaseTemplate):
    shape: list[int] = None
    dtype: str = "float16"
```

Template for element-wise operations. The default computation is `B = A + 1` as a representative element-wise pattern. This template is used to generate configurations for point-wise operations like activations, element-wise math, and broadcasting patterns.

**Usage:**

```python
from tilelang.carver import ElementwiseTemplate

template = ElementwiseTemplate(shape=[1024, 1024], dtype="float32")
hints = template.get_hardware_aware_configs(arch=arch, topk=10)
```

### GeneralReductionTemplate: Reduction Operations

```python
@dataclass
class GeneralReductionTemplate(BaseTemplate):
    structure: str | list[str] = None
    shape: list[int] = None
    dtype: str = "float16"
```

Template for general reduction operations defined by a structure string that encodes axis types.

**Structure Encoding:**

| Character | Meaning |
|-----------|---------|
| `S` | Spatial (data-parallel) axis |
| `R` | Reduction axis |

**Example Structures:**

| Structure | Shape | Description |
|-----------|-------|-------------|
| `"SSR"` | `[128, 64, 32]` | 2D spatial with reduction on last axis |
| `"SR"` | `[1024, 512]` | 1D spatial with reduction |
| `"SSSR"` | `[32, 64, 128, 16]` | 3D spatial with reduction |
| `"SSRR"` | `[128, 64, 32, 16]` | 2D spatial with 2 reduction axes |

**Usage:**

```python
from tilelang.carver import GeneralReductionTemplate

# Row-wise reduction: reduce a (128, 64) tensor to (128,)
template = GeneralReductionTemplate(
    structure="SR",
    shape=[128, 64],
    dtype="float32",
)
hints = template.get_hardware_aware_configs(arch=arch, topk=10)
```

### FlashAttentionTemplate: Attention Configuration

```python
@dataclass
class FlashAttentionTemplate(BaseTemplate):
    batch_size: int = 1
    num_heads: int = 1
    head_dim: int = 1
    seq_length: int = 1
    seq_kv_length: int = 1
    is_causal: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
```

Template for FlashAttention that models the two-stage attention computation as two connected matmul nodes.

**Architecture:**

```
MMA0: Q @ K^T = S    (batch*heads, seq_q, seq_kv, head_dim)
  |
  v  (connected via Edge)
MMA1: S @ V = O      (batch*heads, seq_q, head_dim, seq_kv)
```

**Multi-Node Scheduling:** Unlike single-node templates, FlashAttention uses `OutputNode` graph scheduling:

1. Creates `PrimFuncNode` for each MMA stage
2. Connects them via `Edge` (MMA0 output -> MMA1 input)
3. Uses `get_roller_hints_from_output_nodes()` for multi-node configuration search

### ConvTemplate: Convolution Configuration

```python
@dataclass
class ConvTemplate(BaseTemplate):
    N: int  # Batch size
    C: int  # Input channels
    H: int  # Input height
    W: int  # Input width
    F: int  # Number of filters
    K: int  # Kernel size
    S: int  # Stride
    D: int  # Dilation
    P: int  # Padding
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"
    with_bias: bool = False
```

Template for 2D convolution operations in NHWC format.

**Output Dimensions:**

```python
OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
```

**Tensor Layout:**

| Tensor | Shape | Format |
|--------|-------|--------|
| Input A | `(N, H, W, C)` | NHWC |
| Weight B | `(KH, KW, C, F)` | HWCF |
| Output C | `(N, OH, OW, F)` | NHWC |
| Bias | `(F,)` | 1D |

**Computation:** The convolution is defined with implicit zero-padding using `te.if_then_else` for boundary handling:

```python
def _compute_conv(n, h, w, f):
    h_in = h * S - P + kh * D
    w_in = w * S - P + kw * D
    return te.sum(
        te.if_then_else(
            te.all(h_in >= 0, h_in < H, w_in >= 0, w_in < W),
            A[n, h_in, w_in, c] * B[kh, kw, c, f],
            0,
        ),
        axis=[kh, kw, c],
    )
```

---

## Integration with tilelang.jit and tilelang.compile

The Carver system integrates with TileLang's JIT and compilation infrastructure through the `utils` module:

### get_roller_hints_from_func

```python
def get_roller_hints_from_func(
    func_or_module: tir.PrimFunc | IRModule,
    arch: TileDevice,
    topk: int = 10,
    tensorcore_only: bool = False,
    allow_gemv: bool = False,
) -> list[Hint] | None
```

Entry point for generating hints from a single function. If `tensorcore_only=True`, it first attempts Tensor Core policy and falls back to `None` if the function cannot be tensorized. Otherwise, it tries both Tensor Core and Default policies, preferring Tensor Core when available.

### get_roller_hints_from_output_nodes

```python
def get_roller_hints_from_output_nodes(
    output_nodes: list[OutputNode],
    arch: TileDevice,
    topk: int = 10,
    extra_tags: list[str] | None = None,
) -> list[Hint] | None
```

Entry point for generating hints from a multi-node computation graph. Attempts `TensorCorePolicy` first, falling back to `DefaultPolicy` if no valid configurations are found.

---

## Configuration Search Algorithm Summary

The overall search algorithm in `DefaultPolicy.emit_config()` follows this flow:

```
1. get_base_tile()
   - Find minimum tile that eliminates computation redundancy
   - Score by workload_per_item = compute / num_items

2. _assign_reduce_step(node)
   - For each reduction axis, find optimal step size
   - Score by memory coalescing similarity (Dice coefficient)
   - Enlarge steps greedily until coalescing stops improving

3. dfs_smem_tile(base_tile, rstep_map)
   - BFS-like exploration of tile space
   - Priority queue ordered by (traffic + 1) * num_wave
   - For each tile:
     a. Compute memory traffic
     b. Compute shared memory usage (BestFit allocator)
     c. Validate register and shared memory limits
     d. Compute occupancy (blocks_per_SM)

4. _expand_reduce_axis(td)
   - Enlarge reduction steps within shared memory budget
   - Optimize for memory coalescing

5. assign_block_size(td)
   - For each valid tile configuration:
     a. Factorize block_size into prime factors
     b. Greedily assign factors to spatial dimensions
     c. Assign remaining factors to reduction dimensions
     d. Score by memory bandwidth utilization
     e. Plan vectorization sizes
     f. Plan rasterization strategy
```

For `TensorCorePolicy`, the search additionally:
- Constrains tiles to be multiples of WMMA/MMA shapes (16x16 minimum)
- Distributes threads in terms of warps rather than individual threads
- Multiplies shared memory by pipeline stages for double/triple buffering
- Plans `Rasterization2DColumn` when L2 cache can hold the working set
