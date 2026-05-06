# Apache TVM Reference - Chapter 14: Schedule Primitives

This reference provides a comprehensive guide to TVM's schedule primitives for TensorIR. Schedule primitives are the building blocks for transforming TIR programs. They allow fine-grained control over loop structure, memory hierarchy, parallelism, and mapping to hardware resources. The scheduling API operates on `s_tir.Schedule`, which provides a trace-based interface for recording and replaying schedule transformations.

---

## 14.1 Schedule Class Overview

### 14.1.1 Creating a Schedule

The `Schedule` class wraps an IRModule and provides methods to transform its TIR functions. All transformations are recorded as a trace, enabling reproducibility and integration with auto-tuning systems.

```python
import tvm

# Create a schedule from an IRModule
sch = tvm.s_tir.Schedule(mod)

# Create with debug flag for detailed error messages
sch = tvm.s_tir.Schedule(mod, debug_mask="all")

# Create with a specific seed for reproducibility of random sampling
sch = tvm.s_tir.Schedule(mod, seed=42)
```

**Constructor:**
```python
tvm.s_tir.Schedule(
    mod: ir.IRModule,
    *,
    debug_mask: str = "",
    seed: Optional[int] = None,
) -> Schedule
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mod` | `ir.IRModule` | required | The IRModule to schedule |
| `debug_mask` | `str` | `""` | Controls which internal checks are enabled. `"all"` enables all checks |
| `seed` | `Optional[int]` | `None` | Random seed for sampling primitives |

### 14.1.2 Schedule State

The schedule maintains internal state including:
- **Block reference tracking**: Maps block RVs (random variables) to their corresponding TIR Block SRefs
- **Loop reference tracking**: Maps loop RVs to For loop SRefs
- **Trace log**: Records all applied primitives for reproduction

```python
# Access the underlying modified IRModule
updated_mod = sch.mod

# Access the trace of applied primitives
trace = sch.trace
print(trace)  # Shows the sequence of applied schedule primitives
```

---

## 14.2 Block Access

### 14.2.1 `sch.get_block(name)`

Retrieves a block random variable (RV) by name. The block RV is a handle used by subsequent schedule primitives to identify the target block.

```python
block_rv = sch.get_block("C")
```

**Signature:**
```python
def get_block(
    name: str,
    func_name: str = "main",
) -> BlockRV
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The name of the block (as declared in `T.sblock("name")`) |
| `func_name` | `str` | The name of the PrimFunc containing the block (default: `"main"`) |

**Return value:** A `BlockRV` handle.

```python
from tvm.script import ir as I, tirx as T
import tvm

@I.ir_module
class MyModule:
    @T.prim_func
    def matmul(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

sch = tvm.s_tir.Schedule(MyModule)
block = sch.get_block("matmul")
# block is now a handle to the "matmul" block
```

### 14.2.2 `sch.get_blocks(name)`

Retrieves all blocks matching the given name. Returns a list of block RVs.

```python
blocks = sch.get_blocks("relu")
```

**Signature:**
```python
def get_blocks(
    name: str,
    func_name: str = "main",
) -> List[BlockRV]
```

This is useful when multiple blocks share the same name (e.g., after unrolling or specialization).

---

## 14.3 Loop Access

### 14.3.1 `sch.get_loops(block)`

Returns the loop variables surrounding a block, ordered from outermost to innermost.

```python
block = sch.get_block("matmul")
loops = sch.get_loops(block)
i, j, k = loops  # unpack into named variables
```

**Signature:**
```python
def get_loops(block: BlockRV) -> List[LoopRV]
```

**Return value:** A list of `LoopRV` handles, from outermost to innermost.

```python
# For a 3-dimensional loop nest
block = sch.get_block("C")
loops = sch.get_loops(block)
print(f"Number of surrounding loops: {len(loops)}")
# Output: Number of surrounding loops: 3
```

---

## 14.4 Computation Primitives

Computation primitives change where and how a block's computation is positioned within the overall program structure.

### 14.4.1 `sch.compute_inline(block)`

Inlines a producer block into its consumer. The producer's computation is substituted directly into the consumer's body, eliminating the intermediate buffer.

```python
# Before: A -> [elemwise] -> B -> [matmul] -> C
# After:  A -> [matmul with elemwise fused] -> C

producer = sch.get_block("elemwise")  # the block to inline
sch.compute_inline(producer)
```

**Signature:**
```python
def compute_inline(block: BlockRV) -> None
```

**Preconditions:**
- The block must be a producer of exactly one consumer block
- The block must be element-wise (no reduction axis)
- The consumer must access the entire output region of the producer

**Effect:** The producer block is removed from the IR, and its computation is embedded into the consumer. This eliminates the intermediate buffer allocation and reduces memory traffic.

```python
import tvm
from tvm.script import ir as I, tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def fused(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        # Intermediate buffer
        D = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.sblock("elemwise"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = A[vi, vj] * T.float32(2.0)  # producer
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + D[vi, vk] * B[vk, vj]  # consumer

sch = tvm.s_tir.Schedule(Module)
elemwise_block = sch.get_block("elemwise")
sch.compute_inline(elemwise_block)
# Now the matmul directly reads A * 2.0 instead of reading D
```

### 14.4.2 `sch.compute_root(block)`

Moves a block to the root scope of the function, making it an independent computation step rather than being nested inside another block's loop.

```python
inner_block = sch.get_block("bias_add")
sch.compute_root(inner_block)
```

**Signature:**
```python
def compute_root(block: BlockRV) -> None
```

**Effect:** The block is hoisted out of its parent block's loop nest and placed at the top level. This is the inverse of `compute_inline` in some cases and is useful when a block needs to be scheduled independently.

```python
sch = tvm.s_tir.Schedule(mod)
bias_block = sch.get_block("bias_add")

# Move bias addition to root level for separate optimization
sch.compute_root(bias_block)

# Now we can independently schedule the bias block
bias_loops = sch.get_loops(bias_block)
sch.parallel(bias_loops[0])
```

### 14.4.3 `sch.reverse_compute_inline(block)`

Inlines a consumer block into its producer. Unlike `compute_inline` which inlines the producer into the consumer, this primitive inlines the consumer into the producer's loop body.

```python
# Before: A -> [matmul] -> B -> [relu] -> C
# After:  A -> [matmul with relu fused] -> C

consumer = sch.get_block("relu")  # the consumer to inline
sch.reverse_compute_inline(consumer)
```

**Signature:**
```python
def reverse_compute_inline(block: BlockRV) -> None
```

**Preconditions:**
- The block must be a consumer of exactly one producer block
- The block must be element-wise (no reduction axis)
- The producer must write the entire region that the consumer reads

### 14.4.4 `sch.reverse_compute_root(block)`

The reverse of `compute_root`. Moves a block from the root level back into the loop nest of its producer/consumer.

```python
block = sch.get_block("my_block")
sch.reverse_compute_root(block)
```

**Signature:**
```python
def reverse_compute_root(block: BlockRV) -> None
```

---

## 14.5 Loop Manipulation

Loop manipulation primitives restructure the loop nest to expose parallelism, improve locality, or match hardware constraints.

### 14.5.1 `sch.tile(loop, factors)`

Tiles a loop into multiple nested loops according to the given factors. Loop tiling decomposes a single large iteration space into smaller tiles that fit in fast memory.

```python
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Tile the i and j loops with factors [32, 32]
i_outer, i_inner = sch.split(i, factors=[4, 32])
j_outer, j_inner = sch.split(j, factors=[4, 32])

# Or equivalently, tile both at once using tile
# sch.tile(i, factors=[4, 32])
```

**Signature:**
```python
def tile(
    loop: LoopRV,
    *factors: List[Union[int, None]],
) -> List[LoopRV]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `loop` | `LoopRV` | The loop to tile |
| `factors` | `List[Union[int, None]]` | Tiling factors. Use `None` for the outermost factor to auto-compute |

**Return value:** A list of `LoopRV` from outermost to innermost.

```python
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Tile i into 4 x 32 (total extent = 128)
i_outer, i_inner = sch.split(i, factors=[4, 32])
# Now i_outer iterates 0..3, i_inner iterates 0..31
# Original: for i in range(128)
# Tiled:    for i_outer in range(4): for i_inner in range(32)
```

### 14.5.2 `sch.split(loop, factors)`

Splits a loop into multiple loops with the given factors. The product of all factors must equal the original loop extent.

```python
i_outer, i_inner = sch.split(i, factors=[None, 32])
# None means "auto-compute the remaining factor"
```

**Signature:**
```python
def split(
    loop: LoopRV,
    factors: List[Union[int, None]],
) -> List[LoopRV]
```

**Details:**

- `factors=[4, 32]` splits a loop of extent 128 into an outer loop of extent 4 and an inner loop of extent 32.
- `factors=[None, 32]` auto-computes the outer factor as `ceil(original_extent / 32)`.
- `factors=[4, None]` auto-computes the inner factor as `ceil(original_extent / 4)`.
- Only one factor can be `None`.

```python
# Common GPU tiling pattern
block = sch.get_block("matmul")
i, j, k = sch.get_loops(block)

# Split for thread blocks and threads
i_bx, i_tx = sch.split(i, factors=[None, 64])
j_by, j_ty = sch.split(j, factors=[None, 64])

# Bind to GPU thread hierarchy
sch.bind(i_bx, "blockIdx.x")
sch.bind(j_by, "blockIdx.y")
sch.bind(i_tx, "threadIdx.x")
sch.bind(j_ty, "threadIdx.y")
```

### 14.5.3 `sch.fuse(*loops)`

Fuses multiple consecutive loops into a single loop. The fused loop's extent is the product of the input loops' extents.

```python
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Fuse the i and j loops into a single loop
ij_fused = sch.fuse(i, j)
```

**Signature:**
```python
def fuse(*loops: LoopRV) -> LoopRV
```

**Preconditions:**
- The loops must be perfectly nested (no statements between them)
- The loops must be adjacent in the loop nest
- The loops must have the same parent scope

```python
# Fuse to create a single parallel loop for GPU launch
block = sch.get_block("elementwise")
i, j = sch.get_loops(block)
ij = sch.fuse(i, j)
sch.bind(ij, "threadIdx.x")
```

### 14.5.4 `sch.reorder(*loops)`

Reorders loops in the specified order. This is used to bring loops into an optimal order for data locality or parallelism.

```python
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Reorder to put the reduction loop innermost
sch.reorder(j, i, k)
```

**Signature:**
```python
def reorder(*loops: LoopRV) -> None
```

**Details:**

The reorder primitive accepts an arbitrary number of loop RVs and rearranges them into the specified order. Only the loops that are explicitly provided are reordered; other loops remain in their original positions.

```python
# After tiling, reorder for optimal memory access pattern
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

i_o, i_i = sch.split(i, factors=[4, 32])
j_o, j_i = sch.split(j, factors=[4, 32])

# Reorder: outer tiles first, then inner tiles, then reduction
sch.reorder(i_o, j_o, i_i, j_i, k)
```

### 14.5.5 `sch.add_unit_loop(block)`

Adds a unit loop (extent 1) around a block that has no surrounding loops. This is useful as a scaffolding for subsequent scheduling operations that expect the block to be inside a loop.

```python
block = sch.get_block("my_block")
sch.add_unit_loop(block)
```

**Signature:**
```python
def add_unit_loop(block: BlockRV) -> None
```

### 14.5.6 `sch.add_loop(block)`

Adds a loop around a block with a specified extent.

```python
block = sch.get_block("my_block")
loop = sch.add_loop(block, "i", 0, 128)
```

**Signature:**
```python
def add_loop(
    block: BlockRV,
    loop_var_name: str,
    min_val: int,
    max_val: int,
) -> LoopRV
```

---

## 14.6 Parallelization

Parallelization primitives map loops to execution units: SIMD vector lanes, CPU threads, GPU thread hierarchy, or unrolled sequences.

### 14.6.1 `sch.vectorize(loop)`

Marks a loop for SIMD vectorization. The loop body is compiled into vector instructions that operate on multiple data elements simultaneously.

```python
block = sch.get_block("elementwise")
i, j = sch.get_loops(block)
sch.vectorize(j)  # vectorize the inner loop
```

**Signature:**
```python
def vectorize(loop: LoopRV) -> None
```

**Effect:** The loop is replaced by a single vectorized statement. The loop extent determines the vector width. For example, if the loop has extent 4 and the data type is float32, the resulting vector instruction operates on 4-wide float32x4 vectors.

```python
# Typical CPU scheduling with vectorization
block = sch.get_block("vector_add")
i = sch.get_loops(block)[0]
i_outer, i_inner = sch.split(i, factors=[None, 8])
sch.parallel(i_outer)
sch.vectorize(i_inner)
```

### 14.6.2 `sch.parallel(loop)`

Marks a loop for parallel execution on CPU cores. The loop iterations are distributed across available CPU threads.

```python
block = sch.get_block("large_compute")
i = sch.get_loops(block)[0]
sch.parallel(i)
```

**Signature:**
```python
def parallel(loop: LoopRV) -> None
```

**Effect:** The loop is annotated with `parallel` and will be executed using OpenMP-style parallelism during code generation. Each iteration is assigned to a different CPU thread.

```python
# CPU scheduling pattern for large element-wise operations
block = sch.get_block("elemwise")
i, j = sch.get_loops(block)
i_outer, i_inner = sch.split(i, factors=[None, 64])
j_outer, j_inner = sch.split(j, factors=[None, 64])
sch.reorder(i_outer, j_outer, i_inner, j_inner)
sch.parallel(i_outer)
sch.vectorize(j_inner)
```

### 14.6.3 `sch.unroll(loop, factor)`

Unrolls a loop by the specified factor. Full unrolling uses `factor=0`.

```python
block = sch.get_block("matmul")
i, j, k = sch.get_loops(block)
sch.unroll(k, factor=0)  # fully unroll the reduction loop
```

**Signature:**
```python
def unroll(
    loop: LoopRV,
    factor: int = 0,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `loop` | `LoopRV` | The loop to unroll |
| `factor` | `int` | Unroll factor. `0` means full unrolling. `-1` means unroll with default factor from PassContext |

**Effect:** The loop is annotated for unrolling during code generation. Partial unrolling replicates the loop body `factor` times and adjusts the loop extent. Full unrolling eliminates the loop entirely.

```python
# Common GPU pattern: unroll inner loops for instruction-level parallelism
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

i_o, i_i = sch.split(i, factors=[None, 32])
j_o, j_i = sch.split(j, factors=[None, 32])
sch.reorder(i_o, j_o, k, i_i, j_i)

sch.unroll(i_i, factor=-1)
sch.unroll(j_i, factor=-1)
```

### 14.6.4 `sch.bind(loop, thread_axis)`

Binds a loop to a GPU thread axis. This maps the loop iterations to GPU thread blocks or individual threads.

```python
block = sch.get_block("vector_add")
i = sch.get_loops(block)[0]
sch.bind(i, "threadIdx.x")
```

**Signature:**
```python
def bind(
    loop: LoopRV,
    thread_axis: str,
) -> None
```

**Supported thread axes:**

| Thread Axis | Description | Typical Max Extent |
|-------------|-------------|-------------------|
| `"blockIdx.x"` | X dimension of thread block grid | 2^31 - 1 |
| `"blockIdx.y"` | Y dimension of thread block grid | 65535 |
| `"blockIdx.z"` | Z dimension of thread block grid | 65535 |
| `"threadIdx.x"` | X dimension within a thread block | 1024 |
| `"threadIdx.y"` | Y dimension within a thread block | 1024 |
| `"threadIdx.z"` | Z dimension within a thread block | 64 |
| `"vthread"` | Virtual thread (software threading) | Any |

**GPU scheduling pattern:**

```python
import tvm
from tvm.script import ir as I, tirx as T

@I.ir_module
class MatmulGPU:
    @T.prim_func
    def matmul(
        A: T.Buffer((256, 256), "float32"),
        B: T.Buffer((256, 256), "float32"),
        C: T.Buffer((256, 256), "float32"),
    ):
        for i, j, k in T.grid(256, 256, 256):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

sch = tvm.s_tir.Schedule(MatmulGPU)
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Tile for GPU: 16x16 thread blocks, 16x16 tiles
i_o, i_i = sch.split(i, factors=[16, 16])
j_o, j_i = sch.split(j, factors=[16, 16])

# Bind to GPU thread hierarchy
sch.bind(i_o, "blockIdx.y")
sch.bind(j_o, "blockIdx.x")
sch.bind(i_i, "threadIdx.y")
sch.bind(j_i, "threadIdx.x")
```

### 14.6.5 `sch.blockize(loop)`

Converts a loop into a block-level operation. This wraps the loop body into a new block, which can then be scheduled as a unit.

```python
block = sch.get_block("inner_compute")
loops = sch.get_loops(block)
sch.blockize(loops[-1])
```

**Signature:**
```python
def blockize(loop: LoopRV) -> BlockRV
```

**Return value:** A new `BlockRV` representing the created block.

### 14.6.6 `sch.decompose_reduction(block, loop)`

Decomposes a reduction block into three parts: an initialization block, a parallel reduction body, and a finalization step. This is essential for parallelizing reductions on GPU.

```python
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Split the reduction loop
k_o, k_i = sch.split(k, factors=[None, 16])

# Decompose the reduction at the outer k loop
sch.decompose_reduction(block, k_o)
```

**Signature:**
```python
def decompose_reduction(
    block: BlockRV,
    loop: LoopRV,
) -> BlockRV
```

**Effect:** The reduction block is split into:
1. An **init block** that sets the accumulator to the identity value (e.g., 0 for sum)
2. An **update block** that accumulates partial results
3. Optionally, a **finalize block** that applies any post-reduction operations

```python
# Full GPU reduction decomposition
sch = tvm.s_tir.Schedule(mod)
block = sch.get_block("matmul")
i, j, k = sch.get_loops(block)

# Tile for GPU
i_o, i_i = sch.split(i, factors=[None, 16])
j_o, j_i = sch.split(j, factors=[None, 16])
k_o, k_i = sch.split(k, factors=[None, 16])

# Decompose reduction at the outer k loop
init_block = sch.decompose_reduction(block, k_o)

# Now init_block handles C = 0, and the original block handles C += A*B
```

---

## 14.7 Memory Access

Memory access primitives control data movement between memory hierarchies and reshape how buffers are accessed.

### 14.7.1 `sch.cache_read(block, read_buffer_index, storage_scope)`

Creates a cached copy of an input buffer in a specified storage scope. This is the primary mechanism for loading data into shared memory or registers on GPUs.

```python
block = sch.get_block("C")
# Cache the first input (A) into shared memory
sch.cache_read(block, 0, "shared")
# Cache the second input (B) into shared memory
sch.cache_read(block, 1, "shared")
```

**Signature:**
```python
def cache_read(
    block: BlockRV,
    read_buffer_index: int,
    storage_scope: str,
    consumer_blocks: Optional[List[BlockRV]] = None,
) -> BlockRV
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block` | `BlockRV` | The consumer block |
| `read_buffer_index` | `int` | Index of the input buffer to cache (0-based) |
| `storage_scope` | `str` | Target storage scope |
| `consumer_blocks` | `Optional[List[BlockRV]]` | Specific consumer blocks (default: all) |

**Common storage scopes:**

| Scope | Description | GPU Location |
|-------|-------------|--------------|
| `"local"` | Thread-private (register) memory | Registers |
| `"shared"` | Thread block shared memory | Shared memory |
| `"global"` | Device global memory | Global memory |
| `"global.texture"` | Texture memory (GPU) | Texture cache |
| `"wmma.accumulator"` | WMMA accumulator (tensor core) | Tensor core regs |

**Return value:** A `BlockRV` for the newly created cache read block.

```python
# Full GPU matmul with shared memory caching
sch = tvm.s_tir.Schedule(mod)
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Cache A and B into shared memory
A_shared = sch.cache_read(block, 0, "shared")
B_shared = sch.cache_read(block, 1, "shared")

# Further cache into registers (local memory)
A_local = sch.cache_read(block, 0, "local")
B_local = sch.cache_read(block, 1, "local")

# Now schedule the cache reads to be inside the appropriate loops
# ... tile, reorder, bind as needed
```

### 14.7.2 `sch.cache_write(block, write_buffer_index, storage_scope)`

Creates a cached copy for an output buffer, writing first to the cache and then copying back to the original buffer.

```python
block = sch.get_block("C")
sch.cache_write(block, 0, "local")  # Write to registers first
```

**Signature:**
```python
def cache_write(
    block: BlockRV,
    write_buffer_index: int,
    storage_scope: str,
) -> BlockRV
```

**Effect:** A new intermediate buffer is created in the specified storage scope. The computation writes to this buffer, and a subsequent copy block transfers the result back to the original output buffer.

```python
# Write accumulator to registers, then copy to global memory
block = sch.get_block("C")
C_local = sch.cache_write(block, 0, "local")

# Now C_local is a handle to the intermediate write block
# We can schedule it independently
C_local_loops = sch.get_loops(C_local)
```

### 14.7.3 `sch.reindex(block, buffer_index)`

Transforms the access pattern of a buffer within a block. This changes how indices map to the buffer without changing the buffer itself.

```python
block = sch.get_block("C")
sch.reindex(block, 0)  # Reindex the first buffer access
```

**Signature:**
```python
def reindex(
    block: BlockRV,
    buffer_index: int,
) -> BlockRV
```

**Effect:** Creates a new layout mapping for the buffer access, which can improve memory coalescing on GPUs by ensuring that adjacent threads access adjacent memory addresses.

### 14.7.4 `sch.transform_block_layout(block, index_map)`

Transforms the iteration domain of a block by applying an index mapping function.

```python
from tvm.tir import IndexMap

block = sch.get_block("C")
# Transform the block layout from (i, j) to (i * 4 + j % 4, j // 4)
index_map = IndexMap.from_func(lambda i, j: (i * 4 + j % 4, j // 4))
sch.transform_block_layout(block, index_map)
```

**Signature:**
```python
def transform_block_layout(
    block: BlockRV,
    index_map: tir.IndexMap,
) -> None
```

### 14.7.5 `sch.transform_layout(block, buffer_index, index_map)`

Transforms the memory layout of a buffer access within a block.

```python
from tvm.tir import IndexMap

block = sch.get_block("C")
# Transform from row-major to column-major access
index_map = IndexMap.from_func(lambda i, j: (j, i))
sch.transform_layout(block, 0, index_map)
```

**Signature:**
```python
def transform_layout(
    block: BlockRV,
    buffer_index: int,
    index_map: tir.IndexMap,
) -> None
```

**Common use cases:**
- **Layout transformation**: Convert between NCHW and NHWC layouts
- **Padding**: Add padding dimensions for memory alignment
- **Bank conflict avoidance**: Remap shared memory access patterns to avoid bank conflicts on GPUs

---

## 14.8 Synchronization

### 14.8.1 `sch.storage_align(block, buffer_index, axis, factor, offset)`

Sets storage alignment constraints for a buffer, ensuring that rows are padded to specific boundaries for optimal memory access.

```python
block = sch.get_block("shared_load")
sch.storage_align(block, 0, axis=1, factor=8, offset=0)
```

**Signature:**
```python
def storage_align(
    block: BlockRV,
    buffer_index: int,
    axis: int,
    factor: int,
    offset: int,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block` | `BlockRV` | The block containing the buffer |
| `buffer_index` | `int` | Index of the buffer to align |
| `axis` | `int` | The dimension to align |
| `factor` | `int` | Alignment factor (pad to multiples of this value) |
| `offset` | `int` | Alignment offset |

**Details:**

This primitive is critical for avoiding shared memory bank conflicts on NVIDIA GPUs. Shared memory has 32 banks, and simultaneous accesses to the same bank cause serialization. By padding each row, the access pattern is shifted to distribute accesses across banks.

```python
# Avoid shared memory bank conflicts in GPU matmul
sch = tvm.s_tir.Schedule(mod)
block = sch.get_block("C")

A_shared = sch.cache_read(block, 0, "shared")
B_shared = sch.cache_read(block, 1, "shared")

# Pad shared memory rows to avoid bank conflicts
# 32 banks * 4 bytes = 128 byte offset cycles
sch.storage_align(A_shared, 0, axis=1, factor=8, offset=0)
sch.storage_align(B_shared, 0, axis=1, factor=8, offset=0)
```

### 14.8.2 `sch.set_scope(block, buffer_index, storage_scope)`

Changes the storage scope of a buffer allocation. This is used to move buffers between memory hierarchies.

```python
block = sch.get_block("intermediate")
sch.set_scope(block, 0, "shared")
```

**Signature:**
```python
def set_scope(
    block: BlockRV,
    buffer_index: int,
    storage_scope: str,
) -> None
```

```python
# Change a buffer from global to shared memory
block = sch.get_block("compute")
sch.set_scope(block, 0, "shared")
# The buffer will now be allocated in shared memory instead of global memory
```

---

## 14.9 Annotation

Annotation primitives attach metadata to blocks and loops, guiding subsequent transformation and code generation passes.

### 14.9.1 `sch.annotate(block_or_loop, ann_key, ann_val)`

Adds an annotation (key-value pair) to a block or loop.

```python
block = sch.get_block("C")
sch.annotate(block, "auto_tensorize", True)
sch.annotate(block, "pragma_import_c", "my_custom_kernel.h")
```

**Signature:**
```python
def annotate(
    block_or_loop: Union[BlockRV, LoopRV],
    ann_key: str,
    ann_val: Any,
) -> None
```

**Common annotation keys:**

| Key | Value Type | Description |
|-----|------------|-------------|
| `"pragma_auto_unroll_max_step"` | `int` | Maximum step for auto-unroll |
| `"pragma_unroll_explicit"` | `bool` | Whether to emit explicit unroll |
| `"storage_scope"` | `str` | Storage scope for buffer |
| `"buffer_dim_align"` | `tuple` | Buffer alignment specification |
| `"warp_execution"` | `bool` | Enable warp-level execution |
| `"auto_tensorize"` | `bool/str` | Enable auto-tensorization |

### 14.9.2 `sch.unannotate(block_or_loop, ann_key)`

Removes an annotation from a block or loop.

```python
block = sch.get_block("C")
sch.unannotate(block, "auto_tensorize")
```

**Signature:**
```python
def unannotate(
    block_or_loop: Union[BlockRV, LoopRV],
    ann_key: str,
) -> None
```

### 14.9.3 `sch.tensorize(block_or_loop, intrin)`

Replaces a block or loop body with a hardware tensor intrinsic. This is the mechanism for utilizing specialized hardware units such as NVIDIA Tensor Cores (WMMA), Intel AMX, or ARM dot products.

```python
block = sch.get_block("matmul_16x16")

# Register a tensor intrinsic
from tvm.tir import TensorIntrin
intrin = TensorIntrin.get("wmma_sync_16x16x16_f16f16f32")

sch.tensorize(block, intrin)
```

**Signature:**
```python
def tensorize(
    block_or_loop: Union[BlockRV, LoopRV],
    intrin: Union[str, tir.TensorIntrin],
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_or_loop` | `BlockRV or LoopRV` | The block/loop to tensorize |
| `intrin` | `str or TensorIntrin` | The tensor intrinsic to apply (name or object) |

```python
# Full tensorization workflow for NVIDIA Tensor Cores
from tvm.tir.tensor_intrin import cuda as cuda_intrin

sch = tvm.s_tir.Schedule(mod)
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Tile to match WMMA 16x16x16 dimensions
i_o, i_i = sch.split(i, factors=[None, 16])
j_o, j_i = sch.split(j, factors=[None, 16])
k_o, k_i = sch.split(k, factors=[None, 16])

# Reorder loops for WMMA
sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

# Tensorize the inner block
sch.tensorize(i_i, "wmma_load_16x16x16_f16_shared")
sch.tensorize(j_i, "wmma_load_16x16x16_f16_shared")
sch.tensorize(k_i, "wmma_sync_16x16x16_f16f16f32")
```

---

## 14.10 Sampling Primitives (MetaSchedule)

Sampling primitives introduce randomness into the scheduling process, enabling MetaSchedule to explore different optimization strategies. These primitives return sampled values that are recorded in the schedule trace.

### 14.10.1 `sch.sample_perfect_tile(loop, n, max_innermost_factor)`

Samples a perfect tiling of a loop into `n` factors. The factors are sampled uniformly at random from valid tilings.

```python
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Sample a 3-level tiling for the i loop
factors = sch.sample_perfect_tile(i, n=3, max_innermost_factor=64)
i_o, i_m, i_i = factors
```

**Signature:**
```python
def sample_perfect_tile(
    loop: LoopRV,
    n: int,
    max_innermost_factor: int = 64,
) -> List[LoopRV]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `loop` | `LoopRV` | The loop to tile |
| `n` | `int` | Number of tiling levels |
| `max_innermost_factor` | `int` | Maximum extent of the innermost tile (default: 64) |

**Return value:** A list of `n` `LoopRV` objects from outermost to innermost.

**Details:**

"Perfect tiling" means the product of all factors exactly equals the original loop extent (no remainder handling). The sampling is uniform over valid decompositions, subject to the `max_innermost_factor` constraint. This ensures that the innermost loop is small enough to fit in fast memory.

### 14.10.2 `sch.sample_categorical(candidates, probs)`

Samples a categorical choice from a list of candidates with the given probability distribution.

```python
# Sample a vectorization factor
vec_factor = sch.sample_categorical(
    candidates=[1, 2, 4, 8, 16],
    probs=[0.1, 0.1, 0.3, 0.3, 0.2],
)
```

**Signature:**
```python
def sample_categorical(
    candidates: List[int],
    probs: List[float],
) -> LoopRV
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `candidates` | `List[int]` | List of candidate values |
| `probs` | `List[float]` | Probability for each candidate (must sum to 1.0) |

**Return value:** The index of the sampled candidate (as an integer-valued expression).

```python
# MetaSchedule-style random sampling of tile sizes
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Sample the number of tiling levels
n_levels = sch.sample_categorical([2, 3], [0.5, 0.5])

# Sample tiling for each loop
i_factors = sch.sample_perfect_tile(i, n=n_levels, max_innermost_factor=128)
j_factors = sch.sample_perfect_tile(j, n=n_levels, max_innermost_factor=128)
```

---

## 14.11 Information Queries

### 14.11.1 `sch.get(block_or_loop)`

Returns the underlying IR reference (SRef) for a block or loop random variable.

```python
block_rv = sch.get_block("C")
sref = sch.get(block_rv)
print(sref.stmt)  # Access the actual TIR Block statement
```

**Signature:**
```python
def get(block_or_loop: Union[BlockRV, LoopRV]) -> StmtSRef
```

### 14.11.2 `sch.remove_inner_unit_loop(block)`

Removes inner unit loops (loops with extent 1) surrounding a block. This is a cleanup operation after transformations that may leave trivial loops.

```python
block = sch.get_block("C")
sch.remove_inner_unit_loop(block)
```

**Signature:**
```python
def remove_inner_unit_loop(block: BlockRV) -> None
```

### 14.11.3 `sch.pragma(loop, pragma_type, pragma_value)`

Adds a pragma annotation to a loop. Pragmas guide the code generator's behavior.

```python
loop = sch.get_loops(sch.get_block("compute"))[0]
sch.pragma(loop, "auto_unroll_max_step", 16)
```

**Signature:**
```python
def pragma(
    loop: LoopRV,
    pragma_type: str,
    pragma_value: Union[int, str, bool],
) -> None
```

**Common pragma types:**

| Pragma | Value | Description |
|--------|-------|-------------|
| `"auto_unroll_max_step"` | `int` | Maximum unroll step count |
| `"unroll_explicit"` | `bool` | Generate explicit unrolled code |
| `"vector_length"` | `int` | Force specific vector width |
| `"parallel_launch_point"` | `bool` | Mark parallel launch point |
| `"parallel_stride_pattern"` | `int` | Stride for parallel execution |

---

## 14.12 Complete Scheduling Example

This section demonstrates a complete GPU scheduling workflow for matrix multiplication, combining multiple primitives:

```python
import tvm
from tvm.script import ir as I, tirx as T

@I.ir_module
class MatmulModule:
    @T.prim_func
    def matmul(
        A: T.Buffer((512, 512), "float16"),
        B: T.Buffer((512, 512), "float16"),
        C: T.Buffer((512, 512), "float32"),
    ):
        for i, j, k in T.grid(512, 512, 512):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

sch = tvm.s_tir.Schedule(MatmulModule)
block = sch.get_block("C")
C_local = sch.cache_write(block, 0, "local")

# Get loops of the updated block
i, j, k = sch.get_loops(block)

# Multi-level tiling
i_o, i_i = sch.split(i, factors=[32, 16])
j_o, j_i = sch.split(j, factors=[32, 16])
k_o, k_i = sch.split(k, factors=[64, 8])

# Cache reads into shared memory
A_shared = sch.cache_read(block, 0, "shared")
B_shared = sch.cache_read(block, 1, "shared")

# Cache reads into local (register) memory
A_local = sch.cache_read(block, 0, "local")
B_local = sch.cache_read(block, 1, "local")

# Reorder for optimal access pattern
sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

# Bind to GPU thread hierarchy
sch.bind(i_o, "blockIdx.y")
sch.bind(j_o, "blockIdx.x")
sch.bind(i_i, "threadIdx.y")
sch.bind(j_i, "threadIdx.x")

# Unroll inner loops for instruction-level parallelism
sch.unroll(k_i, factor=-1)

# Apply storage alignment to avoid bank conflicts
sch.storage_align(A_shared, 0, axis=1, factor=8, offset=0)
sch.storage_align(B_shared, 0, axis=1, factor=8, offset=0)

# Verify the schedule
print(sch.mod.script())
```

---

## 14.13 Summary of Schedule Primitives

| Category | Primitive | Description |
|----------|-----------|-------------|
| **Block Access** | `get_block` | Retrieve block by name |
| | `get_blocks` | Retrieve all blocks matching name |
| **Loop Access** | `get_loops` | Get surrounding loops |
| **Computation** | `compute_inline` | Inline producer into consumer |
| | `compute_root` | Move block to root scope |
| | `reverse_compute_inline` | Inline consumer into producer |
| | `reverse_compute_root` | Move block back into parent |
| **Loop** | `split` | Split loop into multiple levels |
| | `tile` | Tile a loop with factors |
| | `fuse` | Merge loops into one |
| | `reorder` | Rearrange loop order |
| | `add_unit_loop` | Add trivial extent-1 loop |
| | `add_loop` | Add a new loop |
| **Parallel** | `vectorize` | SIMD vectorize a loop |
| | `parallel` | Multi-thread a loop |
| | `unroll` | Unroll a loop |
| | `bind` | Bind to GPU thread axis |
| | `blockize` | Convert loop to block |
| | `decompose_reduction` | Split reduction into init+update |
| **Memory** | `cache_read` | Cache input to scope |
| | `cache_write` | Cache output to scope |
| | `reindex` | Transform buffer access pattern |
| | `transform_block_layout` | Transform block iteration domain |
| | `transform_layout` | Transform buffer memory layout |
| **Sync** | `storage_align` | Align buffer storage |
| | `set_scope` | Change buffer storage scope |
| **Annotation** | `annotate` | Add metadata annotation |
| | `unannotate` | Remove annotation |
| | `tensorize` | Apply hardware intrinsic |
| **Sampling** | `sample_perfect_tile` | Random tiling for search |
| | `sample_categorical` | Random categorical choice |
| **Query** | `get` | Get IR reference |
| | `remove_inner_unit_loop` | Cleanup trivial loops |
| | `pragma` | Add codegen pragma |
