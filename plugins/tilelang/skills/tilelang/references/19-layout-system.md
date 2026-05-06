# TileLang Layout System Reference

## 1. Overview

The TileLang layout system provides a powerful abstraction for describing how
multi-dimensional data is mapped to memory addresses and hardware resources.
Layouts are fundamental to TileLang's ability to generate efficient Tensor Core
code, as they encode the complex data rearrangement patterns needed by hardware
matrix instructions.

The layout system consists of three main components:

1. **Layout** - A pure mathematical mapping from input indices to output indices
2. **Fragment** - A Layout extended with thread mapping (which thread owns which elements)
3. **Swizzle layouts** - Specialized layouts for avoiding shared memory bank conflicts

---

## 2. Layout Class

### 2.1 Definition

The `Layout` class is defined in `tilelang/layout/layout.py`:

```python
@tvm_ffi.register_object("tl.Layout")
class Layout(Node):
    def __init__(self, shape, forward_fn):
        """
        Initialize a Layout object.

        Parameters
        ----------
        shape : list of int
            The shape of the layout, defining the number of elements
            along each dimension.
        forward_fn : function
            A function that maps index variables to their computed
            forward index.
        """
```

The Layout is backed by a C++ implementation in `src/layout/layout.cc` that
stores the shape and forward mapping as TVM expressions.

### 2.2 Creating Layouts

#### Basic Layout

```python
from tilelang.layout import Layout

# Create a 2D layout mapping (i, j) -> i * 8 + j (row-major)
layout = Layout([16, 8], lambda i, j: i * 8 + j)
```

#### Column-Major Layout

```python
# Column-major: (i, j) -> j * 16 + i
layout = Layout([16, 8], lambda i, j: j * 16 + i)
```

#### Multi-Dimensional Layout

```python
# 3D layout
layout = Layout([4, 8, 16], lambda i, j, k: i * 128 + j * 16 + k)
```

### 2.3 Properties

#### index

Returns the forward index expression(s) of the layout:

```python
layout = Layout([16, 8], lambda i, j: i * 8 + j)
print(layout.index)  # The forward index PrimExpr
```

#### get_input_shape

Returns the input shape of the layout:

```python
shape = layout.get_input_shape()
# Returns: [16, 8]
```

#### get_output_shape

Returns the output shape of the layout:

```python
output_shape = layout.get_output_shape()
# Returns the shape of the output indices
```

#### get_forward_vars

Returns the iteration variables used in the layout:

```python
vars = layout.get_forward_vars()
# Returns: [IterVar(i0), IterVar(i1)]
```

### 2.4 Methods

#### map_forward_index

Computes the forward index for given input indices:

```python
from tvm.tir import Var

layout = Layout([16, 8], lambda i, j: i * 8 + j)
i, j = Var("i", "int32"), Var("j", "int32")
index = layout.map_forward_index([i, j])
# Returns: i * 8 + j
```

#### __call__ (shorthand)

```python
index = layout(i, j)  # Same as map_forward_index
```

#### repeat

Repeats a layout along a specific dimension:

```python
# Original layout: shape [4, 8]
layout = Layout([4, 8], lambda i, j: i * 8 + j)

# Repeat along dimension 0 by factor 4
repeated = layout.repeat(dim=0, factor=4)
# New shape: [16, 8], maps to [repeat_group, original_index]
```

The `repeat` operation creates a new layout `L'` such that:

```
L'(*idx) = [idx[dim] // extent_dim] + L(idx with idx[dim] % extent_dim)
```

where `extent_dim` is the original extent of the repeated dimension.

Parameters:
- `dim`: The input dimension to repeat (0-based, supports negative indexing)
- `factor`: The repeat factor (must be >= 1)

```python
# factor == 1 is a no-op
assert layout.repeat(0, 1) is layout
```

#### expand

Expands (lifts) the layout by prepending new leading dimensions:

```python
# 2D layout over [J, K]
layout_2d = Layout([8, 16], lambda j, k: j * 16 + k)

# Expand to 3D over [I, J, K]
layout_3d = layout_2d.expand([4])
# [i, j, k] -> [i, *layout_2d(j, k)]
```

Parameters:
- `leading_shape`: int or sequence of ints for new leading dimensions

The new leading dimensions are forwarded unchanged to the output, and the
original layout is applied to the remaining trailing dimensions.

#### inverse

Computes the inverse of the layout transformation:

```python
layout = Layout([16, 8], lambda i, j: i * 8 + j)
inv = layout.inverse()
# The inverse maps output indices back to input indices
```

#### reshape

Reshapes the input shape of the layout:

```python
layout = Layout([16, 8], lambda i, j: i * 8 + j)
reshaped = layout.reshape([8, 2, 8])
# Changes the input shape while preserving the logical mapping
```

Parameters:
- `shape`: New input shape
- `rescale_num`: Rescale numerator for element size changes
- `rescale_den`: Rescale denominator for element size changes

#### is_equal

Checks equality with another layout:

```python
layout1 = Layout([16, 8], lambda i, j: i * 8 + j)
layout2 = Layout([16, 8], lambda i, j: i * 8 + j)
assert layout1.is_equal(layout2)
```

---

## 3. Fragment Class

### 3.1 Definition

The `Fragment` class extends `Layout` with thread mapping information:

```python
@tvm_ffi.register_object("tl.Fragment")
class Fragment(Layout):
    def __init__(self, shape, forward_fn=None, forward_thread_fn=None,
                 replicate=1, forward_index_fn=None):
```

A Fragment captures:
- **shape**: The logical shape of the data
- **forward_index**: How logical indices map to memory indices
- **forward_thread**: How logical indices map to thread IDs
- **thread_replicate**: Replication factor for multi-thread access

### 3.2 Creating Fragments

#### With Combined forward_fn

When `forward_fn` is provided, it returns both thread and index:

```python
from tilelang.layout import Fragment

def my_forward(i, j, rep):
    thread = i * 4 + rep
    index = j
    return thread, index

frag = Fragment([16, 8], forward_fn=my_forward, replicate=4)
```

#### With Separate forward_thread_fn and forward_index_fn

```python
def thread_fn(i, j, rep):
    return i * 4 + rep

def index_fn(i, j):
    return j

frag = Fragment(
    [16, 8],
    forward_thread_fn=thread_fn,
    forward_index_fn=index_fn,
    replicate=4
)
```

### 3.3 Properties

#### thread

Returns the forward_thread expression:

```python
frag = Fragment(...)
thread_expr = frag.thread  # IterVar representing thread mapping
```

#### get_thread_size

Returns the extent of the thread dimension:

```python
thread_size = frag.get_thread_size()
# For a 256-thread fragment: returns 256
```

### 3.4 Methods

#### map_forward_thread

Computes the thread mapping for given indices:

```python
from tvm.tir import Var

frag = Fragment(...)
i, j = Var("i", "int32"), Var("j", "int32")
thread_id = frag.map_forward_thread([i, j])
```

#### repeat

Returns a new Fragment that repeats the iteration space:

```python
repeated = frag.repeat(repeats=2, repeat_on_thread=False, lower_dim_first=True)
```

Parameters:
- `repeats`: Number of times to repeat
- `repeat_on_thread`: If True, repeat along the thread dimension
- `lower_dim_first`: If True, repeat on lower dimensions first

#### replicate

Replicates the Fragment across a new thread dimension:

```python
replicated = frag.replicate(replicate=4)
```

#### condense_rep_var

Condenses the replicate variable into the existing iteration space:

```python
condensed = frag.condense_rep_var()
```

#### is_equal

Checks equality with another fragment:

```python
frag1 = Fragment(...)
frag2 = Fragment(...)
assert frag1.is_equal(frag2)
```

---

## 4. Swizzle Layouts

### 4.1 Purpose

Swizzle layouts rearrange data in shared memory to prevent bank conflicts.
In NVIDIA GPUs, shared memory has 32 banks, and simultaneous access to the
same bank by different threads causes serialization (bank conflicts). Swizzling
uses XOR-based address transformations to distribute accesses evenly.

### 4.2 make_swizzled_layout

```python
from tilelang.layout import make_swizzled_layout

layout = make_swizzled_layout(buffer, k_major=True, allow_pad=True)
```

Creates a swizzled layout suitable for TMA and general copy operations.

Parameters:
- `buffer`: A TIR buffer (or BufferLoad/BufferRegion) to create layout for
- `k_major`: If True, the K dimension is the inner (contiguous) dimension
- `allow_pad`: If True, allows padding to achieve better swizzle patterns

Implementation: `_ffi_api.make_swizzled_layout(buf, k_major, allow_pad)`
in `src/layout/layout.cc`

### 4.3 make_volta_swizzled_layout

```python
from tilelang.layout import make_volta_swizzled_layout

layout = make_volta_swizzled_layout(buffer, is_a=True, k_inner=True)
```

Creates a swizzled layout optimized for Volta (SM70) Tensor Core operations.

Parameters:
- `buffer`: TIR buffer for the matrix operand
- `is_a`: True for matrix A, False for matrix B
- `k_inner`: If True, K dimension is innermost

Volta Tensor Core has specific layout requirements due to the HMMA.884
instruction's register allocation pattern.

### 4.4 make_wgmma_swizzled_layout

```python
from tilelang.layout import make_wgmma_swizzled_layout

layout = make_wgmma_swizzled_layout(buffer, continuity=None, k_major=True)
```

Creates a swizzled layout for Hopper (SM90) WGMMA instructions.

Parameters:
- `buffer`: TIR buffer
- `continuity`: Minimum contiguous access width (None for auto-detect)
- `k_major`: If True, K dimension is the inner (contiguous) dimension

WGMMA instructions read from shared memory using descriptors, and the data
layout must match the hardware's expected swizzle pattern. The supported
swizzle modes are:

| Mode | Size | Pattern |
|---|---|---|
| SWIZZLE_128B | 128 bytes | XOR bits at positions [4,5,6] |
| SWIZZLE_64B | 64 bytes | XOR bits at positions [4,5] |
| SWIZZLE_32B | 32 bytes | XOR bit at position [4] |
| NONE | N/A | No swizzling |

### 4.5 make_tcgen05mma_swizzled_layout

```python
from tilelang.layout import make_tcgen05mma_swizzled_layout

layout = make_tcgen05mma_swizzled_layout(buffer, continuity=None, k_major=True)
```

Creates a swizzled layout for Blackwell (SM100) TCGEN05 MMA instructions.

Parameters:
- `buffer`: TIR buffer
- `continuity`: Minimum contiguous access width
- `k_major`: If True, K dimension is contiguous

TCGEN05 has different swizzle mode encoding compared to WGMMA:

| Mode | Encoding |
|---|---|
| SWIZZLE_128B | 2 |
| SWIZZLE_64B | 4 |
| SWIZZLE_32B | 6 |
| NONE | 0 |

---

## 5. Bank Swizzle Layouts

### 5.1 make_full_bank_swizzled_layout

```python
from tilelang.layout import make_full_bank_swizzled_layout

layout = make_full_bank_swizzled_layout(buffer)
```

Creates a 128-byte swizzle pattern that provides full bank conflict avoidance.

The XOR pattern for 128-byte swizzle:

```
address ^ ((address >> 4) & 0x7) << 4
```

This XORs bits [4,5,6] with bits [0,1,2] of the row index, ensuring that
consecutive rows access different banks.

### 5.2 make_half_bank_swizzled_layout

```python
from tilelang.layout import make_half_bank_swizzled_layout

layout = make_half_bank_swizzled_layout(buffer)
```

Creates a 64-byte swizzle pattern:

```
address ^ ((address >> 4) & 0x3) << 4
```

XORs bits [4,5] with bits [0,1] of the row index.

### 5.3 make_quarter_bank_swizzled_layout

```python
from tilelang.layout import make_quarter_bank_swizzled_layout

layout = make_quarter_bank_swizzled_layout(buffer)
```

Creates a 32-byte swizzle pattern:

```
address ^ ((address >> 4) & 0x1) << 4
```

XORs bit [4] with bit [0] of the row index.

---

## 6. Linear Layout

### 6.1 make_linear_layout

```python
from tilelang.layout import make_linear_layout

layout = make_linear_layout(buffer)
```

Creates a row-major linear layout for any dimension.

For a buffer with shape `[d0, d1, d2, ...]`, the linear layout maps:

```
(i0, i1, i2, ...) -> i0 * (d1 * d2 * ...) + i1 * (d2 * ...) + i2 * (...) + ...
```

This is the simplest layout, equivalent to flattening a multi-dimensional array
into a 1D array in row-major order.

Implementation: `_ffi_api.make_linear_layout(list(shape))`

---

## 7. GEMM Fragment Layouts

### 7.1 make_gemm_fragment_8x8

```python
from tilelang.layout import make_gemm_fragment_8x8

frag = make_gemm_fragment_8x8()
```

Creates a standard 8x8 GEMM fragment layout for `ldmatrix` / `stmatrix` operations.

This layout matches the warp-level matrix multiplication pattern used in
Tensor Core. For a single warp of 32 threads processing an 8x8 tile:

- Each thread holds 2 elements (8 * 8 / 32 = 2)
- The thread-to-element mapping follows the ldmatrix register layout

### 7.2 make_gemm_fragment_8x8_transposed

```python
from tilelang.layout import make_gemm_fragment_8x8_transposed

frag = make_gemm_fragment_8x8_transposed()
```

Creates a transposed 8x8 GEMM fragment layout. This is the transposed version
of `make_gemm_fragment_8x8`, useful for different access patterns in matrix
operations.

---

## 8. Fully Replicated Layout

### 8.1 make_fully_replicated_layout_fragment

```python
from tilelang.layout import make_fully_replicated_layout_fragment

frag = make_fully_replicated_layout_fragment(buffer, threads=256)
```

Creates a fully replicated fragment where all threads hold identical copies of
the entire buffer.

Parameters:
- `buffer`: TIR buffer to get shape information from
- `threads`: Number of threads (replicate extent)

Use cases:
- Index buffers that need uniform access across all threads
- Mask buffers for conditional operations
- Scale factors that are constant across threads

Example:

```python
C_local = T.alloc_fragment((2,), T.float32)
layout = make_fully_replicated_layout_fragment(C_local, 256)
T.annotate_layout({C_local: layout})
# Now all 256 threads have a complete copy of C_local
```

Implementation: `_ffi_api.make_fully_replicated_layout_fragment(list(shape), threads)`

---

## 9. Sparse Layouts

### 9.1 make_cutlass_metadata_layout

```python
from tilelang.layout import make_cutlass_metadata_layout

metadata_layout = make_cutlass_metadata_layout(
    buffer,
    mma_dtype="float16",
    arch=None,  # auto-detect
)
```

Creates a layout compatible with CUTLASS sparse GEMM compression kernels.

Parameters:
- `buffer`: Metadata buffer (8-bit or 16-bit type)
- `mma_dtype`: Data type of the MMA operand (e.g., "float16")
- `arch`: Target architecture (None for auto-detect)
- Extra args for SM90: `block_k` (tiling size along K)

#### SM90 Sparse Metadata Layout

For SM90 (Hopper), the metadata layout follows CUTLASS's sparse WGMMA format:

```python
metadata_layout = make_cutlass_metadata_layout(
    buffer, mma_dtype="float16", block_k=64
)
```

The layout atom depends on the MMA data type:

| mma_dtype | BlockK | Shape_I | Shape_K |
|---|---|---|---|
| float32 | 16 | [8, 2, 4] | [1, 2, NumK] |
| float16/bfloat16 | 32 | [8, 2, 4] | [2, 2, NumK] |
| int8/fp8 | 64 | [64] | [block_k // 8] |

Where `NumK = block_k // BlockK`.

The stride ordering is `[3, 1, 5, 0, 4, 2]` for the combined IK dimensions.

#### SM80 Sparse Metadata Layout

For SM80/SM86 (Ampere), the metadata layout uses a column-major interleaved format:

```python
metadata_layout = make_cutlass_metadata_layout(
    buffer, mma_dtype="float16"
)
```

The layout uses `ColumnMajorInterleaved` ordering:

```python
def ColumnMajorInterleaved(i: int, j: int) -> int:
    i = i // group * group + (i % 8) * interweave + (i % group) // 8
    topright = (1 - (i % 2)) & (j % 2)
    bottomleft = (i % 2) & (1 - (j % 2))
    i += topright - bottomleft
    j -= topright - bottomleft
    offset = (j // 2) * m * 2 + i * 2 + (j % 2)
    return offset // k, offset % k
```

---

## 10. Intrinsic Layouts

### 10.1 MMA Layout Functions

From `tilelang/intrinsics/mma_layout.py`:

These layout functions map between shared memory layouts and MMA register layouts.

#### Shared-to-MMA Layouts (Source -> Register)

```python
# For loading A matrix from shared to registers
shared_16x8_to_mma_32x4_layout_sr_a     # 16x8 shared -> 32x4 register (A)
shared_16x16_to_mma_32x8_layout_sr_a    # 16x16 shared -> 32x8 register (A)
shared_16x32_to_mma_32x16_layout_sr_a   # 16x32 shared -> 32x16 register (A)

# For loading B matrix from shared to registers
shared_16x8_to_mma_32x4_layout_sr_b     # 16x8 shared -> 32x4 register (B)
shared_16x16_to_mma_32x8_layout_sr_b    # 16x16 shared -> 32x8 register (B)
shared_16x32_to_mma_32x16_layout_sr_b   # 16x32 shared -> 32x16 register (B)
```

#### MMA-to-Shared Layouts (Register -> Destination)

```python
# For storing C matrix from registers to shared memory
mma_load_a_32x4_to_shared_16x8_layout
mma_load_b_32x4_to_shared_16x8_layout
mma_load_b_32x8_to_shared_16x16_layout
mma_load_a_32x16_to_shared_16x32_layout
mma_load_b_32x16_to_shared_16x32_layout
mma_load_a_32x8_to_shared_16x16_layout
```

### 10.2 Volta (SM70) Layout Functions

From `tilelang/intrinsics/mma_sm70_layout.py`:

```python
shared_16x4_to_mma_a_32x4_layout           # A matrix, 16x4 shared
shared_4x16_to_mma_b_32x4_layout           # B matrix, 4x16 shared (row-major)
shared_16x4_to_mma_b_32x4_layout_trans      # B matrix, 16x4 shared (transposed)
mma_32x8_to_shared_16x16_layout_fp32        # C output for fp32 accumulation
mma_32x8_to_shared_16x16_layout_fp16        # C output for fp16 accumulation
mma_load_a_32x4_to_shared_16x4_layout
mma_load_b_32x4_to_shared_16x4_layout_trans
mma_load_b_32x4_to_shared_4x16_layout
```

### 10.3 WGMMA Layout Functions

From `tilelang/intrinsics/wgmma_layout.py`:

WGMMA uses descriptor-based access to shared memory, so the layouts focus on
shared memory arrangement rather than register-to-shared mapping.

```python
# WGMMA shared memory layout helpers
# Used for creating swizzled shared memory layouts compatible with WGMMA
# descriptors
```

### 10.4 TCGEN05 Layout Functions

From `tilelang/intrinsics/tcgen05_layout.py`:

TCGEN05 operates with Tensor Memory (TMEM), so layouts map between shared
memory and TMEM.

```python
# TCGEN05 shared memory layout for A and B operands
# TCGEN05 TMEM layout for C accumulator
```

### 10.5 MFMA Layout Functions

From `tilelang/intrinsics/mfma_layout.py`:

MFMA layouts map between shared memory and the register layout expected by
AMD MFMA instructions.

```python
# Shared-to-local (register) layouts
shared_16x4_to_local_64x1_layout_A      # 16x4 shared -> 64x1 register
shared_4x16_to_local_64x1_layout_B      # 4x16 shared -> 64x1 register
shared_16x16_to_local_64x4_layout_A     # 16x16 shared -> 64x4 register
shared_16x16_to_local_64x4_layout_B
shared_16x32_to_local_64x8_layout_A
shared_16x32_to_local_64x8_layout_B
shared_16x64_to_local_64x16_layout_A
shared_16x64_to_local_64x16_layout_B
shared_32x32_to_local_64x16_layout_A
shared_32x32_to_local_64x16_layout_B

# Thread-ID to shared memory access layouts
thread_id_shared_access_64x1_to_16x4_layout_A
thread_id_shared_access_64x1_to_4x16_layout_B
thread_id_shared_access_64x4_to_16x16_layout_A
thread_id_shared_access_64x4_to_16x16_layout_B
thread_id_shared_access_64x8_to_16x32_layout_A
thread_id_shared_access_64x8_to_16x32_layout_B
thread_id_shared_access_64x16_to_16x64_layout_A
thread_id_shared_access_64x16_to_16x64_layout_B
thread_id_shared_access_64x16_to_32x32_layout_A
thread_id_shared_access_64x16_to_32x32_layout_B
```

Note: The "64" in these layout names refers to the 64-thread wavefront size
used by AMD GPUs.

### 10.6 WMMA Layout Functions

From `tilelang/intrinsics/wmma_layout.py`:

```python
# WMMA shared-to-register layouts for RDNA GPUs
# Uses 32-thread wavefronts (smaller than MFMA's 64)
```

---

## 11. Layout Inference System

### 11.1 Overview

The layout inference system automatically determines the optimal data layouts
for TileLang operations based on the target hardware and operation context.

### 11.2 Layout Inference Transform

From `src/transform/layout_inference.cc`:

The `LayoutInference` pass analyzes the TileLang IR and infers layouts for
all buffer operations. It uses the operation-specific `InferLayout` functions
registered for each backend.

### 11.3 Layout Inference for GEMM

The GEMM layout inference determines:

1. **A shared layout:** How matrix A should be arranged in shared memory
2. **B shared layout:** How matrix B should be arranged in shared memory
3. **C fragment layout:** How the accumulator should be arranged in registers

For different backends:

#### CUDA MMA Layout Inference

```python
# From gemm_mma.py
# A shared: swizzled layout based on m16n8k16 requirements
# B shared: swizzled layout based on m16n8k16 requirements
# C fragment: 32-thread fragment layout
```

#### CUDA WGMMA Layout Inference

```python
# From gemm_wgmma.py
# A shared: WGMMA-compatible swizzled layout
# B shared: WGMMA-compatible swizzled layout
# C fragment: warp-group accumulator layout
```

#### ROCm MFMA Layout Inference

```python
# From gemm_mfma.py
# A shared: LDS layout for MFMA loading
# B shared: LDS layout for MFMA loading
# C fragment: 64-thread wavefront fragment layout
```

### 11.4 Layout Inference for Copy

Copy operations infer layouts based on:

1. **Source scope:** Global, shared, or local
2. **Destination scope:** Shared or local
3. **Target hardware:** CUDA, ROCm, Metal, etc.
4. **Data type:** Affects vectorization and access patterns

```python
# SIMT layout inference (used by Metal, CPU, WebGPU)
layout_map = op.InferSIMTLayout(T, level)

# Tensor Core layout inference (used by CUDA, ROCm)
layout_map = op.InferTensorCoreLayout(T, level)
```

### 11.5 Layout Inference Testing

From `testing/python/transform/test_tilelang_transform_layout_inference.py`:

Tests verify that layout inference produces correct and optimal layouts for
various operation and target combinations.

---

## 12. Layout Visualization

### 12.1 Overview

TileLang provides layout visualization tools for debugging and understanding
data layouts.

### 12.2 Plot Layout Tool

From `tilelang/tools/plot_layout.py`:

```python
from tilelang.tools.plot_layout import plot_layout

# Visualize a layout
layout = Layout([16, 8], lambda i, j: i * 8 + j)
plot_layout(layout)
```

### 12.3 Layout Visualizer

From `tilelang/analysis/layout_visual.py`:

```python
from tilelang.analysis.layout_visual import visualize_layout

# Visualize how a layout maps indices
visualize_layout(layout, shape=[16, 8])
```

### 12.4 Fragment Debugging

The `__repr__` method of Layout and Fragment provides a debug string:

```python
layout = Layout([16, 8], lambda i, j: i * 8 + j)
print(layout)  # Shows shape and mapping
```

For Fragment:

```python
frag = Fragment(...)
print(frag)  # Shows thread dimension and index dimension
```

---

## 13. Custom Layout Creation

### 13.1 Custom Layout

```python
from tilelang.layout import Layout

# Create a custom layout with any mapping function
layout = Layout(
    shape=[M, N],
    forward_fn=lambda i, j: custom_mapping(i, j)
)
```

The `forward_fn` receives `len(shape)` IterVar arguments and should return
a PrimExpr or list of PrimExprs representing the output index(es).

### 13.2 Custom Fragment

```python
from tilelang.layout import Fragment

# Custom fragment with explicit thread and index mapping
def my_thread_fn(i, j, rep):
    return (i // 4) * warp_size + rep

def my_index_fn(i, j):
    return (i % 4) * N + j

frag = Fragment(
    shape=[16, 8],
    forward_thread_fn=my_thread_fn,
    forward_index_fn=my_index_fn,
    replicate=4
)
```

### 13.3 Composing Layouts

Layouts can be composed using the `repeat`, `expand`, and `reshape` operations:

```python
# Start with an atom layout
atom = Layout([4, 4], lambda i, j: i * 4 + j)

# Tile it along M
tiled_m = atom.repeat(dim=0, factor=4)  # shape [16, 4]

# Tile it along N
tiled_mn = tiled_m.repeat(dim=1, factor=2)  # shape [16, 8]

# Expand with a batch dimension
batched = tiled_mn.expand([32])  # shape [32, 16, 8]
```

### 13.4 Custom Swizzle Layout

```python
# Custom XOR swizzle
def custom_swizzle(shape, bank_bits):
    def forward(i, j):
        row = i
        col = j ^ ((row >> 0) & ((1 << bank_bits) - 1))
        return row * shape[1] + col
    return Layout(shape, forward)

# Example: 4-bit bank swizzle (16 banks)
layout = custom_swizzle([64, 32], bank_bits=4)
```

---

## 14. Layout and Backend Interaction

### 14.1 Layout Selection by Backend

Each backend selects layouts differently:

| Backend | A Layout | B Layout | C Layout |
|---|---|---|---|
| CUDA MMA | Swizzled + ldmatrix | Swizzled + ldmatrix | 32-thread fragment |
| CUDA WGMMA | WGMMA swizzled | WGMMA swizzled | Warp-group registers |
| CUDA TCGEN05 | TCGEN05 swizzled | TCGEN05 swizzled | TMEM |
| ROCm MFMA | LDS + vec load | LDS + vec load | 64-thread fragment |
| ROCm WMMA | LDS + vec load | LDS + vec load | 32-thread fragment |
| Metal | Row-major | Row-major | SIMT |
| CPU | Row-major | Row-major | Row-major |
| WebGPU | Row-major | Row-major | SIMT |

### 14.2 Layout Propagation

Layouts propagate through the TileLang IR:

1. **Input buffers:** Layout determined by external data arrangement
2. **Shared memory buffers:** Layout inferred from the first consumer operation
3. **Fragment buffers:** Layout determined by the producing GEMM operation
4. **Output buffers:** Layout determined by the store operation

### 14.3 Layout Annotation

Users can override automatic layout inference with explicit annotations:

```python
from tilelang.layout import make_swizzled_layout, make_linear_layout

A_shared = T.alloc_shared((128, 32), "float16")
custom_layout = make_swizzled_layout(A_shared, k_major=True)
T.annotate_layout({A_shared: custom_layout})
```

---

## 15. Advanced Layout Topics

### 15.1 Index Map Integration

Layout uses TVM's `IndexMap` internally for index transformations:

```python
def map_forward_index(self, indices: list[PrimExpr]) -> PrimExpr:
    forward_vars = self.get_forward_vars()
    forward_indexes = self.index
    index_map = IndexMap(
        initial_indices=forward_vars,
        final_indices=forward_indexes,
        inverse_index_map=None,
    )
    return index_map.map_indices(indices)
```

### 15.2 Layout for Sparse GEMM

Sparse GEMM uses a metadata layout that encodes the 2:4 sparsity pattern:

```python
# The metadata layout for SM90 sparse WGMMA
# Each metadata element encodes which 2 of 4 elements are non-zero
metadata_layout = make_cutlass_metadata_layout(
    metadata_buffer,
    mma_dtype="float16",
    block_k=64,
)
```

### 15.3 Layout for Block-Scaled GEMM

Block-scaled GEMM (SM100+) uses separate scale factor layouts:

```python
# Scale factors are stored alongside the main data
# with a specific layout matching the TCGEN05 instruction requirements
```

### 15.4 Layout for FP4/FP6 Sub-Byte Types

Sub-byte data types require special layout handling because multiple elements
are packed into a single byte:

```python
# FP4: 2 elements per byte
# FP6: 4 elements per 3 bytes (or padded to 1 element per byte)
# The layout must account for the packing factor
```

---

## 16. C++ Layout Implementation

### 16.1 Source Files

| File | Purpose |
|---|---|
| `src/layout/layout.cc` | Layout C++ implementation |
| `src/layout/layout.h` | Layout header |
| `src/layout/gemm_layouts.cc` | GEMM-specific layout functions |
| `src/layout/tcgen05_layout.cc` | TCGEN05 layout functions |
| `src/layout/tcgen05_layout.h` | TCGEN05 layout header |
| `src/layout/utils.cc` | Layout utility functions |
| `src/layout/utils.h` | Layout utility header |

### 16.2 FFI Registration

Layout functions are exposed to Python via TVM's FFI:

```cpp
// From src/layout/layout.cc
TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef()
        .def("tl.Layout", LayoutConstructor)
        .def("tl.Layout_index", LayoutIndex)
        .def("tl.Layout_input_shape", LayoutInputShape)
        .def("tl.Layout_output_shape", LayoutOutputShape)
        .def("tl.Layout_forward_vars", LayoutForwardVars)
        .def("tl.Layout_repeat", LayoutRepeat)
        .def("tl.Layout_expand", LayoutExpand)
        .def("tl.Layout_inverse", LayoutInverse)
        .def("tl.Layout_reshape", LayoutReshape)
        .def("tl.Layout_is_equal", LayoutIsEqual)
        .def("tl.make_swizzled_layout", MakeSwizzledLayout)
        .def("tl.make_volta_swizzled_layout", MakeVoltaSwizzledLayout)
        .def("tl.make_wgmma_swizzled_layout", MakeWgmmaSwizzledLayout)
        .def("tl.make_tcgen05mma_swizzled_layout", MakeTcgen05SwizzledLayout)
        .def("tl.make_full_bank_swizzled_layout", MakeFullBankSwizzledLayout)
        .def("tl.make_half_bank_swizzled_layout", MakeHalfBankSwizzledLayout)
        .def("tl.make_quarter_bank_swizzled_layout", MakeQuarterBankSwizzledLayout)
        .def("tl.make_linear_layout", MakeLinearLayout)
        .def("tl.make_gemm_fragment_8x8", MakeGemmFragment8x8)
        .def("tl.make_gemm_fragment_8x8_transposed", MakeGemmFragment8x8Transposed)
        .def("tl.make_fully_replicated_layout_fragment", MakeFullyReplicatedFragment);
}
```

---

## 17. Complete Layout Usage Example

### 17.1 GEMM with Explicit Layouts

```python
import tilelang as tl
from tilelang import language as T
from tilelang.layout import (
    make_swizzled_layout,
    make_gemm_fragment_8x8,
    make_linear_layout,
)

@T.prim_func
def matmul_with_layouts(
    A: T.Buffer((1024, 512), "float16"),
    B: T.Buffer((512, 256), "float16"),
    C: T.Buffer((1024, 256), "float32"),
):
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    with T.Kernel(2, 8, threads=128) as (bx, by):
        # Shared memory with explicit swizzled layouts
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

        # Annotate with swizzled layouts for bank conflict avoidance
        T.annotate_layout({
            A_shared: make_swizzled_layout(A_shared, k_major=True),
            B_shared: make_swizzled_layout(B_shared, k_major=True),
        })

        # Accumulator in registers
        C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        T.clear(C_frag)

        for ko in range(512 // BLOCK_K):
            T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_frag)

        T.copy(C_frag, C[by * BLOCK_M, bx * BLOCK_N])
```

### 17.2 Inspecting Layouts

```python
from tilelang.layout import Layout, Fragment, make_swizzled_layout

# Create and inspect a layout
layout = Layout([16, 8], lambda i, j: i * 8 + j)
print(f"Input shape: {layout.get_input_shape()}")
print(f"Output shape: {layout.get_output_shape()}")
print(f"Forward index: {layout.index}")

# Create and inspect a fragment
frag = Fragment(
    [8, 8],
    forward_fn=lambda i, j, rep: (i * 4 + rep, j),
    replicate=4,
)
print(f"Thread: {frag.thread}")
print(f"Thread size: {frag.get_thread_size()}")
print(f"Input shape: {frag.get_input_shape()}")
```
