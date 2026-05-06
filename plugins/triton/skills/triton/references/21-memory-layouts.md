# Chapter 21: Memory Layouts

Memory layouts describe how data is distributed across GPU threads and stored in different memory spaces.

## LinearLayout

The fundamental layout representation. Maps multi-dimensional indices to thread/memory locations.

```cpp
class LinearLayout {
    // In dimensions: thread/block indices
    // Out dimensions: memory/register locations
    // Represented as a linear transformation
};
```

### Operations

| Operation | Description |
|-----------|-------------|
| `compose(other)` | Compose two layouts |
| `invert()` | Invert layout (if bijective) |
| `pseudoinvert()` | Pseudo-inverse |
| `is_surjective()` | Every output reachable |
| `is_injective()` | Every output unique |
| `is_invertible()` | Both injective and surjective |

### Factory Methods

```python
# 1D identity layout
layout = LinearLayout.identity_1d(size, dim_name)

# 1D strided layout
layout = LinearLayout.strided_1d(size, stride, dim_name)

# From basis vectors
layout = LinearLayout.from_bases(in_dims, bases)
```

## Distributed Layouts

Describe how data is partitioned across threads in a warp/CTA.

### BlockedLayout
Standard blocked partitioning:

```python
# Each thread block processes a tile of data
# shape per CTA, order, and size per warp
layout = BlockedLayout([64, 64], order=[1, 0], size_per_warp=[16, 16])
```

### CoalescedLayout
Optimized for coalesced memory access:

```python
# Memory accesses are coalesced across threads
layout = CoalescedLayout(shape, order)
```

### DotOperandLayout
Optimized layout for dot product operands:

```python
# Matches MMA instruction input requirements
layout = DotOperandLayout(parent, op_idx, k_width)
```

### NVMMADistributedLayout
NVIDIA-specific MMA layout:

```python
# Maps to NVVM MMA instruction layout
layout = NVMMADistributedLayout(shape, instr_shape, wpt)
```

### AMDMFMALayout / AMDWMMALayout
AMD-specific matrix core layouts:

```python
# AMD MFMA instruction layout
layout = AMDMFMALayout(shape, instr_shape)

# AMD WMMA instruction layout
layout = AMDWMMALayout(shape, instr_shape)
```

## Shared Memory Layouts

Describe data arrangement in shared memory.

### SwizzledSharedLayout
Bank-conflict-free shared memory:

```python
# Swizzle pattern to avoid bank conflicts
layout = SwizzledSharedLayout(shape, order, swizzle)
```

### NVMMASharedLayout
Shared memory layout for NVIDIA MMA:

```python
# Optimized for NVVM MMA loads
layout = NVMMASharedLayout(swizzle, element_bitwidth)
```

### PaddedSharedLayout
Padded to avoid bank conflicts:

```python
# Padding to eliminate bank conflicts
layout = PaddedSharedLayout(shape, padding)
```

### SharedLinearLayout
Generic linear layout for shared memory.

## Tensor Memory Layouts (Blackwell)

### TensorMemoryLayout
Layout for tensor memory on Blackwell GPUs:

```python
layout = TensorMemoryLayout(shape, element_bitwidth)
```

### TensorMemoryScalesLayout
Layout for scale factors in tensor memory.

## Layout Conversion

```python
# Convert between layouts
new_tensor = tl.convert_layout(tensor, new_layout)

# In Gluon
from triton.experimental.gluon.language import convert_layout
result = convert_layout(tensor, target_layout)
```

## Layout Inference

Triton automatically infers layouts through the compilation pipeline:

1. **TTIR:** No layout information
2. **TritonToTritonGPU:** Assigns initial blocked layouts
3. **AccelerateMatmul:** Changes dot operand layouts to MMA layouts
4. **OptimizeDotOperands:** Optimizes layouts for dot operands
5. **RemoveLayoutConversions:** Eliminates unnecessary layout changes

## Shared Memory Allocation

```python
# Allocate shared memory in Gluon
from triton.experimental.gluon.language import allocate_shared_memory

buf = allocate_shared_memory(shape, dtype, layout=SwizzledSharedLayout(...))
```

### Allocation Strategy

1. **Analyze** shared memory usage per operation
2. **Assign** offsets using first-fit allocation
3. **Insert** barriers for synchronization
4. **Optimize** by reusing memory across non-overlapping operations

## Bank Conflicts

Shared memory has 32 banks (NVIDIA) or variable banks (AMD). Bank conflicts occur when multiple threads access the same bank simultaneously.

### Avoiding Bank Conflicts

1. **Padding:** Add extra bytes to each row
2. **Swizzling:** XOR-based address transformation
3. **Layout optimization:** Choose conflict-free layouts

```python
# Check for bank conflicts (Gluon)
conflicts = tl.bank_conflicts(layout)
```
