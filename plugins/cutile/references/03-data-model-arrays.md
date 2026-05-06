# Data Model: Global Arrays in cuTile

This chapter provides a comprehensive exploration of cuTile's global array data model—the fundamental abstraction for GPU memory management and data transfer. Understanding arrays deeply is essential for writing efficient, correct cuTile kernels.

## What Are Global Arrays?

Global arrays in cuTile represent contiguous blocks of GPU memory that store multidimensional data. They serve as the primary interface between host (CPU) and device (GPU) code, enabling efficient data transfer and computation.

### Key Characteristics

**Host Allocation:**
Global arrays are allocated by the host code (CPU) using companion libraries like CuPy, PyTorch, or NumPy. The host controls:
- Memory allocation and deallocation
- Initial data population
- Result extraction after computation
- Lifetime management

```python
# Host allocates array using CuPy
import cupy as cp
device_array = cp.zeros((1024, 1024), dtype=cp.float32)

# Array is passed to kernel as argument
@ct.kernel
def process_array(data: ct.Array[float32]):
    # Kernel reads/writes array but doesn't own it
    ...
```

**GPU Storage:**
Once allocated, the array data resides in GPU global memory:
- Large capacity (gigabytes on modern GPUs)
- High latency (hundreds of cycles)
- High bandwidth (hundreds of GB/s)
- Accessible by all thread blocks
- Persistent across kernel launches

**Kernel Arguments:**
Arrays are passed to kernels as read-only or read-write arguments:
- Kernels cannot allocate new global arrays
- Kernels cannot free global arrays
- Kernels operate on pre-allocated arrays
- Array lifetimes exceed kernel execution

### Array vs. Tile Distinction

Understanding the difference between arrays and tiles is crucial:

| Aspect | Array | Tile |
|--------|-------|------|
| **Location** | Global memory | Shared memory/registers |
| **Lifetime** | Kernel execution scope | Kernel execution scope |
| **Allocation** | Host (CPU) | Compiler-generated |
| **Access** | Load/store operations | Direct access |
| **Mutability** | Read-write | Immutable |
| **Speed** | Slow (high latency) | Fast (low latency) |
| **Capacity** | Large (GB) | Small (KB) |
| **Visibility** | All blocks | Single block |

Arrays serve as the data source and destination, while tiles are working copies used for computation.

## Array Shape and Dimensions

The shape of a cuTile array defines its structure and accessibility.

### Shape Representation

Array shape is a tuple of 32-bit integers:
```python
import cuda_tile as ct

# 1D array with 1024 elements
shape_1d = (1024,)

# 2D array with 64 rows, 128 columns
shape_2d = (64, 128)

# 3D array for volume data
shape_3d = (256, 256, 128)

# 4D array for batch of images [batch, channels, height, width]
shape_4d = (32, 3, 224, 224)
```

### Shape Properties

**Number of Dimensions:**
```python
@ct.kernel
def get_ndim(array: ct.Array[float32]):
    # Array.ndim returns number of dimensions
    rank = array.ndim  # e.g., 2 for matrix
    ...
```

**Total Elements:**
```python
@ct.kernel
def get_total_elements(array: ct.Array[float32]):
    # Product of all dimensions
    total = 1
    for i in range(array.ndim):
        total = total * array.shape[i]
    ...
```

**Shape Queries:**
```python
# Runtime shape query (non-constant)
shape = array.shape  # Returns tuple of int32
rows = shape[0]
cols = shape[1]

# Cannot use shape for compile-time constants
# This is a limitation: shapes are runtime values
```

### Dimension Limits

cuTile uses 32-bit integers for dimensions:
- **Maximum per dimension**: 2,147,483,647 elements (2³¹ - 1)
- **Practical limit**: GPU memory capacity
- **Recommended**: Keep dimensions aligned to power-of-2 for efficiency

```python
# Valid shapes
shape_valid = (1024, 1024)  # 1M elements, ~4MB for float32
shape_large = (65536, 65536)  # 4B elements, ~16GB for float32

# Invalid shapes (exceed int32)
shape_invalid = (3000000000,)  # Exceeds int32 max
```

### Shape Semantics

Shape is **runtime information**, not compile-time constant:
```python
@ct.kernel
def flexible_kernel(array: ct.Array[float32]):
    # Cannot use shape in compile-time contexts
    # TILE_SIZE = array.shape[0]  # ERROR: Not a compile-time constant
    
    # Can use shape in runtime computations
    total_size = 1
    for i in range(array.ndim):
        total_size = total_size * array.shape[i]
    
    # Can use shape for bounds checking
    if ct.bid(0) < array.shape[0]:
        ...
```

This limitation exists because:
1. Array shape is determined at runtime (by host allocation)
2. Kernel compilation happens before runtime
3. Compiler cannot know shape values during compilation

## Strided Memory Layout

cuTile arrays use strided memory layout for flexible indexing and efficient subarray operations.

### Memory Layout Fundamentals

GPU memory is fundamentally linear—a single address space. Multimensional arrays are mapped to this linear space using strides.

**Base Address:**
The starting memory address of the array's data.

**Element Size:**
Bytes per element based on data type:
- `float32`, `int32`: 4 bytes
- `float64`, `int64`: 8 bytes
- `float16`, `int16`: 2 bytes
- `int8`, `uint8`: 1 byte

**Strides:**
Number of elements to skip to move to the next position along each dimension.

### Address Calculation Formula

The memory address of element at indices `(i, j, k)` is:

```
address = base_addr + element_size * (stride[0] * i + stride[1] * j + stride[2] * k)
```

More generally, for N-dimensional array with indices `(i₀, i₁, ..., iₙ₋₁)`:

```
address = base_addr + element_size * Σ(stride[d] * i_d)
```

### Layout Examples

**Row-Major (C-Style) Layout:**
```python
# 2D array: 3 rows × 4 columns
array = cp.array([
    [1, 2, 3, 4],     # Row 0
    [5, 6, 7, 8],     # Row 1
    [9, 10, 11, 12]   # Row 2
], dtype=cp.float32)

# Memory layout (linear):
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Strides: (4, 1)
# - stride[0] = 4: Skip 4 elements to next row
# - stride[1] = 1: Adjacent columns are contiguous

# Address calculation for element [2, 3] (value 11):
# address = base + 4 * (4 * 2 + 1 * 3)
#         = base + 4 * 11
#         = base + 44 (11th element)
```

**Column-Major (Fortran-Style) Layout:**
```python
# Transposed array
array_T = array.T  # 4 × 3 array

# Memory layout (same linear storage):
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Strides: (1, 4)
# - stride[0] = 1: Adjacent rows are contiguous
# - stride[1] = 4: Skip 4 elements to next column

# Address calculation for element [3, 2] (value 11):
# address = base + 4 * (1 * 3 + 4 * 2)
#         = base + 4 * 11
#         = base + 44 (same location!)
```

**3D Array Layout:**
```python
# 3D array: 2 × 3 × 4
array_3d = cp.random.randn(2, 3, 4).astype(cp.float32)

# Shape: (2, 3, 4)
# Strides: (12, 4, 1) for C-style
#          - 12 elements between "planes"
#          - 4 elements between rows in same plane
#          - 1 element between adjacent columns

# Element [1, 2, 3]:
# address = base + 4 * (12 * 1 + 4 * 2 + 1 * 3)
#         = base + 4 * (12 + 8 + 3)
#         = base + 4 * 23
#         = base + 92
```

### Stride Implications

**Memory Access Patterns:**
```python
# GOOD: Sequential access (coalesced)
for j in range(cols):
    val = array[0, j]  # Contiguous memory

# LESS EFFICIENT: Strided access
for i in range(rows):
    val = array[i, 0]  # Strided by cols
```

**Performance Considerations:**
- Contiguous access patterns maximize memory bandwidth
- Strided access can reduce efficiency
- Cache utilization depends on access pattern
- Compiler attempts to optimize based on stride information

## Array Creation and Initialization

Arrays are created on the host using various libraries and then passed to cuTile kernels.

### CuPy Array Creation

```python
import cupy as cp

# Zero-initialized array
zeros = cp.zeros((1024, 1024), dtype=cp.float32)

# One-initialized array
ones = cp.ones((256, 256), dtype=cp.float32)

# Random values
random_floats = cp.random.randn(1000, 1000).astype(cp.float32)
random_ints = cp.random.randint(0, 100, size=(500, 500)).astype(cp.int32)

# From existing data
data = [1.0, 2.0, 3.0, 4.0]
from_list = cp.array(data, dtype=cp.float32)

# Arange/linpace
sequence = cp.arange(0, 100, dtype=cp.float32)
spaced = cp.linspace(0, 10, 100, dtype=cp.float32)
```

### PyTorch Tensor Creation

```python
import torch

# Create tensor on GPU
tensor = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')

# From NumPy
import numpy as np
numpy_array = np.random.randn(100, 100).astype(np.float32)
torch_tensor = torch.from_numpy(numpy_array).cuda()

# Zeros/ones
zeros = torch.zeros((256, 256), dtype=torch.float32, device='cuda')
ones = torch.ones((256, 256), dtype=torch.float32, device='cuda')
```

### NumPy Array Creation

```python
import numpy as np

# Created on CPU, will transfer to GPU
numpy_array = np.random.randn(512, 512).astype(np.float32)

# Can be converted to CuPy for GPU
cupy_array = cp.asarray(numpy_array)
```

### Array Type Conversions

```python
# Float32 to float16 (half precision)
float32_array = cp.random.randn(1000, 1000).astype(cp.float32)
float16_array = float32_array.astype(cp.float16)

# Int32 to float32
int_array = cp.randint(0, 100, size=(100, 100)).astype(cp.int32)
float_array = int_array.astype(cp.float32)

# Complex numbers
complex_array = cp.random.randn(100, 100) + 1j * cp.random.randn(100, 100)
complex_array = complex_array.astype(cp.complex64)
```

## Array Views vs. Copies

Understanding the distinction between views and copies is critical for performance and correctness.

### Views Share Memory

A view is a new array object that references the same underlying memory:

```python
import cupy as cp

# Create original array
original = cp.arange(10, dtype=cp.float32)
# Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create view
view = original[2:7]  # Elements [2, 3, 4, 5, 6]
# No memory copy! View points to same memory

# Modify view
view[0] = 999.0

# Original is also modified
print(original)  # [0, 1, 999, 3, 4, 5, 6, 7, 8, 9]

# View sees change too
print(view)  # [999, 3, 4, 5, 6]
```

**View Characteristics:**
- No memory allocation
- No data copying
- Instant creation
- Changes affect original
- Useful for subarray operations

### Copies Create Independent Data

A copy allocates new memory and duplicates data:

```python
# Create copy
copy_array = original.copy()

# Modify copy
copy_array[0] = 777.0

# Original unaffected
print(original)  # [0, 1, 999, 3, 4, 5, 6, 7, 8, 9]
print(copy_array)  # [777, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Copy Characteristics:**
- Allocates new memory
- Copies all data
- Slower for large arrays
- Independent modification
- Useful for independent data

### View Creation Methods

**Slicing:**
```python
array = cp.arange(100, dtype=cp.float32).reshape(10, 10)

# Row slice (view)
row_view = array[3:7, :]  # Rows 3-6, all columns

# Column slice (view)
col_view = array[:, 2:5]  # All rows, columns 2-4

# Combined slice
sub_view = array[2:8, 3:9]  # Both dimensions sliced
```

**Transposition:**
```python
# Transpose creates view with swapped strides
transposed = array.T  # Shape (10, 10), strides swapped
```

**Reshape (when possible):**
```python
# Reshape creates view if memory layout permits
contiguous = cp.arange(100, dtype=cp.float32)
reshaped = contiguous.reshape(10, 10)  # Often a view
```

## Array Slicing Operations

cuTile provides the `Array.slice()` method for creating views of portions of arrays.

### Basic Slicing

```python
@ct.kernel
def slice_example(data: ct.Array[float32]):
    # Slice along axis 0 from index 10 to 20
    subarray = data.slice(axis=0, start=10, stop=20)
    
    # subarray is a view sharing memory with data
    # Shape: (10, ...) if original was (N, ...)
```

### Multi-dimensional Slicing

```python
@ct.kernel
def multi_slice(matrix: ct.Array[float32]):
    # Original shape: (1024, 1024)
    
    # Slice rows 100-200
    row_slice = matrix.slice(axis=0, start=100, stop=200)
    # Shape: (100, 1024)
    
    # Slice columns 50-150
    col_slice = matrix.slice(axis=1, start=50, stop=150)
    # Shape: (1024, 100)
    
    # Combined slicing (chained)
    submatrix = matrix.slice(axis=0, start=100, stop=200).slice(axis=1, start=50, stop=150)
    # Shape: (100, 100)
```

### Slicing Semantics

**Stride Preservation:**
```python
# Slice maintains original stride pattern
# Original: shape (1000, 1000), strides (1000, 1)
sliced = original.slice(axis=0, start=100, stop=200)
# Sliced: shape (100, 1000), strides (1000, 1)
```

**Memory Sharing:**
```python
# Slices are views, not copies
original = cp.arange(1000, dtype=cp.float32)
slice_view = original.slice(0, 100, 200)  # Elements 100-199

# Modifying slice affects original
slice_view[0] = 999.0
# original[100] is now 999.0
```

**Performance Benefits:**
- No memory allocation
- No data copying
- Efficient for subarray operations
- Useful for sliding window operations

## Tiled Views

Tiled views provide a higher-level abstraction for working with array tiles.

### Creating Tiled Views

```python
@ct.kernel
def tiled_computation(matrix: ct.Array[float32]):
    # Create tiled view with 64×64 tiles
    tiled_view = matrix.tiled_view(tile_shape=(64, 64))
    
    # Iterate over tiles
    for tile in tiled_view:
        # Process each tile
        # tile has shape (64, 64) or smaller at edges
        ...
```

### Tile Shape and Padding

**Regular Tiles:**
```python
# Array: (1024, 1024)
# Tile shape: (64, 64)
# Result: 16 × 16 grid of perfect 64×64 tiles

tiled = array.tiled_view(tile_shape=(64, 64))
```

**Edge Handling:**
```python
# Array: (1000, 1000)
# Tile shape: (64, 64)
# Result: 
#   - Most tiles: 64×64
#   - Edge tiles: smaller (e.g., 40×64, 64×40, 40×40)

# Padding mode handling
tiled_padded = array.tiled_view(
    tile_shape=(64, 64),
    padding_mode='constant'  # or 'edge', 'reflect', etc.
)
```

### Tiled View Iteration

```python
@ct.kernel
def process_tiles(data: ct.Array[float32], output: ct.Array[float32]):
    # Create tiled view
    tiles = data.tiled_view(tile_shape=(128, 128))
    
    # Process each tile
    tile_idx = 0
    for tile in tiles:
        # tile is a Tile object (not Array)
        # Compute on tile
        result_tile = process_tile(tile)
        
        # Store result
        # (need to calculate output position)
        store_tile_result(output, tile_idx, result_tile)
        tile_idx = tile_idx + 1
```

## DLPack and CUDA Array Interface

cuTile arrays can interoperate with other GPU frameworks through standardized protocols.

### CUDA Array Interface

The CUDA Array Interface allows zero-copy data sharing between frameworks:

```python
import cupy as cp
import torch

# Create CuPy array
cupy_array = cp.random.randn(1024, 1024).astype(cp.float32)

# Pass to PyTorch without copying
torch_tensor = torch.as_tensor(cupy_array, device='cuda')

# Both reference same GPU memory!
torch_tensor[0, 0] = 999.0
print(cupy_array[0, 0])  # 999.0 (shared memory)
```

### DLPack Protocol

DLPack provides another interoperability standard:

```python
from cupy import DLPackTensor

# Export to DLPack
dlpack_tensor = DLPackTensor(cupy_array)

# Import in other framework
# (depends on receiving framework's DLPack support)
```

### Interoperability Example

```python
import cuda_tile as ct
import cupy as cp
import torch

def interoperability_example():
    # Create data in PyTorch
    torch_tensor = torch.randn(1000, 1000, device='cuda')
    
    # Convert to CuPy (zero-copy)
    cupy_array = cp.fromDlpack(torch.__utils__.to_dlpack(torch_tensor))
    
    # Use in cuTile kernel
    @ct.kernel
    def process_tensor(data: ct.Array[float32], output: ct.Array[float32]):
        i = ct.bid(0)
        if i < 1000000:
            val = ct.load(data, (i,))
            ct.store(output, (i,), val * 2.0)
    
    output = cp.zeros_like(cupy_array)
    stream = ct.Stream()
    ct.launch(stream, (ct.cdiv(1000000, 1),), 
              process_tensor, [cupy_array, output])
    stream.synchronize()
    
    # Convert back to PyTorch (zero-copy)
    torch_output = torch.as_tensor(output, device='cuda')
    
    return torch_output
```

## Array Sources and Conversion

cuTile can work with arrays from multiple sources.

### CuPy Arrays (Primary)

```python
import cupy as cp

# Direct CuPy usage
array = cp.random.randn(1024, 1024).astype(cp.float32)

# Pass to cuTile kernel
@ct.kernel
def use_cupy(data: ct.Array[float32]):
    ...
```

### PyTorch Tensors

```python
import torch

# PyTorch tensor on GPU
tensor = torch.randn(512, 512, dtype=torch.float32, device='cuda')

# Use directly (supports CUDA Array Interface)
@ct.kernel
def use_torch(data: ct.Array[float32]):
    # PyTorch tensor automatically converts
    ...
```

### NumPy Arrays

```python
import numpy as np

# NumPy array (CPU memory)
numpy_array = np.random.randn(256, 256).astype(np.float32)

# Must transfer to GPU
cupy_array = cp.asarray(numpy_array)

# Then use in kernel
@ct.kernel
def use_numpy(data: ct.Array[float32]):
    # Receives GPU copy of NumPy data
    ...
```

### Conversion Best Practices

**Minimize CPU-GPU Transfers:**
```python
# AVOID: Multiple transfers
for i in range(100):
    cpu_data = get_cpu_data()
    gpu_data = cp.asarray(cpu_data)  # Transfer each iteration
    process_gpu(gpu_data)

# BETTER: Single transfer
cpu_data = get_cpu_data()
gpu_data = cp.asarray(cpu_data)  # Transfer once
for i in range(100):
    process_gpu(gpu_data)
```

**Use Appropriate Types:**
```python
# Match kernel expectations
@ct.kernel
def float_kernel(data: ct.Array[float32]):
    ...

# Correct type
array = cp.random.randn(1000).astype(cp.float32)

# Wrong type (will cause errors)
wrong_array = cp.random.randn(1000).astype(cp.float64)
```

## No-Aliasing Rule

cuTile enforces a strict no-aliasing rule for kernel array arguments.

### What is Aliasing?

Aliasing occurs when multiple array arguments refer to overlapping memory regions:

```python
# EXAMPLE OF ALIASING (FORBIDDEN)
array = cp.arange(100, dtype=cp.float32)

# WRONG: a and b refer to same memory
@ct.kernel
def bad_kernel(a: ct.Array[float32], b: ct.Array[float32], c: ct.Array[float32]):
    # Undefined behavior if a and b overlap!
    i = ct.bid(0)
    a_val = ct.load(a, (i,))
    b_val = ct.load(b, (i,))
    ct.store(c, (i,), a_val + b_val)

# This call creates aliasing
bad_kernel(array, array, output)  # a and b are same array!
```

### Why No Aliasing?

**Compiler Optimizations:**
- Compiler assumes independent arrays
- Enables reordering of loads/stores
- Allows common subexpression elimination
- Facilitates memory access optimization

**Correctness:**
- Prevents read-after-write hazards
- Avoids undefined behavior
- Ensures deterministic results
- Maintains data consistency

### Safe Usage

**Independent Arrays:**
```python
# CORRECT: Separate arrays
a = cp.random.randn(1000).astype(cp.float32)
b = cp.random.randn(1000).astype(cp.float32)
c = cp.zeros(1000, dtype=cp.float32)

safe_kernel(a, b, c)  # No overlapping memory
```

**Temporary Copies:**
```python
# If you need in-place operation, make explicit copy
original = cp.random.randn(1000).astype(cp.float32)
working_copy = original.copy()  # Explicit copy

# Now safe to use both
safe_kernel(original, working_copy, output)
```

**Checking for Aliasing:**
```python
def safe_launch(kernel, args):
    """Check for aliased arrays before kernel launch."""
    import numpy as np
    
    # Get base pointers for all array arguments
    base_ptrs = []
    for arg in args:
        if hasattr(arg, 'ptr') or hasattr(arg, '__cuda_array_interface__'):
            # Get base memory address
            if hasattr(arg, 'ptr'):
                ptr = arg.ptr
            else:
                ptr = arg.__cuda_array_interface__['data'][0]
            base_ptrs.append(ptr)
    
    # Check for duplicates
    if len(base_ptrs) != len(set(base_ptrs)):
        raise ValueError("Aliased array arguments detected!")
    
    # Safe to launch
    ct.launch(stream, grid, kernel, args)
```

## Array Shape Querying

Querying array shapes at runtime is essential for flexible, reusable kernels.

### Shape Access

```python
@ct.kernel
def query_shape(data: ct.Array[float32]):
    # Get shape tuple
    shape = data.shape  # Returns tuple of int32
    
    # Access individual dimensions
    dim0 = shape[0]
    dim1 = shape[1]
    
    # Total elements
    total_elements = 1
    for i in range(data.ndim):
        total_elements = total_elements * shape[i]
```

### Runtime Shape vs. Compile-Time Constants

```python
@ct.kernel
def shape_limitations(data: ct.Array[float32]):
    # Shape is RUNTIME value, not compile-time constant
    shape = data.shape  # OK: Runtime query
    
    # Cannot use shape for compile-time decisions
    # TILE_SIZE = shape[0]  # ERROR: Not compile-time constant
    
    # Cannot use shape for static array sizing
    # float local_array[shape[0]]  # ERROR: Size must be constant
    
    # CAN use shape for runtime decisions
    if ct.bid(0) < shape[0]:
        # Dynamic bounds checking
        ...
```

### Shape-Based Computation

```python
@ct.kernel
def flexible_matrix_multiply(
    a: ct.Array[float32],
    b: ct.Array[float32],
    c: ct.Array[float32]
):
    # Get dimensions
    m = a.shape[0]  # Rows in A
    k = a.shape[1]  # Cols in A / Rows in B
    n = b.shape[1]  # Cols in B
    
    # Compute matrix multiplication
    row = ct.bid(0)
    col = ct.bid(1)
    
    if row < m and col < n:
        # Compute dot product
        sum_val = 0.0
        for i in range(k):
            a_val = ct.load(a, (row, i))
            b_val = ct.load(b, (i, col))
            sum_val = sum_val + a_val * b_val
        
        ct.store(c, (row, col), sum_val)
```

## Passing Arrays to Kernels

Arrays are passed to kernels as arguments with type annotations.

### Type Annotations

```python
@ct.kernel
def typed_kernel(
    float_array: ct.Array[float32],
    int_array: ct.Array[int32],
    output: ct.Array[float32]
):
    # Compiler knows types for optimization
    f_val = ct.load(float_array, (0,))
    i_val = ct.load(int_array, (0,))
    result = f_val + float(i_val)
    ct.store(output, (0,), result)
```

### Array Argument Rules

**Must be GPU Arrays:**
```python
# Correct: GPU array
gpu_array = cp.random.randn(1000).astype(cp.float32)
kernel(gpu_array)  # OK

# Incorrect: CPU array
cpu_array = np.random.randn(1000).astype(np.float32)
kernel(cpu_array)  # ERROR: Not on GPU
```

**Matching Types:**
```python
# Kernel declaration
@ct.kernel
def my_kernel(data: ct.Array[float32]):
    ...

# Must match type
float32_array = cp.random.randn(1000).astype(cp.float32)
my_kernel(float32_array)  # OK

float64_array = cp.random.randn(1000).astype(cp.float64)
my_kernel(float64_array)  # ERROR: Type mismatch
```

**Shape Compatibility:**
```python
# Kernels don't enforce shape compatibility
# But incorrect shapes cause runtime errors

small = cp.zeros((10, 10), dtype=cp.float32)
large = cp.zeros((100, 100), dtype=cp.float32)

# May access out of bounds
@ct.kernel
def unsafe_add(a: ct.Array[float32], b: ct.Array[float32]):
    i = ct.bid(0)
    # What if a and b have different sizes?
    val_a = ct.load(a, (i,))  # May be out of bounds for smaller array
    val_b = ct.load(b, (i,))

# Always ensure shapes match before launching
assert a.shape == b.shape, "Shape mismatch!"
```

### Multi-Array Kernels

```python
@ct.kernel
def multi_array_operation(
    input1: ct.Array[float32],
    input2: ct.Array[float32],
    weights: ct.Array[float32],
    output: ct.Array[float32],
    temp: ct.Array[float32]  # Working array
):
    # Complex operation using multiple arrays
    i = ct.bid(0)
    
    if i < input1.shape[0]:
        # Load from multiple inputs
        val1 = ct.load(input1, (i,))
        val2 = ct.load(input2, (i,))
        weight = ct.load(weights, (i,))
        
        # Compute
        temp_val = (val1 + val2) * weight
        ct.store(temp, (i,), temp_val)
        
        # Final computation
        result = temp_val * 0.5
        ct.store(output, (i,), result)
```

## Reading and Writing Array Elements

Element access in cuTile kernels uses explicit load/store operations.

### Load Operations

**Scalar Load:**
```python
@ct.kernel
def load_element(data: ct.Array[float32]):
    # Load single element
    index = ct.bid(0)
    value = ct.load(data, (index,))
    
    # Use loaded value
    result = value * 2.0
```

**Multi-dimensional Load:**
```python
@ct.kernel
def load_2d(matrix: ct.Array[float32]):
    row = ct.bid(0)
    col = ct.bid(1)
    
    if row < matrix.shape[0] and col < matrix.shape[1]:
        value = ct.load(matrix, (row, col))
        ...
```

**Bounds Checking:**
```python
@ct.kernel
def safe_load(data: ct.Array[float32], n: int):
    index = ct.bid(0)
    
    # Always check bounds
    if index < n:
        value = ct.load(data, (index,))
    else:
        value = 0.0  # Default value
```

### Store Operations

**Scalar Store:**
```python
@ct.kernel
def store_element(input: ct.Array[float32], output: ct.Array[float32]):
    index = ct.bid(0)
    
    if index < output.shape[0]:
        value = ct.load(input, (index,))
        result = value * 2.0
        ct.store(output, (index,), result)
```

**Conditional Store:**
```python
@ct.kernel
def conditional_store(
    data: ct.Array[float32],
    mask: ct.Array[bool],  # Boolean mask
    output: ct.Array[float32]
):
    index = ct.bid(0)
    
    if index < data.shape[0]:
        should_store = ct.load(mask, (index,))
        if should_store:
            value = ct.load(data, (index,))
            ct.store(output, (index,), value)
```

### Load-Store Patterns

**Copy Kernel:**
```python
@ct.kernel
def array_copy(src: ct.Array[float32], dst: ct.Array[float32]):
    index = ct.bid(0)
    
    if index < src.shape[0]:
        value = ct.load(src, (index,))
        ct.store(dst, (index,), value)
```

**Scale and Offset:**
```python
@ct.kernel
def scale_offset(
    data: ct.Array[float32],
    scale: float32,
    offset: float32,
    output: ct.Array[float32]
):
    index = ct.bid(0)
    
    if index < data.shape[0]:
        value = ct.load(data, (index,))
        result = value * scale + offset
        ct.store(output, (index,), result)
```

**Gather Operation:**
```python
@ct.kernel
def gather(
    data: ct.Array[float32],
    indices: ct.Array[int32],
    output: ct.Array[float32]
):
    out_idx = ct.bid(0)
    
    if out_idx < output.shape[0]:
        # Get index to read from
        read_idx = ct.load(indices, (out_idx,))
        
        # Bounds check
        if read_idx < data.shape[0]:
            value = ct.load(data, (read_idx,))
            ct.store(output, (out_idx,), value)
```

## Memory Access Coalescing

Efficient memory access is crucial for GPU performance.

### Coalesced Access Patterns

**Good: Sequential Access:**
```python
@ct.kernel
def coalesced_read(data: ct.Array[float32], output: ct.Array[float32]):
    # Adjacent threads access adjacent elements
    index = ct.bid(0)
    
    if index < data.shape[0]:
        value = ct.load(data, (index,))  # Sequential
        ct.store(output, (index,), value * 2.0)
```

**Poor: Strided Access:**
```python
@ct.kernel
def strided_read(data: ct.Array[float32], output: ct.Array[float32]):
    # Threads access elements with stride
    block_idx = ct.bid(0)
    stride = 128  # Large stride
    
    index = block_idx * stride
    
    if index < data.shape[0]:
        value = ct.load(data, (index,))  # Strided access
        ct.store(output, (index,), value)
```

### 2D Access Patterns

**Row-Major Access (Good):**
```python
@ct.kernel
def row_major_access(matrix: ct.Array[float32]):
    row = ct.bid(0)
    col = ct.bid(1)
    
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # Adjacent columns are contiguous
        value = ct.load(matrix, (row, col))  # Good coalescing
```

**Column-Major Access (Poor):**
```python
@ct.kernel
def column_major_access(matrix: ct.Array[float32]):
    row = ct.bid(0)
    col = ct.bid(1)
    
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # Accessing different rows (strided)
        value = ct.load(matrix, (row, col))  # If row varies faster than col
```

### Optimization Guidelines

**1. Use Sequential Access:**
```python
# GOOD
for i in range(n):
    val = array[base + i]  # Sequential

# AVOID
for i in range(n):
    val = array[i * stride]  # Strided
```

**2. Align Data Structures:**
```python
# Align dimensions to powers of 2
good_shape = (1024, 1024)  # Aligned
bad_shape = (1000, 1000)   # May cause misalignment
```

**3. Minimize Stride Variations:**
```python
# Prefer consistent stride patterns
# Access along dimension 1 (stride=1) is fastest
for j in range(cols):
    val = matrix[row, j]  # Sequential along columns
```

## Advanced Array Operations

### Array Broadcasting

```python
@ct.kernel
def broadcast_operation(
    vector: ct.Array[float32],  # Shape: (1024,)
    matrix: ct.Array[float32],  # Shape: (1024, 1024)
    output: ct.Array[float32]   # Shape: (1024, 1024)
):
    # Broadcast vector across matrix columns
    row = ct.bid(0)
    col = ct.bid(1)
    
    if row < matrix.shape[0] and col < matrix.shape[1]:
        vec_val = ct.load(vector, (row,))  # Broadcast
        mat_val = ct.load(matrix, (row, col))
        ct.store(output, (row, col), vec_val + mat_val)
```

### Array Reduction Preparation

```python
@ct.kernel
def prepare_for_reduction(
    input: ct.Array[float32],
    partial_sums: ct.Array[float32]
):
    # Each block computes partial sum
    block_idx = ct.bid(0)
    block_size = 256
    
    # Initialize accumulator
    block_sum = 0.0
    
    # Accumulate elements
    for i in range(block_size):
        global_idx = block_idx * block_size + i
        if global_idx < input.shape[0]:
            val = ct.load(input, (global_idx,))
            block_sum = block_sum + val
    
    # Store partial sum
    ct.store(partial_sums, (block_idx,), block_sum)
```

## Conclusion

Global arrays are the foundation of data management in cuTile, providing a safe, efficient interface for GPU memory operations. Key concepts to remember:

**Creation and Lifecycle:**
- Arrays are allocated on the host using CuPy, PyTorch, or NumPy
- Arrays persist across kernel launches
- Host controls allocation, deallocation, and lifetime

**Memory Layout:**
- Strided layout enables flexible indexing
- Understanding strides is crucial for performance
- Contiguous access patterns maximize bandwidth

**Views vs. Copies:**
- Views share memory without copying
- Copies create independent data
- Choose based on performance vs. independence needs

**Interoperability:**
- CUDA Array Interface enables zero-copy sharing
- DLPack provides standard protocol
- Minimize CPU-GPU transfers

**Safety Rules:**
- No aliasing of kernel array arguments
- Always check array bounds
- Match types to kernel expectations
- Ensure shape compatibility

Mastering cuTile's array model enables efficient, correct GPU programs that leverage the full power of modern GPU hardware while maintaining safety and productivity. The next chapters will build on this foundation to explore more advanced operations and optimization techniques.