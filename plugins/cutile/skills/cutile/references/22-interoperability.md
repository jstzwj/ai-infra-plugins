# Chapter 22: Interoperability

## Overview

cuTile is designed to work seamlessly with the broader CUDA and Python GPU ecosystems. This chapter covers how cuTile interacts with other frameworks, how data flows between different systems, and how cuTile code integrates with existing CUDA applications.

## 22.1 Machine Representation

### 22.1.1 Translation Process

cuTile translates Python tile code into CUDA device code through a multi-stage compilation process:

1. **Python AST Analysis**: The cuTile compiler analyzes the abstract syntax tree of Python tile functions
2. **Type Inference**: Types are inferred for all variables, expressions, and operations
3. **IR Generation**: An intermediate representation (IR) is generated that captures tile semantics
4. **CUDA Code Generation**: The IR is lowered to CUDA C++ code
5. **PTX Generation**: NVIDIA's nvvm compiler generates PTX (Parallel Thread Execution) code
6. **Cubin Generation**: The ptxas assembler creates the final device binary

This process ensures that all cuTile constructs have well-defined machine representations in terms of CUDA C++ entities.

### 22.1.2 Type Mapping

Every cuTile dtype maps directly to a CUDA C++ type. This mapping ensures predictable behavior and allows cuTile to interoperate with other CUDA code that expects standard types.

#### Scalar Type Mapping

| cuTile Type | CUDA C++ Type | Size | Description |
|------------|---------------|------|-------------|
| `cuda.tile.int8` | `int8_t` | 1 byte | 8-bit signed integer |
| `cuda.tile.int16` | `int16_t` | 2 bytes | 16-bit signed integer |
| `cuda.tile.int32` | `int32_t` | 4 bytes | 32-bit signed integer |
| `cuda.tile.int64` | `int64_t` | 8 bytes | 64-bit signed integer |
| `cuda.tile.uint8` | `uint8_t` | 1 byte | 8-bit unsigned integer |
| `cuda.tile.uint16` | `uint16_t` | 2 bytes | 16-bit unsigned integer |
| `cuda.tile.uint32` | `uint32_t` | 4 bytes | 32-bit unsigned integer |
| `cuda.tile.uint64` | `uint64_t` | 8 bytes | 64-bit unsigned integer |
| `cuda.tile.float16` | `__half` | 2 bytes | 16-bit floating point |
| `cuda.tile.float32` | `float` | 4 bytes | 32-bit floating point |
| `cuda.tile.float64` | `double` | 8 bytes | 64-bit floating point |
| `cuda.tile.bool` | `bool` | 1 byte | Boolean value |
| `cuda.tile.bfloat16` | `__nv_bfloat16` | 2 bytes | Brain floating point |
| `cuda.tile.float8_e4m3fn` | `__nv_fp8_e4m3` | 1 byte | 8-bit FP8 (E4M3) |
| `cuda.tile.float8_e5m2` | `__nv_fp8_e5m2` | 1 byte | 8-bit FP8 (E5M2) |

#### Complex Type Mapping

| cuTile Type | CUDA C++ Type | Description |
|------------|---------------|-------------|
| `cuda.tile.complex64` | `complex<float>` or `cuComplex` | 64-bit complex number |
| `cuda.tile.complex128` | `complex<double>` or `cuDoubleComplex` | 128-bit complex number |

### 22.1.3 Object Representation

cuTile objects have specific machine representations:

#### Arrays
```python
import cuda.tile as ct

# cuTile Array → CUDA device pointer + metadata
arr = ct.array([1, 2, 3, 4], dtype=ct.int32)
# Machine representation: void* data_ptr, size_t size, int ndim, size_t* shape
```

In CUDA C++, this corresponds to:
```cpp
struct TileArray {
    void* data_ptr;
    size_t size;
    int ndim;
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
};
```

#### Views
```python
# Tiled views represent strided memory regions
view = arr.tiled_view(tile_shape=(16, 16))
# Machine representation: base_ptr, tile_shape, grid_shape, strides
```

#### Kernels
```python
@ct.tile
def my_kernel(x: ct.tile.float32):
    return x * 2.0
```

Kernels are compiled to CUDA device functions with specific signatures:
```cpp
__device__ float my_kernel(float x) {
    return x * 2.0f;
}
```

### 22.1.4 Function Calling Convention

cuTile functions follow the CUDA device function calling convention:

- Arguments passed through registers when possible
- Return values in registers
- Stack usage only for spills or large aggregates
- Inline expansion where beneficial

## 22.2 DLPack and CUDA Array Interface

### 22.2.1 Overview

cuTile supports two major interoperability standards for GPU arrays:

1. **DLPack**: A common in-memory tensor structure for deep learning frameworks
2. **CUDA Array Interface**: A Python protocol for sharing CUDA arrays between libraries

These standards enable zero-copy data sharing between cuTile and other GPU frameworks.

### 22.2.2 Zero-Copy Data Sharing

The key advantage of DLPack and CUDA Array Interface is zero-copy operation. When you pass a CuPy array, PyTorch tensor, or other GPU array to a cuTile kernel:

1. No data is copied
2. No device-to-host transfer occurs
3. Only a pointer and metadata are exchanged
4. Performance is identical to native cuTile arrays

This makes cuTile ideal for:
- Accelerating specific operations in existing PyTorch/CuPy pipelines
- Writing custom kernels that work with framework data
- Hybrid workflows combining multiple frameworks

### 22.2.3 DLPack Support

DLPack provides a standardized way to share tensor data between frameworks. cuTile can consume DLPack tensors from any framework that implements the protocol.

#### DLPack Type Mapping

| DLPack Type | cuTile Type | CUDA Type |
|-------------|-------------|-----------|
| kDLInt8 | int8 | int8_t |
| kDLInt16 | int16 | int16_t |
| kDLInt32 | int32 | int32_t |
| kDLInt64 | int64 | int64_t |
| kDLUInt8 | uint8 | uint8_t |
| kDLUInt16 | uint16 | uint16_t |
| kDLUInt32 | uint32 | uint32_t |
| kDLUInt64 | uint64 | uint64_t |
| kDLFloat16 | float16 | __half |
| kDLFloat32 | float32 | float |
| kDLFloat64 | float64 | double |

#### Using DLPack Directly

```python
import cuda.tile as ct
import cupy as cp
from dlpack import from_dlpack, to_dlpack

# Create a CuPy array
cupy_arr = cp.random.random(1024)

# Convert to DLPack (zero-copy)
dlpack_capsule = cp.array(cupy_arr).toDlpack()

# Consume in cuTile
ct_arr = ct.from_dlpack(dlpack_capsule)

# Use in cuTile kernel
@ct.tile
def scale(x: ct.tile.float32):
    return x * 2.0

result = ct.launch(stream, grid, scale, (ct_arr,))
```

### 22.2.4 CUDA Array Interface

The CUDA Array Interface is a Python protocol (similar to Python's buffer protocol) for sharing CUDA arrays. It's defined by a `__cuda_array_interface__` attribute on objects.

#### Interface Structure

```python
{
    'data': (ptr, read_only),  # Pointer to data, readonly flag
    'shape': tuple,            # Array shape
    'strides': tuple,          # Byte strides (or None for C-contiguous)
    'typestr': str,            # Typecode (e.g., '<f4')
    'version': 2,              # Interface version
}
```

#### CuPy Interoperability

CuPy has full CUDA Array Interface support. Here's how to use CuPy arrays with cuTile:

```python
import cupy as cp
import cuda.tile as ct

# Create CuPy array
x_cp = cp.random.random(1024).astype(cp.float32)

# Define cuTile kernel
@ct.tile
def saxpy(a: ct.tile.float32, x: ct.tile.float32, y: ct.tile.float32):
    return a * x + y

# Use CuPy arrays directly (zero-copy)
a = cp.float32(2.0)
y_cp = cp.zeros(1024, dtype=cp.float32)

stream = ct.Stream()
grid = (ct.cdiv(1024, 128), 128)
result = ct.launch(stream, grid, saxpy, (a, x_cp, y_cp))

# Result is a CuPy array (zero-copy view)
print(type(result))  # <class 'cupy.ndarray'>
print(result[:5])    # First 5 elements
```

#### PyTorch Interoperability

PyTorch tensors support the CUDA Array Interface, enabling seamless integration:

```python
import torch
import cuda.tile as ct

# Create PyTorch tensor on GPU
x_torch = torch.randn(1024, device='cuda', dtype=torch.float32)
y_torch = torch.zeros(1024, device='cuda', dtype=torch.float32)

# Define cuTile kernel
@ct.tile
def add(x: ct.tile.float32, y: ct.tile.float32):
    return x + y

# Get PyTorch stream
torch_stream = torch.cuda.current_stream()

# Launch cuTile kernel on PyTorch stream
grid = (ct.cdiv(1024, 128), 128)
result = ct.launch(torch_stream, grid, add, (x_torch, y_torch))

# Result is a PyTorch tensor (zero-copy view)
print(type(result))  # <class 'torch.Tensor'>
print(result.device) # cuda:0
```

#### Stream Sharing with PyTorch

When working with PyTorch, you can share streams between frameworks:

```python
import torch
import cuda.tile as ct

# Create PyTorch stream
torch_stream = torch.cuda.Stream()

# Use same stream for cuTile
@ct.tile
def kernel(x: ct.tile.float32):
    return x * 2.0

x = torch.randn(1024, device='cuda')
grid = (ct.cdiv(1024, 128), 128)

# Launch on PyTorch stream
result = ct.launch(torch_stream, grid, kernel, (x,))

# Synchronize
torch_stream.synchronize()
```

#### Numba Interoperability

Numba CUDA arrays also support the CUDA Array Interface:

```python
import numba.cuda as cuda
import cuda.tile as ct

# Create Numba device array
x_numba = cuda.to_device(np.random.random(1024))

# Use in cuTile kernel
@ct.tile
def process(x: ct.tile.float32):
    return x * x

result = ct.launch(stream, grid, process, (x_numba,))
```

### 22.2.5 Multi-Framework Workflows

You can chain operations across frameworks without copying data:

```python
import torch
import cupy as cp
import cuda.tile as ct

# Start with PyTorch
x_torch = torch.randn(1024, device='cuda')

# Apply cuTile kernel
@ct.tile
def custom_op(x: ct.tile.float32):
    return ct.sqrt(x * x + 1.0)

result1 = ct.launch(stream, grid, custom_op, (x_torch,))

# Switch to CuPy for another operation
result1_cupy = cp.asarray(result1)  # Zero-copy
result2 = cp.exp(result1_cupy)      # CuPy operation

# Back to PyTorch
result2_torch = torch.as_tensor(result2_cupy, device='cuda')  # Zero-copy

# Final cuTile operation
@ct.tile
def finalize(x: ct.tile.float32):
    return x / (1.0 + x)

final = ct.launch(stream, grid, finalize, (result2_torch,))
```

All operations in this pipeline occur on the GPU without any host-device transfers.

### 22.2.6 Host Arrays (NumPy)

While NumPy arrays don't have a CUDA Array Interface, they support DLPack for device-host transfers:

```python
import numpy as np
import cuda.tile as ct

# Create NumPy array on host
x_np = np.random.random(1024).astype(np.float32)

# Transfer to device
x_dev = ct.to_device(x_np)

# Process on device
@ct.tile
def process(x: ct.tile.float32):
    return x * 2.0

result_dev = ct.launch(stream, grid, process, (x_dev,))

# Transfer back to host
result_np = result_dev.copy_to_host()
```

For best performance, minimize host-device transfers by keeping data on the GPU.

### 22.2.7 Type Compatibility

When using external arrays, ensure type compatibility:

```python
import torch
import cuda.tile as ct

# PyTorch float32 → cuTile float32 ✓
x = torch.randn(1024, device='cuda', dtype=torch.float32)

# PyTorch float16 → cuTile float16 ✓
y = torch.randn(1024, device='cuda', dtype=torch.float16)

# Mismatch: torch.float64 with ct.float32 kernel → Runtime error
# Type checking occurs at kernel launch
```

### 22.2.8 Memory Layout Considerations

Different frameworks may use different memory layouts:

- **Row-major (C-style)**: Default for NumPy, PyTorch, CuPy
- **Column-major (F-style)**: Can be specified in NumPy

cuTile respects the memory layout of input arrays:

```python
import torch
import cuda.tile as ct

# C-contiguous (row-major)
x_c = torch.randn(1024, 1024, device='cuda')

# F-contiguous (column-major)
x_f = torch.randn(1024, 1024, device='cuda').t()  # Transpose creates F-order

# Both work correctly with cuTile
@ct.tile
def process_row(x: ct.tile.float32):
    # Accesses follow memory layout
    return x * 2.0
```

## 22.3 SIMT Interoperability

### 22.3.1 Overview

cuTile is designed to coexist with traditional SIMT (Single Instruction, Multiple Thread) CUDA programming. Two levels of interoperability are supported:

1. **Inter-kernel**: Tile and SIMT kernels in the same application
2. **Intra-kernel**: Mixing tile and SIMT code in the same kernel (future)

### 22.3.2 Inter-Kernel Interoperability

Tile and SIMT kernels can coexist in the same source file and binary:

```python
import cuda.tile as ct

# Tile kernel
@ct.tile
def tile_kernel(x: ct.tile.float32):
    return x * 2.0

# SIMT kernel (via CUDA Python or ctypes)
def simt_kernel_wrapper(x_ptr, n):
    # Launch traditional SIMT kernel
    # Implementation depends on your approach
    pass
```

#### Building with nvcc

You can compile SIMT kernels with nvcc and link them with cuTile:

```cpp
// simt_ops.cu
__global__ void simt_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
```

Compile and link:
```bash
nvcc --shared -Xcompiler -fPIC simt_ops.cu -o libsimt_ops.so
```

Use from Python:
```python
import ctypes
import cuda.tile as ct
import torch

# Load SIMT library
simt_lib = ctypes.CDLL('./libsimt_ops.so')

# Get function pointer
simt_kernel = simt_lib.simt_kernel

# Use with cuTile
x = torch.randn(1024, device='cuda')

# Apply tile kernel
@ct.tile
def tile_op(x: ct.tile.float32):
    return x + 1.0

result1 = ct.launch(stream, grid, tile_op, (x,))

# Apply SIMT kernel to same array
result1_ptr = ctypes.c_void_t(result1.data_ptr())
simt_kernel(result1_ptr, 1024)
```

#### Stream Synchronization

When mixing tile and SIMT kernels, use stream synchronization:

```python
import cuda.tile as ct
import torch

# Create stream
stream = ct.Stream()

# Launch tile kernel
tile_result = ct.launch(stream, grid, tile_kernel, (data,))

# Synchronize before SIMT kernel
stream.synchronize()

# Launch SIMT kernel on same data
simt_kernel(data_ptr, n)

# Synchronize again
stream.synchronize()

# Continue with tile kernel
final_result = ct.launch(stream, grid, another_tile_kernel, (data,))
```

#### Shared Memory and State

Tile and SIMT kernels can share state through:

1. **Global memory**: Both can read/write the same arrays
2. **CUDA streams**: Synchronize operations
3. **CUDA events**: Cross-framework dependencies

```python
import cuda.tile as ct
import torch

# Shared data
data = torch.zeros(1024, device='cuda')

# Tile kernel writes
@ct.tile
def initialize(x: ct.tile.float32):
    return x + 1.0

ct.launch(stream, grid, initialize, (data,))

# SIMT kernel reads and modifies
# ... launch SIMT kernel ...

# Tile kernel reads modified data
@ct.tile
def finalize(x: ct.tile.float32):
    return x * 2.0

result = ct.launch(stream, grid, finalize, (data,))
```

### 22.3.3 Performance Considerations

When mixing tile and SIMT code:

1. **Minimize transitions**: Each transition has overhead
2. **Batch operations**: Group all tile operations, then all SIMT operations
3. **Stream concurrency**: Use multiple streams when possible
4. **Profile**: Identify bottlenecks from mixing frameworks

```python
# Good: Batch tile operations
result1 = ct.launch(stream1, grid, tile_op1, (data,))
result2 = ct.launch(stream1, grid, tile_op2, (result1,))
result3 = ct.launch(stream1, grid, tile_op3, (result2,))

# Then batch SIMT operations
simt_op1(result3_ptr)
simt_op2(result3_ptr)
simt_op3(result3_ptr)
```

### 22.3.4 Future: Intra-Kernel Interoperability

Future versions of cuTile will support mixing tile and SIMT code within the same kernel:

```python
# Planned feature (not yet available)
@ct.tile
def hybrid_kernel(x: ct.tile.float32):
    # Tile code
    y = x * 2.0
    
    # SIMT code block
    with ct.simt_region():
        # Traditional CUDA code
        pass
    
    # Back to tile code
    return y + 1.0
```

This will enable:
- Fine-grained control over execution model
- Hand-optimized critical paths within tile kernels
- Gradual migration from SIMT to tile

## 22.4 FP8 PyTorch Compatibility

### 22.4.1 Overview

FP8 (8-bit floating point) support requires special handling when working with PyTorch due to DLPack conversion limitations in older PyTorch versions.

### 22.4.2 Version Requirements

For FP8 support with PyTorch:

- **Minimum version**: PyTorch >= 2.10
- **Recommended**: PyTorch >= 2.12

Older PyTorch versions have incomplete FP8 support in DLPack conversion, which can lead to memory leaks or incorrect results.

### 22.4.3 FP8 Types in cuTile

cuTile supports two FP8 formats:

| cuTile Type | Description | Use Case |
|-------------|-------------|----------|
| `float8_e4m3fn` | 4-bit exponent, 3-bit mantissa | Weights, activations |
| `float8_e5m2` | 5-bit exponent, 2-bit mantissa | Gradients |

### 22.4.4 Using FP8 with PyTorch (Recommended Version)

With PyTorch >= 2.10:

```python
import torch
import cuda.tile as ct

# Create FP8 tensor (PyTorch >= 2.10)
x_fp8 = torch.randn(1024, device='cuda', dtype=torch.float8_e4m3fn)

# Use in cuTile kernel
@ct.tile
def fp8_op(x: ct.tile.float8_e4m3fn):
    return x * 2.0  # FP8 arithmetic

result = ct.launch(stream, grid, fp8_op, (x_fp8,))
```

### 22.4.5 FP8 Conversion Patterns

When working with mixed precision:

```python
import torch
import cuda.tile as ct

# Start with FP32
x_fp32 = torch.randn(1024, device='cuda', dtype=torch.float32)

# Convert to FP8 for cuTile kernel
x_fp8 = x_fp32.to(torch.float8_e4m3fn)

# cuTile kernel in FP8
@ct.tile
def fp8_compute(x: ct.tile.float8_e4m3fn):
    return x * x

result_fp8 = ct.launch(stream, grid, fp8_compute, (x_fp8,))

# Convert back to FP32
result_fp32 = result_fp8.to(torch.float32)
```

### 22.4.6 Memory Leak Warning (Older PyTorch)

With PyTorch < 2.10:

```python
# WARNING: Memory leak with FP8 in older PyTorch
import torch
import cuda.tile as ct

# PyTorch < 2.10
x_fp8 = torch.randn(1024, device='cuda', dtype=torch.float8_e4m3fn)

# This may cause memory leak on DLPack conversion failure
@ct.tile
def unsafe_kernel(x: ct.tile.float8_e4m3fn):
    return x + 1.0

# Error or memory leak occurs here
result = ct.launch(stream, grid, unsafe_kernel, (x_fp8,))
```

**Workaround for older versions**:
1. Upgrade PyTorch to >= 2.10
2. Avoid FP8 types, use FP16/BF16 instead
3. Use manual memory management (not recommended)

### 22.4.7 FP8 Performance Considerations

FP8 provides:
- **2x memory reduction** vs FP16
- **2-4x throughput** on H100 GPUs
- **Trade-off**: Reduced precision and range

Best practices:
- Use for inference where precision loss is acceptable
- Verify numerical accuracy for your use case
- Consider mixed-precision approaches (FP8 compute, FP32 master weights)

### 22.4.8 FP8 Autotuning

When using FP8, autotuning becomes more important due to precision trade-offs:

```python
import cuda.tile as ct

# Autotune FP8 kernel configurations
@ct.tile
def fp8_matmul(A: ct.tile.float8_e4m3fn, B: ct.tile.float8_e4m3fn):
    # Implementation
    pass

# Autotune for best accuracy/performance trade-off
best_config = ct.tune.exhaustive_search(
    fp8_matmul,
    inputs=(A_fp8, B_fp8),
    configs=[...],
    metric='accuracy'
)
```

## 22.5 Best Practices

### 22.5.1 Framework Selection

Choose the right framework for each operation:

| Use Case | Recommended Framework |
|----------|----------------------|
| Custom algorithms | cuTile |
| Deep learning primitives | PyTorch native |
| Linear algebra | CuPy |
- Prefer framework-native operations when available
- Use cuTile for custom operations not available elsewhere
- Minimize framework transitions in hot loops

### 22.5.2 Type Consistency

Maintain type consistency across frameworks:

```python
# Good: Consistent types
x = torch.randn(1024, device='cuda', dtype=torch.float32)
result = ct.launch(stream, grid, float32_kernel, (x,))

# Bad: Type mismatch requires conversion
x = torch.randn(1024, device='cuda', dtype=torch.float64)
result = ct.launch(stream, grid, float32_kernel, (x,))  # Error!
```

### 22.5.3 Memory Management

Best practices for memory management:

1. **Reuse arrays**: Avoid allocations in loops
2. **Prefer views**: Use views instead of copies when possible
3. **Explicit synchronization**: Know when data is ready
4. **Profile transfers**: Minimize host-device copies

```python
# Good: Reuse arrays
output = torch.zeros_like(input)
for i in range(iterations):
    temp = ct.launch(stream, grid, kernel, (input,))
    output = ct.launch(stream, grid, finalize, (temp,))
    # Reuse temp buffer (no reallocation)

# Bad: Allocate in loop
for i in range(iterations):
    temp = torch.zeros(...)  # New allocation each iteration
    temp = ct.launch(stream, grid, kernel, (input,))
```

### 22.5.4 Debugging Interop Issues

When troubleshooting interoperability:

1. **Check types**: Ensure dtype compatibility
2. **Verify devices**: Confirm all arrays on same GPU
3. **Synchronize streams**: Add synchronization for debugging
4. **Enable validation**: Use debug builds when available
5. **Isolate issues**: Test each framework independently

```python
# Debugging pattern
import torch
import cuda.tile as ct

# Verify input
print(f"Input dtype: {x.dtype}")  # Should match kernel
print(f"Input device: {x.device}")  # Should be cuda

# Add synchronization
stream.synchronize()
result = ct.launch(stream, grid, kernel, (x,))
stream.synchronize()

# Verify output
print(f"Output shape: {result.shape}")
print(f"Output dtype: {result.dtype}")
```

## 22.6 Summary

cuTile's interoperability features enable:

- **Zero-copy integration** with PyTorch, CuPy, and other frameworks
- **Type-safe interoperability** through DLPack and CUDA Array Interface
- **Mixed-framework workflows** without data transfer overhead
- **SIMT coexistence** for gradual migration
- **FP8 support** with PyTorch >= 2.10

By understanding these interoperability mechanisms, you can seamlessly integrate cuTile into existing GPU applications and workflows.
