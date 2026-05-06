# Compilation and Export

cuTile provides two modes of kernel compilation: Just-In-Time (JIT) compilation via `ct.launch()` and Ahead-Of-Time (AOT) compilation via `export_kernel()`. This chapter covers AOT compilation, which allows you to pre-compile kernels to CUDA binaries (CUBIN) or TileIR bytecode for deployment without Python dependencies.

## JIT vs AOT Compilation

### Just-In-Time (JIT) Compilation

JIT compilation is the default mode when using `ct.launch()`. The compiler automatically specializes your kernel for the concrete arguments provided at launch time:

```python
import cuda.tile as ct
import torch

@ct.kernel
def matmul(a: ct.tensor, b: ct.tensor, c: ct.tensor):
    # Kernel implementation
    pass

# JIT: Auto-specializes for these concrete tensor shapes
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = torch.randn(1024, 1024, device='cuda')

ct.launch(matmul, a, b, c)
```

**Advantages:**
- Automatic specialization for tensor shapes, strides, and dtypes
- No manual signature specification required
- Easy to use for development and prototyping

**Disadvantages:**
- Requires cuTile/Python runtime at deployment
- Compilation happens at runtime (first launch)
- Less control over generated code

### Ahead-Of-Time (AOT) Compilation

AOT compilation uses `export_kernel()` to pre-compile kernels before deployment:

```python
from cuda.tile.compilation import export_kernel, KernelSignature, ScalarConstraint, ArrayConstraint

# Define kernel
@ct.kernel
def matmul(a: ct.tensor, b: ct.tensor, c: ct.tensor):
    # Kernel implementation
    pass

# Build explicit signatures
signatures = [
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype=float32, ndim=2),
            ArrayConstraint(dtype=float32, ndim=2),
            ArrayConstraint(dtype=float32, ndim=2),
        ],
        calling_convention='cutile_python_v1'
    )
]

# Export to CUBIN
with open('matmul.cubin', 'wb') as f:
    export_kernel(
        kernel=matmul,
        signatures=signatures,
        output_file=f,
        gpu_code='sm_80',
        output_format='cubin'
    )
```

**Advantages:**
- No Python dependency at deployment
- Compilation happens once, offline
- More control over generated code
- Can integrate with C/C++ applications
- Enables kernel distribution without source

**Disadvantages:**
- Requires manual signature specification
- Less flexibility for different tensor shapes
- More complex build process

## `export_kernel()` Function

The `export_kernel()` function compiles a cuTile kernel to a binary format.

### Signature

```python
cuda.tile.compilation.export_kernel(
    kernel: Callable,
    signatures: Sequence[KernelSignature],
    output_file: Union[str, BytesIO],
    *,
    gpu_code: str,
    output_format: Literal['cubin', 'tileir_bytecode'],
    bytecode_version: Optional[str] = None
)
```

### Parameters

#### `kernel`: Callable

The `@ct.kernel` decorated function to compile. Must be a valid cuTile kernel:

```python
@ct.kernel
def my_kernel(
    input: ct.tensor,
    output: ct.tensor,
    scale: ct.scalar
):
    pid = ct.program_id(0)
    output[pid] = input[pid] * scale
```

**Requirements:**
- Must be decorated with `@ct.kernel`
- Must use valid cuTile operations
- Must have valid type annotations for all parameters

#### `signatures`: Sequence[KernelSignature]

A sequence of `KernelSignature` objects describing the kernel's parameter types and calling conventions. Each signature represents a specialized version of the kernel:

```python
signatures = [
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype='float32', ndim=2),
            ArrayConstraint(dtype='float32', ndim=2),
            ScalarConstraint(dtype='float32'),
        ],
        calling_convention='cutile_python_v1'
    )
]
```

**Why multiple signatures?**
You can export multiple specializations of the same kernel:

```python
signatures = [
    # FP32 version
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype='float32', ndim=2),
            ArrayConstraint(dtype='float32', ndim=2),
        ],
        calling_convention='cutile_python_v1',
        symbol='matmul_f32'
    ),
    # FP16 version
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype='float16', ndim=2),
            ArrayConstraint(dtype='float16', ndim=2),
        ],
        calling_convention='cutile_python_v1',
        symbol='matmul_f16'
    ),
]
```

#### `output_file`: Union[str, BytesIO]

Destination for the compiled kernel. Either a file path (string) or a `BytesIO` object:

```python
# Write to file
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file='my_kernel.cubin',
    gpu_code='sm_80',
    output_format='cubin'
)

# Write to memory buffer
from io import BytesIO
buffer = BytesIO()
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file=buffer,
    gpu_code='sm_80',
    output_format='cubin'
)
cubin_data = buffer.getvalue()
```

#### `gpu_code`: str

Target GPU architecture. Specifies the compute capability for which to compile:

| GPU Architecture | gpu_code Value |
|---|---|
| NVIDIA Ampere (RTX 30xx, A100) | `'sm_80'`, `'sm_86'` |
| NVIDIA Hopper (H100) | `'sm_90'` |
| NVIDIA Ada Lovelace (RTX 40xx) | `'sm_89'` |
| NVIDIA Volta (V100) | `'sm_70'` |
| NVIDIA Turing (RTX 20xx) | `'sm_75'` |

```python
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file='my_kernel.cubin',
    gpu_code='sm_80',  # Compile for A100/RTX 3090
    output_format='cubin'
)
```

**Finding your GPU architecture:**
```python
import torch
print(torch.cuda.get_device_capability())  # Returns (major, minor)
# For (8, 0), use 'sm_80'
```

#### `output_format`: Literal['cubin', 'tileir_bytecode']

Output format for the compiled kernel:

- `'cubin'`: CUDA binary format. Can be loaded with `cuModuleLoad()` or `cudaLaunchKernel()`
- `'tileir_bytecode'`: TileIR bytecode format. Used for debugging and analysis

```python
# Export as CUDA binary
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file='my_kernel.cubin',
    gpu_code='sm_80',
    output_format='cubin'
)

# Export as TileIR bytecode
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file='my_kernel.tbc',
    gpu_code='sm_80',
    output_format='tileir_bytecode'
)
```

#### `bytecode_version`: Optional[str]

TileIR bytecode version. Specify as `'major.minor'` or `None` for automatic detection:

```python
# Use latest bytecode version
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file='my_kernel.tbc',
    gpu_code='sm_80',
    output_format='tileir_bytecode',
    bytecode_version=None  # Auto-detect latest
)

# Specify specific version
export_kernel(
    kernel=my_kernel,
    signatures=signatures,
    output_file='my_kernel.tbc',
    gpu_code='sm_80',
    output_format='tileir_bytecode',
    bytecode_version='1.2'  # Use version 1.2
)
```

## `KernelSignature` Class

`KernelSignature` describes the parameter types and calling convention for a kernel specialization.

### Constructor

```python
KernelSignature(
    parameters: Sequence[ParameterConstraint],
    calling_convention: str,
    symbol: Optional[str] = None
)
```

### Parameters

#### `parameters`: Sequence[ParameterConstraint]

Sequence of parameter constraints describing each kernel parameter:

```python
KernelSignature(
    parameters=[
        ArrayConstraint(dtype='float32', ndim=2),
        ArrayConstraint(dtype='float32', ndim=2),
        ScalarConstraint(dtype='float32'),
    ],
    calling_convention='cutile_python_v1'
)
```

See "ParameterConstraint Types" below for details on each constraint type.

#### `calling_convention`: str

Binary format for passing kernel arguments. Currently only `'cutile_python_v1'` is supported:

```python
KernelSignature(
    parameters=[...],
    calling_convention='cutile_python_v1'  # Required
)
```

See "Calling Conventions — cutile_python_v1" below for details.

#### `symbol`: Optional[str]

Symbol name for the compiled kernel. If `None`, the symbol is automatically mangled based on the kernel function name:

```python
# Auto-generated symbol (mangled)
sig1 = KernelSignature(
    parameters=[...],
    calling_convention='cutile_python_v1'
)
# Symbol: '_Z8my_kernel...' (mangled)

# Custom symbol
sig2 = KernelSignature(
    parameters=[...],
    calling_convention='cutile_python_v1',
    symbol='my_custom_kernel_name'
)
# Symbol: 'my_custom_kernel_name'
```

### Class Methods

#### `from_kernel_args()`

Create a `KernelSignature` from example kernel arguments. This is primarily for testing and prototyping:

```python
import torch

# Create example arguments
example_a = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
example_b = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
example_scale = 2.0

# Derive signature from examples
sig = KernelSignature.from_kernel_args(
    kernel=my_kernel,
    kernel_args=(example_a, example_b, example_scale),
    calling_convention='cutile_python_v1'
)
```

**Warning:** This method is for testing only. Production code should manually construct signatures for explicit control.

#### `with_mangled_symbol()`

Returns a copy of the signature with a mangled symbol name:

```python
sig = KernelSignature(
    parameters=[...],
    calling_convention='cutile_python_v1'
)

# Create copy with mangled symbol
mangled_sig = sig.with_mangled_symbol('my_kernel')
```

#### `with_symbol()`

Returns a copy of the signature with a custom symbol:

```python
sig = KernelSignature(
    parameters=[...],
    calling_convention='cutile_python_v1'
)

# Create copy with custom symbol
custom_sig = sig.with_symbol('custom_kernel_name')
```

## `ParameterConstraint` Types

Parameter constraints describe the type, shape, and layout properties of kernel parameters.

### `ScalarConstraint`

Describes a scalar (single-value) parameter.

```python
ScalarConstraint(dtype: Union[str, DType])
```

**Parameters:**
- `dtype`: Data type (e.g., `'float32'`, `'int32'`, `ct.float32`)

**Example:**
```python
from cuda.tile.compilation import ScalarConstraint

ScalarConstraint(dtype='float32')
ScalarConstraint(dtype=ct.int32)
```

**Corresponding kernel parameter:**
```python
@ct.kernel
def my_kernel(data: ct.tensor, scale: ct.scalar):
    # scale is a scalar parameter
    pass
```

### `ArrayConstraint`

Describes an array parameter with compile-time assumptions about shape, strides, and aliasing.

```python
ArrayConstraint(
    dtype: Union[str, DType],
    ndim: int,
    *,
    index_dtype: Optional[Union[str, DType]] = None,
    stride_lower_bound_incl: Optional[int] = None,
    alias_groups: Optional[Sequence[int]] = None,
    may_alias_internally: bool = False,
    stride_constant: Optional[bool] = None,
    stride_divisible_by: Optional[Sequence[int]] = None,
    shape_divisible_by: Optional[Sequence[int]] = None,
    base_addr_divisible_by: Optional[int] = None
)
```

**Required Parameters:**

- `dtype`: Element data type (e.g., `'float32'`, `'float16'`, `'int32'`)
- `ndim`: Number of dimensions (e.g., `1`, `2`, `3`)

**Optional Parameters:**

#### `index_dtype`: Optional[Union[str, DType]]

Data type for indexing this array. Default is platform-dependent:

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    index_dtype='int64'  # Use 64-bit indexing
)
```

#### `stride_lower_bound_incl`: Optional[int]

Lower bound (inclusive) for all stride values. Useful for ensuring contiguous or aligned memory:

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    stride_lower_bound_incl=1  # Strides must be >= 1
)
```

#### `alias_groups`: Optional[Sequence[int]]

Which alias groups this array belongs to. Arrays in the same alias group may alias (point to overlapping memory):

```python
# a and b may alias, c does not alias with either
sig = KernelSignature(
    parameters=[
        ArrayConstraint(dtype='float32', ndim=2, alias_groups=[0]),
        ArrayConstraint(dtype='float32', ndim=2, alias_groups=[0]),
        ArrayConstraint(dtype='float32', ndim=2, alias_groups=[1]),
    ],
    calling_convention='cutile_python_v1'
)
```

#### `may_alias_internally`: bool

Whether different elements of this array may alias with each other:

```python
ArrayConstraint(
    dtype='float32',
    ndim=1,
    may_alias_internally=False  # Elements don't alias
)
```

#### `stride_constant`: Optional[bool]

Whether stride values are compile-time constants:

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    stride_constant=True  # Strides are known at compile time
)
```

#### `stride_divisible_by`: Optional[Sequence[int]]

Each stride must be divisible by the corresponding value:

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    stride_divisible_by=[4, 1024]  # stride[0] % 4 == 0, stride[1] % 1024 == 0
)
```

#### `shape_divisible_by`: Optional[Sequence[int]]

Each dimension size must be divisible by the corresponding value:

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    shape_divisible_by=[32, 32]  # shape[0] % 32 == 0, shape[1] % 32 == 0
)
```

#### `base_addr_divisible_by`: Optional[int]

Base address must be aligned to this byte boundary:

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    base_addr_divisible_by=256  # 256-byte alignment
)
```

**Example: Complete Array Constraint**

```python
ArrayConstraint(
    dtype='float32',
    ndim=2,
    index_dtype='int64',
    stride_lower_bound_incl=1,
    stride_divisible_by=[4, 1024],
    shape_divisible_by=[32, 32],
    base_addr_divisible_by=256,
    may_alias_internally=False,
    stride_constant=False
)
```

### `ListConstraint`

Describes a list of arrays parameter.

```python
ListConstraint(
    element: ArrayConstraint,
    *,
    alias_groups: Optional[Sequence[int]] = None,
    elements_may_alias: bool = False
)
```

**Parameters:**

- `element`: `ArrayConstraint` describing each element in the list
- `alias_groups`: Alias groups for the list itself (not individual elements)
- `elements_may_alias`: Whether list elements may alias with each other

**Example:**
```python
from cuda.tile.compilation import ListConstraint

ListConstraint(
    element=ArrayConstraint(dtype='float32', ndim=1),
    elements_may_alias=True
)
```

**Corresponding kernel parameter:**
```python
@ct.kernel
def my_kernel(arrays: ct.list[ct.tensor]):
    # arrays is a list of tensors
    pass
```

### `ConstantConstraint`

Describes a compile-time constant parameter.

```python
ConstantConstraint(value: Any)
```

**Parameters:**

- `value`: Constant value (must be a literal or compile-time constant)

**Example:**
```python
from cuda.tile.compilation import ConstantConstraint

ConstantConstraint(value=32)
ConstantConstraint(value=3.14159)
```

**Corresponding kernel parameter:**
```python
@ct.kernel
def my_kernel(data: ct.tensor, BLOCK_SIZE: ct.constant):
    # BLOCK_SIZE is a compile-time constant
    pass
```

## Calling Conventions — `cutile_python_v1`

The `cutile_python_v1` calling convention defines how kernel arguments are passed in binary form. This is the format used when launching exported kernels from C/C++.

### Binary Format Overview

For each `ParameterConstraint` type, the calling convention specifies how arguments are passed:

| Constraint Type | Number of Arguments | Format |
|---|---|---|
| `ScalarConstraint` | 1 | Single value |
| `ArrayConstraint` | 1 + 2×ndim | Pointer + shape + strides |
| `ListConstraint` | 2 | Pointer + int32 length |
| `ConstantConstraint` | 0 | Omitted from launch arguments |

### ScalarConstraint Format

Single value argument. C type depends on dtype:

| cuTile dtype | C Type | Size |
|---|---|---|
| `float32` | `float` | 4 bytes |
| `float64` | `double` | 8 bytes |
| `int8` | `int8_t` | 1 byte |
| `int16` | `int16_t` | 2 bytes |
| `int32` | `int32_t` | 4 bytes |
| `int64` | `int64_t` | 8 bytes |
| `uint8` | `uint8_t` | 1 byte |
| `uint16` | `uint16_t` | 2 bytes |
| `uint32` | `uint32_t` | 4 bytes |
| `uint64` | `uint64_t` | 8 bytes |

**Example:**
```python
# Python definition
@ct.kernel
def scale_kernel(data: ct.tensor, scale: ct.scalar):
    pid = ct.program_id(0)
    data[pid] = data[pid] * scale

# Corresponding C argument
float scale = 2.0f;  // For float32 scalar
```

### ArrayConstraint Format

1 + 2×ndim arguments:

1. **Pointer**: `void*` to array data
2. **Shape**: `int64_t[ndim]` array of dimension sizes
3. **Strides**: `int64_t[ndim]` array of dimension strides (in bytes)

**Index dtype C type mapping:**

| Index dtype | C Type |
|---|---|
| `int32` | `int32_t` |
| `int64` | `int64_t` |

**Example: 2D Array**

```python
# Python definition
@ct.kernel
def matmul(a: ct.tensor, b: ct.tensor, c: ct.tensor):
    # Implementation
    pass

# Corresponding C arguments for 2D array [1024, 1024]
void* a_ptr = a_data;
int64_t a_shape[2] = {1024, 1024};
int64_t a_strides[2] = {1024 * 4, 4};  // Assuming float32 (4 bytes)
```

**Complete C structure:**
```c
struct ArrayArg {
    void* ptr;
    int64_t shape[ndim];
    int64_t strides[ndim];
};
```

### ListConstraint Format

2 arguments:

1. **Pointer**: `void*` to array of pointers (one per list element)
2. **Length**: `int32_t` number of elements in list

**Example:**
```python
# Python definition
@ct.kernel
def process_list(arrays: ct.list[ct.tensor]):
    idx = ct.program_id(0)
    # Process arrays[idx]
    pass

# Corresponding C arguments
void* array_ptrs[] = {arr1_ptr, arr2_ptr, arr3_ptr};
int32_t length = 3;
```

### ConstantConstraint Format

Constant parameters are **omitted** from launch arguments. The constant value is baked into the compiled kernel.

```python
# Python definition
@ct.kernel
def tiled_kernel(data: ct.tensor, TILE_SIZE: ct.constant):
    # TILE_SIZE is compile-time constant
    pass

# Corresponding signature
sig = KernelSignature(
    parameters=[
        ArrayConstraint(dtype='float32', ndim=1),
        ConstantConstraint(value=32),  # TILE_SIZE = 32
    ],
    calling_convention='cutile_python_v1'
)

# C launch arguments: only data pointer, shape, strides
# TILE_SIZE is not passed at launch time
```

## Complete AOT Example

This example demonstrates the complete AOT workflow: defining a kernel, building signatures, exporting to CUBIN, and launching from C/C++.

### Step 1: Define Kernel with Constant Parameters

```python
import cuda.tile as ct

@ct.kernel
def tiled_matmul(
    a: ct.tensor,
    b: ct.tensor,
    c: ct.tensor,
    BLOCK_SIZE: ct.constant,
    BLOCK_M: ct.constant,
    BLOCK_N: ct.constant,
    BLOCK_K: ct.constant
):
    """
    Matrix multiplication with compile-time tile sizes.
    
    Args:
        a: Input matrix [M, K]
        b: Input matrix [K, N]
        c: Output matrix [M, N]
        BLOCK_SIZE: Block size for tiling
        BLOCK_M: Tile size for M dimension
        BLOCK_N: Tile size for N dimension
        BLOCK_K: Tile size for K dimension
    """
    # Get thread and block IDs
    pid_m = ct.program_id(0)
    pid_n = ct.program_id(1)
    
    # Compute global row and column
    row = pid_m * BLOCK_M + ct.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + ct.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    
    # Loop over K dimension in tiles
    for k in range(0, b.shape[0], BLOCK_K):
        # Load tiles
        a_tile = a[row, k:k+BLOCK_K]
        b_tile = b[k:k+BLOCK_K, col]
        
        # Accumulate
        acc += ct.dot(a_tile, b_tile)
    
    # Write result
    c[row, col] = acc
```

### Step 2: Build Signatures Manually

```python
from cuda.tile.compilation import (
    export_kernel,
    KernelSignature,
    ArrayConstraint,
    ConstantConstraint
)

# Define signature for FP32 matrices with specific tile sizes
signature = KernelSignature(
    parameters=[
        ArrayConstraint(
            dtype='float32',
            ndim=2,
            shape_divisible_by=[128, 128],  # M, K divisible by 128
            stride_divisible_by=[4, 512],   # Alignment requirements
        ),
        ArrayConstraint(
            dtype='float32',
            ndim=2,
            shape_divisible_by=[128, 128],  # K, N divisible by 128
            stride_divisible_by=[4, 512],
        ),
        ArrayConstraint(
            dtype='float32',
            ndim=2,
            stride_divisible_by=[4, 512],
        ),
        ConstantConstraint(value=128),  # BLOCK_SIZE
        ConstantConstraint(value=64),   # BLOCK_M
        ConstantConstraint(value=64),   # BLOCK_N
        ConstantConstraint(value=32),   # BLOCK_K
    ],
    calling_convention='cutile_python_v1',
    symbol='tiled_matmul_f32_64x64x32'
)

# Create multiple signatures for different tile sizes
signatures = [
    # 64x64x32 tiling
    signature,
    # 128x128x32 tiling
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype='float32', ndim=2, shape_divisible_by=[256, 256]),
            ArrayConstraint(dtype='float32', ndim=2, shape_divisible_by=[256, 256]),
            ArrayConstraint(dtype='float32', ndim=2),
            ConstantConstraint(value=128),
            ConstantConstraint(value=128),  # BLOCK_M
            ConstantConstraint(value=128),  # BLOCK_N
            ConstantConstraint(value=32),   # BLOCK_K
        ],
        calling_convention='cutile_python_v1',
        symbol='tiled_matmul_f32_128x128x32'
    ),
]
```

### Step 3: Export to CUBIN

```python
# Export kernel to CUBIN file
with open('tiled_matmul.cubin', 'wb') as f:
    export_kernel(
        kernel=tiled_matmul,
        signatures=signatures,
        output_file=f,
        gpu_code='sm_80',  # Target A100/RTX 3090
        output_format='cubin'
    )

print("Kernel exported to tiled_matmul.cubin")
```

### Step 4: Launch Exported Kernel from C/C++

```c
#include <cuda_runtime.h>
#include <stdio.h>

// Structure for passing array arguments (matches cutile_python_v1 convention)
struct ArrayArg {
    void* ptr;
    int64_t shape[2];
    int64_t strides[2];
};

// Kernel function pointer (loaded from CUBIN)
typedef void (*KernelFunc)(void** args);

int main() {
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Allocate matrices (e.g., 1024x1024)
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    const size_t size_bytes = M * K * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_bytes);
    cudaMalloc(&d_b, size_bytes);
    cudaMalloc(&d_c, size_bytes);
    
    // Initialize data (omitted for brevity)
    // ...
    
    // Load CUBIN module
    CUmodule module;
    cuModuleLoad(&module, "tiled_matmul.cubin");
    
    // Get kernel function
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "tiled_matmul_f32_64x64x32");
    
    // Prepare arguments according to cutile_python_v1 convention
    // Array arguments: ptr + shape + strides
    struct ArrayArg a_arg = {d_a, {M, K}, {K * 4, 4}};  // 4 bytes per float32
    struct ArrayArg b_arg = {d_b, {K, N}, {N * 4, 4}};
    struct ArrayArg c_arg = {d_c, {M, N}, {N * 4, 4}};
    
    // Pack arguments into array of pointers
    void* args[] = {
        &a_arg.ptr,
        a_arg.shape,
        a_arg.strides,
        &b_arg.ptr,
        b_arg.shape,
        b_arg.strides,
        &c_arg.ptr,
        c_arg.shape,
        c_arg.strides,
        // Constants (BLOCK_SIZE, BLOCK_M, BLOCK_N, BLOCK_K) are omitted
    };
    
    // Launch kernel
    dim3 grid(16, 16);  // Grid size for 1024x1024 with 64x64 tiles
    dim3 block(128);    // Block size (must match BLOCK_SIZE constant)
    
    cuLaunchKernel(
        kernel,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, NULL,  // Shared memory and stream
        args, NULL
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Copy result back to host (omitted for brevity)
    // ...
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cuModuleUnload(module);
    
    return 0;
}
```

### Step 5: Compile and Run

```bash
# Compile C++ wrapper
nvcc -o run_matmul main.cu -lcuda

# Run
./run_matmul
```

## Advanced Topics

### Multiple Architectures

To support multiple GPU architectures, export separate CUBIN files:

```python
gpu_archs = ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90']

for arch in gpu_archs:
    filename = f'tiled_matmul_{arch}.cubin'
    with open(filename, 'wb') as f:
        export_kernel(
            kernel=tiled_matmul,
            signatures=signatures,
            output_file=f,
            gpu_code=arch,
            output_format='cubin'
        )
    print(f"Exported for {arch}")
```

Then load the appropriate CUBIN at runtime based on the detected GPU:

```c
// Detect GPU architecture
int major, minor;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

char arch_str[32];
sprintf(arch_str, "tiled_matmul_sm_%d%d.cubin", major, minor);

// Load appropriate CUBIN
cuModuleLoad(&module, arch_str);
```

### TileIR Bytecode for Debugging

Export TileIR bytecode to inspect the intermediate representation:

```python
# Export TileIR bytecode for analysis
with open('kernel_bytecode.tbc', 'wb') as f:
    export_kernel(
        kernel=my_kernel,
        signatures=signatures,
        output_file=f,
        gpu_code='sm_80',
        output_format='tileir_bytecode',
        bytecode_version='1.2'
    )
```

This can be useful for:
- Debugging compilation issues
- Analyzing generated code
- Verifying optimizations

### Integration with Build Systems

For production use, integrate AOT compilation into your build system:

**CMake example:**
```cmake
# CMakeLists.txt
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_command(
    OUTPUT tiled_matmul.cubin
    COMMAND Python3::Interpreter
        export_kernels.py
        -o tiled_matmul.cubin
        --gpu-arch sm_80
    DEPENDS export_kernels.py my_kernels.py
    VERBATIM
)

add_custom_target(kernel_cubin DEPENDS tiled_matmul.cubin)
```

**Makefile example:**
```makefile
# Makefile
tiled_matmul.cubin: my_kernels.py
	python3 export_kernels.py -o $@ --gpu-arch sm_80

.PHONY: kernels
kernels: tiled_matmul.cubin
```

## Best Practices

### 1. Use Specific Constraints

Provide specific constraints for better optimization:

```python
# GOOD: Specific constraints
ArrayConstraint(
    dtype='float32',
    ndim=2,
    shape_divisible_by=[32, 32],
    stride_divisible_by=[4, 128]
)

# AVOID: Overly generic constraints
ArrayConstraint(
    dtype='float32',
    ndim=2
    # No additional constraints
)
```

### 2. Document Calling Convention

Always document the expected C/C++ calling convention:

```python
"""
Kernel: tiled_matmul

Calling Convention: cutile_python_v1

Arguments:
1. a: ptr, shape[2], strides[2]  (float32, 2D)
2. b: ptr, shape[2], strides[2]  (float32, 2D)
3. c: ptr, shape[2], strides[2]  (float32, 2D)

Constants (baked in):
- BLOCK_SIZE = 128
- BLOCK_M = 64
- BLOCK_N = 64
- BLOCK_K = 32

Launch config:
- Grid: (M / BLOCK_M, N / BLOCK_N)
- Block: BLOCK_SIZE threads
"""
@ct.kernel
def tiled_matmul(...):
    pass
```

### 3. Version Control CUBIN Files

Track CUBIN files in version control for reproducibility:

```bash
# Add CUBIN to git
git add tiled_matmul.cubin
git commit -m "Add compiled kernel for sm_80"

# Include in releases
# Users can deploy without Python/cuTile
```

### 4. Validate Signatures

Test signatures with JIT before exporting:

```python
import torch

# Test with example data
a = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
b = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
c = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')

# Launch with JIT to verify correctness
ct.launch(tiled_matmul, a, b, c)

# Now export with confidence
export_kernel(...)
```

### 5. Handle Multiple dtypes

Export separate kernels for different data types:

```python
signatures = [
    # FP32 version
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype='float32', ndim=2),
            ArrayConstraint(dtype='float32', ndim=2),
            ArrayConstraint(dtype='float32', ndim=2),
        ],
        calling_convention='cutile_python_v1',
        symbol='matmul_f32'
    ),
    # FP16 version
    KernelSignature(
        parameters=[
            ArrayConstraint(dtype='float16', ndim=2),
            ArrayConstraint(dtype='float16', ndim=2),
            ArrayConstraint(dtype='float16', ndim=2),
        ],
        calling_convention='cutile_python_v1',
        symbol='matmul_f16'
    ),
]

with open('matmul.cubin', 'wb') as f:
    export_kernel(
        kernel=matmul,
        signatures=signatures,
        output_file=f,
        gpu_code='sm_80',
        output_format='cubin'
    )
```

## Summary

AOT compilation with `export_kernel()` provides:

- **Deployment flexibility**: Distribute kernels without Python
- **Performance control**: Manual specialization for specific use cases
- **Integration**: Load from C/C++ applications
- **Reproducibility**: Pre-compiled binaries for version control

**Key workflow:**

1. Define kernel with `@ct.kernel`
2. Build `KernelSignature` objects with `ParameterConstraint`s
3. Call `export_kernel()` to generate CUBIN or bytecode
4. Load and launch from C/C++ using `cutile_python_v1` convention

AOT compilation is essential for production deployments where Python dependencies are undesirable or where maximum performance through careful specialization is required.
