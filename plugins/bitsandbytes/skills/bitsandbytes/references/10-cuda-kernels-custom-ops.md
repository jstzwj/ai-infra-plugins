# CUDA/C++ Kernels and Custom Ops Reference

This document provides a comprehensive reference for the CUDA/C++ kernel layer and PyTorch custom operator system in bitsandbytes. It covers the native library structure, kernel categories, the ctypes bridge, the `torch.library` custom op system, fake tensor implementations for `torch.compile`, CUDA stream management, cuBLAS integration, and integer GEMM operations.

---

## Table of Contents

1. [csrc/ Directory Structure](#csrc-directory-structure)
2. [Kernel Categories](#kernel-categories)
3. [C Extension Loading](#c-extension-loading)
4. [ctypes Interface](#ctypes-interface)
5. [PyTorch Custom Op System (_ops.py)](#pytorch-custom-op-system)
6. [Fake Tensor Implementations](#fake-tensor-implementations)
7. [CUDA Stream Management](#cuda-stream-management)
8. [cuBLAS Integration](#cublas-integration)
9. [igemm and Batched igemm](#igemm-and-batched-igemm)

---

## csrc/ Directory Structure

The `csrc/` directory contains all native C++ and CUDA source code compiled into shared libraries via CMake.

### File Inventory

| File | Description |
|------|-------------|
| `CMakeLists.txt` (root) | Top-level CMake build configuration |
| `pythonInterface.cpp` | `extern "C"` bridge functions callable from Python via ctypes |
| `ops.cu` | CUDA kernel implementations (quantize, dequantize, optimizer, GEMM, igemm) |
| `ops.cuh` | Template function declarations for CUDA ops (included by `ops.cu` and `pythonInterface.cpp`) |
| `kernels.cu` | Low-level CUDA kernel device functions (quantize/dequantize blockwise, optimizer updates, 4-bit GEMV, etc.) |
| `kernels.cuh` | Kernel function declarations (templates for blockwise quantize/dequantize, optimizer 32-bit/8-bit, 4-bit GEMV, elementwise ops) |
| `common.cuh` | Architecture constants (compute capability, warp size, BF16 availability, threads per SM) |
| `compat.cuh` | Compatibility macros between CUDA and HIP (ROCm) |
| `compat_device.cuh` | Device-level compatibility helpers |
| `common.h` | Shared C/C++ definitions (CPU-side) |
| `cpu_ops.cpp` | CPU-only implementations of quantize/dequantize/optimizer/GEMV kernels |
| `cpu_ops.h` | CPU ops declarations |
| `xpu_ops.cpp` | Intel XPU (SYCL) kernel implementations |
| `xpu_ops.h` | XPU ops declarations |
| `xpu_kernels.cpp` | XPU kernel implementations |
| `xpu_kernels.h` | XPU kernel declarations |
| `mps_ops.mm` | Apple Metal Performance Shaders (Objective-C++) |
| `mps_kernels.metal` | Metal shader source |

### Build Outputs

The CMake system produces one shared library per backend:

| Backend | Output Library | Notes |
|---------|---------------|-------|
| CUDA | `libbitsandbytes_cudaXY.so` (or `.dll`) | XY = CUDA version (e.g., `118`, `121`) |
| ROCm/HIP | `libbitsandbytes_rocmXY.so` | XY = ROCm version |
| CPU | `libbitsandbytes_cpu.so` | Subset of symbols, no GPU kernels |
| XPU | `libbitsandbytes_xpu.so` | Intel GPU via SYCL |
| MPS | `libbitsandbytes_mps.so` | Apple Silicon GPU |

### CMake Key Variables

```cmake
# Select backend
-DCOMPUTE_BACKEND=cpu|cuda|hip|mps|xpu

# CUDA-specific
-DCUDA_VERSION=121                    # Sanity check against detected CUDA
-DCOMPUTE_CAPABILITY=80;86;89;90      # GPU architectures to target
-DPTXAS_VERBOSE=ON                    # Verbose PTX assembler output

# ROCm-specific
-DROCM_VERSION=70                     # Override ROCm version in library name
-DBNB_ROCM_ARCH=gfx90a;gfx942         # AMD GPU targets
```

### CUDA Architecture Support

The build compiles for all detected architectures by default. Minimum supported is CUDA 11.8 (compute capability 5.0+). The build uses "real" cubin for all architectures except the highest, which also gets PTX for forward compatibility.

Supported architectures include: 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 103, 110, 120, 121.

---

## Kernel Categories

### Quantization Kernels

| Kernel | C Function | Description |
|--------|-----------|-------------|
| Blockwise Quantize (8-bit) | `cquantize_blockwise_{fp16,bf16,fp32}` | Quantizes float values to uint8 using blockwise absmax scaling and a dynamic code map |
| Blockwise Quantize (FP4) | `cquantize_blockwise_{dtype}_fp4` | 4-bit quantization using FP4 encoding (1 sign + 2 exponent + 1 mantissa) |
| Blockwise Quantize (NF4) | `cquantize_blockwise_{dtype}_nf4` | 4-bit quantization using NormalFloat4 (quantile-based from N(0,1)) |
| Int8 Vectorwise Quant | `cint8_vector_quant` | Row-wise int8 quantization with optional outlier threshold for LLM.int8() |
| CPU Blockwise Quantize | `cquantize_blockwise_cpu_{fp32,bf16,fp16}` | CPU fallback for blockwise quantization |

### Dequantization Kernels

| Kernel | C Function | Description |
|--------|-----------|-------------|
| Blockwise Dequantize (8-bit) | `cdequantize_blockwise_{fp16,bf16,fp32}` | Dequantize uint8 to float using code map and absmax |
| Blockwise Dequantize (FP4) | `cdequantize_blockwise_{dtype}_fp4` | Dequantize FP4 packed data back to float |
| Blockwise Dequantize (NF4) | `cdequantize_blockwise_{dtype}_nf4` | Dequantize NF4 packed data back to float |
| Int32-to-FP16 Matmul Dequant | `cdequant_mm_int32_fp16` | Dequantize int32 matmul result using row/col statistics |
| CPU Blockwise Dequantize | `cdequantize_blockwise_cpu_{fp4,nf4}_{fp32,bf16,fp16}` | CPU fallback for 4-bit dequantization |

### Optimizer Update Kernels

| Kernel | C Function | Description |
|--------|-----------|-------------|
| 32-bit 2-state | `c{adam,momentum,rmsprop,lion,ademamix}32bit_grad_{fp32,fp16,bf16}` | 32-bit optimizer with two state tensors (e.g., Adam m and v) |
| 32-bit 1-state | `c{momentum,rmsprop}32bit_grad_{32,16}` | 32-bit optimizer with one state tensor (e.g., SGD momentum) |
| 8-bit Blockwise 2-state | `c{adam,momentum,rmsprop,adagrad,lion,ademamix}_8bit_blockwise_grad_{fp16,bf16,fp32}` | 8-bit quantized optimizer with two state tensors |
| 8-bit Blockwise 1-state | `c{momentum,rmsprop,adagrad}_8bit_blockwise_grad_{fp16,bf16,fp32}` | 8-bit quantized optimizer with one state tensor |

### 4-bit GEMV Kernels

| Kernel | C Function | Description |
|--------|-----------|-------------|
| 4-bit GEMV (FP16) | `cgemm_4bit_inference_naive_fp16` | Matrix-vector multiply with 4-bit weights, fp16 activation |
| 4-bit GEMV (BF16) | `cgemm_4bit_inference_naive_bf16` | Same with bfloat16 activation |
| 4-bit GEMV (FP32) | `cgemm_4bit_inference_naive_fp32` | Same with float32 activation |
| CPU 4-bit GEMV | `gemv_4bit_inference_cpu_{fp4,nf4}_bf16` | CPU GEMV using AVX-512 BF16 instructions |

### 8-bit Matmul Kernels

| Kernel | C Function | Description |
|--------|-----------|-------------|
| igemmlt (32-bit output) | `cigemmlt_32` | Integer GEMM using cuBLASLt, 32-bit accumulator |
| igemmlt (8-bit output) | `cigemmlt_8` | Integer GEMM with 8-bit output |
| igemmlt (8-bit + row scale) | `cigemmlt_8_rowscale` | Integer GEMM with row-wise scaling |

### Elementwise Kernels

| Kernel | C Function | Description |
|--------|-----------|-------------|
| Fill (float) | `cfill_fp32` | Fill tensor with a float value |
| Fill (uint8) | `cfill_uint8` | Fill tensor with a uint8 value |
| Arange | `carange_fp32` | Fill tensor with sequential values |
| Multiply | `c_mul_fp32` | Elementwise multiplication |

---

## C Extension Loading

The native library is loaded at import time by `bitsandbytes/cextension.py`. The module exposes a global `lib` object that serves as the ctypes interface to the compiled shared library.

### Loading Flow

```
bitsandbytes/__init__.py
    -> from .cextension import lib
    -> get_native_library()
        -> Detects CUDA specs via get_cuda_specs()
        -> Resolves library path (CUDA > CPU > XPU)
        -> ct.cdll.LoadLibrary(path)
        -> Wraps in CudaBNBNativeLibrary / BNBNativeLibrary / XpuBNBNativeLibrary
```

### Library Classes

```python
class BNBNativeLibrary:
    """Base class wrapping a ctypes CDLL object."""
    _lib: ct.CDLL
    compiled_with_cuda = False

    def __getattr__(self, name):
        # Cached lookup of C function by name
        fn = getattr(self._lib, name, None)
        if fn is not None:
            return fn
        # Return throw-on-call for missing GPU methods in CPU-only build
        return throw_on_call

class CudaBNBNativeLibrary(BNBNativeLibrary):
    """CUDA library with restype overrides for pointer-returning functions."""
    compiled_with_cuda = True

    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        lib.get_context.restype = ct.c_void_p    # Returns opaque context pointer
        lib.cget_managed_ptr.restype = ct.c_void_p  # Returns managed memory pointer

class XpuBNBNativeLibrary(BNBNativeLibrary):
    """XPU library with SYCL USM paged memory support."""
    def __init__(self, lib: ct.CDLL):
        super().__init__(lib)
        if hasattr(lib, "cget_managed_ptr"):
            lib.cget_managed_ptr.restype = ct.c_void_p
```

### Library Path Resolution

```python
def get_native_library() -> BNBNativeLibrary:
    cuda_specs = get_cuda_specs()
    binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu{DYNAMIC_LIBRARY_SUFFIX}"

    if cuda_specs:
        cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)
        if not cuda_binary_path.exists():
            raise RuntimeError(f"Configured {BNB_BACKEND} binary not found at {cuda_binary_path}")
        binary_path = cuda_binary_path

    if torch._C._has_xpu:
        binary_path = PACKAGE_DIR / f"libbitsandbytes_xpu{DYNAMIC_LIBRARY_SUFFIX}"

    dll = ct.cdll.LoadLibrary(str(binary_path))

    if hasattr(dll, "get_context"):  # Only CUDA-built libraries expose this
        return CudaBNBNativeLibrary(dll)
    if torch._C._has_xpu:
        return XpuBNBNativeLibrary(dll)
    return BNBNativeLibrary(dll)
```

### Environment Variable Overrides

| Variable | Effect |
|----------|--------|
| `BNB_CUDA_VERSION` | Override CUDA version used to locate the library (e.g., `121`) |
| `BNB_ROCM_VERSION` | Override ROCm version used to locate the library (e.g., `602`) |

### Error Handling

When the native library fails to load, `ErrorHandlerMockBNBNativeLibrary` is instantiated instead. This mock defers errors until a native method is actually called, preserving backward compatibility for CPU-only environments. It generates detailed diagnostic messages including available library versions, missing dependencies, and troubleshooting steps.

---

## ctypes Interface

The ctypes bridge uses Python's `ctypes` module to call C functions from the loaded shared library. All tensor data is passed as raw pointers.

### Pointer Management

```python
def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    """Gets the memory address of the first element of a tensor."""
    if A is None:
        return None
    return ct.c_void_p(A.data_ptr())
```

The `data_ptr()` method on PyTorch tensors returns the memory address of the underlying storage as a Python integer. Wrapping it in `ct.c_void_p` makes it suitable for passing to C functions expecting `void*` arguments.

### Common ctypes Types Used

| ctypes Type | C Type | Usage |
|-------------|--------|-------|
| `ct.c_void_p` | `void*` | Tensor data pointers, cuBLAS context handles, CUDA streams |
| `ct.c_size_t` | `size_t` | Byte counts for memory allocation (`cget_managed_ptr`) |
| `ct.c_int32` | `int32_t` | Matrix dimensions (m, n, k), leading dimensions (lda, ldb, ldc) |
| `ct.c_int64` | `int64_t` | Element counts for elementwise operations |
| `ct.c_long` | `long` | Stride parameters for batched GEMM |
| `ct.c_uint32` | `uint32_t` | Batch count for batched GEMM |
| `ct.c_float` | `float` | Scalar parameters (value for fill, beta, eps, lr, etc.) |
| `ct.c_bool` | `bool` | Transpose flags, skip_zeros |
| `ct.POINTER(ct.c_int)` | `int*` | Used for numpy ctypes bridge in `get_paged` |

### Example: Calling cigemm

```python
# From functional.py -> igemm()
ptr = CUBLAS_Context.get_instance().get_context(A.device)

lib.cigemm(
    ptr,                         # Context* context        -> ct.c_void_p
    ct.c_bool(transposed_B),     # bool transposeA         -> ct.c_bool
    ct.c_bool(transposed_A),     # bool transposeB         -> ct.c_bool
    ct.c_int32(m),               # int m                   -> ct.c_int32
    ct.c_int32(n),               # int n                   -> ct.c_int32
    ct.c_int32(k),               # int k                   -> ct.c_int32
    get_ptr(B),                  # void* A                 -> ct.c_void_p
    get_ptr(A),                  # void* B                 -> ct.c_void_p
    get_ptr(out),                # void* C                 -> ct.c_void_p
    ct.c_int32(lda),             # int lda                 -> ct.c_int32
    ct.c_int32(ldb),             # int ldb                 -> ct.c_int32
    ct.c_int32(ldc),             # int ldc                 -> ct.c_int32
)
```

### Example: Calling cprefetch (Paged Memory)

```python
lib.cprefetch(
    get_ptr(A),                  # void* ptr               -> ct.c_void_p
    ct.c_size_t(A.nbytes),       # size_t bytes            -> ct.c_size_t
    ct.c_int32(deviceid),        # int device              -> ct.c_int32 (-1 for CPU)
)
```

### Numpy ctypes Bridge (get_paged)

The `get_paged()` function uses numpy's ctypes integration to convert a raw managed pointer into a PyTorch tensor:

```python
def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    num_bytes = dtype.itemsize * prod(shape)
    managed_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(managed_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out
```

---

## PyTorch Custom Op System

The file `bitsandbytes/_ops.py` defines custom PyTorch operators using `torch.library`. This system enables `torch.compile` tracing via fake tensor implementations (registered with `register_fake`), while dispatching to backend-specific C/CUDA implementations via `register_kernel`.

### Version Compatibility

```python
if hasattr(torch.library, "register_fake"):
    # PyTorch >= 2.4
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    # PyTorch <= 2.3
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl
```

### Operator Definitions

Each operator follows a three-step registration pattern:

1. **`torch.library.define()`** -- Declares the operator signature (schema)
2. **`@register_fake()`** -- Registers shape/dtype inference for `torch.compile` tracing
3. **`@register_kernel()`** (optional) -- Registers a Python fallback implementation

#### int8_mixed_scaled_mm

The core mixed-precision int8 matmul with outlier handling.

```python
torch.library.define(
    "bitsandbytes::int8_mixed_scaled_mm",
    "(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB, "
    "Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)",
)

@register_fake("bitsandbytes::int8_mixed_scaled_mm")
def _(A, CA, CB, SCA, SCB, outlier_cols=None, bias=None):
    shapeC = (*CA.shape[:-1], CB.shape[0])
    out = torch.empty(shapeC, device=A.device, dtype=A.dtype)
    outlier_cols = torch.library.get_ctx().new_dynamic_size()
    subA = A.new_empty(outlier_cols, dtype=torch.int64)
    return out, subA
```

**Signature breakdown:**
- `A`: Original fp16 input (used for outlier extraction)
- `CA`: Quantized int8 version of A
- `CB`: Quantized int8 version of B (weight)
- `SCA`: Row-wise scaling factors for A
- `SCB`: Row-wise scaling factors for B
- `outlier_cols`: Optional indices of outlier columns
- `bias`: Optional bias vector
- Returns: `(output_tensor, subA_matrix)` where subA contains the outlier values

#### int8_scaled_mm

Standard int8 matmul with dequantization.

```python
torch.library.define(
    "bitsandbytes::int8_scaled_mm",
    "(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, "
    "Tensor? bias=None, ScalarType? dtype=None) -> Tensor",
)
```

#### int8_linear_matmul

Pure int8 matrix multiplication producing int32 output.

```python
torch.library.define(
    "bitsandbytes::int8_linear_matmul",
    "(Tensor A, Tensor B) -> Tensor",
)

@register_fake("bitsandbytes::int8_linear_matmul")
def _(A, B):
    torch._check(A.dtype == torch.int8, lambda: "A must be int8")
    torch._check(B.dtype == torch.int8, lambda: "B must be int8")
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=torch.int32)
```

The `.out` variant supports in-place output:

```python
torch.library.define(
    "bitsandbytes::int8_linear_matmul.out",
    "(Tensor A, Tensor B, Tensor! out) -> ()",
)
```

The `Tensor!` annotation indicates the output tensor is modified in-place.

#### int8_vectorwise_quant

Row-wise int8 quantization with optional outlier detection.

```python
torch.library.define(
    "bitsandbytes::int8_vectorwise_quant",
    "(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)",
)

@register_fake("bitsandbytes::int8_vectorwise_quant")
def _(A, threshold=0.0):
    out_row = torch.empty(A.shape, device=A.device, dtype=torch.int8)
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)
    if threshold == 0.0:
        return out_row, row_stats, None
    outlier_cols = torch.library.get_ctx().new_dynamic_size()
    return out_row, row_stats, A.new_empty(outlier_cols, dtype=torch.int64)
```

When `threshold > 0.0`, outlier columns are detected and returned. The `new_dynamic_size()` call tells the compiler that the size of `outlier_cols` is determined at runtime.

#### int8_vectorwise_dequant

Dequantize int8 tensor back to float32. Has a default PyTorch-native implementation:

```python
@register_kernel("bitsandbytes::int8_vectorwise_dequant", "default")
def _(A, stats):
    # Dequantize by multiplying by 1/127
    return A * stats.view(-1, 1) * 7.874015718698502e-3
```

#### int8_mm_dequant

Dequantize int32 matmul result using row and column statistics.

```python
torch.library.define(
    "bitsandbytes::int8_mm_dequant",
    "(Tensor A, Tensor row_stats, Tensor col_stats, "
    "ScalarType? dtype=None, Tensor? bias=None) -> Tensor",
)
```

#### int8_double_quant

Double quantization: both row-wise and column-wise int8 quantization, used in the backward pass of MatMul8bitLt.

```python
torch.library.define(
    "bitsandbytes::int8_double_quant",
    "(Tensor A, float threshold=0.0) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor?)",
)

@register_fake("bitsandbytes::int8_double_quant")
def _(A, threshold=0.0):
    out_row = torch.empty_like(A, dtype=torch.int8)      # Row-wise quantized
    out_col = torch.empty_like(A, dtype=torch.int8)      # Column-wise quantized
    row_stats = torch.empty(prod(A.shape[:-1]), device=A.device, dtype=torch.float32)
    col_stats = torch.empty(A.shape[-1], device=A.device, dtype=torch.float32)
    outlier_n = torch.library.get_ctx().new_dynamic_size()
    outlier_cols = A.new_empty(outlier_n, dtype=torch.int64)
    return out_row, out_col, row_stats, col_stats, outlier_cols
```

#### dequantize_4bit / quantize_4bit

4-bit dequantization and quantization operators with both default and `.out` variants.

```python
torch.library.define(
    "bitsandbytes::dequantize_4bit",
    "(Tensor A, Tensor absmax, int blocksize, str quant_type, "
    "int[] shape, ScalarType dtype) -> Tensor",
)

torch.library.define(
    "bitsandbytes::quantize_4bit",
    "(Tensor A, int blocksize, str quant_type, "
    "ScalarType quant_storage) -> (Tensor, Tensor)",
)
```

The quantize fake implementation computes the expected output shape:

```python
@register_fake("bitsandbytes::quantize_4bit")
def _(A, blocksize, quant_type, quant_storage):
    n = A.numel()
    blocks = -(n // -blocksize)  # Ceiling division
    absmax = torch.empty((blocks,), device=A.device, dtype=torch.float32)
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1),
                       device=A.device, dtype=quant_storage)
    return out, absmax
```

The packed output tensor has shape `((n + 1) // (storage_bytes * 2), 1)` because each storage element holds two 4-bit values.

#### dequantize_blockwise / quantize_blockwise

Blockwise 8-bit quantization/dequantization.

```python
torch.library.define(
    "bitsandbytes::dequantize_blockwise",
    "(Tensor A, Tensor absmax, Tensor code, int blocksize, "
    "ScalarType dtype) -> Tensor",
)

torch.library.define(
    "bitsandbytes::quantize_blockwise",
    "(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)",
)
```

#### gemv_4bit

4-bit matrix-vector multiplication for fast inference.

```python
torch.library.define(
    "bitsandbytes::gemv_4bit",
    "(Tensor A, Tensor B, int[] shapeB, Tensor absmax, "
    "Tensor code, int blocksize) -> Tensor",
)

@register_fake("bitsandbytes::gemv_4bit")
def _(A, B, shapeB, absmax, code, blocksize):
    torch._check_is_size(blocksize)
    torch._check(A.numel() == A.size(-1),
                 lambda: f"A must be a vector with leading dimensions of 1, got {A.shape}")
    torch._check(A.dtype in [torch.float16, torch.bfloat16, torch.float32], ...)
    torch._check(B.dtype in [torch.uint8, torch.bfloat16, torch.float16, torch.float32], ...)
    shape = (*A.shape[:-1], shapeB[0])
    return torch.empty(shape, device=A.device, dtype=A.dtype)
```

Constraints enforced:
- A must be a vector (all leading dimensions are 1)
- A must be float16, bfloat16, or float32
- B must have uint8, bfloat16, float16, or float32 storage

#### optimizer_update_32bit

```python
torch.library.define(
    "bitsandbytes::optimizer_update_32bit",
    "(str optimizer_name, Tensor(a0!) g, Tensor(a1!) p, "
    "Tensor(a2!) state1, Tensor(a3!)? state2, Tensor(a4!)? unorm_vec, "
    "float max_unorm, float param_norm, float beta1, float beta2, "
    "float beta3, float alpha, float eps, float weight_decay, "
    "int step, float lr, float gnorm_scale, bool skip_zeros=False) -> ()",
)
```

The `Tensor(a0!)` annotations indicate that the tensors are mutated in-place. The `!` suffix denotes mutation; the `(a0)` alias set groups them for analysis.

#### optimizer_update_8bit_blockwise

```python
torch.library.define(
    "bitsandbytes::optimizer_update_8bit_blockwise",
    "(str optimizer_name, Tensor(a0!) g, Tensor(a1!) p, "
    "Tensor(a2!) state1, Tensor(a3!)? state2, "
    "float beta1, float beta2, float beta3, float alpha, float eps, "
    "int step, float lr, Tensor(a4!) qmap1, Tensor(a5!)? qmap2, "
    "Tensor(a6!) absmax1, Tensor(a7!)? absmax2, "
    "float weight_decay, float gnorm_scale, bool skip_zeros=False) -> ()",
)
```

### .out Variants for In-Place Operations

Several operators have `.out` variants that write into a pre-allocated output tensor instead of allocating a new one:

- `bitsandbytes::int8_linear_matmul.out`
- `bitsandbytes::dequantize_4bit.out`
- `bitsandbytes::dequantize_blockwise.out`
- `bitsandbytes::gemv_4bit.out`

These follow the PyTorch convention: `(Tensor A, Tensor B, ..., Tensor! out) -> ()`.

---

## Fake Tensor Implementations

Fake tensor implementations (registered via `register_fake`) enable `torch.compile` to trace through bitsandbytes operations without executing the actual CUDA kernels. They provide:

### Shape Inference

Each fake implementation computes the output shape from the input shapes:

```python
# int8_linear_matmul: (batch, k) x (n, k) -> (batch, n)
shapeC = (*A.shape[:-1], B.shape[0])

# int8_scaled_mm: (batch, k) x (n, k) -> (batch, n)
shapeC = (*A.shape[:-1], B.shape[0])

# quantize_4bit: n elements -> ceil(n/blocksize) absmax values
n = A.numel()
blocks = -(n // -blocksize)

# gemv_4bit: vector A x matrix B -> vector output
shape = (*A.shape[:-1], shapeB[0])
```

### Dtype Checking

Fake implementations enforce dtype constraints using `torch._check`:

```python
# int8_linear_matmul
torch._check(A.dtype == torch.int8, lambda: "A must be int8")
torch._check(B.dtype == torch.int8, lambda: "B must be int8")

# int8_mm_dequant
torch._check(A.dtype == torch.int32, lambda: "A must be int32")

# dequantize_blockwise
torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")

# optimizer_update_32bit
torch._check(g.dtype in [torch.float16, torch.bfloat16, torch.float32], ...)
torch._check(g.dtype == p.dtype, ...)
```

### Device Consistency

Output tensors are created on the same device as input tensors:

```python
out = torch.empty(shapeC, device=A.device, dtype=A.dtype)
```

The `.out` variants also enforce device consistency:

```python
torch._check(out.device == A.device,
             lambda: f"Expected out.device == {A.device}, got {out.device}")
```

### Dynamic Sizes

For outputs whose size depends on runtime data (e.g., the number of outlier columns), `torch.library.get_ctx().new_dynamic_size()` is used:

```python
outlier_cols = torch.library.get_ctx().new_dynamic_size()
return out_row, row_stats, A.new_empty(outlier_cols, dtype=torch.int64)
```

This tells the compiler that the size is not statically determinable.

### Size Validation

The `torch._check_is_size()` function validates that an integer argument represents a valid size (non-negative):

```python
torch._check_is_size(blocksize)
```

---

## CUDA Stream Management

The `_get_tensor_stream()` function retrieves the current CUDA or XPU stream as a `ct.c_void_p` for passing to C kernels.

### Implementation

```python
def _get_tensor_stream(tensor: Tensor) -> ct.c_void_p:
    if tensor.device.type == "xpu":
        return ct.c_void_p(torch._C._xpu_getCurrentRawStream(tensor.device.index))
    if tensor.device.type == "cuda":
        return ct.c_void_p(torch._C._cuda_getCurrentRawStream(tensor.device.index))
    # CPU tensors: use current device's stream
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return ct.c_void_p(torch._C._xpu_getCurrentRawStream(torch.xpu.current_device()))
    return ct.c_void_p(torch._C._cuda_getCurrentRawStream(torch.cuda.current_device()))
```

### Multi-GPU Device Management

When multiple GPUs are present, the `_cuda_device_of()` context manager switches to the correct device before kernel invocation:

```python
if torch.cuda.device_count() > 1:
    def _cuda_device_of(a: torch.Tensor):
        return torch.cuda.device_of(a)
else:
    # Single GPU: skip overhead of cudaGetDevice/cudaSetDevice
    import contextlib
    def _cuda_device_of(a: torch.Tensor):
        return contextlib.nullcontext()
```

---

## cuBLAS Integration

bitsandbytes uses cuBLAS (or rocBLAS on HIP) for integer matrix multiplication. The integration is managed through the `CUBLAS_Context` singleton.

### CUBLAS_Context

```python
class CUBLAS_Context:
    _instance = None

    def initialize(self):
        self.context = {}  # Maps device index -> cuBLAS handle (as c_void_p)

    def get_context(self, device):
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]
```

### Context (C++ Side)

On the C++ side, `get_context()` creates a cuBLAS handle:

```cpp
// pythonInterface.cpp
Context* get_context() { return new Context(); }

// ops.cuh
class Context {
  public:
    cublasHandle_t m_handle;
    Context() {
        cublasHandle_t handle;
        cublasCreate_v2(&handle);
        m_handle = handle;
    }
};
```

For HIP/ROCm, `cublasHandle_t` is replaced with `rocblas_handle` via compatibility macros.

### cigemm

`cigemm` performs a single integer GEMM using cuBLAS's `cublasGemmEx` (or rocBLAS equivalent):

```cpp
extern "C" void cigemm(
    Context* context, bool transposeA, bool transposeB,
    int m, int n, int k, void* A, void* B, void* C,
    int lda, int ldb, int ldc
) {
    gemmex(context, transposeA, transposeB, m, n, k, A, B, C, lda, ldb, ldc);
}
```

The `gemmex` template function dispatches to `cublasGemmEx` with int8 data type and int32 accumulation (or the HIP equivalent `rocblas_gemm_ex`).

### cbatched_igemm

Batched integer GEMM for 3D tensor contractions:

```cpp
extern "C" void cbatched_igemm(
    Context* context, bool transposeA, bool transposeB,
    int m, int n, int k, void* A, void* B, void* C,
    int lda, int ldb, int ldc,
    long strideA, long strideB, long strideC, int batchCount
) {
    strided_gemmex(context, transposeA, transposeB, m, n, k,
                   A, B, C, lda, ldb, ldc,
                   strideA, strideB, strideC, batchCount);
}
```

---

## igemm and Batched igemm

The `igemm` and `batched_igemm` functions in `functional.py` are the Python-level wrappers for integer GEMM operations. They handle the row-major vs column-major layout conversion between PyTorch and cuBLAS.

### Layout Handling

PyTorch tensors use **row-major** (C-contiguous) memory layout, while cuBLAS expects **column-major** (Fortran-contiguous) layout. The key insight:

```
Column-major: A @ B = C  :  [m, k] @ [k, n] = [m, n]
Row-major:    B^T @ A^T = C^T  :  [k, m] @ [n, k] = [n, m]
```

Therefore, to compute `C = A @ B` in row-major, we pass `B^T` as the first operand and `A^T` as the second to cuBLAS (which sees them as column-major matrices). The dimensions are swapped accordingly.

### igemm (Python)

```python
def igemm(A, B, out=None, transposed_A=False, transposed_B=False):
    sout = check_matmul(A, B, out, transposed_A, transposed_B)
    if out is None:
        out = torch.zeros(size=sout, dtype=torch.int32, device=A.device)

    # 3D x 3D batched case
    if len(A.shape) == 3 and len(B.shape) == 3:
        if A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
            return batched_igemm(A, B, out)

    # Determine actual transpose from stride analysis
    # ... (stride-based transpose detection)

    # Compute dimensions and leading dimensions for cuBLAS
    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    lib.cigemm(
        ptr,
        ct.c_bool(transposed_B),   # Note: B is passed as first operand
        ct.c_bool(transposed_A),   # A is passed as second operand
        ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
        get_ptr(B),               # First matrix (B^T in cuBLAS column-major)
        get_ptr(A),               # Second matrix (A^T in cuBLAS column-major)
        get_ptr(out),
        ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc),
    )
    return out
```

### Batched igemm (Python)

```python
def batched_igemm(A, B, out=None, transposed_A=False, transposed_B=False):
    # Validates 3D tensors
    # Detects transposition from stride patterns
    # Computes batch strides

    num_batch = A.shape[0]
    n = A.shape[1]
    m = B.shape[2]
    k = B.shape[1]

    strideA = B.shape[1] * B.shape[2]  # Elements between batch slices of B
    strideB = A.shape[1] * A.shape[2]  # Elements between batch slices of A
    strideC = A.shape[1] * B.shape[2]  # Elements between batch slices of C

    ptr = CUBLAS_Context.get_instance().get_context(A.device)

    lib.cbatched_igemm(
        ptr,
        ct.c_bool(transposed_B), ct.c_bool(transposed_A),
        ct.c_int32(m), ct.c_int32(n), ct.c_int32(k),
        get_ptr(B), get_ptr(A), get_ptr(out),
        ct.c_int32(lda), ct.c_int32(ldb), ct.c_int32(ldc),
        ct.c_long(strideA), ct.c_long(strideB), ct.c_long(strideC),
        ct.c_uint32(num_batch),
    )
    return out
```

### Dimension Mapping Reference

For a standard 2D matrix multiplication `C = A @ B` where A is `[batch, m, k]` and B is `[k, n]`:

| Concept | PyTorch (row-major) | cuBLAS (column-major) |
|---------|-------------------|---------------------|
| First operand | B: `[k, n]` | B^T: appears as `[n, k]` column-major |
| Second operand | A: `[batch, m, k]` | A^T: appears as `[k, batch*m]` column-major |
| Output | C: `[batch, m, n]` | C^T: appears as `[n, batch*m]` column-major |
| m parameter | `n` (B's columns) | Same |
| n parameter | `batch * m` | Same |
| k parameter | `k` (B's rows) | Same |

For batched 3D operations where A is `[batch, s, i]` and B is `[batch, s, o]` (contracting over s):

| Parameter | Value |
|-----------|-------|
| m | `o` (B's last dim) |
| n | `i` (A's last dim) |
| k | `batch * s` (flattened batch) |
| strideA | `s * o` |
| strideB | `s * i` |
| strideC | `i * o` |

---

## Architecture Constants (common.cuh)

### Compute Capability Constants

```cpp
#define BNB_CC_PASCAL 600
#define BNB_CC_PASCAL_X2 620
#define BNB_CC_VOLTA 700
#define BNB_CC_VOLTA_XAVIER 720
#define BNB_CC_TURING 750
#define BNB_CC_AMPERE 800
#define BNB_CC_AMPERE2 860
#define BNB_CC_AMPERE2_ORIN 870
#define BNB_CC_ADA 890
#define BNB_CC_HOPPER 900
#define BNB_CC_BLACKWELL 1000
```

### Warp Size

- NVIDIA: 32 (all architectures)
- AMD CDNA (gfx9xx): 64
- AMD RDNA (gfx10xx/11xx/12xx): 32

### Feature Availability

| Feature | NVIDIA Minimum Arch | HIP |
|---------|-------------------|-----|
| FP16 MMA | Volta (SM70) | Not available |
| INT8 MMA | Volta Xavier (SM72) | Not available |
| FP8 | Ada (SM89) | Not available |
| BF16 | Ampere (SM80) | All supported archs |

### Threads Per SM

| Architecture | Max Threads/SM |
|-------------|---------------|
| Turing (SM75) | 1024 |
| Ampere/Ada (SM86-SM89) | 1536 |
| All others | 2048 |
| AMD (CDNA2, RDNA3) | 2048 |

---

## HIP/ROCm Compatibility

The codebase uses extensive macro-based compatibility between CUDA and HIP:

```cpp
// From pythonInterface.cpp
#if BUILD_HIP
#define cudaStream_t hipStream_t
#define __nv_bfloat16 hip_bfloat16
#define cublasLtHandle_t hipblasLtHandle_t
#define cudaMallocManaged hipMallocManaged
#define cudaMemAttachHost hipMemAttachHost
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#endif
```

On the ROCm backend, all CUDA API calls are transparently replaced with HIP equivalents. The `Context` class uses `rocblas_handle` instead of `cublasHandle_t`, and `rocblas_create_handle` instead of `cublasCreate_v2`.
