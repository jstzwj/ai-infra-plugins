# 09 - Backend Architecture

bitsandbytes uses a multi-backend architecture to support different hardware accelerators. Each backend provides platform-specific implementations of quantization kernels and operations.

## Backend System Overview

### Supported Backends

| Backend | Device Type | Hardware | Status |
|---------|-----------|----------|--------|
| CUDA | `cuda` | NVIDIA GPU (SM60+) | Full support |
| CPU | `cpu` | x86-64 (AVX2/AVX512) | Full support |
| Triton | `cuda` | NVIDIA GPU (alternative path) | Kernels for 4-bit, 8-bit, optimizers |
| MPS | `mps` | Apple Silicon (M1+) | Slow implementation |
| XPU | `xpu` | Intel GPU (Arc, Data Center) | Full support |
| HPU | `hpu` | Intel Gaudi (Gaudi2/3) | LLM.int8() only |

### Backend Loading

Backends are loaded via conditional imports in `bitsandbytes/__init__.py`:

```python
if torch.cuda.is_available():
    from .backends.cuda import ops as cuda_ops

if hasattr(torch, "xpu") and torch.xpu.is_available():
    from .backends.xpu import ops as xpu_ops

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    from .backends.mps import ops as mps_ops

if importlib.util.find_spec("habana_frameworks"):
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .backends.hpu import ops as hpu_ops
```

### External Backend Autoloading

External packages can register backends via Python entry points:

```python
def _import_backends():
    from importlib.metadata import entry_points
    extensions = entry_points(group="bitsandbytes.backends")
    for ext in extensions:
        entry = ext.load()
        entry()
```

This allows third-party hardware vendors to add support without modifying bitsandbytes core.

## Backend Implementation Structure

Each backend lives in `bitsandbytes/backends/<name>/`:

```
backends/
├── __init__.py          # Backend exports
├── utils.py             # Shared utilities
├── default/
│   ├── __init__.py
│   └── ops.py           # Fallback implementations
├── cuda/
│   ├── __init__.py
│   └── ops.py           # CUDA kernel wrappers
├── cpu/
│   ├── __init__.py
│   └── ops.py           # CPU implementations (AVX512)
├── triton/
│   ├── __init__.py
│   ├── ops.py           # Triton kernel wrappers
│   ├── kernels_4bit.py  # 4-bit dequantization kernels
│   ├── kernels_8bit_quant.py  # 8-bit quantization kernels
│   └── kernels_optim.py # Optimizer update kernels
├── mps/
│   ├── __init__.py
│   └── ops.py           # Metal Performance Shaders
├── xpu/
│   ├── __init__.py
│   └── ops.py           # Intel oneAPI/XPU
└── hpu/
    ├── __init__.py
    └── ops.py           # Intel Gaudi/Habana
```

## PyTorch Custom Op System

bitsandbytes registers all operations as PyTorch custom ops via `torch.library`. This enables:
- Backend-specific kernel dispatch
- `torch.compile` support via fake tensor implementations
- Proper autograd integration
- `out=` variant support for in-place operations

### Operation Definitions

All ops are defined in `bitsandbytes/_ops.py`:

```python
import torch.library

# Example: define a custom op
torch.library.define(
    "bitsandbytes::int8_scaled_mm",
    "(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, "
    "Tensor? bias=None, ScalarType? dtype=None) -> Tensor"
)
```

### Fake Tensor Implementations

Each op has a `register_fake` implementation for `torch.compile` tracing:

```python
@register_fake("bitsandbytes::int8_scaled_mm")
def _(A, B, row_stats, col_stats, bias=None, dtype=None):
    shapeC = (*A.shape[:-1], B.shape[0])
    return torch.empty(shapeC, device=A.device, dtype=dtype or torch.float16)
```

These fake implementations provide:
- **Shape inference**: Output shape based on input shapes
- **Dtype inference**: Output dtype based on parameters
- **Constraint checking**: `torch._check()` assertions for dtype, shape, device

### Backend Dispatch

Backend-specific implementations are registered with `register_kernel`:

```python
@register_kernel("bitsandbytes::int8_scaled_mm", "CUDA")
def _(A, B, row_stats, col_stats, bias=None, dtype=None):
    # CUDA-specific implementation
    ...
```

## Complete Custom Op Reference

### 8-bit Quantization Operations

#### `bitsandbytes::int8_vectorwise_quant`
```
(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)
```
- **Input**: float16 tensor A
- **Output**: (int8 quantized, float32 row_stats, optional int64 outlier_cols)
- **Fake**: output shape matches input, stats shape = prod(input.shape[:-1])

#### `bitsandbytes::int8_vectorwise_dequant`
```
(Tensor A, Tensor stats) -> Tensor
```
- **Input**: int8 A, float32 stats
- **Output**: float32 dequantized tensor
- **Default kernel**: `A * stats.view(-1, 1) * 7.874015718698502e-3` (1/127)

#### `bitsandbytes::int8_double_quant`
```
(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)
```
- **Output**: (int8 row-quantized, int8 col-quantized, float32 row_stats, float32 col_stats, optional outlier_cols)

#### `bitsandbytes::int8_linear_matmul`
```
(Tensor A, Tensor B) -> Tensor
(Tensor A, Tensor B, Tensor! out) -> ()  # .out variant
```
- **Constraints**: A and B must be int8, out must be int32
- **Output shape**: (*A.shape[:-1], B.shape[0])

#### `bitsandbytes::int8_mm_dequant`
```
(Tensor A, Tensor row_stats, Tensor col_stats, ScalarType? dtype=None, Tensor? bias=None) -> Tensor
```
- **Constraints**: A must be int32
- **Output**: dequantized result in specified dtype (default float16)

#### `bitsandbytes::int8_scaled_mm`
```
(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats, Tensor? bias=None, ScalarType? dtype=None) -> Tensor
```
- Combined int8 matmul + dequantize + bias

#### `bitsandbytes::int8_mixed_scaled_mm`
```
(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB,
 Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)
```
- Mixed-precision matmul for LLM.int8() outlier handling
- Returns (output, subA) where subA contains outlier activation values

### 4-bit Quantization Operations

#### `bitsandbytes::quantize_4bit`
```
(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)
```
- **Output**: (packed 4-bit tensor, absmax float32)
- **Fake**: out shape = `((n+1) // (quant_storage.itemsize * 2), 1)`

#### `bitsandbytes::dequantize_4bit`
```
(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor
# With .out variant
(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor! out) -> ()
```

#### `bitsandbytes::gemv_4bit`
```
(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize) -> Tensor
# With .out variant
```
- **Constraints**: A must be a vector (A.numel() == A.size(-1))
- A must be float16/bfloat16/float32
- B must be uint8/bfloat16/float16/float32

### Block-wise Quantization Operations

#### `bitsandbytes::quantize_blockwise`
```
(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)
```
- **Output**: (uint8 quantized, float32 absmax)
- **Fake**: absmax shape = `(ceil(n/blocksize),)`

#### `bitsandbytes::dequantize_blockwise`
```
(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype) -> Tensor
# With .out variant
```
- **Constraints**: A must be uint8

### Optimizer Operations

#### `bitsandbytes::optimizer_update_32bit`
```
(str optimizer_name, Tensor g, Tensor p, Tensor state1, Tensor? state2,
 Tensor? unorm_vec, float max_unorm, float param_norm,
 float beta1, float beta2, float beta3, float alpha,
 float eps, float weight_decay, int step, float lr,
 float gnorm_scale, bool skip_zeros=False) -> ()
```
- **Constraints**: g and p same numel, g must be float16/bfloat16/float32, g.dtype == p.dtype

#### `bitsandbytes::optimizer_update_8bit_blockwise`
```
(str optimizer_name, Tensor g, Tensor p, Tensor state1, Tensor? state2,
 float beta1, float beta2, float beta3, float alpha,
 float eps, int step, float lr,
 Tensor qmap1, Tensor? qmap2, Tensor absmax1, Tensor? absmax2,
 float weight_decay, float gnorm_scale, bool skip_zeros=False) -> ()
```
- **Constraints**: state1/state2 must be uint8, qmap/absmax must be float32

## CUDA-Specific Details

### CUBLAS Context Management

```python
class CUBLAS_Context:
    """Singleton managing cuBLAS handles per device."""

    def get_context(self, device):
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]
```

### Multi-GPU Device Management

```python
if torch.cuda.device_count() > 1:
    def _cuda_device_of(a):
        return torch.cuda.device_of(a)
else:
    def _cuda_device_of(a):
        return contextlib.nullcontext()  # No-op for single GPU
```

When only one GPU is present, the overhead of `cudaGetDevice/cudaSetDevice` is skipped entirely.

### CUDA Stream Management

```python
def _get_tensor_stream(tensor):
    if tensor.device.type == "xpu":
        return ct.c_void_p(torch._C._xpu_getCurrentRawStream(tensor.device.index))
    if tensor.device.type == "cuda":
        return ct.c_void_p(torch._C._cuda_getCurrentRawStream(tensor.device.index))
    # For CPU paged states, use current device stream
    ...
```

Raw stream pointers are used for performance when calling C/CUDA kernels.

## Triton Kernels

The Triton backend provides GPU kernels written in OpenAI Triton:

### kernels_4bit.py
- 4-bit dequantization kernels
- Supports NF4 and FP4 types
- Block-wise dequantization with absmax scaling

### kernels_8bit_quant.py
- 8-bit quantization kernels
- Dynamic quantization map support
- Vector-wise quantization

### kernels_optim.py
- 8-bit optimizer update kernels
- Block-wise quantized state management
- Supports multiple optimizer algorithms

## CPU Backend

### AVX512BF16 Support

```python
def has_avx512bf16():
    """Check if CPU supports AVX512BF16 instructions."""
    try:
        return lib.has_avx512bf16_cpu()
    except (AttributeError, RuntimeError, OSError):
        return False
```

When available, uses AVX512BF16 instructions for fast 4-bit GEMV inference on CPU.

### CPU 4-bit Weight Packing

The CPU backend uses a special packed format for 4-bit weights:
- Block size: 32 elements
- Packs 2 nibbles per byte with interleaving
- See `_convert_weight_packed_for_cpu` in functional.py

## Device Detection

```python
features = {"multi_backend"}

supported_torch_devices = {
    "cpu",    # All platforms
    "cuda",   # NVIDIA/AMD GPU
    "xpu",    # Intel GPU
    "hpu",    # Intel Gaudi
    "npu",    # Ascend NPU
    "mps",    # Apple Silicon
}
```

The `features` dict signals to downstream libraries (Transformers, etc.) that bitsandbytes supports multiple backends.

## Compatibility Matrix

| Feature | CUDA | CPU | Triton | XPU | MPS | HPU |
|---------|------|-----|--------|-----|-----|-----|
| LLM.int8() | Full | Full | Partial | Full | Slow | Full |
| QLoRA 4-bit | Full | Full | Kernels | Full | Slow | Partial |
| 8-bit Optimizers | Full | Full | Kernels | Full | No | No |
| gemv_4bit | Full | AVX512 | No | No | No | No |
| Paged Memory | Full | No | No | No | No | No |
| torch.compile | Yes | Yes | Yes | Yes | Yes | Yes |
