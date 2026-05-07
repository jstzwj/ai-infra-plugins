# 09 - Backend Architecture Reference

This document describes the multi-backend architecture of bitsandbytes, covering the dispatch mechanism, backend implementations, custom op definitions, and device-specific details.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Backend Loading and Discovery](#backend-loading-and-discovery)
- [Backend Implementations](#backend-implementations)
- [Dispatch Mechanism](#dispatch-mechanism)
- [Custom Ops Reference](#custom-ops-reference)
- [CUDA-Specific Details](#cuda-specific-details)
- [Device Detection and Feature Flags](#device-detection-and-feature-flags)

---

## Architecture Overview

bitsandbytes uses a multi-backend architecture built on PyTorch's `torch.library` custom operator system. The key design principles are:

1. **Operator definitions** are centralized in `_ops.py` using `torch.library.define()`
2. **Fake tensor implementations** (for `torch.compile` support) are registered alongside the definitions
3. **Backend-specific kernels** are registered via `register_kernel("op_name", "device_type")` in each backend's `ops.py`
4. **Default implementations** (pure PyTorch) serve as fallbacks when no specialized backend is available

```
bitsandbytes/
  _ops.py                    # Custom op definitions + fake impls
  __init__.py                # Backend auto-loading
  functional.py              # High-level API (dispatches to custom ops)
  backends/
    __init__.py              # Empty (backends loaded from __init__.py)
    utils.py                 # Shared utilities (quant tables, device detection)
    default/ops.py           # Pure PyTorch fallback implementations
    cuda/ops.py              # CUDA kernel implementations (csrc via cextension)
    cpu/ops.py               # CPU implementations (AVX512, SYCL)
    triton/ops.py            # Triton kernel implementations
      triton/kernels_4bit.py       # 4-bit quantization Triton kernels
      triton/kernels_8bit_quant.py # 8-bit quantization Triton kernels
      triton/kernels_optim.py      # Optimizer update Triton kernels
    mps/ops.py               # Apple Metal Performance Shaders
    xpu/ops.py               # Intel XPU (oneAPI / SYCL)
    hpu/ops.py               # Intel Gaudi (Habana)
```

---

## Backend Loading and Discovery

### Built-in Backend Loading

Backends are loaded conditionally in `bitsandbytes/__init__.py` based on hardware availability:

```python
# From bitsandbytes/__init__.py

# Always loaded (fallback)
from .backends.cpu import ops as cpu_ops
from .backends.default import ops as default_ops

# Conditionally loaded
if torch.cuda.is_available():
    from .backends.cuda import ops as cuda_ops

if hasattr(torch, "xpu") and torch.xpu.is_available():
    from .backends.xpu import ops as xpu_ops

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    from .backends.mps import ops as mps_ops

if importlib.util.find_spec("habana_frameworks"):
    import habana_frameworks.torch
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        from .backends.hpu import ops as hpu_ops
```

### External Backend Loading via Entry Points

Third-party backends can be registered as Python entry points and are auto-discovered at import time:

```python
def _import_backends():
    """
    Discover and autoload all available backends installed as separate packages.
    Packages with an entrypoint for "bitsandbytes.backends" will be loaded.
    """
    from importlib.metadata import entry_points

    extensions = entry_points(group="bitsandbytes.backends")

    for ext in extensions:
        try:
            entry = ext.load()
            entry()
        except Exception as e:
            raise RuntimeError(
                f"bitsandbytes: failed to load backend {ext.name}: {e}"
            ) from e

_import_backends()
```

To create an external backend, a package registers an entry point in its `pyproject.toml`:

```toml
[project.entry-points."bitsandbytes.backends"]
my_backend = "my_package.backends:register"
```

The entry point function (`register` in this example) should import and call `register_kernel()` to register backend-specific implementations for the custom ops.

---

## Backend Implementations

### Default Backend (`backends/default/ops.py`)

Pure PyTorch fallback implementations that work on any device. These are registered with the `"default"` device type, meaning they apply when no device-specific kernel is available.

**Registered ops:**
- `bitsandbytes::int8_linear_matmul` -- Naive int8 matmul via `torch.matmul(A.float(), B.float().t()).to(torch.int32)`
- `bitsandbytes::int8_linear_matmul.out` -- In-place variant
- `bitsandbytes::int8_mixed_scaled_mm` -- Mixed precision int8 matmul with outlier handling
- `bitsandbytes::int8_scaled_mm` -- Scaled int8 matmul combining int8 matmul + dequantization
- `bitsandbytes::int8_mm_dequant` -- Int32 matmul result dequantization via `A * row_stats * col_stats * 6.200124e-05`
- `bitsandbytes::int8_vectorwise_quant` -- Row-wise int8 quantization with optional outlier detection
- `bitsandbytes::quantize_blockwise` -- Blockwise 8-bit quantization using code lookup
- `bitsandbytes::dequantize_blockwise` -- Blockwise 8-bit dequantization
- `bitsandbytes::quantize_4bit` -- 4-bit quantization (fp4/nf4) with packing
- `bitsandbytes::dequantize_4bit` -- 4-bit dequantization
- `bitsandbytes::gemv_4bit` -- 4-bit matrix-vector multiplication via dequantize + F.linear
- `bitsandbytes::optimizer_update_32bit` -- Universal 32-bit optimizer update with `@torch.compile`

The default optimizer implementation uses a `@_try_torch_compile` decorator, which attempts `torch.compile()` and falls back to eager execution if compilation fails. It supports all optimizer types via a unified kernel with optimizer ID dispatch:

```python
MOMENTUM = 0  # SGD, LARS
RMSPROP = 1
ADAGRAD = 2
ADAM = 3      # Adam, LAMB
LION = 4
ADEMAMIX = 5

name2optimizer_id = {
    "momentum": MOMENTUM,
    "lars": MOMENTUM,
    "rmsprop": RMSPROP,
    "adagrad": ADAGRAD,
    "adam": ADAM,
    "lamb": ADAM,
    "lion": LION,
    "ademamix": ADEMAMIX,
}
```

The 32-bit optimizer has two phases: a precondition step that computes the update norm (for trust-ratio clipping), and the main update step. Lion runs the precondition after the update (since it needs the new momentum), while all other optimizers run it before.

### CUDA Backend (`backends/cuda/ops.py`)

High-performance CUDA kernel implementations that call into the compiled C++ library (`lib` from `cextension`). This is the primary production backend.

**Registered ops:**
- `bitsandbytes::int8_linear_matmul` -- cuBLASLt int8 matmul via `lib.cigemmlt_32()`
- `bitsandbytes::int8_linear_matmul.out` -- In-place variant
- `bitsandbytes::int8_mm_dequant` -- CUDA kernel via `lib.cdequant_mm_int32_fp16()`
- `bitsandbytes::int8_vectorwise_quant` -- CUDA kernel via `lib.cint8_vector_quant()`
- `bitsandbytes::int8_double_quant` -- Combined row+col quantization
- `bitsandbytes::quantize_blockwise` -- CUDA kernels for fp16/bf16/fp32
- `bitsandbytes::dequantize_blockwise` -- CUDA kernels for fp16/bf16/fp32
- `bitsandbytes::dequantize_blockwise.out` -- In-place variant
- `bitsandbytes::quantize_4bit` -- CUDA kernels for fp4/nf4 across fp16/bf16/fp32
- `bitsandbytes::dequantize_4bit` -- CUDA kernels for fp4/nf4 across fp16/bf16/fp32
- `bitsandbytes::dequantize_4bit.out` -- In-place variant
- `bitsandbytes::gemv_4bit` -- CUDA GEMV kernels for fp16/bf16/fp32
- `bitsandbytes::gemv_4bit.out` -- In-place variant
- `bitsandbytes::optimizer_update_32bit` -- CUDA optimizer kernels
- `bitsandbytes::optimizer_update_8bit_blockwise` -- CUDA 8-bit optimizer kernels

**Optimizer kernel dispatch** uses lookup tables mapping optimizer name + gradient dtype to the appropriate C function:

```python
str2optimizer32bit = {
    "adam":     (lib.cadam32bit_grad_fp32, lib.cadam32bit_grad_fp16, lib.cadam32bit_grad_bf16),
    "momentum": (lib.cmomentum32bit_grad_32, lib.cmomentum32bit_grad_16),
    "rmsprop":  (lib.crmsprop32bit_grad_32, lib.crmsprop32bit_grad_16),
    "lion":     (lib.clion32bit_grad_fp32, lib.clion32bit_grad_fp16, lib.clion32bit_grad_bf16),
    "adagrad":  (lib.cadagrad32bit_grad_32, lib.cadagrad32bit_grad_16),
    "lamb":     (lib.cadam32bit_grad_fp32, lib.cadam32bit_grad_fp16, lib.cadam32bit_grad_bf16),
    "ademamix": (lib.cademamix32bit_grad_fp32, lib.cademamix32bit_grad_fp16, lib.cademamix32bit_grad_bf16),
    "lars":     (lib.cmomentum32bit_grad_32, lib.cmomentum32bit_grad_16),
}

str2optimizer8bit_blockwise = {
    "adam":     (lib.cadam_8bit_blockwise_grad_fp32, ..., lib.cadam_8bit_blockwise_grad_bf16),
    "momentum": (lib.cmomentum_8bit_blockwise_grad_fp32, ..., lib.cmomentum_8bit_blockwise_grad_bf16),
    "rmsprop":  (lib.crmsprop_8bit_blockwise_grad_fp32, ..., lib.crmsprop_8bit_blockwise_grad_bf16),
    "lion":     (lib.clion_8bit_blockwise_grad_fp32, ..., lib.clion_8bit_blockwise_grad_bf16),
    "adagrad":  (lib.cadagrad_8bit_blockwise_grad_fp32, ..., lib.cadagrad_8bit_blockwise_grad_bf16),
    "ademamix": (lib.cademamix_8bit_blockwise_grad_fp32, ..., lib.cademamix_8bit_blockwise_grad_bf16),
}
```

The kernel is selected based on gradient dtype: index 0 for float32, index 1 for float16, index 2 for bfloat16.

**Fallback behavior:** When cuBLASLt has no usable algorithm (e.g., small inner dimensions or HIP/ROCm without a matching algorithm), the CUDA backend falls back to fp32 matmul with a `RuntimeWarning`. This is triggered when `lib.cigemmlt_32()` returns error code 100 (`ERR_NOT_IMPLEMENTED`).

**Int8 matmul constraint:** The inner dimension (`lda`) must be divisible by 4 for cuBLASLt int8 support. When not divisible, the fallback path `torch.matmul(B.float(), A.float().t()).to(torch.int32)` is used.

### CPU Backend (`backends/cpu/ops.py`)

CPU implementations using native C++ kernels (via `lib`) and PyTorch operations.

**Key features:**
- `torch._int_mm` for int8 matmul (requires PyTorch >= 2.6 due to an overflow fix in PyTorch PR #136942)
- AVX512BF16 detection via `has_avx512bf16()` for optimized 4-bit GEMV
- Optional `kernels-community/quantization_bitsandbytes` package for fused CPU gemm_4bit_forward
- The 4-bit dequantization on CPU uses dedicated AVX512 kernels for nf4/fp4 with bf16 output; falls back to the default PyTorch implementation for fp16/fp32 output or blocksizes >= 2048

**Registered ops (when native library is available):**
- `bitsandbytes::int8_linear_matmul` -- `torch._int_mm` (PyTorch >= 2.6 only)
- `bitsandbytes::quantize_blockwise` -- C++ kernel via `lib.cquantize_blockwise_cpu_fp32/bf16/fp16`
- `bitsandbytes::dequantize_blockwise` -- C++ kernel via `lib.cdequantize_blockwise_cpu_fp32/bf16/fp16`
- `bitsandbytes::dequantize_4bit` -- C++ kernels for fp4/nf4 across fp32/bf16/fp16
- `bitsandbytes::gemv_4bit` -- AVX512BF16 optimized or C++ fallback (only when AVX512BF16 is available)
- `bitsandbytes::optimizer_update_32bit` -- Pure PyTorch optimizer update for CPU
- `bitsandbytes::optimizer_update_8bit_blockwise` -- Dequant + update + re-quant cycle

The CPU 8-bit optimizer works by dequantizing states to fp32, performing the optimizer update in fp32, then re-quantizing back to uint8. This ensures numerical correctness while still saving memory between optimizer steps. The 32-bit CPU optimizer implements each algorithm directly in PyTorch operations without relying on compiled C++ code.

### Triton Backend (`backends/triton/ops.py`)

Triton kernel implementations that can be used on devices with Triton support (CUDA, XPU with Triton). The Triton backend is not registered as a standalone device backend; instead, it provides kernel functions that are imported and registered by other backends (primarily XPU).

**Sub-modules:**

- `kernels_4bit.py` -- Triton kernels for 4-bit quantization and dequantization
  - `quantize_fp4_blockwise_kernel` -- FP4 quantization with blockwise absmax. Uses hardcoded threshold comparisons for the 16 FP4 levels, then packs two 4-bit values per byte.
  - `quantize_nf4_blockwise_kernel` -- NF4 quantization with binary tree lookup for the 16 NF4 levels
  - `dequant_fp4_kernel` -- FP4 dequantization using `dequantize_fp4_tree()` which decodes each 4-bit value through nested comparisons
  - `dequant_nf4_kernel` -- NF4 dequantization using `dequantize_nf4_tree()` which maps each 4-bit index to its NF4 value
  - `dequant_4bit_kernel` -- Generic 4-bit dequantization using a code lookup table
  - `quantize_4bit_blockwise_kernel` -- Generic 4-bit quantization using binary search against a code table

- `kernels_8bit_quant.py` -- Triton kernels for 8-bit quantization and dequantization
  - `quantize_8bit_blockwise_kernel` -- Binary search quantization against a code table (8 iterations for 256 entries), computes per-block absmax, outputs uint8
  - `dequant_8bit_kernel` -- Table lookup dequantization with block-wise absmax scaling
  - Helper utilities: `quantize_8bit_blockwise_kernel_util`, `dequant_8bit_blockwise_kernel_util`

- `kernels_optim.py` -- Triton optimizer update kernels
  - Optimizer ID mapping identical to the default backend
  - `_optimizer_precondition_2state_32bit` -- Computes update norm for 2-state optimizers (Adam, AdEMAMix)
  - `_optimizer_precondition_1state_32bit` -- Computes update norm for 1-state optimizers (SGD, Lion, RMSprop, Adagrad)
  - `_optimizer_update_32bit` -- Unified 32-bit optimizer update kernel
  - 8-bit optimizer update: dequantize -> update -> re-quantize cycle using the 8-bit quant utilities

**Registered ops (via `triton/ops.py`):**
- `bitsandbytes::quantize_blockwise` -- Triton 8-bit quantization kernel
- `bitsandbytes::dequantize_blockwise` -- Triton 8-bit dequantization
- `bitsandbytes::dequantize_blockwise.out` -- In-place variant
- `bitsandbytes::quantize_4bit` -- Triton 4-bit quantization
- `bitsandbytes::dequantize_4bit` -- Triton 4-bit dequantization
- `bitsandbytes::dequantize_4bit.out` -- In-place variant
- `bitsandbytes::gemv_4bit` -- Dequantize + F.linear via Triton
- `bitsandbytes::optimizer_update_8bit_blockwise` -- Triton 8-bit optimizer
- `bitsandbytes::optimizer_update_32bit` -- Triton 32-bit optimizer

### MPS Backend (`backends/mps/ops.py`)

Apple Metal Performance Shaders backend for Apple Silicon (M1/M2/M3/M4). Uses Metal kernels from `kernels-community/bitsandbytes-mps` via the HuggingFace Kernels Hub.

**Quant type mapping:**
```python
_QUANT_MAP = {"fp4": 1, "nf4": 2}
```

The kernel is lazily loaded via `kernels.get_kernel("kernels-community/bitsandbytes-mps")` on first use.

**Registered ops:**
- `bitsandbytes::quantize_4bit` -- Metal quantization kernel (blocksizes: 64, 128, 256, 512)
- `bitsandbytes::dequantize_4bit` -- Metal dequantization kernel
- `bitsandbytes::dequantize_4bit.out` -- In-place variant
- `bitsandbytes::gemv_4bit` -- Metal GEMV kernel
- `bitsandbytes::gemv_4bit.out` -- In-place variant

Only 4-bit operations are supported on MPS. 8-bit operations, int8 linear operations, and optimizer updates fall back to the default backend.

### XPU Backend (`backends/xpu/ops.py`)

Intel XPU backend for Intel GPUs (Arc, Data Center GPU Max). Uses SYCL kernels from the native library with Triton fallback.

**Loading priority (determined at import time):**
1. If the native SYCL library is available (`lib` is not `ErrorHandlerMockBNBNativeLibrary`):
   - Dequantization ops use SYCL kernels directly
   - Int8 matmul uses `torch._int_mm` (requires PyTorch >= 2.9)
   - Quantization and optimizer ops delegate to Triton kernels
2. If only Triton is available (no native SYCL library), all ops use Triton
3. Otherwise, falls back to default PyTorch implementations

**Registered ops (SYCL path):**
- `bitsandbytes::int8_linear_matmul` -- `torch._int_mm` (PyTorch >= 2.9)
- `bitsandbytes::quantize_blockwise` -- Triton kernel
- `bitsandbytes::dequantize_blockwise` -- SYCL kernel via `lib.cdequantize_blockwise_fp16/bf16/fp32`
- `bitsandbytes::dequantize_blockwise.out` -- In-place variant
- `bitsandbytes::quantize_4bit` -- Triton kernel
- `bitsandbytes::dequantize_4bit` -- SYCL kernel via `lib.cdequantize_blockwise_*_fp4/nf4`
- `bitsandbytes::gemv_4bit` -- SYCL GEMV via `lib.cgemv_4bit_inference_fp16/bf16/fp32`
- `bitsandbytes::gemv_4bit.out` -- In-place variant
- `bitsandbytes::optimizer_update_32bit` -- Triton optimizer kernel
- `bitsandbytes::optimizer_update_8bit_blockwise` -- Triton 8-bit optimizer

The XPU backend uses `torch.accelerator.current_accelerator().type` for device context management (falling back to `"cuda"` as the attribute name for older PyTorch versions), with `torch.xpu` as the accelerator module. Device context is set using `torch_accelerator_module.device(tensor.device)` to ensure kernels run on the correct accelerator.

### HPU Backend (`backends/hpu/ops.py`)

Intel Gaudi (Habana) backend. Minimal implementation that only supports NF4 dequantization.

**Registered ops:**
- `bitsandbytes::dequantize_4bit` -- Uses `torch.ops.hpu.dequantize_nf4()` native Habana op

**Constraints:**
- Only `quant_type="nf4"` is supported (raises `torch._check` error for "fp4")
- Input dtype must be `torch.bfloat16` or `torch.uint8`

**Backward compatibility:** For Gaudi SW versions < 1.22 (detected via `metadata("habana-torch-plugin")`), the 4-bit compression format is reversed using `_reverse_4bit_compress_format()` to match older format conventions. This reverses the high and low nibble ordering within each byte.

---

## Dispatch Mechanism

### Custom Op Registration Flow

1. **Definition:** `torch.library.define("bitsandbytes::op_name", schema)` in `_ops.py` defines the op and its type signature.

2. **Fake implementation:** `@register_fake("bitsandbytes::op_name")` defines the abstract interpretation for `torch.compile()` and shape inference. These use `torch._check()` assertions to validate dtype and shape constraints.

3. **Backend registration:** Each backend's `ops.py` calls `@register_kernel("bitsandbytes::op_name", "device_type")` to register its implementation for a specific device (e.g., "cuda", "cpu", "xpu", "default").

4. **Runtime dispatch:** When `torch.ops.bitsandbytes.op_name(...)` is called, PyTorch automatically selects the appropriate kernel based on the device of the input tensors. If no device-specific kernel is found, the `"default"` kernel is used.

### API Layer

The `functional.py` module provides high-level functions that call the custom ops:

```python
# Example dispatch chain: functional.py -> _ops.py -> backend kernel

# functional.py (high-level API)
def optimizer_update_32bit(optimizer_name, g, p, state1, ...):
    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())
    is_on_gpu([g, p, state1, state2, unorm_vec])
    torch.ops.bitsandbytes.optimizer_update_32bit(
        optimizer_name, g, p, state1, state2, unorm_vec,
        max_unorm, param_norm, beta1, beta2, beta3, alpha,
        eps, weight_decay, step, lr, gnorm_scale, skip_zeros,
    )

# _ops.py (op definition)
torch.library.define(
    "bitsandbytes::optimizer_update_32bit",
    "(str optimizer_name, Tensor(a0!) g, ...) -> ()"
)

# backends/cuda/ops.py (CUDA implementation)
@register_kernel("bitsandbytes::optimizer_update_32bit", "cuda")
def _(optimizer_name, g, p, state1, ...):
    # Selects and calls the appropriate CUDA kernel
```

### PyTorch Version Compatibility

The dispatch mechanism handles PyTorch version differences:

```python
# _ops.py
_IS_TORCH_GTE_24 = hasattr(torch.library, "register_fake")

if _IS_TORCH_GTE_24:
    register_fake = torch.library.register_fake      # PyTorch >= 2.4
    register_kernel = torch.library.register_kernel   # PyTorch >= 2.4
else:
    register_fake = torch.library.impl_abstract       # PyTorch <= 2.3
    register_kernel = torch.library.impl              # PyTorch <= 2.3
```

---

## Custom Ops Reference

All custom ops are defined in `bitsandbytes/_ops.py`. Below is the complete reference for each operation, including its full signature, fake tensor implementation, dtype constraints, and shape rules.

### bitsandbytes::int8_mixed_scaled_mm

**Signature:**
```
(Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB,
 Tensor? outlier_cols=None, Tensor? bias=None) -> (Tensor, Tensor?)
```

**Description:** Int8 mixed-precision scaled matrix multiplication. Handles outlier columns in full precision while computing the rest in int8. Used by LLM.int8() for the mixed-precision decomposition path.

**Fake impl:**
- Output 1 shape: `(*CA.shape[:-1], CB.shape[0])`, dtype matches `A`
- Output 2 shape: dynamic size (outlier activation submatrix)
- Returns `(output, subA)` tuple

**Default kernel behavior:**
1. If `outlier_cols` is present and non-empty, extracts `A[:, outlier_cols]` and dequantizes the corresponding weight columns via `int8_vectorwise_dequant`
2. Computes main output via `int8_scaled_mm(CA, CB, SCA, SCB, bias, dtype=A.dtype)`
3. Adds outlier contribution via `output.addmm(subA, subB)`

### bitsandbytes::int8_scaled_mm

**Signature:**
```
(Tensor A, Tensor B, Tensor row_stats, Tensor col_stats,
 Tensor? bias=None, ScalarType? dtype=None) -> Tensor
```

**Description:** Int8 scaled matrix multiplication with dequantization. Equivalent to `int8_linear_matmul(A, B)` followed by `int8_mm_dequant(result, row_stats, col_stats, dtype, bias)`.

**Fake impl:**
- Output shape: `(*A.shape[:-1], B.shape[0])`
- Output dtype: `dtype` if specified, otherwise `torch.float16`

**Default kernel:** Chains `int8_linear_matmul` and `int8_mm_dequant`.

### bitsandbytes::int8_linear_matmul (with .out variant)

**Signature:**
```
(Tensor A, Tensor B) -> Tensor
(Tensor A, Tensor B, Tensor! out) -> ()    # .out variant
```

**Description:** Pure int8 matrix multiplication producing int32 output. Computes `A @ B^T` where both inputs are int8.

**Fake impl constraints:**
- `A.dtype == torch.int8`
- `B.dtype == torch.int8`
- Output dtype: `torch.int32`
- Output shape: `(*A.shape[:-1], B.shape[0])`

**.out variant constraints:**
- `out.shape == (*A.shape[:-1], B.shape[0])`
- `out.device == A.device`
- `out.dtype == torch.int32`

**CUDA kernel:** Uses cuBLASLt via `lib.cigemmlt_32()`. Falls back to fp32 matmul when:
- Inner dimension (`lda`) is not divisible by 4
- cuBLASLt returns error code 100 (no usable algorithm, seen on some HIP/ROCm configurations)

**CPU kernel (PyTorch >= 2.6):** Uses `torch._int_mm(A.reshape(-1, A.shape[-1]), B.t()).reshape(*A.shape[:-1], B.shape[0])`.

**XPU kernel (PyTorch >= 2.9):** Same `torch._int_mm` approach.

**Default kernel:** `torch.matmul(A.float(), B.float().t()).to(torch.int32)`.

### bitsandbytes::int8_vectorwise_quant

**Signature:**
```
(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor?)
```

**Description:** Row-wise int8 quantization with optional outlier column detection.

**Returns:**
- `out_row`: Quantized int8 tensor, same shape as `A`
- `row_stats`: Per-row absmax values, shape `(prod(A.shape[:-1]),)`, float32
- `outlier_cols`: Column indices with values >= threshold (int64), or None if `threshold=0.0`

**Fake impl:**
- `out_row` shape matches `A`, dtype int8
- `row_stats` shape: `(prod(A.shape[:-1]),)`, dtype float32
- `outlier_cols`: dynamic size when `threshold > 0.0`, None otherwise

**CUDA kernel:** Input must be `torch.float16`. Uses `lib.cint8_vector_quant()`. When outliers are detected and `rows > 1`, outlier columns in the quantized output are zeroed out.

**Default kernel:** Works with any floating-point dtype. Computes row-wise absmax, quantizes via `round(A * 127 / row_stats)`, zeros out outlier columns. Backs up and restores outlier values in the original tensor.

### bitsandbytes::int8_vectorwise_dequant

**Signature:**
```
(Tensor A, Tensor stats) -> Tensor
```

**Description:** Dequantize int8 tensor using per-row statistics.

**Fake impl constraints:**
- `A.dtype == torch.int8`
- Output dtype: `torch.float32`
- Output shape: same as `A`

**Default kernel:** `A * stats.view(-1, 1) * 7.874015718698502e-3` (where `7.874e-3 = 1/127`).

### bitsandbytes::int8_mm_dequant

**Signature:**
```
(Tensor A, Tensor row_stats, Tensor col_stats,
 ScalarType? dtype=None, Tensor? bias=None) -> Tensor
```

**Description:** Dequantize int32 matmul result using row and column statistics. Applies the formula: `output = A * row_stats * col_stats * (1 / 127^2)`.

**Fake impl constraints:**
- `A.dtype == torch.int32`
- Output dtype: `dtype` if specified, otherwise `torch.float16`
- Output shape: same as `A`

**Default kernel:** `A.view(-1, A.shape[-1]) * (row_stats.unsqueeze(-1) * col_stats.unsqueeze(0)) * 6.200124e-05`, with optional bias addition.

**CUDA kernel:** Always computes output in fp16 first via `lib.cdequant_mm_int32_fp16()`, then casts to requested dtype. Supports fused bias addition for fp16 bias only; non-fp16 bias is added separately after the kernel.

### bitsandbytes::int8_double_quant

**Signature:**
```
(Tensor A, float threshold=0.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor?)
```

**Description:** Double quantization: quantizes the same tensor both row-wise and column-wise.

**Returns:**
- `out_row`: Row-wise quantized int8 tensor
- `out_col`: Column-wise quantized int8 tensor
- `row_stats`: Per-row statistics, shape `(prod(A.shape[:-1]),)`, float32
- `col_stats`: Per-column statistics, shape `(A.shape[-1],)`, float32
- `outlier_cols`: Column indices with outliers (int64), or dynamic size

**Fake impl:**
- `out_row`, `out_col`: same shape as `A`, dtype int8
- `row_stats`: `(prod(A.shape[:-1]),)`, float32
- `col_stats`: `(A.shape[-1],)`, float32
- `outlier_cols`: dynamic size

**CUDA kernel:** Uses `int8_vectorwise_quant` for row-wise quantization, then PyTorch for column-wise quantization (compute per-column absmax, mask outliers, quantize via `round(A * 127 / col_stats)`).

### bitsandbytes::quantize_4bit

**Signature:**
```
(Tensor A, int blocksize, str quant_type,
 ScalarType quant_storage) -> (Tensor, Tensor)
```

**Description:** Quantize a tensor to 4-bit format (fp4 or nf4).

**Parameters:**
- `A`: Input tensor (bfloat16, float16, or float32)
- `blocksize`: Block size for per-block absmax computation (valid: 4096, 2048, 1024, 512, 256, 128, 64, 32)
- `quant_type`: `"nf4"` or `"fp4"`
- `quant_storage`: Storage dtype for packed output (affects packing granularity)

**Returns:**
- `out`: Packed 4-bit quantized tensor, shape `((n+1) // (quant_storage.itemsize * 2), 1)`, dtype `quant_storage`
- `absmax`: Per-block scale factors, shape `(num_blocks,)`, float32

**Fake impl constraints:**
- `blocksize` must be a static size (`torch._check_is_size`)
- Output packed shape: `((n+1) // (quant_storage.itemsize * 2), 1)`

**CUDA kernel:** Dispatches to `lib.cquantize_blockwise_{bf16,fp16,fp32}_{fp4,nf4}` based on input dtype and quant type.

**Default kernel:** Scales each block to [-1, 1] by dividing by block absmax, looks up nearest quant level via `argmin` against the NF4/FP4 code table, then packs two 4-bit values per byte using `left << 4 | right`.

**MPS kernel:** Uses Metal kernel from `kernels-community/bitsandbytes-mps`.

### bitsandbytes::dequantize_4bit (with .out variant)

**Signature:**
```
(Tensor A, Tensor absmax, int blocksize, str quant_type,
 int[] shape, ScalarType dtype) -> Tensor
(Tensor A, Tensor absmax, int blocksize, str quant_type,
 int[] shape, ScalarType dtype, Tensor! out) -> ()    # .out variant
```

**Description:** Dequantize a 4-bit quantized tensor back to full precision.

**Parameters:**
- `A`: Packed 4-bit quantized data (uint8 or viewed as uint8)
- `absmax`: Per-block scale factors, float32
- `blocksize`: Must be a valid size
- `quant_type`: `"nf4"` or `"fp4"`
- `shape`: The original tensor shape before quantization
- `dtype`: Output dtype (float16, bfloat16, or float32)

**Fake impl constraints:**
- `blocksize` must be a static size
- Output shape matches `shape`, dtype matches `dtype`
- `.out` variant: `out.shape == shape`, `out.device == A.device`, `out.dtype == dtype`

**CUDA kernel:** Dispatches to `lib.cdequantize_blockwise_{bf16,fp16,fp32}_{fp4,nf4}` based on output dtype and quant type.

**Default kernel:** Unpacks two 4-bit values per byte using bit shifts (`A >> 4` for high nibble, `A & 0xF` for low nibble), looks up quant values in the NF4/FP4 code table (`CODE[quant_type]`), then multiplies by block-wise absmax.

**CPU kernel:** Uses AVX512 C++ kernels when available; falls back to default PyTorch for fp16/fp32 with blocksize >= 2048 or odd last dimension.

**MPS kernel:** Uses Metal kernel from `kernels-community/bitsandbytes-mps`.

**XPU kernel:** Uses SYCL kernels or Triton fallback.

**HPU kernel:** Uses `torch.ops.hpu.dequantize_nf4()` native Habana op. Only supports `"nf4"` quant type and `bfloat16`/`uint8` input dtype.

### bitsandbytes::quantize_blockwise

**Signature:**
```
(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)
```

**Description:** Blockwise 8-bit quantization using a dynamic quantization map (code).

**Parameters:**
- `A`: Input tensor (float16, bfloat16, or float32)
- `code`: Quantization map (lookup table), float32
- `blocksize`: Block size (valid: 4096, 2048, 1024, 512, 256, 128, 64, 32)

**Returns:**
- `out`: Quantized uint8 tensor, same shape as `A`
- `absmax`: Per-block absmax values, shape `(ceil(n / blocksize),)`, float32

**Fake impl constraints:**
- `blocksize` must be a static size
- `code.dtype == torch.float32`
- Output dtype: uint8, same shape as `A`
- `absmax` shape: `(ceil(n / blocksize),)`, float32

**CUDA kernel:** Dispatches to `lib.cquantize_blockwise_{fp16,bf16,fp32}` based on input dtype.

**Default kernel:** Computes per-block absmax, scales to [-1, 1], then finds nearest code entry via `argmin` of absolute differences.

### bitsandbytes::dequantize_blockwise (with .out variant)

**Signature:**
```
(Tensor A, Tensor absmax, Tensor code, int blocksize,
 ScalarType dtype) -> Tensor
(Tensor A, Tensor absmax, Tensor code, int blocksize,
 ScalarType dtype, Tensor! out) -> ()    # .out variant
```

**Description:** Dequantize a blockwise-quantized uint8 tensor back to full precision.

**Fake impl constraints:**
- `A.dtype == torch.uint8`
- Output shape matches `A.shape`
- Output dtype matches `dtype`
- `.out` variant: `out.shape == A.shape`, `out.device == A.device`, `out.dtype == dtype`

**Default kernel:** `code[A.reshape(-1).int()]` performs the lookup, then reshapes into blocks and multiplies by per-block absmax.

**CUDA kernel:** Dispatches to `lib.cdequantize_blockwise_{fp16,bf16,fp32}`.

**CPU kernel:** Dispatches to `lib.cdequantize_blockwise_cpu_{fp32,bf16,fp16}`.

### bitsandbytes::gemv_4bit (with .out variant)

**Signature:**
```
(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code,
 int blocksize) -> Tensor
(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code,
 int blocksize, Tensor! out) -> ()    # .out variant
```

**Description:** 4-bit quantized matrix-vector multiplication. Computes `A @ dequantize(B)` where B is 4-bit quantized weights.

**Fake impl constraints:**
- `A.numel() == A.size(-1)` (A must be a vector or batch of 1-vectors)
- `A.dtype` in {float16, bfloat16, float32}
- `B.dtype` in {uint8, bfloat16, float16, float32}
- Output shape: `(*A.shape[:-1], shapeB[0])`, dtype matches `A.dtype`
- `.out` variant: `out.shape == (*A.shape[:-1], shapeB[0])`, `out.dtype == A.dtype`, `out.device == A.device`

**CUDA kernel:** Dispatches to `lib.cgemm_4bit_inference_naive_{fp16,bf16,fp32}`. Uses specialized GEMV kernels that fuse dequantization with the matrix-vector multiply.

**Default kernel:** Fully dequantizes B using `dequantize_4bit`, then uses `torch.nn.functional.linear(A, B_dq)`.

**MPS kernel:** Uses Metal GEMV kernel from `kernels-community/bitsandbytes-mps`. Quant type is inferred from the code tensor: `"fp4"` if `code[1] > 0`, otherwise `"nf4"`.

**XPU kernel:** Uses SYCL GEMV kernels or Triton fallback.

### bitsandbytes::optimizer_update_32bit

**Signature:**
```
(str optimizer_name, Tensor(a0!) g, Tensor(a1!) p, Tensor(a2!) state1,
 Tensor(a3!)? state2, Tensor(a4!)? unorm_vec,
 float max_unorm, float param_norm,
 float beta1, float beta2, float beta3, float alpha,
 float eps, float weight_decay, int step, float lr,
 float gnorm_scale, bool skip_zeros=False) -> ()
```

**Description:** In-place 32-bit optimizer state update. All tensor arguments are modified in place (indicated by `Tensor(a0!)` alias annotations).

**Fake impl constraints:**
- `g.numel() == p.numel()` (gradient and parameter must have same number of elements)
- `g.dtype` in {float16, bfloat16, float32}
- `g.dtype == p.dtype` (gradient and parameter must have same dtype)

**Supported optimizer names:** `"adam"`, `"momentum"`, `"rmsprop"`, `"lion"`, `"adagrad"`, `"lamb"`, `"lars"`, `"ademamix"`

**Parameters:**
- `state2`: Second state buffer (None for 1-state optimizers: SGD, Lion, RMSprop, Adagrad)
- `unorm_vec`: Update norm tensor for LAMB/LARS trust-ratio clipping
- `max_unorm`: Maximum update norm relative to parameter norm (0.0 disables clipping)
- `param_norm`: Pre-computed parameter norm for trust-ratio computation
- `beta3`, `alpha`: AdEMAMix-specific parameters (unused by other optimizers)
- `gnorm_scale`: Gradient normalization scale factor (typically 1.0)

**Default kernel:** Two-phase update with `@torch.compile` support:
1. Precondition: compute update norm (for trust-ratio clipping when `max_unorm > 0`)
2. Main update: apply optimizer-specific update rules

**CUDA kernel:** Dispatches to C++ kernels via `str2optimizer32bit` lookup table based on optimizer name and gradient dtype.

**CPU kernel:** Pure PyTorch implementation operating in float32.

### bitsandbytes::optimizer_update_8bit_blockwise

**Signature:**
```
(str optimizer_name, Tensor(a0!) g, Tensor(a1!) p, Tensor(a2!) state1,
 Tensor(a3!)? state2,
 float beta1, float beta2, float beta3, float alpha,
 float eps, int step, float lr,
 Tensor(a4!) qmap1, Tensor(a5!)? qmap2,
 Tensor(a6!) absmax1, Tensor(a7!)? absmax2,
 float weight_decay, float gnorm_scale, bool skip_zeros=False) -> ()
```

**Description:** In-place 8-bit blockwise optimizer state update. States are stored as uint8 with per-block absmax scales and quantization maps.

**Fake impl constraints:**
- `g.numel() == p.numel()`
- `g.dtype` in {float16, bfloat16, float32}
- `g.dtype == p.dtype`
- `state1.dtype == torch.uint8`
- `qmap1.dtype == absmax1.dtype == torch.float32`
- If `state2` is provided: `state2.dtype == torch.uint8`, `qmap2.dtype == absmax2.dtype == torch.float32`

**Supported optimizer names:** `"adam"`, `"momentum"`, `"rmsprop"`, `"lion"`, `"adagrad"`, `"ademamix"` (no `"lamb"` or `"lars"` in 8-bit)

**Quantization workflow (default/CPU):**
1. Dequantize uint8 state to fp32 using blockwise absmax and qmap
2. Perform optimizer update in fp32
3. Re-quantize updated state back to uint8

**CUDA kernel:** Single-pass 8-bit update via `str2optimizer8bit_blockwise` lookup table. The C++ kernels handle quantization/dequantization internally for better performance.

---

## CUDA-Specific Details

### CUBLAS_Context Singleton

The `CUBLAS_Context` class (defined in `functional.py`) manages cuBLAS handles for multi-GPU environments:

```python
class CUBLAS_Context:
    _instance = None

    def initialize(self):
        self.context = {}  # Maps device.index -> cuBLAS handle (ct.c_void_p)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def get_context(self, device):
        if device.index not in self.context:
            prev_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            self.context[device.index] = ct.c_void_p(lib.get_context())
            torch.cuda.set_device(prev_device)
        return self.context[device.index]
```

Key behaviors:
- One cuBLAS handle per CUDA device, stored in a dict keyed by `device.index`
- Handles are created lazily on first access per device
- The singleton pattern ensures consistent handle reuse across the application
- Device switching is performed temporarily during handle creation to ensure the cuBLAS handle is associated with the correct device

### Multi-GPU Device Management

The `_cuda_device_of` context manager ensures CUDA kernels execute on the correct device:

```python
# When multiple GPUs are present
if torch.cuda.device_count() > 1:
    def _cuda_device_of(a: torch.Tensor):
        return torch.cuda.device_of(a)
else:
    # Optimization: skip device switching with single GPU
    import contextlib
    def _cuda_device_of(a: torch.Tensor):
        return contextlib.nullcontext()
```

This avoids the overhead of `cudaGetDevice`/`cudaSetDevice` calls when only one GPU is present. All CUDA kernel calls in `backends/cuda/ops.py` are wrapped with `with _cuda_device_of(tensor):`.

### Stream Management

The `_get_tensor_stream` function (in `functional.py`) retrieves the compute stream associated with a tensor's device:

```python
def _get_tensor_stream(tensor: Tensor) -> ct.c_void_p:
    # Returns a raw stream pointer for async kernel calls
    # Supports CUDA and XPU devices
```

All CUDA kernel calls accept a stream parameter to ensure correct async execution ordering and proper synchronization with PyTorch's stream management.

### Pointer Management

The `get_ptr` function extracts a ctypes void pointer from a PyTorch tensor for passing to C++ kernels:

```python
def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
    if A is None:
        return None
    return ct.c_void_p(A.data_ptr())
```

Returns `None` for `None` inputs, allowing optional tensor parameters to be passed cleanly to C++ functions.

### Paged Memory Management

The `GlobalPageManager` singleton manages unified memory allocation for paged optimizers:

```python
class GlobalPageManager:
    _instance = None

    def initialize(self):
        self.paged_tensors = []

    def prefetch_all(self, to_cpu=False):
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)
```

Paged tensors are allocated via `F.get_paged()` which uses CUDA managed memory (`lib.cget_managed_ptr`). Tensors larger than 100,000 elements are eligible for paging; smaller tensors use regular GPU allocation. Paged tensors are prefetched to GPU before each optimizer step via `prefetch_state()`.

---

## Device Detection and Feature Flags

### Supported Devices

```python
# bitsandbytes/__init__.py
supported_torch_devices = {
    "cpu",      # Universal fallback
    "cuda",     # NVIDIA / AMD GPUs (via ROCm/HIP)
    "xpu",      # Intel GPUs (Arc, Data Center Max)
    "hpu",      # Intel Gaudi (Habana)
    "npu",      # Ascend NPU
    "mps",      # Apple Silicon
}
```

This set serves as a signal for downstream integrations (Hugging Face Transformers, Diffusers) to detect bitsandbytes multi-backend support. Libraries check `hasattr(bnb, 'supported_torch_devices')` to determine whether to use bitsandbytes for device-specific quantization.

### Feature Flags

```python
features = {"multi_backend"}
```

The `features` dict is checked by integration code (e.g., Hugging Face Transformers) to determine the capabilities of the installed bitsandbytes version. The `"multi_backend"` flag indicates that bitsandbytes supports multiple hardware backends beyond CUDA.

### Backend Utilities (`backends/utils.py`)

Shared utilities used across backends:

**NF4/FP4 quantization tables:**
```python
_NF4_QUANT_TABLE = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
    -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,
    0.4407, 0.5626, 0.7230, 1.0,
], dtype=torch.float32)

_FP4_QUANT_TABLE = torch.tensor([
    0.0000, 0.0052, 0.6667, 1.0000, 0.3333, 0.5000,
    0.1667, 0.2500, 0.0000, -0.0052, -0.6667, -1.0000,
    -0.3333, -0.5000, -0.1667, -0.2500,
], dtype=torch.float32)

CODE = {"nf4": _NF4_QUANT_TABLE, "fp4": _FP4_QUANT_TABLE}
```

These tables are used by the default and CPU backends for 4-bit quantization lookup. Device placement defaults to XPU if available, otherwise CPU.

**Triton availability check:**
```python
try:
    import triton
    import triton.language as tl
    triton_available = True
except ImportError:
    triton_available = False
```

**Gaudi SW version detection:**
```python
def get_gaudi_sw_version():
    """Returns the installed version of Gaudi SW."""
    try:
        plugin_metadata = metadata("habana-torch-plugin")
        plugin_version = plugin_metadata.get("Version")
        if plugin_version:
            return version.parse(plugin_version)
    except Exception:
        return None

GAUDI_SW_VER = get_gaudi_sw_version()
```

Used by the HPU backend for backward compatibility with older Gaudi firmware versions (< 1.22) where the 4-bit nibble ordering differs.
