# Reference 06: Functional API Reference

This document provides a comprehensive reference for all functions in `bitsandbytes.functional.py`. These functions implement the core quantization, dequantization, and optimization primitives used by the nn modules and optimizer classes.

---

## Table of Contents

- [Constants](#constants)
- [Quantization Maps](#quantization-maps)
- [Block-wise Quantization (8-bit)](#block-wise-quantization-8-bit)
- [4-bit Quantization](#4-bit-quantization)
- [Int8 Operations](#int8-operations)
- [Optimizer Updates](#optimizer-updates)
- [4-bit Matrix Operations](#4-bit-matrix-operations)
- [Integer Matrix Multiplication](#integer-matrix-multiplication)
- [Paged Memory Management](#paged-memory-management)
- [Utility Functions](#utility-functions)
- [Helper Classes](#helper-classes)
- [CPU Weight Conversion](#cpu-weight-conversion)

---

## Constants

### `C = 127.0`

The normalization constant used for int8 quantization. Values are scaled by dividing by `C` during quantization and multiplied by `row_stats / C` during dequantization.

### `FIRST_CUDA_DEVICE = torch.device("cuda", index=0)`

Default CUDA device used as the target for paged memory allocation.

### `name2qmap: dict`

Module-level dictionary mapping quantization map names to their tensor values. Populated lazily on first use:
- `"dynamic"`: Signed dynamic quantization map (8-bit)
- `"udynamic"`: Unsigned dynamic quantization map (8-bit)

---

## Quantization Maps

### `create_linear_map(signed=True, total_bits=8, add_zero=True) -> Tensor`

Creates a linear (uniform) quantization map.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signed` | `bool` | `True` | Whether to include negative values. |
| `total_bits` | `int` | `8` | Number of bits per quantized value. |
| `add_zero` | `bool` | `True` | Whether to include an explicit zero in the map. |

**Returns:** `Tensor[256]` -- A 256-element tensor of linearly spaced quantization values. For fewer than 8 bits, zero-padded to fill 256 entries.

**Algorithm:**
1. Computes `total_values` based on sign and zero requirements.
2. Creates `torch.linspace(sign, 1.0, total_values)`.
3. If there is a gap to 256, inserts zeros in the middle.

---

### `create_normal_map(offset=0.9677083, use_extra_value=True) -> Tensor[256]`

Creates the NormalFloat4 (NF4) quantization map. Each of the 16 quantization bins has approximately equal probability mass under the standard normal distribution N(0, 1).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offset` | `float` | `0.9677083` | The outermost quantile boundary. Controls the range of the normal distribution covered. The default was empirically optimized. |
| `use_extra_value` | `bool` | `True` | If `True`, creates an asymmetric type with 15 non-zero values (8 negative + zero + 9 positive). If `False`, 14 non-zero values (7 negative + zero + 7 positive). |

**Returns:** `Tensor[256]` -- A 256-element tensor where the first 16 entries are sorted NF4 quantization levels normalized to [-1, 1], and the remaining 240 entries are zero.

**Raises:** `ImportError` if `scipy` is not installed.

**Algorithm:**
1. Uses `scipy.stats.norm.ppf()` (inverse CDF) to compute quantile boundaries.
2. Splits positive and negative regions.
3. Sorts all values and normalizes by dividing by the maximum absolute value.

---

### `create_fp8_map(signed=True, exponent_bits=5, precision_bits=2, total_bits=8) -> Tensor[256]`

Creates a floating-point quantization map with configurable bit layout following IEEE 754-like encoding.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signed` | `bool` | `True` | Whether to include a sign bit. |
| `exponent_bits` | `int` | `5` | Number of bits for the exponent field. |
| `precision_bits` | `int` | `2` | Number of mantissa (fraction) bits. |
| `total_bits` | `int` | `8` | Total bits per value. Must equal `sign + exponent + precision`. |

**Returns:** `Tensor[256]` -- A 256-element tensor of sorted quantization levels normalized to [-1, 1].

**Encoding Details:**
- Exponent bias: `2^(exponent_bits - 1)`
- Normal values: `(1 + mantissa) * 2^(exponent - bias - 1)`
- Subnormal values (exponent field = 0): `mantissa * 2^(-bias)`

**Note:** Despite the name, this function handles any total bit width. For FP4, call: `create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4)`.

---

### `create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8) -> Tensor[256]`

Creates the dynamic quantization map used for 8-bit optimizer state quantization. Uses a dynamic exponent with a shrinking fraction as the exponent increases.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signed` | `bool` | `True` | Whether to include negative values. |
| `max_exponent_bits` | `int` | `7` | Maximum number of exponent bits. |
| `total_bits` | `int` | `8` | Total bits per value. |

**Returns:** `Tensor[256]` -- A 256-element float32 tensor of sorted quantization values.

**Algorithm:**
For each exponent level `i` from 0 to `max_exponent_bits - 1`:
1. Computes `fraction_items = 2^(i + non_sign_bits - max_exponent_bits) + 1`.
2. Creates linearly spaced boundaries in [0.1, 1].
3. Computes means of adjacent boundaries.
4. Scales by `10^(-(max_exponent_bits - 1) + i)`.

Reference: [8-Bit Approximations for Parallelism in Deep Learning](https://arxiv.org/abs/1511.04561)

---

### `get_4bit_type(typename, device=None, blocksize=64) -> Tensor[16]`

Returns the hardcoded quantization lookup table for a 4-bit type. Avoids the scipy dependency at runtime.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `typename` | `str` | required | One of `"nf4"`, `"fp4"`, `"int4"`, or `"af4"`. |
| `device` | | `None` | Target device. Defaults to `"cuda"`. |
| `blocksize` | `int` | `64` | Block size (used only for `"af4"` validation). |

**Returns:** `Tensor[16]` -- A 16-element tensor of quantization values normalized to [-1, 1].

**NF4 Values (hardcoded):**
```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

**FP4 Values (hardcoded):**
```
[0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0,
 -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
```
(After normalization to [-1, 1])

**AF4** (AbnormalFloat4) is only supported for `blocksize=64`. Reference: [NF4 Isn't Information Theoretically Optimal](https://arxiv.org/abs/2306.06965)

---

## Block-wise Quantization (8-bit)

### `quantize_blockwise(A, code=None, absmax=None, out=None, blocksize=4096, nested=False) -> (Tensor, QuantState)`

Quantizes a tensor by dividing it into blocks and independently quantizing each block.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` | required | Input tensor. Supports `float16`, `bfloat16`, `float32`. |
| `code` | `Optional[Tensor]` | `None` | Quantization map. Defaults to signed 8-bit dynamic type. |
| `absmax` | `Optional[Tensor]` | `None` | Pre-allocated tensor for absmax values. Overwritten with computed values. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor. Overwritten with quantized data. |
| `blocksize` | `int` | `4096` | Block size. Valid values: 64, 128, 256, 512, 1024, 2048, 4096. |
| `nested` | `bool` | `False` | If `True`, additionally quantizes the absmax values (double quantization). |

**Returns:** `(Tensor, QuantState)` -- A tuple of the quantized uint8 tensor and the quantization state.

**Nested Quantization:** When `nested=True`:
1. Computes the mean of absmax and subtracts it (centering).
2. Recursively calls `quantize_blockwise` on the centered absmax with the same blocksize.
3. Creates a `QuantState` with `offset`, `state2` (nested QuantState for the absmax).

---

### `dequantize_blockwise(A, quant_state=None, absmax=None, code=None, out=None, blocksize=4096, nested=False) -> Tensor`

Dequantizes a block-wise quantized tensor.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` | required | Quantized input tensor (uint8). |
| `quant_state` | `Optional[QuantState]` | `None` | Quantization state from `quantize_blockwise`. Required if `absmax` is not provided. |
| `absmax` | `Optional[Tensor]` | `None` | Scaling values. Required if `quant_state` is not provided. |
| `code` | `Optional[Tensor]` | `None` | Quantization map. Defaults to signed 8-bit dynamic type. Ignored when `quant_state` is provided. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor. |
| `blocksize` | `int` | `4096` | Block size. Valid values: 64, 128, 256, 512, 1024, 2048, 4096. Ignored when `quant_state` is provided. |

**Returns:** `Tensor` -- The dequantized tensor. Datatype is `quant_state.dtype` (defaults to `float32`).

**Nested Dequantization:** When `quant_state.nested` is `True`:
1. Dequantizes the nested absmax using `quant_state.state2`.
2. Adds back the offset: `absmax = dequantize_blockwise(absmax, state2) + offset`.
3. Converts to float32 if needed.

---

## 4-bit Quantization

### `quantize_4bit(A, absmax=None, out=None, blocksize=None, compress_statistics=False, quant_type="fp4", quant_storage=torch.uint8) -> (Tensor, QuantState)`

Quantizes a tensor using 4-bit blockwise quantization (QLoRA algorithm).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` | required | Input tensor. Supports `float16`, `bfloat16`, `float32`. |
| `absmax` | `Optional[Tensor]` | `None` | Pre-allocated tensor for absmax values. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor. |
| `blocksize` | `Optional[int]` | `None` | Block size. Defaults to 64. Valid values: 32, 64, 128, 256, 512, 1024, 2048, 4096. |
| `compress_statistics` | `bool` | `False` | If `True`, additionally quantizes the absmax values using 8-bit blockwise quantization (double quantization). |
| `quant_type` | `str` | `"fp4"` | Quantization type: `"fp4"` or `"nf4"`. |
| `quant_storage` | `torch.dtype` | `torch.uint8` | Storage dtype for the packed 4-bit output. |

**Returns:** `(Tensor, QuantState)` -- Packed 4-bit tensor and quantization state.

**Output Format:** Two 4-bit values are packed into a single byte (high nibble first, low nibble second). The output tensor has shape `((n + 1) // (quant_storage.itemsize * 2), 1)`.

---

### `dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4") -> Tensor`

Dequantizes a packed 4-bit quantized tensor.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` | required | Quantized input tensor. |
| `quant_state` | `Optional[QuantState]` | `None` | Quantization state from `quantize_4bit`. |
| `absmax` | `Optional[Tensor]` | `None` | Scaling values. Required if `quant_state` is `None`. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor. |
| `blocksize` | `Optional[int]` | `None` | Block size. Defaults to 64. |
| `quant_type` | `str` | `"fp4"` | Quantization type: `"fp4"` or `"nf4"`. |

**Returns:** `Tensor` -- The dequantized tensor with shape `quant_state.shape` and dtype `quant_state.dtype`.

**Note:** If the original tensor was transposed (indicated by `A.shape[0] == 1`), the output is transposed back.

---

### `quantize_fp4(A, absmax=None, out=None, blocksize=None, compress_statistics=False, quant_storage=torch.uint8) -> (Tensor, QuantState)`

Convenience wrapper for `quantize_4bit(..., quant_type="fp4")`.

### `quantize_nf4(A, absmax=None, out=None, blocksize=None, compress_statistics=False, quant_storage=torch.uint8) -> (Tensor, QuantState)`

Convenience wrapper for `quantize_4bit(..., quant_type="nf4")`.

### `dequantize_fp4(A, quant_state=None, absmax=None, out=None, blocksize=None) -> Tensor`

Convenience wrapper for `dequantize_4bit(..., quant_type="fp4")`.

### `dequantize_nf4(A, quant_state=None, absmax=None, out=None, blocksize=None) -> Tensor`

Convenience wrapper for `dequantize_4bit(..., quant_type="nf4")`.

---

## Int8 Operations

### `int8_vectorwise_quant(A, threshold=0.0) -> (int8_tensor, float32_stats, Optional[outlier_cols])`

Quantizes a float16 tensor to int8 using row-wise absolute maximum scaling (LLM.int8() algorithm).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` | required | Input tensor. Must be `torch.float16`. |
| `threshold` | `float` | `0.0` | Outlier threshold. Columns with any value >= threshold are excluded from quantization. Set to `0.0` to disable. |

**Returns:** `(Tensor, Tensor, Optional[Tensor])`
- `int8_tensor`: The quantized int8 tensor.
- `float32_stats`: Row-wise absolute maximum values (shape: `[rows]`).
- `Optional[outlier_cols]`: Column indices containing outlier values (int64), or `None` if `threshold=0.0`.

---

### `int8_vectorwise_dequant(A, stats) -> float32_tensor`

Dequantizes an int8 tensor using the provided row-wise statistics.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `A` | `torch.Tensor` (int8) | The quantized int8 tensor. |
| `stats` | `torch.Tensor` (float32) | Row-wise quantization statistics. |

**Returns:** `Tensor` (float32) -- The dequantized tensor.

**Formula:** `output = A * stats.view(-1, 1) * (1/127)`

---

### `int8_double_quant(A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0) -> (CA, CAt, SCA, SCAt, Optional[outlier_cols])`

Performs double quantization: both row-wise and column-wise (transposed) quantization. Primarily used for training.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` (float16) | required | Input matrix. |
| `col_stats` | `Optional[Tensor]` | `None` | Must be `None`. Pre-allocation not supported. |
| `row_stats` | `Optional[Tensor]` | `None` | Must be `None`. Pre-allocation not supported. |
| `out_col` | `Optional[Tensor]` | `None` | Must be `None`. Pre-allocation not supported. |
| `out_row` | `Optional[Tensor]` | `None` | Must be `None`. Pre-allocation not supported. |
| `threshold` | `float` | `0.0` | Outlier threshold for sparse decomposition. |

**Returns:** `(Tensor, Tensor, Tensor, Tensor, Optional[Tensor])`
- `CA`: Row-wise quantized data (int8).
- `CAt`: Column-wise quantized data (int8).
- `SCA`: Row-wise quantization scales (float32).
- `SCAt`: Column-wise quantization scales (float32).
- `Optional[outlier_cols]`: Column indices with outlier values (int32), or `None`.

---

### `int8_linear_matmul(A, B, out=None, dtype=torch.int32) -> int32_tensor`

Performs an int8 matrix multiplication: `out = A @ B.T`. Utilizes integer tensor cores when available.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `torch.Tensor` (int8) | required | First matrix operand. |
| `B` | `torch.Tensor` (int8) | required | Second matrix operand. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor. |
| `dtype` | `torch.dtype` | `torch.int32` | Expected output dtype. |

**Returns:** `Tensor` (int32) -- The matrix multiplication result.

**Note:** If the inner dimension is not divisible by 4, falls back to fp32 matmul (cuBLASLt limitation).

---

### `int8_mm_dequant(A, row_stats, col_stats, out=None, bias=None) -> float16_tensor`

Dequantizes the result of an int8 matrix multiplication.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `A` | `Tensor` (int32) | Result of `int8_linear_matmul`. |
| `row_stats` | `Tensor` (float32) | Row-wise quantization statistics for the LHS operand. |
| `col_stats` | `Tensor` (float32) | Column-wise quantization statistics for the RHS operand. |
| `out` | `Optional[Tensor]` | Pre-allocated output tensor. |
| `bias` | `Optional[Tensor]` | Optional bias vector to add. |

**Returns:** `Tensor` (float16) -- The dequantized result with optional bias.

**Formula:** `output = A * (row_stats * col_stats) * 6.200124e-05` (approximately `1 / (127 * 127)`)

---

## Optimizer Updates

### `optimizer_update_32bit(optimizer_name, g, p, state1, beta1, eps, step, lr, state2=None, beta2=0.0, beta3=0.0, alpha=0.0, weight_decay=0.0, gnorm_scale=1.0, unorm_vec=None, max_unorm=0.0, skip_zeros=False) -> None`

Performs an in-place optimizer update with 32-bit state tensors.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer_name` | `str` | required | Optimizer name: `"adam"`, `"lamb"`, `"momentum"`, `"lars"`, `"lion"`, `"rmsprop"`, `"adagrad"`, `"ademamix"`. |
| `g` | `Tensor` | required | Gradient tensor. |
| `p` | `Tensor` | required | Parameter tensor. |
| `state1` | `Tensor` | required | First optimizer state (e.g., first moment for Adam). |
| `beta1` | `float` | required | First beta value. |
| `eps` | `float` | required | Epsilon for numerical stability. |
| `step` | `int` | required | Current optimizer step. |
| `lr` | `float` | required | Learning rate. |
| `state2` | `Optional[Tensor]` | `None` | Second optimizer state (e.g., second moment for Adam). |
| `beta2` | `float` | `0.0` | Second beta value. |
| `beta3` | `float` | `0.0` | Third beta value (for AdEMAMix). |
| `alpha` | `float` | `0.0` | Alpha scaling (for AdEMAMix). |
| `weight_decay` | `float` | `0.0` | Weight decay coefficient. |
| `gnorm_scale` | `float` | `1.0` | Gradient norm scaling factor. |
| `unorm_vec` | `Optional[Tensor]` | `None` | Update norm tensor for LAMB/LARS. |
| `max_unorm` | `float` | `0.0` | Maximum update norm relative to weight norm. |
| `skip_zeros` | `bool` | `False` | Whether to skip zero-valued gradients. |

---

### `optimizer_update_8bit_blockwise(optimizer_name, g, p, state1, state2, beta1, beta2, beta3, alpha, eps, step, lr, qmap1, qmap2, absmax1, absmax2, weight_decay=0.0, gnorm_scale=1.0, skip_zeros=False) -> None`

Performs an in-place optimizer update with 8-bit blockwise-quantized state tensors.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer_name` | `str` | required | Optimizer name (same as 32-bit version). |
| `g` | `Tensor` | required | Gradient tensor (float16/bfloat16/float32). |
| `p` | `Tensor` | required | Parameter tensor. |
| `state1` | `Tensor` (uint8) | required | First quantized optimizer state. |
| `state2` | `Optional[Tensor]` (uint8) | required | Second quantized optimizer state. |
| `beta1` | `float` | required | First beta value. |
| `beta2` | `float` | required | Second beta value. |
| `beta3` | `float` | required | Third beta value (for AdEMAMix). |
| `alpha` | `float` | required | Alpha scaling (for AdEMAMix). |
| `eps` | `float` | required | Epsilon. |
| `step` | `int` | required | Current step. |
| `lr` | `float` | required | Learning rate. |
| `qmap1` | `Tensor` (float32) | required | Quantization map for state1. |
| `qmap2` | `Optional[Tensor]` (float32) | required | Quantization map for state2. |
| `absmax1` | `Tensor` (float32) | required | Absmax values for state1 blocks. |
| `absmax2` | `Optional[Tensor]` (float32) | required | Absmax values for state2 blocks. |
| `weight_decay` | `float` | `0.0` | Weight decay. |
| `gnorm_scale` | `float` | `1.0` | Gradient norm scaling. |
| `skip_zeros` | `bool` | `False` | Skip zero gradients. |

---

## 4-bit Matrix Operations

### `gemv_4bit(A, B, out=None, transposed_A=False, transposed_B=False, state=None) -> Tensor`

Performs a matrix-vector multiplication with a 4-bit quantized weight matrix. This is the core operation for QLoRA inference.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` | required | Input vector (float16/bfloat16/float32). Shape: `[1, K]` or `[B, 1, K]`. |
| `B` | `Tensor` | required | 4-bit quantized weight matrix. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor. |
| `transposed_A` | `bool` | `False` | Whether A is transposed. |
| `transposed_B` | `bool` | `False` | Whether B is transposed. |
| `state` | `QuantState` | required | Quantization state from `quantize_4bit`. Must not be `None`. |

**Returns:** `Tensor` -- The matrix-vector product.

**Nested State Handling:** If `state.nested` is `True`, the absmax is dequantized from the nested state before the GEMV operation.

---

## Integer Matrix Multiplication

### `igemm(A, B, out=None, transposed_A=False, transposed_B=False) -> int32_tensor`

Performs integer matrix multiplication using cuBLAS. Handles row-major to column-major conversion for cuBLAS compatibility.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` (int8) | required | First matrix operand. |
| `B` | `Tensor` (int8) | required | Second matrix operand. |
| `out` | `Optional[Tensor]` | `None` | Pre-allocated output tensor (int32). |
| `transposed_A` | `bool` | `False` | Whether A is transposed. |
| `transposed_B` | `bool` | `False` | Whether B is transposed. |

**Returns:** `Tensor` (int32) -- The matrix multiplication result.

**Note:** Uses cuBLAS column-major convention internally. Computes `B^T @ A^T = C^T` to handle the row-major to column-major conversion.

---

### `batched_igemm(A, B, out=None, transposed_A=False, transposed_B=False) -> int32_tensor`

Performs batched integer matrix multiplication.

**Parameters:** Same as `igemm`, but both `A` and `B` must be 3-dimensional tensors with matching batch dimension.

**Returns:** `Tensor` (int32) -- The batched matrix multiplication result.

---

## Paged Memory Management

### `get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE) -> paged Tensor`

Allocates a tensor in CUDA unified (managed) memory that can be paged between GPU and CPU.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `*shape` | `int` | required | Shape of the tensor to allocate. |
| `dtype` | `torch.dtype` | `torch.float32` | Data type of the tensor. |
| `device` | | `cuda:0` | Target device. |

**Returns:** A PyTorch tensor backed by CUDA managed memory with additional attributes:
- `is_paged = True`
- `page_deviceid`: The CUDA device index.

**Implementation:** Calls `lib.cget_managed_ptr()` to allocate managed memory via CUDA unified memory, then wraps it in a PyTorch tensor via `numpy.ctypeslib.as_array`.

---

### `prefetch_tensor(A, to_cpu=False) -> None`

Prefetches a paged tensor to GPU (or CPU) before it is needed.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` | required | A paged tensor (must have `is_paged=True`). |
| `to_cpu` | `bool` | `False` | If `True`, prefetches to CPU (-1). Otherwise prefetches to `A.page_deviceid`. |

**Raises:** `AssertionError` if `A.is_paged` is `False`.

---

### `fill(A, value, device=None, prefetch=True) -> None`

Fills a tensor with a scalar value. Supports paged tensors with automatic prefetching and synchronization.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` | required | Target tensor. |
| `value` | `scalar` | required | Value to fill with. |
| `device` | | `None` | Device (currently unused). |
| `prefetch` | `bool` | `True` | Whether to prefetch paged tensors before filling. |

---

## Utility Functions

### `is_on_gpu(tensors) -> True`

Validates that all input tensors are on the same device.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `tensors` | `Iterable[Optional[Tensor]]` | List of tensors to validate. `None` and paged tensors are allowed. |

**Returns:** `Literal[True]`

**Raises:** `RuntimeError` if tensors are on different devices, or if CPU tensors are mixed with GPU tensors.

**Valid configurations:**
- All tensors on CPU.
- All tensors on the same single GPU.
- Paged tensors mixed with GPU tensors.
- `None` entries are ignored.

---

### `get_ptr(A) -> Optional[c_void_p]`

Gets the memory address of the first element of a tensor.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `A` | `Optional[Tensor]` | A PyTorch tensor, or `None`. |

**Returns:** `Optional[c_void_p]` -- A ctypes void pointer to the tensor data, or `None` if `A` is `None`.

---

### `has_avx512bf16() -> bool`

Checks whether the CPU supports AVX-512 BF16 instructions.

**Returns:** `bool` -- `True` if the CPU supports AVX-512 BF16, `False` otherwise. Returns `False` if the native library is not available or the call fails.

---

### `check_matmul(A, B, out, transposed_A, transposed_B, expected_type=torch.int8) -> shape`

Validates tensor shapes for matrix multiplication and computes the expected output shape.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `Tensor` | required | First operand. |
| `B` | `Tensor` | required | Second operand. |
| `out` | `Optional[Tensor]` | required | Output tensor or `None`. |
| `transposed_A` | `bool` | required | Whether A is transposed. |
| `transposed_B` | `bool` | required | Whether B is transposed. |
| `expected_type` | `torch.dtype` | `torch.int8` | Expected dtype for A and B. |

**Returns:** `tuple` -- The expected output shape.

**Raises:** `TypeError` if A or B are not the expected type. `ValueError` if dimensions are incompatible.

---

## Helper Classes

### `GlobalPageManager`

A singleton that tracks all paged tensors in the system.

```python
GlobalPageManager.get_instance() -> GlobalPageManager
```

**Attributes:**
- `paged_tensors: list[Tensor]` -- List of all paged tensors.

**Methods:**

#### `prefetch_all(to_cpu=False) -> None`

Prefetches all tracked paged tensors. Prefetches in reverse order (last added = first used) to handle potential eviction.

---

### `CUBLAS_Context`

A singleton that manages cuBLAS handles per CUDA device.

```python
CUBLAS_Context.get_instance() -> CUBLAS_Context
```

**Attributes:**
- `context: dict[int, c_void_p]` -- Maps device indices to cuBLAS context pointers.

**Methods:**

#### `get_context(device) -> c_void_p`

Returns the cuBLAS context pointer for the given device. Creates a new context if none exists for the device. Temporarily switches to the target device for context creation.

---

### `QuantState`

A container for quantization state components used with `Params4bit` and blockwise quantization.

```python
QuantState(
    absmax: Tensor,
    shape=None,
    code=None,
    blocksize=None,
    quant_type=None,
    dtype=None,
    offset=None,
    state2=None,
)
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `absmax` | `Tensor` | Absolute maximum values per block. |
| `shape` | `Optional[torch.Size]` | Shape of the original (pre-quantization) tensor. |
| `code` | `Optional[Tensor]` | Quantization lookup table. |
| `blocksize` | `Optional[int]` | Block size used for quantization. |
| `quant_type` | `Optional[str]` | Quantization type: `"fp4"` or `"nf4"`. |
| `dtype` | `Optional[torch.dtype]` | Original data type. |
| `offset` | `Optional[Tensor]` | Mean offset (for nested/double quantization). |
| `state2` | `Optional[QuantState]` | Nested quantization state for absmax. |
| `nested` | `bool` | `True` if `state2` is not `None`. |

**Class Constants:**
- `valid_quant_types = ("fp4", "nf4")`
- `valid_qs_type_keys = ["bitsandbytes__fp4", "bitsandbytes__nf4"]`
- `valid_qs_keys = ["absmax", "quant_map", "nested_absmax", "nested_quant_map", "quant_state", "quant_type", "blocksize", "dtype", "shape", "nested_blocksize", "nested_dtype", "nested_offset"]`

**Methods:**

#### `from_dict(qs_dict, device) -> QuantState`

Class method. Reconstructs a `QuantState` from a serialized dictionary (as produced by `as_dict(packed=True)`). Handles both packed and unpacked formats.

#### `as_dict(packed=False) -> dict`

Serializes the quantization state to a dictionary.

- If `packed=False`: Returns a flat dict with both tensor and non-tensor values.
- If `packed=True`: Returns only tensor values, with non-tensor values packed into a single tensor key `"quant_state.bitsandbytes__{quant_type}"` using `pack_dict_to_tensor()`.

#### `to(device) -> None`

Moves all tensor components (`code`, `absmax`, and nested state tensors) to the specified device.

#### `__eq__(other) -> bool`

Compares two `QuantState` objects for equality using `torch.allclose` on tensor attributes.

#### `__getitem__(idx)`

Provides backward-compatible list-style access. Returns elements in the legacy format: `[absmax, shape, dtype, blocksize, [offset, state2], quant_type]`.

#### `__getattr__(name)`

Supports attribute access for packed state_dict keys like `"bitsandbytes__nf4"` (for FSDP compatibility).

---

## CPU Weight Conversion

### `_convert_weight_packed_for_cpu(qweight, quant_state, block_n=32) -> (Tensor, QuantState)`

Converts 4-bit quantized weights into an AVX-512 BF16-friendly packed format for optimized CPU inference.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `qweight` | `Tensor` | required | Packed 4-bit weight tensor. |
| `quant_state` | `QuantState` | required | Associated quantization state. |
| `block_n` | `int` | `32` | Block size for packing. N must be divisible by this value. |

**Returns:** `(Tensor, QuantState)` -- Repacked weight and modified quantization state.

**Side Effects on QuantState:**
- Sets `packing_format_for_cpu = True`.
- Sets `dtype = torch.bfloat16`.
- If nested, dequantizes absmax and removes nesting.
- Transposes and casts absmax to bfloat16.
- Stores `original_dtype`, `original_nested`, `original_qshape` for inverse operation.

---

### `_convert_weight_packed_for_cpu_inverse(packed_weight, quant_state, block_n=32) -> (Tensor, QuantState)`

Reverses the CPU packing format, restoring the original packed layout.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `packed_weight` | `Tensor` (uint8) | required | CPU-packed weight tensor `[N, K/2]`. |
| `quant_state` | `QuantState` | required | Modified quantization state from `_convert_weight_packed_for_cpu`. |
| `block_n` | `int` | `32` | Block size used during packing. |

**Returns:** `(Tensor, QuantState)` -- Original-format weight and partially restored quantization state.

**Raises:** `AssertionError` if `quant_state.packing_format_for_cpu` is `False`.
