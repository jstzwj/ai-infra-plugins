# QLoRA 4-bit Quantization System

This document provides a comprehensive reference for the QLoRA 4-bit quantization system, covering all modules, parameters, autograd functions, weight packing, and dispatch logic.

---

## Overview

The QLoRA system ([paper](https://arxiv.org/abs/2305.14314)) provides 4-bit quantized linear layers that reduce model memory by ~8x (from float16 to 4-bit) with minimal accuracy loss. The system supports two 4-bit data types (NF4 and FP4) and optional double quantization of scaling factors.

Key components:
- **`Linear4bit`**: The base 4-bit linear layer module.
- **`LinearFP4` / `LinearNF4`**: Convenience subclasses with fixed quant_type.
- **`Params4bit`**: Custom parameter class with lazy quantization on device transfer.
- **`MatMul4Bit`**: Autograd function for dequantize+matmul path (training/multi-batch).
- **`gemv_4bit`**: Fast single-batch inference kernel (CUDA and packed CPU).
- **`matmul_4bit()`**: Top-level dispatch function.
- **`_convert_weight_packed_for_cpu`**: AVX512BF16 optimization for CPU inference.

---

## Module: `Linear4bit`

```python
class Linear4bit(nn.Linear):
```

The base module for 4-bit quantized linear layers. Extends `torch.nn.Linear`.

### Constructor

```python
def __init__(
    self,
    input_features: int,
    output_features: int,
    bias: bool = True,
    compute_dtype: Optional[torch.dtype] = None,
    compress_statistics: bool = True,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    device = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_features` | `int` | required | Number of input features |
| `output_features` | `int` | required | Number of output features |
| `bias` | `bool` | `True` | Whether to include a bias term |
| `compute_dtype` | `torch.dtype` | `None` | Dtype for computation. If None, auto-detected on first forward pass. |
| `compress_statistics` | `bool` | `True` | Enable double quantization of absmax scaling factors |
| `quant_type` | `str` | `"fp4"` | 4-bit data type: `"fp4"` or `"nf4"` |
| `quant_storage` | `torch.dtype` | `torch.uint8` | Storage dtype for packed 4-bit values |
| `device` | `torch.device` | `None` | Device to initialize on |

#### Internal Initialization

```python
# Weight is wrapped in Params4bit
self.weight = Params4bit(
    self.weight.data,
    requires_grad=False,        # Quantized weights are frozen
    compress_statistics=compress_statistics,
    quant_type=quant_type,
    quant_storage=quant_storage,
    module=self,                # Reference back to this module
)
self.compute_dtype = compute_dtype
self.compute_type_is_set = compute_dtype is not None
self.quant_state = None        # Populated after quantization
self.quant_storage = quant_storage
self.support_avx512bf16_for_cpu = has_avx512bf16()  # Runtime check
```

### Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### Forward Flow

1. **Recover quant state**: Call `fix_4bit_weight_quant_state_from_module(self)` to restore `quant_state` if it was lost (e.g., during FSDP parameter flattening).

2. **CPU AVX512BF16 packing** (optimized path): If all of the following conditions are met:
   - Weight is not already in packed CPU format
   - Input is on CPU
   - AVX512BF16 is supported by the CPU
   - Module is NOT in training mode
   - Input does not require gradients

   Then pack the weight for CPU: `self.weight.data, quant_state = _convert_weight_packed_for_cpu(self.weight.data, quant_state)`.

3. **Bias casting**: If bias exists and its dtype differs from input, cast bias to match input dtype.

4. **Auto-detect compute dtype**: On first forward pass (if `compute_type_is_set` is False), call `set_compute_type(x)` to detect and set the compute dtype.

5. **Cast input to compute dtype**: If `compute_dtype` is set, cast `x` to that dtype.

6. **Prepare weight for matmul**:
   - If in packed CPU format: use `self.weight` as-is (not transposed).
   - Otherwise: use `self.weight.t()` (transpose for `F.linear` convention).

7. **Dispatch to `bnb.matmul_4bit()`**: Compute `bnb.matmul_4bit(x, weight, bias=bias, quant_state=quant_state)`.

8. **Cast output back**: Return output in the original input dtype.

### `set_compute_type(x)`

```python
def set_compute_type(self, x: torch.Tensor) -> None
```

Auto-detects the appropriate compute dtype based on input:

| Input dtype | Behavior |
|-------------|----------|
| `float32`, `bfloat16` | Set `compute_dtype = x.dtype` (safe to compute in natively) |
| `float16` | Keep the user-specified `compute_dtype`. If `compute_dtype` is `None` or `float32`, emit a warning about slow inference (float16 input with float32 compute). |

**Warning examples**:
- Single-batch inference with float16 input and float32 compute: "This will lead to slow inference."
- Multi-batch or training with float16 input and float32 compute: "This will lead to slow inference or training speed."

### `_save_to_state_dict()`

```python
def _save_to_state_dict(self, destination, prefix, keep_vars)
```

Saves the module state for serialization:

1. If weight is in packed CPU format, convert it back: `_convert_weight_packed_for_cpu_inverse(self.weight.data, self.weight.quant_state)`.
2. Save standard weight and bias via `super()._save_to_state_dict()`.
3. Save quant state components via `self.weight.quant_state.as_dict(packed=True)`:
   - Each entry is prefixed with `prefix + "weight."`.
   - In packed format, this produces keys like:
     - `"weight.absmax"` -- float32 tensor
     - `"weight.quant_map"` -- float32 tensor (codebook)
     - `"weight.quant_state.bitsandbytes__nf4"` -- uint8 tensor (packed metadata)
     - `"weight.nested_absmax"` -- uint8 tensor (if double quantized)
     - `"weight.nested_quant_map"` -- float32 tensor (if double quantized)

---

## Convenience Subclasses

### `LinearFP4`

```python
class LinearFP4(Linear4bit):
    def __init__(self, input_features, output_features, bias=True,
                 compute_dtype=None, compress_statistics=True,
                 quant_storage=torch.uint8, device=None):
        super().__init__(input_features, output_features, bias,
                         compute_dtype, compress_statistics,
                         "fp4", quant_storage, device)
```

Equivalent to `Linear4bit(..., quant_type="fp4")`.

### `LinearNF4`

```python
class LinearNF4(Linear4bit):
    def __init__(self, input_features, output_features, bias=True,
                 compute_dtype=None, compress_statistics=True,
                 quant_storage=torch.uint8, device=None):
        super().__init__(input_features, output_features, bias,
                         compute_dtype, compress_statistics,
                         "nf4", quant_storage, device)
```

Equivalent to `Linear4bit(..., quant_type="nf4")`.

---

## Parameter: `Params4bit`

```python
class Params4bit(torch.nn.Parameter):
```

A `torch.nn.Parameter` subclass that handles 4-bit weight quantization with lazy quantization on device transfer.

### Constructor (`__new__`)

```python
def __new__(
    cls,
    data: Optional[torch.Tensor] = None,
    requires_grad: bool = False,       # Quantized weights are frozen by default
    quant_state: Optional[QuantState] = None,
    blocksize: Optional[int] = None,
    compress_statistics: bool = True,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    module: Optional["Linear4bit"] = None,
    bnb_quantized: bool = False,
    **kwargs,
) -> "Params4bit"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `torch.Tensor` | `None` (empty tensor) | The weight data (fp16/bf16 before quantization, packed uint8 after) |
| `requires_grad` | `bool` | `False` | Whether gradients are tracked |
| `quant_state` | `QuantState` | `None` | Existing quantization state (if pre-quantized) |
| `blocksize` | `int` | `None` (defaults to 64) | Quantization block size |
| `compress_statistics` | `bool` | `True` | Enable double quantization |
| `quant_type` | `str` | `"fp4"` | `"fp4"` or `"nf4"` |
| `quant_storage` | `torch.dtype` | `torch.uint8` | Storage dtype for packed values |
| `module` | `Linear4bit` | `None` | Reference to parent module (for quant state sharing) |
| `bnb_quantized` | `bool` | `False` | Whether data is already quantized |

#### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `blocksize` | `int` | Elements per quantization block |
| `compress_statistics` | `bool` | Whether double quantization is enabled |
| `quant_type` | `str` | `"fp4"` or `"nf4"` |
| `quant_state` | `QuantState` or `None` | Quantization metadata (populated after quantization) |
| `quant_storage` | `torch.dtype` | Storage dtype for packed values |
| `bnb_quantized` | `bool` | Whether the data has been quantized |
| `module` | `Linear4bit` or `None` | Reference to parent module |

### `_quantize(device)`

```python
def _quantize(self, device) -> "Params4bit"
```

Called lazily on the first `.to(device)` call. Performs the actual 4-bit quantization.

**Algorithm**:
1. Make data contiguous and move to target device: `w = self.data.contiguous().to(device)`.
2. Call `bnb.functional.quantize_4bit()` with the configured parameters:
   ```python
   w_4bit, quant_state = bnb.functional.quantize_4bit(
       w,
       blocksize=self.blocksize,
       compress_statistics=self.compress_statistics,
       quant_type=self.quant_type,
       quant_storage=self.quant_storage,
   )
   ```
3. Update self: `self.data = w_4bit`, `self.quant_state = quant_state`, `self.bnb_quantized = True`.
4. If a module reference exists, sync the quant state: `self.module.quant_state = quant_state`.

### `cpu()`

```python
def cpu(self) -> "Params4bit"
```

Delegates to `self.to(device="cpu")`.

### `cuda(device=None, non_blocking=False)`

```python
def cuda(self, device=None, non_blocking=False) -> "Params4bit"
```

If the weight is currently in packed CPU format (`packing_format_for_cpu` is True), converts it back to standard format via `_convert_weight_packed_for_cpu_inverse()` before moving to CUDA.

Then delegates to `self.to(device="cuda")`.

### `xpu(device=None, non_blocking=False)`

```python
def xpu(self, device=None, non_blocking=False) -> "Params4bit"
```

Same as `cuda()` but for Intel XPU. Converts packed CPU format back to standard if needed, then delegates to `self.to(device="xpu")`.

### `to()`

```python
def to(self, *args, **kwargs) -> "Params4bit"
```

Handles device movement with **lazy quantization**.

#### Behavior

1. Parse target device/dtype from arguments using `torch._C._nn._parse_to()`.
2. **Lazy quantization**: If `device is not None` and `device.type != "meta"` and `not self.bnb_quantized`, call `self._quantize(device)`. This is the trigger for first-time quantization.
3. **Already quantized**: Create a new `Params4bit` with the same configuration:
   ```python
   new_param = Params4bit(
       super().to(device=device, dtype=dtype, non_blocking=non_blocking),
       requires_grad=self.requires_grad,
       quant_state=self.quant_state,
       blocksize=self.blocksize,
       compress_statistics=self.compress_statistics,
       quant_type=self.quant_type,
       quant_storage=self.quant_storage,
       bnb_quantized=self.bnb_quantized,
   )
   ```
   Also moves `quant_state` tensors to the target device.

### FSDP Properties

These `@property` descriptors proxy attributes from `quant_state` to support PyTorch FSDP state_dict traversal. FSDP's `_get_fqns()` resolves dotted fully-qualified names via `getattr`, so these properties allow `weight.absmax`, `weight.code`, etc. to work correctly.

Using `@property` instead of `__getattr__` avoids `torch.compile` graph breaks (Dynamo can trace descriptor protocol but not `__getattr__` on Tensor subclasses).

| Property | Proxies to | Description |
|----------|-----------|-------------|
| `absmax` | `quant_state.absmax` | Per-block scaling factors |
| `code` | `quant_state.code` | Quantization codebook |
| `quant_map` | `quant_state.code` | Alias for code |
| `offset` | `quant_state.offset` | Mean of absmax (double quant) |
| `state2` | `quant_state.state2` | Nested QuantState |
| `nested_absmax` | `quant_state.state2.absmax` | Double-quantized absmax |
| `nested_blocksize` | `quant_state.state2.blocksize` | Nested block size |
| `nested_quant_map` | `quant_state.state2.code` | Nested codebook |
| `nested_dtype` | `quant_state.state2.dtype` | Nested dtype |
| `nested_offset` | `quant_state.offset` | Offset for double quant |

Each property raises `AttributeError` if the underlying `quant_state` (or `state2` for nested properties) is `None`.

### `__torch_function__()`

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None)
```

Intercepts `torch.chunk()` and `torch.split()` operations to preserve quantization state. When a `Params4bit` is chunked or split (e.g., by FSDP for sharded parameters), the resulting pieces are also `Params4bit` instances with the same quantization configuration.

**Behavior**:
- For `torch.chunk` and `torch.split`: Each resulting chunk is wrapped in a new `Params4bit` with the same `quant_state`, `blocksize`, `compress_statistics`, `quant_type`, `quant_storage`, `module`, and `bnb_quantized`.
- For all other functions: Delegates to `super().__torch_function__()`.

### `from_prequantized()`

```python
@classmethod
def from_prequantized(
    cls,
    data: torch.Tensor,
    quantized_stats: dict[str, Any],
    requires_grad: bool = False,
    device: str = "cuda",
    module: Optional["Linear4bit"] = None,
    **kwargs,
) -> "Params4bit"
```

Creates a `Params4bit` from pre-quantized data (e.g., loaded from a safetensors checkpoint).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `torch.Tensor` | Pre-quantized 4-bit packed weight data |
| `quantized_stats` | `dict` | Quantization state dictionary (from `QuantState.as_dict()`) |
| `requires_grad` | `bool` | Whether to track gradients |
| `device` | `str` | Target device |
| `module` | `Linear4bit` | Parent module reference |

**Algorithm**:
1. Create a `Params4bit` subclass from the data on the target device.
2. Reconstruct `QuantState` from `quantized_stats` via `QuantState.from_dict()`.
3. Copy configuration from the reconstructed state (blocksize, compress_statistics, quant_type).
4. Set `bnb_quantized = True`.
5. Sync quant state to parent module if provided.

### `__deepcopy__()` and `__copy__()`

```python
def __deepcopy__(self, memo) -> "Params4bit"
def __copy__(self) -> "Params4bit"
```

Support Python's copy protocol. Deep copy creates independent copies of both `quant_state` and `data`. Shallow copy shares the same `quant_state`.

---

## Autograd Function: `MatMul4Bit`

```python
class MatMul4Bit(torch.autograd.Function):
```

Autograd function for 4-bit quantized matrix multiplication using the dequantize-then-matmul approach. Used for training and multi-batch inference.

### Forward

```python
@staticmethod
def forward(
    ctx,
    A: torch.Tensor,                              # activations
    B: torch.Tensor,                              # 4-bit quantized weights
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    quant_state: Optional[QuantState] = None,
) -> torch.Tensor
```

#### Forward Algorithm

1. **Empty input check**: If `prod(A.shape) == 0`, return an empty tensor of appropriate shape.

2. **Dequantize + Matmul**:
   ```python
   output = torch.nn.functional.linear(
       A,
       F.dequantize_4bit(B, quant_state).to(A.dtype).t(),
       bias,
   )
   ```
   The dequantized weight is transposed because `F.linear` expects `weight` of shape `[out_features, in_features]`, and the 4-bit weight is stored transposed.

3. **Optional output copy**: If `out` is provided, copy result into it.

4. **Save for backward**: Store `quant_state`, input/output dtypes. If gradients are needed, save `(None, B)`.

### Backward

```python
@staticmethod
def backward(ctx, grad_output: torch.Tensor) -> tuple
```

#### Backward Algorithm

1. **Empty input case**: Return zero tensors.

2. **Bias gradient**: `grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)`.

3. **Activation gradient** (`req_gradA`):
   ```python
   grad_A = torch.matmul(
       grad_output,
       F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t()
   )
   ```
   Dequantize B again and multiply.

> **Note**: `grad_B` is always `None` because `Params4bit` has `requires_grad=False`. The 4-bit weights are frozen during QLoRA training; only the LoRA adapter weights receive gradients.

#### Return Signature

```python
return (grad_A, grad_B, None, grad_bias, None)
# Corresponds to: (A, B, out, bias, quant_state)
```

---

## Fast Inference Kernel: `gemv_4bit()`

```python
def gemv_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    transposed_A: bool = False,
    transposed_B: bool = False,
    state: Optional[QuantState] = None,
) -> torch.Tensor
```

Fast single-batch GeMV (general matrix-vector) kernel for 4-bit inference. This is significantly faster than the dequantize+matmul path because it fuses dequantization with the multiply-accumulate operation.

### Shape Requirements

- **A must be a vector**: `A.numel() == A.shape[-1]` (no batch dimension).
- A must have dtype `float16`, `bfloat16`, or `float32`.
- B must be backed by `uint8` storage (packed 4-bit values).

### Nested Absmax Handling

If the quant state has nested (double-quantized) absmax:
```python
if state.nested:
    absmax = dequantize_blockwise(state.absmax, state.state2) + state.offset
```

This dequantizes the absmax values themselves before using them to dequantize the 4-bit weights.

### Custom Op

Implemented as `bitsandbytes::gemv_4bit`:
```
(Tensor A, Tensor B, int[] shapeB, Tensor absmax, Tensor code, int blocksize) -> Tensor
```

Also has an `.out` variant for in-place computation.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `torch.Tensor` | Input vector, shape `[in_features]` |
| `B` | `torch.Tensor` | Packed 4-bit weight matrix |
| `out` | `torch.Tensor` | Pre-allocated output tensor |
| `transposed_A` | `bool` | Whether A is transposed (unused in practice) |
| `transposed_B` | `bool` | Whether B is transposed (unused in practice) |
| `state` | `QuantState` | **Required**. Quantization state with absmax, code, blocksize. |

### Returns

`torch.Tensor` of shape `[out_features]` with the same dtype as A.

---

## Dispatch Function: `matmul_4bit()`

```python
def matmul_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    quant_state: QuantState,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Top-level dispatch for 4-bit matrix multiplication. Selects the optimal path based on device, batch size, and gradient requirements.

### Dispatch Logic

```
1. CPU with packing_format_for_cpu:
   -> gemv_4bit (fast packed CPU AVX512BF16 kernel)
   -> add bias if present

2. CPU without packing:
   -> MatMul4Bit (dequantize + F.linear)

3. GPU + single-batch (A.numel() == A.shape[-1]) + no grad + blocksize divides hidden_dim:
   -> gemv_4bit (fast CUDA kernel)
   -> add bias if present

4. GPU + training or multi-batch:
   -> MatMul4Bit (dequantize + F.linear)

5. GPU + single-batch but blocksize does not divide hidden_dim:
   -> warn about slow path
   -> MatMul4Bit (dequantize + F.linear)
```

### Detailed Conditions

```python
# CPU path
if A.device.type == "cpu":
    if getattr(quant_state, "packing_format_for_cpu", False):
        # Fast packed AVX512BF16 kernel
        out = gemv_4bit(A, B, out, state=quant_state)
        if bias is not None:
            out += bias
        return out
    else:
        # Standard dequantize path
        return MatMul4Bit.apply(A, B, out, bias, quant_state)

# GPU fast path: single-batch, no grad, blocksize-aligned
if (A.numel() == A.shape[-1]           # single vector, no batch
    and A.requires_grad == False        # inference only
    and A.device.type != "hpu"):        # HPU uses dequantize path
    if A.shape[-1] % quant_state.blocksize != 0:
        # Not blocksize-aligned, warn and fall back
        warn("Some matrices hidden dimension is not a multiple of blocksize...")
        return MatMul4Bit.apply(...)
    else:
        # Fast gemv_4bit kernel
        out = gemv_4bit(A, B.t(), out, state=quant_state)
        if bias is not None:
            out += bias
        return out

# GPU training / multi-batch path
return MatMul4Bit.apply(A, B, out, bias, quant_state)
```

Note: On GPU, `B.t()` is passed to `gemv_4bit` because the weight matrix is stored transposed in memory.

---

## CPU Weight Packing: `_convert_weight_packed_for_cpu()`

```python
def _convert_weight_packed_for_cpu(
    qweight: torch.Tensor,
    quant_state: QuantState,
    block_n: int = 32,
) -> tuple[torch.Tensor, QuantState]
```

Converts 4-bit quantized weights to a packed format optimized for CPU AVX512BF16 GEMM kernels. This conversion happens on the first forward pass when all conditions for AVX512BF16 are met (CPU device, AVX512BF16 supported, not training, no gradients).

### Packing Algorithm

1. **Unpack nibbles**: The input `qweight` of shape `(K*N/2, 1)` in `uint8` is unpacked into individual 4-bit values:
   ```python
   unpacked_w[1::2] = qweight & 0xF    # low nibble
   unpacked_w[::2]  = qweight >> 4      # high nibble
   ```

2. **Reshape to 2D**: `qweight_final = unpacked_w.reshape(N, K)` where N = output features, K = input features.

3. **Block packing** with `BLOCK_N = 32`:
   ```python
   new_shape = [N // BLOCK_N, BLOCK_N, K // 2, 2]
   qw = qweight_final.reshape(new_shape)       # (N/B, B, K/2, 2)
   qw = qw.transpose(-3, -2).contiguous()       # (N/B, K/2, B, 2)
   qw = qw.reshape(-1, BIT_COUNT * 2)           # (-1, 64)
   ```

4. **Combine high and low nibbles**:
   ```python
   high = qw[:, BIT_COUNT:]    # high 32 values
   low = qw[:, :BIT_COUNT]     # low 32 values
   packed = ((high << 4) | low).to(torch.uint8)  # 2 nibbles per byte
   ```

5. **Final shape**: `final_qweight = packed.reshape(N, K // 2)`.

6. **Handle nested absmax**: If double quantized, dequantize the nested absmax:
   ```python
   if quant_state.nested:
       absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
       absmax += quant_state.offset
       quant_state.absmax = absmax
       quant_state.nested = False
   ```

7. **Reshape absmax for CPU kernel**:
   ```python
   quant_state.absmax = (
       quant_state.absmax
       .reshape(quant_state.shape[0], quant_state.shape[1] // quant_state.blocksize)
       .T
       .to(torch.bfloat16)
       .contiguous()
   )
   ```

8. **Update state**:
   ```python
   quant_state.dtype = torch.bfloat16
   quant_state.packing_format_for_cpu = True
   ```

### Block Structure Visualization

```
Original: [N, K] uint8 (each element is a 4-bit index)

After reshaping:
  [N/BLOCK_N, BLOCK_N, K/2, 2]
  Each "2" contains a pair of 4-bit values (low, high)

After transposing and packing:
  [N, K/2] uint8
  Each byte contains 2 nibbles from adjacent positions,
  arranged in BLOCK_N=32 blocks for AVX512BF16 GEMM.
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qweight` | `torch.Tensor` | required | Packed 4-bit weight data, shape `(K*N/2, 1)`, dtype `uint8` |
| `quant_state` | `QuantState` | required | Quantization state to modify |
| `block_n` | `int` | `32` | Block size for packing (must divide N) |

### Returns

`(final_qweight, quant_state)` -- The packed weight and modified quant state.

### Side Effects on `quant_state`

- `quant_state.dtype` changed to `torch.bfloat16`.
- `quant_state.absmax` reshaped and cast to `bfloat16`.
- `quant_state.nested` set to `False` (nested absmax is dequantized and flattened).
- `quant_state.packing_format_for_cpu` set to `True`.
- `quant_state.original_dtype`, `quant_state.original_nested`, `quant_state.original_qshape` stored for inverse conversion.

---

## CPU Weight Unpacking: `_convert_weight_packed_for_cpu_inverse()`

```python
def _convert_weight_packed_for_cpu_inverse(
    packed_weight: torch.Tensor,
    quant_state: QuantState,
    block_n: int = 32,
) -> tuple[torch.Tensor, QuantState]
```

Reverses the CPU packing, converting packed weights back to the standard format. Used when:
1. Saving state dict (to save in a portable format).
2. Moving weights from CPU to GPU (via `Params4bit.cuda()`).

### Unpacking Algorithm

1. **Split packed bytes** into high and low nibbles:
   ```python
   packed = packed_weight.reshape(-1, BIT_COUNT)  # [-1, 32]
   high = (packed >> 4) & 0xF
   low = packed & 0xF
   qw = torch.cat([low, high], dim=-1)  # [-1, 64]
   ```

2. **Reverse transpose and reshape**:
   ```python
   qw = qw.reshape(N // BLOCK_N, K_half, BLOCK_N, 2)
   qw = qw.transpose(-3, -2).contiguous()
   qw = qw.reshape(N, K)
   ```

3. **Repack into standard format** (high nibble << 4 | low nibble):
   ```python
   unpacked_w = qweight.reshape(-1).to(torch.int32)
   high4 = (unpacked_w[::2] & 0xF).to(torch.uint8)
   low4 = (unpacked_w[1::2] & 0xF).to(torch.uint8)
   qweight = (high4 << 4) | low4
   ```

4. **Restore quant state** (best-effort):
   - Reshape absmax back to original layout.
   - If originally nested, re-quantize absmax with `quantize_blockwise()`.
   - Restore `original_dtype` and `packing_format_for_cpu = False`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `packed_weight` | `torch.Tensor` | required | Packed weight, shape `[N, K/2]`, dtype `uint8` |
| `quant_state` | `QuantState` | required | Must have `packing_format_for_cpu = True` |
| `block_n` | `int` | `32` | Block size used in original packing |

### Returns

`(qweight, recovered_state)` -- The unpacked weight in standard format and restored quant state.

---

## `fix_4bit_weight_quant_state_from_module()`

```python
def fix_4bit_weight_quant_state_from_module(
    module: Union["Embedding4bit", "Linear4bit"]
) -> None
```

Recovers the quantization state on the weight parameter from the module's stored quant state. This is necessary in FSDP scenarios where the parameter gets flattened/unflattened and loses its `quant_state` attribute.

### Algorithm

1. If `module.weight.quant_state is not None`: Nothing to do, state already present.
2. If `module.quant_state is None`: Log a warning that the module needs `.cuda()` or `.to(device)` first.
3. Ensure `module.weight.shape[1] == 1` (transposed storage format).
4. If `module.weight` is not a `Params4bit`, wrap it: `Params4bit(module.weight, quant_storage=module.quant_storage, bnb_quantized=True)`.
5. Restore: `module.weight.quant_state = module.quant_state`.

### When This Is Needed

- **FSDP**: When PyTorch FSDP flattens parameters, it creates new Parameter objects that lose the `quant_state` attribute. Since the module stores a reference to the quant state, this function recovers it.
- **Serialization round-trip**: After loading a state dict, the weight parameter may not have its quant state attached.

---

## Quantization Flow: End-to-End

### Loading and Quantizing

```
1. Create Linear4bit module (weights are float16/bfloat16)
   |
2. Load fp16 weights via load_state_dict()
   |  (weights stored in Params4bit.data as fp16)
   |
3. Call module.to("cuda")
   |  Triggers Params4bit.to() -> _quantize()
   |  -> bnb.functional.quantize_4bit()
   |  -> self.data becomes packed uint8
   |  -> self.quant_state populated
   |  -> self.bnb_quantized = True
   |
4. First forward pass
   |  fix_4bit_weight_quant_state_from_module()
   |  set_compute_type(x)
   |  dispatch to matmul_4bit()
```

### Inference Path (Single Batch, No Gradients)

```
Input x [hidden_dim]
   |
   v
matmul_4bit dispatch
   |
   +-- CPU (packed format): gemv_4bit()
   |     Fused dequant + GEMV using AVX512BF16
   |
   +-- GPU (blocksize-aligned): gemv_4bit()
   |     Fused dequant + GeMV using CUDA kernel
   |
   +-- Fallback: MatMul4Bit
         dequantize_4bit() -> F.linear()
```

### Training Path (Multi-Batch or With Gradients)

```
Input x [batch, hidden_dim]
   |
   v
matmul_4bit -> MatMul4Bit.apply()
   |
   Forward:
     dequantize_4bit(B, quant_state) -> W (float16/bfloat16)
     F.linear(A, W.t(), bias) -> output
   |
   Backward:
     grad_bias = grad_output.sum(0)
     grad_A = grad_output @ dequantize_4bit(B, quant_state).t()
     grad_B = None (weights frozen)
```

### Saving and Loading

```
Saving:
  _save_to_state_dict()
    -> _convert_weight_packed_for_cpu_inverse() (if CPU packed)
    -> super()._save_to_state_dict() (weight, bias)
    -> quant_state.as_dict(packed=True)
       -> weight.absmax, weight.quant_map, weight.quant_state.bitsandbytes__nf4
       -> weight.nested_absmax, weight.nested_quant_map (if double quant)

Loading:
  load_state_dict()
    -> standard weight loading
    -> QuantState.from_dict() reconstructs quant state from dict keys
    -> Params4bit receives quant_state on .to(device)
```

---

## Embedding Modules

bitsandbytes also provides 4-bit quantized embedding layers:

### `Embedding4bit`

```python
class Embedding4bit(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, dtype=None,
                 quant_type="fp4", quant_storage=torch.uint8, device=None)
```

Wraps the embedding weight in `Params4bit`. On forward:

1. If `embedding_dim % blocksize == 0`: Uses optimized partial dequantize -- only dequantizes the rows selected by the input indices, not the entire weight matrix.
2. Otherwise: Falls back to full dequantize then standard `F.embedding()`.

### `EmbeddingFP4` / `EmbeddingNF4`

Convenience subclasses with fixed `quant_type`:
```python
EmbeddingFP4(..., quant_type="fp4")
EmbeddingNF4(..., quant_type="nf4")
```

### `Embedding8bit`

```python
class Embedding8bit(nn.Embedding):
```

An 8-bit quantized embedding layer. Uses `Int8Params` for the weight. Forward pass:
1. Looks up int8 rows via `F.embedding()`.
2. Looks up per-row scaling factors via `F.embedding()`.
3. Dequantizes: `output = int8_rows * (row_stats / 127.0)`.

---

## `has_avx512bf16()` -- Runtime CPU Feature Detection

```python
def has_avx512bf16() -> bool
```

Checks whether the CPU supports AVX512_BF16 instructions, which are required for the fast packed CPU inference path.

**Implementation**: Calls `lib.has_avx512bf16_cpu()` from the native library. Returns `False` if the symbol is missing or the call fails.

This is called once during `Linear4bit.__init__()` and stored as `self.support_avx512bf16_for_cpu`.
