# LLM.int8() 8-bit Quantization System

This document provides a comprehensive reference for the LLM.int8() 8-bit quantization system, covering all modules, parameters, autograd functions, and dispatch logic.

---

## Overview

The LLM.int8() system ([paper](https://arxiv.org/abs/2208.07339)) provides 8-bit quantized linear layers that:
1. Store weights in int8 format (50% memory reduction vs fp16).
2. Quantize activations on the fly during forward passes.
3. Optionally handle outlier activation features in fp16 via mixed-precision decomposition.

The key insight is that for large transformer models (>=6.7B parameters), certain hidden dimensions develop systematic large-magnitude outliers in activations. If these outliers are naively quantized to int8, they cause significant accuracy degradation. LLM.int8() detects these outlier dimensions and computes them in mixed fp16 precision while the remaining dimensions use efficient int8 arithmetic.

---

## Module: `Linear8bitLt`

```python
class Linear8bitLt(nn.Linear):
```

The primary module for 8-bit quantized linear layers. Extends `torch.nn.Linear`.

### Constructor

```python
def __init__(
    self,
    input_features: int,
    output_features: int,
    bias: bool = True,
    has_fp16_weights: bool = True,
    threshold: float = 0.0,
    index = None,
    device = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_features` | `int` | required | Number of input features (in_features) |
| `output_features` | `int` | required | Number of output features (out_features) |
| `bias` | `bool` | `True` | Whether to include a bias term |
| `has_fp16_weights` | `bool` | `True` | If `False`, weights are quantized to int8 on `.to(device)`. If `True`, weights remain in fp16 and are quantized on-the-fly each forward pass (used for training). |
| `threshold` | `float` | `0.0` | Outlier threshold for mixed-precision decomposition. Columns in activations where any value exceeds this threshold are computed in fp16. Set to `0.0` to disable. |
| `index` | Any | `None` | Internal index for weight reordering |
| `device` | `torch.device` | `None` | Device to initialize on |

#### Internal State

Upon construction:
- `self.weight` is set to an `Int8Params` instance with the given `has_fp16_weights` flag.
- `self.state` is initialized as a new `MatmulLtState` with the specified `threshold` and `has_fp16_weights`.
- If `threshold > 0.0` and `has_fp16_weights == False`, `state.use_pool = True` (enables `GlobalOutlierPooler`).
- A `maybe_rearrange_weight` pre-hook is registered on `_load_from_state_dict` for backward compatibility with older weight formats.

### Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### Forward Flow

1. **Transfer quantization state**: If `self.weight.CB is not None`, call `init_8bit_state()` to move CB/SCB from the parameter to the module's state.
2. **Bias casting**: If bias exists and its dtype differs from input dtype, cast bias to match.
3. **Matmul dispatch**: Call `bnb.matmul(x, self.weight, bias=self.bias, state=self.state)`.
4. **Update weight data**: If weights were quantized (not fp16), store the int8 weight data back to `self.weight.data`.

### `init_8bit_state()`

```python
def init_8bit_state(self) -> None
```

Transfers quantization state from the `Int8Params` weight to the module's `MatmulLtState`:
```python
self.state.CB = self.weight.CB
self.state.SCB = self.weight.SCB
self.weight.CB = None
self.weight.SCB = None
```

This is called once on the first forward pass after `.to(device)`.

### State Dict Save (`_save_to_state_dict`)

```python
def _save_to_state_dict(self, destination, prefix, keep_vars)
```

Saves:
1. Standard weight and bias via `super()._save_to_state_dict()`.
2. If weights are quantized (`has_fp16_weights == False`), saves `SCB` (the weight scaling factors) under `prefix + "SCB"`.
3. Saves a `weight_format` tensor (always `0` for row-major format) for backward compatibility.

### State Dict Load (`_load_from_state_dict`)

```python
def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
```

Handles loading pre-quantized checkpoints:
1. Calls `super()._load_from_state_dict()` first.
2. Looks for an unexpected key `"SCB"` in the loaded state dict.
3. If found, copies the SCB values into `self.weight.SCB` and `self.state.SCB`.
4. Raises `RuntimeError` if loading a quantized checkpoint into a non-quantized module (must call `.cuda()` before `load_state_dict()`).

### `to()`

```python
def to(self, *args, **kwargs) -> "Linear8bitLt"
```

Overrides the default `to()` to also move the module's state tensors (CB, SCB) to the target device.

---

## Parameter: `Int8Params`

```python
class Int8Params(torch.nn.Parameter):
```

A `torch.nn.Parameter` subclass that handles 8-bit weight quantization.

### Constructor (`__new__`)

```python
def __new__(
    cls,
    data: Optional[torch.Tensor] = None,
    requires_grad: bool = True,
    has_fp16_weights: bool = False,
    CB: Optional[torch.Tensor] = None,
    SCB: Optional[torch.Tensor] = None,
    **kwargs,
) -> "Int8Params"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `torch.Tensor` | `None` (empty tensor) | The weight data |
| `requires_grad` | `bool` | `True` | Whether gradients are tracked |
| `has_fp16_weights` | `bool` | `False` | If False, quantizes on `.to(device)` |
| `CB` | `torch.Tensor` | `None` | Cached int8 quantized weight |
| `SCB` | `torch.Tensor` | `None` | Cached weight scaling factors |

### `_quantize(device)`

```python
def _quantize(self, device) -> "Int8Params"
```

Called when the parameter is moved from CPU to GPU (via `.to(device)`) with `has_fp16_weights=False`.

**Algorithm**:
1. If `has_fp16_weights` is True, just calls `super().to(device)` (no quantization).
2. Converts data to contiguous float16 on the target device.
3. Calls `bnb.functional.int8_vectorwise_quant(B)` to quantize the weights.
4. Stores the int8 quantized data in `self.data`, and the scaling factors in `self.SCB`.
5. `self.CB` is set to the int8 weight data (for later transfer to `MatmulLtState`).

### `cpu()`, `cuda()`, `xpu()`

```python
def cpu(self) -> "Int8Params"
def cuda(self, device=None, non_blocking=False) -> "Int8Params"
def xpu(self, device=None, non_blocking=False) -> "Int8Params"
```

Convenience methods that delegate to `self.to()`.

### `to()`

```python
def to(self, *args, **kwargs) -> "Int8Params"
```

Handles device movement with automatic quantization:

1. Parse target device/dtype from arguments.
2. If not yet quantized (`self.data.dtype != torch.int8`) and moving from CPU to a non-meta device, call `self._quantize(device)`.
3. If already quantized, create a new `Int8Params` on the target device and move SCB appropriately.

### `__deepcopy__()`

```python
def __deepcopy__(self, memo) -> "Int8Params"
```

Creates a deep copy including all attributes (data, CB, SCB, has_fp16_weights).

---

## Dataclass: `MatmulLtState`

```python
@dataclass
class MatmulLtState:
```

Mutable state container for the 8-bit matmul operation. Maintained across forward passes to cache quantized weights.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `force_no_igemmlt` | `bool` | `False` | Force fallback path |
| `CB` | `torch.Tensor` | `None` | Quantized int8 weight matrix |
| `SB` | `torch.Tensor` | `None` | (unused in current version) |
| `SCB` | `torch.Tensor` | `None` | Per-row scaling factors for B (weights) |
| `SBt` | `torch.Tensor` | `None` | (unused in current version) |
| `CBt` | `torch.Tensor` | `None` | (unused in current version) |
| `subB` | `torch.Tensor` | `None` | (unused in current version) |
| `outlier_pool` | `GlobalOutlierPooler` | `None` | Reference to the global outlier pooler |
| `has_accumulated_gradients` | `bool` | `False` | (unused in current version) |
| `threshold` | `float` | `0.0` | Outlier threshold for mixed-precision decomposition |
| `idx` | `torch.Tensor` | `None` | Column indices of detected outlier features |
| `is_training` | `bool` | `True` | Whether the parent module is in training mode |
| `has_fp16_weights` | `bool` | `True` | Whether weights are kept in fp16 |
| `use_pool` | `bool` | `False` | Whether to use `GlobalOutlierPooler` |

### Deprecated Fields

The following fields are always `None` and will be removed in a future release. Accessing them triggers a `FutureWarning`:

- `CxB` -- Always `None`.
- `CxBt` -- Always `None`.
- `formatB` -- Always `None`.
- `_tile_indices` -- Always `None`.

### `reset_grads()`

```python
def reset_grads(self) -> None
```

Resets all cached tensors to `None`:
```python
self.CB = None
self.SB = None
self.SCB = None
self.SBt = None
self.CBt = None
```

Called at the beginning of each forward pass when in training mode.

---

## Autograd Function: `MatMul8bitLt`

```python
class MatMul8bitLt(torch.autograd.Function):
```

The primary autograd function for 8-bit matrix multiplication. Used for CUDA and HPU backends.

### Forward

```python
@staticmethod
def forward(
    ctx,
    A: torch.Tensor,          # activations [batch, seq, in_features] or [batch, in_features]
    B: torch.Tensor,          # weights [out_features, in_features] (Int8Params)
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
) -> torch.Tensor
```

#### Forward Algorithm

1. **Empty input check**: If `prod(A.shape) == 0`, return an empty tensor of the appropriate shape.

2. **Reshape**: If A is 3D, reshape to 2D: `[batch*seq, in_features]`.

3. **Quantize A (activations)**:
   - If `ctx.needs_input_grad[1]` (gradient needed for B): Use `int8_double_quant(A)` for both row-wise and column-wise quantization. This produces:
     - `CA`: row-wise quantized A (int8)
     - `CAt`: column-wise quantized A (int8)
     - `SCA`: row-wise scaling factors
     - `SCAt`: column-wise scaling factors
     - `outlier_cols`: indices of outlier columns (if threshold > 0)
   - Otherwise (gradient only needed for A): Use `int8_vectorwise_quant(A)` for row-wise only. This is the fast path.
     - `CA`: row-wise quantized A (int8)
     - `SCA`: row-wise scaling factors
     - `CAt = SCAt = None`

4. **Quantize B (weights)** -- if needed:
   - If `has_fp16_weights` or `state.CB is None`:
     - Check if B needs to be made contiguous.
     - If in training mode without accumulated gradients, or CB/SCB not yet cached, call `state.reset_grads()` then `int8_vectorwise_quant(B)` to populate `state.CB` and `state.SCB`.

5. **Mixed-precision decomposition or standard int8 matmul**:
   - **If threshold > 0**: Call `int8_mixed_scaled_mm(A, CA, CB, SCA, SCB, outlier_cols, bias)`. This returns `(output, subA)` where `subA` is the submatrix of A at outlier column indices for use in backward.
   - **If threshold == 0**: Call `int8_scaled_mm(CA, CB, SCA, SCB, bias, dtype=A.dtype)`. Standard int8 matmul + dequant.

6. **Save for backward**: Save `CAt`, `subA`, `A`, `SCAt`, `state.idx` in context.

7. **Reshape output**: If input was 3D, reshape output back.

### Backward

```python
@staticmethod
def backward(ctx, grad_output: torch.Tensor) -> tuple
```

#### Backward Algorithm

1. **Empty input case**: Return zero tensors matching the shapes of A, B, bias.

2. **Bias gradient**: `grad_bias = grad_output.sum(0)` (sum over batch dimension).

3. **Reshape**: If grad_output is 3D, reshape to 2D.

4. **Weight gradient** (`req_gradB`):
   - Quantize `grad_output` using `int8_double_quant()` to get `Cgrad` and `SCgradt`.
   - Compute `grad_B = int8_scaled_mm(Cgrad.t(), CAt.t(), SCgradt, SCAt, dtype=float16)`.
   - If threshold > 0 and `subA` exists: Add the fp16 contribution from outlier columns: `grad_B[:, idx] += grad_output.t() @ subA`.

5. **Activation gradient** (`req_gradA`):
   - Dequantize B: `CB = state.CB.float() * (state.SCB / 127.0).unsqueeze(1)`.
   - Compute `grad_A = grad_output @ CB`.

#### Return Signature

```python
return (grad_A, grad_B, None, grad_bias, None)
# Corresponds to: (A, B, out, bias, state)
```

---

## Autograd Function: `MatMul8bitFp`

```python
class MatMul8bitFp(torch.autograd.Function):
```

A faster CPU/XPU alternative to `MatMul8bitLt`. Instead of using quantized kernels (which are slow without GPU tensor cores), it dequantizes the weights and uses standard PyTorch matmul. Approximately 3x faster than `MatMul8bitLt` for fine-tuning on CPU/XPU.

### Forward

```python
@staticmethod
def forward(ctx, A, B, out=None, bias=None, state=MatmulLtState) -> torch.Tensor
```

1. **Quantize B if needed**: Same logic as `MatMul8bitLt` -- if fp16 weights or CB not cached, quantize B with `int8_vectorwise_quant`.
2. **Dequantize B**: `CB = state.CB.to(A.dtype) * (state.SCB / 127.0).unsqueeze(1)`.
3. **Compute output**: `output = F.linear(A, CB, bias)` -- standard PyTorch linear.
4. **Save for backward**: Store state, input dtype, grad shape, and original A.

### Backward

```python
@staticmethod
def backward(ctx, grad_output) -> tuple
```

1. **Bias gradient**: `grad_bias = grad_output.sum(0)`.
2. **Weight gradient** (`req_gradB`): `grad_B = A.t() @ grad_output` then transpose.
3. **Activation gradient** (`req_gradA`): Dequantize B, then `grad_A = grad_output @ CB`.

---

## Dispatch Function: `matmul()`

```python
def matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    state: Optional[MatmulLtState] = None,
    threshold: float = 0.0,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Top-level dispatch for 8-bit matrix multiplication. Selects the appropriate autograd function based on device and training mode.

### Dispatch Logic

```python
state = state or MatmulLtState()
if threshold > 0.0:
    state.threshold = threshold

if state.is_training:
    if A.device.type in ("cpu", "xpu"):
        return MatMul8bitFp.apply(A, B, out, bias, state)
return MatMul8bitLt.apply(A, B, out, bias, state)
```

| Condition | Function | Rationale |
|-----------|----------|-----------|
| Training on CPU/XPU | `MatMul8bitFp` | ~3x faster (dequant + matmul vs slow quantized kernels) |
| All other cases (CUDA, HPU, inference) | `MatMul8bitLt` | Uses fast int8 tensor cores |

---

## Low-Level Quantization Functions

### `int8_vectorwise_quant()`

```python
def int8_vectorwise_quant(
    A: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
```

Row-wise quantization of a float16 tensor to int8.

#### Algorithm

1. For each row of A, compute the absolute maximum value.
2. Scale each row: `CA = round(A * 127 / absmax_per_row)` to get int8 values.
3. If `threshold > 0.0`, identify columns where any value exceeds the threshold as outlier features.
4. Suppress outlier values in CA (set to 0 or clip).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `torch.Tensor` (float16) | required | Input tensor |
| `threshold` | `float` | `0.0` | Outlier detection threshold (0 = disabled) |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `CA` | `torch.Tensor` (int8) | Quantized tensor |
| `SCA` | `torch.Tensor` (float32) | Per-row scaling factors (absmax / 127) |
| `outlier_cols` | `torch.Tensor` (int64) or `None` | Column indices with outliers (None if threshold == 0) |

### `int8_double_quant()`

```python
def int8_double_quant(
    A: torch.Tensor,
    col_stats: Optional[torch.Tensor] = None,
    row_stats: Optional[torch.Tensor] = None,
    out_col: Optional[torch.Tensor] = None,
    out_row: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
```

Performs both row-wise and column-wise quantization of a float16 tensor. Used for the backward pass when gradients are needed for both A and B.

> **Note**: The `col_stats`, `row_stats`, `out_col`, `out_row` parameters must all be `None`. Pre-allocated outputs are not supported.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `torch.Tensor` (float16) | required | Input tensor |
| `col_stats` | `torch.Tensor` | `None` | Must be None |
| `row_stats` | `torch.Tensor` | `None` | Must be None |
| `out_col` | `torch.Tensor` | `None` | Must be None |
| `out_row` | `torch.Tensor` | `None` | Must be None |
| `threshold` | `float` | `0.0` | Outlier detection threshold |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `out_row` | `torch.Tensor` (int8) | Row-wise quantized A |
| `out_col` | `torch.Tensor` (int8) | Column-wise (transposed) quantized A |
| `row_stats` | `torch.Tensor` (float32) | Per-row scaling factors |
| `col_stats` | `torch.Tensor` (float32) | Per-column scaling factors |
| `outlier_cols` | `torch.Tensor` (int64) or `None` | Column indices with outliers |

### `int8_linear_matmul()`

```python
def int8_linear_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor
```

Performs pure 8-bit integer matrix multiplication using int8 tensor cores where available.

**Both A and B must be `torch.int8`**. The result is `torch.int32` (accumulated product).

**Operation**: `out = A @ B.T` where A and B are int8 matrices.

On CUDA, this uses cuBLASLt int8 matmul. If the inner dimension is not divisible by 4, falls back to fp32 matmul.

### `int8_mm_dequant()`

```python
def int8_mm_dequant(
    A: torch.Tensor,       # int32 result from int8 matmul
    row_stats: torch.Tensor, # row-wise quantization scales
    col_stats: torch.Tensor, # column-wise quantization scales
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Dequantizes the int32 result of an int8 matmul back to float16 using the row and column scaling statistics.

**Algorithm**: The int8 matmul computes `CA @ CB.T` where `CA = round(A * 127 / row_stats)` and `CB = round(B * 127 / col_stats)`. The int32 result is: `CA @ CB = (A * 127 / row_stats) @ (B * 127 / col_stats) = (127 * 127) * A @ B / (row_stats * col_stats)`. Dequantization reverses this scaling.

**Returns**: `torch.Tensor` of dtype `float16` with optional bias added.

### `int8_scaled_mm()`

```python
# Custom op: bitsandbytes::int8_scaled_mm
# Signature: (Tensor A, Tensor B, Tensor row_stats, Tensor col_stats,
#             Tensor? bias=None, ScalarType? dtype=None) -> Tensor
```

Combined operation: **int8 matmul + dequant + bias** in a single kernel call. This is the primary fast path when threshold == 0.

**Parameters**:
- `A`: Row-wise quantized activations (int8)
- `B`: Quantized weights (int8)
- `row_stats`: Per-row scaling for A (float32)
- `col_stats`: Per-row scaling for B (float32)
- `bias`: Optional bias vector
- `dtype`: Output dtype (default: float16)

**Returns**: Dequantized result with dtype matching the input activation dtype.

### `int8_mixed_scaled_mm()`

```python
# Custom op: bitsandbytes::int8_mixed_scaled_mm
# Signature: (Tensor A, Tensor CA, Tensor CB, Tensor SCA, Tensor SCB,
#             Tensor? outlier_cols=None, Tensor? bias=None)
#             -> (Tensor output, Tensor? subA)
```

Mixed-precision path with outlier handling. Used when `threshold > 0`.

**Parameters**:
- `A`: Original fp16 activations
- `CA`: Row-wise quantized activations (int8)
- `CB`: Quantized weights (int8)
- `SCA`: Per-row scaling for CA (float32)
- `SCB`: Per-row scaling for CB (float32)
- `outlier_cols`: Column indices with outlier features (int64)
- `bias`: Optional bias vector

**Returns**:
- `output`: The result of the int8 matmul for non-outlier columns + fp16 matmul for outlier columns.
- `subA`: Submatrix of original A at outlier column indices (needed for backward pass).

**Algorithm**:
1. Compute int8 matmul for all columns: `int8_scaled_mm(CA, CB, SCA, SCB)`.
2. For outlier columns, zero out the int8 contribution and add fp16 matmul: `A[:, outlier_cols] @ B[:, outlier_cols].T`.
3. Return combined result and subA for backward.

---

## `int8_vectorwise_dequant()`

```python
def int8_vectorwise_dequant(
    A: torch.Tensor,
    stats: torch.Tensor,
) -> torch.Tensor
```

Dequantizes an int8 tensor back to float32 using the provided scaling statistics.

**Algorithm**: `result = A * stats.view(-1, 1) / 127.0`

The default PyTorch-native implementation:
```python
return A * stats.view(-1, 1) * 7.874015718698502e-3  # 1/127
```

---

## `GlobalOutlierPooler`

```python
class GlobalOutlierPooler:
```

A singleton class that pools outlier dimensions across layers. This is particularly important for small models where outlier features are less systematic and occur with low frequency.

### Methods

```python
@classmethod
def get_instance(cls) -> "GlobalOutlierPooler"
```

Returns the singleton instance.

```python
def add_outliers(self, outlier_idx: torch.Tensor, feature_dim: int) -> None
```

Adds outlier column indices to the pool. Only accumulates if the feature dimension matches the first one seen (to avoid mixing FFN layers with different dimensions).

```python
def get_current_outlier_idx(self) -> torch.Tensor
```

Returns all accumulated outlier indices as an int64 tensor.

---

## `OutlierTracer`

```python
class OutlierTracer:
```

A utility for detecting outlier dimensions in model weights. Used by `OutlierAwareLinear`.

### Methods

```python
@classmethod
def get_instance(cls) -> "OutlierTracer"
```

Returns the singleton instance.

```python
def initialize(self, model: torch.nn.Module) -> None
```

Registers `outlier_hook` forward pre-hooks on all `nn.Linear` modules in the model. The hook detects outlier dimensions based on:
- First layer: z-score test of hidden dimension standard deviations + magnitude > 6 test.
- Subsequent layers: outlier dimensions detected from the weight of the previous linear layer.

```python
def get_outliers(self, weight: torch.Tensor) -> Optional[torch.Tensor]
```

Returns the outlier column indices for the given weight tensor, identified by its storage data pointer.

---

## Threshold Behavior

The `threshold` parameter controls mixed-precision decomposition in the LLM.int8() algorithm:

### `threshold = 0.0` (default)

- **No outlier detection**. All activation columns are quantized to int8.
- Forward uses `int8_scaled_mm()` (single kernel: quantize + matmul + dequant + bias).
- This is the fastest path but may lose accuracy for models with systematic outliers (typically >= 6.7B parameters).

### `threshold > 0.0`

- **Outlier detection enabled**. For each activation matrix A, any column where at least one value has absolute magnitude > threshold is marked as an outlier.
- Outlier columns are zeroed out in the int8 path and computed separately in fp16.
- Forward uses `int8_mixed_scaled_mm()` which handles both paths internally.
- The outlier indices are stored in `state.idx` for the backward pass.
- Recommended values: 5.0 -- 6.0 based on the paper's experiments.
- If `use_pool = True`, outliers are accumulated in `GlobalOutlierPooler` across layers.

```python
# Example: Enable mixed-precision with threshold
linear = Linear8bitLt(
    4096, 4096,
    has_fp16_weights=False,
    threshold=6.0,  # columns with values > 6.0 computed in fp16
)
```

---

## Weight Format Storage

The `LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING` defines the supported weight storage formats:

```python
LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING = {
    "row": 0,         # Row-major (current default and only supported format)
    "col32": 1,       # Column-major with 32-element tiles (legacy)
    "col_turing": 2,  # Turing-specific column format (legacy)
    "col_ampere": 3,  # Ampere-specific column format (legacy)
}
```

**Current version**: Only `"row"` format is supported. The `col32`, `col_turing`, and `col_ampere` formats are from older versions and will raise `ValueError` if encountered during loading.

The `maybe_rearrange_weight` hook handles backward compatibility:
- When loading a state dict, if `weight_format` is present and not `"row"`, it raises an error.
- The `weight_format` key is popped from the state dict after processing.
