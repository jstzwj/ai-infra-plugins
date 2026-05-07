# Quantization Fundamentals

This document covers the core quantization algorithms, data types, and state management used throughout bitsandbytes.

---

## Block-wise Quantization Algorithm

Block-wise quantization is the fundamental technique used by all bitsandbytes quantization methods. Instead of using a single global scaling factor for an entire tensor, the tensor is divided into contiguous blocks and each block is independently scaled.

### Algorithm Steps

1. **Divide** the input tensor into contiguous blocks of `blocksize` elements.
2. **Compute** the absolute maximum value within each block: `absmax_i = max(|block_i|)`.
3. **Scale** each block by dividing by its absmax: `scaled_block = block / absmax_i`.
4. **Map** each scaled value to the nearest value in the quantization codebook (lookup table).
5. **Store** the quantized indices alongside the per-block absmax values.

For a tensor with `N` elements and blocksize `B`:
- Number of blocks: `num_blocks = ceil(N / B)`
- Absmax array shape: `(num_blocks,)` of dtype `float32`
- Quantized tensor shape depends on bit width (e.g., for 4-bit: `(ceil(N / 2), 1)` packed into `uint8`)

### Dequantization

Dequantization reverses the process:
1. Look up each quantized index in the codebook to get the scaled value.
2. Multiply each value by its block's absmax: `value = lookup[index] * absmax[block_idx]`.

---

## Quantization Data Types (Codebook Maps)

bitsandbytes uses lookup tables (codebooks) that define the mapping between quantized indices and real values. These are stored as 256-element tensors for 8-bit indexing, with the actual data type values occupying the first `2^bits` entries and the rest zero-padded.

### Dynamic Data Type -- `create_dynamic_map()`

```python
def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8) -> torch.Tensor
```

The dynamic data type uses a **dynamic exponent + fraction** encoding. As the exponent increases, the number of bits available for the fraction (mantissa) shrinks. This provides a non-uniform distribution that covers a wide dynamic range.

#### Algorithm

For each exponent level `i` from 0 to `max_exponent_bits - 1`:
1. Compute the number of fraction items: `2^(i + non_sign_bits - max_exponent_bits) + 1` (signed) or `2^(i + non_sign_bits - max_exponent_bits + 1) + 1` (unsigned).
2. Generate evenly spaced boundaries in `[0.1, 1.0]`.
3. Compute means of adjacent boundaries.
4. Scale by `10^(-(max_exponent_bits - 1) + i)`.
5. If signed, also add the negated values.

Finally, append `0` and `1.0`, sort all values, and return as a 256-element `float32` tensor.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signed` | `bool` | `True` | Whether to include negative values |
| `max_exponent_bits` | `int` | `7` | Maximum number of exponent bits |
| `total_bits` | `int` | `8` | Total number of bits (must be 8) |

#### Returns

`torch.Tensor` of shape `(256,)` with dtype `float32`. The first 256 entries contain the sorted quantization levels.

#### Reference

> 8-Bit Approximations for Parallelism in Deep Learning ([arXiv:1511.04561](https://arxiv.org/abs/1511.04561))

---

### Linear Quantization Map -- `create_linear_map()`

```python
def create_linear_map(signed=True, total_bits=8, add_zero=True) -> torch.Tensor
```

Creates a uniformly spaced quantization map. Values are evenly distributed between `-1.0` (or `0.0` for unsigned) and `1.0`.

#### Algorithm

1. Compute `total_values = 2^total_bits` (or `2^total_bits - 1` for signed with zero-centering).
2. Generate `total_values` evenly spaced values using `torch.linspace(sign, 1.0, total_values)`.
3. If there is a gap between `total_values` and 256, insert zeros in the middle to maintain the zero point.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signed` | `bool` | `True` | If `True`, range is `[-1, 1]`; if `False`, range is `[0, 1]` |
| `total_bits` | `int` | `8` | Total bit width |
| `add_zero` | `bool` | `True` | Whether to ensure zero is in the map |

#### Returns

`torch.Tensor` of shape `(256,)` with dtype `float32`.

---

### NF4 (NormalFloat4) -- `create_normal_map()`

```python
def create_normal_map(offset=0.9677083, use_extra_value=True) -> torch.Tensor
```

NF4 is a **quantile-based** data type specifically designed for normally-distributed neural network weights. It is **not** a floating-point encoding -- it is purely a lookup table where each 4-bit index maps to a specific value.

#### Derivation

1. Compute quantile boundaries of the standard normal distribution N(0,1) using `scipy.stats.norm.ppf()`.
2. For the positive side: `norm.ppf(linspace(offset, 0.5, 9)[:-1])` gives 8 positive values (or 7 without extra).
3. For the negative side: `-norm.ppf(linspace(offset, 0.5, 8)[:-1])` gives 8 negative values.
4. Insert a zero value.
5. Sort all values and normalize by dividing by the maximum absolute value.

#### Key Properties

- **offset = 0.9677083**: The outermost quantile boundary, covering approximately 1.845 standard deviations of N(0,1). This was empirically optimized for typical neural network weight distributions.
- **15 non-zero values** (when `use_extra_value=True`): 8 negative, zero, 8 positive = 17 positions, but 2 are merged to give 15 non-zero values (8 negative + 0 + 7 positive becomes 8 negative + 0 + 8 positive with the extra value being the smallest positive). Actually: 8 positive quantile boundaries, 8 negative quantile boundaries, and one zero, for a total of 17 values, but after removing duplicates and normalizing, 16 unique values remain.
- **16 total values** (4 bits), fitting perfectly in a 4-bit representation.

#### Hardcoded NF4 Values

To avoid a scipy dependency at runtime, the NF4 lookup table is hardcoded in `get_4bit_type("nf4")`:

```python
nf4_values = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
```

These 16 values span `[-1.0, 1.0]` and represent optimal quantile boundaries for normally-distributed data.

#### Why NF4 is NOT a Floating-Point Encoding

Unlike FP4, NF4 has no sign/exponent/mantissa decomposition. The 4-bit index is simply a position in the lookup table. This means:
- The values are not evenly spaced.
- The spacing is denser near zero and coarser at the extremes.
- This is optimal for data following a normal distribution (which neural network weights approximately do).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset` | `float` | `0.9677083` | Outermost quantile boundary |
| `use_extra_value` | `bool` | `True` | If True, asymmetric with extra positive value; if False, symmetric |

#### Returns

`torch.Tensor` of shape `(256,)` with dtype `float32`. First 16 entries are the sorted NF4 levels; remaining 240 entries are zero (padding for 8-bit indexing).

#### Dependencies

Requires `scipy.stats.norm` at call time. Install with `pip install bitsandbytes[test]`.

> **Note**: At runtime, `get_4bit_type("nf4")` returns the hardcoded values and does NOT require scipy. The `create_normal_map()` function is primarily used for testing and verification.

#### Reference

> QLoRA: Efficient Finetuning of Quantized LLMs ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))

---

### FP4 (4-bit Floating Point) -- `create_fp8_map()` with 4 bits

```python
def create_fp8_map(signed=True, exponent_bits=2, precision_bits=1, total_bits=4) -> torch.Tensor
```

FP4 is a true IEEE 754-like floating-point encoding with **1 sign bit + 2 exponent bits + 1 mantissa bit**.

#### Bit Layout

```
| S | E1 | E0 | M0 |
```

- **Sign (S)**: 1 bit
- **Exponent (E1:E0)**: 2 bits, bias = `2^(2-1) = 2`
- **Mantissa (M0)**: 1 bit

#### Encoding Rules

- **Normal values** (exponent != 0): `value = (1 + M0 * 0.5) * 2^(exponent - bias - 1)`
- **Subnormal values** (exponent == 0): `value = M0 * 2^(-bias)`

#### Raw (pre-normalization) Values by Bit Pattern

| Bit Pattern (SEEM) | Unsigned Value | Signed Value |
|---------------------|----------------|--------------|
| 0000 | 0 | 0 |
| 0001 | 0.0625 | 0.0625 |
| 0010 | 8.0 | 8.0 |
| 0011 | 12.0 | 12.0 |
| 0100 | 4.0 | 4.0 |
| 0101 | 6.0 | 6.0 |
| 0110 | 2.0 | 2.0 |
| 0111 | 3.0 | 3.0 |
| 1000 | -0 | -0 |
| 1001 | -0.0625 | -0.0625 |
| 1010 | -8.0 | -8.0 |
| 1011 | -12.0 | -12.0 |
| 1100 | -4.0 | -4.0 |
| 1101 | -6.0 | -6.0 |
| 1110 | -2.0 | -2.0 |
| 1111 | -3.0 | -3.0 |

After normalization (divide by max absolute value = 12.0), all values are in `[-1, 1]`.

#### Hardcoded FP4 Values

```python
fp4_values = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0,
              -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0]
```

After sorting and normalization by `data.div_(data.abs().max())`, the 16 sorted normalized values occupy the first 16 entries of the returned 256-element tensor.

#### Parameters for FP4

| Parameter | Value for FP4 |
|-----------|---------------|
| `signed` | `True` |
| `exponent_bits` | `2` |
| `precision_bits` | `1` |
| `total_bits` | `4` |

#### Returns

`torch.Tensor` of shape `(256,)` with dtype `float32`. First 16 entries are sorted FP4 levels; remaining 240 are zero-padded.

---

### AF4 (AbnormalFloats4)

```python
# Accessed via get_4bit_type("af4", blocksize=64)
```

An alternative to NF4 proposed in the paper "NF4 Isn't Information Theoretically Optimal (and that's Good)" ([arXiv:2306.06965](https://arxiv.org/abs/2306.06965)).

#### Restrictions

- **Only supports blocksize 64**. Other blocksizes raise `NotImplementedError`.

#### Hardcoded AF4 Values (blocksize 64, reversed)

```python
af4_values = [
    -1.0,
    -0.69441008,
    -0.51243739,
    -0.3736951,
    -0.25607552,
    -0.14982478,
    -0.04934812,
    0.0,
    0.04273164,
    0.12934483,
    0.21961274,
    0.31675666,
    0.42563882,
    0.55496234,
    0.72424863,
    1.0,
][::-1]  # reversed in the source
```

---

### INT4 (Integer 4-bit)

```python
# Accessed via get_4bit_type("int4")
int4_values = [7, 6, 5, 4, 3, 2, 1, 0, -0, -1, -2, -3, -4, -5, -6, -7]
```

A simple signed 4-bit integer type. Values range from -7 to 7.

---

## `get_4bit_type()` -- Runtime Codebook Access

```python
def get_4bit_type(typename: str, device=None, blocksize=64) -> torch.Tensor
```

Returns the hardcoded 4-bit quantization lookup table for the given type. This function avoids runtime dependencies on scipy by using pre-computed values.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `typename` | `str` | required | One of `"nf4"`, `"fp4"`, `"int4"`, `"af4"` |
| `device` | `torch.device` or `None` | `None` (defaults to `"cuda"`) | Target device |
| `blocksize` | `int` | `64` | Block size (only affects `"af4"`, which requires 64) |

### Returns

`torch.Tensor` of shape `(16,)` on the specified device, normalized to `[-1, 1]`.

### Raises

- `NotImplementedError`: If `typename` is not recognized, or if `"af4"` is requested with a blocksize other than 64.

---

## QuantState Class

```python
class QuantState:
    """Container for quantization state components."""
```

`QuantState` encapsulates all the metadata needed to dequantize a tensor. It stores the scaling factors, codebook, block size, data type, original shape, and optional nested (double) quantization state.

### Constructor

```python
def __init__(
    self,
    absmax: torch.Tensor,
    shape: Optional[torch.Size] = None,
    code: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    offset: Optional[torch.Tensor] = None,
    state2: Optional["QuantState"] = None,
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `absmax` | `torch.Tensor` | Per-block absolute maximum values (the scaling factors) |
| `shape` | `torch.Size` or `None` | Original shape of the unquantized tensor |
| `code` | `torch.Tensor` or `None` | The quantization codebook (lookup table) |
| `blocksize` | `int` or `None` | Number of elements per quantization block |
| `quant_type` | `str` or `None` | Quantization type: `"fp4"` or `"nf4"` |
| `dtype` | `torch.dtype` or `None` | Original dtype of the unquantized tensor |
| `offset` | `torch.Tensor` or `None` | Mean of absmax (used only for double quantization) |
| `state2` | `QuantState` or `None` | Nested QuantState for double-quantized absmax |

### Class Attributes

```python
valid_quant_types = ("fp4", "nf4")
valid_qs_type_keys = ["bitsandbytes__fp4", "bitsandbytes__nf4"]
valid_qs_keys = [
    "absmax", "quant_map", "nested_absmax", "nested_quant_map",
    "quant_state", "quant_type", "blocksize", "dtype", "shape",
    "nested_blocksize", "nested_dtype", "nested_offset",
]
```

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `absmax` | `torch.Tensor` | Per-block scaling factors (or quantized absmax if nested) |
| `shape` | `torch.Size` | Original tensor shape |
| `code` | `torch.Tensor` | Quantization codebook (256-element tensor) |
| `blocksize` | `int` | Elements per block |
| `quant_type` | `str` | `"fp4"` or `"nf4"` |
| `dtype` | `torch.dtype` | Original tensor dtype |
| `offset` | `torch.Tensor` or `None` | Mean of absmax for double quantization |
| `state2` | `QuantState` or `None` | Nested state for double-quantized absmax |
| `nested` | `bool` | `True` if `state2 is not None` |

### Methods

#### `from_dict(cls, qs_dict, device) -> QuantState`

```python
@classmethod
def from_dict(cls, qs_dict: dict[str, Any], device: torch.device) -> "QuantState"
```

Reconstructs a `QuantState` from a dictionary, typically loaded from a `state_dict`. Handles both packed (safetensors-compatible) and unpacked formats.

**Packed format**: Non-tensor items (quant_type, blocksize, dtype, shape, nested_offset) are serialized as a JSON-encoded uint8 tensor stored under the key `"quant_state.bitsandbytes__[nf4|fp4]"`. This tensor is unpacked via `unpack_tensor_to_dict()`.

**Unpacked format**: All items are separate entries in the dictionary.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `qs_dict` | `dict[str, Any]` | Dictionary with quant state items, stripped of module prefixes |
| `device` | `torch.device` | Target device for all tensors |

**Returns**: Reconstructed `QuantState` instance.

#### `as_dict(packed=False) -> dict[str, Any]`

```python
def as_dict(self, packed: bool = False) -> dict[str, Any]
```

Serializes the quant state to a dictionary. When `packed=True`, non-tensor items are encoded into a single uint8 tensor for safetensors compatibility.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `packed` | `bool` | `False` | If `True`, returns `dict[str, torch.Tensor]` suitable for safetensors |

**Returns**:

- When `packed=False`: A dictionary with both tensor and non-tensor values.
- When `packed=True`: A dictionary with only tensor values. Non-tensor items are packed into a single tensor under the key `"quant_state.bitsandbytes__[nf4|fp4]"`.

**Packed format example**:
```python
{
    "absmax": tensor(...),           # float32
    "quant_map": tensor(...),        # float32 (codebook)
    "quant_state.bitsandbytes__nf4": tensor(...),  # uint8 (JSON-encoded metadata)
    # If double quantized:
    "nested_absmax": tensor(...),    # uint8 (quantized absmax)
    "nested_quant_map": tensor(...), # float32 (nested codebook)
}
```

#### `to(device)`

```python
def to(self, device) -> None
```

Moves all tensors (code, absmax, offset, state2 tensors) to the specified device. Modifies in place.

#### `__eq__(other) -> bool`

Compares two `QuantState` instances for equality. Uses `torch.allclose` with `atol=1e-6` for tensor comparison. Both `offset` and `state2` must be equal (or both `None`) for nested states.

#### `__getitem__(idx)`

Provides backward-compatible list-style access to the quantization state, matching the old list-based layout:

```python
# Nested state:
state[0] == absmax
state[1] == shape
state[2] == dtype
state[3] == blocksize
state[4] == [offset, state2]  # nested info
state[5] == quant_type

# Non-nested state:
state[0] == absmax
state[1] == shape
state[2] == dtype
state[3] == blocksize
state[4] == None
state[5] == quant_type
```

#### `__getattr__(name)`

Supports FSDP state_dict traversal. When PyTorch's FSDP resolves dotted fully-qualified names, it calls `getattr(quant_state, "bitsandbytes__nf4")`. This method delegates to `as_dict(packed=True)` to return the appropriate value.

### Packed Format for Safetensors Serialization

Safetensors can only store tensors, not arbitrary Python objects. The packed format solves this by:

1. Collecting all non-tensor items (quant_type, blocksize, dtype, shape, nested_offset) into a dictionary.
2. Serializing the dictionary to JSON.
3. Encoding the JSON string as a uint8 tensor.
4. Storing this tensor under the key `"quant_state.bitsandbytes__[nf4|fp4]"`.

```python
# Packing
json_str = json.dumps(non_tensor_dict)           # e.g., {"quant_type": "nf4", "blocksize": 64, ...}
json_bytes = json_str.encode("utf-8")             # bytes
tensor = torch.tensor(list(json_bytes), dtype=torch.uint8)  # uint8 tensor

# Unpacking
json_bytes = bytes(tensor.cpu().numpy())          # back to bytes
json_str = json_bytes.decode("utf-8")             # back to string
d = json.loads(json_str)                          # back to dict
```

---

## Valid Blocksizes

### 8-bit Blockwise Quantization

For `quantize_blockwise()` and `dequantize_blockwise()`:

```python
valid_blocksizes_8bit = [64, 128, 256, 512, 1024, 2048, 4096]
# Default: 4096
```

### 4-bit Quantization

For `quantize_4bit()` and `dequantize_4bit()`:

```python
valid_blocksizes_4bit = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# Default: 64
```

Using blocksizes outside these values will result in undefined behavior or errors from the native kernels.

---

## `quant_storage` Dtype

The `quant_storage` parameter controls the dtype of the tensor that stores the packed quantized values. This affects how 4-bit values are packed into bytes.

```python
# Default: each byte stores two 4-bit values
quant_storage = torch.uint8

# Can use other dtypes for different packing strategies
# e.g., torch.float32 stores 8 four-bit values per element
```

For 4-bit quantization with `quant_storage=torch.uint8`:
- Each byte holds two 4-bit values (low nibble + high nibble).
- Output shape: `(ceil(numel / 2), 1)` for a 2D weight matrix.

---

## Double Quantization (compress_statistics)

Double quantization (also called "compress statistics") reduces the memory overhead of the absmax scaling factors themselves.

### Algorithm

1. **Compute offset**: `offset = mean(absmax)`.
2. **Center the absmax**: `centered = absmax - offset`.
3. **Quantize the centered absmax** using block-wise 8-bit quantization with blocksize 256:
   ```python
   qabsmax, state2 = quantize_blockwise(centered, blocksize=256)
   ```
4. **Store** the quantized absmax, offset, and nested state2.

### Memory Savings

Without double quantization:
- `absmax`: `float32` array of size `ceil(N / blocksize)`.

With double quantization:
- `qabsmax`: `uint8` array of size `ceil(N / blocksize)` (4x smaller).
- `offset`: single `float32` scalar.
- `state2.absmax`: `float32` array (much smaller, blocksize 256).
- `state2.code`: 256-element `float32` tensor.

For a 4096x4096 weight matrix with blocksize 64:
- **Without**: `absmax` = 4096*4096/64 * 4 bytes = 1 MB.
- **With**: `qabsmax` = 4096*4096/64 * 1 byte = 256 KB + small overhead.

### Code Example

```python
import torch
import bitsandbytes as bnb

# Without double quantization
q1, state1 = bnb.functional.quantize_4bit(
    weight,
    quant_type="nf4",
    compress_statistics=False,
)

# With double quantization
q2, state2 = bnb.functional.quantize_4bit(
    weight,
    quant_type="nf4",
    compress_statistics=True,  # enables double quantization
)

# state2.nested == True
# state2.offset is the mean of original absmax
# state2.absmax is the 8-bit quantized absmax
# state2.state2 is the QuantState for the nested quantization

# Dequantization automatically handles nested state:
dequantized = bnb.functional.dequantize_4bit(q2, state2)
# Internally: absmax = dequantize_blockwise(state2.absmax, state2.state2) + state2.offset
```

---

## Core Quantization Functions

### `quantize_blockwise()`

```python
def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested: bool = False,
) -> tuple[torch.Tensor, QuantState]
```

Quantizes a tensor using 8-bit block-wise quantization.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `torch.Tensor` | required | Input tensor (float16, bfloat16, or float32) |
| `code` | `torch.Tensor` | `None` | Quantization codebook (defaults to signed 8-bit dynamic map) |
| `absmax` | `torch.Tensor` | `None` | Pre-allocated absmax output tensor |
| `out` | `torch.Tensor` | `None` | Pre-allocated quantized output tensor |
| `blocksize` | `int` | `4096` | Block size (valid: 64, 128, 256, 512, 1024, 2048, 4096) |
| `nested` | `bool` | `False` | If True, additionally quantize the absmax values |

**Returns**: `(quantized_tensor: torch.Tensor[uint8], state: QuantState)`

### `dequantize_blockwise()`

```python
def dequantize_blockwise(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    code: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
    nested: bool = False,
) -> torch.Tensor
```

Dequantizes a block-wise quantized tensor. Either `quant_state` or `absmax` must be provided.

**Returns**: Dequantized tensor with dtype from `quant_state.dtype` (default: `torch.float32`).

### `quantize_4bit()`

```python
def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    compress_statistics: bool = False,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, QuantState]
```

Quantizes a tensor using 4-bit block-wise quantization.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | `torch.Tensor` | required | Input tensor (float16, bfloat16, or float32) |
| `absmax` | `torch.Tensor` | `None` | Pre-allocated absmax output tensor |
| `out` | `torch.Tensor` | `None` | Pre-allocated output tensor |
| `blocksize` | `int` | `None` (defaults to 64) | Block size (valid: 32, 64, 128, 256, 512, 1024, 2048, 4096) |
| `compress_statistics` | `bool` | `False` | Enable double quantization of absmax |
| `quant_type` | `str` | `"fp4"` | `"fp4"` or `"nf4"` |
| `quant_storage` | `torch.dtype` | `torch.uint8` | Storage dtype for packed values |

**Returns**: `(packed_4bit_tensor, QuantState)`

### `dequantize_4bit()`

```python
def dequantize_4bit(
    A: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type: str = "fp4",
) -> torch.Tensor
```

Dequantizes a 4-bit quantized tensor. Either `quant_state` or both `absmax` and `out` must be provided.

**Returns**: Dequantized tensor with the original shape and dtype.

**Special behavior**: If `A.shape[0] == 1` (transposed weight), the output is transposed back before returning.

### Convenience Wrappers

```python
# NF4-specific
quantize_nf4(A, absmax=None, out=None, blocksize=None, compress_statistics=False, quant_storage=torch.uint8)
dequantize_nf4(A, quant_state=None, absmax=None, out=None, blocksize=None)

# FP4-specific
quantize_fp4(A, absmax=None, out=None, blocksize=None, compress_statistics=False, quant_storage=torch.uint8)
dequantize_fp4(A, quant_state=None, absmax=None, out=None, blocksize=None)
```

These are thin wrappers around `quantize_4bit()` and `dequantize_4bit()` with fixed `quant_type`.
