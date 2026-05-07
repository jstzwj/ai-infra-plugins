# Reference 05: NN Modules and Layers

This document provides a comprehensive reference for all neural network module classes in `bitsandbytes.nn.modules`. These modules provide quantized replacements for standard PyTorch layers.

---

## Table of Contents

- [StableEmbedding](#stableembedding)
- [Embedding](#embedding)
- [Linear8bitLt](#linear8bitlt)
- [Linear4bit](#linear4bit)
- [LinearFP4](#linearfp4)
- [LinearNF4](#linearnf4)
- [Int8Params](#int8params)
- [Params4bit](#params4bit)
- [Embedding8bit](#embedding8bit)
- [Embedding4bit](#embedding4bit)
- [EmbeddingFP4](#embeddingfp4)
- [EmbeddingNF4](#embeddingnf4)
- [OutlierAwareLinear](#outlierawarelinear)
- [Utility Functions](#utility-functions)

---

## StableEmbedding

`bitsandbytes.nn.StableEmbedding`

A custom embedding layer designed to improve stability during training for NLP tasks. It wraps `torch.nn.Embedding` with Xavier uniform initialization and an appended `LayerNorm`, and registers the weight parameter with `GlobalOptimManager` for 32-bit optimizer states. This mitigates gradient variance caused by quantization in mixed-precision training.

**Inherits from:** `torch.nn.Embedding`

### Constructor

```python
StableEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    _weight: Optional[Tensor] = None,
    device=None,
    dtype=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_embeddings` | `int` | required | Number of unique embeddings (vocabulary size). |
| `embedding_dim` | `int` | required | Dimensionality of each embedding vector. |
| `padding_idx` | `Optional[int]` | `None` | If specified, pads the output with zeros at the given index. |
| `max_norm` | `Optional[float]` | `None` | Renormalizes embeddings to have a maximum L2 norm. |
| `norm_type` | `float` | `2.0` | The p-norm to compute for the `max_norm` option. |
| `scale_grad_by_freq` | `bool` | `False` | Scale gradient by frequency during backpropagation. |
| `sparse` | `bool` | `False` | If `True`, computes sparse gradients instead of dense. |
| `_weight` | `Optional[Tensor]` | `None` | Pretrained embeddings to initialize with. |
| `device` | | `None` | Device to initialize the layer on. |
| `dtype` | | `None` | Data type of the parameters. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `norm` | `torch.nn.LayerNorm` | Layer normalization applied after embedding lookup. Initialized with `embedding_dim`. |

### Key Behaviors

**Registration with GlobalOptimManager:** On construction, the weight parameter is registered with `GlobalOptimManager.get_instance().register_module_override(self, "weight", {"optim_bits": 32})`. This ensures that the embedding weight always uses 32-bit optimizer states regardless of the global optimizer configuration.

### Methods

#### `reset_parameters() -> None`

Resets the embedding parameters using Xavier uniform initialization (`torch.nn.init.xavier_uniform_`). After initialization, calls `_fill_padding_idx_with_zero()` to zero out the padding index row.

#### `forward(input: Tensor) -> Tensor`

Performs the forward pass through the stable embedding layer:

1. Calls `F.embedding()` with standard PyTorch embedding arguments.
2. Converts the output to `torch.get_default_dtype()` (full precision).
3. Applies `self.norm` (LayerNorm) in full precision.
4. Converts back to `self.weight.dtype` before returning.

```python
# Equivalent pseudocode:
emb = F.embedding(input, self.weight, ...)
emb = emb.to(torch.get_default_dtype())  # always full precision for norm
return self.norm(emb).to(self.weight.dtype)
```

### Usage Example

```python
import torch
import bitsandbytes as bnb

# Create a stable embedding layer
embedding = bnb.nn.StableEmbedding(num_embeddings=50000, embedding_dim=1024)

# The weight is automatically registered for 32-bit optimizer states
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
# embedding.weight will use 32-bit states despite 8-bit optimizer

input_ids = torch.tensor([1, 42, 999])
output = embedding(input_ids)  # Shape: (3, 1024)
```

---

## Embedding

`bitsandbytes.nn.Embedding`

A simplified embedding class that registers weights with `GlobalOptimManager` for 32-bit optimizer states, but does NOT apply LayerNorm. Otherwise identical to `torch.nn.Embedding`.

**Inherits from:** `torch.nn.Embedding`

### Constructor

```python
Embedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    _weight: Optional[Tensor] = None,
    device: Optional[device] = None,
)
```

Parameters are identical to `StableEmbedding` except there is no `dtype` parameter.

### Key Differences from StableEmbedding

- No `LayerNorm` is applied.
- `forward()` is a direct pass-through to `F.embedding()` without dtype conversion.
- Still registered for 32-bit optimizer states via `GlobalOptimManager`.

### Methods

#### `reset_parameters() -> None`

Same as `StableEmbedding`: Xavier uniform initialization followed by padding index zeroing.

#### `forward(input: Tensor) -> Tensor`

Directly returns the result of `F.embedding()` with no additional processing.

---

## Linear8bitLt

`bitsandbytes.nn.Linear8bitLt`

The base module for the [LLM.int8()](https://arxiv.org/abs/2208.07339) algorithm. Supports both on-the-fly quantization during forward pass (training mode) and pre-quantized weights (inference mode).

**Inherits from:** `torch.nn.Linear`

### Constructor

```python
Linear8bitLt(
    input_features: int,
    output_features: int,
    bias: bool = True,
    has_fp16_weights: bool = True,
    threshold: float = 0.0,
    index=None,
    device=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_features` | `int` | required | Number of input features (columns of weight matrix). |
| `output_features` | `int` | required | Number of output features (rows of weight matrix). |
| `bias` | `bool` | `True` | Whether to include a bias term. |
| `has_fp16_weights` | `bool` | `True` | If `False`, weights are quantized to int8 on `.to(device)`. If `True`, weights remain in fp16 and are quantized on-the-fly during each forward pass. |
| `threshold` | `float` | `0.0` | Outlier threshold for mixed-precision decomposition (LLM.int8()). Activation columns where any value exceeds this threshold are computed in fp16 while remaining columns use int8. Set to `0.0` to disable mixed-precision decomposition. |
| `index` | | `None` | Indices for weight reordering (used internally). |
| `device` | | `None` | Device to initialize the layer on. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `state` | `bnb.MatmulLtState` | Internal state object for the LLM.int8() matmul. Contains `threshold`, `has_fp16_weights`, `use_pool`, `CB`, `SCB`. |
| `weight` | `Int8Params` | The weight parameter, stored as an `Int8Params` instance. |

### Methods

#### `init_8bit_state() -> None`

Transfers quantized weight data from `self.weight` to `self.state`. Called on the first forward pass after quantization:

```python
self.state.CB = self.weight.CB    # Quantized weight data
self.state.SCB = self.weight.SCB  # Scaling statistics
self.weight.CB = None
self.weight.SCB = None
```

#### `forward(x: torch.Tensor) -> torch.Tensor`

Performs the forward pass:

1. Sets `self.state.is_training = self.training`.
2. If `self.weight.CB is not None`, calls `init_8bit_state()` (first forward after `.to(device)`).
3. Casts bias to match input dtype if needed.
4. Calls `bnb.matmul(x, self.weight, bias=self.bias, state=self.state)`.
5. If not using fp16 weights and `self.state.CB is not None`, copies `self.state.CB` back to `self.weight.data`.
6. Returns the output tensor.

#### `_save_to_state_dict(destination, prefix, keep_vars) -> None`

Extends the standard save behavior to include quantization statistics:

- Calls `super()._save_to_state_dict()` to save weight and bias.
- Saves `SCB` (row-wise quantization scales) under `prefix + "SCB"`.
- Saves `weight_format` as `torch.tensor(0, dtype=torch.uint8)` indicating row-major format.

#### `_load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None`

Loads a quantized state dict:

- Calls `super()._load_from_state_dict()` for standard parameters.
- Looks for `SCB` key among unexpected keys.
- If the weight's SCB is `None` (not yet quantized), raises `RuntimeError` -- the user must call `.cuda()` before `load_state_dict()`.
- Copies loaded SCB values into both `self.weight.SCB` and `self.state.SCB`.

### Usage Example

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# Create a model with int8 layers
fp16_model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.Linear(4096, 4096),
)

int8_model = nn.Sequential(
    bnb.nn.Linear8bitLt(4096, 4096, has_fp16_weights=False),
    bnb.nn.Linear8bitLt(4096, 4096, has_fp16_weights=False),
)

# Load weights and quantize
int8_model.load_state_dict(fp16_model.state_dict())
int8_model = int8_model.to("cuda")  # Quantization happens here

# Forward pass
x = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
out = int8_model(x)
```

---

## Linear4bit

`bitsandbytes.nn.Linear4bit`

The base module for the [QLoRA](https://arxiv.org/abs/2305.14314) 4-bit quantization algorithm. Supports FP4 and NF4 data types with optional double quantization (compression of quantization statistics).

**Inherits from:** `torch.nn.Linear`

### Constructor

```python
Linear4bit(
    input_features: int,
    output_features: int,
    bias: bool = True,
    compute_dtype=None,
    compress_statistics: bool = True,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    device=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_features` | `int` | required | Number of input features. |
| `output_features` | `int` | required | Number of output features. |
| `bias` | `bool` | `True` | Whether to include a bias term. |
| `compute_dtype` | | `None` | Data type to use for computation. If `None`, determined from input. Supported: `torch.float32`, `torch.float16`, `torch.bfloat16`. |
| `compress_statistics` | `bool` | `True` | Whether to additionally quantize the absmax values (double quantization). Reduces memory for quantization statistics. |
| `quant_type` | `str` | `"fp4"` | The 4-bit quantization type: `"fp4"` (4-bit floating point) or `"nf4"` (4-bit NormalFloat). |
| `quant_storage` | `torch.dtype` | `torch.uint8` | Storage dtype for quantized weights. Can be `torch.uint8`, `torch.float32`, `torch.float16`, or `torch.bfloat16`. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `weight` | `Params4bit` | The weight parameter, stored as a `Params4bit` instance. |
| `compute_dtype` | | The dtype used for forward computation. |
| `compute_type_is_set` | `bool` | Whether `compute_dtype` has been determined from input. |
| `quant_state` | `QuantState` or `None` | Quantization state (set during quantization or loaded from state dict). |
| `quant_storage` | `torch.dtype` | Storage dtype for packed 4-bit values. |
| `support_avx512bf16_for_cpu` | `bool` | Whether CPU supports AVX-512 BF16 for optimized inference. Set by `has_avx512bf16()`. |

### CPU Optimization Path

When `support_avx512bf16_for_cpu` is `True` and the input is on CPU, the layer enters an optimized path during forward:

1. On first forward, calls `_convert_weight_packed_for_cpu()` to repack the weight into an AVX-512-friendly layout.
2. Uses the packed format for subsequent forward passes.
3. When saving, calls `_convert_weight_packed_for_cpu_inverse()` to restore the original format.

### Methods

#### `set_compute_type(x: torch.Tensor) -> None`

Determines the compute dtype from the input tensor on the first forward pass:

- If input is `float32` or `bfloat16`: sets `compute_dtype` to the input dtype.
- If input is `float16`: uses `self.compute_dtype` (or warns if it is `None`/`float32`, since this leads to slow inference).

#### `forward(x: torch.Tensor) -> torch.Tensor`

Performs the forward pass:

1. Calls `fix_4bit_weight_quant_state_from_module(self)` to ensure quantization state is properly attached.
2. On CPU with AVX-512 BF16 support (not training, no gradients): repacks weight for CPU kernel.
3. Casts bias to match input dtype if needed.
4. If `compute_type_is_set` is `False`, calls `set_compute_type(x)`.
5. Casts input to `compute_dtype`.
6. Calls `bnb.matmul_4bit(x, weight, bias=bias, quant_state=quant_state)`.
7. Returns output cast back to the original input dtype.

#### `_save_to_state_dict(destination, prefix, keep_vars) -> None`

Extends the standard save to include all quantization state components:

1. If the weight has a CPU packing format, converts back to standard format.
2. Calls `super()._save_to_state_dict()` for weight and bias.
3. Serializes all `quant_state.as_dict(packed=True)` entries with prefix `"weight."`.

### Usage Example

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

# Create a model with 4-bit layers
fp16_model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.Linear(4096, 4096),
)

# Quantize with NF4 (recommended for best accuracy)
quantized_model = nn.Sequential(
    bnb.nn.Linear4bit(4096, 4096, quant_type="nf4", compute_dtype=torch.bfloat16),
    bnb.nn.Linear4bit(4096, 4096, quant_type="nf4", compute_dtype=torch.bfloat16),
)

quantized_model.load_state_dict(fp16_model.state_dict())
quantized_model = quantized_model.to("cuda")  # Quantization happens here

x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
out = quantized_model(x)
```

---

## LinearFP4

`bitsandbytes.nn.LinearFP4`

A convenience wrapper around `Linear4bit` that hardcodes `quant_type="fp4"`.

**Inherits from:** `Linear4bit`

### Constructor

```python
LinearFP4(
    input_features: int,
    output_features: int,
    bias: bool = True,
    compute_dtype=None,
    compress_statistics: bool = True,
    quant_storage: torch.dtype = torch.uint8,
    device=None,
)
```

All parameters are the same as `Linear4bit` except `quant_type` is always `"fp4"`.

---

## LinearNF4

`bitsandbytes.nn.LinearNF4`

A convenience wrapper around `Linear4bit` that hardcodes `quant_type="nf4"`. NF4 is the recommended quantization type for most use cases, as it is optimized for normally-distributed weight data.

**Inherits from:** `Linear4bit`

### Constructor

```python
LinearNF4(
    input_features: int,
    output_features: int,
    bias: bool = True,
    compute_dtype=None,
    compress_statistics: bool = True,
    quant_storage: torch.dtype = torch.uint8,
    device=None,
)
```

All parameters are the same as `Linear4bit` except `quant_type` is always `"nf4"`.

---

## Int8Params

`bitsandbytes.nn.Int8Params`

A `torch.nn.Parameter` subclass that holds 8-bit quantized weights along with their scaling statistics.

**Inherits from:** `torch.nn.Parameter`

### Constructor (`__new__`)

```python
Int8Params(
    data: Optional[torch.Tensor] = None,
    requires_grad: bool = True,
    has_fp16_weights: bool = False,
    CB: Optional[torch.Tensor] = None,
    SCB: Optional[torch.Tensor] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `Optional[torch.Tensor]` | `None` | The underlying tensor data. If `None`, creates an empty tensor. |
| `requires_grad` | `bool` | `True` | Whether the parameter requires gradients. |
| `has_fp16_weights` | `bool` | `False` | If `True`, weights stay in fp16 (quantized on-the-fly). If `False`, weights are quantized to int8 on `.to(device)`. |
| `CB` | `Optional[torch.Tensor]` | `None` | The quantized weight tensor (int8). |
| `SCB` | `Optional[torch.Tensor]` | `None` | The row-wise quantization scales (float32). |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `CB` | `Optional[torch.Tensor]` | Quantized weight data (int8). Set during `_quantize()`. |
| `SCB` | `Optional[torch.Tensor]` | Row-wise quantization scales (float32). Set during `_quantize()`. |
| `has_fp16_weights` | `bool` | Whether weights remain in fp16 (on-the-fly quantization mode). |

### Methods

#### `_quantize(device) -> Int8Params`

Quantizes the weight tensor to int8:

1. If `has_fp16_weights` is `True`, returns `super().to(device)` (no quantization).
2. Otherwise:
   - Moves data to device as float16.
   - Calls `bnb.functional.int8_vectorwise_quant(B)` to get `CB` (int8 weights) and `SCB` (scales).
   - Sets `self.data = CB`, `self.CB = CB`, `self.SCB = SCB`.

#### `to(*args, **kwargs) -> Int8Params`

Handles device transfers and quantization:

1. Parses target device and dtype.
2. If data is not yet quantized (not int8) and moving to a non-CPU, non-meta device: calls `_quantize(device)`.
3. Otherwise creates a new `Int8Params` on the target device and transfers `CB`/`SCB` if already quantized.

#### `cpu() -> Int8Params`

Equivalent to `self.to(device="cpu")`.

#### `cuda(device=None, non_blocking=False) -> Int8Params`

Equivalent to `self.to(device="cuda")` (or the specified CUDA device).

#### `xpu(device=None, non_blocking=False) -> Int8Params`

Equivalent to `self.to(device="xpu")` (or the specified XPU device).

#### `__deepcopy__(memo) -> Int8Params`

Creates a deep copy of the parameter, including `data`, `CB`, `SCB`, and `has_fp16_weights`.

---

## Params4bit

`bitsandbytes.nn.Params4bit`

A `torch.nn.Parameter` subclass that holds 4-bit quantized weights along with their quantization state.

**Inherits from:** `torch.nn.Parameter`

### Constructor (`__new__`)

```python
Params4bit(
    data: Optional[torch.Tensor] = None,
    requires_grad: bool = False,
    quant_state: Optional[QuantState] = None,
    blocksize: Optional[int] = None,
    compress_statistics: bool = True,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    module: Optional[Linear4bit] = None,
    bnb_quantized: bool = False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `Optional[torch.Tensor]` | `None` | The underlying tensor data. If `None`, creates an empty tensor. |
| `requires_grad` | `bool` | `False` | Quantized weights are frozen by default. |
| `quant_state` | `Optional[QuantState]` | `None` | The quantization state object. |
| `blocksize` | `Optional[int]` | `None` | Block size for quantization. Defaults to `64` if `None`. |
| `compress_statistics` | `bool` | `True` | Whether to use double quantization on absmax values. |
| `quant_type` | `str` | `"fp4"` | Quantization type: `"fp4"` or `"nf4"`. |
| `quant_storage` | `torch.dtype` | `torch.uint8` | Storage dtype for packed 4-bit values. |
| `module` | `Optional[Linear4bit]` | `None` | Reference to the parent module for quantization state recovery. |
| `bnb_quantized` | `bool` | `False` | Whether the data has been quantized. |

### FSDP Compatibility Properties

The following properties proxy attributes from `quant_state` to support FSDP state dict traversal. FSDP resolves dotted FQN keys via `getattr`, so these `@property` descriptors (rather than `__getattr__`) avoid `torch.compile` graph breaks:

| Property | Returns | Description |
|---|---|---|
| `absmax` | `Tensor` | Absolute maximum values per block. |
| `code` | `Tensor` | Quantization lookup table. |
| `quant_map` | `Tensor` | Alias for `code`. |
| `offset` | `Tensor` | Mean offset for nested quantization. |
| `state2` | `QuantState` | Nested quantization state (for double quantization). |
| `nested_absmax` | `Tensor` | Absmax of the nested (double-quantized) state. |
| `nested_blocksize` | `int` | Blocksize of the nested state. |
| `nested_quant_map` | `Tensor` | Quant map of the nested state. |
| `nested_dtype` | `torch.dtype` | Dtype of the nested state. |
| `nested_offset` | `Tensor` | Offset of the nested state (same as top-level). |

All properties raise `AttributeError` if `quant_state` is `None`.

### Methods

#### `_quantize(device) -> Params4bit`

Quantizes the weight data to 4-bit:

1. Makes data contiguous and moves to target device.
2. Calls `bnb.functional.quantize_4bit()` with configured `blocksize`, `compress_statistics`, `quant_type`, `quant_storage`.
3. Sets `self.data` to the packed 4-bit tensor and `self.quant_state` to the returned `QuantState`.
4. Updates `self.module.quant_state` if module reference exists.
5. Sets `self.bnb_quantized = True`.

#### `from_prequantized(data, quantized_stats, requires_grad=False, device="cuda", module=None) -> Params4bit`

Class method to create a `Params4bit` from pre-quantized data:

```python
Params4bit.from_prequantized(
    data: torch.Tensor,
    quantized_stats: dict[str, Any],
    requires_grad: bool = False,
    device="cuda",
    module=None,
)
```

Reconstructs the `QuantState` from the serialized dictionary and attaches it to the parameter.

#### `to(*args, **kwargs) -> Params4bit`

Handles device transfers and quantization:

1. If moving to a non-meta device and not yet quantized (`bnb_quantized=False`): calls `_quantize(device)`.
2. If already quantized: moves `quant_state` to target device, creates a new `Params4bit` with updated data and state.

#### `cuda(device=None, non_blocking=False) -> Params4bit`

Handles CPU-packed format conversion before moving to CUDA. If the weight has a `packing_format_for_cpu` flag, calls `_convert_weight_packed_for_cpu_inverse()` first.

#### `xpu(device=None, non_blocking=False) -> Params4bit`

Same as `cuda()` but for XPU devices.

#### `cpu() -> Params4bit`

Equivalent to `self.to(device="cpu")`.

#### `__deepcopy__(memo) -> Params4bit`

Creates a deep copy including `quant_state` and `data`.

#### `__torch_function__(cls, func, types, args, kwargs)`

Intercepts `torch.chunk` and `torch.split` operations to preserve `Params4bit` type and metadata across the resulting chunks.

---

## Embedding8bit

`bitsandbytes.nn.Embedding8bit`

Implements [LLM.int8()](https://arxiv.org/abs/2208.07339) for embedding layers. The weight matrix is stored as int8 with per-row scaling statistics.

**Inherits from:** `torch.nn.Embedding`

### Constructor

```python
Embedding8bit(
    num_embeddings: int,
    embedding_dim: int,
    device=None,
    dtype=None,
)
```

### Forward Pass

```python
def forward(input: Tensor) -> Tensor:
    # 1. Check that SCB exists (quantization has occurred)
    # 2. Perform standard embedding lookup on int8 weights
    compressed_output = F.embedding(input, rows)
    # 3. Look up the row-wise scales for the selected indices
    compressed_output_stats = F.embedding(input, row_stats.view(num_embeddings, 1))
    # 4. Dequantize by scaling
    output = compressed_output * (compressed_output_stats / 127.0)
    return output.to(self.dtype)
```

The dequantization formula is: `output = int8_weight * (row_scale / 127.0)` where `127.0` is the constant `C` defined in `functional.py`.

### Restrictions

- `_save_to_state_dict()` raises `NotImplementedError`.
- Forward raises `RuntimeError` if SCB is not set (layer not yet quantized).

### Usage Example

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

fp16_module = nn.Embedding(128, 64)
int8_module = bnb.nn.Embedding8bit(128, 64)
int8_module.load_state_dict(fp16_module.state_dict())
int8_module = int8_module.to("cuda")  # Quantization happens here

ids = torch.tensor([0, 5, 42], device="cuda")
out = int8_module(ids)  # Shape: (3, 64), dtype: float16
```

---

## Embedding4bit

`bitsandbytes.nn.Embedding4bit`

Implements [QLoRA](https://arxiv.org/abs/2305.14314) 4-bit quantization for embedding layers.

**Inherits from:** `torch.nn.Embedding`

### Constructor

```python
Embedding4bit(
    num_embeddings: int,
    embedding_dim: int,
    dtype=None,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
    device=None,
)
```

If `embedding_dim % blocksize != 0`, a warning is logged because the partial dequantization path cannot be used.

### Forward Pass

The forward pass has two paths:

#### Fast Path: `_forward_with_partial_dequantize(input)`

Used when `embedding_dim % quant_state.blocksize == 0`. Only dequantizes the rows needed for the input:

1. Views the packed 4-bit data as `uint8` and selects only the required rows via `F.embedding`.
2. Selects the corresponding absmax values via `F.embedding`.
3. Creates a shallow copy of the quantization state with the subset absmax.
4. Dequantizes only the selected rows.
5. Returns the result cast to `self.dtype`.

#### Fallback Path

Used when the embedding dimension is not evenly divisible by the blocksize:

1. Fully dequantizes the entire weight matrix: `dequantized_weight = bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state)`.
2. Performs standard `F.embedding` lookup on the dequantized weight.

### Usage Example

```python
import torch
import torch.nn as nn
import bitsandbytes as bnb

fp16_module = nn.Embedding(128, 64)
quantized_module = bnb.nn.Embedding4bit(128, 64, quant_type="nf4")
quantized_module.load_state_dict(fp16_module.state_dict())
quantized_module = quantized_module.to("cuda")  # Quantization happens here

ids = torch.tensor([0, 5, 42], device="cuda")
out = quantized_module(ids)  # Shape: (3, 64)
```

---

## EmbeddingFP4

`bitsandbytes.nn.EmbeddingFP4`

Convenience wrapper around `Embedding4bit` with `quant_type="fp4"`.

**Inherits from:** `Embedding4bit`

### Constructor

```python
EmbeddingFP4(
    num_embeddings: int,
    embedding_dim: int,
    dtype=None,
    quant_storage: torch.dtype = torch.uint8,
    device=None,
)
```

---

## EmbeddingNF4

`bitsandbytes.nn.EmbeddingNF4`

Convenience wrapper around `Embedding4bit` with `quant_type="nf4"`.

**Inherits from:** `Embedding4bit`

### Constructor

```python
EmbeddingNF4(
    num_embeddings: int,
    embedding_dim: int,
    dtype=None,
    quant_storage: torch.dtype = torch.uint8,
    device=None,
)
```

---

## OutlierAwareLinear

`bitsandbytes.nn.OutlierAwareLinear`

An abstract base class for custom outlier handling in linear layers. Integrates with `OutlierTracer` for automatic outlier detection.

**Inherits from:** `torch.nn.Linear`

### Constructor

```python
OutlierAwareLinear(
    input_features: int,
    output_features: int,
    bias: bool = True,
    device=None,
)
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `outlier_dim` | `None` (initially) | Stores the detected outlier column indices after first forward pass. |
| `is_quantized` | `bool` | Whether the weight has been quantized. Initially `False`. |

### Abstract Methods

These methods must be overridden by subclasses:

#### `forward_with_outliers(x, outlier_idx) -> Tensor`

Defines how to compute the output for outlier dimensions. Must be implemented by the subclass.

#### `quantize_weight(w, outlier_idx) -> Tensor`

Defines how to quantize the weight while accounting for outlier dimensions. Must be implemented by the subclass.

### Forward Pass

```python
def forward(self, x):
    # 1. On first call, detect outlier dimensions via OutlierTracer
    if self.outlier_dim is None:
        tracer = OutlierTracer.get_instance()
        outlier_idx = tracer.get_outliers(self.weight)
        self.outlier_dim = outlier_idx

    # 2. On first call, quantize the weight
    if not self.is_quantized:
        w = self.quantize_weight(self.weight, self.outlier_dim)
        self.weight.data.copy_(w)
        self.is_quantized = True
```

### OutlierTracer Integration

The `OutlierTracer` (from `bitsandbytes.utils`) is a singleton that identifies outlier features across model layers. It pools outlier dimensions to handle small models where outlier features are less systematic.

---

## Utility Functions

### `fix_4bit_weight_quant_state_from_module(module)`

```python
fix_4bit_weight_quant_state_from_module(
    module: Union[Embedding4bit, Linear4bit]
) -> None
```

Recovers the quantization state for a 4-bit module when the `Params4bit` has lost its `quant_state` reference. This can happen during FSDP parameter flattening or other transformations.

- If `module.weight.quant_state` is already set, returns immediately.
- If `module.quant_state` is `None`, logs a warning.
- Creates a new `Params4bit` wrapper if the weight is not already one.
- Copies `module.quant_state` to `module.weight.quant_state`.

### `maybe_rearrange_weight(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)`

A load hook registered on `Linear8bitLt` that handles weight format conversion during state dict loading. Extracts and validates the `weight_format` key from the state dict (only `"row"` format is supported).

---

## Module Comparison Table

| Module | Quantization | Bits | Use Case |
|---|---|---|---|
| `StableEmbedding` | None (32-bit weights) | N/A | Training embeddings with stable gradients |
| `Embedding` | None (32-bit weights) | N/A | Training embeddings without LayerNorm |
| `Linear8bitLt` | LLM.int8() | 8-bit | Inference and mixed-precision training |
| `Linear4bit` | QLoRA | 4-bit (FP4/NF4) | Memory-efficient fine-tuning (LoRA/QLoRA) |
| `LinearFP4` | QLoRA (FP4) | 4-bit | Convenience wrapper for FP4 |
| `LinearNF4` | QLoRA (NF4) | 4-bit | Convenience wrapper for NF4 (recommended) |
| `Embedding8bit` | LLM.int8() | 8-bit | Memory-efficient embedding inference |
| `Embedding4bit` | QLoRA | 4-bit (FP4/NF4) | Memory-efficient embedding for QLoRA |
| `EmbeddingFP4` | QLoRA (FP4) | 4-bit | Convenience wrapper for FP4 embeddings |
| `EmbeddingNF4` | QLoRA (NF4) | 4-bit | Convenience wrapper for NF4 embeddings |
| `OutlierAwareLinear` | Custom (abstract) | Custom | Research/extensibility for outlier handling |
