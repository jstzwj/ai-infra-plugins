# FSDP Integration Reference

This document provides a comprehensive reference for how bitsandbytes integrates with PyTorch's Fully Sharded Data Parallel (FSDP) training. It covers the serialization and deserialization of quantization state, property descriptors for FSDP traversal, optimizer state dict wrapping, and the QLoRA + FSDP training pattern.

---

## Table of Contents

1. [FSDP Compatibility Challenges](#fsdp-compatibility-challenges)
2. [QuantState Serialization](#quantstate-serialization)
3. [QuantState Deserialization](#quantstate-deserialization)
4. [Params4bit FSDP Properties](#params4bit-fsdp-properties)
5. [Linear4bit State Dict](#linear4bit-state-dict)
6. [Optimizer8bit FSDP Compatibility](#optimizer8bit-fsdp-compatibility)
7. [fix_4bit_weight_quant_state_from_module](#fix_4bit_weight_quant_state_from-module)
8. [QLoRA + FSDP Pattern](#qlora--fsdp-pattern)

---

## FSDP Compatibility Challenges

FSDP works by sharding model parameters, optimizer states, and gradients across ranks. It uses `state_dict()` to gather and save the full model state. This creates several challenges for quantized models:

### Challenge 1: Non-Tensor State

Quantization state includes non-tensor data types (strings, ints, `torch.dtype`) that cannot be stored in safetensors format. bitsandbytes solves this with packed serialization.

### Challenge 2: FSDP's _get_fqns() Traversal

FSDP's internal `_get_fqns()` function resolves fully qualified names (FQNs) by traversing dotted keys via `getattr()`. For a key like `weight.absmax`, it calls `getattr(weight, "absmax")`. This means the `Params4bit` parameter must expose quantization state attributes as properties.

### Challenge 3: Parameter Conversion

When FSDP flattens and shards parameters, it may convert `Params4bit` instances to plain tensors, losing the `quant_state`. bitsandbytes recovers this via a module-level fallback attribute.

### Challenge 4: Optimizer State Shape Mismatch

FSDP's `full_optim_state_dict` gathers all tensor states across ranks. However, 8-bit optimizer states (uint8 quantized states, absmax vectors, qmap tensors) have different shapes than model parameters, causing gather operations to fail.

---

## QuantState Serialization

### as_dict(packed=False)

Returns a raw dictionary with tensors and Python objects (strings, ints, dtypes):

```python
qs_dict = {
    "quant_type": "nf4",                        # str
    "absmax": self.absmax,                       # Tensor
    "blocksize": 64,                             # int
    "quant_map": self.code,                      # Tensor
    "dtype": "float16",                          # str
    "shape": (4096, 4096),                       # tuple[int, ...]
}
```

This format is suitable for `torch.save()` (pickle-based) but **not** for safetensors, which only supports tensor values.

### as_dict(packed=True)

Returns a dictionary where **all values are tensors**, making it compatible with safetensors format:

```python
qs_packed_dict = {
    # Tensor values pass through unchanged
    "absmax": self.absmax,                       # Tensor (float32)
    "quant_map": self.code,                      # Tensor (float32)

    # Non-tensor values packed into a single uint8 tensor
    "quant_state.bitsandbytes__nf4": pack_dict_to_tensor({
        "quant_type": "nf4",
        "blocksize": 64,
        "dtype": "float16",
        "shape": (4096, 4096),
    }),                                          # Tensor (uint8)
}
```

#### Packed Key Format

The packed key follows the pattern:

```
quant_state.bitsandbytes__{quant_type}
```

Examples:
- `"quant_state.bitsandbytes__nf4"` for NF4 quantization
- `"quant_state.bitsandbytes__fp4"` for FP4 quantization

#### pack_dict_to_tensor

```python
def pack_dict_to_tensor(source_dict):
    """Pack a dictionary into a torch tensor for state_dict storage."""
    json_str = json.dumps(source_dict)
    json_bytes = json_str.encode("utf-8")
    tensor_data = torch.tensor(list(json_bytes), dtype=torch.uint8)
    return tensor_data
```

The packing process:
1. Serialize the dict to a JSON string
2. Encode the JSON string to UTF-8 bytes
3. Convert each byte to a uint8 tensor element

#### unpack_tensor_to_dict

```python
def unpack_tensor_to_dict(tensor_data):
    """Unpack a torch tensor into a Python dictionary."""
    json_bytes = bytes(tensor_data.cpu().numpy())
    json_str = json_bytes.decode("utf-8")
    return json.loads(json_str)
```

The reverse process:
1. Convert the uint8 tensor to a byte array via numpy
2. Decode the bytes as a UTF-8 JSON string
3. Parse the JSON back into a Python dictionary

#### Nested State (Double Quantization)

When `compress_statistics=True` (double quantization), the packed dict includes additional nested fields:

```python
qs_dict = {
    "quant_type": "nf4",
    "absmax": qabsmax,                           # uint8 (quantized absmax)
    "blocksize": 64,
    "quant_map": code,                           # float32
    "dtype": "float16",
    "shape": (4096, 4096),
    # Nested state components
    "nested_absmax": state2.absmax,              # float32 (second-level absmax)
    "nested_blocksize": 256,                     # int
    "nested_quant_map": state2.code.clone(),     # float32 (un-shared to avoid safetensors issues)
    "nested_dtype": "float32",                   # str
    "nested_offset": offset.item(),              # float
}
```

Note that `nested_quant_map` is explicitly cloned to avoid shared tensor issues when safetensors deduplicates tensors.

---

## QuantState Deserialization

### QuantState.from_dict(qs_dict, device)

Reconstructs a `QuantState` from a dictionary, handling both packed and unpacked formats.

```python
@classmethod
def from_dict(cls, qs_dict: dict[str, Any], device: torch.device) -> "QuantState":
```

#### Step 1: Detect Packed Format

```python
qs_key = [k for k, v in qs_dict.items()
          if "quant_state" in k and isinstance(v, torch.Tensor)]
```

Finds the packed key (e.g., `"quant_state.bitsandbytes__nf4"`).

#### Step 2: Validate and Unpack

```python
if "quant_type" not in qs_dict:
    if not qs_key:
        raise ValueError("Expected packed or unpacked quant_state items, found neither")
    if len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
        raise ValueError(
            f"There should be exactly one `quant_state` item with ending from "
            f"{cls.valid_qs_type_keys}. Detected {qs_key}."
        )

if len(qs_key) == 1:
    first_qs_key = qs_key[0]
    qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))
```

#### Step 3: Strip Prefixes

```python
qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}
```

Converts keys like `"quant_state.bitsandbytes__nf4"` to just the final component where applicable.

#### Step 4: Validate Keys

```python
assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)
```

Valid keys: `absmax`, `quant_map`, `nested_absmax`, `nested_quant_map`, `quant_state`, `quant_type`, `blocksize`, `dtype`, `shape`, `nested_blocksize`, `nested_dtype`, `nested_offset`.

#### Step 5: Reconstruct Nested State

```python
if "nested_absmax" in qs_dict:
    offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
    state2 = cls(
        absmax=qs_dict["nested_absmax"].to(device),
        blocksize=qs_dict["nested_blocksize"],
        code=qs_dict["nested_quant_map"].to(device),
        dtype=getattr(torch, qs_dict["nested_dtype"]),
    )
else:
    offset, state2 = None, None
```

#### Step 6: Create QuantState

```python
quant_state = cls(
    quant_type=qs_dict["quant_type"],
    absmax=qs_dict["absmax"].to(device),
    blocksize=qs_dict["blocksize"],
    code=qs_dict["quant_map"].to(device),
    dtype=getattr(torch, qs_dict["dtype"]),
    shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
    offset=offset,
    state2=state2,
)
```

### QuantState.__getattr__ for FSDP Traversal

The `QuantState` class also supports FSDP's dotted FQN traversal:

```python
def __getattr__(self, name):
    if name.startswith("bitsandbytes__"):
        qs_dict = self.as_dict(packed=True)
        packed_key = "quant_state." + name
        if packed_key in qs_dict:
            return qs_dict[packed_key]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

This allows FSDP to resolve keys like `quant_state.bitsandbytes__nf4` by calling `getattr(quant_state, "bitsandbytes__nf4")`.

---

## Params4bit FSDP Properties

`Params4bit` exposes `QuantState` attributes as `@property` descriptors so FSDP can traverse them during state dict operations.

### Why @property Instead of __getattr__

From the source comments:

> Using @property instead of __getattr__ avoids torch.compile graph breaks (#1904), since Dynamo can trace descriptor protocol access but not __getattr__ on Tensor subclasses.

### Property List

| Property | Proxied From | Type | Description |
|----------|-------------|------|-------------|
| `absmax` | `quant_state.absmax` | Tensor | Block absolute maxima |
| `code` | `quant_state.code` | Tensor | Quantization code map |
| `quant_map` | `quant_state.code` | Tensor | Alias for `code` |
| `offset` | `quant_state.offset` | Tensor or None | Mean offset (double quant) |
| `state2` | `quant_state.state2` | QuantState or None | Nested quantization state |
| `nested_absmax` | `state2.absmax` | Tensor | Second-level absmax |
| `nested_blocksize` | `state2.blocksize` | int | Second-level blocksize |
| `nested_quant_map` | `state2.code` | Tensor | Second-level code map |
| `nested_dtype` | `state2.dtype` | torch.dtype | Second-level dtype |
| `nested_offset` | `quant_state.offset` | Tensor or None | Nested offset |

### Implementation Pattern

All properties follow the same pattern:

```python
@property
def absmax(self):
    qs = self.__dict__.get("quant_state")
    if qs is not None:
        return qs.absmax
    raise AttributeError(f"'{type(self).__name__}' object has no attribute 'absmax'")
```

Key details:
- Uses `self.__dict__.get("quant_state")` instead of `self.quant_state` to avoid infinite recursion through `__getattr__`
- Raises `AttributeError` if `quant_state` is not set, which is the correct behavior for FSDP's attribute traversal
- Nested properties (`nested_*`) check both `quant_state` and `state2` before raising

### Attributes Intentionally Omitted

Some attributes are intentionally not exposed as properties because they collide with `Params4bit` instance attributes or `torch.Tensor` attributes:

- `blocksize` -- exists as a `Params4bit` instance attribute
- `quant_type` -- exists as a `Params4bit` instance attribute
- `dtype` -- exists as a `torch.Tensor` attribute
- `shape` -- exists as a `torch.Tensor` attribute

These are instead packed into the `bitsandbytes__*` blob and not traversed by FSDP individually.

### FSDP State Dict Key Structure

When FSDP traverses a `Params4bit` weight, it generates these state dict keys:

```
weight                                  # The quantized uint8 data (from Tensor.data)
weight.absmax                           # Block absolute maxima (from property)
weight.code                             # Quantization code map (from property)
weight.quant_map                        # Alias for code (from property)
weight.quant_state.bitsandbytes__nf4   # Packed metadata tensor
weight.nested_absmax                    # (if double quantization)
weight.nested_blocksize                 # (if double quantization)
weight.nested_quant_map                 # (if double quantization)
weight.nested_dtype                     # (if double quantization)
weight.nested_offset                    # (if double quantization)
```

---

## Linear4bit State Dict

### _save_to_state_dict

The `Linear4bit._save_to_state_dict` method handles saving both the weight data and the quantization state:

```python
def _save_to_state_dict(self, destination, prefix, keep_vars):
    # Step 1: Unpack CPU format if needed
    if getattr(self.weight, "quant_state", None) is not None and getattr(
        self.weight.quant_state, "packing_format_for_cpu", False
    ):
        self.weight.data, self.weight.quant_state = _convert_weight_packed_for_cpu_inverse(
            self.weight.data, self.weight.quant_state
        )

    # Step 2: Save weight and bias (standard nn.Linear)
    super()._save_to_state_dict(destination, prefix, keep_vars)

    # Step 3: Save quantization state in packed format
    if getattr(self.weight, "quant_state", None) is not None:
        for k, v in self.weight.quant_state.as_dict(packed=True).items():
            destination[prefix + "weight." + k] = v if keep_vars else v.detach()
```

### CPU Packing Format Handling

When the weight is in CPU packing format (optimized for AVX-512 BF16 inference), it is first unpacked back to the standard format before saving. This ensures compatibility with all loading scenarios:

```python
if getattr(self.weight.quant_state, "packing_format_for_cpu", False):
    self.weight.data, self.weight.quant_state = _convert_weight_packed_for_cpu_inverse(
        self.weight.data, self.weight.quant_state
    )
```

The CPU packing format reorganizes the 4-bit weight data for efficient AVX-512 BF16 computation. The inverse function (`_convert_weight_packed_for_cpu_inverse`) reverses this reorganization and restores the standard packed 4-bit layout.

### Resulting State Dict Keys

For a `Linear4bit` layer at prefix `model.layers.0.self_attn.q_proj.`:

```
model.layers.0.self_attn.q_proj.weight                          # Quantized uint8 data
model.layers.0.self_attn.q_proj.weight.absmax                   # float32 tensor
model.layers.0.self_attn.q_proj.weight.quant_map                # float32 tensor
model.layers.0.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4  # uint8 packed metadata
model.layers.0.self_attn.q_proj.bias                            # (optional) float bias
```

With double quantization (`compress_statistics=True`):

```
model.layers.0.self_attn.q_proj.weight.absmax                   # uint8 (quantized)
model.layers.0.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4  # packed metadata
model.layers.0.self_attn.q_proj.weight.nested_absmax            # float32
model.layers.0.self_attn.q_proj.weight.nested_blocksize         # (packed in blob)
model.layers.0.self_attn.q_proj.weight.nested_quant_map         # float32
model.layers.0.self_attn.q_proj.weight.nested_dtype             # (packed in blob)
model.layers.0.self_attn.q_proj.weight.nested_offset            # (packed in blob)
```

---

## Optimizer8bit FSDP Compatibility

The `Optimizer8bit` class wraps and unwraps quantization-specific tensor states in its `state_dict()` and `load_state_dict()` methods for FSDP compatibility.

### The Problem

FSDP's `full_optim_state_dict` gathers all tensor states across ranks using all-gather operations. For 8-bit optimizers, the state contains tensors with different shapes than model parameters:

| State Key | Shape | Notes |
|-----------|-------|-------|
| `state1` | `(numel,)` uint8 | Quantized optimizer state (e.g., Adam's m) |
| `state2` | `(numel,)` uint8 | Quantized optimizer state (e.g., Adam's v) |
| `absmax1` | `(numel/256,)` float32 | Block absmax for state1 |
| `absmax2` | `(numel/256,)` float32 | Block absmax for state2 |
| `qmap1` | `(256,)` float32 | Quantization map |
| `qmap2` | `(256,)` float32 | Quantization map |
| `step` | scalar int | Step counter |

The `absmax`, `qmap`, and uint8 state tensors have incompatible shapes for gathering across ranks, since FSDP expects all optimizer state tensors to have the same shape as the corresponding parameter (or to be scalar).

### Solution: Nested Wrapping

#### state_dict() -- Wrapping

```python
_FSDP_WRAPPED_QUANT_STATE_KEY = "__bnb_optimizer_quant_state__"

def state_dict(self):
    state_dict = super().state_dict()

    # Deep copy the state to avoid modifying the original optimizer state
    # PyTorch's state_dict() only does a shallow copy
    state_dict["state"] = {
        k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in state_dict["state"].items()
    }

    # Wrap quantization-specific tensors in a nested dict to hide from FSDP
    for param_state in state_dict["state"].values():
        if isinstance(param_state, dict):
            quant_state = {}
            keys_to_wrap = [k for k in param_state
                           if k in self.non_castable_tensor_keys]
            for key in keys_to_wrap:
                quant_state[key] = param_state.pop(key)
            if quant_state:
                param_state[self._FSDP_WRAPPED_QUANT_STATE_KEY] = quant_state

    return state_dict
```

**Before wrapping:**
```python
{
    "state": {
        0: {
            "step": 100,
            "state1": tensor([...], dtype=uint8),
            "state2": tensor([...], dtype=uint8),
            "absmax1": tensor([...], dtype=float32),
            "absmax2": tensor([...], dtype=float32),
            "qmap1": tensor([...], dtype=float32),
            "qmap2": tensor([...], dtype=float32),
        }
    },
    "param_groups": [...]
}
```

**After wrapping:**
```python
{
    "state": {
        0: {
            "step": 100,
            "__bnb_optimizer_quant_state__": {
                "state1": tensor([...], dtype=uint8),
                "state2": tensor([...], dtype=uint8),
                "absmax1": tensor([...], dtype=float32),
                "absmax2": tensor([...], dtype=float32),
                "qmap1": tensor([...], dtype=float32),
                "qmap2": tensor([...], dtype=float32),
            }
        }
    },
    "param_groups": [...]
}
```

By nesting these tensors inside a dict value keyed by `"__bnb_optimizer_quant_state__"`, FSDP treats the entire nested dict as a single opaque value and does not attempt to gather individual tensors.

#### load_state_dict() -- Unwrapping

```python
def load_state_dict(self, state_dict, move_to_device=True):
    state_dict = deepcopy(state_dict)

    # Unwrap quantization states that were wrapped for FSDP compatibility
    for param_state in state_dict["state"].values():
        if isinstance(param_state, dict) and self._FSDP_WRAPPED_QUANT_STATE_KEY in param_state:
            quant_state = param_state.pop(self._FSDP_WRAPPED_QUANT_STATE_KEY)
            param_state.update(quant_state)

    # ... standard PyTorch optimizer load_state_dict logic
    # Validates param groups, casts tensors to correct devices and dtypes
```

### Non-Castable Tensor Keys

The keys that are wrapped (because they have incompatible shapes or dtypes for FSDP gathering):

```python
self.non_castable_tensor_keys = {
    "qmap1", "qmap2",       # 256-element float32 quantization maps
    "max1", "max2",          # (unused in current implementation)
    "new_max1", "new_max2",  # (unused in current implementation)
    "state1", "state2",      # uint8 quantized states
    "gnorm_vec",             # Gradient norm vector
    "absmax1", "absmax2",    # Block absmax vectors (different shape from param)
    "unorm_vec",             # Update norm vector
}
```

During `load_state_dict`, these keys are handled specially by the `cast()` function:

```python
def cast(param, value):
    if isinstance(value, torch.Tensor):
        if param.is_floating_point() and value.dtype != torch.uint8:
            value = value.to(param.dtype)
        return value
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in self.non_castable_tensor_keys:
                if move_to_device:
                    value[k] = v.to(param.device)
            else:
                value[k] = cast(param, v)
        return value
```

Quantization tensors (uint8, absmax, qmap) are moved to the correct device but not cast to the parameter's dtype.

---

## fix_4bit_weight_quant_state_from_module

This function recovers the lost `quant_state` after FSDP parameter conversion.

### The Problem

When FSDP flattens and shards parameters, it may convert `Params4bit` instances to plain tensors:

```python
# Before FSDP
module.weight  # Params4bit with quant_state attribute

# After FSDP parameter conversion
module.weight  # Plain tensor or different Params4bit without quant_state
```

This happens because FSDP's `FlatParameter` management creates new tensor views that do not preserve custom subclass attributes.

### The Solution

```python
def fix_4bit_weight_quant_state_from_module(module):
    # If weight already has quant_state, nothing to do
    if getattr(module.weight, "quant_state", None) is not None:
        return

    # Check module-level fallback
    if getattr(module, "quant_state", None) is None:
        logger.warning(
            "FP4 quantization state not initialized. "
            "Please call .cuda() or .to(device) on the LinearFP4 layer first."
        )

    # Recover from module-level quant_state
    assert module.weight.shape[1] == 1  # Packed 4-bit format has shape [N, 1]

    if not isinstance(module.weight, Params4bit):
        module.weight = Params4bit(
            module.weight,
            quant_storage=module.quant_storage,
            bnb_quantized=True,
        )
    module.weight.quant_state = module.quant_state
```

### How Module-Level quant_state Works

When `Params4bit._quantize()` is called (triggered by `.to(device)`), it stores the quantization state on both the parameter and the module:

```python
# In Params4bit._quantize()
self.quant_state = quant_state
if self.module is not None:
    self.module.quant_state = quant_state
```

This dual storage ensures that even if the parameter is converted by FSDP, the quantization state can be recovered from the module.

### When Is This Called

The function is called at the start of every `Linear4bit.forward()`:

```python
class Linear4bit(nn.Linear):
    def forward(self, x):
        fix_4bit_weight_quant_state_from_module(self)
        quant_state = self.weight.quant_state
        # ... rest of forward
```

The check is cheap (a single `getattr` call) when the state is already present, so the overhead is negligible in the common case.

### Shape Assertion

The assertion `module.weight.shape[1] == 1` verifies that the weight is in the packed 4-bit format. In this format, each pair of 4-bit values is packed into a single uint8 element, resulting in a tensor with shape `[num_packed_elements, 1]`.

---

## QLoRA + FSDP Pattern

The typical QLoRA + FSDP training workflow:

### Step 1: Load Model with 4-bit Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map={"": "cpu"},  # Load to CPU first for FSDP
)
```

At this point:
- Each `Linear4bit` layer has a `Params4bit` weight with `quant_state`
- The module also stores `quant_state` as a module attribute (set during `_quantize()`)
- `bnb_quantized=True` on the parameter

### Step 2: Add LoRA Adapters

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
```

Only the LoRA adapter parameters are trainable; the 4-bit base weights remain frozen.

### Step 3: Wrap with FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=policy,
    device_id=rank,
)
```

FSDP will:
1. Flatten parameters into `FlatParameter` groups
2. Shard each flat parameter across ranks
3. During forward/backward, all-gather the needed shard

The 4-bit weights are handled correctly because:
- `Params4bit` exposes `quant_state` attributes via `@property` descriptors
- FSDP's `_get_fqns()` can traverse these properties
- State dict serialization uses the packed format for safetensors compatibility
- `fix_4bit_weight_quant_state_from_module` recovers lost quantization state

### Step 4: Train with 8-bit Optimizer

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The 8-bit optimizer's `state_dict()` wraps quantization tensors in `__bnb_optimizer_quant_state__` for FSDP compatibility.

### Step 5: Save Checkpoint

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Save
save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, "checkpoint.pt")
```

### Step 6: Load Checkpoint

```python
# Load
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    state_dict = torch.load("checkpoint.pt")
    model.load_state_dict(state_dict)
```

The `load_state_dict` methods handle unwrapping the packed quantization state and the nested optimizer quant state.

### FSDP State Dict Flow

```
Saving:
  QuantState.as_dict(packed=True)
    -> {"absmax": tensor, "quant_map": tensor,
        "quant_state.bitsandbytes__nf4": uint8_tensor}
  -> FSDP gathers across ranks
  -> safetensors storage

Loading:
  safetensors storage
  -> FSDP scatters to ranks
  -> QuantState.from_dict(qs_dict, device)
    -> Unpacks "quant_state.bitsandbytes__nf4" tensor
    -> Reconstructs QuantState with proper dtypes, shapes, and nested state
  -> Params4bit.quant_state = quant_state
```

### Key Properties Ensuring FSDP Compatibility

| Component | Mechanism | Purpose |
|-----------|-----------|---------|
| `Params4bit` properties | `@property` descriptors | FSDP attribute traversal without graph breaks |
| `QuantState.as_dict(packed=True)` | JSON to uint8 tensor packing | Safetensors compatibility |
| `QuantState.__getattr__` | `bitsandbytes__*` name handling | FSDP dotted FQN resolution |
| `Linear4bit._save_to_state_dict` | CPU format unpacking + packed state | Correct serialization |
| `fix_4bit_weight_quant_state_from_module` | Module attribute fallback | Recovery after parameter conversion |
| `Optimizer8bit.state_dict` | `__bnb_optimizer_quant_state__` wrapping | Shape mismatch avoidance |
| `Optimizer8bit.load_state_dict` | Unwrapping nested dict | Correct deserialization |
| `Params4bit.__deepcopy__` | Proper state copying | FSDP parameter cloning |

### Important Notes

1. **Load to CPU first**: When using FSDP, load the model to CPU before wrapping with FSDP. Quantization happens when parameters move to GPU via FSDP.

2. **quant_state preservation**: The quantization state is preserved via `module.quant_state` attribute and recovered via `fix_4bit_weight_quant_state_from_module`.

3. **Optimizer state wrapping**: 8-bit optimizer states are automatically wrapped/unwrapped for FSDP compatibility.

4. **Sharding**: FSDP shards the packed uint8 weight tensor. Each rank gets a slice of the quantized data with its corresponding absmax values.
