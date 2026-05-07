# Reference 07: Optimizer Architecture

This document covers the 8-bit optimizer architecture in extreme detail, including the base class hierarchy, state management, paged memory, and configuration override system.

---

## Table of Contents

- [Class Hierarchy](#class-hierarchy)
- [GlobalOptimManager (Singleton)](#globaloptimmanager-singleton)
- [Optimizer8bit (Base Class)](#optimizer8bit-base-class)
- [Optimizer2State (Two-State Optimizers)](#optimizer2state-two-state-optimizers)
- [Optimizer1State (One-State Optimizers)](#optimizer1state-one-state-optimizers)
- [State Management](#state-management)
- [Paged Memory](#paged-memory)
- [FSDP Compatibility](#fsdp-compatibility)
- [min_8bit_size Threshold](#min_8bit_size-threshold)
- [Configuration Override System](#configuration-override-system)

---

## Class Hierarchy

```
torch.optim.Optimizer
  +-- Optimizer8bit
        +-- Optimizer2State
        |     +-- Adam, Adam8bit, Adam32bit
        |     +-- PagedAdam, PagedAdam8bit, PagedAdam32bit
        |     +-- AdamW, AdamW8bit, AdamW32bit
        |     +-- PagedAdamW, PagedAdamW8bit, PagedAdamW32bit
        |     +-- LAMB, LAMB8bit, LAMB32bit
        |     +-- AdEMAMix, AdEMAMix8bit, AdEMAMix32bit
        |     +-- PagedAdEMAMix, PagedAdEMAMix8bit, PagedAdEMAMix32bit
        |
        +-- Optimizer1State
              +-- SGD, SGD8bit, SGD32bit
              +-- Lion, Lion8bit, Lion32bit
              +-- PagedLion, PagedLion8bit, PagedLion32bit
              +-- LARS, LARS8bit, LARS32bit
              +-- Adagrad, Adagrad8bit, Adagrad32bit
              +-- RMSprop, RMSprop8bit, RMSprop32bit
```

---

## GlobalOptimManager (Singleton)

`bitsandbytes.optim.optimizer.GlobalOptimManager`

A global singleton that manages per-parameter optimizer configuration overrides. It maps parameter IDs to custom optimizer settings, enabling fine-grained control over quantization behavior.

### Construction

```python
# Do NOT call __init__ directly; use get_instance()
mng = GlobalOptimManager.get_instance()
```

The constructor raises `RuntimeError` to enforce singleton pattern. The first call to `get_instance()` creates the instance via `__new__` and calls `initialize()`.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `pid2config` | `dict[int, dict]` | Maps `id(parameter)` to a configuration dict. |
| `index2config` | `dict[tuple(int, int), dict]` | Maps `(group_index, param_index)` to a configuration dict. |
| `optimizer` | `None` | Placeholder for optimizer reference (currently unused). |
| `uses_config_override` | `bool` | `True` if any configuration override has been registered. |
| `module_weight_config_triple` | `list[tuple]` | List of `(module, param_name, config)` triples for module-level overrides. |

### Methods

#### `register_parameters(params) -> None`

Registers parameters so their indices can be mapped to configurations. Called before creating the optimizer.

```python
mng = GlobalOptimManager.get_instance()
mng.register_parameters(model.parameters())
```

**Behavior:**
1. Converts `params` to a list of param groups (wrapping in `{"params": ...}` if needed).
2. For each parameter, stores its `id()` in `pid2config` if previously configured.
3. Maps `(group_index, p_index)` to the config in `index2config`.

#### `override_config(parameters, key=None, value=None, key_value_dict=None) -> None`

Overrides optimizer configuration for specific parameters.

```python
# Single key-value override
mng.override_config(model.layer.weight, "optim_bits", 32)

# Multiple overrides via dict
mng.override_config(
    [model.layer.weight, model.layer.bias],
    key_value_dict={"optim_bits": 32, "lr": 1e-4}
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `parameters` | `Parameter`, `Tensor`, or `list` | The parameters to configure. |
| `key` | `Optional[str]` | Single config key to override. |
| `value` | | Single config value. |
| `key_value_dict` | `Optional[dict]` | Dictionary of multiple key-value overrides. |

**Behavior:**
- Sets `uses_config_override = True`.
- If the parameter's `id()` is already in `pid2config`, updates the existing config.
- Otherwise, creates a new config entry.

#### `register_module_override(module, param_name, config) -> None`

Registers a module-level override. Called by `StableEmbedding` and `Embedding` to ensure 32-bit optimizer states.

```python
# Called internally by StableEmbedding:
GlobalOptimManager.get_instance().register_module_override(
    self, "weight", {"optim_bits": 32}
)
```

Stores the `(module, param_name, config)` triple in `module_weight_config_triple`. The override is resolved in `Optimizer8bit.check_overrides()` by matching the module's parameter against the optimizer's parameter groups.

---

## Optimizer8bit (Base Class)

`bitsandbytes.optim.optimizer.Optimizer8bit`

Abstract base class for all 8-bit optimizers. Inherits from `torch.optim.Optimizer`.

### Constructor

```python
Optimizer8bit(
    params,
    defaults,
    optim_bits=32,
    is_paged=False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `params` | `iterable` | required | Model parameters or parameter group dicts. |
| `defaults` | `dict` | required | Default optimizer settings. |
| `optim_bits` | `int` | `32` | Number of bits for optimizer state (32 or 8). |
| `is_paged` | `bool` | `False` | Whether to use paged (unified memory) allocation. |

### Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `initialized` | `bool` | Whether `step()` has been called at least once. |
| `name2qmap` | `dict[str, Tensor]` | Quantization maps: `"dynamic"` (signed) and `"udynamic"` (unsigned). |
| `is_paged` | `bool` | Whether paged memory is enabled. |
| `page_mng` | `GlobalPageManager` | Singleton paged tensor manager. |
| `mng` | `GlobalOptimManager` | Singleton optimizer configuration manager. |
| `non_castable_tensor_keys` | `set[str]` | State keys that should not be dtype-cast during `load_state_dict`. Used for FSDP compatibility. |

### `non_castable_tensor_keys` Set

```python
{
    "qmap1", "qmap2",           # Quantization maps (float32, must not be cast)
    "max1", "max2",             # Legacy: max values
    "new_max1", "new_max2",     # Legacy: new max values
    "state1", "state2",         # Optimizer states (uint8 for 8-bit, must not be cast)
    "gnorm_vec",                # Gradient norm vector
    "absmax1", "absmax2",       # Block-wise absmax values (float32)
    "unorm_vec",                # Update norm vector
}
```

### Methods

#### `fill_qmap() -> None`

Populates `name2qmap` with dynamic quantization maps:

```python
self.name2qmap["dynamic"] = F.create_dynamic_map(signed=True)    # Signed: used for state1
self.name2qmap["udynamic"] = F.create_dynamic_map(signed=False)  # Unsigned: used for state2
```

Called automatically when `optim_bits == 8`.

#### `step(closure=None) -> Optional[Any]`

The main optimization step. Orchestrates the full update pipeline:

1. **closure**: If provided, evaluates it with `torch.enable_grad()` to get the loss.
2. **First-step initialization**: If not `initialized`, calls `check_overrides()` and `to_gpu()`, sets `initialized = True`.
3. **Parameter iteration**: For each parameter group and parameter:
   - Skips parameters without gradients.
   - Initializes state on first encounter: `init_state(group, p, gindex, pindex)`.
   - Prefetches paged state: `prefetch_state(p)`.
   - Performs the update: `update_step(group, p, gindex, pindex)`.
   - Synchronizes GPU: `sync_gpu(p)`.
4. **Final sync**: If `is_paged`, performs a final `sync_gpu()` to ensure all async paged operations complete.

#### `check_overrides() -> None`

Resolves module-level overrides registered via `register_module_override()`. For each `(module, attr, config)` triple:

1. Gets the module's parameter via `getattr(module, attr)`.
2. Searches all parameter groups for a matching parameter (by `id()`).
3. When found, stores the config in both `mng.pid2config` and `mng.index2config`.

#### `to_gpu() -> None`

Moves all optimizer state tensors to the same device as their corresponding parameters. Skips CPU parameters and paged tensors.

#### `get_config(gindex, pindex, group) -> dict`

Merges default configuration with any per-parameter overrides:

```python
config = {
    "betas": group["betas"],
    "eps": group["eps"],
    "weight_decay": group["weight_decay"],
    "lr": group["lr"],
    "alpha": group.get("alpha", 0.0),
    "t_alpha": group.get("t_alpha", None),
    "t_beta3": group.get("t_beta3", None),
    "optim_bits": self.args.optim_bits,
    "min_8bit_size": self.args.min_8bit_size,
    "max_unorm": self.args.max_unorm,
    "skip_zeros": self.args.skip_zeros,
}
```

Override resolution order:
1. Start with defaults from `self.args` and the param group.
2. Check `mng.index2config[(gindex, pindex)]` (from `register_parameters`).
3. Check `mng.pid2config[id(p)]` (from `override_config`).

#### `get_state_buffer(p, dtype=torch.float32) -> Tensor`

Allocates a zero-initialized state buffer for a parameter.

```python
def get_state_buffer(self, p, dtype=torch.float32):
    if p.device.type == "cpu":
        if self.is_paged:
            warnings.warn("Paged optimizers are not supported on CPU. Falling back...")
        return torch.zeros_like(p, dtype=dtype, device=p.device)
    if not self.is_paged or p.numel() < 1e5:
        return torch.zeros_like(p, dtype=dtype, device=p.device)
    else:
        # Paged allocation for large tensors (> ~400KB for float32)
        buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
        F.fill(buff, 0)
        self.page_mng.paged_tensors.append(buff)
        return buff
```

**Decision logic:**
- CPU parameters: Always regular allocation (paged not supported on CPU).
- GPU parameters with `is_paged=False`: Regular allocation.
- GPU parameters with `is_paged=True` and `p.numel() < 100,000`: Regular allocation (too small to benefit from paging).
- GPU parameters with `is_paged=True` and `p.numel() >= 100,000`: Paged allocation via `get_paged()`.

#### `prefetch_state(p) -> None`

For paged optimizers, prefetches state tensors from CPU to GPU before the update step:

```python
def prefetch_state(self, p):
    if self.is_paged:
        state = self.state[p]
        s1 = state["state1"]
        is_paged = getattr(s1, "is_paged", False)
        if is_paged:
            F.prefetch_tensor(state["state1"])
            if "state2" in state:
                F.prefetch_tensor(state["state2"])
```

#### `state_dict() -> dict`

Returns the optimizer state dict with FSDP compatibility wrapping:

1. Calls `super().state_dict()`.
2. Deep copies the state to avoid modifying the original.
3. Extracts all keys in `non_castable_tensor_keys` from each parameter's state.
4. Wraps extracted tensors under a single `__bnb_optimizer_quant_state__` key per parameter state.

This wrapping prevents FSDP from trying to gather quantization tensors (which have different shapes than model parameters).

#### `load_state_dict(state_dict, move_to_device=True) -> None`

Loads an optimizer state dict with FSDP unwrapping:

1. Deep copies the state dict.
2. Unwraps `__bnb_optimizer_quant_state__` entries back into the state dict.
3. Validates parameter group counts and sizes.
4. Creates an `id_map` mapping old parameter IDs to new ones.
5. Casts tensors to appropriate types (skipping `non_castable_tensor_keys` from casting, but still moving to device if `move_to_device=True`).

---

## Optimizer2State (Two-State Optimizers)

`bitsandbytes.optim.optimizer.Optimizer2State`

Base class for optimizers with two state tensors (e.g., Adam with first and second moments). Inherits from `Optimizer8bit`.

### Constructor

```python
Optimizer2State(
    optimizer_name: str,
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    optim_bits=32,
    args=None,
    min_8bit_size=4096,
    max_unorm=0.0,
    skip_zeros=False,
    is_paged=False,
    alpha=0.0,
    t_alpha: Optional[int] = None,
    t_beta3: Optional[int] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer_name` | `str` | required | Internal optimizer identifier: `"adam"`, `"lamb"`, `"ademamix"`. |
| `params` | `iterable` | required | Parameters to optimize. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `betas` | `tuple` | `(0.9, 0.999)` | Beta values for moment decay. |
| `eps` | `float` | `1e-8` | Epsilon for numerical stability. |
| `weight_decay` | `float` | `0.0` | Weight decay coefficient. |
| `optim_bits` | `int` | `32` | State precision: 32 (float32) or 8 (uint8 blockwise). |
| `args` | `object` | `None` | Override object with `optim_bits`, `min_8bit_size`, `max_unorm`, `skip_zeros` attributes. |
| `min_8bit_size` | `int` | `4096` | Minimum parameter elements for 8-bit quantization. |
| `max_unorm` | `float` | `0.0` | Maximum update norm relative to parameter norm (for LAMB). |
| `skip_zeros` | `bool` | `False` | Skip zero-valued gradients. |
| `is_paged` | `bool` | `False` | Enable paged memory. |
| `alpha` | `float` | `0.0` | Alpha scaling (for AdEMAMix). |
| `t_alpha` | `Optional[int]` | `None` | Iteration count for alpha scheduling (AdEMAMix). |
| `t_beta3` | `Optional[int]` | `None` | Iteration count for beta3 scheduling (AdEMAMix). |

### `init_state(group, p, gindex, pindex) -> None`

Initializes the optimizer state for a parameter. Called once per parameter on the first `step()`.

**Decision tree for state dtype:**
1. Get merged config via `get_config()`.
2. If `config["optim_bits"] == 32`: dtype = `torch.float32`.
3. If `config["optim_bits"] == 8`: dtype = `torch.uint8`.
4. If `p.numel() < config["min_8bit_size"]`: force dtype = `torch.float32` regardless.

**32-bit state initialization:**
```python
state["step"] = 0
state["state1"] = get_state_buffer(p, dtype=torch.float32)  # First moment
state["state2"] = get_state_buffer(p, dtype=torch.float32)  # Second moment
if config["max_unorm"] > 0.0:
    state["unorm_vec"] = torch.zeros((1,), device=p.device)
```

**8-bit state initialization:**
```python
state["step"] = 0
state["state1"] = get_state_buffer(p, dtype=torch.uint8)
state["qmap1"] = name2qmap["dynamic"].to(p.device)  # Signed dynamic map
state["state2"] = get_state_buffer(p, dtype=torch.uint8)
state["qmap2"] = name2qmap["udynamic"].to(p.device)  # Unsigned dynamic map

# Blockwise absmax (blocksize=256)
blocksize = 256
n = p.numel()
blocks = (n // blocksize) + bool(n % blocksize)
state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)
state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

if config["max_unorm"] > 0.0:
    state["unorm_vec"] = torch.zeros((1,), device=p.device)
```

### `update_step(group, p, gindex, pindex) -> None`

Performs a single parameter update:

1. Ensures contiguous memory: `p.data = p.data.contiguous()` and `p.grad = p.grad.contiguous()`.
2. Increments `state["step"]`.
3. Gets merged config.
4. Dispatches based on state dtype:

**32-bit path:** Calls `F.optimizer_update_32bit()` with the optimizer name, gradients, parameters, states, and hyperparameters.

**8-bit path:** Calls `F.optimizer_update_8bit_blockwise()` with quantized states, quantization maps, and absmax values.

---

## Optimizer1State (One-State Optimizers)

`bitsandbytes.optim.optimizer.Optimizer1State`

Base class for optimizers with a single state tensor (e.g., SGD with momentum). Inherits from `Optimizer8bit`.

### Constructor

```python
Optimizer1State(
    optimizer_name: str,
    params,
    lr=1e-3,
    betas=(0.9, 0.0),
    eps=1e-8,
    weight_decay=0.0,
    optim_bits=32,
    args=None,
    min_8bit_size=4096,
    max_unorm=0.0,
    skip_zeros=False,
    is_paged=False,
)
```

Similar to `Optimizer2State` but simpler:
- Only `betas` (not `alpha`, `t_alpha`, `t_beta3`).
- Default `betas = (0.9, 0.0)` instead of `(0.9, 0.999)`.
- Uses optimizer names: `"momentum"`, `"lion"`, `"rmsprop"`, `"adagrad"`, `"lars"`.

### `init_state(group, p, gindex, pindex) -> None`

Same decision tree as `Optimizer2State`, but allocates only one state tensor:

**32-bit state:**
```python
state["step"] = 0
state["state1"] = get_state_buffer(p, dtype=torch.float32)
if config["max_unorm"] > 0.0:
    state["unorm_vec"] = torch.zeros((1,), device=p.device)
```

**8-bit state:**
```python
state["step"] = 0
state["state1"] = get_state_buffer(p, dtype=torch.uint8)
state["qmap1"] = name2qmap["dynamic"].to(p.device)

blocksize = 256
n = p.numel()
blocks = (n // blocksize) + bool(n % blocksize)
state["absmax1"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

if config["max_unorm"] > 0.0:
    state["unorm_vec"] = torch.zeros((1,), device=p.device)
```

### `update_step(group, p, gindex, pindex) -> None`

Same structure as `Optimizer2State.update_step()`, but passes `None` for `state2`, `qmap2`, `absmax2` in the 8-bit path, and `beta2=0.0`, `beta3=0.0`, `alpha=0.0` in the 32-bit path.

---

## State Management

### State Dictionary Keys

Each parameter's state dict contains:

**32-bit optimizers:**
| Key | Type | Description |
|---|---|---|
| `step` | `int` | Current step count. |
| `state1` | `Tensor` (float32) | First state (e.g., first moment m). |
| `state2` | `Tensor` (float32) | Second state (e.g., second moment v). Only for 2-state optimizers. |
| `unorm_vec` | `Tensor` | Update norm (only if `max_unorm > 0.0`). |

**8-bit optimizers:**
| Key | Type | Description |
|---|---|---|
| `step` | `int` | Current step count. |
| `state1` | `Tensor` (uint8) | Quantized first state. |
| `qmap1` | `Tensor` (float32) | Quantization map for state1 (signed dynamic). |
| `absmax1` | `Tensor` (float32) | Per-block absmax for state1. |
| `state2` | `Tensor` (uint8) | Quantized second state (2-state only). |
| `qmap2` | `Tensor` (float32) | Quantization map for state2 (unsigned dynamic). |
| `absmax2` | `Tensor` (float32) | Per-block absmax for state2 (2-state only). |
| `unorm_vec` | `Tensor` | Update norm (only if `max_unorm > 0.0`). |

### Block Size for 8-bit States

8-bit optimizer states use a fixed block size of **256 elements**. This means:
- Each block of 256 consecutive state values is independently quantized.
- The absmax tensor has `ceil(numel / 256)` entries.

---

## Paged Memory

### When Paging is Activated

Paging uses CUDA unified memory (managed memory) to allow optimizer states to be automatically migrated between GPU and CPU memory. It is activated when:

1. `is_paged=True` is set on the optimizer.
2. The parameter has `p.numel() >= 100,000` elements.
3. The parameter is on a GPU device.

For parameters with fewer than 100,000 elements or on CPU, regular allocation is used even when `is_paged=True`.

### Paged Memory Flow

```
Step N-1                    Step N                      Step N (continued)
┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐
│ State on CPU     │       │ prefetch_state() │        │ update_step()    │
│ (paged out)      │  ---> │ → prefetch to GPU│  --->  │ → compute on GPU │
│                  │       │                  │        │ → result on GPU  │
└──────────────────┘       └──────────────────┘        └──────────────────┘
                                                              │
                                                              v
                                                       ┌──────────────────┐
                                                       │ sync_gpu()       │
                                                       │ → ensure async   │
                                                       │   ops complete   │
                                                       └──────────────────┘
```

### GlobalPageManager

The `GlobalPageManager` singleton tracks all paged tensors:

```python
page_mng = GlobalPageManager.get_instance()
page_mng.paged_tensors  # List of all paged state tensors
```

When a new paged tensor is allocated via `get_state_buffer()`, it is appended to `paged_tensors`.

### Prefetching

Before each parameter update, `prefetch_state(p)` is called:

```python
def prefetch_state(self, p):
    if self.is_paged:
        state = self.state[p]
        s1 = state["state1"]
        if getattr(s1, "is_paged", False):
            F.prefetch_tensor(state["state1"])  # Move to GPU
            if "state2" in state:
                F.prefetch_tensor(state["state2"])  # Move to GPU
```

This ensures the state is on the GPU before the update kernel runs. CUDA's unified memory system handles the actual data migration.

### Synchronization

After each parameter update, `sync_gpu(p)` is called to ensure all async operations complete:

```python
# In step():
self.update_step(group, p, gindex, pindex)
sync_gpu(p)  # Per-parameter sync

# After all parameters:
if self.is_paged and p is not None:
    sync_gpu(p)  # Final sync for all async paged operations
```

### Memory Savings

For a model with parameters of total size P:
- **32-bit, non-paged**: State memory = 2P (state1 + state2, both float32 on GPU).
- **8-bit, non-paged**: State memory = P/4 + P/4 + small overhead (uint8 states on GPU).
- **8-bit, paged**: GPU memory for states is near-zero (states live in unified memory, paged to CPU when not in use).

---

## FSDP Compatibility

### State Dict Wrapping

FSDP's `full_optim_state_dict` gathers all tensor states across ranks. Quantization states have different shapes than model parameters, which would cause gather operations to fail. To handle this:

**Saving (`state_dict()`):**
```python
# Before wrapping:
state = {
    "step": 42,
    "state1": uint8_tensor,    # Different shape from parameter
    "qmap1": float32_tensor,
    "absmax1": float32_tensor,
    "state2": uint8_tensor,
    "qmap2": float32_tensor,
    "absmax2": float32_tensor,
}

# After wrapping:
state = {
    "step": 42,
    "__bnb_optimizer_quant_state__": {
        "state1": uint8_tensor,
        "qmap1": float32_tensor,
        "absmax1": float32_tensor,
        "state2": uint8_tensor,
        "qmap2": float32_tensor,
        "absmax2": float32_tensor,
    }
}
```

**Loading (`load_state_dict()`):**
Reverses the wrapping by popping `__bnb_optimizer_quant_state__` and merging back into the state dict.

### Non-Castable Tensors

During `load_state_dict()`, tensors in `non_castable_tensor_keys` are:
- Moved to the parameter's device (if `move_to_device=True`).
- NOT cast to the parameter's dtype (they must remain uint8 or float32 as appropriate).

---

## min_8bit_size Threshold

The `min_8bit_size` parameter (default: 4096) controls the minimum parameter size for 8-bit quantization of optimizer states.

### Behavior

```python
# In init_state():
if p.numel() < config["min_8bit_size"]:
    dtype = torch.float32  # Force 32-bit regardless of optim_bits
```

### Rationale

Small parameters (bias vectors, small embedding layers, layer norms) have few enough elements that:
1. The overhead of quantization maps and absmax arrays can exceed the memory savings.
2. Quantization noise is proportionally larger for small tensors.
3. The computational overhead of quantize/dequantize is not amortized.

### Default Value

The default threshold of **4096 elements** was chosen as a balance:
- A 4096-element float32 tensor occupies 16 KB.
- An 8-bit version saves only ~12 KB (plus overhead for absmax and qmap).
- Below this threshold, the savings are negligible and may hurt training quality.

### Per-Parameter Override

Individual parameters can have different thresholds via `GlobalOptimManager`:

```python
mng = GlobalOptimManager.get_instance()
mng.override_config(model.small_layer.weight, "min_8bit_size", 0)  # Allow 8-bit for small params
```

---

## Configuration Override System

### Override Resolution

When computing the configuration for a parameter update, the following precedence applies:

1. **Base defaults**: From `self.args` (set in constructor).
2. **Param group defaults**: From the `group` dict in `param_groups`.
3. **Index-based override**: From `mng.index2config[(gindex, pindex)]` (set by `register_parameters` + `override_config`).
4. **ID-based override**: From `mng.pid2config[id(p)]` (set by `override_config`).

Later entries override earlier ones.

### Override Lifecycle

```python
# 1. Register parameters BEFORE creating optimizer
mng = GlobalOptimManager.get_instance()
mng.register_parameters(model.parameters())

# 2. Create optimizer
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)

# 3. Override specific parameters
mng.override_config(model.embed.weight, "optim_bits", 32)
mng.override_config(model.embed.weight, "min_8bit_size", 0)

# 4. Module-level overrides are resolved on first step()
# (StableEmbedding and Embedding auto-register via register_module_override)
```

### Overrideable Configuration Keys

| Key | Type | Description |
|---|---|---|
| `optim_bits` | `int` | State precision (8 or 32). |
| `min_8bit_size` | `int` | Minimum elements for 8-bit. |
| `max_unorm` | `float` | Maximum update norm. |
| `skip_zeros` | `bool` | Skip zero gradients. |
| `lr` | `float` | Learning rate. |
| `betas` | `tuple` | Beta values. |
| `eps` | `float` | Epsilon. |
| `weight_decay` | `float` | Weight decay. |
| `alpha` | `float` | Alpha (AdEMAMix). |
| `t_alpha` | `int` | Alpha scheduling iterations (AdEMAMix). |
| `t_beta3` | `int` | Beta3 scheduling iterations (AdEMAMix). |
