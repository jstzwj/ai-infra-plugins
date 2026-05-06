# LoRA Adapters Reference

This document provides a comprehensive reference for the LoRA (Low-Rank Adaptation) implementation in vLLM, covering the architecture, request lifecycle, layer types, multi-LoRA serving, weight management, and configuration.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Configuration: LoRAConfig](#2-configuration-loraconfig)
3. [LoRA Request Lifecycle](#3-lora-request-lifecycle)
4. [LoRARequest](#4-lorarequest)
5. [PEFTHelper](#5-pefthelper)
6. [LoRALayerWeights](#6-loralayerweights)
7. [PackedLoRALayerWeights](#7-packedloralayerweights)
8. [LoRAModel](#8-loramodel)
9. [LoRAModelManager](#9-loramodelmanager)
10. [WorkerLoRAManager](#10-workerlormanager)
11. [LoRA Layer Types](#11-lora-layer-types)
12. [Base Classes and Interfaces](#12-base-classes-and-interfaces)
13. [Punica Kernels for Multi-LoRA Serving](#13-punica-kernels-for-multi-lora-serving)
14. [LoRA Mapping and Utilities](#14-lora-mapping-and-utilities)
15. [LoRA Resolver](#15-lora-resolver)
16. [V1 LoRA Support (LoRAModelRunnerMixin)](#16-v1-lora-support-loramodelrunnermixin)
17. [Tensor Parallelism for LoRA](#17-tensor-parallelism-for-lora)
18. [Cache and Memory Management](#18-cache-and-memory-management)
19. [Multimodal LoRA Support](#19-multimodal-lora-support)
20. [Utility Functions](#20-utility-functions)
21. [Prompt Adapters](#21-prompt-adapters)
22. [Full Parameter Reference](#22-full-parameter-reference)

---

## 1. Architecture Overview

vLLM's LoRA system implements multi-LoRA serving, allowing multiple LoRA adapters to be active simultaneously in a single inference engine. The architecture follows a hierarchical model:

```
Engine
  --> Executor
       --> Worker(s)
            --> WorkerLoRAManager (LRU cache-based adapter management)
                 --> LoRAModelManager (manages multiple LoRAModel instances)
                      --> LoRAModel (represents a single LoRA adapter)
                           --> LoRALayerWeights / PackedLoRALayerWeights
```

Key design principles:
- **LRU eviction**: Adapters are evicted when the cache exceeds `max_loras`
- **Punica kernels**: Triton-based kernels for efficient multi-LoRA batched inference
- **Tensor parallelism**: Support for fully-sharded and non-sharded LoRA across TP ranks
- **Layer swapping**: Original model layers are replaced with LoRA-aware versions at load time

### Source Files

| File | Purpose |
|------|---------|
| `vllm/lora/request.py` | LoRARequest data structure |
| `vllm/lora/lora_model.py` | LoRAModel class |
| `vllm/lora/lora_weights.py` | LoRALayerWeights and PackedLoRALayerWeights |
| `vllm/lora/model_manager.py` | LoRAModelManager, AdapterLRUCache, factory functions |
| `vllm/lora/worker_manager.py` | WorkerLoRAManager and LRUCacheWorkerLoRAManager |
| `vllm/lora/layers/` | LoRA layer implementations |
| `vllm/lora/punica_wrapper/` | Punica kernel wrappers |
| `vllm/lora/utils.py` | Utility functions |
| `vllm/lora/resolver.py` | LoRAResolver ABC and registry |
| `vllm/lora/peft_helper.py` | PEFTHelper for PEFT config validation |
| `vllm/config/lora.py` | LoRAConfig |
| `vllm/v1/worker/lora_model_runner_mixin.py` | V1 model runner LoRA mixin |

---

## 2. Configuration: LoRAConfig

**File**: `vllm/config/lora.py`

### Class: `LoRAConfig`

```python
class LoRAConfig:
    max_lora_rank: int              # Maximum LoRA rank across all adapters
    max_loras: int                  # Maximum number of LoRAs to keep in memory
    fully_sharded_loras: bool       # Whether to fully shard LoRA across TP ranks
    max_cpu_loras: int | None       # Maximum LoRAs to store on CPU (None = unlimited)
    lora_dtype: str | None          # Data type for LoRA weights (None = auto)
    lora_extra_vocab_size: int      # Extra vocabulary size for LoRA
    # (deprecated: use lora_extra_vocab_size instead)
    lora_vocab_size: int            # Vocabulary size passed to LoRA layers
    long_lora_max_len: int | None   # Max context length for long LoRA scaling
    target_modules: list[str] | None # Modules to apply LoRA to (None = auto-detect)
    default_mm_loras: dict[str, str] | None  # Default multimodal LoRAs per modality
    enable_tower_connector_lora: bool  # Enable LoRA for vision tower connector
    specialize_active_lora: bool       # Specialize compilation per active LoRA
```

### Key Parameters

- **`max_lora_rank`**: Determines the maximum rank (r) of any LoRA adapter. This affects pre-allocated GPU memory for LoRA weights. Default: `16`.
- **`max_loras`**: Maximum number of simultaneously cached LoRA adapters. When exceeded, LRU eviction is triggered. Default: `4`.
- **`fully_sharded_loras`**: When `True`, LoRA weights are fully sharded across tensor parallel ranks. When `False`, each rank holds a complete copy of non-parallel LoRA weights. Default: `False`.
- **`max_cpu_loras`**: If set, limits the number of LoRA adapters cached on CPU memory. `None` means no limit.
- **`lora_dtype`**: Override the data type for LoRA computations. When `None`, the model's data type is used.
- **`target_modules`**: Explicit list of module name patterns to apply LoRA to. When `None`, the system auto-detects based on the LoRA checkpoint's target modules.
- **`default_mm_loras`**: Mapping from modality name to LoRA path for default multimodal LoRA adapters. Example: `{"IMAGE": "/path/to/vision_lora"}`.
- **`enable_tower_connector_lora`**: Enables LoRA support on the vision tower connector layers in multimodal models.
- **`specialize_active_lora`**: When `True`, CUDA graphs are captured separately for each LoRA configuration, enabling better optimization.

### Construction

```python
LoRAConfig(
    max_lora_rank=16,
    max_loras=4,
    fully_sharded_loras=False,
    max_cpu_loras=None,
    lora_dtype=None,
    target_modules=None,
    default_mm_loras=None,
    enable_tower_connector_lora=False,
    specialize_active_lora=False,
)
```

---

## 3. LoRA Request Lifecycle

The lifecycle of a LoRA request follows these stages:

1. **Request Creation**: Client creates a `LoRARequest` with name, integer ID, and path.
2. **Submission**: Request is passed to the engine via `add_lora()`.
3. **Worker Processing**: Each worker's `WorkerLoRAManager.add_adapter()` loads the LoRA.
4. **Loading**: `LoRAModelManager` loads the LoRA weights from disk (safetensors/bin/tensorizer).
5. **Model Integration**: LoRA weights are applied to model layers via `set_lora()` calls.
6. **Activation**: During forward pass, active LoRAs are set via `set_active_adapters()`.
7. **Inference**: Punica kernels perform batched multi-LoRA inference.
8. **Eviction**: When `max_loras` is exceeded, least-recently-used adapters are evicted.
9. **Removal**: Explicit removal via `remove_lora()` or `remove_all_adapters()`.

### Flow Diagram

```
Client --> Engine.add_lora(LoRARequest)
    --> Executor.collective_rpc("add_lora", LoRARequest)
        --> Worker.add_lora(LoRARequest)
            --> LRUCacheWorkerLoRAManager.add_adapter(LoRARequest)
                --> LoRAModelManager.add_adapter(LoRARequest)
                    --> LoRAModel.from_local_checkpoint(path)
                    --> Load weights into LoRALayerWeights
                    --> Set weights on model layers via set_lora()
```

---

## 4. LoRARequest

**File**: `vllm/lora/request.py`

### Class: `LoRARequest`

```python
class LoRARequest(msgspec.Struct, frozen=False):
    lora_name: str                          # Human-readable name
    lora_int_id: int                        # Integer ID (1-indexed, 0 = no LoRA)
    lora_path: str                          # Path to LoRA weights on disk
    base_model_name: str | None = None      # Base model name override
    tensorizer_config_dict: dict | None = None  # Tensorizer serialization config
    load_inplace: bool = False              # Whether to load weights in-place
```

### Key Behaviors

- **Equality and Hashing**: Two `LoRARequest` objects are equal if their `lora_name` matches. This means requests with the same name but different paths are considered identical.
- **Frozen**: The struct is marked `frozen=False` to allow mutation after creation (e.g., updating `lora_path`).
- **ID Convention**: `lora_int_id` uses 1-indexed IDs. ID `0` is reserved for "no LoRA" (base model).

### Usage

```python
request = LoRARequest(
    lora_name="my_adapter",
    lora_int_id=1,
    lora_path="/path/to/lora_weights",
)
```

---

## 5. PEFTHelper

**File**: `vllm/lora/peft_helper.py`

### Class: `PEFTHelper`

A dataclass that parses and validates PEFT/LoRA configuration files.

```python
@dataclass(repr=False)
class PEFTHelper:
    name: str | None
    config: dict
    # Derived from config:
    target_modules: set[str]
    module_filter: str | None
    target_alibi_modules: set[str]
    boft_target_modules: set[str]
    ia3_target_modules: set[str]
    # LoRA parameters:
    task_type: str
    r: int                    # LoRA rank
    lora_alpha: int           # LoRA scaling factor
    lora_dropout: float       # Dropout probability (must be 0 for vLLM)
    bias: str                 # Bias handling strategy
    # Scaling:
    scaling: float            # Computed as lora_alpha / r
    use_rslora: bool          # Whether rank-stabilized LoRA is used
```

### Key Methods

- **`from_config(lora_name, config_dict, max_lora_rank, expected_lora_alpha)`**: Factory method that creates a `PEFTHelper` from a PEFT config dictionary. Validates that rank does not exceed `max_lora_rank`.
- **`scaling`**: Property that computes the effective scaling factor. Uses `lora_alpha / sqrt(r)` for RSLoRA, `lora_alpha / r` otherwise.

### Validation Rules

- `lora_dropout` must be `0` (vLLM does not support dropout during inference)
- `bias` must be `"none"` (bias is not supported in LoRA layers)
- `r` must not exceed `max_lora_rank`
- `lora_alpha` must match the expected value (when specified)

---

## 6. LoRALayerWeights

**File**: `vllm/lora/lora_weights.py`

### Class: `LoRALayerWeights`

Represents the LoRA weights for a single model layer.

```python
@dataclass
class LoRALayerWeights:
    module_name: str                 # Original module name (e.g., "q_proj")
    rank: int                        # LoRA rank (r)
    lora_alpha: int                  # LoRA alpha for scaling
    lora_a: torch.Tensor             # A matrix, shape [r, in_features]
    lora_b: torch.Tensor             # B matrix, shape [out_features, r]
    scaling: float                   # Effective scaling = lora_alpha / rank
    embeddings: torch.Tensor | None  # For embedding layers
    output_tensor: torch.Tensor      # Cached A^T @ B^T result
```

### Key Properties and Methods

- **`lora_a`**: The "down-projection" weight matrix. Shape: `[rank, input_dim]`.
- **`lora_b`**: The "up-projection" weight matrix. Shape: `[output_dim, rank]`.
- **`scaling`**: Precomputed scaling factor: `lora_alpha / rank` (or `lora_alpha / sqrt(rank)` for RSLoRA).
- **`obj()`**: Returns a memory-efficient representation for caching.

### LoRA Computation

For a base weight matrix `W` and input `x`, the LoRA output is:
```
output = x @ W^T + scaling * x @ lora_a^T @ lora_b^T
```

This is equivalent to:
```
output = x @ (W + scaling * lora_a^T @ lora_b^T)^T
```

---

## 7. PackedLoRALayerWeights

**File**: `vllm/lora/lora_weights.py`

### Class: `PackedLoRALayerWeights`

Represents fused/packed LoRA weights for merged column parallel layers or MoE (Mixture of Experts) layers.

```python
@dataclass
class PackedLoRALayerWeights:
    module_name: str                    # Module name
    lora_alphas: list[int]              # Alpha per sub-layer
    lora_a: torch.Tensor                # Packed A matrices
    lora_b: torch.Tensor                # Packed B matrices
    scaling: list[float]                # Scaling per sub-layer
    sub_modules: list[str]              # Names of constituent sub-modules
    shapes: list[tuple[int, ...]]       # Shapes of constituent weight tensors
```

### Class Methods

#### `pack(weights_list, repack_qkv, qkv_proj_repacking_index)`

Packs multiple `LoRALayerWeights` into a single `PackedLoRALayerWeights`.

```python
@classmethod
def pack(
    cls,
    weights_list: list[LoRALayerWeights],
    repack_qkv: bool = False,
    qkv_proj_repacking_index: list[int] | None = None,
) -> "PackedLoRALayerWeights"
```

**Parameters:**
- `weights_list`: List of individual LoRA weights to pack.
- `repack_qkv`: Whether to repack for QKV parallel projection (reorders dimensions).
- `qkv_proj_repacking_index`: Index mapping for QKV repacking.

**Behavior**: Concatenates `lora_a` and `lora_b` matrices along the feature dimension to create a single fused weight.

#### `pack_moe(lora_a_list, lora_b_list, module_name, lora_alphas, sub_modules, shapes)`

Packs LoRA weights for MoE (Mixture of Experts) layers.

```python
@classmethod
def pack_moe(
    cls,
    lora_a_list: list[torch.Tensor],
    lora_b_list: list[torch.Tensor],
    module_name: str,
    lora_alphas: list[int],
    sub_modules: list[str],
    shapes: list[tuple[int, ...]],
) -> "PackedLoRALayerWeights"
```

---

## 8. LoRAModel

**File**: `vllm/lora/lora_model.py`

### Class: `LoRAModel`

Represents a single LoRA adapter with all its layer weights.

```python
class LoRAModel:
    model_name: str                              # Name of the LoRA adapter
    id: int                                      # Integer ID
    rank: int                                    # LoRA rank
    lora_alpha: int                              # LoRA alpha
    scaling: float                               # Effective scaling
    target_modules: dict[str, LoRALayerWeights | PackedLoRALayerWeights]  # Layer weights
    lora_a_weights: dict[str, torch.Tensor]      # A weights per layer
    lora_b_weights: dict[str, torch.Tensor]      # B weights per layer
```

### Key Methods

#### `from_lora_tensors(tensor_name_mapping, lora_model_id, lora_model_name, device, dtype, embeddings, target_modules, column_parallel_layers, row_parallel_layers, base_model, mixin, qkv_proj_repacking, packed_modules_mapping, lora_alpha, scaling)`

Creates a `LoRAModel` from pre-loaded LoRA tensors.

**Parameters:**
- `tensor_name_mapping`: Mapping from weight names to tensors.
- `lora_model_id`: Integer ID for this LoRA adapter.
- `lora_model_name`: Human-readable name.
- `device`: Target device for weights.
- `dtype`: Data type for weights.
- `embeddings`: Embedding weights (for embedding LoRA).
- `target_modules`: Set of module names to apply LoRA to.
- `column_parallel_layers`: List of column parallel layer names.
- `row_parallel_layers`: List of row parallel layer names.
- `base_model`: Base model module for shape inference.
- `mixin`: LoRAModelManager mixin for type resolution.
- `qkv_proj_repacking`: Whether to repack QKV projections.
- `packed_modules_mapping`: Mapping for packed modules (e.g., gate_up_proj).
- `lora_alpha`: LoRA alpha value.
- `scaling`: Effective scaling factor.

#### `from_local_checkpoint(lora_dir, expected_lora_rank, max_lora_rank, lora_model_id, device, dtype, target_modules, column_parallel_layers, row_parallel_layers, base_model, mixin, qkv_proj_repacking, packed_modules_mapping, lora_alpha, scaling)`

Creates a `LoRAModel` from a local checkpoint directory.

**Supported formats:**
- Safetensors (`.safetensors`)
- PyTorch binary (`.bin`)
- PyTorch pickle (`.pt`)
- Tensorizer (vLLM serialization format)

The method first looks for `adapter_config.json` (PEFT format) to determine LoRA parameters, then loads the weight files.

#### `clone(base_model=None, device=None, pin_memory=False)`

Creates a deep copy of the LoRAModel, optionally on a different device.

```python
def clone(
    self,
    base_model: nn.Module | None = None,
    device: torch.device | None = None,
    pin_memory: bool = False,
) -> "LoRAModel"
```

---

## 9. LoRAModelManager

**File**: `vllm/lora/model_manager.py`

### Class: `LoRAModelManager`

Manages multiple `LoRAModel` instances and their integration with the base model.

```python
class LoRAModelManager:
    model: nn.Module                       # Base model with LoRA layers
    lora_uid_to_state: dict[int, LoRAModelState]  # Active LoRA states
    max_num_slots: int                     # Maximum adapter slots
    lora_index: dict[str, int]             # Name to index mapping
    id_to_lora: dict[int, LoRAModel]       # ID to LoRAModel mapping
```

### Key Methods

#### `__init__(model, max_num_slots, max_lora_rank, vocab_size, lora_extra_vocab_size, lora_dtype, device, lora_backend, max_cipher_size, enable_tower_connector_lora)`

Initialize the model manager.

**Parameters:**
- `model`: The base model with LoRA layers already injected.
- `max_num_slots`: Maximum number of adapter slots (equals `max_loras`).
- `max_lora_rank`: Maximum LoRA rank.
- `vocab_size`: Base model vocabulary size.
- `lora_extra_vocab_size`: Additional vocabulary size for LoRA.
- `lora_dtype`: Data type for LoRA weights.
- `device`: Target device.
- `lora_backend`: Punica backend for multi-LoRA inference.
- `max_cipher_size`: Maximum cipher size for prompt adapters.
- `enable_tower_connector_lora`: Whether to enable LoRA for vision tower connectors.

#### `add_adapter(lora_request)`

Adds a LoRA adapter to the manager.

```python
def add_adapter(self, lora_request: LoRARequest) -> bool
```

Returns `True` if the adapter was added, `False` if it was already present.

#### `remove_adapter(lora_id)`

Removes a LoRA adapter by its integer ID.

```python
def remove_adapter(self, lora_id: int) -> bool
```

#### `pin_adapter(lora_id)`

Pins a LoRA adapter so it cannot be evicted by the LRU cache.

```python
def pin_adapter(self, lora_id: int) -> bool
```

#### `list_adapters()`

Returns the set of currently active LoRA adapter IDs.

```python
def list_adapters(self) -> set[int]
```

#### `set_active_adapters(lora_requests, lora_mapping)`

Sets the currently active LoRA adapters and their token-level mapping for the current batch.

```python
def set_active_adapters(
    self,
    lora_requests: set[LoRARequest],
    lora_mapping: LoRAMapping,
) -> None
```

### Class: `AdapterLRUCache`

An LRU (Least Recently Used) cache for LoRA adapters.

```python
class AdapterLRUCache:
    capacity: int                              # Maximum number of adapters
    cache: OrderedDict[int, LoRAModel]         # Ordered dict for LRU ordering
    pinned: set[int]                           # Set of pinned adapter IDs
```

**Methods:**
- `get(lora_id)`: Get an adapter, updating its position in the LRU order.
- `put(lora_id, lora_model)`: Add an adapter, evicting the LRU entry if at capacity.
- `remove(lora_id)`: Remove an adapter from the cache.
- `pin(lora_id)`: Pin an adapter to prevent eviction.
- `unpin(lora_id)`: Unpin an adapter.

### Factory Functions

#### `create_lora_manager(model, vllm_config, ...)`

Factory function that creates a `LoRAModelManager` by:
1. Identifying LoRA-compatible layers in the model
2. Replacing them with LoRA-aware layer implementations
3. Wrapping multimodal tower/connector layers if needed
4. Initializing the Punica backend

```python
def create_lora_manager(
    model: nn.Module,
    vllm_config: VllmConfig,
    lora_config: LoRAConfig,
    device: torch.device,
    embedding_modules: dict[str, str],
    lora_model_cls: type[LoRAModel] | None = None,
    max_cipher_size: int | None = None,
    lora_manager_cls: type[LoRAModelManager] | None = None,
) -> LoRAModelManager
```

### Multimodal LoRA Support

The model manager supports multimodal LoRA through tower and connector wrappers:

- **Tower LoRA**: Applied to the vision tower (encoder) layers
- **Connector LoRA**: Applied to the connector layers between vision tower and LLM

When `enable_tower_connector_lora` is `True`, the model manager wraps LoRA layers to support separate `LoRAMappingType` for:
- `LANGUAGE`: Standard language model LoRA
- `TOWER`: Vision tower LoRA
- `CONNECTOR`: Vision-language connector LoRA

---

## 10. WorkerLoRAManager

**File**: `vllm/lora/worker_manager.py`

### Class: `WorkerLoRAManager`

Base class for worker-side LoRA management. Manages the loading, caching, and application of LoRA adapters on each worker.

```python
class WorkerLoRAManager:
    lora_config: LoRAConfig
    device: torch.device
    lora_manager: LoRAModelManager
```

### Key Methods

#### `__init__(vllm_config, device, embedding_modules, lora_manager_cls, max_cipher_size)`

Initialize the worker LoRA manager.

**Parameters:**
- `vllm_config`: Complete vLLM configuration.
- `device`: Target device.
- `embedding_modules`: Mapping of embedding module names.
- `lora_manager_cls`: Optional override for the LoRAModelManager class.
- `max_cipher_size`: Maximum cipher size for prompt adapters.

#### `create_lora_manager(model, vllm_config)`

Creates and initializes the `LoRAModelManager` for the given model. This replaces model layers with LoRA-aware versions.

```python
def create_lora_manager(
    self,
    model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module
```

#### `add_adapter(lora_request)`

Adds a LoRA adapter, loading it from disk if not already cached.

```python
def add_adapter(self, lora_request: LoRARequest) -> bool
```

#### `remove_adapter(lora_id)`

Removes a LoRA adapter from the worker.

#### `pin_adapter(lora_id)`

Pins a LoRA adapter to prevent eviction.

#### `list_adapters()`

Returns the set of active LoRA adapter IDs.

#### `set_active_adapters(lora_requests, lora_mapping)`

Sets the active adapters for the current batch.

#### `add_dummy_lora(lora_request, rank)`

Adds a dummy LoRA adapter for warmup/profiling purposes. Used during CUDA graph capture.

```python
def add_dummy_lora(
    self,
    lora_request: LoRARequest,
    rank: int,
) -> None
```

#### `dummy_lora_cache()` (context manager)

Context manager that creates a temporary cache for dummy LoRAs, ensuring they are properly cleaned up after use.

### Class: `LRUCacheWorkerLoRAManager`

Extends `WorkerLoRAManager` with LRU-based caching.

```python
class LRUCacheWorkerLoRAManager(WorkerLoRAManager):
    # Adds LRU eviction when max_loras is exceeded
    # Adds CPU caching with max_cpu_loras limit
```

**Additional Methods:**

#### `get_dummy_lora_warmup_rank(suggested_rank)`

Returns the appropriate LoRA rank for warmup, clamped to `max_lora_rank`.

```python
def get_dummy_lora_warmup_rank(self, suggested_rank: int) -> int
```

---

## 11. LoRA Layer Types

vLLM supports LoRA on the following layer types:

### LoRA Layer Hierarchy

```
BaseLayerWithLoRA (abstract base)
  --> BaseLinearLayerWithLoRA (base for linear layers)
      --> ReplicatedLinearWithLoRA
      --> ColumnParallelLinearWithLoRA
      --> MergedColumnParallelLinearWithLoRA
      --> QKVParallelLinearWithLoRA
      --> MergedQKVParallelLinearWithLoRA
      --> ColumnParallelLinearWithShardedLoRA
      --> RowParallelLinearWithLoRA
      --> RowParallelLinearWithShardedLoRA
      --> FusedMoEWithLoRA
      --> FusedMoE3DWithLoRA
  --> VocabParallelEmbeddingWithLoRA
  --> LogitsProcessorWithLoRA
```

### Layer Files

| File | Layers |
|------|--------|
| `vllm/lora/layers/base.py` | `BaseLayerWithLoRA` |
| `vllm/lora/layers/base_linear.py` | `BaseLinearLayerWithLoRA` |
| `vllm/lora/layers/replicated_linear.py` | `ReplicatedLinearWithLoRA` |
| `vllm/lora/layers/column_parallel_linear.py` | `ColumnParallelLinearWithLoRA`, `MergedColumnParallelLinearWithLoRA`, `QKVParallelLinearWithLoRA`, `MergedQKVParallelLinearWithLoRA`, `ColumnParallelLinearWithShardedLoRA`, `MergedColumnParallelLinearVariableSliceWithLoRA` |
| `vllm/lora/layers/row_parallel_linear.py` | `RowParallelLinearWithLoRA`, `RowParallelLinearWithShardedLoRA` |
| `vllm/lora/layers/logits_processor.py` | `LogitsProcessorWithLoRA` |
| `vllm/lora/layers/vocal_parallel_embedding.py` | `VocabParallelEmbeddingWithLoRA` |
| `vllm/lora/layers/fused_moe.py` | `FusedMoEWithLoRA`, `FusedMoE3DWithLoRA` |
| `vllm/lora/layers/utils.py` | `LoRAMappingType`, `LoRAMapping`, helper decorators |

### BaseLinearLayerWithLoRA

**File**: `vllm/lora/layers/base_linear.py`

```python
class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    # Dual CUDA stream support for async LoRA computation
    lora_a_stacked: dict[int, torch.Tensor]  # Stacked A weights per rank
    lora_b_stacked: dict[int, torch.Tensor]  # Stacked B weights per rank
```

**Key Methods:**

- **`create_lora_weights(max_loras, lora_config, model_config, ...) `**: Pre-allocates GPU tensors for stacked LoRA weights based on `max_loras` and `max_lora_rank`.
- **`set_lora(index, lora_a, lora_b, ...)`**: Sets LoRA weights for a specific adapter slot.
- **`set_mapping(lora_mapping, ...) `**: Sets the token-to-adapter mapping for the current batch.
- **`reset_lora()`**: Resets all LoRA weights to zero.
- **`apply(input_tensor, ...)`**: Applies LoRA computation. Supports dual-stream async execution.

**Dual Stream Support** (`VLLM_LORA_ENABLE_DUAL_STREAM`): When enabled, the LoRA computation is split across two CUDA streams for overlap:
- `_apply_async_impl()`: Performs shrink (x @ lora_a) and expand (result @ lora_b) on separate streams.

### ReplicatedLinearWithLoRA

**File**: `vllm/lora/layers/replicated_linear.py`

Applies LoRA to a non-parallel replicated linear layer.

```python
class ReplicatedLinearWithLoRA(BaseLinearLayerWithLoRA):
    # Wraps vllm.model_executor.layers.linear.ReplicatedLinear
```

**`can_replace_layer`**: Returns `True` when the original layer is `ReplicatedLinear` and LoRA is enabled.

### ColumnParallelLinearWithLoRA

**File**: `vllm/lora/layers/column_parallel_linear.py`

Applies LoRA to a column-parallel linear layer (output is sharded along the column dimension across TP ranks).

```python
class ColumnParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    # Wraps vllm.model_executor.layers.linear.ColumnParallelLinear
```

**LoRA Application**: The `_mcp_apply()` helper function handles multi-column-parallel LoRA:
1. Performs `add_shrink` (input @ lora_a for all active adapters)
2. Performs `add_expand` (shrink_result @ lora_b, added to base output)

### MergedColumnParallelLinearWithLoRA

**File**: `vllm/lora/layers/column_parallel_linear.py`

Applies LoRA to a merged column parallel layer (e.g., `gate_up_proj` which combines gate and up projections).

```python
class MergedColumnParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    # Handles packed modules with PackedLoRALayerWeights
```

Uses `PackedLoRALayerWeights` to handle the fused weight matrices from multiple sub-layers.

### QKVParallelLinearWithLoRA

**File**: `vllm/lora/layers/column_parallel_linear.py`

Applies LoRA to QKV parallel projections (query, key, value combined).

```python
class QKVParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    # Handles QKV repacking for TP sharding
```

**Special handling**: The QKV weights may need repacking to align with the model's TP sharding scheme. The `qkv_proj_repacking_index` parameter controls the reordering.

### MergedQKVParallelLinearWithLoRA

**File**: `vllm/lora/layers/column_parallel_linear.py`

Applies LoRA to merged QKV parallel projections that combine Q, K, V into a single weight.

### ColumnParallelLinearWithShardedLoRA

**File**: `vllm/lora/layers/column_parallel_linear.py`

Applies LoRA with fully-sharded weights on column-parallel layers. Each TP rank holds a shard of both the LoRA A and B matrices.

### RowParallelLinearWithLoRA

**File**: `vllm/lora/layers/row_parallel_linear.py`

Applies LoRA to a row-parallel linear layer (input is sharded along the row dimension, output is all-reduced).

```python
class RowParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    # Wraps vllm.model_executor.layers.linear.RowParallelLinear
```

### RowParallelLinearWithShardedLoRA

**File**: `vllm/lora/layers/row_parallel_linear.py`

Applies fully-sharded LoRA on row-parallel layers.

### LogitsProcessorWithLoRA

**File**: `vllm/lora/layers/logits_processor.py`

Applies LoRA to the logits processor, handling vocabulary size changes from LoRA adapters.

```python
class LogitsProcessorWithLoRA(BaseLayerWithLoRA):
    # Handles vocab reindexing for LoRA with extra vocabulary
```

**Special handling**: When a LoRA adds extra vocabulary tokens, the logits processor must remap token IDs between the base vocabulary and the expanded vocabulary.

**`_get_logits()`** method:
1. Computes logits from hidden states
2. Applies LoRA to the logits computation
3. Handles vocabulary reindexing for extra tokens

### VocabParallelEmbeddingWithLoRA

**File**: `vllm/lora/layers/vocal_parallel_embedding.py`

Applies LoRA to vocabulary parallel embedding layers.

```python
class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    # Wraps VocabParallelEmbedding with LoRA support
```

Handles both the embedding lookup and LoRA modification of embeddings.

### FusedMoEWithLoRA

**File**: `vllm/lora/layers/fused_moe.py`

Applies LoRA to fused Mixture of Experts (MoE) layers.

```python
class FusedMoEWithLoRA(BaseLinearLayerWithLoRA):
    # Handles MoE-specific LoRA with MoELoRAContext
```

**MoE-Specific Details**:
- Uses `PackedLoRALayerWeights.pack_moe()` for packing MoE LoRA weights.
- The `MoELoRAContext` manages per-expert LoRA computation.
- Supports both standard MoE (`FusedMoE`) and 3D MoE (`FusedMoE3D`).

---

## 12. Base Classes and Interfaces

### BaseLayerWithLoRA

**File**: `vllm/lora/layers/base.py`

```python
class BaseLayerWithLoRA(abc.ABC):
    @abc.abstractmethod
    def create_lora_weights(self, max_loras, lora_config, model_config, ...):
        """Pre-allocate GPU tensors for LoRA weights."""
        ...

    @abc.abstractmethod
    def reset_lora(self):
        """Reset all LoRA weights to zero."""
        ...

    @abc.abstractmethod
    def set_lora(self, index, lora_a, lora_b, ...):
        """Set LoRA weights for a specific adapter slot."""
        ...

    @abc.abstractmethod
    def set_mapping(self, lora_mapping, ...):
        """Set token-to-adapter mapping for current batch."""
        ...

    @staticmethod
    def can_replace_layer(layer, lora_config, ...):
        """Check if this LoRA layer can replace the given original layer."""
        ...
```

### LoRAMappingType

**File**: `vllm/lora/layers/utils.py`

```python
class LoRAMappingType(enum.Enum):
    LANGUAGE = "language"       # Standard language model LoRA
    TOWER = "tower"             # Vision tower LoRA
    CONNECTOR = "connector"     # Vision-language connector LoRA
```

### LoRAMapping

**File**: `vllm/lora/layers/utils.py`

```python
@dataclass
class LoRAMapping:
    token_lora_mapping: tuple[int, ...]  # Per-token LoRA ID (0 = base model)
    prompt_lora_mapping: tuple[int, ...] # Per-request LoRA ID (0 = base model)
    is_prefill: bool                     # Whether this is a prefill step
    type: LoRAMappingType               # Type of LoRA mapping
```

### Helper Decorators

**File**: `vllm/lora/layers/utils.py`

- **`_not_fully_sharded_can_replace`**: Decorator that checks if a layer can be replaced with a non-sharded LoRA variant. Used when `fully_sharded_loras=False`.
- **`_fully_sharded_can_replace`**: Decorator that checks if a layer can be replaced with a fully-sharded LoRA variant. Used when `fully_sharded_loras=True`.

Both decorators check:
1. LoRA is enabled (`lora_config` is not None)
2. The original layer type matches
3. The module name is in the target modules list

---

## 13. Punica Kernels for Multi-LoRA Serving

### Architecture

vLLM uses Punica-style kernels for efficient multi-LoRA batched inference. The key operations are:

1. **Shrink**: `output = input @ lora_a^T` for all active adapters in a single batched kernel
2. **Expand**: `output = shrink_result @ lora_b^T` for all active adapters
3. **Scaling**: Multiply by `scaling` factor and add to base output

### PunicaWrapperABC

**File**: `vllm/lora/punica_wrapper/punica_base.py`

```python
class PunicaWrapperABC(abc.ABC):
    @abc.abstractmethod
    def add_shrink(self, y, x, lora_a_stacked, scale, ...):
        """Compute y += scale * x @ lora_a^T for all active adapters."""
        ...

    @abc.abstractmethod
    def add_expand(self, y, x, lora_b_stacked, ...):
        """Compute y += x @ lora_b^T for all active adapters."""
        ...

    @abc.abstractmethod
    def add_lora_embedding(self, y, x, lora_b_stacked, ...):
        """Apply LoRA to embedding layer output."""
        ...

    @abc.abstractmethod
    def add_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale, ...):
        """Combined shrink + expand for linear layers."""
        ...

    @abc.abstractmethod
    def add_lora_logits(self, y, x, lora_a_stacked, lora_b_stacked, scale, ...):
        """Apply LoRA to logits computation."""
        ...

    @abc.abstractmethod
    def moe_lora_align_block_size(self, ...):
        """Align block sizes for MoE LoRA."""
        ...

    @abc.abstractmethod
    def add_lora_fused_moe(self, ...):
        """Apply LoRA to fused MoE layers."""
        ...
```

### PunicaWrapperGPU

**File**: `vllm/lora/punica_wrapper/punica_gpu.py`

GPU implementation using Triton kernels with `LoRAKernelMeta` for JIT compilation.

```python
class PunicaWrapperGPU(PunicaWrapperBase):
    # Implements all abstract methods using Triton kernels
```

### Punica Selector

**File**: `vllm/lora/punica_wrapper/punica_selector.py`

```python
def get_punica_wrapper(lora_config: LoRAConfig) -> PunicaWrapperABC:
    """Factory function that returns the platform-specific Punica wrapper."""
```

Uses the platform's `punica_qualname` to resolve the correct implementation.

### Utility Functions

**File**: `vllm/lora/punica_wrapper/utils.py`

#### `compute_meta(indices, num_layers, num_slots, ...)`

Computes metadata tensors for Punica kernels.

```python
def compute_meta(
    indices: torch.Tensor,   # Per-token adapter indices
    num_layers: int,         # Number of LoRA layers
    num_slots: int,          # Maximum adapter slots
) -> tuple[torch.Tensor, ...]:
```

#### `convert_mapping(mapping, num_loras, num_slots, ...)`

Converts a `LoRAMapping` into index tensors for Punica kernels.

```python
def convert_mapping(
    mapping: LoRAMapping,
    num_loras: int,
    num_slots: int,
    vocab_size: int,
    scaling: float,
    index_stacked: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
```

This function creates:
- **`index_stacked`**: Per-token adapter slot index tensor, shape `[num_tokens]`
- Converts LoRA IDs (1-indexed) to slot indices (0-indexed)
- Handles "no LoRA" tokens (ID 0) by mapping to a special null slot

---

## 14. LoRA Mapping and Utilities

### LoRAMapping Utilities

**File**: `vllm/lora/layers/utils.py`

#### `try_get_optimal_moe_lora_config(...)`

Determines the optimal MoE LoRA configuration based on model parameters and hardware.

```python
def try_get_optimal_moe_lora_config(
    lora_config: LoRAConfig | None,
    num_experts: int,
    ...
) -> dict | None:
```

### Utility Functions

**File**: `vllm/lora/utils.py`

#### `get_captured_lora_counts(cudagraph_capture_sizes, lora_config, ...)`

Computes the number of LoRA configurations needed for CUDA graph capture.

```python
def get_captured_lora_counts(
    cudagraph_capture_sizes: list[int],
    lora_config: LoRAConfig,
    ...
) -> dict[int, int]:
```

#### `get_lora_id(lora_request)`

Returns the LoRA integer ID from a request, or 0 if None.

```python
def get_lora_id(lora_request: LoRARequest | None) -> int:
```

#### `from_layer(module_name, original_layer, lora_config, ...)`

Creates a LoRA layer wrapper from the original model layer.

```python
def from_layer(
    module_name: str,
    original_layer: nn.Module,
    lora_config: LoRAConfig,
    ...
) -> BaseLayerWithLoRA | None:
```

#### `parse_fine_tuned_lora_name(name)`

Parses a LoRA weight name from a PEFT checkpoint.

```python
def parse_fine_tuned_lora_name(name: str) -> tuple[str, str]:
    """Returns (module_name, weight_type) where weight_type is 'lora_A' or 'lora_B'."""
```

#### `get_supported_lora_modules(model)`

Returns the list of module names that support LoRA in the given model.

```python
def get_supported_lora_modules(model: nn.Module) -> set[str]:
```

#### `is_in_target_modules(module_name, target_modules)`

Checks if a module name matches any of the target module patterns.

```python
def is_in_target_modules(
    module_name: str,
    target_modules: set[str] | None,
) -> bool:
```

#### `get_adapter_absolute_path(lora_path, adapter_name)`

Resolves an adapter path to an absolute path.

```python
def get_adapter_absolute_path(lora_path: str, adapter_name: str) -> str:
```

#### `process_packed_modules_mapping(packed_modules_mapping, target_modules)`

Processes the packed modules mapping to determine which modules need packed LoRA weights.

```python
def process_packed_modules_mapping(
    packed_modules_mapping: dict[str, list[str]],
    target_modules: set[str],
) -> dict[str, list[str]]:
```

---

## 15. LoRA Resolver

**File**: `vllm/lora/resolver.py`

### Class: `LoRAResolver`

Abstract base class for resolving LoRA adapters from requests.

```python
class LoRAResolver(abc.ABC):
    @abc.abstractmethod
    def resolve(self, request_id: str, lora_name: str) -> LoRARequest | None:
        """Resolve a LoRA request from a request ID and LoRA name."""
        ...
```

### Class: `_LoRAResolverRegistry`

Singleton registry for LoRA resolvers.

```python
class _LoRAResolverRegistry:
    _resolvers: dict[str, LoRAResolver]

    def register(self, name: str, resolver: LoRAResolver):
        """Register a LoRA resolver."""
        ...

    def resolve(self, request_id: str, lora_name: str) -> LoRARequest | None:
        """Resolve a LoRA request using registered resolvers."""
        ...
```

---

## 16. V1 LoRA Support (LoRAModelRunnerMixin)

**File**: `vllm/v1/worker/lora_model_runner_mixin.py`

### Class: `LoRAModelRunnerMixin`

Mixin class for V1 model runners that provides LoRA functionality.

```python
class LoRAModelRunnerMixin:
    lora_manager: LRUCacheWorkerLoRAManager
```

### Key Methods

#### `load_lora_model(model, vllm_config, device)`

Loads LoRA support into the model runner.

```python
def load_lora_model(
    self,
    model: nn.Module,
    vllm_config: VllmConfig,
    device: torch.device,
) -> nn.Module
```

1. Validates the model supports LoRA via `supports_lora(model)`
2. Creates an `LRUCacheWorkerLoRAManager`
3. Calls `create_lora_manager()` to inject LoRA layers

#### `set_active_loras(input_batch, num_scheduled_tokens, num_sampled_tokens, mapping_type)`

Sets the active LoRA adapters for the current batch based on the input batch state.

```python
def set_active_loras(
    self,
    input_batch: InputBatch,
    num_scheduled_tokens: np.ndarray,
    num_sampled_tokens: np.ndarray | None = None,
    mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
) -> None
```

1. Generates LoRA mappings from the input batch's `request_lora_mapping`
2. Creates `LoRAMapping` with `is_prefill=True` (uses SGMV kernels on non-CUDA)
3. Calls `lora_manager.set_active_adapters()`

#### `maybe_setup_dummy_loras(lora_config, remove_lora)` (context manager)

Context manager that sets up dummy LoRAs for CUDA graph capture warmup.

```python
@contextmanager
def maybe_setup_dummy_loras(
    self,
    lora_config: LoRAConfig | None,
    remove_lora: bool = True,
):
```

1. If `lora_config` is None, yields without action
2. Creates dummy `LoRARequest` objects with warmup rank
3. Adds dummy LoRAs via `lora_manager.add_dummy_lora()`
4. After context exit, optionally removes all adapters

#### `maybe_select_dummy_loras(lora_config, num_scheduled_tokens, ...)` (context manager)

Context manager that selects dummy LoRAs for capture/warmup with specific active LoRA counts.

```python
@contextmanager
def maybe_select_dummy_loras(
    self,
    lora_config: LoRAConfig | None,
    num_scheduled_tokens: np.ndarray,
    mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    num_sampled_tokens: np.ndarray | None = None,
    num_active_loras: int = 0,
):
```

**Parameters:**
- `num_active_loras`: Number of distinct active LoRAs:
  - `0`: No LoRA active (zero mappings)
  - `> max_loras`: Uses `max_loras` adapters plus no-LoRA tokens (-1)
  - `1 to max_loras`: Uses exactly that many distinct LoRAs

#### `maybe_dummy_run_with_lora(lora_config, ...)` (context manager)

Combined context manager that both sets up and selects dummy LoRAs.

```python
@contextmanager
def maybe_dummy_run_with_lora(
    self,
    lora_config: LoRAConfig | None,
    num_scheduled_tokens: np.ndarray,
    num_sampled_tokens: np.ndarray,
    remove_lora: bool = True,
    num_active_loras: int = 0,
    mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
):
```

#### `maybe_remove_all_loras(lora_config)`

Removes all LoRA adapters if LoRA is enabled.

#### `add_lora(lora_request)`, `remove_lora(lora_id)`, `pin_lora(lora_id)`, `list_loras()`

Delegate to `lora_manager` methods. Raise `RuntimeError` if LoRA is not enabled.

---

## 17. Tensor Parallelism for LoRA

### Non-Sharded LoRA (default)

When `fully_sharded_loras=False`:
- Each TP rank holds a **complete copy** of non-parallel LoRA weights
- Column-parallel layers: LoRA B is sharded along output dimension, A is replicated
- Row-parallel layers: LoRA A is sharded along input dimension, B is replicated

### Fully-Sharded LoRA

When `fully_sharded_loras=True`:
- LoRA weights are **fully sharded** across TP ranks
- Both A and B matrices are partitioned
- Uses `ColumnParallelLinearWithShardedLoRA` and `RowParallelLinearWithShardedLoRA`

### S-LoRA Based Tensor Parallelism

For column-parallel and row-parallel LoRA layers, vLLM implements S-LoRA style tensor parallelism:

- **Column Parallel**: The LoRA output (after shrink + expand) is sharded across TP ranks, matching the base weight's column partitioning.
- **Row Parallel**: The LoRA input is sharded across TP ranks, matching the base weight's row partitioning. The expand output is all-reduced.

---

## 18. Cache and Memory Management

### Memory Allocation

LoRA weights require pre-allocated GPU memory. The allocation is determined by:

```
memory_per_layer = max_loras * max_lora_rank * (input_dim + output_dim) * dtype_size
```

Total LoRA memory:
```
total = sum(memory_per_layer for each LoRA layer)
```

### LRU Eviction Policy

When the number of active LoRAs exceeds `max_loras`:
1. The least recently used adapter is identified
2. Its weights are cleared from GPU memory
3. The slot is freed for the new adapter
4. Pinned adapters are never evicted

### CPU Caching

When `max_cpu_loras` is set:
- Evicted GPU LoRAs are cached on CPU
- If the same LoRA is requested again, it can be loaded from CPU cache instead of disk
- The CPU cache has its own LRU eviction policy

---

## 19. Multimodal LoRA Support

### Overview

vLLM supports LoRA on multimodal models (e.g., Llava, Phi-3-Vision) through:

1. **Tower LoRA**: LoRA applied to the vision encoder (tower) layers
2. **Connector LoRA**: LoRA applied to the connector layers between vision and language
3. **Language LoRA**: Standard LoRA on the language model layers

### Configuration

```python
LoRAConfig(
    enable_tower_connector_lora=True,
    default_mm_loras={"IMAGE": "/path/to/default_lora"},
)
```

### Mapping Types

```python
class LoRAMappingType(enum.Enum):
    LANGUAGE = "language"       # Language model layers
    TOWER = "tower"             # Vision tower layers
    CONNECTOR = "connector"     # Connector layers
```

### Implementation

The model manager wraps LoRA layers with separate mapping types for each component. During the forward pass:
1. Language LoRA is applied with `LoRAMappingType.LANGUAGE`
2. Tower LoRA is applied with `LoRAMappingType.TOWER`
3. Connector LoRA is applied with `LoRAMappingType.CONNECTOR`

Each mapping type has its own token-to-adapter mapping, allowing different adapters per component.

---

## 20. Utility Functions

### _get_lora_device()

**File**: `vllm/lora/layers/utils.py`

Returns the device for LoRA tensors, preferring CUDA if available.

### Layer Replacement Flow

```python
# In create_lora_manager():
for name, module in model.named_modules():
    if is_in_target_modules(name, target_modules):
        if isinstance(module, ReplicatedLinear):
            replacement = ReplicatedLinearWithLoRA(...)
        elif isinstance(module, ColumnParallelLinear):
            replacement = ColumnParallelLinearWithLoRA(...)
        # ... etc
        replace_module(model, name, replacement)
```

---

## 21. Prompt Adapters

vLLM also supports prompt tuning through prompt adapters, which are managed alongside LoRA adapters.

### Prompt Adapter Configuration

Prompt adapters are configured through the `LoRAConfig`:
- `max_cipher_size`: Maximum cipher size for prompt adapters

### Integration

Prompt adapters share the same management infrastructure as LoRA:
- Same LRU cache
- Same slot allocation system
- Same mapping mechanism

---

## 22. Full Parameter Reference

### LoRAConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lora_rank` | `int` | `16` | Maximum LoRA rank |
| `max_loras` | `int` | `4` | Maximum cached LoRA adapters |
| `fully_sharded_loras` | `bool` | `False` | Fully shard LoRA across TP |
| `max_cpu_loras` | `int \| None` | `None` | Max CPU-cached LoRAs |
| `lora_dtype` | `str \| None` | `None` | LoRA weight data type |
| `lora_extra_vocab_size` | `int` | `256` | Extra vocabulary size |
| `target_modules` | `list[str] \| None` | `None` | Modules to apply LoRA |
| `default_mm_loras` | `dict \| None` | `None` | Default multimodal LoRAs |
| `enable_tower_connector_lora` | `bool` | `False` | Enable tower/connector LoRA |
| `specialize_active_lora` | `bool` | `False` | Specialize CG per LoRA config |

### LoRARequest Fields

| Field | Type | Description |
|-------|------|-------------|
| `lora_name` | `str` | Human-readable adapter name |
| `lora_int_id` | `int` | Integer ID (1-indexed, 0=no LoRA) |
| `lora_path` | `str` | Path to LoRA weights |
| `base_model_name` | `str \| None` | Base model name override |
| `tensorizer_config_dict` | `dict \| None` | Tensorizer config |
| `load_inplace` | `bool` | Load weights in-place |

### LoRAMapping Fields

| Field | Type | Description |
|-------|------|-------------|
| `token_lora_mapping` | `tuple[int, ...]` | Per-token LoRA ID |
| `prompt_lora_mapping` | `tuple[int, ...]` | Per-request LoRA ID |
| `is_prefill` | `bool` | Prefill vs decode |
| `type` | `LoRAMappingType` | LANGUAGE/TOWER/CONNECTOR |

### Supported Layer Types for LoRA

| Layer Type | LoRA Class | Sharded Variant |
|-----------|------------|-----------------|
| `ReplicatedLinear` | `ReplicatedLinearWithLoRA` | N/A |
| `ColumnParallelLinear` | `ColumnParallelLinearWithLoRA` | `ColumnParallelLinearWithShardedLoRA` |
| `MergedColumnParallelLinear` | `MergedColumnParallelLinearWithLoRA` | - |
| `QKVParallelLinear` | `QKVParallelLinearWithLoRA` | - |
| `RowParallelLinear` | `RowParallelLinearWithLoRA` | `RowParallelLinearWithShardedLoRA` |
| `VocabParallelEmbedding` | `VocabParallelEmbeddingWithLoRA` | - |
| `LogitsProcessor` | `LogitsProcessorWithLoRA` | - |
| `FusedMoE` | `FusedMoEWithLoRA` | - |
| `FusedMoE3D` | `FusedMoE3DWithLoRA` | - |
