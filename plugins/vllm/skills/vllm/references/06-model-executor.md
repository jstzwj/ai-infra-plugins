# Model Executor Reference

This document provides a comprehensive reference for the vLLM Model Executor subsystem, covering model loading, weight management, layer implementations, attention mechanisms, rotary embeddings, quantization, fused MoE, and model offloading.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Model Registry](#2-model-registry)
3. [Model Interfaces](#3-model-interfaces)
4. [Model Loaders](#4-model-loaders)
5. [Weight Loading Utilities](#5-weight-loading-utilities)
6. [Linear Layers](#6-linear-layers)
7. [Normalization Layers](#7-normalization-layers)
8. [Activation Functions](#8-activation-functions)
9. [Attention Layers](#9-attention-layers)
10. [MLA Attention](#10-mla-attention)
11. [Rotary Embeddings](#11-rotary-embeddings)
12. [Fused MoE (Mixture of Experts)](#12-fused-moe-mixture-of-experts)
13. [Quantization](#13-quantization)
14. [Model Offloading](#14-model-offloading)
15. [Kernel Warmup](#15-kernel-warmup)
16. [Supported Model Architectures](#16-supported-model-architectures)

---

## 1. Architecture Overview

The Model Executor is the subsystem responsible for:

- **Model Registration**: Mapping HuggingFace architecture names to vLLM model classes
- **Model Loading**: Initializing model weights from disk, downloading from HuggingFace Hub, and applying quantization
- **Layer Execution**: Providing optimized implementations of linear, attention, normalization, and activation layers
- **Parallel Execution**: Tensor parallel (TP), pipeline parallel (PP), data parallel (DP), and expert parallel (EP) support
- **Weight Management**: Sharding, offloading, prefetching, and quantization of model weights

### Key Directory Structure

```
vllm/model_executor/
  models/           # Model implementations (Llama, Mistral, DeepSeek, etc.)
    registry.py     # ModelRegistry - architecture-to-class mapping
    interfaces.py   # Protocol interfaces (SupportsMultiModal, SupportsLoRA, etc.)
    interfaces_base.py  # Base interfaces (VllmModel, VllmModelForTextGeneration)
  model_loader/     # Weight loading pipeline
    base_loader.py  # BaseModelLoader abstract class
    default_loader.py  # DefaultModelLoader for safetensors/bin/pt
    gguf_loader.py  # GGUFModelLoader for GGUF quantized files
    dummy_loader.py # DummyModelLoader for random weight initialization
    bitsandbytes_loader.py  # BitsAndBytes quantized model loader
    tensorizer_loader.py    # Tensorizer model loader
    sharded_state_loader.py # Sharded state loader
    runai_streamer_loader.py # RunAI model streamer loader
    weight_utils.py # Weight downloading, iteration, conversion utilities
    utils.py        # Model initialization and weight processing utilities
  layers/           # Layer implementations
    linear.py       # Linear layers (Replicated, ColumnParallel, RowParallel, QKV)
    layernorm.py    # Normalization layers (RMSNorm, LayerNorm, GemmaRMSNorm)
    activation.py   # Activation functions (SiLU, GELU, SwiGLU, etc.)
    attention/      # Attention layer implementations
    rotary_embedding/ # Rotary positional embeddings
    fused_moe/      # Fused Mixture of Experts implementation
    quantization/   # Quantization methods
  offloader/        # Weight offloading (CPU, UVA, prefetch)
  warmup/           # Kernel warmup utilities
```

---

## 2. Model Registry

**Source**: `vllm/model_executor/models/registry.py`

### ModelRegistry

A singleton instance of `_ModelRegistry` that maps HuggingFace architecture names to vLLM model implementation classes.

#### Architecture Categories

The registry organizes models into the following categories:

| Category | Description |
|----------|-------------|
| `_TEXT_GENERATION_MODELS` | Decoder-only causal LM models |
| `_EMBEDDING_MODELS` | Text embedding models |
| `_LATE_INTERACTION_MODELS` | ColBERT-style late interaction models |
| `_REWARD_MODELS` | Reward/process reward models |
| `_TOKEN_CLASSIFICATION_MODELS` | Token classification models |
| `_SEQUENCE_CLASSIFICATION_MODELS` | Sequence classification models |
| `_MULTIMODAL_MODELS` | Vision/language/audio multimodal models |
| `_SPECULATIVE_DECODING_MODELS` | Speculative decoding draft models (EAGLE, MTP, Medusa) |
| `_TRANSFORMERS_SUPPORTED_MODELS` | Models with explicit Transformers backend mapping |
| `_TRANSFORMERS_BACKEND_MODELS` | Generic Transformers backend models |

#### Key Methods

**`ModelRegistry.resolve_model_cls(architectures, model_config)`**

Resolves a list of HuggingFace architecture names to a concrete model class.

- **Parameters**:
  - `architectures` (`str | list[str]`): Architecture name(s) from `config.json`
  - `model_config` (`ModelConfig`): Model configuration
- **Returns**: `tuple[type[nn.Module], str]` - The model class and matched architecture name
- **Resolution Order**:
  1. Check for `transformers` backend requirement (`model_impl="transformers"`)
  2. Try fallback to transformers backend for unknown architectures (`model_impl="auto"`)
  3. Iterate architectures, normalize via `_normalize_arch`, attempt load
  4. Final fallback to transformers backend

**`ModelRegistry.inspect_model_cls(architectures, model_config)`**

Inspects model capabilities without loading the class.

- **Returns**: `tuple[_ModelInfo, str]` - Model metadata and architecture name

**`ModelRegistry.register_model(model_arch, model_cls)`**

Registers an external model. `model_cls` can be:
- A `torch.nn.Module` class directly
- A string `"module.path:ClassName"` for lazy import

**`ModelRegistry.is_text_generation_model(architectures, model_config)`** -> `bool`

**`ModelRegistry.is_pooling_model(architectures, model_config)`** -> `bool`

**`ModelRegistry.is_multimodal_model(architectures, model_config)`** -> `bool`

**`ModelRegistry.is_pp_supported_model(architectures, model_config)`** -> `bool`

**`ModelRegistry.is_attention_free_model(architectures, model_config)`** -> `bool`

**`ModelRegistry.is_hybrid_model(architectures, model_config)`** -> `bool`

#### _ModelInfo

A frozen dataclass containing model capability metadata:

| Field | Type | Description |
|-------|------|-------------|
| `architecture` | `str` | Model class name |
| `is_text_generation_model` | `bool` | Whether model generates text |
| `is_pooling_model` | `bool` | Whether model supports pooling |
| `attn_type` | `AttnTypeStr` | Attention type ("decoder", "encoder", "encoder_decoder") |
| `default_seq_pooling_type` | `SequencePoolingType` | Default sequence pooling type |
| `default_tok_pooling_type` | `TokenPoolingType` | Default token pooling type |
| `score_type` | `ScoreType` | Scoring type ("bi-encoder", "cross-encoder", "late-interaction") |
| `supports_multimodal` | `bool` | Multi-modal input support |
| `supports_multimodal_raw_input_only` | `bool` | Raw-only multimodal processing |
| `requires_raw_input_tokens` | `bool` | Requires raw input token IDs |
| `supports_multimodal_encoder_tp_data` | `bool` | Encoder TP data parallelism |
| `supports_pp` | `bool` | Pipeline parallel support |
| `has_inner_state` | `bool` | Has internal state (e.g., Mamba) |
| `is_attention_free` | `bool` | No attention layers (e.g., Mamba) |
| `is_hybrid` | `bool` | Hybrid attention+mamba (e.g., Jamba) |
| `has_noops` | `bool` | Has no-op layers |
| `supports_mamba_prefix_caching` | `bool` | Mamba prefix caching support |
| `supports_transcription` | `bool` | ASR transcription support |
| `supports_transcription_only` | `bool` | Transcription-only (no text gen) |

#### Lazy Loading

Models are registered via `_LazyRegisteredModel` which stores `(module_name, class_name)` and only imports the module when actually needed. This avoids initializing CUDA during import and enables caching of model info via JSON files in `$VLLM_CACHE_ROOT/modelinfos/`.

---

## 3. Model Interfaces

**Source**: `vllm/model_executor/models/interfaces.py`, `interfaces_base.py`

### Base Interfaces

#### VllmModel

The minimum interface required for all models in vLLM.

```python
class VllmModel(Protocol[T_co]):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> T_co: ...
```

#### VllmModelForTextGeneration

Extends `VllmModel` for generative models.

```python
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    def compute_logits(self, hidden_states: T) -> T | None: ...
```

#### VllmModelForPooling

Extends `VllmModel` for pooling/embedding models.

| Class Variable | Type | Default | Description |
|---------------|------|---------|-------------|
| `is_pooling_model` | `ClassVar[Literal[True]]` | - | Pooling model flag |
| `default_seq_pooling_type` | `ClassVar[SequencePoolingType]` | `"LAST"` | Default sequence pooling |
| `default_tok_pooling_type` | `ClassVar[TokenPoolingType]` | `"ALL"` | Default token pooling |
| `attn_type` | `ClassVar[AttnTypeStr]` | `"decoder"` | Attention type |
| `score_type` | `ClassVar[ScoreType]` | `"bi-encoder"` | Scoring type |
| `pooler` | `Pooler` | - | Pooler instance |

### Capability Interfaces (Protocols)

#### SupportsMultiModal

Required for multi-modal models (images, audio, video).

| Class Variable | Type | Default | Description |
|---------------|------|---------|-------------|
| `supports_multimodal` | `ClassVar[Literal[True]]` | - | Multi-modal support flag |
| `supports_multimodal_raw_input_only` | `ClassVar[bool]` | `False` | Process raw multimodal inputs |
| `supports_encoder_tp_data` | `ClassVar[bool]` | `False` | Encoder TP data mode support |
| `requires_raw_input_tokens` | `ClassVar[bool]` | `False` | Requires raw token IDs |

**Key Methods**:

- `get_placeholder_str(modality, i)` -> `str | None`: Get placeholder text for the i-th item
- `embed_multimodal(**kwargs)` -> `MultiModalEmbeddings`: Generate multimodal embeddings
- `embed_input_ids(input_ids, multimodal_embeddings, is_multimodal)` -> `Tensor`: Merge text and multimodal embeddings
- `get_language_model()` -> `VllmModel`: Get underlying language model
- `configure_mm_token_handling(vocab_size, mm_token_ids)`: Handle out-of-vocabulary multimodal tokens

#### SupportsLoRA

Required for LoRA adapter support.

| Class Variable | Type | Default | Description |
|---------------|------|---------|-------------|
| `supports_lora` | `ClassVar[Literal[True]]` | - | LoRA support flag |
| `is_3d_moe_weight` | `ClassVar[bool]` | `False` | 3D MoE weight format |
| `is_non_gated_moe` | `ClassVar[bool]` | `False` | Non-gated MoE |
| `embedding_modules` | `ClassVar[dict[str, str]]` | `{}` | Embedding module mapping |
| `packed_modules_mapping` | `dict[str, list[str]]` | `{}` | Packed modules mapping |
| `lora_skip_prefixes` | `ClassVar[list[str]]` | `[]` | Module prefixes to skip |

#### SupportsPP

Required for pipeline parallel support.

```python
class SupportsPP(Protocol):
    supports_pp: ClassVar[Literal[True]]
    def make_empty_intermediate_tensors(self, batch_size, dtype, device) -> IntermediateTensors: ...
    def forward(self, input_ids, positions, *, intermediate_tensors) -> IntermediateTensors | None: ...
```

#### HasInnerState

For models with internal state (e.g., Mamba, Jamba).

```python
class HasInnerState(Protocol):
    has_inner_state: ClassVar[Literal[True]]
```

#### IsAttentionFree

For models without attention (e.g., Mamba).

```python
class IsAttentionFree(Protocol):
    is_attention_free: ClassVar[Literal[True]]
```

#### IsHybrid

For hybrid attention+mamba models (e.g., Jamba, Bamba).

```python
class IsHybrid(Protocol):
    is_hybrid: ClassVar[Literal[True]]
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple[tuple, tuple]: ...
    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]: ...
```

#### SupportsMRoPE

For models supporting Multi-dimensional Rotary Position Embedding (e.g., Qwen2-VL).

```python
class SupportsMRoPE(Protocol):
    supports_mrope: ClassVar[Literal[True]]
    def get_mrope_input_positions(self, input_tokens, mm_features) -> tuple[Tensor, int]: ...
```

#### SupportsXDRoPE

For models supporting XD-RoPE (4D positional embeddings).

```python
class SupportsXDRoPE(Protocol):
    supports_xdrope: ClassVar[Literal[True]]
    def get_xdrope_input_positions(self, input_tokens, mm_features) -> Tensor: ...
```

#### SupportsTranscription

For ASR/speech-to-text models (e.g., Whisper).

| Class Variable | Type | Default |
|---------------|------|---------|
| `supported_languages` | `Mapping[str, str]` | Required |
| `supports_transcription` | `ClassVar[Literal[True]]` | - |
| `supports_transcription_only` | `ClassVar[bool]` | `False` |
| `supports_segment_timestamp` | `ClassVar[bool]` | `False` |
| `supports_explicit_language_detection` | `ClassVar[bool]` | `False` |
| `no_space_languages` | `ClassVar[set[str]]` | `{"ja", "zh"}` |

#### SupportsEagle / SupportsEagle3

For EAGLE speculative decoding draft models.

```python
class SupportsEagle(SupportsEagleBase, Protocol):
    supports_eagle: ClassVar[Literal[True]]
    has_own_lm_head: bool = False
    has_own_embed_tokens: bool = False

class SupportsEagle3(SupportsEagleBase, Protocol):
    supports_eagle3: ClassVar[Literal[True]]
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None: ...
    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]: ...
```

#### MixtureOfExperts

For MoE models with Expert Parallel Load Balancing (EPLB).

```python
class MixtureOfExperts(Protocol):
    expert_weights: MutableSequence[Sequence[Tensor]]
    num_moe_layers: int
    num_expert_groups: int
    num_logical_experts: int
    num_physical_experts: int
    num_local_physical_experts: int
    num_routed_experts: int
    num_shared_experts: int
    num_redundant_experts: int
    moe_layers: Iterable[nn.Module]
```

#### SupportsQuant

Mixin for models with custom quantization mappings.

```python
class SupportsQuant:
    hf_to_vllm_mapper: ClassVar[WeightsMapper | None] = None
    packed_modules_mapping: ClassVar[dict[str, list[str]] | None] = None
    quant_config: QuantizationConfig | None = None
```

#### SupportsEncoderCudaGraph

For models with vision encoders that support CUDA graph capture/replay.

---

## 4. Model Loaders

**Source**: `vllm/model_executor/model_loader/`

### Load Format Registry

The module-level `LoadFormats` literal defines supported formats:

| Format | Loader Class | Description |
|--------|-------------|-------------|
| `"auto"` | `DefaultModelLoader` | Auto-detect (defaults to HF or Mistral) |
| `"hf"` | `DefaultModelLoader` | HuggingFace format (safetensors + bin) |
| `"bitsandbytes"` | `BitsAndBytesModelLoader` | BitsAndBytes 4-bit/8-bit quantized |
| `"dummy"` | `DummyModelLoader` | Random weights for profiling |
| `"fastsafetensors"` | `DefaultModelLoader` | Fast safetensors with GDS |
| `"gguf"` | `GGUFModelLoader` | GGUF quantized format |
| `"instanttensor"` | `DefaultModelLoader` | InstantTensor library loader |
| `"mistral"` | `DefaultModelLoader` | Mistral consolidated format |
| `"npcache"` | `DefaultModelLoader` | NumPy cache format |
| `"pt"` | `DefaultModelLoader` | PyTorch `.pt` format |
| `"runai_streamer"` | `RunaiModelStreamerLoader` | RunAI streaming loader |
| `"runai_streamer_sharded"` | `ShardedStateLoader` | RunAI sharded streaming |
| `"safetensors"` | `DefaultModelLoader` | Pure safetensors |
| `"sharded_state"` | `ShardedStateLoader` | Sharded state format |
| `"tensorizer"` | `TensorizerLoader` | Tensorizer serialization |

### Registering Custom Loaders

```python
@register_model_loader("my_loader")
class MyModelLoader(BaseModelLoader):
    def download_model(self, model_config): ...
    def load_weights(self, model, model_config): ...
```

### get_model()

```python
def get_model(*, vllm_config, model_config=None, prefix="", load_config=None) -> nn.Module
```

High-level function that:
1. Gets the loader from `load_config`
2. Calls `loader.load_model(vllm_config, model_config, prefix)`

### BaseModelLoader

```python
class BaseModelLoader(ABC):
    def __init__(self, load_config: LoadConfig)
    
    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None
    
    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None
    
    def load_model(self, vllm_config, model_config, prefix="") -> nn.Module
```

#### `load_model()` Pipeline

1. Determine `load_device` from `device_config` or `load_config.device`
2. Set default torch dtype to `model_config.dtype`
3. Place model on target device
4. Call `initialize_model()` to create model instance
5. Log model inspection if `VLLM_LOG_MODEL_INSPECTION=1`
6. Call `self.load_weights(model, model_config)`
7. If online quantization detected, call `finalize_layerwise_processing()`
8. Call `process_weights_after_loading(model, model_config, target_device)`
9. Return `model.eval()`

### DefaultModelLoader

The primary loader supporting safetensors, PyTorch bin, and NP cache formats.

#### Source Dataclass

```python
@dataclasses.dataclass
class Source:
    model_or_path: str          # Model ID or path
    revision: str | None        # Optional model revision
    subfolder: str | None = None  # Subfolder in repo
    prefix: str = ""            # Weight name prefix
    fall_back_to_pt: bool = True  # Allow .pt fallback
    allow_patterns_overrides: list[str] | None = None
```

#### Weight Loading Pipeline

1. **`_prepare_weights()`**: Download from HuggingFace Hub or ModelScope, detect format, filter files
2. **`_get_weights_iterator()`**: Create appropriate iterator based on format:
   - `safetensors_weights_iterator()` - Standard safetensors
   - `multi_thread_safetensors_weights_iterator()` - Multi-threaded safetensors
   - `fastsafetensors_weights_iterator()` - FastSafetensors with GDS
   - `instanttensor_weights_iterator()` - InstantTensor library
   - `pt_weights_iterator()` - PyTorch `.bin`/`.pt`
   - `multi_thread_pt_weights_iterator()` - Multi-threaded PyTorch
   - `np_cache_weights_iterator()` - NumPy cache
3. **`get_all_weights()`**: Iterate primary + secondary weights
4. **`load_weights()`**: Call `model.load_weights()` with the iterator, track loaded weights

#### EP Weight Filtering

When Expert Parallelism is active, `_init_ep_weight_filter()` computes local expert IDs so that non-local expert weights are skipped before reading from disk. This reduces I/O for MoE models.

### DummyModelLoader

Initializes models with random weights for profiling and benchmarking.

- Calls `initialize_dummy_weights()` for all parameters
- Handles online quantization layers by materializing, applying dummy weights, and running `process_weights_after_loading()`

### GGUFModelLoader

Loads models in GGUF quantized format.

- Supports single files, sharded files, and `repo_id:quant_type` notation
- Uses `gguf_quant_weights_iterator()` and `gguf_quant_weights_iterator_multi()` for sharded models
- Two-pass loading: first yield weight types, then yield weight data

---

## 5. Weight Loading Utilities

**Source**: `vllm/model_executor/model_loader/weight_utils.py`

### Download Functions

#### `download_weights_from_hf(model_name_or_path, cache_dir, allow_patterns, revision, subfolder, ignore_patterns)` -> `str`

Downloads model weights from HuggingFace Hub with file locking to prevent concurrent downloads.

#### `maybe_download_from_modelscope(model, revision, download_dir, ignore_patterns, allow_patterns)` -> `str | None`

Downloads from ModelScope if `VLLM_USE_MODELSCOPE` is enabled.

#### `download_gguf(repo_id, quant_type, cache_dir, revision, ignore_patterns)` -> `str`

Downloads GGUF files matching a specific quantization type.

#### `download_safetensors_index_file_from_hf(model_name_or_path, index_file, cache_dir, subfolder, revision)`

Downloads safetensors index file for deduplication.

### Weight Iterator Functions

#### `safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids)` -> `Generator[tuple[str, Tensor]]`

Iterates over safetensors weights with automatic prefetching on network filesystems (NFS/Lustre).

- **`safetensors_load_strategy`** options:
  - `None` (auto): Prefetch on NFS/Lustre if checkpoint fits in RAM
  - `"eager"`: Load entire files into memory
  - `"prefetch"`: Force prefetch to page cache
  - `"torchao"`: Load torchao serialized tensors

#### `multi_thread_safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load, max_workers=4)`

Multi-threaded safetensors loading using `ThreadPoolExecutor`.

#### `pt_weights_iterator(hf_weights_files, use_tqdm_on_load, pt_load_map_location="cpu")`

Iterates over PyTorch `.bin`/`.pt` files.

#### `multi_thread_pt_weights_iterator(hf_weights_files, use_tqdm_on_load, pt_load_map_location="cpu", max_workers=4)`

Multi-threaded PyTorch loading.

#### `np_cache_weights_iterator(model_name_or_path, cache_dir, hf_folder, hf_weights_files, use_tqdm_on_load)`

Converts `.bin` weights to NumPy format for faster subsequent loads.

#### `gguf_quant_weights_iterator(gguf_file, gguf_to_hf_name_map)` -> `Generator`

Two-pass GGUF loading: yields weight types first, then weight data.

#### `gguf_quant_weights_iterator_multi(gguf_files, gguf_to_hf_name_map)` -> `Generator`

Multi-shard GGUF loading.

### Weight Loader Functions

#### `default_weight_loader(param, loaded_weight)`

Default weight loader - copies `loaded_weight` into `param.data`.

#### `row_parallel_weight_loader(param, loaded_weight)`

Shards weights along dimension 0 for row-parallel linear layers.

#### `sharded_weight_loader(shard_axis)` -> `LoaderFunction`

Creates a weight loader that shards along a specific axis.

#### `composed_weight_loader(loader, fn)` -> `LoaderFunction`

Composes a loader with a post-processing function.

### Utility Functions

#### `get_quant_config(model_config, load_config)` -> `QuantizationConfig`

Reads quantization configuration from:
1. `model_config.hf_config.quantization_config`
2. `model_config.hf_config.compression_config` (compressed-tensors)
3. `model_config.hf_overrides.quantization_config_file`
4. `model_config.hf_overrides.quantization_config_dict_json`
5. Online quantization config from `model_config.quantization_config`
6. Separate JSON config file on disk

#### `initialize_dummy_weights(model, model_config, low=-1e-3, high=1e-3, seed=1234)`

Initializes all model weights with uniform random values in `[low, high]`. Uses per-parameter seeding for consistency across TP ranks.

#### `maybe_remap_kv_scale_name(name, params_dict)` -> `str | None`

Remaps FP8 KV scale parameter names from checkpoint format to vLLM internal format. Handles multiple naming conventions across different quantization tools.

#### `atomic_writer(filepath, mode, encoding)` -> `Generator[IO]`

Context manager for atomic file writes using temporary files and `os.replace()`.

### Model Initialization

**Source**: `vllm/model_executor/model_loader/utils.py`

#### `initialize_model(vllm_config, prefix, model_class, model_config)` -> `nn.Module`

1. Resolves model class via `get_model_architecture()`
2. Configures quantization config if present
3. Instantiates model with `(vllm_config, prefix)` arguments
4. Records metadata for reloading

#### `process_weights_after_loading(model, model_config, target_device)`

Post-load weight processing:
1. Iterates all modules, calls `quant_method.process_weights_after_loading()` on each
2. Handles device loading context for CPU-offloaded parameters
3. Initializes attention weights for `Attention`, `MLAAttention`, `MMEncoderAttention`
4. Sets torchao reload attributes if applicable

#### `get_model_architecture(model_config)` -> `tuple[type[nn.Module], str]`

Resolves model architecture with caching. Supports `convert_type` for embedding/classification conversion.

#### `ParamMapping`

Bidirectional mapping between packed parameters and their constituents.

```python
@dataclass
class ParamMapping:
    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]]
```

---

## 6. Linear Layers

**Source**: `vllm/model_executor/layers/linear.py`

### Class Hierarchy

```
LinearBase (PluggableLayer)
  +-- ReplicatedLinear
  +-- ColumnParallelLinear
  |     +-- MergedColumnParallelLinear
  |     +-- QKVParallelLinear
  +-- RowParallelLinear
```

### LinearBase

Base class for all linear layers.

```python
class LinearBase(PluggableLayer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | required | Input dimension |
| `output_size` | `int` | required | Output dimension |
| `bias` | `bool` | `False` | Add bias parameter |
| `skip_bias_add` | `bool` | `False` | Return bias separately |
| `params_dtype` | `torch.dtype` | `None` | Parameter dtype (uses default if None) |
| `quant_config` | `QuantizationConfig` | `None` | Quantization configuration |
| `prefix` | `str` | `""` | Layer name in state dict |
| `return_bias` | `bool` | `True` | Include bias in return value |
| `disable_tp` | `bool` | `False` | Disable tensor parallelism |

### ReplicatedLinear

Fully replicated linear layer (no TP sharding).

```python
class ReplicatedLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=True, skip_bias_add=False,
                 params_dtype=None, quant_config=None, prefix="",
                 *, return_bias=True, disable_tp=False)
    
    def weight_loader(self, param, loaded_weight)
    def forward(self, x) -> Tensor | tuple[Tensor, Parameter | None]
```

### ColumnParallelLinear

Linear layer with column (output) parallelism. Weight matrix A is split along columns: `A = [A_1, ..., A_p]`.

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=True,
                 gather_output=False, skip_bias_add=False,
                 params_dtype=None, quant_config=None, prefix="",
                 *, return_bias=True, disable_tp=False)
```

| Extra Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `gather_output` | `bool` | `False` | All-gather output across TP ranks |

**Forward**: `output = all_gather(quant_method(x, bias))` if `gather_output` else `quant_method(x, bias)`

### MergedColumnParallelLinear

Packed column-parallel layer for fused MLP projections (e.g., `gate_up_proj`).

```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_sizes, bias=True,
                 gather_output=False, skip_bias_add=False,
                 params_dtype=None, quant_config=None, prefix="",
                 *, return_bias=True, disable_tp=False)
```

| Extra Parameter | Type | Description |
|----------------|------|-------------|
| `output_sizes` | `list[int]` | Output dimensions for each logical weight |

**Weight Loader**: Handles both pre-fused (e.g., `gate_up_proj`) and separate (e.g., `gate_proj`, `up_proj`) loading patterns.

### QKVParallelLinear

Column-parallel layer for QKV attention projections. Handles GQA/MQA head replication.

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads,
                 total_num_kv_heads=None, bias=True, skip_bias_add=False,
                 params_dtype=None, quant_config=None, prefix="",
                 *, return_bias=True, disable_tp=False, v_head_size=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_size` | `int` | Input hidden dimension |
| `head_size` | `int` | Size of each attention head |
| `total_num_heads` | `int` | Total query heads |
| `total_num_kv_heads` | `int | None` | Total KV heads (None = same as query) |
| `v_head_size` | `int | None` | V head dimension (None = head_size) |

**Output sizes**: `[num_heads * head_size, num_kv_heads * head_size, num_kv_heads * v_head_size]` (replicated across TP)

**Weight Loader Shard IDs**: `"q"`, `"k"`, `"v"` strings instead of integers.

### RowParallelLinear

Linear layer with row (input) parallelism. Weight matrix A is split along rows, input X is split along columns.

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=True, skip_bias_add=False,
                 params_dtype=None, reduce_results=True,
                 quant_config=None, prefix="",
                 *, return_bias=True, disable_tp=False)
```

| Extra Parameter | Type | Default | Description |
|----------------|------|---------|-------------|
| `input_is_parallel` | `bool` | `True` | Input is already split across TP ranks |
| `reduce_results` | `bool` | `True` | All-reduce output across TP ranks |

**Forward**:
1. Split input if not already parallel
2. `output_parallel = quant_method(input_parallel, bias)` (bias only on rank 0)
3. `output = all_reduce(output_parallel)` if `reduce_results` else `output_parallel`

### LinearMethodBase

Abstract base for quantized linear methods.

```python
class LinearMethodBase(QuantizeMethodBase):
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs)
    def apply(self, layer, x, bias=None) -> Tensor
```

### UnquantizedLinearMethod

Default unquantized linear implementation.

- **`create_weights()`**: Creates `ModelWeightParameter` with shape `(sum(output_partition_sizes), input_size_per_partition)`
- **`apply()`**: Dispatches to platform-optimized GEMM (CUDA, CPU, XPU)
- **`process_weights_after_loading()`**: On CPU, dispatches to optimized CPU GEMM

### Weight Loader V2

Selected linear methods support an optimized weight loader V2 that uses `BasevLLMParameter` subclasses for more efficient weight handling. Checked via `WEIGHT_LOADER_V2_SUPPORTED` list.

---

## 7. Normalization Layers

**Source**: `vllm/model_executor/layers/layernorm.py`

### RMSNorm

Root Mean Square Normalization. Computes `x -> w * x / sqrt(E[x^2] + eps)`.

```python
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | required | Hidden dimension size |
| `eps` | `float` | `1e-6` | Epsilon for numerical stability |
| `var_hidden_size` | `int | None` | `None` | Override variance computation dimension |
| `has_weight` | `bool` | `True` | Whether to use learnable weight |
| `dtype` | `torch.dtype` | `None` | Weight data type |

**Forward signature**: `forward(x, residual=None) -> Tensor | tuple[Tensor, Tensor]`

When `residual` is provided, performs fused `residual + x` then RMSNorm (fused_add_rms_norm). Supports native and CUDA kernel implementations.

### GemmaRMSNorm

RMS normalization for Gemma models. Uses `x * (1 + w)` instead of `x * w`.

```python
class GemmaRMSNorm(CustomOp):
    def __init__(self, hidden_size: int, eps: float = 1e-6)
```

Weight is initialized to zeros (not ones), since the formula is `(1 + w)`.

### RMSNormGated

RMS normalization with optional gating and group normalization.

```python
class RMSNormGated(CustomOp):
    def __init__(self, hidden_size, eps=1e-5, group_size=None,
                 norm_before_gate=False, device=None, dtype=None, activation="swish")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_size` | `int | None` | `None` | Group RMS norm group size |
| `norm_before_gate` | `bool` | `False` | Apply norm before gating |
| `activation` | `str` | `"swish"` | Gate activation ("silu", "sigmoid", "swish") |

**Formula**:
- `norm_before_gate=True`: `out = norm(x) * silu(z)`
- `norm_before_gate=False`: `out = norm(x * silu(z))`

### LayerNorm

Standard Layer Normalization using `F.layer_norm`.

```python
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6)
```

### PolyNorm

Polynomial normalization with custom kernel.

```python
def poly_norm(x, weight, bias, variance_epsilon) -> Tensor
```

---

## 8. Activation Functions

**Source**: `vllm/model_executor/layers/activation.py`

### Gated Activation Functions (split-and-multiply)

These functions split the input tensor in half along the last dimension, apply activation to the first half, and multiply by the second half.

#### SiluAndMul (SwiGLU)

`x -> silu(x[:d]) * x[d:]` where `d = x.shape[-1] // 2`

```python
@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    def __init__(self, *, compile_native=True)
```

#### GeluAndMul (GeGLU)

`x -> GELU(x[:d]) * x[d:]`

```python
@CustomOp.register("gelu_and_mul")
class GeluAndMul(CustomOp):
    def __init__(self, approximate: str = "none")  # "none" or "tanh"
```

#### MulAndSilu

`x -> x[:d] * silu(x[d:])` (reversed order)

```python
@CustomOp.register("mul_and_silu")
class MulAndSilu(CustomOp)
```

#### FatreluAndMul

`x -> FATReLU(x[:d]) * x[d:]` with threshold

```python
@CustomOp.register("fatrelu_and_mul")
class FatreluAndMul(CustomOp):
    def __init__(self, threshold: float = 0.0)
```

#### SiluAndMulWithClamp

SwiGLU with input clamping for MoE shared experts.

```python
@CustomOp.register("silu_and_mul_with_clamp")
class SiluAndMulWithClamp(CustomOp):
    def __init__(self, swiglu_limit: float, *, compile_native=True)
```

#### SwigluOAIAndMul

OpenAI-style SwiGLU: `gate = clamp(x[..., ::2], max=limit)`, `up = clamp(x[..., 1::2], -limit, limit)`, `out = (up + 1) * gate * sigmoid(gate * alpha)`.

```python
@CustomOp.register("swigluoai_and_mul")
class SwigluOAIAndMul(CustomOp):
    def __init__(self, alpha: float = 1.702, limit: float = 7.0)
```

#### SwigluStepAndMul

SwiGLU with step clamping: `silu(x[:d]).clamp(max=limit) * x[d:].clamp(-limit, limit)`

```python
@CustomOp.register("swiglustep_and_mul")
class SwigluStepAndMul(CustomOp):
    def __init__(self, limit: float = 7.0)
```

#### GeluAndMulSparse

GELU with Gaussian top-k sparsity for Gemma3n.

```python
@CustomOp.register("gelu_and_mul_sparse")
class GeluAndMulSparse(CustomOp):
    def __init__(self, activation_sparsity: float, approximate: str = "none")
```

### Simple Activation Functions

| Class | Registration | Formula |
|-------|-------------|---------|
| `GELU` | `"gelu"` | `F.gelu(x, approximate="none")` |
| `NewGELU` | `"gelu_new"` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` |
| `FastGELU` | `"gelu_fast"` | `0.5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x^2)))` |
| `QuickGELU` | `"quick_gelu"` | `x * sigmoid(1.702 * x)` |
| `ReLUSquaredActivation` | `"relu2"` | `relu(x)^2` |
| `XIELU` | `"xielu"` | Piecewise: `alpha_p * x^2 + beta * x` (x > 0), otherwise negative branch |

### Activation Registry

```python
_ACTIVATION_REGISTRY = {
    "gelu": GELU,
    "gelu_fast": FastGELU,
    "gelu_new": NewGELU,
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "silu": nn.SiLU,
    "quick_gelu": QuickGELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "xielu": XIELU,
}

_ACTIVATION_AND_MUL_REGISTRY = {
    "gelu": GeluAndMul,
    "silu": SiluAndMul,
    "geglu": GeluAndMul,
    "swigluoai": SwigluOAIAndMul,
}
```

#### `get_act_fn(act_fn_name)` -> `nn.Module`

Returns activation function by name.

#### `get_act_and_mul_fn(act_fn_name)` -> `nn.Module`

Returns gated activation function by name.

### ScaledActivation

Activation with post-scale parameters (used for AWQ dequantization).

```python
class ScaledActivation(nn.Module):
    def __init__(self, act_module, intermediate_size, input_is_parallel=True, params_dtype=None)
    def forward(self, x) -> Tensor:  # act(x) / scales
```

---

## 9. Attention Layers

**Source**: `vllm/model_executor/layers/attention/`

### Attention

The primary attention layer supporting MHA, MQA, and GQA.

```python
class Attention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        ...
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | `int` | required | Number of query heads (per TP rank) |
| `head_size` | `int` | required | Dimension of each head |
| `scale` | `float` | required | Attention scale factor (typically `1/sqrt(head_size)`) |
| `num_kv_heads` | `int | None` | `None` | Number of KV heads (None = same as num_heads) |
| `alibi_slopes` | `list[float] | None` | `None` | ALiBI slopes per head |
| `cache_config` | `CacheConfig | None` | `None` | KV cache configuration |
| `quant_config` | `QuantizationConfig | None` | `None` | Quantization config |
| `logits_soft_cap` | `float | None` | `None` | Logits soft capping value |
| `per_layer_sliding_window` | `int | None` | `None` | Per-layer sliding window size |

**Key Attributes**:
- `kv_cache_dtype`: Data type for KV cache (from `cache_config`)
- `attn_type`: `AttentionType` enum (FULL, WINDOWED, SLIDING_WINDOW)
- `_k_scale`, `_v_scale`, `_q_scale`, `_prob_scale`: FP8 scale tensors
- `kv_sharing_target_layer_name`: Layer name for KV cache sharing

**Forward**: Processes query/key/value through the attention backend, stores KV in cache, returns attention output.

### MLAAttention

Multi-head Latent Attention for DeepSeek-V2/V3 models. Uses compressed KV cache.

Key characteristics:
- Compressed KV representation: KV cache stores latent vectors instead of full K/V
- Two compute paths:
  - **Compute-friendly** (prefill): Decompress KV, run standard MHA
  - **Memory-friendly** (decode): Run attention in compressed latent space (MQA-style)

### MMEncoderAttention

Attention layer for multimodal encoder models.

### EncoderOnlyAttention

Attention for encoder-only models (e.g., BERT embeddings).

### CrossAttention

Cross-attention layer for encoder-decoder models.

### ChunkedLocalAttention

Local attention with chunking for long sequences.

### StaticSinkAttention

Attention with static sink tokens.

### KV Transfer Utilities

**Source**: `vllm/model_executor/layers/attention/kv_transfer_utils.py`

`maybe_transfer_kv_layer()` - Handles KV cache transfer between layers for KV sharing.

---

## 10. MLA Attention

**Source**: `vllm/model_executor/layers/attention/mla_attention.py`

### Architecture

MLA (Multi-head Latent Attention) compresses KV pairs into a latent representation, reducing memory bandwidth during decode.

#### Vector/Matrix Definitions

| Symbol | Shape | Description |
|--------|-------|-------------|
| `h_t` | `[Sq, H]` | Hidden states input |
| `q_c` | `[Sq, Lq]` | Latent/compressed Q |
| `q_nope` | `[Sq, N, P]` | Uncompressed Q (no RoPE) |
| `q_pe` | `[Sq, N, R]` | Uncompressed Q (with RoPE) |
| `kv_c` | `[Skv, Lkv]` | Latent/compressed KV |
| `k_pe` | `[Skv, R]` | Decoupled K position embeddings |

#### Compute-Friendly Path (Prefill/MHA)

1. Compress Q and KV: `q_c = h_t @ W_DQ`, `kv_c = h_t @ W_DKV`
2. Decompress: `q_nope = q_c @ W_UQ`, `k_nope = kv_c @ W_UK`, `v = kv_c @ W_UV`
3. Apply RoPE to position dims
4. Run standard MHA with QK head dim = `P + R`, V head dim = `V`

#### Data-Movement Friendly Path (Decode/MQA)

1. Project Q into latent KV space: `q_latent = einsum(q, W_UK)`
2. Run MQA attention in compressed latent space
3. Project output through `W_UV` and `W_O`

---

## 11. Rotary Embeddings

**Source**: `vllm/model_executor/layers/rotary_embedding/`

### get_rope() Factory

```python
def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding
```

Factory function that creates the appropriate rotary embedding class based on `rope_parameters["rope_type"]`.

Instances are cached in `_ROPE_DICT` for reuse.

### RotaryEmbedding (Base)

```python
class RotaryEmbeddingBase(CustomOp):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,  # Default: 10000
        is_neox_style: bool,  # GPT-NeoX style vs GPT-J style
        dtype: torch.dtype,
        init_cache: bool = True,
    )
```

**Key Attributes**:
- `cos_sin_cache`: Precomputed cos/sin values, shape `[max_position, rotary_dim]`
- `apply_rotary_emb`: `ApplyRotaryEmb` helper for dispatch

### Supported RoPE Types

| `rope_type` | Class | Description |
|-------------|-------|-------------|
| `"default"` | `RotaryEmbedding` | Standard RoPE |
| `"linear"` | `LinearScalingRotaryEmbedding` | Linear position scaling |
| `"ntk"` | `NTKScalingRotaryEmbedding` | NTK-aware scaling |
| `"dynamic"` | `DynamicNTKScalingRotaryEmbedding` | Dynamic NTK scaling (with `factor`) |
| `"dynamic"` | `DynamicNTKAlphaRotaryEmbedding` | Dynamic NTK scaling (with `alpha`) |
| `"llama3"` | `Llama3RotaryEmbedding` | Llama 3 low/high freq factor scaling |
| `"yarn"` | `YaRNScalingRotaryEmbedding` | YaRN position extrapolation |
| `"deepseek_yarn"` | `DeepseekScalingRotaryEmbedding` | DeepSeek YaRN variant |
| `"deepseek_llama_scaling"` | `DeepseekScalingRotaryEmbedding` | DeepSeek Llama scaling |
| `"longrope"` | `Phi3LongRoPEScaledRotaryEmbedding` | Phi-3 LongRoPE scaling |
| `"mllama4"` | `Llama4VisionRotaryEmbedding` | Llama 4 vision RoPE |
| `"proportional"` | `Gemma4RotaryEmbedding` | Gemma 4 proportional RoPE |
| `"xdrope"` | `XDRotaryEmbedding` | XD-RoPE (4D positional) |
| `"openpangu"` | `MRotaryEmbeddingInterleaved` | Pangu interleaved M-RoPE |
| `"telechat3-yarn"` | `TeleChat3RoPEScaledRotaryEmbedding` | TeleChat3 YaRN variant |

### Special RoPE Types

#### MRotaryEmbedding (M-RoPE)

Multi-dimensional RoPE used by Qwen2-VL and similar models. Supports spatial (H, W) and temporal (T) position encoding with configurable section sizes.

#### DualChunkRotaryEmbedding

RoPE for dual-chunk attention with chunk and local sizes.

#### FourierRotaryEmbedding (FOPE)

Fourier-based rotary embedding with configurable frequency count.

### RoPE Parameters

Common parameters in `rope_parameters` dict:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rope_theta` | `float` | `10000` | Base frequency |
| `rope_type` | `str` | `"default"` | Scaling type |
| `rope_dim` | `int` | `head_size` | Rotary dimension |
| `partial_rotary_factor` | `float` | `1.0` | Fraction of head dim to rotate |
| `factor` | `float` | - | Scaling factor |
| `original_max_position_embeddings` | `int` | - | Original max position before scaling |
| `low_freq_factor` | `float` | - | Llama3 low freq factor |
| `high_freq_factor` | `float` | - | Llama3 high freq factor |
| `mrope_section` | `list[int]` | - | M-RoPE section sizes |
| `xdrope_section` | `list[int]` | - | XD-RoPE section sizes |
| `short_factor` | `list[float]` | - | LongRoPE short factors |
| `long_factor` | `list[float]` | - | LongRoPE long factors |
| `extrapolation_factor` | `float` | - | YaRN extrapolation factor |
| `attn_factor` | `float` | - | YaRN attention factor |
| `beta_fast` | `float` | - | YaRN beta fast |
| `beta_slow` | `float` | - | YaRN beta slow |

---

## 12. Fused MoE (Mixture of Experts)

**Source**: `vllm/model_executor/layers/fused_moe/`

### FusedMoE Layer

The primary MoE layer implementation.

```python
class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
```

### Expert Map Calculation

#### `determine_expert_map(ep_size, ep_rank, global_num_experts, expert_placement_strategy, num_fused_shared_experts, return_expert_mask)` -> `tuple[int, Tensor | None, Tensor | None]`

Distributes experts across EP ranks.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ep_size` | `int` | Expert parallel size |
| `ep_rank` | `int` | Current EP rank |
| `global_num_experts` | `int` | Total experts in model |
| `expert_placement_strategy` | `str` | `"linear"` or `"round_robin"` |

**Returns**: `(local_num_experts, expert_map, expert_mask)`

- `expert_map`: Tensor mapping global to local indices (-1 for non-local)
- `expert_mask`: Binary mask for AITER MOE

### FusedMoE Configuration

#### FusedMoEConfig

Configuration for MoE layer behavior.

#### FusedMoEParallelConfig

Parallel execution configuration including all2all backend settings.

#### FusedMoEQuantConfig

Quantization configuration for MoE weights.

### MoE Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| `MoERunner` | `runner/` | Main MoE execution runner |
| `SharedExperts` | `runner/` | Shared expert implementation |
| `MoEActivation` | `activation.py` | MoE activation functions |
| Router | `router/` | Expert routing (top-k, load balancing) |
| Experts | `experts/` | Expert weight management |
| Prepare/Finalize | `prepare_finalize/` | Input/output permutation |
| Modular Kernel | `modular_kernel.py` | Modular MoE kernel interface |

### FusedMoEMethodBase

Abstract base class for MoE execution methods.

### UnquantizedFusedMoEMethod

Default unquantized MoE implementation.

### Supported MoE Backends

| Backend | File | Description |
|---------|------|-------------|
| Triton | `fused_moe.py` | Triton-based fused MoE |
| Marlin | `fused_marlin_moe.py` | Marlin-quantized MoE |
| Batched | `fused_batched_moe.py` | Batched GEMM approach |
| Cutlass | `flashinfer_cutlass_moe.py` | FlashInfer CUTLASS MoE |
| Triton-CUTLASS | `triton_cutlass_moe.py` | Triton + CUTLASS hybrid |
| Deep GEMM | `triton_deep_gemm_moe.py` | Deep GEMM MoE |
| Humming | `fused_humming_moe.py` | Humming quantized MoE |
| ROCm AITER | `rocm_aiter_fused_moe.py` | AMD ROCm AITER MoE |
| CPU | `cpu_fused_moe.py` | CPU MoE implementation |

### EPLB (Expert Parallel Load Balancing)

MoE layers can register with the EPLB system via `set_eplb_state()` which receives:
- `expert_load_view`: View of expert load metrics
- `logical_to_physical_map`: Mapping from logical to physical experts
- `logical_replica_count`: Replica count per logical expert

---

## 13. Quantization

**Source**: `vllm/model_executor/layers/quantization/`

### Supported Quantization Methods

| Method | Config Class | Description |
|--------|-------------|-------------|
| `"awq"` | `AWQConfig` | Activation-aware Weight Quantization |
| `"awq_marlin"` | `AWQMarlinConfig` | AWQ with Marlin kernel |
| `"fp8"` | `Fp8Config` | FP8 quantization |
| `"fbgemm_fp8"` | `FBGEMMFp8Config` | FBGEMM FP8 (deprecated) |
| `"fp_quant"` | `FPQuantConfig` | FP quantization (deprecated) |
| `"modelopt"` | `ModelOptFp8Config` | NVIDIA ModelOpt FP8 |
| `"modelopt_fp4"` | `ModelOptNvFp4Config` | NVIDIA ModelOpt FP4 |
| `"modelopt_mxfp8"` | `ModelOptMxFp8Config` | ModelOpt MX-FP8 |
| `"modelopt_mixed"` | `ModelOptMixedPrecisionConfig` | ModelOpt mixed precision |
| `"gguf"` | `GGUFConfig` | GGUF quantization |
| `"gptq"` | `GPTQConfig` | GPTQ quantization |
| `"gptq_marlin"` | `GPTQMarlinConfig` | GPTQ with Marlin kernel |
| `"gptq_marlin_24"` | - | GPTQ Marlin 2:4 sparsity |
| `"compressed-tensors"` | `CompressedTensorsConfig` | Neural Magic compressed tensors |
| `"bitsandbytes"` | `BitsAndBytesConfig` | BitsAndBytes 4/8-bit |
| `"experts_int8"` | `ExpertsInt8Config` | INT8 expert quantization |
| `"quark"` | `QuarkConfig` | AMD Quark quantization |
| `"moe_wna16"` | `MoeWNA16Config` | MoE Weight-Only INT8/INT4 with FP16 |
| `"torchao"` | `TorchAOConfig` | TorchAO quantization |
| `"inc"` / `"auto-round"` | `INCConfig` | Intel Neural Compressor |
| `"mxfp4"` | `Mxfp4Config` | MX-FP4 quantization |
| `"gpt_oss_mxfp4"` | `GptOssMxfp4Config` | GPT-OSS MX-FP4 |
| `"deepseek_v4_fp8"` | `DeepseekV4FP8Config` | DeepSeek V4 FP8 |
| `"cpu_awq"` | `CPUAWQConfig` | CPU AWQ quantization |
| `"humming"` | `HummingConfig` | Humming quantization |
| `"online"` | `OnlineQuantizationConfig` | Online/dynamic quantization |
| `"fp8_per_tensor"` | `OnlineQuantizationConfig` | Shorthand for online FP8 per-tensor |
| `"fp8_per_block"` | `OnlineQuantizationConfig` | Shorthand for online FP8 per-block |
| `"int8_per_channel_weight_only"` | `OnlineQuantizationConfig` | Shorthand for online INT8 |
| `"mxfp8"` | `OnlineQuantizationConfig` | Shorthand for online MX-FP8 |

### QuantizationConfig Base

```python
class QuantizationConfig:
    @classmethod
    def from_config(cls, config) -> QuantizationConfig
    
    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase | None
    
    def get_scaled_num_groups(self, grouping) -> int
    
    @staticmethod
    def get_config_filenames() -> list[str]
```

### QuantizeMethodBase

Abstract base for quantization methods applied to layers.

```python
class QuantizeMethodBase:
    def create_weights(self, layer, ...) -> None
    def apply(self, layer, x, bias) -> Tensor
    def process_weights_after_loading(self, layer) -> None
    def weight_loader(self, ...) -> Callable
```

### KV Cache Quantization

FP8 KV cache quantization is handled by `BaseKVCacheMethod` which manages `k_scale` and `v_scale` parameters:

- `QuantFP8`: Input quantization to FP8
- `kv_cache.py`: KV cache quantization methods

### Input Quantization

`input_quant_fp8.py` provides `QuantFP8` for quantizing attention inputs to FP8.

### Registering Custom Quantization

```python
@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    ...
```

---

## 14. Model Offloading

**Source**: `vllm/model_executor/offloader/`

### BaseOffloader

Abstract base class for weight offloading strategies.

```python
class BaseOffloader(ABC):
    def wrap_modules(self, modules_generator) -> list[nn.Module]
    def post_init(self)
    def sync_prev_onload(self) -> None
    def join_after_forward(self) -> None
    def _wait_for_layer(self, layer_idx) -> None
    def _start_prefetch(self, layer_idx) -> None
```

### NoopOffloader

No-op implementation that returns modules as-is.

### UVAOffloader

Unified Virtual Memory offloader. Uses CUDA UVA (Unified Virtual Addressing) to access CPU-allocated weights from GPU without explicit transfers. Weights are pinned for faster access.

### PrefetchOffloader

Layer-by-layer prefetching offloader that:
- Stores weights on CPU
- Asynchronously prefetches the next layer's weights to GPU
- Waits for prefetch completion before each layer's forward pass

### Utility Functions

#### `should_pin_memory()` -> `bool`

Checks if pinned memory should be used. Respects `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` env var. On unified-memory systems (e.g., GH200), pinned memory consumes GPU memory.

---

## 15. Kernel Warmup

**Source**: `vllm/model_executor/warmup/`

### kernel_warmup(worker)

Main entry point for kernel warmup:

1. **Deep GEMM warmup**: If `VLLM_USE_DEEP_GEMM` and Deep GEMM is supported
2. **FlashInfer autotune**: On Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs, runs FlashInfer autotuning benchmarks
3. **FlashInfer attention warmup**: Warm-up run with mixed prefill/decode batch

### flashinfer_autotune(runner)

Runs FlashInfer autotuning benchmarks to select optimal kernel implementations. Results are cached for future use.

### deep_gemm_warmup(model, max_tokens)

Warms up Deep GEMM kernels used in MoE expert computation.

---

## 16. Supported Model Architectures

### Text Generation Models

| HuggingFace Architecture | Module | vLLM Class |
|-------------------------|--------|------------|
| `LlamaForCausalLM` | `llama` | `LlamaForCausalLM` |
| `MistralForCausalLM` | `mistral` | `MistralForCausalLM` |
| `MixtralForCausalLM` | `mixtral` | `MixtralForCausalLM` |
| `Qwen2ForCausalLM` | `qwen2` | `Qwen2ForCausalLM` |
| `Qwen3ForCausalLM` | `qwen3` | `Qwen3ForCausalLM` |
| `Qwen3MoeForCausalLM` | `qwen3_moe` | `Qwen3MoeForCausalLM` |
| `DeepseekV2ForCausalLM` | `deepseek_v2` | `DeepseekV2ForCausalLM` |
| `DeepseekV3ForCausalLM` | `deepseek_v2` | `DeepseekV3ForCausalLM` |
| `DeepseekV4ForCausalLM` | `deepseek_v4` | `DeepseekV4ForCausalLM` |
| `GlmForCausalLM` | `glm` | `GlmForCausalLM` |
| `Glm4ForCausalLM` | `glm4` | `Glm4ForCausalLM` |
| `Glm4MoeForCausalLM` | `glm4_moe` | `Glm4MoeForCausalLM` |
| `Gemma2ForCausalLM` | `gemma2` | `Gemma2ForCausalLM` |
| `Gemma3ForCausalLM` | `gemma3` | `Gemma3ForCausalLM` |
| `Gemma4ForCausalLM` | `gemma4` | `Gemma4ForCausalLM` |
| `Phi3ForCausalLM` | `phi3` | `Phi3ForCausalLM` |
| `FalconForCausalLM` | `falcon` | `FalconForCausalLM` |
| `BloomForCausalLM` | `bloom` | `BloomForCausalLM` |
| `GPT2LMHeadModel` | `gpt2` | `GPT2LMHeadModel` |
| `OPTForCausalLM` | `opt` | `OPTForCausalLM` |
| `GPTNeoXForCausalLM` | `gpt_neox` | `GPTNeoXForCausalLM` |
| `JambaForCausalLM` | `jamba` | `JambaForCausalLM` |
| `MambaForCausalLM` | `mamba` | `MambaForCausalLM` |
| `ArcticForCausalLM` | `arctic` | `ArcticForCausalLM` |
| `DbrxForCausalLM` | `dbrx` | `DbrxForCausalLM` |
| `Llama4ForCausalLM` | `llama4` | `Llama4ForCausalLM` |
| `CohereForCausalLM` | `commandr` | `CohereForCausalLM` |
| `MiniMaxText01ForCausalLM` | `minimax_text_01` | `MiniMaxText01ForCausalLM` |
| `BambaForCausalLM` | `bamba` | `BambaForCausalLM` |

### Multimodal Models

| HuggingFace Architecture | Module | vLLM Class |
|-------------------------|--------|------------|
| `LlavaForConditionalGeneration` | `llava` | `LlavaForConditionalGeneration` |
| `LlavaNextForConditionalGeneration` | `llava_next` | `LlavaNextForConditionalGeneration` |
| `Qwen2VLForConditionalGeneration` | `qwen2_vl` | `Qwen2VLForConditionalGeneration` |
| `Qwen2_5_VLForConditionalGeneration` | `qwen2_5_vl` | `Qwen2_5_VLForConditionalGeneration` |
| `Qwen3VLForConditionalGeneration` | `qwen3_vl` | `Qwen3VLForConditionalGeneration` |
| `Gemma3ForConditionalGeneration` | `gemma3_mm` | `Gemma3ForConditionalGeneration` |
| `Gemma4ForConditionalGeneration` | `gemma4_mm` | `Gemma4ForConditionalGeneration` |
| `InternVLChatModel` | `internvl` | `InternVLChatModel` |
| `Phi3VForCausalLM` | `phi3v` | `Phi3VForCausalLM` |
| `PixtralForConditionalGeneration` | `pixtral` | `PixtralForConditionalGeneration` |
| `DeepseekVLV2ForCausalLM` | `deepseek_vl2` | `DeepseekVLV2ForCausalLM` |
| `WhisperForConditionalGeneration` | `whisper` | `WhisperForConditionalGeneration` |
| `MiniCPMO` | `minicpmo` | `MiniCPMO` |
| `ChameleonForConditionalGeneration` | `chameleon` | `ChameleonForConditionalGeneration` |

### Speculative Decoding Models

| Architecture | Module | Type |
|-------------|--------|------|
| `EagleLlamaForCausalLM` | `llama_eagle` | EAGLE-1/2 |
| `Eagle3LlamaForCausalLM` | `llama_eagle3` | EAGLE-3 |
| `DeepSeekMTPModel` | `deepseek_mtp` | Multi-Token Prediction |
| `MedusaModel` | `medusa` | Medusa |
| `MiMoMTPModel` | `mimo_mtp` | MTP |

### Architecture Aliasing

Many architectures are aliased to existing implementations:

| Alias | Target |
|-------|--------|
| `AquilaModel`, `AquilaForCausalLM`, `InternLMForCausalLM`, `InternLM3ForCausalLM`, `XverseForCausalLM`, `LLaMAForCausalLM` | `LlamaForCausalLM` |
| `Ministral3ForCausalLM` | `MistralForCausalLM` |
| `MPTForCausalLM`, `MptForCausalLM` | `MPTForCausalLM` |
| `RWForCausalLM` | `FalconForCausalLM` |

### Previously Supported Models (Removed)

| Architecture | Last Version |
|-------------|-------------|
| `MotifForCausalLM` | v0.10.2 |
| `Phi3SmallForCausalLM` | v0.9.2 |
| `Phi4FlashForCausalLM` | v0.10.2 |
| `MllamaForConditionalGeneration` | v0.10.2 |
| `DonutForConditionalGeneration` | v0.10.2 |
| `Phi4MultimodalForCausalLM` | v0.12.0 |

### Out-of-Tree Supported Models

| Architecture | Plugin URL |
|-------------|------------|
| `BartModel` | `github.com/vllm-project/bart-plugin` |
| `Florence2ForConditionalGeneration` | `github.com/vllm-project/bart-plugin` |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_LOG_MODEL_INSPECTION` | `0` | Log model structure on load |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` | `0` | Disable pinned memory for offloading |
| `VLLM_BATCH_INVARIANT` | `0` | Enable batch-invariant optimizations |
| `VLLM_USE_MODELSCOPE` | `0` | Download from ModelScope instead of HuggingFace |
| `VLLM_USE_DEEP_GEMM` | `0` | Enable Deep GEMM for MoE |
| `VLLM_DEEP_GEMM_WARMUP` | - | Deep GEMM warmup mode |
| `Q_SCALE_CONSTANT` | - | Q scale constant for FP8 attention |
| `K_SCALE_CONSTANT` | - | K scale constant for FP8 attention |
| `V_SCALE_CONSTANT` | - | V scale constant for FP8 attention |
