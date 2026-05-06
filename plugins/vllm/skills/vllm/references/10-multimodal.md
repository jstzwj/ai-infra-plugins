# vLLM Multimodal Processing Reference

This document provides comprehensive coverage of vLLM's multimodal processing pipeline,
including image, audio, and video processing, data classes, the registry system,
caching, encoder budget management, and integration with vision-language models.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Input Data Types and Type Aliases](#input-data-types-and-type-aliases)
3. [Multimodal Data Parsing](#multimodal-data-parsing)
4. [Multimodal Input Data Classes](#multimodal-input-data-classes)
5. [PlaceholderRange and Field Configuration](#placeholderrange-and-field-configuration)
6. [Processing Context](#processing-context)
7. [BaseMultiModalProcessor](#basemultimodalprocessor)
8. [Prompt Update System](#prompt-update-system)
9. [MultiModalRegistry](#multimodalregistry)
10. [Multimodal Cache System](#multimodal-cache-system)
11. [Encoder Budget Management](#encoder-budget-management)
12. [Media I/O System](#media-io-system)
13. [Image Processing](#image-processing)
14. [Audio Processing](#audio-processing)
15. [Video Processing](#video-processing)
16. [Multimodal Hashing](#multimodal-hashing)
17. [Utility Functions](#utility-functions)
18. [Engine Input Types](#engine-input-types)
19. [Dummy Inputs Builder](#dummy-inputs-builder)

---

## Architecture Overview

vLLM's multimodal system processes image, audio, and video inputs through a pipeline:

1. **API Layer**: Raw multimodal data arrives as `MultiModalDataDict` via the API
2. **Parsing**: `MultiModalDataParser` normalizes raw data into `MultiModalDataItems`
3. **Processing**: `BaseMultiModalProcessor` applies HuggingFace processors, generates
   prompt placeholders, and produces `MultiModalKwargsItems`
4. **Caching**: Processed items are cached using `BaseMultiModalProcessorCache` (P0 side)
   and `BaseMultiModalReceiverCache` (P1 side)
5. **Registry**: `MultiModalRegistry` dispatches processing per model architecture
6. **Budget**: `MultiModalBudget` computes encoder/decoder budgets based on model constraints

Key modules:
- `vllm/multimodal/inputs.py` - Data types and field configurations
- `vllm/multimodal/parse.py` - Data parsing and normalization
- `vllm/multimodal/processing/` - Processor framework
- `vllm/multimodal/registry.py` - Model dispatch registry
- `vllm/multimodal/cache.py` - Caching infrastructure
- `vllm/multimodal/encoder_budget.py` - Budget computation
- `vllm/multimodal/media/` - Media I/O connectors
- `vllm/multimodal/image.py` - Image utilities
- `vllm/multimodal/audio.py` - Audio utilities
- `vllm/multimodal/video.py` - Video utilities
- `vllm/multimodal/hasher.py` - Content hashing for caching

---

## Input Data Types and Type Aliases

### HfImageItem

```python
HfImageItem: TypeAlias = Union[Image, np.ndarray, torch.Tensor]
```

A `transformers.image_utils.ImageInput` representing a single image item,
which can be passed to a HuggingFace `ImageProcessor`.

### HfVideoItem

```python
HfVideoItem: TypeAlias = Union[
    list[Image], np.ndarray, torch.Tensor, list[np.ndarray], list[torch.Tensor]
]
```

A `transformers.image_utils.VideoInput` representing a single video item.

### HfAudioItem

```python
HfAudioItem: TypeAlias = Union[list[float], np.ndarray, torch.Tensor]
```

Represents a single audio item, which can be passed to a HuggingFace `AudioProcessor`.

### ImageItem

```python
ImageItem: TypeAlias = Union[HfImageItem, torch.Tensor, MediaWithBytes[HfImageItem]]
```

Extended image item that also accepts:
- A 3-D tensor or batch of 2-D tensors (treated as image embeddings, passed directly)
- `MediaWithBytes` wrapper coupling media with original bytes

### VideoItem

```python
VideoItem: TypeAlias = Union[HfVideoItem, torch.Tensor, tuple[HfVideoItem, dict[str, Any]]]
```

Extended video item that also accepts:
- A 3-D tensor or batch of 2-D tensors (treated as video embeddings)
- A tuple of `(video_data, metadata)` for videos with metadata

### AudioItem

```python
AudioItem: TypeAlias = Union[HfAudioItem, tuple[np.ndarray, float], torch.Tensor]
```

Extended audio item that also accepts:
- A tuple `(audio, sampling_rate)` for resampling to model's expected rate
- A 3-D tensor or batch of 2-D tensors (treated as audio embeddings)

### VisionChunkImage

```python
class VisionChunkImage(TypedDict):
    type: Literal["image"]
    image: Image
    uuid: str | None
```

Represents an image wrapped as a vision chunk.

### VisionChunkVideo

```python
class VisionChunkVideo(TypedDict):
    type: Literal["video_chunk"]
    video_chunk: list[Image]
    uuid: str | None
    prompt: str
    video_idx: int
```

Represents a video chunk with metadata.

### NestedTensors

```python
NestedTensors: TypeAlias = Union[
    list[NestedTensors],
    list[torch.Tensor],
    torch.Tensor,
    tuple[torch.Tensor, ...],
]
```

Uses a list instead of a tensor if the dimensions of each element do not match.

### BatchedTensorInputs

```python
BatchedTensorInputs: TypeAlias = dict[str, NestedTensors]
```

A dictionary containing nested tensors which have been batched via
`MultiModalKwargsItems.get_data`.

---

## Multimodal Data Parsing

### MultiModalDataParser

Location: `vllm/multimodal/parse.py`

```python
class MultiModalDataParser:
    def __init__(
        self,
        *,
        target_sr: float | None = None,
        target_channels: int | None = None,
        audio_resample_method: Literal["pyav", "scipy"] = "pyav",
        video_needs_metadata: bool = False,
        expected_hidden_size: int | None = None,
    ) -> None
```

Parses `MultiModalDataDict` into `MultiModalDataItems`.

**Parameters:**
- `target_sr` - Target sampling rate for automatic audio resampling
- `target_channels` - Target number of audio channels for normalization
- `audio_resample_method` - Method for audio resampling ("pyav" or "scipy")
- `video_needs_metadata` - Whether video metadata is required
- `expected_hidden_size` - Expected hidden dimension for embedding validation

**Key Methods:**

#### `is_embeddings(data: object) -> TypeGuard[torch.Tensor | list[torch.Tensor]]`

```python
@classmethod
def is_embeddings(cls, data: object) -> TypeGuard[torch.Tensor | list[torch.Tensor]]
```

Returns True if the data represents pre-computed embeddings:
- A single 3D tensor `(batch, seq_len, hidden_size)`
- A list of 2D tensors `[(seq_len, hidden_size), ...]`

#### `parse_mm_data(mm_data: MultiModalDataDict) -> MultiModalDataItems`

```python
def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems
```

Main entry point: converts raw multimodal data dict into structured items.
Handles audio resampling, channel normalization, and type detection.

### ModalityDataItems (Base Class)

```python
class ModalityDataItems(ABC, Generic[_T, _I]):
    def __init__(self, data: _T, modality: str) -> None
```

Abstract base for data items of a single modality.

**Abstract Methods:**
- `get_count() -> int` - Number of data items
- `get(index: int) -> _I` - Get item by index
- `get_processor_data() -> Mapping[str, object]` - Data for HF processor
- `get_passthrough_data() -> Mapping[str, object]` - Data passed directly to model

**Concrete Methods:**
- `get_all() -> list[_I]` - Get all data items
- `get_item_for_hash(index: int) -> object` - Get item for hashing (preserves MediaWithBytes)
- `get_all_items_for_hash() -> list[object]` - Get all items for hashing

### ProcessorBatchItems

```python
class ProcessorBatchItems(ModalityDataItems[Sequence[_T], _T]):
```

Base class for data items arranged in a list. Automatically unwraps `MediaWithBytes` wrappers.

**Subclasses:**

#### `AudioProcessorItems`
```python
class AudioProcessorItems(ProcessorBatchItems[HfAudioItem | None]):
    def __init__(self, data: Sequence[HfAudioItem | None]) -> None
    def get_audio_length(self, item_idx: int) -> int
```

#### `ImageProcessorItems`
```python
class ImageProcessorItems(ProcessorBatchItems[HfImageItem | None]):
    def __init__(self, data: Sequence[HfImageItem | None]) -> None
    def get_image_size(self, item_idx: int) -> ImageSize
```

Returns `ImageSize(width: int, height: int)` NamedTuple.

#### `VideoProcessorItems`
```python
class VideoProcessorItems(ProcessorBatchItems[HfVideoItem | None]):
    def __init__(
        self,
        data: Sequence[HfVideoItem | None],
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> None
    def get_num_frames(self, item_idx: int) -> int
    def get_frame_size(self, item_idx: int) -> ImageSize
```

#### `VisionChunkProcessorItems`
```python
class VisionChunkProcessorItems(ProcessorBatchItems[Any]):
    def __init__(self, data: Sequence[Any]) -> None
```

### EmbeddingItems

```python
class EmbeddingItems(ModalityDataItems[torch.Tensor | list[torch.Tensor], torch.Tensor]):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        modality: str,
        expected_hidden_size: int | None = None,
    ) -> None
```

Base class for data items expressed as embedding tensors. Validates ndim (2D or 3D)
and optionally hidden dimension size.

**Subclasses:**
- `AudioEmbeddingItems(EmbeddingItems)` - modality="audio"
- `ImageEmbeddingItems(EmbeddingItems)` - modality="image"
- `VideoEmbeddingItems(EmbeddingItems)` - modality="video"

**Methods:**
- `get_feature_size(item_idx: int) -> int` - Returns sequence length of embedding
- `get_processor_data() -> Mapping[str, object]` - Returns `{}`
- `get_passthrough_data() -> Mapping[str, object]` - Returns `{modality}_embeds: data`

### DictEmbeddingItems

```python
class DictEmbeddingItems(
    ModalityDataItems[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        modality: str,
        required_fields: set[str],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None
```

Data items expressed as a dictionary of tensors (matching HF processor output format).

### MultiModalDataItems

```python
class MultiModalDataItems(UserDict[str, ModalityDataItems[Any, Any]]):
```

Normalized `MultiModalDataDict` where each entry corresponds to typed items.

**Methods:**

```python
def select(self, modalities: Set[str]) -> MultiModalDataItems
```
Construct new instance containing only selected modalities.

```python
def get_count(self, modality: str, *, strict: bool = True) -> int
```
Get number of items for a modality. Returns 0 if `strict=False` and modality not found.

```python
def get_all_counts(self) -> Mapping[str, int]
```
Get item counts for all modalities.

```python
def get_items(
    self, modality: str, typ: type[_D] | tuple[type[_D], ...]
) -> _D
```
Get items for a modality, requiring they belong to a certain type.

---

## Multimodal Input Data Classes

### PlaceholderRange

```python
@dataclass(frozen=True)
class PlaceholderRange:
    offset: int
    """The start index of the placeholder in the prompt."""
    length: int
    """The length of the placeholder."""
    is_embed: torch.Tensor | None = None
    """Boolean mask of shape (length,) indicating which positions get embeddings."""
```

Tracks multimodal placeholder positions in the prompt token sequence.

**Properties and Methods:**

```python
@property
def embeds_cumsum(self) -> list[int] | None
```
Cumulative sum of `is_embed` mask for efficient index computation.

```python
def get_num_embeds(self) -> int
```
Number of embedding positions (accounting for `is_embed` mask).

```python
def get_embeds_indices_in_range(
    self, start_idx: int, end_idx: int
) -> tuple[int, int]
```
Returns start/end indices of encoder output embeddings within the given range.

```python
def extract_embeds_range(self) -> list[tuple[int, int]]
```
Extract start/end index pairs of embedded regions in the prompt.

### MultiModalFeatureSpec

```python
@dataclass
class MultiModalFeatureSpec:
    data: MultiModalKwargsItem | None
    """Processed multimodal data; None if cached (skip IPC)."""
    modality: str
    """Input modality: "image", "audio", "video"."""
    identifier: str
    """Hash for caching encoder outputs (with LoRA prefix)."""
    mm_position: PlaceholderRange
    """Location of modality tokens in the prompt."""
    mm_hash: str | None = None
    """Hash for caching processor outputs (without LoRA prefix)."""
```

Represents a single multimodal input with processed data and metadata.

```python
@staticmethod
def gather_kwargs(
    features: list[MultiModalFeatureSpec],
    keys: set[str],
) -> dict[str, list[NestedTensors]]
```
Gather specified kwargs from a list of features.

### MultiModalFieldElem

```python
@dataclass
class MultiModalFieldElem:
    data: NestedTensors
    """Tensor data of this field."""
    field: BaseMultiModalField
    """Defines how to combine with other fields for batching."""
```

Represents a processed keyword argument to pass to a model for a single item.

### MultiModalKwargsItem

```python
class MultiModalKwargsItem(UserDict[str, MultiModalFieldElem]):
```

Dictionary of processed keyword arguments for a single multimodal item.

**Methods:**

```python
@staticmethod
def dummy(nbytes: int = 1) -> MultiModalKwargsItem
```
Create a dummy item for testing.

```python
def get_data(self) -> dict[str, NestedTensors]
```
Extract raw tensor data from all fields.

### MultiModalKwargsItems

```python
class MultiModalKwargsItems(UserDict[str, Sequence[_I]]):
```

Dictionary of processed multimodal inputs organized by modality.

Example structure:
```python
MultiModalKwargsItems({
    "image": [
        MultiModalKwargsItem({"pixel_values": ..., "image_grid_thw": ...}),
        MultiModalKwargsItem({"pixel_values": ..., "image_grid_thw": ...}),
    ],
    "audio": [
        MultiModalKwargsItem({"input_audio_features": ...}),
    ],
})
```

**Methods:**

```python
@staticmethod
def from_hf_inputs(
    hf_inputs: BatchFeature,
    config_by_key: Mapping[str, MultiModalFieldConfig],
) -> MultiModalKwargsItems
```
Construct from HuggingFace processor output using field configurations.

```python
def require_data(self) -> MultiModalKwargsItems[MultiModalKwargsItem]
```
Verify all items have data (not None). Raises RuntimeError otherwise.

```python
def get_data(
    self,
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> BatchedTensorInputs
```
Construct batched dictionary of keyword arguments for model inference.

---

## PlaceholderRange and Field Configuration

### BaseMultiModalField

```python
@dataclass(frozen=True, kw_only=True)
class BaseMultiModalField(ABC):
    keep_on_cpu: bool = False
```

Defines how to interpret tensor data for keyword arguments.

**Abstract Methods:**
```python
@abstractmethod
def build_elems(
    self, modality: str, key: str, data: NestedTensors
) -> Sequence[MultiModalFieldElem]

@abstractmethod
def _reduce_data(
    self, batch: list[NestedTensors], *, pin_memory: bool
) -> NestedTensors
```

**Concrete Method:**
```python
def reduce_data(
    self,
    elems: list[MultiModalFieldElem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> NestedTensors
```
Merge data from multiple `MultiModalFieldElem` instances.

### MultiModalBatchedField

```python
@dataclass(frozen=True, kw_only=True)
class MultiModalBatchedField(BaseMultiModalField):
```

Each element is obtained by indexing into the first dimension. Uses `torch.stack` for reduction.

Example:
```
Input:  [[AAAA], [BBBB], [CCCC]]
Output: Element 1: [AAAA], Element 2: [BBBB], Element 3: [CCCC]
```

### MultiModalFlatField

```python
@dataclass(frozen=True, kw_only=True)
class MultiModalFlatField(BaseMultiModalField):
    slices: Sequence[slice] | Sequence[Sequence[slice]]
    dim: int = 0
```

Each element is obtained by slicing along a dimension. Uses `torch.concat` for reduction.

Supports variable-length tensors by zero-padding and slice-assigning.

### MultiModalSharedField

```python
@dataclass(frozen=True, kw_only=True)
class MultiModalSharedField(BaseMultiModalField):
    batch_size: int
```

All elements share the same data (no splitting). Returns first element on reduction.

### MultiModalFieldConfig

```python
@dataclass(frozen=True)
class MultiModalFieldConfig:
    field: BaseMultiModalField
    modality: str
```

Configuration factory for field types.

**Static Factory Methods:**

```python
@staticmethod
def batched(modality: str, *, keep_on_cpu: bool = False) -> MultiModalFieldConfig
```
Defines a field where elements are obtained by first-dimension indexing.

```python
@staticmethod
def flat(
    modality: str,
    slices: Sequence[slice] | Sequence[Sequence[slice]],
    dim: int = 0,
    *,
    keep_on_cpu: bool = False,
) -> MultiModalFieldConfig
```
Defines a field where elements are obtained by slicing.

```python
@staticmethod
def flat_from_sizes(
    modality: str,
    size_per_item: torch.Tensor,
    dim: int = 0,
    *,
    keep_on_cpu: bool = False,
) -> MultiModalFieldConfig
```
Like `flat` but computes slices from per-item sizes.

```python
@staticmethod
def shared(
    modality: str,
    batch_size: int,
    *,
    keep_on_cpu: bool = False,
) -> MultiModalFieldConfig
```
Defines a field where all elements share the same data.

---

## Processing Context

### TimingContext

```python
@dataclass
class TimingContext:
    enabled: bool = True
    stage_secs: dict[str, float] = field(default_factory=dict)
```

Records execution times during multi-modal processing.

**Methods:**
```python
@property
def total_secs(self) -> float
```

```python
@contextmanager
def record(self, stage: str)
```
Record execution time for a processing stage.

```python
def get_stats_dict(self) -> dict[str, float]
```

### InputProcessingContext

```python
@dataclass(frozen=True)
class InputProcessingContext:
    model_config: ModelConfig
    tokenizer: TokenizerLike | None
```

Contains model information used to modify inputs.

**Methods:**

```python
def get_tokenizer(self) -> TokenizerLike
```

```python
@overload
def get_hf_config(self, /) -> PretrainedConfig: ...
@overload
def get_hf_config(self, typ: type[_C] | tuple[type[_C], ...], /) -> _C: ...
```

```python
def get_hf_image_processor_config(self) -> dict[str, Any]
```

```python
def get_mm_config(self)
```

```python
@overload
def get_hf_processor(self, /, **kwargs: object) -> ProcessorMixin: ...
@overload
def get_hf_processor(
    self, typ: type[_P] | tuple[type[_P], ...], /, **kwargs: object
) -> _P: ...
```

```python
def init_processor(self, typ: type[_T], /, **kwargs: object) -> _T
```

```python
def call_hf_processor(
    self,
    hf_processor: Callable[..., BatchFeature] | ProcessorMixin,
    data: Mapping[str, object],
    kwargs: Mapping[str, object] = {},
    *,
    num_tries: int = 1,
    max_tries: int = 5,
) -> BatchFeature
```
Call HF processor with configurable options, converting output to model dtype.

### BaseProcessingInfo

```python
class BaseProcessingInfo:
    def __init__(self, ctx: InputProcessingContext) -> None
```

Base class providing information necessary for data processing.

**Properties:**
- `model_id: str` - Model identifier
- `supported_mm_limits: Mapping[str, int | None]` - Max items per modality
- `allowed_mm_limits: Mapping[str, int]` - User-constrained limits
- `skip_prompt_length_check: bool` - Whether to skip length checks
- `default_tok_params: TokenizeParams` - Default tokenization parameters

**Methods:**

```python
@abstractmethod
def get_supported_mm_limits(self) -> Mapping[str, int | None]
```
Return maximum supported number of items per modality. `None` means unlimited.

```python
def validate_num_items(self, modality: str, num_items: int) -> None
```
Raise `ValueError` if item count exceeds limits.

```python
def parse_mm_data(
    self, mm_data: MultiModalDataDict, *, validate: bool = True
) -> MultiModalDataItems
```
Normalize `MultiModalDataDict` to `MultiModalDataItems`.

```python
def get_mm_max_tokens_per_item(
    self, seq_len: int, mm_counts: Mapping[str, int]
) -> Mapping[str, int] | None
```
Return maximum tokens per item for each modality. Override for faster startup.

---

## BaseMultiModalProcessor

Location: `vllm/multimodal/processing/processor.py`

```python
class BaseMultiModalProcessor(ABC, Generic[_I]):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None
```

Abstract base class to process multimodal inputs for vLLM.

**Key Abstract Methods:**

```python
@abstractmethod
def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]
```
Given HF-processed data, output metadata of each field.

```python
@abstractmethod
def _get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    out_mm_kwargs: MultiModalKwargsItems,
) -> Sequence[PromptUpdate]
```
Given original multimodal items and HF-processed data, output prompt updates.

**Main Entry Point:**

```python
def apply(
    self,
    inputs: ProcessorInputs,
    timing_ctx: TimingContext,
) -> MultiModalInput
```
Process multimodal inputs. Steps:
1. Apply HF Processor on prompt text and multimodal data
2. Find and update token sequences with placeholder tokens
3. Extract placeholder information from processed tokens

```python
def __call__(
    self,
    prompt: str,
    mm_items: MultiModalDataItems,
    mm_uuid_items: MultiModalUUIDItems | None = None,
    hf_processor_mm_kwargs: Mapping[str, object] | None = None,
) -> MultiModalInput
```
Convenience method that wraps `apply()`.

**HF Processor Methods:**

```python
def _call_hf_processor(
    self,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
    tok_kwargs: Mapping[str, object],
) -> BatchFeature
```

```python
def _apply_hf_processor_text_mm(
    self,
    prompt_text: str,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    tokenization_kwargs: Mapping[str, object],
) -> tuple[list[int], BatchFeature, bool]
```
Apply HF processor on prompt text + multimodal data together.

```python
def _apply_hf_processor_text_only(
    self,
    prompt_text: str,
    tokenization_kwargs: Mapping[str, object],
) -> list[int]
```
Apply HF processor on text only (creates dummy MM data).

```python
def _apply_hf_processor_mm_only(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    tokenization_kwargs: Mapping[str, object],
) -> BatchFeature
```
Apply HF processor on multimodal data only (generates dummy text).

### EncDecMultiModalProcessor

```python
class EncDecMultiModalProcessor(BaseMultiModalProcessor[_I]):
    skip_decoder_start_token: bool = False
```

Processor variant for encoder-decoder models.

**Abstract Method:**
```python
@abstractmethod
def create_encoder_prompt(
    self,
    prompt: str | list[int],
    mm_items: MultiModalDataItems,
) -> str | list[int]
```

**Method:**
```python
def create_decoder_prompt(
    self,
    prompt: str | list[int],
    mm_items: MultiModalDataItems,
) -> str | list[int]
```

---

## Prompt Update System

### PromptUpdate

```python
@dataclass
class PromptUpdate(ABC):
    modality: str
    target: PromptUpdateTarget
```

Defines how to update a prompt with placeholder tokens.

**Properties:**
```python
@property
@abstractmethod
def content(self) -> PromptUpdateContent

@property
@abstractmethod
def mode(self) -> UpdateMode
```

```python
def resolve(self, item_idx: int) -> ResolvedPromptUpdate
```

### PromptInsertion

```python
@dataclass
class PromptInsertion(PromptUpdate):
    insertion: PromptUpdateContent = field(repr=False)
```

Inserts placeholder tokens after `target`.

`mode` is always `UpdateMode.INSERT`.

### PromptReplacement

```python
@dataclass
class PromptReplacement(PromptUpdate):
    replacement: PromptUpdateContent = field(repr=False)
```

Replaces occurrences of `target` with placeholder tokens.

`mode` is always `UpdateMode.REPLACE`.

### PromptIndexTargets

```python
class PromptIndexTargets:
    @staticmethod
    def start() -> PromptIndex
    """Resolves to start of prompt."""

    @staticmethod
    def prefix(seq: PromptSeq) -> PromptIndex
    """Resolves to location after the given prefix."""

    @staticmethod
    def end() -> PromptIndex
    """Resolves to end of prompt."""
```

### PromptUpdateDetails

```python
@dataclass
class PromptUpdateDetails(Generic[_S]):
    full: _S
    """The full content."""
    is_embed: Callable[[TokenizerLike | None, PromptSeq], torch.Tensor] | None = None
    """Boolean mask indicating which positions get embeddings."""
```

**Static Factory Methods:**
```python
@staticmethod
def from_seq(seq: _S) -> PromptUpdateDetails[_S]

@staticmethod
def select_text(seq: _S, embed_text: str) -> PromptUpdateDetails[_S]

@staticmethod
def select_token_id(seq: _S, embed_token_id: int) -> PromptUpdateDetails[_S]

@staticmethod
def select_token_ids(seq: _S, embed_token_ids: list[int]) -> PromptUpdateDetails[_S]
```

### ResolvedPromptUpdate

```python
@dataclass(frozen=True)
class ResolvedPromptUpdate:
    modality: str
    item_idx: int
    mode: UpdateMode
    target: UpdateTarget
    content: PromptUpdateDetails = field(repr=False)
```

A `PromptUpdate` with lazy attributes resolved.

**Methods:**
```python
def iter_token_matches(
    self, prompt: list[int], tokenizer: TokenizerLike | None, *, start_idx: int = 0
) -> Generator[PromptTargetMatch]

def iter_text_matches(
    self, prompt: str, tokenizer: TokenizerLike | None, *, start_idx: int = 0
) -> Generator[PromptTargetMatch]

def iter_matches(
    self, prompt: list[int] | str, tokenizer: TokenizerLike | None, *, start_idx: int = 0
) -> Generator[PromptTargetMatch]
```

### Helper Functions

```python
def iter_token_matches(
    token_ids: list[int],
    match_ids: list[int],
    *,
    start_idx: int = 0,
) -> Generator[_TokenMatch]
```
Yield each occurrence of `match_ids` in `token_ids`.

```python
def replace_token_matches(
    token_ids: list[int],
    match_ids: list[int],
    new_ids: list[int],
) -> list[int]
```
Replace each occurrence of `match_ids` with `new_ids`.

```python
def apply_token_matches(
    prompt: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> tuple[list[int], MultiModalPromptUpdatesApplyResult]

def apply_text_matches(
    prompt: str,
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> tuple[str, MultiModalPromptUpdatesApplyResult]
```

```python
def find_mm_placeholders(
    prompt: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> Mapping[str, list[PlaceholderFeaturesInfo]]
```

### Type Aliases

```python
PromptSeq: TypeAlias = str | list[int]
UpdateTarget: TypeAlias = PromptSeq | PromptIndex
PromptUpdateTarget: TypeAlias = Callable[[int], UpdateTarget] | UpdateTarget
PromptUpdateInfo: TypeAlias = PromptSeq | PromptUpdateDetails
PromptUpdateContent: TypeAlias = Callable[[int], PromptUpdateInfo] | PromptUpdateInfo
MultiModalIsCached: TypeAlias = dict[str, list[bool]]
MultiModalPromptUpdates: TypeAlias = Mapping[str, list[Sequence[ResolvedPromptUpdate]]]
MultiModalPromptUpdatesApplyResult: TypeAlias = Mapping[str, list[int | None]]
```

---

## MultiModalRegistry

Location: `vllm/multimodal/registry.py`

```python
class MultiModalRegistry:
```

A registry that dispatches data processing according to the model.

**Methods:**

```python
def supports_multimodal_inputs(self, model_config: ModelConfig) -> bool
```
Check if the model supports multimodal inputs (any modality with non-zero limit).

```python
def register_processor(
    self,
    processor: MultiModalProcessorFactory[_I],
    *,
    info: ProcessingInfoFactory[_I],
    dummy_inputs: DummyInputsBuilderFactory[_I],
) -> Callable[[N], N]
```
Register a multi-modal processor to a model class. Returns a decorator.

```python
def create_processor(
    self,
    model_config: ModelConfig,
    *,
    tokenizer: TokenizerLike | None = None,
    cache: BaseMultiModalProcessorCache | None = None,
) -> BaseMultiModalProcessor[BaseProcessingInfo]
```
Create a multi-modal processor for a specific model and tokenizer.

```python
def get_processing_info(
    self, model_config: ModelConfig
) -> BaseProcessingInfo
```

```python
def get_dummy_mm_inputs(
    self,
    model_config: ModelConfig,
    mm_counts: Mapping[str, int],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
    processor: BaseMultiModalProcessor | None = None,
) -> MultiModalInput
```
Create dummy data for profiling memory usage.

```python
def processor_cache_from_config(
    self, vllm_config: VllmConfig
) -> BaseMultiModalProcessorCache | None
```
Return appropriate processor cache based on configuration.

```python
def engine_receiver_cache_from_config(
    self, vllm_config: VllmConfig
) -> BaseMultiModalReceiverCache | None
```

```python
def worker_receiver_cache_from_config(
    self, vllm_config: VllmConfig, shared_worker_lock: LockType
) -> BaseMultiModalReceiverCache | None
```

### MultiModalTimingRegistry

```python
class MultiModalTimingRegistry:
    def __init__(self, observability_config: ObservabilityConfig | None) -> None
    def get(self, request_id: str) -> TimingContext
    def stat(self) -> dict[str, dict[str, float]]
```

### Protocol Types

```python
class ProcessingInfoFactory(Protocol[_I_co]):
    def __call__(self, ctx: InputProcessingContext) -> _I_co: ...

class DummyInputsBuilderFactory(Protocol[_I]):
    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]: ...

class MultiModalProcessorFactory(Protocol[_I]):
    def __call__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[_I]: ...
```

### Global Instance

```python
MULTIMODAL_REGISTRY = MultiModalRegistry()
```

---

## Multimodal Cache System

Location: `vllm/multimodal/cache.py`

### Cache Architecture

The caching system uses a client-server model:
- **P0** (frontend/API process): Sender cache
- **P1** (core/worker process): Receiver cache

```
              is_cached() x N    get_and_update()
P0: From API -----------------> -----------------> To P1

             get_and_update()
P1: From P0 -----------------> To model
```

### MultiModalCache

```python
class MultiModalCache:
```

Utility class for computing cache item sizes.

**Class Methods:**

```python
@classmethod
def get_leaf_size(cls, leaf: object) -> int
```
Get byte size of a single leaf value (tensor, array, etc.).

```python
@classmethod
def get_item_size(
    cls, value: MultiModalCacheValue, *, debug: bool = False
) -> int
```
Compute total byte size of a cache value.

```python
@classmethod
def get_item_complexity(cls, value: MultiModalCacheValue) -> int
```
Count leaf elements in a cache value.

```python
@classmethod
def get_lru_cache(
    cls,
    capacity_gb: float,
    value_type: type[_V],
    *,
    debug: bool = False,
) -> LRUCache[str, _V]
```
Create an LRU cache with capacity in GiB.

### BaseMultiModalCache

```python
class BaseMultiModalCache(ABC, Generic[_I, _O]):
```

Abstract base class for reading/writing multimodal items from cache.

**Abstract Methods:**
```python
@abstractmethod
def get_and_update_item(self, mm_item: _I, mm_hash: str) -> _O

@abstractmethod
def clear_cache(self) -> None
```

**Concrete Methods:**
```python
def get_and_update(
    self, mm_items: Sequence[_I], mm_hashes: list[str]
) -> list[_O]
```

### BaseMultiModalProcessorCache

```python
class BaseMultiModalProcessorCache(
    BaseMultiModalCache[MultiModalProcessorCacheInItem, MultiModalProcessorCacheOutItem]
):
```

Required interface for caches on P0 (frontend).

**Methods:**

```python
@abstractmethod
def is_cached_item(self, mm_hash: str) -> bool

def is_cached(self, mm_hashes: list[str]) -> list[bool]

@abstractmethod
def touch_sender_cache_item(self, mm_hash: str) -> None

@abstractmethod
def make_stats(self, *, delta: bool = False) -> CacheInfo
```

### MultiModalProcessorOnlyCache

```python
class MultiModalProcessorOnlyCache(BaseMultiModalProcessorCache):
    def __init__(self, model_config: ModelConfig) -> None
```

Cache used on P0 when IPC caching is disabled. Stores full item data.

### MultiModalProcessorSenderCache

```python
class MultiModalProcessorSenderCache(BaseMultiModalProcessorCache):
    def __init__(self, model_config: ModelConfig) -> None
```

Cache used on P0 when IPC caching is enabled. Stores only metadata (not tensor data).

### ShmObjectStoreSenderCache

```python
class ShmObjectStoreSenderCache(BaseMultiModalProcessorCache):
    def __init__(self, vllm_config: VllmConfig) -> None
```

Cache used on P0 with shared memory IPC. Stores data in shared memory ring buffer.

**Methods:**
```python
def remove_dangling_items(self) -> None

def address_as_item(
    self, address: int, monotonic_id: int
) -> MultiModalKwargsItem
```

### BaseMultiModalReceiverCache

```python
class BaseMultiModalReceiverCache(
    BaseMultiModalCache[MultiModalKwargsItem | None, MultiModalKwargsItem]
):
```

Required interface for caches on P1 (core/worker).

**Methods:**
```python
def get_and_update_features(
    self, mm_features: list[MultiModalFeatureSpec]
) -> list[MultiModalFeatureSpec]

@abstractmethod
def touch_receiver_cache_item(
    self, mm_hash: str, mm_item: MultiModalKwargsItem | None = None
) -> None
```

### MultiModalReceiverCache

```python
class MultiModalReceiverCache(BaseMultiModalReceiverCache):
    def __init__(self, model_config: ModelConfig) -> None
```

Cache on P1 when IPC caching enabled. Stores full item data.

### ShmObjectStoreReceiverCache

```python
class ShmObjectStoreReceiverCache(BaseMultiModalReceiverCache):
    def __init__(
        self, vllm_config: VllmConfig, shared_worker_lock: LockType
    ) -> None
```

Cache on P1 worker process using shared memory. Reads data from shared memory ring buffer.

### Cache Item Types

```python
class MultiModalProcessorCacheItem:
    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence[ResolvedPromptUpdate],
    ) -> None

class MultiModalProcessorCacheItemMetadata:
    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence[ResolvedPromptUpdate],
    ) -> None
    # Only stores item_size, not actual data
```

---

## Encoder Budget Management

Location: `vllm/multimodal/encoder_budget.py`

### get_mm_max_toks_per_item

```python
def get_mm_max_toks_per_item(
    model_config: ModelConfig,
    mm_registry: MultiModalRegistry,
    processor: BaseMultiModalProcessor,
    mm_counts: Mapping[str, int],
) -> Mapping[str, int]
```

Get maximum tokens per data item from each modality. Uses processor's
`get_mm_max_tokens_per_item` if available, otherwise generates dummy inputs.

### MultiModalBudget

```python
class MultiModalBudget:
    def __init__(
        self,
        vllm_config: VllmConfig,
        mm_registry: MultiModalRegistry,
    ) -> None
```

Helper class to calculate budget information for multimodal models.

**Attributes:**
- `max_model_len: int` - Maximum model sequence length
- `max_num_reqs: int` - Maximum number of sequences
- `encoder_compute_budget: int` - Encoder compute budget (tokens)
- `encoder_cache_size: int` - Encoder cache size (tokens)
- `mm_max_toks_per_item: Mapping[str, int]` - Max tokens per item per modality
- `mm_max_items_per_prompt: Mapping[str, int]` - Max items per prompt
- `mm_max_items_per_batch: Mapping[str, int]` - Max items per batch

**Methods:**

```python
def get_modality_with_max_tokens(self) -> str
```
Get the modality that has the highest token count per item.

```python
def get_encoder_budget(self) -> int
```
Get effective encoder budget (min of compute budget and cache size).

```python
def reset_cache(self) -> None
```
Clear the processor cache.

---

## Media I/O System

Location: `vllm/multimodal/media/`

### MediaWithBytes

```python
@dataclass
class MediaWithBytes(Generic[_T]):
    media: _T
    original_bytes: bytes = field(repr=False)
```

Wrapper coupling media with its original encoded bytes. Prevents cache corruption
from in-place modifications. Delegates attribute access to the underlying media.

### MediaIO

```python
class MediaIO(ABC, Generic[_T]):
```

Abstract base for media I/O operations.

**Class Methods:**
```python
@classmethod
def merge_kwargs(
    cls,
    default_kwargs: dict[str, Any] | None,
    runtime_kwargs: dict[str, Any] | None,
) -> dict[str, Any]
```
Merge config-level and request-level kwargs.

**Abstract Methods:**
```python
@abstractmethod
def load_bytes(self, data: bytes) -> _T

@abstractmethod
def load_base64(self, media_type: str, data: str) -> _T

@abstractmethod
def load_file(self, filepath: Path) -> _T
```

### ImageMediaIO

```python
class ImageMediaIO(MediaIO[Image.Image]):
    def __init__(self, image_mode: str = "RGB", **kwargs) -> None
```

**Parameters:**
- `image_mode` - Target image mode (default: "RGB")
- `rgba_background_color` - Background for RGBA to RGB conversion (default: `(255,255,255)`)

**Methods:**
```python
def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]
def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]
def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]
def encode_base64(self, media: Image.Image, *, image_format: str = "PNG") -> str
```

### ImageEmbeddingMediaIO

```python
class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
```

Handles image embedding tensors. Supports pickle, numpy, and torch formats.

### AudioMediaIO

```python
class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    def __init__(self, **kwargs) -> None
```

**Methods:**
```python
def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]
def load_base64(self, media_type: str, data: str) -> tuple[npt.NDArray, float]
def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]
def encode_base64(
    self, media: tuple[npt.NDArray, int], *, audio_format: str = "WAV"
) -> str
```

### AudioEmbeddingMediaIO

```python
class AudioEmbeddingMediaIO(MediaIO[torch.Tensor]):
```

### VideoMediaIO

```python
class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None
```

**Parameters:**
- `image_io` - ImageMediaIO for frame encoding/decoding
- `num_frames` - Default number of frames to sample
- `video_backend` - Backend override (from kwargs or env)

**Key Method:**
```python
def load_base64(
    self, media_type: str, data: str
) -> tuple[npt.NDArray, dict[str, Any]]
```
Supports `video/jpeg` media type with comma-separated frames.

### MediaConnector

```python
@MEDIA_CONNECTOR_REGISTRY.register("http")
class MediaConnector:
    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        connection: HTTPConnection = global_http_connection,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None
```

**Fetch Methods:**
```python
def fetch_audio(self, audio_url: str) -> tuple[np.ndarray, int | float]
async def fetch_audio_async(self, audio_url: str) -> tuple[np.ndarray, int | float]

def fetch_image(self, image_url: str, *, image_mode: str = "RGB") -> Image.Image
async def fetch_image_async(self, image_url: str, *, image_mode: str = "RGB") -> Image.Image

def fetch_video(self, video_url: str, *, image_mode: str = "RGB") -> tuple[npt.NDArray, dict[str, Any]]
async def fetch_video_async(self, video_url: str, *, image_mode: str = "RGB") -> tuple[npt.NDArray, dict[str, Any]]
```

**Cache Features:**
- Configurable via `VLLM_MEDIA_CACHE` environment variable
- LRU eviction policy
- TTL-based expiration
- Atomic writes for cache integrity

---

## Image Processing

Location: `vllm/multimodal/image.py`

```python
def rescale_image_size(
    image: Image.Image, size_factor: float, transpose: int = -1
) -> Image.Image
```
Rescale image dimensions by a constant factor.

```python
def rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] | list[int] = (255, 255, 255),
) -> Image.Image
```
Convert RGBA to RGB with filled background color.

```python
def convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image
```
Convert image to target mode, handling RGBA->RGB specially.

---

## Audio Processing

Location: `vllm/multimodal/audio.py`

### AudioSpec

```python
@dataclass
class AudioSpec:
    target_channels: int | None = 1
    channel_reduction: ChannelReduction = ChannelReduction.MEAN
```

**Predefined Specs:**
```python
MONO_AUDIO_SPEC = AudioSpec(target_channels=1, channel_reduction=ChannelReduction.MEAN)
PASSTHROUGH_AUDIO_SPEC = AudioSpec(target_channels=None)
```

### ChannelReduction

```python
class ChannelReduction(str, Enum):
    MEAN = "mean"
    FIRST = "first"
    MAX = "max"
    SUM = "sum"
```

### Audio Functions

```python
def get_audio_duration(*, y: npt.NDArray[np.floating], sr: float = 22050) -> float
```

```python
def normalize_audio(
    audio: npt.NDArray[np.floating] | torch.Tensor,
    spec: AudioSpec,
) -> npt.NDArray[np.floating] | torch.Tensor
```
Normalize audio to target channel count. Handles 1D (mono), 2D (channels, time),
and auto-detects (time, channels) format.

### AudioResampler

```python
class AudioResampler:
    def __init__(
        self,
        target_sr: float | None = None,
        method: Literal["pyav", "scipy"] = "pyav",
    ) -> None

    def resample(
        self, audio: npt.NDArray[np.floating], *, orig_sr: float
    ) -> npt.NDArray[np.floating]
```

### Audio Splitting

```python
def split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    max_clip_duration_s: float,
    overlap_duration_s: float,
    min_energy_window_size: int,
) -> list[np.ndarray]
```
Split audio into chunks at low-energy regions.

```python
def find_split_point(
    wav: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_energy_window: int,
) -> int
```
Find quietest point in audio region for clean splitting.

---

## Video Processing

Location: `vllm/multimodal/video.py`

### Metadata Types

```python
class VideoTargetMetadata(NamedTuple):
    num_frames: int
    fps: float
    max_duration: float

class VideoSourceMetadata(NamedTuple):
    total_frames_num: int
    original_fps: float
    duration: float
```

### VideoLoader (Abstract Base)

```python
class VideoLoader:
    @classmethod
    def compute_frames_index_to_sample(
        cls, source: VideoSourceMetadata, target: VideoTargetMetadata, **kwargs
    ) -> list[int]

    @classmethod
    @abstractmethod
    def load_bytes(cls, data: bytes, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]

    @classmethod
    def create_hf_metadata(
        cls, source: VideoSourceMetadata, valid_frame_indices: list[int],
        video_backend: str
    ) -> dict[str, Any]
```

### VideoBackend (Default)

```python
@VIDEO_LOADER_REGISTRY.register("opencv")
class VideoBackend(VideoLoader, OpenCVVideoBackendMixin, PyAVVideoBackendMixin):
```

Uniform-sampling video backend. Supports `opencv` and `pyav` decoding.

```python
@classmethod
def load_bytes(
    cls,
    data: bytes,
    num_frames: int = -1,
    fps: int = -1,
    max_duration: int = 300,
    frame_recovery: bool = False,
    *,
    backend: Literal["opencv", "pyav"] = "opencv",
    **kwargs,
) -> tuple[npt.NDArray, dict[str, Any]]
```

### DynamicVideoBackend

```python
@VIDEO_LOADER_REGISTRY.register("opencv_dynamic")
class DynamicVideoBackend(VideoBackend):
```

Duration-aware dynamic-sampling. Samples at `fps` up to `max_duration`, falling back
to uniform sampling for longer videos.

### Molmo2VideoBackend

```python
@VIDEO_LOADER_REGISTRY.register("molmo2")
class Molmo2VideoBackend(VideoLoader, OpenCVVideoBackendMixin):
```

Molmo2-specific video backend with configurable frame sampling modes.

### NemotronVLVideoBackend

```python
@VIDEO_LOADER_REGISTRY.register("nemotron_vl")
class NemotronVLVideoBackend(VideoBackend):
```
Includes original video bytes in metadata.

### OpenCVDynamicOpenPanguVideoBackend

```python
@VIDEO_LOADER_REGISTRY.register("openpangu")
class OpenCVDynamicOpenPanguVideoBackend(VideoLoader, OpenCVVideoBackendMixin):
```
OpenPangu-specific video backend with dynamic timestamp-based sampling.

### OpenCV Backend Mixin

```python
class OpenCVVideoBackendMixin:
    @staticmethod
    def get_cv2_video_api()

    @classmethod
    def open_video_capture(cls, data: bytes) -> cv2.VideoCapture

    @staticmethod
    def get_video_metadata(cap: cv2.VideoCapture) -> VideoSourceMetadata

    @classmethod
    def read_frames(
        cls,
        cap: cv2.VideoCapture,
        frame_idx: list[int],
        total_frames_num: int,
        *,
        frame_recovery: bool = False,
    ) -> tuple[npt.NDArray, list[int]]
```

Features forward-scan recovery for failed frames.

### PyAV Backend Mixin

```python
class PyAVVideoBackendMixin:
    @staticmethod
    def get_metadata(container: av.container.InputContainer) -> VideoSourceMetadata

    @staticmethod
    def decode_frames(
        container: av.container.InputContainer,
        frame_indices: list[int],
        fps: float,
        duration: float,
    ) -> tuple[npt.NDArray, list[int]]
```

### Video Utility Functions

```python
def resize_video(frames: npt.NDArray, size: tuple[int, int]) -> npt.NDArray
def rescale_video_size(frames: npt.NDArray, size_factor: float) -> npt.NDArray
def sample_frames_from_video(frames: npt.NDArray, num_frames: int) -> npt.NDArray
```

---

## Multimodal Hashing

Location: `vllm/multimodal/hasher.py`

### MultiModalHasher

```python
class MultiModalHasher:
```

Content hasher for multimodal caching. Supports blake3 (default), sha256, and sha512
algorithms via `VLLM_MM_HASHER_ALGORITHM` environment variable.

**Class Methods:**

```python
@classmethod
def serialize_item(cls, obj: object) -> Iterable[bytes | memoryview]
```
Serialize various types for hashing:
- `bytes`/`memoryview`: direct
- `str`: UTF-8 encode
- `int`/`float`: numpy array bytes
- `PIL.Image`: mode + pixel data (or UUID from EXIF)
- `MediaWithBytes`: uses original bytes
- `torch.Tensor`: numpy conversion (handles bfloat16)
- `np.ndarray`: dtype + shape + data
- Other: pickle fallback

```python
@classmethod
def iter_item_to_bytes(
    cls, key: str, obj: object
) -> Iterable[bytes | memoryview]
```
Recursively convert nested structures (lists, dicts) to bytes.

```python
@classmethod
def hash_kwargs(cls, **kwargs: object) -> str
```
Compute deterministic hash of keyword arguments. Sorts keys alphabetically.

---

## Utility Functions

Location: `vllm/multimodal/utils.py`

### Encoding Functions

```python
def encode_audio_base64(
    audio: np.ndarray, sampling_rate: int, *, format: str = "WAV"
) -> str

def encode_audio_url(
    audio: np.ndarray, sampling_rate: int, *, format: str = "WAV"
) -> str

def encode_image_base64(
    image: Image.Image, *, image_mode: str = "RGB", format: str = "PNG"
) -> str

def encode_image_url(
    image: Image.Image, *, image_mode: str = "RGB", format: str = "PNG"
) -> str

def encode_video_base64(frames: npt.NDArray, *, format: str = "JPEG") -> str

def encode_video_url(frames: npt.NDArray, *, format: str = "JPEG") -> str
```

### Placeholder Functions

```python
def argsort_mm_positions(
    mm_positions: MultiModalPlaceholders,
) -> list[tuple[str, int]]
```
Sort multimodal placeholder positions by offset in ascending order.

### Batching Functions

```python
def group_and_batch_mm_items(
    items: Sequence[MultiModalKwargsItem],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[int, BatchedTensorInputs]]
```
Group consecutive items into valid batches. Splits on different fields or shared field values.

```python
def group_and_batch_mm_kwargs(
    mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    *,
    device: torch.types.Device = None,
    pin_memory: bool = False,
) -> Generator[tuple[str, int, BatchedTensorInputs], None, None]
```
Group items by modality, then batch within each modality group.

### Fetch Functions

```python
def fetch_audio(
    audio_url: str, audio_io_kwargs: dict[str, Any] | None = None
) -> tuple[np.ndarray, int | float]

def fetch_image(
    image_url: str, image_io_kwargs: dict[str, Any] | None = None
) -> Image.Image

def fetch_video(
    video_url: str, video_io_kwargs: dict[str, Any] | None = None
) -> tuple[npt.NDArray, dict[str, Any]]
```

---

## Engine Input Types

Location: `vllm/inputs/engine.py`

### TokensInput

```python
class TokensInput(_InputOptions):
    type: Literal["token"]
    prompt_token_ids: list[int]
    prompt: NotRequired[str]
```

```python
def tokens_input(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokensInput
```

### EmbedsInput

```python
class EmbedsInput(_InputOptions):
    type: Literal["embeds"]
    prompt_embeds: torch.Tensor
    prompt: NotRequired[str]
    prompt_token_ids: NotRequired[list[int]]
    is_token_ids: NotRequired[list[bool]]
```

```python
def embeds_input(
    prompt_embeds: torch.Tensor,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
    prompt_token_ids: list[int] | None = None,
    is_token_ids: list[bool] | None = None,
) -> EmbedsInput
```

### MultiModalInput

```python
class MultiModalInput(_InputOptions):
    type: Literal["multimodal"]
    prompt_token_ids: list[int]
    prompt: NotRequired[str]
    mm_kwargs: MultiModalKwargsOptionalItems
    mm_hashes: MultiModalHashes
    mm_placeholders: MultiModalPlaceholders
```

```python
def mm_input(
    prompt_token_ids: list[int],
    mm_kwargs: MultiModalKwargsOptionalItems,
    mm_hashes: MultiModalHashes,
    mm_placeholders: MultiModalPlaceholders,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> MultiModalInput
```

### MultiModalEncDecInput

```python
class MultiModalEncDecInput(MultiModalInput):
    encoder_prompt_token_ids: list[int]
    encoder_prompt: NotRequired[str]
```

```python
def mm_enc_dec_input(
    encoder_inputs: MultiModalInput,
    decoder_prompt_token_ids: list[int],
    *,
    decoder_prompt: str | None = None,
) -> MultiModalEncDecInput
```

### Type Aliases

```python
MultiModalHashes: TypeAlias = Mapping[str, list[str]]
MultiModalPlaceholders: TypeAlias = Mapping[str, Sequence[PlaceholderRange]]
DecoderOnlyEngineInput: TypeAlias = TokensInput | EmbedsInput | MultiModalInput
SingletonInput: TypeAlias = DecoderOnlyEngineInput | MultiModalEncDecInput
EngineInput: TypeAlias = DecoderOnlyEngineInput | EncoderDecoderInput
```

### Encoder-Decoder Functions

```python
def build_enc_dec_input(
    encoder_input: SingletonInput,
    decoder_input: SingletonInput | None,
    decoder_start_token_id: int,
    skip_decoder_start_token: bool = False,
) -> EncoderDecoderInput

def split_enc_dec_input(
    inputs: EngineInput,
) -> tuple[SingletonInput | None, SingletonInput]
```

---

## Dummy Inputs Builder

Location: `vllm/multimodal/processing/dummy_inputs.py`

### BaseDummyInputsBuilder

```python
class BaseDummyInputsBuilder(ABC, Generic[_I]):
    def __init__(self, info: _I) -> None
```

Constructs dummy data for profiling multimodal models.

**Abstract Methods:**

```python
@abstractmethod
def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str

@abstractmethod
def get_dummy_mm_data(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
    mm_options: Mapping[str, BaseDummyOptions],
) -> MultiModalDataDict
```

**Concrete Method:**

```python
def get_dummy_processor_inputs(
    self,
    seq_len: int,
    mm_counts: Mapping[str, int],
    mm_options: Mapping[str, BaseDummyOptions],
) -> ProcessorInputs
```

**Helper Methods:**

```python
def _get_dummy_audios(
    self, *, length: int, num_audios: int,
    overrides: AudioDummyOptions | None = None,
) -> list[npt.NDArray]

def _get_dummy_images(
    self, *, width: int, height: int, num_images: int,
    overrides: ImageDummyOptions | None = None,
) -> list[Image.Image]

def _get_dummy_videos(
    self, *, width: int, height: int, num_frames: int, num_videos: int,
    overrides: VideoDummyOptions | None = None,
) -> list[npt.NDArray]
```

---

## ProcessorInputs

```python
@dataclass
class ProcessorInputs:
    prompt: str | list[int]
    mm_data_items: MultiModalDataItems
    mm_uuid_items: MultiModalUUIDItems | None = None
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenization_kwargs: Mapping[str, object] = field(default_factory=dict)
```

**Methods:**

```python
def get_mm_hashes(self, model_id: str) -> MultiModalHashes
```
Compute multimodal hashes for caching. Uses UUID items when provided,
with fallback to content-based hashing.
