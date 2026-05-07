# 19 - Hybrid and MIMO Models Reference

This document provides a comprehensive reference for Megatron-LM's hybrid SSM-Transformer
models and the MIMO (Multi-Input Multi-Output) framework. Hybrid models combine state-
space model (SSM) layers with attention and MLP layers in configurable patterns, while
the MIMO framework provides a flexible architecture for building multimodal models.

## Part 1: Hybrid SSM-Transformer Architecture

### Overview

The hybrid model architecture, implemented in `megatron/core/models/hybrid/`, enables
mixing different layer types within a single model. This supports architectures like
Nemotron-NAS (which combines Mamba SSM layers with attention layers) and provides the
flexibility to place any combination of layer types at any position in the model.

### Layer Type Symbols

Defined in `hybrid_layer_allocation.py` class `Symbols`:

| Symbol | Layer Type | Description |
|---|---|---|
| `M` | Mamba (SSM) | State-space model layer using MambaMixer |
| `G` | GDN (Gated DeltaNet) | Gated DeltaNet attention layer |
| `*` | Attention | Standard self-attention transformer layer |
| `D` | DS-Attention | DeepSeek-style MLA/DSA attention layer |
| `-` | MLP | Standalone MLP layer (no attention) |
| `E` | MoE | Mixture of Experts transformer layer |
| `\|` | Pipeline separator | Delimits pipeline stage boundaries |
| `/` | MTP separator | Separates main pattern from MTP patterns |

### HybridStackSubmodules

The `HybridStackSubmodules` dataclass defines module specs for each layer type:

```python
@dataclass
class HybridStackSubmodules:
    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    gdn_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    dsa_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None
```

### HybridStack

**File**: `megatron/core/models/hybrid/hybrid_block.py`

`HybridStack` is the core execution engine that builds and runs a stack of heterogeneous
layers based on a layer type pattern.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `TransformerConfig` | required | Model configuration |
| `submodules` | `HybridStackSubmodules` | required | Module specs for each layer type |
| `pre_process` | `bool` | `True` | Include embedding layer |
| `layer_type_list` | `list[str]` | required | Pre-computed layer types for this segment |
| `pp_layer_offset` | `int` | `0` | Global layer offset for this segment |
| `post_layer_norm` | `bool` | `True` | Include final layer norm |
| `post_process` | `bool` | `True` | Include output layer |
| `pg_collection` | `ProcessGroupCollection` | required | Process groups |
| `is_mtp_layer` | `bool` | `False` | Whether this is an MTP layer |

#### Layer Building

Layers are constructed by iterating over `layer_type_list` and instantiating the
corresponding module from `submodules`:

```python
for i, layer_type in enumerate(self.layer_type_list):
    layer_number = i + 1 + pp_layer_offset
    if layer_type == LayerSymbols.MAMBA:
        layer = build_module(submodules.mamba_layer, ...)
    elif layer_type == LayerSymbols.ATTENTION:
        layer = build_module(submodules.attention_layer, ...)
    elif layer_type == LayerSymbols.DS_ATTENTION:
        layer = build_module(submodules.dsa_layer, ...)
    elif layer_type == LayerSymbols.MLP:
        layer = build_module(submodules.mlp_layer, ...)
    elif layer_type == LayerSymbols.MOE:
        layer = build_module(submodules.moe_layer, ...)
    elif layer_type == LayerSymbols.GDN:
        layer = build_module(submodules.gdn_layer, ...)
```

#### Forward Pass

The forward pass iterates through all layers, applying quantization context managers
as needed:

```python
def forward(self, hidden_states, attention_mask, inference_context=None,
            rotary_pos_emb=None, packed_seq_params=None, padding_mask=None):
    for layer in self.layers:
        if isinstance(layer, TransformerLayer):
            hidden_states, _ = layer(hidden_states, attention_mask, ...)
        else:  # MambaLayer, Expert, or MLP
            hidden_states = layer(hidden_states, attention_mask, ...)
    if post_process and post_layer_norm:
        hidden_states = self.final_norm(hidden_states)
```

### HybridModel

**File**: `megatron/core/models/hybrid/hybrid_model.py`

`HybridModel` is the top-level model class that manages embeddings, position encoding,
the hybrid decoder stack, MTP blocks, and the output layer.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `TransformerConfig` | required | Model configuration |
| `hybrid_stack_spec` | `ModuleSpec` | required | HybridStack module spec |
| `vocab_size` | `int` | required | Vocabulary size |
| `max_sequence_length` | `int` | required | Maximum sequence length |
| `hybrid_layer_pattern` | `str` | `None` | Layer type pattern string |
| `hybrid_attention_ratio` | `float` | `None` | Deprecated: attention ratio |
| `hybrid_mlp_ratio` | `float` | `None` | Deprecated: MLP ratio |
| `hybrid_override_pattern` | `str` | `None` | Deprecated: override pattern |
| `pre_process` | `bool` | `True` | Include embeddings |
| `post_process` | `bool` | `True` | Include output layer |
| `parallel_output` | `bool` | `True` | Keep outputs split across TP |
| `share_embeddings_and_output_weights` | `bool` | `False` | Share embedding/output weights |
| `position_embedding_type` | `str` | `'none'` | Position embedding type |
| `rotary_percent` | `float` | `1.0` | RoPE rotary percent |
| `rotary_base` | `int` | `10000` | RoPE base frequency |
| `pg_collection` | `ProcessGroupCollection` | `None` | Process groups |
| `vp_stage` | `int` | `None` | Virtual pipeline stage |

### Layer Pattern Specification

The `hybrid_layer_pattern` string controls the architecture layout.

#### Basic Pattern

A string of layer type symbols defines the layer stack:

```
"M*M*M*M*"     # 5-layer model: Mamba-Attn-Mamba-Attn-Mamba-Attn
"MMMM"          # 4 Mamba layers only
"MMMM/*M/*M/*M" # 4 Mamba main + 3 MTP depths of Attn-Mamba
```

#### Pipeline Parallelism with Pipe Separators

The `|` symbol divides layers across pipeline stages:

```
"M-M-|M-M*-|M-M-|M-M*-"   # 4 pipeline segments
"MM*|MM*|MM*"              # 3 pipeline segments of 3 layers each
```

#### MTP (Multi-Token Prediction) with Slash Separators

The `/` symbol introduces MTP patterns. Each repeated group after the main pattern
represents one MTP prediction depth:

```
"M*M*/MM/MM"     # main="M*M*", MTP pattern="MM", 2 depths
"MMMM/*M/*M/*M"  # main="MMMM", MTP pattern="*M", 3 depths
```

#### ParsedHybridPattern

The `parse_hybrid_pattern()` function returns a `ParsedHybridPattern`:

```python
@dataclass
class ParsedHybridPattern:
    main_pattern: Optional[str]   # e.g., "M*M*"
    mtp_pattern: Optional[str]    # e.g., "MM"
    mtp_num_depths: int           # e.g., 2
```

### Pipeline Segment Selection

`select_pipeline_segment()` determines which layers run on each pipeline rank:

1. **Pipe-based**: When `|` separators are present, each segment is assigned to a PP rank
2. **Even split**: Without pipes but PP > 1, layers are split evenly
3. **Uneven split**: With `first_stage_layers` / `last_stage_layers`, asymmetric splits are supported

For Virtual Pipeline Parallelism (VPP), the segment index is `vp_rel * pp_size + pp_rank`.

### Default Layer Specs

**File**: `megatron/core/models/hybrid/hybrid_layer_specs.py`

#### `hybrid_stack_spec` (Training)

The default training spec uses Transformer Engine components:

| Layer Type | Module | Key Components |
|---|---|---|
| Mamba | `MambaLayer` | `MambaMixer` with `TELayerNormColumnParallelLinear` + `TERowParallelLinear` |
| Attention | `TransformerLayer` | `SelfAttention` with `TEDotProductAttention`, causal mask |
| DSA | `TransformerLayer` | `MLASelfAttention` with `DSAttention` indexer |
| MLP | `MLPLayer` | `MLP` with TE linear layers |
| MoE | `MoETransformerLayer` | MoE module with grouped GEMM |
| GDN | `TransformerLayer` | `GatedDeltaNet` with TE layers |

#### `hybrid_inference_stack_spec` (Inference)

Uses inference-optimized layers:
- `InferenceLayerNormColumnParallelLinear` for linear layers
- `InferenceRowParallelLinear` for output projections
- Inference-optimized MoE module spec

#### MTP Block Spec

The `_hybrid_mtp_block_spec` provides multi-token prediction support:

```python
_hybrid_mtp_block_spec = ModuleSpec(
    module=MultiTokenPredictionBlock,
    submodules=MultiTokenPredictionBlockSubmodules(
        layer_specs=[ModuleSpec(
            module=MultiTokenPredictionLayer,
            submodules=MultiTokenPredictionLayerSubmodules(
                enorm=TENorm, hnorm=TENorm,
                eh_proj=TEColumnParallelLinear,
                mtp_model_layer=None,  # Built via pattern + hybrid_submodules
                layer_norm=TENorm,
            ),
        )]
    ),
)
```

### Backward-Compatible Aliases

For backward compatibility with older code:

```python
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec
```

### Pattern Utility Functions

#### `pattern_from_ratios()`

Generates an evenly-spaced hybrid pattern from deprecated ratio arguments:

```python
pattern = pattern_from_ratios(
    num_layers=12,
    attention_ratio=0.33,  # 4 attention layers
    mlp_ratio=0.0,         # No standalone MLP layers
)
# Result: "MM*MM*MM*MM*MM"
```

#### `get_hybrid_total_layer_count()`

Returns total main decoder layers, stripping pipe separators:

```python
get_hybrid_total_layer_count("M*M*|M*M*")  # Returns 6
```

#### `get_hybrid_layer_counts()`

Returns a dictionary of layer counts by type:

```python
get_hybrid_layer_counts("M*M*/MM/MM")
# Returns: {'*': 2, 'G': 0, 'D': 0, 'M': 6, '-': 0, 'E': 0}
```

---

## Part 2: MIMO (Multi-Input Multi-Output) Framework

### Overview

**File**: `megatron/core/models/mimo/`

The MIMO framework provides a flexible architecture for building multimodal models that
process multiple input modalities (vision, audio, etc.) alongside text. It supports
arbitrary combinations of encoders and decoders with configurable parallelism strategies.

**Warning**: MIMO is experimental and under active development. The API may change.

### Architecture

```
                  +-----------+     +-----------+
Images ---------> | Vision    |     | Audio     |
                  | Encoder   |     | Encoder   | ...
                  +-----+-----+     +-----+-----+
                        |                 |
                  +-----v-----+   +-------v-----+
                  | Input     |   | Input       |
                  | Projection|   | Projection  |
                  +-----+-----+   +-------+-----+
                        |                 |
                        +--------+--------+
                                 |
                    +------------v-------------+
                    | Align Embeddings by      |
                    | Token Positions          |
                    +------------+-------------+
                                 |
                    +------------v-------------+
                    |     Language Model        |
                    +------------+-------------+
                                 |
                    +------------v-------------+
                    |       Output              |
                    +--------------------------+
```

### MimoModelConfig

**File**: `megatron/core/models/mimo/config/base_configs.py`

```python
@dataclass
class MimoModelConfig:
    language_model_spec: ModuleSpec           # Language model specification
    modality_submodules_spec: Dict[str, ModuleSpec]  # Modality name -> submodule spec
    special_token_ids: Dict[str, int]         # Modality name -> special token ID
    module_to_grid_map: Optional[Dict[str, HyperCommGrid]]  # Module grid mappings
    kv_format: str = "sbhd"                   # KV format: "sbhd" or "thd"
```

### MimoModel

**File**: `megatron/core/models/mimo/model/base.py`

The main model class that orchestrates multimodal processing.

#### Forward Pass

```python
def forward(
    self,
    input_ids: torch.Tensor,           # [B, S] token IDs with special modality tokens
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    packing_kwargs: Optional[dict] = None,
):
```

The `modality_inputs` dictionary maps modality names to their encoder inputs:

```python
modality_inputs = {
    "vision": {
        "clip_encoder": {"pixel_values": images},
    },
    "audio": {
        "whisper_encoder": {"input_features": features},
    }
}
```

#### Embedding Alignment

`align_embeddings_by_token_positions()` merges embeddings from different modalities
using special token positions in `input_ids`:

1. Create a combined tensor `[B, S, H]` initialized to zeros
2. For each modality, locate its special tokens in `input_ids`
3. Scatter the modality embeddings into the combined tensor at those positions
4. For text, scatter at all non-special-token positions
5. Transpose to `[S, B, H]` for the language model

#### Partition Adapter

When context parallelism or sequence parallelism is enabled, a `PartitionAdapter` handles
sharding the combined embeddings across ranks:

```python
if self.partition_adapter is not None:
    combined_embeddings, labels, loss_mask, _, packed_seq_params = \
        self.partition_adapter.shard(
            embeddings=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
```

### Module Layout Modes

**File**: `megatron/core/models/mimo/config/role.py`

#### COLOCATED Mode

All modules (encoders + language model) share the same ranks. This is the default
when no `module_to_grid_map` is provided, or when all grids span the same ranks.

```python
class ModuleLayout(Enum):
    COLOCATED = "colocated"       # All modules on all ranks
    NON_COLOCATED = "non_colocated"  # Modules on disjoint rank sets
```

In COLOCATED mode, `_forward_all_modules()` runs all encoders and the language model
on every rank, then merges embeddings before passing to the LM.

#### NON_COLOCATED Mode

Modules are distributed across different rank sets using `module_to_grid_map`.
Each rank runs EITHER encoders OR the language model.

`RankRole` determines what each rank does:

```python
@dataclass
class RankRole:
    modules: Dict[str, ModuleStageInfo]  # module_name -> stage info
    mode: ModuleLayout

    @property
    def has_modality_modules(self) -> bool: ...
    @property
    def has_language_module(self) -> bool: ...
    @property
    def modality_module_names(self) -> List[str]: ...
```

### ModalitySubmodules

**File**: `megatron/core/models/mimo/submodules/base.py`

Abstract base class for modality-specific processing pipelines.

#### Structure

Each `ModalitySubmodules` instance manages:

| Component | Description |
|---|---|
| `encoders` | `ModuleDict` of encoder modules |
| `decoders` | `ModuleDict` of decoder modules |
| `input_projections` | `ModuleList` of input projection layers |
| `output_projections` | `ModuleList` of output projection layers |

#### Pipeline Stages

The pipeline is stage-aware for pipeline parallelism:

- **First stage**: Encodes inputs, builds output projections
- **Last stage**: Applies input projections
- **Middle stages**: Passes hidden states through

#### Forward Pass

```python
def forward(
    self,
    encoder_inputs: Optional[Dict[str, Any]] = None,
    hidden_states: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if is_first_stage:
        embeddings = self.encode(encoder_inputs)
        combined = self.combine_embeddings(embeddings)
    else:
        combined = hidden_states

    if is_last_stage:
        return self.project_embeddings([combined], is_input=True)
    return combined
```

#### Factory Method

```python
@classmethod
def from_spec(cls, module_spec, is_first_stage=True, is_last_stage=True):
```

Builds the submodule from a `ModuleSpec`, conditionally creating encoders, decoders,
and projections based on pipeline stage position.

### Colocated Communication

**File**: `megatron/core/models/mimo/comm/colocated_communicator.py`

When modules share ranks but have different TP/DP configurations (heterogeneous
parallelism), `ColocatedBridgeCommunicator` handles the transformation of embeddings
between different parallelism layouts:

```python
# In MimoModel._build_colocated_communicators():
for mod_name in modality_submodules_spec:
    self.colocated_comms[(mod_name, lang_key)] = ColocatedBridgeCommunicator(
        src_grid=grid_map[mod_name],
        dest_grid=grid_map[lang_key],
        src_module_name=mod_name,
        dest_module_name=lang_key,
        dim_mapping={'b': 0, 'h': 1},
    )
```

### Vision and Audio Submodules

**Files**: `megatron/core/models/mimo/submodules/vision.py`, `audio.py`

Specialized submodule implementations for specific modalities:

- `VisionSubmodules`: Processes image/video inputs through vision encoders
- `AudioSubmodules`: Processes audio inputs through audio encoders

Both extend `ModalitySubmodules` and provide modality-specific encoding, decoding,
and projection logic.

### Sharded State Dict

MIMO models implement custom `sharded_state_dict()` methods that inject the correct
`dp_cp_group` from each module's `pg_collection`, avoiding global `parallel_state`
fallbacks. This enables heterogeneous parallelism where different modules may have
different DP/CP configurations.

## Configuration Quick Reference

### Hybrid Model Configuration

| Parameter | Description | Example |
|---|---|---|
| `--hybrid-layer-pattern` | Layer type pattern | `"M*M*M*M*M*"` |
| `--num-layers` | Total number of layers | `12` |
| `--position-embedding-type` | Position embedding | `rope`, `none` |
| `--rotary-percent` | RoPE percentage | `1.0` |
| `--mtp-num-layers` | MTP depth | `2` |

### MIMO Configuration

| Parameter | Description |
|---|---|
| `language_model_spec` | ModuleSpec for the language model |
| `modality_submodules_spec` | Dict mapping modality names to submodule specs |
| `special_token_ids` | Dict mapping modality names to token IDs |
| `module_to_grid_map` | Optional grid mappings for non-colocated mode |
| `kv_format` | KV format: `"sbhd"` or `"thd"` |
