# Mamba and Hybrid Model Architecture

This reference documents the Mamba/Hybrid model implementation in Megatron-Core. The original `MambaModel` class has been deprecated in favor of `HybridModel`, which supports heterogeneous layer stacks combining Mamba SSM layers, attention layers, MLP layers, MoE layers, Gated Delta Network (GDN) layers, and DeepSeek Sparse Attention (DSA) layers in arbitrary configurations.

## Source Files

| Component | Path |
|-----------|------|
| HybridModel | `megatron/core/models/hybrid/hybrid_model.py` |
| HybridStack | `megatron/core/models/hybrid/hybrid_block.py` |
| Layer Specs | `megatron/core/models/hybrid/hybrid_layer_specs.py` |
| Layer Allocation | `megatron/core/models/hybrid/hybrid_layer_allocation.py` |
| MambaModel (deprecated) | `megatron/core/models/mamba/mamba_model.py` |
| Mamba Layer Specs | `megatron/core/models/mamba/mamba_layer_specs.py` |

## Deprecation: MambaModel -> HybridModel

```python
class MambaModel(HybridModel):
    """Backward-compatible wrapper that accepts the deprecated mamba_stack_spec kwarg."""

    def __init__(self, *args, mamba_stack_spec: ModuleSpec = None, **kwargs):
        log_single_rank(
            logger, logging.WARNING, "MambaModel has been deprecated. Use HybridModel instead."
        )
        if mamba_stack_spec is not None:
            kwargs['hybrid_stack_spec'] = mamba_stack_spec
        super().__init__(*args, **kwargs)
```

The `MambaModel` class now simply extends `HybridModel` with backward-compatible argument handling. The `mamba_stack_spec` parameter is remapped to `hybrid_stack_spec`. Users should migrate to using `HybridModel` directly.

The `mamba_layer_specs.py` file re-exports from `hybrid_layer_specs.py`:

```python
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec
```

## HybridModel Class

`HybridModel` extends both `LanguageModule` and `GraphableMegatronModule`, implementing a configurable mixture of layer types in a single model. It supports Mamba SSM layers, attention layers, MLP-only layers, MoE layers, GDN layers, and DSA layers in any arrangement.

### Constructor

```python
class HybridModel(LanguageModule, GraphableMegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        hybrid_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        hybrid_layer_pattern: Optional[str] = None,
        hybrid_attention_ratio: Optional[float] = None,
        hybrid_mlp_ratio: Optional[float] = None,
        hybrid_override_pattern: Optional[str] = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'yarn', 'none'
        ] = 'none',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    )
```

### Key Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | `TransformerConfig` | required | Central model configuration |
| `hybrid_stack_spec` | `ModuleSpec` | required | Layer specs for all layer types |
| `hybrid_layer_pattern` | `str` | `None` | Unified pattern defining layer arrangement |
| `position_embedding_type` | `str` | `'none'` | Position embedding (default none for Mamba) |
| `hybrid_attention_ratio` | `float` | `None` | Deprecated. Use `hybrid_layer_pattern` instead |
| `hybrid_mlp_ratio` | `float` | `None` | Deprecated. Use `hybrid_layer_pattern` instead |
| `hybrid_override_pattern` | `str` | `None` | Deprecated. Use `hybrid_layer_pattern` instead |

### Backward Compatibility

The constructor handles deprecated parameters:
1. `hybrid_override_pattern` is mapped to `hybrid_layer_pattern` with a deprecation warning.
2. `hybrid_attention_ratio` and `hybrid_mlp_ratio` are converted to a pattern via `pattern_from_ratios()` with a deprecation warning.
3. If both deprecated and new parameters are provided, an error is raised.

### Internal Architecture

1. **Pattern Parsing**:
   ```python
   parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
   self.mtp_pattern = parsed.mtp_pattern
   self.mtp_num_depths = parsed.mtp_num_depths
   ```

2. **Pipeline Segment Selection**:
   ```python
   layer_type_list, layer_offset = select_pipeline_segment(
       parsed.main_pattern or '',
       self.pg_collection.pp,
       vp_stage,
       first_stage_layers=self.config.num_layers_in_first_pipeline_stage,
       last_stage_layers=self.config.num_layers_in_last_pipeline_stage,
   )
   ```

3. **Embedding Layer** (`LanguageModelEmbedding`): Created when `pre_process=True` or `mtp_process=True`. Default position embedding is `'none'` since Mamba layers do not require positional information.

4. **Rotary Position Embeddings**:
   - `'rope'` (without MLA): Creates `RotaryEmbedding`.
   - `'yarn'`: Creates `YarnRotaryEmbedding`.
   - `'none'`: No position embeddings (default for pure Mamba or MLA models).

5. **Decoder** (`HybridStack`): The main model stack built from the pre-computed `layer_type_list`.

6. **MTP Block** (`MultiTokenPredictionBlock`): Optional multi-token prediction layers using the `mtp_block_spec` from `hybrid_stack_spec.submodules`.

7. **Output Layer** (`ColumnParallelLinear`): Projects hidden states to vocabulary logits.

## Layer Pattern System

### Layer Type Symbols

The `Symbols` class defines the pattern language:

| Symbol | Layer Type | Module | Description |
|--------|-----------|--------|-------------|
| `M` | MAMBA | `MambaLayer` | Mamba SSM layer with `MambaMixer` |
| `*` | ATTENTION | `TransformerLayer` | Standard self-attention layer |
| `-` | MLP | `MLPLayer` | MLP-only layer (no attention) |
| `E` | MOE | `MoETransformerLayer` | Mixture of Experts layer |
| `G` | GDN | `TransformerLayer` with `GatedDeltaNet` | Gated Delta Network layer |
| `D` | DS_ATTENTION | `TransformerLayer` with `MLASelfAttention` + `DSAttention` | DeepSeek Sparse Attention with MLA |
| `\|` | PIPE | -- | Pipeline stage boundary separator |
| `/` | MTP_SEPARATOR | -- | MTP pattern separator |

### Pattern Format

The unified pattern format is:

```
<main_pattern>/<mtp_pattern>/<mtp_pattern>/...
```

Examples:

| Pattern | Main Decoder | MTP | Description |
|---------|-------------|-----|-------------|
| `M*M*` | `M*M*` | None | 4 layers alternating Mamba/Attention, no MTP |
| `M*M*/MM/MM` | `M*M*` | `MM`, 2 depths | 4 main layers + 2 MTP depths of MM each |
| `MMMM/*M/*M/*M` | `MMMM` | `*M`, 3 depths | 4 Mamba main + 3 MTP depths of Attention+Mamba |
| `M-M-\|M-M*-/MM/MM` | `M-M-\|M-M*-` | `MM`, 2 depths | 2 PP stages + MTP |
| `M-M-\|M-M*-\|M-M-\|M-M*-` | 4 PP segments | None | 4 pipeline stages with alternating patterns |

### Pattern Parsing

```python
@dataclass
class ParsedHybridPattern:
    main_pattern: Optional[str]     # Main decoder layers
    mtp_pattern: Optional[str]      # MTP layer pattern per depth
    mtp_num_depths: int             # Number of MTP prediction depths
```

The `parse_hybrid_pattern()` function:
1. Splits the pattern by `/` to separate main and MTP components.
2. Validates that all MTP patterns are identical.
3. Validates that all symbols are valid layer types.
4. Disallows mixing Attention (`*`) and DS_Attention (`D`) in the same model.

### Pipeline Segment Selection

```python
def select_pipeline_segment(
    main_pattern: str,
    pp_group: Optional[torch.distributed.ProcessGroup],
    vp_stage: Optional[int],
    first_stage_layers: Optional[int] = None,
    last_stage_layers: Optional[int] = None,
) -> Tuple[List[str], int]
```

Handles two modes:

**Pipe-based (recommended)**: When `|` separators are present in the pattern, splits into segments and selects by `vp_rel * pp_size + pp_rank`.

**Legacy (deprecated)**: When no `|` separators exist but PP > 1, evenly splits the pattern across PP ranks. Supports uneven splits via `first_stage_layers` and `last_stage_layers`. VPP is not supported without pipe separators.

### Pattern from Ratios (Deprecated)

```python
def pattern_from_ratios(
    num_layers: int, attention_ratio: float = 0.0, mlp_ratio: float = 0.0
) -> str
```

Generates an evenly-spaced pattern from target ratios. Attention layers are placed first, then MLP layers replace Mamba layers.

## HybridStack Class

`HybridStack` extends `MegatronModule` and implements the actual heterogeneous layer execution.

### Constructor

```python
class HybridStack(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: Optional[list[str]] = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    )
```

### Layer Building

For each symbol in `layer_type_list`, the appropriate layer spec is selected from `submodules` and built:

```python
for i, layer_type in enumerate(self.layer_type_list):
    with quant_init_context:
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
    self.layers.append(layer)
```

### Quantization Context

FP8/FP4 contexts are applied per-layer during initialization:

```python
if self.config.fp8:
    quant_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
elif self.config.fp4:
    quant_init_context = get_fp4_context(self.config, i + pp_layer_offset, is_init=True)
else:
    quant_init_context = nullcontext()
```

### Forward Pass

```python
def forward(
    self,
    hidden_states: Union[Tensor, WrappedTensor],
    attention_mask: Tensor,
    inference_context: Optional[BaseInferenceContext] = None,
    rotary_pos_emb: Optional[Tensor] = None,
    *,
    inference_params=None,
    packed_seq_params=None,
    padding_mask=None,
)
```

1. **Input Handling**: For non-preprocess stages, uses `self.input_tensor`. Unwraps `WrappedTensor` for inference.

2. **Quantization Context Management**:
   - **Delayed FP8**: Single outer context wrapping the entire forward pass.
   - **Non-delayed FP8/FP4**: Per-layer inner context for fine-grained quantization control.

3. **Layer Execution**:
   ```python
   with outer_fp8_context:
       for layer in self.layers:
           with inner_quant_context:
               if isinstance(layer, TransformerLayer):
                   hidden_states, _ = layer(
                       hidden_states, attention_mask, inference_context,
                       rotary_pos_emb, sequence_len_offset, packed_seq_params, padding_mask
                   )
               else:  # MambaLayer, MLPLayer
                   hidden_states = layer(
                       hidden_states, attention_mask, inference_context, packed_seq_params
                   )
   ```

4. **Final Layer Norm**: Applied when `post_process=True` and `post_layer_norm=True`:
   ```python
   self.final_norm = TENorm(config, hidden_size, eps=layernorm_epsilon)
   ```

### Mamba State Shapes

```python
def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
    """Returns conv and SSM state shapes if this block contains Mamba layers."""
    for layer_type, layer in zip(self.layer_type_list, self.layers):
        if layer_type == LayerSymbols.MAMBA:
            return layer.mamba_state_shapes_per_request()
    return None
```

Used for inference state management to pre-allocate Mamba SSM state buffers.

## HybridStackSubmodules

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

Each field maps to a layer type symbol. Unused layer types default to `IdentityOp` (no-op). The `mtp_block_spec` provides the MTP block configuration.

## Layer Specification Details

### hybrid_stack_spec (Training)

Defined in `hybrid_layer_specs.py` with Transformer Engine components:

| Layer Type | Module | Key Components |
|-----------|--------|----------------|
| `mamba_layer` | `MambaLayer` | `MambaMixer` with `TELayerNormColumnParallelLinear` (in_proj) and `TERowParallelLinear` (out_proj) |
| `attention_layer` | `TransformerLayer` | `SelfAttention` with `TELayerNormColumnParallelLinear` (QKV), `TEDotProductAttention`, `TERowParallelLinear` (proj) |
| `dsa_layer` | `TransformerLayer` | `MLASelfAttention` + `DSAttention` with `DSAIndexer` |
| `mlp_layer` | `MLPLayer` | `MLP` with `TELayerNormColumnParallelLinear` (fc1) and `TERowParallelLinear` (fc2) |
| `moe_layer` | `MoETransformerLayer` | MoE module with TE GroupedMLP |
| `gdn_layer` | `TransformerLayer` | `GatedDeltaNet` with `TELayerNormColumnParallelLinear`, `TENorm`, `TERowParallelLinear` |

### hybrid_inference_stack_spec

Uses inference-optimized linear layers:
- `InferenceLayerNormColumnParallelLinear` instead of `TELayerNormColumnParallelLinear`.
- `InferenceRowParallelLinear` instead of `TERowParallelLinear`.
- `InferenceColumnParallelLinear` for MTP block projections.
- Inference-optimized MoE spec via `get_inference_optimized_moe_spec()`.

## CUDA Graph Support

HybridModel supports CUDA graphs through `GraphableMegatronModule`:

```python
def _should_call_local_cudagraph(self, *args, **kwargs):
    return (
        not self.training
        and hasattr(self, 'cudagraph_manager')
        and inference_context is not None
        and CudaGraphScope.full_iteration_inference in self.config.cuda_graph_scope
        and using_cuda_graph
    )
```

When CUDA graphs are active, the model wraps output in `[output]` format for graph replay compatibility.

```python
def create_mcore_cudagraph_manager(self, config):
    if CudaGraphScope.full_iteration_inference in config.cuda_graph_scope:
        from megatron.core.transformer.cuda_graphs import CudaGraphManager
        self.cudagraph_manager = CudaGraphManager(config)
```

## Forward Pass (HybridModel)

```python
def forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context=None,
    runtime_gather_output=None,
    *,
    inference_params=None,
    loss_mask=None,
    packed_seq_params=None,
    padding_mask=None,
) -> Tensor
```

### Pipeline

```
input_ids --> embedding --> RoPE/YaRN --> HybridStack decoder --> MTP (optional) --> output_layer --> loss/logits
```

1. **Embedding**: When `pre_process=True`, applies `self.embedding(input_ids, position_ids)`.

2. **Position Embeddings**: Computes RoPE or YaRN embeddings when configured. Mamba layers do not use position embeddings; only attention layers consume them.

3. **Decoder Forward**: Passes through `HybridStack` which iterates through heterogeneous layers.

4. **MTP Processing**: When `mtp_process=True` and not in inference/spec-decode mode, runs `self.mtp()` for multi-token prediction.

5. **Output Layer**: Projects hidden states through `ColumnParallelLinear` to vocabulary logits.

6. **MuP Scaling**: Applies `self._scale_logits()` when `config.use_mup=True`.

7. **Inference Optimizations**: Supports `materialize_only_last_token_logits` for last-token-only logit extraction.

## Fine-Grained Activation Offloading

```python
def preprocess_for_fine_grained_offloading(self):
    off_interface.init_chunk_handler(
        vp_size=self.config.virtual_pipeline_model_parallel_size,
        vp_stage=self.vp_stage,
        min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
    )
    for param in self.decoder.parameters():
        off_interface.mark_not_offloadable(param)
    if self.mtp_process:
        for param in self.mtp.parameters():
            off_interface.mark_not_offloadable(param)
    if self.post_process:
        for param in self.output_layer.parameters():
            off_interface.mark_not_offloadable(param)
```

## Sharded State Dict

`HybridStack` implements custom sharded state dict logic to handle layer numbering across pipeline stages:

```python
def sharded_state_dict(self, prefix='', sharded_offsets=None, metadata=None):
    for local_layer_idx, layer in enumerate(self.layers):
        global_layer_offset = layer.layer_number - 1
        state_dict_prefix = f'{layer_prefix}{local_layer_idx}.'
        sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
        # Replace prefix for correct global layer indexing
        replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
```

This ensures that checkpoints use globally consistent layer numbering regardless of pipeline parallelism configuration.

## Layer Counting Utilities

```python
def get_hybrid_total_layer_count(pattern: str) -> int
def get_hybrid_total_pipeline_segment_count(pattern: str) -> int
def get_hybrid_layer_counts(pattern: str) -> Dict[str, int]
def get_layer_maps_from_layer_type_list(layer_type_list: list[str]) -> dict[str, dict[int, int]]
```

These utilities provide information about the pattern without instantiating the model:
- `get_hybrid_layer_counts("M*M*/MM/MM")` returns `{'M': 4, '*': 2, '-': 0, 'E': 0, 'G': 0, 'D': 0}`.
- `get_layer_maps_from_layer_type_list` maps global layer indices to per-type local indices.

## Mamba SSM Layer Details

### MambaLayer Architecture

Each Mamba layer contains a `MambaMixer` which implements the Selective State Space Model:

```
Input [s, b, h]
    |
    v
in_proj (TELayerNormColumnParallelLinear): h -> 2*ssm_state_dim + 2*head_dim
    |
    +-----> A projection (state dynamics)
    +-----> B projection (input matrix)
    +-----> C projection (output matrix)
    +-----> dt (discretization parameter)
    |
    v
Selective Scan (SSM core computation)
    |
    v
out_proj (TERowParallelLinear): ssm_state_dim -> h
    |
    v
Output [s, b, h]
```

### TransformerConfig Parameters for Mamba

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mamba_state_dim` | int | 128 | SSM state dimension |
| `mamba_head_dim` | int | 64 | Head dimension in Mamba layers |
| `mamba_num_groups` | int | 8 | Number of groups in Mamba layers |
| `mamba_num_heads` | int | None | Number of Mamba heads (auto-computed from hidden_size / head_dim) |
| `use_mamba_mem_eff_path` | bool | True | Use memory-efficient computation path |

### Memory-Efficient Path

When `use_mamba_mem_eff_path=True` (default), the Mamba layer uses a memory-optimized implementation that:
1. Fuses the discretization step with the scan operation.
2. Avoids materializing intermediate tensors.
3. Uses custom CUDA kernels for the selective scan.

## Comparison: Mamba vs Transformer

| Aspect | Transformer | Mamba SSM |
|--------|------------|-----------|
| Time Complexity | O(L^2) for attention | O(L) for SSM scan |
| Memory (activations) | O(L^2) for KV cache | O(L) for state |
| Long Sequence Handling | Needs context parallelism | Natural linear scaling |
| Expressiveness | High (global attention) | Moderate (state-based) |
| Inference Speed | KV cache grows with length | Fixed-size state |

## Configuration Examples

### Pure Mamba Model

```python
config = TransformerConfig(
    num_layers=64,
    hidden_size=5120,
    bf16=True,
)

model = HybridModel(
    config=config,
    hybrid_stack_spec=hybrid_stack_spec,
    hybrid_layer_pattern="M" * 64,  # 64 Mamba layers
    vocab_size=32000,
    max_sequence_length=4096,
    position_embedding_type='none',
)
```

### Mamba-Attention Hybrid (Jamba-style)

```python
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    bf16=True,
)

# Alternating Mamba and Attention: M*M*M*M*...
pattern = "M*" * 16  # 32 layers total, 16 Mamba + 16 Attention

model = HybridModel(
    config=config,
    hybrid_stack_spec=hybrid_stack_spec,
    hybrid_layer_pattern=pattern,
    vocab_size=32000,
    max_sequence_length=4096,
    position_embedding_type='rope',
)
```

### Hybrid with MLP and MoE

```python
# Pattern: M-M-E-M-M-E (6 layers: 4 Mamba, 0 Attention, 2 MoE)
pattern = "MMEMME"

model = HybridModel(
    config=config,
    hybrid_stack_spec=hybrid_stack_spec,
    hybrid_layer_pattern=pattern,
    vocab_size=32000,
    max_sequence_length=4096,
    position_embedding_type='rope',
)
```

### Hybrid with Pipeline Parallelism

```python
# 8 layers split across 2 PP stages, with MTP
pattern = "M*M*M*M*|M*M*M*M*/MM/MM"

model = HybridModel(
    config=config,
    hybrid_stack_spec=hybrid_stack_spec,
    hybrid_layer_pattern=pattern,
    vocab_size=32000,
    max_sequence_length=4096,
    position_embedding_type='rope',
)
```

### DeepSeek-V2 Style (with DSA)

```python
# Pattern: M-M-M-D-M-M-M-D (Mamba + DeepSeek Sparse Attention)
pattern = "MMMDMMMD"

model = HybridModel(
    config=config,
    hybrid_stack_spec=hybrid_stack_spec,
    hybrid_layer_pattern=pattern,
    vocab_size=32000,
    max_sequence_length=4096,
    position_embedding_type='none',  # MLA has its own RoPE
)
```

## Backward-Compatible Aliases

```python
# In hybrid_block.py
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack

# In hybrid_layer_specs.py
mamba_stack_spec = hybrid_stack_spec
mamba_inference_stack_spec = hybrid_inference_stack_spec
```

## Related Documentation

- **13-gpt-model.md**: GPT model architecture (standard transformer)
- **19-hybrid-mimo-models.md**: MIMO multi-modal framework using HybridModel
- **20-moe-architecture.md**: MoE layer architecture for expert layers
