# Chapter 05: Transformer Building Blocks Reference

## Source Files
- `sources/Megatron-LM/megatron/core/transformer/transformer_layer.py`
- `sources/Megatron-LM/megatron/core/transformer/transformer_block.py`
- `sources/Megatron-LM/megatron/core/transformer/mlp.py`

## Overview

This reference covers the three fundamental building blocks of a Megatron-Core transformer model: `TransformerLayer` (a single decoder layer), `TransformerBlock` (a stack of layers), and `MLP` (the feed-forward network). These classes work together through a modular spec system that allows flexible customization of every submodule.

```
TransformerBlock
  |-- TransformerLayer (x N)
  |     |-- input_layernorm
  |     |-- self_attention
  |     |-- self_attn_bda (bias-dropout-add)
  |     |-- pre_cross_attn_layernorm
  |     |-- cross_attention
  |     |-- cross_attn_bda
  |     |-- pre_mlp_layernorm
  |     |-- MLP (or MoELayer)
  |     |-- mlp_bda
  |-- final_layernorm
```

---

## TransformerLayerSubmodules

The `TransformerLayerSubmodules` dataclass defines the specification for all submodules within a single transformer layer. Each field accepts a `ModuleSpec`, a builder type, or `IdentityOp` (to disable).

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `input_layernorm` | `LayerNormBuilder` | `IdentityOp` | Input layer normalization |
| `self_attention` | `Union[ModuleSpec, type]` | `IdentityOp` | Self-attention module spec |
| `self_attn_bda` | `Union[ModuleSpec, type]` | `IdentityFuncOp` | Bias-dropout-add after self-attention |
| `pre_cross_attn_layernorm` | `LayerNormBuilder` | `IdentityOp` | Layer norm before cross-attention |
| `cross_attention` | `Union[ModuleSpec, type]` | `IdentityOp` | Cross-attention module spec |
| `cross_attn_bda` | `Union[ModuleSpec, type]` | `IdentityFuncOp` | Bias-dropout-add after cross-attention |
| `pre_mlp_layernorm` | `LayerNormBuilder` | `IdentityOp` | Layer norm before MLP |
| `mlp` | `Union[ModuleSpec, type]` | `IdentityOp` | MLP or MoE layer spec |
| `mlp_bda` | `Union[ModuleSpec, type]` | `IdentityFuncOp` | Bias-dropout-add after MLP |
| `sharded_state_dict_keys_map` | `Dict[str, str]` | `{}` | Mapping for sharded tensor keys |

---

## TransformerLayer

`TransformerLayer` is a single transformer decoder layer. It takes input of shape `[s, b, h]` (sequence length, batch size, hidden size) and returns output of the same size.

**Inheritance**: `GraphableMegatronModule` -> `MegatronModule` -> `torch.nn.Module`, plus `BaseTransformerLayer` (ABC).

### __init__

```python
class TransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
    ):
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `TransformerConfig` | required | Transformer configuration |
| `submodules` | `TransformerLayerSubmodules` | required | Submodule specifications |
| `layer_number` | `int` | `1` | 1-based local layer index within this pipeline stage |
| `hidden_dropout` | `Optional[float]` | `None` | Override dropout; defaults to `config.hidden_dropout` |
| `pg_collection` | `Optional[ProcessGroupCollection]` | `None` | Process group collection for TP/PP/CP/EP. Defaults to MPU groups |
| `vp_stage` | `Optional[int]` | `None` | Virtual pipeline stage index |
| `is_mtp_layer` | `bool` | `False` | Whether this is a Multi-Token Prediction layer |
| `add_layer_offset` | `bool` | `True` | Whether to add pipeline stage offset to layer_number |
| `pp_layer_offset` | `Optional[int]` | `None` | Explicit pipeline parallel layer offset |

#### Internal Modules Created

The constructor creates the following modules in order:

1. **input_layernorm** - Input layer normalization
2. **self_attention** - Self-attention block (from `submodules.self_attention`)
3. **self_attn_bda** - Bias-dropout-add after self-attention
4. **pre_cross_attn_layernorm** - Layer norm before cross-attention
5. **cross_attention** - Cross-attention block
6. **cross_attn_bda** - Bias-dropout-add after cross-attention
7. **pre_mlp_layernorm** - Layer norm before MLP
8. **mlp** - MLP or MoELayer (detects MoE automatically)
9. **mlp_bda** - Bias-dropout-add after MLP

The `is_moe_layer` attribute is set automatically by checking if the MLP is an instance of `MoELayer`.

#### Layer Number Computation

The global `layer_number` is computed by adding a pipeline-stage offset. For standard layers (not MTP), the offset is computed via `get_transformer_layer_offset(config, vp_stage, pp_rank)`. MTP layers use their own numbering scheme.

### forward

```python
def forward(
    self,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
    context: Optional[Tensor] = None,
    context_mask: Optional[Tensor] = None,
    rotary_pos_emb: Optional[Tensor] = None,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    rotary_pos_cos_sin: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[Tensor] = None,
    padding_mask: Optional[Tensor] = None,
    *,
    inference_params: Optional[Any] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
```

#### Forward Parameters

| Parameter | Shape | Description |
|---|---|---|
| `hidden_states` | `[s, b, h]` | Input tensor |
| `attention_mask` | `[1, 1, s, s]` | Self-attention mask |
| `context` | `[s, b, h]` | Cross-attention context (encoder output) |
| `context_mask` | `[1, 1, s, s]` | Cross-attention mask |
| `rotary_pos_emb` | `Tuple[Tensor, Tensor]` | Rotary position embeddings (q, k) |
| `rotary_pos_cos` | `[s, 1, 1, d]` | Rotary cosine for flash decode |
| `rotary_pos_sin` | `[s, 1, 1, d]` | Rotary sine for flash decode |
| `rotary_pos_cos_sin` | `[s, 1, 1, 2d]` | Combined cos/sin for flashinfer RoPE |
| `attention_bias` | `[b, num_head, sq, skv]` | Attention bias for Q*K^T |
| `inference_context` | `BaseInferenceContext` | Inference KV cache manager |
| `packed_seq_params` | `PackedSeqParams` | Packed sequence parameters |
| `sequence_len_offset` | `Tensor` | Sequence offset during inference |
| `padding_mask` | `[bsz, seq_length]` | Padding mask for MoE aux loss |

#### Returns

A tuple `(output, context)`:
- **output**: Tensor of shape `[s, b, h]` - the transformed hidden states.
- **context**: Updated context tensor if cross-attention is used, otherwise `None`.

#### Forward Flow

1. **_forward_attention**: Applies input layernorm -> self-attention -> bias-dropout-add -> pre-cross-attn layernorm -> cross-attention -> bias-dropout-add.
2. **_forward_mlp**: Applies pre-MLP layernorm -> MLP/MoE -> bias-dropout-add.

### _forward_attention

Executes the attention portion of the layer. Key behaviors:

- Optionally recomputes `input_layernorm` output if `recompute_modules` includes `"layernorm"`.
- Supports fp32 residual connections when `config.fp32_residual_connection=True`.
- Supports fused TP inference kernel when `config.inference_fuse_tp_communication=True`.
- Offloads attention norm input to CPU when fine-grained activation offloading is enabled.

### _forward_mlp

Executes the MLP/MoE portion. Key behaviors:

- Optionally recomputes `pre_mlp_layernorm` output.
- Optionally recomputes the dense MLP when `recompute_modules` includes `"mlp"`.
- Supports MLP chunking during prefill/training to reduce peak activation memory.
- For MoE layers, passes `padding_mask` to exclude padding tokens from aux loss computation.
- Handles MoE-specific CUDA graph partial capture.

### MoETransformerLayer

A subclass of `TransformerLayer` specialized for MoE layers. It overrides:

- **`_should_call_local_cudagraph`**: Controls full-layer vs. partial CUDA graph capture.
- **`transition_cudagraph_scope`**: Switches between full-layer (inference) and partial (training) CUDA graph modes.
- **`_forward_mlp`**: Orchestrates partial CUDA graph execution by splitting the forward into router, expert_compute, and postprocess phases.

---

## TransformerBlockSubmodules

```python
@dataclass
class TransformerBlockSubmodules:
    layer_specs: Optional[List[ModuleSpec]] = None
    layer_norm: LayerNormBuilder | None = None
```

| Field | Type | Default | Description |
|---|---|---|---|
| `layer_specs` | `Optional[List[ModuleSpec]]` | `None` | Specs for each layer. Fan-out from a single spec if all layers are identical. |
| `layer_norm` | `LayerNormBuilder \| None` | `None` | Final layer normalization builder. |

---

## TransformerBlock

`TransformerBlock` is the top-level container that stacks multiple `TransformerLayer` instances, handles pipeline parallel input/output, activation checkpointing, and final layer normalization.

**Inheritance**: `GraphableMegatronModule`, `MegatronModule`

### __init__

```python
class TransformerBlock(GraphableMegatronModule, MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `TransformerConfig` | required | Transformer configuration |
| `spec` | `Union[TransformerBlockSubmodules, ModuleSpec]` | required | Layer specifications. A single `ModuleSpec` for a `BaseTransformerLayer` is automatically fanned out to all layers. |
| `post_layer_norm` | `bool` | `True` | Apply final layer norm after all layers |
| `pre_process` | `bool` | `True` | Whether this block is at the start of the pipeline (receives input directly) |
| `post_process` | `bool` | `True` | Whether this block is at the end of the pipeline |
| `pg_collection` | `Optional[ProcessGroupCollection]` | `None` | Process groups |
| `vp_stage` | `Optional[int]` | `None` | Virtual pipeline stage |

#### Key Attributes

| Attribute | Type | Description |
|---|---|---|
| `layers` | `torch.nn.ModuleList` | The stacked transformer layers |
| `num_layers_per_pipeline_rank` | `int` | Number of layers in this pipeline stage |
| `final_layernorm` | `Module or None` | Final layer normalization (only on last stage) |
| `input_tensor` | `Tensor or None` | Tensor received from previous pipeline stage |
| `checkpoint_core_attention` | `bool` | Whether selective core attention checkpointing is active |

#### Layer Building

`_build_layers()` constructs the layers. For each layer spec:
1. Computes the global layer number.
2. For heterogeneous architectures (`heterogeneous_block_specs=True`), retrieves per-layer config.
3. Wraps layer construction in an FP8/FP4 quantization context if applicable.
4. Creates a final layernorm if this is the last pipeline stage and `post_layer_norm=True`.

### forward

```python
def forward(
    self,
    hidden_states: Union[Tensor, WrappedTensor],
    attention_mask: Optional[Tensor],
    context: Optional[Tensor] = None,
    context_mask: Optional[Tensor] = None,
    rotary_pos_emb: Optional[Tensor] = None,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    rotary_pos_cos_sin: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[Tensor] = None,
    padding_mask: Optional[Tensor] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    dynamic_inference_decode_only: Optional[bool] = None,
) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
```

#### Special Parameters

| Parameter | Description |
|---|---|
| `extract_layer_indices` | Set of global layer indices (0-based) at which to extract intermediate hidden states |
| `dynamic_inference_decode_only` | Identifies decode vs. non-decode CUDA graph runners |

#### Forward Flow

1. **Input handling**: If `pre_process=False`, uses `self.input_tensor` (set by pipeline schedule). Wraps in viewless tensor.
2. **RNG context**: Forks CUDA RNG tracker if sequence parallel is enabled.
3. **Quantization context**: Wraps forward in FP8/FP4 context as appropriate.
4. **Activation recompute**: If `recompute_granularity='full'`, delegates to `_checkpointed_forward`.
5. **Layer-by-layer forward**: Iterates through all layers, passing hidden states through each.
6. **CPU offloading**: Commits offloaded activations after each layer if enabled.
7. **Feature extraction**: Collects intermediate hidden states at specified layer indices.
8. **Final layernorm**: Applies final layer normalization if present.

#### Returns

- If `extract_layer_indices` is empty: just the output `Tensor` of shape `[s, b, h]`.
- If `extract_layer_indices` is non-empty: a tuple `(output, intermediate_hidden_states)`.

### _checkpointed_forward

Handles full activation checkpointing with two methods:

**Uniform**: Divides layers into chunks of `recompute_num_layers` and checkpoints each chunk. Only the last layer of each chunk can have features extracted.

**Block**: Checkpoints only the first `recompute_num_layers` layers and runs the rest without checkpointing. Supports feature extraction at any layer.

### set_input_tensor

```python
def set_input_tensor(self, input_tensor: Tensor):
```

Sets the input tensor for pipeline parallelism. Called by the pipeline schedule before `forward()` to pass the output from the previous pipeline stage.

### get_num_layers_to_build (module-level function)

```python
def get_num_layers_to_build(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> int:
```

Computes the number of transformer layers to build for the current pipeline stage. Accounts for uneven pipeline parallelism (`num_layers_in_first_pipeline_stage`, `num_layers_in_last_pipeline_stage`), virtual pipeline parallelism, and embedding/loss layer placement.

---

## MLPSubmodules

```python
@dataclass
class MLPSubmodules:
    linear_fc1: LinearFc1Builder
    linear_fc2: LinearFc2Builder
    activation_func: TEActivationFunctionBuilder | None = None
```

| Field | Type | Default | Description |
|---|---|---|---|
| `linear_fc1` | `LinearFc1Builder` | required | First linear projection (hidden_size -> ffn_hidden_size) |
| `linear_fc2` | `LinearFc2Builder` | required | Second linear projection (ffn_hidden_size -> hidden_size) |
| `activation_func` | `TEActivationFunctionBuilder \| None` | `None` | TE activation function builder (only used with `use_te_activation_func=True`) |

---

## MLP

`MLP` implements the feed-forward network: `hidden_size -> ffn_hidden_size -> activation -> hidden_size`. With gated linear units (SwiGLU), the fc1 output is doubled to accommodate gating.

**Inheritance**: `MegatronModule` -> `torch.nn.Module`

### __init__

```python
class MLP(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `TransformerConfig` | required | Transformer configuration |
| `submodules` | `MLPSubmodules` | required | Submodule specifications for linear layers |
| `is_expert` | `bool` | `False` | Whether this MLP is an MoE expert |
| `input_size` | `Optional[int]` | `None` | Input dimension; defaults to `config.hidden_size` |
| `ffn_hidden_size` | `Optional[int]` | `None` | FFN hidden size; defaults to `config.ffn_hidden_size` |
| `tp_group` | `Optional[ProcessGroup]` | `None` | Tensor parallel process group |

#### Internal Modules

1. **linear_fc1**: Column-parallel linear projection. When `gated_linear_unit=True`, the output width is `2 * ffn_hidden_size` with stride=2 for weight interleaving.
2. **activation_func**: Set from `config.activation_func` (e.g., `F.gelu`, `F.silu`) or TE activation function builder.
3. **linear_fc2**: Row-parallel linear projection back to hidden size.

#### MoE Latent Projections

When `config.moe_latent_size` is set and `is_expert=True`, the MLP uses `moe_latent_size` instead of `hidden_size` for the input/output of the linear projections.

### forward

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    per_token_scale: torch.Tensor | None = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

#### Parameters

| Parameter | Shape | Description |
|---|---|---|
| `hidden_states` | `[s, b, h]` | Input tensor |
| `per_token_scale` | `[s, b]` | Per-token scaling factor for MoE weight quantization |

#### Returns

A tuple `(output, output_bias)`:
- **output**: Tensor of shape `[s, b, h]`
- **output_bias**: Bias tensor or `None` (when bias is consumed internally)

#### Forward Flow

1. **linear_fc1**: Projects input from hidden_size to ffn_hidden_size (column-parallel).
2. **Activation**: Applies the configured activation function. Supports multiple paths:
   - **TE activation** (`use_te_activation_func=True`): Uses TE's fused activation.
   - **Bias-activation fusion** (`bias_activation_fusion=True`): Fuses bias add with activation (supports `gelu`, `swiglu`, `quick_gelu`).
   - **Non-fused**: Adds bias separately, then applies activation. For GLU, splits the fc1 output into two halves and applies `activation_func(x_glu) * (x_linear + offset)`.
3. **linear_fc2**: Projects back from ffn_hidden_size to hidden_size (row-parallel).
4. Returns output and bias (bias may be None if consumed during fusion or by per_token_scale).

---

## Layer Specs and Module Specs

Megatron-Core uses a spec system to compose models flexibly. A `ModuleSpec` wraps a module class with optional submodules and additional keyword arguments:

```python
from megatron.core.transformer.spec_utils import ModuleSpec

# ModuleSpec(module_class, submodules, **kwargs)
layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=LayerNormImpl,
        self_attention=ModuleSpec(
            module=SelfAttention,
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
            ),
        ),
        self_attn_bda=GetBiasDropoutAdd,
        pre_mlp_layernorm=LayerNormImpl,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        ),
        mlp_bda=GetBiasDropoutAdd,
    ),
)
```

### Spec Fan-Out

When a `TransformerBlock` receives a single `ModuleSpec` for a `BaseTransformerLayer`, it automatically fans out that spec into N copies (one per layer) via `_get_block_submodules`. This means all layers share the same architecture. For heterogeneous architectures (e.g., Nemotron-NAS), set `heterogeneous_block_specs=True` and provide a `TransformerBlockSubmodules` with individual layer specs.

### Custom Layers

To create custom layers:

1. **Subclass `BaseTransformerLayer`** so the block recognizes it during fan-out.
2. **Define custom submodules** dataclass following the pattern of `TransformerLayerSubmodules`.
3. **Provide a `ModuleSpec`** or `TransformerBlockSubmodules` to `TransformerBlock`.

Example of a custom MLP layer:

```python
from megatron.core.transformer import MLP, MLPSubmodules

class CustomMLP(MLP):
    def forward(self, hidden_states, **kwargs):
        # Custom forward logic
        output, bias = super().forward(hidden_states, **kwargs)
        # Post-processing
        return output, bias
```

Example of using custom specs:

```python
from megatron.core.transformer import TransformerBlock, TransformerBlockSubmodules

# Custom block spec with different layer types
block_spec = TransformerBlockSubmodules(
    layer_specs=[
        layer_spec_1,  # First layer uses one config
        layer_spec_2,  # Second layer uses another
        # ... more layers
    ],
    layer_norm=LayerNormImpl,
)

model = TransformerBlock(config=config, spec=block_spec)
```

---

## Selective Recomputation in Layers

The `TransformerLayer` constructor checks `config.recompute_modules` to determine which submodules to checkpoint during selective activation recomputation:

| Module | Recompute Variable | Checkpointing Method |
|---|---|---|
| `"layernorm"` (input) | `recompute_input_layernorm` | `CheckpointWithoutOutput` (output-discarding) |
| `"layernorm"` (pre-MLP) | `recompute_pre_mlp_layernorm` | `CheckpointWithoutOutput` (output-discarding) |
| `"core_attn"` | `checkpoint_core_attention` | Standard checkpoint |
| `"mlp"` | `recompute_mlp` | Standard checkpoint |
| `"moe_act"` | Handled in MoELayer | Output-discarding checkpoint |
| `"shared_experts"` | Handled in MoELayer | Standard checkpoint |

Output-discarding checkpointing saves the forward output temporarily, then discards it and registers a recompute hook on the downstream output tensor. This is more memory-efficient for cheap-to-recompute operations like layernorm.

---

## Fused TP Communication for Inference

When `config.inference_fuse_tp_communication=True`, the `TransformerBlock` sets up fused TP communication across layers. This fuses the reduce-scatter, residual addition, layer norm, and all-gather into a single kernel per linear layer. The setup:

1. Each layer's attention `linear_proj` receives the next layer's MLP fc1 layernorm weights.
2. Each layer's MLP `linear_fc2` receives the next layer's attention QKV layernorm weights.
3. The first layer skips QKV norm and all-gather (weights already available).
4. The last layer uses dummy weights for fc2.

This eliminates separate all-gather/reduce-scatter calls, improving inference throughput at the cost of some memory overhead.
