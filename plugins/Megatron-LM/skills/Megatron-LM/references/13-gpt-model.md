# GPT Model Architecture

This reference documents the GPT (Generative Pre-Trained Transformer) model implementation in Megatron-Core, covering GPTModel, its layer specifications, forward pass pipeline, Multi-Token Prediction (MTP), fine-grained scheduling, and related features.

## Source Files

| Component | Path |
|-----------|------|
| GPTModel | `megatron/core/models/gpt/gpt_model.py` |
| Layer Specs | `megatron/core/models/gpt/gpt_layer_specs.py` |
| MoE Module Specs | `megatron/core/models/gpt/moe_module_specs.py` |
| Fine-Grained Callables | `megatron/core/models/gpt/fine_grained_callables.py` |
| Heterogeneous Layer Specs | `megatron/core/models/gpt/heterogeneous/` |

## GPTModel Class

`GPTModel` extends `LanguageModule` and implements a causal transformer language model for autoregressive text generation and training. It is the primary model class used for GPT-style pre-training, fine-tuning, and inference.

### Constructor

```python
class GPTModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    )
```

### Key Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | `TransformerConfig` | required | Central model configuration |
| `transformer_layer_spec` | `ModuleSpec` | required | Layer implementation specification |
| `vocab_size` | `int` | required | Vocabulary size for embeddings and output |
| `max_sequence_length` | `int` | required | Maximum sequence length for positional embeddings |
| `pre_process` | `bool` | `True` | Include embedding layer (PP first stage) |
| `post_process` | `bool` | `True` | Include output layer (PP last stage) |
| `parallel_output` | `bool` | `True` | Keep outputs split across TP ranks |
| `share_embeddings_and_output_weights` | `bool` | `False` | Tie input embedding and output logit weights |
| `position_embedding_type` | `str` | `'learned_absolute'` | Position embedding strategy |
| `mtp_block_spec` | `ModuleSpec` | `None` | Multi-Token Prediction block spec |
| `vp_stage` | `int` | `None` | Virtual pipeline stage index |

### Internal Architecture

The GPTModel builds these core components during initialization:

1. **Embedding Layer** (`LanguageModelEmbedding`): Created when `pre_process=True` or `mtp_process=True`. Handles word embeddings and position embeddings (learned absolute or none).

2. **Rotary Position Embeddings**: Selected based on `position_embedding_type`:
   - `'rope'` (without MLA): Creates `RotaryEmbedding` with configurable `rotary_percent`, `rotary_base`, and `seq_len_interpolation_factor`.
   - `'yarn'`: Creates `YarnRotaryEmbedding` with additional YaRN-specific parameters (`scaling_factor`, `original_max_position_embeddings`, `beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim`).
   - `'mrope'` (without MLA): Creates `MultimodalRotaryEmbedding` with `mrope_section` configuration.
   - `'none'`: No position embeddings (used with MLA, which has its own decoupled RoPE).

3. **Decoder** (`TransformerBlock`): The main transformer stack. Configured with `transformer_layer_spec`, `pre_process`, `post_process`, and `vp_stage`.

4. **MTP Block** (`MultiTokenPredictionBlock`): Optional multi-token prediction layers for speculative decoding and training with future token prediction. Created when `mtp_block_spec` is provided and `mtp_on_this_rank()` returns True.

5. **Output Layer** (`ColumnParallelLinear`): Projects hidden states to vocabulary logits. Supports weight tying with embeddings and deferred embedding weight gradient computation.

### Model Type

```python
self.model_type = ModelType.encoder_or_decoder
```

This is required for Megatron-Core pipeline parallelism scheduling. GPTModel uses the `encoder_or_decoder` type (despite being decoder-only) because the pipeline engine uses this designation for single-direction models.

## Forward Pass Pipeline

The GPTModel forward pass follows a three-phase pipeline:

```
input_ids --> _preprocess() --> decoder --> _postprocess() --> loss/logits
```

### Phase 1: _preprocess()

```python
def _preprocess(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    decoder_input: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    padding_mask: Optional[Tensor] = None,
)
```

Handles embedding lookup and position embedding computation:

1. **Embedding Lookup**: When `decoder_input` is None and `pre_process=True`, applies `self.embedding(input_ids, position_ids)`.

2. **Padding Mask Handling**: When `padding_mask` is provided and sequence parallel is enabled, scatters the padding mask across the sequence parallel region.

3. **RoPE Computation**: Depending on `position_embedding_type`:
   - For `'rope'`: Computes rotary embeddings with optional flash-decode optimization using precomputed cos/sin. Supports `flash_decode` and `flashinfer_fused_rope` modes during inference.
   - For `'yarn'`: Computes YaRN rotary embeddings with length extrapolation.
   - For `'mrope'`: Computes multimodal rotary embeddings using position IDs and `mrope_section`.

4. **Inference Optimizations**:
   - `sequence_len_offset` tensor for flash decoding with static batching.
   - Clears padding token outputs for dynamic batching with quantization scales.
   - Wraps `decoder_input` in `WrappedTensor` for early garbage collection during inference.

5. **Returns**: A 6-tuple or 7-tuple (with optional `rotary_pos_cos_sin` for flashinfer fused rope).

### Phase 2: Decoder Forward

```python
hidden_states = self.decoder(
    hidden_states=decoder_input,
    attention_mask=attention_mask,
    inference_context=inference_context,
    rotary_pos_emb=rotary_pos_emb,
    rotary_pos_cos=rotary_pos_cos,
    rotary_pos_sin=rotary_pos_sin,
    rotary_pos_cos_sin=rotary_pos_cos_sin,
    packed_seq_params=packed_seq_params,
    sequence_len_offset=sequence_len_offset,
    padding_mask=padding_mask,
    **(extra_block_kwargs or {}),
)
```

The `TransformerBlock` iterates through its layers, applying each `TransformerLayer` (or `MoETransformerLayer`) sequentially. Each layer performs:
- Self-attention with causal masking
- Bias-dropout-add residual connection
- MLP (or MoE) computation
- Bias-dropout-add residual connection

### Phase 3: _postprocess()

```python
def _postprocess(
    self, hidden_states, input_ids, position_ids, labels,
    rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
    mtp_in_postprocess=None, loss_mask=None, decoder_input=None,
    attention_mask=None, inference_params=None, packed_seq_params=None,
    sequence_len_offset=None, runtime_gather_output=None,
    extra_block_kwargs=None, inference_context=None,
)
```

Handles MTP processing, output projection, and loss computation:

1. **MTP Processing (Training)**: When `mtp_in_postprocess` is True and not in inference mode, runs `self.mtp()` to compute multi-token prediction heads. The MTP block processes additional prediction depths and `process_mtp_loss()` handles loss aggregation.

2. **Speculative Decoding Check**: During inference with speculative decoding (`num_speculative_tokens > 0`), MTP is deferred until after token verification. Decoder hidden states are cached in `self._decoder_hidden_states_cache`.

3. **Output Layer**: Projects hidden states through `self.output_layer` (ColumnParallelLinear) to vocabulary logits. Uses shared embedding weights when `share_embeddings_and_output_weights=True`.

4. **MuP Scaling**: Applies `self._scale_logits(logits)` for Maximal Update Parameterization (MuP) when configured.

5. **Inference Optimizations**: For `materialize_only_last_token_logits`, extracts only the last token's logits to reduce memory usage during inference.

6. **Loss Computation**: When `labels` are provided, computes cross-entropy loss via `compute_language_model_loss()`. Otherwise, returns logits transposed to `[batch, seq, hidden]` format.

## Pipeline Parallelism

GPTModel supports pipeline parallelism through the `pre_process` and `post_process` flags:

- **First PP stage** (`pre_process=True`): Contains the embedding layer, computes word and position embeddings.
- **Intermediate PP stages**: Only contain transformer layers. Receive hidden states via `set_input_tensor()`.
- **Last PP stage** (`post_process=True`): Contains the output layer and computes loss/logits.

```python
def set_input_tensor(self, input_tensor: Tensor) -> None:
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
    assert len(input_tensor) == 1
    self.decoder.set_input_tensor(input_tensor[0])
```

The model uses a single input tensor (unlike T5 which uses 2 for encoder-decoder skip connections).

## Position Embedding Types

### Learned Absolute

Default position embedding. Embeddings are learned parameters stored in `LanguageModelEmbedding.position_embeddings`. Suitable for fixed-length sequences.

### RoPE (Rotary Position Embedding)

Applied in `_preprocess()` using `RotaryEmbedding`. Key parameters:

- `rotary_percent`: Fraction of attention head dimension to apply RoPE to (default 1.0).
- `rotary_base`: Base frequency (default 10000).
- `rotary_interleaved`: Whether to interleave rotary dimensions.
- `seq_len_interpolation_factor`: Linear interpolation factor for extending context length.

Flash-decode optimization precomputes cos/sin tensors cached by `max_sequence_length`.

### YaRN (Yet another RoPE extensioN)

Extended RoPE method for length extrapolation. Uses `YarnRotaryEmbedding` with additional parameters:
- `yarn_rotary_scaling_factor`: Scaling factor for extended sequences.
- `yarn_original_max_position_embeddings`: Original training context length.
- `yarn_beta_fast`, `yarn_beta_slow`: Attention magnitude correction factors.
- `yarn_mscale`, `yarn_mscale_all_dim`: Scaling parameters.

### mRoPE (Multimodal RoPE)

Used for multimodal models. Requires `mrope_section` configuration in `TransformerConfig`. Supports different RoPE sections for different modalities.

### None

No position embeddings. Used with Multi-Latent Attention (MLA), which implements its own decoupled RoPE within the attention mechanism.

## Layer Specifications

### Spec System

Megatron-Core uses a spec-based architecture system where `ModuleSpec` objects define which implementation to use for each component. This allows swapping between Transformer Engine (TE) and local Megatron-Core implementations without changing model code.

### get_gpt_layer_with_transformer_engine_submodules()

Creates layer specs using Transformer Engine components (required for FP8 training):

```python
def get_gpt_layer_with_transformer_engine_submodules(
    num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False,
    qk_l2_norm=False, use_te_op_fuser=False,
    use_kitchen=False, use_te_activation_func=False,
    use_kitchen_attention=False, kitchen_attention_backend="sdpa",
    mla_down_proj_fusion=False,
) -> TransformerLayerSubmodules
```

Features:
- Supports MLA with optional `mla_down_proj_fusion` for fused q/kv down-projection and input layernorm.
- Supports Kitchen backend for custom attention implementations.
- Returns different submodules depending on `multi_latent_attention` (MLASelfAttention vs SelfAttention).

### get_gpt_layer_local_submodules()

Creates layer specs using only Megatron-Core modules:

```python
def get_gpt_layer_local_submodules(
    num_experts=None, moe_grouped_gemm=False,
    qk_layernorm=False, multi_latent_attention=False,
    normalization=None, qk_l2_norm=False,
    use_kitchen=False, use_kitchen_attention=False,
    kitchen_attention_backend="sdpa",
) -> TransformerLayerSubmodules
```

Supports `normalization='RMSNorm'` for LLaMA-style architectures.

### get_gpt_layer_with_inference_submodules()

Creates inference-optimized layer specs:

```python
def get_gpt_layer_with_inference_submodules(
    qk_layernorm=False, multi_latent_attention=False,
    qk_l2_norm=False, num_experts=None,
    moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False,
) -> TransformerLayerSubmodules
```

Uses inference-optimized linear layers from TE for faster inference performance.

### Block Spec Construction

```python
def get_gpt_decoder_block_spec(
    config, use_transformer_engine, normalization=None,
    qk_l2_norm=False, vp_stage=None, pp_rank=None,
) -> TransformerBlockSubmodules
```

Constructs the full block spec including:
1. Dense layer specs and MoE layer specs (separate specs for dense and expert layers).
2. MoE layer frequency pattern parsing from `config.moe_layer_freq`:
   - Integer N: One expert layer every N layers.
   - List: Explicit pattern of 0 (dense) and 1 (expert) values.
3. Layer slicing for pipeline parallelism via `get_num_layers_to_build()`.
4. Support for `PipelineParallelLayerLayout` for custom PP layer assignments.

### MTP Block Spec

```python
def get_gpt_mtp_block_spec(
    config, spec, use_transformer_engine,
    vp_stage=None, pp_rank=None,
) -> MultiTokenPredictionBlockSubmodules
```

Constructs MTP block specs. MTP layers use the spec from the last decoder layer. Supports `mtp_use_repeated_layer` to reuse a single layer spec for all MTP depths.

## Fine-Grained Scheduling (1F1B Overlap)

The `fine_grained_callables.py` module decomposes transformer layers into fine-grained callables for overlapping computation and communication in the 1F1B pipeline schedule.

### Layer Decomposition

Each transformer layer is split into up to 5 callables:

| Callable | Type | Description |
|----------|------|-------------|
| `attn` | Computation | Self-attention forward + pre-MLP layernorm + router + dispatch preprocess |
| `dispatch` | Communication | Token dispatch to experts (MoE only) |
| `mlp` / `moe_forward` | Computation | MLP forward (dense) or expert computation (MoE) |
| `combine` | Communication | Token combine from experts + postprocess + bias-dropout-add (MoE only) |
| `None` / `mtp_postprocess` | -- | Placeholder or MTP postprocess |

### Schedule Node Types

- **PreProcessNode**: Handles embedding computation and RoPE setup before transformer layers.
- **PostProcessNode**: Handles output layer computation and loss after transformer layers.
- **TransformerLayerNode**: Base class for layer computation nodes with forward/backward/weight-gradient support.

### Memory Management

```python
def should_free_input(name, is_moe, config, num_local_experts)
```

Controls when input tensors should be freed during the forward pass to reduce peak memory:
- For dense layers: Input is needed during backward pass, never freed.
- For MoE layers: Input can be freed after dispatch (communication) and after expert computation when using FP8/FP4 or certain dispatcher configurations.

### MTP Layer Callables

```python
def build_mtp_layer_callables(layer)
```

Extends the standard transformer layer callables with MTP-specific operations:
- `submodule_mtp_attn_forward`: Handles embedding extraction, concatenation, and attention.
- `submodule_mtp_postprocess_forward`: Post-processes MTP output and accumulates hidden states.

## Deferred Embedding Weight Gradient

When `config.defer_embedding_wgrad_compute=True`, the output layer stores activation and gradient buffers:

```python
self.embedding_activation_buffer = []
self.grad_output_buffer = []
```

These buffers accumulate across micro-batches and compute the embedding weight gradient during the pipeline flush phase, reducing memory fragmentation and improving overlap.

## Sharded State Dict

```python
def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None) -> ShardedStateDict
```

The GPTModel's sharded state dict removes the output layer's `_extra_state` key for backward compatibility with older checkpoints that only stored the weight key.

## Quantization Support

GPTModel supports FP8 and FP4 quantization through `get_quant_config_or_none()`:

```python
for name, module in self.named_modules():
    if hasattr(module, 'finish_init'):
        quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
        module.finish_init(quant_config)
```

This initializes quantization parameters for each module after the model is fully constructed.

## Fine-Grained Activation Offloading

When `config.fine_grained_activation_offloading=True`, the model marks decoder, MTP, and output layer parameters as not offloadable:

```python
def preprocess_for_fine_grained_offloading(self):
    off_interface.init_chunk_handler(
        vp_size=self.config.virtual_pipeline_model_parallel_size,
        vp_stage=self.vp_stage,
        min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
    )
    for param in self.decoder.parameters():
        off_interface.mark_not_offloadable(param)
```

## Schedule Plan

```python
def build_schedule_plan(self, input_ids, position_ids, attention_mask, ...)
```

Builds a `TransformerModelChunkSchedulePlan` for the model chunk, enabling the fine-grained scheduling system to manage preprocessing, transformer layers, and postprocessing as separate schedulable operations.

## Configuration Example

A typical GPT-3 175B configuration:

```python
config = TransformerConfig(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    seq_length=2048,
    bf16=True,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=1,
)

model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
    vocab_size=51200,
    max_sequence_length=2048,
    position_embedding_type='learned_absolute',
)
```

A LLaMA-style configuration with RoPE:

```python
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    seq_length=4096,
    bf16=True,
    position_embedding_type='rope',
    rotary_percent=1.0,
    normalization='RMSNorm',
    add_bias_linear=False,
    gated_linear_unit=True,
)

model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_layer_local_spec(normalization='RMSNorm'),
    vocab_size=32000,
    max_sequence_length=4096,
    position_embedding_type='rope',
    share_embeddings_and_output_weights=False,
)
```

## Related Documentation

- **14-bert-model.md**: BERT encoder model architecture
- **15-t5-model.md**: T5 encoder-decoder model architecture
- **20-moe-architecture.md**: MoE layer architecture for sparse models
- **19-hybrid-mimo-models.md**: Hybrid model architecture for Mamba/Transformer mixtures
