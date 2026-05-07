# T5 Model Architecture

This reference documents the T5 (Text-to-Text Transfer Transformer) model implementation in Megatron-Core, covering T5Model, its encoder-decoder architecture, cross-attention mechanism, position embeddings, and pipeline parallelism support.

## Source Files

| Component | Path |
|-----------|------|
| T5Model | `megatron/core/models/T5/t5_model.py` |
| T5 Spec | `megatron/core/models/T5/t5_spec.py` |

## T5Model Class

`T5Model` extends `LanguageModule` and implements an encoder-decoder transformer model. Unlike GPT (decoder-only) and BERT (encoder-only), T5 combines both an encoder and decoder with cross-attention connections between them.

### Constructor

```python
class T5Model(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        encoder_config: TransformerConfig,
        transformer_encoder_layer_spec: ModuleSpec,
        transformer_decoder_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'relative'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection = None,
    )
```

### Key Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | `TransformerConfig` | required | Decoder configuration |
| `encoder_config` | `TransformerConfig` | required | Separate encoder configuration |
| `transformer_encoder_layer_spec` | `ModuleSpec` | required | Encoder layer implementation spec |
| `transformer_decoder_layer_spec` | `ModuleSpec` | required | Decoder layer implementation spec |
| `vocab_size` | `int` | required | Vocabulary size (shared between encoder and decoder) |
| `max_sequence_length` | `int` | required | Maximum sequence length |
| `position_embedding_type` | `str` | `'learned_absolute'` | One of `'learned_absolute'`, `'rope'`, `'relative'` |
| `relative_attention_num_buckets` | `int` | `32` | Number of buckets for relative position embeddings |
| `relative_attention_max_distance` | `int` | `128` | Maximum distance for relative position embeddings |
| `add_encoder` | `bool` | `True` | Create the encoder (PP control) |
| `add_decoder` | `bool` | `True` | Create the decoder (PP control) |

### Cross-Attention Flag

```python
self.xattn_needed = True
```

This flag tells pipeline scheduling (`schedules.py`) that the T5 model has a skip connection between the encoder output and decoder input. Both encoder and decoder tensors are required for correct backpropagation. This is critical for pipeline schedule correctness.

### Internal Architecture

T5Model builds these components during initialization:

1. **Embedding Layer** (`LanguageModelEmbedding`): Shared between encoder and decoder. Created when `pre_process=True`. Supports learned absolute, RoPE, and relative position embeddings.

2. **Rotary Position Embeddings** (`RotaryEmbedding`): Created when `position_embedding_type='rope'`. Configured with `kv_channels`, `rotary_percent`, `rotary_interleaved`, and context parallel group.

3. **Relative Position Embeddings**: Created when `position_embedding_type='relative'`:
   - `encoder_relative_pos_emb`: Bidirectional relative position embedding for encoder self-attention.
   - `decoder_relative_pos_emb`: Unidirectional (causal) relative position embedding for decoder self-attention.
   Both use `RelativePositionEmbedding` with configurable `num_buckets` and `max_distance`.

4. **Encoder** (`TransformerBlock`): The bidirectional encoder stack, created when `add_encoder=True`. Uses `encoder_config` (which may differ from `config` for the decoder).

5. **Decoder** (`TransformerBlock`): The causal decoder stack with cross-attention, created when `add_decoder=True`. Uses the main `config`.

6. **LM Head** (`T5LMHead`): Output projection layer, created when `post_process=True`.

### Separate Encoder Configuration

T5Model supports separate `encoder_config` and decoder `config`:

```python
# Encoder may have different num_layers, hidden_size, etc.
self.encoder = TransformerBlock(
    config=self.encoder_config,  # Separate encoder config
    spec=encoder_spec,
    ...
)
self.decoder = TransformerBlock(
    config=self.config,  # Decoder config
    spec=decoder_spec,
    ...
)
```

This enables asymmetric architectures where the encoder and decoder have different depths, widths, or attention configurations.

## T5LMHead

```python
class T5LMHead(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        parallel_output: bool,
        vocab_size: int,
        pre_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    )
```

The T5 LM head is a ColumnParallelLinear layer that projects decoder hidden states to vocabulary logits:

```python
def forward(self, hidden_states: Tensor, word_embeddings_weight: Tensor) -> Tensor:
    logits, _ = self.output_layer(hidden_states, weight=word_embeddings_weight)
    return logits
```

When `share_embeddings_and_output_weights=True`, the LM head reuses the embedding weight matrix instead of maintaining a separate parameter. The weight is passed explicitly via the `word_embeddings_weight` argument.

## Forward Pass

```python
def forward(
    self,
    encoder_input_ids: Tensor,
    decoder_input_ids: Tensor,
    encoder_attn_mask: Tensor,
    decoder_attn_mask: Tensor,
    encoder_decoder_attn_mask: Tensor,
    lm_labels: Tensor = None,
    encoder_hidden_states: Tensor = None,
    output_encoder_hidden_only: bool = False,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
) -> Tensor
```

### Forward Pipeline

```
encoder_input_ids --> encoder_embedding --> encoder --> encoder_hidden_states
                                                          |
decoder_input_ids --> decoder_embedding --> decoder ------+
                          |                               |
                     cross-attention    <--- encoder_hidden_states
                          |
                     lm_head --> logits --> loss
```

### Encoder Forward

1. **Position IDs**: `t5_position_ids(encoder_input_ids)` creates sequential position IDs.

2. **Encoder Embedding** (when `pre_process=True`):
   ```python
   encoder_input = self.embedding(
       input_ids=encoder_input_ids, position_ids=encoder_position_ids
   )
   ```

3. **RoPE** (when `position_embedding_type='rope'`):
   ```python
   rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
       inference_context, self.encoder, encoder_input, self.config, packed_seq_params
   )
   rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
   ```

4. **Relative Position Embeddings** (when `position_embedding_type='relative'`):
   ```python
   attention_bias = self.encoder_relative_pos_emb(query_seq_length, key_seq_length)
   ```
   The attention bias is scattered to tensor parallel ranks:
   - Reshape `[1, num_heads, seq_q, seq_kv]` -> `[1, seq_q, seq_kv, num_heads]`.
   - Scatter along the last dimension (heads) to TP ranks.
   - Revert to `[1, num_heads_per_tp, seq_q, seq_kv]`.

5. **Encoder Execution**:
   ```python
   encoder_hidden_states = self.encoder(
       hidden_states=encoder_input,
       attention_mask=encoder_attn_mask,
       inference_context=inference_context,
       rotary_pos_emb=rotary_pos_emb,
       attention_bias=encoder_attention_bias_parallel,
   )
   ```

6. **Early Exit**: If `output_encoder_hidden_only=True` or `not add_decoder`, returns encoder hidden states directly.

### Decoder Forward

1. **Decoder Position IDs**: `t5_position_ids(decoder_input_ids)`.

2. **Decoder Embedding** (when `pre_process=True`):
   ```python
   decoder_input = self.embedding(
       input_ids=decoder_input_ids, position_ids=decoder_position_ids
   )
   ```

3. **Decoder Position Embeddings**: Same RoPE/relative logic as encoder, but using `self.decoder_relative_pos_emb` (unidirectional).

4. **Decoder Execution** with cross-attention:
   ```python
   decoder_hidden_states = self.decoder(
       hidden_states=decoder_input,
       attention_mask=decoder_attn_mask,
       context=encoder_hidden_states,           # Encoder output for cross-attention
       context_mask=encoder_decoder_attn_mask,   # Cross-attention mask
       inference_context=inference_context,
       rotary_pos_emb=rotary_pos_emb,
       attention_bias=decoder_attention_bias_parallel,
   )
   ```

5. **Output Processing** (when `post_process=True`):
   ```python
   output_weight = self.shared_embedding_or_output_weight() if share_weights else None
   lm_logits = self.lm_head(decoder_hidden_states, word_embeddings_weight=output_weight)
   ```

6. **Loss or Logits**:
   - When `lm_labels is None`: Returns logits `[b, s, h]`.
   - When `lm_labels` provided: Returns cross-entropy loss.

## Pipeline Parallelism

T5Model has more complex pipeline parallelism than GPT/BERT due to the encoder-decoder architecture.

### set_input_tensor()

```python
def set_input_tensor(self, input_tensor):
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]

    if self.add_encoder and self.add_decoder:
        # Same stage has both encoder and decoder
        assert len(input_tensor) == 1
        self.encoder.set_input_tensor(input_tensor[0])
    elif self.add_encoder:
        # Encoder-only stage
        assert len(input_tensor) == 1
        self.encoder.set_input_tensor(input_tensor[0])
    elif self.add_decoder:
        # Decoder stage receives encoder output as skip connection
        if len(input_tensor) == 2:
            self.decoder.set_input_tensor(input_tensor[0])
            self.encoder_hidden_state = input_tensor[1]  # Encoder skip connection
        elif len(input_tensor) == 1:
            self.decoder.set_input_tensor(None)
            self.encoder_hidden_state = input_tensor[0]  # Encoder output only
```

Key differences from GPT/BERT:
- The decoder stage can receive 2 tensors: [decoder_input, encoder_output].
- `encoder_hidden_state` is stored as a member variable for the decoder's cross-attention.
- This enables placing encoder and decoder on different PP stages.

### Pipeline Stage Configurations

| Configuration | Description |
|---------------|-------------|
| Both on same stage | `add_encoder=True`, `add_decoder=True`. Standard single-GPU or tensor-parallel setup. |
| Separate stages | Encoder on earlier PP stages, decoder on later PP stages. Uses `add_encoder`/`add_decoder` to control which is created. |
| Mixed stages | Some PP stages have both, others have only one. Requires careful scheduling. |

## Position Embedding Types

### Learned Absolute

Default T5 behavior. Position embeddings are stored in `LanguageModelEmbedding.position_embeddings`:

```python
if position_embedding_type == "learned_absolute":
    self.position_embeddings = self.embedding.position_embeddings
```

The position embeddings are tracked separately for gradient allreduce in `finalize_model_grads._allreduce_position_embedding_grads`.

### Rotary (RoPE)

Applied via `RotaryEmbedding` with standard parameters. Both encoder and decoder use the same rotary embedding instance.

### Relative

T5's native position encoding method. Uses `RelativePositionEmbedding` with bucketed distances:

- **Encoder**: `bidirectional=True` -- relative positions can be positive or negative.
- **Decoder**: `bidirectional=False` -- only causal (non-negative) positions.
- `relative_attention_num_buckets`: Number of discrete position buckets (default 32).
- `relative_attention_max_distance`: Maximum distance before clipping (default 128).

The attention bias is scattered to TP ranks since different TP ranks handle different attention heads.

## T5 Layer Specifications

### Encoder Layer Spec (TE)

```python
def encoder_model_with_transformer_engine_default_spec() -> ModuleSpec
```

Encoder layer structure:
- Self-attention with padding mask type
- TELayerNormColumnParallelLinear for QKV
- TEDotProductAttention for core attention
- TERowParallelLinear for output projection
- MLP with TELayerNormColumnParallelLinear (fc1) and TERowParallelLinear (fc2)
- Bias-dropout-add for both attention and MLP

### Decoder Layer Spec (TE)

```python
def decoder_model_with_transformer_engine_default_spec() -> ModuleSpec
```

Decoder layer structure (encoder + cross-attention + MLP):
- Self-attention with causal mask type
- Pre-cross-attention layer norm (TENorm)
- Cross-attention with padding mask type:
  - Separate Q projection: `TEColumnParallelLinear`
  - Separate K,V projection: `TEColumnParallelLinear`
  - Core attention: `TEDotProductAttention`
  - Output projection: `TERowParallelLinear`
- MLP with standard fc1/fc2

### Local Specs

```python
def encoder_model_with_local_spec() -> ModuleSpec
def decoder_model_with_local_spec() -> ModuleSpec
```

Local specs use Megatron-Core components (ColumnParallelLinear, RowParallelLinear, DotProductAttention) with explicit layer norms and `sharded_state_dict_keys_map` for checkpoint compatibility.

### Block Specs

```python
def get_t5_encoder_with_transformer_engine_block_spec(num_layers) -> TransformerBlockSubmodules
def get_t5_decoder_with_transformer_engine_block_spec(num_layers) -> TransformerBlockSubmodules
def get_t5_encoder_with_local_block_spec(num_layers) -> TransformerBlockSubmodules
def get_t5_decoder_with_local_block_spec(num_layers) -> TransformerBlockSubmodules
```

Each creates a `TransformerBlockSubmodules` with repeated layer specs and a final layer norm.

## Helper Functions

### t5_extended_attention_mask()

```python
def t5_extended_attention_mask(attention_mask_list: List[Tensor]) -> List[Tensor]
```

Converts attention masks from `[b, s, s]` to `[b, 1, s, s]` by unsqueezing along dimension 1. Handles None masks by returning None unchanged.

### t5_position_ids()

```python
def t5_position_ids(token_ids: Tensor) -> Tensor
```

Creates sequential position IDs `[0, 1, 2, ..., seq_length-1]` expanded to match the batch size. Simple positional indexing matching the original T5 implementation.

## Sharded State Dict

```python
def sharded_state_dict(
    self, prefix='', sharded_offsets=(), metadata=None,
) -> ShardedStateDict
```

T5Model's sharded state dict delegates to the parent `LanguageModule` implementation. Some layers (output, embedding) are shared between encoder and decoder. The `replica_id` for these shared layers is set to ensure only one instance with `replica_id (0, 0, 0)`.

## Embedding Weight Sharing

```python
def shared_embedding_or_output_weight(self) -> Tensor:
    if self.pre_process:
        return self.embedding.word_embeddings.weight
    elif self.post_process:
        return self.lm_head.output_layer.weight
    return None
```

When `share_embeddings_and_output_weights=True`, the input embeddings, encoder output, and decoder output all share the same weight matrix. The `setup_embeddings_and_output_layer()` method from `LanguageModule` handles the weight tying across pipeline stages.

## Configuration Example

Standard T5-Base configuration:

```python
encoder_config = TransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    seq_length=512,
)

decoder_config = TransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    seq_length=512,
)

model = T5Model(
    config=decoder_config,
    encoder_config=encoder_config,
    transformer_encoder_layer_spec=get_t5_encoder_with_transformer_engine_block_spec(
        num_layers=12
    ),
    transformer_decoder_layer_spec=get_t5_decoder_with_transformer_engine_block_spec(
        num_layers=12
    ),
    vocab_size=32128,
    max_sequence_length=512,
    position_embedding_type='relative',
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
)
```

T5-Large with RoPE:

```python
encoder_config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
)

model = T5Model(
    config=encoder_config,
    encoder_config=encoder_config,
    transformer_encoder_layer_spec=get_t5_encoder_with_transformer_engine_block_spec(24),
    transformer_decoder_layer_spec=get_t5_decoder_with_transformer_engine_block_spec(24),
    vocab_size=32128,
    max_sequence_length=512,
    position_embedding_type='rope',
    rotary_percent=1.0,
)
```

## Key Differences from GPTModel and BertModel

| Feature | T5 | GPT | BERT |
|---------|-----|-----|------|
| Architecture | Encoder-decoder | Decoder-only | Encoder-only |
| Cross-attention | Yes | No | No |
| Separate configs | encoder_config + config | config only | config only |
| PP input tensors | 1 or 2 | 1 | 1 |
| Position embeddings | learned_absolute, rope, relative | learned_absolute, rope, yarn, mrope, none | learned_absolute, rope |
| Binary head | No | No | Yes |
| Pooler | No | No | Yes |
| LM head type | T5LMHead (ColumnParallelLinear) | ColumnParallelLinear | BertLMHead (dense + gelu + layernorm) |
| xattn_needed | True | False | False |

## Related Documentation

- **13-gpt-model.md**: GPT decoder-only model architecture
- **14-bert-model.md**: BERT encoder-only model architecture
- **20-moe-architecture.md**: MoE layer architecture for sparse models
