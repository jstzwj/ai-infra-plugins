# BERT Model Architecture

This reference documents the BERT (Bidirectional Encoder Representations from Transformers) model implementation in Megatron-Core, covering BertModel, its encoder architecture, masked language modeling loss, next sentence prediction, and related components.

## Source Files

| Component | Path |
|-----------|------|
| BertModel | `megatron/core/models/bert/bert_model.py` |
| BertLMHead | `megatron/core/models/bert/bert_lm_head.py` |
| Pooler | `megatron/core/models/bert/pooler.py` |
| Layer Specs | `megatron/core/models/bert/bert_layer_specs.py` |

## BertModel Class

`BertModel` extends `LanguageModule` and implements a bidirectional transformer encoder for pre-training tasks including Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)/Sentence Order Prediction (SOP).

### Constructor

```python
class BertModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        num_tokentypes: int,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        add_binary_head=True,
        return_embeddings=False,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    )
```

### Key Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | `TransformerConfig` | required | Central model configuration |
| `num_tokentypes` | `int` | required | Number of token type IDs (2 when `add_binary_head=True`, 0 otherwise) |
| `transformer_layer_spec` | `ModuleSpec` | required | Layer implementation specification |
| `vocab_size` | `int` | required | Vocabulary size |
| `max_sequence_length` | `int` | required | Maximum sequence length for positional embeddings |
| `pre_process` | `bool` | `True` | Include embedding layer (PP first stage) |
| `post_process` | `bool` | `True` | Include output layer (PP last stage) |
| `add_binary_head` | `bool` | `True` | Include NSP/SOP binary classification head |
| `return_embeddings` | `bool` | `False` | Return pooled embeddings instead of logits/loss |
| `position_embedding_type` | `str` | `'learned_absolute'` | Position embedding strategy |
| `share_embeddings_and_output_weights` | `bool` | `False` | Tie input and output weights |

### Internal Architecture

BertModel builds these components during initialization:

1. **Embedding Layer** (`LanguageModelEmbedding`): Created when `pre_process=True`. Handles word embeddings, position embeddings (learned absolute or RoPE), and token type embeddings (`num_tokentypes`).

2. **Rotary Position Embeddings** (`RotaryEmbedding`): Created when `position_embedding_type='rope'`. Configured with `rotary_percent`, `rotary_interleaved`, and `seq_len_interpolation_factor`.

3. **Encoder** (`TransformerBlock`): The main bidirectional transformer stack. Unlike GPT, BERT uses padding attention masks rather than causal masks.

4. **LM Head** (`BertLMHead`): Dense layer followed by GELU activation and layer normalization. Created when `post_process=True`.

5. **Output Layer** (`ColumnParallelLinear`): Projects from hidden size to vocabulary size. Supports weight tying with embeddings and MuP initialization.

6. **Binary Head** and **Pooler**: Created when `post_process=True` and `add_binary_head=True`:
   - `Pooler`: Takes the first token's hidden state (CLS) and projects through a dense layer with Tanh activation.
   - `binary_head`: A simple linear layer (hidden_size -> 2) for NSP/SOP classification.

### Model Type

```python
self.model_type = ModelType.encoder_or_decoder
```

## Attention Mask Handling

### Sanity Check and Dimensions

```python
def _sanity_check_attention_and_get_attn_mask_dimension(self) -> str
```

Returns the attention mask dimension format based on the TE version and attention backend:

| Condition | Dimension | Mask Type |
|-----------|-----------|-----------|
| MCore local attention | `[b, 1, s, s]` | arbitrary |
| TE >= 1.10 | `[b, 1, 1, s]` | padding |
| TE >= 1.7, flash/fused | `[b, 1, 1, s]` | padding |
| TE >= 1.7, unfused | `[b, 1, s, s]` | arbitrary |
| TE < 1.7 | `[b, 1, s, s]` | padding |

### bert_extended_attention_mask()

```python
def bert_extended_attention_mask(self, attention_mask: Tensor) -> Tensor
```

Converts a binary attention mask of shape `[batch, seq_len]` into the extended format required by the attention mechanism:

**For `[b, 1, s, s]` format:**
1. Expand mask to `[b, 1, s]` and `[b, s, 1]`.
2. Multiply to get pairwise mask `[b, s, s]`.
3. Unsqueeze to `[b, 1, s, s]`.
4. Convert to binary: `extended_mask < 0.5` (True = masked, False = valid).

**For `[b, 1, 1, s]` format:**
1. Unsqueeze twice to `[b, 1, 1, s]`.
2. Convert to binary.

### bert_position_ids()

```python
def bert_position_ids(self, token_ids) -> Tensor
```

Creates position IDs as a simple range `[0, 1, 2, ..., seq_length-1]` expanded to match the batch size.

## Forward Pass

```python
def forward(
    self,
    input_ids: Tensor,
    attention_mask: Tensor,
    tokentype_ids: Tensor = None,
    lm_labels: Tensor = None,
    inference_context=None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
)
```

### Forward Pipeline

```
input_ids --> embedding --> encoder --> pooler --> lm_head --> output_layer --> loss/logits
                 |                        |
           tokentype_ids             binary_head
```

1. **Attention Mask Extension**: Convert `[b, s]` mask to extended format.

2. **Position IDs**: Generate position IDs from input token IDs.

3. **Embedding**: When `pre_process=True`:
   ```python
   encoder_input = self.embedding(
       input_ids=input_ids, position_ids=position_ids, tokentype_ids=tokentype_ids
   )
   ```

4. **RoPE Computation** (when `position_embedding_type='rope'`):
   ```python
   rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
       inference_context, self.encoder, encoder_input, self.config
   )
   rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
   ```

5. **Encoder Forward**:
   ```python
   hidden_states = self.encoder(
       hidden_states=encoder_input,
       attention_mask=extended_attention_mask,
       inference_context=inference_context,
       rotary_pos_emb=rotary_pos_emb,
   )
   ```

6. **Early Return** (when `not post_process`): Returns raw hidden states.

7. **Pooling** (when `add_binary_head`): Extracts CLS token representation.
   ```python
   pooled_output = self.pooler(hidden_states, 0)  # index 0 = CLS token
   ```

8. **Embedding Return Mode** (when `return_embeddings=True`):
   - Transposes hidden states to `[batch, seq, hidden]`.
   - Computes mean of non-padding token embeddings per sample.
   - Returns averaged embeddings of shape `[batch, hidden_size]`.

9. **LM Head and Output Layer**:
   ```python
   hidden_states_after_lm_head = self.lm_head(hidden_states=hidden_states)
   logits, _ = self.output_layer(hidden_states_after_lm_head, weight=output_weight)
   ```

10. **Binary Head** (when `binary_head` is not None):
    ```python
    binary_logits = self.binary_head(pooled_output)
    ```

11. **Return Values**:
    - When `lm_labels is None`: Returns `(logits [b, s, h], binary_logits)`.
    - When `lm_labels` provided: Returns `(mlm_loss, binary_logits)`.

## BertLMHead

```python
class BertLMHead(MegatronModule):
    def __init__(self, hidden_size: int, config: TransformerConfig)
```

The BERT MLM head applies a three-step transformation:

1. **Dense linear projection**: `hidden_size -> hidden_size`
2. **GELU activation**: Non-linear transformation
3. **Layer normalization**: Standard layer norm

```python
def forward(self, hidden_states: Tensor) -> Tensor:
    hidden_states = self.dense(hidden_states)
    hidden_states = self.gelu(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    return hidden_states
```

This matches the original BERT paper's MLM head design, transforming hidden states before the final vocabulary projection.

## Pooler

```python
class Pooler(MegatronModule):
    def __init__(self, hidden_size, init_method, config, sequence_parallel)
```

The pooler extracts the CLS token representation and projects it:

```python
def forward(self, hidden_states, sequence_index=0):
    # hidden_states: [s, b, h] (sequence-first format)
    first_token_tensor = hidden_states[sequence_index]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = torch.tanh(pooled_output)
    return pooled_output
```

## Pipeline Parallelism

```python
def set_input_tensor(self, input_tensor: Tensor) -> None:
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
    assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
    self.encoder.set_input_tensor(input_tensor[0])
```

BERT uses the same PP scheme as GPT: a single input tensor for pipeline communication. The encoder receives hidden states from the previous pipeline stage.

## Embedding and Output Weight Sharing

When `share_embeddings_and_output_weights=True`, the output layer reuses the embedding weight matrix:

```python
output_weight = self.shared_embedding_or_output_weight()
logits, _ = self.output_layer(hidden_states_after_lm_head, weight=output_weight)
```

This is inherited from `LanguageModule.setup_embeddings_and_output_layer()` which handles the weight tying mechanism across pipeline stages.

## MuP (Maximal Update Parameterization) Support

When `config.use_mup=True`:
- The output layer uses `config.embedding_init_method` instead of `config.init_method` for initialization (unless weights are shared).
- Logits are scaled by `self._scale_logits()` inherited from `LanguageModule`.

## Return Embeddings Mode

When `return_embeddings=True`, the model returns averaged token embeddings instead of logits/loss:

```python
embeddings = torch.transpose(hidden_states, 0, 1)  # [s, b, h] -> [b, s, h]
masks = torch.sum(attention_mask, dim=1)
output = torch.zeros(size=(embeddings.shape[0], embeddings.shape[2]), ...)
for i, (embedding, mask) in enumerate(zip(embeddings, masks)):
    output[i, :] = torch.mean(embedding[1 : mask - 1], dim=0)  # Exclude CLS and SEP
```

This mode is useful for extracting sentence embeddings for downstream tasks like semantic similarity or retrieval. It requires `post_process=True` and `add_binary_head=True`.

## BERT Layer Specifications

### TE-based Spec

```python
def get_bert_layer_with_transformer_engine_spec(
    num_tokentypes: int = 0,
) -> ModuleSpec
```

Creates BERT layer specs using Transformer Engine components with padding attention mask type:

- Self-attention with `AttnMaskType.padding`
- TE LayerNorm + ColumnParallelLinear for QKV projection
- TE DotProductAttention for core attention
- TE RowParallelLinear for output projection
- MLP with TE LayerNorm + ColumnParallelLinear (fc1) and RowParallelLinear (fc2)

### Local Spec

```python
def get_bert_layer_local_spec(
    num_tokentypes: int = 0,
) -> ModuleSpec
```

Creates BERT layer specs using Megatron-Core components with arbitrary attention mask type and explicit layer norms.

## BERT Pre-Training Tasks

### Masked Language Modeling (MLM)

The primary pre-training objective. Random tokens are replaced with `[MASK]` tokens, and the model predicts the original tokens. The loss is computed via `compute_language_model_loss(lm_labels, logits)` which applies cross-entropy loss only on masked positions.

### Next Sentence Prediction (NSP) / Sentence Order Prediction (SOP)

The binary classification head (`binary_head`) predicts whether sentence B follows sentence A (NSP) or whether the sentence order is correct (SOP). The binary logits are returned alongside the MLM loss:

```python
loss = self.compute_language_model_loss(lm_labels, logits)
binary_logits = self.binary_head(pooled_output)
return loss, binary_logits
```

## Configuration Example

Standard BERT-Base configuration:

```python
config = TransformerConfig(
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    seq_length=512,
    bf16=True,
)

model = BertModel(
    config=config,
    num_tokentypes=2,  # Segment A and B
    transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
    vocab_size=30522,
    max_sequence_length=512,
    add_binary_head=True,
    position_embedding_type='learned_absolute',
)
```

BERT-Large configuration:

```python
config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
    seq_length=512,
    bf16=True,
)

model = BertModel(
    config=config,
    num_tokentypes=2,
    transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
    vocab_size=30522,
    max_sequence_length=512,
)
```

## Key Differences from GPTModel

| Feature | BERT | GPT |
|---------|------|-----|
| Attention type | Bidirectional (padding mask) | Causal (causal mask) |
| Position embedding | learned_absolute or rope | learned_absolute, rope, yarn, mrope, none |
| LM Head | BertLMHead (dense + gelu + layernorm) | Direct output layer |
| Binary head | Yes (NSP/SOP) | No |
| Pooler | Yes (CLS token) | No |
| Token types | Yes (segment embeddings) | No |
| Return embeddings | Yes | No |
| MTP support | No | Yes |
| Flash decode | No | Yes |

## Related Documentation

- **13-gpt-model.md**: GPT decoder-only model architecture
- **15-t5-model.md**: T5 encoder-decoder model architecture
- **20-moe-architecture.md**: MoE layer architecture
