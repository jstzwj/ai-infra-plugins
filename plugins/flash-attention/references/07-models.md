# FlashAttention Models Reference

This document provides comprehensive reference documentation for all model implementations in the FlashAttention library. These models are optimized implementations that leverage FlashAttention kernels and fused operations for maximum training throughput.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [GPT Model (gpt.py)](#gpt-model)
3. [BERT Model (bert.py)](#bert-model)
4. [LLaMA Model (llama.py)](#llama-model)
5. [GPT-NeoX Model (gpt_neox.py)](#gpt-neox-model)
6. [GPT-J Model (gptj.py)](#gpt-j-model)
7. [OPT Model (opt.py)](#opt-model)
8. [Falcon Model (falcon.py)](#falcon-model)
9. [Vision Transformer (vit.py)](#vision-transformer)
10. [Baichuan Model (baichuan.py)](#baichuan-model)
11. [BigCode Model (bigcode.py)](#bigcode-model)
12. [BTLM Model (btlm.py)](#btlm-model)
13. [Shared Design Patterns](#shared-design-patterns)
14. [Configuration Guide](#configuration-guide)
15. [Tensor Parallel Utilities](#tensor-parallel-utilities)
16. [State Dict Remapping](#state-dict-remapping)

---

## Architecture Overview

All models in FlashAttention follow a unified architecture pattern based on the `GPT2Config` from HuggingFace Transformers. Rather than implementing each model from scratch, the library provides:

1. **Configuration converters** that map model-specific configs (e.g., `LlamaConfig`, `FalconConfig`) to a unified `GPT2Config` with additional custom fields
2. **State dict remappers** that convert pretrained weights from HuggingFace format to the FlashAttention internal format
3. **Optimized building blocks** (`Block`, `MHA`, `Mlp`, `GatedMlp`) that leverage fused CUDA kernels and FlashAttention

This approach allows all supported models to share the same optimized GPT backbone while maintaining compatibility with their original pretrained weights.

### Supported Models Summary

| Model | Module | Architecture Type | Attention | Position Embedding | Activation |
|-------|--------|-------------------|-----------|-------------------|------------|
| GPT-2 | `gpt.py` | Decoder-only | Multi-head | Learned absolute | GELU |
| BERT | `bert.py` | Encoder | Multi-head | Learned absolute | GELU |
| LLaMA | `llama.py` | Decoder-only | GQA | Rotary (interleaved) | SwiGLU |
| GPT-NeoX | `gpt_neox.py` | Decoder-only | Multi-head | Rotary | GELU |
| GPT-J | `gptj.py` | Decoder-only | Multi-head (parallel) | Rotary (interleaved) | GELU |
| OPT | `opt.py` | Decoder-only | Multi-head | Learned absolute | GELU |
| Falcon | `falcon.py` | Decoder-only | MQA | Rotary | GELU |
| ViT | `vit.py` | Encoder | Multi-head | Learned absolute | GELU |
| Baichuan | `baichuan.py` | Decoder-only | Multi-head | Rotary/ALiBi | SwiGLU |
| BigCode | `bigcode.py` | Decoder-only | MQA | Learned absolute | GELU |
| BTLM | `btlm.py` | Decoder-only | Multi-head | ALiBi | GELU |

---

## GPT Model

**File:** `flash_attn/models/gpt.py`

The GPT model is the central model implementation in FlashAttention. It serves as the unified backbone for all decoder-only transformer models (GPT-2, GPT-NeoX, GPT-J, OPT, Falcon, LLaMA, Baichuan, BigCode, BTLM).

### Classes

#### `GPTPreTrainedModel`

Abstract base class for GPT models. Handles weight initialization and pretrained model loading.

```python
class GPTPreTrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs)
```

**Parameters:**
- `config` (GPT2Config): Must be a `GPT2Config` instance. Extended with custom fields for FlashAttention features.

**Class Methods:**

##### `from_pretrained`

```python
@classmethod
def from_pretrained(
    cls,
    model_name,
    config,
    *args,
    strict=True,
    device=None,
    dtype=None,
    world_size=1,
    rank=0,
    **kwargs,
)
```

Loads a pretrained model from HuggingFace Hub or local checkpoint.

**Parameters:**
- `model_name` (str): Model identifier (e.g., `"gpt2"`, `"facebook/opt-350m"`, `"meta-llama/Llama-2-7b-hf"`)
- `config` (GPT2Config): Model configuration
- `strict` (bool): Whether to strictly enforce weight key matching
- `device` (torch.device): Target device
- `dtype` (torch.dtype): Target dtype
- `world_size` (int): Number of processes for tensor parallel
- `rank` (int): Process rank for tensor parallel

**Supported model identifiers:**
- `"gpt2"`, `"gpt2-medium"`, `"gpt2-large"`, `"gpt2-xl"` -- GPT-2 models
- `"facebook/opt-*"` -- OPT models
- `"EleutherAI/gpt-j-*"` -- GPT-J models
- `"EleutherAI/gpt-neox-*"`, `"EleutherAI/pythia-*"` -- GPT-NeoX/Pythia models
- `"tiiuae/falcon-*"` -- Falcon models
- `"meta-llama/Llama-*"` -- LLaMA models
- `"bigcode/*"`, `"WizardLM/*"` -- BigCode/StarCoder models

#### `GPTModel`

The core transformer model without the language modeling head.

```python
class GPTModel(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None)
```

**Parameters:**
- `config` (GPT2Config): Model configuration
- `process_group` (ProcessGroup, optional): For tensor parallelism
- `device` (torch.device, optional): Target device
- `dtype` (torch.dtype, optional): Target dtype

**Supported activation functions:**
- `"gelu"`, `"gelu_new"`, `"gelu_fast"`, `"gelu_approx"`, `"gelu_pytorch_tanh"` -- GELU variants
- `"relu"` -- Rectified Linear Unit
- `"sqrelu"` -- Squared ReLU (from Primer paper)
- `"glu"` -- Gated Linear Unit with sigmoid
- `"swiglu"` -- SwiGLU (SiLU-gated, used in LLaMA)
- `"geglu"` -- GELU-gated linear unit

**Key attributes:**
- `embeddings` -- `GPT2Embeddings` or `ParallelGPT2Embeddings`
- `layers` -- `nn.ModuleList` of `Block` or `ParallelBlock` modules
- `ln_f` -- Final layer norm or RMS norm
- `drop_f` -- Dropout before final norm (prenorm architecture)

**Methods:**

##### `forward`

```python
def forward(self, input_ids, position_ids=None, inference_params=None)
```

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs, shape `(batch, seqlen)`
- `position_ids` (torch.Tensor, optional): Position IDs, shape `(batch, seqlen)`
- `inference_params` (InferenceParams, optional): For autoregressive generation with KV cache

**Returns:**
- `hidden_states` (torch.Tensor): Shape `(batch, seqlen, hidden_size)`

##### `allocate_inference_cache`

```python
def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs)
```

Allocates KV cache for autoregressive generation. Returns a dict mapping layer index to the layer's inference cache.

#### `GPTLMHeadModel`

GPT model with a language modeling head for next-token prediction.

```python
class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin):
    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None)
```

**Additional attributes:**
- `transformer` -- The `GPTModel` backbone
- `lm_head` -- `nn.Linear` or `ColumnParallelLinear` for vocabulary projection

**Methods:**

##### `forward`

```python
def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0)
```

**Parameters:**
- `input_ids` (torch.Tensor): Shape `(batch, seqlen)`
- `position_ids` (torch.Tensor, optional): Position IDs
- `inference_params` (InferenceParams, optional): For generation with KV cache
- `num_last_tokens` (int): If > 0, only return logits for the last N tokens

**Returns:**
- `CausalLMOutput` named tuple with `logits` field, shape `(batch, seqlen, vocab_size)` or `(batch, num_last_tokens, vocab_size)`

### Helper Functions

#### `create_mixer_cls`

```python
def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None)
```

Creates the attention mixer class (MHA or ParallelMHA) based on config.

**Config parameters read:**
- `head_dim` -- Head dimension (default: `hidden_size // num_attention_heads`)
- `mup_scale_qk_dot_by_d` -- muP scaling
- `scale_attn_weights` -- Whether to scale attention by 1/sqrt(d)
- `mup_attn_multiplier` -- muP attention multiplier
- `scale_attn_by_inverse_layer_idx` -- Scale attention by 1/(layer+1)
- `attn_dwconv` -- Use depthwise convolution
- `qkv_proj_bias` -- Bias in QKV projection (default: True)
- `out_proj_bias` -- Bias in output projection (default: True)
- `rotary_emb_fraction` -- Fraction of head dim for rotary embedding
- `rotary_emb_base` -- Base frequency for rotary embedding (default: 10000)
- `rotary_emb_scale_base` -- Base for rotary scale (for dynamic scaling)
- `rotary_emb_interleaved` -- Interleaved rotary pattern
- `use_alibi` -- Use ALiBi positional encoding
- `window_size` -- Sliding window attention size
- `use_flash_attn` -- Enable FlashAttention
- `fused_bias_fc` -- Use fused bias FC
- `n_head_kv` -- Number of KV heads for GQA/MQA

#### `create_mlp_cls`

```python
def create_mlp_cls(config, layer_idx=None, process_group=None, device=None, dtype=None)
```

Creates the MLP class based on config. Supports Mlp, GatedMlp, FusedMLP, ParallelFusedMLP, and FusedDenseSqreluDense.

**Config parameters read:**
- `mlp_fc1_bias`, `mlp_fc2_bias` -- Bias in MLP layers
- `fused_mlp` -- Use fused CUDA MLP
- `fused_dense_sqrelu_dense` -- Use Triton fused Dense-SqReLU-Dense
- `mlp_checkpoint_lvl` -- Gradient checkpointing level (0, 1, or 2)
- `mlp_multiple_of` -- Round MLP hidden dim to multiple of this

#### `create_block`

```python
def create_block(config, layer_idx=None, process_group=None, device=None, dtype=None)
```

Creates a transformer block (`Block` or `ParallelBlock`) with attention and MLP sub-blocks.

**Config parameters read:**
- `rms_norm` -- Use RMSNorm instead of LayerNorm
- `residual_in_fp32` -- Force residual in fp32
- `prenorm` -- Use pre-norm architecture (default: True)
- `parallel_block` -- Use parallel attention+MLP block (GPT-J style)
- `parallel_block_tied_norm` -- Share norm between attention and MLP
- `fused_dropout_add_ln` -- Fuse dropout + add + layer norm

#### `shard_state_dict_tp`

```python
def shard_state_dict_tp(state_dict, config, world_size, rank)
```

Shards a state dict for tensor parallelism. Splits weight tensors across ranks for:
- Word embeddings (first dimension)
- QKV projections (by head dimension)
- Output projections (last dimension)
- MLP fc1/fc2 weights
- LM head weights

**Parameters:**
- `state_dict` (dict): Full model state dict
- `config` (GPT2Config): Model configuration
- `world_size` (int): Number of tensor parallel ranks
- `rank` (int): Current rank

**Returns:** Modified state dict with tensors sharded for the current rank.

#### `combine_state_dicts_tp`

```python
def combine_state_dicts_tp(state_dicts: List[Dict[str, torch.Tensor]], config: GPT2Config)
```

Inverse of `shard_state_dict_tp`. Combines sharded state dicts from all ranks into a single full state dict.

### Weight Initialization

#### `_init_weights`

```python
def _init_weights(
    module, n_layer, initializer_range=0.02, mup_width_scale=1.0, rescale_prenorm_residual=True
)
```

Initializes weights following the GPT-2 scheme with optional muP (maximal update parameterization) scaling.

**Key behaviors:**
- Linear weights: Normal initialization with `std = initializer_range * sqrt(mup_width_scale)`
- Embeddings: Normal initialization with `std = initializer_range`
- Residual outputs (out_proj, fc2): Scaled by `1/sqrt(2 * n_layer)` to account for residual accumulation

### State Dict Remapping Functions

#### `remap_state_dict_hf_gpt2`

```python
def remap_state_dict_hf_gpt2(state_dict, config)
```

Converts HuggingFace GPT-2 state dict to FlashAttention format. Key transformations:
- `wpe.` -> `transformer.embeddings.position_embeddings.`
- `wte.weight` -> `transformer.embeddings.word_embeddings.weight` (with vocab padding)
- `h.{i}.ln_{1,2}.` -> `transformer.layers.{i}.norm{1,2}.`
- `h.{i}.mlp.c_fc.weight` -> Transposed to `transformer.layers.{i}.mlp.fc1.weight`
- `h.{i}.attn.c_attn.weight` -> Transposed and remapped to `transformer.layers.{i}.mixer.Wqkv.weight`

#### `remap_state_dict_megatron`

```python
def remap_state_dict_megatron(state_dict, config)
```

Converts Megatron-LM state dict to FlashAttention format. Handles the different Wqkv layout where Megatron stores as `(nheads, 3, headdim)` while FlashAttention uses `(3, nheads, headdim)`.

---

## BERT Model

**File:** `flash_attn/models/bert.py`

Optimized BERT implementation based on MLPerf 2.0/2.1 training implementations. Supports both pretraining (MLM + NSP) and downstream tasks.

### Classes

#### `BertPreTrainedModel`

Base class for BERT models.

```python
class BertPreTrainedModel(nn.Module):
    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs)
```

#### `BertModel`

Core BERT encoder model.

```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer=True)
```

**Methods:**

##### `forward`

```python
def forward(
    self,
    input_ids,
    position_ids=None,
    token_type_ids=None,
    attention_mask=None,
    masked_tokens_mask=None,
)
```

**Parameters:**
- `input_ids` (torch.Tensor): Token IDs, shape `(batch, seqlen)`
- `position_ids` (torch.Tensor, optional): Position IDs
- `token_type_ids` (torch.Tensor, optional): Segment IDs
- `attention_mask` (torch.Tensor, optional): Padding mask, shape `(batch, seqlen)`
- `masked_tokens_mask` (torch.Tensor, optional): Boolean mask for MLM tokens

**Returns:** `BaseModelOutputWithPoolingAndCrossAttentions` with:
- `last_hidden_state`: Sequence output
- `pooler_output`: CLS token pooled output

**Optimizations:**
- **Unpadded attention**: When `use_flash_attn=True` and padding mask is provided, input is unpadded before processing to avoid wasted computation on padding tokens
- **Subset attention**: For `last_layer_subset=True`, the last layer uses cross-attention where only masked tokens serve as queries against the full sequence, reducing computation by ~85%
- **Fused operations**: Layer norm + dropout + residual are fused when `fused_dropout_add_ln=True`

#### `BertForPreTraining`

BERT model for pretraining with MLM and NSP heads.

```python
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config: BertConfig)
```

**Config extensions:**
- `dense_seq_output` (bool): Only pass masked token hidden states to classifier heads
- `last_layer_subset` (bool): Only compute last layer for the subset of tokens needed for MLM loss and NSP prediction
- `use_xentropy` (bool): Use optimized cross-entropy loss

##### `forward`

```python
def forward(
    self,
    input_ids,
    position_ids=None,
    token_type_ids=None,
    attention_mask=None,
    labels=None,
    next_sentence_label=None,
)
```

**Returns:** `BertForPreTrainingOutput` with:
- `loss`: Total loss (MLM + NSP)
- `prediction_logits`: MLM prediction logits
- `seq_relationship_logits`: NSP prediction logits

#### `BertEncoder`

The BERT encoder stack.

```python
class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig)
```

##### `forward`

```python
def forward(self, hidden_states, key_padding_mask=None, subset_mask=None)
```

Handles two execution paths:
1. **Standard**: Process all tokens through all layers
2. **Unpadded**: Remove padding, process through layers, restore padding
3. **Subset**: Process all tokens through layers except the last, which uses cross-attention for only the needed subset

#### `BertPooler`

```python
class BertPooler(nn.Module):
    def forward(self, hidden_states, pool=True)
```

Extracts the CLS token representation and projects through a linear + tanh.

#### `BertLMPredictionHead`

```python
class BertLMPredictionHead(nn.Module):
    def forward(self, hidden_states)
```

Transforms hidden states through dense -> GELU -> LayerNorm -> linear (vocab projection).

### Configuration

```python
# BERT-Base example
from transformers import BertConfig

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu_new",
    # FlashAttention-specific
    use_flash_attn=True,
    fused_mlp=True,
    fused_bias_fc=True,
    fused_dropout_add_ln=True,
    dense_seq_output=True,
    last_layer_subset=True,
)
```

---

## LLaMA Model

**File:** `flash_attn/models/llama.py`

Support for Meta's LLaMA models (1, 2, 3, CodeLLaMA) with Grouped Query Attention (GQA), RMSNorm, SwiGLU activation, and rotary embeddings.

### State Dict Remapping

#### `remap_state_dict_meta_llama`

```python
def remap_state_dict_meta_llama(state_dict, config)
```

Converts Meta-format (original distributed checkpoint) state dict.

**Key transformations:**
- `tok_embeddings.` -> `transformer.embeddings.word_embeddings.`
- `output.weight` -> `lm_head.weight`
- `norm.` -> `transformer.ln_f.`
- `layers.{i}.attention_norm.` -> `transformer.layers.{i}.norm1.`
- `layers.{i}.ffn_norm.` -> `transformer.layers.{i}.norm2.`
- `feed_forward.w1.weight` + `feed_forward.w3.weight` -> `mlp.fc1.weight` (concatenated as [w3, w1] for SwiGLU)
- `feed_forward.w2.` -> `mlp.fc2.`
- `attention.wq/wk/wv.weight` -> `mixer.Wqkv.weight` (concatenated)
- `attention.wo.` -> `mixer.out_proj.`

#### `remap_state_dict_hf_llama`

```python
def remap_state_dict_hf_llama(state_dict, config)
```

Converts HuggingFace format LLaMA state dict. Also handles the interleaved rotary weight permutation that HF applies.

#### `inv_remap_state_dict_hf_llama`

```python
def inv_remap_state_dict_hf_llama(state_dict, config)
```

Inverse mapping: converts FlashAttention format back to HuggingFace LLaMA format.

### Configuration Conversion

#### `llama_config_to_gpt2_config`

```python
def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config
```

Converts HuggingFace `LlamaConfig` to extended `GPT2Config`.

**Key mappings:**
- `activation_function` = `"swiglu"` (hardcoded)
- `rms_norm` = `True`
- `rotary_emb_fraction` = `1.0` (full head dim)
- `rotary_emb_interleaved` = `True`
- `tie_word_embeddings` = `False`
- `qkv_proj_bias` = `False`
- `mlp_fc1_bias` = `False`
- `mlp_fc2_bias` = `False`
- `n_head_kv` = `llama_config.num_key_value_heads` (for GQA)

#### `config_from_meta_checkpoint`

```python
def config_from_meta_checkpoint(checkpoint_path, model_name) -> LlamaConfig
```

Loads LLaMA config from Meta's `params.json` format. Computes the MLP intermediate size following Meta's formula:

```python
intermediate_size = 4 * hidden_size
intermediate_size = int(2 * intermediate_size / 3)
if ffn_dim_multiplier is not None:
    intermediate_size = int(ffn_dim_multiplier * intermediate_size)
intermediate_size = multiple_of * ceil(intermediate_size / multiple_of)
```

### Usage Example

```python
from transformers import LlamaConfig
from flash_attn.models.llama import config_from_checkpoint, state_dicts_from_checkpoint, llama_config_to_gpt2_config
from flash_attn.models.gpt import GPTLMHeadModel

# From Meta checkpoint
llama_config = config_from_checkpoint("/path/to/llama", "7B", checkpoint_format="meta")
config = llama_config_to_gpt2_config(llama_config)
state_dicts = state_dicts_from_checkpoint("/path/to/llama", "7B")

# From HuggingFace checkpoint
from transformers import AutoConfig
llama_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config = llama_config_to_gpt2_config(llama_config)
model = GPTLMHeadModel.from_pretrained("meta-llama/Llama-2-7b-hf", config)
```

---

## GPT-NeoX Model

**File:** `flash_attn/models/gpt_neox.py`

Support for GPT-NeoX and Pythia models.

### State Dict Remapping

#### `remap_state_dict_hf_gpt_neox`

```python
def remap_state_dict_hf_gpt_neox(state_dict, config)
```

**Key transformations:**
- `gpt_neox.` -> `transformer.`
- `embed_in.` -> `transformer.embeddings.word_embeddings.`
- `layers.{i}.input_layernorm.` -> `transformer.layers.{i}.norm1.`
- `layers.{i}.post_attention_layernorm.` -> `transformer.layers.{i}.norm2.`
- `mlp.dense_h_to_4d.` -> `mlp.fc1.`
- `mlp.dense_4h_to_h.` -> `mlp.fc2.`
- `attention.query_key_value.weight` -> Rearranged from `(nheads, 3, headdim)` to `(3, nheads, headdim)`

### Configuration Conversion

#### `gpt_neox_config_to_gpt2_config`

```python
def gpt_neox_config_to_gpt2_config(gpt_neox_config: GPTNeoXConfig) -> GPT2Config
```

**Key mappings:**
- `prenorm` = `True`
- `parallel_block` = `gpt_neox_config.use_parallel_residual`
- `parallel_block_tied_norm` = `False`
- `rotary_emb_fraction` = `gpt_neox_config.rotary_pct`

---

## GPT-J Model

**File:** `flash_attn/models/gptj.py`

Support for GPT-J and GPT-JT models with parallel attention+MLP architecture.

### State Dict Remapping

#### `remap_state_dict_hf_gptj`

```python
def remap_state_dict_hf_gptj(state_dict, config)
```

**Key transformations:**
- Separate Q, K, V weights are concatenated into a single `Wqkv` weight
- `transformer.h.` -> `transformer.layers.`
- `attn.q_proj/k_proj/v_proj.weight` -> `mixer.Wqkv.weight`

### Configuration Conversion

#### `gptj_config_to_gpt2_config`

```python
def gptj_config_to_gpt2_config(gptj_config: GPTJConfig) -> GPT2Config
```

**Key mappings:**
- `parallel_block` = `True` (GPT-J uses parallel attention + MLP)
- `parallel_block_tied_norm` = `True`
- `rotary_emb_fraction` = `gptj_config.rotary_dim / headdim`
- `rotary_emb_interleaved` = `True`
- `qkv_proj_bias` = `False`
- `out_proj_bias` = `False`
- `lm_head_bias` = `True`

---

## OPT Model

**File:** `flash_attn/models/opt.py`

Support for Meta's OPT (Open Pre-trained Transformer) models, including the special OPT-350m variant.

### State Dict Remapping

#### `remap_state_dict_hf_opt`

```python
def remap_state_dict_hf_opt(state_dict, config)
```

**Key transformations:**
- `model.decoder.` -> `transformer.` (also handles `decoder.` for OPT-350m)
- `embed_tokens.` -> `transformer.embeddings.word_embeddings.`
- `embed_positions.weight` -> Position embeddings with first 2 indices removed (used for padding)
- Separate Q, K, V weights concatenated into `Wqkv`
- `project_in.` / `project_out.` -- Handled for OPT-350m's word_embed_proj_dim

### Configuration Conversion

#### `opt_config_to_gpt2_config`

```python
def opt_config_to_gpt2_config(opt_config: OPTConfig) -> GPT2Config
```

**Key mappings:**
- `prenorm` = `opt_config.do_layer_norm_before`
- `word_embed_proj_dim` -- For OPT-350m which has a smaller embedding dimension than hidden dimension

---

## Falcon Model

**File:** `flash_attn/models/falcon.py`

Support for TII Falcon models with multi-query attention (MQA).

### State Dict Remapping

#### `remap_state_dict_hf_falcon`

```python
def remap_state_dict_hf_falcon(state_dict, config)
```

**Key transformations:**
- Handles the unique Falcon Wqkv layout where Q and KV are grouped differently for MQA
- Input layout: `(group, ratio=n_head/n_head_kv + 2, headdim)` -> Output: `(3, nheads, headdim)`
- `self_attention.query_key_value.` -> `mixer.Wqkv.`

### Configuration Conversion

#### `falcon_config_to_gpt2_config`

```python
def falcon_config_to_gpt2_config(falcon_config: FalconConfig) -> GPT2Config
```

**Key mappings:**
- `parallel_block` = `falcon_config.parallel_attn`
- `n_head_kv` -- Inferred from `n_head_kv` or `multi_query` config field
- `parallel_block_tied_norm` -- Inferred from whether n_head_kv == 1 (MQA)
- `rotary_emb_fraction` = `1.0`
- `rotary_emb_interleaved` = `False`

---

## Vision Transformer

**File:** `flash_attn/models/vit.py`

Optimized Vision Transformer for image classification, based on timm's implementation.

### Classes

#### `VisionTransformer`

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        class_token=True,
        no_embed_class=False,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        use_flash_attn=False,
        fused_bias_fc=False,
        fused_mlp=False,
        fused_dropout_add_ln=False,
    )
```

**Parameters:**
- `img_size` (int): Input image size (default: 224)
- `patch_size` (int): Patch size for embedding (default: 16)
- `embed_dim` (int): Embedding dimension (default: 768)
- `depth` (int): Number of transformer blocks (default: 12)
- `num_heads` (int): Number of attention heads (default: 12)
- `mlp_ratio` (float): MLP hidden dim ratio (default: 4.0)
- `use_flash_attn` (bool): Enable FlashAttention

**Methods:**

##### `forward`

```python
def forward(self, x)
```

**Parameters:**
- `x` (torch.Tensor): Input images, shape `(batch, channels, height, width)`

**Returns:** Classification logits, shape `(batch, num_classes)`

##### `forward_features`

```python
def forward_features(self, x, all_tokens=True)
```

When `global_pool == "token"` and `all_tokens=False`, the last layer uses cross-attention where only the CLS token attends to the full sequence, reducing computation.

##### `load_state_dict`

```python
def load_state_dict(self, state_dict, strict=True)
```

Handles conversion from Conv2d patch embedding to Linear, and converts `attn.qkv` to `mixer.Wqkv`.

### Factory Functions

```python
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ViT-Base (ViT-B/16)"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return VisionTransformer(**model_kwargs)
```

---

## Baichuan Model

**File:** `flash_attn/models/baichuan.py`

Support for Baichuan models (7B and 13B). The 7B model uses rotary embeddings while the 13B model uses ALiBi.

### State Dict Remapping

#### `remap_state_dict_hf_baichuan`

```python
def remap_state_dict_hf_baichuan(state_dict, config)
```

Handles the Baichuan-specific `W_pack` attention format (combined QKV in a single weight matrix).

### Configuration Conversion

#### `baichuan_config_to_gpt2_config`

```python
def baichuan_config_to_gpt2_config(baichuan_config: PretrainedConfig) -> GPT2Config
```

**Key behavior:**
- Infers whether to use rotary or ALiBi from hidden_size (< 5000 = rotary, >= 5000 = ALiBi)
- Infers norm_head from vocab_size (> 70000 uses norm_head)
- Uses SwiGLU activation and RMSNorm

---

## BigCode Model

**File:** `flash_attn/models/bigcode.py`

Support for BigCode/StarCoder models with multi-query attention.

### State Dict Remapping

#### `remap_state_dict_hf_bigcode`

```python
def remap_state_dict_hf_bigcode(state_dict, config)
```

Handles MQA weight format where K and V have a single head, which are tiled to match Q's multiple heads.

#### `inv_remap_state_dict_hf_bigcode`

```python
def inv_remap_state_dict_hf_bigcode(state_dict, config)
```

Inverse mapping that reduces tiled K/V weights back to single-head format.

---

## BTLM Model

**File:** `flash_attn/models/btlm.py`

Support for Cerebras BTLM (Bilingual Transformer Language Model) with ALiBi and muP scaling.

### Configuration Conversion

#### `btlm_config_to_gpt2_config`

```python
def btlm_config_to_gpt2_config(btlm_config: PretrainedConfig) -> GPT2Config
```

**Key muP parameters:**
- `mup_width_scale` -- Width scaling for muP
- `mup_embeddings_multiplier` -- Embedding scale
- `mup_output_multiplier` -- Output scale
- `mup_scale_qk_dot_by_d` -- Scale QK dot product by dimension

**Position embedding:**
- When `position_embedding_type == "alibi"`: Uses ALiBi with `use_flash_attn=True`
- Otherwise: Uses learned absolute position embeddings

---

## Shared Design Patterns

### Block Architecture

All models use one of two block architectures:

#### Sequential Block (standard)

```
x -> LayerNorm -> Attention -> Dropout -> Add -> LayerNorm -> MLP -> Dropout -> Add
```

This is the standard pre-norm transformer block, used in GPT-2, LLaMA, BERT, etc.

#### Parallel Block (GPT-J/GPT-NeoX/Falcon style)

```
x -> LayerNorm -> Attention -> Dropout --\
                     Add <---------------/
x -> LayerNorm -> MLP     -> Dropout --/
```

Attention and MLP are computed in parallel from the same input, then summed.

### Optimized Operations

All models can leverage the following fused operations (when configured):

1. **FlashAttention**: `use_flash_attn=True` -- Memory-efficient attention
2. **Fused Dense**: `fused_bias_fc=True` -- Fused matmul + bias
3. **Fused MLP**: `fused_mlp=True` -- Fused two-layer MLP with activation
4. **Fused Dropout+Add+LN**: `fused_dropout_add_ln=True` -- Single kernel for dropout + residual + layer norm
5. **Fused Rotary**: Built into the MHA module when `rotary_emb_fraction > 0`

---

## Configuration Guide

### GPT2Config Extensions

The standard HuggingFace `GPT2Config` is extended with the following custom fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_flash_attn` | bool | False | Enable FlashAttention |
| `fused_bias_fc` | bool | False | Use fused bias FC kernel |
| `fused_mlp` | bool | False | Use fused MLP kernel |
| `fused_dense_sqrelu_dense` | bool | False | Use Triton fused Dense-SqReLU-Dense |
| `fused_dropout_add_ln` | bool | False | Fuse dropout + add + layernorm |
| `residual_in_fp32` | bool | False | Force residual in fp32 |
| `pad_vocab_size_multiple` | int | 1 | Pad vocab to multiple of this |
| `prenorm` | bool | True | Use pre-norm architecture |
| `rms_norm` | bool | False | Use RMSNorm instead of LayerNorm |
| `rotary_emb_fraction` | float | 0.0 | Fraction of head dim for rotary |
| `rotary_emb_base` | float | 10000.0 | Rotary base frequency |
| `rotary_emb_scale_base` | float | None | Base for rotary scale |
| `rotary_emb_interleaved` | bool | False | Interleaved rotary pattern |
| `use_alibi` | bool | False | Use ALiBi positional encoding |
| `window_size` | tuple | (-1, -1) | Sliding window size |
| `n_head_kv` | int | None | Number of KV heads (GQA) |
| `parallel_block` | bool | False | Parallel attention + MLP |
| `parallel_block_tied_norm` | bool | False | Share norm in parallel block |
| `qkv_proj_bias` | bool | True | Bias in QKV projection |
| `out_proj_bias` | bool | True | Bias in output projection |
| `mlp_fc1_bias` | bool | True | Bias in MLP fc1 |
| `mlp_fc2_bias` | bool | True | Bias in MLP fc2 |
| `lm_head_bias` | bool | False | Bias in LM head |
| `tie_word_embeddings` | bool | True | Tie input/output embeddings |
| `norm_head` | bool | False | Normalize LM head weights |
| `word_embed_proj_dim` | int | None | Separate embedding dimension (OPT-350m) |
| `mlp_checkpoint_lvl` | int/list | 0 | Gradient checkpoint level |
| `mlp_multiple_of` | int | 128 | Round MLP hidden dim to multiple |
| `mup_width_scale` | float | 1.0 | muP width scaling |
| `mup_embeddings_multiplier` | float | 1.0 | muP embedding multiplier |
| `mup_output_multiplier` | float | 1.0 | muP output multiplier |
| `mup_attn_multiplier` | float | 1.0 | muP attention multiplier |
| `mup_scale_qk_dot_by_d` | bool | False | muP QK scaling |

### Example Configurations

#### GPT-2 Small with FlashAttention

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    activation_function="gelu_new",
    # FlashAttention optimizations
    use_flash_attn=True,
    fused_mlp=True,
    fused_bias_fc=True,
    fused_dropout_add_ln=True,
    pad_vocab_size_multiple=8,
)
```

#### LLaMA-7B

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=32000,
    n_positions=0,
    n_embd=4096,
    n_layer=32,
    n_head=32,
    n_inner=11008,
    activation_function="swiglu",
    # LLaMA-specific
    rms_norm=True,
    rotary_emb_fraction=1.0,
    rotary_emb_interleaved=True,
    tie_word_embeddings=False,
    qkv_proj_bias=False,
    out_proj_bias=False,
    mlp_fc1_bias=False,
    mlp_fc2_bias=False,
    n_head_kv=32,
    use_flash_attn=True,
)
```

#### GPT-J 6B

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50400,
    n_positions=0,
    n_embd=4096,
    n_layer=28,
    n_head=16,
    n_inner=16384,
    activation_function="gelu_new",
    # GPT-J specific
    prenorm=True,
    parallel_block=True,
    parallel_block_tied_norm=True,
    rotary_emb_fraction=0.25,
    rotary_emb_interleaved=True,
    qkv_proj_bias=False,
    out_proj_bias=False,
    lm_head_bias=True,
    use_flash_attn=True,
)
```

---

## Tensor Parallel Utilities

### `shard_state_dict_tp`

Splits a model's state dict for tensor parallel training. Each rank receives a shard of the relevant weight tensors.

**Sharding strategy by tensor type:**
- **Word embeddings**: Split along vocabulary dimension (dim 0)
- **QKV weights**: Split by number of heads, handling GQA correctly
- **Output projection**: Split along hidden dimension (dim -1)
- **MLP fc1**: Split along output dimension, with special handling for gated MLPs (SwiGLU)
- **MLP fc2**: Split along input dimension (dim -1)
- **LM head**: Split along vocabulary dimension
- **Biases**: Only rank 0 retains output projection biases

### `combine_state_dicts_tp`

Reconstructs the full state dict from sharded state dicts. The inverse operation of `shard_state_dict_tp`.

---

## State Dict Remapping

### Common Pattern

All remapping functions follow the same pattern:

1. **Rename keys** using regex substitutions to match FlashAttention's naming convention
2. **Combine separate weights** (e.g., separate Q, K, V -> combined Wqkv)
3. **Transpose weights** (some frameworks use row-major vs column-major)
4. **Pad vocabulary** to the configured multiple
5. **Handle bias tying** (e.g., tied word embeddings = LM head weights)

### Inverse Remapping

Several models provide `inv_remap_state_dict_*` functions that convert back to the original format:

- `inv_remap_state_dict` (BERT)
- `inv_remap_state_dict_hf_llama` (LLaMA)
- `inv_remap_state_dict_hf_bigcode` (BigCode)

These are useful for converting fine-tuned models back to HuggingFace format for inference with standard libraries.
