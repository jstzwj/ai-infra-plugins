# 17 - Multimodal Models Reference

This document provides a comprehensive reference for Megatron-LM's multimodal model
architecture, focusing on the Vision-Language Model (VLM) implementation, the LLaVA-style
model architecture, vision encoders, projection layers, and data loading for multimodal
training.

## Architecture Overview

Megatron-LM implements multimodal models primarily through the `LLaVAModel` class located
at `megatron/core/models/multimodal/llava_model.py`. The architecture follows a three-
component design:

1. **Vision Encoder** -- Extracts visual features from images or video frames.
2. **Vision Projection** -- Maps vision encoder outputs to the language model hidden dimension.
3. **Language Model** -- Processes combined text and visual embeddings.

The data flow is:

```
Images -> Vision Encoder -> [Drop Class Token] -> [Pixel Shuffle] -> Vision Projection
                                                                             |
Text IDs -> Text Embedding -----> Preprocess Data -> Combine Embeddings -> Language Model -> Output
```

## LLaVAModel Class

**File**: `megatron/core/models/multimodal/llava_model.py`

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `language_transformer_config` | `TransformerConfig` | required | Config for the language model |
| `language_transformer_layer_spec` | `ModuleSpec` | required | Language model layer spec |
| `language_vocab_size` | `int` | required | Language model vocabulary size |
| `language_max_sequence_length` | `int` | required | Maximum sequence length |
| `vision_transformer_config` | `TransformerConfig` | required | Config for the vision model |
| `vision_transformer_layer_spec` | `ModuleSpec` | required | Vision model layer spec |
| `drop_vision_class_token` | `bool` | required | Drop vision class token before LM |
| `vision_projection_config` | `TransformerConfig` | required | Vision projection config |
| `vision_projection_layer_spec` | `ModuleSpec` | required | Vision projection layer spec |
| `vision_projection_type` | `str` | `"mlp"` | Projection type: `"mlp"` or `"affine"` |
| `allow_missing_vision_projection_checkpoint` | `bool` | `False` | Allow missing projection weights |
| `parallel_output` | `bool` | `True` | Keep outputs split across TP ranks |
| `share_embeddings_and_output_weights` | `bool` | `False` | Share embedding/output weights |
| `language_position_embedding_type` | `str` | `'learned_absolute'` | Position embedding type |
| `language_rotary_percent` | `float` | `1.0` | RoPE rotary percent |
| `pre_process` | `bool` | `True` | Include embedding layer |
| `post_process` | `bool` | `True` | Include output layer |
| `add_encoder` | `bool` | `True` | Construct the encoder |
| `add_decoder` | `bool` | `True` | Construct the decoder |
| `img_h` | `int` | `336` | Input image height |
| `img_w` | `int` | `336` | Input image width |
| `patch_dim` | `int` | `14` | Patch size for image tokenization |
| `image_token_index` | `int` | `-200` | Token ID for `<image>` placeholder |
| `pixel_shuffle` | `bool` | `False` | Enable pixel shuffle (InternVL-style) |
| `tile_tags` | `list` | `None` | Optional tile tags (NVLM-style) |
| `max_num_tiles` | `int` | `0` | Maximum number of tiles per image |
| `use_vision_backbone_fp8_arch` | `bool` | `False` | Use FP8 vision backbone architecture |

### Supported Language Model Types

The `LLaVAModel` supports three language model backends, selected via
`language_transformer_config.language_model_type`:

- **GPTModel** (default): Standard GPT decoder-only model
- **HybridModel** (prefix `nemotron5-hybrid`): Hybrid SSM-Transformer architecture
- **HuggingFace model** (prefix `hf://`): External HF model integration

### Forward Method

```python
def forward(
    self,
    images: torch.Tensor,           # [num_tiles, img_h, img_w]
    input_ids: torch.Tensor,         # [batch, text_seq_len]
    position_ids: torch.Tensor,      # [batch, text_seq_len]
    attention_mask: torch.Tensor,    # [batch, 1, 1, combined_seq_len]
    labels: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    num_image_tiles: Optional[List[int]] = None,
    image_token_index: Optional[int] = None,
    runtime_gather_output: Optional[bool] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Returns**: `(output, loss_mask)` where output is loss `[b, s]` if labels are provided,
otherwise logits `[b, s, vocab_size]`.

## Vision Encoder Backends

The vision encoder is selected via `vision_transformer_config.vision_model_type`:

### CLIP ViT (`"clip"`, `"siglip"`, `"internvit"`, `"internvit300M"`)

Uses `CLIPViTModel` from `megatron/core/models/vision/clip_vit_model.py`.

**Model subtype details**:

| Subtype | Class Token | `ln_pre` | `ln_post` | Conv Bias | Padding |
|---|---|---|---|---|---|
| `clip` | Yes (len=1) | Yes | No | False | 0 |
| `siglip` | No | No | Yes | True | `"valid"` |
| `internvit` | Yes (len=1) | No | No | True | 0 |
| `internvit300M` | Yes (len=1) | No | No | True | 0 |

**Processing pipeline for CLIP ViT**:
1. `conv1`: 2D convolution with kernel_size=stride=patch_dim, maps 3 channels to hidden_size
2. Reshape to `[batch, grid^2, hidden_size]`
3. Prepend class token(s) if enabled
4. Add learned absolute position embeddings
5. Apply `ln_pre` (CLIP subtype only)
6. Pass through `TransformerBlock` decoder
7. Apply `ln_post` (SigLIP subtype only)

### RADIO ViT (`"radio"`, `"radio-g"`, `"cradio-g"`)

Uses `RADIOViTModel` from `megatron/core/models/vision/radio.py`.

**Model subtype details**:

| Subtype | Class Token Len | Max Image Size | Embedder Bias | Mask Token |
|---|---|---|---|---|
| `radio` | 8 | 2048x2048 | False | False |
| `radio-g` | 5 | 1792x1792 | True | True |
| `cradio-g` | 8 | 2048x2048 | False | False |

Key features of RADIO:
- Uses `ColumnParallelLinear` as embedder (not Conv2d), requiring `einops` package
- Implements Conditional Positional Encoding (CPE) with dynamic image resolution support
- CPE training applies random scaling and aspect ratio jittering to position embeddings
- CPE inference uses bilinear interpolation for resolution adaptation
- FP8 support requires class_token_len to be padded to multiple of 16 (or 32 for MXFP8)

### HuggingFace Models (prefix `"hf://"`)

Loads any HuggingFace vision model via `build_hf_model()`.

## Vision Projection Layer

**File**: `megatron/core/models/vision/multimodal_projector.py`

The `MultimodalProjector` maps vision encoder output dimension to the language model
hidden dimension. Two projection types are available:

### MLP Projection (`vision_projection_type="mlp"`)

Uses a full `MLP` module (two linear layers with activation). The input size equals the
vision encoder's hidden_size (multiplied by 4 if pixel_shuffle is enabled).

### Affine Projection (`vision_projection_type="affine"`)

Uses a single linear layer (`linear_fc1` from submodules) for direct dimension mapping.
This is a simpler, lower-parameter alternative to MLP projection.

### FP8 Support

The projector uses FP8 context managers for forward pass when configured:
```python
fp8_context = get_fp8_context(self.config)
with fp8_context:
    encoder_output, encoder_output_bias = apply_module(self.encoder)(hidden_states)
```

## Image Token Handling

### Token Index and Placeholders

- `DEFAULT_IMAGE_TOKEN_INDEX = -200`: Placeholder index in `input_ids` for image positions
- `IMAGE_TOKEN = "<image>"`: String representation
- `VIDEO_TOKEN = "<video>"`: String representation for video inputs

### Image Sequence Length Calculation

`get_num_image_embeddings()` calculates the number of image embedding tokens per tile:

```python
num_patches = (img_h // patch_dim) * (img_w // patch_dim)
num_embeddings = num_patches + class_token_len  # if class token kept

# With pixel shuffle (InternVL-style):
num_embeddings = int(num_embeddings * 0.25)  # 0.5^2

# With tile tags:
num_embeddings += 5  # or 6 for nemotron5 tokenizer
```

### Embedding Preprocessing (`_preprocess_data`)

The preprocessing function merges image and text embeddings:

1. Identify image token positions in `input_ids` via `image_token_index`
2. Calculate new sequence length: `text_seq_len + num_tiles * img_seq_len - num_images`
3. Place text embeddings at shifted text positions
4. Place image embeddings at image mask positions
5. Create corresponding labels and loss masks
6. Truncate to `language_max_sequence_length` if needed

The loss mask zeros out:
- Image token positions (no prediction loss on image embeddings)
- The text position immediately before an image token

## Pixel Shuffle

Implements InternVL-style pixel shuffle for reducing spatial tokens:

```python
def pixel_shuffle(x, scale_factor=0.5, version=2):
    # Reduces spatial dimension by scale_factor^2
    # Increases channel dimension by 1/scale_factor^2
    # Input: [num_tiles, seq_len, h_vision]
    # Output: [num_tiles, seq_len * 0.25, h_vision * 4]
```

## Tile Tagging (NVLM-style)

When `tile_tags` is configured, the model prepends learned tile tags to image embeddings:

```python
# tile_tag_input_ids shape: [num_tiles, tile_seq_len]
# Tags like <tile_1>, <tile_2>, etc.
tile_tag_embeds = self.language_model.embedding(tile_tag_input_ids, position_ids=None)
image_embeddings = torch.cat([tile_tag_embeds, image_embeddings])
```

## Multimodal Sequence and Context Parallelism

**File**: `megatron/core/models/multimodal/context_parallel.py`

### Padding Calculation

`get_padding()` computes the padding needed for combined text+image sequences:

| Condition | Padding Factor |
|---|---|
| SP + CP | `tp_size * cp_size * 2` |
| CP only | `cp_size * 2` |
| SP only | `tp_size` |
| FP8 (MXFP8) | 32 |
| FP8 (other) | 16 |

### Packed Sequence Parameters

`get_packed_seq_params()` constructs `PackedSeqParams` for context parallelism:

- Calculates combined valid sequence length: `text_seq_len + img_seq_len - padding`
- Computes `cu_seqlens` for each batch sample
- When CP > 1 with padding, switches to THD format

### Token Parallel Processing

`_process_embedding_token_parallel()` handles SP/CP distribution:

1. **CP sharding**: Distributes combined sequence across CP ranks using
   `get_batch_on_this_cp_rank()` or THD partitioned indices
2. **SP scattering**: Scatters combined embeddings across TP ranks via
   `scatter_to_sequence_parallel_region()`

## Model Freezing

The `freeze()` method allows selective module freezing:

```python
model.freeze(
    freeze_language_model=True,
    freeze_vision_model=True,
    freeze_vision_projection=False,
)
```

This sets `requires_grad = False` for parameters in selected modules. Useful for:
- Stage 1 pretraining: Freeze LM and vision encoder, train projection only
- Stage 2 fine-tuning: Unfreeze all or specific components

## Decoder Model Layer Specs

**File**: `megatron/core/models/multimodal/llava_spec.py`

Two decoder specifications are provided:

### Transformer Engine Spec (`decoder_model_with_transformer_engine_default_spec`)

Uses TE components for optimal performance:
- `TELayerNormColumnParallelLinear` for QKV projection
- `TEDotProductAttention` for core attention
- `TERowParallelLinear` for output projection
- Supports MoE via `num_experts` and `moe_grouped_gemm` parameters
- Optional QK LayerNorm via `qk_layernorm` parameter

### Local Spec (`decoder_model_with_local_default_spec`)

Uses Megatron-Core native components:
- `ColumnParallelLinear` / `RowParallelLinear`
- `DotProductAttention`
- `FusedLayerNorm` (with Apex) or `WrappedTorchNorm` (fallback)
- Explicit `input_layernorm` and `pre_mlp_layernorm`

## Pipeline Parallelism with VLM

`LLaVAModel` supports pipeline parallelism through `pre_process`, `post_process`,
`add_encoder`, and `add_decoder` flags:

| Stage | `add_encoder` | `add_decoder` | `pre_process` | `post_process` |
|---|---|---|---|---|
| First PP stage | True | True | True | False |
| Middle PP stage | False | True | False | False |
| Last PP stage | False | True | False | True |

`set_input_tensor()` handles tensor passing between pipeline stages:
- Encoder+Decoder: passes to vision model
- Encoder only: passes to vision model
- Pre-process only: stores as `encoder_hidden_state`
- Otherwise: passes to language model

## Inference Support

### KV Cache

During inference with KV cache, image tokens are computed once and cached:

```python
use_inference_kv_cache = (
    inference_context is not None
    and "image_tokens_count" in inference_context.key_value_memory_dict
)
```

When cached, image embedding computation is skipped and the token count is stored
as an offset for subsequent KV cache operations.

### Checkpoint Loading Hooks

Two state dict hooks handle checkpoint compatibility:

- `_load_state_dict_hook_ignore_param_names`: Allows missing projection weights when
  loading from separate vision/language checkpoints
- `_load_state_dict_hook_ignore_extra_state`: Ignores Transformer Engine `_extra_state`
  keys for FP8 backward compatibility

## Key Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `--vision-model-type` | Vision encoder backend | `"clip"` |
| `--disable-vision-class-token` | Drop class token | `False` |
| `--img-h` / `--img-w` | Image dimensions | `336` |
| `--patch-dim` | Patch size | `14` |
| `--pixel-shuffle` | Enable pixel shuffle | `False` |
| `--image-token-index` | Image token ID | `-200` |
| `--allow-missing-vision-projection-checkpoint` | Allow missing projection | `False` |
| `--vision-projection-type` | `"mlp"` or `"affine"` | `"mlp"` |
| `--freeze-language-model` | Freeze LM during training | `False` |
| `--freeze-vision-model` | Freeze vision encoder | `False` |
| `--freeze-vision-projection` | Freeze projection layer | `False` |
