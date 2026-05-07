# 18 - Vision Models Reference

This document provides a comprehensive reference for Megatron-LM's vision model
implementations, including the ViT (Vision Transformer) architecture, the CLIP ViT
model, the RADIO model, image classification backbones, patch embedding, and vision-
specific configurations.

## Overview

Megatron-LM's vision models are located under `megatron/core/models/vision/` and serve as
encoder backbones for multimodal models. The two primary implementations are:

1. **CLIPViTModel** -- A standard ViT implementation supporting CLIP, SigLIP, and InternViT
   variants.
2. **RADIOViTModel** -- An advanced ViT implementation supporting NVIDIA RADIO models with
   dynamic resolution and conditional positional encoding.

Both inherit from `VisionModule` (at `megatron/core/models/common/vision_module/vision_module.py`)
which itself extends `MegatronModule`.

## CLIPViTModel

**File**: `megatron/core/models/vision/clip_vit_model.py`

### Architecture

CLIPViTModel implements the Vision Transformer architecture from "An Image is Worth 16x16
Words" (Dosovitskiy et al., 2020), adapted for CLIP-style pretraining. The model processes
images through the following pipeline:

```
Input Image [B, 3, H, W]
    |
    v
Conv2d (3 -> hidden_size, kernel=patch_dim, stride=patch_dim)
    |
    v
Reshape [B, hidden_size, grid_h, grid_w] -> [B, num_patches, hidden_size]
    |
    v
[Optional] Prepend Class Token(s)
    |
    v
Add Learned Absolute Position Embeddings
    |
    v
[Optional] LayerNorm Pre (CLIP only)
    |
    v
Permute [B, S, H] -> [S, B, H]
    |
    v
TransformerBlock (N layers)
    |
    v
Permute [S, B, H] -> [B, S, H]
    |
    v
[Optional] LayerNorm Post (SigLIP only)
    |
    v
Output [B, seq_length, hidden_size]
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transformer_config` | `TransformerConfig` | required | Model configuration |
| `transformer_layer_spec` | `ModuleSpec` | required | Layer spec for transformer blocks |
| `ln_pre_impl` | `ModuleSpec` or type | `TENorm` or `LayerNorm` | Pre-norm implementation |
| `ln_post_impl` | `ModuleSpec` or type | `TENorm` or `LayerNorm` | Post-norm implementation |
| `add_class_token` | `bool` | `True` | Include class token(s) |
| `class_token_len` | `int` | `1` | Number of class tokens |
| `patch_dim` | `int` | `14` | Patch size |
| `img_h` | `int` | `336` | Image height |
| `img_w` | `int` | `336` | Image width |
| `model_subtype` | `str` | `"clip"` | Subtype: `"clip"`, `"siglip"`, `"internvit"`, `"internvit300M"` |
| `pg_collection` | `ProcessGroupCollection` | `None` | Process groups |
| `vp_stage` | `int` | `None` | Virtual pipeline stage |

### Model Subtypes

#### CLIP (`model_subtype="clip"`)

The original CLIP ViT variant:
- Applies `ln_pre` (LayerNorm) before the transformer
- No `ln_post` after the transformer
- Conv2d with no bias and zero padding
- Class token length of 1

#### SigLIP (`model_subtype="siglip"`)

The SigLIP (Sigmoid Loss for Language-Image Pre-Training) variant:
- No `ln_pre` before the transformer
- Applies `ln_post` (LayerNorm) after the transformer
- Conv2d with bias and `"valid"` padding
- No class tokens (class_token_len=0, add_class_token=False)
- Asserts that `drop_vision_class_token` is not True

#### InternViT (`model_subtype="internvit"` or `"internvit300M"`)

The InternViT variant from InternVL:
- No `ln_pre` and no `ln_post`
- Conv2d with bias and zero padding
- Class token length of 1

### Patch Embedding

The patch embedding uses a 2D convolutional layer:

```python
self.conv1 = torch.nn.Conv2d(
    in_channels=3,
    out_channels=visual_hidden_size,   # from transformer_config.hidden_size
    kernel_size=patch_dim,
    stride=patch_dim,
    bias=conv_bias,                    # subtype-dependent
    padding=padding,                   # subtype-dependent
)
```

The number of patches is computed as:
```python
num_patches_per_dim_h = img_h // patch_dim
num_patches_per_dim_w = img_w // patch_dim
num_patches = num_patches_per_dim_h * num_patches_per_dim_w
```

For a 336x336 image with patch_dim=14: `24 * 24 = 576` patches.

### Position Embeddings

Learned absolute position embeddings:
```python
self.position_ids = torch.arange(seq_length).expand(1, -1).cuda()
self.position_embeddings = torch.nn.Embedding(
    seq_length, visual_hidden_size, dtype=params_dtype
)
```

Where `seq_length = num_patches + class_token_len` (if class tokens are used).

### Class Token

When enabled, a learnable class token is prepended to the patch embeddings:

```python
self.class_token = torch.nn.Parameter(
    torch.randn(1, class_token_len, visual_hidden_size, dtype=params_dtype)
)
```

The class token is expanded to batch size and concatenated:
```python
class_token = self.class_token.expand(batch_size, -1, -1)
x = torch.cat([class_token, x], dim=1)
```

### Forward Pass

```python
def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
    # x: [batch, 3, img_h, img_w] or [batch, img_h, img_w]
    x = self.conv1(x)                    # [B, hidden_size, grid_h, grid_w]
    x = x.reshape(B, hidden_size, -1)    # [B, hidden_size, num_patches]
    x = x.permute(0, 2, 1)              # [B, num_patches, hidden_size]
    if add_class_token:
        x = cat([class_token, x], dim=1) # [B, num_patches+cls, hidden_size]
    x = x + position_embeddings(position_ids)
    if ln_pre: x = ln_pre(x)
    x = x.permute(1, 0, 2)              # [S, B, H]
    x = self.decoder(x, attention_mask)  # TransformerBlock
    x = x.permute(1, 0, 2)              # [B, S, H]
    if ln_post: x = ln_post(x)
    return x
```

### get_num_image_embeddings()

Utility function that calculates the number of image embedding tokens per tile:

```python
def get_num_image_embeddings(
    img_h, img_w, patch_dim, vision_model_type,
    disable_vision_class_token, class_token_len,
    pixel_shuffle, use_tile_tags=False,
    max_num_tiles=0, tokenizer_type=None,
) -> int:
```

Considers:
- Number of patches from image dimensions and patch size
- Whether class tokens are kept (model-type dependent)
- Pixel shuffle reduction (factor of 0.25)
- Tile tag overhead (5 or 6 tokens depending on tokenizer)

## RADIOViTModel

**File**: `megatron/core/models/vision/radio.py`

### Overview

RADIO (Reference Algorithm for Distribution-based Image Output) is an advanced ViT
implementation that supports dynamic image resolution through Conditional Positional
Encoding (CPE). It is designed for flexible resolution handling and can process images
of varying sizes up to configurable maximums.

### Architecture

```
Input Image [B, 3, H, W]
    |
    v
Rearrange (einops) -> [B, num_patches, 3 * patch_dim^2]
    |
    v
ColumnParallelLinear (embedder) -> [B, num_patches, hidden_size]
    |
    v
Apply CPE Position Encoding
    |
    v
[Optional] Prepend Class Token(s)
    |
    v
[Optional] ln_pre
    |
    v
Permute [B, S, H] -> [S, B, H]
    |
    v
TransformerBlock
    |
    v
Permute [S, B, H] -> [B, S, H]
    |
    v
[Optional] ln_post
    |
    v
Output [B, seq_length, hidden_size]
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transformer_config` | `TransformerConfig` | required | Model configuration |
| `transformer_layer_spec` | `ModuleSpec` | required | Layer spec for transformer blocks |
| `ln_pre_impl` | `ModuleSpec` or type | `None` | Pre-norm implementation |
| `ln_post_impl` | `ModuleSpec` or type | `None` | Post-norm implementation |
| `use_mask_token` | `bool` | `False` | Use RADIO mask token |
| `add_class_token` | `bool` | `True` | Include class token(s) |
| `class_token_len` | `int` | `8` | Number of class tokens |
| `patch_dim` | `int` | `16` | Patch size |
| `img_h` | `int` | `224` | Image height |
| `img_w` | `int` | `224` | Image width |
| `max_img_h` | `int` | `2048` | Maximum image height |
| `max_img_w` | `int` | `2048` | Maximum image width |
| `pos_dropout` | `int` | `0` | Positional encoding dropout |
| `has_cpe` | `bool` | `True` | Use conditional positional encoding |
| `embedder_bias` | `bool` | `False` | Bias in embedder linear layer |
| `pg_collection` | `ProcessGroupCollection` | `None` | Process groups |
| `vp_stage` | `int` | `None` | Virtual pipeline stage |

### RADIO Variants

| Variant | class_token_len | Max Resolution | Embedder Bias | Mask Token | ln_post |
|---|---|---|---|---|---|
| `radio` | 8 | 2048 | No | No | None |
| `radio-g` | 5 | 1792 | Yes | Yes | TENorm |
| `cradio-g` | 8 | 2048 | No | No | None |

### Patch Embedding (Linear Embedder)

Unlike CLIPViT which uses Conv2d, RADIO uses a `ColumnParallelLinear` as the patch
embedder, enabling tensor parallelism on the embedding dimension:

```python
self.embedder = ColumnParallelLinear(
    input_size=3 * patch_dim * patch_dim,
    output_size=visual_hidden_size,
    bias=embedder_bias,
    config=transformer_config,
    gather_output=True,
    init_method=lambda t: torch.nn.init.normal_(t, mean=0.0, std=1.0),
)
```

Input patches are rearranged using `einops`:
```python
x = rearrange(x, "b c (py yy) (px xx) -> b (py px) (c yy xx)",
              py=py, yy=patch_dim, px=px, xx=patch_dim)
```

### Conditional Positional Encoding (CPE)

RADIO implements dynamic resolution handling through CPE. The position embeddings are
stored as a large tensor covering the maximum resolution:

```python
self.position_embeddings = nn.Parameter(
    torch.randn(1, max_num_patches, hidden_size) * pos_scale
)
```

Where `max_num_patches = (max_img_h // patch_dim) * (max_img_w // patch_dim)`.

#### CPE Training Behavior

During training, CPE applies random augmentation to position embeddings:

```python
# Random scale (0.316 to 1.0)
scale = torch.rand(batch, 1, 1) * (1 - sqrt(0.1)) + sqrt(0.1)
# Random aspect ratio (0.75 to 1.33)
aspect = exp(rand * (log(3/4) - log(4/3)) + log(4/3))
# Random position offset
pos_xy = rand(batch, 1, 1, 2) * (1 - scale_xy)
# Bilinear grid sample
pos_embed = F.grid_sample(pos_embed, grid_xy, mode="bilinear")
```

#### CPE Inference Behavior

During inference, position embeddings are interpolated to match the input resolution:

```python
max_dim = max(input_dims)
pos_embed = F.interpolate(pos_embed, size=(max_dim, max_dim),
                          align_corners=True, mode="bilinear")
pos_embed = window_select(pos_embed)  # Crop to actual dimensions
```

### FP8 Support

RADIO supports FP8 training with special padding for class tokens:

```python
def fp8_pad_hook(module, state_dict, prefix, ...):
    # FP8 requires class_token_len to be a multiple of 16 (or 32 for MXFP8)
    pad = 32 if module.config.fp8_recipe == "mxfp8" else 16
    if class_token.shape[0] % pad != 0:
        pad_len = pad - (class_token.shape[0] % pad)
        class_token = torch.cat([pad_tensor, class_token], dim=0)
```

This hook is registered as a `load_state_dict_pre_hook` when FP8 is enabled.

### Position Dropout

When `pos_dropout > 0`, position embeddings are randomly dropped during training:

```python
if self.training and self.pos_dropout > 0:
    keeps = torch.rand(batch, 1, 1) > self.pos_dropout
    pos_enc_drop = torch.where(keeps, pos_enc, 0)
```

## MultimodalProjector

**File**: `megatron/core/models/vision/multimodal_projector.py`

The projector bridges the gap between vision encoder output dimension and language model
input dimension.

### Constructor

```python
class MultimodalProjector(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        projector_type: str,         # "mlp" or "affine"
        input_size: int,             # Vision encoder hidden_size
        tp_group=None,
    )
```

### Projection Types

#### MLP (`projector_type="mlp"`)

Full two-layer MLP with activation:
```python
self.encoder = MLP(config=config, submodules=submodules, input_size=input_size)
```

#### Affine (`projector_type="affine"`)

Single linear projection:
```python
self.encoder = submodules.linear_fc1(
    input_size, config.hidden_size,
    config=config, gather_output=True,
    bias=config.add_bias_linear, skip_bias_add=True,
)
```

### Forward Pass

```python
def forward(self, hidden_states):
    fp8_context = get_fp8_context(self.config)
    with fp8_context:
        encoder_output, encoder_output_bias = self.encoder(hidden_states)
        if encoder_output_bias is not None:
            encoder_output = encoder_output + encoder_output_bias
        encoder_output = make_viewless_tensor(encoder_output, requires_grad=True)
    return encoder_output
```

## Vision Transformer Layer Specs

**File**: `megatron/core/models/vision/vit_layer_specs.py`

### TE Vision Spec

`get_vit_layer_with_transformer_engine_spec()` returns a spec optimized for Transformer
Engine:

- `TELayerNormColumnParallelLinear` for QKV projection (fused LayerNorm + Linear)
- `TEDotProductAttention` for core attention
- `TERowParallelLinear` for output projection
- `MLP` with `TELayerNormColumnParallelLinear` / `TERowParallelLinear`
- `IdentityOp` for `pre_mlp_layernorm` (fused into TE linear_fc1)
- `AttnMaskType.no_mask` (bidirectional attention for ViT)

### Local Vision Spec

`get_vit_layer_with_local_spec()` returns a spec using Megatron-Core native layers:

- `ColumnParallelLinear` for QKV
- `DotProductAttention` for core attention
- `RowParallelLinear` for output projection
- `FusedLayerNorm` (Apex) or `WrappedTorchNorm` for normalization
- `AttnMaskType.causal` for attention masking
- Explicit `input_layernorm` and `pre_mlp_layernorm`

### Key Differences from GPT Specs

Vision layer specs differ from GPT decoder specs in several ways:

1. **Attention Mask**: ViT uses `AttnMaskType.no_mask` (bidirectional), while GPT uses
   `AttnMaskType.causal`
2. **No Cross-Attention**: Vision models use only self-attention
3. **Position Embedding**: ViT uses learned absolute embeddings (or CPE for RADIO),
   not RoPE

## Integration with Multimodal Models

Vision models are integrated into the LLaVA multimodal framework through the
`LLaVAModel` class:

```python
# In LLaVAModel.__init__():
if vision_model_type.startswith(("clip", "siglip", "internvit")):
    self.vision_model = CLIPViTModel(
        vision_transformer_config, vision_transformer_layer_spec,
        img_h=img_h, img_w=img_w,
        class_token_len=class_token_len, patch_dim=patch_dim,
        model_subtype=vision_model_type,
        add_class_token=add_class_token,
    )
elif vision_model_type in ("radio", "radio-g", "cradio-g"):
    self.vision_model = RADIOViTModel(
        vision_transformer_config, vision_transformer_layer_spec,
        ln_post_impl=ln_post_impl,
        img_h=img_h, img_w=img_w,
        max_img_h=max_img_h, max_img_w=max_img_w,
        class_token_len=class_token_len, patch_dim=patch_dim,
        embedder_bias=embedder_bias,
        use_mask_token=use_mask_token,
    )
```

After vision encoding, the output goes through optional post-processing:

1. **Drop class token** (if `drop_vision_class_token=True`)
2. **Pixel shuffle** (if `pixel_shuffle=True`)
3. **Permute** to `[img_seq_len, num_tiles, h_vision]`
4. **Vision projection** to map to language model dimension
5. **Tile tagging** (if tile tags are configured)

## Key Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `--vision-model-type` | Vision encoder: clip, siglip, internvit, radio, radio-g, cradio-g | `"clip"` |
| `--img-h` | Image height | `336` |
| `--img-w` | Image width | `336` |
| `--patch-dim` | Patch size | `14` |
| `--disable-vision-class-token` | Drop vision class tokens | `False` |
| `--vision-projection-type` | Projection: mlp or affine | `"mlp"` |
| `--use-vision-backbone-fp8-arch` | FP8 vision backbone | `False` |

## Dependencies

- **einops**: Required for RADIO model (`pip install einops`)
- **Transformer Engine**: Required for TE vision layer specs and FP8 training
- **Apex**: Optional, used for FusedLayerNorm in local specs
