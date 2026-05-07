# Chapter 06: Attention Mechanisms Reference

## Source Files
- `sources/Megatron-LM/megatron/core/transformer/attention.py`
- `sources/Megatron-LM/megatron/core/transformer/dot_product_attention.py`

## Overview

Megatron-Core provides a flexible attention system supporting multiple backends, Grouped Query Attention (GQA), Multi-Latent Attention (MLA), sliding window attention, and various inference optimizations. The attention architecture follows a three-level hierarchy:

```
Attention (abstract base)
  |-- SelfAttention  (self-attention for decoder)
  |-- CrossAttention (cross-attention for encoder-decoder)

CoreAttention (protocol)
  |-- DotProductAttention  (local PyTorch implementation)
  |-- TEDotProductAttention (TransformerEngine implementation)
```

---

## SelfAttentionSubmodules

```python
@dataclass
class SelfAttentionSubmodules:
    linear_qkv: LinearQkvBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: LayerNormBuilder | None = None
    k_layernorm: LayerNormBuilder | None = None
```

| Field | Type | Default | Description |
|---|---|---|---|
| `linear_qkv` | `LinearQkvBuilder` | required | QKV projection linear layer builder |
| `core_attention` | `CoreAttentionBuilder` | required | Core attention mechanism builder |
| `linear_proj` | `Union[ModuleSpec, type]` | `None` | Output projection linear layer |
| `q_layernorm` | `LayerNormBuilder \| None` | `None` | Optional Q normalization layer |
| `k_layernorm` | `LayerNormBuilder \| None` | `None` | Optional K normalization layer |

---

## CrossAttentionSubmodules

```python
@dataclass
class CrossAttentionSubmodules:
    linear_q: LinearLayerBuilder
    linear_kv: LinearLayerBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: Union[ModuleSpec, type] = None
```

| Field | Type | Default | Description |
|---|---|---|---|
| `linear_q` | `LinearLayerBuilder` | required | Query projection from hidden_states |
| `linear_kv` | `LinearLayerBuilder` | required | Key-value projection from context |
| `core_attention` | `CoreAttentionBuilder` | required | Core attention mechanism |
| `linear_proj` | `Union[ModuleSpec, type]` | `None` | Output projection |

---

## Attention (Abstract Base Class)

`Attention` is the abstract base class for both `SelfAttention` and `CrossAttention`. It provides the shared infrastructure for QKV processing, rotary embeddings, inference KV caching, and output projection.

### __init__

```python
class Attention(MegatronModule, ABC):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
```

#### Parameters

| Parameter | Type | Description |
|---|---|---|
| `config` | `TransformerConfig` | Transformer configuration |
| `submodules` | `SelfAttentionSubmodules` or `CrossAttentionSubmodules` | Submodule specs |
| `layer_number` | `int` | 1-based global layer number |
| `attn_mask_type` | `AttnMaskType` | Mask type (`padding`, `causal`, `no_mask`, etc.) |
| `attention_type` | `str` | `"self"` or `"cross"` |
| `cp_comm_type` | `str \| None` | Context parallel communication type |
| `pg_collection` | `ProcessGroupCollection \| None` | Process groups |
| `pp_layer_offset` | `Optional[int]` | Pipeline parallel layer offset |

#### Key Computed Attributes

| Attribute | Type | Description |
|---|---|---|
| `query_projection_size` | `int` | `kv_channels * num_attention_heads` |
| `kv_projection_size` | `int` | `kv_channels * num_query_groups` |
| `hidden_size_per_attention_head` | `int` | `query_projection_size / num_attention_heads` |
| `num_attention_heads_per_partition` | `int` | Heads per TP rank |
| `num_query_groups_per_partition` | `int` | KV groups per TP rank |
| `core_attention` | `Module` | The core attention module (e.g., DotProductAttention) |
| `linear_proj` | `Module` | Output projection (row-parallel linear) |
| `checkpoint_core_attention` | `bool` | Whether to checkpoint core attention |
| `offload_qkv_linear` | `bool` | Whether to offload QKV computation input |
| `offload_core_attention` | `bool` | Whether to offload core attention input |
| `offload_attn_proj` | `bool` | Whether to offload output projection input |

### forward

```python
def forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    key_value_states: Optional[Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    rotary_pos_cos_sin: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[Tensor] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
) -> Tuple[Tensor, Tensor]:
```

#### Forward Parameters

| Parameter | Shape | Description |
|---|---|---|
| `hidden_states` | `[s, b, h]` | Input hidden states |
| `attention_mask` | `[1, 1, sq, sk]` | Attention mask |
| `key_value_states` | `[s, b, h]` | Cross-attention encoder output |
| `inference_context` | `BaseInferenceContext` | KV cache manager for inference |
| `rotary_pos_emb` | `Tuple[Tensor, Tensor]` | (q_emb, k_emb) rotary embeddings |
| `rotary_pos_cos` | `[max_s, 1, 1, d]` | Cosine for flash decode |
| `rotary_pos_sin` | `[max_s, 1, 1, d]` | Sine for flash decode |
| `rotary_pos_cos_sin` | `[max_s, 1, 1, 2d]` | Combined for flashinfer RoPE |
| `attention_bias` | `[b, num_head, sq, skv]` | Bias for Q*K^T |
| `packed_seq_params` | `PackedSeqParams` | THD format parameters |
| `sequence_len_offset` | `Tensor` | Inference decode offset |

#### Returns

A tuple `(output, bias)` where:
- **output**: `[s, b, h]` - attention output
- **bias**: Output bias tensor or `None`

#### Forward Flow

1. **RoPE skip check**: If `no_rope_freq` indicates RoPE should be skipped for this layer, sets `rotary_pos_emb = None`.
2. **QKV computation**: Calls `get_query_key_value_tensors()` to produce query, key, value tensors. Supports fused QKV+RoPE when `fused_single_qkv_rope=True`.
3. **Inference KV cache**: Calls `_adjust_key_value_for_inference()` for static batching inference (appends KV to cache). For flash decode, applies RoPE before storing and calls `flash_decode()`.
4. **Rotary position embeddings**: Applies RoPE to query and key. Supports THD format (packed sequences).
5. **Core attention**: Runs through `core_attention` module. Uses selective checkpointing if enabled. For dynamic batching inference, uses `flash_decode_and_prefill()`.
6. **Output gate**: If `attention_output_gate=True`, applies sigmoid gating.
7. **Output projection**: Projects from head dimension back to hidden size via `linear_proj`.

---

## SelfAttention

`SelfAttention` extends `Attention` for decoder self-attention. It creates a fused `linear_qkv` projection and optional Q/K layernorms.

### __init__

```python
class SelfAttention(Attention):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
    ):
```

#### Internal Modules

1. **linear_qkv**: Column-parallel linear projection from `hidden_size` to `query_projection_size + 2 * kv_projection_size` (plus gate dimensions if `attention_output_gate=True`).
2. **q_layernorm**: Optional per-head Q normalization. Created when `qk_layernorm=True` or `qk_l2_norm=True`. Uses `TENorm` or `L2Norm` respectively.
3. **k_layernorm**: Optional per-head K normalization.

### get_query_key_value_tensors

```python
def get_query_key_value_tensors(
    self,
    hidden_states: Tensor,
    key_value_states: Optional[Tensor] = None,
    output_gate: bool = False,
    split_qkv: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, List[int]]]:
```

Projects `hidden_states` through `linear_qkv` and splits the result into separate Q, K, V tensors.

**QKV Weight Layout**: The `linear_qkv` output is organized as interleaved groups: `[q_heads, k_head, v_head]` per query group. With GQA where `num_query_groups < tp_size`, weights are interleaved as `q1 q2 k1 v1 | q3 q4 k2 v2 | ...` and an all-gather plus slicing is performed.

**GQA Expansion**: When `num_attention_heads > num_query_groups`, each KV head is shared by multiple query heads. The key and value are logically repeated to match the number of query heads during attention computation.

**Output Gate**: If `output_gate=True`, returns `(query, key, value, gate)` where gate has the same shape as query.

**Fused QKV+RoPE**: If `split_qkv=False`, returns `(mixed_qkv, split_arg_list)` for use with `apply_fused_qkv_rotary_pos_emb`.

### QK Clipping (Experimental)

When `config.qk_clip=True`, `SelfAttention.clip_qk()` implements attention logit clipping:

```python
# Per-group balancing factor
eta = clamp(threshold / max_attention_logits, max=1.0)
# Apply to weights
Q_weight *= eta ** alpha
K_weight *= eta ** (1 - alpha)
```

### Real-Time Tests

When `config.test_mode=True`, `run_realtime_tests()` verifies that Q/K layernorm parameters are consistent across TP and DP ranks. This catches silent hardware failures (memory corruption, network errors).

---

## CrossAttention

`CrossAttention` extends `Attention` for encoder-decoder cross-attention. It uses separate `linear_q` and `linear_kv` projections.

### __init__

```python
class CrossAttention(Attention):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: CrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
```

**Constraint**: GQA (`num_query_groups != num_attention_heads`) is not supported in cross-attention.

### get_query_key_value_tensors

```python
def get_query_key_value_tensors(
    self,
    hidden_states: Tensor,
    key_value_states: Optional[Tensor],
    output_gate: bool = False,
    split_qkv: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
```

Projects `hidden_states` to query via `linear_q`, and `key_value_states` to key/value via `linear_kv`. Always splits (no fused QKV+RoPE support).

---

## DotProductAttention

`DotProductAttention` is the local (PyTorch-only) implementation of the core attention mechanism. It does not support context parallelism or packed sequences (use `TEDotProductAttention` for those features).

### __init__

```python
class DotProductAttention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
```

#### Key Attributes

| Attribute | Description |
|---|---|
| `hidden_size_per_partition` | Attention dimension per TP partition |
| `hidden_size_per_attention_head` | Per-head dimension |
| `num_attention_heads_per_partition` | Heads per TP rank |
| `num_query_groups_per_partition` | KV groups per TP rank |
| `softmax_scale` | Q*K^T scaling factor (1/sqrt(d_head) by default) |
| `scale_mask_softmax` | FusedScaleMaskSoftmax module |
| `attention_dropout` | nn.Dropout layer |

### forward

```python
def forward(
    self,
    query: Tensor,     # [sq, b, np, hn]
    key: Tensor,       # [sk, b, ng, hn]
    value: Tensor,     # [sv, b, ng, hn]
    attention_mask: Optional[Tensor],
    attn_mask_type: Optional[AttnMaskType] = None,
    attention_bias: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> Tensor:  # [sq, b, hp]
```

#### Algorithm

1. **GQA Key/Value Expansion**: If `num_attention_heads > num_query_groups`, repeats key/value along the head dimension using `repeat_interleave` to match the number of query heads.

2. **Reshape for batched matmul**: Reshapes Q to `[sq, b*np, hn]` and K to `[sk, b*np, hn]`.

3. **QK^T computation**: Uses `torch.baddbmm` to compute `softmax_scale * Q @ K^T`, producing attention scores of shape `[b*np, sq, sk]`.

4. **Softmax + mask**: Applies `FusedScaleMaskSoftmax` which handles masking and softmax in potentially fused fp32.

5. **Dropout**: Applies attention dropout (with RNG fork for non-sequence-parallel).

6. **Attention x Value**: `torch.bmm(attention_probs, value)` produces context of shape `[b*np, sq, hn]`.

7. **Reshape output**: Permutes and reshapes to `[sq, b, hp]` where `hp = np * hn`.

#### Softmax Type Variants

| Type | Behavior |
|---|---|
| `"vanilla"` | Standard softmax |
| `"off-by-one"` | Fixed per-head offset added before softmax |
| `"learnable"` | Learnable per-head offset parameter (initialized with `init_method`) |

---

## Attention Backends

Megatron-Core supports multiple attention backends selected via `config.attention_backend`:

### Auto (`AttnBackend.auto`)
The default. TransformerEngine selects the best available backend (cuDNN Fused Attention or Flash Attention).

### Local (`AttnBackend.local`)
Uses `DotProductAttention` - the pure PyTorch implementation. No context parallelism or packed sequence support. Useful for debugging.

### Flash (`AttnBackend.flash`)
Uses FlashAttention kernels (v2, v3, or v4 depending on availability). Supports:
- Sliding window attention
- Packed sequences (THD format)
- Inference KV cache via `flash_attn_with_kvcache`
- Batch-invariant mode for deterministic execution

### TE (`AttnBackend.te`)
Uses TransformerEngine's attention implementation. Supports:
- Context parallelism
- FP8 attention
- Fused QKV + RoPE
- CUDA graph capture
- cuDNN Fused Attention backend

---

## GQA (Grouped Query Attention)

GQA reduces memory and compute by sharing key/value heads across multiple query heads. Configuration:

```python
config = TransformerConfig(
    num_attention_heads=32,     # 32 query heads
    num_query_groups=8,         # 8 KV groups (each shared by 4 query heads)
    # ... other params
)
```

### How GQA Works in Megatron-Core

1. **Projection**: `linear_qkv` outputs `[q_heads_per_group + 1 k_head + 1 v_head]` per group.
2. **Split**: The output is split into query (`[np/ng * hn]`), key (`[hn]`), and value (`[hn]`) per group.
3. **Query reshape**: Query is reshaped from `[sq, b, ng, np/ng * hn]` to `[sq, b, np, hn]`.
4. **KV expansion**: In `DotProductAttention`, key and value are repeated via `repeat_interleave` to match query heads. In TE/Flash backends, GQA is handled natively without explicit expansion.

### Special Case: num_query_groups < tp_size

When the number of KV groups is smaller than the tensor parallel degree, Megatron-Core performs an all-gather on the QKV output before splitting. This ensures each rank gets the correct QKV slice for its assigned query heads.

---

## MLA (Multi-Latent Attention)

MLA (from DeepSeek) projects Q, K, V into lower-dimensional latent spaces, significantly reducing KV cache size during inference.

### Configuration

```python
from megatron.core.transformer import MLATransformerConfig

config = MLATransformerConfig(
    q_lora_rank=512,       # Query low-rank dimension
    kv_lora_rank=512,      # KV low-rank dimension
    qk_head_dim=128,       # QK head dimension (excluding position)
    qk_pos_emb_head_dim=64, # Position embedding dimension in QK
    v_head_dim=128,         # Value head dimension
    rope_type="yarn",       # YaRN RoPE
    # ... standard params
)
```

### MLA Architecture

```
hidden_states
  |-- [q_down_proj] --> q_latent [sq, b, q_lora_rank]
  |     |-- [q_up_proj] --> query [sq, b, np, qk_head_dim + qk_pos_emb_head_dim]
  |           |-- split --> q_nope [sq, b, np, qk_head_dim]
  |           |            q_rope  [sq, b, np, qk_pos_emb_head_dim]
  |
  |-- [kv_down_proj] --> kv_latent [sq, b, kv_lora_rank]
        |-- [kv_up_proj] --> k_nope [sq, b, 1, qk_head_dim]
        |                   v      [sq, b, 1, v_head_dim]
        |-- apply RoPE to kv_latent --> k_rope [sq, b, 1, qk_pos_emb_head_dim]

key = concat(k_nope, k_rope)  # [sq, b, 1, qk_head_dim + qk_pos_emb_head_dim]
value = v                       # [sq, b, 1, v_head_dim]
```

### MLA Inference Optimizations

- **Latent caching** (`cache_mla_latents=True`): Caches the low-dimensional KV latent instead of the full KV, reducing cache size. Requires Flash MLA.
- **Down-projection fusion** (`mla_down_proj_fusion=True`): Fuses the Q/KV down-projection with input layernorm.

---

## RoPE (Rotary Position Embeddings)

### Standard RoPE

```python
config = TransformerConfig(
    rotary_interleaved=False,  # LLaMA style (first/second half)
    apply_rope_fusion=True,    # Use fused kernel
)
```

Applied to query and key before core attention. The `apply_rotary_pos_emb` function handles the rotation.

### YaRN RoPE (for MLA)

```python
config = MLATransformerConfig(
    rope_type="yarn",
    rotary_base=10000,
    rotary_scaling_factor=40,
    original_max_position_embeddings=4096,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=0.0,
)
```

YaRN (Yet another RoPE extensioN) extends context length by modifying the RoPE frequencies with concentration factors.

### RoPE Skipping

```python
config = TransformerConfig(
    no_rope_freq=4,  # Skip RoPE every 4th layer
)
# Or per-layer control:
config = TransformerConfig(
    no_rope_freq=[0, 1, 1, 0],  # 0=apply, 1=skip
)
```

### Fused QKV + RoPE

```python
config = TransformerConfig(
    fused_single_qkv_rope=True,
)
```

Avoids splitting QKV before RoPE application. The RoPE is applied to the unsplit mixed QKV tensor using TE's `apply_fused_qkv_rotary_pos_emb`, which is more efficient as it avoids the split-concat round-trip.

---

## Sliding Window Attention

Sliding window attention limits each token to attend only to a local window of tokens, reducing memory from O(s^2) to O(s * w).

### Configuration

```python
config = TransformerConfig(
    window_size=(256, -1),       # (left, right); -1 = infinite
    window_attn_skip_freq=4,     # Full attention every 4th layer
)
```

### How It Works

1. In `DotProductAttention`, the `FusedScaleMaskSoftmax` module receives the `window_size` and applies the window mask in addition to any causal/padding mask.
2. In `SelfAttention.forward()`, `is_layer_window_attention()` checks whether the current layer should use windowed attention based on `window_size` and `window_attn_skip_freq`.
3. For TE/Flash backends, the window size is passed directly to the attention kernel.

### Window Skip Pattern

`window_attn_skip_freq` controls the interleaving of full and windowed attention:
- Integer N: Full attention at every N-th layer, SWA in between.
- List: Custom binary pattern (1 = SWA, 0 = full attention).

---

## Inference Attention Optimizations

### Static Batching (Flash Decode)

```python
config = TransformerConfig(
    flash_decode=True,
    use_inference_optimized_layers=True,
)
```

Flash decode fuses RoPE computation, KV cache update, and flash attention into a single kernel call during decode:

1. Compute RoPE with precomputed cos/sin tensors
2. Update the KV cache
3. Perform flash attention against the full cached sequence

### Dynamic Batching

Dynamic batching uses variable-length sequences with paged KV cache (block tables):

```python
# Handled automatically by inference_context
# Uses flash_attn_varlen_func or FA3/FA4 with block_table support
```

The `flash_decode_and_prefill()` method handles mixed decode/prefill batches:
- **Prefill**: Uses `flash_attn_varlen_func` with cumulative sequence lengths.
- **Decode**: Uses `flash_attn_with_kvcache` with block tables for paged attention.
- **MLA decode**: Uses Flash MLA kernel with specialized metadata.

### Inference Attention Backends

| Backend | FA4 | FA3 | FA2 |
|---|---|---|---|
| Prefill | `flash_attn4_varlen_func` | `_flash_attn_forward` wrapper | `flash_attn_varlen_func` |
| Decode | `flash_attn4_varlen_func` | `flash_attn3_with_kvcache` | `flash_attn_with_kvcache` |
| MLA Decode | N/A | N/A | `flash_mla_with_kvcache` |

---

## Code Examples

### Standard Self-Attention with TE

```python
from megatron.core.transformer import TransformerConfig, SelfAttention, SelfAttentionSubmodules
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TEDotProductAttention,
)

config = TransformerConfig(
    num_attention_heads=32,
    hidden_size=4096,
    num_query_groups=8,  # GQA
    bf16=True,
)

attn_spec = SelfAttentionSubmodules(
    linear_qkv=TEColumnParallelLinear,
    core_attention=TEDotProductAttention,
    linear_proj=TERowParallelLinear,
)

self_attn = SelfAttention(
    config=config,
    submodules=attn_spec,
    layer_number=1,
)
```

### Local Attention with DotProductAttention

```python
from megatron.core.transformer import DotProductAttention
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

attn_spec = SelfAttentionSubmodules(
    linear_qkv=ColumnParallelLinear,
    core_attention=DotProductAttention,
    linear_proj=RowParallelLinear,
)

config = TransformerConfig(
    num_attention_heads=32,
    hidden_size=4096,
    attention_backend="local",
)

self_attn = SelfAttention(config=config, submodules=attn_spec, layer_number=1)
```

### Cross-Attention

```python
from megatron.core.transformer import CrossAttention, CrossAttentionSubmodules

cross_attn_spec = CrossAttentionSubmodules(
    linear_q=TEColumnParallelLinear,
    linear_kv=TEColumnParallelLinear,
    core_attention=TEDotProductAttention,
    linear_proj=TERowParallelLinear,
)

cross_attn = CrossAttention(
    config=config,
    submodules=cross_attn_spec,
    layer_number=1,
)

# Forward pass
output, bias = cross_attn(
    hidden_states=decoder_hidden,    # [sq, b, h]
    attention_mask=None,
    key_value_states=encoder_output,  # [sk, b, h]
)
```

### Sliding Window Attention

```python
config = TransformerConfig(
    num_attention_heads=32,
    hidden_size=4096,
    window_size=(4096, -1),     # Attend to last 4096 tokens
    window_attn_skip_freq=4,    # Full attention every 4th layer
)
# The window mask is automatically applied in the core attention
```

### GQA with QK Norm

```python
config = TransformerConfig(
    num_attention_heads=64,
    hidden_size=8192,
    num_query_groups=8,
    qk_layernorm=True,         # Normalize Q and K after projection
    layernorm_epsilon=1e-6,
    normalization="RMSNorm",
)

attn_spec = SelfAttentionSubmodules(
    linear_qkv=TEColumnParallelLinear,
    core_attention=TEDotProductAttention,
    linear_proj=TERowParallelLinear,
    # q_layernorm and k_layernorm will use TENorm by default
    # when qk_layernorm=True
)
```

### MLA Configuration

```python
from megatron.core.transformer import MLATransformerConfig

config = MLATransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    q_lora_rank=768,
    kv_lora_rank=512,
    qk_head_dim=192,            # Includes position embedding
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
    rope_type="yarn",
    rotary_base=10000,
    rotary_scaling_factor=40,
    cache_mla_latents=True,     # Cache latents for inference
    mla_down_proj_fusion=True,  # Fuse down-projection
)
```

### Attention with Output Gate

```python
config = TransformerConfig(
    num_attention_heads=32,
    hidden_size=4096,
    attention_output_gate=True,
)

# SelfAttention.get_query_key_value_tensors returns (query, key, value, gate)
# The gate is applied as: output * sigmoid(gate)
```

### Using per-token Scale with MoE Expert MLP

```python
# Per-token scale is used in quantized MoE inference
# MLP.forward accepts per_token_scale parameter
output, bias = mlp(
    hidden_states=hidden_states,
    per_token_scale=per_token_scale,  # [s, b] scale factor
)
# The scale is applied to the fc1 output before activation
```
