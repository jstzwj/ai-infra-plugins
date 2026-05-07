# Chapter 04: TransformerConfig Reference

## Source Files
- `sources/Megatron-LM/megatron/core/transformer/transformer_config.py`

## Overview

`TransformerConfig` extends `ModelParallelConfig` and serves as the central configuration dataclass for all megatron-core transformer models. It controls model architecture, attention mechanisms, MoE configuration, quantization (FP8/FP4), activation recomputation, CUDA graphs, inference optimization, and more. A specialized subclass, `MLATransformerConfig`, adds parameters for Multi-Latent Attention models such as DeepSeek.

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    bf16=True,
    sequence_parallel=True,
    tensor_model_parallel_size=4,
)
```

---

## Model Architecture

### num_layers
- **Type**: `int`
- **Default**: `0`
- **Description**: Total number of transformer layers (decoder blocks) in the model. This is the global count across all pipeline stages.

### mtp_num_layers
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of Multi-Token Prediction (MTP) layers. MTP extends prediction to multiple future tokens at each position using D sequential modules to predict D additional tokens.

### mtp_loss_scaling_factor
- **Type**: `Optional[float]`
- **Default**: `0.1`
- **Description**: Weighting factor for MTP loss. The average MTP loss across all depths is multiplied by this factor.

### mtp_use_repeated_layer
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use a single MTP layer repeatedly instead of multiple separate layers.

### mtp_hybrid_override_pattern
- **Type**: `Optional[str]`
- **Default**: `None`
- **Description**: DEPRECATED. Use the unified `hybrid_layer_pattern` instead. Legacy argument for loading old checkpoints.

### num_layers_in_first_pipeline_stage
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of transformer layers on the first pipeline stage. `None` implies equal layer division across PP ranks.

### num_layers_in_last_pipeline_stage
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of transformer layers on the last pipeline stage. `None` implies equal layer division across PP ranks.

### pipeline_model_parallel_layout
- **Type**: `Optional[Union[str, list, PipelineParallelLayerLayout]]`
- **Default**: `None`
- **Description**: Custom pipeline parallel partitioning definition. Accepts a string (e.g., `'Et*3|(tt|)*29,m|L'`), a list of layer lists, or a `PipelineParallelLayerLayout` object. See the TransformerConfig docstring for detailed syntax examples.

### account_for_embedding_in_pipeline_split
- **Type**: `bool`
- **Default**: `False`
- **Description**: Treat the embedding layer as a standard transformer layer for PP partitioning and placement.

### account_for_loss_in_pipeline_split
- **Type**: `bool`
- **Default**: `False`
- **Description**: Treat the loss layer as a standard transformer layer for PP partitioning and placement.

### hidden_size
- **Type**: `int`
- **Default**: `0`
- **Description**: Transformer hidden size (denoted `h`). The dimensionality of the hidden representations throughout the model.

### num_attention_heads
- **Type**: `int`
- **Default**: `0`
- **Description**: Number of attention heads. Must be divisible by `tensor_model_parallel_size`.

### ffn_hidden_size
- **Type**: `Optional[int]`
- **Default**: `None`
- **Auto-computed**: Set to `4 * hidden_size` if not provided.
- **Description**: Feed-Forward Network hidden dimension. The intermediate projection size in the MLP block.

### kv_channels
- **Type**: `Optional[int]`
- **Default**: `None`
- **Auto-computed**: Set to `hidden_size // num_attention_heads` if not provided.
- **Description**: Per-head projection dimension in multi-head attention. Equals the head dimension.

### hidden_dropout
- **Type**: `float`
- **Default**: `0.1`
- **Description**: Dropout probability applied after the residual connection (bias-dropout-add) in both attention and MLP paths.

### fp32_residual_connection
- **Type**: `bool`
- **Default**: `False`
- **Description**: Cast residual connections to fp32. When enabled, also forces `pipeline_dtype` to `torch.float`.

### apply_residual_connection_post_layernorm
- **Type**: `bool`
- **Default**: `False`
- **Description**: If True, uses the original BERT residual connection ordering (post-layernorm).

### layernorm_epsilon
- **Type**: `float`
- **Default**: `1e-5`
- **Description**: Epsilon value for LayerNorm/RMSNorm operations to avoid division by zero.

### layernorm_zero_centered_gamma
- **Type**: `bool`
- **Default**: `False`
- **Description**: If True, centers LayerNorm gamma values around zero for improved numerical stability (also called LayerNorm 1p).

### add_bias_linear
- **Type**: `bool`
- **Default**: `True`
- **Description**: Include bias in all linear layers (QKV projections, output projection, MLP fc1/fc2).

### add_qkv_bias
- **Type**: `bool`
- **Default**: `False`
- **Description**: Add bias only to QKV projections (overrides the general bias setting for QKV).

### gated_linear_unit
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use a gated linear unit (SwiGLU/GeGLU) in the MLP. When enabled, `linear_fc1` output width is doubled to accommodate the gating mechanism.

### activation_func
- **Type**: `Callable[[torch.Tensor], torch.Tensor]`
- **Default**: `F.gelu`
- **Description**: Activation function for the MLP. Common choices: `F.gelu`, `F.silu`, `F.relu`.

### activation_func_fp8_input_store
- **Type**: `bool`
- **Default**: `False`
- **Description**: Store the input of the MLP activation function in FP8 for backprop to save memory. Only supported with SwiGLU (`activation_func=F.silu` and `gated_linear_unit=True`).

### glu_linear_offset
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Offset in the GLU activation: `activation_func(x[0]) * (x[1] + offset)`. Only used when `gated_linear_unit=True`.

### activation_func_clamp_value
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Clamp value for linear_fc1 output. Only used when `activation_func` is `quick_gelu`.

### normalization
- **Type**: `Literal['LayerNorm', 'RMSNorm']`
- **Default**: `"LayerNorm"`
- **Description**: Normalization layer type.

### is_hybrid_model
- **Type**: `bool`
- **Default**: `False`
- **Description**: Indicates whether this is a hybrid model (affects initialization scaling).

### heterogeneous_block_specs
- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether to use heterogeneous block specs (Nemotron-NAS style architecture).

### hetereogenous_dist_checkpoint
- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether to use heterogeneous layers in distributed checkpoint.

### transformer_impl
- **Type**: `Literal['local', 'transformer_engine', 'inference_optimized']`
- **Default**: `"transformer_engine"`
- **Description**: Transformer implementation backend. `"transformer_engine"` uses NVIDIA TransformerEngine, `"local"` uses Megatron-Core's PyTorch implementation, and `"inference_optimized"` uses inference-specific layers.

---

## Attention

### attention_backend
- **Type**: `AttnBackend`
- **Default**: `AttnBackend.auto`
- **Description**: Attention backend selection. Options: `auto` (let TE decide), `local` (PyTorch), `flash` (FlashAttention), `te` (TransformerEngine cuDNN/Fused).

### softmax_scale
- **Type**: `Optional[float]`
- **Default**: `None`
- **Auto-computed**: Set to `1.0 / sqrt(kv_channels)` by default (or adjusted by MuP).
- **Description**: Scale factor applied to Q*K^T before softmax. When `None`, defaults to `1/sqrt(d_head)`.

### softmax_type
- **Type**: `Literal['vanilla', 'off-by-one', 'learnable']`
- **Default**: `'vanilla'`
- **Description**: Softmax variant. `"off-by-one"` and `"learnable"` apply a per-head offset as described in [Attention Is Off By One](https://www.evanmiller.org/attention-is-off-by-one.html).

### num_query_groups
- **Type**: `Optional[int]`
- **Default**: `None`
- **Auto-computed**: Set to `num_attention_heads` if not provided (i.e., standard MHA).
- **Description**: Number of KV groups for Grouped Query Attention (GQA). Must be a multiple or divisor of `tensor_model_parallel_size`.

### window_size
- **Type**: `Optional[Tuple[int, int]]`
- **Default**: `None`
- **Description**: Sliding window attention window. Tuple of (left, right) sizes. `-1` means infinite. `None` disables sliding window.

### window_attn_skip_freq
- **Type**: `Optional[Union[int, List[int]]]`
- **Default**: `None`
- **Description**: Frequency of full-attention layers among sliding-window layers. An integer N means one full-attention layer after every (N-1) SWA layers. A list defines a custom binary pattern.

### multi_latent_attention
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable Multi-Latent Attention (MLA) as used in DeepSeek models.

### no_rope_freq
- **Type**: `Optional[Union[int, List[int]]]`
- **Default**: `None`
- **Description**: Controls which layers apply RoPE. An integer N skips RoPE every N-th layer. A list defines per-layer RoPE application (1 = skip, 0 = apply).

### qk_layernorm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Apply normalization to query and key embeddings after the QKV projection.

### qk_l2_norm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Apply LLaMA 4-style L2 normalization to query and key embeddings.

### qk_clip
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable QK clipping to prevent attention logit explosion.

### qk_clip_alpha
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Balancing alpha for QK clipping: `Q = Q * (eta ** alpha)`.

### qk_clip_threshold
- **Type**: `float`
- **Default**: `100`
- **Description**: Threshold for QK clipping: `eta = min(threshold / max_attention_logits, 1.0)`.

### log_max_attention_logit
- **Type**: `bool`
- **Default**: `False`
- **Description**: Log the maximum attention logit across the whole model.

### attention_output_gate
- **Type**: `bool`
- **Default**: `False`
- **Description**: Apply a learned output gate to the attention layers.

### rotary_interleaved
- **Type**: `bool`
- **Default**: `False`
- **Description**: RoPE interleaving style. `False` = LLaMA style (first half/second half), `True` = RoFormer style (even/odd pairs).

### mrope_section
- **Type**: `Optional[List[int]]`
- **Default**: `None`
- **Description**: Multimodal RoPE section dimensions for temporal, height, and width channels.

---

## DSA (Data-Dependent Sparse Attention)

### experimental_attention_variant
- **Type**: `Optional[Literal['gated_delta_net', 'dsa']]`
- **Default**: `None`
- **Description**: Experimental attention variant. `"gated_delta_net"` enables the gated delta net linear attention. `"dsa"` enables Data-Dependent Sparse Attention.

### dsa_indexer_n_heads
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of DSA indexer heads.

### dsa_indexer_head_dim
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Dimension per DSA indexer head.

### dsa_indexer_topk
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of top-k tokens to select in DSA indexer.

### dsa_indexer_loss_coeff
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Coefficient for DSA indexer KL divergence loss.

### dsa_indexer_use_sparse_loss
- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether to use sparse DSA indexer loss (computed using top-k indices).

---

## Linear Attention

### linear_attention_freq
- **Type**: `Optional[Union[int, List[int]]]`
- **Default**: `None`
- **Description**: Frequency between LA (linear attention) and SDPA layers. Integer N = (N-1) LA layers for every 1 SDPA layer. A list defines a custom pattern.

### linear_conv_kernel_dim
- **Type**: `Optional[int]`
- **Default**: `4`
- **Description**: Conv kernel dimension for the gated delta net.

### linear_key_head_dim
- **Type**: `Optional[int]`
- **Default**: `128`
- **Description**: Query and key head dimension for linear attention.

### linear_value_head_dim
- **Type**: `Optional[int]`
- **Default**: `128`
- **Description**: Value and gate head dimension for linear attention.

### linear_num_key_heads
- **Type**: `Optional[int]`
- **Default**: `16`
- **Description**: Number of query and key heads for linear attention. Must be divisible by `TP * CP`.

### linear_num_value_heads
- **Type**: `Optional[int]`
- **Default**: `32`
- **Description**: Number of value and gate heads for linear attention. Must be divisible by `TP * CP` and a multiple of `linear_num_key_heads`.

---

## Initialization

### init_method
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Auto-computed**: Set to `init_method_normal(init_method_std)` (with MuP scaling if enabled).
- **Description**: Weight initialization function. Takes a single tensor and initializes it. Bias is always initialized to zero.

### output_layer_init_method
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Auto-computed**: Set to `scaled_init_method_normal(init_method_std, num_layers)`.
- **Description**: Initialization for output layers (attention projection, MLP fc2). Uses scaled init: `std / sqrt(2 * num_layers)`.

### init_method_std
- **Type**: `float`
- **Default**: `0.02`
- **Description**: Standard deviation for the default zero-mean normal initialization.

### embedding_init_method
- **Type**: `Optional[Callable]`
- **Default**: `None`
- **Auto-computed**: Set using `embedding_init_method_std`.
- **Description**: Embedding layer initialization function.

### embedding_init_method_std
- **Type**: `Optional[float]`
- **Default**: `None`
- **Auto-computed**: Set to `init_method_std`.
- **Description**: Standard deviation for embedding initialization. Setting this avoids loss spikes during training (see [arXiv:2312.16903](https://arxiv.org/abs/2312.16903)). Also skips weight decay on embeddings.

### init_model_with_meta_device
- **Type**: `bool`
- **Default**: `False`
- **Description**: Initialize model on meta device for large model training. Only works with Megatron FSDP.

---

## MuP (Maximal Update Parameterization)

### use_mup
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable MuP for hyperparameter transfer across model widths. Scales learning rates and initialization according to the width multiplier.

### mup_width_mult
- **Type**: `float`
- **Default**: `1.0`
- **Auto-computed**: `hidden_size / mup_base_hidden_size` when `use_mup=True`.
- **Description**: Width multiplier for MuP scaling.

### mup_base_hidden_size
- **Type**: `Optional[int]`
- **Default**: `None`
- **Auto-computed**: Set to `hidden_size` (base model case).
- **Description**: Base hidden size for MuP width scaling. Set to your proxy model's hidden size.

### mup_embedding_mult
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Multiplier for embedding layer output.

### mup_output_mult
- **Type**: `float`
- **Default**: `1.0`
- **Auto-computed**: Set to `1.0 / mup_width_mult` when MuP is enabled and width differs.
- **Description**: Multiplier for output logits before softmax. Keeps output variance stable across widths.

### mup_base_head_dim
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Base head dimension for MuP attention scaling. When set, `softmax_scale = sqrt(mup_base_head_dim) / (kv_channels ** mup_attn_scale_power)`.

### mup_attn_scale_power
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Power for attention scaling. `0.5` = standard (1/sqrt(d_head)), `1.0` = MuP (1/d_head).

---

## MoE (Mixture of Experts)

### num_moe_experts
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of experts. When set, MLP is replaced with a MoE layer. `None` disables MoE.

### moe_layer_freq
- **Type**: `Union[int, List[int]]`
- **Default**: `1`
- **Description**: Frequency of MoE layers. Integer N = one expert layer for every N-1 dense layers. A list defines a custom pattern.

### moe_ffn_hidden_size
- **Type**: `Optional[int]`
- **Default**: `None`
- **Auto-computed**: Set to `ffn_hidden_size` if `num_moe_experts` is set.
- **Description**: Per-expert FFN hidden size.

### moe_router_topk
- **Type**: `int`
- **Default**: `2`
- **Description**: Number of experts each token is routed to.

### moe_router_load_balancing_type
- **Type**: `Union[str, List[str]]`
- **Default**: `"aux_loss"`
- **Description**: Load balancing strategy. Options: `"aux_loss"` (GShard), `"seq_aux_loss"` (DeepSeek), `"global_aux_loss"`, `"sinkhorn"`, `"none"`. A list combines multiple types.

### moe_aux_loss_coeff
- **Type**: `Union[float, List[float]]`
- **Default**: `0.0`
- **Description**: Scaling coefficient for aux loss. Recommended starting value: `1e-2`. If a list is given for load balancing types, a matching list of coefficients is required.

### moe_z_loss_coeff
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Scaling coefficient for z-loss. Recommended starting value: `1e-3`.

### moe_grouped_gemm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use grouped GEMM to batch multiple expert GEMMs into a single kernel launch for better utilization.

### moe_token_dispatcher_type
- **Type**: `Literal['allgather', 'alltoall', 'flex']`
- **Default**: `"allgather"`
- **Description**: Token dispatcher type. `"allgather"` is the default; `"alltoall"` supports expert parallelism; `"flex"` supports DeepEP and HybridEP.

### moe_router_score_function
- **Type**: `Literal['softmax', 'sigmoid', 'sqrtsoftplus']`
- **Default**: `"softmax"`
- **Description**: Score function for routing.

### moe_router_pre_softmax
- **Type**: `bool`
- **Default**: `False`
- **Description**: Apply softmax before top-k selection. By default, softmax is applied after top-k.

### moe_router_topk_scaling_factor
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Scaling factor for routing score in top-k selection (only with pre-softmax).

### moe_router_dtype
- **Type**: `Optional[Literal['fp32', 'fp64']]`
- **Default**: `None`
- **Description**: Data type for routing computations. Recommended for large expert counts (>=32).

### moe_router_enable_expert_bias
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable dynamic per-expert bias for aux-loss-free load balancing (DeepSeekV3 style). Requires `sigmoid` or `sqrtsoftplus` score function.

### moe_router_bias_update_rate
- **Type**: `float`
- **Default**: `1e-3`
- **Description**: Update rate for expert bias in aux-loss-free routing.

### moe_router_force_load_balancing
- **Type**: `bool`
- **Default**: `False`
- **Description**: [Experimental] Force load balancing with random logits for benchmarking.

### moe_router_force_biased
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: [Experimental] Apply random bias with specified std to router logits. Positive = per-forward, negative = per-layer.

### moe_router_num_groups
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of groups for group-limited routing (DeepSeek-V2/V3 style). Must divide `num_moe_experts` evenly.

### moe_router_group_topk
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of groups to select in group-limited routing. Must be <= `moe_router_num_groups`.

### moe_router_padding_for_quantization
- **Type**: `Optional[bool]`
- **Default**: `False`
- **Description**: Pad routing map so tokens per expert are multiples of 16/32 for quantized precision (FP8/FP4).

### moe_enable_routing_replay
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable routing replay for MoE layers.

### moe_shared_expert_intermediate_size
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Total shared expert FFN size = `num_shared_experts * ffn_size_per_expert`. `None` means no shared expert.

### moe_shared_expert_gate
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable gating for the shared expert.

### moe_shared_expert_overlap
- **Type**: `bool`
- **Default**: `False`
- **Description**: Overlap shared expert computation with token dispatcher communication.

### moe_input_jitter_eps
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Add input jitter noise with specified epsilon.

### moe_token_dropping
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable token dropping (currently unsupported, must remain False).

### moe_expert_capacity_factor
- **Type**: `Optional[float]`
- **Default**: `None`
- **Description**: Expert capacity factor. `None` = no token dropping.

### moe_pad_expert_input_to_capacity
- **Type**: `bool`
- **Default**: `False`
- **Description**: Pad expert input to match capacity. Requires `moe_expert_capacity_factor`.

### moe_pad_experts_for_cuda_graph_inference
- **Type**: `bool`
- **Default**: `False`
- **Description**: Switch to drop-and-pad routing during inference decode.

### moe_token_drop_policy
- **Type**: `Literal['probs', 'position']`
- **Default**: `"probs"`
- **Description**: Token drop policy. `"probs"` drops lowest-probability tokens; `"position"` drops end-of-batch tokens.

### moe_layer_recompute
- **Type**: `bool`
- **Default**: `False`
- **Description**: DEPRECATED. Use `recompute_granularity='selective'` with `recompute_modules=['moe']`.

### moe_permute_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Fuse token rearrangement ops during dispatching. Requires TE >= 2.1.0.

### moe_router_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Fuse MoE TopK routing and aux-loss computation. Requires TE >= 2.7.0.

### moe_apply_probs_on_input
- **Type**: `bool`
- **Default**: `False`
- **Description**: Apply routing probabilities on expert input instead of after activation.

### moe_latent_size
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Latent projection dimension for MoE. Enables latent MoE projections.

### moe_per_layer_logging
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable per-layer MoE logging (aux loss, z-loss).

### moe_enable_deepep
- **Type**: `bool`
- **Default**: `False`
- **Description**: [Experimental] Enable DeepEP for efficient MoE token dispatching.

### moe_flex_dispatcher_backend
- **Type**: `Literal['deepep', 'hybridep']`
- **Default**: `"deepep"`
- **Description**: Backend for flex token dispatcher. `"hybridep"` supports MNNVL.

### moe_permute_fusion_into_hybridep
- **Type**: `bool`
- **Default**: `False`
- **Description**: Fuse token rearrangement into HybridEP dispatching.

### moe_deepep_num_sms
- **Type**: `int`
- **Default**: `20`
- **Description**: Number of SMs for DeepEP.

### moe_hybridep_num_sms
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of SMs for HybridEP.

### moe_hybridep_num_blocks_permute
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: CUDA thread blocks for HybridEP permute part.

### moe_hybridep_num_blocks_unpermute
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: CUDA thread blocks for HybridEP unpermute part.

---

## FP8 Quantization

### fp8
- **Type**: `Optional[Literal['e4m3', 'hybrid']]`
- **Default**: `None`
- **Description**: Enable FP8 precision via TransformerEngine. `"e4m3"` uses e4m3 for all FP8 tensors. `"hybrid"` uses e4m3 for forward (activations/weights) and e5m2 for backward (gradients).

### fp8_recipe
- **Type**: `Optional[Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise', 'custom']]`
- **Default**: `"delayed"`
- **Description**: FP8 scaling recipe. `"delayed"` = delayed scaling, `"mxfp8"` = MX FP8 (Blackwell), `"blockwise"` = blockwise scaling, `"tensorwise"` = per-tensor scaling, `"custom"` = custom quantizer.

### fp8_param
- **Type**: `bool`
- **Default**: `False`
- **Description**: Store parameters in FP8 to save memory. Must be used with `fp8` mode enabled.

### fp8_quantizer_factory
- **Type**: `Optional[str]`
- **Default**: `None`
- **Description**: Python import path to a custom quantizer factory. Required when `fp8_recipe='custom'`.

### fp8_margin
- **Type**: `int`
- **Default**: `0`
- **Description**: Margin for scaling factor computation.

### fp8_interval
- **Type**: `int`
- **Default**: `1`
- **Description**: DEPRECATED since TE v1.8.0. Controls scaling factor recomputation frequency.

### fp8_amax_history_len
- **Type**: `int`
- **Default**: `1`
- **Description**: Length of the amax history window for scaling factor computation.

### fp8_amax_compute_algo
- **Type**: `Literal['most_recent', 'max']`
- **Default**: `"most_recent"`
- **Description**: Algorithm for choosing amax. `"max"` = largest in history, `"most_recent"` = latest value.

### fp8_wgrad
- **Type**: `bool`
- **Default**: `True`
- **Description**: When False, override FP8 and compute wgrad in higher precision.

### fp8_dot_product_attention
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use FP8 implementation of Dot Product Attention.

### fp8_multi_head_attention
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use FP8 implementation of Multi Head Attention.

### tp_only_amax_red
- **Type**: `bool`
- **Default**: `False`
- **Description**: Reduce FP8 AMAX only within TP or TP-CP domain.

### first_last_layers_bf16
- **Type**: `bool`
- **Default**: `False`
- **Description**: Keep first and last N layers in BF16 instead of FP8.

### num_layers_at_start_in_bf16
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of starting layers kept in BF16 when `first_last_layers_bf16=True`.

### num_layers_at_end_in_bf16
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of ending layers kept in BF16 when `first_last_layers_bf16=True`.

### use_kitchen
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use the kitchen extension for transformer quantization.

### use_kitchen_attention
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use kitchen attention instead of TE attention.

### kitchen_attention_backend
- **Type**: `Literal["sdpa", "fa"]`
- **Default**: `"sdpa"`
- **Description**: Kitchen attention backend when `use_kitchen_attention=True`.

---

## FP4 Quantization

### fp4
- **Type**: `Optional[Literal['e2m1']]`
- **Default**: `None`
- **Description**: Enable FP4 precision via TransformerEngine. Only `'e2m1'` (NVFP4BlockScaling) is supported, requiring TE >= 2.7.0 and Blackwell+.

### fp4_recipe
- **Type**: `Optional[Literal['nvfp4', 'custom']]`
- **Default**: `"nvfp4"`
- **Description**: FP4 scaling recipe. `"nvfp4"` uses NVFP4BlockScaling for Blackwell+.

### fp4_param
- **Type**: `bool`
- **Default**: `False`
- **Description**: Store parameters in FP4 to save memory. Must be used with `fp4` mode.

### fp4_quantizer_factory
- **Type**: `Optional[str]`
- **Default**: `None`
- **Description**: Python import path for custom FP4 quantizer factory. Required when `fp4_recipe='custom'`.

---

## Mixed Precision

### apply_query_key_layer_scaling
- **Type**: `bool`
- **Default**: `False`
- **Description**: Scale Q*K^T by 1/layer-number. Improves fp16 stability. Also forces `attention_softmax_in_fp32=True`.

### attention_softmax_in_fp32
- **Type**: `bool`
- **Default**: `True`
- **Description**: Run attention masking and softmax in fp32.

### disable_bf16_reduced_precision_matmul
- **Type**: `bool`
- **Default**: `False`
- **Description**: Disable BF16 reduced-precision matmul accumulation.

---

## Fusion

### bias_activation_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Fuse bias addition with activation function. Supports `gelu`, `swiglu`, `quick_gelu`.

### masked_softmax_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use fused softmax kernel.

### persist_layer_norm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use persistent fused LayerNorm kernel (supports fixed set of hidden sizes).

### memory_efficient_layer_norm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use memory-efficient fused LayerNorm kernel from Apex (local layers only).

### bias_dropout_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use fused bias-dropout-add kernel.

### apply_rope_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use fused RoPE kernel. Requires TE >= 1.4.

### use_fused_weighted_squared_relu
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use fused weighted squared ReLU kernel for MoE.

### fused_single_qkv_rope
- **Type**: `bool`
- **Default**: `False`
- **Description**: Avoid splitting QKV before RoPE forward and concatenating RoPE dgrads.

### fused_residual_rmsnorm
- **Type**: `bool`
- **Default**: `False`
- **Description**: Fuse residual connection and RMSNorm backward pass (TE only). Requires `normalization='RMSNorm'`.

### use_te_activation_func
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use TE-implemented activation functions. Supports `gelu`, `silu`, `relu`.

---

## Activation Recomputation

### recompute_granularity
- **Type**: `Optional[Literal['full', 'selective']]`
- **Default**: `None`
- **Description**: Activation recompute mode. `"full"` checkpoints the entire layer. `"selective"` checkpoints specified submodules. `None` = no recompute.

### recompute_method
- **Type**: `Optional[Literal['uniform', 'block']]`
- **Default**: `None`
- **Description**: Layer selection for recompute. `"uniform"` divides layers into equal chunks. `"block"` recomputes a contiguous set of layers per pipeline stage.

### recompute_num_layers
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of layers per recompute unit (`uniform`) or per pipeline stage (`block`). Must be `None` for `selective` granularity.

### distribute_saved_activations
- **Type**: `Optional[bool]`
- **Default**: `False`
- **Description**: Distribute recomputed activations across model parallel group. Incompatible with sequence parallel.

### recompute_modules
- **Type**: `Optional[List[str]]`
- **Default**: `None` (auto-set to `["core_attn"]`)
- **Description**: Submodules to recompute under `selective` granularity. Choices: `"core_attn"`, `"moe_act"`, `"layernorm"`, `"mla_up_proj"`, `"mlp"`, `"moe"`, `"shared_experts"`.

---

## Context Parallel

### cp_comm_type
- **Type**: `Optional[Union[str, List[str]]]`
- **Default**: `None`
- **Description**: Communication type for context parallelism. Per-layer string or list. Options: `"p2p"`, `"all_gather"`, `"a2a"`, `"a2a+p2p"`.

---

## CUDA Graphs

### enable_cuda_graph
- **Type**: `bool`
- **Default**: `False`
- **Description**: DEPRECATED. Use `cuda_graph_impl` instead.

### cuda_graph_impl
- **Type**: `Literal['none', 'local', 'transformer_engine']`
- **Default**: `"none"`
- **Description**: CUDA graph capture implementation. `"local"` uses MCore's implementation. `"transformer_engine"` uses TE's `make_graphed_callables()`.

### cuda_graph_scope
- **Type**: `Union[str, CudaGraphScope, List[str], List[CudaGraphScope]]`
- **Default**: `"full"`
- **Description**: CUDA graph capture scope. Options: `"attn"`, `"mlp"`, `"moe"`, `"moe_router"`, `"moe_preprocess"`, `"mamba"`, `"full_iteration"` (local impl only). Empty list = full layer.

### cuda_graph_use_single_mempool
- **Type**: `bool`
- **Default**: `False`
- **Description**: [Local impl only] Use a single memory pool for all CUDA graphs.

### cuda_graph_retain_backward_graph
- **Type**: `bool`
- **Default**: `False`
- **Description**: Capture backward passes with `retain_grad=True`.

### cuda_graph_warmup_steps
- **Type**: `int`
- **Default**: `3`
- **Description**: Number of warmup steps for CUDA graph capture.

### external_cuda_graph
- **Type**: `bool`
- **Default**: `False`
- **Description**: DEPRECATED. Use `cuda_graph_impl='transformer_engine'`.

---

## Inference

### use_inference_optimized_layers
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use inference-optimized transformer layers during inference.

### flash_decode
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use optimized flash decoding kernel during inference decode.

### inference_fuse_tp_communication
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use fused reduce-scatter-residual-norm-allgather kernel during inference.

### inference_disable_triton_nvls_kernels
- **Type**: `bool`
- **Default**: `False`
- **Description**: Disable Triton NVLS kernels during inference.

### inference_grouped_gemm_backend
- **Type**: `Literal['flashinfer', 'torch', 'vllm']`
- **Default**: `"vllm"`
- **Description**: Grouped GEMM backend for inference MoE. `"flashinfer"` = FlashInfer cutlass_fused_moe, `"torch"` = mcore_fused_moe (Triton), `"vllm"` = vLLM Triton fused MoE.

### inference_moe_disable_fused_quant_kernels
- **Type**: `bool`
- **Default**: `False`
- **Description**: Disable fused permute/activation+MXFP8 quantization kernels during inference.

### inference_moe_token_dispatcher_type
- **Type**: `Literal['nccl', 'nvls']`
- **Default**: `'nvls'`
- **Description**: Token dispatcher for MoE inference EP. `"nvls"` requires Hopper+ with NVLink and symmetric memory.

### batch_invariant_mode
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use batch-invariant kernels for deterministic execution regardless of batch size. Only supports FlashAttention.

### use_te_rng_tracker
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use TE's RNG tracker instead of MCore's.

### inference_rng_tracker
- **Type**: `bool`
- **Default**: `False`
- **Description**: Instantiate a separate RNG tracker for inference.

### inference_sampling_seed
- **Type**: `int`
- **Default**: `42`
- **Description**: Random seed for sampling during inference.

---

## Miscellaneous

### calculate_per_token_loss
- **Type**: `bool`
- **Default**: `False`
- **Description**: Calculate cross-entropy loss over actual non-padded tokens rather than assuming all tokens are valid.

### clone_scatter_output_in_embedding
- **Type**: `bool`
- **Default**: `True`
- **Description**: Clone scatter output in the embedding layer to facilitate garbage collection.

### disable_parameter_transpose_cache
- **Type**: `bool`
- **Default**: `False`
- **Description**: Disable parameter transpose caching.

### config_logger_dir
- **Type**: `str`
- **Default**: `""`
- **Description**: Directory to dump entry-point configs.

### test_mode
- **Type**: `bool`
- **Default**: `False`
- **Description**: Run real-time consistency tests across devices.

### symmetric_ar_type
- **Type**: `Optional[Literal['two_shot', 'one_shot', 'multimem_all_reduce']]`
- **Default**: `None`
- **Description**: Symmetric all-reduce type using symmetric memory.

### nccl_all_reduce_for_prefill
- **Type**: `bool`
- **Default**: `False`
- **Description**: Use NCCL all-reduce during prefill when symmetric all-reduce is enabled.

### quant_recipe
- **Type**: `Optional[RecipeConfig]`
- **Default**: `None`
- **Description**: Per-module quantization configuration.

### mlp_chunks_for_prefill
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of sequence-dimension chunks for MLP during prefill inference.

### mlp_chunks_for_training
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of sequence-dimension chunks for MLP during training.

### mamba_state_dim
- **Type**: `int`
- **Default**: `128`
- **Description**: State dimensionality for Mamba layers.

### mamba_head_dim
- **Type**: `int`
- **Default**: `64`
- **Description**: Head dimensionality for Mamba layers.

### mamba_num_groups
- **Type**: `int`
- **Default**: `8`
- **Description**: Number of groups for Mamba layers.

### mamba_num_heads
- **Type**: `Optional[int]`
- **Default**: `None`
- **Description**: Number of heads for Mamba layers.

### use_mamba_mem_eff_path
- **Type**: `bool`
- **Default**: `True`
- **Description**: Use memory-efficient path for Mamba layers.

---

## Fine-Grained Activation Offloading

### fine_grained_activation_offloading
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable module-level activation offloading to CPU. Incompatible with `cpu_offloading`.

### offload_modules
- **Type**: `Optional[list[str]]`
- **Default**: `[]`
- **Description**: Submodules to offload. Choices: `"attn_norm"`, `"qkv_linear"`, `"core_attn"`, `"attn_proj"`, `"mlp_norm"`, `"expert_fc1"`, `"moe_act"`.

### min_offloaded_tensor_size
- **Type**: `int`
- **Default**: `1024 * 1024`
- **Description**: Minimum tensor size for offloading.

---

## MLATransformerConfig

`MLATransformerConfig` extends `TransformerConfig` with Multi-Latent Attention parameters (DeepSeek-style). It forces `multi_latent_attention=True` and `normalization='RMSNorm'`.

```python
from megatron.core.transformer import MLATransformerConfig

config = MLATransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_head_dim=128,
    v_head_dim=128,
)
```

### multi_latent_attention
- **Type**: `bool`
- **Default**: `True`
- **Description**: Always True for MLA. Enables multi-latent attention architecture.

### q_lora_rank
- **Type**: `int`
- **Default**: `512`
- **Description**: Rank of the query tensor's low-rank representation.

### kv_lora_rank
- **Type**: `int`
- **Default**: `512`
- **Description**: Rank of the key/value tensors' low-rank representation.

### qk_head_dim
- **Type**: `int`
- **Default**: `128`
- **Description**: Dimension of the head in the QK projection. Total `q_head_dim = qk_head_dim + qk_pos_emb_head_dim`.

### qk_pos_emb_head_dim
- **Type**: `int`
- **Default**: `64`
- **Description**: Dimension of the position embedding component in the QK projection.

### v_head_dim
- **Type**: `int`
- **Default**: `128`
- **Description**: Dimension of the head in the V projection.

### rope_type
- **Type**: `str`
- **Default**: `"yarn"`
- **Description**: RoPE type. Options: `"rope"` (standard), `"yarn"` (YaRN).

### rotary_base
- **Type**: `float`
- **Default**: `10000`
- **Description**: Base frequency for rotary embeddings.

### rotary_percent
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Fraction of dimensions to apply rotary embeddings (for `"rope"` type).

### rotary_scaling_factor
- **Type**: `float`
- **Default**: `40`
- **Description**: Scaling factor for YaRN rotary embeddings.

### original_max_position_embeddings
- **Type**: `int`
- **Default**: `4096`
- **Description**: Original maximum position embeddings (used by YaRN).

### beta_fast
- **Type**: `float`
- **Default**: `32`
- **Description**: Fast beta for YaRN RoPE.

### beta_slow
- **Type**: `float`
- **Default**: `1`
- **Description**: Slow beta for YaRN RoPE.

### mscale
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Mscale for YaRN RoPE.

### mscale_all_dim
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Mscale for all dimensions in YaRN RoPE.

### cache_mla_latents
- **Type**: `bool`
- **Default**: `False`
- **Description**: Cache low-dimensional MLA latents instead of full KV cache. Requires Flash MLA. Only for dynamic inference backend.

### mla_down_proj_fusion
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable fused Q/KV down-projection and fused input layernorm.

---

## Key `__post_init__` Validation Rules

The following validations are enforced during initialization:

1. **Mutually exclusive precisions**: `fp16` and `bf16` cannot both be True.
2. **Attention head divisibility**: `num_attention_heads` must be divisible by `tensor_model_parallel_size`.
3. **Query group divisibility**: `num_query_groups` must be a multiple or divisor of `tensor_model_parallel_size`.
4. **Default auto-computations**: `ffn_hidden_size` defaults to `4 * hidden_size`; `kv_channels` defaults to `hidden_size // num_attention_heads`; `num_query_groups` defaults to `num_attention_heads`.
5. **FP8 param requires FP8**: `fp8_param` requires `fp8` to be set.
6. **FP4 param requires FP4**: `fp4_param` requires `fp4` to be set.
7. **FP4 and FP8 are mutually exclusive**: Cannot enable both simultaneously.
8. **Custom FP8/FP4 recipes require quantizer**: `fp8_recipe='custom'` requires `fp8_quantizer_factory`; `fp4_recipe='custom'` requires `fp4_quantizer_factory`.
9. **First/last BF16 layers**: Incompatible with `fp8_recipe='delayed'`. Layer counts must be valid.
10. **Expert parallelism requires experts**: `expert_model_parallel_size > 1` requires `num_moe_experts` to be set.
11. **Inference-optimized MoE constraints**: No expert tensor parallelism, no token dropping, requires `moe_router_dtype='fp32'`, no GLU.
12. **Recompute granularity**: Must be `'full'` or `'selective'` if set. Method must be `'block'` or `'uniform'` for full granularity. `recompute_num_layers` must be `None` for selective.
13. **Distributed activations**: Incompatible with `sequence_parallel`.
14. **MuP initialization**: Warns if custom `init_method` or `output_layer_init_method` is set, as this may break MuP assumptions.
15. **Bias in MoE**: When `num_moe_experts` is set and `add_bias_linear=True`, requires `expert_tensor_parallel_size == 1`.
16. **Top-1 routing**: Requires `moe_router_pre_softmax=True` when using softmax score function.
17. **MLA + interleaved RoPE**: `rotary_interleaved` is incompatible with `multi_latent_attention`.
18. **MLA + rope_fusion**: `apply_rope_fusion` for MLA only works with `rope_type='yarn'`.
19. **Context parallel CP comm type**: If a list, must have length equal to `num_layers`.
20. **CUDA graph scope validation**: Various constraints on scope combinations (e.g., `moe` and `moe_router` are mutually exclusive).
21. **Inference-optimized constraints**: Requires `RMSNorm`, no bias, no QKV bias, no kitchen.
22. **DSA constraints**: Currently incompatible with context parallelism and RoPE fusion.
