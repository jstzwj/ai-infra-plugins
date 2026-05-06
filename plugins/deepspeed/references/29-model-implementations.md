# DeepSpeed Model Implementations Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [DeepSpeed Transformer Layer](#deepspeed-transformer-layer)
4. [DeepSpeed Transformer Config](#deepspeed-transformer-config)
5. [DeepSpeed Transformer Function](#deepspeed-transformer-function)
6. [Training Model Implementations](#training-model-implementations)
7. [Inference V2 Model Implementations](#inference-v2-model-implementations)
8. [Common Parameters Module](#common-parameters-module)
9. [Sharding Module](#sharding-module)
10. [Diffusion Model Support](#diffusion-model-support)
11. [Feature Extraction Support](#feature-extraction-support)
12. [Per-Model Architecture Details](#per-model-architecture-details)
13. [Configuration Examples](#configuration-examples)
14. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed provides optimized implementations for a wide range of model architectures spanning transformer-based language models, diffusion models, and feature extraction models. These implementations are organized across three main subsystems:

1. **Training models** (`deepspeed/model_implementations/` and `deepspeed/ops/transformer/`): Fused transformer layer implementations for training with kernel injection and AutoTP
2. **Inference V2 models** (`deepspeed/inference/v2/model_implementations/`): Next-generation inference with ragged batching, blocked KV cache, and quantization support
3. **Diffusion models**: Support for Stable Diffusion and related architectures through specialized containers

Each model implementation includes:
- Architecture-specific policy for module injection
- Weight mapping and transfer logic
- Configuration extraction from HuggingFace checkpoints
- Tensor parallelism partitioning plans
- Quantization support (where applicable)

---

## Architecture

### Source Code Structure

```
deepspeed/
    model_implementations/           # Training model implementations
        __init__.py
        transformers/                # Transformer model wrappers
            __init__.py
            ds_transformer.py        # Generic DS transformer wrapper
            ds_gpt.py                # GPT-specific wrapper
            ds_llama2.py             # LLaMA 2/3 wrapper
            ds_bert.py               # BERT-specific wrapper
        diffusers/                   # Diffusion model support
            __init__.py
            ds_stable_diffusion.py   # Stable Diffusion wrapper
        features/                    # Feature extraction models
            __init__.py
    
    ops/transformer/                 # Core fused transformer ops
        __init__.py
        deepspeed_transformer.py     # DeepSpeedTransformerLayer class
        ds_transformer_config.py     # DeepSpeedTransformerConfig
        ds_transformer_function.py   # DeepSpeedTransformerFunction (autograd)
        
    inference/v2/model_implementations/  # Inference V2 models
        __init__.py
        inference_model_base.py      # DSInferenceModelBase
        transformer_base.py          # DSTransformerModelBase
        moe_base.py                  # DSMOETransformerModelBase
        common/                      # Shared parameter handling
            __init__.py
            parameters.py            # Common parameter definitions
        sharding/                    # Parameter sharding for TP
            __init__.py
            param_sharder.py         # Parameter sharding logic
        llama_v2/                    # LLaMA 2/3 inference
            __init__.py
            policy.py                # Llama2Policy
            container.py             # Llama2TransformerContainer
        mistral/                     # Mistral inference
            __init__.py
            policy.py
            container.py
        mixtral/                     # Mixtral MoE inference
            __init__.py
            policy.py
            container.py
        falcon/                      # Falcon inference
            __init__.py
            policy.py
            container.py
        phi/                         # Phi inference
            __init__.py
            policy.py
            container.py
        phi3/                        # Phi-3 inference
            __init__.py
            policy.py
            container.py
        opt/                         # OPT inference
            __init__.py
            policy.py
            container.py
        qwen/                        # Qwen inference
            __init__.py
            policy.py
            container.py
        qwen_v2/                     # Qwen2 inference
            __init__.py
            policy.py
            container.py
        qwen_v2_moe/                 # Qwen2-MoE inference
            __init__.py
            policy.py
            container.py
        exaone4/                     # Exaone4 inference
            __init__.py
            policy.py
            container.py
```

---

## DeepSpeed Transformer Layer

The `DeepSpeedTransformerLayer` in `deepspeed/ops/transformer/deepspeed_transformer.py` is the core optimized transformer implementation used for training.

### Class Definition

```python
# deepspeed/ops/transformer/deepspeed_transformer.py

class DeepSpeedTransformerLayer(nn.Module):
    """Fused transformer layer for high-performance training.
    
    Implements a complete transformer encoder/decoder layer with:
    - Pre-LN or Post-LN layer normalization
    - Multi-head self-attention with optional rotary embeddings
    - Feed-forward network with configurable activation
    - Residual connections and dropout
    - Optional stochastic depth
    - Support for FP16, BF16, and FP32
    
    All operations are fused into optimized CUDA kernels to minimize
    kernel launch overhead and memory traffic.
    """
    
    def __init__(self, config: "DeepSpeedTransformerConfig"):
        super().__init__()
        self.config = config
        
        # Attention module
        self.attn = DeepSpeedSelfAttention(config)
        
        # MLP module
        if config.mlp_after_attn:
            self.mlp = DeepSpeedMLP(config)
        
        # Layer normalization
        if config.pre_layer_norm:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
```

### Forward Pass

```python
def forward(self, input, input_mask=None, attention_mask=None, 
            self_attn_layer_past=None, self_attn_grad_mask=None,
            output_attentions=False, layer_head_mask=None):
    """Forward pass through the fused transformer layer.
    
    Args:
        input (Tensor): [batch_size, seq_length, hidden_size]
        input_mask (Tensor): Attention mask [batch_size, 1, 1, seq_length]
        attention_mask (Tensor): Extended attention mask
        self_attn_layer_past (Tuple): Cached K, V for inference
        self_attn_grad_mask: Gradient mask for attention
        output_attentions (bool): Return attention weights
        layer_head_mask (Tensor): Head mask for head pruning
    
    Returns:
        Tuple of (output, present_kv, attention_weights)
    """
    # Use custom autograd function for fused backward
    return DeepSpeedTransformerFunction.apply(
        input, input_mask, attention_mask,
        self_attn_layer_past, self_attn_grad_mask,
        output_attentions, layer_head_mask,
        self.attn, self.mlp, self.norm1, self.norm2,
        self.config,
    )
```

### Component Modules

#### DeepSpeedSelfAttention

```python
class DeepSpeedSelfAttention(nn.Module):
    """Fused multi-head self-attention with rotary embedding support."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Fused QKV weight: [3 * hidden_size, hidden_size] or
        # GQA: [(num_q_heads + 2 * num_kv_heads) * head_size, hidden_size]
        qkv_size = 3 * config.hidden_size
        self.qkvw = nn.Parameter(torch.empty(qkv_size, config.hidden_size))
        
        # Output projection: [hidden_size, hidden_size]
        self.ow = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        
        # Biases (optional)
        if config.add_bias:
            self.qkvb = nn.Parameter(torch.empty(qkv_size))
            self.ob = nn.Parameter(torch.empty(config.hidden_size))
    
    def forward(self, input, mask, layer_past=None, head_mask=None,
                output_attentions=False):
        # QKV projection
        qkv = torch.nn.functional.linear(input, self.qkvw, self.qkvb)
        
        # Split into Q, K, V
        query, key, value = qkv.chunk(3, dim=-1)
        
        # Apply rotary embeddings (if enabled)
        if self.config.rotary_embedding:
            query, key = apply_rotary_pos_emb(query, key, freqs)
        
        # Multi-head attention
        attn_output = scaled_dot_product_attention(
            query, key, value, mask
        )
        
        # Output projection
        output = torch.nn.functional.linear(attn_output, self.ow, self.ob)
        
        return output
```

#### DeepSpeedMLP

```python
class DeepSpeedMLP(nn.Module):
    """Fused feed-forward network module."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        intermediate_size = config.intermediate_size or 4 * config.hidden_size
        
        # First linear: [hidden_size, intermediate_size]
        self.w1 = nn.Parameter(torch.empty(intermediate_size, config.hidden_size))
        
        # Second linear: [hidden_size, intermediate_size] (for SwiGLU)
        self.w2 = nn.Parameter(torch.empty(intermediate_size, config.hidden_size))
        
        # Output linear: [intermediate_size, hidden_size]
        self.w3 = nn.Parameter(torch.empty(config.hidden_size, intermediate_size))
    
    def forward(self, input):
        # Gated MLP (SwiGLU) or standard MLP
        if self.config.activation == "silu":
            # SwiGLU: output = (input @ w1.T * SiLU(input @ w2.T)) @ w3.T
            gate = torch.nn.functional.linear(input, self.w1)
            up = torch.nn.functional.linear(input, self.w2)
            intermediate = gate * torch.nn.functional.silu(up)
        else:
            # Standard: output = activation(input @ w1.T) @ w3.T
            intermediate = torch.nn.functional.linear(input, self.w1)
            intermediate = torch.nn.functional.gelu(intermediate)
        
        output = torch.nn.functional.linear(intermediate, self.w3)
        return output
```

---

## DeepSpeed Transformer Config

The `DeepSpeedTransformerConfig` in `deepspeed/ops/transformer/ds_transformer_config.py` defines all configuration parameters for the fused transformer layer.

### Full Parameter Reference

```python
# deepspeed/ops/transformer/ds_transformer_config.py

@dataclass
class DeepSpeedTransformerConfig:
    """Configuration for DeepSpeed transformer layer.
    
    All parameters have sensible defaults for common architectures.
    """
    
    # === Architecture Parameters ===
    batch_size: int = -1                # Training batch size (-1 for dynamic)
    max_seq_length: int = -1            # Maximum sequence length (-1 for dynamic)
    hidden_size: int = -1               # Hidden dimension size
    heads: int = -1                     # Number of attention heads
    intermediate_size: int = -1         # FFN intermediate size (-1 = 4 * hidden_size)
    num_hidden_layers: int = -1         # Total number of transformer layers
    
    # === Attention Parameters ===
    attn_dropout_ratio: float = 0.0     # Attention dropout probability
    hidden_dropout_ratio: float = 0.0   # Hidden dropout probability
    add_bias: bool = True               # Add bias to linear layers
    
    # === Normalization ===
    pre_layer_norm: bool = True         # Pre-LN (True) vs Post-LN (False)
    layer_norm_eps: float = 1e-5        # Layer norm epsilon
    
    # === Position Embedding ===
    rotary_embedding: bool = False      # Use rotary positional embeddings
    rotary_emb_base: float = 10000.0    # Base frequency for RoPE
    
    # === MLP ===
    mlp_after_attn: bool = True         # Include MLP after attention
    mlp_type: str = "standard"          # MLP type: "standard", "residual"
    activation: str = "gelu"            # Activation: "gelu", "silu", "relu"
    
    # === Precision ===
    fp16: bool = False                  # Enable FP16 training
    bf16: bool = False                  # Enable BF16 training
    
    # === Training ===
    training: bool = True               # Training mode (vs inference)
    stochastic_mode: bool = False       # Stochastic depth
    initializer_range: float = 0.02     # Weight init range
    
    # === Distributed ===
    local_rank: int = -1                # Local GPU rank
    mp_size: int = 1                    # Model parallel (TP) size
    
    # === Special ===
    return_tuple: bool = True           # Return tuple instead of dict
    max_grad_norm: float = 1.0          # Max gradient norm
```

### Configuration Table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `-1` | Training batch size (-1 = dynamic) |
| `max_seq_length` | int | `-1` | Maximum sequence length (-1 = dynamic) |
| `hidden_size` | int | `-1` | Hidden dimension (e.g., 4096 for LLaMA-7B) |
| `heads` | int | `-1` | Number of attention heads |
| `intermediate_size` | int | `-1` | FFN intermediate size (-1 = 4x hidden_size) |
| `num_hidden_layers` | int | `-1` | Total number of layers |
| `attn_dropout_ratio` | float | `0.0` | Attention dropout |
| `hidden_dropout_ratio` | float | `0.0` | Hidden state dropout |
| `add_bias` | bool | `True` | Include bias in linear layers |
| `pre_layer_norm` | bool | `True` | Pre-LN vs Post-LN |
| `layer_norm_eps` | float | `1e-5` | Layer norm epsilon |
| `rotary_embedding` | bool | `False` | Enable rotary position embeddings |
| `rotary_emb_base` | float | `10000.0` | RoPE base frequency |
| `mlp_after_attn` | bool | `True` | Include MLP after attention |
| `mlp_type` | str | `"standard"` | MLP architecture type |
| `activation` | str | `"gelu"` | Activation function |
| `fp16` | bool | `False` | FP16 training |
| `bf16` | bool | `False` | BF16 training |
| `training` | bool | `True` | Training vs inference mode |
| `stochastic_mode` | bool | `False` | Stochastic depth |
| `initializer_range` | float | `0.02` | Weight initialization range |
| `local_rank` | int | `-1` | GPU rank for distributed |
| `mp_size` | int | `1` | Model parallel size |

### Per-Model Configuration Presets

#### LLaMA 7B

```python
DeepSpeedTransformerConfig(
    hidden_size=4096,
    heads=32,
    intermediate_size=11008,
    num_hidden_layers=32,
    pre_layer_norm=True,
    rotary_embedding=True,
    rotary_emb_base=10000.0,
    activation="silu",
    add_bias=False,
    mlp_after_attn=True,
)
```

#### LLaMA 70B (with GQA)

```python
DeepSpeedTransformerConfig(
    hidden_size=8192,
    heads=64,
    intermediate_size=28672,
    num_hidden_layers=80,
    pre_layer_norm=True,
    rotary_embedding=True,
    rotary_emb_base=10000.0,
    activation="silu",
    add_bias=False,
    mlp_after_attn=True,
    # GQA: 64 query heads, 8 KV heads
    num_kv_heads=8,
)
```

#### OPT 175B

```python
DeepSpeedTransformerConfig(
    hidden_size=12288,
    heads=96,
    intermediate_size=49152,
    num_hidden_layers=96,
    pre_layer_norm=False,
    rotary_embedding=False,
    activation="relu",
    add_bias=True,
    mlp_after_attn=True,
)
```

#### BLOOM 176B

```python
DeepSpeedTransformerConfig(
    hidden_size=14336,
    heads=112,
    intermediate_size=57344,
    num_hidden_layers=70,
    pre_layer_norm=True,
    rotary_embedding=False,  # BLOOM uses ALiBi
    activation="gelu",
    add_bias=True,
    mlp_after_attn=True,
)
```

---

## DeepSpeed Transformer Function

The `DeepSpeedTransformerFunction` in `deepspeed/ops/transformer/ds_transformer_function.py` implements the custom autograd function for the fused transformer layer.

### Class Definition

```python
# deepspeed/ops/transformer/ds_transformer_function.py

class DeepSpeedTransformerFunction(torch.autograd.Function):
    """Custom autograd function for fused transformer layer.
    
    Implements forward and backward passes using custom CUDA kernels.
    The backward pass fuses gradient computations to minimize memory
    traffic and kernel launch overhead.
    """
    
    @staticmethod
    def forward(ctx, input, input_mask, attention_mask,
                layer_past, grad_mask, output_attentions,
                head_mask, attn, mlp, norm1, norm2, config):
        """Forward pass through the fused transformer layer."""
        
        # Pre-layer norm or post-layer norm path
        if config.pre_layer_norm:
            # Pre-LN: normalize before attention
            normed_input = norm1(input)
            attn_output = attn(normed_input, input_mask, layer_past, head_mask)
            hidden = input + attn_output  # Residual
            
            normed_hidden = norm2(hidden)
            mlp_output = mlp(normed_hidden)
            output = hidden + mlp_output  # Residual
        else:
            # Post-LN: normalize after residual
            attn_output = attn(input, input_mask, layer_past, head_mask)
            hidden = norm1(input + attn_output)  # Residual + LN
            
            mlp_output = mlp(hidden)
            output = norm2(hidden + mlp_output)  # Residual + LN
        
        # Save for backward
        ctx.save_for_backward(input, attn_output, mlp_output,
                              input_mask, head_mask)
        ctx.config = config
        ctx.attn = attn
        ctx.mlp = mlp
        ctx.norm1 = norm1
        ctx.norm2 = norm2
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with fused gradient computation."""
        input, attn_output, mlp_output, input_mask, head_mask = ctx.saved_tensors
        config = ctx.config
        
        # Compute gradients using fused CUDA kernels
        # (gradients are fused to minimize memory writes)
        grad_input, grad_attn, grad_mlp = fused_backward(
            grad_output, input, attn_output, mlp_output,
            input_mask, head_mask, config
        )
        
        return grad_input, None, None, None, None, None, None, None, None, None, None, None
```

---

## Training Model Implementations

### DS Transformer Wrapper

```python
# deepspeed/model_implementations/transformers/ds_transformer.py

class DSTransformerModel(nn.Module):
    """Generic DeepSpeed transformer model wrapper.
    
    Wraps a standard transformer model with DeepSpeed optimizations:
    - Fused transformer layers
    - Mixed precision training
    - Gradient checkpointing
    - Tensor parallelism
    """
    
    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        self.config = config
        self._ds_config = None
    
    def inject_fused_layers(self, ds_config=None):
        """Replace standard layers with fused implementations."""
        from deepspeed.module_inject import module_inject
        self.model, _ = module_inject(
            self.model, config=ds_config or self.config
        )
```

### DS GPT Wrapper

```python
# deepspeed/model_implementations/transformers/ds_gpt.py

class DSGPTModel(DSTransformerModel):
    """DeepSpeed wrapper for GPT-style (causal LM) models.
    
    Supports: GPT-2, GPT-Neo, GPT-NeoX, GPT-J, OPT, BLOOM
    
    Features:
    - Causal attention mask generation
    - Past key-value caching for generation
    - Position embedding handling
    """
    
    def __init__(self, model, config=None):
        super().__init__(model, config)
        self._is_causal = True
    
    def generate(self, input_ids, max_length=100, **kwargs):
        """Autoregressive generation with KV caching."""
        # ... generation logic with DeepSpeed optimizations
        pass
```

### DS LLaMA2 Wrapper

```python
# deepspeed/model_implementations/transformers/ds_llama2.py

class DSLLaMA2Model(DSTransformerModel):
    """DeepSpeed wrapper for LLaMA 2/3 models.
    
    Features:
    - Rotary position embeddings (RoPE)
    - Grouped Query Attention (GQA) for 70B
    - SwiGLU activation in MLP
    - RMSNorm instead of LayerNorm
    - No bias in any linear layer
    """
    
    def __init__(self, model, config=None):
        super().__init__(model, config)
        self._uses_rope = True
        self._uses_gqa = getattr(model.config, 'num_key_value_heads', 
                                  model.config.num_attention_heads) != model.config.num_attention_heads
```

### DS BERT Wrapper

```python
# deepspeed/model_implementations/transformers/ds_bert.py

class DSBERTModel(DSTransformerModel):
    """DeepSpeed wrapper for BERT-style (masked LM) models.
    
    Supports: BERT, RoBERTa, DistilBERT, DeBERTa
    
    Features:
    - Bidirectional attention
    - [CLS] token handling
    - Segment embeddings
    - Masked language modeling head
    """
    
    def __init__(self, model, config=None):
        super().__init__(model, config)
        self._is_causal = False
```

---

## Inference V2 Model Implementations

The inference V2 engine has its own model implementation framework in `deepspeed/inference/v2/model_implementations/`. Each model consists of a **policy** (defining how to parse and replace the original model) and a **container** (the optimized inference implementation).

### Base Classes

#### DSInferenceModelBase

```python
# deepspeed/inference/v2/model_implementations/inference_model_base.py

class DSInferenceModelBase:
    """Base class for all inference V2 model implementations.
    
    Defines the interface for:
    - Model parsing and policy application
    - Parameter sharding for tensor parallelism
    - Weight loading and conversion
    - Inference execution
    """
    
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._policy = None
        self._containers = []
    
    def parse(self):
        """Parse the model and create inference containers.
        
        Walks the model tree, identifies replaceable modules,
        and creates optimized containers for each.
        """
        raise NotImplementedError
    
    def allocate_kv_cache(self, state_manager):
        """Allocate KV cache for inference.
        
        Args:
            state_manager (DSStateManager): The inference state manager
        """
        for container in self._containers:
            container.allocate_kv_cache(state_manager)
    
    def forward(self, batch):
        """Execute inference forward pass.
        
        Args:
            batch (RaggedBatch): The ragged inference batch
        
        Returns:
            Tensor: Output logits
        """
        raise NotImplementedError
```

#### DSTransformerModelBase

```python
# deepspeed/inference/v2/model_implementations/transformer_base.py

class DSTransformerModelBase(DSInferenceModelBase):
    """Base class for transformer-based inference models.
    
    Provides common functionality for all transformer models:
    - Layer-by-layer execution
    - KV cache management
    - Tensor parallel communication
    """
    
    def __init__(self, model, config):
        super().__init__(model, config)
        self._embedding = None
        self._layers = []
        self._final_norm = None
        self._lm_head = None
    
    def forward(self, batch):
        """Execute transformer forward pass layer by layer.
        
        Flow:
        1. Embedding lookup
        2. For each transformer layer:
           a. Layer norm
           b. Self-attention (with KV cache update)
           c. Residual connection
           d. Feed-forward network
           e. Residual connection
        3. Final layer norm
        4. LM head projection
        """
        # Embedding
        hidden = self._embedding(batch.token_ids)
        
        # Transformer layers
        for layer in self._layers:
            hidden = layer(hidden, batch)
        
        # Final norm + LM head
        hidden = self._final_norm(hidden)
        logits = self._lm_head(hidden)
        
        return logits
```

#### DSMOETransformerModelBase

```python
# deepspeed/inference/v2/model_implementations/moe_base.py

class DSMOETransformerModelBase(DSTransformerModelBase):
    """Base class for MoE (Mixture of Experts) inference models.
    
    Extends DSTransformerModelBase with:
    - Expert routing
    - Expert sharding across GPUs
    - Load balancing
    """
    
    def __init__(self, model, config):
        super().__init__(model, config)
        self._num_experts = 0
        self._num_experts_per_tok = 0
        self._expert_layers = []
    
    def forward(self, batch):
        """Execute MoE transformer forward pass.
        
        For MoE layers:
        1. Router computes expert assignments
        2. Tokens dispatched to assigned experts
        3. Expert computation (potentially sharded)
        4. Token collection and combination
        """
        hidden = self._embedding(batch.token_ids)
        
        for layer in self._layers:
            if layer.is_moe:
                hidden = self._moe_forward(layer, hidden, batch)
            else:
                hidden = layer(hidden, batch)
        
        hidden = self._final_norm(hidden)
        logits = self._lm_head(hidden)
        return logits
    
    def _moe_forward(self, layer, hidden, batch):
        """Execute MoE layer with expert routing.
        
        1. Compute router logits: router_logits = hidden @ router_weight
        2. Select top-k experts per token
        3. For each expert:
           a. Gather tokens assigned to this expert
           b. Compute expert output
           c. Scatter results back
        4. Combine expert outputs with router weights
        """
        # Router
        router_logits = layer.router(hidden)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self._num_experts_per_tok)
        
        # Expert computation
        output = torch.zeros_like(hidden)
        for k in range(self._num_experts_per_tok):
            expert_idx = top_k_indices[:, k]
            expert_weight = top_k_probs[:, k]
            
            # Route to expert
            expert_output = layer.experts[expert_idx](hidden)
            output += expert_weight.unsqueeze(-1) * expert_output
        
        return output
```

### Per-Model Inference V2 Implementations

#### LLaMA V2

```python
# deepspeed/inference/v2/model_implementations/llama_v2/

class Llama2Policy:
    """Policy for LLaMA 2/3 model inference.
    
    Architecture:
    - RMSNorm (not LayerNorm)
    - Rotary position embeddings (RoPE)
    - SwiGLU MLP (gate + up + down projections)
    - No bias in any linear layer
    - Optional Grouped Query Attention (GQA)
    """
    
    def get_attention_params(self):
        return {
            "q_proj": {"weight": "column"},
            "k_proj": {"weight": "column"},
            "v_proj": {"weight": "column"},
            "o_proj": {"weight": "row"},
        }
    
    def get_mlp_params(self):
        return {
            "gate_proj": {"weight": "column"},
            "up_proj": {"weight": "column"},
            "down_proj": {"weight": "row"},
        }

class Llama2TransformerContainer:
    """Container for optimized LLaMA 2/3 inference."""
    
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        self.attention = None
        self.mlp = None
        self.norm1 = None  # RMSNorm
        self.norm2 = None  # RMSNorm
    
    def forward(self, hidden, batch):
        # Pre-norm attention
        normed = self.norm1(hidden)
        attn_out = self.attention(normed, batch)
        hidden = hidden + attn_out
        
        # Pre-norm MLP
        normed = self.norm2(hidden)
        mlp_out = self.mlp(normed)
        hidden = hidden + mlp_out
        
        return hidden
```

#### Mistral

```python
# deepspeed/inference/v2/model_implementations/mistral/

class MistralPolicy(Llama2Policy):
    """Policy for Mistral model inference.
    
    Similar to LLaMA with differences:
    - Sliding window attention (window_size=4096)
    - Default GQA with 8 KV heads (vs 32 query heads)
    - Different RoPE base frequency
    """
    
    def get_attention_params(self):
        params = super().get_attention_params()
        params["sliding_window"] = 4096
        return params
```

#### Mixtral

```python
# deepspeed/inference/v2/model_implementations/mixtral/

class MixtralPolicy(MistralPolicy):
    """Policy for Mixtral 8x7B MoE inference.
    
    Extends Mistral with:
    - 8 experts per MoE layer
    - Top-2 routing per token
    - Expert parallelism across GPUs
    """
    
    def get_moe_params(self):
        return {
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "expert_params": {
                "w1": {"weight": "column"},  # gate_proj
                "w2": {"weight": "row"},     # down_proj
                "w3": {"weight": "column"},  # up_proj
            },
            "router": {"weight": "replicate"},
        }
```

#### Falcon

```python
# deepspeed/inference/v2/model_implementations/falcon/

class FalconPolicy:
    """Policy for Falcon model inference.
    
    Architecture:
    - LayerNorm (not RMSNorm)
    - Parallel attention + MLP (not sequential)
    - Multi-query attention (single KV head)
    - GELU activation
    """
    
    def get_attention_params(self):
        return {
            "query_key_value": {"weight": "column"},  # Fused QKV
            "dense": {"weight": "row"},
        }
    
    def get_mlp_params(self):
        return {
            "dense_h_to_4h": {"weight": "column"},
            "dense_4h_to_h": {"weight": "row"},
        }
```

#### Phi / Phi-3

```python
# deepspeed/inference/v2/model_implementations/phi/

class PhiPolicy:
    """Policy for Phi-2 model inference.
    
    Architecture:
    - LayerNorm
    - Multi-head attention (standard)
    - GELU activation
    - Parallel MLP
    """

class Phi3Policy:
    """Policy for Phi-3 model inference.
    
    Architecture:
    - RMSNorm
    - GQA (Grouped Query Attention)
    - SwiGLU MLP
    - RoPE
    Similar to LLaMA architecture.
    """
```

#### OPT

```python
# deepspeed/inference/v2/model_implementations/opt/

class OPTPolicy:
    """Policy for OPT model inference.
    
    Architecture:
    - LayerNorm (post-norm)
    - Multi-head attention
    - ReLU activation
    - Bias in all linear layers
    - No rotary embeddings
    """
    
    def get_attention_params(self):
        return {
            "q_proj": {"weight": "column", "bias": "column"},
            "k_proj": {"weight": "column", "bias": "column"},
            "v_proj": {"weight": "column", "bias": "column"},
            "out_proj": {"weight": "row", "bias": "row"},
        }
    
    def get_mlp_params(self):
        return {
            "fc1": {"weight": "column", "bias": "column"},
            "fc2": {"weight": "row", "bias": "row"},
        }
```

#### Qwen / Qwen2 / Qwen2-MoE

```python
# deepspeed/inference/v2/model_implementations/qwen/

class QwenPolicy:
    """Policy for Qwen model inference.
    
    Similar to LLaMA architecture.
    """

# deepspeed/inference/v2/model_implementations/qwen_v2/

class QwenV2Policy:
    """Policy for Qwen2 model inference.
    
    Architecture:
    - RMSNorm
    - GQA
    - SwiGLU MLP
    - RoPE with dynamic frequency scaling
    """

# deepspeed/inference/v2/model_implementations/qwen_v2_moe/

class QwenV2MoEPolicy(QwenV2Policy):
    """Policy for Qwen2-MoE inference.
    
    Extends Qwen2 with Mixture of Experts.
    """
    
    def get_moe_params(self):
        return {
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "shared_expert": True,  # Qwen2-MoE has a shared expert
        }
```

#### Exaone4

```python
# deepspeed/inference/v2/model_implementations/exaone4/

class Exaone4Policy:
    """Policy for Exaone4 model inference.
    
    Architecture (LG AI Research):
    - RMSNorm
    - GQA
    - SwiGLU MLP
    - RoPE
    """
```

---

## Common Parameters Module

The common parameters module (`deepspeed/inference/v2/model_implementations/common/parameters.py`) defines shared parameter structures used across all model implementations.

### Parameter Definitions

```python
# deepspeed/inference/v2/model_implementations/common/parameters.py

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class ParameterType(Enum):
    """Parameter types for weight classification."""
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_OUTPUT = "attention_output"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    EMBEDDING = "embedding"
    LAYER_NORM = "layer_norm"
    LM_HEAD = "lm_head"
    BIAS = "bias"
    ROUTER = "router"
    EXPERT_WEIGHT = "expert_weight"

class PartitionStrategy(Enum):
    """Tensor parallelism partition strategies."""
    COLUMN = "column"       # Partition output dimension
    ROW = "row"             # Partition input dimension
    REPLICATE = "replicate" # No partitioning (full copy)
    VOCAB = "vocab"         # Partition vocabulary dimension

@dataclass
class ParameterSpec:
    """Specification for a model parameter."""
    name: str
    shape: Tuple[int, int]
    dtype: torch.dtype
    param_type: ParameterType
    partition_strategy: PartitionStrategy = PartitionStrategy.REPLICATE
    partition_dim: int = -1
    requires_grad: bool = False
    
    @property
    def local_shape(self):
        """Shape after TP partitioning."""
        if self.partition_strategy == PartitionStrategy.COLUMN:
            shape = list(self.shape)
            shape[0] //= self.tp_size
            return tuple(shape)
        elif self.partition_strategy == PartitionStrategy.ROW:
            shape = list(self.shape)
            shape[1] //= self.tp_size
            return tuple(shape)
        return self.shape

@dataclass
class TransformerLayerParameters:
    """Parameters for a single transformer layer."""
    # Attention
    q_weight: ParameterSpec
    k_weight: ParameterSpec
    v_weight: ParameterSpec
    o_weight: ParameterSpec
    q_bias: Optional[ParameterSpec] = None
    k_bias: Optional[ParameterSpec] = None
    v_bias: Optional[ParameterSpec] = None
    o_bias: Optional[ParameterSpec] = None
    
    # MLP
    gate_weight: Optional[ParameterSpec] = None
    up_weight: Optional[ParameterSpec] = None
    down_weight: Optional[ParameterSpec] = = None
    fc1_weight: Optional[ParameterSpec] = None
    fc2_weight: Optional[ParameterSpec] = None
    
    # Norms
    attn_norm_weight: ParameterSpec
    attn_norm_bias: Optional[ParameterSpec] = None
    mlp_norm_weight: ParameterSpec
    mlp_norm_bias: Optional[ParameterSpec] = None
    
    # MoE (optional)
    router_weight: Optional[ParameterSpec] = None
    expert_weights: Optional[list] = None
```

---

## Sharding Module

The sharding module (`deepspeed/inference/v2/model_implementations/sharding/`) handles parameter distribution across tensor-parallel GPUs.

### Parameter Sharding

```python
# deepspeed/inference/v2/model_implementations/sharding/param_sharder.py

class ParameterSharder:
    """Distributes model parameters across tensor-parallel GPUs.
    
    Handles:
    - Column-parallel partitioning (split output dimension)
    - Row-parallel partitioning (split input dimension)
    - Vocabulary partitioning (split embedding table)
    - Replicated parameters (no partitioning)
    """
    
    def __init__(self, tp_size: int, tp_rank: int):
        """
        Args:
            tp_size (int): Tensor parallel degree
            tp_rank (int): This GPU's rank in the TP group
        """
        self.tp_size = tp_size
        self.tp_rank = tp_rank
    
    def shard_parameter(self, param: torch.Tensor, 
                        strategy: PartitionStrategy,
                        dim: int = 0) -> torch.Tensor:
        """Shard a parameter tensor according to the partition strategy.
        
        Args:
            param: Full parameter tensor
            strategy: Partition strategy
            dim: Dimension to partition
        
        Returns:
            torch.Tensor: This rank's shard of the parameter
        """
        if strategy == PartitionStrategy.REPLICATE:
            return param.clone()
        
        chunk_size = param.shape[dim] // self.tp_size
        start = self.tp_rank * chunk_size
        end = start + chunk_size
        
        if dim == 0:
            return param[start:end].clone()
        elif dim == 1:
            return param[:, start:end].clone()
        else:
            indices = torch.arange(start, end)
            return torch.index_select(param, dim, indices).clone()
    
    def shard_qkv(self, q_weight, k_weight, v_weight,
                  num_q_heads, num_kv_heads, head_size):
        """Shard QKV weights for GQA models.
        
        For GQA, Q has more heads than K and V. The sharding must
        account for different head counts.
        
        Args:
            q_weight: [num_q_heads * head_size, hidden_size]
            k_weight: [num_kv_heads * head_size, hidden_size]
            v_weight: [num_kv_heads * head_size, hidden_size]
            num_q_heads: Number of query heads
            num_kv_heads: Number of KV heads
            head_size: Size per head
        
        Returns:
            Sharded (q, k, v) weights for this rank
        """
        # Shard Q: each rank gets num_q_heads / tp_size heads
        q_heads_per_rank = num_q_heads // self.tp_size
        q_start = self.tp_rank * q_heads_per_rank * head_size
        q_end = q_start + q_heads_per_rank * head_size
        q_shard = q_weight[q_start:q_end]
        
        # Shard K, V: each rank gets num_kv_heads / tp_size heads
        kv_heads_per_rank = num_kv_heads // self.tp_size
        k_start = self.tp_rank * kv_heads_per_rank * head_size
        k_end = k_start + kv_heads_per_rank * head_size
        k_shard = k_weight[k_start:k_end]
        v_shard = v_weight[k_start:k_end]
        
        return q_shard, k_shard, v_shard
    
    def gather_output(self, tensor, strategy):
        """Gather sharded outputs from all TP ranks.
        
        Args:
            tensor: Local shard output
            strategy: Partition strategy used for sharding
        
        Returns:
            torch.Tensor: Full gathered tensor
        """
        if strategy == PartitionStrategy.COLUMN:
            # All-gather along first dimension
            gathered = [torch.empty_like(tensor) for _ in range(self.tp_size)]
            torch.distributed.all_gather(gathered, tensor, group=self.tp_group)
            return torch.cat(gathered, dim=0)
        elif strategy == PartitionStrategy.ROW:
            # All-reduce (sum)
            torch.distributed.all_reduce(tensor, group=self.tp_group)
            return tensor
        return tensor
    
    def reduce_scatter(self, tensor, strategy):
        """Reduce-scatter for row-parallel forward.
        
        Each rank sends its partial result and receives its shard
        of the reduced (summed) result.
        """
        if strategy == PartitionStrategy.ROW:
            output = torch.empty_like(tensor)
            torch.distributed.reduce_scatter(
                output, [tensor], group=self.tp_group
            )
            return output
        return tensor
```

### Sharding Communication Patterns

```
Column-Parallel Forward:
  Rank 0: X @ W_0 -> Y_0
  Rank 1: X @ W_1 -> Y_1
  All-Gather: Y = [Y_0, Y_1, Y_2, Y_3]

Row-Parallel Forward:
  Rank 0: X_0 @ W_0 -> partial_0
  Rank 1: X_1 @ W_1 -> partial_1
  All-Reduce: Y = partial_0 + partial_1 + partial_2 + partial_3

Combined (Transformer Layer):
  Input -> Column(QKV) -> Attention -> Row(O_proj) -> All-Reduce ->
  Column(gate+up) -> Activation -> Row(down) -> All-Reduce -> Output
```

---

## Diffusion Model Support

DeepSpeed provides limited support for diffusion models, primarily targeting Stable Diffusion through the `DSStableDiffusionContainer`.

### Stable Diffusion Container

```python
# deepspeed/module_inject/containers/stable_diffusion.py

class DSStableDiffusionContainer(DSModelContainer):
    """Container for Stable Diffusion models.
    
    Supports:
    - UNet optimization (fused attention and convolution)
    - VAE optimization
    - Text encoder (CLIP) optimization
    - Mixed precision training/inference
    
    Limitations:
    - No tensor parallelism for diffusion models
    - Limited kernel injection support
    - Primarily for inference optimization
    """
    
    def __init__(self, model, config=None):
        super().__init__(model, config)
        self._unet = None
        self._vae = None
        self._text_encoder = None
    
    def parse_model(self):
        """Parse Stable Diffusion components."""
        if hasattr(self.model, 'unet'):
            self._unet = self.model.unet
        if hasattr(self.model, 'vae'):
            self._vae = self.model.vae
        if hasattr(self.model, 'text_encoder'):
            self._text_encoder = self.model.text_encoder
```

### Diffusion Model Configuration

```json
{
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}
```

---

## Feature Extraction Support

The feature extraction module supports using transformer models for downstream tasks.

### Feature Extraction API

```python
# deepspeed/model_implementations/features/

class DSFeatureExtractor:
    """DeepSpeed feature extraction wrapper.
    
    Enables efficient extraction of intermediate representations
    from transformer models for downstream tasks.
    """
    
    def __init__(self, model, layers=None, pooler=True):
        """
        Args:
            model: The transformer model
            layers: Layer indices to extract features from
            pooler: Include the pooler output
        """
        self.model = model
        self.target_layers = layers
        self.include_pooler = pooler
        self._hooks = []
    
    def extract(self, input_ids, attention_mask=None):
        """Extract features from specified layers.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            dict: Layer index -> feature tensor
        """
        features = {}
        
        def hook_fn(layer_idx):
            def hook(module, input, output):
                features[layer_idx] = output.detach()
            return hook
        
        # Register hooks
        for idx in self.target_layers:
            layer = self._get_layer(idx)
            self._hooks.append(
                layer.register_forward_hook(hook_fn(idx))
            )
        
        # Forward pass
        with torch.no_grad():
            self.model(input_ids, attention_mask=attention_mask)
        
        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
        return features
```

---

## Per-Model Architecture Details

### Architecture Comparison Table

| Model | Norm | Attention | MLP | Activation | Position | Bias | KV Heads |
|-------|------|-----------|-----|------------|----------|------|----------|
| **LLaMA 2/3** | RMSNorm | MHA/GQA | SwiGLU | SiLU | RoPE | No | 32/8 |
| **Mistral** | RMSNorm | GQA | SwiGLU | SiLU | RoPE | No | 8 |
| **Mixtral** | RMSNorm | GQA | SwiGLU + MoE | SiLU | RoPE | No | 8 |
| **Falcon** | LayerNorm | MQA | Parallel | GELU | Rotary | Yes | 1 |
| **Phi-2** | LayerNorm | MHA | Standard | GELU | RoPE | Yes | 32 |
| **Phi-3** | RMSNorm | GQA | SwiGLU | SiLU | RoPE | No | 8 |
| **OPT** | LayerNorm | MHA | Standard | ReLU | Learned | Yes | 96 |
| **BLOOM** | LayerNorm | MHA | Standard | GELU | ALiBi | Yes | 112 |
| **GPT-NeoX** | LayerNorm | MHA | Standard | GELU | RoPE | No | 32 |
| **GPT-J** | LayerNorm | MHA | Parallel | GELU | RoPE | No | 16 |
| **Qwen2** | RMSNorm | GQA | SwiGLU | SiLU | RoPE | No | 4-8 |
| **Qwen2-MoE** | RMSNorm | GQA | SwiGLU + MoE | SiLU | RoPE | No | 4-8 |
| **Exaone4** | RMSNorm | GQA | SwiGLU | SiLU | RoPE | No | Varies |

### Model Size Reference

| Model | Parameters | Hidden | Heads | Layers | Intermediate | KV Heads |
|-------|-----------|--------|-------|--------|-------------|----------|
| LLaMA-7B | 6.7B | 4096 | 32 | 32 | 11008 | 32 |
| LLaMA-13B | 13.0B | 5120 | 40 | 40 | 13824 | 40 |
| LLaMA-70B | 68.9B | 8192 | 64 | 80 | 28672 | 8 |
| Mistral-7B | 7.2B | 4096 | 32 | 32 | 14336 | 8 |
| Mixtral-8x7B | 46.7B | 4096 | 32 | 32 | 14336 | 8 |
| OPT-125M | 125M | 768 | 12 | 12 | 3072 | 12 |
| OPT-1.3B | 1.3B | 2048 | 32 | 24 | 8192 | 32 |
| OPT-175B | 175B | 12288 | 96 | 96 | 49152 | 96 |
| BLOOM-176B | 176B | 14336 | 112 | 70 | 57344 | 112 |
| GPT-NeoX-20B | 20B | 6144 | 64 | 44 | 24576 | 64 |
| Falcon-180B | 180B | 14848 | 232 | 80 | 59392 | 1 |
| Qwen-72B | 72B | 8192 | 64 | 80 | 24576 | 8 |

### Injection Policy Per Model

#### LLaMA

```
Original Layer:                          DeepSpeed Layer:
+---------------------------+           +---------------------------+
| input_layernorm (RMSNorm) |           | norm1 (RMSNorm)           |
| self_attn.q_proj (Linear) |           | attn.qkvw (fused QKV)     |
| self_attn.k_proj (Linear) |   -->     | attn.ow (output proj)     |
| self_attn.v_proj (Linear) |           | norm2 (RMSNorm)           |
| self_attn.o_proj (Linear) |           | mlp.w1 (gate)             |
| post_attention_layernorm  |           | mlp.w2 (up)               |
| mlp.gate_proj (Linear)    |           | mlp.w3 (down)             |
| mlp.up_proj (Linear)      |           +---------------------------+
| mlp.down_proj (Linear)    |
+---------------------------+
```

Weight mapping:
```python
{
    "self_attn.q_proj.weight": "attn.qkvw[0:head_size*num_heads]",
    "self_attn.k_proj.weight": "attn.qkvw[head_size*num_heads:head_size*(num_heads+kv_heads)]",
    "self_attn.v_proj.weight": "attn.qkvw[head_size*(num_heads+kv_heads):]",
    "self_attn.o_proj.weight": "attn.ow",
    "input_layernorm.weight": "norm1.weight",
    "post_attention_layernorm.weight": "norm2.weight",
    "mlp.gate_proj.weight": "mlp.w1",
    "mlp.up_proj.weight": "mlp.w2",
    "mlp.down_proj.weight": "mlp.w3",
}
```

#### OPT

```
Original Layer:                          DeepSpeed Layer:
+---------------------------+           +---------------------------+
| self_attn.q_proj (Linear) |           | attn.qkvw (fused QKV)     |
| self_attn.k_proj (Linear) |   -->     | attn.ow (output proj)     |
| self_attn.v_proj (Linear) |           | norm1 (LayerNorm)         |
| self_attn.out_proj(Linear)|           | norm2 (LayerNorm)         |
| self_attn_layer_norm      |           | mlp.w1 (fc1)              |
| fc1 (Linear)              |           | mlp.w3 (fc2)              |
| fc2 (Linear)              |           +---------------------------+
| final_layer_norm          |
+---------------------------+
```

---

## Configuration Examples

### LLaMA-7B Training with Injection

```json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 8,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    }
}
```

```python
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

### Mistral-7B Inference V2

```python
import deepspeed

model_config = {
    "tensor_parallel": {"tp_size": 2},
    "dtype": "float16",
    "max_batch_size": 32,
    "max_seq_length": 4096,
}

engine = deepspeed.init_inference(
    model="mistralai/Mistral-7B-v0.2",
    mp_size=2,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)
```

### Mixtral-8x7B MoE Inference

```python
import deepspeed
import torch

# Mixtral with 4-way TP
engine = deepspeed.init_inference(
    model="mistralai/Mixtral-8x7B-v0.1",
    mp_size=4,                             # 4-way tensor parallelism
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)

# Generate
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
with torch.no_grad():
    output = engine.generate(input_ids, max_new_tokens=100)
```

### OPT-175B with ZeRO-3 and AutoTP

```json
{
    "train_batch_size": 512,
    "gradient_accumulation_steps": 16,
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e6,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "tensor_parallel": {
        "enabled": true,
        "tp_size": 8
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 6e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    }
}
```

### Falcon-180B with Kernel Injection

```python
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "tiiie/falcon-180B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

engine = deepspeed.init_inference(
    model=model,
    mp_size=8,
    dtype=torch.bfloat16,
    replace_with_kernel_inject=True,
)
```

---

## Troubleshooting

### Model Type Not Detected

**Symptom**: `ValueError: Unknown model type` or injection not happening.

**Solutions**:
1. Check `model.config.model_type`:
   ```python
   print(model.config.model_type)
   ```
2. Set model type manually if not auto-detected
3. Ensure you are using a supported model from the list above

### Weight Shape Mismatch After Injection

**Symptom**: `RuntimeError: shape '[X, Y]' is invalid for input of size Z`

**Solutions**:
1. Check if GQA heads are correctly configured:
   ```python
   print(model.config.num_attention_heads)
   print(model.config.num_key_value_heads)
   ```
2. Ensure model was loaded with correct `torch_dtype`
3. Verify the intermediate_size matches the model config

### Inference V2 Model Not Supported

**Symptom**: `NotImplementedError` or `KeyError` for model type in V2 engine.

**Solutions**:
1. Check if the model is in the V2 supported list (not all training models have V2 implementations)
2. Fall back to inference V1 engine:
   ```python
   engine = deepspeed.init_inference(model=model, ...)
   ```
3. Check if the model type name matches exactly (e.g., "qwen2" not "qwen")

### MoE Expert Sharding Errors

**Symptom**: Shape or device errors in MoE layers.

**Solutions**:
1. Ensure the number of experts is divisible by the TP degree
2. For Mixtral-8x7B, use `mp_size` of 2, 4, or 8 (divisors of 8)
3. Check expert parallel configuration:
   ```json
   {
       "tensor_parallel": {
           "tp_size": 4,
           "moe_tp_mode": "auto"
       }
   }
   ```

### Quantization Compatibility Issues

**Symptom**: Accuracy loss or NaN outputs after quantization.

**Solutions**:
1. Use INT8 instead of INT4 for better accuracy
2. Increase `q_groups` for finer quantization granularity
3. Disable quantization for sensitive layers (e.g., logits head)
4. Verify the quantized model produces reasonable outputs before deployment:
   ```python
   # Quick validation
   with torch.no_grad():
       output = engine.generate(test_input, max_new_tokens=10)
       print(tokenizer.decode(output[0]))
   ```
