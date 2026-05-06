# DeepSpeed Module Injection and AutoTP Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Injection System](#module-injection-system)
4. [Injection Policies](#injection-policies)
5. [AutoTP: Automatic Tensor Parallelism](#autotp-automatic-tensor-parallelism)
6. [Model Containers](#model-containers)
7. [Module Quantization](#module-quantization)
8. [FusedQKV Utilities](#fusedqkv-utilities)
9. [TP Plan Conversion](#tp-plan-conversion)
10. [Model Parsing and Policy Application](#model-parsing-and-policy-application)
11. [Supported Model List](#supported-model-list)
12. [Configuration Examples](#configuration-examples)
13. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed's module injection system (`deepspeed/module_inject/`) replaces standard PyTorch `nn.Module` layers with highly optimized DeepSpeed implementations at runtime. This mechanism enables:

1. **Transparent optimization**: Replace HuggingFace/standard PyTorch model layers with fused CUDA kernels without modifying model definition code
2. **Automatic Tensor Parallelism (AutoTP)**: Automatically partition model layers across GPUs for tensor-parallel training and inference
3. **Quantization injection**: Replace linear layers with quantized implementations for efficient inference
4. **Policy-driven replacement**: A configurable policy system maps standard layers to optimized replacements

The injection system is used by both the DeepSpeed training engine (for fused transformer layers) and the inference engines (V1 and V2) for kernel-injected inference.

---

## Architecture

### Directory Structure

```
deepspeed/module_inject/
    __init__.py                  # Public API exports
    inject.py                    # Main module_inject() function
    replace_policy.py            # Layer replacement policies
    replace_module.py            # Module replacement engine
    auto_tp.py                   # AutoTP: automatic tensor parallelism
    auto_tp_model_utils.py       # Model-specific AutoTP utilities
    layers.py                    # Optimized layer implementations
    policy.py                    # Policy definitions and helpers
    module_quantize.py           # Module quantization (INT8/INT4)
    fusedqkv_utils.py            # Fused QKV projection utilities
    tp_plan_converter.py         # Convert HuggingFace tp_plan to DeepSpeed format
    
    containers/                  # Model-specific container implementations
        __init__.py
        base.py                  # DSModelContainer base class
        base_moe.py              # DSMoEModelContainer for MoE models
        distil_bert.py           # DistilBERT container
        gptneo.py                # GPT-Neo container
        llama2.py                # LLaMA 2 container
        opt.py                   # OPT container
        bert.py                  # BERT container
        gptj.py                  # GPT-J container
        gpt_neox.py              # GPT-NeoX container
        bloom.py                 # BLOOM container
        mistral.py               # Mistral container
        mixtral.py               # Mixtral MoE container
        falcon.py                # Falcon container
        phi.py                   # Phi container
        qwen.py                  # Qwen container
        stable_diffusion.py      # Stable Diffusion container
```

### Class Hierarchy

```
DSModelContainer (base.py)
    |
    +-- DSDistilBertContainer (distil_bert.py)
    +-- DSGPTNeoContainer (gptneo.py)
    +-- DSLLAMA2Container (llama2.py)
    +-- DSOPTContainer (opt.py)
    +-- DSBERTContainer (bert.py)
    +-- DSGPTJContainer (gptj.py)
    +-- DSGPTNeoXContainer (gpt_neox.py)
    +-- DSBloomContainer (bloom.py)
    +-- DSMistralContainer (mistral.py)
    +-- DSMixtralContainer (mixtral.py)
    +-- DSFalconContainer (falcon.py)
    +-- DSPhiContainer (phi.py)
    +-- DSQwenContainer (qwen.py)
    +-- DSStableDiffusionContainer (stable_diffusion.py)

DSMoEModelContainer (base_moe.py)
    |
    +-- DSMixtralContainer (mixtral.py)
```

### Injection Flow

```
User Model (HuggingFace/PyTorch)
    |
    v
module_inject() called
    |
    v
1. Parse model architecture
    |-- Identify transformer layers
    |-- Count layers, heads, hidden size
    |
    v
2. Find matching injection policy
    |-- Check model type (e.g., "llama", "gpt_neox", "opt")
    |-- Load policy rules for this model
    |
    v
3. Apply AutoTP (if enabled)
    |-- Determine TP degree
    |-- Convert tp_plan or generate default
    |-- Configure parameter partitioning
    |
    v
4. Replace modules
    |-- For each layer matching a policy rule:
    |       |-- Create optimized replacement
    |       |-- Copy weights from original
    |       |-- Replace module in model tree
    |
    v
5. Apply quantization (if enabled)
    |-- Replace linear layers with quantized versions
    |-- Convert weights to INT8/INT4
    |
    v
6. Return modified model
```

---

## Module Injection System

### `module_inject()` Main Function

The `module_inject()` function in `deepspeed/module_inject/inject.py` is the primary entry point for module replacement.

```python
# deepspeed/module_inject/inject.py

def module_inject(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[dict] = None,
    hidden_size: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    interpolate_pos: bool = False,
    onnx: bool = False,
    training: bool = True,
    replace_with_kernel: bool = False,
    linear_layer_replacement: bool = True,
    max_out_dim: Optional[int] = None,
    min_out_dim: Optional[int] = None,
    seed: Optional[int] = None,
   三角化: bool = False,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
    """Replace model modules with optimized DeepSpeed implementations.
    
    Args:
        model: The PyTorch model to optimize
        optimizer: Optional optimizer to preserve state for
        config: DeepSpeed configuration dict
        hidden_size: Override hidden dimension size
        num_attention_heads: Override number of attention heads
        interpolate_pos: Interpolate positional embeddings
        onnx: Export mode for ONNX compatibility
        training: Whether the model is for training (vs inference)
        replace_with_kernel: Force kernel replacement
        linear_layer_replacement: Replace linear layers
        max_out_dim: Maximum output dimension for layer replacement
        min_out_dim: Minimum output dimension for layer replacement
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified_model, optimizer)
    """
```

### Injection Process Detail

```python
def module_inject(model, optimizer=None, config=None, **kwargs):
    # Step 1: Get the appropriate replacement policy
    policy = get_replace_policy(model, config)
    
    if policy is None:
        logger.warning("No matching injection policy found for model type")
        return model, optimizer
    
    # Step 2: Configure replacement parameters
    replace_ctx = ReplaceContext(
        model=model,
        policy=policy,
        config=config,
        training=kwargs.get('training', True),
        replace_with_kernel=kwargs.get('replace_with_kernel', False),
    )
    
    # Step 3: Walk model tree and replace modules
    replaced_model = replace_modules(replace_ctx)
    
    # Step 4: Update optimizer if provided
    if optimizer is not None:
        optimizer = update_optimizer(replaced_model, optimizer)
    
    return replaced_model, optimizer
```

---

## Injection Policies

Injection policies define mapping rules from standard PyTorch/HuggingFace modules to DeepSpeed optimized implementations. Each policy specifies:

1. **Source module class**: The original `nn.Module` class to replace
2. **Target module class**: The DeepSpeed optimized replacement
3. **Weight transfer rules**: How to copy weights from source to target
4. **Configuration extraction**: How to derive configuration from the source module

### Policy Definition

```python
# deepspeed/module_inject/replace_policy.py

class ReplacePolicy:
    """Base class for module replacement policies."""
    
    def __init__(self):
        self.source_cls = None           # Original module class
        self.target_cls = None           # Replacement module class
        self.weight_mapping = {}         # Parameter name mapping
        self.config_extractor = None     # Function to extract config
    
    def match(self, module: nn.Module) -> bool:
        """Check if this policy applies to the given module."""
        return isinstance(module, self.source_cls)
    
    def get_config(self, module: nn.Module) -> dict:
        """Extract configuration from the source module."""
        if self.config_extractor:
            return self.config_extractor(module)
        return {}
    
    def transfer_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target module."""
        for src_name, tgt_name in self.weight_mapping.items():
            src_param = getattr(source, src_name)
            tgt_param = getattr(target, tgt_name)
            tgt_param.data.copy_(src_param.data)
```

### Policy Registration

```python
# Policy registry for different model architectures

POLICIES = {
    # Transformer Layer Policies
    "bert": BertPolicy,
    "gpt_neox": GPTNeoXPolicy,
    "gptj": GPTJPolicy,
    "gpt_neo": GPTNeoPolicy,
    "opt": OPTPolicy,
    "bloom": BloomPolicy,
    "llama": LLAMAPolicy,
    "mistral": MistralPolicy,
    "mixtral": MixtralPolicy,
    "falcon": FalconPolicy,
    "phi": PhiPolicy,
    "qwen": QwenPolicy,
    
    # Diffusion Model Policies
    "stable_diffusion": StableDiffusionPolicy,
}

def get_replace_policy(model, config=None):
    """Find the matching replacement policy for a model.
    
    Uses model class name, config attributes, and module structure
    to determine the appropriate policy.
    """
    model_type = detect_model_type(model)
    policy_cls = POLICIES.get(model_type)
    if policy_cls is None:
        return None
    return policy_cls()
```

### Example Policy: LLAMA

```python
class LLAMAPolicy(ReplacePolicy):
    """Replacement policy for LLaMA models."""
    
    def __init__(self):
        super().__init__()
        self.source_cls = LlamaDecoderLayer  # HuggingFace LlamaDecoderLayer
        self.target_cls = DeepSpeedTransformerLayer  # DeepSpeed fused layer
        
    def get_config(self, module):
        config = {
            "hidden_size": module.hidden_size,
            "num_attention_heads": module.num_heads,
            "intermediate_size": module.intermediate_size,
            "pre_layer_norm": True,  # LLaMA uses pre-LN
            "rotary_embedding": True,  # LLaMA uses RoPE
            "mlp_after_attn": True,
            "activation": "silu",  # LLaMA uses SiLU (SwiGLU)
        }
        return config
    
    def transfer_weights(self, source, target):
        # Map QKV weights
        target.attn.qkvw.data.copy_(
            torch.cat([source.self_attn.q_proj.weight,
                       source.self_attn.k_proj.weight,
                       source.self_attn.v_proj.weight], dim=0)
        )
        # Map output projection
        target.attn.ow.data.copy_(source.self_attn.o_proj.weight)
        # Map MLP weights
        target.mlp.w1.data.copy_(source.mlp.gate_proj.weight)
        target.mlp.w2.data.copy_(source.mlp.up_proj.weight)
        target.mlp.w3.data.copy_(source.mlp.down_proj.weight)
        # Map layer norms
        target.norm1.weight.data.copy_(source.input_layernorm.weight)
        target.norm2.weight.data.copy_(source.post_attention_layernorm.weight)
```

### Example Policy: OPT

```python
class OPTPolicy(ReplacePolicy):
    """Replacement policy for OPT models."""
    
    def __init__(self):
        super().__init__()
        self.source_cls = OPTDecoderLayer
        self.target_cls = DeepSpeedTransformerLayer
    
    def get_config(self, module):
        config = {
            "hidden_size": module.hidden_size,
            "num_attention_heads": module.num_heads,
            "intermediate_size": module.fc1.in_features * 4,
            "pre_layer_norm": False,  # OPT uses post-LN
            "rotary_embedding": False,
            "mlp_after_attn": True,
            "activation": "relu",  # OPT uses ReLU
        }
        return config
    
    def transfer_weights(self, source, target):
        # OPT has separate Q, K, V projections
        target.attn.qkvw.data.copy_(
            torch.cat([source.self_attn.q_proj.weight,
                       source.self_attn.k_proj.weight,
                       source.self_attn.v_proj.weight], dim=0)
        )
        target.attn.ow.data.copy_(source.self_attn.out_proj.weight)
        target.mlp.w1.data.copy_(source.fc1.weight)
        target.mlp.w2.data.copy_(source.fc2.weight)
        target.norm1.weight.data.copy_(source.self_attn_layer_norm.weight)
        target.norm1.bias.data.copy_(source.self_attn_layer_norm.bias)
        target.norm2.weight.data.copy_(source.final_layer_norm.weight)
        target.norm2.bias.data.copy_(source.final_layer_norm.bias)
```

---

## AutoTP: Automatic Tensor Parallelism

AutoTP (`deepspeed/module_inject/auto_tp.py`) automatically partitions model parameters across GPUs for tensor-parallel training and inference. It analyzes the model architecture and generates a partition plan without requiring manual specification.

### AutoTP Architecture

```python
# deepspeed/module_inject/auto_tp.py

class AutoTP:
    """Automatic Tensor Parallelism engine.
    
    Analyzes model architecture and creates a partition plan
    that distributes parameters across tensor-parallel GPUs.
    """
    
    def __init__(self, model, tp_degree, config=None):
        """
        Args:
            model (nn.Module): The model to partition
            tp_degree (int): Number of tensor-parallel GPUs
            config (dict): Optional DeepSpeed configuration
        """
        self.model = model
        self.tp_degree = tp_degree
        self.config = config
        self.tp_plan = None
    
    def analyze_and_partition(self):
        """Main entry point: analyze model and apply tensor parallelism.
        
        Steps:
        1. Detect model architecture
        2. Find or generate TP plan
        3. Partition linear layers
        4. Adjust embedding layers
        5. Insert communication primitives
        """
        # 1. Detect architecture
        model_type = self._detect_model_type()
        
        # 2. Get TP plan
        self.tp_plan = self._get_tp_plan(model_type)
        
        # 3. Apply partitioning
        self._apply_partitioning()
        
        # 4. Insert communication ops
        self._insert_communication()
        
        return self.model
```

### TP Plan Generation

```python
def _get_tp_plan(self, model_type):
    """Get or generate the tensor parallelism plan.
    
    Priority:
    1. HuggingFace model.config.tp_plan (if available)
    2. DeepSpeed built-in plan for known models
    3. Auto-generated plan based on heuristics
    """
    # Try HuggingFace tp_plan first
    if hasattr(self.model, 'config') and hasattr(self.model.config, 'tp_plan'):
        from deepspeed.module_inject.tp_plan_converter import convert_tp_plan
        return convert_tp_plan(self.model.config.tp_plan)
    
    # Use built-in plans
    builtin_plans = {
        "llama": self._llama_tp_plan(),
        "opt": self._opt_tp_plan(),
        "bloom": self._bloom_tp_plan(),
        "gpt_neox": self._gpt_neox_tp_plan(),
        "mistral": self._mistral_tp_plan(),
        "mixtral": self._mixtral_tp_plan(),
        "falcon": self._falcon_tp_plan(),
        "qwen": self._qwen_tp_plan(),
    }
    
    plan = builtin_plans.get(model_type)
    if plan is not None:
        return plan
    
    # Auto-generate plan
    return self._auto_generate_plan()
```

### Column-Parallel and Row-Parallel Linear Layers

AutoTP partitions linear layers using two strategies:

#### Column-Parallel (Partition output dimension)

```
Original: Y = X @ W        where W is [in_features, out_features]

Partitioned:
  Y_0 = X @ W_0    (W_0 is [in_features, out_features/N])
  Y_1 = X @ W_1    (W_1 is [in_features, out_features/N])
  ...
  Y = [Y_0, Y_1, ..., Y_{N-1}]  (concatenate along columns)

Memory per GPU: in_features * out_features / N
```

Use for: QKV projections, first MLP linear layer

#### Row-Parallel (Partition input dimension)

```
Original: Y = X @ W        where W is [in_features, out_features]

Partitioned:
  Y_0 = X_0 @ W_0    (W_0 is [in_features/N, out_features])
  Y_1 = X_1 @ W_1    (W_1 is [in_features/N, out_features])
  ...
  Y = Y_0 + Y_1 + ... + Y_{N-1}  (all-reduce)

Memory per GPU: in_features * out_features / N
```

Use for: Output projection, second MLP linear layer

### TP Plan Format

```python
# TP plan is a dict mapping parameter names to partition specs
tp_plan = {
    # "param_name": ("column" or "row", dimension_index)
    
    # Attention
    "self_attn.q_proj.weight": ("column", 0),
    "self_attn.k_proj.weight": ("column", 0),
    "self_attn.v_proj.weight": ("column", 0),
    "self_attn.o_proj.weight": ("row", 1),
    
    # MLP
    "mlp.gate_proj.weight": ("column", 0),
    "mlp.up_proj.weight": ("column", 0),
    "mlp.down_proj.weight": ("row", 1),
}
```

### Communication Insertion

```python
def _insert_communication(self):
    """Insert all-reduce / all-gather ops for tensor parallelism.
    
    Pattern for each transformer layer:
    
    Input --> [Column-Parallel Linear] --> ... --> [Row-Parallel Linear] --> All-Reduce --> Output
    
    The all-reduce after the row-parallel layer synchronizes partial
    results from all TP ranks.
    """
    for name, module in self.model.named_modules():
        if self._is_transformer_layer(module):
            self._insert_all_reduce_after_row_parallel(module)
            self._insert_all_gather_before_column_parallel(module)
```

### AutoTP Configuration

```json
{
    "tensor_parallel": {
        "enabled": true,
        "tp_size": 4,
        "tp_mode": "auto",
        "moe_tp_mode": "auto"
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable automatic tensor parallelism |
| `tp_size` | int | `1` | Tensor parallel degree (number of GPUs) |
| `tp_mode` | str | `"auto"` | Mode: "auto", "manual", "off" |
| `moe_tp_mode` | str | `"auto"` | MoE-specific TP mode |
| `tp_plan` | dict | `null` | Manual TP plan override |
| `verify_tp` | bool | `false` | Verify correctness of TP partitioning |

### AutoTP Model Utilities

`auto_tp_model_utils.py` provides model-specific helper functions:

```python
# deepspeed/module_inject/auto_tp_model_utils.py

def get_llama_tp_plan(model_config):
    """Generate TP plan for LLaMA models.
    
    LLaMA architecture:
    - Q, K, V projections: column parallel
    - O projection: row parallel
    - gate/up projections: column parallel
    - down projection: row parallel
    """
    return {
        "self_attn.q_proj": {"weight": ("column", 0), "bias": None},
        "self_attn.k_proj": {"weight": ("column", 0), "bias": None},
        "self_attn.v_proj": {"weight": ("column", 0), "bias": None},
        "self_attn.o_proj": {"weight": ("row", 1), "bias": None},
        "mlp.gate_proj": {"weight": ("column", 0), "bias": None},
        "mlp.up_proj": {"weight": ("column", 0), "bias": None},
        "mlp.down_proj": {"weight": ("row", 1), "bias": None},
    }

def get_mistral_tp_plan(model_config):
    """Generate TP plan for Mistral models.
    
    Similar to LLaMA but supports GQA (Grouped Query Attention).
    K, V projections may have fewer heads than Q.
    """
    plan = get_llama_tp_plan(model_config)
    return plan

def get_mixtral_tp_plan(model_config):
    """Generate TP plan for Mixtral MoE models.
    
    Extends Mistral plan with expert parallelism:
    - Expert MLP layers can be partitioned across TP or EP
    - Router is replicated (small)
    """
    plan = get_mistral_tp_plan(model_config)
    # Add expert-specific partitioning
    for i in range(model_config.num_local_experts):
        plan[f"block_sparse_moe.experts.{i}.w1"] = {"weight": ("column", 0)}
        plan[f"block_sparse_moe.experts.{i}.w2"] = {"weight": ("row", 1)}
        plan[f"block_sparse_moe.experts.{i}.w3"] = {"weight": ("column", 0)}
    return plan
```

---

## Model Containers

Model containers (`deepspeed/module_inject/containers/`) provide model-specific wrappers that handle weight loading, configuration extraction, and module replacement for each supported architecture.

### Base Container

```python
# deepspeed/module_inject/containers/base.py

class DSModelContainer:
    """Base class for model containers.
    
    Each container wraps a specific model architecture and provides:
    - Configuration extraction from the original model
    - Weight mapping between original and DeepSpeed layers
    - Module replacement logic
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        self._parsed = False
        self._hidden_size = None
        self._num_heads = None
        self._num_layers = None
        self._intermediate_size = None
        self._max_seq_length = None
    
    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def num_heads(self):
        return self._num_heads
    
    @property
    def num_layers(self):
        return self._num_layers
    
    @property
    def intermediate_size(self):
        return self._intermediate_size
    
    def parse_model(self):
        """Extract architecture parameters from the model.
        
        Must be implemented by each container subclass.
        """
        raise NotImplementedError
    
    def get_decoder_blocks(self):
        """Return list of transformer decoder blocks."""
        raise NotImplementedError
    
    def get_encoder_blocks(self):
        """Return list of transformer encoder blocks (if applicable)."""
        return []
    
    def create_config(self, ds_config=None):
        """Create DeepSpeedTransformerConfig from model parameters.
        
        Args:
            ds_config: Optional DeepSpeed configuration dict
        
        Returns:
            DeepSpeedTransformerConfig
        """
        from deepspeed.ops.transformer import DeepSpeedTransformerConfig
        
        return DeepSpeedTransformerConfig(
            batch_size=self._get_batch_size(ds_config),
            max_seq_length=self._max_seq_length or 2048,
            hidden_size=self._hidden_size,
            heads=self._num_heads,
            intermediate_size=self._intermediate_size,
            num_hidden_layers=self._num_layers,
            fp16=ds_config.get("fp16", {}).get("enabled", False) if ds_config else False,
            bf16=ds_config.get("bf16", {}).get("enabled", False) if ds_config else False,
            pre_layer_norm=self._uses_pre_layernorm(),
            rotary_embedding=self._uses_rotary_embedding(),
        )
    
    def _uses_pre_layernorm(self):
        return True  # Default: most modern models use pre-LN
    
    def _uses_rotary_embedding(self):
        return False  # Override in models that use RoPE
    
    def _get_batch_size(self, ds_config):
        if ds_config and "train_batch_size" in ds_config:
            return ds_config["train_batch_size"]
        return 1
```

### LLAMA2 Container

```python
# deepspeed/module_inject/containers/llama2.py

class DSLLAMA2Container(DSModelContainer):
    """Container for LLaMA 2 and LLaMA 3 models."""
    
    def parse_model(self):
        """Parse LLaMA model architecture."""
        # LLaMA structure: model.layers[i].* 
        if hasattr(self.model, 'model'):
            base_model = self.model.model
        else:
            base_model = self.model
        
        # Extract config
        if hasattr(self.model, 'config'):
            cfg = self.model.config
            self._hidden_size = cfg.hidden_size
            self._num_heads = cfg.num_attention_heads
            self._num_key_value_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
            self._intermediate_size = cfg.intermediate_size
            self._num_layers = cfg.num_hidden_layers
            self._max_seq_length = getattr(cfg, 'max_position_embeddings', 4096)
            self._vocab_size = cfg.vocab_size
            self._rms_norm_eps = getattr(cfg, 'rms_norm_eps', 1e-5)
        
        self._parsed = True
    
    def get_decoder_blocks(self):
        if hasattr(self.model, 'model'):
            return list(self.model.model.layers)
        return list(self.model.layers)
    
    def _uses_pre_layernorm(self):
        return True
    
    def _uses_rotary_embedding(self):
        return True
```

### MoE Container

```python
# deepspeed/module_inject/containers/base_moe.py

class DSMoEModelContainer(DSModelContainer):
    """Base container for Mixture of Experts models.
    
    Extends DSModelContainer with MoE-specific functionality:
    - Expert counting and configuration
    - Router weight extraction
    - Expert parallel partitioning
    """
    
    def __init__(self, model, config=None):
        super().__init__(model, config)
        self._num_experts = None
        self._num_experts_per_tok = None
        self._expert_type = None
    
    @property
    def num_experts(self):
        return self._num_experts
    
    @property
    def num_experts_per_tok(self):
        return self._num_experts_per_tok
    
    def get_expert_blocks(self, layer_idx):
        """Return expert modules for a given MoE layer."""
        raise NotImplementedError
    
    def get_router(self, layer_idx):
        """Return the router/gate module for a given MoE layer."""
        raise NotImplementedError
```

### Container Registry

```python
# Model type to container mapping
CONTAINER_REGISTRY = {
    "llama": DSLLAMA2Container,
    "mistral": DSMistralContainer,
    "mixtral": DSMixtralContainer,
    "opt": DSOPTContainer,
    "bloom": DSBloomContainer,
    "gpt_neox": DSGPTNeoXContainer,
    "gptj": DSGPTJContainer,
    "gpt_neo": DSGPTNeoContainer,
    "falcon": DSFalconContainer,
    "phi": DSPhiContainer,
    "qwen": DSQwenContainer,
    "distilbert": DSDistilBertContainer,
    "bert": DSBERTContainer,
}

def get_container(model, config=None):
    """Get the appropriate container for a model."""
    model_type = detect_model_type(model)
    container_cls = CONTAINER_REGISTRY.get(model_type)
    if container_cls is None:
        raise ValueError(f"No container found for model type '{model_type}'")
    return container_cls(model, config)
```

---

## Module Quantization

The `module_quantize.py` module replaces standard linear layers with quantized implementations for efficient inference.

### Quantization API

```python
# deepspeed/module_inject/module_quantize.py

def quantize_module(
    model: nn.Module,
    q_type: str = "int8",
    q_groups: int = 1,
    merge_count: int = 1,
    mlp_extra_grouping: bool = False,
    quantize_rms_norm: bool = True,
    quantize_rotary_embedding: bool = True,
    quantize_logits: bool = False,
) -> nn.Module:
    """Quantize model linear layers for efficient inference.
    
    Args:
        model: The model to quantize
        q_type: Quantization type ("int8", "int4", "fp8")
        q_groups: Number of quantization groups per tensor
        merge_count: Number of layers to merge for grouping
        mlp_extra_grouping: Additional grouping for MLP layers
        quantize_rms_norm: Quantize RMS normalization layers
        quantize_rotary_embedding: Quantize rotary embedding computation
        quantize_logits: Quantize the final logits computation
    
    Returns:
        nn.Module: Model with quantized linear layers
    """
```

### Quantization Process

```python
def quantize_module(model, q_type="int8", q_groups=1, **kwargs):
    for name, module in model.named_modules():
        parent_name, child_name = _get_parent_child_name(name)
        
        if isinstance(module, nn.Linear):
            # Replace with quantized linear layer
            quantized = _create_quantized_linear(module, q_type, q_groups)
            _replace_module(model, parent_name, child_name, quantized)
        
        elif isinstance(module, nn.LayerNorm) and kwargs.get('quantize_rms_norm'):
            # Replace with quantized norm
            quantized = _create_quantized_norm(module, q_type)
            _replace_module(model, parent_name, child_name, quantized)
    
    return model

def _create_quantized_linear(linear, q_type, q_groups):
    """Create a quantized version of a linear layer."""
    from deepspeed.ops.quantizer import quantize
    
    weight = linear.weight.data
    bias = linear.bias.data if linear.bias is not None else None
    
    # Quantize weights
    q_weight, q_scale, q_min = quantize(
        weight, q_type=q_type, q_groups=q_groups
    )
    
    return QuantizedLinear(
        original_shape=weight.shape,
        q_weight=q_weight,
        q_scale=q_scale,
        q_min=q_min,
        bias=bias,
        q_type=q_type,
        in_features=linear.in_features,
        out_features=linear.out_features,
    )
```

### QuantizedLinear Module

```python
class QuantizedLinear(nn.Module):
    """Quantized linear layer for efficient inference.
    
    Stores weights in quantized format (INT8 or INT4) and
    dequantizes on-the-fly during forward pass, or uses
    specialized quantized GEMM kernels.
    """
    
    def __init__(self, original_shape, q_weight, q_scale, q_min, 
                 bias, q_type, in_features, out_features):
        super().__init__()
        self.original_shape = original_shape
        self.q_weight = nn.Parameter(q_weight, requires_grad=False)
        self.q_scale = nn.Parameter(q_scale, requires_grad=False)
        self.q_min = nn.Parameter(q_min, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.q_type = q_type
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, input):
        # Dequantize and compute
        from deepspeed.ops.quantizer import dequantize
        weight = dequantize(self.q_weight, self.q_scale, self.q_min, 
                           q_type=self.q_type)
        return torch.nn.functional.linear(input, weight, self.bias)
```

### Quantization Configuration

```json
{
    "injection_config": {
        "quantize": {
            "enabled": true,
            "q_type": "int8",
            "q_groups": 1,
            "quantize_rms_norm": true,
            "quantize_rotary_embedding": true
        }
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable quantization |
| `q_type` | str | `"int8"` | Quantization type: "int8", "int4", "fp8" |
| `q_groups` | int | `1` | Number of quantization groups |
| `quantize_rms_norm` | bool | `true` | Quantize RMS norm layers |
| `quantize_rotary_embedding` | bool | `true` | Quantize rotary embedding |
| `quantize_logits` | bool | `false` | Quantize final logits layer |

### Memory Savings from Quantization

| Precision | Bytes per Parameter | 7B Model Size | Compression Ratio |
|-----------|-------------------|---------------|------------------|
| FP32 | 4 | 28 GB | 1x |
| FP16 | 2 | 14 GB | 2x |
| BF16 | 2 | 14 GB | 2x |
| INT8 | 1 | 7 GB | 4x |
| INT4 | 0.5 | 3.5 GB | 8x |

---

## FusedQKV Utilities

The `fusedqkv_utils.py` module provides utilities for fusing separate Q, K, V projections into a single fused QKV operation.

### FusedQKV Class

```python
# deepspeed/module_inject/fusedqkv_utils.py

class FusedQKV:
    """Utility for fusing separate Q, K, V linear layers.
    
    Fuses three separate linear projections (Q, K, V) into a single
    linear layer with concatenated weights, reducing kernel launches
    from 3 to 1 for the QKV projection.
    """
    
    def __init__(self, q_proj, k_proj, v_proj, add_bias=True):
        """
        Args:
            q_proj (nn.Linear): Query projection
            k_proj (nn.Linear): Key projection
            v_proj (nn.Linear): Value projection
            add_bias (bool): Whether to include bias
        """
        self.hidden_size = q_proj.in_features
        self.head_size = q_proj.out_features // q_proj.num_heads
        self.num_heads = q_proj.num_heads
        self.add_bias = add_bias
    
    @staticmethod
    def fuse_weights(q_proj, k_proj, v_proj):
        """Fuse Q, K, V weights into a single tensor.
        
        Returns:
            Tuple of (fused_weight, fused_bias) tensors
        """
        fused_weight = torch.cat([
            q_proj.weight.data,
            k_proj.weight.data,
            v_proj.weight.data,
        ], dim=0)
        
        fused_bias = None
        if q_proj.bias is not None:
            fused_bias = torch.cat([
                q_proj.bias.data,
                k_proj.bias.data,
                v_proj.bias.data,
            ], dim=0)
        
        return fused_weight, fused_bias
    
    @staticmethod
    def unfuse_weights(fused_weight, fused_bias, num_heads, head_size, kv_heads=None):
        """Unfuse a fused QKV weight back into separate Q, K, V weights.
        
        Args:
            fused_weight: [3 * num_heads * head_size, hidden_size] or
                          [(num_heads + 2 * kv_heads) * head_size, hidden_size]
            fused_bias: Optional bias tensor
            num_heads: Number of query heads
            head_size: Size of each head
            kv_heads: Number of KV heads (for GQA, default: num_heads)
        
        Returns:
            Tuple of (q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)
        """
        if kv_heads is None:
            kv_heads = num_heads
        
        q_dim = num_heads * head_size
        k_dim = kv_heads * head_size
        v_dim = kv_heads * head_size
        
        q_weight = fused_weight[:q_dim]
        k_weight = fused_weight[q_dim:q_dim + k_dim]
        v_weight = fused_weight[q_dim + k_dim:]
        
        q_bias = k_bias = v_bias = None
        if fused_bias is not None:
            q_bias = fused_bias[:q_dim]
            k_bias = fused_bias[q_dim:q_dim + k_dim]
            v_bias = fused_bias[q_dim + k_dim:]
        
        return q_weight, k_weight, v_weight, q_bias, k_bias, v_bias
```

### Grouped Query Attention (GQA) Support

For models using GQA (like LLaMA 2 70B, Mistral), K and V projections have fewer heads than Q:

```python
def fuse_gqa_weights(q_proj, k_proj, v_proj, num_q_heads, num_kv_heads, head_size):
    """Fuse QKV weights for GQA models.
    
    Q: [num_q_heads * head_size, hidden_size]
    K: [num_kv_heads * head_size, hidden_size]
    V: [num_kv_heads * head_size, hidden_size]
    
    Fused: [(num_q_heads + 2 * num_kv_heads) * head_size, hidden_size]
    """
    fused_weight = torch.cat([
        q_proj.weight.data,    # [num_q_heads * head_size, hidden]
        k_proj.weight.data,    # [num_kv_heads * head_size, hidden]
        v_proj.weight.data,    # [num_kv_heads * head_size, hidden]
    ], dim=0)
    
    return fused_weight
```

---

## TP Plan Conversion

The `tp_plan_converter.py` module converts HuggingFace's `tp_plan` format to DeepSpeed's internal tensor parallelism plan.

### HuggingFace tp_plan Format

HuggingFace models may include a `tp_plan` in their configuration:

```python
# HuggingFace tp_plan format (example for LLaMA)
tp_plan = {
    "model.embed_tokens": "embed",
    "model.layers.*.self_attn.q_proj": "column",
    "model.layers.*.self_attn.k_proj": "column",
    "model.layers.*.self_attn.v_proj": "column",
    "model.layers.*.self_attn.o_proj": "row",
    "model.layers.*.mlp.gate_proj": "column",
    "model.layers.*.mlp.up_proj": "column",
    "model.layers.*.mlp.down_proj": "row",
    "lm_head": "column",
}
```

### Conversion Function

```python
# deepspeed/module_inject/tp_plan_converter.py

def convert_tp_plan(hf_tp_plan, num_layers=None):
    """Convert HuggingFace tp_plan to DeepSpeed format.
    
    Args:
        hf_tp_plan (dict): HuggingFace tp_plan from model config
        num_layers (int): Number of transformer layers (for expanding wildcards)
    
    Returns:
        dict: DeepSpeed TP plan with concrete parameter mappings
    """
    ds_plan = {}
    
    for pattern, strategy in hf_tp_plan.items():
        if strategy == "embed":
            # Embedding: partition along vocab dimension
            ds_plan[pattern + ".weight"] = ("column", 1)
        
        elif strategy == "column":
            if "*" in pattern:
                # Expand wildcard for all layers
                for i in range(num_layers):
                    concrete = pattern.replace("*", str(i))
                    ds_plan[concrete + ".weight"] = ("column", 0)
                    if _has_bias(pattern, model):
                        ds_plan[concrete + ".bias"] = ("column", 0)
            else:
                ds_plan[pattern + ".weight"] = ("column", 0)
        
        elif strategy == "row":
            if "*" in pattern:
                for i in range(num_layers):
                    concrete = pattern.replace("*", str(i))
                    ds_plan[concrete + ".weight"] = ("row", 1)
                    if _has_bias(pattern, model):
                        ds_plan[concrete + ".bias"] = ("row", 0)
            else:
                ds_plan[pattern + ".weight"] = ("row", 1)
    
    return ds_plan
```

### TP Plan for Inference V2

For the inference V2 engine, TP plans are converted to a different format:

```python
def convert_to_v2_plan(ds_plan):
    """Convert DeepSpeed TP plan to inference V2 format.
    
    V2 uses a per-layer specification:
    {
        "layer_type": {
            "param_name": PartitionSpec(...)
        }
    }
    """
    from deepspeed.inference.v2.modules import PartitionSpec
    
    v2_plan = {}
    for param_path, (strategy, dim) in ds_plan.items():
        parts = param_path.split(".")
        
        # Group by layer type
        layer_key = ".".join(parts[:-2])  # e.g., "self_attn.q_proj"
        param_name = parts[-1]            # e.g., "weight"
        
        if layer_key not in v2_plan:
            v2_plan[layer_key] = {}
        
        if strategy == "column":
            v2_plan[layer_key][param_name] = PartitionSpec(
                partition_dim=dim,
                partition_type="column",
                replicate=False,
            )
        elif strategy == "row":
            v2_plan[layer_key][param_name] = PartitionSpec(
                partition_dim=dim,
                partition_type="row",
                replicate=False,
            )
    
    return v2_plan
```

---

## Model Parsing and Policy Application

### Model Type Detection

```python
def detect_model_type(model):
    """Detect the model architecture type from the model object.
    
    Checks:
    1. model.config.model_type (HuggingFace models)
    2. Class name patterns
    3. Module structure analysis
    
    Returns:
        str: Model type identifier (e.g., "llama", "opt", "bloom")
    """
    # Check HuggingFace config
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type
        # Normalize variations
        type_mapping = {
            "llama": "llama",
            "mistral": "mistral",
            "mixtral": "mixtral",
            "opt": "opt",
            "bloom": "bloom",
            "gpt_neox": "gpt_neox",
            "gptj": "gptj",
            "gpt_neo": "gpt_neo",
            "falcon": "falcon",
            "phi": "phi",
            "qwen2": "qwen",
            "stable-diffusion": "stable_diffusion",
        }
        return type_mapping.get(model_type, model_type)
    
    # Check class name
    class_name = model.__class__.__name__.lower()
    for key in ["llama", "mistral", "mixtral", "opt", "bloom", "gpt", "falcon", "phi", "qwen"]:
        if key in class_name:
            return key
    
    return None
```

### Policy Application

```python
def apply_policy(model, policy, container, config=None):
    """Apply injection policy to a model.
    
    Walks the model tree and replaces modules according to the policy.
    
    Args:
        model: The model to modify
        policy: The replacement policy
        container: The model container with configuration
        config: Optional DeepSpeed configuration
    
    Returns:
        nn.Module: The modified model
    """
    # Create DeepSpeed config from container
    ds_config = container.create_config(config)
    
    # Get decoder blocks to replace
    decoder_blocks = container.get_decoder_blocks()
    
    for i, block in enumerate(decoder_blocks):
        # Create DeepSpeed transformer layer
        ds_layer = DeepSpeedTransformerLayer(ds_config)
        
        # Transfer weights from original to DeepSpeed layer
        policy.transfer_weights(block, ds_layer)
        
        # Replace in model tree
        _replace_decoder_block(model, i, ds_layer)
    
    return model
```

### Module Replacement Utility

```python
def _replace_decoder_block(model, layer_idx, new_block):
    """Replace a decoder block in the model tree.
    
    Args:
        model: The model containing the decoder blocks
        layer_idx: Index of the block to replace
        new_block: The replacement module
    """
    # Handle different model structures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers[layer_idx] = new_block
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        model.transformer.h[layer_idx] = new_block
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        model.decoder.layers[layer_idx] = new_block
    else:
        raise ValueError(f"Cannot find decoder blocks in model structure")
```

---

## Supported Model List

### Supported Models for Module Injection

| Model | Architecture | Container | Injection | AutoTP | MoE |
|-------|-------------|-----------|-----------|--------|-----|
| **LLaMA 1/2/3** | Causal LM | `DSLLAMA2Container` | Yes | Yes | No |
| **LLaMA 2 70B** | Causal LM (GQA) | `DSLLAMA2Container` | Yes | Yes | No |
| **Mistral 7B** | Causal LM (GQA) | `DSMistralContainer` | Yes | Yes | No |
| **Mixtral 8x7B** | MoE Causal LM | `DSMixtralContainer` | Yes | Yes | Yes |
| **OPT** | Causal LM | `DSOPTContainer` | Yes | Yes | No |
| **BLOOM** | Causal LM | `DSBloomContainer` | Yes | Yes | No |
| **GPT-NeoX** | Causal LM | `DSGPTNeoXContainer` | Yes | Yes | No |
| **GPT-J** | Causal LM | `DSGPTJContainer` | Yes | Yes | No |
| **GPT-Neo** | Causal LM | `DSGPTNeoContainer` | Yes | Yes | No |
| **Falcon** | Causal LM | `DSFalconContainer` | Yes | Yes | No |
| **Phi-1/2** | Causal LM | `DSPhiContainer` | Yes | Yes | No |
| **Qwen/Qwen2** | Causal LM | `DSQwenContainer` | Yes | Yes | No |
| **BERT** | Masked LM | `DSBERTContainer` | Yes | Yes | No |
| **DistilBERT** | Masked LM | `DSDistilBertContainer` | Yes | Yes | No |
| **Stable Diffusion** | Diffusion | `DSStableDiffusionContainer` | Partial | No | No |

### Supported Models for Inference V2

The inference V2 engine has its own set of model implementations in `deepspeed/inference/v2/model_implementations/`:

| Model | Architecture | GQA | MoE | Quantization |
|-------|-------------|-----|-----|-------------|
| LLaMA 2/3 | Causal LM | Yes | No | INT8, INT4 |
| Mistral | Causal LM | Yes | No | INT8, INT4 |
| Mixtral | MoE Causal LM | Yes | Yes | INT8 |
| Falcon | Causal LM | No | No | INT8 |
| Phi-2 | Causal LM | No | No | INT8 |
| Phi-3 | Causal LM | Yes | No | INT8 |
| OPT | Causal LM | No | No | INT8, INT4 |
| Qwen | Causal LM | No | No | INT8 |
| Qwen2 | Causal LM | Yes | No | INT8 |
| Qwen2-MoE | MoE Causal LM | Yes | Yes | INT8 |
| Exaone4 | Causal LM | Yes | No | INT8 |

---

## Configuration Examples

### Basic Module Injection for Training

```python
import deepspeed
from deepspeed.module_inject import module_inject

# Load HuggingFace model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Apply module injection for fused kernels
model, optimizer = module_inject(
    model=model,
    optimizer=optimizer,
    config=ds_config,
    training=True,
    replace_with_kernel=True,
)

# Or via deepspeed.initialize()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
)
```

```json
{
    "train_batch_size": 32,
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3
    }
}
```

### AutoTP Configuration

```python
# Enable AutoTP via DeepSpeed config
ds_config = {
    "train_batch_size": 32,
    "tensor_parallel": {
        "enabled": True,
        "tp_size": 4,
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2
    }
}

# Launch with 4 GPUs for TP
# deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### Inference with Kernel Injection

```python
from deepspeed.inference.engine import InferenceEngine

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Initialize inference engine with kernel injection
engine = InferenceEngine(
    model=model,
    tensor_parallel={"tp_size": 2},
    dtype=torch.float16,
    replace_with_kernel_inject=True,  # Enable kernel injection
)

# Run inference
outputs = engine.generate(input_ids)
```

### Inference with Quantization

```python
from deepspeed.module_inject.module_quantize import quantize_module

# Load model in FP32
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float32)

# Quantize to INT8
model = quantize_module(
    model,
    q_type="int8",
    q_groups=1,
    quantize_rms_norm=True,
)

# Initialize inference engine
engine = InferenceEngine(model=model, tensor_parallel={"tp_size": 1})
```

### Mixtral MoE with AutoTP

```json
{
    "train_batch_size": 16,
    "tensor_parallel": {
        "enabled": true,
        "tp_size": 4,
        "moe_tp_mode": "auto"
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5
        }
    }
}
```

### Custom TP Plan

```python
# Define custom TP plan manually
from deepspeed.module_inject.auto_tp import AutoTP

tp_plan = {
    "model.layers.*.self_attn.q_proj.weight": ("column", 0),
    "model.layers.*.self_attn.k_proj.weight": ("column", 0),
    "model.layers.*.self_attn.v_proj.weight": ("column", 0),
    "model.layers.*.self_attn.o_proj.weight": ("row", 1),
    "model.layers.*.mlp.gate_proj.weight": ("column", 0),
    "model.layers.*.mlp.up_proj.weight": ("column", 0),
    "model.layers.*.mlp.down_proj.weight": ("row", 1),
}

auto_tp = AutoTP(model, tp_degree=4)
auto_tp.tp_plan = tp_plan
model = auto_tp.analyze_and_partition()
```

### Combined Injection + TP + Quantization for Inference

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.2")

# DeepSpeed inference configuration
ds_config = {
    "tensor_parallel": {
        "tp_size": 2,
    },
    "injection_config": {
        "quantize": {
            "enabled": True,
            "q_type": "int8",
            "q_groups": 1,
        }
    }
}

# Initialize
model_engine = deepspeed.init_inference(
    model=model,
    mp_size=2,                             # Model parallel (TP) size
    dtype=torch.float16,
    replace_with_kernel_inject=True,       # Enable fused kernels
    quantization_config=ds_config.get("injection_config", {}).get("quantize"),
)

# Generate
with torch.no_grad():
    output = model_engine.generate(input_ids, max_new_tokens=100)
```

---

## Troubleshooting

### Injection Policy Not Found

**Symptom**: `Warning: No matching injection policy found for model type`

**Solutions**:
1. Check if the model type is in the supported list
2. Verify model has a `config.model_type` attribute
3. Try specifying the model type manually:
   ```python
   from deepspeed.module_inject import set_model_type
   set_model_type("llama")
   ```

### Weight Transfer Errors

**Symptom**: `RuntimeError: shape mismatch in weight transfer`

**Solutions**:
1. Ensure the model architecture matches the expected structure
2. Check that no weights have been modified before injection
3. Try loading model with `torch_dtype=torch.float16`:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "model_name", torch_dtype=torch.float16
   )
   ```

### AutoTP Shape Errors

**Symptom**: `RuntimeError: Expected all tensors to be on the same device` or shape mismatches during forward pass

**Solutions**:
1. Verify TP degree matches number of GPUs: `--num_gpus=4` with `"tp_size": 4`
2. Check that all linear layers are covered by the TP plan
3. Enable verification: `"verify_tp": true`

### Quantization Accuracy Loss

**Symptom**: Significant accuracy degradation after INT8/INT4 quantization

**Solutions**:
1. Use more quantization groups: `"q_groups": 4` or `"q_groups": 8`
2. Skip quantizing sensitive layers (like the final logits head):
   ```python
   quantize_module(model, q_type="int8", quantize_logits=False)
   ```
3. Use calibration: run a few forward passes with representative data before quantizing

### Fused QKV GQA Mismatch

**Symptom**: Shape errors when fusing QKV for GQA models (Mistral, LLaMA 2 70B)

**Solutions**:
1. Ensure the number of KV heads is correctly detected:
   ```python
   print(model.config.num_key_value_heads)
   ```
2. If incorrect, set manually:
   ```python
   from deepspeed.module_inject.fusedqkv_utils import FusedQKV
   fused = FusedQKV(q_proj, k_proj, v_proj)
   fused.kv_heads = model.config.num_key_value_heads
   ```

### Module Injection with Pipeline Parallelism

**Symptom**: Errors when combining module injection with pipeline parallelism

**Solutions**:
1. Module injection should be applied before pipeline partitioning
2. Ensure injection targets the correct stage's layers
3. Consider disabling injection for pipeline parallelism and using ZeRO Stage 3 instead
