# DeepSpeed Inference Engine V1 Reference

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [DeepSpeedInferenceConfig](#deepspeedinferenceconfig)
4. [InferenceEngine Class](#inferenceengine-class)
5. [Kernel Injection Mechanism](#kernel-injection-mechanism)
6. [Supported Models](#supported-models)
7. [Precision and Dtype Conversion](#precision-and-dtype-conversion)
8. [CUDA Graph Support](#cuda-graph-support)
9. [Triton Kernel Integration](#triton-kernel-integration)
10. [Checkpoint Loading](#checkpoint-loading)
11. [Quantization Support](#quantization-support)
12. [MoE Inference Support](#moe-inference-support)
13. [Profiling and Generation](#profiling-and-generation)
14. [Tensor Parallel Inference](#tensor-parallel-inference)
15. [Configuration Examples](#configuration-examples)
16. [Code Examples](#code-examples)
17. [Troubleshooting](#troubleshooting)

---

## Overview

DeepSpeed Inference Engine V1 provides high-performance inference for transformer models through kernel injection, quantization, tensor parallelism, and CUDA graph optimization. It replaces standard PyTorch modules with optimized CUDA kernels that deliver significant speedups over naive implementations.

Key capabilities:
- **Kernel injection**: Automatically replace standard transformer ops with fused CUDA kernels
- **Multiple precision support**: fp16, bf16, fp32, and int8 quantization
- **Tensor parallel inference**: Distribute model across multiple GPUs for large models
- **CUDA graph capture**: Eliminate kernel launch overhead for fixed-shape inference
- **MoE inference**: Efficient sparse expert evaluation
- **Triton kernels**: Custom Triton-based kernels for emerging operations
- **Checkpoint loading**: Direct loading from DeepSpeed and HuggingFace checkpoints
- **Automatic dtype conversion**: Seamless precision conversion at load time

---

## Architecture

### Module Location

```
deepspeed/inference/
  __init__.py
  engine.py               # InferenceEngine class
  config.py               # DeepSpeedInferenceConfig
  rp_utils.py             # Replacement policy utilities
```

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Engine Pipeline                     │
│                                                                  │
│  1. Load Model                                                  │
│     ┌───────────────┐                                           │
│     │ HuggingFace   │──┐                                        │
│     │ Checkpoint    │  │                                        │
│     └───────────────┘  │   ┌───────────────────┐               │
│                        ├──►│ InferenceEngine    │               │
│     ┌───────────────┐  │   │                    │               │
│     │ DeepSpeed     │──┘   │  2. Apply Injection │               │
│     │ Checkpoint    │      │     Policy          │               │
│     └───────────────┘      │                    │               │
│                            │  3. Convert Dtype   │               │
│                            │                    │               │
│                            │  4. Load Checkpoint │               │
│                            │                    │               │
│                            │  5. Apply TP        │               │
│                            │                    │               │
│                            │  6. Capture CUDA    │               │
│                            │     Graph (opt.)    │               │
│                            └────────┬───────────┘               │
│                                     │                            │
│                            ┌────────▼───────────┐               │
│                            │ Optimized Model     │               │
│                            │ (CUDA Kernels)      │               │
│                            └────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## DeepSpeedInferenceConfig

The configuration class for the inference engine, controlling all aspects of inference optimization.

### Full Configuration

```python
@dataclass
class DeepSpeedInferenceConfig:
    """Configuration for DeepSpeed Inference Engine V1."""

    # --- Core Settings ---
    replace_with_kernel_inject: bool = True
    dtype: torch.dtype = torch.float16

    # --- Tensor Parallel ---
    tensor_parallel: Optional["DeepSpeedTPConfig"] = None

    # --- Performance ---
    enable_cuda_graph: bool = False
    use_triton: bool = False
    triangular_masking: bool = True

    # --- MoE ---
    moe: Optional["DeepSpeedMoEConfig"] = None

    # --- Memory Management ---
    keep_module_on_host: bool = False
    checkpoint: Optional[Dict] = None
    base_dir: Optional[str] = None
    set_empty_params: bool = False
    save_mp_checkpoint_path: Optional[str] = None

    # --- Model Behavior ---
    return_tuple: bool = True
    training_mp_size: int = 1
    injection_policy: Optional[Dict] = None

    # --- Generation ---
    max_out_tokens: int = 1024
    min_out_tokens: int = 1

    # --- Quantization ---
    quant: Optional["QuantizationConfig"] = None
```

### Parameter Reference

#### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `replace_with_kernel_inject` | `bool` | `True` | Replace PyTorch modules with optimized CUDA kernels |
| `dtype` | `torch.dtype` | `torch.float16` | Target data type for inference. Options: `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.int8` |

#### Tensor Parallel

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel` | `DeepSpeedTPConfig` | `None` | Tensor parallel configuration |

**DeepSpeedTPConfig:**

```python
@dataclass
class DeepSpeedTPConfig:
    enabled: bool = False
    tp_size: int = 1
    tp_grain_size: int = 64
    mpu: Optional[object] = None  # Model parallel unit
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable tensor parallel inference |
| `tp_size` | `int` | `1` | Number of GPUs for tensor parallelism |
| `tp_grain_size` | `int` | `64` | Minimum grain size for weight partitioning |
| `mpu` | `object` | `None` | Custom model parallel unit |

#### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_cuda_graph` | `bool` | `False` | Capture CUDA graphs for reduced kernel launch overhead |
| `use_triton` | `bool` | `False` | Use Triton kernels for custom operations |
| `triangular_masking` | `bool` | `True` | Use causal (triangular) attention masking |

#### MoE

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `moe` | `DeepSpeedMoEConfig` | `None` | MoE configuration for sparse expert models |

**DeepSpeedMoEConfig:**

```python
@dataclass
class DeepSpeedMoEConfig:
    enabled: bool = False
    ep_size: int = 1
    moe_experts: int = 1
    type: str = "standard"
    ep_mp_group: Optional[object] = None
    ep_group: Optional[object] = None
```

#### Memory Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keep_module_on_host` | `bool` | `False` | Keep model weights on CPU, transfer to GPU on demand |
| `checkpoint` | `Dict` | `None` | Checkpoint dictionary for direct weight loading |
| `base_dir` | `str` | `None` | Base directory for checkpoint files |
| `set_empty_params` | `bool` | `False` | Initialize model with empty (uninitialized) parameters |
| `save_mp_checkpoint_path` | `str` | `None` | Path to save model-parallel checkpoint after loading |

#### Model Behavior

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_tuple` | `bool` | `True` | Return outputs as tuple (instead of dict/tensor) |
| `training_mp_size` | `int` | `1` | Model parallel size used during training (for checkpoint compatibility) |
| `injection_policy` | `Dict` | `None` | Custom injection policy for unsupported models |

#### Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_out_tokens` | `int` | `1024` | Maximum number of output tokens for generation |
| `min_out_tokens` | `int` | `1` | Minimum number of output tokens for generation |

#### Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quant` | `QuantizationConfig` | `None` | Quantization configuration |

### JSON Configuration

```json
{
    "replace_with_kernel_inject": true,
    "dtype": "fp16",
    "tensor_parallel": {
        "enabled": true,
        "tp_size": 4,
        "tp_grain_size": 64
    },
    "enable_cuda_graph": false,
    "use_triton": false,
    "triangular_masking": true,
    "max_out_tokens": 1024,
    "min_out_tokens": 1,
    "moe": {
        "enabled": false
    }
}
```

---

## InferenceEngine Class

The main inference engine class that orchestrates model optimization and inference.

### Class Definition

```python
class InferenceEngine(torch.nn.Module):
    """DeepSpeed Inference Engine V1.

    Wraps a PyTorch model and optimizes it for inference through:
    - Kernel injection (replacing modules with CUDA kernels)
    - Precision conversion
    - Tensor parallelism
    - CUDA graph capture
    - Quantization
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: DeepSpeedInferenceConfig,
    ):
        super().__init__()
        self.config = config
        self.model = model

        # Initialize inference engine
        self._apply_injection_policy()
        self._convert_to_dtype()
        self._load_checkpoint()
        self._apply_tensor_parallelism()
        self._capture_cuda_graph()
```

### Key Methods

#### \_\_init\_\_

```python
def __init__(self, model, config):
    """Initialize the inference engine.

    Args:
        model: PyTorch model to optimize.
        config: DeepSpeedInferenceConfig with inference settings.
    """
    super().__init__()
    self.config = config
    self.model = model
    self._validate_config()
    self._setup_engine()
```

#### forward

```python
def forward(self, *inputs, **kwargs):
    """Run forward pass through the optimized model.

    The forward pass uses injected CUDA kernels when available,
    falling back to standard PyTorch operations otherwise.
    """
    return self.model(*inputs, **kwargs)
```

#### generate

```python
def generate(self, input_ids, **kwargs):
    """Generate tokens from the model.

    Overrides the model's generate method with optimized inference.

    Args:
        input_ids: Input token IDs [batch_size, seq_len].
        **kwargs: Additional generation arguments (max_length, temperature, etc.)

    Returns:
        Generated token IDs [batch_size, output_seq_len].
    """
    # Set generation config
    self.config.max_out_tokens = kwargs.pop("max_length", self.config.max_out_tokens)
    self.config.min_out_tokens = kwargs.pop("min_length", self.config.min_out_tokens)

    return self.model.generate(input_ids, **kwargs)
```

---

## Kernel Injection Mechanism

Kernel injection replaces standard PyTorch transformer modules with optimized CUDA kernel implementations. This is the core optimization mechanism of InferenceEngine V1.

### _apply_injection_policy()

```python
def _apply_injection_policy(self):
    """Apply kernel injection based on the configured policy.

    The injection policy maps module types to their optimized replacements:
    1. Identify target modules by class type
    2. Replace each target with an optimized CUDA kernel module
    3. Copy weights from the original module to the replacement
    """
    if not self.config.replace_with_kernel_inject:
        return

    # Determine injection policy
    if self.config.injection_policy is not None:
        policy = self.config.injection_policy
    else:
        policy = self._get_default_injection_policy()

    # Apply policy to model
    for module_name, module in list(self.model.named_modules()):
        module_type = type(module).__name__
        if module_type in policy:
            replacement_cls = policy[module_type]
            self._replace_module(module_name, module, replacement_cls)
```

### Injection Policy Structure

An injection policy is a dictionary mapping module class names to their replacement classes:

```python
injection_policy = {
    "BertLayer": DeepSpeedBERTLayer,
    "GPT2Block": DeepSpeedGPT2Block,
    "LlamaDecoderLayer": DeepSpeedLlamaLayer,
    # ... etc.
}
```

### Default Injection Policies by Model

```python
def _get_default_injection_policy(self):
    """Auto-detect the model type and return the appropriate injection policy."""
    model_class = type(self.model).__name__

    policies = {
        "BertModel": BERT_POLICY,
        "BertForMaskedLM": BERT_POLICY,
        "BertForSequenceClassification": BERT_POLICY,
        "GPT2LMHeadModel": GPT2_POLICY,
        "GPTNeoForCausalLM": GPTNEO_POLICY,
        "GPTNeoXForCausalLM": GPTNEOX_POLICY,
        "GPTJForCausalLM": GPTJ_POLICY,
        "LlamaForCausalLM": LLAMA_POLICY,
        "OPTForCausalLM": OPT_POLICY,
        "BloomForCausalLM": BLOOM_POLICY,
        "MistralForCausalLM": MISTRAL_POLICY,
        "MixtralForCausalLM": MIXTRAL_POLICY,
    }

    # Check by model class name
    for key, policy in policies.items():
        if key in model_class:
            return policy

    # Try to infer from model structure
    return self._infer_policy_from_structure()
```

### Module Replacement Process

```python
def _replace_module(self, module_name, original_module, replacement_cls):
    """Replace a module with its optimized counterpart.

    Steps:
    1. Instantiate the replacement module
    2. Copy weights from original to replacement
    3. Set the replacement in the model hierarchy
    """
    # Navigate to parent module
    parts = module_name.split(".")
    parent = self.model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Create replacement
    replacement = replacement_cls(
        config=self.config,
        original_module=original_module,
    )

    # Set replacement
    setattr(parent, parts[-1], replacement)
```

### Kernel Types

DeepSpeed uses several categories of optimized kernels:

| Kernel Type | Operation | Speedup |
|------------|-----------|---------|
| Fused Attention | QKV + attention + output projection | 2-4x |
| Fused FFN | Gate + up + down projection | 1.5-2x |
| Fused LayerNorm | Layer normalization | 1.5-3x |
| Fused Bias+GeLU | Bias addition + GeLU activation | 2-3x |
| Fused QKV | Separate Q, K, V projections combined | 1.5-2x |
| Vectorized Adam | Optimized Adam kernel (training) | 1.5-2x |

---

## Supported Models

### Built-in Injection Policies

| Model Family | HuggingFace Class | Injection Policy | Kernel Support |
|-------------|-------------------|-----------------|---------------|
| BERT | `BertModel`, `BertForMaskedLM` | `BERT_POLICY` | Attention, FFN, LayerNorm |
| GPT-2 | `GPT2LMHeadModel` | `GPT2_POLICY` | Attention, FFN, LayerNorm |
| GPT-Neo | `GPTNeoForCausalLM` | `GPTNEO_POLICY` | Attention, FFN |
| GPT-NeoX | `GPTNeoXForCausalLM` | `GPTNEOX_POLICY` | Attention, FFN, LayerNorm |
| GPT-J | `GPTJForCausalLM` | `GPTJ_POLICY` | Attention, FFN |
| LLaMA | `LlamaForCausalLM` | `LLAMA_POLICY` | Attention, FFN, RMSNorm |
| OPT | `OPTForCausalLM` | `OPT_POLICY` | Attention, FFN, LayerNorm |
| BLOOM | `BloomForCausalLM` | `BLOOM_POLICY` | Attention, FFN, LayerNorm |
| Mistral | `MistralForCausalLM` | `MISTRAL_POLICY` | Attention, FFN, RMSNorm |
| Mixtral | `MixtralForCausalLM` | `MIXTRAL_POLICY` | Attention, MoE FFN, RMSNorm |
| Falcon | `FalconForCausalLM` | `FALCON_POLICY` | Attention, FFN, LayerNorm |

### Custom Injection Policy

For models not in the built-in list, provide a custom injection policy:

```python
# Define custom injection policy
custom_policy = {
    "CustomTransformerLayer": CustomOptimizedLayer,
}

# Use with InferenceEngine
engine = deepspeed.init_inference(
    model=model,
    injection_policy=custom_policy,
    dtype=torch.float16,
)
```

### Injection Policy for a New Model

```python
from deepspeed.ops.op_builder import InferenceBuilder

# Define which submodules to replace
class CustomPolicy:
    """Injection policy for a custom transformer model."""

    def __init__(self):
        self.replacement_map = {
            # (module_class, submodule_name) -> replacement_function
            (CustomAttention, "self_attn"): self._replace_attention,
            (CustomFFN, "ffn"): self._replace_ffn,
        }

    def _replace_attention(self, original):
        """Replace attention with fused CUDA kernel."""
        return FusedAttention(
            hidden_size=original.hidden_size,
            num_heads=original.num_heads,
            dtype=self.dtype,
        )

    def _replace_ffn(self, original):
        """Replace FFN with fused CUDA kernel."""
        return FusedFFN(
            hidden_size=original.hidden_size,
            intermediate_size=original.intermediate_size,
            dtype=self.dtype,
        )
```

---

## Precision and Dtype Conversion

### _convert_to_dtype()

```python
def _convert_to_dtype(self):
    """Convert model parameters to the target dtype.

    Supports:
    - torch.float16 (fp16): 2 bytes per parameter
    - torch.bfloat16 (bf16): 2 bytes per parameter, larger dynamic range
    - torch.float32 (fp32): 4 bytes per parameter (no conversion)
    - torch.int8: 1 byte per parameter (requires quantization)
    """
    target_dtype = self.config.dtype

    if target_dtype == torch.int8:
        # Int8 requires quantization, handled by quant module
        self._apply_quantization()
        return

    for param in self.model.parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)

    # Also convert buffers (e.g., position embeddings)
    for buffer in self.model.buffers():
        if buffer.dtype != target_dtype:
            buffer.data = buffer.data.to(target_dtype)
```

### Dtype Selection Guide

| Dtype | Memory | Dynamic Range | Speed | Recommended Use |
|-------|--------|--------------|-------|----------------|
| `torch.float32` | 4x | Full | Baseline | Debugging, numerical analysis |
| `torch.float16` | 2x | 5.96e-8 to 65504 | Fast | General inference (GPU with FP16 support) |
| `torch.bfloat16` | 2x | 9.2e-41 to 3.39e38 | Fast | Training-like inference, A100/H100 |
| `torch.int8` | 1x | -128 to 127 | Fastest | Maximum throughput, latency-sensitive |

### Dtype Conversion Examples

```python
# FP16 inference
engine = deepspeed.init_inference(model, dtype=torch.float16)

# BF16 inference (recommended for A100/H100)
engine = deepspeed.init_inference(model, dtype=torch.bfloat16)

# INT8 quantized inference
engine = deepspeed.init_inference(
    model,
    dtype=torch.int8,
    quant=QuantizationConfig(
        enabled=True,
        weight_quant=WeightQuantConfig(enabled=True),
    ),
)
```

---

## CUDA Graph Support

CUDA graph capture records the GPU operations into a graph that can be replayed with minimal CPU overhead, eliminating kernel launch latency.

### How CUDA Graphs Work

```
Without CUDA Graph:
  CPU: Launch kernel 1 → Wait → Launch kernel 2 → Wait → ...
  GPU: [kernel1] [idle] [kernel2] [idle] [kernel3] ...

With CUDA Graph:
  CPU: Launch graph (single call)
  GPU: [kernel1][kernel2][kernel3]... (no gaps)
```

### Enabling CUDA Graphs

```python
engine = deepspeed.init_inference(
    model,
    enable_cuda_graph=True,
    dtype=torch.float16,
)
```

### CUDA Graph Requirements

1. **Fixed input shapes**: Input tensor shapes must remain constant across calls
2. **No dynamic control flow**: No data-dependent branching or variable-length sequences
3. **Sufficient warm-up**: At least one forward pass before graph capture

### CUDA Graph Capture Process

```python
def _capture_cuda_graph(self):
    """Capture the forward pass as a CUDA graph."""
    if not self.config.enable_cuda_graph:
        return

    # Warm-up: run forward pass once to initialize all buffers
    dummy_input = self._create_dummy_input()
    _ = self.model(dummy_input)
    torch.cuda.synchronize()

    # Capture graph
    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph):
        self._static_output = self.model(dummy_input)
```

### CUDA Graph Replay

```python
def forward(self, *inputs, **kwargs):
    """Forward with optional CUDA graph replay."""
    if self.config.enable_cuda_graph:
        # Copy input to static buffer
        self._static_input.copy_(inputs[0])
        # Replay graph
        self.graph.replay()
        return self._static_output
    else:
        return self.model(*inputs, **kwargs)
```

---

## Triton Kernel Integration

DeepSpeed can use Triton kernels as an alternative to CUDA C++ kernels for certain operations, providing flexibility and portability.

### Enabling Triton Kernels

```python
engine = deepspeed.init_inference(
    model,
    use_triton=True,
    dtype=torch.float16,
)
```

### Triton Kernel Operations

| Operation | Triton Kernel | Advantage |
|-----------|--------------|-----------|
| Attention | `triton_attention` | Easier to customize attention patterns |
| LayerNorm | `triton_layernorm` | Competitive with CUDA, more portable |
| RMSNorm | `triton_rmsnorm` | Native support for RMS normalization |
| Activation | `triton_fused_bias_gelu` | Fused bias + activation |
| Linear | `triton_linear` | Custom tiling strategies |

### Custom Triton Kernel

```python
import triton
import triton.language as tl

@triton.jit
def fused_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for fused bias + GeLU."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)

    # GeLU(x + b)
    y = x + b
    y = y * 0.5 * (1.0 + tl.math.erf(y * 0.707106781))

    tl.store(Y_ptr + offsets, y, mask=mask)
```

---

## Checkpoint Loading

### load_model_with_checkpoint()

```python
def load_model_with_checkpoint(self):
    """Load model weights from a checkpoint.

    Supports multiple checkpoint formats:
    - DeepSpeed training checkpoints (ZeRO format)
    - HuggingFace model weights
    - Custom checkpoint dictionaries
    """
    if self.config.checkpoint is not None:
        self._load_from_dict(self.config.checkpoint)
    elif self.config.base_dir is not None:
        self._load_from_directory(self.config.base_dir)
```

### Loading from DeepSpeed Checkpoint

```python
engine = deepspeed.init_inference(
    model=model,
    base_dir="/path/to/deepspeed/checkpoint",
    dtype=torch.float16,
    training_mp_size=4,  # MP size used during training
)
```

### Loading from HuggingFace

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
engine = deepspeed.init_inference(model, dtype=torch.float16)
```

### Loading with set_empty_params

When `set_empty_params=True`, parameters are allocated but not initialized, allowing deferred loading:

```python
engine = deepspeed.init_inference(
    model=model,
    set_empty_params=True,
    dtype=torch.float16,
)
# Load weights manually later
engine.load_weights(checkpoint_path)
```

### Saving Model-Parallel Checkpoints

```python
engine = deepspeed.init_inference(
    model=model,
    save_mp_checkpoint_path="/path/to/mp_checkpoint",
    tensor_parallel={"enabled": True, "tp_size": 4},
    dtype=torch.float16,
)
# Checkpoint is saved automatically after loading and partitioning
```

---

## Quantization Support

DeepSpeed supports multiple quantization schemes for reducing memory footprint and improving inference throughput.

### QuantizationConfig

```python
@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    enabled: bool = False
    weight_quant: Optional["WeightQuantConfig"] = None
    activation_quant: Optional["ActivationQuantConfig"] = None
    qkv_quant: Optional["QKVQuantConfig"] = None
```

### WeightQuantConfig

```python
@dataclass
class WeightQuantConfig:
    """Weight quantization configuration."""
    enabled: bool = False
    num_bits: int = 8           # Quantization bits (4 or 8)
    group_size: int = 64        # Group size for group quantization
    sym: bool = True            # Symmetric quantization
    percentile: float = 99.9    # Percentile for calibration
```

### ActivationQuantConfig

```python
@dataclass
class ActivationQuantConfig:
    """Activation quantization configuration."""
    enabled: bool = False
    num_bits: int = 8
    group_size: int = 64
    sym: bool = True
    percentile: float = 99.9
```

### QKVQuantConfig

```python
@dataclass
class QKVQuantConfig:
    """QKV projection quantization configuration."""
    enabled: bool = False
    num_bits: int = 8
    group_size: int = 64
    sym: bool = True
```

### Quantization Examples

#### INT8 Weight Quantization

```python
from deepspeed.inference.config import QuantizationConfig, WeightQuantConfig

quant_config = QuantizationConfig(
    enabled=True,
    weight_quant=WeightQuantConfig(
        enabled=True,
        num_bits=8,
        group_size=64,
        sym=True,
    ),
)

engine = deepspeed.init_inference(
    model=model,
    dtype=torch.int8,
    quant=quant_config,
)
```

#### INT4 Weight-Only Quantization

```python
quant_config = QuantizationConfig(
    enabled=True,
    weight_quant=WeightQuantConfig(
        enabled=True,
        num_bits=4,
        group_size=128,
        sym=False,
    ),
)

engine = deepspeed.init_inference(
    model=model,
    dtype=torch.float16,  # Compute in fp16, weights in int4
    quant=quant_config,
)
```

### Quantization Memory Savings

| Quantization | Memory per 7B Model | Accuracy Impact |
|-------------|--------------------|-----------------|
| FP32 | 28 GB | Baseline |
| FP16 | 14 GB | Negligible |
| BF16 | 14 GB | Negligible |
| INT8 (weight) | 7 GB | < 0.1% perplexity increase |
| INT4 (weight) | 3.5 GB | 0.5-2% perplexity increase |

---

## MoE Inference Support

InferenceEngine V1 supports sparse MoE models with optimized expert evaluation.

### MoE Configuration for Inference

```python
from deepspeed.inference.config import DeepSpeedMoEConfig

moe_config = DeepSpeedMoEConfig(
    enabled=True,
    ep_size=4,
    moe_experts=8,
    type="standard",
)

engine = deepspeed.init_inference(
    model=model,
    moe=moe_config,
    dtype=torch.float16,
)
```

### MoE Inference Optimizations

1. **Expert batching**: Batch tokens assigned to the same expert
2. **Sparse activation**: Only compute through selected experts (k=2 out of E)
3. **Expert weight caching**: Keep expert weights in GPU memory

```python
# MoE inference with Mixtral
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
engine = deepspeed.init_inference(
    model=model,
    dtype=torch.float16,
    moe=DeepSpeedMoEConfig(enabled=True, ep_size=4, moe_experts=8),
    tensor_parallel={"enabled": True, "tp_size": 4},
)
```

---

## Profiling and Generation

### profile_model_time()

```python
def profile_model_time(self, input_ids, num_iterations=100):
    """Profile the model's inference time.

    Args:
        input_ids: Sample input tensor.
        num_iterations: Number of iterations for profiling.

    Returns:
        Dict with timing statistics:
        - mean_latency_ms: Mean latency per forward pass
        - p50_latency_ms: 50th percentile latency
        - p99_latency_ms: 99th percentile latency
        - throughput_tokens_per_sec: Tokens processed per second
    """
    # Warm-up
    for _ in range(10):
        _ = self.model(input_ids)
    torch.cuda.synchronize()

    # Profile
    latencies = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        _ = self.model(input_ids)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)  # ms

    latencies = sorted(latencies)
    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": latencies[len(latencies) // 2],
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        "throughput_tokens_per_sec": (
            input_ids.shape[0] * input_ids.shape[1]
            / (sum(latencies) / 1000 / len(latencies))
        ),
    }
```

### Usage

```python
# Profile inference
input_ids = torch.randint(0, 32000, (1, 128)).cuda()
stats = engine.profile_model_time(input_ids, num_iterations=100)
print(f"Mean latency: {stats['mean_latency_ms']:.1f} ms")
print(f"P99 latency: {stats['p99_latency_ms']:.1f} ms")
print(f"Throughput: {stats['throughput_tokens_per_sec']:.0f} tokens/s")
```

### generate() Method Override

```python
def generate(self, input_ids, **kwargs):
    """Generate tokens with DeepSpeed optimizations.

    Supports standard HuggingFace generation kwargs:
    - max_length, max_new_tokens
    - temperature, top_k, top_p
    - repetition_penalty
    - do_sample
    """
    max_out_tokens = kwargs.pop("max_new_tokens", self.config.max_out_tokens)
    min_out_tokens = kwargs.pop("min_new_tokens", self.config.min_out_tokens)

    # Use model's generate with optimized forward
    return self.model.generate(
        input_ids,
        max_new_tokens=max_out_tokens,
        min_new_tokens=min_out_tokens,
        **kwargs,
    )
```

---

## Tensor Parallel Inference

### Multi-GPU Inference with TP

```python
import deepspeed
import torch.distributed as dist

# Initialize distributed
dist.init_process_group("nccl")
tp_size = 4

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

engine = deepspeed.init_inference(
    model=model,
    mp_size=tp_size,
    dtype=torch.float16,
    tensor_parallel={
        "enabled": True,
        "tp_size": tp_size,
        "tp_grain_size": 64,
    },
)

# Run inference (each GPU holds 1/tp_size of model weights)
output = engine.generate(input_ids.cuda(), max_new_tokens=100)
```

### TP Inference Communication

During tensor parallel inference, all-reduce operations are inserted after row-parallel layers:

```
Forward pass on each GPU:
  1. Column parallel projection (local)
  2. Attention computation (local)
  3. Row parallel projection (local)
  4. All-reduce to combine results (communication)
  5. FFN (local)
```

### Launching Multi-GPU Inference

```bash
# 4-GPU tensor parallel inference
deepspeed --num_gpus=4 inference.py \
    --model_name meta-llama/Llama-2-70b-hf \
    --dtype fp16 \
    --tp_size 4
```

---

## Configuration Examples

### Example 1: Basic FP16 Inference

```json
{
    "replace_with_kernel_inject": true,
    "dtype": "fp16",
    "triangular_masking": true,
    "max_out_tokens": 512
}
```

```python
import deepspeed
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
engine = deepspeed.init_inference(model, dtype=torch.float16)
output = engine.generate(input_ids.cuda(), max_new_tokens=100)
```

### Example 2: BF16 with CUDA Graph

```python
engine = deepspeed.init_inference(
    model=model,
    dtype=torch.bfloat16,
    enable_cuda_graph=True,
)
```

### Example 3: Multi-GPU TP Inference

```python
engine = deepspeed.init_inference(
    model=model,
    mp_size=4,
    dtype=torch.float16,
    tensor_parallel={"enabled": True, "tp_size": 4},
)
```

### Example 4: Quantized INT8 Inference

```python
from deepspeed.inference.config import QuantizationConfig, WeightQuantConfig

engine = deepspeed.init_inference(
    model=model,
    dtype=torch.int8,
    quant=QuantizationConfig(
        enabled=True,
        weight_quant=WeightQuantConfig(enabled=True, num_bits=8, group_size=64),
    ),
)
```

### Example 5: MoE Inference (Mixtral)

```python
engine = deepspeed.init_inference(
    model=model,
    dtype=torch.float16,
    moe={"enabled": True, "ep_size": 4, "moe_experts": 8},
    tensor_parallel={"enabled": True, "tp_size": 4},
)
```

### Example 6: Custom Injection Policy

```python
# Define custom policy for unsupported model
policy = {
    "MyAttentionLayer": DeepSpeedAttentionReplacement,
    "MyFFNLayer": DeepSpeedFFNReplacement,
}

engine = deepspeed.init_inference(
    model=model,
    injection_policy=policy,
    dtype=torch.float16,
)
```

---

## Code Examples

### Example 1: Complete Inference Pipeline

```python
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Initialize DeepSpeed inference
engine = deepspeed.init_inference(
    model,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    triangular_masking=True,
    max_out_tokens=256,
)

# Generate
prompt = "Deep learning is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(engine.device)

output_ids = engine.generate(input_ids, max_new_tokens=100)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

### Example 2: Benchmarking Inference

```python
import torch
import deepspeed
import time

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
engine = deepspeed.init_inference(model, dtype=torch.float16)

# Benchmark different sequence lengths
for seq_len in [32, 64, 128, 256, 512, 1024]:
    input_ids = torch.randint(0, 32000, (1, seq_len)).cuda()

    # Profile
    stats = engine.profile_model_time(input_ids, num_iterations=50)
    print(f"Seq={seq_len}: mean={stats['mean_latency_ms']:.1f}ms, "
          f"p99={stats['p99_latency_ms']:.1f}ms, "
          f"throughput={stats['throughput_tokens_per_sec']:.0f} tok/s")
```

### Example 3: Multi-GPU Serving

```python
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model_name = "meta-llama/Llama-2-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": local_rank},
)

engine = deepspeed.init_inference(
    model,
    mp_size=4,
    dtype=torch.float16,
)

# Serve (simplified)
if local_rank == 0:
    prompt = "Explain quantum computing in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    output = engine.generate(input_ids, max_new_tokens=200)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Troubleshooting

### Common Issues

**1. Kernel injection fails for unsupported model**

```
RuntimeError: No injection policy found for model type CustomModel
```

Provide a custom injection policy or disable kernel injection:
```python
engine = deepspeed.init_inference(
    model,
    replace_with_kernel_inject=False,  # Disable injection
    dtype=torch.float16,
)
```

**2. CUDA graph capture fails with dynamic shapes**

```
RuntimeError: CUDA graphs do not support dynamic shapes
```

Ensure input shapes are fixed when using CUDA graphs, or disable:
```python
engine = deepspeed.init_inference(model, enable_cuda_graph=False)
```

**3. OOM during model loading**

For very large models, use CPU offloading:
```python
engine = deepspeed.init_inference(
    model,
    keep_module_on_host=True,
    dtype=torch.float16,
)
```

**4. Quantization accuracy degradation**

Use calibration data for better quantization:
```python
# Calibrate with representative data
engine.calibrate(calibration_dataloader)
# Then quantize
engine.quantize()
```

**5. NCCL timeout during TP inference**

Ensure all ranks are synchronized:
```python
dist.barrier()
output = engine.generate(input_ids)
dist.barrier()
```
