# Layers and Operations Reference

This document provides a comprehensive reference for all neural network layer implementations and operations in SGLang. It covers attention layers, linear layers, MoE, embeddings, positional encodings, normalization, activations, quantization, LoRA, multimodal encoders, and alternative attention architectures.

---

## Table of Contents

1. [Attention Layers](#attention-layers)
2. [Linear Layers](#linear-layers)
3. [MoE (Mixture of Experts) Layers](#moe-mixture-of-experts-layers)
4. [Embedding Layers](#embedding-layers)
5. [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
6. [Layer Normalization](#layer-normalization)
7. [Activation Functions](#activation-functions)
8. [Quantization Layers](#quantization-layers)
9. [LoRA Layers](#lora-layers)
10. [Multimodal Encoder Layers](#multimodal-encoder-layers)
11. [Mamba and Linear Attention Layers](#mamba-and-linear-attention-layers)
12. [Model Architecture Registry](#model-architecture-registry)
13. [Cross-Platform Support](#cross-platform-support)

---

## Attention Layers

### RadixAttention

The core attention module is `RadixAttention`, defined in `python/sglang/srt/layers/radix_attention.py`.

#### Attention Types

```python
class AttentionType(Enum):
    DECODER = "decoder"                     # Standard causal decoder attention
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"  # Bidirectional for image tokens
    ENCODER_ONLY = "encoder_only"           # Encoder-only attention
```

#### RadixAttention Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_heads` | int | Number of query heads |
| `head_dim` | int | Dimension per head |
| `scaling` | float | Scale factor for QK product |
| `num_kv_heads` | int | Number of key/value heads (for GQA/MQA) |
| `layer_id` | int | Layer index in the model |
| `logit_cap` | float | Maximum logit value (0 = no cap) |
| `v_head_dim` | int | Value head dimension (-1 = same as head_dim) |
| `sliding_window_size` | int | Sliding window size (-1 = full attention) |
| `is_cross_attention` | bool | Whether this is cross-attention |
| `pos_encoding_mode` | str | Position encoding mode ("NONE", "ROPE_LLAMA", etc.) |
| `logit_capping_method` | str | Method for logit capping ("tanh", etc.) |
| `quant_config` | QuantizationConfig | Quantization configuration |
| `attn_type` | AttentionType | Type of attention |
| `use_irope` | bool | Whether to use interleaved RoPE |

### Attention Backends

SGLang supports multiple attention backends. Each implements the `AttentionBackend` ABC from `base_attn_backend.py`.

#### Available Backends

| Backend | File | Hardware | Description |
|---------|------|----------|-------------|
| **FlashInfer** | `flashinfer_backend.py` | NVIDIA | Default backend using FlashInfer library. Supports paged KV cache, CUDA graphs. |
| **FlashAttention 3 (FA3)** | `flashattention_backend.py` | NVIDIA (H100+) | Uses FlashAttention 3 for improved performance on Hopper GPUs. |
| **FlashMLA** | `flashmla_backend.py` | NVIDIA | Optimized for MLA (Multi-Latent Attention) models like DeepSeek. |
| **FlashInfer MLA** | `flashinfer_mla_backend.py` | NVIDIA | FlashInfer-based MLA attention. |
| **CUTLASS MLA** | `cutlass_mla_backend.py` | NVIDIA | CUTLASS-based MLA attention implementation. |
| **Triton** | `triton_backend.py` | NVIDIA | Pure Triton kernel implementation. |
| **Torch Native** | `torch_native_backend.py` | All | PyTorch native attention (fallback). |
| **Torch Flex** | `torch_flex_backend.py` | All | PyTorch FlexAttention-based implementation. |
| **TRT-LLM MHA** | `trtllm_mha_backend.py` | NVIDIA | TensorRT-LLM multi-head attention backend. |
| **TRT-LLM MLA** | `trtllm_mla_backend.py` | NVIDIA | TensorRT-LLM MLA backend. |
| **Wave** | `wave_backend.py` | NVIDIA | Wave attention backend for specialized kernels. |
| **NSA** | `nsa_backend.py` | NVIDIA | Native Sparse Attention for DeepSeek models. |
| **Dual Chunk FA** | `dual_chunk_flashattention_backend.py` | NVIDIA | Dual-chunk FlashAttention for long contexts. |
| **Hybrid** | `hybrid_attn_backend.py` | All | Hybrid attention combining multiple backends. |
| **Hybrid Linear** | `hybrid_linear_attn_backend.py` | All | Hybrid linear attention for SSM models. |
| **TBO** | `tbo_backend.py` | All | Two-batch overlap attention backend. |
| **Intel AMX** | `intel_amx_backend.py` | Intel CPU | Intel AMX-accelerated attention. |
| **XPU** | `xpu_backend.py` | Intel GPU | Intel XPU attention. |

#### Attention Backend Interface

```python
class AttentionBackend(ABC):
    def init_forward_metadata(self, forward_batch: ForwardBatch)
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int)
    def init_forward_metadata_capture_cuda_graph(self, ...)
    def init_forward_metadata_replay_cuda_graph(self, ...)
    def get_cuda_graph_seq_len_fill_value(self)
```

#### NSA (Native Sparse Attention)

The NSA backend supports DeepSeek's sparse attention architecture:

- Located in `python/sglang/srt/layers/attention/nsa/`
- Includes specialized indexer for top-k token selection
- Supports `flashmla_sparse` decode backend
- Quantized K-cache for memory efficiency

### Attention Backend Selection

```bash
# Select attention backend
python -m sglang.launch_server \
    --attention-backend flashinfer  # Options: flashinfer, fa3, triton, etc.
```

---

## Linear Layers

Linear layers are implemented in `python/sglang/srt/layers/linear.py`.

### Linear Layer Types

#### QKVParallelLinear

Fused QKV projection with tensor parallelism:

- Concatenates Q, K, V projections into a single operation.
- Shards Q heads across TP ranks.
- Shards KV heads across TP ranks (for GQA).

#### ColumnParallelLinear

Column-parallel linear layer:

- Weight is sharded along the output dimension.
- Each TP rank holds `output_size / tp_size` columns.
- Output is gathered across ranks (all-reduce or all-gather).

#### RowParallelLinear

Row-parallel linear layer:

- Weight is sharded along the input dimension.
- Each TP rank holds `input_size / tp_size` rows.
- Results are reduced across ranks.

### Fused Operations

#### SiluAndMul (Fused Gate-Up Projection)

Combines the SwiGLU activation with the gate and up projections:

```python
# Separate: gate = silu(gate_proj(x)) * up_proj(x)
# Fused:   output = silu_and_mul([gate_proj(x), up_proj(x)])
```

This is a single fused kernel that splits the input tensor, applies SiLU to the first half, and multiplies with the second half.

### Weight Loading Methods

SGLang supports multiple weight loading formats (via `WEIGHT_LOADER_V2_SUPPORTED`):

| Method | Description |
|--------|-------------|
| `CompressedTensorsLinearMethod` | Neural Magic compressed tensors |
| `AWQLinearMethod` | AWQ quantized weights |
| `GPTQMarlinLinearMethod` | GPTQ with Marlin kernels |
| `Fp8LinearMethod` | FP8 quantization |
| `BlockInt8LinearMethod` | Block-wise INT8 |
| `MarlinLinearMethod` | Marline sparse quantization |
| `GPTQLinearMethod` | GPTQ standard |
| `ModelOptFp8LinearMethod` | NVIDIA ModelOpt FP8 |
| `PetitNvFp4LinearMethod` | Petit FP4 |
| `QuarkInt4Fp8LinearMethod` | Quark INT4-FP8 mixed |

### Parameter Types

The linear layer system uses specialized parameter types for weight loading:

| Parameter Class | Description |
|----------------|-------------|
| `PackedColumnParameter` | Column-parallel packed weights |
| `RowvLLMParameter` | Row-parallel weights |
| `PerTensorScaleParameter` | Per-tensor quantization scales |
| `BlockQuantScaleParameter` | Block-wise quantization scales |

---

## MoE (Mixture of Experts) Layers

Mixture of Experts layers are located in `python/sglang/srt/layers/moe/`.

### MoE Architecture

```
Input Tokens
    │
    ▼
Router (gate) → Top-K Selection
    │
    ├── Expert 0 ──┐
    ├── Expert 1 ──┤
    ├── ...        ├──→ Weighted Sum → Output
    └── Expert N-1 ┘
```

### Key Components

#### Router (`router.py`)

- Computes routing logits for each token.
- Applies top-k selection.
- Supports auxiliary loss for load balancing.

#### Top-K Selection (`topk.py`)

- Selects the top-k experts for each token.
- Supports different top-k algorithms and configurations.

#### Token Dispatcher (`token_dispatcher/`)

- Routes tokens to their selected experts.
- Handles the all-to-all communication for expert parallelism.

#### MoE Runners

| Runner | File | Description |
|--------|------|-------------|
| **Fused MoE Triton** | `fused_moe_triton/` | Triton-based fused MoE kernels |
| **Fused MoE Native** | `fused_moe_native.py` | PyTorch-native MoE implementation |
| **CUTLASS MoE** | `cutlass_moe.py` | CUTLASS-accelerated MoE |
| **CUTLASS W4A8 MoE** | `cutlass_w4a8_moe.py` | CUTLASS with W4A8 quantization |
| **FlashInfer TRT-LLM** | `flashinfer_trtllm_moe.py` | FlashInfer + TRT-LLM MoE |
| **FlashInfer CuteDSL** | `flashinfer_cutedsl_moe.py` | FlashInfer CuteDSL MoE |
| **EP MoE** | `ep_moe/` | Expert parallelism MoE |
| **KT EP Wrapper** | `kt_ep_wrapper.py` | Kernel-tuned expert parallelism |

### Expert Parallelism

SGLang supports expert parallelism (EP) for MoE models:

- **EP size**: Number of expert parallel groups.
- **Token routing**: All-to-all communication between EP ranks.
- **Load balancing**: Auxiliary loss or expert choice routing.

### MoE Configuration

```bash
# Enable expert parallelism for MoE models
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --ep-size 8 \
    --moe-dp-size 1
```

---

## Embedding Layers

### VocabParallelEmbedding

Defined in `python/sglang/srt/layers/vocab_parallel_embedding.py`:

- Distributes the embedding table across tensor parallel ranks.
- Each rank holds `vocab_size / tp_size` embeddings.
- Vocabulary size is padded to be divisible by `tp_size` (default padding: 64).

#### Key Features

- **Parallel lookup**: Each rank looks up its local portion.
- **All-reduce**: Results are summed across ranks.
- **Quantization support**: Embeddings can be quantized.
- **AMX support**: Intel AMX acceleration for CPU inference.

### Embedding Parameters

| Parameter | Description |
|-----------|-------------|
| `num_embeddings` | Vocabulary size |
| `embedding_dim` | Embedding dimension |
| `DEFAULT_VOCAB_PADDING_SIZE` | 64 (alignment for TP) |

### Pooler Layer

The `pooler.py` module provides pooling operations for embedding and classification models:

- **Mean pooling**: Average of token embeddings.
- **CLS token**: First token embedding.
- **Last token**: Last token embedding.

### Sparse Pooler

`sparse_pooler.py` provides sparse pooling for models that use sparse attention patterns.

---

## Rotary Position Embedding (RoPE)

RoPE implementations are in `python/sglang/srt/layers/rotary_embedding/`.

### Factory Pattern

The `factory.py` provides a factory for creating position embeddings based on model configuration:

```python
def create_rotary_embedding(config, head_dim, ...)
```

### RoPE Variants

| File | Description |
|------|-------------|
| `base.py` | Base rotary embedding class |
| `triton_kernels.py` | Triton-accelerated RoPE kernels |
| `mrope.py` | Multimodal RoPE (M-RoPE) for vision-language models |
| `mrope_rope_index.py` | RoPE index computation for M-RoPE |
| `yarn.py` | YaRN (Yet another RoPE extensioN) for length extrapolation |
| `rope_variant.py` | RoPE variant configurations |
| `utils.py` | Utility functions for RoPE computation |

### Standard RoPE

The standard RoPE applies rotation matrices to query and key vectors:

```
q_rotated = q * cos(theta) + rotate_half(q) * sin(theta)
k_rotated = k * cos(theta) + rotate_half(k) * sin(theta)
```

Where `theta` is the position-dependent angle.

### M-RoPE (Multimodal RoPE)

M-RoPE extends standard RoPE for multimodal models by:

- Using separate position indices for temporal, height, and width dimensions.
- Applying different rotation frequencies for each dimension.
- Supporting 1D (text), 2D (image), and 3D (video) inputs.

### YaRN

YaRN extends RoPE for length extrapolation:

- Scales the rotation frequencies for longer sequences.
- Supports dynamic frequency scaling based on target length.

### Configuration

```bash
# RoPE parameters (usually auto-detected from model config)
--rope-scaling {}  # JSON configuration for RoPE scaling
--rope-scaling-factor 1.0
```

---

## Layer Normalization

Normalization layers are in `python/sglang/srt/layers/layernorm.py`.

### RMSNorm

Root Mean Square Layer Normalization:

```
output = x * weight / sqrt(mean(x^2) + eps)
```

#### Fused Operations

| Operation | CUDA Kernel | Description |
|-----------|-------------|-------------|
| `rmsnorm` | sgl_kernel | Standard RMSNorm |
| `fused_add_rmsnorm` | sgl_kernel | Fused residual add + RMSNorm |
| `gemma_rmsnorm` | sgl_kernel | Gemma-style RMSNorm (different scaling) |
| `gemma_fused_add_rmsnorm` | sgl_kernel | Fused add + Gemma RMSNorm |

### LayerNorm

Standard Layer Normalization:

```
output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

### FlashInfer LayerNorm

When FlashInfer is available, an optimized LayerNorm from `flashinfer.norm` is used.

### AITER LayerNorm (AMD)

On AMD GPUs with AITER:

- `rmsnorm2d_fwd`: AITER RMSNorm
- `rmsnorm2d_fwd_with_add`: AITER fused add + RMSNorm
- `layernorm2d_fwd`: AITER LayerNorm

### Batch-Invariant Normalization

When deterministic inference is enabled (`--enable-deterministic-inference`):

- Uses `rms_norm_batch_invariant` from `batch_invariant_ops`.
- Ensures identical results regardless of batch size.

### Platform Support

| Platform | RMSNorm | LayerNorm | Fused Add+Norm |
|----------|---------|-----------|----------------|
| NVIDIA CUDA | sgl_kernel | FlashInfer | sgl_kernel |
| AMD ROCm | AITER | AITER | AITER |
| Intel XPU | sgl_kernel | sgl_kernel | sgl_kernel |
| Intel AMX (CPU) | sgl_kernel | sgl_kernel | sgl_kernel |
| NPU | torch_npu | torch_npu | torch_npu |

---

## Activation Functions

Activation functions are in `python/sglang/srt/layers/activation.py`.

### Available Activations

| Name | Class | Formula | Description |
|------|-------|---------|-------------|
| `silu` | `SiluAndMul` | `silu(gate) * up` | SwiGLU (fused gate + up projection) |
| `gelu` | `GeluAndMul` | `gelu(gate) * up` | GeLU gate |
| `gelu_pytorch_tanh` | `GeluAndMul` | `gelu_tanh(gate) * up` | Approximate GeLU |
| `gelu_new` | `NewGELU` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` | New GELU |
| `relu2` | `ReLU2` | `max(0, x)^2` | Squared ReLU |
| `xielu` | `XIELU` | Parametric activation | xIELU from https://arxiv.org/abs/2411.13010 |
| `quick_gelu` | `QuickGELU` | `x * sigmoid(1.702 * x)` | Quick GELU (CLIP) |

### Fused Gate-Up Activations

The most performance-critical activation is the fused gate-up projection:

#### SiluAndMul

```python
# Input: [gate_proj, up_proj] concatenated along last dim
# Output: silu(gate_proj) * up_proj
d = x.shape[-1] // 2
output = silu(x[..., :d]) * x[..., d:]
```

#### GeluAndMul

```python
# Input: [gate_proj, up_proj] concatenated
# Output: gelu(gate_proj) * up_proj
d = x.shape[-1] // 2
output = gelu(x[..., :d]) * up_proj  # x[..., d:]
```

### Platform-Specific Implementations

| Platform | SiluAndMul | GeluAndMul | QuickGELU |
|----------|------------|------------|-----------|
| CUDA | JIT kernel | sgl_kernel | Native |
| XPU | sgl_kernel | sgl_kernel | Native |
| ROCm | sgl_kernel | sgl_kernel | sgl_kernel |
| CPU (AMX) | sgl_kernel | sgl_kernel | Native |
| NPU | torch_npu | torch_npu | torch_npu |
| MUSA | nn.SwishGLU | N/A | Native |

### ScaledActivation

Some quantization methods (e.g., AWQ) require post-activation scaling:

```python
class ScaledActivation:
    def forward(self, x):
        return self.act(x) / self.scales
```

### Activation Registry

Activations are registered in `_ACTIVATION_REGISTRY`:

```python
_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "gelu_new": NewGELU(),
    "relu2": ReLU2(),
    "xielu": XIELU(),
}
```

Use `get_act_fn(name, quant_config, ...)` to retrieve activation functions by name.

---

## Quantization Layers

Quantization layers are in `python/sglang/srt/layers/quantization/`.

### Supported Quantization Methods

| Method | File | Description |
|--------|------|-------------|
| **FP8** | `fp8.py`, `fp8_utils.py` | Float-8 quantization (E4M3, E5M2) |
| **FP4** | `fp4_utils.py` | Float-4 quantization |
| **MXFP4** | `mxfp4.py`, `mxfp4_tensor.py` | Microscaling FP4 |
| **INT8** | `w8a8_int8.py`, `int8_utils.py` | Weight-only or weight-activation INT8 |
| **Block INT8** | `blockwise_int8.py` | Block-wise INT8 quantization |
| **GPTQ** | `gptq.py` | GPTQ post-training quantization |
| **GPTQ CPU** | `gptq_cpu.py` | GPTQ for CPU inference |
| **AWQ** | `awq/` | Activation-aware weight quantization |
| **Marlin** | `marlin_utils.py` | Marline sparse quantization |
| **BitsAndBytes** | `bitsandbytes.py` | BitsAndBytes 4/8-bit quantization |
| **GGUF** | `gguf.py` | GGUF format quantization |
| **Compressed Tensors** | `compressed_tensors/` | Neural Magic compressed format |
| **ModelOpt** | `modelopt_quant.py` | NVIDIA ModelOpt quantization |
| **KV Cache** | `kv_cache.py` | KV cache quantization |
| **FP4 KV Cache** | `fp4_kv_cache_quant_method.py` | FP4 quantized KV cache |
| **W4A FP8** | `w4afp8.py` | W4A with FP8 activation |
| **Petit** | `petit.py`, `petit_utils.py` | Petit quantization |
| **Quark** | `quark/` | Quark quantization framework |
| **QOQ** | `qoq.py` | QOQ quantization |
| **Unquantized** | `unquant.py` | No quantization (baseline) |

### FP8 Quantization

FP8 is the most commonly used quantization in SGLang:

```bash
# Enable FP8 quantization
python -m sglang.launch_server \
    --quantization fp8 \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

#### FP8 Configuration

- **E4M3**: 4 exponent bits, 3 mantissa bits (for weights and activations)
- **E5M2**: 5 exponent bits, 2 mantissa bits (for gradients)
- **FNUZ**: Alternative FP8 format (Flush Non-zero to Zero)

### KV Cache Quantization

KV cache can be quantized to reduce memory usage:

| Format | Description |
|--------|-------------|
| `fp8_e4m3` | FP8 E4M3 KV cache |
| `fp8_e5m2` | FP8 E5M2 KV cache |
| `fp4` | FP4 KV cache |

```bash
# Enable FP8 KV cache
python -m sglang.launch_server \
    --kv-cache-dtype fp8_e4m3
```

### Quantization Base Classes

```python
class QuantizationConfig:
    # Base configuration for quantization methods
    def get_scaled_act_names(self) -> Set[str]
    def get_quant_method(self, layer) -> QuantizeMethodBase

class QuantizeMethodBase:
    # Base class for quantization implementations
    def process_weights_after_loading(self, layer)
    def apply(self, layer, *args, **kwargs)
```

---

## LoRA Layers

LoRA (Low-Rank Adaptation) support enables serving multiple fine-tuned models from a single base model.

### LoRA Components

| Component | Description |
|-----------|-------------|
| `LoRADrainer` | Manages LoRA adapter loading and unloading |
| `LoRAOverlapLoader` | Overlaps LoRA weight loading with computation |
| LoRA weight merging | Supports merging LoRA weights into base weights |

### LoRA Configuration

```bash
# Enable LoRA
python -m sglang.launch_server \
    --enable-lora \
    --max-loras-per-batch 4 \
    --lora-target-modules q_proj v_proj \
    --max-lora-rank 64
```

### LoRA Adapter Management

```bash
# Load a LoRA adapter
curl -X POST http://localhost:30000/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "my_adapter", "lora_path": "/path/to/adapter"}'

# Unload a LoRA adapter
curl -X POST http://localhost:30000/unload_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "my_adapter"}'
```

### LoRA Pool Metrics

- `sglang:lora_pool_slots_used`: Number of active LoRA slots
- `sglang:lora_pool_slots_total`: Total available LoRA slots
- `sglang:lora_pool_utilization`: LoRA pool utilization ratio

---

## Multimodal Encoder Layers

Multimodal support is handled through several layers:

### Vision Encoders

Located in `python/sglang/srt/layers/attention/vision.py`:

- **Vision attention**: Specialized attention for vision transformers.
- **Vision utils**: Helper functions for image preprocessing.

### Multimodal Processing

The multimodal processor (`python/sglang/srt/managers/multimodal_processor.py`) handles:

- **Image processing**: Resize, normalize, and convert images to pixel values.
- **Audio processing**: Convert audio inputs to spectrograms or embeddings.
- **Video processing**: Frame extraction and encoding.

### Supported Multimodal Models

SGLang supports a wide range of multimodal models:

| Model Type | Architecture | Vision Encoder |
|------------|-------------|----------------|
| LLaVA | CLIP + LLM | CLIP ViT |
| Qwen2-VL | Qwen2 + ViT | Custom ViT |
| Gemma3-MM | Gemma3 + SigLIP | SigLIP |
| InternVL | InternLM2 + ViT | InternViT |
| MiniCPM-V | MiniCPM + ViT | Custom ViT |
| DeepSeek-VL2 | DeepSeek + SigLIP | SigLIP2 |
| Phi-4-MM | Phi-4 + ViT | Custom ViT |
| Kimi-VL | Kimi + MoonViT | MoonViT |
| Llama-4 | Llama-4 + Vision | Custom Vision |
| Pixtral | Mistral + ViT | Custom ViT |

### Multimodal Cache

The multimodal cache (`python/sglang/srt/mem_cache/multimodal_cache.py`) stores:

- Processed image embeddings.
- Audio features.
- Video frame embeddings.

This avoids reprocessing the same multimodal inputs across requests.

---

## Mamba and Linear Attention Layers

### Mamba / SSM Layers

Located in `python/sglang/srt/layers/attention/mamba/`:

- **Selective state space models**: Mamba SSM implementation.
- **Linear attention**: Attention with linear complexity.
- **Hybrid models**: Models combining Mamba and attention layers (e.g., Jamba).

#### Mamba State Management

- `MambaPool`: Manages Mamba SSM state storage.
- `HiMambaRadixCache`: Hierarchical cache for Mamba states.
- `MambaRadixCache`: Radix tree cache adapted for Mamba states.

### FLA (Flash Linear Attention)

Located in `python/sglang/srt/layers/attention/fla/`:

- Flash Linear Attention kernels.
- Chunk delta computation with configurable chunk size (`FLA_CHUNK_SIZE`).

### Hybrid Linear Attention

The `hybrid_linear_attn_backend.py` implements:

- Switching between full attention and linear attention based on layer configuration.
- Support for models that mix attention types across layers.

### Radix Linear Attention

`radix_linear_attention.py` provides radix tree caching for linear attention models, similar to how RadixAttention works for standard attention.

---

## Model Architecture Registry

SGLang supports a vast number of model architectures through its model registry (`python/sglang/srt/models/registry.py`).

### Supported Model Families

| Family | Models | Key Features |
|--------|--------|--------------|
| **Llama** | Llama, Llama-2, Llama-3, Llama-3.1, Llama-3.2, Llama-4 | GQA, RoPE, SwiGLU |
| **Qwen** | Qwen, Qwen2, Qwen2.5, Qwen3, Qwen3-VL | GQA, MLA, MoE |
| **DeepSeek** | DeepSeek, DeepSeek-V2, DeepSeek-V3 | MLA, MoE, NSA |
| **Gemma** | Gemma, Gemma2, Gemma3, Gemma4 | SWA, GQA |
| **Mistral** | Mistral, Mixtral, Mistral-Large-3 | SWA, MoE |
| **GLM** | GLM-4, GLM-4-MoE, GLM-4v | MoE, Vision |
| **Phi** | Phi, Phi3, Phi-4-MM | GQA |
| **MoE Models** | Mixtral, DBRX, Qwen-MoE, ExaOne-MoE | Expert parallelism |
| **Vision Models** | LLaVA, Qwen-VL, InternVL, Pixtral | Multimodal |
| **Reward Models** | Llama-Reward, Qwen-RM, InternLM-Reward | Reward scoring |
| **Eagle Speculative** | Llama-Eagle, Qwen-Eagle, Mistral-Eagle | Speculative decoding |
| **Mamba Models** | Falcon-H1, GraniteMoE-Hybrid | SSM + Attention hybrid |
| **Diffusion Models** | (via SGLang-Diffusion) | WAN, FLUX, Qwen-Image |

### Model Implementation Pattern

Each model file follows a standard pattern:

```python
class ModelName(nn.Module):
    def __init__(self, config, quant_config, ...):
        # Initialize embeddings, layers, norm, head
        pass

    def forward(self, input_ids, positions, forward_batch):
        # Standard forward pass
        pass

    def get_input_embeddings(self):
        return self.embed_tokens

    # Optional: load_weights for custom weight loading
```

### Weight Loading

Models support multiple weight loading formats:

- **Safetensors**: Primary format (recommended).
- **PyTorch bin**: Legacy format.
- **GGUF**: Quantized format.
- **Dummy**: Random weights for benchmarking.

### Model Override Arguments

```bash
# Override model architecture for testing
python -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --json-model-override-args '{"num_hidden_layers": 2}'
```

---

## Cross-Platform Support

SGLang's layer implementations support multiple hardware platforms through the `MultiPlatformOp` abstraction.

### Platform Detection

```python
is_cuda()     # NVIDIA CUDA
is_hip()      # AMD ROCm
is_xpu()      # Intel XPU
is_cpu()      # CPU (with AMX support check)
is_npu()      # Ascend NPU
is_musa()     # Moore Threads MUSA
is_mps()      # Apple Silicon (MLX)
```

### MultiPlatformOp Pattern

```python
class MyOp(MultiPlatformOp):
    def forward_cuda(self, x): ...   # NVIDIA implementation
    def forward_hip(self, x): ...    # AMD implementation
    def forward_xpu(self, x): ...    # Intel implementation
    def forward_cpu(self, x): ...    # CPU implementation
    def forward_npu(self, x): ...    # Ascend implementation
    def forward_native(self, x): ... # Fallback PyTorch
```

### Kernel Libraries

| Platform | Kernel Library |
|----------|---------------|
| NVIDIA | sgl_kernel, FlashInfer, CUTLASS |
| AMD | sgl_kernel, AITER |
| Intel XPU | sgl_kernel |
| Intel CPU | sgl_kernel (AMX) |
| Ascend NPU | torch_npu |
| Apple Silicon | MLX |

---

*This reference covers the layer and operation implementations in SGLang. For model-specific details, consult the individual model files in `python/sglang/srt/models/`. For quantization configuration, see the quantization documentation.*
