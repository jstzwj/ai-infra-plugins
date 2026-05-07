# Chapter 01: Overview and Architecture

## Source Files
- `sources/Megatron-LM/megatron/core/` - Megatron Core library
- `sources/Megatron-LM/megatron/` - Megatron-LM framework
- `sources/Megatron-LM/docs/` - Documentation

## System Architecture

Megatron-LM consists of two main components:

### 1. Megatron Core (`megatron/core/`)
A composable, modular Python library providing GPU-optimized building blocks for constructing custom training frameworks. Designed for framework developers and ML engineers who need fine-grained control.

**Key modules:**
```
megatron/core/
├── model_parallel_config.py        # Base parallelism configuration
├── transformer/
│   ├── transformer_config.py       # Transformer architecture config
│   ├── transformer_layer.py        # Core transformer layer
│   ├── transformer_block.py        # Multi-layer transformer block
│   ├── attention.py                # Multi-head attention
│   ├── mlp.py                      # MLP layers
│   ├── dot_product_attention.py    # Optimized attention
│   └── custom_layers/              # CUDA-optimized layers
├── tensor_parallel/                # Tensor parallelism
├── pipeline_parallel/              # Pipeline parallelism
├── distributed/                    # DDP, FSDP
│   └── fsdp/                       # Fully Sharded Data Parallel
├── models/
│   ├── gpt/                        # GPT model
│   ├── bert/                       # BERT model
│   ├── T5/                         # T5 model
│   ├── mamba/                      # Mamba SSM
│   ├── multimodal/                 # Vision-language models
│   ├── vision/                     # Vision models
│   ├── hybrid/                     # Hybrid architectures
│   └── mimo/                       # Multi-input multi-output
├── optimizer/                      # Optimizers with CPU offloading
├── quantization/                   # FP8, FP4 quantization
├── inference/                      # Inference engine
├── export/trtllm/                  # TensorRT-LLM export
├── tokenizers/                     # Text and vision tokenizers
├── datasets/                       # Dataset loaders
├── post_training/                  # Post-training optimization
├── cuda_graphs.py                  # CUDA graph support
├── fp8_utils.py                    # FP8 utilities
└── parallel_state.py               # Parallel process groups
```

### 2. Megatron-LM (Reference Implementation)
Pre-configured training scripts and examples built on top of Megatron Core. Designed for research teams and production deployments.

**Key directories:**
```
Megatron-LM/
├── pretrain_gpt.py                 # GPT pretraining entry point
├── pretrain_bert.py                # BERT pretraining
├── pretrain_t5.py                  # T5 pretraining
├── pretrain_vlm.py                 # Vision-language model pretraining
├── pretrain_mamba.py               # Mamba pretraining
├── pretrain_hybrid.py              # Hybrid model pretraining
├── train_rl.py                     # RL training
├── megatron/
│   ├── training/                   # Training scripts and utilities
│   ├── legacy/                     # Legacy components
│   ├── rl/                         # Reinforcement learning
│   └── elastification/             # Elastic training
├── examples/                       # Ready-to-use examples
│   ├── gptoss/                     # GPT-OSS training
│   ├── multimodal/                 # Multimodal training
│   ├── mamba/                      # Mamba examples
│   ├── llama/                      # LLaMA training
│   ├── mixtral/                    # Mixtral MoE training
│   └── rl/                         # RL examples
├── tools/                          # Development tools
├── scripts/                        # Utility scripts
├── tasks/                          # Task definitions
├── tests/                          # Test suite
├── docs/                           # Documentation
└── docker/                         # Docker configurations
```

## Parallelism Hierarchy

Megatron-LM supports five complementary parallelism strategies that can be combined:

```
Total GPUs = TP × PP = DP × CP × EP

Where:
  TP = Tensor Model Parallelism (intra-layer)
  PP = Pipeline Model Parallelism (inter-layer)
  DP = Data Parallelism (gradient sync)
  CP = Context Parallelism (sequence splitting)
  EP = Expert Parallelism (MoE expert distribution)
```

### Process Group Layout
```
┌─────────────────────────────────────────────────┐
│              WORLD_GROUP (all GPUs)              │
│  ┌──────────────────────────────────────────┐   │
│  │       DATA_PARALLEL_GROUP                │   │
│  │  ┌──────────────────────────────────┐    │   │
│  │  │   CONTEXT_PARALLEL_GROUP         │    │   │
│  │  │  ┌──────────────────────────┐    │    │   │
│  │  │  │ PIPELINE_MODEL_PARALLEL  │    │   │
│  │  │  │ GROUP                    │    │   │
│  │  │  │  ┌────────────────────┐  │    │   │
│  │  │  │  │TENSOR_MODEL_PARALLEL│  │    │   │
│  │  │  │  │GROUP               │  │    │   │
│  │  │  │  └────────────────────┘  │    │   │
│  │  │  └──────────────────────────┘    │    │   │
│  │  └──────────────────────────────────┘    │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Composability**: Megatron Core components can be used independently or combined
2. **Scalability**: Proven to scale to thousands of GPUs training 462B+ parameter models
3. **Performance**: Up to 47% MFU (Model FLOP Utilization) on H100 GPUs
4. **Flexibility**: Supports decoder-only, encoder-only, and encoder-decoder architectures
5. **Production-Ready**: Includes TensorRT-LLM export, HTTP inference servers, and Docker images

## Supported Model Architectures

| Architecture | Type | Models | Max Size |
|---|---|---|---|
| GPT | Decoder-only | GPT-2, GPT-3, LLaMA, Mistral, Qwen | 462B+ |
| BERT | Encoder-only | BERT, RoBERTa | ~1B |
| T5 | Encoder-Decoder | T5, mT5 | ~11B |
| Mamba | SSM | Mamba, Hybrid SSM-Transformer | ~8B |
| MoE | Mixture of Experts | Mixtral, DeepSeek-V3, Qwen3-MoE | 462B+ |
| Multimodal | Vision-Language | CLIP+Mistral, VLM | Various |
| Hybrid | SSM+Attention | Hybrid Mamba-Transformer | Various |

## Performance Benchmarks

### H100 GPU Scaling (GPT-3 175B)
| GPUs | TP | PP | DP | Throughput (tokens/s) | MFU |
|---|---|---|---|---|---|
| 64 | 8 | 1 | 8 | ~52K | 42% |
| 128 | 8 | 2 | 8 | ~100K | 44% |
| 256 | 8 | 4 | 8 | ~190K | 45% |
| 512 | 8 | 8 | 8 | ~370K | 46% |
| 1024 | 8 | 8 | 16 | ~710K | 47% |

### Communication Patterns
- **TP (Tensor Parallel)**: AllReduce within node (NVLink) - ~2× per layer forward, ~2× backward
- **PP (Pipeline Parallel)**: P2P between stages (NVLink/IB) - 1 send + 1 recv per microbatch
- **DP (Data Parallel)**: AllReduce across DP group (IB) - 1× per training step
- **CP (Context Parallel)**: P2P or AllGather within CP group - depends on strategy

## Software Stack Dependencies

```
Megatron-LM
├── PyTorch >= 2.1
├── TransformerEngine >= 1.4 (for FP8, fused attention)
├── NVIDIA Apex (for fused kernels)
├── NCCL >= 2.18 (for GPU communication)
├── CUDA >= 11.8
├── cuDNN >= 8.9
├── FlashAttention >= 2.0 (optional, for faster attention)
├── FlashInfer (optional, for MoE inference)
├── grouped_gemm (optional, for MoE grouped GEMM)
├── TensorRT-LLM (optional, for export)
├── Megatron Energon (optional, for multimodal data)
└── NVIDIA Docker (optional, for containers)
```
