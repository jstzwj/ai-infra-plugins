---
name: Megatron-LM
description: NVIDIA Megatron-LM & Megatron Core - GPU-optimized framework for training large language models with tensor parallelism, pipeline parallelism, data parallelism (DDP/FSDP), context parallelism, expert parallelism, FP8/FP4 quantization, CUDA graphs, MoE (Mixture of Experts), multimodal models, and TensorRT-LLM export. Supports GPT, BERT, T5, Mamba, LLaMA, Mixtral, DeepSeek-V3, Qwen3, and custom architectures from 2B to 462B parameters with up to 47% MFU on H100 GPUs.
version: 25.07
---

# Megatron-LM Reference Manual

NVIDIA Megatron-LM is a production-grade, open-source framework for training large transformer models at scale. It provides both a composable library (Megatron Core) and reference training scripts for pretraining, fine-tuning, and inference of models ranging from millions to hundreds of billions of parameters.

## How to Use

- **Architecture & Setup Questions**: Start with [01-overview-and-architecture.md](references/01-overview-and-architecture.md) and [02-installation-and-setup.md](references/02-installation-and-setup.md)
- **Configuration**: [03-model-parallel-config.md](references/03-model-parallel-config.md) and [04-transformer-config.md](references/04-transformer-config.md) cover all 200+ configuration parameters
- **Parallelism Strategy**: [07-tensor-parallelism.md](references/07-tensor-parallelism.md) through [12-distributed-checkpointing.md](references/12-distributed-checkpointing.md)
- **Model Implementation**: [13-gpt-model.md](references/13-gpt-model.md) through [20-moe-architecture.md](references/20-moe-architecture.md)
- **Training & Optimization**: [21-optimizer-and-training.md](references/21-optimizer-and-training.md) through [28-post-training.md](references/28-post-training.md)
- **Inference & Export**: [29-inference-engine.md](references/29-inference-engine.md) through [34-profiling-and-debugging.md](references/34-profiling-and-debugging.md)
- **Examples & Deployment**: [35-example-scripts.md](references/35-example-scripts.md) through [40-troubleshooting-faq.md](references/40-troubleshooting-faq.md)

## Quick Reference

### Basic Training Launch
```bash
# GPT pretraining with tensor parallelism (TP=2) and pipeline parallelism (PP=2)
torchrun --nproc_per_node=4 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --train-iters 100000 \
    --lr 1e-4 \
    --bf16
```

### Megatron Core Usage
```python
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig, TransformerBlock
from megatron.core.models.gpt import GPTModel

# Initialize parallel groups
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
)

# Configure model
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    seq_length=4096,
    bf16=True,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
)

# Build model
model = GPTModel(config=config, ...)
```

### Key Configuration Patterns
```python
# FP8 training
config = TransformerConfig(fp8='e4m3', fp8_recipe='delayed', fp8_param=True, ...)

# MoE (Mixture of Experts)
config = TransformerConfig(
    num_moe_experts=8,
    moe_router_topk=2,
    expert_model_parallel_size=4,
    moe_grouped_gemm=True,
    ...
)

# Sequence Parallelism + Context Parallelism
config = TransformerConfig(
    sequence_parallel=True,
    context_parallel_size=4,
    tp_comm_overlap=True,
    ...
)
```

## Key Concepts

- **Tensor Parallelism (TP)**: Splits individual weight tensors across GPUs; communication-heavy but reduces per-GPU memory
- **Pipeline Parallelism (PP)**: Splits layers across GPUs; reduces communication but introduces pipeline bubbles
- **Data Parallelism (DP)**: Replicates model across GPUs with gradient synchronization; includes DDP and FSDP variants
- **Context Parallelism (CP)**: Splits long sequences across GPUs for training with 8K+ token sequences
- **Expert Parallelism (EP)**: Distributes MoE experts across GPUs for Mixture-of-Experts models
- **Sequence Parallelism**: Makes TP more memory-efficient by parallelizing LayerNorm and dropout
- **FP8/FP4 Quantization**: Reduces memory and improves throughput via TransformerEngine
- **CUDA Graphs**: Captures GPU operations for reduced kernel launch overhead
- **Activation Recomputation**: Trades compute for memory by selectively recomputing activations

## Documentation Map

### Part I: Core Architecture
1. [Overview and Architecture](references/01-overview-and-architecture.md) - System architecture, Megatron Core vs Megatron-LM, component overview
2. [Installation and Setup](references/02-installation-and-setup.md) - PyPI, source, NGC container, dependencies, Docker
3. [Model Parallel Config](references/03-model-parallel-config.md) - `ModelParallelConfig` dataclass with all parallelism, training, and optimization parameters
4. [Transformer Config](references/04-transformer-config.md) - `TransformerConfig` and `MLATransformerConfig` with model architecture, FP8, MoE, CUDA graph parameters
5. [Transformer Building Blocks](references/05-transformer-building-blocks.md) - TransformerLayer, TransformerBlock, MLP, custom layers
6. [Attention Mechanisms](references/06-attention-mechanisms.md) - Multi-head attention, GQA, MLA, flash attention, RoPE, sliding window

### Part II: Parallelism Strategies
7. [Tensor Parallelism](references/07-tensor-parallelism.md) - Column/Row parallel linear layers, sequence parallelism, communication overlap
8. [Pipeline Parallelism](references/08-pipeline-parallelism.md) - 1F1B schedule, interleaved PP, virtual pipeline stages, p2p communication
9. [Data Parallelism](references/09-data-parallelism.md) - DDP, Megatron-FSDP, distributed optimizer (ZeRO-1/2/3), gradient synchronization
10. [Context Parallelism](references/10-context-parallelism.md) - Ring attention, all-gather CP, hybrid CP, variable-length sequences
11. [Expert Parallelism](references/11-expert-parallelism.md) - MoE expert distribution, token dispatching (allgather/alltoall/flex), DeepEP
12. [Distributed Checkpointing](references/12-distributed-checkpointing.md) - Checkpoint save/load, distributed checkpoint format, rescaling

### Part III: Model Implementations
13. [GPT Model](references/13-gpt-model.md) - GPT/GPT-2/GPT-3 architecture, embedding, language model, forward pass
14. [BERT Model](references/14-bert-model.md) - BERT encoder, MLM pretraining, classification heads
15. [T5 Model](references/15-t5-model.md) - Encoder-decoder architecture, span corruption pretraining
16. [Mamba Model](references/16-mamba-model.md) - State space models, hybrid SSM-Transformer architecture
17. [Multimodal Models](references/17-multimodal-models.md) - Vision-language models, CLIP, ViT integration, image/video encoders
18. [Vision Models](references/18-vision-models.md) - ViT, Radio, image classification backbones
19. [Hybrid and MIMO Models](references/19-hybrid-mimo-models.md) - Hybrid SSM-attention, multi-input multi-output architectures
20. [MoE Architecture](references/20-moe-architecture.md) - Router, token dispatcher, grouped MLP, shared experts, load balancing

### Part IV: Training and Optimization
21. [Optimizer and Training Loop](references/21-optimizer-and-training.md) - AdamW, distributed optimizer, CPU offloading, gradient accumulation fusion
22. [FP8 and Quantization](references/22-fp8-and-quantization.md) - FP8 formats (e4m3/hybrid), FP4, recipes (delayed/MX/blockwise), quantization config
23. [CUDA Graphs](references/23-cuda-graphs.md) - Graph capture, partial/full iteration graphs, TE integration, warmup
24. [Activation Checkpointing](references/24-activation-checkpointing.md) - Full/selective recompute, recompute modules, uniform/block methods
25. [Data Loading and Datasets](references/25-data-loading-and-datasets.md) - JSONL format, blending, Megatron Energon multimodal data, data preprocessing
26. [Tokenizers](references/26-tokenizers.md) - BPE, SentencePiece, HuggingFace tokenizer integration, text and vision tokenizers
27. [RL Training](references/27-rl-training.md) - Reinforcement learning from human feedback, GRPO, PPO, reward models
28. [Post-Training](references/28-post-training.md) - Model optimization, quantization-aware training, distillation

### Part V: Inference and Export
29. [Inference Engine](references/29-inference-engine.md) - Model inference wrappers, optimized inference layers, batched generation
30. [Text Generation Server](references/30-text-generation-server.md) - HTTP inference server, sampling strategies, streaming generation
31. [TensorRT-LLM Export](references/31-tensorrt-llm-export.md) - Weight conversion, engine building, optimized deployment
32. [Megatron Bridge](references/32-megatron-bridge.md) - HuggingFace checkpoint conversion, bidirectional model porting
33. [Elastic Training](references/33-elastic-training.md) - Flextron config, memory management, elastic scaling
34. [Profiling and Debugging](references/34-profiling-and-debugging.md) - Nsight integration, memory profiling, performance analysis

### Part VI: Examples and Configuration
35. [Example Scripts](references/35-example-scripts.md) - GPT, LLaMA, Mixtral, BERT, T5, Mamba training examples
36. [Docker and Deployment](references/36-docker-and-deployment.md) - Dockerfiles, NGC containers, SLURM integration, multi-node setup
37. [Testing Framework](references/37-testing-framework.md) - Unit tests, integration tests, CI/CD pipeline, test writing guide
38. [CLI Arguments Reference](references/38-cli-arguments-reference.md) - Complete command-line arguments catalog (200+ flags)
39. [Performance Tuning Guide](references/39-performance-tuning-guide.md) - MFU optimization, communication overlap, memory tuning, scaling strategies
40. [Troubleshooting FAQ](references/40-troubleshooting-faq.md) - Common errors, OOM debugging, hang diagnosis, numerical issues

## Source Files

- `sources/Megatron-LM/megatron/core/` - Megatron Core library
- `sources/Megatron-LM/megatron/core/transformer/` - Transformer building blocks
- `sources/Megatron-LM/megatron/core/models/` - Model implementations (GPT, BERT, T5, Mamba, etc.)
- `sources/Megatron-LM/megatron/core/tensor_parallel/` - Tensor parallelism
- `sources/Megatron-LM/megatron/core/pipeline_parallel/` - Pipeline parallelism
- `sources/Megatron-LM/megatron/core/distributed/` - Distributed training (DDP, FSDP)
- `sources/Megatron-LM/megatron/core/optimizer/` - Optimizers
- `sources/Megatron-LM/megatron/core/inference/` - Inference engine
- `sources/Megatron-LM/megatron/core/quantization/` - Quantization (FP8, FP4)
- `sources/Megatron-LM/examples/` - Training examples
- `sources/Megatron-LM/docs/` - Documentation
