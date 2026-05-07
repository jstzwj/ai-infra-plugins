# Chapter 35: Example Scripts

## Source Files
- `sources/Megatron-LM/examples/` - Training examples
- `sources/Megatron-LM/examples/llama/` - LLaMA training
- `sources/Megatron-LM/examples/multimodal/` - Multimodal training
- `sources/Megatron-LM/examples/mamba/` - Mamba training
- `sources/Megatron-LM/examples/gptoss/` - GPT-OSS training
- `sources/Megatron-LM/examples/rl/` - RL training

## Overview

Megatron-LM ships with ready-to-use example scripts demonstrating various training configurations. These scripts serve as starting points for production training.

## LLaMA Training

### LLaMA-3 8B on H100 with FP8
```bash
# File: examples/llama/train_llama3_8b_h100_fp8.sh
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --ffn-hidden-size 14336 \
    --normalization RMSNorm \
    --swiglu \
    --position-embedding-type rope \
    --fp8-format e4m3 \
    --fp8-recipe delayed \
    --fp8-param-gather \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --train-iters 100000 \
    --bf16 \
    --use-distributed-optimizer \
    --sequence-parallel
```

### LLaMA-3 70B Configuration
```bash
torchrun --nproc_per_node=8 --nnodes=8 pretrain_gpt.py \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --ffn-hidden-size 28672 \
    --normalization RMSNorm \
    --swiglu \
    --position-embedding-type rope \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --context-parallel-size 2 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --bf16
```

### LLaMA-3.1 405B Configuration
```bash
torchrun --nproc_per_node=8 --nnodes=128 pretrain_gpt.py \
    --num-layers 126 \
    --hidden-size 16384 \
    --num-attention-heads 128 \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 131072 \
    --ffn-hidden-size 53248 \
    --normalization RMSNorm \
    --swiglu \
    --position-embedding-type rope \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --context-parallel-size 2 \
    --expert-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 2048 \
    --bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --sequence-parallel \
    --recompute-granularity selective \
    --recompute-method uniform
```

## Mixtral MoE Training

### Mixtral 8x7B
```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --seq-length 4096 \
    --ffn-hidden-size 14336 \
    --normalization RMSNorm \
    --swiglu \
    --position-embedding-type rope \
    --num-experts 8 \
    --expert-model-parallel-size 8 \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 0.01 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 1 \
    --global-batch-size 512 \
    --bf16
```

### DeepSeek-V3 671B Configuration
```bash
torchrun --nproc_per_node=8 --nnodes=128 pretrain_gpt.py \
    --num-layers 61 \
    --hidden-size 7168 \
    --num-attention-heads 128 \
    --num-query-groups 128 \
    --seq-length 4096 \
    --ffn-hidden-size 18432 \
    --normalization RMSNorm \
    --swiglu \
    --position-embedding-type rope \
    --multi-latent-attention \
    --num-experts 256 \
    --expert-model-parallel-size 64 \
    --moe-router-topk 8 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-grouped-gemm \
    --moe-shared-expert-intermediate-size 18432 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 16 \
    --micro-batch-size 1 \
    --global-batch-size 4096 \
    --bf16
```

## GPT Training

### GPT-3 175B Standard Configuration
```bash
torchrun --nproc_per_node=8 --nnodes=16 pretrain_gpt.py \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --ffn-hidden-size 49152 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --lr 6e-5 \
    --min-lr 6e-6 \
    --lr-decay-style cosine \
    --bf16 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --recompute-granularity selective
```

## BERT Training

```bash
torchrun --nproc_per_node=8 pretrain_bert.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --ffn-hidden-size 4096 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --lr 1e-4 \
    --bf16
```

## T5 Training

```bash
torchrun --nproc_per_node=8 pretrain_t5.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 512 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --ffn-hidden-size 4096 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --bf16
```

## Mamba Training

```bash
torchrun --nproc_per_node=8 pretrain_hybrid.py \
    --num-layers 64 \
    --hidden-size 5120 \
    --seq-length 4096 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --bf16 \
    --mamba-state-dim 128 \
    --mamba-head-dim 64 \
    --mamba-num-groups 8
```

## Multimodal Training (VLM)

```bash
torchrun --nproc_per_node=8 pretrain_vlm.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --bf16 \
    --use-vision-transformer \
    --vision-model-path /path/to/vit \
    --image-size 336 \
    --patch-size 14
```

## RL Training (GRPO)

```bash
torchrun --nproc_per_node=8 train_rl.py \
    --tensor-model-parallel-size 4 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --bf16 \
    --rl-algorithm grpo \
    --rl-kl-coeff 0.1 \
    --rl-entropy-coeff 0.01 \
    --rl-advantage-clip 0.2
```

## Minimal Megatron Core Example

```python
# examples/run_simple_mcore_train_loop.py
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt import GPTModel

# Initialize
parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)

config = TransformerConfig(
    num_layers=2, hidden_size=128, num_attention_heads=4,
    seq_length=128, bf16=True, tensor_model_parallel_size=2,
)

model = GPTModel(config)
# Run training loop...
```
