# Chapter 32: Megatron Bridge

## Source Files
- `sources/Megatron-LM/docs/llama_mistral.md` - Model conversion guide
- `sources/Megatron-LM/tools/` - Conversion utilities

## Overview

Megatron Bridge enables bidirectional conversion between HuggingFace and Megatron-LM checkpoint formats. This allows leveraging HuggingFace pretrained models in Megatron-LM training and exporting Megatron-trained models back to the HuggingFace ecosystem.

## Supported Models

| Model | HuggingFace → Megatron | Megatron → HuggingFace |
|---|---|---|
| GPT-2 | Yes | Yes |
| GPT-NeoX | Yes | Yes |
| LLaMA / LLaMA-2 / LLaMA-3 | Yes | Yes |
| Mistral | Yes | Yes |
| Mixtral (MoE) | Yes | Yes |
| BERT | Yes | Yes |
| T5 | Yes | Yes |
| Qwen | Yes | Yes |
| DeepSeek-V3 | Yes | Yes |

## Conversion: HuggingFace → Megatron

### Step 1: Download HuggingFace Model
```bash
# Using huggingface-cli
huggingface-cli download meta-llama/Meta-Llama-3-8B \
    --local-dir /path/to/llama3-8b-hf

# Or use from_pretrained in Python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

### Step 2: Convert to Megatron Format
```bash
python tools/checkpoint/loader_llama_hf.py \
    --load /path/to/llama3-8b-hf \
    --save /path/to/llama3-8b-megatron \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --model-type llama
```

### Conversion Options
| Parameter | Description |
|---|---|
| `--load` | HuggingFace model directory |
| `--save` | Output Megatron checkpoint directory |
| `--target-tensor-parallel-size` | TP degree for Megatron checkpoint |
| `--target-pipeline-parallel-size` | PP degree for Megatron checkpoint |
| `--model-type` | Model architecture (llama, gpt, bert, t5) |
| `--dtype` | Target precision (fp32, fp16, bf16) |

### Step 3: Train with Megatron-LM
```bash
torchrun --nproc_per_node=4 pretrain_gpt.py \
    --load /path/to/llama3-8b-megatron \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --bf16 \
    [additional training args...]
```

## Conversion: Megatron → HuggingFace

```bash
python tools/checkpoint/saver_llama_hf.py \
    --load /path/to/megatron/checkpoint \
    --save /path/to/output-hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --model-type llama
```

## Checkpoint Rescaling

Change parallelism configuration of an existing checkpoint:

```bash
# Rescale from TP=8 to TP=4
python tools/checkpoint/util.py \
    --load /path/to/checkpoint-tp8 \
    --save /path/to/checkpoint-tp4 \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --model-type GPT
```

### Supported Rescaling Operations

| Operation | Command |
|---|---|
| Change TP | `--target-tensor-parallel-size NEW_TP` |
| Change PP | `--target-pipeline-parallel-size NEW_PP` |
| Change DP | Handled by training framework |
| Merge TP shards | Set `--target-tensor-parallel-size 1` |
| Split for more TP | Set `--target-tensor-parallel-size N` |

## LLaMA-Specific Configuration

### LLaMA-3 8B
```bash
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
    --bf16 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1
```

### LLaMA-3 70B
```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
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
    --bf16 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4
```

## Model Architecture Mapping

| HuggingFace Config | Megatron Argument |
|---|---|
| `hidden_size` | `--hidden-size` |
| `num_hidden_layers` | `--num-layers` |
| `num_attention_heads` | `--num-attention-heads` |
| `num_key_value_heads` | `--num-query-groups` |
| `intermediate_size` | `--ffn-hidden-size` |
| `max_position_embeddings` | `--max-position-embeddings` |
| `rms_norm_eps` | `--norm-epsilon` |
| `rope_theta` | `--rotary-base` |
| `hidden_act` (silu+gate) | `--swiglu` |
| `layer_norm_epsilon` | `--norm-epsilon` |
| `layer_norm_type` | `--normalization` (RMSNorm/LayerNorm) |

## Common Issues

### Vocabulary Size Mismatch
Ensure `--tokenizer-model` matches the HuggingFace tokenizer. The vocabulary size must be consistent.

### Position Embedding Mismatch
LLaMA uses RoPE - ensure `--position-embedding-type rope` is set.

### Query Group Configuration
GQA models require `--num-query-groups` to match the HuggingFace `num_key_value_heads`.
