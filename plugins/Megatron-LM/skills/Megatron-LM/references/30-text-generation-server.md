# Chapter 30: Text Generation Server

## Source Files
- `sources/Megatron-LM/megatron/core/text_generation_server.py` - HTTP server
- `sources/Megatron-LM/tools/run_text_generation_server.py` - Server launcher
- `sources/Megatron-LM/tools/text_generation_cli.py` - CLI client
- `sources/Megatron-LM/tools/run_dynamic_text_generation_server.py` - Dynamic batching
- `sources/Megatron-LM/tools/run_vlm_text_generation.py` - VLM generation

## Overview

Megatron-LM includes an HTTP-based text generation server that supports both single-request and dynamic batched generation. The server wraps the inference engine and provides a REST API for integration with applications.

## Server Architecture

```
┌────────────┐     HTTP     ┌──────────────────┐
│   Client    │ ──────────► │ Generation Server │
│  (CLI/API)  │ ◄────────── │                  │
└────────────┘              │ ┌──────────────┐ │
                            │ │ Inference    │ │
                            │ │ Engine       │ │
                            │ └──────────────┘ │
                            │ ┌──────────────┐ │
                            │ │ Sampling     │ │
                            │ │ Module       │ │
                            │ └──────────────┘ │
                            │ ┌──────────────┐ │
                            │ │ KV Cache     │ │
                            │ │ Manager      │ │
                            │ └──────────────┘ │
                            └──────────────────┘
```

## Launching the Server

### Basic Server
```bash
torchrun --nproc_per_node=4 tools/run_text_generation_server.py \
    --load /path/to/checkpoint \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --seed 42
```

### Dynamic Batched Server
```bash
torchrun --nproc_per_node=4 tools/run_dynamic_text_generation_server.py \
    --load /path/to/checkpoint \
    --tensor-model-parallel-size 4 \
    --max-batch-size 32 \
    --max-tokens 512
```

### Multi-Node Server
```bash
# Master node
torchrun --nproc_per_node=8 --nnodes=2 \
    --master_addr=$MASTER_IP --master_port=6000 \
    tools/run_text_generation_server.py \
    --load /path/to/checkpoint \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 2
```

## Server CLI Arguments

### Model Loading
| Argument | Description | Default |
|---|---|---|
| `--load` | Checkpoint directory path | Required |
| `--tensor-model-parallel-size` | TP degree | 1 |
| `--pipeline-model-parallel-size` | PP degree | 1 |
| `--seq-length` | Input sequence length | 1024 |
| `--max-position-embeddings` | Max position embeddings | Same as seq-length |

### Generation Parameters
| Argument | Description | Default |
|---|---|---|
| `--max-tokens` | Maximum tokens to generate | 128 |
| `--temperature` | Sampling temperature | 1.0 |
| `--top-k` | Top-K sampling parameter | 0 (disabled) |
| `--top-p` | Top-P (nucleus) sampling | 0.0 (disabled) |
| `--seed` | Random seed | 1234 |
| `--repetition-penalty` | Repetition penalty factor | 1.0 |
| `--num-beams` | Number of beams for beam search | 1 |

### Server Configuration
| Argument | Description | Default |
|---|---|---|
| `--port` | HTTP server port | 5000 |
| `--max-batch-size` | Maximum concurrent requests | 1 |
| `--micro-batch-size` | Micro batch size for generation | Variable |

## HTTP API

### Generate Endpoint
```bash
# Single prompt
curl -X POST http://localhost:5000/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompts": ["Once upon a time"],
        "tokens_to_generate": 128,
        "temperature": 0.7,
        "top_p": 0.9
    }'
```

### Response Format
```json
{
    "text": ["Once upon a time there was a kingdom far away..."],
    "tokens_generated": 128,
    "prompt_tokens": 4,
    "total_tokens": 132
}
```

### Streaming Generation
```bash
curl -X POST http://localhost:5000/api/generate \
    -H "Content-Type: application/json" \
    -d '{"prompts": ["Hello"], "tokens_to_generate": 50, "stream": true}'
```

## CLI Client

```bash
# Interactive generation
python tools/text_generation_cli.py \
    --port 5000 \
    --temperature 0.7 \
    --max-tokens 128

# Batch generation from file
python tools/text_generation_cli.py \
    --port 5000 \
    --input-file prompts.txt \
    --output-file outputs.jsonl
```

## Vision-Language Model Generation

```bash
python tools/run_vlm_text_generation.py \
    --load /path/to/vlm/checkpoint \
    --tensor-model-parallel-size 4 \
    --image-path /path/to/image.jpg \
    --prompt "Describe this image"
```

## Performance Tuning

### Throughput Optimization
```bash
# Increase batch size for higher throughput
--max-batch-size 64
--micro-batch-size 8

# Enable CUDA graphs for reduced latency
--cuda-graph-impl local
```

### Latency Optimization
```bash
# Reduce batch size for lower latency
--max-batch-size 1

# Enable flash decode
--flash-decode

# Fuse TP communication
--inference-fuse-tp-communication
```

### Memory Optimization
```bash
# Use FP8 for reduced memory
--fp8-format e4m3
--fp8-recipe mxfp8
--fp8-param-gather

# Enable KV cache compression (MLA)
--cache-mla-latents
```
