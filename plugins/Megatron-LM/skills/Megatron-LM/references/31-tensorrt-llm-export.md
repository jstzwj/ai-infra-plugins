# Chapter 31: TensorRT-LLM Export

## Source Files
- `sources/Megatron-LM/megatron/core/export/trtllm/` - Export utilities
- `sources/Megatron-LM/megatron/core/export/trtllm/engine_builder/` - Engine builder
- `sources/Megatron-LM/megatron/core/export/trtllm/weights_converter/` - Weight conversion

## Overview

Megatron-LM supports exporting trained models to TensorRT-LLM format for optimized production inference. The export pipeline converts Megatron checkpoints to TensorRT-LLM weights and builds optimized inference engines.

## Export Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Megatron-LM    │────►│  Weight           │────►│  TensorRT-LLM    │
│  Checkpoint     │     │  Converter        │     │  Engine          │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                              │                         │
                              ▼                         ▼
                        ┌──────────────────┐     ┌───────────────────┐
                        │  Weight Files    │     │  Optimized Model  │
                        │  (NPZ/Safetensors)│    │  (TRT Engine)     │
                        └──────────────────┘     └───────────────────┘
```

## Weight Conversion

### Convert Megatron Checkpoint to TRT-LLM Weights
```python
from megatron.core.export.trtllm.weights_converter import convert_weights

convert_weights(
    model_path="/path/to/megatron/checkpoint",
    output_path="/path/to/trtllm/weights",
    model_type="gpt",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
)
```

### Command-Line Conversion
```bash
python -m megatron.core.export.trtllm.weights_converter \
    --model-path /path/to/megatron/checkpoint \
    --output-path /path/to/trtllm/weights \
    --model-type gpt \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1
```

### Supported Model Types

| Model Type | Description | Supported Features |
|---|---|---|
| `gpt` | GPT/GPT-2/GPT-3 decoder-only | FP16, FP8, GQA, RoPE |
| `llama` | LLaMA/Llama-2/Llama-3 | GQA, RoPE, SwiGLU |
| `mixtral` | Mixtral MoE | MoE, GQA |
| `bert` | BERT encoder | FP16, position embeddings |
| `t5` | T5 encoder-decoder | Encoder-decoder, span attention |

### Weight Conversion Options

| Parameter | Type | Description |
|---|---|---|
| `model_path` | str | Path to Megatron checkpoint |
| `output_path` | str | Output directory for TRT-LLM weights |
| `model_type` | str | Model architecture type |
| `tensor_parallel_size` | int | TP degree for the target engine |
| `pipeline_parallel_size` | int | PP degree for the target engine |
| `dtype` | str | Output precision (fp16, bf16, fp8) |
| `use_fp8` | bool | Enable FP8 quantization |

## Engine Building

### Build TRT-LLM Engine
```python
from megatron.core.export.trtllm.engine_builder import build_engine

build_engine(
    weights_path="/path/to/trtllm/weights",
    engine_path="/path/to/output/engine",
    model_type="gpt",
    max_batch_size=32,
    max_input_len=2048,
    max_output_len=512,
    tensor_parallel_size=4,
)
```

### Command-Line Engine Building
```bash
python -m megatron.core.export.trtllm.engine_builder \
    --weights-path /path/to/trtllm/weights \
    --engine-path /path/to/output/engine \
    --model-type gpt \
    --max-batch-size 32 \
    --max-input-len 2048 \
    --max-output-len 512 \
    --tensor-parallel-size 4
```

### Engine Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_batch_size` | int | 32 | Maximum batch size for inference |
| `max_input_len` | int | 2048 | Maximum input sequence length |
| `max_output_len` | int | 512 | Maximum output sequence length |
| `max_num_tokens` | int | Auto | Maximum total tokens (input + output) |
| `tensor_parallel_size` | int | 1 | TP degree for inference |
| `pipeline_parallel_size` | int | 1 | PP degree for inference |
| `use_fp8` | bool | False | Enable FP8 quantization |
| `use_beam_search` | bool | False | Enable beam search |
| `num_beams` | int | 1 | Number of beams |

## FP8 Export

```bash
# Export with FP8 quantization
python -m megatron.core.export.trtllm.weights_converter \
    --model-path /path/to/checkpoint \
    --output-path /path/to/trtllm/weights \
    --model-type gpt \
    --use-fp8

# Build FP8 engine
python -m megatron.core.export.trtllm.engine_builder \
    --weights-path /path/to/trtllm/weights \
    --engine-path /path/to/output/engine \
    --use-fp8
```

## End-to-End Export Pipeline

```bash
# Step 1: Train model with Megatron-LM
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --save /path/to/checkpoint \
    [training args...]

# Step 2: Convert checkpoint
python -m megatron.core.export.trtllm.weights_converter \
    --model-path /path/to/checkpoint \
    --output-path /path/to/trtllm/weights \
    --model-type gpt

# Step 3: Build engine
python -m megatron.core.export.trtllm.engine_builder \
    --weights-path /path/to/trtllm/weights \
    --engine-path /path/to/trtllm/engine \
    --max-batch-size 32 \
    --max-input-len 2048 \
    --max-output-len 512

# Step 4: Run inference with TRT-LLM
mpirun -np 4 python run_trtllm_inference.py \
    --engine-path /path/to/trtllm/engine
```

## Performance Comparison

| Configuration | Megatron Inference | TRT-LLM Inference | Speedup |
|---|---|---|---|
| GPT-3 175B, TP=8 | ~15 tokens/s | ~45 tokens/s | 3x |
| LLaMA-2 70B, TP=4 | ~25 tokens/s | ~80 tokens/s | 3.2x |
| Mixtral 8x7B, TP=4 | ~20 tokens/s | ~65 tokens/s | 3.25x |

*Approximate throughput for single-stream generation on H100 GPUs*
