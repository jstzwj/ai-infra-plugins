---
name: vllm
description: >
  Comprehensive reference documentation and skill for vLLM - a high-throughput and memory-efficient
  inference and serving engine for large language models (LLMs). Covers vLLM architecture (V0 and V1),
  engine APIs (LLMEngine, AsyncLLMEngine, LLM), OpenAI-compatible API server, configuration system,
  model executor and layers, attention mechanisms (PagedAttention, FlashAttention, MLA), KV cache
  management, sampling and decoding (beam search, speculative decoding, structured outputs), distributed
  inference (tensor/pipeline/data/expert parallelism), multimodal processing (image/audio/video),
  compilation and CUDA graphs, custom kernels, quantization (FP8, GPTQ, AWQ, INT4/INT8, etc.),
  LoRA and adapters, scheduling and memory management, supported model architectures (200+ models),
  speculative decoding (EAGLE, Medusa, n-gram), observability and profiling, and hardware platforms
  (NVIDIA GPU, AMD GPU, CPU, TPU, Intel GPU/XPU).
version: 0.9.1
---

# vLLM - High-Throughput LLM Inference and Serving Engine

## Overview

vLLM is a fast and easy-to-use library for LLM inference and serving, originally developed at UC Berkeley's Sky Computing Lab. It provides state-of-the-art serving throughput through efficient attention key/value memory management via **PagedAttention**, continuous batching, chunked prefill, and extensive hardware acceleration support.

**Key Capabilities:**
- State-of-the-art serving throughput with PagedAttention
- Continuous batching of incoming requests with chunked prefill
- Fast model execution with CUDA/HIP graphs and torch.compile
- Extensive quantization support (FP8, GPTQ, AWQ, INT4/INT8, GGUF, etc.)
- Optimized attention kernels (FlashAttention, FlashInfer, FlashMLA, Triton)
- Speculative decoding (n-gram, EAGLE, Medusa, DFlash)
- Distributed inference (tensor, pipeline, data, expert parallelism)
- OpenAI-compatible API server with streaming
- Multi-LoRA serving support
- Multimodal model support (image, audio, video)
- 200+ supported model architectures
- Structured output generation (JSON, regex, CFG)
- Disaggregated prefill/decode/encode

**Supported Hardware:** NVIDIA GPUs (CUDA), AMD GPUs (ROCm), x86/ARM/PowerPC CPUs, Google TPUs, Intel Gaudi/XPU, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

**vLLM Version:** 0.9.1 | **Python:** 3.9-3.12 | **License:** Apache 2.0

## Architecture Overview

### High-Level System Architecture

```
+------------------------------------------------------------------+
|                        Client Applications                        |
|  OpenAI API  |  Python SDK (LLM)  |  gRPC  |  Custom Clients    |
+------------------------------------------------------------------+
|                         API Server Layer                          |
|  /v1/completions  |  /v1/chat/completions  |  /v1/embeddings    |
|  FastAPI/Starlette  |  SSL/TLS  |  Streaming SSE               |
+------------------------------------------------------------------+
|                          Engine Layer                             |
|  AsyncLLMEngine  |  LLMEngine  |  V1 AsyncLLM  |  V1 LLMEngine |
|  Request Queue   |  Output Queue  |  Tokenizer  |  Detokenizer  |
+------------------------------------------------------------------+
|                    Scheduling & Coordination                      |
|  V1 Core (Scheduler)  |  KV Cache Manager  |  Block Pool        |
|  KV Cache Coordinator |  Encoder Cache  |  Prefix Cache         |
+------------------------------------------------------------------+
|                      Model Execution                              |
|  GPU Model Runner  |  CPU Model Runner  |  XPU Model Runner     |
|  CUDA Graphs  |  torch.compile  |  Piecewise Compilation        |
+------------------------------------------------------------------+
|                    Worker & Executor Layer                         |
|  GPU Worker  |  CPU Worker  |  TPU Worker  |  XPU Worker        |
|  Uniproc  |  Multiproc  |  Ray Executor  |  Data Parallel       |
+------------------------------------------------------------------+
|                    Distributed Computing                           |
|  Tensor Parallel  |  Pipeline Parallel  |  Expert Parallel      |
|  Data Parallel  |  Disaggregated Prefill  |  KV Transfer (NIXL) |
+------------------------------------------------------------------+
|                       Model & Layers                              |
|  Attention (Flash/MLA)  |  Linear  |  MoE  |  Embedding         |
|  Rotary Embedding  |  Layernorm  |  Activation  |  Quantization |
|  LoRA Layers  |  Multimodal Encoders  |  Pooling                |
+------------------------------------------------------------------+
|                     Memory Management                             |
|  PagedAttention KV Cache  |  Block Tables  |  Workspace          |
|  KV Cache Offloading  |  GPU Memory Allocator  |  Swapping      |
+------------------------------------------------------------------+
|                     Kernels & Operators                           |
|  FlashAttention  |  FlashInfer  |  Triton  |  CUTLASS           |
|  Custom CUDA  |  aiter_ops (AMD)  |  xpu_ops (Intel)            |
+------------------------------------------------------------------+
|                     Hardware Backends                             |
|  NVIDIA CUDA  |  AMD ROCm  |  CPU  |  Google TPU  |  Intel XPU |
+------------------------------------------------------------------+
```

### V1 Architecture (Current Default)

vLLM V1 is the redesigned architecture with improved performance:

```
                        +-------------------+
                        |  API Server       |
                        |  (FastAPI/gRPC)   |
                        +--------+----------+
                                 |
                        +--------v----------+
                        |  AsyncLLM (V1)    |
                        |  - InputProcessor |
                        |  - Detokenizer    |
                        |  - OutputProcessor|
                        +--------+----------+
                                 |
                  +--------------+--------------+
                  |                             |
         +--------v----------+        +--------v----------+
         |  Engine Core      |        |  Engine Core      |
         |  (Scheduler)      |  ...   |  (Scheduler)      |
         |  - KV Cache Mgr   |        |  - KV Cache Mgr   |
         |  - Block Pool     |        |  - Block Pool     |
         +--------+----------+        +--------+----------+
                  |                             |
         +--------v----------+        +--------v----------+
         |  GPU Worker       |        |  GPU Worker       |
         |  - Model Runner   |        |  - Model Runner   |
         |  - CUDA Graphs    |        |  - CUDA Graphs    |
         +-------------------+        +-------------------+
```

### Request Lifecycle

```
1. Client sends request to API server
   |
2. API server parses request, creates SamplingParams
   |
3. AsyncLLMEngine.add_request() -> InputProcessor processes
   |
4. Engine Core schedules request (KV cache allocation)
   |
5. GPU Model Runner executes forward pass
   |  - Embedding lookup
   |  - Transformer layers (attention + FFN)
   |  - Sampling / Logits processing
   |
6. OutputProcessor collects results
   |
7. Detokenizer converts tokens to text
   |
8. Response streamed back to client
```

## Core Abstractions

### Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `LLM` | `vllm.entrypoints.llm` | Offline inference entrypoint |
| `LLMEngine` | `vllm.engine.llm_engine` | Synchronous engine (V0) |
| `AsyncLLMEngine` | `vllm.engine.async_llm_engine` | Async engine with streaming (V0) |
| `EngineArgs` | `vllm.engine.arg_utils` | Engine configuration arguments |
| `SamplingParams` | `vllm.sampling_params` | Sampling parameters for generation |
| `PoolingParams` | `vllm.pooling_params` | Parameters for pooling/embedding |
| `RequestOutput` | `vllm.outputs` | Complete output for a request |
| `CompletionOutput` | `vllm.outputs` | Output for a single completion |
| `ModelRegistry` | `vllm.model_executor.models` | Registry of supported models |

### Data Flow Types

| Type | Description |
|------|-------------|
| `TextPrompt` | Text input with optional multi-modal data |
| `TokensPrompt` | Token IDs input |
| `PromptType` | Union of all prompt types |
| `MultiModalDataDict` | Multi-modal input data (image/audio/video) |
| `MultiModalKwargs` | Processed multi-modal features |
| `DecoderOnlyInputs` | Inputs for decoder-only models |
| `EncoderDecoderInputs` | Inputs for encoder-decoder models |

## Quick Start Examples

### Offline Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# Generate
outputs = llm.generate(["Hello, how are you?"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### Online Serving (OpenAI-compatible)

```bash
# Start the server
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

# Query with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### Async Engine

```python
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate():
    results = engine.generate(
        "Write a poem about AI",
        SamplingParams(max_tokens=256),
        request_id="req-1"
    )
    async for output in results:
        if output.finished:
            print(output.outputs[0].text)
```

## Key Modules

| Module | Path | Description |
|--------|------|-------------|
| Engine | `vllm/engine/` | LLMEngine and AsyncLLMEngine (V0) |
| V1 Engine | `vllm/v1/engine/` | V1 engine architecture |
| V1 Core | `vllm/v1/core/` | Scheduler, KV cache management |
| V1 Worker | `vllm/v1/worker/` | GPU/CPU/XPU model runners |
| V1 Executor | `vllm/v1/executor/` | Uniproc/Multiproc/Ray executors |
| Entrypoints | `vllm/entrypoints/` | API server, LLM offline, gRPC |
| Config | `vllm/config/` | All configuration classes |
| Model Executor | `vllm/model_executor/` | Models, layers, quantization |
| Kernels | `vllm/kernels/` | Custom GPU kernels |
| Compilation | `vllm/compilation/` | torch.compile, CUDA graphs |
| Distributed | `vllm/distributed/` | Parallel state, communication |
| Multimodal | `vllm/multimodal/` | Image/audio/video processing |
| LoRA | `vllm/lora/` | LoRA adapter management |
| Sampling | `vllm/v1/sample/` | Sampling implementations |
| Structured Output | `vllm/v1/structured_output/` | JSON/regex/CFG guided decoding |
| Speculative Decoding | `vllm/v1/spec_decode/` | EAGLE, Medusa, n-gram |
| Attention | `vllm/v1/attention/` | Attention backends |
| Profiler | `vllm/profiler/` | Profiling tools |
| Tracing | `vllm/tracing/` | OpenTelemetry tracing |
| Platforms | `vllm/platforms/` | Hardware platform abstractions |

## Supported Features

### Quantization Methods
- FP8 (E4M3, E5M2)
- MXFP8, MXFP4, NVFP4
- INT8 weight-only, INT4 weight-only
- GPTQ, AWQ, SqueezeLLM
- GGUF quantization
- Compressed-tensors
- ModelOpt, TorchAO
- BitsAndBytes (NF4)
- FBGEMM (FP8, INT8)

### Attention Backends
- FlashAttention 2/3 (NVIDIA)
- FlashInfer
- FlashMLA (DeepSeek)
- TRTLLM-GEN
- ROCm FlashAttention (AMD)
- XFormers
- Triton-based attention

### Parallelism Strategies
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Data Parallelism (DP)
- Expert Parallelism (EP) for MoE
- Context Parallelism (CP)
- Disaggregated Prefill/Decode

### Model Types
- Decoder-only LLMs (Llama, Qwen, Gemma, Mistral, etc.)
- Mixture-of-Expert LLMs (Mixtral, DeepSeek-V3, Qwen-MoE)
- Hybrid attention/SSM models (Mamba, Jamba)
- Multi-modal models (LLaVA, Qwen-VL, Pixtral)
- Embedding/retrieval models (E5-Mistral, GTE, ColBERT)
- Reward/classification models
- Encoder-decoder models (T5, BART)
- Audio models (Whisper, Ultravox)
- Speech-to-text models

## Reference Documents

For detailed documentation on each topic, see the reference documents:

1. [Overview and Architecture](references/01-overview-architecture.md) - Project overview, architecture diagrams, core abstractions, data flow
2. [Installation and Setup](references/02-installation-setup.md) - Installation methods, Docker, hardware requirements, environment setup
3. [Engine and Serving](references/03-engine-and-serving.md) - LLMEngine, AsyncLLMEngine, V1 engine, request lifecycle
4. [API Server Reference](references/04-api-server-reference.md) - OpenAI-compatible API, endpoints, request/response formats
5. [Configuration Reference](references/05-configuration-reference.md) - All configuration classes, parameters, defaults
6. [Model Executor](references/06-model-executor.md) - Models, layers, quantization, weight loading
7. [Attention and KV Cache](references/07-attention-kv-cache.md) - PagedAttention, KV cache management, block tables
8. [Sampling and Decoding](references/08-sampling-decoding.md) - Sampling strategies, beam search, structured output
9. [Distributed Computing](references/09-distributed.md) - TP, PP, DP, EP, distributed KV cache
10. [Multimodal](references/10-multimodal.md) - Image/audio/video processing, multi-modal models
11. [Compilation and CUDA Graphs](references/11-compilation-cuda-graphs.md) - torch.compile, CUDA graph capture, piecewise compilation
12. [Kernels and Operators](references/12-kernels-operators.md) - Custom kernels, FlashAttention, MoE kernels
13. [LoRA and Adapters](references/13-lora-adapters.md) - LoRA management, multi-LoRA, prompt adapters
14. [Scheduling and Memory](references/14-scheduling-memory.md) - Scheduler, block management, memory strategies
15. [V1 Worker and Executor](references/15-v1-worker-executor.md) - GPU model runner, workers, ubatching
16. [Supported Models](references/16-supported-models.md) - Complete catalog of 200+ supported architectures
17. [Speculative Decoding](references/17-speculative-decoding.md) - EAGLE, Medusa, n-gram, draft models
18. [Observability and Profiling](references/18-observability-profiling.md) - Metrics, tracing, profiling, logging

## Environment Variables

Key environment variables for controlling vLLM behavior:

| Variable | Description |
|----------|-------------|
| `VLLM_USE_V1` | Use V1 architecture (default: 1) |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker spawn method (spawn/fork) |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | Enable runtime LoRA updates |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow long context lengths |
| `VLLM_API_KEY` | API key for server authentication |
| `VLLM_HOST` | Server host address |
| `VLLM_PORT` | Server port |
| `VLLM_USE_PRECOMPILED` | Use pre-compiled C extensions |
| `VLLM_TARGET_DEVICE` | Target device (cuda/rocm/cpu/tpu) |

See `vllm/envs.py` for the complete list of environment variables.

## Common Usage Patterns

### Chat with Vision Model

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
sampling_params = SamplingParams(max_tokens=256, temperature=0.7)

from vllm.inputs import TextPrompt
output = llm.generate([
    TextPrompt(
        prompt="<|image_pad|>What is in this image?",
        multi_modal_data={"image": "path/to/image.jpg"}
    )
], sampling_params)
```

### Multi-LoRA Serving

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-lora \
  --lora-modules lora1=/path/to/lora1 lora2=/path/to/lora2
```

### Distributed Inference

```bash
# Tensor parallelism across 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

# Pipeline parallelism across 2 nodes
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

### Quantized Model

```bash
# AWQ quantized model
vllm serve TheBloke/Llama-2-7B-AWQ --quantization awq

# GPTQ quantized model
vllm serve TheBloke/Llama-2-7B-GPTQ --quantization gptq

# FP8 quantization
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8
```

### Structured Output

```python
from vllm import LLM, SamplingParams
import json

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
params = SamplingParams(
    max_tokens=256,
    guided_json={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
)
output = llm.generate(["Generate a person profile"], params)
result = json.loads(output[0].outputs[0].text)
```
