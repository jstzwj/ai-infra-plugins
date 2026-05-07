---
name: sglang
description: >
  Comprehensive reference documentation and skill for SGLang - a high-performance serving framework
  for large language models and multimodal models. Covers SGLang architecture, ServerArgs configuration,
  OpenAI-compatible API server, native API, offline engine API, attention backends (FlashInfer, FlashAttention,
  Triton, FlashMLA, cutlass_mla), KV cache management with RadixAttention, paged attention, sampling and
  decoding (structured outputs, constrained decoding, speculative decoding with EAGLE/ngram/DFlash), distributed
  inference (tensor/pipeline/expert/data parallelism), PD disaggregation, EPD disaggregation, quantization
  (FP8, FP4/MXFP4, GPTQ, AWQ, INT4/INT8, Marlin, bitsandbytes, GGUF, modelopt), multi-LoRA batching,
  multimodal processing (image/audio/video), CUDA graphs, torch.compile, piecewise CUDA graphs, sgl-kernel,
  sgl-model-gateway (Rust), HiCache, HiSparse, RL/post-training support, checkpoint engine, diffusion models,
  observability, profiling, supported model architectures (200+ models), and hardware platforms (NVIDIA, AMD,
  Intel, Google TPU, Ascend NPU, Apple Metal, CPU). Based on SGLang source code analysis.
version: 0.5.9
---

# SGLang - High-Performance LLM Serving Framework

## Overview

SGLang is a fast serving framework for large language models and vision language models, designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters. It powers over 400,000 GPUs worldwide and generates trillions of tokens daily in production.

**Key Capabilities:**
- Fast serving with **RadixAttention** for automatic prefix caching
- **Zero-overhead CPU scheduler** with continuous batching and chunked prefill
- **Prefill-decode disaggregation** (PD) and **encode-prefill-decode disaggregation** (EPD)
- **Speculative decoding** (EAGLE, ngram, DFlash, adaptive)
- Optimized attention kernels (FlashInfer, FlashAttention 3/4, FlashMLA, Triton, cutlass_mla)
- **Tensor/pipeline/expert/data parallelism** for distributed inference
- Extensive **quantization** support (FP8, FP4/MXFP4, GPTQ, AWQ, INT4/INT8, Marlin, GGUF, modelopt)
- **Multi-LoRA** serving with batching support
- **Structured outputs** (JSON, regex, EBNF) via XGrammar/Outlines/llguidance
- OpenAI-compatible API server with streaming
- 200+ supported model architectures
- Multimodal model support (image, audio, video)
- **Diffusion model** acceleration (WAN, Qwen-Image, FLUX)
- **HiCache** for hierarchical KV cache offloading
- **RL/post-training backbone** with native integrations (AReaL, verl, slime, Tunix)

**Supported Hardware:** NVIDIA GPUs (GB200/GB300/B300/H100/A100/L40S/Spark/5090), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, Intel XPU, Apple Metal (MPS), Mthreads GPU

**SGLang Version:** 0.5.9 | **Python:** 3.10+ | **PyTorch:** 2.11.0 | **License:** Apache 2.0

## Architecture Overview

```
+------------------------------------------------------------------+
|                        Client Applications                        |
|  OpenAI API  |  Native API (/generate)  |  gRPC  |  Offline LLM  |
+------------------------------------------------------------------+
|                         API Server Layer                          |
|  FastAPI/Uvicorn  |  SSL/TLS  |  HTTP/2  |  Streaming SSE       |
|  /v1/chat/completions  |  /v1/completions  |  /v1/embeddings     |
|  /generate  |  /v1/files  |  /health  |  /get_model_info        |
+------------------------------------------------------------------+
|                    Tokenizer Manager (ZMQ)                         |
|  Batch Tokenize  |  Batch Detokenize  |  Chat Template           |
+------------------------------------------------------------------+
|                     Scheduler (CPU-side)                           |
|  FCFS/Priority Scheduling  |  RadixAttention Tree Cache          |
|  Chunked Prefill  |  Continuous Batching  |  Memory Management   |
|  Prefill Delayer  |  Dynamic Chunking  |  SWA Eviction           |
+------------------------------------------------------------------+
|                     DP Controller                                  |
|  Load Balancer  |  Request Routing  |  Cache-Aware Dispatch      |
+------------------------------------------------------------------+
|                    Model Execution (GPU-side)                      |
|  Model Runner  |  CUDA Graphs  |  Piecewise CUDA Graphs          |
|  torch.compile  |  FlashInfer Batch  |  Overlap Schedule          |
|  Two-Batch Overlap  |  Single-Batch Overlap                      |
+------------------------------------------------------------------+
|                    Distributed Computing                           |
|  Tensor Parallel  |  Pipeline Parallel  |  Expert Parallel        |
|  Data Parallel  |  PD Disaggregation  |  EPD Disaggregation      |
|  Elastic EP  |  EP Load Balancing  |  Context Parallelism        |
+------------------------------------------------------------------+
|                    Model & Layers                                  |
|  Attention (Flash/MLA/NSA)  |  Linear  |  MoE  |  Embedding       |
|  Rotary Embedding  |  Layernorm  |  Activation  |  Quantization |
|  LoRA Layers  |  Multimodal Encoders  |  Mamba/Linear Attn       |
+------------------------------------------------------------------+
|                    Memory Management                               |
|  Paged KV Cache  |  Radix Tree  |  Token-to-Slot Pool            |
|  HiCache (Hierarchical)  |  KV Cache Offloading  |  GPU Memory   |
+------------------------------------------------------------------+
|                    Kernels & Operators                             |
|  FlashInfer  |  FlashAttention 3/4  |  sgl-kernel (CUDA)         |
|  DeepGEMM (FP8/FP4)  |  Triton  |  CUTLASS  |  Marlin            |
|  DeepEP  |  Aiter (AMD)  |  cutedsl  |  Flash RL                  |
+------------------------------------------------------------------+
|                    Hardware Backends                               |
|  NVIDIA CUDA  |  AMD ROCm  |  Intel XPU  |  Google TPU/Jax       |
|  Ascend NPU  |  Apple MPS  |  x86 CPU  |  Mthreads MUSA          |
+------------------------------------------------------------------+
```

## Quick Reference

### Installation
```bash
# pip/uv (recommended)
pip install --upgrade pip && pip install uv && uv pip install sglang

# From source
git clone https://github.com/sgl-project/sglang.git && cd sglang
pip install -e "python"

# Docker
docker run --gpus all --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000
```

### Launch Server
```bash
# Basic server
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

# With tensor parallelism
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --tp 8

# With LoRA support
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora --lora-paths my_adapter=/path/to/adapter

# PD disaggregation
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill --port 30000

# With speculative decoding (EAGLE)
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-num-steps 5 --speculative-eagle-topk 8

# With quantization (FP8)
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 --kv-cache-dtype fp8_e5m2
```

### Send Requests
```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

# Chat completions
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=128,
)

# Streaming
for chunk in client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Native API
import requests
response = requests.post("http://localhost:30000/generate", json={
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
})
```

### Offline Batch Inference
```python
import sglang as sgl

llm = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello!", "What is AI?"], {"temperature": 0.7})
for output in outputs:
    print(output["text"])
llm.shutdown()
```

## Reference Documents

| Document | Description |
|---|---|
| [01 - Overview and Architecture](references/01-overview-architecture.md) | Project overview, system architecture, core abstractions, data flow, module structure |
| [02 - Installation and Setup](references/02-installation-setup.md) | Installation methods, Docker, hardware support, environment setup |
| [03 - Server Configuration](references/03-server-configuration.md) | Complete ServerArgs reference, all CLI flags, defaults, and validation |
| [04 - API Reference](references/04-api-reference.md) | OpenAI API, native /generate API, gRPC, offline engine, embeddings, scoring |
| [05 - Supported Models](references/05-supported-models.md) | All supported model architectures, configurations, and usage examples |
| [06 - Attention and KV Cache](references/06-attention-kv-cache.md) | Attention backends, RadixAttention, paged attention, KV cache management |
| [07 - Sampling and Decoding](references/07-sampling-decoding.md) | Sampling parameters, structured outputs, constrained decoding, custom logit processors |
| [08 - Distributed Serving](references/08-distributed-serving.md) | TP, PP, EP, DP, PD disaggregation, EPD, multi-node, context parallelism |
| [09 - Quantization](references/09-quantization.md) | All quantization methods: FP8, FP4, GPTQ, AWQ, Marlin, GGUF, modelopt, etc. |
| [10 - Speculative Decoding](references/10-speculative-decoding.md) | EAGLE, ngram, DFlash, adaptive speculative decoding configurations |
| [11 - LoRA Adapters](references/11-lora-adapters.md) | Multi-LoRA serving, loading, batching, eviction policies |
| [12 - Multimodal Support](references/12-multimodal.md) | VLM support, image/video/audio processing, vision encoders |
| [13 - Kernels and Compilation](references/13-kernels-compilation.md) | sgl-kernel, CUDA graphs, torch.compile, piecewise graphs, compilation pipeline |
| [14 - Scheduling and Memory](references/14-scheduling-memory.md) | Scheduler design, memory management, batch scheduling, chunked prefill |
| [15 - Observability and Profiling](references/15-observability-profiling.md) | Prometheus metrics, OpenTelemetry tracing, profiling tools, debugging |
| [16 - Layers and Operations](references/16-layers-operations.md) | Model layers, attention implementations, linear layers, MoE, normalizations |
| [17 - Advanced Features](references/17-advanced-features.md) | HiCache, HiSparse, RL support, checkpoint engine, diffusion models, forward hooks |
| [18 - Benchmark and Deployment](references/18-benchmark-deployment.md) | Benchmark tools, Docker deployment, Kubernetes, multi-node setup |
| [19 - Hardware Platforms](references/19-hardware-platforms.md) | NVIDIA, AMD, Intel, TPU, Ascend NPU, CPU, MPS platform-specific guides |
| [20 - Environment Variables](references/20-environment-variables.md) | Complete environment variable reference with defaults and descriptions |
