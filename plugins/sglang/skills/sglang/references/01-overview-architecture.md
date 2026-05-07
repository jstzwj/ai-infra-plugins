# SGLang Overview and Architecture Reference

This document provides a comprehensive overview of SGLang, including its mission, architecture,
core abstractions, data flow, module organization, and public API surface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Abstractions](#core-abstractions)
4. [Data Flow: Request to Response](#data-flow-request-to-response)
5. [Key Modules and Responsibilities](#key-modules-and-responsibilities)
6. [Source Code Organization](#source-code-organization)
7. [Public API Reference](#public-api-reference)

---

## Project Overview

### Mission Statement

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal
models. It is designed to deliver low-latency and high-throughput inference across a wide range of
setups, from a single GPU to large distributed clusters. SGLang is hosted under the non-profit
open-source organization LMSYS and has become the de facto industry standard LLM inference engine,
with deployments running on over 400,000 GPUs worldwide.

### Key Capabilities

- **Fast Runtime**: Efficient serving with RadixAttention for prefix caching, a zero-overhead CPU
  scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged
  attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill,
  quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, embedding models
  (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image).
  Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: NVIDIA GPUs (GB200/B300/H100/A100/Spark/5090), AMD GPUs
  (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: Open-source with widespread industry adoption, powering trillions of tokens
  daily.
- **RL and Post-Training Backbone**: Proven rollout backend used for training many frontier models,
  with native RL integrations and adoption by frameworks such as AReaL, Miles, slime, Tunix, verl.

### Project Metadata

| Field              | Value                                            |
|--------------------|--------------------------------------------------|
| Name               | SGLang                                            |
| License            | Apache License 2.0                                |
| PyPI Package       | sglang                                            |
| Python Version     | 3.10+                                             |
| GPU Compute        | sm80+ (A10, A100, L4, L40S, H100, B200, etc.)   |
| Primary Language   | Python                                            |
| Repository         | https://github.com/sgl-project/sglang             |
| Documentation      | https://docs.sglang.io/                           |
| Docker Hub         | lmsysorg/sglang                                   |
| Organization       | LMSYS (LMSYS Org)                                 |

---

## High-Level Architecture

SGLang follows a layered architecture with clear separation of concerns across API endpoints,
tokenization, scheduling, model execution, and memory management.

```
+--------------------------------------------------------------------+
|                        CLIENT LAYER                                 |
|  cURL / OpenAI SDK / Ollama CLI / Python requests / Offline Engine  |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                      API / ENTRYPOINT LAYER                         |
|                                                                      |
|  +-------------------+  +------------------+  +------------------+  |
|  |  HTTP Server      |  |  gRPC Server     |  |  Offline Engine  |  |
|  |  (FastAPI/Uvicorn)|  |  (optional)      |  |  (sgl.Engine)    |  |
|  +-------------------+  +------------------+  +------------------+  |
|                                                                      |
|  OpenAI-compat: /v1/chat/completions, /v1/completions, /v1/embed.. |
|  Native: /generate, /encode, /v1/rerank, /v1/score, /classify      |
|  Ollama: /api/chat, /api/generate, /api/tags, /api/show            |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                    TOKENIZER MANAGER LAYER                           |
|                                                                      |
|  - TokenizerManager: tokenizes requests, manages worker processes   |
|  - Multi-modal input processing (image, audio, video)               |
|  - Chat template application                                        |
|  - Request routing to Scheduler via ZMQ                              |
|  - Multiple tokenizer workers (configurable)                        |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                     SCHEDULER LAYER                                  |
|                                                                      |
|  +-------------------+  +------------------+  +------------------+  |
|  | Scheduler         |  | Cache-aware      |  | Priority         |  |
|  | (FCFS / priority) |  | Load Balancer    |  | Scheduler        |  |
|  +-------------------+  +------------------+  +------------------+  |
|                                                                      |
|  - Continuous batching                                              |
|  - RadixAttention (prefix caching) with LRU/LFU/SLRU eviction      |
|  - Chunked prefill scheduling                                       |
|  - Prefill delayer for throughput optimization                      |
|  - Memory management and token budget allocation                    |
|  - Data parallelism routing                                         |
|  - Expert parallelism coordination                                  |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                    MODEL RUNNER LAYER                                |
|                                                                      |
|  +-------------------+  +------------------+  +------------------+  |
|  | ModelRunner       |  | CUDA Graphs      |  | Batch Manager    |  |
|  | (forward pass)    |  | (optimized path) |  | (prefill/decode) |  |
|  +-------------------+  +------------------+  +------------------+  |
|                                                                      |
|  - Model forward execution (Transformer layers)                     |
|  - Attention backends: FlashInfer, FA3, FA4, Triton, Cutlass MLA   |
|  - Sampling backends: FlashInfer, PyTorch                           |
|  - Speculative decoding (EAGLE, Medusa, ngram)                     |
|  - Torch.compile integration                                        |
|  - Tensor parallelism (TP) execution                                |
|  - Pipeline parallelism (PP) execution                              |
|  - Expert parallelism (EP) execution                                |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                    MEMORY / INFRASTRUCTURE LAYER                     |
|                                                                      |
|  +-------------------+  +------------------+  +------------------+  |
|  | KV Cache          |  | Radix Tree       |  | HiCache          |  |
|  | (paged attention) |  | (prefix cache)   |  | (hierarchical)   |  |
|  +-------------------+  +------------------+  +------------------+  |
|                                                                      |
|  +-------------------+  +------------------+  +------------------+  |
|  | GPU Memory Pool   |  | Weight Loader    |  | NCCL / Comms     |  |
|  | (token allocator) |  | (safetensors,pt) |  | (distributed)    |  |
|  +-------------------+  +------------------+  +------------------+  |
|                                                                      |
|  - Paged KV-cache management                                        |
|  - GPU memory allocation and tracking                               |
|  - Model weight loading (safetensors, PyTorch, GGUF, etc.)          |
|  - NCCL/Torch distributed communication                             |
|  - CUDA graph capture and replay                                    |
|  - LoRA weight management                                           |
|  - Quantization support (FP8, FP4, AWQ, GPTQ, etc.)                |
+--------------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------------+
|                      HARDWARE LAYER                                  |
|                                                                      |
|  NVIDIA GPUs (CUDA) | AMD GPUs (ROCm/HIP) | Intel CPUs (AMX)       |
|  Google TPUs (JAX)  | Ascend NPUs         | Intel XPU              |
+--------------------------------------------------------------------+
```

### Architecture Principles

1. **Separation of Tokenization and Scheduling**: The TokenizerManager handles text processing
   while the Scheduler handles batch formation and memory management. This allows independent
   scaling and optimization.

2. **Zero-Overhead CPU Scheduling**: The scheduler runs on the CPU without blocking GPU
   operations. It prepares batches while the GPU is executing the previous batch, achieving
   overlap between scheduling and computation.

3. **RadixAttention**: A radix-tree-based prefix caching system that automatically detects and
   reuses common prompt prefixes across requests, dramatically reducing redundant computation.

4. **Continuous Batching**: Requests are dynamically added to and removed from the active batch
   without waiting for the entire batch to complete, maximizing GPU utilization.

5. **Disaggregated Serving**: Prefill and decode phases can be separated onto different GPU
   clusters, allowing each phase to be independently optimized.

---

## Core Abstractions

### ServerArgs

The central configuration object for the SGLang server. It is a Python dataclass that holds all
server parameters, from model path and hardware configuration to API settings and optimization
flags. Defined in `python/sglang/srt/server_args.py`.

Key aspects:
- Contains 200+ configurable parameters organized into functional groups
- Supports CLI argument parsing via `add_cli_args()` and `from_cli_args()`
- Performs extensive validation in `__post_init__()` including cross-parameter dependency checks
- Supports YAML/JSON configuration through programmatic construction

### Engine (Offline)

The offline inference engine (`sgl.Engine`) provides direct access to the SGLang runtime without
an HTTP server. It is ideal for batch inference and custom server implementations.

```python
import sglang as sgl

llm = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello, world!"], {"temperature": 0.7})
llm.shutdown()
```

### TokenizerManager

Manages the tokenization pipeline. Responsibilities include:
- Tokenizing incoming text requests into token IDs
- Processing multi-modal inputs (images, audio, video)
- Applying chat templates
- Routing requests to the Scheduler via ZMQ (inter-process communication)
- Managing multiple tokenizer workers for parallel tokenization
- Handling request-level parameter validation

### Scheduler

The heart of the SGLang runtime. The Scheduler:
- Implements continuous batching with configurable policies (FCFS, priority)
- Manages the KV-cache radix tree for prefix reuse
- Controls memory allocation and token budget
- Coordinates prefill and decode phases
- Handles chunked prefill for long sequences
- Routes requests across data-parallel workers
- Implements the prefill delayer for throughput optimization

The Scheduler runs as a separate process and communicates with the TokenizerManager and
ModelRunner via ZMQ.

### ModelRunner

Executes the model forward pass on the GPU. Responsibilities include:
- Loading model weights and distributing across GPUs
- Executing prefill and decode forward passes
- Managing CUDA graph capture and replay for decode optimization
- Implementing tensor parallelism via NCCL
- Supporting speculative decoding with draft models
- Handling quantized model execution

### RadixAttention

SGLang's signature feature for KV-cache prefix reuse. It uses a radix tree data structure to:
- Automatically detect common prefixes across requests
- Share KV-cache entries for matching prefixes
- Implement configurable eviction policies (LRU, LFU, SLRU, priority)
- Support hierarchical caching (HiCache) for offloading to host memory

### IoStruct (Request/Response Types)

Defines the wire protocol between the API layer and the scheduler:
- `GenerateReqInput`: Incoming generation request with text, sampling params, and options
- `EmbeddingReqInput`: Embedding request for encoding models
- `TokenizeReqInput`: Tokenization request
- `BatchTokenIDOut`: Batched output from the model runner
- Various response types for different API endpoints

---

## Data Flow: Request to Response

### Step-by-Step: Chat Completion Request

This section traces a single `/v1/chat/completions` request through the entire system.

```
1. CLIENT sends HTTP POST to /v1/chat/completions
   |
   v
2. HTTP SERVER (FastAPI) receives the request
   - Parses JSON body (model, messages, temperature, etc.)
   - Validates API key (if configured)
   - Resolves LoRA adapter if model name contains ":adapter"
   |
   v
3. TOKENIZER MANAGER processes the request
   - Applies chat template to messages
   - Tokenizes the rendered prompt into token IDs
   - Processes multi-modal inputs (images, audio) if present
   - Creates a GenerateReqInput object
   - Sends request to Scheduler via ZMQ
   |
   v
4. SCHEDULER receives the request
   - Checks for matching prefix in RadixAttention tree
   - If prefix match found, reuses existing KV-cache entries
   - Allocates new KV-cache slots for non-cached tokens
   - Adds request to waiting queue
   - On next scheduling iteration:
     a. Selects requests from queue (FCFS or priority)
     b. Forms a batch respecting memory constraints
     c. Determines prefill vs decode for each request
     d. Sends batch to ModelRunner
   |
   v
5. MODEL RUNNER executes forward pass
   - For PREFILL requests:
     a. Processes prompt tokens in chunks (chunked prefill)
     b. Computes attention using configured backend
     c. Stores KV-cache entries
     d. Returns logits for last token position
   - For DECODE requests:
     a. Uses CUDA graph for optimized execution
     b. Processes one new token per request
     c. Computes attention against full KV-cache
     d. Returns logits for next token
   |
   v
6. SAMPLING on GPU/CPU
   - Applies temperature scaling
   - Applies top-p, top-k, min-p filtering
   - Applies frequency/presence penalties
   - Handles structured output constraints (JSON, regex, EBNF)
   - Samples next token
   |
   v
7. SCHEDULER processes sampled tokens
   - Checks for EOS token or stop sequences
   - Checks for max_new_tokens limit
   - If generation not complete, re-queues for next decode step
   - If generation complete, prepares final response
   |
   v
8. TOKENIZER MANAGER receives completed request
   - Detokenizes output token IDs to text
   - Strips special tokens and stop sequences
   - Constructs response object
   |
   v
9. HTTP SERVER sends response to CLIENT
   - Non-streaming: Complete JSON response
   - Streaming: Server-Sent Events (SSE) with incremental updates
```

### Step-by-Step: Offline Engine Request

The offline engine follows a similar path but without the HTTP layer:

```
1. User creates Engine instance
   llm = sgl.Engine(model_path="model_name")

2. User calls generate()
   outputs = llm.generate(prompts, sampling_params)

3. Engine sends requests directly to TokenizerManager
   (no HTTP/ZMQ overhead for the API layer)

4. TokenizerManager processes and routes to Scheduler

5. Scheduler batches and sends to ModelRunner

6. Results flow back through the same path

7. Engine returns results to caller

8. User calls llm.shutdown() to release resources
```

### Streaming Data Flow

For streaming responses, the data flow differs in the response path:

```
1-6. Same as non-streaming (steps 1-6 above)

7. SCHEDULER emits tokens incrementally
   - After each decode step, sends available tokens to TokenizerManager
   - TokenizerManager detokenizes and forwards to HTTP server

8. HTTP SERVER sends Server-Sent Events
   - Each token delta is sent as: data: {"text": "...", ...}
   - Final message: data: [DONE]
```

---

## Key Modules and Responsibilities

The following table lists all major modules under `python/sglang/` and their responsibilities.

| Module | Path | Description |
|--------|------|-------------|
| **sglang** | `python/sglang/` | Top-level package. Exports public API. |
| **srt** | `python/sglang/srt/` | SGLang Runtime (SRT). Core inference engine. |
| **srt.server_args** | `python/sglang/srt/server_args.py` | ServerArgs dataclass with all configuration parameters. |
| **srt.constants** | `python/sglang/srt/constants.py` | Global constants (GPU memory types, health check prefixes). |
| **srt.entrypoints** | `python/sglang/srt/entrypoints/` | API entry points (HTTP, gRPC, Engine). |
| **srt.entrypoints.http_server** | `.../entrypoints/http_server.py` | FastAPI HTTP server with OpenAI, native, and Ollama endpoints. |
| **srt.entrypoints.engine** | `.../entrypoints/engine.py` | Offline Engine class for batch inference. |
| **srt.managers** | `python/sglang/srt/managers/` | TokenizerManager and Scheduler process management. |
| **srt.managers.tokenizer_manager** | `.../managers/tokenizer_manager.py` | TokenizerManager: tokenization, multi-modal processing, request routing. |
| **srt.managers.scheduler** | `.../managers/scheduler.py` | Scheduler: batching, memory, prefix caching, continuous batching. |
| **srt.model_executor** | `python/sglang/srt/model_executor/` | Model execution on GPU. |
| **srt.model_runner** | `.../model_executor/model_runner.py` | ModelRunner: forward pass, CUDA graphs, weight loading. |
| **srt.layers** | `python/sglang/srt/layers/` | Neural network layer implementations (attention, MLP, MoE, etc.). |
| **srt.layers.attention** | `.../layers/attention/` | Attention backends (FlashInfer, FA3, FA4, Triton, etc.). |
| **srt.models** | `python/sglang/srt/models/` | Model architecture implementations (Llama, Qwen, DeepSeek, etc.). |
| **srt.mem_cache** | `python/sglang/srt/mem_cache/` | Memory cache implementations (radix tree, paged attention). |
| **srt.sampling** | `python/sglang/srt/sampling/` | Sampling logic (temperature, top-p, top-k, penalties, structured outputs). |
| **srt.tokenizer** | `python/sglang/srt/tokenizer/` | Tokenizer wrapper (HuggingFace, sentencepiece). |
| **srt.lora** | `python/sglang/srt/lora/` | LoRA adapter loading, management, and batching. |
| **srt.speculative** | `python/sglang/srt/speculative/` | Speculative decoding (EAGLE, Medusa, ngram). |
| **srt.disaggregation** | `python/sglang/srt/disaggregation/` | Prefill-decode disaggregation support. |
| **srt.constrained** | `python/sglang/srt/constrained/` | Constrained/structured output (xgrammar, outlines, llguidance). |
| **srt.distributed** | `python/sglang/srt/distributed/` | Distributed communication (NCCL, custom all-reduce). |
| **srt.model_loader** | `python/sglang/srt/model_loader/` | Model weight loading (safetensors, PyTorch, GGUF, etc.). |
| **srt.configs** | `python/sglang/srt/configs/` | Model configuration parsing and validation. |
| **srt.multimodal** | `python/sglang/srt/multimodal/` | Multi-modal input processing (images, audio, video). |
| **srt.multiplex** | `python/sglang/srt/multiplex/` | PD-Multiplexing support. |
| **srt.observability** | `python/sglang/srt/observability/` | Metrics, tracing, and observability. |
| **srt.parser** | `python/sglang/srt/parser/` | Reasoning and tool call parsing. |
| **srt.function_call** | `python/sglang/srt/function_call/` | Function/tool call parsing and handling. |
| **srt.grpc** | `python/sglang/srt/grpc/` | gRPC server implementation. |
| **srt.platforms** | `python/sglang/srt/platforms/` | Hardware platform abstraction (CUDA, ROCm, CPU, TPU, NPU, XPU). |
| **srt.hicache** | `python/sglang/srt/mem_cache/` | Hierarchical cache (HiCache) for host memory offloading. |
| **srt.checkpoint_engine** | `.../srt/checkpoint_engine/` | Checkpoint management for weight updates. |
| **srt.batch_overlap** | `python/sglang/srt/batch_overlap/` | Two-batch overlap scheduling. |
| **srt.compilation** | `python/sglang/srt/compilation/` | torch.compile integration. |
| **srt.connector** | `python/sglang/srt/connector/` | External storage connectors (RunAI, etc.). |
| **srt.dllm** | `python/sglang/srt/dllm/` | Diffusion LLM support. |
| **srt.elastic_ep** | `python/sglang/srt/elastic_ep/` | Elastic expert parallelism. |
| **srt.eplb** | `python/sglang/srt/eplb/` | Expert parallelism load balancing. |
| **srt.plugins** | `python/sglang/srt/plugins/` | Plugin system for extending SGLang. |
| **srt.ray** | `python/sglang/srt/ray/` | Ray integration for distributed deployment. |
| **srt.session** | `python/sglang/srt/session/` | Streaming session management. |
| **srt.state_capturer** | `.../srt/state_capturer/` | State capture for debugging. |
| **srt.utils** | `python/sglang/srt/utils/` | Utility functions (device detection, memory, etc.). |
| **srt.weight_sync** | `python/sglang/srt/weight_sync/` | Weight synchronization for distributed serving. |
| **lang** | `python/sglang/lang/` | Frontend language APIs (SGLang programming language). |
| **lang.backend** | `.../lang/backend/` | Backend connectors (RuntimeEndpoint, OpenAI, Anthropic, LiteLLM). |
| **benchmark** | `python/sglang/benchmark/` | Benchmarking tools. |
| **test** | `python/sglang/test/` | Test suites. |
| **jit_kernel** | `python/sglang/jit_kernel/` | JIT kernel compilation support. |
| **multimodal_gen** | `.../sglang/multimodal_gen/` | Multi-modal generation tools. |
| **cli** | `python/sglang/cli/` | Command-line interface. |
| **eval** | `python/sglang/eval/` | Evaluation tools. |

---

## Source Code Organization

Below is the complete directory tree of the SGLang Python package with descriptions for each
major component.

```
python/sglang/
|
|-- __init__.py                    # Public API exports (Engine, Runtime, gen, etc.)
|-- version.py                     # Version string
|-- global_config.py               # Global configuration singleton
|-- launch_server.py               # Server launcher entry point
|-- utils.py                       # General utilities (LazyImport, etc.)
|-- profiler.py                    # Profiling utilities
|-- check_env.py                   # Environment check utility
|-- _triton_stub.py                # Triton stub for non-CUDA platforms (macOS)
|-- _mps_stub.py                   # MPS stub for macOS
|-- kernel_api_logging.py          # Kernel API logging utilities
|-- README.md                      # Package README
|
|-- cli/                           # Command-line interface
|   |-- ...                        # CLI commands and argument parsing
|
|-- lang/                          # SGLang Frontend Language
|   |-- __init__.py                # Language module exports
|   |-- api.py                     # Core API functions (gen, select, function, etc.)
|   |-- choices.py                 # Choice selection strategies
|   |-- grammar.py                 # Grammar definitions
|   |-- interpreter.py             # Program interpreter
|   |-- ir.py                      # Intermediate representation
|   |-- tracer.py                  # Execution tracer
|   |-- backend/                   # Backend connectors
|   |   |-- runtime_endpoint.py    # SGLang Runtime endpoint
|   |   |-- openai.py              # OpenAI backend
|   |   |-- anthropic.py           # Anthropic backend
|   |   |-- litellm.py             # LiteLLM backend
|   |   |-- vertexai.py            # Vertex AI backend
|   |-- ...                        # Additional language support files
|
|-- srt/                           # SGLang Runtime (SRT) - Core Engine
|   |-- server_args.py             # ServerArgs dataclass (200+ parameters)
|   |-- constants.py               # Global constants
|   |-- io_struct.py               # Request/response wire protocol types
|   |-- sampling_params.py         # SamplingParams definition
|   |-- layers/                    # Neural network layer implementations
|   |   |-- attention/             # Attention mechanism backends
|   |   |   |-- triton_backend.py  # Triton attention kernel
|   |   |   |-- flashinfer_backend.py  # FlashInfer attention
|   |   |   |-- fa3_backend.py     # FlashAttention 3
|   |   |   |-- fa4_backend.py     # FlashAttention 4
|   |   |   |-- cutlass_mla_backend.py  # CUTLASS MLA
|   |   |   |-- flashmla_backend.py     # FlashMLA
|   |   |   |-- ...                # More attention backends
|   |   |-- moe/                   # Mixture of Experts layers
|   |   |   |-- ...                # MoE routing and computation
|   |   |-- radix_attention.py     # RadixAttention implementation
|   |   |-- sampler.py             # Token sampler
|   |   |-- logits_processor.py    # Logits processing
|   |   |-- ...                    # Other layers (embed, linear, norm, etc.)
|   |
|   |-- managers/                  # Process managers
|   |   |-- tokenizer_manager.py  # TokenizerManager process
|   |   |-- scheduler.py           # Scheduler process
|   |   |-- data_parallel_controller.py  # Data parallelism controller
|   |   |-- detokenizer_manager.py # Detokenizer process
|   |   |-- image_processor.py     # Image processing
|   |   |-- ...                    # Supporting managers
|   |
|   |-- model_executor/            # Model execution
|   |   |-- model_runner.py        # ModelRunner (GPU forward pass)
|   |   |-- forward_batch_info.py  # Forward batch metadata
|   |   |-- cuda_graph_runner.py   # CUDA graph capture and replay
|   |
|   |-- models/                    # Model architecture implementations
|   |   |-- llama.py               # LLaMA / Llama 2/3/4
|   |   |-- qwen.py                # Qwen / Qwen2 / Qwen2.5
|   |   |-- deepseek.py            # DeepSeek V2/V3/R1
|   |   |-- gemma.py               # Gemma 2/3
|   |   |-- mistral.py             # Mistral / Mixtral
|   |   |-- glm.py                 # GLM-4 / ChatGLM
|   |   |-- gpt2.py                # GPT-2 (base architecture)
|   |   |-- grok.py                # Grok
|   |   |-- exaone.py              # EXAONE
|   |   |-- ...                    # Many more model architectures
|   |
|   |-- mem_cache/                 # Memory and cache management
|   |   |-- radix_cache/           # RadixAttention radix tree cache
|   |   |-- paged_attention/       # Paged attention allocator
|   |   |-- hicache/               # Hierarchical cache (host memory)
|   |   |-- mamba_cache.py         # Mamba/SSM state cache
|   |   |-- ...                    # Cache utilities
|   |
|   |-- entrypoints/               # API entry points
|   |   |-- http_server.py         # FastAPI HTTP server
|   |   |-- engine.py              # Offline Engine
|   |   |-- grpc_server.py         # gRPC server (optional)
|   |   |-- openai/                # OpenAI-compatible API handlers
|   |   |-- ollama/                # Ollama-compatible API handlers
|   |   |-- ...                    # Other entry point support
|   |
|   |-- tokenizer/                 # Tokenizer implementations
|   |   |-- ...                    # HuggingFace tokenizer wrapper
|   |
|   |-- sampling/                  # Sampling logic
|   |   |-- sampling_params.py     # Sampling parameters
|   |   |-- custom_logit_processor.py  # Custom logit processor
|   |   |-- ...                    # Sampling utilities
|   |
|   |-- lora/                      # LoRA adapter support
|   |   |-- lora_registry.py       # LoRA adapter registry
|   |   |-- ...                    # LoRA loading and management
|   |
|   |-- speculative/               # Speculative decoding
|   |   |-- eagle/                 # EAGLE speculative decoding
|   |   |-- medusa/                # Medusa speculative decoding
|   |   |-- ngram/                 # N-gram speculative decoding
|   |   |-- ...                    # Speculative utilities
|   |
|   |-- constrained/               # Structured output constraints
|   |   |-- ...                    # xgrammar, outlines, llguidance backends
|   |
|   |-- distributed/               # Distributed communication
|   |   |-- ...                    # NCCL wrappers, custom all-reduce
|   |
|   |-- model_loader/              # Weight loading
|   |   |-- ...                    # safetensors, PyTorch, GGUF loaders
|   |
|   |-- configs/                   # Model configuration
|   |   |-- ...                    # Config parsing and validation
|   |
|   |-- multimodal/                # Multi-modal processing
|   |   |-- ...                    # Image, audio, video processors
|   |
|   |-- disaggregation/            # Prefill-decode disaggregation
|   |   |-- ...                    # Transfer backends, coordination
|   |
|   |-- platforms/                 # Hardware platform abstraction
|   |   |-- cuda_device.py         # NVIDIA CUDA platform
|   |   |-- amd_device.py          # AMD ROCm platform
|   |   |-- cpu_device.py          # Intel CPU platform
|   |   |-- tpu_device.py          # Google TPU platform
|   |   |-- npu_device.py          # Ascend NPU platform
|   |   |-- xpu_device.py          # Intel XPU platform
|   |
|   |-- observability/             # Observability and metrics
|   |   |-- ...                    # Prometheus metrics, tracing
|   |
|   |-- parser/                    # Output parsing
|   |   |-- reasoning_parser.py    # Reasoning/thinking content parser
|   |
|   |-- function_call/             # Tool/function call handling
|   |   |-- ...                    # Function call parsing
|   |
|   |-- utils/                     # Utility functions
|   |   |-- common.py              # Common utilities
|   |   |-- network.py             # Network utilities
|   |   |-- hf_transformers_utils.py  # HuggingFace patches
|   |   |-- ...                    # More utilities
|   |
|   |-- ...                        # Additional SRT submodules
|
|-- benchmark/                     # Benchmarking
|   |-- ...                        # Benchmark scripts and datasets
|
|-- test/                          # Test suites
|   |-- ...                        # Unit and integration tests
|
|-- jit_kernel/                    # JIT kernel compilation
|   |-- ...                        # Triton kernels, CUDA kernels
|
|-- multimodal_gen/                # Multi-modal generation
|   |-- ...                        # Generation tools and configs
|
|-- eval/                          # Evaluation
|   |-- ...                        # Evaluation benchmarks
```

---

## Public API Reference

The public API of SGLang is exported through `python/sglang/__init__.py`. The following symbols
are available via `import sglang as sgl` or `from sglang import ...`.

### Runtime Engine APIs

These are the primary APIs for running inference.

#### `sgl.Engine`

The offline inference engine for batch processing without an HTTP server.

```python
import sglang as sgl

# Create engine
llm = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    # All ServerArgs parameters are accepted here
)

# Synchronous generation
outputs = llm.generate(
    prompts=["Hello, world!", "What is AI?"],
    sampling_params={"temperature": 0.7, "max_new_tokens": 128}
)

# Asynchronous generation
outputs = await llm.async_generate(
    prompts=["Hello, world!"],
    sampling_params={"temperature": 0.7}
)

# Shutdown (releases GPU resources)
llm.shutdown()
```

Constructor accepts all `ServerArgs` parameters as keyword arguments.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model_path, **kwargs)` | Create engine with model and ServerArgs parameters |
| `generate` | `(prompts, sampling_params) -> list` | Synchronous batch generation |
| `async_generate` | `(prompts, sampling_params) -> list` | Async batch generation |
| `shutdown` | `()` | Release all resources and stop workers |

#### `sgl.ServerArgs`

Lazy-imported configuration dataclass. See the Server Configuration reference document for
complete parameter documentation.

```python
from sglang import ServerArgs

args = ServerArgs(model_path="meta-llama/Llama-3.1-8B-Instruct", tp_size=4)
```

### Frontend Language APIs

These APIs form the SGLang frontend programming language for composing LLM programs.

#### Prompt Primitive Functions

| API | Signature | Description |
|-----|-----------|-------------|
| `gen` | `(name, **kwargs)` | Generate text with optional constraints |
| `gen_int` | `(name, **kwargs)` | Generate an integer value |
| `gen_string` | `(name, **kwargs)` | Generate a string value |
| `select` | `(name, choices, **kwargs)` | Select from a list of choices |
| `image` | `(url)` | Embed an image into the prompt |
| `video` | `(url)` | Embed a video into the prompt |

#### Role Tag Functions

These functions insert role-specific markers into the prompt:

| API | Description |
|-----|-------------|
| `system` | Insert a system message |
| `system_begin` | Begin a system message block |
| `system_end` | End a system message block |
| `user` | Insert a user message |
| `user_begin` | Begin a user message block |
| `user_end` | End a user message block |
| `assistant` | Insert an assistant message |
| `assistant_begin` | Begin an assistant message block |
| `assistant_end` | End an assistant message block |

#### Control Flow

| API | Description |
|-----|-------------|
| `function` | Define a callable SGLang function |
| `Runtime` | Create a runtime for executing SGLang programs |
| `flush_cache` | Flush the KV-cache on the backend |
| `get_server_info` | Get server information |
| `set_default_backend` | Set the default backend for SGLang programs |
| `separate_reasoning` | Configure reasoning content separation |

#### Backend Connectors

| API | Description |
|-----|-------------|
| `RuntimeEndpoint` | Connect to an SGLang Runtime server |
| `OpenAI` | Connect to OpenAI API (lazy import) |
| `Anthropic` | Connect to Anthropic API (lazy import) |
| `LiteLLM` | Connect via LiteLLM (lazy import) |
| `VertexAI` | Connect to Google Vertex AI (lazy import) |

#### Choice Selection Strategies

| API | Description |
|-----|-------------|
| `greedy_token_selection` | Select highest probability token |
| `token_length_normalized` | Normalize by token length |
| `unconditional_likelihood_normalized` | Normalize by unconditional likelihood |

### Configuration

#### `global_config`

The global configuration singleton for SGLang frontend settings.

```python
from sglang import global_config

# Access configuration values
print(global_config.default_backend)
```

#### `__version__`

The installed SGLang version string.

```python
from sglang import __version__
print(__version__)  # e.g., "0.5.9"
```

### Constants

Defined in `python/sglang/srt/constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `GPU_MEMORY_TYPE_KV_CACHE` | `"kv_cache"` | GPU memory used for KV-cache |
| `GPU_MEMORY_TYPE_WEIGHTS` | `"weights"` | GPU memory used for model weights |
| `GPU_MEMORY_TYPE_CUDA_GRAPH` | `"cuda_graph"` | GPU memory used for CUDA graphs |
| `GPU_MEMORY_ALL_TYPES` | `[kv_cache, weights, cuda_graph]` | All GPU memory type categories |
| `HEALTH_CHECK_RID_PREFIX` | `"HEALTH_CHECK"` | Prefix for health check request IDs |

---

## Version History Highlights

| Version | Date | Key Features |
|---------|------|-------------|
| v0.5.9 | 2026 | Latest stable release |
| v0.4 | 2024-12 | Zero-overhead batch scheduler, cache-aware load balancer, faster structured outputs |
| v0.3 | 2024-09 | 7x faster DeepSeek MLA, 1.5x faster torch.compile, multi-image/video LLaVA-OneVision |
| v0.2 | 2024-07 | Faster Llama3 serving vs TensorRT-LLM, vLLM |
| v0.1 | 2024-01 | Initial release with RadixAttention (5x faster inference) |

---

## Related Documentation

- [Installation and Setup](./02-installation-setup.md)
- [Server Configuration Reference](./03-server-configuration.md)
- [API Reference](./04-api-reference.md)
