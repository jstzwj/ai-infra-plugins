# vLLM Overview and Architecture

## Table of Contents

- [Project Overview](#project-overview)
- [High-Level Architecture](#high-level-architecture)
- [Core Abstractions](#core-abstractions)
- [Data Flow: Request to Response](#data-flow-request-to-response)
- [Key Modules and Responsibilities](#key-modules-and-responsibilities)
- [V0 vs V1 Architecture](#v0-vs-v1-architecture)
- [Supported Hardware Platforms](#supported-hardware-platforms)
- [Public API Reference](#public-api-reference)
  - [Module: vllm.\_\_init\_\_](#module-vllm__init__)
  - [Module: vllm.sequence](#module-vllmsequence)
  - [Module: vllm.outputs](#module-vllmoutputs)
  - [Module: vllm.exceptions](#module-vllmexceptions)
  - [Module: vllm.forward\_context](#module-vllmforward_context)
  - [Module: vllm.logger](#module-vllmlogger)
  - [Module: vllm.envs](#module-vllmenvs)
  - [Module: vllm.tasks](#module-vllmtasks)
  - [Module: vllm.scalar\_type](#module-vllmscalar_type)
  - [Module: vllm.version](#module-vllmversion)

---

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). Originally developed at UC Berkeley's Sky Computing Lab, it has grown into one of the most active open-source AI projects with over 2000 contributors.

### Mission

vLLM aims to provide **easy, fast, and cheap LLM serving for everyone** through:

- **State-of-the-art serving throughput** via PagedAttention, continuous batching, chunked prefill, and prefix caching
- **Flexible model execution** with CUDA/HIP graphs, torch.compile, and optimized kernels
- **Comprehensive quantization support**: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, and more
- **Optimized attention kernels** including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- **Speculative decoding** with n-gram, suffix, EAGLE, and DFlash
- **Disaggregated prefill, decode, and encode** serving
- **Seamless HuggingFace integration** supporting 200+ model architectures
- **Tensor, pipeline, data, expert, and context parallelism** for distributed inference
- **Structured outputs**, tool calling, and reasoning parsers
- **OpenAI-compatible API server**, Anthropic Messages API, and gRPC support

### Project Metadata

| Property | Value |
|---|---|
| **Name** | vllm |
| **License** | Apache-2.0 |
| **Python** | >=3.10, <3.15 |
| **PyTorch** | 2.11.0 |
| **Homepage** | https://github.com/vllm-project/vllm |
| **Documentation** | https://docs.vllm.ai |
| **Entry Point** | `vllm.entrypoints.cli.main:main` |

---

## High-Level Architecture

```
                          +---------------------+
                          |    Client / API     |
                          | (OpenAI / gRPC /    |
                          |  Python SDK / LLM)  |
                          +----------+----------+
                                     |
                          +----------v----------+
                          |   Entrypoints       |
                          | (serve / generate /  |
                          |  encode / score)     |
                          +----------+----------+
                                     |
                          +----------v----------+
                          |   AsyncLLM /        |
                          |   LLMEngine         |
                          | (API Coordinator)    |
                          +----------+----------+
                                     |
                    +----------------+----------------+
                    |                                 |
          +---------v----------+          +-----------v-----------+
          |  InputProcessor    |          |  OutputProcessor      |
          |  (Tokenizer + MM   |          |  (Detokenizer +       |
          |   Preprocessing)   |          |   Logprobs + Stats)   |
          +---------+----------+          +-----------+-----------+
                    |                                 ^
                    v                                 |
          +---------+----------+          +-----------+-----------+
          |  EngineCoreClient  |<--------->|   EngineCore          |
          |  (IPC / InProc)    |          |   (Scheduler +        |
          +--------------------+          |    ModelExecutor)      |
                                          +---+-------+-------+---+
                                              |       |       |
                                    +---------+  +----+-----+ +--------+
                                    | Scheduler | | Model    | | KV     |
                                    |           | | Runner   | | Cache  |
                                    +-----------+ +----------+ +--------+
```

### Component Overview

```
+-------------------------------------------------------------------+
|                        vLLM Engine Stack                          |
|                                                                   |
|  +-------------------------------------------------------------+ |
|  |                      Entrypoints Layer                       | |
|  |  - OpenAI API Server (FastAPI)                               | |
|  |  - Anthropic Messages API                                    | |
|  |  - gRPC Server                                               | |
|  |  - Python LLM class (offline)                                | |
|  +-------------------------------------------------------------+ |
|                                                                   |
|  +-------------------------------------------------------------+ |
|  |                    Engine Layer                              | |
|  |  - AsyncLLM (async, streaming)                               | |
|  |  - LLMEngine (sync, batch)                                   | |
|  |  - InputProcessor (tokenize, preprocess MM)                  | |
|  |  - OutputProcessor (detokenize, logprobs, stats)             | |
|  +-------------------------------------------------------------+ |
|                                                                   |
|  +-------------------------------------------------------------+ |
|  |                    Core Layer                                | |
|  |  - EngineCore (scheduler + executor loop)                    | |
|  |  - SchedulerInterface (scheduling policy)                    | |
|  |  - KV Cache Manager (block allocation, prefix caching)       | |
|  |  - Structured Output Manager (grammar, JSON schema)          | |
|  +-------------------------------------------------------------+ |
|                                                                   |
|  +-------------------------------------------------------------+ |
|  |                   Executor Layer                             | |
|  |  - MultiprocExecutor / UniprocExecutor                       | |
|  |  - RayExecutor / MPExecutor                                  | |
|  |  - ModelRunner (forward pass)                                | |
|  |  - Worker (GPU/CPU/TPU)                                      | |
|  +-------------------------------------------------------------+ |
|                                                                   |
|  +-------------------------------------------------------------+ |
|  |                  Model Layer                                 | |
|  |  - ModelRegistry (200+ architectures)                        | |
|  |  - Attention backends (FlashInfer, FA2, FA3, etc.)           | |
|  |  - Quantization layers (FP8, GPTQ, AWQ, GGUF, etc.)         | |
|  |  - MoE kernels (DeepGEMM, CUTLASS, Triton)                  | |
|  +-------------------------------------------------------------+ |
+-------------------------------------------------------------------+
```

---

## Core Abstractions

### IntermediateTensors

Pipeline parallelism intermediate data structure for passing hidden states and residuals between stages.

```python
@dataclass
class IntermediateTensors:
    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None

    def __getitem__(self, key: str | slice) -> torch.Tensor | IntermediateTensors
    def __setitem__(self, key: str, value: torch.Tensor) -> None
    def items(self) -> ItemsView[str, torch.Tensor]
    def __len__(self) -> int
    def __eq__(self, other: object) -> bool
    def __repr__(self) -> str

    @staticmethod
    def empty_like(intermediate_tensors: IntermediateTensors) -> IntermediateTensors
```

### CompletionOutput

The output data of one completion output of a request.

```python
@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: Sequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    routed_experts: np.ndarray | None = None  # [seq_len, layer_num, topk]
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    lora_request: LoRARequest | None = None

    def finished(self) -> bool
    def __repr__(self) -> str
```

### PoolingOutput

Output data for pooling (embedding/classification) operations.

```python
@dataclass
class PoolingOutput:
    data: torch.Tensor

    def __repr__(self) -> str
    def __eq__(self, other: object) -> bool
```

### RequestOutput

The output data of a completion request to the LLM.

```python
class RequestOutput:
    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestStateStats | None = None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None

    # Attributes
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    prompt_logprobs: PromptLogprobs | None
    outputs: list[CompletionOutput]
    finished: bool
    metrics: RequestStateStats | None
    lora_request: LoRARequest | None
    encoder_prompt: str | None
    encoder_prompt_token_ids: list[int] | None
    num_cached_tokens: int | None
    kv_transfer_params: dict[str, Any] | None

    def add(self, next_output: RequestOutput, aggregate: bool) -> None
    def __repr__(self) -> str
```

### PoolingRequestOutput

Generic output for pooling requests (embeddings, classification, scoring).

```python
class PoolingRequestOutput(Generic[_O]):
    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
    )

    request_id: str
    prompt_token_ids: list[int]
    num_cached_tokens: int
    finished: bool
    outputs: _O

    def __repr__(self) -> str
```

### EmbeddingOutput

```python
@dataclass
class EmbeddingOutput:
    embedding: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> EmbeddingOutput

    @property
    def hidden_size(self) -> int

    def __repr__(self) -> str
```

### ClassificationOutput

```python
@dataclass
class ClassificationOutput:
    probs: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> ClassificationOutput

    @property
    def num_classes(self) -> int

    def __repr__(self) -> str
```

### ScoringOutput

```python
@dataclass
class ScoringOutput:
    score: float

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> ScoringOutput

    def __repr__(self) -> str
```

### EmbeddingRequestOutput

```python
class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput) -> EmbeddingRequestOutput
```

### ClassificationRequestOutput

```python
class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput) -> ClassificationRequestOutput
```

### ScoringRequestOutput

```python
class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput) -> ScoringRequestOutput
```

---

## Data Flow: Request to Response

### Generation Request Flow

```
1. Client submits request (text/token_ids + SamplingParams)
       |
2. Entrypoint (API Server / LLM class) receives request
       |
3. InputProcessor.process_inputs()
   - Tokenize text (if needed)
   - Process multimodal inputs
   - Validate parameters
   - Create EngineCoreRequest
       |
4. EngineCoreClient.add_request()
   - Send EngineCoreRequest to EngineCore via IPC (ZMQ) or in-process
       |
5. EngineCore.add_request()
   - Create internal Request object
   - Compute block hashes for prefix caching
   - Add to Scheduler's waiting queue
       |
6. Scheduler.schedule()
   - Select requests for next batch
   - Allocate KV cache blocks
   - Determine prefill vs decode
   - Create SchedulerOutput
       |
7. ModelExecutor.execute_model()
   - Run forward pass on GPU/TPU/CPU
   - Sample tokens (if decode)
   - Return ModelRunnerOutput
       |
8. Scheduler.update_from_output()
   - Update request status
   - Check finish conditions
   - Return EngineCoreOutputs
       |
9. OutputProcessor.process_outputs()
   - Detokenize token IDs to text
   - Process logprobs
   - Create RequestOutput
       |
10. Client receives streaming/completed output
```

### Pooling Request Flow

```
1-5. Same as generation flow
       |
6. Scheduler schedules encoder + pooling
       |
7. ModelExecutor runs encoder + pooler forward pass
       |
8. OutputProcessor creates PoolingRequestOutput
   - EmbeddingOutput, ClassificationOutput, or ScoringOutput
       |
9. Client receives result
```

---

## Key Modules and Responsibilities

### Module Map

| Module | Responsibility |
|--------|---------------|
| `vllm.engine` | Engine coordination (LLMEngine, AsyncLLMEngine) |
| `vllm.v1.engine` | V1 engine implementation (Core, AsyncLLM, Coordinator) |
| `vllm.v1.core` | Scheduler, KV cache management |
| `vllm.v1.worker` | GPU worker, model runner |
| `vllm.v1.executor` | Execution backends (multiproc, ray, uni) |
| `vllm.model_executor` | Model loading, attention, quantization |
| `vllm.entrypoints` | API servers (OpenAI, gRPC, CLI) |
| `vllm.inputs` | Input processing, multimodal |
| `vllm.outputs` | Output types (Completion, Pooling, Embedding) |
| `vllm.sampling_params` | Sampling configuration |
| `vllm.pooling_params` | Pooling configuration |
| `vllm.platforms` | Hardware platform abstraction |
| `vllm.config` | Configuration dataclasses |
| `vllm.distributed` | Distributed communication (TP, PP, DP, EP) |
| `vllm.lora` | LoRA adapter management |
| `vllm.tracing` | OpenTelemetry tracing |
| `vllm.transformers_utils` | HuggingFace model/tokenizer utilities |

---

## V0 vs V1 Architecture

As of the current version, V1 is the default and only architecture. The V0 engine classes (`vllm.engine.llm_engine.LLMEngine` and `vllm.engine.async_llm_engine.AsyncLLMEngine`) are now thin aliases that delegate to their V1 counterparts.

### V1 Architecture (Current)

```
                    +--------------------+
                    |     AsyncLLM       |
                    |  (or LLMEngine)    |
                    +----+--------+------+
                         |        |
              +----------+   +----+-----------+
              |               |                |
    +---------v-----+  +------v------+  +------v------+
    | InputProcessor |  | OutputProc  |  | CoreClient  |
    +---------+------+  +-------------+  +------+------+
              |                                  |
              |                    +-------------v-------------+
              |                    |     EngineCore            |
              |                    |  +-------+  +----------+ |
              +------------------->|  | Sched  |  | Executor | |
                                   |  +-------+  +----------+ |
                                   +---------------------------+
```

**Key V1 characteristics:**

1. **Multiprocessing by default**: EngineCore runs in a separate process, communicating via ZMQ IPC
2. **Separated concerns**: InputProcessor (tokenizer + MM), OutputProcessor (detokenizer + stats), EngineCore (scheduler + executor)
3. **EngineCoreClient**: Abstract IPC layer with `InprocClient`, `SyncMPClient`, `AsyncMPClient`
4. **Data parallelism**: Built-in load balancer across DP ranks
5. **Structured outputs**: Native grammar-constrained generation via xgrammar/outlines
6. **Streaming inputs**: Multi-turn streaming session support
7. **torch.compile**: Automatic kernel generation and graph-level optimizations
8. **CUDA graphs**: Full and piecewise CUDA graph support
9. **Speculative decoding**: n-gram, EAGLE, suffix-based

---

## Supported Hardware Platforms

### Platform Detection

vLLM uses a plugin-based platform detection system. The `current_platform` singleton is lazily resolved from `vllm.platforms`.

```python
class PlatformEnum(enum.Enum):
    CUDA = enum.auto()      # NVIDIA GPUs
    ROCM = enum.auto()      # AMD GPUs (ROCm)
    TPU = enum.auto()       # Google TPUs
    XPU = enum.auto()       # Intel XPU (Gaudi, Arc)
    CPU = enum.auto()       # x86, ARM, PowerPC, s390x, RISC-V
    OOT = enum.auto()       # Out-of-tree plugins
    UNSPECIFIED = enum.auto()
```

### Platform Capabilities

Each platform implements the `Platform` abstract base class with methods for:

- Device name and memory queries
- Device capability detection (compute capability)
- Attention backend selection
- Distributed communication backend
- Custom all-reduce support
- Device control (sleep, wake up)
- Environment setup

### Supported Architectures

| Platform | Hardware | Key Requirements |
|----------|----------|-----------------|
| CUDA | NVIDIA GPUs (H100, A100, L40S, etc.) | CUDA 12.x+, compute capability 7.0+ |
| ROCm | AMD GPUs (MI300X, MI250, etc.) | ROCm 6.x+, hip runtime |
| TPU | Google TPU v4, v5, v5e, v6e | libtpu, JAX/XLA runtime |
| XPU | Intel Gaudi, Arc | oneAPI, Level Zero |
| CPU | x86_64, ARM, PowerPC, s390x, RISC-V | AVX2 or AVX512 (x86) |

---

## Public API Reference

### Module: vllm.\_\_init\_\_

The top-level vLLM module. Uses lazy imports for performance.

**Public attributes (lazy-loaded):**

| Name | Type | Source |
|------|------|--------|
| `LLM` | class | `vllm.entrypoints.llm:LLM` |
| `LLMEngine` | class | `vllm.engine.llm_engine:LLMEngine` |
| `AsyncLLMEngine` | class | `vllm.engine.async_llm_engine:AsyncLLMEngine` |
| `EngineArgs` | dataclass | `vllm.engine.arg_utils:EngineArgs` |
| `AsyncEngineArgs` | dataclass | `vllm.engine.arg_utils:AsyncEngineArgs` |
| `SamplingParams` | class | `vllm.sampling_params:SamplingParams` |
| `PoolingParams` | class | `vllm.pooling_params:PoolingParams` |
| `RequestOutput` | class | `vllm.outputs:RequestOutput` |
| `CompletionOutput` | dataclass | `vllm.outputs:CompletionOutput` |
| `PoolingOutput` | dataclass | `vllm.outputs:PoolingOutput` |
| `PoolingRequestOutput` | class | `vllm.outputs:PoolingRequestOutput` |
| `EmbeddingOutput` | dataclass | `vllm.outputs:EmbeddingOutput` |
| `EmbeddingRequestOutput` | class | `vllm.outputs:EmbeddingRequestOutput` |
| `ClassificationOutput` | dataclass | `vllm.outputs:ClassificationOutput` |
| `ClassificationRequestOutput` | class | `vllm.outputs:ClassificationRequestOutput` |
| `ScoringOutput` | dataclass | `vllm.outputs:ScoringOutput` |
| `ScoringRequestOutput` | class | `vllm.outputs:ScoringRequestOutput` |
| `PromptType` | type alias | `vllm.inputs:PromptType` |
| `TextPrompt` | class | `vllm.inputs:TextPrompt` |
| `TokensPrompt` | class | `vllm.inputs:TokensPrompt` |
| `ModelRegistry` | class | `vllm.model_executor.models:ModelRegistry` |
| `initialize_ray_cluster` | function | `vllm.v1.executor.ray_utils:initialize_ray_cluster` |
| `__version__` | str | Version string |
| `__version_tuple__` | tuple | Version tuple |

---

### Module: vllm.sequence

Provides `IntermediateTensors` for pipeline parallelism.

#### IntermediateTensors

```python
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, contains hidden states
    and residuals to be sent to the next stage."""

    tensors: dict[str, torch.Tensor]
    kv_connector_output: KVConnectorOutput | None

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        kv_connector_output: KVConnectorOutput | None = None,
    ) -> None
    # NOTE: Manually defined __init__ so Dynamo knows the source file.

    def __getitem__(self, key: str | slice)
        # str key: returns self.tensors[key]
        # slice key: returns new IntermediateTensors with sliced tensors

    def __setitem__(self, key: str, value: torch.Tensor)
        # Sets self.tensors[key] = value

    def items(self) -> ItemsView[str, torch.Tensor]

    def __len__(self) -> int
        # Returns len(self.tensors)

    def __eq__(self, other: object) -> bool
        # Compares tensor keys and values

    def __repr__(self) -> str

    @staticmethod
    def empty_like(
        intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors
        # Creates new IntermediateTensors with empty tensors of same shape
```

---

### Module: vllm.outputs

All output data classes for vLLM requests.

#### STREAM_FINISHED

```python
STREAM_FINISHED: RequestOutput
# Sentinel to indicate request is finished (used with streaming inputs)
```

#### CompletionOutput

```python
@dataclass
class CompletionOutput:
    """The output data of one completion output of a request."""

    index: int
        # The index of the output in the request (for n > 1)

    text: str
        # The generated output text

    token_ids: Sequence[int]
        # The token IDs of the generated output text

    cumulative_logprob: float | None
        # The cumulative log probability of the generated output text

    logprobs: SampleLogprobs | None
        # The log probabilities of the top probability words at each
        # position if the logprobs are requested

    routed_experts: np.ndarray | None = None
        # Shape: [seq_len, layer_num, topk]
        # Expert routing information for MoE models

    finish_reason: str | None = None
        # The reason why the sequence is finished

    stop_reason: int | str | None = None
        # The stop string or token id that caused the completion to stop

    lora_request: LoRARequest | None = None
        # The LoRA request used to generate the output

    def finished(self) -> bool
        # Returns True if finish_reason is not None
```

#### PoolingOutput

```python
@dataclass
class PoolingOutput:
    """The output data of one pooling output of a request."""

    data: torch.Tensor
        # The extracted hidden states / embedding / classification scores
```

#### RequestOutput

```python
class RequestOutput:
    """The output data of a completion request to the LLM."""

    def __init__(
        self,
        request_id: str,
            # Unique ID of the request
        prompt: str | None,
            # Prompt string (decoder input for enc/dec models)
        prompt_token_ids: list[int] | None,
            # Token IDs of the prompt (decoder input for enc/dec models)
        prompt_logprobs: PromptLogprobs | None,
            # Log probabilities per prompt token
        outputs: list[CompletionOutput],
            # Output sequences of the request
        finished: bool,
            # Whether the whole request is finished
        metrics: RequestStateStats | None = None,
            # Metrics associated with the request
        lora_request: LoRARequest | None = None,
            # LoRA request used for generation
        encoder_prompt: str | None = None,
            # Encoder prompt string (None for decoder-only)
        encoder_prompt_token_ids: list[int] | None = None,
            # Token IDs of encoder prompt (None for decoder-only)
        num_cached_tokens: int | None = None,
            # Number of tokens with prefix cache hit
        *,
        kv_transfer_params: dict[str, Any] | None = None,
            # Params for remote KV transfer
        **kwargs: Any,
            # Forward compatibility kwargs
    ) -> None

    def add(self, next_output: RequestOutput, aggregate: bool) -> None
        # Merge subsequent RequestOutput into this one
        # If aggregate=True, merges outputs with same index
        # If aggregate=False, replaces the output with the new one
```

#### PoolingRequestOutput

```python
class PoolingRequestOutput(Generic[_O]):
    """Generic output for pooling requests."""

    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
    ) -> None
```

#### EmbeddingOutput

```python
@dataclass
class EmbeddingOutput:
    """Embedding vector output."""

    embedding: list[float]
        # The embedding vector (length = hidden_size)

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> EmbeddingOutput
        # Converts PoolingOutput to EmbeddingOutput
        # Raises ValueError if pooled_data is not 1-D

    @property
    def hidden_size(self) -> int
        # Returns len(self.embedding)
```

#### EmbeddingRequestOutput

```python
class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput) -> EmbeddingRequestOutput
```

#### ClassificationOutput

```python
@dataclass
class ClassificationOutput:
    """Classification probability output."""

    probs: list[float]
        # Probability vector (length = num_classes)

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> ClassificationOutput

    @property
    def num_classes(self) -> int
```

#### ClassificationRequestOutput

```python
class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput) -> ClassificationRequestOutput
```

#### ScoringOutput

```python
@dataclass
class ScoringOutput:
    """Similarity score output (scalar)."""

    score: float

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> ScoringOutput
```

#### ScoringRequestOutput

```python
class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput) -> ScoringRequestOutput
```

---

### Module: vllm.exceptions

Custom exception classes for vLLM.

#### VLLMValidationError

```python
class VLLMValidationError(ValueError):
    """vLLM-specific validation error for request validation failures."""

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: Any = None,
    ) -> None

    parameter: str | None
        # The parameter name that failed validation

    value: Any
        # The value that was rejected

    def __str__(self) -> str
        # Returns message with optional parameter and value info
```

#### VLLMNotFoundError

```python
class VLLMNotFoundError(Exception):
    """vLLM-specific NotFoundError."""
    pass
```

#### LoRAAdapterNotFoundError

```python
class LoRAAdapterNotFoundError(VLLMNotFoundError):
    """Exception raised when a LoRA adapter is not found."""

    message: str

    def __init__(self, lora_name: str, lora_path: str) -> None
    def __str__(self) -> str
```

---

### Module: vllm.forward_context

Provides the forward context management for model execution.

#### BatchDescriptor

```python
@dataclass(frozen=True)
class BatchDescriptor:
    """Batch descriptor for CUDA graph dispatching."""

    num_tokens: int
        # Number of tokens in the batch

    num_reqs: int | None = None
        # Number of requests (None for PIECEWISE CUDA graphs)

    uniform: bool = False
        # True if all requests have the same number of tokens

    has_lora: bool = False
        # Whether this batch has active LoRA adapters

    num_active_loras: int = 0
        # Number of distinct active LoRA adapters
```

#### DPMetadata

```python
@dataclass
class DPMetadata:
    """Data parallel metadata for the forward context."""

    num_tokens_across_dp_cpu: torch.Tensor
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> DPMetadata

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int)
        # Context manager for setting local_sizes with SP

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None

    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor
        # Cumulative tokens across sequence parallel ranks
```

#### ForwardContext

```python
@dataclass
class ForwardContext:
    """Forward context for model execution."""

    no_compile_layers: dict[str, Any]
        # Static forward context from compilation config

    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
        # Attention metadata per layer (or list for DBO)

    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
        # Slot mapping for KV cache writes

    dp_metadata: DPMetadata | None = None
        # Data parallel metadata

    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
        # CUDA graph mode: FULL, PIECEWISE, or NONE

    batch_descriptor: BatchDescriptor | None = None
        # Batch descriptor for CUDA graph

    ubatch_slices: UBatchSlices | None = None
        # Microbatch slices for DBO

    skip_compiled: bool = False
        # If True, bypass compiled model call

    all_moe_layers: list[str] | None = None
        # List of MoE layer names for torch.compile cold start

    moe_layer_index: int = 0
        # Counter for MoE layer string access

    additional_kwargs: dict[str, Any] = field(default_factory=dict)
        # Additional kwargs for the forward pass
```

#### Module-level Functions

```python
def get_forward_context() -> ForwardContext
    # Get the current forward context
    # Raises AssertionError if not set

def is_forward_context_available() -> bool
    # Check if forward context is available

def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    additional_kwargs: dict[str, Any] | None = None,
    skip_compiled: bool = False,
) -> ForwardContext
    # Factory function to create a ForwardContext

@contextmanager
def override_forward_context(forward_context: ForwardContext | None)
    # Context manager to override the current forward context

@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
)
    # Context manager that stores the current forward context
    # Injects common logic for every model forward pass
```

---

### Module: vllm.logger

Logging configuration and utilities for vLLM.

#### LogScope

```python
LogScope = Literal["process", "global", "local"]
# "process": Log in current process (always)
# "global": Log only on global first rank
# "local": Log only on local first rank
```

#### init_logger

```python
def init_logger(name: str) -> _VllmLogger
    # Get a vLLM logger with patched methods (debug_once, info_once, warning_once)
```

#### _VllmLogger (Extended Logger)

```python
class _VllmLogger(Logger):
    def debug_once(self, msg: str, *args: Hashable, scope: LogScope = "local") -> None
        # Log debug message only once (subsequent identical messages dropped)

    def info_once(self, msg: str, *args: Hashable, scope: LogScope = "local") -> None
        # Log info message only once

    def warning_once(self, msg: str, *args: Hashable, scope: LogScope = "local") -> None
        # Log warning message only once
```

#### suppress_logging

```python
@contextmanager
def suppress_logging(level: int = logging.INFO) -> Generator[None, Any, None]
    # Context manager to temporarily suppress logging
```

#### current_formatter_type

```python
def current_formatter_type(logger: Logger) -> Literal["color", "newline", None]
    # Returns the formatter type used by the logger
```

#### enable_trace_function_call

```python
def enable_trace_function_call(log_file_path: str, root_dir: str | None = None) -> None
    # Enable tracing of every function call under root_dir
    # Useful for debugging hangs or crashes
    # Note: Thread-level, slows down code significantly
```

#### Logging Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CONFIGURE_LOGGING` | `1` | Whether vLLM configures logging |
| `VLLM_LOGGING_LEVEL` | `INFO` | Logging level |
| `VLLM_LOGGING_PREFIX` | `""` | Prefix for all log messages |
| `VLLM_LOGGING_STREAM` | `ext://sys.stdout` | Output stream |
| `VLLM_LOGGING_CONFIG_PATH` | None | Path to custom logging config JSON |
| `VLLM_LOGGING_COLOR` | `auto` | Color mode: auto, 1, or 0 |
| `NO_COLOR` | `0` | Standard unix color disable |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | Stats logging interval in seconds |
| `VLLM_TRACE_FUNCTION` | `0` | Enable function call tracing |

---

### Module: vllm.envs

Environment variable management for vLLM. All environment variables are accessed through lazy-evaluated callables.

#### Key Environment Variables (Categorized)

**Installation Time:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TARGET_DEVICE` | `cuda` | Target device: cuda, rocm, cpu, tpu, xpu |
| `VLLM_MAIN_CUDA_VERSION` | `13.0` | Main CUDA version |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `highest` | PyTorch float32 matmul precision |
| `VLLM_BATCH_INVARIANT` | `0` | Batch-invariant mode (requires compute >= 9.0) |
| `MAX_JOBS` | None | Maximum parallel compilation jobs |
| `NVCC_THREADS` | None | Number of nvcc threads |
| `VLLM_USE_PRECOMPILED` | `0` | Use precompiled binaries |
| `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX` | `0` | Skip +precompiled version suffix |
| `VLLM_DOCKER_BUILD_CONTEXT` | `0` | Mark Docker build context |
| `CMAKE_BUILD_TYPE` | None | CMake build type |
| `VERBOSE` | `0` | Verbose install logs |

**Runtime Core:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST_IP` | `""` | Node IP for multi-node |
| `VLLM_PORT` | None | Communication port |
| `VLLM_RPC_BASE_PATH` | temp dir | IPC path for frontend-backend |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `60` | Engine iteration timeout |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` | Engine startup timeout |
| `VLLM_API_KEY` | None | API server key |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | Worker multiprocessing method |
| `VLLM_MAX_N_SEQUENCES` | `16384` | Maximum sequences |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `1` | Enable V1 multiprocessing |
| `VLLM_RPC_TIMEOUT` | `10000` | RPC timeout (ms) |

**Logging:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `-1` | Batch size log interval |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | Stats log interval (seconds) |
| `VLLM_TRACE_FUNCTION` | `0` | Function call tracing |

**Compilation:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_AOT_COMPILE` | auto | AOT compilation |
| `VLLM_USE_BYTECODE_HOOK` | `1` | Bytecode hook |
| `VLLM_FORCE_AOT_LOAD` | `0` | Force AOT load from disk |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | auto | Mega AOT artifact loading |
| `VLLM_DISABLE_COMPILE_CACHE` | `0` | Disable compile cache |
| `VLLM_USE_STANDALONE_COMPILE` | `1` | Standalone compile |
| `VLLM_ENABLE_PREGRAD_PASSES` | `1` | Enable pre-grad passes |

**CPU Backend:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CPU_KVCACHE_SPACE` | None | CPU KV cache space (bytes) |
| `VLLM_CPU_OMP_THREADS_BIND` | `auto` | OpenMP thread binding |
| `VLLM_CPU_SGL_KERNEL` | `0` | SGL kernels for small batch |
| `VLLM_CPU_ATTN_SPLIT_KV` | `1` | Attention split KV |
| `VLLM_ZENTORCH_WEIGHT_PREPACK` | `1` | ZenDNN weight prepacking |

**GPU/CUDA:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | FlashInfer top-k/top-p sampler |
| `VLLM_USE_DEEP_GEMM` | `1` | DeepGEMM kernels |
| `VLLM_MOE_USE_DEEP_GEMM` | `1` | DeepGEMM for MoE |
| `VLLM_SKIP_P2P_CHECK` | `0` | Skip P2P check |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `0` | CUDA graph garbage collection |

**ROCm:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ROCM_USE_AITER` | `0` | AITER kernels |
| `VLLM_ROCM_FP8_PADDING` | `1` | FP8 padding |
| `VLLM_ROCM_MOE_PADDING` | `1` | MoE padding |

**Media/Multimodal:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE_FETCH_TIMEOUT` | `5` | Image fetch timeout (seconds) |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `30` | Video fetch timeout (seconds) |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `10` | Audio fetch timeout (seconds) |
| `VLLM_MEDIA_CACHE_MAX_SIZE_MB` | `5120` | Media cache max size |
| `VLLM_MM_HASHER_ALGORITHM` | `blake3` | Hash algorithm |

**Distributed:**

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DP_RANK` | `0` | Data parallel rank |
| `VLLM_DP_SIZE` | `1` | Data parallel size |
| `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` | `1` | Ray V2 executor |

#### Helper Functions

```python
def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]
    # Create a lambda that validates env var against allowed choices

def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]
    # Validate comma-separated env var values against choices

def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]
    # Same as env_list_with_choices but returns a set

def get_vllm_port() -> int | None
    # Get port from VLLM_PORT env var
    # Raises ValueError if port is a URI

def get_env_or_set_default(
    env_name: str,
    default_factory: Callable[[], str],
) -> Callable[[], str]
    # Returns env var value if set, or generates and sets a default

def maybe_convert_int(value: str | None) -> int | None
def maybe_convert_bool(value: str | None) -> bool | None
def maybe_convert_json_str_or_file(value: str | None) -> dict[str, Any] | None

def disable_compile_cache() -> bool
def use_aot_compile() -> bool
def use_mega_aot_artifact() -> bool
```

---

### Module: vllm.tasks

Task type definitions for vLLM.

```python
# Generation tasks
GenerationTask = Literal["generate", "transcription", "realtime"]
GENERATION_TASKS: tuple[GenerationTask, ...] = ("generate", "transcription", "realtime")

# Pooling tasks
PoolingTask = Literal[
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
    "embed&token_classify",
]
POOLING_TASKS: tuple[PoolingTask, ...]

# Score types
ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]
SCORE_TYPE_MAP: dict[PoolingTask, ScoreType] = {
    "embed": "bi-encoder",
    "classify": "cross-encoder",
    "token_embed": "late-interaction",
}

# Frontend tasks
FrontendTask = Literal["render"]
FRONTEND_TASKS: tuple[FrontendTask, ...] = ("render",)

# Combined
SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]
```

---

### Module: vllm.scalar_type

Custom scalar type representation for quantization, mirroring the C++ `ScalarType` class.

#### NanRepr

```python
class NanRepr(Enum):
    NONE = 0           # NaNs not supported
    IEEE_754 = 1       # Standard IEEE 754 NaN encoding
    EXTD_RANGE_MAX_MIN = 2  # NaN = Exp all 1s, mantissa all 1s
```

#### ScalarType

```python
@dataclass(frozen=True)
class ScalarType:
    """Represents floating point and integer types, including sub-byte types."""

    exponent: int
        # Bits in exponent (0 for integer types)

    mantissa: int
        # Bits in mantissa (float) or magnitude (integer, excluding sign)

    signed: bool
        # Whether type has a sign bit

    bias: int
        # Bias: stored_value = value + bias

    _finite_values_only: bool = False
        # Whether infs are not supported

    nan_repr: NanRepr = NanRepr.IEEE_754
        # How NaNs are represented

    # Properties
    @property
    def id(self) -> int
        # Integer ID for passing to PyTorch custom ops (cached)

    @property
    def size_bits(self) -> int
        # Total size in bits = exponent + mantissa + int(signed)

    def min(self) -> int | float
        # Minimum representable value (accounting for bias)

    def max(self) -> int | float
        # Maximum representable value (accounting for bias)

    def is_signed(self) -> bool
    def is_floating_point(self) -> bool
    def is_integer(self) -> bool
    def has_bias(self) -> bool
    def has_infs(self) -> bool
    def has_nans(self) -> bool
    def is_ieee_754(self) -> bool

    # Convenience Constructors
    @classmethod
    def int_(cls, size_bits: int, bias: int | None) -> ScalarType
        # Create signed integer type (size_bits includes sign bit)

    @classmethod
    def uint(cls, size_bits: int, bias: int | None) -> ScalarType
        # Create unsigned integer type

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> ScalarType
        # Create standard IEEE 754 floating point type

    @classmethod
    def float_(
        cls, exponent: int, mantissa: int, finite_values_only: bool, nan_repr: NanRepr
    ) -> ScalarType
        # Create non-standard floating point type

    @classmethod
    def from_id(cls, scalar_type_id: int) -> ScalarType
        # Look up ScalarType by its integer ID
```

#### scalar_types (Predefined Types)

```python
class scalar_types:
    # Integer types
    int4 = ScalarType.int_(4, None)
    uint4 = ScalarType.uint(4, None)
    int8 = ScalarType.int_(8, None)
    uint8 = ScalarType.uint(8, None)

    # FP8 types
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)
    float8_e5m2 = ScalarType.float_IEEE754(5, 2)
    float8_e8m0fnu = ScalarType(8, 0, False, 0, True, NanRepr.EXTD_RANGE_MAX_MIN)

    # FP16 types
    float16_e8m7 = ScalarType.float_IEEE754(8, 7)  # bfloat16
    float16_e5m10 = ScalarType.float_IEEE754(5, 10)  # float16

    # FP6 types
    float6_e3m2f = ScalarType.float_(3, 2, True, NanRepr.NONE)
    float6_e2m3f = ScalarType.float_(2, 3, True, NanRepr.NONE)

    # FP4 types
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)

    # GPTQ types
    uint2b2 = ScalarType.uint(2, 2)
    uint3b4 = ScalarType.uint(3, 4)
    uint4b8 = ScalarType.uint(4, 8)
    uint8b128 = ScalarType.uint(8, 128)

    # Colloquial names
    bfloat16 = float16_e8m7
    float16 = float16_e5m10
```

---

### Module: vllm.version

Version management for vLLM.

```python
__version__: str
    # The vLLM version string (e.g., "0.8.1" or "dev")
    # Loaded from _version.py (generated by setuptools-scm)

__version_tuple__: tuple
    # Version as a tuple

def _prev_minor_version_was(version_str: str) -> bool
    # Check if version_str matches the previous minor version
    # Used for --show-hidden-metrics-for-version

def _prev_minor_version() -> str
    # Return previous minor version number (for testing)
```

---

### Module: vllm.collect_env

Environment information collection for debugging and bug reports.

#### SystemEnv

```python
SystemEnv = namedtuple("SystemEnv", [
    "torch_version",
    "is_debug_build",
    "cuda_compiled_version",
    "gcc_version",
    "clang_version",
    "cmake_version",
    "os",
    "libc_version",
    "python_version",
    "python_platform",
    "is_cuda_available",
    "cuda_runtime_version",
    "cuda_module_loading",
    "nvidia_driver_version",
    "nvidia_gpu_models",
    "cudnn_version",
    "xpu_available",
    "xpu_runtime_version",
    "intel_graphics_compiler_version",
    "intel_gpu_models",
    "oneapi_compiler_version",
    "level_zero_loader_version",
    "level_zero_driver_version",
    "oneccl_version",
    "libigdgmm_version",
    "vllm_xpu_kernels_version",
    "sycl_version",
    "pip_version",
    "pip_packages",
    "conda_packages",
    "hip_compiled_version",
    "hip_runtime_version",
    "miopen_runtime_version",
    "caching_allocator_config",
    "is_xnnpack_available",
    "cpu_info",
    "rocm_version",
    "vllm_version",
    "vllm_build_flags",
    "gpu_topo",
    "env_vars",
])
```

#### Functions

```python
def get_env_info() -> SystemEnv
    # Collect all system environment information

def get_pretty_env_info() -> str
    # Get formatted environment information string

def get_vllm_version() -> str
    # Get vLLM version with git SHA details

def summarize_vllm_build_flags() -> str
    # Summarize CUDA, ROCm, XPU build flags
```

---

### Module: vllm.platforms

Platform detection and abstraction layer.

#### PlatformEnum

```python
class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()
```

#### CpuArchEnum

```python
class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    S390X = enum.auto()
    RISCV = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()
```

#### DeviceCapability

```python
class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str
    def to_int(self) -> int
    # Supports comparison operators: <, <=, ==, >=, >
```

#### Platform (Abstract Base)

```python
class Platform(ABC):
    """Abstract base class for hardware platform support."""

    # Class attributes
    platform_name: str
    platform_type: PlatformEnum
    device_type: str
    dispatch_key: str
    ray_device_key: str
    device_control_key: str | None
    dist_backend: str
    supported_quantization: list[str]

    # Abstract / overridable methods
    @abstractmethod
    def get_device_name(self, device_id: int = 0) -> str

    @abstractmethod
    def get_device_uuid(self, device_id: int = 0) -> str

    @abstractmethod
    def get_device_total_memory(self, device_id: int = 0) -> int

    @abstractmethod
    def is_cuda(self) -> bool
    def is_rocm(self) -> bool
    def is_tpu(self) -> bool
    def is_xpu(self) -> bool
    def is_cpu(self) -> bool
    def is_neuron(self) -> bool

    def get_punica_wrapper(self) -> str | None
    def get_current_memory_usage(self, device: torch.device | None = None) -> float
    def get_device_capability(self, device_id: int = 0) -> DeviceCapability | None
    def get_cpu_architecture(self) -> CpuArchEnum
    def supports_cuda_graph(self, ...) -> bool
    def inference_mode(self) -> torch.inference_mode
    def sleep(self, device: torch.device, level: int = 1) -> None
    def wake_up(self, device: torch.device) -> None
    def set_device(self, device: torch.device) -> None
    def pre_register_and_update(self, parser: ... = None) -> None
    def set_additional_forward_context(self, ...) -> dict[str, Any] | None

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None
```

#### current_platform

```python
current_platform: Platform
# Lazy-initialized singleton for the detected hardware platform
# Automatically detected on first access
```

#### Platform Plugins

```python
builtin_platform_plugins = {
    "tpu": tpu_platform_plugin,
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
}

def resolve_current_platform_cls_qualname() -> str
    # Resolve the current platform class from built-in and external plugins
```
