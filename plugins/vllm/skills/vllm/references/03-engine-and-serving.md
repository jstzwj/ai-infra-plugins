# vLLM Engine and Serving

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [V1 LLMEngine (Synchronous)](#v1-llmengine-synchronous)
- [V1 AsyncLLM (Asynchronous)](#v1-asyncllm-asynchronous)
- [EngineCore](#enginecore)
- [EngineCoreClient](#enginecoreclient)
- [Engine Arguments (EngineArgs)](#engine-arguments-engneargs)
- [Request Lifecycle](#request-lifecycle)
- [Input Processing](#input-processing)
- [Output Processing](#output-processing)
- [Detokenization](#detokenization)
- [Request and Output Types](#request-and-output-types)
- [Error Handling](#error-handling)
- [EngineClient Protocol](#engineclient-protocol)

---

## Architecture Overview

The V1 engine architecture separates concerns across three main processes:

```
+-------------------------------------------------------------------+
|                    Frontend Process (API Server)                    |
|                                                                    |
|  AsyncLLM / LLMEngine                                              |
|  +------------------+  +------------------+  +------------------+  |
|  | InputProcessor   |  | OutputProcessor  |  | EngineCoreClient |  |
|  | (Tokenizer + MM) |  | (Detokenizer +   |  | (ZMQ IPC)        |  |
|  |                  |  |  Logprobs)       |  |                  |  |
|  +------------------+  +------------------+  +--+------+-------+  |
+--------------------------------------------------|------|---------+
                                                   |      |
                              ZMQ IPC (async/sync) |      |
                                                   v      v
+-------------------------------------------------------------------+
|                    EngineCore Process                               |
|                                                                    |
|  EngineCore                                                        |
|  +------------------+  +------------------+  +------------------+  |
|  | Scheduler        |  | ModelExecutor    |  | KV Cache Mgr     |  |
|  | (Scheduling +    |  | (GPU Workers +   |  | (Block alloc +   |  |
|  |  KV allocation)  |  |  Forward pass)   |  |  Prefix caching) |  |
|  +------------------+  +------------------+  +------------------+  |
+-------------------------------------------------------------------+
```

### Key Design Decisions (V1)

1. **Multiprocessing**: EngineCore runs in a separate process, communicating via ZMQ
2. **Separated input/output processing**: Tokenizer and detokenizer run in the frontend process
3. **EngineCoreClient abstraction**: Supports in-process, sync multiprocess, and async multiprocess
4. **Scheduler-driven**: The scheduler drives model execution (schedule -> execute -> update)
5. **Data parallelism**: Built-in load balancer across DP ranks

---

## V1 LLMEngine (Synchronous)

The synchronous engine for batch inference and offline processing.

**Module**: `vllm.v1.engine.llm_engine`

Note: `vllm.engine.llm_engine.LLMEngine` is an alias for `vllm.v1.engine.llm_engine.LLMEngine`.

### Class: LLMEngine

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm_config` | `VllmConfig` | required | Global vLLM configuration |
| `executor_class` | `type[Executor]` | required | Executor implementation class |
| `log_stats` | `bool` | required | Whether to log statistics |
| `aggregate_engine_logging` | `bool` | `False` | Aggregate stats across DP |
| `usage_context` | `UsageContext` | `ENGINE_CONTEXT` | Usage context for defaults |
| `stat_loggers` | `list[StatLoggerFactory] \| None` | `None` | Custom stat loggers |
| `mm_registry` | `MultiModalRegistry` | `MULTIMODAL_REGISTRY` | Multi-modal registry |
| `multiprocess_mode` | `bool` | `False` | Run EngineCore in separate process |

#### Class Methods

```python
@classmethod
def from_vllm_config(
    cls,
    vllm_config: VllmConfig,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    disable_log_stats: bool = False,
) -> LLMEngine
    # Create LLMEngine from VllmConfig

@classmethod
def from_engine_args(
    cls,
    engine_args: EngineArgs,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    enable_multiprocessing: bool = False,
) -> LLMEngine
    # Create LLMEngine from EngineArgs
```

#### Instance Methods

```python
def get_num_unfinished_requests(self) -> int
    # Returns count of unfinished requests in output processor

def has_unfinished_requests(self) -> bool
    # Check if any requests are unfinished

def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool
    # Aggregate unfinished status across DP ranks

def get_supported_tasks(self) -> tuple[SupportedTask, ...]
    # Get supported task types for the loaded model
```

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vllm_config` | `VllmConfig` | Global configuration |
| `model_config` | `ModelConfig` | Model configuration |
| `observability_config` | `ObservabilityConfig` | Observability config |
| `renderer` | `BaseRenderer` | Input renderer |
| `input_processor` | `InputProcessor` | Input processing |
| `output_processor` | `OutputProcessor` | Output processing |
| `engine_core` | `EngineCoreClient` | Core engine client |
| `logger_manager` | `StatLoggerManager \| None` | Stats logging |
| `model_executor` | `Executor` | Model executor (v0 compat) |

---

## V1 AsyncLLM (Asynchronous)

The asynchronous engine for serving with streaming support.

**Module**: `vllm.v1.engine.async_llm`

Note: `vllm.engine.async_llm_engine.AsyncLLMEngine` is an alias for `vllm.v1.engine.async_llm.AsyncLLM`.

### Class: AsyncLLM

```python
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm_config` | `VllmConfig` | required | Global configuration |
| `executor_class` | `type[Executor]` | required | Executor class |
| `log_stats` | `bool` | required | Log statistics |
| `usage_context` | `UsageContext` | `ENGINE_CONTEXT` | Usage context |
| `mm_registry` | `MultiModalRegistry` | `MULTIMODAL_REGISTRY` | MM registry |
| `log_requests` | `bool` | `True` | Log request info |
| `start_engine_loop` | `bool` | `True` | Start engine loop |
| `stat_loggers` | `list[StatLoggerFactory] \| None` | `None` | Custom stat loggers |
| `aggregate_engine_logging` | `bool` | `False` | Aggregate DP stats |
| `client_addresses` | `dict[str, str] \| None` | `None` | Client ZMQ addresses |
| `client_count` | `int` | `1` | Number of client processes |
| `client_index` | `int` | `0` | Index of this client |

#### Class Methods

```python
@classmethod
def from_vllm_config(
    cls,
    vllm_config: VllmConfig,
    start_engine_loop: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
    enable_log_requests: bool = False,
    aggregate_engine_logging: bool = False,
    disable_log_stats: bool = False,
    client_addresses: dict[str, str] | None = None,
    client_count: int = 1,
    client_index: int = 0,
) -> AsyncLLM
    # Create AsyncLLM from VllmConfig

@classmethod
def from_engine_args(
    cls,
    engine_args: AsyncEngineArgs,
    start_engine_loop: bool = True,
    usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    stat_loggers: list[StatLoggerFactory] | None = None,
) -> AsyncLLM
    # Create AsyncLLM from AsyncEngineArgs
```

#### Instance Methods

```python
async def get_supported_tasks(self) -> tuple[SupportedTask, ...]
    # Get supported tasks (cached after first call)

async def add_request(
    self,
    request_id: str,
    prompt: EngineCoreRequest | PromptType | EngineInput | AsyncGenerator[StreamingInput, None],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
    prompt_text: str | None = None,
    reasoning_ended: bool | None = None,
    reasoning_parser_kwargs: dict[str, Any] | None = None,
) -> RequestOutputCollector
    # Add a new request to the AsyncLLM
    # Returns a RequestOutputCollector for async iteration

async def _add_request(
    self,
    request: EngineCoreRequest,
    prompt: str | None,
    parent_req: ParentRequest | None,
    index: int,
    queue: RequestOutputCollector,
) -> None
    # Internal: add request to output processor and engine core

async def _add_streaming_input_request(
    self,
    request_id: str,
    input_stream: AsyncGenerator[StreamingInput, None],
    sampling_params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> RequestOutputCollector
    # Add a streaming input request (multi-turn)
```

#### EngineClient Protocol Methods

```python
async def generate(
    self,
    prompt: EngineCoreRequest | PromptType | EngineInput | AsyncGenerator[StreamingInput, None],
    sampling_params: SamplingParams,
    request_id: str,
    *,
    prompt_text: str | None = None,
    lora_request: LoRARequest | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
    reasoning_ended: bool | None = None,
    reasoning_parser_kwargs: dict[str, Any] | None = None,
) -> AsyncGenerator[RequestOutput, None]
    # Generate outputs for a request
    # Yields RequestOutput objects as they become available

async def encode(
    self,
    prompt: PromptType | EngineInput,
    pooling_params: PoolingParams,
    request_id: str,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
    tokenization_kwargs: dict[str, Any] | None = None,
    reasoning_ended: bool | None = None,
) -> AsyncGenerator[PoolingRequestOutput, None]
    # Generate outputs for a pooling model

async def abort(self, request_id: str | Iterable[str]) -> None
    # Abort one or more requests

async def is_tracing_enabled(self) -> bool
async def do_log_stats(self) -> None
async def check_health(self) -> None
    # Raise if unhealthy

async def start_profile(self) -> None
async def stop_profile(self) -> None
async def reset_mm_cache(self) -> None
async def reset_encoder_cache(self) -> None
async def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool

async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None
async def wake_up(self, tags: list[str] | None = None) -> None
async def is_sleeping(self) -> bool

async def add_lora(self, lora_request: LoRARequest) -> bool
async def pause_generation(self, *, mode: PauseMode = "abort", ...) -> None
async def resume_generation(self) -> None
async def is_paused(self) -> bool

def shutdown(self, timeout: float | None = None) -> None
    # Shutdown engine and cleanup

# Properties
@property
def is_running(self) -> bool

@property
def is_stopped(self) -> bool

@property
def errored(self) -> bool

@property
def dead_error(self) -> BaseException
```

### Class: InputStreamError

```python
class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator."""

    def __init__(self, cause: Exception)
    cause: Exception
```

---

## EngineCore

The inner scheduling and execution loop of vLLM.

**Module**: `vllm.v1.engine.core`

### Class: EngineCore

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    )
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vllm_config` | `VllmConfig` | Global configuration |
| `executor_class` | `type[Executor]` | Executor implementation |
| `log_stats` | `bool` | Whether to log statistics |
| `executor_fail_callback` | `Callable \| None` | Callback on executor failure |
| `include_finished_set` | `bool` | Include finished set in outputs |

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vllm_config` | `VllmConfig` | Configuration |
| `model_executor` | `Executor` | Model executor |
| `scheduler` | `SchedulerInterface` | Request scheduler |
| `structured_output_manager` | `StructuredOutputManager` | Grammar manager |
| `batch_queue` | `deque \| None` | Batch queue for PP |
| `log_stats` | `bool` | Stats logging flag |

#### Methods

```python
def get_supported_tasks(self) -> tuple[SupportedTask, ...]
    # Get supported tasks from executor

def add_request(self, request: Request, request_wave: int = 0) -> None
    # Add request to scheduler
    # request_wave: DP wave indicator

def abort_requests(self, request_ids: list[str]) -> None
    # Abort requests (marks as FINISHED_ABORTED)

def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]
    # Schedule, execute, and produce output
    # Returns: (outputs per DP rank, was_model_executed flag)

def step_with_batch_queue(self) -> tuple[dict[int, EngineCoreOutputs] | None, bool]
    # Schedule and execute with batch queue (for PP)
    # Returns None if nothing to output

def post_step(self, model_executed: bool) -> None
    # Post-step processing (speculative decoding draft tokens)
```

#### Internal Methods

```python
@contextmanager
def log_error_detail(self, scheduler_output: SchedulerOutput)
    # Context manager to log detailed info on model execution failure

@contextmanager
def log_iteration_details(self, scheduler_output: SchedulerOutput)
    # Log per-iteration details when enabled

def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig
    # Profile memory and initialize KV caches
    # Steps:
    #   1. Get KV cache specs from model executor
    #   2. Profile available GPU memory
    #   3. Compute KV cache configuration
    #   4. Initialize KV cache blocks
    #   5. Warm up execution

def _process_aborts_queue(self) -> None
    # Process queued abort requests

def collective_rpc(self, method: str, timeout: float | None = None, 
                   args: tuple = (), kwargs: dict | None = None) -> list[Any]
    # Perform collective RPC to all workers
```

### Class: EngineCoreProc

The EngineCore running in a separate process with ZMQ communication.

```python
class EngineCoreProc:
    """EngineCore wrapper that runs in a background process."""

    @staticmethod
    def run_engine_core(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        engine_core_queue: Queue,
        input_addresses: EngineZmqAddresses,
        output_addresses: EngineZmqAddresses,
        ...
    ) -> None
        # Main loop for the EngineCore process
        # Handles: add_request, abort, step, utility, sleep/wake, profile, etc.
```

---

## EngineCoreClient

Abstract client interface for communicating with the EngineCore.

**Module**: `vllm.v1.engine.core_client`

### Class: EngineCoreClient (Abstract)

```python
class EngineCoreClient(ABC):
    """Base class for EngineCore communication."""

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> EngineCoreClient
        # Factory method:
        # multiprocess + asyncio -> AsyncMPClient
        # multiprocess + sync -> SyncMPClient
        # in-process -> InprocClient

    @staticmethod
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> AsyncMPClient
        # Create async multiprocess client
        # Handles DP: returns DPAsyncMPClient or DPLBAsyncMPClient
```

#### Abstract/Overridable Methods

```python
# Core operations
def shutdown(self, timeout: float | None = None) -> None
def get_output(self) -> EngineCoreOutputs
def get_supported_tasks(self) -> tuple[SupportedTask, ...]
def add_request(self, request: EngineCoreRequest) -> None
def abort_requests(self, request_ids: list[str]) -> None

# Cache management
def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None
def reset_mm_cache(self) -> None
def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool
def reset_encoder_cache(self) -> None

# Power management
def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None
def wake_up(self, tags: list[str] | None = None) -> None
def is_sleeping(self) -> bool

# LoRA
def add_lora(self, lora_request: LoRARequest) -> bool
def remove_lora(self, lora_id: int) -> bool
def list_loras(self) -> set[int]
def pin_lora(self, lora_id: int) -> bool

# Utility
def execute_dummy_batch(self) -> None
async def execute_dummy_batch_async(self) -> None
def save_sharded_state(self, path: str, pattern: str | None = None, max_size: int | None = None) -> None
def collective_rpc(self, method: str | Callable, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None) -> list[Any]

# Async variants
async def add_request_async(self, request: EngineCoreRequest) -> None
async def abort_requests_async(self, request_ids: list[str]) -> None
async def get_output_async(self) -> EngineCoreOutputs
async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]

# Properties
@property
def engine_ranks_managed(self) -> list[int]
def dp_engines_running(self) -> bool
```

### InprocClient

```python
class InprocClient(EngineCoreClient):
    """In-process client (no multiprocessing)."""
    # Used for debugging and single-process mode
    # EngineCore runs in the same process
```

### SyncMPClient

```python
class SyncMPClient(EngineCoreClient):
    """Synchronous multiprocess client using ZMQ."""
    # EngineCore runs in a separate process
    # Communication via ZMQ sockets + TensorIpc
```

### AsyncMPClient

```python
class AsyncMPClient(EngineCoreClient):
    """Asynchronous multiprocess client using ZMQ + asyncio."""
    # Used by AsyncLLM
    # Supports async iteration over outputs
```

### DPAsyncMPClient / DPLBAsyncMPClient

```python
class DPAsyncMPClient(AsyncMPClient):
    """Data-parallel async client (external load balancer)."""
    # One client per DP rank

class DPLBAsyncMPClient(AsyncMPClient):
    """Data-parallel load-balancing async client (internal LB)."""
    # Client balances requests across DP ranks
```

---

## Engine Arguments (EngineArgs)

Comprehensive argument parsing for engine configuration.

**Module**: `vllm.engine.arg_utils`

### Class: EngineArgs

```python
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
```

#### Model Configuration Fields

| Field | Type | Default | CLI Flag |
|-------|------|---------|----------|
| `model` | `str` | ModelConfig.model | `--model` |
| `enable_return_routed_experts` | `bool` | False | `--enable-return-routed-experts` |
| `model_weights` | `str` | ModelConfig.model_weights | - |
| `served_model_name` | `str \| list[str] \| None` | None | `--served-model-name` |
| `tokenizer` | `str \| None` | None | `--tokenizer` |
| `hf_config_path` | `str \| None` | None | `--hf-config-path` |
| `runner` | `RunnerOption` | ModelConfig.runner | `--runner` |
| `convert` | `ConvertOption` | ModelConfig.convert | `--convert` |
| `skip_tokenizer_init` | `bool` | False | `--skip-tokenizer-init` |
| `enable_prompt_embeds` | `bool` | False | `--enable-prompt-embeds` |
| `tokenizer_mode` | `TokenizerMode \| str` | ModelConfig.tokenizer_mode | `--tokenizer-mode` |
| `trust_remote_code` | `bool` | False | `--trust-remote-code` |
| `allowed_local_media_path` | `str` | ModelConfig.allowed_local_media_path | `--allowed-local-media-path` |
| `allowed_media_domains` | `list[str] \| None` | None | `--allowed-media-domains` |
| `download_dir` | `str \| None` | None | `--download-dir` |
| `safetensors_load_strategy` | `str \| None` | None | `--safetensors-load-strategy` |
| `load_format` | `str \| LoadFormats` | LoadConfig.load_format | `--load-format` |
| `config_format` | `str` | ModelConfig.config_format | `--config-format` |
| `dtype` | `ModelDType` | ModelConfig.dtype | `--dtype` |
| `seed` | `int` | ModelConfig.seed | `--seed` |
| `max_model_len` | `int` | ModelConfig.max_model_len | `--max-model-len` |
| `max_logprobs` | `int` | ModelConfig.max_logprobs | `--max-logprobs` |
| `logprobs_mode` | `LogprobsMode` | ModelConfig.logprobs_mode | `--logprobs-mode` |
| `revision` | `str \| None` | None | `--revision` |
| `code_revision` | `str \| None` | None | `--code-revision` |
| `hf_token` | `bool \| str \| None` | None | `--hf-token` |
| `hf_overrides` | `HfOverrides` | - | `--hf-overrides` |
| `tokenizer_revision` | `str \| None` | None | `--tokenizer-revision` |
| `quantization` | `QuantizationMethods \| str \| None` | None | `--quantization, -q` |
| `quantization_config` | `dict \| OnlineQuantizationConfigArgs \| None` | None | - |
| `allow_deprecated_quantization` | `bool` | False | `--allow-deprecated-quantization` |
| `enforce_eager` | `bool` | False | `--enforce-eager` |
| `disable_sliding_window` | `bool` | False | `--disable-sliding-window` |
| `disable_cascade_attn` | `bool` | False | `--disable-cascade-attn` |
| `generation_config` | `str` | ModelConfig.generation_config | `--generation-config` |
| `override_generation_config` | `dict[str, Any]` | - | `--override-generation-config` |
| `enable_sleep_mode` | `bool` | False | `--enable-sleep-mode` |
| `model_impl` | `str` | ModelConfig.model_impl | `--model-impl` |
| `override_attention_dtype` | `str \| None` | None | `--override-attention-dtype` |
| `logits_processors` | `list[str \| type[LogitsProcessor]] \| None` | None | `--logits-processors` |

#### Parallel Configuration Fields

| Field | Type | Default | CLI Flag |
|-------|------|---------|----------|
| `distributed_executor_backend` | `str \| DistributedExecutorBackend \| type[Executor] \| None` | None | `--distributed-executor-backend` |
| `pipeline_parallel_size` | `int` | 1 | `--pipeline-parallel-size, -pp` |
| `tensor_parallel_size` | `int` | 1 | `--tensor-parallel-size, -tp` |
| `data_parallel_size` | `int` | 1 | `--data-parallel-size, -dp` |
| `data_parallel_rank` | `int \| None` | None | `--data-parallel-rank, -dpn` |
| `data_parallel_size_local` | `int \| None` | None | `--data-parallel-size-local, -dpl` |
| `data_parallel_address` | `str \| None` | None | `--data-parallel-address, -dpa` |
| `data_parallel_rpc_port` | `int \| None` | None | `--data-parallel-rpc-port, -dpp` |
| `data_parallel_backend` | `DataParallelBackend` | "mp" | `--data-parallel-backend, -dpb` |
| `data_parallel_hybrid_lb` | `bool` | False | `--data-parallel-hybrid-lb, -dph` |
| `data_parallel_external_lb` | `bool` | False | `--data-parallel-external-lb, -dpe` |
| `enable_expert_parallel` | `bool` | False | `--enable-expert-parallel, -ep` |
| `master_addr` | `str` | ParallelConfig.master_addr | `--master-addr` |
| `master_port` | `int` | ParallelConfig.master_port | `--master-port` |
| `nnodes` | `int` | 1 | `--nnodes, -n` |
| `node_rank` | `int` | 0 | `--node-rank, -r` |
| `distributed_timeout_seconds` | `int \| None` | None | `--distributed-timeout-seconds` |
| `disable_custom_all_reduce` | `bool` | False | `--disable-custom-all-reduce` |
| `worker_cls` | `str` | ParallelConfig.worker_cls | `--worker-cls` |
| `worker_extension_cls` | `str` | ParallelConfig.worker_extension_cls | `--worker-extension-cls` |
| `enable_dbo` | `bool` | False | `--enable-dbo` |

#### Cache Configuration Fields

| Field | Type | Default | CLI Flag |
|-------|------|---------|----------|
| `block_size` | `int \| None` | None | `--block-size` |
| `enable_prefix_caching` | `bool \| None` | None | `--enable-prefix-caching` |
| `prefix_caching_hash_algo` | `PrefixCachingHashAlgo` | CacheConfig default | `--prefix-caching-hash-algo` |
| `gpu_memory_utilization` | `float` | 0.9 | `--gpu-memory-utilization` |
| `kv_cache_memory_bytes` | `int \| None` | None | `--kv-cache-memory-bytes` |
| `kv_cache_dtype` | `CacheDType` | CacheConfig.cache_dtype | `--kv-cache-dtype` |
| `num_gpu_blocks_override` | `int \| None` | None | `--num-gpu-blocks-override` |
| `calculate_kv_scales` | `bool` | False | `--calculate-kv-scales` |
| `kv_cache_dtype_skip_layers` | `list[str]` | - | `--kv-cache-dtype-skip-layers` |
| `kv_sharing_fast_prefill` | `bool` | False | `--kv-sharing-fast-prefill` |
| `kv_offloading_size` | `float \| None` | None | `--kv-offloading-size` |
| `kv_offloading_backend` | `KVOffloadingBackend` | CacheConfig default | `--kv-offloading-backend` |

#### Scheduler Configuration Fields

| Field | Type | Default | CLI Flag |
|-------|------|---------|----------|
| `max_num_batched_tokens` | `int \| None` | None | `--max-num-batched-tokens` |
| `max_num_seqs` | `int \| None` | None | `--max-num-seqs` |
| `max_num_partial_prefills` | `int` | 1 | `--max-num-partial-prefills` |
| `max_long_partial_prefills` | `int` | 1 | `--max-long-partial-prefills` |
| `long_prefill_token_threshold` | `int` | SchedulerConfig default | `--long-prefill-token-threshold` |
| `enable_chunked_prefill` | `bool \| None` | None | `--enable-chunked-prefill` |
| `disable_chunked_mm_input` | `bool` | False | `--disable-chunked-mm-input` |
| `scheduling_policy` | `SchedulerPolicy` | SchedulerConfig.policy | `--scheduling-policy` |
| `scheduler_cls` | `str \| type \| None` | None | `--scheduler-cls` |
| `scheduler_reserve_full_isl` | `bool` | False | `--scheduler-reserve-full-isl` |
| `disable_hybrid_kv_cache_manager` | `bool \| None` | None | `--disable-hybrid-kv-cache-manager` |
| `async_scheduling` | `bool \| None` | None | `--async-scheduling` |
| `stream_interval` | `int` | SchedulerConfig.stream_interval | `--stream-interval` |

#### Compilation Configuration Fields

| Field | Type | Default | CLI Flag |
|-------|------|---------|----------|
| `cudagraph_capture_sizes` | `list[int] \| None` | None | `--cudagraph-capture-sizes` |
| `max_cudagraph_capture_size` | `int \| None` | None | `--max-cudagraph-capture-size` |
| `compilation_config` | `CompilationConfig` | default | `--compilation-config, -cc` |

#### Other Fields

| Field | Type | Default | CLI Flag |
|-------|------|---------|----------|
| `disable_log_stats` | `bool` | False | `--disable-log-stats` |
| `aggregate_engine_logging` | `bool` | False | `--aggregate-engine-logging` |
| `shutdown_timeout` | `int` | 0 | `--shutdown-timeout` |
| `fail_on_environ_validation` | `bool` | False | `--fail-on-environ-validation` |
| `tokens_only` | `bool` | False | - |
| `optimization_level` | `OptimizationLevel` | VllmConfig.default | `--optimization-level` |
| `performance_mode` | `PerformanceMode` | VllmConfig.default | `--performance-mode` |

#### EngineArgs Methods

```python
@staticmethod
def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser
    # Add all CLI arguments to the parser

@classmethod
def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs
    # Create EngineArgs from parsed CLI arguments

def create_model_config(self) -> ModelConfig
    # Create ModelConfig from engine args

def create_load_config(self) -> LoadConfig
    # Create LoadConfig from engine args

def create_speculative_config(
    self,
    target_model_config: ModelConfig,
    target_parallel_config: ParallelConfig,
) -> SpeculativeConfig | None
    # Create SpeculativeConfig if configured

def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig
    # Create the complete VllmConfig
    # Resolves all defaults, validates features, computes cache config
```

### Class: AsyncEngineArgs

```python
@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    enable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
        parser: FlexibleArgumentParser,
        async_args_only: bool = False,
    ) -> FlexibleArgumentParser
        # Add async-specific args (--enable-log-requests)
```

---

## Request Lifecycle

### 1. Request Submission

```python
# AsyncLLM.add_request() flow:
1. Validate parameters (SamplingParams / PoolingParams)
2. Check for streaming input (AsyncGenerator)
3. Process inputs via InputProcessor:
   - Tokenize text
   - Process multimodal inputs
   - Create EngineCoreRequest
4. Create RequestOutputCollector (queue for async iteration)
5. For n > 1: fan out child requests via ParentRequest
6. Add to OutputProcessor (track state)
7. Send to EngineCore via EngineCoreClient
```

### 2. Scheduling

```python
# EngineCore.step() flow:
1. Check if scheduler has requests
2. scheduler.schedule():
   - Select requests from waiting queue
   - Allocate KV cache blocks
   - Determine prefill vs decode batches
   - Handle prefix caching hits
   - Create SchedulerOutput
3. model_executor.execute_model(scheduler_output):
   - Run model forward pass
   - Return ModelRunnerOutput
4. scheduler.update_from_output():
   - Update request states
   - Check finish conditions (stop tokens, max length, etc.)
   - Return EngineCoreOutputs
```

### 3. Output Processing

```python
# OutputProcessor.process_outputs() flow:
1. For each EngineCoreOutput:
   - Get RequestState for the request
   - Update detokenizer with new token IDs
   - Process logprobs
   - Check stop strings
   - Create RequestOutput
2. Stream outputs to RequestOutputCollector
3. Log stats if enabled
```

### 4. Client Receives Output

```python
# Async iteration:
async for output in await engine.generate(prompt, params, request_id):
    # Each output is a RequestOutput
    # Can be delta (streaming) or full
    print(output.outputs[0].text)
```

---

## Input Processing

### Class: InputProcessor

**Module**: `vllm.v1.engine.input_processor`

```python
class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None

    @property
    def tokenizer(self) -> TokenizerLike | None

    def get_tokenizer(self) -> TokenizerLike

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest
        # Process input and create EngineCoreRequest
        # Steps:
        #   1. Validate params
        #   2. Validate LoRA
        #   3. Preprocess input (tokenize, extract MM)
        #   4. Create EngineCoreRequest with all metadata

    def assign_request_id(self, request: EngineCoreRequest) -> None
        # Assign unique request ID if not already set

    def inject_into_mm_cache(
        self,
        mm_hashes: dict[str, list[str]],
        mm_kwargs: dict[str, list],
    ) -> None
        # Inject pre-processed multimodal data into cache
```

---

## Output Processing

### Class: OutputProcessor

**Module**: `vllm.v1.engine.output_processor`

```python
class OutputProcessor:
    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        log_stats: bool,
        stream_interval: int,
        tracing_enabled: bool = False,
    ) -> None

    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ) -> None
        # Register a new request for output tracking

    def process_outputs(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput
        # Process engine outputs into RequestOutputs
        # Returns OutputProcessorOutput with:
        #   request_outputs: list of completed outputs
        #   reqs_to_abort: list of request IDs to abort

    def get_num_unfinished_requests(self) -> int
    def has_unfinished_requests(self) -> bool
```

### Class: RequestOutputCollector

```python
class RequestOutputCollector:
    """Collects streamed RequestOutputs for async iteration."""

    def __init__(self, output_kind: RequestOutputKind, request_id: str)

    aggregate: bool
        # True for DELTA mode (merge outputs)

    request_id: str

    output: RequestOutput | PoolingRequestOutput | Exception | None

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None
        # Non-blocking put (merges for DELTA mode)

    async def get(self) -> RequestOutput | PoolingRequestOutput
        # Async get (blocks until output available)

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None
        # Non-blocking get

    def close(self) -> None
        # Cancel input stream task if any
```

### Class: RequestState

```python
@dataclass
class RequestState:
    """Tracks the state of a request in the output processor."""

    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
    )

    # Key attributes:
    request_id: str
    external_req_id: str
    parent_req: ParentRequest | None
    request_index: int
    output_kind: RequestOutputKind
    prompt: str | None
    prompt_token_ids: list[int] | None
    is_prefilling: bool
    queue: RequestOutputCollector | None
    detokenizer: IncrementalDetokenizer | None
    logprobs_processor: LogprobsProcessor | None
    stats: RequestStateStats | None

    def apply_streaming_update(self, update: StreamingUpdate) -> None
        # Apply streaming input update to the request state
```

---

## Detokenization

### Class: IncrementalDetokenizer (Base)

**Module**: `vllm.v1.engine.detokenizer`

```python
class IncrementalDetokenizer:
    """Base class for incremental token-to-text conversion."""

    token_ids: list[int]

    @property
    def output_token_ids(self) -> list[int]

    def num_output_tokens(self) -> int

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None
        # Update with new token IDs, return matched stop string or None

    def get_next_output_text(self, finished: bool, delta: bool) -> str
        # Get output text (full or delta)

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> IncrementalDetokenizer
        # Factory:
        # No tokenizer -> IncrementalDetokenizer (no-op)
        # Fast tokenizer -> FastIncrementalDetokenizer
        # Other -> SlowIncrementalDetokenizer
```

### Class: BaseIncrementalDetokenizer

```python
class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    def __init__(self, request: EngineCoreRequest)
        # Initializes:
        #   self.stop: list of stop strings
        #   self.min_tokens: minimum tokens before stop
        #   self.include_stop_str_in_output: whether to include stop in output
        #   self.stop_buffer_length: chars to hold back for stop detection
        #   self.output_text: accumulated text

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None
        # 1) Detokenize new tokens incrementally
        # 2) Evaluate stop criteria
        # Returns matched stop string or None

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str

    def get_next_output_text(self, finished: bool, delta: bool) -> str
        # delta=True: only new text since last call
        # finished=True: include stop buffer text
```

### Class: FastIncrementalDetokenizer

```python
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """Uses tokenizers library DecodeStream for fast incremental decoding."""
    # Requires tokenizers >= 0.22.0
    # Uses native prefill with prompt tokens
```

### Class: SlowIncrementalDetokenizer

```python
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """Fallback using Python-based incremental detokenization."""
    # Uses vllm.tokenizers.detokenizer_utils.detokenize_incrementally
```

---

## Request and Output Types

### Class: Request (V1 Internal)

**Module**: `vllm.v1.request`

```python
class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: LoRARequest | None = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[[Request], list[BlockHash]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> None

    # Properties
    @property
    def use_structured_output(self) -> bool
    @property
    def num_tokens(self) -> int       # Total tokens (prompt + output)
    @property
    def num_tokens_with_spec(self) -> int  # Including speculative tokens
    @property
    def num_output_tokens(self) -> int
    @property
    def num_encoder_inputs(self) -> int
    @property
    def has_encoder_inputs(self) -> bool

    # Methods
    def append_output_token_ids(self, token_ids: int | list[int]) -> None
    def update_block_hashes(self) -> None
    def is_finished(self) -> bool
    def get_finished_reason(self) -> FinishReason | None
    def get_num_encoder_embeds(self, input_id: int) -> int
    def record_event(self, event_type: EngineCoreEventType, timestamp: float | None = None) -> None
    def take_events(self) -> list[EngineCoreEvent] | None
    def take_prefill_stats(self) -> PrefillStats | None

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest, block_hasher) -> Request
```

### Class: RequestStatus

```python
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # After PREEMPTED = finished statuses
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    @staticmethod
    def is_finished(status: RequestStatus) -> bool
        # Returns True if status > PREEMPTED

    @staticmethod
    def get_finished_reason(status: RequestStatus) -> FinishReason | None
```

### V1 Output Types

**Module**: `vllm.v1.outputs`

#### SamplerOutput

```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
        # [num_reqs, max_num_generated_tokens]
    logprobs_tensors: LogprobsTensors | None
```

#### ModelRunnerOutput

```python
@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    logprobs: LogprobsLists | None
    prompt_logprobs_dict: dict[str, LogprobsTensors | None]
    pooler_output: list[torch.Tensor | None] | None
    kv_connector_output: KVConnectorOutput | None
    ec_connector_output: ECConnectorOutput | None
    num_nans_in_logits: dict[str, int] | None
    cudagraph_stats: CUDAGraphStat | None
```

#### AsyncModelRunnerOutput

```python
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput
        # Blocking call to get the output (may involve device-to-host copy)
```

#### LogprobsTensors / LogprobsLists

```python
class LogprobsTensors(NamedTuple):
    logprob_token_ids: torch.Tensor   # [num_positions, max_logprobs + 1]
    logprobs: torch.Tensor             # [num_positions, max_logprobs + 1]
    selected_token_ranks: torch.Tensor # [num_positions]
    cu_num_generated_tokens: list[int] | None = None

    def tolists(self, cu_num_generated_tokens=None) -> LogprobsLists
    def to_cpu_nonblocking(self) -> LogprobsTensors
    def filter(self, mask: torch.Tensor) -> LogprobsTensors

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> LogprobsTensors

class LogprobsLists(NamedTuple):
    logprob_token_ids: np.ndarray
    logprobs: np.ndarray
    sampled_token_ranks: np.ndarray
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int) -> LogprobsLists
```

---

## Error Handling

### Engine Death Detection

```python
class EngineDeadError(Exception):
    """Raised when the engine process has died."""
    # Module: vllm.v1.engine.exceptions
```

### Engine Generate Error

```python
class EngineGenerateError(Exception):
    """Raised when generation fails."""
    # Module: vllm.v1.engine.exceptions
```

### Error Flow

```
1. EngineCore process crashes
       |
2. EngineCoreClient detects via ZMQ socket close
       |
3. Sets internal error state
       |
4. AsyncLLM.generate() raises EngineDeadError
       |
5. API server returns 500 error to client
```

### InputStreamError

```python
class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator."""
    cause: Exception
```

---

## EngineClient Protocol

**Module**: `vllm.engine.protocol`

### Class: EngineClient (Abstract)

```python
class EngineClient(ABC):
    """Protocol class for Clients to Engine."""

    # Required attributes
    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: InputProcessor

    # Abstract properties
    @property
    @abstractmethod
    def is_running(self) -> bool

    @property
    @abstractmethod
    def is_stopped(self) -> bool

    @property
    @abstractmethod
    def errored(self) -> bool

    @property
    @abstractmethod
    def dead_error(self) -> BaseException

    # Abstract methods
    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest | PromptType | EngineInput | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]

    @abstractmethod
    def encode(
        self,
        prompt: PromptType | EngineInput,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]

    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None

    @abstractmethod
    async def is_tracing_enabled(self) -> bool

    @abstractmethod
    async def do_log_stats(self) -> None

    @abstractmethod
    async def check_health(self) -> None

    @abstractmethod
    async def start_profile(self) -> None

    @abstractmethod
    async def stop_profile(self) -> None

    @abstractmethod
    async def reset_mm_cache(self) -> None

    @abstractmethod
    async def reset_encoder_cache(self) -> None

    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool

    @abstractmethod
    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None

    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None

    @abstractmethod
    async def is_sleeping(self) -> bool

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool

    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None

    @abstractmethod
    async def resume_generation(self) -> None

    @abstractmethod
    async def is_paused(self) -> bool

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None

    # Methods with default implementations
    async def scale_elastic_ep(self, new_data_parallel_size: int, drain_timeout: int = 300) -> None
    async def collective_rpc(self, method: str, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None)
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]
    async def init_weight_transfer_engine(self, init_request: WeightTransferInitRequest) -> None
    async def update_weights(self, request: WeightTransferUpdateRequest) -> None
```

### StreamingInput

```python
@dataclass
class StreamingInput:
    """Input data for a streaming generation request."""

    prompt: EngineInput
    sampling_params: SamplingParams | None = None
```

### PauseMode

```python
PauseMode = Literal["abort", "wait", "keep"]
# "abort": Abort in-flight requests immediately
# "wait": Wait for in-flight requests to complete
# "keep": Freeze queue; resume on resume_generation()
```
