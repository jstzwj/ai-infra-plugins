# V1 Worker and Executor Reference

This document provides a comprehensive reference for the V1 worker and executor architecture in vLLM, covering GPUModelRunner, GPUWorker, InputBatch processing, ubatching, block tables, CUDA graphs, executor types, and connector mixins.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Executor Framework](#2-executor-framework)
3. [UniProcExecutor](#3-uniprocexecutor)
4. [MultiprocExecutor](#4-multiprocexecutor)
5. [RayDistributedExecutor](#5-raydistributedexecutor)
6. [WorkerBase](#6-workerbase)
7. [WorkerWrapperBase](#7-workerwrapperbase)
8. [GPUWorker](#8-gpuworker)
9. [GPUModelRunner](#9-gpumodelrunner)
10. [InputBatch (GPUInputBatch)](#10-inputbatch-gpuinputbatch)
11. [CachedRequestState](#11-cachedrequeststate)
12. [Block Table Management](#12-block-table-management)
13. [UBatching Framework](#13-ubatching-framework)
14. [Workspace Management](#14-workspace-management)
15. [CUDA Graph Handling](#15-cuda-graph-handling)
16. [KV Connector Model Runner Mixin](#16-kv-connector-model-runner-mixin)
17. [EC Connector Model Runner Mixin](#17-ec-connector-model-runner-mixin)
18. [LoRA Model Runner Mixin](#18-lora-model-runner-mixin)
19. [Async Scheduling Support](#19-async-scheduling-support)
20. [Speculative Decoding in Model Runner](#20-speculative-decoding-in-model-runner)
21. [Attention Metadata Building](#21-attention-metadata-building)
22. [Utility Classes](#22-utility-classes)
23. [Full Parameter Reference](#23-full-parameter-reference)

---

## 1. Architecture Overview

The V1 execution architecture follows a layered design:

```
Engine
  --> Executor (UniProc / Multiproc / Ray)
       --> Worker (GPU / CPU / TPU)
            --> ModelRunner (GPUModelRunner / CPUModelRunner)
                 --> Model (nn.Module)
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `Executor` (ABC) | `vllm/v1/executor/abstract.py` | Abstract executor interface |
| `UniProcExecutor` | `vllm/v1/executor/uniproc_executor.py` | Single-process executor |
| `MultiprocExecutor` | `vllm/v1/executor/multiproc_executor.py` | Multi-process executor |
| `RayDistributedExecutor` | `vllm/v1/executor/ray_executor.py` | Ray-based distributed executor |
| `WorkerBase` | `vllm/v1/worker/worker_base.py` | Abstract worker interface |
| `WorkerWrapperBase` | `vllm/v1/worker/worker_base.py` | Lazy worker initialization wrapper |
| `Worker` (GPU) | `vllm/v1/worker/gpu_worker.py` | GPU worker implementation |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py` | GPU model execution runner |
| `InputBatch` | `vllm/v1/worker/gpu_input_batch.py` | GPU input batch state management |
| `BlockTable` | `vllm/v1/worker/block_table.py` | KV cache block table |
| `UBatchWrapper` | `vllm/v1/worker/gpu_ubatch_wrapper.py` | Microbatching wrapper |
| `WorkspaceManager` | `vllm/v1/worker/workspace.py` | Workspace memory management |

---

## 2. Executor Framework

### Executor (Abstract Base Class)

**File**: `vllm/v1/executor/abstract.py`

```python
class Executor(ABC):
    uses_ray: bool = False
    supports_pp: bool = False

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        """Factory method that returns the appropriate executor class
        based on the distributed_executor_backend configuration."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None: ...

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize KV caches and compile/warm up models."""

    def determine_available_memory(self) -> list[int]: ...

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]: ...

    @abstractmethod
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None,
                       non_block=False) -> list | Future: ...

    @abstractmethod
    def execute_model(self, scheduler_output, non_block=False): ...

    @abstractmethod
    def sample_tokens(self, grammar_output, non_block=False): ...

    @abstractmethod
    def check_health(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...
```

### Executor Selection

The executor class is selected based on `parallel_config.distributed_executor_backend`:

| Backend | Executor Class | Description |
|---------|---------------|-------------|
| `"uni"` | `UniProcExecutor` | Single-process, single-GPU |
| `"mp"` | `MultiprocExecutor` | Multi-process, multi-GPU |
| `"ray"` | `RayDistributedExecutor` | Ray-based distributed |
| `"external_launcher"` | `ExecutorWithExternalLauncher` | Torchrun-compatible |

### `collective_rpc(method, timeout, args, kwargs, non_block)`

Executes a method on all workers.

**Parameters:**
- `method`: Method name (str) or callable to execute on each worker
- `timeout`: Maximum wait time in seconds
- `args`: Positional arguments
- `kwargs`: Keyword arguments
- `non_block`: If `True`, returns a `Future` instead of blocking

**Returns**: List of results from each worker (or `Future` if `non_block=True`)

---

## 3. UniProcExecutor

**File**: `vllm/v1/executor/uniproc_executor.py`

### Class: `UniProcExecutor`

Single-process executor for single-GPU inference.

```python
class UniProcExecutor(Executor):
    driver_worker: WorkerWrapperBase
    async_output_thread: ThreadPoolExecutor | None
```

#### `_init_executor()`

1. Creates a `WorkerWrapperBase` with `rpc_rank=0`
2. Computes distributed arguments (init method, rank, local_rank)
3. Initializes the worker via `init_worker()`
4. Initializes the device via `init_device()`
5. Loads the model via `load_model()`

#### `execute_model(scheduler_output, non_block)`

Delegates to `driver_worker.execute_model()` via `collective_rpc()`.

#### `sample_tokens(grammar_output, non_block)`

Delegates to `driver_worker.sample_tokens()` via `collective_rpc()`.

#### `collective_rpc(method, timeout, args, kwargs, non_block, single_value)`

For single-process, directly calls the method on the driver worker.

**Parameters:**
- `single_value`: If `True`, returns a single result instead of a list

### ExecutorWithExternalLauncher

Subclass of `UniProcExecutor` for torchrun-compatible distributed inference. Uses environment variables (`RANK`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`) for distributed initialization.

---

## 4. MultiprocExecutor

**File**: `vllm/v1/executor/multiproc_executor.py`

### Class: `MultiprocExecutor`

Multi-process executor for multi-GPU inference using Python multiprocessing.

```python
class MultiprocExecutor(Executor):
    supports_pp: bool = True
    workers: list[WorkerProc]       # Worker process handles
    rpc_broadcast_mq: MessageQueue  # Broadcast message queue
```

#### `_init_executor()`

1. Creates a shared worker lock
2. Spawns worker processes using `get_mp_context()`
3. Each worker initializes independently in its own process
4. Waits for all workers to become ready
5. Sets up message queues for `SchedulerOutput` broadcast

#### Worker Process Architecture

Each worker process runs:
1. `WorkerProc.make_worker_process()`: Creates and starts the process
2. Worker initializes device, loads model, allocates KV cache
3. Worker signals readiness via a pipe
4. Worker listens for RPC calls via shared memory broadcast

#### Communication

- **SchedulerOutput**: Broadcast via `MessageQueue` (shared memory)
- **ModelRunnerOutput**: Sent back via pipes or shared memory
- **RPC calls**: Via `collective_rpc()` using broadcast

#### `collective_rpc(method, timeout, args, kwargs, non_block)`

Broadcasts the method call to all worker processes and collects results.

### FutureWrapper

A `Future` subclass that processes results in FIFO order from the futures queue.

```python
class FutureWrapper(Future):
    def result(self, timeout=None):
        # Drain any futures ahead of us in the queue
        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()
```

---

## 5. RayDistributedExecutor

**File**: `vllm/v1/executor/ray_executor.py`

Ray-based distributed executor that uses Ray actors for worker management. Similar to `MultiprocExecutor` but uses Ray for process orchestration, enabling multi-node inference.

---

## 6. WorkerBase

**File**: `vllm/v1/worker/worker_base.py`

### Class: `WorkerBase`

Abstract base class for workers.

```python
class WorkerBase:
    vllm_config: VllmConfig
    model_config: ModelConfig
    cache_config: CacheConfig
    lora_config: LoRAConfig | None
    load_config: LoadConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    device_config: DeviceConfig
    speculative_config: SpeculativeConfig | None
    observability_config: ObservabilityConfig | None
    kv_transfer_config: KVTransferConfig | None
    compilation_config: CompilationConfig

    local_rank: int
    rank: int
    distributed_init_method: str
    is_driver_worker: bool

    device: torch.device | None
    model_runner: nn.Module | None
```

#### Abstract Methods

| Method | Description |
|--------|-------------|
| `get_kv_cache_spec()` | Return KV cache specifications |
| `compile_or_warm_up_model()` | Compile/warm up model, return `CompilationTimes` |
| `check_health()` | Health check |
| `init_device()` | Initialize device, load model |
| `get_model()` | Return the loaded model |
| `load_model(load_dummy_weights)` | Load model onto device |
| `execute_model(scheduler_output)` | Execute model forward pass |
| `sample_tokens(grammar_output)` | Sample tokens from logits |
| `get_cache_block_size_bytes()` | Return cache block size |

#### Concrete Methods

- **`apply_model(fn)`**: Apply a function on the model inside this worker.
- **`reset_mm_cache()`**: Reset multimodal cache.
- **`get_model_inspection()`**: Return a hierarchical view of the model.
- **`add_lora(lora_request)`**: Add a LoRA adapter.
- **`remove_lora(lora_id)`**: Remove a LoRA adapter.
- **`pin_lora(lora_id)`**: Pin a LoRA adapter.
- **`list_loras()`**: List active LoRA adapter IDs.
- **`vocab_size`** (property): Get vocabulary size from model configuration.
- **`shutdown()`**: Clean up resources.

### CompilationTimes

```python
class CompilationTimes(NamedTuple):
    language_model: float  # Compilation time for language model (seconds)
    encoder: float         # Compilation time for encoder (seconds)
```

---

## 7. WorkerWrapperBase

**File**: `vllm/v1/worker/worker_base.py`

### Class: `WorkerWrapperBase`

Lazy initialization wrapper for workers. The wrapper remembers the worker module and class name, then performs actual initialization in `init_worker()`.

```python
class WorkerWrapperBase:
    def __init__(self, rpc_rank: int = 0, global_rank: int | None = None):
        self.rpc_rank = rpc_rank
        self.global_rank = global_rank
        self.worker: WorkerBase | None = None

    def init_worker(self, all_kwargs: list[dict]): ...

    def init_device(self): ...

    def load_model(self, *, load_dummy_weights: bool = False): ...
```

**Why a wrapper?**: The wrapper enables deferred initialization, allowing environment variables to be set before the worker is created. This is needed for distributed setups where rank information may not be available at construction time.

---

## 8. GPUWorker

**File**: `vllm/v1/worker/gpu_worker.py`

### Class: `Worker` (GPUWorker)

GPU worker implementation extending `WorkerBase`.

```python
class Worker(WorkerBase):
    model_runner: GPUModelRunner
    weight_transfer_engine: WeightTransferEngine | None
    profiler: TorchProfilerWrapper | CudaProfilerWrapper | None
```

#### Constructor

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    is_driver_worker: bool = False,
):
```

Initializes:
- Float32 matmul precision from `VLLM_FLOAT32_MATMUL_PRECISION`
- ElasticEP scaling executor
- Sleep-mode saved buffers
- Weight transfer engine (for live weight updates)
- Profiler configuration

#### `sleep(level)`

Puts the GPU to sleep, offloading memory to free GPU RAM.

**Parameters:**
- `level`: Sleep level:
  - `1`: Offload weights only
  - `2`: Offload weights + save buffers

Uses `CuMemAllocator` for memory pool management.

#### `wake_up(tags)`

Wakes up the GPU from sleep mode, restoring offloaded memory.

**Parameters:**
- `tags`: List of tags to restore (`"weights"`, `"kv_cache"`)

Also restores model buffers saved during level-2 sleep.

#### `init_device()`

1. Sets CUDA device
2. Checks dtype support
3. Initializes distributed environment
4. Takes memory snapshot
5. Requests memory for KV cache
6. Creates workspace manager
7. Creates `GPUModelRunner`

#### `determine_available_memory() -> int`

Profiles the peak memory usage and determines available KV cache memory:

1. Executes a profiling forward pass
2. Measures torch peak memory
3. Estimates CUDA graph memory
4. Calculates: `available = requested_memory - non_kv_memory - cudagraph_estimate`

#### `initialize_from_config(kv_cache_config)`

Allocates GPU KV cache based on the configuration:

1. Updates `cache_config.num_gpu_blocks`
2. Initializes KV transfer connector
3. Calls `model_runner.initialize_kv_cache()`
4. Initializes routed experts capturer (if enabled)
5. Initializes KV zero metadata (for block zeroing)

#### `compile_or_warm_up_model() -> CompilationTimes`

1. Runs compilation warmup for specified sizes
2. Runs kernel warmup
3. Captures CUDA graphs (if not eager mode)
4. Warms up sampler (V1) or JIT compiles triton kernels (V2)
5. Resets random seed
6. Activates Triton JIT monitor

#### `execute_model(scheduler_output)`

1. Ensures previous PP sends are complete
2. Handles pipeline parallel intermediate tensors
3. Delegates to `model_runner.execute_model()`
4. For non-last PP ranks, sends intermediate tensors

#### `sample_tokens(grammar_output)`

Delegates to `model_runner.sample_tokens()`.

### AsyncIntermediateTensors

```python
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization."""

    def wait_for_comm(self) -> None:
        """Wait for all communication handles to complete."""

    def __getattribute__(self, name):
        # Ensure .tensors is ready before use
        if name == "tensors" and not self._comm_waited:
            self.wait_for_comm()
        return super().__getattribute__(name)
```

### `init_worker_distributed_environment(...)`

Initializes the distributed environment:

```python
def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
```

1. Initializes batch invariance
2. Overrides EPLB environment variables
3. Sets custom all-reduce configuration
4. Initializes distributed environment (NCCL)
5. Initializes model parallel groups (TP, PP, PCP, DCP)
6. Initializes EC connector

---

## 9. GPUModelRunner

**File**: `vllm/v1/worker/gpu_model_runner.py`

### Class: `GPUModelRunner`

The main model execution runner. Inherits from `LoRAModelRunnerMixin`, `KVConnectorModelRunnerMixin`, and `ECConnectorModelRunnerMixin`.

```python
class GPUModelRunner(
    LoRAModelRunnerMixin,
    KVConnectorModelRunnerMixin,
    ECConnectorModelRunnerMixin,
):
```

#### Constructor

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    device: torch.device,
):
```

**Initializes:**
- Model configuration references
- Device and dtype settings
- KV cache data type
- Pooling model flag
- Multimodal support flags
- Sampler
- Speculative decoding drafter (if configured)
- Input batch
- Persistent GPU buffers for CUDA graphs
- Encoder cache
- Workspace manager

#### Key Attributes

```python
# Model
self.model: nn.Module          # Set after load_model()
self.kv_caches: list[torch.Tensor]  # KV cache tensors

# Batch state
self.input_batch: InputBatch
self.requests: dict[str, CachedRequestState]

# Persistent GPU buffers
self.input_ids: CpuGpuBuffer           # [max_num_tokens]
self.positions: torch.Tensor           # [max_num_tokens]
self.query_start_loc: CpuGpuBuffer    # [max_num_reqs + 1]
self.seq_lens: torch.Tensor           # [max_num_reqs]
self.num_computed_tokens: torch.Tensor  # [max_num_reqs]
self.req_indices: CpuGpuBuffer        # [max_num_tokens]
self.num_scheduled_tokens: CpuGpuBuffer  # [max_num_reqs]

# Position encoding
self.mrope_positions: CpuGpuBuffer | None   # [3, max_num_tokens + 1]
self.xdrope_positions: CpuGpuBuffer | None  # [xdim, max_num_tokens + 1]

# CUDA graph
self.cudagraph_batch_sizes: list[int]
self.cudagraph_dispatcher: CudagraphDispatcher

# Speculative decoding
self.drafter: NgramProposer | EagleProposer | DraftModelProposer | ...
self.rejection_sampler: RejectionSampler
self.num_spec_tokens: int
```

#### `load_model(load_dummy_weights)`

Loads the model onto the GPU device using the configured model loader.

#### `initialize_kv_cache(kv_cache_config)`

Allocates KV cache tensors based on the configuration:

1. Creates block tables for each KV cache group
2. Sets up attention groups (backend + metadata builders)
3. Allocates KV cache tensors
4. Binds KV caches to model layers
5. Handles KV sharing layers
6. Initializes mamba state management (if applicable)

#### `execute_model(scheduler_output, intermediate_tensors) -> ModelRunnerOutput | IntermediateTensors | ExecuteModelState | None`

The main model execution method.

**Process:**
1. **Update states**: `_update_states(scheduler_output)` - Updates cached request states and persistent batch
2. **Early exit**: If no tokens scheduled, handle KV connector output and return
3. **Determine batch**: `_determine_batch_execution_and_padding()` - Compute batch size and padding
4. **Prepare inputs**: `_prepare_inputs()` - Build input tensors (input_ids, positions, attention metadata)
5. **Forward pass**: Execute the model
6. **Handle outputs**: Process logits, sampling, speculative decoding
7. **Return**: ModelRunnerOutput or ExecuteModelState (for split execute/sample)

#### `_update_states(scheduler_output) -> Callable | None`

Updates cached request states and the persistent batch from the scheduler output.

**Process:**
1. Remove finished requests from cached states
2. Zero newly allocated cache blocks
3. Free cached encoder outputs
4. Remove unscheduled requests from batch
5. Add new requests to cached states and batch
6. Update running/resumed request states
7. Handle speculative decode corrections (async scheduling)
8. Condense the batch (compact after removals)
9. Reorder batch (attention backend optimization)
10. Refresh metadata (sampling parameters)

Returns a correction callback for async speculative decode.

#### `_prepare_inputs(scheduler_output, num_scheduled_tokens) -> tuple[logits_indices, spec_decode_metadata]`

Prepares all input tensors for the model forward pass.

**Process:**
1. Compute request indices and cumulative token counts
2. Compute positions (standard, M-RoPE, or XD-RoPE)
3. Build input_ids from token_ids_cpu
4. Handle prompt embeddings
5. Build query_start_loc
6. Compute optimistic sequence lengths
7. Build discard_request_mask
8. Sync num_accepted_tokens (speculative decode)
9. Update num_computed_tokens on GPU
10. Compute slot mapping for KV cache
11. Copy input_ids to GPU
12. Handle LoRA activation
13. Build attention metadata

#### `profile_run()`

Executes a dummy forward pass for memory profiling.

#### `profile_cudagraph_memory() -> int`

Estimates CUDA graph memory consumption.

#### `_dummy_run(num_tokens, ...) -> tuple[hidden_states, last_hidden_states]`

Executes a dummy forward pass with the given number of tokens. Used for:
- Memory profiling
- Compilation warmup
- Sampler warmup
- CUDA graph capture

#### `capture_model() -> int`

Captures CUDA graphs for specified batch sizes. Returns CUDA graph memory usage in bytes.

#### `sample_tokens(grammar_output) -> ModelRunnerOutput`

Samples tokens from the model's output logits.

**Process:**
1. Retrieves `ExecuteModelState` from previous `execute_model()` call
2. Applies grammar bitmask (structured output)
3. Samples from logits using `Sampler`
4. Handles speculative decoding (rejection sampling)
5. Processes draft token IDs
6. Builds output (async or sync)

### ExecuteModelState

```python
class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
```

### AsyncGPUModelRunnerOutput

```python
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Overlapped execution output that copies tensors to CPU asynchronously."""

    def get_output(self) -> ModelRunnerOutput:
        """Block until copy is done, then return the output."""
```

---

## 10. InputBatch (GPUInputBatch)

**File**: `vllm/v1/worker/gpu_input_batch.py`

### Class: `InputBatch`

Manages the persistent batch state for GPU model execution. All request metadata is stored in contiguous arrays for efficient GPU transfer.

```python
class InputBatch:
    max_num_reqs: int
    max_model_len: int
    max_num_batched_tokens: int
    device: torch.device
    pin_memory: bool
    vocab_size: int
```

#### Constructor

```python
def __init__(
    self,
    max_num_reqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    device: torch.device,
    pin_memory: bool,
    vocab_size: int,
    block_sizes: list[int],
    kernel_block_sizes: list[int],
    max_num_blocks_per_req: list[int] | None = None,
    logitsprocs: LogitsProcessors | None = None,
    logitsprocs_need_output_token_ids: bool = False,
    num_spec_tokens: int = 0,
    is_pooling_model: bool = False,
    cp_kv_cache_interleave_size: int = 1,
    reasoning_config: ReasoningConfig | None = None,
):
```

**Allocates:**
- Token ID buffers (CPU + GPU)
- Sampling parameter buffers (temperature, top_p, top_k, penalties)
- Block tables for each KV cache group
- LoRA mapping arrays
- Generator dictionary
- Logits processor state
- Pooling parameter storage
- Speculative decode token storage

#### Key State Arrays

| Array | Shape | Device | Description |
|-------|-------|--------|-------------|
| `token_ids_cpu` | `[max_num_reqs, max_model_len]` | CPU | Token IDs per request |
| `is_token_ids` | `[max_num_reqs, max_model_len]` | CPU | Is-token-vs-embed mask |
| `num_tokens_no_spec` | `[max_num_reqs]` | CPU | Token count without spec |
| `num_prompt_tokens` | `[max_num_reqs]` | CPU | Prompt token count |
| `num_computed_tokens_cpu` | `[max_num_reqs]` | CPU | Computed token count |
| `temperature` | `[max_num_reqs]` | GPU | Sampling temperature |
| `top_p` | `[max_num_reqs]` | GPU | Top-p sampling |
| `top_k` | `[max_num_reqs]` | GPU | Top-k sampling |
| `frequency_penalties` | `[max_num_reqs]` | GPU | Frequency penalty |
| `presence_penalties` | `[max_num_reqs]` | GPU | Presence penalty |
| `repetition_penalties` | `[max_num_reqs]` | GPU | Repetition penalty |
| `request_lora_mapping` | `[max_num_reqs]` | CPU | Per-request LoRA ID |
| `block_table` | `MultiGroupBlockTable` | CPU+GPU | KV cache block tables |

#### Methods

##### `add_request(request: CachedRequestState) -> int`

Adds a request to the batch, returning its index.

**Process:**
1. Allocate a slot (fill empty indices first, append otherwise)
2. Copy token IDs to `token_ids_cpu`
3. Copy sampling parameters to CPU/GPU buffers
4. Add block table entries
5. Set up LoRA mapping
6. Set up generators (for seeded sampling)
7. Set up logprob tracking

##### `remove_request(req_id: str) -> int | None`

Removes a request from the batch.

**Process:**
1. Look up request index
2. Clear token IDs, sampling params, block table
3. Remove LoRA mapping
4. Clean up generator, logprobs, penalties
5. Return the freed index

##### `swap_states(i1: int, i2: int)`

Swaps all state between two batch indices. Used for batch reordering.

##### `condense()`

Compacts the batch by moving active requests to lower indices. Called after removing requests to maintain contiguous batch layout.

##### `refresh_metadata()`

Updates sampling metadata after batch changes. Applies logits processor state updates.

##### `make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)`

Generates LoRA mapping inputs for the current batch.

```python
def make_lora_inputs(
    self,
    num_scheduled_tokens: np.ndarray,
    num_sampled_tokens: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
```

**Returns:**
1. `prompt_lora_mapping`: Per-sampled-token LoRA IDs
2. `token_lora_mapping`: Per-scheduled-token LoRA IDs
3. `lora_requests`: Set of active LoRA requests

#### Properties

| Property | Description |
|----------|-------------|
| `num_reqs` | Number of active requests |
| `req_ids` | List of active request IDs |
| `all_greedy` | Whether all requests use greedy sampling |
| `all_random` | Whether all requests use random sampling |
| `no_top_p` | Whether no requests use top-p |
| `no_top_k` | Whether no requests use top-k |
| `no_penalties` | Whether no requests use penalties |
| `max_num_logprobs` | Maximum logprobs count across requests |

---

## 11. CachedRequestState

**File**: `vllm/v1/worker/gpu_input_batch.py`

### Class: `CachedRequestState`

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]
    mrope_positions: torch.Tensor | None
    mrope_position_delta: int | None
    xdrope_positions: torch.Tensor | None
    lora_request: LoRARequest | None
    prompt_embeds: torch.Tensor | None
    in_progress_prompt_logprobs_cpu: LogprobsTensors | None
    prompt_is_token_ids: list[bool] | None
    prev_num_draft_len: int = 0
    pooling_params: PoolingParams | None = None
    pooling_states: PoolingStates | None = None
```

#### Key Properties

- **`num_prompt_tokens`**: Length of prompt tokens or embeddings (set in `__post_init__`)
- **`num_tokens`**: Total tokens = prompt tokens + output tokens

#### Methods

##### `get_token_id(idx: int) -> int`

Returns the token ID at the given index. Handles both prompt and output tokens. Returns `-1` for indices beyond the current token count.

---

## 12. Block Table Management

**File**: `vllm/v1/worker/block_table.py`

### Class: `BlockTable`

Manages the KV cache block table for a single KV cache group.

```python
class BlockTable:
    block_size: int                    # Kernel block size
    blocks_per_kv_block: int           # Ratio of allocation to kernel block size
    use_hybrid_blocks: bool            # Whether block sizes differ
    max_num_reqs: int
    max_num_blocks_per_req: int
    block_table: CpuGpuBuffer          # [max_num_reqs, max_num_blocks_per_req]
    num_blocks_per_row: np.ndarray     # [max_num_reqs]
    slot_mapping: CpuGpuBuffer         # [max_num_batched_tokens]
```

#### Hybrid Block Support

When the KV cache allocation block size differs from the kernel block size:

```python
if block_size % kernel_block_size != 0:
    raise ValueError("kernel_block_size must divide block_size evenly")

self.blocks_per_kv_block = block_size // kernel_block_size
self.use_hybrid_blocks = True
```

Example: Allocation block size = 32, kernel block size = 16. Each allocation block maps to 2 kernel blocks.

#### Methods

##### `append_row(block_ids, row_idx)`

Appends block IDs to an existing row.

##### `add_row(block_ids, row_idx)`

Replaces block IDs for a row (clears first).

##### `clear_row(row_idx)`

Clears all block IDs for a row.

##### `move_row(src, tgt)`

Copies block IDs from source row to target row.

##### `swap_row(src, tgt)`

Swaps block IDs between two rows.

##### `compute_slot_mapping(num_reqs, query_start_loc, positions)`

Computes the slot mapping tensor using a Triton kernel. Maps each token position to its corresponding KV cache slot.

**Triton Kernel**: `_compute_slot_mapping_kernel`

Handles:
- Block ID lookup from block table
- Position-to-block mapping
- Context parallel (CP) interleaving
- Padding for CUDA graph compatibility

##### `commit_block_table(num_reqs)`

Copies the CPU block table to GPU for the active number of requests.

##### `get_device_tensor(num_reqs)`

Returns the GPU block table tensor.

### Class: `MultiGroupBlockTable`

Manages block tables for multiple KV cache groups.

```python
class MultiGroupBlockTable:
    block_tables: list[BlockTable]  # One per KV cache group
```

Delegates operations to each individual `BlockTable`.

### `map_to_kernel_blocks(kv_manager_block_ids, blocks_per_kv_block, kernel_block_arange)`

Static method that converts allocation block IDs to kernel block IDs.

```python
# Example:
# kv_manager_block_ids = [0, 1, 2], blocks_per_kv_block = 2
# Result: [0, 1, 2, 3, 4, 5]
```

---

## 13. UBatching Framework

### Overview

UBatching (microbatching) splits a batch into multiple micro-batches for execution. This enables:
- **DBO (Decoupled Batch Optimization)**: Overlapping communication with computation in MoE models
- **SM Control**: Partitioning streaming multiprocessors between communication and computation

### UBatchSlice

**File**: `vllm/v1/worker/ubatch_utils.py`

```python
@dataclass
class UBatchSlice:
    request_slice: slice    # Range of requests in this micro-batch
    token_slice: slice      # Range of tokens in this micro-batch

    def is_empty(self) -> bool:
        return (self.request_slice.start == self.request_slice.stop
                or self.token_slice.start == self.token_slice.stop)

    @property
    def num_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start
```

### `check_ubatch_thresholds(config, num_tokens, uniform_decode) -> bool`

Determines whether ubatching should be used based on token count thresholds.

```python
def check_ubatch_thresholds(
    config: ParallelConfig,
    num_tokens: int,
    uniform_decode: bool,
) -> bool:
```

- For decode: `num_tokens >= dbo_decode_token_threshold`
- For prefill: `num_tokens >= dbo_prefill_token_threshold`

### `maybe_create_ubatch_slices(...) -> tuple[UBatchSlices | None, UBatchSlices | None]`

Creates micro-batch slices by splitting the token sequence at regular intervals.

```python
def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
```

**Returns:** `(ubatch_slices, ubatch_slices_padded)` - Unpadded and padded variants.

### UBatchWrapper

**File**: `vllm/v1/worker/gpu_ubatch_wrapper.py`

```python
class UBatchWrapper:
    runnable: Callable
    vllm_config: VllmConfig
    comm_stream: torch.cuda.Stream
    ready_barrier: threading.Barrier
    cudagraphs: dict[int, CUDAGraphMetaData]
    sm_control: SMControlContextManager
```

Wraps a model runnable to support microbatched execution:

1. Splits the batch into micro-batches
2. Executes each micro-batch on a separate thread
3. Controls SM allocation between communication and computation
4. Captures CUDA graphs for micro-batches

### SMControlContextManager

```python
class SMControlContextManager:
    """Context manager for controlling SM allocation."""
    total_sms: int
    compute_sms: int      # = total_sms - comm_sms
    comm_sms: int
```

On enter: Sets communication and computation SM counts.
On exit: Restores all SMs for general use.

---

## 14. Workspace Management

**File**: `vllm/v1/worker/workspace.py`

### Class: `WorkspaceManager`

```python
class WorkspaceManager:
    _device: torch.device
    _num_ubatches: int
    _current_workspaces: list[torch.Tensor | None]  # One per ubatch slot
    _locked: bool
```

Manages workspace buffer allocation for operations that need temporary GPU memory (e.g., MoE expert computation).

#### Methods

##### `get_simultaneous(*shapes_and_dtypes) -> list[torch.Tensor]`

Allocates multiple workspace tensors from a single contiguous buffer.

```python
def get_simultaneous(
    self,
    *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
) -> list[torch.Tensor]:
```

**Behavior:**
1. Computes total bytes needed (with 256-byte alignment)
2. Calls `_ensure_workspace_size(total_bytes)` to allocate/grow
3. Returns views into the workspace buffer

##### `lock()` / `unlock()`

Lock/unlock the workspace to prevent/allow growth during execution.

##### `_ensure_workspace_size(required_bytes) -> torch.Tensor`

Ensures the workspace is large enough:
1. Checks current size
2. If too small and unlocked, allocates a new buffer
3. If too small and locked, raises `AssertionError`

### Global Functions

```python
def init_workspace_manager(device, num_ubatches=None): ...
def is_workspace_manager_initialized() -> bool: ...
def lock_workspace(): ...  # Context manager
```

---

## 15. CUDA Graph Handling

### CudagraphDispatcher

**File**: `vllm/v1/cudagraph_dispatcher.py`

Dispatches model execution to the appropriate CUDA graph based on batch size and configuration.

### CUDA Graph Capture Flow

1. **Determine capture sizes**: Based on `cudagraph_capture_sizes` and `cudagraph_mode`
2. **For each size**:
   a. Create dummy inputs
   b. Set up LoRA (if enabled)
   c. Capture CUDA graph
3. **Store graphs**: Indexed by batch size and configuration

### CUDA Graph Execution

At runtime:
1. Look up the graph for the current batch size
2. Copy inputs to the graph's input buffers
3. Replay the graph
4. Read outputs from the graph's output buffers

### Encoder CUDA Graphs

**File**: `vllm/v1/worker/encoder_cudagraph.py`

The `EncoderCudaGraphManager` captures and replays CUDA graphs for encoder (vision) model forward passes. This is particularly beneficial for multimodal models where the encoder is called repeatedly.

---

## 16. KV Connector Model Runner Mixin

**File**: `vllm/v1/worker/kv_connector_model_runner_mixin.py`

### Class: `KVConnectorModelRunnerMixin`

Mixin for KV transfer connector support in model runners.

```python
class KVConnectorModelRunnerMixin:
    @staticmethod
    def kv_connector_no_forward(scheduler_output, vllm_config): ...

    @staticmethod
    def maybe_get_kv_connector_output(scheduler_output, defer_finalize=False): ...

    @staticmethod
    def finalize_kv_connector(): ...

    @staticmethod
    def use_uniform_kv_cache(attn_groups, cache_dtype): ...

    @staticmethod
    def allocate_uniform_kv_caches(kv_cache_config, attn_groups, cache_dtype,
                                   device, kernel_block_sizes): ...
```

### Key Methods

#### `kv_connector_no_forward(scheduler_output, vllm_config)`

Handles KV send/recv even when no model forward pass is needed (e.g., when all requests are just loading KV cache).

#### `maybe_get_kv_connector_output(scheduler_output, defer_finalize)`

Context manager that manages the KV connector lifecycle:

1. Binds connector metadata from scheduler output
2. Starts background KV cache transfers (`start_load_kv`)
3. Yields the output
4. Waits for saves to complete (`wait_for_save`)
5. Gets finished requests (`get_finished`)
6. Clears connector metadata

#### `use_uniform_kv_cache(attn_groups, cache_dtype) -> bool`

Determines whether to use a uniform KV layout where all layers share the same tensor. Required for efficient KV transfer across layers.

**Conditions:**
1. Single KV cache group with same page size
2. KV connector configured and prefers cross-layer blocks
3. Attention backend supports the layout

#### `allocate_uniform_kv_caches(...)`

Allocates a single shared KV cache tensor for all layers, enabling efficient cross-layer KV transfer.

---

## 17. EC Connector Model Runner Mixin

**File**: `vllm/v1/worker/ec_connector_model_runner_mixin.py`

### Class: `ECConnectorModelRunnerMixin`

Mixin for encoder cache transfer connector support.

```python
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(encoder_cache, mm_hash): ...

    @staticmethod
    def maybe_get_ec_connector_output(scheduler_output, encoder_cache, **kwargs): ...
```

### Key Methods

#### `maybe_save_ec_to_connector(encoder_cache, mm_hash)`

Saves encoder outputs to the EC connector for transfer to consumer workers.

#### `maybe_get_ec_connector_output(scheduler_output, encoder_cache, **kwargs)`

Context manager that manages the EC connector lifecycle:

1. Binds connector metadata
2. Loads caches for consumer workers (`start_load_caches`)
3. Yields the output
4. Gets finished requests
5. Clears connector metadata

---

## 18. LoRA Model Runner Mixin

See Document 13 (LoRA Adapters Reference) for full details.

**File**: `vllm/v1/worker/lora_model_runner_mixin.py`

The mixin provides:
- `load_lora_model()`: Initialize LoRA support
- `set_active_loras()`: Set active adapters for current batch
- `maybe_setup_dummy_loras()`: Context manager for warmup
- `maybe_select_dummy_loras()`: Context manager for CG capture
- `maybe_dummy_run_with_lora()`: Combined warmup context manager
- `add_lora()`, `remove_lora()`, `pin_lora()`, `list_loras()`: Adapter management

---

## 19. Async Scheduling Support

### Overview

Async scheduling overlaps the scheduler and model execution. While the model processes batch N, the scheduler prepares batch N+1.

### Implementation

#### GPUModelRunner Async Support

- **`async_output_copy_stream`**: Separate CUDA stream for copying outputs to CPU
- **`prepare_inputs_event`**: CUDA event for synchronizing input preparation
- **`AsyncGPUModelRunnerOutput`**: Wraps output with async CPU copy
- **`prev_req_id_to_index`**: Maps request IDs to previous batch positions for GPU-based reindexing

#### Async Scheduling Flow

1. **Step N**: Model executes batch N
2. **Step N+1 (overlap)**:
   a. Scheduler produces batch N+1
   b. Model runner calls `_update_states()` for batch N+1
   c. `_prepare_inputs()` builds GPU inputs
   d. For decode-only batches, input IDs are gathered from GPU tensors (no CPU round-trip)
3. **Synchronization**: After model execution completes, outputs are copied to CPU asynchronously

### Async Speculative Decode

When both async scheduling and speculative decoding are enabled:
1. Draft tokens from the previous step are used optimistically
2. GPU-based correction is applied after sampling
3. `update_num_computed_tokens_for_batch_change()` kernel adjusts computed tokens on GPU

---

## 20. Speculative Decoding in Model Runner

### Supported Methods

| Method | Class | Description |
|--------|-------|-------------|
| `"ngram"` | `NgramProposer` | CPU-based n-gram matching |
| `"ngram_gpu"` | `NgramProposerGPU` | GPU-accelerated n-gram |
| `"eagle"` / `"eagle3"` | `EagleProposer` | EAGLE hidden-state drafting |
| `"medusa"` | `MedusaProposer` | Medusa multi-head drafting |
| `"dflash"` | `DFlashProposer` | DFlash speculative decoding |
| `"suffix"` | `SuffixDecodingProposer` | Suffix array-based drafting |
| `"extract_hidden_states"` | `ExtractHiddenStatesProposer` | Hidden state extraction |
| (draft model) | `DraftModelProposer` | Small draft model |

### Speculative Decode Flow in ModelRunner

1. **Draft generation**: Before scheduling, the drafter proposes draft tokens
2. **Scheduling**: Scheduler includes draft tokens in the output
3. **Input preparation**: `_prepare_inputs()` creates `SpecDecodeMetadata`
4. **Forward pass**: Model processes all tokens (including draft)
5. **Logits extraction**: Extract logits at draft positions
6. **Rejection sampling**: `RejectionSampler` accepts/rejects draft tokens
7. **Output**: Return accepted tokens + newly sampled tokens

### RejectionSampler

```python
class RejectionSampler:
    def __init__(self, sampler, speculative_config, device):
        self.sampler = sampler
        ...

    @staticmethod
    def parse_output(sampled_token_ids, vocab_size, invalid_req_indices,
                     logprobs_tensors=None):
        """Parse rejection sampling output, removing rejected tokens."""
```

---

## 21. Attention Metadata Building

### `_build_attention_metadata(...)`

```python
def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
    slot_mappings: dict[int, torch.Tensor] | None = None,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
```

**Process:**
1. Build `CommonAttentionMetadata` with query_start_loc, seq_lens, block tables
2. For each KV cache group and attention group:
   a. Get the metadata builder
   b. Build attention metadata (or reuse cached if possible)
   c. Store per-layer metadata
3. Handle cascade attention prefix lengths
4. Handle mm_prefix_lm (bidirectional attention for images)
5. Handle speculative decode metadata

### Cascade Attention

```python
def _compute_cascade_attn_prefix_len(
    self,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_common_prefix_blocks: int,
    kv_cache_spec: KVCacheSpec,
    attn_metadata_builder: AttentionMetadataBuilder,
) -> int:
```

Computes the common prefix length for cascade attention, which splits attention into:
1. **Prefix attention**: Bi-directional attention over shared prefix
2. **Suffix attention**: Causal attention over remaining tokens

The prefix length is:
- Capped by the minimum `num_computed_tokens` across requests
- Rounded down to block size
- Determined by the attention backend's `use_cascade_attention()` heuristic

---

## 22. Utility Classes

### KVBlockZeroer

**File**: `vllm/v1/worker/utils.py`

```python
class KVBlockZeroer:
    """Manages efficient zeroing of KV cache blocks via a Triton kernel."""
```

Zeros newly allocated KV cache blocks to prevent stale data from corrupting attention computation.

### AttentionGroup

**File**: `vllm/v1/worker/utils.py`

```python
@dataclass
class AttentionGroup:
    kv_cache_group_id: int
    attn_group_id: int
    layer_names: list[str]
    backend: AttentionBackend
    kv_cache_spec: KVCacheSpec
    metadata_builders: list[AttentionMetadataBuilder]  # One per ubatch
```

Groups attention layers that share the same backend and KV cache specification.

### Utility Functions

**File**: `vllm/v1/worker/utils.py`

- **`bind_kv_cache(kv_caches, forward_context, ...) `**: Binds KV cache tensors to model attention layers.
- **`prepare_kernel_block_sizes(...)`**: Computes kernel block sizes for each KV cache group.
- **`add_kv_sharing_layers_to_kv_cache_groups(...)`**: Adds KV sharing layers to cache groups.
- **`sanity_check_mm_encoder_outputs(...)`**: Validates multimodal encoder output shapes.

---

## 23. Full Parameter Reference

### GPUModelRunner Constructor Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `vllm_config` | Constructor | Complete vLLM configuration |
| `device` | Constructor | Target GPU device |

### InputBatch Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_num_reqs` | `int` | Max concurrent requests |
| `max_model_len` | `int` | Max context length |
| `max_num_batched_tokens` | `int` | Max tokens per batch |
| `device` | `torch.device` | Target device |
| `pin_memory` | `bool` | Use pinned memory |
| `vocab_size` | `int` | Vocabulary size |
| `block_sizes` | `list[int]` | Block sizes per KV cache group |
| `kernel_block_sizes` | `list[int]` | Kernel block sizes per group |
| `max_num_blocks_per_req` | `list[int] \| None` | Max blocks per request per group |
| `logitsprocs` | `LogitsProcessors \| None` | Custom logits processors |
| `num_spec_tokens` | `int` | Number of speculative tokens |
| `is_pooling_model` | `bool` | Whether this is a pooling model |

### BlockTable Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_size` | `int` | KV cache block size |
| `max_num_reqs` | `int` | Max concurrent requests |
| `max_num_blocks_per_req` | `int` | Max blocks per request |
| `max_num_batched_tokens` | `int` | Max tokens per batch |
| `pin_memory` | `bool` | Use pinned memory |
| `device` | `torch.device` | Target device |
| `kernel_block_size` | `int` | Attention kernel block size |
| `cp_kv_cache_interleave_size` | `int` | CP interleave size |

### Executor Selection Reference

| Config | Executor | Use Case |
|--------|----------|----------|
| `distributed_executor_backend="uni"` | `UniProcExecutor` | Single GPU |
| `distributed_executor_backend="mp"` | `MultiprocExecutor` | Multi-GPU (same node) |
| `distributed_executor_backend="ray"` | `RayDistributedExecutor` | Multi-GPU / multi-node |
| `distributed_executor_backend="external_launcher"` | `ExecutorWithExternalLauncher` | Torchrun offline inference |

### Worker Method Reference

| Method | Description |
|--------|-------------|
| `init_device()` | Initialize GPU, set up distributed env |
| `load_model(load_dummy_weights)` | Load model weights onto GPU |
| `determine_available_memory()` | Profile memory and return available bytes |
| `initialize_from_config(kv_cache_config)` | Allocate KV cache |
| `compile_or_warm_up_model()` | Compile and capture CUDA graphs |
| `execute_model(scheduler_output)` | Execute model forward pass |
| `sample_tokens(grammar_output)` | Sample tokens from logits |
| `sleep(level)` | Offload GPU memory |
| `wake_up(tags)` | Restore offloaded memory |
| `add_lora(lora_request)` | Add LoRA adapter |
| `remove_lora(lora_id)` | Remove LoRA adapter |
| `update_weights(update_info)` | Live weight update |
| `check_health()` | Health check |
| `shutdown()` | Clean up resources |
