# Scheduling and Memory Management Reference

This document provides a comprehensive reference for the V1 scheduling system and KV cache memory management in vLLM, covering the scheduler architecture, scheduling policies, KV cache block management, prefix caching, preemption, and configuration.

---

## Table of Contents

1. [Scheduler Architecture Overview](#1-scheduler-architecture-overview)
2. [Configuration: SchedulerConfig](#2-configuration-schedulerconfig)
3. [Configuration: CacheConfig](#3-configuration-cacheconfig)
4. [SchedulerInterface](#4-schedulerinterface)
5. [Scheduler Output Types](#5-scheduler-output-types)
6. [Request Queues](#6-request-queues)
7. [Main Scheduler Implementation](#7-main-scheduler-implementation)
8. [KV Cache Block Management](#8-kv-cache-block-management)
9. [Block Pool](#9-block-pool)
10. [KVCacheManager](#10-kvcachemanager)
11. [SingleTypeKVCacheManager](#11-singletypekvcachemanager)
12. [KVCacheCoordinator](#12-kvcachecoordinator)
13. [KV Cache Utilities](#13-kv-cache-utilities)
14. [Encoder Cache Management](#14-encoder-cache-management)
15. [Prefix Caching](#15-prefix-caching)
16. [Preemption Strategies](#16-preemption-strategies)
17. [Speculative Decoding Support](#17-speculative-decoding-support)
18. [KV Connector for P/D Disaggregation](#18-kv-connector-for-pd-disaggregation)
19. [Watermarks and Thresholds](#19-watermarks-and-thresholds)
20. [PauseState Management](#20-pausestate-management)
21. [Full Parameter Reference](#21-full-parameter-reference)

---

## 1. Scheduler Architecture Overview

The V1 scheduler is responsible for deciding which requests to process in each scheduling step (forward pass). It manages:

- **Request queues**: Waiting, running, and finished requests
- **KV cache allocation**: Block-level memory management for key-value caches
- **Scheduling policies**: FCFS (First-Come-First-Served) or Priority-based
- **Preemption**: Evicting running requests when memory is insufficient
- **Prefix caching**: Reusing KV cache blocks across requests with shared prefixes

### Architecture Diagram

```
EngineCore
  --> Scheduler
       --> RequestQueue (FCFS or Priority)
       --> KVCacheManager
            --> KVCacheCoordinator (NoPrefixCache, Unitary, or Hybrid)
                 --> BlockPool
                      --> FreeKVCacheBlockQueue (doubly linked list)
                      --> KVCacheBlock (hash-based dedup for prefix caching)
       --> EncoderCacheManager (multimodal encoder outputs)
```

### Source Files

| File | Purpose |
|------|---------|
| `vllm/config/scheduler.py` | SchedulerConfig |
| `vllm/config/cache.py` | CacheConfig |
| `vllm/v1/core/sched/interface.py` | SchedulerInterface ABC, PauseState |
| `vllm/v1/core/sched/output.py` | SchedulerOutput, GrammarOutput dataclasses |
| `vllm/v1/core/sched/scheduler.py` | Main Scheduler implementation (~2300 lines) |
| `vllm/v1/core/sched/request_queue.py` | RequestQueue, FCFSRequestQueue, PriorityRequestQueue |
| `vllm/v1/core/kv_cache_manager.py` | KVCacheManager |
| `vllm/v1/core/kv_cache_coordinator.py` | KVCacheCoordinator hierarchy |
| `vllm/v1/core/block_pool.py` | BlockPool, FreeKVCacheBlockQueue |
| `vllm/v1/core/kv_cache_utils.py` | KVCacheBlock, hash utilities, block size computation |
| `vllm/v1/core/single_type_kv_cache_manager.py` | SingleTypeKVCacheManager |
| `vllm/v1/core/encoder_cache_manager.py` | EncoderCacheManager |

---

## 2. Configuration: SchedulerConfig

**File**: `vllm/config/scheduler.py`

### Class: `SchedulerConfig`

```python
@config
class SchedulerConfig:
    max_num_batched_tokens: int          # Max total tokens per batch
    max_num_seqs: int                    # Max concurrent sequences
    max_model_len: int                   # Max model context length
    num_scheduler_steps: int             # Scheduler steps per iteration
    multi_step_stream_outputs: bool      # Stream multi-step outputs
    enable_chunked_prefill: bool         # Enable chunked prefill scheduling
    max_num_partial_prefills: int        # Max concurrent partial prefills
    max_long_partial_prefills: int       # Max concurrent long partial prefills
    long_prefill_token_threshold: int    # Token threshold for long prefill classification
    policy: str                          # Scheduling policy: "fcfs" or "priority"
    scheduler_cls: str                   # Scheduler class qualified name
    async_scheduling: bool               # Enable async scheduling
    stream_interval: int                 # Streaming output interval
    scheduler_reserve_full_isl: bool     # Reserve full input sequence length
    max_num_encoder_input_tokens: int    # Max encoder input tokens (multimodal)
```

### Key Parameters

- **`max_num_batched_tokens`**: Maximum total number of tokens that can be processed in a single batch. This is the primary knob for throughput vs latency tradeoff. Higher values increase throughput but may increase latency.
- **`max_num_seqs`**: Maximum number of sequences (requests) that can be active simultaneously. Limits the number of concurrent requests regardless of their token counts.
- **`enable_chunked_prefill`**: When `True`, long prefill requests are split into chunks, allowing them to be interleaved with decode requests. This prevents long prefills from blocking decodes.
- **`max_num_partial_prefills`**: Maximum number of requests that can be in a partially-prefilled state simultaneously.
- **`policy`**: Scheduling policy:
  - `"fcfs"`: First-Come-First-Served (default). Requests are processed in arrival order.
  - `"priority"`: Priority-based scheduling. Requests are scheduled by priority value.
- **`async_scheduling`**: When `True`, scheduling and model execution overlap. The scheduler runs ahead while the model processes the previous batch.
- **`scheduler_reserve_full_isl`**: When `True`, the scheduler reserves KV cache space for the full input sequence length at scheduling time, preventing later preemption.

---

## 3. Configuration: CacheConfig

**File**: `vllm/config/cache.py`

### Class: `CacheConfig`

```python
@config
class CacheConfig:
    block_size: int                          # KV cache block size in tokens
    gpu_memory_utilization: float            # Fraction of GPU memory to use
    cache_dtype: str                         # KV cache data type ("auto", "fp8", etc.)
    enable_prefix_caching: bool              # Enable prefix caching
    prefix_caching_hash_algo: str            # Hash algorithm for prefix caching
    sliding_window: int | None               # Sliding window attention size
    kv_cache_memory_bytes: int | None        # Explicit KV cache memory size
    kv_offloading_size: int | None           # KV offloading memory size
    kv_offloading_backend: str | None        # KV offloading backend
    calculate_kv_scales: bool                # Calculate KV quantization scales
    mamba_cache_mode: str                    # Mamba cache mode ("align" or "default")
```

### Key Parameters

- **`block_size`**: Number of tokens per KV cache block. Default: `16`. This determines the granularity of KV cache allocation and prefix caching.
- **`gpu_memory_utilization`**: Fraction of total GPU memory to allocate for KV cache. Default: `0.9`. The remaining memory is used for model weights, activations, and CUDA graphs.
- **`cache_dtype`**: Data type for KV cache. `"auto"` uses the model's dtype. `"fp8_e5m2"` or `"fp8_e4m3"` enable FP8 quantization for reduced memory usage.
- **`enable_prefix_caching`**: When `True`, KV cache blocks are deduplicated across requests that share the same prompt prefix. This can significantly reduce memory usage and computation for shared prefixes.
- **`prefix_caching_hash_algo`**: Hash algorithm for prefix caching block identification. Options: `"sha256"`, `"builtin"`.
- **`kv_cache_memory_bytes`**: When set, explicitly specifies the KV cache memory size in bytes, bypassing the automatic memory profiling.
- **`sliding_window`**: When set, enables sliding window attention. Only the most recent `sliding_window` tokens are kept in KV cache for each request.

---

## 4. SchedulerInterface

**File**: `vllm/v1/core/sched/interface.py`

### Class: `SchedulerInterface` (ABC)

Abstract base class defining the scheduler contract.

```python
class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(self, vllm_config, kv_cache_config, structured_output_manager,
                 block_size, hash_block_size, mm_registry, include_finished_set,
                 log_stats) -> None: ...

    @abstractmethod
    def schedule(self) -> SchedulerOutput: ...

    @abstractmethod
    def get_grammar_bitmask(self, scheduler_output) -> GrammarOutput | None: ...

    @abstractmethod
    def update_from_output(self, scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]: ...

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids) -> None: ...

    @abstractmethod
    def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output) -> None: ...

    @abstractmethod
    def add_request(self, request) -> None: ...

    @abstractmethod
    def finish_requests(self, request_ids, finished_status) -> list[tuple[str, int]]: ...

    @abstractmethod
    def get_num_unfinished_requests(self) -> int: ...

    @abstractmethod
    def has_finished_requests(self) -> bool: ...

    @abstractmethod
    def pause_state(self) -> PauseState: ...

    @abstractmethod
    def set_pause_state(self, pause_state) -> None: ...

    @abstractmethod
    def reset_prefix_cache(self, reset_running_requests, reset_connector) -> bool: ...

    @abstractmethod
    def reset_encoder_cache(self) -> None: ...

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]: ...

    @abstractmethod
    def make_stats(self) -> SchedulerStats | None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    def get_kv_connector(self) -> KVConnectorBase_V1 | None: ...
```

### PauseState Enum

```python
class PauseState(enum.IntEnum):
    UNPAUSED = 0     # Normal operation
    PAUSED_NEW = 1   # No new requests scheduled; running requests continue
    PAUSED_ALL = 2   # No requests scheduled at all
```

---

## 5. Scheduler Output Types

**File**: `vllm/v1/core/sched/output.py`

### NewRequestData

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    prompt_embeds: torch.Tensor | None
    prompt_is_token_ids: list[bool] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
```

### CachedRequestData

```python
@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    num_computed_tokens: list[int]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_output_tokens: list[int]
    new_token_ids: list[list[int]]  # For non-last PP rank
    all_token_ids: dict[str, list[int]]  # For async scheduling
```

### SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    total_num_scheduled_tokens: int
    num_scheduled_tokens: dict[str, int]        # req_id -> num_tokens
    total_num_scheduled_spec_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]  # req_id -> draft tokens
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    finished_req_ids: set[str]
    free_encoder_mm_hashes: set[str]
    kv_connector_metadata: KVConnectorMetadata | None
    ec_connector_metadata: ECConnectorMetadata | None
    structured_output_request_ids: dict[str, int]
    new_block_ids_to_zero: list[int]
```

### GrammarOutput

```python
@dataclass
class GrammarOutput:
    bitmask: torch.Tensor | None
    vocab_size: int
    context_logits: torch.Tensor | None
    structured_output_request_ids: dict[str, int]
```

---

## 6. Request Queues

**File**: `vllm/v1/core/sched/request_queue.py`

### RequestQueue (ABC)

```python
class RequestQueue(ABC):
    @abstractmethod
    def push(self, request: Request) -> None: ...

    @abstractmethod
    def pop(self) -> Request: ...

    @abstractmethod
    def peek(self) -> Request: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __bool__(self) -> bool: ...
```

### FCFSRequestQueue

First-Come-First-Served queue using `collections.deque`.

```python
class FCFSRequestQueue(RequestQueue):
    _queue: deque[Request]

    def push(self, request): self._queue.append(request)
    def pop(self): return self._queue.popleft()
    def peek(self): return self._queue[0]
```

### PriorityRequestQueue

Priority-based queue using `heapq`. Requests with higher priority (lower priority value) are scheduled first.

```python
class PriorityRequestQueue(RequestQueue):
    _heap: list[tuple[int, int, Request]]  # (priority, counter, request)
    _counter: int  # Tie-breaking counter for stable ordering

    def push(self, request):
        priority = request.priority or 0
        heapq.heappush(self._heap, (priority, self._counter, request))
        self._counter += 1

    def pop(self):
        return heapq.heappop(self._heap)[2]
```

---

## 7. Main Scheduler Implementation

**File**: `vllm/v1/core/sched/scheduler.py`

### Class: `Scheduler`

The main scheduler implementation, approximately 2300 lines. It manages request queues, KV cache allocation, preemption, and all scheduling decisions.

#### Constructor

```python
class Scheduler(
    SchedulerInterface,
    KVCacheManager,
    EncoderCacheManager,
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
```

**Initializes:**
- Request queues (FCFS or Priority based on `policy`)
- KV cache coordinator (NoPrefixCache, Unitary, or Hybrid)
- Block pool
- Encoder cache
- Speculative decoding state
- KV connector state
- Structured output manager

#### Key Attributes

```python
# Configuration
self.vllm_config: VllmConfig
self.scheduler_config: SchedulerConfig
self.cache_config: CacheConfig
self.lora_config: LoRAConfig | None

# State
self._pause_state: PauseState = PauseState.UNPAUSED
self.request_queue: RequestQueue
self.running: list[Request] = []
self.finished_req_ids: set[str] = set()
self._request_states: dict[str, Request] = {}

# KV Cache
self.block_pool: BlockPool
self.coordinator: KVCacheCoordinator

# Encoder cache (multimodal)
self.encoder_cache: dict[str, torch.Tensor]
self.encoder_cache_manager: EncoderCacheManager

# Speculative decoding
self.speculative_config: SpeculativeConfig | None
self.draft_manager: DraftManager | None
```

#### `schedule() -> SchedulerOutput`

The main scheduling method. Called once per scheduling step.

**Algorithm:**

1. **Check pause state**: If paused, return empty output
2. **Process waiting queue**:
   a. For each waiting request:
      - Compute encoder inputs (multimodal)
      - Check if the request fits in the budget
      - Allocate KV cache blocks
      - Move request to running
3. **Schedule running requests**:
   a. For each running request:
      - Determine number of tokens to schedule
      - Handle chunked prefill if enabled
      - Handle speculative decoding tokens
      - Allocate new KV cache blocks if needed
   b. If memory insufficient, preempt requests
4. **Build output**: Create `SchedulerOutput` with all scheduled requests

**Key scheduling parameters used:**
- `max_num_batched_tokens`: Limits total tokens per batch
- `max_num_seqs`: Limits concurrent requests
- `enable_chunked_prefill`: Enables prefill chunking
- `max_num_partial_prefills`: Limits partial prefills

#### `update_from_output(scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]`

Updates scheduler state after model execution.

**Process:**
1. Processes generated token IDs
2. Checks for stop conditions (EOS, stop strings, max tokens)
3. Updates speculative decoding state
4. Handles structured output validation
5. Returns outputs organized by client index

#### `_preempt_request(request, preemption_mode)`

Preempts a running request when memory is insufficient.

**Parameters:**
- `request`: The request to preempt
- `preemption_mode`: Either `"swap"` (move to waiting queue) or `"recompute"` (discard KV cache)

**Process:**
1. Free all KV cache blocks allocated to the request
2. Reset the request's computed tokens count
3. Move the request back to the waiting queue
4. If `preemption_mode` is `"swap"`, preserve the request state

#### `_try_schedule_encoder_inputs(requests, ...) -> list[Request]`

Attempts to schedule multimodal encoder computations for waiting requests.

**Process:**
1. For each request with multimodal inputs:
   - Check encoder cache budget
   - Allocate encoder cache space
   - Schedule encoder computation
2. Returns list of requests whose encoder inputs are ready

#### KV Connector Support

The scheduler supports KV transfer for prefill-decode disaggregation:

- **Prefill worker**: Sends KV cache to decode workers
- **Decode worker**: Receives KV cache from prefill workers
- Uses `kv_connector_metadata` in `SchedulerOutput` to communicate transfer information

#### EC Connector Support

The scheduler supports encoder cache transfer for multimodal P/D disaggregation:

- **Producer**: Sends encoder outputs to consumer workers
- **Consumer**: Receives encoder outputs from producer workers
- Uses `ec_connector_metadata` in `SchedulerOutput`

---

## 8. KV Cache Block Management

### KVCacheBlock

**File**: `vllm/v1/core/kv_cache_utils.py`

```python
@dataclass
class KVCacheBlock:
    block_id: int                          # Unique block identifier
    ref_count: int = 0                     # Reference count for sharing
    _hash: BlockHash | None = None         # Hash for prefix caching
    _prev_free: KVCacheBlock | None = None # Previous in free list
    _next_free: KVCacheBlock | None = None # Next in free list

    @property
    def block_hash(self) -> BlockHash | None:
        return self._hash

    def add_ref(self) -> None:
        self.ref_count += 1

    def remove_ref(self) -> int:
        assert self.ref_count > 0
        self.ref_count -= 1
        return self.ref_count

    def reset_hash(self) -> None:
        self._hash = None

    def set_hash(self, block_hash: BlockHash) -> None:
        self._hash = block_hash

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0
```

**Reference Counting**: When prefix caching is enabled, multiple requests can share the same KV cache block. The `ref_count` tracks how many requests are using the block. A block is freed only when `ref_count` reaches 0.

### FreeKVCacheBlockQueue

**File**: `vllm/v1/core/kv_cache_utils.py`

A doubly-linked list of free KV cache blocks for O(1) allocation and deallocation.

```python
class FreeKVCacheBlockQueue:
    _head: KVCacheBlock | None
    _tail: KVCacheBlock | None
    _num_free_blocks: int

    def push(self, block: KVCacheBlock) -> None:
        """Add a block to the tail of the free list."""

    def pop(self) -> KVCacheBlock:
        """Remove and return the block at the head of the free list."""

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a specific block from the free list (for prefix cache hits)."""

    @property
    def num_free_blocks(self) -> int: ...

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return all free blocks."""
```

### BlockHash Types

**File**: `vllm/v1/core/kv_cache_utils.py`

```python
# BlockHash is a tuple of (prefix_hash, last_token_ids)
# prefix_hash: Hash of all preceding blocks
# last_token_ids: Tuple of token IDs in the current block
BlockHash = tuple[int, tuple[int, ...]]

# BlockHashToBlockMap maps hash values to blocks for prefix caching
BlockHashToBlockMap = dict[BlockHash, KVCacheBlock]
```

### `hash_block_tokens(block_size, prev_block_hash, cur_block_tokens, ...)`

Computes the hash for a KV cache block.

```python
def hash_block_tokens(
    block_size: int,
    prev_block_hash: int | None,
    cur_block_tokens: tuple[int, ...],
    *,
    hash_algo: str = "builtin",
    extra_keys: tuple[int, ...] | None = None,
) -> BlockHash:
```

**Parameters:**
- `block_size`: Number of tokens per block
- `prev_block_hash`: Hash of the previous block (for chaining)
- `cur_block_tokens`: Token IDs in the current block
- `hash_algo`: Hash algorithm (`"builtin"` or `"sha256"`)
- `extra_keys`: Additional hash keys (e.g., for multi-modal features)

**Returns**: `BlockHash` tuple containing the computed hash and token IDs.

### `resolve_kv_cache_block_sizes(model_config, parallel_config, cache_config)`

Determines the block sizes for each KV cache group based on the model configuration.

```python
def resolve_kv_cache_block_sizes(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
) -> tuple[list[int], list[int], list[int]]:
    # Returns (block_sizes, gpu_block_sizes, num_cpu_blocks)
```

### `get_kv_cache_configs(kv_cache_spec, available_memory, block_sizes, ...)`

Computes the number of KV cache blocks that fit in available memory.

```python
def get_kv_cache_configs(
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
    block_sizes: list[int],
    ...
) -> list[KVCacheConfig]:
```

---

## 9. Block Pool

**File**: `vllm/v1/core/block_pool.py`

### Class: `BlockPool`

Manages the pool of KV cache blocks, handling allocation, deallocation, eviction, and prefix caching.

```python
class BlockPool:
    _block_size: int
    _num_total_blocks: int
    _free_blocks: FreeKVCacheBlockQueue
    _blocks: list[KVCacheBlock]
    _block_hashes: BlockHashToBlockMap  # For prefix caching
```

#### Key Methods

##### `__init__(num_blocks, block_size, enable_caching)`

Initialize the block pool with the given number of blocks.

**Parameters:**
- `num_blocks`: Total number of KV cache blocks
- `block_size`: Number of tokens per block
- `enable_caching`: Whether to enable prefix caching (hash-based dedup)

##### `get_cached_block(block_hash) -> KVCacheBlock | None`

Looks up a cached block by its hash. Returns `None` if not found.

```python
def get_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
```

When a cache hit occurs:
1. The cached block's `ref_count` is incremented
2. The block is removed from the free list (if it was free)
3. The block is returned

##### `cache_full_block(block, ...)`

Caches a fully computed block for future prefix cache hits.

```python
def cache_full_block(
    self,
    block: KVCacheBlock,
    ...
) -> None:
```

##### `get_new_blocks(num_blocks) -> list[KVCacheBlock]`

Allocates `num_blocks` new blocks from the free list.

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
```

Raises `ValueError` if insufficient free blocks.

##### `free_blocks(blocks, ...)`

Frees a list of blocks, decrementing their reference counts.

```python
def free_blocks(
    self,
    blocks: list[KVCacheBlock],
    ...
) -> None:
```

For each block:
1. Decrement `ref_count`
2. If `ref_count` reaches 0, add to free list
3. If prefix caching is enabled, keep the hash mapping

##### `touch(block)`

Updates the LRU ordering for a cached block. Called when a block is accessed.

##### `evict_blocks(num_blocks) -> list[KVCacheBlock]`

Evicts the least recently used blocks to free up space.

```python
def evict_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
```

Eviction strategy:
1. Iterate through blocks in LRU order
2. Skip blocks with `ref_count > 0` (in-use blocks)
3. Free blocks until `num_blocks` are reclaimed

##### `reset_prefix_cache() -> bool`

Resets all prefix cache entries. Frees all cached blocks.

```python
def reset_prefix_cache(self) -> bool:
```

Returns `True` if the cache was successfully reset.

##### Properties

- `num_free_blocks`: Number of blocks in the free list
- `num_cached_blocks`: Number of blocks with hash entries (prefix cache)
- `get_all_blocks`: Returns all blocks

---

## 10. KVCacheManager

**File**: `vllm/v1/core/kv_cache_manager.py`

### Class: `KVCacheBlocks`

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[list[KVCacheBlock], ...]  # Blocks per KV cache group
```

### Class: `KVCacheManager`

Manages KV cache allocation for all requests across multiple KV cache groups.

```python
class KVCacheManager:
    block_pool: BlockPool
    request_blocks: dict[str, KVCacheBlocks]  # req_id -> allocated blocks
    coordinator: KVCacheCoordinator
```

#### Key Methods

##### `get_computed_blocks(request) -> KVCacheBlocks`

Returns the already-computed (cached) blocks for a request. Used for prefix cache hits.

```python
def get_computed_blocks(self, request: Request) -> KVCacheBlocks:
```

##### `allocate_slots(request, num_tokens) -> list[KVCacheBlock]`

Allocates new KV cache blocks for a request to store `num_tokens` additional tokens.

```python
def allocate_slots(
    self,
    request: Request,
    num_tokens: int,
) -> list[KVCacheBlock]:
```

##### `free(request)`

Frees all KV cache blocks for a finished/preempted request.

```python
def free(self, request: Request) -> None:
```

##### `evict_blocks(num_blocks)`

Evicts blocks from the least recently used requests.

##### `reset_prefix_cache(reset_running_requests)`

Resets the prefix cache. If `reset_running_requests` is `True`, also preempts running requests.

##### `get_num_common_prefix_blocks(requests, kv_cache_group_id) -> int`

Returns the number of common prefix blocks shared by all given requests. Used for cascade attention.

##### `cache_blocks(request, kv_cache_blocks)`

Caches the given blocks for prefix cache lookup.

---

## 11. SingleTypeKVCacheManager

**File**: `vllm/v1/core/single_type_kv_cache_manager.py`

### Class: `SingleTypeKVCacheManager`

Abstract base class for managing KV cache of a single type (e.g., full attention, sliding window).

```python
class SingleTypeKVCacheManager(ABC):
    @abstractmethod
    def get_block_size(self) -> int: ...

    @abstractmethod
    def get_num_blocks(self) -> int: ...

    @abstractmethod
    def allocate(self, num_tokens: int) -> list[KVCacheBlock]: ...

    @abstractmethod
    def free(self, blocks: list[KVCacheBlock]) -> None: ...
```

---

## 12. KVCacheCoordinator

**File**: `vllm/v1/core/kv_cache_coordinator.py`

### Hierarchy

```
KVCacheCoordinator (ABC)
  --> KVCacheCoordinatorNoPrefixCache
  --> UnitaryKVCacheCoordinator
  --> HybridKVCacheCoordinator
```

### KVCacheCoordinator (ABC)

```python
class KVCacheCoordinator(ABC):
    @abstractmethod
    def find_longest_cache_hit(
        self,
        request: Request,
        block_pool: BlockPool,
    ) -> tuple[list[KVCacheBlock], int]:
        """Find the longest prefix cache hit for a request.

        Returns:
            Tuple of (cached_blocks, num_computed_tokens)
        """
        ...

    @abstractmethod
    def update_cache(
        self,
        request: Request,
        new_blocks: list[KVCacheBlock],
    ) -> None:
        """Update the cache with newly computed blocks."""
        ...
```

### KVCacheCoordinatorNoPrefixCache

No prefix caching. Always returns empty cache hits.

```python
class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    def find_longest_cache_hit(self, request, block_pool):
        return [], 0

    def update_cache(self, request, new_blocks):
        pass  # No caching
```

### UnitaryKVCacheManager

Single KV cache group with prefix caching enabled.

```python
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(self, request, block_pool):
        # Iterate through request tokens in block-sized chunks
        # Hash each block and look up in block_pool
        # Return matched blocks and token count
        ...

    def update_cache(self, request, new_blocks):
        # Hash each new block and cache it in block_pool
        ...
```

### HybridKVCacheCoordinator

Multiple KV cache groups (e.g., full attention + sliding window + chunked local) with prefix caching.

Uses an **iterative fixed-point algorithm** to find the longest cache hit across all groups simultaneously:

```python
class HybridKVCacheCoordinator(KVCacheCoordinator):
    def find_longest_cache_hit(self, request, block_pool):
        # Iterative fixed-point:
        # 1. For each group, compute candidate cache hit length
        # 2. Use the minimum across all groups as the effective length
        # 3. Re-check all groups with the constrained length
        # 4. Repeat until stable
        ...
```

The fixed-point algorithm is necessary because different attention types (full, sliding window, chunked local) may have different block sizes and different cache hit lengths. The effective cache hit must be the intersection of all groups' hits.

---

## 13. KV Cache Utilities

**File**: `vllm/v1/core/kv_cache_utils.py`

### Key Functions

#### `hash_block_tokens(block_size, prev_block_hash, cur_block_tokens, hash_algo, extra_keys)`

Computes the hash for a KV cache block using chained hashing.

**Hash computation:**
1. Start with `prev_block_hash` (or 0 for the first block)
2. Add token IDs from `cur_block_tokens`
3. Add any `extra_keys` (multimodal features)
4. Apply the hash function (`builtin` uses Python's `hash()`, `sha256` uses cryptographic hash)

#### `resolve_kv_cache_block_sizes(model_config, parallel_config, cache_config)`

Determines block sizes for each KV cache group. For models with hybrid attention (full + sliding window + chunked local), different block sizes may be needed for different groups.

#### `get_kv_cache_configs(kv_cache_spec, available_memory, block_sizes, ...)`

Computes the KV cache configuration based on available memory:

1. Calculates the memory per block for each KV cache group
2. Divides available memory by the maximum memory per block
3. Returns the number of blocks per group

#### Utility Classes and Functions

- **`AttentionSpec`**: Defines the KV cache specification for attention layers
- **`SlidingWindowSpec`**: Specification for sliding window attention
- **`FullAttentionSpec`**: Specification for full attention
- **`ChunkedLocalAttentionSpec`**: Specification for chunked local attention
- **`MambaSpec`**: Specification for Mamba (SSM) state cache
- **`KVCacheGroupSpec`**: Groups layers with the same KV cache specification
- **`UniformTypeKVCacheSpecs`**: Wrapper for uniform KV cache specs across layers

---

## 14. Encoder Cache Management

**File**: `vllm/v1/core/encoder_cache_manager.py`

### Class: `EncoderCacheManager`

Manages the cache for multimodal encoder outputs (e.g., vision embeddings).

```python
class EncoderCacheManager:
    _cache: dict[str, torch.Tensor]    # mm_hash -> encoder output
    _cache_size: int                    # Current cache size in tokens
    _max_cache_size: int                # Maximum cache size in tokens
```

#### Key Methods

- **`allocate(mm_hash, num_tokens)`**: Allocate cache space for an encoder output.
- **`get(mm_hash)`**: Retrieve a cached encoder output.
- **`free(mm_hash)`**: Free a cached encoder output.
- **`can_allocate(num_tokens)`**: Check if there's space for `num_tokens`.
- **`evict(num_tokens)`**: Evict cached entries to make space.

### Class: `EncoderDecoderCacheManager`

Extended encoder cache for encoder-decoder models.

```python
class EncoderDecoderCacheManager(EncoderCacheManager):
    # Adds support for cross-attention encoder cache
```

### `compute_mm_encoder_budget(model_config, scheduler_config)`

Computes the budget (maximum token count) for the encoder cache.

```python
def compute_mm_encoder_budget(
    model_config: ModelConfig,
    scheduler_config: SchedulerConfig,
) -> int:
```

---

## 15. Prefix Caching

### Overview

Prefix caching enables KV cache reuse across requests that share the same prompt prefix. When enabled:

1. Each KV cache block is hashed based on its content (token IDs + prefix hash)
2. When a new request arrives, the scheduler checks for matching hashes
3. Matching blocks are shared (ref_count incremented) instead of recomputed
4. This significantly reduces memory usage and computation for shared system prompts

### Hash Chain

```
Block 0: hash(0, [token_0, token_1, ..., token_{B-1}]) -> H0
Block 1: hash(H0, [token_B, token_{B+1}, ..., token_{2B-1}]) -> H1
Block 2: hash(H1, [token_2B, ...]) -> H2
```

Each block's hash depends on:
1. The hash of the previous block (chaining)
2. The token IDs in the current block
3. Optional extra keys (multimodal features)

### Cache Lookup Algorithm

1. For a new request, iterate through its tokens in block-sized chunks
2. Compute the hash for each block
3. Look up in `BlockPool._block_hashes`
4. If found, increment `ref_count` and skip computation
5. If not found, allocate a new block and compute the KV

### Cache Eviction

When the block pool is full:
1. Evict the least recently used blocks with `ref_count == 0`
2. Remove their hash entries from `_block_hashes`
3. Add them to the free list

### Configuration

```python
CacheConfig(
    enable_prefix_caching=True,
    prefix_caching_hash_algo="builtin",  # or "sha256"
)
```

---

## 16. Preemption Strategies

When there is insufficient KV cache memory for a new request, the scheduler must preempt existing requests.

### Preemption Trigger

Preemption is triggered when:
1. A new request needs KV cache blocks
2. The block pool has insufficient free blocks
3. Eviction of cached (prefix cache) blocks is not enough

### Preemption Process

1. **Identify victim**: The lowest-priority (or last-scheduled) running request
2. **Free blocks**: Release all KV cache blocks allocated to the victim
3. **Reset state**: Reset the victim's `num_computed_tokens` to 0
4. **Re-queue**: Move the victim back to the waiting queue

### Preemption Modes

- **Recompute**: Free all blocks and recompute from scratch when rescheduled. This is the default mode.
- **Swap**: (Future) Swap KV cache to CPU memory and restore when rescheduled.

### Watermark-based Preemption Prevention

The scheduler uses watermarks to avoid preemption:

- **`scheduler_reserve_full_isl`**: When `True`, reserves enough blocks for the full input sequence length at scheduling time. This prevents preemption but may reduce throughput.

---

## 17. Speculative Decoding Support

### Overview

The scheduler supports speculative decoding (EAGLE, Medusa, N-gram, etc.) by:

1. **Draft token scheduling**: Including draft tokens in the scheduled output
2. **Verification**: Processing draft tokens through the model
3. **Acceptance/rejection**: Deciding which draft tokens to keep

### Speculative Decoding Flow

1. **Draft generation**: A drafter (small model, n-gram, etc.) proposes draft tokens
2. **Scheduling**: The scheduler includes draft tokens in `scheduled_spec_decode_tokens`
3. **Model execution**: The model processes both draft and non-draft tokens
4. **Output processing**: `update_from_output()` determines accepted/rejected tokens
5. **State update**: Accepted tokens become part of the request's output

### EAGLE Support

EAGLE uses the model's own hidden states for drafting. The scheduler:
1. Stores draft tokens and their hidden states
2. Passes them to the model runner for verification
3. Handles the KV cache management for draft tokens

### Configuration

```python
SpeculativeConfig(
    method="eagle",               # or "ngram", "medusa", etc.
    num_speculative_tokens=5,     # Number of draft tokens
    ...
)
```

---

## 18. KV Connector for P/D Disaggregation

### Overview

The KV connector enables prefill-decode disaggregation, where:
- **Prefill workers** compute the KV cache for prompts
- **Decode workers** receive the KV cache and generate tokens

### How It Works

1. **Prefill worker**: After computing KV cache, sends it to decode workers via `kv_connector.save()`
2. **Decode worker**: Receives KV cache via `kv_connector.load()` before starting decode
3. **Scheduler role**: The scheduler tracks which blocks have been loaded/sent and manages the transfer metadata

### Scheduler Integration

- `kv_connector_metadata` in `SchedulerOutput` contains transfer instructions
- The scheduler tracks loaded blocks to avoid re-loading
- Block invalidation handles load failures gracefully

### EC Connector

The encoder cache (EC) connector extends this to multimodal models:
- Producer sends encoder outputs (vision embeddings)
- Consumer receives and caches them
- Uses `ec_connector_metadata` in `SchedulerOutput`

---

## 19. Watermarks and Thresholds

### Memory Watermarks

The scheduler uses several watermarks to manage memory:

| Watermark | Description |
|-----------|-------------|
| `gpu_memory_utilization` | Fraction of GPU memory used for KV cache |
| `scheduler_reserve_full_isl` | Reserve full ISL at scheduling time |

### Scheduling Thresholds

| Threshold | Description |
|-----------|-------------|
| `max_num_batched_tokens` | Max tokens per batch |
| `max_num_seqs` | Max concurrent requests |
| `max_num_partial_prefills` | Max partial prefills |
| `max_long_partial_prefills` | Max long partial prefills |
| `long_prefill_token_threshold` | Token count for long prefill classification |

### Preemption Thresholds

The scheduler preemptively evicts when:
- Free blocks < required blocks for the next step
- Running requests exceed `max_num_seqs`

---

## 20. PauseState Management

### States

```python
class PauseState(enum.IntEnum):
    UNPAUSED = 0     # Normal scheduling
    PAUSED_NEW = 1   # Only running requests scheduled
    PAUSED_ALL = 2   # No scheduling
```

### Usage

- `set_pause_state(PAUSE_NEW)`: Used during model weight updates to prevent new requests from being scheduled
- `set_pause_state(PAUSE_ALL)`: Used during complete system pause
- `set_pause_state(UNPAUSED)`: Resume normal scheduling

---

## 21. Full Parameter Reference

### SchedulerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_batched_tokens` | `int` | - | Max tokens per batch |
| `max_num_seqs` | `int` | `128` | Max concurrent sequences |
| `max_model_len` | `int` | - | Max context length |
| `num_scheduler_steps` | `int` | `1` | Steps per iteration |
| `multi_step_stream_outputs` | `bool` | `False` | Stream multi-step outputs |
| `enable_chunked_prefill` | `bool` | `True` | Enable chunked prefill |
| `max_num_partial_prefills` | `int` | `1` | Max partial prefills |
| `max_long_partial_prefills` | `int` | `1` | Max long partial prefills |
| `long_prefill_token_threshold` | `int` | `4096` | Long prefill threshold |
| `policy` | `str` | `"fcfs"` | Scheduling policy |
| `scheduler_cls` | `str` | - | Scheduler class name |
| `async_scheduling` | `bool` | `False` | Enable async scheduling |
| `stream_interval` | `int` | `1` | Output stream interval |
| `scheduler_reserve_full_isl` | `bool` | `False` | Reserve full ISL |
| `max_num_encoder_input_tokens` | `int` | - | Max encoder tokens |

### CacheConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | `int` | `16` | KV cache block size |
| `gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction |
| `cache_dtype` | `str` | `"auto"` | KV cache dtype |
| `enable_prefix_caching` | `bool` | `False` | Enable prefix caching |
| `prefix_caching_hash_algo` | `str` | `"builtin"` | Hash algorithm |
| `sliding_window` | `int \| None` | `None` | Sliding window size |
| `kv_cache_memory_bytes` | `int \| None` | `None` | Explicit KV memory |
| `calculate_kv_scales` | `bool` | `False` | Calculate FP8 scales |
| `mamba_cache_mode` | `str` | `"default"` | Mamba cache mode |
