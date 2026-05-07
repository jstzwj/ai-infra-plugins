# Scheduling and Memory Management Reference

This document provides a comprehensive reference for SGLang's scheduling system, memory management, KV cache allocation, and related subsystems. It covers the scheduler architecture, scheduling policies, continuous batching, chunked prefill, RadixAttention tree cache, and performance tuning parameters.

---

## Table of Contents

1. [Scheduler Architecture Overview](#scheduler-architecture-overview)
2. [Scheduling Policies](#scheduling-policies)
3. [Continuous Batching](#continuous-batching)
4. [Chunked Prefill](#chunked-prefill)
5. [Dynamic Chunking](#dynamic-chunking)
6. [Memory Management](#memory-management)
7. [RadixAttention Tree Cache](#radixattention-tree-cache)
8. [Eviction Policies](#eviction-policies)
9. [Prefill Delayer](#prefill-delayer)
10. [Schedule Conservativeness](#schedule-conservativeness)
11. [Overlap Scheduling](#overlap-scheduling)
12. [Two-Batch Overlap](#two-batch-overlap)
13. [Memory Fraction Static](#memory-fraction-static)
14. [Sliding Window Attention (SWA) Eviction](#sliding-window-attention-swa-eviction)
15. [Multi-Item Scoring (MIS)](#multi-item-scoring-mis)
16. [Request Lifecycle](#request-lifecycle)
17. [Server Arguments Reference](#server-arguments-reference)

---

## Scheduler Architecture Overview

The SGLang scheduler is the central component that manages incoming requests, allocates GPU resources, and orchestrates the execution of inference batches. It operates within the `Scheduler` class located in `python/sglang/srt/managers/scheduler.py`.

### Scheduler Class Hierarchy

The `Scheduler` class inherits from multiple mixin classes, each providing a specific capability:

```
Scheduler
├── SchedulerOutputProcessorMixin    -- Processes forward pass results
├── SchedulerUpdateWeightsMixin      -- Handles weight updates for RL
├── SchedulerProfilerMixin           -- Profiling integration
├── SchedulerMetricsMixin            -- Prometheus metrics collection
├── SchedulerDisaggregationDecodeMixin  -- PD disaggregation (decode)
├── SchedulerDisaggregationPrefillMixin -- PD disaggregation (prefill)
├── SchedulerMultiplexMixin          -- Model multiplexing
├── SchedulerRuntimeCheckerMixin     -- Runtime health checks
├── SchedulerPPMixin                 -- Pipeline parallelism
├── SchedulerDPAttnMixin             -- Data parallel attention
├── SchedulerDllmMixin               -- Diffusion LLM support
└── SchedulerMlxOverlapMixin         -- MLX overlap scheduling
```

### Core Scheduler Components

The scheduler maintains several key data structures:

- **`waiting_queue: List[Req]`** -- Requests waiting to be scheduled.
- **`running_batch: Optional[ScheduleBatch]`** -- Currently running batch of decode requests.
- **`tree_cache: BasePrefixCache`** -- The RadixAttention tree cache for KV cache reuse.
- **`token_to_kv_pool_allocator`** -- Allocator managing GPU memory for KV cache.
- **`req_to_token_pool: ReqToTokenPool`** -- Maps requests to their token locations in KV cache.

### Event Loop

The scheduler runs in an event loop. There are two modes:

1. **Non-overlap mode** (`event_loop_normal`): Sequential processing where the scheduler waits for GPU computation to complete before scheduling the next batch.
2. **Overlap mode** (`event_loop_overlap`): Overlaps CPU processing (scheduling, detokenization) with GPU computation for higher throughput.

### Key Initialization Parameters

When the scheduler is initialized, it configures:

- **Parallelism settings**: `tp_size`, `dp_size`, `pp_size`, `ep_size`, `moe_dp_size`
- **Scheduling policy**: From `server_args.schedule_policy`
- **Priority scheduling**: From `server_args.enable_priority_scheduling`
- **Overlap scheduling**: From `server_args.disable_overlap_schedule`
- **Memory management**: `page_size`, `enable_hierarchical_cache`, `enable_hisparse`
- **Prefill delayer**: Conditional initialization based on `enable_dp_attention`
- **LoRA support**: `enable_lora`, `enable_lora_overlap_loading`, `max_loras_per_batch`

---

## Scheduling Policies

SGLang supports multiple scheduling policies that control the order in which waiting requests are admitted to the running batch. Policies are implemented in `python/sglang/srt/managers/schedule_policy.py`.

### Cache-Aware Policies

Cache-aware policies consider the state of the RadixAttention tree cache when making scheduling decisions:

#### LPM (Longest Prefix Match)

- **Value**: `"lpm"`
- **Behavior**: Sorts the waiting queue so that requests with the longest matching prefix in the tree cache are scheduled first. This maximizes cache hit rates.
- **Trade-off**: Higher scheduling overhead due to prefix matching computation.
- **Fallback**: When the waiting queue exceeds 128 requests, automatically falls back to FCFS to avoid excessive overhead.
- **In-Batch Prefix Caching**: When enabled, if multiple requests share the same small prefix, only one is prioritized while others are temporarily deprioritized to increase overall cache hit rate.

#### DFS-Weight (Depth-First Search Weighting)

- **Value**: `"dfs-weight"`
- **Behavior**: Uses a depth-first traversal of the radix tree to weight requests. Requests whose last node has more accumulated weight (i.e., share a common path in the tree) are prioritized.
- **Use case**: Workloads with hierarchical prefix sharing patterns.

### Cache-Agnostic Policies

Cache-agnostic policies do not consider the tree cache state:

#### FCFS (First Come First Serve)

- **Value**: `"fcfs"` (default)
- **Behavior**: Requests are scheduled in the order they arrive.
- **With priority scheduling**: When `--enable-priority-scheduling` is enabled, requests are sorted by priority first, then by arrival time (`wait_queue_entry_time`).

#### LOF (Longest Output First)

- **Value**: `"lof"`
- **Behavior**: Requests with larger `max_new_tokens` are scheduled first.
- **With priority scheduling**: Sorted by priority first, then by output length.

#### RANDOM

- **Value**: `"random"`
- **Behavior**: Requests are shuffled randomly in the waiting queue.

#### ROUTING-KEY

- **Value**: `"routing-key"`
- **Behavior**: Prioritizes requests whose routing key appears more frequently in the current running batch. This helps consolidate requests with the same routing key for better batching efficiency.

### Priority Scheduling

When `--enable-priority-scheduling` is enabled:

- Each request carries a `priority` value.
- `--schedule-low-priority-values-first` controls whether lower or higher numeric priority values are scheduled first.
- `--priority-scheduling-preemption-threshold` defines the minimum priority difference required for preemption.

### Preemption

When the scheduler cannot fit a new high-priority request, it can preempt lower-priority running requests:

1. Running requests are sorted by priority (opposite direction of the new request).
2. The scheduler identifies requests whose priority difference exceeds the threshold.
3. Preemptible requests are released (KV cache freed) and returned to the waiting queue.
4. The new request is then scheduled in the freed space.

---

## Continuous Batching

SGLang uses continuous batching (also called iteration-level batching) to maximize GPU utilization:

### How It Works

1. **Iteration-level scheduling**: At each iteration, the scheduler decides which requests to include in the batch.
2. **Dynamic batch composition**: New requests are added and completed requests are removed between iterations.
3. **Prefill and decode separation**: The scheduler handles prefill (processing input tokens) and decode (generating output tokens) in separate or mixed batches.

### Batch Lifecycle

A request goes through these scheduler phases:

```
Arrival -> Waiting Queue -> Prefill Batch -> Running Batch (Decode) -> Completion
```

### Batch Size Controls

- **`--max-running-requests`**: Maximum number of concurrently running requests.
- **`--max-prefill-requests`**: Maximum number of requests in a single prefill batch.
- **`--schedule-conservativeness`**: Controls how aggressively new requests are admitted.

---

## Chunked Prefill

Chunked prefill splits long prefill sequences into smaller chunks, allowing the scheduler to interleave prefill and decode operations.

### Configuration

- **`--chunked-prefill-size`**: Maximum number of tokens in a single prefill chunk (default: varies by model, typically 8192).
- **`--max-prefill-tokens`**: Maximum total prefill tokens per batch.

### How Chunked Prefill Works

1. When a request's input length exceeds `chunked_prefill_size`, it is split into multiple chunks.
2. The first chunk is processed in the current batch.
3. Subsequent chunks are processed in future iterations, interleaved with decode steps.
4. The request transitions to the decode phase only after all chunks are processed.

### PrefillAdder and Budget Management

The `PrefillAdder` class manages the token budget for prefill:

- **`rem_input_tokens`**: Remaining input token budget for this batch.
- **`rem_chunk_tokens`**: Remaining tokens within the current chunk.
- **`rem_total_tokens`**: Total available KV cache slots minus reservations.
- **`page_size`**: Memory allocation granularity.

When adding a request to the prefill batch:

1. Compute the required tokens: `extend_input_len + max_new_tokens + page_overhead`.
2. Check against `rem_total_tokens` (available + evictable KV cache).
3. Check against `rem_input_tokens` (input token budget).
4. If chunked, truncate to `rem_chunk_tokens` and mark as chunked.
5. Lock the matched prefix in the tree cache to prevent eviction.
6. Update all budget counters.

### Chunked Request State

A chunked request tracks:

- `is_chunked`: Counter indicating how many chunks remain.
- `extend_input_len`: Tokens to process in the current chunk.
- `fill_ids`: The token IDs to process, truncated to the current chunk.
- `prefix_indices`: KV cache indices for the matched prefix.

---

## Dynamic Chunking

Dynamic chunking adjusts the chunk size based on the current system state to balance TTFT (Time To First Token) and throughput:

### Mechanism

- The scheduler evaluates available memory and running batch state.
- Chunk sizes are dynamically reduced when memory is constrained.
- The `new_token_ratio` parameter estimates how many output tokens each request will generate, affecting admission decisions.

### CLIP_MAX_NEW_TOKENS

The environment variable `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` (default: 4096) clips the estimated `max_new_tokens` for scheduling purposes. This prevents the server from being overly conservative when requests specify very large `max_new_tokens` but actually stop much earlier due to EOS tokens.

---

## Memory Management

SGLang has a two-level memory pool system for managing KV cache.

### ReqToTokenPool

Located in `python/sglang/srt/mem_cache/memory_pool.py`:

- **Purpose**: Maps each request to its token locations in the KV cache.
- **Structure**: A 2D tensor of shape `(max_requests + 1, max_context_len)` with `int32` dtype.
- **Index 0**: Reserved as a padding row for CUDA graph dummy reads/writes.
- **Allocation**: When a request is admitted, a row is allocated from the free list.
- **Free**: When a request completes, its row is returned to the free list.

```python
class ReqToTokenPool:
    req_to_token: torch.Tensor  # [max_requests+1, max_context_len]
    free_slots: List[int]       # Available row indices
```

### TokenToKVPoolAllocator

Manages the physical KV cache indices:

- **Available tokens**: Free slots that can be allocated for new KV entries.
- **Evictable tokens**: Slots holding cached data that can be reclaimed.
- **Operations**: `alloc_extend()` for prefill, `alloc_decode()` for decode.

### KV Cache Data Storage

The actual KV cache tensors are stored per-layer, with shape depending on the model architecture:

- **MHA (Multi-Head Attention)**: `[max_total_num_tokens, num_kv_heads, head_dim]` per layer, per K and V.
- **MLA (Multi-Latent Attention)**: Compressed representation with different dimensions.
- **FP8 Quantization**: Optional FP8 storage for reduced memory footprint.

### Memory Allocation Formula

```
Total GPU Memory = Model Weights + KV Cache Pool + CUDA Graph Buffers + Activations
```

The `--mem-fraction-static` parameter controls:

```
mem_fraction_static = (Model Weights + KV Cache Pool) / GPU Memory Capacity
```

---

## RadixAttention Tree Cache

The RadixAttention tree cache is the core data structure for KV cache reuse. It is implemented in `python/sglang/srt/mem_cache/radix_cache.py`.

### RadixTree Structure

```
Root Node
├── Child Node [tokens: [1, 2, 3]] -> KV indices [10, 11, 12]
│   ├── Child Node [tokens: [4, 5]] -> KV indices [13, 14]
│   │   └── Leaf Node (locked by request A)
│   └── Child Node [tokens: [6, 7]] -> KV indices [15, 16]
│       └── Leaf Node (locked by request B)
└── Child Node [tokens: [8, 9]] -> KV indices [17, 18]
```

### TreeNode Properties

Each `TreeNode` maintains:

- **`token_ids`**: The token sequence stored in this node.
- **`parent`**: Reference to the parent node.
- **`children`**: Dictionary mapping child keys to child nodes.
- **`lock_ref`**: Number of active requests referencing this node.
- **`last_access_time`**: Timestamp of last access (for LRU eviction).
- **`hit_count`**: Number of cache hits (for LFU eviction).
- **`priority`**: Priority value (for priority-based eviction).
- **`value`**: Tensor of KV cache indices.

### RadixKey

The `RadixKey` class represents a key for the radix tree:

- **`token_ids`**: The raw token sequence.
- **`extra_key`**: Optional extra key for namespace separation (e.g., LoRA adapter ID, cache salt).
- **`is_bigram`**: Whether to use bigram view (for Eagle speculative decoding).

### Prefix Matching

The `match_prefix` operation:

1. Traverses the tree from the root.
2. At each node, compares incoming tokens with the node's stored tokens.
3. When `page_size > 1`, matching is performed at page granularity.
4. If a match terminates within a node, the node is automatically split.
5. Returns: matched KV indices, last matched node, host hit length.

### Insert Operation

The `insert` operation:

1. Finds the longest matching prefix.
2. Splits the matching node if needed (partial match).
3. Creates a new child node for the remaining tokens.
4. Allocates KV cache slots for the new tokens.
5. Stores the KV indices in the new node.

### Eviction

When memory is needed:

1. Nodes with `lock_ref == 0` are eligible for eviction.
2. The eviction policy determines the eviction order.
3. Evicted nodes are removed from the tree.
4. Their KV cache slots are returned to the free pool.

### Lock Reference Management

- **`inc_lock_ref`**: Increments the lock count on a node path when a request starts using it.
- **`dec_lock_ref`**: Decrements the lock count when a request no longer needs the cached data.
- **Protected nodes**: Nodes with `lock_ref > 0` cannot be evicted.

---

## Eviction Policies

Eviction policies determine which cached KV data is removed when memory is needed. They are implemented in `python/sglang/srt/mem_cache/evict_policy.py`.

### Available Policies

| Policy | Class | Sort Key | Description |
|--------|-------|----------|-------------|
| LRU | `LRUStrategy` | `last_access_time` | Least recently used entries evicted first |
| LFU | `LFUStrategy` | `(hit_count, last_access_time)` | Least frequently used entries evicted first; ties broken by LRU |
| FIFO | `FIFOStrategy` | `creation_time` | Oldest entries evicted first |
| FILO | `FILOStrategy` | `-creation_time` | Newest entries evicted first |
| MRU | `MRUStrategy` | `-last_access_time` | Most recently used entries evicted first |
| Priority | `PriorityStrategy` | `(priority, last_access_time)` | Lower priority values evicted first; ties broken by LRU |
| SLRU | `SLRUStrategy` | `(segment, last_access_time)` | Two-segment LRU with probationary and protected segments |

### SLRU Details

The Segmented LRU (SLRU) policy divides entries into two segments:

- **Probationary (Segment 0)**: Entries with `hit_count < protected_threshold` (default: 2).
- **Protected (Segment 1)**: Entries with `hit_count >= protected_threshold`.

Probationary entries are always evicted before protected entries. Within each segment, older entries are evicted first.

### Policy Selection

The eviction policy is configured via the tree cache initialization. The default is LRU.

---

## Prefill Delayer

The Prefill Delayer is a mechanism to improve decode throughput in data-parallel attention deployments by delaying prefill batches until the decode batch reaches sufficient size.

### Purpose

When using DP attention, prefill batches temporarily reduce decode throughput because some DP ranks handle prefill while others handle decode. The Prefill Delayer ensures prefill is only triggered when the system has accumulated enough decode requests to maintain throughput.

### Implementation

Located in `python/sglang/srt/managers/prefill_delayer.py`.

### Configuration

- **`--prefill-deliver-max-delay-passes`** (`SGLANG_PREFILL_DELIVER_MAX_DELAY_PASSES`): Maximum number of forward passes to delay prefill (default varies).
- **`--prefill-deliver-token-usage-low-watermark`**: Token usage threshold below which prefill is force-allowed.

### Negotiation Logic

The Prefill Delayer uses a negotiation protocol across DP ranks:

1. Each rank reports whether it has prefillable requests.
2. If all ranks are prefillable ("all"), the system checks if the running batch is large enough.
3. If some ranks are prefillable and others are not ("mixed"):
   - If token usage is below the low watermark, prefill is force-allowed.
   - Otherwise, prefill is delayed up to `max_delay_passes`.
4. If no ranks are prefillable ("none"), prefill proceeds trivially.

### Negotiation Outcomes

| State | Outcome | Description |
|-------|---------|-------------|
| All prefillable, batch ready | Allow | Normal prefill |
| All prefillable, batch too small | Delay | Wait for more decode requests |
| Mixed, low watermark | Allow | Force allow due to low memory usage |
| Mixed, high watermark | Delay/Timeout | Delay with configurable timeout |
| None prefillable | Allow | No prefill to schedule |

---

## Schedule Conservativeness

The `--schedule-conservativeness` parameter controls how aggressively the scheduler admits new requests.

### How It Works

- **Range**: Positive float values (typically 0.1 to 2.0).
- **Default**: 1.0
- **Lower values** (e.g., 0.3): The scheduler is less conservative, admitting more requests. Use when `token_usage < 0.9` despite having queued requests.
- **Higher values** (e.g., 1.3): The scheduler is more conservative, admitting fewer requests. Use when "KV cache pool is full" warnings appear frequently.

### new_token_ratio

The `new_token_ratio` is derived from schedule conservativeness:

- It estimates the fraction of `max_new_tokens` that a request will actually generate.
- A ratio of 1.0 means the scheduler assumes every request will generate its full `max_new_tokens`.
- The scheduler adjusts this ratio dynamically based on actual generation patterns.

### Retraction

When the KV cache pool becomes full mid-generation:

1. Running requests with the most remaining output tokens are retracted.
2. Their KV cache is freed.
3. They are returned to the waiting queue for rescheduling.
4. The `new_token_ratio` is adjusted upward to prevent future retractions.

---

## Overlap Scheduling

Overlap scheduling is the default scheduling mode that overlaps CPU processing with GPU computation for improved throughput.

### Architecture

```
Timeline:
GPU:  [Forward Batch N] [Forward Batch N+1]
CPU:                   [Process Batch N] [Schedule Batch N+1]
                       ^^^^^^^^^^^^^^^^^ overlapped with GPU
```

### How It Works

1. **Batch N** is submitted to the GPU for forward computation.
2. While the GPU is busy, the CPU processes results from Batch N-1 (detokenization, response sending).
3. The CPU simultaneously schedules Batch N+1.
4. When the GPU finishes Batch N, Batch N+1 is immediately ready.

### Enabling Overlap

- **Default**: Enabled unless `--disable-overlap-schedule` is set.
- **Requirement**: The scheduler must be in overlap-capable mode (CUDA streams, async memory copy).

### Overlap Disable Conditions

Overlap is disabled for specific batches when:

1. Two consecutive prefill batches occur (to improve TTFT of the first batch).
2. All DP ranks do not agree on the overlap decision (to avoid deadlock).
3. Speculative decoding with grammar constraints is active.

---

## Two-Batch Overlap

Two-batch overlap is an advanced scheduling technique where two batches are kept in flight simultaneously.

### Mechanism

In the overlap scheduler's `event_loop_overlap` method:

1. The scheduler maintains a "current" batch and prepares a "next" batch.
2. While the current batch executes on GPU, the next batch is prepared on CPU.
3. The GPU alternates between the two batches.
4. This creates a pipeline where scheduling and computation are fully overlapped.

### Key Implementation Details

- **FutureMap**: A mechanism for asynchronously accessing results from the GPU worker.
- **Batch result processing**: Results are copied to CPU asynchronously, enabling true overlap.
- **CUDA stream management**: Separate streams for computation and data transfer.

---

## Memory Fraction Static

### Overview

`--mem-fraction-static` controls the fraction of GPU memory allocated for static data (model weights + KV cache pool).

### Memory Layout

```
┌─────────────────────────────────────────────┐
│              GPU Memory                      │
├─────────────────────────────────────────────┤
│  Model Weights                              │
├─────────────────────────────────────────────┤
│  KV Cache Pool                              │  } mem_fraction_static
├─────────────────────────────────────────────┤
│  CUDA Graph Buffers                         │
├─────────────────────────────────────────────┤
│  Activations (temporary)                    │
└─────────────────────────────────────────────┘
```

### Tuning Guidelines

1. **Check `available_gpu_mem`** in startup logs (target: 5-8 GB free).
2. **If too much free memory** (>10 GB): Increase `--mem-fraction-static` in increments of 0.01.
3. **If OOM occurs**: Decrease `--mem-fraction-static` to 0.8 or 0.7.
4. **If OOM during prefill**: Reduce `--chunked-prefill-size` to 4096 or 2048.
5. **If OOM during decode**: Lower `--max-running-requests`.

---

## Sliding Window Attention (SWA) Eviction

Some models (e.g., Gemma2) use hybrid attention with a sliding window for recent tokens and full attention for a subset.

### SWATokenToKVPoolAllocator

Located in `python/sglang/srt/mem_cache/swa_memory_pool.py`:

- Manages two pools: a full attention pool and a sliding window pool.
- The sliding window pool holds only the most recent tokens (up to `sliding_window_size`).
- The full pool holds all tokens.

### SWA Budget Calculation

For each request, the SWA budget accounts for:

- **Chunk N** (running, not yet in tree): Tokens from the current chunk.
- **Sliding window** (locked in tree): Tokens within the sliding window.
- **Chunk N+1** (new allocation): Tokens for the next chunk.

The budget formula ensures:
```
swa_budget = max(extend_input_len, sliding_window_size) + page_size
```

### SWA Radix Cache

The `SWARadixCache` extends the standard radix cache with SWA-aware eviction:

- Tokens outside the sliding window are evicted from the SWA pool.
- Full attention tokens remain in the full pool.
- The cache tracks both full and SWA evictable sizes separately.

---

## Multi-Item Scoring (MIS)

Multi-Item Scoring is a feature for models that produce multiple scores per request (e.g., reward models, classification models).

### Overview

- MIS allows a single request to produce multiple outputs, each with its own scoring context.
- The scheduler handles batch composition for MIS requests, ensuring they are processed together.
- Output tensors are returned as a list rather than a single tensor.

### Scheduler Integration

- MIS requests are tracked with their item count.
- The batch scheduler accounts for MIS when computing token budgets.
- Output processing handles the variable-length outputs from MIS requests.

---

## Request Lifecycle

### Data Flow

```
TokenizerManager (receives HTTP request)
  -> Tokenization
  -> ZMQ send to Scheduler

Scheduler (manages batching)
  -> Add to waiting_queue
  -> Scheduling policy sorts queue
  -> PrefillAdder admits requests
  -> Allocate KV cache (ReqToTokenPool + TokenToKVPool)
  -> Lock prefix in tree cache
  -> Submit to TpWorker

TpWorker (GPU execution)
  -> ModelRunner.forward_batch()
  -> ForwardBatch construction
  -> Model forward pass
  -> Return logits

Scheduler (process results)
  -> Sampling
  -> Detokenization
  -> Check completion
  -> Release KV cache (or keep for tree cache)
  -> ZMQ send to TokenizerManager

TokenizerManager (send response)
  -> HTTP streaming or batch response
```

### Batch Data Structures

Three levels of batch data structures exist:

1. **`ScheduleBatch`**: High-level scheduling data managed by the Scheduler. Contains request objects, scheduling metadata, mostly on CPU.

2. **`ModelWorkerBatch`**: Subset of ScheduleBatch for model forward pass on GPU. Transferred from CPU scheduler to GPU model runner.

3. **`ForwardBatch`**: Low-level tensor data for the model forward pass. Contains GPU tensors for attention metadata, positions, etc.

### Request States

A request (`Req`) tracks:

- **`origin_input_ids`**: Original input token IDs.
- **`output_ids`**: Generated output token IDs.
- **`fill_ids`**: Tokens to be processed in the current step.
- **`prefix_indices`**: KV cache indices for matched prefix.
- **`extend_input_len`**: Number of new tokens to process.
- **`req_pool_idx`**: Index in the ReqToTokenPool.
- **`last_node`**: Last matched node in the tree cache.
- **`finished()`**: Whether the request has completed.
- **`is_chunked`**: Counter for remaining chunks.
- **`priority`**: Scheduling priority value.
- **`routing_key`**: Optional routing key for ROUTING_KEY policy.

---

## Server Arguments Reference

### Scheduling Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--schedule-policy` | `fcfs` | Scheduling policy: `fcfs`, `lpm`, `lof`, `random`, `dfs-weight`, `routing-key` |
| `--schedule-conservativeness` | `1.0` | How aggressively to admit new requests |
| `--max-running-requests` | Model-dependent | Maximum concurrent running requests |
| `--max-prefill-requests` | None | Maximum requests per prefill batch |
| `--disable-overlap-schedule` | False | Disable CPU/GPU overlap scheduling |

### Memory Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mem-fraction-static` | Auto | Fraction of GPU memory for weights + KV cache |
| `--chunked-prefill-size` | 8192 | Maximum tokens per prefill chunk |
| `--max-prefill-tokens` | 16384 | Maximum total prefill tokens per batch |
| `--page-size` | 1 | Tokens per page for KV cache management |

### Priority Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-priority-scheduling` | False | Enable priority-based scheduling |
| `--schedule-low-priority-values-first` | False | Schedule lower priority values first |
| `--priority-scheduling-preemption-threshold` | 0 | Priority difference threshold for preemption |
| `--default-priority-value` | None | Default priority for requests without priority |

### Cache Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--disable-radix-cache` | False | Disable RadixAttention tree cache |
| `--enable-hierarchical-cache` | False | Enable HiCache (hierarchical KV cache) |
| `--hicache-ratio` | 2 | Host-to-device KV cache ratio |
| `--hicache-size` | 0 | Host KV cache size in GB |

### SWA Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--swa-full-tokens-ratio` | Varies | Ratio for full attention tokens in SWA models |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | 4096 | Clip max_new_tokens estimation for scheduling |
| `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` | 32 | Threshold for in-batch prefix caching |
| `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` | 32 | Deprioritize threshold for in-batch prefix caching |
| `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` | 256 | Maximum requests received per scheduler poll |
| `SGLANG_PREFILL_DELIVER_MAX_DELAY_PASSES` | Varies | Maximum prefill delay passes |
| `SGLANG_TEST_RETRACT` | 0 | Enable test retract for debugging |

---

*This reference covers the scheduling and memory management internals of SGLang. For performance tuning guidance, see the Hyperparameter Tuning documentation.*
