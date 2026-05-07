# Advanced Features Reference

This document provides a comprehensive reference for SGLang's advanced features, including HiCache, HiSparse, RL/post-training support, checkpoint engine, forward hooks, deterministic inference, object storage, R-Fork, diffusion model serving, encoder disaggregation, and session management.

---

## Table of Contents

1. [HiCache (Hierarchical KV Cache)](#hicache-hierarchical-kv-cache)
2. [HiSparse (Hierarchical Sparse Attention)](#hisparse-hierarchical-sparse-attention)
3. [RL and Post-Training Support](#rl-and-post-training-support)
4. [Checkpoint Engine](#checkpoint-engine)
5. [Forward Hooks](#forward-hooks)
6. [Deterministic Inference](#deterministic-inference)
7. [Object Storage Integration](#object-storage-integration)
8. [R-Fork (Request Fork)](#r-fork-request-fork)
9. [Diffusion Model Serving](#diffusion-model-serving)
10. [Encoder Disaggregation](#encoder-disaggregation)
11. [Session Management](#session-management)

---

## HiCache (Hierarchical KV Cache)

### Overview

HiCache extends SGLang's RadixAttention with a three-tier hierarchical KV caching system inspired by CPU cache hierarchies:

- **L1 (GPU Memory)**: Fast, private to each inference instance.
- **L2 (Host/CPU Memory)**: Larger, private to each inference instance.
- **L3 (Distributed Storage)**: Largest, shared across all instances.

### Why HiCache

In LLM inference, the prefill phase converts input tokens into KV cache for subsequent decoding. When multiple requests share prefixes, the KV cache is identical. HiCache exploits this by:

1. Caching KV data in GPU memory (L1) via RadixAttention.
2. Extending to host memory (L2) for larger capacity.
3. Integrating distributed storage (L3) for cluster-wide sharing.

### HiRadixTree Architecture

HiCache uses HiRadixTree, which extends the standard RadixTree:

- Each node records where KV data is stored (L1, L2, L3, or multiple tiers).
- Local metadata is maintained precisely for L1 and L2.
- L3 metadata is queried in real time from the backend to reduce synchronization overhead.

### Workflow

#### 1. Local Match

When a new request arrives:

1. Traverse HiRadixTree from root.
2. Match incoming tokens against cached prefixes.
3. Return the longest matching prefix split between L1 (GPU) and L2 (CPU).
4. This is extremely fast (tree traversal only, no data copy).

#### 2. Prefetch from L3

After local matching, for unmatched portions:

1. Query L3 backend for the next continuous matching KV caches.
2. If L3 hit length exceeds threshold (default: 256 tokens), trigger prefetch.
3. Load KV data from L3 to L2 (host memory).

**Prefetch Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `best_effort` | Stop immediately when GPU can compute | Latency-sensitive |
| `wait_complete` | Wait for all prefetch to finish | Maximum cache hits |
| `timeout` | Stop after specified time | Balanced (recommended for production) |

**Timeout formula:**
```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_tokens / 1024
```

#### 3. Write-back

After prefill, KV data can be written back:

| Policy | Description | Trade-off |
|--------|-------------|-----------|
| `write_through` | Write to all tiers immediately | Strongest cache benefit, highest I/O |
| `write_through_selective` | Write only hot data (above hit threshold) | Reduced I/O, still backs up hot data |
| `write_back` | Write only on eviction from upper tier | Minimum I/O, delayed backup |

**Cross-instance sharing:** When data is written to L3, all SGLang instances can access it.

### Data Transfer Optimization

#### Zero-Copy Transfers

- L2 to L3: Pass memory addresses directly without copying.
- Enabled by `page_first` and `page_first_direct` memory layouts.

#### Memory Layouts

| Layout | Description | I/O Performance |
|--------|-------------|-----------------|
| `layer_first` | Compatible with GPU computation | Standard |
| `page_first` | All KV data for same page contiguous | Optimized for zero-copy I/O |
| `page_first_direct` | Page-layer grouping for direct I/O | Best for direct I/O backend |

#### CPU-to-GPU Transfer

- **Compute-Transfer Overlap**: Load layer N+1 KV while computing layer N.
- **GPU-assisted I/O Kernels**: Custom CUDA kernels for KV transfer (up to 3x faster than cudaMemcpy).

#### MLA Optimization

For MLA models (e.g., DeepSeek-V2/V3):
- All TP ranks hold identical KV data.
- Only one rank initiates write-back to avoid redundant storage.

### L3 Storage Backends

| Backend | Description | Key Feature |
|---------|-------------|-------------|
| **Mooncake** | RDMA-based caching system | Zero-copy, ultra-fast transfers |
| **HF3FS (3FS)** | Kubernetes-native distributed storage | Operator-based deployment |
| **NIXL** | Unified API for multiple storage plugins | GDS, S3, 3FS support |
| **AIBrix KVCache** | Production-ready KV cache framework | Low-overhead cross-engine reuse |
| **LMCache** | Alternative enterprise KV cache layer | Alternative to HiCache |
| **HiCacheFile** | Simple file-based backend | Demonstration purposes |

### Multi-Rank Synchronization

During TP inference:

- `all_reduce(op=min)` ensures all ranks get same L3 hit count.
- After prefetch, `all_reduce(op=min)` guarantees prefix length consensus.
- Prevents inconsistent decisions about whether to use prefetched data.

### Configuration

#### Core Parameters

```bash
--enable-hierarchical-cache           # Enable HiCache
--hicache-ratio 2                     # Host:device ratio
--hicache-size 100                    # Host memory size in GB (overrides ratio)
--page-size 64                        # Page granularity
--hicache-io-backend kernel           # I/O backend (direct or kernel)
--hicache-mem-layout page_first_direct # Memory layout
--hicache-write-policy write_through   # Write-back policy
--hicache-storage-backend hf3fs       # L3 storage backend
--hicache-storage-prefetch-policy timeout  # Prefetch strategy
```

#### Advanced Parameters

```bash
--hicache-storage-backend-extra-config '{"prefetch_threshold": 512, "prefetch_timeout_base": 0.5, "prefetch_timeout_per_ki_token": 0.25, "tp_lcm_size": 8}'
```

#### Runtime Attach/Detach

HiCache storage backend can be attached/detached at runtime without restart:

```bash
# Attach storage backend
curl -X POST http://localhost:30000/attach_hicache_storage \
    -H "Content-Type: application/json" \
    -d '{"backend": "hf3fs", "config": {...}}'

# Detach storage backend
curl -X POST http://localhost:30000/detach_hicache_storage
```

### Deployment Examples

#### With HF3FS

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/DeepSeek-R1/ \
    --tp 8 --host 0.0.0.0 --port 10000 \
    --enable-metrics --enable-cache-report \
    --mem-fraction-static 0.85 --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-size 0 \
    --hicache-mem-layout page_first_direct \
    --hicache-io-backend direct \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs \
    --hicache-storage-prefetch-policy wait_complete
```

#### With Mooncake

```bash
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH --tp 8 --page-size 64 \
    --enable-hierarchical-cache --hicache-ratio 2 \
    --hicache-mem-layout page_first_direct \
    --hicache-io-backend direct \
    --hicache-storage-backend mooncake \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy timeout
```

#### With PD Disaggregation

```bash
# Prefill node
python3 -m sglang.launch_server \
    --model-path /path/to/model --tp 8 \
    --enable-hierarchical-cache --hicache-ratio 2 \
    --hicache-storage-backend hf3fs \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake

# Decode node (with async offload for multi-turn KV reuse)
python3 -m sglang.launch_server \
    --model-path /path/to/model --tp 8 \
    --hicache-ratio 2 --hicache-storage-backend hf3fs \
    --disaggregation-decode-enable-offload-kvcache \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake
```

### Custom Storage Backend

To integrate a new backend:

1. Implement three core methods: `get(key)`, `exists(key)`, `set(key, value)`.
2. Register in `BackendFactory` or use dynamic loading:

```bash
python3 -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend dynamic \
    --hicache-storage-backend-extra-config '{
        "backend_name": "my_backend",
        "module_path": "my_package.my_module",
        "class_name": "MyHiCacheStorage"
    }'
```

### Heterogeneous TP Support

For cross-cluster KV reuse with different TP sizes:

```bash
--hicache-storage-backend-extra-config '{"tp_lcm_size": 8}'
```

Set `tp_lcm_size` to the LCM of all TP sizes sharing the same HiCache storage.

---

## HiSparse (Hierarchical Sparse Attention)

### Overview

HiSparse reduces per-request GPU memory during decode by keeping only a small "hot" KV buffer on GPU while storing the complete KV data in CPU pinned memory. Designed for models with DeepSeek Sparse Attention (DSA) architectures.

### Prerequisites

- Models with DSA architecture (DeepSeek-V3.2, GLM-5).
- PD disaggregation mode.
- Enabled on decode instance only.

### How It Works

1. **Forward decode**: Generate next token.
2. **Top-k selection**: Select most relevant token positions via attention scores.
3. **Swap-in**: CUDA kernel loads top-k KV from host to device buffer.
4. **Decode attention**: Compute attention using top-k device locations.
5. **Eager backup**: Async copy previous token's KV from device to host.

### PD Disaggregation Integration (Direct-to-Host)

```
Prefill GPU --RDMA--> Decode Host Pool (CPU pinned)
                            |
                            v
                    alloc device buffer (4KB)
                            |
                            v
                    swap-in kernel (on-demand top-k)
```

KV cache transfers directly to host pool via RDMA, bypassing GPU entirely on the decode side.

### Configuration

```bash
# Decode instance with HiSparse
python3 -m sglang.launch_server \
    --model-path /path/to/model \
    --tp-size 8 --dp-size 8 --enable-dp-attention \
    --kv-cache-dtype bfloat16 \
    --nsa-decode-backend flashmla_sparse \
    --disaggregation-mode decode \
    --enable-hisparse \
    --hisparse-config='{"top_k": 2048, "device_buffer_size": 6144, "host_to_device_ratio": 10}'
```

### HiSparse Config Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `top_k` | int | Number of top-k entries for sparse attention |
| `device_buffer_size` | int | Per-request GPU device buffer size (in tokens) |
| `host_to_device_ratio` | int | Ratio of host to device pool size |

### host_to_device_ratio Guidelines

| Host Memory | Recommended Value |
|-------------|-------------------|
| ~1 TB | 5 |
| ~2 TB | 10 |

### Required Flags on Decode Instance

- `--kv-cache-dtype bfloat16`: Currently only bfloat16 supported.
- `--nsa-decode-backend flashmla_sparse`: Only this backend supported.
- `--enable-hisparse`: Enable HiSparse.
- `--hisparse-config`: Configuration JSON.

---

## RL and Post-Training Support

### Overview

SGLang provides first-class support for reinforcement learning and post-training systems, following the principle "Be a library, not a framework."

### Five Key Capabilities

1. **Fine-Grained Engine Sleep and Wake Up**: Maximize GPU utilization for rollout and training.
2. **Open-To-Use Refit Functionality**: Multiple weight update methods for co-located or disaggregated setups.
3. **Easy To Postpone Generation**: Partial rollout and dedicated rollout control.
4. **Deterministic Inference**: Zero training-inference mismatch.
5. **Load Balancing Router**: Cache-aware load balancing for high-throughput rollout.

### Engine Sleep and Wake Up

#### Release Memory

```bash
curl -X POST http://localhost:30000/release_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{"tags": ["kv_cache", "weights"]}'
```

- Releases KV cache and/or model weights while keeping the server process alive.
- Uses CUDA-graph-aware weight offload via `torch_memory_saver`.
- Must be called when no requests are in progress.

#### Resume Memory

```bash
curl -X POST http://localhost:30000/resume_memory_occupation \
    -H "Content-Type: application/json" \
    -d '{"tags": ["kv_cache", "weights"]}'
```

- Resumes previously released memory regions.
- No need for full restart or CUDA graph recapture.

**Server flag:** `--enable-memory-saver`

### Weight Update (Refit) Methods

#### Method 1: From Disk

**Best for:** Elastic rollout scaling, checkpoint-based workflows.

```bash
curl -X POST http://localhost:30000/update_weights_from_disk \
    -H "Content-Type: application/json" \
    -d '{
        "model_path": "/path/to/new/checkpoint",
        "load_format": null,
        "abort_all_requests": false,
        "weight_version": "step_100",
        "is_async": false,
        "flush_cache": true,
        "recapture_cuda_graph": false,
        "token_step": 100
    }'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | Required | Path to new checkpoint |
| `load_format` | str | None | Weight loading format |
| `abort_all_requests` | bool | false | Abort running requests before update |
| `weight_version` | str | None | Version label for tracking |
| `is_async` | bool | false | Async weight loading |
| `flush_cache` | bool | true | Flush KV cache after update |
| `recapture_cuda_graph` | bool | false | Recapture CUDA graphs |
| `token_step` | int | 0 | Trainer step ID |
| `keep_pause` | bool | false | Keep scheduler paused |

#### Method 2: From Tensor

**Best for:** Co-located training/rollout with in-memory tensor sharing.

```bash
curl -X POST http://localhost:30000/update_weights_from_tensor \
    -H "Content-Type: application/json" \
    -d '{
        "serialized_named_tensors": ["..."],
        "load_format": "direct",
        "flush_cache": true,
        "abort_all_requests": false,
        "weight_version": "step_100"
    }'
```

**Constraints:**
- Requires training and rollout to share GPU memory.
- Tensors must be serialized with `MultiprocessingSerializer.serialize(...)`.

**Python API:** `engine.update_weights_from_tensor(named_tensors, load_format=None)`

#### Method 3: From Distributed Group

**Best for:** Disaggregated training/rollout with NCCL/IB communication.

```bash
# Initialize communication group
curl -X POST http://localhost:30000/init_weights_update_group \
    -H "Content-Type: application/json" \
    -d '{
        "master_address": "10.0.0.1",
        "master_port": 29500,
        "rank_offset": 0,
        "world_size": 16,
        "group_name": "weight_update_group",
        "backend": "nccl"
    }'

# Update weights
curl -X POST http://localhost:30000/update_weights_from_distributed \
    -H "Content-Type: application/json" \
    -d '{
        "names": ["model.layers.0.weight"],
        "dtypes": ["float16"],
        "shapes": [[4096, 4096]],
        "group_name": "weight_update_group",
        "flush_cache": true
    }'

# Destroy group
curl -X POST http://localhost:30000/destroy_weights_update_group
```

**Python APIs:**
- `engine.init_weights_update_group(...)`
- `engine.update_weights_from_distributed(names, dtypes, shapes, ...)`
- `engine.destroy_weights_update_group(group_name)`

### Pause and Continue Generation

#### Pause Generation

```bash
curl -X POST http://localhost:30000/pause_generation \
    -H "Content-Type: application/json" \
    -d '{"mode": "retract"}'
```

**Modes:**

| Mode | Description |
|------|-------------|
| `abort` | Default. Abort all pending/running requests, return to caller. |
| `retract` | Pause engine, move running requests back to waiting queue. KV cache can be flushed. |
| `in_place` | Pause engine without changing request state. Running requests retain KV cache. |

#### Continue Generation

```bash
curl -X POST http://localhost:30000/continue_generation
```

**Correct flow:** `pause_generation` -> `update_weights` -> `continue_generation`

### Load Balancing Router (SGLang Model Gateway)

For large-scale RL rollouts, SGLang Model Gateway provides:

- **Async non-blocking efficiency**: Native async server/router architecture.
- **Elasticity and fault tolerance**: Independent servers with automatic failover.
- **Training-inference alignment**: Same engine for training and deployment.
- **Dynamic load balancing**: Request-level dispatching for multi-turn RL.

---

## Checkpoint Engine

### Overview

The checkpoint engine integration provides efficient distributed model weight loading, significantly reducing startup time for large models.

### Architecture

```
Checkpoint Engine Workers (torchrun)  <-->  SGLang Server
        |                                       |
    Load from disk                        --wait-for-initial-weights
        |                                       |
    Broadcast/P2P to SGLang            Receives weights via NCCL
```

### Installation

```bash
pip install 'checkpoint-engine[p2p]'
```

### Update Methods

| Method | Description |
|--------|-------------|
| `broadcast` | Weights broadcast from loading processes to inference processes |
| `p2p` | Direct peer-to-peer weight transfer |
| `all` | Combination of broadcast and P2P |

### Usage

#### Single Node

```bash
# Terminal 1: SGLang server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp 8 \
    --load-format dummy \
    --wait-for-initial-weights

# Terminal 2: Checkpoint engine
python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/Qwen/Qwen3-8B/ \
    --inference-parallel-size 8
```

#### Multi-Node

```bash
# Node 0: SGLang server + checkpoint engine
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --tp 8 \
    --load-format dummy --wait-for-initial-weights

python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/checkpoint \
    --inference-parallel-size 16

# Node 1: Same setup
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --tp 8 \
    --load-format dummy --wait-for-initial-weights \
    --dist-init-addr [IP]:9120 --nnodes 2 --node-rank 1

python -m sglang.srt.checkpoint_engine.update \
    --update-method broadcast \
    --checkpoint-path /path/to/checkpoint \
    --inference-parallel-size 16
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--update-method` | Required | `broadcast`, `p2p`, or `all` |
| `--checkpoint-path` | Required | Path to model checkpoint |
| `--inference-parallel-size` | Required | Number of inference parallel processes |
| `--endpoint` | `http://localhost:19730` | SGLang server endpoint |
| `--checkpoint-name` | `my-checkpoint-iter-0` | Checkpoint name |
| `--weight-version` | None | Weight version identifier |

### Performance Benefits

1. **Multi-node loading**: Each node loads portion of weights, increasing effective disk bandwidth. ~20s acceleration for DeepSeek-R1 on H20-3e with two nodes.
2. **Single process optimization**: Overlapping disk-to-CPU transfer with CUDA graph capture.

---

## Forward Hooks

### Overview

SGLang supports attaching PyTorch forward hooks to model submodules via JSON configuration.

### Use Cases

- Logging intermediate activations
- Debugging model internals
- Exporting hidden states to external tooling

### Configuration

```json
{
    "forward_hooks": [
        {
            "name": "capture_attention",
            "target_modules": ["model.layers.*.self_attn"],
            "hook_factory": "my_project.hooks:attention_hook_factory",
            "config": {
                "tag": "attention_output",
                "log_shape": true
            }
        }
    ]
}
```

### Hook Spec Schema

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Human-readable name for logging |
| `target_modules` | Yes | List of glob patterns to match module names |
| `hook_factory` | Yes | Python callable path (`module:factory` or `module.factory`) |
| `config` | No | Arbitrary JSON passed to the hook factory |

### Target Module Patterns

Uses `fnmatch.fnmatch` for pattern matching:

- `"model.layers.0"` matches exactly.
- `"model.layers.*"` matches all children.
- `"model.layers.*.mlp"` matches MLP modules across all layers.

### Writing a Hook Factory

```python
def attention_hook_factory(config):
    """Factory that creates a forward hook."""
    tag = config.get("tag", "default")

    def hook(module, inputs, output):
        print(f"[{tag}] Module: {type(module).__name__}, Output shape: {output.shape}")
        return output  # Must return output if not modifying

    return hook
```

### Hook Lifecycle

- Hooks are registered in `ModelRunner.initialize()`.
- Attached once at startup, run on every forward pass.
- Forward hooks only (`register_forward_hook`).
- Non-matching patterns produce warnings, not errors.
- Hook factory returning `None` is non-fatal.

---

## Deterministic Inference

### Overview

Deterministic inference ensures consistent LLM outputs across runs, critical for RL training, testing, and production reliability.

### Root Cause of Non-Determinism

The main source is varying batch sizes causing different reduction orders in GPU kernels. Due to floating-point non-associativity: `(a + b) + c != a + (b + c)`.

### SGLang's Solution

Built on batch-invariant operators from Thinking Machines Lab.

### Supported Backends

| Backend | CUDA Graph | Chunked Prefill | Radix Cache | Non-greedy Sampling |
|---------|------------|-----------------|-------------|---------------------|
| FlashInfer | Yes | Yes | No | Yes |
| FA3 | Yes | Yes | Yes | Yes |
| Triton | Yes | Yes | Yes | Yes |

### Usage

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### Non-Greedy Sampling with Deterministic Inference

```python
import requests

# Different seeds produce different but reproducible responses
for seed in [42, 43, 44]:
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": "Tell me a joke",
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 128,
                "sampling_seed": seed
            }
        }
    )
```

### Verification

```bash
# Single test: same prompt, varying batch sizes
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# Prefix test: different prefix lengths
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50

# Radix cache consistency
python3 -m sglang.test.test_deterministic --test-mode radix_cache
```

Expected: `Unique samples: 1` (perfectly deterministic).

---

## Object Storage Integration

### Overview

SGLang supports loading models directly from object storage without full local download.

### Supported Backends

| Backend | URI Format |
|---------|-----------|
| Amazon S3 | `s3://bucket-name/path/to/model/` |
| Google Cloud Storage | `gs://bucket-name/path/to/model/` |
| Azure Blob | `az://container-name/path/to/model/` |
| S3-compatible | `s3://bucket-name/path/to/model/` |

### Usage

```bash
# Auto-detected with object storage URI
python -m sglang.launch_server \
    --model-path s3://my-bucket/models/llama-3-8b/

# With tensor parallelism
python -m sglang.launch_server \
    --model-path gs://my-bucket/models/llama-70b/ \
    --tp 4 \
    --model-loader-extra-config '{"distributed": true}'
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distributed` | bool | Auto | Enable distributed streaming for multi-GPU |
| `concurrency` | int | 4 | Number of concurrent download streams |
| `memory_limit` | int | System-dependent | Streaming buffer memory limit (bytes) |

### Limitations

- Only `.safetensors` format supported.
- Distributed streaming only on CUDA devices.

---

## R-Fork (Request Fork)

### Overview

R-Fork (Tensor Remote Fork) enables zero-copy GPU-to-GPU weight loading from a running SGLang instance to a new instance, reducing boot-up time from minutes to seconds.

### Backends

| Backend | Description |
|---------|-------------|
| `nccl` | NCCL-based direct GPU-to-GPU transfer |
| `transfer_engine` | Mooncake TransferEngine with RDMA |
| `modelexpress` | ModelExpress coordination service with RDMA |

### Usage

#### NCCL Backend

```bash
# Seed instance
python -m sglang.launch_server [args]

# Client instance
python -m sglang.launch_server [args] \
    --load-format remote_instance \
    --remote-instance-weight-loader-seed-instance-ip [seed_ip] \
    --remote-instance-weight-loader-seed-instance-service-port [port] \
    --remote-instance-weight-loader-send-weights-group-ports [port_list] \
    --remote-instance-weight-loader-backend nccl
```

#### TransferEngine Backend

```bash
# Seed instance
python -m sglang.launch_server [args] \
    --remote-instance-weight-loader-start-seed-via-transfer-engine

# Client instance
python -m sglang.launch_server [args] \
    --load-format remote_instance \
    --remote-instance-weight-loader-seed-instance-ip [seed_ip] \
    --remote-instance-weight-loader-seed-instance-service-port [port] \
    --remote-instance-weight-loader-backend transfer_engine
```

#### ModelExpress Backend

```bash
# Seed instance
python -m sglang.launch_server [args] \
    --modelexpress-config '{"url": "[grpc_host:port]", "model_name": "[name]", "source": true}'

# Client instance
python -m sglang.launch_server [args] \
    --load-format remote_instance \
    --remote-instance-weight-loader-backend modelexpress \
    --modelexpress-config '{"url": "[grpc_host:port]", "model_name": "[name]"}'
```

### Configuration Arguments

| Argument | Description |
|----------|-------------|
| `--load-format remote_instance` | Enable R-Fork |
| `--remote-instance-weight-loader-backend` | Backend: `nccl`, `transfer_engine`, `modelexpress` |
| `--remote-instance-weight-loader-seed-instance-ip` | Seed instance IP |
| `--remote-instance-weight-loader-seed-instance-service-port` | Seed instance HTTP port |
| `--remote-instance-weight-loader-send-weights-group-ports` | NCCL group ports (NCCL only) |
| `--remote-instance-weight-loader-start-seed-via-transfer-engine` | Start seed for TransferEngine |
| `--modelexpress-config` | JSON config for ModelExpress |

---

## Diffusion Model Serving

### Overview

SGLang Diffusion is a high-performance inference framework for image and video generation with diffusion models.

### Key Features

- Broad model support: WAN, Hunyuan, Qwen-Image, FLUX, Z-Image, GLM-Image, LTX, and more.
- Fast inference with `sgl-kernel`, JIT kernels, scheduler improvements, and caching.
- Multiple interfaces: `sglang generate`, `sglang serve`, and OpenAI-compatible API.
- Multi-platform: NVIDIA, AMD, Intel XPU, Ascend, Apple Silicon, Moore Threads.

### Installation

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

### Quick Start

```bash
# One-off generation
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains" \
    --save-output

# Server mode
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

### Supported Models

| Model | Type | Features |
|-------|------|----------|
| Qwen-Image | Image generation | High quality, fast |
| WAN | Video generation | State-of-the-art video |
| FLUX | Image generation | High fidelity |
| Hunyuan | Image/Video | Tencent Hunyuan |
| Z-Image | Image generation | Fast inference |
| GLM-Image | Image generation | Zhipu GLM |
| LTX | Video generation | Fast video |

### Caching Acceleration

- **Cache-DiT**: Cache intermediate features to reduce denoising cost.
- **TeaCache**: Temporal-aware caching for diffusion transformers.

### Quantization

Supports loading quantized transformer checkpoints for reduced memory usage and faster inference.

### Post-Processing

- Frame interpolation
- Upscaling

### Weight Update

The diffusion engine supports `POST /update_weights_from_disk` with:

- **All-or-nothing with rollback**: Failed modules roll back to original weights.
- **Offload-aware**: Writes to CPU buffers when layerwise offload is active.
- **DTensor-aware**: Correct sharding for tensor-parallel parameters.

---

## Encoder Disaggregation

### Overview

SGLang supports encoder disaggregation, where the encoder (vision/audio) and decoder (LLM) run on separate GPU instances.

### Use Cases

- **Heterogeneous hardware**: Encoder on smaller GPUs, decoder on larger ones.
- **Resource optimization**: Share a single encoder across multiple decoder instances.
- **Cost reduction**: Use cheaper hardware for encoder computation.

### Implementation

Encoder disaggregation is handled through the disaggregation framework:

1. Prefill instance processes the encoder (vision/audio) inputs.
2. Encoder embeddings are transferred to the decode instance.
3. Decode instance uses the embeddings for language model generation.

### Configuration

The encoder disaggregation uses the same PD disaggregation infrastructure with encoder-specific parameters.

---

## Session Management

### Overview

SGLang provides session management for persistent conversations and stateful interactions.

### Session Types

#### Standard Sessions

```bash
# Open a session
curl -X POST http://localhost:30000/open_session \
    -H "Content-Type: application/json" \
    -d '{"session_id": "my-session"}'

# Close a session
curl -X POST http://localhost:30000/close_session \
    -H "Content-Type: application/json" \
    -d '{"session_id": "my-session"}'
```

#### Streaming Sessions

Streaming sessions maintain state across multiple requests, keeping KV cache alive for conversation continuity.

### Session Controller

The `SessionController` manages session lifecycle:

- **Session creation**: Allocates resources and initializes state.
- **Session tracking**: Maintains active sessions and their associated KV cache.
- **Session cleanup**: Releases resources when sessions expire or are closed.

### Metrics

| Metric | Description |
|--------|-------------|
| `sglang:num_streaming_sessions` | Number of active streaming sessions |
| `sglang:streaming_session_held_tokens` | Tokens held by streaming sessions |

### Cache Protection

Sessions protect their KV cache from eviction:

- Tokens held by active sessions cannot be evicted.
- `cache_protected_len` tracks the protected prefix length.
- When a session closes, its tokens become eligible for eviction.

---

*This reference covers the advanced features of SGLang. For scheduling and memory management, see the Scheduling and Memory Management reference. For layer implementations, see the Layers and Operations reference. For observability, see the Observability and Profiling reference.*
