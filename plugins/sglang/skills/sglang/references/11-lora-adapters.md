# SGLang LoRA Adapters Reference

This document provides a comprehensive reference for LoRA (Low-Rank Adaptation) serving in SGLang, including multi-LoRA support, backends, dynamic loading, GPU pinning, eviction policies, and performance tuning.

## Table of Contents

- [Overview](#overview)
- [Server Arguments](#server-arguments)
- [Serving a Single Adapter](#serving-a-single-adapter)
- [Serving Multiple Adapters](#serving-multiple-adapters)
- [Dynamic LoRA Loading and Unloading](#dynamic-lora-loading-and-unloading)
- [OpenAI-Compatible API](#openai-compatible-api)
- [LoRA Backends](#lora-backends)
- [GPU Pinning for Adapters](#gpu-pinning-for-adapters)
- [LoRA Overlap Loading](#lora-overlap-loading)
- [Eviction Policies](#eviction-policies)
- [Target Module Configuration](#target-module-configuration)
- [Tensor Parallelism with LoRA](#tensor-parallelism-with-lora)
- [Performance Tuning](#performance-tuning)
- [Source Code Structure](#source-code-structure)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

SGLang enables the use of [LoRA adapters](https://arxiv.org/abs/2106.09685) with a base model. By incorporating techniques from [S-LoRA](https://arxiv.org/pdf/2311.03285) and [Punica](https://arxiv.org/pdf/2310.18547), SGLang can efficiently support multiple LoRA adapters for different sequences within a single batch of inputs.

Key capabilities:
- **Multi-LoRA serving**: Serve hundreds of LoRA adapters simultaneously
- **Dynamic loading/unloading**: Load and unload adapters at runtime via API
- **GPU pinning**: Pin frequently-used adapters to GPU memory
- **Overlap loading**: Asynchronously load adapter weights behind GPU compute
- **Multiple backends**: Triton and Chunked SGMV (csgmv) backends
- **Tensor parallelism**: Full TP support with S-LoRA sharding strategy
- **Eviction policies**: LRU and FIFO eviction for GPU memory management

---

## Server Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-lora` | flag | `False` | Enable LoRA support. Auto-set to `True` if `--lora-paths` is provided. |
| `--enable-lora-overlap-loading` | flag | `False` | Enable asynchronous LoRA weight loading to overlap H2D transfers with GPU compute. Enable if LoRA loading is a bottleneck. |
| `--lora-paths` | list | `None` | List of LoRA adapters to load. Format: `<PATH>` or `<NAME>=<PATH>` or JSON `{"lora_name":str,"lora_path":str,"pinned":bool}`. |
| `--max-loras-per-batch` | int | `8` | Maximum number of adapters used by each batch. Affects GPU memory reserved. Set smaller when memory is scarce. |
| `--max-loaded-loras` | int | `None` | Maximum number of LoRA adapters loaded in CPU memory at a time. Must be >= `max-loras-per-batch`. |
| `--lora-eviction-policy` | str | `lru` | Eviction policy when GPU memory pool is full. Choices: `lru` (Least Recently Used), `fifo` (First-In-First-Out). |
| `--lora-backend` | str | `triton` | GEMM kernel backend for LoRA modules. Choices: `triton`, `csgmv`. |
| `--max-lora-rank` | int | `None` | Maximum LoRA rank supported. Auto-inferred from `--lora-paths` if not specified. Needed for dynamic loading of larger-rank adapters. |
| `--lora-target-modules` | str | `None` | Union set of target modules where LoRA should be applied (e.g., `q_proj`, `k_proj`, `gate_proj`). Auto-inferred from `--lora-paths`. Set to `all` for all supported modules. |
| `--max-lora-chunk-size` | int | `16` | Maximum chunk size for ChunkedSGMV LoRA backend. Only used with `--lora-backend csgmv`. Larger values may improve performance. |
| `--lora-drain-wait-threshold` | float | `0` | When any LoRA adapter request waits longer than this threshold (seconds), the scheduler drains one running adapter. Set to `0` to disable. |
| `--tp-size` | int | `1` | Number of GPUs for tensor parallelism. LoRA serving is compatible with TP. |

---

## Serving a Single Adapter

### Native API (`/generate`)

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
    --max-loras-per-batch 2 \
    --log-level warning
```

```python
import requests

url = "http://127.0.0.1:30000"
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    "lora_path": ["lora0", None],  # First uses lora0, second uses base model
}
response = requests.post(url + "/generate", json=json_data)
print(response.json()[0]["text"])  # With lora0
print(response.json()[1]["text"])  # Base model
```

### Passing `None` for Base Model

When `lora_path` is set to `None` for a given input, the base model (without any adapter) is used for that sequence.

---

## Serving Multiple Adapters

### Launch with Multiple Adapters

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
                 lora1=Nutanix/Meta-Llama-3.1-8B-Instruct_SFT_lora_4_alpha_16_humaneval_raw_json \
    --max-loras-per-batch 2 \
    --log-level warning
```

### Request with Multiple Adapters

```python
json_data = {
    "text": [
        "List 3 countries and their capitals.",
        "List 3 countries and their capitals.",
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    "lora_path": ["lora0", "lora1"],  # Each input uses a different adapter
}
response = requests.post(url + "/generate", json=json_data)
```

### Adapter Path Formats

The `--lora-paths` argument supports three formats:

1. **Path only**: `<PATH>` -- The adapter name is derived from the path
2. **Named**: `<NAME>=<PATH>` -- Explicit adapter name
3. **JSON**: `{"lora_name":"name","lora_path":"path","pinned":true}` -- Full configuration including pinning

---

## Dynamic LoRA Loading and Unloading

Instead of specifying all adapters at startup, you can load and unload adapters dynamically via API endpoints.

### Recommendations for Dynamic Loading

When using dynamic LoRA loading, explicitly specify both `--max-lora-rank` and `--lora-target-modules` at startup:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --cuda-graph-max-bs 2 \
    --max-loras-per-batch 2 \
    --max-lora-rank 256 \
    --lora-target-modules all \
    --log-level warning
```

If not explicitly set, SGLang infers these values from `--lora-paths`. In that case, dynamically loaded adapters must share the same shape (rank and target modules) as the initial adapters or be strictly "smaller".

### Load Adapter API

```python
response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora0",
        "lora_path": "path/to/lora/adapter",
    },
)

if response.status_code == 200:
    print("LoRA adapter loaded successfully.", response.json())
else:
    print("Failed to load LoRA adapter.", response.json())
```

### Unload Adapter API

```python
response = requests.post(
    url + "/unload_lora_adapter",
    json={
        "lora_name": "lora0",
    },
)
```

### Replace an Adapter

```python
# Unload old adapter
requests.post(url + "/unload_lora_adapter", json={"lora_name": "lora0"})

# Load new adapter in its place
requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora0",
        "lora_path": "path/to/new/adapter",
    },
)
```

### Load Adapter with Pinning

```python
response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora1",
        "lora_path": "path/to/adapter",
        "pinned": True,  # Pin the adapter to GPU
    },
)
```

---

## OpenAI-Compatible API

SGLang supports LoRA adapters via the OpenAI-compatible APIs (`/v1/chat/completions`, `/v1/completions`) using the `model:adapter-name` syntax.

### Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

# Use base model
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)

# Use LoRA adapter
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct:lora0",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
```

The format is `base-model:adapter-name`. The base model name must match the `--model-path` argument used to launch the server, and the adapter name must match the name used in `--lora-paths` or loaded via the dynamic API.

---

## LoRA Backends

SGLang supports two LoRA backends, selected via `--lora-backend`:

### Triton Backend (`triton`)

The basic Triton-based backend. Suitable for general use cases.

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-backend triton \
    --lora-paths lora1=path/to/lora1
```

### Chunked SGMV Backend (`csgmv`)

Default chunked SGMV backend optimized for high concurrency scenarios. Achieves 20% to 80% latency improvements over the basic Triton backend.

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-backend csgmv \
    --max-loras-per-batch 16 \
    --lora-paths lora1=path/to/lora1 lora2=path/to/lora2
```

The `csgmv` backend supports the `--max-lora-chunk-size` parameter for tuning:

```bash
--lora-backend csgmv --max-lora-chunk-size 32
```

### Backend Performance Comparison

| Backend | Best For | Latency vs Triton |
|---------|----------|-------------------|
| `triton` | General use, compatibility | Baseline |
| `csgmv` | High concurrency, many adapters | 20-80% improvement |

### Backend Source Files

| File | Description |
|------|-------------|
| `lora/backend/base_backend.py` | Base LoRA backend interface |
| `lora/backend/triton_backend.py` | Triton LoRA backend |
| `lora/backend/chunked_backend.py` | Chunked SGMV (csgmv) backend |
| `lora/backend/torch_backend.py` | PyTorch native backend |
| `lora/backend/lmhead_mixing.py` | LMHead mixing backend |
| `lora/triton_ops/` | Triton kernel implementations |
| `lora/torch_ops/` | PyTorch native implementations |

---

## GPU Pinning for Adapters

GPU pinning permanently assigns an adapter to one of the available GPU pool slots (configured by `--max-loras-per-batch`). Pinned adapters are not evicted from GPU memory and remain resident until explicitly unloaded.

### Benefits

- Avoids repeated memory transfers and reinitialization overhead
- Improves performance for frequently-used adapters

### Constraints

- Reduces flexibility for dynamically loading other adapters
- Maximum pinned adapters is limited to `max-loras-per-batch - 1`
- If too many adapters are pinned, unpinned requests may be degraded or halted

### Usage at Startup

Specify `pinned: true` in the JSON format:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --cuda-graph-max-bs 8 \
    --max-loras-per-batch 3 \
    --max-lora-rank 256 \
    --lora-target-modules all \
    --lora-paths \
        {"lora_name":"lora0","lora_path":"path/to/lora0","pinned":true} \
        {"lora_name":"lora1","lora_path":"path/to/lora1"} \
        lora2=path/to/lora2 \
    --log-level warning
```

In this example:
- `lora0` is pinned (always resident in GPU memory)
- `lora1` and `lora2` are regular (can be evicted)

### Usage via Dynamic API

```python
# Load adapter as pinned
response = requests.post(
    url + "/load_lora_adapter",
    json={
        "lora_name": "lora1",
        "lora_path": "path/to/adapter",
        "pinned": True,
    },
)
```

---

## LoRA Overlap Loading

By enabling `--enable-lora-overlap-loading`, the SGLang engine overlaps the loading of LoRA weights with prefill and decode compute, hiding data movement behind GPU computation.

### Performance Impact

Under adversarial conditions, enabling overlap loading can result in approximately 35% reduction in median TTFT.

### Usage

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --enable-lora-overlap-loading \
    --lora-paths lora0=path/to/lora0 \
                 lora1=path/to/lora1 \
                 lora2=path/to/lora2 \
    --max-lora-rank 256 \
    --max-loras-per-batch 2 \
    --max-loaded-loras 4
```

### How It Works

With overlap loading enabled:
1. Adapters are loaded asynchronously (H2D transfers happen in background)
2. GPU compute (prefill/decode) continues while adapter weights transfer
3. When an adapter finishes loading, requests using it can be scheduled

### Limitations

**1. Pinned CPU Memory Requirement**

Asynchronous H2D memory copies require LoRA weights to be pinned in CPU memory, which is a finite system resource. SGLang restricts `max_loaded_loras` to at most 2x `max_loras_per_batch` when overlap loading is enabled.

**2. Reduced Multi-Adapter Prefill Batching**

With overlap loading, adapters become available on the GPU at different times (each loaded asynchronously). This reduces the scheduler's ability to form multi-adapter prefill batches. Requests for different adapters are scheduled in separate or smaller prefill batches.

### When Overlap Loading Can Increase Latency

Consider four LoRA adapters where loading takes 2ms and prefill takes 20ms:

- **Without overlap**: Load all 4 adapters synchronously, then run one combined prefill batch: ~28ms total
- **With overlap**: Load adapters one-by-one, scheduling individual prefills: ~82ms total

In this scenario, the loss of multi-adapter prefill batching dominates and leads to higher TTFT. Enable overlap loading only when LoRA weight loading is the bottleneck (high adapter churn, heavy weights, PCIe-bottlenecked workloads).

---

## Eviction Policies

When the GPU memory pool for LoRA adapters is full, SGLang evicts adapters based on the configured policy:

### LRU (Least Recently Used) - Default

Evicts the adapter that was least recently accessed. Better cache efficiency for workloads with temporal locality.

```bash
--lora-eviction-policy lru
```

### FIFO (First-In-First-Out)

Evicts the adapter that was loaded earliest. Simpler but may evict frequently-used adapters.

```bash
--lora-eviction-policy fifo
```

### Adapter Draining

When `--lora-drain-wait-threshold` is set to a positive value, the scheduler will selectively drain one running adapter to make room for a waiting adapter. This mitigates extreme tail latency under high or skewed workloads.

```bash
--lora-drain-wait-threshold 5.0  # Drain after 5 seconds of waiting
```

Set to `0` (default) to disable draining.

---

## Target Module Configuration

### Auto-Detection

If `--lora-target-modules` is not specified, SGLang automatically infers the target modules from the adapters provided in `--lora-paths`.

### Explicit Configuration

```bash
--lora-target-modules q_proj k_proj v_proj o_proj gate_proj
```

### All Modules

```bash
--lora-target-modules all
```

Enabling LoRA on additional modules introduces a minor performance overhead. For performance-sensitive applications, specify only the modules for which you plan to load adapters.

### Common Target Modules

| Module | Description |
|--------|-------------|
| `q_proj` | Query projection in attention |
| `k_proj` | Key projection in attention |
| `v_proj` | Value projection in attention |
| `o_proj` | Output projection in attention |
| `gate_proj` | Gate projection in MLP |
| `up_proj` | Up projection in MLP |
| `down_proj` | Down projection in MLP |

---

## Tensor Parallelism with LoRA

LoRA serving is fully compatible with Tensor Parallelism (TP). The `--tp-size` argument controls the number of GPUs for tensor parallelism.

The tensor sharding strategy follows the S-LoRA paper, where LoRA weights are partitioned across TP ranks to match the base model's sharding.

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tp-size 4 \
    --enable-lora \
    --lora-paths lora0=path/to/lora0 \
    --max-loras-per-batch 4 \
    --log-level warning
```

---

## Performance Tuning

### Memory Management

1. **Set `max-loras-per-batch` appropriately**: Smaller values reduce GPU memory reserved for adapters but limit concurrent adapter usage.

2. **Use `max-loaded-loras`**: Limit the total number of adapters in CPU memory to control host memory usage.

3. **Choose the right eviction policy**: LRU works well for temporal workloads; FIFO for uniform access patterns.

### Throughput Optimization

1. **Use the `csgmv` backend**: Achieves 20-80% latency improvement over `triton` for high-concurrency scenarios.

2. **Pin frequently-used adapters**: Avoid eviction overhead for hot adapters.

3. **Enable overlap loading**: When adapter loading is a bottleneck (heavy adapters, high churn).

4. **Tune `max-lora-chunk-size`**: For the `csgmv` backend, larger chunk sizes may improve performance.

### Balancing Trade-offs

| Scenario | Recommendation |
|----------|---------------|
| Few adapters, always in memory | Pin all adapters, use `triton` backend |
| Many adapters, high churn | Use `csgmv` backend, enable overlap loading, set appropriate `max-loaded-loras` |
| Skewed workload (some adapters hot) | Pin hot adapters, leave cold ones unpinned with LRU eviction |
| Limited GPU memory | Reduce `max-loras-per-batch`, use smaller `max-lora-rank` |

---

## Source Code Structure

The LoRA implementation is located in `python/sglang/srt/lora/`:

| File | Description |
|------|-------------|
| `lora.py` | Core LoRA layer implementation |
| `lora_config.py` | LoRA configuration parsing |
| `lora_manager.py` | Manages LoRA adapter lifecycle (loading, unloading, GPU memory) |
| `lora_registry.py` | Registry of loaded LoRA adapters |
| `lora_overlap_loader.py` | Asynchronous LoRA weight loading for overlap |
| `lora_drainer.py` | Adapter draining logic for wait threshold |
| `lora_moe_runner_marlin.py` | Marlin MoE runner for LoRA |
| `lora_moe_runners.py` | MoE runner implementations for LoRA |
| `layers.py` | LoRA-aware layer implementations |
| `mem_pool.py` | GPU memory pool for LoRA adapter weights |
| `eviction_policy.py` | LRU and FIFO eviction policies |
| `utils.py` | LoRA utility functions |
| `backend/` | LoRA backend implementations |
| `backend/base_backend.py` | Base backend interface |
| `backend/triton_backend.py` | Triton LoRA backend |
| `backend/chunked_backend.py` | Chunked SGMV (csgmv) backend |
| `backend/torch_backend.py` | PyTorch native backend |
| `backend/lmhead_mixing.py` | LMHead mixing backend |
| `backend/lora_registry.py` | Backend registry |
| `triton_ops/` | Triton kernel implementations |
| `torch_ops/` | PyTorch native implementations |

---

## Limitations and Future Work

### Current Limitations

- **Pinned adapters limit**: Maximum pinned adapters is `max-loras-per-batch - 1`
- **Overlap loading memory**: Requires pinned CPU memory; `max_loaded_loras` limited to 2x `max_loras_per_batch`
- **Overlap loading batching**: May reduce multi-adapter prefill batching efficiency
- **Mixed adapter sizes**: Dynamically loaded adapters should share shapes with initial adapters or be smaller (unless `max-lora-rank` and `lora-target-modules` are explicitly set)

### Planned Features

The development roadmap for LoRA features is tracked in [GitHub Issue #2929](https://github.com/sgl-project/sglang/issues/2929). Planned features include:

- Embedding Layer support
- Unified Paging
- CUTLASS backend
- Additional performance optimizations

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [S-LoRA Paper](https://arxiv.org/pdf/2311.03285) - Serving Thousands of Concurrent LoRA Adapters
- [Punica Paper](https://arxiv.org/pdf/2310.18547) - Multi-Tenant LoRA Serving
- [LoRA Overlap Loading PR](https://github.com/sgl-project/sglang/pull/15512) - Detailed benchmarks
- [SGLang LoRA Development Roadmap](https://github.com/sgl-project/sglang/issues/2929)
