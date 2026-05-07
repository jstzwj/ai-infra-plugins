# SGLang Distributed Serving Reference

This document provides a comprehensive reference for all distributed serving strategies in SGLang, including tensor parallelism, pipeline parallelism, expert parallelism, data parallelism, PD disaggregation, EPD disaggregation, context parallelism, multi-node deployment, and the SGLang Router.

## Table of Contents

- [Overview](#overview)
- [Tensor Parallelism (TP)](#tensor-parallelism-tp)
- [Pipeline Parallelism (PP)](#pipeline-parallelism-pp)
- [Expert Parallelism (EP)](#expert-parallelism-ep)
- [Data Parallelism (DP)](#data-parallelism-dp)
- [Data Parallelism Attention (DPA)](#data-parallelism-attention-dpa)
- [PD Disaggregation](#pd-disaggregation)
- [EPD Disaggregation](#epd-disaggregation)
- [Context Parallelism](#context-parallelism)
- [Multi-Node Deployment](#multi-node-deployment)
- [SGLang Model Gateway (Router)](#sglang-model-gateway-router)
- [EPLB (Expert Parallel Load Balancing)](#eplb-expert-parallel-load-balancing)
- [Quick Reference Matrix](#quick-reference-matrix)

---

## Overview

SGLang supports a rich set of distributed serving strategies that can be combined to serve models from 1B to 1T+ parameters across single-GPU to multi-node clusters. The key parallelism dimensions are:

| Strategy | Flag | Primary Use Case |
|---|---|---|
| Tensor Parallelism | `--tp-size` | Intra-node model sharding |
| Pipeline Parallelism | `--pp-size` | Cross-node long-context TTFT reduction |
| Expert Parallelism | `--ep-size` | MoE expert distribution across GPUs |
| Data Parallelism | `--dp-size` | Throughput scaling via request distribution |
| Data Parallelism Attention | `--enable-dp-attention` | MLA model KV cache deduplication |
| PD Disaggregation | `--disaggregation-mode` | Separate prefill/decode serving |
| EPD Disaggregation | `--encoder-only` / `--language-only` | Separate encoder/prefill/decode for VLMs |

---

## Tensor Parallelism (TP)

Tensor Parallelism (TP) shards model weights, attention heads, and intermediate computations across multiple GPUs. Each GPU computes a portion of each layer and communicates results via all-reduce operations.

### Key Characteristics

- Most common parallelism strategy for intra-node scaling
- Splits attention heads and FFN dimensions across GPUs
- Requires high-bandwidth interconnect (NVLink preferred)
- Communication overhead increases with TP size, especially across nodes

### Usage

```bash
# 8-way TP on a single 8-GPU node
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tp-size 8 \
  --host 0.0.0.0 --port 30000

# 16-way TP across 2 nodes
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp-size 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 0
```

### TP Size Guidelines

| Model Size | Recommended TP | Nodes |
|---|---|---|
| 7B-13B | 1-2 | 1 node |
| 30B-70B | 4-8 | 1 node |
| 100B-400B | 8-16 | 1-2 nodes |
| 400B+ | 16+ | 2+ nodes |

---

## Pipeline Parallelism (PP)

Pipeline Parallelism (PP) partitions model layers across GPUs/nodes. Different pipeline stages process different chunks of the input simultaneously, reducing TTFT for ultra-long sequences.

### Why Pipeline Parallelism?

- Reduces TTFT for ultra-long sequences by parallelizing chunk processing across nodes
- Only requires cross-node communication at pipeline stage boundaries
- Better computation-communication overlap compared to large TP across nodes
- Works with Dynamic Chunked Prefill to partition input tokens into pipeline-parallel chunks

### Implementation

SGLang implements a Micro-batching Event Loop with non-blocking asynchronous P2P communication:

- **Decoupled Sync/Async Logic**: `async_send` returns handles; actual synchronization deferred
- **Multi-Stream Execution**: `default_stream`, `forward_stream`, and `copy_stream` for overlapping GPU computation with CPU metadata processing and PP communication

### Dynamic Chunking

Fixed chunk sizes can cause pipeline bubbles, especially with large PP sizes. Dynamic chunking predicts optimal chunk sizes based on a quadratic runtime model:

```
Runtime(L + Next Chunk Size) - Runtime(L) = Runtime(Initial Chunk Size)
```

As prefix length L grows, next chunk size progressively decreases to maintain aligned execution time across stages.

### Configuration

```bash
# Fixed chunked prefill
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 \
  --tp 8 --pp-size 4 \
  --chunked-prefill-size 4096 \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr <MASTER_IP>

# Dynamic chunking
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1 \
  --tp 8 --pp-size 4 \
  --chunked-prefill-size 12288 --enable-dynamic-chunking \
  --nnodes 4 --node-rank 0 \
  --dist-init-addr <MASTER_IP>
```

### Tuning Guidance

1. **Step 1**: Find optimal fixed chunked prefill size for your PP size
2. **Step 2**: Set initial chunk size to 2-3x the optimal fixed size for dynamic chunking
3. **Step 3**: Adjust smooth factor (recommended range: 0.6-0.85)
   - 1.0: Strict model following
   - 0.6-0.85: Best balance
   - 0: Disables dynamic adjustment

### Layer Partition Optimization

Put the larger partition in the higher PP rank when layers are not evenly divisible:
```bash
export SGLANG_PP_LAYER_PARTITION=15,15,15,16  # Better than 16,15,15,15
```

### Case Studies

**DeepSeek-V3.1 with 128K ITL on NVIDIA H20 (4 nodes)**:
- Fixed: chunked-prefill-size=4096
- Dynamic: chunked-prefill-size=12288, smooth_factor=0.65

**Qwen3-235B-A22B-FP8 with 128K ITL on NVIDIA H20 (4 nodes)**:
- Fixed: chunked-prefill-size=6144
- Dynamic: chunked-prefill-size=18432, smooth_factor=0.8

---

## Expert Parallelism (EP)

Expert Parallelism distributes MoE expert weights across multiple devices. It is critical for serving large-scale MoE models like DeepSeek V3/R1.

### Configuration Flags

- `--ep-size`: Number of expert parallel partitions
- `--moe-a2a-backend`: Backend for all-to-all communication
- `--moe-runner-backend`: Backend for MoE computation (grouped GEMMs)

### All-to-All Communication Backends

| Backend | Description | Use Case |
|---|---|---|
| `none` (default) | Disables all-to-all; uses All-Reduce/All-Gather | Hybrid EP and TP setups |
| `deepep` | DeepEP communication library for efficient token shuffling | Large-scale EP deployments |
| `mooncake` | Extension of DeepEP with RDMA for elastic inference | Elastic EP serving |
| `nixl` | NIXL-EP built on NVIDIA's NIXL framework | Elastic EP with fault tolerance |
| `mori` | AMD's native all-to-all for ROCm | AMD GPU deployments |
| `flashinfer` | FlashInfer all-to-all implementation | Large-scale EP |
| `ascend_fuseep` | Ascend NPU native fused all-to-all | Ascend NPU |

### DeepEP Modes

DeepEP supports two dispatch modes:
- **normal**: Optimized for prefill workloads (high throughput)
- **low_latency**: Optimized for decode workloads (low latency, CUDA Graph compatible)
- **auto** (recommended): Automatic mode switching during runtime

```bash
--deepep-mode auto  # Recommended
--deepep-mode normal  # Debug/development
--deepep-mode low_latency  # Debug/development
```

### MoE Computation Backends

| Backend | Description | Use Case |
|---|---|---|
| `auto` (default) | Auto-selects optimal backend | General-purpose |
| `triton` | Triton-based grouped GEMMs | Custom kernel development |
| `deep_gemm` | DeepGEMM optimized for FP8 block-wise quantization | Large-scale EP with FP8 |
| `cutlass` | CUTLASS-based GEMMs | NVIDIA architectures |
| `flashinfer_trtllm` | FlashInfer + TensorRT-LLM | Blackwell with TRT-LLM |
| `flashinfer_trtllm_routed` | Routed MoE with TRT-LLM | Blackwell with TRT-LLM |
| `flashinfer_cutlass` | FlashInfer + CUTLASS | Blackwell with FP4/FP8 |
| `flashinfer_mxfp4` | MXFP4 quantization | Low-precision models |
| `flashinfer_cutedsl` | Custom DSL with ModelOpt FP4 | NVFP4 models |

### Example: DeepSeek-V3 with EP

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --tp 8 --ep 8 \
  --trust-remote-code
```

### Computation-Communication Overlap

#### Two-Batch Overlap (TBO)

Splits requests into micro-batches, interleaving attention computation with dispatch/combine operations. Enable with `--enable-two-batch-overlap` for up to 2x throughput improvement.

#### Single-Batch Overlap (SBO)

Overlaps shared expert computation with communication within a single batch via a dispatcher-hook system. Enable with `--enable-single-batch-overlap`.

### EP with Speculative Decoding

Use `--speculative-moe-runner-backend` and `--speculative-moe-a2a-backend` to customize MoE for the draft model:

```bash
--moe-runner-backend flashinfer_trtllm \
--speculative-moe-runner-backend triton
```

### Constraints

- DeepEP, Mooncake, NIXL-EP, ascend_fuseep, and MORI only support `ep_size = tp_size`
- For hybrid EP and TP (`ep_size < tp_size`), only the `none` backend is supported

---

## Data Parallelism (DP)

Data Parallelism replicates the entire model across multiple GPU sets and processes different batches of requests in parallel.

### Key Characteristics

- Each replica has a full copy of the model
- Requests are distributed across replicas
- No inter-replica communication during inference
- Throughput scales nearly linearly with proper routing

### Native DP Mode (Not Recommended for Production)

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4
```

**Limitations**: Built-in load balancing only, no cache-aware routing, no fault tolerance, limited observability.

### SMG-Based DP (Recommended)

Use the SGLang Model Gateway for production DP:

```bash
# Co-launch workers and SMG
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 --port 30000
```

---

## Data Parallelism Attention (DPA)

DPA applies data parallelism specifically to the attention component. It is most beneficial for MLA models (DeepSeek, MiniMax, Kimi-K2) where tensor parallelism leads to duplicated KV cache.

### How DPA Works

- Each DP replica processes different batches independently
- Each replica maintains its own KV cache (no duplication)
- Enables significantly larger batch sizes due to memory savings
- Replicas can be in different forward modes (prefill, decode, idle)

### Enabling DPA

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 --dp-size 8 --enable-dp-attention \
  --trust-remote-code
```

**Requirements**:
- `--dp-size` must be greater than 1
- `tp_size % dp_size == 0` must be satisfied
- `--enable-dp-attention` must be set

### DPA with Expert Parallelism

For MoE models, DPA pairs with EP for best throughput:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 --dp-size 8 --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm
```

---

## PD Disaggregation

PD (Prefill-Decode) Disaggregation separates the prefill and decode phases into independent service instances, enabling tailored optimizations for each.

### Why PD Disaggregation?

1. **Prefill Interruption**: In unified scheduling, incoming prefill batches interrupt ongoing decode batches, causing token generation delays
2. **DP Attention Imbalance**: One DP worker may process prefill while another handles decode, increasing decode latency

### Transfer Engines

| Engine | Installation | Notes |
|---|---|---|
| Mooncake | `uv pip install mooncake-transfer-engine` | Default; supports NVLink, RDMA |
| NIXL | `pip install nixl` | Supports UCX (default) and LIBFABRIC backends |
| Ascend | `pip install memfabric-hybrid` | For Ascend NPU deployments |

### Single-Node Example (Mooncake)

```bash
# Prefill instance
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

# Decode instance
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### Multi-Node DeepSeek Example

```bash
# Prefill node 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode prefill \
  --host ${local_ip} --port 30000 \
  --trust-remote-code \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8

# Decode node 0
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-ib-device ${device_name} \
  --disaggregation-mode decode \
  --host ${local_ip} --port 30001 \
  --trust-remote-code \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

### NIXL Backend

```bash
# Use NIXL instead of Mooncake
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --port 30000

# Select NIXL plugin backend
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC  # or UCX (default)
```

### NVLink Transport (Mooncake)

For NVL72 deployments:
```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

### Heterogeneous TP with GPU Staging Buffer

When prefill and decode use different TP sizes, the GPU staging buffer provides 2-5x throughput improvement:

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096
```

**Note**: Designed for non-MLA models (GQA, MHA). MLA models should NOT enable this.

### Environment Variables

#### Prefill Server

| Variable | Description | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | Worker threads per TP rank for KV transfer | Dynamic: `max(4, min(12, 0.75*cpu_count//8))` |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | Parallel transfer queues | 4 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | Timeout for receiving KV indices | 300s |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL` | Cleanup interval for bootstrap entries | 120s |

#### Decode Server

| Variable | Description | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | Health check interval | 5.0s |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | Failures before marking prefill offline | 2 |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | Timeout for receiving KV cache | 300s |

---

## EPD Disaggregation

EPD (Encoder-Prefill-Decode) Disaggregation further separates vision encoding from language processing for VLM inference. This creates a three-tier architecture:

1. **Encoder**: Vision preprocessing and ViT image encoding (compute-intensive)
2. **Prefill**: Full multimodal input processing to initialize KV cache
3. **Decode**: Autoregressive token generation (memory-intensive)

### Usage

```bash
# Encoder-only instance
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Language-only instance (handles prefill + decode)
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

### Full EPD Disaggregation

```bash
# Encoder 0
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Encoder 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30001

# Prefill
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill \
  --language-only \
  --encoder-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002

# Decode
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode decode \
  --port 30003

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_HOST:30002 \
  --decode http://$DECODE_HOST:30003 \
  --port 8000
```

### Transfer Backends

| Backend | Description |
|---|---|
| `zmq_to_scheduler` | Default; ZeroMQ to scheduler |
| `zmq_to_tokenizer` | ZeroMQ to tokenizer manager |
| `mooncake` | Mooncake RDMA transfer |

### Global Multimodal Embedding Cache

Enable Mooncake-backed global cache for repeated image inputs on encoder servers:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only \
  --enable-mm-global-cache \
  --port 30000
```

### gRPC Encoder

Run the encoder as a gRPC server:

```bash
# gRPC encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --encoder-only --grpc-mode \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30000

# Prefill (HTTP) using gRPC receiver
SGLANG_ENCODER_MM_RECEIVER_MODE=grpc \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --disaggregation-mode prefill --language-only \
  --encoder-urls grpc://127.0.0.1:30000 \
  --encoder-transfer-backend zmq_to_scheduler \
  --port 30002
```

---

## Context Parallelism

Context Parallelism distributes the processing of long contexts across multiple devices. In SGLang, this is primarily handled through:

- **Pipeline Parallelism**: Different pipeline stages process different chunks of the context
- **Dual Chunk FlashAttention**: Specialized backend for ultra-long context models
- **Chunked Prefill**: Splits long prefill sequences into manageable chunks

---

## Multi-Node Deployment

### Llama 3.1 405B on Two Nodes

```bash
# Node 0
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 0

# Node 1
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
  --tp 16 \
  --dist-init-addr 172.16.4.52:20000 \
  --nnodes 2 \
  --node-rank 1
```

**Note**: Llama 405B FP8 can fit on a single 8-GPU node:
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 \
  --tp 8
```

### SLURM Cluster Deployment

```bash
#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

model=MODEL_PATH
tp_size=16

HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"

srun --ntasks=2 --nodes=2 \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --tp "$tp_size" \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank "$SLURM_NODEID" &

while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
done

wait
```

### Key Multi-Node Parameters

| Parameter | Description |
|---|---|
| `--nnodes` | Total number of nodes |
| `--node-rank` | Rank of this node (0-indexed) |
| `--dist-init-addr` | Address of the master node (IP:PORT) |

---

## SGLang Model Gateway (Router)

The SGLang Model Gateway (SMG) is a production-ready routing system built in Rust for extreme performance.

### Installation

```bash
pip install sglang-router
# or
pip install "sglang[all]"
```

### Launch Options

#### Option A: Co-launch Workers and SMG

```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 --port 30000
```

#### Option B: Separate Launch (Multi-Node)

```bash
# Workers on each node
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000

# SMG pointing to workers
python -m sglang_router.launch_router \
  --worker-urls http://node1:8000 http://node2:8000 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

#### Option C: Dynamic Worker Registration

```bash
# Launch SMG first
python -m sglang_router.launch_router \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000

# Register workers dynamically
curl -X POST http://localhost:30000/workers \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker1:8000"}'
```

### Load Balancing Policies

| Policy | Description | Best For |
|---|---|---|
| `cache_aware` (default) | Combines cache locality with load balancing | Most workloads (recommended) |
| `round_robin` | Cycles through workers in order | Simple, predictable distribution |
| `random` | Random worker selection | Baseline, testing |
| `power_of_two` | Samples two workers, picks lighter one | Low latency requirements |

### Cache-Aware Policy Configuration

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 32 \
  --balance-rel-threshold 1.5 \
  --eviction-interval-secs 120 \
  --max-tree-size 67108864
```

How cache-aware routing works:
1. Maintains an approximate radix tree for each worker based on request history
2. Routes requests to workers with highest prefix match (cache hit)
3. Falls back to shortest-queue routing when load is imbalanced
4. Automatically evicts old entries to prevent memory overflow

### SMG Performance

Cache-aware routing provides significant improvements:
- Throughput: 82,665 -> 158,596 token/s (+92%)
- Cache hit rate: 20% -> 75% (+275%)

### SMG vs Native DP

| Feature | Native DP | SMG-Based DP |
|---|---|---|
| Load Balancing | Built-in in-process | Advanced (cache-aware, power-of-two) |
| Cache Awareness | No | Yes |
| Multi-Node | Limited | Full support |
| Health Monitoring | Basic | Circuit breakers, health checks |
| Reliability | Basic | Retries, rate limiting, queuing |
| Observability | Basic metrics | 40+ Prometheus metrics, OpenTelemetry |
| Hot Worker Add/Remove | No | Yes |

### Recommended Production Setup

```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --router-policy cache_aware \
  --router-health-check-interval-secs 30 \
  --router-prometheus-port 10001 \
  --host 0.0.0.0 --port 30000
```

### Verifying Traffic Distribution

```bash
# Check worker status
curl http://localhost:30000/workers

# Check load distribution
curl http://localhost:30000/get_loads

# Key Prometheus metrics
smg_router_requests_total{model="..."}
smg_worker_requests_active{worker="..."}
sglang_cache_hit_rate{source="..."}
```

---

## EPLB (Expert Parallel Load Balancing)

SGLang integrates the EPLB from DeepSeek to address routing imbalances in MoE models.

### How EPLB Works

1. Analyzes expert activation statistics across requests
2. Computes optimal expert arrangement
3. Strategically places or replicates experts to minimize GPU utilization variance
4. Reduces idle cycles and enhances scalability

### Enabling EPLB

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 --ep 8 \
  --enable-eplb \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm
```

### Best Practices

- Increase batch sizes to stabilize activation statistics
- Configure periodic rebalancing (e.g., every 1000 requests) to adapt to evolving workloads
- Monitor expert load distribution via metrics

---

## Quick Reference Matrix

| Strategy | Flag | Use Case | Key Benefit |
|---|---|---|---|
| **TP** | `--tp-size N` | Model sharding across GPUs | Scales model size |
| **PP** | `--pp-size N` | Long-context TTFT reduction | Parallelizes chunk processing |
| **EP** | `--ep-size N` | MoE expert distribution | Handles large MoE models |
| **DP** | `--dp-size N` | Throughput scaling | Linear throughput scaling |
| **DPA** | `--enable-dp-attention` | MLA model optimization | Eliminates KV cache duplication |
| **PD Disagg** | `--disaggregation-mode` | Prefill/decode separation | Tailored phase optimization |
| **EPD Disagg** | `--encoder-only`/`--language-only` | VLM three-tier separation | Independent encoder scaling |

### Recommended Production Setup for DeepSeek

1. Enable **DPA** for attention: `--dp-size 8 --enable-dp-attention`
2. Enable **EP** for MoE: `--ep 8 --moe-a2a-backend deepep`
3. Use **SMG** with **cache_aware** policy
4. Enable **EPLB** for load balancing: `--enable-eplb`
5. Use **FA3** or **TRTLLM MLA** attention backend

```bash
python -m sglang_router.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 --dp-size 8 --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --enable-eplb \
  --attention-backend fa3 \
  --router-policy cache_aware \
  --trust-remote-code
```
