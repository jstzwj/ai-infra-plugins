# SGLang Benchmark and Deployment Reference

This document provides a comprehensive reference for benchmarking SGLang performance and deploying SGLang in production environments. It covers benchmark tools, datasets, profiling techniques, multi-node deployment, Docker, Kubernetes, and production best practices.

---

## Table of Contents

1. [Benchmark Tools Overview](#benchmark-tools-overview)
2. [bench_serving - Online Serving Benchmark](#bench_serving---online-serving-benchmark)
3. [bench_one_batch - Kernel-Level Benchmark](#bench_one_batch---kernel-level-benchmark)
4. [bench_one_batch_server - Single Batch HTTP Benchmark](#bench_one_batch_server---single-batch-http-benchmark)
5. [bench_offline_throughput - Maximum Throughput Benchmark](#bench_offline_throughput---maximum-throughput-benchmark)
6. [Benchmark Datasets](#benchmark-datasets)
7. [Benchmark Parameters and Configuration](#benchmark-parameters-and-configuration)
8. [Metrics Explained](#metrics-explained)
9. [Profiling with PyTorch Profiler](#profiling-with-pytorch-profiler)
10. [Profiling with Nsight Systems](#profiling-with-nsight-systems)
11. [Profiling in PD Disaggregation Mode](#profiling-in-pd-disaggregation-mode)
12. [Multi-Node Deployment (SSH/Ray)](#multi-node-deployment-sshray)
13. [Docker Deployment](#docker-deployment)
14. [Kubernetes Deployment with LWS](#kubernetes-deployment-with-lws)
15. [PD Disaggregation Deployment on Kubernetes](#pd-disaggregation-deployment-on-kubernetes)
16. [Production Best Practices](#production-best-practices)
17. [Performance Optimization Tips](#performance-optimization-tips)
18. [Troubleshooting](#troubleshooting)

---

## Benchmark Tools Overview

SGLang provides four benchmark tools that operate at different levels of the stack:

| Tool | HTTP Server | Scheduler | Use Case |
|------|-------------|-----------|----------|
| `bench_serving` | Yes (async HTTP client to a running server) | Yes (indirectly, via server) | Realistic online serving benchmarks with latency metrics (TTFT, TPOT, ITL) |
| `bench_one_batch_server` | Yes (sends HTTP requests to a running server) | Yes (indirectly, via server) | End-to-end single-batch latency including HTTP and scheduler overhead |
| `bench_offline_throughput` | No | Yes (directly uses `Engine` in-process) | Maximum throughput measurement without HTTP overhead |
| `bench_one_batch` | No | No (directly calls `ModelRunner`) | Kernel-level latency profiling of a single static batch |

**Use `bench_serving` by default** unless there are specific needs for kernel-level profiling or offline throughput measurement.

---

## bench_serving - Online Serving Benchmark

`bench_serving` is an async HTTP load-testing client that sends requests at controlled rates with configurable concurrency to a running server. It measures realistic online serving metrics including TTFT, TPOT, ITL, and throughput.

### Supported Backends and Endpoints

- `sglang` / `sglang-native`: `POST /generate`
- `sglang-oai`, `vllm`, `lmdeploy`: `POST /v1/completions`
- `sglang-oai-chat`, `vllm-chat`, `lmdeploy-chat`: `POST /v1/chat/completions`
- `trt` (TensorRT-LLM): `POST /v2/models/ensemble/generate_stream`
- `gserver`: Custom server (not implemented yet)
- `truss`: `POST /v1/models/model:predict`

### Quick Start

```bash
# Terminal 1: Launch server
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Terminal 2: Run benchmark
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Steady-State Measurement

Use `num-prompts >= 5 * max-concurrency` to measure steady-state performance:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --max-concurrency 16 \
  --num-prompts 80 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

### Key Options

| Option | Description |
|--------|-------------|
| `--backend` | Backend type (sglang, vllm, trt, etc.) |
| `--base-url` | Base URL for the server |
| `--host` / `--port` | Server host and port (alternative to base-url) |
| `--model` | Model name (auto-detected from /v1/models if not provided) |
| `--tokenizer` | Tokenizer path (defaults to model) |
| `--num-prompts` | Number of requests to send |
| `--dataset-name` | Dataset type (sharegpt, random, random-ids, image, etc.) |
| `--request-rate` | Requests per second. `inf` sends all immediately (burst) |
| `--max-concurrency` | Caps concurrent in-flight requests |
| `--disable-stream` | Switch to non-streaming mode |
| `--output-file FILE.jsonl` | Append JSONL results to file |
| `--output-details` | Include per-request arrays in output |
| `--extra-request-body` | JSON merged into payload (sampling params, etc.) |
| `--warmup-requests N` | Run warmup requests first (default 1) |
| `--flush-cache` | Call `/flush_cache` before main run |
| `--profile` | Call `/start_profile` and `/stop_profile` |
| `--lora-name` | Pick one LoRA per request |
| `--tokenize-prompt` | Send integer IDs instead of text (sglang only) |
| `--apply-chat-template` | Apply tokenizer chat template |
| `--disable-ignore-eos` | Pass through EOS behavior |
| `--fake-prefill` | Skip real prefill for decode-only benchmarking (PD mode) |

### End-to-End Examples

**Sglang native `/generate` (streaming):**

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --num-prompts 2000 \
  --request-rate 100 \
  --max-concurrency 512 \
  --output-file sglang_random.jsonl --output-details
```

**OpenAI-compatible Chat Completions (streaming):**

```bash
python3 -m sglang.bench_serving \
  --backend vllm-chat \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --apply-chat-template
```

**Image/VLM benchmark:**

```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --dataset-name image \
  --num-prompts 500 \
  --image-count 3 \
  --image-resolution 720p \
  --random-input-len 512 \
  --random-output-len 512
```

**Fake decode stress testing (PD disaggregation):**

```bash
# Server must use --disaggregation-transfer-backend fake
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --port 30001

# Benchmark
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30001 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 1024 --random-output-len 256 \
  --fake-prefill
```

**Mooncake trace evaluation:**

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model model-name \
  --dataset-name mooncake \
  --mooncake-slowdown-factor 1.0 \
  --mooncake-num-rounds 1000 \
  --mooncake-workload conversation \
  --use-trace-timestamps true \
  --random-output-len 256
```

### Authentication

For servers requiring OpenAI-style auth:

```bash
export OPENAI_API_KEY=sk-...yourkey...
```

The script adds `Authorization: Bearer $OPENAI_API_KEY` automatically.

---

## bench_one_batch - Kernel-Level Benchmark

The lowest-level tool. Directly instantiates a `ModelRunner` and calls `extend()` / `decode()` on a fixed static batch, bypassing the scheduler entirely. Prefill and decode phases are run separately.

```bash
python3 -m sglang.bench_one_batch \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch-size 32 \
  --input-len 256 \
  --output-len 32
```

### Dummy Weights and Model Overrides

You can benchmark with dummy weights (only needs `config.json`):

```bash
python -m sglang.bench_one_batch \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch 32 --input-len 256 --output-len 32 \
  --load-format dummy
```

You can override model config for faster testing:

```bash
python -m sglang.bench_one_batch \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch 32 --input-len 256 --output-len 32 \
  --load-format dummy \
  --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
```

---

## bench_one_batch_server - Single Batch HTTP Benchmark

Sends a single batch as one HTTP request to a running server. Due to having only a single batch, the server is never in steady-state and metrics will be biased.

```bash
python3 -m sglang.bench_one_batch_server \
  --base-url http://127.0.0.1:30000 \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch-size 32 --input-len 256 --output-len 32
```

---

## bench_offline_throughput - Maximum Throughput Benchmark

Directly instantiates the `Engine` object in-process (no HTTP server) and submits all requests at once via `engine.generate()`. Measures maximum achievable throughput without network overhead.

```bash
python3 -m sglang.bench_offline_throughput \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-prompts 10
```

With profiling:

```bash
python -m sglang.bench_offline_throughput \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 10 \
  --profile --mem-frac=0.8
```

---

## Benchmark Datasets

### ShareGPT (default)

Loads ShareGPT-style prompt-response pairs. Optionally restrict context length and override output lengths.

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name sharegpt \
  --sharegpt-context-len 2048 \
  --sharegpt-output-len 256 \
  --num-prompts 1000
```

- `--dataset-path PATH`: File path for ShareGPT JSON; downloads and caches if not provided.

### Random

Generates random text lengths, sampled from ShareGPT token space.

```bash
python3 -m sglang.bench_serving \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio 0.5 \
  --num-prompts 3000
```

### Random-IDs

Generates random token IDs (can lead to gibberish output).

### Image

Generates images and wraps them in chat messages for VLM benchmarking.

```bash
python3 -m sglang.bench_serving \
  --dataset-name image \
  --image-count 2 \
  --image-resolution 720p \
  --image-format jpeg \
  --image-content random \
  --num-prompts 200
```

Image dataset flags:
- `--image-count`: Number of images per request
- `--image-resolution`: Presets (4k, 1080p, 720p, 360p) or custom `HEIGHTxWIDTH`
- `--image-format`: jpeg or png
- `--image-content`: random or blank

### Generated Shared Prefix

Synthetic dataset with shared long system prompts and short questions. Tests KV cache sharing.

```bash
python3 -m sglang.bench_serving \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256 \
  --num-prompts 1024
```

### MMMU

Samples from MMMU (Math split) and includes images for multimodal benchmarking.

### Mooncake

Evaluates large-scale KVCache sharing with mooncake trace data.

---

## Benchmark Parameters and Configuration

### Rate and Concurrency Control

| Parameter | Description |
|-----------|-------------|
| `--request-rate` | Requests per second. `inf` sends all immediately (burst). Non-infinite rate uses a Poisson process for arrival times. |
| `--max-concurrency` | Caps concurrent in-flight requests regardless of arrival rate. |

### Model and Tokenizer

- `--model` is required unless the backend exposes `GET /v1/models`
- `--tokenizer` defaults to `--model`
- For ModelScope: set `SGLANG_USE_MODELSCOPE=true`

### Output Configuration

- `--output-file FILE.jsonl`: Append JSONL results to file; auto-named if unspecified
- `--output-details`: Include per-request arrays (generated texts, errors, ttfts, itls, input/output lens)

---

## Metrics Explained

The following metrics are printed after each benchmark run:

| Metric | Description |
|--------|-------------|
| Request throughput (req/s) | Completed requests per second |
| Input token throughput (tok/s) | Input tokens per second (includes text and vision tokens) |
| Output token throughput (tok/s) | Output tokens per second |
| Total token throughput (tok/s) | Total tokens per second (includes text and vision) |
| Concurrency | Aggregate time of all requests divided by wall time |
| End-to-End Latency (ms) | Mean/median/std/p99 per-request total latency |
| Time to First Token (TTFT, ms) | Mean/median/std/p99 for streaming mode |
| Inter-Token Latency (ITL, ms) | Mean/median/std/p95/p99/max between tokens |
| TPOT (ms) | Token processing time after first token: `(latency - ttft) / (tokens - 1)` |
| Accept length | Speculative decoding accept length (sglang-only, if available) |

---

## Profiling with PyTorch Profiler

### Profile with bench_serving

```bash
# Set trace path
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# Start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Send profiling request from client
python -m sglang.bench_serving --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 10 --sharegpt-output-len 100 --profile
```

### Profile with bench_offline_throughput

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

python3 -m sglang.bench_one_batch \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --batch 32 --input-len 1024 --output-len 10 --profile

python -m sglang.bench_offline_throughput \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random --num-prompts 10 --profile --mem-frac=0.8
```

### HTTP API Profiling Endpoints

**Start profiling:**

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "start_step": 5,
    "num_steps": 10,
    "activities": ["CPU", "GPU"]
  }'
```

Parameters:
- `output_dir` (optional): Directory for profile traces. Falls back to `SGLANG_TORCH_PROFILER_DIR` then `/tmp`
- `num_steps` (optional): Number of steps to profile. Continues until manually stopped if not specified
- `start_step` (optional): Step number to start profiling (inclusive). Skips warmup
- `activities` (optional): List of activities, e.g. `["CPU", "GPU"]`. Default `["CPU", "GPU"]`
- `merge_profiles` (optional): Whether to merge distributed traces. Default `false`

**Stop profiling:**

```bash
curl -X POST http://127.0.0.1:30000/stop_profile
```

### Live Profiling with sglang.profiler

```bash
# Terminal 1: Send a generation request
python3 -m sglang.test.send_one

# Terminal 2: Start profiling (before request finishes)
python3 -m sglang.profiler

# Or combine in a single command:
python3 -m sglang.test.send_one --profile
```

### Distributed Trace Merger

For multi-node setups with shared storage (NFS, Lustre):

```bash
curl -X POST <BASE_URL>/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/profiles",
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "merge_profiles": true
  }'
```

Output files:
- Individual rank traces: `{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- Merged trace: `merged-{profile_id}.trace.json.gz`

### Viewing Traces

- https://ui.perfetto.dev/ (any browser)
- `chrome://tracing` (Chrome browser only)

For large trace files, reduce `--num-prompts` and `--sharegpt-output-len`.

### PyTorch Bug Workaround

If encountering `RuntimeError: !stack.empty()`:

```bash
export SGLANG_PROFILE_WITH_STACK=False
```

---

## Profiling with Nsight Systems

Nsight Systems exposes more profiling details including register and shared memory usage, annotated code regions, and low-level CUDA APIs.

### Installation

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

### Profile a Single Batch

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  python3 -m sglang.bench_one_batch \
  --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512
```

### Profile a Server

```bash
# Server
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  -o sglang.out --delay 60 --duration 70 \
  python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache

# Client
python3 -m sglang.bench_serving --backend sglang \
  --num-prompts 1000 --dataset-name random \
  --random-input 1024 --random-output 512
```

### Manual Profiler Stop

```bash
# List sessions
nsys sessions list

# Stop specific session
nsys stop --session=profile-XXXXX
```

### Layer-wise NVTX Profiling

SGLang provides built-in layerwise NVTX annotations:

```bash
# Terminal 1: Start server with layerwise NVTX under nsys
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o layerwise_profile \
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-layerwise-nvtx-marker \
    --disable-cuda-graph

# Terminal 2: Start CUDA profiling
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{"start_step": 3, "num_steps": 10, "activities": ["CUDA_PROFILER"]}'

# Terminal 3: Send workload
python -m sglang.bench_serving --backend sglang --num-prompts 100
```

Note: NVTX markers are not emitted for kernel launches captured by CUDA graphs. Use `--disable-cuda-graph` for full visibility.

---

## Profiling in PD Disaggregation Mode

Prefill and decode workers must be profiled separately.

### Profile Prefill Workers

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 10 --sharegpt-output-len 100 \
  --profile --pd-separated \
  --profile-prefill-url http://127.0.0.1:30000
```

### Profile Decode Workers

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 10 --sharegpt-output-len 100 \
  --profile --pd-separated \
  --profile-decode-url http://127.0.0.1:30001
```

Important notes:
- `--profile-prefill-url` and `--profile-decode-url` are mutually exclusive
- Both support multiple worker URLs for multi-instance setups
- Set `SGLANG_TORCH_PROFILER_DIR` on all worker nodes

---

## Multi-Node Deployment (SSH/Ray)

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

Llama 405B (FP8) can also run on a single node:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8
```

### Multi-Node on SLURM

```bash
#!/bin/bash -l
#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.err
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=224GB
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

---

## Docker Deployment

### Dockerfile

The SGLang Dockerfile (`docker/Dockerfile`) uses a multi-stage build with parallel builder stages:

**Build stages:**
1. **base**: NVIDIA CUDA base with Python 3.12, system dependencies, InfiniBand/RDMA libraries
2. **torch_deps**: Installs PyTorch and sglang dependencies
3. **deepep_builder**: Builds DeepEP wheel for MoE models
4. **flashinfer_cache**: Pre-caches FlashInfer JIT kernels
5. **devtools_builder**: Builds development tools (cmake, clangd, oh-my-zsh)
6. **gateway_builder**: Builds sgl-model-gateway Rust binary
7. **framework**: Combines all artifacts, installs SGLang from source
8. **runtime**: Production image (smaller, includes JIT compilation support)

**Key build arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `CUDA_VERSION` | CUDA version | 13.0.1 |
| `BUILD_TYPE` | Package extras (all, srt, etc.) | all |
| `BRANCH_TYPE` | Source type (local, remote) | remote |
| `SGL_VERSION` | SGLang version tag | (none) |
| `USE_LATEST_SGLANG` | Use latest main branch | 0 |
| `FLASHINFER_VERSION` | FlashInfer version | 0.6.8.post1 |
| `MOONCAKE_VERSION` | Mooncake transfer engine version | 0.3.10.post2 |

**Build example:**

```bash
docker build -t sglang:latest -f docker/Dockerfile .
```

### Docker Compose

The `docker/compose.yaml` provides a simple deployment:

```yaml
services:
  sglang:
    image: lmsysorg/sglang:latest
    container_name: sglang
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    restart: always
    network_mode: host    # required by RDMA
    privileged: true      # required by RDMA
    environment:
      - HF_TOKEN=<secret>
    entrypoint: python3 -m sglang.launch_server
    command: --model-path meta-llama/Llama-3.1-8B-Instruct
      --host 0.0.0.0
      --port 30000
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Running with Docker

```bash
docker run -it --rm --network=host --privileged \
    --ipc=host --shm-size 16G \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=<secret> \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000
```

---

## Kubernetes Deployment with LWS

[LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws) is a Kubernetes API for AI/ML inference workloads, particularly multi-host distributed inference.

### Prerequisites

1. At least two Kubernetes nodes with GPUs (e.g., H20 systems with 8 GPUs each)
2. LWS correctly installed on the K8S cluster (v0.6.0+ recommended)
3. Mellanox NICs with RoCE for RDMA scenarios

### Basic LWS YAML (RDMA RoCE)

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: sglang
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-leader
            image: sglang:latest
            securityContext:
              privileged: true
            env:
              - name: NCCL_IB_GID_INDEX
                value: "3"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --mem-fraction-static
              - "0.93"
              - --tp
              - "16"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --host
              - "0.0.0.0"
              - --port
              - "40000"
            resources:
              limits:
                nvidia.com/gpu: "8"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model
                mountPath: /work/models
              - name: ib
                mountPath: /dev/infiniband
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model
            hostPath:
              path: '<your models dir>'
          - name: ib
            hostPath:
              path: /dev/infiniband
    workerTemplate:
      spec:
        dnsPolicy: ClusterFirstWithHostNet
        hostNetwork: true
        hostIPC: true
        containers:
          - name: sglang-worker
            image: sglang:latest
            securityContext:
              privileged: true
            env:
              - name: NCCL_IB_GID_INDEX
                value: "3"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --tp
              - "16"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20000
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
            resources:
              limits:
                nvidia.com/gpu: "8"
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model
                mountPath: /work/models
              - name: ib
                mountPath: /dev/infiniband
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: ib
            hostPath:
              path: /dev/infiniband
          - name: model
            hostPath:
              path: /data1/models
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-leader
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: sglang
    role: leader
  ports:
    - protocol: TCP
      port: 40000
      targetPort: 40000
```

### Deploy and Verify

```bash
kubectl apply -f lws.yaml
kubectl get pods
kubectl logs -f sglang-0
```

### NCCL Debugging

```bash
# Set in container environment
NCCL_DEBUG=TRACE
```

### Key RDMA Configuration

- Mount `/dev/infiniband` into containers
- Set `NCCL_IB_GID_INDEX=3` for RoCE
- Use `hostNetwork: true` and `hostIPC: true`
- Verify IB status with `ibstatus`, `rdma link show`, `ibv_devices`

---

## PD Disaggregation Deployment on Kubernetes

### Architecture

PD (Prefill-Decode) disaggregation separates prefill and decode onto different worker pools, with a load balancer routing requests.

### Components

1. **Prefill LWS**: Handles prompt processing
2. **Decode LWS**: Handles token generation
3. **Load Balancer**: Routes requests (sglang_router)

### Prefill Deployment

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseekr10528-prefill-main
spec:
  leaderWorkerTemplate:
    leaderTemplate:
      spec:
        containers:
          - name: sglang-leader
            image: lmsysorg/sglang:deepep
            command:
              - python3
              - -m
              - sglang.launch_server
              - --port
              - "30000"
              - --model-path
              - /work/models
              - --disaggregation-ib-device
              - mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
              - --chunked-prefill-size
              - "524288"
              - --enable-dp-attention
              - --enable-dp-lm-head
              - --dp-size
              - "16"
              - --moe-a2a-backend
              - deepep
              - --disaggregation-mode
              - prefill
              - --mem-fraction-static
              - "0.7"
              - --tp
              - "16"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20102
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
            env:
              - name: SGLANG_ENABLE_JIT_DEEPGEMM
                value: "1"
              - name: NCCL_IB_GID_INDEX
                value: "3"
            # ... volume mounts and other config
```

### Decode Deployment

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: deepseekr10528-decode-main
spec:
  leaderWorkerTemplate:
    leaderTemplate:
      spec:
        containers:
          - name: sglang-leader
            image: lmsysorg/sglang:deepep
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - /work/models
              - --moe-a2a-backend
              - deepep
              - --disaggregation-mode
              - decode
              - --mem-fraction-static
              - "0.849"
              - --disaggregation-ib-device
              - "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
              - --cuda-graph-max-bs
              - "64"
              - --max-running-requests
              - "2048"
              - --tp-size
              - "16"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):20102
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
            env:
              - name: SGLANG_ENABLE_JIT_DEEPGEMM
                value: "1"
              - name: NCCL_IB_GID_INDEX
                value: "3"
```

### Load Balancer Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepseekr10528-lb-main
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: sgl-minilb
          image: lmsysorg/sglang:deepep
          command:
            - python
            - -m
            - sglang_router.launch_router
            - --pd-disaggregation
            - --prefill
            - http://deepseekr10528-prefill-main:30000
            - --decode
            - http://deepseekr10528-decode-main:30000
            - --host
            - 0.0.0.0
            - --port
            - "8000"
---
apiVersion: v1
kind: Service
metadata:
  name: deepseekr10528-lb-service
spec:
  type: NodePort
  selector:
    app: deepseekr10528-lb
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
      nodePort: 30800
```

### Deploy and Test

```bash
kubectl apply -f p.yaml
kubectl apply -f d.yaml
kubectl apply -f p-svc.yaml
kubectl apply -f d-svc.yaml
kubectl apply -f lb.yaml

# Test
curl -X POST "http://{nodeIP}:30800/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "r1", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

---

## Production Best Practices

### Server Configuration

1. **Memory fraction**: Use `--mem-fraction-static 0.88-0.93` for most GPUs
2. **Tensor parallelism**: Match `--tp` to GPU count per node
3. **CUDA graphs**: Keep enabled for decode performance (default)
4. **Radix cache**: Enable for prefix sharing workloads (default)
5. **Chunked prefill**: Tune `--chunked-prefill-size` based on workload

### Multi-Node Configuration

1. Use RDMA (InfiniBand or RoCE) for inter-node communication
2. Set `NCCL_IB_GID_INDEX=3` for RoCE environments
3. Use `hostNetwork: true` in Kubernetes to prevent performance degradation
4. Mount `/dev/infiniband` for RDMA access
5. Use shared memory (`emptyDir` with `medium: Memory`)

### Health Checks

Configure readiness probes:

```yaml
readinessProbe:
  tcpSocket:
    port: 30000
  initialDelaySeconds: 15
  periodSeconds: 10
```

Or use the HTTP health endpoint:

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
```

### Resource Limits

```yaml
resources:
  limits:
    nvidia.com/gpu: "8"
```

---

## Performance Optimization Tips

### Benchmark Best Practices

1. **Steady-state measurement**: Use `num-prompts >= 5 * max-concurrency`
2. **Warmup**: Allow at least 1 warmup request (default)
3. **Cache flush**: Use `--flush-cache` for consistent results
4. **Token counting**: Use `--apply-chat-template` with instruct models for accurate token counting

### Server Optimization

1. **Increase memory fraction**: `--mem-fraction-static 0.93` for H100/H200
2. **Tune max running requests**: Higher values increase throughput but may increase latency
3. **Use torch compile**: `--enable-torch-compile --torch-compile-max-bs 8`
4. **Disable radix cache for random workloads**: `--disable-radix-cache`
5. **Quantization**: Use FP8 or W8A8 for memory savings and throughput improvement

### Network Optimization

1. Use InfiniBand/RoCE for multi-node deployments
2. Set `NCCL_IB_QPS_PER_CONNECTION=8` and `NCCL_IB_SPLIT_DATA_ON_QPS=1`
3. Set `NCCL_IB_TC=136` for traffic class optimization
4. Configure CPU affinity: `SGLANG_SET_CPU_AFFINITY=true`

### Profiling Workflow

1. Start with `bench_serving` for realistic performance numbers
2. Use `bench_one_batch` for kernel-level profiling
3. Use PyTorch Profiler for GPU kernel analysis
4. Use Nsight Systems for low-level CUDA analysis
5. Use layerwise NVTX for per-layer bottleneck identification

---

## Troubleshooting

### Benchmark Issues

- **All requests failed**: Verify `--backend`, server URL/port, `--model`, and authentication
- **Low throughput**: Adjust `--request-rate` and `--max-concurrency`; verify server batch size
- **Odd token counts**: Use chat/instruct models with proper chat templates
- **Image/MMMU datasets**: Install extra deps (`pillow`, `datasets`, `pybase64`)

### Multi-Node Issues

- **NCCL timeout**: Set `NCCL_DEBUG=TRACE` to check communication
- **RDMA not working**: Verify `ibstatus`, `rdma link show`, mount `/dev/infiniband`
- **Container image issues**: Avoid Ubuntu 18.04-based images; try different images
- **GLOO_SOCKET_IFNAME**: May need correct configuration in containerized environments

### Kubernetes Issues

- **Pod not ready**: Check `kubectl logs -f sglang-0` for startup errors
- **RDMA devices not accessible**: Verify device mounts and host paths
- **Performance degradation**: Use `hostNetwork: true` and `hostIPC: true`
- **LWS version**: Use v0.6.0+ for native `LWS_WORKER_INDEX` support

### Docker Issues

- **Permission errors**: Use `--privileged` flag
- **RDMA not working**: Use `--network=host` and `--privileged`
- **Out of shared memory**: Use `--ipc=host` or `--shm-size=16g`
