# Observability and Profiling Reference

This document provides a comprehensive reference for monitoring, tracing, and profiling SGLang inference servers. It covers Prometheus metrics, OpenTelemetry tracing, request logging, profiling tools, crash dump, MFU metrics, benchmark tools, and production request tracing.

---

## Table of Contents

1. [Prometheus Metrics](#prometheus-metrics)
2. [OpenTelemetry Tracing](#opentelemetry-tracing)
3. [Request Logging](#request-logging)
4. [Request Dump and Replay](#request-dump-and-replay)
5. [Crash Dump](#crash-dump)
6. [Profiling Tools](#profiling-tools)
7. [PyTorch Profiler Integration](#pytorch-profiler-integration)
8. [Nsight Systems Integration](#nsight-systems-integration)
9. [NVTX Layerwise Profiling](#nvtx-layerwise-profiling)
10. [MFU Metrics](#mfu-metrics)
11. [Request Time Statistics](#request-time-statistics)
12. [Health Check Endpoints](#health-check-endpoints)
13. [Benchmark Tools](#benchmark-tools)
14. [Production Request Tracing](#production-request-tracing)
15. [Grafana Dashboard Setup](#grafana-dashboard-setup)

---

## Prometheus Metrics

SGLang exposes Prometheus metrics via the `/metrics` HTTP endpoint when launched with `--enable-metrics`.

### Enabling Metrics

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --enable-metrics
```

### Counter Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:prompt_tokens_total` | Counter | `model_name` | Total number of prefill (prompt) tokens processed |
| `sglang:generation_tokens_total` | Counter | `model_name` | Total number of generation tokens produced |
| `sglang:estimated_flops_per_gpu_total` | Counter | `model_name` | Estimated floating-point operations per GPU (requires `--enable-mfu-metrics`) |
| `sglang:estimated_read_bytes_per_gpu_total` | Counter | `model_name` | Estimated bytes read from memory per GPU (requires `--enable-mfu-metrics`) |
| `sglang:estimated_write_bytes_per_gpu_total` | Counter | `model_name` | Estimated bytes written to memory per GPU (requires `--enable-mfu-metrics`) |

### Gauge Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:token_usage` | Gauge | `model_name` | Overall KV cache memory utilization (0.0-1.0) |
| `sglang:full_token_usage` | Gauge | `model_name` | Full-attention KV cache pool usage ratio |
| `sglang:swa_token_usage` | Gauge | `model_name` | Sliding-window attention KV cache pool usage ratio |
| `sglang:mamba_usage` | Gauge | `model_name` | Mamba SSM state pool usage ratio |
| `sglang:cache_hit_rate` | Gauge | `model_name` | RadixAttention cache hit rate |
| `sglang:num_running_reqs` | Gauge | `model_name` | Number of currently running requests |
| `sglang:num_used_tokens` | Gauge | `model_name` | Number of actively used KV cache tokens |
| `sglang:gen_throughput` | Gauge | `model_name` | Current generation throughput in tokens/second |
| `sglang:num_queue_reqs` | Gauge | `model_name` | Number of requests in the waiting queue |
| `sglang:is_cuda_graph` | Gauge | `model_name` | Whether CUDA graph is active for current batch |
| `sglang:new_token_ratio` | Gauge | `model_name` | Current new token ratio used by scheduler |

### KV Cache Token Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:kv_available_tokens` | Gauge | `model_name` | Free (unallocated) slots in the KV cache pool |
| `sglang:kv_evictable_tokens` | Gauge | `model_name` | Slots holding radix-cached data that can be evicted |
| `sglang:kv_used_tokens` | Gauge | `model_name` | Actively used slots (locked by running requests) |
| `sglang:max_total_num_tokens` | Gauge | `model_name` | Maximum total KV cache token capacity (emitted at startup) |

### Histogram Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:time_to_first_token_seconds` | Histogram | `model_name` | Time to first token (TTFT) in seconds |
| `sglang:e2e_request_latency_seconds` | Histogram | `model_name` | End-to-end request latency in seconds |
| `sglang:time_per_output_token_seconds` | Histogram | `model_name` | Time per output token (TPOT) in seconds |
| `sglang:inter_token_latency_seconds` | Histogram | `model_name` | Inter-token latency (ITL) in seconds |
| `sglang:func_latency_seconds` | Histogram | `name` | Function-level latency for instrumented functions |

### Histogram Bucket Configurations

**TTFT Buckets**: 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, +Inf

**E2E Request Latency Buckets**: 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, +Inf

**TPOT Buckets**: 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, +Inf

### Speculative Decoding Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:spec_accept_length` | Gauge | `model_name` | Average acceptance length for speculative decoding |
| `sglang:spec_accept_rate` | Gauge | `model_name` | Acceptance rate for speculative decoding |

### PD Disaggregation Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:num_prefill_bootstrap_queue_reqs` | Gauge | `model_name` | Requests in prefill bootstrap queue |
| `sglang:num_prefill_inflight_queue_reqs` | Gauge | `model_name` | Requests in prefill inflight queue |
| `sglang:num_decode_prealloc_queue_reqs` | Gauge | `model_name` | Requests in decode pre-allocation queue |
| `sglang:num_decode_transfer_queue_reqs` | Gauge | `model_name` | Requests in decode transfer queue |
| `sglang:kv_transfer_speed_gb_s` | Gauge | `model_name` | KV transfer speed in GB/s |
| `sglang:kv_transfer_latency_ms` | Gauge | `model_name` | KV transfer latency in milliseconds |

### LoRA Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:lora_pool_slots_used` | Gauge | `model_name` | Number of LoRA adapter slots in use |
| `sglang:lora_pool_slots_total` | Gauge | `model_name` | Total number of LoRA adapter slots |
| `sglang:lora_pool_utilization` | Gauge | `model_name` | LoRA pool utilization ratio |

### HiCache Metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `sglang:hicache_host_used_tokens` | Gauge | `model_name` | Host memory KV cache tokens in use |
| `sglang:hicache_host_total_tokens` | Gauge | `model_name` | Total host memory KV cache token capacity |

### Priority Scheduling Metrics

When `--enable-priority-scheduling` is active, the following metrics include per-priority breakdowns:

- `sglang:num_running_reqs` with additional `priority` label
- `sglang:num_queue_reqs` with additional `priority` label

---

## OpenTelemetry Tracing

SGLang supports distributed request tracing using OpenTelemetry, allowing you to visualize the lifecycle of requests across multiple processes and threads.

### Enabling Tracing

```bash
pip install -e "python[tracing]"

python -m sglang.launch_server \
    --enable-trace \
    --otlp-traces-endpoint 0.0.0.0:4317 \
    --port 30000
```

### Setup with Jaeger

1. Start the OpenTelemetry Collector and Jaeger:

```bash
docker compose -f examples/monitoring/tracing_compose.yaml up -d
```

2. Launch the SGLang server with tracing enabled.

3. Access Jaeger UI at `http://localhost:16686`.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS` | Delay between OTLP exports (default: 500) |
| `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE` | Maximum batch size for OTLP exports (default: 64) |
| `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Protocol for trace export (`grpc` or `http/protobuf`) |

### Trace Levels

The trace level can be dynamically adjusted:

```bash
curl http://0.0.0.0:30000/set_trace_level?level=2
```

| Level | Description |
|-------|-------------|
| 0 | Disable tracing |
| 1 | Trace important slices only |
| 2 | Trace all slices except nested ones |
| 3 | Trace all slices (most verbose) |

### Trace Context Architecture

The tracing framework uses a three-level context structure:

```
TraceReqContext (req_id="req-123")
├── TraceThreadContext(thread_label="scheduler", tp_rank=0)
│     └── TraceSliceContext(slice_name="prefill")
│
└── TraceThreadContext(thread_label="scheduler", tp_rank=1)
      └── TraceSliceContext(slice_name="prefill")
```

- **`TraceReqContext`**: Global context per traced request, creates a request span.
- **`TraceThreadContext`**: Per-thread context, creates a thread span nested within the request.
- **`TraceSliceContext`**: Per-operation context, records individual code slices.

### Traced Request Stages

The following stages are traced:

- **TOKENIZE**: Tokenization of input text.
- **SCHEDULER**: Request scheduling decisions.
- **PREFILL**: Prefill forward pass.
- **DECODE**: Decode forward pass.
- **DETOKENIZE**: Detokenization of output tokens.
- **KV_TRANSFER**: KV cache transfer (PD disaggregation).

### Adding Custom Trace Spans

To add custom tracing for additional operations:

```python
# Initialize per-process
process_tracing_init(otlp_traces_endpoint, server_name)

# Initialize per-thread
trace_set_thread_info("thread_label", tp_rank, dp_rank)

# Create request context
trace_ctx = TraceReqContext()

# Mark request start/end
trace_ctx.trace_req_start()
trace_ctx.trace_req_finish()

# Add slice tracing
trace_ctx.trace_slice_start("my_operation")
trace_ctx.trace_slice_end("my_operation")
```

---

## Request Logging

### Basic Logging

```bash
python -m sglang.launch_server \
    --log-requests \
    --log-request-level 1 \
    --port 30000
```

### Log Levels

| Level | Description |
|-------|-------------|
| 0 | No request logging (default) |
| 1 | Basic request info (IDs, token counts) |
| 2 | Detailed request info including sampling parameters |

---

## Request Dump and Replay

### Starting a Request Dump

```bash
python3 -m sglang.srt.managers.configure_logging \
    --url http://localhost:30000 \
    --dump-requests-folder /tmp/sglang_request_dump \
    --dump-requests-threshold 100
```

This dumps all requests into pickle files, creating a new file every 100 requests.

### Replaying a Request Dump

```bash
python3 scripts/playground/replay_request_dump.py \
    --url http://localhost:30000 \
    --dump-folder /tmp/sglang_request_dump
```

---

## Crash Dump

### Enabling Crash Dump

```bash
python -m sglang.launch_server \
    --crash-dump-folder /tmp/crash_dump \
    --port 30000
```

### Behavior

- When the server crashes, all requests from the 5 minutes before the crash are automatically dumped.
- The dump includes request IDs, input tokens, sampling parameters, and generated output.
- Dumps can be replayed using the same `replay_request_dump.py` script for debugging.

---

## Profiling Tools

SGLang provides multiple profiling approaches at different levels of the stack.

### Profiling Methods Overview

| Method | Level | Overhead | Best For |
|--------|-------|----------|----------|
| PyTorch Profiler | Kernel-level | Medium | Identifying slow kernels, call stacks |
| Nsight Systems | System-level | Low-Medium | Detailed CUDA analysis, timeline |
| NVTX Layerwise | Layer-level | Low | Per-layer performance breakdown |
| HTTP API | Server-level | Low | Production profiling |
| `sglang.profiler` | Live profiling | Low | Quick profiling of running server |

---

## PyTorch Profiler Integration

### Profile with bench_serving

```bash
# Set trace output directory
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# Start server
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Send profiling request
python -m sglang.bench_serving \
    --backend sglang \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-prompts 10 \
    --sharegpt-output-len 100 \
    --profile
```

### Profile with bench_one_batch

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

python3 -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --batch 32 \
    --input-len 1024 \
    --output-len 10 \
    --profile
```

### Profile with bench_offline_throughput

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

python -m sglang.bench_offline_throughput \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --num-prompts 10 \
    --profile \
    --mem-frac=0.8
```

### Profile with sglang.profiler

```bash
# Terminal 1: Send a generation request
python3 -m sglang.test.send_one

# Terminal 2: Profile the running server
python3 -m sglang.profiler

# Or combined
python3 -m sglang.test.send_one --profile
```

### PD Disaggregation Profiling

Prefill and decode workers must be profiled separately:

```bash
# Profile prefill workers
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 10 \
    --profile \
    --pd-separated \
    --profile-prefill-url http://127.0.0.1:30000

# Profile decode workers
python -m sglang.bench_serving \
    --backend sglang \
    --num-prompts 10 \
    --profile \
    --pd-separated \
    --profile-decode-url http://127.0.0.1:30001
```

### Trace Merger for Distributed Setups

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
- Individual rank: `{profile_id}-TP-{tp}-DP-{dp}-PP-{pp}-EP-{ep}.trace.json.gz`
- Merged: `merged-{profile_id}.trace.json.gz`

### Viewing Traces

- **Perfetto UI**: https://ui.perfetto.dev/ (any browser)
- **Chrome Tracing**: `chrome://tracing` (Chrome only)

### Known Issues

If you encounter `RuntimeError: !stack.empty() INTERNAL ASSERT FAILED`:
```bash
export SGLANG_PROFILE_WITH_STACK=False
```

---

## Nsight Systems Integration

### Installation

```bash
apt update && apt install -y --no-install-recommends gnupg
apt install nsight-systems-cli
```

### Profile a Single Batch

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    python3 -m sglang.bench_one_batch \
    --model meta-llama/Meta-Llama-3-8B \
    --batch-size 64 \
    --input-len 512
```

### Profile a Server

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
    -o sglang.out --delay 60 --duration 70 \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

To manually stop:
```bash
nsys sessions list
nsys stop --session=profile-XXXXX
```

---

## NVTX Layerwise Profiling

### Overview

SGLang provides built-in layerwise NVTX annotations combined with `--enable-layerwise-nvtx-marker`.

### Method 1: With CUDA Profiler API Control

```bash
# Terminal 1: Server with NVTX under nsys
nsys profile --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -o layerwise_profile \
    python -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --enable-layerwise-nvtx-marker \
        --disable-cuda-graph

# Terminal 2: Start profiling
curl -X POST http://127.0.0.1:30000/start_profile \
    -H "Content-Type: application/json" \
    -d '{"start_step": 3, "num_steps": 10, "activities": ["CUDA_PROFILER"]}'
```

### Method 2: Simple Approach

```bash
# Terminal 1: Server with NVTX
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-layerwise-nvtx-marker \
    --disable-cuda-graph

# Terminal 2: Profile the benchmark
nsys profile --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    -o layerwise_profile \
    python -m sglang.bench_serving --backend sglang --num-prompts 10
```

### NVTX Marker Information

Each layerwise NVTX marker includes:

- **Full module path**: e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct.model.layers.0.self_attn.qkv_proj`
- **Layer type**: Attention, MLP, embedding, etc.
- **Input/output shapes**: Tensor dimensions
- **Parameter shapes**: Weight tensor dimensions

### Adding Custom NVTX Annotations

```python
import nvtx

with nvtx.annotate("description", color="color"):
    # Critical code section
    pass
```

---

## MFU Metrics

Model FLOPs Utilization (MFU) metrics estimate hardware utilization.

### Enabling

```bash
python -m sglang.launch_server \
    --enable-metrics \
    --enable-mfu-metrics
```

### Available Metrics

| Metric | Description |
|--------|-------------|
| `sglang:estimated_flops_per_gpu_total` | Estimated floating-point operations per GPU |
| `sglang:estimated_read_bytes_per_gpu_total` | Estimated bytes read from memory per GPU |
| `sglang:estimated_write_bytes_per_gpu_total` | Estimated bytes written to memory per GPU |

### PromQL Examples

Average TFLOPS per GPU:
```promql
rate(sglang:estimated_flops_per_gpu_total[1m]) / 1e12
```

Average memory bandwidth in GB/s:
```promql
(rate(sglang:estimated_read_bytes_per_gpu_total[1m]) +
 rate(sglang:estimated_write_bytes_per_gpu_total[1m])) / 1e9
```

### Caveats

- These are estimates, not direct hardware counters.
- They reflect modeled traffic and are intended for observability and trend analysis.

---

## Request Time Statistics

The request time statistics module tracks timing information for each request through the pipeline stages.

### Tracked Stages

| Stage | Level | Description |
|-------|-------|-------------|
| `TOKENIZE` | 1 | Input tokenization |
| `API_SERVER_DISPATCH` | 2 | API server request dispatch |
| `SCHEDULE` | 1 | Scheduling decision |
| `PREFILL` | 1 | Prefill forward pass |
| `DECODE` | 1 | Decode forward pass |
| `DECODE_LOOP` | 3 | Complete decode loop |
| `DETOKENIZE` | 1 | Output detokenization |
| `KV_TRANSFER` | 1 | KV cache transfer (disaggregation) |

### Timing Architecture

The timing system uses monotonic time (`time.perf_counter`) for internal measurements and converts to wall-clock time (`time.time`) for external reporting. Periodic calibration handles NTP drift.

### Per-Stage Metrics

When `--enable-metrics` is active, per-stage latency histograms are emitted:

```
sglang:func_latency_seconds{name="tokenize"} ...
sglang:func_latency_seconds{name="schedule"} ...
sglang:func_latency_seconds{name="prefill"} ...
sglang:func_latency_seconds{name="decode"} ...
```

### Prefill Delayer Metrics

When the prefill delayer is active:

| Metric | Description |
|--------|-------------|
| `forward_passes` | Number of passes prefill was delayed |
| `wait_seconds` | Total seconds prefill was delayed |
| `input_estimation` | Estimation of prefillable status (`all`, `none`, `mixed`) |
| `output_allow` | Whether prefill was allowed |
| `output_reason` | Reason for allow/deny decision |
| `actual_execution` | Whether prefill actually executed |

---

## Health Check Endpoints

### `/health`

Returns the server health status.

```bash
curl http://localhost:30000/health
```

Response: HTTP 200 with `{"status": "ok"}` when the server is ready.

### `/health_generate`

Performs a generation-based health check by generating a single token.

```bash
curl http://localhost:30000/health_generate
```

This endpoint is useful for verifying that the entire inference pipeline is functional, including model loading, scheduling, and generation.

### Internal Health Check

The scheduler periodically sends health check requests with the prefix `HEALTH_CHECK_RID_PREFIX` to verify the pipeline is functioning correctly.

---

## Benchmark Tools

SGLang provides four benchmark tools at different levels of the stack.

### Tool Comparison

| Tool | HTTP Server | Scheduler | Use Case |
|------|-------------|-----------|----------|
| `bench_serving` | Yes (async HTTP) | Yes (via server) | Realistic online serving with latency metrics |
| `bench_one_batch_server` | Yes (HTTP) | Yes (via server) | Single-batch E2E latency with HTTP overhead |
| `bench_offline_throughput` | No | Yes (in-process) | Maximum throughput without HTTP overhead |
| `bench_one_batch` | No | No (direct ModelRunner) | Kernel-level profiling of static batch |

### bench_serving

The primary benchmarking tool for realistic serving scenarios:

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --max-concurrency 16 \
    --num-prompts 80 \
    --random-input-len 256 \
    --random-output-len 32 \
    --dataset-name random
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--max-concurrency` | Maximum concurrent requests |
| `--num-prompts` | Total number of prompts (use >= 5x max-concurrency) |
| `--random-input-len` | Input length for random dataset |
| `--random-output-len` | Output length for random dataset |
| `--dataset-name` | Dataset type: `random`, `sharegpt`, etc. |
| `--request-rate` | Requests per second (inf for burst) |
| `--profile` | Enable profiling during benchmark |

**Output metrics:**

- TTFT (Time To First Token): median, mean, p90, p95, p99
- TPOT (Time Per Output Token): median, mean, p90, p95, p99
- ITL (Inter-Token Latency): median, mean, p90, p95, p99
- E2E latency: median, mean, p90, p95, p99
- Throughput: output tokens/second, requests/second

### bench_one_batch_server

```bash
python3 -m sglang.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --batch-size 32 \
    --input-len 256 \
    --output-len 32
```

### bench_offline_throughput

```bash
python3 -m sglang.bench_offline_throughput \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-prompts 10
```

### bench_one_batch

```bash
python3 -m sglang.bench_one_batch \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --batch-size 32 \
    --input-len 256 \
    --output-len 32
```

**Additional options:**

| Option | Description |
|--------|-------------|
| `--load-format dummy` | Use random weights (no real model needed, only config.json) |
| `--json-model-override-args` | Override model config (e.g., fewer layers) |
| `--profile` | Enable profiling |

Example with dummy weights:
```bash
python3 -m sglang.bench_one_batch \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --batch 32 --input-len 256 --output-len 32 \
    --load-format dummy \
    --json-model-override-args '{"num_hidden_layers": 1, "num_key_value_heads": 1}'
```

---

## Production Request Tracing

### Architecture

The production request tracing system provides end-to-end visibility into request processing:

```
Request Tracing Pipeline:
HTTP Request -> TokenizerManager -> ZMQ -> Scheduler -> TpWorker -> GPU -> Results -> Response
     |               |                          |              |
     v               v                          v              v
  [Trace Start]  [Tokenize Slice]     [Schedule Slice]  [Forward Slice]
```

### Observability Source Files

| File | Description |
|------|-------------|
| `metrics_collector.py` | Prometheus metrics collection and emission |
| `req_time_stats.py` | Per-request time statistics tracking |
| `trace.py` | OpenTelemetry trace integration |
| `func_timer.py` | Function-level timing decorator |
| `scheduler_metrics_mixin.py` | Scheduler-specific metrics |
| `request_metrics_exporter.py` | Request-level metric export |
| `cpu_monitor.py` | CPU utilization monitoring |
| `label_transform.py` | Prometheus label transformation |

### SchedulerStats Dataclass

The `SchedulerStats` dataclass collects comprehensive scheduler state for metrics emission:

```python
@dataclass
class SchedulerStats:
    # Basics
    num_running_reqs: QueueCount
    num_queue_reqs: QueueCount
    gen_throughput: float
    cache_hit_rate: float

    # Memory pool usage
    token_usage: float          # Overall (max of full, swa, mamba)
    full_token_usage: float     # Full-attention KV cache
    swa_token_usage: float      # Sliding window attention
    mamba_usage: float          # Mamba SSM state

    # Absolute token counts
    num_used_tokens: int
    kv_available_tokens: int
    kv_evictable_tokens: int
    kv_used_tokens: int

    # Speculative decoding
    spec_accept_length: float
    spec_accept_rate: float

    # Retract
    num_retracted_reqs: int
    num_paused_reqs: int

    # Utilization
    utilization: float
    fwd_occupancy: float

    # CUDA graph
    is_cuda_graph: int
```

---

## Grafana Dashboard Setup

### Quick Start

1. Start the SGLang server with metrics:
```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-metrics \
    --enable-mfu-metrics \
    --port 30000
```

2. Start the monitoring stack:
```bash
cd examples/monitoring
docker compose up -d
```

3. Access:
   - **Grafana**: http://localhost:3000 (admin/admin)
   - **Prometheus**: http://localhost:9090

### Configuration Files

| File | Purpose |
|------|---------|
| `docker-compose.yaml` | Prometheus and Grafana services |
| `prometheus.yaml` | Scrape configuration |
| `grafana/datasources/datasource.yaml` | Prometheus data source |
| `grafana/dashboards/config/dashboard.yaml` | Dashboard auto-loading |
| `grafana/dashboards/json/sglang-dashboard.json` | Pre-built dashboard |

### Dashboard Panels

The pre-built Grafana dashboard includes:

- **Throughput**: Generation throughput (tokens/s), request throughput (req/s)
- **Latency**: TTFT, TPOT, ITL percentiles (p50, p90, p95, p99)
- **Cache**: Cache hit rate, token usage
- **Queue**: Running requests, queued requests
- **Memory**: KV cache utilization, available/evictable/used tokens
- **Hardware**: MFU (estimated TFLOPS), memory bandwidth

### Troubleshooting

**Port Conflicts:**
```bash
docker ps  # Check running containers
lsof -i :<port>  # Check port usage
docker stop <container_id>  # Stop conflicting container
```

**No Data:**
- Generate traffic: `python3 -m sglang.bench_serving --backend sglang --num-prompts 100`
- Check Prometheus targets: http://localhost:9090 -> Status -> Targets
- Verify `model_name` labels match dashboard variables

**Docker Networking:**
- If SGLang is on the host and Prometheus is in Docker, use `host.docker.internal` or the host's IP in `prometheus.yaml`.

---

## HTTP Profiling API Reference

### POST /start_profile

Start profiling on the server.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | `SGLANG_TORCH_PROFILER_DIR` or `/tmp` | Directory for trace files |
| `num_steps` | int | None (manual stop) | Number of steps to profile |
| `start_step` | int | 0 | Step to start profiling (inclusive) |
| `activities` | list | `["CPU", "GPU"]` | Activities to profile |
| `merge_profiles` | bool | false | Merge distributed traces |

**Special activities:**
- `"CUDA_PROFILER"`: Triggers `cudaProfilerStart/Stop` for Nsight Systems integration.

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
    -H "Content-Type: application/json" \
    -d '{"output_dir": "/tmp/profiles", "start_step": 5, "num_steps": 10}'
```

### POST /stop_profile

Stop profiling and save traces.

```bash
curl -X POST http://127.0.0.1:30000/stop_profile
```

---

*This reference covers the observability and profiling capabilities of SGLang. For scheduling internals, see the Scheduling and Memory Management reference. For performance tuning, see the Hyperparameter Tuning documentation.*
