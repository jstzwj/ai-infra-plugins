# vLLM Observability and Profiling Reference

This document provides a comprehensive reference for vLLM's observability systems including metrics, tracing, profiling, logging, and usage statistics.

---

## Table of Contents

1. [Overview](#overview)
2. [Observability Configuration](#observability-configuration)
3. [Prometheus Metrics](#prometheus-metrics)
4. [Metric Types and Definitions](#metric-types-and-definitions)
5. [Metrics Reader API](#metrics-reader-api)
6. [Stats Collection](#stats-collection)
7. [Stat Loggers](#stat-loggers)
8. [Performance Metrics (MFU)](#performance-metrics-mfu)
9. [OpenTelemetry Tracing](#opentelemetry-tracing)
10. [Tracing Utilities](#tracing-utilities)
11. [Profiler System](#profiler-system)
12. [Profiler Configuration](#profiler-configuration)
13. [Layerwise Profiling](#layerwise-profiling)
14. [Logging](#logging)
15. [Usage Statistics](#usage-statistics)
16. [Multiprocess Prometheus](#multiprocess-prometheus)
17. [Metrics Utilities](#metrics-utilities)
18. [KV Cache Residency Metrics](#kv-cache-residency-metrics)

---

## Overview

vLLM provides comprehensive observability through multiple layers:

- **Prometheus Metrics**: Real-time counters, gauges, and histograms for inference performance
- **OpenTelemetry Tracing**: Distributed tracing with span attributes for request lifecycle
- **GPU Profiling**: PyTorch profiler and CUDA profiler integration
- **Layerwise Profiling**: Detailed per-layer performance analysis
- **Logging**: Structured logging with customizable formatters
- **Usage Statistics**: Anonymous telemetry for development insights

---

## Observability Configuration

**Source:** `vllm/config/observability.py`

```python
@config
class ObservabilityConfig:
    show_hidden_metrics_for_version: str | None = None
    otlp_traces_endpoint: str | None = None
    collect_detailed_traces: bool = False
    kv_cache_metrics: bool = False
    kv_cache_metrics_sample: int = 0
    cudagraph_metrics: bool = False
    enable_layerwise_nvtx_tracing: bool = False
    enable_mfu_metrics: bool = False
    enable_mm_processor_stats: bool = False
    enable_logging_iteration_details: bool = False
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `show_hidden_metrics_for_version` | `str \| None` | `None` | Show metrics hidden after this version |
| `otlp_traces_endpoint` | `str \| None` | `None` | OTLP traces endpoint URL (gRPC or HTTP) |
| `collect_detailed_traces` | `bool` | `False` | Collect detailed traces for all requests |
| `kv_cache_metrics` | `bool` | `False` | Enable KV cache residency metrics |
| `kv_cache_metrics_sample` | `int` | `0` | Sample rate for KV cache metrics (0 = disabled) |
| `cudagraph_metrics` | `bool` | `False` | Enable CUDA graph metrics |
| `enable_layerwise_nvtx_tracing` | `bool` | `False` | Enable per-layer NVTX tracing |
| `enable_mfu_metrics` | `bool` | `False` | Enable Model FLOPs Utilization metrics |
| `enable_mm_processor_stats` | `bool` | `False` | Enable multimodal processor statistics |
| `enable_logging_iteration_details` | `bool` | `False` | Log detailed per-iteration information |

---

## Prometheus Metrics

**Source:** `vllm/v1/metrics/loggers.py` (PrometheusStatLogger)

vLLM exposes a comprehensive set of Prometheus metrics. All metrics are prefixed with `vllm:`.

### Gauge Metrics

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `vllm:num_requests_running` | `model_name` | Number of requests currently running |
| `vllm:num_requests_waiting` | `model_name` | Number of requests waiting to be processed |
| `vllm:num_requests_waiting_by_reason` | `model_name`, `reason` | Waiting requests grouped by reason |
| `vllm:kv_cache_usage_perc` | `model_name` | KV cache usage percentage |
| `vllm:engine_sleep_state` | `model_name`, `engine_id` | Whether the engine is sleeping (0 or 1) |
| `vllm:lora_requests_info` | `model_name`, `running`, `max`, `waiting` | LoRA adapter request information |

### Counter Metrics

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `vllm:prefix_cache_queries_total` | `model_name` | Total prefix cache query count |
| `vllm:prefix_cache_hits_total` | `model_name` | Total prefix cache hit count |
| `vllm:external_prefix_cache_queries_total` | `model_name` | External prefix cache queries |
| `vllm:external_prefix_cache_hits_total` | `model_name` | External prefix cache hits |
| `vllm:mm_cache_queries_total` | `model_name` | Multimodal cache query count |
| `vllm:mm_cache_hits_total` | `model_name` | Multimodal cache hit count |
| `vllm:num_preemptions_total` | `model_name` | Total number of preemptions |
| `vllm:prompt_tokens_total` | `model_name` | Total prompt tokens processed |
| `vllm:prompt_tokens_by_source_total` | `model_name`, `source` | Prompt tokens grouped by source (computed, external_cache, etc.) |
| `vllm:prompt_tokens_cached_total` | `model_name` | Prompt tokens served from cache |
| `vllm:generation_tokens_total` | `model_name` | Total generation tokens produced |
| `vllm:request_success_total` | `model_name`, `finishing_reason` | Successful request completions by reason |
| `vllm:corrupted_requests_total` | `model_name` | Number of corrupted requests |
| `vllm:spec_decode_num_drafts_total` | `model_name` | Speculative decoding draft steps |
| `vllm:spec_decode_num_draft_tokens_total` | `model_name` | Speculative decoding drafted tokens |
| `vllm:spec_decode_num_accepted_tokens_total` | `model_name` | Speculative decoding accepted tokens |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | `model_name`, `position` | Accepted tokens per spec position |

### Histogram Metrics

| Metric Name | Labels | Buckets | Description |
|-------------|--------|---------|-------------|
| `vllm:request_prompt_tokens` | `model_name` | Exponential 1-131072 | Prompt tokens per request |
| `vllm:request_generation_tokens` | `model_name` | Exponential 1-131072 | Generation tokens per request |
| `vllm:iteration_tokens_total` | `model_name` | Exponential 1-8192 | Total tokens per iteration |
| `vllm:request_max_num_generation_tokens` | `model_name` | Exponential 1-131072 | Max generation tokens per request |
| `vllm:request_params_n` | `model_name` | Exponential 1-128 | Parameter `n` per request |
| `vllm:request_params_max_tokens` | `model_name` | Exponential 1-131072 | `max_tokens` parameter per request |
| `vllm:time_to_first_token_seconds` | `model_name` | Exponential 0.001-10 | Time to first token (TTFT) |
| `vllm:inter_token_latency_seconds` | `model_name` | Exponential 0.001-10 | Inter-token latency (ITL) |
| `vllm:request_time_per_output_token_seconds` | `model_name` | Exponential 0.001-10 | Time per output token (TPOT) |
| `vllm:e2e_request_latency_seconds` | `model_name` | Exponential 1-300 | End-to-end request latency |
| `vllm:request_queue_time_seconds` | `model_name` | Exponential 0.001-10 | Time spent in queue |
| `vllm:request_inference_time_seconds` | `model_name` | Exponential 1-300 | Inference time per request |
| `vllm:request_prefill_time_seconds` | `model_name` | Exponential 0.01-60 | Prefill time per request |
| `vllm:request_decode_time_seconds` | `model_name` | Exponential 0.01-60 | Decode time per request |
| `vllm:request_prefill_kv_computed_tokens` | `model_name` | Exponential 1-131072 | KV computed tokens during prefill |
| `vllm:kv_block_lifetime_seconds` | `model_name` | Exponential 0.1-3600 | KV block lifetime |
| `vllm:kv_block_idle_before_evict_seconds` | `model_name` | Exponential 0.1-3600 | KV block idle time before eviction |
| `vllm:kv_block_reuse_gap_seconds` | `model_name` | Exponential 0.1-3600 | KV block reuse gap time |

### MFU Prometheus Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:estimated_flops_per_gpu_total` | Counter | Estimated FLOPs per GPU |
| `vllm:estimated_read_bytes_per_gpu_total` | Counter | Estimated bytes read per GPU |
| `vllm:estimated_write_bytes_per_gpu_total` | Counter | Estimated bytes written per GPU |

---

## Metric Types and Definitions

**Source:** `vllm/v1/metrics/reader.py`

The metrics reader provides a typed API for accessing in-memory Prometheus metrics.

### Metric Data Classes

```python
@dataclass
class Metric:
    name: str
    labels: dict[str, str]

@dataclass
class Counter(Metric):
    value: int

@dataclass
class Vector(Metric):
    values: list[int]

@dataclass
class Gauge(Metric):
    value: float

@dataclass
class Histogram(Metric):
    count: int
    sum: float
    buckets: dict[str, int]
```

The `Vector` type is a vLLM-specific metric that models `vllm:spec_decode_num_accepted_tokens_per_pos` as an ordered array of integer counters.

---

## Metrics Reader API

**Source:** `vllm/v1/metrics/reader.py`

### get_metrics_snapshot

```python
def get_metrics_snapshot() -> list[Metric]
```

Returns a snapshot of all vLLM Prometheus metrics. Only metrics with names starting with `vllm:` are included.

Example usage:
```python
for metric in llm.get_metrics():
    if isinstance(metric, Counter):
        print(f"{metric.name} = {metric.value}")
    elif isinstance(metric, Gauge):
        print(f"{metric.name} = {metric.value}")
    elif isinstance(metric, Histogram):
        print(f"{metric.name}: sum={metric.sum}, count={metric.count}")
        for bucket_le, value in metric.buckets.items():
            print(f"  {bucket_le} = {value}")
    elif isinstance(metric, Vector):
        print(f"{metric.name}: {metric.values}")
```

### Histogram Digestion

The `_digest_histogram` function processes raw Prometheus histogram samples into structured `Histogram` objects. It handles data parallel (DP) deployments where histogram data comes from multiple engine instances, each labeled with an `idx` label.

Input samples:
```
labels = {bucket: 100, idx: 0}, value = 2
labels = {bucket: 200, idx: 0}, value = 4
labels = {bucket: Inf, idx: 0}, value = 10
labels = {bucket: 100, idx: 1}, value = 1
```

Output:
```
{idx: 0}, {"100": 2, "200": 4, "Inf": 10}, 10, 2000
{idx: 1}, {"100": 1, "200": 5, "Inf": 7},   7, 1200
```

### Vector Digestion

The `_digest_num_accepted_by_pos_samples` function processes per-position acceptance samples from speculative decoding:

Input:
```
labels = {pos: 0, idx: 0}, value = 10
labels = {pos: 1, idx: 0}, value = 7
labels = {pos: 2, idx: 0}, value = 2
```

Output:
```
{idx: 0}, [10, 7, 2]
```

---

## Stats Collection

**Source:** `vllm/v1/metrics/stats.py`

### BaseCacheStats

```python
@dataclass
class BaseCacheStats:
    requests: int = 0
    queries: int = 0
    hits: int = 0
```

### PrefixCacheStats

```python
@dataclass
class PrefixCacheStats(BaseCacheStats):
    external_queries: int = 0
    external_hits: int = 0
```

### MultiModalCacheStats

```python
@dataclass
class MultiModalCacheStats(BaseCacheStats):
    pass
```

### CachingMetrics

```python
@dataclass
class CachingMetrics:
    """Tracks cache hit rates over time using sliding window."""
    requests: deque[int]       # Recent request counts
    hits: deque[int]           # Recent hit counts
    sliding_window_size: int   # Window size for averaging

    def update(self, requests: int, hits: int) -> None
    def hit_rate(self) -> float  # Returns sliding window hit rate
```

### SchedulerStats

```python
@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_cache_usage: float = 0.0
    num_corrupted_requests: int = 0
    prefix_cache_stats: PrefixCacheStats | None = None
    mm_cache_stats: MultiModalCacheStats | None = None
    spec_decoding_stats: SpecDecodingStats | None = None
    lora_stats: LoRAStats | None = None
    waiting_reason_counts: dict[str, int] | None = None
```

### RequestStateStats

```python
@dataclass
class RequestStateStats:
    num_computed_tokens: int = 0
    num_resumed_tokens: int = 0
```

### FinishedRequestStats

```python
@dataclass
class FinishedRequestStats:
    arrival_time: float = 0.0
    last_token_time: float = 0.0
    first_token_time: float = 0.0
    time_in_queue: float = 0.0
    finished_time: float = 0.0
    queue_time: float = 0.0
    inference_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    spec_decoding_stats: SpecDecodingStats | None = None
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_gen_tokens: int = 0
    n_param: int = 0
    max_tokens_param: int = 0
    finished_reason: str = ""
    prompt_source: str = ""
    num_prefill_kv_computed_tokens: int = 0
    model_name: str = ""
```

### PrefillStats

```python
@dataclass
class PrefillStats:
    prefill_tokens: int = 0
    prefill_time: float = 0.0
    prefill_num_computed_tokens: int = 0
```

### PromptTokenStats

```python
@dataclass
class PromptTokenStats:
    source: str = ""
    num_prompt_tokens: int = 0
    num_prompt_tokens_cached: int = 0
```

### IterationStats

```python
@dataclass
class IterationStats:
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    num_tokens: int = 0
    total_num_tokens: int = 0
    num_preempted_requests: int = 0
    num_running_requests: int = 0
    num_waiting_requests: int = 0
    spec_decoding_stats: SpecDecodingStats | None = None
    timestamp: float = 0.0
    first_token_time: float | None = None
    time_to_first_token: float | None = None
    inter_token_latency: float | None = None
    prefill_stats: PrefillStats | None = None
    prompt_token_stats_list: list[PromptTokenStats] | None = None
    is_sleeping: bool = False
    engine_sleep_state: int | None = None
```

### LoRAStats

```python
@dataclass
class LoRAStats:
    max_lora: int
    running_lora_adapters: list[str]
    waiting_lora_adapters: list[str]
```

### LoRARequestStates

```python
@dataclass
class LoRARequestStates:
    request_states: dict[str, RequestStateStats]
```

---

## Stat Loggers

**Source:** `vllm/v1/metrics/loggers.py`

### StatLoggerBase (ABC)

```python
class StatLoggerBase(ABC):
    @abstractmethod
    def log(self, scheduler_stats: SchedulerStats, iteration_stats: IterationStats) -> None
    @abstractmethod
    def record_request_finished(self, finished_stats: FinishedRequestStats) -> None
    @abstractmethod
    def observe_kv_cache_block_lifecycle(self, block_id: int, event: str, **kwargs) -> None
```

### LoggingStatLogger

Text-based logging logger that prints periodic statistics:

- Throughput (tokens/second for prompt and generation)
- Cache hit rate (prefix cache sliding window)
- Speculative decoding acceptance rate
- Running/waiting request counts

Key methods:
- `log(scheduler_stats, iteration_stats)` - Logs per-iteration statistics
- `record_request_finished(finished_stats)` - Logs finished request stats

### AggregatedLoggingStatLogger

Aggregates statistics across data parallel (DP) engines before logging. Useful for multi-engine deployments.

### PrometheusStatLogger

The main Prometheus metrics logger. Registers and updates all vLLM Prometheus metrics.

#### Initialization

```python
class PrometheusStatLogger(StatLoggerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_ids: list[int],
    )
```

Creates per-engine labeled metrics. Each metric is labeled with `model_name` and optionally `engine_id`.

#### Metric Registration

Metrics are registered at initialization time using the `prometheus_client` library. All metrics use the `vllm:` prefix convention.

#### Key Methods

- `log(scheduler_stats, iteration_stats)` - Updates gauge and counter metrics per iteration
- `record_request_finished(finished_stats)` - Records histogram observations for finished requests
- `observe_kv_cache_block_lifecycle(block_id, event, **kwargs)` - Observes KV cache block lifecycle events

### StatLoggerManager

```python
class StatLoggerManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        dp_size: int,
        engine_ids: list[int],
    )
```

Manages multiple stat loggers. Handles DP aggregation by creating per-engine loggers and an aggregated logger.

Methods:
- `log(scheduler_stats_per_engine, iteration_stats_per_engine)` - Logs stats for all engines
- `record_request_finished(finished_stats)` - Records a finished request
- `observe_kv_cache_block_lifecycle(block_id, event, **kwargs)` - Observes KV cache block events

---

## Performance Metrics (MFU)

**Source:** `vllm/v1/metrics/perf.py`

vLLM calculates Model FLOPs Utilization (MFU) metrics by estimating the computational cost of each inference iteration and comparing it against hardware peak performance.

### ExecutionContext

```python
@dataclass
class ExecutionContext:
    batch_size: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    avg_seq_len: int = 0
    max_seq_len: int = 0
```

Tracks the execution context for a batch, distinguishing between prefill and decode tokens.

### ComponentMetrics (ABC)

```python
class ComponentMetrics(ABC):
    @abstractmethod
    def get_num_flops_breakdown(self, ctx: ExecutionContext) -> dict[str, float]

    @abstractmethod
    def get_read_bytes_breakdown(self, ctx: ExecutionContext) -> dict[str, float]

    @abstractmethod
    def get_write_bytes_breakdown(self, ctx: ExecutionContext) -> dict[str, float]
```

Abstract base for per-component performance estimation.

### AttentionMetrics

```python
class AttentionMetrics(ComponentMetrics):
    def __init__(self, hidden_size, num_heads, head_size, num_kv_heads)
```

Estimates FLOPs, read bytes, and write bytes for attention layers:

- **FLOPs**: QKV projection + output projection + attention computation
  - Prefill: `batch_size * num_tokens * (hidden_size * (hidden_size + head_size)) + batch_size * num_tokens * seq_len * head_size * num_heads`
  - Decode: `batch_size * (hidden_size * (hidden_size + head_size) * num_tokens + seq_len * head_size * num_heads * num_tokens)`
- **Read bytes**: Weight matrices + KV cache reads
- **Write bytes**: KV cache writes + output

### FfnMetrics

```python
class FfnMetrics(ComponentMetrics):
    def __init__(self, hidden_size, intermediate_size, num_layers, ...)
```

Estimates FLOPs for feed-forward layers:

- **Dense FFN**: `4 * hidden_size * intermediate_size * num_tokens * num_layers`
- **MoE FFN**: Accounts for expert routing, expert FFN computation, and shared expert (if applicable)

### UnembedMetrics

```python
class UnembedMetrics(ComponentMetrics):
    def __init__(self, hidden_size, vocab_size)
```

Estimates FLOPs for the unembedding (LM head) layer:
- **FLOPs**: `2 * hidden_size * vocab_size * num_tokens`

### ModelMetrics

```python
class ModelMetrics:
    def __init__(self, attention: AttentionMetrics, ffn: FfnMetrics, unembed: UnembedMetrics)
```

Aggregates all component metrics. Provides total FLOPs, read bytes, and write bytes.

### PerfMetricsProm

```python
class PerfMetricsProm:
    # Prometheus counters
    vllm_estimated_flops_per_gpu: Counter
    vllm_estimated_read_bytes_per_gpu: Counter
    vllm_estimated_write_bytes_per_gpu: Counter
```

Publishes estimated performance metrics to Prometheus.

### Parser Chain

The MFU system uses a chain of config parsers to extract model architecture parameters:

```
BaseConfigParser
  -> BaseAttentionConfigParser
    -> AttentionQuantizationConfigParser
  -> BaseFfnConfigParser
    -> FfnQuantizationConfigParser
```

Each parser extracts relevant parameters from the model's HuggingFace config and returns a `ComponentMetrics` instance.

---

## OpenTelemetry Tracing

**Source:** `vllm/tracing/otel.py` and `vllm/tracing/__init__.py`

vLLM integrates with OpenTelemetry for distributed tracing of request processing.

### Initialization

#### init_tracer

```python
def init_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
)
```

Initializes the OTel tracer with:
- `TracerProvider` with resource attributes
- `BatchSpanProcessor` for efficient span export
- Optional extra attributes attached to all spans

#### init_otel_tracer

```python
def init_otel_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
) -> Tracer
```

Creates an OTel tracer with:
- Resource with service name and extra attributes
- gRPC or HTTP span exporter based on endpoint URL
- BatchSpanProcessor with 5-second export interval

#### init_otel_worker_tracer

```python
def init_otel_worker_tracer(
    instrumenting_module_name: str,
    process_kind: str,
    process_name: str,
) -> Tracer | None
```

Initializes a tracer for worker processes. Adds `process_kind` and `process_name` as resource attributes.

### Span Exporters

#### get_span_exporter

```python
def get_span_exporter(otlp_traces_endpoint: str) -> SpanExporter
```

Creates the appropriate span exporter based on the endpoint URL:
- URLs containing `https://` or `http://`: Uses `OTLPSpanExporter` with protobuf over HTTP
- All other URLs: Uses `OTLPSpanExporter` with gRPC

### Context Propagation

#### extract_trace_context

```python
def extract_trace_context(headers: dict[str, str]) -> Context | None
```

Extracts trace context from HTTP headers. Looks for W3C `traceparent` and `tracestate` headers.

#### propagate_trace_to_env

```python
def propagate_trace_to_env(context: Context | None) -> dict[str, str] | None
```

Propagates trace context to environment variables for subprocess tracing.

### Instrumentation

#### instrument (decorator)

```python
def instrument(
    obj: Callable | None = None,
    *,
    span_name: str = "",
    attributes: dict[str, str] | None = None,
    record_exception: bool = True,
)
```

Decorator that creates an OTel span around a function. Can be used with or without arguments:

```python
@instrument
def my_function():
    pass

@instrument(span_name="custom_name", attributes={"key": "value"})
def my_function():
    pass
```

#### instrument_manual

```python
def instrument_manual(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Any = None,
    kind: Any = None,
)
```

Manually creates a span with explicit timestamps. Useful for instrumenting code that doesn't fit the decorator pattern.

### Backend Registration

```python
_REGISTERED_TRACING_BACKENDS: dict[str, tuple[
    BackendAvailableFunc,
    InitTracerFunc,
    InitWorkerTracerFunc,
    InstrumentFunc,
    InstrumentManualFunc,
]] = {
    "otel": (
        is_otel_available,
        init_otel_tracer,
        init_otel_worker_tracer,
        instrument_otel,
        manual_instrument_otel,
    ),
}
```

---

## Tracing Utilities

**Source:** `vllm/tracing/utils.py`

### SpanAttributes

Standard span attribute names following OpenTelemetry semantic conventions:

| Attribute | Description |
|-----------|-------------|
| `gen_ai.usage.completion_tokens` | Number of completion tokens |
| `gen_ai.usage.prompt_tokens` | Number of prompt tokens |
| `gen_ai.request.max_tokens` | Maximum tokens requested |
| `gen_ai.request.top_p` | Top-p sampling parameter |
| `gen_ai.request.temperature` | Temperature sampling parameter |
| `gen_ai.response.model` | Model name in response |
| `gen_ai.request.id` | Request ID |
| `gen_ai.request.n` | Number of completions requested |
| `gen_ai.usage.num_sequences` | Number of sequences generated |
| `gen_ai.latency.time_in_queue` | Queue wait time |
| `gen_ai.latency.time_to_first_token` | TTFT |
| `gen_ai.latency.e2e` | End-to-end latency |
| `gen_ai.latency.time_in_scheduler` | Scheduler time |
| `gen_ai.latency.time_in_model_forward` | Model forward pass time |
| `gen_ai.latency.time_in_model_execute` | Model execute time |
| `gen_ai.latency.time_in_model_prefill` | Prefill time |
| `gen_ai.latency.time_in_model_decode` | Decode time |
| `gen_ai.latency.time_in_model_inference` | Total inference time |

### LoadingSpanAttributes

Code-level tracing attributes:

| Attribute | Description |
|-----------|-------------|
| `code.namespace` | Code namespace |
| `code.function` | Function name |
| `code.filepath` | Source file path |
| `code.lineno` | Line number |

### Helper Functions

```python
TRACE_HEADERS = ["traceparent", "tracestate"]

def contains_trace_headers(headers: Mapping[str, str]) -> bool
def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]
def log_tracing_disabled_warning() -> None  # run_once decorated
```

---

## Profiler System

**Source:** `vllm/profiler/wrapper.py`

### WorkerProfiler (ABC)

```python
class WorkerProfiler(ABC):
    def __init__(self, profiler_config: ProfilerConfig) -> None
```

Abstract base class for worker profilers. Manages delayed start, max iteration limits, and lifecycle.

Key attributes:
- `_delay_iters` - Iterations to skip before starting
- `_max_iters` - Maximum profiling iterations
- `_active_iteration_count` - Current iteration count since start
- `_active` - Whether profiler has been triggered
- `_running` - Whether profiler is actually recording

Methods:
- `start()` - Start the profiler (accounting for delay)
- `step()` - Advance one iteration (handles delayed start and max limits)
- `stop()` - Stop the profiler
- `shutdown()` - Ensure profiler is stopped on shutdown
- `annotate_context_manager(name: str)` - Return context manager for trace annotation
- `_profiler_step() -> bool` - Called each step; returns True if data was recorded

### TorchProfilerWrapper

```python
class TorchProfilerWrapper(WorkerProfiler):
    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
        activities: list[TorchProfilerActivity],
        on_trace_ready: Callable | None = None,
    )
```

Wraps PyTorch profiler with vLLM-specific configuration.

Configuration options (from ProfilerConfig):
- `torch_profiler_dir` - Output directory
- `torch_profiler_with_stack` - Enable stack tracing (default True)
- `torch_profiler_with_flops` - Enable FLOPS counting (default False)
- `torch_profiler_use_gzip` - Gzip traces (default True)
- `torch_profiler_dump_cuda_time_total` - Dump CUDA time table (default True)
- `torch_profiler_record_shapes` - Record tensor shapes (default False)
- `torch_profiler_with_memory` - Enable memory profiling (default False)

Schedule support:
- `warmup_iterations` - Warmup steps where data is discarded
- `active_iterations` - Active recording steps (default 5)
- `wait_iterations` - Wait steps before warmup (default 0)

Methods:
- `_start()` - Start `torch.profiler.profile`
- `_stop()` - Stop profiler and optionally dump tables
- `_profiler_step() -> bool` - Call `profiler.step()` for schedule-based profiling
- `_build_profiler_table(sort_key, row_limit) -> str` - Build sorted profiler table
- `_write_profiler_table(rank, table)` - Write table to file
- `annotate_context_manager(name)` - Returns `torch.profiler.record_function(name)`

### CudaProfilerWrapper

```python
class CudaProfilerWrapper(WorkerProfiler):
    def __init__(self, profiler_config: ProfilerConfig)
```

Wraps CUDA profiler (`torch.cuda.profiler`). Simple start/stop with NVTX annotation support.

Methods:
- `_start()` - Start CUDA profiler
- `_stop()` - Stop CUDA profiler
- `annotate_context_manager(name)` - Returns `torch.cuda.nvtx.range(name)`

### Activity Types

```python
TorchProfilerActivity = Literal["CPU", "CUDA", "XPU"]

TorchProfilerActivityMap = {
    "CPU": torch.profiler.ProfilerActivity.CPU,
    "CUDA": torch.profiler.ProfilerActivity.CUDA,
    "XPU": torch.profiler.ProfilerActivity.XPU,
}
```

---

## Profiler Configuration

**Source:** `vllm/config/profiler.py`

```python
@config
class ProfilerConfig:
    profiler: ProfilerKind | None = None
    torch_profiler_dir: str = ""
    torch_profiler_with_stack: bool = True
    torch_profiler_with_flops: bool = False
    torch_profiler_use_gzip: bool = True
    torch_profiler_dump_cuda_time_total: bool = True
    torch_profiler_record_shapes: bool = False
    torch_profiler_with_memory: bool = False
    ignore_frontend: bool = False
    delay_iterations: int = 0
    max_iterations: int = 0
    warmup_iterations: int = 0
    active_iterations: int = 5
    wait_iterations: int = 0
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `profiler` | `"torch" \| "cuda" \| None` | `None` | Which profiler to use |
| `torch_profiler_dir` | `str` | `""` | Output directory for torch profiler traces |
| `torch_profiler_with_stack` | `bool` | `True` | Enable stack tracing |
| `torch_profiler_with_flops` | `bool` | `False` | Enable FLOPS counting |
| `torch_profiler_use_gzip` | `bool` | `True` | Gzip-compress traces |
| `torch_profiler_dump_cuda_time_total` | `bool` | `True` | Dump CUDA time summary |
| `torch_profiler_record_shapes` | `bool` | `False` | Record tensor shapes |
| `torch_profiler_with_memory` | `bool` | `False` | Enable memory profiling |
| `ignore_frontend` | `bool` | `False` | Skip frontend profiling |
| `delay_iterations` | `int` | `0` | Iterations to skip before starting |
| `max_iterations` | `int` | `0` | Max iterations to profile (0 = unlimited) |
| `warmup_iterations` | `int` | `0` | Warmup iterations for schedule |
| `active_iterations` | `int` | `5` | Active recording iterations |
| `wait_iterations` | `int` | `0` | Wait iterations before warmup |

### Validation

- `torch_profiler_dir` is required when `profiler == "torch"`
- Supports URI paths (gs://, s3://, hdfs://) for cloud storage
- Local paths are expanded and made absolute
- Warns when using delay/limit without `ignore_frontend`

---

## Layerwise Profiling

**Source:** `vllm/profiler/layerwise_profile.py` and `vllm/profiler/utils.py`

vLLM provides detailed layerwise profiling that breaks down GPU time by model layer and operation.

### layerwise_profile

```python
class layerwise_profile(profile):
    def __init__(self, num_running_seqs: int | None = None)
```

Context manager for layerwise profiling. Uses PyTorch profiler with CPU and CUDA activities, recording shapes, stacks, and modules.

```python
with layerwise_profile(num_running_seqs=4) as prof:
    model(input_ids, positions)
results = prof.results  # LayerwiseProfileResults
```

### LayerwiseProfileResults

```python
@dataclass
class LayerwiseProfileResults(profile):
    _kineto_results: _ProfilerResult
    _kineto_event_correlation_map: dict[int, list[_KinetoEvent]]
    _event_correlation_map: dict[int, list[FunctionEvent]]
    _module_tree: list[_ModuleTreeNode]
    _model_stats_tree: list[_StatsTreeNode[ModelStatsEntry]]
    _summary_stats_tree: list[_StatsTreeNode[SummaryStatsEntry]]
    num_running_seqs: int | None = None
```

Key methods:

- `print_model_table(column_widths=None)` - Print model-layer performance table with columns: name, cpu_time_us, cuda_time_us, pct_cuda_time, trace
- `print_summary_table(column_widths=None)` - Print aggregated summary table with columns: name, cuda_time_us, pct_cuda_time, invocations
- `export_model_stats_table_csv(filename)` - Export model stats to CSV
- `export_summary_stats_table_csv(filename)` - Export summary stats to CSV
- `convert_stats_to_dict() -> dict` - Convert to JSON-compatible dictionary

### Data Classes

#### ModelStatsEntry

```python
@dataclass
class ModelStatsEntry:
    name: str
    cpu_time_us: float
    cuda_time_us: float
    pct_cuda_time: float
    trace: str
```

#### SummaryStatsEntry

```python
@dataclass
class SummaryStatsEntry:
    name: str
    cuda_time_us: float
    pct_cuda_time: float
    invocations: int
```

### Internal Tree Building

The profiler builds two tree structures:

1. **Module Tree** (`_module_tree`): Represents the actual module call hierarchy. Built from PyTorch profiler's experimental event tree by traversing events with module information.

2. **Stats Trees**: Derived from the module tree:
   - **Summary Stats Tree**: Aggregates identical operations (same name and parent chain). Shows total time and invocation count for each unique operation.
   - **Model Stats Tree**: Preserves the full call hierarchy. Shows individual timing for each layer invocation.

### Profiler Utilities

**Source:** `vllm/profiler/utils.py`

#### String Functions

```python
def trim_string_front(string: str, width: int) -> str
def trim_string_back(string: str, width: int) -> str
def indent_string(string: str, indent: int, indent_style: Callable | str = " ") -> str
```

#### TablePrinter

```python
class TablePrinter:
    def __init__(self, row_cls: type, column_widths: dict[str, int])
    def print_table(self, rows: list)
```

Prints formatted tables with configurable column widths.

#### Event Inspection Functions

```python
def event_has_module(event: _ProfilerEvent) -> bool
def event_is_torch_op(event: _ProfilerEvent) -> bool
def event_arg_repr(arg) -> str
def event_torch_op_repr(event: _ProfilerEvent) -> str
def event_module_repr(event: _ProfilerEvent) -> str
def event_torch_op_stack_trace(event: _ProfilerEvent, until: Callable) -> str
```

---

## Logging

**Source:** `vllm/logging_utils/formatter.py`

### NewLineFormatter

```python
class NewLineFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None, style="%")
```

Custom log formatter that:
- Adds logging prefix to newlines to align multi-line messages
- In DEBUG mode, shows relative paths with shortened file paths
- Uses `\r\n` for line continuations to maintain alignment

Path shortening rules:
- Removes leading `vllm` folder
- For `v1` paths: keeps first two and last two levels, collapses middle as `...`
- Otherwise: keeps first and last two levels
- Examples:
  - `vllm/model_executor/layers/quantization/utils/fp8_utils.py` -> `model_executor/.../quantization/utils/fp8_utils.py`
  - `vllm/model_executor/layers/quantization/awq.py` -> `model_executor/layers/quantization/awq.py`

### ColoredFormatter

```python
class ColoredFormatter(NewLineFormatter):
    COLORS = {
        "DEBUG": "\033[37m",      # White
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    GREY = "\033[90m"
    RESET = "\033[0m"
```

Extends `NewLineFormatter` with ANSI color codes:
- Timestamps and file info are colored grey
- Log level names use severity-appropriate colors

---

## Usage Statistics

**Source:** `vllm/usage/usage_lib.py`

vLLM collects anonymous usage statistics to help improve the project. This can be disabled via environment variables.

### UsageContext

```python
class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"
```

### UsageMessage

```python
class UsageMessage:
    def __init__(self)
```

Collects and sends platform information to the usage stats server.

#### Collected Data

| Field | Description |
|-------|-------------|
| `uuid` | Random UUID for this vLLM instance |
| `provider` | Cloud provider (AWS, GCP, AZURE, OCI, RUNPOD, UNKNOWN) |
| `num_cpu` | Number of CPUs |
| `cpu_type` | CPU brand name |
| `cpu_family_model_stepping` | CPU family, model, stepping |
| `total_memory` | Total system memory |
| `architecture` | Machine architecture |
| `platform` | Platform string |
| `xpu_runtime` | XPU runtime version |
| `cuda_runtime` | CUDA runtime version |
| `gpu_count` | Number of GPUs |
| `gpu_type` | GPU model name |
| `gpu_memory_per_device` | GPU memory per device |
| `env_var_json` | Selected environment variables |
| `model_architecture` | Model architecture name |
| `vllm_version` | vLLM version |
| `context` | Usage context |
| `log_time` | Log timestamp (nanoseconds) |
| `source` | Usage source |

#### Cloud Provider Detection

`_detect_cloud_provider()` checks:
- DMI files: `/sys/class/dmi/id/product_version`, `bios_vendor`, `product_name`, `chassis_asset_tag`, `sys_vendor`
- Maps to: amazon -> AWS, microsoft -> AZURE, google -> GCP, oraclecloud -> OCI
- Environment variables: `RUNPOD_DC_ID` -> RUNPOD

#### Methods

- `report_usage(model_architecture, usage_context, extra_kvs=None)` - Reports usage in a background thread
- `_report_usage_once(...)` - Sends initial platform information
- `_report_continuous_usage()` - Sends heartbeat every 10 minutes with runtime data
- `_send_to_server(data)` - POSTs JSON to usage stats server
- `_write_to_file(data)` - Appends JSON to `$VLLM_CONFIG_ROOT/usage_stats.json`

### Disabling Usage Statistics

Usage stats are disabled if any of these conditions are true:
- `VLLM_DO_NOT_TRACK=1`
- `DO_NOT_TRACK=1`
- `VLLM_NO_USAGE_STATS=1`
- File `$HOME/.config/vllm/do_not_track` exists

### Runtime Data

```python
def set_runtime_usage_data(key: str, value: str | int | bool) -> None
```

Sets global key-value pairs that are sent with every heartbeat.

### Environment Variables Collected

```python
_USAGE_ENV_VARS_TO_COLLECT = [
    "VLLM_USE_MODELSCOPE",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_PP_LAYER_PARTITION",
    "VLLM_USE_TRITON_AWQ",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
]
```

---

## Multiprocess Prometheus

**Source:** `vllm/v1/metrics/prometheus.py`

### Functions

```python
def setup_multiprocess_prometheus() -> None
```

Sets up Prometheus for multiprocess mode. Must be called before any metrics are registered. Uses `prometheus_client`'s multiprocess mode with a shared directory.

```python
def get_prometheus_registry() -> CollectorRegistry
```

Returns the appropriate Prometheus registry for the current process.

```python
def unregister_vllm_metrics() -> None
```

Unregisters all vLLM metrics from the registry. Used during testing and re-initialization.

```python
def shutdown_prometheus() -> None
```

Shuts down the Prometheus client, cleaning up multiprocess directories.

---

## Metrics Utilities

**Source:** `vllm/v1/metrics/utils.py`

### Type Alias

```python
PromMetric: TypeAlias = Gauge | Counter | Histogram
```

### create_metric_per_engine

```python
def create_metric_per_engine(
    metric: PromMetric,
    per_engine_labelvalues: dict[int, list[object]],
) -> dict[int, PromMetric]
```

Creates a labeled metric child for each engine index. Used in DP deployments where each engine needs its own labeled metric instance.

---

## KV Cache Residency Metrics

When `kv_cache_metrics` is enabled in `ObservabilityConfig`, vLLM tracks the lifecycle of KV cache blocks:

### Lifecycle Events

| Event | Description |
|-------|-------------|
| `allocate` | Block is allocated for a sequence |
| `free` | Block is freed back to the pool |
| `cache_hit` | Block is reused from the cache |

### Histogram Metrics

| Metric | Description |
|--------|-------------|
| `vllm:kv_block_lifetime_seconds` | Time from allocation to free |
| `vllm:kv_block_idle_before_evict_seconds` | Time a block sits idle before being evicted |
| `vllm:kv_block_reuse_gap_seconds` | Time between successive uses of a cached block |

### Sampling

When `kv_cache_metrics_sample > 0`, only a fraction of block events are tracked to reduce overhead. The sample rate determines how many blocks out of each batch are monitored.
