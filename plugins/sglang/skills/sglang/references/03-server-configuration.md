# SGLang Server Configuration Reference

This document provides the complete reference for all SGLang server configuration parameters.
The `ServerArgs` dataclass in `python/sglang/srt/server_args.py` defines 200+ configurable
parameters organized into functional groups.

---

## Table of Contents

1. [ServerArgs Overview](#serverargs-overview)
2. [Model and Tokenizer Parameters](#model-and-tokenizer-parameters)
3. [HTTP Server Parameters](#http-server-parameters)
4. [SSL/TLS Parameters](#ssltls-parameters)
5. [Quantization and Data Type Parameters](#quantization-and-data-type-parameters)
6. [Memory and Scheduling Parameters](#memory-and-scheduling-parameters)
7. [Runtime Options](#runtime-options)
8. [Logging Parameters](#logging-parameters)
9. [API Parameters](#api-parameters)
10. [Data Parallelism Parameters](#data-parallelism-parameters)
11. [Multi-Node Distributed Parameters](#multi-node-distributed-parameters)
12. [Tensor Parallelism Parameters](#tensor-parallelism-parameters)
13. [Expert Parallelism Parameters](#expert-parallelism-parameters)
14. [Pipeline Parallelism Parameters](#pipeline-parallelism-parameters)
15. [LoRA Parameters](#lora-parameters)
16. [Kernel Backend Parameters](#kernel-backend-parameters)
17. [Speculative Decoding Parameters](#speculative-decoding-parameters)
18. [N-gram Speculative Parameters](#n-gram-speculative-parameters)
19. [Mamba Cache Parameters](#mamba-cache-parameters)
20. [Hierarchical Cache Parameters](#hierarchical-cache-parameters)
21. [KV-Cache Parameters](#kv-cache-parameters)
22. [CUDA Graph Parameters](#cuda-graph-parameters)
23. [Disaggregation Parameters](#disaggregation-parameters)
24. [Encoder Disaggregation Parameters](#encoder-disaggregation-parameters)
25. [Multi-Modal Parameters](#multi-modal-parameters)
26. [Offloading Parameters](#offloading-parameters)
27. [Debug and Optimization Parameters](#debug-and-optimization-parameters)
28. [Context Parallelism Parameters](#context-parallelism-parameters)
29. [Dynamic Batch Tokenizer Parameters](#dynamic-batch-tokenizer-parameters)
30. [Debug Tensor Dump Parameters](#debug-tensor-dump-parameters)
31. [Weight Loading Parameters](#weight-loading-parameters)
32. [Scoring Parameters](#scoring-parameters)
33. [Diffusion LLM Parameters](#diffusion-llm-parameters)
34. [Validation Rules and Cross-Dependencies](#validation-rules-and-cross-dependencies)
35. [CLI Flag Mapping](#cli-flag-mapping)

---

## ServerArgs Overview

`ServerArgs` is a Python dataclass that holds all configuration for the SGLang server. It is
defined in `python/sglang/srt/server_args.py` and supports:

- **CLI argument parsing**: via `add_cli_args()` and `from_cli_args()` static methods
- **Programmatic construction**: instantiate directly with keyword arguments
- **Extensive validation**: `__post_init__()` validates all parameters and sets defaults
- **Environment variable overrides**: many settings can be overridden via environment variables

### Creating ServerArgs

```python
from sglang import ServerArgs

# From keyword arguments
args = ServerArgs(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tp_size=4,
    host="0.0.0.0",
    port=30000,
)

# From CLI arguments
import argparse
parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
args = ServerArgs.from_cli_args(parser.parse_args())

# For offline engine (same parameters)
import sglang as sgl
llm = sgl.Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    tp_size=4,
)
```

---

## Model and Tokenizer Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `model_path` | `str` | (required) | `--model-path` | Path or HuggingFace model ID to load. Required parameter. |
| `tokenizer_path` | `Optional[str]` | `None` | `--tokenizer-path` | Path to tokenizer. Defaults to `model_path` if not specified. |
| `tokenizer_mode` | `str` | `"auto"` | `--tokenizer-mode` | Tokenizer mode. Options: `"auto"`, `"slow"`. |
| `tokenizer_backend` | `str` | `"huggingface"` | `--tokenizer-backend` | Tokenizer backend. Options: `"huggingface"`. |
| `tokenizer_worker_num` | `int` | `1` | `--tokenizer-worker-num` | Number of tokenizer worker processes for parallel tokenization. |
| `skip_tokenizer_init` | `bool` | `False` | `--skip-tokenizer-init` | Skip tokenizer initialization. Useful for token-ID-only workflows. |
| `load_format` | `str` | `"auto"` | `--load-format` | Model weight loading format. See Load Format Choices below. |
| `model_loader_extra_config` | `str` | `"{}"` | `--model-loader-extra-config` | Extra JSON config for model loader. |
| `trust_remote_code` | `bool` | `False` | `--trust-remote-code` | Allow execution of remote code from HuggingFace model repos. |
| `context_length` | `Optional[int]` | `None` | `--context-length` | Override the maximum context length. If not set, uses model default. |
| `is_embedding` | `bool` | `False` | `--is-embedding` | Run model in embedding mode instead of generation mode. |
| `enable_multimodal` | `Optional[bool]` | `None` | `--enable-multimodal` | Explicitly enable/disable multimodal support. Auto-detected if None. |
| `revision` | `Optional[str]` | `None` | `--revision` | Specific model revision (branch, tag, or commit hash). |
| `model_impl` | `str` | `"auto"` | `--model-impl` | Model implementation. Options: `"auto"`, `"sglang"`, `"transformers"`. |

### Load Format Choices

The `load_format` parameter accepts the following values:

| Value | Description |
|-------|-------------|
| `auto` | Automatically detect format from model files |
| `pt` | PyTorch `.pt` or `.bin` files |
| `safetensors` | SafeTensors format (recommended) |
| `npcache` | NumPy cache format |
| `dummy` | Load dummy weights (for testing/benchmarking) |
| `sharded_state` | Sharded state dict |
| `gguf` | GGUF format (for quantized models) |
| `bitsandbytes` | BitsAndBytes quantized format |
| `mistral` | Mistral-specific format |
| `layered` | Layer-by-layer loading (memory efficient) |
| `flash_rl` | FlashRL format |
| `remote` | Remote loading |
| `remote_instance` | Remote instance loading |
| `fastsafetensors` | Fast SafeTensors loading |
| `private` | Private model loading |
| `runai_streamer` | RunAI streaming loader |

---

## HTTP Server Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `host` | `str` | `"127.0.0.1"` | `--host` | Server bind address. Use `0.0.0.0` for all interfaces. |
| `port` | `int` | `30000` | `--port` | Server bind port. |
| `fastapi_root_path` | `str` | `""` | `--fastapi-root-path` | FastAPI root_path for reverse proxy deployments. |
| `grpc_mode` | `bool` | `False` | `--grpc-mode` | Enable gRPC server alongside HTTP. |
| `skip_server_warmup` | `bool` | `False` | `--skip-server-warmup` | Skip server warmup phase (faster startup, slower first request). |
| `warmups` | `Optional[str]` | `None` | `--warmups` | Custom warmup configuration. |
| `nccl_port` | `Optional[int]` | `None` | `--nccl-port` | Custom NCCL port for distributed communication. |
| `checkpoint_engine_wait_weights_before_ready` | `bool` | `False` | `--checkpoint-engine-wait-weights-before-ready` | Wait for weights before marking server ready. |

---

## SSL/TLS Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `ssl_keyfile` | `Optional[str]` | `None` | `--ssl-keyfile` | Path to SSL private key file. Required if `ssl_certfile` is set. |
| `ssl_certfile` | `Optional[str]` | `None` | `--ssl-certfile` | Path to SSL certificate file. Required if `ssl_keyfile` is set. |
| `ssl_ca_certs` | `Optional[str]` | `None` | `--ssl-ca-certs` | Path to CA certificates file. Requires SSL key and cert. |
| `ssl_keyfile_password` | `Optional[str]` | `None` | `--ssl-keyfile-password` | Password for SSL private key. Requires SSL key and cert. |
| `enable_ssl_refresh` | `bool` | `False` | `--enable-ssl-refresh` | Enable SSL certificate hot-reloading. Requires SSL key and cert. Not compatible with HTTP/2. |
| `enable_http2` | `bool` | `False` | `--enable-http2` | Enable HTTP/2 support via Granian server. Requires `pip install "sglang[http2]"`. Not compatible with `--enable-ssl-refresh` or `--tokenizer-worker-num > 1`. |

### SSL Validation Rules

- `ssl_keyfile` requires `ssl_certfile` and vice versa
- `ssl_ca_certs` requires both `ssl_keyfile` and `ssl_certfile`
- `ssl_keyfile_password` requires both `ssl_keyfile` and `ssl_certfile`
- `enable_ssl_refresh` requires both `ssl_keyfile` and `ssl_certfile`
- `enable_http2` requires the `granian` package
- `enable_http2` is incompatible with `enable_ssl_refresh`
- `enable_http2` does not yet support `tokenizer_worker_num > 1`
- All SSL file paths are validated to exist on disk before server startup

---

## Quantization and Data Type Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `dtype` | `str` | `"auto"` | `--dtype` | Model data type. Options: `"auto"`, `"float16"`, `"bfloat16"`, `"float32"`. Auto selects BF16 on supported hardware. |
| `quantization` | `Optional[str]` | `None` | `--quantization` | Quantization method. See Quantization Choices below. |
| `quantization_param_path` | `Optional[str]` | `None` | `--quantization-param-path` | Path to quantization parameters file. |
| `kv_cache_dtype` | `str` | `"auto"` | `--kv-cache-dtype` | KV-cache data type. Options: `"auto"`, `"fp8_e5m2"`, `"fp8_e4m3"`. |
| `enable_fp32_lm_head` | `bool` | `False` | `--enable-fp32-lm-head` | Use FP32 precision for the language model head. |
| `modelopt_quant` | `Optional[Union[str, Dict]]` | `None` | `--modelopt-quant` | ModelOpt quantization configuration. |
| `modelopt_checkpoint_restore_path` | `Optional[str]` | `None` | `--modelopt-checkpoint-restore-path` | Path to restore ModelOpt checkpoint. |
| `modelopt_checkpoint_save_path` | `Optional[str]` | `None` | `--modelopt-checkpoint-save-path` | Path to save ModelOpt checkpoint. |
| `modelopt_export_path` | `Optional[str]` | `None` | `--modelopt-export-path` | Path to export ModelOpt model. |
| `quantize_and_serve` | `bool` | `False` | `--quantize-and-serve` | Quantize the model on-the-fly and serve. |
| `rl_quant_profile` | `Optional[str]` | `None` | `--rl-quant-profile` | RL quantization profile for FlashRL. |

### Quantization Choices

The `quantization` parameter accepts the following values:

| Value | Description |
|-------|-------------|
| `None` (default) | No quantization; use model's native format |
| `awq` | Activation-aware Weight Quantization (4-bit) |
| `fp8` | FP8 (8-bit floating point) quantization |
| `mxfp8` | MXFP8 microscaling FP8 |
| `gptq` | GPTQ quantization (4-bit) |
| `marlin` | Marlin quantization |
| `gptq_marlin` | GPTQ-Marlin hybrid |
| `awq_marlin` | AWQ-Marlin hybrid |
| `bitsandbytes` | BitsAndBytes quantization |
| `gguf` | GGUF quantization |
| `modelopt` | NVIDIA ModelOpt quantization |
| `modelopt_fp8` | ModelOpt FP8 |
| `modelopt_fp4` | ModelOpt FP4 |
| `modelopt_mixed` | ModelOpt mixed precision |
| `petit_nvfp4` | Petit NVFP4 |
| `w8a8_int8` | Weight-only 8-bit INT8 |
| `w8a8_fp8` | Weight-and-activation FP8 |
| `moe_wna16` | MoE weight-only 16-bit |
| `qoq` | QOQ quantization |
| `w4afp8` | 4-bit weight with FP8 activation |
| `mxfp4` | MXFP4 microscaling |
| `auto-round` | AutoRound quantization |
| `compressed-tensors` | Compressed tensors (Ktransformers) |
| `modelslim` | ModelSlim quantization (NPU) |
| `quark` | AMD Quark quantizer (FP8/MXFP4/Int4FP8) |
| `quark_int4fp8_moe` | Quark INT4-FP8 for MoE |
| `unquant` | Explicitly disable quantization |

---

## Memory and Scheduling Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `mem_fraction_static` | `Optional[float]` | `None` | `--mem-fraction-static` | Fraction of GPU memory used for KV-cache. Auto-computed if None. |
| `max_running_requests` | `Optional[int]` | `None` | `--max-running-requests` | Maximum number of concurrently running requests. |
| `max_queued_requests` | `Optional[int]` | `None` | `--max-queued-requests` | Maximum number of queued requests. |
| `max_total_tokens` | `Optional[int]` | `None` | `--max-total-tokens` | Maximum total tokens across all requests (KV-cache size). |
| `chunked_prefill_size` | `Optional[int]` | `None` | `--chunked-prefill-size` | Number of tokens per prefill chunk. `-1` disables chunked prefill. |
| `enable_dynamic_chunking` | `bool` | `False` | `--enable-dynamic-chunking` | Enable dynamic chunk size adjustment. |
| `max_prefill_tokens` | `int` | `16384` | `--max-prefill-tokens` | Maximum tokens in a single prefill batch. |
| `prefill_max_requests` | `Optional[int]` | `None` | `--prefill-max-requests` | Maximum concurrent prefill requests. |
| `schedule_policy` | `str` | `"fcfs"` | `--schedule-policy` | Scheduling policy. Options: `"fcfs"` (first-come, first-served). |
| `enable_priority_scheduling` | `bool` | `False` | `--enable-priority-scheduling` | Enable priority-based request scheduling. |
| `disable_priority_preemption` | `bool` | `False` | `--disable-priority-preemption` | Disable preemption of lower-priority requests. |
| `default_priority_value` | `Optional[int]` | `None` | `--default-priority-value` | Default priority for requests without explicit priority. |
| `abort_on_priority_when_disabled` | `bool` | `False` | `--abort-on-priority-when-disabled` | Abort requests with priority when priority scheduling is disabled. |
| `schedule_low_priority_values_first` | `bool` | `False` | `--schedule-low-priority-values-first` | Schedule lower priority values first. |
| `priority_scheduling_preemption_threshold` | `int` | `10` | `--priority-scheduling-preemption-threshold` | Threshold for preemption decisions. |
| `schedule_conservativeness` | `float` | `1.0` | `--schedule-conservativeness` | How conservative the scheduler is with memory allocation (0.0-2.0). |
| `page_size` | `Optional[int]` | `None` | `--page-size` | KV-cache page size in tokens. Auto-selected based on hardware. |
| `swa_full_tokens_ratio` | `float` | `0.8` | `--swa-full-tokens-ratio` | Sliding window attention full tokens ratio. |
| `disable_hybrid_swa_memory` | `bool` | `False` | `--disable-hybrid-swa-memory` | Disable hybrid sliding window attention memory optimization. |
| `radix_eviction_policy` | `str` | `"lru"` | `--radix-eviction-policy` | KV-cache eviction policy. Options: `"lru"`, `"lfu"`, `"slru"`, `"priority"`. |

### Prefill Delayer Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_prefill_delayer` | `bool` | `False` | `--enable-prefill-delayer` | Enable prefill delayer for throughput optimization. |
| `prefill_delayer_max_delay_passes` | `int` | `30` | `--prefill-delayer-max-delay-passes` | Maximum delay passes before forcing prefill. |
| `prefill_delayer_token_usage_low_watermark` | `Optional[float]` | `None` | `--prefill-delayer-token-usage-low-watermark` | Token usage low watermark for prefill delayer. |
| `prefill_delayer_forward_passes_buckets` | `Optional[List[float]]` | `None` | `--prefill-delayer-forward-passes-buckets` | Forward passes bucket boundaries. |
| `prefill_delayer_wait_seconds_buckets` | `Optional[List[float]]` | `None` | `--prefill-delayer-wait-seconds-buckets` | Wait seconds bucket boundaries. |

Environment variable overrides for prefill delayer:
- `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1` enables prefill delayer
- `SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=<int>` sets max delay passes
- `SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK=<float>` sets low watermark

---

## Runtime Options

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `device` | `Optional[str]` | `None` | `--device` | Compute device. Options: `"cuda"`, `"cpu"`, `"hpu"`, `"npu"`, `"xpu"`, `"mps"`. Auto-detected if None. |
| `tp_size` | `int` | `1` | `--tp-size` | Tensor parallelism size (number of GPUs for TP). |
| `pp_size` | `int` | `1` | `--pp-size` | Pipeline parallelism size (number of pipeline stages). |
| `pp_max_micro_batch_size` | `Optional[int]` | `None` | `--pp-max-micro-batch-size` | Maximum micro-batch size for pipeline parallelism. |
| `pp_async_batch_depth` | `int` | `0` | `--pp-async-batch-depth` | Async batch depth for pipeline parallelism. |
| `stream_interval` | `int` | `1` | `--stream-interval` | Interval (in tokens) between streaming responses. |
| `batch_notify_size` | `int` | `16` | `--batch-notify-size` | Batch notification size for streaming. |
| `stream_response_default_include_usage` | `bool` | `False` | `--stream-response-default-include-usage` | Include usage stats in streaming responses by default. |
| `incremental_streaming_output` | `bool` | `False` | `--incremental-streaming-output` | Enable incremental streaming output (delta only). |
| `enable_streaming_session` | `bool` | `False` | `--enable-streaming-session` | Enable streaming session management. |
| `random_seed` | `Optional[int]` | `None` | `--random-seed` | Random seed for reproducibility. Auto-generated if None. |
| `constrained_json_whitespace_pattern` | `Optional[str]` | `None` | `--constrained-json-whitespace-pattern` | Whitespace pattern for constrained JSON generation. |
| `constrained_json_disable_any_whitespace` | `bool` | `False` | `--constrained-json-disable-any-whitespace` | Disable any whitespace in constrained JSON. |
| `watchdog_timeout` | `float` | `300` | `--watchdog-timeout` | Watchdog timeout in seconds. |
| `soft_watchdog_timeout` | `Optional[float]` | `None` | `--soft-watchdog-timeout` | Soft watchdog timeout in seconds. |
| `dist_timeout` | `Optional[int]` | `None` | `--dist-timeout` | Timeout for torch.distributed operations. |
| `download_dir` | `Optional[str]` | `None` | `--download-dir` | Directory for downloading model files. |
| `model_checksum` | `Optional[str]` | `None` | `--model-checksum` | Expected model checksum for verification. |
| `base_gpu_id` | `int` | `0` | `--base-gpu-id` | Starting GPU ID for tensor parallelism. |
| `gpu_id_step` | `int` | `1` | `--gpu-id-step` | Step between GPU IDs. |
| `sleep_on_idle` | `bool` | `False` | `--sleep-on-idle` | Put GPU to sleep when idle to save power. |
| `use_ray` | `bool` | `False` | `--use-ray` | Use Ray for distributed serving. |

---

## Logging Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `log_level` | `str` | `"info"` | `--log-level` | Logging level. Options: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`. |
| `log_level_http` | `Optional[str]` | `None` | `--log-level-http` | HTTP access log level. |
| `log_requests` | `bool` | `False` | `--log-requests` | Log all incoming requests. |
| `log_requests_level` | `int` | `2` | `--log-requests-level` | Request logging detail level. |
| `log_requests_format` | `str` | `"text"` | `--log-requests-format` | Request log format. Options: `"text"`, `"json"`. |
| `log_requests_target` | `Optional[List[str]]` | `None` | `--log-requests-target` | Filter request logging to specific endpoints. |
| `uvicorn_access_log_exclude_prefixes` | `List[str]` | `()` | `--uvicorn-access-log-exclude-prefixes` | URL prefixes to exclude from access logs. |
| `crash_dump_folder` | `Optional[str]` | `None` | `--crash-dump-folder` | Directory to save crash dumps. |
| `show_time_cost` | `bool` | `False` | `--show-time-cost` | Show time cost for each operation. |
| `enable_metrics` | `bool` | `False` | `--enable-metrics` | Enable Prometheus metrics endpoint. |
| `grpc_http_sidecar_port` | `Optional[int]` | `None` | `--grpc-http-sidecar-port` | Port for gRPC HTTP sidecar. |
| `enable_mfu_metrics` | `bool` | `False` | `--enable-mfu-metrics` | Enable Model FLOPs Utilization (MFU) metrics. |
| `enable_metrics_for_all_schedulers` | `bool` | `False` | `--enable-metrics-for-all-schedulers` | Enable metrics for all scheduler instances. |
| `tokenizer_metrics_custom_labels_header` | `str` | `"x-custom-labels"` | `--tokenizer-metrics-custom-labels-header` | HTTP header for custom metric labels. |
| `tokenizer_metrics_allowed_custom_labels` | `Optional[List[str]]` | `None` | `--tokenizer-metrics-allowed-custom-labels` | Allowed custom label names for metrics. |
| `extra_metric_labels` | `Optional[Dict[str, str]]` | `None` | `--extra-metric-labels` | Additional labels to add to all metrics. |
| `bucket_time_to_first_token` | `Optional[List[float]]` | `None` | `--bucket-time-to-first-token` | Histogram buckets for TTFT metric. |
| `bucket_inter_token_latency` | `Optional[List[float]]` | `None` | `--bucket-inter-token-latency` | Histogram buckets for ITL metric. |
| `bucket_e2e_request_latency` | `Optional[List[float]]` | `None` | `--bucket-e2e-request-latency` | Histogram buckets for E2E latency metric. |
| `prompt_tokens_buckets` | `Optional[List[str]]` | `None` | `--prompt-tokens-buckets` | Histogram buckets for prompt token count. |
| `generation_tokens_buckets` | `Optional[List[str]]` | `None` | `--generation-tokens-buckets` | Histogram buckets for generation token count. |
| `gc_warning_threshold_secs` | `float` | `0.0` | `--gc-warning-threshold-secs` | Garbage collection warning threshold. |
| `decode_log_interval` | `int` | `40` | `--decode-log-interval` | Interval between decode step log messages. |
| `enable_request_time_stats_logging` | `bool` | `False` | `--enable-request-time-stats-logging` | Enable request time statistics logging. |
| `kv_events_config` | `Optional[str]` | `None` | `--kv-events-config` | KV events configuration for observability. |
| `enable_trace` | `bool` | `False` | `--enable-trace` | Enable OpenTelemetry tracing. |
| `otlp_traces_endpoint` | `str` | `"localhost:4317"` | `--otlp-traces-endpoint` | OpenTelemetry traces endpoint. |
| `export_metrics_to_file` | `bool` | `False` | `--export-metrics-to-file` | Export metrics to a file. |
| `export_metrics_to_file_dir` | `Optional[str]` | `None` | `--export-metrics-to-file-dir` | Directory for exported metrics files. |

---

## API Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `api_key` | `Optional[str]` | `None` | `--api-key` | API key for client authentication. |
| `admin_api_key` | `Optional[str]` | `None` | `--admin-api-key` | Admin API key for management endpoints. |
| `served_model_name` | `Optional[str]` | `None` | `--served-model-name` | Name exposed in API responses. Defaults to `model_path`. |
| `weight_version` | `str` | `"default"` | `--weight-version` | Weight version identifier. |
| `chat_template` | `Optional[str]` | `None` | `--chat-template` | Path to custom chat template file. Auto-detected from tokenizer if None. |
| `hf_chat_template_name` | `Optional[str]` | `None` | `--hf-chat-template-name` | HuggingFace chat template name to use. |
| `completion_template` | `Optional[str]` | `None` | `--completion-template` | Path to custom completion template. |
| `file_storage_path` | `str` | `"sglang_storage"` | `--file-storage-path` | Directory for file uploads and storage. |
| `enable_cache_report` | `bool` | `False` | `--enable-cache-report` | Include cache hit information in responses. |
| `reasoning_parser` | `Optional[str]` | `None` | `--reasoning-parser` | Parser for reasoning/thinking content. Options: `"deepseek-r1"`, `"deepseek-v3"`, `"qwen3"`, `"qwen3-thinking"`, `"kimi"`, `"gpt-oss"`. |
| `strip_thinking_cache` | `bool` | `False` | `--strip-thinking-cache` | Strip thinking content from cached responses. |
| `tool_call_parser` | `Optional[str]` | `None` | `--tool-call-parser` | Parser for tool/function calls. Options: `"qwen"`, `"glm"`, `"llama3"`, etc. |
| `tool_server` | `Optional[str]` | `None` | `--tool-server` | Tool server URL for function calling. |
| `sampling_defaults` | `str` | `"model"` | `--sampling-defaults` | Default sampling parameter source. `"model"` uses generation_config.json, `"openai"` uses constant defaults. |
| `preferred_sampling_params` | `Optional[str]` | `None` | `--preferred-sampling-params` | Preferred sampling parameters for the model. |
| `json_model_override_args` | `str` | `"{}"` | `--json-model-override-args` | JSON string of model config overrides. |

---

## Data Parallelism Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `dp_size` | `int` | `1` | `--dp-size` | Data parallelism size (number of data-parallel replicas). |
| `load_balance_method` | `str` | `"auto"` | `--load-balance-method` | Load balancing method across DP workers. Options: `"auto"`, `"round_robin"`, `"follow_bootstrap_room"`, `"shortest_queue"`. Auto selects based on disaggregation mode. |
| `attn_cp_size` | `int` | `1` | `--attn-cp-size` | Attention context parallelism size. |
| `moe_dp_size` | `int` | `1` | `--moe-dp-size` | MoE data parallelism size. |

---

## Multi-Node Distributed Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `dist_init_addr` | `Optional[str]` | `None` | `--dist-init-addr` | Address of the master node for distributed initialization (e.g., `"host:port"`). |
| `nnodes` | `int` | `1` | `--nnodes` | Number of nodes in the distributed cluster. |
| `node_rank` | `int` | `0` | `--node-rank` | Rank of this node in the distributed cluster (0-indexed). |

### Multi-Node Example

```bash
# Node 0 (master)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp-size 8 --nnodes 2 --node-rank 0 \
    --dist-init-addr "MASTER_IP:20000"

# Node 1 (worker)
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp-size 8 --nnodes 2 --node-rank 1 \
    --dist-init-addr "MASTER_IP:20000"
```

---

## Tensor Parallelism Parameters

Tensor parallelism is configured primarily through the `tp_size` parameter in Runtime Options.
The following related parameters control GPU selection:

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `tp_size` | `int` | `1` | `--tp-size` | Number of GPUs for tensor parallelism. |
| `base_gpu_id` | `int` | `0` | `--base-gpu-id` | Starting GPU ID. |
| `gpu_id_step` | `int` | `1` | `--gpu-id-step` | Step between GPU IDs. |

---

## Expert Parallelism Parameters

For Mixture-of-Experts (MoE) models such as DeepSeek-V3, DeepSeek-R1, and Mixtral.

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `ep_size` | `int` | `1` | `--ep-size` | Expert parallelism size (number of EP workers). |
| `moe_a2a_backend` | `Literal` | `"none"` | `--moe-a2a-backend` | MoE all-to-all backend. Options: `"none"`, `"deepep"`, `"mooncake"`, `"nixl"`, `"mori"`, `"ascend_fuseep"`, `"flashinfer"`. |
| `moe_runner_backend` | `str` | `"auto"` | `--moe-runner-backend` | MoE runner backend. Options: `"auto"`, `"deep_gemm"`, `"triton"`, `"flashinfer_trtllm"`, `"cutlass"`, `"aiter"`, `"marlin"`, etc. |
| `record_nolora_graph` | `bool` | `True` | `--record-nolora-graph` | Record CUDA graph for non-LoRA path. |
| `flashinfer_mxfp4_moe_precision` | `Literal` | `"default"` | `--flashinfer-mxfp4-moe-precision` | MXFP4 MoE precision. Options: `"default"`, `"bf16"`. |
| `enable_flashinfer_allreduce_fusion` | `bool` | `False` | `--enable-flashinfer-allreduce-fusion` | Enable FlashInfer allreduce fusion for MoE. |
| `enforce_disable_flashinfer_allreduce_fusion` | `bool` | `False` | `--enforce-disable-flashinfer-allreduce-fusion` | Force disable FlashInfer allreduce fusion. |
| `enable_aiter_allreduce_fusion` | `bool` | `False` | `--enable-aiter-allreduce-fusion` | Enable AITER allreduce fusion (AMD). |
| `deepep_mode` | `Literal` | `"auto"` | `--deepep-mode` | DeepEP mode. Options: `"auto"`, `"normal"`, `"low_latency"`. |
| `ep_num_redundant_experts` | `int` | `0` | `--ep-num-redundant-experts` | Number of redundant experts for EP load balancing. |
| `ep_dispatch_algorithm` | `Optional[Literal]` | `None` | `--ep-dispatch-algorithm` | Expert dispatch algorithm. Options: `"static"`, `"dynamic"`, `"fake"`. |
| `init_expert_location` | `str` | `"trivial"` | `--init-expert-location` | Initial expert placement strategy. |
| `enable_eplb` | `bool` | `False` | `--enable-eplb` | Enable Expert Parallelism Load Balancing. |
| `eplb_algorithm` | `str` | `"auto"` | `--eplb-algorithm` | EPLB algorithm. |
| `eplb_rebalance_num_iterations` | `int` | `1000` | `--eplb-rebalance-num-iterations` | Iterations between EPLB rebalancing. |
| `eplb_rebalance_layers_per_chunk` | `Optional[int]` | `None` | `--eplb-rebalance-layers-per-chunk` | Layers per chunk for EPLB rebalancing. |
| `eplb_min_rebalancing_utilization_threshold` | `float` | `1.0` | `--eplb-min-rebalancing-utilization-threshold` | Minimum utilization threshold for EPLB. |
| `expert_distribution_recorder_mode` | `Optional[Literal]` | `None` | `--expert-distribution-recorder-mode` | Expert distribution recording mode. Options: `"stat"`, `"stat_approx"`, `"per_pass"`, `"per_token"`. |
| `expert_distribution_recorder_buffer_size` | `Optional[int]` | `None` | `--expert-distribution-recorder-buffer-size` | Buffer size for expert distribution recording. |
| `enable_expert_distribution_metrics` | `bool` | `False` | `--enable-expert-distribution-metrics` | Enable expert distribution metrics. |
| `deepep_config` | `Optional[str]` | `None` | `--deepep-config` | DeepEP configuration JSON. |
| `moe_dense_tp_size` | `Optional[int]` | `None` | `--moe-dense-tp-size` | TP size for dense MoE layers. |
| `elastic_ep_backend` | `Literal` | `None` | `--elastic-ep-backend` | Elastic EP backend. Options: `None`, `"mooncake"`, `"nixl"`. |
| `enable_elastic_expert_backup` | `bool` | `False` | `--enable-elastic-expert-backup` | Enable elastic expert backup. |
| `mooncake_ib_device` | `Optional[str]` | `None` | `--mooncake-ib-device` | Mooncake InfiniBand device. |
| `elastic_ep_rejoin` | `bool` | `False` | `--elastic-ep-rejoin` | Enable elastic EP rejoin. |

---

## Pipeline Parallelism Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `pp_size` | `int` | `1` | `--pp-size` | Pipeline parallelism size. |
| `pp_max_micro_batch_size` | `Optional[int]` | `None` | `--pp-max-micro-batch-size` | Maximum micro-batch size for PP. |
| `pp_async_batch_depth` | `int` | `0` | `--pp-async-batch-depth` | Async batch depth for PP. |

---

## LoRA Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_lora` | `Optional[bool]` | `None` | `--enable-lora` | Enable LoRA adapter support. Auto-detected if None. |
| `enable_lora_overlap_loading` | `Optional[bool]` | `None` | `--enable-lora-overlap-loading` | Enable overlapping LoRA weight loading with computation. |
| `max_lora_rank` | `Optional[int]` | `None` | `--max-lora-rank` | Maximum LoRA rank supported. |
| `lora_target_modules` | `Optional[Union[set, List]]` | `None` | `--lora-target-modules` | Target modules for LoRA. Use `"all"` for all linear layers. |
| `lora_paths` | `Optional[Union[dict, List]]` | `None` | `--lora-paths` | Named LoRA adapter paths. Format: `name=path name2=path2` or JSON. |
| `max_loaded_loras` | `Optional[int]` | `None` | `--max-loaded-loras` | Maximum number of LoRA adapters loaded simultaneously. |
| `max_loras_per_batch` | `int` | `8` | `--max-loras-per-batch` | Maximum LoRA adapters per batch. |
| `lora_eviction_policy` | `str` | `"lru"` | `--lora-eviction-policy` | LoRA cache eviction policy. Default: `"lru"`. |
| `lora_backend` | `str` | `"csgmv"` | `--lora-backend` | LoRA computation backend. Options: `"triton"`, `"csgmv"`, `"ascend"`, `"torch_native"`. |
| `max_lora_chunk_size` | `Optional[int]` | `16` | `--max-lora-chunk-size` | Maximum LoRA chunk size for batching. |
| `experts_shared_outer_loras` | `Optional[bool]` | `None` | `--experts-shared-outer-loras` | Share outer LoRA across experts (MoE). |
| `lora_use_virtual_experts` | `bool` | `False` | `--lora-use-virtual-experts` | Use virtual experts for LoRA in MoE models. |
| `lora_strict_loading` | `bool` | `False` | `--lora-strict-loading` | Strictly validate LoRA weight shapes on loading. |
| `lora_drain_wait_threshold` | `float` | `0.0` | `--lora-drain-wait-threshold` | Threshold for draining LoRA adapters. |

### LoRA Example

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths adapter_a=/path/to/adapter_a adapter_b=/path/to/adapter_b \
    --max-loras-per-batch 4
```

---

## Kernel Backend Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `attention_backend` | `Optional[str]` | `None` | `--attention-backend` | Attention kernel backend. Auto-selected if None. See Attention Backend Choices. |
| `decode_attention_backend` | `Optional[str]` | `None` | `--decode-attention-backend` | Separate attention backend for decode phase. |
| `prefill_attention_backend` | `Optional[str]` | `None` | `--prefill-attention-backend` | Separate attention backend for prefill phase. |
| `sampling_backend` | `Optional[str]` | `None` | `--sampling-backend` | Sampling kernel backend. Options: `"flashinfer"`, `"pytorch"`, `"ascend"`. |
| `grammar_backend` | `Optional[str]` | `None` | `--grammar-backend` | Structured output grammar backend. Options: `"xgrammar"`, `"outlines"`, `"llguidance"`, `"none"`. |
| `mm_attention_backend` | `Optional[str]` | `None` | `--mm-attention-backend` | Multi-modal attention backend. |
| `fp8_gemm_runner_backend` | `str` | `"auto"` | `--fp8-gemm-runner-backend` | FP8 GEMM runner backend. Options: `"auto"`, `"deep_gemm"`, `"flashinfer_trtllm"`, `"cutlass"`, etc. |
| `fp4_gemm_runner_backend` | `str` | `"auto"` | `--fp4-gemm-runner-backend` | FP4 GEMM runner backend. Options: `"auto"`, `"cutlass"`, `"flashinfer_cudnn"`, etc. |
| `nsa_prefill_backend` | `Optional[str]` | `None` | `--nsa-prefill-backend` | Native Sparse Attention prefill backend. Auto-detect if None. |
| `nsa_decode_backend` | `Optional[str]` | `None` | `--nsa-decode-backend` | Native Sparse Attention decode backend. Auto-detect if None. |
| `disable_flashinfer_autotune` | `bool` | `False` | `--disable-flashinfer-autotune` | Disable FlashInfer kernel autotuning. |
| `mamba_backend` | `str` | `"triton"` | `--mamba-backend` | Mamba/SSM backend. Options: `"triton"`, `"flashinfer"`. |

### Attention Backend Choices

| Backend | Platform | Description |
|---------|----------|-------------|
| `flashinfer` | NVIDIA | FlashInfer attention (default for sm80+) |
| `fa3` | NVIDIA | FlashAttention 3 |
| `fa4` | NVIDIA | FlashAttention 4 |
| `flashmla` | NVIDIA | FlashMLA for Multi-head Latent Attention |
| `cutlass_mla` | NVIDIA | CUTLASS MLA |
| `trtllm_mla` | NVIDIA | TensorRT-LLM MLA |
| `trtllm_mha` | NVIDIA | TensorRT-LLM MHA |
| `dual_chunk_flash_attn` | NVIDIA | Dual-chunk flash attention |
| `nsa` | NVIDIA | Native Sparse Attention |
| `triton` | All | Triton attention kernel (fallback) |
| `torch_native` | All | PyTorch native attention (slow, fallback) |
| `flex_attention` | All | PyTorch flex_attention |
| `aiter` | AMD | AITER attention (AMD) |
| `wave` | AMD | Wave attention (AMD) |
| `intel_amx` | Intel | AMX attention (CPU) |
| `ascend` | Ascend | Ascend NPU attention |
| `intel_xpu` | Intel | Intel XPU attention |

### Grammar Backend Choices

| Backend | Description | Supported Constraints |
|---------|-------------|----------------------|
| `xgrammar` (default) | XGrammar engine | JSON schema, regex, EBNF |
| `outlines` | Outlines engine | JSON schema, regex |
| `llguidance` | LLGuidance engine | JSON schema, regex, EBNF |
| `none` | Disable grammar backend | None |

---

## Speculative Decoding Parameters

### General Speculative Decoding

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `speculative_algorithm` | `Optional[str]` | `None` | `--speculative-algorithm` | Speculative decoding algorithm. Options: `"eagle"`, `"medusa"`, `"ngram"`, `"eagle3"`. |
| `speculative_draft_model_path` | `Optional[str]` | `None` | `--speculative-draft-model-path` | Path to draft model for speculative decoding. |
| `speculative_draft_model_revision` | `Optional[str]` | `None` | `--speculative-draft-model-revision` | Revision of the draft model. |
| `speculative_draft_load_format` | `Optional[str]` | `None` | `--speculative-draft-load-format` | Load format for the draft model. |
| `speculative_num_steps` | `Optional[int]` | `None` | `--speculative-num-steps` | Number of speculative steps per iteration. |
| `speculative_eagle_topk` | `Optional[int]` | `None` | `--speculative-eagle-topk` | Top-K for EAGLE speculative decoding. |
| `speculative_num_draft_tokens` | `Optional[int]` | `None` | `--speculative-num-draft-tokens` | Number of draft tokens to generate. |
| `speculative_dflash_block_size` | `Optional[int]` | `None` | `--speculative-dflash-block-size` | Block size for DFlash attention in speculation. |
| `speculative_dflash_draft_window_size` | `Optional[int]` | `None` | `--speculative-dflash-draft-window-size` | Draft window size for DFlash. |
| `speculative_accept_threshold_single` | `float` | `1.0` | `--speculative-accept-threshold-single` | Acceptance threshold for single-token verification. |
| `speculative_accept_threshold_acc` | `float` | `1.0` | `--speculative-accept-threshold-acc` | Acceptance threshold for accumulated verification. |
| `speculative_token_map` | `Optional[str]` | `None` | `--speculative-token-map` | Token map for speculative decoding. |
| `speculative_attention_mode` | `str` | `"prefill"` | `--speculative-attention-mode` | Attention mode for speculation. Options: `"prefill"`, `"decode"`. |
| `speculative_draft_attention_backend` | `Optional[str]` | `None` | `--speculative-draft-attention-backend` | Attention backend for draft model. |
| `speculative_moe_runner_backend` | `Optional[str]` | `None` | `--speculative-moe-runner-backend` | MoE runner backend for draft model. |
| `speculative_moe_a2a_backend` | `Optional[str]` | `None` | `--speculative-moe-a2a-backend` | MoE all-to-all backend for draft model. |
| `speculative_draft_model_quantization` | `Optional[str]` | `None` | `--speculative-draft-model-quantization` | Quantization for draft model. Same choices as `quantization`. |
| `speculative_adaptive` | `bool` | `False` | `--speculative-adaptive` | Enable adaptive speculative decoding. |
| `speculative_adaptive_config` | `Optional[str]` | `None` | `--speculative-adaptive-config` | Config for adaptive speculative decoding. |
| `speculative_skip_dp_mlp_sync` | `bool` | `False` | `--speculative-skip-dp-mlp-sync` | Skip DP MLP sync during speculation. |
| `enable_multi_layer_eagle` | `bool` | `False` | `--enable-multi-layer-eagle` | Enable multi-layer EAGLE speculative decoding. |

---

## N-gram Speculative Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `speculative_ngram_min_bfs_breadth` | `int` | `1` | `--speculative-ngram-min-bfs-breadth` | Minimum BFS breadth for n-gram speculation. |
| `speculative_ngram_max_bfs_breadth` | `int` | `10` | `--speculative-ngram-max-bfs-breadth` | Maximum BFS breadth for n-gram speculation. |
| `speculative_ngram_match_type` | `Literal` | `"BFS"` | `--speculative-ngram-match-type` | N-gram match type. Options: `"BFS"`, `"PROB"`. |
| `speculative_ngram_max_trie_depth` | `int` | `18` | `--speculative-ngram-max-trie-depth` | Maximum trie depth for n-gram speculation. |
| `speculative_ngram_capacity` | `int` | `10000000` | `--speculative-ngram-capacity` | Capacity of n-gram trie. |
| `speculative_ngram_external_corpus_path` | `Optional[str]` | `None` | `--speculative-ngram-external-corpus-path` | Path to external corpus for n-gram speculation. |
| `speculative_ngram_external_sam_budget` | `int` | `0` | `--speculative-ngram-external-sam-budget` | SAM budget for external corpus. |
| `speculative_ngram_external_corpus_max_tokens` | `int` | `10000000` | `--speculative-ngram-external-corpus-max-tokens` | Maximum tokens from external corpus. |

---

## Mamba Cache Parameters

For SSM/Mamba models (e.g., Mamba, Jamba).

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `max_mamba_cache_size` | `Optional[int]` | `None` | `--max-mamba-cache-size` | Maximum Mamba cache size. |
| `mamba_ssm_dtype` | `Optional[str]` | `None` | `--mamba-ssm-dtype` | Data type for Mamba SSM states. |
| `mamba_full_memory_ratio` | `float` | `0.9` | `--mamba-full-memory-ratio` | Full memory ratio for Mamba cache. |
| `mamba_scheduler_strategy` | `str` | `"auto"` | `--mamba-scheduler-strategy` | Mamba scheduler strategy. Options: `"auto"`, `"no_buffer"`, `"extra_buffer"`. |
| `mamba_track_interval` | `int` | `256` | `--mamba-track-interval` | Track interval for Mamba cache management. |
| `linear_attn_backend` | `str` | `"triton"` | `--linear-attn-backend` | Linear attention backend. Options: `"triton"`, `"cutedsl"`, `"flashinfer"`. |
| `linear_attn_decode_backend` | `Optional[str]` | `None` | `--linear-attn-decode-backend` | Linear attention decode backend. |
| `linear_attn_prefill_backend` | `Optional[str]` | `None` | `--linear-attn-prefill-backend` | Linear attention prefill backend. |

---

## Hierarchical Cache Parameters

HiCache enables offloading KV-cache entries to host (CPU) memory for longer context support.

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_hierarchical_cache` | `bool` | `False` | `--enable-hierarchical-cache` | Enable hierarchical cache (HiCache). |
| `hicache_ratio` | `float` | `2.0` | `--hicache-ratio` | HiCache size ratio relative to GPU KV-cache. |
| `hicache_size` | `int` | `0` | `--hicache-size` | HiCache size in bytes. Overrides `hicache_ratio` if non-zero. |
| `hicache_write_policy` | `str` | `"write_through"` | `--hicache-write-policy` | Write policy. Options: `"write_through"`, `"write_back"`. |
| `hicache_io_backend` | `str` | `"kernel"` | `--hicache-io-backend` | IO backend. Options: `"kernel"`, `"direct"`. |
| `hicache_mem_layout` | `str` | `"layer_first"` | `--hicache-mem-layout` | Memory layout. Options: `"layer_first"`, `"token_first"`. |
| `hicache_storage_backend` | `Optional[str]` | `None` | `--hicache-storage-backend` | Storage backend for persistent HiCache. |
| `hicache_storage_prefetch_policy` | `str` | `"best_effort"` | `--hicache-storage-prefetch-policy` | Prefetch policy. Options: `"best_effort"`, `"aggressive"`. |
| `hicache_storage_backend_extra_config` | `Optional[str]` | `None` | `--hicache-storage-backend-extra-config` | Extra config for storage backend. |

### Additional Cache Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_hisparse` | `bool` | `False` | `--enable-hisparse` | Enable hierarchical sparse attention. |
| `hisparse_config` | `Optional[str]` | `None` | `--hisparse-config` | HiSparse configuration JSON. |
| `enable_lmcache` | `bool` | `False` | `--enable-lmcache` | Enable LMCache integration. |

---

## CUDA Graph Parameters

CUDA graphs capture GPU operations for efficient replay, significantly reducing kernel launch
overhead during decode.

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `cuda_graph_max_bs` | `Optional[int]` | `None` | `--cuda-graph-max-bs` | Maximum batch size for CUDA graph capture. |
| `cuda_graph_bs` | `Optional[List[int]]` | `None` | `--cuda-graph-bs` | Custom batch sizes for CUDA graph capture. |
| `disable_cuda_graph` | `bool` | `False` | `--disable-cuda-graph` | Disable CUDA graph optimization entirely. |
| `disable_cuda_graph_padding` | `bool` | `False` | `--disable-cuda-graph-padding` | Disable padding in CUDA graphs. |
| `enable_breakable_cuda_graph` | `bool` | `False` | `--enable-breakable-cuda-graph` | Enable breakable CUDA graphs. |
| `enable_profile_cuda_graph` | `bool` | `False` | `--enable-profile-cuda-graph` | Profile CUDA graph capture. |
| `enable_cudagraph_gc` | `bool` | `False` | `--enable-cudagraph-gc` | Enable garbage collection for CUDA graphs. |
| `debug_cuda_graph` | `bool` | `False` | `--debug-cuda-graph` | Debug CUDA graph capture issues. |
| `disable_piecewise_cuda_graph` | `bool` | `False` | `--disable-piecewise-cuda-graph` | Disable piecewise CUDA graph capture. |
| `enforce_piecewise_cuda_graph` | `bool` | `False` | `--enforce-piecewise-cuda-graph` | Force piecewise CUDA graph capture. |
| `piecewise_cuda_graph_max_tokens` | `Optional[int]` | `None` | `--piecewise-cuda-graph-max-tokens` | Max tokens for piecewise CUDA graph. |
| `piecewise_cuda_graph_tokens` | `Optional[List[int]]` | `None` | `--piecewise-cuda-graph-tokens` | Custom token counts for piecewise CUDA graph. |
| `piecewise_cuda_graph_compiler` | `str` | `"eager"` | `--piecewise-cuda-graph-compiler` | Compiler for piecewise CUDA graphs. |

---

## Disaggregation Parameters

Prefill-decode (PD) disaggregation separates the prefill and decode phases onto different GPU
clusters for independent scaling and optimization.

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `disaggregation_mode` | `Literal` | `"null"` | `--disaggregation-mode` | Disaggregation mode. Options: `"null"` (not disaggregated), `"prefill"` (prefill-only), `"decode"` (decode-only). |
| `disaggregation_transfer_backend` | `str` | `"mooncake"` | `--disaggregation-transfer-backend` | Transfer backend for PD disaggregation. Options: `"mooncake"`, `"nixl"`, `"ascend"`, `"fake"`, `"mori"`. |
| `disaggregation_bootstrap_port` | `int` | `8998` | `--disaggregation-bootstrap-port` | Bootstrap port for PD coordination. |
| `disaggregation_ib_device` | `Optional[str]` | `None` | `--disaggregation-ib-device` | InfiniBand device for PD transfer. |
| `disaggregation_decode_enable_radix_cache` | `bool` | `False` | `--disaggregation-decode-enable-radix-cache` | Enable radix cache on decode workers. |
| `disaggregation_decode_enable_offload_kvcache` | `bool` | `False` | `--disaggregation-decode-enable-offload-kvcache` | Enable KV-cache offloading on decode workers. |
| `num_reserved_decode_tokens` | `int` | `512` | `--num-reserved-decode-tokens` | Reserved tokens for decode KV-cache offload. |
| `disaggregation_decode_polling_interval` | `int` | `1` | `--disaggregation-decode-polling-interval` | Polling interval for decode workers (ms). |

---

## Encoder Disaggregation Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `encoder_only` | `bool` | `False` | `--encoder-only` | Run in encoder-only mode (for disaggregated serving). |
| `language_only` | `bool` | `False` | `--language-only` | Run in language-only mode (no multimodal). |
| `encoder_transfer_backend` | `str` | `"zmq_to_scheduler"` | `--encoder-transfer-backend` | Backend for encoder transfer. Options: `"zmq_to_scheduler"`, `"zmq_to_tokenizer"`, `"mooncake"`. |
| `encoder_urls` | `List[str]` | `[]` | `--encoder-urls` | URLs for remote encoder workers. |
| `enable_adaptive_dispatch_to_encoder` | `bool` | `False` | `--enable-adaptive-dispatch-to-encoder` | Adaptive dispatch to encoder workers. |

---

## Multi-Modal Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_broadcast_mm_inputs_process` | `bool` | `False` | `--enable-broadcast-mm-inputs-process` | Broadcast multimodal inputs across workers. |
| `enable_prefix_mm_cache` | `bool` | `False` | `--enable-prefix-mm-cache` | Enable prefix caching for multimodal inputs. |
| `mm_enable_dp_encoder` | `bool` | `False` | `--mm-enable-dp-encoder` | Enable data parallelism for multimodal encoder. |
| `mm_process_config` | `Optional[Dict]` | `None` | `--mm-process-config` | Multimodal processor configuration. Must be a dict with optional `image`, `video`, `audio` sub-dicts. |
| `limit_mm_data_per_request` | `Optional[Union[str, Dict]]` | `None` | `--limit-mm-data-per-request` | Limit multimodal data items per request. |
| `enable_mm_global_cache` | `bool` | `False` | `--enable-mm-global-cache` | Enable global multimodal feature cache. |
| `disable_fast_image_processor` | `bool` | `False` | `--disable-fast-image-processor` | Disable fast image processor. |
| `keep_mm_feature_on_device` | `bool` | `False` | `--keep-mm-feature-on-device` | Keep multimodal features on device (GPU). |

---

## Offloading Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `cpu_offload_gb` | `int` | `0` | `--cpu-offload-gb` | Amount of CPU memory (GB) to use for weight offloading. 0 disables. |
| `offload_group_size` | `int` | `-1` | `--offload-group-size` | Group size for offloading. `-1` for auto. |
| `offload_num_in_group` | `int` | `1` | `--offload-num-in-group` | Number of items per offload group. |
| `offload_prefetch_step` | `int` | `1` | `--offload-prefetch-step` | Prefetch step for offloading. |
| `offload_mode` | `str` | `"cpu"` | `--offload-mode` | Offload destination. Options: `"cpu"`. |

---

## Debug and Optimization Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `disable_radix_cache` | `bool` | `False` | `--disable-radix-cache` | Disable RadixAttention prefix caching. |
| `enable_layerwise_nvtx_marker` | `bool` | `False` | `--enable-layerwise-nvtx-marker` | Enable NVTX markers per layer (for NVIDIA Nsight profiling). |
| `enable_nccl_nvls` | `bool` | `False` | `--enable-nccl-nvls` | Enable NCCL NVLS (NVLink Sharp). |
| `enable_symm_mem` | `bool` | `False` | `--enable-symm-mem` | Enable symmetric memory. |
| `disable_flashinfer_cutlass_moe_fp4_allgather` | `bool` | `False` | `--disable-flashinfer-cutlass-moe-fp4-allgather` | Disable FlashInfer CUTLASS MoE FP4 allgather. |
| `enable_tokenizer_batch_encode` | `bool` | `False` | `--enable-tokenizer-batch-encode` | Enable batch tokenization. |
| `disable_tokenizer_batch_decode` | `bool` | `False` | `--disable-tokenizer-batch-decode` | Disable batch detokenization. |
| `disable_outlines_disk_cache` | `bool` | `False` | `--disable-outlines-disk-cache` | Disable Outlines disk cache for grammars. |
| `disable_custom_all_reduce` | `bool` | `False` | `--disable-custom-all-reduce` | Disable custom all-reduce (use NCCL). |
| `enable_mscclpp` | `bool` | `False` | `--enable-mscclpp` | Enable MSCCL++ custom all-reduce. |
| `enable_torch_symm_mem` | `bool` | `False` | `--enable-torch-symm-mem` | Enable torch symmetric memory. |
| `pre_warm_nccl` | `bool` | `auto` | `--pre-warm-nccl` | Pre-warm NCCL/RCCL to reduce P99 TTFT. Default: True for AMD/HIP, False otherwise. |
| `disable_overlap_schedule` | `bool` | `False` | `--disable-overlap-schedule` | Disable overlap scheduling. |
| `enable_mixed_chunk` | `bool` | `False` | `--enable-mixed-chunk` | Enable mixed chunk prefill/decode. |
| `enable_dp_attention` | `bool` | `False` | `--enable-dp-attention` | Enable data-parallel attention. |
| `enable_dp_attention_local_control_broadcast` | `bool` | `False` | `--enable-dp-attention-local-control-broadcast` | Enable DP attention local control broadcast. |
| `enable_dp_lm_head` | `bool` | `False` | `--enable-dp-lm-head` | Enable data-parallel LM head. |
| `enable_two_batch_overlap` | `bool` | `False` | `--enable-two-batch-overlap` | Enable two-batch overlap scheduling. |
| `enable_single_batch_overlap` | `bool` | `False` | `--enable-single-batch-overlap` | Enable single-batch overlap scheduling. |
| `tbo_token_distribution_threshold` | `float` | `0.48` | `--tbo-token-distribution-threshold` | Token distribution threshold for TBO. |
| `enable_torch_compile` | `bool` | `False` | `--enable-torch-compile` | Enable torch.compile optimization. |
| `enable_torch_compile_debug_mode` | `bool` | `False` | `--enable-torch-compile-debug-mode` | Debug mode for torch.compile. |
| `torch_compile_max_bs` | `int` | `32` | `--torch-compile-max-bs` | Maximum batch size for torch.compile. |
| `torchao_config` | `str` | `""` | `--torchao-config` | TorchAO configuration string. |
| `enable_nan_detection` | `bool` | `False` | `--enable-nan-detection` | (Deprecated) Enable NaN detection. Use env vars instead. |
| `enable_p2p_check` | `bool` | `False` | `--enable-p2p-check` | Enable P2P access check between GPUs. |
| `triton_attention_reduce_in_fp32` | `bool` | `False` | `--triton-attention-reduce-in-fp32` | Use FP32 for attention reduction in Triton. |
| `triton_attention_num_kv_splits` | `int` | `8` | `--triton-attention-num-kv-splits` | Number of KV splits for Triton attention. |
| `triton_attention_split_tile_size` | `Optional[int]` | `None` | `--triton-attention-split-tile-size` | Split tile size for Triton attention. |
| `num_continuous_decode_steps` | `int` | `1` | `--num-continuous-decode-steps` | Number of continuous decode steps per batch. |
| `delete_ckpt_after_loading` | `bool` | `False` | `--delete-ckpt-after-loading` | Delete checkpoint files after loading. |
| `enable_memory_saver` | `bool` | `False` | `--enable-memory-saver` | Enable memory saver mode. |
| `enable_weights_cpu_backup` | `bool` | `False` | `--enable-weights-cpu-backup` | Backup model weights to CPU memory. |
| `enable_draft_weights_cpu_backup` | `bool` | `False` | `--enable-draft-weights-cpu-backup` | Backup draft model weights to CPU. |
| `allow_auto_truncate` | `bool` | `False` | `--allow-auto-truncate` | Automatically truncate long prompts. |
| `enable_custom_logit_processor` | `bool` | `False` | `--enable-custom-logit-processor` | Enable custom logit processor support. |
| `flashinfer_mla_disable_ragged` | `bool` | `False` | `--flashinfer-mla-disable-ragged` | Disable ragged tensors in FlashInfer MLA. |
| `disable_shared_experts_fusion` | `bool` | `False` | `--disable-shared-experts-fusion` | Disable shared experts fusion (MoE). |
| `enforce_shared_experts_fusion` | `bool` | `False` | `--enforce-shared-experts-fusion` | Force shared experts fusion (MoE). |
| `disable_chunked_prefix_cache` | `bool` | `False` | `--disable-chunked-prefix-cache` | Disable chunked prefix caching. |
| `enable_return_hidden_states` | `bool` | `False` | `--enable-return-hidden-states` | Return hidden states in responses. |
| `enable_return_routed_experts` | `bool` | `False` | `--enable-return-routed-experts` | Return expert routing data (MoE). |
| `enable_return_indexer_topk` | `bool` | `False` | `--enable-return-indexer-topk` | Return indexer top-K data. |
| `scheduler_recv_interval` | `int` | `1` | `--scheduler-recv-interval` | Scheduler receive interval. |
| `numa_node` | `Optional[List[int]]` | `None` | `--numa-node` | NUMA node affinity. |
| `enable_deterministic_inference` | `bool` | `False` | `--enable-deterministic-inference` | Enable deterministic inference mode. |
| `rl_on_policy_target` | `Optional[str]` | `None` | `--rl-on-policy-target` | RL on-policy target. Options: `"fsdp"`. |
| `enable_attn_tp_input_scattered` | `bool` | `False` | `--enable-attn-tp-input-scattered` | Enable scattered input for attention TP. |
| `gc_threshold` | `Optional[List[int]]` | `None` | `--gc-threshold` | Garbage collection threshold values. |
| `enable_fused_qk_norm_rope` | `bool` | `False` | `--enable-fused-qk-norm-rope` | Enable fused QK norm + RoPE. |
| `enable_precise_embedding_interpolation` | `bool` | `False` | `--enable-precise-embedding-interpolation` | Enable precise embedding interpolation. |
| `enable_fused_moe_sum_all_reduce` | `bool` | `False` | `--enable-fused-moe-sum-all-reduce` | Enable fused MoE sum all-reduce. |
| `enable_nsa_prefill_context_parallel` | `bool` | `False` | `--enable-nsa-prefill-context-parallel` | Enable NSA prefill context parallelism. |
| `nsa_prefill_cp_mode` | `str` | `"round-robin-split"` | `--nsa-prefill-cp-mode` | NSA prefill CP mode. Options: `"in-seq-split"`, `"round-robin-split"`. |

---

## Context Parallelism Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_prefill_context_parallel` | `bool` | `False` | `--enable-prefill-context-parallel` | Enable context parallelism during prefill. |
| `prefill_cp_mode` | `str` | `"in-seq-split"` | `--prefill-cp-mode` | Context parallelism mode. Options: `"in-seq-split"`. |

---

## Dynamic Batch Tokenizer Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_dynamic_batch_tokenizer` | `bool` | `False` | `--enable-dynamic-batch-tokenizer` | Enable dynamic batching for tokenization. |
| `dynamic_batch_tokenizer_batch_size` | `int` | `32` | `--dynamic-batch-tokenizer-batch-size` | Batch size for dynamic batch tokenizer. |
| `dynamic_batch_tokenizer_batch_timeout` | `float` | `0.002` | `--dynamic-batch-tokenizer-batch-timeout` | Timeout (seconds) for dynamic batch tokenizer. |

---

## Debug Tensor Dump Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `debug_tensor_dump_output_folder` | `Optional[str]` | `None` | `--debug-tensor-dump-output-folder` | Output folder for tensor dumps. |
| `debug_tensor_dump_layers` | `Optional[List[int]]` | `None` | `--debug-tensor-dump-layers` | Specific layers to dump (None = all layers). |
| `debug_tensor_dump_input_file` | `Optional[str]` | `None` | `--debug-tensor-dump-input-file` | Input file for tensor injection. |
| `debug_tensor_dump_inject` | `bool` | `False` | `--debug-tensor-dump-inject` | Inject dumped tensors for replay debugging. |

---

## Weight Loading Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `custom_weight_loader` | `Optional[List[str]]` | `None` | `--custom-weight-loader` | Custom weight loader module paths. |
| `weight_loader_disable_mmap` | `bool` | `False` | `--weight-loader-disable-mmap` | Disable memory-mapped file loading. |
| `weight_loader_prefetch_checkpoints` | `bool` | `False` | `--weight-loader-prefetch-checkpoints` | Enable checkpoint prefetching. |
| `weight_loader_prefetch_num_threads` | `int` | `4` | `--weight-loader-prefetch-num-threads` | Number of threads for prefetching. |
| `remote_instance_weight_loader_seed_instance_ip` | `Optional[str]` | `None` | `--remote-instance-weight-loader-seed-instance-ip` | IP of seed instance for remote loading. |
| `remote_instance_weight_loader_seed_instance_service_port` | `Optional[int]` | `None` | `--remote-instance-weight-loader-seed-instance-service-port` | Service port of seed instance. |
| `remote_instance_weight_loader_send_weights_group_ports` | `Optional[List[int]]` | `None` | `--remote-instance-weight-loader-send-weights-group-ports` | Ports for weight transfer group. |
| `remote_instance_weight_loader_backend` | `Literal` | `"nccl"` | `--remote-instance-weight-loader-backend` | Backend for remote weight loading. Options: `"transfer_engine"`, `"nccl"`, `"modelexpress"`. |
| `remote_instance_weight_loader_start_seed_via_transfer_engine` | `bool` | `False` | `--remote-instance-weight-loader-start-seed-via-transfer-engine` | Start seed via transfer engine. |
| `engine_info_bootstrap_port` | `int` | `6789` | `--engine-info-bootstrap-port` | Bootstrap port for engine info. |
| `modelexpress_config` | `Optional[str]` | `None` | `--modelexpress-config` | ModelExpress configuration. |

---

## Scoring Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_mis` | `bool` | `False` | `--enable-mis` | Enable Multi-Item Scoring optimization. Combines query and items into single sequence. |

---

## Diffusion LLM Parameters

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `dllm_algorithm` | `Optional[str]` | `None` | `--dllm-algorithm` | Diffusion LLM algorithm. |
| `dllm_algorithm_config` | `Optional[str]` | `None` | `--dllm-algorithm-config` | Diffusion LLM algorithm configuration. |

---

## Additional Parameters

### PD-Multiplexing

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_pdmux` | `bool` | `False` | `--enable-pdmux` | Enable PD-Multiplexing. |
| `pdmux_config_path` | `Optional[str]` | `None` | `--pdmux-config-path` | Path to PDMux configuration file. |
| `sm_group_num` | `int` | `8` | `--sm-group-num` | SM group number for PDMux. |

### Checkpoint Decryption

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `decrypted_config_file` | `Optional[str]` | `None` | `--decrypted-config-file` | Decrypted model config file path. |
| `decrypted_draft_config_file` | `Optional[str]` | `None` | `--decrypted-draft-config-file` | Decrypted draft model config file path. |

### Forward Hooks

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `forward_hooks` | `Optional[List[dict]]` | `None` | `--forward-hooks` | Forward hook configurations. |

### Communications Compression

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `enable_quant_communications` | `Optional[bool]` | `False` | `--enable-quant-communications` | Enable quantized communications. |

### msProbe

| Parameter | Type | Default | CLI Flag | Description |
|-----------|------|---------|----------|-------------|
| `msprobe_dump_config` | `Optional[str]` | `None` | `--msprobe-dump-config` | msProbe dump configuration. |

---

## Validation Rules and Cross-Dependencies

The `__post_init__` method in `ServerArgs` performs extensive validation. Below are the key
cross-dependency rules:

### SSL Validation
- `ssl_keyfile` requires `ssl_certfile` and vice versa
- `ssl_ca_certs` requires both `ssl_keyfile` and `ssl_certfile`
- `ssl_keyfile_password` requires both `ssl_keyfile` and `ssl_certfile`
- `enable_ssl_refresh` requires both `ssl_keyfile` and `ssl_certfile`
- `enable_http2` requires `granian` package installed
- `enable_http2` is incompatible with `enable_ssl_refresh`
- `enable_http2` does not support `tokenizer_worker_num > 1`

### Load Balance Defaults
- `load_balance_method="auto"` resolves to:
  - `"follow_bootstrap_room"` when `disaggregation_mode == "prefill"`
  - `"round_robin"` for all other modes

### Default Value Resolution
- `tokenizer_path` defaults to `model_path`
- `served_model_name` defaults to `model_path`
- `device` is auto-detected based on available hardware
- `random_seed` is auto-generated if not specified
- `mm_process_config` defaults to `{}`

### Disaggregation Validation
- `disaggregation_mode` must be `"null"`, `"prefill"`, or `"decode"`
- PD disaggregation requires coordinated prefill and decode clusters

### Deprecated Arguments
- `tool_call_parser="qwen25"` is deprecated; use `"qwen"` instead
- `tool_call_parser="glm45"` is deprecated; use `"glm"` instead
- `enable_nan_detection` is deprecated; use `SGLANG_SPEC_NAN_DETECTION=1` and `SGLANG_SPEC_OOB_DETECTION=1` environment variables

---

## CLI Flag Mapping

CLI flags use a `--kebab-case` convention that maps to `snake_case` Python attribute names.

| Python Attribute | CLI Flag |
|-----------------|----------|
| `model_path` | `--model-path` |
| `tokenizer_path` | `--tokenizer-path` |
| `tp_size` | `--tp-size` |
| `pp_size` | `--pp-size` |
| `dp_size` | `--dp-size` |
| `ep_size` | `--ep-size` |
| `mem_fraction_static` | `--mem-fraction-static` |
| `chunked_prefill_size` | `--chunked-prefill-size` |
| `enable_lora` | `--enable-lora` |
| `lora_paths` | `--lora-paths` |
| `quantization` | `--quantization` |
| `kv_cache_dtype` | `--kv-cache-dtype` |
| `attention_backend` | `--attention-backend` |
| `sampling_backend` | `--sampling-backend` |
| `grammar_backend` | `--grammar-backend` |
| `host` | `--host` |
| `port` | `--port` |
| `log_level` | `--log-level` |
| `api_key` | `--api-key` |
| `chat_template` | `--chat-template` |
| `reasoning_parser` | `--reasoning-parser` |
| `context_length` | `--context-length` |
| `disaggregation_mode` | `--disaggregation-mode` |
| `speculative_algorithm` | `--speculative-algorithm` |

### Environment Variable Overrides

| Environment Variable | Affects |
|---------------------|---------|
| `SGLANG_USE_MODELSCOPE` | Use ModelScope for model downloads |
| `SGLANG_ENABLE_GRPC` | Enable gRPC server |
| `SGLANG_GRPC_PORT` | gRPC server port |
| `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE` | Enable prefill delayer |
| `SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES` | Prefill delayer max passes |
| `SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK` | Prefill delayer watermark |
| `SGLANG_SPEC_NAN_DETECTION` | Enable NaN detection in speculative |
| `SGLANG_SPEC_OOB_DETECTION` | Enable OOB detection in speculative |
| `CUDA_HOME` | CUDA installation path |
| `HF_TOKEN` | HuggingFace authentication token |
| `HF_ENDPOINT` | HuggingFace endpoint (for mirrors) |
| `TRITON_PTXAS_PATH` | Path to ptxas binary (Blackwell) |

---

## Related Documentation

- [Overview and Architecture](./01-overview-architecture.md)
- [Installation and Setup](./02-installation-setup.md)
- [API Reference](./04-api-reference.md)
