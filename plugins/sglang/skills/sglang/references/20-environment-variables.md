# SGLang Environment Variables Reference

This document provides a comprehensive reference for all environment variables used by SGLang. These variables control every aspect of the SGLang runtime, from model loading and memory management to scheduling, profiling, quantization, and hardware-specific behavior.

---

## Table of Contents

1. [Environment Variable System Overview](#environment-variable-system-overview)
2. [Model and File Download](#model-and-file-download)
3. [Logging Options](#logging-options)
4. [Scheduler Configuration](#scheduler-configuration)
5. [Scheduler New Token Ratio](#scheduler-new-token-ratio)
6. [Scheduler Receive Interval](#scheduler-receive-interval)
7. [Scheduler Memory Management](#scheduler-memory-management)
8. [Scheduler Other Settings](#scheduler-other-settings)
9. [PD Disaggregation (Runtime)](#pd-disaggregation-runtime)
10. [Model Parallel](#model-parallel)
11. [Quantization](#quantization)
12. [FlashInfer](#flashinfer)
13. [Triton](#triton)
14. [Torch Compile](#torch-compile)
15. [DeepGemm](#deepgemm)
16. [DeepEP](#deepep)
17. [NIXL-EP](#nixl-ep)
18. [EPLB (Expert Parallel Load Balancing)](#eplb-expert-parallel-load-balancing)
19. [TBO (Two-Batch Overlap)](#tbo-two-batch-overlap)
20. [DeepSeek MHA Optimization](#deepseek-mha-optimization)
21. [NSA Backend](#nsa-backend)
22. [Flash Attention](#flash-attention)
23. [Kernel Configuration](#kernel-configuration)
24. [Speculative Decoding](#speculative-decoding)
25. [Vision Language Models (VLM)](#vision-language-models-vlm)
26. [VLM Item CUDA IPC Transport](#vlm-item-cuda-ipc-transport)
27. [Mamba](#mamba)
28. [AMD and ROCm](#amd-and-rocm)
29. [Ascend NPU](#ascend-npu)
30. [Moore Threads (MUSA)](#moore-threads-musa)
31. [MPS (Apple Silicon)](#mps-apple-silicon)
32. [Hi-Cache](#hi-cache)
33. [Mooncake Store](#mooncake-store)
34. [Mooncake KV Transfer](#mooncake-kv-transfer)
35. [Tool Calling](#tool-calling)
36. [Test and Debug](#test-and-debug)
37. [CI and Testing](#ci-and-testing)
38. [Constrained Decoding (Grammar)](#constrained-decoding-grammar)
39. [Profiler Configuration](#profiler-configuration)
40. [HTTP Server](#http-server)
41. [gRPC Server](#grpc-server)
42. [Health Check](#health-check)
43. [Encoder gRPC](#encoder-grpc)
44. [External Models](#external-models)
45. [NUMA Binding](#numa-binding)
46. [Metrics](#metrics)
47. [Tokenizer](#tokenizer)
48. [TokenizerManager](#tokenizermanager)
49. [Symmetric Memory](#symmetric-memory)
50. [Aiter](#aiter)
51. [EPD (Encoder Prefill Decode)](#epd-encoder-prefill-decode)
52. [RoPE Cache Configuration](#rope-cache-configuration)
53. [Unified Radix Tree](#unified-radix-tree)
54. [CUDA Graph](#cuda-graph)
55. [Memory Saver](#memory-saver)
56. [Sparse Embeddings](#sparse-embeddings)
57. [Logits Processor](#logits-processor)
58. [Tool-Call Behavior](#tool-call-behavior)
59. [Ngram](#ngram)
60. [Warmup](#warmup)
61. [SGLang Cache](#sglang-cache)
62. [Plugin System](#plugin-system)
63. [Deterministic Inference](#deterministic-inference)
64. [Deprecated Variables](#deprecated-variables)
65. [Usage Examples](#usage-examples)

---

## Environment Variable System Overview

SGLang environment variables are defined in `sglang/srt/environ.py`. They use a typed field system with automatic parsing and validation:

- **EnvBool**: Boolean values (`true`, `1`, `yes`, `y` / `false`, `0`, `no`, `n`)
- **EnvInt**: Integer values
- **EnvFloat**: Float values
- **EnvStr**: String values
- **EnvTuple**: Comma-separated tuple values

All variables are accessed via `envs.VARIABLE_NAME.get()`. Using `envs.VARIABLE_NAME` directly (as a boolean) raises a RuntimeError.

The environment variable system also handles automatic migration from deprecated `SGL_*` prefixed variables to `SGLANG_*` prefixed ones.

---

## Model and File Download

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_MODELSCOPE` | bool | `false` | Use ModelScope for model downloads instead of HuggingFace |
| `SGLANG_SORT_WEIGHT_FILES` | bool | `false` | Sort weight files before loading |
| `SGLANG_DISABLED_MODEL_ARCHS` | tuple | `()` | Comma-separated list of disabled model architectures |
| `SGLANG_PREFETCH_BLOCK_SIZE_MB` | int | `16` | Block size in MB for prefetching model weights |

---

## Logging Options

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_LOG_GC` | bool | `false` | Enable garbage collection logging |
| `SGLANG_LOG_FORWARD_ITERS` | bool | `false` | Log forward iteration counts |
| `SGLANG_LOG_MS` | bool | `false` | Log millisecond-level timing |
| `SGLANG_LOG_REQUEST_EXCEEDED_MS` | int | `-1` | Log requests that exceed this many milliseconds. -1 disables. |
| `SGLANG_LOG_REQUEST_HEADERS` | tuple | `()` | Comma-separated list of request headers to log |
| `SGLANG_LOG_SCHEDULER_STATUS_TARGET` | str | `""` | Target for scheduler status logging |
| `SGLANG_LOG_SCHEDULER_STATUS_INTERVAL` | float | `60.0` | Interval in seconds for scheduler status logging |

---

## Scheduler Configuration

### Memory Leak Test

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_TEST_RETRACT` | bool | `false` | Test retract functionality in scheduler |
| `SGLANG_TEST_RETRACT_INTERVAL` | int | `3` | Interval for retract testing |
| `SGLANG_TEST_RETRACT_NO_PREFILL_BS` | int | `2^31` | Max batch size without prefill during retract test |
| `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY` | int | `0` | Enable strict memory checks during busy periods (0=off, 1=on) |
| `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE` | bool | `true` | Enable strict memory checks during idle periods |

### New Token Ratio Hyperparameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_INIT_NEW_TOKEN_RATIO` | float | `0.7` | Initial new token ratio for scheduler |
| `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR` | float | `0.14` | Minimum new token ratio factor |
| `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS` | int | `600` | Number of steps for new token ratio decay |
| `SGLANG_RETRACT_DECODE_STEPS` | int | `20` | Number of decode steps before retract |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | int | `4096` | Maximum estimation for new tokens clipping |

### Receive Interval

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT` | int | `1000` | Default weight for receive skipper |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE` | int | `1` | Decode weight for receive skipper |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY` | int | `1` | Target verify weight for receive skipper |
| `SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE` | int | `1` | None weight for receive skipper |

### Other Scheduler Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_EMPTY_CACHE_INTERVAL` | float | `-1` | Interval in seconds for emptying cache. Set if you observe high memory accumulation over long serving periods. -1 disables. |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` | bool | `false` | Disable consecutive prefill overlap |
| `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` | int | `-1` | Maximum receives per scheduler poll. -1 for unlimited. |
| `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` | bool | `false` | Use experimental C++ radix tree implementation |
| `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` | float | `0.75` | Smooth factor for dynamic chunking |
| `SGLANG_SCHEDULER_SKIP_ALL_GATHER` | bool | `false` | Skip all-gather in scheduler |
| `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE` | bool | `false` | Decrease prefill during idle (deprecated: use `--enable-prefill-delayer`) |
| `SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES` | int | `None` | Maximum delay passes for prefill delayer (deprecated: use CLI flag) |
| `SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK` | float | `None` | Token usage low watermark for prefill delayer (deprecated: use CLI flag) |
| `SGLANG_DATA_PARALLEL_BUDGET_INTERVAL` | int | `1` | Data parallel budget interval |
| `SGLANG_REQ_WAITING_TIMEOUT` | float | `-1` | Request waiting timeout in seconds. -1 disables. |
| `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH` | bool | `false` | Enable NCCL all-gather in overlap scheduler sync batch |
| `SGLANG_REQ_RUNNING_TIMEOUT` | float | `-1` | Request running timeout in seconds. -1 disables. |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL` | int | `120` | Cleanup interval in seconds for disaggregation bootstrap entries |
| `SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER` | float | `1.0` | Sliding window attention eviction interval multiplier |
| `SGLANG_FORCE_STREAM_INTERVAL` | int | `50` | For non-streaming requests, flush intermediate output every N decoded tokens to record TTFT. Lower for accurate TTFT benchmarking. |
| `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE` | bool | `false` | Deprecated. Use `--enable-prefill-delayer` CLI flag instead. |

---

## PD Disaggregation (Runtime)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | int | `None` | Thread pool size for PD disaggregation. Default computed dynamically from CPU count. |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | int | `4` | Queue size for PD disaggregation operations |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | int | `300` | Bootstrap timeout in seconds for PD disaggregation |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | float | `5.0` | Heartbeat interval in seconds |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | int | `2` | Maximum heartbeat failures before considering node down |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | int | `300` | Waiting timeout in seconds for disaggregation |
| `SGLANG_DISAGGREGATION_NIXL_BACKEND` | str | `"UCX"` | NIXL backend for disaggregation transfer |
| `SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS` | str | `"{}"` | JSON parameters for NIXL backend |
| `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` | bool | `false` | Enable all CP ranks transfer in disaggregation |
| `SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK` | bool | `false` | Force query prefill DP rank in disaggregation |
| `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS` | int | `0` | Extra slots in req_to_token_pool for decode workers. Only effective when max_num_reqs > 32. |

### Staging Buffer

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_DISAGG_STAGING_BUFFER` | bool | `false` | Enable staging buffer for heterogeneous TP KV transfer |
| `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` | int | `64` | Staging buffer size in MB |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | int | `4096` | Staging pool size in MB |
| `SGLANG_STAGING_USE_TORCH` | bool | `false` | Use torch fallback for staging buffer instead of Triton |

### Test: PD Disaggregation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_TEST_PD_DISAGG_BACKEND` | str | `"mooncake"` | Backend for PD disaggregation testing |
| `SGLANG_TEST_PD_DISAGG_DEVICES` | str | `None` | Devices for PD disaggregation testing |

---

## Model Parallel

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_MESSAGE_QUEUE_BROADCASTER` | bool | `true` | Use message queue broadcaster for model parallel |
| `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` | bool | `false` | Restrict each process to one visible GPU |
| `SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE` | str | `None` | Override distributed init method. Set to "env://" to use externally-created TCPStore via MASTER_ADDR/MASTER_PORT. |
| `SGLANG_TCP_STORE_PORT` | int | `29600` | Port for TCPStore in distributed communication |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` | bool | `true` | Enable TP memory imbalance check |
| `SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK` | bool | `false` | Deprecated. Use SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK. |
| `SGLANG_IS_FIRST_RANK_ON_NODE` | bool | `true` | Whether this is the first rank on the node |
| `SGLANG_SYNC_TOKEN_IDS_ACROSS_TP` | bool | `false` | Synchronize token IDs across tensor parallel ranks |
| `SGLANG_ENABLE_COLOCATED_BATCH_GEN` | bool | `false` | Enable colocated batch generation |

---

## Quantization

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_INT4_WEIGHT` | bool | `false` | Enable INT4 weight quantization |
| `SGLANG_CPU_QUANTIZATION` | bool | `false` | Enable CPU quantization |
| `SGLANG_USE_DYNAMIC_MXFP4_LINEAR` | bool | `false` | Enable dynamic MXFP4 linear quantization |
| `SGLANG_FORCE_FP8_MARLIN` | bool | `false` | Force FP8 Marlin kernel usage |
| `SGLANG_MOE_NVFP4_DISPATCH` | bool | `false` | Enable NVFP4 dispatch for MoE models |
| `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` | bool | `false` | Use FP8 GEMM in attention with NVFP4 checkpoint |
| `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` | bool | `false` | Use FP8 for next-N MoE with NVFP4 checkpoint |
| `SGLANG_QUANT_ALLOW_DOWNCASTING` | bool | `false` | Allow downcasting in quantization |
| `SGLANG_FP8_IGNORED_LAYERS` | str | `""` | Comma-separated list of layers to ignore for FP8 quantization |

---

## FlashInfer

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_IS_FLASHINFER_AVAILABLE` | bool | `true` | Whether FlashInfer is available |
| `SGLANG_FLASHINFER_USE_PAGED` | bool | `false` | Use paged attention in FlashInfer |
| `SGLANG_FLASHINFER_WORKSPACE_SIZE` | int | `384 * 1024 * 1024` (384 MB) | Workspace size for FlashInfer |
| `SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR` | float | `None` | Skip-softmax threshold scale factor for TRT-LLM attention (prefill). None = standard attention. |
| `SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR` | float | `None` | Skip-softmax threshold scale factor for TRT-LLM attention (decode). None = standard attention. |
| `SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT` | bool | `None` | Force POSIX FD transport for FlashInfer allreduce-fusion. Workaround for GB200/GB300 transport issues. |
| `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE` | int | `4096` | Tile size for FlashInfer prefill split |
| `SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE` | int | `2048` | Tile size for FlashInfer decode split |

---

## Triton

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS` | bool | `false` | Enable static KV splits for Triton decode attention |
| `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE` | bool | `false` | Use custom Triton kernel cache |
| `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE` | int | `4096` | Alignment size for Triton prefill truncation |
| `SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE` | int | `256` | Tile size for Triton decode split |

---

## Torch Compile

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_TORCH_COMPILE` | bool | `false` | Enable torch.compile optimization |

---

## DeepGemm

DeepGemm provides JIT-compiled FP8/microscaling GEMM kernels.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_JIT_DEEPGEMM` | bool | `true` | Enable JIT DeepGemm kernel compilation |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | bool | `true` | Pre-compile DeepGemm kernels during startup |
| `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` | bool | `false` | Use fast warmup for DeepGemm compilation |
| `SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS` | int | `4` | Number of compile workers for DeepGemm |
| `SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE` | bool | `false` | Internal flag indicating pre-compile stage |
| `SGLANG_DG_CACHE_DIR` | str | `~/.cache/deep_gemm` | Cache directory for DeepGemm compiled kernels |
| `SGLANG_DG_USE_NVRTC` | bool | `false` | Use NVRTC for DeepGemm compilation instead of nvcc |
| `SGLANG_USE_DEEPGEMM_BMM` | bool | `false` | Use DeepGemm batched matrix multiply |
| `SGLANG_DEEPGEMM_SANITY_CHECK` | bool | `false` | Enable sanity checks for DeepGemm kernels |

---

## DeepEP

DeepEP provides high-performance all-to-all communication for MoE expert parallelism.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_DEEPEP_BF16_DISPATCH` | bool | `false` | Use BF16 for DeepEP dispatch |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | int | `128` | Maximum number of dispatched tokens per rank |
| `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` | int | `32` | Number of SMs for low-latency combine send |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | bool | `false` | Overlap shared experts outside SBO on Blackwell |

---

## NIXL-EP

NIXL-EP provides an alternative all-to-all backend for MoE.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_NIXL_EP_BF16_DISPATCH` | bool | `false` | Use BF16 for NIXL-EP dispatch |
| `SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | int | `128` | Maximum number of dispatched tokens per rank |

---

## EPLB (Expert Parallel Load Balancing)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT` | bool | `false` | Log input to expert location updater |
| `SGLANG_EXPERT_LOCATION_UPDATER_CANARY` | bool | `false` | Enable canary testing for expert location updater |
| `SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS` | bool | `false` | Log metrics from expert location updater |
| `SGLANG_LOG_EXPERT_LOCATION_METADATA` | bool | `false` | Log expert location metadata |
| `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` | str | `"/tmp"` | Directory for expert distribution recorder output |
| `SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` | int | `0` | Interval for EPLB heatmap collection. 0 disables. |
| `SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC` | bool | `false` | Enable EPLB balancedness metric |

---

## TBO (Two-Batch Overlap)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_TBO_DEBUG` | bool | `false` | Enable debug logging for Two-Batch Overlap |

---

## DeepSeek MHA Optimization

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD` | int | `8192` | Threshold for chunked prefix caching in DeepSeek MHA |

---

## NSA Backend

NSA (Native Sparse Attention) backend for DeepSeek V3.2.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_NSA_FUSE_TOPK` | bool | `true` | Fuse TopK operation in NSA |
| `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` | bool | `true` | Enable MTP precompute metadata for NSA |
| `SGLANG_USE_FUSED_METADATA_COPY` | bool | `true` | Use fused metadata copy |
| `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` | int | `2048` | KV length threshold for switching to dense attention during prefill |

---

## Flash Attention

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_SGL_FA3_KERNEL` | bool | `true` | Use SGL's FA3 kernel for flash attention |

---

## Kernel Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_TRITON_W8A8_FP8_KERNEL` | bool | `false` | Use Triton kernel for W8A8 FP8 operations |
| `SGLANG_RETURN_ORIGINAL_LOGPROB` | bool | `false` | Return original log probabilities |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | bool | `false` | Allow overwriting longer context length |
| `SGLANG_MOE_PADDING` | bool | `false` | Enable padding for MoE operations |
| `SGLANG_CUTLASS_MOE` | bool | `false` | Use CUTLASS MoE kernels |
| `HF_HUB_DISABLE_XET` | bool | `false` | Disable XET for HuggingFace Hub |
| `DISABLE_OPENAPI_DOC` | bool | `false` | Disable OpenAPI documentation endpoint |
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | bool | `false` | Enable torch inference mode |
| `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK` | bool | `false` | Skip sgl-kernel version compatibility check |

---

## Speculative Decoding

### Overlap Spec V2

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_SPEC_V2` | bool | `true` | Enable speculative decoding V2 |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | bool | `false` | Enable overlap plan stream for speculative decoding |

### Spec Config

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK` | bool | `true` | Enable strict filter checking for speculative decoding |
| `SGLANG_SPEC_NAN_DETECTION` | bool | `false` | Enable NaN detection in speculative decoding |
| `SGLANG_SPEC_OOB_DETECTION` | bool | `false` | Enable out-of-bounds detection in speculative decoding |

---

## Vision Language Models (VLM)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_VLM_CACHE_SIZE_MB` | int | `100` | Cache size in MB for VLM operations |
| `SGLANG_IMAGE_MAX_PIXELS` | int | `16384 * 28 * 28` | Maximum pixels for image processing |
| `SGLANG_RESIZE_RESAMPLE` | str | `""` | Resampling method for image resize |
| `SGLANG_MM_BUFFER_SIZE_MB` | int | `0` | Buffer size in MB for multimodal operations |
| `SGLANG_MM_PRECOMPUTE_HASH` | bool | `false` | Pre-compute hash for multimodal items |
| `SGLANG_VIT_ENABLE_CUDA_GRAPH` | bool | `false` | Enable CUDA graph for vision transformer |
| `SGLANG_MM_SKIP_COMPUTE_HASH` | bool | `false` | Skip compute hash for multimodal items |

### VLM Item CUDA IPC Transport

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | bool | `false` | Use CUDA IPC transport for VLM items |
| `SGLANG_USE_IPC_POOL_HANDLE_CACHE` | bool | `false` | Use IPC pool handle cache |
| `SGLANG_MM_FEATURE_CACHE_MB` | int | `1024` | Feature cache size in MB for multimodal |
| `SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC` | float | `0.05` | Recycle interval in seconds for MM item memory pool |

---

## Mamba

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_MAMBA_CONV_DTYPE` | str | `"bfloat16"` | Data type for Mamba convolution |
| `SGLANG_MAMBA_SSM_DTYPE` | str | `None` | Data type for Mamba SSM |

---

## AMD and ROCm

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_AITER` | bool | `false` | Enable Aiter acceleration for AMD GPUs |
| `SGLANG_USE_AITER_UNIFIED_ATTN` | bool | `false` | Use Aiter unified attention |
| `SGLANG_ROCM_FUSED_DECODE_MLA` | bool | `false` | Enable fused decode MLA for ROCm |
| `SGLANG_ROCM_DISABLE_LINEARQUANT` | bool | `false` | Disable linear quantization on ROCm |
| `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | int | `4096` | Maximum dispatch tokens per rank for MoRI |
| `SGLANG_USE_AITER_FP8_PER_TOKEN` | bool | `false` | Use Aiter FP8 per-token quantization |
| `SGLANG_USE_1STAGE_ALLREDUCE` | bool | `false` | Use 1-stage all-reduce kernel on AMD (deterministic). Auto-enabled with --enable-deterministic-inference. |

---

## Ascend NPU

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT` | bool | `false` | Disable ACL format weight casting on NPU |
| `SGLANG_NPU_USE_MULTI_STREAM` | bool | `false` | Enable dual-stream computation for shared/routing experts |
| `SGLANG_NPU_USE_MLAPO` | bool | `false` | Use MLAPO fusion operator for MLA attention preprocessing |
| `SGLANG_NPU_FORWARD_NATIVE_GELUTANH` | bool | `false` | Use native GELU tanh activation for specific models |
| `SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM` | bool | `false` | Use native Gemma RMS norm for specific models |
| `SGLANG_USE_AG_AFTER_QLORA` | bool | `false` | Delay all-gather after QLoRA for DeepSeek v3.2 |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | bool | `false` | Quantize x to int8 in dispatch operator |
| `SGLANG_NPU_FUSED_MOE_MODE` | int | `1` | Fused MoE mode for NPU |
| `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE` | bool | `false` | Enable Ascend transfer with Mooncake |
| `ASCEND_NPU_PHY_ID` | int | `-1` | Physical ID for Ascend NPU |

---

## Moore Threads (MUSA)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA` | bool | `false` | Force update metadata for MUSA FA3 kernel |

---

## MPS (Apple Silicon)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_MLX` | bool | `false` | Use MLX framework for Apple Silicon |

---

## Hi-Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_HICACHE_HF3FS_CONFIG_PATH` | str | `None` | Configuration path for HF3FS Hi-Cache |
| `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE` | int | `None` | Decode offload stride for Hi-Cache |
| `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` | str | `None` | Storage directory for file-based Hi-Cache backend |
| `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR` | str | `None` | Storage directory for NIXL-based Hi-Cache backend |
| `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` | str | `None` | Configuration path for Mooncake Hi-Cache |
| `SGLANG_HICACHE_MOONCAKE_REUSE_TE` | bool | `true` | Reuse transfer engine for Mooncake Hi-Cache |

---

## Mooncake Store

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MOONCAKE_MASTER` | str | `None` | Mooncake master address |
| `MOONCAKE_CLIENT` | str | `None` | Mooncake client address |
| `MOONCAKE_LOCAL_HOSTNAME` | str | `"localhost"` | Local hostname for Mooncake |
| `MOONCAKE_TE_META_DATA_SERVER` | str | `"P2PHANDSHAKE"` | Metadata server for Mooncake transfer engine |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | str | `"4gb"` | Global segment size for Mooncake |
| `MOONCAKE_PROTOCOL` | str | `"tcp"` | Protocol for Mooncake communication |
| `MOONCAKE_DEVICE` | str | `""` | Device for Mooncake |
| `MOONCAKE_MASTER_METRICS_PORT` | int | `9003` | Metrics port for Mooncake master |
| `MOONCAKE_CHECK_SERVER` | bool | `false` | Check Mooncake server availability |
| `MOONCAKE_STANDALONE_STORAGE` | bool | `false` | Use standalone storage for Mooncake |

### Mooncake KV Transfer

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | str | `None` | Custom memory pool for Mooncake KV transfer |
| `SGLANG_MOONCAKE_SEND_AUX_TCP` | bool | `false` | Send auxiliary data via TCP for Mooncake |
| `SGLANG_MOONCAKE_TRANS_THREAD` | str/EnvInt | varies | Number of transfer threads for Mooncake (used in PD deployment) |

---

## Tool Calling

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_FORWARD_UNKNOWN_TOOLS` | bool | `false` | Forward unknown tools to the model |
| `SGLANG_TOOL_STRICT_LEVEL` | int | `0` (OFF) | Strictness level for tool call parsing. 0=OFF, 1=FUNCTION, 2=PARAMETER |

---

## Test and Debug

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_DETECT_SLOW_RANK` | bool | `false` | Detect slow ranks in distributed inference |
| `SGLANG_TEST_STUCK_DETOKENIZER` | float | `0` | Test stuck detokenizer with timeout |
| `SGLANG_TEST_STUCK_DP_CONTROLLER` | float | `0` | Test stuck DP controller with timeout |
| `SGLANG_TEST_STUCK_SCHEDULER_INIT` | float | `0` | Test stuck scheduler initialization |
| `SGLANG_TEST_STUCK_TOKENIZER` | float | `0` | Test stuck tokenizer with timeout |
| `SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS` | int | `0` | Crash after N stream outputs (for testing) |
| `IS_H200` | bool | `false` | Simulate H200 GPU for testing |
| `SGLANG_SET_CPU_AFFINITY` | bool | `false` | Set CPU affinity for workers |
| `SGLANG_RECORD_STEP_TIME` | bool | `false` | Record per-step timing |
| `SGLANG_FORCE_SHUTDOWN` | bool | `false` | Force shutdown without graceful cleanup |
| `SGLANG_DEBUG_MEMORY_POOL` | bool | `false` | Debug memory pool operations |
| `SGLANG_TEST_REQUEST_TIME_STATS` | bool | `false` | Enable request time statistics for testing |
| `SGLANG_SIMULATE_ACC_LEN` | float | `-1` | Simulate accept length for speculative decoding testing |
| `SGLANG_SIMULATE_ACC_METHOD` | str | `"match-expected"` | Method for simulating accept length |
| `SGLANG_NATIVE_MOVE_KV_CACHE` | bool | `false` | Use native move KV cache (for testing) |

---

## CI and Testing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_IS_IN_CI` | bool | `false` | Indicates running in CI environment |
| `SGLANG_IS_IN_CI_AMD` | bool | `false` | Indicates running in AMD CI environment |
| `SGLANG_CUDA_COREDUMP` | bool | `false` | Enable CUDA core dump on errors |
| `SGLANG_CUDA_COREDUMP_DIR` | str | `"/tmp/sglang_cuda_coredumps"` | Directory for CUDA core dumps |
| `SGLANG_TEST_MAX_RETRY` | int | `None` | Maximum retries for CI tests |

---

## Constrained Decoding (Grammar)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_GRAMMAR_POLL_INTERVAL` | float | `0.005` | Poll interval in seconds for grammar constrained decoding |
| `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` | int | `10000` | Maximum poll iterations for grammar constrained decoding |
| `SGLANG_DISABLE_OUTLINES_DISK_CACHE` | bool | `false` | Disable outlines disk cache for grammar |

---

## Profiler Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_TORCH_PROFILER_DIR` | str | `"/tmp"` | Directory for PyTorch profiler trace output |
| `SGLANG_PROFILE_WITH_STACK` | bool | `true` | Include stack traces in profiler output |
| `SGLANG_PROFILE_RECORD_SHAPES` | bool | `true` | Record tensor shapes in profiler |
| `SGLANG_PROFILE_V2` | bool | `false` | Use profiler V2 |
| `SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS` | int | `500` | Schedule delay in milliseconds for OTLP exporter |
| `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE` | int | `64` | Maximum export batch size for OTLP exporter |
| `SGLANG_ENABLE_METRICS_DEVICE_TIMER` | bool | `false` | Enable device timer for metrics |
| `SGLANG_ENABLE_METRICS_DP_ATTENTION` | bool | `false` | Enable DP attention metrics |

---

## HTTP Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_TIMEOUT_KEEP_ALIVE` | int | `5` | Keep-alive timeout in seconds for HTTP server |

### HTTP/2 Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_GRANIAN_PARENT_PID` | int | `None` | Parent PID for Granian HTTP/2 server |

---

## gRPC Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_GRPC_PORT` | int | `None` | Port for native gRPC server |
| `SGLANG_ENABLE_GRPC` | bool | `false` | Enable native gRPC server |

---

## Health Check

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` | bool | `true` | Enable generation-based health endpoint |

---

## Encoder gRPC

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENCODER_GRPC_TIMEOUT_SECS` | int | `60` | Timeout in seconds for encoder gRPC |
| `SGLANG_ENCODER_MM_RECEIVER_MODE` | str | `"http"` | Encoder receiver mode: http or grpc |
| `SGLANG_ENCODER_RECV_TIMEOUT` | float | `180.0` | Timeout in seconds for encoder receive |
| `SGLANG_ENCODER_SEND_TIMEOUT` | float | `180.0` | Timeout in seconds for encoder send |
| `SGLANG_ENCODER_DISPATCH_MIN_ITEMS` | int | `2` | Minimum items for encoder dispatch |

---

## External Models

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_EXTERNAL_MODEL_PACKAGE` | str | `""` | External model package path |
| `SGLANG_EXTERNAL_MM_MODEL_ARCH` | str | `""` | External multimodal model architecture |
| `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE` | str | `""` | External multimodal processor package |

---

## NUMA Binding

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_NUMA_BIND_V2` | bool | `true` | Enable NUMA binding V2 |
| `SGLANG_AUTO_NUMA_BIND` | bool | `false` | Automatically bind to NUMA nodes |

---

## Tokenizer

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_PATCH_TOKENIZER` | bool | `true` | Patch tokenizer for all_special_tokens/all_special_ids caching (improves ITL by up to 10x at high batch sizes) |

---

## TokenizerManager

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_REQUEST_STATE_WAIT_TIMEOUT` | int | `4` | Timeout in seconds for request state waiting |

---

## Symmetric Memory

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE` | int | `-1` | Pre-allocated symmetric memory size in GB. -1 disables. |
| `SGLANG_DEBUG_SYMM_MEM` | bool | `false` | Debug symmetric memory operations |

---

## Aiter

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_AITER_FP8_PER_TOKEN` | bool | `false` | Use Aiter FP8 per-token quantization |

---

## EPD (Encoder Prefill Decode)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENCODER_RECV_TIMEOUT` | float | `180.0` | Timeout for encoder receive in EPD |
| `SGLANG_ENCODER_SEND_TIMEOUT` | float | `180.0` | Timeout for encoder send in EPD |
| `SGLANG_ENCODER_DISPATCH_MIN_ITEMS` | int | `2` | Minimum items for EPD dispatch |

---

## RoPE Cache Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_SPEC_EXPANSION_SAFETY_FACTOR` | int | `2` | Safety factor for speculative expansion |
| `SGLANG_ROPE_CACHE_SAFETY_MARGIN` | int | `256` | Safety margin for RoPE cache |
| `SGLANG_ROPE_CACHE_ALIGN` | int | `128` | Alignment for RoPE cache |

---

## Unified Radix Tree

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_UNIFIED_RADIX_TREE` | bool | `false` | Enable unified radix tree implementation |

---

## CUDA Graph

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_USE_BREAKABLE_CUDA_GRAPH` | bool | `false` | Enable breakable CUDA graph |

---

## Memory Saver

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_MEMORY_SAVER_CUDA_GRAPH` | bool | `false` | Enable memory saver for CUDA graph |

---

## Sparse Embeddings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_EMBEDDINGS_SPARSE_HEAD` | str | `None` | Sparse head for embedding models |

---

## Logits Processor

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK` | bool | `false` | Enable chunked logits processing |
| `SGLANG_LOGITS_PROCESSER_CHUNK_SIZE` | int | `2048` | Chunk size for logits processing |

---

## Ngram

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | bool | `false` | Force greedy verification for Ngram speculative decoding |

---

## Warmup

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_WARMUP_TIMEOUT` | float | `-1` | Timeout in seconds for warmup forward batch. If exceeded, server crashes to prevent hanging. Increase to 1800 for kernel JIT precache (e.g., DeepGEMM). -1 disables. |

---

## Elastic EP Backup Port

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_BACKUP_PORT_BASE` | int | `10000` | Base port for Elastic EP backup |

---

## SGLang Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_CACHE_DIR` | str | `~/.cache/sglang` | Cache directory for SGLang |

---

## Plugin System

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_PLATFORM` | str | `""` | Select platform plugin by entry_point name (e.g., `kunlun`, `demo_cuda`). Required when multiple plugins would activate. |
| `SGLANG_PLUGINS` | str | `""` | Comma-separated whitelist of general plugin names to load (group: `sglang.srt.plugins`). If unset, all discovered plugins are loaded. |

---

## Deterministic Inference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_DETERMINISTIC_INFERENCE` | bool | `false` | Enable deterministic inference mode |
| `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` | bool | `false` | Use custom all-reduce V2 kernel |

---

## Deprecated Variables

The following environment variables are deprecated and automatically migrated to their replacements:

| Deprecated Variable | Replacement | Notes |
|--------------------|-------------|-------|
| `SGL_*` prefix | `SGLANG_*` prefix | All `SGL_` variables are automatically converted to `SGLANG_` |
| `SGLANG_GC_LOG` | `SGLANG_LOG_GC` | Renamed |
| `SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH` | `SGLANG_MOE_NVFP4_DISPATCH` | Renamed |
| `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK` | `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` | Renamed (inverted logic) |
| `SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2` | (removed) | No replacement |
| `SGLANG_USE_JIT_ALL_REDUCE` | `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` | Renamed |
| `SGLANG_QUEUED_TIMEOUT_MS` | `SGLANG_REQ_WAITING_TIMEOUT` | Converted from ms to seconds |
| `SGLANG_FORWARD_TIMEOUT_MS` | `SGLANG_REQ_RUNNING_TIMEOUT` | Converted from ms to seconds |
| `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE` | `--enable-prefill-delayer` CLI flag | Migrated to CLI flag |
| `SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES` | `--prefill-delayer-max-delay-passes` CLI flag | Migrated to CLI flag |
| `SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK` | `--prefill-delayer-token-usage-low-watermark` CLI flag | Migrated to CLI flag |

---

## Usage Examples

### Setting environment variables

```bash
# Enable profiling
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Enable Aiter on AMD
export SGLANG_USE_AITER=1
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# CPU affinity for multi-node
export SGLANG_SET_CPU_AFFINITY=true

# Increase warmup timeout for JIT compilation
export SGLANG_WARMUP_TIMEOUT=1800

# Enable DeepGemm JIT compilation
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=1

# Set CPU affinity for Ascend NPU
export SGLANG_SET_CPU_AFFINITY=1

# Fix PyTorch profiler stack error
export SGLANG_PROFILE_WITH_STACK=False
```

### Using override context manager

```python
from sglang.srt.environ import envs

# Override in code (restores after context)
with envs.SGLANG_TEST_RETRACT.override(True):
    # SGLANG_TEST_RETRACT is True here
    pass
# SGLANG_TEST_RETRACT is restored

# Set explicitly
envs.SGLANG_TEST_RETRACT.set(True)
assert envs.SGLANG_TEST_RETRACT.get() is True

# Check if set
if envs.SGLANG_TEST_RETRACT.is_set():
    print("Variable is set")

# Clear
envs.SGLANG_TEST_RETRACT.clear()
```

### Using with subprocess

```python
import subprocess
from sglang.srt.environ import envs

command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
with envs.SGLANG_TEST_RETRACT.override(True):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    output = process.stdout.read().decode('utf-8').strip()
    assert output == "True"
```

### Using temp_set_env for non-SGLang variables

```python
from sglang.srt.environ import temp_set_env

with temp_set_env(CUSTOM_VAR="value"):
    # CUSTOM_VAR is set here
    pass
# CUSTOM_VAR is restored

# SGLANG_ variables are rejected by default:
with temp_set_env(SGLANG_LOG_GC="true"):  # Raises ValueError
    pass

# Use allow_sglang=True for special cases:
with temp_set_env(allow_sglang=True, SGLANG_LOG_GC="true"):
    pass
```

### Using with ExitStack (unit tests)

```python
from contextlib import ExitStack
from sglang.srt.environ import envs

exit_stack = ExitStack()
exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
assert envs.SGLANG_TEST_RETRACT.get() is False
exit_stack.close()
assert envs.SGLANG_TEST_RETRACT.get() is None
```
