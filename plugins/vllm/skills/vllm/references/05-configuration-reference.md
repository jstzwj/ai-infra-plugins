# vLLM Configuration Reference

This document provides comprehensive reference documentation for all vLLM configuration classes.
Every configuration parameter is documented with its type, default value, and description.

---

## Table of Contents

1. [VllmConfig (Master Config)](#vllmconfig-master-config)
2. [ModelConfig](#modelconfig)
3. [CacheConfig](#cacheconfig)
4. [ParallelConfig](#parallelconfig)
5. [SchedulerConfig](#schedulerconfig)
6. [DeviceConfig](#deviceconfig)
7. [AttentionConfig](#attentionconfig)
8. [CompilationConfig](#compilationconfig)
9. [QuantizationConfig](#quantizationconfig)
10. [SpeculativeConfig](#speculativeconfig)
11. [LoRAConfig](#loraconfig)
12. [LoadConfig](#loadconfig)
13. [MultimodalConfig](#multimodalconfig)
14. [ObservabilityConfig](#observabilityconfig)
15. [KV Transfer Configs](#kv-transfer-configs)
16. [KV Events Config](#kv-events-config)
17. [EC Transfer Config](#ec-transfer-config)
18. [Weight Transfer Config](#weight-transfer-config)
19. [KernelConfig](#kernelconfig)
20. [MambaConfig](#mambaconfig)
21. [PoolerConfig](#poolerconfig)
22. [ProfilerConfig](#profilerconfig)
23. [ReasoningConfig](#reasoningconfig)
24. [StructuredOutputsConfig](#structuredoutputsconfig)
25. [SpeechToTextConfig](#speechtotextconfig)
26. [OffloadConfig](#offloadconfig)
27. [ModelArchitectureConfig](#modelarchitectureconfig)
28. [Optimization Levels](#optimization-levels)
29. [Helper Types and Utilities](#helper-types-and-utilities)

---

## VllmConfig (Master Config)

The `VllmConfig` class (`vllm/config/vllm.py`) is the top-level configuration container
that holds all sub-configurations for a vLLM instance.

### Class: `VllmConfig`

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_config` | `ModelConfig` | `None` | Model configuration |
| `cache_config` | `CacheConfig` | `CacheConfig()` | Cache configuration |
| `parallel_config` | `ParallelConfig` | `ParallelConfig()` | Parallel execution configuration |
| `scheduler_config` | `SchedulerConfig` | `SchedulerConfig.default_factory()` | Scheduler configuration |
| `device_config` | `DeviceConfig` | `DeviceConfig()` | Device configuration |
| `load_config` | `LoadConfig` | `LoadConfig()` | Model loading configuration |
| `offload_config` | `OffloadConfig` | `OffloadConfig()` | Model weight offloading configuration |
| `attention_config` | `AttentionConfig` | `AttentionConfig()` | Attention mechanism configuration |
| `mamba_config` | `MambaConfig` | `MambaConfig()` | Mamba SSM configuration |
| `kernel_config` | `KernelConfig` | `KernelConfig()` | Kernel selection configuration |
| `lora_config` | `LoRAConfig | None` | `None` | LoRA adapter configuration |
| `speculative_config` | `SpeculativeConfig | None` | `None` | Speculative decoding configuration |
| `structured_outputs_config` | `StructuredOutputsConfig` | `StructuredOutputsConfig()` | Structured output configuration |
| `observability_config` | `ObservabilityConfig` | `ObservabilityConfig()` | Observability/telemetry configuration |
| `quant_config` | `QuantizationConfig | None` | `None` | Quantization configuration (auto-detected) |
| `compilation_config` | `CompilationConfig` | `CompilationConfig()` | torch.compile and CUDA graph configuration |
| `profiler_config` | `ProfilerConfig` | `ProfilerConfig()` | Profiling configuration |
| `kv_transfer_config` | `KVTransferConfig | None` | `None` | Distributed KV cache transfer configuration |
| `kv_events_config` | `KVEventsConfig | None` | `None` | KV cache event publishing configuration |
| `ec_transfer_config` | `ECTransferConfig | None` | `None` | Distributed EC cache transfer configuration |
| `reasoning_config` | `ReasoningConfig | None` | `None` | Reasoning model configuration |
| `additional_config` | `dict | SupportsHash` | `{}` | Platform-specific additional configuration |
| `instance_id` | `str` | `""` | vLLM instance identifier (auto-generated) |
| `optimization_level` | `OptimizationLevel` | `OptimizationLevel.O2` | Optimization level (O0-O3) |
| `performance_mode` | `PerformanceMode` | `"balanced"` | Performance mode: "balanced", "interactivity", or "throughput" |
| `weight_transfer_config` | `WeightTransferConfig | None` | `None` | Weight transfer for RL training |
| `shutdown_timeout` | `int` | `0` | Shutdown grace period in seconds |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_speculative_tokens` | `int` | Number of speculative tokens (0 if no spec decoding) |
| `needs_dp_coordinator` | `bool` | Whether DP coordinator process is needed |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_hash` | `() -> str` | Hash identifying computation graph structure |
| `with_hf_config` | `(hf_config, architectures=None) -> VllmConfig` | Return copy with updated HF config |
| `enable_trace_function_call_for_thread` | `() -> None` | Set up function tracing for current thread |
| `validate_block_size` | `() -> None` | Validate block_size against DCP/Mamba constraints |
| `__str__` | `() -> str` | String representation with key config values |

### Context Manager

```python
set_current_vllm_config(vllm_config: VllmConfig, check_compile=False, prefix=None)
```

Temporarily sets the current vLLM config for use by CustomOps and other modules.

### Helper Functions

```python
get_current_vllm_config() -> VllmConfig
get_current_vllm_config_or_none() -> VllmConfig | None
get_layers_from_vllm_config(vllm_config, layer_type, layer_names=None) -> dict[str, T]
```

---

## ModelConfig

The `ModelConfig` class (`vllm/config/model.py`) configures the model architecture and behavior.

### Class: `ModelConfig`

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig
```

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"Qwen/Qwen3-0.6B"` | HuggingFace model name or path |
| `model_weights` | `str` | `""` | Original model weights path (for object storage) |
| `runner` | `RunnerOption` | `"auto"` | Runner type: "auto", "generate", "pooling", "draft" |
| `convert` | `ConvertOption` | `"auto"` | Model conversion: "auto", "none", "embed", "classify" |
| `tokenizer` | `str` | `None` | Tokenizer name/path (defaults to model path) |
| `tokenizer_mode` | `TokenizerMode | str` | `"auto"` | Tokenizer mode: "auto", "hf", "slow", "mistral", "deepseek_v32", "deepseek_v4" |
| `trust_remote_code` | `bool` | `False` | Trust remote code from HuggingFace |
| `dtype` | `ModelDType | torch.dtype` | `"auto"` | Data type: "auto", "half", "float16", "bfloat16", "float", "float32" |
| `seed` | `int` | `0` | Random seed for reproducibility |
| `hf_config` | `PretrainedConfig` | (auto) | HuggingFace config (auto-loaded) |
| `hf_text_config` | `PretrainedConfig` | (auto) | HuggingFace text model config |
| `hf_config_path` | `str | None` | `None` | Override path for HF config |
| `allowed_local_media_path` | `str` | `""` | Local media paths allowed for API requests |
| `allowed_media_domains` | `list[str] | None` | `None` | Allowed media URL domains |
| `revision` | `str | None` | `None` | Model version (branch, tag, or commit) |
| `code_revision` | `str | None` | `None` | Code revision for model |
| `tokenizer_revision` | `str | None` | `None` | Tokenizer revision |
| `max_model_len` | `int` | `None` | Maximum context length (prompt + output). -1 = auto |
| `spec_target_max_model_len` | `int | None` | `None` | Max length for spec decoding draft models |
| `quantization` | `QuantizationMethods | str | None` | `None` | Quantization method |
| `enforce_eager` | `bool` | `False` | Disable CUDA graphs, always use eager mode |
| `enable_return_routed_experts` | `bool` | `False` | Return routed expert indices |
| `max_logprobs` | `int` | `20` | Maximum number of logprobs per token. -1 = unlimited |
| `disable_sliding_window` | `bool` | `False` | Disable sliding window attention |
| `skip_tokenizer_init` | `bool` | `False` | Skip tokenizer initialization |
| `served_model_name` | `list[str] | None` | `None` | Model name(s) exposed in the API |
| `limit_mm_per_prompt` | `dict[str, int] | None` | `None` | Max multi-modal items per prompt per modality |
| `use_async_external_dp` | `bool` | `False` | Use async external data parallelism |
| `hf_token` | `bool | str | None` | `None` | HuggingFace authentication token |
| `hf_overrides` | `HfOverrides` | `None` | HuggingFace config overrides |
| `mm_processor_kwargs` | `dict | None` | `None` | Multi-modal processor kwargs |
| `pooler_config` | `PoolerConfig | None` | `None` | Pooler configuration |
| `model_impl` | `ModelImpl` | `"auto"` | Model implementation: "auto", "vllm", "transformers" |
| `override_pooler_config` | `PoolerConfig | None` | `None` | Override pooler config |
| ` logits_processors` | `list | None` | `None` | Custom logits processors |
| `disable_cascade_attn` | `bool` | `False` | Disable cascade attention |
| `enable_prompt_embeds` | `bool` | `False` | Enable prompt embedding inputs |
| `config_format` | `ConfigFormat` | (auto) | Config format: "hf", "gguf", etc. |

### Type Aliases

```python
RunnerOption = Literal["auto", "generate", "pooling", "draft"]
ConvertOption = Literal["auto", "none", "embed", "classify"]
TokenizerMode = Literal["auto", "hf", "slow", "mistral", "deepseek_v32", "deepseek_v4"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
ModelImpl = Literal["auto", "vllm", "transformers", "terratorch"]
```

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_hidden_size` | `() -> int` | Get model hidden dimension |
| `get_head_size` | `() -> int` | Get attention head dimension |
| `get_total_num_kv_heads` | `() -> int` | Total number of KV heads |
| `get_num_kv_heads` | `(parallel_config) -> int` | KV heads per partition |
| `get_num_attention_heads` | `(parallel_config) -> int` | Attention heads per partition |
| `get_vocab_size` | `() -> int` | Vocabulary size |
| `get_diff_sampling_param` | `() -> dict` | Non-default sampling parameters from generation config |
| `verify_with_parallel_config` | `(parallel_config)` | Verify consistency with parallel config |
| `get_pooling_task` | `(supported_tasks)` | Determine pooling task |
| `compute_hash` | `() -> str` | Hash for computation graph |

---

## CacheConfig

The `CacheConfig` class (`vllm/config/cache.py`) configures KV cache behavior.

### Class: `CacheConfig`

```python
@config
class CacheConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `block_size` | `int` | `None` (resolves to 16) | Tokens per cache block |
| `user_specified_block_size` | `bool` | `False` | Whether block_size was explicitly set |
| `hash_block_size` | `int | None` | `None` | Block size for hash computation (finer granularity) |
| `gpu_memory_utilization` | `float` | `0.92` | Fraction of GPU memory for executor (0, 1] |
| `cache_dtype` | `CacheDType` | `"auto"` | KV cache data type |
| `is_attention_free` | `bool` | `False` | Whether model is attention-free |
| `num_gpu_blocks_override` | `int | None` | `None` | Override GPU block count (testing) |
| `sliding_window` | `int | None` | `None` | Sliding window size |
| `enable_prefix_caching` | `bool` | `True` | Enable prefix caching |
| `prefix_caching_hash_algo` | `PrefixCachingHashAlgo` | `"sha256"` | Hash algorithm for prefix caching |
| `calculate_kv_scales` | `bool` | `False` | Deprecated: dynamic KV scale calculation |
| `kv_cache_dtype_skip_layers` | `list[str]` | `[]` | Layers to skip KV cache quantization |
| `mamba_page_size_padded` | `int | None` | `None` | Override mamba page size |
| `mamba_block_size` | `int | None` | `None` | Mamba cache block size (must be multiple of 8) |
| `mamba_cache_dtype` | `MambaDType` | `"auto"` | Mamba cache data type |
| `mamba_ssm_cache_dtype` | `MambaDType` | `"auto"` | Mamba SSM state data type |
| `mamba_cache_mode` | `MambaCacheMode` | `"none"` | Mamba cache strategy: "none", "all", "align" |
| `num_gpu_blocks` | `int | None` | `None` | GPU blocks (set after profiling) |
| `num_cpu_blocks` | `int | None` | `None` | CPU blocks (set after profiling) |
| `kv_sharing_fast_prefill` | `bool` | `False` | Enable KV sharing fast prefill (WIP) |
| `kv_cache_memory_bytes` | `int | None` | `None` | KV cache size in bytes (overrides gpu_memory_utilization) |
| `kv_offloading_size` | `float | None` | `None` | KV offloading buffer size in GiB |
| `kv_offloading_backend` | `KVOffloadingBackend` | `"native"` | KV offloading backend: "native" or "lmcache" |

### Type Aliases

```python
CacheDType = Literal["auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
    "fp8_inc", "fp8_ds_mla", "turboquant_k8v4", "turboquant_4bit_nc",
    "turboquant_k3v4_nc", "turboquant_3bit_nc", "int8_per_token_head",
    "fp8_per_token_head", "nvfp4"]
MambaDType = Literal["auto", "float32", "float16"]
MambaCacheMode = Literal["all", "align", "none"]
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]
KVOffloadingBackend = Literal["native", "lmcache"]
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_BLOCK_SIZE` | `16` | Default block size in tokens |

---

## ParallelConfig

The `ParallelConfig` class (`vllm/config/parallel.py`) configures distributed execution.

### Class: `ParallelConfig`

```python
@config
class ParallelConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline_parallel_size` | `int` | `1` | Number of pipeline parallel groups |
| `tensor_parallel_size` | `int` | `1` | Number of tensor parallel groups |
| `prefill_context_parallel_size` | `int` | `1` | Prefill context parallel groups |
| `data_parallel_size` | `int` | `1` | Data parallel groups (MoE sharded by TP*DP) |
| `data_parallel_size_local` | `int` | `1` | Local data parallel groups |
| `data_parallel_rank` | `int` | `0` | Data parallel rank |
| `data_parallel_rank_local` | `int | None` | `None` | Local DP rank (SPMD mode) |
| `data_parallel_master_ip` | `str` | `"127.0.0.1"` | DP master IP |
| `data_parallel_rpc_port` | `int` | `29550` | DP messaging port |
| `data_parallel_master_port` | `int` | `29500` | DP master port |
| `data_parallel_backend` | `DataParallelBackend` | `"mp"` | DP backend: "mp" or "ray" |
| `data_parallel_external_lb` | `bool` | `False` | Use external DP load balancer |
| `data_parallel_hybrid_lb` | `bool` | `False` | Use hybrid DP load balancer |
| `is_moe_model` | `bool | None` | `None` | Whether model is MoE |
| `enable_expert_parallel` | `bool` | `False` | Use expert parallelism for MoE |
| `enable_ep_weight_filter` | `bool` | `False` | Skip non-local expert weights during loading |
| `enable_eplb` | `bool` | `False` | Enable expert parallel load balancing |
| `eplb_config` | `EPLBConfig` | `EPLBConfig()` | EPLB configuration |
| `expert_placement_strategy` | `ExpertPlacementStrategy` | `"linear"` | Expert placement: "linear" or "round_robin" |
| `all2all_backend` | `All2AllBackend` | `"allgather_reducescatter"` | All2All communication backend |
| `max_parallel_loading_workers` | `int | None` | `None` | Max parallel loading workers |
| `disable_custom_all_reduce` | `bool` | `False` | Disable custom all-reduce kernel |
| `enable_elastic_ep` | `bool` | `False` | Enable elastic expert parallelism |
| `enable_dbo` | `bool` | `False` | Enable dual batch overlap |
| `ubatch_size` | `int` | `0` | Microbatch size |
| `dbo_decode_token_threshold` | `int` | `32` | DBO threshold for decode-only batches |
| `dbo_prefill_token_threshold` | `int` | `512` | DBO threshold for prefill batches |
| `disable_nccl_for_dp_synchronization` | `bool | None` | `None` | Force Gloo for DP sync |
| `ray_workers_use_nsight` | `bool` | `False` | Profile Ray workers with nsight |
| `distributed_executor_backend` | `str | DistributedExecutorBackend | type | None` | `None` | Executor: "ray", "mp", "uni", "external_launcher" |
| `worker_cls` | `str` | `"auto"` | Worker class name |
| `sd_worker_cls` | `str` | `"auto"` | Speculative decoding worker class |
| `worker_extension_cls` | `str` | `""` | Worker extension class |
| `master_addr` | `str` | `"127.0.0.1"` | Multi-node master address |
| `master_port` | `int` | `29501` | Multi-node master port |
| `node_rank` | `int` | `0` | Multi-node rank |
| `nnodes` | `int` | `1` | Number of nodes |
| `numa_bind` | `bool` | `False` | Enable NUMA binding |
| `numa_bind_nodes` | `list[int] | None` | `None` | NUMA node per GPU |
| `numa_bind_cpus` | `list[str] | None` | `None` | CPU list per GPU |
| `distributed_timeout_seconds` | `int | None` | `None` | Distributed operation timeout |
| `rank` | `int` | `0` | Global rank |
| `decode_context_parallel_size` | `int` | `1` | Decode context parallel groups |
| `dcp_kv_cache_interleave_size` | `int` | `1` | DCP KV cache interleave size |
| `dcp_comm_backend` | `DCPCommBackend` | `"ag_rs"` | DCP backend: "ag_rs" or "a2a" |
| `cp_kv_cache_interleave_size` | `int` | `1` | CP KV cache interleave size |

### EPLBConfig Subclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `window_size` | `int` | `1000` | Expert load recording window |
| `step_interval` | `int` | `3000` | Expert rearrangement interval |
| `num_redundant_experts` | `int` | `0` | Number of redundant experts |
| `log_balancedness` | `bool` | `False` | Log balancedness per step |
| `log_balancedness_interval` | `int` | `1` | Balancedness logging interval |
| `use_async` | `bool` | `False` | Use non-blocking EPLB |
| `policy` | `EPLBPolicyOption` | `"default"` | EPLB policy |
| `communicator` | `EPLBCommunicatorBackend | None` | `None` | EPLB communicator backend |

### Type Aliases

```python
ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
All2AllBackend = Literal["naive", "pplx", "deepep_high_throughput", "deepep_low_latency",
    "mori", "nixl_ep", "allgather_reducescatter", "flashinfer_all2allv",
    "flashinfer_nvlink_two_sided", "flashinfer_nvlink_one_sided"]
DCPCommBackend = Literal["ag_rs", "a2a"]
EPLBCommunicatorBackend = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl"]
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `world_size` | `int` | TP * PP * PCP |
| `world_size_across_dp` | `int` | world_size * DP |
| `use_ubatching` | `bool` | Whether microbatching is active |
| `num_ubatches` | `int` | Number of microbatches |
| `local_engines_only` | `bool` | Whether client manages local engines only |
| `use_ray` | `bool` | Whether Ray backend is used |
| `use_sequence_parallel_moe` | `bool` | Whether SP MoE is used |
| `use_batched_dp_moe` | `bool` | Whether batched DP MoE is used |

---

## SchedulerConfig

The `SchedulerConfig` class (`vllm/config/scheduler.py`) configures request scheduling.

### Class: `SchedulerConfig`

```python
@config
class SchedulerConfig
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_model_len` | `InitVar[int]` | required | Maximum sequence length (stored in ModelConfig) |
| `is_encoder_decoder` | `InitVar[bool]` | required | Whether encoder-decoder model |
| `runner_type` | `RunnerType` | `"generate"` | Runner type: "generate", "pooling", "draft" |
| `max_num_batched_tokens` | `int` | `2048` | Max tokens per scheduler iteration |
| `max_num_scheduled_tokens` | `int | None` | `None` | Max tokens scheduler may issue |
| `max_num_seqs` | `int` | `128` | Max sequences per iteration |
| `max_num_partial_prefills` | `int` | `1` | Max concurrent partial prefills |
| `max_long_partial_prefills` | `int` | `1` | Max concurrent long partial prefills |
| `long_prefill_token_threshold` | `int` | `0` | Token count for "long" prompt classification |
| `enable_chunked_prefill` | `bool` | `True` | Enable chunked prefill |
| `is_multimodal_model` | `bool` | `False` | Whether model is multimodal |
| `policy` | `SchedulerPolicy` | `"fcfs"` | Scheduling policy: "fcfs" or "priority" |
| `disable_chunked_mm_input` | `bool` | `False` | Disable chunked multi-modal input |
| `scheduler_cls` | `str | type | None` | `None` | Scheduler class override |
| `disable_hybrid_kv_cache_manager` | `bool | None` | `None` | Disable hybrid KV cache manager |
| `scheduler_reserve_full_isl` | `bool` | `True` | Reserve full ISL before admitting |
| `async_scheduling` | `bool | None` | `None` | Enable async scheduling |

### Type Aliases

```python
RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]
```

---

## DeviceConfig

The `DeviceConfig` class (`vllm/config/device.py`) configures the compute device.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | `str` | (auto-detected) | Device type: "cuda", "cpu", "tpu", "hpu", "xpu" |

---

## AttentionConfig

The `AttentionConfig` class (`vllm/config/attention.py`) configures attention mechanisms.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `str | None` | `None` | Attention backend name override |
| `kv_cache_dtype` | `str | None` | `None` | KV cache data type override |

---

## CompilationConfig

The `CompilationConfig` class (`vllm/config/compilation.py`) configures torch.compile and CUDA graphs.

### Class: `CompilationConfig`

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `CompilationMode | None` | `None` | Compilation mode: NONE(0), STOCK_TORCH_COMPILE(1), DYNAMO_TRACE_ONCE(2), VLLM_COMPILE(3) |
| `cudagraph_mode` | `CUDAGraphMode` | (from optimization level) | CUDA graph mode: NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, FULL_AND_PIECEWISE |
| `backend` | `str` | `"inductor"` | Compilation backend: "inductor" or "eager" |
| `cudagraph_capture_sizes` | `list[int] | None` | `None` | Explicit CUDA graph capture sizes |
| `max_cudagraph_capture_size` | `int | None` | `None` | Maximum CUDA graph capture size |
| `cudagraph_num_of_warmups` | `int` | `0` | Number of CUDA graph warmup runs |
| `custom_ops` | `list[str]` | `[]` | Custom op enable/disable patterns |
| `pass_config` | `PassConfig` | `PassConfig()` | Custom Inductor pass configuration |
| `use_inductor_graph_partition` | `bool` | `False` | Use Inductor graph partitioning |
| `debug_dump_path` | `Path | None` | `None` | Path for debug dumps |
| `compile_ranges_endpoints` | `list[int] | None` | `None` | Compile range endpoints |
| `static_forward_context` | `dict` | `{}` | Static forward context (set at runtime) |
| `fast_moe_cold_start` | `bool | None` | `None` | Fast MoE cold start optimization |

### PassConfig Subclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fuse_norm_quant` | `bool | None` | `None` | Fuse RMSNorm + quant ops |
| `fuse_act_quant` | `bool | None` | `None` | Fuse SiluMul + quant ops |
| `fuse_attn_quant` | `bool | None` | `None` | Fuse Attention + quant ops |
| `eliminate_noops` | `bool` | `True` | Eliminate no-op operations |
| `enable_sp` | `bool | None` | `None` | Enable sequence parallelism |
| `fuse_gemm_comms` | `bool | None` | `None` | Enable async TP (GEMM+comm fusion) |
| `fuse_allreduce_rms` | `bool | None` | `None` | Fuse allreduce + RMSNorm |
| `fuse_minimax_qk_norm` | `bool | None` | `None` | Fuse MiniMax QK norm |
| `enable_qk_norm_rope_fusion` | `bool` | `False` | Fuse Q/K RMSNorm + RoPE |
| `fuse_act_padding` | `bool | None` | `None` | Fuse RMSNorm + padding (ROCm) |
| `fuse_mla_dual_rms_norm` | `bool | None` | `None` | Fuse MLA dual RMS norm |
| `fuse_rope_kvcache` | `bool | None` | `None` | Fuse RoPE + KV cache update |
| `rope_kvcache_fusion_max_token_num` | `int` | `256` | Max tokens for RoPE+KV fusion |
| `sp_min_token_num` | `int | None` | `None` | Minimum tokens for sequence parallelism |

### CompilationMode Enum

| Value | Description |
|-------|-------------|
| `NONE = 0` | No torch.compile, fully eager |
| `STOCK_TORCH_COMPILE = 1` | Standard torch.compile pipeline |
| `DYNAMO_TRACE_ONCE = 2` | Single Dynamo trace, no recompilation |
| `VLLM_COMPILE = 3` | Custom vLLM Inductor backend with caching |

### CUDAGraphMode Enum

| Value | Description |
|-------|-------------|
| `NONE = 0` | No CUDA graphs |
| `PIECEWISE = 1` | Piecewise CUDA graphs |
| `FULL = 2` | Full CUDA graphs |
| `FULL_DECODE_ONLY = (FULL, NONE)` | Full for decode, none for prefill |
| `FULL_AND_PIECEWISE = (FULL, PIECEWISE)` | Full for decode, piecewise for prefill |

---

## QuantizationConfig

### Class: `OnlineQuantizationConfigArgs`

```python
@config
class OnlineQuantizationConfigArgs
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `global_scheme` | `OnlineQuantScheme | None` | `None` | Quantization scheme for all layers |
| `linear_scheme_override` | `OnlineQuantScheme | None` | `None` | Override for LinearBase layers |
| `moe_scheme_override` | `OnlineQuantScheme | None` | `None` | Override for FusedMoE layers |
| `ignore` | `list[str]` | `[]` | Layers to skip (exact names or regex) |

### OnlineQuantScheme Enum

| Value | Description |
|-------|-------------|
| `FP8_PER_TENSOR` | FP8, weights and activations scaled per-tensor |
| `FP8_PER_BLOCK` | FP8, 1x128 activation / 128x128 weight blocks (DeepSeek) |
| `INT8_PER_CHANNEL_WEIGHT_ONLY` | INT8 weight-only per-channel for MoE experts |
| `MXFP8` | MXFP8, 1x32 weight blocks (microscaling) |

### Supported Quantization Methods

| Method | Description |
|--------|-------------|
| `awq` | Activation-aware Weight Quantization |
| `gptq` | GPTQ weight quantization |
| `fp8` | FP8 quantization (experimental) |
| `squeezellm` | SqueezeLLM quantization |
| `bitsandbytes` | bitsandbytes quantization |
| `gguf` | GGUF format quantization |
| `fbgemm_fp8` | FBGEMM FP8 quantization |
| `modelopt` | NVIDIA ModelOpt quantization |
| `mxfp4` | MXFP4 quantization |
| `online` | Online (runtime) quantization |
| `compressed_tensors` | Compressed Tensors format |
| `inc` | Intel Neural Compressor |
| `torchao` | TorchAO quantization |
| `experts_int8` | INT8 expert quantization for MoE |
| `humming` | Humming quantization |
| `qutlass` | QUTLASS quantization |

---

## SpeculativeConfig

The `SpeculativeConfig` class (`vllm/config/speculative.py`) configures speculative decoding.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enforce_eager` | `bool | None` | `None` | Override enforce_eager for spec decoding |
| `num_speculative_tokens` | `int` | `None` | Number of speculative tokens |
| `model` | `str | None` | `None` | Draft model name/path |
| `method` | `SpeculativeMethod | None` | `None` | Speculative method |
| `draft_tensor_parallel_size` | `int | None` | `None` | TP size for draft model |
| `prompt_lookup_max` | `int` | `None` | Max tokens for ngram lookup |
| `prompt_lookup_min` | `int` | `None` | Min tokens for ngram lookup |
| `disable_padded_drafter_batch` | `bool` | `False` | Disable padded drafter batch |

### Speculative Methods

```python
SpeculativeMethod = Literal["ngram", "medusa", "mlp_speculator", "draft_model",
    "suffix", "eagle", "eagle3", "extract_hidden_states", "ngram_gpu",
    "deepseek_mtp", "mimo_mtp", "mimo_v2_mtp", "glm4_moe_mtp",
    "glm_ocr_mtp", "ernie_mtp", "nemotron_h_mtp", "exaone_moe_mtp",
    "exaone4_5_mtp", "qwen3_next_mtp", "qwen3_5_mtp", "longcat_flash_mtp",
    "mtp", "pangu_ultra_moe_mtp", "step3p5_mtp", "hy_v3_mtp", "dflash"]
```

---

## LoRAConfig

The `LoRAConfig` class (`vllm/config/lora.py`) configures LoRA adapter support.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_lora_rank` | `MaxLoRARanks` | `16` | Maximum LoRA rank |
| `max_loras` | `int` | `1` | Max LoRAs in a single batch |
| `fully_sharded_loras` | `bool` | `False` | Fully shard LoRA with TP |
| `max_cpu_loras` | `int | None` | `None` | Max LoRAs stored in CPU memory |
| `lora_dtype` | `LoRADType` | `"auto"` | LoRA data type |
| `target_modules` | `list[str] | None` | `None` | Restrict LoRA to specific module suffixes |
| `default_mm_loras` | `dict[str, str] | None` | `None` | Modality-to-LoRA mapping |
| `enable_tower_connector_lora` | `bool` | `False` | Enable LoRA for vision encoder |
| `specialize_active_lora` | `bool` | `False` | Specialize CUDA graphs by active LoRA count |

### Type Aliases

```python
LoRADType = Literal["auto", "float16", "bfloat16"]
MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
```

---

## LoadConfig

The `LoadConfig` class (`vllm/config/load.py`) configures model loading.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `load_format` | `str` | `"auto"` | Model weight format: "auto", "pt", "safetensors", "npcache", "dummy", "gguf", "bitsandbytes", "sharded_state", "runai_streamer", etc. |
| `download_dir` | `str | None` | `None` | Directory for downloaded model files |
| `model_loader_extra_config` | `dict` | `{}` | Extra config for model loader |

---

## MultimodalConfig

The `MultimodalConfig` class (`vllm/config/multimodal.py`) configures multi-modal support.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `limit_per_prompt` | `dict[str, int] | None` | `None` | Max items per prompt per modality |
| `enable_mm_embeds` | `bool` | `False` | Enable multi-modal embedding inputs |
| `mm_encoder_tp_mode` | `MMEncoderTPMode` | (auto) | Multi-modal encoder TP mode |
| `mm_tensor_ipc` | `MMTensorIPC` | (auto) | Tensor IPC mechanism for MM data |
| `media_io_kwargs` | `dict | None` | `None` | Media IO configuration kwargs |
| `interleave_mm_strings` | `bool` | `False` | Interleave MM strings in text |
| `cache_config` | `MMCacheType | None` | `None` | MM cache configuration |

---

## ObservabilityConfig

The `ObservabilityConfig` class (`vllm/config/observability.py`) configures telemetry.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `otlp_traces_endpoint` | `str | None` | `None` | OpenTelemetry traces endpoint |
| `collect_detailed_traces` | `bool` | `False` | Collect detailed traces |

---

## KV Transfer Configs

### Class: `KVTransferConfig`

Configuration for distributed KV cache transfer between prefill and decode instances.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kv_connector` | `str | None` | `None` | KV connector class name |
| `kv_role` | `str` | (auto) | KV role: "kv_producer", "kv_consumer", "kv_both" |
| `kv_connector_extra_config` | `dict` | `{}` | Extra config for KV connector |
| `kv_buffer_size` | `float | None` | `None` | KV buffer size in GiB |
| `kv_ip` | `str` | `"127.0.0.1"` | KV transfer IP |
| `kv_port` | `int` | `None` | KV transfer port |

---

## KV Events Config

### Class: `KVEventsConfig`

Configuration for KV cache event publishing.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_kv_cache_events` | `bool` | `False` | Enable KV cache events |
| `publisher` | `str` | `"null"` | Event publisher backend |

---

## EC Transfer Config

### Class: `ECTransferConfig`

Configuration for distributed EC (expert cache) transfer.

---

## Weight Transfer Config

### Class: `WeightTransferConfig`

Configuration for weight transfer during RL training.

---

## KernelConfig

The `KernelConfig` class (`vllm/config/kernel.py`) configures kernel selection.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_flashinfer_autotune` | `bool | None` | `None` | Enable FlashInfer autotuning |
| `ir_op_priority` | `IROpPriority` | (auto) | IR operation priority configuration |

---

## MambaConfig

The `MambaConfig` class (`vllm/config/mamba.py`) configures Mamba SSM layers.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_stochastic_rounding` | `bool` | `False` | Enable stochastic rounding for Mamba cache |

---

## PoolerConfig

The `PoolerConfig` class (`vllm/config/pooler.py`) configures the pooling output layer.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seq_pooling_type` | `str | None` | `None` | Sequence pooling type: "MEAN", "LAST", "STEP", etc. |
| `use_activation` | `bool | None` | `None` | Apply activation function |
| `head_dtype` | `str | None` | `None` | Head output dtype |
| `head_dim` | `int | None` | `None` | Head output dimension |
| `task` | `PoolingTask | None` | `None` | Pooling task type |
| `dimensions` | `int | None` | `None` | Output dimensions (Matryoshka) |

---

## ProfilerConfig

The `ProfilerConfig` class (`vllm/config/profiler.py`) configures profiling.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `profiled_locations` | `list | None` | `None` | Locations to profile |
| `profiler_output_dir` | `str | None` | `None` | Profiler output directory |

---

## ReasoningConfig

The `ReasoningConfig` class (`vllm/config/reasoning.py`) configures reasoning model support.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reasoning_parser` | `str | None` | `None` | Reasoning content parser name |
| `enabled` | `bool` | `False` | Whether reasoning is enabled |
| `reasoning_start_str` | `str | None` | `None` | Start marker for reasoning |
| `reasoning_end_str` | `str | None` | `None` | End marker for reasoning |

---

## StructuredOutputsConfig

The `StructuredOutputsConfig` class (`vllm/config/structured_outputs.py`) configures structured output generation.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `str` | `"auto"` | Backend: "auto", "xgrammar", "outlines", "guidance", "lm-format-enforcer" |
| `disable_by_batch_size` | `int | None` | `None` | Disable when batch exceeds this size |
| `reasoning_parser` | `str | None` | `None` | Reasoning parser for structured thinking |
| `enable_in_reasoning` | `bool` | `False` | Enable structured outputs in reasoning |

---

## SpeechToTextConfig

Configuration for speech-to-text models.

---

## OffloadConfig

The `OffloadConfig` class (`vllm/config/offload.py`) configures model weight offloading.

### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cpu_offload_gb` | `float` | `0` | CPU memory for weight offloading (GiB) |
| `offload_group_size` | `int` | `0` | Prefetch offloading group size |
| `offload_num_in_group` | `int` | `1` | Layers to offload per group |
| `offload_prefetch_step` | `int` | `1` | Prefetch lookahead steps |
| `offload_params` | `set[str]` | `set()` | Parameter name segments to offload |

---

## ModelArchitectureConfig

Configuration for model architecture-specific behavior. Different architectures
may have custom config classes.

---

## Optimization Levels

### Class: `OptimizationLevel`

| Level | Startup Time | Performance | Description |
|-------|-------------|-------------|-------------|
| `O0` | Fastest | Lowest | No compilation, no CUDA graphs, no optimizations |
| `O1` | Fast | Good | Dynamo+Inductor compilation, Piecewise CUDA graphs |
| `O2` | Moderate | Best (default) | Full + Piecewise CUDA graphs, all optimizations |
| `O3` | Moderate | Best | Same as O2 + FlashInfer autotuning |

### Performance Modes

| Mode | Description |
|------|-------------|
| `"balanced"` | Default. Good balance of latency and throughput |
| `"interactivity"` | Low latency at small batches. Fine-grained CUDA graphs |
| `"throughput"` | High aggregate tokens/sec at high concurrency |

---

## Helper Types and Utilities

### Types

```python
PerformanceMode = Literal["balanced", "interactivity", "throughput"]
HfOverrides = dict[str, Any] | Callable[[PretrainedConfig], PretrainedConfig]
```

### Decorator: `@config`

The `@config` decorator (`vllm/config/utils.py`) transforms a Pydantic dataclass into
a vLLM config class that:
- Supports CLI argument generation
- Computes hashes for computation graph identification
- Integrates with the VllmConfig system

### Function: `replace(config_obj, **kwargs)`

Returns a new config object with specified fields replaced (immutable update).

### Function: `is_init_field(cls, field_name) -> bool`

Checks whether a field is an init field (not computed/derived).
