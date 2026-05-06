# ONNX Runtime Reference - Chapter 5: Python API - Configuration

Complete reference for SessionOptions, RunOptions, and all configuration keys.

---

## 5.1 SessionOptions

```python
import onnxruntime as ort

opts = ort.SessionOptions()
```

### 5.1.1 Threading Configuration

```python
# Intra-op threads: parallelize within a single operator
opts.intra_op_num_threads = 4    # 0 = use all logical cores

# Inter-op threads: parallelize across independent operators
opts.inter_op_num_threads = 1    # 0 = use all logical cores

# Execution mode
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Sequential execution
opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL    # Parallel execution
```

### 5.1.2 Graph Optimization

```python
# Optimization level
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL    # No optimizations
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC   # Level 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED # Level 2
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL     # All levels
```

**Level 1 - Basic Optimizations:**
- Constant folding
- Dead code elimination
- Redundant node elimination (Identity, Noop, Dropout, Slice)
- Semantics-preserving node fusion

**Level 2 - Extended Optimizations:**
- All Level 1 optimizations
- Complex operator fusion (Conv+BN, Gemm+Activation, Attention)
- Embedding layer fusion
- Bias+Gelu fusion

**Level 3 - Layout Optimizations:**
- All Level 2 optimizations
- NCHW → NHWC layout transformation
- NCHWc format optimization

**Level 99 - All Optimizations:**
- All Level 3 optimizations
- QDQ (QuantizeLinear/DequantizeLinear) handling
- MatMulNBits conversion
- BFloat16 conversion

### 5.1.3 Memory Configuration

```python
# Memory pattern optimization (pre-allocate memory for known patterns)
opts.enable_mem_pattern = True    # Default: True
opts.enable_mem_reuse = True      # Default: True

# Disable for large models or variable batch sizes
opts.enable_mem_pattern = False
```

### 5.1.4 Profiling

```python
opts.enable_profiling = True
opts.profile_file_prefix = "ort_profile_"
```

### 5.1.5 Optimized Model Export

```python
# Save optimized model after graph optimizations
opts.optimized_model_filepath = "model_optimized.onnx"

# Save in ORT format
opts.add_session_config_entry("session.save_model_format", "ORT")
opts.optimized_model_filepath = "model_optimized.ort"
```

### 5.1.6 Logging

```python
opts.session_log_id = "my_session"
opts.session_log_severity_level = 2    # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
opts.session_log_verbosity_level = 0   # Verbosity when severity is Verbose
```

### 5.1.7 Custom Operators

```python
# Register custom ops from shared library
opts.register_custom_ops_library("my_custom_ops.so")

# Register using V2 API
opts.register_custom_ops_library("my_custom_ops.so")
```

---

## 5.2 Complete Session Config Keys Reference

All configuration keys passed via `opts.add_session_config_entry(key, value)`:

### 5.2.1 Session Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `session.disable_prepacking` | "0"/"1" | "0" | Disable pre-packing of constant initializers |
| `session.use_env_allocators` | "0"/"1" | "0" | Use environment-level allocators |
| `session.load_model_format` | "ONNX"/"ORT" | auto | Force model format on load |
| `session.save_model_format` | "ONNX"/"ORT" | auto | Force model format when saving |
| `session.set_denormal_as_zero` | "0"/"1" | "0" | Flush denormals to zero |
| `session.use_device_allocator_for_initializers` | "0"/"1" | "0" | Use device allocator for initializers |
| `session.use_ort_model_bytes_directly` | "0"/"1" | "0" | Use model bytes directly (no copy) |
| `session.use_ort_model_bytes_for_initializers` | "0"/"1" | "0" | Use model bytes for initializers |
| `session.use_memory_mapped_ort_model` | "0"/"1" | "0" | Memory-map ORT model file |
| `session.disable_cpu_ep_fallback` | "0"/"1" | "0" | Disable CPU EP fallback |
| `session.strict_shape_type_inference` | "0"/"1" | "0" | Fail on shape/type inconsistencies |
| `session.allow_released_opsets_only` | "0"/"1" | "0" | Fail on unreleased opsets |
| `session.dynamic_block_base` | positive int | "" | Dynamic block-sizing for multithreading |
| `session.force_spinning_stop` | "0"/"1" | "0" | Stop thread spinning between runs |

### 5.2.2 Optimization Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `optimization.enable_gelu_approximation` | "0"/"1" | "0" | Enable GELU approximation |
| `optimization.enable_cast_chain_elimination` | "0"/"1" | "0" | Enable Cast chain elimination |
| `optimization.disable_specified_optimizers` | comma-separated | "" | Disable named optimizers |
| `optimization.minimal_build_optimizations` | "save"/"apply"/"" | "" | Minimal build optimization mode |
| `session.graph_optimizations_loop_level` | "0"/"1"/"2" | "1" | Graph optimization loop level |

### 5.2.3 Quantization Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `session.disable_quant_qdq` | "0"/"1" | "0" | Disable QDQ fusion |
| `session.disable_qdq_constant_folding` | "0"/"1" | "0" | Prevent DQ constant folding |
| `session.disable_double_qdq_remover` | "0"/"1" | "0" | Disable double QDQ removal |
| `session.enable_quant_qdq_cleanup` | "0"/"1" | "0" | Enable QDQ cleanup after handling |
| `session.qdqisint8allowed` | "0"/"1" | "1" | Allow INT8 in QDQ |
| `session.x64quantprecision` | "0"/"1" | "0" | Use U8U8 on AVX2/AVX512 |
| `session.qdq_matmulnbits_accuracy_level` | int | "4" | MatMulNBits accuracy level |
| `session.qdq_matmulnbits_block_size` | int | "32" | MatMulNBits block size |
| `session.enable_dq_matmulnbits_fusion` | "0"/"1" | "0" | Enable DQ→MatMulNBits fusion |

### 5.2.4 Threading Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `session.intra_op.allow_spinning` | "0"/"1" | "1" | Allow intra-op thread spinning |
| `session.inter_op.allow_spinning` | "0"/"1" | "1" | Allow inter-op thread spinning |
| `session.intra_op.spin_duration_us` | int | default | Spin duration in microseconds |
| `session.inter_op.spin_duration_us` | int | default | Spin duration in microseconds |
| `session.intra_op.spin_backoff_max` | int | "1" | Exponential backoff cap |
| `session.inter_op.spin_backoff_max` | int | "1" | Exponential backoff cap |
| `session.intra_op_thread_affinities` | string | "" | Thread affinity string |

### 5.2.5 EP Context Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `ep.context_enable` | "0"/"1" | "0" | Enable EP context model |
| `ep.context_file_path` | path | auto | EP context file path |
| `ep.context_embed_mode` | "0"/"1" | "0" | Embed EP context in ONNX |
| `ep.context_node_name_prefix` | string | "" | EPContext node name prefix |
| `ep.share_ep_contexts` | "0"/"1" | "0" | Share EP contexts across sessions |
| `ep.stop_share_ep_contexts` | "0"/"1" | "0" | Stop sharing EP contexts |
| `ep.enable_weightless_ep_context_nodes` | "0"/"1" | "0" | Enable weightless EP context nodes |

### 5.2.6 Memory and Model Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `session.optimized_model_external_initializers_file_name` | path | "" | External initializers file |
| `session.optimized_model_external_initializers_min_size_in_bytes` | int | "" | Min initializer size for external |
| `session.model_external_initializers_file_folder_path` | path | "" | External data file folder |
| `session.save_external_prepacked_constant_initializers` | "0"/"1" | "0" | Save pre-packed initializers externally |
| `session.collect_node_memory_stats_to_file` | path | "" | File path for node memory stats |
| `session.resource_cuda_partitioning_settings` | CSV | "" | CUDA capacity-aware partitioning |
| `session.layer_assignment_settings` | string | "" | Layer assignment configuration |
| `session.node_partition_config_file` | path | "" | Node partition config file |
| `session.record_ep_graph_assignment_info` | "0"/"1" | "0" | Record EP graph assignment info |
| `session.disable_model_compile` | "0"/"1" | "0" | Disable EP model compilation |
| `session.fail_on_suboptimal_compiled_model` | "0"/"1" | "0" | Fail on suboptimal compiled model |

### 5.2.7 MLAS Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `mlas.enable_gemm_fastmath_arm64_bfloat16` | "0"/"1" | "0" | Enable BF16 GEMM fast math (ARM64) |
| `mlas.use_lut_gemm` | "0"/"1" | "0" | Use LUT-based GEMM |
| `mlas.disable_kleidiai` | "0"/"1" | "0" | Disable KleidiAI kernels |

### 5.2.8 Debug Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `session.debug_layout_transformation` | "0"/"1" | "0" | Dump model after layout transforms |

### 5.2.9 EP-Specific Configuration

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `ep.nnapi.partitioning_stop_ops` | comma-separated | default | NNAPI stop ops |
| `session.disable_aot_function_inlining` | "0"/"1" | "0" | Disable AOT function inlining |

---

## 5.3 RunOptions

```python
run_opts = ort.RunOptions()

# Logging
run_opts.log_severity_level = 2        # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
run_opts.log_verbosity_level = 0       # Verbosity when severity is Verbose
run_opts.log_tag = "inference_run"     # Tag for profiling

# Termination
run_opts.add_run_config_entry(" ort.terminategnalign", "1")  # Request termination
```

---

## 5.4 ThreadingOptions

```python
tp = ort.ThreadingOptions()
tp.global_intra_op_num_threads = 4
tp.global_inter_op_num_threads = 1
tp.global_spin_control = 1  # Allow spinning
```

---

## 5.5 Environment Configuration

```python
import onnxruntime as ort

# Set language projection (for telemetry)
ort.set_default_logger_severity(2)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

# Get version
print(ort.__version__)

# Get available providers
print(ort.get_available_providers())
# Returns: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# Get device ID
print(ort.get_device())  # Returns: 0 (GPU) or -1 (CPU only)
```

---

## 5.6 Complete Configuration Example

```python
import onnxruntime as ort

# Create session options
opts = ort.SessionOptions()

# Threading
opts.intra_op_num_threads = 4
opts.inter_op_num_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Optimization
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Memory
opts.enable_mem_pattern = True
opts.enable_mem_reuse = True

# Profiling
opts.enable_profiling = True
opts.profile_file_prefix = "profile_"

# Logging
opts.session_log_severity_level = 2
opts.session_log_verbosity_level = 0

# Save optimized model
opts.optimized_model_filepath = "model_optimized.onnx"
opts.add_session_config_entry("session.save_model_format", "ORT")

# Disable specific optimizations
opts.add_session_config_entry("optimization.disable_specified_optimizers",
    "GeluFusion,BiasGeluFusion")

# Enable GELU approximation
opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")

# Disable CPU fallback
opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

# Thread spinning
opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
opts.add_session_config_entry("session.inter_op.allow_spinning", "0")

# QDQ handling
opts.add_session_config_entry("session.disable_quant_qdq", "0")
opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")

# Create session with CUDA
providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
    }),
    "CPUExecutionProvider",
]

sess = ort.InferenceSession("model.onnx", sess_options=opts, providers=providers)
```
