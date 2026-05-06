# ONNX Runtime Reference - Chapter 4: Python API - InferenceSession

Complete reference for the ONNX Runtime Python InferenceSession API.

---

## 4.1 Installation

```bash
# CPU only
pip install onnxruntime

# GPU (CUDA)
pip install onnxruntime-gpu

# Training support
pip install onnxruntime-training

# DirectML (Windows)
pip install onnxruntime-directml

# Specific version
pip install onnxruntime==1.22.0
```

---

## 4.2 InferenceSession

```python
import onnxruntime as ort

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: Union[str, bytes, os.PathLike],
        sess_options: Optional[SessionOptions] = None,
        providers: Optional[Sequence[Union[str, Tuple[str, Dict]]]] = None,
        provider_options: Optional[Sequence[Dict]] = None,
        **kwargs
    ):
        """
        Create an inference session.

        Args:
            path_or_bytes: Path to ONNX model file OR model bytes
            sess_options: SessionOptions for configuration
            providers: List of provider names or (name, options) tuples
            provider_options: List of provider option dicts
        """
```

### 4.2.1 Session Creation

```python
import onnxruntime as ort
import numpy as np

# Basic creation (CPU only)
sess = ort.InferenceSession("model.onnx")

# With execution providers
sess = ort.InferenceSession("model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# With provider options
sess = ort.InferenceSession("model.onnx",
    providers=[
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
        }),
        "CPUExecutionProvider",
    ])

# From bytes
with open("model.onnx", "rb") as f:
    model_bytes = f.read()
sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

# With session options
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("model.onnx", sess_options=opts,
                             providers=["CUDAExecutionProvider"])
```

### 4.2.2 Running Inference

```python
# Basic run with numpy arrays
results = sess.run(
    output_names=None,           # None = all outputs
    input_feed={"input": input_array},
    run_options=None
)

# Specific outputs
results = sess.run(
    output_names=["output1", "output2"],
    input_feed={"input": input_array}
)

# Multiple inputs
results = sess.run(
    None,
    input_feed={
        "input_ids": ids_array,
        "attention_mask": mask_array,
        "position_ids": pos_array,
    }
)

# With RunOptions
run_opts = ort.RunOptions()
run_opts.log_severity_level = 3
run_opts.log_verbosity_level = 0
run_opts.add_run_config_entry("enable_mem_pattern", "1")
results = sess.run(None, {"input": input_array}, run_options=run_opts)
```

### 4.2.3 Run with IO Binding

```python
# Create IO binding
io_binding = sess.io_binding()

# Bind inputs
io_binding.bind_cpu_input("input", input_array)

# Bind OrtValue inputs (GPU)
ort_value = ort.OrtValue.ortvalue_from_numpy(input_array, "cuda", 0)
io_binding.bind_ortvalue_input("input", ort_value)

# Bind outputs
io_binding.bind_output("output", "cuda", 0)  # GPU output
# or
io_binding.bind_output("output")  # CPU output

# Run
sess.run_with_iobinding(io_binding)

# Get outputs
outputs = io_binding.copy_outputs_to_cpu()
```

---

## 4.3 Model Introspection

### 4.3.1 Input/Output Information

```python
# Get input info
inputs = sess.get_inputs()
for inp in inputs:
    print(f"Name: {inp.name}")
    print(f"Type: {inp.type}")           # e.g., 'tensor(float)'
    print(f"Shape: {inp.shape}")         # e.g., [1, 3, 224, 224] or ['batch', 3, 'H', 'W']

# Quick access
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape

# Get output info
outputs = sess.get_outputs()
for out in outputs:
    print(f"Name: {out.name}")
    print(f"Type: {out.type}")
    print(f"Shape: {out.shape}")

# Get overridable initializers (parameters)
initializers = sess.get_overridable_initializers()
for init in initializers:
    print(f"Name: {init.name}, Shape: {init.shape}, Type: {init.type}")
```

### 4.3.2 Model Metadata

```python
metadata = sess.get_modelmeta()
print(f"Producer: {metadata.producer_name}")
print(f"Graph name: {metadata.graph_name}")
print(f"Description: {metadata.description}")
print(f"Domain: {metadata.domain}")
print(f"Version: {metadata.version}")
print(f"Custom metadata: {metadata.custom_metadata_map}")
```

### 4.3.3 Provider Information

```python
# Get available providers
providers = sess.get_providers()
# Returns: ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Get provider options
provider_options = sess.get_provider_options()
# Returns: {'CUDAExecutionProvider': {...}, 'CPUExecutionProvider': {...}}

# Set providers after creation
sess.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

# Set providers with options
sess.set_providers(
    providers=["CUDAExecutionProvider"],
    provider_options=[{"device_id": 0}]
)
```

### 4.3.4 Session Options

```python
# Get current session options
opts = sess.get_session_options()
print(f"Intra-op threads: {opts.intra_op_num_threads}")
print(f"Inter-op threads: {opts.inter_op_num_threads}")
```

### 4.3.5 Profiling

```python
# Enable profiling via SessionOptions
opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "ort_profile"

sess = ort.InferenceSession("model.onnx", sess_options=opts)
results = sess.run(None, {"input": input_array})

# End profiling and get file path
profile_path = sess.end_profiling()
print(f"Profile saved to: {profile_path}")
```

---

## 4.4 Input/Output Handling

### 4.4.1 Numpy Array Inputs

```python
import numpy as np

# Float32 input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
results = sess.run(None, {"input": input_data})

# Int64 input
input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
attention_mask = np.array([[1, 1, 1, 1]], dtype=np.int64)
results = sess.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask
})

# Multiple outputs
output1, output2 = sess.run(
    ["last_hidden_state", "pooler_output"],
    {"input_ids": input_ids}
)
```

### 4.4.2 String Inputs

```python
# String tensor input
text_data = np.array([["hello", "world"]], dtype=object)
results = sess.run(None, {"text_input": text_data})
```

### 4.4.3 Overriding Initializers

```python
# Override model weights at runtime
import numpy as np

weight_data = np.random.randn(768, 768).astype(np.float32)
results = sess.run(
    None,
    {"input": input_data},
    {"transformer.weight": weight_data}  # Override initializer
)
```

---

## 4.5 Execution Provider Configuration

### 4.5.1 CUDA EP

```python
providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
        "cudnn_conv_use_max_workspace": True,
        "enable_cuda_graph": False,
        "enable_skip_layout_transform": False,
        "enable_cublas_lt_gemm": True,
        "prefer_nhwc": False,
        "use_cublas_lt_for_fp16_ckpt_gemm": True,
        "use_tf32": True,
        "use_cudnn_conv": True,
        "use_blockwise_quantization": True,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.2 TensorRT EP

```python
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_max_workspace_size": 1 << 30,  # 1GB
        "trt_fp16_enable": True,
        "trt_int8_enable": False,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
        "trt_builder_opt_level": 3,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "./trt_cache/timing_cache.bin",
        "trt_min_subgraph_size": 1,
        "trt_max_partition_iterations": 1000,
        "trt_dump_subgraphs": False,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.3 OpenVINO EP

```python
providers = [
    ("OpenVINOExecutionProvider", {
        "device_type": "CPU",       # CPU, GPU, NPU, AUTO, HETERO, MULTI
        "enable_opencl_throttling": False,
        "enable_dynamic_shapes": True,
        "device_id": "",
        "num_of_threads": 0,        # 0 = auto
        "cache_dir": "./ov_cache",
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.4 CoreML EP

```python
providers = [
    ("CoreMLExecutionProvider", {
        "use_cpu_only": False,
        "enable_on_subgraph": True,
        "coreml_flags": 0,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.5 NNAPI EP

```python
providers = [
    ("NNAPIExecutionProvider", {
        "use_fp16": False,
        "use_nchwc": True,
        "use_cpu_only": False,
        "partitioning_stop_ops": "",
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.6 DNNL EP

```python
providers = [
    ("DnnlExecutionProvider", {
        "use_arena": True,
        "use_pinned_mem": True,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.7 DirectML EP

```python
providers = [
    ("DmlExecutionProvider", {
        "device_id": 0,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.8 WebGPU EP

```python
providers = [
    ("WebGPUExecutionProvider", {
        "device_id": 0,
        "enable_graph_capture": False,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.9 QNN EP

```python
providers = [
    ("QNNExecutionProvider", {
        "backend_path": "libQnnHtp.so",
        "htp_performance_mode": "default",
        "soc_model": "SM8550",
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 4.5.10 XNNPACK EP

```python
providers = [
    ("XnnpackExecutionProvider", {
        "num_threads": 4,
        "flags": 0,
    }),
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

---

## 4.6 Session Fallback Behavior

```python
# Enable provider fallback (default)
sess = ort.InferenceSession("model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# If CUDA fails, CPU is used automatically

# Check which providers are actually active
active_providers = sess.get_providers()
# e.g., ['CPUExecutionProvider'] if CUDA failed

# Disable fallback (require CUDA)
opts = ort.SessionOptions()
opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession("model.onnx", sess_options=opts,
                             providers=["CUDAExecutionProvider"])
# Will raise if CUDA is not available
```

---

## 4.7 Advanced Patterns

### 4.7.1 Batched Inference

```python
# Process multiple inputs efficiently
batch_size = 32
input_batch = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
results = sess.run(None, {"input": input_batch})
```

### 4.7.2 Dynamic Shapes

```python
# Model with dynamic batch dimension
input_dynamic = np.random.randn(8, 3, 224, 224).astype(np.float32)
results = sess.run(None, {"input": input_dynamic})

input_dynamic2 = np.random.randn(1, 3, 224, 224).astype(np.float32)
results2 = sess.run(None, {"input": input_dynamic2})
```

### 4.7.3 Multi-Threaded Inference

```python
import threading
import numpy as np

def inference_worker(session, input_data, thread_id):
    results = session.run(None, {"input": input_data})
    print(f"Thread {thread_id}: output shape = {results[0].shape}")

# Create session with thread-safe options
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess = ort.InferenceSession("model.onnx", sess_options=opts)

# Session is thread-safe for Run()
threads = []
for i in range(4):
    data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    t = threading.Thread(target=inference_worker, args=(sess, data, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

### 4.7.4 Async Inference (Python)

```python
import asyncio
import numpy as np
import onnxruntime as ort

async def async_inference(session, input_data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, session.run, None, {"input": input_data}
    )
    return result
```

### 4.7.5 Memory Optimization

```python
opts = ort.SessionOptions()
opts.enable_mem_pattern = True      # Enable memory pattern optimization
opts.enable_mem_reuse = True        # Enable memory reuse
opts.add_session_config_entry("session.set_denormal_as_zero", "1")

# For large models, disable memory patterns to reduce peak memory
opts.enable_mem_pattern = False

# Use memory-mapped ORT format models
opts.add_session_config_entry("session.use_memory_mapped_ort_model", "1")
```
