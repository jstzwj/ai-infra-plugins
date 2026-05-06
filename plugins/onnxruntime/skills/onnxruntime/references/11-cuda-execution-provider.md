# ONNX Runtime Reference - Chapter 11: CUDA Execution Provider

Complete reference for the NVIDIA CUDA Execution Provider - GPU acceleration for ONNX Runtime.

---

## 11.1 Overview

The CUDA EP accelerates ONNX model inference on NVIDIA GPUs using CUDA, cuDNN, and cuBLAS libraries.

**Requirements:**
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.8+ or 12.x
- cuDNN 8.x+
- cuBLAS

---

## 11.2 Provider Options

### 11.2.1 Python Configuration

```python
providers = [
    ("CUDAExecutionProvider", {
        # Device configuration
        "device_id": 0,                              # GPU device ID

        # Memory management
        "arena_extend_strategy": "kNextPowerOfTwo",  # Arena extension strategy
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,    # GPU memory limit (bytes)
        "use_cudnn_conv_algo_search": True,          # Deprecated, use cudnn_conv_algo_search

        # cuDNN configuration
        "cudnn_conv_algo_search": "EXHAUSTIVE",       # EXHAUSTIVE|HEURISTIC|DEFAULT
        "cudnn_conv_use_max_workspace": False,        # Use max workspace for cuDNN conv
        "use_cudnn_conv": True,                       # Use cuDNN for convolutions

        # Stream management
        "do_copy_in_default_stream": True,            # Copy in default stream
        "use_ep_level_unified_stream": False,         # EP-level unified stream

        # CUDA Graph
        "enable_cuda_graph": False,                   # Enable CUDA graph capture

        # Layout
        "enable_skip_layout_transform": False,        # Skip NHWC↔NCHW transforms
        "prefer_nhwc": False,                         # Prefer NHWC layout

        # Precision
        "use_tf32": True,                             # Use TF32 for matmul
        "use_cublas_lt_gemm": False,                  # Enable cuBLASLt GEMM

        # Quantization
        "use_blockwise_quantization": True,           # Use blockwise quantization

        # Tunable operators
        "enable_cuda_tunable_op": False,              # Enable tunable operators

        # External stream
        "user_compute_stream": None,                  # External CUDA stream (void*)
    }),
    "CPUExecutionProvider",
]

sess = ort.InferenceSession("model.onnx", providers=providers)
```

### 11.2.2 C++ Configuration

```cpp
Ort::CUDAProviderOptionsV2 cuda_options;

std::vector<const char*> keys = {
    "device_id",
    "arena_extend_strategy",
    "gpu_mem_limit",
    "cudnn_conv_algo_search",
    "enable_cuda_graph",
};

std::vector<const char*> values = {
    "0",
    "kNextPowerOfTwo",
    "2147483648",
    "EXHAUSTIVE",
    "0",
};

cuda_options.Update(keys, values);

Ort::SessionOptions opts;
opts.AppendExecutionProvider_CUDA(cuda_options);
```

---

## 11.3 Arena Extension Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `kNextPowerOfTwo` | Extend to next power of 2 (default) | General use, better for variable sizes |
| `kSameAsRequested` | Extend by exact requested size | Memory-constrained environments |

---

## 11.4 cuDNN Convolution Algorithm Search

| Mode | Description | Speed | Coverage |
|------|-------------|-------|----------|
| `EXHAUSTIVE` | Try all algorithms, pick fastest | Slow first run, fast inference | Best |
| `HEURISTIC` | Use heuristics to pick algorithm | Medium first run | Good |
| `DEFAULT` | Use default algorithm | Fast first run | Basic |

---

## 11.5 CUDA Graph Support

```python
# Enable CUDA graph for repeated inference with same input shape
providers = [
    ("CUDAExecutionProvider", {
        "enable_cuda_graph": True,
    }),
]

opts = ort.SessionOptions()
sess = ort.InferenceSession("model.onnx", sess_options=opts, providers=providers)

# Warmup run (captures CUDA graph)
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
sess.run(None, {"input": input_data})

# Subsequent runs use captured CUDA graph (much faster)
sess.run(None, {"input": input_data})
```

---

## 11.6 Multi-GPU Support

```python
# Use specific GPU
providers = [
    ("CUDAExecutionProvider", {"device_id": 0}),  # GPU 0
]

# Use multiple sessions on different GPUs
sess0 = ort.InferenceSession("model.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": 0})])
sess1 = ort.InferenceSession("model.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": 1})])
```

---

## 11.7 Memory Management

```python
# Limit GPU memory usage
providers = [
    ("CUDAExecutionProvider", {
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
        "arena_extend_strategy": "kSameAsRequested",  # Precise allocation
    }),
]

# Check CUDA memory availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Memory allocated: {torch.cuda.memory_allocated()}")
print(f"Memory reserved: {torch.cuda.memory_reserved()}")
```

---

## 11.8 NHWC Layout Support

```python
# Prefer NHWC layout for better GPU performance
providers = [
    ("CUDAExecutionProvider", {
        "prefer_nhwc": True,
    }),
]

# Skip layout transformations if model is already NHWC
providers = [
    ("CUDAExecutionProvider", {
        "enable_skip_layout_transform": True,
    }),
]
```

---

## 11.9 Mixed Precision

```python
# FP16 inference (model must be FP16 or auto-converted)
# ORT automatically uses FP16 kernels when available

# TF32 for Ampere+ GPUs
providers = [
    ("CUDAExecutionProvider", {
        "use_tf32": True,  # Default on Ampere+
    }),
]

# cuBLASLt for better FP16/BF16 performance
providers = [
    ("CUDAExecutionProvider", {
        "use_cublas_lt_gemm": True,
    }),
]
```

---

## 11.10 Build Requirements

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| CUDA Toolkit | 11.8 | 12.x |
| cuDNN | 8.2 | 8.9+ |
| cuBLAS | 11.x | 12.x |
| Python | 3.8+ | 3.11+ |
| GCC | 9+ | 11+ |
| CMake | 3.28+ | 3.28+ |

```bash
# Build with CUDA
python build.sh --config Release --use_cuda --cuda_version 12.2 \
    --cuda_home /usr/local/cuda --cudnn_home /usr/local/cudnn
```

---

## 11.11 Supported Operators

CUDA EP supports the full ONNX operator set with GPU-accelerated implementations:
- Arithmetic: Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log
- Neural: Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool
- Activation: Relu, Sigmoid, Tanh, Gelu, FastGelu, LeakyRelu, Selu
- Normalization: BatchNorm, LayerNorm, InstanceNorm, GroupNorm
- Linear: MatMul, Gemm, MatMulInteger
- Attention: MultiHeadAttention, Attention
- Reduction: ReduceMean, ReduceSum, ReduceMax, ReduceMin
- Transform: Reshape, Transpose, Concat, Split, Gather, ScatterND, Expand
- Comparison: Equal, NotEqual, Greater, Less
- Logical: And, Or, Xor, Not
- Quantized: QLinearConv, QLinearMatMul, MatMulInteger
- And many more...

Plus contrib ops: BiasGelu, EmbedLayerNorm, Attention, SkipLayerNorm, FastGelu
