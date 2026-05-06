# ONNX Runtime Reference - Chapters 37-42: Build, Deploy, and Tools

---

## 37. Build System (CMake)

### Minimum Requirements
- CMake 3.28+
- C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+)
- Python 3.8+

### Core CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `onnxruntime_BUILD_SHARED_LIB` | OFF | Build shared library |
| `onnxruntime_BUILD_UNIT_TESTS` | ON | Build unit tests |
| `onnxruntime_ENABLE_PYTHON` | OFF | Build Python bindings |
| `onnxruntime_ENABLE_CUDA` | OFF | Enable CUDA EP |
| `onnxruntime_USE_TENSORRT` | OFF | Enable TensorRT EP |
| `onnxruntime_USE_OPENVINO` | OFF | Enable OpenVINO EP |
| `onnxruntime_USE_DNNL` | OFF | Enable oneDNN EP |
| `onnxruntime_USE_COREML` | OFF | Enable CoreML EP |
| `onnxruntime_USE_DML` | OFF | Enable DirectML EP |
| `onnxruntime_USE_WEBGPU` | OFF | Enable WebGPU EP |
| `onnxruntime_USE_QNN` | OFF | Enable QNN EP |
| `onnxruntime_USE_XNNPACK` | OFF | Enable XNNPACK EP |
| `onnxruntime_ENABLE_TRAINING` | OFF | Enable training support |
| `onnxruntime_ENABLE_LORA` | OFF | Enable LoRA support |
| `onnxruntime_MINIMAL_BUILD` | OFF | Minimal build |
| `onnxruntime_DISABLE_CONTRIB_OPS` | OFF | Disable contrib ops |
| `onnxruntime_USE_FLASH_ATTENTION` | OFF | Flash Attention kernels |
| `onnxruntime_ENABLE_LTO` | OFF | Link-time optimization |
| `onnxruntime_BUILD_FOR_NATIVE_MACHINE` | OFF | Optimize for host CPU |

### Build Commands

```bash
# CPU-only build
python build.sh --config Release --parallel

# With CUDA
python build.sh --config Release --use_cuda --cuda_version 12.2 \
    --cuda_home /usr/local/cuda --cudnn_home /usr/local/cudnn

# Python wheel
python build.sh --config Release --enable_pybind --build_wheel

# WebAssembly
python build.sh --config Release --build_wasm

# Android
python build.sh --config Release --android --android_sdk_path /path/to/sdk \
    --android_ndk_path /path/to/ndk --android_abi arm64-v8a

# iOS
python build.sh --config Release --ios --ios_sysroot iphoneos \
    --ios_arch arm64
```

### Cross-Compilation
```bash
# ARM64 Linux
python build.sh --config Release --cmake_extra_defines \
    CMAKE_TOOLCHAIN_FILE=cmake/linux_arm64_crosscompile_toolchain.cmake

# Windows ARM64
python build.sh --config Release --cmake_extra_defines \
    CMAKE_SYSTEM_NAME=Windows CMAKE_SYSTEM_PROCESSOR=ARM64
```

---

## 38. Docker and Container Deployment

### Available Images
```dockerfile
# CUDA GPU inference
FROM mcr.microsoft.com/onnxruntime/server:latest-gpu

# CPU inference
FROM mcr.microsoft.com/onnxruntime/server:latest

# TensorRT
FROM mcr.microsoft.com/onnxruntime/server:latest-tensorrt
```

### Docker Build
```bash
# Build CUDA image
docker build -f dockerfiles/Dockerfile.cuda -t ort-cuda .

# Build TensorRT image
docker build -f dockerfiles/Dockerfile.tensorrt -t ort-tensorrt .
```

### Running with GPU
```bash
docker run --gpus all -v /path/to/models:/models ort-cuda \
    python inference.py --model /models/model.onnx
```

---

## 39. Profiling and Performance

### Enabling Profiling
```python
opts = ort.SessionOptions()
opts.enable_profiling = True
opts.profile_file_prefix = "ort_profile"

sess = ort.InferenceSession("model.onnx", sess_options=opts)
results = sess.run(None, {"input": input_data})
profile_file = sess.end_profiling()
# Open profile_file in chrome://tracing
```

### Performance Tuning Tips

1. **Thread Count**: Set `intra_op_num_threads` to number of physical cores
2. **Optimization Level**: Use `ORT_ENABLE_ALL` for best performance
3. **Memory Patterns**: Enable `enable_mem_pattern` for fixed-shape models
4. **CUDA Workspace**: Use `cudnn_conv_algo_search: EXHAUSTIVE` for best conv perf
5. **IO Binding**: Use IO Binding for GPU inference to avoid CPU↔GPU copies
6. **Batch Size**: Larger batches amortize kernel launch overhead
7. **Quantization**: Use INT8 quantization for 2-4x speedup with minimal accuracy loss
8. **Model Format**: Use ORT format (.ort) for faster load times

### Benchmarking
```python
import time
import numpy as np

# Warmup
for _ in range(10):
    sess.run(None, {"input": input_data})

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    sess.run(None, {"input": input_data})
    times.append(time.perf_counter() - start)

print(f"Mean: {np.mean(times)*1000:.2f} ms")
print(f"P50:  {np.percentile(times, 50)*1000:.2f} ms")
print(f"P99:  {np.percentile(times, 99)*1000:.2f} ms")
```

---

## 40. Mobile Deployment

### Android (NNAPI + XNNPACK)
```python
# Build Android AAR
python build.sh --config Release --android \
    --android_sdk_path $ANDROID_SDK_ROOT \
    --android_ndk_path $ANDROID_NDK_ROOT \
    --android_abi arm64-v8a \
    --use_nnapi --use_xnnpack
```

### iOS (CoreML)
```python
# Build iOS framework
python build.sh --config Release --ios \
    --ios_sysroot iphoneos \
    --ios_arch arm64 \
    --use_coreml
```

### Model Optimization for Mobile
```python
# Quantize for mobile
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quant.onnx", weight_type=QuantType.QUInt8)

# Use ORT format for smaller size
opts.add_session_config_entry("session.save_model_format", "ORT")
opts.optimized_model_filepath = "model.ort"
```

---

## 41. MLAS Acceleration

MLAS (Machine Learning Acceleration System) provides optimized CPU kernels.

### Supported Operations
- **GEMM**: SGEMM/DGEMM for all x86/ARM architectures
- **Quantized GEMM**: U8S8, U8U8 for AVX2/AVX512/NEON
- **Convolution**: Im2col + GEMM, Winograd, Direct
- **Attention**: Scaled dot-product attention
- **Activation**: ReLU, GELU, FastGELU, Sigmoid, Tanh
- **Normalization**: LayerNorm, BatchNorm

### Architecture Support
| Architecture | Features |
|-------------|----------|
| x86 SSE4.1 | Basic SIMD |
| x86 AVX2 | 256-bit SIMD, FMA |
| x86 AVX-512 | 512-bit SIMD, VNNI |
| ARM NEON | 128-bit SIMD |
| ARM SVE | Scalable vectors |

### Configuration
```python
# Enable BF16 GEMM fast math (ARM64)
opts.add_session_config_entry("mlas.enable_gemm_fastmath_arm64_bfloat16", "1")

# Enable LUT-based GEMM for quantized models
opts.add_session_config_entry("mlas.use_lut_gemm", "1")

# Disable KleidiAI (ARM)
opts.add_session_config_entry("mlas.disable_kleidiai", "0")
```

---

## 42. Samples and Tutorials

### C++ Inference Sample
```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    Ort::Session session(env, L"model.onnx", opts);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = session.GetInputName(0, allocator);
    auto input_info = session.GetInputTypeInfo(0);
    auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
    auto input_shape = tensor_info.GetShape();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input_data(3 * 224 * 224, 1.0f);
    std::vector<int64_t> shape = {1, 3, 224, 224};

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), shape.data(), shape.size());

    const char* input_names[] = {input_name};
    const char* output_names[] = {session.GetOutputName(0, allocator)};

    auto outputs = session.Run(Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1, output_names, 1);

    auto* output_data = outputs[0].GetTensorData<float>();
    auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();

    std::cout << "Output shape: [";
    for (auto d : output_shape) std::cout << d << " ";
    std::cout << "]" << std::endl;

    return 0;
}
```

### Python Basic Inference
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx")
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
results = sess.run(None, {"input": input_data})
print(f"Output shape: {results[0].shape}")
```

### Python GPU Inference
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": 0})])

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# IO Binding for zero-copy GPU inference
io_binding = sess.io_binding()
ort_input = ort.OrtValue.ortvalue_from_numpy(input_data, "cuda", 0)
io_binding.bind_ortvalue_input("input", ort_input)
io_binding.bind_output("output", "cuda", 0)
sess.run_with_iobinding(io_binding)
output = io_binding.copy_outputs_to_cpu()[0]
```

### Python Quantization
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)
sess = ort.InferenceSession("model_int8.onnx")
results = sess.run(None, {"input": input_data})
```
