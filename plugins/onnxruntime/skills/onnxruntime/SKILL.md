---
name: onnxruntime
description: >
  Comprehensive reference documentation and skill for ONNX Runtime - the cross-platform
  high-performance inference and training engine for ONNX models. Covers C/C++ API,
  Python API (InferenceSession, OrtValue, SessionOptions), all Execution Providers
  (CUDA, TensorRT, OpenVINO, DNNL, CoreML, NNAPI, WebGPU, DirectML, QNN, etc.),
  graph optimization pipeline, operator kernel system, shape inference, custom operators,
  training (ORTModule, TrainingSession), quantization, LoRA adapters, IO Binding,
  language bindings (C#, Java, JavaScript, Rust, Objective-C), WebAssembly deployment,
  build system (CMake), MLAS acceleration, mobile deployment, profiling, and plugin
  development. Based on ONNX Runtime source code analysis.
version: 1.22
---

# ONNX Runtime - Cross-Platform Inference & Training Engine

## Overview

ONNX Runtime is a cross-platform, high-performance inference and training engine for ONNX (Open Neural Network Exchange) models. It provides optimal performance by leveraging hardware accelerators alongside graph optimizations and transforms. ONNX Runtime supports models from deep learning frameworks such as PyTorch and TensorFlow/Keras, as well as classical ML libraries like scikit-learn, LightGBM, and XGBoost.

**Key Capabilities:**
- **Inference**: Accelerate model inference across CPUs, GPUs, NPUs, and edge devices
- **Training**: Accelerate multi-node GPU training for transformer models with ORTModule
- **Quantization**: Post-training quantization and quantization-aware training support
- **20+ Execution Providers**: Hardware-specific backends for optimal performance
- **Cross-Platform**: Linux, macOS, Windows, Android, iOS, WebAssembly

**Supported Hardware:** NVIDIA GPUs (CUDA/TensorRT), AMD GPUs (ROCm/MIGraphX), Intel GPUs (OpenVINO/DNNL), Apple Silicon (CoreML), Qualcomm NPUs (QNN), ARM NPUs (NNAPI), WebGPU, DirectML, and CPU

**Supported Languages:** C, C++, Python, C#, Java, JavaScript/TypeScript, Rust, Objective-C

**ONNX Runtime Version:** 1.22

## Key Architecture Concepts

- **InferenceSession**: The primary object for loading and running ONNX models
- **Execution Provider (EP)**: Hardware-specific backend for accelerating operator execution (20+ EPs)
- **Graph Optimizer**: Multi-level graph transformation pipeline (Level 1-4)
- **OpKernel**: The base class for operator implementations on specific EPs
- **KernelRegistry**: Registry mapping (op_type, EP) pairs to kernel implementations
- **IExecutionProvider**: Interface for all execution providers
- **Graph/GraphViewer**: IR representation of ONNX models
- **OrtValue**: Unified tensor/sparse-tensor container
- **MLAS**: Machine Learning Acceleration System for optimized CPU kernels
- **ORT Module**: PyTorch-compatible training acceleration module

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Language Bindings                                │
│  Python │ C# │ Java │ JavaScript/TS │ Rust │ Objective-C │ C/C++   │
├──────────────────────┼──────────────────────────────────────────────┤
│              C API (onnxruntime_c_api.h)                            │
├──────────────────────┼──────────────────────────────────────────────┤
│              InferenceSession / TrainingSession                     │
├──────────────────────┼──────────────────────────────────────────────┤
│  Graph │ Optimizer │ Partitioning │ Kernel Registry │ Execution     │
│  IR    │ L1-L4     │ EP Assignment│                 │ Scheduling    │
├──────────────────────┼──────────────────────────────────────────────┤
│               Execution Providers (20+)                             │
│  CPU │ CUDA │ TensorRT │ OpenVINO │ DNNL │ CoreML │ NNAPI │ QNN... │
├──────────────────────┼──────────────────────────────────────────────┤
│               MLAS │ Platform │ Memory │ Threading                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Pipeline

```
Load Model → Parse ONNX → Build Graph IR → Optimize Graph (L1→L2→L3→L4)
                                                    ↓
Run() ← Execute Kernels ← Assign Kernels ← Partition by EPs
```

## Quick Reference

### C++ Inference
```cpp
#include <onnxruntime_cxx_api.h>

// Create environment and session
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "my_app");
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(4);
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

// Load model
Ort::Session session(env, L"model.onnx", session_options);

// Prepare input
std::vector<int64_t> input_shape = {1, 3, 224, 224};
std::vector<float> input_data(1 * 3 * 224 * 224, 1.0f);
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
auto input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_data.data(), input_data.size(),
    input_shape.data(), input_shape.size());

// Run inference
const char* input_names[] = {"input"};
const char* output_names[] = {"output"};
auto output_tensors = session.Run(Ort::RunOptions{nullptr},
    input_names, &input_tensor, 1, output_names, 1);
```

### Python Inference
```python
import onnxruntime as ort
import numpy as np

# Create session
sess = ort.InferenceSession("model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
results = sess.run(None, {"input": input_data})

# Or use newer API
io_binding = sess.io_binding()
io_binding.bind_cpu_input("input", input_data)
io_binding.bind_output("output")
sess.run_with_iobinding(io_binding)
output = io_binding.copy_outputs_to_cpu()[0]
```

### CUDA Execution Provider
```python
import onnxruntime as ort

# With options
providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }),
    "CPUExecutionProvider",
]
sess = ort.InferenceSession("model.onnx", providers=providers)
```

### Quantization
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization
quantize_dynamic("model.onnx", "model_quant.onnx",
    weight_type=QuantType.QUInt8)

# Static quantization with calibration
from onnxruntime.quantization import quantize_static, CalibrationDataReader
quantize_static("model.onnx", "model_quant.onnx",
    calibration_data_reader=my_reader)
```

### Session Options
```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.enable_mem_pattern = True
opts.enable_mem_reuse = True
opts.add_session_config_entry("session.disable_prepacking", "0")
```

## Reference Chapters

### Core Architecture

1. [Overview and Architecture](references/01-overview-architecture.md) - Design philosophy, system architecture, code layout, core pipeline
2. [C API Reference](references/02-c-api-reference.md) - Complete C API functions, types, enums, error handling
3. [C++ API Reference](references/03-cpp-api-reference.md) - C++ wrapper classes, RAII, smart pointers, session management
4. [Python API - InferenceSession](references/04-python-api-inference-session.md) - InferenceSession, model loading, inference execution
5. [Python API - Configuration](references/05-python-api-configuration.md) - SessionOptions, RunOptions, all config keys
6. [Python API - OrtValue and Tensors](references/06-python-api-ortvalue-tensors.md) - OrtValue creation, numpy integration, sparse tensors
7. [Data Types and Memory System](references/07-data-types-and-memory.md) - All tensor element types, MemoryInfo, allocators
8. [Environment and Session Lifecycle](references/08-environment-session-lifecycle.md) - Env creation, session initialization, model loading
9. [Error Handling and Logging](references/09-error-handling-logging.md) - Status, error codes, logging system, profiling
10. [Execution Providers Overview](references/10-execution-providers-overview.md) - EP architecture, registration, partitioning, fallback

### Execution Providers

11. [CUDA Execution Provider](references/11-cuda-execution-provider.md) - NVIDIA GPU acceleration, options, kernel implementations
12. [TensorRT Execution Provider](references/12-tensorrt-execution-provider.md) - TensorRT integration, options, trt engine caching
13. [OpenVINO Execution Provider](references/13-openvino-execution-provider.md) - Intel hardware acceleration, device selection
14. [DNNL (oneDNN) Execution Provider](references/14-dnnl-execution-provider.md) - CPU acceleration with oneDNN primitives
15. [CoreML Execution Provider](references/15-coreml-execution-provider.md) - Apple Silicon acceleration
16. [NNAPI Execution Provider](references/16-nnapi-execution-provider.md) - Android NPU acceleration
17. [WebGPU Execution Provider](references/17-webgpu-execution-provider.md) - GPU acceleration for web via WebGPU
18. [DirectML Execution Provider](references/18-directml-execution-provider.md) - Windows GPU acceleration via DirectML
19. [QNN and Other Execution Providers](references/19-other-execution-providers.md) - QNN, ACL, CANN, Vitis-AI, XNNPACK, RKNPU, TVM

### Graph and Operators

20. [Graph System Architecture](references/20-graph-system-architecture.md) - Graph, GraphViewer, Node, NodeArg, Model classes
21. [Graph Optimization Passes](references/21-graph-optimization-passes.md) - All optimizer passes (L1-L4), fusion, elimination, layout transforms
22. [Operator Kernel System](references/22-operator-kernel-system.md) - OpKernel, KernelRegistry, kernel registration, build system
23. [Shape Inference System](references/23-shape-inference-system.md) - Type/shape inference, op schema, inference context
24. [Custom Operator Registration](references/24-custom-operator-registration.md) - Custom op API, kernel implementation, attribute handling
25. [Partitioning and Graph Splitting](references/25-partitioning-graph-splitting.md) - EP capability, node assignment, data transfer

### Training and Optimization

26. [Training API - ORTModule](references/26-training-api-ortmodule.md) - ORTModule, TrainingSession, gradient ops, optimizers
27. [Quantization API](references/27-quantization-api.md) - Dynamic/static quantization, calibration, QDQ format
28. [LoRA Adapter Support](references/28-lora-adapter-support.md) - LoRA adapters, dynamic loading, adapter management
29. [IO Binding and Advanced Inference](references/29-io-binding-advanced-inference.md) - IOBinding, pinned memory, zero-copy inference
30. [Auto Mixed Precision](references/30-auto-mixed-precision.md) - FP16/BF16 mixed precision, loss scaling

### Language Bindings

31. [C# API Reference](references/31-csharp-api-reference.md) - InferenceSession, tensors, providers, async inference
32. [Java API Reference](references/32-java-api-reference.md) - OrtSession, OnnxTensor, OnnxValue, provider options
33. [JavaScript/TypeScript API Reference](references/33-javascript-api-reference.md) - InferenceSession, tensors, WebGPU, Node.js/Web
34. [WebAssembly Deployment](references/34-webassembly-deployment.md) - WASM build, web deployment, WebGL/WebGPU
35. [Rust API Reference](references/35-rust-api-reference.md) - Rust bindings, session, tensor types
36. [Objective-C API Reference](references/36-objectivec-api-reference.md) - ORTSession, ORTValue, CoreML integration

### Build, Deploy, and Tools

37. [Build System (CMake)](references/37-build-system-cmake.md) - All CMake options, build targets, cross-compilation
38. [Docker and Container Deployment](references/38-docker-container-deployment.md) - Dockerfiles, container images, deployment
39. [Profiling and Performance](references/39-profiling-performance.md) - Profiler, tracing, performance tuning, benchmarking
40. [Mobile Deployment](references/40-mobile-deployment.md) - Mobile build, NNAPI, CoreML, model optimization
41. [MLAS Acceleration](references/41-mlas-acceleration.md) - Machine Learning Acceleration System, optimized kernels
42. [Samples and Tutorials](references/42-samples-tutorials.md) - Complete example code for all languages and scenarios
43. [Plugin Development Guide](references/43-plugin-development-guide.md) - EP plugin development, CUDA plugin EP, WebGPU plugin EP
44. [ONNX Format and Model Loading](references/44-onnx-format-model-loading.md) - ONNX format, ORT format, external data, serialization
45. [Model Conversion and Optimization Tools](references/45-model-conversion-tools.md) - Model optimizer, converter, shape inference tools
46. [Allocator and Memory Management](references/46-allocator-memory-management.md) - Memory patterns, arena allocators, memory planning
47. [Threading and Parallelism](references/47-threading-parallelism.md) - Thread pools, intra/inter op parallelism, stream handling
48. [Contributed Operators](references/48-contributed-operators.md) - Non-standard ops by EP, contrib kernel registration
49. [CUDA EP Plugin](references/49-cuda-ep-plugin.md) - CUDA EP plugin architecture, kernel patterns, build system
50. [EP Context and Compiled Models](references/50-ep-context-compiled-models.md) - EP context model, pre-compiled models, weight sharing
