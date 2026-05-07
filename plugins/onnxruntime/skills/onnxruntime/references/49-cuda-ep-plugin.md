# ONNX Runtime Reference - Chapter 49: CUDA EP Plugin (Detailed)

This chapter provides an in-depth examination of the CUDA EP Plugin, covering its directory structure, Python bindings, build system, API differences from the in-tree CUDA EP, plugin loading, CUDA kernel patterns, stream management, memory management, error handling, and testing.

---

## 49.1 Plugin EP CUDA Directory Structure

### 49.1.1 Complete Layout

```
plugin-ep-cuda/
├── CMakeLists.txt                          # Top-level build configuration
├── cmake/
│   ├── CudaEpPluginConfig.cmake.in         # CMake package config template
│   ├── FindCUDAToolkit.cmake               # CUDA toolkit finder
│   └── CompilerFlags.cmake                 # Compiler flag configuration
├── src/
│   ├── cuda_ep.h                           # CudaEp class declaration
│   ├── cuda_ep.cc                          # CudaEp implementation
│   ├── cuda_ep_factory.h                   # Factory class
│   ├── cuda_ep_factory.cc                  # Factory implementation
│   ├── cuda_allocator.h                    # GPU memory allocator
│   ├── cuda_allocator.cc                   # BFC arena allocator for CUDA
│   ├── cuda_pinned_allocator.h             # Pinned host memory allocator
│   ├── cuda_pinned_allocator.cc
│   ├── cuda_stream.h                       # CUDA stream wrapper
│   ├── cuda_stream.cc                      # CudaSyncStream implementation
│   ├── cuda_fence.h                        # Cross-stream synchronization
│   ├── cuda_fence.cc                       # CudaFence implementation
│   ├── cuda_data_transfer.h                # Host <-> Device data transfer
│   ├── cuda_data_transfer.cc
│   ├── cuda_call.h                         # CUDA error checking macros
│   ├── cudnn_call.h                        # cuDNN error checking macros
│   ├── cublas_call.h                       # cuBLAS error checking macros
│   ├── cuda_kernel_adapter.h               # Base class for CUDA kernels
│   ├── cuda_kernel_adapter.cc
│   ├── cuda_utils.h                        # CUDA utility functions
│   ├── cuda_utils.cc
│   ├── cuda_graph.h                        # CUDA Graph support
│   ├── cuda_graph.cc
│   ├── fused_kernels/                      # Fused kernel implementations
│   │   ├── fused_matmul_bias_gelu.h
│   │   ├── fused_matmul_bias_gelu.cu       # CUDA kernel (.cu files)
│   │   ├── fused_conv_bias_relu.h
│   │   ├── fused_conv_bias_relu.cu
│   │   └── ...
│   └── kernels/
│       ├── cuda_kernel_registry.h          # Kernel registry
│       ├── cuda_kernel_registry.cc
│       ├── core/
│       │   ├── matmul.h
│       │   ├── matmul.cc                   # cuBLAS GEMM wrapper
│       │   ├── conv.h
│       │   ├── conv.cc                     # cuDNN convolution wrapper
│       │   ├── reduction.h
│       │   ├── reduction.cu                # Custom CUDA reduction kernel
│       │   ├── elementwise_ops.h
│       │   ├── elementwise_ops.cu          # Element-wise CUDA kernels
│       │   ├── activation_ops.h
│       │   ├── activation_ops.cu           # ReLU, Sigmoid, Tanh, etc.
│       │   ├── normalization.h
│       │   ├── normalization.cu            # LayerNorm, BatchNorm
│       │   ├── pool_ops.h
│       │   ├── pool_ops.cu                 # Pooling operations
│       │   ├── softmax.h
│       │   ├── softmax.cu
│       │   ├── gather.h
│       │   ├── gather.cu
│       │   ├── scatter.h
│       │   ├── scatter.cu
│       │   ├── transpose.h
│       │   ├── transpose.cu
│       │   ├── clip.h
│       │   ├── clip.cu
│       │   ├── concat.h
│       │   ├── concat.cu
│       │   ├── split.h
│       │   ├── split.cu
│       │   ├── where.h
│       │   ├── where.cu
│       │   └── ...
│       ├── contrib/
│       │   ├── attention.h
│       │   ├── attention.cu                # Flash Attention, etc.
│       │   ├── bias_gelu.h
│       │   ├── bias_gelu.cu
│       │   ├── embed_layer_norm.h
│       │   ├── embed_layer_norm.cu
│       │   ├── skip_layer_norm.h
│       │   ├── skip_layer_norm.cu
│       │   ├── fast_gelu.h
│       │   ├── fast_gelu.cu
│       │   ├── matmul_nbits.h
│       │   ├── matmul_nbits.cu             # Quantized MatMul
│       │   └── ...
│       ├── cutlass/
│       │   ├── gemm_kernel.h
│       │   ├── gemm_kernel.cu              # CUTLASS-based GEMM
│       │   ├── gemm_tensor_core.h
│       │   ├── gemm_tensor_core.cu         # Tensor Core GEMM
│       │   └── ...
│       └── utils/
│           ├── cuda_type_utils.h            # Type conversion utilities
│           ├── cuda_math_utils.h            # Math function wrappers
│           └── cuda_device_utils.h          # Device property utilities
├── python/
│   ├── __init__.py                         # Package initialization
│   ├── cuda_ep.py                          # Python API for CUDA EP
│   ├── binding.cc                          # pybind11 C++ bindings
│   └── setup.py                            # Python package setup
├── test/
│   ├── CMakeLists.txt                      # Test build configuration
│   ├── test_cuda_ep.cc                     # EP unit tests
│   ├── test_cuda_kernels.cc                # Kernel unit tests
│   ├── test_cuda_allocator.cc              # Allocator tests
│   ├── test_cuda_stream.cc                 # Stream tests
│   ├── test_data/
│   │   ├── simple_model.onnx
│   │   ├── bert_model.onnx
│   │   ├── resnet_model.onnx
│   │   └── ...
│   └── python/
│       ├── test_cuda_ep.py                 # Python EP tests
│       ├── test_cuda_kernels.py            # Python kernel tests
│       └── conftest.py                     # pytest configuration
├── docs/
│   ├── README.md
│   ├── BUILD.md                            # Build instructions
│   ├── API.md                              # API documentation
│   └── CHANGELOG.md                        # Release notes
└── tools/
    ├── benchmark.py                        # Benchmarking tool
    └── profile.py                          # Profiling tool
```

---

## 49.2 Python Bindings for CUDA Plugin EP

### 49.2.1 Python API Module

```python
# python/cuda_ep.py
"""Python bindings for ONNX Runtime CUDA EP Plugin."""

from typing import Any, Dict, List, Optional, Union
import onnxruntime as ort


class CudaEpOptions:
    """Configuration options for CUDA EP Plugin."""

    def __init__(
        self,
        device_id: int = 0,
        gpu_mem_limit: int = 0,
        arena_extend_strategy: str = "kNextPowerOfTwo",
        cudnn_conv_algo_search: str = "EXHAUSTIVE",
        do_copy_in_default_stream: bool = True,
        user_compute_stream: Optional[int] = None,
        enable_cuda_graph: bool = False,
        enable_skip_layout_transform: bool = False,
        prefer_nhwc: bool = False,
        use_cudnn_conv: bool = True,
        use_cublas_mm: bool = True,
        use_tensor_core: bool = True,
        enable_flash_attention: bool = True,
        enable_memory_efficient_attention: bool = True,
        use_contrib_ops: bool = True,
        enable_quantized_ops: bool = True,
    ):
        self.device_id = device_id
        self.gpu_mem_limit = gpu_mem_limit
        self.arena_extend_strategy = arena_extend_strategy
        self.cudnn_conv_algo_search = cudnn_conv_algo_search
        self.do_copy_in_default_stream = do_copy_in_default_stream
        self.user_compute_stream = user_compute_stream
        self.enable_cuda_graph = enable_cuda_graph
        self.enable_skip_layout_transform = enable_skip_layout_transform
        self.prefer_nhwc = prefer_nhwc
        self.use_cudnn_conv = use_cudnn_conv
        self.use_cublas_mm = use_cublas_mm
        self.use_tensor_core = use_tensor_core
        self.enable_flash_attention = enable_flash_attention
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.use_contrib_ops = use_contrib_ops
        self.enable_quantized_ops = enable_quantized_ops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "gpu_mem_limit": self.gpu_mem_limit,
            "arena_extend_strategy": self.arena_extend_strategy,
            "cudnn_conv_algo_search": self.cudnn_conv_algo_search,
            "do_copy_in_default_stream": self.do_copy_in_default_stream,
            "user_compute_stream": self.user_compute_stream,
            "enable_cuda_graph": self.enable_cuda_graph,
            "enable_skip_layout_transform": self.enable_skip_layout_transform,
            "prefer_nhwc": self.prefer_nhwc,
            "use_cudnn_conv": self.use_cudnn_conv,
            "use_cublas_mm": self.use_cublas_mm,
            "use_tensor_core": self.use_tensor_core,
            "enable_flash_attention": self.enable_flash_attention,
            "enable_memory_efficient_attention": self.enable_memory_efficient_attention,
            "use_contrib_ops": self.use_contrib_ops,
            "enable_quantized_ops": self.enable_quantized_ops,
        }


def load_cuda_ep_plugin(library_path: Optional[str] = None) -> None:
    """Load the CUDA EP plugin shared library.

    Args:
        library_path: Path to the plugin shared library. If None,
            searches in standard locations.
    """
    if library_path is None:
        library_path = _find_cuda_ep_library()

    ort.load("cuda_plugin", library_path)


def create_cuda_session(
    model_path: str,
    options: Optional[CudaEpOptions] = None,
    session_options: Optional[ort.SessionOptions] = None,
) -> ort.InferenceSession:
    """Create an inference session with CUDA EP.

    Args:
        model_path: Path to the ONNX model.
        options: CUDA EP configuration options.
        session_options: Additional session options.

    Returns:
        An InferenceSession with CUDA EP configured.
    """
    if options is None:
        options = CudaEpOptions()

    if session_options is None:
        session_options = ort.SessionOptions()

    # Configure CUDA EP
    provider_options = options.to_dict()

    session = ort.InferenceSession(
        model_path,
        session_options,
        providers=["CUDAExecutionProvider"],
        provider_options=[provider_options],
    )

    return session


def _find_cuda_ep_library() -> str:
    """Find the CUDA EP plugin library in standard locations."""
    import glob
    import os
    import sys

    # Search paths
    search_paths = [
        os.path.dirname(__file__),
        os.path.join(sys.prefix, "lib"),
        "/usr/local/lib",
        "/usr/lib",
    ]

    patterns = [
        "libort-plugin-ep-cuda.so",
        "libort-plugin-ep-cuda.so.*",
        "ort-plugin-ep-cuda.dll",
    ]

    for path in search_paths:
        for pattern in patterns:
            matches = glob.glob(os.path.join(path, pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        "Could not find CUDA EP plugin library. "
        "Please specify the library path explicitly."
    )
```

### 49.2.2 pybind11 Bindings

```cpp
// python/binding.cc
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuda_ep.h"
#include "cuda_ep_factory.h"
#include "cuda_allocator.h"

namespace py = pybind11;

PYBIND11_MODULE(_cuda_ep_plugin, m) {
    m.doc() = "CUDA EP Plugin for ONNX Runtime";

    // CudaEpOptions
    py::class_<CudaEpOptions>(m, "CudaEpOptions")
        .def(py::init<>())
        .def_readwrite("device_id", &CudaEpOptions::device_id)
        .def_readwrite("gpu_mem_limit", &CudaEpOptions::gpu_mem_limit)
        .def_readwrite("arena_extend_strategy", &CudaEpOptions::arena_extend_strategy)
        .def_readwrite("cudnn_conv_algo_search", &CudaEpOptions::cudnn_conv_algo_search)
        .def_readwrite("do_copy_in_default_stream", &CudaEpOptions::do_copy_in_default_stream)
        .def_readwrite("enable_cuda_graph", &CudaEpOptions::enable_cuda_graph)
        .def_readwrite("prefer_nhwc", &CudaEpOptions::prefer_nhwc);

    // CudaEpInfo - Query CUDA device information
    py::class_<CudaEpInfo>(m, "CudaEpInfo")
        .def_static("get_device_count", &CudaEpInfo::GetDeviceCount)
        .def_static("get_device_name", &CudaEpInfo::GetDeviceName)
        .def_static("get_device_memory", &CudaEpInfo::GetDeviceMemory)
        .def_static("get_compute_capability", &CudaEpInfo::GetComputeCapability)
        .def_static("is_flash_attention_available", &CudaEpInfo::IsFlashAttentionAvailable);

    // Memory stats
    py::class_<CudaMemoryStats>(m, "CudaMemoryStats")
        .def_readonly("total_allocated", &CudaMemoryStats::total_allocated)
        .def_readonly("total_reserved", &CudaMemoryStats::total_reserved)
        .def_readonly("peak_allocated", &CudaMemoryStats::peak_allocated);

    // Load function
    m.def("load_plugin", [](const std::string& path) {
        // Implementation delegates to ort.load
    }, py::arg("path"));

    // Get CUDA device info
    m.def("get_cuda_device_info", []() {
        py::dict info;
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        info["device_count"] = device_count;

        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::string key = "device_" + std::to_string(i);
            info[key.c_str()] = py::dict(
                "name"_a = prop.name,
                "compute_capability"_a = py::str(
                    std::to_string(prop.major) + "." + std::to_string(prop.minor)),
                "total_memory"_a = prop.totalGlobalMem,
                "multiprocessor_count"_a = prop.multiProcessorCount
            );
        }
        return info;
    });
}
```

---

## 49.3 Build System for CUDA Plugin

### 49.3.1 Top-Level CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.24)
project(ort-plugin-ep-cuda VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_FLASH_ATTENTION "Enable Flash Attention kernel" ON)
option(ENABLE_CUTLASS "Enable CUTLASS GEMM kernels" ON)
option(ENABLE_NVIDIA_TF32 "Enable TF32 on Ampere+" ON)

# Find dependencies
find_package(CUDAToolkit 11.8 REQUIRED)
find_package(onnxruntime 1.20.0 REQUIRED)

# CUDA architecture flags
set(CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")

if(ENABLE_NVIDIA_TF32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()

# Source files
file(GLOB_RECURSE CUDA_EP_SOURCES
    "src/*.cc"
    "src/*.cu"
    "src/*.cpp"
)

# Shared library
add_library(ort-plugin-ep-cuda SHARED ${CUDA_EP_SOURCES})

target_include_directories(ort-plugin-ep-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${onnxruntime_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ort-plugin-ep-cuda
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cudnn
        CUDA::curand
        ${onnxruntime_LIBRARIES}
)

# CUTLASS (optional)
if(ENABLE_CUTLASS)
    # Find or fetch CUTLASS
    include(cmake/FindCutlass.cmake)
    target_link_libraries(ort-plugin-ep-cuda PRIVATE cutlass::cutlass)
    target_compile_definitions(ort-plugin-ep-cuda PRIVATE ENABLE_CUTLASS)
endif()

# Flash Attention (optional)
if(ENABLE_FLASH_ATTENTION)
    include(cmake/FindFlashAttention.cmake)
    target_link_libraries(ort-plugin-ep-cuda PRIVATE flash_attention)
    target_compile_definitions(ort-plugin-ep-cuda PRIVATE ENABLE_FLASH_ATTENTION)
endif()

# Export symbols
set_target_properties(ort-plugin-ep-cuda PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "ort-plugin-ep-cuda"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Install rules
include(GNUInstallDirs)
install(TARGETS ort-plugin-ep-cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ort-plugin-ep-cuda
    FILES_MATCHING PATTERN "*.h"
)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_cuda_ep_plugin python/binding.cc)
    target_link_libraries(_cuda_ep_plugin PRIVATE ort-plugin-ep-cuda)
    target_include_directories(_cuda_ep_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    install(TARGETS _cuda_ep_plugin LIBRARY DESTINATION python)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()
```

### 49.3.2 Build Commands

```bash
# Configure
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES="80;86;89" \
    -Donnxruntime_DIR=/path/to/ort/cmake \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DENABLE_FLASH_ATTENTION=ON \
    -DENABLE_CUTLASS=ON

# Build
cmake --build . -j$(nproc)

# Install
cmake --install . --prefix=/usr/local

# Run tests
ctest --output-on-failure
```

---

## 49.4 API Differences from In-Tree CUDA EP

### 49.4.1 Loading API

```python
# In-tree CUDA EP (built into ONNX Runtime)
import onnxruntime as ort
options = ort.SessionOptions()
options.append_execution_provider("CUDA", {"device_id": 0})
session = ort.InferenceSession("model.onnx", options)

# Plugin CUDA EP (loaded at runtime)
import onnxruntime as ort
ort.load("cuda_plugin", "/path/to/libort-plugin-ep-cuda.so")
options = ort.SessionOptions()
options.append_execution_provider("CUDA", {"device_id": 0})
session = ort.InferenceSession("model.onnx", options)
```

### 49.4.2 Configuration Differences

| Option | In-Tree | Plugin | Notes |
|--------|---------|--------|-------|
| Session-level config | `SessionOptions` entries | Same + plugin-specific | Plugin has extra options |
| EP registration | `append_execution_provider` | Same API | Transparent to user |
| Memory management | BFC arena | BFC arena | Same implementation |
| Stream handling | Internal | Internal | Same model |
| Error reporting | ORT status | ORT status | Same error codes |
| Logging | ORT logger | ORT logger | Same logging |
| Build flags | Compile-time | CMake options | Plugin is more flexible |

### 49.4.3 Functional Differences

| Feature | In-Tree | Plugin |
|---------|---------|--------|
| All standard ONNX ops | Full | Full |
| Contrib ops | Full | Growing subset |
| Flash Attention | Yes | Yes |
| CUTLASS GEMM | Yes | Yes |
| cuDNN conv algorithms | All | All |
| CUDA Graphs | Yes | Yes |
| NHWC layout | Yes | Yes |
| Weight pre-packing | Yes | Yes |
| Multi-GPU | Yes | Yes |
| FP16/BF16 | Yes | Yes |
| INT8 (QLinear ops) | Full | Partial |
| 4-bit quantization | Yes | Yes |
| FP8 | Yes | Limited |
| Profiling | ORT profiler | ORT profiler |
| Training mode | Yes | No (inference only) |

---

## 49.5 Plugin Loading and Registration

### 49.5.1 Loading Flow

```
1. ort.load("cuda_plugin", path)
   │
   ├── dlopen(path)
   │
   ├── GetEpApiVersion() → verify version
   │
   ├── GetEpFactory() → get CudaEpFactory
   │
   ├── Register factory with OrtEnv
   │
   └── Factory available for session creation

2. Session creation with "CUDA" provider
   │
   ├── Lookup factory by name ("CUDA")
   │
   ├── factory->CreateEp(session_options, logger)
   │
   ├── CudaEpFactory::CreateEpImpl()
   │   ├── Parse options from session_options
   │   ├── cudaSetDevice(device_id)
   │   ├── cudaStreamCreate(&compute_stream)
   │   ├── cudnnCreate(&cudnn_handle)
   │   ├── cublasCreate(&cublas_handle)
   │   ├── Create BFC arena allocator
   │   ├── Register CUDA kernels
   │   └── Return CudaEp instance
   │
   └── EP is active for this session
```

### 49.5.2 Registration at Session Creation

```cpp
// When session is created with CUDA EP:
Status InferenceSession::AddExecutionProvider(const std::string& ep_name,
                                               const ProviderOptions& options) {
    // Check if this is a plugin EP
    auto* factory = env_->GetPluginEpFactory(ep_name);
    if (factory != nullptr) {
        // Use plugin factory
        OrtEp* ep = nullptr;
        factory->CreateEp(factory, session_options_, logger_, &ep);

        // Register the EP
        auto provider = std::unique_ptr<IExecutionProvider>(
            reinterpret_cast<IExecutionProvider*>(ep));
        ORT_RETURN_IF_ERROR(RegisterExecutionProvider(std::move(provider)));
    }
    return Status::OK();
}
```

---

## 49.6 CUDA Kernel Patterns

### 49.6.1 Standard CUDA Kernel Pattern

```cpp
// Element-wise kernel template
template <typename T, typename Functor>
__global__ void ElementWiseKernel(const T* input, T* output,
                                   int64_t num_elements, Functor functor) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = functor(input[idx]);
    }
}

// Kernel launch wrapper
template <typename T, typename Functor>
Status LaunchElementWiseKernel(cudaStream_t stream, const T* input, T* output,
                                int64_t num_elements, Functor functor) {
    constexpr int block_size = 256;
    int64_t grid_size = (num_elements + block_size - 1) / block_size;

    ElementWiseKernel<T, Functor><<<grid_size, block_size, 0, stream>>>(
        input, output, num_elements, functor);

    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
}
```

### 49.6.2 Reduction Kernel Pattern

```cpp
// Warp-level reduction
template <int BLOCK_SIZE>
__device__ float WarpReduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void ReduceKernel(const float* input, float* output,
                              int64_t num_elements) {
    __shared__ float shared_mem[BLOCK_SIZE / 32];  // One per warp

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride loop
    for (int64_t i = tid; i < num_elements; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Warp reduction
    sum = WarpReduce<BLOCK_SIZE>(sum);

    // Write warp result to shared memory
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_mem[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (threadIdx.x < BLOCK_SIZE / 32) ? shared_mem[lane] : 0.0f;
        sum = WarpReduce<BLOCK_SIZE / 32>(sum);
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}
```

### 49.6.3 GEMM Kernel Pattern (cuBLAS)

```cpp
Status CudaMatMulKernel::ComputeInternal(OpKernelContext* context) const {
    const Tensor* A = context->Input<Tensor>(0);
    const Tensor* B = context->Input<Tensor>(1);
    Tensor* Y = context->Output(0, ComputeOutputShape(A->Shape(), B->Shape()));

    const float* a_data = A->Data<float>();
    const float* b_data = B->Data<float>();
    float* y_data = Y->MutableData<float>();

    auto a_shape = A->Shape();
    auto b_shape = B->Shape();

    int M = static_cast<int>(a_shape[0]);
    int N = static_cast<int>(b_shape[1]);
    int K = static_cast<int>(a_shape[1]);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Use Tensor Cores when available (FP16 accumulation with FP32 output)
    cublasStatus_t status = cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b_data, CUDA_R_32F, N,
        a_data, CUDA_R_32F, K,
        &beta,
        y_data, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,                   // Compute type
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);       // Use Tensor Cores

    CUBLAS_RETURN_IF_ERROR(status);
    return Status::OK();
}
```

### 49.6.4 Convolution Kernel Pattern (cuDNN)

```cpp
Status CudaConvKernel::ComputeInternal(OpKernelContext* context) const {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* W = context->Input<Tensor>(1);
    const Tensor* B = context->Input<Tensor>(2);  // Optional bias
    Tensor* Y = context->Output(0, ComputeOutputShape());

    // Create cuDNN tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_RETURN_IF_ERROR(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_RETURN_IF_ERROR(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_RETURN_IF_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set descriptors
    // ... (set dimensions, padding, stride, dilation)

    // Find the best algorithm
    int algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    CUDNN_RETURN_IF_ERROR(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle_,
        x_desc, w_desc, conv_desc, y_desc,
        1, &algo_count, &algo_perf));

    cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;

    // Allocate workspace
    size_t workspace_size = 0;
    CUDNN_RETURN_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle_, x_desc, w_desc, conv_desc, y_desc, algo,
        &workspace_size));

    auto workspace = GetScratchBuffer(workspace_size);

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_RETURN_IF_ERROR(cudnnConvolutionForward(
        cudnn_handle_,
        &alpha,
        x_desc, X->Data<float>(),
        w_desc, W->Data<float>(),
        conv_desc, algo,
        workspace->MutableData(), workspace_size,
        &beta,
        y_desc, Y->MutableData<float>()));

    // Add bias if present
    if (B != nullptr) {
        // ... add bias using cuDNN or custom kernel
    }

    return Status::OK();
}
```

---

## 49.7 Stream Management

### 49.7.1 Stream Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CudaEp Instance                     │
│                                                      │
│  compute_stream_ (cudaStream_t)                      │
│  ├── Used for all GPU compute operations             │
│  ├── Non-blocking (created with cudaStreamNonBlocking)│
│  └── Synchronized via ep->Sync()                     │
│                                                      │
│  copy_stream_ (optional)                             │
│  ├── Used for async H2D/D2H copies                   │
│  ├── Separate from compute to overlap copy/compute   │
│  └── Created when do_copy_in_default_stream=false    │
│                                                      │
│  cuDNN/cuBLAS handles                                │
│  ├── Bound to compute_stream_                        │
│  └── All operations go through compute_stream_       │
└─────────────────────────────────────────────────────┘
```

### 49.7.2 Stream Synchronization Points

```
Input copy (H2D) ──→ Compute ──→ Output copy (D2H)
       │                              │               │
       │  copy_stream_                │  compute_     │  copy_stream_
       │                              │  stream_      │
       │         cudaEvent            │               │
       └──────── Record ──────────────┘               │
                   Wait                              │
                                                     │
                                        cudaEvent ────┘
                                          Record
                                          Wait (on host stream)
```

### 49.7.3 Stream Configuration

```python
# Stream configuration via session options
options = ort.SessionOptions()

# Default: copy and compute on the same stream (simpler, less overlap)
# options.add_config_entry("ep.cuda.do_copy_in_default_stream", "1")

# Separate copy and compute streams (better overlap)
options.add_config_entry("ep.cuda.do_copy_in_default_stream", "0")

# User-provided compute stream
options.add_config_entry("ep.cuda.user_compute_stream",
                         str(cuda_stream_ptr))
```

---

## 49.8 Memory Management

### 49.8.1 Memory Allocation Hierarchy

```
User Request
    │
    ├── IAllocator::Alloc(size)
    │   │
    │   ├── BFCArena::Alloc(size)
    │   │   ├── Check free bins for best-fit chunk
    │   │   ├── If found: return chunk (may split)
    │   │   └── If not found: extend arena via cudaMalloc
    │   │
    │   └── Returns device pointer
    │
    └── Memory is freed via BFCArena::Free(ptr)
        ├── Mark chunk as free
        ├── Coalesce adjacent free chunks
        └── Return to free bin
```

### 49.8.2 Memory Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()

# GPU memory limit (0 = all available memory)
provider_options = {
    "device_id": 0,
    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB

    # Arena extension strategy
    # "kNextPowerOfTwo" (default) - double each time
    # "kSameAsRequested" - allocate exact size
    "arena_extend_strategy": "kNextPowerOfTwo",

    # Enable CUDA memory arena (default: True)
    # Set to False for user-managed memory
    "enable_cuda_memory_arena": True,
}
```

### 49.8.3 Scratch Space Management

```cpp
// Scratch buffer for temporary GPU memory (e.g., cuDNN workspace)
class CudaScratchSpace {
public:
    explicit CudaScratchSpace(cudaStream_t stream)
        : stream_(stream) {}

    // Allocate scratch buffer (reused across calls)
    void* Allocate(size_t size) {
        if (size > buffer_size_) {
            // Grow buffer
            if (buffer_ != nullptr) {
                CUDA_CALL(cudaFreeAsync(buffer_, stream_));
            }
            CUDA_CALL(cudaMallocAsync(&buffer_, size, stream_));
            buffer_size_ = size;
        }
        return buffer_;
    }

    ~CudaScratchSpace() {
        if (buffer_ != nullptr) {
            CUDA_CALL(cudaFreeAsync(buffer_, stream_));
        }
    }

private:
    cudaStream_t stream_;
    void* buffer_ = nullptr;
    size_t buffer_size_ = 0;
};
```

---

## 49.9 Error Handling

### 49.9.1 CUDA Error Macros

```cpp
// src/cuda_call.h

// Basic CUDA error check
#define CUDA_RETURN_IF_ERROR(expr)                                     \
    do {                                                                \
        cudaError_t status = (expr);                                    \
        if (status != cudaSuccess) {                                    \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                   \
                "CUDA error: ", cudaGetErrorString(status),             \
                " in ", __FILE__, ":", __LINE__);                       \
        }                                                               \
    } while (0)

// CUDA error with context
#define CUDA_CALL(expr)                                                \
    do {                                                                \
        cudaError_t status = (expr);                                    \
        if (status != cudaSuccess) {                                    \
            ORT_THROW("CUDA error at ", __FILE__, ":", __LINE__,       \
                      ": ", cudaGetErrorString(status),                 \
                      " (", static_cast<int>(status), ")");             \
        }                                                               \
    } while (0)

// cuDNN error check
#define CUDNN_RETURN_IF_ERROR(expr)                                    \
    do {                                                                \
        cudnnStatus_t status = (expr);                                  \
        if (status != CUDNN_STATUS_SUCCESS) {                           \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                   \
                "cuDNN error: ", cudnnGetErrorString(status),           \
                " in ", __FILE__, ":", __LINE__);                       \
        }                                                               \
    } while (0)

// cuBLAS error check
#define CUBLAS_RETURN_IF_ERROR(expr)                                   \
    do {                                                                \
        cublasStatus_t status = (expr);                                 \
        if (status != CUBLAS_STATUS_SUCCESS) {                          \
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,                   \
                "cuBLAS error (", static_cast<int>(status), ")"         \
                " in ", __FILE__, ":", __LINE__);                       \
        }                                                               \
    } while (0)

// CUDA kernel launch error check
#define CUDA_CHECK_KERNEL_LAUNCH()                                      \
    do {                                                                \
        CUDA_RETURN_IF_ERROR(cudaGetLastError());                       \
    } while (0)
```

### 49.9.2 Error Propagation

```
CUDA Error (e.g., cudaErrorMemoryAllocation)
    │
    ├── CUDA_RETURN_IF_ERROR macro
    │   └── Converts to ORT Status with error details
    │
    ├── Propagated up through Compute() chain
    │
    ├── ORT Session::Run() catches Status
    │   └── Converts to OrtException
    │
    └── Python: onnxruntime.RuntimeException
```

### 49.9.3 Graceful Error Recovery

```cpp
Status CudaEp::OnRunStart() {
    try {
        CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id_));

        // Check if device is still available
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id_);
        if (err != cudaSuccess) {
            LOGS(logger_, ERROR) << "CUDA device " << device_id_
                                  << " is not available: "
                                  << cudaGetErrorString(err);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                "CUDA device not available");
        }

        return Status::OK();
    } catch (const std::exception& e) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
            "OnRunStart failed: ", e.what());
    }
}
```

---

## 49.10 Testing

### 49.10.1 Unit Test Framework

```cpp
// test/test_cuda_kernels.cc
#include "gtest/gtest.h"
#include "cuda_ep.h"
#include "kernels/cuda_kernel_registry.h"

class CudaKernelTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize CUDA EP once for all tests
        OrtSessionOptions* opts = nullptr;
        OrtSessionOptionsCreate(&opts);
        logger_ = OrtLogger::GetDefaultLogger();
        ep_ = std::make_unique<CudaEp>(opts, *logger_, 0);
    }

    static void TearDownTestSuite() {
        ep_.reset();
    }

    void SetUp() override {
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaFree(0));  // Ensure CUDA context is ready
    }

    static std::unique_ptr<CudaEp> ep_;
    static const OrtLogger* logger_;
};

std::unique_ptr<CudaEp> CudaKernelTest::ep_;
const OrtLogger* CudaKernelTest::logger_;

// Test MatMul kernel
TEST_F(CudaKernelTest, MatMul_Basic) {
    auto allocator = ep_->GetAllocator(0, OrtMemType::OrtMemType_Default);

    // Allocate inputs on GPU
    int M = 128, K = 64, N = 256;
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t y_bytes = M * N * sizeof(float);

    float* d_A = static_cast<float*>(allocator->Alloc(a_bytes));
    float* d_B = static_cast<float*>(allocator->Alloc(b_bytes));
    float* d_Y = static_cast<float*>(allocator->Alloc(y_bytes));

    // Initialize inputs on host and copy to device
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i) / (M * K);
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(i) / (K * N);

    CUDA_CALL(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    // Run MatMul
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CALL(cublasSgemm(
        ep_->GetCublasHandle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_Y, N));

    CUDA_CALL(cudaStreamSynchronize(ep_->GetComputeStream()));

    // Copy result back and verify
    std::vector<float> h_Y(M * N);
    CUDA_CALL(cudaMemcpy(h_Y.data(), d_Y, y_bytes, cudaMemcpyDeviceToHost));

    // Basic sanity check
    EXPECT_GT(h_Y[0], 0.0f);

    // Cleanup
    allocator->Free(d_A);
    allocator->Free(d_B);
    allocator->Free(d_Y);
}

// Test allocator
TEST_F(CudaKernelTest, Allocator_Basic) {
    auto alloc = ep_->GetAllocator(0, OrtMemType::OrtMemType_Default);
    ASSERT_NE(alloc, nullptr);

    // Allocate
    void* ptr = alloc->Alloc(1024 * 1024);  // 1MB
    ASSERT_NE(ptr, nullptr);

    // Write to GPU memory
    float value = 42.0f;
    CUDA_CALL(cudaMemcpy(ptr, &value, sizeof(float), cudaMemcpyHostToDevice));

    // Read back
    float result;
    CUDA_CALL(cudaMemcpy(&result, ptr, sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(result, 42.0f);

    // Free
    alloc->Free(ptr);
}

// Test stream synchronization
TEST_F(CudaKernelTest, Stream_Sync) {
    auto status = ep_->Sync();
    EXPECT_TRUE(status.IsOK());
}
```

### 49.10.2 Python Integration Tests

```python
# test/python/test_cuda_ep.py
import pytest
import numpy as np
import onnxruntime as ort


@pytest.fixture(scope="module")
def cuda_session():
    """Create a CUDA EP session for testing."""
    ort.load("cuda_plugin", "/path/to/libort-plugin-ep-cuda.so")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        "test_data/simple_model.onnx",
        options,
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": 0}]
    )
    yield session


class TestCudaEp:
    def test_basic_inference(self, cuda_session):
        """Test basic inference with CUDA EP."""
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        input_name = cuda_session.get_inputs()[0].name

        output = cuda_session.run(None, {input_name: input_data})
        assert output is not None
        assert output[0].shape == (1, 64, 112, 112)

    def test_matmul_accuracy(self):
        """Test MatMul accuracy on CUDA EP."""
        # Create MatMul model
        import onnx
        from onnx import helper, TensorProto

        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [4, 8])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [8, 6])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4, 6])

        node = helper.make_node('MatMul', ['A', 'B'], ['Y'])
        graph = helper.make_graph([node], 'matmul_test', [A], [B], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        onnx.save(model, "/tmp/test_matmul.onnx")

        options = ort.SessionOptions()
        session = ort.InferenceSession("/tmp/test_matmul.onnx", options,
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": 0}])

        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 6).astype(np.float32)
        output = session.run(None, {'A': a, 'B': b})

        expected = np.matmul(a, b)
        np.testing.assert_allclose(output[0], expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_dynamic_batch(self, batch_size):
        """Test dynamic batch size handling."""
        # ... test with various batch sizes

    def test_memory_limit(self):
        """Test GPU memory limit enforcement."""
        options = ort.SessionOptions()
        provider_options = {
            "device_id": 0,
            "gpu_mem_limit": 256 * 1024 * 1024,  # 256MB
        }
        session = ort.InferenceSession(
            "test_data/simple_model.onnx", options,
            providers=["CUDAExecutionProvider"],
            provider_options=[provider_options])

        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = session.run(None, {"input": input_data})
        assert output is not None
```

### 49.10.3 Performance Benchmarks

```python
# tools/benchmark.py
"""Benchmark tool for CUDA EP Plugin."""

import time
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    model_name: str
    batch_size: int
    num_iterations: int
    warmup_iterations: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float


def benchmark_session(session, input_data, num_iterations=100,
                      warmup_iterations=10):
    """Benchmark an inference session."""
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup_iterations):
        session.run(None, {input_name: input_data})

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies.sort()
    return {
        "avg_ms": np.mean(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "throughput_qps": 1000.0 / np.mean(latencies),
    }
```

---

## 49.11 Summary

| Topic | Key Points |
|-------|-----------|
| Directory Structure | `src/` for C++/CUDA code, `python/` for bindings, `test/` for tests |
| Python Bindings | pybind11-based, `CudaEpOptions` for configuration |
| Build System | CMake with CUDA toolkit, optional CUTLASS/Flash Attention |
| API Differences | Same user API as in-tree, loaded via `ort.load()` |
| Plugin Loading | dlopen → version check → factory registration |
| CUDA Kernel Patterns | Element-wise, reduction, GEMM (cuBLAS), Conv (cuDNN) |
| Stream Management | Compute stream + optional copy stream, event-based sync |
| Memory Management | BFC arena with cudaMalloc, scratch space for cuDNN |
| Error Handling | CUDA_RETURN_IF_ERROR macros, graceful recovery |
| Testing | GTest (C++), pytest (Python), benchmark tools |
