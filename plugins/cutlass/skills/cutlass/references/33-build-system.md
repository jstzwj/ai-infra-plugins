# CUTLASS Build System - Chapter 33: CMake Configuration and Compilation

This reference covers the CUTLASS build system, including CMake configuration, compilation options, dependency management, library builds, and troubleshooting.

---

## 33.1 Overview

CUTLASS uses CMake as its primary build system. The project supports multiple build configurations ranging from header-only usage to fully compiled static/shared libraries. CUTLASS is template-heavy, so compile times can be substantial. The build system provides several mechanisms to control which kernels are compiled, targeting specific architectures and operation types.

### Minimum Requirements

| Requirement | Minimum Version | Notes |
|---|---|---|
| CMake | 3.18+ | Required for CUDA language support |
| CUDA Toolkit | 11.4+ | 11.8+ recommended for SM90; 12.x for SM100+ |
| C++ Standard | C++17 | Required by CUTLASS 3.x |
| Host Compiler | GCC 9+, MSVC 2019+, Clang 10+ | Must support C++17 |
| Python | 3.6+ | For profiler and code generation tools |

---

## 33.2 Main CMakeLists.txt Structure

The top-level `CMakeLists.txt` in the CUTLASS repository defines the overall project structure:

```
cutlass/
  CMakeLists.txt              # Top-level project configuration
  include/
    cutlass/                  # Header-only core library
  examples/
    CMakeLists.txt            # Example programs
  test/
    unit/                     # Unit tests
    CMakeLists.txt            # Test configuration
  tools/
    profiler/                 # CUTLASS Profiler
    library/                  # Pre-compiled kernel library
    util/                     # Utility headers
```

### Project Declaration

```cmake
cmake_minimum_required(VERSION 3.18)

# Enable CUDA language support
project(CUTLASS LANGUAGES CXX CUDA)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

### Key CMake Option Defaults

```cmake
# Default build options
option(CUTLASS_ENABLE_TESTS "Enable tests" OFF)
option(CUTLASS_ENABLE_PROFILER "Enable profiler" ON)
option(CUTLASS_ENABLE_EXAMPLES "Enable examples" ON)
option(CUTLASS_ENABLE_GITHUB_UNITTEST_WORKAROUNDS "Workaround for GitHub Actions" OFF)
option(CUTLASS_NVCC_ARCHS "Target NVCC architectures" "")
option(CUTLASS_LIBRARY_KERNELS "Filter kernels to compile" "")
```

---

## 33.3 Dependencies

### Required Dependencies

**CUDA Toolkit**: CUTLASS fundamentally requires the CUDA toolkit. The minimum supported version depends on the target architecture:

| Target Architecture | Minimum CUDA Version |
|---|---|
| SM70-SM75 | CUDA 11.4 |
| SM80-SM89 | CUDA 11.4 |
| SM90 (Hopper) | CUDA 11.8 (12.0+ recommended) |
| SM100+ (Blackwell) | CUDA 12.x |

```cmake
# Verify CUDA is available
find_package(CUDAToolkit REQUIRED)

# The CUDA compiler (nvcc) is used directly via CMake's CUDA language
enable_language(CUDA)
```

### Optional Dependencies

| Dependency | Purpose | How to Enable |
|---|---|---|
| cuBLAS | Reference validation in tests and profiler | Automatic if CUDA toolkit found |
| cuDNN | Convolution reference validation | Set `CUDNN_ROOT_PATH` |
| Google Test | Unit test framework | Downloaded automatically |
| Python 3 | Profiler post-processing, code generation | Found via `find_package` |
| doxygen | Documentation generation | `BUILD_DOCS` option |

```cmake
# cuBLAS (typically comes with CUDA toolkit)
find_library(CUBLAS cublas HINTS ${CUDAToolkit_LIBRARY_DIR})

# cuDNN (optional)
if(CUTLASS_ENABLE_PROFILER OR CUTLASS_ENABLE_TESTS)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_PATH})
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_PATH})
endif()
```

---

## 33.4 Build Options

### 33.4.1 CUTLASS_LIBRARY_KERNELS - Selective Kernel Compilation

CUTLASS is a template library with potentially thousands of kernel instantiations. Compiling all of them would take hours. The `CUTLASS_LIBRARY_KERNELS` option uses CMake generator expressions to filter which kernels are compiled into the library.

```bash
# Compile only SM90 FP16 GEMM kernels
cmake .. -DCUTLASS_LIBRARY_KERNELS="sm90*gemm*fp16*"

# Compile specific kernel names
cmake .. -DCUTLASS_LIBRARY_KERNELS="cutlass_sm90_tensorop_gemm_f16*"

# Compile all GEMM kernels for SM80
cmake .. -DCUTLASS_LIBRARY_KERNELS="sm80*gemm*"

# Multiple patterns (semicolon-separated)
cmake .. -DCUTLASS_LIBRARY_KERNELS="sm90*gemm*f16*;sm90*gemm*bf16*"

# Compile everything (warning: very slow)
cmake .. -DCUTLASS_LIBRARY_KERNELS="all"
```

The kernel naming convention follows this pattern:
```
cutlass_<arch>_<opclass>_<operation>_<type>_<layout>_<tile>
```

Common filter patterns:

```bash
# All TensorOp GEMMs
-D_CUTLASS_LIBRARY_KERNELS="*tensorop*gemm*"

# All SIMT GEMMs (no Tensor Cores)
-D_CUTLASS_LIBRARY_KERNELS="*simt*gemm*"

# Specific tile size
-D_CUTLASS_LIBRARY_KERNELS="*128x128x*"

# Only grouped GEMM
-D_CUTLASS_LIBRARY_KERNELS="*grouped*"

# Only Rank-K or Rank-2K update
-D_CUTLASS_LIBRARY_KERNELS="*rank_k*;*rank_2k*"

# Only SYMM/HEMM
-D_CUTLASS_LIBRARY_KERNELS="*symm*;*hemm*"
```

### 33.4.2 CUTLASS_ENABLE_TESTS

Controls whether unit tests are built. Tests add significant compilation time.

```bash
# Enable tests
cmake .. -DCUTLASS_ENABLE_TESTS=ON

# Build only specific test targets
cmake --build . --target cutlass_test_unit_gemm_device_simt

# Run tests
ctest --output-on-failure -R "simt"
```

Test targets are organized hierarchically:
- `cutlass_test_unit_gemm_device` - All device GEMM tests
- `cutlass_test_unit_gemm_device_simt` - SIMT GEMM tests
- `cutlass_test_unit_gemm_device_tensorop` - TensorOp GEMM tests
- `cutlass_test_unit_epilogue` - Epilogue tests
- `cutlass_test_unit_layout` - Layout tests
- `cutlass_test_unit_util` - Utility tests

### 33.4.3 CUTLASS_ENABLE_PROFILER

The CUTLASS Profiler is a powerful tool for benchmarking kernels. It adds compilation time but is essential for performance tuning.

```bash
# Enable profiler (default)
cmake .. -DCUTLASS_ENABLE_PROFILER=ON

# Disable profiler for faster builds
cmake .. -DCUTLASS_ENABLE_PROFILER=OFF

# Run the profiler
./tools/profiler/cutlass_profiler --kinds=gemm --m=1024 --n=1024 --k=1024
```

### 33.4.4 CUTLASS_NVCC_ARCHS - Target Architecture

Specifies which GPU architectures to compile for. This controls which PTX and SASS are generated.

```bash
# Single architecture
cmake .. -DCUTLASS_NVCC_ARCHS="90a"

# Multiple architectures
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a"

# All common architectures
cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80;86;89;90a"

# Blackwell (SM100)
cmake .. -DCUTLASS_NVCC_ARCHS="100"
```

Architecture string reference:

| Architecture | NVCC Arch String | Notes |
|---|---|---|
| Volta | `70` | V100 |
| Turing | `75` | T4, RTX 2080 |
| Ampere | `80` | A100 |
| Ampere | `86` | RTX 3090 |
| Ada | `89` | RTX 4090, L40 |
| Hopper | `90a` | H100 (the `a` variant enables cluster features) |
| Blackwell | `100` | B100 |
| Blackwell | `101` | B200 |
| Blackwell | `103` | RTX 5090 |

> **Important**: For Hopper SM90, use `90a` (not `90`) to enable all features including thread block clusters and TMA. The plain `90` variant disables some features.

If `CUTLASS_NVCC_ARCHS` is not set, CMake will auto-detect the GPUs present in the system:

```cmake
# Auto-detection logic (simplified)
if(NOT CUTLASS_NVCC_ARCHS)
  execute_process(COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --list-gpu-arch
    OUTPUT_VARIABLE DETECTED_ARCHS)
endif()
```

### 33.4.5 CUTLASS_DEBUG_TRACE_LEVEL

Controls the verbosity of debug tracing output embedded in CUTLASS kernels. Higher levels produce more output but may affect performance.

```bash
# No debug tracing (default, production builds)
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=0

# Basic tracing
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=1

# Detailed tracing
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=2

# Verbose tracing (useful for debugging, slow)
cmake .. -DCUTLASS_DEBUG_TRACE_LEVEL=3
```

Debug trace levels:

| Level | Description | Performance Impact |
|---|---|---|
| 0 | No tracing | None |
| 1 | Kernel launch info, tile iteration counts | Minimal |
| 2 | Per-stage data movement tracing | Moderate |
| 3 | Full per-thread tracing | Significant |

---

## 33.5 Building as a Header-Only Library

CUTLASS is primarily designed as a header-only library. Most usage does not require building CUTLASS itself; instead, you include the headers and instantiate templates in your own project.

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyGemmProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Option 1: Add CUTLASS as a subdirectory (no library build needed)
add_subdirectory(${CUTLASS_PATH} cutlass_build)

# Option 2: Simply add the include path
include_directories(${CUTLASS_PATH}/include)
include_directories(${CUTLASS_PATH}/tools/util/include)

add_executable(my_gemm my_gemm.cpp)
target_compile_options(my_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_90a>)
```

Header-only usage example in your own project:

```cpp
// my_gemm.cpp - no CUTLASS library needed
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Kernel is fully instantiated in your translation unit
// The compiler generates code for exactly the kernel you use
```

### Adding CUTLASS to Your Project via FetchContent

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG        v3.8.0
)

# Disable parts we don't need to speed up configuration
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(cutlass)

add_executable(my_kernel my_kernel.cu)
target_link_libraries(my_kernel PRIVATE cutlass::cutlass)
```

---

## 33.6 Building Shared and Static Libraries

CUTLASS can also be compiled into a pre-built library containing specific kernel instantiations. This is useful for deploying without the full CUTLASS source tree.

### Building the CUTLASS Library

```bash
# Configure with specific kernels
cmake .. \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_LIBRARY_KERNELS="sm90*gemm*f16*;sm90*gemm*bf16*"

# Build the library
cmake --build . --target cutlass_library

# The library output is in:
#   build/tools/library/libcutlass.so   (shared)
#   build/tools/library/libcutlass.a    (static)
```

### CMake Configuration for Library Build

```cmake
# The library target is defined in tools/library/CMakeLists.txt
# It compiles all kernels matching CUTLASS_LIBRARY_KERNELS into
# a single library with a C-style API for kernel dispatch.

# Library includes:
# - cutlass_library.h: API declarations
# - All compiled kernel instantiations
# - Handle types for GEMM operations
```

### Using the Pre-Built Library

```cmake
# In your project's CMakeLists.txt
find_package(CUTLASS REQUIRED)

add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE cutlass::cutlass)
```

```cpp
// Using the library API
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"
#include "cutlass/library/singleton.h"

cutlass::library::Handle handle(stream);

// Find the appropriate operation
auto op = cutlass::library::Singleton::get().operation_table.find_op(
  cutlass::library::GemmFunctionalKey(
    cutlass::library::Provider::kCUTLASS,
    cutlass::library::GemmKind::kGemm,
    cutlass::half_t(), cutlass::layout::RowMajor(),
    cutlass::half_t(), cutlass::layout::ColumnMajor(),
    float(), cutlass::layout::RowMajor(),
    cutlass::half_t(), cutlass::layout::RowMajor(),
    float()
  )
);

// Execute via the library handle
handle.gemm_problem_size({M, N, K});
handle.set_arguments(op, {...});
handle.run();
```

---

## 33.7 Generator Expressions for Kernel Filtering

The `CUTLASS_LIBRARY_KERNELS` option uses CMake generator expressions to filter kernels at configuration time. This mechanism matches against kernel names registered in the `cutlass_library` target.

### Filter Syntax

Filters use glob patterns applied to kernel operation names:

```cmake
# In tools/library/CMakeLists.txt
# Each kernel is added with a generator expression:
target_sources(cutlass_library PRIVATE
  $<$<STREQUAL:"${CUTLASS_LIBRARY_KERNELS}","all">:kernel_sm90_gemm_f16.cu>
  $<$<IN_LIST:kernel_sm90_gemm_f16,${CUTLASS_LIBRARY_KERNELS}>:kernel_sm90_gemm_f16.cu>
)
```

### Filter Patterns by Feature

```bash
# Filter by operation type
-D_CUTLASS_LIBRARY_KERNELS="*gemm*"          # Only GEMM
-D_CUTLASS_LIBRARY_KERNELS="*conv*"           # Only convolution
-D_CUTLASS_LIBRARY_KERNELS="*rank_k*"         # Only Rank-K updates

# Filter by data type
-D_CUTLASS_LIBRARY_KERNELS="*f16*"            # FP16 only
-D_CUTLASS_LIBRARY_KERNELS="*bf16*"           # BF16 only
-D_CUTLASS_LIBRARY_KERNELS="*f32*"            # FP32 only
-D_CUTLASS_LIBRARY_KERNELS="*tf32*"           # TF32 only
-D_CUTLASS_LIBRARY_KERNELS="*s8*"             # INT8 only
-D_CUTLASS_LIBRARY_KERNELS="*e4m3*"           # FP8 E4M3 only
-D_CUTLASS_LIBRARY_KERNELS="*e5m2*"           # FP8 E5M2 only

# Filter by architecture
-D_CUTLASS_LIBRARY_KERNELS="sm80*;sm90*"      # Ampere + Hopper

# Filter by operation class
-D_CUTLASS_LIBRARY_KERNELS="*tensorop*"       # Tensor Core only
-D_CUTLASS_LIBRARY_KERNELS="*simt*"           # SIMT (no Tensor Cores)

# Combined filters
-D_CUTLASS_LIBRARY_KERNELS="sm90*gemm*f16*;sm90*gemm*bf16*"
```

---

## 33.8 Multi-Architecture Builds

When targeting multiple GPU architectures, CUTLASS will compile separate kernel variants for each architecture. This increases build time and binary size but allows a single binary to run on multiple GPU generations.

```bash
# Multi-architecture build
cmake .. \
  -DCUTLASS_NVCC_ARCHS="80;86;89;90a" \
  -DCUTLASS_LIBRARY_KERNELS="*gemm*f16*"

# This compiles:
# - SM80 FP16 GEMM kernels
# - SM86 FP16 GEMM kernels
# - SM89 FP16 GEMM kernels
# - SM90 FP16 GEMM kernels
```

### Fat Binary Considerations

```cmake
# CMake generates appropriate -gencode flags for each architecture:
# -gencode arch=compute_80,code=sm_80
# -gencode arch=compute_86,code=sm_86
# -gencode arch=compute_89,code=sm_89
# -gencode arch=compute_90a,code=sm_90a

# For forward compatibility, also include PTX for the highest arch:
# -gencode arch=compute_90a,code=compute_90a
```

### Reducing Multi-Arch Build Time

```bash
# Use Ninja for parallel builds (recommended)
cmake .. -GNinja -DCUTLASS_NVCC_ARCHS="80;90a"
ninja -j8  # Limit parallel jobs to avoid memory exhaustion

# Or use Make with job limits
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a"
make -j$(nproc)

# Build only specific targets
ninja cutlass_library
ninja cutlass_profiler
```

---

## 33.9 CUDA Toolkit Version Requirements

Different CUTLASS features require different CUDA toolkit versions:

| Feature | Minimum CUDA Version | Notes |
|---|---|---|
| CUTLASS 2.x (SM70-SM80) | 11.0 | Basic functionality |
| CUTLASS 3.x (SM80-SM89) | 11.4 | C++17 required |
| TMA (SM90) | 11.8 | Tensor Memory Accelerator |
| WGMMA (SM90) | 11.8 | Warp Group MMA |
| Thread Block Clusters (SM90) | 12.0 | Recommended for full support |
| UMMA (SM100) | 12.x | Blackwell features |

```cmake
# Check CUDA version at configure time
if(CUDAToolkit_VERSION VERSION_LESS "11.8")
  message(WARNING "CUDA < 11.8: SM90 features will be limited")
endif()

if(CUDAToolkit_VERSION VERSION_LESS "12.0")
  message(WARNING "CUDA < 12.0: Some SM90 cluster features unavailable")
endif()
```

### NVCC-Specific Flags

```cmake
# Common NVCC flags used by CUTLASS
target_compile_options(my_target PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr    # Allow constexpr in device code
    --expt-extended-lambda      # Extended lambdas in device code
    -Xcudafe --display_error_number
  >
)

# Disable specific warnings
target_compile_options(my_target PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
  >
)
```

---

## 33.10 C++17 Standard Requirements

CUTLASS 3.x requires C++17 for both host and device compilation:

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
```

C++17 features used by CUTLASS:
- `constexpr if` for compile-time branching
- Structured bindings
- `std::aligned_storage_t`
- Fold expressions
- Inline variables
- Nested namespace declarations
- `std::byte`

---

## 33.11 Example CMake Configurations

### 33.11.1 Minimal Development Build

```bash
# Fast iteration for development - header-only, single arch
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF
```

### 33.11.2 Full Release Build with Profiler

```bash
# Complete build with profiler for benchmarking
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_LIBRARY_KERNELS="sm90*gemm*"
```

### 33.11.3 Library Build for Deployment

```bash
# Build a minimal library for deployment
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS="80;90a" \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_LIBRARY_KERNELS="sm80*gemm*f16*;sm90*gemm*f16*;sm90*gemm*bf16*"

cmake --build . --target cutlass_library
```

### 33.11.4 CI/CD Build Configuration

```bash
# Fast CI build with limited kernel set
cmake .. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_ENABLE_TESTS=ON \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_LIBRARY_KERNELS="sm90*simt*gemm*" \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0
```

### 33.11.5 Embedding CUTLASS in a Larger Project

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyDLFramework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add CUTLASS as subdirectory with minimal features
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/cutlass)

# Your library target
add_library(my_kernels
  src/gemm_kernels.cu
  src/attention_kernels.cu
)

target_include_directories(my_kernels PRIVATE
  ${CUTLASS_PATH}/include
  ${CUTLASS_PATH}/tools/util/include
)

target_compile_options(my_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -arch=sm_90a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -O3
  >
)
```

---

## 33.12 Build Troubleshooting

### 33.12.1 Out of Memory During Compilation

CUTLASS template instantiation is memory-intensive. NVCC can consume 10+ GB per translation unit.

```bash
# Reduce parallel jobs
make -j4  # Instead of -j$(nproc)

# Or limit with Ninja
ninja -j4

# Use the CUTLASS_LIBRARY_KERNELS filter to reduce kernel count
cmake .. -DCUTLASS_LIBRARY_KERNELS="sm90*gemm*f16*128x128*"

# Some kernels require more memory than others
# Reduce tile sizes in kernel instantiation to lower memory pressure
```

### 33.12.2 Long Compile Times

```bash
# Disable unused features
cmake .. \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_PROFILER=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF

# Use ccache for incremental builds
cmake .. -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Filter kernels aggressively
cmake .. -DCUTLASS_LIBRARY_KERNELS="sm90*gemm*f16*"

# Use single architecture
cmake .. -DCUTLASS_NVCC_ARCHS="90a"
```

### 33.12.3 NVCC Internal Compiler Errors

```bash
# Update CUDA toolkit to latest patch version
# Common issue with older CUDA versions

# Try reducing optimization level for problematic files
target_compile_options(my_target PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O2>  # Instead of -O3
)

# Increase GPU maximum register count if needed
target_compile_options(my_target PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=255>
)
```

### 33.12.4 Undefined Reference Errors

```bash
# Ensure CUTLASS headers are in the include path
target_include_directories(my_target PRIVATE
  ${CUTLASS_PATH}/include
  ${CUTLASS_PATH}/tools/util/include
)

# If using the library, ensure it is linked
target_link_libraries(my_target PRIVATE cutlass)

# Check that kernel instantiations are present
nm -C libcutlass.a | grep "gemm.*sm90.*f16"
```

### 33.12.5 Architecture Mismatch

```bash
# Verify the compiled architecture matches your GPU
nvidia-smi --query-gpu=compute_cap --format=csv

# Ensure CUTLASS_NVCC_ARCHS includes your GPU's compute capability
cmake .. -DCUTLASS_NVCC_ARCHS="90a"  # For H100

# Check runtime error messages for arch mismatch
# "no kernel image is available for execution on the device"
```

### 33.12.6 CMake Configuration Errors

```bash
# Clean build directory completely
rm -rf build && mkdir build && cd build

# Verbose CMake configuration
cmake .. --debug-output

# Check CUDA toolkit detection
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Explicitly set CUDA toolkit root
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda
```

### 33.12.7 Device Symbol Resolution

When linking CUTLASS device code, ensure proper separable compilation settings:

```cmake
# Enable separable compilation if needed
set_target_properties(my_target PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

---

## 33.13 Environment Variables

CUTLASS respects several environment variables during the build process:

| Variable | Purpose | Example |
|---|---|---|
| `CUTLASS_PATH` | Root of CUTLASS source tree | `/path/to/cutlass` |
| `CUDNN_ROOT_PATH` | Root of cuDNN installation | `/usr/local/cudnn` |
| `CUDA_TOOLKIT_ROOT_DIR` | Override CUDA toolkit location | `/usr/local/cuda` |
| `CUTLASS_LIBRARY_KERNELS` | Kernel filter (can also be CMake var) | `sm90*gemm*` |

### Runtime Environment Variables

| Variable | Purpose | Example |
|---|---|---|
| `CUTLASS_DEBUG_TRACE_LEVEL` | Override compile-time debug level | `2` |
| `CUDA_LAUNCH_BLOCKING` | Synchronize all kernel launches | `1` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Limit concurrent kernel streams | `4` |
