# CUTLASS: Quick Start Guide

## Prerequisites

### Software Requirements

| Requirement | Minimum Version | Recommended Version |
|---|---|---|
| CUDA Toolkit | 11.4+ | 12.0+ (for SM90/SM100) |
| CMake | 3.18+ | 3.24+ |
| C++ Compiler | C++17 support | GCC 9+, MSVC 2019+, Clang 12+ |
| Python | 3.6+ | 3.8+ (for Python bindings) |
| OS | Linux x86_64 | Ubuntu 20.04+ |

**CUDA Toolkit version notes:**
- CUDA 11.4+ required for basic CUTLASS functionality
- CUDA 11.8+ required for SM90 (Hopper) features including TMA and warp-group MMA
- CUDA 12.0+ recommended for SM90a (full Hopper feature set)
- CUDA 12.6+ required for SM100 (Blackwell) features including block-scaled MMA
- nvcc must be available in the system PATH

### Hardware Requirements

| GPU Architecture | Minimum GPU | Compute Capability |
|---|---|---|
| Volta | Tesla V100, GV100 | SM70 |
| Turing | Tesla T4, RTX 2080 | SM75 |
| Ampere | A100, A10, RTX 3090 | SM80 |
| Ada | L40, L40S, RTX 4090 | SM89 |
| Hopper | H100, H200 | SM90 / SM90a |
| Blackwell | B200, B100 | SM100 / SM100a |

A CUDA-capable NVIDIA GPU with Volta architecture or later is required. CUTLASS does not support pre-Volta GPUs for Tensor Core operations.

---

## Clone and Build Instructions

### Step 1: Clone the Repository

```bash
# Clone CUTLASS with submodules
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Initialize submodules (required for test dependencies)
git submodule update --init --recursive
```

### Step 2: Configure with CMake

```bash
# Create build directory
mkdir build && cd build

# Basic configuration (builds for current GPU architecture)
cmake ..

# Recommended: Specify target architecture(s)
cmake .. -DCUTLASS_NVCC_ARCHS="90a"
```

### Step 3: Build the Library

```bash
# Build CUTLASS (release mode)
cmake --build . -j$(nproc)

# Or using make directly
make -j$(nproc)
```

---

## Building for Specific Architectures

The `CUTLASS_NVCC_ARCHS` CMake variable controls which GPU architectures the kernels are compiled for. This is the most important build configuration option.

### Single Architecture

```bash
# Hopper H100 (SM90a -- full feature set including FP64 Tensor Cores)
cmake .. -DCUTLASS_NVCC_ARCHS="90a"

# Blackwell B200 (SM100a)
cmake .. -DCUTLASS_NVCC_ARCHS="100a"

# Blackwell B100 (SM100)
cmake .. -DCUTLASS_NVCC_ARCHS="100"

# Ampere A100 (SM80)
cmake .. -DCUTLASS_NVCC_ARCHS="80"

# Ada L40 (SM89)
cmake .. -DCUTLASS_NVCC_ARCHS="89"

# Turing T4 (SM75)
cmake .. -DCUTLASS_NVCC_ARCHS="75"

# Volta V100 (SM70)
cmake .. -DCUTLASS_NVCC_ARCHS="70"
```

### Multiple Architectures

```bash
# Build for both Ampere and Hopper
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a"

# Build for all major architectures
cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80;89;90a"
```

**Note:** Building for multiple architectures significantly increases compile time and binary size. For development, target only the architecture you are testing on.

### Additional CMake Configuration Options

```bash
cmake .. \
    -DCUTLASS_NVCC_ARCHS="90a" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DCUTLASS_ENABLE_TESTS=ON \
    -DCUTLASS_ENABLE_PROFILER=ON \
    -DCUTLASS_LIBRARY_OPERATIONS="gemm" \
    -DCUTLASS_LIBRARY_KERNELS="cutlass_tensorop_s*"
```

| CMake Option | Default | Description |
|---|---|---|
| `CUTLASS_NVCC_ARCHS` | Native GPU | Target architecture(s) |
| `CUTLASS_ENABLE_TESTS` | OFF | Build unit tests |
| `CUTLASS_ENABLE_PROFILER` | ON | Build the CUTLASS profiler |
| `CUTLASS_ENABLE_TOOLS` | ON | Build tools |
| `CUTLASS_ENABLE_LIBRARY` | ON | Build the CUTLASS library |
| `CUTLATT_ENABLE_EXAMPLES` | ON | Build SDK examples |
| `CUTLASS_LIBRARY_KERNELS` | "" | Filter which kernels to compile |
| `CUTLASS_LIBRARY_OPERATIONS` | "" | Filter which operation types |
| `CUTLASS_DEBUG_TRACE_LEVEL` | 0 | Debug trace verbosity (0-3) |

---

## Building the CUTLASS Profiler

The CUTLASS Profiler (`cutlass_profiler`) is a command-line tool for benchmarking and validating GEMM, convolution, and other operations across all supported data types and architectures.

### Build the Profiler

```bash
cmake .. -DCUTLASS_ENABLE_PROFILER=ON -DCUTLASS_NVCC_ARCHS="90a"
make cutlass_profiler -j$(nproc)
```

The executable will be at `tools/profiler/cutlass_profiler`.

### Basic Profiler Usage

```bash
# Profile all FP16 GEMM kernels on current GPU
./tools/profiler/cutlass_profiler --kernels=cutlass_simt_h* --m=1024 --n=1024 --k=1024

# Profile SM90 TMA FP16 kernels
./tools/profiler/cutlass_profiler --kernels=cutlass_sm90*e4m3* --m=2048 --n=2048 --k=2048

# Profile all GEMM operations and save results
./tools/profiler/cutlass_profiler --m=1024 --n=1024 --k=1024 --output=profile_results.csv

# Verify correctness of FP16 GEMM
./tools/profiler/cutlass_profiler --kernels=cutlass_simt_h* --verification-enabled=true

# Profile with specific batch size
./tools/profiler/cutlass_profiler --kernels=cutlass_simt* --m=1024 --n=1024 --k=1024 --batch-count=4
```

### Profiler Arguments

| Argument | Description |
|---|---|
| `--kernels=<pattern>` | Filter kernels by name pattern |
| `--m=<int>` | GEMM M dimension |
| `--n=<int>` | GEMM N dimension |
| `--k=<int>` | GEMM K dimension |
| `--batch-count=<int>` | Number of batches |
| `--alpha=<float>` | GEMM alpha scalar |
| `--beta=<float>` | GEMM beta scalar |
| `--verification-enabled=<bool>` | Enable/disable correctness verification |
| `--warmup-iterations=<int>` | Number of warmup iterations |
| `--profiling-iterations=<int>` | Number of profiling iterations |
| `--output=<path>` | Output CSV file path |

---

## Building Unit Tests

```bash
# Enable tests in CMake configuration
cmake .. -DCUTLASS_ENABLE_TESTS=ON -DCUTLASS_NVCC_ARCHS="90a"

# Build all tests
make cutlass_test_unit -j$(nproc)

# Build specific test categories
make cutlass_test_unit_gemm_device -j$(nproc)
make cutlass_test_unit_gemm_warp -j$(nproc)
make cutlass_test_unit_gemm_thread -j$(nproc)
make cutlass_test_unit_layout -j$(nproc)

# Run all unit tests
ctest --output-on-failure

# Run specific test category
ctest -R gemm_device --output-on-failure
```

### Test Categories

| Test Target | Description |
|---|---|
| `cutlass_test_unit_gemm_device` | Device-level GEMM tests |
| `cutlass_test_unit_gemm_kernel` | Kernel-level tests |
| `cutlass_test_unit_gemm_warp` | Warp-level MMA tests |
| `cutlass_test_unit_gemm_thread` | Thread-level GEMM tests |
| `cutlass_test_unit_layout` | Layout tests |
| `cutlass_test_unit_epilogue` | Epilogue tests |
| `cutlass_test_unit_conv_device` | Convolution device tests |
| `cutlass_test_unit_transform` | Transform operation tests |
| `cutlass_test_unit_reduction` | Reduction tests |

---

## Using CUTLASS as a Header-Only Library

CUTLASS is primarily a header-only library. You can include it in your project without building it as a standalone library.

### Method 1: Subdirectory in CMake

```cmake
# In your project's CMakeLists.txt
add_subdirectory(third_party/cutlass)

target_include_directories(your_target PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include
)

# Enable CUTLASS language support for CUDA
enable_language(CUDA)
```

### Method 2: FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v3.5.0
)
FetchContent_MakeAvailable(cutlass)

target_link_libraries(your_target PRIVATE cutlass)
```

### Method 3: Direct Include

```cmake
# Simply add the include directories
target_include_directories(your_target PRIVATE
    /path/to/cutlass/include
    /path/to/cutlass/tools/util/include
)
```

```cpp
// In your CUDA source file
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
```

---

## First GEMM Kernel Example

### CUTLASS 2.x GEMM Example

The following demonstrates a complete FP16 GEMM kernel using the CUTLASS 2.x API:

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

// Define the GEMM configuration
using GemmConfig = cutlass::gemm::device::Gemm<
    // Data types
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::RowMajor,                 // LayoutC
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OperationClass
    cutlass::arch::Sm80,                       // Architecture

    // Threadblock-level configuration
    cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape (Tensor Core)

    // Epilogue configuration
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,                       // ElementOutput
        8,                                      // ElementsPerAccess
        float,                                  // ElementAccumulator
        float                                   // ElementCompute
    >,

    // Threadblock swizzling
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,

    // Number of pipeline stages
    3                                          // Stages
>;

int run_gemm_2x(int M, int N, int K,
                cutlass::half_t* A, cutlass::half_t* B,
                cutlass::half_t* C, cutlass::half_t* D,
                float alpha, float beta,
                cudaStream_t stream = 0) {

    // Create GEMM arguments
    typename GemmConfig::Arguments args(
        {M, N, K},          // Problem size
        {A, K},              // PtrA and leading dimension (RowMajor -> lda=K)
        {B, K},              // PtrB and leading dimension (ColMajor -> ldb=K)
        {C, N},              // PtrC and leading dimension (RowMajor -> ldc=N)
        {D, N},              // PtrD and leading dimension (RowMajor -> ldd=N)
        {alpha, beta}        // Epilogue scalars
    );

    // Get workspace size
    size_t workspace_size = GemmConfig::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize and run GEMM
    GemmConfig gemm_op;
    cutlass::Status status = gemm_op.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        return -1;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        return -1;
    }

    return 0;
}
```

### CUTLASS 3.x GEMM Example (Hopper SM90)

The following demonstrates an FP16 GEMM kernel using the CUTLASS 3.x API with TMA and warp specialization for Hopper:

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/epilogue_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/util/packed_stride.hpp"

// Define shapes using CuTe compile-time constants
using ProblemShape = cutlass::Shape<int, int, int, int>;  // M, N, K, L
using TileShape = cutlass::Shape<_128, _128, _64>;         // TM, TN, TK
using ClusterShape = cutlass::Shape<_1, _2, _1>;           // CM, CN, CK

// Data types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

// Layouts
using LayoutA = cutlass::layout::RowMajor;  // K-major for optimal TMA
using LayoutB = cutlass::layout::ColumnMajor; // K-major for optimal TMA
using LayoutC = cutlass::layout::RowMajor;

// Use the Collective Builder to automatically select the best mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                // Architecture
    cutlass::gemm::collective::OpClassTensorOp,  // OpClass
    ElementA, LayoutA, 8,               // A: type, layout, alignment
    ElementB, LayoutB, 8,               // B: type, layout, alignment
    ElementAccumulator,                  // Accumulator type
    TileShape,                          // Tile shape
    ClusterShape,                       // Cluster shape
    cutlass::gemm::collective::StageCountAutoCarveout<0>,  // Auto stages
    cutlass::gemm::collective::KernelScheduleAuto          // Auto schedule
>::CollectiveOp;

// Epilogue: simple linear combination
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor,          // LayoutC
    cutlass::layout::RowMajor,          // LayoutD
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

// Build the kernel and device adapter
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int run_gemm_3x(int M, int N, int K,
                cutlass::half_t* A, cutlass::half_t* B,
                cutlass::half_t* C, cutlass::half_t* D,
                float alpha, float beta,
                cudaStream_t stream = 0) {

    // Create GEMM arguments with packed strides
    typename GemmDevice::GemmKernel::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,  // Mode
        {M, N, K, 1},                              // Problem shape (M, N, K, L)
        {alpha, beta},                              // Epilogue scalars
        cutlass::make_tuple(
            cutlass::make_Coord(M, K),             // A extent
            cutlass::make_Coord(K, N)              // B extent
        ),
        A,
        B,
        C,
        D,
        {K, 1},      // StrideA (RowMajor)
        {K, 1},      // StrideB (ColMajor)
        {N, 1},      // StrideC
        {N, 1}       // StrideD
    );

    // Get workspace and launch
    typename GemmDevice::GemmKernel::Params params =
        typename GemmDevice::GemmKernel::Params(args);

    dim3 grid = GemmDevice::GemmKernel::get_grid_shape(params);
    int smem_size = GemmDevice::GemmKernel::SharedStorageSize;

    // Launch kernel
    cutlass::Status status = cutlass::launch_kernel_on_workspace(
        stream, nullptr, 0,
        kernel_fn<GemmDevice::GemmKernel>,
        grid, params.cta_shape, smem_size,
        params
    );

    return (status == cutlass::Status::kSuccess) ? 0 : -1;
}
```

### Simplified CUTLASS 3.x Example Using API Wrappers

For the most common use cases, CUTLASS 3.x provides simplified API wrappers:

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/device_memory.h"

// Use the device-level API directly
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        cutlass::gemm::device::GemmShape<128, 128, 64>,
        cutlass::half_t, cutlass::layout::RowMajor,    // A
        cutlass::half_t, cutlass::layout::ColumnMajor,  // B
        cutlass::half_t, cutlass::layout::RowMajor,     // C/D
        float,                                            // Accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm90,
        cutlass::gemm::kernel::GemmUniversalMode::kGemm
    >
>;

// Or use the 2.x-compatible interface
cutlass::gemm::GemmCoord problem_size = {M, N, K};
typename Gemm::Arguments args(
    problem_size,
    A, {K},  // A ptr, lda
    B, {K},  // B ptr, ldb
    C, {N},  // C ptr, ldc
    D, {N},  // D ptr, ldd
    {alpha, beta}
);

Gemm gemm_op;
cutlass::Status status = gemm_op.initialize(args);
status = gemm_op();
```

---

## Launching a GEMM Kernel in CUDA

### Full Launch Sequence

The typical sequence for launching a CUTLASS GEMM kernel is:

```cpp
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

// 1. Allocate device memory
cutlass::half_t *d_A, *d_B, *d_C, *d_D;
size_t size_A = M * K * sizeof(cutlass::half_t);
size_t size_B = K * N * sizeof(cutlass::half_t);
size_t size_C = M * N * sizeof(cutlass::half_t);

cudaMalloc(&d_A, size_A);
cudaMalloc(&d_B, size_B);
cudaMalloc(&d_C, size_C);
cudaMalloc(&d_D, size_C);

// 2. Copy input data to device
cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

// 3. Create arguments
typename Gemm::Arguments args(
    {M, N, K},
    {d_A, K},   // RowMajor A: leading dim = K
    {d_B, K},   // ColumnMajor B: leading dim = K
    {d_C, N},
    {d_D, N},
    {1.0f, 0.0f}  // alpha=1, beta=0
);

// 4. Allocate workspace (if needed)
size_t workspace_size = Gemm::get_workspace_size(args);
void* workspace = nullptr;
if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
}

// 5. Initialize the GEMM operation
Gemm gemm_op;
cutlass::Status status = gemm_op.initialize(args, workspace);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "Initialization failed: "
              << cutlassGetStatusString(status) << std::endl;
    return -1;
}

// 6. Run the GEMM
status = gemm_op();
if (status != cutlass::Status::kSuccess) {
    std::cerr << "Execution failed: "
              << cutlassGetStatusString(status) << std::endl;
    return -1;
}

// 7. Wait for completion
cudaDeviceSynchronize();

// 8. Copy result back
cudaMemcpy(h_D, d_D, size_C, cudaMemcpyDeviceToHost);

// 9. Cleanup
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
cudaFree(d_D);
if (workspace) cudaFree(workspace);
```

---

## CUTLASS Library Generation and Usage

CUTLASS includes a library generator that produces compiled kernel libraries for integration into other frameworks (e.g., PyTorch, TensorFlow).

### Generating the Library

```bash
# Configure to build the library
cmake .. -DCUTLASS_ENABLE_LIBRARY=ON -DCUTLASS_NVCC_ARCHS="90a"

# Build the library
make cutlass_library -j$(nproc)
```

### Using Generated Library

```cpp
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

// Create a library handle
cutlass::library::Handle handle;

// Find a GEMM operation in the library
cutlass::library::GemmDescription const* gemm_desc;
auto status = cutlass::library::find_gemm_operation(
    &gemm_desc,
    cutlass::library::GemmFunctionalKey(
        cutlass::library::Provider::kCUTLASS,
        cutlass::library::GemmKind::kUniversal,
        cutlass::half_t,           // ElementA
        cutlass::layout::RowMajor, // LayoutA
        cutlass::half_t,           // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        cutlass::half_t,           // ElementC
        cutlass::layout::RowMajor, // LayoutC
        float,                     // ElementAccumulator
        float                      // ElementCompute
    )
);

// Launch the operation
cutlass::library::GemmArguments gemm_args;
gemm_args.problem_size = {M, N, K};
gemm_args.A = d_A;
gemm_args.B = d_B;
gemm_args.C = d_C;
gemm_args.D = d_D;
gemm_args.lda = K;
gemm_args.ldb = K;
gemm_args.ldc = N;
gemm_args.ldd = N;
gemm_args.alpha = &alpha;
gemm_args.beta = &beta;

status = handle.gemm(gemm_desc, gemm_args);
```

---

## Selective Compilation with CUTLASS_LIBRARY_KERNELS

To reduce compile time and binary size, you can selectively compile only specific kernels:

```bash
# Compile only SM90 FP16 kernels
cmake .. -DCUTLASS_LIBRARY_KERNELS="cutlass_sm90*e4m3*"

# Compile only SM80 FP16 SIMT kernels
cmake .. -DCUTLASS_LIBRARY_KERNELS="cutlass_simt_h*"

# Compile a specific kernel configuration
cmake .. -DCUTLASS_LIBRARY_KERNELS="cutlass_tensorop_s16816gemm_f16_128x128_32x3_nt_align8"

# Multiple patterns separated by semicolons
cmake .. -DCUTLASS_LIBRARY_KERNELS="cutlass_sm90*e4m3*;cutlass_sm80*f16*"
```

### Kernel Naming Convention

CUTLASS kernel names encode their configuration:

```
cutlass_{arch}_{opclass}_{MxNxK}gemm_{dtype}_{tileM}x{tileN}x{tileK}_{stages}_{layout}_{alignment}
```

Example: `cutlass_tensorop_s16816gemm_f16_128x128_32x3_tn_align8`
- `tensorop`: Using Tensor Cores
- `s16816`: Instruction shape 16x8x16
- `f16`: FP16 data type
- `128x128x32`: Threadblock tile shape
- `3`: Pipeline stages
- `tn`: Transpose A, non-transpose B
- `align8`: 8-element alignment

---

## Blackwell SM100 Kernel Instantiation Examples

CUTLASS 3.x provides specialized kernels for Blackwell (SM100) architecture including block-scaled MMA:

```bash
# Build for Blackwell
cmake .. -DCUTLASS_NVCC_ARCHS="100a"

# Profile Blackwell kernels
./tools/profiler/cutlass_profiler --kernels=cutlass_sm100*e4m3* --m=2048 --n=2048 --k=2048
```

### Block-Scaled GEMM on Blackwell

```cpp
// Block-scaled GEMM using NVFP4 with per-block scaling factors
// (CUTLASS 3.x, SM100+)
using BlockScaledGemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        cutlass::gemm::collective::CollectiveMma<
            cutlass::gemm::collective::KernelTmaWarpSpecializedBlockScaled,
            TileShape,
            ElementA, LayoutA,    // NVFP4 A with block scale factors
            ElementB, LayoutB,    // NVFP4 B with block scale factors
            TiledMma,
            ...
        >,
        CollectiveEpilogue
    >
>;
```

---

## Common Build Issues and Troubleshooting

### Issue 1: "Unsupported CUDA architecture"

```
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```

**Solution:** Ensure your CUDA toolkit version supports the target architecture. Check the compatibility table in the Prerequisites section.

### Issue 2: Excessive Compile Time

CUTLASS is template-heavy and can take a very long time to compile.

**Solutions:**
- Use `CUTLASS_LIBRARY_KERNELS` to compile only needed kernels
- Reduce the number of target architectures
- Use `CUTLASS_ENABLE_EXAMPLES=OFF` and `CUTLASS_ENABLE_TESTS=OFF`
- Use Ninja instead of Make: `cmake .. -G Ninja`

```bash
cmake .. -G Ninja \
    -DCUTLASS_NVCC_ARCHS="90a" \
    -DCUTLASS_ENABLE_EXAMPLES=OFF \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_LIBRARY_KERNELS="cutlass_sm90*e4m3*"
ninja -j$(nproc)
```

### Issue 3: Out of Memory During Compilation

CUTLASS template instantiation can require significant RAM during compilation.

**Solutions:**
- Reduce parallelism: `make -j4` instead of `make -j$(nproc)`
- Reduce target architectures to one
- Use selective kernel compilation

### Issue 4: "CMake Error: Could NOT find CUDAToolkit"

**Solution:** Ensure CUDA toolkit is installed and `nvcc` is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue 5: "undefined reference to cutlass::..."

**Solution:** CUTLASS is header-only for the template library. Ensure you are including the correct headers and that `CUTLASS_ENABLE_LIBRARY` is set if you are using the generated library.

### Issue 6: Kernel Fails at Runtime

**Solutions:**
- Verify the GPU architecture matches the compiled target (`nvidia-smi` to check GPU, `CUDA_VISIBLE_DEVICES` to select)
- Check alignment requirements (A and B pointers must be aligned to the type's alignment)
- Verify leading dimensions are correct for the layout
- Run with `CUDA_LAUNCH_BLOCKING=1` to get exact error locations

### Issue 7: Incorrect Results

**Solutions:**
- Verify matrix layouts match (RowMajor vs ColumnMajor)
- Check leading dimension values (RowMajor: lda = K for A, ColumnMajor: lda = M for A)
- Ensure alpha and beta values are correct
- Use the CUTLASS profiler with verification enabled to compare against reference implementations

### Issue 8: Build Fails with "static_assert: CuTe layout mismatch"

**Solution:** This indicates an incompatibility between the specified layout and the operation's requirements. Check that:
- Layout types match between A, B, and the MMA operation
- Alignment is sufficient for the chosen tile size
- The operation class (TensorOp vs SIMT) is appropriate for the architecture

---

## Summary

This quick start guide covers the essential steps to get CUTLASS running:
1. Install prerequisites (CUDA 11.4+, CMake 3.18+, C++17 compiler)
2. Clone the repository with submodules
3. Configure with CMake specifying the target architecture
4. Build the library, profiler, or integrate as a header-only dependency
5. Write your first GEMM kernel using either the 2.x or 3.x API
6. Use the profiler to benchmark and validate kernels
7. Selectively compile kernels to reduce build time

For detailed information about the code organization, data types, and layout system, see the subsequent reference chapters.
