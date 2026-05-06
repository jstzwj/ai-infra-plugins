# CUTLASS: Code Organization

## Main Directory Structure

The CUTLASS repository is organized into several top-level directories, each serving a distinct purpose. Understanding this structure is essential for navigating the codebase and finding the components you need.

```
cutlass/
├── include/           # Core template library (header-only)
│   ├── cutlass/       # Main CUTLASS headers
│   ├── cute/          # CuTe tensor library (3.x)
│   └── cutlass_cpp/   # C++ standard library shims
├── tools/             # Profiler, library generator, utilities
│   ├── profiler/      # cutlass_profiler tool
│   ├── library/       # Library generation framework
│   └── util/          # Utility headers (device memory, reference, etc.)
├── examples/          # 95+ SDK examples
├── test/              # Unit tests and benchmarks
├── python/            # Python bindings and DSL
├── docs/              # Generated documentation
├── media/             # Documentation source files and images
└── cmake/             # CMake modules and utilities
```

---

## include/cutlass/ -- Core Template Library

The `include/cutlass/` directory contains the entire CUTLASS template library as header files. This is the primary directory for kernel developers. It is organized by functional scope:

```
include/cutlass/
├── arch/              # Architecture-specific features and instructions
├── gemm/              # GEMM operations at all hierarchy levels
├── conv/              # Convolution operations
├── epilogue/          # Post-GEMM processing (epilogue) operations
├── layout/            # Matrix and tensor layout descriptors
├── transform/         # Layout and domain transformations
├── reduction/         # Reduction kernel components
├── pipeline/          # Execution pipeline abstractions
├── thread/            # Thread-level operations and primitives
├── platform/          # Platform abstractions
├── numeric_types.h    # Core numeric type definitions
├── cutlass.h          # Main header, version info, macros
├── array.h            # Static array type
├── tensor_ref.h       # Tensor reference (pointer + layout)
├── tensor_view.h      # Tensor view (pointer + layout + extent)
├── subbyte_ref.h      # Sub-byte element references
├── coord.h            # Multi-dimensional coordinate types
├── matrix.h           # Matrix coordinate and shape types
├── matrix_coord.h     # Matrix coordinate arithmetic
├── fast_math.h        # Fast math approximations
├── functional.h       # Arithmetic functors (multiply, plus, etc.)
├── numeric_conversion.h  # Type conversion utilities
├── half.h             # FP16 type definition
├── bfloat16.h         # BF16 type definition
├── tfloat32.h         # TF32 type definition
├── float8.h           # FP8 type definitions
├── uint_integer.h     # Sub-byte integer types
├── complex.h          # Complex number support
├── device_kernel.h    # CUDA kernel launch utilities
├── kernel_hardware_info.hpp  # Hardware capability queries
└── cluster_query.hpp  # Cluster launch capability queries
```

---

## arch/ -- Architecture-Specific Features

The `arch/` directory contains abstractions for GPU hardware instructions and features, providing architecture-specific implementations that are selected at compile time.

```
include/cutlass/arch/
├── arch.h               # Architecture tag definitions (SM70, SM75, SM80, SM90, SM100)
├── mma.h                # Base MMA operation declarations
├── mma_sm50.h           # Maxwell (SM50) SIMT matrix multiply
├── mma_sm60.h           # Pascal (SM60) SIMT matrix multiply
├── mma_sm70.h           # Volta (SM70) Tensor Core FP16 MMA (16x16x4)
├── mma_sm75.h           # Turing (SM75) Tensor Core MMA (FP16, INT8, INT4, INT1)
├── mma_sm80.h           # Ampere (SM80) Tensor Core MMA (FP16, BF16, TF32, FP64, INT8)
├── mma_sm89.h           # Ada (SM89) Tensor Core MMA (FP8)
├── mma_sm90.h           # Hopper (SM90) warp-group MMA (wgmma.mma_async)
├── mma_sm100.h          # Blackwell (SM100) block-scaled MMA
├── mma_generic.h        # Generic MMA fallback using SIMT
├── cache_operation.h    # Cache hint definitions (evict_first, evict_last, etc.)
├── memory.h             # Architecture-specific memory operations (ldg, sts, etc.)
├── memory_sm75.h        # Turing async memory operations
├── memory_sm80.h        # Ampere cp.async operations
├── memory_sm90.h        # Hopper TMA (Tensor Memory Accelerator) operations
├── barrier.h            # Barrier and synchronization primitives
├── cluster_barrier.hpp  # Cluster-level barrier (SM90+)
├── reg_reconfig.h       # Register reconfiguration for warp-group operations
├── warp.h               # Warp-level primitives
└── wmma.h               # WMMA (Warp Matrix Multiply-Accumulate) API
```

### Key Architecture Files

**`arch.h`** defines architecture tag types used throughout CUTLASS for compile-time dispatch:

```cpp
namespace cutlass::arch {
struct Sm70 {};   // Volta
struct Sm75 {};   // Turing
struct Sm80 {};   // Ampere
struct Sm89 {};   // Ada
struct Sm90 {};   // Hopper
struct Sm100 {};  // Blackwell

// Convenience aliases
using SmMin = Sm70;
using SmMax = Sm100;
}
```

**`mma_sm*.h`** files contain the actual Tensor Core instruction wrappers. For example, SM80 FP16 MMA:

```cpp
// 16x8x16 FP16 MMA instruction (Ampere Tensor Core)
template <>
struct Mma<gemm::GemmShape<16, 8, 16>, 1, half_t, layout::ColumnMajor,
           half_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
    // D = A * B + C
    CUTLASS_DEVICE void operator()(Array<float, 4> &D,
                                   Array<half_t, 8> const &A,
                                   Array<half_t, 4> const &B,
                                   Array<float, 4> const &C) {
        // Maps to nvcuda::wmma::mma_sync or mma.sync instruction
    }
};
```

---

## gemm/ -- GEMM Operations

The `gemm/` directory contains GEMM-specific components at every level of the execution hierarchy. This is the largest and most important directory in CUTLASS.

```
include/cutlass/gemm/
├── gemm.h                # GEMM shape, coord, and enumeration definitions
├── gemm_enumerated_types.h  # Enumerated types for GEMM configuration
│
│   # ---- Device Layer ----
├── device/
│   ├── gemm.h                  # CUTLASS 2.x device-level GEMM (legacy)
│   ├── gemm_universal.h        # CUTLASS 2.x universal GEMM device
│   ├── gemm_universal_adapter.h # CUTLASS 3.x device adapter
│   ├── gemm_array.h            # Array GEMM device (batched)
│   ├── gemm_splitk_parallel.h  # Split-K parallel GEMM device
│   └── gemm_complex.h          # Complex number GEMM device
│
│   # ---- Kernel Layer ----
├── kernel/
│   ├── gemm_universal.hpp      # CUTLASS 3.x universal GEMM kernel
│   ├── gemm_universal.h        # CUTLASS 2.x universal GEMM kernel
│   ├── gemm_pipelined.h        # 2.x pipelined GEMM kernel
│   ├── gemm_batched.h          # 2.x batched GEMM kernel
│   ├── default_gemm_universal.h # Default GEMM kernel configuration
│   └── tile_scheduler.hpp      # Tile scheduling strategies
│
│   # ---- Collective Layer (3.x) ----
├── collective/
│   ├── collective_builder.hpp  # Auto-selects optimal collective
│   ├── collective_mma.hpp      # Collective MMA mainloop
│   ├── collective_mma_sm90.hpp # SM90 TMA warp-specialized MMA
│   ├── collective_mma_sm100.hpp # SM100 block-scaled MMA
│   ├── mma_pv.hpp              # Persistent visit tile scheduler
│   └── default_sm90.hpp        # SM90 default configurations
│
│   # ---- Threadblock Layer ----
├── threadblock/
│   ├── mma_pipelined.h         # 2.x pipelined threadblock MMA
│   ├── mma_singlestage.h       # 2.x single-stage threadblock MMA
│   ├── mma_multistage.h        # 2.x multi-stage threadblock MMA
│   ├── mma_base.h              # 2.x threadblock MMA base class
│   ├── threadblock_swizzle.h   # Threadblock swizzling strategies
│   └── default_mma.h           # Default MMA configuration
│
│   # ---- Warp Layer ----
├── warp/
│   ├── mma_simt.h              # SIMT (scalar) warp MMA
│   ├── mma_simt_tile_iterator.h # SIMT tile iterator
│   ├── mma_tensor_op.h         # Tensor Core warp MMA
│   ├── mma_tensor_op_tile_iterator.h # Tensor Core tile iterator
│   ├── mma_tensor_op_tile_iterator_sm80.h # SM80-specific iterator
│   ├── mma_tensor_op_fast_f32.h # Fast FP32 using Tensor Cores
│   └── default_mma_tensor_op.h # Default warp MMA for Tensor Cores
│
│   # ---- Thread Layer ----
├── thread/
│   ├── mma.h                   # Thread-level matrix multiply
│   ├── mma_generic.h           # Generic thread-level MMA
│   └── block_swizzle.h         # Block-level swizzling
│
│   # ---- Scheduler ----
├── threadblock_index_mma.h     # Threadblock index computation
├── dispatch_tracker.hpp        # Kernel dispatch tracking
└── gemm_make_features.h        # Feature flag computation
```

### GEMM Template Hierarchy (CUTLASS 2.x)

In CUTLASS 2.x, the GEMM hierarchy is expressed as nested template parameters:

```
device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OpClass, ArchTag,
    ThreadblockShape,           // gemm/threadblock/ level
    WarpShape,                  // gemm/warp/ level
    InstructionShape,           // arch/ level
    Epilogue,
    ThreadblockSwizzle,
    Stages,
    SplitKSerial,
    Operator
>
```

### GEMM Template Hierarchy (CUTLASS 3.x)

In CUTLASS 3.x, the hierarchy is flattened into collective operations:

```
device::GemmUniversalAdapter<
    kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,      // Replaces threadblock + warp layers
        CollectiveEpilogue,      // Replaces epilogue layer
        TileScheduler
    >
>
```

---

## conv/ -- Convolution Operations

```
include/cutlass/conv/
├── conv.h                    # Convolution enumerations and types
├── conv2d_problem_size.h     # 2D convolution problem size
├── conv3d_problem_size.h     # 3D convolution problem size
├── device/
│   ├── implicit_gemm_convolution.h  # Implicit GEMM convolution device
│   └── implicit_gemm_multistage.h   # Multi-stage implicit GEMM
├── kernel/
│   ├── implicit_gemm_convolution.h  # Implicit GEMM convolution kernel
│   ├── implicit_gemm_convolution_fusion.h  # Fused convolution kernel
│   └── default_conv2d.h              # Default 2D convolution config
├── threadblock/
│   ├── implicit_gemm_pipelined.h     # Pipelined implicit GEMM
│   ├── implicit_gemm_multistage.h    # Multi-stage implicit GEMM
│   └── conv2d_tile_iterator.h        # Convolution tile iterator
├── warp/
│   └── implicit_gemm_wmma.h          # WMMA-based implicit GEMM
├── collective/
│   ├── collective_conv.hpp            # 3.x collective convolution
│   └── conv3d_sm90.hpp               # SM90 3D convolution
└── default/
    ├── conv2d_direct.h                 # Direct 2D convolution config
    └── conv3d_direct.h                 # Direct 3D convolution config
```

---

## epilogue/ -- Post-Processing Operations

The `epilogue/` directory contains components that process GEMM results after the MMA operation, including writing to global memory and applying element-wise operations.

```
include/cutlass/epilogue/
├── epilogue.h                     # Epilogue base types
├── threadblock/
│   ├── epilogue.h                 # 2.x default epilogue
│   ├── epilogue_base.h            # Epilogue base class
│   ├── default_epilogue.h         # Default epilogue configuration
│   ├── epilogue_workspace.h       # Workspace management
│   └── output_tile_thread_map.h   # Output tile thread mapping
├── warp/
│   ├── tile_iterator.h            # Warp-level output tile iterator
│   ├── fragment_iterator.h        # Fragment iterator for epilogue
│   └── default_warp_iterator.h    # Default warp iterator
├── thread/
│   ├── linear_combination.h       # D = alpha * AB + beta * C
│   ├── linear_combination_bias.h  # D = alpha * AB + beta * C + bias
│   ├── linear_combination_relu.h  # D = relu(alpha * AB + beta * C)
│   ├── linear_combination_gelu.h  # D = gelu(alpha * AB + beta * C)
│   ├── linear_combination_sigmoid.h  # D = sigmoid(alpha * AB + beta * C)
│   ├── linear_combination_clamp.h    # D = clamp(alpha * AB + beta * C)
│   ├── conversion_op.h            # Type conversion in epilogue
│   └── reduction.h                # Epilogue reduction
├── collective/
│   ├── default_epilogue.hpp       # 3.x default epilogue
│   ├── epilogue_builder.hpp       # 3.x epilogue auto-builder
│   ├── sm70_epilogue_vectorized.hpp  # SM70 vectorized epilogue
│   ├── sm90_epilogue_tma_warpspecialized.hpp  # SM90 TMA epilogue
│   └── sm100_epilogue.hpp         # SM100 epilogue
└── fusion/
    ├── visitor.hpp                # Epilogue visitor pattern
    ├── builder.hpp                # Epilogue fusion builder
    └── sm90_visitor_tma_warpspecialized.hpp  # SM90 TMA fusion visitor
```

### Epilogue Fusion Examples

CUTLASS supports fusing element-wise operations into the GEMM epilogue:

```cpp
// Linear combination with ReLU activation
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,     // Output type
    8,                    // Elements per access
    float,                // Accumulator type
    float,                // Compute type
    cutlass::epilogue::thread::Identity  // Additional activation (Identity = none)
>;

// Linear combination with bias and GELU
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasGelu<
    cutlass::half_t,     // Output type
    8,                    // Elements per access
    float,                // Accumulator type
    float,                // Compute type
    cutlass::epilogue::thread::GELU  // GELU activation
>;
```

---

## layout/ -- Matrix and Tensor Layouts

```
include/cutlass/layout/
├── layout.h              # Layout base types and tags
├── matrix.h              # RowMajor, ColumnMajor layout definitions
├── tensor.h              # Tensor layout definitions (NHWC, etc.)
├── pitch_linear.h        # PitchLinear layout for CUDA memory
├── interleaved.h         # Interleaved layouts for Tensor Core efficiency
├── tensor_op_multiplicand.h  # TensorOp multiplicand layouts per SM
├── column_major_interleaved.h # Column-major interleaved layout
├── row_major_interleaved.h    # Row-major interleaved layout
├── permute.h             # Permutation layout transforms
├── transpose.h           # Layout transpose operations
└── vector.h              # Vector layout utilities
```

### Layout Tags

```cpp
namespace cutlass::layout {
struct RowMajor {          // Row-major: stride = {N, 1}
    static Layout::kRowMajor operator()();
};
struct ColumnMajor {       // Column-major: stride = {1, M}
    static Layout::kColumnMajor operator()();
};
struct RowMajorInterleaved<int InterleavedK>;  // Interleaved row-major
struct ColumnMajorInterleaved<int InterleavedK>; // Interleaved col-major
}
```

---

## transform/ -- Layout and Domain Transformations

```
include/cutlass/transform/
├── threadblock/
│   ├── regular_tile_iterator.h           # Regular tile iterator
│   ├── regular_tile_access_iterator.h    # Regular tile access iterator
│   ├── predicated_tile_access_iterator.h # Predicated tile access
│   ├── predicated_tile_iterator.h        # Predicated tile iterator
│   ├── tile_iterator.h                   # Base tile iterator
│   └── transpose.h                       # Transpose transforms
├── thread/
│   ├── tile_iterator.h                   # Thread-level tile iterator
│   └── tile_iterator_simt.h              # SIMT tile iterator
├── warp/
│   └── tile_iterator.h                   # Warp-level tile iterator
├── pitch_linear_thread_map.h             # Pitch-linear thread mapping
└── filter.hpp                            # Predicated filtering
```

---

## reduction/ -- Reduction Kernels

```
include/cutlass/reduction/
├── tensor_reduce.h           # Tensor reduction operation
├── tensor_reduce_affine.h    # Affine reduction
├── tensor_reduce_affine_strided.h  # Strided affine reduction
├── device/
│   ├── tensor_reduce.h       # Device-level tensor reduction
│   └── tensor_reduce_affine_strided.h
├── thread/
│   └── reduction.h           # Thread-level reduction operations
├── kernel/
│   ├── tensor_reduce.hpp     # Reduction kernel
│   └── reduce_split_k.h      # Split-K reduction kernel
└── split_tensor_reduce.h     # Split reduction
```

---

## pipeline/ -- Execution Pipelines

```
include/cutlass/pipeline/
├── pipeline.hpp              # Base pipeline types and concepts
├── pipeline_tma.hpp          # TMA-based pipeline (SM90+)
├── pipeline_sync.hpp         # Synchronization primitives
├── pipeline_sm70.hpp         # Volta pipeline (double buffering)
├── pipeline_sm80.hpp         # Ampere pipeline (cp.async based)
├── pipeline_sm90.hpp         # Hopper pipeline (TMA based)
├── pipeline_sm90_tma_gather.hpp   # TMA gather pipeline
├── pipeline_sm90_tma_store.hpp    # TMA store pipeline
└── warpgroup_pipeline.hpp    # Warpgroup pipeline coordination
```

---

## thread/ -- Thread-Level Operations

```
include/cutlass/thread/
├── thread.h            # Thread-level basic operations
├── matrix.h            # Thread-level matrix operations
├── reduction.h         # Thread-level reduction
└── multiply_add.h      # Thread-level multiply-add functors
```

---

## include/cute/ -- CuTe Tensor Library

CuTe is the tensor algebra library introduced in CUTLASS 3.x. It provides composable abstractions for tensors, layouts, and memory operations.

```
include/cute/
├── cute.hpp              # Main CuTe include-all header
├── config.hpp            # CuTe configuration macros
│
├── algorithm/            # Algorithmic primitives
│   ├── copy.hpp          # copy() - memory copy with auto-vectorization
│   ├── gemm.hpp          # gemm() - matrix multiply-accumulate
│   ├── coalesce.hpp      # Layout coalescing
│   ├── fill.hpp          # Tensor fill operations
│   ├── axpby.hpp         # A*X + B*Y operations
│   ├── axpby_if.hpp      # Conditional A*X + B*Y
│   ├── memset.hpp        # Memory set operations
│   └── tuple_algorithms.hpp # Tuple manipulation algorithms
│
├── arch/                 # Architecture-specific CuTe operations
│   ├── copy.hpp          # Architecture-specific copy implementations
│   ├── copy_sm50.hpp     # SM50 copy
│   ├── copy_sm75.hpp     # SM75 copy
│   ├── copy_sm80.hpp     # SM80 copy (cp.async)
│   ├── copy_sm90.hpp     # SM90 copy (TMA)
│   ├── copy_sm100.hpp    # SM100 copy
│   ├── mma.hpp           # Architecture-specific MMA implementations
│   ├── mma_sm61.hpp      # SM61 MMA (DP4A)
│   ├── mma_sm70.hpp      # SM70 MMA (wmma)
│   ├── mma_sm75.hpp      # SM75 MMA
│   ├── mma_sm80.hpp      # SM80 MMA (mma.sync)
│   ├── mma_sm90.hpp      # SM90 MMA (wgmma)
│   ├── mma_sm100.hpp     # SM100 MMA (block-scaled)
│   ├── clustered_copy.hpp  # Cluster-aware copy
│   └── tmem.hpp          # Tensor memory operations (SM100+)
│
├── atom/                 # Atomic operations and MMA atoms
│   ├── copy_atom.hpp     # Copy atom definitions
│   ├── mma_atom.hpp      # MMA atom definitions
│   ├── sm90_traits.hpp   # SM90 MMA traits
│   ├── sm100_traits.hpp  # SM100 MMA traits
│   └── auto_copy.hpp     # Auto copy atom selection
│
├── container/            # Container types
│   ├── array.hpp         # Static array
│   ├── tuple.hpp         # Compile-time tuple
│   ├── alignment.hpp     # Alignment utilities
│   └── intrusive_ptr.hpp # Intrusive pointer
│
├── numeric/              # Numeric types and math
│   ├── numeric_types.hpp # CuTe numeric type definitions
│   ├── real.hpp          # Real number operations
│   ├── complex.hpp       # Complex number operations
│   ├── bfloat16.hpp      # BF16 type
│   ├── float8.hpp        # FP8 types
│   ├── int.hpp           # Integer types
│   └── math.hpp          # Math functions
│
├── util/                 # Utilities
│   ├── debug.hpp         # Debug printing for layouts and tensors
│   ├── print.hpp         # Pretty printing
│   ├── type_traits.hpp   # Type traits
│   ├── env.hpp           # Environment variable utilities
│   └── footprint.hpp     # Memory footprint calculation
│
├── tensor.hpp            # Tensor class definition
├── layout.hpp            # Layout class definition
├── layout_layout.hpp     # Layout composition
├── shape.hpp             # Shape hierarchy
├── stride.hpp            # Stride hierarchy
├── int_tuple.hpp         # Integer tuple operations
├── coordinate.hpp        # Coordinate types
├── smem_layout.hpp       # Shared memory layout utilities
├── swizzle.hpp           # Swizzle operations
├── tile.hpp              # Tiling operations
├── pointer.hpp           # Pointer and memory space types
├── memory.hpp            # Memory space tags (global, shared, register)
├── predicate.hpp         # Predicated operations
├── array.hpp             # CuTe array
├── numeric_conversion.hpp # Numeric type conversion
└── instance.hpp          # Instance/fragment utilities
```

### Key CuTe Concepts and Files

**`tensor.hpp`**: Defines `Tensor<Engine, Layout>` -- the core tensor abstraction.

**`layout.hpp`**: Defines `Layout<Shape, Stride>` -- the mapping from coordinates to offsets.

**`shape.hpp`** and **`stride.hpp`**: Define the hierarchical shape and stride types that compose layouts.

**`algorithm/copy.hpp`**: The `copy()` function that performs vectorized memory copies between tensors in any memory space.

**`algorithm/gemm.hpp`**: The `gemm()` function that wraps Tensor Core MMA operations.

**`arch/mma_sm*.hpp`**: Architecture-specific implementations of MMA operations that map to Tensor Core instructions.

---

## tools/ -- Profiler, Library Generator, Utilities

```
tools/
├── profiler/
│   ├── include/
│   │   └── cutlass_profiler/     # Profiler library headers
│   │       ├── profiler.h        # Main profiler class
│   │       ├── gemm_profiler.h   # GEMM profiling
│   │       ├── conv_profiler.h   # Convolution profiling
│   │       ├── performance_result.h  # Result types
│   │       ├── options.h         # Command-line options
│   │       └── problem_space.h   # Problem space definition
│   ├── src/                       # Profiler implementation
│   └── CMakeLists.txt
│
├── library/
│   ├── include/
│   │   └── cutlass/library/      # Library headers
│   │       ├── library.h          # Library API
│   │       ├── handle.h           # Operation handle
│   │       ├── gemm_operation.h   # GEMM operation descriptors
│   │       ├── conv_operation.h   # Convolution operation descriptors
│   │       ├── manifest.h         # Operation manifest
│   │       ├── gemm_types.h       # GEMM type definitions
│   │       └── internal.h         # Internal types
│   ├── src/                       # Library implementation
│   └── CMakeLists.txt
│
├── util/
│   ├── include/
│   │   └── cutlass/util/
│   │       ├── device_memory.h    # Device memory allocation
│   │       ├── reference/
│   │       │   ├── device/        # Device reference implementations
│   │       │   └── host/          # Host reference implementations
│   │       ├── gemm_testbed.h     # GEMM test utilities
│   │       ├── debug.h            # Debug utilities
│   │       ├── tensor_host_io.h   # Host tensor I/O
│   │       ├── trace_command.h    # Tracing utilities
│   │       ├── packed_stride.hpp  # Stride packing utilities
│   │       ├── command_line.h     # Command-line parsing
│   │       └── exceptions.h       # Exception types
│   └── CMakeLists.txt
│
└── generator/
    ├── include/
    │   └── cutlass/generator/     # Kernel generator utilities
    └── src/
```

---

## examples/ -- SDK Examples

The `examples/` directory contains 95+ example programs demonstrating CUTLASS usage across various configurations, architectures, and operations.

```
examples/
├── 00_basic_gemm/               # Basic GEMM example (2.x)
├── 01_cutlass_utilities/        # CUTLASS utility classes
├── 02_dump_reg_shmem/           # Register and shared memory analysis
├── 03_visualize_layout/         # Layout visualization
├── 04_tile_iterator/            # Tile iterator usage
├── 05_batched_gemm/             # Batched GEMM
├── 06_splitK_gemm/              # Split-K GEMM
├── 07_volta_tensorop_gemm/      # Volta (SM70) Tensor Core GEMM
├── 08_turing_tensorop_gemm/     # Turing (SM75) Tensor Core GEMM
├── 09_ampere_tensorop_gemm/     # Ampere (SM80) Tensor Core GEMM
├── 10_planar_complex/           # Planar complex GEMM
├── 11_gemm_bias_relu/           # GEMM with bias and ReLU fusion
├── 12_gemm_bias_epilogue/       # GEMM with bias epilogue
├── 13_two_tensor_op_fusion/     # Fused dual-GEMM operation
├── 14_ampere_tf32_tensorop_gemm/  # TF32 Tensor Core GEMM
├── 15_ampere_bf16_tensorop_gemm/  # BF16 Tensor Core GEMM
├── 16_ampere_int8_tensorop_gemm/  # INT8 Tensor Core GEMM
├── 17_ampere_fused_gemm/        # Fused GEMM operations
├── 18_ampere_gemm_universal/    # Universal GEMM for Ampere
├── 19_ampere_tensorop_conv/     # Tensor Core convolution
├── 20_simt_gemm/                # SIMT (non-Tensor-Core) GEMM
├── 21_quaternion_gemm/          # Quaternion GEMM
├── 22_quaternion_conv/          # Quaternion convolution
├── 23_ampere_gemm/              # General Ampere GEMM
├── 24_gemm_grouped/             # Grouped GEMM
├── 25_ampere_gemm_workspace/    # Workspace usage
├── 26_amperem_bf16_dual_gemm/   # Dual BF16 GEMM
├── 27_gemm_softmax_gemm/        # GEMM-Softmax-GEMM fusion
├── 28_gemm_permute/             # GEMM with permutation
├── 29_ampere_gemm_epi_tma/     # GEMM with TMA epilogue
├── 30_hopper_gemm_universal/    # Hopper universal GEMM (3.x)
├── 31_hopper_warp_specialized_gemm/  # Warp-specialized GEMM
├── 32_hopper_multiple_stages/   # Multiple pipeline stages
├── 33_hopper_gemm_with_epilogue/  # Hopper GEMM with epilogue fusion
├── 34_hopper_gemm_fused/        # Fused Hopper GEMM
├── 35_hopper_gemm_broadcast/    # GEMM with broadcast
├── 36_hopper_tensorop_gemm/     # Hopper Tensor Core GEMM
├── 37_hopper_gemm_grouped/      # Grouped GEMM on Hopper
├── 38_hopper_conv/              # Hopper convolution
├── 39_gather_gemm/              # Gather GEMM
├── 40_cutlass_py/               # Python interface examples
├── 41_hopper_ptr_array_gemm/    # Pointer-array GEMM on Hopper
├── 42_hopper_gemm_epi_collective/  # Collective epilogue examples
├── 43_stream_k/                 # Stream-K GEMM
├── 44_hopper_gemm_epilogue_fusion/  # Epilogue fusion examples
├── 45_blackwell_gemm/           # Blackwell GEMM (SM100)
├── 46_blackwell_block_scaled/   # Block-scaled GEMM (SM100)
├── 47_blackwell_mixed_dtype/    # Mixed-dtype GEMM (SM100)
├── 48_hopper_gemm_dual_gemm/    # Dual GEMM on Hopper
├── 49_hopper_gemm_layernorm/    # GEMM + LayerNorm fusion
├── 50_hopper_fused_attn/        # Fused attention kernel
├── 51_hopper_ptr_array_batched_gemm/  # Ptr-array batched GEMM
├── 52_blackwell_nvfp4_gemm/     # NVFP4 GEMM on Blackwell
├── 53_sm100_cp_reduce/          # SM100 copy-reduce
└── common/                      # Shared example utilities
    ├── cutlass_b2b_gemm.h       # Back-to-back GEMM utilities
    ├── helper.h                  # CUDA helper macros
    └── print.hpp                 # Printing utilities
```

### Example Organization by Architecture

| Prefix | Architecture | API |
|---|---|---|
| `07_` | Volta SM70 | 2.x |
| `08_` | Turing SM75 | 2.x |
| `09_` - `29_` | Ampere SM80 | 2.x |
| `30_` - `43_` | Hopper SM90 | 3.x |
| `45_` - `53_` | Blackwell SM100 | 3.x |

---

## test/ -- Unit Tests and Benchmarks

```
test/
├── unit/
│   ├── cutlass/
│   │   ├── core/             # Core type tests
│   │   │   ├── test_coord.cu
│   │   │   ├── test_layout.cu
│   │   │   ├── test_matrix.cu
│   │   │   ├── test_numeric_conversion.cu
│   │   │   └── test_tensor.cu
│   │   ├── gemm/
│   │   │   ├── device/       # Device-level GEMM tests
│   │   │   │   ├── test_gemm_device_*.cu   # Various data type tests
│   │   │   │   └── simt_*.cu               # SIMT-specific tests
│   │   │   ├── kernel/       # Kernel-level tests
│   │   │   ├── warp/         # Warp-level MMA tests
│   │   │   │   ├── test_mma_sm70.cu
│   │   │   │   ├── test_mma_sm75.cu
│   │   │   │   ├── test_mma_sm80.cu
│   │   │   │   ├── test_mma_sm90.cu
│   │   │   │   └── test_mma_sm100.cu
│   │   │   └── thread/       # Thread-level MMA tests
│   │   ├── conv/
│   │   │   └── device/       # Convolution device tests
│   │   ├── epilogue/         # Epilogue tests
│   │   ├── transform/        # Transform tests
│   │   ├── reduction/        # Reduction tests
│   │   └── arch/             # Architecture tests
│   ├── cute/
│   │   ├── test_layout.cpp   # CuTe layout tests
│   │   ├── test_tensor.cpp   # CuTe tensor tests
│   │   ├── test_algorithm.cpp # CuTe algorithm tests
│   │   └── test_mma.cpp      # CuTe MMA tests
│   └── CMakeLists.txt
│
├── examples/                 # Example verification tests
│   └── CMakeLists.txt
│
├── python/                   # Python binding tests
│   ├── test_gemm.py
│   ├── test_conv.py
│   └── test_layout.py
│
└── utils/                    # Test utility functions
    ├── test_memory.h
    └── testbed.h
```

---

## python/ -- Python Bindings and DSL

```
python/
├── cutlass/                   # Python package
│   ├── __init__.py
│   ├── bindings.py            # pybind11 C++ bindings
│   ├── gemm.py                # GEMM Python interface
│   ├── conv.py                # Convolution Python interface
│   ├── layout.py              # Layout Python types
│   ├── datatype.py            # Data type definitions
│   ├── epilogue.py            # Epilogue Python interface
│   └── emit/
│       ├── pycuda.py          # PyCUDA integration
│       └── pytorch.py         # PyTorch integration
├── examples/
│   ├── 00_basic_gemm.py       # Basic GEMM in Python
│   ├── 01_epilogue_fusion.py  # Epilogue fusion in Python
│   ├── 02_stream_k.py         # Stream-K in Python
│   └── 03_batched_gemm.py     # Batched GEMM in Python
├── setup.py                   # Package setup
├── CMakeLists.txt             # CMake build for bindings
└── pyproject.toml             # Python project configuration
```

---

## docs/ and media/ -- Documentation

```
docs/
├── doxygen/                   # Doxygen configuration
│   └── Doxyfile
├── html/                      # Generated HTML documentation (after build)
└── latex/                     # Generated LaTeX documentation (after build)

media/
├── docs/                      # Documentation source files
│   ├── cutlass.svg            # CUTLASS architecture diagram
│   ├── hierarchy.svg          # Hierarchy visualization
│   ├── gemm_pipeline.svg      # GEMM pipeline diagram
│   └── *.png                  # Various documentation images
└── tools/                     # Media generation tools
```

---

## CMake Build System

```
cmake/
├── cutlass.cmake              # CUTLASS-specific CMake utilities
├── cuda.cmake                 # CUDA compilation utilities
├── gencode.cmake              # GPU architecture code generation
├── noeval.cmake               # Build configuration
└── test.cmake                 # Test configuration
```

---

## Summary

The CUTLASS codebase is organized around a clear hierarchy:

1. **`include/cutlass/`** contains the header-only template library, organized by functional scope (arch, gemm, conv, epilogue, layout, transform, reduction, pipeline, thread)
2. **`include/cute/`** contains the CuTe tensor algebra library with its own sub-structure (algorithm, arch, atom, container, numeric, util)
3. **`tools/`** provides the profiler, library generator, and utility headers
4. **`examples/`** contains 95+ example programs organized by architecture and operation type
5. **`test/`** provides comprehensive unit tests at every hierarchy level
6. **`python/`** provides Python bindings and a high-level DSL

Understanding this structure helps locate the right components for building custom kernels, debugging issues, and extending CUTLASS functionality.
