# CUTLASS - Chapter 30: Examples Catalog

This reference provides a comprehensive catalog of all CUTLASS examples, organized by category. Each example is described with its purpose, target architecture, key concepts demonstrated, and location within the CUTLASS repository.

---

## 30.1 How to Use This Catalog

All examples are located in the `examples/` directory of the CUTLASS repository. Each example is a self-contained CMake project that can be built independently.

```bash
# Build all examples
cd cutlass/build
make examples -j$(nproc)

# Build a specific example
make cutlass_example_00_basic_gemm -j$(nproc)

# Run an example
./examples/00_basic_gemm/00_basic_gemm
```

### Conventions

| Convention | Meaning |
|------------|---------|
| **Arch Target** | The minimum GPU architecture required (SM70, SM75, SM80, SM90, SM100) |
| **Key Concepts** | The primary CUTLASS features demonstrated |
| **API Level** | CUTLASS 2.x (legacy) or CUTLASS 3.x (recommended) |

---

## 30.2 Basic GEMM Examples

### Example 00: Basic GEMM

- **File**: `examples/00_basic_gemm/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Device-level GEMM API, `cutlass::gemm::device::Gemm`, problem sizing, host-device memory management
- **Description**: The most fundamental CUTLASS example. Demonstrates how to define a GEMM operation using the CUTLASS 2.x device API with FP16 inputs, FP32 accumulation, and FP32 output. Shows the complete flow from tensor allocation through kernel launch to result verification.
- **Data Types**: FP16 input, FP32 accumulator/output
- **Layout**: RowMajor A, ColumnMajor B, RowMajor C

### Example 01: Basic GEMM with Tuple

- **File**: `examples/01_basic_gemm/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Using tuple-based API for GEMM configuration, explicit type aliases
- **Description**: Similar to Example 00 but uses C++ type aliases more explicitly to define the GEMM operation. Shows how to organize the type definitions for clarity.

### Example 05: Batched GEMM

- **File**: `examples/05_batched_gemm/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Batched matrix multiplication, stride handling, batched GEMM problem size
- **Description**: Demonstrates batched GEMM where multiple independent matrix multiplications are performed in a single kernel launch. Shows how to set up 3D tensors with batch strides and configure the batched GEMM operation.
- **Data Types**: FP16 input, FP32 accumulator
- **Key Parameters**: `batch_count`, batch stride for A, B, C, D

### Example 06: Split-K GEMM

- **File**: `examples/06_splitK_gemm/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Split-K parallel reduction, workspace allocation, reduction kernel
- **Description**: Shows how to use Split-K decomposition to parallelize the K dimension of GEMM across multiple thread blocks. Demonstrates workspace allocation for partial results and the automatic reduction kernel that combines them.
- **Data Types**: FP16 input, FP32 accumulator
- **Key Parameters**: `split_k_slices`, workspace allocation

### Example 07: GEMM with Tensor Cores (Volta)

- **File**: `examples/07_volta_tensorop_gemm/`
- **Arch Target**: SM70 (Volta)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: WMMA API, Volta Tensor Core instructions, 4x4x4 WMMA tiles
- **Description**: Demonstrates GEMM using Volta's WMMA (Warp Matrix Multiply-Accumulate) instructions. Shows the Volta-specific instruction shape (16x16x4 for FP16) and tile configuration.
- **Data Types**: FP16
- **Instruction Shape**: 16x16x4 (WMMA)

### Example 08: GEMM with Tensor Cores (Turing)

- **File**: `examples/08_turing_tensorop_gemm/`
- **Arch Target**: SM75 (Turing)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: MMA instructions, Turing Tensor Cores, multi-stage pipeline
- **Description**: Demonstrates GEMM using Turing's MMA (Matrix Multiply-Accumulate) instructions. Shows the smaller instruction shapes (8x8x16, 16x8x8) and multi-stage pipeline design.
- **Data Types**: FP16, INT8, INT4
- **Instruction Shape**: 8x8x16, 16x8x8

---

## 30.3 Mixed Precision Examples

### Example 10: Planar Complex GEMM

- **File**: `examples/10_planar_complex/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Complex number GEMM, planar complex layout (real and imaginary stored separately)
- **Description**: Demonstrates complex-valued GEMM where real and imaginary parts of the input matrices are stored in separate (planar) tensors. Shows the planar complex layout adapter and how to configure GEMM for complex arithmetic.
- **Data Types**: FP32 complex (planar layout)

### Example 11: Planar Complex Array GEMM

- **File**: `examples/11_planar_complex_array/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Batched complex GEMM, array of problem sizes
- **Description**: Extends the planar complex example to support an array of batched GEMM problems with potentially different sizes. Shows how to configure the batched complex GEMM with problem-specific dimensions.

### Example 26: Mixed Precision GEMM (SM80)

- **File**: `examples/26_ampere_tensorop_gemm/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Mixed precision GEMM, TF32 Tensor Cores, BF16 support, async copy
- **Description**: Demonstrates mixed-precision GEMM on Ampere with various data type combinations: TF32, BF16, FP16, and INT8. Shows how to configure the epilogue for type conversion between accumulator and output types.
- **Data Types**: TF32, BF16, FP16, INT8, FP32 accumulator

---

## 30.4 Fusion Examples

### Example 13: Two Tensor Op Fusion

- **File**: `examples/13_two_tensor_op_fusion/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Epilogue fusion, back-to-back GEMM, fused activation
- **Description**: Demonstrates fusing two tensor operations (two GEMMs) into a single kernel by using the output of the first GEMM as the input to the second without writing to global memory. Shows the fused epilogue pattern.
- **Data Types**: FP16, TF32

### Example 44: Multi-GEMM IR and Code Generation

- **File**: `examples/44_multi_gemm_ir_and_codegen/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Multi-GEMM fusion, intermediate representation, automatic code generation
- **Description**: Shows how to define a sequence of GEMM operations as an intermediate representation and automatically generate fused kernel code. Demonstrates the code generation pipeline for multi-GEMM patterns.
- **Data Types**: FP16, BF16, FP32

### Example 46: fused two gemms

- **File**: `examples/46_fused_two_gemms/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Fused GEMM-GEMM, epilogue visitor tree
- **Description**: Demonstrates fusing two GEMM operations (e.g., for MLP layers) where the output of the first GEMM feeds directly into the second without an intermediate global memory write. Uses the epilogue visitor tree pattern.
- **Data Types**: FP16, FP32

---

## 30.5 Ampere Tensor Operation Examples

### Example 14: Ampere Tensor Op Flow

- **File**: `examples/14_ampere_tensorop_flow/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Ampere async copy (cp.async), multi-stage pipeline, Tensor Core dispatch
- **Description**: Demonstrates the complete Ampere data flow: async memory copy from global to shared memory using cp.async, multi-stage software pipeline, and Tensor Core MMA operations.

### Example 17: CUTLASS Tensor Op Conv

- **File**: `examples/17_cutlass_tensor_op_conv/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Convolution via implicit GEMM, tensor operation convolution
- **Description**: Shows how CUTLASS implements convolution as an implicit GEMM operation. Demonstrates the mapping from convolution parameters (padding, stride, dilation) to the GEMM problem size.

### Example 18: Ampere FP64 Tensor Op

- **File**: `examples/18_ampere_fp64_tensor_op/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: FP64 Tensor Core GEMM, double precision MMA
- **Description**: Demonstrates double-precision (FP64) GEMM using Ampere's FP64 Tensor Core instructions. Shows the FP64-specific instruction shapes and pipeline configuration.
- **Data Types**: FP64
- **Instruction Shape**: 8x8x4 (FP64)

### Example 19: Ampere TF32 Tensor Op

- **File**: `examples/19_ampere_tf32_tensor_op/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: TF32 data type, TF32 Tensor Core operations, implicit TF32 conversion
- **Description**: Demonstrates TF32 (TensorFloat-32) GEMM on Ampere. Shows how FP32 inputs are implicitly converted to TF32 for the Tensor Core operation while maintaining FP32 accumulation.
- **Data Types**: TF32 input, FP32 accumulator

### Example 22: Ampere Tensor Op Conv2D

- **File**: `examples/22_ampere_tensor_op_conv2d/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Implicit GEMM Conv2D, Ampere-specific conv optimizations
- **Description**: Full Conv2D implementation using Ampere Tensor Cores and the implicit GEMM approach. Demonstrates forward convolution (fprop), backward data gradient (dgrad), and backward weight gradient (wgrad).
- **Data Types**: FP16, TF32

### Example 24: Ampere Tensor Op GEMM (CUTLASS 3.x)

- **File**: `examples/24_ampere_tensorop_gemm/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: CUTLASS 3.x API, CollectiveBuilder, GemmUniversalAdapter
- **Description**: Demonstrates the CUTLASS 3.x GEMM API using CollectiveBuilder for Ampere. Shows the modern, simplified approach to defining GEMM kernels with automatic tile and schedule selection.
- **Data Types**: FP16, BF16, TF32, FP32

### Example 59: Ampere Gather/Scatter Conv

- **File**: `examples/59_ampere_gather_scatter_conv/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Gather/scatter memory access, irregular convolution patterns
- **Description**: Demonstrates convolution with gather/scatter memory access patterns. Useful for irregular access patterns where the input or output tensor has non-contiguous elements.
- **Data Types**: FP16, FP32

---

## 30.6 Sparse GEMM Examples

### Example 15: Sparse Tensor Op GEMM

- **File**: `examples/15_sparse_tensorop_gemm/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: 2:4 structured sparsity, sparse metadata, sparse GEMM kernel
- **Description**: Demonstrates GEMM with 2:4 structured sparsity on Ampere. Shows how the sparse metadata tensor encodes the 2:4 pattern and how the sparse Tensor Core instructions skip zero elements for improved throughput.

### Example 41: Sparse CUTLASS 3.x GEMM

- **File**: `examples/41_sparse_cutlass3_gemm/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: CUTLASS 3.x sparse GEMM, CollectiveBuilder for sparse operations
- **Description**: Sparse GEMM using the CUTLASS 3.x API with CollectiveBuilder. Shows the modern approach to defining sparse GEMM kernels with automatic configuration.

---

## 30.7 Python Examples

### Example 40: CUTLASS Python

- **File**: `examples/40_cutlass_py/`
- **Arch Target**: SM80+
- **API Level**: PyCUTLASS
- **Key Concepts**: Python GEMM API, PyCUTLASS bindings, Python-based kernel configuration
- **Description**: Demonstrates the PyCUTLASS Python interface for defining and running GEMM operations. Shows how to configure data types, layouts, tile sizes, and launch kernels entirely from Python.
- **Language**: Python

---

## 30.8 Hopper (SM90) Examples

### Example 55: Hopper Mixed Dtype GEMM

- **File**: `examples/55_hopper_mixed_dtype_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Mixed data type GEMM on Hopper, WGMMA instructions, FP8/FP16 mixed precision
- **Description**: Demonstrates mixed data type GEMM on Hopper using the WGMMA (Warp Group Matrix Multiply-Accumulate) instruction. Shows how different input types (FP8, FP16, BF16) can be combined with different accumulator and output types.
- **Data Types**: E4M3, E5M2, FP16, BF16, FP32
- **Schedule**: KernelTmaWarpSpecialized

### Example 56: Hopper WGMMA GEMM

- **File**: `examples/56_hopper_wgmma_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: WGMMA instruction, warp group MMA, async matrix multiply
- **Description**: Low-level example showing WGMMA instruction usage in CUTLASS. Demonstrates the register layout requirements and synchronization for warp-group-level matrix operations.

### Example 57: Hopper Tensor Memory Adapter

- **File**: `examples/57_hopper_tma_tensor/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: TMA (Tensor Memory Accelerator) load/store, TMA descriptors
- **Description**: Demonstrates the Tensor Memory Accelerator (TMA) on Hopper for efficient global-to-shared memory transfers. Shows how to create TMA descriptors and perform bulk tensor operations.
- **Features**: TMA load, TMA store, TMA descriptors

### Example 58: Hopper WARP Specialized GEMM

- **File**: `examples/58_hopper_warp_specialized_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Warp specialization, producer-consumer pattern, DMA warp, MMA warp group
- **Description**: Demonstrates warp-specialized GEMM on Hopper where separate warp groups handle data movement (producer/DMA) and computation (consumer/MMA). Shows the named barrier synchronization between producer and consumer.
- **Schedule**: KernelTmaWarpSpecialized

### Example 60: Hopper Collective Builder GEMM

- **File**: `examples/60_hopper_collective_builder/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: CollectiveBuilder, automatic schedule selection, GemmUniversalAdapter
- **Description**: Comprehensive example showing the CollectiveBuilder API for Hopper. Demonstrates how CollectiveBuilder automatically selects the optimal kernel schedule, tile size, and stage count based on the problem configuration.
- **Features**: CollectiveBuilder, auto schedule, auto stage count

### Example 61: Hopper Grouped GEMM

- **File**: `examples/61_hopper_grouped_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Grouped GEMM, variable problem sizes, load-balanced scheduling
- **Description**: Demonstrates grouped GEMM where multiple GEMM problems with potentially different sizes are processed in a single kernel launch. Shows load-balanced scheduling across thread blocks.
- **Data Types**: FP16, BF16, FP32

### Example 63: Hopper GEMM with Weight Prefetch

- **File**: `examples/63_hopper_gemm_with_weight_prefetch/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Weight prefetching, persistent kernels, multi-cast TMA, cluster-level data sharing
- **Description**: Demonstrates weight prefetching in Hopper GEMM where weights are pre-loaded into shared memory before they are needed. Shows how to use persistent kernel patterns and multi-cast TMA for efficient weight sharing across thread blocks in a cluster.
- **Features**: Weight prefetch, TMA multi-cast, persistent kernels

### Example 65: Distributed GEMM

- **File**: `examples/65_distributed_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Tensor Parallel GEMM, NVLink communication, multi-GPU GEMM
- **Description**: Demonstrates distributed GEMM across multiple GPUs using Tensor Parallelism. Shows how to split the GEMM workload across devices and overlap computation with NVLink communication.
- **Features**: Tensor Parallelism, NVLink, all-reduce, reduce-scatter

### Example 66: Hopper Epilogue Visitor Tree

- **File**: `examples/66_hopper_epilogue_visitor_tree/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Epilogue visitor tree, fused epilogue operations, chained element-wise ops
- **Description**: Demonstrates the epilogue visitor tree pattern on Hopper for composing multiple post-GEMM operations (bias, activation, scaling, etc.) into a single fused epilogue pass.
- **Features**: Visitor tree, fused activation, bias fusion

### Example 69: Hopper Mixed Dtype Grouped GEMM

- **File**: `examples/69_hopper_mixed_dtype_grouped_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Grouped GEMM with mixed data types, variable batch sizes
- **Description**: Combines grouped GEMM (multiple problems in one launch) with mixed data types (e.g., FP8 inputs, FP16 output). Shows how to handle type conversion in the grouped GEMM context.
- **Data Types**: E4M3, E5M2, FP16, BF16, FP32

### Example 70: Hopper Stream-K GEMM

- **File**: `examples/70_hopper_stream_k_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Stream-K decomposition, workload splitting, fixed-size tiles
- **Description**: Demonstrates the Stream-K algorithm for GEMM on Hopper, which splits the workload into fixed-size tiles to ensure all thread blocks perform equal amounts of work, improving load balancing for non-square or small problems.
- **Features**: Stream-K, load balancing, tile splitting

### Example 88: Hopper FMHA

- **File**: `examples/88_hopper_fmha/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Fused Multi-Head Attention, flash attention, online softmax, TMA-based attention
- **Description**: Implements Fused Multi-Head Attention (FMHA) on Hopper using TMA for data movement and WGMMA for the core attention computation. Demonstrates the flash attention algorithm with online softmax.
- **Features**: FMHA, flash attention, TMA, WGMMA, online softmax, causal masking

### Example 111: Hopper SSD

- **File**: `examples/111_hopper_ssd/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: State Space Decomposition, selective state space models, Mamba-style kernels
- **Description**: Implements the State Space Decomposition (SSD) operation on Hopper, used in selective state space models (Mamba). Demonstrates how to map the recurrent scan operation to efficient GPU kernels using TMA and WGMMA.
- **Features**: SSD, state space models, recurrent scan, TMA, WGMMA

---

## 30.9 Blackwell (SM100+) Examples

### Example 70: Blackwell GEMM

- **File**: `examples/70_blackwell_gemm/`
- **Arch Target**: SM100 (Blackwell)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: UMMA (Unified MMA), Blackwell Tensor Core instructions
- **Description**: Basic GEMM example for Blackwell demonstrating the UMMA (Unified Matrix Multiply-Accumulate) instruction. Shows the Blackwell-specific data flow and pipeline configuration.
- **Data Types**: FP16, BF16, TF32, FP32

### Example 72: Blackwell Narrow Precision GEMM

- **File**: `examples/72_blackwell_narrow_precision_gemm/`
- **Arch Target**: SM100 (Blackwell)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: NVFP4, MXFP4/6/8, block-scaled GEMM, scale factor tensors
- **Description**: Demonstrates narrow precision GEMM on Blackwell using block-scaled data types: NVFP4 (NVIDIA FP4), MXFP4, MXFP6, and MXFP8. Shows how scale factor tensors (SFA, SFB) are used alongside the narrow-precision data.
- **Data Types**: NVFP4, MXFP4, MXFP6, MXFP8, FP16, FP32

### Example 77: Blackwell FMHA

- **File**: `examples/77_blackwell_fmha/`
- **Arch Target**: SM100 (Blackwell)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: FMHA on Blackwell, MLA (Multi-head Latent Attention), weight absorption
- **Description**: Implements Fused Multi-Head Attention on Blackwell with support for MLA (Multi-head Latent Attention), which uses low-rank key-value compression. Demonstrates weight absorption for efficient MLA computation.
- **Features**: FMHA, MLA, weight absorption, UMMA

### Example 81: Blackwell GEMM Blockwise

- **File**: `examples/81_blackwell_gemm_blockwise/`
- **Arch Target**: SM100 (Blackwell)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Blockwise GEMM, groupwise quantization, scale factors per block
- **Description**: Demonstrates blockwise GEMM on Blackwell where scale factors are applied per block (group) of elements rather than globally. Essential for quantized inference with block-scaled types.
- **Features**: Blockwise scaling, groupwise quantization, SFA/SFB

### Example 82: Blackwell Distributed GEMM

- **File**: `examples/82_blackwell_distributed_gemm/`
- **Arch Target**: SM100 (Blackwell)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Distributed GEMM on Blackwell, multi-GPU Tensor Parallelism, NVLink communication
- **Description**: Demonstrates distributed GEMM across multiple Blackwell GPUs using Tensor Parallelism with NVLink. Shows Blackwell-specific optimizations for overlapping communication and computation.
- **Features**: Tensor Parallelism, NVLink, all-reduce

### Example 95: Blackwell GEMM Green Context

- **File**: `examples/95_blackwell_gemm_green_context/`
- **Arch Target**: SM100 (Blackwell)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Green Contexts, SM resource partitioning, dynamic persistent CLC scheduler
- **Description**: Demonstrates GEMM using Blackwell's Green Contexts feature, which allows partitioning SM resources for concurrent kernel execution. Shows the persistent CLC (Command List Controller) scheduler for managing green contexts.
- **Features**: Green Contexts, persistent scheduler, CLC, SM partitioning

---

## 30.10 Additional Examples

### Example 09: Interleaved Layout

- **File**: `examples/09_interleaved_layout/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Interleaved memory layout, bank conflict avoidance, column-interleaved layout
- **Description**: Demonstrates the interleaved layout where matrix elements are stored in an interleaved pattern to avoid shared memory bank conflicts during Tensor Core operations.

### Example 12: Sliced K GEMM (Epilogue)

- **File**: `examples/12_sliced_k_gemm/`
- **Arch Target**: SM70+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Sliced-K reduction, partial accumulator reduction, epilogue integration
- **Description**: Shows how to perform a sliced-K (similar to Split-K) GEMM where the K dimension is partitioned across warps within a thread block and the partial results are reduced in the epilogue.

### Example 20: SIMT GEMM

- **File**: `examples/20_simt_gemm/`
- **Arch Target**: SM50+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: SIMT (Single Instruction Multiple Thread) GEMM, scalar CUDA core operations, no Tensor Cores
- **Description**: Demonstrates GEMM using only CUDA scalar cores (no Tensor Cores). Useful for architectures without Tensor Cores or for data types not supported by Tensor Cores.
- **Data Types**: FP32, FP64

### Example 23: Ampere INT8 GEMM

- **File**: `examples/23_ampere_int8_tensorop/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: INT8 GEMM, integer Tensor Core operations
- **Description**: Demonstrates INT8 GEMM using Ampere Tensor Cores. Shows integer matrix multiply with various accumulator types (INT32).
- **Data Types**: INT8 input, INT32 accumulator

### Example 27: Ampere BF16 Tensor Op

- **File**: `examples/27_ampere_bf16_tensorop/`
- **Arch Target**: SM80 (Ampere)
- **API Level**: CUTLASS 2.x
- **Key Concepts**: BF16 data type, BF16 Tensor Core operations
- **Description**: Demonstrates BF16 (Brain Float 16) GEMM on Ampere. Shows the BF16-specific instruction shapes and pipeline configuration.

### Example 28: Ampere FP8 (SM89, Ada)

- **File**: `examples/28_ampere_fp8/`
- **Arch Target**: SM89 (Ada Lovelace)
- **API Level**: CUTLASS 2.x/3.x
- **Key Concepts**: FP8 E4M3, FP8 E5M2, Ada-specific FP8 support
- **Description**: Demonstrates FP8 GEMM on Ada (SM89). Shows the E4M3 and E5M2 floating-point formats and how they map to Tensor Core operations on Ada.

### Example 29: Ampere Tensor Op GEMM with Broadcast

- **File**: `examples/29_ampere_broadcast/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Broadcast in epilogue, bias addition fusion
- **Description**: Demonstrates broadcasting a bias vector in the GEMM epilogue. Shows how to fuse a bias addition operation with the GEMM output.

### Example 30: Ampere Tensor Op GEMM with Reduction

- **File**: `examples/30_ampere_reduction/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Epilogue reduction, row/column reduction fused with GEMM
- **Description**: Demonstrates performing a reduction (e.g., row-wise sum) in the GEMM epilogue without writing intermediate results to global memory.

### Example 31: Ampere GEMM with Epilogue

- **File**: `examples/31_ampere_gemm_epilogue/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Custom epilogue, activation functions, linear combination
- **Description**: Shows various epilogue configurations for Ampere GEMM including different activation functions (ReLU, GELU, sigmoid) and linear combination parameters.

### Example 35: GEMM with Stream-K

- **File**: `examples/35_gemm_stream_k/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Stream-K decomposition, load balancing, tile splitting
- **Description**: Demonstrates the Stream-K algorithm for improving load balancing in GEMM when the problem size does not evenly divide into thread block tiles.

### Example 36: GEMM with Gather/Scatter

- **File**: `examples/36_gemm_gather_scatter/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Gather/scatter iterator, indexed access, non-contiguous tensors
- **Description**: Demonstrates GEMM with gather/scatter memory access where input elements are accessed through an index tensor rather than contiguously.

### Example 37: GEMM with rank_2k/3k Update

- **File**: `examples/37_gemm_rank_k_update/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 2.x
- **Key Concepts**: Symmetric rank-K update, Hermitian matrix operations
- **Description**: Demonstrates symmetric rank-2K and rank-3K matrix update operations using GEMM infrastructure.

### Example 38: GEMM with 2:4 Sparsity

- **File**: `examples/38_gemm_2_4_sparsity/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: 2:4 structured sparsity, sparse metadata, CUTLASS 3.x sparse API
- **Description**: Demonstrates 2:4 structured sparsity using the CUTLASS 3.x API. Shows how to create the sparse metadata tensor and launch sparse GEMM operations.

### Example 39: GEMM with Epilogue Visitor

- **File**: `examples/39_gemm_epilogue_visitor/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Epilogue visitor pattern, compositional epilogue
- **Description**: Demonstrates the epilogue visitor pattern for composing complex post-GEMM operations as a tree of visitor nodes.

### Example 42: CUTLASS 3.x GEMM Basic

- **File**: `examples/42_cutlass3_gemm_basic/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: CUTLASS 3.x basic GEMM, GemmUniversal, CollectiveBuilder
- **Description**: A clean, minimal example of CUTLASS 3.x GEMM using the modern API. Recommended starting point for new CUTLASS 3.x users.

### Example 43: CUTLASS 3.x GEMM with Custom Mainloop

- **File**: `examples/43_cutlass3_gemm_custom_mainloop/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Custom mainloop, custom collective operation, extending CUTLASS
- **Description**: Shows how to define a custom collective operation (mainloop) for GEMM. Useful for implementing novel data movement patterns or computation strategies.

### Example 45: CUTLASS 3.x Epilogue Fusion

- **File**: `examples/45_cutlass3_epilogue_fusion/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Epilogue fusion, activation fusion, bias fusion, CUTLASS 3.x epilogue API
- **Description**: Demonstrates epilogue fusion patterns in CUTLASS 3.x including bias addition, ReLU activation, and composed fusion operations.

### Example 47: CUTLASS 3.x GEMM with CuTe

- **File**: `examples/47_cutlass3_gemm_cute/`
- **Arch Target**: SM80+
- **API Level**: CUTLASS 3.x / CuTe
- **Key Concepts**: CuTe library, CuTe layout, CuTe tensors, CuTe algorithms
- **Description**: Demonstrates GEMM using the CuTe library directly (without the higher-level GEMM API). Shows CuTe layout algebra, tensor partitioning, and the copy/gemm algorithms.

### Example 48: CuTe BLAS-Level Operations

- **File**: `examples/48_cute_blas/`
- **Arch Target**: SM80+
- **API Level**: CuTe
- **Key Concepts**: CuTe BLAS operations, tiled MMA, custom MMA atoms
- **Description**: Demonstrates BLAS-level operations (GEMM, SYRK, HERK) using CuTe primitives. Shows how to build custom operations from CuTe atoms.

### Example 49: CUTLASS 3.x SM90 GEMM

- **File**: `examples/49_cutlass3_sm90_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: SM90 GEMM, TMA, warp specialization, CollectiveBuilder for SM90
- **Description**: Comprehensive SM90 GEMM example showing TMA-based data movement and warp-specialized execution. Demonstrates the full Hopper GEMM pipeline.

### Example 50: CUTLASS 3.x SM90 Sparse GEMM

- **File**: `examples/50_cutlass3_sm90_sparse_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Sparse GEMM on Hopper, 2:4 sparsity with TMA
- **Description**: Demonstrates sparse GEMM on Hopper combining 2:4 structured sparsity with TMA-based data movement and warp specialization.

### Example 51: CUTLASS 3.x SM90 Mixed Precision

- **File**: `examples/51_cutlass3_sm90_mixed_precision/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Mixed precision on SM90, FP8/FP16/FP32 combinations, WGMMA dtype flexibility
- **Description**: Demonstrates various mixed precision configurations on Hopper including FP8 inputs with FP32 accumulation and FP16 output.

### Example 52: CUTLASS 3.x SM90 Epilogue

- **File**: `examples/52_cutlass3_sm90_epilogue/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: SM90 epilogue, TMA store, epilogue fusion on Hopper
- **Description**: Demonstrates the SM90 epilogue with TMA store for output writing. Shows fused activation and bias operations in the Hopper epilogue pipeline.

### Example 53: CUTLASS 3.x SM90 Stream-K

- **File**: `examples/53_cutlass3_sm90_stream_k/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Stream-K on SM90, load balancing with TMA and warp specialization
- **Description**: Stream-K decomposition on Hopper combining the load-balancing benefits of Stream-K with Hopper's TMA and warp specialization features.

### Example 54: CUTLASS 3.x SM90 Grouped GEMM

- **File**: `examples/54_cutlass3_sm90_grouped_gemm/`
- **Arch Target**: SM90 (Hopper)
- **API Level**: CUTLASS 3.x
- **Key Concepts**: Grouped GEMM on SM90, variable-size problems, TMA descriptors
- **Description**: Grouped GEMM on Hopper using TMA descriptors for efficient handling of variable-size problems in a single kernel launch.

---

## 30.11 Complete Example Index

The following table provides a quick reference index of all CUTLASS examples:

| ID | Name | Arch | API | Category | Key Data Types |
|----|------|------|-----|----------|----------------|
| 00 | Basic GEMM | SM70+ | 2.x | Basic | FP16, FP32 |
| 05 | Batched GEMM | SM70+ | 2.x | Basic | FP16, FP32 |
| 06 | Split-K GEMM | SM70+ | 2.x | Basic | FP16, FP32 |
| 07 | Volta TensorOp | SM70 | 2.x | TensorOp | FP16 |
| 08 | Turing TensorOp | SM75 | 2.x | TensorOp | FP16, INT8 |
| 09 | Interleaved Layout | SM70+ | 2.x | Layout | FP16 |
| 10 | Planar Complex | SM70+ | 2.x | Mixed Precision | FP32 complex |
| 11 | Planar Complex Array | SM70+ | 2.x | Mixed Precision | FP32 complex |
| 12 | Sliced K GEMM | SM70+ | 2.x | Reduction | FP16, FP32 |
| 13 | Two TensorOp Fusion | SM80+ | 2.x | Fusion | FP16, TF32 |
| 14 | Ampere TensorOp Flow | SM80 | 2.x | TensorOp | FP16, TF32 |
| 15 | Sparse TensorOp | SM80 | 2.x | Sparse | FP16, TF32 |
| 17 | TensorOp Conv | SM80+ | 2.x | Convolution | FP16, TF32 |
| 18 | Ampere FP64 | SM80 | 2.x | TensorOp | FP64 |
| 19 | Ampere TF32 | SM80 | 2.x | TensorOp | TF32 |
| 20 | SIMT GEMM | SM50+ | 2.x | Basic | FP32, FP64 |
| 22 | Ampere Conv2D | SM80 | 2.x | Convolution | FP16, TF32 |
| 23 | Ampere INT8 | SM80 | 2.x | TensorOp | INT8 |
| 24 | Ampere TensorOp 3.x | SM80+ | 3.x | Basic | FP16, BF16, TF32 |
| 26 | Ampere Mixed Prec | SM80 | 2.x | Mixed Precision | TF32, BF16 |
| 27 | Ampere BF16 | SM80 | 2.x | TensorOp | BF16 |
| 28 | Ada FP8 | SM89 | 2.x/3.x | Mixed Precision | FP8 |
| 29 | Ampere Broadcast | SM80+ | 2.x | Epilogue | FP16, FP32 |
| 30 | Ampere Reduction | SM80+ | 2.x | Epilogue | FP16, FP32 |
| 31 | Ampere Epilogue | SM80+ | 2.x | Epilogue | FP16, FP32 |
| 35 | Stream-K GEMM | SM80+ | 3.x | Advanced | FP16, BF16 |
| 36 | Gather/Scatter | SM80+ | 3.x | Advanced | FP16 |
| 37 | Rank-K Update | SM80+ | 2.x | Advanced | FP16, FP32 |
| 38 | 2:4 Sparsity 3.x | SM80+ | 3.x | Sparse | FP16, BF16 |
| 39 | Epilogue Visitor | SM80+ | 3.x | Epilogue | FP16, FP32 |
| 40 | Python GEMM | SM80+ | Python | Python | FP16, FP32 |
| 41 | Sparse 3.x | SM80+ | 3.x | Sparse | FP16, BF16 |
| 42 | 3.x Basic GEMM | SM80+ | 3.x | Basic | FP16, BF16 |
| 43 | Custom Mainloop | SM80+ | 3.x | Advanced | FP16 |
| 44 | Multi-GEMM IR | SM80+ | 3.x | Fusion | FP16, BF16 |
| 45 | 3.x Epilogue Fusion | SM80+ | 3.x | Epilogue | FP16, FP32 |
| 46 | Fused Two GEMMs | SM80+ | 3.x | Fusion | FP16, FP32 |
| 47 | GEMM with CuTe | SM80+ | CuTe | CuTe | FP16 |
| 48 | CuTe BLAS | SM80+ | CuTe | CuTe | FP16, FP32 |
| 49 | SM90 GEMM | SM90 | 3.x | Hopper | FP16, BF16, FP8 |
| 50 | SM90 Sparse | SM90 | 3.x | Hopper | FP16, FP8 |
| 51 | SM90 Mixed Prec | SM90 | 3.x | Hopper | FP8, FP16, FP32 |
| 52 | SM90 Epilogue | SM90 | 3.x | Hopper | FP16, FP32 |
| 53 | SM90 Stream-K | SM90 | 3.x | Hopper | FP16, BF16 |
| 54 | SM90 Grouped | SM90 | 3.x | Hopper | FP16, BF16 |
| 55 | Mixed Dtype SM90 | SM90 | 3.x | Hopper | FP8, FP16, BF16 |
| 56 | WGMMA GEMM | SM90 | 3.x | Hopper | FP16 |
| 57 | TMA Tensor | SM90 | 3.x | Hopper | FP16 |
| 58 | Warp Specialized | SM90 | 3.x | Hopper | FP16, BF16 |
| 59 | Gather/Scatter Conv | SM80+ | 3.x | Convolution | FP16 |
| 60 | Collective Builder | SM90 | 3.x | Hopper | FP16, BF16 |
| 61 | Grouped GEMM SM90 | SM90 | 3.x | Hopper | FP16, BF16 |
| 63 | Weight Prefetch | SM90 | 3.x | Hopper | FP16, BF16 |
| 65 | Distributed GEMM | SM90 | 3.x | Hopper | FP16, BF16 |
| 66 | Epilogue Visitor SM90 | SM90 | 3.x | Hopper | FP16, FP32 |
| 69 | Mixed Dtype Grouped | SM90 | 3.x | Hopper | FP8, FP16 |
| 70 | Blackwell GEMM | SM100 | 3.x | Blackwell | FP16, BF16 |
| 72 | Narrow Precision | SM100 | 3.x | Blackwell | NVFP4, MXFP8 |
| 77 | Blackwell FMHA | SM100 | 3.x | Blackwell | FP16, BF16 |
| 81 | Blockwise GEMM | SM100 | 3.x | Blackwell | NVFP4, MXFP |
| 82 | Distributed SM100 | SM100 | 3.x | Blackwell | FP16, BF16 |
| 88 | Hopper FMHA | SM90 | 3.x | Hopper | FP16, BF16 |
| 95 | Green Context | SM100 | 3.x | Blackwell | FP16, BF16 |
| 111 | Hopper SSD | SM90 | 3.x | Hopper | FP16, BF16 |

---

## 30.12 Building and Running Examples

### Building a Single Example

```bash
cd cutlass/build

# Build example 42 (CUTLASS 3.x basic GEMM)
make cutlass_example_42_cutlass3_gemm_basic -j$(nproc)

# Run
./examples/42_cutlass3_gemm_basic/42_cutlass3_gemm_basic
```

### Building All Examples

```bash
cd cutlass/build
make examples -j$(nproc)
```

### Building with Specific Architecture

```bash
cmake .. -DCUTLASS_NVCC_ARCHS="90"
make cutlass_example_49_cutlass3_sm90_gemm -j$(nproc)
```

### Example Output

Most examples produce output similar to:

```
M=1024, N=1024, K=1024
Running GEMM...
Status: success
Runtime: 0.234 ms
GFLOPS: 9216.4
```

---

## 30.13 Summary

The CUTLASS examples directory provides over 95 examples covering:

- **Basic GEMM**: From simple single-GEMM to batched and Split-K variants
- **Mixed Precision**: TF32, BF16, FP8, and multi-type combinations
- **Fusion**: Epilogue fusion, multi-GEMM fusion, activation fusion
- **Tensor Operations**: Architecture-specific Tensor Core usage from Volta to Blackwell
- **Convolution**: Implicit GEMM Conv2D/Conv3D with various optimizations
- **Sparse GEMM**: 2:4 structured sparsity across architectures
- **Python**: Complete Python API usage examples
- **Hopper (SM90)**: TMA, WGMMA, warp specialization, FMHA, SSD
- **Blackwell (SM100+)**: UMMA, narrow precision, green contexts, distributed GEMM

These examples serve as the primary learning resource for CUTLASS and provide ready-to-use code templates for common GEMM and convolution patterns.
