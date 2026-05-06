# XLA Reference - Chapter 15: GPU Backend

This reference provides comprehensive documentation on XLA's GPU backend, covering the compilation pipeline, code generation strategies, runtime infrastructure, and hardware support.

---

## 15.1 Overview

The XLA GPU backend compiles HLO programs into efficient GPU kernels and library calls that execute on NVIDIA GPUs (and, with appropriate configuration, AMD GPUs via ROCm). The backend is the most mature and feature-rich of XLA's backends, supporting a wide range of optimization techniques and code generation strategies.

### 15.1.1 Three Code Generation Approaches

The GPU backend uses three distinct approaches to generate executable code:

1. **External library calls**: Dispatching to highly optimized vendor libraries (cuBLAS, cuDNN, NCCL, cuFFT) for well-defined operations like matrix multiplication, convolution, and collective communication. These libraries are hand-tuned by hardware vendors and provide near-peak performance for their supported operations.

2. **Triton tiling**: Generating Triton IR (Intermediate Representation) for complex fusion patterns that involve matrix multiplications, softmax, layer normalization, and other structured operations. Triton provides a Python-like programming model that maps efficiently to GPU hardware while giving the compiler control over shared memory and synchronization.

3. **XLA emitters (progressive lowering to LLVM IR)**: Directly lowering HLO operations to LLVM IR, which is then compiled to PTX by the LLVM NVPTX backend. This approach handles all operations that cannot be dispatched to libraries or Triton, including elementwise operations, reductions, transposes, gather/scatter, and other general-purpose computations.

The choice of approach is made during the compilation pipeline based on pattern matching and cost analysis. The goal is to use the most efficient code generation strategy for each operation while maximizing fusion opportunities.

### 15.1.2 GPU Pipeline Architecture

The GPU compilation pipeline consists of the following major stages:

```
HLO Module (from frontend: JAX, TensorFlow, PyTorch)
    |
    v
[1. HLO Optimization Pipeline]
    - Algebraic simplification
    - Constant folding
    - Dead code elimination
    - Fusion
    - Layout assignment
    - Sharding propagation
    - Rematerialization
    |
    v
[2. GPU-Specific Rewriting]
    - CudnnFusedConvRewriter
    - CudnnNormRewriter
    - GemmRewriter (cuBLAS)
    - TritonFusionRewriter
    - CollectiveRewriter (NCCL)
    |
    v
[3. Buffer Assignment]
    - Instruction scheduling
    - Buffer allocation and reuse
    |
    v
[4. Code Generation]
    - Library call emission (cuBLAS, cuDNN, NCCL)
    - Triton IR generation
    - LLVM IR emission
    - PTX compilation (via LLVM NVPTX)
    |
    v
[5. Executable]
    - GpuExecutable containing:
      - PTX binary
      - Library call configurations
      - Buffer allocation plan
      - Thunk sequence (execution schedule)
```

---

## 15.2 Running Example

To illustrate the compilation pipeline, consider a simple JAX program that computes a matrix multiplication followed by a GELU activation:

```python
import jax
import jax.numpy as jnp

@jax.jit
def matmul_gelu(x, w, b):
    # Matrix multiplication: [M, K] x [K, N] -> [M, N]
    hidden = jnp.dot(x, w) + b
    # GELU activation
    return jax.nn.gelu(hidden)
```

### 15.2.1 Step 1: HLO Module Generation

JAX converts this to the following HLO module:

```
HloModule matmul_gelu

ENTRY %entry (x: f32[128,512], w: f32[512,256], b: f32[256]) -> f32[128,256] {
  %x = parameter(0), f32[128,512]
  %w = parameter(1), f32[512,256]
  %b = parameter(2), f32[256]

  // Matrix multiplication
  %dot = dot(%x, %w),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    shape=f32[128,256]

  // Bias addition (broadcast b from [256] to [128,256])
  %bcast = broadcast(%b), dimensions={1}, shape=f32[128,256]
  %add = add(%dot, %bcast), shape=f32[128,256]

  // GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
  %half = constant(0.5)
  %bcast_half = broadcast(%half), dimensions={}, shape=f32[128,256]
  %mul_half = multiply(%add, %bcast_half)

  %one = constant(1.0)
  %sqrt2 = constant(1.41421356...)
  %div = divide(%add, %bcast_sqrt2)
  %erf = erf(%div)
  %add_one = add(%bcast_one, %erf)
  %gelu = multiply(%mul_half, %add_one)

  ROOT %result = %gelu, shape=f32[128,256]
}
```

### 15.2.2 Step 2: Optimization

The optimization pipeline applies several transformations:

1. **AlgebraicSimplifier**: No simplifications apply to this module.

2. **Fusion**: The elementwise chain (bias add + GELU computation) is fused into a single kernel:
   ```
   %dot = dot(%x, %w)  // Remains separate (handled by cuBLAS)

   %fused = fusion(%dot, %b) {
     %p0 = parameter(0), f32[128,256]  // dot result
     %p1 = parameter(1), f32[256]      // bias

     %bcast = broadcast(%p1), dimensions={1}
     %add = add(%p0, %bcast)

     %half = constant(0.5)
     %mul_half = multiply(%add, %half)

     %sqrt2 = constant(1.41421356...)
     %div = divide(%add, %sqrt2)
     %erf = erf(%div)

     %one = constant(1.0)
     %add_one = add(%one, %erf)
     ROOT %gelu = multiply(%mul_half, %add_one)
   }
   ```

3. **GemmRewriter**: The dot operation is rewritten to a cuBLAS custom call:
   ```
   %dot = custom-call(%x, %w),
     custom_call_target="__cublas$gemm",
     backend_config={
       lhs_batch_dimensions: [],
       rhs_batch_dimensions: [],
       lhs_contracting_dimensions: [1],
       rhs_contracting_dimensions: [0],
       algorithm: auto
     }
   ```

4. **Layout Assignment**: Assigns row-major (dimension order {0,1}) layout to all tensors.

### 15.2.3 Step 3: Code Generation

The code generation phase produces:

1. **For the dot operation**: A cuBLAS GEMM call configuration.
2. **For the fused elementwise chain**: An LLVM IR kernel that computes the GELU activation.

### 15.2.4 Step 4: Executable

The final `GpuExecutable` contains:
- A cuBLAS call thunk for the matrix multiplication.
- A PTX kernel for the fused GELU computation.
- A buffer allocation plan that allocates memory for `%x`, `%w`, `%b`, intermediate dot result, and final output.

---

## 15.3 Key Components

### 15.3.1 SPMD Partitioner

The SPMD (Single Program, Multiple Data) partitioner implements the GSPMD algorithm for multi-device computation. It partitions a single HLO program across multiple devices, inserting communication operations where necessary.

**GSPMD Paper**: The GSPMD algorithm was described in the paper "GSPMD: General and Scalable Parallelization for ML Computation Graphs" (Google, 2021). The key insight is that sharding annotations can be propagated through the computation graph, and the same program can be compiled for all devices with only the data distribution differing.

**Sharding annotations**: Users annotate key tensors with sharding specifications using XLA's `sharding` instruction annotation:

```python
# JAX example: shard a matrix across 4 devices along dimension 1
from jax.sharding import PartitionSpec as P, NamedSharding
from jax import devices

mesh = jax.sharding.Mesh(devices(), ('devices',))
sharding = NamedSharding(mesh, P(None, 'devices'))

x_sharded = jax.device_put(x, sharding)
w_sharded = jax.device_put(w, P('devices', None))

result = jax.pmap(jnp.dot)(x_sharded, w_sharded)
```

**Propagation**: The `ShardingPropagation` pass propagates shardings from annotated instructions through the graph. For each operation type, it determines the output sharding from the input shardings:

| Operation | Sharding Rule |
|-----------|--------------|
| Elementwise (add, mul, ...) | Output has same sharding as inputs (inputs must agree) |
| Dot | Depends on contracting dimensions; may insert all-reduce |
| Reshape | Adjust sharding to new shape |
| Reduce | Sharded dimensions are reduced locally; all-reduce for final result |
| Broadcast | Extend sharding to new dimensions |
| Transpose | Permute sharding dimensions |

**Communication overlap**: The partitioner inserts communication operations (all-reduce, all-gather, reduce-scatter, collective-permute) and attempts to overlap them with computation where possible. For example, when a dot product requires an all-reduce on its output, the partitioner may split the dot into partial computations that allow the all-reduce to overlap with remaining computation.

**Implementation in the GPU pipeline:**

```
SPMD Partitioning Pipeline:
  1. ShardingPropagation -- propagate shardings
  2. SpmdPartitioner -- partition computations
  3. CollectiveQuantizer -- insert communication ops
  4. CommunicationRewriter -- optimize communication patterns
  5. ShardingPropagation (fixpoint) -- propagate again after partitioning
```

### 15.3.2 Layout Assignment

Layout assignment on the GPU determines the physical memory layout (dimension ordering) for every tensor. The GPU backend has specific layout preferences driven by library requirements and hardware characteristics.

**Logical shape vs. physical layout:**

```
Logical: f32[128, 256]  -- describes mathematical structure
Physical layout: {0, 1}  -- row-major: element [i,j] at offset i*256 + j
Physical layout: {1, 0}  -- column-major: element [i,j] at offset j*128 + i
```

**GPU layout preferences:**

1. **cuDNN convolutions**: Prefer NHWC layout (dimensions ordered as batch, height, width, channels). This is because cuDNN's optimized kernels are designed for NHWC access patterns.

2. **cuBLAS GEMM**: Expects column-major inputs (Fortran order). However, XLA can pass row-major inputs and adjust the GEMM parameters (transpose flags) accordingly. The layout assignment pass chooses the layout that minimizes the total number of copy operations.

3. **Triton kernels**: Triton can handle any layout, but performance is best when the innermost dimension is the one being iterated over in the innermost loop.

4. **Elementwise kernels**: Performance is largely layout-independent because the GPU memory coalescing hardware handles both row-major and column-major access patterns well for contiguous elementwise operations.

**Copy insertion for layout conflicts:**

When a tensor is consumed by operations with different layout preferences, the pass inserts a `copy` (transposition) instruction:

```
%conv = custom-call(%input, %weights), target="__cudnn$convForward"
  // conv output layout: {0,1,2,3} (NHWC)

%copy = copy(%conv)
  // copy output layout: {0,3,1,2} (NCHW) for use by subsequent operation

%dot = custom-call(%copy, %weights2), target="__cublas$gemm"
```

The cost model minimizes the total number of copy bytes by choosing layouts that satisfy the most constrained operations first (convolutions and GEMM) and then propagating layouts to surrounding elementwise operations.

### 15.3.3 Fusion

Fusion on the GPU is critical for performance because global memory bandwidth (1-2 TB/s on modern GPUs) is the primary bottleneck, not compute throughput (tens to hundreds of TFLOPS). Fusion reduces memory traffic by keeping intermediate results in registers.

**Fusion invariants on GPU:**

1. **Shared iteration space**: All operations in a fusion must share the same iteration space (or be broadcastable to it). This means a fusion can contain elementwise operations and reductions, but not arbitrary combinations of operations with different shapes.

2. **No library calls inside fusion**: Fused computations cannot contain operations that require library calls (dot, convolution). These are always emitted as separate thunks.

3. **Register pressure**: The fusion must not use more registers than the GPU's register file allows. Excessive register usage reduces occupancy (the number of concurrent warps) and can hurt performance.

4. **Shared memory**: Some fusion patterns (transpose emitter, reduction emitter) use shared memory for inter-thread communication. The fusion must not exceed the GPU's shared memory capacity.

**Kernel generation for fused computations:**

Each fusion is emitted as a single GPU kernel. The kernel generation process:

1. **Determine launch dimensions**: The number of threads and blocks is determined by the output shape and the fusion's access pattern.
2. **Generate element function**: For each thread, the element function computes one or more output elements by traversing the fused computation graph.
3. **Optimize memory access**: The code generator applies memory coalescing, vectorized loads, and shared memory optimizations.
4. **Insert synchronization**: If the fusion uses shared memory (transpose, reduction), insert `__syncthreads()` barriers.

**Memory savings from fusion:**

```
Without fusion (3 separate kernels):
  Kernel 1 (exp):     read input [128,256] + write exp_result [128,256] = 2 * 128K * 4B = 1 MB
  Kernel 2 (add):     read exp_result + read bias + write add_result = 3 * 128K * 4B = 1.5 MB
  Kernel 3 (tanh):    read add_result + write output = 2 * 128K * 4B = 1 MB
  Total: 3.5 MB

With fusion (1 kernel):
  Fused kernel:       read input + read bias + write output = 2 * 128K * 4B + 256 * 4B = ~1 MB
  Total: ~1 MB (3.5x reduction)
```

### 15.3.4 Buffer Assignment

Buffer assignment determines the memory allocation for every tensor in the program. It aims to minimize total memory usage by reusing buffers that are no longer needed.

**Static allocation**: The GPU backend uses static buffer allocation -- all memory is allocated before execution begins. This avoids the overhead of dynamic memory allocation during program execution.

**Algorithm:**

1. **Instruction scheduling**: Order instructions to minimize peak memory usage. The scheduler considers data dependencies and chooses an order that allows buffers to be freed as early as possible.

2. **Buffer liveness analysis**: For each buffer, compute the interval [first_use, last_use] during which it must be alive.

3. **Buffer coloring (allocation)**: Assign buffers to memory offsets using a greedy interval coloring algorithm. Buffers with non-overlapping liveness intervals can share the same memory offset.

```
Time:         t0    t1    t2    t3    t4    t5    t6
Buffer A:     |=========================|
Buffer B:                 |=========================|
Buffer C:                           |=========================|

Allocation:
  Offset 0: Buffer A (t0-t3), Buffer C (t4-t6) -- A and C don't overlap
  Offset 1: Buffer B (t2-t5)
  Total: 2 * max_buffer_size bytes
```

4. **In-place operations**: Where possible, operations write their output directly into the same buffer as their input. This is safe when the input buffer is not needed after the operation. The `BufferAssignment` pass identifies such opportunities using `HloAliasAnalysis`.

5. **Temporary buffers**: Some operations require temporary scratch space (e.g., sorting algorithms, FFTs). The buffer allocator includes these temporary buffers in the allocation plan.

**Buffer assignment output:**

The result of buffer assignment is a `BufferAssignment` object that maps each HLO instruction to its buffer allocation:

```cpp
struct BufferAllocation {
  int64_t index;           // Allocation index
  int64_t offset;          // Offset in the allocation pool
  int64_t size;            // Size in bytes
  bool is_thread_local;    // Whether this is a thread-local buffer
  bool is_reusable;        // Whether this buffer can be reused
  HloInstruction* instruction;  // The instruction that owns this buffer
};
```

---

## 15.4 Code Generation Strategies

### 15.4.1 Library Selection

The GPU backend dispatches specific operations to optimized vendor libraries:

**cuBLAS for matrix multiplication:**

All dot-product operations (including batched matmul, matrix-vector multiply) are dispatched to cuBLAS. The `GemmRewriter` pass converts dot operations into cuBLAS custom calls with appropriate parameters:

```
custom_call_target = "__cublas$gemm"
backend_config = {
  dot_dimension_numbers: {
    lhs_contracting_dimensions: [1]
    rhs_contracting_dimensions: [0]
    lhs_batch_dimensions: []
    rhs_batch_dimensions: []
  }
  algorithm: GEMM_DEFAULT_TENSOR_OP  // Use tensor core if available
  alpha: 1.0
  beta: 0.0
}
```

cuBLAS supports multiple GEMM algorithms. XLA selects the best algorithm using:
- **Auto-tuning**: Run multiple algorithms on small inputs and select the fastest.
- **Heuristics**: Based on matrix sizes and compute capability.
- **User override**: Via `--xla_gpucublas_gemm_algorithm` flag.

**cuDNN for convolutions:**

Convolution operations are dispatched to cuDNN, which provides highly optimized implementations for various convolution types:

- **Forward convolution**: Standard convolution for training and inference.
- **Backward data convolution**: Gradient computation with respect to input.
- **Backward filter convolution**: Gradient computation with respect to filter.
- **Fused convolution**: Convolution + bias + activation in a single call.

The `CudnnConvRewriter` (and `CudnnFusedConvRewriter`) passes convert convolution operations into cuDNN custom calls:

```
custom_call_target = "__cudnn$convForward"
backend_config = {
  stream: 0
  window: {size: [3,3], stride: [1,1], pad: [1,1,1,1], dil: [1,1]}
  input_shape: f32[128,28,28,3]
  filter_shape: f32[3,3,3,64]
  output_shape: f32[128,28,28,64]
  algorithm: -1  // Auto-select
}
```

cuDNN algorithm selection uses:
- **cudnnFindConvolutionForwardAlgorithm**: Benchmark available algorithms.
- **Workspace size constraints**: Some algorithms require significant workspace memory.
- **Tensor core availability**: Use tensor core algorithms when available.

**NCCL for collective operations:**

Collective operations (all-reduce, all-gather, reduce-scatter, broadcast) are dispatched to NCCL, which provides optimized multi-GPU and multi-node communication:

```
custom_call_target = "__nccl$all_reduce"
backend_config = {
  reduction_kind: SUM
  operand_count: 1
}
```

NCCL operations are inserted by the SPMD partitioner and handle:
- **Topology-aware communication**: NCCL detects the GPU topology (NVLink, PCIe, NVSwitch) and selects the optimal communication pattern.
- **Overlap with computation**: NCCL operations can run on a separate CUDA stream, overlapping with computation on the default stream.

**cuFFT for FFTs:**

FFT operations are dispatched to cuFFT when available.

### 15.4.2 Direct LLVM IR Emission

For operations that cannot be dispatched to libraries, XLA generates LLVM IR directly. This covers:

- **Elementwise operations**: Simple element-wise kernels (add, mul, exp, log, etc.) that are generated by iterating over all elements and applying the operation.
- **Reductions**: Parallel reductions that use shared memory for inter-thread communication.
- **Transposes**: Shared-memory-based transpose kernels.
- **Gather/Scatter**: Index-based access patterns.
- **Sort**: Bitonic sort or other GPU-friendly sorting algorithms.
- **Select/Clamp**: Conditional operations.
- **Rng**: Random number generation.

**LLVM IR generation pipeline:**

```
HLO Fusion Computation
    |
    v
[Elemental IR Emitter]
    - Generate LLVM IR for each element
    - Apply indexing transformations
    |
    v
[LLVM IR Module]
    |
    v
[LLVM Optimization Pipeline]
    - Mem2reg
    - Loop invariant code motion
    - Dead instruction elimination
    |
    v
[LLVM NVPTX Backend]
    - Compile LLVM IR to PTX
    |
    v
[PTX Binary]
    |
    v
[CUBIN] (via CUDA driver)
```

**Elementwise kernel example:**

For a fusion `exp(x) + bias`, the generated LLVM IR looks like:

```llvm
define void @fusion(ptr %input, ptr %bias, ptr %output, i64 %n) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %gid = add i32 %tid, mul(%ctaid, %ntid)

  %cond = icmp ult i32 %gid, %n
  br i1 %cond, %body, %exit

body:
  %addr_in = getelementptr float, ptr %input, i32 %gid
  %val = load float, ptr %addr_in
  %exp_val = call float @llvm.exp.f32(float %val)

  %addr_bias = getelementptr float, ptr %bias, i32 %gid
  %b = load float, ptr %addr_bias
  %sum = fadd float %exp_val, %b

  %addr_out = getelementptr float, ptr %output, i32 %gid
  store float %sum, ptr %addr_out
  br label %exit

exit:
  ret void
}
```

### 15.4.3 Triton IR Emission

Triton IR emission is used for complex fusion patterns that involve matrix multiplications and structured operations. Triton provides better performance than LLVM IR emission for these patterns because it:

1. **Manages shared memory explicitly**: Triton gives the compiler control over what data is loaded into shared memory and when.
2. **Supports tiling natively**: Triton's programming model is based on tiles (blocks of data), which maps well to GPU shared memory and register blocking.
3. **Handles matmul efficiently**: Triton can emit efficient tiled matmul kernels that leverage tensor cores.

**Patterns handled by Triton:**

| Pattern | Triton Kernel Name |
|---------|-------------------|
| Matmul + bias + activation | `triton_matmul` |
| Softmax | `triton_softmax` |
| Flash attention (QK^T * V) | `triton_flash_attention` |
| Layer normalization | `triton_layer_norm` |
| Matmul + matmul (fused) | `triton_fused_matmul` |

**Triton compilation pipeline:**

```
HLO Fusion Computation
    |
    v
[Triton IR Generator]
    - Identify tile dimensions
    - Generate Triton IR (tt.load, tt.dot, tt.store)
    |
    v
[Triton IR Optimization]
    - Tiling optimization
    - Memory access coalescing
    |
    v
[TTIR -> TTGIR] (Triton GPU IR)
    - Convert to GPU-specific IR
    - Shared memory allocation
    |
    v
[TTGIR -> LLVM IR]
    - Lower to LLVM IR with GPU intrinsics
    |
    v
[PTX Binary]
```

---

## 15.5 Runtime

### 15.5.1 RuntimeIR (MLIR Dialect)

XLA's GPU runtime uses an MLIR-based intermediate representation called `RuntimeIR` (or `xla_gpu` dialect) to describe the execution plan. This IR captures:

- **Thunk sequence**: The ordered list of operations to execute (kernel launches, library calls, memory copies).
- **Buffer references**: Which buffers are used by each operation.
- **Synchronization**: Dependencies between operations that require synchronization.

The `xla_gpu` MLIR dialect includes operations like:

```mlir
// Launch a GPU kernel
xla_gpu.launch_kernel @fusion_kernel
  args(%buffer0, %buffer1, %buffer2)
  grid<128, 1, 1>
  block<256, 1, 1>

// Call cuBLAS GEMM
xla_gpu.custom_call @__cublas$gemm
  args(%input, %weights, %output)
  backend_config = {...}

// Copy between host and device
xla_gpu.copy %host_buffer to %device_buffer
```

### 15.5.2 CUDA Graph Extraction

XLA can extract CUDA graphs from compiled programs for reduced launch overhead. A CUDA graph captures the entire execution plan (kernel launches, memory operations, synchronization) as a single graph that can be replayed with minimal CPU overhead.

**How it works:**

1. **Graph capture**: During the first execution, XLA records all CUDA operations into a graph.
2. **Graph instantiation**: The captured graph is instantiated as an executable CUDA graph.
3. **Graph replay**: On subsequent executions, the graph is replayed directly, bypassing the CPU-side launch overhead.

CUDA graphs are particularly beneficial for:
- **Small kernels**: Where launch overhead is a significant fraction of execution time.
- **Repeated execution**: Where the same sequence of operations is executed many times (e.g., inference).
- **Multi-GPU programs**: Where the reduced CPU overhead improves scaling.

**Enablement:**

```python
# JAX: Enable CUDA graphs
with jax.jit(cuda_graph=True):
    result = f(x)
```

### 15.5.3 CPU Executable Compilation

The `GpuExecutable` contains not only GPU code but also CPU-side "thunks" that orchestrate execution. These thunks are compiled into a CPU executable that:

1. Manages buffer allocation and deallocation.
2. Launches GPU kernels in the correct order.
3. Calls into library APIs (cuBLAS, cuDNN, NCCL).
4. Handles host-device synchronization.
5. Manages CUDA streams and events.

### 15.5.4 Ahead-Of-Time Compilation

XLA supports ahead-of-time (AOT) compilation for GPU programs, producing a compiled binary that can be loaded and executed without the full XLA compiler:

```
// Compile:
xla_aot_compile --input_module=program.hlo --target=gpu --output=output.o

// The output contains:
// - PTX binary for each kernel
// - Library call configurations
// - Buffer allocation plan
// - Metadata for the runtime
```

AOT compilation is used for:
- **Deployment**: Distributing compiled models without the XLA compiler dependency.
- **Startup time**: Eliminating JIT compilation overhead at application startup.
- **Determinism**: Ensuring the exact same code is executed every time.

---

## 15.6 CUDA Support

### 15.6.1 NVIDIA GPU Support via LLVM NVPTX Backend

XLA generates GPU code by compiling LLVM IR to PTX using the LLVM NVPTX backend. The NVPTX backend is part of the LLVM project and supports all NVIDIA GPU architectures.

**Compilation flow:**

```
LLVM IR (target: nvptx64-nvidia-cuda)
    |
    v
[LLVM NVPTX Backend]
    - Instruction selection
    - Register allocation
    - Instruction scheduling
    |
    v
[PTX Assembly]
    |
    v
[CUDA Driver (cuModuleLoadData)]
    |
    v
[CUBIN (native GPU code)]
```

### 15.6.2 Compute Capability Detection

XLA queries the GPU's compute capability at runtime and adjusts code generation accordingly:

| Compute Capability | GPU Family | Key Features |
|-------------------|------------|--------------|
| 7.0 | V100 | Tensor cores (FP16), 16 GB HBM2 |
| 7.5 | T4, RTX 2080 | Turing tensor cores (INT8, INT4) |
| 8.0 | A100 | TF32 tensor cores, 40/80 GB HBM2e, async copy |
| 8.6 | RTX 3090 | Ampere consumer, 24 GB GDDR6X |
| 8.9 | RTX 4090 | Ada Lovelace, FP8 tensor cores |
| 9.0 | H100 | FP8, FP16, BF16 tensor cores, TMA, 80 GB HBM3 |
| 10.0 | B200 | Blackwell, next-gen tensor cores |

**Feature flags based on compute capability:**

- **Tensor cores**: Available from compute capability 7.0 (Volta). XLA uses tensor cores for matrix multiplication when the data types are compatible (FP16, BF16, TF32, FP8, INT8).
- **Shared memory capacity**: Varies by architecture (96 KB on A100, up to 228 KB on H100). The fusion emitter adjusts tile sizes based on available shared memory.
- **Warp-level operations**: Available from compute capability 7.0. Used for efficient reductions and matrix operations.
- **Async copy (cp.async)**: Available from compute capability 8.0. Allows overlapping memory loads with computation.
- **Tensor Memory Accelerator (TMA)**: Available from compute capability 9.0. Provides hardware-accelerated tiled memory access.

### 15.6.3 ROCm Support

XLA can also target AMD GPUs via ROCm. The ROCm backend uses the LLVM AMDGPU backend instead of NVPTX:

**Differences from NVIDIA:**

| Aspect | NVIDIA | AMD (ROCm) |
|--------|--------|-------------|
| IR Backend | LLVM NVPTX | LLVM AMDGPU |
| PTX equivalent | PTX | AMDGPU ISA |
| cuBLAS equivalent | rocBLAS | |
| cuDNN equivalent | MIOpen | |
| NCCL equivalent | RCCL | |
| Compute capability | CUDA compute capability | gfx architecture |

The ROCm backend shares most of the XLA codebase with the NVIDIA backend. The main differences are in:
- Library call targets (rocBLAS instead of cuBLAS, MIOpen instead of cuDNN).
- Code generation target (AMDGPU instead of NVPTX).
- Hardware capability queries.

ROCm support is enabled by building XLA with `--config=rocm` and is used by JAX on AMD GPUs.

---

## 15.7 Debugging the GPU Backend

### 15.7.1 Dump Compilation Artifacts

```bash
# Dump all compilation stages
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_program.py

# Dump specific stages
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_to=/tmp/xla_dump" \
  python my_program.py
```

The dump directory will contain:
- `module_*.hlo` -- HLO modules at various compilation stages
- `kernel_*.ptx` -- Generated PTX for each kernel
- `kernel_*.cubin` -- Compiled CUBIN for each kernel

### 15.7.2 PTX Inspection

```bash
# Dump PTX for all kernels
XLA_FLAGS="--xla_gpu_dump_ptx_to=/tmp/ptx" python my_program.py

# Inspect a specific PTX file
ptxas -v kernel.ptx  # Show register usage and other stats
```

### 15.7.3 Performance Profiling

```bash
# Enable XLA profiling
XLA_FLAGS="--xla_hlo_profile" python my_program.py

# Use NVIDIA Nsight Systems
nsys profile -t cuda python my_program.py

# Use NVIDIA Nsight Compute for kernel-level profiling
ncu --set full -k fusion_kernel python my_program.py
```

### 15.7.4 Auto-Tuning

XLA can auto-tune kernel parameters and library algorithm selection:

```bash
# Enable cuBLAS auto-tuning
XLA_FLAGS="--xla_gpu_autotune_level=4" python my_program.py

# Enable cuDNN benchmarking
XLA_FLAGS="--xla_gpu_cudnn_conv_benchmark" python my_program.py
```

---

## 15.8 GPU Compilation Flags

| Flag | Description |
|------|-------------|
| `--xla_gpu_cuda_data_dir` | Path to CUDA toolkit |
| `--xla_gpu_ftz` | Flush denormals to zero |
| `--xla_gpu_precision` | Default precision (F32 or F16) |
| `--xla_gpu_triton_gemm` | Enable Triton GEMM kernels |
| `--xla_gpu_enable_triton_softmax_fusion` | Enable Triton softmax fusion |
| `--xla_gpu_enable_triton_flash_attention` | Enable Triton flash attention |
| `--xla_gpu_autotune_level` | Auto-tuning aggressiveness (0-4) |
| `--xla_gpu_cudnn_conv_benchmark` | Enable cuDNN conv benchmarking |
| `--xla_gpu_enable_license_check` | Check library licenses |
| `--xla_gpu_max_kernel_unroll_factor` | Max loop unroll factor |
| `--xla_gpu_num_rearrange_fusion_threads` | Threads for fusion rearrangement |
| `--xla_gpu_enable_persistent_cache` | Enable kernel binary caching |
| `--xla_gpu_persistent_cache_dir` | Directory for cached binaries |
