# XLA Reference - Chapter 1: Overview and Architecture

This reference provides a comprehensive overview of XLA (Accelerated Linear Algebra), its design objectives, compilation pipeline, hardware backend support, and code generation strategies. XLA is a domain-specific compiler for linear algebra that optimizes computations described as computation graphs, producing efficient machine code for CPUs, GPUs, and TPUs.

---

## 1.1 What Is XLA?

XLA stands for **Accelerated Linear Algebra**. It is an open-source domain-specific compiler developed primarily by Google that targets numerical computations arising in machine learning workloads. XLA accepts a high-level computation graph description -- expressed in its internal intermediate representation called **HLO** (High-Level Optimizer IR) -- and applies a sequence of target-independent and target-dependent optimizations before lowering to machine code via LLVM or other code generation backends.

At its core, XLA addresses a fundamental problem in machine learning frameworks: the **operator-level execution gap**. When a framework like TensorFlow or PyTorch executes a computation graph, it typically invokes a pre-implemented kernel for each operation (matrix multiply, convolution, element-wise addition, and so on). Each such invocation incurs overhead from framework dispatch, memory bandwidth costs for reading and writing intermediate tensors, and lost optimization opportunities because the boundaries between operators prevent cross-operator optimizations such as kernel fusion.

XLA eliminates this gap by treating the entire computation graph as a single compilation unit. It analyzes the full graph, applies whole-program optimizations (fusion, memory layout assignment, rematerialization, and more), and produces a single optimized executable. This approach can yield significant improvements in both execution speed and memory usage.

### Key Characteristics

- **Graph-level optimization**: XLA operates on complete computation graphs, not individual operations. This enables optimizations impossible at the single-operator level.
- **Ahead-of-time (AOT) and just-in-time (JIT) compilation**: XLA supports both compilation modes, enabling flexibility for development and deployment.
- **Hardware portability**: The same HLO program can be compiled to run efficiently on CPUs, NVIDIA GPUs, AMD GPUs, and Google TPUs.
- **Deterministic optimization passes**: XLA's optimization pipeline is designed to be deterministic -- the same input always produces the same output, which is critical for reproducibility.

---

## 1.2 Key Objectives

XLA was designed with four primary objectives.

### 1.2.1 Improve Execution Speed

XLA improves execution speed through several mechanisms:

- **Operator fusion**: The most impactful optimization. XLA fuses multiple operations into a single kernel, eliminating the need to write intermediate results to memory and read them back. For example, a pattern like `output = relu(matmul(A, B) + bias)` can be fused into a single kernel that reads A, B, and bias, computes the matrix multiply, adds the bias, applies ReLU, and writes the output -- all without ever storing the intermediate matmul result or the sum to memory.
- **Layout optimization**: XLA assigns optimal memory layouts to all tensors, minimizing memory bank conflicts, improving cache locality, and enabling vectorized memory accesses.
- **Rematerialization**: XLA can choose to recompute values rather than store them in memory, trading computation for memory bandwidth, which is often the bottleneck.
- **Target-specific code generation**: XLA generates code tailored to the target hardware, leveraging hardware-specific instructions and intrinsics.

In practice, XLA can provide speedups ranging from 10% to 10x depending on the model architecture and hardware. Models with many small operations (such as Transformer-based models with many attention layers) tend to benefit the most from fusion.

### 1.2.2 Improve Memory Usage

XLA reduces memory consumption through:

- **Buffer aliasing**: XLA tracks the lifetime of all tensors and reuses memory buffers when tensors are no longer needed. Two tensors whose lifetimes do not overlap can share the same memory.
- **Fusion**: By eliminating intermediate tensors, fusion directly reduces peak memory usage.
- **Rematerialization**: Recomputing values instead of storing them can dramatically reduce peak memory at the cost of additional computation.
- **Memory layout optimization**: Compact layouts reduce wasted memory from padding and alignment.

### 1.2.3 Reduce the Need for Custom Operations

Without XLA, framework users often need to write custom CUDA kernels or C++ operations to achieve good performance for operation patterns that the framework's default kernel library does not handle efficiently. XLA automatically optimizes these patterns through fusion and other passes, reducing or eliminating the need for hand-written custom kernels.

For example, a user who wants to fuse a bias addition with a ReLU activation after a convolution would typically need to write a custom CUDA kernel. With XLA, this fusion happens automatically.

### 1.2.4 Improve Portability

XLA provides a unified compilation path across hardware targets. A machine learning model described in HLO can be compiled to run on:

- x86 and ARM CPUs
- NVIDIA GPUs (via CUDA/PTX code generation)
- AMD GPUs (via ROCm code generation)
- Google TPUs (via custom TPU code generation)
- Other accelerators through extensible backend interfaces

This means framework developers can write their operation implementations once and rely on XLA to generate efficient code for each target, rather than maintaining separate kernel implementations for each hardware platform.

---

## 1.3 Supported Frameworks

XLA serves as the compilation backend for several major machine learning frameworks.

### 1.3.1 JAX

JAX is the framework most tightly integrated with XLA. In JAX, every computation is compiled through XLA by default. JAX's functional programming model -- where all operations are pure functions with no hidden state -- maps naturally onto XLA's computation graph model.

Key aspects of JAX's XLA integration:

- **`jax.jit`**: The `jit` (just-in-time compilation) decorator traces a Python function into a JAXpr (JAX's intermediate representation), which is then lowered to HLO and compiled by XLA. Compilation happens on the first call with a given input shape/dtype signature; subsequent calls with the same signature execute the cached compiled code.
- **`jax.pmap`**: Parallel mapping across multiple devices, compiled through XLA with SPMD (Single Program, Multiple Data) partitioning.
- **`jax.grad` and transforms**: Higher-order transforms like `grad`, `vmap`, `pmap` are all ultimately compiled through XLA.

```python
import jax
import jax.numpy as jnp

@jax.jit
def train_step(params, x, y):
    # This entire function is compiled into a single XLA executable
    predictions = jnp.dot(x, params['w']) + params['b']
    loss = jnp.mean((predictions - y) ** 2)
    return loss

# First call triggers XLA compilation
loss = train_step(params, x_data, y_data)
```

### 1.3.2 TensorFlow

TensorFlow was the first framework to integrate XLA. Integration can be enabled at several granularity levels:

- **Auto-clustering** (`tf.config.optimizer.set_jit(True)`): TensorFlow automatically identifies clusters of XLA-compatible operations within a `tf.function` and compiles them as a group. This is the easiest but least predictable mode.
- **Explicit compilation** (`tf.function(jit_compile=True)`): The entire function is compiled through XLA. All operations within the function must be XLA-compatible.
- **`tf.raw_ops.XlaCallModule`**: For loading pre-compiled XLA AOT modules.

```python
import tensorflow as tf

# Explicit XLA compilation
@tf.function(jit_compile=True)
def train_step(params, x, y):
    predictions = tf.matmul(x, params['w']) + params['b']
    loss = tf.reduce_mean(tf.square(predictions - y))
    return loss
```

### 1.3.3 PyTorch

PyTorch integrates with XLA through the **`torch_xla`** package (also known as PyTorch/XLA), developed with Google collaboration. PyTorch/XLA provides:

- **Lazy tensor execution**: PyTorch operations are recorded as a graph and compiled by XLA when results are materialized (e.g., when printing, moving to CPU, or calling `xm.mark_step()`).
- **Device support**: XLA devices appear as `xla:0`, `xla:1`, etc., and can be used as drop-in replacements for CUDA devices.
- **Distributed training**: Integration with PyTorch's distributed training via `xla` backend, supporting multi-host TPU pods and GPU clusters.

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)

for data, target in loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)
```

---

## 1.4 The OpenXLA Project

In 2022, Google announced the **OpenXLA Project**, an open-source initiative to make XLA a community-driven, industry-standard compiler for machine learning. The project is hosted on GitHub under the `openxla` organization.

### Project Structure

The OpenXLA project encompasses several repositories and components:

| Repository | Description |
|------------|-------------|
| `xla` | The core XLA compiler, including HLO, optimization passes, and backends |
| `stablehlo` | The StableHLO intermediate representation for portability |
| `mlir-hlo` | MLIR-based dialects for HLO operations |
| `iree` | A retargetable MLIR-based compiler and runtime |
| `xla-stream-tracer` | Tooling for tracing and replaying XLA computations |

### Governance and Community

OpenXLA follows an open governance model with contributions from Google, NVIDIA, AMD, Intel, ARM, and other industry participants. Technical decisions are made through a steering committee and community discussion on GitHub and mailing lists.

### Key Principles

- **Stability**: The StableHLO opset provides a stable interface that guarantees backward compatibility, enabling compiled models to run on future versions of XLA and compatible runtimes without recompilation.
- **Extensibility**: The architecture is designed to support new hardware backends, new optimization passes, and new frontends.
- **Interoperability**: OpenXLA components can be used independently or together, and they interoperate with MLIR infrastructure.

---

## 1.5 Architecture Overview

XLA's architecture follows a layered design with well-defined interfaces between layers. The compilation pipeline flows through the following stages:

```
+------------------+     +------------+     +-----+     +--------+     +----------+
| Frontend (JAX,   | --> | StableHLO  | --> | HLO | --> |Backend | --> | Codegen  |
| TF, PyTorch)     |     | (Portable  |     | (Opt |     |Target  |     | (LLVM,   |
|                  |     |  IR)       |     |  IR) |     | (CPU,  |     |  Triton, |
|                  |     |            |     |      |     |  GPU,  |     |  libs)   |
+------------------+     +------------+     +-----+     |  TPU)  |     +----------+
                                                         +--------+
```

### 1.5.1 Frontend Layer

The frontend layer is responsible for translating framework-specific computation graphs into XLA-compatible representations. Each framework provides its own frontend:

- **JAX**: Traces Python functions into `Jaxpr`, then lowers to StableHLO/HLO.
- **TensorFlow**: Extracts graph subgraphs from `tf.function`, converts TF ops to HLO via `tf2xla` bridge.
- **PyTorch**: Uses `torch_xla` to trace PyTorch operations into a lazy execution graph, then lowers to HLO.

The frontend is responsible for:
- Shape inference and type checking
- Lowering framework operations to HLO/StableHLO operations
- Handling framework-specific constructs (control flow, random state, etc.)

### 1.5.2 StableHLO Layer

**StableHLO** (Stable High-Level Operations) is a portable, versioned operation set designed to serve as a stable interface between ML frameworks and compiler backends. Key properties:

- **Stability guarantees**: StableHLO guarantees backward compatibility for compiled programs. A program compiled against StableHLO version N will continue to work with any runtime implementing version N or later.
- **Operation set**: StableHLO defines approximately 100 operations covering the common operations needed for ML workloads -- arithmetic, comparison, tensor manipulation, control flow, and more.
- **MLIR-based**: StableHLO is implemented as an MLIR dialect, leveraging MLIR's infrastructure for type systems, pattern rewriting, and pass management.
- **Serialization**: StableHLO programs can be serialized to a portable binary format (using MLIR's bytecode), enabling storage and transport of compiled programs.

The StableHLO layer serves as a **contract** between frontends and backends. Frontends lower to StableHLO, and backends consume StableHLO (typically by converting it to HLO for optimization and code generation).

### 1.5.3 HLO Layer

**HLO** (High-Level Optimizer IR) is XLA's primary intermediate representation and the locus of most optimization work. HLO is a dataflow graph where:

- Each **node** represents an operation (e.g., `dot`, `convolution`, `add`, `fusion`).
- Each **edge** represents a data dependency (a tensor flowing from one operation to another).
- Each operation has an associated **shape** describing the type, dimensions, and memory layout of its output.

The HLO layer includes:

- **HLO Module**: The top-level compilation unit, containing one or more computation graphs (entry computation and potentially called computations).
- **HLO Computation**: A single computation graph with a list of instructions.
- **HLO Instruction**: A single operation node with its opcode, operands, shape, and metadata.
- **HLO Pass Manager**: Manages the optimization pipeline, running passes in sequence with scheduling and verification.

Example HLO textual representation:

```
HloModule main

ENTRY main {
  parameter.0 = f32[128,256] parameter(0)
  parameter.1 = f32[256,512] parameter(1)
  parameter.2 = f32[512] parameter(2)
  dot.0 = f32[128,512] dot(parameter.0, parameter.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast.0 = f32[128,512] broadcast(parameter.2), dimensions={1}
  add.0 = f32[128,512] add(dot.0, broadcast.0)
  ROOT relu.0 = f32[128,512] max(add.0, f32[] constant(0))
}
```

### 1.5.4 Backend Layer

The backend layer is responsible for target-specific decisions. XLA supports multiple backends, each implementing a common interface (`se::StreamExecutor` or the newer `PjRtClient` interface):

| Backend | Class | Target |
|---------|-------|--------|
| `xla::cpu::CpuCompiler` | CPU backend | x86-64, ARM, RISC-V |
| `xla::gpu::GpuCompiler` | GPU backend (NVIDIA) | NVIDIA GPUs via CUDA/PTX |
| `xla::gpu::GpuCompiler` | GPU backend (AMD) | AMD GPUs via ROCm |
| `xla::tpu::TpuCompiler` | TPU backend | Google TPU v2/v3/v4/v5 |

Each backend is responsible for:
- **Device memory management**: Allocating and managing device memory buffers.
- **Executable generation**: Converting optimized HLO to device-specific executable code.
- **Kernel launching**: Executing compiled kernels on the device.
- **Stream/command queue management**: Managing asynchronous execution on the device.

### 1.5.5 Code Generation Layer

The code generation layer translates optimized HLO operations into executable machine code. XLA employs several code generation strategies:

1. **LLVM-based code generation**: The primary approach. XLA lowers HLO to LLVM IR, applies LLVM optimization passes, and generates machine code via LLVM's target backends. This is used for both CPU (x86, ARM) and GPU (NVIDIA PTX, AMD ROCm).

2. **Triton-based code generation**: For certain GPU operations (particularly fusion regions), XLA can generate Triton IR, which is then compiled to PTX by the Triton compiler. Triton provides a higher-level abstraction for GPU programming that can produce efficient code for complex fusion patterns.

3. **Library calls (cuBLAS, cuDNN, etc.)**: For well-known operations like matrix multiplication and convolution, XLA can emit calls to highly optimized vendor libraries rather than generating custom code. The compiler makes cost-based decisions about when to use library calls vs. custom code.

4. **TPU custom code generation**: For TPU targets, XLA uses a custom code generation path that targets the TPU's systolic array architecture, generating specialized instructions for the TPU's matrix multiply units, vector units, and inter-chip interconnect.

---

## 1.6 Compilation Pipeline Stages

The XLA compilation pipeline proceeds through the following major stages. Each stage may consist of multiple optimization passes.

### Stage 1: Frontend Export

The framework frontend traces or exports the user's computation graph:

1. **Graph capture**: The framework captures the computation graph from user code (via tracing, AST analysis, or graph extraction).
2. **Shape inference**: All tensor shapes and dtypes are inferred and validated.
3. **Operation lowering**: Framework operations are lowered to StableHLO/HLO operations.
4. **Module construction**: The complete HLO module is constructed with entry computation, called computations (for conditionals, loops, etc.), and metadata.

### Stage 2: HLO Optimization (Target-Independent)

This stage applies a series of target-independent optimization passes to the HLO module. These passes transform the HLO graph to improve performance without considering target-specific details.

Key optimization passes include:

| Pass | Description |
|------|-------------|
| **Algebraic Simplifier** | Simplifies algebraic expressions (e.g., `x + 0 = x`, `x * 1 = x`, `multiply by power of 2 -> shift`) |
| **Constant Folding** | Evaluates operations on constant inputs at compile time |
| **Operator Fusion** | Identifies and fuses compatible operation patterns into fusion instructions |
| **Reshape Mover** | Moves reshapes to enable additional fusion opportunities |
| **Transpose Folding** | Fuses transposes into adjacent operations (e.g., into dot operations) |
| **Dot Dimension Sink** | Normalizes dot operations to canonical dimension ordering |
| **Layout Assignment** | Assigns optimal memory layouts to all instructions |
| **Rematerialization** | Identifies opportunities to recompute values to reduce memory pressure |
| **Reduction Decomposer** | Decomposes complex reductions into simpler patterns |
| **Convolution Canonicalization** | Normalizes convolution operations |
| **While Loop Simplifier** | Simplifies while loops (unrolling, trip count analysis) |
| **Slicing** | Moves slice operations to enable fusion |
| **HLO DCE** | Dead Code Elimination -- removes unused instructions |
| **CSE** | Common Subexpression Elimination |
| **Tuple Simplifier** | Simplifies tuple operations |
| **Broadcast Canonicalizer** | Normalizes broadcast operations |
| **Convolution PadExtractor** | Extracts padding from convolution into explicit pad operations |

The fusion pass is particularly important and operates with the following strategy:

1. **Sibling fusion**: Operations that share the same operand are fused into a single kernel.
2. **Producer-consumer fusion**: An operation is fused with its consumer if the fusion is profitable (typically when the producer is element-wise and small relative to the consumer).
3. **Multi-output fusion**: Multiple operations with the same loop structure are fused into a single kernel that produces multiple outputs.

The fusion decision is guided by a cost model that considers:
- Whether the fused operation fits within the kernel launch overhead budget
- Whether the fused computation can be efficiently parallelized
- Memory access patterns and register pressure
- Hardware-specific constraints (e.g., shared memory limits on GPUs)

### Stage 3: HLO Scheduling and Buffer Assignment

After optimization, the HLO module is scheduled and memory is assigned:

1. **Instruction scheduling**: Instructions are ordered to minimize peak memory usage while respecting data dependencies. The scheduler considers liveness analysis to order instructions such that buffers can be reused as early as possible.
2. **Buffer allocation**: Virtual buffers are assigned to physical memory offsets. The buffer allocator uses:
   - **Buffer aliasing**: Multiple instructions whose live ranges do not overlap share the same memory buffer.
   - **Color-based allocation**: Buffers are assigned to memory spaces (e.g., default memory vs. alternate memory for GPU shared memory).
3. **Heap simulation**: A heap simulation determines the total memory allocation and ensures it stays within device limits.

### Stage 4: Target-Specific Lowering

The optimized HLO is lowered to target-specific representations:

1. **Legalization**: Target-unsupported HLO operations are legalized to supported operations. For example, complex operations may be decomposed into simpler primitives that the target supports.
2. **Library call matching**: Patterns that match highly optimized library implementations (cuBLAS, cuDNN, etc.) are replaced with library calls.
3. **Thunk/emitter generation**: For each HLO instruction, the backend generates a "thunk" (a deferred execution command) or invokes an emitter that produces the actual device code.

### Stage 5: Code Emission

Device code is generated for each kernel:

1. **LLVM IR generation**: HLO operations are lowered to LLVM IR. Element-wise operations become LLVM vector operations. Data movement operations become LLVM memory operations. Control flow becomes LLVM branches.
2. **LLVM optimization**: Standard LLVM optimization passes are applied (inlining, loop optimization, SROA, etc.).
3. **Machine code generation**: LLVM generates target machine code (x86, ARM, PTX, AMDGPU).
4. **Linking**: Generated code is linked with runtime libraries and library calls.

### Stage 6: Executable Assembly

The final stage assembles the compiled executable:

1. **Executable construction**: All generated kernels, library call thunks, and buffer assignment plans are assembled into an `XlaExecutable` object.
2. **Serialization** (optional): The executable may be serialized for later loading, enabling AOT deployment.
3. **Execution**: At runtime, the executable is invoked with input buffers. The runtime:
   - Allocates output buffers according to the buffer assignment plan.
   - Enqueues kernel launches and library calls in the correct order.
   - Synchronizes on completion.

---

## 1.7 Hardware Backends

### 1.7.1 CPU Backend

The CPU backend targets x86-64 and ARM architectures. Key characteristics:

- **Code generation via LLVM**: All kernels are compiled through LLVM, targeting x86-64 (with AVX2, AVX-512 support) or ARM (with NEON, SVE support).
- **Multi-threaded execution**: The CPU backend uses a thread pool (Eigen or OpenMP) to parallelize across CPU cores.
- **Operations**: Supports all standard HLO operations. Matrix multiplications are dispatched to Eigen or oneDNN.
- **Layout**: Uses row-major (C-contiguous) layout by default, matching CPU cache behavior.

The CPU backend is useful for:
- Development and testing without GPU/TPU hardware
- Deployment on server hardware without accelerators
- Edge and mobile deployment

### 1.7.2 GPU Backend (NVIDIA)

The NVIDIA GPU backend is the most mature GPU backend. Key characteristics:

- **Code generation**: Uses LLVM to generate PTX code, which is then JIT-compiled to native GPU machine code by the CUDA driver.
- **Library integration**: Dispatches to cuBLAS (matrix multiply), cuDNN (convolution, attention), cuFFT (FFT), NCCL (collective communication).
- **Fusion**: Uses custom emitters for fused operations, generating optimized GPU kernels that minimize memory traffic.
- **Memory management**: Manages GPU memory allocation, including support for unified memory and memory bandwidth optimization.
- **Stream execution**: Operations are enqueued on CUDA streams for asynchronous execution.
- **Multi-GPU**: Supports multi-GPU execution with data parallelism and model parallelism through collective operations (AllReduce, AllGather, etc.).

The GPU backend supports:
- Compute capabilities from 6.0 (Pascal) through 9.0 (Hopper) and beyond
- Mixed precision (FP32, FP16, BF16, FP8, INT8)
- Tensor Core for accelerated matrix operations

### 1.7.3 GPU Backend (AMD)

The AMD GPU backend follows a similar architecture to the NVIDIA backend but targets AMD hardware:

- **Code generation**: Uses LLVM to generate AMDGPU code.
- **Library integration**: Dispatches to MIOpen (equivalent to cuDNN), rocBLAS (equivalent to cuBLAS), and other ROCm libraries.
- **Triton support**: Can use Triton for fusion code generation, similar to the NVIDIA backend.

### 1.7.4 TPU Backend

The TPU backend targets Google's Tensor Processing Units. Key characteristics:

- **Custom ISA**: TPU has a custom instruction set architecture. XLA generates TPU-specific microcode.
- **Systolic array**: The TPU's systolic array is programmed via specialized instructions for matrix operations.
- **Inter-chip interconnect**: TPU pods communicate via high-speed interconnect (ICI). XLA generates cross-chip communication instructions.
- **SPMD partitioning**: XLA's SPMD partitioner automatically partitions computations across TPU cores and chips.

TPU generations:
- **TPU v2**: 180 TFLOPS (BF16), 16 GB HBM per core
- **TPU v3**: 420 TFLOPS (BF16), 32 GB HBM per core
- **TPU v4**: 275 TFLOPS (BF16), 32 GB HBM per core, improved interconnect
- **TPU v5**: Higher performance with support for FP8 and INT8

---

## 1.8 Code Generation Approaches

### 1.8.1 LLVM-Based Code Generation

LLVM-based code generation is the primary approach for CPU and GPU targets. The flow is:

```
HLO Instruction
    |
    v
Emitter (target-specific)
    |
    v
LLVM IR Module
    |
    v
LLVM Optimization Passes
    |
    v
LLVM Backend (x86, ARM, NVPTX, AMDGPU)
    |
    v
Machine Code / PTX
```

For the CPU backend, each fused HLO instruction is emitted as an LLVM function. The emitter:
1. Creates an LLVM function with appropriate parameters (input/output buffer pointers, dimensions, etc.).
2. Generates LLVM IR for the computation, using LLVM vector types and intrinsics.
3. Applies LLVM optimization passes (O2/O3 level).
4. Generates x86-64 or ARM machine code.

For the GPU backend, each fused HLO instruction is emitted as a GPU kernel (a `__global__` function in CUDA terms). The emitter:
1. Creates an LLVM function representing the GPU kernel.
2. Generates LLVM IR with GPU-specific constructs (thread indices, shared memory, etc.).
3. Applies LLVM optimization passes.
4. Generates PTX code, which is further compiled by the CUDA driver to native GPU code.

### 1.8.2 Triton-Based Code Generation

For GPU targets, XLA can use the Triton compiler for code generation in certain cases. Triton provides a higher-level GPU programming model:

- **Block-level programming**: Triton programs operate on blocks of data, abstracting away thread-level details.
- **Automatic memory management**: Triton handles shared memory allocation and data movement automatically.
- **Efficient code generation**: Triton generates optimized PTX code through LLVM.

XLA uses Triton for:
- Complex fusion patterns that benefit from Triton's programming model
- Operations where Triton's auto-tuning can find optimal configurations
- Emerging operation patterns where custom emitters do not yet exist

### 1.8.3 Library-Based Code Generation

For well-known operations, XLA can emit calls to optimized libraries rather than generating custom code. This approach is used when:

- The library implementation is significantly more optimized than what XLA could generate (e.g., cuBLAS matrix multiply with Tensor Core utilization).
- The operation requires hardware-specific knowledge that is difficult to express in LLVM IR (e.g., cuDNN convolution with implicit GEMM).
- The operation benefits from runtime auto-tuning (e.g., cuDNN's convolution algorithm selection).

Libraries used:

| Library | Operations | Backend |
|---------|-----------|---------|
| cuBLAS | MatMul, BatchMatMul, Triton | NVIDIA GPU |
| cuDNN | Convolution, ConvBackprop, BatchNorm, Attention | NVIDIA GPU |
| MIOpen | Convolution | AMD GPU |
| rocBLAS | MatMul | AMD GPU |
| oneDNN | MatMul, Convolution | CPU |
| NCCL / RCCL | AllReduce, AllGather, ReduceScatter, CollectiveBroadcast | GPU |
| cuFFT / rocFFT | FFT, IFFT | GPU |

The decision of whether to use a library call or generate custom code is made by the backend's **instruction canonicalization** and **library call selection** passes, which use cost models and heuristics.

---

## 1.9 Compilation Modes

### 1.9.1 Just-In-Time (JIT) Compilation

In JIT mode, XLA compiles the computation graph at runtime, just before execution. This is the primary mode used during model development and interactive work.

JIT compilation flow:
1. The framework traces the computation and produces an HLO module.
2. XLA compiles the HLO module to an executable (this may take seconds to minutes for large models).
3. The executable is cached keyed by the input shape/dtype signature.
4. Subsequent calls with the same signature reuse the cached executable.
5. Calls with different signatures trigger recompilation.

### 1.9.2 Ahead-Of-Time (AOT) Compilation

In AOT mode, XLA compiles the computation graph ahead of time, producing a serialized executable that can be loaded and run without compilation overhead.

AOT compilation flow:
1. The computation graph is exported to StableHLO or HLO.
2. XLA compiles the module for the target hardware.
3. The compiled executable is serialized (e.g., as a `.a` file for CPU, `.so` for GPU).
4. At deployment time, the serialized executable is loaded and run directly.

AOT compilation is used for:
- Production deployment where compilation latency is unacceptable
- Edge and mobile deployment where the compilation infrastructure is not available
- Cross-compilation (compiling on one machine for execution on another)

---

## 1.10 Runtime Architecture

The XLA runtime is responsible for executing compiled executables and managing device resources. The modern runtime architecture is based on **PjRt** (Pretty JAX Runtime).

### PjRt Architecture

PjRt defines a common interface between frameworks and device runtimes:

```
+-------------------+
| Framework (JAX,   |
| TF, PyTorch)      |
+--------+----------+
         |
         v
+--------+----------+
| PjRt Client        |  <-- Framework-facing API
+--------+----------+
         |
         v
+--------+----------+
| PjRt StreamExecutor|  <-- Device-specific implementation
| (GPU, TPU, CPU)    |
+-------------------+
```

Key PjRt components:

- **`PjRtClient`**: The top-level interface. Provides methods for device management, executable compilation, and buffer transfer.
- **`PjRtDevice`**: Represents a single device (GPU, TPU core, CPU). Provides methods for buffer allocation and execution.
- **`PjRtLoadedExecutable`**: A compiled executable bound to specific devices. Provides methods for execution and input/output shape queries.
- **`PjRtBuffer`**: A data buffer residing on a device. Provides methods for data transfer, shape queries, and layout information.

### Execution Flow

1. The framework creates a `PjRtClient` for the target device type.
2. The framework compiles a computation graph to a `PjRtLoadedExecutable` via `client.compile()`.
3. To execute, the framework transfers input data to device buffers and calls `executable.execute()`.
4. The runtime enqueues the execution on the device stream.
5. Output buffers are returned to the framework.

---

## 1.11 Debugging and Introspection

XLA provides several tools and environment variables for debugging and performance analysis.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `XLA_FLAGS=--xla_dump_to=<dir>` | Dump HLO, LLVM IR, and assembly to the specified directory |
| `XLA_FLAGS=--xla_dump_hlo_as_text` | Dump HLO as human-readable text |
| `XLA_FLAGS=--xla_dump_hlo_as_proto` | Dump HLO as binary protobuf |
| `XLA_FLAGS=--xla_dump_hlo_snapshots` | Dump HLO snapshots at various compilation stages |
| `XLA_FLAGS=--xla_disable_hlo_passes=<pass1>,<pass2>` | Disable specific optimization passes |
| `XLA_FLAGS=--xla_enable_hlo_passes=<pass>` | Enable specific optimization passes |
| `XLA_FLAGS=--xla_backend_optimization_level=<O0,O1,O2,O3>` | Control LLVM optimization level |
| `TF_XLA_FLAGS=--tf_xla_auto_jit=2` | Enable auto-clustering in TensorFlow |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=<frac>` | Fraction of device memory to preallocate (JAX) |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Disable memory preallocation (JAX) |

### HLO Inspection

The `xla.hlo` module and `jax.hlo` utilities allow programmatic inspection of compiled HLO:

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x, y):
    return jnp.dot(x, y) + jnp.sin(x)

# Print the HLO
print(f.lower(jnp.ones((2, 3)), jnp.ones((3, 4))).as_text())

# Print the optimized HLO
print(f.lower(jnp.ones((2, 3)), jnp.ones((3, 4))).compiler_ir(dialect='hlo').as_hlo_text())
```

### Performance Analysis

- **XLA profiling**: Integrates with TensorBoard via the XLA profiler plugin.
- **NVIDIA Nsight**: GPU executables can be profiled with NVIDIA Nsight Systems and Nsight Compute.
- **XLA Compilation metrics**: JAX exposes compilation time and cache hit metrics.

---

## 1.12 Summary

XLA is a domain-specific compiler that sits between ML frameworks and hardware, providing graph-level optimization and efficient code generation. Its architecture is organized into clear layers -- frontend, StableHLO, HLO, backend, and code generation -- each with well-defined responsibilities. The optimization pipeline applies target-independent passes (fusion, layout assignment, rematerialization) followed by target-specific lowering and code generation via LLVM, Triton, or library calls. XLA supports CPU, NVIDIA GPU, AMD GPU, and TPU backends, and is integrated with JAX, TensorFlow, and PyTorch through the OpenXLA project and PjRt runtime interface.
