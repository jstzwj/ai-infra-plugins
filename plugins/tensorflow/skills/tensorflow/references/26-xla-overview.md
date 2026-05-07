# XLA (Accelerated Linear Algebra) Overview Reference

This document provides a comprehensive reference for XLA, TensorFlow's
optimizing compiler for machine learning workloads. XLA compiles subgraphs
from TensorFlow graphs into optimized machine code for various hardware
backends.

## Table of Contents

1. [XLA Overview](#xla-overview)
2. [Compilation Pipeline](#compilation-pipeline)
3. [XLA Backends](#xla-backends)
4. [XLA Client Architecture](#xla-client-architecture)
5. [PjRtClient Interface](#pjrtclient-interface)
6. [Executable](#executable)
7. [HloModule and HloComputation](#hlomodule-and-hlocomputation)
8. [Shape System](#shape-system)
9. [Literal Values](#literal-values)
10. [XLA in TensorFlow](#xla-in-tensorflow)
11. [Performance Benefits](#performance-benefits)
12. [XLA Debugging](#xla-debugging)
13. [XLA Environment Variables](#xla-environment-variables)
14. [XLA Optimization Passes](#xla-optimization-passes)
15. [XLA Compilation Cache](#xla-compilation-cache)

---

## XLA Overview

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
algebra that optimizes TensorFlow computations. It was developed by Google
to improve the performance and memory usage of machine learning models.

### Purpose and Goals

1. **Execution Speed**: Fuse operations to reduce memory bandwidth pressure
   and eliminate kernel launch overhead
2. **Memory Efficiency**: Optimize memory allocation and reuse, reducing
   peak memory usage
3. **Portability**: Single compiler infrastructure targeting CPU, GPU, and TPU
4. **Custom Hardware**: Enable efficient compilation for new hardware via
   pluggable backend architecture

### Key Concepts

- **HLO (High-Level Optimizer) IR**: XLA's intermediate representation
- **Compilation Unit**: An `HloModule` containing one or more `HloComputation`s
- **Fusion**: Combining multiple operations into a single kernel
- **Layout**: Memory layout of tensors (dimension ordering, tiling)

---

## Compilation Pipeline

The XLA compilation pipeline transforms high-level computations into
optimized machine code.

### Pipeline Stages

```
1. TensorFlow Graph
   |
2. Auto-clustering / Explicit Compilation Markers
   |
3. Graph -> HLO Conversion
   |
4. HLO Module
   |
5. HLO Optimization Passes
   |   - Algebraic simplification
   |   - Fusion
   |   - Layout assignment
   |   - Scheduling
   |   - Buffer assignment
   |
6. Backend-Specific Lowering
   |   - CPU: LLVM IR -> x86/ARM machine code
   |   - GPU: LLVM IR -> PTX/NVIDIA assembly
   |   - TPU: Custom code generation
   |
7. Executable
   |
8. Execution on Device
```

### Detailed Stage Descriptions

#### Stage 1-2: TensorFlow Graph to Clusters

TensorFlow identifies subgraphs for XLA compilation:

- **Auto-clustering** (`auto_mixed_precision` + XLA auto-clustering): Automatically
  identifies clusters of XLA-compatible ops
- **Explicit compilation**: User marks functions with `@tf.function(jit_compile=True)`
  or `tf.xla.experimental.compile()`

#### Stage 3: Graph to HLO Conversion

Each TensorFlow cluster is converted to an HLO module:

- TensorFlow ops map to HLO instructions
- Tensors map to HLO shapes
- Device placement is preserved
- Function calls are inlined or converted to HLO computations

#### Stage 5: HLO Optimization Passes

The core optimization pipeline includes:

1. **HloConstantFolding**: Fold constant expressions
2. **AlgebraicSimplifier**: Simplify arithmetic (remove identity ops, etc.)
3. **HloCSE**: Common subexpression elimination
4. **HloDCE**: Dead code elimination
5. **Fusion**: Merge compatible operations into single kernels
6. **LayoutAssignment**: Assign optimal memory layouts
7. **HloSchedule**: Schedule instruction execution order
8. **BufferAssignment**: Allocate memory buffers

#### Stage 6: Backend Lowering

Each backend translates HLO to machine code:

**CPU Backend**:
```
HLO -> LLVM IR -> LLVM optimization -> x86/ARM machine code
```

**GPU Backend**:
```
HLO -> LLVM IR -> NVPTX backend -> PTX -> SASS (via driver)
```

**TPU Backend**:
```
HLO -> Custom code generation -> TPU instructions
```

---

## XLA Backends

### CPU Backend

Uses LLVM to generate optimized x86 and ARM code.

**Features**:
- Multi-threaded execution via Eigen/ThreadPool
- SIMD vectorization (SSE, AVX, AVX-512, NEON)
- Loop fusion for memory bandwidth optimization
- oneDNN integration for CPU-optimized kernels

**LLVM Pipeline**:
```
HLO -> llvm::IRBuilder -> LLVM Module
    -> LLVM PassManager (optimization)
    -> TargetMachine (code emission)
    -> Object file
    -> JIT execution
```

### GPU Backend

Generates PTX/NVIDIA assembly for GPU execution.

**Features**:
- Kernel fusion for reduced memory bandwidth
- Shared memory optimization
- Warp-level primitives
- Tensor Core utilization for mixed precision
- Multi-stream execution

**Kernel Emission**:
```
HLO Fusion Kernel -> LLVM GPU Kernel -> PTX -> SASS
```

**Key optimizations**:
- Thread coarsening for small kernels
- Shared memory tiling
- Register allocation optimization
- Reduction tree optimization

### TPU Backend

Google's custom TPU hardware backend.

**Features**:
- Systolic array matrix multiplication
- Bfloat16 native support
- High-speed inter-chip interconnect
- Custom collective operations

### Interpreter Backend

Reference implementation for testing and debugging.

```cpp
class HloEvaluator {
 public:
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal> arguments);
};
```

---

## XLA Client Architecture

XLA provides several client interfaces for compilation and execution.

### LocalClient

In-process client for local device compilation and execution:

```cpp
class LocalClient : public Client {
 public:
  // Compile a computation to an executable
  StatusOr<std::unique_ptr<LocalExecutable>> Compile(
      const XlaComputation& computation,
      absl::Span<const Shape> argument_shapes,
      const ExecutableBuildOptions& options);

  // Transfer data to device
  StatusOr<ScopedShapedBuffer> LiteralToShapedBuffer(
      const LiteralSlice& literal, int device_ordinal);

  // Transfer data from device
  StatusOr<Literal> ShapedBufferToLiteral(
      const ShapedBuffer& shaped_buffer);

  // Execute
  StatusOr<ScopedShapedBuffer> Execute(
      const LocalExecutable& executable,
      absl::Span<const ShapedBuffer* const> arguments);
};
```

### Client (Remote)

Remote client for distributed XLA execution:

```cpp
class Client {
 public:
  virtual StatusOr<std::unique_ptr<ProgramShape>> GetComputationShape(
      const XlaComputation& computation) = 0;

  virtual StatusOr<std::unique_ptr<GlobalData>> Execute(
      const XlaComputation& computation,
      absl::Span<const GlobalData*> arguments) = 0;

  virtual StatusOr<std::vector<std::unique_ptr<GlobalData>>> ExecuteParallel(
      const std::vector<const XlaComputation*>& computations,
      const std::vector<absl::Span<const GlobalData*>>& arguments) = 0;

  virtual StatusOr<Literal> Transfer(const GlobalData& data) = 0;
};
```

### XlaBuilder

Builds XLA computations programmatically:

```cpp
class XlaBuilder {
 public:
  explicit XlaBuilder(const string& computation_name);

  // Parameter
  XlaOp Parameter(int64_t parameter_number, const Shape& shape,
                  const string& name);

  // Constants
  XlaOp ConstantLiteral(const Literal& literal);
  XlaOp ConstantR0(float value);
  XlaOp ConstantR1(absl::Span<const float> values);

  // Operations
  XlaOp Add(XlaOp lhs, XlaOp rhs);
  XlaOp Mul(XlaOp lhs, XlaOp rhs);
  XlaOp Dot(XlaOp lhs, XlaOp rhs);
  XlaOp Conv(XlaOp lhs, XlaOp rhs, int64_t feature_group_count,
             const ConvolutionDimensionNumbers& dims,
             const Window& window);

  // Build the computation
  StatusOr<XlaComputation> Build(XlaOp root);
};
```

---

## PjRtClient Interface

PjRtClient (Portable JIT Runtime) is a device-agnostic runtime interface
designed for use with JAX, TensorFlow, and other ML frameworks.

### Interface

```cpp
class PjRtClient {
 public:
  // Device information
  virtual int device_count() const = 0;
  virtual int addressable_device_count() const = 0;
  virtual PjRtDevice* device(int id) const = 0;
  virtual PjRtDevice* addressable_device(int id) const = 0;

  // Compilation
  virtual StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation,
      CompileOptions options) = 0;

  // Buffer management
  virtual StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, const Shape& shape,
      HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) = 0;

  // Execution
  virtual StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
  Execute(PjRtLoadedExecutable* executable,
          const std::vector<std::vector<PjRtBuffer*>>& arguments,
          const ExecuteOptions& options) = 0;
};
```

### PjRtDevice

```cpp
class PjRtDevice {
 public:
  virtual int id() const = 0;
  virtual string_view device_kind() const = 0;
  virtual string_view ToString() const = 0;
  virtual Status TransferToInfeed(const LiteralSlice& literal) = 0;
  virtual Status TransferFromOutfeed(MutableBorrowingLiteral literal) = 0;
};
```

### PjRtBuffer

```cpp
class PjRtBuffer {
 public:
  virtual Shape shape() const = 0;
  virtual Shape on_device_shape() const = 0;
  virtual StatusOr<std::unique_ptr<Literal>> ToLiteralSync() = 0;
  virtual bool IsDeleted() const = 0;
};
```

### PjRt Implementations

| Implementation | Target | Description |
|----------------|--------|-------------|
| `PjRtStreamExecutorClient` | GPU/CPU | StreamExecutor-based client |
| `TfrtCpuClient` | CPU | TFRT-based CPU client |
| `GpuClient` | GPU | GPU client with CUDA/ROCm |
| `TpuClient` | TPU | TPU client |

---

## Executable

### LocalExecutable

A compiled XLA program that can be executed on a specific device:

```cpp
class LocalExecutable {
 public:
  // Execute the computation
  StatusOr<ScopedShapedBuffer> Execute(
      absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions options);

  // Execute asynchronously
  StatusOr<std::pair<ScopedShapedBuffer, ExecutionProfile>>
  ExecuteWithProfiling(absl::Span<const ShapedBuffer* const> arguments,
                       ExecutableRunOptions options);

  // Execute multiple computations
  StatusOr<std::vector<ScopedShapedBuffer>> ExecuteOnStreams(
      absl::Span<const ExecuteArguments> arguments);

  // Access the HLO module
  const HloModule& module() const;
};
```

### ExecutableRunOptions

```cpp
struct ExecutableRunOptions {
  int device_ordinal = 0;
  Stream* stream = nullptr;
  ThreadPool* intra_op_thread_pool = nullptr;
  RngState* rng_state = nullptr;
  int64_t run_id = -1;
  bool allocator_wants_profiling = false;
};
```

### ExecutionProfile

```cpp
struct ExecutionProfile {
  bool compilation_cache_hit = false;
  int64_t compile_time_ms = 0;
  int64_t compute_cycle_count = 0;
  int64_t compute_time_ns = 0;
  int64_t compute_and_transfer_time_ns = 0;
  int64_t executable_size_in_bytes = 0;
};
```

---

## HloModule and HloComputation

### HloModule

The top-level compilation unit in XLA. Contains one entry computation
and zero or more called computations.

```cpp
class HloModule {
 public:
  const std::string& name() const;
  const HloComputation* entry_computation() const;
  const std::vector<HloComputation*>& computations() const;

  // Add/remove computations
  HloComputation* AddComputation(std::unique_ptr<HloComputation> computation,
                                  bool is_entry);
  void RemoveComputation(HloComputation* computation);

  // Configuration
  const HloModuleConfig& config() const;
  const BackendConfig& backend_config() const;
};
```

### HloModuleConfig

```cpp
struct HloModuleConfig {
  // Replication parameters
  int64_t replica_count = 1;
  int64_t num_partitions = 1;

  // Debug options
  const DebugOptions& debug_options() const;

  // Entry computation layout
  std::optional<ProgramShape> entry_computation_layout;
};
```

### HloComputation

A function-level IR unit. Contains a control flow graph of HLO instructions.

```cpp
class HloComputation {
 public:
  const std::string& name() const;

  // Instructions
  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction);
  const std::vector<HloInstruction*>& instructions() const;

  // Root instruction (the computation result)
  HloInstruction* root_instruction() const;

  // Parameters
  int64_t num_parameters() const;
  HloInstruction* parameter_instruction(int64_t param_no) const;

  // Shape
  const Shape& shape() const;

  // Parent module
  HloModule* parent() const;
};
```

---

## Shape System

### Shape

XLA uses the `Shape` class (from `xla/shape.h`) to describe tensor and tuple
types.

```cpp
class Shape {
 public:
  // Constructors
  Shape();  // Invalid shape
  explicit Shape(PrimitiveType element_type);  // Token/opaque/buffer
  Shape(PrimitiveType element_type,
        absl::Span<const int64_t> dimensions,
        absl::Span<const bool> dynamic_dimensions = {});  // Array
  explicit Shape(std::vector<Shape> tuple_shapes);  // Tuple

  // From/to proto
  static StatusOr<Shape> FromProto(const ShapeProto& proto);
  ShapeProto ToProto() const;

  // Type queries
  bool IsArray() const;      // Array (non-tuple, non-token, etc.)
  bool IsTuple() const;      // Tuple type
  bool IsToken() const;      // Token type
  bool IsBuffer() const;     // Buffer type
  bool IsOpaque() const;     // Opaque type
  bool IsValid() const;      // Well-formed shape

  // Array properties
  PrimitiveType element_type() const;
  int64_t rank() const;
  int64_t dimensions(int64_t dimension) const;
  absl::Span<const int64_t> dimensions() const;
  bool is_dynamic_dimension(int64_t dimension) const;
  int64_t num_elements() const;

  // Tuple properties
  const std::vector<Shape>& tuple_shapes() const;

  // Layout
  bool has_layout() const;
  const Layout& layout() const;
  Layout* mutable_layout();

  // String representation
  std::string ToString(bool print_layout = false) const;
};
```

### ShapeProto

From `xla/xla_data.proto`:

```protobuf
message ShapeProto {
  PrimitiveType element_type = 2;
  repeated int64 dimensions = 3;
  repeated bool is_dynamic_dimension = 6;
  repeated ShapeProto tuple_shapes = 4;
  LayoutProto layout = 5;
}
```

### PrimitiveType Enum

```protobuf
enum PrimitiveType {
  PRIMITIVE_TYPE_INVALID = 0;
  PRED = 1;         // Boolean
  S2 = 26; S4 = 21; S8 = 2; S16 = 3; S32 = 4; S64 = 5;
  U2 = 27; U4 = 22; U8 = 6; U16 = 7; U32 = 8; U64 = 9;
  F16 = 10; F32 = 11; F64 = 12; BF16 = 16;
  F8E5M2 = 19; F8E4M3 = 28; F8E4M3FN = 20;
  C64 = 15; C128 = 18;
  TUPLE = 13; OPAQUE_TYPE = 14; TOKEN = 17; BUFFER = 34;
}
```

### Layout

Describes how data is laid out in memory:

```protobuf
message LayoutProto {
  repeated int64 minor_to_major = 1;       // Dimension ordering
  repeated DimLevelType dim_level_types = 9; // Sparse/dense encoding
  repeated TileProto tiles = 6;             // Tiling configuration
  int64 element_size_in_bits = 7;
  int64 memory_space = 8;                   // Memory space identifier
  int64 tail_padding_alignment_in_elements = 16;
  repeated SplitConfigProto split_configs = 17;
}
```

### Layout Example

For a 2D array [128, 64]:
```
Layout {0, 1} means: dimension 0 is minor (fastest varying), dim 1 is major
  Memory layout: row-major equivalent
  Element [i, j] is at offset: i * 64 + j

Layout {1, 0} means: dimension 1 is minor, dim 0 is major
  Memory layout: column-major equivalent
  Element [i, j] is at offset: j * 128 + i
```

---

## Literal Values

### Literal

A `Literal` holds concrete values in XLA, analogous to a constant tensor:

```cpp
class Literal {
 public:
  // Construction
  static Literal CreateFromShape(const Shape& shape);
  static Literal CreateFromDimensions(PrimitiveType primitive_type,
                                       absl::Span<const int64_t> dimensions);

  // Data access (rank-specific)
  float Get<float>(absl::Span<const int64_t> multi_index) const;
  void Set<float>(absl::Span<const int64_t> multi_index, float value);

  // Full data access
  void* data();
  const void* data() const;
  int64_t size_bytes() const;

  // Shape
  const Shape& shape() const;

  // Serialization
  LiteralProto ToProto() const;
  static StatusOr<Literal> CreateFromProto(const LiteralProto& proto);
};
```

### LiteralProto

```protobuf
message LiteralProto {
  ShapeProto shape = 1;
  repeated bool preds = 2;
  bytes s8s = 15;
  repeated int32 s32s = 4;
  repeated int64 s64s = 5;
  bytes u8s = 3;
  repeated uint32 u32s = 6;
  repeated uint64 u64s = 7;
  repeated float f32s = 8;
  repeated double f64s = 9;
  bytes f16s = 11;
  bytes bf16s = 13;
  repeated float c64s = 12;
  repeated double c128s = 18;
  repeated LiteralProto tuple_literals = 10;
  repeated int64 sparse_indices = 14;
}
```

---

## XLA in TensorFlow

### Auto-Clustering

XLA automatically identifies clusters of ops that can be compiled together:

```python
# Enable auto-clustering (default in TF 2.x)
tf.config.optimizer.set_jit(True)

# Disable auto-clustering
tf.config.optimizer.set_jit(False)
```

Auto-clustering rules:
1. Ops must be XLA-compatible (have HLO lowering)
2. Ops must be on the same device
3. Cluster boundaries at non-XLA ops, control flow, or device transfers
4. Minimum cluster size threshold

### Explicit Compilation

```python
# Using jit_compile=True
@tf.function(jit_compile=True)
def my_compiled_function(x):
    return tf.matmul(x, x) + tf.reduce_sum(x)

# Using experimental_compile
@tf.function(experimental_compile=True)
def my_compiled_function(x):
    return tf.matmul(x, x)

# Using tf.xla.experimental.compile
def my_function(x):
    return tf.matmul(x, x)
result = tf.xla.experimental.compile(my_function, [input_tensor])
```

### Compilation Flow in TensorFlow

```
1. tf.function(jit_compile=True) or auto-cluster identified
   |
2. TensorFlow captures the concrete function
   |
3. Grappler runs (graph optimizations)
   |
4. XLA cluster extracted
   |
5. TF ops -> HLO instructions (MLIR-based lowering)
   |
6. HLO optimization pipeline
   |
7. Backend compilation (CPU/GPU/TPU)
   |
8. XLA Executable cached
   |
9. Execution on subsequent calls (cache hit)
```

---

## Performance Benefits

### Fusion Benefits

1. **Reduced Memory Bandwidth**: Intermediate results stay in registers/cache
2. **Fewer Kernel Launches**: One fused kernel replaces many small kernels
3. **Better Instruction-Level Parallelism**: Compiler can schedule operations

### Example: MatMul + BiasAdd + ReLU

```
Without XLA (3 kernels):
  1. MatMul: load A, B -> write C
  2. BiasAdd: load C, bias -> write D
  3. ReLU: load D -> write E
  Total: 4 reads + 3 writes = 7 memory ops

With XLA (1 fused kernel):
  1. FusedMatMulBiasAddRelu: load A, B, bias -> write E
  Total: 3 reads + 1 write = 4 memory ops

Speedup: ~1.5-3x for this pattern
```

### Memory Benefits

1. **Buffer Reuse**: Dead buffers are reused for new allocations
2. **Reduced Peak Memory**: Intermediate results are not materialized
3. **In-Place Operations**: Some operations can be done in-place

---

## XLA Debugging

### Dump Configuration

```bash
# Dump HLO text representation
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"

# Dump HLO proto representation
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_proto"

# Dump both
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text --xla_dump_hlo_as_proto"

# Dump all passes
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*"
```

### Dump File Naming

```
module_name.before_optimizations.txt
module_name.after_optimizations.txt
module_name.before_fusion.txt
module_name.after_fusion.txt
module_name.before_layout_assignment.txt
module_name.after_layout_assignment.txt
module_name.gpu.asm                    # PTX for GPU
module_name.cpu.o                      # Object file for CPU
```

### Per-Pass Dumping

```bash
# Dump specific optimization passes
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_pass_re=.*fusion.*"
```

### HLO Module Printing

```python
# Get HLO text from a compiled function
@tf.function(jit_compile=True)
def my_fn(x):
    return tf.matmul(x, x)

# Enable XLA dumping before calling
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump'
my_fn(tf.ones([10, 10]))
```

### Debugging Tools

1. **xla.hlo.dump**: Dumps HLO module to files
2. **XLA Inspector**: Examine HLO computations
3. **TensorBoard XLA Profiler**: Visualize XLA execution
4. **HloModule Dumper**: Dump during compilation

### Common Debugging Patterns

```python
# Enable XLA debugging
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_cluster=true'
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla'

# Disable XLA for debugging
tf.config.optimizer.set_jit(False)

# Force XLA compilation
@tf.function(jit_compile=True)
def debug_fn(x):
    # This must compile entirely with XLA
    return x + 1
```

---

## XLA Environment Variables

### Compilation Control

| Variable | Description |
|----------|-------------|
| `XLA_FLAGS` | General XLA flags (dumping, optimization) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of device memory to preallocate (default: 0.75) |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | Preallocate device memory (default: true) |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | Memory allocator type ("default", "platform", "bfc") |
| `XLA_PYTHON_CLIENT_METRIC_sample_rate` | Metric sampling rate |

### Dump Flags

| Flag | Description |
|------|-------------|
| `--xla_dump_to` | Directory for dump output |
| `--xla_dump_hlo_as_text` | Dump HLO as text files |
| `--xla_dump_hlo_as_proto` | Dump HLO as protobuf files |
| `--xla_dump_hlo_pass_re` | Regex for pass names to dump |
| `--xla_dump_hlo_snapshots` | Dump snapshots during compilation |

### Optimization Flags

| Flag | Description |
|------|-------------|
| `--xla_disable_all_hlo_passes` | Disable all HLO optimization passes |
| `--xla_gpu_enable_triton_gemm` | Enable Triton-based GEMM for GPU |
| `--xla_gpu_autotune_level` | Auto-tuning level for GPU kernels (0-4) |
| `--xla_backend_optimization_level` | Backend optimization level |

### GPU-Specific Flags

| Flag | Description |
|------|-------------|
| `--xla_gpu_cuda_data_dir` | CUDA toolkit directory |
| `--xla_gpu_force_compilation_parallel_to_llvm` | Parallel LLVM compilation |
| `--xla_gpu_max_kernel_unroll_factor` | Max kernel unroll factor |
| `--xla_gpu_enable_pipelined_fusion` | Enable pipelined fusion |

### CPU-Specific Flags

| Flag | Description |
|------|-------------|
| `--xla_cpu_use_thunk_runtime` | Use thunk-based CPU runtime |
| `--xla_llvm_compiler_options` | Additional LLVM compiler options |

---

## XLA Optimization Passes

### Core Optimization Passes

| Pass | Description |
|------|-------------|
| `HloConstantFolding` | Evaluate constant expressions at compile time |
| `AlgebraicSimplifier` | Simplify arithmetic (remove identity, cancel inverses) |
| `HloCSE` | Common subexpression elimination |
| `HloDCE` | Dead code elimination |
| `HloVerifier` | Verify HLO correctness |
| `CallInliner` | Inline called computations |
| `WhileLoopConstantSinking` | Move constants into while loops |
| `WhileLoopSimplifier` | Simplify while loop structures |
| `TupleSimplifier` | Simplify tuple operations |
| `ShapeInference` | Infer shapes through the computation |
| `SortSimplifier` | Simplify sort operations |
| `BroadcastFolding` | Fold broadcasts into preceding ops |
| `ConvolutionFolding` | Fold operations into convolutions |

### Fusion Passes

| Pass | Description |
|------|-------------|
| `GpuInstructionFusion` | GPU-specific fusion patterns |
| `CpuInstructionFusion` | CPU-specific fusion patterns |
| `MultiOutputFusion` | Fuse ops with multiple outputs |
| `FusionMerger` | Merge compatible fusion nodes |
| `FusionBlockize` | Create tiled fusion for large reductions |

### Layout Passes

| Pass | Description |
|------|-------------|
| `LayoutAssignment` | Assign memory layouts to all instructions |
| `GpuLayoutAssignment` | GPU-specific layout assignment |
| `CpuLayoutAssignment` | CPU-specific layout assignment |

### Memory Passes

| Pass | Description |
|------|-------------|
| `HloSchedule` | Schedule instruction execution order |
| `BufferAssignment` | Allocate memory buffers |
| `MemorySpaceAssignment` | Assign buffers to memory spaces |
| `HloReplicationAnalysis` | Analyze replicated values |

### GPU-Specific Passes

| Pass | Description |
|------|-------------|
| `GpuFusionAnalysis` | Analyze fusion candidates |
| `GpuHloSchedule` | GPU-specific scheduling |
| `GpuVectorize` | Vectorize GPU kernels |
| `ReductionDimensionGrouper` | Group reduction dimensions |
| `DotDimensionSorter` | Sort dot operation dimensions |

---

## XLA Compilation Cache

### Cache Behavior

XLA caches compiled executables to avoid recompilation:

1. **Cache Key**: Based on computation hash, argument shapes, and compilation options
2. **Cache Hit**: Returns existing executable without recompilation
3. **Cache Miss**: Compiles new executable and stores in cache

### Cache Configuration

```python
# Increase compilation cache size
tf.config.optimizer.set_experimental_options(
    {'xla_compilation_cache_size': 100}
)
```

### Cache Invalidation

The cache is invalidated when:
- Different computation (different HLO)
- Different argument shapes (dynamic shapes)
- Different compilation options
- Process restart

---

## Window and Convolution Configuration

### WindowDimension

```protobuf
message WindowDimension {
  int64 size = 1;            // Window size
  int64 stride = 2;          // Stride between positions
  int64 padding_low = 3;     // Low padding
  int64 padding_high = 4;    // High padding
  int64 window_dilation = 5; // Window dilation factor
  int64 base_dilation = 6;   // Base dilation factor
  bool window_reversal = 7;  // Window reversal flag
}
```

### ConvolutionDimensionNumbers

```protobuf
message ConvolutionDimensionNumbers {
  int64 input_batch_dimension = 7;
  int64 input_feature_dimension = 8;
  repeated int64 input_spatial_dimensions = 11;
  int64 kernel_input_feature_dimension = 3;
  int64 kernel_output_feature_dimension = 4;
  repeated int64 kernel_spatial_dimensions = 6;
  int64 output_batch_dimension = 9;
  int64 output_feature_dimension = 10;
  repeated int64 output_spatial_dimensions = 12;
}
```

### DotDimensionNumbers

```protobuf
message DotDimensionNumbers {
  repeated int64 lhs_contracting_dimensions = 1;
  repeated int64 rhs_contracting_dimensions = 2;
  repeated int64 lhs_batch_dimensions = 3;
  repeated int64 rhs_batch_dimensions = 4;
}
```

---

## Channel and Device Assignment

### ChannelHandle

```protobuf
message ChannelHandle {
  int64 handle = 1;
  enum ChannelType {
    CHANNEL_TYPE_INVALID = 0;
    DEVICE_TO_DEVICE = 1;
    DEVICE_TO_HOST = 2;
    HOST_TO_DEVICE = 3;
  }
  ChannelType type = 2;
}
```

### DeviceAssignment

```protobuf
message DeviceAssignmentProto {
  int32 replica_count = 1;
  int32 computation_count = 2;
  message ComputationDevice {
    repeated int64 replica_device_ids = 1;
  }
  repeated ComputationDevice computation_devices = 3;
}
```

---

## OpMetadata

```protobuf
message OpMetadata {
  string op_type = 1;            // Framework op name
  string op_name = 2;            // User-specified name
  string source_file = 3;        // Source file
  int32 source_line = 4;         // Source line number
  int64 size_of_generated_code_in_bytes = 8;
  int64 size_of_memory_working_set_in_bytes = 9;
}
```

This metadata is attached to HLO instructions for profiling and debugging.

---

## FrontendAttributes

```protobuf
message FrontendAttributes {
  map<string, string> map = 1;
}
```

Generic key-value attributes passed from frontend to XLA backend.
Used for hints like:
- `xla_framework`: Name of the frontend framework
- `xla_allow_restructuring`: Allow graph restructuring
- `use_uniform_channel`: Use uniform communication channels
