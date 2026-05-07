# XLA Compiler

This document provides a comprehensive reference for the XLA (Accelerated Linear Algebra)
compiler infrastructure in TensorFlow. XLA is a domain-specific compiler for linear algebra
that optimizes TensorFlow computations through a series of HLO-level analysis and
transformation passes, followed by backend-specific code generation.

## Table of Contents

1. [Compiler Interface](#compiler-interface)
2. [Backend Architecture](#backend-architecture)
3. [CPU Backend](#cpu-backend)
4. [GPU Backend](#gpu-backend)
5. [TPU Backend](#tpu-backend)
6. [Optimization Passes](#optimization-passes)
7. [Fusion Strategies](#fusion-strategies)
8. [Buffer Assignment](#buffer-assignment)
9. [Layout Assignment](#layout-assignment)
10. [Scheduling](#scheduling)
11. [Cost Model](#cost-model)
12. [HloDCE](#flodce)
13. [HloVerifier](#hloverifier)
14. [TuplePointsToAnalysis](#tuplepointstoanalysis)

---

## Compiler Interface

The XLA compiler provides an abstract interface (`xla::Compiler`) that is subclassed for
each supported hardware platform. The compiler ties together high-level optimization (HLO)
and low-level optimization/code generation to produce efficient executables.

### Compiler Class Hierarchy

```
xla::Compiler (abstract base)
  |-- xla::cpu::CpuCompiler
  |-- xla::gpu::GpuCompiler
  |-- xla::tpu::TpuCompiler
```

### Key Methods

#### `Compiler::Compile`

Compiles an HLO module into an executable. This is the primary entry point for
just-in-time compilation.

```cpp
// Defined in: xla/service/compiler.h
class Compiler {
 public:
  // Compiles an HLO module into an executable.
  // Parameters:
  //   - module: The HLO module to compile
  //   - options: Compilation options including device allocator, thread pool, etc.
  // Returns: StatusOr<std::unique_ptr<Executable>>
  virtual absl::StatusOr<std::unique_ptr<Executable>> Compile(
      std::unique_ptr<HloModule> module,
      const CompileOptions& options) = 0;
};
```

The `CompileOptions` struct contains:

| Field | Type | Description |
|-------|------|-------------|
| `device_allocator` | `se::DeviceAddressAllocator*` | Optional allocator for temp device memory during compilation |
| `thread_pool` | `tsl::thread::ThreadPool*` | Optional thread pool for parallel compilation |
| `layout_canonicalization_callback` | `function` | Optional callback for layout canonicalization |
| `gpu_topology` | `optional<GpuTopology>` | GPU topology information |
| `cpu_target_config` | `optional<CpuTargetConfig>` | CPU target machine options |
| `slice_size` | `int64_t` | Number of devices in a fast-interconnect domain |

#### `Compiler::RunBackend`

Runs the backend-specific compilation pipeline on an already-optimized HLO module.
This is separated from the full `Compile` flow to allow reuse of optimization passes.

```cpp
virtual absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) = 0;
```

#### `Compiler::CompileXla`

Compiles an XLA computation directly, typically used for ahead-of-time compilation.

```cpp
virtual absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
CompileXla(const XlaComputation& computation,
           const absl::Span<const Shape> argument_layouts,
           const ExecutableBuildOptions& options) = 0;
```

### Compiler Registry

Compilers are registered through a factory mechanism:

```cpp
// Register a compiler factory for a platform
static void RegisterCompilerFactory(
    se::Platform::Id platform_id,
    std::function<std::unique_ptr<Compiler>()> factory);

// Get the compiler for a platform
static Compiler* GetForPlatform(se::Platform* platform);
```

The compiler singletons are registered via module initializers in their corresponding
XLA compiler libraries. Multiple XLA clients may request compilation concurrently,
so compiler subclasses must be thread-safe.

### Backend Class

The `Backend` class (defined in `xla/service/backend.h`) encapsulates everything
necessary to compile and execute computations on a particular platform:

```cpp
class Backend {
 public:
  static absl::StatusOr<std::unique_ptr<Backend>> CreateBackend(
      const BackendOptions& options);
  static absl::StatusOr<std::unique_ptr<Backend>> CreateDefaultBackend();

  // Accessors
  se::Platform* platform() const;
  Compiler* compiler() const;
  se::DeviceAddressAllocator* memory_allocator() const;
  TransferManager* transfer_manager() const;
  ComputationPlacer* computation_placer() const;

  // Stream management
  absl::StatusOr<StreamPool::Ptr> BorrowStream(int device_ordinal);
  int device_count() const;
  bool device_ordinal_supported(int device_ordinal) const;

  // Device equivalence check
  absl::StatusOr<bool> devices_equivalent(int a, int b) const;
};
```

`BackendOptions` configures the backend:

| Field | Default | Description |
|-------|---------|-------------|
| `platform` | `nullptr` (default platform) | The platform backing the backend |
| `intra_op_parallelism_threads` | `-1` (num cores) | Thread pool size for parallel operator execution |
| `allowed_devices` | `nullopt` | Selectively construct stream executors for these devices |

---

## Backend Architecture

The XLA compilation pipeline follows a layered architecture:

```
TensorFlow Graph
      |
      v
  TF -> HLO Conversion (tf2xla)
      |
      v
  HLO Module
      |
      v
  +---+---+
  | HLO   |  Frontend Optimization Passes
  | Pass  |  (platform-independent)
  | Pipeline|
  +---+---+
      |
      v
  +---+---+
  | Backend|  Backend-Specific Passes
  | Passes |  (layout assignment, fusion, lowering)
  +---+---+
      |
      v
  +---+---+
  | Code   |  Code Generation
  | Gen    |  (LLVM IR -> object code, PTX, etc.)
  +---+---+
      |
      v
  Executable
```

---

## CPU Backend

The CPU backend (`xla::cpu::CpuCompiler`) generates optimized native code for x86-64
and AArch64 processors using LLVM.

### LLVM IR Generation

The CPU backend translates HLO instructions to LLVM IR through the following pipeline:

1. **HLO -> LLVM IR Translation**: Each HLO operation is mapped to LLVM IR instructions.
   Element-wise operations become LLVM vector operations. Reductions become loops with
   accumulator operations.

2. **IR Builder**: The `IrArray` class wraps LLVM values representing HLO array-shaped
   instructions, providing methods to emit LLVM IR for element access.

3. **Vectorization**: The CPU backend targets SIMD vectorization using LLVM's vector
   types. For example, an element-wise add on an array of 1024 floats might be compiled
   to 256 x 4-wide SIMD add instructions on a processor with 128-bit SSE registers.

### CPU Instruction Selection

Instruction selection is handled by LLVM's target-specific instruction selectors:

- **x86-64**: Uses SSE/AVX/AVX2/AVX-512 instructions for vectorized operations
- **AArch64**: Uses NEON instructions for vectorized operations

The target features are controlled via `TargetMachineOptions`:

```cpp
struct TargetMachineOptions {
  std::string target_triple;     // e.g., "x86_64-unknown-linux-gnu"
  std::string target_cpu;        // e.g., "haswell", "skylake-avx512"
  std::string target_features;   // e.g., "+avx2,+fma"
  bool enable_fast_math = false;
};
```

### Layout Assignment for CPU

The CPU backend assigns layouts that favor the natural memory access patterns of the
target architecture. By default, it prefers row-major (dim 0 major) layouts. The layout
assignment pass (see `LayoutAssignment` section) propagates constraints from instructions
that have layout preferences (like `kConvolution`, `kDot`) and resolves conflicts.

### Vectorization Strategies

The CPU backend employs several vectorization strategies:

1. **SLP Vectorization**: Superword-Level Parallelism auto-vectorization provided by LLVM
2. **Loop Vectorization**: LLVM's loop vectorizer for reduction and map patterns
3. **Explicit SIMD**: Some HLO operations explicitly emit SIMD LLVM IR

### Key Source Files

| File | Description |
|------|-------------|
| `xla/service/cpu/cpu_compiler.h` | CPU compiler class |
| `xla/service/cpu/cpu_instruction_selection.h` | Instruction selection |
| `xla/service/cpu/llvm_ir_gen.h` | LLVM IR generation |
| `xla/service/cpu/target_machine_features.h` | Target machine feature detection |

---

## GPU Backend

The GPU backend (`xla::gpu::GpuCompiler`) generates optimized GPU kernels, primarily
targeting NVIDIA GPUs via PTX/SASS and AMD GPUs via ROCm.

### GPU Emitter

The GPU emitter translates HLO fusion nodes into GPU kernels. Each fusion node is
emitted as a single GPU kernel:

```
Fusion HLO Node
      |
      v
  Kernel Thunk
      |
      v
  GPU Kernel (PTX/SASS)
```

The emitter handles:
- Thread/block dimension computation based on output shape
- Shared memory management for reductions
- Register allocation for intermediate values
- Cooperative kernel launches for large reductions

### Kernel Generation

Kernel generation follows this process:

1. **Thunk Construction**: Each HLO instruction maps to a "thunk" (a unit of GPU work)
2. **Kernel Emission**: Fusion thunks are emitted as GPU kernels
3. **Sequential Thunk Ordering**: Thunks are ordered respecting data dependencies
4. **Buffer Arguments**: Kernel arguments are bound to allocated buffers

### NCCL Integration

The GPU backend integrates NCCL (NVIDIA Collective Communications Library) for
collective operations:

| HLO Op | NCCL Operation |
|--------|---------------|
| `kAllReduce` | `ncclAllReduce` |
| `kAllGather` | `ncclAllGather` |
| `kReduceScatter` | `ncclReduceScatter` |
| `kAllToAll` | `ncclAllToAll` |
| `kCollectivePermute` | `ncclSend/Recv` |

NCCL integration requires:
- Matching data types (FP32, FP16, BF16, INT32, INT64)
- Matching reduction operations (Sum, Min, Max, Product)
- Compatible replica groups

### Thunks

Thunks are the GPU backend's abstraction for executable units of work. The thunk
hierarchy:

```
Thunk (base)
  |-- KernelThunk          - Launches a GPU kernel
  |-- CopyThunk            - Device-to-device memory copy
  |-- NcclCollectiveThunk  - NCCL collective operation
  |-- ConditionalThunk     - Conditional execution
  |-- WhileThunk           - While loop execution
  |-- CustomCallThunk      - Custom call to user-defined function
  |-- CommandBufferThunk   - Encapsulates multiple thunks in a command buffer
```

### Stream Assignment

Stream assignment determines which GPU streams execute which operations. The goal is
to overlap independent operations for better GPU utilization:

```
Stream 0: [MatMul1] ----> [MatMul2] ---->
Stream 1:    [Reduce1] ----> [Reduce2] ---->
```

The `StreamAssignment` pass analyzes data dependencies and assigns operations to
streams such that:
- Dependent operations execute on the same stream (or are synchronized)
- Independent operations execute on different streams
- The number of streams is bounded by the hardware limit

### Buffer Assignment for GPU

Buffer assignment for GPU includes special handling for:
- **Fragment buffers**: Temporary buffers within a kernel
- **Shared memory**: Fast on-chip memory for reductions
- **Prefetch buffers**: Host-pinned memory for async transfers

---

## TPU Backend

The TPU (Tensor Processing Unit) backend compiles HLO for execution on Google's TPU
hardware.

### TPU Compilation

The TPU compilation pipeline:

1. **HLO Optimization**: Standard HLO optimization passes plus TPU-specific passes
2. **HLO -> TPU Instruction Conversion**: HLO operations are converted to TPU-specific
   instructions
3. **TPU Program Generation**: Instructions are assembled into a TPU program

### HLO to TPU Instruction Conversion

Key HLO operations map to TPU instructions as follows:

| HLO Operation | TPU Instruction |
|--------------|-----------------|
| `kDot` | Matrix multiply unit instruction |
| `kConvolution` | Convolution unit instruction |
| `kReduce` | Scalar reduction instruction |
| `kSort` | Sort network instruction |
| `kTranspose` | Memory layout transformation |

### TPU-Specific Optimizations

- **Layout optimization**: TPU prefers specific data layouts for matrix operations
- **Memory space assignment**: Uses on-chip SRAM (VMEM) vs HBM based on access patterns
- **Cross-replica optimization**: Leverages high-speed inter-core interconnects

---

## Optimization Passes

XLA applies a series of optimization passes to the HLO module before code generation.
These passes are organized into a pipeline that runs sequentially.

### Optimization Pass Pipeline Overview

```
HLO Module Input
      |
      v
  CallInliner
      |
      v
  AlgebraicSimplifier
      |
      v
  BitcastDtypesIntervalSimplifier
      |
      v
  HloConstantFolding
      |
      v
  HloCSE (Common Subexpression Elimination)
      |
      v
  HloDCE (Dead Code Elimination)
      |
      v
  InstructionFusion
      |
      v
  LayoutAssignment
      |
      v
  LayoutSimplifier
      |
      v
  TransposeFolding
      |
      v
  ReshapeMover
      |
      v
  WhileLoopConstantSinking
      |
      v
  WhileLoopSimplifier
      |
      v
  WhileLoopTripCountSimplifier
      |
      v
  TupleSimplifier
      |
      v
  PadSimplifier / SliceSimplifier / SortSimplifier / ScatterExpander
      |
      v
  (Backend-specific passes)
      |
      v
  HloVerifier (verification)
```

### AlgebraicSimplifier

The `AlgebraicSimplifier` pass simplifies algebraic expressions in the HLO graph.
It performs pattern matching on HLO operations and replaces them with simpler equivalents.

**Key simplifications:**

| Pattern | Replacement |
|---------|-------------|
| `add(x, 0)` | `x` |
| `multiply(x, 1)` | `x` |
| `multiply(x, 0)` | `broadcast(0)` |
| `subtract(x, x)` | `broadcast(0)` |
| `divide(x, 1)` | `x` |
| `power(x, 1)` | `x` |
| `power(x, 0)` | `broadcast(1)` |
| `max(x, x)` | `x` |
| `min(x, x)` | `x` |
| `transpose(x, {0,1,...,n-1})` | `x` (identity transpose) |
| `reshape(x, same_shape)` | `x` |
| `broadcast(x, no_dims)` | `x` |

Additional simplifications include:
- Collapsing consecutive reshapes: `reshape(reshape(x, s1), s2)` -> `reshape(x, s3)`
- Collapsing consecutive transposes
- Simplifying broadcast-into-binary operations
- Eliminating trivial slices and pads

### BitcastDtypesIntervalSimplifier

Simplifies bitcast operations that convert between compatible dtypes where the data
representation is preserved. This handles cases where bitcasting between BF16 and F16
or between different integer types can be eliminated.

### CallInliner

The `CallInliner` pass inlines all `kCall` instructions by replacing the call with
the body of the called computation. This enables subsequent optimization passes to
see the full computation graph.

```cpp
// Before:
%call = call(%param), to_apply=%called_computation

// After (inlined):
%result = <body of called_computation with %param substituted>
```

### CSE (Common Subexpression Elimination)

The `HloCSE` pass identifies and eliminates common subexpressions. Two instructions
are considered equivalent if:
- They have the same opcode
- They have the same operands (by identity)
- They have the same attributes (shape, dimensions, etc.)

```cpp
// Before CSE:
%add1 = add(%x, %y)
%add2 = add(%x, %y)   // Same as %add1
%result = multiply(%add1, %add2)

// After CSE:
%add1 = add(%x, %y)
%result = multiply(%add1, %add1)   // %add2 replaced with %add1
```

The pass uses `HloCseConstantKey` for hashing and comparing constant instructions,
ensuring that constants with the same literal values are deduplicated.

### DCE (Dead Code Elimination)

The `HloDCE` pass removes instructions whose results are not used by any other
instruction (except the root instruction of the computation). It traverses the
HLO graph in reverse topological order and removes dead instructions.

```cpp
// Before DCE:
%add = add(%x, %y)
%mul = multiply(%x, %y)   // Dead - not used
%result = %add

// After DCE:
%add = add(%x, %y)
%result = %add
```

### Defuse

The `Defuse` pass (de-fusion) breaks apart fusion nodes that may have been created
suboptimally or that need to be split for backend-specific reasons. This enables
re-fusion with better strategies.

### FlattenerInstructionSimplifier

Simplifies instructions that operate on flattened representations, such as collapsing
nested tuple structures and simplifying `GetTupleElement` chains.

### Fusion (InstructionFusion)

The `InstructionFusion` pass is one of the most critical optimizations in XLA. It
combines multiple HLO instructions into a single "fusion" instruction that is compiled
into a single kernel (on GPU) or a single loop nest (on CPU).

The fusion decision is governed by the `FusionDecision` class:

```cpp
class FusionDecision {
 public:
  static FusionDecision Allow();
  static FusionDecision Forbid(absl::string_view explanation);
  bool CanFuse() const;
  FusionDecision Or(const FusionDecision& other) const;
  FusionDecision And(const FusionDecision& other) const;
};
```

**Fusion criteria:**
- The producer instruction has exactly one user
- The fusion would not create excessive code expansion
- The resulting fusion node is within size limits
- The instructions are compatible (same element type, compatible shapes)

See the [Fusion Strategies](#fusion-strategies) section for detailed fusion types.

### HloConstantFolding

The `HloConstantFolding` pass evaluates operations on constant operands at compile
time and replaces the results with new constant instructions:

```cpp
// Before:
%x = constant({1.0, 2.0, 3.0})
%y = constant({4.0, 5.0, 6.0})
%add = add(%x, %y)

// After:
%add = constant({5.0, 7.0, 9.0})
```

This is especially valuable for:
- Folding shape operations during compilation
- Pre-computing constant expressions in models with static weights
- Simplifying broadcast and reshape operations on constants

### LayoutAssignment

See the [Layout Assignment](#layout-assignment) section below for details.

### LayoutSimplifier

After layout assignment, this pass simplifies unnecessary copies and layout
transformations. It eliminates redundant `kCopy` instructions that were inserted
by layout assignment when the source and destination layouts are identical.

### LiteralSimplifier

The `LiteralSimplifier` pass simplifies operations on literal (constant) values.
It evaluates operations at compile time when all operands are constants, similar to
constant folding but with a focus on literal-level transformations.

### PadSimplifier

The `PadSimplifier` pass removes trivial padding operations:
- Padding with zeros on a `kAdd` is simplified to just the operand
- Identity padding (no actual padding applied) is removed entirely
- Negative padding (which can be represented as slicing) is converted

### ReshapeMover

The `ReshapeMover` pass moves reshape operations past element-wise operations when
it is safe to do so. This can enable better fusion opportunities:

```cpp
// Before:
%reshape = reshape(%x), shape=[N,M]
%add = add(%reshape, %y)

// After (if safe):
%add_inner = add(%x, %y_reshaped)
%reshape = reshape(%add_inner), shape=[N,M]
```

### ScatterExpander

The `ScatterExpander` pass decomposes scatter operations into simpler primitives
when the backend does not natively support scatter or when the decomposition is
beneficial:

```cpp
// Scatter -> (Loop of DynamicUpdateSlice)
```

### SliceSimplifier

Simplifies slice operations:
- Removes identity slices (slice that returns the entire input)
- Collapses consecutive slices into a single slice
- Converts trivial slices to reshapes or bitcasts

### SortSimplifier

Simplifies sort operations:
- Removes sorts on single-element or empty tensors
- Simplifies sorts where the input is already sorted (detected via analysis)
- Converts sorts with trivial comparators

### TransposeFolding

The `TransposeFolding` pass folds transpose operations into adjacent dot or
convolution operations. On GPU, this is critical because GEMM operations can
handle transposed inputs natively:

```cpp
// Before:
%transpose = transpose(%x), dims={1,0}
%dot = dot(%transpose, %y)

// After (transposed input folded into dot):
%dot = dot(%x, %y), lhs_transpose={1,0}
```

This eliminates a separate transpose kernel and reduces memory bandwidth usage.

### TupleSimplifier

Simplifies tuple operations:
- Eliminates `Tuple(GetTupleElement(x, i))` patterns
- Simplifies chains of `GetTupleElement` operations
- Removes unused tuple elements

### WhileLoopConstantSinking

The `WhileLoopConstantSimplifier` pass sinks constants into while loop bodies,
eliminating the need to pass constants through loop iterations:

```cpp
// Before:
%const = constant(42)
while_body(%iter_var, %const) { ... }

// After:
while_body(%iter_var) {
  %const = constant(42)
  ...
}
```

### WhileLoopSimplifier

Simplifies while loops by:
- Removing unused loop-carried values
- Converting while loops with known trip counts to bounded loops
- Simplifying loop conditions that are always true or always false
- Detecting and eliminating redundant induction variables

### WhileLoopTripCountSimplifier

Analyzes while loops to determine static trip counts when possible and uses this
information for optimization. If the trip count is known:
- The loop can be unrolled (up to a limit)
- Buffer liveness analysis becomes more precise
- Memory planning can be more aggressive

---

## Fusion Strategies

XLA supports multiple fusion strategies that determine how HLO instructions are grouped
into fusion nodes.

### Fusion Types

| Fusion Kind | Description | Best For |
|------------|-------------|----------|
| `kLoop` | Loop fusion - all ops execute with same element loop | Element-wise chains |
| `kInput` | Input fusion - reduce into a fusion producer | Reduction + element-wise |
| `kOutput` | Output fusion - scatter into a fusion consumer | Scatter patterns |
| `kCustom` | Custom fusion defined by backend | Backend-specific patterns |

### kLoop Fusion

Loop fusion combines element-wise operations into a single loop. All operations
execute with the same loop structure:

```cpp
// Before:
%a = add(%x, %y)
%b = multiply(%a, %z)
%c = tanh(%b)

// After (kLoop fusion):
%fused = fusion(%x, %y, %z), kind=kLoop {
  %a = add(%x, %y)
  %b = multiply(%a, %z)
  %c = tanh(%b)
  ROOT %c
}
```

**Advantages:**
- Minimizes memory bandwidth by keeping intermediate results in registers
- Simple code generation
- Works well for element-wise chains

**Limitations:**
- Cannot fuse operations with different iteration spaces
- Limited benefit when the fusion becomes too large (register pressure)

### kInput Fusion

Input fusion combines a reduction with its producer operations. The reduction is the
root of the fusion, and all producers feed into it:

```cpp
// Before:
%a = multiply(%x, %y)
%b = reduce(%a), dimensions={0}

// After (kInput fusion):
%fused = fusion(%x, %y), kind=kInput {
  %a = multiply(%x, %y)
  %b = reduce(%a), dimensions={0}
  ROOT %b
}
```

**Advantages:**
- Eliminates intermediate buffer for the reduction input
- Enables tiling and shared memory optimization on GPU

### kOutput Fusion

Output fusion combines scatter operations with their consumers:

```cpp
// Before:
%scatter = scatter(%buffer, %indices, %updates)
%result = add(%scatter, %bias)

// After (kOutput fusion):
%fused = fusion(%buffer, %indices, %updates, %bias), kind=kOutput {
  %scatter = scatter(%buffer, %indices, %updates)
  %result = add(%scatter, %bias)
  ROOT %result
}
```

### kCustom Fusion

Custom fusion strategies are defined by the backend. Examples include:
- **Multi-output fusion**: A single fusion node produces multiple outputs
- **Transpose+GEMM fusion**: Fusing transpose into matrix multiplication
- **Reduction+Transpose fusion**: Common pattern in transformer models

### Fusion Decision Framework

The `FusionDecision` class provides a composable framework for making fusion decisions:

```cpp
// Composing fusion decisions
FusionDecision shouldFuse = IsProfitable(instruction)
    .And(HasSingleUser(producer))
    .And(WithinSizeLimit(fusion_node));
```

The `FusionNodeIndexingEvaluation` class evaluates the cost of indexing into a fusion
node, which affects whether fusion is beneficial:

```cpp
// High indexing cost = less benefit from fusion
int64_t indexing_cost = FusionNodeIndexingEvaluation::ComputeIndexingCost(
    fusion, operand);
```

### Fusion Queue

The `FusionQueue` class manages the order in which instructions are considered for
fusion. Different backends can customize the queue ordering:

- **Bottom-up**: Start from outputs and fuse producers (default for GPU)
- **Top-down**: Start from inputs and fuse consumers
- **Greedy**: Greedily fuse the most beneficial pairs first

---

## Buffer Assignment

Buffer assignment determines how logical buffers (HLO values) map to physical memory
allocations. This is one of the most critical phases of XLA compilation.

### Overview

The `BufferAssignment` class maps each `HloValue` (a value produced by an HLO
instruction at a particular shape index) to a `BufferAllocation::Slice` (a contiguous
range of bytes within a `BufferAllocation`).

```
HloValue #1 ----> BufferAllocation #0, offset=0, size=1024
HloValue #2 ----> BufferAllocation #0, offset=1024, size=512  (reuses #1's allocation)
HloValue #3 ----> BufferAllocation #1, offset=0, size=2048
```

### BufferAllocation

The `BufferAllocation` class represents a contiguous block of memory:

```cpp
class BufferAllocation {
 public:
  using Index = int64_t;

  Index index() const;             // Unique allocation index
  int64_t size() const;            // Size in bytes
  LogicalBuffer::Color color();    // Memory space color
  bool is_thread_local() const;    // Thread-local allocation
  bool is_tuple() const;           // Holds tuple data
  bool is_entry_computation_parameter() const;
  bool maybe_live_out() const;     // May be live out of computation
  bool is_constant() const;        // Constant data
  bool is_reusable() const;        // Can hold multiple buffers

  // Slice represents a contiguous portion
  class Slice {
   public:
    const BufferAllocation* allocation() const;
    int64_t offset() const;
    int64_t size() const;
    bool OverlapsWith(const Slice& other) const;
    bool Contains(const Slice& other) const;
  };
};
```

### Buffer Assigner

The `BufferAssigner` class constructs a `BufferAssignment`:

```cpp
class BufferAssigner {
 public:
  struct Options {
    bool allocate_buffers_for_constants = false;
    Colorer colorer = DefaultColorer();
    std::optional<MustNotLiveOut> must_not_live_out;
    BufferOrder buffer_order = BufferOrder::kBiggestFirst;
    // ...
  };

  static absl::StatusOr<std::unique_ptr<BufferAssignment>> Run(
      const HloModule* module,
      std::unique_ptr<HloOrdering> hlo_ordering,
      BufferValue::SizeFunction buffer_size,
      const AliasInfo* alias_info,
      LogicalBuffer::AlignmentFunction color_alignment,
      Options options);
};
```

### Prescoring

Prescoring assigns preliminary buffer assignments based on analysis of the HLO graph
before full heap simulation. Prescored assignments are typically used for:
- Entry computation parameters (fixed memory locations)
- Constants (placed in read-only memory)
- Values with preset memory space assignments

### Coloring

Buffer coloring assigns a "color" to each buffer, representing which memory space
it belongs to. This is used for:
- **Default color (0)**: Main memory (HBM for GPU, RAM for CPU)
- **Alternate colors**: On-chip memory (VMEM for TPU), shared memory (GPU)

```cpp
// Default colorer assigns colors based on shape layout memory space
static Colorer DefaultColorer() {
  return [](HloAliasAnalysis* alias_analysis, const HloOrdering&) {
    for (HloValue* value : alias_analysis->dataflow_analysis().values()) {
      const HloPosition& pos = value->defining_position();
      if (pos.shape().has_layout()) {
        value->set_color(BufferValue::Color(
            pos.shape().layout().memory_space()));
      } else {
        value->set_color(BufferValue::Color(0));
      }
    }
    return absl::OkStatus();
  };
}
```

### Heap Simulation

For temporary buffers, XLA uses heap simulation to pack buffers efficiently:

1. **GlobalDecreasingSizeBestFitHeap**: Allocates buffers using a best-fit strategy,
   ordering buffers by decreasing size. This is the default algorithm.

2. **Whole-module heap simulation**: Runs heap simulation across all computations
   in the module, enabling cross-computation buffer reuse.

The heap simulator produces `HeapSimulatorTrace` objects that record the sequence of
allocations and deallocations:

```
HeapSimulatorTrace:
  Alloc: buffer=#1, size=1024
  Alloc: buffer=#2, size=512
  Free:  buffer=#1
  Alloc: buffer=#3, size=1024  (reuses #1's space)
  Free:  buffer=#2
  Free:  buffer=#3
```

### Buffer Assignment Statistics

The `BufferAssignment::Stats` struct tracks allocation statistics:

```cpp
struct Stats {
  int64_t parameter_allocation_count;
  int64_t parameter_allocation_bytes;
  int64_t constant_allocation_count;
  int64_t constant_allocation_bytes;
  int64_t maybe_live_out_allocation_count;
  int64_t maybe_live_out_allocation_bytes;
  int64_t preallocated_temp_allocation_count;
  int64_t preallocated_temp_allocation_bytes;
  int64_t preallocated_temp_fragmentation_bytes;
  int64_t total_allocation_count;
  int64_t total_allocation_bytes;
};
```

### Computation Classification

Computations are classified into two categories for buffer assignment:

```cpp
absl::Status GatherComputationsByAllocationType(
    const HloModule* module,
    std::vector<const HloComputation*>* thread_local_computations,
    std::vector<const HloComputation*>* global_computations);
```

- **Thread-local**: Computations called in parallel contexts (map, reduce) that need
  thread-local allocations using `alloca`.
- **Global**: Computations with sequentially ordered instructions that use global
  heap allocation.

---

## Layout Assignment

Layout assignment determines the in-memory layout (dimension ordering) of tensors
in the HLO module. This is critical for performance because different layouts have
different memory access patterns.

### Layout Constraints

The `LayoutAssignment` pass is an `HloModulePass` that assigns layouts to all
instructions. It operates through a constraint-based system:

```
LayoutConstraint (abstract base)
  |-- BufferLayoutConstraint    - constrains a LogicalBuffer's layout
  |-- OperandLayoutConstraint   - constrains an instruction's operand layout
  |-- ComputationLayoutConstraint - constrains computation I/O layout
```

Each constraint has properties:
- **Mandatory**: Cannot be overridden
- **DFS propagation**: Propagates in depth-first order
- **Priority**: Higher priority constraints win conflicts

```cpp
class LayoutConstraint {
 public:
  bool mandatory() const;     // Cannot be overridden
  bool dfs() const;           // DFS vs BFS propagation
  int64_t priority() const;   // Higher wins conflicts

  static constexpr int64_t kDefaultPriority = -2;
  static constexpr int64_t kBeginningPriority = 0;
  static constexpr int64_t kGivenPriority = 3;
};
```

### Layout Assignment Process

The layout assignment process follows these steps:

1. **Initialize**: Create `LayoutConstraints` for each computation
2. **Add mandatory constraints**: Required for correctness
3. **Add backend constraints**: Platform-specific layout preferences
4. **Propagate constraints**: Spread layout requirements through the graph
5. **Resolve conflicts**: Higher-priority constraints win
6. **Assign layouts**: Set final layouts on all instructions
7. **Insert copies**: Add `kCopy` instructions where layouts don't match

### Tentative Layout

During propagation, layouts are tentatively assigned. The propagation runs in rounds
(default: 2 rounds, controlled by `kNumberOfPropagationRounds`):

```cpp
static constexpr int64_t kNumberOfPropagationRounds = 2;
```

In each round:
1. Propagate buffer constraints to operands and users
2. Propagate operand constraints to the instruction's output
3. Propagate computation result/parameter constraints
4. Resolve conflicts using priority

### Final Layout

After all rounds, final layouts are assigned. If an instruction's required layout
doesn't match its assigned layout, a `kCopy` instruction is inserted:

```cpp
// If instruction %x has layout {1,0} but %y needs layout {0,1}:
%x = parameter(0), layout={1,0}
%copy = copy(%x), layout={0,1}  // inserted copy
%y = some_op(%copy)
```

### Channel Layout Constraints

For cross-device communication (send/recv), channel layout constraints ensure both
ends use compatible layouts:

```cpp
class ChannelLayoutConstraints {
 public:
  bool IsChannelConstrained(int64_t channel_id) const;
  Shape LayoutShapeForChannel(Shape shape, int64_t channel_id) const;
  const Layout& LayoutForChannel(int64_t channel_id) const;
  const Layout* ConstrainChannel(int64_t channel_id, const Layout& layout);
};
```

### Backend-Specific Layout Hooks

Backends can customize layout assignment by overriding:

```cpp
virtual absl::Status AddBackendConstraints(LayoutConstraints* constraints);
virtual std::unique_ptr<Layout> ChooseOperandLayoutFromOutputLayout(
    const Layout& output_layout, const HloInstruction* instruction,
    int64_t operand_no);
virtual std::unique_ptr<Layout> ChooseOutputLayoutFromOperandLayout(
    const Layout& operand_layout, const HloInstruction* user,
    int64_t operand_no);
virtual bool InstructionCanChangeLayout(const HloInstruction* instruction);
```

---

## Scheduling

Instruction scheduling determines the execution order of HLO instructions.

### Instruction Scheduling

The `InstructionScheduler` assigns a sequential order to HLO instructions within
each computation, respecting data dependencies. The scheduling affects:

- Buffer liveness (which buffers are live simultaneously)
- Memory pressure (peak memory usage)
- Latency hiding (overlap of compute and communication)

### Memory-Aware Scheduling

Memory-aware scheduling optimizes the instruction order to minimize peak memory usage:

```
High memory usage schedule:
  [Alloc A: 1GB] -> [Alloc B: 1GB] -> [Use A] -> [Use B] -> [Free A] -> [Free B]
  Peak: 2GB

Optimized schedule:
  [Alloc A: 1GB] -> [Use A] -> [Free A] -> [Alloc B: 1GB] -> [Use B] -> [Free B]
  Peak: 1GB
```

The `LatencyHidingScheduler` goes further by overlapping communication operations
with computation to hide network latency.

### Schedule Configuration

```cpp
class ScheduleConfig {
 public:
  // Type of scheduling algorithm
  enum Algorithm {
    kDefault,
    kMemoryAware,
    kLatencyHiding,
  };
};
```

---

## Cost Model

### HloCostAnalysis

The `HloCostAnalysis` class estimates the computational cost of HLO operations.
It traverses the HLO graph and accumulates cost metrics.

```cpp
class HloCostAnalysis : public ConstDfsHloVisitor {
 public:
  // Key cost metrics
  static constexpr absl::string_view kFlopsKey = "flops";
  static constexpr absl::string_view kTranscendentalsKey = "transcendentals";
  static constexpr absl::string_view kBytesAccessedKey = "bytes accessed";
  static constexpr absl::string_view kOptimalSecondsKey = "optimal_seconds";
  static constexpr absl::string_view kUtilizationKey = "utilization";
```

### Properties System

The cost analysis uses a `Properties` class that acts like a hash map with
fast-paths for common keys:

```cpp
class Properties {
 public:
  float& operator[](absl::string_view property);
  float operator[](absl::string_view property) const;

  // Fast-path accessors
  float flops();
  float transcendentals();
  float bytes_accessed();
  float optimal_seconds();

  // Per-operand metrics
  float operand_utilization(int64_t operand, const ShapeIndex& index = {});
  float operand_bytes_accessed(int64_t operand, const ShapeIndex& index = {});
  float output_bytes_accessed(const ShapeIndex& index = {});
};
```

### Hardware Options

```cpp
struct Options {
  ShapeSizeFunction shape_size = DefaultShapeSize;
  Properties per_second_rates = {};        // FLOPs/s, bytes/s, etc.
  Properties min_latencies_seconds;        // Minimum operation latencies
  bool count_multiple_input_accesses = false;

  void set_flops_per_second(float value);
  void set_bytes_per_second(float value);
  void set_transcendentals_per_second(float value);
};
```

### Cost Estimation per Operation

| HLO Operation | Cost Model |
|--------------|------------|
| `kDot` | `2 * M * N * K` FLOPs (standard matrix multiply) |
| `kConvolution` | `2 * output_elements * filter_elements_per_position` |
| `kReduce` | `N * (cost_of_reduction_function)` |
| `kElementwise` | `output_elements * (1 or 2)` |
| `kTranspose` | `size_in_bytes` (memory copy cost) |
| `kBroadcast` | `0` (no computation, metadata only) |
| `kCopy` | `size_in_bytes` |

### Per-Instruction Cost Access

```cpp
// Get cost for a specific instruction
int64_t flop_count(const HloInstruction& hlo) const;
int64_t bytes_accessed(const HloInstruction& hlo) const;
float operand_utilization(const HloInstruction& hlo, int64_t operand_num) const;
float optimal_seconds(const HloInstruction& hlo) const;

// Get cost for the entire computation
float flop_count() const;
float bytes_accessed() const;
float optimal_seconds() const;
```

---

## HloDCE

Dead Code Elimination removes HLO instructions whose results are never used.

### Algorithm

1. Mark all instructions reachable from the root instruction (or entry computation
   outputs) as "live"
2. Remove all unmarked instructions
3. Repeat until no more instructions can be removed (fixed point)

### Special Cases

- **While loops**: The condition and body computations must be analyzed recursively
- **Fusion nodes**: Fused instructions within a fusion node are considered as a unit
- **Entry computation parameters**: Always kept alive (they are inputs)
- **Side-effecting instructions**: Always kept (send, outfeed, etc.)

---

## HloVerifier

The `HloVerifier` pass validates HLO module invariants after optimization passes.

### Verification Levels

```cpp
struct HloVerifierOpts {
  bool layout_sensitive = false;          // Check layout compatibility
  bool allow_mixed_precision = false;     // Allow F32/BF16 mixing
  bool verify_broadcast_dimensions_order = false;
  bool verify_reshape_is_bitcast = false;
  bool verify_no_collective_deadlocks = false;
  bool check_replica_groups = true;
  ShapeSizeFn shape_size;
  HloPredicate instruction_can_change_layout;
};
```

### ShapeVerifier

The `ShapeVerifier` verifies that each instruction's output shape matches the
shape inferred from its operands and attributes:

```cpp
class ShapeVerifier : public DfsHloVisitor {
 public:
  // Verifies each HLO instruction type
  absl::Status HandleDot(HloInstruction* dot) override;
  absl::Status HandleConvolution(HloInstruction* convolution) override;
  absl::Status HandleReduce(HloInstruction* reduce) override;
  // ... all other HLO instruction types
};
```

### InstructionVerifier

The `InstructionVerifier` checks non-shape invariants:
- Valid sharding configurations
- Correct parameter counts
- Valid fusion configurations
- Consistent memory space assignments

### TargetVerifierMetadata

Backends can provide custom verification behavior:

```cpp
class TargetVerifierMetadata {
 public:
  virtual std::unique_ptr<ShapeVerifier> GetVerifier() const = 0;
  const HloVerifierOpts& GetVerifierOpts() const;
};
```

---

## TuplePointsToAnalysis

The `TuplePointsToAnalysis` performs points-to analysis for tuple-shaped values in
the HLO module. This analysis is essential for:

- **Buffer aliasing**: Determining when two tuple elements share the same underlying buffer
- **Copy insertion**: Knowing when copies are needed to prevent aliasing
- **Layout assignment**: Understanding data flow through tuple operations

### Points-To Set

For each instruction, the analysis computes a "points-to set" that describes which
logical buffers the instruction's outputs may reference:

```
%tuple = tuple(%a, %b)
%gte0 = get_tuple_element(%tuple), index=0
%gte1 = get_tuple_element(%tuple), index=1

PointsTo(%tuple) = {%a, %b}
PointsTo(%gte0) = {%a}
PointsTo(%gte1) = {%b}
```

### Data Flow Analysis Integration

The points-to analysis integrates with `HloDataflowAnalysis` and `HloAliasAnalysis`
to provide a complete picture of value flow:

```
HloDataflowAnalysis  - traces values through the computation
       |
       v
HloAliasAnalysis     - identifies aliases between values
       |
       v
TuplePointsToAnalysis - resolves tuple-level aliasing
```

### Uses in Compilation

1. **Buffer assignment**: Points-to analysis determines which buffers can share
   allocations
2. **Copy insertion**: Ensures that aliased buffers don't violate single-assignment
   semantics
3. **Layout assignment**: Propagates layout constraints through tuple structures
4. **HloVerifier**: Validates that tuple operations maintain correct aliasing

---

## Compilation Environments

The `CompilationEnvironments` class manages the environment in which compilation
occurs, including debug options, target configurations, and platform-specific settings.

```cpp
class CompilationEnvironments {
 public:
  template <typename T>
  T* GetOrCreate();
  void Set(std::unique_ptr<CompilationEnvironment> env);
};
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `xla/service/compiler.h` | Compiler abstract interface |
| `xla/service/backend.h` | Backend encapsulation |
| `xla/service/buffer_assignment.h` | Buffer assignment |
| `xla/service/layout_assignment.h` | Layout assignment |
| `xla/service/hlo_cost_analysis.h` | Cost analysis |
| `xla/service/hlo_verifier.h` | HLO verification |
| `xla/service/instruction_fusion.h` | Instruction fusion |
| `xla/service/hlo_cse.h` | Common subexpression elimination |
| `xla/service/call_inliner.h` | Call inlining |
| `xla/service/transpose_folding.h` | Transpose folding |
| `xla/service/scatter_expander.h` | Scatter expansion |
| `xla/service/while_loop_simplifier.h` | While loop simplification |
| `xla/service/while_loop_constant_sinking.h` | Constant sinking |
| `xla/service/conditional_simplifier.h` | Conditional simplification |
| `xla/service/all_reduce_simplifier.h` | All-reduce simplification |
