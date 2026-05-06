# XLA Compilation Pipeline Reference

This reference provides comprehensive documentation of the XLA compilation pipeline, from frontend input through target-independent optimization, target-specific optimization, and final code generation. Understanding the pipeline is essential for debugging performance issues, interpreting HLO dumps, and extending XLA with custom operations or optimization passes.

---

## 13.1 Overview

The XLA compilation pipeline transforms a high-level program description into optimized machine code for the target device (GPU, CPU, or TPU). The pipeline consists of several major stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     XLA Compilation Pipeline                        │
│                                                                     │
│  1. Frontend Input (StableHLO / MHLO)                              │
│     └─ JAX / TensorFlow / PyTorch export to StableHLO               │
│                                                                     │
│  2. StableHLO → HLO Conversion                                     │
│     └─ Legalize StableHLO ops to HLO instructions                   │
│                                                                     │
│  3. Target-Independent Optimizations                                │
│     └─ CSE, algebraic simplification, fusion, DCE, etc.             │
│                                                                     │
│  4. Target-Specific Optimizations                                   │
│     └─ Layout assignment, library selection, rematerialization       │
│                                                                     │
│  5. Code Generation                                                 │
│     └─ HLO → LLVM IR → native code (PTX / x86 / TPU ISA)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.1.1 Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Ahead-of-time (AOT) compilation** | XLA compiles the entire computation graph before execution, enabling whole-program optimization |
| **Backend abstraction** | Target-independent optimizations are shared across all backends |
| **Shape-specialized compilation** | Each compilation is for a specific set of input shapes and types (no runtime polymorphism) |
| **Just-in-time (JIT) caching** | Compiled programs are cached by shape signature; recompilation only occurs for new shapes |
| **Deterministic compilation** | Same input + same flags = same output (within a given XLA version) |

### 13.1.2 Entry Points

The compilation pipeline is invoked through several entry points:

| Entry Point | Description |
|-------------|-------------|
| `XlaBuilder::Build()` | Builds an HLO module from XLA builder operations (legacy) |
| `StableHLO -> XLA` | Converts StableHLO to HLO via `stablehlo::createStablehloToHloPass()` |
| `CompileGraphService` | Service-level compilation entry point |
| `PjRtClient::Compile()` | PjRt (JAX runtime) compilation interface |

---

## 13.2 Pipeline Stages in Detail

### 13.2.1 StableHLO to HLO Conversion

StableHLO is the portable, stable intermediate format that frontends (JAX, TensorFlow, PyTorch via torch-xla2) target. The conversion from StableHLO to XLA HLO involves:

#### Conversion Process

```
StableHLO Module (MLIR)
    │
    ├── StableHLO canonicalization passes
    │     └─ Simplify types, fold constants, normalize ops
    │
    ├── Legalize StableHLO → HLO dialect
    │     └─ Map StableHLO ops to HLO ops
    │     └─ Handle type conversions
    │     └─ Resolve naming differences
    │
    ├── HLO dialect → HloModule protobuf/text
    │     └─ Serialize to HloModule structure
    │
    v
XLA HloModule
```

#### Key Conversions

| StableHLO Op | HLO Instruction | Notes |
|-------------|-----------------|-------|
| `stablehlo.add` | `kAdd` | Direct mapping |
| `stablehlo.dot_general` | `kDot` | Dimension number conversion |
| `stablehlo.convolution` | `kConvolution` | Window and dim label conversion |
| `stablehlo.reduce` | `kReduce` | Body computation extraction |
| `stablehlo.while` | `kWhile` | Condition/body computation extraction |
| `stablehlo.dynamic_slice` | `kDynamicSlice` | Direct mapping |
| `stablehlo.scatter` | `kScatter` | Scatter dimension number conversion |
| `stablehlo.rng_bit_generator` | `kRngBitGenerator` | Algorithm enum mapping |
| `stablehlo.custom_call` | `kCustomCall` | Backend config passthrough |
| `stablehlo.cond` | `kConditional` | Branch computation extraction |
| `stablehlo.broadcast_in_dim` | `kBroadcast` | Dimension mapping |
| `stablehlo.transpose` | `kTranspose` | Direct mapping |

#### Shape and Type Handling

- StableHLO uses MLIR types (`tensor<128xf32>`) which are converted to XLA shapes (`f32[128]`).
- Dynamic dimensions in StableHLO (`tensor<?xf32>`) are converted to `SetDimensionSize`/`GetDimensionSize` patterns in HLO.
- Layout information from StableHLO is propagated as layout hints.

---

### 13.2.2 CSE (Common Subexpression Elimination)

`HloCSE` eliminates redundant computations by detecting instructions that compute the same value.

#### Algorithm

1. Hash each instruction based on: opcode, operand IDs, literal values (for constants), and attributes.
2. Build a hash map from instruction hash to instruction.
3. For each instruction, check if an equivalent instruction already exists.
4. Replace the redundant instruction with the existing one.

#### What CSE Detects

| Pattern | Example |
|---------|---------|
| Duplicate constants | Two `constant(42)` instructions merged |
| Duplicate elementwise ops | Two `add(%x, %y)` instructions merged |
| Duplicate reshapes | Two `reshape(%a)` with same shape merged |
| Duplicate broadcasts | Two `broadcast(%s, dims={})` merged |
| Duplicate transposes | Two `transpose(%m, {1,0})` merged |

#### What CSE Does Not Detect

- Semantically equivalent but syntactically different expressions (e.g., `add(x, y)` vs `add(y, x)` for commutative ops -- though the algebraic simplifier may normalize these first).
- Instructions with side effects (infeed, outfeed, send, recv, custom calls with `has_side_effect=true`).

#### Configuration

```cpp
struct CseConfig {
  bool is_layout_sensitive = false;     // Whether to consider layout in equality
  bool only_floating_point_literals = false;  // Whether to CSE only FP constants
};
```

---

### 13.2.3 Fusion Optimization

Fusion is one of the most important optimizations in XLA. It combines multiple HLO instructions into a single "fusion" instruction that will be emitted as a single GPU kernel, eliminating intermediate memory reads and writes.

#### Fusion Strategies

##### Loop Fusion

Loop fusion applies to elementwise operations where each output element depends on exactly one input element at the same position. All fused operations can be computed in a single loop over the output:

```
Before fusion:
  %a = f32[128] parameter(0)
  %b = f32[128] add(%a, %c)        // kernel 1: write %b to memory
  %d = f32[128] multiply(%b, %e)   // kernel 2: read %b from memory

After loop fusion:
  %d = f32[128] fusion(%a, %c, %e), kind=kLoop
  // Single kernel: for each i: d[i] = (a[i] + c[i]) * e[i]
  // %b never materialized in memory
```

##### Input Fusion (Reduce Fusion)

Input fusion attaches elementwise operations as producers to a reduce or reduce-window operation:

```
Before fusion:
  %a = f32[128, 64] parameter(0)
  %b = f32[128, 64] multiply(%a, %scale)
  %c = f32[128] reduce(%b, %zero), dimensions={1}

After input fusion:
  %c = f32[128] fusion(%a, %scale), kind=kInput
  // Single kernel: for each row, compute multiply then reduce
```

##### Output Fusion

Output fusion combines a reduce with its consumers:

```
Before fusion:
  %reduced = f32[128] reduce(%mat, %zero), dimensions={1}
  %result = f32[128] multiply(%reduced, %scale)

After output fusion:
  %result = f32[128] fusion(%mat, %zero, %scale), kind=kOutput
```

##### Multi-Output Fusion

Multi-output fusion produces multiple results from a single kernel:

```
Before fusion:
  %a = f32[128] add(%x, %y)
  %b = f32[128] multiply(%x, %y)

After multi-output fusion:
  %result = (f32[128], f32[128]) fusion(%x, %y), kind=kLoop
  %a = f32[128] get-tuple-element(%result), index=0
  %b = f32[128] get-tuple-element(%result), index=1
```

#### Fusion Criteria

An instruction is fusible if it meets all of the following:

| Criterion | Description |
|-----------|-------------|
| **Elementwise** | The instruction is elementwise (or a scalar constant, broadcast, or bitcast) |
| **No side effects** | The instruction has no side effects |
| **Shape compatible** | All fused instructions have compatible shapes (broadcastable to a common shape) |
| **Not too large** | The fusion does not exceed the backend's fusion limit |
| **Not a fusion sink** | Certain operations (e.g., rng, infeed) are not fusible as producers |

#### Fusion on GPU

On the GPU backend, additional considerations apply:

- **Register pressure**: Fusing too many operations can exhaust GPU registers, causing spills to local memory.
- **Shared memory**: Some fusions (e.g., reduce fusion) use shared memory for partial reductions.
- **Thread efficiency**: The fusion emitter must map the computation to GPU threads efficiently.
- **Launch bounds**: Fused kernels may need `__launch_bounds__` annotations.

The GPU fusion pipeline includes:

1. **InstructionFusion** -- First pass: basic loop and input fusion.
2. **FusionMerger** -- Merge small fusions into larger ones.
3. **GpuInstructionFusion** -- GPU-specific fusion patterns.
4. **FusionBlockize** -- Split large fusions into smaller blocks.

---

### 13.2.4 Buffer Analysis

Buffer analysis determines the memory requirements of the computation. It consists of two main components: buffer liveness analysis and buffer alias analysis.

#### Buffer Liveness Analysis

Buffer liveness determines when each buffer is first written (birth) and last read (death):

```
Instruction         | Buffers Born          | Buffers Dead
────────────────────┼───────────────────────┼─────────────────
%p0 = parameter(0)  | %p0 buffer            |
%a = add(%p0, %c)   | %a buffer             |
%b = multiply(%a,%d)| %b buffer             | %a buffer
%c = reduce(%b,...) | %c buffer             | %b buffer
ROOT %result = ...   |                       | %c buffer, %p0
```

#### Buffer Alias Analysis

Alias analysis determines when two logical buffers can share the same physical memory:

| Alias Type | Description |
|------------|-------------|
| **Parameter-output alias** | Input parameter and output share memory (in-place) |
| **Instruction buffer alias** | Two instructions share the same underlying buffer |
| **Tuple element alias** | Tuple elements alias their source instruction buffers |

The `BufferAliasAnalysis` class computes these aliases:

```cpp
class BufferAliasAnalysis {
  // Returns the set of buffers that may alias the given buffer
  const BufferAliasSet& GetBufferAliases(const Buffer& buffer) const;

  // Returns whether two buffers may alias
  bool MayAlias(const Buffer& a, const Buffer& b) const;
};
```

---

### 13.2.5 Layout Assignment

Layout assignment determines the physical memory layout (dimension ordering and tiling) for each tensor in the computation. Layout choices significantly impact performance because they determine memory access patterns.

#### Layout Representation

In XLA, a layout specifies the order of dimensions from minor (innermost, contiguous in memory) to major (outermost):

```
Shape: f32[128, 64]
Layout {1, 0}: row-major (dim 1 is minor) -- elements in same row are contiguous
  Memory: [row0_col0, row0_col1, ..., row0_col63, row1_col0, ...]

Layout {0, 1}: column-major (dim 0 is minor) -- elements in same column are contiguous
  Memory: [row0_col0, row1_col0, ..., row127_col0, row0_col1, ...]
```

#### Layout Assignment Algorithm

```
1. Assign layouts to parameters (from entry computation layout)
2. Propagate layouts forward through the computation:
   - Elementwise ops: inherit operand layout
   - Dot/Conv: assign optimal layout for the GEMM/conv kernel
   - Reduce: assign layout for efficient reduction
   - Tuple: assign layouts to each element
3. For conflicts (different operands want different layouts):
   - Insert copy instructions to convert between layouts
   - Choose the layout that minimizes total copies
4. Verify all instructions have assigned layouts
```

#### Layout Constraints

| Constraint Type | Description |
|----------------|-------------|
| **Operand layout** | An instruction may require its operands to have specific layouts |
| **Result layout** | The entry computation may specify output layouts |
| **Performance layout** | Backend-specific layouts for optimal performance |
| **Library layout** | Library calls (cuBLAS, cuDNN) require specific layouts |

#### GPU Tiled Layouts

On GPU, XLA supports tiled layouts that group elements into 2D tiles for improved memory coalescing:

```
// Standard layout
f32[256, 256]{1, 0}  // row-major

// Tiled layout for GPU shared memory efficiency
f32[256, 256]{1, 0:T(32, 32)}
// Elements are organized in 32x32 tiles
```

Tiled layouts are particularly important for matrix multiplication, where the tiling matches the GPU's shared memory access patterns.

---

### 13.2.6 SPMD Partitioning

SPMD (Single Program, Multiple Data) partitioning splits a single HLO program across multiple devices. Each device executes the same program but operates on a different shard of the data.

#### Partitioning Process

```
1. Sharding Annotation
   - Framework (JAX) annotates each HLO instruction with a sharding specification
   - Sharding specifies how each tensor is split across devices

2. Partitioner
   - Transforms the HLO module to operate on local shards
   - Inserts collective operations (all-gather, all-reduce, collective-permute) for cross-device communication
   - Uses PartitionId to determine which shard to process

3. Verification
   - Verifies that the partitioned program produces correct results
   - Checks that all necessary communication is inserted
```

#### Sharding Specification

Sharding is specified via `HloSharding` objects attached to instructions as frontend attributes:

```
%x = f32[1024] parameter(0),
    sharding={devices=[4,1]<=[4]}

// After partitioning (on device 2):
%local_x = f32[256] parameter(0)  // local shard of size 1024/4=256
```

#### Sharding Types

| Type | Description | Example |
|------|-------------|---------|
| **Tuple sharding** | Each tuple element has its own sharding | `{devices=[2,2], devices=[4]}` |
| **Array sharding** | Per-dimension device assignment | `{devices=[2,2]}` |
| **Replicated** | Same data on all devices | `{replicated}` |
| **Manual** | User manages the partitioning | `{manual}` |
| **Tile assignment** | N-dimensional device grid | `{devices=[2,2,1]T(2,2)}` |

#### Inserted Collective Operations

The partitioner inserts collective operations at points where data dependencies cross device boundaries:

```
// Original: %result = dot(%x, %y)
// If %x is sharded on dim 0 and %y is sharded on dim 1:
%local_x = f32[256, 64] parameter(0)    // local shard of x
%local_y = f32[64, 256] parameter(1)    // local shard of y

// Partial dot product on local shard
%partial = f32[256, 256] dot(%local_x, %local_y)

// All-reduce to sum partial results across devices
%result = f32[256, 256] all-reduce(%partial), to_apply=%add,
    replica_groups={{0, 1, 2, 3}}
```

---

### 13.2.7 HLO to LLVM IR Lowering

After all HLO optimizations are complete, the HLO module is lowered to LLVM IR. This stage converts each HLO instruction into LLVM IR that implements the operation.

#### Lowering Strategy

```
HloModule
    │
    ├── Lower to kLoop fusion bodies
    │     └─ Each fusion becomes an LLVM function
    │
    ├── Lower elementwise operations
    │     └─ Map to LLVM vector instructions
    │
    ├── Lower reduce operations
    │     └─ Generate reduction loops with shuffle instructions
    │
    ├── Lower dot operations
    │     └─ Call library functions (cuBLAS) or emit tiled GEMM
    │
    ├── Lower convolution operations
    │     └─ Call library functions (cuDNN) or emit winograd/FFT
    │
    ├── Lower data movement operations
    │     └─ Generate memcpy or elementwise copy loops
    │
    ├── Lower control flow
    │     └─ Generate LLVM function calls for while/conditional
    │
    v
LLVM IR Module
```

#### GPU-Specific Lowering (NVPTX)

On NVIDIA GPUs, LLVM IR is further lowered to PTX (Parallel Thread Execution) via LLVM's NVPTX backend:

```
LLVM IR -> NVPTX Machine Code -> PTX Assembly -> Cubin (by ptxas)
```

The GPU lowering emits:
- `__global__` kernel functions for each fusion or standalone operation.
- Thread index calculations (`threadIdx`, `blockIdx`, `blockDim`).
- Shared memory declarations for reductions.
- Calls to device library functions (`__nv_expf`, `__nv_logf`, etc.).

#### CPU-Specific Lowering

On CPU, LLVM IR is lowered to x86 machine code via LLVM's X86 backend:

```
LLVM IR -> X86 Machine Code -> Object File
```

The CPU lowering leverages:
- LLVM's auto-vectorization for elementwise operations.
- Eigen library calls for matrix operations.
- LLVM's vector intrinsics (AVX2, AVX-512).

---

### 13.2.8 LLVM Optimization

After lowering to LLVM IR, LLVM's optimization pipeline runs:

```
LLVM IR (unoptimized)
    │
    ├── -mem2reg: Promote stack allocations to registers
    ├── -instcombine: Combine redundant instructions
    ├── -reassociate: Reassociate expressions for better CSE
    ├── -gvn: Global value numbering (LLVM's CSE)
    ├── -simplifycfg: Simplify control flow
    ├── -loop-vectorize: Auto-vectorize loops (CPU)
    ├── -slp-vectorize: SLP vectorization (CPU)
    ├── -inline: Inline function calls
    │
    v
LLVM IR (optimized)
```

On GPU, additional passes specific to the NVPTX backend run:

```
    ├── NVPTX-specific passes
    ├── -nvvm-reflect: Handle NVVM reflect parameters
    ├── Lower GPU intrinsics
    ├── Optimize for PTX execution model
```

---

### 13.2.9 Native Code Generation

The final stage generates executable code for the target:

#### GPU Code Generation

```
Optimized LLVM IR (NVPTX)
    │
    ├── LLVM NVPTX backend
    │     └─ Generate PTX assembly
    │
    ├── PTX → Cubin (via ptxas or driver JIT)
    │     └─ Compile to GPU machine code
    │
    ├── Kernel metadata extraction
    │     └─ Grid size, block size, shared memory size
    │
    v
Executable (xla::Executable)
    └─ Contains: cubin, kernel entry points, buffer assignment info
```

#### CPU Code Generation

```
Optimized LLVM IR (X86)
    │
    ├── LLVM X86 backend
    │     └─ Generate object file
    │
    ├── Linking
    │     └─ Link with runtime libraries (Eigen, etc.)
    │
    v
Executable (xla::Executable)
    └─ Contains: object code, function pointers, buffer assignment info
```

---

## 13.3 GPU Compilation Pipeline

The GPU compilation pipeline is the most complex pipeline in XLA, with several GPU-specific stages.

### 13.3.1 GPU Pipeline Overview

```
StableHLO Input
    │
    v
[HLO Conversion]
    │
    v
[Target-Independent Optimizations]
    ├── AlgebraicSimplifier
    ├── CSE
    ├── DCE
    ├── ConvolutionRewriter
    ├── DotMerger
    ├── InstructionFusion (pass 1)
    ├── WhileLoopSimplifier
    ├── TupleSimplifier
    ├── SortSimplifier
    ├── CallInliner
    ├── ConstantFolding
    ├── WindowReversion
    ├── ScatterExpander
    ├── ZeroSizedHloElimination
    │
    v
[Layout Assignment]
    │
    v
[GPU-Specific Optimizations]
    ├── GpuConvRewrite (cuDNN selection)
    ├── GpuDotRewrite (cuBLAS selection)
    ├── GpuFusion (pass 2)
    ├── GpuMultiOutputFusion
    ├── GpuRewrite (backend-specific rewrites)
    ├── GemmRewriter (Triton GEMM)
    ├── GemmAlgorithmPicker (autotuning)
    ├── ConvolutionAlgorithmPicker (autotuning)
    ├── AllReduceRewrite (NCCL selection)
    ├── GpuCopyInsertion
    ├── GpuDefragmenter
    │
    v
[Memory Planning]
    ├── BufferAssignment
    ├── HeapSimulator
    │
    v
[Code Generation]
    ├── HLO → LLVM IR
    ├── LLVM Optimization
    ├── NVPTX Code Generation
    ├── PTX → Cubin
    │
    v
[Runtime IR Generation]
    ├── Thunk sequence generation
    ├── Buffer allocation schedule
    │
    v
Executable
```

### 13.3.2 Library Selection

One of the most impactful GPU optimizations is selecting the right library for each operation.

#### cuBLAS Selection (GemmRewriter)

The `GemmRewriter` pass rewrites dot operations as custom calls to cuBLAS:

```
Before:
  %result = f32[M, N] dot(%lhs, %rhs), ...

After:
  %result = f32[M, N] custom-call(%lhs, %rhs),
      call_target_name="__cublas$gemm",
      backend_config="{...algorithm...}"
```

The rewrite considers:
- Matrix dimensions (selects batched vs. non-batched GEMM).
- Data types (selects FP32, FP16, BF16, TF32, INT8 kernels).
- Transposition requirements (inserts transposes or uses cuBLAS trans flags).
- Algorithm selection via autotuning.

#### cuDNN Selection (GpuConvRewrite)

The `GpuConvRewrite` pass rewrites convolution operations as custom calls to cuDNN:

```
Before:
  %result = f32[N, H, W, C_out] convolution(%input, %filter), ...

After:
  %result = f32[N, H, W, C_out] custom-call(%input, %filter),
      call_target_name="__cudnn$convForward",
      backend_config="{...algorithm, workspace_size...}"
```

Convolution rewrite decisions:
- Forward vs. backward convolution.
- Algorithm selection (implicit GEMM, Winograd, FFT, etc.) via autotuning.
- Workspace memory allocation.
- Fused activation (Relu, BiasAdd) via cuDNN's fused operations.

#### NCCL Selection (AllReduceRewrite)

Collective operations are rewritten to use NCCL:

```
Before:
  %result = f32[M, N] all-reduce(%operand), to_apply=%add, ...

After:
  %result = f32[M, N] custom-call(%operand),
      call_target_name="__nccl$AllReduce",
      backend_config="{...reduction_operation=add...}"
```

NCCL selection handles:
- AllReduce, AllGather, ReduceScatter, AllToAll, CollectivePermute.
- Channel configuration for overlapping communication.
- Ring vs. tree algorithms.

### 13.3.3 Direct LLVM IR Emission

Operations that cannot be delegated to libraries are emitted directly as LLVM IR. This includes:

| Operation Category | Emission Strategy |
|-------------------|-------------------|
| Elementwise unary | Map to LLVM vector operations or PTX intrinsics |
| Elementwise binary | Map to LLVM vector operations |
| Reduce | Generate reduction with warp shuffles or shared memory |
| Transpose | Generate tiled transpose kernels |
| Concatenate, Slice | Generate index arithmetic |
| Pad | Generate conditional writes |
| Sort | Generate bitonic sort network |
| Gather/Scatter | Generate gather/scatter memory access |
| Select | Generate conditional moves |
| Broadcast | Generate stride-based indexing |

### 13.3.4 Triton Code Generation

XLA can use Triton to generate optimized GEMM (matrix multiplication) kernels. This provides an alternative to cuBLAS for some operations.

#### Triton GEMM Pipeline

```
HLO Dot Operation
    │
    ├── Analyze matrix dimensions and types
    │
    ├── Select Triton GEMM configuration
    │     └─ Tile sizes, pipeline stages, data layout
    │
    ├── Generate Triton IR
    │     └─ Kernel with tiled load/compute/store
    │
    ├── Triton → LLVM IR → PTX
    │
    ├── Autotune over configurations
    │     └─ Try multiple tile sizes, pipeline depths
    │
    v
Optimized GEMM kernel (as custom call)
```

#### Enabling Triton GEMM

```python
# JAX flag to enable Triton GEMM
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=true'

# Autotuning level
# 0: no autotuning (use heuristics)
# 1: autotune on first compilation
# 2: autotune with more configurations
# 3: exhaustive autotuning
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=2'
```

Triton GEMM is particularly beneficial for:
- Shapes that cuBLAS does not optimize well (non-standard tile sizes).
- Fused operations (GEMM + activation, GEMM + epilogue).
- Custom data types or layouts.

### 13.3.5 Buffer Assignment and Scheduling

#### Buffer Assignment

Buffer assignment maps each HLO instruction's output to a physical memory buffer. The goal is to minimize total memory usage while respecting data dependencies.

```
Buffer Assignment Algorithm:

1. Compute buffer liveness:
   - For each instruction, determine when its output buffer is first and last used.

2. Build interference graph:
   - Two buffers interfere if they are alive at the same time.

3. Allocate buffers:
   - Greedily assign offsets in a contiguous memory pool.
   - Buffers that don't interfere can share the same memory (buffer reuse).

4. Color the interference graph:
   - Assign memory offsets using a greedy coloring heuristic.
```

Buffer assignment output:

```cpp
struct BufferAssignment {
  // Maps each HLO instruction to its buffer allocation
  absl::flat_hash_map<const HloInstruction*, BufferAllocation*> allocation_map;

  // All buffer allocations
  std::vector<BufferAllocation> allocations;

  // Total memory required
  int64_t total_size() const;
};
```

#### Heap Simulator

The heap simulator models the memory allocation process:

```
Instruction sequence:
  %a = parameter(0)     -> Alloc buffer_a [1024 bytes]
  %b = parameter(1)     -> Alloc buffer_b [1024 bytes]
  %c = add(%a, %b)      -> Alloc buffer_c [1024 bytes]
                          Free buffer_a? No, still needed by %e
  %d = multiply(%c, %c) -> Alloc buffer_d [1024 bytes]
                          Free buffer_c (last use)
  %e = add(%a, %d)      -> Alloc buffer_e [1024 bytes]
                          Free buffer_a (last use)
                          Free buffer_b (last use)
                          Free buffer_d (last use)

Peak memory: 5 * 1024 = 5120 bytes
With buffer reuse: buffer_c and buffer_e can share (non-overlapping liveness)
  -> Peak memory: 4 * 1024 = 4096 bytes
```

#### Instruction Scheduling

Instruction scheduling determines the execution order of HLO instructions. The scheduler must respect data dependencies while optimizing for memory usage.

```
Scheduling strategies:

1. Post-order scheduling:
   - Schedule instructions in post-order of the dependency graph.
   - Simple but may not minimize peak memory.

2. Memory-minimizing scheduling:
   - Schedule instructions to minimize peak memory usage.
   - Uses a cost model based on buffer sizes and liveness.
   - Priority: schedule instructions that free the most memory first.

3. Latency-hiding scheduling:
   - Overlap computation with communication (async operations).
   - Schedule independent operations in parallel.
```

### 13.3.6 Runtime IR Generation

After code generation, XLA creates a "thunk sequence" -- a list of runtime operations to execute:

```
ThunkSequence:
  ├── NcclAllReduceThunk (all-reduce on gradient)
  ├── KernelThunk (execute fusion kernel 1)
  ├── CublasGemmThunk (execute cuBLAS GEMM)
  ├── KernelThunk (execute fusion kernel 2)
  ├── CopyThunk (copy between buffers)
  ├── CudnnConvThunk (execute cuDNN convolution)
  ├── KernelThunk (execute fusion kernel 3)
  └── ...
```

Each thunk encapsulates:
- The operation to perform (kernel launch, library call, memory copy).
- Input and output buffer locations.
- Stream assignment (which GPU stream to execute on).
- Dependencies on other thunks.

---

## 13.4 CPU Compilation Pipeline

The CPU compilation pipeline shares the target-independent optimization stages with GPU but diverges at layout assignment and code generation.

### 13.4.1 CPU Pipeline Overview

```
StableHLO Input
    │
    v
[HLO Conversion]
    │
    v
[Target-Independent Optimizations]
    │
    v
[CPU-Specific Optimizations]
    ├── CpuLayoutAssignment
    ├── CpuInstructionFusion
    ├── DotRewriter (Eigen-based)
    ├── ConvRewriter (Eigen-based)
    ├── ParallelTaskAssignment
    │
    v
[Code Generation]
    ├── HLO → LLVM IR (via IrEmitterCPU)
    │     └─ Uses Eigen for matrix ops
    │     └─ Uses LLVM intrinsics for vector ops
    ├── LLVM Optimization
    │     ├── -O2 optimization pipeline
    │     ├── Loop vectorization
    │     ├── SLP vectorization
    │     └── Inline Eigen kernels
    ├── X86 Code Generation
    │     └─ AVX2 / AVX-512 / SSE support
    │
    v
Executable (xla::LocalExecutable)
```

### 13.4.2 Eigen-Based Lowering

The CPU backend uses the Eigen library for matrix operations:

| Operation | Eigen Implementation |
|-----------|---------------------|
| Dot (matrix multiply) | `Eigen::MatrixXf::multiply()` or tensor contractions |
| Convolution | `Eigen::Tensor::convolve()` |
| Reduce | `Eigen::Tensor::reduce()` |
| Broadcast | `Eigen::Tensor::broadcast()` |
| Chip/Slice | `Eigen::Tensor::chip()` |

### 13.4.3 Multi-Threading Support

The CPU backend supports multi-threaded execution:

```
Configuration:
  --xla_cpu_multi_thread_eigen=true    // Enable multi-threading
  --xla_cpu_force_n_threads=N          // Force N threads (default: all cores)

Threading model:
  - Eigen thread pool for parallel operations
  - Parallel loop scheduling for elementwise operations
  - Parallel reduction with partial sum aggregation
```

Threading is applied at two levels:

1. **Instruction-level parallelism**: Independent HLO instructions are executed in parallel using a thread pool.
2. **Data parallelism**: Large elementwise operations are split across threads.

---

## 13.5 Buffer Assignment (Detailed)

### 13.5.1 Static Memory Allocation

XLA uses static memory allocation: all buffers are assigned at compile time, with no runtime allocation overhead. This is possible because XLA compiles for specific input shapes.

```
Static allocation process:

1. Run the HLO scheduling to determine instruction execution order.
2. Compute live ranges for each buffer:
   - Birth: instruction that writes the buffer.
   - Death: last instruction that reads the buffer.
3. Build a live range interference graph.
4. Assign offsets in a contiguous memory pool using greedy coloring.
5. Output: {buffer_id -> (offset, size)} mapping.
```

### 13.5.2 Live Range Analysis

Live range analysis determines the interval during which each buffer must be preserved:

```
Example computation:
  %0 = parameter(0)                  // buffer[0] born
  %1 = parameter(1)                  // buffer[1] born
  %2 = add(%0, %1)                   // buffer[2] born; buffer[0], buffer[1] still alive
  %3 = multiply(%2, %2)             // buffer[3] born; buffer[2] still alive
                                    // buffer[0], buffer[1] die after %2
  ROOT %4 = subtract(%3, %2)        // buffer[4] born
                                    // buffer[2] dies after %4
                                    // buffer[3] dies after %4

Live ranges:
  buffer[0]: [%0, %2]  -- parameter must live until used
  buffer[1]: [%1, %2]
  buffer[2]: [%2, %4]  -- add result used by multiply and subtract
  buffer[3]: [%3, %4]
  buffer[4]: [%4, %4]  -- output, lives to end

Interference:
  buffer[2] interferes with buffer[0], buffer[1], buffer[3]
  buffer[3] interferes with buffer[2]
```

### 13.5.3 Buffer Reuse

Buffer reuse allows two non-interfering buffers to occupy the same memory:

```
Memory layout (with reuse):

Offset 0:    buffer[0] (1024 bytes) | buffer[3] (1024 bytes)
Offset 1024: buffer[1] (1024 bytes) | buffer[4] (1024 bytes)
Offset 2048: buffer[2] (1024 bytes)

Total: 3072 bytes (instead of 5120 without reuse)

Reuse decisions:
  buffer[0] and buffer[3] don't interfere -> same memory
  buffer[1] and buffer[4] don't interfere -> same memory
```

Buffer reuse is implemented by the `HeapSimulator` class, which supports multiple allocation strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `kNoReuse` | No buffer reuse (baseline) | Debugging |
| `kReuseBySize` | Reuse by matching buffer sizes | General purpose |
| `kReuseBySizeTraversalOrder` | Reuse by size + scheduling order | Default for most backends |
| `kNumStrategies` | Number of strategies | -- |

### 13.5.4 Alias Analysis for Buffer Assignment

Buffer assignment leverages alias analysis to enable in-place operations:

```
Parameter-output alias:
  If the program output is the same data as an input parameter
  (e.g., a no-op), XLA can alias the parameter buffer to the
  output buffer, avoiding a copy.

  ENTRY %main (x: f32[128]) -> f32[128] {
    %x = f32[128] parameter(0)
    ROOT %result = f32[128] copy(%x)
  }
  // %result can alias %x if the caller allows it
```

---

## 13.6 Compilation Flags and Options

XLA provides extensive compilation flags for controlling the pipeline. These are set via:

- `XLA_FLAGS` environment variable (space-separated).
- `--xla_*` flags passed to the XLA service.
- `jax.jit` compiler options in JAX.

### 13.6.1 General Compilation Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--xla_dump_to` | "" | Directory for HLO dumps |
| `--xla_dump_hlo_pass_re` | "" | Regex for which passes to dump (empty = all) |
| `--xla_dump_hlo_as_text` | true | Dump HLO as text files |
| `--xla_dump_hlo_as_proto` | true | Dump HLO as protobuf files |
| `--xla_dump_hlo_as_html` | false | Dump HLO as HTML visualization |
| `--xla_dump_hlo_snapshots` | false | Dump HLO snapshots |
| `--xla_disable_all_hlo_passes` | false | Disable all optimization passes |
| `--xla_disable_hlo_passes` | "" | Comma-separated list of passes to disable |
| `--xla_enable_hlo_passes` | "" | Comma-separated list of passes to enable |
| `--xla_hlo_profile` | false | Enable HLO profiling |
| `--xla_backend_optimization_level` | 2 | Backend optimization level (0-3) |
| `--xla_force_all_intermediate_buffer_occupancy` | false | Force all intermediate buffers to be allocated |

### 13.6.2 GPU-Specific Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--xla_gpu_enable_triton_gemm` | false | Enable Triton GEMM code generation |
| `--xla_gpu_triton_gemm_version` | "" | Specific Triton GEMM version to use |
| `--xla_gpu_autotune_level` | 2 | Autotuning level (0=off, 1-3=increasing) |
| `--xla_gpu_force_compilation_parallelism` | 0 | Max parallel compilation threads |
| `--xla_gpu_crash_on_verification_fail` | false | Crash on cuDNN/cuBLAS verification failure |
| `--xla_gpu_max_kernel_unroll_factor` | 0 | Max unroll factor for GPU kernels (0=auto) |
| `--xla_gpu_enable_pipelined_fusion` | true | Enable pipelined fusion on GPU |
| `--xla_gpu_enable_reduce_scatter_fusion` | true | Enable reduce-scatter fusion |
| `--xla_gpu_enable_while_loop_double_buffering` | true | Enable double buffering for while loops |
| `--xla_gpu_enable_all_gather_pipeline_fusion` | false | Enable all-gather pipeline fusion |
| `--xla_gpu_enable_cublaslt` | true | Enable cuBLASLt for GEMM |
| `--xla_gpu_enable_cudnn_convolution` | true | Enable cuDNN for convolution |
| `--xla_gpu_enable_nccl` | true | Enable NCCL for collective operations |
| `--xla_gpu_data_dir` | "" | Directory for autotuning cache |
| `--xla_gpu_dump_autotune_results_to` | "" | File to dump autotuning results |
| `--xla_gpu_load_autotune_results_from` | "" | File to load autotuning results from |
| `--xla_gpu_address_compiler_hints` | false | Enable address compiler hints |
| `--xla_gpu_enable_triton_sgmm` | false | Enable Triton for sparse GEMM |
| `--xla_gpu_enable_schedule_pass` | true | Enable GPU scheduling pass |
| `--xla_gpu_enable_compilation_environment_cache` | true | Cache compilation environments |

### 13.6.3 CPU-Specific Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--xla_cpu_multi_thread_eigen` | true | Enable multi-threaded Eigen execution |
| `--xla_cpu_force_n_threads` | 0 | Force number of threads (0=all cores) |
| `--xla_cpu_use_thunk_runtime` | false | Use thunk-based runtime for CPU |
| `--xla_cpu_llvm_clang_opt_level` | "" | LLVM optimization level for CPU |
| `--xla_cpu_enable_fast_math` | false | Enable fast-math for CPU LLVM |
| `--xla_cpu_enable_profiling` | false | Enable CPU profiling |

### 13.6.4 Memory and Buffer Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--xla_gpu_enable_highest_priority_api` | false | Enable highest priority stream for API calls |
| `--xla_gpu_persistent_cache_dir` | "" | Directory for persistent kernel cache |
| `--xla_gpu_persistent_cache_per_thread` | false | Per-thread persistent cache |
| `--xla_gpu_enable_triton_fusion` | false | Enable Triton for non-GEMM fusion |

### 13.6.5 Debugging Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--xla_dump_to` | "" | Dump HLO to directory |
| `--xla_dump_fusion_profiles` | false | Dump fusion performance profiles |
| `--xla_experimental_custom_calls` | "" | Enable experimental custom call features |
| `--xla_gpu_enable_llvmir_compilation_dump` | false | Dump LLVM IR during compilation |
| `--xla_gpu_dump_llvmir` | "" | Dump LLVM IR to file |
| `--xla_gpu_verify_ir` | false | Verify LLVM IR after each pass |
| `--xla_debug_memory_use` | false | Debug memory allocation |

### 13.6.6 Using Flags in JAX

```python
import os

# Method 1: Environment variable (must be set before importing JAX)
os.environ['XLA_FLAGS'] = (
    '--xla_dump_to=/tmp/hlo_dumps '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_gpu_enable_triton_gemm=true '
    '--xla_gpu_autotune_level=3'
)

import jax
import jax.numpy as jnp

# Method 2: Per-function JIT options (limited)
@jax.jit
def f(x):
    return x + 1

# Method 3: Using jax.debug for HLO inspection
import jax
jax.debug_hlo("before_optimizations")(f)(x)
```

### 13.6.7 Compilation Caching

XLA caches compiled executables to avoid recompilation for the same input shapes:

```
Cache key = hash(program_shape + compilation_options)

Cache hit: Reuse existing executable
Cache miss: Compile new executable and add to cache
```

Cache-related flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--xla_cache_computation` | true | Enable computation caching |
| `--xla_cache_key_include_platform_info` | true | Include platform info in cache key |
| `--xla_persistent_cache_path` | "" | Path for persistent compilation cache |
| `--xla_persistent_cache_read_only` | false | Read-only mode for persistent cache |

---

## 13.7 Compilation Pipeline Example

### End-to-End Example: Matrix Multiply + ReLU

```python
import jax
import jax.numpy as jnp

@jax.jit
def matmul_relu(x, w, b):
    return jnp.maximum(x @ w + b, 0.0)

x = jnp.ones((128, 64))
w = jnp.ones((64, 32))
b = jnp.ones((32,))
result = matmul_relu(x, w, b)
```

#### Pipeline Trace

```
Stage 1: JAX traces the function
  -> Produces StableHLO:
     %0 = stablehlo.dot_general(%x, %w), contracting_dims={1, 0}
     %1 = stablehlo.add(%0, %b)
     %2 = stablehlo.maximum(%1, stablehlo.constant(0.0))

Stage 2: StableHLO -> HLO Conversion
  -> Produces HLO:
     %dot = f32[128, 32] dot(%x, %w),
         dot_dimension_numbers={lhs_contracting={1}, rhs_contracting={0}}
     %bias = f32[128, 32] add(%dot, %b)
     %result = f32[128, 32] maximum(%bias, f32[] constant(0))

Stage 3: Target-Independent Optimizations
  -> CSE: no common subexpressions found
  -> AlgebraicSimplifier: no simplifications possible
  -> ConstantFolding: constants folded
  -> Fusion: fuse add + maximum into one kernel

Stage 4: Layout Assignment
  -> %x: f32[128, 64]{1, 0}  (row-major)
  -> %w: f32[64, 32]{1, 0}   (row-major)
  -> %dot: f32[128, 32]{1, 0} (row-major)
  -> %bias, %result: f32[128, 32]{1, 0} (row-major)

Stage 5: GPU-Specific Optimizations
  -> DotRewrite: dot -> custom_call("__cublas$gemm", ...)
  -> Fusion: add + maximum fused into loop fusion
  -> Result:
     %dot = custom_call(%x, %w), call_target_name="__cublas$gemm"
     %result = fusion(%dot, %b, %zero), kind=kLoop

Stage 6: Buffer Assignment
  -> Buffer 0: %x input (128*64*4 = 32768 bytes)
  -> Buffer 1: %w input (64*32*4 = 8192 bytes)
  -> Buffer 2: %b input (32*4 = 128 bytes)
  -> Buffer 3: %dot result (128*32*4 = 16384 bytes) -> reused by %result
  -> Total: ~57.6 KB

Stage 7: Code Generation
  -> Kernel 1: cuBLAS sgemm for %dot
  -> Kernel 2: LLVM IR for fused add + maximum
     define void @fusion(float* %dot, float* %b, float* %result) {
       for i in range(128*32):
         %val = %dot[i] + %b[i % 32]
         %result[i] = max(%val, 0.0)
     }

Stage 8: LLVM Optimization
  -> Vectorize the fusion kernel (AVX2 on CPU, SIMD on GPU)

Stage 9: Native Code Generation
  -> GPU: PTX + cubin
  -> CPU: x86 object code
```

#### Inspecting the Compilation

```python
# Dump HLO to inspect compilation stages
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/matmul_relu_hlo'

# After running, inspect the dumps:
# /tmp/matmul_relu_hlo/module_000.before_optimizations.txt
# /tmp/matmul_relu_hlo/module_001.after_optimizations.txt
# /tmp/matmul_relu_hlo/module_002.after_fusion.txt
# /tmp/matmul_relu_hlo/module_003.after_layout.txt
# /tmp/matmul_relu_hlo/module_004.after_backend_optimizations.txt
```

---

## 13.8 Advanced Topics

### 13.8.1 Async Compilation

XLA supports asynchronous compilation where compilation happens in a background thread:

```python
import jax

# Compile asynchronously
compiled = jax.jit(matmul_relu).lower(x, w, b).compile()

# Check compilation status
print(compiled.runtime_compilation_time)

# Execute
result = compiled(x, w, b)
```

### 13.8.2 AOT Compilation

Ahead-of-time compilation produces a serialized executable that can be loaded without the XLA compiler:

```python
import jax
from jax import export

# Export as StableHLO
exported = export.export(jax.jit(matmul_relu))(x, w, b)

# Serialize
serialized = exported.serialize()

# The serialized form can be saved, transmitted, and loaded
# on a different machine for execution without needing the
# original Python code or XLA compiler
```

### 13.8.3 Compilation Time Analysis

To understand where compilation time is spent:

```
XLA compilation time breakdown (typical):

1. HLO optimization passes:        20-40%
   - Fusion:                       10-20%
   - Layout assignment:            5-10%
   - Other passes:                 5-10%

2. Code generation:                30-50%
   - LLVM IR generation:           10-20%
   - LLVM optimization:            10-20%
   - PTX generation + ptxas:       10-15%

3. Autotuning:                     10-30%
   - GEMM algorithm selection:     5-15%
   - Conv algorithm selection:     5-15%
   - Triton autotuning:            variable
```

### 13.8.4 Memory Budget Constraints

XLA can be constrained to stay within a memory budget:

```
--xla_gpu_memory_limit_bytes=N

When set, XLA will:
1. Aggressively rematerialize operations instead of caching results.
2. Prefer smaller fusion kernels.
3. Use smaller workspace sizes for library calls.
4. Reduce the number of concurrent buffers.
```

### 13.8.5 Multi-Device Compilation

For multi-device (multi-GPU) compilation:

```
--xla_num_replicas=N           // Number of data-parallel replicas
--xla_num_partitions=N         // Number of SPMD partitions

The compilation pipeline adds:
1. SPMD partitioning pass
2. NCCL collective operation insertion
3. Cross-device scheduling
4. Per-device buffer assignment
```
