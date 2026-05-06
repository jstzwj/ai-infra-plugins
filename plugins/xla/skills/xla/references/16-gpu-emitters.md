# XLA Reference - Chapter 16: GPU Emitters

This reference provides comprehensive documentation on XLA's GPU emitter infrastructure, covering the three code generation approaches, the hero-based codegen framework, partitioning, elemental emission, and the specific emitter pipelines.

---

## 16.1 Overview

GPU emitters are responsible for converting HLO fusion computations into executable GPU kernels. The emitter infrastructure supports multiple code generation strategies and automatically selects the best approach based on the fusion's structure. This chapter covers the emitter architecture in detail, based on the design described in the XLA GPU emitters documentation.

---

## 16.2 Three Code Generation Approaches

The GPU backend uses three distinct approaches for generating GPU code from HLO:

### 16.2.1 External Library Calls

External library calls delegate execution to vendor-optimized libraries. This is the preferred approach for operations where highly optimized implementations exist:

- **cuBLAS**: All matrix multiplication operations (dot products, batched matmul).
- **cuDNN**: Convolution operations, normalization, and fused convolution patterns.
- **NCCL**: Collective communication (all-reduce, all-gather, etc.).
- **cuFFT**: Fast Fourier transforms.
- **cuRAND**: Random number generation.

Library calls are not "emitted" in the traditional sense -- instead, the compiler generates a thunk (runtime descriptor) that specifies the library function to call, the input and output buffers, and the algorithm parameters. The actual code is in the library itself.

**Advantages:**
- Near-peak performance for supported operations.
- Vendor-maintained and updated with each hardware generation.
- Handles architecture-specific optimizations transparently.

**Limitations:**
- Limited fusion: libraries typically handle a single operation (or a fixed fusion pattern like conv+bias+relu).
- Black box: the compiler cannot optimize across library call boundaries.
- Not all operations have library implementations.

### 16.2.2 Triton Tiling

Triton tiling generates code using the Triton compiler, which provides a Python-like programming model for GPU programming. Triton is used for complex fusion patterns that involve:

- Matrix multiplications fused with elementwise operations.
- Softmax and flash attention.
- Layer normalization fused with surrounding operations.
- Any pattern that benefits from explicit shared memory management and tiling.

The Triton approach generates Triton IR (TTIR), which is then lowered through the Triton compiler pipeline to PTX. Triton gives the compiler control over tile sizes, shared memory usage, and memory access patterns, enabling efficient code generation for structured operations.

**Advantages:**
- Efficient handling of matmul-based fusion patterns.
- Explicit shared memory management.
- Good performance on tensor cores.
- Growing ecosystem with community-contributed kernels.

**Limitations:**
- Only supports specific patterns (matmul-centric fusions).
- Triton compiler is an additional dependency.
- May not handle all HLO operation types.

### 16.2.3 XLA Emitters (Progressive Lowering to LLVM IR)

XLA emitters directly lower HLO operations to LLVM IR through a progressive lowering pipeline. This approach handles all operations that are not dispatched to libraries or Triton, including:

- Elementwise operations (add, mul, exp, log, etc.).
- Reductions (sum, max, min, etc.).
- Transposes.
- Gather and scatter.
- Sort.
- Concatenate and slice.
- Conditional and tuple operations.

The progressive lowering pipeline goes through multiple intermediate representations:

```
HLO Fusion Computation
    |
    v
[Partitioner] -- Split fusion into emit-able partitions
    |
    v
[Emitter] -- Convert partitioned HLO to MLIR (xla_gpu dialect)
    |
    v
[MLIR Optimization Pipeline]
    - Inlining
    - xla_gpu -> scf conversion
    - Tensor flattening
    - Vectorization
    - Loop unrolling
    |
    v
[MLIR -> LLVM IR Lowering]
    |
    v
[LLVM NVPTX Backend]
    |
    v
[PTX]
```

**Advantages:**
- Handles all HLO operation types.
- Full compiler control over code generation.
- Can apply LLVM optimizations.

**Limitations:**
- May not achieve library-level performance for matmul/conv.
- Code generation quality depends on the emitter implementation.
- Increasingly complex pipeline with multiple lowering stages.

---

## 16.3 Hero-Based Codegen

The XLA GPU emitter uses a "hero-based" code generation framework. The idea is that each fusion computation is analyzed to identify the most complex operation -- the "hero" -- which determines the emission strategy. The hero is the operation that drives the kernel's structure (loop nesting, shared memory usage, synchronization).

### 16.3.1 Hero Types

There are 7 hero emitter types:

#### 1. Loop Emitter (Default)

The loop emitter is the default and most commonly used emitter. It handles fusions where all operations can be emitted in a simple loop nest -- primarily elementwise operations and broadcast patterns.

**Characteristics:**
- Each GPU thread computes one or more output elements.
- No shared memory usage.
- No inter-thread synchronization (beyond implicit block-level synchronization).
- Simple indexing: thread ID maps directly to output element index.

**When used:** When the fusion contains only elementwise operations, broadcasts, constants, and simple reshapes.

#### 2. Transpose Emitter

The transpose emitter handles fusions containing a transpose operation that benefits from shared-memory-based optimization.

**Characteristics:**
- Uses shared memory for coalesced memory access.
- Two-phase execution: coalesced read -> shared memory -> coalesced write.
- Requires thread synchronization (`__syncthreads()`).
- Tile-based: operates on tiles of the input matrix.

**When used:** When the fusion contains a non-trivial transpose (permutation) and the surrounding operations are elementwise.

#### 3. Reduction Emitter

The reduction emitter handles fusions containing reduction operations (sum, max, min, etc.).

**Characteristics:**
- Uses shared memory for partial reduction results.
- Hierarchical reduction: thread-level -> warp-level -> block-level.
- May require multiple kernel launches for large reductions.
- Handles row/column reductions differently.

**When used:** When the fusion contains a reduce operation as the hero.

#### 4. Scatter Emitter

Handles fusions containing scatter operations (indexed writes).

**Characteristics:**
- Handles atomic and non-atomic scatter modes.
- Manages write conflicts through atomics or serialization.
- Supports scatter with updates and indices.

**When used:** When the fusion contains a scatter as the hero.

#### 5. Sort Emitter

Handles sorting operations within a fusion.

**Characteristics:**
- Implements bitonic sort or odd-even merge sort.
- Uses shared memory for efficient comparison and swap operations.
- Requires multiple passes over the data.

**When used:** When the fusion contains a sort operation.

#### 6. Select-And-Scatter Emitter

Handles the select-and-scatter pattern used in max/average pooling backward pass.

**Characteristics:**
- Combines element selection with scatter operations.
- Handles window-based selection patterns.

**When used:** When the fusion contains select-and-scatter.

#### 7. Custom Emitter

Handles backend-specific custom operations.

**Characteristics:**
- Allows backend-specific code generation.
- May use hand-written PTX or specialized LLVM IR.

**When used:** For operations with custom emission requirements.

### 16.3.2 Hero Selection

The hero selection algorithm identifies the "hero" operation in a fusion:

1. Enumerate all operations in the fusion computation.
2. Classify each operation by its emitter type (loop, transpose, reduction, etc.).
3. Select the operation with the highest "complexity" as the hero. The complexity ordering is roughly: sort > scatter > reduction > transpose > loop.
4. Verify that the remaining operations can be emitted around the hero (i.e., they are elementwise with respect to the hero's output or compatible inputs).

If no single hero can be identified (e.g., the fusion contains incompatible operations), the fusion is split into multiple fusions, each with its own hero.

---

## 16.4 High-Level Building Blocks

The emitter infrastructure consists of three high-level building blocks:

### 16.4.1 Computation Partitioner

The computation partitioner splits a fusion computation into smaller "partitions" that can each be emitted independently. This is necessary because not all operations in a fusion can be emitted together using a single emission strategy.

**Partitioning criteria:**

1. **Safety**: Two operations can be in the same partition if they are compatible with the same emitter type.
2. **Efficiency**: Operations should be partitioned to maximize the amount of work done in each partition (minimizing inter-partition memory traffic).
3. **Correctness**: The partitioning must preserve the dataflow semantics of the original fusion.

**Partition representation:**

```
Fusion computation:
  %0 = exp(%input)
  %1 = transpose(%0, {1, 0})
  %2 = add(%1, %bias)

Partitions:
  Partition 0 (Transpose hero):
    %0 = exp(%input)
    %1 = transpose(%0, {1, 0})
  Partition 1 (Loop hero):
    %2 = add(%1, %bias)
```

### 16.4.2 Emitters

Emitters convert partitioned HLO into MLIR using the `xla_gpu` dialect. Each emitter type generates specific MLIR patterns:

- **Loop emitter**: Generates `xla_gpu.loop` operations that iterate over output elements.
- **Transpose emitter**: Generates shared memory allocation, load, synchronize, and store operations.
- **Reduction emitter**: Generates partial reduction loops and final reduction operations.

### 16.4.3 Compilation Pipeline

The compilation pipeline optimizes and lowers the emitted MLIR to LLVM IR and then to PTX:

```
xla_gpu MLIR
    |
    v
[Inliner] -- Inline partitioned function calls
    |
    v
[xla_gpu -> scf] -- Convert xla_gpu ops to structured control flow
    |
    v
[Flatten tensors] -- Convert multi-dimensional tensors to 1D
    |
    v
[Vectorization] -- Generate vectorized memory operations
    |
    v
[Loop unrolling] -- Unroll small loops for better ILP
    |
    v
[Convert to LLVM] -- Lower to LLVM IR dialect
    |
    v
[Translate to LLVM] -- Convert MLIR LLVM dialect to LLVM IR
    |
    v
[NVPTX Backend] -- Compile to PTX
```

---

## 16.5 Partitioning in Detail

### 16.5.1 Safety Criteria for Emitting Together

Two operations can be emitted in the same partition if and only if:

1. **Same hero type**: Both operations are compatible with the same hero emitter. For example, two elementwise operations can share a loop emitter; a transpose and a reduction cannot share a single emitter.

2. **No data hazard**: There are no write-after-read or write-after-write conflicts between the operations when executed in the kernel's execution model.

3. **Compatible indexing**: Both operations use compatible indexing schemes. For example, a transpose changes the indexing of its output relative to its input. If the next operation expects the original indexing, the two cannot be emitted together without additional index transformation.

4. **Memory safety**: The combined operation does not exceed shared memory or register limits.

### 16.5.2 Multi-User Instruction Handling

An instruction with multiple users (consumers) requires special handling during partitioning:

**Case 1: All users in the same partition.**
The instruction is emitted once, and its result is used by all consumers.

**Case 2: Users in different partitions.**
The instruction's result must be written to global memory so that downstream partitions can read it. This adds memory traffic but is sometimes unavoidable.

```
%0 = exp(%input)
%1 = transpose(%0, {1, 0})  // User 1: uses %0 as input to transpose
%2 = reduce(%0, axes={0})   // User 2: uses %0 as input to reduce

// %0 has two users. If transpose and reduce are in different partitions,
// %0 must be written to global memory.
```

**Case 3: Users have different hero types.**
The fusion is split so that each partition handles one user. The common input is materialized in global memory.

### 16.5.3 Example Partitioning Scenarios

**Scenario 1: Simple elementwise chain.**
```
%0 = exp(%input)
%1 = add(%0, %bias)
%2 = tanh(%1)

All operations are elementwise -> single partition with loop emitter.
```

**Scenario 2: Transpose with elementwise.**
```
%0 = transpose(%input, {1, 0})
%1 = add(%0, %bias)

Two possible partitions:
  Option A: Single partition with transpose emitter (add is emitted inside the transpose kernel).
  Option B: Two partitions - transpose kernel + elementwise add kernel.

Option A is preferred because it avoids writing the transpose result to global memory.
```

**Scenario 3: Reduction followed by broadcast.**
```
%0 = reduce(%input, axes={1})  // [N, M] -> [N]
%1 = broadcast(%0)              // [N] -> [N, M]

Two partitions:
  Partition 0: Reduction (hero = reduce)
  Partition 1: Broadcast (hero = loop)

The reduction result is written to global memory and read by the broadcast kernel.
```

**Scenario 4: Multiple reductions.**
```
%0 = reduce_sum(%input, axes={1})
%1 = reduce_max(%input, axes={1})

If the reductions share the same input, they can sometimes be fused into a single
kernel that computes both reductions in one pass over the input.
```

---

## 16.6 Elemental Emission

Elemental emission is the process of generating code that computes a single output element of a fusion. The emitter walks the fusion's dataflow graph from the output back to the inputs, generating code for each operation.

### 16.6.1 Indexing Transformations

Each operation transforms the output element index into input element indices. The elemental emitter must apply these transformations correctly:

#### Transpose

A transpose permutes the dimensions, which changes how an output index maps to an input index:

```
// Transpose with permutation {1, 0} on a [M, N] -> [N, M] transpose:
// Output index [i_out, j_out] maps to input index [j_out, i_out]
// In linearized form:
//   output_linear = i_out * N + j_out
//   input_linear = j_out * M + i_out

// In code:
// Given output linear index `idx`:
//   j_out = idx % N;  i_out = idx / N;
//   input_idx = j_out * M + i_out;
```

#### Broadcast

A broadcast adds new dimensions that don't correspond to any input dimension. When mapping from output index to input index, these dimensions are dropped:

```
// Broadcast scalar to [M, N]:
// Output index [i, j] maps to input index [] (scalar has no dimensions)

// Broadcast [N] to [M, N]:
// Output index [i, j] maps to input index [j]

// In general: drop dimensions that are broadcast.
```

#### Reshape

A reshape changes the shape without changing the data. The mapping from output index to input index is computed by:
1. Delinearize the output index into multi-dimensional coordinates using the output shape.
2. Relinearize the multi-dimensional coordinates using the input shape.

```
// Reshape [M*N] -> [M, N]:
// Output index [i, j] -> linear = i * N + j -> input index [linear]

// Reshape [M, N] -> [N, M]:
// Output index [i, j] -> linear = i * M + j -> input index [j, i]
//   (delinearize with input shape: [j, i] = [linear / N, linear % N])
```

#### Slice

A slice extracts a sub-range of each dimension. The mapping adds an offset:

```
// Slice [10, 20] -> [5, 10], start_indices={2, 5}:
// Output index [i, j] maps to input index [i + 2, j + 5]
```

#### Reverse

A reverse flips one or more dimensions. The mapping inverts the index along reversed dimensions:

```
// Reverse dimension 0 of [10, 20]:
// Output index [i, j] maps to input index [9 - i, j]
```

### 16.6.2 Tuple Handling

Tuple operations (e.g., the root of a multi-output fusion) are handled by emitting each tuple element separately. Each element gets its own output buffer, and the tuple operation is just a pointer-level operation (no data movement):

```
// Multi-output fusion:
// ROOT %tuple = tuple(%result1, %result2)

// Emission:
// 1. Emit %result1 computation -> write to output_buffer_0
// 2. Emit %result2 computation -> write to output_buffer_1
// 3. Tuple is implicit (output_buffer_0 and output_buffer_1 are the results)
```

### 16.6.3 Gather Support

Gather operations are emitted by computing the gather index for each output element and then loading the corresponding input element:

```
// Gather: output[i, j] = input[indices[i], j]
// For each output element (i, j):
//   1. Load index = indices[i]
//   2. Load value = input[index, j]
//   3. Store value to output[i, j]
```

The gather emitter supports:
- **Batch dimensions**: Dimensions that are passed through from input to output.
- **Offset dimensions**: Dimensions in the output that correspond to the slice of the input.
- **Collapsed slice dimensions**: Dimensions in the slice that are collapsed.
- **Start index map**: Maps from index dimensions to input dimensions.
- **Index vector dimension**: The dimension of the indices tensor that contains the index vector.

Gather operations can be expensive due to the random memory access pattern. The emitter applies the following optimizations:
- **Index caching**: Cache frequently used indices in shared memory or registers.
- **Coalesced access**: When consecutive output elements access consecutive input elements, the emitter generates coalesced loads.

---

## 16.7 Loop Emitter Pipeline

The loop emitter is the most common emitter. It generates a simple parallel loop over output elements. The pipeline consists of the following stages:

### 16.7.1 Stage 1: MLIR Conversion

Convert the partitioned HLO to MLIR using the `xla_gpu` dialect:

```mlir
// Generated MLIR for a fusion exp(x) + bias:
func.func @fusion(%input: tensor<128x256xf32>, %bias: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %result = xla_gpu.loop over tensor<128x256xf32> : tensor<128x256xf32> {
    ^bb0(%i: index, %j: index):
      %idx = arith.constant 0 : index
      %val_in = xla_gpu.pure_call @load_element(%input, %i, %j) : (tensor<128x256xf32>, index, index) -> f32
      %exp_val = math.exp %val_in : f32
      %val_bias = xla_gpu.pure_call @load_element(%bias, %i, %j) : (tensor<128x256xf32>, index, index) -> f32
      %sum = arith.addf %exp_val, %val_bias : f32
      xla_gpu.yield %sum : f32
  }
  return %result : tensor<128x256xf32>
}
```

The `xla_gpu.loop` operation represents a parallel loop over the output tensor. Each iteration of the loop computes one output element.

### 16.7.2 Stage 2: Inliner

The MLIR inliner inlines all `pure_call` operations, replacing them with the body of the called function. This is necessary because the downstream passes expect flat IR without function calls:

```mlir
// After inlining:
func.func @fusion(%input: tensor<128x256xf32>, %bias: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %result = xla_gpu.loop over tensor<128x256xf32> : tensor<128x256xf32> {
    ^bb0(%i: index, %j: index):
      // Inlined load_element:
      %linear_idx = arith.muli %i, %c256 : index
      %full_idx = arith.addi %linear_idx, %j : index
      %val_in = tensor.extract %input[%full_idx] : tensor<128x256xf32>
      // Original computation:
      %exp_val = math.exp %val_in : f32
      %val_bias = tensor.extract %bias[%full_idx] : tensor<128x256xf32>
      %sum = arith.addf %exp_val, %val_bias : f32
      xla_gpu.yield %sum : f32
  }
  return %result : tensor<128x256xf32>
}
```

### 16.7.3 Stage 3: xla_gpu to scf Conversion

Convert `xla_gpu.loop` operations to standard MLIR `scf.for` or `scf.parallel` operations:

```mlir
// After conversion:
func.func @fusion(%input: memref<32768xf32>, %bias: memref<32768xf32>, %output: memref<32768xf32>) {
  %c0 = arith.constant 0 : index
  %c32768 = arith.constant 32768 : index
  %c1 = arith.constant 1 : index
  scf.for %tid = %c0 to %c32768 step %c1 {
    %val_in = memref.load %input[%tid] : memref<32768xf32>
    %exp_val = math.exp %val_in : f32
    %val_bias = memref.load %bias[%tid] : memref<32768xf32>
    %sum = arith.addf %exp_val, %val_bias : f32
    memref.store %sum, %output[%tid] : memref<32768xf32>
  }
}
```

Note that tensors have been converted to memrefs (buffers) at this stage, as the scf dialect operates on memrefs.

### 16.7.4 Stage 4: Flatten Tensors

Multi-dimensional tensors are flattened to 1D memrefs. This simplifies indexing and enables vectorized access:

```mlir
// tensor<128x256xf32> -> memref<32768xf32>
// Index [i, j] -> linear: i * 256 + j
```

The flattening pass:
1. Computes the linearized size of each tensor.
2. Replaces multi-dimensional indexing with linear indexing.
3. Converts `tensor.extract` and `tensor.insert` to `memref.load` and `memref.store`.

### 16.7.5 Stage 5: Vectorization

The vectorization pass identifies contiguous memory access patterns and replaces scalar loads/stores with vectorized equivalents:

```mlir
// Before vectorization:
%v0 = memref.load %input[%tid] : memref<32768xf32>
%v1 = memref.load %input[%tid1] : memref<32768xf32>
%v2 = memref.load %input[%tid2] : memref<32768xf32>
%v3 = memref.load %input[%tid3] : memref<32768xf32>

// After vectorization (vector<4xf32>):
%vec = vector.load %input[%tid] : memref<32768xf32>, vector<4xf32>
```

Vectorization is applied to:
- **Contiguous loads**: When consecutive threads load consecutive memory addresses.
- **Contiguous stores**: When consecutive threads store to consecutive memory addresses.
- **Elementwise operations**: When the same operation is applied to all elements of a vector.

The vectorization factor is determined by:
- The data type (e.g., 4 for f32, 8 for f16).
- The alignment of the memory access.
- The GPU's memory coalescing requirements.

### 16.7.6 Stage 6: Loop Unrolling

Small loops are unrolled to reduce loop overhead and increase instruction-level parallelism:

```mlir
// Before unrolling:
scf.for %i = %c0 to %c4 step %c1 {
  %val = memref.load %input[%i] : memref<4xf32>
  %exp = math.exp %val : f32
  memref.store %exp, %output[%i] : memref<4xf32>
}

// After unrolling (factor 4):
%val0 = memref.load %input[%c0] : memref<4xf32>
%exp0 = math.exp %val0 : f32
memref.store %exp0, %output[%c0] : memref<4xf32>
%val1 = memref.load %input[%c1] : memref<4xf32>
%exp1 = math.exp %val1 : f32
memref.store %exp1, %output[%c1] : memref<4xf32>
%val2 = memref.load %input[%c2] : memref<4xf32>
%exp2 = math.exp %val2 : f32
memref.store %exp2, %output[%c2] : memref<4xf32>
%val3 = memref.load %input[%c3] : memref<4xf32>
%exp3 = math.exp %val3 : f32
memref.store %exp3, %output[%c3] : memref<4xf32>
```

The unroll factor is chosen based on:
- The loop trip count (only unroll if the trip count is small and known at compile time).
- The register pressure (unrolling increases register usage).
- The GPU's instruction cache size.

### 16.7.7 Stage 7: Conversion to LLVM

The final stage converts the optimized MLIR to LLVM IR dialect, which is then translated to LLVM IR and compiled to PTX:

```mlir
// After LLVM conversion:
llvm.func @fusion(%input: !llvm.ptr<f32>, %bias: !llvm.ptr<f32>, %output: !llvm.ptr<f32>, %n: i64) {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64

  // Compute thread ID
  %tid = llvm.call @llvm.nvvm.read.ptx.sreg.tid.x() : () -> i32
  %ctaid = llvm.call @llvm.nvvm.read.ptx.sreg.ctaid.x() : () -> i32
  %ntid = llvm.call @llvm.nvvm.read.ptx.sreg.ntid.x() : () -> i32
  %gid = llvm.add %tid, llvm.mul %ctaid, %ntid : i32
  %gid64 = llvm.zext %gid : i32 to i64

  // Bounds check
  %cond = llvm.icmp "ult" %gid64, %n : i1
  llvm.cond_br %cond, ^body, ^exit

^body:
  // Load input
  %input_ptr = llvm.getelementptr %input[%gid64] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %val_in = llvm.load %input_ptr : !llvm.ptr<f32> -> f32

  // exp
  %exp_val = llvm.call @llvm.exp.f32(%val_in) : (f32) -> f32

  // Load bias
  %bias_ptr = llvm.getelementptr %bias[%gid64] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %val_bias = llvm.load %bias_ptr : !llvm.ptr<f32> -> f32

  // Add
  %sum = llvm.fadd %exp_val, %val_bias : f32

  // Store
  %output_ptr = llvm.getelementptr %output[%gid64] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  llvm.store %sum, %output_ptr : f32, !llvm.ptr<f32>

  llvm.br ^exit

^exit:
  llvm.return
}
```

---

## 16.8 Transpose Emitter

The transpose emitter handles fusions where the hero is a non-trivial transpose operation. The key optimization is using shared memory to ensure coalesced memory access on both the read and write sides.

### 16.8.1 The Problem with Naive Transpose

A naive transpose kernel reads input elements in the order of the output layout, which means non-contiguous memory access on the input:

```
Input layout (row-major): [M, N]
Output layout (row-major): [N, M] (transposed)

Naive approach:
  For each output element [i, j]:
    Read input[j, i]  // Non-contiguous access! Input element is at offset j*N+i

  If thread 0 reads [0,0], thread 1 reads [0,1], etc.
  Input offsets: 0, N, 2N, 3N, ...  (stride N, non-coalesced!)
```

Non-coalesced memory access is extremely inefficient on GPUs because each memory transaction fetches an entire cache line (128 bytes). When threads access non-contiguous addresses, most of the fetched data is wasted.

### 16.8.2 Two-Phase Shared Memory Approach

The transpose emitter uses a two-phase approach with shared memory:

**Phase 1: Coalesced read from global memory to shared memory.**
Each thread reads a contiguous element from the input and stores it in shared memory:

```
// Tile size: TILE_SIZE x TILE_SIZE (e.g., 32x32)
// Each block processes one tile

// Phase 1: Coalesced read
shared_memory[threadIdx.y][threadIdx.x] = input[row * N + col];
// row = blockIdx.y * TILE_SIZE + threadIdx.y
// col = blockIdx.x * TILE_SIZE + threadIdx.x
// Access pattern: contiguous along threadIdx.x -> coalesced!
```

**Phase 2: Synchronize, then coalesced write from shared memory to global memory.**
After all threads have written to shared memory, synchronize (`__syncthreads()`). Then read from shared memory in transposed order and write to global memory:

```
// Phase 2: Coalesced write (with transposition in shared memory)
output[col * M + row] = shared_memory[threadIdx.x][threadIdx.y];
// Note: threadIdx.x and threadIdx.y are swapped!
// This transposes the tile.
// Access pattern: contiguous along threadIdx.x -> coalesced!
```

### 16.8.3 Shared Memory Bank Conflicts

A critical optimization is avoiding shared memory bank conflicts. GPU shared memory is divided into 32 banks, and simultaneous accesses to the same bank by different threads are serialized. Reading along a row of shared memory is bank-conflict-free (each element is in a different bank), but reading along a column causes bank conflicts.

**Solution: Pad each row of shared memory by one element:**

```
// Without padding:
__shared__ float tile[32][32];  // Reading column: all elements in same bank!

// With padding:
__shared__ float tile[32][33];  // Extra element breaks the bank conflict pattern
// Column access: tile[0][0], tile[1][0], tile[2][0], ...
// Bank: 0, 0, 0, ... -> CONFLICT
// With padding: tile[0][0], tile[1][0], tile[2][0], ...
// Bank: 0*33%32=0, 1*33%32=1, 2*33%32=2, ... -> NO CONFLICT!
```

The XLA transpose emitter automatically applies shared memory padding based on the tile size and data type.

### 16.8.4 Thread Synchronization

The transpose emitter inserts `__syncthreads()` barriers between the read and write phases. This ensures that all threads have finished writing to shared memory before any thread reads from it:

```
Phase 1: Read from global -> shared memory
__syncthreads()  // Wait for all threads
Phase 2: Read from shared memory (transposed) -> global memory
```

Without this barrier, a thread might read shared memory locations that other threads haven't written yet, resulting in incorrect results.

### 16.8.5 Fusing Elementwise Operations

The transpose emitter can fuse elementwise operations before and after the transpose:

```
// Fusion pattern:
%0 = exp(%input)
%1 = transpose(%0, {1, 0})
%2 = add(%1, %bias)

// Emitted kernel:
Phase 1: Read input -> apply exp -> store in shared memory
__syncthreads()
Phase 2: Read shared memory (transposed) -> apply add -> write to output
```

This eliminates the need for separate kernels for `exp` and `add`, saving two global memory reads and writes.

---

## 16.9 Complete GELU Example

This section provides a complete walkthrough of the HLO -> MLIR -> LLVM IR -> PTX pipeline for a GELU activation fusion.

### 16.9.1 HLO Input

```
%fused_computation (param_0: f32[128,256], param_1: f32[256]) -> f32[128,256] {
  %param_0 = parameter(0), f32[128,256]
  %param_1 = parameter(1), f32[256]

  // Bias addition
  %broadcast = broadcast(%param_1), dimensions={1}
  %add = add(%param_0, %broadcast)

  // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
  %half = constant(0.5)
  %sqrt2 = constant(1.41421356237)
  %bcast_half = broadcast(%half), dimensions={}
  %bcast_sqrt2 = broadcast(%sqrt2), dimensions={}

  %mul_half = multiply(%add, %bcast_half)
  %div = divide(%add, %bcast_sqrt2)
  %erf = erf(%div)
  %one = constant(1.0)
  %bcast_one = broadcast(%one), dimensions={}
  %add_one = add(%bcast_one, %erf)

  ROOT %gelu = multiply(%mul_half, %add_one)
}
```

### 16.9.2 Step 1: Hero Selection

The fusion contains only elementwise operations (add, multiply, divide, erf, broadcast, constant). The hero is a **loop** emitter.

### 16.9.3 Step 2: Partitioning

All operations are elementwise and compatible with the loop emitter. No partitioning is needed -- the entire fusion is a single partition.

### 16.9.4 Step 3: MLIR Generation (xla_gpu dialect)

```mlir
func.func @gelu_fusion(
  %input: tensor<128x256xf32>,
  %bias: tensor<256xf32>
) -> tensor<128x256xf32> {
  %result = xla_gpu.loop over tensor<128x256xf32> iter_args(%output = %input) -> tensor<128x256xf32> {
    ^bb0(%row: index, %col: index, %out: tensor<128x256xf32>):
      // Load input element
      %val = tensor.extract %input[%row, %col] : tensor<128x256xf32>

      // Load bias element (bias is [256], indexed by col only)
      %b = tensor.extract %bias[%col] : tensor<256xf32>

      // add
      %sum = arith.addf %val, %b : f32

      // GELU computation
      %c0_5 = arith.constant 0.5 : f32
      %c_sqrt2 = arith.constant 1.41421356 : f32
      %c1 = arith.constant 1.0 : f32

      %mul_half = arith.mulf %sum, %c0_5 : f32
      %div = arith.divf %sum, %c_sqrt2 : f32
      %erf_val = math.erf %div : f32
      %add_one = arith.addf %c1, %erf_val : f32
      %gelu = arith.mulf %mul_half, %add_one : f32

      xla_gpu.yield %gelu : f32
  }
  return %result : tensor<128x256xf32>
}
```

### 16.9.5 Step 4: Inlining

No function calls to inline in this simple case.

### 16.9.6 Step 5: xla_gpu to scf Conversion

```mlir
func.func @gelu_fusion(
  %input: memref<32768xf32>,
  %bias: memref<256xf32>,
  %output: memref<32768xf32>
) {
  %c0 = arith.constant 0 : index
  %c32768 = arith.constant 32768 : index
  %c1 = arith.constant 1 : index
  %c256 = arith.constant 256 : index

  scf.for %tid = %c0 to %c32768 step %c1 {
    // Load input
    %val = memref.load %input[%tid] : memref<32768xf32>

    // Load bias (bias index = tid % 256)
    %col = arith.remui %tid, %c256 : index
    %b = memref.load %bias[%col] : memref<256xf32>

    // Compute GELU
    %sum = arith.addf %val, %b : f32
    %c0_5 = arith.constant 0.5 : f32
    %c_sqrt2 = arith.constant 1.41421356 : f32
    %c1 = arith.constant 1.0 : f32

    %mul_half = arith.mulf %sum, %c0_5 : f32
    %div = arith.divf %sum, %c_sqrt2 : f32
    %erf_val = math.erf %div : f32
    %add_one = arith.addf %c1, %erf_val : f32
    %gelu = arith.mulf %mul_half, %add_one : f32

    memref.store %gelu, %output[%tid] : memref<32768xf32>
  }
}
```

### 16.9.7 Step 6: Vectorization

```mlir
// With vector factor 4:
func.func @gelu_fusion_vectorized(
  %input: memref<32768xf32>,
  %bias: memref<256xf32>,
  %output: memref<32768xf32>
) {
  %c0 = arith.constant 0 : index
  %c8192 = arith.constant 8192 : index  // 32768 / 4
  %c4 = arith.constant 4 : index

  scf.for %tid = %c0 to %c8192 step %c1 {
    %base = arith.muli %tid, %c4 : index

    // Vector load (4 elements)
    %val = vector.load %input[%base] : memref<32768xf32>, vector<4xf32>

    // Vector load bias (4 elements)
    %b = vector.load %bias[%base] : memref<256xf32>, vector<4xf32>

    // Vector GELU
    %sum = arith.addf %val, %b : vector<4xf32>
    %c0_5 = arith.constant dense<0.5> : vector<4xf32>
    %c_sqrt2 = arith.constant dense<1.41421356> : vector<4xf32>
    %c1 = arith.constant dense<1.0> : vector<4xf32>

    %mul_half = arith.mulf %sum, %c0_5 : vector<4xf32>
    %div = arith.divf %sum, %c_sqrt2 : vector<4xf32>
    %erf_val = math.erf %div : vector<4xf32>
    %add_one = arith.addf %c1, %erf_val : vector<4xf32>
    %gelu = arith.mulf %mul_half, %add_one : vector<4xf32>

    vector.store %gelu, %output[%base] : memref<32768xf32>, vector<4xf32>
  }
}
```

### 16.9.8 Step 7: Convert to LLVM IR and PTX

The final LLVM IR (simplified):

```llvm
define void @gelu_fusion(
  ptr %input, ptr %bias, ptr %output, i64 %n
) #0 {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %gid = add i32 %tid, mul(i32 %ctaid, i32 %ntid)
  %gid64 = zext i32 %gid to i64

  %total = mul i64 128, 256
  %cond = icmp ult i64 %gid64, %total
  br i1 %cond, label %body, label %exit

body:
  ; Load input[gid]
  %in_ptr = getelementptr float, ptr %input, i64 %gid64
  %val = load float, ptr %in_ptr

  ; Load bias[gid % 256]
  %col = urem i64 %gid64, 256
  %bias_ptr = getelementptr float, ptr %bias, i64 %col
  %b = load float, ptr %bias_ptr

  ; add
  %sum = fadd float %val, %b

  ; GELU
  %half = fmul float %sum, 0x3FE0000000000000  ; 0.5
  %sqrt2_inv = fdiv float %sum, 0x3FF6A09E667F3BCD  ; 1/sqrt(2)
  %erf_val = call float @llvm.erf.f32(float %sqrt2_inv)
  %one_plus = fadd float 1.0, %erf_val
  %gelu = fmul float %half, %one_plus

  ; Store output[gid]
  %out_ptr = getelementptr float, ptr %output, i64 %gid64
  store float %gelu, ptr %out_ptr
  br label %exit

exit:
  ret void
}
```

The PTX output (highly simplified):

```ptx
.visible .entry gelu_fusion(
  .param .u64 input,
  .param .u64 bias,
  .param .u64 output,
  .param .u64 n
) {
  .reg .f32 %f<10>;
  .reg .u32 %r<5>;
  .reg .u64 %rd<10>;
  .reg .pred %p;

  // Compute thread ID
  ld.param.u64 %rd1, [input];
  ld.param.u64 %rd2, [bias];
  ld.param.u64 %rd3, [output];
  mov.u32 %r1, %tid.x;
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mad.lo.u32 %r4, %r2, %r3, %r1;  // gid = ctaid * ntid + tid

  // Bounds check
  cvt.u64.u32 %rd4, %r4;
  setp.lt.u64 %p, %rd4, 32768;
  @!p bra exit;

  // Load input[gid]
  shl.b64 %rd5, %rd4, 2;           // gid * 4 (sizeof float)
  add.u64 %rd6, %rd1, %rd5;
  ld.global.f32 %f1, [%rd6];

  // Load bias[gid % 256]
  rem.u64 %rd7, %rd4, 256;
  shl.b64 %rd8, %rd7, 2;
  add.u64 %rd9, %rd2, %rd8;
  ld.global.f32 %f2, [%rd9];

  // add
  add.f32 %f3, %f1, %f2;

  // GELU computation
  mul.f32 %f4, %f3, 0f3F000000;     // * 0.5
  div.f32 %f5, %f3, 0f3FB504F3;     // / sqrt(2)
  // erf approximation (expanded by LLVM)
  ...
  add.f32 %f8, 0f3F800000, %f7;     // 1 + erf
  mul.f32 %f9, %f4, %f8;            // * 0.5 * (1 + erf)

  // Store output[gid]
  add.u64 %rd10, %rd3, %rd5;
  st.global.f32 [%rd10], %f9;

exit:
  ret;
}
```

This complete example shows how a GELU fusion goes from HLO through MLIR, LLVM IR, and finally PTX, with each stage applying specific transformations and optimizations.

---

## 16.10 Emitter Configuration Flags

| Flag | Description |
|------|-------------|
| `--xla_gpu_max_kernel_unroll_factor` | Maximum loop unroll factor for generated kernels |
| `--xla_gpu_enable_vectorization` | Enable/disable vectorization of memory operations |
| `--xla_gpu_emit_loop_fusion` | Force loop emitter for all fusions |
| `--xla_gpu_triton_gemm` | Enable Triton-based GEMM emission |
| `--xla_gpu_enable_triton_softmax_fusion` | Enable Triton softmax fusion emitter |
| `--xla_gpu_enable_triton_flash_attention` | Enable Triton flash attention emitter |
