# XLA Reference - Chapter 17: CPU Backend

This reference provides comprehensive documentation on XLA's CPU backend, covering the architecture, code generation pipeline, CPU-specific optimizations, Eigen integration, supported instruction set architectures, and build configuration.

---

## 17.1 Architecture

The XLA CPU backend compiles HLO programs into native machine code that executes on the host CPU. It uses LLVM as its code generation backend, leveraging LLVM's mature support for a wide range of CPU architectures and instruction sets.

### 17.1.1 LLVM-Based Code Generation

The CPU backend generates code through the LLVM compiler infrastructure. The compilation pipeline converts HLO operations to LLVM IR, which LLVM then compiles to native machine code for the target CPU:

```
HLO Module (from frontend)
    |
    v
[1. HLO Optimization Pipeline]
    - Same hardware-independent passes as GPU/TPU
    - Algebraic simplification, constant folding, DCE
    - Fusion, layout assignment, rematerialization
    - CPU-specific passes (ConvCanonicalization, ParallelTaskAssigner)
    |
    v
[2. HLO -> LLVM IR Conversion]
    - Each HLO operation is lowered to LLVM IR
    - Elementwise ops -> LLVM vector instructions
    - Dot/Conv -> calls to Eigen routines
    - Reduce -> LLVM loop with SIMD reduction
    |
    v
[3. LLVM Optimization]
    - Standard LLVM optimization passes
    - Loop vectorization
    - SLP vectorization
    - Interprocedural optimization
    |
    v
[4. LLVM Backend]
    - Instruction selection for target ISA
    - Register allocation
    - Instruction scheduling
    |
    v
[5. Native Object Code]
    - ELF object file (Linux)
    - Mach-O object file (macOS)
    - COFF object file (Windows)
```

The LLVM IR generation is handled by the `IrEmitter` class, which traverses the HLO computation graph and emits LLVM IR for each instruction. The emitted IR is organized into functions, with one function per HLO computation.

### 17.1.2 JIT Compilation

In JIT (Just-In-Time) mode, the CPU backend compiles HLO modules at runtime, immediately before execution. The JIT compilation flow:

1. **Module construction**: The frontend (JAX, TensorFlow) constructs an HLO module.
2. **Optimization**: The HLO optimization pipeline runs on the module.
3. **LLVM IR generation**: The optimized HLO is converted to an LLVM module.
4. **LLVM JIT compilation**: The LLVM module is compiled using LLVM's JIT engine (ORC JIT v2).
5. **Execution**: The compiled function is loaded into memory and executed.

The JIT mode is the default for interactive use (e.g., Jupyter notebooks, REPL) and development. It provides fast compilation times but does not persist the compiled code across process restarts.

**LLVM ORC JIT v2**: The CPU backend uses LLVM's ORC JIT v2, which provides:
- **Lazy compilation**: Functions are compiled only when first called.
- **Concurrent compilation**: Multiple functions can be compiled in parallel.
- **Resource tracking**: Compiled code and associated resources are automatically freed when no longer needed.
- **Symbol resolution**: Automatic resolution of symbols from the host process and shared libraries.

### 17.1.3 AOT Compilation

In AOT (Ahead-Of-Time) mode, the CPU backend compiles HLO modules to native object files that can be linked into a standalone executable or shared library:

```
# Compile HLO module to object file
xla_aot_compile \
  --input_module=program.hlo \
  --target=cpu \
  --output=/tmp/program.o

# Link into executable
g++ /tmp/program.o -o my_program -lxla_runtime -lpthread
```

**AOT compilation advantages:**
- **Startup time**: No JIT compilation overhead at runtime.
- **Deployment**: Compiled code can be distributed without the XLA compiler.
- **Determinism**: The same binary is executed every time.
- **Cross-compilation**: Code can be compiled for a different target architecture.

**AOT compilation metadata:**

The compiled object file includes metadata that describes:
- Function signatures (input/output shapes and types).
- Buffer allocation requirements (sizes and alignments).
- Constant data (embedded in the object file).
- Entry point names and calling conventions.

The AOT runtime (`xla_runtime`) provides the execution infrastructure:
- Buffer management (allocation, deallocation).
- Thread pool management for parallel execution.
- Profiling hooks.

---

## 17.2 CPU-Specific Optimizations

The CPU backend applies several optimizations that are specific to CPU execution characteristics.

### 17.2.1 ConvCanonicalization

`ConvCanonicalization` is a CPU-specific pass that rewrites convolution operations into a canonical form suitable for Eigen's convolution implementation. Eigen is a C++ template library for linear algebra that provides highly optimized implementations of matrix operations and convolutions.

**What ConvCanonicalization does:**

1. **Dimension ordering**: Ensures that convolution dimensions are in NHWC format (batch, height, width, channels), which is the format expected by Eigen:

   ```
   // Input: NCHW format
   %input = parameter(0), f32[batch, channels, height, width]

   // After canonicalization: NHWC format
   %transposed = transpose(%input, {0, 2, 3, 1}), f32[batch, height, width, channels]
   %conv = convolution(%transposed, %filter), ...
   ```

2. **Padding normalization**: Converts padding specifications to Eigen's format. Eigen uses symmetric padding (same amount on both sides of each dimension), so asymmetric padding is handled by inserting explicit pad operations:

   ```
   // Asymmetric padding: {top=1, bottom=2, left=1, right=2}
   %padded = pad(%input, 0, {{0,0}, {1,2}, {1,2}, {0,0}})
   %conv = convolution(%padded, %filter), padding={{0,0}, {0,0}, {0,0}, {0,0}}
   ```

3. **Dilation handling**: When Eigen does not natively support the required dilation configuration, the pass decomposes dilated convolutions:

   ```
   // Dilated convolution with dilation [2, 2]:
   %dilated_filter = insert_zeros(%filter, [2, 2])  // Insert zeros between filter elements
   %conv = convolution(%input, %dilated_filter)
   ```

4. **Grouped convolution decomposition**: For grouped convolutions where Eigen lacks direct support, the pass decomposes the operation into multiple smaller convolutions:

   ```
   // Grouped convolution with 4 groups:
   // Input: [N, H, W, C_in], Filter: [H_f, W_f, C_in/groups, C_out/groups]
   %group_0 = convolution(%input[:,:,:,:C_in/4], %filter[:,:,:C_in/4, :C_out/4])
   %group_1 = convolution(%input[:,:,:,C_in/4:C_in/2], %filter[:,:,:,C_in/4:C_in/2, :C_out/4])
   // ... groups 2, 3
   %result = concatenate(%group_0, %group_1, %group_2, %group_3), dimension=3
   ```

### 17.2.2 ParallelTaskAssigner

`ParallelTaskAssigner` is a CPU-specific pass that identifies opportunities for parallel execution and assigns instructions to threads. This enables multi-threaded execution of HLO programs on multi-core CPUs.

**Algorithm:**

1. **Cost estimation**: For each instruction, estimate the computational cost using `HloCostAnalysis`. The cost model accounts for:
   - FLOP count.
   - Memory access volume (bytes read/written).
   - Memory hierarchy effects (cache-friendly access patterns cost less).
   - Operation type (elementwise, reduction, dot, etc.).

2. **Parallelism boundary identification**: Identify which instructions can execute in parallel:
   - Instructions with no data dependency can execute in parallel.
   - Different branches of a conditional can execute in parallel.
   - Different iterations of a loop can execute in parallel if they are independent.

3. **Task decomposition**: Decompose expensive operations into parallel sub-tasks:
   - **Elementwise operations**: Partition the output into chunks, one per thread.
   - **Reductions**: Partition the reduction dimension into chunks, then combine partial results.
   - **Dot products**: Partition the matrix multiplication along the output dimensions.
   - **Convolutions**: Partition along the batch and/or spatial dimensions.

4. **Thread assignment**: Assign tasks to threads using a work-stealing scheduler:
   - Initially, tasks are distributed evenly across threads.
   - If a thread finishes early, it steals tasks from other threads' queues.
   - This balances the load even when tasks have unpredictable execution times.

**Configuration:**

```bash
# Set the number of threads for CPU execution
XLA_FLAGS="--xla_cpu_parallel_strategy=parallel" \
  OMP_NUM_THREADS=8 \
  python my_program.py

# Disable parallel execution
XLA_FLAGS="--xla_cpu_parallel_strategy=sequential" python my_program.py
```

**Implementation details:**

The parallel execution uses a thread pool managed by XLA's runtime. Each parallel task is a function that:
1. Computes the sub-range of the output that this task is responsible for.
2. Executes the computation for that sub-range.
3. Synchronizes with other tasks at join points (e.g., when a reduction needs to combine partial results).

```cpp
// Simplified parallel task execution:
void ExecuteParallel(HloInstruction* instruction, ThreadPool* pool) {
  auto cost = CostAnalysis::GetCost(instruction);
  int64_t num_tasks = std::min(cost / min_cost_per_task, pool->num_threads());

  // Partition the output
  auto partitions = PartitionOutput(instruction->shape(), num_tasks);

  // Launch parallel tasks
  std::vector<std::future<void>> futures;
  for (int i = 0; i < num_tasks; ++i) {
    futures.push_back(pool->Schedule([&]() {
      ExecutePartition(instruction, partitions[i]);
    }));
  }

  // Wait for all tasks to complete
  for (auto& f : futures) {
    f.wait();
  }
}
```

### 17.2.3 Vectorization via LLVM

The CPU backend relies heavily on LLVM's auto-vectorization passes to generate SIMD (Single Instruction, Multiple Data) code. LLVM provides two vectorization approaches:

**Loop vectorization**: Identifies loops where multiple iterations can be executed simultaneously using SIMD instructions:

```c
// Before vectorization:
for (int i = 0; i < N; ++i) {
  output[i] = exp(input[i]) + bias[i];
}

// After vectorization (AVX2, 8 floats per vector):
for (int i = 0; i < N; i += 8) {
  __m256 v = _mm256_load_ps(&input[i]);
  __m256 e = _mm256_exp_ps(v);          // 8-element exp
  __m256 b = _mm256_load_ps(&bias[i]);
  __m256 r = _mm256_add_ps(e, b);
  _mm256_store_ps(&output[i], r);
}
```

**SLP (Superword-Level Parallelism) vectorization**: Identifies independent operations within a basic block that can be combined into SIMD operations:

```c
// Before SLP vectorization:
float a0 = input[0] + bias[0];
float a1 = input[1] + bias[1];
float a2 = input[2] + bias[2];
float a3 = input[3] + bias[3];

// After SLP vectorization:
__m128 v_in = _mm_load_ps(input);
__m128 v_bias = _mm_load_ps(bias);
__m128 v_out = _mm_add_ps(v_in, v_bias);
_mm_store_ps(output, v_out);
```

**Vectorization challenges on CPU:**

1. **Alignment**: SIMD load/store instructions perform best with aligned addresses (16-byte for SSE, 32-byte for AVX, 64-byte for AVX-512). The CPU backend ensures that buffers are allocated with proper alignment.

2. **Remainder handling**: When the data size is not a multiple of the SIMD width, the remaining elements must be handled separately (scalar loop or masked operations).

3. **Gather/Scatter**: Random access patterns (gather, scatter) are expensive on CPUs. LLVM generates gather/scatter instructions on hardware that supports them (AVX-512) or falls back to scalar loads/stores.

4. **Reduction**: SIMD reductions require horizontal operations (e.g., `_mm256_hadd_ps`) that combine partial results across SIMD lanes.

---

## 17.3 Eigen Integration

The CPU backend uses the Eigen library for optimized implementations of matrix multiplication and convolution operations. Eigen is a high-performance C++ template library that provides:

- **Matrix operations**: General matrix multiplication (GEMM), triangular solves, matrix decompositions.
- **Tensor operations**: Convolutions, pooling, tensor contractions.
- **SIMD abstraction**: Portable SIMD operations across different CPU architectures.

### 17.3.1 Matrix Multiplication

HLO dot-product operations are lowered to Eigen matrix multiplications. The lowering handles:

1. **Batched matmul**: HLO dot operations with batch dimensions are decomposed into batched Eigen calls:

   ```
   // HLO: dot(%a, %b), lhs_batch_dims={0}, rhs_batch_dims={0},
   //       lhs_contracting_dims={2}, rhs_contracting_dims={1}
   // Shapes: a[batch, M, K], b[batch, K, N]

   // Lowered to:
   for (int b = 0; b < batch; ++b) {
     EigenMatrixMultiply(a[b], b[b], output[b]);
   }
   ```

2. **Transpose folding**: If the dot product includes transposes of its operands, these are folded into the Eigen call by selecting the appropriate Eigen matrix type:

   ```cpp
   // If rhs is transposed:
   // Eigen: output.noalias() = lhs * rhs.transpose();
   // vs. creating an explicit transpose buffer:
   // Eigen: auto rhs_t = rhs.transpose(); output = lhs * rhs_t;
   ```

3. **Contraction detection**: General HLO contractions (einsum-like operations) are detected and mapped to Eigen tensor contractions:

   ```cpp
   // HLO contraction: output[i,j] = sum_k input1[i,k] * input2[k,j]
   // Eigen:
   // output.device(dev) = input1.contract(input2, dims);
   ```

4. **Data type handling**: Eigen supports float32, float64, int32, and other data types. The lowering selects the appropriate Eigen type based on the HLO element type:

   ```cpp
   template <typename T>
   void EigenMatMul(const T* a, const T* b, T* c, int M, int K, int N) {
     auto A = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(a, M, K);
     auto B = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(b, K, N);
     auto C = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(c, M, N);
     C.noalias() = A * B;
   }
   ```

### 17.3.2 Convolution Operations

HLO convolution operations are lowered to Eigen's tensor convolution operations:

1. **Forward convolution**: The input and filter are reshaped to Eigen tensor format, and Eigen's `convolve()` method is called:

   ```cpp
   Eigen::Tensor<float, 4> input(batch, height, width, channels);
   Eigen::Tensor<float, 4> filter(filter_h, filter_w, in_channels, out_channels);

   // Eigen convolution
   Eigen::array<ptrdiff_t, 2> dims({1, 2}); // Spatial dimensions
   auto output = input.convolve(filter, dims);
   ```

2. **Backward data convolution**: Implemented as a convolution with flipped filter:

   ```cpp
   // Backward data: gradient w.r.t. input
   // Equivalent to convolving the output gradient with the flipped filter
   auto flipped_filter = filter.reverse({0, 1});
   auto input_grad = output_grad.convolve(flipped_filter, dims);
   ```

3. **Backward filter convolution**: Implemented as a convolution of input with output gradient:

   ```cpp
   // Backward filter: gradient w.r.t. filter
   // Equivalent to convolving the input with the output gradient
   auto filter_grad = input.convolve(output_grad, dims);
   ```

### 17.3.3 Eigen Thread Pool

The CPU backend uses Eigen's thread pool for parallel execution of Eigen operations. The thread pool is created during runtime initialization and shared across all operations:

```cpp
// Thread pool initialization
int num_threads = std::thread::hardware_concurrency();
Eigen::ThreadPool thread_pool(num_threads);
Eigen::ThreadPoolDevice device(&thread_pool, num_threads);

// Using the device for parallel operations
output.device(device) = input1.contract(input2, dims);
```

The thread pool uses a work-stealing algorithm for load balancing and supports nested parallelism (operations within operations can use the same thread pool).

---

## 17.4 Supported CPU ISAs

The CPU backend can generate code for a wide range of instruction set architectures through LLVM's target support.

### 17.4.1 x86 Architecture

**SSE (Streaming SIMD Extensions)**:
- Available on all x86-64 processors.
- 128-bit registers (XMM0-XMM15).
- 4 single-precision floats or 2 double-precision floats per operation.
- Operations: add, sub, mul, div, sqrt, min, max, compare, bitwise.
- Baseline for all x86-64 code generation.

**AVX (Advanced Vector Extensions)**:
- Available on Intel Sandy Bridge (2011) and later, AMD Bulldozer and later.
- 256-bit registers (YMM0-YMM15).
- 8 single-precision floats or 4 double-precision floats per operation.
- Fused multiply-add (FMA3) on some processors.
- Includes VEX-encoded versions of all SSE instructions.

**AVX2**:
- Available on Intel Haswell (2013) and later, AMD Excavator and later.
- Extends AVX with integer operations and gather instructions.
- 256-bit integer SIMD operations.
- FMA3 (Fused Multiply-Add) is standard.
- Most common target for modern x86 CPUs.

**AVX-512**:
- Available on Intel Skylake-X (2017), Ice Lake (2019), and later. AMD Zen 4 and later.
- 512-bit registers (ZMM0-ZMM31).
- 16 single-precision floats or 8 double-precision floats per operation.
- Mask registers (k0-k7) for predicated operations.
- Gather/Scatter instructions.
- AVX-512BW, AVX-512DQ, AVX-512VL, AVX-512VNNI (vector neural network instructions).
- AVX-512BF16 (bfloat16 support) on Cooper Lake and later.

**Detection and dispatch:**

The CPU backend detects the available ISA at runtime using CPUID and generates code for the highest available ISA:

```cpp
// ISA detection (simplified)
bool has_avx2 = __builtin_cpu_supports("avx2");
bool has_avx512f = __builtin_cpu_supports("avx512f");
bool has_avx512bw = __builtin_cpu_supports("avx512bw");

if (has_avx512bw) {
  // Use AVX-512 code path (32 float elements per vector op)
} else if (has_avx512f) {
  // Use AVX-512F code path
} else if (has_avx2) {
  // Use AVX2 code path (8 float elements per vector op)
} else {
  // Use SSE code path (4 float elements per vector op)
}
```

For AOT compilation, the target ISA is specified via LLVM target features:

```bash
# Compile for AVX2
xla_aot_compile --target=cpu --cpu=haswell --host_cpu_features="+avx2,+fma"

# Compile for AVX-512
xla_aot_compile --target=cpu --cpu=skylake-avx512 --host_cpu_features="+avx512f,+avx512bw"
```

### 17.4.2 ARM Architecture

**NEON**:
- Available on all ARMv7-A and ARMv8-A processors.
- 128-bit registers (Q0-Q31 / D0-D31).
- 4 single-precision floats per operation.
- ARMv8-A provides 32 128-bit registers (vs. 16 on ARMv7).
- Operations: add, sub, mul, fused multiply-add, min, max, compare.

**SVE (Scalable Vector Extension)**:
- Available on ARMv8.2-A and later (e.g., AWS Graviton3, Fujitsu A64FX).
- Variable-length vectors (128 to 2048 bits, determined by hardware).
- Predicate registers for masked operations.
- Gather/Scatter instructions.
- Per-lane predication.

**Detection and dispatch:**

```cpp
// ARM ISA detection
#if defined(__ARM_NEON)
  // NEON is available
  // Use vld1q_f32, vaddq_f32, etc.
#endif

#if defined(__ARM_FEATURE_SVE)
  // SVE is available
  // Use svld1_f32, svadd_f32_z, etc.
#endif
```

For cross-compilation targeting ARM:

```bash
# Compile for ARM NEON
xla_aot_compile --target=cpu --target_triple=aarch64-linux-gnu --host_cpu_features="+neon"

# Compile for ARM SVE
xla_aot_compile --target=cpu --target_triple=aarch64-linux-gnu --host_cpu_features="+sve"
```

### 17.4.3 Other Architectures

The CPU backend also supports (through LLVM):

- **PowerPC (PPC64LE)**: Altivec/VSX SIMD instructions.
- **RISC-V**: Vector extension (V extension).
- **WebAssembly (Wasm)**: SIMD128 instructions.

These architectures are less commonly used with XLA but are supported through LLVM's target infrastructure.

---

## 17.5 Build and Configuration

### 17.5.1 Backend Selection

The CPU backend is selected using the `--backend=cpu` flag:

```bash
# Using XLA directly
xla_compiler --backend=cpu --input_module=program.hlo

# Using JAX with CPU backend
import jax
jax.config.update('jax_platforms', 'cpu')

# Using TensorFlow with CPU
tf.config.set_visible_devices([], 'GPU')  # Hide GPUs -> use CPU
```

### 17.5.2 LLVM Triple Configuration

The target triple specifies the CPU architecture, vendor, operating system, and ABI:

```bash
# Common target triples:
# x86_64 Linux:    x86_64-unknown-linux-gnu
# x86_64 macOS:    x86_64-apple-darwin
# x86_64 Windows:  x86_64-pc-windows-msvc
# ARM64 Linux:     aarch64-unknown-linux-gnu
# ARM64 macOS:     arm64-apple-darwin

# Specify target triple
XLA_FLAGS="--xla_cpu_target_triple=x86_64-unknown-linux-gnu" python my_program.py
```

### 17.5.3 CPU Feature Flags

Control which CPU features are used during code generation:

```bash
# Enable specific CPU features
XLA_FLAGS="--xla_cpu_host_cpu_features=+avx2,+fma,+avx512f" python my_program.py

# Disable specific features
XLA_FLAGS="--xla_cpu_host_cpu_features=+sse4.2,-avx,-avx2" python my_program.py

# Use the host CPU's features (default)
XLA_FLAGS="--xla_cpu_use_host_cpu_features=true" python my_program.py
```

### 17.5.4 Compilation Flags

| Flag | Description |
|------|-------------|
| `--xla_cpu_parallel_strategy` | Parallel execution strategy: `sequential`, `parallel` |
| `--xla_cpu_num_threads` | Number of threads for parallel execution |
| `--xla_cpu_target_triple` | LLVM target triple |
| `--xla_cpu_host_cpu_features` | CPU features to enable/disable |
| `--xla_cpu_use_host_cpu_features` | Auto-detect CPU features |
| `--xla_cpu_opt_level` | LLVM optimization level (O0, O1, O2, O3) |
| `--xla_cpu_intrinsic_barrier_limit` | Limit for LLVM intrinsic expansion |
| `--xla_cpu_llvm_clang_opt_level` | Clang optimization level for AOT |
| `--xla_cpu_max_kernel_unroll_factor` | Maximum loop unroll factor |
| `--xla_cpu_enable_fast_math` | Enable LLVM fast-math flags |
| `--xla_cpu_enable_xla_sm` | Enable XLA shared memory optimizations |

### 17.5.5 Building XLA with CPU Backend

```bash
# Build XLA with CPU backend support (Bazel)
bazel build //xla:all --config=cpu

# Build for a specific target architecture (cross-compilation)
bazel build //xla:all --config=cpu --config=aarch64

# Run CPU backend tests
bazel test //xla/service/cpu:all
bazel test //xla/tests:cpu_backend_test
```

### 17.5.6 Memory Allocation

The CPU backend uses a simple memory allocator that allocates buffers from the host process's heap:

- **Buffer alignment**: Buffers are aligned to the natural alignment of the data type (e.g., 16 bytes for float32 vectors).
- **Memory planning**: The buffer assignment pass determines the allocation plan, and the runtime allocates all buffers before execution begins.
- **Temporary buffers**: Scratch space for operations that require it is allocated from a separate pool.

```cpp
// CPU buffer allocation
class CpuBufferAllocator : public BufferAllocator {
  StatusOr<BorrowingSlice> AllocateBytes(int64_t byte_size,
                                          int64_t alignment) override {
    void* ptr = absl::aligned_malloc(byte_size, alignment);
    return BorrowingSlice(ptr, byte_size);
  }

  void DeallocateBytes(BorrowingSlice slice) override {
    absl::aligned_free(slice.data());
  }
};
```

### 17.5.7 Debugging the CPU Backend

**LLVM IR dumping:**

```bash
# Dump LLVM IR for each computation
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_ir" python my_program.py

# The dump directory contains:
# - module_name.before_optimizations.ll
# - module_name.after_optimizations.ll
# - module_name.optimized.ll
```

**Disassembly:**

```bash
# Dump generated machine code
objdump -d /tmp/xla_dump/kernel.o > kernel.asm

# Or use LLVM's objdump for more readable output
llvm-objdump -d --x86-asm-syntax=intel /tmp/xla_dump/kernel.o
```

**Performance analysis:**

```bash
# Use perf for profiling
perf record -g python my_program.py
perf report

# Use LLVM's perf integration
XLA_FLAGS="--xla_cpu_llvm_profile=true" python my_program.py
```

**GDB debugging:**

```bash
# Attach GDB to the process
gdb -p $(pgrep python)

# Set breakpoints on generated functions
# (function names are derived from HLO computation names)
break entry_computation
```
