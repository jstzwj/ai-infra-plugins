# XLA Reference - Chapter 18: TPU Backend

This reference provides comprehensive documentation on XLA's TPU (Tensor Processing Unit) backend, covering the TPU architecture, TPU-specific optimizations, memory model, MegaScale debugging, and memory space identifiers.

---

## 18.1 TPU Architecture

The TPU backend targets Google's Tensor Processing Units, which are application-specific integrated circuits (ASICs) designed for high-performance machine learning workloads. XLA serves as the primary compiler for TPU, converting HLO programs into instructions that execute on the TPU hardware.

### 18.1.1 TPUs Supported

XLA's TPU backend supports multiple generations of TPU hardware:

**TPU v2 (Cloud TPU v2)**:
- Introduced in 2017.
- 180 TFLOPS (BF16).
- 64 GB HBM (High Bandwidth Memory) per pod.
- 2D systolic array (128x128).
- Used for training and inference.

**TPU v3 (Cloud TPU v3)**:
- Introduced in 2018.
- 420 TFLOPS (BF16).
- 128 GB HBM per pod.
- 2D systolic array (256x256).
- Liquid cooling for higher clock speeds.
- Improved interconnect bandwidth.

**TPU v4 (Cloud TPU v4)**:
- Introduced in 2020.
- 275 TFLOPS per core (BF16), 2x over v3.
- MatMul units with MXU (Matrix Multiply Unit).
- SparseCores for embedding lookup acceleration.
- Improved inter-chip interconnect (ICI) bandwidth.
- 3D torus topology for pod connectivity.

**TPU v5 (Cloud TPU v5p)**:
- Introduced in 2023.
- Significantly higher compute throughput than v4.
- Larger HBM capacity.
- Enhanced SparseCore support.
- Improved power efficiency.

**TPU v5e (Cloud TPU v5e)**:
- Cost-optimized variant.
- Designed for inference and medium-scale training.
- 16 GB HBM per chip.
- Lower power consumption than v5p.

Each TPU generation has specific hardware characteristics that the XLA compiler must account for:

| Feature | v2 | v3 | v4 | v5p | v5e |
|---------|----|----|----|----|-----|
| Systolic array | 128x128 | 128x128 | 256x256 | Larger | Smaller |
| BF16 support | Yes | Yes | Yes | Yes | Yes |
| INT8 support | No | No | Yes | Yes | Yes |
| SparseCore | No | No | Yes | Yes | No |
| HBM per chip | 16 GB | 32 GB | 32 GB | 95 GB | 16 GB |
| ICI bandwidth | 496 Gbps | 496 Gbps | 4.8 Tbps | Higher | Lower |

### 18.1.2 TPU Hardware Organization

Each TPU chip contains:

1. **Matrix Multiply Unit (MXU)**: A systolic array optimized for matrix multiplication. The MXU performs multiply-accumulate operations in BF16 precision with FP32 accumulation. It is the primary compute unit for dense linear algebra operations.

2. **Vector Unit**: Handles elementwise operations, reductions, and data movement. The vector unit operates on vectors of data and supports a wide range of operations including arithmetic, comparison, and type conversion.

3. **Scalar Unit**: Handles control flow, address computation, and scalar operations.

4. **SparseCore (v4+)**: A specialized unit for embedding lookup operations common in recommendation systems and other sparse workloads. SparseCores handle sparse-dense matrix multiplications efficiently.

5. **On-chip memory**: Fast on-chip memory (VMEM) used for intermediate results and frequently accessed data. VMEM has much higher bandwidth than HBM but is limited in size.

6. **High Bandwidth Memory (HBM)**: Off-chip memory with high bandwidth (hundreds of GB/s to TB/s). HBM stores the model parameters, activations, and other large tensors.

7. **Inter-Chip Interconnect (ICI)**: High-speed links connecting TPU chips in a pod. ICI enables all-to-all communication between chips with low latency and high bandwidth.

### 18.1.3 Memory Model

The TPU has a hierarchical memory model with distinct memory spaces:

```
+------------------------------------------+
|              HBM (S(0))                   |  Large, high bandwidth
|   - Model parameters                     |  ~hundreds of GB/s
|   - Large activations                    |  - Moderate latency
|   - Gradient accumulators                |
+------------------------------------------+
         |  High bandwidth link
         v
+------------------------------------------+
|            VMEM (S(1))                    |  Small, very high bandwidth
|   - Intermediate computation results     |  ~multiple TB/s
|   - Tile data for systolic array         |  - Very low latency
|   - Frequently reused data               |  - Limited size
+------------------------------------------+
         |
         v
+------------------------------------------+
|       MXU / Vector Unit registers        |  Registers
|   - Current computation operands         |  - Highest bandwidth
|   - Accumulators                         |  - Smallest capacity
+------------------------------------------+
```

**HBM (High Bandwidth Memory) - S(0)**:
- The primary storage for tensors.
- Addressed via the `S(0)` memory space identifier.
- Accessible by all compute units.
- Typical size: 16-95 GB depending on TPU generation.
- Bandwidth: 600 GB/s to 3.2 TB/s depending on generation.

**VMEM (Vector Memory) - S(1)**:
- Fast on-chip memory for intermediate results.
- Addressed via the `S(1)` memory space identifier.
- Used by the compiler for data that needs to be accessed repeatedly.
- Typical size: 8-32 MB depending on TPU generation.
- Bandwidth: Multiple TB/s (much higher than HBM).

**Additional device memory - S(2), S(3)**:
- Some TPU generations have additional memory spaces.
- `S(2)` and `S(3)` may refer to specialized memory regions (e.g., SparseCore memory, instruction memory).
- Usage varies by TPU generation.

### 18.1.4 SparseCore Support

Starting with TPU v4, SparseCores provide hardware acceleration for embedding lookups and other sparse operations. SparseCores are programmable units that operate independently from the main MXU:

**SparseCore capabilities:**
- **Embedding lookup**: Efficiently gather embedding vectors from large tables using indices.
- **Sparse-dense matrix multiplication**: Multiply a sparse matrix (represented in compressed format) with a dense matrix.
- **Scatter-add**: Accumulate values into specific positions of a large tensor using indices.

**XLA support for SparseCores:**
- HLO operations like `gather`, `scatter`, and `dot` with sparse operands can be directed to SparseCores.
- The compiler generates SparseCore instructions (SC instructions) that execute on the SparseCore unit.
- SparseCore operations can overlap with MXU operations for improved utilization.

**SparseCore programming model:**

The SparseCore is programmed through a set of specialized instructions:
- `SC.GATHER`: Load elements from a table using indices.
- `SC.SCATTER`: Write elements to specific positions.
- `SC.UPDATE`: Update elements in-place (e.g., for gradient updates).
- `SC.MAX`: Find maximum values along a sparse dimension.

---

## 18.2 TPU-Specific Optimizations

The TPU backend applies several optimizations that are specific to TPU hardware characteristics.

### 18.2.1 Spatial Partitioning

Spatial partitioning is a TPU-specific optimization that breaks large computations into tiles that fit within the TPU's systolic array dimensions. When a matrix multiplication or convolution is too large to fit in the MXU in a single pass, spatial partitioning divides it into smaller tiles.

**Motivation**: The TPU's systolic array has fixed dimensions (e.g., 128x128 for v2, 256x256 for v4). If a matrix multiplication has dimensions that exceed these limits, it must be decomposed:

```
// Matrix multiplication: [M, K] x [K, N] -> [M, N]
// If M > MXU_SIZE or N > MXU_SIZE or K > MXU_SIZE:
//   Tile the computation:
//   result[i:i+TILE, j:j+TILE] = sum_k input1[i:i+TILE, k:k+TILE] * input2[k:k+TILE, j:j+TILE]
```

**Partitioning algorithm:**

1. **Identify tileable dimensions**: For matrix multiplications, the M, N, and K dimensions can be tiled independently. For convolutions, the batch, output spatial, and channel dimensions can be tiled.

2. **Determine tile sizes**: Choose tile sizes that fit within the MXU dimensions while maximizing utilization:
   ```
   tile_m = min(M, MXU_ROWS)
   tile_n = min(N, MXU_COLS)
   tile_k = min(K, MXU_ROWS)  // or MXU_COLS depending on layout
   ```

3. **Generate tiled loops**: Generate loops that iterate over tiles:
   ```
   for i in range(0, M, tile_m):
     for j in range(0, N, tile_n):
       for k in range(0, K, tile_k):
         result[i:i+tile_m, j:j+tile_n] += input1[i:i+tile_m, k:k+tile_k] * input2[k:k+tile_k, j:j+tile_n]
   ```

4. **Insert data movement**: Generate instructions to load tile data from HBM to VMEM, execute the computation, and store results back to HBM:
   ```
   // Load input1 tile from HBM to VMEM
   copy HBM[S(0):input1_offset] -> VMEM[S(1):vbuf1]
   // Load input2 tile from HBM to VMEM
   copy HBM[S(0):input2_offset] -> VMEM[S(1):vbuf2]
   // Execute matrix multiply
   matmul VMEM[S(1):vbuf1] * VMEM[S(1):vbuf2] -> VMEM[S(1):vbuf3]
   // Accumulate result
   accumulate VMEM[S(1):vbuf3] -> VMEM[S(1):result]
   // Store result tile back to HBM
   copy VMEM[S(1):result] -> HBM[S(0):output_offset]
   ```

5. **Overlap data movement with computation**: The compiler attempts to overlap HBM<->VMEM copies with computation to hide memory latency:
   ```
   Time:  |--load tile (1,1)--|--compute tile (1,1)--|--store tile (1,1)--|
          |                   |--load tile (1,2)--|  |  |--compute tile (1,2)--|
          // Overlap loading next tile with computing current tile
   ```

### 18.2.2 BFloat16 Handling

TPUs natively operate in BF16 (Brain Float 16) precision. The BFloat16 handling pass ensures that all operations use the appropriate precision:

**BF16 format**: The BF16 format uses 1 sign bit, 8 exponent bits, and 7 mantissa bits. It has the same dynamic range as FP32 (8 exponent bits) but reduced precision (7 vs. 24 mantissa bits). This makes it well-suited for deep learning workloads where the dynamic range matters more than exact precision.

```
FP32:  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM  (1+8+23 = 32 bits)
BF16:  S EEEEEEEE MMMMMMM1                  (1+8+7 = 16 bits)
```

**Conversion strategy:**

1. **Input conversion**: All FP32 inputs are converted to BF16 at the TPU boundary:
   ```
   // HLO:
   %input = parameter(0), f32[128, 512]
   %bf16_input = convert(%input), bf16[128, 512]
   ```

2. **Computation in BF16**: All arithmetic operations execute in BF16:
   ```
   // HLO:
   %result = dot(%bf16_input, %bf16_weights), bf16[128, 256]
   ```
   The MXU performs BF16 multiplication with FP32 accumulation, providing good numerical stability for the accumulation.

3. **FP32 accumulation**: Matrix multiplications accumulate in FP32 internally, even though the inputs and outputs are BF16:
   ```
   // Internal MXU behavior:
   // for each element:
   //   accumulator += (float32)input_a * (float32)input_b
   // output = (bfloat16)accumulator
   ```

4. **Output conversion**: Results may be converted back to FP32 for operations that require higher precision:
   ```
   // HLO:
   %fp32_result = convert(%bf16_result), f32[128, 256]
   ```

**Operations that remain in FP32:**

Some operations are kept in FP32 for numerical stability:
- **Loss functions**: Cross-entropy loss, softmax denominators.
- **Normalization**: Layer normalization variance computation, batch normalization statistics.
- **Gradient computations**: Certain gradient computations that are sensitive to precision.
- **Reductions**: Sum reductions of large arrays (to avoid accumulation errors).

**BFloat16Normalization pass:**

The `BFloat16Normalization` pass enforces BF16 precision constraints:
1. Identifies operations that should operate in BF16.
2. Inserts explicit `convert` operations at FP32/BF16 boundaries.
3. Verifies that all operations have consistent precision.
4. Supports mixed-precision models where some operations intentionally use FP32.

### 18.2.3 Layout Optimization for TPU

Layout optimization on TPU determines the physical layout (tiling, padding, and dimension ordering) for all tensors. TPU layout requirements differ significantly from CPU and GPU:

**TPU layout concepts:**

1. **Tiling**: TPU hardware operates on fixed-size tiles. The layout must specify how tensors are divided into tiles:
   - **Major-to-minor ordering**: Dimensions are ordered from most major to most minor.
   - **Tiling specification**: Each tile has a fixed size (e.g., 128x128 for the MXU).
   - **Padding**: Tensors whose dimensions are not multiples of the tile size must be padded.

2. **Dimension ordering**: The order of dimensions in the physical layout determines how data flows through the systolic array:
   ```
   Logical: [batch, height, width, channels]
   TPU layout: {batch/tiling, height/tiling, width/tiling, channels/tiling}
   where tiling divides each dimension into tile-sized blocks
   ```

3. **Bitfield representation**: TPU layouts are represented as bitfields that encode the tiling structure:
   ```
   Layout bitfield:
   - Major dimension: bits 0-7
   - Minor dimension: bits 8-15
   - Tile sizes: bits 16-31
   ```

**Layout assignment for TPU:**

The TPU layout assignment pass (`TpuLayoutAssignment`) selects layouts that:
- Minimize the number of layout transformations (copies) between operations.
- Maximize MXU utilization (tile dimensions should match the MXU size).
- Minimize padding waste.
- Support the specific operation's access pattern (e.g., convolution requires specific dimension ordering).

**Common TPU layouts:**

```
// Matrix multiplication: [M, K] x [K, N] -> [M, N]
// Optimal layout for MXU:
// Input1: [M/tile, K/tile, tile, tile] (row-major within tile)
// Input2: [K/tile, N/tile, tile, tile] (column-major within tile)
// Output: [M/tile, N/tile, tile, tile] (row-major within tile)

// Convolution: [batch, height, width, channels]
// Optimal layout for TPU convolution unit:
// Input: [batch/8, height*tile, width*tile, channels/128, 8, 128]
// Filter: [filter_h*filter_w, input_channels/128, output_channels/128, 128, 128]
```

---

## 18.3 TPU Compilation Pipeline

The TPU compilation pipeline includes hardware-independent optimizations plus TPU-specific stages:

```
HLO Module (from frontend)
    |
    v
[1. Hardware-Independent Optimization]
    - Algebraic simplification
    - Constant folding
    - DCE
    - Fusion
    - Sharding propagation (SPMD)
    |
    v
[2. TPU-Specific Rewriting]
    - BFloat16Normalization
    - TpuLayoutAssignment
    - SpatialPartitioning
    - TPU-specific fusion patterns
    |
    v
[3. Buffer Assignment]
    - Memory space assignment (HBM, VMEM)
    - Buffer allocation and reuse
    - Copy insertion for memory space transitions
    |
    v
[4. Code Generation]
    - HLO -> TPU instruction sequence
    - MXU instructions for matrix ops
    - Vector unit instructions for elementwise ops
    - SparseCore instructions for sparse ops
    - Data movement instructions
    |
    v
[5. TPU Executable]
    - Instruction sequence
    - Buffer allocation plan
    - Communication schedule (for multi-chip)
```

### 18.3.1 SPMD on TPU

TPU's SPMD implementation is particularly important because TPU pods consist of thousands of chips that must work together:

**TPU pod topology:**
- **v2/v3 pods**: Up to 2048 chips connected in a 2D torus.
- **v4 pods**: Up to 4096 chips connected in a 3D torus.
- **v5 pods**: Even larger configurations.

**SPMD partitioning on TPU:**

1. **Mesh configuration**: Define a logical mesh over the physical chips:
   ```python
   # JAX example
   import jax
   from jax.sharding import Mesh, PartitionSpec

   devices = jax.devices()  # All TPU devices in the pod
   mesh = Mesh(devices.reshape((num_hosts, chips_per_host)), ('host', 'chip'))
   ```

2. **Sharding specification**: Annotate tensors with sharding across the mesh:
   ```python
   # Shard a tensor across chips
   spec = PartitionSpec(None, 'chip')  # Shard dimension 1 across chips
   ```

3. **Communication insertion**: The compiler inserts communication operations:
   - **All-reduce**: For reductions across sharded dimensions.
   - **All-gather**: For gathering sharded dimensions.
   - **Collective-permute**: For shifting data between specific chips.
   - **All-to-all**: For resharding operations.

4. **Communication overlap**: The compiler overlaps communication with computation where possible, using TPU's dedicated communication hardware that operates independently from the compute units.

### 18.3.2 TPU Fusion

TPU fusion combines multiple operations into a single instruction sequence that keeps intermediate results in VMEM. TPU fusion differs from GPU fusion in several ways:

1. **VMEM-resident intermediates**: Fused operations keep intermediate results in VMEM (S(1)) instead of writing them back to HBM (S(0)).

2. **Larger fusion windows**: The TPU can fuse more operations than the GPU because VMEM is managed by the compiler and the fusion does not need to fit in GPU registers.

3. **Instruction-level fusion**: On TPU, fusion is expressed as a sequence of instructions that execute without returning to HBM:
   ```
   // Unfused:
   copy HBM->VMEM input
   matmul VMEM->VMEM result
   copy VMEM->HBM result
   copy HBM->VMEM result
   add VMEM->VMEM biased
   copy VMEM->HBM biased

   // Fused:
   copy HBM->VMEM input
   copy HBM->VMEM bias
   matmul VMEM->VMEM result
   add VMEM->VMEM biased
   copy VMEM->HBM biased
   ```

---

## 18.4 MegaScale Debugging

MegaScale is the TPU debugging infrastructure that provides tools for diagnosing performance issues, correctness bugs, and system-level problems in TPU programs.

### 18.4.1 Debugging Workflow

The MegaScale debugging workflow follows a structured approach:

**Step 1: Reproduce the issue.**

```bash
# Capture the HLO module for offline debugging
XLA_FLAGS="--xla_dump_to=/tmp/tpu_dump --xla_dump_hlo_as_text" \
  python my_tpu_program.py

# The dump directory contains HLO at each compilation stage:
# - before_optimizations.hlo
# - after_algebraic_simplifier.hlo
# - after_layout_assignment.hlo
# - after_buffer_assignment.hlo
# - optimized.hlo (final)
```

**Step 2: Analyze the HLO module.**

```bash
# Run the TPU compilation pipeline offline
hlo-opt --backend=tpu --optimize /tmp/tpu_dump/optimized.hlo

# Run specific passes to isolate the issue
hlo-opt --pass=tpu-layout-assignment /tmp/tpu_dump/before_layout_assignment.hlo
hlo_opt --pass=spatial-partitioning /tmp/tpu_dump/before_partitioning.hlo
```

**Step 3: Compare with reference.**

```bash
# Compare TPU output with CPU reference
XLA_FLAGS="--xla_dump_to=/tmp/cpu_dump" \
  JAX_PLATFORMS=cpu python my_tpu_program.py

# Compare HLO modules
diff /tmp/cpu_dump/optimized.hlo /tmp/tpu_dump/optimized.hlo
```

**Step 4: Profile execution.**

```bash
# TPU profiling
import jax
profiler = jax.profiler
with profiler.trace("/tmp/tpu_trace"):
    result = my_tpu_function(x)

# View in TensorBoard
tensorboard --logdir=/tmp/tpu_trace
```

**Step 5: Debug numerical issues.**

```bash
# Enable float64 on TPU for numerical comparison
jax.config.update('jax_enable_x64', True)

# Compare BF16 vs FP32 results
# Run with FP32 accumulation
XLA_FLAGS="--xla_tpu_enable_aggressive_fp32_matmul" python my_program.py
```

### 18.4.2 MegaScale Overview

MegaScale provides several debugging capabilities:

1. **HLO visualization**: Visualize the HLO computation graph to understand the structure and identify optimization opportunities.

2. **Instruction-level profiling**: Profile each TPU instruction to identify bottlenecks:
   - MXU utilization (how busy the systolic array is).
   - VMEM usage (is the program VMEM-bound?).
   - HBM bandwidth utilization.
   - ICI communication volume and latency.

3. **Memory analysis**: Track memory usage throughout execution:
   - HBM allocation and deallocation events.
   - VMEM allocation and utilization over time.
   - Peak memory usage and fragmentation.

4. **Correctness checking**: Verify that the compiled program produces correct results:
   - **Numeric comparison**: Compare TPU output with CPU reference output.
   - **Bit-exact checking**: For deterministic operations, verify bit-exact matching.
   - **Tolerance checking**: For non-deterministic operations (e.g., reductions in BF16), verify results are within acceptable tolerance.

5. **Communication analysis**: For multi-chip programs:
   - Collective communication volume and frequency.
   - Communication-computation overlap efficiency.
   - ICI utilization and congestion.

6. **Performance counter access**: Access hardware performance counters:
   - MXU FLOPs delivered vs. theoretical peak.
   - HBM bytes transferred vs. theoretical bandwidth.
   - VMEM hits and misses.
   - Instruction issue rate.

---

## 18.5 Memory Space Identifiers for TPU

The TPU has distinct memory spaces that are identified by integer identifiers. These identifiers are used in HLO operations and buffer assignment to specify where data should be stored.

### 18.5.1 S(0) - HBM (High Bandwidth Memory)

**Identifier**: `S(0)` or `memory_space = 0`

HBM is the primary storage for tensors on the TPU. It provides:
- **Capacity**: 16-95 GB per chip depending on TPU generation.
- **Bandwidth**: 600 GB/s to 3.2 TB/s.
- **Latency**: Hundreds of cycles for random access.

**Usage:**
- Model parameters (weights, biases).
- Large activation tensors.
- Gradient accumulators.
- Input and output data.
- Any tensor that is too large to fit in VMEM.

**HBM allocation:**

HBM is allocated by the buffer assignment pass. Buffers in HBM are allocated for the entire duration of the program (static allocation) and are assigned fixed offsets within the HBM address space:

```cpp
struct HbmBufferAllocation {
  int64_t offset;        // Offset in HBM
  int64_t size;          // Size in bytes
  HloInstruction* instr; // Instruction that owns this buffer
  bool is_input;         // Whether this buffer holds an input parameter
  bool is_output;        // Whether this buffer holds an output
};
```

**HBM management:**

The XLA runtime manages HBM allocation:
- Total HBM usage must not exceed the available capacity.
- Buffers with non-overlapping liveness can share the same HBM region.
- The compiler ensures that HBM usage stays within the chip's capacity.

### 18.5.2 S(1) - VMEM (Vector Memory)

**Identifier**: `S(1)` or `memory_space = 1`

VMEM is fast on-chip memory used for intermediate computation results. It provides:
- **Capacity**: 8-32 MB per chip depending on TPU generation.
- **Bandwidth**: Multiple TB/s (much higher than HBM).
- **Latency**: Tens of cycles for random access.

**Usage:**
- Intermediate results between fused operations.
- Tile data loaded from HBM for processing by the MXU.
- Frequently reused data (e.g., lookup tables).
- Accumulator buffers for tiled computations.

**VMEM allocation:**

VMEM is a scarce resource and is managed carefully by the compiler. The VMEM allocation algorithm:

1. **Estimate VMEM usage**: For each operation, compute the VMEM needed for its inputs, outputs, and intermediate results.

2. **VMEM-aware scheduling**: Schedule operations to minimize peak VMEM usage. Operations that require a lot of VMEM may be scheduled when other large VMEM allocations have been freed.

3. **VMEM-aware fusion**: Limit fusion to avoid exceeding VMEM capacity. If a fusion would require more VMEM than available, the compiler splits the fusion:

   ```
   // Original fusion (too large for VMEM):
   %fused = fusion(%a, %b) {
     %p0 = parameter(0)  // [1024, 1024] = 4 MB in BF16
     %p1 = parameter(1)  // [1024, 1024] = 4 MB in BF16
     %r1 = dot(%p0, %p1) // [1024, 1024] = 4 MB in BF16 (in VMEM)
     %r2 = exp(%r1)      // [1024, 1024] = 4 MB in BF16 (in VMEM)
     %r3 = add(%r2, %p0) // [1024, 1024] = 4 MB in BF16 (in VMEM)
     // Total VMEM: 4 + 4 + 4 + 4 = 16 MB -- may exceed VMEM capacity!
   }

   // Split fusion:
   // Fusion 1: dot + exp (8 MB VMEM)
   // Fusion 2: add (4 MB VMEM)
   ```

4. **VMEM spilling**: When VMEM is exhausted, the compiler inserts spill operations that write VMEM contents back to HBM and reload them later:

   ```
   // VMEM spill:
   copy VMEM[S(1):temp_buffer] -> HBM[S(0):spill_slot]
   // ... other operations ...
   copy HBM[S(0):spill_slot] -> VMEM[S(1):temp_buffer]
   ```

### 18.5.3 S(2), S(3) - Additional Device Memory

**Identifier**: `S(2)` or `memory_space = 2`, and `S(3)` or `memory_space = 3`

These memory spaces refer to additional on-chip or near-chip memory regions. Their exact semantics vary by TPU generation:

**S(2) - SparseCore memory / Instruction memory**:
- On TPU v4+, `S(2)` may refer to memory used by SparseCores for embedding table lookups.
- SparseCore memory stores embedding table data that is frequently accessed during sparse operations.
- Size varies by TPU generation but is typically smaller than VMEM.

**S(3) - Additional scratch space**:
- May refer to additional on-chip scratch memory.
- Used for specialized operations that require temporary storage.
- Availability and size vary by TPU generation.

**Memory space assignment:**

The `MemorySpaceAssignment` pass determines which memory space each buffer should be allocated in:

```cpp
enum class TpuMemorySpace {
  kHbm = 0,    // S(0) - High Bandwidth Memory
  kVmem = 1,   // S(1) - Vector Memory
  kSparse = 2, // S(2) - SparseCore memory
  kScratch = 3 // S(3) - Additional scratch
};

struct MemorySpaceAssignment {
  // Determine the optimal memory space for a buffer
  TpuMemorySpace AssignMemorySpace(const HloValue& value,
                                     const HloDataflowAnalysis& analysis) {
    // Small, frequently accessed values -> VMEM
    if (value.instruction()->operand_count() <= 2 &&
        value.shape().byte_size() <= vmem_threshold) {
      return TpuMemorySpace::kVmem;
    }

    // SparseCore operands -> S(2)
    if (IsSparseCoreOperation(value.instruction())) {
      return TpuMemorySpace::kSparse;
    }

    // Default -> HBM
    return TpuMemorySpace::kHbm;
  }
};
```

**Memory space transitions:**

When data needs to move between memory spaces, the compiler inserts copy operations:

```
// Copy from HBM to VMEM (load for processing)
copy HBM[S(0):input] -> VMEM[S(1):vbuf]

// Process in VMEM
matmul VMEM[S(1):vbuf] * VMEM[S(1):weights] -> VMEM[S(1):result]

// Copy from VMEM to HBM (store result)
copy VMEM[S(1):result] -> HBM[S(0):output]
```

These copy operations consume memory bandwidth and add latency, so the compiler minimizes the number of memory space transitions by:
- Keeping data in the same memory space across consecutive operations.
- Fusing operations that can share VMEM-resident data.
- Prefetching data into VMEM before it is needed (overlapping with computation).

---

## 18.6 TPU Compilation Flags

| Flag | Description |
|------|-------------|
| `--xla_tpu_enable_aggressive_fp32_matmul` | Use FP32 accumulation for all matmuls |
| `--xla_tpu_enable_bf16` | Enable BF16 mode (default: true) |
| `--xla_tpu_use_spmd` | Enable SPMD partitioning |
| `--xla_tpu_num_partitions` | Number of partitions for SPMD |
| `--xla_tpu_enable_tracing` | Enable TPU instruction tracing |
| `--xla_tpu_vmem_limit_bytes` | Set VMEM allocation limit |
| `--xla_tpu_enable_sparse_cores` | Enable SparseCore usage |
| `--xla_tpu_profile` | Enable TPU profiling |
| `--xla_tpu_dump_layout_assignment` | Dump layout assignment results |
| `--xla_tpu_max_spatial_partition_tiles` | Maximum number of spatial partition tiles |

---

## 18.7 TPU vs GPU Comparison

| Aspect | TPU | GPU |
|---------|-----|-----|
| Primary compute | Systolic array (MXU) | CUDA cores + Tensor cores |
| Precision | BF16 native | FP16/BF16/TF32/FP8 |
| Memory model | HBM + VMEM (explicit) | HBM + shared memory + L2 |
| Code generation | Custom instruction sequences | PTX via LLVM |
| Fusion | VMEM-resident, large windows | Register-resident, smaller windows |
| Libraries | Custom TPU ops | cuBLAS, cuDNN, NCCL |
| Interconnect | ICI (dedicated) | NVLink/PCIe (general purpose) |
| Sparse ops | SparseCore hardware | Software-based |
| Multi-chip | Pod (thousands of chips) | Multi-GPU (tens of GPUs) |
