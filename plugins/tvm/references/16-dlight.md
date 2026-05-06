# DLight — Pre-defined High-Performance Schedule Rules

This reference covers DLight, TVM's framework for applying pre-defined, high-performance scheduling rules to common computational patterns. Unlike MetaSchedule's automated search, DLight uses deterministic pattern matching to identify computation structures and apply hand-crafted schedules that are known to perform well. DLight provides fast, predictable compilation with near-optimal performance for supported patterns.

---

## 16.1 Overview

### 16.1.1 Motivation

MetaSchedule's evolutionary search can find excellent schedules, but it requires significant tuning time (thousands of trials per task). For production deployment scenarios where compilation speed matters, or for well-understood computational patterns where optimal strategies are already known, DLight provides:

1. **Zero-tuning compilation**: No search is required. The schedule is determined immediately from the program structure.
2. **Deterministic results**: The same input always produces the same schedule, enabling reproducible builds.
3. **Near-optimal performance**: The rules encode expert knowledge about GPU/CPU optimization, matching or exceeding auto-tuning for supported patterns.
4. **Composability**: Rules can be combined and extended for new patterns.

### 16.1.2 DLight Architecture

DLight follows a rule-based architecture where each rule:

1. Inspects the PrimFunc to detect a known pattern
2. If the pattern matches, applies a pre-defined schedule
3. If no pattern matches, falls back to a default strategy

```
PrimFunc
    |
    v
+-------------------+
| Pattern Matcher   |  Examines block structure, loop patterns, access regions
+--------+----------+
         |
   Match found?
    /         \
  Yes          No
   |            |
   v            v
+------+   +---------+
| Apply |   | Try next|  or fallback to default rule
| Rule  |   |  Rule   |
+------+   +---------+
   |
   v
Scheduled PrimFunc
```

The architecture separates concerns into three layers:

- **Pattern detection**: Analyze the block iteration structure, classify axes as spatial or reduction, examine access patterns and data dependencies.
- **Schedule generation**: Given a matched pattern, construct a sequence of TensorIR schedule primitives (split, bind, cache_read, vectorize, etc.).
- **Target adaptation**: Adjust tile sizes, thread counts, and memory strategies based on the target GPU's specifications (shared memory size, warp size, max threads per block).

### 16.1.3 Basic Usage

```python
import tvm
from tvm import dlight as dl

# Apply DLight GPU rules to an entire IRModule
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Decode(),
        dl.gpu.Fallback(),
    )(mod)
```

The `ApplyDefaultSchedule` function takes an ordered list of rules and tries each one in sequence against each PrimFunc in the IRModule. The first rule that matches and successfully applies its schedule is used. If no rule matches, the PrimFunc is left unchanged.

### 16.1.4 DLight Module Organization

DLight is organized into submodules by target:

```
tvm.dlight/
    __init__.py           # ApplyDefaultSchedule, base classes
    gpu/
        __init__.py       # GPU rule exports
        gemv.py           # GEMV rule
        matmul.py         # Matmul rule
        reduction.py      # Reduction rule
        decode.py         # Decode / DecodeGEMV rule
        general_reduction.py  # GeneralReduction rule
        fallback.py       # GPU fallback rule
    cpu/
        __init__.py       # CPU rule exports
        fallback.py       # CPU fallback rule
    analysis/
        __init__.py       # Pattern analysis utilities
        reduction.py      # Reduction detection
        broadcast.py      # Broadcast epilogue detection
        dominated_var.py  # Dominated variable analysis
```

---

## 16.2 GPU Rules

### 16.2.1 GEMV Rule

Optimizes General Matrix-Vector Multiplication (GEMV) patterns: `y = A @ x` where A is a matrix and x is a vector. GEMV is memory-bandwidth-bound on GPUs, so the optimization strategy focuses on maximizing memory throughput.

**Pattern characteristics:**

- A reduction over one dimension of a 2D input, producing a 1D output
- The non-reduced dimension maps to the output
- The reduced dimension multiplies corresponding elements and accumulates

**Detection criteria:**

- The block has exactly one spatial axis and one reduce axis
- The spatial axis corresponds to the output dimension
- The reduce axis iterates over the "contraction" dimension
- Buffer access patterns show A[spatial, reduce] and x[reduce]

```python
from tvm import dlight as dl

# Apply the GEMV rule
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
```

**Schedule strategy:**

| Step | Transformation | Purpose |
|------|---------------|---------|
| 1 | Bind spatial axis to `blockIdx.x` | Distribute rows across thread blocks |
| 2 | Split spatial axis: `threadIdx.x` for inner | Map threads within a block |
| 3 | Vectorize the innermost load | Maximize memory bandwidth |
| 4 | Unroll the reduction loop | Reduce loop overhead |
| 5 | Apply cache_read if beneficial | Reduce global memory traffic |

The GEMV rule's thread binding strategy is:

```
Original loops:
  for i in range(M):        # spatial
    for k in range(K):      # reduce
      y[i] += A[i, k] * x[k]

After scheduling:
  blockIdx.x -> i_outer     # Each block handles a group of output rows
  threadIdx.x -> i_inner    # Threads within a block handle individual rows
  vectorize on k access     # Coalesced memory reads of A and x
  reduction unrolled        # Full unrolling of inner reduction loop
```

**Example GEMV pattern in TVMScript:**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class GEMVModule:
    @T.prim_func
    def gemv(
        A: T.Buffer((4096, 4096), "float16"),
        x: T.Buffer((4096,), "float16"),
        y: T.Buffer((4096,), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for i, k in T.grid(4096, 4096):
            with T.block("gemv"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(A[vi, vk], x[vk])
                T.writes(y[vi])
                with T.init():
                    y[vi] = T.float32(0)
                y[vi] = y[vi] + T.cast(A[vi, vk], "float32") * T.cast(x[vk], "float32")

# Apply DLight GEMV rule
import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(GEMVModule)
```

**Performance characteristics:**

- Achieves >80% of peak memory bandwidth for large GEMV operations
- Performance is bounded by memory bandwidth, not compute throughput
- Optimal tile sizes are determined by the target GPU's memory hierarchy
- For M=4096, K=4096 on A100: approximately 1.3 TB/s sustained bandwidth

**Target-specific parameters:**

| Target | Threads per block | Vector width | Shared memory usage |
|--------|-------------------|--------------|---------------------|
| NVIDIA A100 | 256 | 8 (float16) | None (direct global reads) |
| NVIDIA H100 | 512 | 8 (float16) | None (direct global reads) |
| NVIDIA T4 | 256 | 4 (float16) | Optional for x buffering |

---

### 16.2.2 Matmul Rule

Optimizes General Matrix-Matrix Multiplication (GEMM) patterns: `C = A @ B`. Matmul is compute-bound for large matrices, so the strategy focuses on maximizing arithmetic throughput through tiling and shared memory usage.

**Pattern characteristics:**

- Two spatial axes (output dimensions) and one reduce axis
- The computation is a sum of products along the reduce axis
- Both input matrices are 2D

**Detection criteria:**

- The block has two spatial axes and one reduce axis
- The spatial axes correspond to the M and N dimensions of the output matrix
- The reduce axis corresponds to the K dimension
- Access patterns: A[spatial_M, reduce_K], B[reduce_K, spatial_N], C[spatial_M, spatial_N]

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
```

**Schedule strategy:**

The Matmul rule implements a hierarchical tiling strategy inspired by CUTLASS:

| Level | Tile Size (typical) | Mapping |
|-------|---------------------|---------|
| Thread block tile | 128 x 128 | `blockIdx.y`, `blockIdx.x` |
| Warp tile | 64 x 64 | warp-level scheduling |
| Thread tile | 8 x 8 | `threadIdx.y`, `threadIdx.x` |
| Micro-kernel | 4x4 or WMMA | Per-thread computation |

**Detailed transformation steps:**

1. **Cache reads to shared memory**: Load tiles of A and B into shared memory for fast access. The tile size for the K dimension is chosen to maximize shared memory utilization.
2. **Multi-level tiling**: Split M, N, K dimensions into block/warp/thread levels. Each level has a specific tile size that balances parallelism with data reuse.
3. **Thread binding**: Map outer tiles to GPU thread blocks and inner tiles to threads. Warp-level grouping ensures coalesced memory access.
4. **Pipeline shared memory loads**: Overlap computation with data loading where supported. This hides memory latency behind useful computation.
5. **Vectorize memory access**: Use vector loads for global-to-shared transfers. A typical vector width is 128 bits (8 float16 elements).
6. **Storage alignment**: Pad shared memory to avoid bank conflicts. A common padding is 1 element per row.
7. **Unroll inner loops**: Maximize instruction-level parallelism by fully unrolling the innermost computation loop.

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class MatmulModule:
    @T.prim_func
    def matmul(
        A: T.Buffer((1024, 1024), "float16"),
        B: T.Buffer((1024, 1024), "float16"),
        C: T.Buffer((1024, 1024), "float32"),
    ):
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")

import tvm
from tvm import dlight as dl

# Apply DLight Matmul rule
with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(MatmulModule)
```

**Shared memory tiling visualization:**

```
Matrix A (M x K)              Matrix B (K x N)
+-------------------+          +-------------------+
| Tile A_block      |          | Tile B_block      |
| (128 x tile_k)    |          | (tile_k x 128)    |
|   |               |          |   |               |
|   v               |          |   v               |
|  shared memory    |          |  shared memory    |
+-------------------+          +-------------------+

Each thread block loads:
  - A_block: 128 rows x tile_k columns from A
  - B_block: tile_k rows x 128 columns from B

Each thread computes:
  - Thread tile: 8 x 8 = 64 multiply-accumulate operations
  - Accumulated across all tile_k steps along K dimension
```

**Performance characteristics:**

- Achieves >70% of peak FLOPS for large square matrices (M=N=K >= 512)
- Automatically selects tile sizes based on the target GPU's specifications
- Supports mixed-precision computation (float16 inputs, float32 accumulation)
- For M=N=K=1024 on A100: approximately 180 TFLOPS (FP16 with FP32 accumulation)

**Tile size selection by target:**

| Target | Block tile | Warp tile | Thread tile | K tile |
|--------|-----------|-----------|-------------|--------|
| NVIDIA A100 | 128x128 | 32x64 | 8x8 | 32 |
| NVIDIA H100 | 128x128 | 64x64 | 8x16 | 32 |
| NVIDIA T4 | 64x64 | 32x32 | 4x8 | 16 |

---

### 16.2.3 Reduction Rule

Optimizes general reduction operations: computing a scalar or lower-dimensional tensor by combining elements along one or more axes.

**Pattern characteristics:**

- One or more spatial axes and one or more reduce axes
- The output has fewer dimensions than the input
- Common operations: sum, max, min, mean, norm

**Detection criteria:**

- The block has at least one reduce axis
- The reduce axis is not part of a matmul pattern (otherwise Matmul rule takes precedence)
- The output is a scalar or has reduced dimensionality
- The reduction operation is associative (sum, max, min, etc.)

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(mod)
```

**Schedule strategy:**

| Step | Transformation | Purpose |
|------|---------------|---------|
| 1 | Bind spatial axes to `blockIdx` | Distribute independent reductions across thread blocks |
| 2 | Bind reduce axis to `threadIdx` | Enable parallel reduction within a block |
| 3 | Use warp shuffle for reduction | Efficient cross-thread communication |
| 4 | Handle remainder elements | Support non-power-of-2 dimensions |

**Thread mapping for reductions:**

```
Original:
  for i in range(N):        # spatial (output)
    for j in range(M):      # reduce
      out[i] = reduce_op(out[i], in[i, j])

After scheduling:
  blockIdx.x -> i            # Each block computes one output element
  threadIdx.x -> j_partial   # Threads cooperatively reduce over j
  warp-level reduction       # Final reduction within each warp
  cross-warp reduction       # Combine warp results via shared memory
```

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class ReductionModule:
    @T.prim_func
    def row_sum(
        A: T.Buffer((1024, 4096), "float32"),
        B: T.Buffer((1024,), "float32"),
    ):
        for i, j in T.grid(1024, 4096):
            with T.block("sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vj]

import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(ReductionModule)
```

**Multi-axis reduction example:**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class MultiAxisReduction:
    @T.prim_func
    def sum_over_hw(
        A: T.Buffer((16, 64, 64, 3), "float32"),   # [N, H, W, C]
        B: T.Buffer((16, 3), "float32"),             # [N, C]
    ):
        for n, h, w, c in T.grid(16, 64, 64, 3):
            with T.block("sum"):
                vn, vh, vw, vc = T.axis.remap("SRRS", [n, h, w, c])
                T.reads(A[vn, vh, vw, vc])
                T.writes(B[vn, vc])
                with T.init():
                    B[vn, vc] = T.float32(0)
                B[vn, vc] = B[vn, vc] + A[vn, vh, vw, vc]
```

For multi-axis reductions, the Reduction rule flattens the reduce axes into a single dimension and applies the standard parallel reduction strategy.

---

### 16.2.4 Decode Rule

Optimizes decode-phase operations commonly found in LLM inference. During the decode phase, each step processes a single token and updates the KV cache. The computation is essentially a GEMV-like operation: `output = W @ x` where x is a vector (the current token embedding).

**Pattern characteristics:**

- GEMV-like structure (matrix-vector multiply)
- The "matrix" is a weight tensor from the model
- The "vector" is a single token's representation
- Often includes dequantization for quantized weights
- Batch size is typically 1

**Detection criteria:**

- The block has one spatial axis and one reduce axis (like GEMV)
- The spatial extent is large (hidden dimension of the model)
- The reduce extent may also be large (for MLP layers)
- The computation pattern matches a decode-phase linear layer

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Decode())(mod)
```

**Schedule strategy:**

The Decode rule applies similar optimizations to GEMV but with additional considerations for:

1. **Batch size = 1**: No batch dimension to parallelize over. All parallelism must come from the output dimension.
2. **Weight pre-fetching**: Overlap weight loading with computation. Since the weight matrix is large and accessed sequentially, pre-fetching can hide memory latency.
3. **Dequantization fusion**: Fuse quantize/dequantize operations into the GEMV kernel. This avoids extra memory traffic for intermediate dequantized values.

**Decode vs GEMV distinction:**

| Aspect | GEMV Rule | Decode Rule |
|--------|-----------|-------------|
| Batch size | Any | Typically 1 |
| Weight layout | Standard row-major | May include quantization metadata |
| Dequantization | Not fused | Fused when detected |
| Thread strategy | Standard 1D | Optimized for single-vector case |
| Target use case | General GEMV | LLM decode phase |

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class DecodeModule:
    @T.prim_func
    def decode_linear(
        W: T.Buffer((4096, 4096), "float16"),  # Weight matrix
        x: T.Buffer((4096,), "float16"),         # Input vector (single token)
        y: T.Buffer((4096,), "float32"),          # Output vector
    ):
        for i, k in T.grid(4096, 4096):
            with T.block("decode_gemm"):
                vi, vk = T.axis.remap("SR", [i, k])
                T.reads(W[vi, vk], x[vk])
                T.writes(y[vi])
                with T.init():
                    y[vi] = T.float32(0)
                y[vi] = y[vi] + T.cast(W[vi, vk], "float32") * T.cast(x[vk], "float32")

import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.Decode())(DecodeModule)
```

**Quantized decode example (INT4 weights):**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class QuantizedDecodeModule:
    @T.prim_func
    def decode_int4(
        W_q: T.Buffer((4096, 1024), "uint8"),   # INT4 packed weights (2 per byte)
        scale: T.Buffer((4096, 1), "float16"),   # Per-row scale factors
        zero_point: T.Buffer((4096, 1), "float16"),  # Per-row zero points
        x: T.Buffer((4096,), "float16"),          # Input vector
        y: T.Buffer((4096,), "float32"),           # Output vector
    ):
        for i, k_packed in T.grid(4096, 1024):
            with T.block("decode_int4"):
                vi, vk_packed = T.axis.remap("SR", [i, k_packed])
                T.reads(W_q[vi, vk_packed], scale[vi, 0], zero_point[vi, 0], x[2*vk_packed], x[2*vk_packed+1])
                T.writes(y[vi])
                with T.init():
                    y[vi] = T.float32(0)
                # Dequantize the packed INT4 values and compute
                w_lo = T.cast(T.bitwise_and(T.cast(W_q[vi, vk_packed], "int32"), 0xF), "float32")
                w_hi = T.cast(T.shift_right(T.cast(W_q[vi, vk_packed], "int32"), 4), "float32")
                w_deq_lo = (w_lo - T.cast(zero_point[vi, 0], "float32")) * T.cast(scale[vi, 0], "float32")
                w_deq_hi = (w_hi - T.cast(zero_point[vi, 0], "float32")) * T.cast(scale[vi, 0], "float32")
                y[vi] = y[vi] + w_deq_lo * T.cast(x[2*vk_packed], "float32")
                         + w_deq_hi * T.cast(x[2*vk_packed+1], "float32")
```

---

### 16.2.5 DecodeGEMV Rule

A variant of the Decode rule specifically optimized for the GEMV structure in decode-phase operations. This rule applies when the decode operation has a pure GEMV pattern without additional fused operations.

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.DecodeGEMV())(mod)
```

**Distinction from Decode rule:**

- DecodeGEMV specifically targets the case where the weight matrix is accessed in a regular pattern
- May use different tiling strategies for weight loading
- Can apply more aggressive vectorization for the weight access
- Slightly different thread binding: may use 2D thread blocks for better utilization

**When DecodeGEMV is preferred over Decode:**

1. The weight matrix is stored in standard float16/bfloat16 format (not quantized)
2. No dequantization or activation fusion is needed
3. The input vector is a pure contiguous buffer
4. Performance profiling shows the standard GEMV pattern is memory-bandwidth-bound

---

### 16.2.6 GeneralReduction Rule

Optimizes general reduction patterns that do not match the specific GEMV or Matmul patterns. This includes reductions over multiple dimensions, reductions with non-trivial access patterns, and reductions with epilogue operations.

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(mod)
```

**Schedule strategy:**

1. Analyze the reduction dimensions to determine the optimal thread mapping
2. Bind spatial axes to `blockIdx` for independent output elements
3. Flatten multiple reduce axes into a single dimension
4. Use shared memory for intermediate reduction results if needed
5. Apply cross-thread reduction for the innermost reduce axis
6. Handle non-power-of-2 sizes with predicated threads

**Example: Multi-dimensional reduction**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class GeneralRedModule:
    @T.prim_func
    def norm(
        A: T.Buffer((256, 256, 256), "float32"),
        B: T.Buffer((256,), "float32"),
    ):
        for i, j, k in T.grid(256, 256, 256):
            with T.block("sum"):
                vi, vj, vk = T.axis.remap("SRR", [i, j, k])
                T.reads(A[vi, vj, vk])
                T.writes(B[vi])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vj, vk]

import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(GeneralRedModule)
```

**Example: Reduction with epilogue (layer norm)**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class LayerNormModule:
    @T.prim_func
    def layernorm(
        X: T.Buffer((128, 1024), "float32"),
        Gamma: T.Buffer((1024,), "float32"),
        Beta: T.Buffer((1024,), "float32"),
        Y: T.Buffer((128, 1024), "float32"),
    ):
        # Compute mean
        mean = T.alloc_buffer((128,), "float32")
        for i, j in T.grid(128, 1024):
            with T.block("mean"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(X[vi, vj])
                T.writes(mean[vi])
                with T.init():
                    mean[vi] = T.float32(0)
                mean[vi] = mean[vi] + X[vi, vj] / T.float32(1024)

        # Compute variance
        var = T.alloc_buffer((128,), "float32")
        for i, j in T.grid(128, 1024):
            with T.block("var"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(X[vi, vj], mean[vi])
                T.writes(var[vi])
                with T.init():
                    var[vi] = T.float32(0)
                var[vi] = var[vi] + (X[vi, vj] - mean[vi]) ** 2 / T.float32(1024)

        # Normalize and scale
        for i, j in T.grid(128, 1024):
            with T.block("output"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(X[vi, vj], mean[vi], var[vi], Gamma[vj], Beta[vj])
                T.writes(Y[vi, vj])
                Y[vi, vj] = (X[vi, vj] - mean[vi]) / T.sqrt(var[vi] + T.float32(1e-5)) * Gamma[vj] + Beta[vj]
```

---

### 16.2.7 Fallback Rule (GPU)

The GPU fallback rule applies a generic scheduling strategy when no specific rule matches. It ensures that every PrimFunc receives at least a basic GPU schedule.

```python
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Fallback(),  # Catch-all for unmatched patterns
    )(mod)
```

**Fallback strategy:**

1. Find all blocks in the function
2. For element-wise blocks: bind to `blockIdx.x` and `threadIdx.x` with vectorization
3. For reduction blocks: apply basic cross-thread reduction
4. Inline all trivial producers/consumers
5. Bind outermost loop to `blockIdx` and inner loop to `threadIdx`
6. Vectorize the innermost contiguous access where possible

**What the fallback handles:**

- Element-wise operations (relu, sigmoid, element-wise add/mul)
- Copy/transpose operations
- Reshape and broadcast operations
- Padding and slicing operations
- Any other operation not matched by a specific rule

---

## 16.3 CPU Rules

### 16.3.1 Fallback Rule (CPU)

The CPU fallback rule applies a generic scheduling strategy for CPU targets. It handles all patterns that do not have CPU-specific optimizations.

```python
from tvm import dlight as dl

with tvm.target.Target("cpu"):
    mod = dl.ApplyDefaultSchedule(
        dl.cpu.Fallback(),
    )(mod)
```

**Fallback strategy:**

1. Inline all trivial (element-wise, injective) blocks
2. Parallelize the outermost loop using OpenMP or thread pool
3. Vectorize the innermost loop with a target-appropriate width (e.g., AVX-512 = 16 floats)
4. Unroll small inner loops to reduce overhead

**CPU vectorization widths by target:**

| Target | SIMD width (float32) | SIMD width (float16) |
|--------|----------------------|----------------------|
| x86-64 (SSE) | 4 | N/A |
| x86-64 (AVX2) | 8 | N/A |
| x86-64 (AVX-512) | 16 | N/A |
| ARM (NEON) | 4 | 8 |
| ARM (SVE) | Variable | Variable |
| RISC-V (Vector) | Variable | Variable |

```python
from tvm.script import ir as I, tir as T
import tvm
from tvm import dlight as dl

@I.ir_module
class CPUModule:
    @T.prim_func
    def elementwise(
        A: T.Buffer((4096, 4096), "float32"),
        B: T.Buffer((4096, 4096), "float32"),
    ):
        for i, j in T.grid(4096, 4096):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * T.float32(2.0) + T.float32(1.0)

# Apply CPU fallback
with tvm.target.Target("cpu"):
    scheduled_mod = dl.ApplyDefaultSchedule(dl.cpu.Fallback())(CPUModule)
```

---

## 16.4 Analysis Helpers

DLight uses several internal analysis helpers to detect patterns and make scheduling decisions. These are available for users writing custom rules.

### 16.4.1 `is_broadcast_epilogue`

Checks whether a reduction block has a broadcast epilogue -- a situation where the final operation after reduction broadcasts a value across multiple output elements.

```python
from tvm.dlight import analysis as dl_analysis

# Check if a block has a broadcast epilogue
has_broadcast = dl_analysis.is_broadcast_epilogue(block, spatial_axes, reduce_axis)
```

**Usage in rule matching:**

Broadcast epilogues require special handling because the output elements may not be independent. If a reduction has a broadcast epilogue (e.g., softmax normalization, layer norm), the scheduling strategy must ensure that all threads contributing to the same output element synchronize properly.

**Common patterns with broadcast epilogues:**

| Pattern | Broadcast Dimension | Example |
|---------|---------------------|---------|
| Softmax | Over class dimension | `softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))` |
| LayerNorm | Over hidden dimension | `layernorm(x)[i] = (x[i] - mean) / sqrt(var)` |
| BatchNorm | Over spatial dimensions | `batchnorm(x)[i,j] = (x[i,j] - mean[j]) / sqrt(var[j])` |
| RMSNorm | Over hidden dimension | `rmsnorm(x)[i] = x[i] / sqrt(mean(x[j]^2))` |

### 16.4.2 `detect_dominated_var`

Detects whether a variable's value is uniquely determined by the surrounding loop iteration variables. Used to validate that loop transformations are safe.

```python
from tvm.dlight import analysis as dl_analysis

is_dominated = dl_analysis.detect_dominated_var(expr, var, loop_vars)
```

**Usage in DLight rules:**

Before applying a tiling transformation, the rule checks that buffer indices are dominated by the loop variables. If an index expression uses variables outside the loop nest, the transformation may produce incorrect results.

**Example:**

```python
# In this block, the index vi is dominated by loop var i
for i in range(128):
    with T.block("B"):
        vi = T.axis.remap("S", [i])
        A[vi]  # vi is dominated by i -> safe to tile

# In this block, the index uses an external variable
# external_var is NOT dominated by loop var i -> tiling may be unsafe
for i in range(128):
    with T.block("B"):
        vi = T.axis.remap("S", [i])
        A[vi + external_var]  # Contains non-dominated variable
```

### 16.4.3 `detect_reduction`

Analyzes a block to determine if it contains a reduction pattern and extracts the reduction axes.

```python
from tvm.dlight import analysis as dl_analysis

reduction_info = dl_analysis.detect_reduction(block)
if reduction_info is not None:
    spatial_axes, reduce_axes = reduction_info
    print(f"Spatial axes: {spatial_axes}")
    print(f"Reduce axes: {reduce_axes}")
```

**Return value:** A tuple `(spatial_axes, reduce_axes)` if the block contains a reduction, or `None` if it does not.

**Details:**

This analysis is the primary pattern-matching mechanism for DLight rules. It examines the block's iteration variables and classifies them as:

- **Spatial**: Iteration variables that map directly to output elements (one-to-one)
- **Reduce**: Iteration variables that are collapsed through an associative reduction operation

The classification determines which DLight rule applies:

| Spatial axes | Reduce axes | Matching rule |
|-------------|-------------|---------------|
| 1 | 1 | GEMV or Decode |
| 2 | 1 | Matmul |
| N (N >= 1) | M (M >= 1) | GeneralReduction |
| N | 0 | Element-wise (Fallback) |

### 16.4.4 `detect_linear_equation`

Detects whether buffer index expressions form linear equations over the iteration variables. This is used to determine whether tiling and reindexing transformations preserve correctness.

```python
from tvm.dlight import analysis as dl_analysis

# Check if index expressions are linear in the loop variables
is_linear = dl_analysis.detect_linear_equation(indices, loop_vars)
```

**Why linear equations matter:**

Tiling transformations (split, reorder, fuse) work correctly only when buffer access patterns are linear in the iteration variables. If an index is a non-linear function of the loop variables (e.g., `A[i * i]`), tiling may produce incorrect results because the relationship between the tile's boundaries and the accessed memory region is not well-defined.

---

## 16.5 Rule Application Flow

### 16.5.1 Pattern Matching Pipeline

When `ApplyDefaultSchedule` is called with multiple rules, it processes each PrimFunc in the IRModule through the following pipeline:

```
For each PrimFunc in IRModule:
    For each rule in the provided list (in order):
        1. Analyze the PrimFunc's block structure
        2. Extract iteration variable classifications (spatial/reduce)
        3. Check access patterns and data dependencies
        4. If the pattern matches:
            a. Create a schedule: sch = tvm.tir.Schedule(mod)
            b. Apply the rule's transformation
            c. Return the scheduled module
    If no rule matches:
        Return the original module unchanged
```

**Important:** Rules are tried in the order they are provided. The first matching rule is applied. If a more specific rule is listed after a more general one, the general rule will match first. Always list rules from most specific to most general:

```python
# CORRECT order: specific rules first, fallback last
mod = dl.ApplyDefaultSchedule(
    dl.gpu.GEMV(),              # Most specific: 1 spatial + 1 reduce
    dl.gpu.Decode(),            # Specific: decode-phase GEMV
    dl.gpu.Matmul(),            # Specific: 2 spatial + 1 reduce
    dl.gpu.Reduction(),         # General: any reduction
    dl.gpu.GeneralReduction(),  # Most general reduction
    dl.gpu.Fallback(),          # Catch-all
)(mod)

# WRONG order: fallback would match everything
# mod = dl.ApplyDefaultSchedule(
#     dl.gpu.Fallback(),      # This would match first!
#     dl.gpu.Matmul(),        # Never reached
# )(mod)
```

### 16.5.2 Applying Rules to Specific Functions

To apply DLight rules to a specific PrimFunc rather than the entire module:

```python
import tvm
from tvm import dlight as dl

# Get a specific function from the module
func_name = "my_matmul"
func = mod[func_name]

# Apply the Matmul rule to this specific function
with tvm.target.Target("nvidia/nvidia-a100"):
    rule = dl.gpu.Matmul()
    sch = rule.apply(func, tvm.target.Target("nvidia/nvidia-a100"))
    mod = sch.mod
```

### 16.5.3 Inspecting the Applied Schedule

After applying DLight, you can inspect the scheduled TIR to verify the transformations:

```python
import tvm
from tvm import dlight as dl

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Fallback(),
    )(mod)

# Print the scheduled TIR
for gv, func in scheduled_mod.functions.items():
    print(f"Function: {gv.name_hint}")
    print(func.script())
    print()

# Verify that GPU scheduling was applied (look for threadIdx/blockIdx bindings)
# A properly scheduled GPU kernel will contain:
# - T.launch_thread for blockIdx and threadIdx
# - Shared memory allocations (T.alloc_buffer with scope="shared")
# - Vectorized loads
```

---

## 16.6 Integration with Optimization Pipelines

### 16.6.1 DLight in the Relax Pipeline

DLight can be used as the scheduling backend in the Relax optimization pipeline:

```python
from tvm import relax
from tvm import dlight as dl

# Option 1: Use DLight directly in the pipeline
target = tvm.target.Target("nvidia/nvidia-a100")

# First apply standard Relax optimizations
mod = relax.get_pipeline("zero")(mod)

# Then apply DLight scheduling to the TIR functions
with target:
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Decode(),
        dl.gpu.Fallback(),
    )(mod)

# Build the optimized module
exec = relax.build(mod, target=target)
```

### 16.6.2 Combining DLight with MetaSchedule

DLight and MetaSchedule can be combined: use DLight for well-understood patterns and MetaSchedule for custom or unusual operations.

```python
from tvm import relax, meta_schedule as ms
from tvm import dlight as dl
from tvm.meta_schedule.database import JSONDatabase

# Step 1: Extract tasks
tasks = ms.extract_tasks(mod, target=target)

# Step 2: Classify tasks
dlight_tasks = []    # Tasks that match DLight patterns
tuning_tasks = []    # Tasks that need MetaSchedule tuning

for task in tasks:
    if is_dlight_pattern(task):  # Custom classification logic
        dlight_tasks.append(task)
    else:
        tuning_tasks.append(task)

# Step 3: Apply DLight to matching tasks
with target:
    for task in dlight_tasks:
        mod[task.task_name] = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.Fallback(),
        )(task.mod)[task.task_name]

# Step 4: Tune remaining tasks with MetaSchedule
db = ms.tune_tasks(
    tasks=tuning_tasks,
    target=target,
    max_trials_global=5000,
)

# Step 5: Build the final module
exec = relax.build(mod, target=target)
```

### 16.6.3 DLight as MetaSchedule Default

When using MetaSchedule's `ApplyDefaultSchedule`, DLight rules serve as the default scheduling strategy when no tuning data is available:

```python
from tvm.meta_schedule.tune import tune_tasks
from tvm import dlight as dl

# MetaSchedule will fall back to DLight rules for un-tuned operators
mod = relax.get_pipeline("static_shape_tuning")(
    mod,
    target=target,
    database=db,  # May be empty for first run
    default_schedule=dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Fallback(),
    ),
)
```

### 16.6.4 Using DLight in the Zero Pipeline

The "zero" pipeline is TVM's recommended pipeline for LLM inference and includes DLight rules by default:

```python
from tvm import relax

# The "zero" pipeline includes DLight GPU rules
target = tvm.target.Target("nvidia/nvidia-a100")
mod = relax.get_pipeline("zero")(mod)

# After the zero pipeline, TIR functions are already scheduled with DLight
# You can directly build
exec = relax.build(mod, target=target)

# The zero pipeline internally applies:
# 1. LegalizeOps (convert Relax ops to TIR)
# 2. DLight GPU rules (GEMV, Matmul, Reduction, Decode, Fallback)
# 3. FuseOpsByPattern (operator fusion)
# 4. FuseTIR (merge TIR functions)
```

---

## 16.7 Performance Characteristics

### 16.7.1 Expected Performance

| Pattern | DLight Performance | MetaSchedule (2000 trials) | Notes |
|---------|-------------------|---------------------------|-------|
| GEMV (M=4096) | ~85% peak BW | ~87% peak BW | Memory-bound, DLight is close to optimal |
| Matmul (1024x1024) | ~70% peak FLOPS | ~75% peak FLOPS | Compute-bound, MetaSchedule slightly better |
| Reduction | ~80% peak BW | ~82% peak BW | Memory-bound, similar performance |
| Element-wise | ~90% peak BW | ~90% peak BW | Trivially parallelizable |
| Decode (LLM) | ~80% peak BW | ~83% peak BW | Memory-bound with specific access pattern |

### 16.7.2 Compilation Time

| Approach | Compilation Time | Tuning Time |
|----------|-----------------|-------------|
| DLight | 1-5 seconds | None |
| MetaSchedule | 1-5 seconds | 30 min - 2 hours |
| Manual Schedule | Hours (development) | None |

### 16.7.3 When DLight Performance is Sufficient

DLight is the recommended approach when:

- The model uses standard operations (linear, attention, normalization, element-wise)
- Compilation speed is important (CI/CD, rapid prototyping)
- The target hardware is a well-known GPU (NVIDIA A100, H100, etc.)
- The model has been previously validated with DLight schedules
- You need deterministic, reproducible builds

### 16.7.4 When to Use MetaSchedule Instead

MetaSchedule is recommended when:

- The model contains custom or unusual operators
- Maximum performance is critical and worth the tuning investment
- The target hardware has unusual characteristics
- The operation has dimensions that don't match standard tiling patterns
- Previous tuning data is available (reducing the effective tuning time)

---

## 16.8 Example: Applying DLight to a Custom Operator

This section demonstrates the complete workflow of applying DLight rules to a custom operator.

### 16.8.1 Custom Softmax Operator

```python
from tvm.script import ir as I, tir as T
import tvm
from tvm import dlight as dl

@I.ir_module
class SoftmaxModule:
    @T.prim_func
    def softmax(
        A: T.Buffer((128, 1024), "float32"),
        B: T.Buffer((128, 1024), "float32"),
    ):
        # Step 1: Compute max for numerical stability
        max_val = T.alloc_buffer((128,), "float32")
        for i, j in T.grid(128, 1024):
            with T.block("max"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(A[vi, vj])
                T.writes(max_val[vi])
                with T.init():
                    max_val[vi] = T.float32("-inf")
                max_val[vi] = T.max(max_val[vi], A[vi, vj])

        # Step 2: Compute exp(x - max)
        exp_val = T.alloc_buffer((128, 1024), "float32")
        for i, j in T.grid(128, 1024):
            with T.block("exp"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj], max_val[vi])
                T.writes(exp_val[vi, vj])
                exp_val[vi, vj] = T.exp(A[vi, vj] - max_val[vi])

        # Step 3: Compute sum of exp values
        sum_val = T.alloc_buffer((128,), "float32")
        for i, j in T.grid(128, 1024):
            with T.block("sum"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(exp_val[vi, vj])
                T.writes(sum_val[vi])
                with T.init():
                    sum_val[vi] = T.float32(0)
                sum_val[vi] = sum_val[vi] + exp_val[vi, vj]

        # Step 4: Normalize
        for i, j in T.grid(128, 1024):
            with T.block("normalize"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(exp_val[vi, vj], sum_val[vi])
                T.writes(B[vi, vj])
                B[vi, vj] = exp_val[vi, vj] / sum_val[vi]

# Apply DLight rules
with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Reduction(),  # Matches "max" and "sum" blocks
        dl.gpu.Fallback(),   # Matches "exp" and "normalize" blocks
    )(SoftmaxModule)

print("Scheduled softmax kernel:")
print(scheduled_mod.script())
```

### 16.8.2 Custom Fused Attention Operator

```python
from tvm.script import ir as I, tir as T
import tvm
from tvm import dlight as dl

@I.ir_module
class AttentionModule:
    @T.prim_func
    def attention(
        Q: T.Buffer((1, 8, 128, 64), "float16"),   # [batch, heads, seq, dim]
        K: T.Buffer((1, 8, 128, 64), "float16"),
        V: T.Buffer((1, 8, 128, 64), "float16"),
        O: T.Buffer((1, 8, 128, 64), "float16"),
    ):
        # QK^T: [batch, heads, seq_q, seq_k]
        QK = T.alloc_buffer((1, 8, 128, 128), "float32")
        for b, h, i, j, k in T.grid(1, 8, 128, 128, 64):
            with T.block("QK"):
                vb, vh, vi, vj, vk = T.axis.remap("SSSSR", [b, h, i, j, k])
                T.reads(Q[vb, vh, vi, vk], K[vb, vh, vj, vk])
                T.writes(QK[vb, vh, vi, vj])
                with T.init():
                    QK[vb, vh, vi, vj] = T.float32(0)
                QK[vb, vh, vi, vj] = QK[vb, vh, vi, vj] + \
                    T.cast(Q[vb, vh, vi, vk], "float32") * T.cast(K[vb, vh, vj, vk], "float32")

        # Softmax is omitted for brevity (see Section 16.8.1)

        # Attention * V: [batch, heads, seq_q, dim]
        for b, h, i, j, k in T.grid(1, 8, 128, 64, 128):
            with T.block("AttnV"):
                vb, vh, vi, vj, vk = T.axis.remap("SSSSR", [b, h, i, j, k])
                T.reads(QK[vb, vh, vi, vk], V[vb, vh, vk, vj])
                T.writes(O[vb, vh, vi, vj])
                with T.init():
                    O[vb, vh, vi, vj] = T.float32(0)
                O[vb, vh, vi, vj] = O[vb, vh, vi, vj] + \
                    QK[vb, vh, vi, vk] * T.cast(V[vb, vh, vk, vj], "float32")

# Apply DLight rules
with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),    # Matches the batched matmul patterns
        dl.gpu.Fallback(),
    )(AttentionModule)
```

### 16.8.3 RMSNorm with DLight

```python
from tvm.script import ir as I, tir as T
import tvm
from tvm import dlight as dl

@I.ir_module
class RMSNormModule:
    @T.prim_func
    def rmsnorm(
        X: T.Buffer((1, 4096), "float32"),
        Weight: T.Buffer((4096,), "float32"),
        Y: T.Buffer((1, 4096), "float32"),
    ):
        # Compute sum of squares
        sum_sq = T.alloc_buffer((1,), "float32")
        for i, j in T.grid(1, 4096):
            with T.block("sum_sq"):
                vi, vj = T.axis.remap("SR", [i, j])
                T.reads(X[vi, vj])
                T.writes(sum_sq[vi])
                with T.init():
                    sum_sq[vi] = T.float32(0)
                sum_sq[vi] = sum_sq[vi] + X[vi, vj] * X[vi, vj]

        # Normalize and scale
        for i, j in T.grid(1, 4096):
            with T.block("output"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(X[vi, vj], sum_sq[vi], Weight[vj])
                T.writes(Y[vi, vj])
                Y[vi, vj] = X[vi, vj] / T.sqrt(sum_sq[vi] / T.float32(4096) + T.float32(1e-6)) * Weight[vj]

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Reduction(),  # Matches sum_sq reduction
        dl.gpu.Fallback(),   # Matches output element-wise
    )(RMSNormModule)
```

---

## 16.9 Writing Custom DLight Rules

Users can write custom DLight rules for domain-specific patterns.

### 16.9.1 Basic Custom Rule

```python
import tvm
from tvm import dlight as dl
from tvm import tir

class MyCustomGPURule(dl.ScheduleRule):
    """A custom DLight rule for a specific computational pattern."""

    def apply(self, func: tir.PrimFunc, target: tvm.target.Target) -> tvm.tir.Schedule:
        """
        Apply the scheduling rule to a PrimFunc.

        Parameters
        ----------
        func : tir.PrimFunc
            The function to schedule.
        target : Target
            The compilation target.

        Returns
        -------
        sch : tir.Schedule
            The scheduled module, or None if the rule does not match.
        """
        # Step 1: Analyze the function structure
        sch = tvm.tir.Schedule(tvm.IRModule.from_expr(func))

        # Step 2: Detect the pattern
        # ... custom pattern matching logic ...

        # Step 3: Apply the schedule
        # ... schedule transformations ...

        return sch

# Use the custom rule
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(
        MyCustomGPURule(),
        dl.gpu.Fallback(),  # Fallback for unmatched patterns
    )(mod)
```

### 16.9.2 Custom Rule with Pattern Detection

```python
class CustomConv2DRule(dl.ScheduleRule):
    """Custom rule for 2D convolution with specific tiling strategy."""

    def apply(self, func: tir.PrimFunc, target: tvm.target.Target):
        from tvm.dlight.analysis import detect_reduction

        sch = tvm.tir.Schedule(tvm.IRModule.from_expr(func))

        # Find all blocks
        blocks = sch.get_child_blocks(sch.get_loops(sch.get_block("root")))

        for block in blocks:
            block_rv = sch.get_block(block.name_hint)
            # Check if this is a reduction block with 4 spatial + 1 reduce
            reduction_info = detect_reduction(block)
            if reduction_info is None:
                continue

            spatial_axes, reduce_axes = reduction_info
            if len(spatial_axes) != 4 or len(reduce_axes) != 1:
                continue

            # Apply custom tiling for conv2d
            # ... tile NHWC and RS dimensions ...

        return sch
```

### 16.9.3 Guidelines for Writing Effective DLight Rules

1. **Be specific in pattern matching**: Return `None` early if the pattern does not match. Do not apply transformations that assume a specific structure without verification.

2. **Use analysis helpers**: Leverage `detect_reduction`, `is_broadcast_epilogue`, and other analysis functions to understand the block structure.

3. **Consider the target**: Check the target's attributes (shared memory size, max threads, etc.) before applying target-specific transformations.

4. **Validate the schedule**: After applying transformations, consider using `verify_gpu_code` to ensure correctness.

5. **Document the expected pattern**: Clearly document what pattern the rule matches and what schedule it produces.

6. **Handle edge cases**: Consider non-standard sizes, non-power-of-2 dimensions, empty dimensions, and degenerate cases.

7. **Test against known-good implementations**: Validate the scheduled output against reference implementations.

---

## 16.10 DLight vs MetaSchedule Comparison

### 16.10.1 Feature Comparison

| Feature | DLight | MetaSchedule |
|---------|--------|--------------|
| Speed | Fast (no search) | Slow (search-based) |
| Quality | Good for known patterns | Potentially better for unknown patterns |
| Use case | Common patterns | Custom/new patterns |
| Config | Minimal | Extensive |
| Determinism | Fully deterministic | Depends on search strategy |
| Reproducibility | Same input = same output | May vary across runs |
| Tuning data needed | None | Required for best results |
| Target flexibility | Works on any supported target | Target-specific tuning |

### 16.10.2 Decision Flowchart

```
Need to schedule a PrimFunc?
    |
    v
Is the pattern one of:
GEMV, Matmul, Reduction, Decode?
    |
    +-- Yes --> Use DLight
    |
    +-- No --> Is the pattern similar to a known pattern?
                    |
                    +-- Yes --> Write a custom DLight rule
                    |
                    +-- No --> Is compile time critical?
                                    |
                                    +-- Yes --> Use DLight Fallback
                                    |
                                    +-- No --> Use MetaSchedule
```

### 16.10.3 Migration from MetaSchedule to DLight

If you have been using MetaSchedule and want to migrate to DLight:

```python
# Before: MetaSchedule
from tvm import meta_schedule as ms

db = ms.database.JSONDatabase("tuning_records.json")
mod = ms.tune_tir(
    mod,
    target=target,
    max_trials_global=2000,
    database=db,
)

# After: DLight
from tvm import dlight as dl

with target:
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.GEMV(),
        dl.gpu.Matmul(),
        dl.gpu.Reduction(),
        dl.gpu.Decode(),
        dl.gpu.Fallback(),
    )(mod)

# Performance comparison
# Run both versions and compare latency
# If DLight is within 5-10% of MetaSchedule, it's usually worth the
# compilation speed improvement
```

---

## 16.11 Troubleshooting

### 16.11.1 Common Issues

**Issue: "No rule matched the PrimFunc"**

If no DLight rule matches your PrimFunc, the function will be left unscheduled and may fail during GPU code generation. To diagnose:

```python
# Check the PrimFunc structure
print(mod["my_func"].script())

# Look for:
# 1. Block structure - does it have T.block() calls?
# 2. Axis classification - are axes properly remapped as "S" or "R"?
# 3. Access patterns - are buffer reads/writes correct?
```

**Issue: Incorrect results after DLight scheduling**

This typically indicates a pattern mismatch where the rule applies a transformation that does not preserve the computation's semantics. To debug:

```python
# Step 1: Apply DLight and compare results
import numpy as np

# Build without DLight (CPU reference)
ref_mod = tvm.build(mod, target="llvm")
# Build with DLight (GPU)
with tvm.target.Target("nvidia/nvidia-a100"):
    gpu_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    gpu_exec = tvm.build(gpu_mod, target="nvidia/nvidia-a100")

# Compare outputs
```

**Issue: Poor performance with DLight**

Some patterns may not be well-optimized by the default DLight rules. To investigate:

```python
# Profile individual operators
import tvm

with tvm.target.Target("nvidia/nvidia-a100"):
    scheduled_mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.Fallback(),
    )(mod)

# Check the generated kernel
for gv, func in scheduled_mod.functions.items():
    print(f"Function: {gv.name_hint}")
    # Look for:
    # - Proper thread binding (blockIdx, threadIdx)
    # - Shared memory usage (for compute-bound ops)
    # - Vectorization (for memory-bound ops)
```

### 16.11.2 Debugging DLight Application

```python
import tvm
from tvm import dlight as dl
import logging

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)

# Apply DLight step by step
with tvm.target.Target("nvidia/nvidia-a100"):
    # Try each rule individually to see which one matches
    try:
        result = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
        print("GEMV rule matched")
    except Exception as e:
        print(f"GEMV rule did not match: {e}")

    try:
        result = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
        print("Matmul rule matched")
    except Exception as e:
        print(f"Matmul rule did not match: {e}")
```

---

## 16.12 Summary

| Component | Description | Key Rules |
|-----------|-------------|-----------|
| **GPU Rules** | Optimized schedules for GPU targets | GEMV, Matmul, Reduction, Decode, DecodeGEMV, GeneralReduction, Fallback |
| **CPU Rules** | Optimized schedules for CPU targets | Fallback |
| **Analysis Helpers** | Pattern detection utilities | `is_broadcast_epilogue`, `detect_dominated_var`, `detect_reduction`, `detect_linear_equation` |
| **Application** | Rule composition and ordering | `ApplyDefaultSchedule` |
| **Integration** | Pipeline and MetaSchedule integration | Relax pipeline, combined tuning, zero pipeline |
| **Custom Rules** | Extensibility framework | `ScheduleRule` base class |

DLight is TVM's recommended approach for scheduling standard operators, particularly in production deployments where compilation speed and reproducibility matter. For custom or unusual operators, it can be combined with MetaSchedule for the best of both worlds.
