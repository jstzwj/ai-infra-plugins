# CUTLASS - Chapter 27: Reduction Operations

This reference covers CUTLASS reduction operations, which are fundamental building blocks used throughout the library for combining partial results, performing Split-K reduction in GEMM, and implementing element-wise operations with reduction semantics.

---

## 27.1 Overview

Reduction operations in CUTLASS compute a single summary value (or a tensor of summary values) from a set of input elements by applying a binary operator associatively across the data. Reductions are used in several critical paths:

- **Split-K GEMM reduction**: Accumulating partial GEMM results computed by independent thread blocks across the K dimension.
- **Epilogue reduction**: Combining accumulator values with bias or residual tensors.
- **Tensor reduction**: Full-device reductions across arbitrary dimensions of a tensor.
- **Warp-level reduction**: Efficient intra-warp reductions using shuffle instructions.
- **Thread-level reduction**: Small fixed-size reductions within a single thread.

CUTLASS provides a layered hierarchy of reduction abstractions that map efficiently onto the GPU execution model:

| Level | Scope | Description |
|-------|-------|-------------|
| Thread | Single thread | Reduce a fixed-size array within one thread |
| Warp | Single warp (32 threads) | Reduce across threads in a warp using shuffles |
| Threadblock | Single CTA | Reduce across warps using shared memory |
| Device | Full grid | Multi-kernel or single-kernel full tensor reduction |

---

## 27.2 Thread-Level Reduction

### 27.2.1 `reduce<T, N, Op>`

The simplest reduction primitive operates on a fixed-size array within a single thread. It is defined in `cutlass/reduction/thread/reduce.h`.

```cpp
#include "cutlass/reduction/thread/reduce.h"

// Reduce an array of N elements using the specified operator
// T: element type
// N: number of elements (must be a power of two)
// Op: binary reduction operator (e.g., plus, maximum)

using ReduceSum = cutlass::reduction::thread::Reduce<float, 8, cutlass::plus<float>>;

float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
float result = ReduceSum::apply(data, cutlass::plus<float>());
// result == 36.0f
```

The implementation uses a logarithmic tree reduction pattern:

```cpp
namespace cutlass {
namespace reduction {
namespace thread {

template <typename T, int N, typename Op>
struct Reduce {
  static_assert(!(N & (N - 1)), "N must be a power of two");

  CUTLASS_HOST_DEVICE
  static T apply(T const *ptr, Op op) {
    T storage[N];

    // Copy input
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      storage[i] = ptr[i];
    }

    // Tree reduction
    CUTLASS_PRAGMA_UNROLL
    for (int n = N; n > 1; n /= 2) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < n / 2; ++i) {
        storage[i] = op(storage[i], storage[i + n / 2]);
      }
    }

    return storage[0];
  }
};

} // namespace thread
} // namespace reduction
} // namespace cutlass
```

### 27.2.2 Supported Operators

CUTLASS provides several function objects suitable for use as reduction operators:

| Operator | Header | Description |
|----------|--------|-------------|
| `cutlass::plus<T>` | `cutlass/functional.h` | Summation (a + b) |
| `cutlass::multiplies<T>` | `cutlass/functional.h` | Product (a * b) |
| `cutlass::maximum<T>` | `cutlass/functional.h` | Maximum value |
| `cutlass::minimum<T>` | `cutlass/functional.h` | Minimum value |
| `cutlass::bit_and<T>` | `cutlass/functional.h` | Bitwise AND |
| `cutlass::bit_or<T>` | `cutlass/functional.h` | Bitwise OR |
| `cutlass::logical_and<T>` | `cutlass/functional.h` | Logical AND |
| `cutlass::logical_or<T>` | `cutlass/functional.h` | Logical OR |

Example using maximum reduction:

```cpp
#include "cutlass/reduction/thread/reduce.h"
#include "cutlass/functional.h"

// Find the maximum of 4 float values
using ReduceMax = cutlass::reduction::thread::Reduce<float, 4, cutlass::maximum<float>>;

float values[4] = {3.14f, 2.71f, 1.41f, 1.73f};
float max_val = ReduceMax::apply(values, cutlass::maximum<float>());
// max_val == 3.14f
```

### 27.2.3 Custom Reduction Operators

You can define custom operators by creating a callable struct:

```cpp
struct CustomWeightedSum {
  float weight_a;
  float weight_b;

  CUTLASS_HOST_DEVICE
  float operator()(float a, float b) const {
    return weight_a * a + weight_b * b;
  }
};

// Use with thread-level reduce
using ReduceWeighted = cutlass::reduction::thread::Reduce<float, 4, CustomWeightedSum>;
CustomWeightedSum op{0.7f, 0.3f};
float result = ReduceWeighted::apply(data, op);
```

---

## 27.3 Warp-Level Reduction

### 27.3.1 Shuffle-Based Reduction

Warp-level reductions use CUDA shuffle intrinsics (`__shfl_down_sync`) to exchange data between lanes without using shared memory. This is highly efficient for reductions across the 32 threads in a warp.

```cpp
#include "cutlass/reduction/warp/reduce.h"

// Reduce across a warp using shuffle-down pattern
// This is typically used for reductions where each thread holds one value

template <typename T, typename Op>
CUTLASS_DEVICE T warp_reduce(T val, Op op) {
  // Assumes warpSize == 32
  val = op(val, __shfl_down_sync(0xffffffff, val, 16));
  val = op(val, __shfl_down_sync(0xffffffff, val, 8));
  val = op(val, __shfl_down_sync(0xffffffff, val, 4));
  val = op(val, __shfl_down_sync(0xffffffff, val, 2));
  val = op(val, __shfl_down_sync(0xffffffff, val, 1));
  return val;
}
```

### 27.3.2 `WarpReduce` in CUTLASS

CUTLASS provides a structured warp reduction in `cutlass/reduction/warp/reduce.h` that integrates with the layout system:

```cpp
#include "cutlass/reduction/warp/reduce.h"

// Define a warp reduction for a row-major arrangement
using WarpReduce = cutlass::reduction::warp::Reduce<
  float,                              // Element type
  cutlass::layout::RowMajor,          // Layout within warp
  4,                                  // Elements per thread per step
  8,                                  // Number of iterations
  cutlass::plus<float>                // Reduction operator
>;

// Within device code
__shared__ typename WarpReduce::Storage shared_storage;
WarpReduce warp_reduce(shared_storage);

// Each thread contributes its accumulated values
float thread_accumulator = /* partial sum */;
float warp_result = warp_reduce(thread_accumulator);
// warp_result is the sum across all threads in the warp
// (only valid in lane 0, or replicated depending on configuration)
```

### 27.3.3 Partial Warp Reductions

Sometimes only a subset of lanes participate in the reduction:

```cpp
// Reduce only the first N lanes of a warp
template <int N, typename T, typename Op>
CUTLASS_DEVICE T partial_warp_reduce(T val, Op op) {
  static_assert(N <= 32, "N must be <= warp size");
  unsigned mask = (1u << N) - 1;

  if (N >= 32) val = op(val, __shfl_down_sync(mask, val, 16));
  if (N >= 16) val = op(val, __shfl_down_sync(mask, val, 8));
  if (N >= 8)  val = op(val, __shfl_down_sync(mask, val, 4));
  if (N >= 4)  val = op(val, __shfl_down_sync(mask, val, 2));
  if (N >= 2)  val = op(val, __shfl_down_sync(mask, val, 1));

  return val;
}
```

---

## 27.4 Threadblock-Level Reduction

### 27.4.1 Cross-Warp Reduction

When a reduction must span multiple warps within a threadblock, shared memory is used as an intermediary. Each warp performs a warp-level reduction, writes its partial result to shared memory, and then a single warp reads all partial results and performs a final reduction.

```cpp
#include "cutlass/reduction/threadblock/reduce.h"

// Configuration
using ThreadblockReduce = cutlass::reduction::threadblock::Reduce<
  float,                              // Element type
  128,                                // Threadblock size (threads)
  4,                                  // Elements per thread
  cutlass::plus<float>                // Reduction operator
>;

// Shared memory for inter-warp communication
__shared__ typename ThreadblockReduce::Storage reduce_shared_storage;

// In device code
float thread_partial_sum = /* computed partial result */;

float block_result = ThreadblockReduce::reduce(
  reduce_shared_storage,
  thread_partial_sum,
  cutlass::plus<float>()
);
// block_result contains the sum across all threads in the block
// (valid only in thread 0)
```

### 27.4.2 Strided Reduction

When threads hold strided elements (e.g., in GEMM epilogue where each thread writes contiguous rows), CUTLASS provides strided reduction utilities:

```cpp
// Reduce elements where threads hold strided data
// Used in Split-K reduction where thread blocks write partial results
// to a workspace tensor

template <typename Element, int Threads, int ElementsPerThread>
struct StridedReduction {
  CUTLASS_DEVICE
  static Element reduce(
    Element *shared_ptr,
    Element thread_value,
    int thread_idx,
    int stride
  ) {
    // Write thread value to shared memory at strided position
    shared_ptr[thread_idx] = thread_value;
    __syncthreads();

    // Sequential reduction in the first warp
    if (thread_idx < Threads / 32) {
      Element sum = Element(0);
      for (int i = thread_idx; i < Threads; i += Threads / 32) {
        sum += shared_ptr[i];
      }
      shared_ptr[thread_idx] = sum;
    }
    __syncthreads();

    // Final warp reduction
    if (thread_idx < 32) {
      // Warp shuffle reduction on shared_ptr[thread_idx]
      // ...
    }

    return shared_ptr[0];
  }
};
```

---

## 27.5 Device-Level Tensor Reduction

### 27.5.1 `TensorReduce` Overview

The `TensorReduce` device-level operation reduces a tensor along one or more dimensions. It is defined in `cutlass/reduction/device/tensor_reduce.h` and provides a high-level interface similar to other CUTLASS device operations.

```cpp
#include "cutlass/reduction/device/tensor_reduce.h"
#include "cutlass/reduction/device/reduce_split_k.h"

// Reduce a 4D tensor along dimension 0 (batch dimension)
using TensorReduceOp = cutlass::reduction::device::TensorReduction<
  float,                              // Element type (output)
  float,                              // Element type (workspace/accumulator)
  cutlass::layout::TensorNHWC,        // Layout
  cutlass::plus<float>,               // Reduction operator
  128,                                // VectorLength (elements per memory access)
  64,                                 // Threads
  4                                   // ElementsPerAccess
>;

TensorReduceOp reduce_op;

// Arguments
typename TensorReduceOp::Arguments args(
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC>(src_ptr, src_stride),
  cutlass::TensorRef<float, cutlass::layout::TensorNHWC>(dst_ptr, dst_stride),
  cutlass::plus<float>(),
  float(0)  // identity element
);

// Initialize and run
auto status = reduce_op(args);
if (status != cutlass::Status::kSuccess) {
  // Handle error
}
```

### 27.5.2 TensorReduce Configuration

The `TensorReduction` template takes several important parameters:

| Parameter | Description |
|-----------|-------------|
| `ElementOutput` | Data type of the output tensor |
| `ElementAccumulator` | Data type used for accumulation (often wider, e.g., `float` for `half_t` input) |
| `Layout` | Tensor layout (must match input and output) |
| `Op` | Binary reduction operator |
| `VectorLength` | Number of contiguous elements processed per memory operation |
| `Threads` | Number of threads per threadblock |
| `ElementsPerAccess` | Elements loaded per thread per iteration |

### 27.5.3 Reducing Across Multiple Dimensions

To reduce across multiple dimensions, the reduction is performed sequentially one dimension at a time:

```cpp
// Reduce a NHWC tensor across H and W dimensions (spatial reduction)
// Step 1: Reduce across W
using ReduceW = cutlass::reduction::device::TensorReduction<
  float, float,
  cutlass::layout::TensorNHWC,
  cutlass::plus<float>, 128, 64, 4
>;

// Step 2: Reduce across H
using ReduceH = cutlass::reduction::device::TensorReduction<
  float, float,
  cutlass::layout::TensorNHWC,
  cutlass::plus<float>, 128, 64, 4
>;

// Execution:
// Input: [N, H, W, C]
// After ReduceW: [N, H, 1, C] (reduce dim=2)
// After ReduceH: [N, 1, 1, C] (reduce dim=1)
```

### 27.5.4 Workspace Requirements

TensorReduce may require device workspace memory for multi-pass reductions:

```cpp
typename TensorReduceOp::Arguments args(...);
size_t workspace_size = TensorReduceOp::get_workspace_size(args);

void *workspace = nullptr;
cudaMalloc(&workspace, workspace_size);

args.workspace.reset(workspace);

// Run reduction
reduce_op(args, workspace, stream);

cudaFree(workspace);
```

---

## 27.6 Split-K Reduction in GEMM

### 27.6.1 Motivation

In standard GEMM, a single thread block computes the entire K dimension for its assigned output tile. When K is very large, this can lead to:

1. Insufficient parallelism (not enough output tiles to fill the GPU).
2. Long kernel execution time per tile.
3. Register pressure from holding too many accumulators.

**Split-K** addresses these issues by partitioning the K dimension into `split_k_slices` independent chunks, each processed by a separate thread block. The partial results are then combined using a reduction.

### 27.6.2 Split-K in CUTLASS 2.x

```cpp
#include "cutlass/gemm/device/gemm.h"

using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t, cutlass::layout::RowMajor,
  cutlass::half_t, cutlass::layout::ColumnMajor,
  float, cutlass::layout::RowMajor,
  float,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,
  cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2,                                  // Stages
  cutlass::arch::OpMultiplyAdd
>;

// Launch with Split-K
int split_k_slices = 4;

cutlass::gemm::GemmCoord problem_size(M, N, K);

Gemm gemm_op;
auto status = gemm_op.initialize({
  problem_size,
  split_k_slices                      // Split-K dimension
});
status = gemm_op(stream);

// The GEMM kernel writes partial results to workspace,
// then an internal reduction kernel combines them.
```

### 27.6.3 Split-K Reduction Kernel

The Split-K reduction kernel reads partial results from the workspace tensor and combines them using a specified operator (typically summation):

```cpp
#include "cutlass/reduction/device/reduce_split_k.h"

// Define the Split-K reduction operation
using SplitKReduce = cutlass::reduction::device::ReduceSplitK<
  cutlass::layout::RowMajor,          // Output layout
  float,                              // Element type (output)
  float,                              // Element type (accumulator)
  cutlass::plus<float>                // Reduction operator
>;

// After GEMM produces partial results in workspace:
// workspace shape: [split_k_slices, M, N]
// output shape: [M, N]

SplitKReduce reduction_op;

typename SplitKReduce::Arguments reduce_args(
  cutlass::TensorRef<float, cutlass::layout::RowMajor>(
    workspace_ptr, cutlass::layout::RowMajor::Stride(N)),
  cutlass::TensorRef<float, cutlass::layout::RowMajor>(
    output_ptr, cutlass::layout::RowMajor::Stride(N)),
  cutlass::TensorRef<float, cutlass::layout::RowMajor>(
    output_ptr, cutlass::layout::RowMajor::Stride(N)),
  {M, N, split_k_slices},             // Extent: rows, cols, slices
  cutlass::plus<float>(),
  1.0f,                               // alpha
  0.0f                                // beta
);

reduction_op(reduce_args, workspace, stream);
```

### 27.6.4 Split-K with Epilogue Fusion

Split-K reduction can be combined with epilogue operations. The reduction kernel applies the epilogue transformation after combining partial results:

```cpp
// Epilogue with bias addition after Split-K reduction
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
  float,                              // ElementOutput
  4,                                  // ElementsPerAccess
  float,                              // ElementAccumulator
  float,                              // ElementCompute
  cutlass::epilogue::thread::ScaleType::NoBetaScaling
>;

// The Split-K reduction applies the epilogue:
// D = alpha * sum(partial_i) + bias
// rather than:
// D = sum(partial_i) and then apply epilogue separately
```

### 27.6.5 Choosing the Split-K Factor

The optimal split-K factor depends on the problem dimensions and hardware:

```cpp
// Heuristic for choosing split_k_slices
int choose_split_k(int M, int N, int K, int sm_count) {
  // Estimate the number of output tiles
  int tile_m = 128;
  int tile_n = 128;
  int num_tiles = (M / tile_m) * (N / tile_n);

  // If we have many more SMs than tiles, use Split-K
  if (sm_count > num_tiles * 2) {
    int min_k_per_slice = 256; // minimum K per slice for efficiency
    int max_slices = K / min_k_per_slice;
    int desired_slices = sm_count / max(num_tiles, 1);
    return min(max_slices, max(1, desired_slices));
  }

  return 1; // No Split-K needed
}
```

Guidelines for Split-K:
- Use Split-K when the number of output tiles (`ceil(M/tile_m) * ceil(N/tile_n)`) is small relative to SM count.
- Each slice should have at least 256 elements in K to amortize startup overhead.
- Split-K introduces extra global memory traffic for the workspace; the performance tradeoff depends on the compute-to-memory ratio.
- For very small K (K < 256), Split-K is unlikely to help.

---

## 27.7 Reduction with Element-wise Operations

### 27.7.1 Fused Reduction and Element-wise Operations

CUTLASS supports fusing element-wise operations with reductions, avoiding redundant memory round-trips. This is commonly used in:

- **Softmax**: Compute the maximum (reduce with `max`), then the sum (reduce with `+`), then divide.
- **Layer normalization**: Compute mean and variance via reduction, then normalize.
- **Bias addition with reduction**: Add bias during the Split-K reduction phase.

```cpp
// Fused bias addition during Split-K reduction
using ReduceSplitKWithBias = cutlass::reduction::device::ReduceSplitK<
  cutlass::layout::RowMajor,
  cutlass::half_t,                    // Output type
  float,                              // Accumulator type
  cutlass::plus<float>,
  cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float>
>;

// The reduction applies:
// D = alpha * reduce(partial_results) + beta * C + bias
```

### 27.7.2 Reduction Followed by Transformation

A common pattern is to perform a reduction and then transform the result:

```cpp
// Softmax reduction pattern (conceptual)
// 1. Reduce-max across the last dimension
// 2. Subtract the max from each element
// 3. Compute exp
// 4. Reduce-sum across the last dimension
// 5. Divide each element by the sum

// Step 1: Max reduction
using MaxReduce = cutlass::reduction::device::TensorReduction<
  float, float,
  cutlass::layout::RowMajor,
  cutlass::maximum<float>, 1, 128, 1
>;

// Step 4: Sum reduction
using SumReduce = cutlass::reduction::device::TensorReduction<
  float, float,
  cutlass::layout::RowMajor,
  cutlass::plus<float>, 1, 128, 1
>;
```

---

## 27.8 TensorReduce Configuration Details

### 27.8.1 Arguments Structure

```cpp
struct Arguments {
  // Input tensor reference
  cutlass::TensorRef<Element, Layout> src;

  // Output tensor reference (reduced tensor)
  cutlass::TensorRef<Element, Layout> dst;

  // Reduction operator instance
  Op reduction_op;

  // Identity element for the reduction
  Element identity;

  // Problem size (extents of each dimension)
  cutlass::Coord<Rank> extent;

  // Dimension to reduce along
  int reduction_dim;
};
```

### 27.8.2 Tuning Parameters

The performance of TensorReduce depends on several tuning knobs:

```cpp
// High-throughput configuration for large tensors
using FastReduce = cutlass::reduction::device::TensorReduction<
  float, float,
  cutlass::layout::RowMajor,
  cutlass::plus<float>,
  128,    // VectorLength: process 128 contiguous elements per access
  256,    // Threads: large threadblock for occupancy
  8       // ElementsPerAccess: high memory throughput
>;

// Low-latency configuration for small tensors
using SmallReduce = cutlass::reduction::device::TensorReduction<
  float, float,
  cutlass::layout::RowMajor,
  cutlass::plus<float>,
  4,      // VectorLength: small vector
  32,     // Threads: single warp
  1       // ElementsPerAccess
>;
```

### 27.8.3 Grid Launch Configuration

```cpp
// CUTLASS computes the grid dimensions automatically based on the
// problem size and the reduction dimension
typename TensorReduceOp::Params params(args);

dim3 grid = TensorReduceOp::grid_shape(params);
dim3 block = TensorReduceOp::block_shape();

// Typical grid computation:
// grid.x = ceil(extent[non-reduced-dims] / (Threads * ElementsPerAccess))
// grid.y = 1
// grid.z = 1
```

---

## 27.9 Reduction Across Arbitrary Dimensions

### 27.9.1 Dimension Selection

CUTLASS reductions can target any dimension of a multi-dimensional tensor. The choice of reduction dimension affects the memory access pattern:

```cpp
// Reducing along contiguous (innermost) dimension is most efficient
// because it enables vectorized memory access.

// NHWC layout, reducing C (dim=3, contiguous):
//   - Optimal: contiguous elements can be loaded with vectorized reads
//   - Each thread reduces a contiguous chunk

// NHWC layout, reducing N (dim=0, outermost):
//   - Non-optimal: strided access pattern
//   - Requires each thread to stride through memory
//   - May benefit from transposing or using a different layout
```

### 27.9.2 Multi-Dimension Reduction Strategy

For reducing multiple dimensions, CUTLASS typically uses a sequential approach:

```cpp
// Reduce a [N, H, W, C] tensor to [N, C] by reducing H and W

// Option 1: Sequential single-dimension reductions
//   Step 1: Reduce dim=2 (W) -> [N, H, 1, C]
//   Step 2: Reduce dim=1 (H) -> [N, 1, 1, C] = [N, C]

// Option 2: Transpose + reduce
//   Step 1: Transpose to [N, C, H, W]
//   Step 2: Reduce dim=2,3 together -> [N, C]
//   This may be more efficient for large H*W due to better memory access

// Option 3: Custom kernel for specific pattern (e.g., fused attention)
```

### 27.9.3 Reduction with Predication

When the tensor extent is not aligned to the threadblock tile size, boundary handling is required:

```cpp
// The reduction kernel uses predicated access to handle edge cases
CUTLASS_DEVICE
void reduce_with_predication(
  float *output,
  float const *input,
  int num_elements,
  int thread_idx,
  int block_dim
) {
  float sum = 0.0f;

  // Each thread processes strided elements
  for (int idx = thread_idx; idx < num_elements; idx += block_dim) {
    sum += input[idx];
  }

  // Warp-level reduction
  sum = warp_reduce(sum, cutlass::plus<float>());

  // Write result (only lane 0)
  if (thread_idx % 32 == 0) {
    atomicAdd(output, sum);
  }
}
```

---

## 27.10 Advanced Reduction Patterns

### 27.10.1 Segmented Reduction

Segmented reduction performs independent reductions on non-overlapping segments of a tensor:

```cpp
// Reduce each row of a matrix independently
// Input: [M, K], Output: [M], reduction along dim=1

template <int BlockSize, int ElementsPerThread>
__global__ void segmented_row_reduce(
  float *output,
  float const *input,
  int M, int K
) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= M) return;

  float const *row_ptr = input + row * K;
  float sum = 0.0f;

  // Each thread reduces a portion of the row
  for (int col = tid * ElementsPerThread; col < K; col += BlockSize * ElementsPerThread) {
    #pragma unroll
    for (int i = 0; i < ElementsPerThread && (col + i) < K; ++i) {
      sum += row_ptr[col + i];
    }
  }

  // Warp-level reduction
  sum = warp_reduce(sum, cutlass::plus<float>());

  // Cross-warp reduction via shared memory
  __shared__ float shared_sum[BlockSize / 32];
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0) {
    shared_sum[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane_id < BlockSize / 32) ? shared_sum[lane_id] : 0.0f;
    sum = warp_reduce(sum, cutlass::plus<float>());

    if (lane_id == 0) {
      output[row] = sum;
    }
  }
}
```

### 27.10.2 Reduction with Index Tracking (ArgMin/ArgMax)

CUTLASS provides utilities for reductions that also track the index of the extremal element:

```cpp
// ArgMax reduction: find maximum value and its index
template <typename T>
struct ArgMaxOp {
  struct Pair {
    T value;
    int index;
  };

  CUTLASS_HOST_DEVICE
  Pair operator()(Pair const &a, Pair const &b) const {
    return (a.value >= b.value) ? a : b;
  }
};

// Usage in warp-level argmax
CUTLASS_DEVICE
ArgMaxOp<float>::Pair warp_argmax(ArgMaxOp<float>::Pair val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    auto other = __shfl_down_sync(0xffffffff, val, offset);
    if (other.value > val.value) {
      val = other;
    }
  }
  return val;
}
```

### 27.10.3 Matrix Reduction (Row/Column)

CUTLASS provides specialized reduction for matrix row and column reduction:

```cpp
#include "cutlass/reduction/kernel/reduce_split_k.h"

// Row reduction: reduce each row to a single value
// Column reduction: reduce each column to a single value

// Row reduction (reduce along columns)
using RowReduce = cutlass::reduction::kernel::ReduceSplitK<
  cutlass::layout::RowMajor,
  float,                              // Output
  float,                              // Accumulator
  128,                                // Threads
  4,                                  // ElementsPerAccess
  cutlass::plus<float>                // Operator
>;
```

---

## 27.11 Performance Considerations

### 27.11.1 Memory Access Patterns

Efficient reductions require careful attention to memory access patterns:

1. **Coalesced reads**: Ensure consecutive threads read consecutive memory addresses.
2. **Vectorized loads**: Use `VectorLength > 1` when the data is contiguous in the reduction dimension.
3. **Bank conflict avoidance**: In shared memory reduction, use padding to avoid bank conflicts.

```cpp
// Shared memory reduction with bank conflict avoidance
__shared__ float smem[128 + 4]; // +4 padding to avoid bank conflicts

// Write partial results
smem[threadIdx.x] = partial_sum;
__syncthreads();

// Reduction with bank-conflict-free access
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (threadIdx.x < stride) {
    smem[threadIdx.x] += smem[threadIdx.x + stride];
  }
  __syncthreads();
}
```

### 27.11.2 Occupancy Considerations

Reduction kernels are typically memory-bandwidth-bound. Key tuning knobs:

| Parameter | Recommendation |
|-----------|---------------|
| Block size | 128-256 threads for good occupancy |
| Elements per thread | 4-8 for bandwidth saturation |
| Grid size | At least 2x SM count for good utilization |
| Shared memory | Minimize usage to allow more concurrent blocks |

### 27.11.3 Numerical Considerations

When reducing floating-point values, the order of operations affects the result:

```cpp
// For highest accuracy, use Kahan summation during reduction
struct KahanSum {
  float sum = 0.0f;
  float compensation = 0.0f;

  CUTLASS_HOST_DEVICE
  void add(float value) {
    float y = value - compensation;
    float t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
};
```

---

## 27.12 Summary

CUTLASS reduction operations provide a comprehensive hierarchy from thread-level to device-level reductions:

- **Thread-level** (`reduce<T, N, Op>`): Fixed-size logarithmic tree reduction within a single thread.
- **Warp-level**: Shuffle-based reduction across 32 threads, the most efficient building block.
- **Threadblock-level**: Shared-memory reduction spanning multiple warps.
- **Device-level** (`TensorReduce`): Full-grid tensor reduction with workspace support.
- **Split-K reduction**: Essential for GEMM parallelism when K is large relative to M and N.
- **Fused reductions**: Combine element-wise operations with reduction to avoid extra memory traffic.

These primitives are used throughout CUTLASS internally (Split-K, epilogue fusion) and can also be used directly by applications that need custom reduction patterns.
