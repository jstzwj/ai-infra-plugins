# CUTLASS - Chapter 12: Convolution Operations

This reference covers convolution operations in CUTLASS, which are implemented as implicit GEMM (General Matrix-Matrix Multiply) operations. CUTLASS provides highly optimized convolution kernels for forward propagation (fprop), data gradient (dgrad), weight gradient (wgrad), and deconvolution (deconv) across 2D and 3D convolution scenarios.

---

## 12.1 Convolution as Implicit GEMM

CUTLASS implements convolution operations by reducing them to GEMM problems. This technique, called **implicit GEMM**, reformulates the convolution operation as a matrix multiplication where:

- The input activation tensor is treated as one matrix operand.
- The filter (weight) tensor is treated as the other matrix operand.
- The convolution operation is computed as a matrix product with implicit data rearrangement.

The key insight is that a 2D convolution with input tensor `(N, H, W, C)` and filter tensor `(R, S, C, K)` producing output `(N, P, Q, K)` can be expressed as a GEMM:

```
Output (NPQ x K) = ActMatrix (NPQ x RSC) * FilterMatrix (RSC x K)
```

Where:
- `NPQ = N * P * Q` represents the output spatial dimensions flattened.
- `RSC = R * S * C` represents the filter spatial and channel dimensions flattened.
- `P = (H + padding_h - R) / stride_h + 1` is the output height.
- `Q = (W + padding_w - S) / stride_h + 1` is the output width.

The "implicit" part means that the activation matrix is not materialized in memory. Instead, the GEMM kernel reads activation data directly from its original layout, computing the implicit index mapping on the fly. This avoids the memory overhead and bandwidth cost of explicit im2col (image-to-column) transformation.

---

## 12.2 Convolution Types

CUTLASS supports four primary convolution types, each corresponding to a different phase of neural network training or inference.

### 12.2.1 kFprop (Forward Propagation)

Forward convolution computes the output activations from input activations and filters. This is the standard convolution used during inference and the forward pass of training.

**Mathematical definition:**

```
Output(n, p, q, k) = sum_{r, s, c} Input(n, h, w, c) * Filter(r, s, c, k)
```

where `h = p * stride_h + r - pad_h` and `w = q * stride_w + s - pad_w`.

**Implicit GEMM formulation:**

```
GEMM M dimension = N * P * Q  (number of output activations)
GEMM N dimension = K           (number of output channels)
GEMM K dimension = R * S * C  (filter elements per output channel)
```

**CUTLASS enum:**

```cpp
cutlass::conv::Operator::kFprop
```

### 12.2.2 kDgrad (Data Gradient / Backward Input)

Data gradient convolution computes the gradient of the loss with respect to the input activations. This is used during the backward pass of training to propagate error signals.

**Mathematical definition:**

```
GradInput(n, h, w, c) = sum_{p, q, k} GradOutput(n, p, q, k) * Filter(r, s, c, k)
```

where `r = h - (p * stride_h - pad_h)` and `s = w - (q * stride_w - pad_w)`.

This is equivalent to a transposed convolution (fractionally strided convolution) with the filter flipped.

**Implicit GEMM formulation:**

```
GEMM M dimension = N * H * W  (number of input activations)
GEMM N dimension = C           (number of input channels)
GEMM K dimension = P * Q * K  (output elements contributing to each input)
```

**CUTLASS enum:**

```cpp
cutlass::conv::Operator::kDgrad
```

### 12.2.3 kWgrad (Weight Gradient / Backward Filter)

Weight gradient convolution computes the gradient of the loss with respect to the filter weights.

**Mathematical definition:**

```
GradFilter(r, s, c, k) = sum_{n, p, q} Input(n, h, w, c) * GradOutput(n, p, q, k)
```

where `h = p * stride_h + r - pad_h` and `w = q * stride_w + s - pad_w`.

**Implicit GEMM formulation:**

```
GEMM M dimension = R * S * C  (filter spatial + input channel)
GEMM N dimension = K           (output channels)
GEMM K dimension = N * P * Q  (batch * output spatial)
```

**CUTLASS enum:**

```cpp
cutlass::conv::Operator::kWgrad
```

### 12.2.4 kDeconv (Deconvolution)

Deconvolution (also known as transposed convolution) is the reverse operation of convolution. It can be implemented as a dgrad-style convolution with appropriate parameter interpretation.

**CUTLASS enum:**

```cpp
cutlass::conv::Operator::kDeconv
```

---

## 12.3 Conv2D Problem Size

The `Conv2dProblemSize` struct encapsulates all parameters needed to define a 2D convolution problem. It uses the NHWC (batch-height-width-channel) tensor layout convention.

### 12.3.1 Struct Definition

```cpp
struct Conv2dProblemSize {
    // Tensor sizes (NHWC layout)
    int N;          // Batch size
    int H;          // Input height
    int W;          // Input width
    int C;          // Input channels

    // Filter size
    int K;          // Output channels (number of filters)
    int R;          // Filter height
    int S;          // Filter width

    // Padding
    int pad_h;      // Padding in height dimension
    int pad_w;      // Padding in width dimension

    // Stride
    int stride_h;   // Stride in height dimension
    int stride_w;   // Stride in width dimension

    // Dilation
    int dilation_h; // Dilation in height dimension
    int dilation_w; // Dilation in width dimension

    // Output size (computed from above)
    int P;          // Output height
    int Q;          // Output width

    // Optional: mode and split-k
    cutlass::conv::Mode mode;       // kCrossCorrelation or kConvolution
    int split_k_slices;             // Number of split-k slices
    int groups;                     // Number of convolution groups
};
```

### 12.3.2 Constructor Usage

```cpp
#include "cutlass/conv/convolution.h"

// Define a convolution problem size
cutlass::conv::Conv2dProblemSize problem_size(
    /*N=*/32,        // Batch size
    /*H=*/28,        // Input height
    /*W=*/28,        // Input width
    /*C=*/64,        // Input channels
    /*K=*/128,       // Output channels
    /*R=*/3,         // Filter height
    /*S=*/3,         // Filter width
    /*pad_h=*/1,     // Padding height
    /*pad_w=*/1,     // Padding width
    /*stride_h=*/1,  // Stride height
    /*stride_w=*/1,  // Stride width
    /*dilation_h=*/1,// Dilation height
    /*dilation_w=*/1,// Dilation width
    cutlass::conv::Mode::kCrossCorrelation,
    /*split_k_slices=*/1,
    /*groups=*/1
);

// Access computed output dimensions
int P = problem_size.P;  // Output height = 28
int Q = problem_size.Q;  // Output width = 28
```

### 12.3.3 Output Size Computation

The output spatial dimensions are computed automatically:

```
P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1
Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1
```

### 12.3.4 Dilation

Dilation introduces spacing between filter elements, effectively increasing the receptive field without increasing the number of parameters:

```cpp
// Standard 3x3 convolution (dilation = 1)
// Filter samples: (0,0), (0,1), (0,2), (1,0), ..., (2,2)

// Dilated 3x3 convolution (dilation = 2)
// Filter samples: (0,0), (0,2), (0,4), (2,0), ..., (4,4)
// Effective receptive field: 5x5

cutlass::conv::Conv2dProblemSize dilated_problem(
    1, 32, 32, 64,    // N, H, W, C
    128, 3, 3,         // K, R, S
    2, 2,              // pad_h, pad_w
    1, 1,              // stride_h, stride_w
    2, 2,              // dilation_h, dilation_w <-- dilation = 2
    cutlass::conv::Mode::kCrossCorrelation,
    1, 1
);
// Output: P = (32 + 4 - 4 - 1)/1 + 1 = 32, Q = 32
```

### 12.3.5 Split-K

Split-K parallelizes the reduction dimension (K in implicit GEMM) across multiple threadblocks, each computing a partial result that is later reduced:

```cpp
// Split-K with 4 slices
cutlass::conv::Conv2dProblemSize split_k_problem(
    32, 28, 28, 64,
    128, 3, 3,
    1, 1,
    1, 1,
    1, 1,
    cutlass::conv::Mode::kCrossCorrelation,
    4,  // split_k_slices = 4
    1
);

// Each threadblock computes 1/4 of the K dimension
// A separate reduction kernel combines the partial results
```

### 12.3.6 Grouped Convolution

Grouped convolution divides input and output channels into groups, with each group using independent filters:

```cpp
// Grouped convolution with 4 groups
cutlass::conv::Conv2dProblemSize grouped_problem(
    32, 28, 28, 64,    // N, H, W, C = 64
    128, 3, 3,          // K = 128, R, S
    1, 1,
    1, 1,
    1, 1,
    cutlass::conv::Mode::kCrossCorrelation,
    1,
    4   // groups = 4, so each group has C/G=16 input channels and K/G=32 output channels
);
```

---

## 12.4 Conv3D Problem Size

The `Conv3dProblemSize` struct handles 3D convolutions used in video processing, medical imaging, and volumetric data analysis. It uses the NDHWC (batch-depth-height-width-channel) layout.

```cpp
struct Conv3dProblemSize {
    int N;          // Batch size
    int D;          // Input depth
    int H;          // Input height
    int W;          // Input width
    int C;          // Input channels

    int K;          // Output channels
    int T;          // Filter depth (time)
    int R;          // Filter height
    int S;          // Filter width

    int pad_d;      // Padding in depth dimension
    int pad_h;      // Padding in height dimension
    int pad_w;      // Padding in width dimension

    int stride_d;   // Stride in depth dimension
    int stride_h;   // Stride in height dimension
    int stride_w;   // Stride in width dimension

    int dilation_d; // Dilation in depth dimension
    int dilation_h; // Dilation in height dimension
    int dilation_w; // Dilation in width dimension

    int Z;          // Output depth
    int P;          // Output height
    int Q;          // Output width

    cutlass::conv::Mode mode;
    int split_k_slices;
    int groups;
};
```

**Example: 3D convolution for video processing**

```cpp
cutlass::conv::Conv3dProblemSize conv3d_problem(
    /*N=*/8,          // Batch size (8 video clips)
    /*D=*/16,         // Depth (16 frames)
    /*H=*/112,        // Height
    /*W=*/112,        // Width
    /*C=*/3,          // Channels (RGB)
    /*K=*/64,         // Output channels
    /*T=*/3,          // Temporal filter size
    /*R=*/3,          // Spatial filter height
    /*S=*/3,          // Spatial filter width
    /*pad_d=*/1, /*pad_h=*/1, /*pad_w=*/1,
    /*stride_d=*/1, /*stride_h=*/1, /*stride_w=*/1,
    /*dilation_d=*/1, /*dilation_h=*/1, /*dilation_w=*/1,
    cutlass::conv::Mode::kCrossCorrelation,
    1, 1
);
// Output: Z=16, P=112, Q=112
```

---

## 12.5 Convolution Modes

### 12.5.1 kCrossCorrelation (Default)

Cross-correlation is the standard mode used in deep learning frameworks. The filter is applied without flipping:

```
Output(n, p, q, k) = sum_{r,s,c} Input(n, p*stride_h + r, q*stride_w + s, c) * Filter(r, s, c, k)
```

This is the default in PyTorch, TensorFlow, and most deep learning frameworks.

```cpp
cutlass::conv::Mode::kCrossCorrelation
```

### 12.5.2 kConvolution

True convolution applies the filter with a 180-degree rotation:

```
Output(n, p, q, k) = sum_{r,s,c} Input(n, p*stride_h - r, q*stride_w - s, c) * Filter(r, s, c, k)
```

```cpp
cutlass::conv::Mode::kConvolution
```

The difference between the two modes is whether the filter is flipped before application. In practice, `kCrossCorrelation` is almost always used in deep learning.

---

## 12.6 Convolution Algorithms

CUTLASS provides multiple algorithm variants optimized for different convolution scenarios:

### 12.6.1 kAnalytic

The analytic algorithm uses a direct index mapping that is mathematically correct for all parameter combinations (including non-unit strides, dilation, and arbitrary padding).

```cpp
cutlass::conv::Algorithm::kAnalytic
```

- **Pros**: Correct for all parameter combinations; handles edge cases properly.
- **Cons**: May not achieve peak performance for specific common cases.
- **Use case**: General-purpose convolution with arbitrary parameters.

### 12.6.2 kOptimized

The optimized algorithm uses precomputed index tables and specialized index computation to reduce the overhead of the implicit GEMM indexing.

```cpp
cutlass::conv::Algorithm::kOptimized
```

- **Pros**: Better performance for common configurations (unit dilation, aligned strides).
- **Cons**: May have restrictions on supported parameter combinations.
- **Use case**: Production inference and training with standard configurations.

### 12.6.3 kFixedChannels

The fixed-channels algorithm is optimized for cases where the number of input channels (C) is fixed to a specific value (e.g., C = 4 for RGBA images, C = 3 for RGB, C = 1 for grayscale).

```cpp
cutlass::conv::Algorithm::kFixedChannels
```

- **Use case**: Specialized inference for specific channel counts.

### 12.6.4 kFewChannels

The few-channels algorithm is optimized for small numbers of input channels (e.g., C <= 8), which is common in early layers of CNNs and point cloud processing.

```cpp
cutlass::conv::Algorithm::kFewChannels
```

- **Use case**: First convolutional layer of CNNs, depthwise convolution, point cloud networks.

---

## 12.7 Device-Level Convolution API

### 12.7.1 CUTLASS 2.x Device Convolution

CUTLASS 2.x provides device-level convolution operations through the `Conv2d` and `Conv3d` device classes:

```cpp
#include "cutlass/conv/device/implicit_gemm_convolution.h"

// Define types
using ElementA = cutlass::half_t;        // Activation type
using ElementB = cutlass::half_t;        // Filter type
using ElementC = float;                  // Source/output type
using ElementAccumulator = float;        // Accumulator type

using LayoutA = cutlass::layout::TensorNHWC;
using LayoutB = cutlass::layout::TensorNHWC;
using LayoutC = cutlass::layout::TensorNHWC;

// Define the convolution operation
using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass::conv::kernel::ImplicitGemmConvolution<
        cutlass::conv::threadblock::ImplicitGemmPipelined<
            cutlass::gemm::GemmShape<128, 128, 32>,    // Threadblock shape
            cutlass::gemm::GemmShape<64, 64, 32>,      // Warp shape
            cutlass::gemm::GemmShape<16, 8, 16>,       // Instruction shape
            ElementA, LayoutA,
            ElementB, LayoutB,
            ElementC, LayoutC,
            ElementAccumulator,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::conv::Operator::kFprop
        >
    >
>;

// Set up arguments
typename Conv2dFprop::Arguments args(
    problem_size,
    {ptr_A, stride_A},     // Activation tensor (NHWC)
    {ptr_B, stride_B},     // Filter tensor (KRSC)
    {ptr_C, stride_C},     // Source tensor (NPQK)
    {ptr_D, stride_D},     // Output tensor (NPQK)
    {alpha, beta},         // Epilogue scalars
    split_k_slices
);

// Create and run the operation
Conv2dFprop conv_op;
cutlass::Status status = conv_op(args);

// Handle workspace for split-k
size_t workspace_size = conv_op.get_workspace_size(args);
void *workspace;
cudaMalloc(&workspace, workspace_size);
status = conv_op.initialize(args, workspace);
status = conv_op.run();
```

### 12.7.2 Stride Computation

Strides for NHWC tensors are computed as follows:

```cpp
// Input activation tensor (N, H, W, C) - NHWC layout
int64_t stride_A = H * W * C;     // Stride to next batch element
// Within a batch element, elements are stored as (h, w, c) with stride:
// stride_h = W * C
// stride_w = C
// stride_c = 1

// Filter tensor (K, R, S, C) - KRSC layout for fprop
int64_t stride_B = R * S * C;     // Stride to next output channel
// stride_r = S * C
// stride_s = C
// stride_c = 1

// Output tensor (N, P, Q, K) - NPQK layout
int64_t stride_C = P * Q * K;     // Stride to next batch element
// stride_p = Q * K
// stride_q = K
// stride_k = 1

// Using cutlass TensorRef:
cutlass::TensorRef<ElementA, LayoutA> ref_A(ptr_A, LayoutA::packed({N, H, W, C}));
cutlass::TensorRef<ElementB, LayoutB> ref_B(ptr_B, LayoutB::packed({K, R, S, C}));
cutlass::TensorRef<ElementC, LayoutC> ref_C(ptr_C, LayoutC::packed({N, P, Q, K}));
```

---

## 12.8 Implicit GEMM Implementation

### 12.8.1 Index Mapping

The core of the implicit GEMM approach is the index mapping that translates a GEMM (M, N, K) index to the corresponding convolution tensor indices.

**For fprop convolution:**

```
GEMM index (m, n, k) maps to:
  Output: (n_m, p_m, q_m, K_n) where:
    n_m = m / (P * Q)
    pq = m % (P * Q)
    p_m = pq / Q
    q_m = pq % Q
    K_n = n

  Activation: (n_m, h, w, c_k) where:
    k_decomp: rs = k / C, c_k = k % C
    r_rs = rs / S, s_rs = rs % S
    h = p_m * stride_h + r_rs * dilation_h - pad_h
    w = q_m * stride_w + s_rs * dilation_w - pad_w

  Filter: (K_n, r_rs, s_rs, c_k)
```

### 12.8.2 ImplicitGemmPipelined

The `ImplicitGemmPipelined` class implements a multi-stage pipelined implicit GEMM:

```cpp
// The pipeline loads tiles of the activation and filter matrices
// into shared memory, then performs warp-level MMA operations

// Key stages:
// 1. Compute GEMM coordinates from thread/block IDs
// 2. Load activation tile from global memory (using implicit indexing)
// 3. Load filter tile from global memory
// 4. Store to shared memory
// 5. Load from shared memory to registers (warp-level)
// 6. MMA accumulate
// 7. Repeat for K dimension
```

### 12.8.3 Threadblock-level Index Computation

Each threadblock is assigned a tile of the output GEMM. The threadblock computes which elements of the activation and filter tensors it needs:

```cpp
// Threadblock tile position
int gemm_m = blockIdx.x;  // Output spatial + batch dimension
int gemm_n = blockIdx.y;  // Output channel dimension

// The K dimension is iterated within the threadblock mainloop
// Each iteration loads a tile of activations and filters

// Activation tile: rows [gemm_m * TM : (gemm_m+1) * TM]
//                  cols [k_iter * TK : (k_iter+1) * TK]
// But columns are in "implicit GEMM" space, so the actual
// activation indices are computed from the K index:
//   k -> (r, s, c) -> (h, w, c) using the convolution parameters
```

---

## 12.9 Gather/Scatter Convolution (Ampere+)

Starting with the Ampere architecture (SM80+), CUTLASS supports gather/scatter convolution operations that enable sparse or irregular access patterns.

### 12.9.1 Gather Convolution

Gather convolution allows non-contiguous access to the activation tensor, indexed by a gather map:

```cpp
// Gather convolution enables:
// - Sparse convolution where only a subset of spatial positions are computed
// - Point cloud convolution with irregular neighborhoods
// - Masked convolution for variable-length sequences

// The gather map specifies which input positions to read:
// gather_map[m] = index into the NHWC activation tensor
// Instead of computing h and w from the implicit GEMM index,
// the gather map directly specifies the activation position.
```

### 12.9.2 Scatter Convolution

Scatter convolution allows writing output to non-contiguous positions:

```cpp
// Scatter convolution enables:
// - Writing output to non-contiguous memory locations
// - Reduction-based output for split-k
// - Gradient accumulation in specific positions
```

### 12.9.3 Usage Pattern

```cpp
// Gather/scatter convolution is configured through the arguments:
typename ConvOp::Arguments args(
    problem_size,
    {ptr_A, stride_A, ptr_gather_A},   // Include gather index for A
    {ptr_B, stride_B},                  // No gather for B (filter)
    {ptr_C, stride_C},
    {ptr_D, stride_D},
    {alpha, beta},
    split_k_slices
);

// ptr_gather_A: pointer to int32 array of gather indices
// gather_A[m] = linear index into the activation tensor for GEMM row m
```

---

## 12.10 Convolution Tensor Layouts

CUTLASS supports multiple tensor layouts for convolution operations:

### 12.10.1 TensorNHWC

The standard NHWC layout is the default for most convolution operations:

```cpp
using LayoutNHWC = cutlass::layout::TensorNHWC;
// Storage order: N -> H -> W -> C
// C is the contiguous dimension
// Stride: {H*W*C, W*C, C, 1}
```

### 12.10.2 TensorNCxHWx

The `TensorNCxHWx` layout interleaves channels for better vectorization and memory access patterns:

```cpp
// Interleaved channel layout: C elements are stored in groups of Cx
// For Cx = 4: [c0,c1,c2,c3, c4,c5,c6,c7, ...] interleaved with spatial positions
using LayoutNCxHWx = cutlass::layout::TensorNCxHWx<4>;

// This layout improves memory coalescing when the number of channels
// is small or when accessing channels in groups
// Common for Winograd-based implementations and quantized convolution
```

### 12.10.3 TensorNDHWC (3D)

For 3D convolutions:

```cpp
using LayoutNDHWC = cutlass::layout::TensorNDHWC;
// Storage order: N -> D -> H -> W -> C
// C is the contiguous dimension
// Stride: {D*H*W*C, H*W*C, W*C, C, 1}
```

### 12.10.4 Filter Layouts

Filter tensors use different layouts depending on the convolution type:

| Conv Type | Filter Layout | Shape |
|---|---|---|
| fprop | TensorNHWC / KRSC | (K, R, S, C) |
| dgrad | TensorNHWC / KRSC | (K, R, S, C) |
| wgrad | TensorNHWC / CSRK | (C, R, S, K) |

```cpp
// For fprop: filter is (K, R, S, C) = KRSC layout
using FilterLayoutFprop = cutlass::layout::TensorNHWC;

// For wgrad: filter gradient is (C, R, S, K) = CSRK layout
using FilterLayoutWgrad = cutlass::layout::TensorNHWC;
// Same layout type, but the logical dimensions are permuted
```

---

## 12.11 Epilogue Integration for Convolution

The convolution epilogue applies post-GEMM operations such as bias addition, activation functions, and type conversion. CUTLASS leverages the same epilogue framework used for GEMM.

### 12.11.1 Standard Epilogue

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                    // Output element type
    8,                           // Elements per access
    ElementAccumulator,          // Accumulator type
    ElementCompute,              // Compute type (for alpha, beta)
    cutlass::epilogue::thread::ScaleType::Default
>;

// The epilogue computes: D = alpha * Conv(A, B) + beta * C
```

### 12.11.2 Fused Bias and Activation

```cpp
// Fused bias + ReLU epilogue
using BiasReLU = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    8,
    ElementAccumulator,
    ElementCompute
>;

// Fused bias + leaky ReLU
using BiasLeakyReLU = cutlass::epilogue::thread::LinearCombinationLeakyRelu<
    ElementC,
    8,
    ElementAccumulator,
    ElementCompute
>;

// Element-wise operations in epilogue:
// D[n, p, q, k] = activation(alpha * sum(A * B) + beta * C + bias[k])
```

### 12.11.3 Convolution-Specific Epilogue Considerations

Convolution epilogues must account for the NHWC output layout:

```cpp
// The output iterator in the epilogue must write to an NHWC tensor
// with proper predication for the spatial boundaries

// For split-k convolution, the epilogue must also handle
// partial accumulation and reduction:
// D = sum_over_k_slices(partial_D[k_slice]) + bias
```

---

## 12.12 Performance Considerations

### 12.12.1 Tile Size Selection

Choosing appropriate tile sizes is critical for convolution performance:

```cpp
// General guidelines for tile size selection:

// For fprop with large spatial dimensions:
// - ThreadblockShape: 128x128x32 (M, N, K)
// - Large M (spatial*batch) and N (output channels)

// For fprop with small spatial dimensions:
// - ThreadblockShape: 64x128x64 or 128x64x64
// - Adjust for better occupancy when M is small

// For wgrad (typically large K dimension):
// - ThreadblockShape: 128x128x32
// - K = N*P*Q can be very large

// For dgrad:
// - Similar to fprop but with different output shape
```

### 12.12.2 Stage Count

The number of pipeline stages (double/triple buffering) affects the overlap between computation and memory access:

```cpp
// More stages = more shared memory = better overlap
// But shared memory is limited, so more stages may reduce occupancy

// For SM80 (Ampere) with 128x128x32 tile:
// - 3 stages: 3 * 128*32 * 2 * sizeof(half) = 48KB for A alone
// - Shared memory per SM: 164KB max
// - Balance between stages and occupancy
```

### 12.12.3 Split-K Strategy

Split-K can improve performance when the reduction dimension (RSC for fprop) is large:

```cpp
// When to use split-K:
// - K dimension (R*S*C) is large relative to M and N
// - Limited parallelism from M and N dimensions
// - Trade-off: more threadblocks but requires reduction

// Optimal split-k slices:
// - Typically 2-8 slices
// - Each slice should have enough work per threadblock
// - The reduction overhead should be amortized
```

### 12.12.4 Memory Access Optimization

```cpp
// Key optimizations for convolution memory access:
// 1. Vectorized loads: Use 128-bit loads (8 half_t values) for coalesced access
// 2. Swizzling: Avoid shared memory bank conflicts in the tile layout
// 3. Padding: Add padding to shared memory tiles to prevent bank conflicts
// 4. Cache management: Use .cg (cache-global) hint for streaming loads
// 5. Prefetching: Load the next tile while computing the current tile (pipelining)
```

### 12.12.5 Convolution-Specific Profiling

```bash
# Profile convolution using CUTLASS profiler
./tools/profiler/cutlass_profiler --kernels=conv2d*fprop* --operation=Conv2d

# Profile specific configuration
./tools/profiler/cutlass_profiler \
    --kernels=cutlass_tensorop*s8*conv2d*fprop* \
    --n=32 --h=28 --w=28 --c=64 --k=128 --r=3 --s=3 \
    --pad_h=1 --pad_w=1 --stride_h=1 --stride_w=1

# Compare analytic vs optimized
./tools/profiler/cutlass_profiler --kernels=*conv*fprop*analytic*
./tools/profiler/cutlass_profiler --kernels=*conv*fprop*optimized*
```

---

## 12.13 Complete Code Example: 2D Forward Convolution

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/threadblock/implicit_gemm_multistage.h"

// Define the convolution kernel type
using ConvKernel = typename cutlass::conv::kernel::ImplicitGemmConvolution<
    cutlass::conv::threadblock::ImplicitGemmMultistage<
        cutlass::gemm::GemmShape<128, 128, 64>,     // Threadblock shape
        cutlass::gemm::GemmShape<64, 64, 64>,       // Warp shape
        cutlass::gemm::GemmShape<16, 8, 16>,        // Instruction shape (FP16 Tensor Core)
        float,                                        // ElementA (activations, stored as FP32)
        cutlass::layout::TensorNHWC,                 // LayoutA
        cutlass::half_t,                             // ElementB (filters)
        cutlass::layout::TensorNHWC,                 // LayoutB
        float,                                        // ElementC
        cutlass::layout::TensorNHWC,                 // LayoutC
        float,                                        // ElementAccumulator
        cutlass::arch::OpClassTensorOp,              // OpClass
        cutlass::arch::Sm80,                         // Architecture
        cutlass::conv::Operator::kFprop,             // Convolution type
        3,                                           // Pipeline stages
        cutlass::gemm::SharedMemoryClearOption::kNone
    >
>;

// Wrap in device-level operation
using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<ConvKernel>;

// Launch function
cutlass::Status run_conv2d_fprop(
    int N, int H, int W, int C,    // Input dimensions
    int K, int R, int S,            // Filter dimensions
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cutlass::half_t *ptr_activations,
    cutlass::half_t *ptr_filters,
    float *ptr_output,
    cudaStream_t stream = 0)
{
    // Create problem size
    cutlass::conv::Conv2dProblemSize problem_size(
        N, H, W, C,
        K, R, S,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        cutlass::conv::Mode::kCrossCorrelation,
        1,  // split_k_slices
        1   // groups
    );

    // Compute strides
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC>
        ref_A(ptr_activations, cutlass::layout::TensorNHWC::packed({N, H, W, C}));
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC>
        ref_B(ptr_filters, cutlass::layout::TensorNHWC::packed({K, R, S, C}));
    cutlass::TensorRef<float, cutlass::layout::TensorNHWC>
        ref_C(ptr_output, cutlass::layout::TensorNHWC::packed(
            {N, problem_size.P, problem_size.Q, K}));

    // Arguments
    typename Conv2dFprop::Arguments args(
        problem_size,
        {ptr_activations, ref_A.stride(0)},
        {ptr_filters, ref_B.stride(0)},
        {ptr_output, ref_C.stride(0)},     // C (source for beta term)
        {ptr_output, ref_C.stride(0)},     // D (destination)
        {1.0f, 0.0f},                       // alpha, beta
        1                                    // split_k_slices
    );

    Conv2dFprop conv_op;
    size_t workspace_size = conv_op.get_workspace_size(args);

    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    cutlass::Status status = conv_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = conv_op.run(stream);

    if (workspace) {
        cudaFree(workspace);
    }

    return status;
}
```

---

## 12.14 Complete Code Example: 3D Forward Convolution

```cpp
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"

using Conv3dKernel = typename cutlass::conv::kernel::ImplicitGemmConvolution<
    cutlass::conv::threadblock::ImplicitGemmMultistage<
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::half_t,
        cutlass::layout::TensorNDHWC,
        cutlass::half_t,
        cutlass::layout::TensorNDHWC,
        float,
        cutlass::layout::TensorNDHWC,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::conv::Operator::kFprop,
        3
    >
>;

using Conv3dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv3dKernel>;

// Usage:
// cutlass::conv::Conv3dProblemSize problem_3d(
//     N, D, H, W, C, K, T, R, S,
//     pad_d, pad_h, pad_w,
//     stride_d, stride_h, stride_w,
//     dilation_d, dilation_h, dilation_w,
//     cutlass::conv::Mode::kCrossCorrelation, 1, 1
// );
// typename Conv3dFprop::Arguments args(problem_3d, ...);
// Conv3dFprop conv3d;
// conv3d(args);
```

---

## 12.15 Key Header Files Reference

| Header | Purpose |
|---|---|
| `cutlass/conv/convolution.h` | Core convolution types, Conv2dProblemSize, Conv3dProblemSize |
| `cutlass/conv/device/implicit_gemm_convolution.h` | Device-level implicit GEMM convolution |
| `cutlass/conv/kernel/implicit_gemm_convolution.h` | Kernel-level implicit GEMM convolution |
| `cutlass/conv/threadblock/implicit_gemm_pipelined.h` | Pipelined implicit GEMM threadblock |
| `cutlass/conv/threadblock/implicit_gemm_multistage.h` | Multi-stage implicit GEMM threadblock |
| `cutlass/conv/threadblock/conv2d_tile_iterator.h` | Tile iterator for 2D convolution |
| `cutlass/conv/threadblock/conv3d_tile_iterator.h` | Tile iterator for 3D convolution |
| `cutlass/conv/threadblock/conv2d_predicated_tile_iterator.h` | Predicated tile iterator for conv2d |
| `cutlass/conv/threadblock/predicated_conv2d_tile_access_iterator.h` | Predicated access iterator |
| `cutlass/conv/warpconv2d/mma_complex.h` | Warp-level MMA for convolution |
| `cutlass/layout/tensor.h` | TensorNHWC layout definition |
| `cutlass/layout/tensor_interleaved.h` | TensorNCxHWx interleaved layout |

---

## 12.16 Summary

CUTLASS implements convolution through the implicit GEMM approach, which reformulates convolution as a matrix multiplication without materializing the expanded activation matrix. Key aspects include:

1. **Four convolution types**: fprop (forward), dgrad (data gradient), wgrad (weight gradient), and deconv (deconvolution), each mapping to different GEMM formulations.
2. **Problem size structs**: `Conv2dProblemSize` and `Conv3dProblemSize` encapsulate all parameters including padding, stride, dilation, split-k, and groups.
3. **Multiple algorithms**: Analytic (general-purpose), optimized (fast paths), fixed-channels, and few-channels variants.
4. **Device-level API**: The `ImplicitGemmConvolution` device class provides a simple interface for launching convolution kernels.
5. **Gather/scatter support**: Ampere+ architectures support indexed access patterns for sparse and irregular convolutions.
6. **Epilogue integration**: Convolution epilogues support fused bias, activation, and type conversion operations.
7. **Performance tuning**: Tile size selection, stage count, split-k strategy, and memory access patterns are key performance knobs.
