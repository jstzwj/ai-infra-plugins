# CUTLASS Reference - Chapter 26: Epilogue Fusion Patterns

This reference covers epilogue fusion patterns in CUTLASS, including back-to-back GEMM fusion, activation fusion, bias addition, custom epilogue operators, and the epilogue visitor pattern. Epilogue fusion is a powerful technique for reducing memory traffic by fusing post-GEMM operations directly into the GEMM kernel.

---

## 26.1 Epilogue Fusion Concept

### 26.1.1 What is the Epilogue?

In CUTLASS, the **epilogue** is the phase after the main GEMM loop that processes the accumulated result. The standard epilogue computes:

```
D = alpha * A * B + beta * C
```

The epilogue performs:
1. **Scale**: Multiply the accumulated result by `alpha`.
2. **Bias addition**: Add `beta * C`.
3. **Type conversion**: Convert from accumulator type (e.g., FP32) to output type (e.g., FP16).
4. **Store**: Write the result to global memory.

### 26.1.2 Why Fuse the Epilogue?

Without fusion, a typical pipeline involves:
```
GEMM kernel:    Compute D = alpha * A * B + beta * C   --> write D to global memory
Bias kernel:    Compute D = D + bias                    --> read D, write D
ReLU kernel:    Compute D = max(D, 0)                   --> read D, write D
```

Each intermediate kernel reads and writes the full output tensor to global memory. For a 4096x4096 FP16 matrix, each read/write is 32 MB. With 3 kernels, that is 6 round trips (192 MB of global memory traffic).

With epilogue fusion:
```
Fused GEMM kernel:  Compute D = ReLU(alpha * A * B + beta * C + bias)
                    --> write D once to global memory
```

The fused kernel performs all operations in registers or shared memory before the final store, reducing global memory traffic to a single write (32 MB).

### 26.1.3 Fusion Opportunities

CUTLASS supports fusing the following operations into the epilogue:

| Operation | Description | Memory Saved |
|---|---|---|
| Linear combination | `alpha * acc + beta * C` | Baseline (always done) |
| Bias addition | `+ bias[row]` or `+ bias[col]` | 1 read + 1 write |
| ReLU | `max(x, 0)` | 1 read + 1 write |
| GELU | Gaussian Error Linear Unit | 1 read + 1 write |
| SiLU / Swish | `x * sigmoid(x)` | 1 read + 1 write |
| HardSwish | `x * clip(x/6 + 0.5, 0, 1)` | 1 read + 1 write |
| Type conversion | Accumulator to output type | Baseline |
| Scale per row/col | Per-row or per-column scaling | 1 read + 1 write |
| Reduction | Sum or max reduction of the output | 1 read + 1 write |
| Back-to-back GEMM | Output feeds directly into next GEMM | 1 read + 1 write |

---

## 26.2 Back-to-Back GEMM Fusion

### 26.2.1 Concept

Back-to-back (B2B) GEMM fusion chains two GEMM operations together so that the output of the first GEMM feeds directly into the second GEMM without going through global memory:

```
Standard (unfused):
  GEMM 0: D0 = relu(alpha0 * A0 * B0)         --> write D0 to global memory
  GEMM 1: D1 = relu(alpha1 * D0 * B1 + beta1 * C1)  --> read D0 from global memory

Fused (B2B):
  B2B GEMM: D1 = relu(alpha1 * relu(alpha0 * A0 * B0) * B1 + beta1 * C1)
            D0 stays in register file / shared memory between GEMM 0 and GEMM 1
            Only D1 is written to global memory
```

### 26.2.2 Register File Residency Optimization

The key optimization in B2B GEMM is keeping the intermediate result `D0` in the register file. On Hopper (SM90), the register file can hold up to 256 KB per SM, which is sufficient for moderate-sized output tiles:

- A 64x64 FP16 tile = 64 * 64 * 2 = 8 KB (fits easily in registers).
- A 128x128 FP16 tile = 128 * 128 * 2 = 32 KB (still fits).
- A 256x128 FP16 tile = 256 * 128 * 2 = 64 KB (challenging but possible with warp specialization).

The register file residency eliminates the round-trip to global memory for the intermediate result.

### 26.2.3 Constraints for Full Fusion

Full B2B GEMM fusion has several constraints:

1. **Tile size compatibility**: The output tile of GEMM 0 must match the input tile of GEMM 1. Specifically, the M dimension of GEMM 0's output tile must equal the M dimension of GEMM 1's input tile.

2. **Thread block mapping**: Both GEMMs must use the same grid mapping so that the same thread block processes the corresponding tiles in both GEMMs.

3. **Memory budget**: The intermediate tile must fit in registers plus shared memory. If the tile is too large, it may spill to local memory (which is cached in L1 but still slower than registers).

4. **Accumulator type**: The intermediate type (after activation) should be the same as the input type for GEMM 1.

5. **Activation function**: The intermediate activation must be applied before the result is used as input to GEMM 1. Only element-wise activations are supported (ReLU, GELU, SiLU, etc.).

### 26.2.4 B2B GEMM API

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

// Back-to-back GEMM configuration
// GEMM 0: D0 = relu(alpha0 * A0 * B0)
// GEMM 1: D1 = relu(alpha1 * D0 * B1 + beta1 * C1)

using ElementA0 = cutlass::half_t;
using ElementB0 = cutlass::half_t;
using ElementC0 = cutlass::half_t;  // Not used (beta = 0)
using ElementD0 = cutlass::half_t;  // Intermediate, stays in registers

using ElementA1 = ElementD0;         // GEMM 1 input = GEMM 0 output
using ElementB1 = cutlass::half_t;
using ElementC1 = cutlass::half_t;
using ElementD1 = cutlass::half_t;   // Final output

using ElementAccumulator = float;

// GEMM 0 configuration
using Gemm0Shape = cutlass::gemm::GemmShape<128, 128, 64>;
using Gemm1Shape = cutlass::gemm::GemmShape<128, 128, 64>;

// The B2B GEMM kernel combines both GEMMs
using B2BGemmKernel = cutlass::gemm::kernel::B2bGemm<
    Gemm0Shape, Gemm1Shape,
    ElementA0, cutlass::layout::RowMajor,
    ElementB0, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    ElementA1,
    ElementB1, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,    // Warp shape
    cutlass::gemm::GemmShape<16, 8, 16>,     // Instruction shape
    cutlass::epilogue::thread::LinearCombinationRelu<
        ElementD0, 4, ElementAccumulator, ElementAccumulator>,  // GEMM 0 epilogue: ReLU
    cutlass::epilogue::thread::LinearCombinationRelu<
        ElementD1, 4, ElementAccumulator, ElementAccumulator>,  // GEMM 1 epilogue: ReLU
    4  // Stages
>;

using B2BGemm = cutlass::gemm::device::GemmUniversalAdapter<B2BGemmKernel>;
```

### 26.2.5 Launching B2B GEMM

```cpp
void run_b2b_gemm(
    int M, int N0, int K0, int N1, int K1,
    const cutlass::half_t* A0, int64_t lda0,
    const cutlass::half_t* B0, int64_t ldb0,
    const cutlass::half_t* B1, int64_t ldb1,
    const cutlass::half_t* C1, int64_t ldc1,
    cutlass::half_t* D1, int64_t ldd1,
    float alpha0, float alpha1, float beta1,
    cudaStream_t stream = 0
) {
    B2BGemm b2b_gemm;

    typename B2BGemm::Arguments args{
        {M, N0, K0, N1, K1},  // Problem dimensions for both GEMMs
        {A0, lda0},
        {B0, ldb0},
        {B1, ldb1},
        {C1, ldc1},
        {D1, ldd1},
        {alpha0, alpha1, beta1}
    };

    size_t workspace_size = b2b_gemm.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = b2b_gemm.initialize(args, workspace.get(), stream);
    if (status == cutlass::Status::kSuccess) {
        status = b2b_gemm(stream);
    }
}
```

---

## 26.3 Activation Fusion

### 26.3.1 Supported Activation Functions

CUTLASS provides built-in epilogue operators that fuse activation functions directly into the GEMM output:

| Activation | Formula | CUTLASS Class |
|---|---|---|
| ReLU | `max(x, 0)` | `LinearCombinationRelu` |
| Leaky ReLU | `max(x, alpha * x)` | `LinearCombinationLeakyRelu` |
| GELU (tanh) | `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` | `LinearCombinationGelu` |
| GELU (exact) | `x * Phi(x)` where Phi is the standard normal CDF | Custom implementation |
| SiLU / Swish | `x * sigmoid(x) = x / (1 + exp(-x))` | `LinearCombinationSilu` |
| HardSwish | `x * clip(x/6 + 0.5, 0, 1)` | Custom implementation |
| Tanh | `tanh(x)` | Custom implementation |
| Sigmoid | `1 / (1 + exp(-x))` | Custom implementation |

### 26.3.2 ReLU Fusion

The most common activation fusion is ReLU:

```cpp
#include "cutlass/epilogue/thread/linear_combination_relu.h"

// CUTLASS 2.x: ReLU fused epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per access
    float,                 // ElementAccumulator
    float,                 // ElementCompute
    cutlass::epilogue::thread::ReLu<float>  // Activation function
>;

using GemmWithRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp
>;
```

For CUTLASS 3.x, activation fusion is specified through the epilogue collective:

```cpp
// CUTLASS 3.x: ReLU fused epilogue
#include "cutlass/epilogue/collective/epilogue_fusion.hpp"

using EpilogueOp = cutlass::epilogue::collective::Epilogue<
    cutlass::layout::RowMajor,     // LayoutD
    cutlass::layout::RowMajor,     // LayoutC
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::epilogue::thread::ReLu<float>  // Activation
>;
```

### 26.3.3 GELU Fusion

GELU (Gaussian Error Linear Unit) is widely used in transformer models:

```cpp
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

// GELU fused epilogue
using EpilogueGelu = cutlass::epilogue::thread::LinearCombinationGelu<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per access
    float,                 // ElementAccumulator
    float                  // ElementCompute
>;
```

### 26.3.4 SiLU Fusion

SiLU (Sigmoid Linear Unit) is used in models like LLaMA:

```cpp
#include "cutlass/epilogue/thread/linear_combination_silu.h"

// SiLU fused epilogue (if available in your CUTLASS version)
// Otherwise, implement a custom activation:
struct Silu {
    float operator()(float x) const {
        return x / (1.0f + expf(-x));  // x * sigmoid(x)
    }
};

using EpilogueSilu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float,
    cutlass::epilogue::thread::SiLu<float>
>;
```

---

## 26.4 Bias Addition Fusion

### 26.4.1 Row Bias (Broadcast Along Rows)

Row bias is the most common bias pattern: a single bias vector is added to every row of the output:

```
D[i, j] = alpha * sum_k(A[i,k] * B[k,j]) + beta * C[i,j] + bias[j]
```

```cpp
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

// Bias + ReLU fused epilogue
using EpilogueBiasRelu = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per access
    float,                 // ElementAccumulator
    float,                 // ElementCompute
    cutlass::layout::RowMajor  // Bias layout (RowMajor = row broadcast)
>;

// The bias tensor is passed as an additional argument:
// args = { ..., {bias_ptr, 0} }  // stride 0 for row broadcast
```

### 26.4.2 Column Bias (Broadcast Along Columns)

Column bias adds a bias vector to every column:

```
D[i, j] = alpha * sum_k(A[i,k] * B[k,j]) + beta * C[i,j] + bias[i]
```

```cpp
// Column bias fused epilogue
using EpilogueColBias = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    4,
    float,
    float,
    cutlass::layout::ColumnMajor  // ColumnMajor = column broadcast
>;
```

### 26.4.3 Per-Channel Bias (for Quantized GEMM)

For INT8 quantized GEMM with per-channel scales and biases:

```cpp
// Per-channel bias for quantized inference
// D = (A_int8 * B_int8) * scale_per_channel + bias_per_channel
using EpiloguePerChannel = cutlass::epilogue::thread::LinearCombinationPerChannel<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per access
    int32_t,              // ElementAccumulator (INT8 * INT8 = INT32)
    float,                 // ElementCompute (scale and bias in FP32)
    cutlass::layout::RowMajor
>;
```

---

## 26.5 Epilogue Output Operations

### 26.5.1 LinearCombination

The basic epilogue operation computes `D = alpha * acc + beta * C`:

```cpp
#include "cutlass/epilogue/thread/linear_combination.h"

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per vectorized access
    float,                 // ElementAccumulator
    float                  // ElementCompute (for alpha, beta)
>;

// Template parameters explained:
// - ElementOutput: The type of the output tensor D
// - ElementsPerAccess: Number of elements written per instruction (affects vectorization)
// - ElementAccumulator: The type of the GEMM accumulator
// - ElementCompute: The type for alpha/beta computation
```

### 26.5.2 LinearCombination with Activation

Fusing an activation function after the linear combination:

```cpp
// LinearCombination + ReLU
using EpilogueRelu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float,
    cutlass::epilogue::thread::ReLu<float>  // 5th parameter: activation
>;

// LinearCombination + GELU
using EpilogueGelu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float,
    cutlass::epilogue::thread::Gelu<float>
>;

// LinearCombination + SiLU
using EpilogueSilu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float,
    cutlass::epilogue::thread::SiLu<float>
>;

// LinearCombination + Tanh
using EpilogueTanh = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float,
    cutlass::epilogue::thread::Tanh<float>
>;
```

### 26.5.3 LinearCombinationBiasRelu

The combined bias + ReLU epilogue is a common pattern in neural network layers:

```cpp
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

// D = ReLU(alpha * A * B + bias)
using EpilogueBiasRelu = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per access
    float,                 // ElementAccumulator
    float,                 // ElementCompute
    cutlass::layout::RowMajor  // Broadcast dimension for bias
>;

// Arguments include the bias tensor:
// args.bias_ptr = bias_data;  // Pointer to bias vector of length N
// args.bias_stride = 0;       // Stride 0 for broadcast
```

### 26.5.4 Custom Epilogue Operators

For custom activation functions not built into CUTLASS:

```cpp
// Custom activation function
struct MyActivation {
    // Scalar version
    float operator()(float x) const {
        // Example: Mish activation = x * tanh(softplus(x))
        float sp = logf(1.0f + expf(x));
        return x * tanhf(sp);
    }

    // Vectorized version (optional, for better performance)
    cutlass::Array<float, 4> operator()(cutlass::Array<float, 4> const& x) const {
        cutlass::Array<float, 4> result;
        for (int i = 0; i < 4; ++i) {
            result[i] = this->operator()(x[i]);
        }
        return result;
    }
};

// Use custom activation in the epilogue
using EpilogueCustom = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 4, float, float,
    MyActivation  // Custom activation functor
>;
```

---

## 26.6 Fusion with Reduction

### 26.6.1 Reduction Epilogue

Some workloads require reducing the GEMM output along one dimension (e.g., computing the sum or max of each row). CUTLASS supports fusing this reduction into the epilogue:

```
D = alpha * A * B + beta * C    (GEMM)
R[i] = reduce_j(D[i, j])        (Reduction along N dimension)
```

Common use cases:
- **Layer normalization**: Computing the mean and variance of each row.
- **Softmax**: Computing the max and sum of each row.
- **Loss computation**: Summing log-probabilities.

```cpp
#include "cutlass/epilogue/thread/linear_combination.h"

// The reduction is handled by a separate reduction kernel that reads
// the GEMM output. For fused reduction, use the EpilogueVisitor pattern.

// Step 1: GEMM with standard epilogue writes D
// Step 2: Separate reduction kernel computes R = reduce(D)
// For fused operation, see the Epilogue Visitor pattern below.
```

---

## 26.7 Epilogue Visitor Pattern

### 26.7.1 Concept

The Epilogue Visitor pattern is CUTLASS 3.x's composable epilogue system. It allows arbitrary chains of operations to be fused into the epilogue using a visitor pattern:

```
acc (from GEMM) --> Op1 --> Op2 --> Op3 --> store to D
```

Each operation is a "visitor" that processes the accumulator tile. Operations can include:
- Linear combination (alpha * acc + beta * C).
- Bias addition.
- Activation functions.
- Type conversion.
- Custom element-wise operations.

### 26.7.2 Visitor Composition

Visitors are composed using template parameters:

```cpp
#include "cutlass/epilogue/collective/epilogue_visitor.hpp"

// Define the visitor chain:
// 1. LinearCombination: acc * alpha + beta * C
// 2. BiasAdd: + bias
// 3. ReLU activation
// 4. Convert: FP32 -> FP16
// 5. Store to D

using VisitorChain = cutlass::epilogue::visitor::Chain<
    cutlass::epilogue::visitor::Compute<
        cutlass::epilogue::visitor::OpLambda<float>,  // alpha * acc + beta * C
        float
    >,
    cutlass::epilogue::visitor::Compute<
        cutlass::epilogue::visitor::OpBiasAdd<float>,  // + bias
        float
    >,
    cutlass::epilogue::visitor::Compute<
        cutlass::epilogue::visitor::OpReLU<float>,      // max(x, 0)
        float
    >,
    cutlass::epilogue::visitor::Convert<cutlass::half_t>, // FP32 -> FP16
    cutlass::epilogue::visitor::StoreD<
        cutlass::half_t, cutlass::layout::RowMajor
    >
>;
```

### 26.7.3 CUTLASS 3.x Epilogue Collective

In CUTLASS 3.x, the epilogue is configured through the `Epilogue` collective:

```cpp
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Standard epilogue (alpha * A * B + beta * C)
using EpilogueStandard = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor,  // LayoutD
    cutlass::layout::RowMajor,  // LayoutC
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

// Epilogue with visitor chain (custom fusion)
using EpilogueCustom = cutlass::epilogue::collective::Epilogue<
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    VisitorChain  // Custom visitor chain
>;
```

---

## 26.8 Code Generation for B2B GEMMs

### 26.8.1 CUTLASS 2.x B2B GEMM Code Generation

CUTLASS provides a code generation utility for B2B GEMMs that automatically selects the optimal tile sizes and configurations:

```cpp
// B2B GEMM with code generation
// The following example shows a fused two-layer MLP:
// Layer 1: D0 = ReLU(A0 * B0 + bias0)
// Layer 2: D1 = ReLU(D0 * B1 + bias1)

using B2BGemmKernel = cutlass::gemm::kernel::B2bGemm<
    // GEMM 0 configuration
    cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape0
    cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape0
    cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape0

    // GEMM 1 configuration
    cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape1
    cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape1
    cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape1

    // Data types
    cutlass::half_t, cutlass::layout::RowMajor,   // A0
    cutlass::half_t, cutlass::layout::ColumnMajor, // B0
    float,                                          // Accumulator0
    cutlass::half_t,                                // Intermediate type
    cutlass::half_t, cutlass::layout::ColumnMajor, // B1
    float,                                          // Accumulator1

    // Operation class and architecture
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,

    // Epilogues
    cutlass::epilogue::thread::LinearCombinationRelu<
        cutlass::half_t, 4, float, float>,          // GEMM 0: ReLU
    cutlass::epilogue::thread::LinearCombinationRelu<
        cutlass::half_t, 4, float, float>,          // GEMM 1: ReLU
    4  // Stages
>;
```

### 26.8.2 B2B GEMM Problem Dimensions

B2B GEMMs require specifying the problem dimensions for both GEMMs:

```cpp
// Problem dimensions:
// GEMM 0: D0 (M x N0) = A0 (M x K0) * B0 (K0 x N0)
// GEMM 1: D1 (M x N1) = D0 (M x N0) * B1 (N0 x N1)
// Note: K1 = N0 (the intermediate dimension must match)

struct B2BProblemSize {
    int M;
    int N0;    // Also K1 (intermediate dimension)
    int K0;    // K dimension for first GEMM
    int N1;    // N dimension for second GEMM
};

// Initialize B2B GEMM arguments
B2BProblemSize problem{1024, 512, 256, 1024};
// GEMM 0: 1024 x 512 = 1024 x 256 * 256 x 512
// GEMM 1: 1024 x 1024 = 1024 x 512 * 512 x 1024
```

### 26.8.3 Intermediate Tile Size Constraints

For B2B GEMM fusion, the intermediate tile (from GEMM 0 to GEMM 1) must satisfy:

1. **M dimension**: The M tile size must be the same for both GEMMs (same number of rows processed by the same thread block).
2. **N/K dimension**: The N tile of GEMM 0 must fit in registers/shared memory as the K tile of GEMM 1.

```cpp
// Valid B2B configuration:
// GEMM 0 threadblock tile: (128, 128, 64)
// GEMM 1 threadblock tile: (128, 128, 128)
// Intermediate tile: 128 x 128 = 32KB (FP16), fits in registers

// Invalid (too large):
// GEMM 0 threadblock tile: (256, 256, 64)
// GEMM 1 threadblock tile: (256, 256, 256)
// Intermediate tile: 256 x 256 = 128KB (FP16), exceeds register budget
```

---

## 26.9 Complete Epilogue Fusion Examples

### 26.9.1 GEMM + Bias + ReLU Fusion

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

// GEMM with bias + ReLU fusion
using GemmBiasRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombinationBiasRelu<
        cutlass::half_t, 4, float, float,
        cutlass::layout::RowMajor
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;

void run_gemm_bias_relu(
    int M, int N, int K,
    const cutlass::half_t* A, int lda,
    const cutlass::half_t* B, int ldb,
    const cutlass::half_t* C, int ldc,
    cutlass::half_t* D, int ldd,
    const cutlass::half_t* bias,  // Bias vector of length N
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    GemmBiasRelu gemm_op;

    typename GemmBiasRelu::Arguments args(
        {M, N, K},
        alpha, beta,
        A, lda,
        B, ldb,
        C, ldc,
        D, ldd,
        bias  // Additional bias argument
    );

    auto status = gemm_op.initialize(args);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op(stream);
    }
}
```

### 26.9.2 SM90 GEMM with Epilogue Fusion (3.x API)

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// SM90 GEMM with fused ReLU epilogue using CUTLASS 3.x
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 8,
    ElementB, cutlass::layout::ColumnMajor, 8,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Epilogue with ReLU activation
using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::epilogue::thread::ReLu<float>  // Fused ReLU
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

void run_sm90_gemm_relu(
    int M, int N, int K,
    const cutlass::half_t* A, int64_t lda,
    const cutlass::half_t* B, int64_t ldb,
    const cutlass::half_t* C, int64_t ldc,
    cutlass::half_t* D, int64_t ldd,
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    Gemm gemm_op;

    typename Gemm::Arguments args{
        {M, N, K},
        {A, lda},
        {B, ldb},
        {C, ldc},
        {D, ldd},
        {alpha, beta}
    };

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm_op.initialize(args, workspace.get(), stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op(stream);
    }
}
```

### 26.9.3 GEMM + Bias + GELU Fusion

```cpp
// Custom epilogue: GEMM + bias + GELU
// Using a custom visitor chain in CUTLASS 3.x

#include "cutlass/epilogue/thread/linear_combination.h"

// Define GELU activation
struct GeluFusion {
    static float gelu(float x) {
        // Tanh approximation of GELU
        return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }

    float operator()(float x) const {
        return gelu(x);
    }
};

using EpilogueGelu = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,      // ElementOutput
    4,                     // Elements per access
    float,                 // ElementAccumulator
    float,                 // ElementCompute
    GeluFusion             // Custom activation
>;

using GemmGelu = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueGelu,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4
>;
```

### 26.9.4 Full B2B GEMM Example (MLP Forward)

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/b2b_gemm.h"

// Two-layer MLP: D1 = ReLU(ReLU(A0 * B0) * B1)
// GEMM 0: Intermediate = ReLU(A0 * B0)  -- stays in registers
// GEMM 1: D1 = ReLU(Intermediate * B1)  -- only final output goes to memory

void run_mlp_forward_b2b(
    int M, int K0, int N0, int N1,
    const cutlass::half_t* A0, int lda0,
    const cutlass::half_t* B0, int ldb0,
    const cutlass::half_t* B1, int ldb1,
    const cutlass::half_t* C1, int ldc1,
    cutlass::half_t* D1, int ldd1,
    float alpha, float beta,
    cudaStream_t stream = 0
) {
    // GEMM 0: M x N0 = M x K0 * K0 x N0
    // GEMM 1: M x N1 = M x N0 * N0 x N1

    using B2BGemm = /* B2B GEMM type as defined in 26.8.1 */;
    B2BGemm b2b_gemm;

    typename B2BGemm::Arguments args(
        {M, N0, K0, N1, N0},  // M, N0, K0, N1, K1=N0
        alpha, alpha, beta,
        A0, lda0,
        B0, ldb0,
        B1, ldb1,
        C1, ldc1,
        D1, ldd1
    );

    auto status = b2b_gemm.initialize(args);
    if (status == cutlass::Status::kSuccess) {
        status = b2b_gemm(stream);
    }
}
```

---

## 26.10 Performance Impact of Epilogue Fusion

### 26.10.1 Memory Traffic Reduction

For a 4096 x 4096 FP16 matrix:

| Scenario | Global Memory Traffic |
|---|---|
| Separate GEMM + ReLU kernels | 3 reads + 2 writes = 160 MB |
| Fused GEMM + ReLU | 2 reads + 1 write = 96 MB |
| Fused GEMM + Bias + ReLU | 2 reads + 1 read (bias) + 1 write = 100 MB |
| B2B GEMM (saves intermediate) | 3 reads (A0, B0, B1) + 1 write = 80 MB |
| Unfused B2B (two separate GEMMs) | 5 reads + 3 writes = 256 MB |

### 26.10.2 Register Pressure Considerations

Fusing more operations increases register pressure:
- **Standard epilogue**: ~8-16 registers for the output tile.
- **Bias + activation**: +2-4 registers for the bias values.
- **B2B GEMM**: The intermediate tile requires 16-64 registers (depends on tile size).

If register pressure exceeds the budget, the compiler spills to local memory, which can negate the fusion benefit. CUTLASS's tile sizes are designed to avoid this, but very large tiles in B2B configurations may need tuning.

### 26.10.3 Best Practices

1. **Profile before and after fusion**: Verify that the fusion actually improves performance by measuring end-to-end time.
2. **Match activation to workload**: ReLU is nearly free. GELU and SiLU have higher compute cost due to transcendental functions.
3. **B2B for small intermediates**: B2B fusion is most beneficial when the intermediate dimension (N0/K1) is not too large relative to the register file.
4. **Use CollectiveBuilder**: Let CUTLASS choose the optimal configuration for your fusion pattern.
5. **Watch for type conversion overhead**: Converting between FP32 accumulator and FP16 output in the epilogue is usually well-optimized, but adding more conversions (e.g., to FP8) adds latency.

---

## 26.11 Summary

Epilogue fusion is a critical optimization technique in CUTLASS:

1. **Activation fusion**: ReLU, GELU, SiLU, and custom activations can be fused directly into the GEMM epilogue.
2. **Bias fusion**: Per-row or per-column bias addition is fused into the epilogue, avoiding an extra kernel launch and memory round-trip.
3. **Back-to-back GEMM**: Two chained GEMMs can be fused so the intermediate result stays in registers, cutting memory traffic in half.
4. **Epilogue visitor pattern**: CUTLASS 3.x provides a composable visitor chain for arbitrary epilogue fusion patterns.
5. **Memory savings**: Fusion eliminates intermediate reads/writes to global memory, reducing traffic by 30-70%.
6. **Register pressure**: More fusion increases register pressure; CUTLASS's default tile sizes are designed to stay within budget.
7. **CUTLASS 2.x vs 3.x**: Both versions support epilogue fusion, but 3.x provides a more composable API with the visitor pattern.
