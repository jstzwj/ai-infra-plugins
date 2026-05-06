# Epilogue in CUTLASS

The epilogue is the post-GEMM processing stage that transforms raw accumulator values into the final output. It handles scaling the accumulated result, adding a residual (source) matrix, applying activation functions, performing type conversion, and writing the output to global memory. Understanding the epilogue is essential for fusing post-GEMM operations and achieving optimal performance.

---

## Epilogue Concept

### What the Epilogue Does

After the mainloop completes the matrix multiply-accumulate (AB), the epilogue performs the following steps:

1. **Scale the accumulators**: multiply by `alpha`
2. **Add the source matrix**: if `beta != 0`, add `beta * C` to the scaled result
3. **Apply activation functions**: optional element-wise transformations (ReLU, sigmoid, etc.)
4. **Type conversion**: convert from accumulator type to output type (e.g., float to half_t)
5. **Store output**: write results to global memory at the destination pointer

### Mathematical Formulation

```
D = activation(alpha * A * B + beta * C + bias)
```

Where:
- `A * B` is the accumulated matrix product from the mainloop
- `alpha` and `beta` are scalar scaling factors
- `C` is the optional source matrix (for residual connections)
- `bias` is an optional per-row or per-column bias
- `activation` is an optional element-wise function (identity, ReLU, etc.)

### Position in the GEMM Pipeline

```
Mainloop:   accum[i][j] += A[i][k] * B[k][j]   (for all k)
                |
Epilogue:   D[i][j] = f(alpha * accum[i][j] + beta * C[i][j])
                |
Output:     Write D to global memory
```

---

## Epilogue Operators (2.x Thread-Level)

Epilogue operators in CUTLASS 2.x are defined in the `cutlass::epilogue::thread` namespace. Each operator defines how a single thread processes its portion of the accumulator tile.

### LinearCombination

The most basic epilogue: `D = alpha * AB + beta * C`

```cpp
#include "cutlass/epilogue/thread/linear_combination.h"

template <
  typename ElementOutput_,        // Output element type (e.g., half_t, float)
  int Count,                      // Number of elements per vectorized access
  typename ElementAccumulator_,   // Accumulator type (e.g., float)
  typename ElementCompute_,       // Compute type for alpha/beta operations
  cutlass::epilogue::thread::ScaleType::Kind Scale =
      cutlass::epilogue::thread::ScaleType::Default
>
class LinearCombination;
```

#### Template Parameters

```cpp
// ElementOutput_: the data type of the output matrix D
using ElementOutput = cutlass::half_t;  // or float, int8_t, etc.

// Count: number of elements processed per memory access
// Should be 128 / sizeof_bits<ElementOutput>::value for optimal vectorization
// For FP16: 128 / 16 = 8 elements
// For FP32: 128 / 32 = 4 elements
int Count = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// ElementAccumulator_: the type of the MMA accumulator
// Typically float for FP16/BF16 inputs, int32_t for INT8 inputs
using ElementAccumulator = float;

// ElementCompute_: the type used for computing alpha*AB + beta*C
// Usually the same as ElementAccumulator, but can differ for mixed precision
using ElementCompute = float;
```

#### Usage Example

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,                                   // ElementOutput
    8,                                                  // Count (128-bit access)
    float,                                              // ElementAccumulator
    float                                               // ElementCompute
>;

// The Params structure:
// EpilogueOp::Params params(alpha, beta);
//   alpha: scaling factor for AB product
//   beta: scaling factor for C (source matrix)
//
// When beta = 0, the C matrix is not loaded (saves memory bandwidth)
```

### LinearCombinationRelu

Adds ReLU activation: `D = max(0, alpha * AB + beta * C)`

```cpp
#include "cutlass/epilogue/thread/linear_combination_relu.h"

template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_,
  typename ElementCompute_,
  cutlass::epilogue::thread::ScaleType::Kind Scale =
      cutlass::epilogue::thread::ScaleType::Default
>
class LinearCombinationRelu;

// Usage:
using EpilogueRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
    8,
    float,
    float
>;

// Params:
// EpilogueRelu::Params params(alpha, beta);
// Output: max(0, alpha * AB + beta * C)
```

### LinearCombinationBiasRelu

Adds bias vector and ReLU: `D = max(0, alpha * AB + beta * C + bias)`

```cpp
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementBias_ = ElementCompute_,
  cutlass::epilogue::thread::ScaleType::Kind Scale =
      cutlass::epilogue::thread::ScaleType::Default
>
class LinearCombinationBiasRelu;

// Usage:
using EpilogueBiasRelu = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    8,
    float,
    float,
    float           // ElementBias type
>;

// Params:
// EpilogueBiasRelu::Params params(alpha, beta, bias_ptr);
//   bias_ptr: pointer to bias vector in device memory
//   Bias is broadcast along the M dimension (one bias per column)
// Output: max(0, alpha * AB + beta * C + bias[j])
```

### LinearCombinationBias

Adds bias without activation: `D = alpha * AB + beta * C + bias`

```cpp
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"

// Generic bias + elementwise operation
template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_,
  typename ElementCompute_,
  typename ElementBias_,
  typename ElementwiseFunctor_
>
class LinearCombinationBiasElementwise;

// For simple bias addition (identity elementwise):
using EpilogueBias = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    cutlass::half_t,
    8,
    float,
    float,
    float,
    cutlass::epilogue::thread::Identity          // No activation
>;
```

### Additional Epilogue Operators

```cpp
// LinearCombinationClamp: clamp output to [min, max] range
// Used for quantized output types (INT8, UINT8)
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
using EpilogueClamp = cutlass::epilogue::thread::LinearCombinationClamp<
    int8_t, 8, int32_t, float
>;
// Output: clamp(alpha * AB + beta * C, INT8_MIN, INT8_MAX)

// LinearCombinationGelu: GELU activation
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
// Output: GELU(alpha * AB + beta * C)

// LinearCombinationSigmoid: sigmoid activation
// Output: sigmoid(alpha * AB + beta * C)

// LinearCombinationHardSwish: hard-swish activation
// Output: x * clamp(x + 3, 0, 6) / 6

// LinearCombinationPlanarComplex: for complex-number GEMM
// Handles real and imaginary parts of complex outputs
```

---

## Epilogue Template Parameters

### ElementOutput

```cpp
// The output element type determines:
// 1. The type conversion from accumulator to output
// 2. The memory access width and alignment
// 3. The vectorized store instructions used

// Common output types:
using ElementOutput = float;              // Full precision output
using ElementOutput = cutlass::half_t;    // FP16 output (common for training)
using ElementOutput = cutlass::bfloat16_t;// BF16 output
using ElementOutput = int8_t;             // Quantized output
using ElementOutput = cutlass::float_e4m3_t; // FP8 output
```

### Count (Elements Per Access)

```cpp
// Count controls the vectorization width of output stores
// It should match the memory access pattern of the epilogue

// For 128-bit memory access:
// FP32:  128 / 32 = 4 elements
// FP16:  128 / 16 = 8 elements
// BF16:  128 / 16 = 8 elements
// INT8:  128 / 8  = 16 elements
// FP8:   128 / 8  = 16 elements

// Using cutlass::sizeof_bits for portability:
int Count = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// The Count must also satisfy alignment requirements:
// The leading dimension of C/D must be a multiple of Count
```

### ElementAccumulator

```cpp
// The accumulator type comes from the MMA operation:
// FP16/BF16 inputs -> float accumulator
// INT8 inputs -> int32_t accumulator
// FP64 inputs -> double accumulator
// TF32 inputs -> float accumulator

// The epilogue must be compatible with the accumulator type:
using ElementAccumulator = float;  // For FP16/BF16/TF32 GEMM
using ElementAccumulator = int32_t;  // For INT8 GEMM
```

### ElementCompute

```cpp
// ElementCompute determines the precision of alpha*AB + beta*C computation
// Higher precision reduces rounding errors in the scaling

// Same as accumulator (most common):
using ElementCompute = float;  // Matches accumulator

// For higher precision scaling:
using ElementCompute = double;  // Double precision scaling

// Note: ElementCompute affects register usage and may impact occupancy
```

---

## Scaling Modes

CUTLASS provides several scaling modes that control how alpha and beta are applied.

### ScaleType::Default

```cpp
// Default: D = alpha * AB + beta * C
// Both alpha and beta are applied normally
// C is always loaded from memory
using ScaleDefault = cutlass::epilogue::thread::ScaleType::Default;
```

### NoBetaScaling

```cpp
// NoBetaScaling: D = alpha * AB + C
// Beta is implicitly 1, but C is still loaded and added
// Saves one multiplication per element
using ScaleNoBeta = cutlass::epilogue::thread::ScaleType::NoBetaScaling;
// Output: alpha * AB + C
```

### OnlyAlphaScaling

```cpp
// OnlyAlphaScaling: D = alpha * AB
// C is not loaded at all (beta is implicitly 0)
// Saves all memory reads for the C matrix
using ScaleAlpha = cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling;
// Output: alpha * AB
// Useful when beta = 0 and C is not needed
```

### OnlyAlphaPerRowScaling

```cpp
// OnlyAlphaPerRowScaling: D = alpha_per_row[i] * AB
// Alpha varies per row (used in some quantization schemes)
// Alpha is an array with one value per row of the output
using ScaleAlphaPerRow = cutlass::epilogue::thread::ScaleType::OnlyAlphaPerRowScaling;
```

### Scaling Mode Selection in Operators

```cpp
// Specify the scaling mode as the last template parameter:
using EpilogueNoBeta = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    8,
    float,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling
>;

using EpilogueAlphaOnly = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    8,
    float,
    float,
    cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
>;
```

---

## Params Structure

Each epilogue operator defines a `Params` structure that holds the runtime parameters.

### Direct Values

```cpp
// LinearCombination::Params with direct scalar values
typename LinearCombination<half_t, 8, float, float>::Params params(
    float alpha,     // Scaling factor for AB
    float beta       // Scaling factor for C
);

// When alpha and beta are simple scalars:
float alpha = 1.0f;
float beta = 0.0f;
auto params = EpilogueOp::Params(alpha, beta);
```

### Pointers for Per-Row/Column Scaling

```cpp
// For per-row or per-column scaling, alpha/beta can be arrays
// pointed to by device pointers:

// LinearCombinationBiasRelu::Params
typename LinearCombinationBiasRelu<half_t, 8, float, float>::Params params(
    float alpha,
    float beta,
    float const* bias_ptr   // Device pointer to bias vector
);

// The bias is broadcast: bias[j] is added to every element in column j
```

### Arrays and Batched Parameters

```cpp
// For batched GEMM, the params can include batch-specific values:
// alpha and beta can be the same across all batches, or
// batch_stride can be specified for per-batch alpha/beta

// For split-K, the params include reduction parameters:
// alpha is applied after the split-K reduction
```

---

## Epilogue Threadblock Operations

The threadblock-level epilogue orchestrates how threads in a threadblock collectively write their accumulator results to the output matrix.

### Output Tile Iteration

```cpp
#include "cutlass/epilogue/threadblock/epilogue.h"

// The threadblock epilogue iterates over the output tile in steps
// determined by the warp-level accumulator layout and the output
// access width.

template <
  typename Shape_,                 // Output tile shape (matches ThreadblockShape)
  typename WarpMmaOperator_,       // Warp-level MMA type
  int PartitionsK,                 // Number of partitions in K dimension
  typename OutputTileIterator_,    // Iterator for writing output tiles
  typename AccumulatorFragmentIterator_,  // Iterator over accumulator fragments
  typename WarpTileIterator_,      // Iterator for warp-level accumulator
  typename SharedLoadIterator_,    // Iterator for shared memory loads
  typename OutputOp_,              // Output operator (e.g., LinearCombination)
  typename Padding_                // Padding for bank-conflict-free access
>
class Epilogue;

// Execution flow:
// 1. Each warp stores its accumulator fragment to shared memory
// 2. Shared memory is rearranged for vectorized output access
// 3. OutputTileIterator loads from shared memory in vectorized chunks
// 4. OutputOp transforms each chunk (scale, activate, convert type)
// 5. OutputTileIterator stores the transformed chunk to global memory
```

### Accumulator Storage and Layout

```cpp
// Accumulators are stored in register fragments per warp
// The layout in registers depends on the MMA instruction:
//
// For HMMA.16816 (FP16 TensorOp on SM80):
//   Each warp produces a 16x16 (or larger) tile of accumulators
//   Stored in 8 registers (float) per thread
//
// For SIMT:
//   Each thread produces a small NxN tile
//   Stored directly in thread-local registers

// The epilogue must transform this register layout into the
// output matrix layout. This transformation involves:
// 1. Warp-level: shared memory transpose/rearrange
// 2. Threadblock-level: vectorized store to global memory

// Shared memory layout in the epilogue:
// Aligned to avoid bank conflicts during the warp->shared store
// and the shared->global load

union SharedStorage {
    typename Epilogue::SharedStorage epilogue_smem;
    // Contains the rearranged accumulator data
    // Plus space for C source data (if beta != 0)
};
```

### Type Conversion in Epilogue

```cpp
// The epilogue handles type conversion from accumulator to output type
// This is done element-wise by the output operator

// Conversion examples:
// float -> half_t:   rounding to nearest FP16 value
// float -> bfloat16_t: rounding to BF16 (keep top 16 bits)
// float -> int8_t:   clamp + round + cast
// float -> float_e4m3_t: rounding to FP8 E4M3
// int32_t -> int8_t: clamp to [-128, 127] + cast

// CUTLASS uses cutlass::NumericConverter for type conversion:
// cutlass::NumericConverter<half_t, float> convert;
// half_t result = convert(source_float);

// Special converters for different rounding modes:
// cutlass::NumericConverterCutlass<half_t, float>  -- standard rounding
// cutlass::NumericConverterClamp<int8_t, float>    -- clamp + cast
// cutlass::NumericConverterWithAbsMax<half_t, float> -- absolute max scaling
```

---

## Epilogue Fusion Patterns

Epilogue fusion combines multiple post-GEMM operations into a single pass, avoiding extra memory round-trips.

### Pattern 1: GEMM + Bias + ReLU

```cpp
// Fused: D = max(0, alpha * AB + bias)
// Avoids: writing AB to memory, then reading back for bias+ReLU

using EpilogueFusedBiasRelu = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    8,
    float,
    float,
    float   // bias type
>;

// This is a single fused kernel that:
// 1. Reads accumulators from MMA
// 2. Scales by alpha
// 3. Adds per-column bias
// 4. Applies ReLU
// 5. Converts to half_t
// 6. Stores to global memory
```

### Pattern 2: GEMM + Residual + ReLU

```cpp
// Fused: D = max(0, alpha * AB + beta * C)
// C is the residual connection (e.g., from layer normalization)

using EpilogueResidualRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
    8,
    float,
    float
>;

// Params: alpha = 1.0, beta = 1.0 (typical for residual)
// D = max(0, AB + C)
```

### Pattern 3: GEMM + Scale + Clamp (Quantization)

```cpp
// Fused: D = clamp(alpha * AB, min, max)
// Used for post-GEMM quantization

using EpilogueQuantize = cutlass::epilogue::thread::LinearCombinationClamp<
    int8_t,     // Output in INT8
    16,          // 128-bit access = 16 x INT8
    int32_t,     // Accumulator
    float        // Compute type
>;

// Params: alpha = quantization_scale, beta = 0
// Output: clamp(quant_scale * AB, -128, 127) as int8_t
```

### Pattern 4: GEMM + Bias + GELU

```cpp
// Fused: D = GELU(alpha * AB + bias)
// Common in transformer models

using EpilogueGelu = cutlass::epilogue::thread::LinearCombinationGelu<
    cutlass::half_t,
    8,
    float,
    float
>;

// GELU(x) = x * Phi(x) where Phi is the CDF of standard normal distribution
// Approximated as: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

---

## 3.x Epilogue Collective API

CUTLASS 3.x redesigns the epilogue as a collective operation, mirroring the mainloop design.

### EpilogueBuilder

```cpp
#include "cutlass/epilogue/collective/epilogue_builder.hpp"

// The EpilogueBuilder selects the optimal epilogue implementation
// based on architecture, data types, and schedule.

template <
  typename ArchTag_,               // Target architecture
  typename TileShape_,             // CTA tile shape (must match mainloop)
  typename CollectiveMainloop_,    // Mainloop type (for accumulator layout info)
  typename ElementC_,              // Source element type
  typename LayoutC_,               // Source layout
  int AlignmentC_,                 // Source alignment
  typename ElementD_,              // Destination element type
  typename LayoutD_,               // Destination layout
  int AlignmentD_,                 // Destination alignment
  typename EpilogueSchedule_       // Epilogue schedule policy
>
struct EpilogueBuilder;
```

### Epilogue Schedules

```cpp
// EpilogueScheduleAuto: automatic selection
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

// SM90-specific schedules:
// - TMA store-based epilogue (uses TMA for writing output)
// - Warp-specialized epilogue (DMA warps handle output writes)
// - Sub-tiled epilogue (for tiles larger than TMA atom)
```

### 3.x Epilogue Arguments

```cpp
// In 3.x, the epilogue arguments are part of the kernel arguments:
struct EpilogueArgs {
    ElementC const* ptr_C;                    // Source matrix pointer
    cute::Stride<int64_t, int64_t, int64_t> stride_C;  // Source stride
    ElementD* ptr_D;                          // Destination pointer
    cute::Stride<int64_t, int64_t, int64_t> stride_D;  // Destination stride
    ElementCompute alpha;                      // Alpha scaling factor
    ElementCompute beta;                       // Beta scaling factor
};

// For batched GEMM, additional batch strides are included:
struct BatchedEpilogueArgs {
    ElementC const* ptr_C;
    cute::Stride<int64_t, int64_t, int64_t> stride_C;
    int64_t batch_stride_C;
    ElementD* ptr_D;
    cute::Stride<int64_t, int64_t, int64_t> stride_D;
    int64_t batch_stride_D;
    ElementCompute alpha;
    ElementCompute beta;
};
```

### 3.x Epilogue Fusion

```cpp
// In 3.x, epilogue fusion is handled through visitor patterns
// and template-based composition

// Built-in fusion examples:

// 1. Bias addition:
// The epilogue visitor can accept a bias tensor and add it
// before type conversion

// 2. Activation functions:
// Visitors for ReLU, GELU, SiLU, etc. wrap the output

// 3. Auxiliary tensor store:
// The epilogue can write intermediate results to auxiliary tensors
// (e.g., writing pre-activation values for backward pass)

// Fusion composition:
// auto fused_epilogue = compose(
//     LoadC(beta),
//     ScaleAccum(alpha),
//     AddBias(bias_ptr),
//     ApplyReLU(),
//     ConvertTo<half_t>(),
//     StoreD(ptr_D, stride_D)
// );
```

---

## Custom Epilogue Implementation Guide

### Implementing a Custom Thread-Level Operator (2.x)

```cpp
// Step 1: Define the output operator
template <
  typename ElementOutput,
  int Count,
  typename ElementAccumulator,
  typename ElementCompute
>
class CustomEpilogueOp {
public:
    // Required type aliases
    using ElementOutput_ = ElementOutput;
    using ElementAccumulator_ = ElementAccumulator;
    using ElementCompute_ = ElementCompute;
    static int const kCount = Count;

    // Fragment type: the register storage for accumulator elements
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentCompute = cutlass::Array<ElementCompute, kCount>;

    // Params structure: holds runtime parameters
    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        // Add custom parameters here:
        // ElementCompute custom_param;
        // ElementOutput const* auxiliary_ptr;

        Params() : alpha(ElementCompute(1)), beta(ElementCompute(0)) {}

        Params(
            ElementCompute alpha_,
            ElementCompute beta_
        ) : alpha(alpha_), beta(beta_) {}
    };

private:
    Params params_;

public:
    // Constructor
    CustomEpilogueOp(Params const& params) : params_(params) {}

    // Required: indicates whether source (C) matrix is needed
    bool is_source_needed() const {
        return params_.beta != ElementCompute(0);
    }

    // operator(): process accumulator fragment when source is NOT needed
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const& accum) const {
        // Convert accumulator to compute type
        cutlass::NumericConverter<FragmentCompute, FragmentAccumulator> to_compute;
        FragmentCompute converted = to_compute(accum);

        // Apply alpha scaling
        cutlass::multiplies<FragmentCompute> scale;
        FragmentCompute scaled = scale(params_.alpha, converted);

        // Apply custom operation here
        // Example: square the output
        // for (int i = 0; i < kCount; ++i) {
        //     scaled[i] = scaled[i] * scaled[i];
        // }

        // Convert to output type
        cutlass::NumericConverter<FragmentOutput, FragmentCompute> to_output;
        return to_output(scaled);
    }

    // operator(): process accumulator fragment when source IS needed
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const& accum,
        FragmentOutput const& source
    ) const {
        // Convert both to compute type
        cutlass::NumericConverter<FragmentCompute, FragmentAccumulator> accum_to_compute;
        cutlass::NumericConverter<FragmentCompute, FragmentOutput> source_to_compute;

        FragmentCompute converted_accum = accum_to_compute(accum);
        FragmentCompute converted_source = source_to_compute(source);

        // Compute alpha * accum + beta * source
        cutlass::multiply_add<FragmentCompute> fma;
        FragmentCompute result = fma(params_.alpha, converted_accum,
                                     params_.beta, converted_source);

        // Convert to output type
        cutlass::NumericConverter<FragmentOutput, FragmentCompute> to_output;
        return to_output(result);
    }
};
```

### Using a Custom Epilogue in a GEMM

```cpp
// Plug the custom epilogue into the CUTLASS GEMM:
using CustomEpilogue = CustomEpilogueOp<
    cutlass::half_t,   // ElementOutput
    8,                  // Count
    float,              // ElementAccumulator
    float               // ElementCompute
>;

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CustomEpilogue,                              // <-- Custom epilogue here
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3
>;

// Use it:
float alpha = 1.0f;
float beta = 0.5f;
typename Gemm::Arguments args(
    {M, N, K},
    {ptr_A, K}, {ptr_B, N},
    {ptr_C, N}, {ptr_D, N},
    CustomEpilogue::Params(alpha, beta)
);
```

---

## Performance Considerations for Epilogue

### Memory Bandwidth

```cpp
// The epilogue is often memory-bandwidth-bound because:
// 1. It writes the full output matrix (M * N elements)
// 2. It may read the source matrix C (another M * N elements)
// 3. The data movement can dominate the kernel time for small K

// Optimization: avoid loading C when beta = 0
// LinearCombination and friends check is_source_needed()
// When beta = 0, the C load is completely eliminated

// Always set beta = 0 when C is not needed:
float beta = 0.0f;
auto params = EpilogueOp::Params(alpha, beta);
// The epilogue will skip all C reads
```

### Vectorization

```cpp
// The Count parameter controls vectorization width
// Always use the maximum alignment-friendly width:

// Correct: use 128-bit aligned access
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,  // = 8
    float, float
>;

// Sub-optimal: use 32-bit access
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    2,  // Only 32-bit access
    float, float
>;
// This reduces memory throughput by 4x
```

### Shared Memory Usage

```cpp
// The epilogue uses shared memory for rearranging accumulator data
// Shared memory is shared between mainloop and epilogue (union)

// For large tiles, the epilogue shared memory can be significant:
// epilogue_smem_size = output_tile_elements * sizeof(ElementAccumulator)

// Trade-off:
// Larger tiles -> more epilogue smem -> fewer pipeline stages for mainloop
// This is why StageCountAuto exists: it balances mainloop and epilogue smem needs
```

### Type Conversion Overhead

```cpp
// Type conversion in the epilogue has a cost:
// - float -> half_t: requires conversion instructions (HFMA2, etc.)
// - float -> int8_t: requires clamp + round + convert
// - float -> bfloat16_t: relatively cheap (just masking)

// For maximum throughput, match accumulator and output types:
// float accum -> float output: no conversion needed
// half_t accum -> half_t output: no conversion needed

// However, using float accumulation with half_t output is usually worth
// the conversion cost for numerical accuracy in training workloads
```

### Epilogue Fusion Benefits

```cpp
// Fusion eliminates extra kernel launches and memory round-trips

// Without fusion (separate kernels):
// GEMM kernel:    D1 = AB                    (write M*N elements)
// Bias+ReLU kernel: D2 = max(0, D1 + bias)  (read M*N, write M*N)
// Total: 1 write + 1 read + 1 write = 3 * M * N memory ops

// With fused epilogue:
// GEMM kernel:    D = max(0, AB + bias)      (write M*N elements)
// Total: 1 write = 1 * M * N memory ops
// Savings: 3x reduction in memory traffic for the post-GEMM operations
```

---

## Code Examples

### Example 1: Basic GEMM with LinearCombination Epilogue

```cpp
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    8,
    float,
    float
>;

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp
>;

// D = 1.0 * A * B + 0.0 * C
float alpha = 1.0f;
float beta = 0.0f;  // C is not loaded
typename Gemm::Arguments args(
    {M, N, K},
    {ptr_A, K}, {ptr_B, N},
    {ptr_C, N}, {ptr_D, N},
    {alpha, beta}
);
```

### Example 2: GEMM with Bias + ReLU Fusion

```cpp
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"

using EpilogueBiasRelu = cutlass::epilogue::thread::LinearCombinationBiasRelu<
    cutlass::half_t,
    8,
    float,
    float,
    float   // Bias type
>;

using GemmBiasRelu = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueBiasRelu
>;

// D = max(0, alpha * A * B + bias)
float alpha = 1.0f;
float beta = 0.0f;
float* d_bias;  // device pointer to bias vector of size N

typename GemmBiasRelu::Arguments args(
    {M, N, K},
    {ptr_A, K}, {ptr_B, N},
    {ptr_C, N}, {ptr_D, N},
    {alpha, beta, d_bias}  // Includes bias pointer
);
```

### Example 3: GEMM with Residual Connection

```cpp
// D = alpha * A * B + beta * C (C is the residual)
using EpilogueResidual = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
    8,
    float,
    float
>;

// Typical residual connection: D = AB + C
float alpha = 1.0f;
float beta = 1.0f;  // C is loaded and added

typename Gemm::Arguments args(
    {M, N, K},
    {ptr_A, K}, {ptr_B, N},
    {ptr_C, N},  // ptr_C points to residual data
    {ptr_D, N},  // ptr_D can be same as ptr_C for in-place
    {alpha, beta}
);
```

### Example 4: 3.x Epilogue with EpilogueBuilder

```cpp
#include "cutlass/epilogue/collective/epilogue_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

// Define mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Define epilogue using EpilogueBuilder
using CollectiveEpilogue = typename cutlass::epilogue::collective::EpilogueBuilder<
    cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 64>,
    CollectiveMainloop,
    cutlass::half_t,           // ElementC (source)
    cutlass::layout::RowMajor, // LayoutC
    8,                         // AlignmentC
    cutlass::half_t,           // ElementD (destination)
    cutlass::layout::RowMajor, // LayoutD
    8,                         // AlignmentD
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Build the full GEMM
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentTileScheduler
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Run with epilogue parameters
int M = 4096, N = 3072, K = 2048;
float alpha = 1.0f, beta = 0.5f;

typename Gemm::Arguments args {
    cute::make_shape(M, N, K, 1),
    { ptr_A, stride_A, ptr_B, stride_B },
    { ptr_C, stride_C, ptr_D, stride_D, alpha, beta }
};

Gemm gemm_op;
gemm_op(args);
```

### Example 5: Quantized Output (FP32 Accumulator to INT8)

```cpp
// GEMM with FP16 inputs, FP32 accumulation, INT8 quantized output
using EpilogueQuantize = cutlass::epilogue::thread::LinearCombinationClamp<
    int8_t,      // Output type: INT8
    16,           // 128 / 8 = 16 elements per access
    float,        // Accumulator type: FP32
    float         // Compute type: FP32
>;

using GemmQuantize = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    int8_t,          // Output: INT8
    cutlass::layout::RowMajor,
    float,           // Accumulator: FP32
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueQuantize
>;

// D = clamp(quant_scale * A * B, -128, 127)
float quant_scale = 0.1f;
typename GemmQuantize::Arguments args(
    {M, N, K},
    {ptr_A, K}, {ptr_B, N},
    {ptr_C, N}, {ptr_D, N},
    {quant_scale, 0.0f}  // alpha = quant_scale, beta = 0
);
```

---

## Summary

### Epilogue Quick Reference

| Operator | Formula | Use Case |
|---|---|---|
| `LinearCombination` | `D = alpha*AB + beta*C` | Basic GEMM output |
| `LinearCombinationRelu` | `D = max(0, alpha*AB + beta*C)` | GEMM + ReLU fusion |
| `LinearCombinationBiasRelu` | `D = max(0, alpha*AB + beta*C + bias)` | GEMM + bias + ReLU |
| `LinearCombinationBias` | `D = alpha*AB + beta*C + bias` | GEMM + bias addition |
| `LinearCombinationClamp` | `D = clamp(alpha*AB + beta*C, min, max)` | Quantized output |
| `LinearCombinationGelu` | `D = GELU(alpha*AB + beta*C)` | Transformer models |
| `LinearCombinationSigmoid` | `D = sigmoid(alpha*AB + beta*C)` | Logistic regression |

### Key Design Principles

1. **Always fuse when possible**: combining operations in the epilogue eliminates memory round-trips and kernel launch overhead.

2. **Use beta = 0 when C is not needed**: this skips the entire source matrix load, saving significant bandwidth.

3. **Match Count to alignment**: use `128 / sizeof_bits<ElementOutput>` for maximum vectorization.

4. **Consider accumulator precision**: float accumulation with half_t output gives better accuracy with minimal conversion cost.

5. **3.x epilogue**: use `EpilogueBuilder` with `EpilogueScheduleAuto` for SM90+ targets; it automatically selects the best implementation.

6. **Custom operators**: implement the required interface (Params, is_source_needed, operator()) and plug into the CUTLASS GEMM template.
