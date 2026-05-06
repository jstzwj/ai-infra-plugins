# CUTLASS Reference - Chapter 21: Mixed Precision GEMM

This reference covers mixed precision in CUTLASS: using different numeric types for input operands, accumulation, and output in GEMM operations. Mixed precision is critical for achieving high throughput on Tensor Cores while maintaining numerical accuracy.

---

## 21.1 Mixed Precision GEMM Concept

### 21.1.1 Motivation

Modern NVIDIA Tensor Cores achieve peak throughput by operating on reduced-precision data types (FP16, BF16, TF32, FP8, INT8) while accumulating at higher precision (typically FP32). Mixed precision GEMM exploits this by:

- Storing inputs in a compact format (e.g., FP16 or BF16) to reduce memory bandwidth and storage.
- Performing multiply-accumulate in a higher-precision accumulator (e.g., FP32) to preserve numerical fidelity.
- Writing the result in the desired output format (e.g., FP16, BF16, or FP32).

The general form is:

```
D = alpha * A * B + beta * C
```

Where:
- `A` has element type `ElementA`
- `B` has element type `ElementB`
- `C` has element type `ElementC`
- `D` has element type `ElementD`
- Accumulation occurs in `ElementAccumulator`

These five types need not be the same, enabling a wide range of mixed-precision configurations.

### 21.1.2 Type Conversion Pipeline

The data flows through a type conversion pipeline during GEMM execution:

```
ElementA (memory) --> ElementA (register) --> ElementAccumulator (register)
                                                            |
ElementB (memory) --> ElementB (register) -->               |
                                                            v
                                              ElementAccumulator (result)
                                                            |
                                                            v
                                              ElementOutput (register) --> ElementOutput (memory)
```

Each arrow represents a potential type conversion. CUTLASS handles these conversions automatically based on the template parameters, using the `NumericConverter` and `NumericArrayConverter` facilities.

### 21.1.3 Common Mixed Precision Configurations

| Configuration | ElementA/B | ElementAccumulator | ElementOutput | Use Case |
|---|---|---|---|---|
| FP16 training | `half_t` | `float` | `half_t` | Standard mixed precision training |
| BF16 training | `bfloat16_t` | `float` | `bfloat16_t` | Large model training |
| TF32 training | `tfloat32_t` | `float` | `float` | FP32-compatible training |
| FP16 inference | `half_t` | `half_t` | `half_t` | Low-latency inference |
| FP8 training | `float_e4m3_t` / `float_e5m2_t` | `float` | `float_e4m3_t` | Hopper+ high-throughput training |
| INT8 quantized | `int8_t` | `int32_t` | `int8_t` | Quantized inference |

---

## 21.2 Numeric Converters

### 21.2.1 NumericConverter

`NumericConverter<Target, Source, Round>` performs element-wise conversion from `Source` to `Target` type. It is the fundamental building block for type transitions in the GEMM pipeline.

```cpp
#include "cutlass/numeric_conversion.h"

// Convert FP32 to FP16 with round-to-nearest-even
cutlass::NumericConverter<cutlass::half_t, float, cutlass::FloatRoundStyle::round_to_nearest> converter;

float value_fp32 = 1.5f;
cutlass::half_t value_fp16 = converter(value_fp32);
```

The `Round` template parameter controls the rounding behavior. CUTLASS provides several rounding styles:

| Rounding Style | Description |
|---|---|
| `round_to_nearest` | Round to nearest representable value, ties to even |
| `round_toward_zero` | Truncate toward zero |
| `round_half_ulp_truncate` | Round with half-ULP truncation (for TF32) |
| `round_half_ulp_trunc_dntz` | Combine half-ULP truncation with round-toward-zero for small values |

### 21.2.2 NumericArrayConverter

`NumericArrayConverter<Target, Source, N, Round>` converts an array of `N` elements at once, enabling vectorized type conversion. This is used heavily in the GEMM mainloop where elements are processed in SIMD batches.

```cpp
#include "cutlass/numeric_conversion.h"

// Convert an array of 4 FP32 values to FP16
using Converter = cutlass::NumericArrayConverter<cutlass::half_t, float, 4>;
Converter converter;

cutlass::Array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
cutlass::Array<cutlass::half_t, 4> dst = converter(src);
```

### 21.2.3 Specialized Converters

CUTLASS provides specialized converters for specific type pairs that leverage hardware instructions for maximum performance:

```cpp
// FP32 -> BF16 using hardware cvt instruction (Ampere+)
cutlass::NumericConverter<cutlass::bfloat16_t, float> bf16_converter;

// FP32 -> TF32 with half-ULP rounding
cutlass::NumericConverter<cutlass::tfloat32_t, float,
    cutlass::FloatRoundStyle::round_half_ulp_truncate> tf32_converter;

// FP32 -> FP16 using __half2 conversion for 2 elements at once
cutlass::NumericArrayConverter<cutlass::half_t, float, 2> fp16_pair_converter;

// FP32 -> FP8 e4m3 (Hopper+)
cutlass::NumericConverter<cutlass::float_e4m3_t, float> fp8_converter;

// FP32 -> FP8 e5m2 (Hopper+)
cutlass::NumericConverter<cutlass::float_e5m2_t, float> fp8_e5m2_converter;
```

For sub-byte types (FP8, INT4, INT2, binary), CUTLASS packs multiple elements into a single storage word. The converter handles packing and unpacking automatically:

```cpp
// INT4 array conversion: pack 8 INT4 values into one 32-bit word
using Int4Converter = cutlass::NumericArrayConverter<
    cutlass::int4b_t,    // Target: 4-bit signed integer
    int,                  // Source: 32-bit integer
    8                     // 8 elements per conversion
>;
```

---

## 21.3 Fast Math Operations

### 21.3.1 OpMultiplyAddFastF32

On SM80 (Ampere) and later architectures, Tensor Cores can perform FP32 GEMM using TF32 format at 8x the throughput of standard FP32. The `OpMultiplyAddFastF32` operation instructs the Tensor Core to convert FP32 inputs to TF32 internally:

```cpp
#include "cutlass/gemm/device/gemm.h"

// CUTLASS 2.x: Fast FP32 GEMM using TF32 Tensor Cores
using GemmFastF32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,               // ElementA, LayoutA
    float, cutlass::layout::ColumnMajor,             // ElementB, LayoutB
    float, cutlass::layout::RowMajor,                // ElementC, LayoutC
    float,                                           // ElementAccumulator
    cutlass::arch::OpClassTensorOp,                  // OpClass
    cutlass::arch::Sm80,                             // ArchTag
    cutlass::gemm::GemmShape<128, 128, 32>,          // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,            // WarpShape
    cutlass::gemm::GemmShape<16, 8, 8>,              // InstructionShape (TF32 mma.sync)
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,                                               // Stages
    cutlass::arch::OpMultiplyAddFastF32              // OperationClass: TF32 emulation
>;
```

When using `OpMultiplyAddFastF32`:
- The inputs are FP32 in global memory.
- They are converted to TF32 before entering the Tensor Core (losing ~13 mantissa bits, keeping 10).
- Accumulation is in full FP32.
- Output is FP32.

This provides a good tradeoff: FP32 memory layout and code compatibility with ~8x throughput boost over FP32 scalar cores, at the cost of reduced mantissa precision.

### 21.3.2 OpMultiplyAddFastBF16

Similarly, `OpMultiplyAddFastBF16` uses a faster BF16 path on hardware that supports it:

```cpp
// Fast BF16 operation on Ampere+
using GemmFastBF16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,             // BF16 mma.sync instruction
    cutlass::epilogue::thread::LinearCombination<
        float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAddFastBF16              // Fast BF16 path
>;
```

### 21.3.3 Fast Operation Flags Summary

| Operation | SM Version | Input | Accumulator | Description |
|---|---|---|---|---|
| `OpMultiplyAdd` | All | Native type | Native type | Standard precision |
| `OpMultiplyAddFastF32` | SM80+ | FP32 -> TF32 | FP32 | TF32 emulation for FP32 inputs |
| `OpMultiplyAddFastBF16` | SM80+ | BF16 | FP32 | Fast BF16 multiply-add |
| `OpMultiplyAddFastF16` | SM80+ | FP16 | FP32 | Fast FP16 multiply-add |

---

## 21.4 TF32 Emulation for FP32 Operations

### 21.4.1 What is TF32?

TF32 (TensorFloat-32) is a 19-bit floating-point format introduced with Ampere Tensor Cores:
- **1 sign bit** (same as FP32)
- **8 exponent bits** (same as FP32, same dynamic range)
- **10 mantissa bits** (vs. 23 for FP32, 10 for FP16)

TF32 provides the same dynamic range as FP32 but with the precision of FP16. The conversion from FP32 to TF32 simply truncates 13 mantissa bits.

### 21.4.2 CUTLASS 3.x TF32 GEMM

In CUTLASS 3.x, the CollectiveBuilder handles TF32 configuration automatically when you specify `tfloat32_t` as the input type:

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// TF32 GEMM using CollectiveBuilder
using ElementA = cutlass::tfloat32_t;
using ElementB = cutlass::tfloat32_t;
using ElementC = float;
using ElementD = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 4,    // Alignment 4 for TF32
    ElementB, cutlass::layout::ColumnMajor, 4,
    ElementC,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

### 21.4.3 Explicit TF32 Conversion

You can also explicitly convert FP32 data to TF32 before passing it to CUTLASS:

```cpp
#include "cutlass/numeric_conversion.h"

// Convert FP32 data to TF32 before GEMM
void convert_fp32_to_tf32(const float* src, cutlass::tfloat32_t* dst, int count) {
    cutlass::NumericConverter<cutlass::tfloat32_t, float,
        cutlass::FloatRoundStyle::round_half_ulp_truncate> converter;

    for (int i = 0; i < count; ++i) {
        dst[i] = converter(src[i]);
    }
}
```

---

## 21.5 Mixed Input Type GEMM (SM90 Hopper)

### 21.5.1 Different Types for A and B

Starting with Hopper (SM90), CUTLASS supports GEMM operations where operands A and B have different element types. This is useful for scenarios such as:

- **Quantized inference**: Operand A in FP16, operand B in INT8 (weight quantization).
- **FP8 training**: Operand A in `float_e4m3_t` (activations), operand B in `float_e5m2_t` (gradients).
- **Mixed BF16/FP16**: Operand A in BF16, operand B in FP16.

The CollectiveBuilder in CUTLASS 3.x supports this natively:

```cpp
// Mixed input type GEMM: FP8 e4m3 for A, FP8 e5m2 for B
using ElementA = cutlass::float_e4m3_t;   // Activations: higher precision range
using ElementB = cutlass::float_e5m2_t;   // Weights/gradients: higher dynamic range
using ElementAccumulator = float;
using ElementOutput = cutlass::float_e4m3_t;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,    // FP8: alignment 16
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### 21.5.2 Register File Path for Smaller Types

When the input type is smaller than the accumulator type, Hopper's GMMA (GeMM MMA Assembly) instructions can keep the smaller type in registers during the multiply-accumulate pipeline. This reduces register pressure and shared memory usage:

- **FP16 inputs**: 2 bytes per element, stored as `__half2` pairs in registers.
- **BF16 inputs**: 2 bytes per element, stored similarly to FP16.
- **FP8 inputs**: 1 byte per element, 4x density compared to FP32.
- **INT8 inputs**: 1 byte per element.

The GMMA hardware automatically handles type promotion from the input type to the accumulator type during the multiply-add operation. No explicit conversion code is needed.

### 21.5.3 Type Upcasting Strategy

For mixed types where one operand has lower precision, CUTLASS applies a type upcasting strategy:

```
ElementA (smaller type) ---\
                            +--> ElementAccumulator (larger type)
ElementB (smaller type) ---/
```

The upcasting happens at the instruction level:
- For WGMMA instructions (SM90), the hardware natively supports mixed-type inputs.
- For MMA instructions (SM80), types must match; the CollectiveBuilder handles conversion.

```cpp
// Example: BF16 inputs with FP32 accumulation on SM90
using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,
    float,   // ElementAccumulator: upcast from BF16 to FP32
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

---

## 21.6 Mixed dtype Schedules

### 21.6.1 Kernel Schedule Selection

CUTLASS 3.x provides several kernel schedules optimized for different mixed-precision workloads. The `KernelScheduleAuto` policy selects the best schedule automatically, but you can also specify them explicitly:

| Schedule | Description | Best For |
|---|---|---|
| `KernelTmaWarpSpecialized` | TMA load + warp-specialized MMA | General mixed precision on SM90 |
| `KernelTmaWarpSpecializedPingpong` | Ping-pong between two warp groups | High arithmetic intensity |
| `KernelTmaWarpSpecializedCooperative` | Cooperative MMA across warp groups | Large tile sizes |
| `KernelCpAsyncWarpSpecialized` | cp.async load + warp-specialized | Fallback for non-TMA scenarios |

### 21.6.2 KernelTmaWarpSpecialized

This is the default schedule for SM90 GEMM. It uses:
- **TMA (Tensor Memory Accelerator)** for loading tiles from global memory to shared memory.
- **Warp specialization**: One warp group acts as the producer (loads data via TMA), the other acts as the consumer (executes MMA on Tensor Cores).

```cpp
#include "cutlass/gemm/collective/collective_builder.hpp"

using Schedule = cutlass::gemm::collective::KernelTmaWarpSpecialized;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    Schedule
>::CollectiveOp;
```

### 21.6.3 KernelTmaWarpSpecializedPingpong

The Pingpong schedule doubles throughput by having two warp groups alternate between loading and computing:

```
Time -->  | Load tile 0 | Load tile 1 | Load tile 2 | ...
Warp Grp0 | Compute 0   | (idle)      | Compute 1   | ...
Warp Grp1 | (idle)      | Compute 0   | (idle)      | ...
```

This is effective when the compute time is approximately equal to the load time. It works well with FP8 and FP16 mixed-precision workloads where the Tensor Core throughput is very high.

```cpp
using Schedule = cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    Schedule
>::CollectiveOp;
```

### 21.6.4 KernelTmaWarpSpecializedCooperative

The Cooperative schedule splits the MMA work across multiple warp groups within a thread block. This is useful for very large tile sizes where a single warp group cannot cover the entire output tile efficiently.

```cpp
using Schedule = cutlass::gemm::collective::KernelTmaWarpSpecializedCooperative;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    Schedule
>::CollectiveOp;
```

---

## 21.7 Performance Implications of Mixed Precision

### 21.7.1 Throughput Comparison

The following table shows approximate relative throughput for different mixed-precision configurations on Hopper (SM90), normalized to FP32 scalar:

| Configuration | Relative Throughput | Accuracy |
|---|---|---|
| FP32 scalar | 1x | Full FP32 |
| TF32 Tensor Core | ~8x | ~FP16 mantissa precision |
| BF16 Tensor Core | ~16x | ~FP16 mantissa precision, FP32 range |
| FP16 Tensor Core | ~16x | FP16 precision and range |
| FP8 (e4m3) Tensor Core | ~32x | 4-bit mantissa, 4-bit exponent |
| INT8 Tensor Core | ~32x | Integer, no decimal |

### 21.7.2 Memory Bandwidth Benefits

Mixed precision reduces memory bandwidth requirements proportionally to the reduction in element size:

| Type | Bytes per element | Bandwidth vs FP32 |
|---|---|---|
| FP32 | 4 | 1x |
| TF32 | 4 (same storage as FP32) | 1x (but compute is 8x faster) |
| BF16 | 2 | 2x |
| FP16 | 2 | 2x |
| FP8 | 1 | 4x |
| INT8 | 1 | 4x |
| INT4 | 0.5 | 8x |

### 21.7.3 Accuracy Considerations

When selecting a mixed-precision configuration, consider:

1. **Accumulator type**: Always use FP32 accumulation for FP16/BF16/TF32/FP8 inputs to prevent catastrophic cancellation during summation.
2. **Dynamic range**: BF16 has the same exponent range as FP32, making it more robust for values far from 1.0. FP16 can overflow/underflow more easily.
3. **Mantissa precision**: TF32 and BF16 both have 7-8 bits of mantissa. FP16 has 10 bits. FP8 e4m3 has 3 bits.
4. **Gradient handling**: In training, gradients often require higher dynamic range. Consider using `float_e5m2_t` for gradients (5-bit exponent) and `float_e4m3_t` for activations (4-bit mantissa).

---

## 21.8 Rounding Modes and Saturation

### 21.8.1 Rounding During Conversion

Type conversion in the GEMM pipeline involves rounding when the target type has fewer mantissa bits. CUTLASS provides fine-grained control:

```cpp
#include "cutlass/numeric_conversion.h"

// Round-to-nearest-even (default for most operations)
using RoundNearest = cutlass::FloatRoundStyle::round_to_nearest;

// Truncate toward zero (faster, less accurate)
using RoundTowardZero = cutlass::FloatRoundStyle::round_toward_zero;

// Convert FP32 accumulator to FP16 output with round-to-nearest
using EpilogueConverter = cutlass::NumericConverter<
    cutlass::half_t,      // Output type
    float,                // Source type
    RoundNearest          // Rounding style
>;
```

### 21.8.2 Saturation Arithmetic

When converting from a type with larger dynamic range (e.g., FP32) to a smaller range (e.g., FP16), values outside the representable range are saturated:

- **Overflow**: Values larger than the maximum representable value are clamped to max.
- **Underflow**: Values smaller than the minimum representable value are clamped to min (or zero for denormals).
- **NaN handling**: NaN inputs typically produce NaN outputs.

```cpp
// CUTLASS handles saturation automatically during type conversion
cutlass::NumericConverter<cutlass::half_t, float> converter;

float overflow_value = 70000.0f;   // Beyond FP16 range (max ~65504)
cutlass::half_t result = converter(overflow_value);  // Saturated to FP16 max

float underflow_value = 1e-8f;     // Below FP16 minimum normal
result = converter(underflow_value);  // Flush to zero or denormal
```

### 21.8.3 Integer Saturation

For integer types, saturation is explicit:

```cpp
// INT32 to INT8 with saturation
cutlass::NumericConverter<int8_t, int, cutlass::FloatRoundStyle::round_to_nearest> int8_converter;

int large_value = 300;  // Beyond INT8 range (-128 to 127)
int8_t result = int8_converter(large_value);  // Saturated to 127
```

---

## 21.9 Complete Mixed Precision GEMM Examples

### 21.9.1 FP16 Input / FP32 Accumulator / FP16 Output (Training)

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// FP16 mixed precision training GEMM on Hopper
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Alignment: 8 elements for FP16 (16 bytes)
constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    LayoutD, LayoutC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    CollectiveOp, EpilogueOp
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

// Launch the GEMM
void run_fp16_mixed_precision(
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
        {M, N, K},                          // GemmCoord problem_size
        {A, lda},                            // TensorRef for A
        {B, ldb},                            // TensorRef for B
        {C, ldc},                            // TensorRef for C
        {D, ldd},                            // TensorRef for D
        {alpha, beta}                        // Epilogue scalars
    };

    // Check if the kernel can be executed with these arguments
    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(args, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        // Handle initialization error
        return;
    }

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        // Handle execution error
        return;
    }
}
```

### 21.9.2 BF16 Input / FP32 Accumulator / FP32 Output

```cpp
// BF16 mixed precision GEMM with FP32 output
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;   // Input bias in FP32
using ElementD = float;   // Output in FP32
using ElementAccumulator = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 8,
    ElementB, cutlass::layout::ColumnMajor, 8,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

### 21.9.3 FP8 Mixed Type GEMM (e4m3 x e5m2)

```cpp
// FP8 mixed type GEMM: e4m3 for A, e5m2 for B
using ElementA = cutlass::float_e4m3_t;   // Higher mantissa precision (4 bits)
using ElementB = cutlass::float_e5m2_t;   // Higher exponent range (5 bits)
using ElementAccumulator = float;
using ElementOutput = cutlass::float_e4m3_t;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

### 21.9.4 TF32 Training GEMM

```cpp
// TF32 GEMM for FP32-compatible training
using ElementA = cutlass::tfloat32_t;
using ElementB = cutlass::tfloat32_t;
using ElementAccumulator = float;
using ElementOutput = float;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm80,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 4,
    ElementB, cutlass::layout::ColumnMajor, 4,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

### 21.9.5 INT8 Quantized GEMM

```cpp
// INT8 quantized GEMM with INT32 accumulation
using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementD = int8_t;
using ElementAccumulator = int32_t;

using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 16,
    ElementB, cutlass::layout::ColumnMajor, 16,
    ElementAccumulator,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using EpilogueOp = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::layout::RowMajor, cutlass::layout::RowMajor,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>;

using Kernel = cutlass::gemm::kernel::GemmUniversal<CollectiveOp, EpilogueOp>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
```

---

## 21.10 Best Practices for Mixed Precision

### 21.10.1 Choosing the Right Configuration

1. **Training**: Use FP16 or BF16 inputs with FP32 accumulation. BF16 is preferred for large models due to its wider dynamic range.
2. **Fine-tuning**: TF32 provides a good balance when you need FP32-like behavior with Tensor Core acceleration.
3. **Inference on Hopper+**: FP8 (e4m3) provides the highest throughput. Use e5m2 only for gradients.
4. **Quantized inference**: INT8 with appropriate scale factors provides deterministic integer arithmetic.

### 21.10.2 Alignment Requirements

Mixed-precision types have different alignment requirements for optimal TMA access:

| Type | Recommended Alignment (elements) | Bytes |
|---|---|---|
| FP32 / TF32 | 4 | 16 |
| BF16 / FP16 | 8 | 16 |
| FP8 / INT8 | 16 | 16 |
| INT4 | 32 | 16 |

All alignments target 16-byte boundaries for optimal TMA transaction size.

### 21.10.3 Avoiding Common Pitfalls

1. **Don't use FP16 accumulation for large K**: The FP16 mantissa (10 bits) cannot accurately represent the sum of thousands of products. Always use FP32 accumulation for K > ~512.
2. **Watch for overflow in FP16**: FP16 max is ~65504. Softmax logits or attention scores can exceed this. Use BF16 or FP32 for such values.
3. **Loss scaling for FP16 training**: To prevent gradient underflow, apply a loss scale factor before backpropagation and un-scale before the weight update.
4. **Check alignment**: Misaligned tensors fall back to slower load paths. Ensure leading dimensions are multiples of the alignment requirement.
