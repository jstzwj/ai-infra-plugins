# CUTLASS: Data Types

## Overview

CUTLASS supports a comprehensive set of data types for GPU tensor computation, ranging from standard IEEE floating-point formats to specialized Tensor Core types like FP8, block-scaled formats, and sub-byte integers. Each type has specific alignment requirements, Tensor Core support, and conversion behaviors. Understanding these types is essential for achieving optimal performance and correctness.

All CUTLASS numeric types are defined in header files under `include/cutlass/` and are designed to be compatible with CUDA kernel code (`__device__` callable).

---

## Floating-Point Types

### double (FP64)

Standard IEEE 754 double-precision floating-point (64-bit).

```cpp
// Type alias
using Element = double;

// Properties
// Size:        8 bytes (64 bits)
// Sign:        1 bit
// Exponent:    11 bits
// Mantissa:    52 bits
// Alignment:   8 bytes (64-bit aligned)
// Range:       ~2.23e-308 to ~1.80e+308
// Precision:   ~15-17 decimal digits
```

**Tensor Core support:**
- SM80+ (Ampere): FP64 Tensor Cores via `mma.sync` 8x8x4 instruction
- SM90+ (Hopper): FP64 Tensor Cores via `wgmma.mma_async`
- SM90a: Full FP64 Tensor Core feature set

**Usage notes:**
- FP64 is used in scientific computing, HPC, and applications requiring high numerical precision
- On consumer GPUs (GeForce RTX), FP64 throughput is typically 1/32 of FP32 throughput
- On data center GPUs (A100, H100), FP64 throughput is 1/2 of FP32

```cpp
// Example: FP64 GEMM on SM80
using GemmFp64 = cutlass::gemm::device::Gemm<
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::RowMajor,
    double,                               // Accumulator also FP64
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 8>,
    cutlass::gemm::GemmShape<16, 32, 8>,
    cutlass::gemm::GemmShape<8, 8, 4>     // FP64 instruction shape
>;
```

### float (FP32)

Standard IEEE 754 single-precision floating-point (32-bit).

```cpp
// Type alias
using Element = float;

// Properties
// Size:        4 bytes (32 bits)
// Sign:        1 bit
// Exponent:    8 bits
// Mantissa:    23 bits
// Alignment:   4 bytes (32-bit aligned)
// Range:       ~1.18e-38 to ~3.40e+38
// Precision:   ~6-9 decimal digits
```

**Tensor Core support:**
- SM80+ (Ampere): TF32 mode (via `mma.sync` with TF32 inputs, FP32 accumulate)
- SM80+: No native FP32 Tensor Core; use TF32 for Tensor Core acceleration or SIMT path for full FP32 precision

**Usage notes:**
- FP32 is the default accumulator type for most mixed-precision operations
- For Tensor Core acceleration, convert FP32 inputs to TF32 via `cutlass::tfloat32_t`
- SIMT path (`OpClassSimt`) provides full FP32 precision without Tensor Cores

```cpp
// Example: FP32 SIMT GEMM (no Tensor Cores, full precision)
using GemmFp32Simt = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,           // SIMT path for full FP32
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>     // SIMT: instruction shape is 1x1x1
>;
```

### half_t (FP16)

IEEE 754 half-precision floating-point (16-bit), defined as `cutlass::half_t`.

```cpp
// Type definition
#include <cutlass/half.h>
using Element = cutlass::half_t;

// Properties
// Size:        2 bytes (16 bits)
// Sign:        1 bit
// Exponent:    5 bits
// Mantissa:    10 bits
// Alignment:   2 bytes (16-bit aligned)
// Range:       ~6.10e-5 to 65504
// Precision:   ~3-4 decimal digits
// Has negative zero: Yes
// Supports NaN and Inf: Yes
```

**Tensor Core support:**
- SM70+ (Volta): FP16 Tensor Cores via `wmma` or `mma.sync`, 16x16x4 (Volta), 16x8x8/16x8x16 (Turing+)
- All subsequent architectures support FP16 Tensor Cores
- Most widely used type for deep learning inference and training

**Alignment considerations:**
- For optimal Tensor Core performance, align memory accesses to 8 bytes (4 elements) or more
- CUTLASS typically requires `kAlignment = 8` for FP16 in Tensor Core operations (8 elements = 16 bytes)

```cpp
// Example: FP16 GEMM on SM80
using GemmFp16 = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,                                // FP32 accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>   // FP16 instruction shape
>;
```

**Interoperability with `__half`:**
```cpp
cutlass::half_t h = cutlass::half_t(1.5f);
__half nv_half = __half(h);              // Convert to CUDA native __half
cutlass::half_t h2 = cutlass::half_t(nv_half);  // Convert back
float f = float(h);                       // Convert to float
```

### bfloat16_t (BF16)

Google Brain floating-point format (16-bit), defined as `cutlass::bfloat16_t`.

```cpp
// Type definition
#include <cutlass/bfloat16.h>
using Element = cutlass::bfloat16_t;

// Properties
// Size:        2 bytes (16 bits)
// Sign:        1 bit
// Exponent:    8 bits  (same as FP32)
// Mantissa:    7 bits
// Alignment:   2 bytes (16-bit aligned)
// Range:       ~1.18e-38 to ~3.40e+38 (same range as FP32)
// Precision:   ~2-3 decimal digits (less than FP16)
// Has negative zero: Yes (treated as zero in arithmetic)
```

**Key difference from FP16:** BF16 trades precision for range. It has the same exponent range as FP32 but only 7 bits of mantissa (vs FP16's 10 bits). This makes BF16 better for training deep learning models because it reduces the risk of overflow/underflow.

**Tensor Core support:**
- SM80+ (Ampere): BF16 Tensor Cores via `mma.sync` 16x8x16
- SM90+ (Hopper): BF16 Tensor Cores via `wgmma.mma_async`
- SM100+ (Blackwell): BF16 Tensor Cores

```cpp
// Example: BF16 GEMM on SM80
using GemmBF16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;
```

### tfloat32_t (TF32)

TensorFloat-32 format (19-bit), defined as `cutlass::tfloat32_t`.

```cpp
// Type definition
#include <cutlass/tfloat32.h>
using Element = cutlass::tfloat32_t;

// Properties
// Size:        4 bytes (32 bits stored, 19 bits significant)
// Sign:        1 bit
// Exponent:    8 bits  (same as FP32)
// Mantissa:    10 bits (only 10 bits used from FP32's 23)
// Alignment:   4 bytes (32-bit aligned)
// Range:       Same as FP32
// Precision:   ~3-4 decimal digits (similar to FP16)
```

**Purpose:** TF32 provides a way to use FP32 Tensor Cores with reduced precision but FP32 range. The hardware rounds FP32 inputs to TF32 (10-bit mantissa) internally during the MMA operation, accumulating in full FP32.

**Tensor Core support:**
- SM80+ (Ampere): TF32 Tensor Cores via `mma.sync` 16x8x8
- SM90+ (Hopper): TF32 Tensor Cores via `wgmma.mma_async`

```cpp
// TF32 is typically used implicitly -- FP32 inputs are rounded to TF32
// The inputs are declared as tfloat32_t but stored in 32-bit containers
using GemmTF32 = cutlass::gemm::device::Gemm<
    cutlass::tfloat32_t, cutlass::layout::RowMajor,
    cutlass::tfloat32_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,      // Output in full FP32
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>     // TF32 instruction shape
>;
```

---

## FP8 Types

CUTLASS supports two IEEE FP8 formats introduced with the OCP (Open Compute Project) FP8 specification.

### float_e4m3_t (FP8 E4M3)

```cpp
// Type definition
#include <cutlass/float8.h>
using Element = cutlass::float_e4m3_t;

// Properties
// Size:        1 byte (8 bits)
// Sign:        1 bit
// Exponent:    4 bits  (bias = 7)
// Mantissa:    3 bits
// Alignment:   1 byte
// Range:       ±448
// Precision:   ~1 decimal digit
// Supports NaN: Yes
// Supports Inf: No (max exponent value is valid)
```

**Usage:** FP8 E4M3 is the preferred format for forward-pass activations and weights in deep learning because it provides higher precision than E5M2. The absence of Inf representation allows more values in the dynamic range.

**Tensor Core support:**
- SM89+ (Ada): FP8 Tensor Cores via `mma.sync`
- SM90+ (Hopper): FP8 Tensor Cores via `wgmma.mma_async`
- SM100+ (Blackwell): FP8 Tensor Cores

```cpp
// Example: FP8 E4M3 GEMM on SM90
using GemmFP8E4M3 = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::gemm::collective::OpClassTensorOp,
            cutlass::float_e4m3_t, LayoutA, 16,   // 16-element alignment
            cutlass::float_e4m3_t, LayoutB, 16,
            float,                                 // Accumulator
            TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<0>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp,
        CollectiveEpilogue
    >
>;
```

### float_e5m2_t (FP8 E5M2)

```cpp
// Type definition
#include <cutlass/float8.h>
using Element = cutlass::float_e5m2_t;

// Properties
// Size:        1 byte (8 bits)
// Sign:        1 bit
// Exponent:    5 bits  (bias = 15)
// Mantissa:    2 bits
// Alignment:   1 byte
// Range:       ±57344
// Precision:   ~0.5 decimal digits
// Supports NaN: Yes
// Supports Inf: Yes
```

**Usage:** FP8 E5M2 provides higher dynamic range at the cost of precision, making it suitable for gradient representation in backward passes and for applications that need a wider range.

**Typical mixed-FP8 strategy:** Use E4M3 for forward pass (weights, activations) and E5M2 for backward pass (gradients).

---

## Block-Scaled Types

CUTLASS supports block-scaled floating-point formats introduced with Blackwell (SM100) and the Microscaling (MX) specification.

### float_e2m1_t (FP4)

```cpp
// Type definition
#include <cutlass/float8.h>  // or block_scaled types header
using Element = cutlass::float_e2m1_t;

// Properties
// Size:        4 bits (half byte, stored as packed pairs)
// Sign:        1 bit
// Exponent:    2 bits  (bias = 1)
// Mantissa:    1 bit
// Range:       ±6 (without scaling)
// Precision:   Very low (only 1 bit of mantissa)
```

**Usage:** FP4 is used as the base element type in block-scaled MMA operations. Each block of FP4 elements shares a common scaling factor, enabling efficient representation of weights and activations at very low precision.

### NVFP4

```cpp
// NVFP4 block-scaled format (SM100+)
// Uses float_e2m1_t elements with per-block scale factors (float_e8m0_t)
// Block size is typically 16 elements per scale factor
```

### MX Formats (MXFP4, MXFP6, MXFP8)

The Microscaling (MX) specification defines block-scaled formats with standardized block sizes:

| Format | Element Size | Block Size | Scale Factor Type |
|---|---|---|---|
| MXFP4 | 4 bits | 32 elements | FP8 E8M0 |
| MXFP6 (FP6 E2M3) | 6 bits | 32 elements | FP8 E8M0 |
| MXFP6 (FP6 E3M2) | 6 bits | 32 elements | FP8 E8M0 |
| MXFP8 (E4M3) | 8 bits | 32 elements | FP8 E8M0 |
| MXFP8 (E5M2) | 8 bits | 32 elements | FP8 E8M0 |

```cpp
// Block-scaled GEMM using NVFP4 on Blackwell (conceptual)
using BlockScaledGemm = /* uses float_e2m1_t elements with float_e8m0_t scales */;
```

**Tensor Core support for block-scaled types:**
- SM100+ (Blackwell): Native block-scaled MMA via specialized `wgmma` instructions
- Earlier architectures: Software emulation only (no hardware acceleration)

---

## Integer Types

### Standard Integer Types

```cpp
// Standard integer types usable in CUTLASS
using Element = int8_t;    // 8-bit signed integer
using Element = uint8_t;   // 8-bit unsigned integer
using Element = int32_t;   // 32-bit signed integer (commonly used as accumulator for int8 GEMM)
using Element = int64_t;   // 64-bit signed integer
```

### Sub-byte Integer Types

CUTLASS defines packed sub-byte integer types for quantized operations:

```cpp
#include <cutlass/integer_subbyte.h>

// 4-bit signed integer (packed two per byte)
using Element = cutlass::int4b_t;

// 4-bit unsigned integer (packed two per byte)
using Element = cutlass::uint4b_t;

// 2-bit unsigned integer (packed four per byte)
using Element = cutlass::uint2b_t;

// 1-bit binary (packed eight per byte)
using Element = cutlass::bin1_t;
```

### Sub-byte Type Properties

| Type | Bits per Element | Packing | Alignment | Tensor Core Support |
|---|---|---|---|---|
| `int4b_t` | 4 | 2 per byte | 1 byte | SM75+ (INT4 Tensor Cores) |
| `uint4b_t` | 4 | 2 per byte | 1 byte | SM75+ (INT4 Tensor Cores) |
| `uint2b_t` | 2 | 4 per byte | 1 byte | Limited |
| `bin1_t` | 1 | 8 per byte | 1 byte | SM75+ (INT1 Tensor Cores) |
| `int8_t` | 8 | 1 per byte | 1 byte | SM75+ (INT8 Tensor Cores) |
| `uint8_t` | 8 | 1 per byte | 1 byte | SM75+ (INT8 Tensor Cores) |

### INT8 GEMM Example

```cpp
// INT8 GEMM with INT32 accumulator
using GemmInt8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,     // A: INT8
    int8_t, cutlass::layout::ColumnMajor,   // B: INT8
    int32_t, cutlass::layout::RowMajor,     // C/D: INT32
    int32_t,                                 // Accumulator: INT32
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>      // INT8 instruction shape
>;
```

---

## Complex Number Types

CUTLASS provides complex number support for both floating-point and integer types.

```cpp
#include <cutlass/complex.h>

// Complex type template
using Element = cutlass::complex<float>;      // complex<float>
using Element = cutlass::complex<double>;     // complex<double>
using Element = cutlass::complex<cutlass::half_t>;  // complex<half_t>

// Properties of complex<T>
// Size:        2 * sizeof(T)
// Alignment:   2 * alignof(T)
// Members:     real(), imag()
```

### Complex Number Operations

```cpp
cutlass::complex<float> a(1.0f, 2.0f);   // 1 + 2i
cutlass::complex<float> b(3.0f, 4.0f);   // 3 + 4i

// Arithmetic
auto sum = a + b;         // (4, 6)
auto prod = a * b;        // (1*3 - 2*4, 1*4 + 2*3) = (-5, 10)
auto conj_a = conj(a);    // (1, -2)
float abs_a = abs(a);     // sqrt(1 + 4) = sqrt(5)
```

### Complex GEMM

CUTLASS supports both genuine complex GEMM (complex multiply-add) and planar complex GEMM (real and imaginary parts stored in separate matrices):

```cpp
// Complex GEMM: C = alpha * A * B + beta * C with complex arithmetic
using GemmComplex = cutlass::gemm::device::GemmComplex<
    cutlass::complex<half_t>, cutlass::layout::RowMajor,
    cutlass::complex<half_t>, cutlass::layout::ColumnMajor,
    cutlass::complex<float>, cutlass::layout::RowMajor,
    cutlass::complex<float>,                    // Accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;
```

---

## Numeric Conversion

CUTLASS provides conversion utilities for safely converting between numeric types.

### NumericConverter

The `NumericConverter` template handles scalar type conversions with appropriate rounding:

```cpp
#include <cutlass/numeric_conversion.h>

// Converter from float to half_t
cutlass::NumericConverter<cutlass::half_t, float> converter;
cutlass::half_t result = converter(1.5f);

// Converter from float to bfloat16_t
cutlass::NumericConverter<cutlass::bfloat16_t, float> bf16_converter;
cutlass::bfloat16_t bf16_result = bf16_converter(3.14f);

// Converter from half_t to float
cutlass::NumericConverter<float, cutlass::half_t> to_float;
float f = to_float(half_val);
```

### NumericArrayConverter

For converting arrays of values efficiently:

```cpp
// Convert array of float to array of half_t
cutlass::NumericArrayConverter<cutlass::half_t, float, 4> array_converter;
cutlass::Array<cutlass::half_t, 4> half_array = array_converter(float_array);

// Convert array of half_t to array of float
cutlass::NumericArrayConverter<float, cutlass::half_t, 4> to_float_array;
cutlass::Array<float, 4> float_array = to_float_array(half_array);
```

### Rounding Modes

CUTLASS supports several rounding modes for type conversion:

```cpp
namespace cutlass {

// Rounding modes for half-precision conversions
struct half_rn {};      // Round to nearest even (default)
struct half_rz {};      // Round toward zero (truncate)
struct half_rp {};      // Round toward positive infinity
struct half_rm {};      // Round toward negative infinity

// Rounding modes for half2 (paired half) conversions
struct half_2_rn {};    // Round to nearest even, both elements
struct half_2_rz {};    // Truncate, both elements
struct half_2_rp {};    // Round toward +inf, both elements
struct half_2_rm {};    // Round toward -inf, both elements

// Convenience aliases
using FastLinearCombinationClamp = /* saturating conversion with clamp */;
using FloatRoundStyle = cutlass::FloatRoundStyle;  // Enum for rounding mode

}  // namespace cutlass
```

### Conversion with Saturation

```cpp
// Saturating conversion: clamps to destination type's range
// float -> int8_t: values outside [-128, 127] are clamped
cutlass::NumericConverter<int8_t, float,
    cutlass::SaturationFreeConverterTag  // No saturation (wrap-around)
> converter_no_sat;

cutlass::NumericConverter<int8_t, float,
    cutlass::platform::is_same<float, float>::value  // Default: saturating
> converter_sat;
```

---

## Type Traits

CUTLASS provides type traits for querying properties of numeric types:

```cpp
#include <cutlass/numeric_types.h>

// Check if a type supports negative zero
constexpr bool has_neg_zero = cutlass::has_negative_zero<cutlass::half_t>::value;
// half_t: true, bfloat16_t: true (but treated as zero in arithmetic)

// Get the unpacked element type (for sub-byte types, returns the storage type)
using Unpacked = cutlass::get_unpacked_element_type<cutlass::int4b_t>::type;
// Unpacked = int8_t (int4b_t is stored as int8_t with two values packed)

// Check if a type is a sub-byte type
constexpr bool is_subbyte = cutlass::is_subbyte<cutlass::int4b_t>::value;  // true
constexpr bool is_subbyte_fp = cutlass::is_subbyte<cutlass::half_t>::value; // false

// Get the number of bits per element
constexpr int bits = cutlass::sizeof_bits<cutlass::half_t>::value;  // 16
constexpr int bits4 = cutlass::sizeof_bits<cutlass::int4b_t>::value; // 4

// Check alignment requirements
constexpr int alignment = cutlass::AlignmentOf<cutlass::half_t>::value;  // 2
constexpr int alignment_fp32 = cutlass::AlignmentOf<float>::value;       // 4
```

---

## Subbyte References and Storage

Sub-byte types require special handling because they cannot be directly addressed in memory. CUTLASS provides specialized reference types:

```cpp
#include <cutlass/subbyte_ref.h>

// Subbyte reference for int4b_t
// Stores a reference to the byte containing the packed 4-bit value
cutlass::SubbyteReference<int4b_t> ref(ptr, bit_offset);

// Reading a sub-byte element
int4b_t value = ref.get();

// Writing a sub-byte element
ref.set(int4b_t(5));

// Subbyte pointer arithmetic
cutlass::subbyte_iterator<int4b_t> it(ptr);
++it;  // Advances by 4 bits
```

### Sub-byte Storage Layout

```
Byte:  [0]          [1]          [2]          [3]
       |lo  |hi  | |lo  |hi  | |lo  |hi  | |lo  |hi  |
       |e0  |e1  | |e2  |e3  | |e4  |e5  | |e6  |e7  |

For int4b_t (4 bits per element):
- Element 0 is in bits [0:3] of byte 0
- Element 1 is in bits [4:7] of byte 0
- Element 2 is in bits [0:3] of byte 1
- ...

For uint2b_t (2 bits per element):
- Element 0 is in bits [0:1] of byte 0
- Element 1 is in bits [2:3] of byte 0
- Element 2 is in bits [4:5] of byte 0
- Element 3 is in bits [6:7] of byte 0
- ...

For bin1_t (1 bit per element):
- 8 elements packed per byte
```

---

## Alignment Requirements

Each CUTLASS data type has alignment requirements that must be met for correct and efficient Tensor Core operations:

| Type | Size (bytes) | Minimum Alignment | Recommended Tensor Core Alignment |
|---|---|---|---|
| `double` | 8 | 8 bytes | 16 bytes |
| `float` | 4 | 4 bytes | 16 bytes |
| `tfloat32_t` | 4 | 4 bytes | 16 bytes |
| `half_t` | 2 | 2 bytes | 16 bytes (8 elements) |
| `bfloat16_t` | 2 | 2 bytes | 16 bytes (8 elements) |
| `float_e4m3_t` | 1 | 1 byte | 16 bytes (16 elements) |
| `float_e5m2_t` | 1 | 1 byte | 16 bytes (16 elements) |
| `int8_t` | 1 | 1 byte | 16 bytes (16 elements) |
| `int4b_t` | 0.5 | 1 byte | 16 bytes (32 elements) |
| `bin1_t` | 0.125 | 1 byte | 4 bytes (32 elements) |

**Memory allocation alignment:**

```cpp
// Always align allocations to at least 16 bytes for CUTLASS
void* ptr;
cudaMalloc(&ptr, size);  // cudaMalloc returns 256-byte aligned memory

// Or use CUTLASS utilities
cutlass::device_memory::allocation<cutlass::half_t> tensor(M * N);
// Internally uses cudaMalloc with proper alignment
```

---

## Data Type Selection Guide

### By Use Case

| Use Case | Input Type | Accumulator | Output Type | Rationale |
|---|---|---|---|---|
| LLM Training (forward) | FP16 or BF16 | FP32 | FP16 or BF16 | Standard mixed-precision training |
| LLM Training (backward) | FP16 or BF16 | FP32 | FP16 or BF16 | Same as forward, with gradient scaling |
| LLM Inference | FP8 E4M3 | FP32 or FP16 | FP16 | Maximum throughput with acceptable accuracy |
| LLM Quantized Inference | INT8 | INT32 | FP16 or INT8 | Post-training quantization |
| Scientific Computing | FP64 | FP64 | FP64 | Maximum precision required |
| CV Training (forward) | FP16 | FP32 | FP16 | Standard for vision models |
| CV Training (TF32) | TF32 | FP32 | FP32 | Better range than FP16 |
| Edge Inference | INT8 | INT32 | INT8 | Quantized for edge deployment |
| Ultra-low precision | FP4 (block-scaled) | FP32 | FP16 | Blackwell block-scaled MMA |
| Quantized Inference | INT4 | INT32 | FP16 or INT8 | Aggressive quantization |

### By Architecture

| Architecture | Best Throughput Type | Notes |
|---|---|---|
| SM70 (Volta) | FP16 | Only FP16 Tensor Cores available |
| SM75 (Turing) | FP16, INT8 | INT4 and INT1 also supported |
| SM80 (Ampere) | BF16, TF32, FP16 | BF16 and TF32 are new |
| SM89 (Ada) | FP8 E4M3 | FP8 support added |
| SM90 (Hopper) | FP8 E4M3/E5M2 | Best throughput with FP8 on Tensor Cores |
| SM100 (Blackwell) | FP4 block-scaled | Block-scaled formats provide highest throughput |

---

## Tensor Core Data Type Support Per Architecture

### Summary Table

| Data Type | SM70 | SM75 | SM80 | SM89 | SM90 | SM100 |
|---|---|---|---|---|---|---|
| FP64 | -- | -- | Yes | Yes | Yes (90a) | Yes |
| FP32 (TF32) | -- | -- | Yes | Yes | Yes | Yes |
| FP16 | Yes | Yes | Yes | Yes | Yes | Yes |
| BF16 | -- | -- | Yes | Yes | Yes | Yes |
| FP8 E4M3 | -- | -- | -- | Yes | Yes | Yes |
| FP8 E5M2 | -- | -- | -- | Yes | Yes | Yes |
| INT8 | -- | Yes | Yes | Yes | Yes | Yes |
| INT4 | -- | Yes | Yes | Yes | Yes | Yes |
| INT1 (binary) | -- | Yes | Yes | Yes | Yes | Yes |
| FP4 block-scaled | -- | -- | -- | -- | -- | Yes |
| MXFP4/6/8 | -- | -- | -- | -- | -- | Yes |

### MMA Instruction Sizes by Type and Architecture

| Type | SM70 (wmma) | SM75 (mma.sync) | SM80 (mma.sync) | SM90 (wgmma) | SM100 (wgmma) |
|---|---|---|---|---|---|
| FP64 | -- | -- | 8x8x4 | 64xN x K | 64xN x K |
| FP32/TF32 | -- | -- | 16x8x8 | 64xN x K | 64xN x K |
| FP16 | 16x16x4 | 16x8x8/16x8x16 | 16x8x16 | 64xN x K | 64xN x K |
| BF16 | -- | -- | 16x8x16 | 64xN x K | 64xN x K |
| FP8 | -- | -- | -- | 64xN x K | 64xN x K |
| INT8 | -- | 8x8x16 | 16x8x32 | 64xN x K | 64xN x K |
| INT4 | -- | 8x8x32 | 8x8x32 | -- | -- |
| Block-scaled | -- | -- | -- | -- | 64xN x K |

---

## Summary

CUTLASS provides a comprehensive type system covering all NVIDIA Tensor Core data types from FP64 down to 1-bit binary. Key takeaways:

1. **Floating-point types** (FP64, FP32, FP16, BF16, TF32) cover the standard precision range for training and inference
2. **FP8 types** (E4M3, E5M2) provide 2x throughput over FP16 on SM89+ hardware
3. **Block-scaled types** (FP4, NVFP4, MX formats) enable ultra-low-precision computation on Blackwell (SM100+)
4. **Integer types** (INT8, INT4, INT1) support quantized inference workloads
5. **Complex types** support scientific computing workloads
6. **Numeric conversion** utilities handle safe type conversion with configurable rounding modes
7. **Sub-byte types** are packed and require specialized reference types
8. **Alignment requirements** must be respected for correct and efficient Tensor Core operation
