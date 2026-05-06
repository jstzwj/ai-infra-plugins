# CUTLASS Subbyte Reference and Numeric Conversion - Chapter 37: Subbyte Types, Numeric Converters, and Type Traits

This reference covers sub-byte element access, numeric type conversion, rounding modes, and related type traits used in CUTLASS for working with quantized and reduced-precision data types.

---

## 37.1 Overview

CUTLASS supports data types that are smaller than one byte (INT4, INT2, binary 1-bit, and sub-byte floating-point formats). These sub-byte types require special handling because standard C++ memory access operates at byte granularity. CUTLASS provides `SubbyteReference` and `ConstSubbyteReference` to handle packing and unpacking of sub-byte elements within standard byte-addressable memory.

Additionally, CUTLASS provides `NumericConverter` and `NumericArrayConverter` for converting between different numeric types with controlled rounding behavior, which is essential for mixed-precision training and inference.

---

## 37.2 SubbyteReference

### 37.2.1 Template Definition

```cpp
template <typename Element_, int NumBits_ = sizeof_bits<Element_>::value>
class SubbyteReference;
```

| Parameter | Description |
|---|---|
| `Element_` | The logical element type (e.g., `int4b_t`, `uint4b_t`, `bin1_t`) |
| `NumBits_` | Number of bits per element (default: computed from element type) |

### 37.2.2 Supported Sub-byte Types

| Type | Bits | Range | Description |
|---|---|---|---|
| `int4b_t` | 4 | -8 to 7 | 4-bit signed integer |
| `uint4b_t` | 4 | 0 to 15 | 4-bit unsigned integer |
| `int2b_t` | 2 | -2 to 1 | 2-bit signed integer |
| `uint2b_t` | 2 | 0 to 3 | 2-bit unsigned integer |
| `bin1_t` | 1 | 0 to 1 | Binary (1-bit) value |

### 37.2.3 Storage Model

Sub-byte elements are packed into standard byte-sized storage. Multiple elements share a single byte:

```
For 4-bit types (2 elements per byte):
  Byte[0]: [Element1 (high nibble) | Element0 (low nibble)]
  Byte[1]: [Element3 (high nibble) | Element2 (low nibble)]

For 2-bit types (4 elements per byte):
  Byte[0]: [Elem3 | Elem2 | Elem1 | Elem0]
  Each element occupies 2 bits

For 1-bit types (8 elements per byte):
  Byte[0]: [b7 | b6 | b5 | b4 | b3 | b2 | b1 | b0]
  Each element occupies 1 bit
```

### 37.2.4 Construction and Usage

```cpp
#include "cutlass/subbyte_reference.h"

// Create a subbyte reference from a pointer and element index
// The pointer must be aligned to at least 1 byte (8 bits)
uint8_t* storage = ...;

// Reference to element at index 0 (first 4 bits in byte 0)
cutlass::SubbyteReference<int4b_t> ref0(storage, 0);

// Reference to element at index 1 (second 4 bits in byte 0)
cutlass::SubbyteReference<int4b_t> ref1(storage, 1);

// Reference to element at index 2 (first 4 bits in byte 1)
cutlass::SubbyteReference<int4b_t> ref2(storage, 2);

// Read an element
int4b_t value = ref0;  // Reads the packed 4-bit value and unpacks it

// Write an element
ref0 = int4b_t(5);  // Packs 5 into the low nibble of byte[0]

// Pointer arithmetic is in units of elements, not bytes
ref0 += 4;  // Advances by 4 elements = 2 bytes for 4-bit type
```

### 37.2.5 Read/Write Operations

```cpp
// SubbyteReference acts like a regular reference through operator= and implicit conversion:

cutlass::SubbyteReference<int4b_t> ref(storage, index);

// Read via implicit conversion
int4b_t value = ref;

// Write via operator=
ref = int4b_t(3);

// Read-modify-write
int4b_t old = ref;
ref = int4b_t(old + 1);
```

### 37.2.6 Pointer Arithmetic

```cpp
// SubbyteReference supports pointer arithmetic in element units:
cutlass::SubbyteReference<int4b_t> ref(storage, 0);

// Increment by one element
++ref;  // Moves to the next 4-bit element

// Increment by N elements
ref += 4;  // Moves forward by 4 elements

// Decrement
--ref;
ref -= 2;

// Distance between two references
auto ref_a = cutlass::SubbyteReference<int4b_t>(storage, 0);
auto ref_b = cutlass::SubbyteReference<int4b_t>(storage, 8);
// Distance = 8 elements = 4 bytes

// Conversion to raw pointer (returns the byte pointer and bit offset)
uint8_t* byte_ptr = ref.byte_pointer();
int bit_offset = ref.bit_offset();
```

### 37.2.7 Alignment Requirements

```cpp
// SubbyteReference has specific alignment requirements:
// - The underlying byte pointer should be at least 1-byte aligned
// - For optimal performance, align to 4-byte or 8-byte boundaries
// - The bit offset is computed as (index * NumBits) % 8

// Elements per byte computation:
// Elements per byte = 8 / NumBits
// For 4-bit: 2 elements per byte
// For 2-bit: 4 elements per byte
// For 1-bit: 8 elements per byte

// Byte offset for element at index i:
// byte_offset = (i * NumBits) / 8
// bit_offset  = (i * NumBits) % 8

// Example: accessing element 5 of int4b_t array
// byte_offset = (5 * 4) / 8 = 20 / 8 = 2
// bit_offset  = (5 * 4) % 8 = 20 % 8 = 4
// So element 5 starts at bit 4 of byte 2 (high nibble)
```

### 37.2.8 Atomic Operations for Concurrent Access

```cpp
// SubbyteReference provides atomic operations for thread-safe access:
// These are necessary because packing multiple elements in one byte
// means concurrent writes to different elements in the same byte
// could corrupt each other.

cutlass::SubbyteReference<int4b_t> ref(storage, index);

// Atomic store
ref.store(int4b_t(5));  // Uses atomic compare-and-swap internally

// Atomic load
int4b_t val = ref.load();  // Reads the packed value atomically

// Note: Atomic subbyte operations are significantly slower than
// non-atomic ones due to the read-modify-write cycle required.
// Prefer batch operations when possible.
```

---

## 37.3 ConstSubbyteReference

A read-only variant of `SubbyteReference` that does not provide write access:

```cpp
template <typename Element_, int NumBits_ = sizeof_bits<Element_>::value>
class ConstSubbyteReference {
public:
    using Element = Element_;

    // Construction from pointer and element index
    CUTLASS_HOST_DEVICE
    ConstSubbyteReference(uint8_t const *ptr, int index);

    // Construction from SubbyteReference (implicit conversion)
    CUTLASS_HOST_DEVICE
    ConstSubbyteReference(SubbyteReference<Element_, NumBits_> const &ref);

    // Read access only
    CUTLASS_HOST_DEVICE
    Element get() const;

    CUTLASS_HOST_DEVICE
    operator Element() const;

    // Pointer access
    CUTLASS_HOST_DEVICE
    uint8_t const *byte_pointer() const;

    CUTLASS_HOST_DEVICE
    int bit_offset() const;

    // No operator= for writing
};

// Usage:
uint8_t const* read_only_data = ...;
cutlass::ConstSubbyteReference<int4b_t> cref(read_only_data, 0);
int4b_t value = cref;  // OK: read
// cref = int4b_t(3);  // ERROR: cannot write to ConstSubbyteReference
```

---

## 37.4 Packed and Unpacked Storage

### 37.4.1 Storage vs Logical Types

```cpp
// Packed storage type: the physical type used to store elements in memory
// For sub-byte types, this is uint8_t (or uint32_t for wider packed access)

// The packed storage type for an element type:
template <typename Element>
struct storage_type {
    using type = typename platform::conditional<
        (cutlass::sizeof_bits<Element>::value < 8),
        uint8_t,
        Element
    >::type;
};

// For int4b_t: storage_type<int4b_t>::type = uint8_t
// For int8_t:  storage_type<int8_t>::type = int8_t
// For half_t:  storage_type<half_t>::type = half_t

// Number of elements stored in one storage unit:
template <typename Element>
struct elements_per_storage {
    static int const value = 8 / cutlass::sizeof_bits<Element>::value;
    // For int4b_t: 8/4 = 2
    // For int2b_t: 8/2 = 4
    // For bin1_t:  8/1 = 8
};
```

### 37.4.2 Packing and Unpacking

```cpp
// Pack multiple sub-byte elements into a storage word
template <typename Element>
CUTLASS_HOST_DEVICE
uint8_t pack(Element e0, Element e1);  // For 4-bit types

// Unpack a storage word into sub-byte elements
template <typename Element>
CUTLASS_HOST_DEVICE
Element unpack(uint8_t packed, int index);

// Example for int4b_t:
uint8_t packed = (int4b_t(3) & 0xF) | ((int4b_t(-2) & 0xF) << 4);
// packed = 0b11100011 = 0xE3
// Element 0 (low nibble): 3
// Element 1 (high nibble): -2 (stored as 0xE = 14 in unsigned, interpreted as -2)

// Unpack:
int4b_t e0 = static_cast<int4b_t>(packed & 0xF);         // 3
int4b_t e1 = static_cast<int4b_t>((packed >> 4) & 0xF);  // -2
```

---

## 37.5 Numeric Conversion

### 37.5.1 NumericConverter

`NumericConverter` handles conversion between different numeric types with controlled rounding behavior.

```cpp
template <typename Source, typename Target,
          FloatRoundStyle Round = FloatRoundStyle::round_indeterminate>
struct NumericConverter;
```

| Parameter | Description |
|---|---|
| `Source` | Input type |
| `Target` | Output type |
| `Round` | Rounding mode (default: `round_indeterminate`) |

```cpp
#include "cutlass/numeric_conversion.h"

// Basic usage: convert float to half
cutlass::NumericConverter<float, cutlass::half_t> converter;
cutlass::half_t result = converter(3.14159f);

// With explicit rounding mode
cutlass::NumericConverter<float, cutlass::half_t, cutlass::FloatRoundStyle::round_toward_zero>
    truncating_converter;
cutlass::half_t truncated = truncating_converter(3.14159f);

// Convert from accumulator (float) to output (int8_t)
cutlass::NumericConverter<float, int8_t, cutlass::FloatRoundStyle::round_half_ulp_truncate>
    quantize_converter;
int8_t quantized = quantize_converter(2.7f);  // Rounds toward zero with half-ULP adjustment
```

### 37.5.2 NumericArrayConverter

`NumericArrayConverter` handles vectorized conversion of arrays of elements, leveraging hardware vector instructions when available.

```cpp
template <typename Source, typename Target, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_indeterminate>
struct NumericArrayConverter;
```

```cpp
// Convert an array of 4 floats to 4 half_t values
using ArrayConverter = cutlass::NumericArrayConverter<float, cutlass::half_t, 4>;
ArrayConverter converter;

cutlass::Array<float, 4> src = {1.0f, 2.0f, 3.0f, 4.0f};
cutlass::Array<cutlass::half_t, 4> dst = converter(src);

// Convert FP32 accumulator array to BF16 output array
using Fp32ToBf16 = cutlass::NumericArrayConverter<float, cutlass::bfloat16_t, 8>;
Fp32ToBf16 bf16_converter;
cutlass::Array<float, 8> fp32_array = {...};
cutlass::Array<cutlass::bfloat16_t, 8> bf16_array = bf16_converter(fp32_array);

// FP16 to FP32 (widening conversion, no rounding needed)
using Fp16ToFp32 = cutlass::NumericArrayConverter<cutlass::half_t, float, 8>;
Fp16ToFp32 widen;
cutlass::Array<cutlass::half_t, 8> fp16_data = {...};
cutlass::Array<float, 8> fp32_data = widen(fp16_data);
```

### 37.5.3 Rounding Modes

CUTLASS defines several rounding modes via the `FloatRoundStyle` enum:

```cpp
enum class FloatRoundStyle {
    round_indeterminate,           // Fastest, rounding mode unspecified
    round_toward_zero,             // Truncate toward zero
    round_to_nearest,              // Round to nearest, ties to even
    round_toward_infinity,         // Round toward positive infinity
    round_toward_neg_infinity,     // Round toward negative infinity
    round_half_ulp_truncate,       // Truncate with half-ULP adjustment for accuracy
    round_half_ulp_truncate_dntz   // Combination of half-ULP truncate and DNTZ
};
```

#### round_indeterminate

The default rounding mode. Uses whatever the hardware provides, which is typically round-to-nearest-even. Fastest but results may vary.

```cpp
cutlass::NumericConverter<double, float, cutlass::FloatRoundStyle::round_indeterminate> conv;
float result = conv(1.23456789012345);  // Hardware-default rounding
```

#### round_toward_zero (Truncation)

Discards fractional bits. Always rounds toward zero.

```cpp
cutlass::NumericConverter<float, int8_t, cutlass::FloatRoundStyle::round_toward_zero> conv;
int8_t result = conv(2.7f);   // Returns 2
int8_t result2 = conv(-2.7f); // Returns -2
```

#### round_to_nearest

Rounds to the nearest representable value. Ties go to even.

```cpp
cutlass::NumericConverter<float, cutlass::half_t, cutlass::FloatRoundStyle::round_to_nearest> conv;
// Standard IEEE 754 rounding behavior
```

#### round_half_ulp_truncate

Special rounding mode for downcasting that adds a half-ULP bias before truncation. This provides better accuracy for cascaded reductions by compensating for systematic truncation bias.

```cpp
// Used when converting from higher precision to lower precision
// in reduction operations to avoid bias accumulation:
cutlass::NumericConverter<float, cutlass::half_t,
    cutlass::FloatRoundStyle::round_half_ulp_truncate> conv;

// Internally:
// 1. Add 0.5 ULP of the target type to the source value
// 2. Truncate toward zero
// This ensures the average rounding error is close to zero
```

### 37.5.4 Type-Specific Conversion Specializations

#### Float to Half (FP32 to FP16)

```cpp
// Standard conversion
template <>
struct NumericConverter<float, cutlass::half_t, FloatRoundStyle::round_indeterminate> {
    CUTLASS_HOST_DEVICE
    cutlass::half_t operator()(float const &source) const {
        return cutlass::half_t(source);  // Hardware conversion
    }
};

// Truncating conversion (for FP32 -> FP16 in backward pass)
template <>
struct NumericConverter<float, cutlass::half_t, FloatRoundStyle::round_toward_zero> {
    CUTLASS_HOST_DEVICE
    cutlass::half_t operator()(float const &source) const {
        // Truncate to FP16 range without rounding
        uint32_t bits = reinterpret_cast<uint32_t const &>(source);
        bits = bits & 0xFFFFE000;  // Clear low mantissa bits
        float truncated = reinterpret_cast<float &>(bits);
        return cutlass::half_t(truncated);
    }
};
```

#### Float to BF16 (FP32 to BF16)

```cpp
// BF16 keeps the same exponent as FP32 but truncates mantissa to 7 bits
template <>
struct NumericConverter<float, cutlass::bfloat16_t, FloatRoundStyle::round_indeterminate> {
    CUTLASS_HOST_DEVICE
    cutlass::bfloat16_t operator()(float const &source) const {
        // BF16 is the upper 16 bits of FP32
        uint32_t bits = reinterpret_cast<uint32_t const &>(source);
        return cutlass::bfloat16_t(reinterpret_cast<cutlass::bfloat16_t::storage_type const &>(bits >> 16));
    }
};

// Rounding BF16 conversion
template <>
struct NumericConverter<float, cutlass::bfloat16_t, FloatRoundStyle::round_to_nearest> {
    CUTLASS_HOST_DEVICE
    cutlass::bfloat16_t operator()(float const &source) const {
        uint32_t bits = reinterpret_cast<uint32_t const &>(source);
        // Add rounding bias
        bits += 0x8000;  // Add 0.5 ULP of BF16
        return cutlass::bfloat16_t(reinterpret_cast<cutlass::bfloat16_t::storage_type const &>(bits >> 16));
    }
};
```

#### Float to TF32

```cpp
// TF32 keeps FP32 range but with 10-bit mantissa (19 bits total)
template <>
struct NumericConverter<float, cutlass::tfloat32_t, FloatRoundStyle::round_indeterminate> {
    CUTLASS_HOST_DEVICE
    cutlass::tfloat32_t operator()(float const &source) const {
        uint32_t bits = reinterpret_cast<uint32_t const &>(source);
        // TF32: 1 sign + 8 exponent + 10 mantissa = 19 bits
        // Mask out lower 13 mantissa bits of FP32
        bits &= 0xFFFFE000;
        return cutlass::tfloat32_t(reinterpret_cast<float const &>(bits));
    }
};

// Rounding TF32 conversion
template <>
struct NumericConverter<float, cutlass::tfloat32_t, FloatRoundStyle::round_half_ulp_truncate> {
    CUTLASS_HOST_DEVICE
    cutlass::tfloat32_t operator()(float const &source) const {
        uint32_t bits = reinterpret_cast<uint32_t const &>(source);
        // Add half-ULP of TF32 before truncating
        bits += 0x1000;  // 0.5 ULP at TF32 precision
        bits &= 0xFFFFE000;
        return cutlass::tfloat32_t(reinterpret_cast<float const &>(bits));
    }
};
```

#### Float to FP8 (E4M3 and E5M2)

```cpp
// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits
// Range: up to 448, no infinity, NaN = 0x7F
cutlass::NumericConverter<float, cutlass::float_e4m3_t> conv_e4m3;
cutlass::float_e4m3_t fp8_val = conv_e4m3(1.5f);

// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa = 8 bits
// Range: up to 57344, supports infinity and NaN
cutlass::NumericConverter<float, cutlass::float_e5m2_t> conv_e5m2;
cutlass::float_e5m2_t fp8_val2 = conv_e5m2(1.5f);

// Vectorized FP8 conversion (8 at a time)
using Fp32ToFp8Array = cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, 8>;
Fp32ToFp8Array array_conv;
cutlass::Array<float, 8> fp32_arr = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
cutlass::Array<cutlass::float_e4m3_t, 8> fp8_arr = array_conv(fp32_arr);
```

#### Integer Conversions

```cpp
// Float to INT8 with rounding
cutlass::NumericConverter<float, int8_t, cutlass::FloatRoundStyle::round_to_nearest> to_int8;
int8_t q = to_int8(2.7f);  // Returns 3

// Float to INT4 with clamping
cutlass::NumericConverter<float, cutlass::int4b_t, cutlass::FloatRoundStyle::round_toward_zero> to_int4;
cutlass::int4b_t q4 = to_int4(5.3f);  // Clamped to 7 (max int4)

// INT8 to Float (widening, no rounding)
cutlass::NumericConverter<int8_t, float> int8_to_float;
float val = int8_to_float(q);  // Exact conversion: 3.0f
```

---

## 37.6 Type Traits

### 37.6.1 has_negative_zero

```cpp
// Trait indicating whether a type has a distinct negative zero representation
template <typename T>
struct has_negative_zero;

// Floating-point types: true (IEEE 754 -0.0)
template <> struct has_negative_zero<float> { static bool const value = true; };
template <> struct has_negative_zero<double> { static bool const value = true; };
template <> struct has_negative_zero<cutlass::half_t> { static bool const value = true; };
template <> struct has_negative_zero<cutlass::bfloat16_t> { static bool const value = true; };

// Integer types: false (no -0)
template <> struct has_negative_zero<int8_t> { static bool const value = false; };
template <> struct has_negative_zero<int32_t> { static bool const value = false; };

// Usage: important for correctly handling sign bit in reductions
// When has_negative_zero<T>::value is true, need special handling
// to ensure -0 and +0 compare equal
```

### 37.6.2 get_unpacked_element_type

```cpp
// Trait for getting the unpacked (logical) element type from a storage type
// For sub-byte types, returns the logical type (e.g., int4b_t)
// For regular types, returns the type itself

template <typename T>
struct get_unpacked_element_type {
    using type = T;  // Identity for regular types
};

// For sub-byte types referenced through SubbyteReference:
template <typename T>
struct get_unpacked_element_type<cutlass::SubbyteReference<T>> {
    using type = T;  // Unwrap to the logical element type
};

// Usage:
using UnpackedA = typename cutlass::get_unpacked_element_type<ElementA>::type;
// For int4b_t: UnpackedA = int4b_t
// For SubbyteReference<int4b_t>: UnpackedA = int4b_t
// For half_t: UnpackedA = half_t
```

### 37.6.3 Register Type Extraction

```cpp
// Determines the register-level type used to hold elements during computation
// For most types, this is the element type itself
// For sub-byte types, this is a wider packed type

template <typename Element>
struct RegisterType {
    using type = Element;
};

// For 4-bit types, registers hold packed pairs
template <>
struct RegisterType<int4b_t> {
    using type = uint8_t;  // Two int4b_t packed into uint8_t
};

template <>
struct RegisterType<uint4b_t> {
    using type = uint8_t;
};

// For 1-bit types, registers hold packed groups
template <>
struct RegisterType<cutlass::bin1_t> {
    using type = uint32_t;  // 32 binary values packed into uint32_t
};

// Usage in kernel implementation:
using RegType = typename cutlass::RegisterType<ElementA>::type;
// For int4b_t: RegType = uint8_t
// For half_t: RegType = half_t
// For float: RegType = float
```

### 37.6.4 sizeof_bits

```cpp
// Returns the number of bits required to store a type
template <typename T>
struct sizeof_bits {
    static constexpr int value = sizeof(T) * 8;
};

// Specializations for sub-byte types:
template <>
struct sizeof_bits<int4b_t> {
    static constexpr int value = 4;
};

template <>
struct sizeof_bits<uint4b_t> {
    static constexpr int value = 4;
};

template <>
struct sizeof_bits<int2b_t> {
    static constexpr int value = 2;
};

template <>
struct sizeof_bits<uint2b_t> {
    static constexpr int value = 2;
};

template <>
struct sizeof_bits<cutlass::bin1_t> {
    static constexpr int value = 1;
};

// Usage:
static_assert(cutlass::sizeof_bits<cutlass::half_t>::value == 16, "");
static_assert(cutlass::sizeof_bits<int4b_t>::value == 4, "");
static_assert(cutlass::sizeof_bits<float>::value == 32, "");

// Compute alignment for Tensor Core operations:
int alignment = 128 / cutlass::sizeof_bits<ElementA>::value;
// For FP16: 128 / 16 = 8 elements
// For INT8: 128 / 8 = 16 elements
// For INT4: 128 / 4 = 32 elements
```

---

## 37.7 Platform-Specific Optimizations

### 37.7.1 Hardware Conversion Instructions

CUTLASS leverages hardware conversion instructions when available:

```cpp
// __half2float: FP16 -> FP32 (SM53+)
// __float2half: FP32 -> FP16 (SM53+)
// __float2bf16_rn: FP32 -> BF16 (SM80+)
// __nv_fp8_e4m3_to_fp32: FP8 E4M3 -> FP32 (SM90+)
// __nv_fp8_e5m2_to_fp32: FP8 E5M2 -> FP32 (SM90+)

// The NumericConverter specializations use these instructions automatically:
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Use hardware BF16 conversion on Ampere+
    return cutlass::bfloat16_t(__float2bfloat16_rn(source));
#else
    // Software fallback for older architectures
    uint32_t bits = reinterpret_cast<uint32_t const &>(source);
    bits = (bits + 0x8000) & 0xFFFF0000;
    return cutlass::bfloat16_t::bitcast(uint16_t(bits >> 16));
#endif
```

### 37.7.2 Vectorized Array Conversion

```cpp
// NumericArrayConverter uses SIMD instructions for batch conversion:
// FP32x4 -> FP16x4: uses __float2half2_rn on SM53+
// FP32x8 -> FP16x8: uses two __float2half2_rn calls
// FP32x4 -> BF16x4: uses __float2bfloat162_rn on SM80+

// Example: Array<float, 4> -> Array<half_t, 4>
template <>
struct NumericArrayConverter<float, cutlass::half_t, 4, FloatRoundStyle::round_to_nearest> {
    CUTLASS_HOST_DEVICE
    cutlass::Array<cutlass::half_t, 4> operator()(
        cutlass::Array<float, 4> const &source
    ) const {
        cutlass::Array<cutlass::half_t, 4> result;

        // Use half2 conversion for pairs
        __half2 h2_0 = __float22half2_rn(source[0], source[1]);
        __half2 h2_1 = __float22half2_rn(source[2], source[3]);

        result[0] = cutlass::half_t(h2_0.x);
        result[1] = cutlass::half_t(h2_0.y);
        result[2] = cutlass::half_t(h2_1.x);
        result[3] = cutlass::half_t(h2_1.y);

        return result;
    }
};
```

---

## 37.8 Code Examples

### 37.8.1 Quantized GEMM with INT4 Weights

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/subbyte_reference.h"
#include "cutlass/numeric_conversion.h"

// GEMM with INT4 weights packed in memory
// A: FP16 [M, K], B: INT4 [K, N] (packed), C: FP16 [M, N]

// Step 1: Allocate packed storage for INT4 weights
int K = 512, N = 256;
size_t packed_B_size = (K * N + 1) / 2;  // 2 elements per byte
uint8_t* packed_B;
cudaMalloc(&packed_B, packed_B_size);

// Step 2: Use SubbyteReference for accessing packed elements
// In host code (for packing):
for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; n += 2) {
        int4b_t val0 = quantize(weight[k * N + n]);
        int4b_t val1 = quantize(weight[k * N + n + 1]);
        packed_B[(k * N + n) / 2] = (val0 & 0xF) | ((val1 & 0xF) << 4);
    }
}

// Step 3: In CUTLASS kernel, SubbyteReference handles element access
cutlass::SubbyteReference<int4b_t> B_ref(packed_B, k * N + n);
int4b_t weight_val = B_ref;
```

### 37.8.2 Mixed-Precision Conversion Pipeline

```cpp
#include "cutlass/numeric_conversion.h"

// Typical mixed-precision GEMM pipeline:
// 1. Load FP16 inputs
// 2. Convert to TF32 for Tensor Core multiplication
// 3. Accumulate in FP32
// 4. Convert output to FP16

// Converters for each stage:
using LoadConverterA = cutlass::NumericConverter<cutlass::half_t, cutlass::tfloat32_t>;
using LoadConverterB = cutlass::NumericConverter<cutlass::half_t, cutlass::tfloat32_t>;
using OutputConverter = cutlass::NumericConverter<float, cutlass::half_t>;

// In the kernel mainloop:
LoadConverterA load_a;
LoadConverterB load_b;
OutputConverter output_conv;

// Load FP16 -> TF32
cutlass::tfloat32_t a_tf32 = load_a(a_fp16);
cutlass::tfloat32_t b_tf32 = load_b(b_fp16);

// Accumulate in FP32 (TF32 * TF32 -> FP32 via Tensor Core)
accumulator = a_tf32 * b_tf32 + accumulator;  // Uses TF32 Tensor Core

// Convert FP32 -> FP16 output
cutlass::half_t d_fp16 = output_conv(accumulator);
```

### 37.8.3 Subbyte Element Access Pattern

```cpp
// Kernel that processes sub-byte elements in a packed array
template <typename SubbyteElement>
__global__ void process_subbyte(
    uint8_t* __restrict__ output,
    uint8_t const* __restrict__ input,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Create subbyte reference
    cutlass::ConstSubbyteReference<SubbyteElement> in_ref(input, idx);
    cutlass::SubbyteReference<SubbyteElement> out_ref(output, idx);

    // Read the sub-byte element
    SubbyteElement val = in_ref;

    // Process (example: negate for signed types)
    // For int4b_t: val = -val
    // For uint4b_t: val = val + 1

    // Write back
    out_ref = val;
}

// Launch:
int num_elements = 1024;
int block_size = 256;
int grid_size = (num_elements + block_size - 1) / block_size;
process_subbyte<int4b_t><<<grid_size, block_size>>>(output, input, num_elements);
```

---

## 37.9 Summary Table

| Component | Purpose | Key Types |
|---|---|---|
| `SubbyteReference` | Read/write access to sub-byte elements | `int4b_t`, `uint4b_t`, `int2b_t`, `uint2b_t`, `bin1_t` |
| `ConstSubbyteReference` | Read-only access to sub-byte elements | Same as above |
| `NumericConverter` | Single-element type conversion | All CUTLASS numeric types |
| `NumericArrayConverter` | Vectorized array conversion | `Array<T, N>` types |
| `sizeof_bits` | Bit-width of a type | Any CUTLASS type |
| `has_negative_zero` | Whether type has -0.0 | Floating-point types |
| `get_unpacked_element_type` | Logical element type from storage | SubbyteReference types |
| `RegisterType` | Register-level storage type | Sub-byte types |
