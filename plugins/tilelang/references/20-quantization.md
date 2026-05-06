# TileLang Quantization Reference

## 1. Overview

TileLang provides comprehensive support for quantized operations, enabling efficient
execution of low-precision matrix multiplications on GPU hardware. The quantization
system covers data type conversions between packed low-precision formats and compute
formats, hardware-accelerated dequantization using LOP3 intrinsics, MXFP (Mixed-Precision
Floating Point) support, and utility functions for weight packing and interleaving.

### Supported Quantization Formats

| Format | Bits | Type | Hardware Support |
|---|---|---|---|
| INT4 | 4 | Integer | CUDA, ROCm |
| INT8 | 8 | Integer | CUDA, ROCm |
| UINT4 | 4 | Unsigned integer | CUDA |
| UINT8 | 8 | Unsigned integer | CUDA, ROCm |
| FP8 E4M3 | 8 | Floating point | CUDA SM90+, ROCm gfx940+ |
| FP8 E5M2 | 8 | Floating point | CUDA SM90+, ROCm gfx940+ |
| FP4 E2M1 | 4 | Floating point | CUDA SM100+ |
| FP6 E2M3 | 6 | Floating point | CUDA (experimental) |
| FP6 E3M2 | 6 | Floating point | CUDA (experimental) |
| MXFP4 | 4 | Block-scaled float | CUDA SM90+ |
| INT2 | 2 | Integer | CUDA |
| INT1 | 1 | Integer | CUDA |

---

## 2. Core Conversion Functions

### 2.1 _tir_packed_int_to_int_convert

Converts packed signed integers to their sign-extended form.

```python
from tilelang.quantize.quantization import _tir_packed_int_to_int_convert

# Create a converter for 8-bit storage
convert = _tir_packed_int_to_int_convert(storage_type="uint", storage_nbit=8)

# Use in TileLang kernel
# val: uint8 containing packed 4-bit signed integers
# pos: which 4-bit field to extract (0 or 1)
# dtype: target data type (e.g., "float16")
result = convert(nbit=4, val=val, pos=pos, dtype="float16")
```

Implementation:

```python
def _tir_packed_int_to_int_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)
    def f_convert(nbit, val, pos, dtype):
        mask = tir.const((1 << nbit) - 1, T.int32)
        unextended = (val >> (pos.astype(T.int32) * tir.const(nbit, T.int32))) & mask
        # Sign extend: shift left by (32 - nbit), then arithmetic shift right
        return tir.Cast(
            dtype,
            (unextended << tir.const(32 - nbit, T.int32)) >> tir.const(32 - nbit, T.int32))
    return f_convert
```

The sign extension works by:
1. Extracting the n-bit field from the packed storage
2. Shifting left to position the sign bit at bit 31
3. Performing an arithmetic right shift to propagate the sign bit

### 2.2 _tir_packed_to_signed_convert

Converts packed values to signed representation by subtracting the offset.

```python
from tilelang.quantize.quantization import _tir_packed_to_signed_convert

convert = _tir_packed_to_signed_convert(storage_type="uint", storage_nbit=8)
result = convert(nbit=4, val=val, pos=pos, dtype="float16")
```

Implementation:

```python
def _tir_packed_to_signed_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)
    def f_convert(nbit, val, pos, dtype):
        max_int_value = (1 << (nbit - 1))
        return ((val >> (pos.astype(T.uint32) * tir.const(nbit, T.uint32))) &
                tir.const((1 << nbit) - 1, "uint32")).astype(dtype) - tir.const(max_int_value, dtype)
    return f_convert
```

For 4-bit values:
- Unsigned range: 0 to 15
- Signed offset: 8 (2^(nbit-1))
- Signed range after conversion: -8 to +7

### 2.3 _tir_packed_to_unsigned_convert

Extracts packed unsigned values without modification.

```python
from tilelang.quantize.quantization import _tir_packed_to_unsigned_convert

convert = _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8)
result = convert(nbit=4, val=val, pos=pos, dtype="float16")
```

Implementation:

```python
def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)
    def f_convert(nbit, val, pos, dtype):
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        return ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(dtype)
    return f_convert
```

### 2.4 _tir_packed_to_unsigned_convert_with_zeros

Extracts packed unsigned values and subtracts a zero point.

```python
from tilelang.quantize.quantization import _tir_packed_to_unsigned_convert_with_zeros

convert = _tir_packed_to_unsigned_convert_with_zeros(storage_type="uint", storage_nbit=8)
result = convert(nbit=4, val=val, pos=pos, zero=zero_point, dtype="float16")
```

Implementation:

```python
def _tir_packed_to_unsigned_convert_with_zeros(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)
    def f_convert(nbit, val, pos, zero, dtype):
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        return (((val >> (pos * nbit).astype(storage_dtype)) & mask) - zero).astype(dtype)
    return f_convert
```

This implements the standard quantization formula:

```
dequantized = (quantized - zero_point) * scale
```

---

## 3. Floating Point Conversion Functions

### 3.1 _tir_packed_to_fp4_to_f16

Converts packed FP4 (E2M1) values to FP16.

```python
from tilelang.quantize.quantization import _tir_packed_to_fp4_to_f16

# Create a converter for uint32 storage
convert = _tir_packed_to_fp4_to_f16(storage_type="uint", storage_nbit=8)
result = convert(nbit=4, val=val, pos=pos, dtype="float16")
```

FP4 (E2M1) format:
- 1 sign bit
- 2 exponent bits (bias = 2)
- 1 mantissa bit

Implementation (for uint32 storage):

```python
def _tir_packed_to_fp4_to_f16(nbit, val, pos, dtype):
    assert nbit == 4
    assert dtype == T.float16
    assert val.dtype == T.uint32

    mask = tvm.tir.const((1 << nbit) - 1, T.uint16)
    f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
    s = f4 >> tir.const(3, T.uint16)        # sign bit
    e_f4 = f4 & tir.const(7, T.uint16)      # exponent + mantissa

    # FP4 exponent bias = 2, FP16 exponent bias = 15
    # e_f4 != 0 -> e_f16 = e_f4 + (15 - 2) = e_f4 | (1000)_2 = e_f4 | 8
    e_f16 = e_f4 | tir.const(8, T.uint16)

    val_f16 = tir.reinterpret(T.float16,
        ((e_f16 | (s << tir.const(5, T.uint16))) << tir.const(10, T.uint16)).astype(T.uint16))

    # Zero handling: e_f4 == 0 means zero
    return tir.Select(e_f4 == tir.const(0, T.uint16), tir.const(0, T.float16), val_f16)
```

The conversion maps FP4 values to FP16:

| FP4 Value | FP16 Value |
|---|---|
| 0b0000 | +0 |
| 0b0001 | +2^-2 * 1.0 |
| 0b0010 | +2^-1 * 1.0 |
| 0b0011 | +2^-1 * 1.5 |
| 0b0100 | +2^0 * 1.0 |
| 0b0101 | +2^0 * 1.5 |
| 0b0110 | +2^1 * 1.0 |
| 0b0111 | +2^1 * 1.5 |
| 1xxx | (negative versions) |

### 3.2 _tir_u8_to_f8_e4m3_to_f16

Converts FP8 (E4M3) to FP16.

```python
from tilelang.quantize.quantization import _tir_u8_to_f8_e4m3_to_f16

result = _tir_u8_to_f8_e4m3_to_f16(nbit=8, val=val, dtype="float16")
```

FP8 (E4M3) format:
- 1 sign bit
- 4 exponent bits (bias = 7)
- 3 mantissa bits

Implementation:

```python
def _tir_u8_to_f8_e4m3_to_f16(nbit, val, dtype):
    assert nbit == 8
    assert dtype == T.float16
    s_f16 = (val >> tir.const(7, T.uint16)) << tir.const(15, T.uint16)
    e4 = val & tir.const(0x40, T.uint16)
    prefix = tir.Select(e4 == tir.const(0, T.uint16),
                        tir.const(0x2000, T.uint16),
                        tir.const(0x4000, T.uint16))
    e_f16 = ((val & tir.const(63, T.uint16)) << tir.const(7, T.uint16)) | prefix
    e_f16 = e_f16 ^ tir.const(0x2000, T.uint16)
    return tir.reinterpret(T.float16, s_f16 | e_f16)
```

The conversion handles the FP8 E4M3 bias of 7 and maps to FP16 bias of 15.

### 3.3 _tir_u8_to_f8_e5m2_to_f16

Converts FP8 (E5M2) to FP16.

```python
from tilelang.quantize.quantization import _tir_u8_to_f8_e5m2_to_f16

result = _tir_u8_to_f8_e5m2_to_f16(nbit=8, val=val, dtype="float16")
```

Implementation:

```python
def _tir_u8_to_f8_e5m2_to_f16(nbit, val, dtype):
    assert nbit == 8
    assert dtype == T.float16
    return tir.reinterpret("float8_e5m2", val).astype(T.float16)
```

FP8 E5M2 has the same exponent width as FP16, making the conversion simpler
(only mantissa bits differ).

### 3.4 _tir_u8_to_f4_to_bf16

Converts FP4 (E2M1) to BFloat16.

```python
from tilelang.quantize.quantization import _tir_u8_to_f4_to_bf16

result = _tir_u8_to_f4_to_bf16(nbit=4, val=val, pos=pos, scale=scale, dtype="bfloat16")
```

Implementation:

```python
def _tir_u8_to_f4_to_bf16(nbit, val, pos, scale, dtype):
    assert nbit == 4
    assert dtype == T.bfloat16
    assert val.dtype == T.uint8

    mask = tir.const((1 << nbit) - 1, T.uint16)
    f4 = (val >> (pos.astype(T.uint16) * tir.const(nbit, T.uint16))) & mask
    s = f4 >> tir.const(3, T.uint16)
    e_f4 = (f4 & tir.const(6, T.uint16)) >> tir.const(1, T.uint16)

    # Bias difference: BF16 bias = 127, FP4 bias = 1 -> diff = 126
    e_bf16 = e_f4 + tir.const(126, T.uint16)
    e_bf16 = min(e_bf16 + scale, tir.const((1 << 8) - 1, T.uint16))

    m_f4 = f4 & tir.const(1, T.uint16)
    val_bf16 = tir.reinterpret(T.bfloat16,
        ((((s << tir.const(8, T.uint16)) | e_bf16) << tir.const(7, T.uint16))
         | (m_f4 << tir.const(6, T.uint16))).astype(T.uint16))
    return val_bf16
```

This function supports an additional `scale` parameter for exponent scaling.

### 3.5 _tir_u32_to_f4_to_f32

Converts FP4 (E2M1) to FP32.

```python
from tilelang.quantize.quantization import _tir_u32_to_f4_to_f32

result = _tir_u32_to_f4_to_f32(nbit=4, val=val, pos=pos, dtype="float32")
```

FP4 to FP32 conversion uses the bias difference of 120 (FP32 bias = 127, FP4 bias = 2,
but implementation uses 120 for the OR trick):

```python
# e_f4 == 0 -> e_f32 = 0
# e_f4 != 0 -> e_f32 = e_f4 + 120 = e_f4 | (1111000)_2
e_f32 = e_f4 | tir.const(120, T.uint32)
```

---

## 4. Float-to-Float Packing Functions

### 4.1 _tir_f32_to_uint_to_f4

Converts FP32 to packed FP4 format.

```python
from tilelang.quantize.quantization import _tir_f32_to_uint_to_f4

result = _tir_f32_to_uint_to_f4(val=fp32_value)
```

Implementation:

```python
def _tir_f32_to_uint_to_f4(val):
    assert val.dtype == T.float32
    val_u32 = tir.reinterpret(T.uint32, val)
    m_h = (val_u32 >> tir.const(22, T.uint32)) & tir.const(1, T.uint32)
    e_f32 = (val_u32 >> tir.const(23, T.uint32)) & tir.const(255, T.uint32)
    s = (val_u32 >> tir.const(31, T.uint32))

    e_f4 = tir.Select(
        e_f32 > tir.const(120, T.uint32),
        tir.Min(e_f32 - tir.const(120, T.uint32) + m_h, tir.const(7, T.uint32)),
        tir.Select(e_f32 == tir.const(120, T.uint32),
                   tir.const(1, T.uint32),
                   tir.const(0, T.uint32)))
    return (s << tir.const(3, T.uint32)) | e_f4
```

The rounding uses round-to-nearest-even by including the hidden mantissa bit.

### 4.2 _tir_f16_to_uint_to_f4

Converts FP16 to packed FP4 format.

```python
from tilelang.quantize.quantization import _tir_f16_to_uint_to_f4

result = _tir_f16_to_uint_to_f4(val=fp16_value)
```

Similar to the FP32 version but with FP16 bias handling.

---

## 5. BF16 Packing Utilities

### 5.1 _tir_f32x2_to_bf16x2_to_u32

Packs two FP32 values into a single uint32 as two BF16 values:

```python
from tilelang.quantize.quantization import _tir_f32x2_to_bf16x2_to_u32

packed = _tir_f32x2_to_bf16x2_to_u32(v0=fp32_a, v1=fp32_b, round_to_even=True)
```

Implementation:

```python
def _tir_f32x2_to_bf16x2_to_u32(v0, v1, round_to_even=True):
    mask = tir.const((1 << 16) - 1, T.uint32)
    res = []
    for data in [v0, v1]:
        u32_val = tir.reinterpret(T.uint32, data)
        if round_to_even:
            rounding_bias = ((u32_val >> tir.const(16, T.uint32)) &
                            tir.const(1, T.uint32)) + tir.const(0x7FFF, T.uint32)
            u32_val += rounding_bias
        res.append((u32_val >> tir.const(16, T.uint32)) & mask)
    return res[0] | (res[1] << tir.const(16, T.uint32))
```

### 5.2 _tir_u32_to_bf16x2_to_f32x2

Unpacks a uint32 containing two BF16 values into two FP32 values:

```python
from tilelang.quantize.quantization import _tir_u32_to_bf16x2_to_f32x2

f32_a, f32_b = _tir_u32_to_bf16x2_to_f32x2(packed_u32)
```

---

## 6. MXFP (Mixed-Precision Floating Point)

### 6.1 Overview

MXFP (Microscaling Formats) is a block-scaled floating point format where groups
of elements share a common scale factor. TileLang supports MXFP4 for efficient
quantized GEMM operations.

### 6.2 get_mxfp_intrin_group

```python
from tilelang.quantize.mxfp import get_mxfp_intrin_group

intrin = get_mxfp_intrin_group(
    out_dtype=T.bfloat16,     # Output data type
    source_format=T.uint,     # Source format (int or uint)
    source_bit=4,             # Source bit width
    storage_dtype=T.uint8,    # Storage data type
    use_twiddling=False,      # Use bit-twiddling optimization
)
```

Parameters:
- `out_dtype`: Target floating-point type for decoded values (T.float16 or T.bfloat16)
- `source_format`: Integer source representation ("int" or "uint")
- `source_bit`: Bit width of the packed source format (e.g., 4)
- `storage_dtype`: Underlying storage integer dtype (T.int32, T.int8, or T.uint8)
- `use_twiddling`: When True, use the optimized bit-twiddling variant

Returns:
```python
{
    "func_name": "decode_fp4_to_bf16",      # C function name
    "c_source": "/* CUDA device function */", # C source code
}
```

### 6.3 FP4 to BF16 Twiddling Optimization

The twiddling variant uses PTX bitwise operations for efficient FP4-to-BF16
conversion. From `tilelang/quantize/mxfp.py`:

```cuda
template<typename T1, typename T2>
__device__ void decode_fp4_to_bf16_twiddling(T1 *B_local, T2 *B_local_decode,
                                              const int N = 8) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        uint B_dequantize_local_vec[4];
        uint tmp, bias, d0, d1, d2, d3, d4, d5, d6;
        asm volatile(
            "prmt.b32 %13, %4, 0, 0x0123;"
            "mov.b32 %12, 0x7e807e80;"
            "and.b32 %0, %13, 0b10000001110000001000000111000000;"
            "mul.bf16x2 %0, %0, %12;"
            "shl.b32 %1, %13, 3;"
            "and.b32 %1, %1, 0b10000001110000001000000111000000;"
            "mul.bf16x2 %1, %1, %12;"
            // ... more operations
        );
    }
}
```

This approach converts 8 FP4 values at once using `bf16x2` multiply operations,
which is significantly faster than scalar conversion.

### 6.4 MXFP Usage Example

```python
import tilelang as tl
from tilelang import language as T
from tilelang.quantize.mxfp import get_mxfp_intrin_group

# Get the MXFP dequantization intrinsic
intrin = get_mxfp_intrin_group(
    out_dtype=T.bfloat16,
    source_format=T.uint,
    source_bit=4,
    storage_dtype=T.uint8,
    use_twiddling=True,
)

@T.prim_func
def mxfp4_gemm(
    A: T.Buffer((M, K), "bfloat16"),
    B_packed: T.Buffer((K // 2, N), "uint8"),  # Packed FP4
    scale: T.Buffer((K // 32, N), "bfloat16"), # Per-block scale
    C: T.Buffer((M, N), "float32"),
):
    # Dequantize FP4 weights to BF16
    # Use the intrinsic function
    T.import_c_source(intrin["c_source"])

    with T.Kernel(...) as (bx, by):
        B_local_decode = T.alloc_fragment((BLOCK_K, BLOCK_N), "bfloat16")
        # Call the imported dequantization function
        T.call_extern(intrin["func_name"], B_packed_ptr, B_local_decode_ptr, N)
        T.gemm(A_shared, B_local_decode, C_frag)
```

---

## 7. LOP3 Intrinsics for Quantization

### 7.1 Overview

LOP3 (Three-Operand Logic) is a CUDA instruction that performs arbitrary bitwise
logical operations on three inputs in a single instruction. TileLang uses LOP3
for efficient dequantization of INT4, INT2, and INT1 values to FP16.

### 7.2 get_lop3_intrin_group

```python
from tilelang.quantize.lop3 import get_lop3_intrin_group

intrin = get_lop3_intrin_group(
    out_dtype=T.float16,
    source_format=T.uint,
    source_bit=4,
    storage_dtype=T.int8,
    with_scaling=False,
    with_zeros=False,
    zeros_mode="original",
    storage_scope="local",
)
```

Parameters:
- `out_dtype`: Output data type (T.float16, T.int8, or T.int4)
- `source_format`: Source integer format (T.int or T.uint)
- `source_bit`: Bit width of packed values (1, 2, or 4)
- `storage_dtype`: Storage type (T.int32 or T.int8)
- `with_scaling`: Whether to apply scaling factor
- `with_zeros`: Whether to subtract zero points
- `zeros_mode`: "original", "rescale", or "quantized"
- `storage_scope`: "local" or "warp" (affects offset handling)

Returns:
```python
{
    "func_name": "decode_i4u_to_f16",
    "c_source": "/* CUDA device function with LOP3 */",
}
```

### 7.3 INT4 to FP16 Dequantization

The LOP3-based INT4 to FP16 dequantization works in two steps:

1. **LOP3 extraction:** Extract 4-bit values and combine with a magic number
   using LOP3 to create intermediate FP16 bit patterns
2. **Subtraction:** Subtract a bias value to get the final FP16 result

```cuda
// Step 1: LOP3 extracts 4-bit fields and combines with FP16 magic number
static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;  // = 0xE8 = 0b11101000
static constexpr uint BOTTOM_MASK = 0x000f000f;
static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;

asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
             : "=r"(h[i])
             : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK),
               "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));

// Step 2: Subtract bias to get correct FP16 value
// For unsigned: 0x64006400 (bias = 1024 in FP16)
// For signed: 0x64086408 (bias = 1032 in FP16)
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
```

The LOP3 lookup table `0xE8` computes `A AND B OR C` which:
1. Masks the 4-bit values with `0x000f000f` (bottom 4 bits of each 16-bit half)
2. ORs with `0x64006400` to create valid FP16 numbers around 1024.0
3. The subtraction then maps 0..15 to -8..+7 (signed) or 0..15 (unsigned)

### 7.4 INT4 with Scaling

```cuda
// After extraction and bias subtraction:
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
// Apply scale factor
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
             : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
```

### 7.5 INT4 with Zero Points

Three modes of zero-point application:

#### Original: `target = (dequantize_weight - zero_point) * scale`

```cuda
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_zeros));
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
```

#### Rescale: `target = dequantize_weight * scale - zero_point`

```cuda
uint const packed_zeros = 0x80008000 | __pack_half2(zero_r, zero_r);
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
             : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(packed_zeros));
```

#### Quantized: `target = (dequantize_weight - dequantize_zeros) * scale`

```cuda
int16_t const zero_r = *((int16_t*)zeros);
uint median_num = ((0xe400 | zero_r) << 16) | (0xe400 | zero_r);
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" ...);
asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(median_num));
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales), "r"(0));
```

### 7.6 INT2 to FP16 Dequantization

Similar to INT4 but with different masks:

```cuda
static constexpr uint BOTTOM_MASK = 0x00030003;  // 2-bit mask
static constexpr uint MEDIAN_NUM = isSigned ? 0x64026402 : 0x64006400;
```

INT2 uses 2-bit extraction with the same LOP3 pattern. The input is first
interleaved to separate even and odd bits:

```cuda
int16_t const i2s_i16 = *reinterpret_cast<int16_t *>(_i2s);
int i2s = (i2s_i16 & 0x00ff);
i2s |= ((i2s_i16 & 0xff00) << 8);
```

### 7.7 INT1 to FP16 Dequantization

1-bit quantization with 1-bit mask:

```cuda
static constexpr uint BOTTOM_MASK = 0x00010001;  // 1-bit mask
static constexpr uint MEDIAN_NUM = 0x64006400;
```

For signed INT1, an additional transform converts 0/1 to -1/+1:

```cuda
static constexpr uint TRANSFORM_SUBTRACT = 0xbc00bc00; // for 2x - 1
asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(h[i]));     // 2x
asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(TRANSFORM_SUBTRACT)); // -1
```

### 7.8 INT4 to INT8 Conversion

For converting between integer precisions using LOP3:

```cuda
static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;
static constexpr uint I4b_TO_I8s_MAGIC_NUM = 0x00000000;
static constexpr uint MEDIAN_NUM = 0x07070707;

asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(i8s[i])
             : "r"(i4b[0] >> (4 * i)), "n"(BOTTOM_MASK),
               "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));
i8s[i] = __vsubss4(i8s[i], MEDIAN_NUM);  // Vector subtract with saturation
```

### 7.9 INT2 to INT4 Conversion

```cuda
static constexpr uint BOTTOM_MASK = 0x33333333;
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(i4s[i])
             : "r"(i2b[i / 2] >> (2 * (i % 2))), "n"(BOTTOM_MASK),
               "n"(I4b_TO_I8s_MAGIC_NUM), "n"(immLut));
```

### 7.10 Offset Variants

Several functions support split-scale/zero-point with offset for scenarios where
different groups within the same warp use different scale factors:

```cuda
// With offset for left/right halves
T3 const scale_l = *scale;
T3 const scale_r = *(scale + offset);
uint const packed_scales_l = __pack_half2(scale_l, scale_l);
uint const packed_scales_r = __pack_half2(scale_r, scale_r);

// First half uses scale_l
for (int i = 0; i < (N / 4); i++) {
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_l), "r"(0));
}
// Second half uses scale_r
for (int i = (N / 4); i < (N / 2); i++) {
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[i]) : "r"(h[i]), "r"(packed_scales_r), "r"(0));
}
```

---

## 8. Utility Functions

### 8.1 gen_quant4

Generates quantized INT4 weights for testing:

```python
from tilelang.quantize.utils import gen_quant4

original_w, linear, scales, quantized_w = gen_quant4(
    k=4096,           # Input dimension
    n=4096,           # Output dimension
    groupsize=-1,     # Group size for per-group quantization (-1 = per-channel)
)
```

The function:
1. Generates random float16 weights
2. Computes per-group or per-channel scales
3. Quantizes to INT4 (unsigned storage)
4. Returns the original weights, a reference linear layer, scales, and quantized weights

### 8.2 general_compress

Compresses low-precision weights into packed storage format:

```python
from tilelang.quantize.utils import general_compress

compressed = general_compress(
    lowprecision_weight,  # Tensor of quantized values
    source_bits=4,        # Number of bits per element
    storage_dtype=None,   # Output dtype (default: torch.int8)
)
```

Implementation:

```python
def general_compress(lowprecision_weight, source_bits=4, storage_dtype=None):
    elems_per_byte = 8 // source_bits
    int8_weight = torch.zeros(
        (*lowprecision_weight.shape[:-1],
         lowprecision_weight.shape[-1] // elems_per_byte),
        dtype=torch.int8,
    )
    for j in range(lowprecision_weight.shape[-1] // elems_per_byte):
        for k in range(elems_per_byte):
            int8_weight[..., j] |= (
                lowprecision_weight[..., j * elems_per_byte + k] << (source_bits * k)
            ).to(torch.int8)
    return int8_weight.to(storage_dtype)
```

For INT4 with int8 storage: each byte stores 2 INT4 values.

### 8.3 interleave_weight

Reorders quantized weights for efficient hardware dequantization:

```python
from tilelang.quantize.utils import interleave_weight

interleaved = interleave_weight(
    qweight,            # Quantized weight tensor
    nbits=4,            # Bits per element
    target_dtype="float16",  # Target dtype for interleaving
)
```

The interleaving reorders elements so that the LOP3-based dequantization
can process multiple elements simultaneously:

```python
# For INT4 to FP16:
bits_stride = 16  # 16 bits = 1 FP16 element
mask = (1 << 4) - 1
num_groups = 32 // 16    # = 2
elems_per_group = 16 // 4  # = 4

for i in range(num_groups):
    for j in range(elems_per_group):
        offset = i * elems_per_group + j
        shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
        new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift
```

Special cases for different bit widths and target types:

- **INT4 to FP16:** Standard interleaving with 16-bit stride
- **INT2 to FP16:** Special reordering with byte-level shifts
- **INT1 to FP16:** Complex bit manipulation with 4-bit group handling
- **INT1 to INT8:** Different interleaving for integer output

---

## 9. Quantized GEMM Patterns

### 9.1 INT4 Weight-Only Quantization

The most common pattern: INT4 weights with FP16/BF16 activation:

```python
import tilelang as tl
from tilelang import language as T

@T.prim_func
def int4_weight_only_gemm(
    A: T.Buffer((M, K), "float16"),           # FP16 activation
    B_packed: T.Buffer((K // 2, N), "uint8"), # Packed INT4 weights (2 per byte)
    scale: T.Buffer((1, N), "float16"),        # Per-column scale
    C: T.Buffer((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        A_shared = T.alloc_shared((128, 32), "float16")
        B_packed_shared = T.alloc_shared((32 // 2, 128), "uint8")
        B_dequant = T.alloc_fragment((32, 128), "float16")
        C_frag = T.alloc_fragment((128, 128), "float16")

        T.clear(C_frag)

        for ko in range(K // 32):
            T.copy(A[by * 128, ko * 32], A_shared)
            T.copy(B_packed[ko * 32 // 2, bx * 128], B_packed_shared)

            # Dequantize INT4 to FP16
            for i, j in T.Parallel(32, 128):
                packed = B_packed_shared[i // 2, j]
                B_dequant[i, j] = _tir_packed_to_unsigned_convert(
                    nbit=4, val=packed, pos=i % 2, dtype="float16"
                ) * scale[0, bx * 128 + j]

            T.gemm(A_shared, B_dequant, C_frag)

        T.copy(C_frag, C[by * 128, bx * 128])
```

### 9.2 FP8 GEMM

FP8 GEMM uses hardware Tensor Core support directly:

```python
@T.prim_func
def fp8_gemm(
    A: T.Buffer((M, K), "float8_e4m3"),
    B: T.Buffer((K, N), "float8_e4m3"),
    C: T.Buffer((M, N), "float32"),
):
    with T.Kernel(...) as (bx, by):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float8_e4m3")
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float8_e4m3")
        C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

        T.clear(C_frag)

        for ko in range(K // BLOCK_K):
            T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_frag)  # Uses FP8 Tensor Core

        T.copy(C_frag, C[by * BLOCK_M, bx * BLOCK_N])
```

FP8 GEMM is supported on:
- CUDA SM90+ (Hopper): MMA with fp8_e4m3 inputs
- ROCm gfx940+ (MI300X): MFMA with fp8 inputs

### 9.3 MXFP4 GEMM with Block Scaling

```python
@T.prim_func
def mxfp4_gemm(
    A: T.Buffer((M, K), "bfloat16"),
    B_packed: T.Buffer((K // 2, N), "uint8"),
    scale: T.Buffer((K // 32, N), "bfloat16"),
    C: T.Buffer((M, N), "float32"),
):
    with T.Kernel(...) as (bx, by):
        # Load and dequantize in blocks
        # Each 32 elements of K share one scale factor
        for ko in range(K // BLOCK_K):
            for ki in range(BLOCK_K // 32):
                scale_val = scale[ko * BLOCK_K // 32 + ki, bx * BLOCK_N:...]
                # Dequantize 32 FP4 elements with shared scale
                # ...
            T.gemm(A_shared, B_dequant, C_frag)
```

### 9.4 W4A8 Quantized GEMM

Weight-4-bit, Activation-8-bit quantization:

```python
@T.prim_func
def w4a8_gemm(
    A: T.Buffer((M, K), "int8"),
    B_packed: T.Buffer((K // 2, N), "uint8"),
    scale_a: T.Buffer((M, K // 32), "float32"),
    scale_b: T.Buffer((K // 2, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    # Dequantize A from INT8 using per-block scale
    # Dequantize B from INT4 using per-column scale
    # GEMM in FP32
```

---

## 10. Dequantization in TileLang Kernels

### 10.1 Inline Dequantization

The simplest approach is to dequantize inline within the kernel:

```python
for i, j in T.Parallel(BLOCK_K, BLOCK_N):
    packed = B_packed_shared[i // 2, j]
    low = (packed >> 0) & 0xF
    high = (packed >> 4) & 0xF
    B_dequant[i, j] = scale[0, j] * (low - 8).astype("float16")
    B_dequant[i + 1, j] = scale[0, j] * (high - 8).astype("float16")
```

### 10.2 Extern Function Dequantization

Using pre-built CUDA/HIP functions:

```python
# Import the dequantization function
T.import_c_source(intrin["c_source"])

# Call it with appropriate buffers
T.call_extern(
    intrin["func_name"],
    B_packed_ptr,
    B_dequant_ptr,
    N,
    scale_ptr,
    zeros_ptr,
)
```

### 10.3 LOP3-Based Dequantization

The fastest approach uses LOP3 intrinsics:

```python
intrin = get_lop3_intrin_group(
    out_dtype=T.float16,
    source_format=T.uint,
    source_bit=4,
    with_scaling=True,
    with_zeros=True,
    zeros_mode="original",
)

T.import_c_source(intrin["c_source"])
T.call_extern(intrin["func_name"], packed_ptr, dequant_ptr, scale_ptr, zeros_ptr, N)
```

---

## 11. Weight Packing and Unpacking

### 11.1 Packing Process

Weights are packed for efficient storage and access:

```
Original weights (INT4):  [w0, w1, w2, w3, ...]  (each 4 bits)
Packed storage (uint8):   [w1<<4 | w0, w3<<4 | w2, ...]
```

### 11.2 Interleaving for Hardware Efficiency

After packing, weights are interleaved so that simultaneous dequantization
can process multiple elements efficiently:

```
Original order:   [w0, w1, w2, w3, w4, w5, w6, w7]
Interleaved:      [w0, w4, w1, w5, w2, w6, w3, w7]
```

This matches the access pattern of LOP3-based dequantization where two
4-bit values are extracted and converted to FP16 simultaneously using
`f16x2` (packed half2) operations.

### 11.3 Scale Factor Layout

Scale factors can be organized as:

| Method | Shape | Description |
|---|---|---|
| Per-tensor | (1, 1) | Single scale for entire weight |
| Per-channel | (1, N) | One scale per output channel |
| Per-group | (K // group_size, N) | One scale per group of K rows |
| Per-block (MXFP) | (K // 32, N) | One scale per 32-element block |

---

## 12. INT4 Quantization Workflow

### 12.1 Complete INT4 GEMM Workflow

1. **Quantize weights** (offline):

```python
from tilelang.quantize.utils import gen_quant4, general_compress, interleave_weight

# Generate test data
original_w, linear, scales, quantized_w = gen_quant4(K, N, groupsize=128)

# Compress to packed format
packed_w = general_compress(quantized_w, source_bits=4, storage_dtype=torch.int8)

# Interleave for LOP3 dequantization
interleaved_w = interleave_weight(packed_w, nbits=4, target_dtype="float16")
```

2. **Write TileLang kernel**:

```python
@T.prim_func
def int4_gemm(A, B_packed, scale, C):
    # ... kernel with dequantization
```

3. **Compile and run**:

```python
kernel = tl.compile(int4_gemm, target="cuda")
result = kernel(A, interleaved_w, scales)
```

---

## 13. FP8 Quantization

### 13.1 FP8 E4M3 Format

| Bit | Field |
|---|---|
| 7 | Sign |
| 6-3 | Exponent (bias = 7) |
| 2-0 | Mantissa |

Range: approximately [-448, 448] with no infinities/NaN

### 13.2 FP8 E5M2 Format

| Bit | Field |
|---|---|
| 7 | Sign |
| 6-2 | Exponent (bias = 15) |
| 1-0 | Mantissa |

Range: approximately [-57344, 57344] with support for infinities/NaN

### 13.3 FP8 in TileLang

FP8 is used directly in GEMM without explicit dequantization:

```python
# FP8 GEMM with hardware support
@T.prim_func
def fp8_gemm(
    A: T.Buffer((M, K), "float8_e4m3"),  # Directly use FP8
    B: T.Buffer((K, N), "float8_e4m3"),
    C: T.Buffer((M, N), "float32"),
):
    T.gemm(A_shared, B_shared, C_frag)  # Hardware handles conversion
```

### 13.4 FP8 Target Support

| Target | FP8 E4M3 | FP8 E5M2 |
|---|---|---|
| CUDA SM90+ | MMA/WGMMA | MMA/WGMMA |
| CUDA SM100+ | TCGEN05 | TCGEN05 |
| ROCm gfx940+ | MFMA | MFMA |
| Other | Not supported | Not supported |

---

## 14. Quantization File Organization

### 14.1 Source Files

| File | Purpose |
|---|---|
| `tilelang/quantize/__init__.py` | Package init, exports |
| `tilelang/quantize/quantization.py` | Core TIR conversion functions |
| `tilelang/quantize/mxfp.py` | MXFP intrinsics and dequantization |
| `tilelang/quantize/lop3.py` | LOP3-based fast dequantization |
| `tilelang/quantize/utils.py` | Weight packing, compression, interleaving |

### 14.2 Test Files

Quantization tests are located at:
- `testing/python/` (general testing)
- `examples/dequantize_gemm/` (end-to-end examples)

Example files:
- `example_dequant_gemm_fine_grained.py` - Fine-grained dequantization
- `example_dequant_gemm_bf16_fp4_hopper.py` - BF16/FP4 on Hopper
- `example_dequant_gemm_bf16_mxfp4_hopper.py` - MXFP4 on Hopper
- `example_dequant_gemv_fp16xint4.py` - FP16xINT4 GEMV
- `example_dequant_gemm_w4a8.py` - W4A8 quantized GEMM
- `example_dequant_gemm_fp4_hopper.py` - FP4 on Hopper

### 14.3 CUDA Template Files

| File | Purpose |
|---|---|
| `src/tl_templates/cuda/cuda_fp8.h` | FP8 data type definitions |
| `src/tl_templates/cuda/cuda_fp4.h` | FP4 data type definitions |
| `src/tl_templates/cuda/compress_sm90.cu` | SM90 compression/decompression |

---

## 15. Advanced Quantization Topics

### 15.1 Sub-Byte Data in Tensor Core

Different Tensor Core instructions support different sub-byte formats:

| Instruction | INT4 | INT2 | INT1 | FP4 | FP6 |
|---|---|---|---|---|---|
| mma.sync (SM80+) | m16n8k64 | - | - | m16n8k64 | - |
| WGMMA (SM90) | - | - | - | - | - |
| TCGEN05 (SM100) | - | - | - | m32nNkK | - |

INT4 MMA on SM80 uses `mma.sync.aligned.m16n8k64.row.col.satfinite.s4.s4.s32`
which processes 64 INT4 elements per K instruction.

### 15.2 Vectorizable Cast Detection

The `IsCudaVectorizableCast` function determines if a type conversion can be
vectorized on CUDA:

```cpp
bool IsCudaVectorizableCast(DataType from_ty, DataType target_ty) {
    // float16 <-> float32
    // bfloat16 <-> float32
    // float32 <-> float8 (E4M3/E5M2)
    // float8_e8m0 <-> bfloat16/float32/float64
    // float4_e2m1fn <-> float16/float32/float64/bfloat16
    // ...
}
```

### 15.3 Quantized GEMM Performance Considerations

1. **Dequantization overhead:** Inline dequantization adds computation;
   balance between dequantization cost and Tensor Core utilization.

2. **Memory bandwidth:** Quantized formats reduce memory bandwidth by 2-4x,
   which is often the bottleneck.

3. **Scale factor access:** Per-block scale factors add memory accesses;
   consider caching scale factors in shared memory.

4. **Packing alignment:** Ensure packed data is aligned for vector loads.

5. **Bank conflicts:** Dequantized data in shared memory should use
   swizzled layouts to avoid bank conflicts.
