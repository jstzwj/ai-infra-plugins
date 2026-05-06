# TileLang Type System Reference

This reference covers the complete TileLang type system, including basic scalar types, reduced-precision floating-point formats, vector types, type specification methods, type properties, conversion utilities, accumulator types for GEMM operations, mixed precision patterns, type promotion rules, and type casting within kernels.

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Scalar Types](#basic-scalar-types)
3. [Float8 Types](#float8-types)
4. [Float6 Types](#float6-types)
5. [Float4 Types](#float4-types)
6. [Vector Types](#vector-types)
7. [Type Specification Methods](#type-specification-methods)
8. [Type Properties](#type-properties)
9. [Type Conversion Methods](#type-conversion-methods)
10. [Callable Types and Constants](#callable-types-and-constants)
11. [Accumulator Types for GEMM](#accumulator-types-for-gemm)
12. [Mixed Precision Patterns](#mixed-precision-patterns)
13. [Type Promotion Rules](#type-promotion-rules)
14. [Type Casting Within Kernels](#type-casting-within-kernels)
15. [Practical Examples](#practical-examples)

---

## Overview

TileLang provides a comprehensive type system that supports the full range of data types used in modern GPU computing. The type system is designed to:

1. **Match GPU hardware types**: Every TileLang type maps directly to a hardware-supported data type.
2. **Enable mixed precision**: Different types can be used for inputs, accumulators, and outputs in the same kernel.
3. **Support emerging formats**: Include experimental types like float4, float6, and MX-format scale factors.
4. **Provide multiple specification methods**: Types can be specified as strings, TileLang type objects, or framework-native types (PyTorch, NumPy).

### Type Hierarchy

```
TileLang Types
|
+-- Integer Types
|   +-- Signed:   int4, int8, int16, int32, int64
|   +-- Unsigned: uint8, uint16, uint32, uint64
|   +-- Boolean:  bool
|
+-- Floating-Point Types
|   +-- Standard:  float16, float32, float64
|   +-- Brain:     bfloat16
|   +-- Float8:    float8_e4m3, float8_e5m2, float8_e4m3fnuz, float8_e5m2fnuz
|   +-- Float6:    float6_e2m3fn, float6_e3m2fn
|   +-- Float4:    float4_e2m1fn, float4_e2m1fnx2
|
+-- Scale Factor Types
|   +-- float8_e8m0  (MX format scale factor, pure exponent)
|
+-- Vector Types
    +-- int8x2, int8x4, int8x8, int8x16, int8x32, int8x64
    +-- uint8x2, uint8x4, uint8x8, uint8x16, uint8x32, uint8x64
    +-- float16x2, float16x4, float16x8, float16x16, float16x32, float16x64
    +-- bfloat16x2, bfloat16x4, bfloat16x8, bfloat16x16, bfloat16x32, bfloat16x64
    +-- float32x2, float32x4, float32x8, float32x16
    +-- uint32x2, uint32x4, uint32x8
```

---

## Basic Scalar Types

### Integer Types

| Type | TileLang Name | Bits | Range | Hardware Support |
|------|-------------|------|-------|-----------------|
| `int4` | `"int4"`, `T.int4` | 4 | -8 to 7 | SM 90+ (tensor cores) |
| `int8` | `"int8"`, `T.int8` | 8 | -128 to 127 | SM 75+ (tensor cores) |
| `int16` | `"int16"`, `T.int16` | 16 | -32768 to 32767 | All SMs |
| `int32` | `"int32"`, `T.int32` | 32 | -2^31 to 2^31-1 | All SMs |
| `int64` | `"int64"`, `T.int64` | 64 | -2^63 to 2^63-1 | All SMs |
| `uint8` | `"uint8"`, `T.uint8` | 8 | 0 to 255 | All SMs |
| `uint16` | `"uint16"`, `T.uint16` | 16 | 0 to 65535 | All SMs |
| `uint32` | `"uint32"`, `T.uint32` | 32 | 0 to 2^32-1 | All SMs |
| `uint64` | `"uint64"`, `T.uint64` | 64 | 0 to 2^64-1 | All SMs |
| `bool` | `"bool"`, `T.bool` | 1 | True/False | All SMs |

### Standard Floating-Point Types

| Type | TileLang Name | Bits | Mantissa | Exponent | Range | Precision |
|------|-------------|------|----------|----------|-------|-----------|
| `float16` | `"float16"`, `T.float16` | 16 | 10 bits | 5 bits | +/- 65504 | ~3 decimal digits |
| `bfloat16` | `"bfloat16"`, `T.bfloat16` | 16 | 7 bits | 8 bits | +/- 3.4e38 | ~2 decimal digits |
| `float32` | `"float32"`, `T.float32` | 32 | 23 bits | 8 bits | +/- 3.4e38 | ~7 decimal digits |
| `float64` | `"float64"`, `T.float64` | 64 | 52 bits | 11 bits | +/- 1.8e308 | ~15 decimal digits |

### Float16 vs Bfloat16 Comparison

```
float16 (IEEE half precision):
  Sign: 1 bit | Exponent: 5 bits | Mantissa: 10 bits
  Range: 5.96e-8 to 6.55e4
  Use case: Inference, mixed precision training on Ampere+

bfloat16 (Brain floating point):
  Sign: 1 bit | Exponent: 8 bits | Mantissa: 7 bits
  Range: 1.18e-38 to 3.4e38 (same range as float32)
  Use case: Training, deep learning workloads

Key difference:
  bfloat16 has the same exponent range as float32 but less mantissa precision.
  float16 has more precision but a much smaller range (requires loss scaling).
  bfloat16 avoids the need for loss scaling in training.
```

### Boolean Type

```python
import tilelang.language as T

# Boolean type for masks and conditions
mask = T.alloc_shared([M, N], "bool")

# Boolean operations
with T.Parallel(M, N) as i, j:
    mask[i, j] = data[i, j] > threshold  # Comparison produces bool
    if mask[i, j]:
        output[i, j] = data[i, j]  # Conditional on bool
```

---

## Float8 Types

Float8 types are 8-bit floating-point representations designed for deep learning workloads. They were introduced in the paper "FP8 Formats for Deep Learning" (NVIDIA, AMD, Intel, ARM, 2022).

### float8_e4m3 (FN)

| Property | Value |
|----------|-------|
| TileLang Name | `"float8_e4m3"`, `T.float8_e4m3` |
| Total Bits | 8 |
| Sign | 1 bit |
| Exponent | 4 bits (bias 7) |
| Mantissa | 3 bits |
| Max Value | 448.0 |
| Min Normal | 2^-6 = 0.015625 |
| Subnormals | Yes |
| NaN | Yes (exponent=15, mantissa=111) |
| Infinity | No |

```
Bit layout: S EEEE MMM
            0 0000 000 = 0
            0 0111 000 = 1.0
            0 1111 110 = 448.0 (max)
            0 1111 111 = NaN
```

**Primary use**: Forward-pass activations and weights in FP8 training. The larger mantissa (3 bits) provides better precision for representing activation values.

### float8_e5m2

| Property | Value |
|----------|-------|
| TileLang Name | `"float8_e5m2"`, `T.float8_e5m2` |
| Total Bits | 8 |
| Sign | 1 bit |
| Exponent | 5 bits (bias 15) |
| Mantissa | 2 bits |
| Max Value | 57344.0 |
| Min Normal | 2^-14 = 0.000061 |
| Subnormals | Yes |
| NaN | Yes |
| Infinity | Yes |

```
Bit layout: S EEEEE MM
            0 00000 00 = 0
            0 01111 00 = 1.0
            0 11110 11 = 57344.0 (max)
            0 11111 00 = Inf
            0 11111 01 = NaN
```

**Primary use**: Gradients and error propagation in FP8 training. The larger exponent range (5 bits) prevents overflow in gradient computation.

### float8_e4m3fnuz

| Property | Value |
|----------|-------|
| TileLang Name | `"float8_e4m3fnuz"`, `T.float8_e4m3fnuz` |
| Total Bits | 8 |
| Sign | 1 bit |
| Exponent | 4 bits (bias 8) |
| Mantissa | 3 bits |
| Max Value | 240.0 |
| NaN | No |
| Infinity | No |

**Primary use**: Variant of E4M3 used by some hardware vendors. The "fnuz" suffix indicates "no NaN, no infinity, unbiased zero". The bias is different from the standard E4M3.

### float8_e5m2fnuz

| Property | Value |
|----------|-------|
| TileLang Name | `"float8_e5m2fnuz"`, `T.float8_e5m2fnuz` |
| Total Bits | 8 |
| Sign | 1 bit |
| Exponent | 5 bits (bias 16) |
| Mantissa | 2 bits |
| Max Value | 57344.0 |
| NaN | No |
| Infinity | No |

**Primary use**: Variant of E5M2 used by some hardware vendors.

### Float8 Type Comparison

| Type | Exponent | Mantissa | Range | NaN | Inf | Use Case |
|------|----------|----------|-------|-----|-----|----------|
| `float8_e4m3` | 4 | 3 | [0, 448] | Yes | No | Activations, weights |
| `float8_e5m2` | 5 | 2 | [0, 57344] | Yes | Yes | Gradients, errors |
| `float8_e4m3fnuz` | 4 | 3 | [0, 240] | No | No | Alternative forward pass |
| `float8_e5m2fnuz` | 5 | 2 | [0, 57344] | No | No | Alternative backward pass |

### Example: FP8 GEMM

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def fp8_matmul(M, N, K, block_M=128, block_N=128, block_K=64):
    # FP8 inputs: E4M3 for A, E5M2 for B (optimal for forward pass)
    A_smem = T.alloc_shared([block_M, block_K], "float8_e4m3")
    B_smem = T.alloc_shared([block_K, block_N], "float8_e5m2")
    # FP32 accumulation
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    T.copy(C_local, C_global)
    return C_global
```

---

## Float6 Types

Float6 types are 6-bit floating-point representations designed for extreme compression in deep learning inference and training.

### float6_e2m3fn

| Property | Value |
|----------|-------|
| TileLang Name | `"float6_e2m3fn"`, `T.float6_e2m3fn` |
| Total Bits | 6 |
| Sign | 1 bit |
| Exponent | 2 bits (bias 1) |
| Mantissa | 3 bits |
| Max Value | 7.5 |
| NaN | Yes |

```
Bit layout: S EE MMM
            0 00 000 = 0
            0 01 000 = 1.0
            0 11 110 = 7.5 (max)
```

**Primary use**: Narrow-precision value representation in MX (Microscaling) format blocks. Optimized for maximum mantissa precision with limited range.

### float6_e3m2fn

| Property | Value |
|----------|-------|
| TileLang Name | `"float6_e3m2fn"`, `T.float6_e3m2fn` |
| Total Bits | 6 |
| Sign | 1 bit |
| Exponent | 3 bits (bias 3) |
| Mantissa | 2 bits |
| Max Value | 28.0 |
| NaN | Yes |

```
Bit layout: S EEE MM
            0 000 00 = 0
            0 011 00 = 1.0
            0 110 11 = 28.0 (max)
```

**Primary use**: Alternative 6-bit format with more exponent range, suitable for values with larger dynamic range.

### Float6 Type Comparison

| Type | Exponent | Mantissa | Range | Use Case |
|------|----------|----------|-------|----------|
| `float6_e2m3fn` | 2 | 3 | [0, 7.5] | High-precision MX values |
| `float6_e3m2fn` | 3 | 2 | [0, 28.0] | Wider-range MX values |

---

## Float4 Types

Float4 types are 4-bit floating-point representations, the narrowest practical format for deep learning.

### float4_e2m1fn

| Property | Value |
|----------|-------|
| TileLang Name | `"float4_e2m1fn"`, `T.float4_e2m1fn` |
| Total Bits | 4 |
| Sign | 1 bit |
| Exponent | 2 bits (bias 1) |
| Mantissa | 1 bit |
| Representable Values | {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} (+ signed versions) |
| NaN | No |

```
Bit layout: S EE M
            0 00 0 = 0
            0 01 0 = 1.0
            0 01 1 = 1.5
            0 10 0 = 2.0
            0 10 1 = 3.0
            0 11 0 = 4.0
            0 11 1 = 6.0 (max)
```

**Primary use**: Extreme compression in MX format blocks. Only 8 distinct positive values are representable, so block scaling is essential.

### float4_e2m1fnx2

| Property | Value |
|----------|-------|
| TileLang Name | `"float4_e2m1fnx2"`, `T.float4_e2m1fnx2` |
| Total Bits | 4 per element, stored as pairs (8 bits per pair) |
| Description | Packed float4: two `float4_e2m1fn` values stored in a single byte. |

**Primary use**: Memory-efficient storage format for float4 data. Two values are packed into one byte for compact shared/global memory representation.

### MX Format Scale Factor: float8_e8m0

| Property | Value |
|----------|-------|
| TileLang Name | `"float8_e8m0"`, `T.float8_e8m0` |
| Total Bits | 8 |
| Sign | 0 bits (always positive) |
| Exponent | 8 bits (bias 127) |
| Mantissa | 0 bits |
| Representable Values | Powers of 2: 2^(e-127) where e in [0, 254] |
| Max Value | 2^127 |
| Special | e=255 represents NaN or infinity |

```
Bit layout: EEEEEEEE
            00000000 = 2^-127 (min)
            01111111 = 2^0 = 1.0
            11111110 = 2^127 (max)
```

**Primary use**: Block scale factors in MX format. Each scale factor is a power of 2, applied to a block of 32 narrow-format values.

### Example: MX Format Storage

```python
import tilelang.language as T

# MX format: 32 values share one scale factor
block_size = 32

# Values in float4 (4 bits each)
values = T.alloc_shared([M, K], "float4_e2m1fn")

# Scale factors (one per block of 32 values)
scales = T.alloc_shared([M, K // block_size], "float8_e8m0")

# Reconstructed values: values[i, j] * scales[i, j // 32]
# Used in block-scaled GEMM: T.tcgen05_gemm_blockscaled(...)
```

---

## Vector Types

Vector types represent multiple elements of the same scalar type packed into a single register or memory word. They enable efficient vectorized memory access and computation.

### Vector Type Naming Convention

Vector types follow the pattern `{scalar_type}x{count}`:

```
float16x4 = 4 x float16 values packed together (64 bits)
int8x16   = 16 x int8 values packed together (128 bits)
uint32x8  = 8 x uint32 values packed together (256 bits)
```

### Available Vector Types

#### int8 Vector Types

| Type | Elements | Total Bits | Register Mapping |
|------|----------|-----------|-----------------|
| `int8x2` | 2 | 16 | Half register |
| `int8x4` | 4 | 32 | Full register |
| `int8x8` | 8 | 64 | Double register |
| `int8x16` | 16 | 128 | Quad register |
| `int8x32` | 32 | 256 | 8 registers |
| `int8x64` | 64 | 512 | 16 registers |

#### float16 Vector Types

| Type | Elements | Total Bits | Register Mapping |
|------|----------|-----------|-----------------|
| `float16x2` | 2 | 32 | Full register (native GPU type) |
| `float16x4` | 4 | 64 | Double register |
| `float16x8` | 8 | 128 | Quad register |
| `float16x16` | 16 | 256 | 8 registers |
| `float16x32` | 32 | 512 | 16 registers |
| `float16x64` | 64 | 1024 | 32 registers |

#### bfloat16 Vector Types

| Type | Elements | Total Bits | Register Mapping |
|------|----------|-----------|-----------------|
| `bfloat16x2` | 2 | 32 | Full register |
| `bfloat16x4` | 4 | 64 | Double register |
| `bfloat16x8` | 8 | 128 | Quad register |
| `bfloat16x16` | 16 | 256 | 8 registers |
| `bfloat16x32` | 32 | 512 | 16 registers |
| `bfloat16x64` | 64 | 1024 | 32 registers |

#### float32 Vector Types

| Type | Elements | Total Bits | Register Mapping |
|------|----------|-----------|-----------------|
| `float32x2` | 2 | 64 | Double register |
| `float32x4` | 4 | 128 | Quad register |
| `float32x8` | 8 | 256 | 8 registers |
| `float32x16` | 16 | 512 | 16 registers |

#### uint32 Vector Types

| Type | Elements | Total Bits | Register Mapping |
|------|----------|-----------|-----------------|
| `uint32x2` | 2 | 64 | Double register |
| `uint32x4` | 4 | 128 | Quad register |
| `uint32x8` | 8 | 256 | 8 registers |

### Example: Using Vector Types

```python
import tilelang.language as T

# Vector types are used implicitly by T.copy and T.Parallel
# with appropriate vectorization settings

# Explicit vector type for buffer allocation
vec_buffer = T.alloc_shared([N], "float16x4")  # N/4 elements, each holding 4 float16 values

# Vector load: load 4 float16 values at once
for i in T.Vectorized(0, N // 4):
    vec_buffer[i] = T.load("float16x4", src, i * 4)
```

### Vector Type Element Access

```python
# Access individual elements within a vector
vec = T.alloc_local([1], "float16x4")
vec[0] = some_value  # Assign full vector

# Access element 0 of the vector
element_0 = vec[0].x  # Or vec.lo.lo depending on hardware
element_1 = vec[0].y
```

---

## Type Specification Methods

TileLang supports three methods for specifying data types:

### Method 1: String Name

The most common method. Specify the type as a string:

```python
# String type specification
A = T.alloc_shared([M, K], "float16")
B = T.alloc_shared([K, N], "bfloat16")
C = T.alloc_local([M, N], "float32")
D = T.alloc_shared([M, K], "float8_e4m3")
E = T.alloc_shared([M, K], "int8")
```

### Method 2: TileLang Type Object

Use the type objects defined in the `T` module:

```python
import tilelang.language as T

# TileLang type object specification
A = T.alloc_shared([M, K], T.float16)
B = T.alloc_shared([K, N], T.bfloat16)
C = T.alloc_local([M, N], T.float32)
D = T.alloc_shared([M, K], T.float8_e4m3)
E = T.alloc_shared([M, K], T.int8)
```

### Method 3: Framework-Native Type

Use PyTorch or NumPy dtypes:

```python
import torch
import numpy as np
import tilelang.language as T

# PyTorch dtype specification
A = T.alloc_shared([M, K], torch.float16)
B = T.alloc_shared([K, N], torch.bfloat16)
C = T.alloc_local([M, N], torch.float32)

# NumPy dtype specification
D = T.alloc_shared([M, K], np.float16)
E = T.alloc_shared([M, K], np.int8)
```

### Type Specification Equivalence

| String | TileLang Object | PyTorch | NumPy |
|--------|----------------|---------|-------|
| `"float16"` | `T.float16` | `torch.float16` | `np.float16` |
| `"float32"` | `T.float32` | `torch.float32` | `np.float32` |
| `"float64"` | `T.float64` | `torch.float64` | `np.float64` |
| `"bfloat16"` | `T.bfloat16` | `torch.bfloat16` | -- |
| `"int8"` | `T.int8` | `torch.int8` | `np.int8` |
| `"int16"` | `T.int16` | `torch.int16` | `np.int16` |
| `"int32"` | `T.int32` | `torch.int32` | `np.int32` |
| `"int64"` | `T.int64` | `torch.int64` | `np.int64` |
| `"uint8"` | `T.uint8` | `torch.uint8` | `np.uint8` |
| `"bool"` | `T.bool` | `torch.bool` | `np.bool_` |
| `"float8_e4m3"` | `T.float8_e4m3` | `torch.float8_e4m3fn` | -- |
| `"float8_e5m2"` | `T.float8_e5m2` | `torch.float8_e5m2` | -- |

---

## Type Properties

Each TileLang type has several properties that can be queried:

### .bits -- Bit Width

Returns the number of bits per element:

```python
T.float16.bits       # 16
T.float32.bits       # 32
T.int8.bits          # 8
T.float8_e4m3.bits   # 8
T.float4_e2m1fn.bits # 4
T.bool.bits          # 1
```

### .bytes -- Byte Width

Returns the number of bytes per element (bits / 8, rounded up):

```python
T.float16.bytes      # 2
T.float32.bytes      # 4
T.int8.bytes         # 1
T.float8_e4m3.bytes  # 1
T.float4_e2m1fn.bytes # 1 (minimum addressable unit)
```

### .lanes -- Vector Lanes

For vector types, returns the number of scalar elements packed into one vector:

```python
T.float16.lanes      # 1 (scalar type)
T.float16x4.lanes    # 4 (vector of 4 float16)
T.float16x8.lanes    # 8
T.int8x32.lanes      # 32
```

### Property Reference Table

| Type | .bits | .bytes | .lanes |
|------|-------|--------|--------|
| `T.bool` | 1 | 1 | 1 |
| `T.int4` | 4 | 1 | 1 |
| `T.int8` | 8 | 1 | 1 |
| `T.int16` | 16 | 2 | 1 |
| `T.int32` | 32 | 4 | 1 |
| `T.int64` | 64 | 8 | 1 |
| `T.uint8` | 8 | 1 | 1 |
| `T.uint16` | 16 | 2 | 1 |
| `T.uint32` | 32 | 4 | 1 |
| `T.uint64` | 64 | 8 | 1 |
| `T.float16` | 16 | 2 | 1 |
| `T.bfloat16` | 16 | 2 | 1 |
| `T.float32` | 32 | 4 | 1 |
| `T.float64` | 64 | 8 | 1 |
| `T.float8_e4m3` | 8 | 1 | 1 |
| `T.float8_e5m2` | 8 | 1 | 1 |
| `T.float6_e2m3fn` | 6 | 1 | 1 |
| `T.float6_e3m2fn` | 6 | 1 | 1 |
| `T.float4_e2m1fn` | 4 | 1 | 1 |
| `T.float8_e8m0` | 8 | 1 | 1 |
| `T.float16x2` | 32 | 4 | 2 |
| `T.float16x4` | 64 | 8 | 4 |
| `T.float16x8` | 128 | 16 | 8 |
| `T.int8x4` | 32 | 4 | 4 |
| `T.int8x16` | 128 | 16 | 16 |
| `T.uint32x4` | 128 | 16 | 4 |

---

## Type Conversion Methods

### .as_torch() -- Convert to PyTorch dtype

Returns the equivalent PyTorch dtype:

```python
T.float16.as_torch()       # torch.float16
T.float32.as_torch()       # torch.float32
T.bfloat16.as_torch()      # torch.bfloat16
T.int8.as_torch()          # torch.int8
T.float8_e4m3.as_torch()   # torch.float8_e4m3fn
T.float8_e5m2.as_torch()   # torch.float8_e5m2
T.bool.as_torch()          # torch.bool
```

### .as_numpy() -- Convert to NumPy dtype

Returns the equivalent NumPy dtype:

```python
T.float16.as_numpy()    # np.float16
T.float32.as_numpy()    # np.float32
T.float64.as_numpy()    # np.float64
T.int8.as_numpy()       # np.int8
T.int32.as_numpy()      # np.int32
T.uint8.as_numpy()      # np.uint8
T.bool.as_numpy()       # np.bool_
```

Note: NumPy does not have native equivalents for all TileLang types. Types without NumPy equivalents (e.g., `bfloat16`, `float8_e4m3`, `float4_e2m1fn`) will raise an error when `.as_numpy()` is called.

### Conversion in Practice

```python
import torch
import tilelang.language as T

# Create a PyTorch tensor with the matching dtype
dtype = T.float16
tensor = torch.randn(M, K, dtype=dtype.as_torch(), device="cuda")

# Pass to a TileLang kernel
kernel = my_jit_kernel(M=M, K=K, in_dtype="float16")
result = kernel(tensor)
```

---

## Callable Types and Constants

### Creating Constants with Type Constructors

TileLang type objects can be called as functions to create typed constants:

```python
import tilelang.language as T

# Create a float32 constant
pi = T.float32(3.14159)

# Create a float16 constant
half_one = T.float16(1.0)

# Create an int32 constant
count = T.int32(42)

# Create a bfloat16 constant
bf_val = T.bfloat16(2.718)
```

### How Constants Work in Kernels

When a type is called as a function inside a TileLang kernel, it creates a constant value with the specified type:

```python
@tilelang.jit(out_idx=[0])
def scale_kernel(M, N, dtype="float16"):
    buffer = T.alloc_shared([M, N], dtype)

    with T.Parallel(M, N) as i, j:
        # Create typed constant inside kernel
        scale = T.float32(0.125)  # Creates a float32 constant 0.125
        buffer[i, j] = buffer[i, j] * scale

    T.copy(buffer, out_global)
    return out_global
```

### Constant Type Inference

When numeric literals are used in TileLang kernels, the type is inferred from context:

```python
# Integer literal -> int32
x = 42

# Float literal -> float32
y = 3.14

# Type is inferred from the buffer being assigned to
buffer_float16 = T.alloc_shared([N], "float16")
buffer_float16[i] = 1.0  # 1.0 is implicitly converted to float16
```

### Type Coercion Rules for Constants

When a constant is used in an expression with a buffer, the constant is coerced to the buffer's type:

```python
# float16 buffer * float32 constant -> float16 result (constant is downcast)
result = float16_buffer * 2.0  # 2.0 is coerced to float16

# float32 buffer * float16 constant -> float32 result (constant is upcast)
result = float32_buffer * T.float16(1.5)  # 1.5 is promoted to float32
```

---

## Accumulator Types for GEMM

In GEMM operations, the accumulator type determines the precision of the intermediate sum-of-products. Choosing the correct accumulator type is critical for numerical accuracy.

### Recommended Accumulator Types

| Input Type | Recommended Accumulator | Rationale |
|-----------|------------------------|-----------|
| `float16` | `float32` | Prevent overflow: max float16 * max float16 * K_iterations can exceed float16 range |
| `bfloat16` | `float32` | Same rationale as float16 |
| `tf32` | `float32` | Natural pairing, same range |
| `float8_e4m3` | `float32` | Very narrow mantissa (3 bits) needs wide accumulation |
| `float8_e5m2` | `float32` | Same as above |
| `int8` | `int32` | 127 * 127 * K can overflow int8 or int16 |
| `int4` | `int32` | 7 * 7 * K can overflow int8 or int16 |
| `float16` (inference) | `float16` | Acceptable for inference if K is small (< 256) |

### Why Wide Accumulation is Necessary

Consider a float16 GEMM with K=1024:

```
Worst case: sum of 1024 products of max float16 values
= 1024 * 65504 * 65504
= 4.4e12

float16 max = 65504 -> OVERFLOW!
float32 max = 3.4e38 -> OK!
```

Without float32 accumulation, the sum would overflow and produce incorrect results.

### Accumulator Type Specification

```python
import tilelang.language as T

# Specify accumulator type explicitly
C_local = T.alloc_local([block_M, block_N], "float32")  # Accumulator

# GEMM will accumulate in float32 even though inputs are float16
A = T.alloc_shared([block_M, block_K], "float16")
B = T.alloc_shared([block_K, block_N], "float16")
T.gemm(A, B, C_local)  # Accumulates in C_local's type (float32)
```

### Accumulator Precision and Numerical Error

| Accumulator | Max Relative Error (for K=1024) | Safe K Range |
|------------|-------------------------------|-------------|
| float16 | ~0.1% (highly scenario-dependent) | K < 128 |
| float32 | ~0.001% | K < 10^6 |
| int32 (for int8 input) | Exact | K < 2^17 / 127^2 |

---

## Mixed Precision Patterns

Mixed precision is the practice of using different data types for different parts of a computation to balance performance and accuracy.

### Pattern 1: FP16 Input, FP32 Accumulation, FP16 Output

The most common mixed precision pattern for deep learning:

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def mixed_precision_gemm(M, N, K, block_M=128, block_N=128, block_K=32):
    # Input: float16 (2x throughput vs float32)
    A_smem = T.alloc_shared([block_M, block_K], "float16")
    B_smem = T.alloc_shared([block_K, block_N], "float16")

    # Accumulator: float32 (numerical stability)
    C_local = T.alloc_local([block_M, block_N], "float32")

    # Output: float16 (compact storage)
    # (C_global is float16, conversion happens at T.copy)

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Automatic float32 -> float16 conversion during copy
    T.copy(C_local, C_global)
    return C_global
```

### Pattern 2: FP8 Input, FP32 Accumulation, FP16 Output

For Hopper (SM 90) FP8 tensor cores:

```python
@tilelang.jit(out_idx=[2])
def fp8_gemm(M, N, K, block_M=128, block_N=128, block_K=64):
    # FP8 inputs (4x throughput vs float16)
    A_smem = T.alloc_shared([block_M, block_K], "float8_e4m3")
    B_smem = T.alloc_shared([block_K, block_N], "float8_e5m2")

    # FP32 accumulation
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Output as float16
    T.copy(C_local, C_global)
    return C_global
```

### Pattern 3: INT8 Input, INT32 Accumulation, FP16 Output

For quantized inference:

```python
@tilelang.jit(out_idx=[2])
def int8_gemm(M, N, K, block_M=128, block_N=128, block_K=64):
    # Quantized int8 inputs
    A_smem = T.alloc_shared([block_M, block_K], "int8")
    B_smem = T.alloc_shared([block_K, block_N], "int8")

    # Int32 accumulation (exact integer arithmetic)
    C_local = T.alloc_local([block_M, block_N], "int32")

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Dequantize: int32 -> float16 with scale factor
    # C_float = C_int32 * scale_a * scale_b
    for i in range(block_M):
        for j in range(block_N):
            C_local[i, j] = C_local[i, j] * scale_a * scale_b

    T.copy(C_local, C_global)
    return C_global
```

### Pattern 4: Block-Scaled MX Format

For Blackwell (SM 100) block-scaled GEMM:

```python
@tilelang.jit(out_idx=[2])
def mx_gemm(M, N, K, block_M=128, block_N=256, block_K=128):
    block_size = 32

    # MX-format values (float4 per element, 8x compression vs float32)
    A_vals = T.alloc_shared([block_M, block_K], "float4_e2m1fn")
    B_vals = T.alloc_shared([block_K, block_N], "float4_e2m1fn")

    # Block scale factors
    A_scales = T.alloc_tmem([block_M, block_K // block_size], "float8_e8m0")
    B_scales = T.alloc_tmem([block_K // block_size, block_N], "float8_e8m0")

    # FP32 accumulation
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_vals)
        T.copy(B_global[k:k+block_K], B_vals)
        T.copy(A_scales_global[k:k+block_K:block_size], A_scales)
        T.copy(B_scales_global[k:k+block_K:block_size], B_scales)
        T.sync_shared_memory()

        T.tcgen05_gemm_blockscaled(
            A_vals, B_vals, C_local,
            A_scales, B_scales
        )

    T.copy(C_local, C_global)
    return C_global
```

### Mixed Precision Pattern Summary

| Pattern | Input | Accumulator | Output | Throughput Gain | HW Requirement |
|--------|-------|-------------|--------|----------------|----------------|
| Standard | FP16 | FP32 | FP16 | 2x vs FP32 | SM 70+ |
| BF16 | BF16 | FP32 | BF16 | 2x vs FP32 | SM 80+ |
| TF32 | TF32 | FP32 | FP32 | 2x vs FP32 | SM 80+ |
| FP8 | FP8 | FP32 | FP16 | 4x vs FP16 | SM 90+ |
| INT8 | INT8 | INT32 | FP16 | 2x vs FP16 | SM 75+ |
| MX-FP4 | FP4 | FP32 | FP16 | 8x vs FP16 | SM 100+ |
| MX-FP8 | FP8 | FP32 | FP16 | 4x vs FP16 | SM 100+ |

---

## Type Promotion Rules

When operations involve operands of different types, TileLang applies type promotion rules to determine the result type.

### Binary Operation Promotion

For a binary operation `A op B` where A and B have different types:

```
Rule 1: If either operand is float64, the result is float64.
Rule 2: If either operand is float32, the result is float32.
Rule 3: If both operands are the same type, the result is that type.
Rule 4: For mixed integer types, the result is the wider type.
Rule 5: For float16/bfloat16 mixed with float32, the result is float32.
Rule 6: For narrow float types mixed with float32, the result is float32.
```

### Promotion Table

| Type A | Type B | Result Type |
|--------|--------|------------|
| float16 | float16 | float16 |
| float16 | float32 | float32 |
| float16 | bfloat16 | float32 |
| bfloat16 | float32 | float32 |
| float32 | float32 | float32 |
| float8_e4m3 | float32 | float32 |
| float8_e4m3 | float16 | float16 |
| float8_e4m3 | float8_e5m2 | float32 |
| int8 | int32 | int32 |
| int8 | float16 | float16 |
| int32 | float32 | float32 |

### Explicit Type Control

To override promotion rules, use explicit type casting:

```python
# Force float16 result even though one operand is float32
result = T.cast(A_float32, "float16") * B_float16

# Force float32 accumulation even though inputs are float16
acc = T.float32(0.0)  # Explicit float32 initial value
for k in range(K):
    acc = acc + T.cast(A_float16[k], "float32") * T.cast(B_float16[k], "float32")
```

---

## Type Casting Within Kernels

### Explicit Type Casting

TileLang provides `T.cast` for explicit type conversion:

```python
# Cast buffer from one type to another
float32_values = T.cast(float16_buffer, "float32")
float16_values = T.cast(float32_buffer, "float16")
int8_values = T.cast(float32_buffer, "int8")
float32_values = T.cast(int8_buffer, "float32")
```

### Cast Behavior

| Source | Target | Behavior |
|--------|--------|---------|
| float16 -> float32 | Upcast | Exact (no precision loss) |
| float32 -> float16 | Downcast | Round to nearest even |
| bfloat16 -> float32 | Upcast | Exact (no precision loss) |
| float32 -> bfloat16 | Downcast | Round to nearest even |
| float8 -> float32 | Upcast | Exact |
| float32 -> float8 | Downcast | Round to nearest, may overflow |
| float32 -> int8 | Quantize | Round + clamp to [-128, 127] |
| int8 -> float32 | Dequantize | Exact |
| float16 -> bfloat16 | Reinterpret | Different mantissa/exponent split |
| float32 -> float4 | Downcast | Extreme quantization, 4 bits |

### Implicit Casting in Operations

When a buffer of one type is assigned to a buffer of another type, implicit casting occurs:

```python
# Implicit float16 -> float32 upcast
A_fp16 = T.alloc_shared([M, K], "float16")
B_fp32 = T.alloc_local([M, K], "float32")
B_fp32[i, j] = A_fp16[i, j]  # Implicit upcast

# Implicit float32 -> float16 downcast
C_fp16 = T.alloc_shared([M, K], "float16")
C_fp16[i, j] = B_fp32[i, j]  # Implicit downcast with rounding
```

### Quantization and Dequantization

```python
# Quantization: float32 -> int8 with scale factor
scale = T.float32(0.1)
quantized = T.cast(float32_buffer / scale, "int8")

# Dequantization: int8 -> float32 with scale factor
dequantized = T.cast(int8_buffer, "float32") * scale
```

### Safe Downcasting

When downcasting to a narrower type, values outside the target range are clamped:

```python
# float32 -> float16: values outside [-65504, 65504] are clamped
# float32 -> int8: values outside [-128, 127] are clamped
# float32 -> float8_e4m3: values outside [0, 448] are clamped to 448

# This is handled automatically by T.cast
clamped = T.cast(large_float32_values, "float16")  # Safe: overflow is clamped
```

---

## Practical Examples

### Complete Mixed-Precision Transformer Block

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[0])
def transformer_ffn(
    batch, seq_len, dim, hidden_dim,
    block_M=64, block_N=64, block_K=32,
):
    # All linear layers use FP16 input, FP32 accumulation
    # QKV projection: [batch, seq_len, dim] @ [dim, 3*dim]
    A_smem = T.alloc_shared([block_M, block_K], "float16")
    B_smem = T.alloc_shared([block_K, block_N], "float16")
    C_local = T.alloc_local([block_M, block_N], "float32")
    activation = T.alloc_shared([block_M, block_N], "float16")

    T.clear(C_local)

    for k in T.serial(0, dim, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Apply GELU activation (in float32, then cast back to float16)
    with T.Parallel(block_M, block_N) as i, j:
        x = C_local[i, j]
        # GELU approximation
        gelu = x * 0.5 * (1.0 + T.tanh(0.7978845608 * (x + 0.044715 * x * x)))
        activation[i, j] = gelu  # Implicit float32 -> float16 cast

    T.copy(activation, out_global)
    return out_global
```

### FP8 Training: Forward and Backward Pass Types

```python
import tilelang.language as T

# Forward pass type configuration
forward_types = {
    "activation_dtype": "float8_e4m3",  # E4M3 for forward (more precision)
    "weight_dtype": "float8_e4m3",
    "accumulator_dtype": "float32",
    "output_dtype": "float8_e4m3",
}

# Backward pass type configuration
backward_types = {
    "gradient_dtype": "float8_e5m2",    # E5M2 for gradients (more range)
    "weight_dtype": "float8_e4m3",
    "accumulator_dtype": "float32",
    "output_dtype": "float8_e5m2",
}

# Full training GEMM with proper FP8 type selection
@tilelang.jit(out_idx=[2])
def fp8_forward_gemm(M, N, K, block_M=128, block_N=128, block_K=64):
    # E4M3 for activations and weights
    A_smem = T.alloc_shared([block_M, block_K], "float8_e4m3")
    B_smem = T.alloc_shared([block_K, block_N], "float8_e4m3")
    C_local = T.alloc_local([block_M, block_N], "float32")

    T.clear(C_local)

    for k in T.serial(0, K, block_K):
        T.copy(A_global[k:k+block_K], A_smem)
        T.copy(B_global[k:k+block_K], B_smem)
        T.sync_shared_memory()
        T.gemm(A_smem, B_smem, C_local)

    # Cast output back to E4M3 for downstream consumption
    C_fp8 = T.alloc_shared([block_M, block_N], "float8_e4m3")
    for i in range(block_M):
        for j in range(block_N):
            C_fp8[i, j] = C_local[i, j]  # Implicit float32 -> float8_e4m3

    T.copy(C_fp8, C_global)
    return C_global
```

### Dynamic Type Selection

```python
import tilelang
import tilelang.language as T

def create_gemm_kernel(in_dtype="float16", out_dtype="float16", accum_dtype="float32"):
    """Create a GEMM kernel with configurable types."""

    @tilelang.jit(out_idx=[2])
    def gemm_kernel(M, N, K, block_M=128, block_N=128, block_K=32):
        A_smem = T.alloc_shared([block_M, block_K], in_dtype)
        B_smem = T.alloc_shared([block_K, block_N], in_dtype)
        C_local = T.alloc_local([block_M, block_N], accum_dtype)

        T.clear(C_local)

        for k in T.serial(0, K, block_K):
            T.copy(A_global[k:k+block_K], A_smem)
            T.copy(B_global[k:k+block_K], B_smem)
            T.sync_shared_memory()
            T.gemm(A_smem, B_smem, C_local)

        T.copy(C_local, C_global)
        return C_global

    return gemm_kernel

# Create kernels for different precision levels
fp16_kernel = create_gemm_kernel("float16", "float16", "float32")
bf16_kernel = create_gemm_kernel("bfloat16", "bfloat16", "float32")
fp8_kernel = create_gemm_kernel("float8_e4m3", "float16", "float32")
int8_kernel = create_gemm_kernel("int8", "float16", "int32")
```

### Type-aware Memory Size Calculation

```python
import tilelang.language as T

def calculate_smem_usage(block_M, block_N, block_K, num_stages, in_dtype):
    """Calculate shared memory usage for a tiled GEMM."""
    dtype = T.get_type(in_dtype)
    bytes_per_element = dtype.bytes

    # Two buffers: A and B, each with num_stages copies
    a_bytes = num_stages * block_M * block_K * bytes_per_element
    b_bytes = num_stages * block_K * block_N * bytes_per_element

    total = a_bytes + b_bytes
    return total

# Example: FP16 GEMM with 128x128x32 tiles and 3 stages
smem = calculate_smem_usage(128, 128, 32, 3, "float16")
# = 3 * 128 * 32 * 2 + 3 * 32 * 128 * 2 = 24576 + 24576 = 49152 bytes = 48 KB

# Example: FP8 GEMM with same tiles
smem_fp8 = calculate_smem_usage(128, 128, 32, 3, "float8_e4m3")
# = 3 * 128 * 32 * 1 + 3 * 32 * 128 * 1 = 12288 + 12288 = 24576 bytes = 24 KB
```

---

## Summary

### Complete Type Reference

| Category | Types | Key Characteristics |
|----------|-------|-------------------|
| Standard Float | float16, bfloat16, float32, float64 | IEEE 754 and brain float formats |
| Float8 | float8_e4m3, float8_e5m2, float8_e4m3fnuz, float8_e5m2fnuz | 8-bit floating point for AI workloads |
| Float6 | float6_e2m3fn, float6_e3m2fn | 6-bit formats for extreme compression |
| Float4 | float4_e2m1fn, float4_e2m1fnx2 | Minimum practical FP format |
| Scale | float8_e8m0 | MX format pure-exponent scale factor |
| Integer | int4, int8, int16, int32, int64 | Signed integers |
| Unsigned | uint8, uint16, uint32, uint64 | Unsigned integers |
| Boolean | bool | Binary true/false |
| Vector | {type}x{2,4,8,16,32,64} | Packed multi-element types |

### Type Selection Decision Tree

```
Need highest throughput?
  -> FP8 (float8_e4m3/float8_e5m2) on SM 90+
  -> INT8 on SM 75+
  -> MX format (float4_e2m1fn + float8_e8m0) on SM 100+

Need good balance of speed and accuracy?
  -> float16 + float32 accumulation (standard mixed precision)
  -> bfloat16 + float32 accumulation (for training)

Need exact arithmetic?
  -> float32 for general computation
  -> float64 for scientific computing
  -> int32/int64 for indexing and counting

Need minimum memory usage?
  -> float4_e2m1fn (4 bits per element, MX format)
  -> float8_e4m3 (8 bits per element)
  -> int4 (4 bits per element, tensor core only)
```

The TileLang type system is designed to provide seamless support for the full spectrum of data types used in modern GPU computing, from standard IEEE formats to emerging reduced-precision formats. The system handles type promotion, casting, and conversion automatically while giving developers explicit control when needed for performance-critical mixed-precision kernels.
