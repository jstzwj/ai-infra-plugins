# Data Types Reference

## Overview

Data types are fundamental to cuTile programming, determining how values are stored, computed, and promoted through operations. This chapter provides a comprehensive reference for all cuTile data types, their characteristics, promotion rules, and behavior in different computational contexts.

## The DType Class

The `cuda.tile.DType` class describes the type of elements stored in arrays, tiles, and used in operations. Understanding this class is essential for type-safe and efficient cuTile programming.

### DType Properties

**bitwidth**: The number of bits used to store each element of the data type.

```python
@ct.function
def dtype_bitwidth():
    int32_type = ct.int32
    float32_type = ct.float32
    
    # Access bitwidth property
    int32_bits = int32_type.bitwidth   # 32
    float32_bits = float32_type.bitwidth # 32
    
    # Can be used in compile-time calculations
    total_int_bits = int32_bits * 16  # 512 bits for 16 elements
    
    return int32_bits, float32_bits
```

**name**: The string name of the data type, useful for debugging and type introspection.

```python
@ct.function
def dtype_names():
    int_type = ct.int32
    float_type = ct.float32
    bool_type = ct.bool_
    
    # Access name property
    int_name = int_type.name    # "int32"
    float_name = float_type.name # "float32"
    bool_name = bool_type.name  # "bool"
    
    return int_name, float_name, bool_name
```

### DType Characteristics

**Immutability**: DType objects are immutable and cannot be modified after creation. This ensures type stability throughout compilation and execution.

```python
# DType objects are immutable
int32_type = ct.int32
# int32_type.bitwidth = 64  # This would raise an error
```

**Host and Tile Code Usage**: DTypes can be used in both host code and tile code, providing a unified type system across execution spaces.

```python
# Host code
def host_function():
    return ct.float32  # Can return dtype from host function

@ct.function
def tile_function():
    return ct.int32  # Can use dtype in tile code
```

**Kernel Parameters**: DTypes can be passed as kernel parameters, enabling generic kernels that work with different data types.

```python
@ct.kernel
def generic_kernel(input_array: ct.Array, output_array: ct.Array, 
                   element_type: ct.DType):
    # Can use dtype parameter for type-specific operations
    tile = ct.load(input_array)
    processed = tile.astype(element_type)
    ct.store(output_array, processed)
```

## Complete Data Type Reference

cuTile provides a comprehensive set of data types optimized for different use cases, from general-purpose computing to specialized machine learning applications.

### Boolean Types

**bool_**: 8-bit boolean type representing `True` or `False` values.

```python
@ct.function
def boolean_operations():
    # Create boolean tiles
    true_tile = ct.full((8, 8), True, dtype=ct.bool_)
    false_tile = ct.full((8, 8), False, dtype=ct.bool_)
    
    # Boolean operations
    and_result = true_tile & false_tile  # Logical AND
    or_result = true_tile | false_tile   # Logical OR
    not_result = ~true_tile              # Logical NOT
    
    # Comparison operations return boolean tiles
    int_tile = ct.full((8, 8), 5, dtype=ct.int32)
    greater_than = int_tile > 3          # All True
    less_than = int_tile < 10            # All True
    
    return and_result, or_result, not_result, greater_than, less_than
```

### Unsigned Integer Types

Unsigned integers represent non-negative values and are commonly used for indexing, counting, and bit manipulation operations.

**uint8**: 8-bit unsigned integer, range [0, 255].

```python
@ct.function
def uint8_operations():
    # Create uint8 tiles
    small_vals = ct.arange(256, dtype=ct.uint8)  # 0 to 255
    
    # Uint8 is useful for pixel data, small counters
    pixels = ct.full((32, 32), 128, dtype=ct.uint8)
    
    # Overflow wraps around
    max_val = ct.uint8(255)
    overflow = max_val + ct.uint8(1)  # Wraps to 0
    
    return small_vals, pixels, overflow
```

**uint16**: 16-bit unsigned integer, range [0, 65,535].

```python
@ct.function
def uint16_operations():
    # Create uint16 tiles
    medium_vals = ct.arange(1000, dtype=ct.uint16)
    
    # Useful for larger counters, indices
    indices = ct.arange(65536, dtype=ct.uint16)  # Full range
    
    return medium_vals, indices
```

**uint32**: 32-bit unsigned integer, range [0, 4,294,967,295].

```python
@ct.function
def uint32_operations():
    # Large unsigned integers
    large_vals = ct.full((16, 16), 1000000, dtype=ct.uint32)
    
    # Common for array indexing, memory offsets
    max_uint32 = ct.uint32(4294967295)  # Maximum value
    
    return large_vals, max_uint32
```

**uint64**: 64-bit unsigned integer, range [0, 18,446,744,073,709,551,615].

```python
@ct.function
def uint64_operations():
    # Very large unsigned integers
    huge_vals = ct.full((8, 8), 10000000000, dtype=ct.uint64)
    
    # Used for file sizes, memory addresses, large counters
    max_uint64 = ct.uint64(18446744073709551615)  # Maximum value
    
    return huge_vals, max_uint64
```

### Signed Integer Types

Signed integers represent both positive and negative values and are used for general arithmetic operations.

**int8**: 8-bit signed integer, range [-128, 127].

```python
@ct.function
def int8_operations():
    # Small signed integers
    small_signed = ct.full((16, 16), -64, dtype=ct.int8)
    
    # Range limits
    min_val = ct.int8(-128)
    max_val = ct.int8(127)
    
    # Overflow wraps around
    overflow = max_val + ct.int8(1)  # Wraps to -128
    
    return small_signed, min_val, max_val, overflow
```

**int16**: 16-bit signed integer, range [-32,768, 32,767].

```python
@ct.function
def int16_operations():
    # Medium signed integers
    medium_signed = ct.arange(-1000, 1000, dtype=ct.int16)
    
    # Audio data often uses int16
    audio_sample = ct.int16(0)  # Silence
    
    return medium_signed, audio_sample
```

**int32**: 32-bit signed integer, range [-2,147,483,648, 2,147,483,647].

```python
@ct.function
def int32_operations():
    # Standard integer type
    standard_int = ct.full((32, 32), 42, dtype=ct.int32)
    
    # Loop counters, array indices
    counter = ct.arange(1000000, dtype=ct.int32)
    
    # Common arithmetic
    result = standard_int * 1000
    
    return standard_int, counter, result
```

**int64**: 64-bit signed integer, range [-9,223,372,036,854,775,808, 9,223,372,036,854,775,807].

```python
@ct.function
def int64_operations():
    # Large signed integers
    large_signed = ct.full((16, 16), 1000000000, dtype=ct.int64)
    
    # Timestamps, large counters
    timestamp = ct.int64(1640000000000)
    
    # Precision arithmetic
    precise_result = large_signed * 1000
    
    return large_signed, timestamp, precise_result
```

### Floating-Point Types

Floating-point types represent real numbers with varying precision and are essential for scientific computing and machine learning.

**float16**: IEEE 754 half-precision (16-bit) floating-point.

```python
@ct.function
def float16_operations():
    # Half precision for memory-constrained applications
    fp16_tile = ct.full((32, 32), 3.14, dtype=ct.float16)
    
    # Reduced precision, reduced memory usage
    values = [ct.float16(1.0), ct.float16(2.0), ct.float16(3.0)]
    
    # Useful for ML inference, mobile applications
    result = fp16_tile * ct.float16(2.0)
    
    return fp16_tile, result
```

**float32**: IEEE 754 single-precision (32-bit) floating-point.

```python
@ct.function
def float32_operations():
    # Standard floating-point type
    fp32_tile = ct.full((32, 32), 3.14159, dtype=ct.float32)
    
    # General-purpose scientific computing
    values = ct.arange(0.0, 1.0, 0.1, dtype=ct.float32)
    
    # Common operations
    result = ct.sqrt(fp32_tile) + ct.exp(fp32_tile)
    
    return fp32_tile, values, result
```

**float64**: IEEE 754 double-precision (64-bit) floating-point.

```python
@ct.function
def float64_operations():
    # Double precision for high accuracy
    fp64_tile = ct.full((16, 16), 2.718281828459045, dtype=ct.float64)
    
    # Financial calculations, scientific research
    precise_value = ct.float64(1.7976931348623157e+308)
    
    # High-precision arithmetic
    result = fp64_tile / ct.float64(3.0)
    
    return fp64_tile, precise_value, result
```

### Machine Learning Floating-Point Types

Specialized floating-point formats optimized for machine learning workloads.

**bfloat16**: Brain floating-point format with 1 sign bit, 8 exponent bits, and 7 mantissa bits.

```python
@ct.function
def bfloat16_operations():
    # BFloat16 for ML training
    bf16_tile = ct.full((32, 32), 1.0, dtype=ct.bfloat16)
    
    # Same dynamic range as float32, reduced precision
    weights = ct.full((128, 128), 0.5, dtype=ct.bfloat16)
    
    # Common in deep learning frameworks
    result = weights * ct.bfloat16(2.0)
    
    return bf16_tile, weights, result
```

**tfloat32**: TensorFloat-32 format with 1 sign bit, 8 exponent bits, and 10 mantissa bits (stored in 32-bit container).

```python
@ct.function
def tfloat32_operations():
    # TF32 for accelerated ML on Ampere GPUs
    tf32_tile = ct.full((32, 32), 1.0, dtype=ct.tfloat32)
    
    # NVIDIA format for ML acceleration
    activations = ct.full((64, 64), 0.1, dtype=ct.tfloat32)
    
    # Used in tensor core operations
    result = activations * ct.tfloat32(10.0)
    
    return tf32_tile, activations, result
```

**float8_e4m3fn**: 8-bit floating-point with 1 sign bit, 4 exponent bits, and 3 mantissa bits (FN variant).

```python
@ct.function
def float8_e4m3fn_operations():
    # FP8 E4M3 for ML inference
    fp8_tile = ct.full((32, 32), 1.0, dtype=ct.float8_e4m3fn)
    
    # Maximum compression for ML models
    weights = ct.full((128, 128), 0.25, dtype=ct.float8_e4m3fn)
    
    # Used in quantized inference
    result = weights * ct.float8_e4m3fn(4.0)
    
    return fp8_tile, weights, result
```

**float8_e5m2**: 8-bit floating-point with 1 sign bit, 5 exponent bits, and 2 mantissa bits.

```python
@ct.function
def float8_e5m2_operations():
    # FP8 E5M2 for ML training
    fp8_tile = ct.full((32, 32), 1.0, dtype=ct.float8_e5m2)
    
    # Wider dynamic range than E4M3
    gradients = ct.full((64, 64), 0.01, dtype=ct.float8_e5m2)
    
    # Used in training scenarios
    result = gradients * ct.float8_e5m2(100.0)
    
    return fp8_tile, gradients, result
```

## Data Type Characteristics

### Numeric vs Arithmetic Types

cuTile distinguishes between **numeric types** (which can store values) and **arithmetic types** (which can participate in arithmetic operations).

**Numeric Types**: All types that can represent numeric values, including boolean and all integer/float types.

**Arithmetic Types**: Subset of numeric types that support arithmetic operations (excludes boolean in many contexts).

```python
@ct.function
def numeric_vs_arithmetic():
    # Numeric types
    bool_val = ct.bool_(True)         # Numeric but not arithmetic
    int_val = ct.int32(42)            # Numeric and arithmetic
    float_val = ct.float32(3.14)      # Numeric and arithmetic
    
    # Arithmetic operations
    int_result = int_val * 2          # Works
    float_result = float_val + 1.0    # Works
    # bool_result = bool_val * 2       # May not work as expected
    
    return int_result, float_result
```

## Arithmetic Promotion Rules

When operations involve operands of different types, cuTile applies arithmetic promotion rules to determine the result type. This process ensures type safety and computational consistency.

### Five-Step Promotion Process

#### Step 1: Classification into Categories

Operands are classified into one of three categories in order of precedence:

1. **Floating-point**: All floating-point types (float16, float32, float64, bfloat16, tfloat32, float8_e4m3fn, float8_e5m2)
2. **Integral**: All signed and unsigned integer types
3. **Boolean**: The bool_ type

```python
@ct.function
def category_classification():
    # Floating-point operands
    fp16_val = ct.float16(1.0)
    fp32_val = ct.float32(2.0)
    bf16_val = ct.bfloat16(3.0)
    
    # Integral operands
    int32_val = ct.int32(42)
    uint64_val = ct.uint64(100)
    
    # Boolean operand
    bool_val = ct.bool_(True)
    
    # Category determines promotion behavior
    # Floating-point > Integral > Boolean
    
    return fp16_val, fp32_val, int32_val, bool_val
```

#### Step 2: Loosely Typed Constants Get Concrete Dtype

Loosely typed constants (numeric literals without explicit dtype) acquire a concrete dtype based on the context.

```python
@ct.function
def loose_constant_concretization():
    # Loosely typed constants
    loose_int = 42           # Infinite precision integer
    loose_float = 3.14       # Double precision float
    
    # When combined with typed operand, acquire concrete type
    int32_tile = ct.full((8, 8), 1, dtype=ct.int32)
    
    # Loose constant promotes to match tile dtype
    result = int32_tile + loose_int  # Result is int32
    
    # Float constant may determine result type
    float_result = int32_tile + loose_float  # Result may be float32
    
    return result, float_result
```

#### Step 3: Higher Category Wins

When operands are from different categories, the higher category determines the result type.

```python
@ct.function
def category_precedence():
    int_val = ct.int32(42)
    float_val = ct.float32(3.14)
    bool_val = ct.bool_(True)
    
    # Float > Int
    float_result = int_val + float_val  # Result is float32
    
    # Int > Bool
    int_result = int_val + bool_val     # Result is int32
    
    # Float > Bool
    float_result2 = float_val + bool_val  # Result is float32
    
    return float_result, int_result, float_result2
```

#### Step 4: Typed Operand Wins Over Loosely Typed

When mixing typed and loosely typed operands, the typed operand determines the result type.

```python
@ct.function
def typed_vs_loose():
    typed_int = ct.int32(42)
    typed_float = ct.float32(3.14)
    loose_int = 100
    loose_float = 2.718
    
    # Typed wins over loose
    result1 = typed_int + loose_int     # int32
    result2 = typed_float + loose_float # float32
    result3 = typed_int + loose_float   # May promote to float32
    
    return result1, result2, result3
```

#### Step 5: Use Promotion Table

When both operands are typed and in the same category, the promotion table determines the result type.

### Complete Arithmetic Promotion Table

The following table shows the result type for binary operations between all pairs of data types. "ERR" indicates unsupported combinations that will raise compilation errors.

|  | bool_ | uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | float16 | float32 | float64 | bfloat16 | tfloat32 | float8_e4m3fn | float8_e5m2 |
|--|-------|-------|--------|--------|--------|------|-------|-------|-------|---------|---------|---------|----------|----------|---------------|-------------|
| **bool_** | bool_ | uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | float16 | float32 | float64 | bfloat16 | ERR | ERR | ERR |
| **uint8** | uint8 | uint8 | uint16 | uint32 | uint64 | int16 | int16 | int32 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **uint16** | uint16 | uint16 | uint16 | uint32 | uint64 | int32 | int32 | int32 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **uint32** | uint32 | uint32 | uint32 | uint32 | uint64 | int64 | int64 | int32 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **uint64** | uint64 | uint64 | uint64 | uint64 | uint64 | ERR | ERR | int64 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **int8** | int8 | int16 | int32 | int64 | ERR | int8 | int16 | int32 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **int16** | int16 | int16 | int32 | int64 | ERR | int16 | int16 | int32 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **int32** | int32 | int32 | int32 | int32 | int64 | int32 | int32 | int32 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **int64** | int64 | int64 | int64 | int64 | int64 | int64 | int64 | int64 | int64 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **float16** | float16 | float16 | float16 | float16 | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float64 | ERR | ERR | ERR | ERR |
| **float32** | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 | ERR | ERR | ERR | ERR |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | ERR | ERR | ERR | ERR |
| **bfloat16** | bfloat16 | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | bfloat16 | ERR | ERR | ERR |
| **tfloat32** | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | tfloat32 | ERR | ERR |
| **float8_e4m3fn** | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | float8_e4m3fn | ERR |
| **float8_e5m2** | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | ERR | float8_e5m2 |

### Important Promotion Restrictions

**bfloat16 and float16 Cannot Promote**: These two 16-bit floating-point formats cannot be mixed in arithmetic operations.

```python
@ct.function
def bfloat16_float16_restriction():
    bf16_val = ct.bfloat16(1.0)
    fp16_val = ct.float16(2.0)
    
    # This will raise a compilation error
    # result = bf16_val + fp16_val  # ERR: unsupported combination
    
    # Must explicitly convert to common type
    result = ct.float32(bf16_val) + ct.float32(fp16_val)
    
    return result
```

**ML Floating-Point Types Only Self-Promote**: The specialized machine learning floating-point types (tfloat32, float8_e4m3fn, float8_e5m2) can only promote to themselves.

```python
@ct.function
def ml_types_self_promotion():
    tf32_val = ct.tfloat32(1.0)
    fp8_e4m3 = ct.float8_e4m3fn(2.0)
    fp8_e5m2 = ct.float8_e5m2(3.0)
    
    # Same type operations work
    result1 = tf32_val * tf32_val         # OK: tfloat32
    result2 = fp8_e4m3 + fp8_e4m3         # OK: float8_e4m3fn
    result3 = fp8_e5m2 * fp8_e5m2         # OK: float8_e5m2
    
    # Mixed operations require explicit conversion
    # mixed = tf32_val + fp8_e4m3         # ERR: unsupported
    mixed = ct.float32(tf32_val) + ct.float32(fp8_e4m3)  # OK
    
    return result1, result2, result3, mixed
```

## Rounding Modes

cuTile supports various rounding modes for floating-point operations, providing control over precision vs. performance trade-offs.

### Available Rounding Modes

**RN (Round to Nearest)**: Standard IEEE 754 rounding mode. Rounds to the nearest value, with ties rounded to even mantissa bits.

```python
@ct.function
def rounding_rn():
    val = ct.float32(1.234567890)
    result = ct.round(val, mode='RN')  # Standard rounding
    return result
```

**RZ (Round toward Zero)**: Rounds toward zero, effectively truncating the result.

```python
@ct.function
def rounding_rz():
    val = ct.float32(1.999)
    result = ct.round(val, mode='RZ')  # Truncates to 1.0
    return result
```

**RM (Round toward Negative Infinity)**: Rounds toward negative infinity (floor).

```python
@ct.function
def rounding_rm():
    val = ct.float32(-1.234)
    result = ct.round(val, mode='RM')  # Rounds to -2.0
    return result
```

**RP (Round toward Positive Infinity)**: Rounds toward positive infinity (ceiling).

```python
@ct.function
def rounding_rp():
    val = ct.float32(1.234)
    result = ct.round(val, mode='RP')  # Rounds to 2.0
    return result
```

**FULL**: Full precision mode, uses maximum available precision.

```python
@ct.function
def rounding_full():
    val = ct.float32(1.0 / 3.0)
    result = ct.round(val, mode='FULL')  # Maximum precision
    return result
```

**APPROX**: Approximate mode, may trade precision for performance.

```python
@ct.function
def rounding_approx():
    val = ct.float32(1.234567890)
    result = ct.round(val, mode='APPROX')  # Faster, less precise
    return result
```

**RZI (Round to Integer)**: Rounds to nearest integer value.

```python
@ct.function
def rounding_rzi():
    val = ct.float32(3.7)
    result = ct.round(val, mode='RZI')  # Rounds to 4.0
    return result
```

### Rounding Mode Applications

Rounding modes affect various floating-point operations:

```python
@ct.function
def rounding_in_operations():
    val1 = ct.float32(1.234567890)
    val2 = ct.float32(2.345678901)
    
    # Arithmetic operations with specific rounding
    sum_rn = ct.add(val1, val2, mode='RN')
    product_rz = ct.mul(val1, val2, mode='RZ')
    
    # Type conversion with rounding control
    converted = ct.float64(val1, mode='RM')
    
    return sum_rn, product_rz, converted
```

## Padding Modes

Padding modes define how out-of-bounds accesses are handled in gather/scatter operations.

### Available Padding Modes

**UNDETERMINED**: Out-of-bounds behavior is unspecified (default, fastest).

```python
@ct.function
def padding_undetermined(global_array: ct.Array, indices: ct.Tile):
    # Fast but undefined behavior for out-of-bounds
    result = ct.gather(global_array, indices, padding='UNDETERMINED')
    return result
```

**ZERO**: Out-of-bounds accesses return zero.

```python
@ct.function
def padding_zero(global_array: ct.Array, indices: ct.Tile):
    # Safe: out-of-bounds returns 0
    result = ct.gather(global_array, indices, padding='ZERO')
    return result
```

**NEG_ZERO**: Out-of-bounds accesses return negative zero (-0.0).

```python
@ct.function
def padding_neg_zero(global_array: ct.Array, indices: ct.Tile):
    # Returns -0.0 for out-of-bounds
    result = ct.gather(global_array, indices, padding='NEG_ZERO')
    return result
```

**NAN**: Out-of-bounds accesses return NaN (Not a Number).

```python
@ct.function
def padding_nan(global_array: ct.Array, indices: ct.Tile):
    # Returns NaN for out-of-bounds (floating-point only)
    result = ct.gather(global_array, indices, padding='NAN')
    return result
```

**POS_INF**: Out-of-bounds accesses return positive infinity.

```python
@ct.function
def padding_pos_inf(global_array: ct.Array, indices: ct.Tile):
    # Returns +inf for out-of-bounds
    result = ct.gather(global_array, indices, padding='POS_INF')
    return result
```

**NEG_INF**: Out-of-bounds accesses return negative infinity.

```python
@ct.function
def padding_neg_inf(global_array: ct.Array, indices: ct.Tile):
    # Returns -inf for out-of-bounds
    result = ct.gather(global_array, indices, padding='NEG_INF')
    return result
```

### Padding Mode Usage

Padding modes are essential for safe memory access patterns:

```python
@ct.function
def safe_gather(global_array: ct.Array, indices: ct.Tile):
    # Use ZERO padding for safe out-of-bounds handling
    safe_result = ct.gather(global_array, indices, padding='ZERO')
    
    # Use NAN for debugging (helps identify issues)
    debug_result = ct.gather(global_array, indices, padding='NAN')
    
    return safe_result, debug_result
```

## Type Conversion

### Explicit Type Conversion

Explicit type conversion ensures predictable behavior and avoids unintended promotions.

```python
@ct.function
def explicit_conversion():
    # Convert between types
    int_val = ct.int32(42)
    float_val = ct.float32(int_val)       # int32 -> float32
    double_val = ct.float64(float_val)    # float32 -> float64
    
    # Float to integer conversion (truncates)
    int_from_float = ct.int32(float_val)  # float32 -> int32
    
    # Between integer types
    int64_val = ct.int64(int_val)         # int32 -> int64
    
    return float_val, double_val, int_from_float, int64_val
```

### Safe Type Conversion

Safe conversion checks for overflow and range issues:

```python
@ct.function
def safe_conversion():
    # Large value
    large_val = ct.int64(10000000000)
    
    # Safe conversion (checks range)
    try:
        small_val = ct.int32(large_val)  # May overflow
    except OverflowError:
        # Handle overflow
        small_val = ct.int32(2147483647)  # Maximum int32
    
    return small_val
```

### Saturated Conversion

Saturated conversion clamps values to the target type's range:

```python
@ct.function
def saturated_conversion():
    # Value outside int8 range
    large_val = ct.int32(1000)
    
    # Saturated conversion (clamps to int8 range)
    saturated = ct.sat_cast(large_val, dtype=ct.int8)  # Returns 127
    
    return saturated
```

## Performance Considerations

### Type Selection for Performance

Choosing appropriate data types significantly impacts performance:

```python
# Fast types (aligned with hardware)
fast_types = [
    ct.int32,    # Native integer size
    ct.float32,  # Native float size
]

# Moderate types
moderate_types = [
    ct.int64,    # May be slower on some architectures
    ct.float64,  # Double precision
]

# Slower types (software emulation or conversion overhead)
slow_types = [
    ct.int8,     # May require packing/unpacking
    ct.float16,  # May require conversion
]
```

### Memory Bandwidth Considerations

Smaller types reduce memory bandwidth requirements:

```python
@ct.function
def memory_bandwidth():
    # 8x less memory usage than float32
    fp8_data = ct.full((1024, 1024), 1.0, dtype=ct.float8_e4m3fn)
    
    # 2x less memory usage than float32
    fp16_data = ct.full((1024, 1024), 1.0, dtype=ct.float16)
    
    # Standard memory usage
    fp32_data = ct.full((1024, 1024), 1.0, dtype=ct.float32)
    
    # 2x more memory usage than float32
    fp64_data = ct.full((1024, 1024), 1.0, dtype=ct.float64)
    
    return fp8_data, fp16_data, fp32_data, fp64_data
```

### Computational Precision vs Speed

Trade-off between precision and computational speed:

```python
@ct.function
def precision_speed_tradeoff():
    # Fast but less precise
    fp16_result = ct.float16(1.0) / ct.float16(3.0)
    
    # Balanced
    fp32_result = ct.float32(1.0) / ct.float32(3.0)
    
    # Precise but slower
    fp64_result = ct.float64(1.0) / ct.float64(3.0)
    
    return fp16_result, fp32_result, fp64_result
```

## Best Practices

### Type Selection Guidelines

1. **Use appropriate precision**: Choose the lowest precision that meets accuracy requirements.
2. **Consider hardware capabilities**: Match types to hardware features (e.g., tensor cores).
3. **Minimize type conversions**: Reduce conversions between different types.

```python
@ct.function
def best_practice_types():
    # For machine learning inference
    weights = ct.full((128, 128), 1.0, dtype=ct.bfloat16)
    activations = ct.full((128, 128), 1.0, dtype=ct.bfloat16)
    
    # For scientific computing
    scientific_data = ct.full((256, 256), 1.0, dtype=ct.float64)
    
    # For indexing
    indices = ct.arange(1000000, dtype=ct.int32)
    
    return weights, activations, scientific_data, indices
```

### Safe Arithmetic Operations

1. **Be aware of promotion rules**: Understand how types promote in mixed operations.
2. **Use explicit conversions**: Avoid implicit conversions when precision matters.
3. **Check for overflow**: Be cautious with integer arithmetic near range limits.

```python
@ct.function
def safe_arithmetic():
    # Explicit conversion for precision control
    int_val = ct.int32(42)
    float_val = ct.float32(int_val)
    
    # Avoid unintended promotions
    result = float_val * ct.float32(2.0)  # Explicit float operand
    
    return result
```

### Machine Learning Type Selection

1. **Training**: Use bfloat16 or float32 for gradients.
2. **Inference**: Use bfloat16, float16, or float8 for quantized models.
3. **Accumulation**: Use float32 for accumulation even with lower precision inputs.

```python
@ct.function
def ml_type_patterns():
    # Training pattern
    inputs = ct.full((64, 64), 1.0, dtype=ct.float32)
    weights = ct.full((64, 64), 1.0, dtype=ct.bfloat16)
    gradients = ct.full((64, 64), 1.0, dtype=ct.float32)
    
    # Inference pattern
    quantized_weights = ct.full((64, 64), 1.0, dtype=ct.float8_e4m3fn)
    quantized_inputs = ct.full((64, 64), 1.0, dtype=ct.float8_e4m3fn)
    
    return inputs, weights, gradients, quantized_weights, quantized_inputs
```

## Conclusion

Understanding cuTile's data type system is essential for writing efficient and correct programs. The key takeaways are:

- Data types determine storage, precision, and promotion behavior
- Arithmetic promotion follows a well-defined five-step process
- Specialized ML floating-point types have specific promotion restrictions
- Rounding and padding modes provide control over precision and safety
- Type selection significantly impacts performance and memory usage
- Explicit type conversion ensures predictable behavior

Mastering these concepts enables you to write cuTile programs that are both efficient and correct, leveraging the full capabilities of the hardware while maintaining type safety and numerical accuracy.
