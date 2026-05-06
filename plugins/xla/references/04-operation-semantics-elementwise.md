# XLA Reference - Chapter 4: Operation Semantics - Element-wise Operations

This reference provides comprehensive documentation for all element-wise (unary) XLA operations. Element-wise operations apply a scalar function independently to each element of the input array, producing an output array of the same shape. The output element at each position depends only on the input element at the same position.

---

## 4.1 General Semantics

All element-wise operations in XLA share the following properties:

- **Shape preservation**: The output shape is identical to the input shape.
- **Element independence**: Each output element `result[i]` depends only on the corresponding input element `input[i]`.
- **Type constraints**: The input element type must be compatible with the operation. For example, trigonometric operations require floating-point or complex types; bitwise operations require integer types.
- **Fusion friendliness**: Element-wise operations are the most common candidates for fusion with other operations.

### Function Signature Convention

All unary element-wise operations follow this general pattern:

```
result_shape = opcode(input_shape)
```

Where:
- `input_shape` must be an array shape (not a tuple, token, or opaque).
- `result_shape` has the same dimensions as `input_shape`.
- The result element type may differ from the input element type for some operations.

### StableHLO Cross-Reference

All element-wise operations described in this chapter have direct counterparts in the StableHLO opset. The mapping is typically one-to-one, with the StableHLO operation having the same name. Where there are differences, they are noted in the individual operation descriptions. The StableHLO specification can be found in the `openxla/stablehlo` repository.

---

## 4.2 Abs

### Function Signature

```
result = abs(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of S, U, F, or C type | The input array |

### Semantics

Computes the absolute value of each element. For real-valued types (signed integers, unsigned integers, floating-point), the result is the magnitude of the input value. For complex types, the result is the magnitude of the complex number: `sqrt(real^2 + imag^2)`, returned as a real-valued type.

| Input Type | Output Type | Formula |
|------------|-------------|---------|
| Signed integer (S) | Same | `x >= 0 ? x : -x` |
| Unsigned integer (U) | Same | `x` (identity, since unsigned values are non-negative) |
| Floating-point (F) | Same | `|x|` (IEEE 754 absolute value) |
| Complex (C64) | F32 | `sqrt(real(x)^2 + imag(x)^2)` |
| Complex (C128) | F64 | `sqrt(real(x)^2 + imag(x)^2)` |

### Special Values

| Input | Output |
|-------|--------|
| `-0.0` | `+0.0` |
| `-inf` | `+inf` |
| `+inf` | `+inf` |
| `NaN` | `NaN` |

### Examples

```
// Integer absolute value
input = s32[3] constant({-3, 0, 7})
result = s32[3] abs(input)   // {3, 0, 7}

// Floating-point absolute value
input = f32[4] constant({-1.5, 0.0, 3.14, -inf})
result = f32[4] abs(input)   // {1.5, 0.0, 3.14, +inf}

// Complex absolute value (magnitude)
input = c64[2] constant({(3.0, 4.0), (0.0, 0.0)})
result = f32[2] abs(input)   // {5.0, 0.0}
```

### StableHLO Cross-Reference

StableHLO `abs` operation: identical semantics.

---

## 4.3 Cbrt

### Function Signature

```
result = cbrt(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F type | The input array (floating-point only) |

### Semantics

Computes the cube root of each element: `result[i] = input[i]^(1/3)`.

The cube root function is defined for all real numbers, including negative values, and preserves the sign of the input:
- `cbrt(x) < 0` if `x < 0`
- `cbrt(x) = 0` if `x = 0`
- `cbrt(x) > 0` if `x > 0`

### Special Values

| Input | Output |
|-------|--------|
| `-0.0` | `-0.0` |
| `+0.0` | `+0.0` |
| `-inf` | `-inf` |
| `+inf` | `+inf` |
| `NaN` | `NaN` |

### Examples

```
input = f32[4] constant({-8.0, -1.0, 0.0, 27.0})
result = f32[4] cbrt(input)   // {-2.0, -1.0, 0.0, 3.0}

input = f32[2] constant({1.0, 0.125})
result = f32[2] cbrt(input)   // {1.0, 0.5}
```

### StableHLO Cross-Reference

StableHLO `cbrt` operation: identical semantics.

---

## 4.4 Ceil

### Function Signature

```
result = ceil(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F type | The input array (floating-point only) |

### Semantics

Computes the ceiling of each element -- the smallest integer value greater than or equal to the input. Returns a floating-point result (not an integer type).

`ceil(x)` rounds toward positive infinity.

### Special Values

| Input | Output |
|-------|--------|
| `-1.5` | `-1.0` |
| `-0.5` | `0.0` (note: rounds toward +inf) |
| `0.0` | `0.0` |
| `0.5` | `1.0` |
| `1.0` | `1.0` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `-inf` |

### Examples

```
input = f32[4] constant({-2.7, -0.3, 1.2, 3.0})
result = f32[4] ceil(input)   // {-2.0, 0.0, 2.0, 3.0}
```

### StableHLO Cross-Reference

StableHLO `ceil` operation: identical semantics.

---

## 4.5 Cos

### Function Signature

```
result = cos(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array (angle in radians) |

### Semantics

Computes the cosine of each element. The input is interpreted as an angle in radians. For complex inputs, computes the complex cosine.

For real inputs: `cos(x) = cos(x)` where `x` is in radians.

For complex inputs: `cos(z) = (e^(iz) + e^(-iz)) / 2`.

### Accuracy

The result accuracy parameter may be specified to control the precision of the computation:
- Default: standard library precision (typically within 1-4 ULP of the mathematical result).
- Fast-math mode: may use less accurate approximations for improved performance.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `1.0` |
| `pi/2` | `0.0` (approximately) |
| `pi` | `-1.0` (approximately) |
| `NaN` | `NaN` |
| `+inf` | `NaN` |
| `-inf` | `NaN` |

### Examples

```
input = f32[3] constant({0.0, 1.57079632679, 3.14159265359})
result = f32[3] cos(input)   // {1.0, 0.0, -1.0} (approximately)
```

### StableHLO Cross-Reference

StableHLO `cosine` operation: identical semantics (note: StableHLO uses `cosine` as the operation name).

---

## 4.6 Cosh

### Function Signature

```
result = cosh(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes the hyperbolic cosine of each element.

For real inputs: `cosh(x) = (e^x + e^(-x)) / 2`.

For complex inputs: `cosh(z) = (e^z + e^(-z)) / 2`.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `1.0` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `+inf` |

Note: `cosh` is an even function: `cosh(-x) = cosh(x)`.

### Examples

```
input = f32[3] constant({0.0, 1.0, -1.0})
result = f32[3] cosh(input)   // {1.0, 1.54308..., 1.54308...}
```

### StableHLO Cross-Reference

StableHLO `cosh` operation: identical semantics.

---

## 4.7 Exp

### Function Signature

```
result = exp(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array (exponent) |

### Semantics

Computes the natural exponential (base-e exponential) of each element: `result[i] = e^(input[i])`.

For complex inputs: `exp(z) = e^(real(z)) * (cos(imag(z)) + i*sin(imag(z)))`.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `1.0` |
| `1.0` | `e ≈ 2.71828...` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `0.0` |

### Examples

```
input = f32[4] constant({0.0, 1.0, -1.0, 2.0})
result = f32[4] exp(input)   // {1.0, 2.71828..., 0.36787..., 7.38905...}
```

### StableHLO Cross-Reference

StableHLO `exponential` operation: identical semantics (note: StableHLO uses `exponential` as the operation name).

---

## 4.8 Expm1

### Function Signature

```
result = expm1(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes `e^x - 1` for each element. This function provides better numerical accuracy than computing `exp(x) - 1` for values of `x` near zero, where the subtraction would lose precision due to catastrophic cancellation.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `-1.0` |

### Examples

```
input = f32[3] constant({0.0, 1e-15, -1e-15})
result = f32[3] expm1(input)   // {0.0, 1.00000000000000e-15, -9.99999999999999e-16}
// Compare with exp(input) - 1:
// exp(1e-15) - 1 ≈ 0.0 due to floating-point precision loss
// expm1(1e-15) ≈ 1e-15 with full precision
```

### StableHLO Cross-Reference

StableHLO `exponential_minus_one` operation: identical semantics (note: StableHLO uses `exponential_minus_one` as the operation name).

---

## 4.9 Floor

### Function Signature

```
result = floor(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F type | The input array (floating-point only) |

### Semantics

Computes the floor of each element -- the largest integer value less than or equal to the input. Returns a floating-point result.

`floor(x)` rounds toward negative infinity.

### Special Values

| Input | Output |
|-------|--------|
| `-2.7` | `-3.0` |
| `-0.3` | `-1.0` |
| `0.0` | `0.0` |
| `0.5` | `0.0` |
| `1.0` | `1.0` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `-inf` |

### Examples

```
input = f32[4] constant({-2.7, -0.3, 1.2, 3.0})
result = f32[4] floor(input)   // {-3.0, -1.0, 1.0, 3.0}
```

### StableHLO Cross-Reference

StableHLO `floor` operation: identical semantics.

---

## 4.10 Log

### Function Signature

```
result = log(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes the natural logarithm (base-e logarithm) of each element: `result[i] = ln(input[i])`.

For real inputs, the natural logarithm is defined only for positive values. For negative inputs, the result is `NaN`.

For complex inputs: `log(z) = ln(|z|) + i * arg(z)` where `arg(z)` is the principal argument.

### Special Values

| Input | Output |
|-------|--------|
| `1.0` | `0.0` |
| `e` | `1.0` |
| `0.0` | `-inf` |
| `-0.0` | `-inf` |
| `-1.0` | `NaN` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |

### Examples

```
input = f32[3] constant({1.0, 2.71828, 10.0})
result = f32[3] log(input)   // {0.0, 1.0, 2.30258...}
```

### StableHLO Cross-Reference

StableHLO `log` operation: identical semantics.

---

## 4.11 Log1p

### Function Signature

```
result = log1p(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes `ln(1 + x)` for each element. This function provides better numerical accuracy than computing `log(1 + x)` for values of `x` near zero, where the addition would lose precision.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `-1.0` | `-inf` |
| `< -1.0` | `NaN` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `NaN` |

### Examples

```
input = f32[3] constant({0.0, 1e-15, -1e-15})
result = f32[3] log1p(input)   // {0.0, 9.99999999999999e-16, -1.00000000000000e-15}
// Compare with log(1 + input):
// log(1 + 1e-15) ≈ 0.0 due to floating-point precision loss
// log1p(1e-15) ≈ 1e-15 with full precision
```

### StableHLO Cross-Reference

StableHLO `log_plus_one` operation: identical semantics (note: StableHLO uses `log_plus_one` as the operation name).

---

## 4.12 Logistic

### Function Signature

```
result = logistic(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes the logistic (sigmoid) function of each element: `result[i] = 1 / (1 + exp(-input[i]))`.

This is the standard sigmoid activation function used in neural networks. For real inputs, the result is always in the range `(0, 1)`.

The computation is implemented as `logistic(x) = 1 / (1 + exp(-x))` but may use a more numerically stable formulation internally.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.5` |
| `+inf` | `1.0` |
| `-inf` | `0.0` |
| `NaN` | `NaN` |
| Large positive | `1.0` (approximately) |
| Large negative | `0.0` (approximately) |

### Examples

```
input = f32[4] constant({-10.0, 0.0, 1.0, 10.0})
result = f32[4] logistic(input)   // {~0.0, 0.5, 0.73105..., ~1.0}
```

### StableHLO Cross-Reference

StableHLO `logistic` operation: identical semantics.

---

## 4.13 Negate

### Function Signature

```
result = negate(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of S, F, or C type | The input array |

### Semantics

Computes the arithmetic negation of each element: `result[i] = -input[i]`.

For signed integers: two's complement negation.
For floating-point: IEEE 754 negation (flips the sign bit).
For complex: negates both real and imaginary parts.

Note: Not supported for unsigned integer types (negating an unsigned value is undefined).

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `-0.0` |
| `-0.0` | `0.0` |
| `+inf` | `-inf` |
| `-inf` | `+inf` |
| `NaN` | `NaN` (sign may or may not change, per IEEE 754) |

### Examples

```
input = s32[3] constant({-5, 0, 3})
result = s32[3] negate(input)   // {5, 0, -3}

input = f32[3] constant({-1.5, 0.0, 2.7})
result = f32[3] negate(input)   // {1.5, -0.0, -2.7}
```

### StableHLO Cross-Reference

StableHLO `negate` operation: identical semantics.

---

## 4.14 Sign

### Function Signature

```
result = sign(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of S, F, or C type | The input array |

### Semantics

Computes the sign function (signum) of each element:

For real types (signed integers, floating-point):
- `sign(x) = -1` if `x < 0`
- `sign(x) = 0` if `x == 0`
- `sign(x) = 1` if `x > 0`

For complex types:
- `sign(z) = z / |z|` (unit complex number in the direction of z)
- `sign(0) = 0`

### Special Values

| Input | Output |
|-------|--------|
| `-0.0` | `0.0` |
| `+0.0` | `0.0` |
| `NaN` | `NaN` |
| `+inf` | `1.0` |
| `-inf` | `-1.0` |

### Examples

```
input = f32[5] constant({-3.7, -0.0, 0.0, 0.5, 100.0})
result = f32[5] sign(input)   // {-1.0, 0.0, 0.0, 1.0, 1.0}
```

### StableHLO Cross-Reference

StableHLO `sign` operation: identical semantics.

---

## 4.15 Sin

### Function Signature

```
result = sin(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array (angle in radians) |

### Semantics

Computes the sine of each element. The input is interpreted as an angle in radians.

For real inputs: `sin(x)` where `x` is in radians.
For complex inputs: `sin(z) = (e^(iz) - e^(-iz)) / (2i)`.

### Accuracy

When the result accuracy parameter is specified, XLA may use faster but less accurate approximations. Without this parameter, the standard library precision is used.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `pi/2` | `1.0` (approximately) |
| `pi` | `0.0` (approximately) |
| `NaN` | `NaN` |
| `+inf` | `NaN` |
| `-inf` | `NaN` |

### Examples

```
input = f32[3] constant({0.0, 1.57079632679, 3.14159265359})
result = f32[3] sin(input)   // {0.0, 1.0, 0.0} (approximately)
```

### StableHLO Cross-Reference

StableHLO `sine` operation: identical semantics (note: StableHLO uses `sine` as the operation name).

---

## 4.16 Sinh

### Function Signature

```
result = sinh(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes the hyperbolic sine of each element.

For real inputs: `sinh(x) = (e^x - e^(-x)) / 2`.

For complex inputs: `sinh(z) = (e^z - e^(-z)) / 2`.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `-inf` |

Note: `sinh` is an odd function: `sinh(-x) = -sinh(x)`.

### Examples

```
input = f32[3] constant({0.0, 1.0, -1.0})
result = f32[3] sinh(input)   // {0.0, 1.17520..., -1.17520...}
```

### StableHLO Cross-Reference

StableHLO `sinh` operation: identical semantics.

---

## 4.17 Sqrt

### Function Signature

```
result = sqrt(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes the square root of each element: `result[i] = sqrt(input[i])`.

For real inputs, `sqrt` is defined only for non-negative values. For negative inputs, the result is `NaN`. Use `sqrt(complex(x))` for complex square roots of negative real values.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `1.0` | `1.0` |
| `4.0` | `2.0` |
| `-1.0` | `NaN` |
| `NaN` | `NaN` |
| `+inf` | `+inf` |

### Examples

```
input = f32[4] constant({0.0, 1.0, 4.0, 9.0})
result = f32[4] sqrt(input)   // {0.0, 1.0, 2.0, 3.0}
```

### StableHLO Cross-Reference

StableHLO `sqrt` operation: identical semantics.

---

## 4.18 Tan

### Function Signature

```
result = tan(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array (angle in radians) |

### Semantics

Computes the tangent of each element: `tan(x) = sin(x) / cos(x)`.

For real inputs: `tan(x)` where `x` is in radians. The function has poles at `x = (n + 0.5) * pi` for integer `n`, where the result approaches positive or negative infinity.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `pi/4` | `1.0` (approximately) |
| `pi/2` | Large value (approaching infinity) |
| `NaN` | `NaN` |
| `+inf` | `NaN` |
| `-inf` | `NaN` |

### Examples

```
input = f32[3] constant({0.0, 0.78539816339, 1.0})
result = f32[3] tan(input)   // {0.0, 1.0 (approx), 1.55740...}
```

### StableHLO Cross-Reference

StableHLO `tan` operation: identical semantics.

---

## 4.19 Tanh

### Function Signature

```
result = tanh(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F or C type | The input array |

### Semantics

Computes the hyperbolic tangent of each element: `tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`.

For real inputs, the result is always in the range `(-1, 1)`.

This is one of the most commonly used activation functions in neural networks (historically, though ReLU and GELU are now more common).

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `+inf` | `1.0` |
| `-inf` | `-1.0` |
| `NaN` | `NaN` |
| Large positive | `1.0` (approximately) |
| Large negative | `-1.0` (approximately) |

### Examples

```
input = f32[4] constant({-10.0, -1.0, 0.0, 1.0})
result = f32[4] tanh(input)   // {-1.0 (approx), -0.76159..., 0.0, 0.76159...}
```

### StableHLO Cross-Reference

StableHLO `tanh` operation: identical semantics.

---

## 4.20 Round

### Function Signature

```
result = round(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of F type | The input array (floating-point only) |

### Semantics

Computes the rounding of each element to the nearest integer using banker's rounding (round half to even). This is also known as "round half to even" or "commercial rounding."

- If the fractional part is less than 0.5, rounds toward zero.
- If the fractional part is greater than 0.5, rounds away from zero.
- If the fractional part is exactly 0.5, rounds to the nearest even integer.

### Special Values

| Input | Output |
|-------|--------|
| `0.0` | `0.0` |
| `0.5` | `0.0` (rounds to even) |
| `1.5` | `2.0` (rounds to even) |
| `2.5` | `2.0` (rounds to even) |
| `3.5` | `4.0` (rounds to even) |
| `-0.5` | `0.0` (rounds to even) |
| `-1.5` | `-2.0` (rounds to even) |
| `NaN` | `NaN` |
| `+inf` | `+inf` |
| `-inf` | `-inf` |

### Examples

```
input = f32[6] constant({0.3, 0.5, 0.7, 1.5, 2.5, -1.5})
result = f32[6] round(input)   // {0.0, 0.0, 1.0, 2.0, 2.0, -2.0}
```

### StableHLO Cross-Reference

StableHLO `round_nearest_even` operation: identical semantics (note: StableHLO uses `round_nearest_even` as the operation name).

---

## 4.21 Not (Bitwise Complement)

### Function Signature

```
result = not(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of S, U, or PRED type | The input array (integer or boolean) |

### Semantics

Computes the bitwise NOT (complement) of each element.

For integer types: `result[i] = ~input[i]` (inverts all bits).
For boolean (PRED) type: `result[i] = !input[i]` (logical NOT).

### Examples

```
// Bitwise NOT on unsigned integers
input = u8[3] constant({0b00001111, 0b11110000, 0b10101010})
result = u8[3] not(input)   // {0b11110000, 0b00001111, 0b01010101}

// Logical NOT on booleans
input = pred[3] constant({true, false, true})
result = pred[3] not(input)   // {false, true, false}
```

### StableHLO Cross-Reference

StableHLO `not` operation: identical semantics. Note that StableHLO distinguishes between `not` for booleans and bitwise complement for integers; in XLA HLO, `not` handles both.

---

## 4.22 Clz (Count Leading Zeros)

### Function Signature

```
result = clz(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of S or U type | The input array (integer only) |

### Semantics

Counts the number of leading zero bits in each element. For an N-bit integer type, the result is in the range `[0, N]`.

For unsigned integers, this is the count of consecutive zero bits starting from the most significant bit (MSB). For signed integers, the value is interpreted as its unsigned bit pattern.

Special cases:
- If the input is 0, the result is the bit width of the type (e.g., 32 for `s32`/`u32`).
- If the MSB is 1, the result is 0.

### Examples

```
// For u8 (8-bit unsigned):
input = u8[4] constant({0, 1, 128, 255})
result = u8[4] clz(input)   // {8, 7, 0, 0}

// For u32 (32-bit unsigned):
input = u32[3] constant({0, 1, 65536})
result = u32[3] clz(input)   // {32, 31, 16}
```

### StableHLO Cross-Reference

StableHLO `count_leading_zeros` operation: identical semantics (note: StableHLO uses `count_leading_zeros` as the operation name).

---

## 4.23 PopulationCount

### Function Signature

```
result = population_count(input)
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `input` | Array of S or U type | The input array (integer only) |

### Semantics

Counts the number of set bits (1-bits, also called the "population count" or "Hamming weight") in each element.

For unsigned integers, this is the sum of all bit values. For signed integers, the value is interpreted as its unsigned bit pattern.

### Examples

```
// For u8 (8-bit unsigned):
input = u8[4] constant({0, 1, 15, 255})
result = u8[4] population_count(input)   // {0, 1, 4, 8}

// For u32 (32-bit unsigned):
input = u32[3] constant({0, 1, 0xAAAAAAAA})
result = u32[3] population_count(input)   // {0, 1, 16}
```

### StableHLO Cross-Reference

StableHLO `population_count` operation: identical semantics.

---

## 4.24 Result Accuracy Parameter

Several of the operations described above support a **result accuracy** parameter (also known as the `result_type` or `accuracy` attribute). This parameter controls the precision of the computation:

### Accuracy Levels

| Level | Description |
|-------|-------------|
| Default | Use standard library precision (typically within 1-4 ULP). |
| `fast` | Use faster but less accurate approximations (may sacrifice ULP accuracy for speed). |
| `high` | Use higher precision internally for improved accuracy. |

### Operations Supporting Result Accuracy

The following operations may support configurable result accuracy:
- Trigonometric functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic functions: `sinh`, `cosh`, `tanh`
- Exponential and logarithmic functions: `exp`, `log`, `log1p`, `expm1`
- `sqrt`, `rsqrt`, `logistic`

### Implementation Notes

- The result accuracy parameter is a hint to the compiler, not a guarantee. The actual accuracy depends on the backend and hardware.
- On GPUs, fast-math mode may use hardware instructions like `SFU` (Special Function Unit) that have limited precision (typically 2^(-22) relative error for single precision).
- On TPUs, the BF16 precision inherently limits accuracy regardless of the accuracy parameter.
- The result accuracy parameter is typically propagated from the framework's fast-math or precision settings.

---

## 4.25 Additional Element-wise Operations

The following element-wise operations are also available in XLA but are less commonly used. They are listed here for completeness.

### Rsqrt

```
result = rsqrt(input)    // 1 / sqrt(input)
```

Computes the reciprocal square root. Equivalent to `1.0 / sqrt(input)` but typically more efficient and numerically stable as a single operation.

**StableHLO**: `rsqrt` operation.

### Sqrt

(Already documented in Section 4.17.)

### Reciprocal

```
result = reciprocal(input)    // 1 / input
```

Computes the reciprocal. Equivalent to `1.0 / input` but may be implemented more efficiently.

Note: `reciprocal` is not a standalone HLO opcode in all XLA versions. It may be represented as `divide(constant(1.0), input)`.

### IsFinite

```
result = is_finite(input)
```

Returns `true` if the element is a finite number (not infinity or NaN). Input must be floating-point. Output is PRED (boolean).

**StableHLO**: `is_finite` operation.

### Real

```
result = real(input)
```

Extracts the real part of a complex number. Input must be complex (C64 or C128). Output is the corresponding real type (F32 or F64).

**StableHLO**: `real` operation.

### Imag

```
result = imag(input)
```

Extracts the imaginary part of a complex number. Input must be complex (C64 or C128). Output is the corresponding real type (F32 or F64).

**StableHLO**: `imag` operation.

### Conj

```
result = conj(input)
```

Computes the complex conjugate. For `a + bi`, returns `a - bi`. Input and output are complex types.

**StableHLO**: `conj` operation.

### Convert

```
result = convert(input)
```

Performs element-wise type conversion. For example, converts `f32` to `f16`, `s32` to `f32`, etc. This is not a mathematical operation but a type conversion.

Conversion semantics follow the source element type and target element type:
- Float-to-float: rounding according to the target format.
- Int-to-float: exact conversion (if the target has sufficient precision) or rounding.
- Float-to-int: truncation toward zero.
- Int-to-int: truncation or sign extension.

**StableHLO**: `convert` operation.

### BitcastConvert

```
result = bitcast_convert(input)
```

Reinterprets the bit pattern of each element as a different type. Unlike `convert`, this does not perform any numerical conversion -- it simply reinterprets the bits.

For example, `bitcast_convert(f32 value)` to `u32` gives the integer whose bit pattern matches the floating-point representation.

The source and destination types must have the same bit width.

**StableHLO**: `bitcast_convert` operation.

### Count Leading Zeros / Clz

(Already documented in Section 4.22.)

### Reverse

While not strictly element-wise (it changes the position of elements), `reverse` is an array manipulation operation that reverses the order of elements along specified dimensions:

```
result = reverse(input, dimensions={dim0, dim1, ...})
```

**StableHLO**: `reverse` operation.

---

## 4.26 Summary of Element-wise Operations

| Operation | Input Types | Output Type | Category |
|-----------|-------------|-------------|----------|
| `abs` | S, U, F, C | S, U, F, F | Arithmetic |
| `cbrt` | F, C | F, C | Power |
| `ceil` | F | F | Rounding |
| `cos` | F, C | F, C | Trigonometric |
| `cosh` | F, C | F, C | Hyperbolic |
| `exp` | F, C | F, C | Exponential |
| `expm1` | F, C | F, C | Exponential |
| `floor` | F | F | Rounding |
| `log` | F, C | F, C | Logarithmic |
| `log1p` | F, C | F, C | Logarithmic |
| `logistic` | F, C | F, C | Activation |
| `negate` | S, F, C | S, F, C | Arithmetic |
| `sign` | S, F, C | S, F, C | Arithmetic |
| `sin` | F, C | F, C | Trigonometric |
| `sinh` | F, C | F, C | Hyperbolic |
| `sqrt` | F, C | F, C | Power |
| `tan` | F, C | F, C | Trigonometric |
| `tanh` | F, C | F, C | Hyperbolic/Activation |
| `round` | F | F | Rounding |
| `not` | S, U, PRED | S, U, PRED | Bitwise/Logical |
| `clz` | S, U | S, U | Bitwise |
| `population_count` | S, U | S, U | Bitwise |
| `rsqrt` | F, C | F, C | Power |
| `is_finite` | F | PRED | Classification |
| `real` | C | F | Complex |
| `imag` | C | F | Complex |
| `conj` | C | C | Complex |
| `convert` | Any | Any | Type conversion |
| `bitcast_convert` | Any (same bit width) | Any | Type reinterpretation |

---

## 4.27 Fusion of Element-wise Operations

Element-wise operations are prime candidates for fusion. When multiple element-wise operations are applied sequentially, XLA's fusion pass combines them into a single kernel:

```
// Before fusion:
sin_result = sin(input)
exp_result = exp(sin_result)
tanh_result = tanh(exp_result)

// After fusion:
fused_result = fusion(input),
    calls=fused_computation({
      sin_result = sin(param)
      exp_result = exp(sin_result)
      ROOT tanh_result = tanh(exp_result)
    })
```

The fused kernel reads each input element once, computes the entire chain of operations in registers, and writes the final output once. This eliminates all intermediate memory reads and writes, which can provide significant speedups for chains of element-wise operations.

Fusion rules for element-wise operations:
1. Any chain of element-wise operations can be fused.
2. Element-wise operations can be fused with reductions (element-wise as the "map" before the reduce).
3. Element-wise operations can be fused with broadcasts (the broadcast becomes part of the fused kernel's input handling).
4. Element-wise operations cannot be fused across non-element-wise operations (e.g., cannot fuse across a `dot` or `convolution`).
