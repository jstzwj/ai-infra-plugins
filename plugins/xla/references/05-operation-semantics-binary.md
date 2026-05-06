# XLA Reference - Chapter 5: Operation Semantics - Binary Operations

This reference provides comprehensive documentation for all binary XLA operations. Binary operations take two input arrays and produce one output array. They include arithmetic operations, comparison operations, bitwise operations, and shift operations. Many binary operations support broadcasting, allowing operands of different shapes to be combined.

---

## 5.1 General Binary Operation Semantics

All binary operations in XLA share the following general properties:

### Shape Rules

- **Same-rank operands**: When both operands have the same rank, the operation is applied element-wise. If any dimension sizes differ, InDim broadcasting applies (the differing dimension must have size 1 in one operand).
- **Different-rank operands**: When operands have different ranks, the `broadcast_dimensions` attribute must be specified (see Section 5.3 for details).
- **Scalar operands**: A scalar (rank-0) operand is implicitly broadcast to match the other operand's shape.

### Function Signature Convention

Binary operations follow this general pattern:

```
// Without broadcasting
result = opcode(lhs, rhs)

// With broadcasting (for different-rank operands)
result = opcode(lhs, rhs), broadcast_dimensions={dim0, dim1, ...}
```

### StableHLO Cross-Reference

All binary operations described in this chapter have direct counterparts in the StableHLO opset. The StableHLO operations use the same names unless noted. StableHLO uses a `broadcast_dimensions` attribute (called `broadcast_dimensions` in HLO and `lhs_broadcast_dimensions` in some StableHLO versions) to specify broadcasting.

---

## 5.2 Broadcasting in Binary Operations

### 5.2.1 Implicit Scalar Broadcasting

When one operand is a scalar, it is automatically broadcast to the shape of the other operand:

```
// Scalar + array
scalar = f32[] constant(5.0)
array = f32[3] constant({1.0, 2.0, 3.0})
result = f32[3] add(scalar, array)   // {6.0, 7.0, 8.0}
```

### 5.2.2 InDim Broadcasting (Same-Rank, Degenerate Dimensions)

When both operands have the same rank but differ in some dimensions where one has size 1, InDim broadcasting applies:

```
lhs = f32[2, 1] constant({{1.0}, {2.0}})
rhs = f32[1, 3] constant({{10.0, 20.0, 30.0}})
result = f32[2, 3] add(lhs, rhs)
// {{11.0, 21.0, 31.0},
//  {12.0, 22.0, 32.0}}
```

### 5.2.3 Explicit Broadcasting with broadcast_dimensions

When operands have different ranks, the `broadcast_dimensions` attribute specifies how the lower-rank operand maps to the higher-rank shape. This attribute is a list of integers specifying which dimensions of the higher-rank result correspond to the dimensions of the lower-rank operand.

```
// Broadcast a vector to a matrix
vector = f32[3] parameter(0)         // [1, 2, 3]
matrix = f32[2, 3] parameter(1)      // [[10, 20, 30], [40, 50, 60]]

// broadcast_dimensions={1} maps vector dim 0 to result dim 1
result = f32[2, 3] add(vector, matrix), broadcast_dimensions={1}
// [[11, 22, 33], [41, 52, 63]]
```

See Chapter 3 (Broadcasting) for a complete explanation of `broadcast_dimensions` semantics.

---

## 5.3 Add

### Function Signature

```
// Without broadcasting
result = add(lhs, rhs)

// With broadcasting
result = add(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F, or C type | Left-hand side operand |
| `rhs` | Array of S, U, F, or C type | Right-hand side operand |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise addition: `result[i] = lhs[i] + rhs[i]`.

For integer types: modular arithmetic (wraps around on overflow).
For floating-point types: IEEE 754 addition.
For complex types: component-wise addition `(a + bi) + (c + di) = (a+c) + (b+d)i`.

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| `+0.0` | `+0.0` | `+0.0` |
| `-0.0` | `+0.0` | `+0.0` |
| `+inf` | `-inf` | `NaN` |
| `NaN` | any | `NaN` |
| any | `NaN` | `NaN` |

### Examples

```
// Integer addition
lhs = s32[3] constant({1, 2, 3})
rhs = s32[3] constant({10, 20, 30})
result = s32[3] add(lhs, rhs)   // {11, 22, 33}

// Floating-point addition
lhs = f32[3] constant({1.5, 2.5, 3.5})
rhs = f32[3] constant({0.5, 0.5, 0.5})
result = f32[3] add(lhs, rhs)   // {2.0, 3.0, 4.0}

// Scalar broadcast addition
scalar = f32[] constant(10.0)
array = f32[3] constant({1.0, 2.0, 3.0})
result = f32[3] add(scalar, array)   // {11.0, 12.0, 13.0}

// Explicit broadcast: vector + matrix
vector = f32[3] constant({1.0, 2.0, 3.0})
matrix = f32[2, 3] constant({{10, 20, 30}, {40, 50, 60}})
result = f32[2, 3] add(vector, matrix), broadcast_dimensions={1}
// {{11.0, 22.0, 33.0}, {41.0, 52.0, 63.0}}
```

### StableHLO Cross-Reference

StableHLO `add` operation: identical semantics.

---

## 5.4 Sub

### Function Signature

```
result = sub(lhs, rhs)
result = sub(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F, or C type | Left-hand side (minuend) |
| `rhs` | Array of S, U, F, or C type | Right-hand side (subtrahend) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise subtraction: `result[i] = lhs[i] - rhs[i]`.

For integer types: modular arithmetic (wraps around on overflow/underflow).
For floating-point types: IEEE 754 subtraction.
For complex types: component-wise subtraction `(a + bi) - (c + di) = (a-c) + (b-d)i`.

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| `+0.0` | `+0.0` | `+0.0` |
| `-0.0` | `-0.0` | `+0.0` |
| `+0.0` | `-0.0` | `+0.0` |
| `-0.0` | `+0.0` | `-0.0` |
| `+inf` | `+inf` | `NaN` |
| `NaN` | any | `NaN` |

### Examples

```
// Integer subtraction
lhs = s32[3] constant({10, 20, 30})
rhs = s32[3] constant({1, 2, 3})
result = s32[3] sub(lhs, rhs)   // {9, 18, 27}

// Scalar subtraction (bias removal)
bias = f32[512] parameter(0)
activations = f32[128, 512] parameter(1)
result = f32[128, 512] sub(activations, bias), broadcast_dimensions={1}
```

### StableHLO Cross-Reference

StableHLO `subtract` operation: identical semantics.

---

## 5.5 Mul

### Function Signature

```
result = mul(lhs, rhs)
result = mul(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F, or C type | Left-hand side (multiplicand) |
| `rhs` | Array of S, U, F, or C type | Right-hand side (multiplier) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise multiplication: `result[i] = lhs[i] * rhs[i]`.

For integer types: modular arithmetic.
For floating-point types: IEEE 754 multiplication.
For complex types: `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| `+0.0` | `+inf` | `NaN` |
| `-0.0` | `+inf` | `NaN` |
| `+0.0` | `-inf` | `NaN` |
| `+inf` | `+inf` | `+inf` |
| `+inf` | `-inf` | `-inf` |
| `NaN` | any | `NaN` |

### Examples

```
// Element-wise multiplication
lhs = f32[3] constant({1.0, 2.0, 3.0})
rhs = f32[3] constant({4.0, 5.0, 6.0})
result = f32[3] mul(lhs, rhs)   // {4.0, 10.0, 18.0}

// Scaling by a scalar
scale = f32[] constant(2.0)
array = f32[3] constant({1.0, 2.0, 3.0})
result = f32[3] mul(scale, array)   // {2.0, 4.0, 6.0}

// Masking with boolean multiplication
mask = f32[4] constant({1.0, 0.0, 1.0, 0.0})
values = f32[4] constant({10.0, 20.0, 30.0, 40.0})
result = f32[4] mul(mask, values)   // {10.0, 0.0, 30.0, 0.0}
```

### StableHLO Cross-Reference

StableHLO `multiply` operation: identical semantics.

---

## 5.6 Div

### Function Signature

```
result = div(lhs, rhs)
result = div(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F, or C type | Left-hand side (dividend / numerator) |
| `rhs` | Array of S, U, F, or C type | Right-hand side (divisor / denominator) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise division: `result[i] = lhs[i] / rhs[i]`.

For integer types: **truncation toward zero** (not floor division). For example, `-7 / 2 = -3` (not -4).
For floating-point types: IEEE 754 division.
For complex types: complex division.

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| `+0.0` | `+0.0` | `NaN` |
| any non-zero | `+0.0` | `+inf` or `-inf` (depending on sign) |
| `+inf` | `+inf` | `NaN` |
| `+inf` | finite | `+inf` or `-inf` |
| finite | `+inf` | `+0.0` |
| `NaN` | any | `NaN` |

### Integer Division Notes

For signed integers, division truncates toward zero:
- `7 / 2 = 3`
- `-7 / 2 = -3`
- `7 / -2 = -3`
- `-7 / -2 = 3`

For unsigned integers, division always produces a non-negative result.

Division by zero for integer types produces an implementation-defined result (may trap, return zero, or return the maximum value).

### Examples

```
// Floating-point division
lhs = f32[3] constant({10.0, 20.0, 30.0})
rhs = f32[3] constant({2.0, 4.0, 5.0})
result = f32[3] div(lhs, rhs)   // {5.0, 5.0, 6.0}

// Integer division (truncation toward zero)
lhs = s32[3] constant({7, -7, -7})
rhs = s32[3] constant({2, 2, -2})
result = s32[3] div(lhs, rhs)   // {3, -3, 3}

// Normalization (divide by max)
values = f32[4] constant({1.0, 2.0, 3.0, 4.0})
max_val = f32[] constant(4.0)
result = f32[4] div(values, max_val)   // {0.25, 0.5, 0.75, 1.0}
```

### StableHLO Cross-Reference

StableHLO `divide` operation: identical semantics.

---

## 5.7 Rem

### Function Signature

```
result = rem(lhs, rhs)
result = rem(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F type | Left-hand side (dividend) |
| `rhs` | Array of S, U, F type | Right-hand side (divisor) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type. Complex types are not supported.

### Semantics

Computes the element-wise remainder: `result[i] = lhs[i] - trunc(lhs[i] / rhs[i]) * rhs[i]`.

This is the **truncation remainder** (equivalent to C/C++ `%` operator for integers):
- The result has the same sign as the dividend (lhs).
- `rem(x, y) = x - trunc(x/y) * y`

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| any | `+inf` | lhs |
| any | `-inf` | lhs |
| `+0.0` | any non-zero | `+0.0` |
| `-0.0` | any non-zero | `-0.0` |
| `NaN` | any | `NaN` |
| any | `NaN` | `NaN` |
| any | `0.0` | `NaN` |

### Examples

```
// Integer remainder
lhs = s32[4] constant({7, -7, 7, -7})
rhs = s32[4] constant({3, 3, -3, -3})
result = s32[4] rem(lhs, rhs)   // {1, -1, 1, -1}

// Floating-point remainder
lhs = f32[3] constant({5.5, -5.5, 10.0})
rhs = f32[3] constant({2.0, 2.0, 3.0})
result = f32[3] rem(lhs, rhs)   // {1.5, -1.5, 1.0}
```

### StableHLO Cross-Reference

StableHLO `remainder` operation: identical semantics.

---

## 5.8 Max

### Function Signature

```
result = max(lhs, rhs)
result = max(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F type | Left-hand side |
| `rhs` | Array of S, U, F type | Right-hand side |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes the element-wise maximum: `result[i] = max(lhs[i], rhs[i])`.

For floating-point types, if either operand is NaN, the result is the other operand. If both are NaN, the result is NaN.

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| `NaN` | non-NaN | non-NaN |
| non-NaN | `NaN` | non-NaN |
| `NaN` | `NaN` | `NaN` |
| `-0.0` | `+0.0` | `+0.0` |
| `+0.0` | `-0.0` | `+0.0` |

Note: This is the `maxNum` behavior from IEEE 754-2008 (NaN is treated as missing data).

### Examples

```
// Element-wise max
lhs = f32[4] constant({1.0, 5.0, 3.0, 7.0})
rhs = f32[4] constant({4.0, 2.0, 6.0, 1.0})
result = f32[4] max(lhs, rhs)   // {4.0, 5.0, 6.0, 7.0}

// ReLU implementation using max
values = f32[4] constant({-1.0, 2.0, -3.0, 4.0})
zero = f32[] constant(0.0)
result = f32[4] max(values, zero)   // {0.0, 2.0, 0.0, 4.0}
```

### StableHLO Cross-Reference

StableHLO `maximum` operation: identical semantics.

---

## 5.9 Min

### Function Signature

```
result = min(lhs, rhs)
result = min(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F type | Left-hand side |
| `rhs` | Array of S, U, F type | Right-hand side |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes the element-wise minimum: `result[i] = min(lhs[i], rhs[i])`.

For floating-point types, if either operand is NaN, the result is the other operand. If both are NaN, the result is NaN.

### Special Values (Floating-Point)

| lhs | rhs | Result |
|-----|-----|--------|
| `NaN` | non-NaN | non-NaN |
| non-NaN | `NaN` | non-NaN |
| `NaN` | `NaN` | `NaN` |
| `-0.0` | `+0.0` | `-0.0` |
| `+0.0` | `-0.0` | `-0.0` |

Note: This is the `minNum` behavior from IEEE 754-2008.

### Examples

```
// Element-wise min
lhs = f32[4] constant({1.0, 5.0, 3.0, 7.0})
rhs = f32[4] constant({4.0, 2.0, 6.0, 1.0})
result = f32[4] min(lhs, rhs)   // {1.0, 2.0, 3.0, 1.0}

// Clipping values to range [0, 1]
values = f32[4] constant({-0.5, 0.3, 1.5, 0.8})
clipped_low = f32[4] max(values, f32[] constant(0.0))   // {0.0, 0.3, 1.5, 0.8}
result = f32[4] min(clipped_low, f32[] constant(1.0))    // {0.0, 0.3, 1.0, 0.8}
```

### StableHLO Cross-Reference

StableHLO `minimum` operation: identical semantics.

---

## 5.10 And (Bitwise/Logical AND)

### Function Signature

```
result = and(lhs, rhs)
result = and(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, or PRED type | Left-hand side |
| `rhs` | Array of S, U, or PRED type | Right-hand side |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

For integer types: computes bitwise AND: `result[i] = lhs[i] & rhs[i]`.
For boolean (PRED) type: computes logical AND: `result[i] = lhs[i] && rhs[i]`.

### Examples

```
// Bitwise AND on unsigned integers
lhs = u8[3] constant({0xFF, 0xF0, 0x0F})
rhs = u8[3] constant({0x0F, 0xF0, 0x0F})
result = u8[3] and(lhs, rhs)   // {0x0F, 0xF0, 0x0F}

// Logical AND on booleans
lhs = pred[3] constant({true, true, false})
rhs = pred[3] constant({true, false, true})
result = pred[3] and(lhs, rhs)   // {true, false, false}

// Masking: extract lower 16 bits
value = u32[2] constant({0x12345678, 0xAABBCCDD})
mask = u32[] constant(0x0000FFFF)
result = u32[2] and(value, mask)   // {0x00005678, 0x0000CCDD}
```

### StableHLO Cross-Reference

StableHLO `and` operation: identical semantics.

---

## 5.11 Or (Bitwise/Logical OR)

### Function Signature

```
result = or(lhs, rhs)
result = or(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, or PRED type | Left-hand side |
| `rhs` | Array of S, U, or PRED type | Right-hand side |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

For integer types: computes bitwise OR: `result[i] = lhs[i] | rhs[i]`.
For boolean (PRED) type: computes logical OR: `result[i] = lhs[i] || rhs[i]`.

### Examples

```
// Bitwise OR on unsigned integers
lhs = u8[3] constant({0xF0, 0x0F, 0x00})
rhs = u8[3] constant({0x0F, 0x0F, 0xFF})
result = u8[3] or(lhs, rhs)   // {0xFF, 0x0F, 0xFF}

// Logical OR on booleans
lhs = pred[3] constant({true, false, false})
rhs = pred[3] constant({false, true, false})
result = pred[3] or(lhs, rhs)   // {true, true, false}

// Setting specific bits
value = u32[2] constant({0x00000000, 0x00000001})
flags = u32[] constant(0x80000000)
result = u32[2] or(value, flags)   // {0x80000000, 0x80000001}
```

### StableHLO Cross-Reference

StableHLO `or` operation: identical semantics.

---

## 5.12 Xor (Bitwise/Logical XOR)

### Function Signature

```
result = xor(lhs, rhs)
result = xor(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, or PRED type | Left-hand side |
| `rhs` | Array of S, U, or PRED type | Right-hand side |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

For integer types: computes bitwise XOR: `result[i] = lhs[i] ^ rhs[i]`.
For boolean (PRED) type: computes logical XOR: `result[i] = lhs[i] != rhs[i]`.

### Examples

```
// Bitwise XOR
lhs = u8[3] constant({0xFF, 0xFF, 0x00})
rhs = u8[3] constant({0xFF, 0x0F, 0xFF})
result = u8[3] xor(lhs, rhs)   // {0x00, 0xF0, 0xFF}

// Logical XOR
lhs = pred[3] constant({true, true, false})
rhs = pred[3] constant({true, false, false})
result = pred[3] xor(lhs, rhs)   // {false, true, false}

// Toggle bits
value = u32[2] constant({0x00000000, 0xFFFFFFFF})
toggle = u32[] constant(0xFFFFFFFF)
result = u32[2] xor(value, toggle)   // {0xFFFFFFFF, 0x00000000}
```

### StableHLO Cross-Reference

StableHLO `xor` operation: identical semantics.

---

## 5.13 Atan2

### Function Signature

```
result = atan2(lhs, rhs)
result = atan2(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of F or C type | Left-hand side (y coordinate) |
| `rhs` | Array of F or C type | Right-hand side (x coordinate) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes the two-argument arctangent: `result[i] = atan2(lhs[i], rhs[i])`, which returns the angle in radians from the positive x-axis to the point `(rhs[i], lhs[i])`.

The function takes `y` as the first argument and `x` as the second argument. This gives correct quadrant information:

| Quadrant | lhs (y) | rhs (x) | atan2 range |
|----------|---------|---------|-------------|
| I | positive | positive | (0, pi/2) |
| II | positive | negative | (pi/2, pi) |
| III | negative | negative | (-pi, -pi/2) |
| IV | negative | positive | (-pi/2, 0) |

### Special Values

| lhs (y) | rhs (x) | Result |
|---------|---------|--------|
| `0.0` | `>0` | `0.0` |
| `0.0` | `+0.0` | `0.0` |
| `0.0` | `-0.0` | `pi` |
| `-0.0` | `+0.0` | `-0.0` |
| `-0.0` | `-0.0` | `-pi` |
| `>0` | `0.0` | `pi/2` |
| `<0` | `0.0` | `-pi/2` |
| `NaN` | any | `NaN` |
| any | `NaN` | `NaN` |

### Examples

```
// Compute angle of vector (3, 4)
y = f32[1] constant({4.0})
x = f32[1] constant({3.0})
result = f32[1] atan2(y, x)   // {0.92729...} (approximately 53.13 degrees)

// Compute angles for multiple points
y_vals = f32[4] constant({1.0, 1.0, -1.0, -1.0})
x_vals = f32[4] constant({1.0, -1.0, 1.0, -1.0})
result = f32[4] atan2(y_vals, x_vals)
// {pi/4, 3*pi/4, -pi/4, -3*pi/4} (approximately)
```

### StableHLO Cross-Reference

StableHLO `atan2` operation: identical semantics.

---

## 5.14 Complex

### Function Signature

```
result = complex(lhs, rhs)
result = complex(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of F32 or F64 | Real part |
| `rhs` | Array of F32 or F64 | Imaginary part |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same floating-point type.

### Semantics

Constructs complex numbers from real and imaginary parts:
- If `lhs` and `rhs` are `F32`, the result type is `C64`.
- If `lhs` and `rhs` are `F64`, the result type is `C128`.

`result[i] = (lhs[i] + rhs[i] * i)` where `i` is the imaginary unit.

### Examples

```
// Construct complex numbers
real_parts = f32[3] constant({1.0, 3.0, 0.0})
imag_parts = f32[3] constant({0.0, 4.0, -2.0})
result = c64[3] complex(real_parts, imag_parts)
// {(1.0, 0.0), (3.0, 4.0), (0.0, -2.0)}

// Scalar complex number
real = f32[] constant(1.0)
imag = f32[] constant(1.0)
result = c64[] complex(real, imag)   // (1.0, 1.0) = 1+i
```

### StableHLO Cross-Reference

StableHLO `complex` operation: identical semantics.

---

## 5.15 Power

### Function Signature

```
result = power(lhs, rhs)
result = power(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F, or C type | Base |
| `rhs` | Array of S, U, F, or C type | Exponent |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise exponentiation: `result[i] = lhs[i] ^ rhs[i]`.

For integer types: `lhs ^ rhs` where `rhs` must be non-negative. Computes repeated multiplication. `0^0 = 1`.

For floating-point types: `pow(lhs, rhs)` following IEEE 754 semantics. `pow(x, y) = exp(y * log(x))`.

For complex types: complex exponentiation.

### Special Values (Floating-Point)

| lhs (base) | rhs (exp) | Result |
|------------|-----------|--------|
| `x` | `0.0` | `1.0` (including `x = 0`, `x = NaN`, `x = inf`) |
| `1.0` | `y` | `1.0` (including `y = NaN`, `y = inf`) |
| `x > 0` | `y` | `x^y` |
| `x < 0` | non-integer `y` | `NaN` |
| `0.0` | `y > 0` | `0.0` |
| `0.0` | `y < 0` | `+inf` |
| `+inf` | `y > 0` | `+inf` |
| `+inf` | `y < 0` | `0.0` |
| `NaN` | non-zero | `NaN` |

### Examples

```
// Integer power
base = s32[3] constant({2, 3, 2})
exp = s32[3] constant({10, 3, 0})
result = s32[3] power(base, exp)   // {1024, 27, 1}

// Floating-point power
base = f32[3] constant({2.0, 4.0, 9.0})
exp = f32[3] constant({0.5, 0.5, 0.5})
result = f32[3] power(base, exp)   // {1.41421..., 2.0, 3.0} (square root)

// Squaring
values = f32[3] constant({2.0, 3.0, 4.0})
two = f32[] constant(2.0)
result = f32[3] power(values, two)   // {4.0, 9.0, 16.0}
```

### StableHLO Cross-Reference

StableHLO `power` operation: identical semantics.

---

## 5.16 ShiftLeft

### Function Signature

```
result = shift_left(lhs, rhs)
result = shift_left(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S or U type | Value to shift |
| `rhs` | Array of S or U type | Number of bits to shift (unsigned interpretation) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise left shift: `result[i] = lhs[i] << rhs[i]`.

The shift amount is interpreted as an unsigned value. If the shift amount is greater than or equal to the bit width of the type, the result is 0.

This is a logical shift -- zeros are shifted in from the right.

### Examples

```
// Left shift
value = u32[3] constant({1, 1, 255})
amount = u32[3] constant({0, 4, 8})
result = u32[3] shift_left(value, amount)   // {1, 16, 65280}

// Equivalent to multiply by power of 2
value = s32[3] constant({3, 3, 3})
amount = s32[3] constant({1, 2, 3})
result = s32[3] shift_left(value, amount)   // {6, 12, 24}

// Shift by zero
value = u32[2] constant({42, 100})
amount = u32[2] constant({0, 0})
result = u32[2] shift_left(value, amount)   // {42, 100}

// Shift by >= bit width (result is 0)
value = u32[2] constant({42, 42})
amount = u32[2] constant({32, 64})
result = u32[2] shift_left(value, amount)   // {0, 0}
```

### StableHLO Cross-Reference

StableHLO `shift_left` operation: identical semantics.

---

## 5.17 ShiftRightArithmetic

### Function Signature

```
result = shift_right_arithmetic(lhs, rhs)
result = shift_right_arithmetic(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S or U type | Value to shift |
| `rhs` | Array of S or U type | Number of bits to shift (unsigned interpretation) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise arithmetic right shift: `result[i] = lhs[i] >> rhs[i]` (arithmetic shift).

Arithmetic right shift preserves the sign bit:
- For signed types: the sign bit (MSB) is replicated. Negative values remain negative.
- For unsigned types: equivalent to logical right shift (zeros shifted in from the left).

If the shift amount is greater than or equal to the bit width of the type:
- For signed types: the result is 0 or -1 (all sign bits), depending on the original sign.
- For unsigned types: the result is 0.

### Examples

```
// Arithmetic right shift on signed integers
value = s32[4] constant({16, 16, -16, -16})
amount = s32[4] constant({1, 4, 1, 4})
result = s32[4] shift_right_arithmetic(value, amount)   // {8, 1, -8, -1}

// Note: -16 >> 4 = -1 (sign extension), not 268435455
// -16 in binary (32-bit): 11111111111111111111111111110000
// >> 4:                   11111111111111111111111111111111 = -1

// Equivalent to floor division by power of 2 (for non-negative values)
value = s32[3] constant({16, 32, 64})
amount = s32[3] constant({1, 2, 3})
result = s32[3] shift_right_arithmetic(value, amount)   // {8, 8, 8}
```

### StableHLO Cross-Reference

StableHLO `shift_right_arithmetic` operation: identical semantics.

---

## 5.18 ShiftRightLogical

### Function Signature

```
result = shift_right_logical(lhs, rhs)
result = shift_right_logical(lhs, rhs), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S or U type | Value to shift |
| `rhs` | Array of S or U type | Number of bits to shift (unsigned interpretation) |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

Both operands must have the same element type.

### Semantics

Computes element-wise logical right shift: `result[i] = lhs[i] >> rhs[i]` (logical shift).

Logical right shift always shifts in zeros from the left, regardless of the sign of the value. This is the key difference from `shift_right_arithmetic`.

If the shift amount is greater than or equal to the bit width of the type, the result is 0.

### Examples

```
// Logical right shift on unsigned integers
value = u32[3] constant({16, 255, 65536})
amount = u32[3] constant({1, 4, 8})
result = u32[3] shift_right_logical(value, amount)   // {8, 15, 256}

// Logical right shift on signed integers (zeros shifted in)
value = s32[2] constant({-16, -1})
amount = s32[2] constant({4, 1})
result = s32[2] shift_right_logical(value, amount)
// -16 >> 4 (logical): 0x0FFFFFFF = 268435455
// -1 >> 1 (logical): 0x7FFFFFFF = 2147483647

// Extracting high bytes
value = u32[2] constant({0x12345678, 0xAABBCCDD})
result = u32[2] shift_right_logical(value, 16)   // {0x00001234, 0x0000AABB}
```

### StableHLO Cross-Reference

StableHLO `shift_right_logical` operation: identical semantics.

---

## 5.19 Comparison Operations

In addition to the arithmetic and bitwise operations described above, XLA provides comparison operations that are often grouped with binary operations. These include:

- **`compare(lhs, rhs, direction)`**: General comparison operation with a direction parameter.

The `direction` parameter specifies the comparison type:

| Direction | Semantics | Result if `lhs[i] < rhs[i]` | Result if `lhs[i] == rhs[i]` | Result if `lhs[i] > rhs[i]` |
|-----------|-----------|----------------------------|------------------------------|----------------------------|
| `EQ` | Equal | `false` | `true` | `false` |
| `NE` | Not equal | `true` | `false` | `true` |
| `LT` | Less than | `true` | `false` | `false` |
| `LE` | Less or equal | `true` | `true` | `false` |
| `GT` | Greater than | `false` | `false` | `true` |
| `GE` | Greater or equal | `false` | `true` | `true` |

### Function Signature

```
result = compare(lhs, rhs), direction={EQ|NE|LT|LE|GT|GE}
result = compare(lhs, rhs, direction), broadcast_dimensions={...}
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `lhs` | Array of S, U, F, PRED, or C type | Left-hand side |
| `rhs` | Array of S, U, F, PRED, or C type | Right-hand side |
| `direction` | String enum | Comparison direction: EQ, NE, LT, LE, GT, GE |
| `broadcast_dimensions` | (optional) list of int64 | Broadcasting dimension mapping |

### Result Type

The result type is always `PRED` (boolean), regardless of the input element type.

### Semantics

For floating-point comparisons:
- `+0.0 == -0.0` is `true`.
- NaN comparisons follow IEEE 754: `NaN != NaN` is `true`, all other comparisons with NaN are `false`.
- Total order: `-inf < finite < +inf`.

For complex comparisons:
- Only `EQ` and `NE` are defined for complex types.
- Complex equality: `(a + bi) == (c + di)` iff `a == c` and `b == d`.

### Examples

```
// Floating-point comparison
lhs = f32[4] constant({1.0, 2.0, 3.0, 4.0})
rhs = f32[4] constant({2.0, 2.0, 2.0, 2.0})
lt_result = pred[4] compare(lhs, rhs), direction=LT   // {true, false, false, false}
eq_result = pred[4] compare(lhs, rhs), direction=EQ   // {false, true, false, false}
ge_result = pred[4] compare(lhs, rhs), direction=GE   // {false, true, true, true}

// Integer comparison
lhs = s32[3] constant({-1, 0, 1})
rhs = s32[3] constant({0, 0, 0})
result = pred[3] compare(lhs, rhs), direction=LT   // {true, false, false}

// NaN handling
lhs = f32[2] constant({NaN, 1.0})
rhs = f32[2] constant({NaN, NaN})
eq = pred[2] compare(lhs, rhs), direction=EQ     // {false, false}
ne = pred[2] compare(lhs, rhs), direction=NE     // {true, true}
lt = pred[2] compare(lhs, rhs), direction=LT     // {false, false}
```

### StableHLO Cross-Reference

StableHLO `compare` operation: identical semantics. StableHLO uses a `comparison_direction` attribute with the same values (EQ, NE, LT, LE, GT, GE).

---

## 5.20 Summary of Binary Operations

| Operation | Input Types | Output Type | Category |
|-----------|-------------|-------------|----------|
| `add` | S, U, F, C | Same as input | Arithmetic |
| `sub` | S, U, F, C | Same as input | Arithmetic |
| `mul` | S, U, F, C | Same as input | Arithmetic |
| `div` | S, U, F, C | Same as input | Arithmetic |
| `rem` | S, U, F | Same as input | Arithmetic |
| `max` | S, U, F | Same as input | Comparison |
| `min` | S, U, F | Same as input | Comparison |
| `and` | S, U, PRED | Same as input | Bitwise/Logical |
| `or` | S, U, PRED | Same as input | Bitwise/Logical |
| `xor` | S, U, PRED | Same as input | Bitwise/Logical |
| `atan2` | F, C | Same as input | Trigonometric |
| `complex` | F32, F64 | C64, C128 | Construction |
| `power` | S, U, F, C | Same as input | Power |
| `shift_left` | S, U | Same as input | Shift |
| `shift_right_arithmetic` | S, U | Same as input | Shift |
| `shift_right_logical` | S, U | Same as input | Shift |
| `compare` | S, U, F, PRED, C | PRED | Comparison |

---

## 5.21 Broadcasting Summary for Binary Operations

### Quick Reference

| Operand Shapes | Broadcasting Mechanism | `broadcast_dimensions` Required? |
|----------------|----------------------|----------------------------------|
| Same shape | None (element-wise) | No |
| Scalar + Array | Implicit scalar broadcast | No |
| Same rank, degenerate dims | InDim broadcast | No |
| Different ranks | Explicit `broadcast_dimensions` | Yes |

### Broadcasting Decision Tree

```
Are lhs and rhs the same shape?
  YES -> No broadcasting needed.
  NO ->
    Is one operand a scalar?
      YES -> Implicit scalar broadcast.
      NO ->
        Do lhs and rhs have the same rank?
          YES ->
            Are mismatched dimensions degenerate (size 1)?
              YES -> InDim broadcast (implicit).
              NO -> ERROR: incompatible shapes.
          NO ->
            Is broadcast_dimensions specified?
              YES -> Explicit broadcast using dimensions.
              NO -> ERROR: must specify broadcast_dimensions.
```

### Performance Notes

1. **Scalar broadcast is free**: Broadcasting a scalar to any shape incurs no memory overhead. The compiler emits code that simply uses the scalar value for every output element.

2. **Degenerate broadcast is nearly free**: Broadcasting a size-1 dimension requires no memory allocation. The compiler emits code that reads the single value and reuses it.

3. **Cross-rank broadcast adds a broadcast instruction**: When `broadcast_dimensions` is used, a `broadcast` HLO instruction is created. This instruction is a good candidate for fusion with the binary operation.

4. **Fusion opportunity**: In practice, the XLA optimizer fuses binary operations with their broadcast operands. The fused kernel reads the broadcast source once and applies the binary operation without materializing the expanded array.

5. **Layout considerations**: The broadcast instruction's output layout affects memory access patterns. The layout assignment pass considers the consumer (binary operation) when assigning layouts to ensure efficient access.
