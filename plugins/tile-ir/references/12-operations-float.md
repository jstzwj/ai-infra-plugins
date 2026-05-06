# Operations: Floating Point

Tile IR contains a set of typed arithmetic operations which implement familiar arithmetic operations on floating-point types. For integer operations, see Integer Operations.

All operations are implemented in a manner that is efficient for the target architecture and device family. In most common cases this means utilizing the underlying hardware's native floating-point operations. Due to Tile IR's stability guarantees and higher-level programming model, some types on some hardware may be emulated. See Stability for more information about the stability guarantees and information about per-device behavior.

## Floating-Point Arithmetic

Standard floating-point types implement the IEEE-754 standard for floating-point arithmetic. On NVIDIA hardware, certain types are non-standard and do not implement the IEEE-754 standard. See Element Types for more details about the different floating-point types, their precision, storage, and formats.

Supports 16-bit, 32-bit, and 64-bit floating-point data types.

## Floating-Point Math

Tile IR contains a set of standard math library operations which implement familiar mathematical functions over tensors supporting 16-bit, 32-bit, and 64-bit floating-point data types.

> **Note:** 32-bit and 64-bit operations typically leverage efficient hardware-specific instructions. Some 16-bit operations are emulated using wider intermediate computations, and may not offer the same performance.

> **Warning:** There are some restrictions based on data type support which are detailed in the Type System section.

---

## `cuda_tile.absf`

Element-wise floating-point absolute value.

```
cuda_tile.absf %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The absolute value of the input tile. |

**Description:**

The absf operation computes the element-wise absolute value of the input float tile. Element-wise floating-point arithmetic operations are performed by the target architecture's native floating-point instructions. If the rounding modifier is specified, the particular rounding mode will be applied to each element of the result.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.addf`

Element-wise floating-point addition.

```
cuda_tile.addf %lhs %rhs %rounding_mode %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The sum of the input tiles. |

**Description:**

The addf operation computes the element-wise addition of two tiles with floating-point element type. The addition of individual elements is performed by the target architecture's native floating-point addition for the given element type unless otherwise specified.

**Rounding modes:**

| Mode | Description |
|------|-------------|
| `nearest_even` | Round to nearest (ties to even) |
| `zero` | Round towards zero (truncate) |
| `negative_inf` | Round towards negative infinity |
| `positive_inf` | Round towards positive infinity |
| `approx` | Approximate rounding mode |
| `full` | Full precision rounding mode |
| `nearest_int_to_zero` | Round towards zero to the nearest integer |

**addf Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |
| rounding\<nearest_even\> | yes | yes | yes | yes |
| rounding\<zero\> | yes | yes | yes* | yes* |
| rounding\<negative_inf\> | yes | yes | yes* | yes* |
| rounding\<positive_inf\> | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.atan2`

Element-wise atan2.

```
cuda_tile.atan2 %x %y
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| x | `tile<f16 \| bf16 \| f32 \| f64>` | The input x float tile. |
| y | `tile<f16 \| bf16 \| f32 \| f64>` | The input y float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The element-wise result tile. |

**Description:**

The atan2 operation calculates the principal value of the arc tangent of the ratio of first and second input arguments x / y. The quadrant of the result is determined by the signs of inputs x and y. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- x, y and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%x = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
%y = constant <f32: [1.0,  1.0, 1.0, 0.0]> : tile<4xf32>
%res = atan2 %x, %y : tile<4xf32>
```

---

## `cuda_tile.ceil`

Element-wise ceiling.

```
cuda_tile.ceil %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The ceiling of the input tile. |

**Description:**

The ceil operation computes the element-wise ceiling on the input floating-point tile. The ceiling operation rounds each element up to the largest integer value that is greater than or equal to the input value.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%result = ceil %source : tile<f32>
```

---

## `cuda_tile.cmpf`

Element-wise floating-point comparison.

```
cuda_tile.cmpf %comparison_predicate %comparison_ordering %lhs %rhs
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| comparison_predicate | `ComparisonPredicate` | The comparison predicate. |
| comparison_ordering | `ComparisonOrdering` | The comparison ordering. |
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1>` | The result of the comparison. |

**Description:**

The cmpf operation is a generic comparison for float-like types. The operands must have the same shape and type, and this type must be a float type. The result is 1 if the comparison is true and 0 otherwise. The comparison is performed element-wise.

**Comparison predicates:**

| Predicate | Description |
|-----------|-------------|
| `equal` | Equal comparison |
| `not_equal` | Not equal comparison |
| `less_than` | Less than comparison |
| `less_than_or_equal` | Less than or equal comparison |
| `greater_than` | Greater than comparison |
| `greater_than_or_equal` | Greater than or equal comparison |

**Comparison ordering:**

| Ordering | Description |
|----------|-------------|
| `unordered` | Unordered comparison |
| `ordered` | Ordered comparison |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs and rhs must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Result type has i1 element type and same shape as operands.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%lhs0 = constant <f16: 0.0> : tile<f16>
%rhs0 = constant <f16: 0.0> : tile<f16>

// Custom form of scalar "ordered equal" comparison.
%x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

%lhs1 = constant <f16: 0.0> : tile<2x2xf16>
%rhs1 = constant <f16: 0.0> : tile<2x2xf16>

// Custom form of scalar "unordered less than" comparison.
%x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>

%lhs2 = constant <f64: 0.0> : tile<2x2xf64>
%rhs2 = constant <f64: 0.0> : tile<2x2xf64>
```

---

## `cuda_tile.cosh`

Element-wise hyperbolic cosine.

```
cuda_tile.cosh %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input floating-point tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The hyperbolic cosine of the input tile. |

**Description:**

The cosh operation computes the element-wise hyperbolic cosine of the input tile with floating-point element type. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.cos`

Element-wise cosine.

```
cuda_tile.cos %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The cosine of the input tile. |

**Description:**

The cos operation computes the element-wise cosine of the input floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = cos %in : tile<4xf32>
```

---

## `cuda_tile.divf`

Element-wise floating-point division.

```
cuda_tile.divf %lhs %rhs %rounding_mode %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The dividend input floating-point tile. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The divisor input floating-point tile. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the divf operation. |

**Description:**

The divf operation computes the element-wise division of two input tiles with floating-point element types. The `approx` rounding mode implements a fast approximation of divide, computed as a multiplication by reciprocal. For |rhs| in normalized range [2^(-126), 2^(126)] the maximum ULP (Unit in the Last Place) error is 2. For 2^(126) < |rhs| < 2^(128), if lhs is infinity the operation returns NaN, otherwise 0.

The `full` rounding mode implements a relatively fast, full-range approximation that scales operands to achieve better accuracy, but is not fully IEEE 754 compliant. The maximum ulp error is 2 across the full range of inputs.

**Rounding modes:**

| Mode | Description |
|------|-------------|
| `nearest_even` | Round to nearest (ties to even) |
| `zero` | Round towards zero (truncate) |
| `negative_inf` | Round towards negative infinity |
| `positive_inf` | Round towards positive infinity |
| `approx` | Approximate rounding mode |
| `full` | Full precision rounding mode |
| `nearest_int_to_zero` | Round towards zero to the nearest integer |

**divf Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |
| approx | yes | no | no | no |
| full | yes | no | no | no |
| rounding\<nearest_even\> | yes | yes | yes* | yes* |
| rounding\<zero\> | yes | yes | yes* | yes* |
| rounding\<negative_inf\> | yes | yes | yes* | yes* |
| rounding\<positive_inf\> | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.exp2`

Element-wise power of two.

```
cuda_tile.exp2 %source %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input floating-point tile. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of raising 2 to the power of the input tile. |

**Description:**

The exp2 operation computes the element-wise power of two of the input floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**exp2 Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = exp2 %in : tile<4xf32>
```

---

## `cuda_tile.exp`

Element-wise exponential.

```
cuda_tile.exp %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The exponential of the input tile. |

**Description:**

The exp operation computes the element-wise exponential of the input floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = exp %in : tile<4xf32>
```

---

## `cuda_tile.floor`

Element-wise floor rounding.

```
cuda_tile.floor %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input tile to the floor operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the floor operation. |

**Description:**

The floor operation computes the element-wise floor on the input floating-point tile rounding each element down to the largest integer that is less than or equal to the element.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%source = constant <f32: 1.5> : tile<f32>
%result = floor %source : tile<f32>
```

---

## `cuda_tile.fma`

Floating-point fused multiply-add.

```
cuda_tile.fma %lhs %rhs %acc %rounding_mode %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |
| acc | `tile<f16 \| bf16 \| f32 \| f64>` | The accumulator operand. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The fused multiply-add of the input tiles. |

**Description:**

Takes three operands lhs, rhs and acc, returns result = lhs * rhs + acc.

**Rounding modes:**

| Mode | Description |
|------|-------------|
| `nearest_even` | Round to nearest (ties to even) |
| `zero` | Round towards zero (truncate) |
| `negative_inf` | Round towards negative infinity |
| `positive_inf` | Round towards positive infinity |
| `approx` | Approximate rounding mode |
| `full` | Full precision rounding mode |
| `nearest_int_to_zero` | Round towards zero to the nearest integer |

**fma Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |
| rounding\<nearest_even\> | yes | yes | yes | yes |
| rounding\<zero\> | yes | yes | yes* | yes* |
| rounding\<negative_inf\> | yes | yes | yes* | yes* |
| rounding\<positive_inf\> | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs, acc and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.log2`

Element-wise base-2 logarithm.

```
cuda_tile.log2 %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input floating-point tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the log2 operation. |

**Description:**

The log2 operation computes the element-wise base-2 logarithm of a floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = log2 %in : tile<4xf32>
```

---

## `cuda_tile.log`

Element-wise natural logarithm.

```
cuda_tile.log %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input floating-point tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the log operation. |

**Description:**

The log operation computes the element-wise natural logarithm of a floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.maxf`

Element-wise floating-point maximum.

```
cuda_tile.maxf %lhs %rhs %propagate_nan %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |
| propagate_nan | `Flag` | When set, maxf (or minf) returns a NaN if either of the two compared elements is NaN. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the maxf operation. |

**Description:**

The maxf operation computes the element-wise maximum of two input tiles with floating-point element types.

The `propagate_nan` controls how maxf will interpret NaN:
- If the `propagate_nan` modifier is set, maxf returns a canonical NaN if either of the compared elements is NaN (IEEE 754-2019's `maximum`).
- If the `propagate_nan` modifier is not set, maxf returns a canonical NaN only if both elements are NaN; otherwise, it returns the non-NaN element (IEEE 754-2019's `maximumNumber`).

If neither element is NaN, maxf will return the greater of the inputs. +0.0 is considered greater than -0.0.

If the `flush_to_zero` modifier is specified, denormal numbers are flushed to sign-preserving zero. The `flush_to_zero` modifier applies only to the f32 data type.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from partition views.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
%c0 = constant <i32: 0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
// IEEE 754-2019's maximum
%4 = maxf %2, %3 propagate_nan : tile<2x4xf32>
// IEEE 754-2019's maximumNumber
%5 = maxf %2, %3 : tile<2x4xf32>
// flush denormal to positive zero
%6 = maxf %2, %3 flush_to_zero : tile<2x4xf32>
```

---

## `cuda_tile.minf`

Element-wise floating-point minimum.

```
cuda_tile.minf %lhs %rhs %propagate_nan %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |
| propagate_nan | `Flag` | When set, maxf (or minf) returns a NaN if either of the two compared elements is NaN. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The minimum of the input tiles. |

**Description:**

The minf operation computes the element-wise minimum of two input tiles with floating-point element types.

The `propagate_nan` controls how minf will interpret NaN:
- If the `propagate_nan` modifier is set, minf returns a canonical NaN if either of the compared elements is NaN (IEEE 754-2019's `minimum`).
- If the `propagate_nan` modifier is not set, minf returns a canonical NaN only if both elements are NaN; otherwise, it returns the non-NaN element (IEEE 754-2019's `minimumNumber`).

If neither element is NaN, minf will return the lowest of the inputs. -0.0 is considered less than +0.0.

If the `flush_to_zero` modifier is specified, denormal numbers are flushed to sign-preserving zero. The `flush_to_zero` modifier applies only to the f32 data type.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from partition views.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
%c0 = constant <i32: 0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
// IEEE 754-2019's minimum
%4 = minf %2, %3 propagate_nan : tile<2x4xf32>
// IEEE 754-2019's minimumNumber
%5 = minf %2, %3 : tile<2x4xf32>
// flush denormal to positive zero
%6 = minf %2, %3 flush_to_zero : tile<2x4xf32>
```

---

## `cuda_tile.mmaf`

Floating-point matrix-multiply-accumulate.

```
cuda_tile.mmaf %lhs %rhs %acc
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64 \| tile<tf32>>` | The left hand side matrix operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64 \| tile<tf32>>` | The right hand side matrix operand. |
| acc | `tile<f16 \| f32 \| f64>` | The accumulator matrix operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| f32 \| f64>` | The result matrix after multiplication and accumulation. |

**Description:**

The mmaf operation implements an MMA (matrix-multiply-accumulate) operation for floating-point tiles. It performs matrix multiplication on the floating-point tiles lhs and rhs, then adds the tile acc to the result. lhs, rhs, and acc must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

The types of all operands must be a supported combination (see table below). Input operands must be of the same element type.

**Shapes:** Unbatched (2D) MMA expects the operands lhs, rhs, and acc to have shapes M x K, K x N, and M x N (respectively). Batched (3D) MMA expects the operands to have shapes B x M x K, B x K x N, and B x M x N (respectively).

**mmaf Supported Data Types:**

| Input Type | Supported Output Types |
|------------|----------------------|
| f8E4M3FN | f16 or f32 |
| f8E5M2 | f16 or f32 |
| f16 | f16 or f32 |
| bf16 | f32 |
| tf32 | f32 |
| f32 | f32 |
| f64 | f64 |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- acc and result must have the same shape and element type (`tile<f16 | f32 | f64>`).
- lhs and rhs must have the same element type (`tile<f16 | bf16 | f32 | f64 | tile<tf32>>`).
- lhs, rhs and acc must have the same rank.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Unbatched MMA: result is tile<4x2xf32>
%lhs0 = constant <f16: 0.0> : tile<4x8xf16>
%rhs0 = constant <f16: 0.0> : tile<8x2xf16>
%acc0 = constant <f32: 0.0> : tile<4x2xf32>

%0 = mmaf %lhs0, %rhs0, %acc0
    : tile<4x8xf16>, tile<8x2xf16>,
      tile<4x2xf32>

// Batched MMA: result is tile<2x4x2xf32>
%lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
%rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
%acc1 = constant <f32: 0.0> : tile<2x4x2xf32>

%1 = mmaf %lhs1, %rhs1, %acc1
    : tile<2x4x8xf16>, tile<2x8x2xf16>,
      tile<2x4x2xf32>
```

---

## `cuda_tile.mulf`

Element-wise floating-point multiplication.

```
cuda_tile.mulf %lhs %rhs %rounding_mode %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The product of the input tiles. |

**Description:**

The mulf operation computes the element-wise product between the two input tiles with floating-point element types. If the flush_to_zero modifier is specified, denormal numbers are flushed to positive zero. If the rounding modifier is specified, the particular rounding mode will be applied to each element of the result.

**mulf Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |
| rounding\<nearest_even\> | yes | yes | yes | yes |
| rounding\<zero\> | yes | yes | yes* | yes* |
| rounding\<negative_inf\> | yes | yes | yes* | yes* |
| rounding\<positive_inf\> | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.negf`

Element-wise floating-point negation.

```
cuda_tile.negf %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The negated floating-point tile. |

**Description:**

negf is an element-wise operation that negates the sign of source.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%source = constant <f32: 0.0> : tile<4xf32>
%result = negf %source : tile<4xf32>
```

---

## `cuda_tile.pow`

Element-wise floating-point exponentiation.

```
cuda_tile.pow %source %exponent
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The base tile. |
| exponent | `tile<f16 \| bf16 \| f32 \| f64>` | The exponent tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the pow operation. |

**Description:**

The pow operation computes the element-wise exponentiation of the source floating-point tile raised to the power of the exponent floating-point tile.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- result, source and exponent must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- source, exponent and result must have the same rank.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%source = constant <f32: 0.0> : tile<4xf32>
%exponent = constant <f32: 2.0> : tile<4xf32>
%result = pow %source, %exponent : tile<4xf32>
```

---

## `cuda_tile.remf`

Element-wise floating-point remainder.

```
cuda_tile.remf %lhs %rhs
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The remainder after division. |

**Description:**

The remf operation computes the element-wise floating-point remainder using truncated division (rounding towards zero). The result has the same sign as the dividend (lhs) and its magnitude is less than the magnitude of divisor (rhs).

**Special cases:**

- If y is zero, returns NaN
- If x is infinite and y is finite, returns NaN
- If x is finite and y is infinite, returns x
- If either argument is NaN, returns NaN

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.rsqrt`

Element-wise reciprocal square root.

```
cuda_tile.rsqrt %source %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input tile to compute the reciprocal square root of. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The reciprocal square root of the input tile. |

**Description:**

The rsqrt operation computes the element-wise reciprocal square root of the input floating-point tile. This operation supports `flush_to_zero`: if set by the user, will flush subnormal inputs and results to sign-preserving zero. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**rsqrt Modifiers:**

| Modifier | Float32 | Float64 |
|----------|---------|---------|
| flush_to_zero | yes | no |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = rsqrt %in : tile<4xf32>

// Rsqrt op with flush to zero modifier
%ftz_res = rsqrt %in flush_to_zero : tile<4xf32>
```

---

## `cuda_tile.sinh`

Element-wise hyperbolic sine.

```
cuda_tile.sinh %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The hyperbolic sine of the input tile. |

**Description:**

The sinh operation computes the element-wise hyperbolic sine of the input floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.sin`

Element-wise sine.

```
cuda_tile.sin %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input float tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The sine of the input tile. |

**Description:**

The sin operation computes the element-wise sine of the input floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res = sin %in : tile<4xf32>
```

---

## `cuda_tile.sqrt`

Element-wise square root.

```
cuda_tile.sqrt %source %rounding_mode %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input tile to compute the square root of. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The square root of the input tile. |

**Description:**

The sqrt operation computes the element-wise square root of a floating-point tile.

**sqrt Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |
| approx | yes | no | no | no |
| rounding\<nearest_even\> | yes | yes | yes | yes |
| rounding\<zero\> | yes | yes | yes* | yes* |
| rounding\<negative_inf\> | yes | yes | yes* | yes* |
| rounding\<positive_inf\> | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.subf`

Element-wise floating-point subtraction.

```
cuda_tile.subf %lhs %rhs %rounding_mode %flush_to_zero
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<f16 \| bf16 \| f32 \| f64>` | The left hand side operand. |
| rhs | `tile<f16 \| bf16 \| f32 \| f64>` | The right hand side operand. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |
| flush_to_zero | `Flag` | If set, flushes subnormal inputs and results to sign-preserving zero. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The result of the subtraction. |

**Description:**

The subf operation computes the element-wise subtraction of the input floating-point tiles.

**subf Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| flush_to_zero | yes | no | no | no |
| rounding\<nearest_even\> | yes | yes | yes | yes |
| rounding\<zero\> | yes | yes | yes* | yes* |
| rounding\<negative_inf\> | yes | yes | yes* | yes* |
| rounding\<positive_inf\> | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.tanh`

Element-wise hyperbolic tangent.

```
cuda_tile.tanh %source %rounding_mode
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input floating-point tile. |
| rounding_mode | `RoundingMode` | The rounding mode for the operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The hyperbolic tangent of the input floating-point tile. |

**Description:**

The tanh operation computes the element-wise hyperbolic tangent of the input floating-point tile. Default rounding mode is `full`.

The `approx` rounding mode implements a fast approximation to hyperbolic tangent. Subnormal results of this fast approximation are not flushed to zero.

The `full` rounding mode implements a relatively fast full-range approximation. The maximum ulp error is 2 across the full range of inputs in FP32 and 1 in FP64.

This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**tanh Modifiers:**

| Modifier | Float32 | Float64 | BFloat16 | Float16 |
|----------|---------|---------|----------|---------|
| approx | yes | no | no | no |
| full | yes | yes | yes* | yes* |

Entries with '*' are emulated in f32.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
%res0 = tanh %in : tile<4xf32>

// tanh with approx modifier
%res1 = tanh %in rounding<approx> : tile<4xf32>
```

---

## `cuda_tile.tan`

Element-wise tangent.

```
cuda_tile.tan %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<f16 \| bf16 \| f32 \| f64>` | The input floating-point tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<f16 \| bf16 \| f32 \| f64>` | The tangent of the input floating-point tile. |

**Description:**

The tan operation computes the element-wise tangent of the input floating-point tile. This operation is emulated in f32 when executed on half-precision inputs (f16 and bf16).

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<f16 | bf16 | f32 | f64>`).
- Operation must infer result types from operands and attributes.
