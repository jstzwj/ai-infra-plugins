# Operations: Conversions

There are no implicit type conversions in Tile IR. This section describes the explicit conversion operations for interconverting between types which have compatible representations or rules for conversion.

## `cuda_tile.bitcast`

Bitcast a tile from one element type to another.

```
cuda_tile.bitcast %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<i1 \| i8 \| i16 \| i32 \| i64 \| f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The source tile to cast |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64 \| f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The casted tile |

**Description:**

Casts the input tile from one element type to another without modifying the underlying bits. Only non-pointer types of the same bit width are allowed (e.g., i32 to f32). Pointer types must use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` instead.

---

## `cuda_tile.exti`

Extend the width of an integer tile.

```
cuda_tile.exti %from %signedness
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The input integer tile to extend |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| to | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The extended integer tile |

**Description:**

Converts a tile of integers of a given width to a strictly larger width. Zero-extension is used for unsigned integers and sign-extension is used for signed integers.

The signedness attribute specifies:

- `unsigned` - Treat the operands as unsigned integers
- `signed` - Treat the operands as signed integers

---

## `cuda_tile.ftof`

Convert between floating-point types.

```
cuda_tile.ftof %from %rounding_mode
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from | `tile<f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The input floating-point tile |
| rounding_mode | `RoundingMode` | The rounding mode for the operation |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| to | `tile<f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The result floating-point tile |

**Description:**

Converts a tile of a given floating-point element type into one of a different floating-point element type. The source type and the result type must be different. Only `nearest_even` rounding mode is supported.

> **Warning:** Different floating-point types have different conversion behaviors for out-of-finite-range values and special values.

**Rounding modes:**

| Mode | Description |
|------|-------------|
| `nearest_even` | Round to nearest (ties to even) |
| `zero` | Round towards zero (truncate) |
| `negative_inf` | Round towards negative infinity |
| `positive_inf` | Round towards positive infinity |
| `approx` | Approximate rounding mode |
| `full` | Full precision rounding mode |

---

## `cuda_tile.ftoi`

Convert a tile from floating-point values to integer values.

```
cuda_tile.ftoi %from %signedness %rounding_mode
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from | `tile<f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The input floating-point tile |
| signedness | `Signedness` | Interpret result as signed or unsigned |
| rounding_mode | `RoundingMode` | The rounding mode for the operation |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| to | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result integer tile |

**Description:**

Converts a floating-point tile into an integer tile, preserving the numerical value rounded towards zero to the nearest integer of the provided type. Only `nearest_int_to_zero` rounding mode is supported.

> **Warning:** If the input value, after rounding, is outside the range of the target integer type, the closest representable value is used. NaN values are converted to 0. Input Inf values are undefined behavior.

---

## `cuda_tile.itof`

Convert integer to floating-point.

```
cuda_tile.itof %from %signedness %rounding_mode
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The input integer tile |
| signedness | `Signedness` | Interpret operand as signed or unsigned |
| rounding_mode | `RoundingMode` | The rounding mode for the operation |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| to | `tile<f16 \| bf16 \| f32 \| f64 \| fp8e4m3fn \| fp8e5m2 \| tf32>` | The converted floating-point tile |

**Description:**

Converts an integer tile into a float tile, preserving the numerical value rounded to the nearest floating-point number of the provided type.

> **Warning:** Different floating-point types have different conversion behaviors for out-of-finite-range values.

---

## `cuda_tile.int_to_ptr`

Convert a tile of integers to a tile of pointers.

```
cuda_tile.int_to_ptr %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<i64>` | The input tile of integers |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `ptr` | The output tile of pointers |

**Description:**

Converts a tile of integers to a tile of pointers. The source operand is interpreted as an unsigned integer. The inverse of this operation is `cuda_tile.ptr_to_int`.

---

## `cuda_tile.ptr_to_int`

Convert a tile of pointers to a tile of integers.

```
cuda_tile.ptr_to_int %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `ptr` | The input tile of pointers |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i64>` | The output tile of integers |

**Description:**

Converts a tile of pointer-type elements to a tile of i64 elements. The result values should be interpreted as unsigned integers. The inverse of this operation is `cuda_tile.int_to_ptr`.

---

## `cuda_tile.ptr_to_ptr`

Reinterpret a tile of one pointer type as another.

```
cuda_tile.ptr_to_ptr %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `ptr` | Tile with source pointer element type |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `ptr` | Tile with target pointer element type |

**Description:**

Casts a tile of pointers from a pointer of one element type to another element type. Casts between pointer and non-pointer types are disallowed. Use `cuda_tile.ptr_to_int` or `cuda_tile.int_to_ptr` for those conversions.

---

## `cuda_tile.trunci`

Truncates the width of an integer tile.

```
cuda_tile.trunci %from %overflow
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| from | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The input integer tile to truncate |
| overflow | `IntegerOverflow` | The overflow behavior of the operation |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| to | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The truncated integer tile |

**Description:**

Converts a tile of integers of a given element type to one with a strictly smaller width. The optional overflow attribute specifies whether an overflow can occur:

| Overflow | Behavior |
|----------|----------|
| `none` | No assumptions about overflow |
| `no_signed_wrap` | Assumes no signed overflow |
| `no_unsigned_wrap` | Assumes no unsigned overflow |
| `no_wrap` | Assumes no signed or unsigned overflow |

These attributes serve as assumptions that the compiler may use. If an overflow occurs at runtime despite the attribute stating otherwise, the behavior is undefined.
