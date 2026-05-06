# Operations: Integer

Tile IR contains a set of typed arithmetic operations which implement familiar arithmetic operations on tiles of integers. For floating-point operations, see Floating Point Operations.

All operations are implemented in a manner that is efficient for the target architecture and device family. In most common cases this means utilizing the underlying hardware's native integer operations. Due to Tile IR's stability guarantees and higher-level programming model, some types on some hardware may be emulated. See Stability for more information.

## Integer Arithmetic

Integer types in Tile IR are signless, which is importantly not the same as unsigned. We store all integers in a two's complement representation and with required operations supporting a signed or unsigned flag as needed. This design allows us to not have to differentiate between signed and unsigned integer types at the IR level and keeps sign information local to the operation.

For the i1 type, unsigned operations see values 0/1, while signed operations see values 0/-1, with all i1 values canonicalized to 0x00 (false) or 0x01 (true) for consistent LSB-only semantics.

---

## `cuda_tile.absi`

Element-wise integer absolute value.

```
cuda_tile.absi %source
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The input integer tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The absolute value of the input tile. |

**Description:**

The absi operation computes the absolute value of the input integer tile. The input tile is always interpreted as a signed integer. The output tile is always interpreted as an unsigned integer.

Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape.
- source and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.addi`

Element-wise integer addition.

```
cuda_tile.addi %lhs %rhs %overflow
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| overflow | `IntegerOverflow` | The overflow behavior of the operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The sum of the input tiles. |

**Description:**

The addi operation computes the element-wise addition of two tiles with integer element types. Element-wise integer arithmetic operations are performed by the target architecture's native integer instructions. The default semantics are wrap-around semantics on overflow or underflow.

**Overflow attribute:**

| Overflow | Description |
|---------|-------------|
| `none` | The compiler makes no assumptions regarding overflow behavior. |
| `no_signed_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed integers. |
| `no_unsigned_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as unsigned integers. |
| `no_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers. |

These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation respects these assumptions dynamically during execution. If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.cmpi`

Element-wise integer comparison.

```
cuda_tile.cmpi %comparison_predicate %lhs %rhs %signedness
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| comparison_predicate | `ComparisonPredicate` | The comparison predicate. |
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1>` | The result of the comparison. |

**Description:**

The cmpi operation is a generic comparison for integer-like types. The operands must have the same shape and type, and this type must be an integer type. The result type has i1 element type and the same shape as the operands. The result is 1 if the comparison is true and 0 otherwise. The comparison is performed element-wise.

**Comparison predicates:**

| Predicate | Description |
|-----------|-------------|
| `equal` | Equal comparison |
| `not_equal` | Not equal comparison |
| `less_than` | Less than comparison |
| `less_than_or_equal` | Less than or equal comparison |
| `greater_than` | Greater than comparison |
| `greater_than_or_equal` | Greater than or equal comparison |

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs and rhs must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Result type has i1 element type and same shape as operands.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%lhs0 = constant <i16: 0> : tile<i16>
%rhs0 = constant <i16: 0> : tile<i16>

// Scalar "signed less than" comparison.
%x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

%lhs1 = constant <i64: 0> : tile<2x2xi64>
%rhs1 = constant <i64: 0> : tile<2x2xi64>

// Tile equality comparison.
// There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
%x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>
```

---

## `cuda_tile.divi`

Element-wise integer division.

```
cuda_tile.divi %lhs %rhs %signedness %rounding
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned. |
| rounding | `RoundingMode` | Set the rounding direction (implementing floordiv/ceildiv). |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result of the division. |

**Description:**

The divi operation computes the element-wise division of two tile values with integer element type. The default rounding is towards zero. The rounding mode can be set to `positive_inf` ("ceiling division"), or `negative_inf` ("floor division"), other values are illegal.

The use of the rounding flag `negative_inf` with `unsigned` is not a valid combination. If the unsigned flag is provided, the operands are treated as unsigned integers, otherwise they are treated as signed integers. The behavior is undefined if the right hand side is zero. A signed division overflow (minimum value divided by -1) is undefined behavior.

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.maxi`

Element-wise integer maximum.

```
cuda_tile.maxi %lhs %rhs %signedness
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result of the maxi operation. |

**Description:**

The maxi operation computes the element-wise maximum between two input integer tiles.

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from them.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%c0 = constant <i32: 0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
// Signless i32 treated as unsigned
%4 = maxi %2, %3 unsigned : tile<2x4xi32>
// Signless i32 treated as signed
%5 = maxi %2, %3 signed : tile<2x4xi32>
```

---

## `cuda_tile.mini`

Element-wise integer minimum.

```
cuda_tile.mini %lhs %rhs %signedness
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The minimum of the input tiles. |

**Description:**

The mini operation computes the element-wise minimum between the two input tiles with integer element types.

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Create tensor view from a pointer to global memory
%0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
%1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
// Convert tensor views to partition views and load tiles from partition views.
%p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
%c0 = constant <i32: 0> : tile<i32>
%2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
%3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
// Signless i32 treated as unsigned
%4 = mini %2, %3 unsigned : tile<2x4xi32>
// Signless i32 treated as signed
%5 = mini %2, %3 signed : tile<2x4xi32>
```

---

## `cuda_tile.mmai`

Integer matrix-multiply-accumulate.

```
cuda_tile.mmai %lhs %rhs %acc %signedness_lhs %signedness_rhs
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i8>` | The left hand side matrix operand. |
| rhs | `tile<i8>` | The right hand side matrix operand. |
| acc | `tile<i32>` | The accumulator matrix operand. |
| signedness_lhs | `Signedness` | The signedness of the lhs operand. |
| signedness_rhs | `Signedness` | The signedness of the rhs operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i32>` | The result matrix after multiplication and accumulation. |

**Description:**

The mmai operation implements an MMA (matrix-multiply-accumulate) operation for integer tiles. It performs matrix multiplication on the integer tiles lhs and rhs, then adds the tile acc to the result. lhs, rhs, and acc must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

Input tiles lhs and rhs must be of integer type i8. The signedness of lhs and rhs are specified separately by the `signedness_lhs` and `signedness_rhs` attributes, respectively. The accumulator tile acc must be of type i32 and is always interpreted as signed. The output tile result is of type i32 and is always interpreted as signed.

**Shapes:** Unbatched (2D) MMA expects the operands lhs, rhs, and acc to have shapes M x K, K x N, and M x N (respectively). Batched (3D) MMA expects the operands to have shapes B x M x K, B x K x N, and B x M x N (respectively).

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- acc and result must have the same shape and element type (`tile<i32>`).
- lhs and rhs must have the same element type (`tile<i8>`).
- lhs, rhs and acc must have the same rank.
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// Unbatched MMA: result is tile<4x2xi32>
%lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
%rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
%acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>

%0 = mmai %lhs0, %rhs0, %acc0 signed signed
    : tile<4x8xi8>, tile<8x2xi8>,
      tile<4x2xi32>

// Batched MMA: result is tile<2x4x2xi32>
%lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
%rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
%acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>

%1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
    : tile<2x4x8xi8>, tile<2x8x2xi8>,
      tile<2x4x2xi32>
```

---

## `cuda_tile.muli`

Element-wise integer multiplication.

```
cuda_tile.muli %lhs %rhs %overflow
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side input integer tile. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side input integer tile. |
| overflow | `IntegerOverflow` | The overflow behavior of the operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The product of the input tiles. |

**Description:**

The muli operation computes the element-wise product between the two input tiles with integer element types.

**Overflow attribute:**

| Overflow | Description |
|---------|-------------|
| `none` | The compiler makes no assumptions regarding overflow behavior. |
| `no_signed_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed integers. |
| `no_unsigned_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as unsigned integers. |
| `no_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers. |

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.mulhii`

Element-wise high bits of integer multiplication.

```
cuda_tile.mulhii %x %y
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| x | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side input integer tile. |
| y | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side input integer tile. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The most significant bits of the product of the input tiles. |

**Description:**

The mulhii operation produces the most significant N bits of the 2N-bit product of two N-bit integer tiles. For i64, this is the most significant 64 bits of the full 128-bit product; for i8, it is the most significant 8 bits of the full 16-bit product; etc.

This is in contrast to muli, which produces the lower N bits of the 2N-bit product. The mulhii operation is only defined for unsigned integers.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- x, y and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
// 2^31 * 2 = 2^32, or 0x100000000.
// The most significant 32 bits of the product are 0x00000001.
// The lower 32 bits of the product are 0x00000000.
%a = constant <i32: 2147483648> : tile<i32>  // %a = 2^31
%b = constant <i32: 2> : tile<i32>           // %b = 2
%res_hi = mulhii %a, %b : tile<i32>          // %res_hi = 1
%res_lo = muli %a, %b : tile<i32>            // %res_lo = 0
```

---

## `cuda_tile.negi`

Element-wise integer negation.

```
cuda_tile.negi %source %overflow
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| source | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The input integer tile. |
| overflow | `IntegerOverflow` | The overflow behavior of the operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The negated integer tile. |

**Description:**

The negi operation computes the element-wise negation of the input integer tile. The input and output tiles are always interpreted as signed integers.

**Overflow attribute:**

| Overflow | Description |
|---------|-------------|
| `none` | The compiler makes no assumptions regarding overflow behavior. |
| `no_signed_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed integers. |
| `no_unsigned_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as unsigned integers. |
| `no_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers. |

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- source and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%source = constant <i16: [0, 1, 2, 3]> : tile<4xi16>
%result = negi %source : tile<4xi16>
// %result = [0, -1, -2, -3]
```

---

## `cuda_tile.ori`

Element-wise bitwise OR.

```
cuda_tile.ori %lhs %rhs
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The bitwise OR of the input tiles. |

**Description:**

The ori operation computes the element-wise bitwise OR of two tiles with integer element types.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.remi`

Element-wise integer remainder.

```
cuda_tile.remi %lhs %rhs %signedness
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The remainder after division. |

**Description:**

The remi operation computes the element-wise remainder of the input tiles with integer element types using truncated division (rounding towards zero). Division by zero is undefined behavior.

If the operation is signed, the sign of the result matches the sign of the dividend (lhs). For example:
- remi(7, 3) = 1
- remi(7, -3) = 1
- remi(-7, 3) = -1
- remi(-7, -3) = -1

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- result, lhs and rhs must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.shli`

Element-wise shift-left.

```
cuda_tile.shli %lhs %rhs %overflow
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand (shift amount). |
| overflow | `IntegerOverflow` | The overflow behavior of the operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result of the left shift operation. |

**Description:**

The shli operation computes the element-wise left shift of the lhs integer operand by the rhs operand. The lower-order bits on the right are filled with zeros. The rhs operand is interpreted as an unsigned integer.

**Overflow attribute:**

| Overflow | Description |
|---------|-------------|
| `none` | The compiler makes no assumptions regarding overflow behavior. |
| `no_signed_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed integers. |
| `no_unsigned_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as unsigned integers. |
| `no_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers. |

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.shri`

Element-wise shift-right.

```
cuda_tile.shri %lhs %rhs %signedness
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand (shift amount). |
| signedness | `Signedness` | Interpret integer(s) as signed or unsigned. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result of the right shift operation. |

**Description:**

The shri operation computes the element-wise right shift of the lhs integer operand by the value of the rhs operand for tiles with integer element types. When unsigned, higher-order bits are zero-filled; when signed, the higher-order bits are filled with the sign bit. The rhs operand is always interpreted as an unsigned integer.

**Signedness:**

| Signedness | Description |
|------------|-------------|
| `unsigned` | Treat the operands as unsigned integers |
| `signed` | Treat the operands as signed integers |

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.subi`

Element-wise integer subtraction.

```
cuda_tile.subi %lhs %rhs %overflow
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |
| overflow | `IntegerOverflow` | The overflow behavior of the operation. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The result of the subtraction. |

**Description:**

The subi operation computes the element-wise subtraction of two input integer tiles.

**Overflow attribute:**

| Overflow | Description |
|---------|-------------|
| `none` | The compiler makes no assumptions regarding overflow behavior. |
| `no_signed_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed integers. |
| `no_unsigned_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as unsigned integers. |
| `no_wrap` | The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers. |

If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

---

## `cuda_tile.xori`

Element-wise bitwise XOR.

```
cuda_tile.xori %lhs %rhs
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The bitwise XOR of the input tiles. |

**Description:**

The xori operation computes the element-wise bitwise exclusive or (XOR) of two tile values with integer element types.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.

**Examples:**

```cuda_tile
%lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
%rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
// This computes the bitwise XOR of each element in `%lhs` and `%rhs`, which
// are tiles of shape `4xi32`, and returns the result as `%result`.
%result = xori %lhs, %rhs : tile<4xi32>
```

---

## `cuda_tile.andi`

Element-wise bitwise logical AND.

```
cuda_tile.andi %lhs %rhs
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The left hand side operand. |
| rhs | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The right hand side operand. |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| result | `tile<i1 \| i8 \| i16 \| i32 \| i64>` | The bitwise AND of the input tiles. |

**Description:**

The andi operation produces a value that is the result of an element-wise, bitwise "and" of two tiles with integer element type.

**Constraints:**

- Operation speculation safety must be determined by operands and attributes.
- Operation must be safe to execute speculatively without side effects.
- Operation must not perform any memory side effects.
- lhs, rhs and result must have the same shape and element type (`tile<i1 | i8 | i16 | i32 | i64>`).
- Operation must infer result types from operands and attributes.
