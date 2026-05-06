# Triton Language Core Reference (`triton.language.core`)

This document provides an exhaustive reference for the `triton.language.core` module, the foundational module of the Triton GPU programming language. It covers the complete type system, tensor operations, memory operations, reductions, atomic operations, compiler hints, debugging utilities, and control flow constructs.

**Source file**: `python/triton/language/core.py` (~3795 lines)

---

## Table of Contents

1. [Type System](#1-type-system)
   - [base_type](#base_type)
   - [base_value](#base_value)
   - [dtype](#dtype)
   - [pointer_type](#pointer_type)
   - [block_type](#block_type)
   - [tuple_type](#tuple_type)
   - [constexpr_type](#constexpr_type)
   - [slice_type](#slice_type)
2. [Scalar Types (Predefined dtypes)](#2-scalar-types)
3. [constexpr Class](#3-constexpr-class)
4. [const Annotation Class](#4-const-annotation-class)
5. [tensor Class](#5-tensor-class)
6. [tuple Class](#6-tuple-class)
7. [slice Class](#7-slice-class)
8. [Programming Model](#8-programming-model)
   - [program_id](#program_id)
   - [num_programs](#num_programs)
9. [Tensor Creation](#9-tensor-creation)
   - [arange](#arange)
   - [full](#full)
   - [zeros](#zeros)
   - [zeros_like](#zeros_like)
   - [to_tensor](#to_tensor)
   - [cast](#cast)
10. [Memory Operations](#10-memory-operations)
    - [load](#load)
    - [store](#store)
    - [make_block_ptr](#make_block_ptr)
    - [advance](#advance)
    - [make_tensor_descriptor](#make_tensor_descriptor)
    - [load_tensor_descriptor](#load_tensor_descriptor)
    - [store_tensor_descriptor](#store_tensor_descriptor)
11. [Linear Algebra](#11-linear-algebra)
    - [dot](#dot)
    - [dot_scaled](#dot_scaled)
12. [Shape Manipulation](#12-shape-manipulation)
    - [broadcast](#broadcast)
    - [broadcast_to](#broadcast_to)
    - [reshape](#reshape)
    - [expand_dims](#expand_dims)
    - [permute](#permute)
    - [trans](#trans)
    - [view](#view)
    - [cat](#cat)
    - [split](#split)
    - [join](#join)
    - [ravel](#ravel)
    - [item](#item)
    - [gather](#gather)
13. [Reductions](#13-reductions)
    - [reduce](#reduce)
    - [associative_scan](#associative_scan)
    - [histogram](#histogram)
    - [_reduce_with_indices](#_reduce_with_indices)
14. [Atomic Operations](#14-atomic-operations)
    - [atomic_cas](#atomic_cas)
    - [atomic_xchg](#atomic_xchg)
    - [atomic_add](#atomic_add)
    - [atomic_max](#atomic_max)
    - [atomic_min](#atomic_min)
    - [atomic_and](#atomic_and)
    - [atomic_or](#atomic_or)
    - [atomic_xor](#atomic_xor)
15. [Conditioning and Math Helpers](#15-conditioning-and-math-helpers)
    - [where](#where)
    - [maximum](#maximum)
    - [minimum](#minimum)
    - [clamp](#clamp)
    - [add](#add)
    - [sub](#sub)
    - [mul](#mul)
    - [builtin_max](#builtin_max)
    - [builtin_min](#builtin_min)
16. [Compiler Hints](#16-compiler-hints)
    - [multiple_of](#multiple_of)
    - [max_contiguous](#max_contiguous)
    - [max_constancy](#max_constancy)
    - [assume](#assume)
17. [Debugging](#17-debugging)
    - [static_print](#static_print)
    - [static_assert](#static_assert)
    - [device_print](#device_print)
    - [device_assert](#device_assert)
    - [debug_barrier](#debug_barrier)
18. [Inline Assembly](#18-inline-assembly)
    - [inline_asm_elementwise](#inline_asm_elementwise)
    - [map_elementwise](#map_elementwise)
19. [Control Flow](#19-control-flow)
    - [condition](#condition)
    - [static_range](#static_range)
    - [range](#range)
20. [tensor_descriptor_base Class](#20-tensor_descriptor_base-class)
21. [tensor_descriptor Class](#21-tensor_descriptor-class)
22. [_block_ptr Class](#22-_block_ptr-class)
23. [External Function Dispatch](#23-external-function-dispatch)
24. [Utility Functions and Decorators](#24-utility-functions-and-decorators)

---

## 1. Type System

### base_type

```python
class base_type
```

Abstract base class for all Triton types. Defines the interface that every type in the Triton type system must implement.

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `__eq__` | `(self, other) -> bool` | Equality comparison. Must be implemented by subclasses. |
| `__ne__` | `(self, other) -> bool` | Inequality comparison. Returns `not (self == other)`. |
| `_unflatten_ir` | `(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]` | Reconstruct a frontend value from IR handles. `cursor` is the index of the first relevant handle; returns the updated cursor position. |
| `mangle` | `(self) -> str` | Return a mangled string representation for function name mangling. |
| `_flatten_ir_types` | `(self, builder: ir.builder, out: List[ir.type]) -> None` | Append the IR types corresponding to this frontend type to `out`. |

### base_value

```python
class base_value
```

Abstract base class for all values that exist in the Triton IR (i.e., not `constexpr`s).

**Attributes**:
- `type: base_type` -- The type of this value.

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `_set_name` | `(self, builder: ir.builder, name: str) -> None` | Set a human-readable name for this value in the IR. Raises `NotImplementedError`. |
| `_flatten_ir` | `(self, handles: List[ir.value]) -> None` | Flatten the frontend value into a sequence of MLIR handles appended to the output list. Raises `NotImplementedError`. |

### dtype

```python
class dtype(base_type)
```

Represents a scalar data type in Triton. This is the primary type class used for specifying element types of tensors.

**Class Attributes**:

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `SINT_TYPES` | `list[str]` | `['int8', 'int16', 'int32', 'int64']` | Signed integer type names |
| `UINT_TYPES` | `list[str]` | `['int1', 'uint8', 'uint16', 'uint32', 'uint64']` | Unsigned integer type names |
| `FP_TYPES` | `list[str]` | `['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']` | Floating-point type names |
| `STANDARD_FP_TYPES` | `list[str]` | `['fp16', 'bf16', 'fp32', 'fp64']` | Standard floating-point type names (not fp8) |
| `OTHER_TYPES` | `list[str]` | `['void']` | Other type names |

**Inner Enums**:

```python
class SIGNEDNESS(Enum):
    SIGNED = 0
    UNSIGNED = 1

class KIND(Enum):
    BOOLEAN = 0
    INTEGRAL = 1
    FLOATING = 2
```

**Constructor**:

```python
dtype(name: str)
```

- `name` -- One of the valid type name strings from `SINT_TYPES`, `UINT_TYPES`, `FP_TYPES`, or `OTHER_TYPES`.

**Instance Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The string name of this dtype (e.g., `'fp32'`, `'int32'`) |
| `primitive_bitwidth` | `int` | The bit width of this type (e.g., 32 for `fp32`, 8 for `int8`) |
| `itemsize` | `int` | Size in bytes: `primitive_bitwidth // 8` |
| `int_signedness` | `SIGNEDNESS` | Signedness for integer types (`SIGNED` or `UNSIGNED`) |
| `int_bitwidth` | `int` | Bit width for integer types |
| `fp_mantissa_width` | `int` | Mantissa width for floating-point types |
| `exponent_bias` | `int` | Exponent bias for floating-point types |

**Floating-Point Properties by Type**:

| dtype | fp_mantissa_width | exponent_bias |
|-------|-------------------|---------------|
| `fp8e4b15` | 3 | 15 |
| `fp8e4nv` | 3 | 7 |
| `fp8e4b8` | 3 | 8 |
| `fp8e5` | 2 | 15 |
| `fp8e5b16` | 2 | 16 |
| `fp16` | 10 | 15 |
| `bf16` | 7 | 127 |
| `fp32` | 23 | 127 |
| `fp64` | 52 | 1023 |

**Type Query Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `is_fp8() -> bool` | `True` if `'fp8'` is in `self.name` | Any fp8 variant |
| `is_fp8e4nv() -> bool` | `True` for `fp8e4nv` | NVidia fp8 E4M3 |
| `is_fp8e4b8() -> bool` | `True` for `fp8e4b8` | fp8 E4M3 with bias 8 |
| `is_fp8e4b15() -> bool` | `True` for `fp8e4b15` | fp8 E4M3 with bias 15 |
| `is_fp8e5() -> bool` | `True` for `fp8e5` | fp8 E5M2 |
| `is_fp8e5b16() -> bool` | `True` for `fp8e5b16` | fp8 E5M2 with bias 16 |
| `is_fp16() -> bool` | `True` for `fp16` | IEEE half precision |
| `is_bf16() -> bool` | `True` for `bf16` | BFloat16 |
| `is_fp32() -> bool` | `True` for `fp32` | IEEE single precision |
| `is_fp64() -> bool` | `True` for `fp64` | IEEE double precision |
| `is_int1() -> bool` | `True` for `int1` | Boolean type |
| `is_int8() -> bool` | `True` for `int8` | 8-bit signed integer |
| `is_int16() -> bool` | `True` for `int16` | 16-bit signed integer |
| `is_int32() -> bool` | `True` for `int32` | 32-bit signed integer |
| `is_int64() -> bool` | `True` for `int64` | 64-bit signed integer |
| `is_uint8() -> bool` | `True` for `uint8` | 8-bit unsigned integer |
| `is_uint16() -> bool` | `True` for `uint16` | 16-bit unsigned integer |
| `is_uint32() -> bool` | `True` for `uint32` | 32-bit unsigned integer |
| `is_uint64() -> bool` | `True` for `uint64` | 64-bit unsigned integer |
| `is_floating() -> bool` | `True` for any FP type | All floating-point types |
| `is_standard_floating() -> bool` | `True` for standard FP types | Excludes fp8 variants |
| `is_int_signed() -> bool` | `True` for signed integers | |
| `is_int_unsigned() -> bool` | `True` for unsigned integers | |
| `is_int() -> bool` | `True` for any integer type | Signed or unsigned |
| `is_bool() -> bool` | `True` for `int1` | Same as `is_int1()` |
| `kind() -> KIND` | `KIND` enum value | Returns `BOOLEAN`, `INTEGRAL`, or `FLOATING` |

**Value Range Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_int_max_value()` | `-> int` | Maximum representable integer value. For signed: `2^(bitwidth-1) - 1`. For unsigned: `2^bitwidth - 1`. |
| `get_int_min_value()` | `-> int` | Minimum representable integer value. For signed: `-2^(bitwidth-1)`. For unsigned: `0`. |

**Static Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_dtype(type_str)` | `-> bool` | Returns `True` if `type_str` is a recognized dtype name |
| `is_void()` | `-> bool` | Raises `RuntimeError("Not implemented")` |
| `is_block()` | `-> bool` | Always returns `False` for scalar dtype |
| `is_ptr()` | `-> bool` | Always returns `False` for scalar dtype |
| `is_const()` | `-> bool` | Always returns `False` for scalar dtype |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `scalar` | `dtype` | Returns `self` (the scalar type) |
| `cache_key_part` | `str` | Returns `self.name` for cache key generation |

**Other Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `__eq__` | `(self, other) -> bool` | Equality based on name comparison |
| `__hash__` | `() -> int` | Hash based on name |
| `__str__` | `() -> str` | Returns `self.name` |
| `__repr__` | `() -> str` | Returns `triton.language.<codegen_name>()` |
| `to_ir` | `(self, builder: ir.builder) -> ir.type` | Convert to MLIR IR type |
| `codegen_name` | `() -> str` | Returns the codegen-compatible name (e.g., `'float32'` for `fp32`, `'bfloat16'` for `bf16`) |
| `mangle` | `() -> str` | Returns mangled name for type mangling (e.g., `'i32'`, `'u8'`, `'fp32'`) |
| `with_element_ty` | `(self, element_ty: dtype) -> dtype` | For non-block types, returns `element_ty` directly |

**Example**:

```python
import triton.language as tl

# Type checking
assert tl.float32.is_floating()
assert tl.int32.is_int_signed()
assert tl.float32.primitive_bitwidth == 32
assert tl.float32.fp_mantissa_width == 23
assert tl.int32.get_int_max_value() == 2147483647
```

### pointer_type

```python
class pointer_type(dtype)
```

Represents a pointer to a scalar type in Triton.

**Constructor**:

```python
pointer_type(element_ty: dtype, address_space: int = 1, const: bool = False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `element_ty` | `dtype` | required | The element type the pointer points to |
| `address_space` | `int` | `1` | The memory address space |
| `const` | `bool` | `False` | Whether this is a const (read-only) pointer |

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `element_ty` | `dtype` | The element type pointed to |
| `address_space` | `int` | Memory address space identifier |
| `const` | `bool` | Whether the pointer is read-only |
| `name` | `str` | `pointer<element_ty>` or `const_pointer<element_ty>` |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `is_ptr()` | `True` | Always `True` for pointer types |
| `is_const()` | `bool` | Returns `self.const` |
| `scalar` | `pointer_type` | Returns `self` (property) |
| `to_ir(builder)` | `ir.pointer_type` | Converts to MLIR pointer type |
| `mangle()` | `str` | Returns `"P" + element_ty.mangle()` |
| `__eq__(other)` | `bool` | Compares element type, address space, and constness |
| `__str__()` | `str` | Returns the pointer name string |
| `__repr__()` | `str` | Same as `__str__` |

**Example**:

```python
import triton.language as tl

ptr_ty = tl.pointer_type(tl.float32)
assert ptr_ty.is_ptr()
assert ptr_ty.element_ty == tl.float32

const_ptr_ty = tl.pointer_type(tl.int32, const=True)
assert const_ptr_ty.is_const()
```

### block_type

```python
class block_type(dtype)
```

Represents an N-dimensional block (tensor) of a scalar type. This is the type of multi-dimensional tensors in Triton kernels.

**Constructor**:

```python
block_type(element_ty: dtype, shape: List)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `element_ty` | `dtype` | The element type of the block |
| `shape` | `list` or `tuple` | The dimensions of the block. Must be a non-empty list/tuple of integers. |

**Note**: 0D block types (empty shape) are forbidden and will raise a `TypeError`.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `element_ty` | `dtype` | The scalar element type |
| `shape` | `tuple[int]` | The block dimensions |
| `numel` | `int` | Total number of elements (product of shape). Validated to be a power of 2 and within `TRITON_MAX_TENSOR_NUMEL`. |
| `name` | `str` | `"<shape, element_ty>"` |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `scalar` | `dtype` | Returns `self.element_ty` |
| `nbytes` | `int` | Total bytes: `numel * (element_ty.primitive_bitwidth // 8)` |

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `is_block()` | `True` | Always `True` for block types |
| `get_block_shapes()` | `Tuple[int]` | Returns `self.shape` |
| `with_element_ty(scalar_ty)` | `block_type` | Returns a new `block_type` with the same shape but different element type |
| `to_ir(builder)` | `ir.block_type` | Converts to MLIR block type |
| `__eq__(other)` | `bool` | Compares element type and shape |
| `__str__()` | `str` | Returns the block name string |
| `mangle()` | `str` | Returns `"<elt_mangle>S<shape_joined>S"` |

**Example**:

```python
import triton.language as tl

bt = tl.core.block_type(tl.float32, [16, 32])
assert bt.is_block()
assert bt.shape == (16, 32)
assert bt.numel == 512
assert bt.scalar == tl.float32
assert bt.nbytes == 512 * 4  # 2048 bytes
```

### tuple_type

```python
class tuple_type(base_type)
```

Represents a tuple of Triton types, used for returning multiple values from functions and representing structured data.

**Constructor**:

```python
tuple_type(types, fields=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `types` | `list[base_type]` | required | List of types in the tuple |
| `fields` | `list[str]` or `None` | `None` | Optional field names for named tuples |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Cached property. If fields: `"[name:ty, ...]"`, else `"[ty, ...]"` |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `__iter__` | `() -> iterator` | Iterates over `self.types` |
| `__getitem__` | `(index: int) -> dtype` | Returns the type at the given index |
| `__eq__` | `(other) -> bool` | Compares types and fields |
| `__str__` | `() -> str` | Returns the name |
| `_flatten_ir_types` | `(builder, out) -> None` | Flattens all member types |
| `_unflatten_ir` | `(handles, cursor) -> (tuple, int)` | Reconstructs a tuple value from IR handles |
| `mangle` | `() -> str` | Returns `"T" + "_".join(manglings) + "T"` |

### constexpr_type

```python
class constexpr_type(base_type)
```

The type of a `constexpr` value. Used internally to represent compile-time constant types.

**Constructor**:

```python
constexpr_type(value)
```

**Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `__eq__(other)` | `bool` | Compares value equality |
| `__repr__()` | `str` | Returns `"constexpr_type[value]"` |
| `__hash__()` | `int` | Hash of the value |
| `mangle()` | `str` | Returns `"c" + repr(value)` or uses value's mangle method |
| `_flatten_ir_types(builder, out)` | `None` | No-op (constexprs have no IR representation) |
| `_unflatten_ir(handles, cursor)` | `(constexpr, int)` | Returns a `constexpr` wrapping the stored value |

### slice_type

```python
class slice_type(dtype)
```

Type for the `slice` built-in.

**Constructor**: `slice_type()` -- takes no arguments. Sets `self.name = 'slice_type'`.

---

## 2. Scalar Types

All predefined scalar type instances are module-level constants:

```python
# Void type
void = dtype('void')

# Boolean
int1 = dtype('int1')

# Signed integers
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')

# Unsigned integers
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')

# 8-bit floating-point types
float8e5    = dtype('fp8e5')      # E5M2 with bias 15
float8e5b16 = dtype('fp8e5b16')   # E5M2 with bias 16
float8e4nv  = dtype('fp8e4nv')    # E4M3 (NVidia) with bias 7
float8e4b8  = dtype('fp8e4b8')    # E4M3 with bias 8
float8e4b15 = dtype('fp8e4b15')   # E4M3 with bias 15

# Standard floating-point types
float16   = dtype('fp16')   # IEEE half precision (16-bit)
bfloat16  = dtype('bf16')   # BFloat16 (16-bit)
float32   = dtype('fp32')   # IEEE single precision (32-bit)
float64   = dtype('fp64')   # IEEE double precision (64-bit)

# Predefined pointer type
pi32_t = pointer_type(int32)
```

### get_int_dtype

```python
def get_int_dtype(bitwidth: int, signed: bool) -> dtype
```

Returns the appropriate integer `dtype` for the given bit width and signedness.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bitwidth` | `int` | The desired bit width (1, 8, 16, 32, or 64) |
| `signed` | `bool` | Whether the type should be signed |

**Returns**: `dtype`

**Raises**: `ValueError` if the bit width is not supported.

**Example**:

```python
import triton.language as tl

assert tl.core.get_int_dtype(32, True) == tl.int32
assert tl.core.get_int_dtype(8, False) == tl.uint8
assert tl.core.get_int_dtype(1, False) == tl.int1
```

---

## 3. constexpr Class

```python
class constexpr(base_value)
```

Wraps a value that is known at compile time. `constexpr` values have no IR representation; they are resolved during kernel compilation and substituted directly into the generated code.

**Constructor**:

```python
constexpr(value)
```

If `value` is already a `constexpr`, it is unwrapped automatically.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `value` | any | The wrapped compile-time value |
| `type` | `constexpr_type` | The type of this constexpr |

**Operator Overloads** (all return new `constexpr` instances):

| Operator | Method | Description |
|----------|--------|-------------|
| `+` | `__add__`, `__radd__` | Addition |
| `-` | `__sub__`, `__rsub__` | Subtraction |
| `*` | `__mul__`, `__rmul__` | Multiplication |
| `%` | `__mod__` | Modulo |
| `/` | `__truediv__`, `__rtruediv__` | True division |
| `//` | `__floordiv__`, `__rfloordiv__` | Floor division |
| `>` | `__gt__`, `__rgt__` | Greater than |
| `>=` | `__ge__`, `__rge__` | Greater or equal |
| `<` | `__lt__`, `__rlt__` | Less than |
| `<=` | `__le__`, `__rle__` | Less or equal |
| `==` | `__eq__` | Equality |
| `!=` | `__ne__` | Inequality |
| `&` | `__and__` | Bitwise AND |
| `\|` | `__or__` | Bitwise OR |
| `^` | `__xor__` | Bitwise XOR |
| `>>` | `__rshift__` | Right shift |
| `<<` | `__lshift__` | Left shift |
| `**` | `__pow__`, `__rpow__` | Exponentiation |
| `-` (unary) | `__neg__` | Negation |
| `+` (unary) | `__pos__` | Positive |
| `~` | `__invert__` | Bitwise NOT |

**Additional Methods**:

| Method | Description |
|--------|-------------|
| `logical_and(other)` | Logical AND (uses `and` keyword) |
| `logical_or(other)` | Logical OR (uses `or` keyword) |
| `__not__()` | Logical NOT |
| `__bool__()` | Returns `bool(self.value)` |
| `__index__()` | Returns `self.value` (for indexing) |
| `__iter__()` | Returns `iter(self.value)` |
| `__call__(*args, **kwds)` | Calls `self.value(*args, **kwds)` |
| `__getitem__(*args)` | Indexes into `self.value` |
| `__hash__()` | Hash based on `(value, type)` |
| `__repr__()` | Returns `"constexpr[value]"` |
| `_set_name(builder, name)` | No-op |
| `_flatten_ir(handles)` | No-op (constexprs have no IR handles) |

**Module-level constant**: `CONSTEXPR_0 = constexpr(0)`

**Example**:

```python
import triton.language as tl

@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE is a constexpr
    offsets = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE resolved at compile time
    data = tl.load(ptr + offsets)
```

### _unwrap_if_constexpr

```python
def _unwrap_if_constexpr(o) -> any
```

If `o` is a `constexpr`, returns `o.value`. Recursively unwraps lists and tuples. Otherwise returns `o` unchanged.

---

## 4. const Annotation Class

```python
class const
```

A type annotation used to mark pointers to constant (read-only) data. When a pointer is annotated with `const`, the `store` function cannot be called with that pointer.

Constness is part of the pointer type, and the usual Triton type consistency rules apply: a function cannot return a constant pointer in one return statement and a non-constant pointer in another.

**Example**:

```python
import triton.language as tl

@triton.jit
def kernel(ptr: tl.const, N: tl.constexpr):
    # ptr is read-only; tl.store(ptr, ...) would be an error
    data = tl.load(ptr + tl.arange(0, N))
```

---

## 5. tensor Class

```python
class tensor(base_value)
```

Represents an N-dimensional array of values or pointers. `tensor` is the fundamental data structure in Triton programs. Most functions in `triton.language` operate on and return tensors.

**Constructor**:

```python
tensor(handle, type: dtype)
```

Not called directly by user code. Created internally by the JIT compiler.

**Instance Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `handle` | `ir.value` | The underlying IR handle |
| `shape` | `tuple[constexpr]` | The shape of the tensor (empty tuple for scalars) |
| `numel` | `constexpr` | Total number of elements (product of shape) |
| `type` | `dtype` | The full type (can be `block_type` for multi-dimensional tensors) |
| `dtype` | `dtype` | The scalar element type |

**String Representation**:
`__str__` returns `"dtype[dim0, dim1, ...]"` (e.g., `"float32[16, 32]"`).

### Arithmetic Operator Overloads

| Operator | Method | Semantic Operation |
|----------|--------|--------------------|
| `a + b` | `__add__`, `__radd__` | `add(a, b, sanitize_overflow=True)` |
| `a - b` | `__sub__`, `__rsub__` | `sub(a, b, sanitize_overflow=True)` |
| `a * b` | `__mul__`, `__rmul__` | `mul(a, b, sanitize_overflow=True)` |
| `a / b` | `__truediv__`, `__rtruediv__` | `semantic.truediv(a, b)` |
| `a // b` | `__floordiv__`, `__rfloordiv__` | `semantic.floordiv(a, b)` |
| `a % b` | `__mod__`, `__rmod__` | `semantic.mod(a, b)` |
| `-a` | `__neg__` | `semantic.minus(a)` |
| `~a` | `__invert__` | `semantic.invert(a)` |

### Bitwise Operator Overloads

| Operator | Method | Semantic Operation |
|----------|--------|--------------------|
| `a & b` | `__and__`, `__rand__` | `semantic.and_(a, b)` |
| `a \| b` | `__or__`, `__ror__` | `semantic.or_(a, b)` |
| `a ^ b` | `__xor__`, `__rxor__` | `semantic.xor_(a, b)` |
| `a << b` | `__lshift__`, `__rlshift__` | `semantic.shl(a, b)` |
| `a >> b` | `__rshift__`, `__rrshift__` | `semantic.ashr(a, b)` (signed) or `semantic.lshr(a, b)` (unsigned) |

**Note**: Shift operations emit a warning if the shift value exceeds the bit width of the operand.

### Comparison Operator Overloads

All comparison operations implicitly broadcast their operands and return a tensor of `int1` (boolean).

| Operator | Method | Returns |
|----------|--------|---------|
| `a > b` | `__gt__`, `__rgt__` | Element-wise greater than |
| `a >= b` | `__ge__`, `__rge__` | Element-wise greater or equal |
| `a < b` | `__lt__`, `__rlt__` | Element-wise less than |
| `a <= b` | `__le__`, `__rle__` | Element-wise less or equal |
| `a == b` | `__eq__`, `__req__` | Element-wise equality |
| `a != b` | `__ne__`, `__rne__` | Element-wise inequality |

### Logical Operator Overloads

| Operator | Method | Description |
|----------|--------|-------------|
| `a and b` | `logical_and` | Logical AND |
| `a or b` | `logical_or` | Logical OR |
| `not a` | `__not__` | Logical NOT |

### Member Functions (Type Stubs)

These are forward declarations of member functions added by the `_tensor_member_fn` decorator. They can be called as `x.method_name(...)` instead of `tl.method_name(x, ...)`.

| Method | Forwarded To |
|--------|-------------|
| `x.broadcast_to(*shape)` | `broadcast_to(x, *shape)` |
| `x.trans(*dims)` | `trans(x, *dims)` |
| `x.permute(*dims)` | `permute(x, *dims)` |
| `x.split()` | `split(x)` |
| `x.view(*shape)` | `view(x, *shape)` |
| `x.reshape(*shape)` | `reshape(x, *shape)` |
| `x.expand_dims(axis)` | `expand_dims(x, axis)` |
| `x.cast(dtype)` | `cast(x, dtype)` |
| `x.store(value, ...)` | `store(x, value, ...)` |
| `x.advance(offsets)` | `advance(x, offsets)` |
| `x.atomic_cas(cmp, val)` | `atomic_cas(x, cmp, val)` |
| `x.atomic_xchg(val, ...)` | `atomic_xchg(x, val, ...)` |
| `x.atomic_add(val, ...)` | `atomic_add(x, val, ...)` |
| `x.atomic_max(val, ...)` | `atomic_max(x, val, ...)` |
| `x.atomic_min(val, ...)` | `atomic_min(x, val, ...)` |
| `x.atomic_and(val, ...)` | `atomic_and(x, val, ...)` |
| `x.atomic_or(val, ...)` | `atomic_or(x, val, ...)` |
| `x.atomic_xor(val, ...)` | `atomic_xor(x, val, ...)` |
| `x.reduce(axis, combine_fn)` | `reduce(x, axis, combine_fn)` |
| `x.associative_scan(axis, combine_fn)` | `associative_scan(x, axis, combine_fn)` |
| `x.gather(indices, axis)` | `gather(x, indices, axis)` |
| `x.histogram(num_bins)` | `histogram(x, num_bins)` |
| `x.cdiv(div)` | `cdiv(x, div)` |
| `x.ravel()` | `ravel(x)` |
| `x.max(axis, ...)` | `max(x, axis, ...)` |
| `x.argmax(axis, ...)` | `argmax(x, axis, ...)` |
| `x.min(axis, ...)` | `min(x, axis, ...)` |
| `x.argmin(axis, ...)` | `argmin(x, axis, ...)` |
| `x.sum(axis, ...)` | `sum(x, axis, ...)` |
| `x.xor_sum(axis, ...)` | `xor_sum(x, axis, ...)` |
| `x.reduce_or(axis, ...)` | `reduce_or(x, axis, ...)` |
| `x.cumsum(axis)` | `cumsum(x, axis)` |
| `x.cumprod(axis)` | `cumprod(x, axis)` |
| `x.sort(dim, descending)` | `sort(x, dim, descending)` |
| `x.flip(dim)` | `flip(x, dim)` |

### Special Methods

#### `x.to(dtype, fp_downcast_rounding=None, bitcast=False)`

Alias for `cast(x, dtype, ...)`. Converts the tensor to the specified dtype.

```python
@triton.jit
def kernel(ptr):
    x = tl.load(ptr)
    y = x.to(tl.float16)  # Equivalent to tl.cast(x, tl.float16)
```

#### `x.T`

Property that transposes a 2D tensor. Raises `AssertionError` if called directly; transposition must be created by the AST Visitor (i.e., use `x.trans()` or `tl.trans(x)`).

#### `x.__getitem__(slices)`

Supports tensor indexing with `None` to add dimensions (unsqueeze). Other slice patterns are not supported in the `__getitem__` implementation.

```python
@triton.jit
def kernel(x):
    # Add new dimensions using None
    y = x[:, None, :]  # Inserts a new axis at dimension 1
```

---

## 6. tuple Class

```python
class tuple(base_value)
```

A Triton tuple value. Used for returning multiple values from functions and for grouping related tensors.

**Constructor**:

```python
tuple(args: Sequence, type: Optional[tuple_type] = None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `args` | `Sequence` | The values in the tuple |
| `type` | `tuple_type` or `None` | The type; if `None`, inferred from values |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `__getitem__(idx)` | `(idx: constexpr) -> value` | Returns the value at the given index |
| `__getattr__(name)` | `-> value` | Returns the value for the named field (if named tuple) |
| `__add__(other)` | `-> tuple` | Concatenates two tuples |
| `__mul__(other)` | `-> tuple` | Repeats tuple N times (constexpr) |
| `__eq__(other)` | `-> constexpr` | Element-wise equality |
| `__hash__()` | `-> int` | Hash of values |
| `__str__()` | `-> str` | String representation |
| `__iter__()` | `-> iterator` | Iterates over values |
| `__len__()` | `-> int` | Number of elements |
| `_setitem(idx, value)` | `-> None` | Sets element at index (internal use) |
| `_set_name(builder, name)` | `-> None` | Sets names for all elements |
| `_flatten_ir(handles)` | `-> None` | Flattens all element IR handles |

---

## 7. slice Class

```python
class slice
```

Represents a slice specification (like Python's built-in `slice`).

**Constructor**:

```python
slice(start, stop, step)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | any | Start index |
| `stop` | any | Stop index |
| `step` | any | Step size |

**Attributes**: `start`, `stop`, `step`, `type` (`slice_type()`).

---

## 8. Programming Model

### program_id

```python
@builtin
def program_id(axis, _semantic=None) -> tensor
```

Returns the ID of the current program instance along the given `axis`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `int` or `constexpr` | The axis of the 3D launch grid. Must be 0, 1, or 2. |

**Returns**: A scalar tensor of type `int32` containing the program ID for the specified axis.

**Example**:

```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Program ID along axis 0
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
```

### num_programs

```python
@builtin
def num_programs(axis, _semantic=None) -> tensor
```

Returns the total number of program instances launched along the given `axis`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `int` or `constexpr` | The axis of the 3D launch grid. Must be 0, 1, or 2. |

**Returns**: A scalar tensor of type `int32` containing the grid size for the specified axis.

**Example**:

```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    # Use pid and num_pids for work distribution
```

---

## 9. Tensor Creation

### arange

```python
@builtin
def arange(start, end, _semantic=None) -> tensor
```

Returns contiguous values within the half-open interval `[start, end)`.

**Constraints**:
- `end - start` must be a power of two.
- `end - start` must be less than or equal to `TRITON_MAX_TENSOR_NUMEL`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `int` or `constexpr` | Start of the interval. Must be a power of two. |
| `end` | `int` or `constexpr` | End of the interval. Must be a power of two greater than `start`. |

**Returns**: A 1D tensor of `int32` with shape `(end - start,)`.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    # Create indices [0, 1, 2, ..., BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    data = tl.load(ptr + offsets)
```

### full

```python
@builtin
def full(shape, value, dtype, _semantic=None) -> tensor
```

Returns a tensor filled with the scalar `value` for the given `shape` and `dtype`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int]` | Shape of the new array, e.g., `(8, 16)` or `(8,)` |
| `value` | scalar | A scalar value to fill the array with |
| `dtype` | `dtype` | Data type of the new array, e.g., `tl.float16` |

**Returns**: A tensor of the specified shape and dtype, filled with `value`.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    ones = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1.0, tl.float32)
    zeros = tl.full((BLOCK_SIZE,), 0, tl.int32)
```

### zeros

```python
@jit
def zeros(shape, dtype) -> tensor
```

Returns a tensor filled with zeros for the given `shape` and `dtype`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int]` | Shape of the new array |
| `dtype` | `dtype` | Data type of the new array |

**Returns**: A zero-filled tensor.

**Example**:

```python
@triton.jit
def kernel(ptr, N: tl.constexpr):
    acc = tl.zeros((N, N), tl.float32)
```

### zeros_like

```python
@jit
def zeros_like(input) -> tensor
```

Returns a tensor of zeros with the same shape and type as the given tensor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | Input tensor to match shape and dtype |

**Returns**: A zero-filled tensor with the same shape and dtype as `input`.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    x = tl.load(ptr + tl.arange(0, BLOCK_SIZE))
    acc = tl.zeros_like(x)
```

### to_tensor

```python
@builtin
def to_tensor(x, _semantic=None) -> tensor
```

Converts a Python scalar or constexpr to a Triton tensor. This is typically called internally.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | scalar or constexpr | Value to convert |

**Returns**: A scalar tensor wrapping the value.

### cast

```python
@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None) -> tensor
```

Casts a tensor to the given `dtype`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `tensor` | required | The input tensor |
| `dtype` | `dtype` | required | The target data type |
| `fp_downcast_rounding` | `str` or `None` | `None` | Rounding mode for downcasting floating-point. Values: `"rtne"` (round to nearest, ties to even), `"rtz"` (round towards zero). Only used when `input` is FP and `dtype` has smaller bitwidth. |
| `bitcast` | `bool` | `False` | If `True`, reinterpret the bits without numerical conversion |

**Returns**: A tensor of the specified `dtype`.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    x = tl.load(ptr + tl.arange(0, BLOCK_SIZE))  # float32
    y = tl.cast(x, tl.float16)                     # cast to float16
    z = tl.cast(x, tl.int32, bitcast=True)         # reinterpret bits as int32
```

---

## 10. Memory Operations

### load

```python
@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="",
         cache_modifier="", eviction_policy="", volatile=False, _semantic=None) -> tensor
```

Loads data from memory at locations defined by `pointer`. Supports three modes:

1. **Scalar pointer**: Load a single scalar value.
2. **Tensor of pointers**: Load an N-dimensional tensor.
3. **Block pointer** (from `make_block_ptr`): Load a block with boundary checking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pointer` | `tensor` | required | Pointer to the data. Can be a scalar pointer, block of pointers, or block pointer. |
| `mask` | `tensor` or `None` | `None` | If `mask[idx]` is `False`, do not load at `pointer[idx]`. Must be `None` for block pointers. |
| `other` | `tensor` or `None` | `None` | If `mask[idx]` is `False`, return `other[idx]`. If `None`, masked-out value is undefined. |
| `boundary_check` | `tuple[int]` | `()` | Dimensions to do boundary checking on (for block pointers). |
| `padding_option` | `str` | `""` | Padding for out-of-bounds: `""` (undefined), `"zero"`, or `"nan"`. |
| `cache_modifier` | `str` | `""` | PTX cache option: `""`, `".ca"` (cache all levels), `".cg"` (cache global level), `".cv"` (don't cache). |
| `eviction_policy` | `str` | `""` | PTX eviction policy. |
| `volatile` | `bool` | `False` | PTX volatile option. |

**Returns**: A tensor containing the loaded values.

**Example**:

```python
@triton.jit
def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Basic load with mask
    x = tl.load(ptr + offsets, mask=mask)

    # Load with default value for masked elements
    x = tl.load(ptr + offsets, mask=mask, other=0.0)

    # Load with cache hint
    x = tl.load(ptr + offsets, mask=mask, cache_modifier=".cg")
```

### store

```python
@_tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None)
```

Stores a tensor of data into memory locations defined by `pointer`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pointer` | `tensor` | required | Memory location(s) to store to. Scalar pointer, block of pointers, or block pointer. |
| `value` | `tensor` | required | The tensor of elements to be stored. Implicitly broadcast and typecast. |
| `mask` | `tensor` or `None` | `None` | If `mask[idx]` is `False`, do not store `value[idx]`. |
| `boundary_check` | `tuple[int]` | `()` | Dimensions for boundary checking (block pointers only). |
| `cache_modifier` | `str` | `""` | PTX cache option: `""`, `".wb"` (write-back), `".cg"` (cache global), `".cs"` (cache streaming), `".wt"` (write-through). |
| `eviction_policy` | `str` | `""` | PTX eviction policy: `""`, `"evict_first"`, `"evict_last"`. |

**Example**:

```python
@triton.jit
def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(ptr + offsets, mask=mask) * 2.0
    tl.store(ptr + offsets, x, mask=mask)
```

### make_block_ptr

```python
@builtin
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _semantic=None) -> _block_ptr
```

Creates a pointer to a block within a parent tensor. This is the deprecated block pointer API; prefer `make_tensor_descriptor`.

> **Deprecated**: Use `TensorDescriptor` or `tl.make_tensor_descriptor` instead.

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `tensor` | The base pointer to the parent tensor (must be a scalar pointer) |
| `shape` | `list[int or tensor]` | The shape of the parent tensor |
| `strides` | `list[int or tensor]` | The strides of the parent tensor |
| `offsets` | `list[int or tensor]` | The initial offsets into the tensor |
| `block_shape` | `list[int]` | The shape of the block to load/store. Must contain positive integers. |
| `order` | `list[int]` | The memory layout order. Must be a permutation of `0..rank-1`. |

**Returns**: A `_block_ptr` object.

**Example**:

```python
@triton.jit
def kernel(ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=[M, N],
        strides=[N, 1],
        offsets=[pid_m * BLOCK_M, pid_n * BLOCK_N],
        block_shape=[BLOCK_M, BLOCK_N],
        order=[1, 0],
    )
    x = tl.load(block_ptr, boundary_check=[0, 1])
```

### advance

```python
@must_use_result("Note that tl.advance does not have any side effects. ...")
@_tensor_member_fn
@builtin
def advance(base, offsets, _semantic=None) -> _block_ptr
```

Advances a block pointer by the given offsets. Returns a new block pointer; the original is not modified.

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `_block_ptr` | The block pointer to advance |
| `offsets` | `tuple[int or tensor]` | The offsets to advance per dimension |

**Returns**: A new `_block_ptr` with updated offsets.

**Important**: `tl.advance` has no side effects. You must assign the result to a variable.

**Example**:

```python
@triton.jit
def kernel(ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    block_ptr = tl.make_block_ptr(
        base=ptr, shape=[M, N], strides=[N, 1],
        offsets=[0, 0], block_shape=[BLOCK_M, BLOCK_N], order=[1, 0],
    )
    for m in range(0, M, BLOCK_M):
        x = tl.load(block_ptr, boundary_check=[0, 1])
        # Process x...
        block_ptr = tl.advance(block_ptr, [BLOCK_M, 0])  # Must assign!
```

### make_tensor_descriptor

```python
@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor
```

Creates a tensor descriptor object that represents a tensor in global memory. On NVIDIA GPUs with TMA (Tensor Memory Accelerator) support, this creates a TMA descriptor and loads/stores are backed by TMA hardware.

**Constraints**:
- `base` must be 16-byte aligned.
- Leading dimension strides must be multiples of 16 bytes.
- The last dimension must be contiguous.
- Currently supports 2-5 dimensional tensors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base` | `tensor` | required | Base pointer of the tensor (16-byte aligned) |
| `shape` | `List[tensor]` | required | Non-negative integers representing tensor shape |
| `strides` | `List[tensor]` | required | Tensor strides; leading dims must be 16-byte multiples, last dim contiguous |
| `block_shape` | `List[constexpr]` | required | Shape of block to load/store from global memory |
| `padding_option` | `str` | `"zero"` | Padding for out-of-bounds access |

**Returns**: A `tensor_descriptor` object.

**Example**:

```python
import triton
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))

# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

M, N = 256, 256
x = torch.randn(M, N, device="cuda")
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```

### load_tensor_descriptor

```python
@builtin
def load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor], _semantic=None) -> tensor
```

Loads a block of data from a tensor descriptor at the given offsets. Values outside the tensor bounds will be filled with zeros.

| Parameter | Type | Description |
|-----------|------|-------------|
| `desc` | `tensor_descriptor_base` | The tensor descriptor |
| `offsets` | `Sequence[constexpr or tensor]` | Element offsets per dimension. Must be a multiple of 16 bytes. |

**Returns**: A tensor containing the loaded block.

**Note**: This is equivalent to calling `desc.load(offsets)`.

### store_tensor_descriptor

```python
@builtin
def store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor],
                            value: tensor, _semantic=None) -> tensor
```

Stores a block of data to a tensor descriptor at the given offsets. Values outside the tensor bounds will be ignored.

| Parameter | Type | Description |
|-----------|------|-------------|
| `desc` | `tensor_descriptor_base` | The tensor descriptor |
| `offsets` | `Sequence[constexpr or tensor]` | Element offsets per dimension |
| `value` | `tensor` | The block to store |

**Note**: This is equivalent to calling `desc.store(offsets, value)`.

---

## 11. Linear Algebra

### dot

```python
@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None,
        max_num_imprecise_acc=None, out_dtype=float32, _semantic=None) -> tensor
```

Computes the matrix product of two blocks. Both inputs must be 2D or 3D tensors with compatible inner dimensions. For 3D inputs, performs batched matrix multiplication (first dimension is batch).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `tensor` | required | First tensor. 2D or 3D. Supported scalar types: `int8`, `float8e5`, `float16`, `bfloat16`, `float32`. |
| `other` | `tensor` | required | Second tensor. Must have compatible inner dimension. |
| `acc` | `tensor` or `None` | `None` | Accumulator tensor. If provided, the result is added to this. |
| `input_precision` | `str` or `None` | `None` | Precision for f32 x f32: `"tf32"`, `"tf32x3"`, `"ieee"`. Default depends on hardware support. |
| `allow_tf32` | `bool` or `None` | `None` | **Deprecated.** If `True`, sets `input_precision="tf32"`. |
| `max_num_imprecise_acc` | `int` or `None` | `None` | Maximum number of imprecise accumulations. |
| `out_dtype` | `dtype` | `float32` | Output data type. |

**Shape Requirements**:
- `input` and `other` must have the same rank (>= 2).
- Batch dimensions must match.
- `input.shape[-1] == other.shape[-2]` (inner dimensions).
- Output shape: `input.shape[:-1] + [other.shape[-1]]`.

**Warning**: When using TF32 precision, float32 inputs may be truncated to TF32 format (19-bit) without rounding, which may bias results. For best results, round to TF32 explicitly or use `TensorDescriptor` with `round_f32_to_tf32=True`.

**Example**:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator, mask=offs_m[:, None] < M)
```

### dot_scaled

```python
@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
               acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True,
               out_dtype=float32, _semantic=None) -> tensor
```

Computes the matrix product of two blocks in microscaling (MX) format, following the OCP Microscaling Formats specification.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lhs` | `tensor` | required | Left-hand side tensor (2D). fp4 packed into uint8, fp8 as uint8/fp8 type, or bf16. |
| `lhs_scale` | `tensor` or `None` | required | Scale factor for lhs. Shape: `[M, K // group_size]` where group_size is 32 for e8m0 scales. |
| `lhs_format` | `str` | required | Format: `"e2m1"`, `"e4m3"`, `"e5m2"`, `"bf16"`, or `"fp16"`. |
| `rhs` | `tensor` | required | Right-hand side tensor (2D). |
| `rhs_scale` | `tensor` or `None` | required | Scale factor for rhs. Shape: `[N, K // group_size]`. Do NOT transpose. |
| `rhs_format` | `str` | required | Format (same options as `lhs_format`). |
| `acc` | `tensor` or `None` | `None` | Accumulator. |
| `fast_math` | `bool` | `False` | Enable fast math mode. |
| `lhs_k_pack` | `bool` | `True` | If `False`, lhs is packed along M dimension instead of K. |
| `rhs_k_pack` | `bool` | `True` | If `False`, rhs is packed along N dimension instead of K. |
| `out_dtype` | `dtype` | `float32` | Currently only `float32` is supported. |

**Returns**: A 2D tensor of `float32`.

**Note**: Software emulation upcasts microscaled inputs to `bf16` (or `fp16` on AMD CDNA3 when one input is `fp16`).

**Example**:

```python
@triton.jit
def mx_matmul(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, M, N, K,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load FP8 inputs and their scales
    a = tl.load(a_ptr + ...)   # [BLOCK_M, BLOCK_K] uint8
    a_scale = tl.load(a_scale_ptr + ...)  # [BLOCK_M, BLOCK_K/32] uint8 (e8m0)
    b = tl.load(b_ptr + ...)   # [BLOCK_K, BLOCK_N] uint8
    b_scale = tl.load(b_scale_ptr + ...)  # [BLOCK_N, BLOCK_K/32] uint8 (e8m0)

    acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3")
    tl.store(c_ptr + ..., acc)
```

---

## 12. Shape Manipulation

### broadcast

```python
@builtin
def broadcast(input, other, _semantic=None) -> tuple[tensor, tensor]
```

Broadcasts two tensors to a common compatible shape. Returns both broadcast tensors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | First input tensor |
| `other` | `tensor` | Second input tensor |

**Returns**: A tuple of two tensors, both broadcast to the common shape.

### broadcast_to

```python
@_tensor_member_fn
@builtin
def broadcast_to(input, *shape, _semantic=None) -> tensor
```

Broadcasts the input tensor to the specified `shape`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `shape` | `tuple[int]` or varargs | The desired shape. Can be passed as a tuple or individual parameters. |

**Returns**: A tensor with the specified shape.

**Example**:

```python
@triton.jit
def kernel(ptr, M: tl.constexpr, N: tl.constexpr):
    x = tl.arange(0, N)           # shape: (N,)
    y = tl.broadcast_to(x, M, N)  # shape: (M, N) - broadcast along dim 0

    # Equivalent to:
    y = tl.broadcast_to(x, (M, N))
```

### reshape

```python
@_tensor_member_fn
@builtin
def reshape(input, *shape, can_reorder=False, _semantic=None, _generator=None) -> tensor
```

Returns a tensor with the same number of elements as `input` but with the provided shape.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `tensor` | required | The input tensor |
| `shape` | `tuple[int]` or varargs | required | The new shape |
| `can_reorder` | `bool` | `False` | If `True`, the compiler may reorder elements for better performance |

**Returns**: A reshaped tensor.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    x = tl.arange(0, BLOCK_SIZE * BLOCK_SIZE)  # (BLOCK_SIZE * BLOCK_SIZE,)
    y = tl.reshape(x, BLOCK_SIZE, BLOCK_SIZE)   # (BLOCK_SIZE, BLOCK_SIZE)
    z = tl.reshape(y, BLOCK_SIZE * BLOCK_SIZE)  # (BLOCK_SIZE * BLOCK_SIZE,)

    # These are equivalent:
    y = tl.reshape(x, (BLOCK_SIZE, BLOCK_SIZE))
    y = tl.reshape(x, BLOCK_SIZE, BLOCK_SIZE)
```

### expand_dims

```python
@_tensor_member_fn
@builtin
def expand_dims(input, axis, _semantic=None) -> tensor
```

Inserts new length-1 dimensions at the specified positions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `axis` | `int` or `Sequence[int]` | The position(s) to insert new axes. Indices are relative to the result shape. Negative indices are supported. |

**Returns**: A tensor with the new dimensions inserted.

**Example**:

```python
@triton.jit
def kernel(ptr, M: tl.constexpr, N: tl.constexpr):
    x = tl.arange(0, N)              # shape: (N,)
    y = tl.expand_dims(x, 0)          # shape: (1, N)
    z = tl.expand_dims(x, 1)          # shape: (N, 1)

    # Insert multiple axes at once
    w = tl.expand_dims(x, [0, 2])     # shape: (1, N, 1)
```

### permute

```python
@_tensor_member_fn
@builtin
def permute(input, *dims, _semantic=None) -> tensor
```

Permutes the dimensions of a tensor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `dims` | `tuple[int]` or varargs | The desired ordering of dimensions. E.g., `(2, 1, 0)` reverses dims of a 3D tensor. |

**Returns**: A tensor with permuted dimensions.

**Example**:

```python
@triton.jit
def kernel(ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    x = tl.full((M, N, K), 1.0, tl.float32)

    # Reverse dimensions
    y = tl.permute(x, 2, 1, 0)  # shape: (K, N, M)

    # These are equivalent:
    y = tl.permute(x, (2, 1, 0))
    y = tl.permute(x, 2, 1, 0)
```

### trans

```python
@_tensor_member_fn
@builtin
def trans(input: tensor, *dims, _semantic=None) -> tensor
```

Permutes the dimensions of a tensor. If `dims` is not specified, defaults to swapping the last two axes (batched 2D transpose).

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `dims` | `tuple[int]` or varargs | The desired ordering. If empty, swaps last two axes. |

**Returns**: A transposed tensor.

**Example**:

```python
@triton.jit
def kernel(ptr, M: tl.constexpr, N: tl.constexpr):
    x = tl.full((M, N), 1.0, tl.float32)

    # Default: swap last two axes
    y = tl.trans(x)          # shape: (N, M)

    # Explicit permutation
    y = tl.trans(x, 1, 0)    # shape: (N, M)
```

### view

```python
@_tensor_member_fn
@builtin
def view(input, *shape, _semantic=None) -> tensor
```

Returns a tensor with the same elements as `input` but a different shape. The order of elements may not be preserved.

> **Deprecated**: Use `reshape` with `can_reorder=True` instead.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `shape` | `tuple[int]` or varargs | The desired shape |

**Returns**: A reshaped tensor (elements may be reordered).

### cat

```python
@builtin
def cat(input, other, can_reorder=False, dim=0, _semantic=None) -> tensor
```

Concatenates two tensors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `tensor` | required | First tensor |
| `other` | `tensor` | required | Second tensor |
| `can_reorder` | `bool` | `False` | Compiler hint: if `True`, may reorder elements. Use when order doesn't matter (e.g., result only used in reductions). |
| `dim` | `int` | `0` | Dimension to concatenate along (when `can_reorder=False`). |

**Returns**: A concatenated tensor.

**Constraints** (when `can_reorder=False`):
- Both tensors must have the same rank.
- Dimensions must match except at `dim`.

**Example**:

```python
@triton.jit
def kernel(ptr, N: tl.constexpr):
    a = tl.arange(0, N)
    b = tl.arange(N, 2 * N)

    # Concatenate along dim 0
    c = tl.cat(a, b, dim=0)  # shape: (2*N,)

    # When order doesn't matter (e.g., for reduction)
    c = tl.cat(a, b, can_reorder=True)
```

### split

```python
@_tensor_member_fn
@builtin
def split(a, _semantic=None, _generator=None) -> tuple[tensor, tensor]
```

Splits a tensor in two along its last dimension, which must have size 2.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `tensor` | The tensor to split. Last dimension must be 2. |

**Returns**: A tuple of two tensors.

**Example**:

```python
@triton.jit
def kernel(ptr, N: tl.constexpr):
    x = tl.full((N, 2), 0.0, tl.float32)
    a, b = tl.split(x)  # a and b both have shape (N,)

    # split is the inverse of join
    joined = tl.join(a, b)  # shape: (N, 2)
```

### join

```python
@builtin
def join(a, b, _semantic=None) -> tensor
```

Joins two tensors along a new minor dimension. The inputs are broadcast to the same shape.

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `tensor` | First tensor |
| `b` | `tensor` | Second tensor |

**Returns**: A tensor with an extra trailing dimension of size 2.

**Example**:

```python
@triton.jit
def kernel(ptr, N: tl.constexpr):
    a = tl.arange(0, N)         # shape: (N,)
    b = tl.arange(N, 2 * N)     # shape: (N,)
    c = tl.join(a, b)            # shape: (N, 2)
```

### ravel

```python
@_tensor_member_fn
@jit
def ravel(x, can_reorder=False) -> tensor
```

Returns a contiguous flattened view of the input tensor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` | required | Input tensor |
| `can_reorder` | `bool` | `False` | Whether elements may be reordered |

**Returns**: A 1D tensor with `x.numel` elements.

**Example**:

```python
@triton.jit
def kernel(ptr, M: tl.constexpr, N: tl.constexpr):
    x = tl.full((M, N), 1.0, tl.float32)
    flat = tl.ravel(x)   # shape: (M * N,)
```

### item

```python
@_tensor_member_fn
@builtin
def item(input, _semantic=None, _generator=None) -> scalar
```

Converts a single-element tensor into a scalar.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | A single-element tensor |

**Returns**: A scalar value.

### gather

```python
@_tensor_member_fn
@builtin
def gather(src, index, axis, _semantic=None) -> tensor
```

Gathers values from a source tensor along a given dimension using indices.

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `tensor` | The source tensor |
| `index` | `tensor` | The index tensor |
| `axis` | `int` | The dimension to gather along |

**Returns**: A tensor of gathered values.

---

## 13. Reductions

### reduce

```python
@_tensor_member_fn
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None) -> tensor or tuple[tensor]
```

Applies the `combine_fn` to all elements along the provided `axis`. The combine function must be associative and commutative.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `tensor` or `tuple[tensor]` | required | Input tensor(s) to reduce. If a tuple, all tensors are reduced together. |
| `axis` | `int` or `None` | required | Dimension to reduce. If `None`, reduce all dimensions. |
| `combine_fn` | `Callable` | required | Function to combine two groups of scalar tensors. Must be marked with `@triton.jit`. |
| `keep_dims` | `bool` | `False` | If `True`, keep reduced dimensions with length 1. |

**Returns**: Reduced tensor(s). If `input` is a tuple, returns a tuple of tensors.

**Example**:

```python
import triton
import triton.language as tl

@triton.jit
def sum_combine(a, b):
    return a + b

@triton.jit
def kernel(ptr, N: tl.constexpr):
    x = tl.load(ptr + tl.arange(0, N))
    total = tl.reduce(x, axis=0, combine_fn=sum_combine)
```

### associative_scan

```python
@_tensor_member_fn
@builtin
def associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None) -> tensor or tuple[tensor]
```

Applies the `combine_fn` as an associative scan (prefix scan) along the given `axis`. Each element is combined with a running carry that is updated along the axis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `tensor` or `tuple[tensor]` | required | Input tensor(s). If a tuple, all tensors are scanned together. |
| `axis` | `int` | required | Dimension along which to scan. |
| `combine_fn` | `Callable` | required | Function to combine two groups of scalar tensors. Must be marked with `@triton.jit`. |
| `reverse` | `bool` | `False` | If `True`, scan in the reverse direction. |

**Returns**: Scanned tensor(s). If `input` is a tuple, returns a tuple of tensors.

**Example**:

```python
@triton.jit
def add_combine(a, b):
    return a + b

@triton.jit
def cumsum_kernel(ptr, N: tl.constexpr):
    x = tl.load(ptr + tl.arange(0, N))
    prefix_sum = tl.associative_scan(x, axis=0, combine_fn=add_combine)
    tl.store(ptr + tl.arange(0, N), prefix_sum)
```

### histogram

```python
@_tensor_member_fn
@builtin
def histogram(input, num_bins, mask=None, _semantic=None, _generator=None) -> tensor
```

Computes a histogram of the input tensor. Bins have width 1 starting at 0.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | Input tensor with integer values |
| `num_bins` | `int` or `constexpr` | Number of histogram bins |
| `mask` | `tensor` or `None` | If `mask[idx]` is `False`, exclude `input[idx]` from the histogram |

**Returns**: A 1D tensor of shape `(num_bins,)` containing the histogram counts.

### _reduce_with_indices

```python
@builtin
def _reduce_with_indices(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None) -> tuple[tensor, tensor]
```

Internal helper for reductions that also track the index of the extremal element (used by `argmax`, `argmin`). Returns both the reduced values and the corresponding indices.

---

## 14. Atomic Operations

All atomic operations return the data stored at `pointer` **before** the atomic operation was performed.

### Common Parameters

All atomic operations share these common parameters (except `atomic_cas`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pointer` | `tensor` | required | Memory location(s) to operate on. Block of `dtype=triton.PointerDType`. |
| `val` | `tensor` | required | Value(s) for the operation. Block of `dtype=pointer.dtype.element_ty`. |
| `mask` | `tensor` or `None` | `None` | If `mask[idx]` is `False`, skip the atomic at `pointer[idx]`. |
| `sem` | `str` or `None` | `None` | Memory semantics: `"acquire"`, `"release"`, `"acq_rel"`, `"relaxed"`. Default: `"acq_rel"`. |
| `scope` | `str` or `None` | `None` | Thread scope: `"gpu"` (default), `"cta"`, or `"sys"`. |

### atomic_cas

```python
@_tensor_member_fn
@builtin
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic compare-and-swap at the memory location specified by `pointer`. If the current value equals `cmp`, it is replaced with `val`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pointer` | `tensor` | Memory location(s). |
| `cmp` | `tensor` | Expected value(s). |
| `val` | `tensor` | New value(s) to store if comparison succeeds. |
| `sem` | `str` or `None` | Memory semantics. |
| `scope` | `str` or `None` | Thread scope. |

**Returns**: The old value at `pointer` (before the operation).

**Example**:

```python
@triton.jit
def lock_kernel(lock_ptr):
    # Try to acquire lock
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass  # Spin until lock is acquired
    # Critical section...
    tl.atomic_xchg(lock_ptr, 0)  # Release lock
```

### atomic_xchg

```python
@_tensor_member_fn
@builtin
def atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic exchange: stores `val` at `pointer` and returns the old value.

### atomic_add

```python
@_tensor_member_fn
@builtin
def atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic addition: `*pointer += val`. Returns the old value.

**Example**:

```python
@triton.jit
def atomic_add_kernel(ptr, val_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    val = tl.load(val_ptr + offsets, mask=mask)
    tl.atomic_add(ptr + offsets, val, mask=mask)
```

### atomic_max

```python
@_tensor_member_fn
@builtin
def atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic maximum: `*pointer = max(*pointer, val)`. Returns the old value.

### atomic_min

```python
@_tensor_member_fn
@builtin
def atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic minimum: `*pointer = min(*pointer, val)`. Returns the old value.

### atomic_and

```python
@_tensor_member_fn
@builtin
def atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic bitwise AND: `*pointer &= val`. Returns the old value.

### atomic_or

```python
@_tensor_member_fn
@builtin
def atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic bitwise OR: `*pointer |= val`. Returns the old value.

### atomic_xor

```python
@_tensor_member_fn
@builtin
def atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None) -> tensor
```

Performs an atomic bitwise XOR: `*pointer ^= val`. Returns the old value.

---

## 15. Conditioning and Math Helpers

### where

```python
@builtin
def where(condition, x, y, _semantic=None) -> tensor
```

Returns elements from `x` or `y` depending on `condition`.

**Important**: `x` and `y` are always evaluated regardless of `condition`. To avoid unintended memory operations, use the `mask` argument in `load`/`store` instead.

The shapes of `x` and `y` are broadcast to the shape of `condition`. `x` and `y` must have the same data type.

| Parameter | Type | Description |
|-----------|------|-------------|
| `condition` | `tensor` | When `True` (nonzero), yield `x`; otherwise yield `y`. Block of `triton.int1`. |
| `x` | `tensor` or scalar | Values selected where `condition` is `True`. |
| `y` | `tensor` or scalar | Values selected where `condition` is `False`. |

**Returns**: A tensor of the same shape as `condition` (after broadcasting).

**Example**:

```python
@triton.jit
def relu_kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0.0)  # ReLU activation
    tl.store(ptr + offsets, y, mask=mask)
```

### maximum

```python
@builtin
def maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None) -> tensor
```

Computes the element-wise maximum of `x` and `y`. Bfloat16 inputs are promoted to float32 internally.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` | required | First input tensor |
| `y` | `tensor` | required | Second input tensor |
| `propagate_nan` | `PropagateNan` | `PropagateNan.NONE` | Whether to propagate NaN values |

**Returns**: Element-wise maximum tensor.

### minimum

```python
@builtin
def minimum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None) -> tensor
```

Computes the element-wise minimum of `x` and `y`. Bfloat16 inputs are promoted to float32 internally.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` | required | First input tensor |
| `y` | `tensor` | required | Second input tensor |
| `propagate_nan` | `PropagateNan` | `PropagateNan.NONE` | Whether to propagate NaN values |

**Returns**: Element-wise minimum tensor.

### clamp

```python
@builtin
def clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None) -> tensor
```

Clamps the input tensor `x` within the range `[min, max]`. Behavior when `min > max` is undefined.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` | required | Input tensor |
| `min` | `tensor` | required | Lower bound |
| `max` | `tensor` | required | Upper bound |
| `propagate_nan` | `PropagateNan` | `PropagateNan.NONE` | Whether to propagate NaN. Applies only to `x`. |

**Returns**: A clamped tensor.

**Example**:

```python
@triton.jit
def clamp_kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(ptr + offsets)
    y = tl.clamp(x, 0.0, 1.0)  # Clip to [0, 1]
    tl.store(ptr + offsets, y)
```

### add

```python
@builtin
def add(x, y, sanitize_overflow: constexpr = True, _semantic=None) -> tensor
```

Element-wise addition of `x` and `y`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` or scalar | required | First operand |
| `y` | `tensor` or scalar | required | Second operand |
| `sanitize_overflow` | `constexpr` | `True` | Whether to sanitize integer overflow |

### sub

```python
@builtin
def sub(x, y, sanitize_overflow: constexpr = True, _semantic=None) -> tensor
```

Element-wise subtraction of `y` from `x`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` or scalar | required | First operand |
| `y` | `tensor` or scalar | required | Second operand |
| `sanitize_overflow` | `constexpr` | `True` | Whether to sanitize integer overflow |

### mul

```python
@builtin
def mul(x, y, sanitize_overflow: constexpr = True, _semantic=None) -> tensor
```

Element-wise multiplication of `x` and `y`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `tensor` or scalar | required | First operand |
| `y` | `tensor` or scalar | required | Second operand |
| `sanitize_overflow` | `constexpr` | `True` | Whether to sanitize integer overflow |

### builtin_max

```python
@builtin
def builtin_max(*args, propagate_nan=_NOTHING, _semantic=None) -> tensor or constexpr
```

Computes the maximum of multiple values. If all arguments are compile-time constants, returns a `constexpr`. Otherwise, uses `maximum` for element-wise computation.

**Note**: Deprecated for non-scalar tensor values; use `tl.maximum` instead.

### builtin_min

```python
@builtin
def builtin_min(*args, propagate_nan=_NOTHING, _semantic=None) -> tensor or constexpr
```

Computes the minimum of multiple values. If all arguments are compile-time constants, returns a `constexpr`. Otherwise, uses `minimum` for element-wise computation.

**Note**: Deprecated for non-scalar tensor values; use `tl.minimum` instead.

---

## 16. Compiler Hints

### multiple_of

```python
@builtin
def multiple_of(input, values, _semantic=None) -> tensor
```

Tells the compiler that all values in `input` are multiples of the corresponding entry in `values`. This enables the compiler to optimize memory access patterns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `values` | `list[constexpr]` or `constexpr` | The divisor(s). Each element must be a `constexpr[int]`. |

**Returns**: The input tensor (unchanged, but annotated with the hint).

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Tell compiler that offsets are multiples of BLOCK_SIZE
    offsets = tl.multiple_of(offsets, BLOCK_SIZE)
    data = tl.load(ptr + offsets)
```

### max_contiguous

```python
@builtin
def max_contiguous(input, values, _semantic=None) -> tensor
```

Tells the compiler that the first `value` elements in `input` are contiguous in memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `values` | `list[constexpr]` or `constexpr` | Number of contiguous elements per group. Each must be `constexpr[int]`. |

**Returns**: The input tensor (with the hint annotation).

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    # First BLOCK_SIZE elements are contiguous
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    data = tl.load(ptr + offsets)
```

### max_constancy

```python
@builtin
def max_constancy(input, values, _semantic=None) -> tensor
```

Tells the compiler that the first `value` elements in `input` are constant (all equal within each group). For example, if `values` is `[4]`, then each group of 4 values should all be equal, e.g., `[0, 0, 0, 0, 1, 1, 1, 1]`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `tensor` | The input tensor |
| `values` | `list[constexpr]` or `constexpr` | Group size(s) for constancy. Each must be `constexpr[int]`. |

**Returns**: The input tensor (with the hint annotation).

### assume

```python
@builtin
def assume(cond, _semantic=None)
```

Allows the compiler to assume that `cond` is `True`. This can enable optimizations that would not otherwise be possible.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cond` | `tensor` | A boolean condition assumed to be `True` |

**Warning**: If the condition is actually `False`, behavior is undefined.

**Example**:

```python
@triton.jit
def kernel(ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Assume N is always a multiple of BLOCK_SIZE
    tl.assume(N % BLOCK_SIZE == 0)
    data = tl.load(ptr + offsets)
```

---

## 17. Debugging

### static_print

```python
@builtin
def static_print(*values, sep: str = " ", end: str = "\n", file=None, flush=False, _semantic=None)
```

Prints values at **compile time**. Parameters are the same as Python's built-in `print`.

**Important**: Calling Python's built-in `print` is NOT the same as `tl.static_print`. The built-in `print` maps to `device_print` which has different requirements.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*values` | any | required | Values to print |
| `sep` | `str` | `" "` | Separator between values |
| `end` | `str` | `"\n"` | End string |
| `file` | any | `None` | Output file |
| `flush` | `bool` | `False` | Whether to flush |

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")  # Printed during compilation
```

### static_assert

```python
@builtin
def static_assert(cond, msg="", _semantic=None)
```

Asserts the condition at **compile time**. Does not require `TRITON_DEBUG` to be set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cond` | `bool` or `constexpr` | required | Condition to assert |
| `msg` | `str` | `""` | Error message if assertion fails |

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE == 1024, "BLOCK_SIZE must be 1024")
```

### device_print

```python
@builtin
def device_print(prefix, *args, hex=False, _semantic=None)
```

Prints values at **runtime** from the GPU device. The first argument must be a string prefix; subsequent arguments can be scalars or tensors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | `str` | required | A prefix string (must be ASCII printable) |
| `*args` | `tensor` or scalar | required | Values to print |
| `hex` | `bool` | `False` | Print values in hexadecimal |

**Notes**:
- Calling Python's built-in `print` is the same as calling this function.
- CUDA printfs use a buffer of limited size (~6912 KiB default). To increase: `triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)`.
- CUDA may raise an error if you try to change the buffer size after running a kernel that uses printfs.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tl.device_print("pid", pid)

    # Using built-in print (equivalent)
    print("pid", pid)
```

### device_assert

```python
@builtin
def device_assert(cond, msg="", mask=None, _semantic=None)
```

Asserts a condition at **runtime** from the GPU device. Requires `TRITON_DEBUG` environment variable to be set (to any value other than `0`) to have any effect.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cond` | `tensor` | required | Boolean condition to assert |
| `msg` | `str` | `""` | Message to print on failure |
| `mask` | `tensor` or `None` | `None` | Optional mask |

Using the Python `assert` statement is the same as calling this function, except the second argument must be provided and must be a string.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tl.device_assert(pid == 0)

    # Using Python assert (equivalent)
    assert pid == 0, "pid != 0"
```

### debug_barrier

```python
@builtin
def debug_barrier(_semantic=None)
```

Inserts a barrier to synchronize all threads in a block (CTA). This is a memory fence that ensures all prior memory operations are visible to all threads in the block before any subsequent operations proceed.

**Example**:

```python
@triton.jit
def kernel(ptr, BLOCK_SIZE: tl.constexpr):
    # ... write some data ...
    tl.debug_barrier()  # Ensure all writes are visible
    # ... read the data ...
```

---

## 18. Inline Assembly

### inline_asm_elementwise

```python
@builtin
def inline_asm_elementwise(asm: str, constraints: str, args: Sequence, dtype: Union[dtype, Sequence[dtype]],
                           is_pure: bool, pack: int, _semantic=None) -> tensor or tuple[tensor]
```

Executes inline assembly over a tensor element-wise. Input tensors are implicitly broadcast to the same shape. Each invocation processes `pack` elements at a time.

| Parameter | Type | Description |
|-----------|------|-------------|
| `asm` | `str` | Assembly code to run. Must match the target's assembly format (e.g., PTX for NVIDIA). |
| `constraints` | `str` | ASM constraints in LLVM format. See [LLVM LangRef](https://llvm.org/docs/LangRef.html#inline-asm-constraint-string). |
| `args` | `Sequence[tensor]` | Input tensors whose values are passed to the ASM block. |
| `dtype` | `dtype` or `Sequence[dtype]` | Element type(s) of the output. Can be a tuple for multiple outputs. |
| `is_pure` | `bool` | If `True`, the compiler assumes the ASM has no side effects. |
| `pack` | `int` | Number of elements processed per ASM invocation. |

**Returns**: One tensor or a tuple of tensors of the given dtypes.

**Notes**:
- Input elements smaller than 4 bytes are packed into 4-byte registers.
- This op does not support empty `dtype` -- the inline ASM must return at least one tensor.
- Which set of input elements a block receives is unspecified.

**Example**:

```python
@triton.jit
def kernel(A, B, C, D, BLOCK: tl.constexpr):
    a = tl.load(A + tl.arange(0, BLOCK))  # uint8 tensor
    b = tl.load(B + tl.arange(0, BLOCK))  # float32 tensor

    (c, d) = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 tmp<4>;
            mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
            cvt.u32.u8 $0, tmp0;
            cvt.u32.u8 $1, tmp1;
            cvt.u32.u8 $2, tmp2;
            cvt.u32.u8 $3, tmp3;
        }
        cvt.rn.f32.s32 $4, $0;
        cvt.rn.f32.s32 $5, $1;
        cvt.rn.f32.s32 $6, $2;
        cvt.rn.f32.s32 $7, $3;
        max.f32 $4, $4, $9;
        max.f32 $5, $5, $10;
        max.f32 $6, $6, $11;
        max.f32 $7, $7, $12;
        """,
        constraints=(
            "=r,=r,=r,=r,=r,=r,=r,=r,"
            "r,r,r,r,r"
        ),
        args=[a, b],
        dtype=(tl.int32, tl.float32),
        is_pure=True,
        pack=4,
    )
    tl.store(C + tl.arange(0, BLOCK), c)
    tl.store(D + tl.arange(0, BLOCK), d)
```

### map_elementwise

```python
@builtin
def map_elementwise(
    scalar_fn: Callable[..., Tuple[tensor, ...]],
    *args: tensor,
    pack=1,
    _semantic=None,
    _generator=None,
) -> tensor or tuple[tensor]
```

Maps a scalar function over tensors element-wise. Input tensors are implicitly broadcast to the same shape. This is useful for control flow over individual elements where `tl.where` would force both branches to execute.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scalar_fn` | `Callable` | required | The function to map. Must be marked with `@triton.jit`. |
| `*args` | `tensor` | required | Input tensors (broadcast to common shape) |
| `pack` | `int` | `1` | Number of elements per function call |

**Returns**: One tensor or a tuple of tensors, depending on the mapped function.

**Example**:

```python
@triton.jit
def selu_scalar(x, alpha):
    if x > 0:
        return x
    else:
        return alpha * (tl.exp(x) - 1)

@triton.jit
def selu(x, alpha):
    return tl.map_elementwise(selu_scalar, x, alpha)
```

---

## 19. Control Flow

### condition

```python
class condition(base_value)
```

Wrapper for while loop conditions that allows passing extra compiler attributes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arg1` | `tensor` | required | The loop condition |
| `disable_licm` | `bool` | `False` | If `True`, prevents the compiler from hoisting loop-invariant code outside the loop. Useful to avoid creating long live ranges. |

**Example**:

```python
@triton.jit
def kernel(...):
    while tl.condition(c, disable_licm=True):
        # Loop body
        pass
```

### static_range

```python
class static_range(base_value)
```

A compile-time range iterator that guides the compiler to aggressively unroll the loop. All parameters must be `constexpr`.

| Constructor Form | Description |
|------------------|-------------|
| `static_range(end)` | Iterates from 0 to `end-1` with step 1 |
| `static_range(start, end)` | Iterates from `start` to `end-1` with step 1 |
| `static_range(start, end, step)` | Iterates from `start` to `end-1` with given `step` |

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `constexpr` | Start value (default: 0) |
| `end` | `constexpr` | End value |
| `step` | `constexpr` | Step value (default: 1) |

**Example**:

```python
@triton.jit
def kernel(ptr, N: tl.constexpr, BLOCK_K: tl.constexpr):
    for k in tl.static_range(0, N, BLOCK_K):
        # This loop is unrolled at compile time
        pass
```

### range

```python
class range(base_value)
```

A range iterator with compiler hints for loop optimization.

| Constructor Form | Description |
|------------------|-------------|
| `range(end)` | Iterates from 0 to `end-1` |
| `range(start, end)` | Iterates from `start` to `end-1` |
| `range(start, end, step)` | With explicit step |

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int or `constexpr` | `0` | Start value |
| `end` | int or `constexpr` | required | End value |
| `step` | int or `constexpr` | `1` | Step value |
| `num_stages` | `int` or `None` | `None` | Pipeline the loop into this many stages. This pipelines most loads in the loop, unlike the kernel-level `num_stages` which only pipelines loads feeding `dot` operations. |
| `loop_unroll_factor` | `int` or `None` | `None` | How many times to unroll the loop. Less than 2 means no unrolling. |
| `disallow_acc_multi_buffer` | `bool` | `False` | Prevent the dot accumulator from being multi-buffered. |
| `flatten` | `bool` | `False` | Automatically flatten the loop nest to create a single loop for better pipelining. |
| `warp_specialize` | `bool` | `False` | Enable automatic warp specialization. Partitions memory, MMA, and vector ops into separate async partitions. Only supported on Blackwell GPUs for simple matmul loops. |
| `disable_licm` | `bool` | `False` | Prevent hoisting loop-invariant code. Useful to avoid long live ranges. |

**Example**:

```python
@triton.jit
def kernel(ptr, N, BLOCK_K: tl.constexpr):
    for k in tl.range(0, N, BLOCK_K, num_stages=3):
        # Loop with software pipelining (3 stages)
        pass

    for k in tl.range(0, N, BLOCK_K, loop_unroll_factor=4):
        # Loop unrolled by factor of 4
        pass

    for k in tl.range(0, N, BLOCK_K, warp_specialize=True):
        # Loop with warp specialization (Blackwell only)
        pass
```

---

## 20. tensor_descriptor_base Class

```python
class tensor_descriptor_base(base_value)
```

A tensor descriptor with unknown shape and strides. This is the base class for tensor descriptors used with TMA (Tensor Memory Accelerator) hardware.

**Constructor** (not called by user code):

```python
tensor_descriptor_base(handle, block_type: block_type)
```

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `handle` | `ir.value` | IR handle |
| `type` | `tensor_descriptor_base_type` | Tensor type (wrapping `block_type`) |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `block_type` | `block_type` | The block type of this descriptor |
| `block_shape` | `tuple[int]` | Shape of the block |
| `dtype` | `dtype` | Element type of the block |

**Methods**:

### `load(offsets)`

```python
@builtin
def load(self, offsets: Sequence[constexpr | tensor], _semantic=None) -> tensor
```

Loads a block from the descriptor starting at the given element offsets. Values outside of the tensor bounds will be filled with zeros.

**Note**: Offset must be a multiple of 16 bytes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `offsets` | `Sequence[constexpr or tensor]` | Element offsets per dimension |

**Returns**: A tensor containing the loaded block.

### `store(offsets, value)`

```python
@builtin
def store(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor
```

Stores a block to the descriptor starting at the given element offsets. Values outside of the tensor bounds will be ignored.

**Note**: Offset must be a multiple of 16 bytes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `offsets` | `Sequence[constexpr or tensor]` | Element offsets per dimension |
| `value` | `tensor` | The block to store |

### Atomic Operations on Descriptors

All atomic operations on tensor descriptors follow the same pattern: `(offsets, value)`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `atomic_add` | `(offsets: Sequence, value: tensor) -> tensor` | Atomic addition via descriptor |
| `atomic_min` | `(offsets: Sequence, value: tensor) -> tensor` | Atomic minimum via descriptor |
| `atomic_max` | `(offsets: Sequence, value: tensor) -> tensor` | Atomic maximum via descriptor |
| `atomic_and` | `(offsets: Sequence, value: tensor) -> tensor` | Atomic bitwise AND via descriptor |
| `atomic_or` | `(offsets: Sequence, value: tensor) -> tensor` | Atomic bitwise OR via descriptor |
| `atomic_xor` | `(offsets: Sequence, value: tensor) -> tensor` | Atomic bitwise XOR via descriptor |

### `gather(x_offsets, y_offset)`

```python
@builtin
def gather(self, *args, _semantic=None) -> tensor
```

Gathers multiple descriptors worth of data. Only supports 2D indexing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x_offsets` | `tensor` | X-dimension offsets |
| `y_offset` | `tensor` | Y-dimension offset |

### `scatter(value, x_offsets, y_offset)`

```python
@builtin
def scatter(self, value, *args, _semantic=None) -> tensor
```

Scatters data to multiple descriptor positions. Only supports 2D indexing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `tensor` | Data to scatter |
| `x_offsets` | `tensor` | X-dimension offsets |
| `y_offset` | `tensor` | Y-dimension offset |

---

## 21. tensor_descriptor Class

```python
class tensor_descriptor(tensor_descriptor_base)
```

A descriptor representing a tensor in global memory with known shape and strides.

**Constructor** (not called by user code):

```python
tensor_descriptor(handle, shape: List[tensor], strides: List[tensor], block_type: block_type)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `handle` | `ir.value` | IR handle |
| `shape` | `List[tensor]` | Global shape of the tensor |
| `strides` | `List[tensor]` | Strides of the tensor |
| `block_type` | `block_type` | Block type for load/store operations |

**Additional Attributes** (beyond base class):

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[tensor]` | Global tensor shape |
| `strides` | `tuple[tensor]` | Tensor strides |
| `type` | `tensor_descriptor_type` | Full type including shape and strides types |

**Methods**:

| Method | Signature | Description |
|--------|-----------|-------------|
| `_set_name(builder, name)` | `-> None` | Sets names for the descriptor, its shape, and strides |
| `_flatten_ir(handles)` | `-> None` | Flattens the descriptor handle, shape, and strides |

---

## 22. _block_ptr Class

```python
@_aggregate
class _block_ptr
    base: tensor
    shape: tuple
    strides: tuple
    offsets: tuple
    block_shape: tuple
    order: tuple
```

An aggregate type representing a block pointer. Created by `make_block_ptr`. Has the special attribute `__triton_block_ptr__ = True`.

**Constructor**:

```python
_block_ptr(base, shape, strides, offsets, block_shape, order, _semantic=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `tensor` | Base pointer (must be a scalar pointer, not a block or pointer-to-block) |
| `shape` | `list[int or tensor]` | Parent tensor shape |
| `strides` | `list[int or tensor]` | Parent tensor strides |
| `offsets` | `list[int or tensor]` | Initial offsets (must be integer tensors) |
| `block_shape` | `list[int]` | Block dimensions (positive integers, all must be power of 2) |
| `order` | `list[int]` | Memory layout order (permutation of `0..rank-1`) |

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `dtype` | `dtype` | Alias for `self.base.dtype` (pointer element type) |

**Validation**:
- `base` must be a scalar pointer type (not a block type, not a pointer to block type).
- `shape`, `strides`, `offsets`, and `order` must all have the same length as `block_shape`.
- `order` must be a permutation of `0..(rank-1)`.
- `block_shape` entries must be positive integers.

**Methods**:

### `_tile_shape()`

```python
def _tile_shape(self) -> list[int]
```

Returns the block shape as a list of plain integers.

### `advance(offsets)`

```python
def advance(self, offsets, _semantic=None) -> _block_ptr
```

Returns a new `_block_ptr` with offsets advanced by the given deltas.

| Parameter | Type | Description |
|-----------|------|-------------|
| `offsets` | `tuple[int or tensor]` | Offset deltas per dimension |

**Returns**: A new `_block_ptr` with updated offsets.

### `load(mask, other, boundary_check, padding_option, cache_modifier, eviction_policy, volatile)`

```python
def load(self, mask=None, other=None, boundary_check=(), padding_option="",
         cache_modifier="", eviction_policy="", volatile=False, _semantic=None) -> tensor
```

Loads data from the block pointer. `mask` and `other` must be `None` (use `boundary_check` and `padding_option` instead).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boundary_check` | `tuple[int]` | `()` | Dimensions to check for boundary violations |
| `padding_option` | `str` | `""` | Out-of-bounds padding: `""` (undefined), `"zero"`, `"nan"` |
| `cache_modifier` | `str` | `""` | PTX cache modifier |
| `eviction_policy` | `str` | `""` | PTX eviction policy |
| `volatile` | `bool` | `False` | PTX volatile flag |

### `store(value, mask, boundary_check, cache_modifier, eviction_policy)`

```python
def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None) -> tensor
```

Stores data to the block pointer. `mask` must be `None`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | `tensor` | required | Data to store |
| `boundary_check` | `tuple[int]` | `()` | Dimensions for boundary checking |
| `cache_modifier` | `str` | `""` | PTX cache modifier |
| `eviction_policy` | `str` | `""` | PTX eviction policy |

---

## 23. External Function Dispatch

### dispatch

```python
def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict,
             ret_type: dtype, is_pure: bool, _semantic) -> tensor
```

Dispatches a function call to an external library.

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | callable | The builder function to create the extern call |
| `lib_name` | `str` | Name of the library |
| `lib_path` | `str` | Path to the library |
| `args` | `list` | Arguments to the function |
| `arg_type_symbol_dict` | `dict` | Mapping from arg type tuples to (symbol, return_type) pairs |
| `ret_type` | `dtype` | Return type |
| `is_pure` | `bool` | Whether the function is pure (no side effects) |

### extern_elementwise

```python
@builtin
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict,
                       is_pure: bool, _semantic=None) -> tensor
```

Dispatches an elementwise function to an external library. Input tensors are implicitly broadcast to a common shape.

### extern (decorator)

```python
def extern(fn) -> builtin
```

A decorator for external functions. Equivalent to `builtin(fn)`.

### binary_op_type_legalization

```python
def binary_op_type_legalization(lhs, rhs, semantic) -> tuple[tensor, tensor]
```

Converts both operands to a single common type using `semantic.binary_op_type_checking_impl`.

---

## 24. Utility Functions and Decorators

### builtin (decorator)

```python
def builtin(fn: T) -> T
```

Marks a function as a Triton builtin. When called outside of a JIT context, raises a `ValueError` with the message "Did you forget to add @triton.jit?".

The decorator:
1. Wraps the function to check for the `_semantic` argument.
2. Sets `__triton_builtin__ = True` on the wrapper.
3. Preserves the function's signature via `inspect.signature`.

### _tensor_member_fn (decorator)

```python
def _tensor_member_fn(fn: T) -> T
```

Decorator that adds a free function as a member function on the `tensor` class. When called as `x.fn(...)`, the first argument is `self` (the tensor).

The decorator:
1. Creates a wrapper with `self` as the first parameter name.
2. Updates the function's docstring to document the member function usage.
3. Registers the function as an attribute on the `tensor` class.
4. If `fn` is a builtin, marks the wrapper as a builtin too.

### must_use_result (decorator)

```python
def must_use_result(x, s=True)
```

Marks that the result of a function must be used (assigned to a variable). If the result is discarded, an error or warning is generated.

Can be used as `@must_use_result` or `@must_use_result("message")`.

### check_bit_width

```python
def check_bit_width(value, shift_value) -> None
```

Warns if a shift value exceeds the bit width of the operand's type.

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `tensor` | The value being shifted |
| `shift_value` | `constexpr` | The shift amount |

### _unwrap_iterable

```python
def _unwrap_iterable(x) -> tuple or list
```

If `x` has one element and that element is iterable, returns `x[0]`. Otherwise returns `x`. Used for parsing variadic shape arguments.

### is_builtin

```python
def is_builtin(fn) -> bool
```

Returns `True` if the function is a registered Triton builtin (has `__triton_builtin__ = True`).

### _normalize_tuple

```python
def _normalize_tuple(t) -> tuple
```

Unwraps constexpr values and converts lists/builtins.tuples to Triton tuples.

### get_int_dtype

```python
def get_int_dtype(bitwidth: int, signed: bool) -> dtype
```

Returns the appropriate integer dtype for the given bit width and signedness. Supported bit widths: 1, 8, 16, 32, 64.

### is_negative_zero

```python
def is_negative_zero(x) -> bool
```

Returns `True` if `x` is negative zero (`-0.0`).

### PropagateNan

```python
PropagateNan = ir.PROPAGATE_NAN
```

An enum-like value used with `maximum`, `minimum`, and `clamp` to control NaN propagation behavior. Available values:
- `PropagateNan.NONE` -- Do not propagate NaN.
- Other values depend on the IR implementation.

### TRITON_BUILTIN

```python
TRITON_BUILTIN = "__triton_builtin__"
```

String constant used as the attribute name for marking Triton builtin functions.

### CONSTEXPR_0

```python
CONSTEXPR_0 = constexpr(0)
```

Pre-built constexpr zero constant for convenience.

---

## Appendix A: Aggregate Type System

Triton supports user-defined aggregate types via the `_aggregate` decorator. These behave like dataclasses but work within the Triton type system.

```python
@dataclass_transform(eq_default=False)
def _aggregate(cls) -> type
```

The decorator:
1. Inspects the class's type annotations to determine fields.
2. Generates an `__init__` method if not provided.
3. Creates an `aggregate_value` class that wraps the user's class.
4. Enforces that all fields are initialized in the constructor.
5. Only allows setting attributes defined in the class annotations.
6. Implements `_flatten_ir` and `_set_name` for IR integration.

**Example** (`_block_ptr` is defined this way):

```python
@_aggregate
class _block_ptr:
    base: tensor
    shape: tuple
    strides: tuple
    offsets: tuple
    block_shape: tuple
    order: tuple

    __triton_block_ptr__ = True

    def __init__(self, base, shape, strides, offsets, block_shape, order, _semantic=None):
        # Custom initialization logic
        ...
```

---

## Appendix B: _add_atomic_docstr (Internal Decorator)

```python
def _add_atomic_docstr(name: str, has_cmp: bool = False) -> Callable[[T], T]
```

Internal decorator that adds standardized documentation to atomic operation functions. Generates parameter documentation for `pointer`, `val` (and `cmp` for `atomic_cas`), `sem`, and `scope`.

---

## Appendix C: _add_reduction_docstr (Internal Decorator)

```python
def _add_reduction_docstr(name: str, return_indices_arg=None, tie_break_arg=None, dtype_arg=None) -> Callable[[T], T]
```

Internal decorator that adds standardized documentation to reduction functions. Generates parameter documentation for `input`, `axis`, `keep_dims`, and optionally `return_indices`, `tie_break_left`, and `dtype`.

---

## Appendix D: Tensor Descriptor Type Hierarchy

```
base_type
  +-- dtype
  |     +-- pointer_type
  |     +-- block_type
  |     +-- slice_type
  +-- constexpr_type
  +-- tuple_type
  +-- tensor_descriptor_base_type
  |     +-- tensor_descriptor_type
  +-- _aggregate_type

base_value
  +-- constexpr
  +-- tensor
  +-- tuple
  +-- tensor_descriptor_base
  |     +-- tensor_descriptor
  +-- _block_ptr (aggregate_value)
  +-- static_range
  +-- range
  +-- condition
```

---

## Appendix E: Quick Reference -- All Public Functions

| Function | Category | Brief Description |
|----------|----------|-------------------|
| `program_id(axis)` | Programming Model | Current program ID along axis |
| `num_programs(axis)` | Programming Model | Number of programs along axis |
| `arange(start, end)` | Tensor Creation | Contiguous values in `[start, end)` |
| `full(shape, value, dtype)` | Tensor Creation | Tensor filled with a scalar |
| `zeros(shape, dtype)` | Tensor Creation | Zero-filled tensor |
| `zeros_like(input)` | Tensor Creation | Zero-filled tensor matching input |
| `to_tensor(x)` | Tensor Creation | Convert scalar to tensor |
| `cast(input, dtype)` | Type Conversion | Cast tensor to new dtype |
| `load(pointer, ...)` | Memory | Load data from memory |
| `store(pointer, value, ...)` | Memory | Store data to memory |
| `make_block_ptr(base, ...)` | Memory | Create block pointer (deprecated) |
| `advance(base, offsets)` | Memory | Advance block pointer |
| `make_tensor_descriptor(...)` | Memory | Create TMA tensor descriptor |
| `load_tensor_descriptor(...)` | Memory | Load from tensor descriptor |
| `store_tensor_descriptor(...)` | Memory | Store to tensor descriptor |
| `dot(input, other, ...)` | Linear Algebra | Matrix multiplication |
| `dot_scaled(lhs, ...)` | Linear Algebra | Scaled matrix multiplication |
| `broadcast(input, other)` | Shape | Broadcast two tensors |
| `broadcast_to(input, shape)` | Shape | Broadcast tensor to shape |
| `reshape(input, shape)` | Shape | Change tensor shape |
| `expand_dims(input, axis)` | Shape | Insert new dimensions |
| `permute(input, dims)` | Shape | Permute dimensions |
| `trans(input, dims)` | Shape | Transpose (permute) dimensions |
| `view(input, shape)` | Shape | Reshape (deprecated) |
| `cat(input, other, ...)` | Shape | Concatenate tensors |
| `split(a)` | Shape | Split tensor along last dim |
| `join(a, b)` | Shape | Join tensors along new dim |
| `ravel(x)` | Shape | Flatten to 1D |
| `item(input)` | Shape | Single-element tensor to scalar |
| `gather(src, index, axis)` | Shape | Gather along axis |
| `reduce(input, axis, fn)` | Reductions | Custom reduction |
| `associative_scan(input, axis, fn)` | Reductions | Prefix scan |
| `histogram(input, num_bins)` | Reductions | Compute histogram |
| `atomic_cas(ptr, cmp, val)` | Atomic | Compare-and-swap |
| `atomic_xchg(ptr, val, ...)` | Atomic | Exchange |
| `atomic_add(ptr, val, ...)` | Atomic | Addition |
| `atomic_max(ptr, val, ...)` | Atomic | Maximum |
| `atomic_min(ptr, val, ...)` | Atomic | Minimum |
| `atomic_and(ptr, val, ...)` | Atomic | Bitwise AND |
| `atomic_or(ptr, val, ...)` | Atomic | Bitwise OR |
| `atomic_xor(ptr, val, ...)` | Atomic | Bitwise XOR |
| `where(cond, x, y)` | Conditioning | Conditional selection |
| `maximum(x, y)` | Math | Element-wise max |
| `minimum(x, y)` | Math | Element-wise min |
| `clamp(x, min, max)` | Math | Clamp to range |
| `add(x, y)` | Math | Addition |
| `sub(x, y)` | Math | Subtraction |
| `mul(x, y)` | Math | Multiplication |
| `multiple_of(input, values)` | Compiler Hint | Annotate multiples |
| `max_contiguous(input, values)` | Compiler Hint | Annotate contiguous |
| `max_constancy(input, values)` | Compiler Hint | Annotate constant groups |
| `assume(cond)` | Compiler Hint | Assume condition true |
| `static_print(*values)` | Debugging | Compile-time print |
| `static_assert(cond, msg)` | Debugging | Compile-time assert |
| `device_print(prefix, *args)` | Debugging | Runtime device print |
| `device_assert(cond, msg)` | Debugging | Runtime device assert |
| `debug_barrier()` | Debugging | Thread barrier |
| `inline_asm_elementwise(...)` | Assembly | Inline GPU assembly |
| `map_elementwise(fn, *args)` | Assembly | Map scalar function over tensor |
