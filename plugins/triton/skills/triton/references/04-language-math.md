# Chapter 4: Language Math Module (`triton.language.math`)

The math module provides mathematical operations optimized for GPU execution. All functions support both tensor and scalar inputs and can be called as member functions on `tensor` objects.

## Exponential Functions

### `tl.exp(x) -> tensor`
Natural exponential function e^x.

```python
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.exp(x)          # e^x
    tl.store(y_ptr + offs, y, mask=mask)
```

**Supported dtypes:** float32, float64
**Member function:** `x.exp()`

### `tl.exp2(x) -> tensor`
Base-2 exponential function 2^x.

**Supported dtypes:** float32, float64
**Member function:** `x.exp2()`

## Logarithmic Functions

### `tl.log(x) -> tensor`
Natural logarithm ln(x).

**Supported dtypes:** float32, float64
**Member function:** `x.log()`

### `tl.log2(x) -> tensor`
Base-2 logarithm log2(x).

**Supported dtypes:** float32, float64
**Member function:** `x.log2()`

## Trigonometric Functions

### `tl.cos(x) -> tensor`
Cosine function.

```python
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(ptr + offs, mask=offs < n)
    y = tl.cos(x)
    tl.store(ptr + offs, y, mask=offs < n)
```

**Supported dtypes:** float32, float64
**Member function:** `x.cos()`

### `tl.sin(x) -> tensor`
Sine function.

**Supported dtypes:** float32, float64
**Member function:** `x.sin()`

## Square Root Functions

### `tl.sqrt(x) -> tensor`
Fast square root. May use approximate hardware instructions.

**Supported dtypes:** float32, float64
**Member function:** `x.sqrt()`

### `tl.sqrt_rn(x) -> tensor`
IEEE-rounding square root. Uses `sqrt.rn.ftz.f32` instruction for precise results.

**Supported dtypes:** float32 only
**Member function:** `x.sqrt_rn()`

### `tl.rsqrt(x) -> tensor`
Inverse square root 1/sqrt(x).

**Supported dtypes:** float32, float64
**Member function:** `x.rsqrt()`

## Absolute Value

### `tl.abs(x) -> tensor`
Absolute value. For float8e4b15, uses bitmask operation.

```python
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(ptr + offs, mask=offs < n)
    y = tl.abs(x)    # Works for float, signed int, unsigned int
    tl.store(ptr + offs, y, mask=offs < n)
```

**Supported dtypes:** All floating-point and integer types
**Member function:** `x.abs()`

## Division Functions

### `tl.fdiv(x, y, ieee_rounding=False) -> tensor`
Fast floating-point division.

- `ieee_rounding=False`: Uses fast approximate division (default)
- `ieee_rounding=True`: Uses IEEE-compliant rounding

**Supported dtypes:** float32, float64
**Member function:** `x.fdiv(y)`

### `tl.div_rn(x, y) -> tensor`
Division with IEEE rounding (precise).

**Supported dtypes:** float32, float64
**Member function:** `x.div_rn(y)`

## Error Function

### `tl.erf(x) -> tensor`
Gauss error function.

**Supported dtypes:** float32, float64
**Member function:** `x.erf()`

## Rounding Functions

### `tl.floor(x) -> tensor`
Floor function (round towards negative infinity).

**Supported dtypes:** float32, float64
**Member function:** `x.floor()`

### `tl.ceil(x) -> tensor`
Ceiling function (round towards positive infinity).

**Supported dtypes:** float32, float64
**Member function:** `x.ceil()`

## Fused Multiply-Add

### `tl.fma(x, y, z) -> tensor`
Fused multiply-add: computes `x * y + z` with a single rounding.

```python
@triton.jit
def kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    result = tl.fma(a, b, c)  # a * b + c (single rounding)
    tl.store(out_ptr + offs, result, mask=mask)
```

**Supported dtypes:** float32, float64
**Member function:** `x.fma(y, z)`

## High Multiply

### `tl.umulhi(x, y) -> tensor`
Unsigned multiply high - returns the upper N bits of a 2N-bit product.

```python
# For int32 inputs, returns upper 32 bits of 64-bit product
hi = tl.umulhi(a, b)
```

**Supported dtypes:** int32, int64, uint32, uint64
**Member function:** `x.umulhi(y)`

## Type Checking

All math functions include strict type checking. Using unsupported types raises `ValueError`:

```python
# This will raise ValueError for integer inputs to exp:
x = tl.load(ptr, mask=mask)  # dtype is int32
y = tl.exp(x)                 # ERROR: exp only supports fp32/fp64

# Fix: cast first
y = tl.exp(tl.cast(x, tl.float32))
```

## External Math Functions (libdevice)

For additional math functions not in the core module, use `triton.language.extra.libdevice`:

```python
from triton.language.extra.libdevice import (
    acos, asin, atan, atan2,
    cospi, sinpi,
    tan, tanh,
    lgamma, tgamma,
    j0, j1, y0, y1,    # Bessel functions
    fmod, remainder,
    copysign, nextafter,
    float2half_rn, half2float,
    rhypot, hypot, norm3d, norm4d,
    clz, popc,          # Bit operations
    sad, umad24,        # Mixed operations
)
```
