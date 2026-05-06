# PyTorch Reference - Chapter 3: Tensor Operations

This chapter covers all tensor mathematical operations including pointwise operations, reductions, comparisons, BLAS/LAPACK operations, and spectral operations. Each function includes its signature, parameter descriptions, examples, and mathematical formulas where applicable.

---

## 3.1 Pointwise Operations

Pointwise operations apply a function independently to each element of the input tensor(s).

### 3.1.1 torch.abs / torch.absolute

Computes the absolute value of each element.

```python
torch.abs(input, *, out=None)
torch.absolute(input, *, out=None)  # Alias for abs
```

**Formula:** `out_i = |input_i|`

**Examples:**
```python
torch.abs(torch.tensor([-1, -2, 3]))        # tensor([1, 2, 3])
torch.abs(torch.tensor([-1.5 + 2.0j]))      # tensor([2.5]) - complex modulus
```

### 3.1.2 torch.acos / torch.arccos

Computes the inverse cosine (arccosine) of each element.

```python
torch.acos(input, *, out=None)
torch.arccos(input, *, out=None)  # Alias
```

**Formula:** `out_i = arccos(input_i)`, input must be in [-1, 1]. Output in [0, pi].

**Examples:**
```python
torch.acos(torch.tensor([0.0, 0.5, 1.0]))   # tensor([1.5708, 1.0472, 0.0000])
```

### 3.1.3 torch.acosh / torch.arccosh

Computes the inverse hyperbolic cosine of each element.

```python
torch.acosh(input, *, out=None)
torch.arccosh(input, *, out=None)  # Alias
```

**Formula:** `out_i = acosh(input_i)`, input must be >= 1.

**Examples:**
```python
torch.acosh(torch.tensor([1.0, 2.0, 3.0]))  # tensor([0.0000, 1.3170, 1.7627])
```

### 3.1.4 torch.add

Adds `other`, scaled by `alpha`, to `input`.

```python
torch.add(input, other, *, alpha=1, out=None)
```

**Parameters:**
- `input` (Tensor): The first input tensor.
- `other` (Tensor or Number): The second input tensor or scalar.
- `alpha` (Number): The scalar multiplier for `other`. Default: 1.

**Formula:** `out_i = input_i + alpha * other_i`

**Examples:**
```python
torch.add(torch.tensor([1, 2]), torch.tensor([3, 4]))    # tensor([4, 6])
torch.add(torch.tensor([1, 2]), 3)                        # tensor([4, 5])
torch.add(torch.tensor([1, 2]), torch.tensor([1, 1]), alpha=10)  # tensor([11, 12])
```

### 3.1.5 torch.addcdiv

Performs `input + value * (tensor1 / tensor2)` element-wise.

```python
torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
```

**Formula:** `out_i = input_i + value * (tensor1_i / tensor2_i)`

**Examples:**
```python
t = torch.randn(1, 3)
a = torch.randn(1, 3)
b = torch.randn(1, 3) + 0.1
torch.addcdiv(t, a, b, value=0.1)
```

### 3.1.6 torch.addcmul

Performs `input + value * (tensor1 * tensor2)` element-wise.

```python
torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
```

**Formula:** `out_i = input_i + value * (tensor1_i * tensor2_i)`

**Examples:**
```python
t = torch.randn(1, 3)
a = torch.randn(1, 3)
b = torch.randn(1, 3)
torch.addcmul(t, a, b, value=0.1)
```

### 3.1.7 torch.angle

Computes the angle (in radians) of complex tensors.

```python
torch.angle(input)
```

**Examples:**
```python
torch.angle(torch.tensor([1+1j, 0-1j, -1+0j]))  # tensor([0.7854, -1.5708, 3.1416])
```

### 3.1.8 torch.asin / torch.arcsin

```python
torch.asin(input, *, out=None)
```

**Formula:** `out_i = arcsin(input_i)`, input in [-1, 1], output in [-pi/2, pi/2].

**Examples:**
```python
torch.asin(torch.tensor([0.0, 0.5, 1.0]))     # tensor([0.0000, 0.5236, 1.5708])
```

### 3.1.9 torch.asinh / torch.arcsinh

```python
torch.asinh(input, *, out=None)
```

**Examples:**
```python
torch.asinh(torch.tensor([-1.0, 0.0, 1.0]))   # tensor([-0.8814, 0.0000, 0.8814])
```

### 3.1.10 torch.atan / torch.arctan

```python
torch.atan(input, *, out=None)
```

**Formula:** `out_i = arctan(input_i)`, output in [-pi/2, pi/2].

### 3.1.11 torch.atan2 / torch.arctan2

```python
torch.atan2(input, other, *, out=None)
```

**Formula:** `out_i = arctan2(input_i, other_i)`, output in [-pi, pi].

**Examples:**
```python
torch.atan2(torch.tensor([1.0]), torch.tensor([1.0]))  # tensor([0.7854]) = pi/4
```

### 3.1.12 torch.atanh / torch.arctanh

```python
torch.atanh(input, *, out=None)
```

**Formula:** `out_i = atanh(input_i)`, input must be in (-1, 1).

### 3.1.13 Bitwise Operations

```python
# Bitwise AND
torch.bitwise_and(input, other, *, out=None)

# Bitwise OR
torch.bitwise_or(input, other, *, out=None)

# Bitwise XOR
torch.bitwise_xor(input, other, *, out=None)

# Bitwise NOT
torch.bitwise_not(input, *, out=None)

# Left shift: input << shift
torch.bitwise_left_shift(input, shift)

# Right shift: input >> shift
torch.bitwise_right_shift(input, shift)
```

**Examples:**
```python
a = torch.tensor([-1, 0, 1], dtype=torch.int8)
b = torch.tensor([1, 0, 1], dtype=torch.int8)
torch.bitwise_and(a, b)        # tensor([1, 0, 1], dtype=torch.int8)
torch.bitwise_or(a, b)         # tensor([-1, 0, 1], dtype=torch.int8)
torch.bitwise_xor(a, b)        # tensor([-2, 0, 0], dtype=torch.int8)
torch.bitwise_not(a)           # tensor([0, -1, -2], dtype=torch.int8)
```

### 3.1.14 torch.ceil

```python
torch.ceil(input, *, out=None)
```

**Formula:** `out_i = ceil(input_i)` (smallest integer >= input)

**Examples:**
```python
torch.ceil(torch.tensor([-0.5, 0.5, 1.7]))    # tensor([0., 1., 2.])
```

### 3.1.15 torch.clamp / torch.clip

Clamps all elements into the range [min, max].

```python
torch.clamp(input, min=None, max=None, *, out=None)
torch.clip(input, min=None, max=None, *, out=None)  # Alias
```

**Parameters:**
- `min` (Number or Tensor, optional): Lower bound.
- `max` (Number or Tensor, optional): Upper bound.

**Formula:**
- If `min` and `max` given: `out_i = min(max(input_i, min_i), max_i)`
- If only `min`: `out_i = max(input_i, min_i)`
- If only `max`: `out_i = min(input_i, max_i)`

**Examples:**
```python
t = torch.randn(5)
torch.clamp(t, min=-0.5, max=0.5)  # Clip to [-0.5, 0.5]
torch.clamp(t, min=0.0)            # ReLU-like (no negative values)
torch.clamp(t, max=1.0)            # Cap at 1.0

# Tensor min/max (per-element)
torch.clamp(t, min=torch.zeros(5), max=torch.ones(5))
```

### 3.1.16 torch.conj / torch.conj_physical

```python
# Returns a view with conjugated elements (lazy, for complex tensors)
torch.conj(input)

# Returns a new tensor with conjugated elements (physical copy)
torch.conj_physical(input)
```

**Examples:**
```python
t = torch.tensor([1+2j, 3-4j])
torch.conj(t)                  # tensor([1-2j, 3+4j])
```

### 3.1.17 torch.cos / torch.cosh

```python
torch.cos(input, *, out=None)    # Cosine
torch.cosh(input, *, out=None)   # Hyperbolic cosine
```

**Examples:**
```python
torch.cos(torch.tensor([0, 3.14159]))  # tensor([1., -1.])
torch.cosh(torch.tensor([0, 1]))       # tensor([1., 1.5431])
```

### 3.1.18 torch.deg2rad / torch.rad2deg

```python
torch.deg2rad(input)    # Degrees to radians: x * pi / 180
torch.rad2deg(input)    # Radians to degrees: x * 180 / pi
```

### 3.1.19 torch.digamma

```python
torch.digamma(input, *, out=None)
```

Computes the logarithmic derivative of the gamma function: `psi(x) = d/dx ln(Gamma(x))`.

**Examples:**
```python
torch.digamma(torch.tensor([1.0, 2.0, 3.0]))  # tensor([-0.5772, 0.4228, 0.9228])
```

### 3.1.20 torch.div / torch.divide / torch.true_divide / torch.floor_divide

```python
torch.div(input, other, *, rounding_mode=None, out=None)
torch.divide(input, other, *, rounding_mode=None, out=None)  # Alias
torch.true_divide(input, other)   # Always floating-point division
torch.floor_divide(input, other)  # Floor division (truncates toward -inf)
```

**Parameters:**
- `rounding_mode` (str, optional):
  - `None`: Default behavior (truncates for integers, true division for floats)
  - `"trunc"`: Truncates toward zero
  - `"floor"`: Floor division (toward negative infinity)

**Examples:**
```python
a = torch.tensor([5, 7, 9])
b = torch.tensor([2, 2, 2])

torch.div(a, b)                           # tensor([2, 3, 4]) - truncation for integers
torch.div(a, b, rounding_mode='floor')    # tensor([2, 3, 4]) - floor
torch.div(a, b, rounding_mode='trunc')    # tensor([2, 3, 4]) - truncation
torch.true_divide(a, b)                   # tensor([2.5, 3.5, 4.5])
torch.floor_divide(a, b)                  # tensor([2, 3, 4])

# Float division
x = torch.tensor([5.0, 7.0, 9.0])
torch.div(x, 2)                           # tensor([2.5, 3.5, 4.5])
```

### 3.1.21 torch.erf / torch.erfc / torch.erfinv

```python
torch.erf(input, *, out=None)      # Error function
torch.erfc(input, *, out=None)     # Complementary error function: 1 - erf(x)
torch.erfinv(input, *, out=None)   # Inverse error function
```

**Examples:**
```python
torch.erf(torch.tensor([0.0, 0.5, 1.0]))       # tensor([0.0000, 0.5205, 0.8427])
torch.erfc(torch.tensor([0.0, 0.5, 1.0]))      # tensor([1.0000, 0.4795, 0.1573])
torch.erfinv(torch.tensor([0.0, 0.5, 0.8427])) # tensor([0.0000, 0.4769, 1.0000])
```

### 3.1.22 torch.exp / torch.exp2 / torch.expm1

```python
torch.exp(input, *, out=None)      # e^x
torch.exp2(input, *, out=None)     # 2^x
torch.expm1(input, *, out=None)    # e^x - 1 (more accurate for small x)
```

**Examples:**
```python
torch.exp(torch.tensor([0.0, 1.0, 2.0]))  # tensor([1.0000, 2.7183, 7.3891])
torch.exp2(torch.tensor([0.0, 1.0, 2.0])) # tensor([1., 2., 4.])
torch.expm1(torch.tensor([0.0, 0.01]))    # tensor([0.0000, 0.0101])
```

### 3.1.23 torch.floor

```python
torch.floor(input, *, out=None)
```

**Formula:** `out_i = floor(input_i)` (largest integer <= input)

### 3.1.24 torch.fmod / torch.remainder

```python
# fmod: C-style modulo (sign of dividend)
torch.fmod(input, other, *, out=None)

# remainder: Python-style modulo (sign of divisor)
torch.remainder(input, other, *, out=None)
```

**Examples:**
```python
torch.fmod(torch.tensor([-3.0, -1.0, 1.0, 3.0]), 2.0)  # tensor([-1., -1., 1., 1.])
torch.remainder(torch.tensor([-3.0, -1.0, 1.0, 3.0]), 2.0)  # tensor([1., 1., 1., 1.])
```

### 3.1.25 torch.frac

```python
torch.frac(input, *, out=None)
```

Computes the fractional portion of each element: `out_i = input_i - floor(|input_i|) * sgn(input_i)`.

**Examples:**
```python
torch.frac(torch.tensor([3.14, -2.7]))   # tensor([0.1400, -0.7000])
```

### 3.1.26 torch.gradient

Estimates the gradient of a function using second-order accurate central differences.

```python
torch.gradient(input, *, dim=None, spacing=1, edge_order=1)
```

**Examples:**
```python
t = torch.tensor([1.0, 2.0, 4.0, 7.0, 11.0])
torch.gradient(t)              # tensor([1.0000, 1.5000, 2.5000, 3.5000, 4.0000])

# With spacing
torch.gradient(t, spacing=2.0)  # Divide by spacing

# Multi-dimensional
t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
dy, dx = torch.gradient(t)    # Gradients along dims 0 and 1
```

### 3.1.27 torch.hypot

```python
torch.hypot(input, other, *, out=None)
```

**Formula:** `out_i = sqrt(input_i^2 + other_i^2)`

### 3.1.28 Special Functions (igamma, igammac, i0, polygamma, ldexp, lgamma, digamma)

```python
torch.i0(input, *, out=None)              # Modified Bessel function of order 0
torch.igamma(input, other, *, out=None)   # Regularized lower incomplete gamma
torch.igammac(input, other, *, out=None)  # Regularized upper incomplete gamma
torch.lgamma(input, *, out=None)          # Log of absolute value of Gamma function
torch.polygamma(n, input, *, out=None)    # n-th derivative of digamma
torch.ldexp(input, other, *, out=None)    # input * 2^other
```

### 3.1.29 torch.lerp

Linear interpolation between two tensors.

```python
torch.lerp(input, end, weight, *, out=None)
```

**Formula:** `out_i = input_i + weight * (end_i - input_i)`

**Examples:**
```python
start = torch.tensor([0.0, 0.0, 0.0])
end = torch.tensor([10.0, 10.0, 10.0])
torch.lerp(start, end, 0.5)    # tensor([5., 5., 5.])
torch.lerp(start, end, torch.tensor([0.0, 0.5, 1.0]))  # tensor([0., 5., 10.])
```

### 3.1.30 torch.log / torch.log2 / torch.log10 / torch.log1p

```python
torch.log(input, *, out=None)      # Natural log: ln(x)
torch.log2(input, *, out=None)     # Log base 2
torch.log10(input, *, out=None)    # Log base 10
torch.log1p(input, *, out=None)    # ln(1 + x), more accurate for small x
```

**Examples:**
```python
torch.log(torch.tensor([1.0, 2.7183, 7.3891]))   # tensor([0.0000, 1.0000, 2.0000])
torch.log2(torch.tensor([1.0, 2.0, 4.0, 8.0]))   # tensor([0., 1., 2., 3.])
torch.log10(torch.tensor([1.0, 10.0, 100.0]))    # tensor([0., 1., 2.])
torch.log1p(torch.tensor([0.0, 0.01, 0.1]))      # tensor([0.0000, 0.0099, 0.0953])
```

### 3.1.31 torch.logaddexp / torch.logaddexp2

```python
torch.logaddexp(input, other, *, out=None)    # ln(exp(input) + exp(other))
torch.logaddexp2(input, other, *, out=None)   # log2(2^input + 2^other)
```

**Examples:**
```python
torch.logaddexp(torch.tensor([1.0]), torch.tensor([2.0]))  # tensor([2.3133])
```

### 3.1.32 Logical Operations

```python
torch.logical_and(input, other, *, out=None)   # Element-wise logical AND
torch.logical_or(input, other, *, out=None)    # Element-wise logical OR
torch.logical_not(input, *, out=None)           # Element-wise logical NOT
torch.logical_xor(input, other, *, out=None)   # Element-wise logical XOR
```

**Examples:**
```python
a = torch.tensor([True, False, True])
b = torch.tensor([True, True, False])
torch.logical_and(a, b)         # tensor([True, False, False])
torch.logical_or(a, b)          # tensor([True, True, True])
torch.logical_not(a)            # tensor([False, True, False])
torch.logical_xor(a, b)         # tensor([False, True, True])

# Works with non-boolean types (0 is False, nonzero is True)
torch.logical_and(torch.tensor([0, 1, 2]), torch.tensor([1, 0, 3]))
# tensor([False, False, True])
```

### 3.1.33 torch.logit

```python
torch.logit(input, eps=None, *, out=None)
```

**Formula:** `out_i = ln(input_i / (1 - input_i))`, with clamping if `eps` is provided.

**Examples:**
```python
torch.logit(torch.tensor([0.25, 0.5, 0.75]))  # tensor([-1.0986, 0.0000, 1.0986])
torch.logit(torch.tensor([0.0, 0.5, 1.0]), eps=1e-6)  # Handles boundary with clamping
```

### 3.1.34 torch.mul / torch.multiply

Element-wise multiplication.

```python
torch.mul(input, other, *, out=None)
torch.multiply(input, other, *, out=None)  # Alias
```

**Formula:** `out_i = input_i * other_i`

**Examples:**
```python
torch.mul(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))  # tensor([4, 10, 18])
torch.mul(torch.tensor([1, 2, 3]), 3)                          # tensor([3, 6, 9])
```

### 3.1.35 torch.neg / torch.negative

```python
torch.neg(input, *, out=None)
torch.negative(input, *, out=None)  # Alias
```

**Formula:** `out_i = -input_i`

### 3.1.36 torch.nan_to_num

Replaces NaN, infinity, and negative infinity with specified values.

```python
torch.nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None)
```

**Examples:**
```python
t = torch.tensor([float('nan'), float('inf'), float('-inf'), 1.0])
torch.nan_to_num(t)              # tensor([0.0, 3.4e+38, -3.4e+38, 1.0])
torch.nan_to_num(t, nan=-1, posinf=999, neginf=-999)
# tensor([-1., 999., -999., 1.])
```

### 3.1.37 torch.nextafter

```python
torch.nextafter(input, other, *, out=None)
```

Returns the next floating-point value after `input` in the direction of `other`.

### 3.1.38 torch.pow

```python
torch.pow(input, exponent, *, out=None)
```

**Formula:** `out_i = input_i ^ exponent`

**Examples:**
```python
torch.pow(torch.tensor([1, 2, 3]), 2)           # tensor([1, 4, 9])
torch.pow(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 0.5]))  # tensor([1.0, 1.4142])
torch.pow(2, torch.tensor([1, 2, 3]))           # tensor([2, 4, 8])
```

### 3.1.39 torch.real / torch.imag

```python
torch.real(input)     # Real part of complex tensor
torch.imag(input)     # Imaginary part of complex tensor
```

### 3.1.40 torch.reciprocal

```python
torch.reciprocal(input, *, out=None)
```

**Formula:** `out_i = 1 / input_i`

### 3.1.41 torch.round

```python
torch.round(input, *, decimals=0, out=None)
```

**Examples:**
```python
torch.round(torch.tensor([0.5, 1.5, 2.3, 2.7]))  # tensor([0., 2., 2., 3.])
torch.round(torch.tensor([1.234, 5.678]), decimals=2)  # tensor([1.23, 5.68])
```

### 3.1.42 torch.rsqrt

```python
torch.rsqrt(input, *, out=None)
```

**Formula:** `out_i = 1 / sqrt(input_i)`

### 3.1.43 torch.sigmoid

```python
torch.sigmoid(input, *, out=None)
```

**Formula:** `out_i = 1 / (1 + exp(-input_i))`

**Examples:**
```python
torch.sigmoid(torch.tensor([0.0, 1.0, -1.0]))  # tensor([0.5000, 0.7311, 0.2689])
```

### 3.1.44 torch.sign / torch.sgn / torch.signbit

```python
torch.sign(input, *, out=None)    # Sign function: -1, 0, or 1
torch.sgn(input, *, out=None)     # Generalized sign (supports complex)
torch.signbit(input, *, out=None) # True if negative (sign bit is set)
```

**Examples:**
```python
torch.sign(torch.tensor([-2.5, 0.0, 3.0]))     # tensor([-1., 0., 1.])
torch.sgn(torch.tensor([-1+2j, 0+0j, 1-1j]))   # Complex sign
torch.signbit(torch.tensor([-1.0, 0.0, 1.0]))   # tensor([True, False, False])
```

### 3.1.45 torch.sin / torch.sinc / torch.sinh

```python
torch.sin(input, *, out=None)     # Sine
torch.sinc(input, *, out=None)    # Normalized sinc: sin(pi*x) / (pi*x)
torch.sinh(input, *, out=None)    # Hyperbolic sine
```

### 3.1.46 torch.sqrt / torch.square

```python
torch.sqrt(input, *, out=None)     # Square root
torch.square(input, *, out=None)   # Element-wise square: x^2
```

### 3.1.47 torch.sub / torch.subtract

```python
torch.sub(input, other, *, alpha=1, out=None)
torch.subtract(input, other, *, alpha=1, out=None)  # Alias
```

**Formula:** `out_i = input_i - alpha * other_i`

### 3.1.48 torch.tan / torch.tanh

```python
torch.tan(input, *, out=None)      # Tangent
torch.tanh(input, *, out=None)     # Hyperbolic tangent
```

**Examples:**
```python
torch.tan(torch.tensor([0.0, 0.7854]))  # tensor([0.0000, 1.0000]) (approx)
torch.tanh(torch.tensor([-1.0, 0.0, 1.0]))  # tensor([-0.7616, 0.0000, 0.7616])
```

### 3.1.49 torch.trunc

```python
torch.trunc(input, *, out=None)
```

Returns the truncated integer value (toward zero).

**Examples:**
```python
torch.trunc(torch.tensor([-1.7, -0.5, 0.5, 1.7]))  # tensor([-1., -0., 0., 1.])
```

---

## 3.2 Reduction Operations

Reduction operations collapse one or more dimensions of a tensor.

### 3.2.1 torch.argmax / torch.argmin

```python
torch.argmax(input, dim=None, keepdim=False)
torch.argmin(input, dim=None, keepdim=False)
```

**Parameters:**
- `dim` (int, optional): The dimension to reduce. If `None`, returns the index of the extremum in the flattened tensor.
- `keepdim` (bool): Whether the output tensor retains the reduced dimension with size 1.

**Examples:**
```python
t = torch.randn(3, 4)
torch.argmax(t)                # Scalar: index of max in flattened tensor
torch.argmax(t, dim=0)         # shape: (4,) - argmax along rows (per column)
torch.argmax(t, dim=1)         # shape: (3,) - argmax along columns (per row)
torch.argmax(t, dim=1, keepdim=True)  # shape: (3, 1)

torch.argmin(t, dim=1)         # shape: (3,) - argmin per row
```

### 3.2.2 torch.amax / torch.amin

```python
torch.amax(input, dim=None, keepdim=False, *, out=None)
torch.amin(input, dim=None, keepdim=False, *, out=None)
```

Returns the maximum/minimum values along a dimension (unlike `max`/`min` which also return indices).

**Examples:**
```python
t = torch.randn(3, 4)
torch.amax(t, dim=0)          # shape: (4,) - max per column
torch.amax(t, dim=1)          # shape: (3,) - max per row
torch.amin(t, dim=0)          # shape: (4,) - min per column
```

### 3.2.3 torch.aminmax

```python
torch.aminmax(input, *, dim=None, keepdim=False, out=None)
```

Returns a named tuple `(min, max)`.

**Examples:**
```python
t = torch.randn(3, 4)
result = torch.aminmax(t, dim=1)
result.min                     # shape: (3,)
result.max                     # shape: (3,)
```

### 3.2.4 torch.all / torch.any

```python
torch.all(input, dim=None, keepdim=False, *, out=None)
torch.any(input, dim=None, keepdim=False, *, out=None)
```

**Examples:**
```python
t = torch.tensor([[True, True], [True, False]])
torch.all(t)                   # tensor(False)
torch.all(t, dim=0)            # tensor([True, False])
torch.all(t, dim=1)            # tensor([True, False])

torch.any(t)                   # tensor(True)
torch.any(t, dim=0)            # tensor([True, True])
```

### 3.2.5 torch.count_nonzero

```python
torch.count_nonzero(input, dim=None)
```

**Examples:**
```python
t = torch.tensor([[0, 1, 0], [1, 1, 0]])
torch.count_nonzero(t)          # tensor(3)
torch.count_nonzero(t, dim=0)   # tensor([1, 2, 0])
torch.count_nonzero(t, dim=1)   # tensor([1, 2])
```

### 3.2.6 torch.dist

```python
torch.dist(input, other, p=2)
```

Computes the p-norm of `(input - other)`.

**Examples:**
```python
a = torch.tensor([1.0, 2.0])
b = torch.tensor([4.0, 6.0])
torch.dist(a, b, p=2)          # tensor(5.) - L2 distance
torch.dist(a, b, p=1)          # tensor(7.) - L1 distance
```

### 3.2.7 torch.logsumexp

```python
torch.logsumexp(input, dim, keepdim=False, *, out=None)
```

**Formula:** `out_i = ln(sum_j(exp(input_{i,j})))`, computed in a numerically stable way.

**Examples:**
```python
t = torch.randn(3, 4)
torch.logsumexp(t, dim=1)      # shape: (3,)
```

### 3.2.8 torch.mean

```python
torch.mean(input, dim=None, keepdim=False, *, dtype=None, out=None)
```

**Examples:**
```python
t = torch.randn(3, 4)
torch.mean(t)                   # scalar - mean of all elements
torch.mean(t, dim=0)            # shape: (4,) - mean per column
torch.mean(t, dim=1)            # shape: (3,) - mean per row
torch.mean(t, dim=1, keepdim=True)  # shape: (3, 1)
torch.mean(t, dtype=torch.float64)  # Compute in float64 precision
```

### 3.2.9 torch.median / torch.mode

```python
torch.median(input, dim=None, keepdim=False, *, out=None)
torch.mode(input, dim=-1, keepdim=False, *, out=None)
```

**Examples:**
```python
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch.median(t, dim=1)
# values: tensor([2, 5]), indices: tensor([1, 1])

torch.mode(t, dim=1)
# values: tensor([1, 4]), indices: tensor([0, 0])
```

### 3.2.10 torch.norm

```python
torch.norm(input, p='fro', dim=None, keepdim=False, *, dtype=None, out=None)
```

**Parameters:**
- `p` (int, float, inf, -inf, 'fro', 'nuc'): The norm order. Default: 'fro' (Frobenius).

**Examples:**
```python
t = torch.randn(3, 4)
torch.norm(t)                  # Frobenius norm (L2 for vectors)
torch.norm(t, p=1)             # L1 norm
torch.norm(t, p=2)             # L2 norm (same as Frobenius for matrices)
torch.norm(t, p=float('inf'))  # Max absolute value
torch.norm(t, dim=1)           # L2 norm per row
torch.norm(t, dim=1, p=1)      # L1 norm per row
```

### 3.2.11 torch.nansum / torch.nanmean

```python
torch.nansum(input, dim=None, keepdim=False, *, dtype=None, out=None)
torch.nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None)
```

Like `sum`/`mean` but treats NaN values as zero (nansum) or ignores them (nanmean).

### 3.2.12 torch.prod

```python
torch.prod(input, dim=None, keepdim=False, *, dtype=None, out=None)
```

**Examples:**
```python
torch.prod(torch.tensor([1, 2, 3, 4]))  # tensor(24)
```

### 3.2.13 torch.quantile / torch.nanquantile

```python
torch.quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None)
```

**Parameters:**
- `q` (float or Tensor): Quantile(s) to compute, in range [0, 1].
- `interpolation` (str): 'linear', 'lower', 'higher', 'midpoint', 'nearest'.

**Examples:**
```python
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
torch.quantile(t, 0.5)         # tensor(3.) - median
torch.quantile(t, torch.tensor([0.25, 0.5, 0.75]))  # Q1, Q2, Q3
```

### 3.2.14 torch.std / torch.var / torch.std_mean / torch.var_mean

```python
# Standard deviation
torch.std(input, dim=None, unbiased=True, keepdim=False, *, correction=1, out=None)

# Variance
torch.var(input, dim=None, unbiased=True, keepdim=False, *, correction=1, out=None)

# Both std and mean
torch.std_mean(input, dim, unbiased=True, keepdim=False, *, correction=1)

# Both var and mean
torch.var_mean(input, dim, unbiased=True, keepdim=False, *, correction=1)
```

**Parameters:**
- `correction` (int): The Bessel correction. Default: 1. Use 0 for population std/var.
- `unbiased` (bool): Deprecated; use `correction` instead.

**Examples:**
```python
t = torch.randn(4, 5)

torch.std(t)                   # Standard deviation of all elements
torch.std(t, dim=0)            # Std per column, shape: (5,)
torch.std(t, dim=1)            # Std per row, shape: (4,)
torch.var(t, dim=1)            # Variance per row

# std and mean together
std, mean = torch.std_mean(t, dim=1)  # Both shape: (4,)

# Population variance (no Bessel correction)
torch.var(t, dim=1, correction=0)
```

### 3.2.15 torch.sum

```python
torch.sum(input, dim=None, keepdim=False, *, dtype=None, out=None)
```

**Examples:**
```python
t = torch.randn(3, 4)
torch.sum(t)                   # Sum of all elements
torch.sum(t, dim=0)            # Sum per column, shape: (4,)
torch.sum(t, dim=1)            # Sum per row, shape: (3,)
torch.sum(t, dim=1, keepdim=True)  # shape: (3, 1)
torch.sum(t, dtype=torch.float64)  # Accumulate in float64
```

### 3.2.16 torch.unique / torch.unique_consecutive

```python
torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)
torch.unique_consecutive(input, return_inverse=False, return_counts=False, dim=None)
```

**Examples:**
```python
t = torch.tensor([3, 1, 2, 1, 3, 2, 1])
torch.unique(t)                # tensor([1, 2, 3])
torch.unique(t, return_inverse=True, return_counts=True)
# (tensor([1, 2, 3]), tensor([1, 0, 2, 0, 1, 2, 0]), tensor([3, 2, 2]))

# Unique consecutive (only removes adjacent duplicates)
t = torch.tensor([1, 1, 2, 2, 1, 1, 3, 3])
torch.unique_consecutive(t)    # tensor([1, 2, 1, 3])
```

### 3.2.17 torch.value_counts

```python
torch.value_counts(input, sorted=True, out=None)
```

**Examples:**
```python
t = torch.tensor([1, 1, 2, 3, 3, 3])
torch.value_counts(t)          # tensor([2, 1, 3]) with values tensor([1, 2, 3])
```

---

## 3.3 Comparison Operations

### 3.3.1 Element-wise Comparisons

```python
torch.eq(input, other, *, out=None)    # Equal: ==
torch.ne(input, other, *, out=None)    # Not equal: !=
torch.gt(input, other, *, out=None)    # Greater than: >
torch.ge(input, other, *, out=None)    # Greater or equal: >=
torch.lt(input, other, *, out=None)    # Less than: <
torch.le(input, other, *, out=None)    # Less or equal: <=
```

All return boolean tensors. Support broadcasting between tensors and scalars.

**Examples:**
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 3, 2])

torch.eq(a, b)                  # tensor([True, False, False])
torch.ne(a, b)                  # tensor([False, True, True])
torch.gt(a, b)                  # tensor([False, False, True])
torch.ge(a, 2)                  # tensor([False, True, True])
torch.lt(a, 3)                  # tensor([True, True, False])
torch.le(a, 2)                  # tensor([True, True, False])
```

### 3.3.2 torch.isclose

```python
torch.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)
```

**Formula:** `|input - other| <= atol + rtol * |other|`

**Examples:**
```python
a = torch.tensor([1.0, 2.0, float('nan')])
b = torch.tensor([1.0, 2.01, float('nan')])
torch.isclose(a, b)             # tensor([True, False, False])
torch.isclose(a, b, atol=0.1)  # tensor([True, True, False])
torch.isclose(a, b, equal_nan=True)  # tensor([True, False, True])
```

### 3.3.3 torch.equal

```python
torch.equal(input, other)
```

Returns `True` if two tensors have the same size and elements, `False` otherwise.

**Examples:**
```python
torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))    # True
torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2.0]))  # False (different dtype)
```

### 3.3.4 torch.max / torch.min

```python
# Without dim: returns the single max/min value
torch.max(input)
torch.min(input)

# With dim: returns (values, indices)
torch.max(input, dim, keepdim=False)
torch.min(input, dim, keepdim=False)

# Element-wise max/min of two tensors
torch.max(input, other)
torch.min(input, other)
```

**Examples:**
```python
t = torch.randn(3, 4)

# Global max/min
torch.max(t)                    # scalar tensor
torch.min(t)                    # scalar tensor

# Per-dimension max/min (returns named tuple)
result = torch.max(t, dim=1)
result.values                   # shape: (3,) - max values per row
result.indices                  # shape: (3,) - argmax per row

# Element-wise
a = torch.tensor([1, 4, 2])
b = torch.tensor([3, 2, 5])
torch.max(a, b)                 # tensor([3, 4, 5])
torch.min(a, b)                 # tensor([1, 2, 2])
```

### 3.3.5 torch.topk

```python
torch.topk(input, k, dim=None, largest=True, sorted=True)
```

Returns the `k` largest (or smallest) elements and their indices.

**Parameters:**
- `k` (int): Number of top elements.
- `dim` (int, optional): Dimension to sort along. Default: last dim.
- `largest` (bool): If `True`, return largest. If `False`, return smallest.
- `sorted` (bool): Whether to sort the result.

**Examples:**
```python
t = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
values, indices = torch.topk(t, 3)
# values: tensor([9, 6, 5])
# indices: tensor([5, 7, 4])

values, indices = torch.topk(t, 3, largest=False)
# values: tensor([1, 1, 2])
# indices: tensor([1, 3, 6])
```

### 3.3.6 torch.sort / torch.argsort

```python
torch.sort(input, dim=-1, descending=False, stable=False)
torch.argsort(input, dim=-1, descending=False, stable=False)
```

**Parameters:**
- `descending` (bool): Sort in descending order.
- `stable` (bool): Preserve the order of equal elements.

**Examples:**
```python
t = torch.tensor([3, 1, 4, 1, 5])

result = torch.sort(t)
result.values                  # tensor([1, 1, 3, 4, 5])
result.indices                 # tensor([1, 3, 0, 2, 4])

torch.sort(t, descending=True).values  # tensor([5, 4, 3, 1, 1])
torch.argsort(t)               # tensor([1, 3, 0, 2, 4])
```

### 3.3.7 torch.kthvalue

```python
torch.kthvalue(input, k, dim=None, keepdim=False)
```

Returns the k-th smallest element and its index.

**Examples:**
```python
t = torch.tensor([3, 1, 4, 1, 5])
values, indices = torch.kthvalue(t, 3)  # 3rd smallest
# values: tensor(3), indices: tensor(0)
```

### 3.3.8 Type Check Functions

```python
torch.isfinite(input)    # True if not NaN/Inf
torch.isinf(input)       # True if +/-Inf
torch.isnan(input)       # True if NaN
torch.isreal(input)      # True if real-valued
```

**Examples:**
```python
t = torch.tensor([1.0, float('inf'), float('-inf'), float('nan')])
torch.isfinite(t)              # tensor([True, False, False, False])
torch.isinf(t)                 # tensor([False, True, True, False])
torch.isnan(t)                 # tensor([False, False, False, True])
```

---

## 3.4 BLAS and LAPACK Operations

### 3.4.1 torch.addbmm

```python
torch.addbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)
```

**Formula:** `out = beta * input + alpha * sum_i(batch1_i @ batch2_i)`

### 3.4.2 torch.addmm

```python
torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None)
```

**Formula:** `out = beta * input + alpha * mat1 @ mat2`

**Examples:**
```python
M = torch.randn(2, 3)
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
torch.addmm(M, mat1, mat2, beta=0.5, alpha=0.5)
```

### 3.4.3 torch.addmv

```python
torch.addmv(input, mat, vec, *, beta=1, alpha=1, out=None)
```

**Formula:** `out = beta * input + alpha * mat @ vec`

### 3.4.4 torch.addr

```python
torch.addr(input, vec1, vec2, *, beta=1, alpha=1, out=None)
```

**Formula:** `out = beta * input + alpha * vec1 outer vec2`

### 3.4.5 torch.baddbmm

```python
torch.baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)
```

**Formula:** `out_i = beta * input_i + alpha * batch1_i @ batch2_i`

### 3.4.6 torch.bmm

Batch matrix multiplication.

```python
torch.bmm(input, mat2, *, out=None)
```

**Parameters:**
- `input`: shape `(b, n, m)`
- `mat2`: shape `(b, m, p)`
- Output: shape `(b, n, p)`

**Examples:**
```python
a = torch.randn(10, 3, 4)
b = torch.randn(10, 4, 5)
c = torch.bmm(a, b)            # shape: (10, 3, 5)
```

### 3.4.7 torch.matmul / torch.mm / torch.mv

```python
# General matrix multiplication (handles 1D and 2D, broadcasting for >2D)
torch.matmul(input, other, *, out=None)

# Matrix-matrix (2D only)
torch.mm(input, mat2, *, out=None)

# Matrix-vector
torch.mv(input, vec, *, out=None)
```

**`torch.matmul` behavior by input dimensions:**

| input shape | other shape | output shape | operation |
|------------|------------|------------|-----------|
| (n,) | (n,) | scalar | dot product |
| (n,) | (n, m) | (m,) | vector-matrix |
| (n, m) | (m,) | (n,) | matrix-vector |
| (n, m) | (m, p) | (n, p) | matrix-matrix |
| (b, n, m) | (m, p) | (b, n, p) | batched matmul |
| (b, n, m) | (b, m, p) | (b, n, p) | batched matmul |
| (b1, b2, n, m) | (b1, b2, m, p) | (b1, b2, n, p) | broadcast batched |

**Examples:**
```python
# Dot product
torch.matmul(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))  # tensor(32)

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
torch.matmul(A, B)              # shape: (3, 5)
torch.mm(A, B)                  # Same, but only for 2D tensors

# Matrix-vector
M = torch.randn(3, 4)
v = torch.randn(4)
torch.matmul(M, v)              # shape: (3,)
torch.mv(M, v)                  # Same, explicit matrix-vector

# Batched
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
torch.matmul(A, B)              # shape: (10, 3, 5)
```

### 3.4.8 torch.dot / torch.outer / torch.ger

```python
torch.dot(input, other)       # Dot product of 1D tensors
torch.outer(input, vec2)      # Outer product of 1D tensors
torch.ger(input, vec2)        # Outer product (deprecated, use torch.outer)
torch.inner(input, other)     # Inner product (generalized dot)
```

**Examples:**
```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
torch.dot(a, b)                 # tensor(32)
torch.outer(a, b)               # shape: (3, 3) - outer product
torch.inner(a, b)               # tensor(32) - same as dot for 1D
```

### 3.4.9 torch.chain_matmul

```python
torch.chain_matmul(*matrices)
```

Efficiently multiplies 2 or more matrices. Deprecated in favor of `torch.linalg.multi_matmul`.

### 3.4.10 Matrix Decompositions and Properties

```python
# Cholesky decomposition
torch.linalg.cholesky(input, *, upper=False, out=None)

# Determinant
torch.linalg.det(input)
torch.linalg.logdet(input)
torch.linalg.slogdet(input)   # Returns (sign, logabsdet)

# Inverse
torch.linalg.inv(input)
torch.linalg.inv_ex(input)    # Returns (inverse, info)

# Pseudo-inverse
torch.linalg.pinv(input)

# Matrix exponential
torch.linalg.matrix_exp(input)

# Matrix power
torch.linalg.matrix_power(input, n)

# Eigenvalues
torch.linalg.eig(input)
torch.linalg.eigvals(input)
torch.linalg.eigh(input)      # For Hermitian/symmetric matrices
torch.linalg.eigvalsh(input)

# SVD
torch.linalg.svd(input, full_matrices=True)
torch.linalg.svdvals(input)   # Only singular values

# Low-rank SVD
torch.svd_lowrank(input, q)

# PCA
torch.pca_lowrank(input, q=None, center=True, niter=5)

# QR decomposition
torch.linalg.qr(input, mode='reduced')

# LU decomposition
torch.linalg.lu(input)
torch.linalg.lu_factor(input)
torch.linalg.lu_solve(LU, pivots, B)

# Solve linear systems
torch.linalg.solve(input, B)
torch.linalg.solve_triangular(input, B, upper=True)

# Least squares
torch.linalg.lstsq(input, B, rcond=None)

# Norms
torch.linalg.norm(input, ord=None, dim=None, keepdim=False)
torch.linalg.matrix_norm(input, ord='fro', dim=(-2, -1))
torch.linalg.vector_norm(input, ord=2, dim=None)

# Condition number
torch.linalg.cond(input)

# Matrix rank
torch.linalg.matrix_rank(input, atol=None, rtol=None)

# Cross product (3D vectors)
torch.linalg.cross(input, other, dim=-1)

# Householder product
torch.linalg.householder_product(input, tau)
```

**Examples:**
```python
A = torch.randn(3, 3)

# Cholesky (requires positive definite)
P = A @ A.T + torch.eye(3) * 0.1  # Make positive definite
L = torch.linalg.cholesky(P)
# P = L @ L.T

# Determinant
torch.linalg.det(A)             # Scalar determinant

# Inverse
A_inv = torch.linalg.inv(A)
torch.allclose(A @ A_inv, torch.eye(3))  # True

# SVD
U, S, Vh = torch.linalg.svd(A)
# A = U @ diag(S) @ Vh

# Eigenvalues
eigenvalues = torch.linalg.eigvals(A)

# Solve linear system
A = torch.randn(3, 3)
B = torch.randn(3, 2)
X = torch.linalg.solve(A, B)   # A @ X = B

# Matrix norms
torch.linalg.matrix_norm(A)     # Frobenius norm
torch.linalg.matrix_norm(A, ord=1)  # Max column sum
torch.linalg.matrix_norm(A, ord=float('inf'))  # Max row sum
```

### 3.4.11 torch.vdot

```python
torch.vdot(input, other, *, out=None)
```

Dot product of complex vectors, conjugating the first argument.

### 3.4.12 torch.kron

```python
torch.kron(input, other)
```

Kronecker product of two tensors.

### 3.4.13 torch.lobpcg

```python
torch.lobpcg(input, k=2, B=None, X=None, niter=100, tol=None, largest=True, method=None)
```

Locally Optimal Block Preconditioned Conjugate Gradient method for eigenvalue problems.

---

## 3.5 Spectral Operations (Brief Overview)

The spectral operations are in `torch.fft`. Here is a brief cross-reference:

```python
# 1D FFT
torch.fft.fft(input, n=None, dim=-1, norm=None)
torch.fft.ifft(input, n=None, dim=-1, norm=None)

# Real FFT (optimized for real inputs)
torch.fft.rfft(input, n=None, dim=-1, norm=None)
torch.fft.irfft(input, n=None, dim=-1, norm=None)

# 2D FFT
torch.fft.fft2(input, s=None, dim=(-2, -1), norm=None)
torch.fft.ifft2(input, s=None, dim=(-2, -1), norm=None)

# N-dimensional FFT
torch.fft.fftn(input, s=None, dim=None, norm=None)
torch.fft.ifftn(input, s=None, dim=None, norm=None)

# Short-time Fourier Transform
torch.fft.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False)
torch.fft.istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False)

# FFT shift
torch.fft.fftshift(input, dim=None)
torch.fft.ifftshift(input, dim=None)

# Frequency bins
torch.fft.fftfreq(n, d=1.0)
torch.fft.rfftfreq(n, d=1.0)

# Window functions
torch.fft.hann_window(window_length, periodic=True)
torch.fft.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46)
torch.fft.bartlett_window(window_length, periodic=True)
torch.fft.blackman_window(window_length, periodic=True)
torch.fft.kaiser_window(window_length, periodic=False, beta=12.0)
```

See the FFT chapter (Chapter 35) for full details on spectral operations.

---

## 3.6 Other Common Operations

### 3.6.1 torch.cumsum / torch.cumprod / torch.cummax / torch.cummin

```python
torch.cumsum(input, dim, *, dtype=None, out=None)     # Cumulative sum
torch.cumprod(input, dim, *, dtype=None, out=None)    # Cumulative product
torch.cummax(input, dim, *, out=None)                  # Cumulative max (returns values, indices)
torch.cummin(input, dim, *, out=None)                  # Cumulative min (returns values, indices)
```

**Examples:**
```python
t = torch.tensor([1, 2, 3, 4, 5])
torch.cumsum(t, dim=0)          # tensor([1, 3, 6, 10, 15])
torch.cumprod(t, dim=0)         # tensor([1, 2, 6, 24, 120])
```

### 3.6.2 torch.diff

```python
torch.diff(input, n=1, dim=-1, prepend=None, append=None)
```

Computes the n-th discrete difference along a given dimension.

**Examples:**
```python
t = torch.tensor([1, 3, 6, 10])
torch.diff(t)                   # tensor([2, 3, 4])
torch.diff(t, n=2)              # tensor([1, 1]) - second difference
```

### 3.6.3 torch.einsum

Einstein summation convention.

```python
torch.einsum(equation, *operands)
```

**Examples:**
```python
# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
torch.einsum('ij,jk->ik', A, B)  # Same as A @ B

# Dot product
a = torch.randn(4)
b = torch.randn(4)
torch.einsum('i,i->', a, b)    # Same as torch.dot(a, b)

# Outer product
torch.einsum('i,j->ij', a, b)  # Same as torch.outer(a, b)

# Batch matrix multiply
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
torch.einsum('bij,bjk->bik', A, B)  # Same as torch.bmm(A, B)

# Trace
M = torch.randn(4, 4)
torch.einsum('ii->', M)        # Same as torch.trace(M)

# Transpose
torch.einsum('ij->ji', A)      # Same as A.T for 2D

# Sum
torch.einsum('ij->', M)        # Sum of all elements

# Diagonal
torch.einsum('ii->i', M)       # Same as torch.diag(M)

# Batch diagonal
B = torch.randn(10, 4, 4)
torch.einsum('bii->bi', B)     # Batch diagonal extraction

# Ellipsis for broadcast dims
torch.einsum('...ij,...jk->...ik', A, B)  # Works with any batch dims
```

### 3.6.4 torch.tensordot

```python
torch.tensordot(input, other, dims=2)
```

**Examples:**
```python
a = torch.arange(6.).reshape(2, 3)
b = torch.arange(9.).reshape(3, 3)
torch.tensordot(a, b, dims=1)  # Contract last dim of a with first dim of b
torch.tensordot(a, b, dims=[[1], [0]])  # Same, explicit dims
```

### 3.6.5 torch.triu_indices / torch.tril_indices

```python
torch.triu_indices(row, col, offset=0, dtype=torch.int64, device='cpu')
torch.tril_indices(row, col, offset=0, dtype=torch.int64, device='cpu')
```

### 3.6.6 torch.meshgrid / torch.cartesian_prod

```python
# Create coordinate grids from 1D tensors
torch.meshgrid(*tensors, indexing='ij')

# Cartesian product of 1D tensors
torch.cartesian_prod(*tensors)
```

**Examples:**
```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])
gx, gy = torch.meshgrid(x, y, indexing='ij')
# gx: tensor([[1, 1], [2, 2], [3, 3]])
# gy: tensor([[4, 5], [4, 5], [4, 5]])

torch.cartesian_prod(x, y)
# tensor([[1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]])
```

---

## 3.7 Summary of Operations by Category

| Category | Operations |
|----------|-----------|
| **Pointwise Math** | `abs`, `acos`, `acosh`, `add`, `addcdiv`, `addcmul`, `angle`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `bitwise_*`, `ceil`, `clamp`, `clip`, `conj`, `cos`, `cosh`, `deg2rad`, `digamma`, `div`, `divide`, `erf`, `erfc`, `erfinv`, `exp`, `exp2`, `expm1`, `floor`, `fmod`, `frac`, `gradient`, `hypot`, `i0`, `igamma`, `igammac`, `imag`, `ldexp`, `lerp`, `lgamma`, `log`, `log2`, `log10`, `log1p`, `logaddexp`, `logical_*`, `logit`, `mul`, `nan_to_num`, `neg`, `nextafter`, `polygamma`, `pow`, `rad2deg`, `real`, `reciprocal`, `remainder`, `round`, `rsqrt`, `sigmoid`, `sign`, `sgn`, `signbit`, `sin`, `sinc`, `sinh`, `sqrt`, `square`, `sub`, `tan`, `tanh`, `trunc` |
| **Reductions** | `argmax`, `argmin`, `amax`, `amin`, `aminmax`, `all`, `any`, `count_nonzero`, `dist`, `logsumexp`, `mean`, `median`, `mode`, `norm`, `nansum`, `nanmean`, `prod`, `quantile`, `std`, `std_mean`, `sum`, `unique`, `var`, `var_mean` |
| **Comparison** | `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `isclose`, `equal`, `max`, `min`, `topk`, `sort`, `argsort`, `kthvalue`, `isfinite`, `isinf`, `isnan`, `isreal` |
| **BLAS/LAPACK** | `addbmm`, `addmm`, `addmv`, `addr`, `baddbmm`, `bmm`, `chain_matmul`, `cholesky`, `det`, `dot`, `ger`, `inner`, `inverse`, `logdet`, `lstsq`, `matmul`, `matrix_exp`, `matrix_power`, `mm`, `mv`, `outer`, `pca_lowrank`, `svd`, `svd_lowrank`, `lobpcg` |
| **Spectral** | `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `stft`, `istft` |
| **Utility** | `cumsum`, `cumprod`, `cummax`, `cummin`, `diff`, `einsum`, `tensordot`, `meshgrid`, `cartesian_prod` |
