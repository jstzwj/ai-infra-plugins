# Math Operations

## Overview

cuTile provides a comprehensive set of element-wise mathematical operations that mirror NumPy and PyTorch functionality while being optimized for GPU execution. These operations work on `Tile` objects and can be used within kernels to perform computations on tiles of data.

This chapter covers all element-wise math operations organized into categories:
- **Arithmetic operations**: Basic mathematical computations
- **Floating-point checks**: Special value detection
- **Exponential and logarithm**: Powers and roots
- **Trigonometric**: Standard trig functions and hyperbolic variants
- **Rounding**: Floor and ceiling operations

## Arithmetic Operations

### ct.add() — Element-wise Addition

#### Syntax
```python
ct.add(a, b) -> Tile
# Or using operator:
a + b
```

#### Description
Performs element-wise addition of two tiles.

#### Parameters
- `a` (`Tile`): First operand
- `b` (`Tile`): Second operand (must be broadcastable to shape of `a`)

#### Returns
- `Tile`: Element-wise sum `a + b`

#### Supported Data Types
| Input Type | Return Type | Notes |
|------------|-------------|-------|
| `int8`, `int16`, `int32`, `int64` | Same as input | May overflow |
| `float16`, `float32`, `float64` | Same as input | IEEE 754 arithmetic |
| `bfloat16` | `bfloat16` | Reduced precision |

#### Example
```python
@ct.kernel
def add_kernel(A: ct.Buffer, B: ct.Buffer, C: ct.Buffer):
    """C = A + B"""
    i, j = ct.bid(0), ct.bid(1)
    
    a_tile = A.load((i, j))
    b_tile = B.load((i, j))
    
    # Using function
    c_tile = ct.add(a_tile, b_tile)
    # Or using operator
    c_tile = a_tile + b_tile
    
    C.store((i, j), c_tile)
```

#### Broadcasting
```python
@ct.kernel
def add_broadcast(A: ct.Buffer, B: ct.Buffer, C: ct.Buffer):
    """Add vector to each row of matrix."""
    i, j = ct.bid(0), ct.bid(1)
    
    # A: (M, N), B: (N,)
    a_tile = A.load((i, j))
    b_tile = B.load((j,))  # Broadcasts across M dimension
    
    c_tile = a_tile + b_tile
    C.store((i, j), c_tile)
```

### ct.sub() — Element-wise Subtraction

#### Syntax
```python
ct.sub(a, b) -> Tile
# Or using operator:
a - b
```

#### Description
Performs element-wise subtraction: `a - b`.

#### Parameters
- `a` (`Tile`): Minuend
- `b` (`Tile`): Subtrahend

#### Returns
- `Tile`: Element-wise difference `a - b`

#### Example
```python
@ct.kernel
def sub_kernel(A: ct.Buffer, B: ct.Buffer, C: ct.Buffer):
    """C = A - B"""
    i, j = ct.bid(0), ct.bid(1)
    
    a_tile = A.load((i, j))
    b_tile = B.load((i, j))
    
    c_tile = a_tile - b_tile
    C.store((i, j), c_tile)
```

### ct.mul() — Element-wise Multiplication

#### Syntax
```python
ct.mul(a, b) -> Tile
# Or using operator:
a * b
```

#### Description
Performs element-wise multiplication (NOT matrix multiplication).

#### Parameters
- `a` (`Tile`): First factor
- `b` (`Tile`): Second factor

#### Returns
- `Tile`: Element-wise product `a * b`

#### Example
```python
@ct.kernel
def scale_kernel(X: ct.Buffer, scale: ct.Buffer, Y: ct.Buffer):
    """Y = X * scale (element-wise scaling)"""
    i, j = ct.bid(0), ct.bid(1)
    
    x_tile = X.load((i, j))
    scale_tile = scale.load((j,))  # Per-dimension scaling
    
    y_tile = x_tile * scale_tile
    Y.store((i, j), y_tile)
```

### ct.truediv() — True Division

#### Syntax
```python
ct.truediv(a, b) -> Tile
# Or using operator:
a / b
```

#### Description
Performs true division, always returning a floating-point result.

#### Parameters
- `a` (`Tile`): Dividend
- `b` (`Tile`): Divisor

#### Returns
- `Tile`: Quotient `a / b` (always floating-point)

#### Supported Data Types
| Input Type | Return Type |
|------------|-------------|
| Any integer type | `float32` |
| `float16`, `float32`, `float64` | Same as input |
| `bfloat16` | `bfloat16` |

#### Example
```python
@ct.kernel
def normalize_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Normalize by dividing by sum."""
    i, j = ct.bid(0), ct.bid(1)
    
    x_tile = X.load((i, j))
    x_sum = ct.sum(x_tile)
    
    # True division returns float
    y_tile = x_tile / x_sum
    Y.store((i, j), y_tile)
```

### ct.floordiv() — Floor Division

#### Syntax
```python
ct.floordiv(a, b) -> Tile
# Or using operator:
a // b
```

#### Description
Performs floor division: `floor(a / b)`.

#### Parameters
- `a` (`Tile`): Dividend
- `b` (`Tile`): Divisor

#### Returns
- `Tile`: Floor of quotient

#### Example
```python
@ct.kernel
def block_index_kernel(global_idx: ct.Buffer, block_size: ct.Buffer, block_idx: ct.Buffer):
    """Compute which block each element belongs to."""
    i = ct.tid()
    
    idx = global_idx.load(i)
    bs = block_size.load(())
    
    # Integer division
    bi = idx // bs
    block_idx.store(i, bi)
```

### ct.cdiv() — Ceiling Division

#### Syntax
```python
ct.cdiv(x, y) -> Tile
```

#### Description
Computes ceiling of division: `ceil(x / y)`. This is commonly used for calculating grid dimensions.

#### Parameters
- `x` (`Tile`): Dividend
- `y` (`Tile`): Divisor

#### Returns
- `Tile`: Smallest integer ≥ x/y

#### Use Case
This function is particularly useful for computing the number of blocks needed in a grid:

```python
def launch_kernel(size: int, block_size: int = 256):
    """Launch kernel with appropriate grid size."""
    # Calculate number of blocks needed
    num_blocks = ct.cdiv(size, block_size)
    
    kernel[(num_blocks,)](size, block_size)
```

#### Example
```python
@ct.kernel
def compute_grid_dims(N: ct.Constant[int], block_size: ct.Constant[int]):
    """Compute grid dimensions for a problem."""
    num_blocks = (N + block_size - 1) // block_size
    # Or using ct.cdiv:
    num_blocks = ct.cdiv(N, block_size)
    
    print(f"Grid size: {num_blocks} blocks")
```

### ct.pow() — Power

#### Syntax
```python
ct.pow(a, b) -> Tile
# Or using operator:
a ** b
```

#### Description
Raises elements of `a` to the power of elements in `b`.

#### Parameters
- `a` (`Tile`): Base
- `b` (`Tile`): Exponent

#### Returns
- `Tile`: `a^b`

#### Supported Data Types
| Base Type | Exponent Type | Return Type | Notes |
|-----------|---------------|-------------|-------|
| Float | Float | Float | Standard power |
| Float | Integer | Float | Integer exponent |
| Integer | Integer (positive) | Float | Returns float |

#### Example
```python
@ct.kernel
def squared_distance_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute squared Euclidean distance."""
    i, j = ct.bid(0), ct.bid(1)
    
    x = X.load(i)
    y = Y.load(j)
    
    # Square the difference
    diff = x - y
    sq_dist = diff ** 2  # or ct.pow(diff, 2)
    
    Y.store((i, j), sq_dist)
```

### ct.mod() — Modulo

#### Syntax
```python
ct.mod(a, b) -> Tile
# Or using operator:
a % b
```

#### Description
Computes remainder of division: `a mod b`.

#### Parameters
- `a` (`Tile`): Dividend
- `b` (`Tile`): Divisor

#### Returns
- `Tile`: Remainder

#### Example
```python
@ct.kernel
def periodic_kernel(x: ct.Buffer, y: ct.Buffer, period: ct.Constant[int]):
    """Apply periodic boundary conditions."""
    i = ct.tid()
    
    xi = x.load(i)
    
    # Wrap to [0, period)
    xi_wrapped = xi % period
    
    y.store(i, xi_wrapped)
```

### ct.minimum() — Element-wise Minimum

#### Syntax
```python
ct.minimum(a, b) -> Tile
```

#### Description
Computes element-wise minimum of two tiles.

#### Parameters
- `a` (`Tile`): First input
- `b` (`Tile`): Second input

#### Returns
- `Tile`: Element-wise minimum

#### Example
```python
@ct.kernel
def relu_kernel(X: ct.Buffer, Y: ct.Buffer):
    """ReLU activation: max(0, x)."""
    i = ct.tid()
    
    x = X.load(i)
    zero = ct.zeros_like(x)
    
    # ReLU = max(0, x) = -min(0, -x) or use ct.maximum
    y = ct.maximum(x, 0)
    Y.store(i, y)
```

### ct.maximum() — Element-wise Maximum

#### Syntax
```python
ct.maximum(a, b) -> Tile
```

#### Description
Computes element-wise maximum of two tiles.

#### Parameters
- `a` (`Tile`): First input
- `b` (`Tile`): Second input

#### Returns
- `Tile`: Element-wise maximum

#### Example
```python
@ct.kernel
def leaky_relu_kernel(X: ct.Buffer, Y: ct.Buffer, alpha: ct.Constant[float] = 0.01):
    """Leaky ReLU: max(alpha*x, x)."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Leaky ReLU
    y = ct.maximum(alpha * x, x)
    Y.store(i, y)
```

### ct.negative() — Negation

#### Syntax
```python
ct.negative(x) -> Tile
# Or using operator:
-x
```

#### Description
Computes element-wise negation.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: Negated values

#### Example
```python
@ct.kernel
def negate_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Y = -X"""
    i = ct.tid()
    
    x = X.load(i)
    y = -x  # or ct.negative(x)
    Y.store(i, y)
```

### ct.abs() — Absolute Value

#### Syntax
```python
ct.abs(x) -> Tile
# Or using function:
abs(x)
```

#### Description
Computes element-wise absolute value.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: Absolute values

#### Supported Data Types
All numeric types are supported. Return type matches input type.

#### Example
```python
@ct.kernel
def abs_error_kernel(predicted: ct.Buffer, actual: ct.Buffer, error: ct.Buffer):
    """Compute absolute error."""
    i = ct.tid()
    
    pred = predicted.load(i)
    act = actual.load(i)
    
    err = ct.abs(pred - act)
    error.store(i, err)
```

## Floating-Point Checks

### ct.isnan() — Check for NaN

#### Syntax
```python
ct.isnan(x) -> Tile
```

#### Description
Tests element-wise for NaN (Not a Number) values.

#### Parameters
- `x` (`Tile`): Input tile (floating-point type)

#### Returns
- `Tile`: Boolean tile with `True` where `x` is NaN

#### Supported Data Types
`float16`, `float32`, `float64`, `bfloat16`

#### Example
```python
@ct.kernel
def check_nan_kernel(X: ct.Buffer, has_nan: ct.Buffer):
    """Check if any NaN values exist."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Check for NaN
    is_nan = ct.isnan(x)
    
    # Store result
    has_nan.store(i, is_nan)

# Usage: Check if any NaN in array
any_nan = ct.zeros((1,), dtype=ct.bool)
check_nan[grid_size](X, any_nan)

# Reduce to get final answer
if any_nan.cpu().any():
    print("NaN detected!")
```

#### Practical Example: NaN-Safe Operations

```python
@ct.kernel
def nan_safe_divide_kernel(
    A: ct.Buffer,
    B: ct.Buffer,
    C: ct.Buffer,
    fill_value: ct.Constant[float] = 0.0
):
    """Division that replaces NaN results with fill_value."""
    i, j = ct.bid(0), ct.bid(1)
    
    a = A.load((i, j))
    b = B.load((i, j))
    
    # Perform division
    result = a / b
    
    # Check for NaN and replace
    is_valid = ~ct.isnan(result)
    final_result = ct.where(is_valid, result, fill_value)
    
    C.store((i, j), final_result)
```

## Exponential and Logarithm Operations

### ct.exp() — Exponential

#### Syntax
```python
ct.exp(x) -> Tile
```

#### Description
Computes element-wise exponential: `e^x`.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: `e^x`

#### Supported Data Types
`float16`, `float32`, `float64`, `bfloat16`

#### Precision
- `float32`: ~1 ULP accuracy
- `float16`: Reduced precision (~2-3 ULP)
- `bfloat16`: Reduced precision

#### Example
```python
@ct.kernel
def softmax_exp_kernel(X: ct.Buffer, Y: ct.Buffer):
    """First step of softmax: compute exp(x)."""
    i, j = ct.bid(0), ct.bid(1)
    
    x = X.load((i, j))
    y = ct.exp(x)
    
    Y.store((i, j), y)
```

### ct.exp2() — Base-2 Exponential

#### Syntax
```python
ct.exp2(x) -> Tile
```

#### Description
Computes element-wise base-2 exponential: `2^x`.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: `2^x`

#### Example
```python
@ct.kernel
def quantization_scale_kernel(exponent: ct.Buffer, scale: ct.Buffer):
    """Compute scale = 2^exponent for quantization."""
    i = ct.tid()
    
    exp = exponent.load(i)
    s = ct.exp2(exp)
    
    scale.store(i, s)
```

### ct.log() — Natural Logarithm

#### Syntax
```python
ct.log(x) -> Tile
```

#### Description
Computes element-wise natural logarithm: `log_e(x)`.

#### Parameters
- `x` (`Tile`): Input tile (must be positive)

#### Returns
- `Tile`: `ln(x)`

#### Supported Data Types
`float16`, `float32`, `float64`, `bfloat16`

#### Domain
Input must be > 0. Results are undefined for non-positive inputs.

#### Example
```python
@ct.kernel
def log_softmax_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Log-softmax: x - log(sum(exp(x)))."""
    i, j = ct.bid(0), ct.bid(1)
    
    x = X.load((i, j))
    
    # Numerically stable log-softmax
    x_max = ct.max(x, axis=1, keepdim=True)
    exp_x = ct.exp(x - x_max)
    sum_exp = ct.sum(exp_x, axis=1, keepdim=True)
    
    y = x - x_max - ct.log(sum_exp)
    Y.store((i, j), y)
```

### ct.log2() — Base-2 Logarithm

#### Syntax
```python
ct.log2(x) -> Tile
```

#### Description
Computes element-wise base-2 logarithm: `log_2(x)`.

#### Parameters
- `x` (`Tile`): Input tile (must be positive)

#### Returns
- `Tile`: `log_2(x)`

#### Example
```python
@ct.kernel
def bit_length_kernel(value: ct.Buffer, bits: ct.Buffer):
    """Compute number of bits needed to represent value."""
    i = ct.tid()
    
    v = value.load(i)
    b = ct.floor(ct.log2(v)) + 1
    
    bits.store(i, b.astype(ct.int32))
```

### ct.sqrt() — Square Root

#### Syntax
```python
ct.sqrt(x) -> Tile
```

#### Description
Computes element-wise square root.

#### Parameters
- `x` (`Tile`): Input tile (must be non-negative)

#### Returns
- `Tile`: `sqrt(x)`

#### Supported Data Types
`float16`, `float32`, `float64`, `bfloat16`

#### Example
```python
@ct.kernel
def l2_normalize_kernel(X: ct.Buffer, Y: ct.Buffer):
    """L2 normalization: x / ||x||_2."""
    i = ct.bid(0)
    
    x = X.load(i)
    
    # Compute L2 norm
    norm = ct.sqrt(ct.sum(x ** 2))
    
    # Normalize
    y = x / norm
    Y.store(i, y)
```

### ct.rsqrt() — Reciprocal Square Root

#### Syntax
```python
ct.rsqrt(x) -> Tile
```

#### Description
Computes element-wise reciprocal square root: `1 / sqrt(x)`.

#### Parameters
- `x` (`Tile`): Input tile (must be positive)

#### Returns
- `Tile`: `1/sqrt(x)`

#### Performance Notes
This is often implemented as a single hardware instruction on GPUs and is faster than computing `1.0 / ct.sqrt(x)`.

#### Example
```python
@ct.kernel
def rms_norm_kernel(
    X: ct.Buffer,
    Gamma: ct.Buffer,
    Y: ct.Buffer,
    eps: ct.Constant[float] = 1e-5
):
    """Root Mean Square Layer Normalization."""
    i, j = ct.bid(0), ct.bid(1)
    
    x = X.load((i, j))
    gamma = Gamma.load((j,))
    
    # Compute RMS
    mean_square = ct.mean(x ** 2, axis=1, keepdim=True)
    rms = ct.rsqrt(mean_square + eps)
    
    # Normalize and scale
    y = x * rms * gamma
    Y.store((i, j), y)
```

## Trigonometric Operations

### ct.sin() — Sine

#### Syntax
```python
ct.sin(x) -> Tile
```

#### Description
Computes element-wise sine of angle (in radians).

#### Parameters
- `x` (`Tile`): Angle in radians

#### Returns
- `Tile`: `sin(x)`

#### Supported Data Types
`float16`, `float32`, `float64`, `bfloat16`

#### Precision
Results are accurate to within 1-2 ULP for `float32`.

#### Example
```python
@ct.kernel
def sin_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute sine of each element."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.sin(x)
    
    Y.store(i, y)
```

### ct.cos() — Cosine

#### Syntax
```python
ct.cos(x) -> Tile
```

#### Description
Computes element-wise cosine of angle (in radians).

#### Parameters
- `x` (`Tile`): Angle in radians

#### Returns
- `Tile`: `cos(x)`

#### Example
```python
@ct.kernel
def rotation_kernel(x: ct.Buffer, y: ct.Buffer, theta: ct.Constant[float]):
    """Apply 2D rotation."""
    i = ct.tid()
    
    xi = x.load(i)
    yi = y.load(i)
    
    # Rotation matrix
    cos_t = ct.cos(theta)
    sin_t = ct.sin(theta)
    
    # Rotate
    x_new = cos_t * xi - sin_t * yi
    y_new = sin_t * xi + cos_t * yi
    
    x.store(i, x_new)
    y.store(i, y_new)
```

### ct.tan() — Tangent

#### Syntax
```python
ct.tan(x) -> Tile
```

#### Description
Computes element-wise tangent of angle (in radians).

#### Parameters
- `x` (`Tile`): Angle in radians

#### Returns
- `Tile`: `tan(x)`

#### Notes
Results approach infinity at `π/2 + kπ` (undefined).

#### Example
```python
@ct.kernel
def tan_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute tangent of each element."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.tan(x)
    
    Y.store(i, y)
```

### ct.sinh() — Hyperbolic Sine

#### Syntax
```python
ct.sinh(x) -> Tile
```

#### Description
Computes element-wise hyperbolic sine: `(e^x - e^(-x)) / 2`.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: `sinh(x)`

#### Example
```python
@ct.kernel
def sinh_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute hyperbolic sine."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.sinh(x)
    
    Y.store(i, y)
```

### ct.cosh() — Hyperbolic Cosine

#### Syntax
```python
ct.cosh(x) -> Tile
```

#### Description
Computes element-wise hyperbolic cosine: `(e^x + e^(-x)) / 2`.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: `cosh(x)`

#### Example
```python
@ct.kernel
def cosh_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute hyperbolic cosine."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.cosh(x)
    
    Y.store(i, y)
```

### ct.tanh() — Hyperbolic Tangent

#### Syntax
```python
ct.tanh(x, rounding_mode: str = None) -> Tile
```

#### Description
Computes element-wise hyperbolic tangent: `sinh(x) / cosh(x)`.

#### Parameters
- `x` (`Tile`): Input tile
- `rounding_mode` (`str`, optional): Rounding mode for computation

#### Returns
- `Tile`: `tanh(x)`

#### Range
Output is in range (-1, 1).

#### Example
```python
@ct.kernel
def tanh_activation_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Tanh activation function."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.tanh(x)
    
    Y.store(i, y)
```

#### Tanh for Gradient Clipping

```python
@ct.kernel
def tanh_clip_kernel(gradient: ct.Buffer, clipped: ct.Buffer):
    """Clip gradients using tanh."""
    i = ct.tid()
    
    grad = gradient.load(i)
    
    # Clip to [-1, 1] using tanh
    clipped_grad = ct.tanh(grad)
    
    clipped.store(i, clipped_grad)
```

## Rounding Operations

### ct.floor() — Floor

#### Syntax
```python
ct.floor(x) -> Tile
```

#### Description
Rounds elements down to the nearest integer.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: Floor of `x` (same dtype as input)

#### Example
```python
@ct.kernel
def floor_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute floor of each element."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.floor(x)
    
    Y.store(i, y)
```

### ct.ceil() — Ceiling

#### Syntax
```python
ct.ceil(x) -> Tile
```

#### Description
Rounds elements up to the nearest integer.

#### Parameters
- `x` (`Tile`): Input tile

#### Returns
- `Tile`: Ceiling of `x` (same dtype as input)

#### Example
```python
@ct.kernel
def ceil_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Compute ceiling of each element."""
    i = ct.tid()
    
    x = X.load(i)
    y = ct.ceil(x)
    
    Y.store(i, y)
```

## Operator Overloading

cuTile tiles support Python's standard operators for element-wise operations:

| Operator | Function | Description |
|----------|----------|-------------|
| `a + b` | `ct.add(a, b)` | Addition |
| `a - b` | `ct.sub(a, b)` | Subtraction |
| `a * b` | `ct.mul(a, b)` | Multiplication |
| `a / b` | `ct.truediv(a, b)` | True division |
| `a // b` | `ct.floordiv(a, b)` | Floor division |
| `a ** b` | `ct.pow(a, b)` | Power |
| `a % b` | `ct.mod(a, b)` | Modulo |
| `-a` | `ct.negative(a)` | Negation |
| `abs(a)` | `ct.abs(a)` | Absolute value |
| `a & b` | `ct.bitwise_and(a, b)` | Bitwise AND |
| `a \| b` | `ct.bitwise_or(a, b)` | Bitwise OR |
| `a ^ b` | `ct.bitwise_xor(a, b)` | Bitwise XOR |
| `a << b` | `ct.bitwise_lshift(a, b)` | Left shift |
| `a >> b` | `ct.bitwise_rshift(a, b)` | Right shift |
| `~a` | `ct.bitwise_not(a)` | Bitwise NOT |
| `a > b` | `ct.greater(a, b)` | Greater than |
| `a >= b` | `ct.greater_equal(a, b)` | Greater or equal |
| `a < b` | `ct.less(a, b)` | Less than |
| `a <= b` | `ct.less_equal(a, b)` | Less or equal |
| `a == b` | `ct.equal(a, b)` | Equality |
| `a != b` | `ct.not_equal(a, b)` | Inequality |

## Complete Examples

### Example 1: Sigmoid Activation

```python
@ct.kernel
def sigmoid_kernel(X: ct.Buffer, Y: ct.Buffer):
    """Sigmoid: 1 / (1 + exp(-x))."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Sigmoid
    y = 1.0 / (1.0 + ct.exp(-x))
    
    Y.store(i, y)
```

### Example 2: GELU Activation

```python
@ct.kernel
def gelu_kernel(X: ct.Buffer, Y: ct.Buffer):
    """
    GELU: Gaussian Error Linear Unit.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    i = ct.tid()
    
    x = X.load(i)
    
    # Constants
    sqrt_2_over_pi = ct.sqrt(2.0 / 3.141592653589793)
    
    # GELU approximation
    cube = x ** 3
    inner = sqrt_2_over_pi * (x + 0.044715 * cube)
    tanh_val = ct.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_val)
    
    Y.store(i, y)
```

### Example 3: Swish Activation

```python
@ct.kernel
def swish_kernel(X: ct.Buffer, Y: ct.Buffer, beta: ct.Constant[float] = 1.0):
    """Swish: x * sigmoid(beta * x)."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Swish
    sigmoid_x = 1.0 / (1.0 + ct.exp(-beta * x))
    y = x * sigmoid_x
    
    Y.store(i, y)
```

### Example 4: Element-wise Linear Combination

```python
@ct.kernel
def linear_combination_kernel(
    X: ct.Buffer,
    Y: ct.Buffer,
    alpha: ct.Constant[float],
    beta: ct.Constant[float],
    Z: ct.Buffer
):
    """Z = alpha * X + beta * Y."""
    i, j = ct.bid(0), ct.bid(1)
    
    x = X.load((i, j))
    y = Y.load((i, j))
    
    # Linear combination
    z = alpha * x + beta * y
    
    Z.store((i, j), z)
```

### Example 5: Clip and Normalize

```python
@ct.kernel
def clip_normalize_kernel(
    X: ct.Buffer,
    Y: ct.Buffer,
    min_val: ct.Constant[float],
    max_val: ct.Constant[float]
):
    """Clip values to [min_val, max_val] and normalize to [0, 1]."""
    i = ct.tid()
    
    x = X.load(i)
    
    # Clip
    x_clipped = ct.minimum(ct.maximum(x, min_val), max_val)
    
    # Normalize to [0, 1]
    range_val = max_val - min_val
    x_normalized = (x_clipped - min_val) / range_val
    
    Y.store(i, x_normalized)
```

### Example 6: Complex Number Operations

```python
@ct.kernel
def complex_multiply_kernel(
    real_a: ct.Buffer,
    imag_a: ct.Buffer,
    real_b: ct.Buffer,
    imag_b: ct.Buffer,
    real_c: ct.Buffer,
    imag_c: ct.Buffer
):
    """
    Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    """
    i = ct.tid()
    
    ar = real_a.load(i)
    ai = imag_a.load(i)
    br = real_b.load(i)
    bi = imag_b.load(i)
    
    # Complex multiplication
    cr = ar * br - ai * bi
    ci = ar * bi + ai * br
    
    real_c.store(i, cr)
    imag_c.store(i, ci)
```

## Summary

cuTile provides comprehensive element-wise math operations covering:

- **Arithmetic**: add, sub, mul, truediv, floordiv, cdiv, pow, mod, minimum, maximum, negative, abs
- **Floating-point checks**: isnan
- **Exponential/logarithm**: exp, exp2, log, log2, sqrt, rsqrt
- **Trigonometric**: sin, cos, tan, sinh, cosh, tanh
- **Rounding**: floor, ceil

**Key takeaways:**
1. All operations support broadcasting for flexible tensor shapes
2. Operator overloading makes code readable (e.g., `a + b` instead of `ct.add(a, b)`)
3. Precision varies by data type (FP32 most accurate, FP16/BF16 reduced precision)
4. Use functions like `ct.rsqrt()` for better performance than `1.0 / ct.sqrt(x)`
5. Check for NaN values when numerical stability is a concern

The next chapter covers bitwise and comparison operations for creating masks and conditional logic.
