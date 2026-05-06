# 18. CUDA Math Functions Reference

This section provides a comprehensive reference for all mathematical functions available in CUDA device code, including standard math functions, intrinsics, non-standard extensions, half-precision math, and the effects of `--use_fast_math`.

---

## 18.1 Standard Math Functions

CUDA device code supports the full set of C/C++ standard math library functions. Each function is available for `double`, and most are also available for `float` (with the `f` suffix) and for `__half` / `__nv_bfloat16` types. These functions are defined in `<math.h>` and are accessible in `__device__` code.

### 18.1.1 Trigonometric Functions

| Function (float / double) | Description |
|---|---|
| `sinf(x)` / `sin(x)` | Sine of `x` (radians) |
| `cosf(x)` / `cos(x)` | Cosine of `x` (radians) |
| `tanf(x)` / `tan(x)` | Tangent of `x` (radians) |
| `asinf(x)` / `asin(x)` | Arc sine of `x` |
| `acosf(x)` / `acos(x)` | Arc cosine of `x` |
| `atanf(x)` / `atan(x)` | Arc tangent of `x` |
| `atan2f(y, x)` / `atan2(y, x)` | Arc tangent of `y/x` using sign of both arguments |

**Accuracy**: These functions meet the accuracy requirements of the IEEE-754 standard. Single-precision functions typically deliver results correctly rounded for the 4 IEEE rounding modes. Double-precision functions are also correctly rounded.

```cpp
__device__ void trig_example(float angle) {
    float s = sinf(angle);
    float c = cosf(angle);
    float t = tanf(angle);

    float angle_back = atan2f(s, c); // recover angle

    // Double-precision variants
    double sd = sin((double)angle);
    double cd = cos((double)angle);
}
```

### 18.1.2 Hyperbolic Functions

| Function (float / double) | Description |
|---|---|
| `sinhf(x)` / `sinh(x)` | Hyperbolic sine of `x` |
| `coshf(x)` / `cosh(x)` | Hyperbolic cosine of `x` |
| `tanhf(x)` / `tanh(x)` | Hyperbolic tangent of `x` |
| `asinhf(x)` / `asinh(x)` | Inverse hyperbolic sine of `x` |
| `acoshf(x)` / `acosh(x)` | Inverse hyperbolic cosine of `x` |
| `atanhf(x)` / `atanh(x)` | Inverse hyperbolic tangent of `x` |

```cpp
__device__ float hyperbolic_example(float x) {
    float sh = sinhf(x);
    float ch = coshf(x);
    float th = tanhf(x);

    // Inverse: recover x from th
    float recovered = atanhf(th);
    return recovered;
}
```

### 18.1.3 Exponential and Logarithmic Functions

| Function (float / double) | Description |
|---|---|
| `expf(x)` / `exp(x)` | e^x |
| `exp2f(x)` / `exp2(x)` | 2^x |
| `exp10f(x)` / `exp10(x)` | 10^x |
| `expm1f(x)` / `expm1(x)` | e^x - 1 (accurate for small x) |
| `logf(x)` / `log(x)` | Natural logarithm of `x` |
| `log2f(x)` / `log2(x)` | Base-2 logarithm of `x` |
| `log10f(x)` / `log10(x)` | Base-10 logarithm of `x` |
| `log1pf(x)` / `log1p(x)` | log(1 + x) (accurate for small x) |

```cpp
__device__ void exp_log_example(float x) {
    float e = expf(x);          // e^x
    float e2 = exp2f(x);        // 2^x
    float e10 = exp10f(x);      // 10^x
    float em1 = expm1f(x);      // e^x - 1, accurate near 0

    float ln = logf(x);         // ln(x)
    float lg2 = log2f(x);       // log2(x)
    float lg10 = log10f(x);     // log10(x)
    float lp1 = log1pf(x);      // ln(1+x), accurate near 0
}
```

### 18.1.4 Power and Root Functions

| Function (float / double) | Description |
|---|---|
| `powf(x, y)` / `pow(x, y)` | x^y |
| `sqrtf(x)` / `sqrt(x)` | Square root of `x` |
| `rsqrtf(x)` / `rsqrt(x)` | Reciprocal square root: 1/sqrt(x) |
| `cbrtf(x)` / `cbrt(x)` | Cube root of `x` |
| `rcbrtf(x)` / `rcbrt(x)` | Reciprocal cube root: 1/cbrt(x) |
| `hypotf(x, y)` / `hypot(x, y)` | sqrt(x^2 + y^2) |

```cpp
__device__ void power_root_example(float x, float y) {
    float p = powf(x, y);       // x^y
    float s = sqrtf(x);         // sqrt(x)
    float rs = rsqrtf(x);       // 1/sqrt(x), faster than 1.0f/sqrtf(x)
    float cb = cbrtf(x);        // cbrt(x)
    float rcb = rcbrtf(x);      // 1/cbrt(x)
    float h = hypotf(x, y);     // sqrt(x^2 + y^2) without overflow
}
```

### 18.1.5 Rounding and Integer Conversion Functions

| Function (float / double) | Description |
|---|---|
| `ceilf(x)` / `ceil(x)` | Ceiling: smallest integer >= x |
| `floorf(x)` / `floor(x)` | Floor: largest integer <= x |
| `truncf(x)` / `trunc(x)` | Truncate toward zero |
| `roundf(x)` / `round(x)` | Round to nearest integer, away from zero |
| `nearbyintf(x)` / `nearbyint(x)` | Round to nearest integer, current rounding mode |
| `rintf(x)` / `rint(x)` | Same as nearbyint, but may raise FE_INEXACT |
| `llrintf(x)` / `llrint(x)` | Round to `long long` using current rounding mode |
| `llroundf(x)` / `llround(x)` | Round to `long long`, away from zero |
| `lrintf(x)` / `lrint(x)` | Round to `long` using current rounding mode |
| `lroundf(x)` / `lround(x)` | Round to `long`, away from zero |

```cpp
__device__ void rounding_example(float x) {
    float c = ceilf(x);       // e.g., 2.3 -> 3.0
    float f = floorf(x);      // e.g., 2.7 -> 2.0
    float t = truncf(x);      // e.g., -2.7 -> -2.0
    float r = roundf(x);      // e.g., 2.5 -> 3.0
    float nr = nearbyintf(x); // depends on rounding mode
    float ri = rintf(x);      // same as nearbyint with potential FE_INEXACT
    long long ll = llrintf(x);
    long l = lrintf(x);
}
```

### 18.1.6 Floating-Point Manipulation Functions

| Function (float / double) | Description |
|---|---|
| `fabsf(x)` / `fabs(x)` | Absolute value |
| `fmodf(x, y)` / `fmod(x, y)` | Floating-point remainder: x - n*y |
| `remainderf(x, y)` / `remainder(x, y)` | IEEE remainder |
| `remquof(x, y, &quo)` / `remquo(x, y, &quo)` | Remainder with partial quotient |
| `copysignf(x, y)` / `copysign(x, y)` | Value of x with sign of y |
| `nextafterf(x, y)` / `nextafter(x, y)` | Next representable value from x toward y |
| `fdimf(x, y)` / `fdim(x, y)` | Positive difference: max(x-y, 0) |
| `fmaxf(x, y)` / `fmax(x, y)` | Maximum of x and y (NaN-safe) |
| `fminf(x, y)` / `fmin(x, y)` | Minimum of x and y (NaN-safe) |
| `modff(x, &iptr)` / `modf(x, &iptr)` | Decompose into fractional and integer parts |
| `frexpf(x, &exp)` / `frexp(x, &exp)` | Decompose into mantissa and exponent (base 2) |
| `ldexpf(x, exp)` / `ldexp(x, exp)` | x * 2^exp |
| `scalbnf(x, n)` / `scalbn(x, n)` | x * FLT_RADIX^n |
| `scalblnf(x, n)` / `scalbln(x, n)` | x * FLT_RADIX^n (long n) |
| `logbf(x)` / `logb(x)` | Extract exponent (floating-point result) |
| `ilogbf(x)` / `ilogb(x)` | Extract exponent (integer result) |

```cpp
__device__ void fp_manipulation_example(float x, float y) {
    float a = fabsf(x);
    float rem = fmodf(x, y);          // remainder of x/y
    float iee = remainderf(x, y);     // IEEE remainder
    int quo;
    float rq = remquof(x, y, &quo);   // remainder + partial quotient

    float cs = copysignf(-3.0f, x);   // -3.0 with sign of x
    float na = nextafterf(x, y);      // next representable float from x toward y
    float pd = fdimf(x, y);           // max(x - y, 0)
    float mx = fmaxf(x, y);           // NaN-safe max
    float mn = fminf(x, y);           // NaN-safe min

    int exp;
    float frac = frexpf(x, &exp);     // x = frac * 2^exp, frac in [0.5, 1.0)
    float recon = ldexpf(frac, exp);  // reconstruct x

    float lg = logbf(x);             // exponent as float
    int il = ilogbf(x);              // exponent as int
}
```

### 18.1.7 Error and Gamma Functions

| Function (float / double) | Description |
|---|---|
| `erff(x)` / `erf(x)` | Error function |
| `erfcf(x)` / `erfc(x)` | Complementary error function: 1 - erf(x) |
| `lgammaf(x)` / `lgamma(x)` | Log of absolute value of gamma function |
| `tgammaf(x)` / `tgamma(x)` | Gamma function |

```cpp
__device__ void error_gamma_example(float x) {
    float e = erff(x);         // error function
    float ec = erfcf(x);       // complementary error function (more accurate for large x)
    float lg = lgammaf(x);     // log|Gamma(x)|
    float g = tgammaf(x);      // Gamma(x)
}
```

### 18.1.8 Fused Multiply-Add and Classification Functions

| Function (float / double) | Description |
|---|---|
| `fmaf(x, y, z)` / `fma(x, y, z)` | Fused multiply-add: x*y + z (single rounding) |
| `nanf(tagp)` / `nan(tagp)` | Quiet NaN |

**Classification and comparison** (return `int` or `bool`):

| Function | Description |
|---|---|
| `isnan(x)` | True if x is NaN |
| `isinf(x)` | True if x is infinite |
| `isfinite(x)` | True if x is neither NaN nor infinite |
| `signbit(x)` | True if sign bit is set (negative or -0) |
| `fpclassify(x)` | Returns FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, or FP_ZERO |
| `isnormal(x)` | True if x is normal (not zero, subnormal, infinite, or NaN) |

```cpp
__device__ void classification_example(float x) {
    // Fused multiply-add: computed as if with infinite precision, then rounded once
    float result = fmaf(3.0f, 4.0f, 5.0f); // = 17.0f exactly, no intermediate rounding

    // Classification
    if (isnan(x)) { /* handle NaN */ }
    if (isinf(x)) { /* handle infinity */ }
    if (isfinite(x)) { /* handle finite value */ }
    if (signbit(x)) { /* handle negative / -0 */ }

    int cls = fpclassify(x);
    switch (cls) {
        case FP_NAN:       break;
        case FP_INFINITE:  break;
        case FP_NORMAL:    break;
        case FP_SUBNORMAL: break;
        case FP_ZERO:      break;
    }
}
```

### 18.1.9 Floating-Point Environment

CUDA supports a subset of the C floating-point environment via `<fenv.h>`:

- `fesetround(round_mode)` / `fegetround()` -- Rounding mode control
- Rounding modes: `FE_TONEAREST`, `FE_UPWARD`, `FE_DOWNWARD`, `FE_TOWARDZERO`

**Note**: The rounding mode is a per-thread state. Changing the rounding mode affects all subsequent floating-point operations in that thread.

```cpp
#include <fenv.h>

__device__ void fenv_example(float x) {
    int old_round = fegetround();

    fesetround(FE_TOWARDZERO);
    float t = nearbyintf(x);  // truncation via rounding mode

    fesetround(FE_UPWARD);
    float u = nearbyintf(x);  // round toward +inf

    fesetround(old_round);    // restore
}
```

### 18.1.10 Support for `__half` and `__nv_bfloat16`

Many standard math functions are overloaded for `__half` (fp16) and `__nv_bfloat16` (bf16) types. These overloads are defined in `<cuda_fp16.h>` and `<cuda_bf16.h>` respectively.

**Available for `__half`**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`, `pow`, `sqrt`, `rsqrt`, `cbrt`, `rcbrt`, `hypot`, `ceil`, `floor`, `trunc`, `round`, `rint`, `fabs`, `fmod`, `remainder`, `fmax`, `fmin`, `fma`, `frexp`, `ldexp`, `modf`, `copysign`, `nextafter`, `signbit`, `isnan`, `isinf`, `isfinite`, `fpclassify`, `normcdf`, `erf`, `erfc`, `lgamma`, `tgamma`, and more.

**Available for `__nv_bfloat16`**: A similar set of overloads is available, though the exact coverage may differ slightly from `__half`.

```cpp
#include <cuda_fp16.h>

__device__ void half_math_example(__half x) {
    __half s = sin(x);        // sine
    __half c = cos(x);        // cosine
    __half e = exp(x);        // exponential
    __half l = log(x);        // natural log
    __half r = sqrt(x);       // square root
    __half rs = rsqrt(x);     // reciprocal sqrt
    __half a = fabs(x);       // absolute value
    __half m = fma(x, x, x);  // fused multiply-add
}
```

---

## 18.2 Intrinsic Functions (Faster, Less Accurate)

Intrinsic functions are available **only in device code** and trade accuracy for performance. They use the same name as the standard function but prefixed with `__`. The primary advantage is significantly reduced instruction count and latency.

### 18.2.1 IEEE-Compliant Arithmetic Intrinsics

These intrinsics provide IEEE-754-compliant operations with explicit rounding mode control. They differ from regular arithmetic in that they **prevent the compiler from merging operations into FFMA (fused multiply-add)** instructions. This is useful when strict IEEE compliance is required.

```cpp
// Add with specified rounding mode
__fadd_rn(x, y)   // round to nearest even
__fadd_rz(x, y)   // round toward zero
__fadd_ru(x, y)   // round toward +inf
__fadd_rd(x, y)   // round toward -inf

// Multiply with specified rounding mode
__fmul_rn(x, y)   // round to nearest even
__fmul_rz(x, y)   // round toward zero
__fmul_ru(x, y)   // round toward +inf
__fmul_rd(x, y)   // round toward -inf

// Fused multiply-add with specified rounding mode
__fmaf_rn(x, y, z)  // round to nearest even
__fmaf_rz(x, y, z)  // round toward zero
__fmaf_ru(x, y, z)  // round toward +inf
__fmaf_rd(x, y, z)  // round toward -inf

// Divide with specified rounding mode
__fdiv_rn(x, y)   // round to nearest even
__fdiv_rz(x, y)   // round toward zero
__fdiv_ru(x, y)   // round toward +inf
__fdiv_rd(x, y)   // round toward -inf

// Square root with specified rounding mode
__fsqrt_rn(x)     // round to nearest even
__fsqrt_rz(x)     // round toward zero
__fsqrt_ru(x)     // round toward +inf
__fsqrt_rd(x)     // round toward -inf

// Reciprocal with specified rounding mode
__frcp_rn(x)      // round to nearest even
__frcp_rz(x)      // round toward zero
__frcp_ru(x)      // round toward +inf
__frcp_rd(x)      // round toward -inf
```

**Rounding modes**:
| Suffix | Mode | Description |
|---|---|---|
| `_rn` | Round to nearest even | Default IEEE rounding mode |
| `_rz` | Round toward zero | Truncation |
| `_ru` | Round toward +infinity | Ceiling |
| `_rd` | Round toward -infinity | Floor |

**Key behavior**: All of the above are **0 ULP error** (exactly IEEE-754 compliant). The `__fadd_rn` / `__fmul_rn` intrinsics specifically prevent the compiler from merging the add or multiply into an FFMA. This is important when reproducing specific numerical results.

```cpp
__device__ float ieee_intrinsic_example(float a, float b, float c) {
    // Prevent FFMA fusion: ensures a+b is computed separately
    float sum = __fadd_rn(a, b);
    float result = __fmul_rn(sum, c);
    // Compiler cannot merge these into fma(a, b, c)

    // Compare with regular arithmetic where compiler may fuse:
    float regular_sum = a + b;
    float regular_result = regular_sum * c;
    // Compiler MAY transform this into fma(a, b, c) -- different result!

    return result;
}
```

**Double-precision variants**: `__dadd_rn`, `__dmul_rn`, `__dmaf_rn`, `__ddiv_rn`, `__dsqrt_rn`, `__drcp_rn` (and all rounding mode suffixes) are also available for `double` operands.

```cpp
__device__ void double_ieee_intrinsics(double a, double b, double c) {
    double s = __dadd_rn(a, b);       // IEEE add, no FFMA merge
    double m = __dmul_rn(a, b);       // IEEE mul, no FFMA merge
    double f = __dmaf_rn(a, b, c);    // IEEE FMA
    double d = __ddiv_rn(a, b);       // IEEE div
    double sq = __dsqrt_rn(a);        // IEEE sqrt
    double r = __drcp_rn(a);          // IEEE reciprocal
}
```

### 18.2.2 Single-Precision Approximate Intrinsics

These intrinsics are **single-precision only** and provide faster but less accurate results compared to the standard math functions.

| Intrinsic | Error | Description |
|---|---|---|
| `__fdividef(x, y)` | 2 ULP | Fast divide: x / y |
| `__expf(x)` | 2 + floor(abs(1.173 * x)) ULP | Fast exponential: e^x |
| `__exp10f(x)` | 2 + floor(abs(2.97 * x)) ULP | Fast base-10 exponential: 10^x |
| `__logf(x)` | 3 ULP (positive range) | Fast natural logarithm |
| `__log2f(x)` | 2 ULP (positive range) | Fast base-2 logarithm |
| `__log10f(x)` | 3 ULP (positive range) | Fast base-10 logarithm |
| `__sinf(x)` | 2^-21.41 absolute error in [-pi, pi] | Fast sine |
| `__cosf(x)` | 2^-21.41 absolute error in [-pi, pi] | Fast cosine |
| `__sincosf(x, s, c)` | Same as individual sin/cos | Fast sine + cosine simultaneously |
| `__tanf(x)` | Derived from __sinf * 1/__cosf | Fast tangent |
| `__powf(x, y)` | Derived from exp2f(y * __log2f(x)) | Fast power: x^y |
| `__tanhf(x)` | Max relative error: 2^-11 | Fast hyperbolic tangent |

```cpp
__device__ void approximate_intrinsics_example(float x, float y) {
    float d = __fdividef(x, y);       // fast divide, 2 ULP
    float e = __expf(x);              // fast exp, ~2 ULP
    float e10 = __exp10f(x);          // fast 10^x, ~2 ULP
    float ln = __logf(x);             // fast log, 3 ULP
    float lg = __log2f(x);            // fast log2, 2 ULP
    float l10 = __log10f(x);          // fast log10, 3 ULP

    float s, c;
    __sincosf(x, &s, &c);            // fast sin + cos simultaneously
    float t = __tanf(x);              // fast tan (derived)
    float p = __powf(x, y);           // fast pow (derived)
    float th = __tanhf(x);            // fast tanh, 2^-11 relative error
}
```

**Important notes on approximate intrinsics**:
- Input range for `__sinf`, `__cosf`, `__sincosf`: the accuracy guarantee of 2^-21.41 absolute error only holds when `x` is in the range `[-pi, pi]`. Outside this range, the error may be larger.
- `__logf(0)` returns negative infinity.
- `__logf(x)` for `x < 0` returns NaN.
- `__fdividef(0.0f, 0.0f)` returns NaN (not a hardware exception).

### 18.2.3 Half-Precision Arithmetic Intrinsics

```cpp
// __half arithmetic (defined in <cuda_fp16.h>)
__hadd(x, y)        // half add
__hsub(x, y)        // half subtract
__hmul(x, y)        // half multiply
__hfma(x, y, z)     // half FMA: x*y + z
__hdiv(x, y)        // half divide
__hneg(x)           // half negate
__habs(x)           // half absolute value

// __nv_bfloat16 arithmetic (defined in <cuda_bf16.h>)
__hadd(x, y)        // bf16 add (overloaded)
__hsub(x, y)        // bf16 subtract (overloaded)
__hmul(x, y)        // bf16 multiply (overloaded)
__hfma(x, y, z)     // bf16 FMA (overloaded)
__hdiv(x, y)        // bf16 divide (overloaded)
```

---

## 18.3 Non-Standard CUDA Math Functions

CUDA provides a number of mathematical functions not found in the C/C++ standard library. These include convenience functions, specialized functions, and approximations.

### 18.3.1 Convenience Functions

| Float / Double | Description |
|---|---|
| `exp10f(x)` / `exp10(x)` | 10^x |
| `rsqrtf(x)` / `rsqrt(x)` | 1 / sqrt(x) |
| `rcbrtf(x)` / `rcbrt(x)` | 1 / cbrt(x) |
| `rhypotf(x, y)` / `rhypot(x, y)` | 1 / hypot(x, y) = 1 / sqrt(x^2 + y^2) |

### 18.3.2 N-Dimensional Norm Functions

| Float / Double | Description |
|---|---|
| `norm3df(x, y, z)` / `norm3d(x, y, z)` | sqrt(x^2 + y^2 + z^2) |
| `norm4df(x, y, z, t)` / `norm4d(x, y, z, t)` | sqrt(x^2 + y^2 + z^2 + t^2) |
| `rnorm3df(x, y, z)` / `rnorm3d(x, y, z)` | 1 / sqrt(x^2 + y^2 + z^2) |
| `rnorm4df(x, y, z, t)` / `rnorm4d(x, y, z, t)` | 1 / sqrt(x^2 + y^2 + z^2 + t^2) |
| `normf(dim, ptr)` / `norm(dim, ptr)` | sqrt(sum of squares of dim-element array) |
| `rnormf(dim, ptr)` / `rnorm(dim, ptr)` | 1 / sqrt(sum of squares of dim-element array) |

```cpp
__device__ void norm_example(float x, float y, float z, float t) {
    // 3D vector magnitude
    float mag3 = norm3df(x, y, z);
    float inv_mag3 = rnorm3df(x, y, z);  // 1/mag3, more efficient

    // 4D vector magnitude
    float mag4 = norm4df(x, y, z, t);
    float inv_mag4 = rnorm4df(x, y, z, t);

    // General N-dimensional norm
    float vec[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float mag5 = normf(5, vec);
    float inv_mag5 = rnormf(5, vec);
}
```

### 18.3.3 Trigonometric Pi Functions

These compute trigonometric functions with the argument scaled by pi, avoiding the need for manual multiplication and improving accuracy.

| Float / Double | Description |
|---|---|
| `sinpif(x)` / `sinpi(x)` | sin(pi * x) |
| `cospif(x)` / `cospi(x)` | cos(pi * x) |
| `sincospif(x, &s, &c)` / `sincospi(x, &s, &c)` | sin(pi * x) and cos(pi * x) simultaneously |

```cpp
__device__ void pi_trig_example(float x) {
    // sin(pi * x) and cos(pi * x) -- more accurate than sinf(x * CUDART_PI_F)
    float s = sinpif(x);     // sin(pi * x)
    float c = cospif(x);     // cos(pi * x)

    float s2, c2;
    sincospif(x, &s2, &c2);  // both at once

    // Exact results at integer boundaries:
    // sinpif(0.0f) = 0.0f, sinpif(1.0f) = 0.0f
    // cospif(0.0f) = 1.0f, cospif(0.5f) = 0.0f
}
```

### 18.3.4 Cumulative Distribution and Inverse Functions

| Float / Double | Description |
|---|---|
| `normcdff(x)` / `normcdf(x)` | Standard normal CDF: Phi(x) |
| `normcdffinv(x)` / `normcdfinv(x)` | Inverse standard normal CDF: Phi^-1(x) |
| `erfcinvf(x)` / `erfcinv(x)` | Inverse complementary error function |
| `erfcxf(x)` / `erfcx(x)` | Scaled complementary error: exp(x^2) * erfc(x) |
| `erfinvf(x)` / `erfinv(x)` | Inverse error function |

```cpp
__device__ void cdf_example(float x) {
    // Standard normal CDF
    float phi = normcdff(x);           // P(Z <= x) for standard normal Z
    float quantile = normcdffinv(phi); // inverse: should return x

    // Error function and relatives
    float erf_val = erff(x);
    float erfc_val = erfcf(x);
    float erfinv_val = erfinvf(erf_val);    // should return x
    float erfcinv_val = erfcinvf(erfc_val);  // should return x
    float erfcx_val = erfcxf(x);            // exp(x^2)*erfc(x), avoids overflow
}
```

### 18.3.5 Bessel Functions

| Float / Double | Description |
|---|---|
| `j0f(x)` / `j0(x)` | Bessel function of the first kind, order 0 |
| `j1f(x)` / `j1(x)` | Bessel function of the first kind, order 1 |
| `jnf(n, x)` / `jn(n, x)` | Bessel function of the first kind, order n |
| `y0f(x)` / `y0(x)` | Bessel function of the second kind, order 0 |
| `y1f(x)` / `y1(x)` | Bessel function of the second kind, order 1 |
| `ynf(n, x)` / `yn(n, x)` | Bessel function of the second kind, order n |
| `cyl_bessel_i0f(x)` / `cyl_bessel_i0(x)` | Modified Bessel function of the first kind, order 0 |
| `cyl_bessel_i1f(x)` / `cyl_bessel_i1(x)` | Modified Bessel function of the first kind, order 1 |

```cpp
__device__ void bessel_example(float x) {
    // Bessel functions of the first kind
    float bj0 = j0f(x);           // J_0(x)
    float bj1 = j1f(x);           // J_1(x)
    float bjn = jnf(3, x);        // J_3(x)

    // Bessel functions of the second kind
    float by0 = y0f(x);           // Y_0(x)
    float by1 = y1f(x);           // Y_1(x)
    float byn = ynf(3, x);        // Y_3(x)

    // Modified Bessel functions of the first kind
    float bi0 = cyl_bessel_i0f(x); // I_0(x)
    float bi1 = cyl_bessel_i1f(x); // I_1(x)
}
```

### 18.3.6 Half-Precision Convenience Functions

These non-standard functions are specific to the `__half` and `__nv_bfloat16` types:

```cpp
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ void half_convenience_example(__half x, __nv_bfloat16 bx) {
    // __half convenience functions
    __half rcp_val = hrcp(x);              // 1/x (approximate reciprocal)
    __half exp10_val = hexp10(x);           // 10^x
    __half rsqrt_val = hrsqrt(x);           // 1/sqrt(x) (approximate)
    __half tanh_approx = htanh_approx(x);  // approximate tanh

    // __nv_bfloat16 convenience functions (same names, overloaded)
    __nv_bfloat16 brcp_val = hrcp(bx);
    __nv_bfloat16 bexp10_val = hexp10(bx);
    __nv_bfloat16 brsqrt_val = hrsqrt(bx);
}
```

---

## 18.4 `--use_fast_math` Compiler Flag

The `--use_fast_math` flag (nvcc) enables a suite of fast-math optimizations that trade accuracy for performance. It is equivalent to individually setting all of the following flags:

| Individual Flag | Effect |
|---|---|
| `--use_fast_math` | Enables all of the following |
| `--ftz=true` | Flush denormals to zero |
| `--prec-div=false` | Less precise division |
| `--prec-sqrt=false` | Less precise square root |
| `--fmad=true` | Enable FMAD contraction |

Additionally, `--use_fast_math` translates standard single-precision math function calls to their intrinsic counterparts:

| Standard Function | Replaced With | Error |
|---|---|---|
| `sinf(x)` | `__sinf(x)` | 2^-21.41 abs error in [-pi, pi] |
| `cosf(x)` | `__cosf(x)` | 2^-21.41 abs error in [-pi, pi] |
| `tanf(x)` | `__tanf(x)` | Derived from __sinf * 1/__cosf |
| `logf(x)` | `__logf(x)` | 3 ULP |
| `log2f(x)` | `__log2f(x)` | 2 ULP |
| `log10f(x)` | `__log10f(x)` | 3 ULP |
| `expf(x)` | `__expf(x)` | 2 + floor(abs(1.173*x)) ULP |
| `powf(x, y)` | `__powf(x, y)` | Derived from exp2f(y * __log2f(x)) |

**Critical**: This replacement happens transparently. If you call `sinf(x)` in your device code and compile with `--use_fast_math`, it will be silently replaced with `__sinf(x)`. Double-precision functions are **not** affected.

```cpp
// Compiled with: nvcc --use_fast_math
__device__ float fast_math_example(float x) {
    // This call is silently replaced with __sinf(x)
    float s = sinf(x);
    // This call is silently replaced with __logf(x)
    float l = logf(x);
    // This call is NOT replaced (double precision unaffected)
    double ds = sin((double)x);
    return s + l;
}
```

**Recommendation**: Use `--use_fast_math` when numerical precision is not critical (e.g., certain deep learning kernels, graphics shaders). Avoid it when bit-exact IEEE results are required. For finer control, call the intrinsic functions directly rather than using the global flag.

---

## 18.5 Half Precision Math (Vector/SIMD Operations)

CUDA provides SIMD-style arithmetic on packed half-precision values (`half2` and `__nv_bfloat162`), which processes two half values simultaneously for improved throughput.

### 18.5.1 half2 Vector Arithmetic

```cpp
#include <cuda_fp16.h>

__device__ void half2_arithmetic_example() {
    // Construct half2 from two half values
    __half a = __float2half(1.5f);
    __half b = __float2half(2.5f);
    half2 pair = __halves2half2(a, b);

    // Or from a single value (both lanes same)
    half2 uniform = __half2half2(a);

    // Vector arithmetic (operates on both lanes simultaneously)
    half2 x = __halves2half2(__float2half(3.0f), __float2half(4.0f));
    half2 y = __halves2half2(__float2half(1.0f), __float2half(2.0f));

    half2 sum  = __hadd2(x, y);    // [3+1, 4+2] = [4.0, 6.0]
    half2 diff = __hsub2(x, y);    // [3-1, 4-2] = [2.0, 2.0]
    half2 prod = __hmul2(x, y);    // [3*1, 4*2] = [3.0, 8.0]
    half2 fma2 = __hfma2(x, y, sum); // [3*1+4, 4*2+6] = [7.0, 14.0]
    half2 neg  = __hneg2(x);       // [-3.0, -4.0]
    half2 abs2 = __habs2(x);       // [3.0, 4.0]
    half2 div  = __h2div(x, y);    // [3.0/1.0, 4.0/2.0]

    // Extract individual lanes
    __half lo = __low2half(pair);    // a = 1.5
    __half hi = __high2half(pair);   // b = 2.5

    // Swizzle operations
    half2 swapped = __halves2half2(__high2half(x), __low2half(x));

    // Comparison (per-lane)
    half2 cmp_gt = __hgt2(x, y);       // per-lane > comparison
    half2 cmp_eq = __heq2(x, y);       // per-lane == comparison
    half2 cmp_lt = __hlt2(x, y);       // per-lane < comparison
}
```

### 18.5.2 __nv_bfloat162 Vector Arithmetic

```cpp
#include <cuda_bf16.h>

__device__ void bf162_arithmetic_example() {
    __nv_bfloat16 a = __float2bfloat16(1.5f);
    __nv_bfloat16 b = __float2bfloat16(2.5f);
    __nv_bfloat162 pair = __halves2bfloat162(a, b);

    __nv_bfloat162 x = __halves2bfloat162(
        __float2bfloat16(3.0f),
        __float2bfloat16(4.0f));
    __nv_bfloat162 y = __halves2bfloat162(
        __float2bfloat16(1.0f),
        __float2bfloat16(2.0f));

    __nv_bfloat162 sum  = __hadd2(x, y);   // SIMD add
    __nv_bfloat162 diff = __hsub2(x, y);   // SIMD sub
    __nv_bfloat162 prod = __hmul2(x, y);   // SIMD mul
    __nv_bfloat162 fma2 = __hfma2(x, y, sum); // SIMD FMA

    // Extract lanes
    __nv_bfloat16 lo = __low2bfloat16(pair);
    __nv_bfloat16 hi = __high2bfloat16(pair);
}
```

### 18.5.3 half2 Math Functions

Vector math functions operate on both half lanes independently:

```cpp
__device__ void half2_math_example(half2 x) {
    half2 s = h2sin(x);      // sin on both lanes
    half2 c = h2cos(x);      // cos on both lanes
    half2 e = h2exp(x);      // exp on both lanes
    half2 l = h2log(x);      // log on both lanes
    half2 sq = h2sqrt(x);    // sqrt on both lanes
    half2 rs = h2rsqrt(x);   // rsqrt on both lanes
}
```

### 18.5.4 Type Conversion Intrinsics

```cpp
__device__ void conversion_example() {
    // float <-> half
    __half h = __float2half(3.14f);
    float f = __half2float(h);

    // float2 <-> half2
    float2 fv = make_float2(1.0f, 2.0f);
    half2 hv = __float22half2_rn(fv);
    float2 fv_back = __half22float2(hv);

    // half <-> short (bit reinterpret)
    __half h_val = __float2half(1.5f);
    short s_val = __half2short(h_val);    // reinterpret bits
    __half h_back = __short2half(s_val);

    // float <-> bfloat16
    __nv_bfloat16 bf = __float2bfloat16(3.14f);
    float fb = __bfloat162float(bf);

    // float2 <-> bfloat162
    float2 fvb = make_float2(1.0f, 2.0f);
    __nv_bfloat162 hvb = __float22bfloat162_rn(fvb);
    float2 fvb_back = __bfloat1622float2(hvb);
}
```

---

## 18.6 FP128 Math (Compute Capability 10.0+)

CUDA introduced support for 128-bit floating-point arithmetic starting with compute capability 10.0 (Hopper-based successors). The `__float128` / `_Float128` type is available on Linux x86 platforms.

### 18.6.1 Type and Basic Usage

```cpp
#if defined(__CUDA_FP128_SUPPORTED__)
__device__ void fp128_example() {
    __float128 a = 1.0q;  // quad-precision literal
    __float128 b = 2.0q;

    __float128 sum = a + b;
    __float128 prod = a * b;
}
#endif
```

### 18.6.2 FP128 Math Functions

FP128 math functions are prefixed with `__nv_fp128_`:

```cpp
__device__ void fp128_math_example(__float128 x, __float128 y) {
#if defined(__CUDA_FP128_SUPPORTED__)
    // Arithmetic
    __float128 s = __nv_fp128_sqrt(x);
    __float128 r = __nv_fp128_fma(x, y, x);  // x*y + x

    // Classification
    bool is_nan = __nv_fp128_isnan(x);
    bool is_inf = __nv_fp128_isinf(x);
    bool is_unordered = __nv_fp128_isunordered(x, y);

    // Comparison
    int cmp = __nv_fp128_cmp(x, y);  // -1, 0, or 1

    // Exponential and logarithmic
    __float128 e = __nv_fp128_exp(x);
    __float128 l = __nv_fp128_log(x);

    // Power
    __float128 p = __nv_fp128_pow(x, y);

    // Trigonometric
    __float128 sin_val = __nv_fp128_sin(x);
    __float128 cos_val = __nv_fp128_cos(x);
#endif
}
```

### 18.6.3 FP128 Available Functions

| Function | Description |
|---|---|
| `__nv_fp128_exp(x)` | e^x |
| `__nv_fp128_exp2(x)` | 2^x |
| `__nv_fp128_exp10(x)` | 10^x |
| `__nv_fp128_log(x)` | Natural log |
| `__nv_fp128_log2(x)` | Base-2 log |
| `__nv_fp128_log10(x)` | Base-10 log |
| `__nv_fp128_pow(x, y)` | x^y |
| `__nv_fp128_sqrt(x)` | Square root |
| `__nv_fp128_fma(x, y, z)` | Fused multiply-add |
| `__nv_fp128_sin(x)` | Sine |
| `__nv_fp128_cos(x)` | Cosine |
| `__nv_fp128_isnan(x)` | NaN test |
| `__nv_fp128_isinf(x)` | Infinity test |
| `__nv_fp128_isunordered(x, y)` | Unordered test (either is NaN) |
| `__nv_fp128_cmp(x, y)` | Three-way comparison |
| `__nv_fp128_fabs(x)` | Absolute value |
| `__nv_fp128_fmax(x, y)` | Maximum (NaN-safe) |
| `__nv_fp128_fmin(x, y)` | Minimum (NaN-safe) |
| `__nv_fp128_copysign(x, y)` | Copysign |

### 18.6.4 Host-Side FP128

On the host side, `_Float128` is a GCC/Clang extension. CUDA provides host-side support for converting and operating on 128-bit floats:

```cpp
// Host-side conversion
__float128 h2d = (__float128)3.14159265358979323846264338327950288q;
double d = (double)h2d;  // loses precision
```

**Note**: FP128 operations are significantly slower than FP64 or FP32 operations. They are implemented in software and should be used only when the extra precision is truly needed (e.g., intermediate accumulation in numerical algorithms).

---

## 18.7 Quick Reference: Accuracy Summary

| Category | Typical Error | Speed |
|---|---|---|
| Standard functions (float) | 0-2 ULP | Baseline |
| Standard functions (double) | 0-1 ULP | Baseline |
| IEEE-compliant intrinsics (`__fadd_rn`, etc.) | 0 ULP | Same or slower (prevents optimization) |
| Approximate intrinsics (`__sinf`, `__logf`, etc.) | 2-3 ULP or 2^-21 abs | Faster |
| `--use_fast_math` replacements | Same as approximate intrinsics | Faster |
| half / half2 functions | Implementation-defined | Fastest (SIMD) |
| FP128 functions | Quad precision | Slowest (software) |

## 18.8 Header Files

| Header | Contents |
|---|---|
| `<math.h>` | Standard math functions (host + device) |
| `<cuda_fp16.h>` | `__half`, `half2` types and intrinsics |
| `<cuda_bf16.h>` | `__nv_bfloat16`, `__nv_bfloat162` types and intrinsics |
| `<cuda_runtime_api.h>` | `CUDART_PI_F`, `CUDART_PI`, `CUDART_INF_F`, etc. |
| `<fenv.h>` | Floating-point environment (rounding modes) |

### Useful Constants

```cpp
CUDART_PI_F          // 3.14159265358979323846264338327950288f
CUDART_PI            // 3.14159265358979323846264338327950288
CUDART_INF_F         // positive infinity (float)
CUDART_INF           // positive infinity (double)
CUDART_NAN_F         // quiet NaN (float)
CUDART_NAN           // quiet NaN (double)
CUDART_MIN_DENORM_F  // minimum denormalized float
CUDART_MAX_NORMAL_F  // maximum normalized float
CUDART_NEG_ZERO_F    // -0.0f
CUDART_E_F           // 2.71828182845904523536028747135266250
CUDART_LN2_F         // 0.693147180559945309417232121458176568
CUDART_LN10_F        // 2.30258509299404568401799145468436421
CUDART_LOG2E_F       // 1.44269504088896340735992468100189214
CUDART_LOG10E_F      // 0.434294481903251827651128918916605082
CUDART_SQRT2_F       // 1.41421356237309504880168872420969808
CUDART_SQRT1_2_F     // 0.707106781186547524400844362104849039
```
