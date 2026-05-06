# CUTLASS Functional Operations - Chapter 35: Math Function Objects, Activation Functions, and Operators

This reference covers all functional operations provided by CUTLASS, including arithmetic operators, math functions, comparisons, fused operations, bitwise operations, complex operations, atomic operations, and activation functions.

---

## 35.1 Overview

CUTLASS provides a comprehensive set of function objects (functors) for element-wise operations used in epilogue fusion, element-wise kernel operations, and reduction operations. These functors are designed to work efficiently on both scalar and vectorized (array) types and are heavily used in epilogue fusion patterns.

All functional operations are defined in `include/cutlass/functional.h` and related headers. They follow a consistent interface:

```cpp
// Every functional operation is a struct with:
struct Operation {
    // The result type
    using result_type = ...;

    // Operator() for scalar types
    CUTLASS_HOST_DEVICE
    result_type operator()(args...) const;

    // May have Array specializations for vectorized execution
};
```

---

## 35.2 Arithmetic Operations

### 35.2.1 plus

Computes the sum of two elements. This is the default reduction operation.

```cpp
template <typename T>
struct plus {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs + rhs;
    }
};

// Usage in epilogue scaling:
// D = alpha * AB + beta * C
// The final addition uses plus<T>()
```

### 35.2.2 minus

Computes the difference of two elements.

```cpp
template <typename T>
struct minus {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs - rhs;
    }
};
```

### 35.2.3 multiplies

Computes the product of two elements. Used extensively in GEMM scaling.

```cpp
template <typename T>
struct multiplies {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs * rhs;
    }
};

// Usage in epilogue:
// alpha * partial_sum uses multiplies
```

### 35.2.4 divides

Computes the quotient of two elements.

```cpp
template <typename T>
struct divides {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs / rhs;
    }
};
```

### 35.2.5 negate

Computes the arithmetic negation (unary).

```cpp
template <typename T>
struct negate {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &operand) const {
        return -operand;
    }
};

// Usage: flipping the sign of GEMM output
using NegateOp = cutlass::negate<float>;
```

---

## 35.3 Math Operations

### 35.3.1 square

Computes the square of an element.

```cpp
template <typename T>
struct square {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &scalar) const {
        return scalar * scalar;
    }
};
```

### 35.3.2 magnitude_squared

Computes the squared magnitude. For real types, this is equivalent to square. For complex types, it computes `|z|^2 = re(z)^2 + im(z)^2`.

```cpp
template <typename T>
struct magnitude_squared {
    using result_type = typename MagnitudeType<T>::type;

    CUTLASS_HOST_DEVICE
    result_type operator()(T const &scalar) const {
        // For real types: scalar * scalar
        // For complex<T>: real^2 + imag^2
        return detail::MagnitudeSquared<T>()(scalar);
    }
};

// Usage: computing squared norm for normalization
using MagSq = cutlass::magnitude_squared<cutlass::complex<float>>;
// For complex(3.0f, 4.0f): returns 25.0f
```

### 35.3.3 inverse_square_root

Computes `1 / sqrt(x)`.

```cpp
template <typename T>
struct inverse_square_root {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &scalar) const {
        return T(1) / sqrt(scalar);
    }
};
```

### 35.3.4 reciprocal_approximate

Computes an approximate reciprocal `1 / x`. On GPU hardware, this may use special approximation instructions for speed.

```cpp
template <typename T>
struct reciprocal_approximate {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &scalar) const {
        // Uses hardware approximate reciprocal when available
        // For float: uses MUFU.RCP instruction
        return T(1) / scalar;
    }
};

// FP16 specialization uses half2 approximation for speed:
template <>
struct reciprocal_approximate<cutlass::half_t> {
    using result_type = cutlass::half_t;

    CUTLASS_HOST_DEVICE
    cutlass::half_t operator()(cutlass::half_t const &scalar) const {
        return cutlass::half_t(1.0f) / scalar;
    }
};
```

---

## 35.4 Comparison Operations

### 35.4.1 greater

Returns true if the left operand is greater than the right.

```cpp
template <typename T>
struct greater {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &lhs, T const &rhs) const {
        return lhs > rhs;
    }
};
```

### 35.4.2 less

Returns true if the left operand is less than the right.

```cpp
template <typename T>
struct less {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &lhs, T const &rhs) const {
        return lhs < rhs;
    }
};
```

### 35.4.3 greater_equal

```cpp
template <typename T>
struct greater_equal {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &lhs, T const &rhs) const {
        return lhs >= rhs;
    }
};
```

### 35.4.4 less_equal

```cpp
template <typename T>
struct less_equal {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &lhs, T const &rhs) const {
        return lhs <= rhs;
    }
};
```

### 35.4.5 maximum

Returns the larger of two elements. For floating-point types, if either operand is NaN, behavior follows the hardware convention.

```cpp
template <typename T>
struct maximum {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return (lhs < rhs) ? rhs : lhs;
    }
};

// NaN-aware specialization for floating-point types:
template <>
struct maximum<float> {
    using result_type = float;

    CUTLASS_HOST_DEVICE
    float operator()(float const &lhs, float const &rhs) const {
        return fmax(lhs, rhs);  // NaN-aware: returns non-NaN if one is NaN
    }
};

// Usage in reduction: compute max of a vector
using MaxOp = cutlass::maximum<float>;
```

### 35.4.6 minimum

Returns the smaller of two elements.

```cpp
template <typename T>
struct minimum {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return (rhs < lhs) ? rhs : lhs;
    }
};

// Usage: clamping values to a range
float clamped = cutlass::minimum<float>()(
    cutlass::maximum<float>()(value, lower_bound),
    upper_bound
);
```

---

## 35.5 Fused Operations

Fused operations combine multiple arithmetic operations into a single functor for efficiency.

### 35.5.1 multiply_add

Computes `a * b + c`. This is the fundamental operation of GEMM and is the default accumulator operation.

```cpp
template <typename A, typename B = A, typename C = A>
struct multiply_add {
    using result_type = C;

    CUTLASS_HOST_DEVICE
    C operator()(A const &a, B const &b, C const &c) const {
        return C(a * b + c);
    }
};

// Mixed-precision multiply_add:
// A=half_t, B=half_t, C=float
// Computes: float(half * half) + float
using FusedOp = cutlass::multiply_add<cutlass::half_t, cutlass::half_t, float>;
// This is the core operation in mixed-precision GEMM:
// accumulator = alpha * element_A * element_B + accumulator
```

### 35.5.2 multiply_add_relu0

Computes `max(0, a * b + c)`. This fuses the ReLU activation with the multiply-add.

```cpp
template <typename A, typename B = A, typename C = A>
struct multiply_add_relu0 {
    using result_type = C;

    CUTLASS_HOST_DEVICE
    C operator()(A const &a, B const &b, C const &c) const {
        C result = C(a * b + c);
        return (result > C(0)) ? result : C(0);
    }
};

// Usage: GEMM with fused ReLU
// D = max(0, alpha * A * B + beta * C)
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput, 4, ElementAccumulator, ElementCompute
>;
```

### 35.5.3 guarded_multiply_add

Computes `a * b + c` but with guard for overflow. Useful for integer types where overflow is a concern.

```cpp
template <typename T>
struct guarded_multiply_add {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &a, T const &b, T const &c) const {
        T product = a * b;
        // Guard: if overflow would occur, saturate
        T result = product + c;
        // Check for overflow
        if ((product > 0) && (c > 0) && (result < 0)) return T(INT_MAX);
        if ((product < 0) && (c < 0) && (result > 0)) return T(INT_MIN);
        return result;
    }
};
```

---

## 35.6 Population Count Operations

These operations compute a weighted population count of the bits of the product, useful for binary and sub-byte operations.

### 35.6.1 and_popc_add

Computes `popcount(a & b) + c`. Used in binary GEMM (1-bit weights).

```cpp
template <typename A, typename B = A, typename C = A>
struct and_popc_add {
    using result_type = C;

    CUTLASS_HOST_DEVICE
    C operator()(A const &a, B const &b, C const &c) const {
        return C(__popc(a & b)) + c;
    }
};
```

### 35.6.2 xor_popc_add

Computes `popcount(a ^ b) + c`. Used in XNOR-based binary neural networks.

```cpp
template <typename A, typename B = A, typename C = A>
struct xor_popc_add {
    using result_type = C;

    CUTLASS_HOST_DEVICE
    C operator()(A const &a, B const &b, C const &c) const {
        return C(__popc(a ^ b)) + c;
    }
};
```

### 35.6.3 or_popc_add

Computes `popcount(a | b) + c`.

```cpp
template <typename A, typename B = A, typename C = A>
struct or_popc_add {
    using result_type = C;

    CUTLASS_HOST_DEVICE
    C operator()(A const &a, B const &b, C const &c) const {
        return C(__popc(a | b)) + c;
    }
};
```

---

## 35.7 Logical Operations

### 35.7.1 logical_and

```cpp
template <typename T>
struct logical_and {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &lhs, T const &rhs) const {
        return bool(lhs) && bool(rhs);
    }
};
```

### 35.7.2 logical_or

```cpp
template <typename T>
struct logical_or {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &lhs, T const &rhs) const {
        return bool(lhs) || bool(rhs);
    }
};
```

### 35.7.3 logical_not

```cpp
template <typename T>
struct logical_not {
    using result_type = bool;

    CUTLASS_HOST_DEVICE
    bool operator()(T const &operand) const {
        return !bool(operand);
    }
};
```

---

## 35.8 Bitwise Operations

### 35.8.1 bit_and

```cpp
template <typename T>
struct bit_and {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs & rhs;
    }
};
```

### 35.8.2 bit_or

```cpp
template <typename T>
struct bit_or {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs | rhs;
    }
};
```

### 35.8.3 bit_not

```cpp
template <typename T>
struct bit_not {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &operand) const {
        return ~operand;
    }
};
```

### 35.8.4 bit_xor

```cpp
template <typename T>
struct bit_xor {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &lhs, T const &rhs) const {
        return lhs ^ rhs;
    }
};
```

---

## 35.9 Complex Operations

### 35.9.1 conjugate

Computes the complex conjugate. For real types, this is identity. For complex types, it negates the imaginary part.

```cpp
template <typename T>
struct conjugate {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return value;  // Identity for real types
    }
};

// Complex specialization
template <typename T>
struct conjugate<cutlass::complex<T>> {
    using result_type = cutlass::complex<T>;

    CUTLASS_HOST_DEVICE
    cutlass::complex<T> operator()(cutlass::complex<T> const &value) const {
        return cutlass::complex<T>(value.real(), -value.imag());
    }
};

// Usage: Hermitian matrix operations require conjugate transpose
// A^H = conjugate(A^T)
using ConjOp = cutlass::conjugate<cutlass::complex<float>>;
```

---

## 35.10 Atomic Operations

### 35.10.1 atomic_add

Performs an atomic addition. Used in reduction operations and when multiple threads write to the same output location.

```cpp
template <typename T>
struct atomic_add {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T *ptr, T const &value) const {
        return atomicAdd(ptr, value);
    }
};

// FP16 specialization uses half2 for efficiency:
template <>
struct atomic_add<cutlass::half_t> {
    using result_type = cutlass::half_t;

    CUTLASS_HOST_DEVICE
    cutlass::half_t operator()(cutlass::half_t *ptr, cutlass::half_t const &value) const {
        return atomicAdd(ptr, value);
    }
};

// Usage in split-K reduction:
// Each split accumulates partial results, then atomically adds to final output
using AtomicAdd = cutlass::atomic_add<float>;
```

### 35.10.2 atomic_maximum

Performs an atomic maximum operation.

```cpp
template <typename T>
struct atomic_maximum {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T *ptr, T const &value) const {
        T old = *ptr;
        while (value > old) {
            T assumed = old;
            old = atomicCAS(ptr, assumed, value);
            if (assumed == old) break;
        }
        return old;
    }
};

// Usage: computing max reduction across threadblocks
using AtomicMax = cutlass::atomic_maximum<float>;
```

---

## 35.11 Activation Functions

Activation functions are defined in `include/cutlass/epilogue/thread/activation.h`. They are used primarily in epilogue fusion patterns.

### 35.11.1 Identity

Pass-through activation. Returns the input unchanged.

```cpp
template <typename T>
struct Identity {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return value;
    }

    // With derivative
    CUTLASS_HOST_DEVICE
    T operator()(T const &value, T const &gradient) const {
        return gradient;  // d/dx(x) = 1
    }
};

// Usage in epilogue: no activation applied
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 4, ElementAccumulator, ElementCompute,
    cutlass::epilogue::thread::Identity<ElementCompute>
>;
```

### 35.11.2 Scale

Multiplies input by a constant scale factor.

```cpp
template <typename T>
struct Scale {
    using result_type = T;

    T scale_factor;

    CUTLASS_HOST_DEVICE
    Scale(T scale_factor_ = T(1)) : scale_factor(scale_factor_) {}

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return value * scale_factor;
    }
};
```

### 35.11.3 ReLU (Rectified Linear Unit)

Computes `max(0, x)`. The most common activation function in deep learning.

```cpp
template <typename T>
struct ReLu {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return (value > T(0)) ? value : T(0);
    }

    // With threshold variant
    CUTLASS_HOST_DEVICE
    T operator()(T const &value, T const &threshold) const {
        return (value > threshold) ? value : T(0);
    }
};

// Usage in fused GEMM+ReLU:
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput, 4, ElementAccumulator, ElementCompute
>;

// Direct usage:
cutlass::epilogue::thread::ReLu<float> relu;
float result = relu(-1.5f);  // Returns 0.0f
float result2 = relu(2.3f);  // Returns 2.3f
```

### 35.11.4 Clamp

Clamps the value to a specified range `[lower, upper]`.

```cpp
template <typename T>
struct Clamp {
    using result_type = T;

    T lower_bound;
    T upper_bound;

    CUTLASS_HOST_DEVICE
    Clamp(T lower = T(0), T upper = T(1)) : lower_bound(lower), upper_bound(upper) {}

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return cutlass::maximum<T>()(
            cutlass::minimum<T>()(value, upper_bound),
            lower_bound
        );
    }
};

// Usage: clamp output to [0, 6] for ReLU6 activation
using ClampOp = cutlass::epilogue::thread::Clamp<float>;
ClampOp clamp_op(0.0f, 6.0f);
float result = clamp_op(8.0f);  // Returns 6.0f
```

### 35.11.5 LeakyReLU

Computes `x > 0 ? x : alpha * x` where alpha is a small positive constant.

```cpp
template <typename T>
struct LeakyReLU {
    using result_type = T;

    T alpha;  // Leak coefficient, typically 0.01 or 0.1

    CUTLASS_HOST_DEVICE
    LeakyReLU(T alpha_ = T(0.01)) : alpha(alpha_) {}

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return (value > T(0)) ? value : alpha * value;
    }
};

// Usage:
using LeakyReluOp = cutlass::epilogue::thread::LeakyReLU<float>;
LeakyReluOp op(0.1f);
float result = op(-2.0f);  // Returns -0.2f
float result2 = op(3.0f);  // Returns 3.0f
```

### 35.11.6 ThresholdReLU

Computes `x > threshold ? x : 0`. A generalization of ReLU with a configurable threshold.

```cpp
template <typename T>
struct ThresholdReLU {
    using result_type = T;

    T threshold;

    CUTLASS_HOST_DEVICE
    ThresholdReLU(T threshold_ = T(0)) : threshold(threshold_) {}

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return (value > threshold) ? value : T(0);
    }
};
```

### 35.11.7 Tanh

Computes the hyperbolic tangent activation.

```cpp
template <typename T>
struct Tanh {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return cutlass::tanh(value);
    }
};

// For FP16:
template <>
struct Tanh<cutlass::half_t> {
    using result_type = cutlass::half_t;

    CUTLASS_HOST_DEVICE
    cutlass::half_t operator()(cutlass::half_t const &value) const {
        return cutlass::half_t(tanhf(float(value)));
    }
};
```

### 35.11.8 Sigmoid

Computes `1 / (1 + exp(-x))`.

```cpp
template <typename T>
struct Sigmoid {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return T(1) / (T(1) + cutlass::exp(-value));
    }
};

// Usage: common in attention mechanisms and gating
using SigmoidOp = cutlass::epilogue::thread::Sigmoid<float>;
float result = SigmoidOp()(0.0f);  // Returns 0.5f
```

### 35.11.9 SiLU (Sigmoid Linear Unit / Swish)

Computes `x * sigmoid(x)`. Also known as Swish activation.

```cpp
template <typename T>
struct SiLU {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return value * (T(1) / (T(1) + cutlass::exp(-value)));
    }
};

// Usage: used in LLaMA and other transformer models
using SiluOp = cutlass::epilogue::thread::SiLU<float>;
```

### 35.11.10 HardSwish

Computes `x * clamp(x + 3, 0, 6) / 6`. Used in MobileNet V3.

```cpp
template <typename T>
struct HardSwish {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        T v = value + T(3);
        T clamped = cutlass::maximum<T>()(
            cutlass::minimum<T>()(v, T(6)),
            T(0)
        );
        return value * clamped / T(6);
    }
};
```

### 35.11.11 GELU (Gaussian Error Linear Unit)

Computes `x * Phi(x)` where `Phi` is the standard Gaussian CDF. Uses the exact formula with `erf`.

```cpp
template <typename T>
struct GELU {
    using result_type = T;

    static constexpr T kSqrtHalf = T(0.7071067811865475);  // 1 / sqrt(2)

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return value * T(0.5) * (T(1) + cutlass::erf(value * kSqrtHalf));
    }
};

// Usage: standard GELU activation used in BERT, GPT, and many transformers
using GeluOp = cutlass::epilogue::thread::GELU<float>;
```

### 35.11.12 GELU_taylor

Approximates GELU using a Taylor series expansion: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. This is the approximation used in most transformer implementations.

```cpp
template <typename T>
struct GELU_taylor {
    using result_type = T;

    static constexpr T kSqrt2OverPi = T(0.7978845608028654);  // sqrt(2/pi)
    static constexpr T kCoeff       = T(0.044715);

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        T inner = kSqrt2OverPi * (value + kCoeff * value * value * value);
        return T(0.5) * value * (T(1) + cutlass::tanh(inner));
    }
};

// Usage: GELU approximation used in GPT-2, GPT-3, etc.
using GeluTaylorOp = cutlass::epilogue::thread::GELU_taylor<float>;
```

### 35.11.13 dReLU (Derivative of ReLU)

Computes the derivative of ReLU: `x > 0 ? 1 : 0`. Used in backward pass computation.

```cpp
template <typename T>
struct dReLU {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        return (value > T(0)) ? T(1) : T(0);
    }
};
```

### 35.11.14 dReLU_Z

Computes the derivative of ReLU with a conditional: `z > 0 ? dy : 0`. Used in fused backward pass kernels where `z` is the pre-activation value and `dy` is the upstream gradient.

```cpp
template <typename T>
struct dReLU_Z {
    using result_type = T;

    CUTLASS_HOST_DEVICE
    T operator()(T const &z, T const &dy) const {
        return (z > T(0)) ? dy : T(0);
    }
};
```

### 35.11.15 dGELU (Derivative of GELU)

Computes the derivative of GELU for the backward pass.

```cpp
template <typename T>
struct dGELU {
    using result_type = T;

    static constexpr T kSqrtHalf = T(0.7071067811865475);

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        T cdf = T(0.5) * (T(1) + cutlass::erf(value * kSqrtHalf));
        T pdf = T(0.3989422804014327) * cutlass::exp(-T(0.5) * value * value);
        return cdf + value * pdf;
    }

    CUTLASS_HOST_DEVICE
    T operator()(T const &value, T const &gradient) const {
        return gradient * (*this)(value);  // Chain rule: dGELU * upstream_gradient
    }
};
```

### 35.11.16 ElementwiseFilter

A generic element-wise filter that applies a predicate function element-by-element.

```cpp
template <typename Predicate>
struct ElementwiseFilter {
    using result_type = bool;

    Predicate predicate;

    CUTLASS_HOST_DEVICE
    ElementwiseFilter(Predicate pred = Predicate()) : predicate(pred) {}

    template <typename T>
    CUTLASS_HOST_DEVICE
    bool operator()(T const &value) const {
        return predicate(value);
    }
};

// Usage: filter elements greater than threshold
using ThresholdFilter = cutlass::epilogue::thread::ElementwiseFilter<
    cutlass::greater<float>
>;
```

---

## 35.12 Heavy Operation Flag

Some operations are marked as "heavy" using the `kIsHeavy` static constant. Heavy operations require significant computation and may affect scheduling decisions.

```cpp
// Operations with kIsHeavy = true:
// - GELU (uses erf)
// - GELU_taylor (uses tanh)
// - Tanh
// - Sigmoid (uses exp)
// - SiLU (uses exp)
// - HardSwish
// - inverse_square_root

// Operations with kIsHeavy = false (default):
// - Identity, Scale, ReLu, Clamp
// - All arithmetic operations
// - All bitwise and logical operations

template <typename T>
struct GELU {
    static constexpr bool kIsHeavy = true;  // Heavy: uses erf()
    // ...
};

template <typename T>
struct ReLu {
    static constexpr bool kIsHeavy = false;  // Light: simple comparison
    // ...
};

// Usage: epilogue fusion may avoid fusing heavy operations
// if register pressure is too high
```

---

## 35.13 Array Specializations

Many functional operations have optimized Array specializations that process multiple elements simultaneously using SIMD instructions.

```cpp
// Scalar multiply_add:
multiply_add<float> op;
float result = op(a, b, c);  // Single element

// Array specialization: processes 4 floats simultaneously
// This is defined as an explicit specialization:
template <>
struct multiply_add<float, float, float> {
    // Scalar version
    CUTLASS_HOST_DEVICE
    float operator()(float const &a, float const &b, float const &c) const {
        return a * b + c;
    }
};

// The Array specialization is automatically invoked when using
// Array<float, 4> as the type:
using ArrayOp = cutlass::multiply_add<float, float, float>;
cutlass::Array<float, 4> a, b, c;
cutlass::Array<float, 4> result;
// result = a * b + c;  // Processes all 4 elements

// Half-precision Array specializations use half2 instructions:
// Array<half_t, 2> uses __hfma2 (2-element fused multiply-add)
// Array<half_t, 4> processes two half2 pairs

// FP16 multiply_add with float accumulator:
template <>
struct multiply_add<cutlass::half_t, cutlass::half_t, float> {
    using result_type = float;

    CUTLASS_HOST_DEVICE
    float operator()(cutlass::half_t const &a, cutlass::half_t const &b,
                     float const &c) const {
        return float(a) * float(b) + c;
    }
};

// Array specialization for FP16 x FP16 -> float:
template <>
struct multiply_add<cutlass::half_t, cutlass::half_t, float> {
    // ... uses __hfma2 for pairs of half_t
};
```

---

## 35.14 Code Examples

### 35.14.1 Custom Epilogue with ReLU Activation

```cpp
#include "cutlass/epilogue/thread/linear_combination_relu.h"

// Fused GEMM + ReLU epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,                   // Output type
    8,                                 // Elements per access
    float,                             // Accumulator type
    float,                             // Compute type (for alpha/beta)
    cutlass::epilogue::thread::ReLu<float>  // Activation
>;

// This computes:
// D = max(0, alpha * A * B + beta * C)
```

### 35.14.2 Custom Activation Function

```cpp
// Define a custom activation: Mish(x) = x * tanh(softplus(x))
template <typename T>
struct Mish {
    using result_type = T;
    static constexpr bool kIsHeavy = true;

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        T softplus = cutlass::log(T(1) + cutlass::exp(value));
        return value * cutlass::tanh(softplus);
    }

    CUTLASS_HOST_DEVICE
    T operator()(T const &value, T const &gradient) const {
        // Derivative of Mish (simplified)
        T sp = cutlass::log(T(1) + cutlass::exp(value));
        T tsp = cutlass::tanh(sp);
        T tsp_grad = T(1) - tsp * tsp;
        T sp_grad = T(1) / (T(1) + cutlass::exp(-value));
        return gradient * (tsp + value * tsp_grad * sp_grad);
    }
};

// Use in a custom epilogue:
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 8, float, float,
    Mish<float>  // Custom activation
>;
```

### 35.14.3 Using Functional Operations in Reductions

```cpp
#include "cutlass/reduction/thread/reduce.h"

// Sum reduction using plus
using ReduceOp = cutlass::reduction::thread::Reduce<
    cutlass::plus<float>,
    cutlass::Array<float, 4>
>;

// Max reduction using maximum
using MaxReduceOp = cutlass::reduction::thread::Reduce<
    cutlass::maximum<float>,
    cutlass::Array<float, 4>
>;

// In kernel code:
__global__ void reduce_kernel(float const *input, float *output, int N) {
    float thread_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        thread_sum = cutlass::plus<float>()(thread_sum, input[i]);
    }
    // Warp-level reduction using CUTLASS primitives
    // ...
}
```

### 35.14.4 Composing Operations

```cpp
// Compose multiple operations: Scale -> ReLU -> Clamp
template <typename T>
struct ScaledReLUClamp {
    T scale;
    T lower;
    T upper;

    CUTLASS_HOST_DEVICE
    ScaledReLUClamp(T s = T(1), T lo = T(0), T hi = T(1))
        : scale(s), lower(lo), upper(hi) {}

    CUTLASS_HOST_DEVICE
    T operator()(T const &value) const {
        T scaled = value * scale;
        T relu = (scaled > T(0)) ? scaled : T(0);
        return cutlass::maximum<T>()(
            cutlass::minimum<T>()(relu, upper),
            lower
        );
    }
};
```

---

## 35.15 Summary Table

| Category | Operations |
|---|---|
| Arithmetic | `plus`, `minus`, `multiplies`, `divides`, `negate` |
| Math | `square`, `magnitude_squared`, `inverse_square_root`, `reciprocal_approximate` |
| Comparison | `greater`, `less`, `greater_equal`, `less_equal`, `maximum`, `minimum` |
| Fused | `multiply_add`, `multiply_add_relu0`, `guarded_multiply_add` |
| Popcount | `and_popc_add`, `xor_popc_add`, `or_popc_add` |
| Logical | `logical_and`, `logical_or`, `logical_not` |
| Bitwise | `bit_and`, `bit_or`, `bit_not`, `bit_xor` |
| Complex | `conjugate` |
| Atomic | `atomic_add`, `atomic_maximum` |
| Activation | `Identity`, `Scale`, `ReLu`, `Clamp`, `LeakyReLU`, `ThresholdReLU` |
| Activation (cont.) | `Tanh`, `Sigmoid`, `SiLU`, `HardSwish` |
| Activation (cont.) | `GELU`, `GELU_taylor`, `dReLU`, `dReLU_Z`, `dGELU` |
| Filter | `ElementwiseFilter` |
