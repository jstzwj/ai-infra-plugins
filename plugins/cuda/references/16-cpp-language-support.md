# 16. C++ Language Support

This document covers CUDA C++ language support, including supported C++ standards, language features available in device code, the CUDA C++ Standard Library (libcu++), C standard library functions available on the device, lambda expressions, and important restrictions.

---

## Table of Contents

1. [Supported C++ Standards](#161-supported-c-standards)
2. [C++11 Features](#162-c11-features)
3. [C++14 Features](#163-c14-features)
4. [C++17 Features](#164-c17-features)
5. [C++20 Features](#165-c20-features)
6. [CUDA C++ Standard Library (libcu++)](#166-cuda-c-standard-library-libcu)
7. [C Standard Library Functions](#167-c-standard-library-functions)
8. [Lambda Expressions](#168-lambda-expressions)
9. [Restrictions](#169-restrictions)

---

## 16.1 Supported C++ Standards

CUDA supports compiling host and device code with different C++ standard levels. The standard is set via the `--std` flag to `nvcc`:

```bash
# Compile with specific C++ standard
nvcc --std=c++17 my_program.cu
nvcc --std=c++20 my_program.cu
```

### Supported Standards

| Flag | Standard | Default (CUDA 12.x) | Notes |
|---|---|---|---|
| `--std=c++03` | C++03 | No | Legacy support |
| `--std=c++11` | C++11 | No | Minimum for many CUDA features |
| `--std=c++14` | C++14 | Yes (CUDA 11.x) | Previous default |
| `--std=c++17` | C++17 | Yes (CUDA 12.x) | Current default |
| `--std=c++20` | C++20 | No | Latest supported |

### Compilation Modes

```bash
# Separate host and device compilation flags
nvcc --std=c++20 -x cu my_program.cu          # Both host and device
nvcc --std=c++17 -x cu -std=c++20 my_program.cu  # Device C++17, Host C++20

# Using -std for host compiler directly
nvcc -std=c++17 -Xcompiler -std=c++17 my_program.cu
```

---

## 16.2 C++11 Features

CUDA provides extensive C++11 support in device code. Below is a comprehensive list of supported features:

### Core Language Features

| Feature | Example | Notes |
|---|---|---|
| **Rvalue references** | `T&& x = std::move(y);` | Move semantics in device code |
| **Variadic templates** | `template<typename... Ts>` | Full parameter pack support |
| **Initializer lists** | `int arr[] = {1, 2, 3};` | Including nested types |
| **Static assertions** | `static_assert(N > 0, "N must be positive");` | Compile-time checks |
| **Auto** | `auto x = 42;` | Type deduction |
| **Trailing return types** | `auto f() -> int;` | |
| **Lambda expressions** | `[=](int x) { return x + 1; }` | See Section 16.8 |
| **decltype** | `decltype(x) y = x;` | Type inspection |
| **Right angle brackets** | `vector<vector<int>>` | No space needed between `>>` |
| **Defaulted functions** | `MyClass() = default;` | |
| **Deleted functions** | `MyClass(const MyClass&) = delete;` | |
| **Delegating constructors** | `MyClass() : MyClass(0) {}` | |
| **Explicit conversion operators** | `explicit operator bool() const;` | |
| **Null pointer literal** | `nullptr` | Replaces `NULL` and `0` |
| **Strongly-typed enums** | `enum class Color { Red, Green };` | Scoped enumerations |
| **Forward declared enums** | `enum E : int;` | |
| **Alias templates** | `template<typename T> using Vec = vector<T>;` | |
| **Unrestricted unions** | Unions with non-trivial members | |
| **Range-based for** | `for (auto& x : container)` | |
| **Extended friend declarations** | `friend T;` | |
| **Extended sizeof** | `sizeof... (Ts)` | For parameter packs |
| **Inline namespaces** | `inline namespace V1 {}` | |
| **Thread-local storage** | `__thread__ int x;` | Per-thread device storage |
| **Non-static data member initializers** | `int x = 0;` in class body | |
| **Constexpr** | `constexpr int f() { return 42; }` | Limited in device code (expanded in C++14) |

### Usage Examples in Device Code

```cpp
// Rvalue references and move semantics
__device__ void moveExample() {
    int* ptr = new int[100];
    int* other = std::move(ptr);  // ptr is now nullptr
    delete[] other;
}

// Variadic templates
template<typename... Args>
__device__ int sum(Args... args) {
    return (... + args);  // C++17 fold expression, but variadic works in C++11
}

// C++11-style variadic recursion
template<typename T>
__device__ T sum11(T value) { return value; }

template<typename T, typename... Args>
__device__ T sum11(T first, Args... rest) {
    return first + sum11(rest...);
}

// Static assertions with type traits
template<typename T>
__device__ void process(T value) {
    static_assert(__is_integral(T), "T must be integral");
    // ...
}

// Strongly-typed enums
enum class ThreadRole : int {
    Producer = 0,
    Consumer = 1
};

__device__ void enumExample() {
    ThreadRole role = ThreadRole::Producer;
    // if (role == 0) -- Error: no implicit conversion
    if (role == ThreadRole::Producer) { /* ok */ }
}

// Alias templates
template<typename T>
using DeviceVector = T*;  // Simplified device vector

// Range-based for with arrays
__device__ void rangeFor() {
    int arr[] = {1, 2, 3, 4, 5};
    for (auto& x : arr) {
        x *= 2;
    }
}

// decltype
__device__ auto add(int a, float b) -> decltype(a + b) {
    return a + b;  // returns float
}
```

---

## 16.3 C++14 Features

### Core Language Features

| Feature | Example | Notes |
|---|---|---|
| **Binary literals** | `0b101010` | `0B101010` also valid |
| **Digit separators** | `1'000'000` | Single quotes as separators |
| **Generic lambdas** | `[](auto x) { return x + 1; }` | Auto in lambda parameters |
| **Lambda init captures** | `[ptr = std::move(p)]()` | Move into lambda |
| **Variable templates** | `template<typename T> constexpr T pi = T(3.14159);` | |
| **Extended constexpr** | `constexpr` with loops, local variables | |
| **Relaxed constexpr** | `constexpr` functions with if/else, local vars | |
| **Return type deduction** | `auto f() { return 42; }` | For normal functions |
| **Aggregate member init** | `struct S { int x = 0; };` | Default member initializers in aggregates |
| **Deprecated attribute** | `[[deprecated]] void f();` | |
| ** Sized deallocation** | `void operator delete(void*, size_t);` | |

### Usage Examples

```cpp
// Binary literals and digit separators
__device__ void literals() {
    int mask = 0b11110000;          // Binary literal: 240
    int big   = 1'000'000;          // Digit separator: 1000000
    int flags = 0xFF'FF'FF'FF;      // Hex with separators
}

// Generic lambdas (C++14)
__device__ void genericLambda() {
    auto add = [](auto a, auto b) {
        return a + b;
    };
    int    x = add(1, 2);       // int
    float  y = add(1.0f, 2.0f); // float
}

// Lambda init captures
__device__ void initCapture() {
    int* data = new int[100];
    auto process = [data = data]() {  // capture by move/init
        data[0] = 42;
    };
    process();
    delete[] data;
}

// Variable templates
template<typename T>
__device__ constexpr T epsilon() {
    return T(1e-6);
}

template<typename T>
__device__ constexpr T pi = T(3.14159265358979323846);

// Extended constexpr with loops
__device__ constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;  // Allowed in C++14 constexpr
}

// Return type deduction for normal functions
__device__ auto getSharedPtr() {
    // Deduced return type
    extern __shared__ int smem[];
    return smem;
}
```

---

## 16.4 C++17 Features

### Core Language Features

| Feature | Example | Notes |
|---|---|---|
| **Fold expressions** | `(args + ...)` | Parameter pack expansion |
| **Structured bindings** | `auto [x, y] = pair;` | Decompose tuples, pairs, structs |
| **constexpr if** | `if constexpr (cond)` | Compile-time conditional |
| **Template argument deduction** | `pair p{1, 2.0};` | Class templates without specifying types |
| **Inline variables** | `inline constexpr int N = 42;` | |
| **Nested namespaces** | `namespace A::B::C {}` | |
| **if/switch with init-statement** | `if (int x = f(); x > 0)` | |
| **UTF-8 character literals** | `u8'x'` | |
| **Hexadecimal floating point** | `0x1.2p3` | |
| **Guaranteed copy elision** | Return prvalues without copy | |
| **__has_include** | `#if __has_include(<header>)` | Check header availability |
| **Static_assert without message** | `static_assert(N > 0)` | No second argument required |

### Usage Examples

```cpp
// Fold expressions
template<typename... Args>
__device__ int sumAll(Args... args) {
    return (args + ...);           // Right fold: (a1 + (a2 + (a3 + a4)))
}

template<typename... Args>
__device__ int sumAllLeft(Args... args) {
    return (... + args);           // Left fold: (((a1 + a2) + a3) + a4)
}

template<typename... Args>
__device__ void printAll(Args... args) {
    ((printf("%d ", args)), ...);  // Comma fold: print each arg
}

// Structured bindings
__device__ void structuredBindings() {
    // With arrays
    int arr[3] = {1, 2, 3};
    auto [a, b, c] = arr;

    // With structs
    struct Point { float x, y, z; };
    Point p = {1.0f, 2.0f, 3.0f};
    auto [x, y, z] = p;

    // With pairs
    auto pair = thrust::make_pair(1, 2.0f);
    auto [first, second] = pair;
}

// constexpr if
template<typename T>
__device__ void process(T value) {
    if constexpr (sizeof(T) == 4) {
        // 32-bit path
        value *= 2;
    } else if constexpr (sizeof(T) == 8) {
        // 64-bit path
        value *= 4;
    } else {
        // Other sizes
    }
}

// Template argument deduction
__device__ void ctad() {
    // Before C++17: thrust::pair<int, float> p(1, 2.0f);
    // C++17: deduction guides handle this
    thrust::pair p(1, 2.0f);  // Deduced to pair<int, float>
}

// Inline variables
// In header file:
inline constexpr int WARP_SIZE = 32;
inline constexpr int MAX_THREADS = 1024;

// Nested namespaces
namespace math::gpu::detail {
    __device__ float inner_product(float a, float b) {
        return a * b;
    }
}

// if with init-statement
__device__ void initIf(int* ptr, int size) {
    if (int* end = ptr + size; end != nullptr) {
        for (int i = 0; i < size; ++i) {
            ptr[i] *= 2;
        }
    }
}

// Static assert without message
template<int N>
__device__ void processArray() {
    static_assert(N > 0);          // No message string required in C++17
    static_assert(N <= 1024);      // Same
    int arr[N];
    // ...
}
```

---

## 16.5 C++20 Features

### Core Language Features

| Feature | Example | Notes |
|---|---|---|
| **Concepts** | `template<Arithmetic T>` | Constrained templates |
| **Coroutines** | `co_await`, `co_yield`, `co_return` | Host only |
| **Modules** | `import std;` | Host only, experimental |
| **Consteval** | `consteval int f()` | Must be evaluated at compile time |
| **Constinit** | `constinit int x = f();` | Compile-time initialization |
| **Three-way comparison** | `auto cmp = a <=> b;` | Spaceship operator |
| **Designated initializers** | `Point{.x=1, .y=2}` | Named member initialization |
| **Range-based for with init** | `for (auto i=0; auto& x : v)` | |
| **Likely/unlikely attributes** | `[[likely]]` | Branch prediction hints |
| **Noexcept in function type** | `void (*fp)() noexcept` | Part of type system |
| **Char8_t** | `char8_t` type | Distinct UTF-8 character type |
| **Aggregate paren initialization** | `Point(1, 2)` for aggregates | |
| **Pack expansion in lambda init-capture** | `[...args = std::move(args)]` | |
| **Remove cv and reference simplification** | `std::remove_cvref_t<T>` | |
| **String literal as template parameter** | `template<auto N> void f()` | NTTP with strings |

### Usage Examples

```cpp
// Concepts (C++20)
template<typename T>
concept Numeric = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a * b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
};

// Or using standard concepts
#include <concepts>
template<std::integral T>
__device__ T gcd(T a, T b) {
    while (b != 0) {
        auto t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Using requires clause
template<typename T>
requires requires(T a, T b) { a + b; }
__device__ auto add(T a, T b) {
    return a + b;
}

// Consteval functions (C++20)
consteval int compileTimeSquare(int x) {
    return x * x;
}
// constexpr int arr[compileTimeSquare(4)];  // OK: evaluated at compile time
// int y; compileTimeSquare(y);              // Error: not constexpr

// Three-way comparison (spaceship operator)
struct Vec3 {
    float x, y, z;
    auto operator<=>(const Vec3&) const = default;
};

__device__ void compare() {
    Vec3 a{1.0f, 2.0f, 3.0f};
    Vec3 b{1.0f, 2.0f, 4.0f};
    auto cmp = a <=> b;
    if (cmp < 0) printf("a < b\n");   // a is less
    if (cmp == 0) printf("a == b\n"); // equal
    if (cmp > 0) printf("a > b\n");   // a is greater
}

// Designated initializers
struct KernelConfig {
    int blockSize = 256;
    int gridSize = 1;
    int sharedMem = 0;
};

__device__ void designatedInit() {
    KernelConfig cfg = {
        .blockSize = 512,
        .gridSize = 10,
        .sharedMem = 4096
    };
}

// Range-based for with init statement (C++20)
__device__ void forWithInit(int* data, int n) {
    // int sum = 0;
    for (int i = 0; auto& x : data) {  // Init statement + range-for (needs bounds)
        // x processing
        if (++i >= n) break;
    }
}

// [[likely]] and [[unlikely]] attributes
__device__ int branchPredict(int x) {
    if (x > 0) [[likely]] {
        return x * 2;     // Common path
    } else [[unlikely]] {
        return -x;        // Rare path
    }
}
```

---

## 16.6 CUDA C++ Standard Library (libcu++)

libcu++ is the CUDA C++ Standard Library, providing a subset of the C++ Standard Library that works on both host and device.

### Overview

```cpp
// libcu++ provides device-usable standard library components
// Available headers typically include:
#include <cuda/std/atomic>       // Atomics
#include <cuda/std/barrier>      // Barriers
#include <cuda/std/chrono>       // Time utilities
#include <cuda/std/complex>      // Complex numbers
#include <cuda/std/functional>   // Function objects
#include <cuda/std/iterator>     // Iterators
#include <cuda/std/limits>       // Numeric limits
#include <cuda/std/memory>       // Smart pointers (limited)
#include <cuda/std/ratio>        // Compile-time ratios
#include <cuda/std/tuple>        // Tuples
#include <cuda/std/type_traits>  // Type traits
#include <cuda/std/utility>      // move, forward, pair
#include <cuda/std/vector>       // Vector (limited device support)
#include <cuda/std/array>        // Array
#include <cuda/std/numeric>      // Numeric algorithms
#include <cuda/std/span>         // Span (C++20 backport)
#include <cuda/std/expected>     // Expected (C++23 backport)
#include <cuda/std/variant>      // Variant (C++17)
#include <cuda/std/optional>     // Optional (C++17)
```

### C++17 Backports of C++20/23/26 Features

libcu++ backports several features from newer C++ standards:

```cpp
// C++20 span (backported to C++14/17 mode)
#include <cuda/std/span>
__device__ void useSpan(cuda::std::span<int> data) {
    for (auto& x : data) { x *= 2; }
    printf("Size: %zu\n", data.size());
}

// C++20 atomic_ref
#include <cuda/std/atomic>
__device__ void atomicRef(int* shared_val) {
    cuda::std::atomic_ref<int> ref(*shared_val);
    ref.fetch_add(1, cuda::std::memory_order_relaxed);
}

// C++23 expected
#include <cuda/std/expected>
__device__ cuda::std::expected<int, int> divide(int a, int b) {
    if (b == 0) return cuda::std::unexpected(0);
    return a / b;
}

// C++17 optional
#include <cuda/std/optional>
__device__ cuda::std::optional<int> findValue(int* arr, int n, int target) {
    for (int i = 0; i < n; ++i) {
        if (arr[i] == target) return i;
    }
    return cuda::std::nullopt;
}

// C++17 variant
#include <cuda/std/variant>
using Value = cuda::std::variant<int, float, double>;

__device__ void processVariant(Value v) {
    cuda::std::visit([](auto&& arg) {
        using T = cuda::std::decay_t<decltype(arg)>;
        if constexpr (cuda::std::is_same_v<T, int>) {
            printf("int: %d\n", arg);
        } else if constexpr (cuda::std::is_same_v<T, float>) {
            printf("float: %f\n", arg);
        }
    }, v);
}
```

### Extended Data Types

```cpp
// __int128 (64-bit minimum CC)
__device__ void int128Example() {
    __int128 big = 0;
    big = (__int128)1 << 100;       // 2^100
    printf("High bits: %lld\n", (long long)(big >> 64));
}

// __half (16-bit floating point)
#include <cuda_fp16.h>
__device__ void halfExample() {
    __half a = __float2half(3.14f);
    __half b = __float2half(2.0f);
    __half c = __hadd(a, b);          // half addition
    float  f = __half2float(c);       // convert back to float
}

// __nv_bfloat16 (16-bit brain floating point)
#include <cuda_bf16.h>
__device__ void bfloat16Example() {
    __nv_bfloat16 a = __float2bfloat16(3.14f);
    __nv_bfloat16 b = __float2bfloat16(2.0f);
    __nv_bfloat16 c = __hadd(a, b);   // bfloat16 addition
    float f = __bfloat162float(c);    // convert back to float
}

// __float128 (128-bit floating point, limited operations)
#if defined(__FLOAT128__)
__device__ void float128Example() {
    __float128 pi = 3.14159265358979323846264338327950288q;
    // Limited arithmetic support
}
#endif

// Compound types with half/bfloat16
__device__ void compoundHalfTypes() {
    // __half2: two half values packed in 32 bits
    __half2 a = __floats2half2_rn(1.0f, 2.0f);
    __half2 b = __floats2half2_rn(3.0f, 4.0f);
    __half2 c = __hadd2(a, b);  // vectorized add: (4.0, 6.0)

    // __nv_bfloat162: two bfloat16 values packed in 32 bits
    __nv_bfloat162 ab = __floats2bfloat162_rn(1.0f, 2.0f);
}
```

### Type Traits in Device Code

```cpp
#include <cuda/std/type_traits>

__device__ void typeTraitsExample() {
    // Standard type traits available on device
    static_assert(cuda::std::is_integral_v<int>);
    static_assert(cuda::std::is_floating_point_v<float>);
    static_assert(cuda::std::is_pointer_v<int*>);
    static_assert(cuda::std::is_const_v<const int>);

    // CUDA-specific type traits
    static_assert(__is_host_callable<decltype(hostFunc)>::value == true);
    static_assert(__is_device_callable<decltype(deviceFunc)>::value == true);
}
```

---

## 16.7 C Standard Library Functions

The following C standard library functions are available in device code:

### Timing Functions

```cpp
// Per-SM cycle counter (low precision, wraps frequently)
__device__ clock_t clock();

// 64-bit cycle counter (recommended)
__device__ long long int clock64();

// Usage example: measure kernel execution time
__global__ void timedKernel(float* data, int N) {
    long long start = clock64();

    // ... kernel work ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;

    long long end = clock64();
    long long cycles = end - start;

    if (idx == 0) {
        printf("Thread 0 took %lld cycles\n", cycles);
    }
}
```

### Printf

```cpp
// Kernel printf with limited buffer (default 1 MiB per device)
__device__ int printf(const char* format, ...);

// Usage
__global__ void printfKernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: value = %f\n", tid, 3.14f);
}

// Configure printf buffer size
// Method 1: Environment variable
// CUDA_PRINTF_LIMIT=10  (in MiB)

// Method 2: Runtime API
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024);  // 10 MiB

// Flush printf buffer
cudaDeviceSynchronize();  // Flushes printf buffer
```

### Memory Functions

```cpp
// Standard memory operations
__device__ void* memcpy(void* dest, const void* src, size_t n);
__device__ void* memset(void* ptr, int value, size_t n);
__device__ void* malloc(size_t size);   // Device-side heap allocation
__device__ void free(void* ptr);         // Device-side heap deallocation

// Stack allocation
__device__ void* alloca(size_t size);    // Allocate on stack, auto-freed

// Usage example
__global__ void memoryKernel(int N) {
    // Device-side heap allocation (slow, use sparingly)
    int* temp = (int*)malloc(N * sizeof(int));
    if (temp == nullptr) {
        printf("Device malloc failed!\n");
        return;
    }

    memset(temp, 0, N * sizeof(int));

    // ... work with temp ...

    free(temp);

    // Stack allocation (fast, limited by stack size)
    int* local = (int*)alloca(64 * sizeof(int));
    // local is automatically freed when function returns
}
```

### Configuring Device Heap and Stack

```cpp
// Set device heap size (default 8 MiB)
cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * 1024 * 1024);  // 32 MiB

// Set per-thread stack size (default varies by CC)
cudaDeviceSetLimit(cudaLimitStackSize, 8192);  // 8 KiB per thread

// Query current limits
size_t heapSize, stackSize;
cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
printf("Device heap: %zu MiB, Stack: %zu bytes/thread\n",
       heapSize / (1024 * 1024), stackSize);
```

### Mathematical Functions

```cpp
// All math.h functions are available on device
// Single precision
__device__ float sqrtf(float x);
__device__ float sinf(float x);
__device__ float cosf(float x);
__device__ float expf(float x);
__device__ float logf(float x);
__device__ float powf(float x, float y);
__device__ float fabsf(float x);
__device__ float fmodf(float x, float y);
__device__ float floorf(float x);
__device__ float ceilf(float x);
__device__ float roundf(float x);
__device__ float fminf(float x, float y);
__device__ float fmaxf(float x, float y);

// Double precision
__device__ double sqrt(double x);
__device__ double sin(double x);
__device__ double cos(double x);
__device__ double exp(double x);
__device__ double log(double x);

// Integer math
__device__ int abs(int x);
__device__ long long llabs(long long x);
__device__ int min(int a, int b);
__device__ int max(int a, int b);
__device__ unsigned umin(unsigned a, unsigned b);
__device__ unsigned umax(unsigned a, unsigned b);
```

---

## 16.8 Lambda Expressions

CUDA provides extended support for lambda expressions in device code.

### Basic Lambdas

```cpp
// Device lambda (defined in device code)
__global__ void lambdaKernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Basic lambda
    auto square = [](int x) { return x * x; };

    // Lambda with capture
    int factor = 3;
    auto scale = [factor](int x) { return x * factor; };  // Capture by value

    // Lambda with reference capture
    int sum = 0;
    auto accumulate = [&sum](int val) { sum += val; };

    if (tid < N) {
        data[tid] = square(data[tid]);
        accumulate(data[tid]);
    }
}
```

### Extended Lambdas (--extended-lambda)

The `--extended-lambda` flag allows defining device lambdas in host code, which is useful for passing lambdas to `__global__` function template parameters:

```bash
nvcc --extended-lambda my_program.cu
```

```cpp
// Extended lambda: device lambda defined in host code
void launchKernel() {
    int* d_data;
    cudaMalloc(&d_data, 256 * sizeof(int));

    // Extended lambda can be used as kernel argument
    auto transform = [] __device__ (int x) {
        return x * 2 + 1;
    };

    // Pass lambda to a kernel (thrust or custom)
    thrust::transform(
        thrust::device,
        d_data, d_data + 256,
        d_data,
        transform
    );
}
```

### Device Lambdas as __global__ Arguments

```cpp
// Passing lambdas to kernel launches
template<typename Transform>
__global__ void transformKernel(int* data, int N, Transform op) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        data[tid] = op(data[tid]);
    }
}

void launchWithLambda() {
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));

    int multiplier = 5;

    // Device lambda as kernel argument
    auto op = [=] __device__ (int x) {
        return x * multiplier + 1;
    };

    transformKernel<<<4, 256>>>(d_data, 1024, op);
    cudaDeviceSynchronize();
}
```

### Type Traits for Lambda Detection

```cpp
// Check if a type is an extended device lambda closure
template<typename T>
__host__ __device__ bool isDeviceLambda() {
    return __nv_is_extended_device_lambda_closure_type(T);
}

// Usage
auto hostLambda = [](int x) { return x; };
auto deviceLambda = [] __device__ (int x) { return x; };

// isDeviceLambda<decltype(hostLambda)>()   -> false
// isDeviceLambda<decltype(deviceLambda)>() -> true
```

### Generic Lambdas (C++14+)

```cpp
__global__ void genericLambdaKernel() {
    // Generic lambda with auto parameters
    auto print_val = [](auto val) {
        printf("Value: ");
        // Use if constexpr to handle different types
        if constexpr (cuda::std::is_same_v<decltype(val), int>) {
            printf("%d\n", val);
        } else if constexpr (cuda::std::is_same_v<decltype(val), float>) {
            printf("%f\n", val);
        }
    };

    print_val(42);
    print_val(3.14f);
}
```

---

## 16.9 Restrictions

CUDA C++ has several important restrictions compared to standard C++ that developers must be aware of.

### No Runtime Type Information (RTTI)

```cpp
// NOT allowed in device code:
// typeid(variable)
// dynamic_cast<Derived*>(base_ptr)

// Use alternatives:
// static_cast (when type is known)
// Custom type tags or enums for polymorphism

struct Shape {
    enum Type { CIRCLE, RECTANGLE } type;
    float area;
};

__device__ float computeArea(const Shape* s) {
    switch (s->type) {
        case Shape::CIRCLE:    return 3.14159f * s->area;
        case Shape::RECTANGLE: return s->area;
    }
    return 0.0f;
}
```

### No Exception Handling

```cpp
// NOT allowed in device code:
// try { } catch (...) { }
// throw expression;

// Use error codes instead:
__device__ int safeDivide(int a, int b, int* error) {
    if (b == 0) {
        *error = 1;  // Error code
        return 0;
    }
    return a / b;
}

// Or use cuda::std::expected (C++23 backport):
__device__ cuda::std::expected<int, int> safeDivide(int a, int b) {
    if (b == 0) return cuda::std::unexpected(1);
    return a / b;
}
```

### No long double

```cpp
// NOT available in device code:
// long double x = 3.14159265358979323846L;

// Use double instead:
__device__ void mathFunc() {
    // long double not supported
    double pi = 3.141592653589793;  // Use double for high precision
    // __float128 available on some platforms for even higher precision
}
```

### __global__ Function Restrictions

```cpp
// __global__ functions have specific restrictions:

// 1. No recursion
// NOT allowed:
// __global__ void recursive() { recursive<<<1,1>>>(); }  // No dynamic parallelism recursion

// 2. No variadic arguments
// NOT allowed:
// __global__ void variadic(int count, ...) { }  // Error

// 3. Maximum 32764 bytes of parameters
// NOT allowed: struct with > 32764 bytes passed to kernel
struct BigParams {
    int data[10000];  // 40000 bytes - TOO LARGE for kernel params
};

// Workaround: pass pointer to device memory
__global__ void bigParamKernel(const int* data, int N) {
    // Read from device memory instead
}

// 4. Must have void return type
__global__ void myKernel() { }  // OK
// __global__ int badKernel() { return 0; }  // Error

// 5. Cannot be a member of a class (must be free function, or static member)
class MyClass {
    __global__ static void kernel();  // OK: static member
    // __global__ void instanceKernel();  // Error: needs 'this'
};
```

### Polymorphic Class Restrictions

```cpp
// Classes with virtual functions cannot be directly copied between host and device

class Base {
public:
    __host__ __device__ virtual void func() { }
};

class Derived : public Base {
public:
    __host__ __device__ void func() override { }
};

// NOT allowed: cudaMemcpy of polymorphic objects between host and device
// Base* h_obj = new Derived();
// Base* d_obj;
// cudaMalloc(&d_obj, sizeof(Base));
// cudaMemcpy(d_obj, h_obj, sizeof(Base), cudaMemcpyHostToDevice);  // UNDEFINED

// Instead: separate data from polymorphism
struct ShapeData {
    int type;
    float params[4];
};

__device__ float computeArea(const ShapeData& s) {
    switch (s.type) {
        case 0: return 3.14159f * s.params[0] * s.params[0];  // Circle
        case 1: return s.params[0] * s.params[1];              // Rectangle
    }
    return 0;
}
```

### Other Restrictions

```cpp
// No standard library I/O in device code
// NOT allowed: std::cout, std::cin, std::cerr (use printf instead)

// No new/delete operators (unless device heap is configured)
// Use cudaMalloc from host, or device malloc with heap configured
// __device__ void* operator new(size_t);  // Available if heap is set up
// __device__ void operator delete(void*); // Available if heap is set up

// Function pointers have limitations
// Cannot take address of __global__ function on device side
__global__ void myKernel() { }

void hostSide() {
    void* kernelPtr;
    cudaGetSymbolAddress(&kernelPtr, myKernel);  // OK from host
    // But cannot use this pointer from device code
}

// Cannot use __shared__ or __constant__ in host code
// __shared__ int x;  // Only in device functions

// __device__ functions called from __global__ must be visible (no separate compilation without -rdc)
// Use -rdc=true for device code linking (relocatable device code)
// nvcc -rdc=true file1.cu file2.cu
```

### Summary of Restrictions

| Restriction | Alternative |
|---|---|
| No RTTI (`typeid`, `dynamic_cast`) | Static casts, enum-based dispatch |
| No exceptions (`try`/`catch`/`throw`) | Error codes, `cuda::std::expected` |
| No `long double` | Use `double` or `__float128` |
| No `__global__` recursion | Iterative algorithms |
| No `__global__` variadic args | Use arrays/structs as params |
| Max 32764 bytes kernel params | Pass device pointers |
| `__global__` must return `void` | Output via pointers |
| No polymorphic host-device copy | Separate data from vtable |
| No `std::cout` / `std::cin` | Use `printf` |
| No `__shared__`/`__constant__` on host | Device functions only |
| No DLL-exported `__global__` functions | Static linking or RTTI patterns |
