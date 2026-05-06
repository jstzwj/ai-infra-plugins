# ATen Operations

## Overview

ATen (A Tensor Library) is PyTorch's core C++ tensor operation library. It provides the fundamental operations for tensor computation -- creation, manipulation, mathematical operations, linear algebra, neural network primitives, and more. ATen serves as the foundation upon which PyTorch's Python API, autograd engine, and JIT compiler are built.

**Source location**: `aten/src/ATen/`

ATen was designed to replace the historical TH/THC libraries (Torch CUDA / Torch CPU) with a modern, unified C++ API. Every `torch.*` Python function maps to one or more ATen C++ operations.

---

## Directory Structure

```
aten/
  src/
    ATen/
      core/              # Core types: Tensor, TensorBase, TensorList
      native/            # Native operator implementations (C++ kernels)
      cpu/               # CPU-specific implementations
      cuda/              # CUDA-specific implementations
      metal/             # Metal-specific implementations
      vulkan/            # Vulkan-specific implementations
      mkldnn/            # oneDNN implementations
      quantized/         # Quantized tensor operations
      sparse/            # Sparse tensor operations
      nn/                # Neural network modules
      ops/               # Organized by operation category
      templates/         # Code generation templates
    ATen/
      ATen.h             # Main include header
      Functions.h         # Generated function declarations
      NativeFunctions.h   # Generated native function declarations
      Tensor.h            # Tensor class (generated methods)
      TensorBody.h        # Tensor method bodies (generated)
```

---

## native_functions.yaml

The `native_functions.yaml` file at `aten/src/ATen/native/native_functions.yaml` is the single source of truth for all ATen native operators. It defines operator schemas, dispatch behavior, and metadata used by the code generator (torchgen).

### Schema Format

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # No device consistency check needed
  structured: True         # Uses structured kernel pattern
  structured_delegate: add.out
  dispatch:
    CPU: add_tensor
    CUDA: add_tensor
    MPS: add_tensor
    SparseCPU: add_sparse
    SparseCUDA: add_sparse
    SparseCsrCPU: add_sparse_csr
    SparseCsrCUDA: add_sparse_csr
  tags: [core, pointwise]

- func: mm(Tensor self, Tensor mat2) -> Tensor
  device_check: ExactSame  # Both tensors must be on same device
  dispatch:
    CPU: mm_cpu
    CUDA: mm_cuda
    MPS: mm_mps

- func: convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
  dispatch:
    CPU: convolution
    CUDA: convolution
```

### Key YAML Fields

| Field | Description |
|-------|-------------|
| `func` | Function schema with name, arguments, and return types |
| `dispatch` | Backend-to-kernel mapping |
| `structured` | Whether to use structured kernel pattern |
| `structured_delegate` | Delegate to another structured op |
| `device_check` | Device consistency check (NoCheck, ExactSame, etc.) |
| `tags` | Operation category tags |
| `manual_kernel_registration` | Skip auto-registration |
| `variants` | function, method, or both |
| `returns` | Return value annotations |

---

## ATen Core Types

### Tensor

```cpp
#include <ATen/core/Tensor.h>

namespace at {

class Tensor : public TensorBase {
public:
  // === Type and Shape ===
  ScalarType scalar_type() const;
  Layout layout() const;
  Device device() const;
  int64_t dim() const;
  int64_t numel() const;
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t size(int64_t dim) const;
  int64_t stride(int64_t dim) const;
  int64_t storage_offset() const;

  // === Data Access ===
  void* data_ptr() const;
  template<typename T> T* data_ptr() const;
  bool is_contiguous() const;
  bool is_contiguous(MemoryFormat format) const;

  // === Device Queries ===
  bool is_cpu() const;
  bool is_cuda() const;
  bool is_xpu() const;
  bool is_mps() const;
  bool is_meta() const;

  // === Tensor Options ===
  TensorOptions options() const;
  bool requires_grad() const;
  Tensor& set_requires_grad(bool requires_grad);

  // === Type Conversion ===
  Tensor to(TensorOptions options, bool non_blocking = false, bool copy = false) const;
  Tensor to(ScalarType dtype) const;
  Tensor to(Device device) const;
  Tensor cuda() const;
  Tensor cpu() const;

  // === Reshape and View ===
  Tensor view(IntArrayRef sizes) const;
  Tensor reshape(IntArrayRef sizes) const;
  Tensor permute(IntArrayRef dims) const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor contiguous(MemoryFormat format = MemoryFormat::Contiguous) const;
  Tensor flatten(int64_t start_dim = 0, int64_t end_dim = -1) const;
  Tensor squeeze(int64_t dim) const;
  Tensor unsqueeze(int64_t dim) const;
  Tensor expand(IntArrayRef sizes, bool implicit = false) const;
  Tensor expand_as(const Tensor& other) const;
  Tensor broadcast_to(IntArrayRef sizes) const;

  // === Item Access ===
  Scalar item() const;
  template<typename T> T item() const;

  // === In-place Operations ===
  Tensor& add_(const Tensor& other, Scalar alpha = 1);
  Tensor& sub_(const Tensor& other, Scalar alpha = 1);
  Tensor& mul_(const Tensor& other);
  Tensor& div_(const Tensor& other);
  Tensor& fill_(Scalar value);
  Tensor& zero_();
  Tensor& copy_(const Tensor& other, bool non_blocking = false);

  // === Indexing ===
  Tensor index(ArrayRef<Tensor> indices) const;
  Tensor index_put(ArrayRef<Tensor> indices, const Tensor& values);
  Tensor index_fill_(int64_t dim, const Tensor& index, Scalar value);
  Tensor index_fill_(int64_t dim, const Tensor& index, const Tensor& value);
  Tensor masked_fill_(const Tensor& mask, Scalar value);
  Tensor gather(int64_t dim, const Tensor& index) const;
  Tensor scatter_(int64_t dim, const Tensor& index, const Tensor& src);

  // === Reductions ===
  Tensor sum(IntArrayRef dim = {}, bool keepdim = false,
             ScalarType dtype = ScalarType::Undefined) const;
  Tensor mean(IntArrayRef dim = {}, bool keepdim = false,
              ScalarType dtype = ScalarType::Undefined) const;
  Tensor max(int64_t dim) const;
  Tensor min(int64_t dim) const;
  Tensor argmax(int64_t dim = -1) const;
  Tensor argmin(int64_t dim = -1) const;
  Tensor norm(Scalar p, IntArrayRef dim = {}, bool keepdim = false) const;
  Tensor prod(int64_t dim = -1, ScalarType dtype = ScalarType::Undefined) const;
  Tensor cumsum(int64_t dim, ScalarType dtype = ScalarType::Undefined) const;
  Tensor cumprod(int64_t dim, ScalarType dtype = ScalarType::Undefined) const;

  // === Arithmetic ===
  Tensor add(const Tensor& other, Scalar alpha = 1) const;
  Tensor sub(const Tensor& other, Scalar alpha = 1) const;
  Tensor mul(const Tensor& other) const;
  Tensor div(const Tensor& other) const;
  Tensor pow(const Tensor& exponent) const;
  Tensor pow(Scalar exponent) const;
  Tensor sqrt() const;
  Tensor abs() const;
  Tensor neg() const;
  Tensor exp() const;
  Tensor log() const;
  Tensor log2() const;
  Tensor log10() const;
  Tensor sin() const;
  Tensor cos() const;
  Tensor tan() const;
  Tensor asin() const;
  Tensor acos() const;
  Tensor atan() const;
  Tensor sinh() const;
  Tensor cosh() const;
  Tensor tanh() const;
  Tensor sigmoid() const;
  Tensor clamp(Scalar min, Scalar max) const;
  Tensor clip(Scalar min, Scalar max) const;

  // === Comparison ===
  Tensor eq(const Tensor& other) const;
  Tensor ne(const Tensor& other) const;
  Tensor lt(const Tensor& other) const;
  Tensor le(const Tensor& other) const;
  Tensor gt(const Tensor& other) const;
  Tensor ge(const Tensor& other) const;
  Tensor equal(const Tensor& other) const;

  // === Linear Algebra ===
  Tensor mm(const Tensor& mat2) const;
  Tensor bmm(const Tensor& mat2) const;
  Tensor mv(const Tensor& vec) const;
  Tensor matmul(const Tensor& other) const;

  // === Type Casting ===
  Tensor toType(ScalarType type) const;
  Tensor castByte() const;
  Tensor castChar() const;
  Tensor castShort() const;
  Tensor castInt() const;
  Tensor castLong() const;
  Tensor castFloat() const;
  Tensor castDouble() const;
  Tensor castHalf() const;
  Tensor castBFloat16() const;

  // === Misc ===
  Tensor clone(MemoryFormat format = MemoryFormat::Preserve) const;
  Tensor type_as(const Tensor& other) const;
  std::string toString() const;
  void print() const;
};

} // namespace at
```

### TensorList

```cpp
#include <ATen/core/TensorList.h>

namespace at {

// Non-owning view of a list of tensors
// Essentially ArrayRef<Tensor>
class TensorList {
public:
  TensorList() = default;
  TensorList(const std::vector<Tensor>& tensors);
  TensorList(ArrayRef<Tensor> tensors);

  const Tensor& operator[](size_t index) const;
  size_t size() const;
  bool empty() const;
  const Tensor* begin() const;
  const Tensor* end() const;
};

} // namespace at
```

---

## Operator Categories

### Creation Operations

#### Tensor Factory Functions

```cpp
// Empty tensor (uninitialized)
at::Tensor empty(IntArrayRef sizes, TensorOptions options = {});
at::Tensor empty_like(const Tensor& other, TensorOptions options = {});

// Zeros
at::Tensor zeros(IntArrayRef sizes, TensorOptions options = {});
at::Tensor zeros_like(const Tensor& other, TensorOptions options = {});

// Ones
at::Tensor ones(IntArrayRef sizes, TensorOptions options = {});
at::Tensor ones_like(const Tensor& other, TensorOptions options = {});

// Full (fill with scalar)
at::Tensor full(IntArrayRef sizes, Scalar value, TensorOptions options = {});
at::Tensor full_like(const Tensor& other, Scalar value, TensorOptions options = {});

// Range and sequence
at::Tensor arange(Scalar end, TensorOptions options = {});
at::Tensor arange(Scalar start, Scalar end, Scalar step = 1, TensorOptions options = {});
at::Tensor linspace(Scalar start, Scalar end, int64_t steps, TensorOptions options = {});
at::Tensor logspace(Scalar start, Scalar end, int64_t steps, double base = 10.0,
                    TensorOptions options = {});
at::Tensor range(Scalar start, Scalar end, Scalar step = 1, TensorOptions options = {});

// Random tensors
at::Tensor rand(IntArrayRef sizes, TensorOptions options = {});
at::Tensor rand_like(const Tensor& other, TensorOptions options = {});
at::Tensor randn(IntArrayRef sizes, TensorOptions options = {});
at::Tensor randn_like(const Tensor& other, TensorOptions options = {});
at::Tensor randint(int64_t high, IntArrayRef sizes, TensorOptions options = {});
at::Tensor randint(int64_t low, int64_t high, IntArrayRef sizes, TensorOptions options = {});
at::Tensor randperm(int64_t n, TensorOptions options = {});

// Identity / diagonal
at::Tensor eye(int64_t n, TensorOptions options = {});
at::Tensor eye(int64_t rows, int64_t cols, TensorOptions options = {});
at::Tensor diag(const Tensor& diagonal, int64_t offset = 0);
at::Tensor diagflat(const Tensor& diagonal, int64_t offset = 0);

// From data
at::Tensor from_blob(void* data, IntArrayRef sizes, TensorOptions options = {});
at::Tensor from_blob(void* data, IntArrayRef sizes, IntArrayRef strides,
                     TensorOptions options = {});
at::Tensor tensor(const std::vector<Scalar>& data, TensorOptions options = {});

// Sparse tensors
at::Tensor sparse_coo_tensor(IntArrayRef sizes, TensorOptions options = {});
at::Tensor sparse_coo_tensor(const Tensor& indices, const Tensor& values,
                              IntArrayRef sizes, TensorOptions options = {});
```

#### Creation Examples

```cpp
// Basic creation
auto t1 = at::zeros({3, 4}, at::kFloat);
auto t2 = at::ones({2, 3, 4});
auto t3 = at::empty({5, 5});
auto t4 = at::full({3, 3}, 7.0);

// Random
auto t5 = at::rand({3, 4});           // uniform [0, 1)
auto t6 = at::randn({3, 4});          // standard normal
auto t7 = at::randint(10, {3, 4});    // random integers [0, 10)

// Sequences
auto t8 = at::arange(10);             // [0, 1, ..., 9]
auto t9 = at::arange(2, 10, 2);       // [2, 4, 6, 8]
auto t10 = at::linspace(0, 1, 5);     // [0, 0.25, 0.5, 0.75, 1.0]

// Identity
auto t11 = at::eye(3);                // 3x3 identity matrix

// From existing data
float data[] = {1, 2, 3, 4, 5, 6};
auto t12 = at::from_blob(data, {2, 3});

// On specific device
auto t13 = at::zeros({3, 4}, at::TensorOptions().device(at::kCUDA));
auto t14 = at::randn({3, 4}, at::kCUDA);  // shorthand

// Specific dtype
auto t15 = at::zeros({3, 4}, at::kInt64);
auto t16 = at::ones({3, 4}, at::kBFloat16);
```

---

### Mathematical Operations

#### Element-wise Arithmetic

```cpp
// Basic arithmetic
at::Tensor add(const Tensor& self, const Tensor& other, Scalar alpha = 1);
at::Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha = 1);
at::Tensor mul(const Tensor& self, const Tensor& other);
at::Tensor div(const Tensor& self, const Tensor& other);
at::Tensor true_divide(const Tensor& self, const Tensor& other);
at::Tensor floor_divide(const Tensor& self, const Tensor& other);
at::Tensor remainder(const Tensor& self, const Tensor& other);
at::Tensor fmod(const Tensor& self, const Tensor& other);
at::Tensor pow(const Tensor& self, const Tensor& exponent);
at::Tensor pow(const Tensor& self, Scalar exponent);
at::Tensor pow(Scalar base, const Tensor& exponent);
at::Tensor sqrt(const Tensor& self);
at::Tensor rsqrt(const Tensor& self);
at::Tensor square(const Tensor& self);
at::Tensor neg(const Tensor& self);
at::Tensor abs(const Tensor& self);
at::Tensor sign(const Tensor& self);
at::Tensor sgn(const Tensor& self);

// Exponential and logarithmic
at::Tensor exp(const Tensor& self);
at::Tensor exp2(const Tensor& self);
at::Tensor expm1(const Tensor& self);
at::Tensor log(const Tensor& self);
at::Tensor log2(const Tensor& self);
at::Tensor log10(const Tensor& self);
at::Tensor log1p(const Tensor& self);
at::Tensor logaddexp(const Tensor& self, const Tensor& other);
at::Tensor logaddexp2(const Tensor& self, const Tensor& other);
at::Tensor xlogy(const Tensor& self, const Tensor& other);

// Trigonometric
at::Tensor sin(const Tensor& self);
at::Tensor cos(const Tensor& self);
at::Tensor tan(const Tensor& self);
at::Tensor asin(const Tensor& self);
at::Tensor acos(const Tensor& self);
at::Tensor atan(const Tensor& self);
at::Tensor atan2(const Tensor& self, const Tensor& other);
at::Tensor sinh(const Tensor& self);
at::Tensor cosh(const Tensor& self);
at::Tensor tanh(const Tensor& self);
at::Tensor asinh(const Tensor& self);
at::Tensor acosh(const Tensor& self);
at::Tensor atanh(const Tensor& self);
at::Tensor sinc(const Tensor& self);
at::Tensor hypot(const Tensor& self, const Tensor& other);

// Rounding and clamping
at::Tensor ceil(const Tensor& self);
at::Tensor floor(const Tensor& self);
at::Tensor round(const Tensor& self);
at::Tensor trunc(const Tensor& self);
at::Tensor frac(const Tensor& self);
at::Tensor clamp(const Tensor& self, std::optional<Scalar> min = {}, std::optional<Scalar> max = {});
at::Tensor clamp_min(const Tensor& self, Scalar min);
at::Tensor clamp_max(const Tensor& self, Scalar max);
at::Tensor clip(const Tensor& self, std::optional<Scalar> min = {}, std::optional<Scalar> max = {});

// Other element-wise
at::Tensor reciprocal(const Tensor& self);
at::Tensor bitwise_not(const Tensor& self);
at::Tensor bitwise_and(const Tensor& self, const Tensor& other);
at::Tensor bitwise_or(const Tensor& self, const Tensor& other);
at::Tensor bitwise_xor(const Tensor& self, const Tensor& other);
at::Tensor bitwise_left_shift(const Tensor& self, const Tensor& other);
at::Tensor bitwise_right_shift(const Tensor& self, const Tensor& other);
at::Tensor lerp(const Tensor& self, const Tensor& end, const Tensor& weight);
at::Tensor lerp(const Tensor& self, const Tensor& end, Scalar weight);
at::Tensor maximum(const Tensor& self, const Tensor& other);
at::Tensor minimum(const Tensor& self, const Tensor& other);
at::Tensor max(const Tensor& self, const Tensor& other);
at::Tensor min(const Tensor& self, const Tensor& other);
at::Tensor fmax(const Tensor& self, const Tensor& other);
at::Tensor fmin(const Tensor& self, const Tensor& other);
at::Tensor nextafter(const Tensor& self, const Tensor& other);

// Special functions
at::Tensor erf(const Tensor& self);
at::Tensor erfc(const Tensor& self);
at::Tensor erfinv(const Tensor& self);
at::Tensor lgamma(const Tensor& self);
at::Tensor digamma(const Tensor& self);
at::Tensor polygamma(int64_t n, const Tensor& self);
at::Tensor mvlgamma(const Tensor& self, int64_t p);
at::Tensor igamma(const Tensor& self, const Tensor& other);
at::Tensor igammac(const Tensor& self, const Tensor& other);
at::Tensor special_expit(const Tensor& self);  // logistic sigmoid
at::Tensor special_logit(const Tensor& self);
at::Tensor special_ndtr(const Tensor& self);
at::Tensor special_i0(const Tensor& self);
at::Tensor special_i0e(const Tensor& self);
at::Tensor special_i1(const Tensor& self);
at::Tensor special_i1e(const Tensor& self);
```

#### Math Operation Examples

```cpp
auto a = at::randn({3, 4});
auto b = at::randn({3, 4});

// Element-wise arithmetic
auto c = at::add(a, b);         // a + b
auto d = at::mul(a, b);         // a * b
auto e = at::add(a, b, 2.5);   // a + 2.5 * b

// Scalar operations
auto f = at::pow(a, 2);         // element-wise square
auto g = at::sqrt(at::abs(a));

// Broadcasting
auto h = at::randn({3, 1});
auto i = at::randn({1, 4});
auto j = at::add(h, i);         // result is {3, 4} via broadcasting

// In-place operations
a.add_(b);       // a = a + b
a.mul_(2);       // a = a * 2

// Chaining
auto result = at::sigmoid(at::matmul(a, b.t()));
```

---

### Reduction Operations

```cpp
// Sum
at::Tensor sum(const Tensor& self);
at::Tensor sum(const Tensor& self, IntArrayRef dim, bool keepdim = false,
               ScalarType dtype = ScalarType::Undefined);
at::Tensor sum_to_size(const Tensor& self, IntArrayRef size);

// Mean
at::Tensor mean(const Tensor& self);
at::Tensor mean(const Tensor& self, IntArrayRef dim, bool keepdim = false,
                ScalarType dtype = ScalarType::Undefined);

// Product
at::Tensor prod(const Tensor& self);
at::Tensor prod(const Tensor& self, int64_t dim, bool keepdim = false,
                ScalarType dtype = ScalarType::Undefined);

// Max and Min
std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim);
std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim);
Tensor max(const Tensor& self);     // all elements
Tensor min(const Tensor& self);     // all elements
Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim = false);
Tensor amin(const Tensor& self, IntArrayRef dim, bool keepdim = false);

// ArgMax and ArgMin
Tensor argmax(const Tensor& self, int64_t dim = -1, bool keepdim = false);
Tensor argmin(const Tensor& self, int64_t dim = -1, bool keepdim = false);

// Norm
Tensor norm(const Tensor& self);
Tensor norm(const Tensor& self, Scalar p);
Tensor norm(const Tensor& self, Scalar p, IntArrayRef dim, bool keepdim = false);
Tensor frobenius_norm(const Tensor& self);
Tensor nuclear_norm(const Tensor& self);

// Cumulative
Tensor cumsum(const Tensor& self, int64_t dim, ScalarType dtype = ScalarType::Undefined);
Tensor cumprod(const Tensor& self, int64_t dim, ScalarType dtype = ScalarType::Undefined);
Tensor cummax(const Tensor& self, int64_t dim);
Tensor cummin(const Tensor& self, int64_t dim);

// LogSumExp
Tensor logsumexp(const Tensor& self, IntArrayRef dim, bool keepdim = false);

// Standard Deviation and Variance
Tensor std(const Tensor& self, bool unbiased = true);
Tensor std(const Tensor& self, IntArrayRef dim, bool unbiased = true, bool keepdim = false);
Tensor var(const Tensor& self, bool unbiased = true);
Tensor var(const Tensor& self, IntArrayRef dim, bool unbiased = true, bool keepdim = false);

// Count non-zero
Tensor count_nonzero(const Tensor& self, IntArrayRef dim = {});

// All / Any
Tensor all(const Tensor& self);
Tensor all(const Tensor& self, int64_t dim, bool keepdim = false);
Tensor any(const Tensor& self);
Tensor any(const Tensor& self, int64_t dim, bool keepdim = false);
```

#### Reduction Examples

```cpp
auto t = at::randn({3, 4, 5});

// Global reductions
at::sum(t);         // sum of all elements -> scalar tensor
at::mean(t);        // mean of all elements -> scalar tensor
at::prod(t);        // product of all elements
at::max(t);         // maximum element value
at::min(t);         // minimum element value
at::norm(t);        // L2 norm of all elements

// Dimensional reductions
at::sum(t, {0});                // sum along dim 0 -> {4, 5}
at::sum(t, {0, 2});             // sum along dims 0 and 2 -> {4}
at::sum(t, {1}, true);          // sum along dim 1, keepdim -> {3, 1, 5}
at::mean(t, {0, 1});            // mean along dims 0 and 1 -> {5}

// Max/Min with indices
auto [values, indices] = at::max(t, 1);  // max along dim 1
// values: {3, 5}, indices: {3, 5}

// ArgMax/ArgMin
auto idx = at::argmax(t);       // index of global max
auto idx2 = at::argmax(t, 1);   // argmax along dim 1

// LogSumExp (numerically stable)
auto lse = at::logsumexp(t, {1});  // {3, 5}
```

---

### BLAS (Basic Linear Algebra Subprograms)

```cpp
// Matrix multiplication
at::Tensor mm(const Tensor& self, const Tensor& mat2);         // 2D x 2D
at::Tensor bmm(const Tensor& self, const Tensor& mat2);        // 3D x 3D (batched)
at::Tensor matmul(const Tensor& self, const Tensor& other);    // generalized
at::Tensor linear(const Tensor& input, const Tensor& weight,
                  const Tensor& bias = {});                      // F.linear

// Matrix-vector
at::Tensor mv(const Tensor& self, const Tensor& vec);          // matrix x vector

// Outer product
at::Tensor ger(const Tensor& self, const Tensor& vec2);        // outer product
at::Tensor outer(const Tensor& self, const Tensor& vec2);

// Addmm and friends
at::Tensor addmm(const Tensor& self, const Tensor& mat1,
                 const Tensor& mat2, Scalar beta = 1, Scalar alpha = 1);
// result = beta * self + alpha * mat1 @ mat2

at::Tensor addmv(const Tensor& self, const Tensor& mat,
                 const Tensor& vec, Scalar beta = 1, Scalar alpha = 1);
// result = beta * self + alpha * mat @ vec

at::Tensor addr(const Tensor& self, const Tensor& vec1,
                const Tensor& vec2, Scalar beta = 1, Scalar alpha = 1);
// result = beta * self + alpha * vec1 outer vec2

at::Tensor addbmm(const Tensor& self, const Tensor& batch1,
                  const Tensor& batch2, Scalar beta = 1, Scalar alpha = 1);
// result = beta * self + alpha * sum(batch1 @ batch2 over batch dim)

at::Tensor baddbmm(const Tensor& self, const Tensor& batch1,
                   const Tensor& batch2, Scalar beta = 1, Scalar alpha = 1);
// result = beta * self + alpha * batch1 @ batch2

// Decompositions
std::tuple<Tensor, Tensor> lstsq(const Tensor& self, const Tensor& A);
std::tuple<Tensor, Tensor> eig(const Tensor& self, bool eigenvectors = false);
std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& self, bool some = true, bool compute_uv = true);
std::tuple<Tensor, Tensor, Tensor> linalg_svd(const Tensor& A, bool full_matrices = true);
std::tuple<Tensor, Tensor> linalg_eig(const Tensor& A);
std::tuple<Tensor, Tensor> linalg_eigh(const Tensor& A, std::string uplo = "L");
Tensor linalg_norm(const Tensor& A, Scalar ord, IntArrayRef dim = {}, bool keepdim = false);
Tensor linalg_inv(const Tensor& A);
Tensor linalg_det(const Tensor& A);
Tensor linalg_solve(const Tensor& A, const Tensor& B);
Tensor linalg_cholesky(const Tensor& A);
Tensor linalg_qr(const Tensor& A, std::string mode = "reduced");

// Trace and diagonal
Tensor trace(const Tensor& self);
Tensor diag(const Tensor& self, int64_t offset = 0);
Tensor diagonal(const Tensor& self, int64_t offset = 0, int64_t dim1 = 0, int64_t dim2 = 1);

// Triangular operations
Tensor triu(const Tensor& self, int64_t diagonal = 0);
Tensor tril(const Tensor& self, int64_t diagonal = 0);
Tensor& triu_(Tensor& self, int64_t diagonal = 0);
Tensor& tril_(Tensor& self, int64_t diagonal = 0);
```

#### BLAS Examples

```cpp
auto A = at::randn({3, 4});
auto B = at::randn({4, 5});
auto C = at::mm(A, B);              // {3, 5}

auto bias = at::randn({5});
auto D = at::addmm(bias, A, B);     // bias + A @ B

auto batchA = at::randn({10, 3, 4});
auto batchB = at::randn({10, 4, 5});
auto batchC = at::bmm(batchA, batchB);  // {10, 3, 5}

// General matmul (handles broadcasting)
auto X = at::randn({2, 3, 4});
auto Y = at::randn({4, 5});
auto Z = at::matmul(X, Y);          // {2, 3, 5}

// Linear layer
auto input = at::randn({8, 16});    // batch_size x in_features
auto weight = at::randn({32, 16});  // out_features x in_features
auto bias2 = at::randn({32});       // out_features
auto output = at::linear(input, weight, bias2);  // {8, 32}

// SVD
auto M = at::randn({5, 3});
auto [U, S, Vh] = at::linalg_svd(M);
```

---

### Comparison Operations

```cpp
// Element-wise comparison (returns bool tensor)
Tensor eq(const Tensor& self, const Tensor& other);
Tensor ne(const Tensor& self, const Tensor& other);
Tensor lt(const Tensor& self, const Tensor& other);
Tensor le(const Tensor& self, const Tensor& other);
Tensor gt(const Tensor& self, const Tensor& other);
Tensor ge(const Tensor& self, const Tensor& other);

// Scalar comparison
Tensor eq(const Tensor& self, Scalar other);
Tensor ne(const Tensor& self, Scalar other);
Tensor lt(const Tensor& self, Scalar other);
Tensor le(const Tensor& self, Scalar other);
Tensor gt(const Tensor& self, Scalar other);
Tensor ge(const Tensor& self, Scalar other);

// Equality test (single bool)
bool equal(const Tensor& self, const Tensor& other);

// Sorting
std::tuple<Tensor, Tensor> sort(const Tensor& self, int64_t dim = -1, bool descending = false);
Tensor argsort(const Tensor& self, int64_t dim = -1, bool descending = false);
std::tuple<Tensor, Tensor> topk(const Tensor& self, int64_t k, int64_t dim = -1,
                                 bool largest = true, bool sorted = true);

// Unique
std::tuple<Tensor, Tensor, Tensor> unique(const Tensor& self, bool sorted = true,
                                            bool return_inverse = false);
std::tuple<Tensor, Tensor, Tensor, Tensor> unique_consecutive(const Tensor& self);

// Where
Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other);
Tensor where(const Tensor& condition);  // returns indices of nonzero elements

// Isin / Isfinite
Tensor isin(const Tensor& elements, const Tensor& test_elements, bool assume_unique = false, bool invert = false);
Tensor isfinite(const Tensor& self);
Tensor isinf(const Tensor& self);
Tensor isnan(const Tensor& self);
Tensor isneginf(const Tensor& self);
Tensor isposinf(const Tensor& self);
Tensor isreal(const Tensor& self);
```

#### Comparison Examples

```cpp
auto a = at::tensor({1, 2, 3, 4, 5});
auto b = at::tensor({3, 2, 1, 4, 6});

// Element-wise comparison
at::eq(a, b);     // [0, 1, 0, 1, 0] (bool)
at::lt(a, b);     // [1, 0, 0, 0, 1] (bool)
at::ge(a, 3);     // [0, 0, 1, 1, 1] (bool)

// Scalar comparison
at::equal(a, b);  // false (single bool)

// Sorting
auto [values, indices] = at::sort(a);
// values: {1, 2, 3, 4, 5}, indices: {0, 1, 2, 3, 4}

auto [vals2, idxs2] = at::sort(a, 0, true);  // descending
// vals2: {5, 4, 3, 2, 1}, idxs2: {4, 3, 2, 1, 0}

// TopK
auto [top_vals, top_idxs] = at::topk(a, 3);  // largest 3
// top_vals: {5, 4, 3}, top_idxs: {4, 3, 2}

// Where
auto condition = at::gt(a, 3);
auto result = at::where(condition, a, at::zeros_like(a));
// result: {0, 0, 0, 4, 5}
```

---

### Manipulation Operations

```cpp
// View / Reshape
Tensor view(const Tensor& self, IntArrayRef sizes);
Tensor reshape(const Tensor& self, IntArrayRef sizes);
Tensor reshape_as(const Tensor& self, const Tensor& other);
Tensor view_as(const Tensor& self, const Tensor& other);

// Permute / Transpose
Tensor permute(const Tensor& self, IntArrayRef dims);
Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1);
Tensor t(const Tensor& self);  // 2D transpose shortcut
Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1);

// Contiguous
Tensor contiguous(const Tensor& self, MemoryFormat format = MemoryFormat::Contiguous);

// Squeeze / Unsqueeze
Tensor squeeze(const Tensor& self);
Tensor squeeze(const Tensor& self, int64_t dim);
Tensor unsqueeze(const Tensor& self, int64_t dim);
Tensor& squeeze_(Tensor& self);
Tensor& squeeze_(Tensor& self, int64_t dim);
Tensor& unsqueeze_(Tensor& self, int64_t dim);

// Flatten / Unflatten
Tensor flatten(const Tensor& self, int64_t start_dim = 0, int64_t end_dim = -1);
Tensor unflatten(const Tensor& self, int64_t dim, IntArrayRef sizes);

// Expand / Repeat
Tensor expand(const Tensor& self, IntArrayRef sizes, bool implicit = false);
Tensor expand_as(const Tensor& self, const Tensor& other);
Tensor repeat(const Tensor& self, IntArrayRef repeats);
Tensor repeat_interleave(const Tensor& repeats, int64_t dim = 0);
Tensor repeat_interleave(const Tensor& self, const Tensor& repeats, int64_t dim = 0);

// Cat / Stack
Tensor cat(TensorList tensors, int64_t dim = 0);
Tensor stack(TensorList tensors, int64_t dim = 0);
Tensor vstack(TensorList tensors);
Tensor hstack(TensorList tensors);
Tensor dstack(TensorList tensors);
Tensor column_stack(TensorList tensors);
Tensor row_stack(TensorList tensors);

// Chunk / Split
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim = 0);
std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim = 0);
std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim = 0);
std::vector<Tensor> tensor_split(const Tensor& self, int64_t sections, int64_t dim = 0);
std::vector<Tensor> vsplit(const Tensor& self, int64_t sections);
std::vector<Tensor> hsplit(const Tensor& self, int64_t sections);
std::vector<Tensor> dsplit(const Tensor& self, int64_t sections);

// Narrow / Select
Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length);
Tensor narrow_copy(const Tensor& self, int64_t dim, int64_t start, int64_t length);
Tensor select(const Tensor& self, int64_t dim, int64_t index);
Tensor slice(const Tensor& self, int64_t dim = 0, int64_t start = 0, int64_t end = INT64_MAX,
             int64_t step = 1);

// Indexing
Tensor index(const Tensor& self, TensorList indices);
Tensor index_put(const Tensor& self, TensorList indices, const Tensor& values);
Tensor masked_select(const Tensor& self, const Tensor& mask);
Tensor take(const Tensor& self, const Tensor& index);
Tensor put_(Tensor& self, const Tensor& index, const Tensor& source);

// Gather / Scatter
Tensor gather(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad = false);
Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, Scalar value);
Tensor scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);

// Flip / Roll
Tensor flip(const Tensor& self, IntArrayRef dims);
Tensor roll(const Tensor& self, IntArrayRef shifts, IntArrayRef dims = {});
Tensor rot90(const Tensor& self, int64_t k = 1, IntArrayRef dims = {0, 1});
```

#### Manipulation Examples

```cpp
auto t = at::arange(24).view({2, 3, 4});

// Reshape
auto r1 = t.reshape({6, 4});
auto r2 = t.view({2, 12});

// Permute dimensions
auto r3 = t.permute({2, 0, 1});  // {4, 2, 3}

// Transpose
auto r4 = at::transpose(t, 0, 1);  // {3, 2, 4}

// Concatenation
auto a = at::randn({2, 3});
auto b = at::randn({2, 3});
auto c = at::cat({a, b}, 0);    // {4, 3} (cat along dim 0)
auto d = at::cat({a, b}, 1);    // {2, 6} (cat along dim 1)
auto e = at::stack({a, b}, 0);  // {2, 2, 3}

// Chunk and split
auto f = at::chunk(c, 2, 0);    // 2 tensors of {2, 3}
auto g = at::split(c, 2, 0);    // 2 tensors of {2, 3}

// Narrow and slice
auto h = at::narrow(c, 0, 1, 2);  // rows 1..2
auto i = at::slice(c, 0, 0, 2);   // rows 0..1

// Gather and scatter
auto src = at::randn({2, 3});
auto idx = at::tensor({{0, 1, 2}, {2, 0, 1}}).to(at::kLong);
auto gathered = at::gather(src, 1, idx);

// Flip
auto flipped = at::flip(t, {0});  // flip along dim 0
```

---

### Convolution Operations

```cpp
// Generic convolution
Tensor convolution(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
                   bool transposed, IntArrayRef output_padding, int64_t groups);

// Specific convolutions
Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
              IntArrayRef stride = {1}, IntArrayRef padding = {0},
              IntArrayRef dilation = {1}, int64_t groups = 1);

Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
              IntArrayRef stride = {1}, IntArrayRef padding = {0},
              IntArrayRef dilation = {1}, int64_t groups = 1);

Tensor conv3d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
              IntArrayRef stride = {1}, IntArrayRef padding = {0},
              IntArrayRef dilation = {1}, int64_t groups = 1);

// Transposed convolutions
Tensor conv_transpose1d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
                        IntArrayRef stride = {1}, IntArrayRef padding = {0},
                        IntArrayRef output_padding = {0}, int64_t groups = 1,
                        IntArrayRef dilation = {1});

Tensor conv_transpose2d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
                        IntArrayRef stride = {1}, IntArrayRef padding = {0},
                        IntArrayRef output_padding = {0}, int64_t groups = 1,
                        IntArrayRef dilation = {1});

Tensor conv_transpose3d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
                        IntArrayRef stride = {1}, IntArrayRef padding = {0},
                        IntArrayRef output_padding = {0}, int64_t groups = 1,
                        IntArrayRef dilation = {1});

// Pooling
Tensor avg_pool1d(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
                  IntArrayRef padding = {0}, bool ceil_mode = false, bool count_include_pad = true);
Tensor avg_pool2d(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
                  IntArrayRef padding = {0}, bool ceil_mode = false, bool count_include_pad = true);
Tensor max_pool1d(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
                  IntArrayRef padding = {0}, bool ceil_mode = false);
Tensor max_pool2d(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
                  IntArrayRef padding = {0}, bool ceil_mode = false);

// Adaptive pooling
Tensor adaptive_avg_pool1d(const Tensor& input, IntArrayRef output_size);
Tensor adaptive_avg_pool2d(const Tensor& input, IntArrayRef output_size);
Tensor adaptive_max_pool2d(const Tensor& input, IntArrayRef output_size);
```

#### Convolution Examples

```cpp
// 2D convolution: input [N, C, H, W], weight [out_C, in_C/groups, kH, kW]
auto input = at::randn({8, 3, 32, 32});
auto weight = at::randn({16, 3, 3, 3});
auto bias = at::zeros({16});
auto output = at::conv2d(input, weight, bias, /*stride=*/{1, 1},
                         /*padding=*/{1, 1});
// output: {8, 16, 32, 32}

// Strided convolution with padding
auto out2 = at::conv2d(input, weight, bias, {2, 2}, {1, 1});
// output: {8, 16, 16, 16}

// Grouped convolution
auto g_weight = at::randn({16, 1, 3, 3});  // in_channels/groups = 3/3 = 1
auto g_input = at::randn({4, 3, 32, 32});
auto g_out = at::conv2d(g_input, g_weight, {}, {1, 1}, {1, 1}, /*dilation=*/{1,1}, /*groups=*/3);

// Max pooling
auto pool_out = at::max_pool2d(output, {2, 2}, {2, 2});
// pool_out: {8, 16, 16, 16}
```

---

### Loss Functions

```cpp
// Negative log-likelihood loss
Tensor nll_loss(const Tensor& input, const Tensor& target,
                const Tensor& weight = {}, int64_t reduction = 1, // 1=mean
                int64_t ignore_index = -100);

// Mean squared error
Tensor mse_loss(const Tensor& input, const Tensor& target, int64_t reduction = 1);

// L1 loss
Tensor l1_loss(const Tensor& input, const Tensor& target, int64_t reduction = 1);

// Cross entropy (combines log_softmax + nll_loss)
Tensor cross_entropy(const Tensor& input, const Tensor& target,
                     const Tensor& weight = {}, int64_t reduction = 1,
                     int64_t ignore_index = -100, double label_smoothing = 0.0);

// Binary cross entropy
Tensor binary_cross_entropy(const Tensor& input, const Tensor& target,
                             const Tensor& weight = {}, int64_t reduction = 1);

// Binary cross entropy with logits
Tensor binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target,
                                         const Tensor& weight = {},
                                         int64_t reduction = 1,
                                         const Tensor& pos_weight = {});

// KL divergence
Tensor kl_div(const Tensor& input, const Tensor& target, int64_t reduction = 2, // 2=batchmean
              bool log_target = false);

// Smooth L1 loss (Huber loss)
Tensor smooth_l1_loss(const Tensor& input, const Tensor& target, int64_t reduction = 1,
                      double beta = 1.0);

// Huber loss
Tensor huber_loss(const Tensor& input, const Tensor& target, int64_t reduction = 1,
                  double delta = 1.0);

// Cosine embedding loss
Tensor cosine_embedding_loss(const Tensor& input1, const Tensor& input2,
                              const Tensor& target, double margin = 0.0,
                              int64_t reduction = 1);

// Hinge embedding loss
Tensor hinge_embedding_loss(const Tensor& self, const Tensor& target,
                             double margin = 1.0, int64_t reduction = 1);

// Margin ranking loss
Tensor margin_ranking_loss(const Tensor& input1, const Tensor& input2,
                            const Tensor& target, double margin = 0.0,
                            int64_t reduction = 1);

// Triplet margin loss
Tensor triplet_margin_loss(const Tensor& anchor, const Tensor& positive,
                           const Tensor& negative, double margin = 1.0,
                           double p = 2.0, double eps = 1e-6, bool swap = false,
                           int64_t reduction = 1);

// CTCLoss
Tensor ctc_loss(const Tensor& log_probs, const Tensor& targets,
                IntArrayRef input_lengths, IntArrayRef target_lengths,
                int64_t blank = 0, int64_t reduction = 1, bool zero_infinity = false);
```

#### Loss Examples

```cpp
// Classification: Cross entropy loss
auto logits = at::randn({8, 10});    // batch_size x num_classes
auto targets = at::randint(0, 10, {8});  // class indices
auto loss = at::cross_entropy(logits, targets);

// Regression: MSE loss
auto pred = at::randn({8, 3});
auto target = at::randn({8, 3});
auto mse = at::mse_loss(pred, target);

// Reduction types: 0=none, 1=mean, 2=sum
auto loss_none = at::cross_entropy(logits, targets, {}, /*reduction=*/0);
auto loss_mean = at::cross_entropy(logits, targets, {}, /*reduction=*/1);
auto loss_sum = at::cross_entropy(logits, targets, {}, /*reduction=*/2);
```

---

### Activation Functions

```cpp
// ReLU
Tensor relu(const Tensor& self);
Tensor& relu_(Tensor& self);          // in-place
Tensor leaky_relu(const Tensor& self, double negative_slope = 0.01);
Tensor& leaky_relu_(Tensor& self, double negative_slope = 0.01);
Tensor selu(const Tensor& self);
Tensor& selu_(Tensor& self);
Tensor elu(const Tensor& self, Scalar alpha = 1.0);
Tensor& elu_(Tensor& self, Scalar alpha = 1.0);
Tensor celu(const Tensor& self, Scalar alpha = 1.0);
Tensor gelu(const Tensor& self, std::string approximate = "none");
Tensor relu6(const Tensor& self);
Tensor prelu(const Tensor& self, const Tensor& weight);
Tensor rrelu(const Tensor& self, Scalar lower = 0.125, Scalar upper = 0.333,
             bool training = false);
Tensor mish(const Tensor& self);
Tensor hardswish(const Tensor& self);
Tensor hardsigmoid(const Tensor& self);
Tensor hardtanh(const Tensor& self, Scalar min_val = -1.0, Scalar max_val = 1.0);
Tensor& hardtanh_(Tensor& self, Scalar min_val = -1.0, Scalar max_val = 1.0);

// Sigmoid
Tensor sigmoid(const Tensor& self);
Tensor& sigmoid_(Tensor& self);
Tensor log_sigmoid(const Tensor& self);

// Tanh
Tensor tanh(const Tensor& self);
Tensor& tanh_(Tensor& self);

// Softmax
Tensor softmax(const Tensor& self, int64_t dim, ScalarType dtype = ScalarType::Undefined);
Tensor log_softmax(const Tensor& self, int64_t dim, ScalarType dtype = ScalarType::Undefined);
Tensor softplus(const Tensor& self, Scalar beta = 1.0, Scalar threshold = 20.0);
Tensor softmin(const Tensor& self, int64_t dim, ScalarType dtype = ScalarType::Undefined);
Tensor softsign(const Tensor& self);

// Gumbel softmax
Tensor gumbel_softmax(const Tensor& logits, double tau = 1.0, bool hard = false);

// GLU (Gated Linear Unit)
Tensor glu(const Tensor& self, int64_t dim = -1);

// SiLU (Sigmoid Linear Unit / Swish)
Tensor silu(const Tensor& self);
Tensor& silu_(Tensor& self);
```

#### Activation Examples

```cpp
auto x = at::randn({4, 5});

// ReLU
auto r = at::relu(x);             // max(0, x)
auto r2 = at::leaky_relu(x, 0.1); // max(0.1*x, x)
auto r3 = at::gelu(x);            // Gaussian Error Linear Unit

// Sigmoid and Tanh
auto s = at::sigmoid(x);          // 1 / (1 + exp(-x))
auto t = at::tanh(x);             // (exp(x) - exp(-x)) / (exp(x) + exp(-x))

// Softmax
auto sm = at::softmax(x, 1);      // softmax along dim 1
auto lsm = at::log_softmax(x, 1); // log softmax (more numerically stable)

// GELU with approximation
auto g1 = at::gelu(x, "none");    // exact GELU
auto g2 = at::gelu(x, "tanh");    // tanh approximation

// GLU
auto x2 = at::randn({4, 10});
auto g = at::glu(x2, -1);         // split and gate: {4, 5}
```

---

### Normalization Operations

```cpp
// Batch normalization
Tensor batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const Tensor& running_mean, const Tensor& running_var,
                  bool training, double momentum, double eps);

// Layer normalization
Tensor layer_norm(const Tensor& input, IntArrayRef normalized_shape,
                  const Tensor& weight = {}, const Tensor& bias = {},
                  double eps = 1e-5);

// Group normalization
Tensor group_norm(const Tensor& input, int64_t num_groups,
                  const Tensor& weight = {}, const Tensor& bias = {},
                  double eps = 1e-5);

// Instance normalization
Tensor instance_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     const Tensor& running_mean, const Tensor& running_var,
                     bool use_input_stats, double momentum, double eps);

// Local response normalization
Tensor local_response_norm(const Tensor& input, int64_t size, double alpha = 1e-4,
                           double beta = 0.75, double k = 1.0);

// RMS normalization
Tensor rms_norm(const Tensor& input, IntArrayRef normalized_shape,
                const Tensor& weight = {}, double eps = 1e-5);

// Normalize (L2 normalize along dimension)
Tensor normalize(const Tensor& input, double p = 2.0, int64_t dim = 1,
                 double eps = 1e-12);
```

#### Normalization Examples

```cpp
// Batch norm: input [N, C, H, W]
auto input = at::randn({8, 3, 32, 32});
auto running_mean = at::zeros({3});
auto running_var = at::ones({3});
auto weight = at::ones({3});
auto bias = at::zeros({3});

auto bn_out = at::batch_norm(input, weight, bias, running_mean, running_var,
                              /*training=*/true, /*momentum=*/0.1, /*eps=*/1e-5);

// Layer norm: normalize over last N dimensions
auto ln_out = at::layer_norm(input, {32, 32});
// or with learnable parameters
auto ln_weight = at::ones({32, 32});
auto ln_bias = at::zeros({32, 32});
auto ln_out2 = at::layer_norm(input, {32, 32}, ln_weight, ln_bias);

// Group norm: groups of channels normalized together
auto gn_out = at::group_norm(input, /*num_groups=*/3);

// Instance norm
auto in_out = at::instance_norm(input, {}, {}, {}, {}, true, 0.1, 1e-5);
```

---

## Dispatch Mechanism

ATen uses a dispatcher to route operator calls to the appropriate backend kernel. When you call `at::add(a, b)`, the dispatcher:

1. Extracts the `DispatchKeySet` from the input tensors
2. Looks up the registered kernel for the highest-priority dispatch key
3. Calls that kernel with the arguments

### How Dispatch Works

```cpp
// When you call:
at::Tensor result = at::add(cpu_tensor, cpu_tensor);

// The dispatcher:
// 1. Computes DispatchKeySet from inputs: {CPU}
// 2. Looks up kernel for "add.Tensor" at key CPU
// 3. Calls the CPU kernel: add_tensor_cpu(self, other, alpha)

// For CUDA tensors:
at::Tensor result2 = at::add(cuda_tensor, cuda_tensor);
// Dispatch key: {CUDA} -> calls add_tensor_cuda

// For tensor requiring grad:
at::Tensor result3 = at::add(tensor_with_grad, other);
// Dispatch key: {Autograd, CPU} -> first calls Autograd kernel
//   Autograd kernel records the operation for backward
//   Then dispatches to CPU kernel for actual computation
```

### Dispatch Priority

Dispatch keys have an ordering. More specific keys are checked first:

1. **Tracer/Functionalize** - highest priority (transforms)
2. **Autograd keys** - gradient computation wrapping
3. **Backend keys** (CPU, CUDA, etc.) - actual computation
4. **Composite keys** - fallback implementations

---

## Operator Registration

### REGISTER_DISPATCH (Internal)

```cpp
// Internal registration macro (generated by torchgen)
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(add_tensor_cpu));
  m.impl("mm", TORCH_FN(mm_cpu));
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("add.Tensor", TORCH_FN(add_tensor_cuda));
  m.impl("mm", TORCH_FN(mm_cuda));
}
```

### Structured Kernels

Structured kernels separate the logic into:
- `structured_compute`: compute the output
- `structured_meta`: compute output metadata (shape, dtype) without data

```cpp
// Example structured kernel pattern
struct structured_add : public at::meta::structured_add {
  void impl(const at::Tensor& self, const at::Tensor& other,
            at::Scalar alpha, const at::Tensor& result) {
    // Actual computation
    at::cpu::add_kernel(self, other, alpha, result);
  }
};
```

---

## Operator Versioning

PyTorch operators are versioned to support model serialization across PyTorch versions. The version is tracked in `torch/_C._get_dispatch_version()` and is incremented when operator behavior changes.

```cpp
// Version bump happens when:
// 1. An operator's semantics change
// 2. New arguments are added with different defaults
// 3. Output format changes

// Version handling in native_functions.yaml:
// Each function has an associated version number
```

---

## Complete Operation Example

```cpp
#include <ATen/ATen.h>
#include <iostream>

void aten_example() {
  // Create tensors
  auto a = at::randn({2, 3});
  auto b = at::randn({3, 4});

  // Matrix multiply
  auto c = at::mm(a, b);  // {2, 4}

  // Element-wise operations
  auto d = at::relu(c);
  auto e = at::add(d, at::randn_like(d));

  // Reduction
  auto mean_val = at::mean(e);
  std::cout << "Mean: " << mean_val.item<float>() << std::endl;

  // Reshape
  auto f = e.reshape({-1});
  auto g = at::softmax(f, 0);

  // Loss
  auto target = at::tensor({1, 0, 0, 1}, at::kLong);
  auto logits = at::randn({1, 4});
  auto loss = at::cross_entropy(logits, target);
  std::cout << "Loss: " << loss.item<float>() << std::endl;
}
```

---

## Summary

ATen provides the complete set of tensor operations that power PyTorch:

1. **Creation**: `empty`, `zeros`, `ones`, `full`, `arange`, `rand`, `randn`, and more
2. **Math**: `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `exp`, `log`, trigonometric and special functions
3. **Reduction**: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `norm`, `prod`
4. **BLAS**: `mm`, `bmm`, `addmm`, `mv`, `matmul`, `linear`, decompositions
5. **Comparison**: `eq`, `ne`, `lt`, `gt`, `sort`, `topk`
6. **Manipulation**: `view`, `reshape`, `permute`, `cat`, `stack`, `chunk`, `split`
7. **Neural Network**: `conv2d`, `batch_norm`, `relu`, `softmax`, `cross_entropy`
8. All operations dispatch through the unified dispatcher to backend-specific kernels defined in `native_functions.yaml`
