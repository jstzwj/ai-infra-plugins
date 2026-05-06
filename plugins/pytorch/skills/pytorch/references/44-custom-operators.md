# Custom Operators

## Overview

Custom operators allow extending PyTorch with new operations that integrate seamlessly with autograd, torch.compile, TorchScript, and the dispatcher. This chapter covers both Python-level and C++-level custom operator registration, schema definition, and integration patterns.

---

## torch.library.define

Defines a new operator schema in the PyTorch dispatcher. The operator has no implementation until one is registered via `torch.library.impl`.

### Signature

```python
torch.library.define(
    qualname: str,                    # qualified name: "namespace::name"
    schema: str,                      # operator schema string
    *,
    lib: Optional[torch.library.Library] = None,  # existing library handle
    tags: Set[torch.Tag] = frozenset(),            # operation tags
) -> None
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `qualname` | `str` | Qualified operator name in `"namespace::name"` format |
| `schema` | `str` | Function schema defining arguments, returns, and alias annotations |
| `lib` | `Library` or None | Optional library to register into (for scoped registration) |
| `tags` | `Set[Tag]` | Tags like `torch.Tag.pt2_compliant_tag` |

### Schema Syntax

```
(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
(Tensor self, int dim, bool keepdim=False) -> Tensor
(Tensor(a!) self, Tensor other) -> Tensor(a!)
(Tensor self, *, bool flag=True) -> Tensor
(Tensor self, int[] dims, ScalarType? dtype=None) -> Tensor
(Tensor[] tensors) -> Tensor
() -> Tensor
```

### Argument Types

| Type | Python Equivalent | Description |
|------|-------------------|-------------|
| `Tensor` | `torch.Tensor` | Required tensor |
| `Tensor?` | `Optional[torch.Tensor]` | Optional tensor |
| `Tensor[]` | `List[torch.Tensor]` | List of tensors |
| `int` | `int` | Integer |
| `int[]` | `List[int]` | List of integers |
| `float` | `float` | Float |
| `bool` | `bool` | Boolean |
| `str` | `str` | String |
| `Scalar` | `int` or `float` | Scalar value |
| `ScalarType` | `torch.dtype` | Data type |
| `Layout` | `torch.layout` | Tensor layout |
| `Device` | `torch.device` | Device |
| `Generator?` | `Optional[torch.Generator]` | RNG generator |

### Examples

```python
import torch
from torch.library import define, impl

# Define a simple element-wise operation
define("myops::add_square(Tensor self, Tensor other) -> Tensor")

# Define with default arguments
define("myops::scale(Tensor self, Scalar factor=1.0) -> Tensor")

# Define with multiple returns
define("myops::minmax(Tensor self) -> (Tensor, Tensor)")

# Define an in-place operation (alias annotation)
define("myops::add_inplace(Tensor(a!) self, Tensor other) -> Tensor(a!)")

# Define with keyword-only arguments
define("myops::norm(Tensor self, *, Scalar p=2) -> Tensor")

# Using a library handle for scoped registration
lib = torch.library.Library("mylib", "DEF")
lib.define("custom_gemm(Tensor A, Tensor B, Tensor? bias=None) -> Tensor")
```

---

## torch.library.impl

Registers an implementation (kernel) for a previously defined operator at a specific dispatch key.

### Signature

```python
torch.library.impl(
    qualname: str,                    # "namespace::name"
    dispatch_key: str,                # "CPU", "CUDA", "CompositeExplicitAutograd", etc.
    *,
    lib: Optional[torch.library.Library] = None,
) -> Callable
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `qualname` | `str` | Qualified operator name |
| `dispatch_key` | `str` | Target dispatch key for this kernel |
| `lib` | `Library` or None | Optional library scope |

### Dispatch Key Strings

```python
# Backend keys
"CPU"                              # CPU implementation
"CUDA"                             # NVIDIA GPU implementation
"XPU"                              # Intel GPU implementation
"MPS"                              # Apple Metal implementation
"Meta"                             # Shape inference (no data)
"PrivateUse1"                      # Custom backend slot

# Autograd keys
"Autograd"                         # All backends autograd
"AutogradCPU"                      # CPU-specific autograd
"AutogradCUDA"                     # CUDA-specific autograd

# Composite keys
"CompositeExplicitAutograd"        # Works on any backend with autograd support
"CompositeImplicitAutograd"        # Works on any backend (implicit autograd)

# Functionality keys
"Functionalize"                    # Functionalization pass
"Autocast"                         # Automatic mixed precision
```

### Examples

```python
import torch
from torch.library import define, impl

# First, define the operator
define("myops::add_square(Tensor self, Tensor other) -> Tensor")

# Register CPU implementation
@impl("myops::add_square", "CPU")
def add_square_cpu(self, other):
    return self * self + other * other

# Register CUDA implementation
@impl("myops::add_square", "CUDA")
def add_square_cuda(self, other):
    return self * self + other * other  # PyTorch auto-routes to CUDA

# Register Meta implementation (shape inference)
@impl("myops::add_square", "Meta")
def add_square_meta(self, other):
    # Return empty tensor with correct shape
    return torch.empty_like(self)

# Register composite implementation (works everywhere)
@impl("myops::scale", "CompositeExplicitAutograd")
def scale_composite(self, factor):
    return self * factor
```

---

## torch.library.register_fake

Registers a FakeTensor implementation for an operator. This is essential for `torch.compile` to work with custom operators. The fake implementation should return tensors with the correct shape and dtype without performing actual computation.

### Signature

```python
torch.library.register_fake(
    qualname: str,
    fn: Callable,
    *,
    lib: Optional[torch.library.Library] = None,
) -> None
```

### Examples

```python
import torch
from torch.library import define, impl, register_fake

define("myops::add_square(Tensor self, Tensor other) -> Tensor")

@register_fake("myops::add_square")
def add_square_fake(self, other):
    # Return tensor with correct metadata (shape, dtype, device)
    # The output of self*self + other*other has the same shape as self
    return torch.empty_like(self)

# More complex fake implementation
define("myops::matmul_relu(Tensor A, Tensor B) -> Tensor")

@impl("myops::matmul_relu", "CompositeExplicitAutograd")
def matmul_relu_impl(A, B):
    return torch.relu(torch.mm(A, B))

@register_fake("myops::matmul_relu")
def matmul_relu_fake(A, B):
    # Shape: (m, k) @ (k, n) -> (m, n)
    m, k1 = A.shape
    k2, n = B.shape
    assert k1 == k2
    return torch.empty(m, n, dtype=A.dtype, device=A.device)

# Fake implementation using existing ops
define("myops::complex_op(Tensor x, int hidden_dim) -> Tensor")

@register_fake("myops::complex_op")
def complex_op_fake(x, hidden_dim):
    # Compose fakes from existing operations
    batch = x.shape[0]
    return torch.empty(batch, hidden_dim, dtype=x.dtype, device=x.device)
```

### Why register_fake is Important

```python
# Without register_fake, torch.compile cannot trace through custom ops:
@torch.compile
def my_function(x):
    return torch.ops.myops.add_square(x, x)
    # Error: "add_square is not supported by torch.compile"
    # Fix: register a fake implementation

# With register_fake:
register_fake("myops::add_square")(add_square_fake)

@torch.compile
def my_function(x):
    return torch.ops.myops.add_square(x, x)
    # Now works! torch.compile uses the fake for shape inference
```

---

## Custom C++ Operators: TORCH_LIBRARY

### Full C++ Registration Pattern

```cpp
// my_ops.cpp
#include <torch/torch.h>

// Step 1: Define operator schema
TORCH_LIBRARY(myops, m) {
    m.def("add_square(Tensor self, Tensor other) -> Tensor");
    m.def("scale(Tensor self, Scalar factor=1.0) -> Tensor");
    m.def("custom_gemm(Tensor A, Tensor B, Tensor? bias=None) -> Tensor");
}

// Step 2: Implement CPU kernel
torch::Tensor add_square_cpu(const torch::Tensor& self, const torch::Tensor& other) {
    TORCH_CHECK(self.sizes() == other.sizes(), "Shape mismatch");
    auto result = torch::empty_like(self);
    auto self_acc = self.accessor<float, 2>();
    auto other_acc = other.accessor<float, 2>();
    auto result_acc = result.accessor<float, 2>();
    for (int i = 0; i < self_acc.size(0); ++i) {
        for (int j = 0; j < self_acc.size(1); ++j) {
            result_acc[i][j] = self_acc[i][j] * self_acc[i][j]
                             + other_acc[i][j] * other_acc[i][j];
        }
    }
    return result;
}

// Step 3: Register CPU kernel
TORCH_LIBRARY_IMPL(myops, CPU, m) {
    m.impl("add_square", &add_square_cpu);
    m.impl("scale", [](const torch::Tensor& self, c10::Scalar factor) {
        return self * factor.toDouble();
    });
}
```

### CMakeLists.txt for Custom C++ Op

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_ops)

set(CMAKE_CXX_STANDARD 17)

# Find PyTorch
find_package(Torch REQUIRED)

# Build the operator library
add_library(my_ops SHARED my_ops.cpp)
target_link_libraries(my_ops "${TORCH_LIBRARIES}")
set_property(TARGET my_ops PROPERTY CXX_STANDARD 17)
```

### Loading in Python

```python
import torch
torch.ops.load_library("build/libmy_ops.so")

# Use the custom operator
x = torch.randn(3, 4)
y = torch.randn(3, 4)
result = torch.ops.myops.add_square(x, y)
```

---

## Custom CUDA Kernels Integration

### CUDA Kernel Registration

```cpp
// my_cuda_ops.cu
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void add_square_kernel(
    const float* self, const float* other, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = self[idx] * self[idx] + other[idx] * other[idx];
    }
}

torch::Tensor add_square_cuda(const torch::Tensor& self, const torch::Tensor& other) {
    TORCH_CHECK(self.is_cuda(), "self must be CUDA tensor");
    TORCH_CHECK(self.sizes() == other.sizes(), "Shape mismatch");

    auto result = torch::empty_like(self);
    int n = self.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_square_kernel<<<blocks, threads>>>(
        self.data_ptr<float>(),
        other.data_ptr<float>(),
        result.data_ptr<float>(),
        n);

    return result;
}

// Register
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
    m.impl("add_square", &add_square_cuda);
}
```

---

## torch.ops Namespace

### Accessing Custom Operators

```python
import torch

# After registering "myops::add_square":

# Access via torch.ops.namespace.op_name
result = torch.ops.myops.add_square(x, y)

# Access via the full qualified name
op = torch.ops.myops.add_square
print(op)  # <OpOverload myops.add_square>

# Check available overloads
print(torch.ops.myops.add_square.overloads())  # ['']

# Call with keyword arguments
result = torch.ops.myops.scale(x, factor=2.0)
```

### OpOverload Object

```python
# Get the overload object
op = torch.ops.myops.add_square

# Schema information
print(op.schema())  # "myops::add_square(Tensor self, Tensor other) -> Tensor"

# Check dispatch keys
print(op.dispatch_kernels())  # shows registered kernels

# Call directly
result = op(x, y)
```

---

## torch._custom_ops

### custom_op Decorator

```python
import torch
from torch._custom_ops import custom_op

# Simplified custom operator definition
@custom_op("mylib::relu_squared")
def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU and then squares the result."""
    return torch.relu(x).square()

# Register implementation for specific device
@relu_squared.impl("CPU")
def relu_squared_cpu(x):
    return torch.relu(x).square()

@relu_squared.impl("CUDA")
def relu_squared_cuda(x):
    return torch.relu(x).square()  # auto-routes to CUDA

# Register FakeTensor implementation
@relu_squared.register_fake
def relu_squared_fake(x):
    return torch.empty_like(x)

# Register autograd implementation
@relu_squared.register_autograd
def relu_squared_autograd(ctx, grad_output):
    # Backward: d/dx (max(0,x)^2) = 2 * max(0,x) if x > 0 else 0
    x = ctx.saved_tensors[0]
    return grad_output * 2 * x * (x > 0).to(x.dtype)

# Usage
x = torch.randn(4, requires_grad=True)
y = torch.ops.mylib.relu_squared(x)
y.sum().backward()
print(x.grad)
```

### CustomOp Class

```python
from torch._custom_ops import CustomOp

# The custom_op decorator returns a CustomOp object
# with these methods:

class CustomOp:
    def impl(self, device_type: str) -> Callable:
        """Register implementation for a device type."""

    def register_fake(self, fn: Callable) -> None:
        """Register FakeTensor implementation for torch.compile."""

    def register_autograd(self, fn: Callable) -> None:
        """Register custom backward implementation."""

    def register_kernel(self, dispatch_key: str, fn: Callable) -> None:
        """Register kernel for arbitrary dispatch key."""

    def __call__(self, *args, **kwargs):
        """Call the operator."""
```

---

## torch.library.register_autograd

Registers a custom backward (autograd) implementation for an operator.

### Signature

```python
torch.library.register_autograd(
    qualname: str,
    fn: Callable,
    *,
    lib: Optional[torch.library.Library] = None,
) -> None
```

### Custom Backward Implementation

```python
import torch
from torch.library import define, impl, register_autograd

# Define a custom sigmoid function with manual backward
define("myops::custom_sigmoid(Tensor self) -> Tensor")

@impl("myops::custom_sigmoid", "CompositeExplicitAutograd")
def custom_sigmoid_forward(self):
    return 1.0 / (1.0 + torch.exp(-self))

@register_autograd("myops::custom_sigmoid")
def custom_sigmoid_backward(ctx, grad_output):
    # ctx contains saved tensors from forward
    result = ctx.saved_tensors[0]  # the output of forward
    grad_input = grad_output * result * (1 - result)
    return grad_input

# The forward function must save tensors for backward
# Modify the forward to use save_for_backward:
define("myops::custom_sigmoid_v2(Tensor self) -> Tensor")

@impl("myops::custom_sigmoid_v2", "CompositeExplicitAutograd")
def custom_sigmoid_v2_forward(self):
    result = 1.0 / (1.0 + torch.exp(-self))
    # Save for backward
    ctx = torch.autograd.function.FunctionCtx()
    ctx.save_for_backward(result)
    return result

# Usage
x = torch.randn(4, requires_grad=True)
y = torch.ops.myops.custom_sigmoid(x)
y.sum().backward()
print(x.grad)  # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
```

---

## Composite Operators from Primitives

Composite operators are built from existing ATen operations. They work on any backend without needing per-backend kernels.

```python
from torch.library import define, impl

# Define a composite operation
define("myops::layer_norm_relu(Tensor input, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor")

@impl("myops::layer_norm_relu", "CompositeExplicitAutograd")
def layer_norm_relu(input, weight, bias, eps=1e-5):
    # Composed from existing ATen operations
    normalized = torch.layer_norm(input, weight.shape, weight, bias, eps)
    return torch.relu(normalized)

# This works on CPU, CUDA, MPS, etc. automatically
# because it uses only existing ATen operations
```

### Benefits of Composite Operations

```python
# 1. No need to write backend-specific kernels
# 2. Autograd is handled automatically
# 3. torch.compile can inline and optimize
# 4. Works on any backend that supports the constituent operations
```

---

## Serialization of Custom Ops

Custom operators must be available when loading a serialized model that uses them.

```python
import torch

# When saving a model that uses custom ops:
class MyModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.myops.custom_sigmoid(x)

model = MyModel()
scripted = torch.jit.script(model)
scripted.save("model.pt")

# When loading:
# The custom op must be registered before loading
# Either import the module that registers it, or load the library
torch.ops.load_library("libmy_ops.so")

loaded = torch.jit.load("model.pt")
result = loaded(torch.randn(4))
```

---

## torch.utils.cpp_extension

### CppExtension

```python
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup

setup(
    name='my_ops',
    ext_modules=[
        CppExtension(
            name='my_ops_cpp',        # Python module name
            sources=['my_ops.cpp'],   # C++ source files
            extra_compile_args=['-O3'],  # compiler flags
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
```

### CUDAExtension

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

setup(
    name='my_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='my_cuda_ops',
            sources=[
                'my_ops.cpp',       # C++ sources
                'my_cuda_kernel.cu' # CUDA sources
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            },
            include_dirs=['include/'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    },
)
```

### BuildExtension Options

```python
BuildExtension.with_options(
    use_ninja=True,            # Use Ninja build system (faster)
    no_python_abi_suffix=False, # Python ABI tag
)
```

### load_inline: JIT Compilation

```python
from torch.utils.cpp_extension import load_inline

# Define C++ source inline and compile on the fly
cpp_source = """
torch::Tensor add_square(torch::Tensor self, torch::Tensor other) {
    return self * self + other * other;
}
"""

my_ops = load_inline(
    name='my_inline_ops',
    cpp_sources=cpp_source,
    functions=['add_square'],     # functions to expose
    verbose=True,                 # print build output
    extra_cflags=['-O3'],         # extra compiler flags
    # build_directory='./build',  # custom build directory
)

# Use immediately
x = torch.randn(3, 4)
y = torch.randn(3, 4)
result = my_ops.add_square(x, y)
```

### load: Load Pre-built Extension

```python
from torch.utils.cpp_extension import load

my_ops = load(
    name='my_ops',
    sources=['my_ops.cpp', 'my_cuda_kernel.cu'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['--use_fast_math'],
    # extra_include_paths=['include/'],
    # extra_ldflags=['-lmylib'],
)
```

---

## Example: Custom CUDA Kernel with Autograd

```python
import torch
from torch.library import define, impl, register_fake, register_autograd

# Step 1: Define the operator
define("myops::softplus(Tensor self, float beta=1.0, float threshold=20.0) -> Tensor")

# Step 2: Forward implementation (composite)
@impl("myops::softplus", "CompositeExplicitAutograd")
def softplus_forward(self, beta=1.0, threshold=20.0):
    # softplus(x) = (1/beta) * log(1 + exp(beta * x))
    # with threshold for numerical stability
    scaled = self * beta
    return torch.where(
        scaled > threshold,
        self,
        torch.log1p(torch.exp(scaled)) / beta
    )

# Step 3: FakeTensor for torch.compile
@register_fake("myops::softplus")
def softplus_fake(self, beta=1.0, threshold=20.0):
    return torch.empty_like(self)

# Step 4: Custom autograd (more efficient than default)
@register_autograd("myops::softplus")
def softplus_backward(ctx, grad_output):
    self, = ctx.saved_tensors
    beta = ctx.beta
    threshold = ctx.threshold
    scaled = self * beta
    sigmoid = torch.sigmoid(scaled)
    grad_input = grad_output * torch.where(
        scaled > threshold,
        torch.ones_like(self),
        sigmoid
    )
    return grad_input

# Usage
x = torch.randn(4, requires_grad=True)
y = torch.ops.myops.softplus(x, beta=2.0)
y.sum().backward()
print(f"Forward: {y}")
print(f"Gradient: {x.grad}")

# Works with torch.compile
@torch.compile
def compiled_softplus(x):
    return torch.ops.myops.softplus(x, beta=2.0)

result = compiled_softplus(torch.randn(4))
```

---

## Example: Custom CPU Operator with C++ Kernel

```cpp
// custom_relu.cpp
#include <torch/torch.h>
#include <algorithm>

// CPU kernel implementation
torch::Tensor custom_relu_cpu(const torch::Tensor& input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    auto input_data = input.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    auto n = input.numel();

    // Parallel-friendly loop (could use OpenMP or parallel_for)
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        output_data[i] = std::max(0.0f, input_data[i]);
    }

    return output;
}

// Register schema
TORCH_LIBRARY(myops, m) {
    m.def("custom_relu(Tensor input) -> Tensor");
}

// Register CPU kernel
TORCH_LIBRARY_IMPL(myops, CPU, m) {
    m.impl("custom_relu", &custom_relu_cpu);
}

// Meta kernel (for shape inference)
TORCH_LIBRARY_IMPL(myops, Meta, m) {
    m.impl("custom_relu", [](const torch::Tensor& input) {
        return torch::empty_like(input);
    });
}
```

Build and use:

```python
# build_and_use.py
from torch.utils.cpp_extension import load

my_ops = load(
    name='custom_relu_op',
    sources=['custom_relu.cpp'],
    extra_cflags=['-fopenmp', '-O3'],
)

# Use the custom operator
import torch
x = torch.randn(5, requires_grad=True)
y = torch.ops.myops.custom_relu(x)
# Note: autograd won't work automatically for C++ only ops
# You need to register an autograd implementation too
```

---

## Registration Patterns and Best Practices

### Pattern 1: Pure Python Composite

Best for: operations that can be composed from existing ATen ops

```python
from torch.library import define, impl

define("myops::swish(Tensor self) -> Tensor")

@impl("myops::swish", "CompositeExplicitAutograd")
def swish(self):
    return self * torch.sigmoid(self)
# Works on all backends, autograd handled automatically
```

### Pattern 2: Python with Custom Autograd

Best for: operations needing custom gradient computation

```python
define("myops::custom_op(Tensor self) -> Tensor")

@impl("myops::custom_op", "CompositeExplicitAutograd")
def custom_op_forward(self):
    result = self * 2  # save for backward
    return result

@register_autograd("myops::custom_op")
def custom_op_backward(ctx, grad_output):
    return grad_output * 2
```

### Pattern 3: C++ with CUDA

Best for: performance-critical operations with custom kernels

```python
# 1. Define in C++ with TORCH_LIBRARY
# 2. Register CPU kernel with TORCH_LIBRARY_IMPL(..., CPU, ...)
# 3. Register CUDA kernel with TORCH_LIBRARY_IMPL(..., CUDA, ...)
# 4. Register Meta kernel with TORCH_LIBRARY_IMPL(..., Meta, ...)
# 5. Register autograd if needed
```

### Pattern 4: Using custom_op Decorator

Best for: quick prototyping with automatic registration

```python
from torch._custom_ops import custom_op

@custom_op("mylib::my_function")
def my_function(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale

@my_function.impl("CPU")
def my_function_cpu(x, scale):
    return x * scale

@my_function.register_fake
def my_function_fake(x, scale):
    return torch.empty_like(x)
```

### Best Practices

1. **Always register a FakeTensor implementation** for torch.compile compatibility
2. **Always register a Meta implementation** for shape inference without data
3. **Use CompositeExplicitAutograd** when possible to avoid per-backend registration
4. **Register custom autograd** only when the default composite autograd is insufficient
5. **Use alias annotations** correctly for in-place and view operations
6. **Namespace your operators** to avoid collisions (e.g., `myproject::op_name`)
7. **Write tests** for each dispatch key you register
8. **Keep kernels stateless** -- lambda captures in TORCH_LIBRARY_IMPL must be trivially copyable
9. **Validate inputs** with TORCH_CHECK at the start of each kernel
10. **Support multiple dtypes** or explicitly check and error on unsupported types
