# Chapter 31: JAX Foreign Function Interface (FFI)

## 31.1 Overview

The JAX Foreign Function Interface (FFI) allows calling C++ and CUDA code
directly from JAX programs. This is the modern replacement for `jax.experimental.custom_ops`
and `jaxlib.custom_call`, providing a cleaner API with better support for
differentiation, batching, and GPU execution.

**Key features:**
- Register C++/CUDA functions callable from JAX
- Support for forward and reverse-mode differentiation
- Automatic batching via `vmap`
- GPU kernel integration with stream-aware execution
- Type-safe dispatch with dtype and shape checking

---

## 31.2 Architecture

### 31.2.1 How FFI Works

```
+-----------------------------------------------------------+
| Python Side                                                |
|                                                            |
|   @jax.custom_jvp  /  @jax.custom_vjp                     |
|   @jax.ffi.register_ffi_target                            |
|   def my_op(*args):                                        |
|       return jax.ffi.ffi_call("my_kernel", ...)           |
|                                                            |
+-----------------------------------------------------------+
          |  XLA FFI dispatch
          v
+-----------------------------------------------------------+
| C++ Side                                                   |
|                                                            |
|   Status my_kernel(FfiCallFrame* frame) {                  |
|     // Access inputs, outputs, attributes                  |
|     // Execute custom logic                                |
|     return OkStatus();                                     |
|   }                                                        |
|                                                            |
|   XLA_FFI_REGISTER_HANDLER(my_kernel, "my_kernel", ...)   |
+-----------------------------------------------------------+
```

### 31.2.2 Components

1. **Python registration**: `jax.ffi.register_ffi_target` registers a Python handler
2. **C++ registration**: `XLA_FFI_REGISTER_HANDLER` registers a C++ handler
3. **Calling convention**: `jax.ffi.ffi_call` invokes the registered handler
4. **Differentiation**: Custom JVP/VJP rules via decorators

---

## 31.3 Python Side: Registering and Calling

### 31.3.1 Basic Registration

```python
import jax
import jax.numpy as jnp
from jax import ffi

def my_add_impl(ctx, a, b):
    """Python implementation of a custom add operation."""
    return [a + b]

# Register the implementation
ffi.register_ffi_target(
    "my_add",                 # Target name (string identifier)
    my_add_impl,              # Implementation function
    api_version=1,            # FFI API version
)

# Call the registered target
def my_add(a, b):
    return ffi.ffi_call(
        "my_add",             # Target name
        a, b,                 # Input arrays
        result_shapes=[       # Output shape specifications
            jax.ShapeDtypeStruct(a.shape, a.dtype)
        ],
    )

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
result = my_add(x, y)
print(result)  # [5.0, 7.0, 9.0]
```

### 31.3.2 The FFI Context

The implementation function receives a context object with metadata:

```python
def my_impl(ctx, x):
    """Implementation with context access."""
    # ctx contains:
    # - devices: the devices the computation runs on
    # - stream: GPU stream (if applicable)
    # - attrs: dictionary of custom attributes
    print(f"Device: {ctx.devices}")
    print(f"Platform: {ctx.platform}")
    return [x * 2]

ffi.register_ffi_target("double", my_impl)
```

### 31.3.3 Custom Attributes

Pass additional parameters through the `attrs` argument:

```python
def scaled_add_impl(ctx, a, b):
    """Implementation using custom attributes."""
    alpha = ctx.attrs["alpha"]  # Scalar attribute
    beta = ctx.attrs["beta"]
    return [alpha * a + beta * b]

ffi.register_ffi_target("scaled_add", scaled_add_impl)

def scaled_add(a, b, alpha=1.0, beta=1.0):
    return ffi.ffi_call(
        "scaled_add",
        a, b,
        result_shapes=[
            jax.ShapeDtypeStruct(a.shape, a.dtype)
        ],
        attrs={"alpha": float(alpha), "beta": float(beta)},
    )

x = jnp.ones(5)
y = jnp.ones(5) * 2
result = scaled_add(x, y, alpha=0.5, beta=1.5)
print(result)  # [3.5, 3.5, 3.5, 3.5, 3.5]
```

### 31.3.4 Multiple Outputs

```python
def split_and_sum_impl(ctx, x):
    """Return both the split result and the sum."""
    mid = x.shape[0] // 2
    first_half = x[:mid]
    second_half = x[mid:]
    total = jnp.sum(x)
    return [first_half, total]

ffi.register_ffi_target("split_sum", split_and_sum_impl)

def split_and_sum(x):
    mid = x.shape[0] // 2
    return ffi.ffi_call(
        "split_sum",
        x,
        result_shapes=[
            jax.ShapeDtypeStruct((x.shape[0] // 2,), x.dtype),
            jax.ShapeDtypeStruct((), x.dtype),
        ],
    )

x = jnp.array([1.0, 2.0, 3.0, 4.0])
half, total = split_and_sum(x)
print(half)   # [1.0, 2.0]
print(total)  # 10.0
```

---

## 31.4 C++ Side

### 31.4.1 Basic C++ Handler

```cpp
// my_kernel.cc
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

// Define the handler
ffi::Status MyAddKernel(ffi::Buffer<ffi::R1<float>> a,
                        ffi::Buffer<ffi::R1<float>> b,
                        ffi::ResultBuffer<ffi::R1<float>> out) {
    // Get raw pointers
    const float* a_data = a.typed_data();
    const float* b_data = b.typed_data();
    float* out_data = out->typed_data();

    int64_t n = a.dimensions(0);
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = a_data[i] + b_data[i];
    }

    return ffi::Status::Ok();
}

// Register the handler
XLA_FFI_REGISTER_HANDLER(
    xla::ffi::HandlerRegistry::GetDefault(),
    MyAddKernel,
    "my_add",                    // Name must match Python registration
    ffi::Cpu,                    // Platform: Cpu, Gpu, Tpu
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::R1<float>>>()    // Input a
        .Arg<ffi::Buffer<ffi::R1<float>>>()    // Input b
        .Ret<ffi::Buffer<ffi::R1<float>>>()    // Output
);
```

### 31.4.2 Buffer Types

The C++ FFI supports various buffer types:

```cpp
// Rank-specific buffers
ffi::Buffer<ffi::R0<float>>   scalar   // 0-dimensional
ffi::Buffer<ffi::R1<float>>   vector   // 1-dimensional
ffi::Buffer<ffi::R2<float>>   matrix   // 2-dimensional
ffi::Buffer<ffi::R3<float>>   tensor3d // 3-dimensional
ffi::Buffer<ffi::R4<float>>   tensor4d // 4-dimensional

// Any-rank buffer
ffi::Buffer<ffi::RuntimeBuffer> buffer  // Any rank, any dtype

// Typed any-rank buffer
ffi::Buffer<float> typed_buffer  // Any rank, float dtype
```

### 31.4.3 Accessing Buffer Data

```cpp
ffi::Status ProcessKernel(ffi::Buffer<ffi::R2<float>> input,
                          ffi::ResultBuffer<ffi::R2<float>> output) {
    // Get dimensions
    int64_t rows = input.dimensions(0);
    int64_t cols = input.dimensions(1);

    // Get raw data pointer
    const float* in_data = input.typed_data();
    float* out_data = output->typed_data();

    // Access individual elements (row-major layout)
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            out_data[i * cols + j] = in_data[i * cols + j] * 2.0f;
        }
    }

    return ffi::Status::Ok();
}
```

### 31.4.4 Custom Attributes in C++

```cpp
ffi::Status ScaledAddKernel(
    float alpha, float beta,                       // Attributes
    ffi::Buffer<ffi::R1<float>> a,
    ffi::Buffer<ffi::R1<float>> b,
    ffi::ResultBuffer<ffi::R1<float>> out) {

    const float* a_data = a.typed_data();
    const float* b_data = b.typed_data();
    float* out_data = out->typed_data();

    int64_t n = a.dimensions(0);
    for (int64_t i = 0; i < n; ++i) {
        out_data[i] = alpha * a_data[i] + beta * b_data[i];
    }

    return ffi::Status::Ok();
}

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::HandlerRegistry::GetDefault(),
    ScaledAddKernel,
    "scaled_add",
    ffi::Cpu,
    ffi::Ffi::Bind()
        .Attr<float>("alpha")
        .Attr<float>("beta")
        .Arg<ffi::Buffer<ffi::R1<float>>>()
        .Arg<ffi::Buffer<ffi::R1<float>>>()
        .Ret<ffi::Buffer<ffi::R1<float>>>()
);
```

---

## 31.5 GPU Support

### 31.5.1 GPU Handler Registration

```cpp
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/ffi.h"

namespace ffi = xla::ffi;

// GPU kernel (CUDA)
ffi::Status GpuAddKernel(
    ffi::Buffer<ffi::R1<float>> a,
    ffi::Buffer<ffi::R1<float>> b,
    ffi::ResultBuffer<ffi::R1<float>> out,
    ffi::Stream stream) {              // GPU stream for async execution

    const float* a_data = a.typed_data();
    const float* b_data = b.typed_data();
    float* out_data = out->typed_data();

    int64_t n = a.dimensions(0);

    // Launch CUDA kernel
    // gpu_add_kernel<<<blocks, threads, 0, stream>>>(a_data, b_data, out_data, n);

    return ffi::Status::Ok();
}

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::HandlerRegistry::GetDefault(),
    GpuAddKernel,
    "gpu_add",
    ffi::Gpu,                     // Platform: Gpu
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::R1<float>>>()
        .Arg<ffi::Buffer<ffi::R1<float>>>()
        .Ret<ffi::Buffer<ffi::R1<float>>>()
);
```

### 31.5.2 Complete CUDA Kernel Example

This example shows a complete CUDA kernel integrated with JAX via FFI:

```cpp
// cuda_kernels.cu
#include <cuda_runtime.h>
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

// CUDA kernel for elementwise ReLU
__global__ void ReluKernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

// CUDA kernel for elementwise scaled sigmoid
__global__ void ScaledSigmoidKernel(const float* input, float* output,
                                     float scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = 1.0f / (1.0f + expf(-input[idx]));
        output[idx] = scale * val;
    }
}

// FFI handler for ReLU on GPU
ffi::Status FfiRelu(
    ffi::Buffer<ffi::RuntimeBuffer> input,
    ffi::ResultBuffer<ffi::RuntimeBuffer> output,
    ffi::Stream stream) {

    const float* in_data = reinterpret_cast<const float*>(input.untyped_data());
    float* out_data = reinterpret_cast<float*>(output->untyped_data());

    int64_t n = input.element_count();
    int64_t threads = 256;
    int64_t blocks = (n + threads - 1) / threads;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    ReluKernel<<<blocks, threads, 0, cuda_stream>>>(in_data, out_data, n);

    return ffi::Status::Ok();
}

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::HandlerRegistry::GetDefault(),
    FfiRelu,
    "custom_relu",
    ffi::Gpu,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::RuntimeBuffer>>()
        .Ret<ffi::Buffer<ffi::RuntimeBuffer>>()
);

// FFI handler for scaled sigmoid on GPU
ffi::Status FfiScaledSigmoid(
    float scale,
    ffi::Buffer<ffi::RuntimeBuffer> input,
    ffi::ResultBuffer<ffi::RuntimeBuffer> output,
    ffi::Stream stream) {

    const float* in_data = reinterpret_cast<const float*>(input.untyped_data());
    float* out_data = reinterpret_cast<float*>(output->untyped_data());

    int64_t n = input.element_count();
    int64_t threads = 256;
    int64_t blocks = (n + threads - 1) / threads;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    ScaledSigmoidKernel<<<blocks, threads, 0, cuda_stream>>>(
        in_data, out_data, scale, n);

    return ffi::Status::Ok();
}

XLA_FFI_REGISTER_HANDLER(
    xla::ffi::HandlerRegistry::GetDefault(),
    FfiScaledSigmoid,
    "scaled_sigmoid",
    ffi::Gpu,
    ffi::Ffi::Bind()
        .Attr<float>("scale")
        .Arg<ffi::Buffer<ffi::RuntimeBuffer>>()
        .Ret<ffi::Buffer<ffi::RuntimeBuffer>>()
);
```

### 31.5.3 Python Side for GPU Kernels

```python
import jax
import jax.numpy as jnp
from jax import ffi

# Register the GPU handler (loaded from compiled .so/.dll)
# This assumes the C++ code has been compiled into a shared library
# and loaded via jax.ffi.register_ffi_target or XLA's handler mechanism

def custom_relu(x):
    """Custom ReLU implemented as a CUDA kernel via FFI."""
    return ffi.ffi_call(
        "custom_relu",
        x,
        result_shapes=[
            jax.ShapeDtypeStruct(x.shape, x.dtype)
        ],
    )

# Usage
x = jnp.array([-1.0, 0.0, 1.0, 2.0])
result = custom_relu(x)
print(result)  # [0.0, 0.0, 1.0, 2.0]

def scaled_sigmoid(x, scale=1.0):
    """Scaled sigmoid with CUDA kernel."""
    return ffi.ffi_call(
        "scaled_sigmoid",
        x,
        result_shapes=[
            jax.ShapeDtypeStruct(x.shape, x.dtype)
        ],
        attrs={"scale": float(scale)},
    )

result = scaled_sigmoid(jnp.array([0.0, 1.0, -1.0]), scale=2.0)
print(result)  # [1.0, 1.4621, 0.5379]
```

---

## 31.6 Batching (vmap Support)

### 31.6.1 Automatic Batching

By default, `vmap` over FFI calls falls back to sequential execution. You
can register a custom batching rule for better performance:

```python
from jax import ffi

def my_op_batched(ctx, batched_args, batch_dims):
    """Custom batching rule for my_op."""
    x, y = batched_args
    x_bdim, y_bdim = batch_dims

    # If both inputs are batched along dimension 0
    if x_bdim == 0 and y_bdim == 0:
        # Call the same FFI target with batched inputs
        result = ffi.ffi_call(
            "my_add",
            x, y,
            result_shapes=[
                jax.ShapeDtypeStruct(x.shape, x.dtype)
            ],
        )
        return result, 0  # Result is batched along dim 0

    # Fall back to sequential
    return None, None

ffi.register_ffi_target(
    "my_add",
    my_add_impl,
    batching_rule=my_op_batched,
)
```

### 31.6.2 Using vmap with FFI

```python
# Define the FFI function
def my_add(a, b):
    return ffi.ffi_call(
        "my_add",
        a, b,
        result_shapes=[jax.ShapeDtypeStruct(a.shape, a.dtype)],
    )

# Vectorize over batch dimension
x = jnp.ones((8, 10))
y = jnp.ones((8, 10)) * 2

batched_add = jax.vmap(my_add)
result = batched_add(x, y)
print(result.shape)  # (8, 10)
```

---

## 31.7 Custom Differentiation

### 31.7.1 Custom JVP (Forward-Mode Differentiation)

```python
import jax
import jax.numpy as jnp
from jax import ffi

# Register the primal implementation
def my_relu_impl(ctx, x):
    return [jnp.maximum(x, 0)]

ffi.register_ffi_target("my_relu", my_relu_impl)

# Define the function with custom JVP
@jax.custom_jvp
def my_relu(x):
    return ffi.ffi_call(
        "my_relu",
        x,
        result_shapes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )

@my_relu.defjvp
def my_relu_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents

    # Forward pass: compute primal output
    primal_out = my_relu(x)

    # Tangent: derivative of ReLU is 0 for x<0, 1 for x>0
    tangent_out = x_dot * (x > 0).astype(x_dot.dtype)

    return primal_out, tangent_out

# Test differentiation
x = jnp.array([-1.0, 0.0, 1.0, 2.0])
print(my_relu(x))  # [0.0, 0.0, 1.0, 2.0]
print(jax.grad(lambda x: jnp.sum(my_relu(x)))(x))
# [0.0, 0.0, 1.0, 1.0]
```

### 31.7.2 Custom VJP (Reverse-Mode Differentiation)

```python
@jax.custom_vjp
def my_scaled_op(x, scale):
    """Custom op: y = scale * x^2, with custom VJP."""
    return ffi.ffi_call(
        "scaled_square",
        x,
        result_shapes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
        attrs={"scale": float(scale)},
    )

def my_scaled_op_fwd(x, scale):
    """Forward pass: compute output and residuals."""
    result = my_scaled_op(x, scale)
    # Save values needed for backward pass
    residuals = (x, scale)
    return result, residuals

def my_scaled_op_bwd(residuals, g):
    """Backward pass: compute gradients."""
    x, scale = residuals
    # dy/dx = 2 * scale * x
    # dy/dscale = x^2 (summed)
    g_x = g * 2 * scale * x
    g_scale = jnp.sum(g * x ** 2)
    return g_x, g_scale

my_scaled_op.defvjp(my_scaled_op_fwd, my_scaled_op_bwd)

# Test
x = jnp.array([1.0, 2.0, 3.0])
scale = 2.0

result = my_scaled_op(x, scale)
print(result)  # [2.0, 8.0, 18.0] = 2 * x^2

grad_x = jax.grad(lambda x: jnp.sum(my_scaled_op(x, scale)))(x)
print(grad_x)  # [4.0, 8.0, 12.0] = 2 * 2 * x

grad_scale = jax.grad(lambda s: jnp.sum(my_scaled_op(x, s)))(scale)
print(grad_scale)  # 14.0 = sum(x^2) = 1 + 4 + 9
```

### 31.7.3 Complete Differentiable CUDA Kernel

This example shows a CUDA kernel with full differentiation support:

```python
import jax
import jax.numpy as jnp
from jax import ffi

# Primal implementation (calls CUDA kernel)
def cuda_softmax_impl(ctx, x):
    """GPU softmax via CUDA kernel."""
    # This would call the registered CUDA handler
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return [exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)]

ffi.register_ffi_target("cuda_softmax", cuda_softmax_impl)

# Define with custom VJP
@jax.custom_vjp
def cuda_softmax(x):
    return ffi.ffi_call(
        "cuda_softmax",
        x,
        result_shapes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
    )

def softmax_fwd(x):
    y = cuda_softmax(x)
    return y, y  # Save output for backward

def softmax_bwd(residuals, g):
    y, = residuals
    # dy/dx = y * (g - sum(g * y))
    sum_gy = jnp.sum(g * y, axis=-1, keepdims=True)
    g_x = y * (g - sum_gy)
    return (g_x,)

cuda_softmax.defvjp(softmax_fwd, softmax_bwd)

# Test in a differentiable pipeline
def loss_fn(params, x, targets):
    logits = jnp.dot(x, params)
    probs = cuda_softmax(logits)
    return -jnp.sum(targets * jnp.log(probs + 1e-8))

params = jnp.ones((5, 3))
x = jnp.ones((2, 5))
targets = jnp.zeros((2, 3)).at[:, 0].set(1.0)

grad = jax.grad(loss_fn)(params, x, targets)
print(grad.shape)  # (5, 3)
```

---

## 31.8 Shape and Type Polymorphism

### 31.8.1 Dynamic Output Shapes

For operations where the output shape depends on the input data (e.g.,
filtering, unique), you need to specify output shapes dynamically:

```python
def dynamic_output_impl(ctx, x):
    """Implementation that produces a dynamically-shaped output."""
    mask = x > 0
    result = x[mask]
    return [result]

ffi.register_ffi_target("positive_only", dynamic_output_impl)

# This requires careful shape handling in the FFI call
def positive_only(x):
    # For dynamic shapes, use result_shape_dtypes as a function
    return ffi.ffi_call(
        "positive_only",
        x,
        result_shapes=[jax.ShapeDtypeStruct((jnp.sum(x > 0),), x.dtype)],
    )

x = jnp.array([-1.0, 2.0, -3.0, 4.0, 5.0])
result = positive_only(x)
print(result)  # [2.0, 4.0, 5.0]
```

### 31.8.2 Type Dispatch

```python
def typed_add_impl(ctx, a, b):
    """Type-polymorphic implementation."""
    if a.dtype == jnp.float32:
        return [a + b]
    elif a.dtype == jnp.float16:
        return [(a + b).astype(jnp.float16)]
    else:
        raise TypeError(f"Unsupported dtype: {a.dtype}")

ffi.register_ffi_target("typed_add", typed_add_impl)
```

---

## 31.9 Loading Compiled Kernels

### 31.9.1 Loading Shared Libraries

```python
import ctypes
import jax
from jax import ffi

# Method 1: Load via ctypes before registration
lib = ctypes.CDLL("path/to/my_kernels.so")

# The shared library's static initializers register the handlers
# with XLA's FFI registry. No additional Python registration needed.

# Method 2: Register Python-side targets
ffi.register_ffi_target("my_op", python_impl)

# Now call via ffi_call
result = ffi.ffi_call(
    "my_op",
    input_array,
    result_shapes=[jax.ShapeDtypeStruct(input_array.shape, input_array.dtype)],
)
```

### 31.9.2 Building C++/CUDA Kernels

```bash
# Compile CPU kernel
g++ -shared -fPIC -o my_kernels.so my_kernel.cc \
    -I/path/to/xla/include \
    -L/path/to/xla/lib -lxlaffi

# Compile CUDA kernel
nvcc -shared -Xcompiler -fPIC -o my_gpu_kernels.so \
    cuda_kernels.cu \
    -I/path/to/xla/include \
    -L/path/to/xla/lib -lxlaffi \
    -lcuda
```

---

## 31.10 Complete End-to-End Example

```python
"""
Complete example: Custom layer normalization with CUDA kernel,
differentiation, and batching support.
"""
import jax
import jax.numpy as jnp
from jax import ffi

# Primal implementation
def layer_norm_impl(ctx, x, gamma, beta):
    eps = ctx.attrs.get("eps", 1e-5)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return [x_norm * gamma + beta]

ffi.register_ffi_target("custom_layer_norm", layer_norm_impl)

# Differentiable wrapper
@jax.custom_vjp
def custom_layer_norm(x, gamma, beta, eps=1e-5):
    return ffi.ffi_call(
        "custom_layer_norm",
        x, gamma, beta,
        result_shapes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
        attrs={"eps": float(eps)},
    )

def ln_fwd(x, gamma, beta, eps):
    y = custom_layer_norm(x, gamma, beta, eps)
    # Save for backward: x, gamma, output
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return y, (x, gamma, mean, var, eps)

def ln_bwd(residuals, g):
    x, gamma, mean, var, eps = residuals
    D = x.shape[-1]
    x_centered = x - mean
    std_inv = 1.0 / jnp.sqrt(var + eps)

    # Gradient w.r.t. x
    dx_norm = g * gamma
    dx = (1.0 / D) * std_inv * (
        D * dx_norm
        - jnp.sum(dx_norm, axis=-1, keepdims=True)
        - x_centered * std_inv**2 * jnp.sum(dx_norm * x_centered, axis=-1, keepdims=True)
    )

    # Gradient w.r.t. gamma
    x_norm = x_centered * std_inv
    dgamma = jnp.sum(g * x_norm, axis=tuple(range(len(g.shape) - 1)))

    # Gradient w.r.t. beta
    dbeta = jnp.sum(g, axis=tuple(range(len(g.shape) - 1)))

    return dx, dgamma, dbeta, None  # None for eps (not differentiable)

custom_layer_norm.defvjp(ln_fwd, ln_bwd)

# Test
B, D = 4, 64
key = jax.random.key(42)
x = jax.random.normal(key, (B, D))
gamma = jnp.ones(D)
beta = jnp.zeros(D)

# Forward
y = custom_layer_norm(x, gamma, beta, eps=1e-5)
print(f"Output mean: {jnp.mean(y):.4f}")  # ~0
print(f"Output std: {jnp.std(y):.4f}")    # ~1

# Backward
def loss_fn(params, x):
    gamma, beta = params
    y = custom_layer_norm(x, gamma, beta)
    return jnp.sum(y ** 2)

grad_fn = jax.grad(loss_fn)
grads = grad_fn((gamma, beta), x)
print(f"Gamma grad shape: {grads[0].shape}")  # (64,)
print(f"Beta grad shape: {grads[1].shape}")    # (64,)

# vmap
x_batch = jax.random.normal(key, (8, B, D))
vmap_ln = jax.vmap(custom_layer_norm, in_axes=(0, None, None, None))
y_batch = vmap_ln(x_batch, gamma, beta, 1e-5)
print(f"Batched output shape: {y_batch.shape}")  # (8, 4, 64)
```

---

## 31.11 Best Practices

### 31.11.1 When to Use FFI

| Use Case | FFI? | Alternative |
|---|---|---|
| Custom CUDA kernels | Yes | Pallas (if possible) |
| Calling existing C++ libraries | Yes | N/A |
| Performance-critical ops not in XLA | Yes | `jax.lax` primitives |
| Simple elementwise ops | No | `jax.vmap`, `jax.numpy` |
| Ops expressible in `jax.lax` | No | Use `jax.lax` directly |

### 31.11.2 Tips

1. **Test the primal first**: Get the forward pass working before adding differentiation
2. **Use Python fallback**: Implement in Python first, then optimize with C++/CUDA
3. **Validate gradients**: Compare FFI gradients with `jax.grad` of the Python implementation
4. **Handle edge cases**: Empty tensors, scalar inputs, dtype mismatches
5. **Profile before optimizing**: Use `jax.profiler` to identify bottlenecks
6. **Keep C++ minimal**: Only the kernel should be in C++; use Python for logic
7. **Document shape constraints**: Clearly document supported shapes and dtypes
