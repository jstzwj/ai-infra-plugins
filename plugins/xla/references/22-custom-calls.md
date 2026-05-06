# XLA Custom Calls (Foreign Function Interface)

This document provides comprehensive documentation about XLA's custom call mechanism, which allows extending XLA with user-defined operations implemented as external functions.

## Table of Contents

- [Overview](#overview)
- [XLA FFI (Foreign Function Interface)](#xla-ffi-foreign-function-interface)
- [FFI Binding Mechanism](#ffi-binding-mechanism)
- [Returning Errors](#returning-errors)
- [Buffer Arguments and Results](#buffer-arguments-and-results)
- [Variadic Arguments](#variadic-arguments)
- [Attributes](#attributes)
- [CPU Custom Call Example](#cpu-custom-call-example)
- [GPU Custom Call Example](#gpu-custom-call-example)
- [Tuple Passing](#tuple-passing)
- [Temp Buffers via Tuple Outputs](#temp-buffers-via-tuple-outputs)

## Overview

XLA's custom call mechanism allows users to define operations that are not natively supported by XLA. Custom calls appear as nodes in the HLO graph and are implemented by user-provided functions that execute at runtime. This is XLA's Foreign Function Interface (FFI), providing a way to integrate hand-written kernels, library calls, or hardware-specific operations into XLA's compilation and execution pipeline.

Custom calls operate in two phases:

1. **Compile-time**: A custom call instruction is placed in the HLO module, specifying the function name, operand shapes, output shapes, and optional attributes. The XLA compiler treats this as an opaque operation and does not attempt to optimize across custom call boundaries.

2. **Runtime**: The named function is looked up in the FFI registry and called with the actual buffer arguments and results. The implementation performs the computation using the provided buffers.

## XLA FFI (Foreign Function Interface)

### Compile-Time: Custom Call in HLO Module

At compile time, a custom call is represented in HLO as a `custom-call` instruction:

```
%result = custom-call(target_name, %operand0, %operand1, ...),
    backend_config="<json or opaque config>",
    has_side_effect={true|false}
```

In MLIR (StableHLO/MHLO), custom calls appear as:

```mlir
%result = stablehlo.custom_call @my_function(%operand0, %operand1)
    {backend_config = "my_config_string"}
    : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
```

### Runtime: Implementation Registration via FFI

At runtime, the FFI handler is registered with XLA. When the custom call instruction is executed, XLA looks up the registered handler and invokes it with the actual buffers.

```cpp
#include "xla/ffi/ffi.h"

// Define the handler
XLA_FFI_DEFINE_HANDLER(MyHandler, MyImplementation,
                        ffi::Ffi::Bind()
                            .Arg<ffi::Buffer<ffi::F32>>()  // first operand
                            .Arg<ffi::Buffer<ffi::F32>>()  // second operand
                            .Ret<ffi::Buffer<ffi::F32>>()  // result
                        );

// Register the handler with the "my_function" name
XLA_FFI_REGISTER_HANDLER(ffi::GetRegistry(), "my_function", MyHandler);
```

## FFI Binding Mechanism

### Template Metaprogramming

The FFI binding mechanism uses C++ template metaprogramming to specify the handler's signature at compile time. This allows the FFI framework to:

1. **Verify types at registration time**: Catch type mismatches before any computation runs.
2. **Generate efficient dispatch code**: Minimal overhead when calling the handler.
3. **Provide type-safe access**: Handlers receive properly typed buffers and attributes.

### Compile-Time Signature Specification

The signature is specified using `ffi::Ffi::Bind()`:

```cpp
// Signature: (F32 buffer, F32 buffer) -> F32 buffer
auto signature = ffi::Ffi::Bind()
    .Arg<ffi::Buffer<ffi::F32>>()   // First argument: F32 buffer
    .Arg<ffi::Buffer<ffi::F32>>()   // Second argument: F32 buffer
    .Ret<ffi::Buffer<ffi::F32>>();  // Result: F32 buffer

// Full handler definition
XLA_FFI_DEFINE_HANDLER(
    MyHandler,
    [](ffi::Buffer<ffi::F32> arg0, ffi::Buffer<ffi::F32> arg1,
       ffi::Result<ffi::Buffer<ffi::F32>> result) -> ffi::Error {
      // Implementation
      return ffi::Error::Success();
    },
    signature);
```

### Minimal Runtime Overhead

The FFI mechanism is designed for minimal runtime overhead:

- **No dynamic dispatch**: The handler signature is resolved at compile time.
- **No memory allocation**: Buffers and results are passed by reference.
- **Inline-able dispatch**: The binding code is simple enough to be inlined by the compiler.
- **Nanosecond-level overhead**: The cost of dispatching through the FFI is measured in nanoseconds, comparable to a virtual function call.

## Returning Errors

### ffi::Error

All FFI handlers return an `ffi::Error` object to indicate success or failure:

```cpp
namespace xla::ffi {

class Error {
 public:
  // Construct from error code and message
  Error(ErrorCode code, std::string message);

  // Success constructor
  static Error Success();

  // Check if error
  explicit operator bool() const;

  // Accessors
  ErrorCode code() const;
  const std::string& message() const;
};

}  // namespace xla::ffi
```

### ErrorCode Enum

The `ErrorCode` enum mirrors XLA's standard error codes:

```cpp
enum class ErrorCode {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
};
```

### Error::Success()

Use `Error::Success()` or return a default-constructed `Error` to indicate success:

```cpp
// All of these indicate success:
return ffi::Error::Success();
return ffi::Error();
return {};
```

For error cases:

```cpp
// Invalid argument error
return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                   "Dimension 0 must be positive");

// Unimplemented error
return ffi::Error(ffi::ErrorCode::kUnimplemented,
                   "F16 type not supported for this operation");

// Internal error
return ffi::Error(ffi::ErrorCode::kInternal,
                   "Unexpected null buffer");
```

## Buffer Arguments and Results

### Destination Passing Style

XLA custom calls use destination passing style: the output buffers are pre-allocated by XLA's buffer assignment, and the handler writes results directly into these buffers. This avoids unnecessary copies and allows XLA to optimize memory allocation.

### AnyBuffer: Generic Buffers

`AnyBuffer` provides access to a buffer without constraining its element type or rank:

```cpp
XLA_FFI_DEFINE_HANDLER(
    GenericHandler,
    [](ffi::AnyBuffer input, ffi::Result<ffi::AnyBuffer> output) -> ffi::Error {
      // Get buffer properties
      auto element_type = input.element_type();
      auto rank = input.rank();
      auto dimensions = input.dimensions();
      auto byte_size = input.byte_size();

      // Access raw data
      void* data = input.data();
      void* out_data = output->data();

      // Must handle all types manually
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());
```

### Buffer<Dtype>: Typed Buffers

`Buffer<Dtype>` constrains the buffer to a specific element type, providing type-safe access:

```cpp
XLA_FFI_DEFINE_HANDLER(
    F32Handler,
    [](ffi::Buffer<ffi::F32> input,
       ffi::Result<ffi::Buffer<ffi::F32>> output) -> ffi::Error {
      // Access typed data
      float* in_data = input.data();
      float* out_data = output->data();

      // Get dimensions (element count is typed)
      int64_t num_elements = input.element_count();

      for (int64_t i = 0; i < num_elements; ++i) {
        out_data[i] = in_data[i] * 2.0f;
      }

      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());
```

Available Dtype specializations:

| Dtype | C++ Type | Description |
|-------|----------|-------------|
| `ffi::F16` | `Eigen::half` | 16-bit floating point |
| `ffi::BF16` | `Eigen::bfloat16` | BFloat16 |
| `ffi::F32` | `float` | 32-bit floating point |
| `ffi::F64` | `double` | 64-bit floating point |
| `ffi::S8` | `int8_t` | Signed 8-bit integer |
| `ffi::S16` | `int16_t` | Signed 16-bit integer |
| `ffi::S32` | `int32_t` | Signed 32-bit integer |
| `ffi::S64` | `int64_t` | Signed 64-bit integer |
| `ffi::U8` | `uint8_t` | Unsigned 8-bit integer |
| `ffi::U16` | `uint16_t` | Unsigned 16-bit integer |
| `ffi::U32` | `uint32_t` | Unsigned 32-bit integer |
| `ffi::U64` | `uint64_t` | Unsigned 64-bit integer |
| `ffi::Pred` | `bool` | Boolean predicate |
| `ffi::C64` | `complex64` | 64-bit complex |
| `ffi::C128` | `complex128` | 128-bit complex |

### BufferR0, BufferR1, BufferR2: Rank-Constrained Buffers

For operations that only make sense at specific ranks, use rank-constrained buffer types:

```cpp
// Scalar buffer (rank 0)
XLA_FFI_DEFINE_HANDLER(
    ScalarHandler,
    [](ffi::BufferR0<ffi::F32> input,
       ffi::Result<ffi::BufferR0<ffi::F32>> output) -> ffi::Error {
      float value = *input.data();  // Single element
      *output->data() = value * 2.0f;
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Ret<ffi::BufferR0<ffi::F32>>());

// 1D buffer (rank 1)
XLA_FFI_DEFINE_HANDLER(
    VectorHandler,
    [](ffi::BufferR1<ffi::F32> input,
       ffi::Result<ffi::BufferR1<ffi::F32>> output) -> ffi::Error {
      int64_t size = input.dimension(0);
      for (int64_t i = 0; i < size; ++i) {
        output->data()[i] = input.data()[i] * 2.0f;
      }
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Ret<ffi::BufferR1<ffi::F32>>());

// 2D buffer (rank 2)
XLA_FFI_DEFINE_HANDLER(
    MatrixHandler,
    [](ffi::BufferR2<ffi::F32> input,
       ffi::Result<ffi::BufferR2<ffi::F32>> output) -> ffi::Error {
      int64_t rows = input.dimension(0);
      int64_t cols = input.dimension(1);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          // Access using (row, col) indexing
          output->data()[r * cols + c] = input.data()[r * cols + c] * 2.0f;
        }
      }
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>());
```

Higher-rank constrained buffers (`BufferR3`, `BufferR4`, etc.) follow the same pattern.

### Result<T> Wrapper

The `Result<T>` wrapper represents an output buffer. It behaves like a pointer to the underlying buffer:

```cpp
// Result<T> dereferences to the underlying Buffer<T>
ffi::Result<ffi::Buffer<ffi::F32>> result;

// Access the buffer
result->data();           // Get the data pointer
result->dimensions();     // Get dimensions
result->element_count();  // Get element count

// The result buffer is pre-allocated; just write into it
float* out = result->data();
```

## Variadic Arguments

### RemainingArgs and RemainingRets

For custom calls that accept a variable number of arguments or return a variable number of results, use `RemainingArgs` and `RemainingRets`:

```cpp
XLA_FFI_DEFINE_HANDLER(
    VariadicHandler,
    [](ffi::RemainingArgs args, ffi::RemainingRets rets) -> ffi::Error {
      // Get the number of arguments
      size_t num_args = args.size();
      size_t num_rets = rets.size();

      // Access each argument
      for (size_t i = 0; i < num_args; ++i) {
        auto arg = args.get<ffi::Buffer<ffi::F32>>(i);
        if (!arg.has_value()) {
          return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                            "All arguments must be F32 buffers");
        }
        // Process arg.value()...
      }

      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .RemainingArgs()
        .RemainingRets());
```

### ErrorOr<T> for Optional Access

When accessing variadic arguments, the `get` method returns an `ErrorOr<T>` that may contain an error if the type does not match:

```cpp
// Safe access with type checking
auto arg = args.get<ffi::Buffer<ffi::F32>>(i);
if (!arg.has_value()) {
  // Handle error: argument i is not an F32 buffer
  return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                     absl::StrFormat("Argument %d must be F32, got %s",
                                     i, arg.error().message()));
}

// Use the buffer
ffi::Buffer<ffi::F32> buffer = arg.value();
```

## Attributes

Attributes provide compile-time constant information to custom call handlers. They are encoded as MLIR `DictionaryAttr` in the HLO module and automatically decoded by the FFI binding.

### Automatic MLIR DictionaryAttr Decoding

The FFI framework automatically decodes MLIR `DictionaryAttr` values into C++ types. Attributes are specified as part of the `ffi::Ffi::Bind()` signature:

```cpp
XLA_FFI_DEFINE_HANDLER(
    AttrHandler,
    [](ffi::Buffer<ffi::F32> input,
       ffi::Result<ffi::Buffer<ffi::F32>> output,
       int32_t num_iters,              // Integer attribute
       std::string_view mode,          // String attribute
       float scale                     // Float attribute
    ) -> ffi::Error {
      for (int32_t i = 0; i < num_iters; ++i) {
        // Use mode and scale
      }
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int32_t>("num_iters")
        .Attr<std::string_view>("mode")
        .Attr<float>("scale"));
```

### Integer Attributes

Integer attributes decode to `int32_t`, `int64_t`, or other integer types:

```cpp
// In the handler signature:
.Attr<int32_t>("tile_size")
.Attr<int64_t>("max_elements")

// In the handler:
// int32_t tile_size = ...;
// int64_t max_elements = ...;
```

### String Attributes

String attributes decode to `std::string_view`:

```cpp
// In the handler signature:
.Attr<std::string_view>("algorithm")

// In the handler:
// std::string_view algorithm = ...;
```

### Float Attributes

Float attributes decode to `float` or `double`:

```cpp
// In the handler signature:
.Attr<float>("learning_rate")
.Attr<double>("epsilon")

// In the handler:
// float learning_rate = ...;
```

### User-Defined Enum Attributes

Use `XLA_FFI_REGISTER_ENUM_ATTR_DECODING` to define custom enum attributes:

```cpp
// Define the enum
enum class ActivationMode {
  kNone,
  kRelu,
  kGelu,
  kSilu,
};

// Register the enum decoder
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(
    ActivationMode,
    {{"none", ActivationMode::kNone},
     {"relu", ActivationMode::kRelu},
     {"gelu", ActivationMode::kGelu},
     {"silu", ActivationMode::kSilu}});

// Use in handler
XLA_FFI_DEFINE_HANDLER(
    ActivationHandler,
    [](ffi::Buffer<ffi::F32> input,
       ffi::Result<ffi::Buffer<ffi::F32>> output,
       ActivationMode mode) -> ffi::Error {
      switch (mode) {
        case ActivationMode::kNone:
          // Copy input to output
          break;
        case ActivationMode::kRelu:
          // Apply ReLU
          break;
        case ActivationMode::kGelu:
          // Apply GELU
          break;
        case ActivationMode::kSilu:
          // Apply SiLU
          break;
      }
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<ActivationMode>("activation_mode"));
```

### User-Defined Struct Attributes

Use `XLA_FFI_REGISTER_STRUCT_ATTR_DECODING` to define custom struct attributes that are decoded from a dictionary:

```cpp
// Define the struct
struct ConvConfig {
  int32_t stride_h;
  int32_t stride_w;
  int32_t padding_h;
  int32_t padding_w;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t groups;
};

// Register the struct decoder
XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    ConvConfig,
    {{"stride_h", &ConvConfig::stride_h},
     {"stride_w", &ConvConfig::stride_w},
     {"padding_h", &ConvConfig::padding_h},
     {"padding_w", &ConvConfig::padding_w},
     {"dilation_h", &ConvConfig::dilation_h},
     {"dilation_w", &ConvConfig::dilation_w},
     {"groups", &ConvConfig::groups}});

// Use in handler
XLA_FFI_DEFINE_HANDLER(
    ConvHandler,
    [](ffi::BufferR4<ffi::F32> input,
       ffi::BufferR4<ffi::F32> kernel,
       ffi::Result<ffi::BufferR4<ffi::F32>> output,
       ConvConfig config) -> ffi::Error {
      // Use config.stride_h, config.stride_w, etc.
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR4<ffi::F32>>()
        .Arg<ffi::BufferR4<ffi::F32>>()
        .Ret<ffi::BufferR4<ffi::F32>>()
        .Attr<ConvConfig>("conv_config"));
```

### Dictionary Access for Lazy Decoding

For complex attribute structures where you want to inspect individual fields on demand, use dictionary access:

```cpp
XLA_FFI_DEFINE_HANDLER(
    LazyAttrHandler,
    [](ffi::Buffer<ffi::F32> input,
       ffi::Result<ffi::Buffer<ffi::F32>> output,
       ffi::Dictionary attrs) -> ffi::Error {
      // Look up attributes on demand
      auto tile_size = attrs.get<int32_t>("tile_size");
      if (!tile_size.has_value()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "Missing 'tile_size' attribute");
      }

      auto algorithm = attrs.get<std::string_view>("algorithm");
      if (!algorithm.has_value()) {
        // Use default
        algorithm = "default";
      }

      // Use tile_size.value() and algorithm.value()
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attrs());
```

## CPU Custom Call Example

Here is a complete example of a CPU custom call that implements an element-wise scaled addition with optional activation:

### Handler Implementation

```cpp
// scaled_add_activation_handler.cc

#include "xla/ffi/ffi.h"
#include <cmath>

namespace my_custom_calls {

enum class Activation { kNone, kRelu, kGelu };

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(
    Activation,
    {{"none", Activation::kNone},
     {"relu", Activation::kRelu},
     {"gelu", Activation::kGelu}});

// Scaled addition with activation: output = activation(alpha * a + beta * b)
XLA_FFI_DEFINE_HANDLER(
    ScaledAddWithActivation,
    [](ffi::Buffer<ffi::F32> a,        // Input A
       ffi::Buffer<ffi::F32> b,        // Input B
       ffi::Result<ffi::Buffer<ffi::F32>> output,  // Output
       float alpha,                     // Scale for A
       float beta,                      // Scale for B
       Activation activation            // Activation function
    ) -> ffi::Error {
      // Validate shapes
      if (a.dimensions() != b.dimensions() || a.dimensions() != output->dimensions()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "All buffers must have the same shape");
      }

      int64_t num_elements = a.element_count();
      const float* a_data = a.data();
      const float* b_data = b.data();
      float* out_data = output->data();

      // Compute scaled sum with activation
      for (int64_t i = 0; i < num_elements; ++i) {
        float val = alpha * a_data[i] + beta * b_data[i];

        switch (activation) {
          case Activation::kNone:
            out_data[i] = val;
            break;
          case Activation::kRelu:
            out_data[i] = val > 0.0f ? val : 0.0f;
            break;
          case Activation::kGelu: {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float cdf = 0.5f * (1.0f + std::tanh(0.7978845608f * (val + 0.044715f * val * val * val)));
            out_data[i] = val * cdf;
            break;
          }
        }
      }

      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()     // a
        .Arg<ffi::Buffer<ffi::F32>>()     // b
        .Ret<ffi::Buffer<ffi::F32>>()     // output
        .Attr<float>("alpha")             // alpha
        .Attr<float>("beta")              // beta
        .Attr<Activation>("activation")); // activation

// Register the handler
XLA_FFI_REGISTER_HANDLER(ffi::GetRegistry(),
                          "scaled_add_with_activation",
                          ScaledAddWithActivation);

}  // namespace my_custom_calls
```

### Using from JAX

```python
import jax
import jax.numpy as jnp
from jax import ffi

# Register the custom call target (in practice, this would be loaded from a shared library)
# For this example, assume the handler is compiled into the XLA runtime

# Define the JAX FFI wrapper
@ffi.register_ffi_target("scaled_add_with_activation")
def scaled_add_with_activation(a, b, *, alpha, beta, activation):
    """Scaled addition with activation: activation(alpha * a + beta * b)"""
    # The actual implementation would call into the C++ handler
    pass

# Use in JAX
def my_function(a, b):
    return scaled_add_with_activation(a, b, alpha=1.0, beta=1.0, activation="relu")

# Compile and run
a = jnp.ones((4, 4))
b = jnp.ones((4, 4)) * 0.5
result = my_function(a, b)
# result = relu(1.0 * 1.0 + 1.0 * 0.5) = relu(1.5) = 1.5
```

## GPU Custom Call Example

Here is a complete example of a GPU custom call with a CUDA kernel for vector addition:

### CUDA Kernel

```cpp
// vector_add_kernel.cu

__global__ void VectorAddKernel(const float* a, const float* b,
                                 float* output, int64_t num_elements,
                                 float alpha, float beta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = alpha * a[idx] + beta * b[idx];
  }
}
```

### GPU Handler Implementation

```cpp
// gpu_vector_add_handler.cc

#include "xla/ffi/ffi.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

// Forward declaration of the CUDA kernel launcher
void LaunchVectorAddKernel(stream_executor::Stream* stream,
                            const float* a, const float* b,
                            float* output, int64_t num_elements,
                            float alpha, float beta);

namespace my_gpu_custom_calls {

XLA_FFI_DEFINE_HANDLER(
    GpuVectorAdd,
    [](ffi::Buffer<ffi::F32> a,
       ffi::Buffer<ffi::F32> b,
       ffi::Result<ffi::Buffer<ffi::F32>> output,
       float alpha,
       float beta,
       ffi::Stream stream) -> ffi::Error {
      // Validate shapes
      if (a.dimensions() != b.dimensions() || a.dimensions() != output->dimensions()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "All buffers must have the same shape");
      }

      int64_t num_elements = a.element_count();

      // Get the GPU stream
      auto* gpu_stream = se::gpu::AsGpuStreamValue(stream);

      // Launch the CUDA kernel
      LaunchVectorAddKernel(gpu_stream,
                            a.data(), b.data(), output->data(),
                            num_elements, alpha, beta);

      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("alpha")
        .Attr<float>("beta")
        .Stream());  // Include the stream for GPU operations

XLA_FFI_REGISTER_HANDLER(ffi::GetRegistry(),
                          "gpu_vector_add",
                          GpuVectorAdd);

}  // namespace my_gpu_custom_calls
```

### CUDA Kernel Launcher

```cpp
// gpu_vector_add_launcher.cu

#include <cuda_runtime.h>
#include "xla/stream_executor/gpu/gpu_stream.h"

// The actual CUDA kernel
__global__ void VectorAddKernel(const float* a, const float* b,
                                 float* output, int64_t num_elements,
                                 float alpha, float beta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = alpha * a[idx] + beta * b[idx];
  }
}

void LaunchVectorAddKernel(stream_executor::Stream* stream,
                            const float* a, const float* b,
                            float* output, int64_t num_elements,
                            float alpha, float beta) {
  // Compute grid and block dimensions
  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  // Get the CUDA stream from the executor stream
  cudaStream_t cuda_stream = se::gpu::AsGpuStreamValue(stream);

  // Launch the kernel
  VectorAddKernel<<<grid_size, block_size, 0, cuda_stream>>>(
      a, b, output, num_elements, alpha, beta);
}
```

## Tuple Passing

Custom calls can accept and return tuples, which are represented as nested buffers. This is useful for operations that return multiple results or accept complex input structures.

### Receiving Tuple Arguments

```cpp
XLA_FFI_DEFINE_HANDLER(
    TupleInputHandler,
    [](ffi::AnyBuffer tuple_input,
       ffi::Result<ffi::Buffer<ffi::F32>> output) -> ffi::Error {
      // A tuple buffer contains pointers to its element buffers
      // The tuple's "data" is an array of device pointers

      // For a tuple of (buffer_a, buffer_b):
      // tuple_input.data() points to an array of 2 device pointers
      void** tuple_ptrs = reinterpret_cast<void**>(tuple_input.data());

      // Access individual elements
      float* a_data = reinterpret_cast<float*>(tuple_ptrs[0]);
      float* b_data = reinterpret_cast<float*>(tuple_ptrs[1]);

      // Process...
      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::Buffer<ffi::F32>>());
```

### Returning Tuple Results

To return a tuple from a custom call, define multiple results:

```cpp
XLA_FFI_DEFINE_HANDLER(
    TupleOutputHandler,
    [](ffi::Buffer<ffi::F32> input,
       ffi::Result<ffi::Buffer<ffi::F32>> output_a,
       ffi::Result<ffi::Buffer<ffi::F32>> output_b) -> ffi::Error {
      int64_t n = input.element_count();
      const float* in_data = input.data();
      float* a_data = output_a->data();
      float* b_data = output_b->data();

      for (int64_t i = 0; i < n; ++i) {
        a_data[i] = in_data[i] * 2.0f;  // Double
        b_data[i] = in_data[i] * 0.5f;  // Half
      }

      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()   // First result
        .Ret<ffi::Buffer<ffi::F32>>()); // Second result
```

### HLO Representation

The HLO for tuple passing looks like:

```
// Tuple input
%tuple = (f32[4], f32[4]) tuple(%a, %b)
%result = custom-call(%tuple), custom_call_target="tuple_input_handler"

// Tuple output
%results:f32[4], f32[4] = custom-call(%input), custom_call_target="tuple_output_handler"
%tuple = (f32[4], f32[4]) tuple(%results.0, %results.1)
```

## Temp Buffers via Tuple Outputs

Custom calls can request temporary workspace buffers that are allocated by XLA and passed to the handler. This is done by including extra buffer results in the output tuple that are used only during computation.

### Mechanism

1. **Define the output shape**: Include the temporary buffer as an additional output in the custom call's output shape tuple.

2. **Write into temp buffers**: The handler writes intermediate results into the temp buffers during computation.

3. **Read from temp buffers**: The handler reads from the temp buffers to produce the final output.

4. **XLA manages lifetime**: XLA's buffer assignment handles the lifetime of temp buffers automatically.

### Example

```cpp
// Custom call that needs a temporary buffer for intermediate results
XLA_FFI_DEFINE_HANDLER(
    NeedsTempBuffer,
    [](ffi::Buffer<ffi::F32> input,
       ffi::Result<ffi::Buffer<ffi::F32>> temp,     // Temporary workspace
       ffi::Result<ffi::Buffer<ffi::F32>> output     // Final output
    ) -> ffi::Error {
      int64_t n = input.element_count();
      const float* in_data = input.data();
      float* temp_data = temp->data();
      float* out_data = output->data();

      // Phase 1: Compute intermediate results into temp buffer
      for (int64_t i = 0; i < n; ++i) {
        temp_data[i] = std::sqrt(in_data[i]);  // Intermediate computation
      }

      // Phase 2: Use temp buffer to compute final output
      for (int64_t i = 0; i < n; ++i) {
        out_data[i] = temp_data[i] * temp_data[i];  // Final computation
      }

      return ffi::Error::Success();
    },
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()   // temp (workspace)
        .Ret<ffi::Buffer<ffi::F32>>()); // output
```

### HLO with Temp Buffer

```
HloModule temp_buffer_example

ENTRY main {
  %input = f32[1024] parameter(0)
  %result:f32[1024], f32[1024] = custom-call(%input),
      custom_call_target="needs_temp_buffer",
      output_tuple={f32[1024], f32[1024]}
  %output = f32[1024] get-tuple-element(%result), index=1
  ROOT %root = (f32[1024]) tuple(%output)
}
```

### Best Practices for Temp Buffers

1. **Minimize temp buffer size**: Only request the minimum amount of workspace needed.
2. **Reuse temp buffers**: If multiple custom calls need similar workspace sizes, XLA may alias them.
3. **Document workspace requirements**: Clearly document the required temp buffer size as a function of the input size.
4. **Consider in-place operations**: Before requesting a temp buffer, check if the computation can be done in-place in the output buffer.
5. **Alignment**: Ensure your kernel respects the alignment requirements of the target hardware.
