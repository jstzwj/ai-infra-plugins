# XLA Operation Semantics: I/O and Miscellaneous Operations

This reference provides comprehensive documentation of XLA's I/O operations, control-flow helpers, random number generation, and miscellaneous operations. These operations form the bridge between the pure functional computation graph and the outside world (devices, host memory, inter-device communication), and they provide essential utilities for advanced program construction.

---

## 11.1 CustomCall

`CustomCall` is XLA's Foreign Function Interface (FFI) mechanism. It allows embedding arbitrary host or device code inside an XLA computation graph, enabling users to invoke handwritten kernels, library calls, or any operation not natively represented in the HLO instruction set.

### 11.1.1 High-Level Interface

At the HLO level, a `CustomCall` instruction specifies:

- A **call target name** (string) identifying the function to invoke.
- A list of **operand shapes** and a **result shape**.
- A **backend_config** string or proto that the target backend interprets.
- Optional **has_side_effect** flag (defaults to `false`).
- Optional **literal** providing a constant backend config.

```
%result = custom-call(%operand0, %operand1, ...),
    call_target_name="my_custom_kernel",
    has_side_effect=false,
    backend_config="{...}",
    shape=f32[128,256]
```

### 11.1.2 XLA FFI Binding Mechanism (C++ API)

The modern XLA FFI (Foreign Function Interface) provides a type-safe, ergonomic C++ API for defining custom call targets. The key components are:

#### Handler Signature

An FFI handler is a function (or callable object) that accepts typed arguments and returns a status or result:

```cpp
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

// Simple handler: no arguments, no results
ffi::Status NoOpHandler() {
  return ffi::Status::Ok();
}

// Handler with typed buffer arguments and results
ffi::Status AddVectors(ffi::Buffer<ffi::F32> a,
                       ffi::Buffer<ffi::F32> b,
                       ffi::Result<ffi::Buffer<ffi::F32>> out) {
  // Implementation goes here
  return ffi::Status::Ok();
}
```

#### Registration Macro

Handlers are registered with XLA using the `XLA_FFI_REGISTER_HANDLER` macro:

```cpp
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiRegistry(),
                         NoOpHandler,
                         "no_op",
                         ffi::Ffi::CallStack()
                             .Ret(ffi::Ffi::RetType::kBuffer));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiRegistry(),
                         AddVectors,
                         "add_vectors",
                         ffi::Ffi::CallStack()
                             .Arg(ffi::Ffi::ArgType::kBuffer)
                             .Arg(ffi::Ffi::ArgType::kBuffer)
                             .Ret(ffi::Ffi::RetType::kBuffer));
```

### 11.1.3 Buffer Arguments and Results

The FFI provides a hierarchy of buffer types that balance type safety and flexibility:

| Type | Description |
|------|-------------|
| `AnyBuffer` | Accepts a buffer of any dtype and any rank. Provides raw pointer access. |
| `Buffer<DType>` | Accepts a buffer of a specific dtype but any rank. E.g., `Buffer<F32>`. |
| `BufferR1<DType>` | Accepts a 1-D buffer of a specific dtype. |
| `BufferR2<DType>` | Accepts a 2-D buffer (matrix) of a specific dtype. |
| `BufferR3<DType>` | Accepts a 3-D buffer of a specific dtype. |
| `BufferR4<DType>` | Accepts a 4-D buffer of a specific dtype. |

All buffer types expose:

- `.data()` -- returns a typed pointer (or `void*` for `AnyBuffer`).
- `.dimensions()` -- returns an `absl::Span<const int64_t>` of dimension sizes.
- `.element_count()` -- total number of elements.
- `.byte_size()` -- total size in bytes.

Example with rank-2 typed buffers:

```cpp
ffi::Status MatMulHandler(ffi::BufferR2<ffi::F32> lhs,
                          ffi::BufferR2<ffi::F32> rhs,
                          ffi::Result<ffi::BufferR2<ffi::F32>> out) {
  int64_t M = lhs.dimensions()[0];
  int64_t K = lhs.dimensions()[1];
  int64_t N = rhs.dimensions()[1];

  const float* a = lhs.data();
  const float* b = rhs.data();
  float* c = out->data();

  // Naive matrix multiply for illustration
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = sum;
    }
  }
  return ffi::Status::Ok();
}
```

#### Result Buffers

Result buffers use the `ffi::Result<T>` wrapper. The `Result` object is default-constructed by the FFI framework and points to a pre-allocated output buffer. The handler writes into the result buffer via `result->data()`.

```cpp
// Single result
ffi::Result<ffi::Buffer<ffi::F32>> out;

// Multiple results (multiple Result parameters)
ffi::Status SplitHandler(ffi::Buffer<ffi::F32> input,
                         ffi::Result<ffi::Buffer<ffi::F32>> out0,
                         ffi::Result<ffi::Buffer<ffi::F32>> out1);
```

### 11.1.4 Constrained Buffer Arguments

Sometimes a handler needs to accept buffers of a specific dtype but leave the rank unconstrained, or vice versa. The FFI provides constrained buffer types for this purpose:

```cpp
// Only constraint: must be F32, rank is free
ffi::Status ProcessF32Data(ffi::Buffer<ffi::F32> data);

// Constraint on layout: must be row-major (checked at call time)
// Specified via the CustomCall target metadata
```

Constraints are validated at call time. If a caller passes a buffer that violates the constraint (wrong dtype, wrong rank), the FFI returns an error status.

### 11.1.5 Variadic Arguments (RemainingArgs and RemainingRets)

For custom calls that accept a variable number of arguments or return a variable number of results, the FFI provides `RemainingArgs` and `RemainingRets`:

```cpp
ffi::Status ConcatMany(ffi::RemainingArgs inputs,
                       ffi::Result<ffi::RemainingRets> outputs) {
  // RemainingArgs supports indexed access
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto buf = inputs.at<ffi::Buffer<ffi::F32>>(i);
    if (!buf.has_value()) {
      return ffi::Status::InvalidArgument("All inputs must be F32 buffers");
    }
    // Process buf->data(), buf->dimensions(), etc.
  }

  // RemainingRets supports indexed access for writing
  for (size_t i = 0; i < outputs->size(); ++i) {
    auto ret = outputs->at<ffi::Buffer<ffi::F32>>(i);
    // Write to ret->data()
  }

  return ffi::Status::Ok();
}
```

Registration with variadic arguments:

```cpp
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiRegistry(),
                         ConcatMany,
                         "concat_many",
                         ffi::Ffi::CallStack()
                             .RemainingArgs()
                             .RemainingRets());
```

`RemainingArgs` API:

| Method | Description |
|--------|-------------|
| `size()` | Number of remaining arguments |
| `at<T>(index)` | Access argument at index as type T; returns `std::optional<T>` |
| `empty()` | Whether there are zero remaining arguments |

`RemainingRets` API (via `Result<RemainingRets>`):

| Method | Description |
|--------|-------------|
| `size()` | Number of remaining results |
| `at<T>(index)` | Access result buffer at index as type T |

### 11.1.6 Attributes

Attributes provide a mechanism to pass non-buffer metadata to custom call handlers. They correspond to key-value pairs in the `custom_call_schedule` or `backend_config`.

#### Scalar Attributes

```cpp
ffi::Status ScaledAddHandler(ffi::Buffer<ffi::F32> x,
                             ffi::Buffer<ffi::F32> y,
                             ffi::Result<ffi::Buffer<ffi::F32>> out,
                             ffi::Attribute<float> alpha,
                             ffi::Attribute<float> beta) {
  float a = *alpha;  // Dereference to get the value
  float b = *beta;
  // out = alpha * x + beta * y
  return ffi::Status::Ok();
}
```

Supported scalar attribute types:

| C++ Type | FFI Type | Description |
|----------|----------|-------------|
| `int32_t` | `ffi::Attribute<int32_t>` | 32-bit signed integer |
| `int64_t` | `ffi::Attribute<int64_t>` | 64-bit signed integer |
| `float` | `ffi::Attribute<float>` | 32-bit floating point |
| `double` | `ffi::Attribute<double>` | 64-bit floating point |
| `bool` | `ffi::Attribute<bool>` | Boolean |

#### String Attributes

```cpp
ffi::Status KernelDispatch(ffi::Buffer<ffi::F32> input,
                           ffi::Result<ffi::Buffer<ffi::F32>> output,
                           ffi::Attribute<std::string_view> kernel_name) {
  std::string_view name = *kernel_name;
  // Dispatch to the named kernel
  return ffi::Status::Ok();
}
```

#### Enum Attributes

Enum attributes allow mapping string values to C++ enum types:

```cpp
enum class ActivationMode {
  kNone,
  kRelu,
  kGelu,
  kSilu,
};

// Register the enum decoding
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(ffi::GetXlaFfiRegistry(),
                                    ActivationMode,
                                    {{"none", ActivationMode::kNone},
                                     {"relu", ActivationMode::kRelu},
                                     {"gelu", ActivationMode::kGelu},
                                     {"silu", ActivationMode::kSilu}});

// Use in handler
ffi::Status ApplyActivation(ffi::Buffer<ffi::F32> input,
                            ffi::Result<ffi::Buffer<ffi::F32>> output,
                            ffi::Attribute<ActivationMode> mode) {
  switch (*mode) {
    case ActivationMode::kRelu:
      // Apply ReLU
      break;
    case ActivationMode::kGelu:
      // Apply GELU
      break;
    // ...
  }
  return ffi::Status::Ok();
}
```

`XLA_FFI_REGISTER_ENUM_ATTR_DECODING` takes the registry, the enum type, and an initializer list of `{string_value, enum_value}` pairs. At call time, the string attribute value is decoded to the enum. If the string does not match any registered value, an error is returned.

#### Struct Attributes

Struct attributes allow grouping multiple related attributes into a single typed object:

```cpp
struct ConvolutionConfig {
  int64_t batch_size;
  int64_t in_channels;
  int64_t out_channels;
  std::array<int64_t, 2> kernel_size;
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> padding;
};

// Register the struct decoding
XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    ffi::GetXlaFfiRegistry(), ConvolutionConfig,
    ffi::StructField<"batch_size", &ConvolutionConfig::batch_size>,
    ffi::StructField<"in_channels", &ConvolutionConfig::in_channels>,
    ffi::StructField<"out_channels", &ConvolutionConfig::out_channels>,
    ffi::StructField<"kernel_size", &ConvolutionConfig::kernel_size>,
    ffi::StructField<"stride", &ConvolutionConfig::stride>,
    ffi::StructField<"padding", &ConvolutionConfig::padding>);

// Use in handler
ffi::Status ConvHandler(ffi::BufferR4<ffi::F32> input,
                        ffi::BufferR4<ffi::F32> kernel,
                        ffi::Result<ffi::BufferR4<ffi::F32>> output,
                        ffi::Attribute<ConvolutionConfig> config) {
  int64_t bs = config->batch_size;
  // Use config->in_channels, config->stride, etc.
  return ffi::Status::Ok();
}
```

`XLA_FFI_REGISTER_STRUCT_ATTR_DECODING` takes the registry, the struct type, and a set of `StructField` descriptors. Each `StructField` maps a named attribute key to a struct member pointer. The FFI framework automatically populates the struct from the attribute dictionary.

Struct fields support nested arrays via `std::array<T, N>`. Each element of the array is read from `"key.0"`, `"key.1"`, ..., `"key.N-1"` attribute keys.

### 11.1.7 Creating Custom Calls on CPU

On the CPU backend, custom calls execute on the host thread that drives XLA execution. The handler function receives raw pointers into the XLA-allocated buffers:

```cpp
// CPU custom call using the C API (legacy)
extern "C" void cpu_custom_add(float* out, const float* a, const float* b,
                                int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    out[i] = a[i] + b[i];
  }
}
```

The CPU backend passes arguments in order: output pointers first, then input pointers, then the `backend_config` as a `const char*`. However, the modern FFI API abstracts this away.

Using JAX to create a CPU custom call:

```python
from jax import lax
import jax.numpy as jnp

# Define the custom call target
def cpu_add(x, y):
    return lax.custom_call(
        target_name="cpu_custom_add",
        operands=[x, y],
        shape=x.shape,           # Output shape same as input
        has_side_effect=False,
        backend_config=b"",
    )

x = jnp.ones(1024, dtype=jnp.float32)
y = jnp.ones(1024, dtype=jnp.float32) * 2.0
result = cpu_add(x, y)
```

### 11.1.8 Creating Custom Calls on GPU

On the GPU backend, custom call handlers are launched as device kernels. The handler receives device pointers and may use CUDA/HIP intrinsics or launch separate kernels:

```cpp
// GPU custom call using XLA FFI
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

#if defined(__CUDA__)
__global__ void AddKernel(const float* a, const float* b, float* out,
                          int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}
#endif

ffi::Status GpuAddHandler(ffi::Buffer<ffi::F32> a,
                          ffi::Buffer<ffi::F32> b,
                          ffi::Result<ffi::Buffer<ffi::F32>> out) {
  int64_t n = a.element_count();
  const float* a_ptr = a.data();
  const float* b_ptr = b.data();
  float* out_ptr = out->data();

#if defined(__CUDA__)
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  AddKernel<<<grid_size, block_size, 0, ffi::GetCurrentStream()>>>(
      a_ptr, b_ptr, out_ptr, n);
#endif

  return ffi::Status::Ok();
}

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiRegistry(),
                         GpuAddHandler,
                         "gpu_custom_add",
                         ffi::Ffi::CallStack()
                             .Arg(ffi::Ffi::ArgType::kBuffer)
                             .Arg(ffi::Ffi::ArgType::kBuffer)
                             .Ret(ffi::Ffi::RetType::kBuffer));
```

The GPU FFI provides access to the current stream via `ffi::GetCurrentStream()`, which returns the stream (e.g., `CUstream` or `hipStream_t`) on which the custom call should enqueue its work.

### 11.1.9 Passing Tuples to Custom Calls

Custom calls can accept tuple operands and produce tuple results. A tuple in XLA is represented as a tree of buffers. When passing a tuple to a custom call, the FFI framework flattens the tuple tree and presents each element buffer individually:

```python
# JAX: passing a tuple to a custom call
def custom_call_with_tuple(x_tuple, y):
    # x_tuple is a tuple (a, b) of arrays
    result = lax.custom_call(
        target_name="tuple_handler",
        operands=[x_tuple[0], x_tuple[1], y],
        shape=jax.ShapeDtypeStruct(shape=(128,), dtype=jnp.float32),
    )
    return result
```

At the HLO level, this is represented as:

```
%a = f32[128] parameter(0)
%b = f32[128] parameter(1)
%y = f32[128] parameter(2)
%tuple = (f32[128], f32[128]) tuple(%a, %b)
%result = f32[128] custom-call(%tuple, %y),
    call_target_name="tuple_handler"
```

The FFI handler for GPU/CPU receives the flattened buffer pointers:

```cpp
ffi::Status TupleHandler(ffi::Buffer<ffi::F32> tuple_elem0,
                         ffi::Buffer<ffi::F32> tuple_elem1,
                         ffi::Buffer<ffi::F32> y,
                         ffi::Result<ffi::Buffer<ffi::F32>> out) {
  // tuple_elem0 and tuple_elem1 are the elements of the tuple operand
  // y is the second operand
  return ffi::Status::Ok();
}
```

### 11.1.10 Tuple Outputs as Temp Buffers

Custom calls can produce tuple outputs, which are useful for returning multiple results or for allocating temporary scratch buffers:

```python
from jax import lax

# Custom call returning a tuple of (result, scratch)
result_shapes = [
    jax.ShapeDtypeStruct(shape=(256,), dtype=jnp.float32),  # result
    jax.ShapeDtypeStruct(shape=(1024,), dtype=jnp.float32), # temp buffer
]

result = lax.custom_call(
    target_name="handler_with_temp",
    operands=[x],
    shape=jax.core.ShapedArray(
        shape=(),
        dtype=jnp.float32,
    ),
    # For tuple outputs, use result_shapes with custom_call_v2 or
    # specify output_tuple_shape
)
```

At the HLO level:

```
%result = (f32[256], f32[1024]) custom-call(%x),
    call_target_name="handler_with_temp"
%output = f32[256] get-tuple-element(%result), index=0
// index=1 is the temp buffer, discarded
```

### 11.1.11 API Versioning

The XLA FFI is versioned to allow backward-compatible evolution. The `custom_call` API has gone through several versions:

| Version | Key Changes |
|---------|-------------|
| Version 0 | Original C API: `void (*)(void** out, void** in, const char* config)` |
| Version 1 | Added dimension arrays passed as arguments |
| Version 2 | Status return support, typed API |
| FFI (current) | Full C++ type-safe API with attributes, enums, structs |

The `custom_call_target_version` attribute in the `backend_config` or module metadata specifies which ABI version the handler expects. When writing new handlers, prefer the FFI API (latest version).

Custom call versioning is important for library authors who need to support multiple XLA versions. The `XLA_FFI_DEFINE_HANDLER` macro can include a version specification:

```cpp
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiRegistry(),
                         MyHandler,
                         "my_handler",
                         ffi::Ffi::CallStack()
                             .Arg(ffi::Ffi::ArgType::kBuffer)
                             .Ret(ffi::Ffi::RetType::kBuffer))
    .Version(2);
```

---

## 11.2 Infeed

`Infeed` reads data from the host into the XLA computation. It is a side-effecting operation that transfers a value of a specified shape from the host's infeed queue into the device.

### Signature

```
%result = infeed(shape=T), token=%token
```

| Field | Type | Description |
|-------|------|-------------|
| `token` | `token` | Input token for ordering with other side-effecting ops |
| Result | `T` | The data read from the infeed queue |
| Token result | `token` | Output token for subsequent side-effecting ops |

### Semantics

- The operation blocks until data is available on the infeed queue.
- The data transferred must match the shape specified in the instruction.
- Infeed is ordered relative to other infeed/outfeed operations via the token chain.
- The layout of the transferred data is determined by the shape on the instruction.
- On multi-device systems, each device has its own infeed queue unless otherwise configured.

### Example (HLO)

```
ENTRY %main {
  %tok0 = token after-all()
  %data = f32[128,128] infeed(), token=%tok0
  %tok1 = token get-tuple-element(%data), index=1
  %result = f32[128,128] get-tuple-element(%data), index=0
  ROOT %out = (f32[128,128], token) tuple(%result, %tok1)
}
```

The infeed instruction returns a tuple `(T, token)` where `T` is the data and the second element is the output token.

### Host-Side API (JAX)

```python
import jax
import jax.numpy as jnp

# In JAX, infeed is typically used via the transfer_to_device
# mechanisms, but the underlying XLA operation can be accessed
# through the HLO builder.
```

---

## 11.3 Outfeed

`Outfeed` transfers data from the XLA computation to the host. It is a side-effecting operation that writes a value to the host's outfeed queue.

### Signature

```
%new_token = outfeed(%operand), token=%token
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | The data to write to the outfeed queue |
| `token` | `token` | Input token |
| Result | `token` | Output token |

### Semantics

- The operation writes `operand` to the outfeed queue and produces a new token.
- The host must be ready to consume the data; otherwise, the device may stall.
- Outfeed is ordered relative to other infeed/outfeed operations via the token chain.
- The layout of the data on the outfeed queue is determined by the operand's shape layout.

### Example (HLO)

```
ENTRY %main {
  %tok0 = token after-all()
  %data = f32[64] custom-call(...), call_target_name="generate_data"
  %tok1 = token outfeed(%data), token=%tok0
  ROOT %result = token after-all(%tok1)
}
```

---

## 11.4 Send and Recv

`Send` and `Recv` are collective communication operations for point-to-point data transfer between devices. They use a `ChannelHandle` to identify the communication endpoint.

### 11.4.1 ChannelHandle

A `ChannelHandle` identifies a communication channel. It consists of:

| Field | Type | Description |
|-------|------|-------------|
| `handle` | `int64` | Unique identifier for the channel |
| `type` | `ChannelHandle::ChannelType` | The type of channel |

Channel types:

| Type | Value | Description |
|------|-------|-------------|
| `CHANNEL_TYPE_INVALID` | 0 | Invalid channel |
| `DEVICE_TO_DEVICE` | 1 | Transfer between devices on the same host |
| `DEVICE_TO_HOST` | 2 | Transfer from device to host |
| `HOST_TO_DEVICE` | 3 | Transfer from host to device |

### 11.4.2 Send

```
%new_token = send(%operand), channel_id=1,
    is_host_transfer=false, token=%token
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Data to send |
| `token` | `token` | Input token |
| `channel_id` | `int64` | Channel handle identifier |
| `is_host_transfer` | `bool` | Whether sending to host |
| Result | `token` | Output token |

The send operation returns a tuple `(token, send-done-token)` in its non-blocking form. The blocking form (`Send` without `SendDone`) blocks until the transfer is complete.

### 11.4.3 Recv

```
%result_and_token = recv(), channel_id=1,
    is_host_transfer=false, token=%token
```

| Field | Type | Description |
|-------|------|-------------|
| `token` | `token` | Input token |
| `channel_id` | `int64` | Channel handle identifier |
| `is_host_transfer` | `bool` | Whether receiving from host |
| Result | `(T, token)` | Tuple of received data and output token |

The non-blocking form splits into `Recv` (start) and `RecvDone` (wait for completion).

### 11.4.4 Frontend Attributes

Send and Recv operations can carry frontend-specific attributes as key-value string pairs. These are opaque to XLA and are used by higher-level frameworks (JAX, TensorFlow) to encode additional semantics:

```python
# Frontend attributes passed through the XLA computation
frontend_attributes = {
    "_xla_send_recv_target": "device_0_to_device_1",
    "_xla_channel_layout_hint": "major_to_minor={1,0}",
}
```

Common frontend attributes:

| Attribute | Description |
|-----------|-------------|
| `_xla_send_recv_target` | Hint for the target device of the transfer |
| `_xla_channel_layout_hint` | Suggested layout for data transfer |
| `_xla_partition_id` | Partition ID for SPMD partitioning |

### 11.4.5 Send/Recv Example (HLO)

```
ENTRY %main {
  %tok0 = token after-all()

  // Device-to-device send
  %send_tok = token send(%data), channel_id=42,
      is_host_transfer=false, token=%tok0

  // Device-to-device recv
  %recv = (f32[256], token) recv(), channel_id=43,
      is_host_transfer=false, token=%tok0
  %recv_data = f32[256] get-tuple-element(%recv), index=0
  %recv_tok = token get-tuple-element(%recv), index=1

  // Synchronize tokens
  ROOT %tok_final = token after-all(%send_tok, %recv_tok)
}
```

---

## 11.5 Token Operations (AfterAll)

### AfterAll

`AfterAll` creates a token that is ordered after all input tokens. It is used to join multiple token chains.

#### Signature

```
%token = after-all(%token0, %token1, ..., %tokenN)
```

| Field | Type | Description |
|-------|------|-------------|
| operands | `token, ...` | Zero or more input tokens |
| Result | `token` | A token ordered after all inputs |

#### Semantics

- With zero operands, `after-all()` produces a token with no ordering constraints (a "fresh" token).
- With one or more operands, the resulting token is ordered after all input tokens, enforcing that all side effects from the input tokens' chains complete before any operation using the output token.
- `AfterAll` is the primary mechanism for merging multiple I/O or communication chains into one.

#### Example

```
// Two independent infeed chains
%tok0 = token after-all()
%tok1 = token after-all()

%infeed1 = (f32[64], token) infeed(), token=%tok0
%infeed2 = (f32[64], token) infeed(), token=%tok1

%data1 = f32[64] get-tuple-element(%infeed1), index=0
%tok1_out = token get-tuple-element(%infeed1), index=1
%data2 = f32[64] get-tuple-element(%infeed2), index=0
%tok2_out = token get-tuple-element(%infeed2), index=1

// Merge: both infeeds must complete before outfeed
%tok_merged = token after-all(%tok1_out, %tok2_out)
%result = f32[64] add(%data1, %data2)
%tok_final = token outfeed(%result), token=%tok_merged
```

---

## 11.6 GetDimensionSize

`GetDimensionSize` returns the size of a specified dimension of its operand as a scalar `u32` value.

### Signature

```
%size = u32[] get-dimension-size(%operand), dimension=d
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Input array |
| `dimension` | `int64` | The dimension index to query |
| Result | `u32[]` | Scalar containing the size of dimension `d` |

### Semantics

- The dimension must be in range `[0, rank(operand))`.
- For static-shape arrays, this returns a compile-time constant.
- For dynamically-shaped arrays (where `SetDimensionSize` was used), this returns the runtime size of the dimension.
- The result is always a scalar `u32`.

### Example

```
%x = f32[3, 4, 5] parameter(0)
%d0 = u32[] get-dimension-size(%x), dimension=0   // returns 3
%d1 = u32[] get-dimension-size(%x), dimension=1   // returns 4
%d2 = u32[] get-dimension-size(%x), dimension=2   // returns 5
```

---

## 11.7 SetDimensionSize

`SetDimensionSize` sets the effective (dynamic) size of a dimension. The underlying buffer is not resized; elements beyond the new bound are considered padding and are ignored by subsequent operations.

### Signature

```
%result = T set-dimension-size(%operand, %size), dimension=d
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Input array |
| `size` | `u32[]` or `s32[]` | New size for the dimension |
| `dimension` | `int64` | The dimension index to resize |
| Result | `T` | Array with the same static shape but dynamic dimension size |

### Semantics

- The static shape of the result is identical to the operand's static shape. Only the dynamic bound changes.
- `size` must be less than or equal to the static dimension size.
- Operations consuming the result will see only the first `size` elements along `dimension`.
- This is the primary mechanism for implementing dynamic shapes in XLA.

### Example

```
%x = f32[10] parameter(0)          // statically 10 elements
%n = u32[] constant(5)             // want only 5 elements
%y = f32[10] set-dimension-size(%x, %n), dimension=0
// %y has static shape f32[10], but dynamic dimension size 5
// Operations on %y will treat it as having 5 elements

%z = f32[5] dynamic-slice(%y), ... // uses only first 5 elements
%s = u32[] get-dimension-size(%y), dimension=0  // returns 5
```

---

## 11.8 CreateToken

`CreateToken` creates a token value. In practice, `AfterAll` with zero arguments is the standard way to create a fresh token, and `CreateToken` is rarely used directly. Some backends may provide it as an explicit instruction.

### Signature

```
%token = create-token()
```

### Semantics

- Produces a token with no predecessor constraints.
- Equivalent to `after-all()` (with no operands).

---

## 11.9 OptimizerBarrier

`OptimizerBarrier` prevents the XLA optimizer from moving operations across it. All operations that are data dependencies of the barrier must execute before the barrier, and all operations that depend on the barrier result must execute after it.

### Signature

```
%result = T optimizer-barrier(%operand)
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Input value |
| Result | `T` | The same value, passthrough |

### Semantics

- The result is identical to the operand (identity function).
- The barrier prevents XLA from reordering operations across it during optimization passes.
- Useful for debugging or for enforcing specific execution order for performance measurement.
- Has no effect on the numerical result.

### Example

```
%a = f32[128] parameter(0)
%b = f32[128] dot(...), lhs=%a, rhs=%weight
%c = f32[128] optimizer-barrier(%b)  // prevents fusion/Reordering of operations using %b
%d = f32[128] add(%c, %bias)
```

---

## 11.10 ReplicaId

`ReplicaId` returns the unique replica ID of the current computation as a scalar `u32`. Replicas are identical computations running on different devices with different data (data parallelism).

### Signature

```
%id = u32[] replica-id()
```

### Semantics

- No operands.
- Returns a scalar `u32` with the replica ID, in the range `[0, num_replicas)`.
- Each replica sees a different value.
- Commonly used to index into sharded data or to control replica-specific behavior.

### Example

```
%id = u32[] replica-id()
%shard = f32[128] dynamic-slice(%full_data, %id), ...
// Each replica gets its own shard of the data
```

### Python (JAX)

```python
import jax
import jax.numpy as jnp

# In JAX, replica ID is typically accessed via jax.lax.axis_index
# in the context of pmap or shard_map
@jax.pmap
def f(x):
    replica_id = jax.lax.axis_index('i')
    return x + jnp.float32(replica_id)
```

---

## 11.11 PartitionId

`PartitionId` returns the partition ID of the current computation in an SPMD-partitioned program. Partitions are created when XLA partitions a single program across multiple devices using SPMD (Single Program, Multiple Data).

### Signature

```
%id = u32[] partition-id()
```

### Semantics

- No operands.
- Returns a scalar `u32` with the partition ID, in the range `[0, num_partitions)`.
- Used in SPMD-partitioned programs to identify which partition is executing.
- Distinct from `ReplicaId`: replicas run independent computations, partitions run the same computation on different data shards.

### Example

```
%pid = u32[] partition-id()
%local_slice = f32[64] dynamic-slice(%sharded_input, %pid), ...
// Each partition reads its own shard
```

---

## 11.12 RngBitGenerator

`RngBitGenerator` generates random bits using a specified pseudo-random number generation algorithm. It returns both the updated state and the generated random bits.

### Signature

```
%result = (T_state, T_output) rng-bit-generator(%state),
    algorithm=A
```

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | `RandomAlgorithm` | The PRNG algorithm to use |
| `state` | `T_state` | The PRNG state (algorithm-dependent shape) |
| Result (index 0) | `T_state` | Updated PRNG state |
| Result (index 1) | `T_output` | Generated random bits with the specified output shape |

### Random Algorithms

XLA supports the following PRNG algorithms:

| Algorithm | State Shape | Description |
|-----------|-------------|-------------|
| `RNG_PHILOX_4x32_10` | `u64[2]` or `u32[4]` | Philox4x32 with 10 rounds. Counter-based PRNG from the Random123 library. The state consists of a 64-bit counter and a 64-bit key. |
| `RNG_THREE_FRY_2x32` | `u32[2]` | ThreeFry2x32. Counter-based PRNG. The state is two 32-bit words. |
| `RNG_XLA_DEFAULT` | backend-dependent | Default algorithm chosen by XLA. |

#### Philox (RNG_PHILOX_4x32_10)

Philox is a counter-based random number generator from the Random123 suite. It produces 128 bits of output from a 64-bit counter and a 64-bit key:

- **State**: `u64[2]` -- `[counter, key]` or equivalently `u32[4]`.
- **Algorithm**: Applies 10 rounds of the Philox4x32 permutation.
- **Output**: For each call, the counter is incremented. The output shape determines how many 128-bit blocks are generated (the counter is advanced by `ceil(output_elements / 4)` steps).
- **Properties**: High-quality statistical properties, fast, GPU-friendly (no carry chain).

```cpp
// Philox state: [counter_lo, counter_hi, key_lo, key_hi]
// Each round: apply S-boxes and LFSR to mix counter bits with key
// After 10 rounds: output = final state
```

#### ThreeFry (RNG_THREE_FRY_2x32)

ThreeFry is another counter-based PRNG from Random123, based on the Threefish block cipher:

- **State**: `u32[2]` -- `[key0, key1]`.
- **Output**: 64 bits per evaluation.
- **Properties**: Simpler than Philox, suitable for lower-quality but faster random generation.

### Output Shape

The output shape is specified as part of the instruction. The generated bits fill the output shape element by element:

```
%state = u64[2] parameter(0)
%result = (u64[2], u32[1024]) rng-bit-generator(%state),
    algorithm=RNG_PHILOX_4x32_10
%new_state = u64[2] get-tuple-element(%result), index=0
%bits = u32[1024] get-tuple-element(%result), index=1
```

The output `u32[1024]` contains 1024 uniformly distributed 32-bit values. These can be transformed to other distributions (uniform floats, normal, etc.) by subsequent operations.

### Usage in JAX

```python
import jax
import jax.numpy as jnp

# JAX uses Philox as its default PRNG
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

# Under the hood, JAX uses RngBitGenerator with Philox
bits = jax.random.bits(subkey, shape=(1024,), dtype=jnp.uint32)
floats = jax.random.uniform(subkey, shape=(1024,))

# The PRNG state is split and advanced explicitly
```

### State Management

The PRNG state is purely functional in XLA: the state is an input, and the updated state is an output. The caller is responsible for threading the state through the computation:

```
ENTRY %main {
  %state0 = u64[2] parameter(0)

  // Generate first batch of random bits
  %r1 = (u64[2], u32[256]) rng-bit-generator(%state0),
      algorithm=RNG_PHILOX_4x32_10
  %state1 = u64[2] get-tuple-element(%r1), index=0
  %bits1 = u32[256] get-tuple-element(%r1), index=1

  // Generate second batch using updated state
  %r2 = (u64[2], u32[256]) rng-bit-generator(%state1),
      algorithm=RNG_PHILOX_4x32_10
  %state2 = u64[2] get-tuple-element(%r2), index=0
  %bits2 = u32[256] get-tuple-element(%r2), index=1

  ROOT %result = (u32[256], u32[256]) tuple(%bits1, %bits2)
}
```

---

## 11.13 RngGetAndUpdateState

`RngGetAndUpdateState` is a side-effecting operation that retrieves the global PRNG state and advances it. Unlike `RngBitGenerator`, which is purely functional, this operation interacts with a per-device global RNG state.

### Signature

```
%state = u64[2] rng-get-and-update-state(), delta=d
```

| Field | Type | Description |
|-------|------|-------------|
| `delta` | `int64` | How much to advance the global PRNG state |
| Result | `u64[2]` | The global PRNG state before the update |

### Semantics

- Reads the current global PRNG state and returns it.
- Advances the global state by `delta`.
- This is a side-effecting operation (each call mutates the global state).
- The order of calls matters: if two `RngGetAndUpdateState` operations appear in the same computation, they are executed in data-dependence order.
- On multiple replicas, each replica has its own global PRNG state.

### Example

```
%state0 = u64[2] rng-get-and-update-state(), delta=10
// %state0 is the state before the update
// The global state has been advanced by 10
%bits = u32[40] rng-bit-generator(%state0), algorithm=RNG_PHILOX_4x32_10
```

---

## 11.14 Bitcast

`Bitcast` reinterprets the bit pattern of its operand as a different element type. The underlying data is not modified; only the type interpretation changes.

### Signature

```
%result = U[D0, D1, ...] bitcast(%operand)
```

Where the operand has type `T[D0, D1, ...]` and `sizeof(T) == sizeof(U)`.

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T[D0, ..., Dn]` | Input array |
| Result | `U[D0, ..., Dn]` | Same shape, different element type |

### Semantics

- The element types `T` and `U` must have the same bit width.
- The shape (dimensions) is unchanged.
- This is a zero-copy operation on most backends.
- Commonly used to convert between `f32` and `u32`, or `f16` and `u16`, for bit-level manipulation.

### Supported Bitcast Pairs

| From | To | Bit Width |
|------|----|-----------|
| `f32` | `u32` | 32 |
| `u32` | `f32` | 32 |
| `f16` | `u16` | 16 |
| `u16` | `f16` | 16 |
| `bf16` | `u16` | 16 |
| `f64` | `u64` | 64 |
| `u64` | `f64` | 64 |
| `s8` | `u8` | 8 |
| `s32` | `u32` | 32 |

### Example

```
%f = f32[128] parameter(0)
%u = u32[128] bitcast(%f)      // reinterpret f32 bits as u32
%mask = u32[128] constant({0x7FFFFFFF, ...})
%masked = u32[128] and(%u, %mask)  // clear sign bit
%result = f32[128] bitcast(%masked) // back to f32 (absolute value)
```

### Python (JAX)

```python
import jax.numpy as jnp

x = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
bits = jnp.asarray(x).view(jnp.uint32)  # Equivalent to bitcast
# bits: array([1065353216, 3221225472, 1077936128], dtype=uint32)
```

---

## 11.15 AddDependency

`AddDependency` creates an explicit data dependency from one value to another, without modifying the value. It is used to enforce execution ordering when no natural data dependency exists.

### Signature

```
%result = T add-dependency(%operand, %dependency)
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | The value to pass through |
| `dependency` | `U` | The value that must be computed first |
| Result | `T` | Identical to `operand` |

### Semantics

- The result is identical to `operand`.
- Execution of any consumer of `%result` will not begin until `%dependency` has been computed.
- Useful for enforcing ordering between side-effecting operations and subsequent computations.
- The `dependency` operand's value is discarded; only its existence as a dependency matters.

### Example

```
// Ensure infeed completes before using the data
%tok0 = token after-all()
%infeed_result = (f32[128], token) infeed(), token=%tok0
%data = f32[128] get-tuple-element(%infeed_result), index=0
%tok1 = token get-tuple-element(%infeed_result), index=1

// Some computation that might be reordered before infeed
%other = f32[128] parameter(1)

// Force %other computation to wait for infeed
%other_dep = f32[128] add-dependency(%other, %tok1)

// Now safe to use %other_dep
ROOT %result = f32[128] add(%data, %other_dep)
```

---

## 11.16 Domain

`Domain` is a no-op operation that serves as a boundary marker between different scheduling domains. It allows the XLA scheduler to partition the graph into regions with different scheduling constraints.

### Signature

```
%result = T domain(%operand), domain={kind, ...}
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Input value |
| `domain` | `DomainMetadata` | Domain metadata (kind-specific) |
| Result | `T` | Identical to `operand` |

### Semantics

- The result is identical to the operand.
- `Domain` operations serve as scheduling barriers. Operations in different domains may be scheduled differently.
- The `DomainMetadata` attached to the instruction identifies the domain kind.
- Useful for separating regions of the computation that have different execution requirements (e.g., regions that should run on different hardware units).
- The XLA scheduler will not move operations across domain boundaries.

### Example

```
%x = f32[128] parameter(0)

// Operations in domain A
%y = f32[128] add(%x, %bias)

// Domain boundary
%z = f32[128] domain(%y), domain={kind="sharding_boundary"}

// Operations in domain B
%w = f32[128] multiply(%z, %scale)
```

### Domain Metadata

Domain metadata is represented as a `DomainMetadata` object, which contains:

| Field | Description |
|-------|-------------|
| `kind` | String identifying the domain type |
| `properties` | Key-value pairs of domain-specific properties |

Common domain kinds used in practice:

| Kind | Description |
|------|-------------|
| `sharding` | Separates differently-sharded regions |
| `scheduling` | Separates scheduling domains for performance |
| `debug` | Marks debug boundaries |

---

## 11.17 Trace

`Trace` is a debugging operation that logs the value of its operand during execution. It has no effect on the computation result.

### Signature

```
%result = T trace(%operand), tag="message"
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Value to trace |
| `tag` | `string` | Message prefix for the log output |
| Result | `T` | Identical to `operand` |

### Semantics

- The result is identical to the operand (passthrough).
- During execution, the runtime prints the operand's value to the debug log, prefixed with `tag`.
- Only active when debugging/tracing is enabled; otherwise it is a no-op.
- Should not be used in production code due to performance impact.
- May be stripped by XLA optimization passes in non-debug builds.

### Example

```
%x = f32[4] parameter(0)
%y = f32[4] add(%x, %ones)
%z = f32[4] trace(%y), tag="after_add"
ROOT %w = f32[4] multiply(%z, %two)
```

When executed with tracing enabled, this would print:
```
[trace] after_add: [1.0, 2.0, 3.0, 4.0]
```

---

## 11.18 BatchGroupGrad

`BatchGroupGrad` is used internally during backpropagation to handle batch dimensions in grouped gradient computations. It is part of XLA's automatic differentiation support and is typically not constructed directly by users.

### Signature

```
%result = T batch-group-grad(%operand), batch_group_count=N
```

| Field | Type | Description |
|-------|------|-------------|
| `operand` | `T` | Gradient input |
| `batch_group_count` | `int64` | Number of batch groups |
| Result | `T` | Transformed gradient |

### Semantics

- Used in the gradient computation of operations that have batch-group semantics (e.g., convolution with batch-group count > 1).
- Rearranges elements to correctly aggregate gradients across batch groups.
- The first dimension of the operand is interpreted as `batch_group_count * batch_size`.
- The result has the same shape but with elements rearranged to account for the grouping.

### Example

During backpropagation of a grouped convolution with `batch_group_count=2`:

```
// Forward: input shape [2*B, H, W, C_in], kernel shape [H, W, C_in, C_out/2]
// Gradient of input has shape [2*B, H, W, C_in]
// BatchGroupGrad rearranges to correctly sum gradients across groups

%grad = f32[16, 28, 28, 64] parameter(0)
%corrected = f32[16, 28, 28, 64] batch-group-grad(%grad), batch_group_count=2
```

---

## 11.19 Comprehensive Summary Table

| Operation | Side Effect | Operands | Results | Key Use |
|-----------|-------------|----------|---------|---------|
| `CustomCall` | Optional | Typed buffers, attributes | Typed buffers | Foreign function interface |
| `Infeed` | Yes | Token | (T, token) | Host-to-device transfer |
| `Outfeed` | Yes | (T, token) | Token | Device-to-host transfer |
| `Send` | Yes | (T, token) | Token | Point-to-point send |
| `Recv` | Yes | Token | (T, token) | Point-to-point receive |
| `AfterAll` | No | Tokens | Token | Token merging |
| `GetDimensionSize` | No | Array | u32 scalar | Dynamic shape query |
| `SetDimensionSize` | No | Array, size | Array | Dynamic shape setting |
| `CreateToken` | No | None | Token | Token creation |
| `OptimizerBarrier` | No | T | T | Scheduling barrier |
| `ReplicaId` | No | None | u32 scalar | Replica identification |
| `PartitionId` | No | None | u32 scalar | SPMD partition ID |
| `RngBitGenerator` | No | PRNG state | (state, bits) | Purely functional RNG |
| `RngGetAndUpdateState` | Yes | None | PRNG state | Global RNG state access |
| `Bitcast` | No | T array | U array | Type reinterpretation |
| `AddDependency` | No | T, U | T | Execution ordering |
| `Domain` | No | T | T | Scheduling domain boundary |
| `Trace` | No* | T | T | Debug logging |
| `BatchGroupGrad` | No | T | T | Gradient grouping |

*Trace has a side effect (logging) but is treated as pure for optimization purposes.

---

## 11.20 Interactions Between Operations

Understanding how these operations interact is essential for correct program construction:

### Token Chain Pattern

All I/O and communication operations are connected via tokens to form a total order within each computation:

```
AfterAll ──► Infeed ──► AfterAll ──► Outfeed ──► AfterAll
                │                        ▲
                └──► Send ──► AfterAll ──┘
```

### Dynamic Shape Pattern

`SetDimensionSize` and `GetDimensionSize` work together to implement dynamic shapes:

```
SetDimensionSize(data, n) ──► [operations respect dynamic bound]
                                         │
                                         ▼
                              GetDimensionSize(result) ──► u32 size
```

### PRNG State Threading Pattern

`RngBitGenerator` is purely functional, so the state must be explicitly threaded:

```
state0 ──► RngBitGenerator ──► state1 ──► RngBitGenerator ──► state2
                                   │                              │
                                   ▼                              ▼
                              random_bits_1                  random_bits_2
```

### CustomCall Composition Pattern

Custom calls can be chained with standard XLA operations and can appear anywhere in the computation graph where standard operations are allowed:

```
%a = f32[128] parameter(0)
%b = f32[128] custom-call(%a), call_target_name="my_op"
%c = f32[128] relu(%b)
%d = f32[128] custom-call(%c), call_target_name="another_op"
```
