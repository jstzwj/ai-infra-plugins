# TensorFlow Lite for Microcontrollers Reference

## Table of Contents

1. [TFLite Micro Architecture Overview](#tflite-micro-architecture-overview)
2. [Memory Model](#memory-model)
3. [MicroInterpreter](#microinterpreter)
4. [OpResolver System](#opresolver-system)
5. [Kernel Implementation](#kernel-implementation)
6. [Platform Support](#platform-support)
7. [Build Systems](#build-systems)
8. [Custom Op Registration](#custom-op-registration)
9. [Memory Planning](#memory-planning)
10. [Profiling and Debugging](#profiling-and-debugging)
11. [Common Applications](#common-applications)
12. [Framework Integrations](#framework-integrations)
13. [Optimization Techniques](#optimization-techniques)

---

## TFLite Micro Architecture Overview

### Design Principles

TensorFlow Lite for Microcontrollers (TFLM) is designed to run machine learning
models on resource-constrained devices with the following constraints:

- **No operating system dependency**: Runs bare-metal without POSIX APIs
- **No dynamic memory allocation**: All memory is pre-allocated at compile time
  using arena-based allocation
- **No standard library dependency**: Avoids most C++ standard library features
  including RTTI, exceptions, and STL containers
- **Minimal binary size**: Designed for devices with as little as 16KB of RAM
- **Deterministic execution**: No heap fragmentation or allocation failures

### Architecture Layers

```
+------------------------------------------+
|           Application Code                |
+------------------------------------------+
|           MicroInterpreter                |
+------------------------------------------+
|           OpResolver                      |
+------------------------------------------+
|     Kernel Implementations                |
|  (reference + platform-optimized)         |
+------------------------------------------+
|         Tensor Arena                      |
|    (pre-allocated memory buffer)          |
+------------------------------------------+
|         FlatBuffer Model                  |
|    (read-only, stored in flash/ROM)       |
+------------------------------------------+
```

### Key Components

1. **MicroInterpreter**: The main inference engine that manages model loading,
   tensor allocation, and op execution.
2. **MicroAllocator**: Handles all memory allocation from a fixed arena,
   including tensors, op data, and scratch buffers.
3. **MicroOpResolver**: Registers only the operations needed by the model.
4. **FlatBuffer Model**: The compiled model stored in read-only memory.
5. **Kernel implementations**: Optimized and reference implementations for
   each supported operation.

### Key Differences from TFLite

| Feature | TFLite | TFLM |
|---|---|---|
| Dynamic memory | Yes (malloc/free) | No (arena only) |
| OS dependency | Requires POSIX/OS | None (bare metal) |
| Standard library | Full C++ stdlib | Minimal subset |
| Exception handling | Yes | No |
| RTTI | Yes | No |
| Tensor arena | Dynamic | Pre-allocated buffer |
| Op resolution | Full built-in set | Selective registration |
| Delegate support | Full | Limited |
| Multi-threading | Yes | No |
| Binary size | ~1-5 MB | ~20-200 KB |

### Reduced Data Structures

TFLM uses simplified versions of TFLite structures when compiled with
`TF_LITE_STATIC_MEMORY`:

```c
// TFLM TfLiteTensor (reduced)
typedef struct TfLiteTensor {
  TfLiteQuantization quantization;
  TfLiteQuantizationParams params;
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  size_t bytes;
  TfLiteType type;
  TfLiteAllocationType allocation_type;
  bool is_variable;
} TfLiteTensor;

// TFLM TfLiteNode (reduced)
typedef struct TfLiteNode {
  TfLiteIntArray* inputs;
  TfLiteIntArray* outputs;
  TfLiteIntArray* intermediates;
  void* user_data;
  void* builtin_data;
  const void* custom_initial_data;
  int custom_initial_data_size;
} TfLiteNode;

// TFLM TfLiteEvalTensor
typedef struct TfLiteEvalTensor {
  TfLitePtrUnion data;
  TfLiteIntArray* dims;
  TfLiteType type;
} TfLiteEvalTensor;
```

The reduced structures omit fields not needed for embedded inference:
- `allocation` pointer
- `buffer_handle` and `delegate` (no delegate support)
- `data_is_stale` (no delegate caching)
- `name` (string names removed for size)
- `sparsity` (no sparse tensor support)
- `dims_signature` (no dynamic shapes)
- `temporaries` and `might_have_side_effect` on TfLiteNode

---

## Memory Model

### Arena-Based Allocation

TFLM uses a single contiguous memory region called the **tensor arena** for
all runtime allocations. The arena is provided by the application as a
byte array:

```c++
// Define the tensor arena
constexpr int kTensorArenaSize = 60 * 1024;  // 60 KB
uint8_t tensor_arena[kTensorArenaSize];
```

### Arena Layout

The arena is divided into multiple sections:

```
High Address
+---------------------------+
|  Head section             |
|  - persistent buffers     |
|  - op data                |
|  - tail section size info |
+---------------------------+
|                           |
|  Free space               |
|  (between head and tail)  |
|                           |
+---------------------------+
|  Tail section             |
|  - tensor buffers         |
|  - scratch buffers        |
+---------------------------+
Low Address
```

### Allocation Types in TFLM

```c
typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,           // No allocation
  kTfLiteMmapRo,                // Read-only (model weights in flash)
  kTfLiteArenaRw,               // Read-write (arena allocated, per-invocation)
  kTfLiteArenaRwPersistent,     // Persistent across invocations
  kTfLiteDynamic,               // Dynamic (NOT used in TFLM)
  kTfLitePersistentRo,          // Persistent read-only (prepare phase)
  kTfLiteCustom,                // Custom allocation
} TfLiteAllocationType;
```

### Memory Allocation Strategy

1. **Model weights**: Stored in read-only memory (flash/ROM). Mapped as
   `kTfLiteMmapRo` tensors. No arena space consumed.
2. **Input/output tensors**: Allocated in the arena tail section.
3. **Intermediate tensors**: Allocated in the arena tail section with
   lifetime-aware placement to maximize memory reuse.
4. **Scratch buffers**: Temporary buffers allocated per-operation. Freed
   after the operation completes, allowing reuse.
5. **Persistent buffers**: Op data and buffers that persist across
   invocations, allocated in the arena head section.
6. **Op data**: Per-operation metadata (kernel parameters, etc.) allocated
   from the head section.

### Choosing Arena Size

The required arena size depends on:
- Model size (number of tensors, tensor sizes)
- Input/output tensor sizes
- Number and type of operations
- Whether scratch buffers are needed

Guidelines for sizing:
```c++
// Start with an estimate and increase until allocation succeeds
constexpr int kTensorArenaSize = 60 * 1024;  // Start with 60 KB
uint8_t tensor_arena[kTensorArenaSize];

// Check if allocation succeeded
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     kTensorArenaSize);
if (interpreter.AllocateTensors() != kTfLiteOk) {
  // Arena too small - increase kTensorArenaSize
}
```

### Memory Debugging

```c++
// Get memory usage statistics
size_t used_bytes = interpreter.arena_used_bytes();
size_t total_bytes = kTensorArenaSize;

// Log memory usage
printf("Arena: %zu / %zu bytes used (%.1f%%)\n",
       used_bytes, total_bytes,
       100.0 * used_bytes / total_bytes);
```

---

## MicroInterpreter

### Creation and Setup

```c++
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Model data (generated by flatc or included as array)
extern const unsigned char g_model_data[];
extern const int g_model_data_len;

// Create the model
const tflite::Model* model = tflite::GetModel(g_model_data);

// Create op resolver
tflite::AllOpsResolver resolver;

// Create interpreter
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     kTensorArenaSize);

// Allocate tensors
interpreter.AllocateTensors();
```

### Constructor Options

```c++
// Full constructor with all options
tflite::MicroInterpreter interpreter(
    model,                          // FlatBuffer model
    resolver,                       // OpResolver
    tensor_arena,                   // Arena buffer
    kTensorArenaSize,               // Arena size in bytes
    nullptr,                        // Optional MicroProfiler
    tflite::MicroResourceVariables::Create()  // Optional resource variables
);
```

### Input and Output Access

```c++
// Get input tensor
TfLiteTensor* input = interpreter.input(0);
// Set input data
input->data.f[0] = 1.0f;
input->data.f[1] = 2.0f;

// Run inference
TfLiteStatus status = interpreter.Invoke();
if (status != kTfLiteOk) {
  // Handle error
}

// Get output tensor
TfLiteTensor* output = interpreter.output(0);
float result = output->data.f[0];

// For multi-input / multi-output models
TfLiteTensor* input1 = interpreter.input(0);
TfLiteTensor* input2 = interpreter.input(1);
TfLiteTensor* output1 = interpreter.output(0);
TfLiteTensor* output2 = interpreter.output(1);
```

### Error Handling

```c++
// AllocateTensors returns status
TfLiteStatus allocate_status = interpreter.AllocateTensors();
if (allocate_status != kTfLiteOk) {
  printf("AllocateTensors() failed\n");
  return;
}

// Invoke returns status
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  printf("Invoke() failed\n");
  return;
}

// Check interpreter state
int input_count = interpreter.inputs_size();
int output_count = interpreter.outputs_size();
```

### Tensor Access Patterns

```c++
// Typed access helpers
float* input_data = tflite::GetTensorData<float>(interpreter.input(0));
int8_t* quantized_input = tflite::GetTensorData<int8_t>(interpreter.input(0));

// Get tensor shape
TfLiteIntArray* input_dims = interpreter.input(0)->dims;
int batch = input_dims->data[0];
int height = input_dims->data[1];
int width = input_dims->data[2];
int channels = input_dims->data[3];

// Access eval tensor (lighter weight, no allocation tracking)
TfLiteEvalTensor* eval_input = interpreter.input_tensor(0);
```

### Interpreter Lifecycle

1. **Construction**: Model, resolver, and arena are provided.
2. **AllocateTensors()**: Allocates all tensors in the arena, prepares the
   execution plan.
3. **Set inputs**: Write input data to input tensors.
4. **Invoke()**: Execute the model.
5. **Read outputs**: Read results from output tensors.
6. **Repeat**: Steps 3-5 can be repeated for multiple inferences.

The interpreter is NOT thread-safe. For multi-instance scenarios, create
separate interpreters with separate arenas.

---

## OpResolver System

### AllOpsResolver

For development and testing, `AllOpsResolver` registers all built-in ops:

```c++
#include "tensorflow/lite/micro/all_ops_resolver.h"

tflite::AllOpsResolver resolver;
// All built-in ops are now available
```

**Warning**: Using AllOpsResolver increases binary size significantly since
all op implementations are linked. For production, use MicroMutableOpResolver.

### MicroMutableOpResolver

For production deployments, `MicroMutableOpResolver` registers only the ops
needed by the model:

```c++
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Create resolver with max 10 ops
tflite::MicroMutableOpResolver<10> resolver;

// Add only the ops your model needs
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddFullyConnected();
resolver.AddRelu();
resolver.AddSoftmax();
resolver.AddReshape();
resolver.AddAveragePool2D();
resolver.AddMaxPool2D();
resolver.AddAdd();
resolver.AddMul();
```

### Template Parameter

The template parameter specifies the maximum number of ops that can be
registered:

```c++
// Register up to 5 ops
tflite::MicroMutableOpResolver<5> resolver;

// Register up to 20 ops
tflite::MicroMutableOpResolver<20> resolver;
```

Choose the smallest value that accommodates your model's ops to minimize
memory usage.

### Available Registration Methods

```c++
// Convolution ops
resolver.AddConv2D(tflite::Register_CONV_2D());
resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D());
resolver.AddTransposeConv(tflite::Register_TRANSPOSE_CONV());

// Pooling ops
resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D());
resolver.AddMaxPool2D(tflite::Register_MAX_POOL_2D());

// Activation ops
resolver.AddRelu();
resolver.AddRelu6();
resolver.AddTanh();
resolver.AddLogistic();
resolver.AddSoftmax();

// FC layer
resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED());

// Element-wise
resolver.AddAdd(tflite::Register_ADD());
resolver.AddSub();
resolver.AddMul(tflite::Register_MUL());
resolver.AddMaximum();
resolver.AddMinimum();

// Array ops
resolver.AddReshape();
resolver.AddConcatenation();
resolver.AddSplit();
resolver.AddPad();
resolver.AddTranspose();
resolver.AddGather();
resolver.AddExpandDims();
resolver.AddSqueeze();
resolver.AddStridedSlice();

// Quantization ops
resolver.AddDequantize();
resolver.AddQuantize();

// Custom op
resolver.AddCustom("MyCustomOp", tflite::Register_MyCustomOp());
```

### Determining Required Ops

To find which ops your model uses:

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

ops = set()
for i, detail in enumerate(interpreter._get_tensor_details()):
    pass

# Alternative: use flatc to inspect
# flatc --json --strict-json model.tflite
# Then look at operator_codes in the JSON output
```

---

## Kernel Implementation

### Kernel Style Guide

TFLM kernel implementations follow a specific style to ensure:
- No dynamic memory allocation
- Minimal stack usage
- Clear separation of reference and optimized code
- Deterministic behavior

### Reference Kernel Structure

```c++
// File: tensorflow/lite/micro/kernels/my_op.cc

#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

namespace {
// Kernel parameters structure
struct MyOpData {
  float alpha;
  int num_iterations;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // Allocate persistent data from the arena
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(buffer != nullptr);

  // Use context->AllocatePersistentBuffer for one-time allocation
  MyOpData* data = static_cast<MyOpData*>(
      context->AllocatePersistentBuffer(context, sizeof(MyOpData)));

  // Parse parameters from flatbuffer
  tflite::MyOpParams params;
  if (tflite::ParseMyOpParams(buffer, length, &params)) {
    data->alpha = params.alpha;
    data->num_iterations = params.num_iterations;
  }

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  // No-op in TFLM (arena-managed memory)
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  MyOpData* data = static_cast<MyOpData*>(node->user_data);

  // Validate input types
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);

  // Set output shape
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  SetTensorShape(output, GetTensorShape(input));

  // Request scratch buffer if needed
  int scratch_buffer_index;
  TF_LITE_ENSURE_OK(context,
      context->RequestScratchBufferInArena(
          context, input->bytes, &scratch_buffer_index));

  data->scratch_buffer_index = scratch_buffer_index;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const MyOpData* data = static_cast<MyOpData*>(node->user_data);

  // Get input and output tensors
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  // Get scratch buffer
  void* scratch = context->GetScratchBuffer(context,
                                            data->scratch_buffer_index);

  // Get tensor data
  const float* input_data = tflite::micro::GetTensorData<float>(input);
  float* output_data = tflite::micro::GetTensorData<float>(output);

  // Compute
  const int num_elements = ElementCount(*input->dims);
  for (int i = 0; i < num_elements; i++) {
    output_data[i] = input_data[i] * data->alpha;
  }

  return kTfLiteOk;
}

}  // namespace

// Registration function
TfLiteRegistration Register_MY_OP() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
```

### Scratch Buffer Management

Scratch buffers are temporary buffers allocated during `Prepare` and available
during `Eval`:

```c++
// In Prepare:
int scratch_index;
context->RequestScratchBufferInArena(context,
    buffer_size, &scratch_index);
// Store index in op data
op_data->scratch_buffer_index = scratch_index;

// In Eval:
void* scratch_buffer = context->GetScratchBuffer(context,
    op_data->scratch_buffer_index);
```

Scratch buffers are reused across operations when their lifetimes do not
overlap, minimizing total arena size.

### Persistent Buffer Allocation

For data that must persist across invocations:

```c++
// In Init:
void* persistent_data = context->AllocatePersistentBuffer(
    context, sizeof(MyPersistentData));
```

Persistent buffers are allocated from the head section and never freed.

### Context Functions in TFLM

```c++
// Memory allocation
void* (*AllocatePersistentBuffer)(TfLiteContext* ctx, size_t bytes);
TfLiteStatus (*RequestScratchBufferInArena)(TfLiteContext* ctx,
                                            size_t bytes, int* buffer_idx);
void* (*GetScratchBuffer)(TfLiteContext* ctx, int buffer_idx);

// Tensor access
TfLiteTensor* (*GetTensor)(const TfLiteContext* context, int tensor_idx);
TfLiteEvalTensor* (*GetEvalTensor)(const TfLiteContext* context,
                                   int tensor_idx);

// Error reporting
void (*ReportError)(TfLiteContext*, const char* msg, ...);

// Tensor resizing
TfLiteStatus (*ResizeTensor)(TfLiteContext*, TfLiteTensor* tensor,
                             TfLiteIntArray* new_size);
```

### Helper Functions

```c++
namespace tflite {
namespace micro {

// Get typed tensor data (returns nullptr if type doesn't match)
template <typename T>
T* GetTensorData(TfLiteEvalTensor* tensor);

template <typename T>
const T* GetTensorData(const TfLiteEvalTensor* tensor);

// Get input/output tensors
TfLiteEvalTensor* GetEvalInput(const TfLiteContext* context,
                               const TfLiteNode* node, int index);
TfLiteEvalTensor* GetEvalOutput(const TfLiteContext* context,
                                const TfLiteNode* node, int index);

// Tensor shape helpers
int ElementCount(const TfLiteIntArray& dims);
size_t EvalTensorBytes(const TfLiteEvalTensor* tensor);

// Gettensor from the full context (heavier weight)
TfLiteTensor* GetMutableInput(TfLiteContext* context,
                              const TfLiteNode* node, int index);
const TfLiteTensor* GetInput(TfLiteContext* context,
                             const TfLiteNode* node, int index);
TfLiteTensor* GetOutput(TfLiteContext* context,
                        const TfLiteNode* node, int index);
TfLiteTensor* GetTemporary(TfLiteContext* context,
                           const TfLiteNode* node, int index);

}  // namespace micro
}  // namespace tflite
```

---

## Platform Support

### ARM Cortex-M Series

| Core | Architecture | FPU | DSP | Typical Devices |
|---|---|---|---|---|
| Cortex-M0 | ARMv6-M | No | No | STM32F0, NRF51 |
| Cortex-M0+ | ARMv6-M | No | No | RP2040, NRF52 |
| Cortex-M3 | ARMv7-M | No | No | STM32F1, STM32F2 |
| Cortex-M4 | ARMv7E-M | Optional | Yes | STM32F3, STM32F4, Kinetis K |
| Cortex-M4F | ARMv7E-M | Yes (FPv4-SP) | Yes | STM32F4, STM32F7, SAMD51 |
| Cortex-M7 | ARMv7E-M | Yes (FPv5) | Yes | STM32F7, STM32H7 |
| Cortex-M23 | ARMv8-M Baseline | No | No | Cortex-M23 devices |
| Cortex-M33 | ARMv8-M Mainline | Optional | Optional | STM32L5, nRF5340 |
| Cortex-M55 | ARMv8.1-M | Yes | Yes (Helium) | Corstone-300, Alif |
| Cortex-M85 | ARMv8.1-M | Yes | Yes (Helium) | Next-gen devices |

### RISC-V Support

TFLM supports RISC-V processors with:
- **RV32I base**: Basic integer instruction set
- **RV32M extension**: Multiplication and division
- **RV32F extension**: Single-precision floating point
- **RV32V extension**: Vector operations (P extension for DSP)

Platforms: SiFive Freedom, ESP32-C3, Kendryte K210.

### Xtensa / Cadence Tensilica

TFLM supports Cadence Tensilica DSP cores:
- **HiFi 3**: Audio/speech processing
- **HiFi 4**: Advanced audio and neural network inference
- **HiFi 5**: Neural network inference optimized
- **Vision P6**: Image and vision processing

### Supported Platforms Summary

| Platform | Architecture | Optimized Kernels |
|---|---|---|
| ARM Cortex-M0/M0+/M3 | ARMv6-M, ARMv7-M | Reference only |
| ARM Cortex-M4/M4F | ARMv7E-M | CMSIS-NN |
| ARM Cortex-M7 | ARMv7E-M | CMSIS-NN |
| ARM Cortex-M33 | ARMv8-M | CMSIS-NN |
| ARM Cortex-M55/M85 | ARMv8.1-M | Helium (MVEI) |
| RISC-V (RV32IMF) | RISC-V | Reference |
| Xtensa HiFi 3/4/5 | Xtensa LX | NNLib |
| Xtensa Vision P6 | Xtensa LX | Vision NNLib |

---

## Build Systems

### CMake Build

```cmake
# CMakeLists.txt for TFLM project
cmake_minimum_required(VERSION 3.16)
project(my_tflm_app C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# Add TFLM as a subdirectory or include via FetchContent
include(FetchContent)
FetchContent_Declare(
  tflm
  GIT_REPOSITORY https://github.com/tensorflow/tflite-micro.git
  GIT_TAG main
)
FetchContent_MakeAvailable(tflm)

# Create executable
add_executable(my_app main.cc)
target_link_libraries(my_app tensorflow-micro)
```

#### Cross-Compilation with CMake

```bash
# ARM Cortex-M4 cross-compilation
cmake -G "Unix Makefiles" \
  -DCMAKE_TOOLCHAIN_FILE=tools/cmake/arm_gcc.cmake \
  -DTARGET=cortex_m4 \
  -DTFLM_BUILD_TYPE=release \
  ..

cmake --build .
```

#### Toolchain File Example

```cmake
# arm_gcc_toolchain.cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)

set(CMAKE_C_FLAGS_INIT "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -fno-exceptions -fno-rtti")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

### Makefile Build

```makefile
# Makefile for TFLM project

# TFLM source directory
TFLM_DIR = $(HOME)/tflite-micro

# Compiler settings
CC = arm-none-eabi-gcc
CXX = arm-none-eabi-g++
CFLAGS = -mcpu=cortex-m4 -mthumb -O2 -Wall
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti -std=c++14

# Include paths
INCLUDES = \
  -I$(TFLM_DIR) \
  -I$(TFLM_DIR)/tensorflow/lite/micro \
  -I$(TFLM_DIR)/tensorflow/lite/micro/kernels \
  -I$(TFLM_DIR)/tensorflow/lite/core/c \
  -I$(TFLM_DIR)/third_party/flatbuffers/include

# Source files
TFLM_SRCS = \
  $(wildcard $(TFLM_DIR)/tensorflow/lite/micro/*.cc) \
  $(wildcard $(TFLM_DIR)/tensorflow/lite/micro/kernels/*.cc) \
  $(wildcard $(TFLM_DIR)/tensorflow/lite/core/api/*.cc) \
  $(wildcard $(TFLM_DIR)/tensorflow/lite/schema/*.cc)

APP_SRCS = main.cc

# Build target
my_app: $(APP_SRCS) $(TFLM_SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@
```

### Bazel Build

```python
# BUILD file for TFLM project

load("@org_tensorflow//tensorflow/lite/micro:build_def.bzl", "tflm_cc_library")

cc_binary(
    name = "my_app",
    srcs = ["main.cc"],
    deps = [
        "@org_tensorflow//tensorflow/lite/micro:micro_interpreter",
        "@org_tensorflow//tensorflow/lite/micro:all_ops_resolver",
        "@org_tensorflow//tensorflow/lite/schema:schema_cc",
        "@org_tensorflow//tensorflow/lite/micro:micro_allocator",
    ],
    copts = [
        "-fno-exceptions",
        "-fno-rtti",
        "-DCMSIS_NN",
    ],
)
```

### Generating Model Arrays

Convert a TFLite model to a C array for embedding:

```bash
xxd -i model.tflite > model_data.cc
```

Or use the TFLM tool:

```bash
python3 tensorflow/lite/micro/tools/generate_cc_arrays.py \
    model.tflite model_data.cc
```

This produces:
```c++
// model_data.cc
const unsigned char g_model_data[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
const unsigned int g_model_data_len = 12345;
```

---

## Custom Op Registration

### MicroMutableOpResolver with Custom Ops

```c++
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Custom op registration
extern "C" {
  TfLiteRegistration* Register_MY_CUSTOM_OP();
}

tflite::MicroMutableOpResolver<6> resolver;
resolver.AddConv2D();
resolver.AddRelu();
resolver.AddReshape();
resolver.AddSoftmax();
resolver.AddFullyConnected();
resolver.AddCustom("MyCustomOp", Register_MY_CUSTOM_OP());
```

### Complete Custom Op Example

```c++
// my_custom_op.h
#ifndef MY_CUSTOM_OP_H_
#define MY_CUSTOM_OP_H_

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
TfLiteRegistration Register_MY_CUSTOM_OP();
}  // namespace tflite

#endif  // MY_CUSTOM_OP_H_

// my_custom_op.cc
#include "my_custom_op.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

struct OpData {
  float threshold;
  int scratch_buffer_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = static_cast<OpData*>(
      context->AllocatePersistentBuffer(context, sizeof(OpData)));
  // Parse threshold from custom_options buffer
  if (buffer != nullptr && length >= sizeof(float)) {
    data->threshold = *reinterpret_cast<const float*>(buffer);
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);

  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  // Output shape same as input
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  context->ResizeTensor(context, output, output_size);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* data = static_cast<OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, 0);

  const float* in = tflite::micro::GetTensorData<float>(input);
  float* out = tflite::micro::GetTensorData<float>(output);

  const int count = ElementCount(*input->dims);
  for (int i = 0; i < count; i++) {
    out[i] = in[i] > data->threshold ? in[i] : 0.0f;
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_MY_CUSTOM_OP() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
```

---

## Memory Planning

### Offline Memory Planning

TFLM uses offline memory planning to determine optimal tensor placement before
execution:

1. **Lifetime analysis**: Each tensor has a defined lifetime (first use to
   last use).
2. **Overlay planning**: Tensors with non-overlapping lifetimes can share
   the same memory.
3. **Greedy allocation**: Tensors are placed in the arena using a greedy
   first-fit algorithm.

### Arena Layout Strategies

**Default (greedy) layout**:
- Most general, works for all models
- May use more memory than optimal

**Single-arena offset planning**:
- All tensors placed in a single arena
- Offsets determined during `AllocateTensors`

**Multi-arena planning** (advanced):
- Separate arenas for different memory types (e.g., internal SRAM vs.
  external SDRAM)
- Platform-specific optimization

### Reducing Memory Usage

1. **Use quantization**: INT8 models use 4x less memory than FP32.
2. **Minimize intermediate tensors**: Some ops can be fused during conversion.
3. **Tune scratch buffer usage**: Request only the minimum needed size.
4. **Use only required ops**: Smaller op resolver reduces persistent data.

### Memory Usage Analysis

```c++
// After AllocateTensors
printf("Arena used: %zu bytes\n", interpreter.arena_used_bytes());

// Per-operation memory usage
for (size_t i = 0; i < interpreter.ops_size(); i++) {
  // Check each operation's memory footprint
}
```

---

## Profiling and Debugging

### Cycle Counting

```c++
#include "tensorflow/lite/micro/micro_profiler.h"

// Create profiler
tflite::MicroProfiler profiler;

// Create interpreter with profiler
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     kTensorArenaSize, &profiler);

// Run inference (profiling is automatic)
interpreter.Invoke();

// Get profiling results
for (int i = 0; i < profiler.GetNumEvents(); i++) {
  printf("Op %d: %lu cycles\n", i, profiler.GetCycleCount(i));
}
```

### Memory Usage Tracking

```c++
// Before allocation
size_t arena_used_before = interpreter.arena_used_bytes();

// After allocation
interpreter.AllocateTensors();
size_t arena_used_after = interpreter.arena_used_bytes();
printf("Allocation used: %zu bytes\n",
       arena_used_after - arena_used_before);
```

### Debug Logging

TFLM uses `TF_LITE_KERNEL_LOG` and `MicroPrintf` for debug output:

```c++
// In kernel code
TF_LITE_KERNEL_LOG(context, "MyOp: processing %d elements", count);

// In application code
MicroPrintf("Inference completed in %d ms", elapsed_ms);
```

To disable logging in production:
```c++
#define TF_LITE_STRIP_ERROR_STRINGS
```

### Common Debugging Issues

1. **Arena too small**: Increase `kTensorArenaSize`.
2. **Op not registered**: Add the missing op to the resolver.
3. **Type mismatch**: Ensure input data types match the model's expectations.
4. **Shape mismatch**: Verify input tensor shapes match the model's signature.
5. **Quantization issues**: Check scale and zero_point values for quantized
   models.

---

## Common Applications

### Keyword Spotting

```c++
// Typical keyword spotting pipeline
// Input: 1-second audio at 16kHz = 16000 samples
// Preprocessing: MFCC features -> 49x10 or 49x40 spectrogram
// Model: DS-CNN, CNN, or GRU with ~50K parameters
// Output: 12 keywords (yes, no, up, down, etc.)

constexpr int kAudioSampleRate = 16000;
constexpr int kFeatureSize = 10;
constexpr int kFeatureCount = 49;
constexpr int kCategoryCount = 12;

// Input tensor: [1, 49, 10, 1] (batch, time, features, channel)
// Output tensor: [1, 12] (probabilities)
```

### Person Detection

```c++
// Visual wake words / person detection
// Input: 96x96 RGB or grayscale image
// Model: MobileNet V1 0.25x or custom CNN (~250K parameters)
// Output: person present / not present

constexpr int kImageWidth = 96;
constexpr int kImageHeight = 96;
constexpr int kImageChannels = 3;  // or 1 for grayscale

// Input tensor: [1, 96, 96, 3]
// Output tensor: [1, 2] (person, not_person)
```

### Gesture Recognition

```c++
// Accelerometer-based gesture recognition
// Input: 128 samples x 3 axes = 384 features
// Model: 2-layer FC network or 1D CNN
// Output: gesture classes

constexpr int kAccelerometerAxes = 3;
constexpr int kSampleCount = 128;
constexpr int kGestureCount = 4;

// Input tensor: [1, 128, 3]
// Output tensor: [1, 4]
```

### Anomaly Detection

```c++
// Time series anomaly detection
// Input: sliding window of sensor data
// Model: autoencoder
// Output: reconstruction error (anomaly score)
```

---

## Framework Integrations

### Arduino Integration

```cpp
// Arduino sketch using TFLM
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h"

constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = tflite::GetModel(g_model_data);
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                     kTensorArenaSize);

void setup() {
  Serial.begin(9600);
  interpreter.AllocateTensors();
}

void loop() {
  // Read sensor data
  float* input = interpreter.input(0)->data.f;
  input[0] = analogRead(A0) / 1024.0f;

  interpreter.Invoke();

  float* output = interpreter.output(0)->data.f;
  Serial.println(output[0]);
  delay(100);
}
```

### ESP-IDF Integration

```cmake
# CMakeLists.txt for ESP-IDF component
idf_component_register(
  SRCS "main.cc"
  INCLUDE_DIRS "."
  REQUIRES tensorflow-lite-micro
)
```

```c++
// main.cc for ESP32
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h"

extern "C" void app_main() {
  // Initialize TFLM
  const tflite::Model* model = tflite::GetModel(g_model_data);

  tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();

  constexpr int kTensorArenaSize = 80 * 1024;
  uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_SPIRAM);

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize);
  interpreter.AllocateTensors();

  while (1) {
    // Run inference
    interpreter.Invoke();
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}
```

### Zephyr RTOS Integration

```cmake
# CMakeLists.txt for Zephyr RTOS
cmake_minimum_required(VERSION 3.13)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

target_sources(app PRIVATE src/main.cc)
target_include_directories(app PRIVATE
  ${TFLM_DIR}/tensorflow/lite/micro
  ${TFLM_DIR}/tensorflow/lite/core/c
)
```

### Mbed OS Integration

```cmake
# CMakeLists.txt for Mbed OS
# Or use mbed-cli / Mbed Studio
mbed_add_library(tflm
  SOURCES ${TFLM_SRCS}
  INCLUDE_DIRS ${TFLM_INCLUDES}
)
```

---

## Optimization Techniques

### CMSIS-NN Optimized Kernels

CMSIS-NN provides optimized kernels for ARM Cortex-M processors:

```c++
// Enable CMSIS-NN during build
// CMake: -DCMSIS_NN=ON
// Makefile: CMSIS_NN=1

// CMSIS-NN provides optimized implementations for:
// - CONV_2D (INT8, INT4)
// - DEPTHWISE_CONV_2D (INT8)
// - FULLY_CONNECTED (INT8)
// - MAX_POOL_2D (INT8)
// - AVERAGE_POOL_2D (INT8)
// - SOFTMAX (INT8, INT16)
// - ADD (INT8)
// - MUL (INT8)
// - SPLIT (INT8)
// - CONCATENATION (INT8)
// - ACTIVATION (INT8)
```

Performance gains with CMSIS-NN vs reference implementations:

| Operation | Speedup |
|---|---|
| INT8 Conv2D | 5-15x |
| INT8 Depthwise Conv2D | 3-8x |
| INT8 Fully Connected | 4-10x |
| INT8 Pool2D | 3-6x |
| INT8 Softmax | 2-4x |

### Xtensa NNLib Optimized Kernels

For Cadence Tensilica processors:

```c++
// Enable Xtensa NNLib during build
// -DXTENSA=1 -DXTENSA_NNLIB=1

// Optimized operations:
// - CONV_2D (INT8)
// - DEPTHWISE_CONV_2D (INT8)
// - FULLY_CONNECTED (INT8)
// - MAX_POOL_2D (INT8)
// - AVERAGE_POOL_2D (INT8)
```

### Quantization for Microcontrollers

INT8 quantization is strongly recommended for TFLM:

```python
# Convert and quantize model for TFLM
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")

# Full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = calibration_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(quantized_model)
```

### Binary Size Optimization

1. **Use MicroMutableOpResolver**: Only link required ops.
2. **Strip error strings**: Define `TF_LITE_STRIP_ERROR_STRINGS`.
3. **Use -Os**: Optimize for size with compiler flags.
4. **Enable LTO**: Link-time optimization (`-flto`).
5. **Use quantized models**: INT8 models have smaller weights.

### Performance Optimization Checklist

1. Use INT8 quantized models
2. Enable platform-optimized kernels (CMSIS-NN, Xtensa NNLib)
3. Use the smallest arena that works
4. Profile per-op performance to identify bottlenecks
5. Consider model architecture changes (fewer parameters)
6. Use fused operations where possible
7. Minimize Python-like overhead in the application

---

## Summary

TensorFlow Lite for Microcontrollers enables machine learning inference on
devices with as little as 16KB of RAM and 256KB of flash:

1. **No OS dependency**: Runs bare-metal on any C/C++ capable MCU.
2. **Arena-based memory**: All allocation from a pre-defined buffer.
3. **Selective op registration**: Only include ops your model needs.
4. **Platform optimizations**: CMSIS-NN for ARM, NNLib for Xtensa.
5. **Broad platform support**: Cortex-M0 to M85, RISC-V, Xtensa.
6. **Multiple build systems**: CMake, Makefile, Bazel.
7. **Framework integrations**: Arduino, ESP-IDF, Zephyr, Mbed.
8. **INT8 quantization**: Critical for efficient MCU inference.
