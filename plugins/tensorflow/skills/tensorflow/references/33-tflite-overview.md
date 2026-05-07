# TensorFlow Lite Overview

This document provides a comprehensive reference for TensorFlow Lite (TFLite),
TensorFlow's lightweight inference framework designed for mobile, embedded, and
edge devices.

## Table of Contents

1. [TFLite Architecture](#tflite-architecture)
2. [FlatBuffer Model Format](#flatbuffer-model-format)
3. [Interpreter](#interpreter)
4. [TFLite Tensors](#tflite-tensors)
5. [InterpreterBuilder](#interpreterbuilder)
6. [Op Registration](#op-registration)
7. [Memory Planning](#memory-planning)
8. [Delegates Overview](#delegates-overview)
9. [TFLite C API](#tflite-c-api)
10. [TFLite Python API](#tflite-python-api)
11. [Model Versioning](#model-versioning)
12. [Flex Delegate](#flex-delegate)
13. [TFLite vs TensorFlow Comparison](#tflite-vs-tensorflow-comparison)
14. [Model Size Optimization](#model-size-optimization)

---

## TFLite Architecture

TFLite is designed for on-device inference with minimal binary size and low latency.

### Architecture Overview

```
TFLite Model (.tflite) - FlatBuffer format
      |
      v
  +---+---+
  | Model  |  FlatBuffer Model parsing
  | Loader |
  +---+---+
      |
      v
  +---+---+
  | Inter- |  Interpreter
  | preter |  - Tensor allocation
  |        |  - Graph execution
  |        |  - Delegate management
  +---+---+
      |
      v
  +---+---+
  | Ops    |  Op Kernel Execution
  | Kernels|  - Builtin ops
  |        |  - Custom ops
  |        |  - Delegate ops
  +---+---+
```

### Key Components

| Component | Description |
|-----------|-------------|
| **FlatBuffer Model** | Serialized model format (`.tflite`) |
| **Interpreter** | Executes the model graph |
| **OpResolver** | Maps op codes to kernel implementations |
| **Delegates** | Hardware acceleration plugins |
| **Tensor** | Data container for inputs, outputs, intermediates |

### Directory Structure

```
tensorflow/lite/
  |-- core/              # Core interpreter and model loading
  |   |-- api/           # Public API (OpResolver, ErrorReporter)
  |   |-- c/             # C API
  |   |-- kernels/       # Builtin kernel implementations
  |-- schema/            # FlatBuffer schema definition
  |-- delegates/         # Hardware delegate implementations
  |-- python/            # Python API and converter
  |-- tools/             # Conversion and optimization tools
  |-- experimental/      # Experimental features
  |-- profiler/          # Profiling infrastructure
  |-- nnapi/             # NNAPI delegate (Android)
  |-- gpu/               # GPU delegate
  |-- swift/             # Swift API (iOS)
  |-- java/              # Java API (Android)
  |-- toco/              # TOCO converter (legacy)
  |-- mlir/              # MLIR-based converter
```

---

## FlatBuffer Model Format

TFLite models are stored in FlatBuffer format, a zero-copy serialization format
optimized for speed and small size.

### Schema Structure

The FlatBuffer schema defines the model structure:

```
Model
  |-- version: int
  |-- operator_codes: [OperatorCode]
  |-- subgraphs: [SubGraph]
  |-- description: string
  |-- buffers: [Buffer]
  |-- metadata: [Metadata]
```

### Key Schema Types

| Type | Description |
|------|-------------|
| `Model` | Top-level model container |
| `SubGraph` | A computation graph |
| `Operator` | A single operation (node) |
| `OperatorCode` | Identifies the operation type |
| `Tensor` | Tensor description (shape, type, buffer) |
| `Buffer` | Raw data (weights, constants) |
| `QuantizationParameters` | Quantization info (scale, zero_point) |

### SubGraph

```flatbuffers
table SubGraph {
  tensors: [Tensor];
  inputs: [int];
  outputs: [int];
  operators: [Operator];
  name: string;
}
```

### Operator

```flatbuffers
table Operator {
  opcode_index: int;
  inputs: [int];
  outputs: [int];
  builtin_options: BuiltinOptions;
  custom_options: [ubyte];
  custom_options_format: CustomOptionsFormat;
  mutating_variable_inputs: [bool];
  intermediates: [int];
}
```

### OperatorCode

```flatbuffers
table OperatorCode {
  builtin_code: BuiltinOperator;
  custom_code: string;
  version: int;
}
```

### Tensor

```flatbuffers
table Tensor {
  shape: [int];
  type: TensorType;
  buffer: int;
  name: string;
  quantization: QuantizationParameters;
  is_variable: bool;
  shape_signature: [int];
}
```

### Buffer

```flatbuffers
table Buffer {
  data: [ubyte];
}
```

### Model Version

The model version indicates the minimum TFLite runtime version required:

```flatbuffers
table Model {
  version: int;  // Minimum runtime version (e.g., 3 for TFLite 2.x)
  // ...
}
```

### Schema Files

The FlatBuffer schema is defined in:

| File | Description |
|------|-------------|
| `tensorflow/lite/schema/schema.fbs` | FlatBuffer schema definition |
| `tensorflow/lite/schema/schema_generated.h` | Generated C++ header |
| `tensorflow/lite/schema/schema_utils.h` | Schema utility functions |
| `tensorflow/lite/schema/upgrade_schema.py` | Schema version upgrade tool |

---

## Interpreter

The `Interpreter` is the core TFLite runtime that loads and executes TFLite models.

### Creation and Usage

```cpp
// C++ API
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

// Load model
auto model = tflite::FlatBufferModel::BuildFromFile("model.tflite");

// Build interpreter
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder builder(*model, resolver);
builder(&interpreter);

// Allocate tensors
interpreter->AllocateTensors();

// Set input
float* input = interpreter->typed_input_tensor<float>(0);
// ... fill input data ...

// Run inference
interpreter->Invoke();

// Get output
float* output = interpreter->typed_output_tensor<float>(0);
```

### Interpreter Class

```cpp
// From: tensorflow/lite/core/interpreter.h
class Interpreter {
 public:
  // Construction
  explicit Interpreter(ErrorReporter* error_reporter = DefaultErrorReporter());

  // Tensor management
  TfLiteStatus AllocateTensors();
  size_t tensors_size() const;

  // Input/Output access
  const std::vector<int>& inputs() const;
  const std::vector<int>& outputs() const;
  TfLiteTensor* input_tensor(size_t index);
  const TfLiteTensor* output_tensor(size_t index);
  TfLiteTensor* tensor(int tensor_index);
  const TfLiteTensor* tensor(int tensor_index) const;

  // Typed tensor access
  template <class T> T* typed_tensor(int tensor_index);
  template <class T> T* typed_input_tensor(int index);
  template <class T> const T* typed_output_tensor(int index) const;

  // Execution
  TfLiteStatus Invoke();

  // Graph modification
  TfLiteStatus AddTensors(int tensors_to_add, int* first_new_tensor_index = nullptr);
  TfLiteStatus AddNodeWithParameters(const std::vector<int>& inputs,
                                      const std::vector<int>& outputs,
                                      const char* init_data,
                                      size_t init_data_size,
                                      void* builtin_data,
                                      const TfLiteRegistration* registration,
                                      int* node_index = nullptr);

  // Input/Output configuration
  TfLiteStatus SetInputs(std::vector<int> inputs);
  TfLiteStatus SetOutputs(std::vector<int> outputs);
  TfLiteStatus SetVariables(std::vector<int> variables);

  // Tensor parameter setting
  TfLiteStatus SetTensorParametersReadOnly(int tensor_index, TfLiteType type,
                                            const char* name,
                                            const std::vector<int>& dims,
                                            TfLiteQuantization quantization,
                                            const char* buffer, size_t bytes);
  TfLiteStatus SetTensorParametersReadWrite(int tensor_index, TfLiteType type,
                                             const char* name,
                                             const std::vector<int>& dims,
                                             TfLiteQuantization quantization,
                                             bool is_variable = false);

  // Delegate support
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  // Execution plan
  const std::vector<int>& execution_plan() const;
  size_t nodes_size() const;

  // Signature support
  std::vector<const std::string*> signature_keys() const;
  SignatureRunner* GetSignatureRunner(const char* signature_key);

  // Cancellation
  TfLiteStatus EnableCancellation();
};
```

### Interpreter Lifecycle

```
1. Create Interpreter
   |
   v
2. Add tensors and nodes (or load from model)
   |
   v
3. AllocateTensors()
   |
   v
4. Set input data
   |
   v
5. Invoke()
   |
   v
6. Read output data
   |
   v
(Repeat 4-6 for each inference)
```

### Thread Safety

The Interpreter is **not** thread-safe. For multi-threaded inference, create
one interpreter per thread.

---

## TFLite Tensors

### TfLiteTensor Structure

```cpp
typedef struct TfLiteTensor {
  TfLiteType type;               // Data type (kTfLiteFloat32, kTfLiteInt8, etc.)
  TfLiteIntArray* dims;          // Shape (dimensions)
  TfLiteQuantizationParams params;  // Legacy quantization params
  char* name;                     // Tensor name
  void* data;                     // Raw data pointer (union of typed pointers)
  TfLiteAllocationType allocation_type;  // Memory allocation type
  const void* allocation;        // Allocation object
  bool is_variable;              // Whether this is a variable tensor
  TfLiteQuantization quantization;  // Quantization info
  const TfLiteAllocation* buffer;  // Buffer allocation
  TfLiteDelegate* delegate;       // Associated delegate
} TfLiteTensor;
```

### Tensor Types

| Type | Enum | Size |
|------|------|------|
| Float32 | `kTfLiteFloat32` | 4 bytes |
| Float16 | `kTfLiteFloat16` | 2 bytes |
| Int32 | `kTfLiteInt32` | 4 bytes |
| Int16 | `kTfLiteInt16` | 2 bytes |
| Int8 | `kTfLiteInt8` | 1 byte |
| UInt8 | `kTfLiteUInt8` | 1 byte |
| Int64 | `kTfLiteInt64` | 8 bytes |
| Bool | `kTfLiteBool` | 1 byte |
| String | `kTfLiteString` | Variable |

### Allocation Types

| Type | Description |
|------|-------------|
| `kTfLiteMemNone` | No allocation |
| `kTfLiteMmapRo` | Read-only memory-mapped (constants) |
| `kTfLiteArenaRw` | Read-write arena allocation (temporary) |
| `kTfLiteArenaRwPersistent` | Persistent arena (variable tensors) |
| `kTfLiteDynamic` | Dynamically allocated |
| `kTfLitePersistentRo` | Persistent read-only |
| `kTfLiteCustom` | Custom allocation |

### Quantization Parameters

```cpp
typedef struct TfLiteQuantizationParams {
  float scale;      // Quantization scale
  int32_t zero_point;  // Zero point
} TfLiteQuantizationParams;

// Extended quantization
typedef struct TfLiteQuantization {
  TfLiteQuantizationType type;  // kTfLiteNoQuantization or kTfLiteAffineQuantization
  void* params;  // Points to TfLiteAffineQuantization if type is affine
} TfLiteQuantization;

typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray* scale;       // Per-channel scales
  TfLiteIntArray* zero_point;    // Per-channel zero points
  int32_t quantized_dimension;   // Axis for per-channel quantization
} TfLiteAffineQuantization;
```

---

## InterpreterBuilder

The `InterpreterBuilder` constructs an Interpreter from a FlatBuffer model and
an OpResolver.

### Usage

```cpp
// From: tensorflow/lite/core/interpreter_builder.h

// Create builder
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder builder(*model, resolver);

// Build interpreter
builder(&interpreter);

// With delegate
builder.AddDelegate(my_delegate);
builder(&interpreter);
```

### InterpreterBuilder Methods

| Method | Description |
|--------|-------------|
| `operator()` | Build the interpreter |
| `AddDelegate` | Add a delegate for hardware acceleration |
| `SetNumThreads` | Set the number of threads for CPU execution |

---

## Op Registration

### OpResolver

The `OpResolver` maps operation codes to their kernel implementations:

```cpp
// From: tensorflow/lite/core/api/op_resolver.h

class OpResolver {
 public:
  // Find a builtin op
  virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                           int version) const = 0;

  // Find a custom op
  virtual const TfLiteRegistration* FindOp(const char* op,
                                           int version) const = 0;

  // Get delegates for the interpreter
  virtual TfLiteDelegatePtrVector GetDelegates(int num_threads) const;

  // Get delegate creators
  virtual TfLiteDelegateCreators GetDelegateCreators() const;
};
```

### BuiltinOpResolver

The `BuiltinOpResolver` registers all built-in TFLite operations:

```cpp
// From: tensorflow/lite/core/kernels/register.h

class BuiltinOpResolver : public MutableOpResolver {
 public:
  BuiltinOpResolver();
};

// Without default delegates
class BuiltinOpResolverWithoutDefaultDelegates : public MutableOpResolver {
 public:
  BuiltinOpResolverWithoutDefaultDelegates();
};
```

### Custom Op Registration

```cpp
// Register a custom operation
class MyCustomOpResolver : public tflite::MutableOpResolver {
 public:
  MyCustomOpResolver() {
    // Register builtin ops
    AddBuiltin(BuiltinOperator_ADD, Register_ADD());
    // Register custom op
    AddCustom("MyCustomOp", Register_MyCustomOp());
  }
};

// TfLiteRegistration structure
typedef struct TfLiteRegistration {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  int32_t builtin_code;
  const char* custom_name;
  int version;
} TfLiteRegistration;
```

### TfLiteRegistration Fields

| Field | Description |
|-------|-------------|
| `init` | Called once when the op is added to the graph. Returns opaque data. |
| `free` | Called to release the opaque data from `init`. |
| `prepare` | Called during `AllocateTensors()`. Sets output shapes. |
| `invoke` | Called during `Invoke()`. Performs the computation. |
| `builtin_code` | Builtin op code (0 for custom ops). |
| `custom_name` | Custom op name (nullptr for builtin ops). |
| `version` | Op version number. |

---

## Memory Planning

Memory planning optimizes tensor memory allocation to minimize peak memory usage.

### ArenaAllocator

The arena-based allocator manages a contiguous memory region for tensors:

```cpp
// SimpleMemoryAllocator allocates from a fixed buffer
class SimpleMemoryAllocator {
 public:
  TfLiteTensor* AllocateTensor(size_t size);
  void Reset();
};
```

### Memory Planning Strategy

1. **Liveness analysis**: Determine which tensors are live at each point in execution
2. **Arena allocation**: Allocate tensors from a shared memory arena
3. **Reuse**: Tensors with non-overlapping lifetimes share the same memory
4. **Persistent allocation**: Variable tensors get permanent allocation

### Memory Arena Layout

```
+---+---+---+---+---+---+---+---+---+---+---+
| T0| T1| T2| T3| T4|   | T5| T6|   | T7|   |
+---+---+---+---+---+---+---+---+---+---+---+
      ^       ^           ^           ^
      |       |           |           |
   (T0 freed, reused by T3 and T5)   |
                                    (T1 freed, reused by T7)
```

### Dynamic Tensors

Tensors with dynamic shapes are allocated at runtime during `invoke()`.
This can cause memory fragmentation and reallocation overhead.

---

## Delegates Overview

Delegates enable hardware acceleration for TFLite operations.

### Delegate Interface

```cpp
typedef struct TfLiteDelegate {
  int64_t node_index;     // First node handled by this delegate
  TfLiteStatus (*Prepare)(TfLiteContext* context, TfLiteDelegate* delegate);
  TfLiteStatus (*CopyFromBufferHandle)(TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        BufferHandle buffer_handle,
                                        TfLiteTensor* tensor);
  TfLiteStatus (*CopyToBufferHandle)(TfLiteContext* context,
                                       TfLiteDelegate* delegate,
                                       BufferHandle buffer_handle,
                                       TfLiteTensor* tensor);
  void (*FreeBufferHandle)(TfLiteContext* context, TfLiteDelegate* delegate,
                            BufferHandle* handle);
  int64_t flags;
} TfLiteDelegate;
```

### Available Delegates

| Delegate | Platform | Description |
|----------|----------|-------------|
| **GPU** | Android, iOS, Linux, macOS | OpenGL/Metal/OpenCL/Vulkan acceleration |
| **NNAPI** | Android | Android Neural Networks API |
| **Core ML** | iOS | Apple Core ML framework |
| **Hexagon** | Android | Qualcomm Hexagon DSP |
| **XNNPACK** | All | Optimized CPU kernels |
| **Flex** | All | TensorFlow op compatibility |
| **Edge TPU** | Coral | Google Edge TPU |
| **Metal** | iOS/macOS | Apple Metal GPU |
| **GPU CL** | Android/Linux | OpenCL GPU acceleration |
| **GPU Vk** | Android/Linux | Vulkan GPU acceleration |

### Delegate Application

```cpp
// Apply delegate to interpreter
TfLiteDelegate* delegate = TfLiteGpuDelegateV2Create(&options);
interpreter->ModifyGraphWithDelegate(delegate);
```

### Delegation Process

1. **Node partitioning**: Identify which nodes the delegate can handle
2. **Graph rewrite**: Replace supported nodes with delegate nodes
3. **Buffer management**: Set up data transfer between TF and delegate
4. **Execution**: Delegate handles supported nodes, TF handles the rest

---

## TFLite C API

### Model Loading

```c
// Create model from file
TfLiteModel* model = TfLiteModelCreateFromFile("model.tflite");

// Create interpreter options
TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
TfLiteInterpreterOptionsSetNumThreads(options, 4);

// Create interpreter
TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

// Allocate tensors
TfLiteInterpreterAllocateTensors(interpreter);

// Get input tensor
TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
TfLiteTensorCopyFromBuffer(input, input_data, input_size);

// Invoke
TfLiteInterpreterInvoke(interpreter);

// Get output tensor
const TfLiteTensor* output = TfLiteInterpreterGetOutputTensor(interpreter, 0);
TfLiteTensorCopyToBuffer(output, output_data, output_size);

// Cleanup
TfLiteInterpreterDelete(interpreter);
TfLiteInterpreterOptionsDelete(options);
TfLiteModelDelete(model);
```

### C API Types

| Type | Description |
|------|-------------|
| `TfLiteModel` | Opaque model handle |
| `TfLiteInterpreter` | Opaque interpreter handle |
| `TfLiteInterpreterOptions` | Opaque options handle |
| `TfLiteTensor` | Tensor structure (public) |
| `TfLiteDelegate` | Opaque delegate handle |

---

## TFLite Python API

### Basic Usage

```python
import numpy as np
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
```

### Interpreter Methods

| Method | Description |
|--------|-------------|
| `__init__` | Create interpreter from model path or content |
| `allocate_tensors()` | Allocate memory for tensors |
| `invoke()` | Run inference |
| `get_input_details()` | Get input tensor metadata |
| `get_output_details()` | Get output tensor metadata |
| `get_tensor_details()` | Get all tensor metadata |
| `get_tensor(index)` | Get tensor data by index |
| `set_tensor(index, value)` | Set tensor data by index |
| `reset_all_variables()` | Reset variable tensors |
| `get_signature_list()` | List available signatures |
| `get_signature_runner(key)` | Get runner for a signature |

### Input/Output Details

```python
input_details = interpreter.get_input_details()
# Returns: [{
#   'name': 'input',
#   'index': 0,
#   'shape': array([  1, 224, 224,   3], dtype=int32),
#   'shape_signature': array([ -1, 224, 224,   3], dtype=int32),
#   'dtype': numpy.float32,
#   'quantization': (0.0, 0),
#   'quantization_parameters': {...},
# }]
```

### Signature Runner

```python
# Get signature runner
runner = interpreter.get_signature_runner('serving_default')

# Run with named inputs
output = runner(input=my_input_data)
```

---

## Model Versioning

### Op Versioning

Each TFLite operation has a version number. Newer versions may introduce
breaking changes:

```flatbuffers
table OperatorCode {
  builtin_code: BuiltinOperator;
  custom_code: string;
  version: int;  // Op version (1 = original, 2+ = updated)
}
```

### Forward Compatibility

TFLite maintains forward compatibility: older runtimes can run models with
newer op versions (up to a limit). The runtime checks op versions and
reports errors if unsupported versions are encountered.

### Schema Versioning

The model schema has its own version number. When the schema changes
incompatibly, the version number is incremented.

### Op Version Upgrades

```python
# Upgrade schema to latest version
from tensorflow.lite.tools import visualize
visualize.update_tensor_names(model_data)
```

---

## Flex Delegate

The Flex delegate enables running TensorFlow operations that are not natively
supported in TFLite.

### How It Works

1. During conversion, unsupported TF ops are marked as Flex ops
2. At runtime, the Flex delegate handles these ops by delegating to the
   TensorFlow runtime
3. This requires the TensorFlow shared library to be available

### Enabling Flex Ops

```python
# During conversion
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,  # Enable Flex ops
]
tflite_model = converter.convert()
```

### Flex Op Considerations

- Increases binary size (requires TF runtime)
- Not suitable for constrained environments
- Performance may be lower than native TFLite ops
- Useful for gradual migration from TF to TFLite

---

## TFLite vs TensorFlow Comparison

| Aspect | TensorFlow | TensorFlow Lite |
|--------|-----------|-----------------|
| **Model format** | SavedModel, GraphDef | FlatBuffer (.tflite) |
| **Runtime size** | ~100+ MB | ~1-5 MB |
| **Execution mode** | Eager, Graph | Graph only |
| **Training** | Supported | Inference only |
| **Dynamic shapes** | Full support | Limited support |
| **Custom ops** | Python, C++ | C/C++ only |
| **Hardware targets** | Server, Desktop | Mobile, Embedded, Edge |
| **Quantization** | Via tf.quantization | Built-in post-training quantization |
| **Delegates** | N/A (uses device directly) | GPU, NNAPI, CoreML, Hexagon, etc. |
| **Ops coverage** | Full TF op set | Subset of TF ops |

### Supported Op Coverage

TFLite supports a subset of TensorFlow operations. Common categories:

| Category | Supported Ops |
|----------|--------------|
| **Arithmetic** | Add, Sub, Mul, Div, FloorDiv, FloorMod |
| **Activation** | Relu, Relu6, Sigmoid, Tanh, Elu, LeakyRelu |
| **Normalization** | BatchNorm, LayerNorm, InstanceNorm |
| **Convolution** | Conv2D, DepthwiseConv2D, TransposeConv |
| **Pooling** | MaxPool, AveragePool, L2Pool |
| **Fully Connected** | FullyConnected |
| **Reduction** | Mean, Sum, Max, Min, Prod, Any, All |
| **Concat/Split** | Concatenation, Split, Slice, Gather |
| **Shape** | Reshape, Transpose, ExpandDims, Squeeze |
| **Comparison** | Equal, NotEqual, Greater, Less |
| **Logical** | LogicalAnd, LogicalOr, LogicalNot |
| **NN** | Softmax, LogSoftmax, L2Normalization |
| **Embedding** | EmbeddingLookup, EmbeddingBag |
| **Sequence** | LSTM, UnidirectionalSequenceLSTM, BidirectionalSequenceLSTM |

---

## Model Size Optimization

### Techniques

1. **Post-training quantization**: Reduce weight precision
2. **Weight pruning**: Remove unimportant weights
3. **Weight clustering**: Group similar weight values
4. **Buffer deduplication**: Remove duplicate constant buffers

### Quantization Impact on Size

| Precision | Relative Size |
|-----------|--------------|
| FP32 | 1.0x (baseline) |
| FP16 | 0.5x |
| INT8 | 0.25x |
| INT4 (experimental) | 0.125x |

### Buffer Deduplication

```python
# From: tensorflow/lite/python/convert.py
# Deduplicate read-only buffers to reduce model size
```

### Selective Op Building

For minimal binary size, build TFLite with only the required ops:

```python
# Use only TFLite builtins (no Flex ops)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
```

### Model Metadata

```python
# Add metadata for model information
from tensorflow.lite.python import metadata
metadata.set_model_metadata(model_data, {"name": "my_model", "version": "1.0"})
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `lite/core/interpreter.h` | Interpreter class |
| `lite/core/interpreter_builder.h` | Interpreter builder |
| `lite/core/model.h` | Model loading |
| `lite/core/api/op_resolver.h` | Op resolver interface |
| `lite/core/api/error_reporter.h` | Error reporting |
| `lite/core/api/profiler.h` | Profiling interface |
| `lite/core/c/common.h` | C API common types |
| `lite/core/kernels/register.h` | Builtin op registration |
| `lite/core/subgraph.h` | Subgraph implementation |
| `lite/core/signature_runner.h` | Signature runner |
| `lite/schema/schema_generated.h` | Generated FlatBuffer types |
| `lite/schema/schema_utils.h` | Schema utilities |

---

## Advanced TFLite Topics

### Subgraph Execution

TFLite supports multiple subgraphs within a single model. The primary subgraph
(index 0) is the main computation, and additional subgraphs handle operations
like control flow:

```cpp
// From: tensorflow/lite/core/subgraph.h
class Subgraph {
 public:
  // Execution
  TfLiteStatus Invoke();

  // Tensor access
  TfLiteTensor* tensor(int tensor_index);
  const TfLiteTensor* tensor(int tensor_index) const;

  // Node access
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const;

  // Graph structure
  size_t tensors_size() const;
  size_t nodes_size() const;
  const std::vector<int>& inputs() const;
  const std::vector<int>& outputs() const;
  const std::vector<int>& variables() const;
  const std::vector<int>& execution_plan() const;

  // Memory management
  TfLiteStatus AllocateTensors();
  TfLiteStatus PrepareOpsAndTensors();

  // Delegation
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);
};
```

### SignatureRunner

The `SignatureRunner` provides named access to model signatures:

```cpp
// From: tensorflow/lite/core/signature_runner.h
class SignatureRunner {
 public:
  // Get input/output names
  const std::vector<const char*>& input_names() const;
  const std::vector<const char*>& output_names() const;

  // Get input/output tensors
  TfLiteTensor* input_tensor(const char* name);
  const TfLiteTensor* output_tensor(const char* name) const;

  // Run inference
  TfLiteStatus Invoke();
};
```

### Interpreter Options

The `InterpreterOptions` class provides fine-grained control over execution:

```cpp
// From: tensorflow/lite/interpreter_options.h
class InterpreterOptions {
 public:
  // Thread count for CPU execution
  void SetNumThreads(int num_threads);

  // Enable/disable NNAPI delegate
  void SetUseNNAPI(bool enable);

  // Allow FP16 precision for FP32 computation
  void SetAllowFp16PrecisionForFp32(bool allow);

  // Enable dynamic tensor allocation
  void SetDynamicAllocationForLargeTensors(bool enable);

  // Set error reporter
  void SetErrorReporter(ErrorReporter* reporter);
};
```

### Profiling Infrastructure

TFLite provides profiling for performance analysis:

```cpp
// From: tensorflow/lite/core/api/profiler.h
class Profiler {
 public:
  // Profile event types
  enum class EventType {
    DEFAULT,
    OPERATOR_INVOKE_EVENT,
    DELEGATE_OPERATOR_INVOKE_EVENT,
    TE_NNAPI_DELEGATE_EVENT,
  };

  // Add a profiling event
  virtual void AddEvent(const char* tag, EventType event_type,
                         uint64_t metric, int64_t op_index) = 0;

  // Begin/end profiling scope
  virtual void BeginEvent(const char* tag, EventType event_type,
                           int64_t op_index) = 0;
  virtual void EndEvent() = 0;
};
```

### Error Reporting

```cpp
// From: tensorflow/lite/core/api/error_reporter.h
class ErrorReporter {
 public:
  virtual int Report(const char* format, ...) = 0;
};

// Default error reporter (writes to stderr)
ErrorReporter* DefaultErrorReporter();
```

### Model Verification

TFLite includes tools for verifying model integrity:

```cpp
// From: tensorflow/lite/core/tools/verifier.h
// Verifies TFLite model integrity
bool Verify(const uint8_t* model_data, size_t model_size);
```

### FlatBuffer Conversions

```cpp
// From: tensorflow/lite/core/api/flatbuffer_conversions.h
// Converts between FlatBuffer types and TFLite runtime types
TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type);
TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_code,
                         void* builtin_data);
```

### TfLiteContext

The `TfLiteContext` is passed to all op kernel functions and provides access
to the interpreter's functionality:

```cpp
typedef struct TfLiteContext {
  // Tensor access
  TfLiteTensor* (*GetTensor)(const TfLiteContext* context, int index);
  const TfLiteTensor* (*GetEvalTensor)(const TfLiteContext* context, int index);

  // Memory management
  TfLiteStatus (*AllocateTensor)(TfLiteContext* context, int index);
  TfLiteStatus (*ResizeTensor)(TfLiteContext* context, TfLiteTensor* tensor,
                                TfLiteIntArray* new_size);

  // Node operations
  int (*GetNode)(TfLiteContext* context, int node_index, TfLiteNode** node);
  int (*GetExecutionPlan)(TfLiteContext* context, int* num_nodes, int* nodes);

  // Error reporting
  void (*ReportError)(TfLiteContext* context, const char* format, ...);

  // Delegation
  TfLiteStatus (*ReplaceNodeSubsetsWithDelegateKernels)(
      TfLiteContext* context, TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

  // Recommended number of threads
  int recommended_num_threads;
} TfLiteContext;
```

### TfLiteNode

```cpp
typedef struct TfLiteNode {
  TfLiteIntArray* inputs;      // Input tensor indices
  TfLiteIntArray* outputs;     // Output tensor indices
  TfLiteIntArray* intermediates; // Intermediate tensor indices
  void* user_data;              // Op-specific data from init()
  void* builtin_data;           // Builtin op parameters
  const TfLiteRegistration* registration;  // Op registration
  TfLiteDelegate* delegate;     // Associated delegate (if any)
} TfLiteNode;
```

### Builtin Op Parameters

Each builtin op has a corresponding parameter structure:

```cpp
// Conv2D parameters
typedef struct TfLiteConvParams {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
  TfLiteDilation dilation_width_factor;
  TfLiteDilation dilation_height_factor;
} TfLiteConvParams;

// FullyConnected parameters
typedef struct TfLiteFullyConnectedParams {
  TfLiteFusedActivation activation;
  TfLiteFullyConnectedWeightsFormat weights_format;
  bool keep_num_dims;
  bool asymmetric_quantize_inputs;
} TfLiteFullyConnectedParams;

// LSTM parameters
typedef struct TfLiteLSTMParams {
  TfLiteFusedActivation activation;
  float cell_clip;
  float proj_clip;
  TfLiteLSTMKernelType kernel_type;
  bool asymmetric_quantize_inputs;
} TfLiteLSTMParams;
```

### Model Building API

TFLite provides a programmatic model building API:

```cpp
// From: tensorflow/lite/core/model_building.h
// API for building TFLite models programmatically
```

This is useful for:
- Generating test models
- Building models without the full TensorFlow conversion pipeline
- Creating specialized models at runtime

### NNAPI Delegate Details

The NNAPI delegate maps TFLite operations to Android Neural Networks API:

```cpp
// From: tensorflow/lite/nnapi/nnapi_handler.h
// NNAPI delegate implementation
```

Supported operations vary by Android version and device. The delegate queries
device capabilities and falls back to CPU for unsupported operations.

### GPU Delegate Details

The GPU delegate uses OpenGL/Metal/OpenCL for acceleration:

```cpp
// GPU delegate creation
TfLiteDelegate* TfLiteGpuDelegateV2Create(const TfLiteGpuDelegateOptions2* options);
void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate);

// Options
typedef struct TfLiteGpuDelegateOptions2 {
  bool is_precision_loss_allowed;
  TfLiteGpuInferencePriority inference_priority1;
  TfLiteGpuInferencePriority inference_priority2;
  TfLiteGpuInferencePriority inference_priority3;
  TfLiteGpuInferenceUsage inference_usage;
} TfLiteGpuDelegateOptions2;
```

### Async Execution

TFLite supports async execution for pipelined inference:

```cpp
// From: tensorflow/lite/core/async/async_signature_runner.h
class AsyncSignatureRunner {
 public:
  TfLiteStatus Invoke();
  TfLiteStatus Wait();
};
```
