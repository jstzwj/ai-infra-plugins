# TensorFlow C API Reference

## Table of Contents

1. [C API Overview](#c-api-overview)
2. [TF_Status](#tf_status)
3. [TF_Buffer](#tf_buffer)
4. [TF_Tensor](#tf_tensor)
5. [TF_SessionOptions](#tf_sessionoptions)
6. [TF_Graph](#tf_graph)
7. [TF_Operation](#tf_operation)
8. [TF_Output and TF_Input](#tf_output-and-tf_input)
9. [TF_Session](#tf_session)
10. [TF_OperationDescription](#tf_operationdescription)
11. [Attribute Setting Functions](#attribute-setting-functions)
12. [Eager C API](#eager-c-api)
13. [SavedModel C API](#savedmodel-c-api)
14. [Memory Management Patterns](#memory-management-patterns)

---

## C API Overview

The TensorFlow C API provides a C-language interface to TensorFlow's core
functionality. It is the foundation for language bindings (Python, Go, Rust,
etc.) and enables TensorFlow usage from any C-compatible language.

### Design Principles

- **Opaque pointers**: All objects are accessed through opaque struct pointers.
- **Explicit lifecycle management**: Creation and destruction functions for all
  objects. Deletion functions are safe to call on `nullptr`.
- **Status-based error handling**: Every operation that can fail takes a
  `TF_Status*` parameter.
- **C89/C99 compatible**: No C++ features required; `unsigned char` used for
  booleans.
- **Thread safety**: Graphs are thread-safe for reads. Sessions are
  thread-safe for concurrent `Run` calls.

### Header Files

```c
#include "tensorflow/c/c_api.h"          // Core C API
#include "tensorflow/c/eager/c_api.h"    // Eager execution C API
#include "tensorflow/c/tf_status.h"      // Status management
#include "tensorflow/c/tf_buffer.h"      // Buffer management
#include "tensorflow/c/tf_tensor.h"      // Tensor operations
#include "tensorflow/c/tf_datatype.h"    // Data type definitions
#include "tensorflow/c/tf_attrtype.h"    // Attribute type definitions
#include "tensorflow/c/tf_tstring.h"    // String type definitions
```

### Naming Convention

All functions and types use the `TF_` prefix:
- Types: `TF_Status`, `TF_Graph`, `TF_Session`, `TF_Tensor`
- Functions: `TF_NewGraph`, `TF_DeleteGraph`, `TF_SessionRun`
- Enum values: `TF_FLOAT`, `TF_INT32`, `TF_OK`

---

## TF_Status

### Overview

`TF_Status` holds error information from API calls. Every function that can
fail accepts a `TF_Status*` parameter that is cleared on success and filled
with error details on failure.

### Functions

```c
// Create a new status object
TF_CAPI_EXPORT extern TF_Status* TF_NewStatus(void);

// Delete a status object (safe on nullptr)
TF_CAPI_EXPORT extern void TF_DeleteStatus(TF_Status* status);

// Get the status code
TF_CAPI_EXPORT extern TF_Code TF_GetCode(const TF_Status* status);

// Get the human-readable error message
TF_CAPI_EXPORT extern const char* TF_Message(const TF_Status* status);

// Set the status explicitly
TF_CAPI_EXPORT extern void TF_SetStatus(TF_Status* status, TF_Code code,
                                         const char* message);
```

### Status Codes

```c
typedef enum TF_Code {
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15,
  TF_UNAUTHENTICATED = 16,
} TF_Code;
```

### Usage Pattern

```c
TF_Status* status = TF_NewStatus();

// Call a function that may fail
TF_Graph* graph = TF_NewGraph();
TF_OperationDescription* desc = TF_NewOperation(graph, "Const", "my_const");

// Check status
if (TF_GetCode(status) != TF_OK) {
    printf("Error: %s\n", TF_Message(status));
    // Handle error
}

// Clean up
TF_DeleteStatus(status);
```

---

## TF_Buffer

### Overview

`TF_Buffer` represents a raw byte buffer used for serialized protocol buffers
and other binary data.

### Functions

```c
// Create a new empty buffer
TF_CAPI_EXPORT extern TF_Buffer* TF_NewBuffer(void);

// Create a buffer from existing data (copies the data)
TF_CAPI_EXPORT extern TF_Buffer* TF_NewBufferFromString(
    const void* proto, size_t proto_len);

// Get buffer from a TF_Buffer (returns pointer to internal data)
TF_CAPI_EXPORT extern void TF_GetBuffer(TF_Buffer* buffer);

// Delete a buffer
TF_CAPI_EXPORT extern void TF_DeleteBuffer(TF_Buffer* buffer);
```

### Structure

```c
typedef struct TF_Buffer {
    const void* data;   // Pointer to buffer data
    size_t length;      // Size of data in bytes
    void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;
```

### Usage

```c++
// Create buffer from serialized proto
TF_Buffer* graph_def = TF_NewBufferFromString(
    serialized_graph_data, serialized_graph_size);

// Read file into buffer
TF_Buffer* ReadFile(const char* filename) {
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBufferFromString(data, size);
    free(data);
    return buf;
}
```

---

## TF_Tensor

### Overview

`TF_Tensor` represents a multidimensional array of data. It is the core data
structure for passing data in and out of TensorFlow operations.

### Creation and Destruction

```c
// Create a new tensor (takes ownership of data)
TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
    TF_DataType type,
    const int64_t* dims,       // Array of dimension sizes
    int num_dims,              // Number of dimensions
    void* data,                // Raw data pointer
    size_t len,                // Data length in bytes
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg);

// Allocate a new tensor (data is zero-initialized)
TF_CAPI_EXPORT extern TF_Tensor* TF_AllocateTensor(
    TF_DataType type,
    const int64_t* dims,
    int num_dims,
    size_t len);

// Delete a tensor (safe on nullptr)
TF_CAPI_EXPORT extern void TF_DeleteTensor(TF_Tensor* tensor);
```

### Tensor Accessors

```c
// Get data type
TF_CAPI_EXPORT extern TF_DataType TF_TensorType(const TF_Tensor* tensor);

// Get number of dimensions
TF_CAPI_EXPORT extern int TF_NumDims(const TF_Tensor* tensor);

// Get size of a specific dimension
TF_CAPI_EXPORT extern int64_t TF_Dim(const TF_Tensor* tensor, int dim_index);

// Get total data size in bytes
TF_CAPI_EXPORT extern size_t TF_TensorByteSize(const TF_Tensor* tensor);

// Get pointer to raw data
TF_CAPI_EXPORT extern void* TF_TensorData(const TF_Tensor* tensor);

// Get total number of elements
TF_CAPI_EXPORT extern int64_t TF_TensorElementCount(const TF_Tensor* tensor);
```

### Data Types

```c
typedef enum TF_DataType {
  TF_FLOAT = 1,
  TF_DOUBLE = 2,
  TF_INT32 = 3,
  TF_UINT8 = 4,
  TF_INT16 = 5,
  TF_INT8 = 6,
  TF_STRING = 7,
  TF_COMPLEX64 = 8,
  TF_INT64 = 9,
  TF_BOOL = 10,
  TF_QINT8 = 11,
  TF_QUINT8 = 12,
  TF_QINT32 = 13,
  TF_BFLOAT16 = 14,
  TF_QINT16 = 15,
  TF_QUINT16 = 16,
  TF_UINT16 = 17,
  TF_COMPLEX128 = 18,
  TF_HALF = 19,
  TF_RESOURCE = 20,
  TF_VARIANT = 21,
  TF_UINT32 = 22,
  TF_UINT64 = 23,
  TF_FLOAT8_E4M3FN = 24,
  TF_FLOAT8_E5M2 = 25,
  TF_FLOAT8_E4M3FNUZ = 26,
  TF_FLOAT8_E5M2FNUZ = 27,
  TF_INT4 = 29,
} TF_DataType;
```

### Creating Tensors

```c++
// Create a float32 tensor with shape [2, 3]
int64_t dims[] = {2, 3};
float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

TF_Tensor* tensor = TF_NewTensor(
    TF_FLOAT, dims, 2,
    data, sizeof(data),
    [](void* data, size_t len, void* arg) {},  // No-op deallocator
    nullptr);

// Allocate a zero-initialized tensor
int64_t dims[] = {10, 20};
TF_Tensor* zeros = TF_AllocateTensor(TF_FLOAT, dims, 2, 10 * 20 * sizeof(float));

// Create int32 tensor
int32_t int_data[] = {1, 2, 3, 4};
int64_t int_dims[] = {4};
TF_Tensor* int_tensor = TF_NewTensor(
    TF_INT32, int_dims, 1,
    int_data, sizeof(int_data),
    nullptr, nullptr);
```

### Tensor Serialization

```c
// Serialize tensor to protocol buffer
TF_CAPI_EXPORT extern void TF_TensorToProto(const TF_Tensor* tensor,
                                             TF_Buffer* proto,
                                             TF_Status* status);

// Deserialize protocol buffer to tensor
TF_CAPI_EXPORT extern void TF_TensorFromProto(const TF_Buffer* proto,
                                               TF_Tensor* tensor,
                                               TF_Status* status);
```

---

## TF_SessionOptions

### Functions

```c
// Create new session options
TF_CAPI_EXPORT extern TF_SessionOptions* TF_NewSessionOptions(void);

// Set the target (e.g., "local", "ip:port")
TF_CAPI_EXPORT extern void TF_SetTarget(TF_SessionOptions* options,
                                         const char* target);

// Set configuration from serialized ConfigProto
TF_CAPI_EXPORT extern void TF_SetConfig(TF_SessionOptions* options,
                                         const void* proto, size_t proto_len,
                                         TF_Status* status);

// Delete session options
TF_CAPI_EXPORT extern void TF_DeleteSessionOptions(TF_SessionOptions* options);
```

### Configuration

```c++
// Create options with GPU configuration
TF_SessionOptions* options = TF_NewSessionOptions();

// Set target to specific server
TF_SetTarget(options, "grpc://localhost:2222");

// Set config (e.g., GPU options)
tensorflow::ConfigProto config;
config.set_allow_soft_placement(true);
config.mutable_gpu_options()->set_allow_growth(true);

std::string config_str;
config.SerializeToString(&config_str);

TF_Status* status = TF_NewStatus();
TF_SetConfig(options, config_str.data(), config_str.size(), status);

if (TF_GetCode(status) != TF_OK) {
    printf("Config error: %s\n", TF_Message(status));
}

TF_DeleteStatus(status);
TF_DeleteSessionOptions(options);
```

---

## TF_Graph

### Overview

`TF_Graph` represents a computation graph. Graphs are thread-safe for
concurrent reads but require external synchronization for modifications.

### Lifecycle

```c
// Create a new graph
TF_CAPI_EXPORT extern TF_Graph* TF_NewGraph(void);

// Delete a graph (valid until no sessions reference it)
TF_CAPI_EXPORT extern void TF_DeleteGraph(TF_Graph* graph);
```

### Tensor Shape Operations

```c
// Set tensor shape (merges with existing shape)
TF_CAPI_EXPORT extern void TF_GraphSetTensorShape(
    TF_Graph* graph, TF_Output output,
    const int64_t* dims, int num_dims,
    TF_Status* status);

// Get number of dimensions (-1 if unknown)
TF_CAPI_EXPORT extern int TF_GraphGetTensorNumDims(
    TF_Graph* graph, TF_Output output, TF_Status* status);

// Get tensor shape into provided dims array
TF_CAPI_EXPORT extern void TF_GraphGetTensorShape(
    TF_Graph* graph, TF_Output output,
    int64_t* dims, int num_dims,
    TF_Status* status);
```

### Graph Import

```c
// Options for graph import
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;

TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions(void);
void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions* opts);

// Set prefix for imported nodes
void TF_ImportGraphDefOptionsSetPrefix(
    TF_ImportGraphDefOptions* opts, const char* prefix);

// Set default device
void TF_ImportGraphDefOptionsSetDefaultDevice(
    TF_ImportGraphDefOptions* opts, const char* device);

// Import a GraphDef proto into the graph
void TF_GraphImportGraphDef(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options,
    TF_Status* status);

// Import with returned mappings
TF_ImportGraphDefResults* TF_GraphImportGraphDefWithResults(
    TF_Graph* graph, const TF_Buffer* graph_def,
    const TF_ImportGraphDefOptions* options,
    TF_Status* status);
```

### Graph Operations

```c
// Get operation by name
TF_CAPI_EXPORT extern TF_Operation* TF_GraphOperationByName(
    TF_Graph* graph, const char* name);

// Iterate operations
TF_CAPI_EXPORT extern TF_Operation* TF_GraphNextOperation(
    TF_Graph* graph, size_t* state);

// Get number of operations
int TF_GraphNumOpettes(TF_Graph* graph);  // Deprecated

// Get OpDef for an operation type
TF_CAPI_EXPORT extern void TF_GraphGetOpDef(
    TF_Graph* graph, const char* op_name,
    TF_Buffer* output_buf, TF_Status* status);

// Get graph versions
TF_CAPI_EXPORT extern void TF_GraphVersions(
    TF_Graph* graph, TF_Buffer* output_buf, TF_Status* status);

// Export to GraphDef
TF_CAPI_EXPORT extern void TF_GraphToGraphDef(
    TF_Graph* graph, TF_Buffer* output_buf, TF_Status* status);
```

### Complete Graph Building Example

```c
TF_Status* status = TF_NewStatus();
TF_Graph* graph = TF_NewGraph();

// Create a constant tensor
int64_t dims[] = {2, 2};
float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
TF_Tensor* const_tensor = TF_NewTensor(
    TF_FLOAT, dims, 2, values, sizeof(values),
    [](void*, size_t, void*) {}, nullptr);

// Create Const operation
TF_OperationDescription* const_desc =
    TF_NewOperation(graph, "Const", "my_const");
TF_SetAttrTensor(const_desc, "value", const_tensor, status);
TF_SetAttrType(const_desc, "dtype", TF_FLOAT);
TF_Operation* const_op = TF_FinishOperation(const_desc, status);

// Create Placeholder operation
TF_OperationDescription* placeholder_desc =
    TF_NewOperation(graph, "Placeholder", "input");
TF_SetAttrType(placeholder_desc, "dtype", TF_FLOAT);
TF_Operation* placeholder_op = TF_FinishOperation(placeholder_desc, status);

// Create MatMul operation
TF_Output matmul_inputs[] = {
    {const_op, 0},
    {placeholder_op, 0}
};
TF_OperationDescription* matmul_desc =
    TF_NewOperation(graph, "MatMul", "result");
TF_AddInput(matmul_desc, matmul_inputs[0]);
TF_AddInput(matmul_desc, matmul_inputs[1]);
TF_SetAttrBool(matmul_desc, "transpose_a", 0);
TF_SetAttrBool(matmul_desc, "transpose_b", 0);
TF_Operation* matmul_op = TF_FinishOperation(matmul_desc, status);

// Export to GraphDef
TF_Buffer* graph_def = TF_NewBuffer();
TF_GraphToGraphDef(graph, graph_def, status);

// Cleanup
TF_DeleteTensor(const_tensor);
TF_DeleteBuffer(graph_def);
TF_DeleteGraph(graph);
TF_DeleteStatus(status);
```

---

## TF_Operation

### Overview

`TF_Operation` represents an immutable operation in a graph. Operations are
valid until the graph is deleted.

### Query Functions

```c
// Get operation name
TF_CAPI_EXPORT extern const char* TF_OperationName(TF_Operation* oper);

// Get operation type (e.g., "MatMul", "Add")
TF_CAPI_EXPORT extern const char* TF_OperationOpType(TF_Operation* oper);

// Get device assignment
TF_CAPI_EXPORT extern const char* TF_OperationDevice(TF_Operation* oper);

// Get number of outputs
TF_CAPI_EXPORT extern int TF_OperationNumOutputs(TF_Operation* oper);

// Get type of a specific output
TF_CAPI_EXPORT extern TF_DataType TF_OperationOutputType(TF_Output oper_out);

// Get output list length for named output
TF_CAPI_EXPORT extern int TF_OperationOutputListLength(
    TF_Operation* oper, const char* arg_name, TF_Status* status);

// Get number of inputs
TF_CAPI_EXPORT extern int TF_OperationNumInputs(TF_Operation* oper);

// Get type of a specific input
TF_CAPI_EXPORT extern TF_DataType TF_OperationInputType(TF_Input oper_in);

// Get input list length for named input
TF_CAPI_EXPORT extern int TF_OperationInputListLength(
    TF_Operation* oper, const char* arg_name, TF_Status* status);

// Get the source of an input (returns the producing TF_Output)
TF_CAPI_EXPORT extern TF_Output TF_OperationInput(TF_Input oper_in);

// Get all inputs at once
TF_CAPI_EXPORT extern void TF_OperationAllInputs(
    TF_Operation* oper, TF_Output* inputs, int max_inputs);
```

### Consumer and Control Flow Functions

```c
// Get number of consumers of an output
TF_CAPI_EXPORT extern int TF_OperationOutputNumConsumers(TF_Output oper_out);

// Get consumers of an output
TF_CAPI_EXPORT extern int TF_OperationOutputConsumers(
    TF_Output oper_out, TF_Input* consumers, int max_consumers);

// Control inputs
TF_CAPI_EXPORT extern int TF_OperationNumControlInputs(TF_Operation* oper);
TF_CAPI_EXPORT extern int TF_OperationGetControlInputs(
    TF_Operation* oper, TF_Operation** control_inputs, int max);

// Control outputs
TF_CAPI_EXPORT extern int TF_OperationNumControlOutputs(TF_Operation* oper);
TF_CAPI_EXPORT extern int TF_OperationGetControlOutputs(
    TF_Operation* oper, TF_Operation** control_outputs, int max);
```

### Attribute Metadata

```c
typedef struct TF_AttrMetadata {
    unsigned char is_list;       // 1 if the attr is a list
    int64_t list_size;           // List length if is_list
    TF_AttrType type;            // Element type
    int64_t total_size;          // Total size (string bytes, shape dims)
} TF_AttrMetadata;

// Get attribute metadata
TF_CAPI_EXPORT extern TF_AttrMetadata TF_OperationGetAttrMetadata(
    TF_Operation* oper, const char* attr_name, TF_Status* status);

// Get attribute values
TF_CAPI_EXPORT extern void TF_OperationGetAttrString(
    TF_Operation* oper, const char* attr_name,
    void* value, size_t max_length, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrInt(
    TF_Operation* oper, const char* attr_name,
    int64_t* value, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrFloat(
    TF_Operation* oper, const char* attr_name,
    float* value, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrBool(
    TF_Operation* oper, const char* attr_name,
    unsigned char* value, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrType(
    TF_Operation* oper, const char* attr_name,
    TF_DataType* value, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrTensor(
    TF_Operation* oper, const char* attr_name,
    TF_Tensor** value, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrShape(
    TF_Operation* oper, const char* attr_name,
    int64_t* value, int num_dims, TF_Status* status);

TF_CAPI_EXPORT extern void TF_OperationGetAttrValueProto(
    TF_Operation* oper, const char* attr_name,
    TF_Buffer* output, TF_Status* status);
```

---

## TF_Output and TF_Input

### TF_Output

Represents a specific output of an operation:

```c
typedef struct TF_Output {
    TF_Operation* oper;
    int index;  // Output index within the operation
} TF_Output;
```

### TF_Input

Represents a specific input of an operation:

```c
typedef struct TF_Input {
    TF_Operation* oper;
    int index;  // Input index within the operation
} TF_Input;
```

### Usage

```c
// Create output reference
TF_Output output = {operation, 0};  // First output of operation

// Get the source of an input
TF_Input input = {consumer_op, 0};
TF_Output source = TF_OperationInput(input);
// source.oper is the producing operation
// source.index is the producing output index
```

---

## TF_Session

### Session Lifecycle

```c
// Create a new session
TF_CAPI_EXPORT extern TF_Session* TF_NewSession(
    const TF_SessionOptions* options, TF_Status* status);

// Close a session (frees resources but session can be deleted later)
TF_CAPI_EXPORT extern void TF_CloseSession(TF_Session* session,
                                            TF_Status* status);

// Delete a session (must be closed first)
TF_CAPI_EXPORT extern void TF_DeleteSession(TF_Session* session,
                                             TF_Status* status);
```

### Session Run

```c
// Run the graph
TF_CAPI_EXPORT extern void TF_SessionRun(
    TF_Session* session,
    const TF_Buffer* run_options,     // RunOptions proto (can be NULL)

    // Inputs
    const TF_Output* inputs,          // Array of input TF_Output
    TF_Tensor* const* input_values,   // Array of input tensors
    int ninputs,                      // Number of inputs

    // Outputs
    const TF_Output* outputs,         // Array of output TF_Output
    TF_Tensor** output_values,        // Array to receive output tensors
    int noutputs,                     // Number of outputs

    // Target operations to run
    const TF_Operation* const* target_opers,
    int ntargets,

    // Output metadata
    TF_Buffer* run_metadata,          // RunMetadata proto (can be NULL)

    TF_Status* status);
```

### Complete Session Run Example

```c
TF_Status* status = TF_NewStatus();

// ... build graph as shown above ...

// Create session
TF_SessionOptions* sess_opts = TF_NewSessionOptions();
TF_Session* session = TF_NewSession(sess_opts, status);
TF_DeleteSessionOptions(sess_opts);

// Prepare inputs
TF_Output input_output = {placeholder_op, 0};
float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
int64_t input_dims[] = {2, 2};
TF_Tensor* input_tensor = TF_NewTensor(
    TF_FLOAT, input_dims, 2,
    input_data, sizeof(input_data),
    [](void*, size_t, void*) {}, nullptr);

// Prepare output
TF_Output output_output = {matmul_op, 0};
TF_Tensor* output_tensor = nullptr;

// Run
TF_SessionRun(session,
    nullptr,                          // No run options
    &input_output, &input_tensor, 1,  // One input
    &output_output, &output_tensor, 1, // One output
    nullptr, 0,                       // No targets
    nullptr,                          // No run metadata
    status);

if (TF_GetCode(status) == TF_OK) {
    float* results = (float*)TF_TensorData(output_tensor);
    int64_t num_elements = TF_TensorElementCount(output_tensor);
    for (int64_t i = 0; i < num_elements; i++) {
        printf("output[%lld] = %f\n", i, results[i]);
    }
    TF_DeleteTensor(output_tensor);
}

TF_DeleteTensor(input_tensor);
TF_CloseSession(session, status);
TF_DeleteSession(session, status);
TF_DeleteStatus(status);
```

### Deprecated TF_Run

```c
// Legacy function (deprecated, use TF_SessionRun)
TF_CAPI_EXPORT extern void TF_Run(
    TF_Session* session,
    const TF_Buffer* run_options,
    const char** input_names,
    TF_Tensor** input_values,
    const char** output_names,
    TF_Tensor** output_values,
    const char** target_names,
    int ninputs, int noutputs, int ntargets,
    TF_Buffer* run_metadata,
    TF_Status* status);
```

---

## TF_OperationDescription

### Overview

`TF_OperationDescription` is used to build operations. Create with
`TF_NewOperation`, configure with `TF_SetAttr*` and `TF_AddInput*`, then
finalize with `TF_FinishOperation`.

### Creating Operations

```c
// Start building an operation
TF_OperationDescription* TF_NewOperation(
    TF_Graph* graph,
    const char* op_type,     // e.g., "MatMul", "Add", "Const"
    const char* oper_name);  // Unique name in the graph

// Thread-safe variant (requires holding graph lock)
TF_OperationDescription* TF_NewOperationLocked(
    TF_Graph* graph, const char* op_type, const char* oper_name);
```

### Device and Placement

```c
// Set the device for the operation
void TF_SetDevice(TF_OperationDescription* desc, const char* device);

// Co-locate with another operation
void TF_ColocateWith(TF_OperationDescription* desc, TF_Operation* op);
```

### Adding Inputs

```c
// Single input
void TF_AddInput(TF_OperationDescription* desc, TF_Input input);

// List of inputs
void TF_AddInputList(TF_OperationDescription* desc,
                     const TF_Output* inputs, int num_inputs);

// Control input (dependency)
void TF_AddControlInput(TF_OperationDescription* desc, TF_Operation* input);
```

### Finalizing

```c
// Add the operation to the graph
TF_Operation* TF_FinishOperation(TF_OperationDescription* desc,
                                  TF_Status* status);

// Thread-safe variant
TF_Operation* TF_FinishOperationLocked(TF_OperationDescription* desc,
                                        TF_Status* status);
```

On success, returns the new `TF_Operation*`. On failure, returns `nullptr`
and the graph is not modified. The `desc` is deleted in either case.

---

## Attribute Setting Functions

### String Attributes

```c
// Set a string attribute
void TF_SetAttrString(TF_OperationDescription* desc,
                       const char* attr_name,
                       const void* value, size_t length);

// Set a list of strings
void TF_SetAttrStringList(TF_OperationDescription* desc,
                           const char* attr_name,
                           const void* const* values,
                           const size_t* lengths,
                           int num_values);

// Set a function name attribute
void TF_SetAttrFuncName(TF_OperationDescription* desc,
                         const char* attr_name,
                         const char* value, size_t length);
```

### Numeric Attributes

```c
// Integer attributes
void TF_SetAttrInt(TF_OperationDescription* desc,
                    const char* attr_name, int64_t value);

void TF_SetAttrIntList(TF_OperationDescription* desc,
                        const char* attr_name,
                        const int64_t* values, int num_values);

// Float attributes
void TF_SetAttrFloat(TF_OperationDescription* desc,
                      const char* attr_name, float value);

void TF_SetAttrFloatList(TF_OperationDescription* desc,
                          const char* attr_name,
                          const float* values, int num_values);

// Boolean attributes (unsigned char used instead of bool)
void TF_SetAttrBool(TF_OperationDescription* desc,
                     const char* attr_name, unsigned char value);

void TF_SetAttrBoolList(TF_OperationDescription* desc,
                         const char* attr_name,
                         const unsigned char* values, int num_values);
```

### Type Attributes

```c
// Data type attribute
void TF_SetAttrType(TF_OperationDescription* desc,
                     const char* attr_name, TF_DataType value);

void TF_SetAttrTypeList(TF_OperationDescription* desc,
                         const char* attr_name,
                         const TF_DataType* values, int num_values);
```

### Shape Attributes

```c
// Shape attribute (-1 for unknown dimension, -1 num_dims for unknown rank)
void TF_SetAttrShape(TF_OperationDescription* desc,
                      const char* attr_name,
                      const int64_t* dims, int num_dims);

void TF_SetAttrShapeList(TF_OperationDescription* desc,
                          const char* attr_name,
                          const int64_t* const* dims,
                          const int* num_dims, int num_shapes);

// From serialized TensorShapeProto
void TF_SetAttrTensorShapeProto(TF_OperationDescription* desc,
                                 const char* attr_name,
                                 const void* proto, size_t proto_len,
                                 TF_Status* status);

void TF_SetAttrTensorShapeProtoList(TF_OperationDescription* desc,
                                     const char* attr_name,
                                     const void* const* protos,
                                     const size_t* proto_lens,
                                     int num_shapes,
                                     TF_Status* status);
```

### Tensor Attributes

```c
// Tensor attribute
void TF_SetAttrTensor(TF_OperationDescription* desc,
                       const char* attr_name,
                       TF_Tensor* value, TF_Status* status);

void TF_SetAttrTensorList(TF_OperationDescription* desc,
                           const char* attr_name,
                           TF_Tensor* const* values,
                           int num_values, TF_Status* status);
```

### Proto Attributes

```c
// Set attribute from serialized AttrValue proto
void TF_SetAttrValueProto(TF_OperationDescription* desc,
                           const char* attr_name,
                           const void* proto, size_t proto_len,
                           TF_Status* status);

// Placeholder attribute (references graph input)
void TF_SetAttrPlaceholder(TF_OperationDescription* desc,
                            const char* attr_name,
                            const char* placeholder);
```

### Common Operation Building Patterns

```c
// Build a Const op
TF_Operation* MakeConst(TF_Graph* graph, const char* name,
                         float* data, int64_t* dims, int ndims) {
    TF_Status* status = TF_NewStatus();

    TF_Tensor* tensor = TF_NewTensor(
        TF_FLOAT, dims, ndims, data,
        ElementCount(dims, ndims) * sizeof(float),
        [](void*, size_t, void*) {}, nullptr);

    TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
    TF_SetAttrTensor(desc, "value", tensor, status);
    TF_SetAttrType(desc, "dtype", TF_FLOAT);

    TF_Operation* op = TF_FinishOperation(desc, status);

    TF_DeleteTensor(tensor);
    TF_DeleteStatus(status);
    return op;
}

// Build a Placeholder op
TF_Operation* MakePlaceholder(TF_Graph* graph, const char* name,
                               TF_DataType type, int64_t* dims, int ndims) {
    TF_Status* status = TF_NewStatus();
    TF_OperationDescription* desc =
        TF_NewOperation(graph, "Placeholder", name);
    TF_SetAttrType(desc, "dtype", type);
    TF_SetAttrShape(desc, "shape", dims, ndims);

    TF_Operation* op = TF_FinishOperation(desc, status);
    TF_DeleteStatus(status);
    return op;
}

// Build a MatMul op
TF_Operation* MakeMatMul(TF_Graph* graph, const char* name,
                          TF_Operation* a, TF_Operation* b) {
    TF_Status* status = TF_NewStatus();
    TF_OperationDescription* desc =
        TF_NewOperation(graph, "MatMul", name);
    TF_AddInput(desc, {a, 0});
    TF_AddInput(desc, {b, 0});
    TF_SetAttrBool(desc, "transpose_a", 0);
    TF_SetAttrBool(desc, "transpose_b", 0);

    TF_Operation* op = TF_FinishOperation(desc, status);
    TF_DeleteStatus(status);
    return op;
}
```

---

## Eager C API

### Overview

The Eager C API provides immediate execution of TensorFlow operations without
building a graph. This is the foundation for TensorFlow 2.x's eager execution
mode.

### Context

```c
typedef struct TFE_Context TFE_Context;
typedef struct TFE_ContextOptions TFE_ContextOptions;

// Create context options
TFE_ContextOptions* TFE_NewContextOptions(void);

// Set config proto
void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options,
                                  const void* proto, size_t proto_len,
                                  TF_Status* status);

// Set async mode
void TFE_ContextOptionsSetAsync(TFE_ContextOptions* options,
                                 unsigned char enable);

// Set device placement policy
typedef enum TFE_ContextDevicePlacementPolicy {
  TFE_DEVICE_PLACEMENT_EXPLICIT = 0,
  TFE_DEVICE_PLACEMENT_WARN = 1,
  TFE_DEVICE_PLACEMENT_SILENT = 2,         // Default
  TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32 = 3,
} TFE_ContextDevicePlacementPolicy;

void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options,
    TFE_ContextDevicePlacementPolicy policy);

// Create context
TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts,
                             TF_Status* status);

// Delete context
void TFE_DeleteContext(TFE_Context* ctx);

// List available devices
TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx,
                                       TF_Status* status);

// Clear internal caches
void TFE_ContextClearCaches(TFE_Context* ctx);
```

### TensorHandle

```c
typedef struct TFE_TensorHandle TFE_TensorHandle;

// Create from TF_Tensor
TFE_TensorHandle* TFE_NewTensorHandle(const TF_Tensor* t,
                                       TF_Status* status);

// Delete handle
void TFE_DeleteTensorHandle(TFE_TensorHandle* h);

// Get data type
TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h);

// Get number of dimensions (blocks until shape is known)
int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status);

// Get dimension size (blocks)
int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                             TF_Status* status);

// Get total elements
int64_t TFE_TensorHandleNumElements(TFE_TensorHandle* h,
                                     TF_Status* status);

// Get device name
const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h,
                                         TF_Status* status);

// Get backing device name
const char* TFE_TensorHandleBackingDeviceName(TFE_TensorHandle* h,
                                               TF_Status* status);

// Resolve to TF_Tensor (blocks until computation completes)
TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h,
                                    TF_Status* status);

// Copy to device
TFE_TensorHandle* TFE_TensorHandleCopyToDevice(
    TFE_TensorHandle* h, TFE_Context* ctx,
    const char* device_name, TF_Status* status);
```

### Eager Operations

```c
typedef struct TFE_Op TFE_Op;

// Create an eager operation
TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_name,
                   TF_Status* status);

// Delete an eager operation
void TFE_DeleteOp(TFE_Op* op);

// Set the device for the operation
void TFE_OpSetDevice(TFE_Op* op, const char* device_name,
                      TF_Status* status);

// Add input
void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* h,
                     TF_Status* status);

// Add input list
void TFE_OpAddInputList(TFE_Op* op, TFE_TensorHandle* const* handles,
                         int num_handles, TF_Status* status);

// Set attributes (same pattern as graph API)
void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name,
                          const void* value, size_t length);
void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value);
void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value);
void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name,
                        unsigned char value);
void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name,
                        TF_DataType value);
void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name,
                         const int64_t* dims, int num_dims);

// Execute the operation
void TFE_Execute(TFE_Op* op, TFE_TensorHandle** outputs,
                  int* num_outputs, TF_Status* status);
```

### Eager Execution Example

```c
TF_Status* status = TF_NewStatus();

// Create context
TFE_ContextOptions* opts = TFE_NewContextOptions();
TFE_Context* ctx = TFE_NewContext(opts, status);
TFE_DeleteContextOptions(opts);

// Create input tensor
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
int64_t dims[] = {2, 2};
TF_Tensor* input_tensor = TF_NewTensor(
    TF_FLOAT, dims, 2, data, sizeof(data),
    [](void*, size_t, void*) {}, nullptr);

TFE_TensorHandle* input_handle =
    TFE_NewTensorHandle(input_tensor, status);
TF_DeleteTensor(input_tensor);

// Create and execute MatMul operation
TFE_Op* matmul_op = TFE_NewOp(ctx, "MatMul", status);
TFE_OpSetAttrBool(matmul_op, "transpose_a", 0);
TFE_OpSetAttrBool(matmul_op, "transpose_b", 0);
TFE_OpAddInput(matmul_op, input_handle, status);
TFE_OpAddInput(matmul_op, input_handle, status);

TFE_TensorHandle* output_handle = nullptr;
int num_outputs = 0;
TFE_Execute(matmul_op, &output_handle, &num_outputs, status);

// Get result
if (TF_GetCode(status) == TF_OK) {
    TF_Tensor* result = TFE_TensorHandleResolve(output_handle, status);
    float* result_data = (float*)TF_TensorData(result);
    // Use result_data...
    TF_DeleteTensor(result);
    TFE_DeleteTensorHandle(output_handle);
}

// Cleanup
TFE_DeleteOp(matmul_op);
TFE_DeleteTensorHandle(input_handle);
TFE_DeleteContext(ctx);
TF_DeleteStatus(status);
```

---

## SavedModel C API

### Loading SavedModels

```c
// Load a session from a SavedModel
TF_CAPI_EXPORT extern TF_Session* TF_LoadSessionFromSavedModel(
    const TF_SessionOptions* session_options,
    const TF_Buffer* run_options,        // Can be NULL
    const char* export_dir,              // Path to SavedModel directory
    const char* const* tags,             // Array of tag strings
    int tags_len,                        // Number of tags
    TF_Graph* graph,                     // Graph to populate
    TF_Buffer* meta_graph_def,           // Output MetaGraphDef (can be NULL)
    TF_Status* status);

// Get the number of graphs in a session
TF_CAPI_EXPORT extern int TF_SessionGraphCount(TF_Session* session,
                                                TF_Status* status);

// Get all graphs
TF_CAPI_EXPORT extern TF_Graph** TF_GetAllSessionGraphs(
    TF_Session* session, TF_Status* status);
```

### SavedModel Loading Example

```c
TF_Status* status = TF_NewStatus();
TF_Graph* graph = TF_NewGraph();
TF_SessionOptions* opts = TF_NewSessionOptions();

const char* tags[] = {"serve"};
TF_Session* session = TF_LoadSessionFromSavedModel(
    opts, nullptr,
    "/path/to/saved_model",
    tags, 1,
    graph,
    nullptr,  // meta_graph_def
    status);

if (TF_GetCode(status) == TF_OK) {
    // Find input and output operations
    TF_Operation* input_op = TF_GraphOperationByName(graph, "serving_default_input");
    TF_Operation* output_op = TF_GraphOperationByName(graph, "StatefulPartitionedCall");

    // Run inference
    TF_Output inputs[] = {{input_op, 0}};
    TF_Output outputs[] = {{output_op, 0}};

    // ... prepare tensors and run ...

    TF_CloseSession(session, status);
}

TF_DeleteSession(session, status);
TF_DeleteSessionOptions(opts);
TF_DeleteGraph(graph);
TF_DeleteStatus(status);
```

### Session Config for SavedModel

```c
// Configure session for SavedModel loading
TF_SessionOptions* opts = TF_NewSessionOptions();

// Set GPU options
tensorflow::ConfigProto config;
config.mutable_gpu_options()->set_allow_growth(true);
config.set_allow_soft_placement(true);

std::string config_str;
config.SerializeToString(&config_str);

TF_Status* status = TF_NewStatus();
TF_SetConfig(opts, config_str.data(), config_str.size(), status);
```

---

## Memory Management Patterns

### Allocation Patterns

Every object created by the C API must be explicitly freed:

```c
// Create
TF_Status* status = TF_NewStatus();
TF_Graph* graph = TF_NewGraph();
TF_SessionOptions* opts = TF_NewSessionOptions();
TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, dims, ndims, size);

// Use ...

// Destroy (order matters for dependent objects)
TF_DeleteTensor(tensor);
TF_DeleteSessionOptions(opts);
TF_DeleteGraph(graph);
TF_DeleteStatus(status);
```

### RAII Wrappers (C++)

```c++
class ScopedStatus {
public:
    ScopedStatus() : status_(TF_NewStatus()) {}
    ~ScopedStatus() { TF_DeleteStatus(status_); }
    TF_Status* get() { return status_; }
    bool ok() { return TF_GetCode(status_) == TF_OK; }
    const char* message() { return TF_Message(status_); }
private:
    TF_Status* status_;
};

class ScopedGraph {
public:
    ScopedGraph() : graph_(TF_NewGraph()) {}
    ~ScopedGraph() { TF_DeleteGraph(graph_); }
    TF_Graph* get() { return graph_; }
private:
    TF_Graph* graph_;
};

class ScopedTensor {
public:
    ScopedTensor(TF_Tensor* t) : tensor_(t) {}
    ~ScopedTensor() { TF_DeleteTensor(tensor_); }
    TF_Tensor* get() { return tensor_; }
    operator TF_Tensor*() { return tensor_; }
private:
    TF_Tensor* tensor_;
};
```

### Deallocator Patterns

```c
// No-op deallocator (data is static or externally managed)
TF_Tensor* tensor = TF_NewTensor(
    type, dims, ndims, data, len,
    [](void*, size_t, void*) {}, nullptr);

// Free on delete
TF_Tensor* tensor = TF_NewTensor(
    type, dims, ndims, malloc(len), len,
    [](void* data, size_t, void*) { free(data); }, nullptr);

// Array delete
float* data = new float[count];
TF_Tensor* tensor = TF_NewTensor(
    TF_FLOAT, dims, ndims, data, count * sizeof(float),
    [](void* data, size_t, void*) { delete[] static_cast<float*>(data); },
    nullptr);
```

### Error Handling Best Practices

```c
// Check status after every operation that can fail
TF_Status* status = TF_NewStatus();

TF_Graph* graph = TF_NewGraph();
TF_OperationDescription* desc = TF_NewOperation(graph, "Const", "c");
TF_SetAttrType(desc, "dtype", TF_FLOAT);

TF_Operation* op = TF_FinishOperation(desc, status);
if (TF_GetCode(status) != TF_OK) {
    printf("Error creating operation: %s\n", TF_Message(status));
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return -1;
}

// Always clean up, even on error paths
TF_DeleteGraph(graph);
TF_DeleteStatus(status);
```

---

## Summary

The TensorFlow C API provides complete access to TensorFlow's core
functionality:

1. **TF_Status**: Error reporting mechanism used throughout the API.
2. **TF_Tensor**: Multidimensional arrays for data exchange.
3. **TF_Graph**: Computation graph construction and manipulation.
4. **TF_Operation**: Immutable graph nodes with full attribute access.
5. **TF_Session**: Graph execution with input/output tensor management.
6. **TF_OperationDescription**: Operation building with comprehensive
   attribute setting functions.
7. **Eager C API**: Immediate execution mode with TFE_Context,
   TFE_TensorHandle, and TFE_Op.
8. **SavedModel C API**: Loading and running pre-trained models from disk.
9. **Memory management**: Explicit allocation/deallocation with clear
   ownership patterns.
