# TensorFlow Lite Delegates Reference

## Table of Contents

1. [Delegate System Overview](#delegate-system-overview)
2. [TfLiteDelegate Structure](#tflitedelegate-structure)
3. [Delegate Lifecycle](#delegate-lifecycle)
4. [Delegate Partitioning](#delegate-partitioning)
5. [GPU Delegate](#gpu-delegate)
6. [NNAPI Delegate](#nnapi-delegate)
7. [CoreML Delegate](#coreml-delegate)
8. [Hexagon Delegate](#hexagon-delegate)
9. [XNNPACK Delegate](#xnnpack-delegate)
10. [External Delegate](#external-delegate)
11. [Flex Delegate](#flex-delegate)
12. [Custom Delegate Implementation](#custom-delegate-implementation)
13. [Delegate Registration and Discovery](#delegate-registration-and-discovery)
14. [Performance Comparison](#performance-comparison)
15. [Platform-Specific Recommendations](#platform-specific-recommendations)

---

## Delegate System Overview

### Purpose and Architecture

TensorFlow Lite delegates provide a mechanism to accelerate graph execution by
delegating portions of the model graph to specialized hardware backends. The
delegate system is designed around the following principles:

- **Graph partitioning**: The TFLite runtime partitions the execution graph into
  subgraphs, delegating supported operations to the delegate while falling back
  to the default CPU implementation for unsupported operations.
- **Transparent acceleration**: Applications can add acceleration without
  modifying the model or the inference code.
- **Backend abstraction**: Each delegate encapsulates a specific backend (GPU,
  DSP, NPU, etc.) behind a uniform interface.
- **Fallback guarantee**: If a delegate fails to initialize or a particular
  operation is not supported, the runtime transparently falls back to the
  built-in CPU kernels.

### Core Abstractions

The delegate system consists of the following key components:

1. **TfLiteDelegate**: The opaque delegate structure registered with the
   interpreter.
2. **TfLiteDelegateParams**: Parameters passed to the delegate kernel,
   including the list of nodes to replace, input tensors, and output tensors.
3. **GraphPartitionHelper**: Utility class that partitions the graph into
   supported and unsupported node subsets.
4. **TfLiteRegistration**: The registration structure for delegate kernels,
   containing function pointers for init, prepare, invoke, and profiling.

### High-Level Delegation Flow

The delegation process follows these steps:

1. Application creates a delegate instance with desired options.
2. Application calls `interpreter->ModifyGraphWithDelegate(delegate)`.
3. The TFLite runtime iterates through the execution plan, calling each node's
   `prepare` function.
4. During delegate `Prepare`, the delegate determines which nodes it supports.
5. The runtime partitions the graph into contiguous subsets of supported nodes.
6. Each subset is replaced with a single delegate kernel (a
   `kTfLiteBuiltinDelegate` op).
7. During `Invoke`, delegate kernels execute their respective subgraphs on the
   target backend.

---

## TfLiteDelegate Structure

### Core Structure Definition

The `TfLiteDelegate` structure is defined in `tensorflow/lite/core/c/common.h`
and represents the interface between the TFLite runtime and a delegate backend.

```c
typedef struct TfLiteDelegate {
  // Data used by the delegate. Owned by the delegate.
  void* data_;

  // Invoked by ModifyGraphWithDelegate. This function does the actual
  // delegation.
  // - Prepares the delegate for handling nodes in the graph.
  // - Returns kTfLiteOk on success.
  TfLiteStatus (*Prepare)(TfLiteContext* context,
                          struct TfLiteDelegate* delegate);

  // Copy data from delegate buffer to CPU-accessible buffer.
  // Called when the tensor data is needed on CPU but is currently in
  // the delegate's buffer.
  TfLiteStatus (*CopyFromBufferHandle)(TfLiteContext* context,
                                        struct TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor);

  // Copy data from CPU buffer to delegate buffer.
  // Called to copy input tensor data to the delegate's memory space.
  TfLiteStatus (*CopyToBufferHandle)(TfLiteContext* context,
                                      struct TfLiteDelegate* delegate,
                                      TfLiteBufferHandle buffer_handle,
                                      TfLiteTensor* tensor);

  // Free the delegate's buffer handle.
  void (*FreeBufferHandle)(TfLiteContext* context,
                           struct TfLiteDelegate* delegate,
                           TfLiteBufferHandle* handle);

  // Bitmask flags. See TfLiteDelegateFlags below.
  int64_t flags;

  // The array of delegate preparations.
  // This is populated by Prepare.
  TfLiteDelegateParams* prepared_nodes_;

  // Number of elements in prepared_nodes_.
  int num_prepared_nodes_;
} TfLiteDelegate;
```

### Delegate Flags

```c
// Delegate is responsible for copying data to/from the delegate context.
// If set, the runtime will NOT call CopyFromBufferHandle/CopyToBufferHandle
// for intermediate tensors.
typedef enum TfLiteDelegateFlags {
  kTfLiteDelegateFlagsNone = 0,
  // The delegate can handle buffer handles that are shared with other
  // delegates or the CPU.
  kTfLiteDelegateFlagsAllowDynamicTensors = 1,
  // The delegate requires per-op profiling information.
  kTfLiteDelegateFlagsPerOpProfiling = 2,
  // The delegate wants the runtime to use the delegate's own custom
  // allocation strategy for tensors.
  kTfLiteDelegateFlagsUseCustomAllocator = 4,
} TfLiteDelegateFlags;
```

### Buffer Handles

Delegates that manage their own memory use buffer handles to identify tensors
stored in the delegate's memory space:

```c
typedef int TfLiteBufferHandle;
enum {
  kTfLiteNullBufferHandle = -1,
};
```

When a tensor has a non-null `buffer_handle` and `delegate` is set, the runtime
knows that the tensor's data resides in the delegate's memory. The
`data_is_stale` field on `TfLiteTensor` indicates that the CPU-side copy is
out-of-date and `CopyFromBufferHandle` must be called to sync the data.

---

## Delegate Lifecycle

### Initialization

Each delegate provides a creation function that returns a `TfLiteDelegate*`:

```c
// Example: Creating an XNNPACK delegate
TfLiteXNNPackDelegateOptions options =
    TfLiteXNNPackDelegateOptionsDefault();
options.num_threads = 4;
TfLiteDelegate* delegate = TfLiteXNNPackDelegateCreate(&options);
```

### Graph Modification

The delegate is applied to the interpreter via `ModifyGraphWithDelegate`:

```c++
// C++ API
interpreter->ModifyGraphWithDelegate(delegate);

// The delegate must outlive the interpreter.
```

During graph modification:

1. The runtime calls `delegate->Prepare(context, delegate)`.
2. Inside Prepare, the delegate inspects the graph using `TfLiteContext`
   methods:
   - `GetExecutionPlan` to enumerate all nodes.
   - `GetNodeAndRegistration` to inspect individual nodes.
3. The delegate builds a list of supported nodes.
4. The delegate calls `ReplaceNodeSubsetsWithDelegateKernels` to replace
   contiguous supported-node subsets with delegate kernels.

### Execution

During `interpreter->Invoke()`:

1. The runtime executes the modified execution plan.
2. For delegate kernel nodes, the runtime calls the delegate kernel's `invoke`
   function.
3. The delegate kernel executes all nodes in its subgraph on the target
   backend.
4. If output tensors have delegate buffer handles, the data remains in the
   delegate's memory until needed by a CPU operation.

### Teardown

When the interpreter is destroyed:

1. The runtime frees all delegate-related resources.
2. The application must destroy the delegate using the delegate-specific
   deletion function:

```c
TfLiteXNNPackDelegateDelete(delegate);
```

---

## Delegate Partitioning

### GraphPartitionHelper

The `GraphPartitionHelper` class (defined in
`tensorflow/lite/delegates/utils.h`) provides utilities for partitioning the
execution graph into supported and unsupported subsets:

```c++
class GraphPartitionHelper {
 public:
  GraphPartitionHelper(TfLiteContext* context,
                       IsNodeSupportedFn is_node_supported_fn);

  // Partition the graph into node subsets
  TfLiteStatus Partition(std::set<std::string>* unsupported_nodes_info,
                         int start_node_index = 0,
                         int end_node_index = INT_MAX);

  // Get the first N largest partitions
  std::vector<TfLiteDelegateParams*> GetFirstNLargestPartitions(
      int n = INT_MAX,
      int min_nodes_per_partition = 0) const;

  int num_total_nodes() const;
  int num_supported_nodes() const;
  int num_partitions() const;
};
```

### Partitioning Algorithm

The partitioning algorithm works as follows:

1. **Node classification**: For each node in the execution plan, the delegate
   reports whether it supports that node via `IsNodeSupported`.
2. **Contiguous grouping**: Supported nodes are grouped into contiguous
   subsets. Non-supported nodes break the contiguous chain.
3. **Partition creation**: Each contiguous subset of supported nodes becomes
   a partition, represented by `TfLiteDelegateParams`:
   ```c
   typedef struct TfLiteDelegateParams {
     TfLiteDelegate* delegate;
     TfLiteIntArray* nodes_to_replace;
     TfLiteIntArray* input_tensors;
     TfLiteIntArray* output_tensors;
   } TfLiteDelegateParams;
   ```
4. **Graph rewriting**: Each partition is replaced with a single delegate
   kernel node. The delegate kernel's `invoke` function is responsible for
   executing all original nodes in the partition.

### FP16 Graph Partitioning

The `FP16GraphPartitionHelper` extends `GraphPartitionHelper` to handle models
with FP16 constant tensors:

- Nodes that accept FP16 inputs via DEQUANTIZE operations are handled
  transparently.
- The partitioner remaps FP32 inputs from DEQUANTIZE nodes back to their
  original FP16 tensors when all consumers of the DEQUANTIZE output are in
  the same partition.
- This eliminates unnecessary dequantization when the delegate natively
  supports FP16.

### Partition Selection Strategies

Delegates can control how many partitions to accept:

```c++
// Only accept partitions with at least 5 nodes
auto partitions = partition_helper.GetFirstNLargestPartitions(
    /*n=*/3, /*min_nodes_per_partition=*/5);
```

A partition with too few nodes may not justify the overhead of delegating.
The `max_delegated_partitions` option in most delegate configurations controls
the upper limit.

---

## GPU Delegate

### Overview

The TFLite GPU delegate accelerates inference using the device's GPU. It
supports multiple GPU APIs:

- **OpenCL**: Primary backend on Android devices with OpenCL support.
- **Metal**: Primary backend on iOS devices.
- **Vulkan**: Fallback backend on Android devices without OpenCL.
- **OpenGL**: Legacy backend, less performant than OpenCL.

### API Definition

Defined in `tensorflow/lite/delegates/gpu/delegate.h`:

```c
// Create GPU delegate
TfLiteDelegate* TfLiteGpuDelegateV2Create(
    const TfLiteGpuDelegateOptionsV2* options);

// Destroy GPU delegate
void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate);

// Asynchronous variant (Android only)
#if defined(__ANDROID__)
TfLiteDelegate* TfLiteGpuDelegateV2CreateAsync(
    const TfLiteGpuDelegateOptionsV2* options);
#endif
```

### Delegate Options

Defined in `tensorflow/lite/delegates/gpu/delegate_options.h`:

```c
typedef struct {
  // [OBSOLETE]: to be removed
  int32_t is_precision_loss_allowed;

  // Inference usage preference
  int32_t inference_preference;
  // Values:
  //   TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = 0
  //   TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED = 1
  //   TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED = 2

  // Ordered priorities (lower index = higher priority)
  int32_t inference_priority1;
  int32_t inference_priority2;
  int32_t inference_priority3;
  // Values:
  //   TFLITE_GPU_INFERENCE_PRIORITY_AUTO = 0
  //   TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION = 1
  //   TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY = 2
  //   TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE = 3

  // Experimental flags bitmask
  int64_t experimental_flags;
  //   TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE = 0
  //   TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT = 1
  //   TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY = 2
  //   TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY = 4
  //   TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION = 8

  // Maximum number of delegated partitions (default: 1)
  int32_t max_delegated_partitions;

  // Serialization directory for caching compiled GPU programs
  const char* serialization_dir;

  // Unique model token for serialization namespace
  const char* model_token;
} TfLiteGpuDelegateOptionsV2;
```

### Precision Modes

The GPU delegate supports multiple precision configurations:

| Priority Setting | Behavior |
|---|---|
| MAX_PRECISION at priority1 | No FP16 allowed, all FP32 |
| MIN_LATENCY at priority1 | FP16 accumulation and storage where possible |
| MIN_MEMORY_USAGE at priority1 | FP16 storage, may reduce memory usage by 50% |

### Shader Generation

The GPU delegate generates GPU shaders (OpenCL kernels or Metal shaders) at
initialization time. The process involves:

1. **Graph analysis**: Traversing the delegated subgraph to identify
   operations and their data types.
2. **Fusion**: Combining sequential operations into single shader programs
   where possible (e.g., Conv2D + BiasAdd + ReLU becomes one kernel).
3. **Shader compilation**: Compiling generated shaders using the GPU driver.
4. **Buffer allocation**: Allocating GPU buffers for intermediate tensors.

### Supported Operations

The GPU delegate supports a wide range of operations:

- **Convolution**: CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV
- **Activation**: RELU, RELU6, TANH, SIGMOID, HARD_SWISH
- **Element-wise**: ADD, SUB, MUL, DIV, MAX, MIN, POW, SQUARED_DIFFERENCE
- **Pooling**: AVERAGE_POOL_2D, MAX_POOL_2D
- **Resize**: RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR
- **Normalization**: L2_NORMALIZATION, SOFTMAX, LOG_SOFTMAX
- **Fully connected**: FULLY_CONNECTED
- **Reshape**: RESHAPE, CONCATENATION, SPLIT, TRANSPOSE, PAD
- **Other**: GATHER, BATCH_TO_SPACE_ND, SPACE_TO_BATCH_ND, SLICE, STRIDED_SLICE

### Usage on Android

```c++
// C++ example
#include "tensorflow/lite/delegates/gpu/delegate.h"

// Create options
TfLiteGpuDelegateOptionsV2 options =
    TfLiteGpuDelegateOptionsV2Default();
options.inference_preference =
    TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
options.inference_priority1 =
    TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;

// Create delegate
TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&options);

// Apply to interpreter
interpreter->ModifyGraphWithDelegate(gpu_delegate);

// Run inference
interpreter->Invoke();

// Cleanup
TfLiteGpuDelegateV2Delete(gpu_delegate);
```

### Usage on iOS (Metal)

```c++
// C++ example for iOS
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"

TfLiteGpuDelegateOptionsV2 options =
    TfLiteGpuDelegateOptionsV2Default();
options.inference_preference =
    TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;

TfLiteDelegate* metal_delegate = TfLiteGpuDelegateV2Create(&options);
interpreter->ModifyGraphWithDelegate(metal_delegate);

// ... use interpreter ...

TfLiteGpuDelegateV2Delete(metal_delegate);
```

### Serialization

The GPU delegate supports serialization of compiled GPU programs to disk,
reducing initialization time on subsequent runs:

```c++
TfLiteGpuDelegateOptionsV2 options =
    TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags |=
    TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
options.serialization_dir = "/data/data/com.example/cache/tflite_gpu";
options.model_token = "unique_model_identifier";
```

Serialization requires:
- A writable directory for cached programs.
- A unique model token that changes when the model changes.
- OpenCL backend (serialization is not supported with OpenGL).

---

## NNAPI Delegate

### Overview

The NNAPI (Neural Networks API) delegate targets Android devices, providing
access to hardware accelerators including:

- **Android Neural Networks API (NNAPI)**: Available on Android 8.1+ (API
  level 27+), with expanded features in subsequent versions.
- **Neural Processing Units (NPUs)**: Vendor-specific AI accelerators.
- **GPU and DSP backends**: Exposed through NNAPI on supported devices.

### API Definition

Defined in `tensorflow/lite/delegates/nnapi/nnapi_delegate.h`:

```c++
namespace tflite {

class StatefulNnApiDelegate : public TfLiteDelegate {
 public:
  struct Options {
    // Execution preference
    enum ExecutionPreference {
      kUndefined = -1,
      kLowPower = 0,
      kFastSingleAnswer = 1,
      kSustainedSpeed = 2,
    };

    ExecutionPreference execution_preference = kUndefined;
    const char* accelerator_name = nullptr;
    const char* cache_dir = nullptr;
    const char* model_token = nullptr;
    bool disallow_nnapi_cpu = true;
    int max_number_delegated_partitions = 3;
    bool allow_fp16 = false;
    int execution_priority = ANEURALNETWORKS_PRIORITY_DEFAULT;
    uint64_t max_compilation_timeout_duration_ns = 0;
    uint64_t max_execution_timeout_duration_ns = 0;
    uint64_t max_execution_loop_timeout_duration_ns = 0;
    bool allow_dynamic_dimensions = false;
    bool use_burst_computation = false;
    uint32_t max_execution_cache_size = 4;
    std::map<int, size_t> tensor_max_size_hints;
    const char* vendor_compilation_hints = nullptr;
    const char* vendor_execution_hints = nullptr;
    NnapiDelegateVendorPlugin* vendor_plugin = nullptr;
    bool disable_debugging_diagnostics_callbacks = false;
  };

  StatefulNnApiDelegate();
  explicit StatefulNnApiDelegate(const NnApi* nnapi);
  explicit StatefulNnApiDelegate(Options options);
  StatefulNnApiDelegate(const NnApi* nnapi, Options options);

  static const Options GetOptions(TfLiteDelegate* delegate);
};

}  // namespace tflite
```

### NNAPI Feature Levels

| Feature Level | Android Version | API Level | Key Features |
|---|---|---|---|
| NNAPI 1.0 | Android 8.1 | 27 | Basic ops: conv, FC, activation, pooling |
| NNAPI 1.1 | Android 9 | 28 | PAD, Squeeze, Transpose, StridedSlice |
| NNAPI 1.2 | Android 10 | 29 | 40+ new ops, quantized ops, execution bounds |
| NNAPI 1.3 | Android 11 | 30 | DEPTH_TO_SPACE, control flow, dynamic shapes |
| NNAPI 1.4+ | Android 12+ | 31+ | Vendor extensions, signature APIs |

### Supported Operations by NNAPI Version

**NNAPI 1.0 (Android 8.1)**:
- CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED
- MAX_POOL_2D, AVERAGE_POOL_2D, L2_POOL_2D
- RELU, RELU1, RELU6, TANH, SIGMOID
- SOFTMAX, CONCATENATION, RESHAPE
- ADD, MUL, RESIZE_BILINEAR

**NNAPI 1.1 (Android 9)**:
- All NNAPI 1.0 ops plus:
- PAD, PAD_V2, SQUEEZE, TRANSPOSE, STRIDED_SLICE
- BATCH_TO_SPACE_ND, SPACE_TO_BATCH_ND
- DEQUANTIZE, QUANTIZE

**NNAPI 1.2 (Android 10)**:
- All NNAPI 1.1 ops plus:
- ABS, EXP, FLOOR, LOG, SIN, SQRT, RSQRT
- ARGMAX, ARGMIN, EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL
- CAST, EXPAND_DIMS, GATHER, GENERATE_PROPOSALS
- HEATMAP_MAX_KEYPOINT, INSTANCE_NORMALIZATION
- L2_NORMALIZATION, LOCAL_RESPONSE_NORMALIZATION
- LOGICAL_AND, LOGICAL_NOT, LOGICAL_OR, LOG_SOFTMAX
- MAXIMUM, MINIMUM, NEG, NOT, PAD_V2
- POW, PRELU, REDUCE_ALL, REDUCE_ANY, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM
- RESIZE_NEAREST_NEIGHBOR, ROI_ALIGN, ROI_POOLING
- SELECT, SLICE, SPLIT, TILE, TOPK_V2
- TRANSPOSE_CONV, UNIDIRECTIONAL_SEQUENCE_LSTM
- UNIDIRECTIONAL_SEQUENCE_RNN, BIDIRECTIONAL_SEQUENCE_RNN

**NNAPI 1.3 (Android 11)**:
- All NNAPI 1.2 ops plus:
- BIDIRECTIONAL_SEQUENCE_LSTM, CACHE ops
- CONTROL FLOW (IF, WHILE), DEPTH_TO_SPACE
- HARD_SWISH, MEAN, RESHAPE (with optional shape input)
- BATCH_MATMUL, DENSIFY

### Execution Preferences

```c++
StatefulNnApiDelegate::Options options;
options.execution_preference =
    StatefulNnApiDelegate::Options::kSustainedSpeed;
// kLowPower: Use low-power accelerator if available
// kFastSingleAnswer: Optimize for single inference
// kSustainedSpeed: Optimize for throughput
```

### Compilation Caching

NNAPI supports compilation caching to avoid recompiling the model on each run:

```c++
StatefulNnApiDelegate::Options options;
options.cache_dir = "/data/data/com.example/cache/nnapi";
options.model_token = "model_v1_unique_token";
```

### Accelerator Selection

```c++
StatefulNnApiDelegate::Options options;
// Use a specific accelerator (e.g., "google-edgetpu")
options.accelerator_name = "google-edgetpu";
// Set to nullptr to use all available accelerators (default)
```

### Burst Mode

Burst computation reduces overhead for repeated inferences:

```c++
StatefulNnApiDelegate::Options options;
options.use_burst_computation = true;
```

---

## CoreML Delegate

### Overview

The CoreML delegate targets Apple devices, leveraging the Core ML framework for
hardware-accelerated inference. Core ML uses the Apple Neural Engine (ANE) on
supported devices, falling back to GPU or CPU as needed.

### API Definition

Defined in `tensorflow/lite/delegates/coreml/coreml_delegate.h`:

```c
typedef enum {
  TfLiteCoreMlDelegateDevicesWithNeuralEngine,
  TfLiteCoreMlDelegateAllDevices,
} TfLiteCoreMlDelegateEnabledDevices;

typedef struct {
  TfLiteCoreMlDelegateEnabledDevices enabled_devices;
  int coreml_version;
  int max_delegated_partitions;
  int min_nodes_per_partition;
} TfLiteCoreMlDelegateOptions;

TfLiteDelegate* TfLiteCoreMlDelegateCreate(
    const TfLiteCoreMlDelegateOptions* options);
void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate);
```

### Delegate Options

| Option | Default | Description |
|---|---|---|
| `enabled_devices` | `TfLiteCoreMlDelegateDevicesWithNeuralEngine` | Restrict to ANE-equipped devices |
| `coreml_version` | Auto | Target Core ML version (2 or 3) |
| `max_delegated_partitions` | 0 (all) | Maximum partitions to delegate |
| `min_nodes_per_partition` | 2 | Minimum nodes per partition |

### Supported Operations

**Core ML 2**:
- CONV_2D, DEPTHWISE_CONV_2D
- AVERAGE_POOL_2D, MAX_POOL_2D
- ADD, MUL, RELU, RELU6, SIGMOID, TANH
- FULLY_CONNECTED, SOFTMAX, CONCATENATION
- RESHAPE, TRANSPOSE

**Core ML 3** (significantly expanded):
- All Core ML 2 ops plus:
- TRANSPOSE_CONV, BATCH_TO_SPACE_ND, SPACE_TO_BATCH_ND
- GATHER, PAD, PADV2, SLICE, STRIDED_SLICE
- EXP, LOG, SQRT, RSQRT, POW, ABS
- REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM, MEAN
- EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL
- CAST, SELECT, RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR
- HARD_SWISH, PRELU, LEAKY_RELU
- LOG_SOFTMAX, L2_NORMALIZATION
- TOPK_V2, SPLIT, SQUEEZE, EXPAND_DIMS

### Device Requirements

- **iOS**: 12.0+ for Core ML 2, 13.0+ for Core ML 3
- **macOS**: 10.14+ for Core ML 2, 10.15+ for Core ML 3
- **Apple Neural Engine**: Available on A11 Bionic and later, M1 and later

### Usage Example

```c++
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

TfLiteCoreMlDelegateOptions options = {};
options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
options.coreml_version = 3;
options.max_delegated_partitions = 0;  // delegate all

TfLiteDelegate* coreml_delegate =
    TfLiteCoreMlDelegateCreate(&options);

if (coreml_delegate) {
  interpreter->ModifyGraphWithDelegate(coreml_delegate);
}

// ... run inference ...

TfLiteCoreMlDelegateDelete(coreml_delegate);
```

---

## Hexagon Delegate

### Overview

The Hexagon delegate targets Qualcomm Snapdragon processors, utilizing the
Hexagon DSP (Digital Signal Processor) and Qualcomm Hexagon NN library for
accelerated inference.

### API Definition

Defined in `tensorflow/lite/delegates/hexagon/hexagon_delegate.h`:

```c
struct TfLiteHexagonDelegateOptions {
  int debug_level;                   // 0 = no debug
  int powersave_level;               // 0 = high performance
  bool print_graph_profile;
  bool print_graph_debug;
  int max_delegated_partitions;
  int min_nodes_per_partition;       // default: 2
  bool enable_dynamic_batch_size;
  int max_batch_size;
  TfLiteIntArray* input_batch_dimensions;
  TfLiteIntArray* output_batch_dimensions;
};

TfLiteDelegate* TfLiteHexagonDelegateCreate(
    const TfLiteHexagonDelegateOptions* options);
TfLiteHexagonDelegateOptions TfLiteHexagonDelegateOptionsDefault();
void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);
void TfLiteHexagonInitWithPath(const char* lib_directory_path);
void TfLiteHexagonInit();
void TfLiteHexagonTearDown();
```

### Initialization Requirements

Before using the Hexagon delegate, the Hexagon NN libraries must be loaded:

```c++
// Initialize with path to Hexagon NN shared libraries
TfLiteHexagonInitWithPath("/path/to/hexagon/libs");
// Or use default initialization if libraries are in system path
TfLiteHexagonInit();
```

The following shared libraries must be present:
- `libhexagon_nn.so`
- `libhexagon_interface.so`
- Other Hexagon NN implementation libraries

### Supported Operations

The Hexagon delegate supports a subset of quantized operations:

- CONV_2D (quantized)
- DEPTHWISE_CONV_2D (quantized)
- FULLY_CONNECTED (quantized)
- MAX_POOL_2D, AVERAGE_POOL_2D (quantized)
- RELU, RELU6, TANH, SIGMOID
- ADD, MUL, SUB
- CONCATENATION, RESHAPE
- SOFTMAX, LOGISTIC
- L2_NORMALIZATION, L2_POOL_2D

### Power Save Levels

| Level | Description |
|---|---|
| 0 | High performance, higher power consumption |
| 1 | Moderate performance |
| 2 | Low power mode |
| 3 | Minimal power, lowest performance |

### Usage Example

```c++
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

// Initialize Hexagon
TfLiteHexagonInit();

TfLiteHexagonDelegateOptions options =
    TfLiteHexagonDelegateOptionsDefault();
options.debug_level = 0;
options.powersave_level = 0;
options.print_graph_profile = true;

TfLiteDelegate* hexagon_delegate =
    TfLiteHexagonDelegateCreate(&options);

interpreter->ModifyGraphWithDelegate(hexagon_delegate);

// ... run inference ...

TfLiteHexagonDelegateDelete(hexagon_delegate);
TfLiteHexagonTearDown();
```

### Device Compatibility

- Qualcomm Snapdragon processors with Hexagon DSP
- Snapdragon 820 and later
- Requires quantized models (INT8) for optimal performance
- Works best with models designed for quantized inference

---

## XNNPACK Delegate

### Overview

The XNNPACK delegate provides highly optimized CPU-based inference using the
XNNPACK library. It offers significant performance improvements over the
default TFLite CPU kernels, especially on ARM and x86 processors.

XNNPACK is the recommended delegate for CPU inference on all platforms because:
- It is the default delegate applied automatically in TFLite.
- It supports FP32, FP16, and quantized (INT8, UINT8) inference.
- It provides optimized implementations for ARM (NEON), x86 (SSE/AVX), and
  WebAssembly.
- It has no hardware dependencies beyond a capable CPU.

### API Definition

Defined in `tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h`:

```c
typedef struct {
  int32_t num_threads;
  uint32_t runtime_flags;
  uint32_t flags;
  struct TfLiteXNNPackDelegateWeightsCache* weights_cache;
  bool handle_variable_ops;  // deprecated
  const char* weight_cache_file_path;
  int weight_cache_file_descriptor;
  void* weight_cache_provider;
  bool weight_cache_lock_memory;
} TfLiteXNNPackDelegateOptions;

TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault();
TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options);
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);
```

### Threading Model

XNNPACK uses a thread pool for parallel inference:

```c++
TfLiteXNNPackDelegateOptions options =
    TfLiteXNNPackDelegateOptionsDefault();
options.num_threads = 4;  // Use 4 threads
// 0 or negative: single-threaded (no thread pool)
```

The threading model distributes work within individual operations (intra-op
parallelism). For example, a large matrix multiplication may be split across
multiple threads.

### Flags

```c
// Enable INT8 quantized inference
#define TFLITE_XNNPACK_DELEGATE_FLAG_QS8 0x00000001

// Enable UINT8 quantized inference
#define TFLITE_XNNPACK_DELEGATE_FLAG_QU8 0x00000002

// Force FP16 inference for FP32 operators
#define TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 0x00000004

// Enable dynamic weights for FULLY_CONNECTED
#define TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED 0x00000008

// Enable variable operators (VAR_HANDLE, READ_VARIABLE, ASSIGN_VARIABLE)
#define TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS 0x00000010

// Enable transient indirection buffer to reduce memory
#define TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER 0x00000020

// Enable latest XNNPACK operators and features
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS 0x00000040

// Enable subgraph reshaping for dynamic tensors
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING 0x00000080

// Consistent arithmetic across codepaths
#define TFLITE_XNNPACK_DELEGATE_FLAG_SLOW_CONSISTENT_ARITHMETIC 0x00000200

// Disable subgraph reshaping
#define TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_SUBGRAPH_RESHAPING 0x00000400

// Disable dynamically quantized ops
#define TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_DYNAMICALLY_QUANTIZED_OPS \
  0x00000800
```

### Weight Cache

XNNPACK supports weight caching to reduce initialization overhead when the
same model is loaded multiple times:

```c++
// Create a shared weights cache
TfLiteXNNPackDelegateWeightsCache* cache =
    TfLiteXNNPackDelegateWeightsCacheCreate();

// Use cache across multiple delegates
TfLiteXNNPackDelegateOptions options1 =
    TfLiteXNNPackDelegateOptionsDefault();
options1.weights_cache = cache;
TfLiteDelegate* delegate1 = TfLiteXNNPackDelegateCreate(&options1);

TfLiteXNNPackDelegateOptions options2 =
    TfLiteXNNPackDelegateOptionsDefault();
options2.weights_cache = cache;
TfLiteDelegate* delegate2 = TfLiteXNNPackDelegateCreate(&options2);

// ... use delegates ...

// Finalize cache when no more delegates will be created
TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(cache);
// or hard finalize for minimum memory:
// TfLiteXNNPackDelegateWeightsCacheFinalizeHard(cache);

// Cleanup
TfLiteXNNPackDelegateDelete(delegate1);
TfLiteXNNPackDelegateDelete(delegate2);
TfLiteXNNPackDelegateWeightsCacheDelete(cache);
```

### Supported Operations

XNNPACK supports a comprehensive set of operations:

**Convolution**:
- CONV_2D (FP32, QS8, QU8)
- DEPTHWISE_CONV_2D (FP32, QS8, QU8)
- TRANSPOSE_CONV (FP32)

**Fully Connected**:
- FULLY_CONNECTED (FP32, QS8, QU8, dynamic weights)

**Activation**:
- RELU, RELU6, RELU_N1_TO_1, LEAKY_RELU
- TANH, SIGMOID, HARD_SWISH, ELU
- LOGISTIC, LOG_SOFTMAX, SOFTMAX

**Element-wise**:
- ADD, SUB, MUL, DIV, MAX, MIN, POW
- SQUARED_DIFF, NEG, ABS, SQUARE
- EXP, LOG, SQRT, RSQRT, SIN, COS
- FLOOR, CEIL, ROUND
- AND, OR, NOT, XOR
- EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL

**Reduction**:
- MEAN, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD, REDUCE_SUM
- REDUCE_ANY, REDUCE_ALL

**Array**:
- RESHAPE, CONCATENATION, SPLIT, SPLIT_V
- TRANSPOSE, PAD, PADV2, GATHER
- EXPAND_DIMS, SQUEEZE, PACK, UNPACK
- STRIDED_SLICE, SLICE, TILE
- SELECT, SELECT_V2, WHERE, BROADCAST_TO

**Pooling**:
- AVERAGE_POOL_2D, MAX_POOL_2D

**Other**:
- CAST, QUANTIZE, DEQUANTIZE
- RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR
- DEPTH_TO_SPACE, SPACE_TO_DEPTH
- BATCH_TO_SPACE_ND, SPACE_TO_BATCH_ND
- PRELU, L2_NORMALIZATION
- TOPK_V2, ARG_MAX, ARG_MIN
- FILL, ZEROS_LIKE, RANGE
- SCATTER_ND, GATHER_ND

### Usage Example

```c++
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

TfLiteXNNPackDelegateOptions options =
    TfLiteXNNPackDelegateOptionsDefault();
options.num_threads = 4;
options.flags = TFLITE_XNNPACK_DELEGATE_FLAG_QS8 |
                TFLITE_XNNPACK_DELEGATE_FLAG_QU8;

TfLiteDelegate* xnnpack_delegate =
    TfLiteXNNPackDelegateCreate(&options);

interpreter->ModifyGraphWithDelegate(xnnpack_delegate);

// ... run inference ...

TfLiteXNNPackDelegateDelete(xnnpack_delegate);
```

---

## External Delegate

### Overview

The external delegate mechanism enables loading delegate implementations from
shared libraries at runtime. This is useful for:

- Vendor-specific delegates distributed separately from TFLite.
- Delegates that cannot be linked at compile time.
- Runtime-selectable acceleration backends.

### API Definition

Defined in `tensorflow/lite/delegates/external/external_delegate.h`:

```c
#define kExternalDelegateMaxOptions 256

typedef struct TfLiteExternalDelegateOptions {
  const char* lib_path;
  int count;
  const char* keys[kExternalDelegateMaxOptions];
  const char* values[kExternalDelegateMaxOptions];
  TfLiteStatus (*insert)(struct TfLiteExternalDelegateOptions* options,
                         const char* key, const char* value);
} TfLiteExternalDelegateOptions;

TfLiteStatus TfLiteExternalDelegateOptionsInsert(
    TfLiteExternalDelegateOptions* options,
    const char* key, const char* value);

TfLiteExternalDelegateOptions TfLiteExternalDelegateOptionsDefault(
    const char* lib_path);

TfLiteDelegate* TfLiteExternalDelegateCreate(
    const TfLiteExternalDelegateOptions* options);

void TfLiteExternalDelegateDelete(TfLiteDelegate* delegate);
```

### Delegate Plugin Interface

External delegates must implement the `tflite_plugin_create_delegate` and
`tflite_plugin_destroy_delegate` functions exported from the shared library:

```c
// Required exports from the delegate shared library
TfLiteDelegate* tflite_plugin_create_delegate(
    const char* const* options_keys,
    const char* const* options_values,
    size_t num_options,
    void (*report_error)(const char*));

void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
```

### Usage Example

```c++
#include "tensorflow/lite/delegates/external/external_delegate.h"

TfLiteExternalDelegateOptions options =
    TfLiteExternalDelegateOptionsDefault(
        "/path/to/libmy_delegate.so");

// Add configuration options
TfLiteExternalDelegateOptionsInsert(&options, "key1", "value1");
TfLiteExternalDelegateOptionsInsert(&options, "key2", "value2");

TfLiteDelegate* external_delegate =
    TfLiteExternalDelegateCreate(&options);

if (external_delegate) {
  interpreter->ModifyGraphWithDelegate(external_delegate);
}

// ... run inference ...

TfLiteExternalDelegateDelete(external_delegate);
```

### Dynamic Loading Process

When an external delegate is created:

1. TFLite dynamically loads the shared library at `lib_path` using
   platform-specific mechanisms (`dlopen` on Linux/Android).
2. It resolves the `tflite_plugin_create_delegate` symbol.
3. It calls `tflite_plugin_create_delegate` with the provided options.
4. The returned delegate is used with the interpreter.
5. On destruction, `tflite_plugin_destroy_delegate` is called and the
   library is unloaded.

---

## Flex Delegate

### Overview

The Flex delegate enables running TensorFlow operations that are not natively
supported in TFLite by executing them through the TensorFlow (TF) eager
runtime. This allows models converted from TF to TFLite to retain access to
the full set of TF operations.

### When to Use

The Flex delegate is useful when:

- A model uses TF ops that do not have TFLite equivalents.
- Custom TF ops need to be executed within a TFLite model.
- Rapid prototyping requires running unconverted ops before creating native
  TFLite implementations.

### API Definition

Defined in `tensorflow/lite/delegates/flex/delegate.h`:

```c++
namespace tflite {

class FlexDelegate : public SimpleDelegateInterface {
 public:
  static TfLiteDelegateUniquePtr Create();
  static TfLiteDelegateUniquePtr Create(
      std::unique_ptr<FlexDelegate> base_delegate);

  void Cancel();
  static bool HasCancelled(void* data);

 protected:
  const char* Name() const override;
  bool IsNodeSupportedByDelegate(...) const override;
  TfLiteStatus Initialize(TfLiteContext* context) override;
  std::unique_ptr<SimpleDelegateKernelInterface>
      CreateDelegateKernelInterface() override;
};

}  // namespace tflite
```

### Usage Example

```c++
#include "tensorflow/lite/delegates/flex/delegate.h"

auto flex_delegate = tflite::FlexDelegate::Create();
if (flex_delegate) {
  interpreter->ModifyGraphWithDelegate(flex_delegate.get());
}

// ... run inference ...

// Flex delegate is automatically cleaned up when unique_ptr is destroyed
```

### Select TF Ops in Model Conversion

When converting a model with the TFLite converter, you can enable Flex ops:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model_dir")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,     # TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS,       # Flex ops
]
tflite_model = converter.convert()
```

### Limitations

- Increases binary size significantly (requires TF runtime).
- Higher memory usage compared to pure TFLite models.
- Not supported on TFLite Micro.
- Performance may be lower than native TFLite ops due to TF eager runtime
  overhead.
- Some TF ops may not be compatible with Flex delegation.

---

## Custom Delegate Implementation

### Overview

Creating a custom delegate involves implementing the `TfLiteDelegate`
interface and the associated delegate kernel. The TFLite framework provides
utility classes to simplify this process.

### Using SimpleDelegateInterface

The `SimpleDelegateInterface` class (in
`tensorflow/lite/delegates/utils/simple_delegate.h`) provides a simplified
approach:

```c++
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

class MyDelegate : public tflite::SimpleDelegateInterface {
 public:
  explicit MyDelegate(const MyDelegateOptions& options)
      : options_(options) {}

  const char* Name() const override {
    return "MyDelegate";
  }

  bool IsNodeSupportedByDelegate(
      const TfLiteRegistration* registration,
      const TfLiteNode* node,
      TfLiteContext* context) const override {
    // Return true if this node is supported by the delegate
    return registration->builtin_code == kTfLiteBuiltinAdd ||
           registration->builtin_code == kTfLiteBuiltinMul;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    // Initialize delegate resources
    return kTfLiteOk;
  }

  Options DelegateOptions() const override {
    Options opts;
    opts.max_delegated_partitions = 0;
    opts.min_nodes_per_partition = 2;
    return opts;
  }

  std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<MyDelegateKernel>();
  }

 private:
  MyDelegateOptions options_;
};
```

### Implementing the Delegate Kernel

```c++
class MyDelegateKernel : public tflite::SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    // Initialize the kernel with the delegated subgraph.
    // 'params' contains:
    //   - params->nodes_to_replace: list of node indices
    //   - params->input_tensors: input tensor indices
    //   - params->output_tensors: output tensor indices
    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteContext* context,
                       TfLiteNode* node) override {
    // Called when input shapes change.
    // Resize output tensors as needed.
    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context,
                    TfLiteNode* node) override {
    // Execute the delegated subgraph.
    // Read input tensors, compute, write output tensors.
    return kTfLiteOk;
  }
};
```

### Using the Custom Delegate

```c++
// Create the delegate interface
auto my_delegate = std::make_unique<MyDelegate>(options);

// Create the TfLiteDelegate wrapper
auto tflite_delegate =
    tflite::TfLiteDelegateFactory::Create(std::move(my_delegate));

// Apply to interpreter
interpreter->ModifyGraphWithDelegate(tflite_delegate.get());
```

### Low-Level Delegate Implementation

For more control, implement `TfLiteDelegate` directly:

```c++
TfLiteDelegate CreateMyDelegate() {
  TfLiteDelegate delegate = {};
  delegate.data_ = new MyDelegateData();
  delegate.Prepare = MyDelegatePrepare;
  delegate.CopyFromBufferHandle = MyCopyFromBuffer;
  delegate.CopyToBufferHandle = MyCopyToBuffer;
  delegate.FreeBufferHandle = MyFreeBufferHandle;
  delegate.flags = kTfLiteDelegateFlagsNone;
  return delegate;
}

TfLiteStatus MyDelegatePrepare(TfLiteContext* context,
                               TfLiteDelegate* delegate) {
  MyDelegateData* data = static_cast<MyDelegateData*>(delegate->data_);

  // Get execution plan
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));

  // Determine supported nodes
  std::vector<int> supported_nodes;
  for (int i = 0; i < plan->size; i++) {
    int node_index = plan->data[i];
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, node_index, &node, &reg);
    if (IsSupported(reg)) {
      supported_nodes.push_back(node_index);
    }
  }

  // Replace supported nodes with delegate kernel
  TfLiteRegistration delegate_kernel_reg = {};
  delegate_kernel_reg.init = MyKernelInit;
  delegate_kernel_reg.free = MyKernelFree;
  delegate_kernel_reg.prepare = MyKernelPrepare;
  delegate_kernel_reg.invoke = MyKernelInvoke;

  TfLiteIntArray* nodes =
      TfLiteIntArrayCreate(supported_nodes.size());
  for (size_t i = 0; i < supported_nodes.size(); i++) {
    nodes->data[i] = supported_nodes[i];
  }

  context->ReplaceNodeSubsetsWithDelegateKernels(
      context, delegate_kernel_reg, nodes, delegate);

  TfLiteIntArrayFree(nodes);
  return kTfLiteOk;
}
```

---

## Delegate Registration and Discovery

### OpResolver Integration

Delegates interact with the TFLite `OpResolver` to register custom delegate
kernels:

```c++
class MutableOpResolver : public OpResolver {
  // Register a custom delegate kernel
  void AddCustom(const char* name, TfLiteRegistration* registration);
};
```

### Multi-Delegate Support

Multiple delegates can be applied to the same interpreter:

```c++
// Apply XNNPACK first (catches most CPU-optimizable ops)
interpreter->ModifyGraphWithDelegate(xnnpack_delegate);

// Then apply GPU delegate for remaining GPU-friendly ops
interpreter->ModifyGraphWithDelegate(gpu_delegate);
```

Order matters: the first delegate applied gets the first chance to claim
nodes. Once a node is claimed by a delegate, it is replaced with a delegate
kernel that cannot be further delegated.

### Delegate Compatibility

| Delegate | Thread Safe | Multiple Interpreters | Dynamic Shapes |
|---|---|---|---|
| XNNPACK | No (per delegate) | With shared cache | Limited |
| GPU | No | No | No |
| NNAPI | No | Limited | Yes (1.3+) |
| CoreML | No | No | Limited |
| Hexagon | No | No | Limited |
| Flex | No | Yes | Yes |

---

## Performance Comparison

### Benchmark Methodology

Performance varies significantly based on:
- Model architecture (CNN, RNN, transformer)
- Input sizes
- Quantization type (FP32, FP16, INT8)
- Device hardware (GPU model, NPU capabilities)

### Relative Performance (Typical Mobile Device)

| Delegate | FP32 Latency | INT8 Latency | Memory | Init Time |
|---|---|---|---|---|
| CPU (default) | 1.0x (baseline) | 0.6x | 1.0x | Fast |
| XNNPACK | 0.4-0.6x | 0.2-0.4x | 1.1x | Medium |
| GPU (OpenCL) | 0.2-0.5x | 0.1-0.3x | 1.5-2x | Slow |
| GPU (Metal) | 0.2-0.4x | 0.1-0.2x | 1.5-2x | Slow |
| NNAPI (NPU) | 0.1-0.3x | 0.05-0.2x | 0.5-1x | Medium-Slow |
| CoreML (ANE) | 0.1-0.3x | 0.05-0.2x | 0.5-1x | Medium |
| Hexagon (DSP) | N/A (quantized only) | 0.1-0.3x | 0.5-1x | Medium |

### CNN Model Benchmarks (MobileNet V2, 224x224)

| Delegate | Latency (ms) | Notes |
|---|---|---|
| CPU only | 30-50 | Single-threaded |
| XNNPACK (4 threads) | 10-20 | Best CPU option |
| GPU (OpenCL) | 3-8 | Device-dependent |
| GPU (Metal) | 2-6 | iPhone 12+ |
| NNAPI (NPU) | 1-4 | Snapdragon 888+ |

### Considerations

- **Initialization overhead**: GPU delegates have higher initialization times
  due to shader compilation. Use serialization for frequently-loaded models.
- **Power consumption**: GPU and NPU delegates typically consume less power
  per inference than CPU-only execution.
- **Accuracy**: FP16 and quantized inference may have slight accuracy
  differences. Always validate model accuracy after enabling a delegate.

---

## Platform-Specific Recommendations

### Android

| Scenario | Recommended Delegate | Rationale |
|---|---|---|
| General CPU | XNNPACK | Best CPU performance, broad op support |
| GPU acceleration | GPU (OpenCL) | Fastest for vision models on GPU |
| NPU acceleration | NNAPI | Access to vendor NPUs |
| Qualcomm DSP | Hexagon | Direct Hexagon DSP access |
| Low-latency inference | GPU or NNAPI | Hardware acceleration |
| Battery-sensitive | NNAPI (NPU) | Lowest power per inference |

### iOS

| Scenario | Recommended Delegate | Rationale |
|---|---|---|
| General CPU | XNNPACK | Best CPU performance |
| GPU acceleration | GPU (Metal) | Native Metal performance |
| Neural Engine | CoreML | Direct ANE access |
| Maximum performance | CoreML + GPU | Try CoreML first, fallback to Metal |

### Desktop / Server

| Scenario | Recommended Delegate | Rationale |
|---|---|---|
| CPU inference | XNNPACK | Optimized SIMD implementations |
| GPU inference | Use full TensorFlow | Not TFLite (use TF with GPU) |
| Multi-model serving | XNNPACK with shared cache | Reduced initialization overhead |

### Embedded / IoT

| Scenario | Recommended Delegate | Rationale |
|---|---|---|
| ARM Cortex-M | None (TFLite Micro) | TFLite delegates not supported on Micro |
| ARM Cortex-A | XNNPACK | Best CPU performance for ARM |
| Edge TPU | Custom Edge TPU delegate | Google Coral specific |
| RISC-V | XNNPACK (with RISC-V support) | Emerging platform support |

### Cross-Platform Strategy

For maximum compatibility across platforms:

```c++
// Platform-adaptive delegate selection
TfLiteDelegate* delegate = nullptr;

#if defined(__ANDROID__)
  // Try NNAPI first for NPU access
  tflite::StatefulNnApiDelegate::Options nnapi_opts;
  nnapi_opts.execution_preference =
      tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
  static tflite::StatefulNnApiDelegate nnapi_delegate(nnapi_opts);
  delegate = &nnapi_delegate;

  // Fall back to GPU if NNAPI is not available
  if (!delegate) {
    static auto gpu_options = TfLiteGpuDelegateOptionsV2Default();
    static TfLiteDelegate* gpu_delegate =
        TfLiteGpuDelegateV2Create(&gpu_options);
    delegate = gpu_delegate;
  }
#elif defined(__APPLE__)
  // Use CoreML on Apple devices
  static TfLiteCoreMlDelegateOptions options = {};
  options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
  static TfLiteDelegate* coreml_delegate =
      TfLiteCoreMlDelegateCreate(&options);
  delegate = coreml_delegate;
#else
  // Use XNNPACK on all other platforms
  static auto xnnpack_options =
      TfLiteXNNPackDelegateOptionsDefault();
  xnnpack_options.num_threads = 4;
  static TfLiteDelegate* xnnpack_delegate =
      TfLiteXNNPackDelegateCreate(&xnnpack_options);
  delegate = xnnpack_delegate;
#endif

if (delegate) {
  interpreter->ModifyGraphWithDelegate(delegate);
}
```

### Fallback Handling

Always handle the case where a delegate fails to initialize:

```c++
TfLiteDelegate* delegate = TryCreateBestDelegate();
if (delegate) {
  TfLiteStatus status =
      interpreter->ModifyGraphWithDelegate(delegate);
  if (status != kTfLiteOk) {
    // Delegate failed, fall back to CPU
    // Rebuild interpreter without delegate
  }
}
```

---

## Debugging Delegate Issues

### Common Issues

1. **Unsupported operation**: Check which ops are not supported by examining
   the delegate's `IsNodeSupportedByDelegate` logic.
2. **Data type mismatch**: Many delegates support only FP32 or only quantized
   types.
3. **Memory layout**: Some delegates require specific tensor layouts (e.g.,
   NHWC only).
4. **Dynamic shapes**: Most delegates do not support dynamic tensor shapes.

### Debugging Tools

```c++
// Enable verbose delegate logging
TfLiteHexagonDelegateOptions options =
    TfLiteHexagonDelegateOptionsDefault();
options.print_graph_profile = true;
options.print_graph_debug = true;

// Use TFLite profiling to identify which nodes are delegated
interpreter->SetProfiler(profiler);
```

### Inspecting Delegation Results

After calling `ModifyGraphWithDelegate`, you can inspect the execution plan
to see which nodes were delegated:

```c++
// After delegation, the execution plan contains delegate kernel nodes
// that represent the delegated partitions.
// Each delegate kernel's builtin_data contains TfLiteDelegateParams
// with the original node indices.
```

---

## Advanced Topics

### Tensor Sharing Between Delegates

When multiple delegates are used, tensor data may need to be copied between
delegate memory spaces. The `CopyFromBufferHandle` and `CopyToBufferHandle`
functions handle this:

```c++
// The runtime automatically calls these when:
// 1. A tensor produced by one delegate is consumed by another delegate
// 2. A tensor produced by a delegate is consumed by a CPU op
// 3. The application requests tensor data from the interpreter
```

### Delegate Serialization

Some delegates (GPU, NNAPI) support serializing their compiled models:

```c++
// GPU delegate serialization
options.experimental_flags |=
    TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
options.serialization_dir = cache_dir;
options.model_token = model_hash;

// NNAPI delegate serialization
nnapi_opts.cache_dir = cache_dir;
nnapi_opts.model_token = model_hash;
```

### Delegate Telemetry

The telemetry system in `tensorflow/lite/delegates/telemetry.h` provides
a mechanism for delegates to report performance and diagnostic information:

```c++
// Delegates can report:
// - Which nodes were delegated
// - Which nodes were unsupported (and why)
// - Performance metrics
// - Error conditions
```

---

## Summary

The TFLite delegate system provides a flexible, extensible mechanism for
hardware acceleration. Key takeaways:

1. **Use XNNPACK as the default CPU delegate** on all platforms for
   significant performance improvements.
2. **Use hardware-specific delegates** (GPU, NNAPI, CoreML, Hexagon) for
   maximum performance on supported devices.
3. **Handle fallback gracefully** when a delegate is not available.
4. **Profile your specific model** on your target device to determine the
   best delegate configuration.
5. **Consider initialization time** for latency-sensitive applications; use
   serialization where available.
6. **Validate accuracy** after enabling delegates, especially with FP16 or
   quantized inference.
7. **Use the Flex delegate** only when necessary for unsupported TF ops,
   understanding the trade-offs in binary size and performance.
