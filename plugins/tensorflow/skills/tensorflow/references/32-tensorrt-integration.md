# TensorFlow-TensorRT Integration (TF-TRT)

This document provides a comprehensive reference for TF-TRT, which integrates
NVIDIA TensorRT with TensorFlow for inference acceleration. TF-TRT automatically
converts TensorFlow subgraphs into TensorRT-optimized engines.

## Table of Contents

1. [TF-TRT Overview](#tf-trt-overview)
2. [Conversion API](#conversion-api)
3. [Conversion Parameters](#conversion-parameters)
4. [TRTEngineOp](#trtengineop)
5. [Graph Segmentation](#graph-segmentation)
6. [Precision Calibration](#precision-calibration)
7. [Engine Caching](#engine-caching)
8. [Dynamic Shapes](#dynamic-shapes)
9. [Op Compatibility](#op-compatibility)
10. [Fallback to TF](#fallback-to-tf)
11. [Performance Tuning](#performance-tuning)
12. [Known Limitations](#known-limitations)
13. [Debugging](#debugging)
14. [TF-TRT with Keras](#tf-trt-with-keras)
15. [Version Compatibility](#version-compatibility)

---

## TF-TRT Overview

TF-TRT integrates NVIDIA TensorRT into TensorFlow to optimize deep learning inference.
It works by:

1. Identifying subgraphs in the TensorFlow graph that are compatible with TensorRT
2. Converting those subgraphs into TensorRT engines
3. Encapsulating the engines as `TRTEngineOp` nodes in the TensorFlow graph
4. Executing the TensorRT engines during inference alongside native TensorFlow ops

### Architecture

```
TensorFlow Graph
      |
      v
  +---+---+
  | Graph  |  Segment Graph
  | Segment|  (identify TRT-compatible subgraphs)
  +---+---+
      |
      v
  +---+---+
  | Node   |  Convert Nodes
  | Convert|  (TF ops -> TensorRT layers)
  +---+---+
      |
      v
  +---+---+
  | Engine |  Build TensorRT Engine
  | Build  |  (optimize for target GPU)
  +---+---+
      |
      v
  TRTEngineOp (in TF graph)
```

### Prerequisites

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA 11.0+
- TensorRT 8.4+
- TensorFlow 2.x with GPU support

### Key Source Directories

```
tensorflow/compiler/tf2tensorrt/
  |-- convert/            # Graph conversion, node converters
  |-- segment/            # Graph segmentation (Union-Find)
  |-- kernels/            # TRTEngineOp kernel implementation
  |-- ops/                # TF op registrations
  |-- plugin/             # TensorRT plugin implementations
  |-- common/             # Common utilities
  |-- utils/              # Helper utilities
  +-- stub/               # TensorRT stub libraries
```

---

## Conversion API

### Python API: TrtGraphConverter

The primary Python API for TF-TRT conversion:

```python
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Create converter
converter = trt.TrtGraphConverter(
    input_graph_def=graph_def,
    nodes_blacklist=['output'],  # Output nodes to keep as TF ops
    max_workspace_size_bytes=1 << 30,  # 1GB workspace
    precision_mode=trt.TrtPrecisionMode.FP16,
    minimum_segment_size=3,
    maximum_cached_engines=1,
    use_calibration=True,
    use_dynamic_shape=True,
    dynamic_shape_profile_strategy=trt.DynamicShapeProfileStrategy.OPTIMAL,
    allow_build_at_runtime=True,
)

# Convert
converted_graph_def = converter.convert()

# Build engines (for static engines)
converted_graph_def = converter.build(input_fn=my_input_fn)

# Save
converter.save('/path/to/saved_model')

# Summary
converter.summary()
```

### C++ API: ConvertAndBuild

```cpp
// From: tensorflow/compiler/tf2tensorrt/trt_convert_api.h

struct TfTrtConversionParams {
  size_t max_workspace_size_bytes;
  TrtPrecisionMode precision_mode = TrtPrecisionMode::FP32;
  int minimum_segment_size = 3;
  int max_cached_engines = 1;
  bool use_calibration = true;
  bool use_dynamic_shape = true;
  ProfileStrategy profile_strategy = ProfileStrategy::kRange;
  bool allow_build_at_runtime = true;
  bool convert_to_static_engine = true;
};

StatusOr<GraphDef> ConvertAndBuild(
    const GraphDef& frozen_graph_def,
    const std::vector<string>& input_names,
    const std::vector<string>& output_names,
    const std::vector<std::vector<tensorflow::Tensor>>& inputs,
    const TfTrtConversionParams& conv_params);

StatusOr<GraphDef> ConvertAndBuild(
    SavedModelBundle* bundle,
    const std::string& signature_key,
    const std::vector<std::vector<tensorflow::Tensor>>& inputs,
    const TfTrtConversionParams& conversion_params);
```

---

## Conversion Parameters

### Precision Modes

| Mode | Description | Accuracy | Speedup |
|------|-------------|----------|---------|
| `FP32` | Full 32-bit floating point | Full | Baseline |
| `FP16` | 16-bit floating point | Slight loss | ~2-3x |
| `INT8` | 8-bit integer (requires calibration) | Moderate loss | ~3-5x |
| `BF16` | Bfloat16 (if supported) | Slight loss | ~2x |

```python
# Set precision mode
converter = trt.TrtGraphConverter(
    input_graph_def=graph_def,
    precision_mode=trt.TrtPrecisionMode.FP16,
)
```

### Workspace Size

The maximum workspace size that TensorRT can use during engine building:

```python
# 1 GB workspace
max_workspace_size_bytes = 1 << 30

# 4 GB workspace (for large models)
max_workspace_size_bytes = 4 << 30
```

Larger workspaces allow TensorRT to consider more optimization strategies but
use more GPU memory during compilation.

### Minimum Segment Size

The minimum number of nodes a subgraph must contain to be converted to TensorRT:

```python
# Only convert subgraphs with at least 3 nodes
minimum_segment_size = 3
```

Many small TRT subgraphs can be detrimental to performance due to the overhead
of switching between TF and TRT execution.

### Maximum Cached Engines

The maximum number of TensorRT engines to cache for dynamic shapes:

```python
max_cached_engines = 1  # Static shape (recommended)
max_cached_engines = 10  # Multiple dynamic shapes
```

### Allow Build at Runtime

Whether to build TensorRT engines at runtime if no cached engine matches:

```python
allow_build_at_runtime = True   # Build on first use (default)
allow_build_at_runtime = False  # Fail if no cached engine
```

Setting to `False` is recommended for production to avoid runtime compilation overhead.

### Dynamic Shape Configuration

```python
use_dynamic_shape = True
dynamic_shape_profile_strategy = trt.DynamicShapeProfileStrategy.OPTIMAL
```

Profile strategies (see Dynamic Shapes section for details):
- `ProfileStrategy.RANGE`
- `ProfileStrategy.OPTIMAL`
- `ProfileStrategy.PRIORITY`

---

## TRTEngineOp

The `TRTEngineOp` encapsulates a TensorRT engine within the TensorFlow graph.

### Op Registration

```cpp
// From: tensorflow/compiler/tf2tensorrt/kernels/trt_engine_op.cc
// Kernel implementation for TRTEngineOp
```

### Engine Execution

When the `TRTEngineOp` is executed:

1. Check the engine cache for a matching engine
2. If no match and `allow_build_at_runtime` is true, build a new engine
3. Set up input bindings (TF tensors -> TRT buffers)
4. Execute the TensorRT engine
5. Copy output bindings (TRT buffers -> TF tensors)
6. Cache the engine for future use

### Engine Serialization

TensorRT engines can be serialized and embedded in the GraphDef:

```python
# Convert with static engine embedding
converter = trt.TrtGraphConverter(
    input_graph_def=graph_def,
    convert_to_static_engine=True,
    max_cached_engines=1,
)
```

Serialized engines are platform-specific (GPU architecture, driver version).

---

## Graph Segmentation

The segmentation process identifies subgraphs of the TensorFlow graph that can
be handled by TensorRT.

### SegmentGraph Function

```cpp
// From: tensorflow/compiler/tf2tensorrt/segment/segment.h

struct SegmentOptions {
  int minimum_segment_size = 2;
  bool use_implicit_batch = true;
  std::optional<int> maximum_batch_size = std::nullopt;
  bool allow_dynamic_non_batch_dim = false;
  std::set<string> exclude_node_list;
};

struct Segment {
  ClusterProperty property;
  std::set<const Node*, NodePtrCompare> nodes;
};

using SegmentVector = std::vector<Segment>;

Status SegmentGraph(
    const Graph* tf_graph,
    const grappler::GraphProperties* graph_properties,
    const std::function<Status(const Node*)>& candidate_fn,
    const std::function<bool(const Edge*)>& input_candidate_fn,
    const std::function<bool(const Edge*)>& output_candidate_fn,
    const SegmentOptions& options,
    SegmentVector* segments);
```

### Union-Find Algorithm

Segmentation uses a Union-Find data structure:

```cpp
// From: tensorflow/compiler/tf2tensorrt/segment/union_find.h
// Union-Find implementation for graph segmentation
```

### Segmentation Rules

A node is a candidate for TRT conversion if:

1. The operation has a registered TensorRT converter
2. The data types are supported
3. The operation is on a GPU device
4. The node is not in the exclusion list

### Segment Merging

After initial candidate identification:
1. Adjacent candidates are merged into segments using Union-Find
2. Small segments (below `minimum_segment_size`) are rejected
3. Segments with unsupported boundary types are split

### EngineConnection

Each segment's inputs and outputs are described by `EngineConnection`:

```cpp
struct EngineConnection {
  const string outside_node_name;
  const int outside_id;
  const int outside_port;
  const string inside_node_name;
  const int inside_id;
  const int inside_port;
  DataType connection_type;
  const bool is_input_edge;
  const int port_number;
};
```

---

## Precision Calibration

### INT8 Calibration

INT8 quantization requires calibration to determine the optimal quantization ranges
for each tensor.

### Calibration Process

1. **Create calibration dataset**: A representative dataset of input samples
2. **Run calibration**: Pass samples through the model to compute activation ranges
3. **Build engine**: Use calibration data to quantize the engine

```python
# Create representative dataset
def calibration_input_fn():
    for data in calibration_dataset:
        yield [data]

# Convert with INT8 calibration
converter = trt.TrtGraphConverter(
    input_graph_def=graph_def,
    precision_mode=trt.TrtPrecisionMode.INT8,
    use_calibration=True,
)
converted_graph = converter.convert()
converted_graph = converter.build(input_fn=calibration_input_fn)
```

### Calibration Cache

Calibration results are cached in the serialized engine, so the calibration
process only needs to run once.

### INT8 Calibration Data Requirements

- Typically 500-1000 representative samples
- Should cover the expected input distribution
- Batch size should match inference batch size

---

## Engine Caching

### Cache Management

TensorRT engines are cached to avoid recompilation:

```python
# Save converted model with cached engines
converter.save('/path/to/saved_model')

# Load and reuse cached engines
converter = trt.TrtGraphConverter(
    input_saved_model_dir='/path/to/saved_model',
)
```

### Serialized Engine Storage

Engines are stored as:
- **Static engines**: Serialized and embedded in the GraphDef as node attributes
- **Dynamic engines**: Stored in a cache directory alongside the SavedModel

### Cache Invalidation

Engines must be regenerated when:
- GPU architecture changes
- TensorRT version changes
- CUDA driver version changes
- Input shapes change significantly

---

## Dynamic Shapes

TF-TRT supports dynamic input shapes through TensorRT's optimization profiles.

### Shape Tensors vs Data Tensors

- **Shape tensors**: Small integer tensors that describe the shape of data tensors
- **Data tensors**: The actual computation data

### Profile Strategies

```python
# From: tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h

class ProfileStrategy:
    kRange = "Range"           # Create profiles covering min-max range
    kOptimal = "Optimal"       # Create profile for optimal shape
    kRangeOptimal = "RangeOptimal"  # Combination of Range + Optimal
    kPriority = "Priority"     # Priority-based profile selection
```

### Profile Creation

Each optimization profile specifies minimum, optimum, and maximum dimensions:

```
Profile 0:
  min:    [1, 224, 224, 3]
  opt:    [8, 224, 224, 3]
  max:    [16, 224, 224, 3]
```

### Dynamic Shape Usage

```python
converter = trt.TrtGraphConverter(
    input_graph_def=graph_def,
    use_dynamic_shape=True,
    dynamic_shape_profile_strategy=trt.DynamicShapeProfileStrategy.OPTIMAL,
)

# Provide sample inputs for profile creation
def input_fn():
    yield [tf.zeros([1, 224, 224, 3])]
    yield [tf.zeros([8, 224, 224, 3])]
    yield [tf.zeros([16, 224, 224, 3])]

converted = converter.build(input_fn=input_fn)
```

---

## Op Compatibility

### Supported Operations

Most common TF operations have TensorRT converters:

| Category | Supported Operations |
|----------|---------------------|
| **Arithmetic** | Add, Sub, Mul, Div, Min, Max, Pow, Sqrt, Rsqrt |
| **Activation** | Relu, Relu6, Sigmoid, Tanh, Elu, LeakyRelu, Selu |
| **Normalization** | BatchNorm, FusedBatchNorm, LayerNorm |
| **Convolution** | Conv1D, Conv2D, Conv3D, DepthwiseConv2D |
| **Pooling** | MaxPool, AvgPool, GlobalAvgPool |
| **Fully Connected** | MatMul, BiasAdd, Dense |
| **Reduction** | Mean, Sum, Min, Max, Prod |
| **Concat/Split** | ConcatV2, Split, Slice, StridedSlice |
| **Reshape** | Reshape, Transpose, ExpandDims, Squeeze |
| **Padding** | Pad, MirrorPad |
| **Softmax** | Softmax, LogSoftmax |
| **Elementwise** | Abs, Ceil, Floor, Round, Sign, Neg |
| **Comparison** | Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual |
| **Logical** | LogicalAnd, LogicalOr, LogicalNot |
| **Misc** | Identity, Cast, ClipByValue, Gather, TopK |

### Unsupported Operations

Operations without TensorRT converters fall back to native TensorFlow:

- Custom user operations
- Operations with unsupported data types
- Operations with unsupported attributes
- String and resource operations

### Op Converter Registry

```cpp
// From: tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h
// Registry of TF op to TensorRT layer converters
```

Each converter registers itself with the registry:

```cpp
// Example converter registration
REGISTER_TRT_OP_CONVERTER(make_name("Conv2D"),
                          make_converter(ConvertConv2D));
```

---

## Fallback to TF

When TensorRT cannot handle an operation, it falls back to native TensorFlow execution.

### Fallback Mechanism

1. Unsupported operations remain as native TF nodes
2. The graph is segmented such that TRT-incompatible operations are outside TRT segments
3. Data flows between TRT segments and TF segments through standard TF tensor passing

### Performance Impact

Frequent switching between TRT and TF execution has overhead:
- Memory copies between TRT and TF buffers
- Context switching
- Kernel launch overhead

Minimize fallbacks by:
- Using only TRT-supported operations
- Setting an appropriate `minimum_segment_size`
- Using a single TRT segment when possible

---

## Performance Tuning

### Workspace Size

Larger workspace allows more optimization strategies:

```python
# For small models
max_workspace_size_bytes = 1 << 29  # 512 MB

# For large models
max_workspace_size_bytes = 1 << 32  # 4 GB

# Maximum (TensorRT 8.4+)
max_workspace_size_bytes = (1 << 63) - 512
```

### Batch Size

Optimal batch size depends on GPU memory and model size:

- Larger batch sizes generally improve throughput
- Batch size must be divisible by warp size (32)
- For dynamic shapes, specify optimal batch in profiles

### Engine Caching Strategy

```python
# Pre-build engines for all expected shapes
shapes = [(1, 224, 224, 3), (4, 224, 224, 3), (8, 224, 224, 3), (16, 224, 224, 3)]
for shape in shapes:
    converter.build(input_fn=lambda: [tf.zeros(shape)])
```

### FP16 Optimization

```python
# FP16 is recommended for most GPU inference
converter = trt.TrtGraphConverter(
    precision_mode=trt.TrtPrecisionMode.FP16,
)
```

FP16 provides significant speedup on NVIDIA GPUs with Tensor Cores (V100, T4, A100, H100).

### TensorRT Level Optimizations

TensorRT applies these optimizations during engine building:

1. **Layer fusion**: Combines consecutive operations
2. **Kernel auto-tuning**: Selects optimal kernel implementations
3. **Precision calibration**: Quantizes to lower precision
4. **Memory optimization**: Minimizes memory footprint
5. **Multi-stream execution**: Overlaps computation and data transfer

---

## Known Limitations

### Operations Not Supported

- Operations on string tensors
- Operations on resource variables (some cases)
- Dynamic control flow (tf.while_loop with dynamic trip count)
- Some custom operations
- Operations with complex types

### Dynamic Shapes Limitations

- Not all operations support dynamic shapes in TensorRT
- Dynamic rank (number of dimensions changing) is not supported
- Shape tensors must be integers and have known bounds
- Some operations require static dimensions for certain axes

### Memory Limitations

- TensorRT engines can be large (hundreds of MB for large models)
- Workspace size affects both compilation time and runtime memory
- Serialized engines may exceed the 2GB protobuf limit for very large models

### Platform Specificity

- Engines are specific to GPU architecture (cannot run A100 engine on T4)
- Engines are specific to TensorRT version
- Engines may not be portable across CUDA driver versions

---

## Debugging

### Environment Variables

```bash
# Enable TF-TRT logging
TF_CPP_MIN_LOG_LEVEL=0

# Set TensorRT logging level
TRT_LOGGER_LEVEL=0  # VERBOSE

# Disable TRT conversion (for debugging)
TF_DISABLE_TRT=1

# Enable TRT engine dump
TRT_ENGINE_DUMP_DIR=/tmp/trt_engines

# Print TRT conversion info
TRT_CONVERT_INFO=1
```

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# TF-TRT logs conversion information
converter = trt.TrtGraphConverter(...)
converted = converter.convert()
converter.summary()  # Print conversion summary
```

### Common Debugging Steps

1. **Check segmentation**: Verify which ops are converted to TRT
2. **Check precision**: Verify FP16/INT8 accuracy
3. **Check shapes**: Ensure dynamic shapes have proper profiles
4. **Check engine cache**: Verify engines are being reused
5. **Check fallbacks**: Identify ops falling back to TF

### Conversion Summary

```python
converter.summary()
```

The summary shows:
- Number of TRT segments
- Number of converted operations
- Operations that fell back to TF
- Engine precision
- Workspace size

---

## TF-TRT with Keras

### Applying to Keras Models

```python
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Save Keras model as SavedModel
model.save('/path/to/saved_model')

# Convert SavedModel with TF-TRT
converter = trt.TrtGraphConverter(
    input_saved_model_dir='/path/to/saved_model',
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=1 << 30,
)
converted_graph = converter.convert()
converter.save('/path/to/trt_saved_model')

# Load and use the converted model
model = tf.saved_model.load('/path/to/trt_saved_model')
output = model(input_data)
```

### Keras Model Compatibility

Most Keras models work with TF-TRT. Common issues:
- Custom layers may not have TRT converters
- Lambda layers with arbitrary Python code cannot be converted
- Models with dynamic control flow may not convert fully

---

## Version Compatibility

### Minimum Versions

| Component | Minimum Version |
|-----------|----------------|
| TensorFlow | 2.0+ |
| TensorRT | 8.4+ |
| CUDA | 11.0+ |
| GPU Compute Capability | 6.0+ (Pascal) |

### Recommended Versions

| Component | Recommended Version |
|-----------|-------------------|
| TensorFlow | 2.12+ |
| TensorRT | 8.6+ |
| CUDA | 12.0+ |
| GPU | Ampere (8.0+) or newer |

### Version Checking

```python
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

# Check TensorRT version
from tensorflow.python.compiler.tensorrt import trt_convert as trt
print(f"TF-TRT available: {trt is not None}")
```

### TensorRT Version Checks in Code

```cpp
// From: tensorflow/compiler/tf2tensorrt/common/utils.h
// Version checking macros
#if IS_TRT_VERSION_GE(8, 4, 0, 0)
  // TensorRT 8.4+ specific code
#endif
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `compiler/tf2tensorrt/trt_convert_api.h` | Conversion API (C++) |
| `compiler/tf2tensorrt/trt_convert_api.cc` | Conversion API implementation |
| `compiler/tf2tensorrt/segment/segment.h` | Graph segmentation |
| `compiler/tf2tensorrt/segment/union_find.h` | Union-Find for segmentation |
| `compiler/tf2tensorrt/convert/convert_nodes.h` | Node converters |
| `compiler/tf2tensorrt/convert/convert_graph.h` | Graph conversion |
| `compiler/tf2tensorrt/convert/op_converter_registry.h` | Converter registry |
| `compiler/tf2tensorrt/convert/trt_parameters.cc` | TRT parameters |
| `compiler/tf2tensorrt/kernels/trt_engine_op.cc` | Engine op kernel |
| `compiler/tf2tensorrt/plugin/trt_plugin.h` | TRT plugin interface |
| `compiler/tf2tensorrt/common/utils.h` | Common utilities |

---

## Advanced TF-TRT Topics

### TRT Plugin System

TF-TRT provides a plugin system for operations not natively supported by TensorRT:

```cpp
// From: tensorflow/compiler/tf2tensorrt/plugin/trt_plugin.h
// Base class for TF-TRT plugins

class TrtPlugin {
 public:
  virtual int getNbOutputs() const = 0;
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                              const nvinfer1::Dims* inputs,
                                              int nbInputDims) = 0;
  virtual void configureWithFormat(const nvinfer1::Dims* inputDims,
                                    int nbInputs,
                                    const nvinfer1::Dims* outputDims,
                                    int nbOutputs,
                                    nvinfer1::DataType type,
                                    nvinfer1::TensorFormat format,
                                    int maxBatchSize) = 0;
  virtual size_t getWorkspaceSize(int maxBatchSize) const = 0;
  virtual int enqueue(int batchSize, const void* const* inputs,
                       void** outputs, void* workspace,
                       cudaStream_t stream) = 0;
};
```

Plugins allow custom GPU kernel implementations to be used within TensorRT engines.

### Algorithm Selection

TensorRT can try multiple algorithm implementations for each layer and select the fastest:

```cpp
// From: tensorflow/compiler/tf2tensorrt/convert/algorithm_selector.h
// Controls algorithm selection for TensorRT layers
```

Algorithm selection can be:
- **Automatic**: TensorRT profiles algorithms and picks the best (default)
- **Timing cache**: Reuses timing data from previous compilations
- **Manual**: User-specified algorithm choices

```cpp
// From: tensorflow/compiler/tf2tensorrt/convert/timing_cache.h
// Timing cache for TensorRT algorithm selection
```

### Weight Conversion

TF-TRT converts TensorFlow weights to TensorRT format:

```cpp
// From: tensorflow/compiler/tf2tensorrt/convert/weights.cc
// Weight conversion utilities
```

Weight handling:
- Constants are extracted from the TF graph
- Weights are converted to the target precision (FP32, FP16, INT8)
- Converted weights are stored in TRT engine

### Logger Registry

TF-TRT uses a logger registry for TensorRT logging:

```cpp
// From: tensorflow/compiler/tf2tensorrt/convert/logger_registry.h
// Registry for TRT logger implementations
```

### Graph Optimization Pass

TF-TRT integrates with TensorFlow's graph optimization as a pass:

```cpp
// From: tensorflow/compiler/tf2tensorrt/convert/trt_optimization_pass.h
// TF-TRT graph optimization pass

// From: tensorflow/compiler/tf2tensorrt/convert/trt_layout_optimization_pass.cc
// Layout optimization pass for TRT
```

### Multi-Engine Models

For models with multiple independent TRT-eligible subgraphs, TF-TRT creates
multiple `TRTEngineOp` instances:

```
TF Graph:
  [TF Ops] -> [TRTEngineOp #1] -> [TF Ops] -> [TRTEngineOp #2] -> [TF Ops]
```

Each engine is independently optimized and cached.

### TRTEngineOp Attributes

The `TRTEngineOp` stores several attributes:

| Attribute | Description |
|-----------|-------------|
| `serialized_engine` | Serialized TensorRT engine bytes |
| `max_workspace_size_bytes` | Maximum workspace for engine building |
| `precision_mode` | Target precision (FP32/FP16/INT8) |
| `calibration_data` | INT8 calibration data |
| `dynamic_batch_size` | Whether batch dimension is dynamic |
| `input_shapes` | Input shape information |
| `output_shapes` | Output shape information |
| `segment_sizes` | Size of each segment |

### TRT Engine Resource Management

TF-TRT manages engine resources through specialized ops:

```cpp
// From: tensorflow/compiler/tf2tensorrt/kernels/trt_engine_resource_ops.cc
// Ops for managing TRT engine resources (creation, deletion)
```

### Calibration Data Operations

```cpp
// From: tensorflow/compiler/tf2tensorrt/kernels/get_calibration_data_op.cc
// Op for retrieving INT8 calibration data
```

### Batch Size Considerations

For optimal performance with TRT:

| Scenario | Recommendation |
|----------|---------------|
| Fixed batch size | Use implicit batch mode |
| Dynamic batch size | Use explicit batch mode with profiles |
| Batch size = 1 | Use optimization profiles with min=1 |
| Large batch sizes | Ensure sufficient GPU memory |

### TRT Engine Building Strategies

TF-TRT supports two engine building strategies:

1. **Static engine building** (during `converter.build()`):
   - Engine is built during conversion
   - Serialized and embedded in the GraphDef
   - No runtime compilation overhead
   - Recommended for production

2. **Dynamic engine building** (at runtime):
   - Engine is built on first use
   - Cached for subsequent uses
   - Adds latency to first inference
   - Useful during development

### Error Recovery

When a TRT engine fails at runtime:
- If `allow_build_at_runtime=True`, a new engine is built
- If the build also fails, the subgraph falls back to native TF
- Errors are logged for diagnosis

### Memory Management

TRT engine memory considerations:
- Engine size: Typically 10-100x the model weight size
- Workspace: Additional GPU memory during execution
- Cache: Multiple cached engines use proportionally more memory
- Fragmentation: Dynamic allocation can cause memory fragmentation

### Model Conversion Workflow

A typical TF-TRT production workflow:

```python
# Step 1: Train and save the model
model.fit(train_data)
model.save('/models/tf_model')

# Step 2: Convert with TF-TRT
from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.TrtGraphConverter.Params(
    precision_mode='FP16',
    max_workspace_size_bytes=1 << 30,
    minimum_segment_size=3,
    allow_build_at_runtime=False,
)
converter = trt.TrtGraphConverter(
    input_saved_model_dir='/models/tf_model',
    conversion_params=conversion_params,
)
converter.convert()
converter.save('/models/trt_model')

# Step 3: Deploy
model = tf.saved_model.load('/models/trt_model')
# Ready for inference with no compilation overhead
```

### TF-TRT and tf.function

TF-TRT works with `tf.function` decorated functions:

```python
@tf.function
def serve_fn(input_tensor):
    return model(input_tensor)

# Save as SavedModel with signature
tf.saved_model.save(
    model,
    '/models/tf_model',
    signatures={'serving_default': serve_fn.get_concrete_function(
        tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)
    )}
)

# Then convert the SavedModel with TF-TRT
```

### Common TF-TRT Patterns for Production

#### Pattern 1: Pre-build All Engines

```python
# Build engines for all expected batch sizes during deployment preparation
for batch_size in [1, 4, 8, 16, 32]:
    converter.build(input_fn=lambda: [tf.zeros([batch_size, 224, 224, 3])])
converter.save('/models/trt_model_multi_batch')
```

#### Pattern 2: Dynamic Shape with Optimal Profile

```python
# Use dynamic shapes with optimal profile for the most common batch size
converter = trt.TrtGraphConverter(
    input_saved_model_dir='/models/tf_model',
    use_dynamic_shape=True,
    dynamic_shape_profile_strategy=trt.DynamicShapeProfileStrategy.OPTIMAL,
)
```

#### Pattern 3: INT8 with Calibration

```python
# Full INT8 pipeline with calibration dataset
def calibration_fn():
    for batch in calibration_dataset.batch(1):
        yield [batch['image']]

converter = trt.TrtGraphConverter(
    input_saved_model_dir='/models/tf_model',
    precision_mode=trt.TrtPrecisionMode.INT8,
)
converter.convert()
converter.build(input_fn=calibration_fn)
converter.save('/models/trt_model_int8')
```

### TF-TRT Conversion Metrics

After conversion, examine these metrics:

- **Conversion ratio**: Percentage of ops converted to TensorRT
- **Number of segments**: How many TRT subgraphs were created
- **Fallback ops**: Which operations fell back to TensorFlow
- **Engine size**: Size of each serialized TensorRT engine
- **Build time**: Time taken to build each engine

### Handling TRT Conversion Failures

Common failure scenarios and solutions:

| Failure | Cause | Solution |
|---------|-------|----------|
| Segfault during conversion | TensorRT version mismatch | Ensure compatible TRT version |
| Out of memory | Workspace too large | Reduce `max_workspace_size_bytes` |
| Unsupported op error | No TRT converter for op | Add to `nodes_blacklist` or use Flex |
| Shape mismatch | Dynamic shapes not configured | Set `use_dynamic_shape=True` |
| Slow conversion | Large model + INT8 calibration | Reduce calibration dataset size |
| Accuracy degradation | INT8 quantization | Use FP16 instead or improve calibration |
