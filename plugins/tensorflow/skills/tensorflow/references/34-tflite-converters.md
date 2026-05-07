# TensorFlow Lite Converters

This document provides a comprehensive reference for the TFLite converter, which
converts TensorFlow models into the TFLite FlatBuffer format. The converter handles
op translation, quantization, and optimization.

## Table of Contents

1. [TFLite Converter API](#tflite-converter-api)
2. [Conversion Process](#conversion-process)
3. [Quantization](#quantization)
4. [Optimizations](#optimizations)
5. [Target Specifications](#target-specifications)
6. [Conversion Configuration](#conversion-configuration)
7. [Custom Ops](#custom-ops)
8. [Error Handling](#error-handling)
9. [Advanced Options](#advanced-options)
10. [Post-Training Quantization Details](#post-training-quantization-details)
11. [Quantization-Aware Training](#quantization-aware-training)
12. [Conversion Debugging](#conversion-debugging)
13. [Export Formats](#export-formats)

---

## TFLite Converter API

The TFLite converter (`tf.lite.TFLiteConverter`) converts TensorFlow models to
the TFLite FlatBuffer format.

### Creating a Converter

```python
import tensorflow as tf

# From a Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# From a SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model(
    '/path/to/saved_model',
    signature_keys=['serving_default'],
    tags={'serve'}
)

# From a concrete function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def my_func(x):
    return tf.matmul(x, weights) + bias

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [my_func.get_concrete_function()]
)

# From a SavedModel with signatures
converter = tf.lite.TFLiteConverter.from_saved_model(
    '/path/to/saved_model',
    signature_keys=['serving_default']
)
```

### Converter Methods

| Method | Description |
|--------|-------------|
| `from_keras_model(model)` | Create converter from a Keras model |
| `from_saved_model(saved_model_dir)` | Create converter from a SavedModel |
| `from_concrete_functions(funcs)` | Create converter from concrete functions |
| `convert()` | Perform the conversion and return bytes |

### Basic Conversion

```python
# Simple conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Converter Attributes

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization settings
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Target specification
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float16]

# Quantization settings
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Experimental options
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True
converter.allow_custom_ops = True
```

---

## Conversion Process

The conversion process transforms a TensorFlow model into a TFLite FlatBuffer
through multiple stages.

### Conversion Pipeline

```
TensorFlow Model (Keras/SavedModel/ConcreteFunction)
      |
      v
  +---+---+
  | Freeze |  Freeze Graph
  | Graph  |  (convert variables to constants)
  +---+---+
      |
      v
  +---+---+
  | Grappler| Graph Optimization
  | Passes  |  (constant folding, layout optimization)
  +---+---+
      |
      v
  +---+---+
  | TF ->   | Op Translation
  | TFLite  |  (TF ops -> TFLite ops)
  +---+---+
      |
      v
  +---+---+
  | Quantize| Quantization (if configured)
  |         |  (weight/activation quantization)
  +---+---+
      |
      v
  +---+---+
  | Sparsify| Sparsification (if configured)
  +---+---+
      |
      v
  TFLite FlatBuffer (.tflite)
```

### MLIR-Based Converter

The new MLIR-based converter (default in TF 2.x) uses MLIR for conversion:

```
TF SavedModel / Keras Model
      |
      v
  Import to MLIR (TF dialect)
      |
      v
  Shape Inference
      |
      v
  Legalize TF -> TFLite
      |
      v
  Quantization (in MLIR)
      |
      v
  Export to FlatBuffer
```

### TOCO (Legacy Converter)

The legacy TOCO (TensorFlow Lite Optimizing Converter) is still available:

```cpp
// From: tensorflow/lite/toco/toco_tooling.h

// Import model
std::unique_ptr<Model> Import(const TocoFlags& toco_flags,
                              const ModelFlags& model_flags,
                              const std::string& input_file_contents);

// Transform model
absl::Status TransformWithStatus(const TocoFlags& toco_flags, Model* model);

// Export model
absl::Status Export(const TocoFlags& toco_flags, const Model& model,
                    bool allow_custom_ops,
                    std::string* output_file_contents);
```

### Conversion Steps in Detail

1. **Model Loading**: Load the TF model (Keras, SavedModel, or concrete function)
2. **Graph Freezing**: Convert variables to constants using `freeze_graph`
3. **Graph Optimization**: Run Grappler passes (constant folding, pruning, etc.)
4. **Op Mapping**: Map each TF operation to its TFLite equivalent
5. **Tensor Type Conversion**: Convert TF tensor types to TFLite types
6. **Quantization**: Apply quantization if configured
7. **Buffer Serialization**: Serialize constant data into FlatBuffer
8. **FlatBuffer Construction**: Build the final `.tflite` file

---

## Quantization

Quantization reduces model size and improves inference speed by using lower-precision
numeric representations.

### Quantization Types

#### Dynamic Range Quantization

The simplest form of quantization. Weights are quantized to 8-bit integers,
but activations remain in floating point:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**Characteristics:**
- Weights: INT8 (4x smaller than FP32)
- Activations: FP32 (computed at runtime)
- No representative dataset needed
- Latency: ~2-3x speedup over FP32
- Size: ~4x reduction

#### Float16 Quantization

Weights are stored in 16-bit floating point:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

**Characteristics:**
- Weights: FP16 (2x smaller than FP32)
- Activations: FP32
- No representative dataset needed
- Minimal accuracy loss
- Size: ~2x reduction
- Good for GPU delegate

#### Full Integer Quantization (INT8)

Both weights and activations are quantized to 8-bit integers:

```python
def representative_dataset():
    for data in calibration_data:
        yield [data]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

**Characteristics:**
- Weights: INT8
- Activations: INT8
- Requires representative dataset for calibration
- Best for Edge TPU, Hexagon DSP
- Latency: ~3-4x speedup over FP32
- Size: ~4x reduction
- Some accuracy loss

#### INT16 Quantization

Weights and activations quantized to 16-bit integers:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
tflite_model = converter.convert()
```

**Characteristics:**
- Activations: INT16
- Weights: INT8
- Better accuracy than full INT8
- Moderate size reduction
- Good for models sensitive to quantization

#### Float Fallback (Mixed Precision)

Some ops quantized, others remain in FP32:

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# Don't force full INT8 - allow FP32 fallback
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
]
tflite_model = converter.convert()
```

### Per-Channel vs Per-Tensor Quantization

#### Per-Tensor Quantization

Each tensor has a single scale and zero point:

```
quantized_value = round(float_value / scale) + zero_point
float_value = (quantized_value - zero_point) * scale
```

#### Per-Channel Quantization

Each channel (along a specified axis) has its own scale and zero point:

```python
# Per-channel quantization is the default for weights in modern TFLite
# It provides better accuracy for weight tensors
```

Per-channel is default for:
- Convolution weights (per output channel)
- Fully connected weights (per output channel)
- Depthwise convolution weights (per input channel)

### Representative Dataset

The representative dataset provides calibration data for quantization:

```python
import numpy as np
import tensorflow as tf

def representative_dataset():
    # Provide 100-500 representative samples
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

# Or using a generator function
def representative_dataset_gen():
    for image in calibration_images:
        yield [np.expand_dims(image, axis=0).astype(np.float32)]
```

**Requirements:**
- Must match the expected input shape and type
- Should cover the expected input distribution
- 100-500 samples is typically sufficient
- Can use a subset of the training data

---

## Optimizations

### Optimize Enum

```python
# From: tensorflow/lite/python/lite.py

class Optimize(enum.Enum):
    DEFAULT = "DEFAULT"
    OPTIMIZE_FOR_SIZE = "OPTIMIZE_FOR_SIZE"      # Deprecated, same as DEFAULT
    OPTIMIZE_FOR_LATENCY = "OPTIMIZE_FOR_LATENCY"  # Deprecated, same as DEFAULT
    EXPERIMENTAL_SPARSITY = "EXPERIMENTAL_SPARSITY"  # Enable sparse weight optimization
```

### Applying Optimizations

```python
# Default optimization (dynamic range quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Default + sparsity
converter.optimizations = [
    tf.lite.Optimize.DEFAULT,
    tf.lite.Optimize.EXPERIMENTAL_SPARSITY,
]
```

### Optimization Effects

| Optimization | Size Reduction | Latency Improvement | Accuracy Impact |
|-------------|---------------|--------------------|-----------------|
| DEFAULT | ~4x | ~2-3x | Minimal |
| DEFAULT + FP16 | ~2x | ~2x | Minimal |
| DEFAULT + INT8 | ~4x | ~3-4x | Moderate |
| EXPERIMENTAL_SPARSITY | Additional 2-4x | Additional 1.5-2x | None (with pruned model) |

---

## Target Specifications

### TargetSpec Class

```python
# From: tensorflow/lite/python/lite.py

class TargetSpec:
    supported_ops: Set[tf.lite.OpsSet]   # Allowed op sets
    supported_types: List[tf.DType]       # Allowed data types
    experimental_supported_backends: List  # Backend targets
```

### OpsSet Options

```python
class OpsSet(enum.Enum):
    TFLITE_BUILTINS = "TFLITE_BUILTINS"             # Standard TFLite ops
    SELECT_TF_OPS = "SELECT_TF_OPS"                  # Flex ops (TF fallback)
    TFLITE_BUILTINS_INT8 = "TFLITE_BUILTINS_INT8"    # INT8-only ops
    EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 = "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8"
```

### Common Target Configurations

```python
# Standard TFLite (all builtin ops)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# TFLite + Flex (for unsupported TF ops)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

# Full INT8 (for Edge TPU / Hexagon DSP)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# FP16 weights (for GPU delegate)
converter.target_spec.supported_types = [tf.float16]
```

---

## Conversion Configuration

### Experimental New Converter

The MLIR-based converter is the default in TF 2.x:

```python
# Enable MLIR-based converter (default True in TF 2.x)
converter.experimental_new_converter = True

# Enable MLIR-based quantizer
converter.experimental_new_quantizer = True
```

### Enable MLIR Converter Flags

```python
# Force MLIR converter
converter.experimental_enable_mlir_converter = True
```

### Conversion Metadata

```python
# From: tensorflow/lite/python/lite.py

# RepresentativeDataset for calibration
class RepresentativeDataset:
    def __init__(self, input_gen):
        self.input_gen = input_gen
```

---

## Custom Ops

### Registering Custom Ops During Conversion

```python
# Define custom op registration
class MyCustomOp:
    pass

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
```

### Custom Op Implementation

Custom ops require a C/C++ implementation at runtime:

```cpp
// Custom op registration
TfLiteRegistration* Register_MY_CUSTOM_OP() {
  static TfLiteRegistration r = {
      .init = my_init,
      .free = my_free,
      .prepare = my_prepare,
      .invoke = my_invoke,
  };
  return &r;
}
```

### Custom Op Converter

For custom ops that need special conversion logic:

```python
# Use Flex ops for custom TF ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
```

---

## Error Handling

### Common Conversion Errors

#### Unsupported Operations

```
RuntimeError: Some ops are not supported by the native TFLite runtime.
If you wish to use Flex ops, please set target_spec.supported_ops
to [TFLITE_BUILTINS, SELECT_TF_OPS].
```

**Solution**: Add `SELECT_TF_OPS` to `supported_ops` or replace the unsupported op.

#### Shape Mismatch

```
ValueError: Tensor shape mismatch. Expected [...], got [...].
```

**Solution**: Ensure input shapes are consistent. Use `input_signature` with
`tf.function`.

#### Dynamic Shape Issues

```
Error: The input array X has a dynamic shape.
```

**Solution**: Use `tf.TensorSpec` with fixed shapes, or enable dynamic shape support.

#### Quantization Errors

```
RuntimeError: Quantization not yet supported for op: XXXX
```

**Solution**: Use mixed precision (allow FP32 fallback) or use `SELECT_TF_OPS`.

### Debugging Conversion

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check which ops are supported
from tensorflow.lite.python import convert
```

---

## Advanced Options

### allow_custom_ops

```python
converter.allow_custom_ops = True  # Allow ops without TFLite converters
```

When set, custom ops are included in the model without conversion. The runtime
must provide the op implementation.

### experimental_new_quantizer

```python
converter.experimental_new_quantizer = True  # Use MLIR-based quantizer
```

The new quantizer provides better accuracy and supports more quantization schemes.

### opset

```python
# Specify the opset version
converter.opset = tf.lite.OperaSetSet.TFLITE_BUILTINS
```

### Input/Output Type Override

```python
# Force integer input/output types
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

This is used with full integer quantization to ensure the model accepts and
produces integer tensors.

### Experimental Options

```python
# Dynamic shape support
converter._experimental_dynamic_update_concat = True

# Disable per-channel quantization
converter.experimental_disable_per_channel = False

# Enable stablehlo pass
converter.experimental_stablehlo_pass = False
```

---

## Post-Training Quantization Details

### Weight Quantization

Weights are quantized during conversion:

```
FP32 weight: [0.1, 0.5, 0.8, -0.3]

Quantization parameters:
  scale = (max - min) / (qmax - qmin) = (0.8 - (-0.3)) / (127 - (-128)) = 0.00431
  zero_point = round(qmin - min / scale) = round(-128 - (-0.3/0.00431)) = round(-128 + 69.6) = -58

INT8 weight: [23, 121, 176, -80]
```

### Activation Quantization

Activations are quantized using calibration data:

1. Run representative data through the model
2. Record min/max of each activation tensor
3. Compute scale and zero_point for each activation
4. Embed quantization parameters in the model

### Quantization Simulation

During quantization-aware training, fake quantization nodes simulate the
quantization effect:

```python
# Fake quantization in training
quantized = tf.quantization.fake_quant_with_min_max_args(
    inputs, min=-6.0, max=6.0, num_bits=8)
```

### Quantization Error Analysis

```python
from tensorflow.lite.tools.optimize.debugging.python.debugger import (
    QuantizationDebugger,
    QuantizationDebugOptions,
)

debugger = QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset,
)
debugger.run()
metrics = debugger.layerwise_debug_metrics
```

---

## Quantization-Aware Training

Quantization-aware training (QAT) simulates quantization during training to
minimize accuracy loss.

### tf.quantization.quantize_model

```python
import tensorflow_model_optimization as tfmot

# Apply quantization-aware training
quant_aware_model = tfmot.quantization.keras.quantize_model(model)

# Train the quantization-aware model
quant_aware_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
quant_aware_model.fit(train_dataset, epochs=10)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Fake Quant Nodes

QAT inserts fake quantization nodes that simulate the quantization/dequantization
process during training:

```
FP32 input -> FakeQuant(FP32 -> INT8 -> FP32) -> Op -> FakeQuant(...) -> FP32 output
```

The fake quantization nodes:
1. Quantize the FP32 tensor to INT8
2. Dequantize back to FP32
3. Pass the "quantized" FP32 tensor to the next operation

This allows the model to learn to compensate for quantization errors.

### Custom Quantization Configurations

```python
# Custom quantization configuration
quantize_config = tfmot.quantization.keras.Default8BitQuantizeConfig()

# Annotate specific layers for quantization
def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_dense,
)
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
```

---

## Conversion Debugging

### MLIR Diagnostic Output

```bash
# Enable MLIR diagnostics during conversion
export TF_CPP_MIN_LOG_LEVEL=1
export TF_CPP_VMODULE="mlir_import=1,tflite_converter=1"
```

### Conversion Phase Tracking

```python
# From: tensorflow/lite/python/convert_phase.py

class Component(enum.Enum):
    CONVERT_GRAPH = "CONVERT_GRAPH"
    CONVERT_OPS = "CONVERT_OPS"
    QUANTIZE = "QUANTIZE"
    SPARSIFY = "SPARSIFY"
    FINALIZE = "FINALIZE"

class SubComponent(enum.Enum):
    # Detailed sub-components
    pass

# Decorator for tracking conversion phases
def convert_phase(component, sub_component=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Include phase information in error
                raise ConverterError(
                    f"Error in {component}/{sub_component}: {e}")
        return wrapper
    return decorator
```

### Visualizing the Model

```bash
# Visualize the TFLite model
python -m tensorflow.lite.tools.visualize model.tflite model.html
```

### Model Analyzer

```python
# From: tensorflow/lite/python/analyzer.py
# Analyze model structure and op compatibility
import tensorflow as tf
from tensorflow.lite.python import analyzer

model_path = 'model.tflite'
report = analyzer.model_analyzer(model_path)
print(report)
```

---

## Export Formats

### TFLite FlatBuffer

The primary export format is the TFLite FlatBuffer (`.tflite`):

```python
# Convert and save
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### SavedModel (with TFLite)

```python
# Save as SavedModel with TFLite model
import shutil
shutil.copy('model.tflite', '/path/to/saved_model/model.tflite')
```

### Export Options

```python
# Export to different formats
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Standard TFLite
tflite_bytes = converter.convert()

# With metadata (for deployment)
from tensorflow.lite.python import metadata as _metadata
# Add metadata to the model
```

### FlatBuffer Utilities

```python
# From: tensorflow/lite/tools/flatbuffer_utils.py

# Read model
model_data = flatbuffer_utils.read_model('model.tflite')

# Write model
flatbuffer_utils.write_model(model_data, 'output.tflite')

# Strip strings (reduce size)
stripped = flatbuffer_utils.strip_strings(model_data)
```

---

## Conversion Flags (TOCO)

### TocoFlags

The legacy TOCO converter accepts flags defined in a protobuf:

```protobuf
// tensorflow/lite/toco/toco_flags.proto

message TocoFlags {
  // Input/output file format
  FileFormat input_format = 1;
  FileFormat output_format = 2;

  // Inference type
  DataType inference_type = 3;

  // Inference input type
  DataType inference_input_type = 4;

  // Default ranges for quantization
  float default_ranges_min = 5;
  float default_ranges_max = 6;

  // Drop control dependency
  bool drop_control_dependency = 7;

  // Reorder ops
  bool reorder_across_fake_quant = 8;

  // Allow custom ops
  bool allow_custom_ops = 9;

  // Post training quantize
  bool post_training_quantize = 10;

  // Quantize weights
  bool quantize_weights = 11;

  // Rnn states
  repeated RnnState rnn_states = 12;

  // Conversion flags
  bool enable_tflite_resource_variables = 13;
  bool enable_select_tf_ops = 14;
}
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `lite/python/lite.py` | Python converter API |
| `lite/python/convert_saved_model.py` | SavedModel conversion |
| `lite/python/convert_phase.py` | Conversion phase tracking |
| `lite/python/analyzer.py` | Model analyzer |
| `lite/python/interpreter.py` | Python interpreter API |
| `lite/toco/toco_tooling.h` | TOCO conversion API |
| `lite/toco/toco_convert.cc` | TOCO converter entry point |
| `lite/toco/toco_cmdline_flags.h` | TOCO command-line flags |
| `lite/toco/model.h` | TOCO model representation |
| `compiler/mlir/lite/tf_tfl_passes.h` | MLIR TF->TFLite passes |
| `compiler/mlir/lite/flatbuffer_translate.h` | FlatBuffer translation |
| `compiler/mlir/lite/flatbuffer_export.h` | FlatBuffer export |
| `compiler/mlir/lite/flatbuffer_import.h` | FlatBuffer import |

---

## Advanced Conversion Topics

### Conversion from JAX HLO

The converter also supports converting JAX HLO computations:

```python
# From: tensorflow/lite/python/convert.py
# convert_jax_hlo function handles JAX HLO conversion
```

### MLIR-Based Conversion Pipeline Details

The MLIR-based converter follows these steps:

```
1. Import SavedModel/Keras to MLIR TF dialect
   - Parse the TF graph into MLIR operations
   - Map TF op types to MLIR TF dialect ops
   - Preserve shape and type information

2. Run shape inference
   - Propagate shapes through the graph
   - Resolve unknown dimensions where possible
   - Report errors for incompatible shapes

3. Legalize TF -> TFLite
   - Convert each TF operation to its TFLite equivalent
   - Insert necessary reshape/transpose for compatibility
   - Handle operations that differ between TF and TFLite

4. Post-legalization optimization
   - Remove redundant reshapes and transposes
   - Fuse adjacent operations
   - Apply TFLite-specific optimizations

5. Quantization (in MLIR)
   - Insert quantize/dequantize ops
   - Propagate quantization parameters
   - Verify quantization correctness

6. Export to FlatBuffer
   - Serialize the MLIR graph to TFLite FlatBuffer format
   - Write constant data as buffers
   - Build the operator tables
```

### Conversion Pass Registration

```cpp
// From: tensorflow/compiler/mlir/lite/tf_tfl_passes.h
// Creates the full TF -> TFLite conversion pass pipeline

// From: tensorflow/compiler/mlir/lite/register_lite_dialects.h
// Registers all dialects needed for TFLite conversion
```

### FlatBuffer Export Flags

```cpp
// From: tensorflow/compiler/mlir/lite/flatbuffer_export_flags.h
// Flags controlling FlatBuffer export behavior
```

### FlatBuffer Operator Definition

```cpp
// From: tensorflow/compiler/mlir/lite/flatbuffer_operator.h
// Defines the mapping between MLIR ops and FlatBuffer operators
```

### Offset Buffer Management

```cpp
// From: tensorflow/compiler/mlir/lite/offset_buffer.h
// Manages offset buffers in the FlatBuffer output
```

### Version Compatibility in Conversion

```cpp
// From: tensorflow/compiler/mlir/lite/version.h
// Handles op version compatibility during conversion
```

### Conversion for Specific Backends

#### GPU Delegate Optimization

When targeting the GPU delegate:

```python
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float16]

# The converter will:
# 1. Keep all ops as TFLite builtins
# 2. Use FP16 weights where possible
# 3. Avoid quantization that GPU doesn't support
```

#### Edge TPU Optimization

For Google Coral Edge TPU:

```python
# Full INT8 quantization is required for Edge TPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

#### NNAPI Targeting

For Android NNAPI:

```python
# Use operations that NNAPI supports efficiently
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
# Quantized models generally work better with NNAPI
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
```

### Conversion with Multiple Signatures

```python
# Convert a SavedModel with multiple signatures
converter = tf.lite.TFLiteConverter.from_saved_model(
    '/path/to/saved_model',
    signature_keys=['serving_default', 'classify', 'regress']
)

# Each signature becomes a separate entry point in the TFLite model
```

### Handling Dynamic Shapes During Conversion

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)  # Dynamic batch
])
def dynamic_model(x):
    return model(x)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [dynamic_model.get_concrete_function()]
)
converter.experimental_enable_dynamic_input_tensors = True
tflite_model = converter.convert()
```

### Weight Clustering (Codebook Quantization)

Weight clustering reduces model size by grouping similar weight values:

```python
import tensorflow_model_optimization as tfmot

# Apply weight clustering
clustering_params = {
    'number_of_clusters': 16,
    'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
}
clustered_model = tfmot.clustering.keras.cluster_weights(model, **clustering_params)

# Fine-tune the clustered model
clustered_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
clustered_model.fit(train_dataset, epochs=5)

# Strip clustering wrappers
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = converter.convert()
```

### Conversion Performance Tips

1. **Use the MLIR converter** (default in TF 2.x) for better compatibility
2. **Freeze variables** before conversion for smaller models
3. **Use concrete functions** with explicit signatures for predictable shapes
4. **Test with representative data** to validate quantization accuracy
5. **Profile the converted model** to identify bottlenecks

### Model Validation After Conversion

```python
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Compare with original TF model
input_data = np.random.random_sample([1, 224, 224, 3]).astype(np.float32)

# TFLite inference
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
interpreter.invoke()
tflite_output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# Original TF inference
tf_output = original_model(input_data).numpy()

# Compare outputs
np.testing.assert_allclose(tflite_output, tf_output, rtol=1e-5, atol=1e-5)
```
