# ONNX Runtime Reference - Chapter 45: Model Conversion Tools

This chapter covers ONNX Runtime's model optimization, shape inference, quantization, format conversion, analysis, and the model transformer API.

---

## 45.1 ONNX Model Optimization

### 45.1.1 ort.optimizer Overview

ONNX Runtime provides a comprehensive model optimizer that applies graph-level transformations to improve inference performance.

```python
import onnxruntime as ort
from onnxruntime.tools import optimizer

# Optimize a model
optimized_model = optimizer.optimize_model("model.onnx")

# Save the optimized model
optimized_model.save_model_to_file("model_optimized.onnx")
```

### 45.1.2 Optimization Levels

```python
from onnxruntime.transformers import optimizer

# Level 1: Basic optimizations
optimized = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',               # Model type hint
    num_heads=12,                     # Number of attention heads
    hidden_size=768,                  # Hidden dimension
    opt_level=1                       # Optimization level
)

# Level 2: Extended optimizations (includes level 1)
optimized = optimizer.optimize_model(
    "model.onnx",
    opt_level=2
)

# Level 99: All optimizations (levels 1 + 2 + layout)
optimized = optimizer.optimize_model(
    "model.onnx",
    opt_level=99
)
```

### 45.1.3 Available Optimization Passes

```python
# List of optimization passes available in ort.optimizer
optimization_passes = {
    # Level 1: Basic optimizations
    "ConstantFolding": {
        "description": "Evaluate constant expressions at compile time",
        "level": 1,
        "enabled_by_default": True,
    },
    "DeadCodeElimination": {
        "description": "Remove unused nodes and graph outputs",
        "level": 1,
        "enabled_by_default": True,
    },
    "CommonSubexpressionElimination": {
        "description": "Identify and merge identical sub-expressions",
        "level": 1,
        "enabled_by_default": True,
    },
    "ConstantPropagation": {
        "description": "Propagate constant values through the graph",
        "level": 1,
        "enabled_by_default": True,
    },
    "ShapeFolding": {
        "description": "Evaluate Shape, Size, and other shape-related ops at compile time",
        "level": 1,
        "enabled_by_default": True,
    },

    # Level 2: Extended optimizations
    "MatMulAddFusion": {
        "description": "Fuse MatMul + Add into MatMul + BiasAdd (or Gemm)",
        "level": 2,
        "enabled_by_default": True,
    },
    "ConvAddFusion": {
        "description": "Fuse Conv + Add into Conv with bias",
        "level": 2,
        "enabled_by_default": True,
    },
    "ConvBNFusion": {
        "description": "Fuse Conv + BatchNorm into Conv with adjusted weights",
        "level": 2,
        "enabled_by_default": True,
    },
    "ConvMulAddFusion": {
        "description": "Fuse Conv + Mul + Add pattern",
        "level": 2,
        "enabled_by_default": True,
    },
    "GeluFusion": {
        "description": "Detect and fuse GELU activation pattern",
        "level": 2,
        "enabled_by_default": True,
    },
    "BiasGeluFusion": {
        "description": "Fuse BiasAdd + GELU into a single operation",
        "level": 2,
        "enabled_by_default": True,
    },
    "LayerNormalizationFusion": {
        "description": "Detect and fuse LayerNorm pattern (ReduceMean + Sub + Pow + Add + Sqrt + Div + Mul + Add)",
        "level": 2,
        "enabled_by_default": True,
    },
    "SkipLayerNormalizationFusion": {
        "description": "Fuse skip connection + LayerNorm pattern",
        "level": 2,
        "enabled_by_default": True,
    },
    "AttentionFusion": {
        "description": "Fuse self-attention pattern into a single Attention node",
        "level": 2,
        "enabled_by_default": True,
    },
    "EmbedLayerNormalizationFusion": {
        "description": "Fuse embedding + LayerNorm pattern for BERT-like models",
        "level": 2,
        "enabled_by_default": True,
    },
    "QLinearConcatFusion": {
        "description": "Fuse quantized Concat operations",
        "level": 2,
        "enabled_by_default": True,
    },
    "ReshapeFusion": {
        "description": "Merge consecutive Reshape operations",
        "level": 2,
        "enabled_by_default": True,
    },
    "TransposeOptimizer": {
        "description": "Eliminate unnecessary Transpose operations",
        "level": 2,
        "enabled_by_default": True,
    },
    "GatherToSliceFusion": {
        "description": "Convert Gather with integer indices to Slice",
        "level": 2,
        "enabled_by_default": True,
    },
    "SliceToDynamicSlice": {
        "description": "Convert Slice to DynamicSlice for compatibility",
        "level": 2,
        "enabled_by_default": True,
    },

    # Level 3/4: Layout optimizations
    "NHWCLayoutTransformation": {
        "description": "Convert NCHW layout to NHWC for better GPU performance",
        "level": 99,
        "enabled_by_default": False,  # EP-dependent
    },
    "ChannelFirstToChannelLast": {
        "description": "Layout transformation for specific EPs",
        "level": 99,
        "enabled_by_default": False,
    },
}
```

### 45.1.4 Transformer-Specific Optimizer

```python
from onnxruntime.transformers import optimizer

# BERT optimization
optimized_model = optimizer.optimize_model(
    "bert_model.onnx",
    model_type='bert',
    num_heads=12,
    hidden_size=768,
    opt_level=2,
    use_gpu=True,                        # Enable GPU-specific optimizations
    opt_only=True,                        # Only optimize, don't change graph structure
    attention_op_type='Attention',        # Use contrib Attention op
    use_multi_head_attention=True,        # Use MultiHeadAttention contrib op
    enable_shape_inference=True,          # Run shape inference
    use_external_data_format=False,       # Keep data inline
)

# GPT-2 optimization
optimized_model = optimizer.optimize_model(
    "gpt2_model.onnx",
    model_type='gpt2',
    num_heads=12,
    hidden_size=768,
    opt_level=2,
)

# T5 optimization
optimized_model = optimizer.optimize_model(
    "t5_model.onnx",
    model_type='t5',
    num_heads=16,
    hidden_size=1024,
)

# Whisper optimization
optimized_model = optimizer.optimize_model(
    "whisper_model.onnx",
    model_type='whisper',
    num_heads=12,
    hidden_size=768,
)

# Save optimized model
optimized_model.save_model_to_file("model_optimized.onnx", use_external_data_format=True)
```

### 45.1.5 Optimization Configuration

```python
from onnxruntime.transformers import optimizer

# Fine-grained optimization control
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',
    # Enable/disable specific passes
    enable_gelu=True,
    enable_layer_norm=True,
    enable_attention=True,
    enable_skip_layer_norm=True,
    enable_embed_layer_norm=True,
    enable_bias_skip_layer_norm=True,
    enable_bias_gelu=True,
    enable_gelu_approximation=False,    # Use exact GELU
    # Performance options
    use_multi_head_attention=False,      # Use contrib MultiHeadAttention
    num_heads=12,
    hidden_size=768,
    # Precision options
    fp16=False,                          # Convert to FP16
    input_int32=False,                   # Use int32 inputs
    # Graph structure options
    disable_shape_inference=False,
    allow_onnx_opset_mode=True,
    opt_level=2,
)
```

---

## 45.2 Shape Inference Tools

### 45.2.1 ONNX Shape Inference

```python
import onnx
from onnx import shape_inference

# Method 1: In-place shape inference
model = onnx.load("model.onnx")
inferred_model = shape_inference.infer_shapes(model)
onnx.save(inferred_model, "model_with_shapes.onnx")

# Method 2: Check model with shape inference
onnx.checker.check_model("model.onnx", full_check=True)

# Method 3: Shape inference with validation
from onnx import shape_inference
inferred_model = shape_inference.infer_shapes(model, check_type=True,
                                               strict_mode=False)
```

### 45.2.2 ORT Shape Inference

```python
import onnxruntime as ort

# Shape inference during session creation
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable shape inference logging
options.log_severity_level = 0  # VERBOSE

session = ort.InferenceSession("model.onnx", options)

# Get input/output shapes
for inp in session.get_inputs():
    print(f"Input: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

for out in session.get_outputs():
    print(f"Output: {out.name}, Shape: {out.shape}, Type: {out.type}")
```

### 45.2.3 Dynamic Shape Handling

```python
import onnx
from onnx import helper, TensorProto, shape_inference

# Model with dynamic dimensions
batch_size = "batch_size"  # Symbolic dimension
seq_length = "seq_length"

X = helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                   [batch_size, seq_length, 768])

# After shape inference, dynamic dimensions are propagated
model = onnx.load("model.onnx")
inferred = shape_inference.infer_shapes(model)

# In ORT, dynamic shapes are resolved at runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
# Dynamic shapes adapt to actual input sizes
output = session.run(None, {
    'X': np.random.randn(4, 128, 768).astype(np.float32)  # batch=4, seq=128
})
output = session.run(None, {
    'X': np.random.randn(1, 512, 768).astype(np.float32)  # batch=1, seq=512
})
```

### 45.2.4 Shape Inference C++ API

```cpp
// onnxruntime/core/graph/graph.cc
Status Graph::PerformTypeShapeInference() {
    // Run ONNX shape inference
    ONNX_NAMESPACE::shape_inference::ShapeInferenceOptions options;
    options.infer_shapes_for_concrete_graph = true;

    // Build inference context
    GraphInferenceContext context(
        opschema_registry_,
        /*resolve_distances=*/false,
        /*enable_data_propagation=*/true,
        model_path_,
        custom_schema_registry_);

    // Infer types and shapes
    ORT_RETURN_IF_ERROR(
        ONNX_NAMESPACE::shape_inference::InferShapes(
            graph_proto_, opset_imports_, *opschema_registry_));

    // Update NodeArg types and shapes from inference results
    for (auto& node_arg : node_args_) {
        const auto& inferred_type = inferred_types[node_arg->Name()];
        node_arg->SetType(inferred_type);
        if (inferred_type.has_tensor_type() &&
            inferred_type.tensor_type().has_shape()) {
            node_arg->SetShape(inferred_type.tensor_type().shape());
        }
    }

    return Status::OK();
}
```

---

## 45.3 Model Quantization Tools

### 45.3.1 Quantization Overview

ONNX Runtime provides comprehensive quantization support through `onnxruntime.quantization`:

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, quantize_qat
from onnxruntime.quantization import QuantType, QuantFormat
```

### 45.3.2 Dynamic Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Basic dynamic quantization
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QUInt8,  # Weight quantization type
    per_channel=True,               # Per-channel weight quantization
    reduce_range=False,             # Use full 8-bit range
    op_types_to_quantize=['MatMul', 'Gemm'],  # Target ops
    extra_options={
        "WeightSymmetric": True,            # Symmetric weight quantization
        "ActivationSymmetric": False,       # Asymmetric activation quantization
        "EnableSubgraph": False,            # Don't quantize subgraphs
        "DisableMatMul4BitsQuant": False,   # Enable 4-bit quantization
        "MatMul4BitsQuantType": "NF4",      # 4-bit quantization format
        "ForceQuantizeNoFloatCheck": False,
    }
)
```

### 45.3.3 Static Quantization

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from onnxruntime.quantization import CalibrationDataReader
import numpy as np

# Step 1: Prepare calibration data reader
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data, input_name):
        self.data = calibration_data
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        data = {self.input_name: self.data[self.index]}
        self.index += 1
        return data

    def rewind(self):
        self.index = 0

calibration_data = [np.random.randn(1, 128, 768).astype(np.float32)
                    for _ in range(100)]
reader = MyCalibrationDataReader(calibration_data, "input_ids")

# Step 2: Quantize with calibration
quantize_static(
    model_input="model.onnx",
    model_output="model_static_quant.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,       # QDQ format (quantize/dequantize nodes)
    weight_type=QuantType.QInt8,        # 8-bit integer weights
    per_channel=True,                    # Per-channel quantization
    fuse_dynamic_quant=True,            # Fuse dynamic quant patterns
    activation_type=QuantType.QUInt8,    # Activation type
    calibrate_method="MinMax",           # Calibration method
    extra_options={
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        "CalibTensorRangeSymmetric": False,
        "CalibMovingAverage": False,
        "CalibMovingAverageConstant": 0.01,
        "UseQDQContribOps": False,
        "DisableMatMul4BitsQuant": False,
        "MatMulBnb4BitQuantType": "nf4",   # Options: "nf4", "fp4"
        "QuantizeMatMulBnb4Bits": True,
        "BlockSizeFor4BitsQuant": 128,
        "ComputeAccuracy": True,
        "QuantizeBias": True,
        "NodesToExclude": [],
        "MinMSEWeight": False,
        "StaticQuantization": True,
    }
)
```

### 45.3.4 Quantization-Aware Training (QAT) Support

```python
from onnxruntime.quantization import quantize_qat, QuantType, QuantFormat

# Quantize a model that was trained with QAT (contains FakeQuant nodes)
quantize_qat(
    model_input="qat_model.onnx",
    model_output="qat_quantized.onnx",
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    per_channel=True,
    extra_options={
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        "UseQDQContribOps": False,
    }
)
```

### 45.3.5 Post-Training Quantization with Mixed Precision

```python
from onnxruntime.quantization import quantize, QuantType, QuantFormat

# Mixed precision quantization (some ops FP16, some INT8)
quantize(
    model_input="model.onnx",
    model_output="model_mixed.onnx",
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    per_channel=True,
    nodes_to_exclude=["MatMul_1", "MatMul_2"],  # Keep these in FP32
    nodes_to_quantize=["Conv_3", "Conv_4"],      # Quantize these
    op_types_to_quantize=["Conv", "MatMul"],
    extra_options={
        "UseQDQContribOps": True,
        "DisableQDQDecoder": True,          # Don't quantize decoder layers
    }
)
```

### 45.3.6 4-Bit Quantization (GPTQ/NF4)

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

# 4-bit weight-only quantization (for LLMs)
quantize_static(
    model_input="llama_model.onnx",
    model_output="llama_4bit.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt4,        # 4-bit quantization
    per_channel=True,
    extra_options={
        "DisableMatMul4BitsQuant": False,
        "MatMulBnb4BitQuantType": "nf4",  # "nf4" or "fp4"
        "BlockSizeFor4BitsQuant": 128,     # Block size for quantization
        "QuantizeMatMulBnb4Bits": True,
        "QuantizeBias": False,             # Keep bias in FP32
    }
)
```

### 45.3.7 Quantization Format Comparison

| Format | Description | Best For |
|--------|-------------|----------|
| `QDQ` (Quantize/Dequantize) | Insert Q/DQ nodes around ops | Most compatible |
| `QOperator` | Replace ops with quantized versions | Legacy support |
| `QQQ` | Weight-only quantization | Large models, memory-bound |
| `MatMulBnb4Bits` | 4-bit BNB quantization | LLMs, extreme compression |

### 45.3.8 QuantType Options

```python
from onnxruntime.quantization import QuantType

# Available quantization types
QuantType.QUInt8    # Unsigned 8-bit integer (0-255), good for activations
QuantType.QInt8     # Signed 8-bit integer (-128 to 127), good for weights
QuantType.QInt4     # Signed 4-bit integer (-8 to 7), for extreme compression
QuantType.QUInt4    # Unsigned 4-bit integer (0-15)
QuantType.QInt16    # Signed 16-bit integer
QuantType.QUInt16   # Unsigned 16-bit integer
QuantType.FLOAT16   # IEEE 754 half-precision
QuantType.BFLOAT16  # Brain floating point (Google format)
```

### 45.3.9 Calibration Methods

```python
# Available calibration methods for static quantization
calibration_methods = {
    "MinMax": {
        "description": "Use min/max of activation values as quantization range",
        "speed": "Fast",
        "accuracy": "Good",
    },
    "Entropy": {
        "description": "Use KL divergence to find optimal quantization range",
        "speed": "Medium",
        "accuracy": "Better",
    },
    "Percentile": {
        "description": "Use percentile of activation distribution",
        "speed": "Medium",
        "accuracy": "Good",
        "default_percentile": 99.999,
    },
    "MSE": {
        "description": "Minimize mean squared error of quantized values",
        "speed": "Slow",
        "accuracy": "Best",
    },
}
```

---

## 45.4 ONNX to ORT Format Conversion

### 45.4.1 Why Convert to ORT Format?

- **Faster loading**: Zero-copy FlatBuffers vs. protobuf parsing
- **Smaller file size**: More compact binary format
- **Pre-optimized**: Graph optimizations are baked in
- **Pre-packed weights**: Weights are pre-packed for target EP

### 45.4.2 Conversion Tool

```python
import onnxruntime as ort
from onnxruntime.tools import convert_onnx_models_to_ort

# Convert ONNX model to ORT format
convert_onnx_models_to_ort(
    model_path="model.onnx",
    output_path="model.ort",
    optimize=True,                    # Apply optimizations before conversion
    graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    execution_providers=["CUDA"],     # Target EP for pre-packing
    provider_options=[{"device_id": 0}],
)
```

### 45.4.3 Conversion via Session API

```python
import onnxruntime as ort

# Method 1: Convert by loading and saving
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.optimized_model_filepath = "model.ort"  # Save optimized model

session = ort.InferenceSession("model.onnx", options)
# Session creation triggers conversion and saves to "model.ort"

# Method 2: Convert via C API
import ctypes
ort_lib = ctypes.CDLL("libonnxruntime.so")
# ... C API calls ...
```

### 45.4.4 Batch Conversion

```python
import os
import onnxruntime as ort
from onnxruntime.tools import convert_onnx_models_to_ort

def batch_convert_to_ort(model_dir, output_dir):
    """Convert all ONNX models in a directory to ORT format."""
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".onnx"):
                input_path = os.path.join(root, f)
                output_path = os.path.join(output_dir, f.replace(".onnx", ".ort"))

                try:
                    convert_onnx_models_to_ort(
                        model_path=input_path,
                        output_path=output_path,
                        optimize=True,
                        execution_providers=["CPUExecutionProvider"],
                    )
                    print(f"Converted: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Failed: {input_path}: {e}")
```

---

## 45.5 Model Analyzer

### 45.5.1 Model Statistics

```python
import onnxruntime as ort
import onnx

def analyze_model(model_path):
    """Analyze an ONNX model and print detailed statistics."""
    model = onnx.load(model_path)
    graph = model.graph

    print("=" * 60)
    print(f"Model Analysis: {model_path}")
    print("=" * 60)

    # Metadata
    print(f"\n--- Metadata ---")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Model Version: {model.model_version}")
    print(f"Domain: {model.domain}")

    # Opset versions
    print(f"\n--- Opset Imports ---")
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx (default)"
        print(f"  {domain}: version {opset.version}")

    # Graph statistics
    print(f"\n--- Graph Statistics ---")
    print(f"Graph Name: {graph.name}")
    print(f"Number of Nodes: {len(graph.node)}")
    print(f"Number of Inputs: {len(graph.input)}")
    print(f"Number of Outputs: {len(graph.output)}")
    print(f"Number of Initializers: {len(graph.initializer)}")
    print(f"Number of Value Info: {len(graph.value_info)}")

    # Operator distribution
    op_counts = {}
    for node in graph.node:
        op_type = node.op_type
        domain = node.domain or ""
        key = f"{domain}::{op_type}" if domain else op_type
        op_counts[key] = op_counts.get(key, 0) + 1

    print(f"\n--- Operator Distribution ---")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")

    # Parameter statistics
    total_params = 0
    total_bytes = 0
    dtype_sizes = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8,
                   10: 2, 11: 8, 12: 4, 13: 8, 16: 2}

    print(f"\n--- Parameter Statistics ---")
    for init in graph.initializer:
        numel = 1
        for dim in init.dims:
            numel *= dim
        total_params += numel

        dtype_size = dtype_sizes.get(init.data_type, 4)
        total_bytes += numel * dtype_size

        if numel > 1000000:
            print(f"  {init.name}: {numel:,} elements "
                  f"({numel * dtype_size / 1024 / 1024:.1f} MB)")

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Parameter Size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Total Model File Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")

    return {
        "num_nodes": len(graph.node),
        "num_params": total_params,
        "param_size_mb": total_bytes / 1024 / 1024,
        "op_counts": op_counts,
    }
```

### 45.5.2 Session-Level Analysis

```python
import onnxruntime as ort

def analyze_session(model_path, providers=None):
    """Analyze model loading and inference characteristics."""
    if providers is None:
        providers = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.log_severity_level = 0  # VERBOSE
    options.profile_file = "profile.json"  # Enable profiling

    # Enable memory and timing profiling
    options.enable_profiling = True

    session = ort.InferenceSession(model_path, options, providers=providers)

    # Get model metadata
    meta = session.get_modelmeta()
    print(f"Producer: {meta.producer_name}")
    print(f"Graph Name: {meta.graph_name}")
    print(f"Description: {meta.description}")
    print(f"Domain: {meta.domain}")
    print(f"Version: {meta.version}")
    print(f"Custom Metadata: {meta.custom_metadata_map}")

    # Get provider info
    print(f"\nProviders: {session.get_providers()}")

    # Get inputs/outputs
    print(f"\nInputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

    print(f"\nOutputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")

    # Overridable initializers
    print(f"\nOverridable Initializers:")
    for init in session.get_overridable_initializers():
        print(f"  {init.name}: shape={init.shape}, type={init.type}")

    # Profiling data
    profile_data = session.end_profiling()
    print(f"\nProfile data saved to: {profile_data}")

    return session
```

### 45.5.3 Node-Level Analysis

```python
import onnx

def analyze_node_types(model_path):
    """Analyze node types and their attributes in the model."""
    model = onnx.load(model_path)
    graph = model.graph

    # Group nodes by type
    nodes_by_type = {}
    for node in graph.node:
        op_type = node.op_type
        if op_type not in nodes_by_type:
            nodes_by_type[op_type] = []
        nodes_by_type[op_type].append(node)

    for op_type, nodes in sorted(nodes_by_type.items()):
        print(f"\n{'=' * 40}")
        print(f"Op: {op_type} ({len(nodes)} instances)")
        print(f"{'=' * 40}")

        # Analyze attributes
        all_attrs = {}
        for node in nodes:
            for attr in node.attribute:
                if attr.name not in all_attrs:
                    all_attrs[attr.name] = set()
                if attr.type == 1:  # FLOAT
                    all_attrs[attr.name].add(attr.f)
                elif attr.type == 2:  # INT
                    all_attrs[attr.name].add(attr.i)
                elif attr.type == 7:  # INTS
                    all_attrs[attr.name].add(tuple(attr.ints))
                elif attr.type == 3:  # STRING
                    all_attrs[attr.name].add(attr.s.decode())

        for attr_name, values in all_attrs.items():
            if len(values) == 1:
                print(f"  {attr_name} = {list(values)[0]}")
            else:
                print(f"  {attr_name} = {len(values)} unique values")

        # Analyze input patterns
        input_counts = set(len(n.input) for n in nodes)
        output_counts = set(len(n.output) for n in nodes)
        print(f"  Inputs: {input_counts}")
        print(f"  Outputs: {output_counts}")
```

---

## 45.6 Model Transformer API

### 45.6.1 Overview

The Model Transformer API provides a programmatic way to apply graph transformations to ONNX models.

```python
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.models import model_utils
```

### 45.6.2 Custom Transformer

```python
import onnx
from onnx import helper, TensorProto
from onnxruntime.transformers import optimizer

class CustomTransformer:
    """Custom model transformer that applies user-defined transformations."""

    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph

    def replace_node(self, old_op_type, new_op_type, new_domain=""):
        """Replace all nodes of old_op_type with new_op_type."""
        for node in self.graph.node:
            if node.op_type == old_op_type:
                node.op_type = new_op_type
                if new_domain:
                    node.domain = new_domain
        return self

    def remove_node(self, op_type):
        """Remove all nodes of the given type."""
        nodes_to_remove = [n for n in self.graph.node if n.op_type == op_type]
        for node in nodes_to_remove:
            self.graph.node.remove(node)
        return self

    def insert_node_after(self, target_op_type, new_node):
        """Insert a new node after all nodes of the target type."""
        for i, node in enumerate(self.graph.node):
            if node.op_type == target_op_type:
                # Rename original output
                original_output = node.output[0]
                intermediate_name = f"{original_output}_pre_insert"

                # Update original node output
                node.output[0] = intermediate_name

                # Set new node inputs/outputs
                new_node.input.append(intermediate_name)
                new_node.output.append(original_output)

                # Insert new node
                self.graph.node.insert(i + 1, new_node)
        return self

    def fuse_nodes(self, pattern, fused_op_type):
        """Fuse a sequence of nodes matching the pattern into a single node."""
        # Build adjacency graph
        output_to_node = {}
        for node in self.graph.node:
            for output in node.output:
                output_to_node[output] = node

        for node in list(self.graph.node):
            if node.op_type == pattern[0]:
                # Try to match the pattern
                matched_nodes = [node]
                current = node
                match = True

                for next_op in pattern[1:]:
                    # Find consumer of current node's output
                    next_node = None
                    for n in self.graph.node:
                        if any(inp in current.output for inp in n.input):
                            if n.op_type == next_op:
                                next_node = n
                                break

                    if next_node is None:
                        match = False
                        break

                    matched_nodes.append(next_node)
                    current = next_node

                if match:
                    # Create fused node
                    fused_node = helper.make_node(
                        fused_op_type,
                        inputs=list(matched_nodes[0].input),
                        outputs=list(matched_nodes[-1].output),
                        name=f"fused_{matched_nodes[0].name}",
                    )

                    # Remove original nodes and add fused node
                    for n in matched_nodes:
                        self.graph.node.remove(n)
                    self.graph.node.append(fused_node)

        return self

    def save(self, output_path):
        """Save the transformed model."""
        onnx.save(self.model, output_path)
        return self

# Usage
transformer = CustomTransformer("model.onnx")
transformer.replace_node("Gelu", "FastGelu", "com.microsoft")
transformer.fuse_nodes(["MatMul", "Add"], "MatMulAdd")
transformer.save("model_transformed.onnx")
```

### 45.6.3 Predefined Transformers

```python
from onnxruntime.transformers import fusion_utils

# Available fusion transformers
class MatMulAddFusion:
    """Fuse MatMul + Add into Gemm or MatMul with bias."""
    def __init__(self, model):
        self.model = model

    def apply(self):
        # Find MatMul nodes followed by Add
        for node in self.model.graph.node:
            if node.op_type == "MatMul":
                # Check if the output feeds into an Add with a constant bias
                matmul_output = node.output[0]
                for consumer in self.model.graph.node:
                    if consumer.op_type == "Add" and matmul_output in consumer.input:
                        # Check for constant bias
                        bias_input = [i for i in consumer.input
                                     if i != matmul_output][0]
                        if self.is_constant(bias_input):
                            # Create Gemm node
                            self.create_gemm(node, consumer, bias_input)
        return self.model

class LayerNormFusion:
    """Fuse ReduceMean + Sub + Pow + Add + Sqrt + Div + Mul + Add into LayerNorm."""
    pattern = ["ReduceMean", "Sub", "Pow", "Add", "Sqrt", "Div", "Mul", "Add"]
    target_op = "LayerNormalization"

class AttentionFusion:
    """Fuse self-attention pattern into Attention contrib op."""
    # Pattern: MatMul(Q) + MatMul(K) + MatMul(V) + Softmax + MatMul
    target_op = "Attention"
```

### 45.6.4 Transformer Pipeline

```python
from onnxruntime.transformers import optimizer

# Apply a pipeline of transformations
model = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',
    opt_level=2,
)

# Apply additional custom transformations
model = model.transform(CustomTransformer)

# Get the ONNX model proto
onnx_model = model.model

# Inspect transformation results
print(f"Nodes after optimization: {len(onnx_model.graph.node)}")
```

---

## 45.7 Python Tools for Model Manipulation

### 45.7.1 ONNX Model Builder

```python
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def build_simple_model():
    """Build a simple ONNX model from scratch."""
    # Define inputs and outputs
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 224, 224])
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [64, 3, 7, 7])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64, 112, 112])

    # Create nodes
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['conv_out'],
        name='Conv_0',
        kernel_shape=[7, 7],
        strides=[2, 2],
        padding=[3, 3],
    )

    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['Y'],
        name='Relu_0',
    )

    # Create graph
    graph = helper.make_graph(
        [conv_node, relu_node],
        'simple_conv_net',
        [X],
        [Y],
        initializer=[numpy_helper.from_array(
            np.random.randn(64, 3, 7, 7).astype(np.float32), name='W')],
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    # Validate
    onnx.checker.check_model(model)

    return model

# Save the model
model = build_simple_model()
onnx.save(model, "simple_model.onnx")
```

### 45.7.2 Model Editor

```python
import onnx
from onnx import helper, TensorProto

class OnnxModelEditor:
    """Utility class for editing ONNX models."""

    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph

    def rename_input(self, old_name, new_name):
        """Rename a model input."""
        for inp in self.graph.input:
            if inp.name == old_name:
                inp.name = new_name
                self._rename_in_graph(old_name, new_name)
                return True
        return False

    def rename_output(self, old_name, new_name):
        """Rename a model output."""
        for out in self.graph.output:
            if out.name == old_name:
                out.name = new_name
                return True
        return False

    def remove_output(self, name):
        """Remove a model output."""
        for i, out in enumerate(self.graph.output):
            if out.name == name:
                del self.graph.output[i]
                return True
        return False

    def add_metadata(self, key, value):
        """Add a metadata entry."""
        entry = helper.make_string_string_entry(key, value)
        self.model.metadata_props.append(entry)

    def set_opset_version(self, domain, version):
        """Set or update an opset version."""
        for opset in self.model.opset_import:
            if opset.domain == domain:
                opset.version = version
                return
        self.model.opset_import.append(
            helper.make_opsetid(domain, version))

    def replace_initializer(self, name, new_value):
        """Replace an initializer (weight) with new values."""
        for init in self.graph.initializer:
            if init.name == name:
                new_tensor = numpy_helper.from_array(new_value, name=name)
                init.CopyFrom(new_tensor)
                return True
        return False

    def extract_subgraph(self, input_names, output_names):
        """Extract a subgraph from the model."""
        # Find all nodes needed to produce output_names from input_names
        required_nodes = set()
        required_initializers = set()

        # Build reverse mapping
        output_to_node = {}
        for node in self.graph.node:
            for output in node.output:
                output_to_node[output] = node

        # Trace back from outputs
        queue = list(output_names)
        visited = set()

        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)

            if name in output_to_node:
                node = output_to_node[name]
                required_nodes.add(id(node))
                for input_name in node.input:
                    if input_name not in input_names:
                        queue.append(input_name)

            # Check if it's an initializer
            for init in self.graph.initializer:
                if init.name == name and name not in input_names:
                    required_initializers.add(name)

        # Build new graph
        new_nodes = [n for n in self.graph.node if id(n) in required_nodes]
        new_inits = [i for i in self.graph.initializer
                     if i.name in required_initializers]

        new_inputs = [inp for inp in self.graph.input
                      if inp.name in input_names]
        new_outputs = [out for out in self.graph.output
                       if out.name in output_names]

        new_graph = helper.make_graph(
            new_nodes,
            self.graph.name + "_subgraph",
            new_inputs,
            new_outputs,
            new_inits,
        )

        new_model = helper.make_model(new_graph)
        return new_model

    def save(self, output_path):
        onnx.save(self.model, output_path)

    def _rename_in_graph(self, old_name, new_name):
        """Rename a tensor throughout the graph."""
        for node in self.graph.node:
            for i, inp in enumerate(node.input):
                if inp == old_name:
                    node.input[i] = new_name
            for i, out in enumerate(node.output):
                if out == old_name:
                    node.output[i] = new_name
```

### 45.7.3 Model Comparison Tool

```python
import onnx
import numpy as np

def compare_models(model_path_1, model_path_2):
    """Compare two ONNX models."""
    model1 = onnx.load(model_path_1)
    model2 = onnx.load(model_path_2)

    g1 = model1.graph
    g2 = model2.graph

    print("=== Model Comparison ===")

    # Compare inputs
    inputs1 = {i.name for i in g1.input}
    inputs2 = {i.name for i in g2.input}
    print(f"\nInputs only in model 1: {inputs1 - inputs2}")
    print(f"Inputs only in model 2: {inputs2 - inputs1}")

    # Compare outputs
    outputs1 = {o.name for o in g1.output}
    outputs2 = {o.name for o in g2.output}
    print(f"\nOutputs only in model 1: {outputs1 - outputs2}")
    print(f"Outputs only in model 2: {outputs2 - outputs1}")

    # Compare initializers
    inits1 = {i.name: i for i in g1.initializer}
    inits2 = {i.name: i for i in g2.initializer}

    common_inits = set(inits1.keys()) & set(inits2.keys())
    print(f"\nCommon initializers: {len(common_inits)}")
    print(f"Only in model 1: {set(inits1.keys()) - set(inits2.keys())}")
    print(f"Only in model 2: {set(inits2.keys()) - set(inits1.keys())}")

    # Check if weights changed
    for name in common_inits:
        w1 = numpy_helper.to_array(inits1[name])
        w2 = numpy_helper.to_array(inits2[name])
        if not np.allclose(w1, w2):
            diff = np.abs(w1 - w2)
            print(f"  Weight changed: {name}, max_diff={diff.max():.6f}, "
                  f"mean_diff={diff.mean():.6f}")

    # Compare node types
    ops1 = [n.op_type for n in g1.node]
    ops2 = [n.op_type for n in g2.node]
    print(f"\nNodes in model 1: {len(ops1)} ({len(set(ops1))} unique types)")
    print(f"Nodes in model 2: {len(ops2)} ({len(set(ops2))} unique types)")
```

### 45.7.4 Model Validation Tool

```python
import onnx
import onnxruntime as ort
import numpy as np

def validate_model(model_path, test_inputs=None, reference_outputs=None):
    """Validate an ONNX model for correctness."""
    issues = []

    # Step 1: ONNX format validation
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model, full_check=True)
        print("[PASS] ONNX format validation")
    except onnx.checker.ValidationError as e:
        issues.append(f"ONNX validation error: {e}")
        print(f"[FAIL] ONNX format validation: {e}")

    # Step 2: Shape inference
    try:
        from onnx import shape_inference
        inferred = shape_inference.infer_shapes(model)
        print("[PASS] Shape inference")
    except Exception as e:
        issues.append(f"Shape inference error: {e}")
        print(f"[FAIL] Shape inference: {e}")

    # Step 3: ORT loading
    try:
        session = ort.InferenceSession(model_path)
        print("[PASS] ORT session creation")
    except Exception as e:
        issues.append(f"ORT loading error: {e}")
        print(f"[FAIL] ORT session creation: {e}")
        return issues

    # Step 4: Inference test
    if test_inputs is not None:
        try:
            # Generate test inputs if not provided
            if test_inputs is None:
                test_inputs = {}
                for inp in session.get_inputs():
                    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                    if inp.type == 'tensor(float)':
                        test_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
                    elif inp.type == 'tensor(int64)':
                        test_inputs[inp.name] = np.random.randint(0, 100, size=shape).astype(np.int64)

            outputs = session.run(None, test_inputs)
            print("[PASS] Inference test")

            # Step 5: Compare with reference
            if reference_outputs is not None:
                for i, (out, ref) in enumerate(zip(outputs, reference_outputs)):
                    if np.allclose(out, ref, rtol=1e-4, atol=1e-4):
                        print(f"[PASS] Output {i} matches reference")
                    else:
                        diff = np.abs(out - ref)
                        print(f"[FAIL] Output {i} mismatch: "
                              f"max_diff={diff.max():.6f}")
                        issues.append(f"Output {i} mismatch")

        except Exception as e:
            issues.append(f"Inference error: {e}")
            print(f"[FAIL] Inference test: {e}")

    return issues
```

### 45.7.5 Model Visualization Tool

```python
def print_model_graph(model_path, max_nodes=50):
    """Print a text representation of the model graph."""
    model = onnx.load(model_path)
    graph = model.graph

    print(f"Graph: {graph.name}")
    print(f"{'=' * 80}")

    # Print inputs
    print("\nInputs:")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    # Print nodes
    print(f"\nNodes ({len(graph.node)} total):")
    for i, node in enumerate(graph.nodes[:max_nodes]):
        inputs = ", ".join(node.input)
        outputs = ", ".join(node.output)
        attrs = {a.name: self._format_attr(a) for a in node.attribute}
        attr_str = f" {attrs}" if attrs else ""
        print(f"  [{i}] {node.op_type}: [{inputs}] -> [{outputs}]{attr_str}")

    if len(graph.node) > max_nodes:
        print(f"  ... ({len(graph.node) - max_nodes} more nodes)")

    # Print outputs
    print("\nOutputs:")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
```

---

## 45.8 Command-Line Tools

### 45.8.1 ONNX Runtime CLI Tools

```bash
# Model optimization
python -m onnxruntime.tools.optimize_model \
    --input model.onnx \
    --output model_optimized.onnx \
    --opt_level 2

# Model conversion to ORT format
python -m onnxruntime.tools.convert_onnx_models_to_ort \
    --input model.onnx \
    --output model.ort

# Quantization (dynamic)
python -m onnxruntime.quantization.quantize \
    --model_input model.onnx \
    --model_output model_int8.onnx \
    --quantization_type dynamic \
    --weight_type uint8

# Model analysis
python -m onnxruntime.tools.analyze_model \
    --model model.onnx \
    --verbose
```

### 45.8.2 ONNX Python Tools

```bash
# ONNX shape inference
python -m onnx shape_inference --input model.onnx --output model_inferred.onnx

# ONNX model checker
python -m onnx checker --input model.onnx --full-check

# ONNX model editor
python -m onnx.tools.update_model_dims \
    --model model.onnx \
    --output model_dynamic.onnx \
    --inputs "input:batch,seq_len" \
    --outputs "output:batch,seq_len"
```

---

## 45.9 Summary

| Tool | Purpose | API |
|------|---------|-----|
| `ort.optimizer` | Graph optimization and fusion | `optimizer.optimize_model()` |
| Shape Inference | Compute tensor shapes statically | `onnx.shape_inference.infer_shapes()` |
| `quantize_dynamic` | Runtime quantization (no calibration) | `onnxruntime.quantization.quantize_dynamic()` |
| `quantize_static` | Post-training static quantization | `onnxruntime.quantization.quantize_static()` |
| `quantize_qat` | Quantization-aware training quantization | `onnxruntime.quantization.quantize_qat()` |
| ORT Conversion | ONNX to ORT FlatBuffers format | `convert_onnx_models_to_ort()` |
| Model Analyzer | Model statistics and diagnostics | `analyze_model()` |
| Model Transformer | Custom graph transformations | `CustomTransformer` class |
| Model Editor | Programmatic model modification | `OnnxModelEditor` class |
| Model Validator | End-to-end model validation | `validate_model()` |
