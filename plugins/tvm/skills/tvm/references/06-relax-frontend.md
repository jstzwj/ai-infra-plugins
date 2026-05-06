# Apache TVM -- Chapter 6: Relax Frontend -- Model Import

This reference covers the Relax frontend modules for importing machine learning models into TVM. The frontends translate models from popular frameworks (PyTorch, ONNX, TensorFlow Lite) into Relax IR, enabling them to be optimized and compiled using TVM's compilation pipeline. Additionally, the Relax NNModule API provides a native PyTorch-like interface for building models directly in TVM.

---

## 6.1 Overview of Relax Frontends

### Why Frontends Matter

Frontends are the entry point to the TVM compilation stack. They bridge the gap between high-level framework representations (PyTorch graphs, ONNX protocols, TFLite flatbuffers) and TVM's internal Relax IR. A good frontend preserves the semantics of the original model while producing IR that is amenable to optimization.

### Available Frontends

| Frontend | Module | Source Format | Primary Use Case |
|----------|--------|---------------|------------------|
| PyTorch | `tvm.relax.frontend.torch` | PyTorch `ExportedProgram` | Research models, custom architectures |
| ONNX | `tvm.relax.frontend.onnx` | ONNX protobuf | Cross-framework model exchange |
| TFLite | `tvm.relax.frontend.tflite` | TFLite flatbuffer | Mobile/edge deployment |
| NNModule | `tvm.relax.frontend.nn` | Python API | Native TVM model definition |

### Common Import Pattern

Regardless of the frontend, the import process follows a common pattern:

```python
import tvm
from tvm import relax

# Step 1: Import the model using a frontend
mod, params = frontend_import(model, ...)

# Step 2: Apply the compilation pipeline
mod = relax.get_pipeline("zero")(mod)

# Step 3: Build and run
exec = relax.vm.build(mod, target="llvm")
vm = relax.VirtualMachine(exec, tvm.runtime.Device("cpu", 0))
result = vm["main"](input_data)
```

---

## 6.2 PyTorch Frontend

### 6.2.1 Overview

The PyTorch frontend imports models via `torch.export.export()`, which produces a `ExportedProgram` containing a traced computation graph. This is the recommended way to import PyTorch models into TVM, as it provides the most accurate representation of the model's computation.

```python
from tvm.relax.frontend.torch import from_exported_program
import torch
import torch.nn as nn
```

### 6.2.2 Basic Import

The simplest way to import a PyTorch model:

```python
import torch
import torch.nn as nn
from tvm.relax.frontend.torch import from_exported_program

# Define a simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Export the model
example_input = torch.randn(1, 784)
exported_program = torch.export.export(model, (example_input,))

# Import into TVM
mod, params = from_exported_program(
    exported_program,
    keep_params_as_input=False,
)

print(mod.script())
```

The resulting `mod` is a `tvm.IRModule` containing a Relax function that represents the model computation, and `params` is a dictionary of parameter tensors.

### 6.2.3 Import Options

#### keep_params_as_input

When `keep_params_as_input=True`, model parameters are kept as function input parameters rather than being embedded as constants in the IRModule. This is useful for:

- Models with very large parameters that should be stored externally.
- Scenarios where parameters may change at runtime (e.g., weight sharing).
- Enabling parameter transformations via `LiftTransformParams`.

```python
# Keep parameters as function inputs
mod, params = from_exported_program(
    exported_program,
    keep_params_as_input=True,
)

# The resulting function signature includes parameter inputs:
# def main(input: R.Tensor((1, 784), "float32"),
#          linear_weight: R.Tensor((256, 784), "float32"),
#          linear_bias: R.Tensor((256,), "float32"),
#          ...) -> R.Tensor((1, 10), "float32"):
```

#### unwrap_unit_return

When `unwrap_unit_return=True`, functions that return a unit tuple (empty tuple) will have their return type unwrapped. This is useful for models that return auxiliary outputs.

```python
mod, params = from_exported_program(
    exported_program,
    unwrap_unit_return=True,
)
```

#### relax_pipeline

The `relax_pipeline` option specifies a transformation pipeline to apply immediately after import. By default, a basic pipeline is applied. You can pass `None` to skip the pipeline or provide a custom pipeline.

```python
from tvm import relax

# No pipeline (raw import)
mod, params = from_exported_program(
    exported_program,
    relax_pipeline=None,
)

# Custom pipeline
custom_pipeline = relax.transform.Sequential([
    relax.transform.DecomposeOpsForInference(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),
])

mod, params = from_exported_program(
    exported_program,
    relax_pipeline=custom_pipeline,
)
```

### 6.2.4 Dynamic Shape Support

The PyTorch frontend supports dynamic shapes through `torch.export` dynamic shape annotations:

```python
import torch
import torch.nn as nn
from tvm.relax.frontend.torch import from_exported_program

class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        return self.linear(x)

model = DynamicModel()

# Define dynamic shapes using torch.export.dynamic_dim
example_input = torch.randn(1, 128)

# Export with dynamic batch dimension
exported_program = torch.export.export(
    model,
    (example_input,),
    dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}}
)

# Import with dynamic shapes preserved
mod, params = from_exported_program(exported_program)

# The resulting Relax function will have symbolic shape variables:
# def main(x: R.Tensor(("batch", 128), "float32"),
#          linear_weight: R.Tensor((10, 128), "float32"),
#          linear_bias: R.Tensor((10,), "float32")):
```

**Dynamic shape example with multiple dynamic dimensions:**

```python
class AttentionModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        attn_out, _ = self.attention(query, key, value)
        return self.linear(attn_out)

model = AttentionModel()

batch = torch.export.Dim("batch", min=1, max=32)
seq_len = torch.export.Dim("seq_len", min=1, max=512)

exported_program = torch.export.export(
    model,
    (torch.randn(1, 10, 64), torch.randn(1, 10, 64), torch.randn(1, 10, 64)),
    dynamic_shapes={
        "query": {0: batch, 1: seq_len},
        "key": {0: batch, 1: seq_len},
        "value": {0: batch, 1: seq_len},
    }
)

mod, params = from_exported_program(exported_program)
```

### 6.2.5 Custom Operator Converters

The PyTorch frontend provides a mechanism for registering custom operator converters. This allows you to handle PyTorch operators that are not natively supported by TVM.

#### Using convert_map

```python
from tvm.relax.frontend.torch import from_exported_program
from tvm import relax
import torch

# Define a custom converter for a specific operation
def convert_my_custom_op(ctx, node):
    """Convert a custom PyTorch operator to Relax operations.

    Args:
        ctx: The TorchImportContext, providing access to the BlockBuilder
        node: The FX graph node representing the operation

    Returns:
        A relax.Var representing the result of the operation
    """
    # Get the input tensors from the context
    input_tensor = ctx.get_tensor(node.args[0])

    # Build the computation using Relax operators
    result = relax.op.relu(input_tensor)
    return result

# Register the converter
convert_map = {
    "my_custom_op.default": convert_my_custom_op,
}

# Use the custom convert_map during import
mod, params = from_exported_program(
    exported_program,
    convert_map=convert_map,
)
```

#### Registering Converters via Decorator

```python
from tvm.relax.frontend.torch import register_torch_op

@register_torch_op("aten.my_custom_add")
def convert_my_custom_add(ctx, node):
    """Register a converter for aten.my_custom_add."""
    lhs = ctx.get_tensor(node.args[0])
    rhs = ctx.get_tensor(node.args[1])
    return ctx.builder.emit(relax.op.add(lhs, rhs))
```

#### Handling Unsupported Operators

When importing a model with unsupported operators, you have several options:

```python
# Option 1: Implement a custom converter (recommended)
mod, params = from_exported_program(
    exported_program,
    convert_map={"unsupported_op.default": my_converter},
)

# Option 2: Decompose the unsupported op in PyTorch before export
# Use torch.export to decompose complex ops
exported_program = torch.export.export(
    model,
    (example_input,),
    decompose_default=True,  # Decompose to simpler ops
)

# Option 3: Use PyTorch's graph manipulation to replace the op
class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        # Replace unsupported ops with supported alternatives
        return self.model(x)

wrapped_model = ModelWrapper(original_model)
exported_program = torch.export.export(wrapped_model, (example_input,))
```

### 6.2.6 Verification: Comparing TVM and PyTorch Outputs

After import, it is critical to verify that the TVM model produces the same results as the original PyTorch model:

```python
import numpy as np
import tvm
from tvm import relax

def verify_torch_model(exported_program, input_data, rtol=1e-5, atol=1e-5):
    """Verify that TVM output matches PyTorch output.

    Args:
        exported_program: The torch.export.ExportedProgram
        input_data: Tuple of input tensors (numpy arrays or torch tensors)
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    import torch

    # Get PyTorch reference output
    with torch.no_grad():
        if isinstance(input_data, torch.Tensor):
            torch_output = exported_program.module()(input_data)
        else:
            torch_output = exported_program.module()(*input_data)

    if isinstance(torch_output, torch.Tensor):
        torch_output = torch_output.numpy()
    else:
        torch_output = tuple(t.numpy() for t in torch_output)

    # Import and compile with TVM
    mod, params = from_exported_program(exported_program)
    mod = relax.get_pipeline("zero")(mod)

    # Build and run
    exec = relax.vm.build(mod, target="llvm")
    device = tvm.runtime.Device("cpu", 0)
    vm = relax.VirtualMachine(exec, device)

    # Convert inputs to TVM format
    tvm_inputs = []
    for inp in (input_data if isinstance(input_data, (tuple, list)) else (input_data,)):
        if isinstance(inp, torch.Tensor):
            tvm_inputs.append(tvm.nd.array(inp.numpy(), device))
        else:
            tvm_inputs.append(tvm.nd.array(inp, device))

    # Set parameters
    for name, param in params.items():
        vm.set_input(name, tvm.nd.array(param, device))

    # Run inference
    tvm_output = vm["main"](*tvm_inputs)

    # Compare outputs
    if isinstance(tvm_output, tvm.runtime.NDArray):
        tvm_output = tvm_output.numpy()
        np.testing.assert_allclose(torch_output, tvm_output, rtol=rtol, atol=atol)
    else:
        for i, (torch_out, tvm_out) in enumerate(zip(torch_output, tvm_output)):
            np.testing.assert_allclose(
                torch_out, tvm_out.numpy(), rtol=rtol, atol=atol,
                err_msg=f"Output {i} mismatch"
            )

    print("Verification passed! TVM output matches PyTorch output.")

# Usage
model = nn.Sequential(nn.Linear(784, 10))
example_input = torch.randn(1, 784)
exported_program = torch.export.export(model, (example_input,))
verify_torch_model(exported_program, example_input)
```

### 6.2.7 Complete PyTorch Import Example

```python
import torch
import torch.nn as nn
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Create and export the model
model = SimpleCNN(num_classes=10)
model.eval()

example_input = torch.randn(1, 3, 32, 32)
exported_program = torch.export.export(model, (example_input,))

# Import into TVM
mod, params = from_exported_program(
    exported_program,
    keep_params_as_input=True,
)

# Apply compilation pipeline
mod = relax.get_pipeline("zero")(mod)

# Build for CPU
exec = relax.vm.build(mod, target="llvm")
device = tvm.runtime.Device("cpu", 0)
vm = relax.VirtualMachine(exec, device)

# Run inference
input_data = tvm.nd.array(np.random.randn(1, 3, 32, 32).astype("float32"), device)
params_tvm = {k: tvm.nd.array(v, device) for k, v in params.items()}
output = vm["main"](input_data, **params_tvm)
print(f"Output shape: {output.shape}")
```

### 6.2.8 PyTorch Frontend: Advanced Patterns

#### Importing Models with Control Flow

```python
class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x, flag):
        if flag:
            h = self.linear1(x)
        else:
            h = self.linear2(x)
        return self.linear3(h)

# Export with specific branch (torch.export traces one path)
model = ConditionalModel()
model.eval()
exported_program = torch.export.export(
    model,
    (torch.randn(1, 128), torch.tensor(True)),
)

mod, params = from_exported_program(exported_program)
```

#### Importing Models with Multiple Outputs

```python
class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(256, 10)   # Classification
        self.head2 = nn.Linear(256, 1)    # Regression

    def forward(self, x):
        features = self.backbone(x)
        class_output = self.head1(features)
        reg_output = self.head2(features)
        return class_output, reg_output

model = MultiOutputModel()
model.eval()

exported_program = torch.export.export(
    model,
    (torch.randn(1, 784),),
)

mod, params = from_exported_program(exported_program)
# The Relax function returns a tuple of two tensors
```

#### Importing Transformer Models

```python
from transformers import BertModel, BertConfig
import torch
from tvm.relax.frontend.torch import from_exported_program

# Load a BERT model
config = BertConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
)
model = BertModel(config)
model.eval()

# Export with dynamic sequence length
seq_len = torch.export.Dim("seq_len", min=1, max=512)
batch_size = 1

exported_program = torch.export.export(
    model,
    (torch.randn(batch_size, 128, dtype=torch.long),),  # input_ids
    dynamic_shapes={
        "input_ids": {0: batch_size, 1: seq_len},
    }
)

# Import into TVM
mod, params = from_exported_program(exported_program)
print(f"Imported BERT model with {len(params)} parameters")
```

#### GPU Target Import and Compilation

```python
import torch
import torch.nn as nn
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# Define model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10),
)
model.eval()

example_input = torch.randn(1, 3, 224, 224)
exported_program = torch.export.export(model, (example_input,))

# Import into TVM
mod, params = from_exported_program(exported_program, keep_params_as_input=True)

# Apply CUDA-optimized pipeline
mod = relax.get_pipeline("static_shape_tuning")(mod)

# Build for CUDA
exec = relax.vm.build(mod, target="cuda")
device = tvm.runtime.Device("cuda", 0)
vm = relax.VirtualMachine(exec, device)

# Run inference on GPU
input_nd = tvm.nd.array(
    torch.randn(1, 3, 224, 224).numpy(), device
)
params_nd = {k: tvm.nd.array(v, device) for k, v in params.items()}
output = vm["main"](input_nd, **params_nd)
print(f"GPU output shape: {output.shape}")
```

---

## 6.3 ONNX Frontend

### 6.3.1 Overview

The ONNX frontend imports models in ONNX (Open Neural Network Exchange) format. ONNX provides a standardized representation for machine learning models that enables interoperability between different frameworks and tools.

```python
from tvm.relax.frontend.onnx import from_onnx
import onnx
```

### 6.3.2 Basic Import

```python
import onnx
from tvm.relax.frontend.onnx import from_onnx

# Load ONNX model
onnx_model = onnx.load("resnet50.onnx")

# Import into TVM with shape information
shape_info = {"input": (1, 3, 224, 224)}
mod, params = from_onnx(onnx_model, shape_info)

print(mod.script())
```

### 6.3.3 Shape and Dtype Customization

The `from_onnx` function accepts shape and dtype specifications to override the information in the ONNX model:

```python
# Specify shapes for all inputs
shape_dict = {
    "input_image": (1, 3, 224, 224),
    "input_mask": (1, 224),
}

# Specify dtypes (optional)
dtype_dict = {
    "input_image": "float32",
    "input_mask": "int64",
}

mod, params = from_onnx(
    onnx_model,
    shape_dict,
    dtype_dict=dtype_dict,
)
```

**Dynamic shapes with ONNX:**

ONNX models may have dynamic dimensions represented as strings (e.g., `"batch_size"`, `"seq_len"`). You can specify concrete values or use symbolic variables:

```python
# Option 1: Provide concrete shapes
shape_dict = {"input": (4, 3, 224, 224)}
mod, params = from_onnx(onnx_model, shape_dict)

# Option 2: Use TVM symbolic variables for dynamic shapes
import tvm
batch = tvm.tir.Var("batch", "int64")
seq_len = tvm.tir.Var("seq_len", "int64")
shape_dict = {
    "input_ids": (batch, seq_len),
    "attention_mask": (batch, seq_len),
}
mod, params = from_onnx(onnx_model, shape_dict)
```

### 6.3.4 Operator Converter Maps

The ONNX frontend uses a converter registry to map ONNX operators to Relax operations. You can customize this mapping to handle unsupported operators or override default conversions.

```python
from tvm.relax.frontend.onnx import from_onnx

# Define custom operator converter
def convert_my_custom_onnx_op(ctx, node, inputs):
    """Convert a custom ONNX operator.

    Args:
        ctx: The import context
        node: The ONNX node
        inputs: List of input expressions

    Returns:
        The converted Relax expression
    """
    from tvm import relax

    x = inputs[0]
    scale = float(node.attrs["scale"])

    result = relax.op.multiply(x, relax.const(scale))
    return result

# Create custom converter map
custom_convert_map = {
    "MyCustomOp": convert_my_custom_onnx_op,
    "MyScaleOp": convert_my_custom_onnx_op,
}

# Import with custom converters
mod, params = from_onnx(
    onnx_model,
    shape_dict,
    op_converter_map=custom_convert_map,
)
```

### 6.3.5 Handling Unsupported Operators

When encountering unsupported ONNX operators, several strategies are available:

**Strategy 1: Register a custom converter (recommended)**

```python
from tvm.relax.frontend.onnx import register_onnx_op

@register_onnx_op("MyUnsupportedOp")
def convert_unsupported_op(ctx, node, inputs):
    """Implement the unsupported operator using existing Relax ops."""
    x = inputs[0]
    alpha = float(node.attrs.get("alpha", 1.0))

    # Implement using existing operations
    pos = relax.op.multiply(x, relax.const(alpha))
    neg = relax.op.multiply(relax.op.negative(x), relax.const(alpha))
    return relax.op.where(relax.op.greater(x, relax.const(0.0)), pos, neg)
```

**Strategy 2: Pre-process the ONNX model to replace unsupported ops**

```python
import onnx
from onnx import helper, numpy_helper

def preprocess_onnx_model(model_path):
    """Pre-process an ONNX model to replace unsupported operations."""
    model = onnx.load(model_path)

    # Replace unsupported ops with supported equivalents
    new_nodes = []
    for node in model.graph.node:
        if node.op_type == "UnsupportedOp":
            # Replace with a sequence of supported ops
            new_nodes.extend(decompose_unsupported_op(node))
        else:
            new_nodes.append(node)

    # Update the graph
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    return model

def decompose_unsupported_op(node):
    """Decompose an unsupported ONNX op into supported ops."""
    # Example: decompose a complex op into add, mul, relu
    intermediate_name = node.output[0] + "_intermediate"
    return [
        helper.make_node("Add", [node.input[0], node.input[1]],
                         [intermediate_name], name=node.name + "_add"),
        helper.make_node("Relu", [intermediate_name],
                         [node.output[0]], name=node.name + "_relu"),
    ]

# Usage
preprocessed_model = preprocess_onnx_model("model.onnx")
mod, params = from_onnx(preprocessed_model, {"input": (1, 3, 224, 224)})
```

**Strategy 3: Use ONNX Runtime for verification and comparison**

```python
import onnxruntime as ort
import numpy as np
import tvm
from tvm import relax

def verify_onnx_import(onnx_path, shape_dict, rtol=1e-5, atol=1e-5):
    """Verify ONNX import by comparing TVM output with ONNX Runtime output."""
    # Load model
    onnx_model = onnx.load(onnx_path)

    # Get ONNX Runtime output
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    # Generate random input
    input_shape = shape_dict[list(shape_dict.keys())[0]]
    input_data = np.random.randn(*input_shape).astype("float32")

    ort_output = session.run(None, {input_name: input_data})[0]

    # Import and compile with TVM
    mod, params = from_onnx(onnx_model, shape_dict)
    mod = relax.get_pipeline("zero")(mod)

    exec = relax.vm.build(mod, target="llvm")
    device = tvm.runtime.Device("cpu", 0)
    vm = relax.VirtualMachine(exec, device)

    # Run TVM inference
    tvm_input = tvm.nd.array(input_data, device)
    for name, param in params.items():
        vm.set_input(name, tvm.nd.array(param.numpy() if hasattr(param, 'numpy') else param, device))

    tvm_output = vm["main"](tvm_input)

    # Compare
    np.testing.assert_allclose(ort_output, tvm_output.numpy(), rtol=rtol, atol=atol)
    print("ONNX import verification passed!")

# Usage
verify_onnx_import("resnet50.onnx", {"input": (1, 3, 224, 224)})
```

### 6.3.6 Complete ONNX Import Example

```python
import numpy as np
import onnx
from onnx import helper, TensorProto
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# Step 1: Create or load an ONNX model
def create_simple_onnx_model():
    """Create a simple ONNX model: y = relu(x * w + b)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 784])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [784, 10])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [10])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])

    matmul_node = helper.make_node("MatMul", ["X", "W"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "B"], ["add_out"])
    relu_node = helper.make_node("Relu", ["add_out"], ["Y"])

    graph = helper.make_graph(
        [matmul_node, add_node, relu_node],
        "simple_model",
        [X, W, B],
        [Y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model

onnx_model = create_simple_onnx_model()

# Step 2: Import into TVM
shape_dict = {"X": (1, 784), "W": (784, 10), "B": (10,)}
mod, params = from_onnx(onnx_model, shape_dict)

print("Imported ONNX model:")
print(mod.script())

# Step 3: Compile and run
mod = relax.get_pipeline("zero")(mod)
exec = relax.vm.build(mod, target="llvm")
device = tvm.runtime.Device("cpu", 0)
vm = relax.VirtualMachine(exec, device)

# Prepare inputs
x_data = np.random.randn(1, 784).astype("float32")
w_data = np.random.randn(784, 10).astype("float32")
b_data = np.random.randn(10).astype("float32")

x_nd = tvm.nd.array(x_data, device)
w_nd = tvm.nd.array(w_data, device)
b_nd = tvm.nd.array(b_data, device)

# Run
output = vm["main"](x_nd, w_nd, b_nd)
print(f"Output shape: {output.shape}")
print(f"Output (first 5): {output.numpy()[0, :5]}")
```

### 6.3.7 ONNX Import with Large Models

For large ONNX models (e.g., LLMs), memory-efficient loading is important:

```python
import onnx
from tvm.relax.frontend.onnx import from_onnx

# Load large ONNX model (saved with external data)
onnx_model = onnx.load("large_model.onnx", load_external_data=True)

# Or load with size limits
onnx_model = onnx.load("large_model.onnx")

# Import with shape information
shape_dict = {
    "input_ids": (1, 128),
    "attention_mask": (1, 128),
}
mod, params = from_onnx(onnx_model, shape_dict)
```

### 6.3.8 ONNX Opset Version Handling

Different ONNX opset versions may have different operator definitions. The frontend handles opset version differences automatically, but you should be aware of potential issues:

```python
# Check opset version
onnx_model = onnx.load("model.onnx")
opset_version = None
for opset in onnx_model.opset_import:
    if opset.domain == "":
        opset_version = opset.version
        break

print(f"ONNX opset version: {opset_version}")

# The TVM ONNX frontend supports opset versions 6-20
# If your model uses a newer opset, you may need to convert it
if opset_version and opset_version > 20:
    # Convert to a supported opset version
    from onnx import version_converter
    onnx_model = version_converter.convert_version(onnx_model, 20)
```

---

## 6.4 TensorFlow Lite Frontend

### 6.4.1 Overview

The TFLite frontend imports models in TensorFlow Lite format, which is commonly used for mobile and edge device deployment. The frontend converts TFLite operators into Relax IR.

```python
from tvm.relax.frontend.tflite import from_tflite
```

### 6.4.2 Basic Import

```python
import tflite
from tvm.relax.frontend.tflite import from_tflite

# Load TFLite model
with open("model.tflite", "rb") as f:
    tflite_model_bytes = f.read()

tflite_model = tflite.Model.GetRootAsModel(tflite_model_bytes, 0)

# Define input shapes and dtypes
shape_dict = {"input": (1, 224, 224, 3)}
dtype_dict = {"input": "float32"}

# Import into TVM
mod, params = from_tflite(
    tflite_model,
    shape_dict=shape_dict,
    dtype_dict=dtype_dict,
)

print(mod.script())
```

### 6.4.3 Shape and Dtype Dictionaries

The `shape_dict` and `dtype_dict` parameters specify the shapes and data types of the model's input tensors:

```python
# Multiple inputs
shape_dict = {
    "image": (1, 224, 224, 3),
    "mask": (1, 224, 224),
}

dtype_dict = {
    "image": "float32",
    "mask": "uint8",
}

mod, params = from_tflite(
    tflite_model,
    shape_dict=shape_dict,
    dtype_dict=dtype_dict,
)
```

**Finding input tensor names:**

If you are unsure of the input tensor names, you can inspect the TFLite model:

```python
import tflite

with open("model.tflite", "rb") as f:
    buf = f.read()

model = tflite.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

# List all input tensors
for i in range(subgraph.InputsLength()):
    tensor_idx = subgraph.Inputs(i)
    tensor = subgraph.Tensors(tensor_idx)
    name = tensor.Name().decode("utf-8")
    shape = tensor.ShapeAsNumpy()
    dtype = tensor.Type()
    print(f"Input {i}: name={name}, shape={shape}, dtype={dtype}")

# List all output tensors
for i in range(subgraph.OutputsLength()):
    tensor_idx = subgraph.Outputs(i)
    tensor = subgraph.Tensors(tensor_idx)
    name = tensor.Name().decode("utf-8")
    print(f"Output {i}: name={name}")
```

### 6.4.4 Custom Operator Converters

The TFLite frontend allows you to register custom converters for TFLite operators:

```python
from tvm.relax.frontend.tflite import from_tflite

# Define a custom converter
def convert_custom_tflite_op(op, tensors, op_inputs, op_outputs):
    """Convert a custom TFLite operator.

    Args:
        op: The TFLite operator
        tensors: The model's tensor list
        op_inputs: Input tensor indices
        op_outputs: Output tensor indices

    Returns:
        A Relax expression
    """
    from tvm import relax

    # Get input tensor data
    input_tensor = tensors[op_inputs[0]]

    # Build the computation
    result = relax.op.relu(input_tensor)
    return result

# Custom operator map
custom_op_map = {
    "CUSTOM": convert_custom_tflite_op,
}

mod, params = from_tflite(
    tflite_model,
    shape_dict=shape_dict,
    dtype_dict=dtype_dict,
    op_converter_map=custom_op_map,
)
```

### 6.4.5 Complete TFLite Import Example

```python
import numpy as np
import tvm
from tvm import relax

# Step 1: Load the TFLite model
import tflite

with open("mobilenet_v2.tflite", "rb") as f:
    tflite_bytes = f.read()

tflite_model = tflite.Model.GetRootAsModel(tflite_bytes, 0)

# Step 2: Define input specifications
shape_dict = {"input": (1, 224, 224, 3)}
dtype_dict = {"input": "float32"}

# Step 3: Import into TVM
from tvm.relax.frontend.tflite import from_tflite
mod, params = from_tflite(tflite_model, shape_dict, dtype_dict)

# Step 4: Compile
mod = relax.get_pipeline("zero")(mod)

# Build for CPU (typical for edge deployment)
exec = relax.vm.build(mod, target="llvm")
device = tvm.runtime.Device("cpu", 0)
vm = relax.VirtualMachine(exec, device)

# Step 5: Run inference
input_data = np.random.randn(1, 224, 224, 3).astype("float32")
input_nd = tvm.nd.array(input_data, device)

# Set parameters
for name, param in params.items():
    vm.set_input(name, tvm.nd.array(param.numpy() if hasattr(param, 'numpy') else param, device))

# Run
output = vm["main"](input_nd)
print(f"Output shape: {output.shape}")

# Get predicted class
predicted_class = np.argmax(output.numpy())
print(f"Predicted class: {predicted_class}")
```

### 6.4.6 TFLite Quantized Models

The TFLite frontend supports importing quantized (int8) models:

```python
# Import a quantized TFLite model
shape_dict = {"input": (1, 224, 224, 3)}
dtype_dict = {"input": "uint8"}  # Quantized input

mod, params = from_tflite(
    tflite_model,
    shape_dict=shape_dict,
    dtype_dict=dtype_dict,
)

# Apply dequantization passes if needed
# TVM will handle quantized operators appropriately
mod = relax.get_pipeline("zero")(mod)
```

### 6.4.7 TFLite Model Inspection Utility

```python
def inspect_tflite_model(model_path):
    """Print a summary of a TFLite model's structure."""
    import tflite

    with open(model_path, "rb") as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)

    print(f"Model version: {model.Version()}")
    print(f"Number of subgraphs: {model.SubgraphsLength()}")
    print(f"Description: {model.Description()}")

    for sg_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(sg_idx)
        print(f"\n--- Subgraph {sg_idx} ---")
        print(f"  Name: {subgraph.Name()}")
        print(f"  Tensors: {subgraph.TensorsLength()}")
        print(f"  Operators: {subgraph.OperatorsLength()}")

        # Print inputs
        print(f"  Inputs ({subgraph.InputsLength()}):")
        for i in range(subgraph.InputsLength()):
            idx = subgraph.Inputs(i)
            tensor = subgraph.Tensors(idx)
            name = tensor.Name().decode("utf-8") if tensor.Name() else f"tensor_{idx}"
            shape = tensor.ShapeAsNumpy() if tensor.ShapeAsNumpy() is not None else "dynamic"
            print(f"    [{i}] {name}: shape={shape}, type={tensor.Type()}")

        # Print outputs
        print(f"  Outputs ({subgraph.OutputsLength()}):")
        for i in range(subgraph.OutputsLength()):
            idx = subgraph.Outputs(i)
            tensor = subgraph.Tensors(idx)
            name = tensor.Name().decode("utf-8") if tensor.Name() else f"tensor_{idx}"
            print(f"    [{i}] {name}")

        # Print operators
        print(f"  Operators ({subgraph.OperatorsLength()}):")
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode = model.OperatorCodes(op.OpcodeIndex())
            op_type = opcode.BuiltinCode()
            print(f"    [{i}] {op_type}")

# Usage
inspect_tflite_model("model.tflite")
```

---

## 6.5 Relax NNModule

### 6.5.1 Overview

The Relax NNModule (`tvm.relax.frontend.nn`) provides a PyTorch-like API for defining models directly in TVM. This eliminates the need for an external framework and provides tight integration with the Relax compilation pipeline.

```python
from tvm.relax.frontend import nn
```

### 6.5.2 Available Layers

The NNModule provides the following layer classes:

| Layer | Description | Equivalent PyTorch |
|-------|-------------|---------------------|
| `nn.Linear` | Fully connected layer | `torch.nn.Linear` |
| `nn.Conv1d` | 1D convolution | `torch.nn.Conv1d` |
| `nn.Conv2d` | 2D convolution | `torch.nn.Conv2d` |
| `nn.Conv3d` | 3D convolution | `torch.nn.Conv3d` |
| `nn.BatchNorm` | Batch normalization | `torch.nn.BatchNorm1d/2d` |
| `nn.LayerNorm` | Layer normalization | `torch.nn.LayerNorm` |
| `nn.RMSNorm` | Root mean square normalization | Custom |
| `nn.Embedding` | Embedding lookup | `torch.nn.Embedding` |
| `nn.MultiheadAttention` | Multi-head attention | `torch.nn.MultiheadAttention` |
| `nn.GroupNorm` | Group normalization | `torch.nn.GroupNorm` |
| `nn.Dropout` | Dropout (identity in inference) | `torch.nn.Dropout` |
| `nn.SiLU` | Sigmoid Linear Unit | `torch.nn.SiLU` |
| `nn.GELU` | Gaussian Error Linear Unit | `torch.nn.GELU` |
| `nn.Tanh` | Hyperbolic tangent | `torch.nn.Tanh` |
| `nn.ReLU` | Rectified Linear Unit | `torch.nn.ReLU` |
| `nn.Softmax` | Softmax activation | `torch.nn.Softmax` |
| `nn.ModuleList` | List of modules | `torch.nn.ModuleList` |

### 6.5.3 Basic Model Definition

```python
from tvm.relax.frontend import nn

class SimpleClassifier(nn.Module):
    def __init__(self, in_features=784, hidden=256, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x: nn.Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Export to TVM IRModule
spec = {"input": {"shape": [1, 784], "dtype": "float32"}}
mod, params = SimpleClassifier.export_tvm(spec)

print(mod.script())
```

### 6.5.4 Spec Module for Shape and Dtype

The `spec` argument to `export_tvm` defines the shape and dtype of each input. This is required because NNModule does not perform automatic shape inference during definition.

```python
# Single input
spec = {
    "input": {
        "shape": [1, 784],
        "dtype": "float32",
    }
}

# Multiple inputs
spec = {
    "input_ids": {
        "shape": [1, 128],
        "dtype": "int64",
    },
    "attention_mask": {
        "shape": [1, 128],
        "dtype": "float32",
    },
}

# Dynamic shapes using symbolic variables
import tvm
batch = tvm.tir.Var("batch", "int64")
seq_len = tvm.tir.Var("seq_len", "int64")

spec = {
    "input_ids": {
        "shape": [batch, seq_len],
        "dtype": "int64",
    },
    "attention_mask": {
        "shape": [batch, seq_len],
        "dtype": "float32",
    },
}
```

### 6.5.5 Convolutional Network Example

```python
from tvm.relax.frontend import nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.pool = nn.max_pool2d
        self.flatten = nn.flatten
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x: nn.Tensor):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.pool(x, kernel_size=[2, 2], stride=[2, 2])

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.relu(x)
        x = self.pool(x, kernel_size=[2, 2], stride=[2, 2])

        # Classification head
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Export
spec = {"input": {"shape": [1, 3, 32, 32], "dtype": "float32"}}
mod, params = ConvNet.export_tvm(spec)
```

### 6.5.6 Transformer Model Example

```python
from tvm.relax.frontend import nn
import tvm

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: nn.Tensor):
        # Self-attention with residual
        attn_out = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=256, num_heads=4,
                 ff_dim=512, num_layers=4, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: nn.Tensor):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        # Global average pooling
        x = nn.mean(x, axis=1)
        x = self.classifier(x)
        return x

# Export with dynamic sequence length
seq_len = tvm.tir.Var("seq_len", "int64")
spec = {"input_ids": {"shape": [1, seq_len], "dtype": "int64"}}
mod, params = Transformer.export_tvm(spec)
```

### 6.5.7 Parameter Management

NNModule provides tools for managing model parameters:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 10)
        self.embedding = nn.Embedding(1000, 128)

    def forward(self, x: nn.Tensor):
        return self.linear(x)

model = MyModel()

# Access parameters by name
for name, param in model.parameters():
    print(f"Parameter: {name}, shape: {param.shape}, dtype: {param.dtype}")

# Export parameters separately
spec = {"input": {"shape": [1, 128], "dtype": "float32"}}
mod, params = model.export_tvm(spec)

# Parameters are stored as a dictionary of numpy arrays
for name, param in params.items():
    print(f"  {name}: shape={param.shape}, dtype={param.dtype}")
```

**Pre-trained weight loading:**

```python
import numpy as np

# Initialize model
model = MyModel()

# Load pre-trained weights
pretrained_weights = {
    "linear.weight": np.random.randn(10, 128).astype("float32"),
    "linear.bias": np.random.randn(10).astype("float32"),
    "embedding.weight": np.random.randn(1000, 128).astype("float32"),
}

# Set weights
for name, weight in pretrained_weights.items():
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    getattr(module, parts[-1]).data = weight

# Export with pre-trained weights
spec = {"input": {"shape": [1, 128], "dtype": "float32"}}
mod, params = model.export_tvm(spec)
```

### 6.5.8 Functional Operations

NNModule also provides functional-style operations that can be used in the `forward` method:

```python
from tvm.relax.frontend import nn

class FunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x: nn.Tensor):
        # Functional operations
        x = self.conv(x)
        x = nn.relu(x)              # Functional activation
        x = nn.max_pool2d(x, kernel_size=[2, 2], stride=[2, 2])
        x = nn.flatten(x)            # Functional flatten
        x = self.fc(x)
        x = nn.softmax(x, axis=-1)   # Functional softmax
        return x
```

**Available functional operations:**

| Function | Description |
|----------|-------------|
| `nn.relu(x)` | ReLU activation |
| `nn.silu(x)` | SiLU activation |
| `nn.gelu(x)` | GELU activation |
| `nn.tanh(x)` | Tanh activation |
| `nn.softmax(x, axis)` | Softmax |
| `nn.sigmoid(x)` | Sigmoid |
| `nn.flatten(x)` | Flatten |
| `nn.max_pool2d(x, ...)` | Max pooling 2D |
| `nn.avg_pool2d(x, ...)` | Average pooling 2D |
| `nn.interpolate(x, ...)` | Interpolation (upsample) |
| `nn.mean(x, axis)` | Mean reduction |
| `nn.sum(x, axis)` | Sum reduction |
| `nn.reshape(x, shape)` | Reshape |
| `nn.transpose(x, axes)` | Transpose |
| `nn.matmul(x, y)` | Matrix multiplication |
| `nn.add(x, y)` | Element-wise add |
| `nn.multiply(x, y)` | Element-wise multiply |
| `nn.concatenate(tensors, axis)` | Concatenation |

### 6.5.9 Complete NNModule Example: Training and Inference

```python
from tvm.relax.frontend import nn
import tvm
from tvm import relax
import numpy as np

# Step 1: Define the model
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: nn.Tensor):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 2: Export to TVM
spec = {"input": {"shape": [1, 784], "dtype": "float32"}}
mod, params = MLP.export_tvm(spec)

# Step 3: Compile
mod = relax.get_pipeline("zero")(mod)
exec = relax.vm.build(mod, target="llvm")
device = tvm.runtime.Device("cpu", 0)
vm = relax.VirtualMachine(exec, device)

# Step 4: Run inference
input_data = np.random.randn(1, 784).astype("float32")
input_nd = tvm.nd.array(input_data, device)

# Set parameters
params_tvm = {k: tvm.nd.array(v, device) for k, v in params.items()}

output = vm["main"](input_nd, **params_tvm)
print(f"Output shape: {output.shape}")
print(f"Predicted class: {np.argmax(output.numpy())}")
```

---

## 6.6 Comparison of Frontends

### 6.6.1 Feature Comparison Table

| Feature | PyTorch | ONNX | TFLite | NNModule |
|---------|---------|------|--------|----------|
| Dynamic shapes | Yes | Limited | No | Yes |
| Custom ops | Yes | Yes | Yes | N/A (native) |
| Ease of use | High | Medium | Medium | High |
| Quantized models | Partial | Yes | Yes | Partial |
| Control flow | Partial | Limited | No | Limited |
| Multiple outputs | Yes | Yes | Yes | Yes |
| Training mode | Yes | No | No | Partial |
| External data | Yes | Yes | Yes | N/A |
| Opset coverage | High | High | Medium | Full |

### 6.6.2 Choosing the Right Frontend

**Use PyTorch when:**

- You are working with a PyTorch model directly.
- You need dynamic shape support.
- You want the most comprehensive operator coverage.
- You need to verify results against PyTorch easily.

**Use ONNX when:**

- Your model is already in ONNX format.
- You need cross-framework compatibility.
- You are working with models exported from non-PyTorch frameworks.
- You need standardized operator definitions.

**Use TFLite when:**

- Your model is already in TFLite format.
- You are targeting mobile or edge deployment.
- You need quantized model support.
- You are working with models from TensorFlow/Keras.

**Use NNModule when:**

- You want to define models directly in TVM.
- You need tight control over the generated IR.
- You are building TVM-specific optimizations.
- You want to avoid framework dependencies.

### 6.6.3 Frontend Performance Considerations

The choice of frontend can affect the quality of the generated code:

```python
# Different frontends may produce different IR for the same model.
# Compare the number of operators and fusion opportunities:

# PyTorch import (typically produces cleaner IR)
mod_torch, params_torch = from_exported_program(exported_program)

# ONNX import (may have additional reshape/transpose operations)
mod_onnx, params_onnx = from_onnx(onnx_model, shape_dict)

# Check IR complexity
print(f"PyTorch: {len(mod_torch.functions)} functions")
print(f"ONNX: {len(mod_onnx.functions)} functions")

# After optimization, both should converge to similar code
mod_torch = relax.get_pipeline("zero")(mod_torch)
mod_onnx = relax.get_pipeline("zero")(mod_onnx)
```

---

## 6.7 Cross-Frontend Verification

### 6.7.1 Comparing PyTorch and ONNX Outputs

When a model is available in both PyTorch and ONNX formats, you can verify that both frontends produce equivalent results:

```python
import torch
import torch.nn as nn
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.frontend.onnx import from_onnx

def compare_frontends(pytorch_model, onnx_path, input_shape, rtol=1e-4, atol=1e-4):
    """Compare PyTorch and ONNX frontend imports."""
    # Generate reference output
    pytorch_model.eval()
    with torch.no_grad():
        input_tensor = torch.randn(*input_shape)
        torch_output = pytorch_model(input_tensor).numpy()

    # Import via PyTorch frontend
    exported_program = torch.export.export(pytorch_model, (input_tensor,))
    mod_torch, params_torch = from_exported_program(exported_program)
    mod_torch = relax.get_pipeline("zero")(mod_torch)

    # Import via ONNX frontend
    import onnx
    onnx_model = onnx.load(onnx_path)
    shape_dict = {"input": input_shape}
    mod_onnx, params_onnx = from_onnx(onnx_model, shape_dict)
    mod_onnx = relax.get_pipeline("zero")(mod_onnx)

    # Build and run both
    device = tvm.runtime.Device("cpu", 0)

    # PyTorch frontend result
    exec_torch = relax.vm.build(mod_torch, target="llvm")
    vm_torch = relax.VirtualMachine(exec_torch, device)
    input_nd = tvm.nd.array(input_tensor.numpy(), device)
    params_torch_nd = {k: tvm.nd.array(v, device) for k, v in params_torch.items()}
    output_torch = vm_torch["main"](input_nd, **params_torch_nd).numpy()

    # ONNX frontend result
    exec_onnx = relax.vm.build(mod_onnx, target="llvm")
    vm_onnx = relax.VirtualMachine(exec_onnx, device)
    input_nd2 = tvm.nd.array(input_tensor.numpy(), device)
    params_onnx_nd = {k: tvm.nd.array(v, device) for k, v in params_onnx.items()}
    output_onnx = vm_onnx["main"](input_nd2, **params_onnx_nd).numpy()

    # Compare all three
    np.testing.assert_allclose(torch_output, output_torch, rtol=rtol, atol=atol)
    np.testing.assert_allclose(torch_output, output_onnx, rtol=rtol, atol=atol)
    np.testing.assert_allclose(output_torch, output_onnx, rtol=rtol, atol=atol)

    print("All three outputs match!")
    return output_torch, output_onnx, torch_output
```

### 6.7.2 Verification Utilities

```python
import numpy as np
import tvm
from tvm import relax

def verify_model_output(mod, params, input_data, expected_output,
                        target="llvm", rtol=1e-5, atol=1e-5):
    """Generic verification utility for any frontend.

    Args:
        mod: TVM IRModule
        params: Parameter dictionary
        input_data: Dict of input name -> numpy array
        expected_output: Expected numpy array output
        target: Compilation target
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    # Apply pipeline
    mod = relax.get_pipeline("zero")(mod)

    # Build
    exec = relax.vm.build(mod, target=target)
    device = tvm.runtime.Device(target.split()[0], 0)
    vm = relax.VirtualMachine(exec, device)

    # Prepare inputs
    inputs = {}
    for name, data in input_data.items():
        inputs[name] = tvm.nd.array(data, device)

    # Set parameters
    for name, param in params.items():
        arr = param.numpy() if hasattr(param, 'numpy') else param
        inputs[name] = tvm.nd.array(arr, device)

    # Run
    output = vm["main"](**inputs)

    # Compare
    if isinstance(output, tvm.runtime.NDArray):
        output_np = output.numpy()
    else:
        output_np = output

    np.testing.assert_allclose(expected_output, output_np, rtol=rtol, atol=atol)
    print("Verification passed!")
    return output_np
```

---

## 6.8 Advanced Frontend Patterns

### 6.8.1 Multi-Framework Pipeline

In some cases, you may want to convert a model through multiple frameworks:

```python
# PyTorch -> ONNX -> TVM
import torch
import torch.nn as nn
import onnx
from tvm.relax.frontend.onnx import from_onnx

model = nn.Sequential(nn.Linear(784, 10))
model.eval()

# Step 1: Export PyTorch to ONNX
example_input = torch.randn(1, 784)
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)

# Step 2: Import ONNX into TVM
onnx_model = onnx.load("model.onnx")
shape_dict = {"input": (1, 784)}
mod, params = from_onnx(onnx_model, shape_dict)
```

### 6.8.2 Model Ensembling

Import multiple models and combine them:

```python
import torch
import torch.nn as nn
from tvm.relax.frontend.torch import from_exported_program
from tvm import relax

# Model A: Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))

# Model B: Classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))

# Export both models
feature_model = FeatureExtractor()
classifier_model = Classifier()

feature_ep = torch.export.export(feature_model, (torch.randn(1, 3, 32, 32),))
classifier_ep = torch.export.export(classifier_model, (torch.randn(1, 64, 1, 1),))

mod_feat, params_feat = from_exported_program(feature_ep, keep_params_as_input=True)
mod_cls, params_cls = from_exported_program(classifier_ep, keep_params_as_input=True)

# Both are separate IRModules that can be compiled independently
# Or manually combined into a single IRModule
```

### 6.8.3 Handling Large Language Models

Importing large language models (LLMs) requires special consideration for memory efficiency and parameter management:

```python
import torch
from tvm.relax.frontend.torch import from_exported_program

# For LLMs, use keep_params_as_input to avoid embedding large weights in the IR
mod, params = from_exported_program(
    exported_llm,
    keep_params_as_input=True,
)

# Apply parameter lifting to separate parameter transformation
mod = relax.transform.LiftTransformParams()(mod)

# Bundle parameters for efficient loading
mod = relax.transform.BundleModelParams()(mod)

# Compile with memory planning
mod = relax.transform.StaticPlanBlockMemory()(mod)
```

### 6.8.4 Import-Time Optimization Hooks

You can apply optimizations at import time by using custom pipelines:

```python
from tvm.relax.frontend.torch import from_exported_program
from tvm import relax

# Define an import-time optimization pipeline
import_pipeline = relax.transform.Sequential([
    relax.transform.DecomposeOpsForInference(),
    relax.transform.CanonicalizeBindings(),
    relax.transform.FoldConstant(),
    relax.transform.SimplifyExpr(),
    relax.transform.FoldConstant(),
])

# Import with immediate optimization
mod, params = from_exported_program(
    exported_program,
    keep_params_as_input=True,
    relax_pipeline=import_pipeline,
)

# The resulting mod is already partially optimized
```

---

## 6.9 Troubleshooting Frontend Issues

### 6.9.1 Common Import Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Unsupported operator: aten.xxx` | PyTorch op not supported | Register custom converter or decompose in PyTorch |
| `Shape mismatch` | Incorrect shape specification | Check model input shapes and provide correct `shape_dict` |
| `Type mismatch` | Incorrect dtype specification | Verify input dtypes match model expectations |
| `Cannot infer shape` | Dynamic shapes not specified | Provide concrete shapes or symbolic variables |
| `ONNX opset version not supported` | Old or new opset version | Convert ONNX model to supported opset version |
| `TFLite model parsing error` | Corrupt or incompatible model | Verify TFLite model with `flatc` tool |

### 6.9.2 Debugging Import

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use TVM's debug mode
with tvm.transform.PassContext(config={"relax.frontend.debug": True}):
    mod, params = from_exported_program(exported_program)

# Print the imported IR for inspection
print(mod.script())

# Check the number of functions and parameters
print(f"Functions: {list(mod.functions.keys())}")
print(f"Parameters: {list(params.keys())}")

# Verify each parameter's shape and dtype
for name, param in params.items():
    arr = param.numpy() if hasattr(param, 'numpy') else param
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
```

### 6.9.3 Operator Coverage Check

To check which operators are used by a model and whether they are supported:

```python
import torch
import torch.nn as nn

def list_model_ops(model, example_input):
    """List all operations used by a PyTorch model."""
    ops = set()

    class OpTracer(torch.nn.Module):
        def forward(self, *args):
            return torch.jit.trace(model, example_input)(*args)

    # Use torch.export to get the graph
    exported = torch.export.export(model, (example_input,))

    # Extract operations from the exported graph
    for node in exported.graph.nodes():
        ops.add(node.target)

    return ops

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
ops = list_model_ops(model, torch.randn(1, 784))
print(f"Operations used: {sorted(ops)}")
```

---

## 6.10 Quick Reference: Frontend APIs

### PyTorch Frontend

```python
from tvm.relax.frontend.torch import from_exported_program

mod, params = from_exported_program(
    exported_program,         # torch.export.ExportedProgram
    keep_params_as_input=False,  # Keep params as function inputs
    unwrap_unit_return=False,    # Unwrap unit tuple returns
    relax_pipeline=None,         # Custom pipeline or None
    convert_map=None,            # Custom op converters
)
```

### ONNX Frontend

```python
from tvm.relax.frontend.onnx import from_onnx

mod, params = from_onnx(
    onnx_model,              # onnx.ModelProto
    shape_dict,              # Dict[str, Tuple[int, ...]]
    dtype_dict=None,         # Dict[str, str]
    op_converter_map=None,   # Custom op converters
)
```

### TFLite Frontend

```python
from tvm.relax.frontend.tflite import from_tflite

mod, params = from_tflite(
    tflite_model,            # tflite.Model
    shape_dict,              # Dict[str, Tuple[int, ...]]
    dtype_dict=None,         # Dict[str, str]
    op_converter_map=None,   # Custom op converters
)
```

### NNModule

```python
from tvm.relax.frontend import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        pass

    def forward(self, x: nn.Tensor):
        # Define computation
        return x

spec = {"input": {"shape": [1, 784], "dtype": "float32"}}
mod, params = MyModel.export_tvm(spec)
```
