# ONNX Export

PyTorch's ONNX export converts models to ONNX format for cross-framework interoperability and deployment.

```python
import torch
import torch.onnx
```

---

## torch.onnx.export

```python
torch.onnx.export(
    model: Union[nn.Module, torch.jit.ScriptModule],
    args: Tuple[Any, ...],           # example inputs
    f: Union[str, BytesIO],          # output file path or buffer
    export_params: bool = True,      # include trained weights
    verbose: bool = False,           # print human-readable graph
    training: torch.onnx.TrainingMode = TrainingMode.EVAL,
    input_names: List[str] = None,   # input tensor names
    output_names: List[str] = None,  # output tensor names
    opset_version: int = 17,         # ONNX opset version
    do_constant_folding: bool = True,
    dynamic_axes: Union[Dict, None] = None,
    keep_initializers_as_inputs: bool = None,
    custom_opsets: Dict[str, int] = None,
)
```

### Basic Export

```python
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
    def forward(self, x):
        return self.linear(x)

model = Model()
model.eval()

torch.onnx.export(
    model,
    (torch.randn(1, 784),),
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)
```

---

## Opset Versions

| Opset | PyTorch | Key Additions |
|-------|---------|---------------|
| 9 | 1.2+ | Broadcast, unfold |
| 11 | 1.6+ | If/Loop/Scan, gather |
| 13 | 1.9+ | Squeeze/Unsqueeze changes |
| 14 | 1.10+ | Reshape, broadcast rules |
| 15 | 1.11+ | Multi-output ops |
| 16 | 1.12+ | QuantizeLinear changes |
| 17 | 2.0+ | LayerNormalization, SkipSimplifiedLayerNormalization |
| 18 | 2.1+ | Optional type support |
| 20 | 2.2+ | GroupNorm, improved control flow |

```python
torch.onnx.export(model, args, "model.onnx", opset_version=17)
```

---

## Dynamic Axes

Specify which input dimensions are dynamic (can change at runtime).

```python
# Single dynamic dimension
torch.onnx.export(
    model,
    (torch.randn(1, 784),),
    "model.onnx",
    dynamic_axes={"input": {0: "batch_size"}},
)

# Multiple dynamic dims
torch.onnx.export(
    model,
    (torch.randn(1, 10, 20),),
    "model.onnx",
    input_names=["x"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "batch", 2: "seq_len"},
        "y": {0: "batch", 1: "seq_len"},
    },
)

# All dims dynamic
torch.onnx.export(
    model, args, "model.onnx",
    dynamic_axes={"input": {}, "output": {}},
)
```

---

## FX-Based Exporter (TorchDynamo)

The newer FX-based exporter uses TorchDynamo for more accurate tracing.

```python
# FX-based export (recommended for PyTorch 2.x)
torch.onnx.export(
    model,
    args,
    "model.onnx",
    dynamo=True,
)

# Export from ExportedProgram
from torch.export import export
exported = export(model, args)
torch.onnx.export_from_exported_program(exported, "model.onnx")
```

---

## Custom Op Registration

```python
from torch.onnx import register_custom_op_symbolic

def my_op_symbolic(g, input, scale):
    return g.op("CustomDomain::MyOp", input, scale_f=scale)

register_custom_op_symbolic("my_namespace::my_op", my_op_symbolic, opset_version=17)
```

### Custom Op with Autograd Function

```python
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        return x * scale
    @staticmethod
    def symbolic(g, x, scale):
        return g.op("CustomDomain::ScaledMul", x, scale_f=scale)
```

---

## Export Modes

```python
from torch.onnx import TrainingMode

# Inference mode (default)
torch.onnx.export(model, args, "model.onnx", training=TrainingMode.EVAL)

# Training mode (preserves dropout, batchnorm in training mode)
torch.onnx.export(model, args, "model.onnx", training=TrainingMode.TRAINING)
```

---

## Verifying Exported Models

```python
import onnx
import onnxruntime as ort
import numpy as np

# Check model validity
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Print model info
print(onnx.helper.printable_graph(onnx_model.graph))

# Run inference with ONNX Runtime
session = ort.InferenceSession("model.onnx")
inputs = {session.get_inputs()[0].name: np.random.randn(1, 784).astype(np.float32)}
outputs = session.run(None, inputs)

# Compare with PyTorch
with torch.no_grad():
    pt_output = model(torch.randn(1, 784)).numpy()
np.testing.assert_allclose(pt_output, outputs[0], rtol=1e-3, atol=1e-5)
```

---

## Common Issues and Solutions

### Unsupported Ops

```python
# Register custom symbolic function
def unsqueeze_symbolic(g, self, dim):
    return g.op("Unsqueeze", self, dim_i=dim)

register_custom_op_symbolic("aten::unsqueeze", unsqueeze_symbolic, 9)
```

### Trace vs Script

```python
# Tracing: cannot capture data-dependent control flow
torch.onnx.export(model, args, "model.onnx")

# Scripting: captures control flow
scripted = torch.jit.script(model)
torch.onnx.export(scripted, args, "model.onnx")
```

### Constant Folding

```python
# Enable constant folding for smaller model
torch.onnx.export(model, args, "model.onnx", do_constant_folding=True)

# Disable to preserve all constants
torch.onnx.export(model, args, "model.onnx", do_constant_folding=False)
```

---

## Complete ResNet Export Example

```python
import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 56 * 56, 1000)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

model = ResNet()
model.eval()

# Export with dynamic batch
torch.onnx.export(
    model,
    (torch.randn(1, 3, 224, 224),),
    "resnet.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    do_constant_folding=True,
)

# Verify
import onnx
onnx.checker.check_model(onnx.load("resnet.onnx"))
print("Export verified successfully")
```

---

## Environment Variables

```bash
# Debug export
TORCH_ONNX_DEBUG=1
TORCH_ONNX_WONDERFUL_HARNESS=1

# Control export behavior
TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK=1
```
