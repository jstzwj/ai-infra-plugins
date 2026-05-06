# torch.export

`torch.export` captures a PyTorch model as a standardized, serializable graph for downstream compilers and deployment.

```python
import torch
from torch.export import export, ExportedProgram, Dim
```

---

## torch.export.export

```python
exported: ExportedProgram = torch.export.export(
    f: Callable,                        # model or function
    args: Tuple[Any, ...],              # example positional inputs
    kwargs: Dict[str, Any] = None,      # example keyword inputs
    *,
    dynamic_shapes: Union[Dict, Tuple] = None,  # dynamic shape spec
    strict: bool = True,                # error on untraceable ops
    preserve_module_call_signature: Tuple[str] = (),
)
```

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
exported = export(model, (torch.randn(3, 10),))
print(exported)
```

---

## ExportedProgram

```python
# Core properties
exported.graph_module          # GraphModule with the exported graph
exported.graph                 # FX Graph
exported.range_constraints     # dynamic shape constraints
exported.graph_signature       # input/output signature metadata

# Access parameters and buffers
for name, param in exported.named_parameters():
    print(name, param.shape)
for name, buf in exported.named_buffers():
    print(name, buf.shape)

# Run the exported program
output = exported.module()(input_tensor)
# Or directly
output = exported(input_tensor)
```

---

## Dynamic Shapes with Dim

```python
from torch.export import Dim

# Define dynamic dimensions
batch = Dim("batch", min=1, max=64)
seq_len = Dim("seq_len", min=1, max=512)

# Specify dynamic dimensions for inputs
exported = export(
    model,
    (torch.randn(3, 10, 20),),
    dynamic_shapes=({0: batch, 1: Dim("features")},),
)

# With kwargs
exported = export(
    model,
    args=(),
    kwargs={"x": torch.randn(3, 10)},
    dynamic_shapes={"x": {0: batch}},
)
```

### Dim Options

```python
Dim(name)                  # Unbounded dynamic dimension
Dim(name, min=1)           # Minimum bound
Dim(name, max=1024)        # Maximum bound
Dim(name, min=1, max=1024) # Both bounds
```

### Derived Dimensions

```python
d = Dim("d")
d2 = d * 2  # derived dimension: output is twice input
```

---

## Save and Load

```python
# Save
torch.export.save(exported, "model.pt2")

# Load
loaded = torch.export.load("model.pt2")
output = loaded(input_tensor)

# Also works with torch.save/torch.load (less recommended)
torch.save(exported, "model.pt")
loaded = torch.load("model.pt")
```

---

## Functionalization

`torch.export` functionalizes the graph -- converting in-place ops and mutations to pure functional form.

- In-place ops like `x.copy_(y)` are converted to functional equivalents
- View ops are preserved as-is
- Mutations on inputs are captured in the graph signature

---

## Custom Decompositions

Control how higher-level ops are decomposed into lower-level ops.

```python
from torch.export import register_decomposition

@register_decomposition(torch.ops.aten.some_op)
def my_decomposition(*args, **kwargs):
    # Custom implementation using simpler ops
    return result

exported = export(model, args)
```

Or use the default decompositions:

```python
decompositions = torch.export.default_decompositions()
exported = export(model, args, decompositions=decompositions)
```

---

## Non-Strict Export

```python
# Non-strict: allows more Python features (experimental)
torch._export.non_strict_export(model, args, kwargs, dynamic_shapes)

# Or use strict=False to allow untraceable ops (less portable)
exported = export(model, args, strict=False)
```

---

## Integration with Downstream Compilers

### ExecuTorch (Edge Devices)

```python
from executorch.exir import to_edge

exported = export(model, args)
edge_program = to_edge(exported)
# Further compile for edge devices (mobile, embedded)
```

### AOTInductor (C++ Deployment)

```python
from torch._inductor import aot_compile

so_path = aot_compile(
    exported,
    example_inputs=(torch.randn(1, 10),),
)
# Produces a shared object loadable in C++
```

### ONNX

```python
torch.onnx.export_from_exported_program(exported, "model.onnx")
```

### TensorRT

```python
# Use torch_tensorrt after export
import torch_tensorrt
trt_model = torch_tensorrt.compile(exported, ...)
```

---

## Inspecting the Exported Graph

```python
# Print the graph IR
print(exported.graph)

# Print as readable Python code
exported.graph_module.print_readable()

# Print tabular format
exported.graph.print_tabular()

# List all nodes
for node in exported.graph.nodes:
    print(f"{node.op}: {node.target} args={node.args}")

# View graph signature
print(exported.graph_signature)
```

---

## Common Patterns

### Handling Non-Tensor Arguments

```python
class Model(torch.nn.Module):
    def forward(self, x, scale: float):
        return x * scale

# Use concrete args for non-tensor inputs
exported = export(model, (torch.randn(1, 10), 2.0))
```

### Data-Dependent Control Flow

```python
# Data-dependent control flow requires restructuring
# Bad: if x.sum() > threshold: ...
# Good: use torch.where or torch.cond

from torch._higher_order_ops.cond import cond

class Model(torch.nn.Module):
    def forward(self, x, flag: bool):
        if flag:
            return x * 2
        else:
            return x + 1

# Must trace with concrete flag
exported = export(model, (torch.randn(1, 10), True))
```

---

## Troubleshooting

```python
# Use strict=False for lenient tracing
exported = export(model, args, strict=False)

# Debug with FX trace
gm = torch.fx.symbolic_trace(model)
print(gm.graph)

# Check for graph breaks
import torch._dynamo
torch._dynamo.explain(model, *args)
```

---

## Complete Example

```python
import torch
from torch.export import export, Dim

class TransformerBlock(torch.nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d, h, batch_first=True)
        self.norm = torch.nn.LayerNorm(d)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d, d * 4), torch.nn.GELU(), torch.nn.Linear(d * 4, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x)
        x = self.norm(x + a)
        return x + self.ff(x)

model = TransformerBlock(256, 8)
model.eval()

# Export with dynamic batch and sequence length
batch = Dim("batch", min=1, max=32)
seq = Dim("seq", min=1, max=128)

exported = export(
    model,
    (torch.randn(2, 16, 256),),
    dynamic_shapes=({0: batch, 1: seq},),
)

# Inspect
print(exported.graph_signature)

# Save and load
torch.export.save(exported, "transformer_block.pt2")
loaded = torch.export.load("transformer_block.pt2")

# Run with different shapes
out = loaded(torch.randn(4, 32, 256))
print(out.shape)  # [4, 32, 256]

# Verify equivalence
with torch.no_grad():
    expected = model(torch.randn(4, 32, 256))
    actual = loaded(torch.randn(4, 32, 256))
    # Note: different random input, just checking shapes
    assert actual.shape == expected.shape
```
