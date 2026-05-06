# TorchScript

TorchScript is PyTorch's model serialization and optimization system. It converts PyTorch models to a statically-typed intermediate representation that can run without Python.

```python
import torch
import torch.nn as nn
```

---

## torch.jit.script

Analyzes Python source code to create a ScriptModule. Supports control flow, loops, and most Python constructs.

```python
# Script a module
scripted = torch.jit.script(model)

# Script a function
@torch.jit.script
def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

# Script with type annotations
@torch.jit.script
def gated_linear(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(gate)
```

---

## torch.jit.trace

Records operations performed on example tensors. Cannot capture data-dependent control flow.

```python
traced = torch.jit.trace(model, example_inputs)

# Trace with multiple inputs
traced = torch.jit.trace(model, (x, y))

# Trace a specific method
traced = torch.jit.trace(model, x, check_trace=True, strict=True)
```

### trace_module

```python
traced = torch.jit.trace_module(model, {
    "forward": example_input,
    "encode": example_encoder_input,
})
```

---

## Save and Load

```python
# Save
torch.jit.save(scripted_model, "model.pt")
torch.jit.save(traced_model, "model.pt")

# Load (works in Python and C++)
loaded = torch.jit.load("model.pt")
output = loaded(input)
```

---

## ScriptModule

ScriptModule behaves like `nn.Module` but with TorchScript restrictions.

```python
# Access attributes
scripted.foo                         # attribute access
scripted.code                        # view generated code
scripted.graph                       # view IR graph
scripted.original_name               # original class name

# Named parameters and buffers
for name, param in scripted.named_parameters():
    print(name, param.shape)
```

---

## Decorators

### torch.jit.export

Mark a method as callable from TorchScript.

```python
class MyModule(nn.Module):
    def forward(self, x):
        return x + 1

    @torch.jit.export
    def predict(self, x):
        return self.forward(x).argmax(1)
```

### torch.jit.ignore

Skip TorchScript compilation for a method.

```python
class MyModule(nn.Module):
    @torch.jit.ignore
    def debug_print(self, x):
        print(x.shape)  # Python-only, not compiled

    def forward(self, x):
        self.debug_print(x)
        return x * 2
```

### torch.jit.unused

Mark a method as unused (no compilation, no call).

```python
class MyModule(nn.Module):
    @torch.jit.unused
    def legacy_method(self, x):
        return x  # not compiled, ignored
```

---

## Supported Features in Script

```python
# If/else with tensor conditions
@torch.jit.script
def threshold(x: torch.Tensor, t: float) -> torch.Tensor:
    if x.sum() > t:      # data-dependent control flow OK in script
        return x
    else:
        return torch.zeros_like(x)

# For loops
@torch.jit.script
def cumsum_loop(x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[0] = x[0]
    for i in range(1, x.size(0)):
        result[i] = result[i-1] + x[i]
    return result

# While loops
@torch.jit.script
def while_sum(x: torch.Tensor, threshold: float) -> torch.Tensor:
    total = torch.tensor(0.0)
    i = 0
    while total < threshold and i < x.size(0):
        total = total + x[i]
        i += 1
    return total

# Tuples and lists
@torch.jit.script
def pair_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = x * 2
    b = x + 3
    return (a, b)

# Dict
@torch.jit.script
def dict_op(x: torch.Tensor) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    result["a"] = x + 1
    result["b"] = x * 2
    return result
```

---

## Scripting nn.Modules

```python
class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP(784, 256, 10)
scripted = torch.jit.script(model)
output = scripted(torch.randn(1, 784))
```

---

## freeze

Fuses constants and removes unused code.

```python
frozen = torch.jit.freeze(scripted_model)
# Removes attributes not used in forward, inlines constants
# Can be combined with optimize_for_inference
```

---

## optimize_for_inference

Optimizes a ScriptModule for inference (disables dropout, fuses operations).

```python
optimized = torch.jit.optimize_for_inference(scripted_model)
# Disables dropout, fuses add+layernorm, etc.
```

---

## Operator Fusion

TorchScript automatically fuses common operation patterns:

- **Linear + ReLU / GELU**: Fused into single kernel
- **Conv + BatchNorm + ReLU**: Fused inference kernel
- **Add + LayerNorm**: Fused kernel

```python
# Fusion happens automatically after scripting + freezing
scripted = torch.jit.script(model)
frozen = torch.jit.freeze(scripted)
# Fused ops visible in frozen.graph
```

---

## Loading in C++

```cpp
#include <torch/script.h>

torch::jit::script::Module module;
module = torch::jit::load("model.pt");

std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::rand({1, 784}));

at::Tensor output = module.forward(inputs).toTensor();
```

---

## Common Limitations

- No dynamic Python features: `eval`, `exec`, `globals()`, `locals()`
- No arbitrary Python objects in tensors (only supported types)
- Must type-annotate for complex signatures
- Tracing cannot capture data-dependent control flow (use `script` instead)
- Third-party Python libraries not supported in script mode

---

## Complete Example

```python
import torch, torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class Model(nn.Module):
    def __init__(self, d: int, n_blocks: int, n_classes: int):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(d) for _ in range(n_blocks)])
        self.head = nn.Linear(d, n_classes)

    @torch.jit.export
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))

model = Model(256, 4, 10)
scripted = torch.jit.script(model)
frozen = torch.jit.freeze(scripted)
optimized = torch.jit.optimize_for_inference(frozen)
torch.jit.save(optimized, "model.pt")

# Load and run
loaded = torch.jit.load("model.pt")
output = loaded(torch.randn(1, 256))
pred = loaded.predict(torch.randn(1, 256))
```
