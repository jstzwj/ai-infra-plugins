# torch.compile

`torch.compile` compiles PyTorch models into optimized code using TorchDynamo and TorchInductor.

```python
import torch
```

---

## torch.compile

```python
compiled_model = torch.compile(
    model: Optional[Callable],
    mode: str = "default",          # "default" | "reduce-overhead" | "max-autotune"
    fullgraph: bool = False,        # compile entire graph, error on graph breaks
    dynamic: bool = False,          # enable dynamic shape support
    backend: Union[str, Callable] = "inductor",  # compilation backend
    options: dict = None,           # backend-specific options
    disable: bool = False,          # disable compilation
)
```

### Modes

| Mode | Optimization | Compile Time | Best For |
|------|-------------|--------------|----------|
| `"default"` | Balanced | Moderate | General use |
| `"reduce-overhead"` | Reduces Python overhead with CUDA graphs | Moderate | Small models, low latency |
| `"max-autotune"` | Tries many kernels, picks fastest | Long | Best throughput |

```python
model = torch.compile(model)                              # default
model = torch.compile(model, mode="reduce-overhead")      # low latency
model = torch.compile(model, mode="max-autotune")         # best throughput
model = torch.compile(model, fullgraph=True)              # no graph breaks
model = torch.compile(model, dynamic=True)                # variable shapes
```

---

## Backends

```python
# List available backends
torch._dynamo.list_backends()
# ['eager', 'aot_eager', 'inductor', 'cudagraphs', 'onnxrt', 'tvm', ...]

# Use specific backend
model = torch.compile(model, backend="eager")        # no optimization, debug
model = torch.compile(model, backend="aot_eager")    # AOT compilation, eager exec
model = torch.compile(model, backend="inductor")     # default, Triton/C++ codegen
```

### Custom Backend

```python
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    # gm is a FX GraphModule; customize optimization here
    print(gm.graph)  # inspect graph
    return gm  # return optimized callable

model = torch.compile(model, backend=my_backend)
```

---

## Dynamo Graph Breaks

Dynamo traces Python bytecode. When it encounters unsupported operations, it "graph breaks" -- falling back to eager mode for that section.

### Common Causes of Graph Breaks

- Data-dependent control flow: `if tensor.item() > 0:`
- Unsupported Python features: `inspect`, `eval`, dynamic `exec`
- Certain tensor operations with side effects
- Calls into C++ extensions not wrapped with `torch.ops`

### Inspecting Graph Breaks

```python
import torch._dynamo as dynamo

# Explain why graph breaks occur
explanation = dynamo.explain(model, *example_inputs)
print(explanation)

# Count graph breaks
n_breaks = dynamo.explain(model, *example_inputs).graph_break_count
```

---

## torch.compiler Utilities

```python
# Disable compilation for a function
@torch.compiler.disable
def my_func(x):
    return x + 1

# Allow a custom function in the graph (no graph break)
@torch.compiler.allow_in_graph
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x * 2
    @staticmethod
    def backward(ctx, grad): return grad * 2

# Disable/enable compilation globally
torch.compiler.disable()
torch.compiler.enable()

# Reset compilation cache
torch._dynamo.reset()
```

### Verbose Logging

```python
# Enable Dynamo logs
import torch._dynamo as dynamo
dynamo.config.log_level = logging.INFO

# Environment variable approach
# TORCH_LOGS="+dynamo"         Dynamo tracing
# TORCH_LOGS="+aot"            AOT autograd
# TORCH_LOGS="+inductor"       Inductor codegen
# TORCH_LOGS="dynamic"         Dynamic shape tracing
# TORCH_LOGS="+diff"           Graph diff on recompile
```

---

## Recompilation and Caches

Dynamo caches compiled graphs keyed on input shapes and dtypes. New shapes trigger recompilation.

```python
# Control cache size
torch._dynamo.config.cache_size_limit = 64  # default 8

# Control max autotune tuning
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_pointwise = True
```

---

## Dynamic Shapes

```python
# Enable dynamic shape support (may reduce optimization opportunities)
model = torch.compile(model, dynamic=True)

# Mark dynamic dimensions
from torch._dynamo import mark_dynamic
# mark_dynamic(tensor, dim)  # mark dim as dynamic
```

---

## Common Issues and Solutions

### "Backend compiler failed"

```python
# Fall back to eager for debugging
model = torch.compile(model, backend="eager")

# Get detailed error
torch._dynamo.config.verbose = True
model = torch.compile(model, fullgraph=True)
```

### "Graph break" warnings

```python
# Identify the cause
import torch._dynamo
torch._dynamo.explain(model, *inputs)

# Fix: avoid data-dependent control flow
# Bad:  if loss.item() > threshold: break
# Good: use torch.where or similar tensor ops
```

### Slow first compilation

```python
# Use "reduce-overhead" for faster compile, or cache compiled model
model = torch.compile(model, mode="default")
# Warm up with representative inputs
with torch.no_grad():
    model(torch.randn(batch_size, *input_shape, device="cuda"))
```

---

## Full Training Example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).cuda()

model = torch.compile(model, mode="reduce-overhead")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Warmup (first call triggers compilation)
with torch.no_grad():
    model(torch.randn(32, 784, device="cuda"))

for epoch in range(10):
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        loss = loss_fn(model(data), target)
        loss.backward()
        optimizer.step()
```

---

## torch._inductor.config Options

```python
import torch._inductor.config

config.triton.unique_kernel_names = True
config.triton.cudagraphs = True           # use CUDA graphs in Inductor
config.triton.max_kernel_autotune = 64    # max autotune attempts
config.cpp_wrapper = False                 # use C++ wrapper (True for deployment)
config.size_asserts = True                 # insert runtime size assertions
config.fx_graph_cache = True              # cache FX graphs across runs
```
