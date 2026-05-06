# torch.fx Graph Manipulation

`torch.fx` provides a Python-to-Python transformation toolkit for PyTorch models via symbolic tracing.

```python
import torch
import torch.fx
```

---

## torch.fx.symbolic_trace

Traces a module and produces a `GraphModule` containing an IR representation.

```python
gm: GraphModule = torch.fx.symbolic_trace(
    root: Union[nn.Module, Callable],
    concrete_args: dict = None,
)
```

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.param = nn.Parameter(torch.randn(10))

    def forward(self, x):
        return self.linear(x) + self.param

m = MyModule()
gm = torch.fx.symbolic_trace(m)
print(gm.graph)      # print the IR
print(gm.code)       # print generated Python code
```

---

## Graph and Node

### Graph

```python
gm.graph               # the Graph object
gm.graph.nodes         # iterable of Node objects
gm.graph.print_tabular()  # print nodes in table format
```

### Node Types

Each Node has an `op` field indicating its type:

| op | Description | Example |
|----|-------------|---------|
| `placeholder` | Function input | `%x : ...` |
| `get_attr` | Get module attribute | `self.linear` |
| `call_function` | Call free function | `torch.add(x, y)` |
| `call_method` | Call tensor method | `x.sum()` |
| `call_module` | Call nn.Module | `self.linear(x)` |
| `output` | Function output | `return result` |

### Node Properties

```python
node.op          # str: operation type
node.target      # str or callable: target of the operation
node.args        # tuple: positional arguments
node.kwargs      # dict: keyword arguments
node.name        # str: unique name for this node
node.type        # type annotation or None
node.meta        # dict: metadata (shape, dtype, etc.)
node.users       # dict: nodes that use this node's output
node.all_input_nodes  # list: nodes that are inputs to this node
node.next        # next node in graph
node.prev        # previous node in graph
node.replace_all_uses_with(other_node)  # reroute all uses
node.replace_input_with(old, new)        # replace one input
node.update_arg(idx, new_val)            # update positional arg
```

---

## GraphModule

GraphModule combines a `Graph` with the original module's attributes.

```python
gm.graph            # Graph object
gm.code             # generated Python source code
gm.print_readable() # pretty-print the code

# Recompile after graph modification
gm.recompile()

# Access original parameters
for name, param in gm.named_parameters():
    print(name, param.shape)
```

---

## Graph Manipulation

### Adding Nodes

```python
graph = gm.graph

# Insert at end
with graph.inserting_after(some_node):
    new_node = graph.call_function(torch.relu, (some_node,))

# Insert at beginning
with graph.inserting_before(some_node):
    new_node = graph.call_function(torch.abs, (some_node,))
```

### Removing Nodes

```python
# Erase a node (must have no users)
graph.erase_node(node)

# Remove all dead code
graph.eliminate_dead_code()
```

### Replacing Operations

```python
# Replace all uses of a node
old_node.replace_all_uses_with(new_node)

# Example: replace all torch.add with custom op
for node in gm.graph.nodes:
    if node.op == "call_function" and node.target is torch.add:
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                my_custom_add, node.args, node.kwargs)
            node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

gm.recompile()
```

---

## torch.fx.wrap

Register a function as a leaf node (not traced through).

```python
torch.fx.wrap("my_custom_func")

def my_custom_func(x):
    # This body is not traced
    return complex_operation(x)

# Now symbolic_trace will treat calls to my_custom_func as a single op
```

---

## Custom Tracer

```python
class MyTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Don't trace into MyCustomLayer
        if isinstance(m, MyCustomLayer):
            return True
        return super().is_leaf_module(m, module_qualified_name)

tracer = MyTracer()
gm = tracer.trace(model)
gm = torch.fx.GraphModule(model, gm)
```

---

## Interpreter

Execute or transform a GraphModule node-by-node.

```python
class ShapeProp(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        n.meta["val"] = result
        n.meta["tensor_meta"] = torch.fx.passes.shape_prop.TensorMeta(result)
        return result

    def propagate(self, *args):
        return super().run(*args)

# Usage
sp = ShapeProp(gm)
fake_args = (torch.randn(1, 10),)
sp.propagate(*fake_args)

for node in gm.graph.nodes:
    if "val" in node.meta:
        print(f"{node.name}: {node.meta['val'].shape}")
```

---

## ShapeProp (Built-in)

```python
from torch.fx.passes.shape_prop import ShapeProp

sp = ShapeProp(gm)
sp.run(torch.randn(1, 10))  # propagates shapes through the graph

for node in gm.graph.nodes:
    if "tensor_meta" in node.meta:
        print(f"{node.name}: {node.meta['tensor_meta']}")
```

---

## Subgraph Rewriting

```python
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

# Find and replace patterns
def replace_pattern(gm, pattern, replacement):
    """Replace all occurrences of pattern with replacement in gm."""
    matches = SubgraphMatcher(pattern).match(gm.graph)
    for match in matches:
        # rewrite matched subgraph
        pass
```

---

## Common Transformations Example

```python
import torch, torch.nn as nn, torch.fx

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Model()
gm = torch.fx.symbolic_trace(model)

# Replace ReLU with GELU
for node in gm.graph.nodes:
    if node.op == "call_function" and node.target is torch.relu:
        with gm.graph.inserting_after(node):
            new = gm.graph.call_function(torch.nn.functional.gelu, node.args)
            node.replace_all_uses_with(new)
            gm.graph.erase_node(node)

gm.recompile()
result = gm(torch.randn(1, 10))
```

---

## fx with torch.compile

```python
# Custom backend using FX
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    # Apply custom passes
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.add:
            # Custom optimization
            pass
    gm.recompile()
    return gm

model = torch.compile(model, backend=my_backend)
```
