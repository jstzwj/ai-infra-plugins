# PyTorch - Chapter 6: nn.Module System

This reference covers the nn.Module base class, Parameter management, hooks, state management, and module utilities.

---

## 6.1 nn.Module Base Class

All neural network components inherit from `torch.nn.Module`.

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x
```

### Key Methods

#### Parameter Access

```python
model = MyModel()

# All parameters (recursive)
for param in model.parameters():
    print(param.shape)

# Named parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    # linear.weight: torch.Size([10, 784])
    # linear.bias: torch.Size([10])

# Buffers (non-parameter tensors like running_mean in BatchNorm)
for buf in model.buffers():
    print(buf.shape)

for name, buf in model.named_buffers():
    print(f"{name}: {buf.shape}")

# Children (direct submodules only)
for child in model.children():
    print(child)

# Named children
for name, child in model.named_children():
    print(f"{name}: {child}")

# All modules (recursive, includes self)
for module in model.modules():
    print(module)

for name, module in model.named_modules():
    print(f"{name}: {module}")
```

#### Parameter Registration

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Method 1: Assign nn.Parameter
        self.weight = nn.Parameter(torch.randn(10, 5))
        self.bias = nn.Parameter(torch.zeros(10))

        # Method 2: register_parameter
        self.register_parameter('scale', nn.Parameter(torch.ones(10)))

        # Method 3: register_buffer (not a parameter, not updated by optimizer)
        self.register_buffer('running_mean', torch.zeros(10))
        self.register_buffer('running_var', torch.ones(10), persistent=True)
        self.register_buffer('step', torch.tensor(0), persistent=False)

        # Method 4: Add submodule
        self.add_module('linear', nn.Linear(5, 10))
        # Equivalent to: self.linear = nn.Linear(5, 10)
```

#### State Management

```python
# Save state
state = model.state_dict()
# OrderedDict([('linear.weight', tensor(...)), ('linear.bias', tensor(...))])

# Load state
model.load_state_dict(state)

# Strict loading (default True)
model.load_state_dict(state, strict=True)   # All keys must match
model.load_state_dict(state, strict=False)  # Allow missing/unexpected keys

# Handle incompatible keys
result = model.load_state_dict(state, strict=False)
result.missing_keys    # ['linear.weight']
result.unexpected_keys # ['old_layer.weight']

# Hooks for state dict
model.register_state_dict_pre_hook(hook)
model.register_state_dict_post_hook(hook)
```

---

## 6.2 Hooks

### Forward Hooks

```python
# Pre-forward hook: called before forward()
def pre_hook(module, input):
    print(f"Input to {module}: {input}")
    return input  # Can modify input

handle = model.linear.register_forward_pre_hook(pre_hook)

# Forward hook: called after forward()
def forward_hook(module, input, output):
    print(f"Output of {module}: {output.shape}")

handle = model.linear.register_forward_hook(forward_hook)

# Remove hook
handle.remove()
```

### Backward Hooks

```python
# Full backward pre-hook: called before backward()
def backward_pre_hook(module, grad_output):
    print(f"Grad output: {grad_output}")

handle = model.linear.register_full_backward_pre_hook(backward_pre_hook)

# Full backward hook: called after backward()
def backward_hook(module, grad_input, grad_output):
    print(f"Grad input: {grad_input}")
    print(f"Grad output: {grad_output}")

handle = module.register_full_backward_hook(backward_hook)
```

### Global Hooks

```python
# Apply to ALL modules
handle = nn.modules.module.register_module_forward_hook(hook)
handle = nn.modules.module.register_module_forward_pre_hook(hook)
handle = nn.modules.module.register_module_backward_hook(hook)
```

### Hook for Weight Registration

```python
nn.modules.module.register_module_parameter_registration_hook(
    lambda module, name, param: print(f"Registered param {name}")
)
nn.modules.module.register_module_buffer_registration_hook(
    lambda module, name, buffer: print(f"Registered buffer {name}")
)
```

---

## 6.3 Training and Evaluation Mode

```python
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.BatchNorm1d(5),
    nn.Dropout(0.5),
)

# Set to training mode
model.train()
# Affects: Dropout (applies), BatchNorm (updates running stats)

# Set to evaluation mode
model.eval()
# Affects: Dropout (disabled), BatchNorm (uses running stats)

# Check mode
model.training  # True/False

# Apply to all submodules
model.apply(lambda m: print(f"{m} training={m.training}"))
```

---

## 6.4 Device and Type Conversion

```python
model = nn.Linear(10, 5)

# Move to device
model = model.to('cuda')
model = model.to(torch.device('cuda:0'))
model = model.cuda()       # Shortcut for CUDA
model = model.cpu()        # Shortcut for CPU
model = model.xpu()        # Intel GPU
model = model.mps()        # Apple Silicon

# Type conversion
model = model.float()      # float32
model = model.double()     # float64
model = model.half()       # float16
model = model.bfloat16()   # bfloat16

# Combined
model = model.to(device='cuda', dtype=torch.float16)

# Apply function recursively
model.apply(lambda m: m.weight.data.normal_() if hasattr(m, 'weight') else None)
```

---

## 6.5 Container Modules

### Sequential

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# With named layers
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(256, 10)),
]))

# Access layers
model[0]           # First layer
model.fc1          # By name
```

### ModuleList

```python
layers = nn.ModuleList([
    nn.Linear(10, 10) for _ in range(5)
])

# Supports list operations
layers.append(nn.Linear(10, 5))
layers.extend([nn.Linear(5, 3)])
layers.insert(0, nn.Linear(10, 10))
layers[0]  # Index access
len(layers)
```

### ModuleDict

```python
acts = nn.ModuleDict({
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
})

acts['relu']  # Access by key
acts.keys(), acts.values(), acts.items()
acts['leaky'] = nn.LeakyReLU()  # Add
```

### ParameterList / ParameterDict

```python
params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for _ in range(3)])
param_dict = nn.ParameterDict({'w': nn.Parameter(torch.randn(5, 5))})
```

---

## 6.6 Lazy Modules

```python
# LazyLinear: in_features inferred from first input
layer = nn.LazyLinear(10)  # Don't need to specify in_features
output = layer(torch.randn(5, 20))  # Now knows in_features=20

# LazyConv variants
conv = nn.LazyConv2d(64, kernel_size=3)

# Check if initialized
layer.has_uninitialized_params()
```

---

## 6.7 Sharing Memory

```python
model = nn.Linear(10, 5)
model.share_memory()  # Move to shared memory for multiprocessing
```

---

## 6.8 Custom repr

```python
class MyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

# Output: MyModule(in_features=10, out_features=5)
```
