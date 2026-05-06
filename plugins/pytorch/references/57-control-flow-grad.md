# PyTorch - Chapter 57: Control Flow and Gradient Checkpointing

This reference covers differentiable control flow and gradient checkpointing.

---

## 57.1 torch.cond

```python
torch.cond(pred, true_fn, false_fn, operands)
```

Differentiable conditional execution.

```python
from torch._higher_order_ops import cond

def true_fn(x):
    return x * 2

def false_fn(x):
    return x * 3

result = torch.cond(torch.tensor(True), true_fn, false_fn, (x,))
```

---

## 57.2 Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# Single function checkpointing
def custom_fn(x):
    return torch.relu(x * 2)

output = checkpoint(custom_fn, input, use_reentrant=False)

# Sequential model checkpointing
model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(10)])
output = checkpoint_sequential(model, segments=4, input=x)

# use_reentrant=False recommended (supports all ops)
output = checkpoint(layer, x, use_reentrant=False)
```

---

## 57.3 saved_tensors_hooks

```python
# Offload saved tensors to CPU during forward, load during backward
def pack_hook(tensor):
    return tensor.to('cpu')

def unpack_hook(packed):
    return packed.to('cuda')

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    output = model(input)
    loss.backward()
```

---

## 57.4 save_on_cpu

```python
with torch.autograd.graph.save_on_cpu(pin_memory=True):
    output = model(input)
    loss.backward()
```
