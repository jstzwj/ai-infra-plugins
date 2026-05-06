# PyTorch - Chapter 4: Automatic Differentiation

This reference covers PyTorch's automatic differentiation system (autograd), including the backward pass, custom Functions, gradient modes, forward-mode AD, and gradient utilities.

---

## 4.1 torch.autograd.backward

```python
torch.autograd.backward(
    tensors,                          # Tensors to differentiate
    grad_tensors=None,                # Gradient of each tensor (default: all ones)
    retain_graph=None,                # Keep computation graph after backward
    create_graph=False,               # Create graph of derivative (for higher-order grads)
    inputs=None,                      # Only compute gradients for these inputs
) -> None
```

Computes the sum of gradients of given tensors with respect to graph leaves.

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()
torch.autograd.backward(z)
print(x.grad)  # tensor([2., 2., 2.])

# Equivalent to:
z.backward()
```

**Parameters**:
- `tensors`: Tensors of which the derivative will be computed
- `grad_tensors`: The "vector" in vector-Jacobian product. If None, defaults to all-ones tensors of the same shape as `tensors`
- `retain_graph`: If False, the graph used to compute the grad will be freed. Set to True if you need to backward through the graph multiple times
- `create_graph`: If True, graph of the derivative will be constructed, allowing computing higher order derivative products
- `inputs`: Inputs w.r.t. which the gradient will be accumulated into `.grad`. All other Tensors will be ignored. If not provided, gradient is accumulated w.r.t. all leaf Tensors

---

## 4.2 torch.autograd.grad

```python
torch.autograd.grad(
    outputs,                          # Output tensors
    inputs,                           # Input tensors
    grad_outputs=None,                # Gradient of outputs
    retain_graph=None,                # Keep graph after computation
    create_graph=False,               # Build graph of derivative
    only_inputs=True,                 # Only compute grad for inputs
    allow_unused=None,                # Return zeros for unused inputs
    is_grads_batched=False,           # Batched gradients
    materialize_grads=False,          # Materialize zero grads instead of None
) -> Tuple[Tensor, ...]
```

Computes and returns the sum of gradients of outputs with respect to the inputs.

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()

grads = torch.autograd.grad(z, x)
print(grads[0])  # tensor([2., 2., 2.])

# Multiple inputs
a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
out = (a * b).sum()
da, db = torch.autograd.grad(out, [a, b])
# da = b, db = a
```

---

## 4.3 torch.autograd.Function

Base class for creating custom autograd operations.

### Core Methods

```python
class torch.autograd.Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Compute the output. Must be a static method."""
        # ctx: context object for saving information
        ctx.save_for_backward(tensor1, tensor2)  # Save tensors for backward
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Set up context for backward. Called after forward."""
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Compute gradients. Must be a static method."""
        input, = ctx.saved_tensors
        return grad_input

    @staticmethod
    def jvp(ctx, *grad_inputs):
        """Compute Jacobian-vector product for forward-mode AD."""

    @staticmethod
    def vmap(info, in_dims, *args):
        """Support for torch.vmap."""
```

### Context (ctx) Methods

```python
ctx.save_for_backward(*tensors)      # Save tensors for backward
ctx.save_for_forward(*tensors)       # Save tensors for forward-mode AD
ctx.saved_tensors                     # Retrieve saved tensors (tuple)
ctx.mark_dirty(*args)                 # Mark inputs as modified in-place
ctx.mark_non_differentiable(*args)    # Mark outputs as non-differentiable
ctx.set_materialize_grads(value)      # If False, grad may be None
ctx.needs_input_grad                  # Tuple of booleans
ctx.is_differentiable                 # Whether in backward pass
```

### Example: Custom Linear Function

```python
class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias

# Usage
output = MyLinearFunction.apply(input, weight, bias)
loss = output.sum()
loss.backward()
```

### Example: Custom ReLU

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).to(grad_output.dtype)

# Usage
output = MyReLU.apply(input)
```

### Example: Custom CUDA Operation

```python
# In C++/CUDA extension
class MyCudaOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = my_cuda_kernel(input)  # Custom CUDA kernel
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = my_cuda_backward_kernel(grad_output, input)
        return grad_input
```

---

## 4.4 Gradient Modes

### no_grad

```python
torch.no_grad()
```
Context manager / decorator that disables gradient computation. Useful for inference to reduce memory usage and speed up computation.

```python
with torch.no_grad():
    output = model(input)    # No gradient tracking

# As decorator
@torch.no_grad()
def inference(model, input):
    return model(input)
```

### enable_grad

```python
torch.enable_grad()
```
Enables gradient computation inside a `no_grad` context.

```python
with torch.no_grad():
    # No gradients tracked here
    with torch.enable_grad():
        output = model(input)  # Gradients ARE tracked
```

### set_grad_enabled

```python
torch.set_grad_enabled(mode)
```
Context manager that sets gradient computation on or off.

```python
torch.set_grad_enabled(False)  # Disable globally
torch.set_grad_enabled(True)   # Enable globally

is_training = model.training
with torch.set_grad_enabled(is_training):
    output = model(input)
```

### inference_mode

```python
torch.inference_mode(mode=True)
```
More aggressive than `no_grad`. Disables both gradient computation and view tracking. Tensors created in inference mode cannot be used in gradient computation after exiting.

```python
with torch.inference_mode():
    output = model(input)  # Fastest inference mode

# Decorator
@torch.inference_mode()
def predict(model, input):
    return model(input)
```

**Difference from no_grad**:
- `no_grad`: Disables gradient tracking only. View tracking still active.
- `inference_mode`: Disables both gradient AND view tracking. Faster but more restrictive.

---

## 4.5 Forward-Mode AD

```python
torch.autograd.forward_ad(func, *args, **kwargs)
```

Forward-mode automatic differentiation computes Jacobian-vector products efficiently.

```python
import torch.autograd.forward_ad as fwAD

# Create dual tensors (value + tangent)
primal = torch.randn(3)
tangent = torch.tensor([1.0, 0.0, 0.0])  # Direction

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)
    dual_output = func(dual_input)
    primal_out, tangent_out = fwAD.unpack_dual(dual_output)
    # tangent_out is JVP (Jacobian-vector product)
```

---

## 4.6 Gradient Checking

```python
torch.autograd.gradcheck(
    func,                  # Function to check
    inputs,                # Input arguments
    eps=1e-6,              # Perturbation for finite differences
    atol=1e-5,             # Absolute tolerance
    rtol=1e-3,             # Relative tolerance
    raise_exception=True,  # Raise on failure
    check_sparse_nnz=False,
    nondet_tol=0.0,
    check_undefined_grad=True,
)

torch.autograd.gradgradcheck(
    func, inputs,          # Checks second-order gradients
    eps=1e-6, atol=1e-5, rtol=1e-3,
)
```

Verifies that computed gradients match numerical (finite difference) gradients.

```python
def my_func(x):
    return x ** 3

x = torch.randn(3, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(my_func, x)
```

**Important**: Use `float64` for accurate numerical differentiation.

---

## 4.7 Anomaly Detection

```python
torch.autograd.set_detect_anomaly(mode, check_nan=True)
# or
with torch.autograd.detect_anomaly():
    loss.backward()
```

Detects NaN/Inf in backward pass and identifies which operation produced them.

```python
torch.autograd.set_detect_anomaly(True)
loss.backward()
# Will print warning: "RuntimeError: Function 'XXX' returned nan values in its 0th output."
```

**Note**: Enabling anomaly detection adds overhead. Only use for debugging.

---

## 4.8 torch.autograd.graph

### Saved Tensor Hooks

```python
torch.autograd.graph.saved_tensors_hooks(pack_fn, unpack_fn)
```

Intercept tensor saving during forward pass (for memory optimization, offloading, compression).

```python
def pack_hook(tensor):
    return tensor.to('cpu')

def unpack_hook(packed):
    return packed.to('cuda')

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    output = model(input)
    loss.backward()
# Saved tensors were offloaded to CPU during forward and loaded back during backward
```

### save_on_cpu

```python
torch.autograd.graph.save_on_cpu(pin_memory=False)
```

Shortcut for offloading saved tensors to CPU.

```python
with torch.autograd.graph.save_on_cpu(pin_memory=True):
    output = model(input)
    loss.backward()
```

---

## 4.9 Gradient Checkpointing

```python
torch.utils.checkpoint.checkpoint(
    function,                  # Function to checkpoint
    *args,                     # Arguments
    use_reentrant=True,        # Use reentrant autograd
    context_fn=None,           # Custom saved tensor hooks
    determinism_check='default',
    debug=False,
)

torch.utils.checkpoint.checkpoint_sequential(
    functions,                 # List of functions or nn.Sequential
    segments,                  # Number of segments
    input,                     # Input tensor
    use_reentrant=True,
)
```

Trades compute for memory by not storing intermediate activations during forward. Instead recomputes them during backward.

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(10)])

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# checkpoint_sequential for sequential models
model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(10)])
output = checkpoint_sequential(model, segments=4, input=x)
```

**use_reentrant**:
- `True` (default, legacy): Uses reentrant autograd. Has some limitations (no in-place ops, no custom autograd functions, tensors must be leaf).
- `False` (recommended): No reentrant autograd. Supports all operations but slightly more memory usage.

---

## 4.10 Higher-Order Gradients

```python
x = torch.randn(3, requires_grad=True)

# First order
y = x ** 3
dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
# dy_dx = 3 * x^2

# Second order
d2y_dx2 = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]
# d2y_dx2 = 6 * x

# Third order
d3y_dx3 = torch.autograd.grad(d2y_dx2.sum(), x)[0]
# d3y_dx3 = 6 (constant)
```

---

## 4.11 Functional API: torch.func

JAX-like function transforms (formerly functorch):

### vmap (Vectorization)

```python
torch.func.vmap(func, in_dims=0, out_dims=0, randomness='error', chunk_size=None)
```

```python
from torch.func import vmap

def f(x):
    return x ** 2

batch_size, feature_dim = 64, 10
inputs = torch.randn(batch_size, feature_dim)

# Instead of a loop:
# outputs = torch.stack([f(inputs[i]) for i in range(batch_size)])
# Use vmap:
outputs = vmap(f)(inputs)
```

### grad

```python
torch.func.grad(func, argnums=0, has_aux=False)
```

```python
from torch.func import grad

def f(x):
    return (x ** 2).sum()

grad_f = grad(f)
x = torch.randn(3)
g = grad_f(x)  # gradient at x: 2*x
```

### jacfwd / jacrev

```python
torch.func.jacfwd(func, argnums=0, has_aux=False, randomness='error')  # Forward-mode
torch.func.jacrev(func, argnums=0, has_aux=False, randomness='error')  # Reverse-mode
```

```python
from torch.func import jacfwd, jacrev

def f(x):
    return x ** 2

x = torch.randn(3)
J_forward = jacfwd(f)(x)  # Forward-mode Jacobian
J_reverse = jacrev(f)(x)  # Reverse-mode Jacobian
# Both: diag(2*x)
```

### hessian

```python
torch.func.hessian(func, argnums=0)
```

```python
from torch.func import hessian

def f(x):
    return (x ** 3).sum()

H = hessian(f)(torch.randn(3))
# H[i,j] = d2f/dxidxj
```

### jvp / vjp

```python
torch.func.jvp(func, primals, tangents, has_aux=False)   # Jacobian-vector product
torch.func.vjp(func, primals, has_aux=False)              # Vector-Jacobian product
```

### Per-Sample Gradients

```python
from torch.func import grad, vmap, functional_call

def compute_loss(params, buffers, sample, target):
    predictions = functional_call(model, (params, buffers), sample)
    return nn.functional.cross_entropy(predictions, target)

# Per-sample gradients
sample_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(
    params, buffers, samples, targets
)
```

---

## 4.12 Profiling Autograd

```python
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    output = model(input)
    loss.backward()

print(prof.key_averages().table(sort_by='cuda_time_total'))
```

### emit_nvtx

```python
with torch.autograd.profiler.emit_nvtx():
    output = model(input)
    loss.backward()
# Visualize in NVIDIA Nsight
```
