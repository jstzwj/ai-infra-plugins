# PyTorch Reference - Chapter 4: Automatic Differentiation

This chapter covers PyTorch's automatic differentiation system (autograd) in comprehensive detail, including the backward pass, custom Functions, gradient modes, forward-mode AD, gradient checking, anomaly detection, and the functional API (torch.func).

---

## 4.1 torch.autograd.backward

Computes the sum of gradients of given tensors with respect to graph leaves.

```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None)
```

**Parameters:**
- `tensors` (Tensors or tuple of Tensors): Tensors of which the derivative will be computed.
- `grad_tensors` (Tensors or tuple of Tensors or None): The "vector" in the vector-Jacobian product. Typically gradients w.r.t. each element of the corresponding tensor. If not provided, defaults to 1 for each tensor. Must match the shape of `tensors`.
- `retain_graph` (bool, optional): If `False`, the graph used to compute the grad will be freed. If `True`, the graph is retained for additional backward passes. Default: `create_graph`.
- `create_graph` (bool, optional): If `True`, graph of the derivative will be constructed, allowing computing higher order derivative products. Default: `False`.
- `inputs` (Tensors or tuple of Tensors or None): Inputs w.r.t. which the gradient will be accumulated into `.grad`. All other Tensors will be ignored. If not provided, the gradient is accumulated into all leaf Tensors.

**Examples:**
```python
import torch

# Basic backward
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward()
print(x.grad)                   # tensor([2., 4., 6.]) = 2*x

# Multiple outputs (grad_tensors acts as upstream gradient)
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = x * 3
torch.autograd.backward([y, z], [torch.ones(2), torch.ones(2)])
print(x.grad)                   # tensor([5., 5.]) = 2 + 3

# Retain graph for multiple backward passes
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
loss = y.sum()
loss.backward(retain_graph=True)
print(x.grad)                   # tensor([2., 4.])
loss.backward(retain_graph=True)
print(x.grad)                   # tensor([4., 8.]) - accumulated!
x.grad.zero_()                  # Clear gradients
loss.backward()
print(x.grad)                   # tensor([2., 4.]) - fresh gradients

# create_graph for higher-order gradients
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
# dy_dx = 3x^2
d2y_dx2 = torch.autograd.grad(dy_dx.sum(), x)[0]
print(d2y_dx2)                  # tensor([6., 12.]) = 6x

# Backward w.r.t. specific inputs
x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([3.0, 4.0], requires_grad=True)
y = x * w
loss = y.sum()
loss.backward(inputs=[x])       # Only x.grad is computed, w.grad is None
```

**Notes:**
- Calling `backward()` accumulates gradients in `.grad`. Use `zero_grad()` before each backward pass.
- Only leaf tensors with `requires_grad=True` get `.grad` populated.
- For non-scalar tensors, you must provide `grad_tensors` matching the shape.

---

## 4.2 torch.autograd.grad

Computes and returns the sum of gradients of outputs with respect to inputs.

```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, materialize_grads=False)
```

**Parameters:**
- `outputs` (Tensor or tuple of Tensors): Outputs of the differentiated function.
- `inputs` (Tensor or tuple of Tensors): Inputs w.r.t. which the gradient will be returned.
- `grad_outputs` (Tensor or tuple of Tensors): The "vector" in the vector-Jacobian product. Defaults to all 1s.
- `retain_graph` (bool, optional): If `False`, the graph is freed after computing the gradient.
- `create_graph` (bool): If `True`, the graph of the gradient is constructed for higher-order derivatives.
- `allow_unused` (bool): If `True`, returns `None` for inputs not connected to outputs instead of raising an error. Default: `False`.
- `materialize_grads` (bool): If `True`, fills in zero tensors for unused inputs instead of returning `None`. Default: `False`.

**Examples:**
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()

# Compute gradient
grad = torch.autograd.grad(z, x)
print(grad[0])                  # tensor([2., 4., 6.])

# Multiple inputs
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)
y = w * x + b
loss = y.sum()
grads = torch.autograd.grad(loss, [w, b])
print(grads[0])                 # tensor([1., 2., 3.]) = x
print(grads[1])                 # tensor([3.]) = count

# allow_unused
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = x * 2  # y is not used
grads = torch.autograd.grad(z, [x, y], allow_unused=True)
print(grads[0])                 # tensor([2.])
print(grads[1])                 # None

# materialize_grads
grads = torch.autograd.grad(z, [x, y], allow_unused=True, materialize_grads=True)
print(grads[1])                 # tensor([0.]) - zero instead of None

# grad_outputs for weighted gradients
x = torch.randn(3, requires_grad=True)
y = x ** 2
weights = torch.tensor([1.0, 0.0, 1.0])
grads = torch.autograd.grad(y, x, grad_outputs=weights)
# Only computes gradient for elements where weight is nonzero
```

---

## 4.3 torch.autograd.Function

Base class for creating custom autograd functions with explicit forward and backward implementations.

### 4.3.1 Class Structure

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # Compute the forward pass
        # ctx is a context object for saving information
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Compute the gradient
        # ctx.saved_tensors retrieves saved tensors
        # Must return one gradient per input (use None for non-differentiable inputs)
        return grad_inputs

    # Optional: for forward-mode AD
    @staticmethod
    def jvp(ctx, *grad_inputs):
        # Compute the Jacobian-vector product for forward-mode AD
        return grad_outputs

    # Optional: for torch.vmap support
    @staticmethod
    def vmap(info, in_dims, *args):
        # Define batching rule for vmap
        pass
```

### 4.3.2 Context Object (ctx) Methods

```python
# Save tensors for backward (recommended over storing as attributes)
ctx.save_for_backward(*tensors)

# Retrieve saved tensors
tensors = ctx.saved_tensors

# Save for forward (used with dual numbers in forward-mode AD)
ctx.save_for_forward(*tensors)

# Mark a tensor as modified in-place (so autograd can handle it)
ctx.mark_dirty(*tensors)

# Mark outputs as non-differentiable
ctx.mark_non_differentiable(*outputs)

# Tell autograd to return zero tensors instead of None for non-differentiable outputs
ctx.set_materialize_grads(False)

# Whether we are in forward-mode AD
ctx.is_forward_mode()

# Needs input gradients (which inputs need gradients)
ctx.needs_input_grad

# Access the input arguments
# (not stored by default - use save_for_backward to persist tensors)
```

### 4.3.3 Custom Linear Function Example

```python
class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias

# Usage
linear = CustomLinearFunction.apply
output = linear(input, weight, bias)
loss = output.sum()
loss.backward()
```

### 4.3.4 Custom ReLU Example

```python
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
x = torch.randn(5, requires_grad=True)
y = CustomReLU.apply(x)
y.sum().backward()
print(x.grad)  # 1 where x >= 0, 0 where x < 0
```

### 4.3.5 Custom Function with Higher-Order Gradients

```python
class ExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.exp()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Use the custom function again to support higher-order gradients
        return grad_output * ExpFunction.apply(input)

# Usage - supports second-order gradients
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = ExpFunction.apply(x)
dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
d2y = torch.autograd.grad(dy.sum(), x)[0]
print(d2y)  # tensor([2.7183, 7.3891]) = exp(x)
```

### 4.3.6 setup_context (Alternative Pattern)

For simpler custom functions, you can use the `setup_context` pattern instead of writing forward with `ctx`:

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias
```

### 4.3.7 Important Notes on Custom Functions

1. Always use `ctx.save_for_backward()` to save tensors, not Python attributes. This properly handles tensor lifecycle and supports features like `torch.utils.checkpoint`.

2. The `backward()` method must return a tuple with one element per input to `forward()`. Use `None` for non-differentiable inputs.

3. `ctx.needs_input_grad` is a tuple of booleans indicating which inputs need gradients. Use it to skip unnecessary computation.

4. Custom functions support forward-mode AD via the optional `jvp()` static method.

5. The `apply()` method is the correct way to invoke custom functions. Do not call `forward()` directly.

6. For in-place operations, use `ctx.mark_dirty()`.

---

## 4.4 Gradient Modes

### 4.4.1 torch.no_grad

Disables gradient computation. Operations inside the block will not be tracked.

```python
torch.no_grad()
# Also usable as decorator:
@torch.no_grad()
def inference_fn(x):
    return model(x)
```

**Examples:**
```python
x = torch.tensor([1.0], requires_grad=True)

with torch.no_grad():
    y = x * 2
    print(y.requires_grad)      # False

# The no_grad context also affects tensor creation
with torch.no_grad():
    t = torch.randn(3, requires_grad=True)
    print(t.requires_grad)      # False (overridden by no_grad)
```

### 4.4.2 torch.enable_grad

Enables gradient computation. Useful inside `no_grad` blocks.

```python
torch.enable_grad()
```

**Examples:**
```python
x = torch.tensor([1.0], requires_grad=True)

with torch.no_grad():
    with torch.enable_grad():
        y = x * 2
        print(y.requires_grad)  # True
```

### 4.4.3 torch.set_grad_enabled

Sets gradient computation on or off.

```python
torch.set_grad_enabled(mode)
```

**Examples:**
```python
is_training = True

with torch.set_grad_enabled(is_training):
    y = model(x)
    # Gradients tracked if is_training is True

# Check current state
torch.is_grad_enabled()         # True or False
```

### 4.4.4 torch.inference_mode

More restrictive than `no_grad`. Disables both gradient computation and version counting. Tensors produced in inference mode cannot be used in autograd after exiting the mode.

```python
torch.inference_mode(mode=True)
# Also usable as decorator:
@torch.inference_mode()
def inference_fn(x):
    return model(x)
```

**Examples:**
```python
x = torch.tensor([1.0], requires_grad=True)

with torch.inference_mode():
    y = x * 2
    print(y.requires_grad)      # False

# Inference mode tensors cannot be used in autograd
# y.backward()  # RuntimeError!

# Check inference mode
torch.is_inference_mode_enabled()
```

**Notes:**
- `inference_mode` is faster than `no_grad` because it also disables view tracking and version counting.
- Use `inference_mode` for pure inference (no backward pass needed).
- Use `no_grad` when you need to temporarily disable gradients but still use the tensors in autograd later.

### 4.4.5 Summary of Gradient Modes

| Mode | Gradients | Version Counting | Use Case |
|------|-----------|-----------------|----------|
| Default (enabled) | Tracked | Yes | Training |
| `no_grad` | Not tracked | Yes | Temporary disable (e.g., validation) |
| `enable_grad` | Tracked | Yes | Re-enable inside `no_grad` |
| `inference_mode` | Not tracked | No | Pure inference (fastest) |

---

## 4.5 Forward-Mode Automatic Differentiation

### 4.5.1 torch.autograd.forward_ad

Context manager for forward-mode AD.

```python
with torch.autograd.forward_ad.dual_level():
    # Create dual tensors (primal + tangent)
    x = torch.autograd.forward_ad.make_dual(primal, tangent)
    y = f(x)
    primal_out, tangent_out = torch.autograd.forward_ad.unpack_dual(y)
```

### 4.5.2 Dual Numbers

Forward-mode AD uses **dual numbers**: each value carries both a primal (the actual value) and a tangent (the directional derivative).

```python
import torch.autograd.forward_ad as fwAD

primal = torch.randn(3)
tangent = torch.tensor([1.0, 0.0, 0.0])  # Direction: differentiate w.r.t. x[0]

with fwAD.dual_level():
    x = fwAD.make_dual(primal, tangent)
    y = x ** 2 + x * 3

    primal_y, tangent_y = fwAD.unpack_dual(y)
    # primal_y: the function value
    # tangent_y: the directional derivative in the direction of `tangent`
    # = 2 * primal[0] + 3 (derivative of x^2 + 3x w.r.t. x[0])
```

### 4.5.3 torch.func.jvp

Computes the Jacobian-vector product (forward-mode AD) using `torch.func`.

```python
from torch.func import jvp

def f(x):
    return x ** 2 + 3 * x

x = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([1.0, 0.0, 0.0])  # Tangent direction

value, jvp_result = jvp(f, (x,), (v,))
# value: f(x) = [4, 10, 18]
# jvp_result: J @ v = directional derivative
```

---

## 4.6 Gradient Checking

### 4.6.1 torch.autograd.gradcheck

Checks gradients computed by autograd against numerical gradients (finite differences).

```python
torch.autograd.gradcheck(func, inputs, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True, check_sparse_nnz=False, nondet_tol=0.0, check_undefined_grad=False, check_grad_dtypes=False, fast_mode=False)
```

**Parameters:**
- `func` (callable): A Python function that takes tensor inputs and returns a tensor or tuple of tensors.
- `inputs` (tuple of Tensors): Input tensors.
- `eps` (float): Perturbation for finite differences. Default: 1e-6.
- `atol` (float): Absolute tolerance. Default: 1e-5.
- `rtol` (float): Relative tolerance. Default: 1e-3.

**Examples:**
```python
def f(x):
    return x ** 2 + 2 * x + 1

x = torch.randn(3, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(f, (x,))  # Returns True if gradients match

# For functions with multiple inputs
def g(x, y):
    return (x * y).sum()

x = torch.randn(3, dtype=torch.float64, requires_grad=True)
y = torch.randn(3, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(g, (x, y))
```

**Notes:**
- Use `float64` inputs for numerical stability of finite differences.
- `gradcheck` perturbs each element independently and computes numerical gradients.

### 4.6.2 torch.autograd.gradgradcheck

Checks second-order gradients.

```python
torch.autograd.gradgradcheck(func, inputs, grad_outputs=None, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True, nondet_tol=0.0, check_undefined_grad=False)
```

---

## 4.7 Anomaly Detection

### 4.7.1 torch.autograd.detect_anomaly

Context manager that detects NaN or Inf values during backward and reports which operation produced them.

```python
torch.autograd.detect_anomaly(check_nan=True)
# Also:
torch.autograd.set_detect_anomaly(True)
```

**Examples:**
```python
with torch.autograd.detect_anomaly():
    x = torch.tensor([1.0], requires_grad=True)
    y = x / 0
    y.backward()
    # Will print: "Function 'DivBackward0' returned nan values in its 0th output."

# Global setting
torch.autograd.set_detect_anomaly(True)
# Now any backward pass with NaN/Inf will raise an error with traceback
```

**Notes:**
- Anomaly detection adds overhead. Only enable for debugging.
- It helps identify which operation in the backward pass produced NaN/Inf.

---

## 4.8 torch.autograd.graph

Utilities for working with the autograd computation graph.

### 4.8.1 Saved Tensor Hooks

Hook into the save/restore of tensors during backward.

```python
# Pack hook: called when a tensor is saved for backward
def pack_hook(tensor):
    return tensor.to('cuda')

# Unpack hook: called when a saved tensor is restored
def unpack_hook(packed):
    return packed.to('cpu')

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    y.sum().backward()
```

**Use case:** Move tensors to GPU during forward (to save CPU memory) and back to CPU during backward.

### 4.8.2 Node and Gradient Edge

```python
# Access the autograd graph
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()

# Get the gradient edge (node, output_idx)
edge = z.grad_fn          # The backward function node
edge.next_functions       # Tuple of (next_node, input_idx) edges
```

---

## 4.9 Compiled Autograd

### 4.9.1 torch._compiled_autograd

Compiled autograd compiles the backward pass using Dynamo, potentially improving backward pass performance.

```python
import torch._compiled_autograd as compiled_autograd

# Enable compiled autograd
compiled_autograd.enable()

# Or use context manager
with compiled_autograd.enable():
    # Training code with compiled backward
    x = torch.randn(3, requires_grad=True)
    y = x ** 2
    y.sum().backward()
```

---

## 4.10 Higher-Order Gradients

PyTorch supports computing gradients of gradients (higher-order derivatives).

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# First-order
y = x ** 3
dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
print(dy)                       # tensor([ 3., 12., 27.]) = 3x^2

# Second-order
d2y = torch.autograd.grad(dy.sum(), x, create_graph=True)[0]
print(d2y)                      # tensor([6., 12., 18.]) = 6x

# Third-order
d3y = torch.autograd.grad(d2y.sum(), x)[0]
print(d3y)                      # tensor([6., 6., 6.]) = 6
```

**Important:**
- `create_graph=True` must be set to enable higher-order gradients.
- The intermediate gradient is itself a differentiable computation.

---

## 4.11 Gradient Checkpointing (torch.utils.checkpoint)

Saves memory by recomputing intermediate activations during backward instead of storing them during forward.

### 4.11.1 torch.utils.checkpoint.checkpoint

```python
torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=True, context_fn=None, determinism_check='default', debug=False, **kwargs)
```

**Parameters:**
- `function` (callable): The function to checkpoint.
- `*args`: Arguments to the function.
- `use_reentrant` (bool): If `True`, uses the older reentrant autograd API. If `False`, uses the non-reentrant API (recommended for new code).
- `context_fn` (callable, optional): Custom context for saving/restoring RNG state.

**Examples:**
```python
import torch.utils.checkpoint as cp

class CheckpointedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = cp.checkpoint(self.block2, x, use_reentrant=False)
        x = self.block3(x)
        return x

# With the non-reentrant API (recommended)
x = torch.randn(32, 784, requires_grad=True)
model = CheckpointedModel()
output = model(x)
output.sum().backward()
```

### 4.11.2 checkpoint_sequential

Checkpoint a sequential module.

```python
torch.utils.checkpoint.checkpoint_sequential(functions, segments, input, use_reentrant=True)
```

**Examples:**
```python
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

x = torch.randn(32, 784, requires_grad=True)
# Checkpoint in 4 segments
output = cp.checkpoint_sequential(model, 4, x, use_reentrant=False)
```

**Memory savings:**
- Without checkpointing: All intermediate activations stored -> O(n) memory for n layers.
- With checkpointing: Only activations at segment boundaries stored -> O(sqrt(n)) memory (approximately).
- Trade-off: Activations are recomputed during backward, increasing compute time by ~30%.

---

## 4.12 Profiling Autograd

### 4.12.1 torch.autograd.profiler

```python
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    # Code to profile
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# Export as Chrome trace
prof.export_chrome_trace("trace.json")
```

### 4.12.2 torch.profiler (Modern API)

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for step, (input, target) in enumerate(dataloader):
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()  # Advance the profiler schedule
```

---

## 4.13 Functional API: torch.func (formerly functorch)

`torch.func` provides JAX-like function transforms that work on pure functions (no `nn.Module` state).

### 4.13.1 torch.func.grad

Transforms a function to return its gradient.

```python
from torch.func import grad

def f(x):
    return x ** 2 + 3 * x + 1

# Create gradient function
df = grad(f)
x = torch.tensor(2.0)
print(df(x))                    # tensor(7.) = 2*2 + 3

# Second derivative
d2f = grad(grad(f))
print(d2f(x))                   # tensor(2.)

# With respect to specific arguments
def g(x, y):
    return (x * y).sum()

dg_dx = grad(g, argnums=0)
dg_dy = grad(g, argnums=1)
```

### 4.13.2 torch.func.grad_and_value

Returns a function that computes both the gradient and the value.

```python
from torch.func import grad_and_value

def f(x):
    return x ** 2

fn = grad_and_value(f)
x = torch.tensor(3.0)
gradient, value = fn(x)
# gradient: tensor(6.), value: tensor(9.)
```

### 4.13.3 torch.func.vmap

Vectorizes a function over a batch dimension (automatic batching).

```python
from torch.func import vmap

def f(x):
    return x ** 2 + 3 * x

# Apply f to each element of the batch
batched_f = vmap(f)
x = torch.randn(100)
result = batched_f(x)           # Equivalent to: x ** 2 + 3 * x

# More useful: batch over a function with weights
def predict(weight, x):
    return weight @ x

weight = torch.randn(10, 5)
batched_predict = vmap(predict, in_dims=(None, 0))
batch_x = torch.randn(64, 5)
result = batched_predict(weight, batch_x)  # shape: (64, 10)

# Nested vmap
def f(x, y):
    return x * y

batched_f = vmap(vmap(f))
x = torch.randn(3, 4)
y = torch.randn(3, 4)
result = batched_f(x, y)        # shape: (3, 4)

# in_dims: specify which dimensions to vectorize
# in_dims=0: batch along first dimension (default)
# in_dims=(0, None): batch first arg, don't batch second
# in_dims=-1: batch along last dimension
```

### 4.13.4 torch.func.jacfwd / torch.func.jacrev

Compute the Jacobian matrix using forward-mode or reverse-mode AD.

```python
from torch.func import jacfwd, jacrev

def f(x):
    return torch.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])

x = torch.tensor([1.0, 2.0])

# Reverse-mode Jacobian (efficient when output dim > input dim)
J_rev = jacrev(f)(x)
# tensor([[2., 0.],
#         [2., 1.],
#         [0., 4.]])

# Forward-mode Jacobian (efficient when input dim > output dim)
J_fwd = jacfwd(f)(x)
# Same result

# Hessian = jacrev(jacrev(f)) or jacfwd(jacrev(f))
from torch.func import hessian

def g(x):
    return (x ** 2).sum()

H = hessian(g)(torch.randn(3))
# Diagonal matrix: diag([2, 2, 2])
```

### 4.13.5 torch.func.hessian

Computes the Hessian matrix.

```python
from torch.func import hessian

def f(x):
    return (x ** 3).sum()

x = torch.tensor([1.0, 2.0, 3.0])
H = hessian(f)(x)
# tensor([[6., 0., 0.],
#         [0., 12., 0.],
#         [0., 0., 18.]])
```

### 4.13.6 torch.func.jvp / torch.func.vjp

Low-level Jacobian-vector and vector-Jacobian products.

```python
from torch.func import jvp, vjp

def f(x):
    return x ** 2

x = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([1.0, 1.0, 1.0])

# Jacobian-vector product (forward-mode)
value, jvp_result = jvp(f, (x,), (v,))
# value: f(x) = [1, 4, 9]
# jvp_result: J @ v = [2, 4, 6]

# Vector-Jacobian product (reverse-mode)
value, vjp_fn = vjp(f, x)
vjp_result = vjp_fn(torch.tensor([1.0, 1.0, 1.0]))
# vjp_result: (tensor([2., 4., 6.]),) = v^T @ J
```

### 4.13.7 torch.func.functional_call

Call an `nn.Module` with specified parameter/buffer overrides.

```python
from torch.func import functional_call

model = torch.nn.Linear(5, 3)
x = torch.randn(5)
params = dict(model.named_parameters())

# Call with explicit parameters (no module state mutation)
output = functional_call(model, params, x)

# Useful with vmap for ensemble models
def predict(params, x):
    return functional_call(model, params, x)

# Batch over multiple parameter sets
ensemble_params = {k: v.unsqueeze(0).repeat(10, 1, 1) for k, v in params.items()}
batched_predict = vmap(predict, in_dims=(0, None))
results = batched_predict(ensemble_params, x)  # shape: (10, 3)
```

### 4.13.8 Summary of torch.func Transforms

| Transform | Description | Mode |
|-----------|------------|------|
| `grad(f)` | Gradient (partial derivatives) | Reverse |
| `grad_and_value(f)` | Gradient + function value | Reverse |
| `vmap(f)` | Automatic vectorization/batching | -- |
| `jacrev(f)` | Jacobian via reverse-mode | Reverse |
| `jacfwd(f)` | Jacobian via forward-mode | Forward |
| `hessian(f)` | Hessian (2nd derivatives) | Mixed |
| `jvp(f)` | Jacobian-vector product | Forward |
| `vjp(f)` | Vector-Jacobian product | Reverse |

---

## 4.14 Autograd Engine Internals

### 4.14.1 Execution Engine

The autograd engine (in C++, `torch/csrc/autograd/engine.cpp`) is responsible for executing the backward pass:

1. **Graph construction**: During forward, each operation creates a `Node` (in C++) that stores:
   - The backward function
   - Saved tensors (via `save_for_backward`)
   - Edges to parent nodes (`next_functions_`)

2. **Topological sort**: The engine performs a topological sort of the computation graph to determine execution order.

3. **Task-based parallelism**: The engine uses a thread pool to parallelize independent backward tasks:
   - Each `Node` becomes a `NodeTask`
   - Tasks are added to a `ReadyQueue` when all their dependencies are satisfied
   - Worker threads pick up tasks and execute them

4. **Gradient accumulation**: Gradients are accumulated atomically in `accumulate_grad` nodes (one per leaf tensor).

### 4.14.2 Gradient Accumulation

```python
# Gradient accumulation over multiple mini-batches
optimizer.zero_grad()

for i, (input, target) in enumerate(dataloader):
    output = model(input)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4.14.3 Gradient Hooks

```python
# Register hook on a tensor
x = torch.randn(3, requires_grad=True)
y = x ** 2

def grad_hook(grad):
    print(f"Gradient: {grad}")
    return grad * 2  # Can modify gradient

y.register_hook(grad_hook)

# Register hook on a module
def module_hook(module, grad_input, grad_output):
    print(f"Module: {module.__class__.__name__}")
    return grad_input

model.linear.register_full_backward_hook(module_hook)
```

---

## 4.15 Summary

PyTorch's autograd system provides:

1. **Automatic differentiation** via `backward()` and `grad()` for computing gradients without manual derivation.
2. **Custom functions** via `torch.autograd.Function` for defining operations with custom backward passes.
3. **Gradient modes** (`no_grad`, `inference_mode`, `enable_grad`) for controlling gradient computation.
4. **Forward-mode AD** via `torch.autograd.forward_ad` and `torch.func.jvp`.
5. **Gradient checking** (`gradcheck`, `gradgradcheck`) for verifying gradient implementations.
6. **Anomaly detection** for debugging NaN/Inf in backward.
7. **Gradient checkpointing** for memory-efficient training.
8. **torch.func** transforms (`vmap`, `grad`, `jacrev`, `jacfwd`, `hessian`) for JAX-like functional programming.
9. **Profiling** tools for analyzing backward pass performance.
10. **Compiled autograd** for JIT-compiling the backward pass.
