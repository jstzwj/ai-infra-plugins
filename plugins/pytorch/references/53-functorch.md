# PyTorch - Chapter 53: Function Transforms (torch.func / functorch)

This reference covers JAX-like function transforms: vmap, grad, jacfwd, jacrev, hessian.

---

## 53.1 vmap (Vectorization)

```python
from torch.func import vmap

def f(x):
    return x ** 2

# Vectorize over dimension 0
batched_f = vmap(f)
output = batched_f(torch.randn(64, 10))  # (64, 10)

# Control which dimension to vectorize
vmap(f, in_dims=1)(torch.randn(10, 64))  # Vectorize over dim 1

# Multiple inputs
vmap(lambda x, y: x + y, in_dims=(0, 0))(torch.randn(64, 10), torch.randn(64, 10))
```

---

## 53.2 grad

```python
from torch.func import grad

def f(x):
    return (x ** 2).sum()

grad_f = grad(f)
g = grad_f(torch.randn(3))  # 2 * x

# With auxiliary data
from torch.func import grad_and_value
grad_fn, value = grad_and_value(f)(torch.randn(3))
```

---

## 53.3 Jacobian

```python
from torch.func import jacfwd, jacrev

def f(x):
    return x ** 2

# Forward-mode (faster for tall Jacobians: more outputs than inputs)
J = jacfwd(f)(torch.randn(3))  # (3, 3) diagonal matrix

# Reverse-mode (faster for wide Jacobians: more inputs than outputs)
J = jacrev(f)(torch.randn(3))  # Same result
```

---

## 53.4 Hessian

```python
from torch.func import hessian

def f(x):
    return (x ** 3).sum()

H = hessian(f)(torch.randn(3))  # (3, 3) Hessian matrix

# Efficient: jacrev(jacfwd(f))
```

---

## 53.5 jvp / vjp

```python
from torch.func import jvp, vjp

def f(x):
    return x ** 2

# Jacobian-vector product (forward-mode)
tangent = torch.ones(3)
primal, jvp_val = jvp(f, (torch.randn(3),), (tangent,))

# Vector-Jacobian product (reverse-mode)
cotangent = torch.ones(3)
primal, vjp_fn = vjp(f, torch.randn(3))
vjp_val = vjp_fn(cotangent)
```

---

## 53.6 Per-Sample Gradients

```python
from torch.func import grad, vmap, functional_call

def compute_loss(params, buffers, x, y):
    pred = functional_call(model, (params, buffers), x)
    return F.cross_entropy(pred, y)

# Per-sample gradients via vmap
sample_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(
    dict(model.named_parameters()),
    dict(model.named_buffers()),
    batch_x, batch_y
)
```
