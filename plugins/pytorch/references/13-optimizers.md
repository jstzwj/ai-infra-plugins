# PyTorch Optimizers - Comprehensive Reference

This chapter covers all optimizers available in `torch.optim`, their parameters, internals, and usage patterns.

## Table of Contents

1. [Optimizer Base Class](#optimizer-base-class)
2. [SGD](#sgd)
3. [Adam](#adam)
4. [AdamW](#adamw)
5. [RMSprop](#rmsprop)
6. [Adagrad](#adagrad)
7. [Adadelta](#adadelta)
8. [Adamax](#adamax)
9. [RAdam](#radam)
10. [NAdam](#nadam)
11. [LBFGS](#lbfgs)
12. [ASGD](#asgd)
13. [SparseAdam](#sparseadam)
14. [Optimizer Hooks](#optimizer-hooks)
15. [Parameter Groups](#parameter-groups)
16. [When to Use Which Optimizer](#when-to-use-which-optimizer)
17. [Fused Optimizer Variants for CUDA](#fused-optimizer-variants-for-cuda)

---

## Optimizer Base Class

All optimizers inherit from `torch.optim.Optimizer`.

### Constructor

```python
torch.optim.Optimizer(params, defaults)
```

**Parameters:**
- `params` (iterable or dict): An iterable of `torch.Tensor` or `dict`s. Specifies what Tensors should be optimized. If a dict, must contain a key `'params'` that maps to the parameter tensors, and may contain additional keys for optimizer-specific options.
- `defaults` (dict): A dict containing default values for optimization options (used as default for all parameter groups).

### Methods

#### `step(closure=None)`

Performs a single optimization step (parameter update).

```python
# Without closure (most optimizers)
optimizer.step()

# With closure (required for LBFGS, optional for others)
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    return loss
optimizer.step(closure)
```

**Parameters:**
- `closure` (callable, optional): A closure that reevaluates the model and returns the loss. Required for some optimizers like LBFGS. For others, it enables multiple evaluations of the function.

**Returns:** The loss value if `closure` is provided and the optimizer uses it, otherwise `None`.

#### `zero_grad(set_to_none=True)`

Sets the gradients of all optimized `torch.Tensor`s to zero.

```python
optimizer.zero_grad()
optimizer.zero_grad(set_to_none=True)  # Default, more memory efficient
optimizer.zero_grad(set_to_none=False)  # Sets gradients to zero tensors
```

**Parameters:**
- `set_to_none` (bool): Instead of setting gradients to zero, sets them to `None`. This generally reduces memory usage and can improve performance slightly because it avoids creating zero tensors. However, it changes some behaviors—for example, `grad` is `None` rather than a zero tensor, and `grad.is_sparse` may be `False` after `zero_grad(set_to_none=True)`.

#### `state_dict()`

Returns the state of the optimizer as a `dict`. Contains two entries:
- `state`: A dict mapping parameter IDs to their optimizer state (momentum buffers, etc.)
- `param_groups`: A list of all parameter groups with their hyperparameters

```python
state = optimizer.state_dict()
# {'state': {0: {'momentum_buffer': tensor(...)}, ...},
#  'param_groups': [{'lr': 0.01, 'momentum': 0.9, 'params': [0, 1, 2]}]}
```

#### `load_state_dict(state_dict)`

Loads the optimizer state from a previously saved state dict.

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, PATH)

checkpoint = torch.load(PATH)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Parameters:**
- `state_dict` (dict): The optimizer state dict (should be from a call to `state_dict()`).

**Behavior:** The optimizer state is loaded from the dict. If the optimizer was constructed with parameter groups, the loaded state must have the same structure. Parameters are matched by their order in the param_groups list.

#### `add_param_group(param_group)`

Adds a parameter group to the optimizer's `param_groups`.

```python
optimizer = torch.optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)

# Add a new parameter group later
optimizer.add_param_group({'params': model.new_layer.parameters(), 'lr': 5e-4})
```

**Parameters:**
- `param_group` (dict): Specifies the parameters to optimize and group-specific optimization options.

### Attributes

- `optimizer.param_groups` (list[dict]): A list of parameter groups. Each group is a dict containing `params` (list of tensors) plus optimizer-specific hyperparameters.
- `optimizer.state` (dict): A dict mapping parameter tensors to their optimizer-specific state (e.g., momentum buffers, running averages).
- `optimizer.defaults` (dict): Default optimizer hyperparameters.

---

## SGD

Stochastic Gradient Descent with optional momentum, weight decay, and Nesterov acceleration.

```python
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0,
                weight_decay=0, nesterov=False, *, maximize=False,
                foreach=None, differentiable=False, fused=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts defining parameter groups |
| `lr` | float | required | Learning rate. Must be positive. |
| `momentum` | float | 0 | Momentum factor. Accelerates SGD in the relevant direction and dampens oscillations. |
| `dampening` | float | 0 | Dampening for momentum. Prevents large momentum when combined with momentum. |
| `weight_decay` | float | 0 | Weight decay (L2 penalty). Encourages smaller weights. |
| `nesterov` | bool | False | Enables Nesterov momentum. Often provides better convergence than standard momentum. |
| `maximize` | bool | False | If True, maximizes the objective instead of minimizing. |
| `foreach` | bool | None | If True, uses the foreach implementation (applies optimizer step to all parameters at once). If None, uses the foreach implementation if available and CUDA is used. |
| `differentiable` | bool | False | If True, the optimizer step is differentiable with respect to the parameters (enables gradient computation through the optimizer step). |
| `fused` | bool | None | If True, uses a fused CUDA kernel implementation. Requires all parameters to be on CUDA. |

### Update Rules

**Without momentum:**
```
param = param - lr * grad
```

**With weight decay:**
```
grad = grad + weight_decay * param
param = param - lr * grad
```

**With momentum:**
```
buf = momentum * buf + grad + dampening * grad   # momentum buffer update
param = param - lr * buf
```

**With Nesterov momentum:**
```
buf = momentum * buf + grad + dampening * grad
param = param - lr * (grad + momentum * buf)
```

### Example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Basic SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum (most common configuration)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with momentum and weight decay
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            weight_decay=1e-4)

# SGD with Nesterov momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            nesterov=True)

# Fused CUDA implementation
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                            fused=True)

# Training loop
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## Adam

Adaptive Moment Estimation. Computes adaptive learning rates for each parameter using first and second moment estimates.

```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=0, amsgrad=False, *, maximize=False,
                 foreach=None, capturable=False, differentiable=False,
                 fused=None, decoupled_weight_decay=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts defining parameter groups |
| `lr` | float | 0.001 | Learning rate |
| `betas` | tuple(float, float) | (0.9, 0.999) | Coefficients for computing running averages of gradient and its square. `beta1` controls the first moment (mean) decay, `beta2` controls the second moment (uncentered variance) decay. |
| `eps` | float | 1e-8 | Term added to the denominator for numerical stability |
| `weight_decay` | float | 0 | Weight decay (L2 penalty) |
| `amsgrad` | bool | False | Whether to use the AMSGrad variant from "On the Convergence of Adam and Beyond" |
| `maximize` | bool | False | Maximize the objective instead of minimizing |
| `foreach` | bool | None | Use the foreach implementation |
| `capturable` | bool | False | Whether the optimizer instance can be captured in a CUDA graph. Requires `lr` to be a tensor. |
| `differentiable` | bool | False | If True, the optimizer step is differentiable |
| `fused` | bool | None | Use fused CUDA kernel. Requires all parameters on CUDA. |
| `decoupled_weight_decay` | bool | False | If True, uses decoupled weight decay (like AdamW). Available since PyTorch 2.6+ for combining Adam with decoupled weight decay without switching optimizer class. |

### Update Rule

The Adam update rule for each parameter `theta` at timestep `t`:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # First moment estimate
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         # Second moment estimate
m_hat_t = m_t / (1 - beta1^t)                         # Bias-corrected first moment
v_hat_t = v_t / (1 - beta2^t)                         # Bias-corrected second moment
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

### AMSGrad Variant

When `amsgrad=True`, maintains the maximum of all second moment estimates:

```
v_hat_max = max(v_hat_max, v_hat_t)
theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_max) + eps)
```

### State

Each parameter stores in `optimizer.state`:
- `step`: Number of optimizer steps taken (int)
- `exp_avg`: First moment estimate (tensor same shape as param)
- `exp_avg_sq`: Second moment estimate (tensor same shape as param)
- `max_exp_avg_sq`: Maximum of second moment estimates (only when `amsgrad=True`)

### Example

```python
# Basic Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Adam with custom betas and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                             betas=(0.9, 0.999), weight_decay=0.01)

# Adam with AMSGrad (for better convergence in some settings)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

# Adam capturable for CUDA graphs
lr_tensor = torch.tensor(0.001, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=lr_tensor, capturable=True)

# Fused Adam for CUDA
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)

# Adam with decoupled weight decay (PyTorch 2.6+)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                             weight_decay=0.01, decoupled_weight_decay=True)
```

---

## AdamW

Adam with decoupled weight decay. The weight decay is applied directly to the weights rather than being added to the gradient.

```python
torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                  weight_decay=0.01, amsgrad=False, *, maximize=False,
                  foreach=None, capturable=False, differentiable=False,
                  fused=None)
```

### Parameters

Same as Adam, except `weight_decay` defaults to `0.01` (instead of `0` in Adam).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_decay` | float | 0.01 | Decoupled weight decay. Applied directly to weights: `param = param - lr * weight_decay * param` |

### Decoupled vs. L2 Weight Decay

**L2 regularization (Adam):**
```
grad = grad + weight_decay * param    # Added to gradient
m_t = beta1 * m + (1 - beta1) * grad  # Moment uses modified gradient
v_t = beta2 * v + (1 - beta2) * grad^2
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

**Decoupled weight decay (AdamW):**
```
m_t = beta1 * m + (1 - beta1) * grad  # Moment uses original gradient
v_t = beta2 * v + (1 - beta2) * grad^2
param = param - lr * m_hat / (sqrt(v_hat) + eps)
param = param - lr * weight_decay * param  # Weight decay applied separately
```

The decoupled approach ensures that weight decay is invariant to the gradient scaling from the adaptive learning rate. This is the preferred approach for training transformers.

### Example

```python
# Standard AdamW (default for transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# AdamW with custom settings for fine-tuning
optimizer = torch.optim.AdamW([
    {'params': model.base.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3},
], weight_decay=0.01, betas=(0.9, 0.999))

# Fused AdamW for CUDA
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                              weight_decay=0.01, fused=True)

# AdamW with AMSGrad
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                              weight_decay=0.01, amsgrad=True)
```

---

## RMSprop

Root Mean Square Propagation. Uses a moving average of squared gradients to normalize the gradient.

```python
torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08,
                    weight_decay=0, momentum=0, centered=False,
                    *, maximize=False, foreach=None, differentiable=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.01 | Learning rate |
| `alpha` | float | 0.99 | Smoothing constant (moving average decay for squared gradients) |
| `eps` | float | 1e-8 | Term added to denominator for numerical stability |
| `weight_decay` | float | 0 | Weight decay (L2 penalty) |
| `momentum` | float | 0 | Momentum factor |
| `centered` | bool | False | If True, compute the centered RMSProp (normalizes by estimated variance instead of raw second moment). Can help convergence. |
| `maximize` | bool | False | Maximize instead of minimize |
| `foreach` | bool | None | Use foreach implementation |
| `differentiable` | bool | False | If True, optimizer step is differentiable |

### Update Rule

**Standard RMSprop:**
```
v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
param = param - lr * grad / (sqrt(v_t) + eps)
```

**With momentum:**
```
v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
buf = momentum * buf + lr * grad / (sqrt(v_t) + eps)
param = param - buf
```

**Centered RMSprop:**
```
v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
m_t = alpha * m_{t-1} + (1 - alpha) * grad
v_hat_t = v_t - m_t^2
param = param - lr * grad / (sqrt(v_hat_t) + eps)
```

### Example

```python
# Basic RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

# RMSprop with momentum (common for RL)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00025,
                                alpha=0.99, momentum=0.95, eps=0.01)

# Centered RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, centered=True)
```

---

## Adagrad

Adaptive Gradient Algorithm. Adapts the learning rate to each parameter by dividing by the square root of the sum of all historical squared gradients.

```python
torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0,
                    initial_accumulator_value=0, eps=1e-10,
                    *, foreach=None, maximize=False, differentiable=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.01 | Learning rate |
| `lr_decay` | float | 0 | Learning rate decay |
| `weight_decay` | float | 0 | Weight decay (L2 penalty) |
| `initial_accumulator_value` | float | 0 | Initial value for the squared gradient accumulator |
| `eps` | float | 1e-10 | Term added to denominator for numerical stability |
| `foreach` | bool | None | Use foreach implementation |
| `maximize` | bool | False | Maximize instead of minimize |
| `differentiable` | bool | False | If True, optimizer step is differentiable |

### Update Rule

```
state_sum = state_sum + grad^2
param = param - lr * grad / (sqrt(state_sum) + eps)
```

The effective learning rate for each parameter decreases over time as the accumulator grows. This means parameters with large gradients get smaller learning rates, and parameters with small gradients get larger learning rates.

**Note:** The learning rate effectively decreases monotonically, which can cause training to stall. Adagrad works well for sparse data (e.g., NLP with rare words) but may converge prematurely on dense problems.

### Example

```python
# Basic Adagrad
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

# Adagrad for sparse features
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1,
                                lr_decay=1e-4, weight_decay=1e-4)

# With initial accumulator value (can help with stability)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01,
                                initial_accumulator_value=0.1)
```

---

## Adadelta

An extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Uses a window of gradient updates rather than accumulating all past squared gradients.

```python
torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06,
                     weight_decay=0, *, foreach=None, maximize=False,
                     differentiable=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 1.0 | Learning rate. Note the default is 1.0 (not 0.01) because Adadelta is self-adjusting. |
| `rho` | float | 0.9 | Coefficient used for computing a running average of squared gradients (decay rate) |
| `eps` | float | 1e-6 | Term added to denominator for numerical stability |
| `weight_decay` | float | 0 | Weight decay (L2 penalty) |
| `foreach` | bool | None | Use foreach implementation |
| `maximize` | bool | False | Maximize instead of minimize |
| `differentiable` | bool | False | If True, optimizer step is differentiable |

### Update Rule

```
E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2           # Running avg of squared gradients
delta_x = -sqrt(E[delta_x^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps) * g_t
E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * delta_x^2  # Running avg of squared updates
param = param + lr * delta_x
```

### Example

```python
optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
```

---

## Adamax

A variant of Adam based on the infinity norm. Generalizes Adam to the L_p norm with p -> infinity.

```python
torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08,
                   weight_decay=0, *, foreach=None, maximize=False,
                   differentiable=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.002 | Learning rate |
| `betas` | tuple(float, float) | (0.9, 0.999) | Coefficients for first moment and infinity norm |
| `eps` | float | 1e-8 | Term added to denominator for numerical stability |
| `weight_decay` | float | 0 | Weight decay (L2 penalty) |
| `foreach` | bool | None | Use foreach implementation |
| `maximize` | bool | False | Maximize instead of minimize |
| `differentiable` | bool | False | If True, optimizer step is differentiable |

### Update Rule

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
u_t = max(beta2 * u_{t-1}, |g_t|)       # Infinity norm estimate
m_hat_t = m_t / (1 - beta1^t)
param = param - lr * m_hat_t / (u_t + eps)
```

Note: Adamax does not need bias correction for the second moment because `u_t` is not a moving average.

### Example

```python
optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999))
```

---

## RAdam

Rectified Adam. A variant of Adam that addresses the variance of the adaptive learning rate during the early stages of training by automatically turning on/off the adaptive learning rate.

```python
torch.optim.RAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                  weight_decay=0, *, foreach=None, maximize=False,
                  differentiable=False, decoupled_weight_decay=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.001 | Learning rate |
| `betas` | tuple(float, float) | (0.9, 0.999) | Coefficients for running averages |
| `eps` | float | 1e-8 | Numerical stability term |
| `weight_decay` | float | 0 | Weight decay |
| `foreach` | bool | None | Use foreach implementation |
| `maximize` | bool | False | Maximize instead of minimize |
| `differentiable` | bool | False | If True, optimizer step is differentiable |
| `decoupled_weight_decay` | bool | False | If True, uses decoupled weight decay |

### Variance Rectification

RAdam computes a "maximum length of the approximated SMA" (`N_sma_max`). Based on the current SMA length:

- If the SMA length is below a threshold (early training), it falls back to a warmup phase using SGD-like updates.
- If the SMA length exceeds the threshold, it uses the full Adam update with a variance rectification term.

This eliminates the need for manual warmup in many cases.

### Example

```python
# RAdam without warmup (often works without explicit warmup)
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)

# RAdam with decoupled weight decay
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3,
                              weight_decay=0.01, decoupled_weight_decay=True)
```

---

## NAdam

Nesterov-accelerated Adam. Combines Adam with Nesterov momentum for better convergence.

```python
torch.optim.NAdam(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08,
                  weight_decay=0, *, momentum_decay=4e-3, foreach=None,
                  maximize=False, differentiable=False,
                  decoupled_weight_decay=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.002 | Learning rate |
| `betas` | tuple(float, float) | (0.9, 0.999) | Coefficients for first and second moment |
| `eps` | float | 1e-8 | Numerical stability term |
| `weight_decay` | float | 0 | Weight decay |
| `momentum_decay` | float | 4e-3 | Momentum decay for the Nesterov momentum term |
| `foreach` | bool | None | Use foreach implementation |
| `maximize` | bool | False | Maximize instead of minimize |
| `differentiable` | bool | False | If True, optimizer step is differentiable |
| `decoupled_weight_decay` | bool | False | If True, uses decoupled weight decay |

### Update Rule

NAdam modifies Adam to use Nesterov momentum by looking ahead in the gradient computation:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat_t = m_t / (1 - beta1^t)
v_hat_t = v_t / (1 - beta2^t)
# Nesterov lookahead: use the future position's momentum
m_nesterov = beta1 * m_hat_t + (1 - beta1) * g_t / (1 - beta1^t)
param = param - lr * m_nesterov / (sqrt(v_hat_t) + eps)
```

### Example

```python
optimizer = torch.optim.NAdam(model.parameters(), lr=0.002,
                              betas=(0.9, 0.999), momentum_decay=4e-3)

# NAdam with decoupled weight decay
optimizer = torch.optim.NAdam(model.parameters(), lr=2e-3,
                              weight_decay=0.01, decoupled_weight_decay=True)
```

---

## LBFGS

Limited-memory BFGS (Broyden-Fletcher-Goldfarb-Shanno). A quasi-Newton method that approximates the inverse Hessian matrix using a limited history of updates.

```python
torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None,
                  tolerance_grad=1e-07, tolerance_change=1e-09,
                  history_size=100, line_search_fn=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 1 | Learning rate (used as initial step size in line search) |
| `max_iter` | int | 20 | Max iterations per optimization step |
| `max_eval` | int | None | Max evaluations per optimization step. If None, defaults to `max_iter * 1.25`. |
| `tolerance_grad` | float | 1e-7 | Termination tolerance on first order optimality (gradient norm) |
| `tolerance_change` | float | 1e-9 | Termination tolerance on function value/parameter changes |
| `history_size` | int | 100 | Update history size for the Hessian approximation |
| `line_search_fn` | str | None | Line search strategy. Either None (no line search, uses fixed `lr`) or "strong_wolfe". |

### Important Notes

- **LBFGS requires a closure** that reevaluates the model and returns the loss.
- It is a full-batch optimizer and should not be used with mini-batches in the typical sense. It works best when you can compute the loss over the full dataset.
- LBFGS is well-suited for small models and full-batch optimization problems.
- It is second-order (approximates Hessian), so it can converge faster than first-order methods in fewer iterations.

### Example

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20,
                              history_size=100, line_search_fn="strong_wolfe")

for epoch in range(num_epochs):
    def closure():
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
```

---

## ASGD

Averaged Stochastic Gradient Descent. Maintains a running average of the parameters which can provide better generalization.

```python
torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0,
                 weight_decay=0, *, foreach=None, maximize=False,
                 differentiable=False)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.01 | Learning rate |
| `lambd` | float | 0.0001 | Decay term |
| `alpha` | float | 0.75 | Power for eta update |
| `t0` | float | 1e6 | Point at which averaging begins |
| `weight_decay` | float | 0 | Weight decay |
| `foreach` | bool | None | Use foreach implementation |
| `maximize` | bool | False | Maximize instead of minimize |
| `differentiable` | bool | False | If True, optimizer step is differentiable |

### Update Rule

```
eta = lr / ((1 + lambd * lr * t)^alpha)
mu = 1 / max(1, t - t0)
ax = (1 - mu) * ax + mu * param
param = param - eta * grad
```

The averaged parameters (`ax`) are what you typically use at inference time.

### Example

```python
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001,
                             alpha=0.75, t0=1e6)
```

---

## SparseAdam

A lazy version of Adam suitable for sparse gradients (as produced by `nn.Embedding` with sparse gradients). It only updates parameters that have non-zero gradients in the current step.

```python
torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | required | Iterable of parameters or dicts |
| `lr` | float | 0.001 | Learning rate |
| `betas` | tuple(float, float) | (0.9, 0.999) | Adam beta coefficients |
| `eps` | float | 1e-8 | Numerical stability term |

### Usage Notes

- Designed specifically for sparse gradient tensors (where `grad.is_sparse == True`).
- Only parameters that received a gradient update in the current step are modified.
- The implementation is "lazy" in that it only allocates state for parameters that have been updated.
- Most commonly used with `nn.Embedding(sparse=True)`.

### Example

```python
embedding = nn.Embedding(100000, 128, sparse=True)
optimizer = torch.optim.SparseAdam(embedding.parameters(), lr=0.001)

# Training loop with sparse gradients
for batch in dataloader:
    optimizer.zero_grad()
    output = embedding(batch)
    loss = compute_loss(output)
    loss.backward()
    optimizer.step()  # Only updates accessed embedding rows
```

---

## Optimizer Hooks

Optimizers support step hooks that allow running custom code before or after each optimizer step.

### `register_step_pre_hook(hook)`

Registers a hook to be called before the optimizer step.

```python
def pre_step_hook(optimizer, args, kwargs):
    # Called before each optimizer.step()
    # Can modify args/kwargs, log, etc.
    current_lr = optimizer.param_groups[0]['lr']
    print(f"About to step with lr={current_lr}")
    return args, kwargs

handle = optimizer.register_step_pre_hook(pre_step_hook)
# Later: handle.remove() to unregister
```

### `register_step_post_hook(hook)`

Registers a hook to be called after the optimizer step.

```python
def post_step_hook(optimizer, args, kwargs):
    # Called after each optimizer.step()
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                print(f"Gradient norm: {grad_norm}")

handle = optimizer.register_step_post_hook(post_step_hook)
```

### Using Hooks for Gradient Clipping

```python
def gradient_clip_hook(optimizer, args, kwargs):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    return args, kwargs

optimizer.register_step_pre_hook(gradient_clip_hook)
```

---

## Parameter Groups

Parameter groups allow different optimizer settings for different parts of the model.

### Creating Multiple Parameter Groups

```python
# Different learning rates for different layers
optimizer = torch.optim.AdamW([
    {
        'params': model.embeddings.parameters(),
        'lr': 1e-4,
        'weight_decay': 0.0  # No weight decay for embeddings
    },
    {
        'params': model.transformer.parameters(),
        'lr': 5e-5,
        'weight_decay': 0.01
    },
    {
        'params': model.head.parameters(),
        'lr': 1e-3,
        'weight_decay': 0.01
    },
], lr=1e-3)  # Default lr for any group that doesn't specify

# Freeze some parameters by excluding them
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
```

### Accessing Parameter Groups

```python
# Access param groups
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}: lr={group['lr']}, {len(group['params'])} params")

# Modify learning rate of a specific group
optimizer.param_groups[0]['lr'] = 1e-4

# Iterate over all parameters with their group settings
for group in optimizer.param_groups:
    for param in group['params']:
        if param.grad is not None:
            print(f"Param shape: {param.shape}, lr: {group['lr']}")
```

### Dynamic Parameter Groups

```python
# Start with base parameters
optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01, momentum=0.9)

# Add new parameters during training (e.g., after unfreezing)
optimizer.add_param_group({
    'params': model.newly_unfrozen.parameters(),
    'lr': 0.001,
    'momentum': 0.9
})
```

---

## When to Use Which Optimizer

### Decision Guide

| Optimizer | Best For | Pros | Cons |
|-----------|----------|------|------|
| **SGD + Momentum** | CNNs, large-scale vision | Good generalization, well-understood | Needs careful lr tuning, slow convergence |
| **SGD + Nesterov** | CNNs, computer vision | Slightly better than standard momentum | Same as SGD |
| **Adam** | NLP, GANs, RL, general purpose | Fast convergence, robust to hyperparameters | May generalize worse than SGD |
| **AdamW** | Transformers, BERT, GPT | Decoupled weight decay, state of the art for transformers | Slightly more memory than SGD |
| **RAdam** | When you want Adam without warmup | Automatic warmup, more stable early training | Slightly slower than Adam |
| **NAdam** | When you want Adam + Nesterov | Better convergence in some cases | Marginal improvement over Adam |
| **RMSprop** | RL, RNNs | Good for non-stationary objectives | Can be sensitive to hyperparameters |
| **Adagrad** | Sparse data, NLP | Per-parameter adaptive lr | Learning rate decreases too aggressively |
| **Adadelta** | When you want improved Adagrad | No need to set global learning rate | Rarely the best choice |
| **Adamax** | When Adam is unstable | More stable than Adam in some cases | Less common, fewer best practices |
| **LBFGS** | Small models, full-batch | Second-order, fast convergence | Memory intensive, requires full batch |
| **ASGD** | When averaging helps generalization | Better generalization | More hyperparameters |
| **SparseAdam** | Sparse embeddings | Efficient for sparse updates | Only for sparse gradients |

### Common Recommendations

1. **Transformer models (BERT, GPT, etc.)**: AdamW with weight_decay=0.01, betas=(0.9, 0.999), lr=1e-5 to 5e-5
2. **ResNet/Vision models**: SGD with momentum=0.9, weight_decay=1e-4, lr=0.1 (with decay schedule)
3. **GANs**: Adam with lr=0.0002, betas=(0.5, 0.999)
4. **Reinforcement Learning**: Adam or RMSprop with low learning rates
5. **Fine-tuning**: AdamW with lower lr for pre-trained layers, higher lr for new layers
6. **Large-scale distributed training**: Fused AdamW or SGD for maximum throughput

---

## Fused Optimizer Variants for CUDA

Fused optimizers combine multiple operations into a single CUDA kernel, reducing kernel launch overhead and improving GPU utilization.

### Using Fused Optimizers

```python
# All parameters must be on CUDA
model = model.cuda()

# Fused SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            fused=True)

# Fused Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)

# Fused AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01,
                              fused=True)
```

### Fused Optimizer Requirements

1. All parameters must be on CUDA
2. All parameters in the same group must be on the same CUDA device
3. If `foreach=None` and CUDA is available, `fused=True` may be automatically selected

### Performance Comparison

```python
import time

# Standard optimizer
optimizer_standard = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Fused optimizer
optimizer_fused = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

# Benchmark
for name, opt in [("Standard", optimizer_standard), ("Fused", optimizer_fused)]:
    times = []
    for _ in range(100):
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model(inputs).sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    print(f"{name}: {sum(times)/len(times)*1000:.2f} ms/step")
```

### NVIDIA Apex Fused Optimizers (Alternative)

For environments using NVIDIA Apex:

```python
from apex.optimizers import FusedAdam
from apex.optimizers import FusedSGD
from apex.optimizers import FusedLAMB

optimizer = FusedAdam(model.parameters(), lr=1e-3, adam_w_mode=True)
```

### torch.optim with CUDA Graphs

For maximum performance with CUDA graphs, use `capturable=True`:

```python
# Step 1: Create optimizer with capturable=True and tensor lr
lr = torch.tensor(0.001, device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, capturable=True)

# Step 2: Warmup (required before capturing)
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs)
    loss.backward()
    optimizer.step()

# Step 3: Capture CUDA graph
static_inputs = inputs.clone()
static_loss = torch.zeros(1, device='cuda')

def run_step():
    optimizer.zero_grad(set_to_none=True)
    static_loss = model(static_inputs).sum()
    static_loss.backward()
    optimizer.step()

# Warmup before graph capture
for _ in range(3):
    run_step()

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    run_step()

# Replay (much faster)
for data in dataloader:
    static_inputs.copy_(data)
    g.replay()
```

---

## Complete Training Loop Example

A complete training loop demonstrating optimizer usage with learning rate scheduling:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, train_dataset, val_dataset, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Split parameters for different weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.01},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=1e-3, betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping (alternative: use pre-step hook)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }, 'best_model.pt')

    return model
```

---

## Optimizer State Management

### Saving and Loading

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']

# Load to different device (e.g., CPU checkpoint to GPU)
checkpoint = torch.load('checkpoint.pt', map_location='cuda:0')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# State tensors are loaded to the device where parameters currently are
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to('cuda:0')
```

### Memory Considerations

The optimizer state can consume significant memory, especially for adaptive optimizers:

- **SGD with momentum**: 1 extra tensor per parameter (momentum buffer) = same size as parameters
- **Adam/AdamW**: 2 extra tensors per parameter (exp_avg + exp_avg_sq) = 2x parameter memory
- **Adam with AMSGrad**: 3 extra tensors per parameter = 3x parameter memory

```python
# Estimate optimizer memory usage
def estimate_optimizer_memory(model, optimizer_type='adam'):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())

    if optimizer_type == 'sgd':
        state_multiplier = 1  # momentum buffer
    elif optimizer_type == 'adam':
        state_multiplier = 2  # exp_avg + exp_avg_sq
    elif optimizer_type == 'adam_amsgrad':
        state_multiplier = 3  # + max_exp_avg_sq

    state_size = param_size * state_multiplier
    print(f"Parameters: {param_size / 1e9:.2f} GB")
    print(f"Optimizer state: {state_size / 1e9:.2f} GB")
    print(f"Total (params + grads + state): {(param_size * 2 + state_size) / 1e9:.2f} GB")
```

### Gradient Accumulation Pattern

When effective batch size exceeds GPU memory, accumulate gradients over multiple mini-batches:

```python
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```
