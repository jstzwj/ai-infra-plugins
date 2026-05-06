# PyTorch Learning Rate Schedulers - Comprehensive Reference

This chapter covers all learning rate schedulers in `torch.optim.lr_scheduler`, their parameters, behaviors, and usage patterns.

## Table of Contents

1. [LRScheduler Base Class](#lrscheduler-base-class)
2. [LambdaLR](#lambdalr)
3. [MultiplicativeLR](#multiplicativelr)
4. [StepLR](#steplr)
5. [MultiStepLR](#multisteplr)
6. [ConstantLR](#constantlr)
7. [LinearLR](#linearlr)
8. [ExponentialLR](#exponentiallr)
9. [CosineAnnealingLR](#cosineannealingelr)
10. [ChainedScheduler](#chainedscheduler)
11. [SequentialLR](#sequentiallr)
12. [PolynomialLR](#polynomiallr)
13. [CosineAnnealingWarmRestarts](#cosineannealingewarmrestarts)
14. [OneCycleLR](#onecyclelr)
15. [CyclicLR](#cycliclr)
16. [ReduceLROnPlateau](#reducelronplateau)
17. [Warm-Up Strategies](#warm-up-strategies)
18. [Combining Schedulers](#combining-schedulers)
19. [Scheduler State Management](#scheduler-state-management)

---

## LRScheduler Base Class

All schedulers (except `ReduceLROnPlateau`) inherit from `torch.optim.lr_scheduler.LRScheduler`.

### Constructor

```python
torch.optim.lr_scheduler.LRScheduler(optimizer, last_epoch=-1, verbose='deprecated')
```

**Parameters:**
- `optimizer` (Optimizer): The optimizer whose learning rate should be scheduled.
- `last_epoch` (int): The index of the last epoch. Used when resuming training. Setting this to a positive value loads the scheduler state as if it had already been called `last_epoch` times. Default: -1 (starts fresh).
- `verbose` (str): Deprecated. Previously controlled printing of learning rate at each step.

### Methods

#### `step(epoch=None)`

Updates the learning rate based on the current epoch or step count.

```python
# Called after each epoch (epoch-based schedulers)
scheduler.step()

# Called after each batch (batch-based schedulers like OneCycleLR, CyclicLR)
scheduler.step()

# Explicit epoch number (for resuming)
scheduler.step(epoch=10)
```

**Important:** For epoch-based schedulers, call `scheduler.step()` after `optimizer.step()` in the epoch loop. For batch-based schedulers, call it after each batch.

#### `get_lr()`

Returns the current learning rate for each parameter group. Override in subclasses to implement the scheduling logic.

```python
current_lrs = scheduler.get_lr()
# Returns a list of learning rates, one per parameter group
```

#### `get_last_lr()`

Returns the learning rates that were last computed by the scheduler.

```python
last_lrs = scheduler.get_last_lr()
```

#### `state_dict()`

Returns the state of the scheduler as a `dict`. Useful for checkpointing.

```python
state = scheduler.state_dict()
```

#### `load_state_dict(state_dict)`

Loads the scheduler state from a previously saved state dict.

```python
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

#### `print_lr(is_verbose, group, lr, epoch=None)`

Deprecated method. Previously used for printing the current learning rate.

### Key Behaviors

- The scheduler is initialized with `last_epoch=-1`, which sets the initial learning rates from the optimizer.
- Calling `step()` increments an internal counter and recomputes learning rates.
- Each scheduler maintains a `_step_count` and `base_lrs` (the initial learning rates).
- Learning rates are updated in-place on the optimizer's `param_groups`.

---

## LambdaLR

Sets the learning rate of each parameter group to the initial lr times a given function (lambda).

```python
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `lr_lambda` | function or list | required | A function which computes a multiplicative factor given an integer parameter epoch, or a list of such functions, one for each group in optimizer.param_groups. |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
lr = initial_lr * lr_lambda(epoch)
```

### Examples

```python
# Simple exponential decay
lambda1 = lambda epoch: 0.95 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# Warmup + cosine decay using lambda
import math

def warmup_cosine(epoch):
    warmup_epochs = 5
    total_epochs = 100
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

# Different lambda for each parameter group
lambda1 = lambda epoch: 0.95 ** epoch   # For first group
lambda2 = lambda epoch: 0.9 ** epoch    # For second group
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
```

### Common Lambda Functions

```python
# Step decay every 30 epochs
step_decay = lambda epoch: 0.1 ** (epoch // 30)

# Exponential decay
exp_decay = lambda epoch: 0.95 ** epoch

# Linear warmup over 10 epochs
linear_warmup = lambda epoch: min(1.0, epoch / 10)

# Custom polynomial decay
def poly_decay(epoch, power=0.9, max_epochs=100):
    return (1 - epoch / max_epochs) ** power

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda e: poly_decay(e, power=0.9, max_epochs=100)
)
```

---

## MultiplicativeLR

Multiplies the learning rate of each parameter group by a factor given by a function.

```python
torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose='deprecated')
```

### Parameters

Same as LambdaLR.

### Update Rule

Unlike LambdaLR which multiplies the *initial* lr, MultiplicativeLR multiplies the *current* lr:

```
lr_new = lr_current * lr_lambda(epoch)
```

### Example

```python
# Halve the learning rate every epoch
lambda1 = lambda epoch: 0.5
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)
# Epoch 0: lr * 0.5
# Epoch 1: lr * 0.5 * 0.5
# Epoch 2: lr * 0.5 * 0.5 * 0.5
```

---

## StepLR

Decays the learning rate by gamma every step_size epochs.

```python
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `step_size` | int | required | Period of learning rate decay (in epochs) |
| `gamma` | float | 0.1 | Multiplicative factor of learning rate decay |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
lr = initial_lr * gamma^(epoch // step_size)
```

### Example

```python
# Decay by 0.1 every 30 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Epoch 0-29: lr = 0.1
# Epoch 30-59: lr = 0.01
# Epoch 60-89: lr = 0.001

# Training loop
for epoch in range(90):
    train_one_epoch()
    validate()
    scheduler.step()
```

---

## MultiStepLR

Decays the learning rate at specific epoch milestones.

```python
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `milestones` | list | required | List of epoch indices. Must be increasing. |
| `gamma` | float | 0.1 | Multiplicative factor of learning rate decay |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
lr = initial_lr * gamma^(number of milestones passed)
```

### Example

```python
# Decay at epochs 30, 60, and 90
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)
# Epoch 0-29: lr = 0.1
# Epoch 30-59: lr = 0.01
# Epoch 60-89: lr = 0.001
# Epoch 90+: lr = 0.0001

# Common for ResNet training
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], gamma=0.1
)
```

---

## ConstantLR

Multiplies the learning rate by a constant factor until a specified number of epochs, then returns to the initial learning rate.

```python
torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0/3, total_iters=5, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `factor` | float | 1/3 | The number we multiply the learning rate until the milestone. |
| `total_iters` | int | 5 | The number of steps that the scheduler decays the learning rate. |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
if step < total_iters:
    lr = initial_lr * factor
else:
    lr = initial_lr
```

### Example

```python
# Use 1/3 of the learning rate for the first 5 epochs, then full lr
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0/3, total_iters=5)

# Use a warmup-like effect: start low and jump to full lr
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=10)
```

---

## LinearLR

Linearly adjusts the learning rate between two factors over a specified number of steps.

```python
torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/3, end_factor=1.0,
                                   total_iters=5, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `start_factor` | float | 1/3 | The number we multiply the learning rate at the start. |
| `end_factor` | float | 1.0 | The number we multiply the learning rate at the end. |
| `total_iters` | int | 5 | The number of steps over which to linearly interpolate. |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
if step < total_iters:
    factor = start_factor + (end_factor - start_factor) * step / total_iters
    lr = initial_lr * factor
else:
    lr = initial_lr * end_factor
```

### Example

```python
# Linear warmup over 10 epochs (start at 1/10 of lr, end at full lr)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
)

# Linear decay over 100 epochs
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.01, total_iters=100
)
```

---

## ExponentialLR

Decays the learning rate by gamma every epoch.

```python
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `gamma` | float | required | Multiplicative factor of learning rate decay per epoch |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
lr = initial_lr * gamma^epoch
```

### Example

```python
# 5% decay per epoch
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 1% decay per epoch (very gentle)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# 10% decay per epoch (more aggressive)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
```

---

## CosineAnnealingLR

Sets the learning rate using a cosine annealing schedule.

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `T_max` | int | required | Maximum number of iterations (half cycle length). The lr will go from initial to eta_min in T_max steps. |
| `eta_min` | float | 0 | Minimum learning rate |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
```

This produces a smooth cosine curve from `initial_lr` down to `eta_min` over `T_max` epochs.

### Example

```python
# Cosine decay over 100 epochs to minimum lr of 1e-6
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# With restart (manual): use T_max=50 for a half-cycle, then reset
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=0
)
```

---

## ChainedScheduler

Chains multiple schedulers together. Each scheduler is applied sequentially, so the output of one becomes the input of the next.

```python
torch.optim.lr_scheduler.ChainedScheduler(schedulers, optimizer=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schedulers` | list | required | List of schedulers to chain |
| `optimizer` | Optimizer | None | The optimizer. If None, uses the first scheduler's optimizer. |

### Behavior

When `step()` is called, each scheduler in the chain is called in order. The learning rate modifications are cumulative.

### Example

```python
# Warmup with LinearLR, then cosine decay
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=90, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])
```

---

## SequentialLR

Receives a list of schedulers and milestones that indicate when to switch between them.

```python
torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `schedulers` | list | required | List of schedulers to run sequentially |
| `milestones` | list | required | List of epoch indices at which to switch to the next scheduler. Length must be len(schedulers) - 1. |
| `last_epoch` | int | -1 | The index of last epoch |

### Example

```python
# Warmup for 10 epochs, then cosine decay for 90 epochs
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=10
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=90, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[10]
)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()
```

---

## PolynomialLR

Decays the learning rate using a polynomial function.

```python
torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `total_iters` | int | 5 | The number of steps that the scheduler decays the learning rate |
| `power` | float | 1.0 | The power of the polynomial |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
if step < total_iters:
    decay_factor = (1 - step / total_iters) ^ power
    lr = initial_lr * decay_factor
else:
    lr = 0  # or initial_lr depending on version
```

When `power=1.0`, this is a linear decay. When `power=2.0`, the decay is more aggressive at the end.

### Example

```python
# Polynomial decay with power=2 over 100 epochs
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=100, power=2.0
)

# Linear decay (power=1)
scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer, total_iters=100, power=1.0
)
```

---

## CosineAnnealingWarmRestarts

Cosine annealing with warm restarts (SGDR). Restarts the cosine schedule after each period.

```python
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0,
                                                       last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `T_0` | int | required | Number of iterations for the first restart |
| `T_mult` | int | 1 | A factor increases T_i after each restart. If T_mult=1, all restarts have the same length. If T_mult=2, each restart is twice as long as the previous. |
| `eta_min` | float | 0 | Minimum learning rate |
| `last_epoch` | int | -1 | The index of last epoch |

### Update Rule

```
T_i = T_0 * (T_mult ^ n_restarts)
lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * (epoch % T_i) / T_i)) / 2
```

### Example

```python
# Cosine annealing with restart every 20 epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=1, eta_min=1e-6
)
# Cycle 1: epochs 0-19, peak lr
# Cycle 2: epochs 20-39, peak lr
# Cycle 3: epochs 40-59, peak lr

# Increasing restart periods
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
# Cycle 1: epochs 0-9 (length 10)
# Cycle 2: epochs 10-29 (length 20)
# Cycle 3: epochs 30-69 (length 40)
```

### Batch-Level Restarts

```python
# CosineAnnealingWarmRestarts can also be called per batch
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=1, eta_min=1e-6
)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        train_step(data, target)
        scheduler.step(epoch + batch_idx / len(train_loader))
```

---

## OneCycleLR

The 1cycle learning rate policy. Changes the learning rate according to the 1cycle scheduling policy from "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates".

```python
torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None,
                                     steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos',
                                     cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                                     div_factor=25.0, final_div_factor=10000.0,
                                     three_phase=False, last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `max_lr` | float or list | required | Upper learning rate boundaries in the cycle. If a list, must have the same length as optimizer.param_groups. |
| `total_steps` | int | None | The total number of steps in the cycle. Must be provided if `epochs` and `steps_per_epoch` are not. |
| `epochs` | int | None | The number of epochs to train for. Used with `steps_per_epoch`. |
| `steps_per_epoch` | int | None | The number of steps (batches) per epoch. Used with `epochs`. |
| `pct_start` | float | 0.3 | The percentage of the cycle (in number of steps) spent increasing the learning rate. |
| `anneal_strategy` | str | 'cos' | Specifies the annealing strategy: 'cos' for cosine, 'linear' for linear. |
| `cycle_momentum` | bool | True | If True, momentum is cycled inversely to learning rate between `base_momentum` and `max_momentum`. |
| `base_momentum` | float or list | 0.85 | Lower momentum boundaries in the cycle. |
| `max_momentum` | float or list | 0.95 | Upper momentum boundaries in the cycle. |
| `div_factor` | float | 25.0 | Determines the initial learning rate via `initial_lr = max_lr / div_factor`. |
| `final_div_factor` | float | 10000.0 | Determines the minimum learning rate via `min_lr = initial_lr / final_div_factor`. |
| `three_phase` | bool | False | If True, use a third phase to annihilate learning rate. |
| `last_epoch` | int | -1 | The index of last epoch |

### How It Works

The 1cycle policy has two (or three) phases:

**Two-phase (default, `three_phase=False`):**
1. **Phase 1 (Increase):** LR linearly/cosinely increases from `max_lr/div_factor` to `max_lr` over `pct_start` fraction of training.
2. **Phase 2 (Decrease):** LR decreases from `max_lr` to `max_lr/(div_factor * final_div_factor)` over the remaining steps.

**Three-phase (`three_phase=True`):**
1. **Phase 1 (Increase):** LR increases from `max_lr/div_factor` to `max_lr`.
2. **Phase 2 (Decrease):** LR decreases from `max_lr` back to `max_lr/div_factor`.
3. **Phase 3 (Annihilation):** LR further decreases to `max_lr/(div_factor * final_div_factor)`.

### Momentum Cycling

When `cycle_momentum=True` and the optimizer has a `momentum` parameter:
- During Phase 1: momentum decreases from `max_momentum` to `base_momentum` (inversely proportional to lr increase)
- During Phase 2: momentum increases from `base_momentum` to `max_momentum` (inversely proportional to lr decrease)

### Example

```python
# Basic OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=len(train_loader) * num_epochs
)

# Per-batch stepping
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step per batch, not per epoch

# With epochs and steps_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=50, steps_per_epoch=len(train_loader)
)

# Three-phase with cosine annealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, total_steps=total_steps,
    pct_start=0.3, anneal_strategy='cos', three_phase=True
)

# Different max_lr per parameter group
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[0.01, 0.001], epochs=50, steps_per_epoch=100
)

# With momentum cycling disabled (for Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=100,
    cycle_momentum=False
)
```

---

## CyclicLR

Sets the learning rate according to a cyclic policy. The learning rate cycles between lower and upper boundaries.

```python
torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000,
                                    step_size_down=None, mode='triangular', gamma=1.0,
                                    scale_fn=None, scale_mode='cycle', cycle_momentum=True,
                                    base_momentum=0.8, max_momentum=0.9,
                                    last_epoch=-1, verbose='deprecated')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `base_lr` | float or list | required | Lower learning rate boundaries in the cycle for each parameter group. |
| `max_lr` | float or list | required | Upper learning rate boundaries in the cycle. |
| `step_size_up` | int | 2000 | Number of training iterations in the increasing half of a cycle. |
| `step_size_down` | int | None | Number of training iterations in the decreasing half of a cycle. If None, defaults to `step_size_up`. |
| `mode` | str | 'triangular' | One of {triangular, triangular2, exp_range}. |
| `gamma` | float | 1.0 | Constant in 'exp_range' scaling function: gamma^(cycle iterations). |
| `scale_fn` | callable | None | Custom scaling policy. A function with single argument that maps cycle position (0-1) to a scaling factor. If provided, `mode` is ignored. |
| `scale_mode` | str | 'cycle' | Whether scale_fn is evaluated on cycle number or cycle position. One of {'cycle', 'iterations'}. |
| `cycle_momentum` | bool | True | If True, momentum is cycled inversely to learning rate. |
| `base_momentum` | float or list | 0.8 | Lower momentum boundaries. |
| `max_momentum` | float or list | 0.9 | Upper momentum boundaries. |
| `last_epoch` | int | -1 | The index of last epoch |

### Modes

**triangular:** Basic triangular cycle with no amplitude scaling.

```
lr = base_lr + (max_lr - base_lr) * cycle_position
```

**triangular2:** The amplitude is halved each cycle.

```
max_lr = base_lr + (initial_max_lr - base_lr) / (2^(cycle_number))
lr = base_lr + (max_lr - base_lr) * cycle_position
```

**exp_range:** The amplitude is scaled by `gamma^(cycle_iterations)`.

```
lr = base_lr + (max_lr - base_lr) * gamma^(iterations) * cycle_position
```

### Custom Scale Function

```python
# Custom scale function: sinusoidal
def custom_scale_fn(x):
    return 0.5 * (1 + math.sin(math.pi * x))

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000,
    scale_fn=custom_scale_fn, scale_mode='cycle'
)
```

### Example

```python
# Basic triangular cycle
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000, mode='triangular'
)

# Triangular2 mode (halving amplitude)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000, mode='triangular2'
)

# Exponential range mode
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000,
    mode='exp_range', gamma=0.99994
)

# Per-batch stepping
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

---

## ReduceLROnPlateau

Reduces the learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor once learning stagnates.

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                             threshold=1e-4, threshold_mode='rel',
                                             cooldown=0, min_lr=0, eps=1e-8, verbose='deprecated')
```

**Note:** `ReduceLROnPlateau` does NOT inherit from `LRScheduler`. It has a different `step()` signature.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | Optimizer | required | Wrapped optimizer |
| `mode` | str | 'min' | One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity has stopped increasing. |
| `factor` | float | 0.1 | Factor by which the learning rate will be reduced. `new_lr = lr * factor`. |
| `patience` | int | 10 | Number of epochs with no improvement after which learning rate will be reduced. |
| `threshold` | float | 1e-4 | Threshold for measuring the new optimum, to only focus on significant changes. |
| `threshold_mode` | str | 'rel' | One of 'rel' or 'abs'. In 'rel' mode, `best = best * (1 + threshold)` for 'min' or `best = best * (1 - threshold)` for 'max'. In 'abs' mode, uses additive threshold. |
| `cooldown` | int | 0 | Number of epochs to wait before resuming normal operation after lr has been reduced. |
| `min_lr` | float or list | 0 | A scalar or a list of scalars. A lower bound on the learning rate. |
| `eps` | float | 1e-8 | Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. |

### Methods

#### `step(metrics, epoch=None)`

Updates the learning rate based on the monitored metric.

```python
scheduler.step(val_loss)  # Must pass the metric value
```

**Note the difference from other schedulers:** You must pass the metric value to `step()`.

#### `state_dict()` and `load_state_dict()`

```python
state = scheduler.state_dict()
scheduler.load_state_dict(state)
```

### Example

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5,
    threshold=1e-4, cooldown=2, min_lr=1e-7
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    # Pass the metric to step()
    scheduler.step(val_loss)

    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']}")
```

### Monitoring Different Metrics

```python
# Monitor validation loss (lower is better)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5
)
scheduler.step(val_loss)

# Monitor validation accuracy (higher is better)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5
)
scheduler.step(val_accuracy)
```

---

## Warm-Up Strategies

Warm-up gradually increases the learning rate from a small value to the target learning rate over a specified number of steps. This is critical for training stability, especially for transformers and large models.

### Linear Warmup with LinearLR

```python
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
)
```

### Linear Warmup with LambdaLR

```python
def linear_warmup_decay(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return (total_steps - step) / (total_steps - warmup_steps)
    return lr_lambda

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=linear_warmup_decay(warmup_steps=1000, total_steps=10000)
)
```

### Cosine Warmup with SequentialLR

```python
warmup_epochs = 5
total_epochs = 100

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
)
```

### Warmup for Transformers

```python
# Standard transformer warmup (from "Attention Is All You Need")
import math

def transformer_warmup(d_model=512, warmup_steps=4000):
    def lr_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return lr_lambda

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=transformer_warmup(d_model=768, warmup_steps=4000)
)
```

---

## Combining Schedulers

### SequentialLR for Phase-Based Training

```python
# Phase 1: Warmup (10 epochs)
# Phase 2: Cosine decay (90 epochs)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=10
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=90, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[10]
)

for epoch in range(100):
    train()
    scheduler.step()
```

### ChainedScheduler for Multiplicative Effects

```python
# Apply exponential decay AND warmup
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=100
)
exponential = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, exponential])
```

### Manual Composition with LambdaLR

```python
def composite_schedule(epoch):
    # Warmup for first 10 epochs
    if epoch < 10:
        return epoch / 10
    # Then cosine decay
    else:
        progress = (epoch - 10) / 90
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=composite_schedule)
```

---

## Scheduler State Management

### Saving and Loading

```python
# Save
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Resume training from the saved epoch
for epoch in range(checkpoint['epoch'], total_epochs):
    train()
    scheduler.step()
```

### ReduceLROnPlateau State

```python
# ReduceLROnPlateau also supports state_dict/load_state_dict
checkpoint = {
    'scheduler_state_dict': scheduler.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pt')

checkpoint = torch.load('checkpoint.pt')
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

### Inspecting Learning Rates

```python
# Get current learning rates
print(scheduler.get_last_lr())

# Print learning rate at each epoch
for epoch in range(10):
    train()
    scheduler.step()
    current_lr = scheduler.get_last_lr()
    print(f"Epoch {epoch}: lr = {current_lr}")
```

---

## Complete Example: Training with Scheduler

```python
import torch
import torch.nn as nn
import math

def train_with_scheduler(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['max_lr'],
        weight_decay=config['weight_decay']
    )

    # OneCycleLR - call per batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['max_lr'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config['device']), target.to(config['device'])

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step per batch for OneCycleLR

        # Validation
        model.eval()
        val_acc = validate(model, val_loader, config['device'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pt')

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}, lr={current_lr:.6f}")

    return model
```

### Typical Scheduler Configurations

```python
# Vision (ResNet): SGD with step decay
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

# Transformer: AdamW with warmup + cosine decay
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=500)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=9500, eta_min=1e-7)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [500])

# Quick experiments: OneCycleLR
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                 epochs=20, steps_per_epoch=100)

# Fine-tuning: ReduceLROnPlateau
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=3)

# Long training with restarts
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```
