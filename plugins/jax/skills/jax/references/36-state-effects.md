# 36 - Stateful Computations in JAX

## Overview

JAX is designed for pure functional programming, but real programs need state (model parameters, optimizer state, RNG keys). This chapter covers patterns for managing state in a functional style.

---

## 1. Why JAX Avoids Mutable State

### The problem with mutation

```python
# This does NOT work with JAX transformations
class BadModel:
    def __init__(self):
        self.params = jnp.ones(3)

    def update(self, x):
        self.params = self.params + x  # Mutation!
        return self.params

# JIT will not track the mutation
model = BadModel()
jax.jit(model.update)(jnp.ones(3))  # Broken!
```

### Functional approach

```python
# State is passed explicitly
class GoodModel:
    def __init__(self, params):
        self.params = params

    def update(self, params, x):
        new_params = params + x  # Create new state
        return new_params

model = GoodModel(jnp.ones(3))
new_params = jax.jit(model.update)(model.params, jnp.ones(3))
```

---

## 2. Training Loop Pattern

### Basic training step

```python
def init_params(layer_sizes):
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params.append({
            'w': jax.random.normal(key, (n_in, n_out)) * 0.01,
            'b': jnp.zeros(n_out)
        })
    return params

@jax.jit
def train_step(state, batch):
    params, opt_state = state
    x, y = batch

    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return (new_params, new_opt_state), loss

# Training loop
state = (params, opt_state)
for epoch in range(num_epochs):
    for batch in dataloader:
        state, loss = train_step(state, batch)
```

### Full training loop with RNG

```python
TrainState = tuple  # (params, opt_state, key)

def create_train_state(key, learning_rate=1e-3):
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    opt_state = optimizer.init(params)
    return (params, opt_state, key)

@jax.jit
def train_step(state, x, y):
    params, opt_state, key = state
    key, dropout_key = jax.random.split(key)

    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, dropout_key)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return (new_params, new_opt_state, key), loss
```

---

## 3. Scan-based Training Loop

```python
@jax.jit
def train_epoch(state, data):
    """Train for one epoch using scan (no Python loop)."""
    def step(state, batch):
        x, y = batch
        (new_params, new_opt_state, key), loss = train_step(state, x, y)
        return (new_params, new_opt_state, key), loss

    final_state, losses = jax.lax.scan(step, state, data)
    return final_state, losses.mean()
```

---

## 4. Optimizer State Pattern

### Using Optax

```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Initialize
params = init_params()
opt_state = optimizer.init(params)

# Training step
@jax.jit
def step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

### Custom optimizer state

```python
def sgd_with_momentum(lr=0.01, momentum=0.9):
    def init(params):
        return {'velocity': jax.tree.map(jnp.zeros_like, params)}

    def update(grads, state, params=None):
        new_velocity = jax.tree.map(
            lambda v, g: momentum * v + g, state['velocity'], grads
        )
        new_params = jax.tree.map(
            lambda p, v: p - lr * v, params, new_velocity
        )
        return new_params, {'velocity': new_velocity}

    return init, update
```

---

## 5. Model + State as Pytree

```python
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class TrainState:
    def __init__(self, params, opt_state, key, step=0):
        self.params = params
        self.opt_state = opt_state
        self.key = key
        self.step = step

    def tree_flatten(self):
        children = (self.params, self.opt_state, self.key)
        aux = {'step': self.step}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        params, opt_state, key = children
        return cls(params, opt_state, key, **aux)
```

---

## 6. Checkpointing State

### Save/Load

```python
import orbax.checkpoint as ocp

# Save
checkpointer = ocp.StandardCheckpointer()
checkpointer.save('/path/to/checkpoint', state)

# Restore
state = checkpointer.restore('/path/to/checkpoint', target=initial_state)
```

### Periodic checkpointing

```python
def train(state, data, num_epochs, checkpoint_every=10):
    for epoch in range(num_epochs):
        state, loss = train_epoch(state, data)
        if epoch % checkpoint_every == 0:
            checkpointer.save(f'/path/ckpt-{epoch}', state)
    return state
```

---

## 7. State in Reinforcement Learning

```python
@jax.jit
def env_step(state, action):
    """Pure function: state → (new_state, reward, info)"""
    new_state = dynamics(state, action)
    reward = compute_reward(new_state, action)
    done = check_done(new_state)
    return new_state, reward, done

def rollout(key, policy, initial_state, max_steps):
    def step(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)
        action = policy(action_key, state)
        new_state, reward, done = env_step(state, action)
        return (new_state, key), (new_state, reward, done)

    (final_state, _), (states, rewards, dones) = jax.lax.scan(
        step, (initial_state, key), None, length=max_steps
    )
    return states, rewards, dones
```

---

## 8. Common Patterns Summary

| Pattern | When to use | Example |
|---|---|---|
| Explicit state passing | All JAX code | `(params, opt_state, key)` tuple |
| Pytree classes | Complex state | `TrainState` with `tree_flatten` |
| Scan loops | Fixed-length iteration | Training epoch, rollout |
| Checkpoint/restore | Long training | Orbax checkpointing |
| RNG key threading | Any random code | Split key at each use |
