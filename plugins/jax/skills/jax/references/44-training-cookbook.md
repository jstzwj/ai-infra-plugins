# 44 - Training Cookbook

## Overview

Practical recipes for common training patterns in JAX, covering everything from basic SGD to advanced distributed training.

---

## 1. Basic Training Loop

```python
import jax
import jax.numpy as jnp
import optax

# Model
def init_params(key, sizes):
    params = {}
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        params[f'w{i}'] = jax.random.normal(k, (n_in, n_out)) * 0.01
        params[f'b{i}'] = jnp.zeros(n_out)
    return params

def predict(params, x):
    n_layers = len(params) // 2
    for i in range(n_layers - 1):
        x = jnp.maximum(0, x @ params[f'w{i}'] + params[f'b{i}'])
    x = x @ params[f'w{n_layers-1}'] + params[f'b{n_layers-1}']
    return x

def loss_fn(params, x, y):
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

# Optimizer
optimizer = optax.adam(1e-3)

# Training
key = jax.random.key(0)
params = init_params(key, [784, 256, 128, 10])
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

for epoch in range(100):
    for batch_x, batch_y in dataloader:
        params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## 2. Classification with Cross-Entropy

```python
def cross_entropy_loss(params, x, y, num_classes=10):
    logits = predict(params, x)
    y_onehot = jax.nn.one_hot(y, num_classes)
    return -jnp.mean(jnp.sum(y_onehot * jax.nn.log_softmax(logits), axis=-1))

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

def accuracy(params, x, y):
    logits = predict(params, x)
    return jnp.mean(jnp.argmax(logits, axis=-1) == y)
```

---

## 3. Training with Dropout

```python
def predict_with_dropout(params, key, x, rate=0.5):
    n_layers = len(params) // 2
    for i in range(n_layers - 1):
        x = jnp.maximum(0, x @ params[f'w{i}'] + params[f'b{i}'])
        key, k = jax.random.split(key)
        mask = jax.random.bernoulli(k, 1 - rate, x.shape)
        x = jnp.where(mask, x / (1 - rate), 0.0)
    return x @ params[f'w{n_layers-1}'] + params[f'b{n_layers-1}']

def loss_with_dropout(params, key, x, y, rate=0.5):
    pred = predict_with_dropout(params, key, x, rate)
    return jnp.mean((pred - y) ** 2)

@jax.jit
def train_step(params, opt_state, key, x, y):
    loss, grads = jax.value_and_grad(loss_with_dropout)(params, key, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

---

## 4. Gradient Accumulation

```python
@jax.jit
def accumulated_grad_step(params, opt_state, xs, ys):
    """Accumulate gradients across mini-batches."""
    def batch_grad(grads_and_loss, batch):
        x, y = batch
        loss, g = jax.value_and_grad(loss_fn)(params, x, y)
        grads, total_loss = grads_and_loss
        new_grads = jax.tree.map(lambda a, b: a + b, grads, g)
        return new_grads, total_loss + loss

    # Split into micro-batches
    micro_batches = (xs.reshape(-1, micro_bs, *xs.shape[1:]),
                     ys.reshape(-1, micro_bs, *ys.shape[1:]))

    zero_grads = jax.tree.map(jnp.zeros_like, params)
    total_grads, total_loss = jax.lax.scan(
        batch_grad, (zero_grads, 0.0),
        xs=jax.tree.map(lambda x: x, micro_batches)
    )

    n_micro = xs.shape[0] // micro_bs
    avg_grads = jax.tree.map(lambda g: g / n_micro, total_grads)
    avg_loss = total_loss / n_micro

    updates, new_opt_state = optimizer.update(avg_grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, avg_loss
```

---

## 5. Learning Rate Scheduling

```python
import optax

# Cosine schedule
schedule = optax.cosine_decay_schedule(
    init_value=1e-3,
    decay_steps=10000,
    alpha=0.1  # Minimum LR as fraction of init
)

# Warmup + cosine
warmup = optax.linear_schedule(0, 1e-3, 1000)
cosine = optax.cosine_decay_schedule(1e-3, 50000)
schedule = optax.join_schedules([warmup, cosine], [1000])

optimizer = optax.adam(schedule)
```

---

## 6. Gradient Clipping

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients
    optax.adam(1e-3),
)
```

---

## 7. Mixed Precision Training

```python
@jax.jit
def mixed_precision_step(params, opt_state, x, y):
    def compute_loss(params, x, y):
        # Cast inputs to bfloat16
        x_bf = x.astype(jnp.bfloat16)
        # Forward in bfloat16
        h = x_bf @ params['w1'].astype(jnp.bfloat16)
        h = jnp.maximum(0, h)
        logits = h @ params['w2'].astype(jnp.bfloat16)
        # Loss in float32
        logits_f32 = logits.astype(jnp.float32)
        return -jnp.mean(jax.nn.log_softmax(logits_f32)[jnp.arange(y.shape[0]), y])

    loss, grads = jax.value_and_grad(compute_loss)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

---

## 8. Multi-GPU Training with shard_map

```python
from jax.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

mesh = Mesh(jax.devices(), ('devices',))
sharding = NamedSharding(mesh, P('devices', None))

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Average gradients across devices
    grads = jax.tree.map(lambda g: jax.lax.pmean(g, 'devices'), grads)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

with mesh:
    x = jax.device_put(data_x, sharding)
    y = jax.device_put(data_y, sharding)
    params, opt_state, loss = train_step(params, opt_state, x, y)
```

---

## 9. Early Stopping

```python
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    params, opt_state, loss = train_epoch(params, opt_state, data)

    val_loss = evaluate(params, val_data)

    if val_loss < best_loss:
        best_loss = val_loss
        best_params = params  # Save
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        params = best_params
        break
```

---

## 10. Evaluation Pattern

```python
@jax.jit
def evaluate_batch(params, x, y):
    logits = predict(params, x)
    loss = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc

def evaluate(params, dataset):
    total_loss = 0
    total_acc = 0
    n = 0
    for x, y in dataset:
        loss, acc = evaluate_batch(params, x, y)
        total_loss += loss * x.shape[0]
        total_acc += acc * x.shape[0]
        n += x.shape[0]
    return total_loss / n, total_acc / n
```

---

## 11. Transfer Learning

```python
# Load pretrained params
pretrained_params = load_pretrained()

# Freeze early layers
frozen_params = {k: v for k, v in pretrained_params.items() if 'w0' in k or 'w1' in k}
trainable_params = {k: v for k, v in pretrained_params.items() if k not in frozen_params}

def loss_fn(trainable_params, frozen_params, x, y):
    all_params = {**frozen_params, **trainable_params}
    pred = predict(all_params, x)
    return jnp.mean((pred - y) ** 2)

grads = jax.grad(loss_fn)(trainable_params, frozen_params, x, y)
```

---

## 12. Complete Training Template

```python
import jax
import jax.numpy as jnp
import optax

class Trainer:
    def __init__(self, model_fn, init_fn, lr=1e-3):
        self.model_fn = model_fn
        key = jax.random.key(0)
        self.params = init_fn(key)
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

    @jax.jit
    def step(self, params, opt_state, x, y):
        loss, grads = jax.value_and_grad(self.model_fn)(params, x, y)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def train(self, data, num_epochs):
        for epoch in range(num_epochs):
            for x, y in data:
                self.params, self.opt_state, loss = self.step(
                    self.params, self.opt_state, x, y
                )
            print(f"Epoch {epoch}: loss={loss:.4f}")
```
