# TensorFlow Keras Training Reference

This document provides comprehensive reference documentation for all aspects of training models with TensorFlow Keras, including model compilation, optimizers, loss functions, metrics, callbacks, and custom training loops.

---

## Table of Contents

1. [Model Compilation](#model-compilation)
2. [Training (fit)](#training-fit)
3. [Evaluation](#evaluation)
4. [Prediction](#prediction)
5. [Optimizers](#optimizers)
6. [Learning Rate Schedules](#learning-rate-schedules)
7. [Loss Functions](#loss-functions)
8. [Metrics](#metrics)
9. [Callbacks](#callbacks)
10. [Custom Training Loops](#custom-training-loops)
11. [Training Utilities](#training-utilities)

---

## Model Compilation

### Model.compile()

Configures the model for training.

```python
model.compile(
    optimizer='rmsprop',                     # String name or optimizer instance
    loss=None,                               # String name, loss instance, or dict/list
    metrics=None,                            # List of metrics to evaluate during training/testing
    loss_weights=None,                       # Dict or list mapping losses to weights (multi-output)
    weighted_metrics=None,                   # List of metrics evaluated with sample_weight or class_weight
    run_eagerly=None,                        # Boolean. If True, runs eagerly (no tf.function tracing)
    steps_per_execution=1,                   # Integer. Number of batches per tf.function call
    jit_compile='auto'                       # 'auto', True, or False. Whether to compile with XLA
)
```

**Usage:**
```python
# Basic compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# With custom optimizer and loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
    ]
)

# Multi-output model
model.compile(
    optimizer='adam',
    loss={
        'output_1': 'binary_crossentropy',
        'output_2': 'categorical_crossentropy'
    },
    loss_weights={
        'output_1': 1.0,
        'output_2': 0.5
    },
    metrics={
        'output_1': ['accuracy'],
        'output_2': ['accuracy', 'top_k_categorical_accuracy']
    }
)

# With XLA compilation for performance
model.compile(
    optimizer='adam',
    loss='mse',
    jit_compile=True
)

# With steps_per_execution for TPU optimization
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    steps_per_execution=100
)
```

---

## Training (fit)

### Model.fit()

Trains the model for a fixed number of epochs.

```python
model.fit(
    x=None,                               # Input data: numpy array, tensor, dict, dataset, generator
    y=None,                               # Target data
    batch_size=None,                      # Integer or None. Number of samples per gradient update
    epochs=1,                             # Integer. Number of epochs to train the model
    verbose='auto',                       # 0 = silent, 1 = progress bar, 2 = one line per epoch
    callbacks=None,                       # List of keras.callbacks.Callback instances
    validation_split=0.0,                 # Float between 0 and 1. Fraction of training data for validation
    validation_data=None,                 # Data for validation: (x_val, y_val) or dataset
    shuffle=True,                         # Boolean or 'batch'. Whether to shuffle training data
    class_weight=None,                    # Dict mapping class indices to weight
    sample_weight=None,                   # Numpy array of weights for training samples
    initial_epoch=0,                      # Integer. Epoch to start training from (for resuming)
    steps_per_epoch=None,                 # Integer or None. Steps before declaring an epoch finished
    validation_steps=None,                # Integer. Number of validation steps per epoch
    validation_batch_size=None,           # Integer or None. Batch size for validation
    validation_freq=1,                    # Integer or list. Only validate every N epochs
    max_queue_size=10,                    # Integer. Max size for generator queue
    workers=1,                            # Integer. Max workers for data loading
    use_multiprocessing=False,            # Boolean. Whether to use multiprocessing
    **kwargs
)
```

**Returns:** A `History` object with `history` attribute containing training loss and metric values.

**Usage:**
```python
# Basic training with numpy arrays
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Training with tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(32)

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Training with class weights (imbalanced data)
class_weights = {0: 1.0, 1: 5.0, 2: 2.0}
model.fit(x_train, y_train, class_weight=class_weights, epochs=10)

# Training with sample weights
sample_weights = np.where(y_train == 1, 2.0, 1.0)
model.fit(x_train, y_train, sample_weight=sample_weights, epochs=10)

# Training with validation frequency
model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    validation_freq=5  # Validate every 5 epochs
)

# Resume training from a specific epoch
model.fit(
    x_train, y_train,
    initial_epoch=50,
    epochs=100
)
```

---

## Evaluation

### Model.evaluate()

Returns the loss value and metrics values for the model.

```python
model.evaluate(
    x=None,                               # Input data
    y=None,                               # Target data
    batch_size=None,                      # Integer or None
    verbose='auto',                       # 0 or 1
    sample_weight=None,                   # Numpy array
    steps=None,                           # Integer
    callbacks=None,                       # List of callbacks
    return_dict=False,                    # Boolean. If True, returns dict instead of list
    **kwargs
)
```

**Usage:**
```python
# Basic evaluation
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

# With dataset
results = model.evaluate(test_dataset, return_dict=True)
# {'loss': 0.35, 'accuracy': 0.89}

# With callbacks
results = model.evaluate(
    test_dataset,
    callbacks=[tf.keras.callbacks.ProgbarLogger()]
)
```

---

## Prediction

### Model.predict()

Generates output predictions for the input samples.

```python
model.predict(
    x,                                    # Input data
    batch_size=None,                      # Integer
    verbose='auto',                       # 0 or 1
    steps=None,                           # Integer
    callbacks=None,                       # List of callbacks
    max_queue_size=10,                    # Integer
    workers=1,                            # Integer
    use_multiprocessing=False,            # Boolean
    **kwargs
)
```

**Usage:**
```python
# Basic prediction
predictions = model.predict(x_test)

# Predict with batch size
predictions = model.predict(x_test, batch_size=64)

# Predict from dataset
predictions = model.predict(test_dataset)

# Get class predictions
predicted_classes = np.argmax(predictions, axis=1)
```

### Model.predict_on_batch()

Returns predictions for a single batch of samples.

```python
predictions = model.predict_on_batch(x_batch)
```

### Model.predict_step()

A single step for prediction. Override for custom behavior.

```python
class CustomModel(tf.keras.Model):
    def predict_step(self, data):
        x = data
        return self(x, training=False)
```

---

## Optimizers

### SGD

Stochastic Gradient Descent optimizer with optional momentum and Nesterov acceleration.

```python
tf.keras.optimizers.SGD(
    learning_rate=0.01,                   # Float or LearningRateSchedule
    momentum=0.0,                         # Float >= 0. Accelerates SGD in relevant direction
    nesterov=False,                       # Boolean. Whether to apply Nesterov momentum
    weight_decay=None,                    # Float. If set, weight decay is applied
    clipnorm=None,                        # Float. Clips gradients to max norm
    clipvalue=None,                       # Float. Clips gradients to [-clipvalue, clipvalue]
    global_clipnorm=None,                 # Float. Clips gradients by global norm
    use_ema=False,                        # Boolean. Whether to use Exponential Moving Average
    ema_momentum=0.99,                    # Float. EMA momentum
    ema_overwrite_frequency=None,         # Int or None. Steps between EMA variable overwrites
    jit_compile=True,                     # Boolean. Whether to use XLA compilation
    name='SGD',                           # String
    **kwargs
)
```

**Usage:**
```python
# Basic SGD
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# With momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Nesterov momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# With gradient clipping
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    clipnorm=1.0
)
```

### Adam

Adam optimizer. Implements the Adam algorithm based on the paper "Adam: A Method for Stochastic Optimization" by Kingma and Ba.

Update rule:
```
t = t + 1
lr_t = learning_rate * sqrt(1 - beta_2^t) / (1 - beta_1^t)
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g * g
variable = variable - lr_t * m_t / (sqrt(v_t) + epsilon)
```

```python
tf.keras.optimizers.Adam(
    learning_rate=0.001,                  # Float or LearningRateSchedule
    beta_1=0.9,                           # Float [0, 1). Exponential decay rate for 1st moment estimates
    beta_2=0.999,                         # Float [0, 1). Exponential decay rate for 2nd moment estimates
    epsilon=1e-07,                        # Float >= 0. Small constant for numerical stability
    amsgrad=False,                        # Boolean. Whether to apply AMSGrad variant
    weight_decay=None,                    # Float. Weight decay coefficient
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Adam',
    **kwargs
)
```

**Usage:**
```python
# Basic Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# With AMSGrad variant
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

# With custom parameters
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

# With learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### AdamW

Adam with decoupled weight decay.

```python
tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.004,                   # Float. Decoupled weight decay coefficient
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='AdamW',
    **kwargs
)
```

**Usage:**
```python
# AdamW as recommended for transformer training
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

# With gradient clipping and schedule
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=warmup_cosine_schedule,
    weight_decay=0.01,
    global_clipnorm=1.0
)
```

### Adagrad

Adagrad optimizer with parameter-specific learning rates.

```python
tf.keras.optimizers.Adagrad(
    learning_rate=0.001,
    initial_accumulator_value=0.1,        # Float >= 0. Starting value for accumulators
    epsilon=1e-07,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Adagrad',
    **kwargs
)
```

### Adadelta

Adadelta optimizer. A more robust extension of Adagrad.

```python
tf.keras.optimizers.Adadelta(
    learning_rate=0.001,
    rho=0.95,                             # Float >= 0. Decay rate for the moving average
    epsilon=1e-07,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Adadelta',
    **kwargs
)
```

### RMSprop

RMSprop optimizer. Good for recurrent neural networks.

```python
tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,                              # Float >= 0. Decay factor for the moving average
    momentum=0.0,                         # Float >= 0. Momentum term
    epsilon=1e-07,
    centered=False,                       # Boolean. If True, compute centered RMSProp
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='RMSprop',
    **kwargs
)
```

**Usage:**
```python
# Standard RMSprop
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# Centered RMSprop (better for some problems)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, centered=True)

# With momentum
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.9)
```

### Ftrl

FTRL (Follow The Regularized Leader) optimizer.

```python
tf.keras.optimizers.Ftrl(
    learning_rate=0.001,
    learning_rate_power=-0.5,             # Float. Controls how learning rate decreases during training
    initial_accumulator_value=0.1,        # Float >= 0. Starting value for accumulators
    l1_regularization_strength=0.0,       # Float >= 0. L1 regularization
    l2_regularization_strength=0.0,       # Float >= 0. L2 regularization
    l2_shrinkage_regularization_strength=0.0,  # Float >= 0. L2 shrinkage regularization
    beta=0.0,                             # Float. Beta value from the FTRL paper
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Ftrl',
    **kwargs
)
```

### Nadam

NAdam optimizer (Adam with Nesterov momentum).

```python
tf.keras.optimizers.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Nadam',
    **kwargs
)
```

### Lion

Lion (EvoLved Sign Momentum) optimizer. More memory-efficient than Adam.

```python
tf.keras.optimizers.Lion(
    learning_rate=0.0001,                 # Typically 3-10x smaller than Adam's learning rate
    beta_1=0.9,                           # Float [0, 1)
    beta_2=0.99,                          # Float [0, 1)
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Lion',
    **kwargs
)
```

### LossScaleOptimizer

Wraps an optimizer to apply loss scaling for mixed precision training.

```python
tf.keras.mixed_precision.LossScaleOptimizer(
    inner_optimizer,                      # Optimizer to wrap
    dynamic=True,                         # Boolean. Whether to use dynamic loss scaling
    initial_scale=2**15,                  # Float. Initial loss scale
    dynamic_growth_steps=1000,            # Integer. Steps between scale increases
    **kwargs
)
```

**Usage:**
```python
# Mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# In custom training loop
with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = loss_fn(targets, predictions)
    scaled_loss = optimizer.get_scaled_loss(loss)

scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
gradients = optimizer.get_unscaled_gradients(scaled_gradients)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

---

## Learning Rate Schedules

### ExponentialDecay

Exponential learning rate decay schedule.

```python
tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,                # Float. Initial learning rate
    decay_steps,                          # Integer. Period of decay
    decay_rate,                           # Float. Decay rate
    staircase=False,                      # Boolean. If True, decay in discrete intervals
    name=None                             # String
)
```

**Formula:** `lr = initial_lr * decay_rate ^ (step / decay_steps)`

**Usage:**
```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### PiecewiseConstantDecay

Piecewise constant learning rate schedule.

```python
tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries,                           # List of step numbers at which to change learning rate
    values,                               # List of learning rate values
    name=None
)
```

**Usage:**
```python
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[5000, 10000, 20000],
    values=[0.1, 0.01, 0.001, 0.0001]
)
# LR: 0.1 for steps 0-4999, 0.01 for 5000-9999, 0.001 for 10000-19999, 0.0001 for 20000+
```

### PolynomialDecay

Polynomial learning rate decay schedule.

```python
tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,                # Float
    decay_steps,                          # Integer
    end_learning_rate=0.0001,             # Float
    power=1.0,                            # Float. The power of the polynomial
    cycle=False,                          # Boolean. Whether to cycle beyond decay_steps
    name=None
)
```

**Usage:**
```python
# Linear warmup then linear decay
warmup_steps = 1000
total_steps = 10000

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.001,
    decay_steps=total_steps - warmup_steps,
    end_learning_rate=0.00001
)
```

### InverseTimeDecay

Inverse time learning rate decay schedule.

```python
tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
```

**Formula:** `lr = initial_lr / (1 + decay_rate * step / decay_steps)`

### CosineDecay

Cosine learning rate decay schedule.

```python
tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,                # Float
    decay_steps,                          # Integer
    alpha=0.0,                            # Float. Minimum learning rate as fraction of initial
    name=None,
    warmup_target=None,                   # Float. Target learning rate after warmup
    warmup_steps=0                        # Integer. Number of warmup steps
)
```

**Usage:**
```python
# Cosine decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=100000,
    alpha=0.0
)

# Cosine decay with warmup
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0,
    warmup_target=0.1,
    warmup_steps=5000,
    decay_steps=100000,
    alpha=0.01
)
```

### CosineDecayRestarts

Cosine decay with restarts (SGDR - Stochastic Gradient Descent with Warm Restarts).

```python
tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate,
    first_decay_steps,                    # Integer. Steps for the first decay cycle
    t_mul=2.0,                            # Float. Factor to increase decay_steps after each restart
    m_mul=1.0,                            # Float. Factor to decrease initial_lr after each restart
    alpha=0.0,
    name=None
)
```

### Custom Learning Rate Schedule

```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
```

---

## Loss Functions

### Regression Losses

#### MeanSquaredError (MSE)

```python
tf.keras.losses.MeanSquaredError(
    reduction='sum_over_batch_size',      # 'sum_over_batch_size', 'sum', 'none'
    name='mean_squared_error'
)
# Alias: 'mse'
```

#### MeanAbsoluteError (MAE)

```python
tf.keras.losses.MeanAbsoluteError(
    reduction='sum_over_batch_size',
    name='mean_absolute_error'
)
# Alias: 'mae'
```

#### MeanAbsolutePercentageError (MAPE)

```python
tf.keras.losses.MeanAbsolutePercentageError(
    reduction='sum_over_batch_size',
    name='mean_absolute_percentage_error'
)
# Formula: 100 * abs((y_true - y_pred) / y_true)
```

#### MeanSquaredLogarithmicError (MSLE)

```python
tf.keras.losses.MeanSquaredLogarithmicError(
    reduction='sum_over_batch_size',
    name='mean_squared_logarithmic_error'
)
# Formula: square(log(y_true + 1) - log(y_pred + 1))
```

#### Huber

```python
tf.keras.losses.Huber(
    delta=1.0,                            # Float. Point where Huber loss transitions from quadratic to linear
    reduction='sum_over_batch_size',
    name='huber_loss'
)
```

**Usage:**
```python
# For robust regression
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.Huber(delta=1.0)
)
```

#### LogCosh

Logarithm of the hyperbolic cosine of the prediction error.

```python
tf.keras.losses.LogCosh(
    reduction='sum_over_batch_size',
    name='log_cosh'
)
```

#### CosineSimilarity

```python
tf.keras.losses.CosineSimilarity(
    axis=-1,                              # Integer. Axis along which cosine similarity is computed
    reduction='sum_over_batch_size',
    name='cosine_similarity'
)
# Note: loss is negative cosine similarity (minimizing maximizes similarity)
```

### Classification Losses

#### BinaryCrossentropy

```python
tf.keras.losses.BinaryCrossentropy(
    from_logits=False,                    # Boolean. Whether predictions are logits or probabilities
    label_smoothing=0.0,                  # Float in [0, 1]. Smoothing factor for labels
    axis=-1,
    reduction='sum_over_batch_size',
    name='binary_crossentropy'
)
```

**Usage:**
```python
# With sigmoid output (probabilities)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# With logits (more numerically stable)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
)

# With label smoothing
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
```

#### CategoricalCrossentropy

```python
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction='sum_over_batch_size',
    name='categorical_crossentropy'
)
```

**Usage:**
```python
# For one-hot encoded labels
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# From logits (recommended for numerical stability)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
)
```

#### SparseCategoricalCrossentropy

```python
tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction='sum_over_batch_size',
    name='sparse_categorical_crossentropy'
)
```

**Usage:**
```python
# For integer labels (not one-hot)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

#### Hinge

```python
tf.keras.losses.Hinge(
    reduction='sum_over_batch_size',
    name='hinge'
)
# For "maximum-margin" classification (SVM-style)
```

#### SquaredHinge

```python
tf.keras.losses.SquaredHinge(
    reduction='sum_over_batch_size',
    name='squared_hinge'
)
```

#### CategoricalHinge

```python
tf.keras.losses.CategoricalHinge(
    reduction='sum_over_batch_size',
    name='categorical_hinge'
)
```

### Probabilistic Losses

#### KLDivergence

```python
tf.keras.losses.KLDivergence(
    reduction='sum_over_batch_size',
    name='kl_divergence'
)
# Formula: y_true * log(y_true / y_pred)
```

#### Poisson

```python
tf.keras.losses.Poisson(
    reduction='sum_over_batch_size',
    name='poisson'
)
# Formula: y_pred - y_true * log(y_pred + epsilon)
```

### Custom Loss Functions

```python
# Custom loss as a function
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred) + 0.01 * tf.reduce_sum(tf.square(y_pred)))

# Custom loss as a class
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.math.pow(1 - y_pred, self.gamma)
        return tf.reduce_mean(weight * cross_entropy)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config
```

---

## Metrics

### Classification Metrics

#### BinaryAccuracy

```python
tf.keras.metrics.BinaryAccuracy(
    name='binary_accuracy',
    dtype=None,
    threshold=0.5                         # Float in [0, 1]. Threshold for binary prediction
)
```

#### CategoricalAccuracy

```python
tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None)
```

#### SparseCategoricalAccuracy

```python
tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy', dtype=None)
```

#### TopKCategoricalAccuracy

```python
tf.keras.metrics.TopKCategoricalAccuracy(
    k=5,                                  # Integer. Number of top elements to consider
    name='top_k_categorical_accuracy',
    dtype=None
)
```

#### SparseTopKCategoricalAccuracy

```python
tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k=5,
    name='sparse_top_k_categorical_accuracy',
    dtype=None
)
```

#### AUC

Approximates the Area Under the Curve (ROC or PR).

```python
tf.keras.metrics.AUC(
    num_thresholds=200,                   # Integer. Number of thresholds for discretization
    curve='ROC',                          # 'ROC' or 'PR' (Precision-Recall)
    summation_method='interpolation',     # 'interpolation', 'minoring', 'majoring'
    name=None,
    dtype=None,
    thresholds=None,                      # Optional list of thresholds
    multi_label=False,                    # Boolean
    num_labels=None,                      # Integer
    label_weights=None,                   # List of floats
    from_logits=False
)
```

#### Precision

```python
tf.keras.metrics.Precision(
    thresholds=None,                      # Float or list of floats
    top_k=None,                           # Integer
    class_id=None,                        # Integer
    name=None,
    dtype=None
)
```

#### Recall

```python
tf.keras.metrics.Recall(
    thresholds=None,
    top_k=None,
    class_id=None,
    name=None,
    dtype=None
)
```

#### TruePositives / FalsePositives / TrueNegatives / FalseNegatives

```python
tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)
```

#### PrecisionAtRecall

```python
tf.keras.metrics.PrecisionAtRecall(
    recall,                               # Float in [0, 1]. Target recall
    num_thresholds=200,
    name=None,
    dtype=None
)
```

#### RecallAtPrecision

```python
tf.keras.metrics.RecallAtPrecision(
    precision,                            # Float in [0, 1]. Target precision
    num_thresholds=200,
    name=None,
    dtype=None
)
```

### Regression Metrics

#### Mean / Sum

```python
tf.keras.metrics.Mean(name='mean', dtype=None)
tf.keras.metrics.Sum(name='sum', dtype=None)
```

#### MeanMetricWrapper

Wraps a stateless loss function into a metric.

```python
tf.keras.metrics.MeanMetricWrapper(fn, name=None, dtype=None, **kwargs)
```

#### CosineSimilarity (Metric)

```python
tf.keras.metrics.CosineSimilarity(
    name='cosine_similarity',
    dtype=None,
    axis=-1
)
```

#### MeanSquaredError (Metric)

```python
tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None)
```

#### RootMeanSquaredError

```python
tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error', dtype=None)
```

#### MeanAbsoluteError (Metric)

```python
tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error', dtype=None)
```

#### MeanAbsolutePercentageError (Metric)

```python
tf.keras.metrics.MeanAbsolutePercentageError(
    name='mean_absolute_percentage_error', dtype=None
)
```

#### MeanSquaredLogarithmicError (Metric)

```python
tf.keras.metrics.MeanSquaredLogarithmicError(
    name='mean_squared_logarithmic_error', dtype=None
)
```

#### LogCoshError

```python
tf.keras.metrics.LogCoshError(name='logcosh', dtype=None)
```

### Segmentation Metrics

#### IoU

```python
tf.keras.metrics.IoU(
    num_classes,                          # Integer
    target_class_ids,                     # List of integers
    name=None,
    dtype=None
)
```

#### MeanIoU

```python
tf.keras.metrics.MeanIoU(
    num_classes,                          # Integer
    name=None,
    dtype=None
)
```

### Custom Metrics

```python
# Custom metric as a class
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.true_positives.assign_add(tf.reduce_sum(y_pred * y_true))
        self.false_positives.assign_add(tf.reduce_sum(y_pred * (1 - y_true)))
        self.false_negatives.assign_add(tf.reduce_sum((1 - y_pred) * y_true))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)
```

---

## Callbacks

### Callback (Base Class)

Base class for all callbacks.

```python
tf.keras.callbacks.Callback()
```

**Available methods to override:**
- `on_train_begin(logs=None)`
- `on_train_end(logs=None)`
- `on_epoch_begin(epoch, logs=None)`
- `on_epoch_end(epoch, logs=None)`
- `on_train_batch_begin(batch, logs=None)`
- `on_train_batch_end(batch, logs=None)`
- `on_test_begin(logs=None)`
- `on_test_end(logs=None)`
- `on_test_batch_begin(batch, logs=None)`
- `on_test_batch_end(batch, logs=None)`
- `on_predict_begin(logs=None)`
- `on_predict_end(logs=None)`
- `on_predict_batch_begin(batch, logs=None)`
- `on_predict_batch_end(batch, logs=None)`

### ModelCheckpoint

Saves the model at the end of each epoch or when a metric improves.

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath,                             # String or PathLike. Path to save the model file
    monitor='val_loss',                   # String. Metric to monitor
    verbose=0,                            # Integer
    save_best_only=False,                 # Boolean. Only save if monitored metric improves
    save_weights_only=False,              # Boolean. If True, only saves weights
    mode='auto',                          # 'auto', 'min', 'max'. Direction of metric improvement
    save_freq='epoch',                    # 'epoch' or integer. If integer, saves every N batches
    initial_value_threshold=None,         # Float. Initial "best" value for comparison
    **kwargs
)
```

**Usage:**
```python
# Save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Save every epoch with epoch number
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model_epoch_{epoch:02d}.keras',
    save_freq='epoch'
)

# Save every 1000 batches
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'checkpoint_{step}.keras',
    save_freq=1000
)
```

### EarlyStopping

Stops training when a monitored metric has stopped improving.

```python
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',                   # String. Metric to monitor
    min_delta=0,                          # Float. Minimum change to qualify as improvement
    patience=0,                           # Integer. Epochs with no improvement before stopping
    verbose=0,                            # Integer
    mode='auto',                          # 'auto', 'min', 'max'
    baseline=None,                        # Float. Baseline value for the monitored quantity
    restore_best_weights=False,           # Boolean. Whether to restore model weights from best epoch
    start_from_epoch=0                    # Integer. Epoch before which no early stopping check
)
```

**Usage:**
```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=1e-4
)
```

### TensorBoard

Enables visualization in TensorBoard.

```python
tf.keras.callbacks.TensorBoard(
    log_dir='logs',                       # String. Path for TensorBoard logs
    histogram_freq=0,                     # Integer. Frequency (in epochs) for weight histograms
    write_graph=True,                     # Boolean. Whether to visualize the graph
    write_images=False,                   # Boolean. Whether to write model weights as images
    write_steps_per_second=False,         # Boolean. Log steps per second
    update_freq='epoch',                  # 'batch', 'epoch', or integer
    profile_batch=2,                      # Integer or '0'. Which batch(es) to profile
    embeddings_freq=0,                    # Integer. Frequency for embedding layers visualization
    embeddings_metadata=None              # Dict or string
)
```

**Usage:**
```python
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
    histogram_freq=1,
    profile_batch='500,520'
)
```

### ReduceLROnPlateau

Reduces learning rate when a metric has stopped improving.

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',                   # String
    factor=0.1,                           # Float. Factor by which LR will be reduced (new_lr = lr * factor)
    patience=10,                          # Integer
    verbose=0,                            # Integer
    mode='auto',                          # 'auto', 'min', 'max'
    min_delta=1e-4,                       # Float
    cooldown=0,                           # Integer. Epochs to wait before resuming normal operation
    min_lr=0,                             # Float. Lower bound on learning rate
    **kwargs
)
```

**Usage:**
```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
```

### CSVLogger

Streams epoch results to a CSV file.

```python
tf.keras.callbacks.CSVLogger(
    filename,                             # String. Path to CSV file
    separator=',',                        # String. Field separator
    append=False                          # Boolean. Whether to append to existing file
)
```

### TerminateOnNaN

Callback that terminates training when a NaN loss is encountered.

```python
tf.keras.callbacks.TerminateOnNaN()
```

### LearningRateScheduler

Learning rate scheduler callback.

```python
tf.keras.callbacks.LearningRateScheduler(
    schedule,                             # Function(epoch, lr) -> new_lr
    verbose=0                             # Integer. 0: quiet, 1: log new LR
)
```

**Usage:**
```python
# Step decay
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# Warm-up schedule
def warmup_schedule(epoch, lr):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return 0.001 * (epoch + 1) / warmup_epochs
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(warmup_schedule)
```

### ProgbarLogger

Callback that prints metrics to stdout.

```python
tf.keras.callbacks.ProgbarLogger(
    count_mode='samples',                 # 'samples' or 'steps'
    stateful_metrics=None                 # List of strings. Metrics that should not be averaged
)
```

### BackupAndRestore

Callback to back up and restore the training state.

```python
tf.keras.callbacks.BackupAndRestore(
    backup_dir,                           # String. Directory for backup
    save_freq='epoch',                    # 'epoch' or integer
    delete_checkpoint=True,               # Boolean. Whether to delete checkpoint after training
    save_before_preemption=False           # Boolean. Whether to save on preemption signal
)
```

### Custom Callback

```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print('Starting training')

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch}: loss = {logs["loss"]:.4f}, '
              f'val_loss = {logs.get("val_loss", "N/A")}')

    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f'Batch {batch}: loss = {logs["loss"]:.4f}')

# Example: Gradient logging callback
class GradientLoggingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            gradients = self.model.optimizer.get_gradients(
                self.model.total_loss,
                self.model.trainable_weights
            )
            for var, grad in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(f'gradients/{var.name}', grad, step=self.model.optimizer.iterations)
```

---

## Custom Training Loops

### Basic GradientTape Loop

```python
# Define model, loss, optimizer
model = create_model()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
        # Add regularization losses
        loss += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_fn(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

# Training loop
for epoch in range(num_epochs):
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for images, labels in train_dataset:
        train_step(images, labels)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    print(f'Epoch {epoch + 1}: '
          f'Loss: {train_loss.result():.4f}, '
          f'Accuracy: {train_accuracy.result() * 100:.2f}%, '
          f'Test Loss: {test_loss.result():.4f}, '
          f'Test Accuracy: {test_accuracy.result() * 100:.2f}%')
```

### Overriding train_step

```python
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```

### Custom Training with Multiple Outputs

```python
@tf.function
def train_step_multi_output(x, y_dict):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        total_loss = 0
        for output_name, y_true in y_dict.items():
            loss = loss_fns[output_name](y_true, predictions[output_name])
            total_loss += loss_weights[output_name] * loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss
```

### Gradient Accumulation

```python
@tf.function
def train_step_with_accumulation(images, labels, accum_steps=4):
    for i in tf.range(accum_steps):
        with tf.GradientTape() as tape:
            start = i * batch_size // accum_steps
            end = (i + 1) * batch_size // accum_steps
            mini_batch_images = images[start:end]
            mini_batch_labels = labels[start:end]
            predictions = model(mini_batch_images, training=True)
            loss = loss_fn(mini_batch_labels, predictions) / accum_steps

        gradients = tape.gradient(loss, model.trainable_variables)
        if i == 0:
            accumulated_gradients = [tf.zeros_like(g) for g in gradients]
        accumulated_gradients = [ag + g for ag, g in zip(accumulated_gradients, gradients)]

    optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
```

### Distributed Training with strategy.run()

```python
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

def compute_loss(labels, predictions):
    per_example_loss = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(num_epochs):
    for batch in dist_dataset:
        loss = distributed_train_step(batch)
```

---

## Training Utilities

### tf.keras.utils.Sequence

Base object for fitting to a sequence of data. Guaranteed to be used in a thread-safe manner.

```python
class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        # Shuffle data at end of each epoch
        indices = np.arange(len(self.x))
        np.random.shuffle(indices)
        self.x = self.x[indices]
        self.y = self.y[indices]
```

### tf.keras.utils.GeneratorEnqueuer

Builds a deque from a generator. Used for multiprocessing data loading.

```python
enqueuer = tf.keras.utils.GeneratorEnqueuer(
    generator,                            # Generator yielding (x, y) tuples
    use_multiprocessing=False,            # Boolean
    random_seed=None                      # Integer
)
enqueuer.start(max_queue_size=10, workers=1)
```

### Mixed Precision Training

```python
# Enable mixed precision globally
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Policy options:
# - 'float32': Default, all computations in float32
# - 'mixed_float16': Mixed precision with float16 compute, float32 master weights
# - 'mixed_bfloat16': Mixed precision with bfloat16 (for TPU and some CPUs)
# - 'float16': All computations in float16 (rarely used)

# Check current policy
policy = tf.keras.mixed_precision.global_policy()
print(policy.name)          # 'mixed_float16'
print(policy.compute_dtype) # 'float16'
print(policy.variable_dtype) # 'float32'

# Per-layer policy
layer = tf.keras.layers.Dense(64, dtype='float32')  # Override for numerical sensitivity

# Loss scaling for mixed precision
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.Adam(0.001)
)
```

### Gradient Clipping

```python
# Option 1: Via optimizer arguments
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0       # Clip by max norm
)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipvalue=0.5      # Clip to [-0.5, 0.5]
)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    global_clipnorm=1.0  # Clip by global norm
)

# Option 2: Manual gradient clipping in custom loop
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    # Clip by global norm
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_norm=1.0)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
```

### Common Training Patterns

```python
# Full training pipeline example
def train_model():
    # 1. Create model
    model = create_model()

    # 2. Compile with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1,
        decay_steps=50000,
        warmup_target=0.1,
        warmup_steps=1000
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq=100
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv'),
        tf.keras.callbacks.BackupAndRestore('./backup')
    ]

    # 4. Train
    history = model.fit(
        train_dataset,
        epochs=200,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    return model, history
```
