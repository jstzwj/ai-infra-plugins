# TensorFlow Keras Overview Reference

## Table of Contents

1. [Keras History and Integration](#keras-history-and-integration)
2. [tf.keras Module Structure](#tfkeras-module-structure)
3. [Layer Class](#layer-class)
4. [Model Class](#model-class)
5. [Sequential API](#sequential-api)
6. [Functional API](#functional-api)
7. [Model Subclassing](#model-subclassing)
8. [Layer Weight Management](#layer-weight-management)
9. [Layer Regularization](#layer-regularization)
10. [Layer Constraints](#layer-constraints)
11. [Layer Activation Functions](#layer-activation-functions)
12. [Model Introspection](#model-introspection)
13. [Layer and Model Serialization](#layer-and-model-serialization)
14. [Custom Layers](#custom-layers)
15. [Custom Models](#custom-models)
16. [Keras Backend Compatibility](#keras-backend-compatibility)
17. [Keras Mixed Precision Integration](#keras-mixed-precision-integration)
18. [Multi-Input/Multi-Output Models](#multi-inputmulti-output-models)
19. [Residual Connections and Model Composition](#residual-connections-and-model-composition)

---

## Keras History and Integration

### Keras Origins

Keras was originally developed by Francois Chollet as a high-level neural
networks API, designed to enable fast experimentation with deep learning.
Key milestones:

- **2015 (March)**: Keras first released as an open-source project
- **2017**: Keras adopted as the official high-level API of TensorFlow
  (bundled as `tf.keras`)
- **2019**: TensorFlow 2.0 makes `tf.keras` the primary API, replacing
  `tf.layers`, `tf.slim`, and `tf.contrib`
- **2023**: Keras 3 (previously Keras Core) introduced as a multi-backend
  framework supporting TensorFlow, JAX, and PyTorch

### Keras 2 vs Keras 3

| Feature | Keras 2 (`tf.keras`) | Keras 3 (`keras`) |
|---------|---------------------|-------------------|
| Backend | TensorFlow only | TensorFlow, JAX, PyTorch |
| Location | `tf.keras` | Standalone `keras` package |
| TF Integration | Deep integration | Loosely coupled |
| Status | Stable, maintained | Active development |
| Compatibility | TF2 ecosystem | Multi-framework |

### tf_keras Package

For TensorFlow users who need the TF-specific Keras 2 API, `tf_keras`
is available as a standalone package:

```python
# Option 1: tf.keras (built into TensorFlow)
import tensorflow as tf
from tensorflow import keras

# Option 2: tf_keras (standalone package for TF-specific features)
import tf_keras
```

---

## tf.keras Module Structure

```
tf.keras/
  engine/             # Core Layer, Model, Sequential classes
  layers/             # Built-in layer implementations
  optimizers/         # Optimizer implementations (Adam, SGD, etc.)
  losses.py           # Loss function implementations
  metrics.py          # Metric implementations
  callbacks.py        # Training callbacks
  initializers/       # Weight initializer implementations
  regularizers.py     # Regularization functions
  constraints.py      # Weight constraint implementations
  activations.py      # Activation function implementations
  backend.py          # Backend abstraction layer
  backend_config.py   # Backend configuration
  models.py           # Model utilities (load_model, clone_model)
  saving/             # Model saving/loading
  mixed_precision/    # Mixed precision training
  distribute/         # Distribution strategy integration
  utils/              # Utility functions
  preprocessing/      # Data preprocessing utilities
  applications/       # Pre-trained model applications
  wrappers/           # Wrapper layers (TimeDistributed, Bidirectional)
  legacy_tf_layers/   # TF1 compatibility layers
```

### Key Engine Files

```
tf.keras/engine/
  base_layer.py       # Layer base class
  base_layer_utils.py # Layer utility functions
  training.py         # Model training logic
  functional.py       # Functional API model
  sequential.py       # Sequential model
  input_layer.py      # Input layer (for Functional API)
  input_spec.py       # Input specification
  node.py             # Layer connectivity node
  keras_tensor.py     # KerasTensor for Functional API
  saving.py           # Model saving internals
```

---

## Layer Class

### Overview

`tf.keras.layers.Layer` is the fundamental building block of Keras models.
It encapsulates state (weights) and computation (forward pass).

Defined in `tensorflow/python/keras/engine/base_layer.py`.

### Layer Lifecycle

1. **`__init__()`**: Configure the layer (hyperparameters)
2. **`build(input_shape)`**: Create weights (lazy initialization)
3. **`call(inputs)`**: Define the forward pass computation
4. **`compute_output_shape(input_shape)`**: Declare output shape
5. **`get_config()`**: Return configuration for serialization

### `__init__`

```python
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units=32, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
```

The `**kwargs` can include:
- `name`: Layer name string
- `dtype`: Default dtype for computations and weights
- `trainable`: Whether the layer is trainable (default True)

### `build(input_shape)`

Called automatically the first time the layer is used. Creates weights:

```python
def build(self, input_shape):
    self.kernel = self.add_weight(
        name='kernel',
        shape=(input_shape[-1], self.units),
        initializer='glorot_uniform',
        trainable=True
    )
    self.bias = self.add_weight(
        name='bias',
        shape=(self.units,),
        initializer='zeros',
        trainable=True
    )
    self.built = True  # Set by add_weight automatically
```

### `call(inputs)`

Defines the forward pass:

```python
def call(self, inputs):
    output = tf.matmul(inputs, self.kernel) + self.bias
    if self.activation is not None:
        output = self.activation(output)
    return output
```

The `call` method can accept additional arguments:
- `training`: Boolean indicating training vs inference mode
- `mask`: Mask tensor for masked layers

```python
def call(self, inputs, training=None, mask=None):
    output = tf.matmul(inputs, self.kernel) + self.bias
    if training:
        output = tf.nn.dropout(output, rate=0.5)
    return output
```

### `compute_output_shape(input_shape)`

Returns the output shape for a given input shape:

```python
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.units)
```

### `get_config()` and `from_config()`

For serialization:

```python
def get_config(self):
    config = super().get_config()
    config.update({
        'units': self.units,
        'activation': tf.keras.activations.serialize(self.activation),
    })
    return config

@classmethod
def from_config(cls, config):
    return cls(**config)
```

---

## Model Class

### Overview

`tf.keras.Model` is a subclass of `Layer` that adds training, evaluation,
and prediction capabilities. It is defined in `tensorflow/python/keras/engine/training.py`.

```python
class Model(tf.keras.layers.Layer):
    """Groups layers into an object with training and inference features."""
```

### Model vs Layer

| Feature | Layer | Model |
|---------|-------|-------|
| `call()` | Yes | Yes |
| `build()` | Yes | Yes |
| `fit()` | No | Yes |
| `evaluate()` | No | Yes |
| `predict()` | No | Yes |
| `compile()` | No | Yes |
| `save()` | No | Yes |
| `train_on_batch()` | No | Yes |

### Model Compilation

```python
model.compile(
    optimizer='adam',                          # Optimizer
    loss='sparse_categorical_crossentropy',    # Loss function
    metrics=['accuracy'],                      # Metrics
    loss_weights=None,                         # Multi-output loss weights
    weighted_metrics=None,                     # Weighted metrics
    run_eagerly=None,                          # Force eager execution
    steps_per_execution=1,                     # Batches per tf.function call
    jit_compile=None,                          # Enable XLA compilation
)

# With custom objects
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc'),
    ]
)
```

### Model Training

```python
# Full training with fit()
history = model.fit(
    x=train_images,
    y=train_labels,
    batch_size=32,
    epochs=10,
    validation_data=(val_images, val_labels),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    ],
    verbose=1,
)

# Training with tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_dataset, epochs=10)

# Custom training loop with GradientTape
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Model Evaluation and Prediction

```python
# Evaluate
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)

# Predict
predictions = model.predict(test_images)
predictions = model(test_images, training=False)  # Direct call
```

---

## Sequential API

### Basic Usage

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
```

### Incremental Building

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))

# Remove the last layer
model.pop()
```

### Build and Summary

```python
# Build explicitly with input shape
model.build(input_shape=(None, 28, 28))

# Print model summary
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                Output Shape              Param #
# =================================================================
# flatten (Flatten)           (None, 784)               0
# dense (Dense)               (None, 128)               100480
# dropout (Dropout)           (None, 128)               0
# dense_1 (Dense)             (None, 10)                1290
# =================================================================
# Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
```

### Sequential Limitations

- Cannot create models with multiple inputs or outputs
- Cannot create models with shared layers
- Cannot create models with residual/skip connections
- For these, use the Functional API or Model subclassing

---

## Functional API

### Basic Usage

```python
# Define input
inputs = tf.keras.Input(shape=(28, 28), name='img')

# Define the forward pass
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10)(x)

# Create model
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
```

### Input Layer

```python
# Various input types
input_1d = tf.keras.Input(shape=(784,))                    # 1D input
input_2d = tf.keras.Input(shape=(28, 28, 1))              # 2D image
input_seq = tf.keras.Input(shape=(None, 128))              # Variable-length sequence
input_dict = {
    'title': tf.keras.Input(shape=(100,), name='title'),
    'body': tf.keras.Input(shape=(200,), name='body'),
}
```

### Connecting Layers

```python
# Each layer call creates a new tensor
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
```

### Multiple Inputs

```python
# Model with multiple inputs
title_input = tf.keras.Input(shape=(100,), name='title')
body_input = tf.keras.Input(shape=(200,), name='body')

title_features = tf.keras.layers.Embedding(10000, 64)(title_input)
title_features = tf.keras.layers.LSTM(128)(title_features)

body_features = tf.keras.layers.Embedding(10000, 64)(body_input)
body_features = tf.keras.layers.LSTM(128)(body_features)

# Concatenate features
x = tf.keras.layers.concatenate([title_features, body_features])
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(
    inputs=[title_input, body_input],
    outputs=outputs
)
```

### Multiple Outputs

```python
inputs = tf.keras.Input(shape=(28, 28, 1))

x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Flatten()(x)

# Multiple output heads
class_output = tf.keras.layers.Dense(10, activation='softmax', name='class')(x)
reg_output = tf.keras.layers.Dense(1, name='box')(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=[class_output, reg_output]
)

# Compile with different losses per output
model.compile(
    optimizer='adam',
    loss={
        'class': 'sparse_categorical_crossentropy',
        'box': 'mse',
    },
    loss_weights={'class': 1.0, 'box': 0.5},
    metrics={'class': ['accuracy'], 'box': ['mae']}
)
```

### Shared Layers

```python
# Shared embedding layer
embedding = tf.keras.layers.Embedding(10000, 128)

# Use the same layer for two different inputs
input_a = tf.keras.Input(shape=(100,))
input_b = tf.keras.Input(shape=(100,))

features_a = embedding(input_a)
features_b = embedding(input_b)

# The embedding weights are shared between both paths
```

---

## Model Subclassing

### Basic Model Subclassing

```python
class MyModel(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)

# Usage
model = MyModel(hidden_dim=128, output_dim=10)
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=5)
```

### Model with Custom Training Step

```python
class CustomModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```

---

## Layer Weight Management

### add_weight

Create weights within a layer:

```python
def build(self, input_shape):
    self.kernel = self.add_weight(
        name='kernel',
        shape=(input_shape[-1], self.units),
        initializer=tf.keras.initializers.GlorotUniform(),
        regularizer=tf.keras.regularizers.l2(0.01),
        constraint=tf.keras.constraints.MaxNorm(3.0),
        trainable=True,
        dtype=self.dtype,
    )

    self.bias = self.add_weight(
        name='bias',
        shape=(self.units,),
        initializer='zeros',
        trainable=True,
    )
```

### Trainable and Non-Trainable Weights

```python
# Access weights
layer = tf.keras.layers.Dense(32)
layer.build((None, 64))

layer.weights                # All weights (trainable + non-trainable)
layer.trainable_weights      # Only trainable weights
layer.non_trainable_weights  # Only non-trainable weights

# Mark a layer as non-trainable
layer.trainable = False      # Freezes all weights in this layer

# Per-weight control
layer = tf.keras.layers.BatchNormalization()
# kernel (trainable), moving_mean (non-trainable), moving_var (non-trainable)
```

### get_weights and set_weights

```python
# Get weights as numpy arrays
weights = layer.get_weights()  # List of numpy arrays

# Set weights from numpy arrays
layer.set_weights(weights)

# Transfer weights between layers
layer1 = tf.keras.layers.Dense(64)
layer1.build((None, 32))
layer2 = tf.keras.layers.Dense(64)
layer2.build((None, 32))
layer2.set_weights(layer1.get_weights())
```

### Weight Initialization

```python
# String identifiers
kernel_initializer='glorot_uniform'    # Xavier uniform
kernel_initializer='glorot_normal'     # Xavier normal
kernel_initializer='he_normal'         # Kaiming normal (for ReLU)
kernel_initializer='he_uniform'        # Kaiming uniform
kernel_initializer='lecun_normal'      # LeCun normal (for SELU)
kernel_initializer='zeros'             # All zeros
kernel_initializer='ones'              # All ones
kernel_initializer='orthogonal'        # Orthogonal matrix

# Class-based initializers
tf.keras.initializers.GlorotUniform(seed=42)
tf.keras.initializers.HeNormal()
tf.keras.initializers.RandomNormal(stddev=0.01)
tf.keras.initializers.TruncatedNormal(stddev=0.02)
tf.keras.initializers.VarianceScaling(
    scale=2.0, mode='fan_in', distribution='truncated_normal'
)
```

---

## Layer Regularization

### Types of Regularization

```python
# Kernel (weight) regularization
tf.keras.layers.Dense(
    64,
    kernel_regularizer=tf.keras.regularizers.l2(0.01)
)

# Bias regularization
tf.keras.layers.Dense(
    64,
    bias_regularizer=tf.keras.regularizers.l1(0.01)
)

# Activity (output) regularization
tf.keras.layers.Dense(
    64,
    activity_regularizer=tf.keras.regularizers.l1(0.01)
)
```

### Built-in Regularizers

```python
# L1 regularization (Lasso)
tf.keras.regularizers.l1(l=0.01)

# L2 regularization (Ridge)
tf.keras.regularizers.l2(l=0.01)

# L1 + L2 combined (Elastic Net)
tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

### Custom Regularizer

```python
class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength=0.01):
        self.strength = strength

    def __call__(self, weight_matrix):
        # Encourage weight columns to be orthogonal
        product = tf.matmul(weight_matrix, weight_matrix, transpose_b=True)
        identity = tf.eye(tf.shape(product)[0])
        return self.strength * tf.reduce_sum(tf.square(product - identity))

    def get_config(self):
        return {'strength': self.strength}

# Usage
layer = tf.keras.layers.Dense(
    64,
    kernel_regularizer=OrthogonalRegularizer(0.01)
)
```

### Accessing Regularization Losses

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, kernel_regularizer='l2'),
    tf.keras.layers.Dense(10, kernel_regularizer='l2'),
])

# Regularization losses are collected per layer
model.losses  # List of regularization loss tensors

# Include in training
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        loss += sum(model.losses)  # Add regularization losses
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

---

## Layer Constraints

### Built-in Constraints

```python
# MaxNorm: constrain weight matrix norm
tf.keras.layers.Dense(64, kernel_constraint=tf.keras.constraints.MaxNorm(3.0))

# NonNeg: constrain weights to be non-negative
tf.keras.layers.Dense(64, kernel_constraint=tf.keras.constraints.NonNeg())

# UnitNorm: constrain weights to have unit norm
tf.keras.layers.Dense(64, kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))

# MinMaxNorm: constrain weight norm to a range
tf.keras.layers.Dense(64,
    kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0))

# RadialNorm: constrain weights to have bounded norm
tf.keras.layers.Dense(64,
    kernel_constraint=tf.keras.constraints.RadialNorm(max_norm=2.0))
```

### Applying Constraints to Bias

```python
tf.keras.layers.Dense(
    64,
    kernel_constraint=tf.keras.constraints.NonNeg(),
    bias_constraint=tf.keras.constraints.MaxNorm(1.0)
)
```

---

## Layer Activation Functions

### Built-in Activations

```python
# String identifiers
tf.keras.layers.Dense(64, activation='relu')
tf.keras.layers.Dense(64, activation='sigmoid')
tf.keras.layers.Dense(64, activation='tanh')
tf.keras.layers.Dense(64, activation='softmax')
tf.keras.layers.Dense(64, activation='selu')
tf.keras.layers.Dense(64, activation='elu')
tf.keras.layers.Dense(64, activation='linear')  # No activation
```

### Available Activations

| Name | Function | Use Case |
|------|----------|----------|
| `relu` | `max(0, x)` | Hidden layers (default) |
| `sigmoid` | `1 / (1 + exp(-x))` | Binary classification |
| `tanh` | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` | Hidden layers (RNNs) |
| `softmax` | `exp(x) / sum(exp(x))` | Multi-class classification |
| `selu` | Scaled ELU | Self-normalizing networks |
| `elu` | Exponential Linear Unit | Hidden layers |
| `softplus` | `log(1 + exp(x))` | Smooth ReLU alternative |
| `softsign` | `x / (1 + abs(x))` | Hidden layers |
| `relu6` | `min(max(0, x), 6)` | Mobile networks |
| `swish` | `x * sigmoid(x)` | Modern hidden layers |
| `mish` | `x * tanh(softplus(x))` | Modern hidden layers |
| `gelu` | Gaussian Error Linear Unit | Transformer models |
| `leaky_relu` | `max(alpha*x, x)` | GANs |
| `prelu` | Parametric ReLU | Learned negative slope |

### Advanced Activation Layers

```python
# LeakyReLU
tf.keras.layers.LeakyReLU(alpha=0.2)

# PReLU (learnable negative slope)
tf.keras.layers.PReLU()

# ELU
tf.keras.layers.ELU(alpha=1.0)

# Thresholded ReLU
tf.keras.layers.ThresholdedReLU(theta=1.0)
```

### Using Activations as Functions

```python
# tf.keras.activations module
x = tf.keras.activations.relu(x)
x = tf.keras.activations.sigmoid(x)
x = tf.keras.activations.softmax(x, axis=-1)

# With custom parameters
x = tf.nn.leaky_relu(x, alpha=0.2)  # Use tf.nn directly
```

---

## Model Introspection

### Model Summary

```python
model.summary()

# Print to string
string = tf.keras.utils.model_to_dot(model).to_string()

# Summary with nested models
model.summary(line_length=120, expand_nested=True)
```

### Accessing Layers

```python
# All layers
model.layers  # List of Layer objects

# Inputs and outputs
model.inputs   # List of input tensors (Functional API)
model.outputs  # List of output tensors (Functional API)

# Get a specific layer by name
layer = model.get_layer('dense_1')

# Get layer by index
layer = model.layers[2]

# Layer input/output shapes
layer.input_shape    # Input shape
layer.output_shape   # Output shape
layer.input          # Input tensor (Functional API)
layer.output         # Output tensor (Functional API)
```

### Variables

```python
# All variables
model.variables                # All trainable + non-trainable
model.trainable_variables      # Only trainable
model.non_trainable_variables  # Only non-trainable

# Count parameters
model.count_params()           # Total
# Per-layer
for layer in model.layers:
    print(f"{layer.name}: {layer.count_params()} params")
```

---

## Layer and Model Serialization

### get_config and from_config

```python
# Get configuration
config = layer.get_config()
# {'units': 64, 'activation': 'relu', 'use_bias': True, ...}

# Recreate from config
new_layer = type(layer).from_config(config)
```

### to_json and to_yaml

```python
# Model to JSON
json_string = model.to_json()
print(json_string)

# Model from JSON
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string)

# Model to YAML
yaml_string = model.to_yaml()

# Model from YAML
from tensorflow.keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

### Model Saving (Full Model)

```python
# Save entire model (architecture + weights + optimizer state)
model.save('my_model.keras')           # Keras 3 format
model.save('my_model')                  # SavedModel format
model.save('my_model.h5')              # HDF5 format (legacy)

# Load model
model = tf.keras.models.load_model('my_model.keras')
model = tf.keras.models.load_model('my_model')

# Save weights only
model.save_weights('weights.keras')
model.save_weights('weights.h5')

# Load weights
model.load_weights('weights.keras')
model.load_weights('weights.h5')

# Load weights with custom objects
model = tf.keras.models.load_model(
    'my_model.keras',
    custom_objects={'MyLayer': MyLayer, 'custom_loss': custom_loss}
)
```

### Saving with Custom Layers

```python
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Must implement get_config for saving
# Must register custom objects
tf.keras.utils.get_custom_objects()['MyLayer'] = MyLayer
```

---

## Custom Layers

### Complete Custom Layer Example

```python
class DenseWithActivation(tf.keras.layers.Layer):
    """A dense layer with built-in activation tracking."""

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            regularizer=None,
            constraint=None,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
        )
        self.built = True

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
        })
        return config
```

### Layer with Training Mode

```python
class DenseWithDropout(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel', shape=(input_shape[-1], self.units))
        self.bias = self.add_weight('bias', shape=(self.units,))

    def call(self, inputs, training=False):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if training:
            output = tf.nn.dropout(output, rate=self.dropout_rate)
        return output
```

### Layer with Masking Support

```python
class MaskedDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel', shape=(input_shape[-1], self.units))
        self.bias = self.add_weight('bias', shape=(self.units,))

    def call(self, inputs, mask=None):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if mask is not None:
            output = output * tf.cast(tf.expand_dims(mask, -1), output.dtype)
        return output

    def compute_mask(self, inputs, mask=None):
        return mask
```

### Layer with Multiple Inputs

```python
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        # input_shape is a list of shapes for multiple inputs
        self.query_dense = tf.keras.layers.Dense(input_shape[0][-1])
        self.key_dense = tf.keras.layers.Dense(input_shape[1][-1])

    def call(self, inputs):
        query, key, value = inputs
        # Attention computation
        scores = tf.matmul(query, key, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, value)
```

---

## Custom Models

### Training Step Customization

```python
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_images = data

        # Generate fake images
        batch_size = tf.shape(real_images)[0]
        random_latent = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated = self.generator(random_latent, training=True)

            real_pred = self.discriminator(real_images, training=True)
            fake_pred = self.discriminator(generated, training=True)

            g_loss = self.loss_fn(
                tf.ones_like(fake_pred), fake_pred)
            d_loss = (self.loss_fn(tf.ones_like(real_pred), real_pred) +
                      self.loss_fn(tf.zeros_like(fake_pred), fake_pred))

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))

        return {'g_loss': g_loss, 'd_loss': d_loss}
```

### Predict Step Customization

```python
class ModelWithPreprocessing(tf.keras.Model):
    def predict_step(self, data):
        x = data
        # Apply preprocessing
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.resize(x, [224, 224])
        return self(x, training=False)
```

---

## Keras Backend Compatibility

### tf.keras.backend Module

The backend module provides a layer of abstraction over the underlying
tensor operations:

```python
import tensorflow.keras.backend as K

# Common operations
K.sum(x, axis=0)          # Sum
K.mean(x, axis=-1)        # Mean
K.reshape(x, shape)       # Reshape
K.batch_flatten(x)        # Flatten batch
K.concatenate([x, y])     # Concatenate
K.zeros(shape)            # Zeros
K.ones(shape)             # Ones
K.random_normal(shape)    # Random normal

# Backend configuration
K.epsilon()               # 1e-7 (fuzz factor)
K.floatx()                # 'float32' (default float type)
K.image_data_format()     # 'channels_last' or 'channels_first'

# Set backend configuration
K.set_floatx('float16')
K.set_image_data_format('channels_first')
K.set_epsilon(1e-5)
```

### Backend-Agnostic Layer Implementation

```python
class BackendAgnosticLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Use backend functions for portability
        return K.relu(K.bias_add(K.dot(inputs, self.kernel), self.bias))
```

---

## Keras Mixed Precision Integration

### Overview

Mixed precision training uses float16 (or bfloat16) for computations while
maintaining float32 master weights for numerical stability.

### Enabling Mixed Precision

```python
# Policy-based mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Or policy for a specific layer
policy = tf.keras.mixed_precision.Policy('mixed_float16')
layer = tf.keras.layers.Dense(512, dtype=policy)

# Available policies:
# 'float32'        - Standard single precision
# 'mixed_float16'  - Float16 compute, float32 master weights
# 'mixed_bfloat16' - BFloat16 compute, float32 master weights
# 'float16'        - Pure float16 (no loss scaling)
```

### Mixed Precision in Custom Layers

```python
class MixedPrecisionLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Compute dtype is float16; variable dtype is float32
        self.kernel = self.add_weight(
            'kernel',
            shape=(input_shape[-1], self.units),
            dtype='float32',  # Master weight in float32
        )

    def call(self, inputs):
        # inputs may be float16
        # Cast kernel to compute dtype for matmul
        kernel = tf.cast(self.kernel, inputs.dtype)
        return tf.matmul(inputs, kernel)
```

### Loss Scaling

When using `mixed_float16`, a loss scaling wrapper is automatically applied:

```python
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    tf.keras.optimizers.Adam(0.001)
)

# The optimizer automatically:
# 1. Scales up the loss before gradient computation
# 2. Scales down the gradients after computation
# 3. Adjusts the scale dynamically
```

---

## Multi-Input/Multi-Output Models

### Multi-Input Model

```python
# Text and metadata input
text_input = tf.keras.Input(shape=(100,), name='text')
meta_input = tf.keras.Input(shape=(10,), name='meta')

# Text processing branch
text_features = tf.keras.layers.Embedding(10000, 64)(text_input)
text_features = tf.keras.layers.LSTM(64)(text_features)

# Metadata processing branch
meta_features = tf.keras.layers.Dense(32, activation='relu')(meta_input)

# Combine branches
combined = tf.keras.layers.concatenate([text_features, meta_features])
combined = tf.keras.layers.Dense(64, activation='relu')(combined)
output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

model = tf.keras.Model(
    inputs=[text_input, meta_input],
    outputs=output
)

# Training with named inputs
model.fit(
    {'text': text_data, 'meta': meta_data},
    labels,
    epochs=10
)
```

### Multi-Output Model

```python
inputs = tf.keras.Input(shape=(224, 224, 3))

# Shared backbone
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Flatten()(x)

# Classification head
class_output = tf.keras.layers.Dense(
    100, activation='softmax', name='classification'
)(x)

# Bounding box regression head
bbox_output = tf.keras.layers.Dense(
    4, activation='sigmoid', name='bbox'
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs={'classification': class_output, 'bbox': bbox_output}
)

# Compile with per-output losses
model.compile(
    optimizer='adam',
    loss={
        'classification': 'categorical_crossentropy',
        'bbox': 'mse',
    },
    loss_weights={
        'classification': 1.0,
        'bbox': 10.0,
    },
    metrics={
        'classification': 'accuracy',
        'bbox': 'mae',
    }
)

# Train with named outputs
model.fit(
    images,
    {'classification': class_labels, 'bbox': bbox_labels},
    epochs=20
)
```

---

## Residual Connections and Model Composition

### Residual Block

```python
def residual_block(x, filters, kernel_size=3, stride=1):
    """Create a residual block."""
    # Shortcut connection
    shortcut = x

    # Main path
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=stride, padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Match dimensions for shortcut
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding='same'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add residual
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x
```

### Building a ResNet-like Model

```python
inputs = tf.keras.Input(shape=(224, 224, 3))

# Stem
x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

# Residual blocks
x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 128, stride=2)
x = residual_block(x, 128)
x = residual_block(x, 256, stride=2)
x = residual_block(x, 256)

# Head
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1000, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
```

### Encoder-Decoder Pattern

```python
# Encoder
encoder_input = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
encoder_output = tf.keras.layers.Flatten()(x)

encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')

# Decoder
decoder_input = tf.keras.Input(shape=(encoder_output.shape[-1],))
x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(decoder_input)
x = tf.keras.layers.Reshape((7, 7, 64))(x)
x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu',
                                      padding='same')(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation='relu',
                                      padding='same')(x)
decoder_output = tf.keras.layers.Conv2D(1, 3, activation='sigmoid',
                                         padding='same')(x)

decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')

# Autoencoder
autoencoder_input = tf.keras.Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = tf.keras.Model(autoencoder_input, decoded, name='autoencoder')
```

### Feature Pyramid Pattern

```python
def feature_pyramid(backbone_outputs):
    """Build feature pyramid from backbone outputs."""
    pyramid = []

    # Top-down pathway
    x = tf.keras.layers.Conv2D(256, 1)(backbone_outputs[-1])

    for backbone_feat in reversed(backbone_outputs[:-1]):
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2D(256, 1)(x)
        lateral = tf.keras.layers.Conv2D(256, 1)(backbone_feat)
        x = tf.keras.layers.Add()([x, lateral])
        x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
        pyramid.append(x)

    return pyramid
```

---

## Summary

TensorFlow Keras provides a comprehensive high-level API with multiple
programming paradigms:

- **Sequential API**: Simple linear stack of layers
- **Functional API**: Flexible model definition with multiple
  inputs/outputs and shared layers
- **Model Subclassing**: Full control over forward pass and training loop

Key components:
- **Layer**: The fundamental building block with build/call/get_config
  lifecycle
- **Model**: Layer subclass with fit/evaluate/predict capabilities
- **Callbacks**: Training hooks for monitoring, checkpointing, and
  scheduling
- **Regularizers**: L1, L2, L1L2 for weight penalty
- **Constraints**: MaxNorm, NonNeg, UnitNorm for weight projection
- **Mixed Precision**: Float16/BFloat16 training with automatic loss scaling
- **Serialization**: Full model saving/loading in Keras, SavedModel,
  and HDF5 formats

The API supports complex architectures including residual connections,
encoder-decoder patterns, attention mechanisms, and feature pyramids,
while maintaining clean code organization through object-oriented design.
