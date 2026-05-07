# TensorFlow Keras Layers Reference

This document provides comprehensive reference documentation for all Keras layer types in TensorFlow. Each layer includes its class signature, all parameters with defaults, input/output shapes, usage examples, and common patterns.

---

## Table of Contents

1. [Core Layers](#core-layers)
2. [Convolution Layers](#convolution-layers)
3. [Pooling Layers](#pooling-layers)
4. [RNN Layers](#rnn-layers)
5. [Attention Layers](#attention-layers)
6. [Normalization Layers](#normalization-layers)
7. [Regularization Layers](#regularization-layers)
8. [Activation Layers](#activation-layers)
9. [Reshaping Layers](#reshaping-layers)
10. [Merge Layers](#merge-layers)
11. [Advanced Layers](#advanced-layers)
12. [Preprocessing Layers](#preprocessing-layers)

---

## Core Layers

### Dense

Fully-connected layer that implements the operation `output = activation(dot(input, kernel) + bias)`.

```python
tf.keras.layers.Dense(
    units,                          # Positive integer, dimensionality of the output space
    activation=None,                # Activation function to use (e.g. 'relu', 'sigmoid', 'tanh')
    use_bias=True,                  # Boolean, whether the layer uses a bias vector
    kernel_initializer='glorot_uniform',  # Initializer for the kernel weights matrix
    bias_initializer='zeros',       # Initializer for the bias vector
    kernel_regularizer=None,        # Regularizer function applied to the kernel weights matrix
    bias_regularizer=None,          # Regularizer function applied to the bias vector
    activity_regularizer=None,      # Regularizer function applied to the output of the layer
    kernel_constraint=None,         # Constraint function applied to the kernel weights matrix
    bias_constraint=None,           # Constraint function applied to the bias vector
    lora_rank=None,                 # Optional int. If set, the layer's forward pass will implement
                                    # LoRA (Low-Rank Adaptation) with the provided rank
)
```

**Input shape:** `(batch_size, input_dim)` or n-D tensor with last dim `input_dim`

**Output shape:** `(batch_size, units)`

**Usage:**
```python
# Basic dense layer
layer = tf.keras.layers.Dense(64, activation='relu')

# In a Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# With regularization
dense = tf.keras.layers.Dense(
    256,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.01),
    bias_regularizer=tf.keras.regularizers.l1(0.01)
)

# With LoRA (Low-Rank Adaptation)
dense_lora = tf.keras.layers.Dense(512, lora_rank=4)
```

### Activation

Applies an activation function to an output.

```python
tf.keras.layers.Activation(
    activation,    # Activation function to use (string name or callable)
    **kwargs
)
```

**Input shape:** Same as output shape

**Output shape:** Same as input shape

**Usage:**
```python
# Using string identifier
layer = tf.keras.layers.Activation('relu')

# Using callable
layer = tf.keras.layers.Activation(tf.nn.leaky_relu)

# In a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Activation('tanh')
])
```

### Dropout

Applies Dropout to the input to prevent overfitting. During training, randomly sets input units to 0 with frequency `rate`. Inputs not set to 0 are scaled up by `1/(1 - rate)`.

```python
tf.keras.layers.Dropout(
    rate,                  # Float between 0 and 1. Fraction of the input units to drop
    noise_shape=None,      # 1D integer tensor representing the shape of the
                           # binary dropout mask that will be multiplied with the input
    seed=None,             # A Python integer to use as random seed
    **kwargs
)
```

**Input shape:** Arbitrary

**Output shape:** Same as input shape

**Usage:**
```python
# Standard dropout
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# With noise_shape for spatial dropout on 1D
dropout = tf.keras.layers.Dropout(0.5, noise_shape=(batch_size, 1, features))

# Deterministic dropout with seed
dropout = tf.keras.layers.Dropout(0.3, seed=42)
```

### Flatten

Flattens the input without affecting the batch size.

```python
tf.keras.layers.Flatten(
    data_format=None,    # 'channels_last' or 'channels_first'
    **kwargs
)
```

**Input shape:** `(batch_size, ...)` - arbitrary dimensions

**Output shape:** `(batch_size, product_of_non_batch_dims)`

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# With data_format
flatten = tf.keras.layers.Flatten(data_format='channels_first')
```

### Input

Used to instantiate a Keras tensor as the entry point of a model.

```python
tf.keras.layers.Input(
    shape=None,             # A shape tuple (integer or None), not including the batch size
    batch_size=None,        # Optional input batch size (integer or None)
    name=None,              # Optional name string for the layer
    dtype=None,             # Data type of the input (e.g. 'float32', 'int32')
    sparse=False,           # Boolean, whether the placeholder created is sparse
    tensor=None,            # Optional existing tensor to wrap into the Input layer
    ragged=False,           # Boolean, whether the placeholder created is ragged
    type_spec=None,         # Optional TypeSpec for the input
    **kwargs
)
```

**Usage:**
```python
# Basic input
inputs = tf.keras.Input(shape=(784,))

# Image input
img_input = tf.keras.Input(shape=(28, 28, 1), name='img')

# Multiple inputs
title_input = tf.keras.Input(shape=(100,), name='title')
body_input = tf.keras.Input(shape=(500,), name='body')

# With specific dtype
input_layer = tf.keras.Input(shape=(10,), dtype='float32')

# Sparse input
sparse_input = tf.keras.Input(shape=(1000,), sparse=True)

# Ragged input for variable-length sequences
ragged_input = tf.keras.Input(shape=(None,), ragged=True)
```

### Reshape

Reshapes an output to a certain shape.

```python
tf.keras.layers.Reshape(
    target_shape,    # Target shape (tuple of integers). Does not include the batch axis
    **kwargs
)
```

**Usage:**
```python
# Reshape 1D to 2D
model = tf.keras.Sequential([
    tf.keras.layers.Dense(784, input_shape=(784,)),
    tf.keras.layers.Reshape((28, 28, 1))
])

# Use -1 for inferred dimension
reshape = tf.keras.layers.Reshape((-1, 128))  # batch dimension inferred
```

### Permute

Permutes the dimensions of the input according to a given pattern.

```python
tf.keras.layers.Permute(
    dims,    # Tuple of integers representing the permutation pattern.
             # Does not include the batch dimension. Indexing starts at 1
    **kwargs
)
```

**Usage:**
```python
# Swap height and width
layer = tf.keras.layers.Permute((2, 1))  # Input (batch, 3, 4) -> (batch, 4, 3)

# For sequence models - swap time and features
permute = tf.keras.layers.Permute((2, 1))  # (batch, timesteps, features) -> (batch, features, timesteps)
```

### RepeatVector

Repeats the input n times.

```python
tf.keras.layers.RepeatVector(
    n,    # Integer, repetition factor
    **kwargs
)
```

**Input shape:** `(batch_size, features)`

**Output shape:** `(batch_size, n, features)`

**Usage:**
```python
# For encoder-decoder models
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    tf.keras.layers.RepeatVector(10),  # Repeat 10 times for sequence output
    tf.keras.layers.LSTM(64, return_sequences=True)
])
```

### Lambda

Wraps arbitrary expressions as a Layer object.

```python
tf.keras.layers.Lambda(
    function,                    # The function to be evaluated. Takes input tensor as first argument
    output_shape=None,           # Expected output shape from function
    mask=None,                   # Either None (no masking) or a callable
    arguments=None,              # Optional dictionary of keyword arguments to pass to function
    **kwargs
)
```

**Usage:**
```python
# Add a constant
add_layer = tf.keras.layers.Lambda(lambda x: x + 1.0)

# With arguments
merger = tf.keras.layers.Lambda(
    lambda x, a: x + a,
    arguments={'a': 0.5}
)

# Custom transformation
layer = tf.keras.layers.Lambda(
    lambda x: tf.math.reduce_mean(x, axis=1),
    output_shape=lambda s: (s[0],)
)

# Lambda with tf operations
norm_layer = tf.keras.layers.Lambda(
    lambda x: tf.math.l2_normalize(x, axis=-1)
)
```

### ActivityRegularization

Applies an update to the cost function based on input activity.

```python
tf.keras.layers.ActivityRegularization(
    l1=0.0,     # L1 regularization factor (positive float)
    l2=0.0,     # L2 regularization factor (positive float)
    **kwargs
)
```

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.ActivityRegularization(l1=0.01, l2=0.01),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Masking

Masks a sequence by using a mask value to skip timesteps.

```python
tf.keras.layers.Masking(
    mask_value=0.0,    # The mask value to skip. For each timestep in the input,
                       # if all values equal mask_value, the timestep is masked
    **kwargs
)
```

**Usage:**
```python
# Mask padding in sequences
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.0, input_shape=(10, 8)),
    tf.keras.layers.LSTM(32)
])
```

### SpatialDropout1D

Drops entire 1D feature maps instead of individual elements.

```python
tf.keras.layers.SpatialDropout1D(
    rate,    # Float between 0 and 1. Fraction of the input units to drop
    **kwargs
)
```

**Input shape:** `(batch_size, timesteps, features)`

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=100),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(64)
])
```

### SpatialDropout2D

Drops entire 2D feature maps instead of individual elements.

```python
tf.keras.layers.SpatialDropout2D(
    rate,                # Float between 0 and 1
    data_format=None,    # 'channels_last' or 'channels_first'
    **kwargs
)
```

**Input shape:**
- channels_last: `(batch_size, rows, cols, channels)`
- channels_first: `(batch_size, channels, rows, cols)`

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.SpatialDropout2D(0.25),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### SpatialDropout3D

Drops entire 3D feature maps.

```python
tf.keras.layers.SpatialDropout3D(
    rate,                # Float between 0 and 1
    data_format=None,    # 'channels_last' or 'channels_first'
    **kwargs
)
```

**Input shape:**
- channels_last: `(batch_size, dim1, dim2, dim3, channels)`
- channels_first: `(batch_size, channels, dim1, dim2, dim3)`

---

## Convolution Layers

### Conv1D

1D convolution layer (e.g., temporal convolution).

```python
tf.keras.layers.Conv1D(
    filters,                          # Integer, the dimensionality of the output space
    kernel_size,                      # Integer or tuple/list of single integer, length of 1D conv window
    strides=1,                        # Integer or tuple/list of single integer
    padding='valid',                  # 'valid', 'same', or 'causal'
    data_format='channels_last',      # 'channels_last' or 'channels_first'
    dilation_rate=1,                  # Integer or tuple/list of single integer
    groups=1,                         # Positive integer, number of groups for grouped convolution
    activation=None,                  # Activation function
    use_bias=True,                    # Boolean
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Input shape:** `(batch_size, length, channels)` (channels_last)

**Output shape:** `(batch_size, new_length, filters)` (channels_last)

**Usage:**
```python
# Text classification with 1D convolution
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=100),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Causal convolution for time series
causal_conv = tf.keras.layers.Conv1D(64, 3, padding='causal', dilation_rate=2)

# Dilated convolution
dilated = tf.keras.layers.Conv1D(32, 3, dilation_rate=4, padding='same')
```

### Conv2D

2D convolution layer (e.g., spatial convolution over images).

```python
tf.keras.layers.Conv2D(
    filters,                          # Integer, dimensionality of the output space
    kernel_size,                      # Integer or tuple of 2 integers
    strides=(1, 1),                   # Tuple of 2 integers or single integer
    padding='valid',                  # 'valid' or 'same'
    data_format=None,                 # 'channels_last' or 'channels_first'
    dilation_rate=(1, 1),             # Tuple of 2 integers or single integer
    groups=1,                         # Positive integer
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Input shape:** `(batch_size, height, width, channels)` (channels_last)

**Output shape:** `(batch_size, new_height, new_width, filters)` (channels_last)

**Usage:**
```python
# Basic Conv2D
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# With dilation
dilated = tf.keras.layers.Conv2D(64, 3, dilation_rate=2, padding='same')

# With stride
strided = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')

# Grouped convolution
grouped = tf.keras.layers.Conv2D(64, 3, groups=2, padding='same')
```

### Conv3D

3D convolution layer (e.g., spatial convolution over volumes).

```python
tf.keras.layers.Conv3D(
    filters,                          # Integer
    kernel_size,                      # Integer or tuple of 3 integers
    strides=(1, 1, 1),               # Tuple of 3 integers
    padding='valid',                  # 'valid' or 'same'
    data_format=None,                 # 'channels_last' or 'channels_first'
    dilation_rate=(1, 1, 1),         # Tuple of 3 integers
    groups=1,                         # Positive integer
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Input shape:** `(batch_size, depth, height, width, channels)` (channels_last)

**Usage:**
```python
# 3D convolution for video data
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu',
                           input_shape=(16, 112, 112, 3)),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling3D(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Conv2DTranspose

Transposed convolution layer (sometimes called deconvolution).

```python
tf.keras.layers.Conv2DTranspose(
    filters,                          # Integer
    kernel_size,                      # Integer or tuple of 2 integers
    strides=(1, 1),                   # Tuple of 2 integers
    padding='valid',                  # 'valid' or 'same'
    output_padding=None,              # Integer or tuple of 2 integers, amount of padding
    data_format=None,                 # 'channels_last' or 'channels_first'
    dilation_rate=(1, 1),            # Tuple of 2 integers
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Usage:**
```python
# U-Net decoder path
up_conv = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')

# Autoencoder
model = tf.keras.Sequential([
    # Encoder
    tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', input_shape=(28, 28, 1)),
    # Decoder
    tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same'),
    tf.keras.layers.Conv2D(1, 3, padding='same')
])
```

### Conv3DTranspose

Transposed 3D convolution layer.

```python
tf.keras.layers.Conv3DTranspose(
    filters,                          # Integer
    kernel_size,                      # Integer or tuple of 3 integers
    strides=(1, 1, 1),               # Tuple of 3 integers
    padding='valid',                  # 'valid' or 'same'
    output_padding=None,              # Integer or tuple of 3 integers
    data_format=None,
    dilation_rate=(1, 1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

### DepthwiseConv1D

Depthwise 1D convolution. Applies a single convolutional filter per input channel.

```python
tf.keras.layers.DepthwiseConv1D(
    kernel_size,                      # Integer or tuple of single integer
    strides=1,                        # Integer or tuple of single integer
    padding='valid',                  # 'valid' or 'same'
    depth_multiplier=1,               # Number of depthwise convolution output channels per input channel
    data_format=None,                 # 'channels_last' or 'channels_first'
    dilation_rate=1,                  # Integer or tuple of single integer
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

### DepthwiseConv2D

Depthwise 2D convolution. Each input channel is convolved separately.

```python
tf.keras.layers.DepthwiseConv2D(
    kernel_size,                      # Integer or tuple of 2 integers
    strides=(1, 1),                   # Tuple of 2 integers
    padding='valid',                  # 'valid' or 'same'
    depth_multiplier=1,               # Number of output channels per input channel
    data_format=None,                 # 'channels_last' or 'channels_first'
    dilation_rate=(1, 1),            # Tuple of 2 integers
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Usage:**
```python
# MobileNet-style depthwise separable convolution
model = tf.keras.Sequential([
    tf.keras.layers.DepthwiseConv2D(3, padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, 1),  # Pointwise convolution
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU()
])
```

### SeparableConv1D

Depthwise separable 1D convolution. Consists of a depthwise convolution followed by a pointwise (1x1) convolution.

```python
tf.keras.layers.SeparableConv1D(
    filters,                          # Integer
    kernel_size,                      # Integer or tuple of single integer
    strides=1,                        # Integer or tuple of single integer
    padding='valid',                  # 'valid' or 'same'
    data_format=None,
    dilation_rate=1,
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    pointwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

### SeparableConv2D

Depthwise separable 2D convolution.

```python
tf.keras.layers.SeparableConv2D(
    filters,                          # Integer
    kernel_size,                      # Integer or tuple of 2 integers
    strides=(1, 1),                   # Tuple of 2 integers
    padding='valid',                  # 'valid' or 'same'
    data_format=None,
    dilation_rate=(1, 1),
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    pointwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Usage:**
```python
# MobileNet-style block
model = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(64, 3, padding='same', input_shape=(224, 224, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.SeparableConv2D(128, 3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU()
])
```

---

## Pooling Layers

### MaxPooling1D

Max pooling operation for 1D temporal data.

```python
tf.keras.layers.MaxPooling1D(
    pool_size=2,                # Integer or tuple of single integer, size of the max pooling window
    strides=None,               # Integer or tuple of single integer. If None, defaults to pool_size
    padding='valid',            # 'valid' or 'same'
    data_format='channels_last' # 'channels_last' or 'channels_first'
)
```

**Input shape:** `(batch_size, timesteps, features)` (channels_last)

**Output shape:** `(batch_size, reduced_timesteps, features)` (channels_last)

### MaxPooling2D

Max pooling operation for 2D spatial data.

```python
tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),           # Integer or tuple of 2 integers
    strides=None,               # Integer or tuple of 2 integers. If None, defaults to pool_size
    padding='valid',            # 'valid' or 'same'
    data_format=None            # 'channels_last' or 'channels_first'
)
```

**Input shape:** `(batch_size, height, width, channels)` (channels_last)

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])
```

### MaxPooling3D

Max pooling operation for 3D data.

```python
tf.keras.layers.MaxPooling3D(
    pool_size=(2, 2, 2),        # Integer or tuple of 3 integers
    strides=None,               # Integer or tuple of 3 integers. If None, defaults to pool_size
    padding='valid',            # 'valid' or 'same'
    data_format=None            # 'channels_last' or 'channels_first'
)
```

### AveragePooling1D

Average pooling for 1D temporal data.

```python
tf.keras.layers.AveragePooling1D(
    pool_size=2,                # Integer
    strides=None,               # Integer. Defaults to pool_size
    padding='valid',            # 'valid' or 'same'
    data_format='channels_last'
)
```

### AveragePooling2D

Average pooling for 2D spatial data.

```python
tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),           # Integer or tuple of 2 integers
    strides=None,               # Defaults to pool_size
    padding='valid',            # 'valid' or 'same'
    data_format=None
)
```

### AveragePooling3D

Average pooling for 3D data.

```python
tf.keras.layers.AveragePooling3D(
    pool_size=(2, 2, 2),        # Integer or tuple of 3 integers
    strides=None,               # Defaults to pool_size
    padding='valid',            # 'valid' or 'same'
    data_format=None
)
```

### GlobalMaxPooling1D

Global max pooling operation for 1D temporal data.

```python
tf.keras.layers.GlobalMaxPooling1D(
    data_format='channels_last',    # 'channels_last' or 'channels_first'
    keepdims=False                  # Boolean. If True, retains reduced dimensions with length 1
)
```

**Input shape:** `(batch_size, timesteps, features)` (channels_last)

**Output shape:** `(batch_size, features)` (channels_last)

**Usage:**
```python
# Text classification
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=200),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### GlobalMaxPooling2D

Global max pooling for 2D data.

```python
tf.keras.layers.GlobalMaxPooling2D(
    data_format=None,
    keepdims=False
)
```

**Input shape:** `(batch_size, height, width, channels)` (channels_last)

**Output shape:** `(batch_size, channels)` (channels_last)

### GlobalMaxPooling3D

Global max pooling for 3D data.

```python
tf.keras.layers.GlobalMaxPooling3D(
    data_format=None,
    keepdims=False
)
```

### GlobalAveragePooling1D

Global average pooling for 1D temporal data.

```python
tf.keras.layers.GlobalAveragePooling1D(
    data_format='channels_last',
    keepdims=False
)
```

### GlobalAveragePooling2D

Global average pooling for 2D data. Often used to replace Flatten + Dense in CNNs.

```python
tf.keras.layers.GlobalAveragePooling2D(
    data_format=None,
    keepdims=False
)
```

**Usage:**
```python
# Common pattern: replace Flatten + Dense with GlobalAveragePooling2D
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),  # Output: (batch, 64)
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### GlobalAveragePooling3D

Global average pooling for 3D data.

```python
tf.keras.layers.GlobalAveragePooling3D(
    data_format=None,
    keepdims=False
)
```

---

## RNN Layers

### RNN (Base Class)

Base class for recurrent layers.

```python
tf.keras.layers.RNN(
    cell,                            # An RNN cell instance or list of cell instances
    return_sequences=False,          # Boolean. Whether to return the last output in the
                                     # output sequence, or the full sequence
    return_state=False,              # Boolean. Whether to return the last state in addition to output
    go_backwards=False,              # Boolean. If True, process the input sequence backwards
    stateful=False,                  # Boolean. If True, the last state for each sample at
                                     # index i will be used as initial state for sample i in the next batch
    unroll=False,                    # Boolean. If True, the network will be unrolled
    time_major=False,                # If True, input shape is (batch, time, ...) becomes (time, batch, ...)
    zero_output_for_mask=False,      # Boolean. If True, output for masked timestep is zero
    implementation=2                 # 1 or 2. 2 is generally faster but may not work on all devices
)
```

**Usage:**
```python
# Using a custom cell
cell = tf.keras.layers.LSTMCell(64)
rnn = tf.keras.layers.RNN(cell, return_sequences=True)

# Stacked RNN cells
cells = [tf.keras.layers.LSTMCell(128), tf.keras.layers.LSTMCell(64)]
rnn = tf.keras.layers.RNN(cells, return_sequences=True)
```

### SimpleRNN

Fully-connected RNN where the output is fed back to input.

```python
tf.keras.layers.SimpleRNN(
    units,                           # Positive integer, dimensionality of the output space
    activation='tanh',               # Activation function
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,                     # Float [0, 1). Fraction of units to drop for linear transformation of inputs
    recurrent_dropout=0.0,           # Float [0, 1). Fraction of units to drop for linear transformation of recurrent state
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    **kwargs
)
```

**Input shape:** `(batch_size, timesteps, features)`

**Output shape:**
- `return_sequences=True`: `(batch_size, timesteps, units)`
- `return_sequences=False`: `(batch_size, units)`

### LSTM

Long Short-Term Memory layer.

```python
tf.keras.layers.LSTM(
    units,                           # Positive integer
    activation='tanh',               # Activation function to use
    recurrent_activation='sigmoid',  # Activation function for recurrent step
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,           # Boolean. If True, add 1 to the bias of the forget gate at initialization
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    implementation=2,               # 1 or 2. 2 uses larger kernel concatenation
    **kwargs
)
```

**Usage:**
```python
# Basic LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(100, 32)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Stacked LSTM with return_sequences
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(100, 32)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Return states for encoder
lstm = tf.keras.layers.LSTM(128, return_state=True)
output, state_h, state_c = lstm(inputs)

# Bidirectional LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=(100, 32)
    ),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Stateful LSTM
lstm = tf.keras.layers.LSTM(64, stateful=True, batch_size=32, input_shape=(10, 32))
```

### GRU

Gated Recurrent Unit layer.

```python
tf.keras.layers.GRU(
    units,                           # Positive integer
    activation='tanh',               # Activation function
    recurrent_activation='sigmoid',  # Activation for recurrent step
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    reset_after=True,                # Boolean. If True, apply reset gate after matrix multiplication
    use_cudnn='auto',                # 'auto', 'never', or 'always'. Whether to use cuDNN implementation
    **kwargs
)
```

**Usage:**
```python
# Basic GRU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=100),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# With cuDNN optimization
gru_cudnn = tf.keras.layers.GRU(128, use_cudnn='always')

# Return states
gru = tf.keras.layers.GRU(64, return_state=True)
output, state = gru(inputs)
```

### Bidirectional

Bidirectional wrapper for RNNs.

```python
tf.keras.layers.Bidirectional(
    layer,                           # RNN layer instance (e.g. LSTM, GRU, SimpleRNN)
    merge_mode='concat',             # Mode by which outputs of the forward and backward RNNs
                                     # will be combined: 'sum', 'mul', 'concat', 'ave', None
    backward_layer=None,             # Optional. Custom backward RNN layer
    **kwargs
)
```

**Usage:**
```python
# Basic bidirectional
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        input_shape=(100, 32)
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# With different merge modes
bidi_sum = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), merge_mode='sum')
bidi_concat = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), merge_mode='concat')
# concat output: 64 * 2 = 128

# With custom backward layer
forward = tf.keras.layers.LSTM(64, activation='tanh')
backward = tf.keras.layers.LSTM(64, activation='relu', go_backwards=True)
bidi = tf.keras.layers.Bidirectional(forward, backward_layer=backward)

# Return state
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_state=True))
output, fwd_h, fwd_c, bwd_h, bwd_c = lstm(inputs)
```

### ConvLSTM2D

Convolutional LSTM layer. Like LSTM but with convolutional operations instead of dense.

```python
tf.keras.layers.ConvLSTM2D(
    filters,                         # Integer, number of output filters
    kernel_size,                     # Integer or tuple of 2 integers
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    activation='tanh',
    recurrent_activation='hard_sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    return_sequences=False,
    go_backwards=False,
    stateful=False,
    dropout=0.0,
    recurrent_dropout=0.0,
    **kwargs
)
```

**Input shape:**
- `(batch, time, height, width, channels)` (channels_last)
- `(batch, time, channels, height, width)` (channels_first)

**Usage:**
```python
# Video prediction model
model = tf.keras.Sequential([
    tf.keras.layers.ConvLSTM2D(
        64, (3, 3), padding='same', return_sequences=True,
        input_shape=(10, 64, 64, 1)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True),
    tf.keras.layers.Conv2D(1, (3, 3), padding='same')
])
```

---

## Attention Layers

### Attention

Dot-product attention layer.

```python
tf.keras.layers.Attention(
    use_scale=True,                  # Boolean. If True, creates a scalar to scale attention scores
    score_mode='dot',                # 'dot' or 'concat'. Function to use to compute attention scores
    dropout=0.0,                     # Float between 0 and 1. Dropout rate for attention weights
    seed=None,                       # Integer for deterministic dropout
    **kwargs
)
```

**Call signature:** `layer(query, value, key=None, attention_mask=None, return_attention_scores=False, training=None)`

**Usage:**
```python
# Basic attention
attention = tf.keras.layers.Attention()
query = tf.keras.Input(shape=(10, 64))
value = tf.keras.Input(shape=(20, 64))
output = attention([query, value])

# With key
key = tf.keras.Input(shape=(20, 64))
output = attention([query, value, key])

# Return attention scores
output, scores = attention([query, value], return_attention_scores=True)
```

### AdditiveAttention

Additive attention layer (Bahdanau-style).

```python
tf.keras.layers.AdditiveAttention(
    use_scale=True,                  # Boolean
    dropout=0.0,                     # Float
    seed=None,
    **kwargs
)
```

**Usage:**
```python
# Bahdanau attention
query = tf.keras.Input(shape=(10, 64))
value = tf.keras.Input(shape=(20, 128))
attention = tf.keras.layers.AdditiveAttention()
output = attention([query, value])
```

### MultiHeadAttention

Multi-head attention mechanism. Implements scaled dot-product attention with multiple heads.

```python
tf.keras.layers.MultiHeadAttention(
    num_heads,                       # Integer, number of attention heads
    key_dim,                         # Integer, size of each attention head for query and key
    value_dim=None,                  # Integer, size of each attention head for value. Defaults to key_dim
    dropout=0.0,                     # Dropout rate
    use_bias=True,                   # Boolean, whether the dense layers use bias vectors
    output_shape=None,               # Integer, expected last dimension of the output shape
    attention_axes=None,             # Axes over which attention is applied. None means all axes except batch
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

**Call signature:**
```python
layer(
    query,                           # Query tensor of shape (batch, seq_len, key_dim)
    value,                           # Value tensor of shape (batch, seq_len, value_dim)
    key=None,                        # Key tensor. If None, uses value
    attention_mask=None,             # Mask of shape (batch, num_heads, query_seq_len, value_seq_len)
    return_attention_scores=False,   # Boolean
    training=None,                   # Boolean
    use_causal_mask=False            # Boolean, whether to apply causal mask
)
```

**Usage:**
```python
# Transformer-style self-attention
mha = tf.keras.layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    dropout=0.1
)

# Self-attention
query = tf.keras.Input(shape=(seq_len, 512))
output = mha(query, query)  # Self-attention: query=value=key

# Cross-attention
encoder_output = tf.keras.Input(shape=(encoder_seq_len, 512))
decoder_output = tf.keras.Input(shape=(decoder_seq_len, 512))
cross_output = mha(decoder_output, encoder_output)  # query=decoder, value=encoder

# With causal mask for autoregressive generation
output = mha(query, query, use_causal_mask=True)

# With custom attention mask
mask = tf.sequence_mask([3, 2, 5], maxlen=5)  # (batch, seq_len)
output = mha(query, query, attention_mask=mask)

# Return attention scores for visualization
output, scores = mha(query, query, return_attention_scores=True)

# Full Transformer encoder block
def transformer_encoder_block(inputs, num_heads, key_dim, ff_dim, dropout_rate):
    # Self-attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(inputs, inputs)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward
    ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2
```

---

## Normalization Layers

### BatchNormalization

Batch normalization layer. Normalizes inputs to zero mean and unit variance per batch.

```python
tf.keras.layers.BatchNormalization(
    axis=-1,                         # Integer, the axis that should be normalized (typically features axis)
    momentum=0.99,                   # Float, momentum for the moving mean and moving variance
    epsilon=0.001,                   # Small float added to variance to avoid dividing by zero
    center=True,                     # Boolean. If True, add offset of beta to normalized tensor
    scale=True,                      # Boolean. If True, multiply by gamma
    beta_initializer='zeros',        # Initializer for the beta weight
    gamma_initializer='ones',        # Initializer for the gamma weight
    moving_mean_initializer='zeros', # Initializer for the moving mean
    moving_variance_initializer='ones', # Initializer for the moving variance
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    synchronized=False,              # Boolean. If True, synchronizes across all replicas
    **kwargs
)
```

**Usage:**
```python
# Standard usage after Conv2D
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, input_shape=(28, 28, 1), use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# With synchronized batch norm for distributed training
bn = tf.keras.layers.BatchNormalization(synchronized=True)
```

### LayerNormalization

Layer normalization layer. Normalizes across features for each sample.

```python
tf.keras.layers.LayerNormalization(
    axis=-1,                         # Integer or list/tuple of integers
    epsilon=0.001,                   # Small float
    center=True,                     # Boolean. Add offset beta
    scale=True,                      # Boolean. Multiply by gamma
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)
```

**Usage:**
```python
# Transformer block
layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

# Pre-norm Transformer
def transformer_block(x, num_heads, key_dim):
    # Pre-norm
    x_norm = tf.keras.layers.LayerNormalization()(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)(x_norm, x_norm)
    x = x + attn
    x_norm = tf.keras.layers.LayerNormalization()(x)
    ff = tf.keras.layers.Dense(2048, activation='relu')(x_norm)
    ff = tf.keras.layers.Dense(x.shape[-1])(ff)
    x = x + ff
    return x
```

### UnitNormalization

Normalizes along the specified axis to unit norm.

```python
tf.keras.layers.UnitNormalization(
    axis=-1,                         # Integer or list/tuple of integers, axis along which to normalize
    **kwargs
)
```

**Usage:**
```python
# Normalize embeddings to unit sphere
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.UnitNormalization(axis=-1)
])
```

### GroupNormalization

Group normalization layer. Divides channels into groups and normalizes within each group.

```python
tf.keras.layers.GroupNormalization(
    groups=32,                       # Integer, number of groups for Group Normalization
    axis=-1,                         # Integer
    epsilon=0.001,                   # Small float
    center=True,                     # Boolean
    scale=True,                      # Boolean
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)
```

**Usage:**
```python
# Group normalization (works well with small batch sizes)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, input_shape=(224, 224, 3), use_bias=False),
    tf.keras.layers.GroupNormalization(groups=8),
    tf.keras.layers.ReLU()
])
```

---

## Regularization Layers

### Dropout

(See Core Layers above)

### SpatialDropout1D / SpatialDropout2D / SpatialDropout3D

(See Core Layers above for SpatialDropout1D/2D/3D)

### GaussianDropout

Applies multiplicative 1-centered Gaussian noise. Only active during training.

```python
tf.keras.layers.GaussianDropout(
    rate,                  # Float, drop probability (and noise standard deviation)
    **kwargs
)
```

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.GaussianDropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### GaussianNoise

Apply additive zero-centered Gaussian noise. Only active during training.

```python
tf.keras.layers.GaussianNoise(
    stddev,                # Float, standard deviation of the noise distribution
    **kwargs
)
```

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.GaussianNoise(0.1, input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### AlphaDropout

Applies Alpha Dropout to the input, maintaining the self-normalizing property of SELU activations.

```python
tf.keras.layers.AlphaDropout(
    rate,                  # Float between 0 and 1
    noise_shape=None,      # Optional
    seed=None,             # Optional
    **kwargs
)
```

**Usage:**
```python
# Self-normalizing network with SELU + AlphaDropout
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='selu', input_shape=(784,)),
    tf.keras.layers.AlphaDropout(0.1),
    tf.keras.layers.Dense(64, activation='selu'),
    tf.keras.layers.AlphaDropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## Activation Layers

### ReLU

Rectified Linear Unit activation function layer.

```python
tf.keras.layers.ReLU(
    max_value=None,       # Float >= 0. Maximum activation value
    negative_slope=0.0,   # Float >= 0. Slope for negative input values
    threshold=0.0,        # Float. Threshold for activation
    **kwargs
)
```

**Formula:** `f(x) = max_value if x >= max_value else (negative_slope * (x - threshold) if x >= threshold else 0)`

**Usage:**
```python
# Standard ReLU
relu = tf.keras.layers.ReLU()

# Leaky ReLU behavior
leaky = tf.keras.layers.ReLU(negative_slope=0.1)

# Capped ReLU
capped = tf.keras.layers.ReLU(max_value=6.0)  # ReLU6
```

### LeakyReLU

Leaky version of a Rectified Linear Unit.

```python
tf.keras.layers.LeakyReLU(
    alpha=0.3,            # Float >= 0. Negative slope coefficient
    **kwargs
)
```

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### PReLU

Parametric Rectified Linear Unit. The slope is learned during training.

```python
tf.keras.layers.PReLU(
    alpha_initializer='zeros',    # Initializer for the learnable alpha
    alpha_regularizer=None,       # Regularizer for alpha
    alpha_constraint=None,        # Constraint for alpha
    shared_axes=None,             # Axes along which to share alpha values
    **kwargs
)
```

**Usage:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Share alpha across spatial dimensions
prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
```

### ELU

Exponential Linear Unit.

```python
tf.keras.layers.ELU(
    alpha=1.0,            # Float. Scale for negative factor
    **kwargs
)
```

**Formula:** `f(x) = x if x > 0 else alpha * (exp(x) - 1)`

### ThresholdedReLU

Thresholded Rectified Linear Unit.

```python
tf.keras.layers.ThresholdedReLU(
    theta=1.0,            # Float >= 0. Threshold for activation
    **kwargs
)
```

**Formula:** `f(x) = x if x > theta else 0`

### Softmax

Softmax activation function.

```python
tf.keras.layers.Softmax(
    axis=-1,              # Integer or list of integers, axis along which softmax is applied
    **kwargs
)
```

**Usage:**
```python
# Multi-class classification output
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])
```

---

## Reshaping Layers

### Reshape

(See Core Layers above)

### Flatten

(See Core Layers above)

### Permute

(See Core Layers above)

### RepeatVector

(See Core Layers above)

### UpSampling1D

Upsampling layer for 1D inputs.

```python
tf.keras.layers.UpSampling1D(
    size=2                # Integer. Upsampling factor
)
```

### UpSampling2D

Upsampling layer for 2D inputs. Repeats rows and columns.

```python
tf.keras.layers.UpSampling2D(
    size=(2, 2),          # Int or tuple of 2 ints. Upsampling factors for rows and columns
    data_format=None,     # 'channels_last' or 'channels_first'
    interpolation='nearest'  # 'nearest' or 'bilinear'
)
```

**Usage:**
```python
# U-Net decoder
up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
```

### UpSampling3D

Upsampling layer for 3D inputs.

```python
tf.keras.layers.UpSampling3D(
    size=(2, 2, 2),       # Int or tuple of 3 ints
    data_format=None,
    **kwargs
)
```

### ZeroPadding1D

Zero-padding layer for 1D input.

```python
tf.keras.layers.ZeroPadding1D(
    padding=1             # Int, or tuple of 2 ints, or dict {'left': int, 'right': int}
)
```

### ZeroPadding2D

Zero-padding layer for 2D input.

```python
tf.keras.layers.ZeroPadding2D(
    padding=(1, 1),       # Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints
    data_format=None
)
```

### ZeroPadding3D

Zero-padding layer for 3D input.

```python
tf.keras.layers.ZeroPadding3D(
    padding=(1, 1, 1),    # Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints
    data_format=None
)
```

### Cropping1D

Cropping layer for 1D input.

```python
tf.keras.layers.Cropping1D(
    cropping=(1, 1)       # Int or tuple of 2 ints. (crop_left, crop_right)
)
```

### Cropping2D

Cropping layer for 2D input.

```python
tf.keras.layers.Cropping2D(
    cropping=((0, 0), (0, 0)),  # Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints
    data_format=None
)
```

### Cropping3D

Cropping layer for 3D input.

```python
tf.keras.layers.Cropping3D(
    cropping=((1, 1), (1, 1), (1, 1)),  # Tuple of 3 tuples of 2 ints
    data_format=None
)
```

---

## Merge Layers

### Add

Layer that adds a list of inputs. All inputs must have the same shape.

```python
tf.keras.layers.Add(**kwargs)
```

**Usage:**
```python
# Residual connection
input_tensor = tf.keras.Input(shape=(28, 28, 64))
x = tf.keras.layers.Conv2D(64, 3, padding='same')(input_tensor)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
output = tf.keras.layers.Add()([input_tensor, x])  # Residual add
```

### Subtract

Layer that subtracts two inputs.

```python
tf.keras.layers.Subtract(**kwargs)
```

### Multiply

Layer that multiplies (element-wise) a list of inputs.

```python
tf.keras.layers.Multiply(**kwargs)
```

### Average

Layer that averages a list of inputs element-wise.

```python
tf.keras.layers.Average(**kwargs)
```

### Maximum

Layer that computes the maximum (element-wise) of a list of inputs.

```python
tf.keras.layers.Maximum(**kwargs)
```

### Minimum

Layer that computes the minimum (element-wise) of a list of inputs.

```python
tf.keras.layers.Minimum(**kwargs)
```

### Concatenate

Layer that concatenates a list of inputs.

```python
tf.keras.layers.Concatenate(
    axis=-1,              # Integer, axis along which to concatenate
    **kwargs
)
```

**Usage:**
```python
# Concatenate features
text_input = tf.keras.Input(shape=(100,))
meta_input = tf.keras.Input(shape=(10,))
merged = tf.keras.layers.Concatenate()([text_input, meta_input])
output = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

# Concatenate along channels
input1 = tf.keras.Input(shape=(28, 28, 32))
input2 = tf.keras.Input(shape=(28, 28, 32))
merged = tf.keras.layers.Concatenate(axis=-1)([input1, input2])  # Shape: (28, 28, 64)
```

### Dot

Layer that computes a dot product between samples in two tensors.

```python
tf.keras.layers.Dot(
    axes,                 # Integer or tuple of integers
    normalize=False,      # Whether to L2-normalize along the dot product axis
    **kwargs
)
```

---

## Advanced Layers

### LocallyConnected1D

Locally-connected layer for 1D inputs. Like Conv1D but with unshared weights.

```python
tf.keras.layers.LocallyConnected1D(
    filters,                          # Integer
    kernel_size,                      # Integer
    strides=1,
    padding='valid',                  # Only 'valid' is supported
    data_format=None,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    implementation=1,                 # 1, 2, or 3
    **kwargs
)
```

### LocallyConnected2D

Locally-connected layer for 2D inputs.

```python
tf.keras.layers.LocallyConnected2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    implementation=1,
    **kwargs
)
```

### Embedding

Turns positive integers (indexes) into dense vectors of fixed size.

```python
tf.keras.layers.Embedding(
    input_dim,                        # Integer >= 1. Size of the vocabulary
    output_dim,                       # Integer >= 1. Dimension of the dense embedding
    embeddings_initializer='uniform', # Initializer for the embeddings matrix
    embeddings_regularizer=None,      # Regularizer for embeddings matrix
    activity_regularizer=None,
    embeddings_constraint=None,       # Constraint for embeddings matrix
    mask_zero=False,                  # Boolean. Whether or not the input value 0 is a special "padding" value
    input_length=None,                # Length of input sequences (for static shape)
    **kwargs
)
```

**Input shape:** `(batch_size, input_length)` - 2D tensor of integers

**Output shape:** `(batch_size, input_length, output_dim)` - 3D tensor

**Usage:**
```python
# Basic embedding
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 128, input_length=100),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# With masking
embedding = tf.keras.layers.Embedding(
    input_dim=5000, output_dim=64, mask_zero=True
)

# With pre-trained embeddings
import numpy as np
embedding_matrix = np.random.random((10000, 300))
embedding = tf.keras.layers.Embedding(
    10000, 300,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False
)
```

### CategoryEncoding

Category encoding layer that encodes categorical features.

```python
tf.keras.layers.CategoryEncoding(
    num_tokens=None,                  # Integer or dict mapping tokens to indices. Total number of tokens
    output_mode='multi_hot',          # 'multi_hot', 'one_hot', 'count', 'tf_idf'
    sparse=False,                     # Boolean. Whether to return a sparse tensor
    **kwargs
)
```

### HashedCrossing

A preprocessing layer that crosses features using the "hashing trick".

```python
tf.keras.layers.HashedCrossing(
    num_bins,                         # Integer, number of hash bins
    output_mode='int',                # 'int', 'one_hot', or 'multi_hot'
    sparse=False,
    **kwargs
)
```

### IntegerLookup

A preprocessing layer that maps integer features to contiguous ranges.

```python
tf.keras.layers.IntegerLookup(
    max_tokens=None,                  # Maximum size of the vocabulary
    num_oov_indices=1,                # Number of out-of-vocabulary indices
    mask_token=None,                  # Integer token that represents masking
    oov_token=-1,                     # Integer token for OOV values
    vocabulary=None,                  # Array of integers defining the vocabulary
    invert=False,                     # If True, maps indices back to tokens
    **kwargs
)
```

### StringLookup

A preprocessing layer that maps string features to integer indices.

```python
tf.keras.layers.StringLookup(
    max_tokens=None,
    num_oov_indices=1,
    mask_token='',
    vocabulary=None,
    encoding=None,                    # String encoding for input strings
    invert=False,
    output_mode='int',                # 'int', 'one_hot', 'multi_hot', 'count', 'tf_idf'
    sparse=False,
    **kwargs
)
```

### TextVectorization

A preprocessing layer that maps text features to integer sequences.

```python
tf.keras.layers.TextVectorization(
    max_tokens=None,                  # Maximum size of the vocabulary
    standardize='lower_and_strip_punctuation',  # 'lower_and_strip_punctuation', 'lower', 'strip_punctuation',
                                                 # or callable
    split='whitespace',               # 'whitespace', 'character', or callable
    ngrams=None,                       # Integer or tuple of integers for n-gram creation
    output_mode='int',                 # 'int', 'multi_hot', 'count', 'tf_idf'
    output_sequence_length=None,       # Integer. Only valid in 'int' mode
    pad_to_max_tokens=False,           # Boolean. Only valid for 'multi_hot', 'count', 'tf_idf'
    vocabulary=None,                   # Array or list of strings defining the vocabulary
    idf_weights=None,                  # Tuple of floats for TF-IDF weighting
    sparse=False,
    ragged=False,
    **kwargs
)
```

**Usage:**
```python
# Basic text vectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=200
)
vectorizer.adapt(train_texts)

# In a model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    tf.keras.layers.Embedding(10000, 128),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# TF-IDF mode
tfidf_layer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode='tf_idf'
)
tfidf_layer.adapt(train_texts)
```

### Normalization (Preprocessing)

A preprocessing layer that normalizes continuous features.

```python
tf.keras.layers.Normalization(
    axis=-1,                          # Integer or tuple of integers
    mean=None,                        # Mean to use for normalization
    variance=None,                    # Variance to use for normalization
    invert=False,                     # If True, applies inverse transformation
    **kwargs
)
```

**Usage:**
```python
# Learn mean/variance from data
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(train_data)

# Use in model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    norm_layer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# With explicit mean/variance
norm = tf.keras.layers.Normalization(mean=5.0, variance=4.0)
```

### Discretization

A preprocessing layer that buckets continuous features into discrete ranges.

```python
tf.keras.layers.Discretization(
    bin_boundaries=None,              # List of boundary values for bins
    num_bins=None,                    # Integer, number of bins to compute
    epsilon=0.01,                     # Small error tolerance for quantile computation
    output_mode='int',                # 'int', 'one_hot', 'multi_hot'
    sparse=False,
    **kwargs
)
```

---

## Preprocessing Layers

### Resizing

Image resizing preprocessing layer.

```python
tf.keras.layers.Resizing(
    height,                           # Integer, output height
    width,                            # Integer, output width
    interpolation='bilinear',         # 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5', etc.
    crop_to_aspect_ratio=False,       # Boolean. If True, resize without distortion and crop excess
    pad_to_aspect_ratio=False,        # Boolean. If True, resize without distortion and pad with zeros
    fill_mode='reflect',             # 'reflect', 'wrap', 'constant', 'nearest'
    fill_value=0.0,                   # Float, fill value when pad_to_aspect_ratio is True
    **kwargs
)
```

### Rescaling

Multiplies inputs by a scale factor and adds an offset.

```python
tf.keras.layers.Rescaling(
    scale=1.0,                        # Float, the scale to apply
    offset=0.0,                       # Float, the offset to apply
    **kwargs
)
```

**Usage:**
```python
# Normalize pixel values from [0, 255] to [0, 1]
rescale = tf.keras.layers.Rescaling(1./255)

# Normalize to [-1, 1]
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# In a model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    # ...
])
```

### RandomCrop

Randomly crops images to a target height and width.

```python
tf.keras.layers.RandomCrop(
    height,                           # Integer, output height
    width,                            # Integer, output width
    seed=None,                        # Optional random seed
    **kwargs
)
```

### RandomFlip

Randomly flips images horizontally and/or vertically.

```python
tf.keras.layers.RandomFlip(
    mode='horizontal',                # 'horizontal', 'vertical', 'horizontal_and_vertical'
    seed=None,
    **kwargs
)
```

**Usage:**
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])
```

### RandomRotation

Randomly rotates images during training.

```python
tf.keras.layers.RandomRotation(
    factor,                           # Float or tuple of 2 floats. Rotation range as fraction of 2*pi.
                                      # E.g., 0.1 means random rotation in [-10%*2pi, 10%*2pi]
    fill_mode='reflect',             # 'reflect', 'wrap', 'constant', 'nearest'
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    **kwargs
)
```

### RandomZoom

Randomly zooms images during training.

```python
tf.keras.layers.RandomZoom(
    height_factor,                    # Float or tuple of 2 floats. Zoom range for height
    width_factor=None,                # Float or tuple of 2 floats. Zoom range for width. None = same as height
    fill_mode='reflect',
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    **kwargs
)
```

### RandomContrast

Adjusts the contrast of images by a random factor during training.

```python
tf.keras.layers.RandomContrast(
    factor,                           # Float or tuple of 2 floats. Contrast range
    seed=None,
    **kwargs
)
```

**Usage:**
```python
augment = tf.keras.layers.RandomContrast(factor=0.2)  # Contrast in [1-0.2, 1+0.2]
```

### RandomBrightness

Adjusts the brightness of images by a random factor during training.

```python
tf.keras.layers.RandomBrightness(
    factor,                           # Float or tuple of 2 floats. Brightness adjustment range
    value_range=(0, 255),            # Tuple of two floats, minimum and maximum allowed pixel values
    seed=None,
    **kwargs
)
```

### RandomTranslation

Randomly translates images during training.

```python
tf.keras.layers.RandomTranslation(
    height_factor,                    # Float or tuple of 2 floats. Translation range for height
    width_factor,                     # Float or tuple of 2 floats. Translation range for width
    fill_mode='reflect',
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    **kwargs
)
```

### RandomHeight

Randomly varies the height of images during training.

```python
tf.keras.layers.RandomHeight(
    factor,                           # Float or tuple of 2 floats
    interpolation='bilinear',
    seed=None,
    **kwargs
)
```

### RandomWidth

Randomly varies the width of images during training.

```python
tf.keras.layers.RandomWidth(
    factor,                           # Float or tuple of 2 floats
    interpolation='bilinear',
    seed=None,
    **kwargs
)
```

---

## Common Layer Patterns

### Residual Block

```python
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x
```

### Inception Module

```python
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5, filters_pool):
    path1 = tf.keras.layers.Conv2D(filters_1x1, 1, activation='relu')(x)

    path2 = tf.keras.layers.Conv2D(filters_3x3_reduce, 1, activation='relu')(x)
    path2 = tf.keras.layers.Conv2D(filters_3x3, 3, padding='same', activation='relu')(path2)

    path3 = tf.keras.layers.Conv2D(filters_5x5_reduce, 1, activation='relu')(x)
    path3 = tf.keras.layers.Conv2D(filters_5x5, 5, padding='same', activation='relu')(path3)

    path4 = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')(x)
    path4 = tf.keras.layers.Conv2D(filters_pool, 1, activation='relu')(path4)

    return tf.keras.layers.Concatenate()([path1, path2, path3, path4])
```

### Transformer Encoder Block

```python
def transformer_encoder(x, num_heads=8, key_dim=64, ff_dim=2048, dropout=0.1):
    # Multi-head self-attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(x, x)
    attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
    ff_output = tf.keras.layers.Dense(x.shape[-1])(ff_output)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x
```

### Depthwise Separable Convolution Block

```python
def depthwise_separable_block(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 1)(x)  # Pointwise convolution
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x
```

### Full Augmentation Pipeline

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Use in training model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./255)(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
# ... rest of model
```

---

## Layer Utility Functions

### get_source_inputs

Returns the list of input tensors necessary to compute a tensor.

```python
tf.keras.utils.get_source_inputs(tensor, layer=None, node_index=None)
```

### Layer Serialization

```python
# Serialize a layer's configuration
config = layer.get_config()

# Recreate a layer from its config
new_layer = type(layer).from_config(config)

# Custom serialization
class MyLayer(tf.keras.layers.Layer):
    def get_config(self):
        config = super().get_config()
        config.update({'my_param': self.my_param})
        return config
```

### Custom Layer Template

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

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

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.kernel) + self.bias)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation)
        })
        return config
```

---

## Layer Version Compatibility Notes

- `tf.keras.layers` in TF2.x follows the Keras API specification
- All layers support eager execution by default
- Layers are compatible with `tf.function` for graph mode optimization
- For GPU acceleration, layers like LSTM/GRU support cuDNN implementation when conditions are met (no custom activation, no recurrent_dropout, unroll=False)
- `use_cudnn='auto'` (default for GRU/LSTM) will automatically use cuDNN when possible
- Preprocessing layers with `adapt()` are stateful and should be included in saved models for portability
