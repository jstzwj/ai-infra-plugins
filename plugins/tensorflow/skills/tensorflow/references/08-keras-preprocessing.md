# TensorFlow Keras Preprocessing Reference

This document provides comprehensive reference documentation for all Keras preprocessing layers, text preprocessing, image preprocessing, numerical preprocessing, feature engineering, and preprocessing patterns in TensorFlow.

---

## Table of Contents

1. [Text Preprocessing](#text-preprocessing)
2. [Image Preprocessing](#image-preprocessing)
3. [Numerical Preprocessing](#numerical-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Adaptive Preprocessing](#adaptive-preprocessing)
6. [Preprocessing in tf.data Pipeline](#preprocessing-in-tfdata-pipeline)
7. [Preprocessing in Model](#preprocessing-in-model)
8. [Export with Preprocessing](#export-with-preprocessing)
9. [Text Processing Details](#text-processing-details)
10. [Image Data Augmentation Patterns](#image-data-augmentation-patterns)
11. [Normalization Strategies](#normalization-strategies)
12. [Custom Preprocessing Layers](#custom-preprocessing-layers)

---

## Text Preprocessing

### TextVectorization

The primary layer for transforming raw strings into token sequences, bag-of-words, or TF-IDF representations.

```python
tf.keras.layers.TextVectorization(
    max_tokens=None,                  # Integer. Maximum vocabulary size. Only the most common
                                      # max_tokens-1 tokens will be kept (one reserved for OOV)
    standardize='lower_and_strip_punctuation',  # How to clean text:
                                      #   'lower_and_strip_punctuation' (default)
                                      #   'lower' - lowercase only
                                      #   'strip_punctuation' - remove punctuation only
                                      #   None - no standardization
                                      #   callable - custom function
    split='whitespace',               # How to split text into tokens:
                                      #   'whitespace' (default)
                                      #   'character' - split into characters
                                      #   None - no splitting (assumes pre-split)
                                      #   callable - custom splitting function
    ngrams=None,                       # Integer or tuple of integers. Create n-grams:
                                      #   None - no n-grams
                                      #   2 - unigrams + bigrams
                                      #   (2, 3) - bigrams + trigrams
    output_mode='int',                 # Output format:
                                      #   'int' - integer token indices
                                      #   'multi_hot' - multi-hot encoding
                                      #   'count' - token count encoding
                                      #   'tf_idf' - TF-IDF weighted encoding
    output_sequence_length=None,       # Integer. Only valid for 'int' mode.
                                      # Pads/truncates sequences to this length
    pad_to_max_tokens=False,           # Boolean. Pad to max_tokens. Only valid for
                                      # 'multi_hot', 'count', 'tf_idf' modes
    vocabulary=None,                   # Array/list of strings defining the vocabulary
    idf_weights=None,                  # Tuple/list of floats for TF-IDF weighting
    sparse=False,                      # Boolean. Return sparse tensor
    ragged=False,                      # Boolean. Return ragged tensor (only 'int' mode)
    encoding='utf-8',                  # String encoding for input strings
    **kwargs
)
```

**Input shape:** `(batch_size,)` or `(batch_size, 1)` of strings

**Output shape:**
- `output_mode='int'`: `(batch_size, output_sequence_length)`
- `output_mode='multi_hot'/'count'/'tf_idf'`: `(batch_size, max_tokens)`

**Usage:**
```python
# Basic integer tokenization
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

# Multi-hot encoding
multi_hot_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode='multi_hot'
)
multi_hot_vectorizer.adapt(train_texts)

# TF-IDF encoding
tfidf_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode='tf_idf'
)
tfidf_vectorizer.adapt(train_texts)

# With custom standardization
import re
def custom_standardize(input_text):
    lowercase = tf.strings.lower(input_text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, r'[^\w\s]', '')

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    standardize=custom_standardize,
    output_mode='int',
    output_sequence_length=200
)

# With n-grams
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    ngrams=2,              # Include unigrams and bigrams
    output_mode='tf_idf'
)

# Character-level tokenization
char_vectorizer = tf.keras.layers.TextVectorization(
    split='character',
    output_mode='int',
    output_sequence_length=500
)

# Ragged output (variable-length sequences)
ragged_vectorizer = tf.keras.layers.TextVectorization(
    output_mode='int',
    ragged=True
)
```

### StringLookup

Maps string features to integer indices.

```python
tf.keras.layers.StringLookup(
    max_tokens=None,                  # Integer. Maximum vocabulary size
    num_oov_indices=1,                # Integer >= 0. Number of OOV bucket indices
    mask_token=None,                  # String. Token that represents masked values
    oov_token='[UNK]',                # String. Token for out-of-vocabulary values
    vocabulary=None,                  # List/array of strings defining vocabulary
    encoding=None,                    # String encoding
    invert=False,                     # Boolean. If True, maps indices back to tokens
    output_mode='int',                # 'int', 'one_hot', 'multi_hot', 'count', 'tf_idf'
    sparse=False,                     # Boolean
    pad_to_max_tokens=False,          # Boolean
    **kwargs
)
```

**Usage:**
```python
# Basic string lookup
lookup = tf.keras.layers.StringLookup(vocabulary=['apple', 'banana', 'cherry'])
lookup(['apple', 'banana', 'date'])  # [1, 2, 0] (0 is OOV)

# With multiple OOV indices
lookup = tf.keras.layers.StringLookup(
    num_oov_indices=2,
    vocabulary=['apple', 'banana', 'cherry']
)

# Adapt from data
lookup = tf.keras.layers.StringLookup(max_tokens=1000)
lookup.adapt(train_string_features)

# Inverse lookup (index to string)
inverse_lookup = tf.keras.layers.StringLookup(
    vocabulary=['apple', 'banana', 'cherry'],
    invert=True
)
inverse_lookup([1, 2, 3])  # ['apple', 'banana', 'cherry']

# One-hot output
one_hot_lookup = tf.keras.layers.StringLookup(
    vocabulary=['cat', 'dog', 'bird'],
    output_mode='one_hot'
)
```

### Hashing

Hashes strings or integers to a fixed number of buckets.

```python
tf.keras.layers.Hashing(
    num_bins,                         # Integer. Number of hash bins
    mask_value=None,                  # Value that should map to 0 (masked)
    salt=None,                        # Integer or tuple of 2 integers. Adds randomness to hashing
    output_mode='int',                # 'int', 'one_hot', 'multi_hot'
    sparse=False,                     # Boolean
    **kwargs
)
```

**Usage:**
```python
# Hash strings to bins
hasher = tf.keras.layers.Hashing(num_bins=100)
hasher(['hello', 'world', 'hello'])  # Maps to integer indices in [0, 100)

# With salt for different hashing
hasher = tf.keras.layers.Hashing(num_bins=100, salt=[123, 456])
```

---

## Image Preprocessing

### Resizing

Resizes images to a target height and width.

```python
tf.keras.layers.Resizing(
    height,                           # Integer. Target height
    width,                            # Integer. Target width
    interpolation='bilinear',         # Interpolation method:
                                      #   'bilinear', 'nearest', 'bicubic', 'area',
                                      #   'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic'
    crop_to_aspect_ratio=False,       # Boolean. Crop to maintain aspect ratio
    pad_to_aspect_ratio=False,        # Boolean. Pad to maintain aspect ratio
    fill_mode='reflect',             # Padding fill mode: 'reflect', 'wrap', 'constant', 'nearest'
    fill_value=0.0,                   # Float. Fill value for padding
    **kwargs
)
```

**Usage:**
```python
# Standard resize
resize = tf.keras.layers.Resizing(224, 224)

# Crop to aspect ratio (no distortion)
resize = tf.keras.layers.Resizing(
    224, 224,
    crop_to_aspect_ratio=True
)

# Pad to aspect ratio
resize = tf.keras.layers.Resizing(
    224, 224,
    pad_to_aspect_ratio=True,
    fill_mode='constant',
    fill_value=0.0
)
```

### Rescaling

Scales input values by a constant and optionally adds an offset.

```python
tf.keras.layers.Rescaling(
    scale=1.0,                        # Float. Multiplication factor
    offset=0.0,                       # Float. Addition offset
    **kwargs
)
```

**Usage:**
```python
# Normalize [0, 255] to [0, 1]
rescale = tf.keras.layers.Rescaling(1./255)

# Normalize [0, 255] to [-1, 1]
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# Normalize ImageNet-style
rescale = tf.keras.layers.Rescaling(1./255)
# Then subtract ImageNet mean per channel
```

### CenterCrop

Crops the center portion of images to a target size.

```python
tf.keras.layers.CenterCrop(
    height,                           # Integer. Target height
    width,                            # Integer. Target width
    **kwargs
)
```

**Usage:**
```python
# Center crop 224x224 from larger images
crop = tf.keras.layers.CenterCrop(224, 224)
```

### RandomCrop

Randomly crops images during training.

```python
tf.keras.layers.RandomCrop(
    height,                           # Integer. Target height
    width,                            # Integer. Target width
    seed=None,                        # Optional random seed
    **kwargs
)
```

**Usage:**
```python
# Random crop augmentation
crop = tf.keras.layers.RandomCrop(224, 224)
# Input must be >= 224x224
```

### RandomFlip

Randomly flips images horizontally and/or vertically during training.

```python
tf.keras.layers.RandomFlip(
    mode='horizontal',                # Flip mode:
                                      #   'horizontal' - flip left-right
                                      #   'vertical' - flip top-bottom
                                      #   'horizontal_and_vertical' - both
    seed=None,
    **kwargs
)
```

**Usage:**
```python
flip = tf.keras.layers.RandomFlip('horizontal')
# Only applies during training; during inference, passes input through unchanged
```

### RandomRotation

Randomly rotates images during training.

```python
tf.keras.layers.RandomRotation(
    factor,                           # Float or tuple of 2 floats.
                                      # Rotation range as fraction of 2*pi.
                                      # 0.1 means random rotation in [-36deg, 36deg]
    fill_mode='reflect',             # Fill mode for newly created pixels:
                                      #   'reflect', 'wrap', 'constant', 'nearest'
    interpolation='bilinear',         # Interpolation method
    seed=None,
    fill_value=0.0,                   # Float. Fill value for 'constant' fill mode
    **kwargs
)
```

**Usage:**
```python
# Rotate up to 20%
rotation = tf.keras.layers.RandomRotation(0.2)

# Rotation with specific range
rotation = tf.keras.layers.RandomRotation((-0.1, 0.1))
```

### RandomZoom

Randomly zooms images during training.

```python
tf.keras.layers.RandomZoom(
    height_factor,                    # Float or tuple. Zoom range for height.
                                      # Positive = zoom out, negative = zoom in
    width_factor=None,                # Float or tuple. Zoom range for width. None = same as height
    fill_mode='reflect',
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    **kwargs
)
```

**Usage:**
```python
# Zoom in by up to 20%
zoom = tf.keras.layers.RandomZoom((-0.2, 0.0))

# Zoom in or out by up to 20%
zoom = tf.keras.layers.RandomZoom((-0.2, 0.2))

# Different zoom for height and width
zoom = tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.1)
```

### RandomContrast

Adjusts image contrast randomly during training.

```python
tf.keras.layers.RandomContrast(
    factor,                           # Float or tuple of 2 floats.
                                      # Contrast adjustment range.
                                      # Output = (1-factor)*input + factor*mean
    seed=None,
    **kwargs
)
```

**Usage:**
```python
# Contrast variation by +/- 30%
contrast = tf.keras.layers.RandomContrast(factor=0.3)
```

### RandomBrightness

Adjusts image brightness randomly during training.

```python
tf.keras.layers.RandomBrightness(
    factor,                           # Float or tuple of 2 floats.
                                      # Brightness adjustment range [-factor, factor]
    value_range=(0, 255),            # Tuple. Allowed output pixel value range
    seed=None,
    **kwargs
)
```

**Usage:**
```python
# Brightness variation for [0, 1] range images
brightness = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1))

# Brightness for uint8 images
brightness = tf.keras.layers.RandomBrightness(factor=0.3, value_range=(0, 255))
```

### RandomTranslation

Randomly translates images during training.

```python
tf.keras.layers.RandomTranslation(
    height_factor,                    # Float or tuple. Translation range for height
    width_factor,                     # Float or tuple. Translation range for width
    fill_mode='reflect',
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    **kwargs
)
```

**Usage:**
```python
# Translate by up to 10% in each direction
translation = tf.keras.layers.RandomTranslation(0.1, 0.1)
```

### RandomHeight

Randomly varies the height of images during training.

```python
tf.keras.layers.RandomHeight(
    factor,                           # Float or tuple. Height variation range
    interpolation='bilinear',
    seed=None,
    **kwargs
)
```

### RandomWidth

Randomly varies the width of images during training.

```python
tf.keras.layers.RandomWidth(
    factor,                           # Float or tuple. Width variation range
    interpolation='bilinear',
    seed=None,
    **kwargs
)
```

---

## Numerical Preprocessing

### Normalization

Normalizes continuous features by subtracting the mean and dividing by the standard deviation.

```python
tf.keras.layers.Normalization(
    axis=-1,                          # Integer or tuple of integers. Axis to normalize
    mean=None,                        # Float or tensor. Mean to use (if not using adapt)
    variance=None,                    # Float or tensor. Variance to use (if not using adapt)
    invert=False,                     # Boolean. If True, applies inverse transformation
    **kwargs
)
```

**Usage:**
```python
# Learn statistics from data
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(train_features)

# Apply normalization
normalized_data = norm(raw_features)

# With explicit mean and variance
norm = tf.keras.layers.Normalization(mean=5.0, variance=4.0)

# Inverse normalization
inv_norm = tf.keras.layers.Normalization(mean=5.0, variance=4.0, invert=True)

# Per-feature normalization
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(train_data)
# Learns separate mean/variance for each feature dimension

# In a model
inputs = tf.keras.Input(shape=(10,))
normalized = tf.keras.layers.Normalization()(inputs)
hidden = tf.keras.layers.Dense(64, activation='relu')(normalized)
outputs = tf.keras.layers.Dense(1)(hidden)
model = tf.keras.Model(inputs, outputs)

# Adapt before building
norm = tf.keras.layers.Normalization()
norm.adapt(x_train)
model = tf.keras.Sequential([
    norm,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

### Discretization

Maps continuous features to discrete bins.

```python
tf.keras.layers.Discretization(
    bin_boundaries=None,              # List of floats. Edges of bins.
                                      # N boundaries create N+1 bins
    num_bins=None,                    # Integer. Compute bin boundaries via quantile
    epsilon=0.01,                     # Float. Error tolerance for quantile computation
    output_mode='int',                # 'int', 'one_hot', 'multi_hot'
    sparse=False,                     # Boolean
    **kwargs
)
```

**Usage:**
```python
# With explicit bin boundaries
discretizer = tf.keras.layers.Discretization(
    bin_boundaries=[0.0, 10.0, 50.0, 100.0]
)
# Values: [-inf, 0) -> 0, [0, 10) -> 1, [10, 50) -> 2, [50, 100) -> 3, [100, inf) -> 4

# Learn bins from data using quantiles
discretizer = tf.keras.layers.Discretization(
    num_bins=5,
    output_mode='one_hot'
)
discretizer.adapt(train_data)

# Integer output
disc = tf.keras.layers.Discretization(bin_boundaries=[0.5, 1.5, 2.5])
disc([0.1, 0.7, 1.2, 2.0, 3.0])  # [0, 1, 1, 2, 3]
```

### CategoryEncoding

Encodes integer categorical features into various representations.

```python
tf.keras.layers.CategoryEncoding(
    num_tokens=None,                  # Integer. Total number of tokens/categories
    output_mode='multi_hot',          # 'multi_hot', 'one_hot', 'count', 'tf_idf'
    sparse=False,                     # Boolean
    **kwargs
)
```

**Usage:**
```python
# One-hot encoding for single categories
encoder = tf.keras.layers.CategoryEncoding(num_tokens=10, output_mode='one_hot')
encoder([3, 5, 7])  # Shape: (3, 10)

# Multi-hot encoding for multi-label
encoder = tf.keras.layers.CategoryEncoding(num_tokens=10, output_mode='multi_hot')
encoder([[1, 3, 5], [2, 4]])  # Shape: (2, 10)

# Count encoding
encoder = tf.keras.layers.CategoryEncoding(num_tokens=10, output_mode='count')
```

### IntegerLookup

Maps integer features to contiguous ranges.

```python
tf.keras.layers.IntegerLookup(
    max_tokens=None,                  # Integer. Maximum vocabulary size
    num_oov_indices=1,                # Integer >= 0
    mask_token=None,                  # Integer. Token representing masked values
    oov_token=-1,                     # Integer. OOV input token
    vocabulary=None,                  # List/array of integers
    invert=False,                     # Boolean. Index to token mapping
    output_mode='int',                # 'int', 'one_hot', 'multi_hot', 'count', 'tf_idf'
    sparse=False,
    pad_to_max_tokens=False,
    **kwargs
)
```

**Usage:**
```python
# Basic integer lookup
lookup = tf.keras.layers.IntegerLookup(vocabulary=[10, 20, 30, 40])
lookup([10, 20, 50])  # [1, 2, 0] (0 = OOV)

# Adapt from data
lookup = tf.keras.layers.IntegerLookup(max_tokens=100)
lookup.adapt(train_integers)

# Inverse mapping
inv_lookup = tf.keras.layers.IntegerLookup(
    vocabulary=[10, 20, 30],
    invert=True
)
inv_lookup([1, 2, 3])  # [10, 20, 30]
```

### HashedCrossing

Crosses features using the "hashing trick" for feature interactions.

```python
tf.keras.layers.HashedCrossing(
    num_bins,                         # Integer. Number of hash bins
    output_mode='int',                # 'int', 'one_hot', 'multi_hot'
    sparse=False,                     # Boolean
    **kwargs
)
```

**Usage:**
```python
# Cross two features
cross = tf.keras.layers.HashedCrossing(num_bins=100)

# Example: crossing age_group and income_bracket
age = tf.keras.layers.Discretization(bin_boundaries=[18, 35, 55])
income = tf.keras.layers.Discretization(bin_boundaries=[30000, 60000, 100000])

crossed = cross([age(age_data), income(income_data)])
```

---

## Feature Engineering

### Keras Preprocessing Layers Approach (Recommended for TF2)

Instead of using TF1 `tf.feature_column`, use Keras preprocessing layers for a more integrated approach.

```python
# Complete feature engineering pipeline
def build_model(num_features, cat_features, text_features):
    inputs = {}

    # Numerical inputs
    num_inputs = {}
    for name in num_features:
        inputs[name] = tf.keras.Input(shape=(1,), name=name)

    # Categorical inputs
    cat_inputs = {}
    for name, vocab in cat_features:
        inputs[name] = tf.keras.Input(shape=(1,), dtype=tf.string, name=name)

    # Text inputs
    text_inputs = {}
    for name in text_features:
        inputs[name] = tf.keras.Input(shape=(1,), dtype=tf.string, name=name)

    # Process numerical features
    num_values = [inputs[name] for name in num_features]
    num_concat = tf.keras.layers.Concatenate()(num_values)
    num_normalized = tf.keras.layers.Normalization()(num_concat)

    # Process categorical features
    cat_encoded = []
    for name, vocab in cat_features:
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
        cat_encoded.append(lookup(inputs[name]))

    # Process text features
    text_encoded = []
    for name in text_features:
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=5000, output_mode='tf_idf'
        )
        text_encoded.append(vectorizer(inputs[name]))

    # Concatenate all features
    all_features = [num_normalized] + cat_encoded + text_encoded
    x = tf.keras.layers.Concatenate()(all_features)

    # Dense layers
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, output)
```

### Feature Crossing Pattern

```python
# Feature crossing with hashing
def create_feature_cross(feature_a, feature_b, num_bins=1000):
    cross = tf.keras.layers.HashedCrossing(num_bins=num_bins)
    embedding = tf.keras.layers.Embedding(num_bins, 8)

    crossed = cross([feature_a, feature_b])
    return embedding(crossed)
```

### Bucketized Continuous Features

```python
# Bucketize and embed continuous features
def bucketize_and_embed(continuous_input, num_bins=10, embed_dim=8):
    discretize = tf.keras.layers.Discretization(num_bins=num_bins)
    embed = tf.keras.layers.Embedding(num_bins, embed_dim)

    bucketized = discretize(continuous_input)
    return embed(bucketized)
```

---

## Adaptive Preprocessing

### The adapt() Method

Stateful preprocessing layers need to learn statistics from training data using `adapt()`.

**Layers that support adapt():**
- `TextVectorization` - learns vocabulary and optionally IDF weights
- `StringLookup` - learns vocabulary
- `IntegerLookup` - learns vocabulary
- `Normalization` - learns mean and variance
- `Discretization` - learns bin boundaries (when `num_bins` is set)

```python
# Adapt a TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000)
text_dataset = tf.data.Dataset.from_tensor_slices([
    "This is a positive review",
    "This is a negative review",
    "Another positive example",
    # ... more data
])
vectorizer.adapt(text_dataset)

# Adapt with batched dataset
vectorizer.adapt(text_dataset.batch(256))

# Adapt a Normalization layer
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(train_numeric_features)

# Adapt a Discretization layer
discretizer = tf.keras.layers.Discretization(num_bins=10)
discretizer.adapt(train_continuous_data)

# Get vocabulary after adapt
vocab = vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocab)}")

# Set vocabulary explicitly
vectorizer.set_vocabulary(['word1', 'word2', 'word3'])
```

### Stateful vs Stateless Preprocessing

```python
# Stateful layers (require adapt or explicit parameters)
# These learn from data and maintain state
stateful_layers = [
    tf.keras.layers.TextVectorization,
    tf.keras.layers.Normalization,
    tf.keras.layers.StringLookup,
    tf.keras.layers.IntegerLookup,
    tf.keras.layers.Discretization,
]

# Stateless layers (no adapt needed)
# These apply deterministic transformations
stateless_layers = [
    tf.keras.layers.Rescaling,
    tf.keras.layers.Resizing,
    tf.keras.layers.CenterCrop,
    tf.keras.layers.CategoryEncoding,
    tf.keras.layers.Hashing,
    tf.keras.layers.HashedCrossing,
    # All Random* augmentation layers
]
```

---

## Preprocessing in tf.data Pipeline

Preprocessing can be done in the tf.data pipeline for efficiency, separating data preprocessing from the model.

```python
# Option 1: Preprocessing in tf.data (for CPU-bound preprocessing)
def preprocess_text(text, label):
    # Text preprocessing in the data pipeline
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r'[^\w\s]', '')
    return text, label

def preprocess_image(image, label):
    # Image preprocessing
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Build dataset with preprocessing
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Option 2: Using Keras preprocessing layers in tf.data
# This is useful when you want preprocessing to happen on CPU
# while the model trains on GPU
resize_layer = tf.keras.layers.Resizing(224, 224)
rescale_layer = tf.keras.layers.Rescaling(1./255)

def augment_image(image, label):
    image = resize_layer(image)
    image = rescale_layer(image)
    # Data augmentation only during training
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

train_dataset = raw_train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
```

### Performance Considerations

```python
# Cache preprocessing results when dataset fits in memory
dataset = (
    tf.data.Dataset.from_tensor_slices((data, labels))
    .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()                    # Cache after preprocessing
    .shuffle(buffer_size=10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# For large datasets, cache to file
dataset = (
    tf.data.Dataset.from_tensor_slices((data, labels))
    .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .cache('/tmp/preprocessed_cache')
    .shuffle(buffer_size=10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

---

## Preprocessing in Model

Including preprocessing layers directly in the model simplifies deployment.

```python
# Image classification with built-in preprocessing
inputs = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32)

# Preprocessing layers (part of model graph)
x = tf.keras.layers.Resizing(224, 224)(inputs)
x = tf.keras.layers.Rescaling(1./255)(x)

# Data augmentation (only active during training)
x = tf.keras.layers.RandomFlip('horizontal')(x)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)

# Model backbone
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Text classification with built-in preprocessing
text_input = tf.keras.Input(shape=(1,), dtype=tf.string)

# Preprocessing
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=200
)
vectorizer.adapt(train_texts)

x = vectorizer(text_input)
x = tf.keras.layers.Embedding(10000, 128)(x)
x = tf.keras.layers.LSTM(64)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(text_input, output)
```

---

## Export with Preprocessing

When preprocessing is included in the model, it is automatically saved with the model.

```python
# Save the model with preprocessing included
model.save('model_with_preprocessing.keras')

# Load - preprocessing is preserved
loaded_model = tf.keras.models.load_model('model_with_preprocessing.keras')

# Export for serving
model.save('export_dir', save_format='tf')

# Inference with raw data (preprocessing handled automatically)
predictions = loaded_model.predict(raw_text_strings)

# For TensorFlow Serving
# The SavedModel includes preprocessing, so clients can send raw data
import tensorflow as tf
served_model = tf.saved_model.load('export_dir')
# Can serve raw strings/images directly
```

### Handling Adapted Layers on Save/Load

```python
# When saving, the adapted state is preserved
vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000)
vectorizer.adapt(train_texts)

# Build model
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
# ... rest of model
model = tf.keras.Model(inputs, outputs)

# Save - vocabulary is saved
model.save('text_model.keras')

# Load - vocabulary is restored automatically
loaded = tf.keras.models.load_model('text_model.keras')
# Can immediately process raw strings

# For explicit vocabulary management
vocab = vectorizer.get_vocabulary()
vectorizer_new = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    vocabulary=vocab
)
```

---

## Text Processing Details

### Tokenization Strategies

```python
# Word-level tokenization (default)
word_vectorizer = tf.keras.layers.TextVectorization(
    split='whitespace',
    output_mode='int',
    output_sequence_length=100
)

# Character-level tokenization
char_vectorizer = tf.keras.layers.TextVectorization(
    split='character',
    output_mode='int',
    output_sequence_length=500
)

# Subword tokenization via custom split function
def subword_split(text):
    # Simple example: split into bigram characters
    words = tf.strings.split(text)
    return tf.strings.regex_replace(words, r'(?<=.{2})', ' ')

subword_vectorizer = tf.keras.layers.TextVectorization(
    split=subword_split,
    output_mode='int'
)

# Custom tokenizer with TensorFlow string ops
def custom_tokenizer(text):
    # Lowercase
    text = tf.strings.lower(text)
    # Remove HTML tags
    text = tf.strings.regex_replace(text, r'<[^>]+>', ' ')
    # Remove special characters
    text = tf.strings.regex_replace(text, r'[^a-z0-9\s]', '')
    # Split on whitespace
    return tf.strings.split(text)
```

### Vocabulary Management

```python
# Get vocabulary
vocab = vectorizer.get_vocabulary()
# Returns list of tokens, index 0 is reserved (empty string or OOV)

# Set vocabulary explicitly
vectorizer.set_vocabulary(['the', 'a', 'is', 'in', 'this', 'and'])

# Vocabulary size
print(f"Vocabulary size: {vectorizer.vocabulary_size()}")

# Limit vocabulary size
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20000  # Keep only top 19999 tokens + 1 OOV
)
```

### TF-IDF Details

```python
# TF-IDF with TextVectorization
tfidf = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='tf_idf'
)
tfidf.adapt(corpus_dataset)

# After adapt, IDF weights are computed
# TF-IDF(t, d) = tf(t, d) * idf(t)
# idf(t) = log((1 + N) / (1 + n_t)) + 1
# where N = total documents, n_t = documents containing term t

# Get IDF weights
idf_weights = tfidf.get_weights()
```

### Bag-of-Words Representations

```python
# Multi-hot (binary presence)
bow = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='multi_hot'
)

# Count (term frequency)
count_bow = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='count'
)

# TF-IDF
tfidf_bow = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='tf_idf'
)
```

---

## Image Data Augmentation Patterns

### Standard Augmentation Pipeline

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Use in model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./255)(x)
# ... rest of model
```

### Aggressive Augmentation (for Small Datasets)

```python
aggressive_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom((-0.3, 0.3)),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomBrightness(0.3),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
    tf.keras.layers.RandomCrop(200, 200),
])
```

### RandAugment-style Pattern

```python
# Implementing random augmentation selection
class RandAugment(tf.keras.layers.Layer):
    def __init__(self, num_layers=2, magnitude=10, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.magnitude = magnitude

    def call(self, inputs, training=None):
        if not training:
            return inputs

        # Randomly apply transformations
        x = inputs
        for _ in range(self.num_layers):
            op = tf.random.uniform((), 0, 4, dtype=tf.int32)
            x = tf.switch_case(op, {
                0: lambda: tf.image.random_flip_left_right(x),
                1: lambda: tf.image.random_brightness(x, self.magnitude / 100.0),
                2: lambda: tf.image.random_contrast(x, 1 - self.magnitude/100.0,
                                                      1 + self.magnitude/100.0),
                3: lambda: tf.image.random_saturation(x, 1 - self.magnitude/100.0,
                                                       1 + self.magnitude/100.0),
            })
        return x
```

### CutMix Implementation

```python
def cutmix(images, labels, alpha=1.0):
    """CutMix augmentation."""
    batch_size = tf.shape(images)[0]
    image_h = tf.shape(images)[1]
    image_w = tf.shape(images)[2]

    # Sample lambda from Beta distribution
    lam = tf.random.uniform((), 0, 1)
    lam = tf.maximum(lam, 1 - lam)

    # Random box coordinates
    cut_rat = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(tf.cast(image_w, tf.float32) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(image_h, tf.float32) * cut_rat, tf.int32)

    cx = tf.random.uniform((), 0, image_w, dtype=tf.int32)
    cy = tf.random.uniform((), 0, image_h, dtype=tf.int32)

    x1 = tf.maximum(cx - cut_w // 2, 0)
    y1 = tf.maximum(cy - cut_h // 2, 0)
    x2 = tf.minimum(cx + cut_w // 2, image_w)
    y2 = tf.minimum(cy + cut_h // 2, image_h)

    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))

    # Create the mixed images
    mask = tf.ones_like(images)
    paddings = [[0, 0], [y1, image_h - y2], [x1, image_w - x2], [0, 0]]
    mask = tf.pad(tf.zeros([batch_size, y2 - y1, x2 - x1, tf.shape(images)[3]]),
                  paddings, constant_values=1.0)
    mixed_images = images * mask + tf.gather(images, indices) * (1 - mask)

    return mixed_images, labels, tf.gather(labels, indices), lam
```

### MixUp Implementation

```python
def mixup(images, labels, alpha=0.2):
    """MixUp augmentation."""
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform((), 0, alpha)

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)

    return mixed_images, mixed_labels
```

---

## Normalization Strategies

### Batch Normalization vs Layer Normalization vs Manual Normalization

```python
# 1. Manual normalization (using Normalization preprocessing layer)
# Best for: input feature normalization
model = tf.keras.Sequential([
    tf.keras.layers.Normalization(),  # Must adapt() first
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 2. Batch Normalization
# Best for: internal normalization in deep networks, CNNs
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, 3, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
])

# 3. Layer Normalization
# Best for: transformers, RNNs, small batch sizes
def transformer_block(x, num_heads, key_dim):
    # Pre-norm
    x_norm = tf.keras.layers.LayerNormalization()(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)(x_norm, x_norm)
    x = tf.keras.layers.Add()([x, attn])
    x_norm = tf.keras.layers.LayerNormalization()(x)
    ff = tf.keras.layers.Dense(2048, activation='relu')(x_norm)
    ff = tf.keras.layers.Dense(x.shape[-1])(ff)
    x = tf.keras.layers.Add()([x, ff])
    return x

# 4. Group Normalization
# Best for: small batch sizes, object detection, segmentation
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, use_bias=False),
    tf.keras.layers.GroupNormalization(groups=8),
    tf.keras.layers.ReLU(),
])

# 5. Instance Normalization (via GroupNormalization with groups=channels)
# Best for: style transfer
instance_norm = tf.keras.layers.GroupNormalization(
    groups=-1  # Number of groups = number of channels
)
```

### ImageNet Normalization

```python
# Standard ImageNet normalization
# Mean: [0.485, 0.456, 0.406]
# Std: [0.229, 0.224, 0.225]

# Option 1: Using Rescaling + Normalization
imagenet_norm = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2]
    )
])

# Option 2: Manual normalization
def imagenet_preprocess(image):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    return (image - mean) / std
```

---

## Custom Preprocessing Layers

### Creating a Custom Preprocessing Layer

```python
class ClipValues(tf.keras.layers.Layer):
    """Clips input values to a specified range."""

    def __init__(self, min_value=0.0, max_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        return config
```

### Stateful Custom Preprocessing Layer

```python
class StandardScaler(tf.keras.layers.Layer):
    """Custom stateful preprocessing layer that standardizes features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight(name='count', initializer='zeros', trainable=False)
        self.mean_sum = self.add_weight(name='mean_sum', initializer='zeros', trainable=False)
        self.var_sum = self.add_weight(name='var_sum', initializer='zeros', trainable=False)

    def adapt(self, data):
        # Compute mean and variance from data
        data = tf.cast(data, tf.float32)
        self.count.assign(tf.cast(tf.shape(data)[0], tf.float32))
        self.mean_sum.assign(tf.reduce_mean(data, axis=0) * self.count)
        self.var_sum.assign(tf.math.reduce_variance(data, axis=0) * self.count)

    def call(self, inputs):
        mean = self.mean_sum / (self.count + 1e-7)
        std = tf.sqrt(self.var_sum / (self.count + 1e-7))
        return (tf.cast(inputs, tf.float32) - mean) / (std + 1e-7)
```

### Custom Text Preprocessing Layer

```python
class TextPreprocessor(tf.keras.layers.Layer):
    """Custom text preprocessing combining cleaning and tokenization."""

    def __init__(self, max_length=200, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.vectorizer = tf.keras.layers.TextVectorization(
            output_mode='int',
            output_sequence_length=max_length
        )

    def adapt(self, texts):
        # Custom standardization during adapt
        def standardize(text):
            text = tf.strings.lower(text)
            text = tf.strings.regex_replace(text, r'<br\s*/?>', ' ')
            text = tf.strings.regex_replace(text, r'[^a-z0-9\s]', '')
            return text

        standardized = texts.map(standardize)
        self.vectorizer.adapt(standardized)

    def call(self, inputs):
        # Standardize then vectorize
        text = tf.strings.lower(inputs)
        text = tf.strings.regex_replace(text, r'<br\s*/?>', ' ')
        text = tf.strings.regex_replace(text, r'[^a-z0-9\s]', '')
        return self.vectorizer(text)
```

### Custom Image Augmentation Layer

```python
class RandomGaussianBlur(tf.keras.layers.Layer):
    """Randomly applies Gaussian blur to images."""

    def __init__(self, kernel_size=3, max_sigma=1.0, probability=0.5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.max_sigma = max_sigma
        self.probability = probability

    def call(self, inputs, training=None):
        if not training:
            return inputs

        # Randomly decide whether to apply blur
        should_blur = tf.random.uniform(()) < self.probability

        def apply_blur():
            sigma = tf.random.uniform((), 0.1, self.max_sigma)
            kernel = self._make_gaussian_kernel(sigma)
            channels = tf.shape(inputs)[-1]
            kernel = tf.expand_dims(kernel, -1)
            kernel = tf.repeat(kernel, channels, axis=-1)
            kernel = tf.reshape(kernel, (*kernel.shape[:2], channels, 1))

            # Apply depthwise convolution
            return tf.nn.depthwise_conv2d(
                inputs, kernel,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )

        return tf.cond(should_blur, apply_blur, lambda: inputs)

    def _make_gaussian_kernel(self, sigma):
        x = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=tf.float32)
        gauss = tf.exp(-x**2 / (2 * sigma**2))
        kernel = tf.einsum('i,j->ij', gauss, gauss)
        kernel = kernel / tf.reduce_sum(kernel)
        return tf.expand_dims(tf.expand_dims(kernel, -1), -1)
```

---

## Complete Preprocessing Pipeline Examples

### Image Classification Pipeline

```python
# Full pipeline for image classification
IMG_SIZE = 224
BATCH_SIZE = 32

# Define augmentation (only during training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Build model with preprocessing included
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = data_augmentation(x)  # Only active during training
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```

### Text Classification Pipeline

```python
# Full pipeline for text classification
MAX_VOCAB = 20000
MAX_LENGTH = 200

# Vectorizer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=MAX_LENGTH,
    output_mode='int'
)
vectorizer.adapt(train_text_dataset)

# Model with preprocessing
text_input = tf.keras.Input(shape=(), dtype=tf.string)
x = vectorizer(text_input)
x = tf.keras.layers.Embedding(MAX_VOCAB, 128)(x)
x = tf.keras.layers.SpatialDropout1D(0.2)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.GlobalMaxPooling1D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(text_input, outputs)
```

### Structured Data Pipeline

```python
# Full pipeline for tabular data with mixed feature types
def build_structured_model(num_features, cat_vocab_sizes, num_classes):
    inputs = []
    encoded_features = []

    # Numerical features
    for i in range(num_features):
        inp = tf.keras.Input(shape=(1,), name=f'num_{i}')
        inputs.append(inp)
        encoded_features.append(inp)

    # Normalize all numerical features together
    num_concat = tf.keras.layers.Concatenate()(encoded_features[:num_features])
    num_norm = tf.keras.layers.Normalization()(num_concat)

    # Categorical features
    cat_encoded = []
    for i, vocab_size in enumerate(cat_vocab_sizes):
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name=f'cat_{i}')
        inputs.append(inp)
        lookup = tf.keras.layers.StringLookup(
            max_tokens=vocab_size, output_mode='one_hot'
        )
        encoded = lookup(inp)
        cat_encoded.append(encoded)

    # Combine features
    all_features = [num_norm] + cat_encoded
    x = tf.keras.layers.Concatenate()(all_features)

    # Deep layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)
```
