# TensorFlow tf.data Reference

This document provides comprehensive reference documentation for the TensorFlow tf.data API, covering dataset creation, transformations, performance optimization, TFRecord format, distributed input, and advanced patterns.

---

## Table of Contents

1. [Dataset Creation](#dataset-creation)
2. [Transformations](#transformations)
3. [Performance Optimization](#performance-optimization)
4. [TFRecord Format](#tfrecord-format)
5. [tf.data Service](#tfdata-service)
6. [Dataset Specs](#dataset-specs)
7. [Checkpointing](#checkpointing)
8. [Distributed Input](#distributed-input)
9. [Streaming Data Patterns](#streaming-data-patterns)
10. [Debugging](#debugging)
11. [Advanced Patterns](#advanced-patterns)
12. [Common Recipes](#common-recipes)

---

## Dataset Creation

### from_tensor_slices

Creates a dataset from tensors by slicing along the first dimension.

```python
tf.data.Dataset.from_tensor_slices(
    tensors                            # A tensor, list, tuple, or dict of tensors
                                       # All tensors must have the same first dimension
)
```

**Usage:**
```python
# Basic usage
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
# Elements: 1, 2, 3, 4, 5

# With features and labels
features = np.random.random((1000, 10))
labels = np.random.randint(0, 2, (1000,))
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
# Elements: (feature_vector, label)

# With dictionary
dataset = tf.data.Dataset.from_tensor_slices({
    'image': images,
    'label': labels,
    'metadata': metadata
})

# Multiple tensors as tuple
dataset = tf.data.Dataset.from_tensor_slices((
    np.arange(10),
    np.arange(10, 20)
))
```

### from_tensors

Creates a dataset with a single element from the given tensors.

```python
tf.data.Dataset.from_tensors(
    tensors                            # A tensor, list, tuple, or dict of tensors
)
```

**Usage:**
```python
# Single element dataset
dataset = tf.data.Dataset.from_tensors([1, 2, 3])
# Contains one element: [1, 2, 3]

# Useful for creating constant datasets
const_dataset = tf.data.Dataset.from_tensors(tf.constant([1.0, 2.0, 3.0]))
```

### from_generator

Creates a dataset from a Python generator function.

```python
tf.data.Dataset.from_generator(
    generator,                         # A callable that returns an iterable
    output_signature=None,             # A tf.TypeSpec, or nested structure of TypeSpecs
                                       # (Recommended over output_types/output_shapes)
    args=None,                         # Optional tuple of args for generator
    output_types=None,                 # (Deprecated) Use output_signature
    output_shapes=None                 # (Deprecated) Use output_signature
)
```

**Usage:**
```python
# Basic generator
def data_generator():
    for i in range(100):
        yield np.random.random((10,)), np.random.randint(0, 2)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(10,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Generator with arguments
def data_generator(start, end, batch_size):
    for i in range(start, end, batch_size):
        yield np.random.random((batch_size, 10))

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
    args=(0, 1000, 32)
)

# Generator yielding dictionaries
def dict_generator():
    for i in range(100):
        yield {
            'feature': np.random.random((10,)),
            'label': np.random.randint(0, 2)
        }

dataset = tf.data.Dataset.from_generator(
    dict_generator,
    output_signature={
        'feature': tf.TensorSpec(shape=(10,), dtype=tf.float32),
        'label': tf.TensorSpec(shape=(), dtype=tf.int32)
    }
)

# Reading files from generator
def file_generator(filenames):
    for filename in filenames:
        data = np.load(filename)
        yield data

dataset = tf.data.Dataset.from_generator(
    lambda: file_generator(file_list),
    output_signature=tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
)
```

### range

Creates a dataset of a range of values.

```python
tf.data.Dataset.range(*args)           # Like Python's range()
                                       # range(stop)
                                       # range(start, stop)
                                       # range(start, stop, step)
```

**Usage:**
```python
dataset = tf.data.Dataset.range(5)               # [0, 1, 2, 3, 4]
dataset = tf.data.Dataset.range(2, 5)            # [2, 3, 4]
dataset = tf.data.Dataset.range(1, 10, 3)        # [1, 4, 7]
```

### TFRecordDataset

Creates a dataset from one or more TFRecord files.

```python
tf.data.TFRecordDataset(
    filenames,                        # String or list of strings. TFRecord file paths
    compression_type=None,            # None, 'GZIP', or 'ZLIB'
    buffer_size=None,                 # Integer. Read buffer size in bytes
    num_parallel_reads=None,          # Integer or AUTOTUNE. Parallel file reads
    name=None                         # String
)
```

**Usage:**
```python
# Read a single TFRecord file
dataset = tf.data.TFRecordDataset('data.tfrecord')

# Read multiple files
dataset = tf.data.TFRecordDataset(['train_0.tfrecord', 'train_1.tfrecord'])

# With GZIP compression
dataset = tf.data.TFRecordDataset(
    'data.tfrecord.gz',
    compression_type='GZIP'
)

# Parallel reads across multiple files
filenames = [f'train_{i}.tfrecord' for i in range(10)]
dataset = tf.data.TFRecordDataset(
    filenames,
    num_parallel_reads=tf.data.AUTOTUNE,
    buffer_size=8 * 1024 * 1024  # 8MB buffer
)

# Parse function
def parse_example(serialized):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(serialized, feature_description)

dataset = tf.data.TFRecordDataset('data.tfrecord').map(parse_example)
```

### TextLineDataset

Creates a dataset from text files, one line per element.

```python
tf.data.TextLineDataset(
    filenames,                        # String or list of strings
    compression_type=None,            # None, 'GZIP', or 'ZLIB'
    buffer_size=None,                 # Integer. Read buffer size
    num_parallel_reads=None,          # Integer or AUTOTUNE
    name=None
)
```

**Usage:**
```python
# Read text file line by line
dataset = tf.data.TextLineDataset('text.txt')

# Multiple text files
dataset = tf.data.TextLineDataset(['file1.txt', 'file2.txt'])

# Skip header line
dataset = tf.data.TextLineDataset('data.csv').skip(1)
```

### CsvDataset

Creates a dataset from CSV files.

```python
tf.data.experimental.CsvDataset(
    filenames,                        # String or list of strings
    record_defaults,                  # List of default values (determines dtype and shape)
    compression_type=None,
    buffer_size=None,
    header=False,                     # Boolean. Whether to skip header line
    field_delim=',',                  # String. Delimiter character
    use_quote_delim=True,             # Boolean
    na_value='',                      # String. Value to treat as NA
    select_cols=None,                 # List of integers. Column indices to select
    name=None
)
```

**Usage:**
```python
# Read CSV with type specification
dataset = tf.data.experimental.CsvDataset(
    'data.csv',
    record_defaults=[0.0, 0.0, 0],    # Two float columns, one int column
    header=True,
    field_delim=','
)

# Select specific columns
dataset = tf.data.experimental.CsvDataset(
    'data.csv',
    record_defaults=[0.0, 0],
    select_cols=[1, 3],              # Only read columns 1 and 3
    header=True
)
```

### FixedLengthRecordDataset

Creates a dataset of fixed-length records from files.

```python
tf.data.FixedLengthRecordDataset(
    filenames,                        # String or list of strings
    record_length,                    # Integer. Length of each record in bytes
    header_bytes=0,                   # Integer. Number of header bytes to skip
    footer_bytes=0,                   # Integer. Number of footer bytes to skip
    buffer_size=None,
    compression_type=None,
    num_parallel_reads=None,
    name=None
)
```

### SqlDataset (Experimental)

Creates a dataset from a SQL query.

```python
tf.data.experimental.SqlDataset(
    driver_name,                      # String. Database driver
    data_source_name,                 # String. Connection string
    query,                            # String. SQL query
    output_types                      # List of tf.DType for query output columns
)
```

---

## Transformations

### map

Maps a function across the elements of the dataset.

```python
dataset.map(
    map_func,                         # Function mapping dataset elements
    num_parallel_calls=None,          # Integer or AUTOTUNE. Number of concurrent map calls
    deterministic=None,               # Boolean. If False, elements may be produced out of order
    name=None
)
```

**Usage:**
```python
# Basic map
dataset = dataset.map(lambda x: x * 2)

# With num_parallel_calls for performance
dataset = dataset.map(
    preprocess_function,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Image preprocessing in map
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Map with multiple outputs
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

# Non-deterministic for better performance
dataset = dataset.map(
    lambda x: expensive_transform(x),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
)
```

### flat_map

Maps a function over the dataset and flattens the result.

```python
dataset.flat_map(
    map_func,                         # Function that returns a tf.data.Dataset
    name=None
)
```

**Usage:**
```python
# Expand each element into multiple elements
def expand_window(x):
    return tf.data.Dataset.from_tensors(x).repeat(3)

dataset = dataset.flat_map(expand_window)
# Each element becomes 3 elements

# Create sliding windows
def make_windows(sequence):
    return tf.data.Dataset.from_tensor_slices(
        tf.signal.frame(sequence, frame_length=5, frame_step=1)
    )

dataset = tf.data.Dataset.from_tensor_slices(sequences).flat_map(make_windows)
```

### filter

Filters elements based on a predicate function.

```python
dataset.filter(
    predicate,                        # Function returning a boolean tensor
    name=None
)
```

**Usage:**
```python
# Filter by value
dataset = dataset.filter(lambda x: x > 0)

# Filter by label
dataset = dataset.filter(lambda x, y: y < 10)

# Filter by length
dataset = dataset.filter(lambda seq: tf.shape(seq)[0] >= 10)

# Complex filter
dataset = dataset.filter(
    lambda features, label: tf.logical_and(
        features['value'] > 0,
        label != -1
    )
)
```

### batch

Combines consecutive elements into batches.

```python
dataset.batch(
    batch_size,                       # Integer. Number of elements to combine
    drop_remainder=False,             # Boolean. If True, drop last batch if incomplete
    num_parallel_calls=None,          # Integer or AUTOTUNE
    deterministic=None,
    name=None
)
```

**Usage:**
```python
# Basic batching
dataset = dataset.batch(32)

# Drop incomplete last batch (important for fixed-shape models)
dataset = dataset.batch(32, drop_remainder=True)

# Batch with parallel calls
dataset = dataset.batch(32, num_parallel_calls=tf.data.AUTOTUNE)
```

### unbatch

Splits elements of a batched dataset into individual elements.

```python
dataset.unbatch(name=None)
```

**Usage:**
```python
# Unbatch and rebatch at different size
dataset = dataset.unbatch().batch(64)
```

### padded_batch

Combines consecutive elements into batches with padding.

```python
dataset.padded_batch(
    batch_size,                       # Integer
    padded_shapes=None,               # Shape to pad to. Can be None, int, or TensorShape
    padding_values=None,              # Value to pad with. Defaults to 0 for numbers, '' for strings
    drop_remainder=False,
    name=None
)
```

**Usage:**
```python
# Pad sequences to the same length within each batch
dataset = dataset.padded_batch(
    batch_size=32,
    padded_shapes=[None],            # Pad first dimension to max length in batch
    padding_values=0.0
)

# Pad variable-length sequences with labels
dataset = dataset.padded_batch(
    32,
    padded_shapes=([100], []),       # Sequence padded to 100, label no padding
    padding_values=(0.0, 0)
)

# Nested padding
dataset = dataset.padded_batch(
    32,
    padded_shapes={
        'tokens': [None],
        'attention_mask': [None],
        'label': []
    },
    padding_values={
        'tokens': tf.constant(0, dtype=tf.int64),
        'attention_mask': tf.constant(0, dtype=tf.int64),
        'label': tf.constant(0, dtype=tf.int64)
    }
)
```

### shuffle

Randomly shuffles the elements of the dataset.

```python
dataset.shuffle(
    buffer_size,                      # Integer. Size of the shuffle buffer
                                       # Should be >= dataset size for perfect shuffling
    seed=None,                        # Integer. Random seed for reproducibility
    reshuffle_each_iteration=None,    # Boolean. Default True
    name=None
)
```

**Usage:**
```python
# Shuffle with buffer size equal to dataset size
dataset = dataset.shuffle(buffer_size=10000)

# Approximate shuffle with smaller buffer
dataset = dataset.shuffle(buffer_size=1000)

# Reproducible shuffle
dataset = dataset.shuffle(buffer_size=1000, seed=42)

# Shuffle only on first iteration
dataset = dataset.shuffle(
    buffer_size=10000,
    reshuffle_each_iteration=False
)
```

### repeat

Repeats the dataset a specified number of times.

```python
dataset.repeat(
    count=None,                       # Integer or None. None = repeat indefinitely
    name=None
)
```

**Usage:**
```python
# Repeat indefinitely (common with steps_per_epoch)
dataset = dataset.repeat()

# Repeat specific number of times
dataset = dataset.repeat(10)
```

### cache

Caches elements of the dataset.

```python
dataset.cache(
    filename='',                      # String. If empty, caches in memory.
                                       # If path, caches to file (use .cache('path/to/cache'))
    name=None
)
```

**Usage:**
```python
# In-memory cache (use before random transformations)
dataset = dataset.map(preprocess).cache().shuffle(1000).batch(32)

# File-based cache
dataset = dataset.map(preprocess).cache('/tmp/data_cache').shuffle(1000).batch(32)

# Cache after expensive preprocessing
dataset = (
    tf.data.TFRecordDataset(files)
    .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # Cache parsed data
    .shuffle(10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

### prefetch

Creates a dataset that prefetches elements.

```python
dataset.prefetch(
    buffer_size,                      # Integer or AUTOTUNE.
                                       # AUTOTUNE lets tf.data choose optimal buffer size
    name=None
)
```

**Usage:**
```python
# Always add prefetch at the end of the pipeline
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### take / skip

Creates a dataset with at most `count` elements, or skips `count` elements.

```python
dataset.take(count, name=None)
dataset.skip(count, name=None)
```

**Usage:**
```python
# Take first 100 elements
subset = dataset.take(100)

# Skip header
dataset = tf.data.TextLineDataset('data.csv').skip(1)

# Combine for train/val split
train_dataset = dataset.take(8000)
val_dataset = dataset.skip(8000).take(2000)
```

### concatenate

Concatenates two datasets.

```python
dataset.concatenate(
    another_dataset                   # Another tf.data.Dataset
)
```

**Usage:**
```python
train_data = tf.data.Dataset.from_tensor_slices(train_features)
val_data = tf.data.Dataset.from_tensor_slices(val_features)
combined = train_data.concatenate(val_data)
```

### zip

Combines multiple datasets into a dataset of tuples.

```python
tf.data.Dataset.zip(datasets)         # Datasets to zip (list or tuple)
```

**Usage:**
```python
# Zip features and labels from separate datasets
features = tf.data.Dataset.from_tensor_slices(feature_array)
labels = tf.data.Dataset.from_tensor_slices(label_array)
dataset = tf.data.Dataset.zip((features, labels))

# Zip three datasets
dataset = tf.data.Dataset.zip((ds1, ds2, ds3))

# Zip with dictionary output
dataset = tf.data.Dataset.zip({
    'features': feature_dataset,
    'labels': label_dataset
})
```

### interleave

Maps a function across the dataset and interleaves the results.

```python
dataset.interleave(
    map_func,                         # Function mapping elements to datasets
    cycle_length=None,                # Integer. Number of input elements to process concurrently
    block_length=1,                   # Integer. Number of consecutive elements from each input
    num_parallel_calls=None,          # Integer or AUTOTUNE
    deterministic=None,
    name=None
)
```

**Usage:**
```python
# Interleaved reading of TFRecord files
files = tf.data.Dataset.list_files('train-*.tfrecord')
dataset = files.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=8,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Interleave with custom parsing
dataset = files.interleave(
    lambda fp: tf.data.TFRecordDataset(fp).map(parse_fn),
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

### enumerate

Enumerates the elements of the dataset.

```python
dataset.enumerate(
    start=0,                          # Integer. Starting value for enumeration
    name=None
)
```

**Usage:**
```python
dataset = tf.data.Dataset.range(3).enumerate()
# Elements: (0, 0), (1, 1), (2, 2)

dataset = tf.data.Dataset.range(3).enumerate(start=10)
# Elements: (10, 0), (11, 1), (12, 2)
```

### window

Combines consecutive elements into windows.

```python
dataset.window(
    size,                             # Integer. Number of elements in each window
    shift=None,                       # Integer. Step between window starts. Default = size
    stride=1,                         # Integer. Step between elements within window
    drop_remainder=False,             # Boolean
    name=None
)
```

**Usage:**
```python
# Sliding windows
dataset = tf.data.Dataset.range(6).window(size=3, shift=1, drop_remainder=True)
# Windows: [0,1,2], [1,2,3], [2,3,4], [3,4,5]

# Non-overlapping windows
dataset = tf.data.Dataset.range(10).window(size=5, shift=5, drop_remainder=True)
# Windows: [0,1,2,3,4], [5,6,7,8,9]

# Convert window datasets to tensors
def window_to_tensor(window):
    return window.batch(window.element_spec.shape[0] or 5, drop_remainder=True)

dataset = dataset.window(5, shift=1, drop_remainder=True).flat_map(window_to_tensor)
```

### reduce

Reduces the dataset to a single element.

```python
dataset.reduce(
    initial_state,                    # Initial value for the reduction
    reduce_func                       # Function (state, value) -> new_state
)
```

**Usage:**
```python
# Sum all elements
total = tf.data.Dataset.range(10).reduce(0, lambda state, value: state + value)
# Result: 45

# Count elements
count = dataset.reduce(0, lambda state, _: state + 1)

# Collect elements into a list
result = dataset.reduce(
    tf.zeros([0, 10]),
    lambda state, value: tf.concat([state, [value]], axis=0)
)
```

### scan

Scans a function over the dataset (stateful map).

```python
dataset.scan(
    initial_state,                    # Initial state
    scan_func                         # Function (state, value) -> (new_state, output)
)
```

**Usage:**
```python
# Running sum
def scan_fn(state, value):
    new_state = state + value
    return new_state, new_state

dataset = tf.data.Dataset.range(5).scan(0, scan_fn)
# Elements: 0, 1, 3, 6, 10

# Running average
def running_avg(state, value):
    total, count = state
    new_total = total + value
    new_count = count + 1
    return (new_total, new_count), new_total / new_count
```

### rebatch

Creates a dataset that rebatches elements.

```python
dataset.rebatch(
    batch_size,                       # Integer or tf.Tensor
    drop_remainder=False,
    name=None
)
```

**Usage:**
```python
# Rebatch from batch_size=128 to batch_size=32
dataset = dataset.rebatch(32)
```

### Other Transformations

```python
# unique - removes duplicate elements
dataset = dataset.unique()

# take_while - takes elements while predicate is true
dataset = dataset.take_while(lambda x: x < 5)

# skip_while - skips elements while predicate is true
dataset = dataset.skip_while(lambda x: x < 5)

# choose_from_datasets - chooses elements from multiple datasets
dataset = tf.data.experimental.choose_from_datasets(
    [dataset1, dataset2],
    choice_dataset
)

# sample_from_datasets - samples from multiple datasets
dataset = tf.data.experimental.sample_from_datasets(
    [dataset1, dataset2],
    weights=[0.7, 0.3]
)
```

---

## Performance Optimization

### AUTOTUNE

The `tf.data.AUTOTUNE` constant (value -1) tells tf.data to automatically determine optimal parameter values.

```python
AUTOTUNE = tf.data.AUTOTUNE

# Use AUTOTUNE for num_parallel_calls
dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
dataset = dataset.interleave(load_fn, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE)
dataset = dataset.prefetch(AUTOTUNE)
```

### Optimal Pipeline Order

The recommended order for dataset transformations is:

1. **Read/Parse** - Read data files (with interleave for parallelism)
2. **Map (preprocess)** - Apply preprocessing with num_parallel_calls=AUTOTUNE
3. **Cache** - Cache preprocessed data (if fits in memory/disk)
4. **Shuffle** - Shuffle with adequate buffer
5. **Batch** - Batch elements
6. **Prefetch** - Always add at the end

```python
# Optimal pipeline template
dataset = (
    tf.data.Dataset.list_files(pattern, shuffle=True)
    .interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # After preprocessing, before random augmentations
    .shuffle(buffer_size=10000)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
```

### Prefetch Buffering

```python
# Prefetch overlaps data preprocessing with model execution
# AUTOTUNE selects optimal buffer size
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# For GPU training, prefetch to GPU device
options = tf.data.Options()
options.experimental_optimization.parallel_batch = True
dataset = dataset.with_options(options)
```

### Interleave Parallelism

```python
# Maximize throughput with interleave
files = tf.data.Dataset.list_files(pattern).shuffle(num_files)

dataset = files.interleave(
    lambda x: tf.data.TFRecordDataset(x, buffer_size=8*1024*1024),
    cycle_length=tf.data.AUTOTUNE,    # Number of files to read concurrently
    block_length=1,                   # Read one element per file before cycling
    num_parallel_calls=tf.data.AUTOTUNE  # Parallel interleave calls
)
```

### Caching Strategies

```python
# Strategy 1: In-memory cache (for small datasets)
dataset = dataset.map(preprocess).cache()

# Strategy 2: File cache (for medium datasets)
dataset = dataset.map(preprocess).cache('/path/to/cache')

# Strategy 3: Snapshot (persisted cache, smarter than file cache)
dataset = dataset.map(preprocess).snapshot('/path/to/snapshot')

# When NOT to cache:
# - After shuffle (cache before shuffle)
# - After random augmentation (cache before augmentation)
# - For very large datasets that don't fit

# Cache before random transformations
dataset = (
    dataset
    .map(deterministic_preprocess)  # Deterministic preprocessing
    .cache()                         # Cache deterministic results
    .map(random_augment)             # Random augmentation (not cached)
    .shuffle(10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
```

### Options for Performance Tuning

```python
options = tf.data.Options()

# Enable auto-tuning of parallelism and buffer sizes
options.autotune.enabled = True
options.autotune.cpu_budget = tf.data.experimental.AUTOTUNE
options.autotune.ram_budget = tf.data.experimental.AUTOTUNE

# Enable graph optimizations
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.filter_fusion = True
options.experimental_optimization.map_and_filter_fusion = True
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_and_batch_fusion = True
options.experimental_optimization.shuffle_and_repeat_fusion = True

# Threading options
options.threading.max_intra_op_parallelism = 1
options.threading.private_threadpool_size = tf.data.AUTOTUNE

dataset = dataset.with_options(options)
```

---

## TFRecord Format

### Writing TFRecord Files

```python
# Helper function to create tf.train.Example
def create_example(features):
    """Create a tf.train.Example from a dictionary of features."""
    feature_dict = {}

    for key, value in features.items():
        if isinstance(value, (int, np.integer)):
            feature_dict[key] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value])
            )
        elif isinstance(value, (float, np.floating)):
            feature_dict[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value])
            )
        elif isinstance(value, bytes):
            feature_dict[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value])
            )
        elif isinstance(value, str):
            feature_dict[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.encode('utf-8')])
            )
        elif isinstance(value, np.ndarray):
            if value.dtype in (np.int32, np.int64):
                feature_dict[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value.flatten())
                )
            elif value.dtype in (np.float32, np.float64):
                feature_dict[key] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value.flatten())
                )

    return tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )

# Write TFRecord file
def write_tfrecord(filename, data_list):
    with tf.io.TFRecordWriter(filename) as writer:
        for data in data_list:
            example = create_example(data)
            writer.write(example.SerializeToString())

# Write compressed TFRecord
options = tf.io.TFRecordOptions(compression_type='GZIP')
with tf.io.TFRecordWriter('data.tfrecord.gz', options=options) as writer:
    for data in data_list:
        example = create_example(data)
        writer.write(example.SerializeToString())
```

### tf.train.Example and Features

```python
# tf.train.Example structure
example = tf.train.Example(
    features=tf.train.Features(feature={
        'image_bytes': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_data])
        ),
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])
        ),
        'height': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[height])
        ),
        'width': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[width])
        ),
        'score': tf.train.Feature(
            float_list=tf.train.FloatList(value=[score])
        ),
        'tags': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b'tag1', b'tag2'])
        ),
        'embedding': tf.train.Feature(
            float_list=tf.train.FloatList(value=embedding.tolist())
        ),
    })
)
```

### tf.train.SequenceExample

For variable-length sequences (e.g., video frames, time series).

```python
sequence_example = tf.train.SequenceExample(
    context=tf.train.Features(feature={
        'video_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b'video_001'])
        ),
        'num_frames': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[100])
        ),
    }),
    feature_lists=tf.train.FeatureLists(feature_list={
        'frames': tf.train.FeatureList(feature=[
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[frame_bytes])
            )
            for frame_bytes in frame_data_list
        ]),
        'timestamps': tf.train.FeatureList(feature=[
            tf.train.Feature(
                float_list=tf.train.FloatList(value=[timestamp])
            )
            for timestamp in timestamp_list
        ]),
    })
)
```

### Parsing TFRecord Features

```python
# FixedLenFeature - fixed shape, required
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),       # Scalar string
    'label': tf.io.FixedLenFeature([], tf.int64),        # Scalar int64
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'embedding': tf.io.FixedLenFeature([128], tf.float32),  # Vector of 128 floats
}

# VarLenFeature - variable length, returns SparseTensor
feature_description_var = {
    'tags': tf.io.VarLenFeature(tf.string),               # Variable-length strings
    'scores': tf.io.VarLenFeature(tf.float32),            # Variable-length floats
}

# FixedLenSequenceFeature - for SequenceExample
context_features = {
    'video_id': tf.io.FixedLenFeature([], tf.string),
}
sequence_features = {
    'frames': tf.io.FixedLenSequenceFeature([], tf.string),
    'timestamps': tf.io.FixedLenSequenceFeature([], tf.float32),
}

# Parse single example
def parse_fn(serialized):
    parsed = tf.io.parse_single_example(serialized, feature_description)
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image = tf.reshape(image, [parsed['height'], parsed['width'], 3])
    return image, parsed['label']

# Parse batch of examples
def parse_batch(serialized_batch):
    parsed = tf.io.parse_example(serialized_batch, feature_description)
    # Process parsed features...
    return parsed

# Parse sequence example
def parse_sequence(serialized):
    context, sequences = tf.io.parse_single_sequence_example(
        serialized,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return context, sequences

# SparseFeature - for sparse data
feature_description_sparse = {
    'sparse_values': tf.io.SparseFeature(
        index_key='sparse_indices',
        value_key='sparse_values',
        dtype=tf.float32,
        size=100
    ),
}
```

### Sharding TFRecord Files

```python
# Write sharded TFRecord files
def write_sharded_tfrecords(data, output_dir, num_shards):
    writers = []
    for i in range(num_shards):
        writers.append(
            tf.io.TFRecordWriter(f'{output_dir}/data-{i:05d}-of-{num_shards:05d}.tfrecord')
        )

    for idx, example in enumerate(data):
        shard = idx % num_shards
        writers[shard].write(example.SerializeToString())

    for writer in writers:
        writer.close()

# Read with sharding
def read_sharded_dataset(output_dir, num_shards):
    files = [f'{output_dir}/data-{i:05d}-of-{num_shards:05d}.tfrecord'
             for i in range(num_shards)]
    return tf.data.TFRecordDataset(files)
```

---

## tf.data Service

Distributed data processing using tf.data service for offloading preprocessing to separate workers.

```python
# Using tf.data service
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.map(expensive_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Register dataset with tf.data service
dataset = dataset.apply(
    tf.data.experimental.service.distribute(
        processing_mode='distributed_epoch',
        service='grpc://localhost:5000',
        job_name='my_job'
    )
)
```

### tf.data Service Cluster Setup

```python
# Start a tf.data service dispatcher
dispatcher = tf.data.experimental.service.DispatchServer(port=5000)

# Start workers
worker = tf.data.experimental.service.WorkerServer(
    port=5001,
    dispatcher_address='localhost:5000'
)
```

---

## Dataset Specs

### element_spec

Returns the type specification of dataset elements.

```python
dataset = tf.data.Dataset.from_tensor_slices(
    (np.zeros((10, 32)), np.zeros(10, dtype=np.int32))
)

print(dataset.element_spec)
# (TensorSpec(shape=(32,), dtype=tf.float64, name=None),
#  TensorSpec(shape=(), dtype=tf.int32, name=None))
```

### tf.data.DatasetSpec

Type specification for tf.data.Dataset objects.

```python
spec = tf.data.DatasetSpec(element_spec=tf.TensorSpec(shape=(None,), dtype=tf.float32))
```

---

## Checkpointing

### Saving and Restoring Iterator State

```python
# Create a checkpointable iterator
dataset = tf.data.Dataset.range(100).batch(10)
iterator = iter(dataset)

# Save iterator state
checkpoint = tf.train.Checkpoint(iterator=iterator)
save_path = checkpoint.save('/tmp/iterator_ckpt')

# Consume some data
for _ in range(3):
    print(next(iterator).numpy())

# Restore iterator state
checkpoint.restore(save_path)
# Iterator resumes from where it was saved

# Integrated with model checkpointing
model = create_model()
dataset = create_dataset()
iterator = iter(dataset)

ckpt = tf.train.Checkpoint(model=model, iterator=iterator, step=tf.Variable(0))
manager = tf.train.CheckpointManager(ckpt, '/tmp/training', max_to_keep=3)

# Training loop with checkpointing
for epoch in range(num_epochs):
    for batch in iterator:
        train_step(batch)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 1000 == 0:
            manager.save()
```

---

## Distributed Input

### strategy.experimental_distribute_dataset

Distributes a dataset across replicas.

```python
strategy = tf.distribute.MirroredStrategy()

# Create global batch dataset
global_batch_size = 32 * strategy.num_replicas_in_sync
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(global_batch_size)

# Distribute the dataset
dist_dataset = strategy.experimental_distribute_dataset(dataset)

# Iterate
for batch in dist_dataset:
    strategy.run(train_step, args=(batch,))
```

### strategy.distribute_datasets_from_function

Distributes dataset creation across replicas using a function.

```python
def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(10000).batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
```

### Input Options

```python
# Control how data is distributed
options = tf.distribute.InputOptions(
    experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,
    experimental_place_dataset_on_device=True,
    experimental_per_replica_buffer_size=1
)

dist_dataset = strategy.experimental_distribute_dataset(dataset, options=options)
```

---

## Streaming Data Patterns

### Windowed Datasets

```python
# Create sliding windows for time series
def create_window_dataset(data, window_size, shift=1, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window_size + 1, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))  # (input, target)
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Usage
windowed_data = create_window_dataset(time_series, window_size=50)
```

### Sliding Windows for Sequence Models

```python
def create_sequence_dataset(sequences, seq_length, batch_size=32):
    def window_to_tensor(window):
        return window.batch(seq_length + 1, drop_remainder=True)

    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.flat_map(
        lambda seq: tf.data.Dataset.from_tensor_slices(seq)
            .window(seq_length + 1, shift=1, drop_remainder=True)
            .flat_map(window_to_tensor)
    )
    dataset = dataset.map(lambda w: (w[:-1], w[1:]))  # Input: all but last, Target: all but first
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

---

## Debugging

### Common Debugging Techniques

```python
# 1. Use take() to inspect first few elements
for element in dataset.take(5):
    print(element)

# 2. Check element spec
print(dataset.element_spec)

# 3. Check dataset cardinality
print(dataset.cardinality())  # Known, Unknown, or Infinite

# 4. Use enumerate for index tracking
for i, element in dataset.enumerate().take(5):
    print(f"Index {i}: {element}")

# 5. Assert next element (experimental)
# dataset = dataset.apply(tf.data.experimental.assert_next('Map'))

# 6. Add debug print in map function
def debug_map(x):
    tf.print("Shape:", tf.shape(x))
    return x
dataset = dataset.map(debug_map)

# 7. Profile data pipeline performance
options = tf.data.Options()
options.experimental_stats.aggregator = tf.data.experimental.StatsAggregator()
dataset = dataset.with_options(options)
```

---

## Advanced Patterns

### Interleaved Reading with Sharding

```python
# Distributed reading: each worker reads its shard
def create_distributed_dataset(filenames, global_batch_size, strategy):
    def dataset_fn(input_context):
        # Shard the dataset for this worker
        local_batch_size = input_context.get_per_replica_batch_size(global_batch_size)

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id
        )
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(local_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    return strategy.distribute_datasets_from_function(dataset_fn)
```

### Bucketing by Sequence Length

```python
def bucket_by_sequence_length(dataset, bucket_boundaries, bucket_batch_sizes,
                               padded_shapes, padding_values=None):
    """Bucket elements by length and batch with dynamic padding."""
    return dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda x, y: tf.shape(x)[0],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            pad_to_bucket_boundary=False
        )
    )

# Usage
dataset = bucket_by_sequence_length(
    dataset,
    bucket_boundaries=[10, 20, 50, 100],
    bucket_batch_sizes=[64, 48, 32, 16, 8],
    padded_shapes=([None], []),
    padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64))
)
```

### Dynamic Padding

```python
# Dynamic padding based on bucket
def dynamic_padding(dataset, batch_size, max_length=None):
    if max_length:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([max_length], []),
            drop_remainder=True
        )
    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], []),
            drop_remainder=True
        )
    return dataset
```

### tf.data.experimental APIs

```python
# Stats aggregation
stats_aggregator = tf.data.experimental.StatsAggregator()
options = tf.data.Options()
options.experimental_stats.aggregator = stats_aggregator
dataset = dataset.with_options(options)

# Optimization configurations
options = tf.data.Options()
options.experimental_optimization.autotune = True
options.experimental_optimization.autotune_algorithm = tf.data.experimental.AutotuneAlgorithm.HILL_CLIMB
dataset = dataset.with_options(options)

# Threading configuration
options = tf.data.Options()
options.threading.max_intra_op_parallelism = 1
options.threading.private_threadpool_size = 4
dataset = dataset.with_options(options)

# Snapshot for persistent caching
dataset = dataset.apply(
    tf.data.experimental.snapshot(
        '/path/to/snapshot',
        compression='GZIP',
        reader_path_prefix=None,
        writer_path_prefix=None,
        shard_size_bytes=10737418240,  # 10GB
        pending_snapshot_expiry_seconds=86400,
        num_reader_threads=tf.data.AUTOTUNE,
        reader_buffer_size=tf.data.AUTOTUNE,
        num_writer_threads=tf.data.AUTOTUNE,
        writer_buffer_size=tf.data.AUTOTUNE
    )
)

# Dense to sparse batch
dataset = dataset.apply(
    tf.data.experimental.dense_to_sparse_batch(
        batch_size=32,
        row_shape=[10]
    )
)
```

---

## Common Recipes

### Image Loading Pipeline

```python
def create_image_dataset(image_paths, labels, batch_size=32, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(len(image_paths))

    def load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0

        if training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)

        return image, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

### Text Loading Pipeline

```python
def create_text_dataset(texts, labels, vocab_size=20000, seq_length=200,
                         batch_size=32, training=True):
    # Create vectorizer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=seq_length,
        output_mode='int'
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(texts))

    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

    if training:
        dataset = dataset.shuffle(10000)

    def vectorize(text, label):
        return vectorizer(text), label

    dataset = dataset.map(vectorize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

### CSV Loading Pipeline

```python
def create_csv_dataset(filepath, label_column, batch_size=32, training=True):
    # Read CSV
    dataset = tf.data.experimental.make_csv_dataset(
        filepath,
        batch_size=batch_size,
        label_name=label_column,
        num_epochs=1,
        shuffle=training,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Alternative with more control
def create_csv_dataset_v2(filepath, feature_columns, label_column, batch_size=32):
    def parse_csv(line):
        defaults = [[0.0]] * len(feature_columns) + [[0]]
        fields = tf.io.decode_csv(line, defaults)
        features = dict(zip(feature_columns + [label_column], fields))
        label = features.pop(label_column)
        return features, label

    dataset = tf.data.TextLineDataset(filepath).skip(1)  # Skip header
    dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
```

### TFRecord Pipeline

```python
def create_tfrecord_dataset(file_pattern, parse_fn, batch_size=32, training=True):
    files = tf.data.Dataset.list_files(file_pattern, shuffle=training)

    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, buffer_size=8*1024*1024),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

### Complete Training Pipeline

```python
def create_training_pipeline(
    train_files, val_files,
    parse_fn, batch_size=32, epochs=100
):
    # Training dataset
    train_ds = tf.data.Dataset.list_files(train_files, shuffle=True)
    train_ds = train_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()

    # Validation dataset
    val_ds = tf.data.Dataset.list_files(val_files, shuffle=False)
    val_ds = val_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=1,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
```
