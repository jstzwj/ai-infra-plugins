# TensorFlow Performance Optimization Reference

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Mixed Precision Training](#mixed-precision-training)
3. [XLA Compilation](#xla-compilation)
4. [Data Pipeline Optimization](#data-pipeline-optimization)
5. [GPU Optimization](#gpu-optimization)
6. [Graph Optimization](#graph-optimization)
7. [Memory Optimization](#memory-optimization)
8. [Distributed Training Optimization](#distributed-training-optimization)
9. [Batch Size Tuning](#batch-size-tuning)
10. [Compilation Caching](#compilation-caching)
11. [Model Optimization](#model-optimization)
12. [Profiling Tools](#profiling-tools)
13. [Common Anti-Patterns](#common-anti-patterns)

---

## Performance Overview

### Identifying Bottlenecks

Performance optimization follows a systematic approach:

1. **Profile first**: Use the TensorFlow Profiler to identify actual
   bottlenecks before optimizing.
2. **Categorize the bottleneck**: Is it compute-bound, memory-bound, or
   data-bound?
3. **Apply targeted optimizations**: Address the identified bottleneck.
4. **Measure improvement**: Re-profile to confirm the optimization helped.

### Common Bottleneck Categories

| Bottleneck | Symptoms | Solutions |
|---|---|---|
| Data loading | Low GPU utilization, high CPU usage | tf.data optimization |
| Compute | High GPU compute utilization, long kernel times | Mixed precision, XLA |
| Memory | OOM errors, excessive memory usage | Gradient checkpointing |
| Communication | High network usage in distributed training | Gradient compression |
| Python overhead | Low device utilization in eager mode | tf.function |

### Performance Workflow

```python
# Step 1: Profile to identify bottlenecks
tf.profiler.experimental.start('./logs')
model.fit(train_dataset, epochs=1)
tf.profiler.experimental.stop()

# Step 2: Analyze in TensorBoard
# tensorboard --logdir=./logs

# Step 3: Apply optimizations (see sections below)

# Step 4: Re-profile and compare
```

---

## Mixed Precision Training

### Overview

Mixed precision training uses FP16 (half precision) for computation while
maintaining FP32 (single precision) for critical operations, providing:

- **2x faster computation** on hardware with Tensor Cores (Volta+ GPUs)
- **Up to 2x memory reduction** for activations
- **Maintained accuracy** through loss scaling

### Policy Configuration

```python
# Enable mixed precision globally
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Available policies:
# 'float32': Full FP32 (default)
# 'mixed_float16': FP16 compute, FP32 master weights
# 'mixed_bfloat16': BFloat16 compute (TPU, some GPUs)
# 'float16': Pure FP16 (not recommended for training)
# 'bfloat16': Pure BFloat16

# Per-layer policy
layer = tf.keras.layers.Dense(
    512,
    dtype='float32'  # Override global policy for this layer
)
```

### LossScaleOptimizer

```python
# Dynamic loss scaling (recommended)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# Dynamic scaling automatically:
# 1. Starts with a large loss scale (e.g., 2^15)
# 2. If gradients contain Inf/NaN, skips the step and reduces scale
# 3. If N consecutive steps succeed, increases the scale

# Custom dynamic loss scale configuration
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    optimizer,
    dynamic=True,
    initial_scale=2**15,
    dynamic_growth_steps=2000
)

# Fixed loss scaling
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    optimizer,
    dynamic=False,
    initial_scale=1024
)
```

### Complete Mixed Precision Training Example

```python
import tensorflow as tf

# 1. Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 2. Build model (layers automatically use FP16 compute)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    # Output layer must be float32 for numerical stability
    tf.keras.layers.Dense(10, dtype='float32'),
])

# 3. Use LossScaleOptimizer
base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)

# 4. Compile
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 5. Train
model.fit(train_dataset, epochs=10)
```

### Hardware Requirements

| Hardware | Support | Notes |
|---|---|---|
| NVIDIA Volta (V100) | Yes | Tensor Cores FP16 |
| NVIDIA Turing (T4, RTX 2080) | Yes | Tensor Cores FP16 |
| NVIDIA Ampere (A100, RTX 3090) | Yes | Tensor Cores FP16/BF16 |
| NVIDIA Hopper (H100) | Yes | Tensor Cores FP16/BF16/FP8 |
| Google TPU v2/v3 | Yes | BF16 native |
| Google TPU v4 | Yes | BF16 native |
| Intel Xeon (SSE/AVX) | Limited | No Tensor Cores |
| AMD GPU (ROCm) | Limited | FP16 via ROCm |

### Common Issues and Solutions

**NaN losses during training**:
```python
# Solution 1: Ensure output layer is float32
output_layer = tf.keras.layers.Dense(10, dtype='float32')

# Solution 2: Check loss scale
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
# Dynamic scaling handles NaN automatically

# Solution 3: Use larger epsilon in layer norm / batch norm
layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
```

**Gradient underflow**:
```python
# If gradients are too small, they become zero in FP16
# Solution: Use LossScaleOptimizer (multiplies loss before backprop)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
    optimizer, initial_scale=2**15)
```

---

## XLA Compilation

### Overview

XLA (Accelerated Linear Algebra) compiles TensorFlow graphs into optimized
machine code for the target device. Benefits include:

- **Kernel fusion**: Combining multiple operations into a single kernel
- **Memory optimization**: Reduced memory bandwidth usage
- **Operation elimination**: Removing redundant computations
- **Target-specific optimization**: Device-specific code generation

### jit_compile

```python
# Enable XLA for a single function
@tf.function(jit_compile=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Enable XLA for Keras model
model.compile(
    optimizer='adam',
    loss='mse',
    jit_compile=True  # Compile entire model with XLA
)
```

### Auto-Clustering

```python
# Enable auto-clustering (XLA for compatible subgraphs)
tf.config.optimizer.set_jit(True)

# Or via environment variable
# TF_XLA_FLAGS="--tf_xla_auto_jit=2" python train.py
```

### XLA Known Issues

```python
# 1. Dynamic shapes: XLA requires static shapes
# Bad:
@tf.function(jit_compile=True)
def bad_dynamic(x):
    # XLA can't handle variable output sizes
    return tf.unique(x)  # Returns variable-sized output

# Good:
@tf.function(jit_compile=True)
def good_static(x):
    return tf.reduce_sum(x)  # Static output shape

# 2. Side effects: XLA may not preserve side-effect ordering
# 3. Custom ops: Must have XLA lowering registered
# 4. String tensors: XLA doesn't support string types
# 5. Some ops lack XLA implementations
```

### XLA Debugging

```python
# Dump XLA computations
TF_XLA_FLAGS="--tf_xla_dump_to=/tmp/xla_dumps"

# Print XLA graph
TF_XLA_FLAGS="--tf_xla_dump_hlo_as_text=true"

# Disable XLA for specific ops
os.environ['TF_XLA_FLAGS'] = '--tf_xla_disable_strict_shape_checks'
```

---

## Data Pipeline Optimization

### Prefetch

```python
# Prefetch overlaps data preprocessing with model execution
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# AUTOTUNE automatically determines the optimal prefetch buffer size
```

### Parallel Mapping

```python
# Map in parallel across multiple CPU cores
dataset = dataset.map(
    preprocess_fn,
    num_parallel_calls=tf.data.AUTOTUNE)

# Ordered parallel mapping (preserves order)
dataset = dataset.map(
    preprocess_fn,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False)  # Allow out-of-order for speed
```

### Caching

```python
# Cache in memory (for datasets that fit in RAM)
dataset = dataset.cache()

# Cache to file
dataset = dataset.cache('/path/to/cache.tfrecord')

# Typical pattern: cache after first epoch
dataset = (tf.data.Dataset.from_tensor_slices(data)
    .cache()                    # Cache after first pass
    .shuffle(buffer_size=10000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE))
```

### Interleave

```python
# Parallel file reading
dataset = tf.data.Dataset.list_files(pattern)
dataset = dataset.interleave(
    tf.data.TFRecordDataset,
    cycle_length=8,             # Number of files to read in parallel
    block_length=1,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False)
```

### Batch Then Map vs Map Then Batch

```python
# SLOW: Map per-element, then batch
dataset = raw_dataset.map(expensive_preprocess).batch(32)

# FAST: Batch first, then vectorized map
dataset = raw_dataset.batch(32).map(vectorized_preprocess)

# The vectorized approach processes entire batches at once
# Example:
def vectorized_preprocess(batch):
    # Operates on batch dimension
    return tf.image.resize(batch, [224, 224])
```

### tf.data Service

```python
# Distributed data preprocessing (separate workers)
dataset = tf.data.experimental.service.distribute(
    processing_mode='distributed_epoch',
    service='grpc://localhost:5000',
    job_name='training_job')
```

### Complete Optimized Pipeline

```python
def create_optimized_dataset(file_pattern, batch_size, is_training=True):
    # 1. List files
    files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    # 2. Interleave for parallel reading
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=16,
        block_length=4,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training)

    # 3. Parse and preprocess in parallel
    dataset = dataset.map(
        parse_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training)

    # 4. Cache if possible
    if not is_training:
        dataset = dataset.cache()

    # 5. Shuffle (training only)
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    # 6. Batch
    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    # 7. Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
```

---

## GPU Optimization

### Memory Growth Configuration

```python
# Enable memory growth (don't allocate all GPU memory at once)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

# Use specific GPU
tf.config.set_visible_devices([gpus[0]], 'GPU')
```

### cuDNN Auto-Tuner

```python
# Enable cuDNN auto-tuner (finds fastest algorithms)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

# Or via environment
# TF_CUDNN_USE_AUTOTUNE=1

# Note: cuDNN auto-tuning happens during the first few iterations
# and may cause variable performance initially
```

### Tensor Core Utilization

Tensor Cores provide 8x throughput improvement for matrix operations when
dimensions are aligned:

```python
# Tensor Core requirements:
# 1. FP16 or BF16 data types (enable mixed precision)
# 2. Dimensions divisible by 8 (for FP16)

# Good: Dimensions aligned to 8
dense_1 = tf.keras.layers.Dense(512)   # 512 % 8 == 0
dense_2 = tf.keras.layers.Dense(256)   # 256 % 8 == 0
dense_3 = tf.keras.layers.Dense(128)   # 128 % 8 == 0

# Bad: Non-aligned dimensions (falls back to non-Tensor-Core path)
dense_bad = tf.keras.layers.Dense(200)  # 200 % 8 != 0

# Convolution Tensor Core alignment:
# - Input channels should be divisible by 8
# - Output channels should be divisible by 8
# - Batch size should be divisible by 8 (or at least 2)
```

### NCCL for Distributed All-Reduce

```python
# Use NCCL for GPU-to-GPU communication
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
    )
)
```

### Gradient Accumulation

```python
# Simulate larger batch sizes by accumulating gradients
@tf.function
def train_with_accumulation(model, optimizer, dataset, accumulation_steps=4):
    gradient_accumulator = [
        tf.Variable(tf.zeros_like(v), trainable=False)
        for v in model.trainable_variables
    ]

    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions) / accumulation_steps

        gradients = tape.gradient(loss, model.trainable_variables)

        # Accumulate gradients
        for acc, grad in zip(gradient_accumulator, gradients):
            acc.assign_add(grad)

        if (step + 1) % accumulation_steps == 0:
            # Apply accumulated gradients
            optimizer.apply_gradients(
                zip(gradient_accumulator, model.trainable_variables))
            # Reset accumulators
            for acc in gradient_accumulator:
                acc.assign(tf.zeros_like(acc))
```

---

## Graph Optimization

### Grappler Passes

TensorFlow Grappler optimizes the computation graph through a series of
passes:

```python
# Configure Grappler via RewriterConfig
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.graph_options.rewriter_options.constant_folding = True
config.graph_options.rewriter_options.arithmetic_optimization = True
config.graph_options.rewriter_options.layout_optimizer = True
config.graph_options.rewriter_options.memory_optimization = (
    tf.compat.v1.RewriterConfig.MANUAL)
```

### Available Grappler Passes

| Pass | Description |
|---|---|
| Constant Folding | Evaluate constant expressions at compile time |
| Arithmetic Optimization | Simplify arithmetic expressions |
| Layout Optimizer | Convert to optimal data layout (NCHW for GPU) |
| Memory Optimizer | Optimize tensor memory allocation |
| Dependency Optimizer | Remove unnecessary control dependencies |
| Function Optimizer | Inline and optimize function calls |
| Shape Optimizer | Simplify shape-related operations |
| Remapper | Replace op sequences with fused implementations |
| Loop Optimizer | Optimize loop operations |
| Debug Stripper | Remove debug operations |

### Enabling Optimizations

```python
# In TF2, most optimizations are enabled by default
# Explicit control:
tf.config.optimizer.set_jit(True)                    # XLA
tf.config.optimizer.set_experimental_options({
    'constant_folding': True,
    'arithmetic_optimization': True,
    'layout_optimizer': True,
    'remapping': True,            # Op remapping for fusion
    'auto_mixed_precision': True,  # Auto mixed precision
    'disable_meta_optimizer': False,
})
```

---

## Memory Optimization

### Gradient Checkpointing (tf.recompute_grad)

```python
# Trade compute for memory: recompute forward pass during backward
@tf.recompute_grad
def expensive_layer(x, training=False):
    return tf.keras.layers.Dense(512, activation='relu')(x)

# Or wrap entire model sections
class CheckpointedModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.block1 = self._make_block()
        self.block2 = self._make_block()
        self.block3 = self._make_block()
        self.classifier = tf.keras.layers.Dense(10)

    def _make_block(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

    @tf.recompute_grad
    def _call_block(self, x, block, training):
        return block(x, training=training)

    def call(self, x, training=False):
        x = self._call_block(x, self.block1, training)
        x = self._call_block(x, self.block2, training)
        x = self._call_block(x, self.block3, training)
        x = tf.reduce_mean(x, axis=[1, 2])
        return self.classifier(x)
```

### Memory-Efficient Attention

```python
# Use memory-efficient attention implementations
# Flash Attention (available in TF 2.12+)
from tensorflow.python.ops import attention

# For transformer models, use efficient attention
class EfficientAttention(tf.keras.layers.Layer):
    def call(self, query, key, value):
        # Use built-in efficient implementations
        return tf.nn.attention(query, key, value)
```

### Reducing Memory Footprint

```python
# 1. Use mixed precision (halves activation memory)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 2. Use smaller data types where possible
# FP16 instead of FP32 for intermediate computations

# 3. Delete unnecessary references
del large_tensor
tf.keras.backend.clear_session()

# 4. Use generator-based data loading
def data_generator():
    while True:
        yield next_batch()

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=tf.TensorSpec(shape=(batch_size, 784), dtype=tf.float32))
```

---

## Distributed Training Optimization

### Gradient Compression

```python
# Compress gradients for communication efficiency
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL,
        # Enable gradient compression
    )
)

# Custom gradient compression
class GradientCompressor:
    def compress(self, gradients):
        # Top-k sparsification
        flat_grad = tf.concat([tf.reshape(g, [-1]) for g in gradients], 0)
        k = int(0.01 * flat_grad.shape[0])  # Keep top 1%
        values, indices = tf.math.top_k(tf.abs(flat_grad), k=k)
        return values, indices

    def decompress(self, values, indices, num_params):
        flat_grad = tf.scatter_nd(
            tf.expand_dims(indices, 1), values, [num_params])
        return flat_grad
```

### Overlapping Computation and Communication

```python
# Use tf.distribute with overlapping
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()

@tf.function
def distributed_train_step(dataset_iterator):
    def step_fn(data):
        x, y = data
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    per_replica_losses = strategy.run(step_fn, args=(next(dataset_iterator),))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

### Choosing a Distribution Strategy

| Strategy | Use Case | Communication | Memory |
|---|---|---|---|
| MirroredStrategy | Single machine, multi-GPU | NCCL all-reduce | Per-GPU copy |
| MultiWorkerMirrored | Multi-machine | NCCL collective | Per-GPU copy |
| TPUStrategy | TPU pods | TPU interconnect | Per-TPU copy |
| ParameterServer | Asynchronous training | PS-to-worker | PS stores params |
| CentralStorage | Small models, multi-GPU | GPU-to-CPU | CPU stores params |

---

## Batch Size Tuning

### Finding Optimal Batch Size

```python
import tensorflow as tf

def find_max_batch_size(model_fn, input_shape, start_batch=16, max_batch=4096):
    """Binary search for the largest batch size that fits in GPU memory."""
    low, high = start_batch, max_batch
    best_batch = start_batch

    while low <= high:
        mid = (low + high) // 2
        try:
            tf.keras.backend.clear_session()
            model = model_fn()
            dummy_input = tf.random.normal([mid] + list(input_shape))
            _ = model(dummy_input)
            best_batch = mid
            low = mid + 1
        except tf.errors.ResourceExhaustedError:
            high = mid - 1
        finally:
            del model
            tf.keras.backend.clear_session()

    return best_batch
```

### Batch Size Considerations

```python
# Larger batch sizes:
# + Better GPU utilization (parallelism)
# + More stable gradient estimates
# + Faster wall-clock time per epoch
# - More memory
# - May need adjusted learning rate
# - May generalize worse (generalization gap)

# Linear scaling rule: if batch size doubles, double the learning rate
base_lr = 0.001
base_batch = 32
actual_batch = 256
scaled_lr = base_lr * (actual_batch / base_batch)

# Warmup: gradually increase learning rate for large batches
class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.base_lr * tf.minimum(1.0, step / self.warmup_steps)
```

### Gradient Accumulation for Large Batches

```python
# Simulate large batch sizes without the memory cost
class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.gradient_accumulator = None

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = loss_fn(y, predictions) / self.accumulation_steps

        gradients = tape.gradient(loss, self.model.trainable_variables)

        if self.gradient_accumulator is None:
            self.gradient_accumulator = [
                tf.Variable(tf.zeros_like(g), trainable=False)
                for g in gradients
            ]

        for acc, grad in zip(self.gradient_accumulator, gradients):
            acc.assign_add(grad)

        return loss

    @tf.function
    def apply_gradients(self):
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulator, self.model.trainable_variables))
        for acc in self.gradient_accumulator:
            acc.assign(tf.zeros_like(acc))
```

---

## Compilation Caching

### XLA Cache

```python
# Enable XLA compilation caching
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=4'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

# Cache XLA compiled programs to disk
os.environ['XLA_FLAGS'] = '--xla_gpu_compilation_cache_dir=/tmp/xla_cache'
```

### tf.function Tracing Optimization

```python
# Reduce retracing by specifying input signatures
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 10], dtype=tf.float32),
])
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Use concrete functions for fixed input shapes
concrete_train = train_step.get_concrete_function(
    tf.TensorSpec([32, 784], tf.float32),
    tf.TensorSpec([32, 10], tf.float32)
)

# Monitor retracing
@tf.function
def monitored_fn(x):
    tf.print("Tracing!")  # Only prints during tracing, not execution
    return x * 2
```

### Autograph Optimization

```python
# tf.function converts Python control flow to graph operations
@tf.function
def optimized_loop(data):
    result = tf.zeros_like(data)
    for i in tf.range(10):  # tf.range for graph mode
        result += data * tf.cast(i, tf.float32)
    return result

# Avoid:
@tf.function
def slow_loop(data):
    result = tf.zeros_like(data)
    for i in range(10):  # Python range: unrolls during tracing
        result += data * float(i)
    return result
```

---

## Model Optimization

### Pruning for Inference

```python
import tensorflow_model_optimization as tfmot

# 1. Train with pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_for_pruning = prune_low_magnitude(model, pruning_params={
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.8,
        begin_step=0,
        end_step=10000)
})

# 2. Train
model_for_pruning.fit(train_data, epochs=10,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

# 3. Strip pruning for export
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# 4. Convert to TFLite (compression aware)
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_model = converter.convert()
```

### Post-Training Quantization

```python
# Dynamic range quantization (simplest)
converter = tf.lite.TFLiteConverter.from_saved_model('model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Full integer quantization
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model('model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
int8_model = converter.convert()

# Float16 quantization
converter = tf.lite.TFLiteConverter.from_saved_model('model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
fp16_model = converter.convert()
```

### Knowledge Distillation

```python
# Teacher-student knowledge distillation
class Distiller(tf.keras.Model):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn,
                distillation_loss_fn, alpha=0.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        # Teacher forward pass (no gradient)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            # Hard loss (student vs true labels)
            student_loss = self.student_loss_fn(y, student_predictions)

            # Soft loss (student vs teacher soft predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1))

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables))

        return {"student_loss": student_loss,
                "distillation_loss": distillation_loss}
```

---

## Profiling Tools

### TensorBoard Profiler

```python
# Profile with TensorBoard callback
callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    profile_batch='10,20',  # Profile batches 10 and 20
)
model.fit(train_data, callbacks=[callback])

# Launch TensorBoard
# tensorboard --logdir=./logs
```

### tf.profiler

```python
# Programmatic profiling
tf.profiler.experimental.start(
    './logs',
    options=tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=1,
    )
)

# Run workload
model.predict(test_data)

tf.profiler.experimental.stop()
```

### Chrome Trace Format

```python
# Export trace as Chrome trace format
tf.profiler.experimental.start('./logs')
# ... run workload ...
tf.profiler.experimental.stop()

# Open chrome://tracing in Chrome and load the trace file
```

### Command-Line Profiling

```bash
# Profile with nvprof (NVIDIA)
nvprof --profile-from-start off python train.py
# Press Ctrl+C to start profiling

# Profile with Nsight Systems
nsys profile -t cuda,osrt,nvtx -o profile python train.py

# Profile with NVIDIA Nsight Compute (kernel-level)
ncu --set full -o profile python train.py
```

---

## Common Anti-Patterns

### Python-Side Loops

```python
# BAD: Python loop with eager execution
results = []
for i in range(1000):
    results.append(model(input_data[i]))

# GOOD: Batched computation
batched_input = tf.stack(input_data)
results = model(batched_input)
```

### Frequent Small Operations

```python
# BAD: Many small operations
def slow_processing(data):
    results = []
    for item in data:
        results.append(tf.square(item))
    return tf.stack(results)

# GOOD: Vectorized operation
def fast_processing(data):
    return tf.square(data)
```

### Unnecessary Host-Device Transfers

```python
# BAD: Frequent transfers between CPU and GPU
for i in range(100):
    gpu_tensor = tf.identity(cpu_array[i])  # Host to device
    result = model(gpu_tensor)
    cpu_result = result.numpy()  # Device to host

# GOOD: Batch and minimize transfers
gpu_batch = tf.constant(cpu_array)  # Single transfer
results = model(gpu_batch)          # All on device
cpu_results = results.numpy()       # Single transfer back
```

### Not Using tf.function

```python
# BAD: Eager execution overhead on every call
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# GOOD: Compile to graph with tf.function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Not Using tf.data

```python
# BAD: Custom data loading
def load_data():
    data = np.load('data.npy')
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# GOOD: Optimized tf.data pipeline
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### Synchronous File I/O

```python
# BAD: Reading files synchronously during training
@tf.function
def train_with_io(x):
    data = tf.io.read_file(filename)  # Blocks training
    return model(x)

# GOOD: Pre-load data with tf.data
files = tf.data.Dataset.list_files(pattern)
dataset = files.interleave(
    tf.data.TFRecordDataset,
    num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### Inefficient Loss Computation

```python
# BAD: Computing loss with explicit one-hot encoding
labels_onehot = tf.one_hot(labels, num_classes)
loss = tf.keras.losses.categorical_crossentropy(labels_onehot, predictions)

# GOOD: Use from_logits with sparse labels
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
    labels, logits)

# BAD: Manual softmax + cross-entropy (numerically unstable)
probs = tf.nn.softmax(logits)
loss = -tf.reduce_sum(labels * tf.math.log(probs))

# GOOD: Use fused softmax cross-entropy
loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
```

### Summary

TensorFlow performance optimization follows these principles:

1. **Profile first**: Identify actual bottlenecks before optimizing.
2. **Use mixed precision**: 2x throughput on Tensor Core hardware.
3. **Optimize data pipeline**: `prefetch`, `cache`, `interleave`, `AUTOTUNE`.
4. **Use tf.function**: Compile to graph for eager overhead elimination.
5. **Use XLA**: Just-in-time compilation for kernel fusion.
6. **Align dimensions**: Ensure Tensor Core compatibility (multiples of 8).
7. **Batch efficiently**: Find the largest batch size that fits in memory.
8. **Use gradient checkpointing**: Trade compute for memory when needed.
9. **Avoid anti-patterns**: Python loops, small ops, host-device transfers.
10. **Quantize for deployment**: INT8 quantization for inference speedup.
