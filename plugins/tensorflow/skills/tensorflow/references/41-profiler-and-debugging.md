# TensorFlow Profiler and Debugging Reference

## Table of Contents

1. [TensorFlow Profiler](#tensorflow-profiler)
2. [TensorBoard Profiling Plugin](#tensorboard-profiling-plugin)
3. [TraceMe Annotations](#traceme-annotations)
4. [tf.debugging](#tfdebugging)
5. [Numerics Checking](#numerics-checking)
6. [tf.print](#tfprint)
7. [tf.summary](#tfsummary)
8. [tf.errors](#tferrors)
9. [Debugging Tools](#debugging-tools)
10. [Common Debugging Patterns](#common-debugging-patterns)

---

## TensorFlow Profiler

### Profiler API

The TensorFlow Profiler collects hardware resource usage metrics during model
execution.

#### Programmatic Profiling

```python
import tensorflow as tf

# Start profiling
tf.profiler.experimental.start(
    logdir='./logs',
    options=tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=1,
    )
)

# Run training
model.fit(train_dataset, epochs=1)

# Stop profiling
tf.profiler.experimental.stop()
```

#### ProfilerService API

```python
# Profile a remote TensorFlow server
tf.profiler.experimental.client.trace(
    service_addr='localhost:6006',
    logdir='./logs',
    duration_ms=10000,
    num_tracing_attempts=3,
)
```

#### tf.profiler.experimental.trace

```python
# Profile a code block
with tf.profiler.experimental.Trace('my_training_step'):
    # Code to profile
    loss = model(train_batch)
    loss.backward()
    optimizer.step()
```

#### Profiler Options

```python
options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=2,        # 0=off, 1=default, 2=verbose
    device_tracer_level=1,       # 0=off, 1=default
    python_tracer_level=1,       # 0=off, 1=enable Python tracing
    delay_ms=None,               # Delay before starting
)
```

### Profiling with Keras Callback

```python
# Profile during training
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    logdir='./logs',
    histogram_freq=1,
    profile_batch='10,20',  # Profile batches 10 and 20
)

model.fit(
    train_dataset,
    epochs=5,
    callbacks=[tensorboard_callback]
)
```

### Sample Profiling Workflow

```python
# Complete profiling workflow
import tensorflow as tf

# 1. Create model
model = tf.keras.applications.ResNet50(weights=None)

# 2. Prepare data
dummy_input = tf.random.normal([32, 224, 224, 3])

# 3. Warm up (avoid profiling initialization overhead)
for _ in range(5):
    _ = model(dummy_input)

# 4. Profile
tf.profiler.experimental.start('./logs')
for _ in range(10):
    _ = model(dummy_input)
tf.profiler.experimental.stop()

# 5. Launch TensorBoard
# tensorboard --logdir=./logs
```

---

## TensorBoard Profiling Plugin

### Overview Page

The overview page provides a high-level summary:

- **Execution Summary**: Total time, device utilization, kernel time
- **Performance Summary**: Key metrics and recommendations
- **Top Operations**: Most time-consuming operations
- **Op Timeline**: Timeline view of operation execution

### Trace Viewer

The trace viewer shows a detailed timeline of operations:

- **CPU threads**: Shows activity on each CPU thread
- **GPU streams**: Shows kernel execution on each GPU stream
- **Op details**: Duration, memory usage, input/output shapes
- **Dependencies**: Data flow between operations

**Using the Trace Viewer**:
1. Profile your model using the Profiler API.
2. Open TensorBoard and navigate to the Profile tab.
3. Select "trace_viewer" from the Tools dropdown.
4. Zoom and pan to explore the timeline.

### Memory Profiler

The memory profiler shows memory usage over time:

- **Peak memory usage**: Maximum memory consumed during execution.
- **Memory breakdown**: Allocation by operation type.
- **Memory timeline**: Memory allocation and deallocation events.
- **Tensor allocation details**: Size and lifetime of each tensor.

### Op Stats

The Op Stats page provides statistics about operations:

- **Operation types**: Count and total time by operation type
- **Slowest operations**: Individual operations sorted by duration
- **Operation histogram**: Distribution of execution times
- **Memory statistics**: Memory allocated per operation type

### Pod Viewer

For distributed training, the Pod Viewer shows:

- **Worker timelines**: Side-by-side view of all workers
- **Communication patterns**: Data transfer between workers
- **Synchronization points**: All-reduce and barrier operations
- **Straggler detection**: Identify slow workers

### Kernel Stats

GPU kernel statistics:

- **Kernel name**: The compiled GPU kernel
- **Total time**: Aggregate time across all launches
- **Launch count**: Number of times the kernel was launched
- **Average time**: Average duration per launch
- **Registers used**: GPU register usage
- **Shared memory**: Shared memory usage per kernel

### Launching TensorBoard

```bash
# Basic launch
tensorboard --logdir=./logs

# With port specification
tensorboard --logdir=./logs --port 6006

# With bind_all for remote access
tensorboard --logdir=./logs --bind_all

# With profilers
tensorboard --logdir=./logs --preload_plugins profile
```

---

## TraceMe Annotations

### Custom Profiling Annotations

```python
import tensorflow as tf

# Annotate a code section for profiling
@tf.function
def my_training_step(x, y):
    with tf.profiler.experimental.Trace('forward_pass'):
        predictions = model(x, training=True)

    with tf.profiler.experimental.Trace('loss_computation'):
        loss = loss_fn(y, predictions)

    with tf.profiler.experimental.Trace('backward_pass'):
        gradients = tf.gradients(loss, model.trainable_variables)

    return loss
```

### TraceMe in Custom Ops

For C++ custom operations:

```c++
#include "tensorflow/core/profiler/lib/traceme.h"

void MyCustomOp::Compute(OpKernelContext* context) {
    tensorflow::profiler::TraceMe trace_me("MyCustomOp::Compute");

    // ... computation ...

    // With additional metadata
    tensorflow::profiler::TraceMe trace_me(
        [&]() { return absl::StrCat("MyCustomOp::Compute",
                                     " batch_size=", batch_size,
                                     " seq_len=", seq_len); });
}
```

### TraceMe Context

```python
# Add context to traces
with tf.profiler.experimental.Trace('data_processing',
    batch_size=32, num_features=784):
    processed_data = preprocess(raw_data)
```

---

## tf.debugging

### Assertion Operations

TensorFlow provides assertion operations that validate tensor properties during
graph execution. In eager mode, assertions raise Python exceptions.

#### assert_equal

```python
tf.debugging.assert_equal(
    x, y,
    message="Values must be equal",
    summarize=None,  # Max entries to print
    name=None)
```

#### assert_not_equal

```python
tf.debugging.assert_not_equal(
    x, y,
    message="Values must not be equal")
```

#### assert_greater / assert_less

```python
tf.debugging.assert_greater(x, y, message="x must be > y")
tf.debugging.assert_greater_equal(x, y, message="x must be >= y")
tf.debugging.assert_less(x, y, message="x must be < y")
tf.debugging.assert_less_equal(x, y, message="x must be <= y")
```

#### assert_near

```python
# Assert two tensors are close within tolerance
tf.debugging.assert_near(
    x, y,
    rtol=1e-5, atol=1e-8,
    message="Values are not close enough")
```

#### assert_positive / assert_negative

```python
tf.debugging.assert_positive(x, message="All values must be positive")
tf.debugging.assert_negative(x, message="All values must be negative")
tf.debugging.assert_non_negative(x, message="Values must be >= 0")
tf.debugging.assert_non_positive(x, message="Values must be <= 0")
```

#### assert_rank

```python
# Assert exact rank
tf.debugging.assert_rank(x, 2, message="Must be rank 2")

# Assert minimum rank
tf.debugging.assert_rank_at_least(x, 2, message="Must be at least rank 2")

# Assert rank in set
tf.debugging.assert_rank_in(x, [2, 3], message="Must be rank 2 or 3")
```

#### assert_shape

```python
# Assert tensor shape matches expected
tf.debugging.assert_shapes([
    (input_tensor, [None, 784]),
    (labels_tensor, [None, 10]),
])
```

#### assert_type

```python
tf.debugging.assert_type(
    x, tf.float32,
    message="Tensor must be float32")
```

#### assert_integer

```python
tf.debugging.assert_integer(x, message="Values must be integers")
```

#### assert_scalar

```python
tf.debugging.assert_scalar(x, message="Must be a scalar tensor")
```

#### assert_ascending

```python
tf.debugging.assert_ascending(x, message="Values must be in ascending order")
```

#### assert_all_finite

```python
# Assert no NaN or Inf values
tf.debugging.assert_all_finite(
    x,
    message="Tensor contains NaN or Inf")
```

#### assert_almost_equal

```python
# Assert approximate equality with decimal tolerance
tf.debugging.assert_almost_equal(
    x, y,
    decimal=6,
    message="Values differ beyond tolerance")
```

#### assert_proper_iterable

```python
tf.debugging.assert_proper_iterable(values)
```

#### assert_same_float_dtype

```python
# Assert all tensors have the same float dtype
dtype = tf.debugging.assert_same_float_dtype(
    [tensor1, tensor2, tensor3],
    dtype=tf.float32)
```

#### assert_none_equal

```python
# Assert no element equals a specific value
tf.debugging.assert_none_equal(
    x, 0.0,
    message="No element should be zero")
```

### Using Assertions in tf.function

```python
@tf.function
def safe_divide(a, b):
    tf.debugging.assert_positive(b, message="Divisor must be positive")
    return a / b

# In graph mode, assertions are control flow operations
# In eager mode, assertions raise Python exceptions
```

---

## Numerics Checking

### enable_check_numerics

```python
# Enable NaN/Inf detection globally
tf.debugging.enable_check_numerics()

# After enabling, any operation producing NaN or Inf will raise an error:
# InvalidArgumentError: Tensor had NaN values

# All floating-point operations are checked:
a = tf.constant(float('nan'))
b = tf.constant([1.0, 2.0, float('inf')])

# This will raise:
c = a + b  # Error: detected NaN or Inf
```

### disable_check_numerics

```python
# Disable NaN/Inf checking
tf.debugging.disable_check_numerics()
```

### Selective Numerics Checking

```python
# Check specific tensors
@tf.function
def safe_computation(x):
    result = some_computation(x)
    tf.debugging.assert_all_finite(result, "Computation produced NaN/Inf")
    return result
```

### Log Numerics Issues

```python
# Use tf.debugging.check_numerics for graph-mode checking
@tf.function
def checked_computation(x):
    x = tf.debugging.check_numerics(x, "Input has NaN/Inf")
    result = model(x)
    result = tf.debugging.check_numerics(result, "Output has NaN/Inf")
    return result
```

---

## tf.print

### Basic Usage

```python
# Print tensor values
x = tf.constant([1.0, 2.0, 3.0])
tf.print("Value of x:", x)
# Output: Value of x: [1 2 3]

# Print multiple tensors
a = tf.constant(42)
b = tf.constant("hello")
tf.print("a =", a, "b =", b)
# Output: a = 42 b = hello
```

### In tf.function

```python
@tf.function
def my_function(x):
    tf.print("Input shape:", tf.shape(x))
    y = x * 2
    tf.print("Output:", y)
    return y

# tf.print outputs during graph execution, not at trace time
my_function(tf.constant([1.0, 2.0]))
```

### Print Options

```python
# Control output stream
tf.print(x, output_stream=sys.stderr)  # To stderr
tf.print(x, output_stream="log")        # To TF logging

# Control number of elements
tf.print(x, summarize=5)  # Print at most 5 elements
```

### Debugging with tf.print

```python
@tf.function
def debug_function(x):
    # Debug intermediate values
    tf.print("Step 1 - input:", x, summarize=10)

    x = tf.nn.relu(x)
    tf.print("Step 2 - after relu:", x, summarize=10)

    x = tf.reduce_mean(x)
    tf.print("Step 3 - after mean:", x)

    return x
```

---

## tf.summary

### Creating a Summary Writer

```python
# Create file writer
writer = tf.summary.create_file_writer('./logs')

# Or with a unique name
import datetime
logdir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)
```

### Scalar Summary

```python
with writer.as_default():
    for step in range(100):
        loss = train_step()
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('learning_rate', lr, step=step)
```

### Histogram Summary

```python
with writer.as_default():
    for step in range(100):
        weights = model.layers[0].get_weights()[0]
        tf.summary.histogram('weights/layer_0', weights, step=step)
        tf.summary.histogram('gradients/layer_0', grads, step=step)
```

### Image Summary

```python
with writer.as_default():
    # Log images (batch of images)
    images = generate_images()
    tf.summary.image('generated_images', images, step=step, max_outputs=4)

    # Log a single image
    img = tf.io.read_file('image.png')
    img = tf.image.decode_png(img)
    img = tf.expand_dims(img, 0)  # Add batch dimension
    tf.summary.image('input_image', img, step=0)
```

### Text Summary

```python
with writer.as_default():
    tf.summary.text('hyperparameters',
        "Learning rate: 0.001\nBatch size: 32\nOptimizer: Adam", step=0)
    tf.summary.text('config', str(config_dict), step=0)
```

### Audio Summary

```python
with writer.as_default():
    # Log audio tensor
    audio = generate_audio()  # Shape: [batch, samples, channels]
    tf.summary.audio('generated_audio', audio,
        sample_rate=16000, step=step, max_outputs=4)
```

### Graph Summary

```python
# Log Keras model graph
with writer.as_default():
    tf.summary.graph(model.predict.get_concrete_function(
        tf.TensorSpec([None, 784], tf.float32)).graph)
```

### Keras Model Summary

```python
# Log model weights as histograms
with writer.as_default():
    for layer in model.layers:
        for weights in layer.weights:
            tf.summary.histogram(
                f'weights/{layer.name}/{weights.name}',
                weights, step=step)
```

### Custom Summary

```python
from tensorboard.plugins.custom_scalar import layout_pb2

# Define custom layout
layout_summary = layout_pb2.Layout(
    category=[
        layout_pb2.Category(
            title='Training Metrics',
            chart=[
                layout_pb2.Chart(
                    title='Loss',
                    multiline=layout_pb2.Multiline(
                        tag_regex='loss/.*')),
            ]),
    ])

with writer.as_default():
    tf.summary.write('custom_layout', layout_summary)
```

### Using with Keras Callback

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,        # Log weight histograms every epoch
    write_graph=True,        # Log the computation graph
    write_images=True,       # Log weight images
    update_freq='batch',     # 'batch', 'epoch', or integer
    profile_batch=2,         # Profile batch 2
    embeddings_freq=1,       # Log embeddings every epoch
    embeddings_metadata=None
)

model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback]
)
```

---

## tf.errors

### Error Hierarchy

TensorFlow errors are organized in a hierarchy rooted at `OpError`:

```
tf.errors.OpError (base)
    tf.errors.AbortedError
    tf.errors.AlreadyExistsError
    tf.errors.CancelledError
    tf.errors.DataLossError
    tf.errors.DeadlineExceededError
    tf.errors.FailedPreconditionError
    tf.errors.InternalError
    tf.errors.InvalidArgumentError
    tf.errors.NotFoundError
    tf.errors.OutOfRangeError
    tf.errors.PermissionDeniedError
    tf.errors.ResourceExhaustedError
    tf.errors.UnauthenticatedError
    tf.errors.UnavailableError
    tf.errors.UnimplementedError
    tf.errors.UnknownError
```

### Error Details

#### OpError (Base)

```python
try:
    result = some_tf_operation()
except tf.errors.OpError as e:
    print(f"Error code: {e.error_code}")
    print(f"Error message: {e.message}")
    print(f"Node name: {e.node_def}")  # May be None
    print(f"Op: {e.op}")                # The operation that failed
```

#### InvalidArgumentError

```python
# Raised for invalid arguments (shape mismatches, type errors)
try:
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5])
    c = a + b  # Shape mismatch
except tf.errors.InvalidArgumentError as e:
    print(f"Invalid argument: {e.message}")
```

#### NotFoundError

```python
# Raised when a resource is not found
# (file not found, variable not found, op not registered)
try:
    model = tf.saved_model.load("/nonexistent/path")
except tf.errors.NotFoundError as e:
    print(f"Not found: {e.message}")
```

#### ResourceExhaustedError

```python
# Raised when resources are exhausted (OOM, connection limits)
try:
    large_tensor = tf.ones([100000, 100000, 100000])
except tf.errors.ResourceExhaustedError as e:
    print(f"Resource exhausted: {e.message}")
```

#### UnimplementedError

```python
# Raised when an operation is not implemented
# (unsupported device, unsupported data type)
try:
    # Attempting unsupported op on specific device
    with tf.device('/gpu:0'):
        string_op = tf.strings.length("hello")  # May fail on some GPUs
except tf.errors.UnimplementedError as e:
    print(f"Not implemented: {e.message}")
```

#### FailedPreconditionError

```python
# Raised when the system is not in a valid state
# (uninitialized variable, wrong graph state)
try:
    reader = tf.raw_ops.ReaderRead(
        reader_handle=uninit_handle, queue_handle=queue_handle)
except tf.errors.FailedPreconditionError as e:
    print(f"Precondition failed: {e.message}")
```

#### InternalError

```python
# Raised for internal TensorFlow errors
# (CUDA errors, runtime errors)
try:
    result = tf.raw_ops.SomeInternalOp(...)
except tf.errors.InternalError as e:
    print(f"Internal error: {e.message}")
```

### Error Handling Patterns

```python
# Retry on transient errors
import time
from tensorflow.python.framework import errors_impl

def run_with_retry(fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn()
        except (tf.errors.UnavailableError,
                tf.errors.AbortedError,
                tf.errors.DeadlineExceededError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# Handle OOM gracefully
def safe_inference(model, input_data, batch_size=32):
    try:
        return model.predict(input_data, batch_size=batch_size)
    except tf.errors.ResourceExhaustedError:
        # Reduce batch size and retry
        return model.predict(input_data, batch_size=batch_size // 2)
```

---

## Debugging Tools

### tfdbg (TensorFlow Debugger) - TF1

The TensorFlow Debugger (tfdbg) was primarily designed for TF1 graph mode.
In TF2 eager mode, use `tf.debugging` assertions and Python debuggers.

```python
# TF1 tfdbg usage (for reference)
from tensorflow.python import debug as tf_debug

# Wrap session with debugger
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
```

### tf.debugging.experimental (TF2)

```python
# Enable experimental debugging features
tf.debugging.experimental.enable_dump_debug_info(
    dump_root='/tmp/tfdbg2',
    tensor_debug_mode='FULL_TENSOR',
    circular_buffer_size=1000)

# Run your model
model.fit(train_dataset, epochs=1)

# Disable
tf.debugging.experimental.disable_dump_debug_info()
```

### Dump Debugging

```python
# Dump tensor values to files
tf.debugging.experimental.enable_dump_debug_info(
    '/tmp/tfdbg_dump',
    tensor_debug_mode='FULL_HEALTH',  # Summary of tensor health
)

# Debug modes:
# 'FULL_TENSOR': Dump full tensor values
# 'FULL_HEALTH': Dump health summary (no NaN, Inf, etc.)
# 'SHAPE': Dump tensor shapes only
# 'REDUCE_MAX_MIN': Dump max and min values
# 'REDUCE_MAX_MIN_NAN_INF': Max, min, NaN count, Inf count
```

### Python Debugger Integration

```python
# Use pdb with eager execution
import pdb

@tf.function
def debug_function(x):
    # Can't use pdb inside tf.function
    # Use tf.print instead
    tf.print("x:", x)
    return x * 2

# Use pdb in eager mode
def eager_debug(x):
    y = model(x)
    pdb.set_trace()  # Works in eager mode
    return y
```

### Gradient Checking

```python
# Numerical gradient checking
def numerical_gradient(f, x, eps=1e-4):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps
        x_minus = x.copy()
        x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def check_gradient(f, x, analytic_grad, eps=1e-4, tol=1e-5):
    num_grad = numerical_gradient(f, x, eps)
    diff = np.abs(analytic_grad - num_grad)
    max_diff = np.max(diff)
    assert max_diff < tol, f"Gradient check failed: max_diff={max_diff}"
```

---

## Common Debugging Patterns

### Shape Mismatches

```python
# Problem: Shape mismatch during model construction
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Debug: Print shapes at each layer
x = tf.random.normal([32, 784])
for layer in model.layers:
    x = layer(x)
    print(f"{layer.name}: {x.shape}")

# Debug: Use tf.debugging.assert_shapes
@tf.function
def forward_pass(inputs):
    tf.debugging.assert_shapes([
        (inputs, [None, 784]),
    ])
    return model(inputs)
```

### NaN Debugging

```python
# Pattern: Find the source of NaN values

# 1. Enable numerics checking
tf.debugging.enable_check_numerics()

# 2. Check inputs
def check_inputs(x):
    has_nan = tf.reduce_any(tf.math.is_nan(x))
    has_inf = tf.reduce_any(tf.math.is_inf(x))
    tf.print("Has NaN:", has_nan, "Has Inf:", has_inf)
    return x

# 3. Check intermediate values
@tf.function
def debug_model(x):
    tf.print("Input range:", tf.reduce_min(x), tf.reduce_max(x))

    x = model.layers[0](x)
    tf.print("After layer 0 range:", tf.reduce_min(x), tf.reduce_max(x))
    tf.debugging.check_numerics(x, "NaN after layer 0")

    x = model.layers[1](x)
    tf.print("After layer 1 range:", tf.reduce_min(x), tf.reduce_max(x))
    tf.debugging.check_numerics(x, "NaN after layer 1")

    return x

# 4. Common NaN sources:
# - Log of negative numbers: tf.math.log(negative)
# - Division by zero: a / 0
# - Invalid gradient computation
# - Overflow in exp(): tf.exp(very_large_number)
# - Mixed precision without loss scaling
```

### Gradient Issues

```python
# Pattern: Debug gradient problems (vanishing/exploding)

# 1. Monitor gradient norms
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    # Check gradient norms
    for var, grad in zip(model.trainable_variables, gradients):
        if grad is not None:
            tf.print(f"{var.name}: grad_norm =",
                tf.norm(grad), "var_norm =", tf.norm(var))

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 2. Detect vanishing gradients
def check_gradient_health(gradients):
    total_norm = 0
    for grad in gradients:
        if grad is not None:
            total_norm += tf.reduce_sum(grad ** 2).numpy()
    total_norm = total_norm ** 0.5

    if total_norm < 1e-7:
        print("WARNING: Vanishing gradients detected!")
    elif total_norm > 1e3:
        print("WARNING: Exploding gradients detected!")
    return total_norm

# 3. Gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step_with_clipping(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    # Clip by global norm
    clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
    return loss, global_norm
```

### OOM (Out of Memory) Debugging

```python
# Pattern: Debug GPU memory issues

# 1. Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 2. Monitor memory usage
tf.config.experimental.get_memory_usage('GPU:0')

# 3. Profile memory
tf.profiler.experimental.start('./logs', options=
    tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,
        device_tracer_level=1))
# Run model
tf.profiler.experimental.stop()

# 4. Reduce memory usage
# - Use mixed precision
# - Reduce batch size
# - Use gradient checkpointing
@tf.function
def train_with_checkpointing(x, y):
    with tf.GradientTape() as tape:
        # Recompute forward pass during backward to save memory
        with tf.recompute_grad(model_call):
            predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Device Placement Issues

```python
# Pattern: Debug device placement

# 1. Log device placement
tf.debugging.set_log_device_placement(True)

# 2. Explicit device placement
with tf.device('/gpu:0'):
    result = model(input_data)

# 3. Soft placement (fall back to CPU if GPU not available)
tf.config.set_soft_device_placement(True)

# 4. Check which device a tensor is on
tensor = tf.constant([1.0, 2.0])
print(tensor.device)  # e.g., "/job:localhost/replica:0/task:0/device:GPU:0"
```

### Model Accuracy Debugging

```python
# Pattern: Debug model accuracy issues

# 1. Check data preprocessing
def debug_preprocessing(raw_data, processed_data):
    print(f"Raw data range: [{raw_data.min()}, {raw_data.max()}]")
    print(f"Processed data range: [{processed_data.min()}, {processed_data.max()}]")
    print(f"Raw data mean/std: {raw_data.mean():.4f} / {raw_data.std():.4f}")
    print(f"Processed mean/std: {processed_data.mean():.4f} / {processed_data.std():.4f}")

# 2. Check label correctness
def debug_labels(labels, predictions):
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Prediction distribution: {np.bincount(np.argmax(predictions, axis=1))}")

# 3. Overfitting a single batch
def test_overfit_single_batch(model, x_batch, y_batch, epochs=100):
    """If model can't overfit a single batch, there's a bug."""
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    history = model.fit(x_batch, y_batch, epochs=epochs, verbose=1)
    final_loss = history.history['loss'][-1]
    assert final_loss < 0.01, f"Model failed to overfit single batch: loss={final_loss}"

# 4. Check for data leakage
# Ensure train/val/test splits don't share samples
# Ensure no future data leaks into training
```

### tf.function Tracing Issues

```python
# Pattern: Debug tf.function retracing

# 1. Monitor retracing
@tf.function
def my_function(x):
    tf.print("Tracing with:", tf.shape(x))
    return x * 2

# Called with different shapes -> retraces each time
my_function(tf.constant([1.0]))        # Trace 1
my_function(tf.constant([1.0, 2.0]))   # Trace 2
my_function(tf.constant([1.0, 2.0]))   # Reuses Trace 2

# 2. Fix by specifying input signature
@tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
def my_function_fixed(x):
    tf.print("Tracing with:", tf.shape(x))
    return x * 2

# 3. Check for Python side effects
counter = 0
@tf.function
def bad_function(x):
    global counter
    counter += 1  # This only runs during tracing, not every call!
    return x + counter  # counter is captured as a constant

# Use tf.Variable instead:
counter_var = tf.Variable(0)
@tf.function
def good_function(x):
    counter_var.assign_add(1)
    return x + tf.cast(counter_var, tf.float32)
```

---

## Summary

TensorFlow provides comprehensive tools for profiling and debugging:

1. **Profiler**: Collect and analyze performance metrics with TensorBoard
   visualization.
2. **tf.debugging**: Comprehensive assertion operations for validating tensor
   properties.
3. **Numerics checking**: Global NaN/Inf detection for numerical stability.
4. **tf.print**: Debug printing that works in both eager and graph modes.
5. **tf.summary**: TensorBoard logging for scalars, histograms, images, and
   more.
6. **tf.errors**: Structured error hierarchy for proper error handling.
7. **Common patterns**: Shape mismatches, NaN debugging, gradient issues,
   OOM, and tf.function tracing problems all have established debugging
   approaches.
