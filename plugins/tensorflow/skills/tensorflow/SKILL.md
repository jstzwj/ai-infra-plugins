---
name: tensorflow
description: >
  Comprehensive reference documentation and skill for TensorFlow - the end-to-end open source platform
  for machine learning. Covers TensorFlow 2.x Python API (tensors, operations, variables, autograd,
  tf.function, tf.data, Keras, distributed training), C++ core (graph execution, kernels, session,
  distributed runtime, grappler), XLA compiler (HLO IR, JIT/AOT compilation, MLIR dialects),
  TensorFlow Lite (converter, delegates, operators, TFLite Micro), C/C++ APIs, SavedModel format,
  profiling, debugging, performance optimization, and deployment. Based on TensorFlow 2.x source code analysis.
version: 2.22
---

# TensorFlow - End-to-End Machine Learning Platform

## Overview

TensorFlow is Google's end-to-end open source platform for machine learning. It provides a comprehensive ecosystem of tools, libraries, and community resources that enables researchers and developers to build and deploy ML-powered applications. Originally developed by the Google Brain team for internal ML research, TensorFlow was released as open source in November 2015.

**Core Capabilities:**
1. **Tensor computation** with strong GPU/TPU acceleration (via CUDA, ROCm, XLA)
2. **Automatic differentiation** via `tf.GradientTape` for gradient computation
3. **High-level Keras API** for rapid model development and training
4. **Distributed training** across multiple GPUs, TPUs, and machines
5. **XLA compiler** for ahead-of-time and just-in-time optimization
6. **TensorFlow Lite** for mobile and edge device deployment
7. **TensorFlow Serving** for production model serving
8. **TFRT runtime** for next-generation graph execution

**Supported Hardware:** NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Google TPUs, Intel CPUs, Apple Silicon, ARM, RISC-V

**Supported Platforms:** Linux, macOS, Windows, Android, iOS, Web (TensorFlow.js)

**TensorFlow Version:** 2.22

## Key Architecture Concepts

- **Tensor**: Multi-dimensional array with uniform data type (`dtype`), living on a specific `Device`
- **Operation (Op)**: A node in the computation graph that performs computation on tensors
- **Graph**: Directed dataflow graph of operations (`tf.Graph`)
- **Session**: Execution context for graph-based computation (TF1) or direct eager execution (TF2)
- **Eager Execution**: Default TF2 mode where operations execute immediately (no session needed)
- **tf.function**: Compiles Python functions into graph operations via AutoGraph tracing
- **Variable**: Mutable tensor that persists across executions, used for model parameters
- **Keras Model**: High-level abstraction combining layers, loss, optimizer, and training loop
- **Dataset (tf.data)**: Input pipeline abstraction for efficient data loading and transformation
- **DistributionStrategy**: API for distributed training across multiple devices/workers
- **SavedModel**: Serialization format for complete TF programs (graph + variables + assets)
- **XLA (Accelerated Linear Algebra)**: Domain-specific compiler for linear algebra optimization
- **HLO (High-Level Optimizer)**: XLA's intermediate representation for computation graphs
- **MLIR (Multi-Level IR)**: Compiler infrastructure for representing and transforming TF programs
- **TFRT (TensorFlow Runtime)**: Next-generation runtime for efficient graph execution

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Python Frontend (TF2)                            │
│  tf.keras  │ tf.data │ tf.distribute │ tf.image │ tf.signal │ tf.linalg │
├─────────────────────────────────────────────────────────────────────────┤
│  tf.function + AutoGraph  │  tf.GradientTape  │  tf.saved_model        │
├─────────────────────────────────────────────────────────────────────────┤
│                        Eager Execution Context                          │
├─────────────────────────────────────────────────────────────────────────┤
│                     pywrap_tensorflow (SWIG/Pybind11)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                         C API (tensorflow/c)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                    C++ Core (tensorflow/core)                           │
│  framework  │ graph │ common_runtime │ kernels │ ops │ platform         │
│  distributed_runtime │ grappler │ profiler │ protobuf                   │
├─────────────────────────────────────────────────────────────────────────┤
│                          Device Layer                                   │
│    CPU Device    │    GPU Device (CUDA/ROCm)    │    TPU Device          │
├─────────────────────────────────────────────────────────────────────────┤
│                       Compiler / XLA / MLIR                             │
│  XLA JIT  │  XLA AOT (tfcompile)  │  MLIR Dialects  │  TF-TRT         │
├─────────────────────────────────────────────────────────────────────────┤
│                       Runtime Layer                                     │
│        Classic Runtime (DirectSession)    │    TFRT Runtime             │
└─────────────────────────────────────────────────────────────────────────┘
```

## TF1 vs TF2 Comparison

| Aspect | TF1.x | TF2.x |
|--------|-------|-------|
| Execution | Graph (Session.run) | Eager (immediate) |
| Variables | tf.get_variable, variable_scope | tf.Variable (object-oriented) |
| Control flow | tf.cond, tf.while_loop (graph ops) | Python native + tf.function tracing |
| Training | Manual session.run, feed_dict | Keras model.fit or custom loops with GradientTape |
| Scoping | name_scope, variable_scope | Python objects and modules |
| Saved models | SavedModel builder | tf.saved_model.save/load |
| Distribution | tf.train.ClusterSpec, tf.train.Server | tf.distribute.Strategy |

## Quick Reference

### Basic Tensor Operations
```python
import tensorflow as tf

# Tensor creation
t = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
z = tf.zeros([3, 4])
o = tf.ones([2, 3])
r = tf.random.normal([5, 5])
u = tf.random.uniform([3, 3], 0, 1)
a = tf.range(0, 10, 2)       # [0, 2, 4, 6, 8]
l = tf.linspace(0.0, 1.0, 5) # [0.0, 0.25, 0.5, 0.75, 1.0]

# Operations
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])
print(x + y)       # tf.Tensor([5. 7. 9.], shape=(3,))
print(tf.reduce_sum(x))  # tf.Tensor(6.0, shape=())
print(tf.matmul(t, t))   # Matrix multiplication
```

### Variables
```python
v = tf.Variable(0.0, name='my_variable')
v.assign(1.0)
v.assign_add(2.0)  # v = 3.0
v.assign_sub(1.0)  # v = 2.0
print(v.numpy())   # 2.0

# Trainable variable
w = tf.Variable(tf.random.normal([784, 256]), name='weights')
b = tf.Variable(tf.zeros([256]), name='bias')
```

### Automatic Differentiation
```python
# Gradient computation
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2
dy_dx = tape.gradient(y, x)  # dy/dx = 2x = 6.0

# Multiple gradients
x = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape:
    y = x ** 3
dy_dx = tape.gradient(y, x)    # 27
d2y_dx2 = tape.gradient(dy_dx, x)  # 18
del tape

# Training step
def train_step(model, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = tf.keras.losses.MSE(y_true, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### tf.function (Graph Tracing)
```python
@tf.function
def add(a, b):
    return a + b

# AutoGraph converts Python control flow
@tf.function
def train_loop(model, data, steps):
    for step in tf.range(steps):
        loss = train_step(model, data)
    return loss

# Input signatures for tracing
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32)
])
def predict(x):
    return model(x)
```

### Keras Model
```python
# Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Functional API
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Custom training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### tf.data Pipeline
```python
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

# From TFRecord files
raw_dataset = tf.data.TFRecordDataset(filenames)
parsed_dataset = raw_dataset.map(parse_fn)

# Performance optimization
dataset = dataset.cache()          # Cache in memory
dataset = dataset.interleave(      # Parallel reading
    tf.data.TFRecordDataset,
    cycle_length=4,
    num_parallel_calls=tf.data.AUTOTUNE
)
```

### Save and Load
```python
# SavedModel format (recommended)
tf.saved_model.save(model, '/path/to/model')
loaded = tf.saved_model.load('/path/to/model')
infer = loaded.signatures['serving_default']
result = infer(input_tensor)

# Keras format
model.save('/path/to/model.keras')
loaded_model = tf.keras.models.load_model('/path/to/model.keras')

# Checkpoints
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint.save('/path/to/ckpt')
checkpoint.restore('/path/to/ckpt-1')

# CheckpointManager
manager = tf.train.CheckpointManager(
    checkpoint, '/path/to/ckpts', max_to_keep=3
)
manager.save()
status = checkpoint.restore(manager.latest_checkpoint)
```

### GPU Configuration
```python
# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Found {len(gpus)} GPUs")

# Memory growth (prevent TF from allocating all GPU memory)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Limit GPU memory
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.VirtualDeviceConfiguration(memory_limit=4096)]
)

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# or for bfloat16 on TPUs
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
```

### Distributed Training
```python
# MirroredStrategy (single machine, multiple GPUs)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

# MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = build_model()

# TPUStrategy
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# Custom distributed training loop
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

## Data Type Reference

| TF dtype | Python type | Description |
|----------|-------------|-------------|
| `tf.float16` | `np.float16` | 16-bit half-precision float |
| `tf.bfloat16` | `np.uint16` | 16-bit brain float |
| `tf.float32` | `np.float32` | 32-bit single-precision float |
| `tf.float64` | `np.float64` | 64-bit double-precision float |
| `tf.int8` | `np.int8` | 8-bit signed integer |
| `tf.int16` | `np.int16` | 16-bit signed integer |
| `tf.int32` | `np.int32` | 32-bit signed integer |
| `tf.int64` | `np.int64` | 64-bit signed integer |
| `tf.uint8` | `np.uint8` | 8-bit unsigned integer |
| `tf.uint16` | `np.uint16` | 16-bit unsigned integer |
| `tf.uint32` | `np.uint32` | 32-bit unsigned integer |
| `tf.uint64` | `np.uint64` | 64-bit unsigned integer |
| `tf.bool` | `bool` | Boolean |
| `tf.complex64` | `np.complex64` | 64-bit complex (32+32) |
| `tf.complex128` | `np.complex128` | 128-bit complex (64+64) |
| `tf.string` | `bytes` | Variable-length byte string |
| `tf.qint8` | - | Quantized 8-bit signed int |
| `tf.qint32` | - | Quantized 32-bit signed int |
| `tf.quint8` | - | Quantized 8-bit unsigned int |
| `tf.resource` | - | Handle to a mutable resource |

## High-Priority Best Practices

1. **Use tf.data for input pipelines** - Build efficient, parallelized data pipelines
2. **Use tf.function for performance** - Decorate hot paths with `@tf.function`
3. **Use Keras high-level API** - Prefer `model.fit()` for standard training
4. **Enable mixed precision** - Use `mixed_float16` policy for faster GPU training
5. **Profile first** - Use TensorFlow Profiler to identify bottlenecks
6. **Use SavedModel format** - Preferred for production deployment
7. **Batch your data** - Always use batched operations for GPU efficiency
8. **Use tf.distribute** - Scale training across multiple devices efficiently
9. **Pin GPU memory growth** - Prevent TF from hogging all GPU memory
10. **Use XLA compilation** - `tf.function(jit_compile=True)` for critical sections

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TF_CPP_MIN_LOG_LEVEL` | 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL |
| `TF_FORCE_GPU_ALLOW_GROWTH` | Enable GPU memory growth ("true"/"false") |
| `TF_GPU_THREAD_MODE` | GPU thread mode ("global", "gpu_private") |
| `TF_XLA_FLAGS` | XLA compiler flags |
| `TF_ENABLE_ONEDNN_OPTS` | Enable oneDNN optimizations ("1"/"0") |
| `TF_USE_LEGACY_KERAS` | Use tf_keras instead of Keras 3 ("1"/"0") |
| `TF_USE_MODULAR_FILESYSTEM` | Enable modular filesystem support |
| `TF_DATA_SERVICE_MAX_WORKERS` | Max tf.data service workers |
| `TF_DETERMINISTIC_OPS` | Enable deterministic ops ("1"/"0") |
| `TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE` | Eager mode streaming |

## Documentation Structure

### Part 1: Python API & High-Level Modules
- [01-overview-and-architecture](references/01-overview-and-architecture.md) - TF architecture, execution modes, API layers, module structure, TF1 vs TF2 migration
- [02-tensors-and-operations](references/02-tensors-and-operations.md) - Tensor class, dtypes, shapes, broadcasting, factory functions, indexing, slicing, math operations
- [03-variables-and-resources](references/03-variables-and-resources.md) - tf.Variable, Variable tracking, ResourceVariable, TensorSpec, composite tensors
- [04-automatic-differentiation](references/04-automatic-differentiation.md) - GradientTape, tf.function, AutoGraph, tracing, concrete functions, input signatures
- [05-keras-overview](references/05-keras-overview.md) - Keras architecture, Model, Layer, Sequential, Functional API, subclassing
- [06-keras-layers](references/06-keras-layers.md) - Core, conv, pooling, RNN, attention, normalization, regularization, activation layers
- [07-keras-training](references/07-keras-training.md) - compile, fit, evaluate, predict, callbacks, optimizers, losses, metrics, custom training loops
- [08-keras-preprocessing](references/08-keras-preprocessing.md) - Preprocessing layers, TextVectorization, Normalization, image augmentation
- [09-tf-data](references/09-tf-data.md) - Dataset API, transformations, TFRecord, performance optimization, service
- [10-distributed-training](references/10-distributed-training.md) - DistributionStrategy, MirroredStrategy, MultiWorkerMirroredStrategy, TPUStrategy, ParameterServerStrategy
- [11-saved-model](references/11-saved-model.md) - SavedModel format, signatures, tf.saved_model save/load, module compatibility
- [12-checkpoints](references/12-checkpoints.md) - Checkpoint, CheckpointManager, Trackable, object-based saving
- [13-ops-reference](references/13-ops-reference.md) - Complete ops reference: math, array, nn, linalg, string, image, signal, random, sparse
- [14-control-flow](references/14-control-flow.md) - tf.cond, tf.while_loop, tf.case, tf.switch_case, tf.map_fn, TensorArray
- [15-special-tensors](references/15-special-tensors.md) - SparseTensor, RaggedTensor, StringTensor, composite tensors, extension types

### Part 2: Core C++ Backend
- [16-cpp-framework](references/16-cpp-framework.md) - C++ Tensor, Scope, Node, output, DataType, TensorShape, PartialTensorShape
- [17-graph-construction](references/17-graph-construction.md) - Graph, Node, Edge, GraphDef, NodeDef, shape inference, graph optimization
- [18-session-and-executor](references/18-session-and-executor.md) - Session, DirectSession, GrpcSession, Executor, ExecutorImpl, partition
- [19-kernels](references/19-kernels.md) - OpKernel, OpKernelContext, AsyncOpKernel, Compute(), kernel registration, device kernels
- [20-ops-registration](references/20-ops-registration.md) - REGISTER_OP, OpDefBuilder, shape functions, gradient functions, op kernel registration
- [21-platform-abstraction](references/21-platform-abstraction.md) - Env, FileSystem, Thread, Device, DeviceAttributes, port::Status
- [22-protobuf-definitions](references/22-protobuf-definitions.md) - graph.proto, op_def.proto, tensor.proto, saved_model.proto, meta_graph.proto
- [23-distributed-runtime](references/23-distributed-runtime.md) - WorkerService, MasterService, RPC, ClusterSpec, Server, Remote tensors
- [24-grappler-optimizer](references/24-grappler-optimizer.md) - Grappler optimizers: ConstantFolding, LayoutOptimizer, ArithmeticOptimizer, Remapper
- [25-tfrt-runtime](references/25-tfrt-runtime.md) - TFRT runtime, BEF format, async value, kernels, device context

### Part 3: Compiler & XLA
- [26-xla-overview](references/26-xla-overview.md) - XLA architecture, compilation pipeline, backends (CPU, GPU, TPU)
- [27-hlo-ir](references/27-hlo-ir.md) - HLO instructions, Shape, HloComputation, HloModule, HloPassInterface
- [28-xla-compiler](references/28-xla-compiler.md) - XLA compiler passes, optimization, fusion, layout assignment, buffer assignment
- [29-jit-compilation](references/29-jit-compilation.md) - XLA JIT, tf.function(jit_compile=True), cluster formation, auto-clustering
- [30-aot-compilation](references/30-aot-compilation.md) - XLA AOT, tfcompile, AOT compilation, header generation, Makefile rules
- [31-mlir-dialects](references/31-mlir-dialects.md) - TF dialect, MHLO dialect, Func dialect, legalizations, lowerings, passes
- [32-tensorrt-integration](references/32-tensorrt-integration.md) - TF-TRT, TensorRT optimization passes, TRT engine caching, FP16/INT8 calibration

### Part 4: TensorFlow Lite
- [33-tflite-overview](references/33-tflite-overview.md) - TFLite architecture, flatbuffer model format, interpreter, tensors, delegates
- [34-tflite-converters](references/34-tflite-converters.md) - TFLite Converter, quantization-aware training, post-training quantization, optimizations
- [35-tflite-delegates](references/35-tflite-delegates.md) - GPU delegate, NNAPI delegate, CoreML delegate, Hexagon delegate, custom delegates
- [36-tflite-operators](references/36-tflite-operators.md) - Built-in ops, custom ops, registration, op versioning, op compatibility
- [37-tflite-micro](references/37-tflite-micro.md) - TFLite Micro architecture, memory planning, ARM Cortex-M, RISC-V, Xtensa support

### Part 5: Tools & Deployment
- [38-c-api](references/38-c-api.md) - C API: TF_Session, TF_Graph, TF_Tensor, TF_Operation, TF_Status, buffer management
- [39-cc-api](references/39-cc-api.md) - C++ API: Scope, ClientSession, ops namespace, GraphDef construction
- [40-serving-and-deployment](references/40-serving-and-deployment.md) - TF Serving, REST/gRPC APIs, batch scheduling, model versioning, TF Hub
- [41-profiler-and-debugging](references/41-profiler-and-debugging.md) - TensorFlow Profiler, TensorBoard, tf.debugging, tf.profiler, TraceMe, trace analysis
- [42-performance-optimization](references/42-performance-optimization.md) - Performance guide, mixed precision, XLA tips, data pipeline optimization, GPU best practices
