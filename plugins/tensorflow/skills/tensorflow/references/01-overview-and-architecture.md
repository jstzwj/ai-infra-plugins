# TensorFlow Overview and Architecture Reference

## Table of Contents

1. [History and Evolution](#history-and-evolution)
2. [TensorFlow 1.x vs TensorFlow 2.x](#tensorflow-1x-vs-tensorflow-2x)
3. [Architecture Layers](#architecture-layers)
4. [Module Structure](#module-structure)
5. [Execution Modes](#execution-modes)
6. [API Layers](#api-layers)
7. [Build System](#build-system)
8. [TF2 Design Principles](#tf2-design-principles)
9. [Key Python Modules](#key-python-modules)
10. [Migration from TF1 to TF2](#migration-from-tf1-to-tf2)

---

## History and Evolution

### Origins at Google Brain

TensorFlow originated from the Google Brain team as a second-generation machine
learning system, succeeding the DistBelief framework. Key milestones:

- **2011**: DistBelief developed internally at Google for large-scale deep
  learning across distributed systems.
- **2015 (November)**: TensorFlow 0.5 released as open source under the Apache
  2.0 license, initially supporting only Linux and GPU computation.
- **2016**: Distributed TensorFlow introduced; TensorFlow Serving released for
  production deployment.
- **2017**: TensorFlow Lite announced for mobile and embedded devices;
  Eager execution introduced as `tf.contrib.eager`.
- **2019 (September)**: TensorFlow 2.0 released with eager execution as the
  default, Keras as the primary high-level API, and removal of legacy APIs.
- **2020**: TensorFlow 2.x matures with improved performance, XLA compilation,
  and TensorFlow.js for browser-based ML.
- **2021-2023**: Continued evolution with improved DTensor (distributed
  computing), JAX-compatible APIs, enhanced TPU support, and mixed precision
  training improvements.
- **2023-2024**: Introduction of experimental sub-8-bit data types
  (float8_e4m3fn, float8_e5m2, float4_e2m1fn), int4/uint4, int2/uint2 types
  for quantization, and improved XLA compiler integration.

### TensorFlow Design Goals

1. **Portability**: Run the same code on CPUs, GPUs, TPUs, mobile devices,
   and web browsers.
2. **Performance**: Optimized C++ runtime with support for distributed
   training across hundreds of machines.
3. **Flexibility**: From research prototyping to production deployment.
4. **Reproducibility**: Deterministic operations where possible, seeded random.
5. **Extensibility**: Custom ops, gradient functions, and model architectures.

---

## TensorFlow 1.x vs TensorFlow 2.x

### TensorFlow 1.x Paradigm (Declarative / Graph Mode)

TensorFlow 1.x used a **declarative programming model**:

```python
import tensorflow as tf

# Build a computation graph
a = tf.constant(3.0, name='a')
b = tf.constant(4.0, name='b')
c = tf.add(a, b, name='c')

# Create a session to execute
with tf.Session() as sess:
    result = sess.run(c)
    print(result)  # 7.0
```

Key characteristics of TF1:
- Graph construction and execution are separate phases
- `tf.Session` required to evaluate tensors
- Variables require explicit initialization via `tf.global_variables_initializer()`
- Name scopes and variable scopes for organizing graphs
- `tf.placeholder` for feeding data during session runs
- Collections (`tf.GraphKeys`) for managing variables and other objects
- `tf.cond` and `tf.while_loop` for control flow in graphs
- `tf.train.Saver` for checkpoint management

### TensorFlow 2.x Paradigm (Imperative / Eager-First)

TensorFlow 2.x uses an **imperative programming model**:

```python
import tensorflow as tf

# Operations execute immediately
a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.add(a, b)
print(c.numpy())  # 7.0
```

Key characteristics of TF2:
- Eager execution is the default; operations execute immediately
- `tf.function` decorator to compile Python functions into graphs for
  performance
- Keras is the primary high-level API (`tf.keras`)
- Variables are initialized upon creation (no explicit init step)
- `tf.GradientTape` for automatic differentiation
- Python control flow works naturally with eager execution
- `tf.Module` and `tf.keras.layers` for variable tracking
- SavedModel as the primary serialization format

### Side-by-Side Comparison

| Feature | TF 1.x | TF 2.x |
|---------|---------|---------|
| Execution | Graph (deferred) | Eager (immediate) |
| Session | `tf.Session()` required | Not needed |
| Variables | `tf.Variable()` + init op | `tf.Variable()` auto-initialized |
| Control Flow | `tf.cond`, `tf.while_loop` | Python `if`, `while` + AutoGraph |
| Gradients | `tf.gradients()` | `tf.GradientTape` |
| High-level API | `tf.layers`, `tf.slim` | `tf.keras` |
| Distribution | `tf.estimator`, `tf.train` | `tf.distribute.Strategy` |
| Saving | `tf.train.Saver` | `tf.saved_model`, checkpoints |
| Data Input | `tf.placeholder` + `feed_dict` | `tf.data.Dataset` |
| Scopes | `tf.variable_scope`, `tf.name_scope` | Object-oriented (`tf.Module`) |

---

## Architecture Layers

TensorFlow has a layered architecture spanning from high-level Python APIs
down to hardware-specific device layers:

```
+----------------------------------------------------------+
|                  Python Frontend (tf.*)                   |
|   tf.keras | tf.data | tf.image | tf.linalg | tf.signal  |
+----------------------------------------------------------+
|         pywrap_tensorflow (Python/C++ Bridge)             |
+----------------------------------------------------------+
|                    C API (tensorflow/c)                   |
+----------------------------------------------------------+
|                   C++ Core (tensorflow/core)              |
|   Graph | Executor | Kernels | Memory | Gradients         |
+----------------------------------------------------------+
|                   Device Layer (Plugins)                  |
|     CPU      |     GPU (CUDA)    |     TPU (XLA)          |
+----------------------------------------------------------+
```

### Layer Details

#### 1. Python Frontend (`tensorflow/python/`)

The Python frontend provides the user-facing API. It includes:
- **Framework**: Tensor, Variable, ops, dtypes, tensor_shape
- **High-level APIs**: keras, data, distribute, train
- **Operations**: math_ops, array_ops, nn_ops, linalg_ops
- **Eager execution**: tape, backprop, context, function
- **Utilities**: saved_model, summary, profiler

The Python code interacts with the C++ core through `pywrap_tensorflow`,
a SWIG/pybind11-generated wrapper that exposes C++ functions to Python.

#### 2. Python/C++ Bridge (`pywrap_tensorflow`)

`pywrap_tensorflow` is the Python extension module that wraps the C API:
- Direct calls to the C API functions
- Wraps `TF_Session`, `TF_Graph`, `TF_Operation` C types
- Handles memory management between Python and C++ heaps
- Provides `EagerTensor` for eager execution (a C++ object exposed to Python)

#### 3. C API (`tensorflow/c/`)

The C API is the stable, language-agnostic interface:
- `TF_Graph`: Computation graph operations
- `TF_Session`: Session management and graph execution
- `TF_Operation`: Node operations
- `TF_Tensor`: Tensor data exchange
- `TF_Status`: Error handling

This layer enables language bindings for Go, Java, Rust, JavaScript, etc.

#### 4. C++ Core (`tensorflow/core/`)

The C++ core implements the fundamental runtime:
- **Graph**: `Graph`, `Node`, `Edge` classes for computation graph
- **Executor**: Runs operations in a graph, scheduling kernels
- **Kernels**: Operation implementations (e.g., `MatMul`, `Conv2D`)
- **Memory**: Allocator abstraction, tensor buffer management
- **Gradients**: Gradient computation infrastructure
- **Proto definitions**: Protocol buffer message types
- **Framework**: Tensor, TensorShape, DataType, OpKernel
- **Common**: Utilities, logging, platform abstraction
- **Distributed runtime**: Master/worker gRPC communication

#### 5. Compiler (`tensorflow/compiler/`)

The compiler layer contains XLA (Accelerated Linear Algebra):
- **XLA**: Whole-program optimization compiler for TF computations
- **TF2XLA**: Converts TensorFlow graphs to XLA HLO (High Level Optimizer)
- **XLA Service**: Compilation, optimization, and code generation
- **Backends**: CPU, GPU (NVPTX), TPU code generation

#### 6. Device Layer

Device-specific implementations:
- **CPU**: Eigen (linear algebra), MKL-DNN/oneDNN integration
- **GPU**: CUDA kernels, cuDNN integration, NCCL for collective ops
- **TPU**: XLA compilation to TPU instructions, cloud TPU runtime
- **Custom devices**: Extensible device plugin architecture

---

## Module Structure

### Top-Level Source Layout

```
tensorflow/
  tensorflow/
    python/           # Python API implementation
    core/             # C++ core runtime
    compiler/         # XLA compiler
    lite/             # TensorFlow Lite (mobile/embedded)
    cc/               # C++ API
    c/                # C API
    java/             # Java API
    go/               # Go API
    js/               # TensorFlow.js (separate repo)
    stream_executor/  # GPU platform abstraction
  third_party/        # External dependencies
  tools/              # Build tools, docs generation
  WORKSPACE           # Bazel workspace definition
  BUILD               # Top-level build rules
```

### Python Module Details (`tensorflow/python/`)

```
tensorflow/python/
  framework/          # Core framework: Tensor, TensorShape, DType, ops
  ops/                # Operation definitions: math, array, nn, linalg
  eager/              # Eager execution: tape, backprop, function, context
  keras/              # High-level Keras API
  data/               # tf.data pipeline API
  distribute/         # Distribution strategy API
  module/             # tf.Module base class
  trackable/          # Checkpoint/save tracking infrastructure
  saved_model/        # SavedModel serialization
  autograph/          # AutoGraph: Python-to-graph conversion
  compiler/           # XLA-related Python code
  layers/             # Legacy tf.layers API
  training/           # Training utilities, optimizers, checkpoints
  summary/            # TensorBoard summary operations
  debug/              # Debugging tools (tfdbg)
  profiler/           # Profiling tools
  lib/                # Internal libraries
  platform/           # Platform-specific utilities
  tools/              # Code generation, docs tools
```

### C++ Core Details (`tensorflow/core/`)

```
tensorflow/core/
  framework/          # Core data structures: Tensor, NodeDef, OpDef
  graph/              # Graph construction and manipulation
  common_runtime/     # Graph execution, session, executor
  kernel_tests/       # Kernel correctness tests
  kernels/            # Operation kernel implementations
  ops/                # Op registrations and shape functions
  util/               # Utility functions
  platform/           # Platform-specific code
  distributed_runtime/ # Distributed execution
  protobuf/           # Protocol buffer definitions
  grappler/           # Graph optimization passes
  mlir/               # MLIR integration
```

### Compiler Details (`tensorflow/compiler/`)

```
tensorflow/compiler/
  tf2xla/             # TF graph to XLA HLO conversion
  xla/                # XLA compiler
    service/          # Compiler service, optimization passes
    client/           # Client library for XLA
    python/           # Python bindings for XLA
    tools/            # XLA debugging and analysis tools
  mlir/               # MLIR dialects for TensorFlow
  aot/                # Ahead-of-time compilation (tfcompile)
  jit/                # Just-in-time compilation
```

### TensorFlow Lite Details (`tensorflow/lite/`)

```
tensorflow/lite/
  core/               # TFLite core interpreter
  kernels/            # TFLite operation kernels
  delegates/          # Hardware delegates (GPU, NNAPI, CoreML)
  tools/              # Conversion tools (TFLite converter)
  schema/             # FlatBuffer schema for models
  python/             # Python converter API
```

---

## Execution Modes

### Eager Execution (Default in TF2)

Eager execution evaluates operations immediately, returning concrete values:

```python
import tensorflow as tf

# All operations execute immediately
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.matmul(x, x)
print(y.numpy())  # Immediately prints the result
```

**Characteristics**:
- Operations return concrete values (not graph nodes)
- Natural Python debugging (pdb, print, isinstance)
- Immediate error reporting with Python stack traces
- Slightly slower than graph execution for many small ops
- `EagerTensor` subclass of `tf.Tensor` (internal detail)

**When to use**: Prototyping, debugging, interactive development.

### Graph Execution (via `tf.function`)

Graph execution compiles a Python function into a TensorFlow graph for
optimized execution:

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

**Characteristics**:
- Python code is traced once to build a static graph
- Subsequent calls execute the graph without Python overhead
- Better performance for distributed training and TPU
- Supports serialization via SavedModel
- Python side effects (print, list.append) only execute during tracing
- AutoGraph converts Python control flow to graph ops

**Tracing process**:
1. `tf.function` is called with input arguments
2. A `ConcreteFunction` is created by tracing the Python code
3. Graph operations are recorded (not executed immediately)
4. The resulting graph is compiled and cached
5. Subsequent calls with compatible inputs reuse the cached graph

### XLA Compiled Execution

XLA (Accelerated Linear Algebra) provides ahead-of-time and just-in-time
compilation for optimized execution:

```python
# Enable XLA compilation for a function
@tf.function(jit_compile=True)
def xla_matmul(a, b):
    return tf.matmul(a, b)

# Cluster-level XLA (experimental)
tf.config.optimizer.set_jit(True)
```

**Characteristics**:
- Whole-program optimization (operator fusion, memory planning)
- Mandatory on TPU, optional on CPU/GPU
- Can significantly improve performance for compute-bound workloads
- Limited op support (not all TF ops have XLA implementations)
- Strict shape/dtype requirements
- `jit_compile=True` forces XLA; `experimental_compile=True` (deprecated) alias

---

## API Layers

### High-Level API: Keras (`tf.keras`)

The primary API for building and training models:

```python
import tensorflow as tf

# Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile and train
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=5)
```

Key modules:
- `tf.keras.layers`: Dense, Conv2D, LSTM, Embedding, etc.
- `tf.keras.models`: Sequential, Model (Functional API)
- `tf.keras.optimizers`: Adam, SGD, RMSprop, etc.
- `tf.keras.losses`: CrossEntropy, MSE, etc.
- `tf.keras.metrics`: Accuracy, AUC, etc.
- `tf.keras.callbacks`: EarlyStopping, ModelCheckpoint, etc.
- `tf.keras.regularizers`: L1, L2, L1L2
- `tf.keras.constraints`: MaxNorm, NonNeg, etc.
- `tf.keras.initializers`: GlorotUniform, HeNormal, etc.

### Mid-Level APIs

#### `tf.data` - Data Pipeline

```python
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
```

#### `tf.train` - Training Utilities

- Checkpoint management: `tf.train.Checkpoint`
- Optimizer integration
- Learning rate schedules
- CheckpointManager for keeping N latest checkpoints

#### `tf.metrics` - Evaluation Metrics

```python
m = tf.keras.metrics.Mean('loss')
m.update_state(loss_value)
print(m.result())
```

#### `tf.image` - Image Processing

```python
decoded = tf.image.decode_png(raw_bytes)
resized = tf.image.resize(decoded, [224, 224])
augmented = tf.image.random_flip_left_right(resized)
```

#### `tf.signal` - Signal Processing

```python
stft = tf.signal.stft(signal, frame_length=256, frame_step=128)
```

### Low-Level APIs

#### `tf.raw_ops` - Direct Op Wrappers

```python
result = tf.raw_ops.MatMul(a=x, b=y, transpose_b=True)
```

Every registered operation has a direct wrapper in `tf.raw_ops`. These
bypass the Python-level argument processing and directly invoke the C++
kernel.

#### C++ API (`tensorflow/cc/`)

Direct C++ API for building graphs and running sessions:
```cpp
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;
using namespace tensorflow::ops;

Scope root = Scope::NewRootScope();
auto a = Const(root, 3.0);
auto b = Const(root, 4.0);
auto c = Add(root, a, b);
ClientSession session(root);
Tensor output;
session.Run({c}, &output);
```

#### C API (`tensorflow/c/`)

The stable C API used for language bindings:
```c
TF_Graph* graph = TF_NewGraph();
TF_Operation* a = tf_constant(3.0, graph);
TF_Operation* b = tf_constant(4.0, graph);
TF_Operation* c = tf_add(a, b, graph);
```

---

## Build System

### Bazel

TensorFlow uses Bazel as its primary build system:

**WORKSPACE file** (`/WORKSPACE`):
- Defines the workspace name and external dependencies
- Configures toolchains (CUDA, Python, etc.)
- Manages repository rules for third-party libraries

**BUILD files**:
- Define compilation targets (cc_library, py_library, tf_gen_op_libs)
- Specify dependencies between packages
- Configure test targets

**configure.py**:
- Interactive configuration script
- Detects CUDA, cuDNN, Python paths
- Generates `.bazelrc` with platform-specific settings

**Key Bazel build patterns**:
```python
# Python library
py_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:dtypes",
    ],
)

# C++ kernel
tf_kernel_library(
    name = "matmul_op",
    srcs = ["matmul_op.cc"],
    deps = [
        "//tensorflow/core:framework",
    ],
)

# Generated op files
tf_gen_op_libs(
    op_lib_names = ["math_ops"],
)
```

**Common build commands**:
```bash
# Build the pip package
bazel build //tensorflow/tools/pip_package:build_pip_package

# Run tests
bazel test //tensorflow/python:math_ops_test

# Build with CUDA support
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package

# Build with specific Python version
bazel build --python_path=/usr/bin/python3 //tensorflow/tools/pip_package:build_pip_package
```

### Build Configuration Options

- `--config=cuda`: Enable CUDA GPU support
- `--config=rocm`: Enable ROCm GPU support
- `--config=mkl`: Enable Intel MKL-DNN/oneDNN
- `--config=noaws`: Disable AWS support
- `--config=nogcp`: Disable GCP support
- `--config=opt`: Optimized build
- `--config=dbg`: Debug build with symbols

---

## TF2 Design Principles

### 1. Eager-First

Operations execute immediately by default. No need to build a graph and
run it in a session. This makes debugging natural and code intuitive.

```python
# TF2: Immediate execution
x = tf.constant([1, 2, 3])
y = tf.reduce_sum(x)
print(y.numpy())  # 6
```

### 2. Functions, Not Sessions

Instead of `tf.Session`, use `tf.function` to create graph-compiled
functions for performance:

```python
# TF2: Use tf.function for graph compilation
@tf.function
def compute(x):
    return tf.reduce_sum(x ** 2)

result = compute(tf.constant([1.0, 2.0, 3.0]))
```

### 3. Python Objects, Not Name Scopes

Instead of string-based name scopes and collections, use object-oriented
patterns:

```python
# TF2: Object-oriented variable management
class MyModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([3, 2]))
        self.b = tf.Variable(tf.zeros([2]))

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

model = MyModel()
print(model.trainable_variables)  # Automatic tracking
```

### 4. Keras as the Primary API

`tf.keras` is the recommended high-level API for most users:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

### 5. tf.data for Input Pipelines

`tf.data` replaces `feed_dict` and queue-based pipelines:

```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
model.fit(dataset, epochs=10)
```

### 6. SavedModel as the Serialization Format

`tf.saved_model` is the standard format for saving and loading models:

```python
# Save
tf.saved_model.save(model, '/path/to/model')

# Load
loaded = tf.saved_model.load('/path/to/model')
```

---

## Key Python Modules

### `tensorflow/python/framework/` - Core Framework

The foundation of the TensorFlow Python API.

| Module | Purpose |
|--------|---------|
| `tensor.py` | `tf.Tensor` class, `tf.TensorSpec`, `BoundedTensorSpec` |
| `tensor_shape.py` | `tf.TensorShape`, `Dimension` for shape representation |
| `dtypes.py` | `tf.DType` class and all dtype constants (float32, int64, etc.) |
| `ops.py` | `tf.Operation`, `tf.Graph`, graph construction |
| `constant_op.py` | `tf.constant`, `tf.zeros`, `tf.ones`, `tf.fill` |
| `composite_tensor.py` | Base class for composite tensors (RaggedTensor, SparseTensor) |
| `type_spec.py` | `tf.TypeSpec` for type specifications |
| `tensor_spec.py` | Re-exports from `tensor.py` |
| `errors.py` | Error classes (InvalidArgumentError, NotFoundError, etc.) |
| `function.py` | `tf.function`-related graph utilities |
| `func_graph.py` | Function graph construction |
| `device.py` | Device specification |
| `config.py` | Runtime configuration (GPU options, threading) |
| `sparse_tensor.py` | `tf.SparseTensor` |
| `indexed_slices.py` | `tf.IndexedSlices` for sparse gradients |
| `random_seed.py` | Random seed management |
| `versions.py` | Version information |
| `smart_cond.py` | Smart conditional execution |

### `tensorflow/python/ops/` - Operations

Operation implementations that wrap C++ kernels.

| Module | Purpose |
|--------|---------|
| `math_ops.py` | Element-wise math, reduction, comparison operations |
| `array_ops.py` | Shape manipulation, slicing, stacking, gathering |
| `nn_ops.py` | Neural network ops: conv, pool, softmax, normalization |
| `nn_impl.py` | High-level NN utilities: sigmoid_cross_entropy, batch_norm |
| `linalg_ops.py` | Linear algebra: matmul, solve, inverse, decomposition |
| `variables.py` | `tf.Variable`, variable management, collections |
| `resource_variable_ops.py` | `ResourceVariable` implementation |
| `control_flow_ops.py` | `tf.cond`, `tf.while_loop`, `tf.switch_case` |
| `gradients_impl.py` | Gradient computation implementation |
| `custom_gradient.py` | `tf.custom_gradient` decorator |
| `state_ops.py` | Variable update operations (assign, assign_add) |
| `random_ops.py` | Random number generation |
| `string_ops.py` | String manipulation operations |
| `image_ops.py` | Image processing operations |
| `clip_ops.py` | Gradient/value clipping |
| `init_ops.py` | Variable initializers (TF1 compatibility) |
| `embedding_ops.py` | Embedding lookup operations |
| `rnn.py` | RNN cell and dynamic RNN |
| `tensor_array_ops.py` | `tf.TensorArray` for dynamic arrays |
| `sparse_ops.py` | Sparse tensor operations |
| `ragged/` | Ragged tensor operations |
| `signal/` | Signal processing (FFT, STFT) |
| `data_flow_grad.py` | Gradients for data flow ops |
| `partitioned_variables.py` | Partitioned variable support |

### `tensorflow/python/eager/` - Eager Execution

Eager mode implementation and gradient computation.

| Module | Purpose |
|--------|---------|
| `tape.py` | `Tape` class for gradient recording |
| `backprop.py` | `tf.GradientTape`, gradient computation in eager mode |
| `def_function.py` | `tf.function` decorator, `Function` class |
| `function.py` | `ConcreteFunction`, `AtomicFunction` |
| `context.py` | Eager execution context management |
| `execute.py` | Operation execution in eager mode |
| `core.py` | Core eager execution primitives |
| `forwardprop.py` | Forward-mode automatic differentiation |
| `imperative_grad.py` | C++ gradient computation bridge |
| `wrap_function.py` | Function wrapping utilities |
| `polymorphic_function/` | Polymorphic function implementation |

### `tensorflow/python/keras/` - Keras API

High-level model building API.

| Module | Purpose |
|--------|---------|
| `engine/` | Layer, Model, Sequential base classes |
| `layers/` | Built-in layers (Dense, Conv2D, LSTM, etc.) |
| `optimizers.py` | Optimizer implementations |
| `losses.py` | Loss function implementations |
| `metrics.py` | Metric implementations |
| `callbacks.py` | Training callbacks |
| `initializers/` | Weight initializers |
| `regularizers.py` | Regularization functions |
| `constraints.py` | Weight constraints |
| `activations.py` | Activation functions |
| `backend.py` | Backend abstraction layer |
| `models.py` | Model utilities (load_model, clone_model) |
| `saving/` | Model saving/loading |
| `mixed_precision/` | Mixed precision training support |
| `distribute/` | Distribution strategy integration |

### `tensorflow/python/autograph/` - AutoGraph

Converts Python code to TensorFlow graph operations.

| Subdirectory | Purpose |
|-------------|---------|
| `converters/` | Control flow converters (if, while, for) |
| `core/` | Core conversion logic |
| `impl/` | Implementation details |
| `lang/` | Language construct handling |
| `operators/` | Operator conversion |
| `pyct/` | Python code transformation toolkit |
| `utils/` | Utility functions |

### Other Important Modules

| Module | Purpose |
|--------|---------|
| `tensorflow/python/data/` | `tf.data.Dataset` pipeline API |
| `tensorflow/python/distribute/` | Distribution strategies |
| `tensorflow/python/module/module.py` | `tf.Module` base class |
| `tensorflow/python/trackable/` | Checkpoint tracking |
| `tensorflow/python/saved_model/` | SavedModel serialization |
| `tensorflow/python/training/` | Training utilities, optimizers |
| `tensorflow/python/summary/` | TensorBoard summaries |
| `tensorflow/python/profiler/` | Profiling tools |

---

## Migration from TF1 to TF2

### API Mapping

| TF 1.x | TF 2.x |
|---------|---------|
| `tf.Session()` | Not needed (eager execution) |
| `tf.placeholder()` | Function arguments |
| `tf.global_variables_initializer()` | Not needed (auto-init) |
| `tf.gradients()` | `tf.GradientTape` |
| `tf.Variable(scope)` | Object-oriented (`tf.Module`, `tf.keras`) |
| `tf.variable_scope` | `tf.Module` / `tf.keras.layers` |
| `tf.name_scope` | Object names or `tf.name_scope` |
| `tf.layers.*` | `tf.keras.layers.*` |
| `tf.contrib.*` | Removed; use `tf.*` directly |
| `tf.train.Saver` | `tf.train.Checkpoint` |
| `tf.GraphKeys` | Object attributes |
| `tf.estimator` | `tf.keras` + `tf.function` |
| `feed_dict` | `tf.data.Dataset` |

### Variable Migration

```python
# TF 1.x
with tf.variable_scope('layer1'):
    w = tf.get_variable('w', [3, 2], initializer=tf.glorot_uniform_initializer())
    b = tf.get_variable('b', [2], initializer=tf.zeros_initializer())

# TF 2.x
class Layer(tf.Module):
    def __init__(self):
        self.w = tf.Variable(tf.initializers.glorot_uniform()([3, 2]))
        self.b = tf.Variable(tf.zeros([2]))
```

### Training Loop Migration

```python
# TF 1.x
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        _, loss_val = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})

# TF 2.x
optimizer = tf.keras.optimizers.Adam(0.001)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for step in range(1000):
    loss_val = train_step(batch_x, batch_y)
```

### Compatibility Shim

TensorFlow 2.x provides `tf.compat.v1` for gradual migration:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Now most TF 1.x code works unchanged
```

Note: `tf.compat.v1` is intended for incremental migration, not long-term
use. New code should use native TF2 APIs.

### Common Migration Issues

1. **Collections removed**: Replace `tf.get_collection()` with object-oriented
   tracking via `tf.Module` or `tf.keras.Model`.

2. **Sessions removed**: Use eager execution or `tf.function`.

3. **Control flow**: Replace `tf.cond`/`tf.while_loop` with Python control
   flow inside `tf.function` (AutoGraph handles conversion).

4. **Feed dict removed**: Use `tf.data.Dataset` for input pipelines.

5. **Variable scopes removed**: Use object-oriented patterns.

6. **`tf.contrib` removed**: Most functionality moved to core `tf.*` or
   separate packages.

7. **Hashability**: Tensors and variables are not hashable in TF2. Use
   `tensor.ref()` or `variable.ref()` for dictionary keys.

8. **Equality**: Tensors use element-wise equality in TF2, not reference
   equality.

9. **Boolean casting**: Using a tensor as a boolean (`if tensor:`) raises
   an error in graph mode. Use `tf.cond` or check specific conditions.

10. **Resource variables**: TF2 always uses `ResourceVariable` instead of
    `RefVariable`. Resource variables are safer (atomic operations) but have
    slightly different semantics for concurrent access.

### Automatic Migration Script

TensorFlow provides `tf_upgrade_v2` for automated code migration:

```bash
tf_upgrade_v2 --infile old_code.py --outfile new_code.py
```

This handles simple API renames and adds `compat.v1` wrappers where needed,
but manual review is required for more complex changes.

---

## Version Information

### Checking TensorFlow Version

```python
import tensorflow as tf

# Version string
print(tf.__version__)  # e.g., "2.15.0"

# Version components
print(tf.version.VERSION)
print(tf.version.GIT_VERSION)
print(tf.version.COMPILER_VERSION)
```

### GPU Support Check

```python
# Check GPU availability
print(tf.config.list_physical_devices('GPU'))

# Check if built with CUDA
print(tf.test.is_built_with_cuda())

# GPU device details
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

---

## Summary

TensorFlow is a comprehensive machine learning framework with a layered
architecture spanning from high-level Python APIs to low-level device-specific
kernels. TF2 represents a major paradigm shift from graph-based to eager-first
execution, with Keras as the primary high-level API and `tf.function` for
performance optimization. The codebase is organized into clear modules:
`framework/` for core types, `ops/` for operations, `eager/` for execution,
`keras/` for the high-level API, and `compiler/` for XLA optimization.

Understanding the architecture is essential for:
- Debugging performance issues
- Writing custom operations and gradients
- Extending TensorFlow with new functionality
- Making informed decisions about API usage
- Migrating TF1 code to TF2

---

## Appendix: Key Entry Points

### TensorFlow Package Init

The main `__init__.py` at `tensorflow/python/__init__.py` orchestrates the
public API surface. Key actions performed during `import tensorflow`:

1. Load the C++ extension module (`pywrap_tensorflow`)
2. Import and re-export submodules (`tf.keras`, `tf.data`, `tf.image`, etc.)
3. Set up eager execution context
4. Configure logging and monitoring
5. Apply TF1/TF2 compatibility settings

### Important File Locations

```
# Python package root
tensorflow/python/__init__.py

# Eager execution setup
tensorflow/python/eager/context.py

# Default TensorFlow 2 behavior
tensorflow/python/tf2.py

# Keras entry point
tensorflow/python/keras/__init__.py

# tf.function implementation
tensorflow/python/eager/polymorphic_function/polymorphic_function.py

# Gradient tape
tensorflow/python/eager/backprop.py

# Variable implementation
tensorflow/python/ops/resource_variable_ops.py
```

### Debugging Architecture Issues

When encountering architecture-related problems:

1. **Import errors**: Check that `pywrap_tensorflow` compiled correctly.
   The C++ extension must match your Python version and OS.

2. **Device errors**: Verify GPU drivers, CUDA toolkit, and cuDNN versions
   match TensorFlow's requirements. Use `tf.config.list_physical_devices()`
   to check available devices.

3. **Memory errors**: Use `tf.config.experimental.set_memory_growth()` to
   prevent TensorFlow from allocating all GPU memory at startup.

4. **Tracing errors in tf.function**: Use `tf.debugging.set_log_device_placement(True)`
   to trace which operations execute where.

5. **Version compatibility**: Check `tf.__version__` and ensure all
   dependencies (NumPy, protobuf, etc.) are compatible.
