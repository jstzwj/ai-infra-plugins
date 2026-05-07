# TensorFlow Variables and Resources Reference

## Table of Contents

1. [tf.Variable](#tfvariable)
2. [Variable Methods](#variable-methods)
3. [Variable Properties](#variable-properties)
4. [ResourceVariable vs RefVariable](#resourcevariable-vs-refvariable)
5. [tf.Module](#tfmodule)
6. [tf.TensorSpec](#tftensorspec)
7. [Device Placement for Variables](#device-placement-for-variables)
8. [Variable Initialization Strategies](#variable-initialization-strategies)
9. [Partitioned Variables](#partitioned-variables)
10. [Variable Aggregation in Distributed Settings](#variable-aggregation-in-distributed-settings)
11. [Constraint Functions on Variables](#constraint-functions-on-variables)
12. [Custom Variable Classes](#custom-variable-classes)
13. [Resource Management and Cleanup](#resource-management-and-cleanup)
14. [Variable Saving and Restoration](#variable-saving-and-restoration)

---

## tf.Variable

### Overview

A `tf.Variable` represents a shared, persistent tensor whose value can be
modified during program execution. Variables are the primary mechanism for
maintaining state in TensorFlow models (weights, biases, counters, etc.).

The base class is defined in `tensorflow/python/ops/variables.py`:

```python
@tf_export("Variable", v1=[])
class Variable(trackable.Trackable, metaclass=VariableMetaclass):
```

The actual implementation used in TF2 is `ResourceVariable`, defined in
`tensorflow/python/ops/resource_variable_ops.py`.

### Creation

```python
import tensorflow as tf

# From a constant value
v = tf.Variable(1.0)
print(v)  # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>

# From a list/array
v = tf.Variable([1.0, 2.0, 3.0])

# From a numpy array
import numpy as np
v = tf.Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))

# From a tensor
v = tf.Variable(tf.random.normal([3, 3]))

# With explicit dtype
v = tf.Variable([1, 2, 3], dtype=tf.float64)

# With a name
v = tf.Variable(0.0, name='counter')

# With trainable flag
v_trainable = tf.Variable(1.0, trainable=True)
v_non_trainable = tf.Variable(1.0, trainable=False)

# With constraint function
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=lambda x: tf.clip_by_norm(x, 1.0))

# From a callable (lazy initialization)
v = tf.Variable(lambda: tf.random.normal([100, 100]))

# With shape override (allows assignment of different shapes later)
v = tf.Variable(1.0, shape=tf.TensorShape(None))
v.assign([[1.0, 2.0]])  # Now shape is (1, 2)
```

### Constructor Arguments

```python
tf.Variable(
    initial_value=None,          # Initial value or callable
    trainable=None,              # Whether GradientTape watches this
    validate_shape=True,         # Verify shape is known
    caching_device=None,         # DEPRECATED: device for caching reads
    name=None,                   # Variable name
    variable_def=None,           # Protocol buffer (for restoration)
    dtype=None,                  # Data type override
    import_scope=None,           # Name scope for import
    constraint=None,             # Projection function after updates
    synchronization=VariableSynchronization.AUTO,
    aggregation=VariableAggregation.NONE,
    shape=None,                  # Override shape (can be TensorShape(None))
    experimental_enable_variable_lifting=True,
)
```

#### `initial_value`

The initial value for the variable. Can be:
- A Python scalar, list, or NumPy array
- A `tf.Tensor`
- A callable (function) that returns a tensor
- If `None`, `variable_def` or `dtype` + `shape` must be provided

When a callable is provided, it is called once to get the initial value.
The callable is useful for:
- Expensive initializations (large random tensors)
- Initializations that depend on other variables
- Lazy initialization patterns

#### `trainable`

Controls whether `tf.GradientTape` automatically watches this variable:
- `True`: GradientTape watches by default
- `False`: GradientTape does NOT watch
- `None` (default): Set to `True` unless `synchronization=ON_READ`

```python
with tf.GradientTape(persistent=True) as tape:
    trainable_var = tf.Variable(1.0, trainable=True)
    non_trainable_var = tf.Variable(2.0, trainable=False)
    loss = trainable_var * 2 + non_trainable_var * 3

tape.gradient(loss, trainable_var)       # 2.0
tape.gradient(loss, non_trainable_var)   # None (not watched)
```

#### `validate_shape`

If `True` (default), the initial value must have a fully-defined shape.
Set to `False` to allow unknown shapes.

#### `constraint`

An optional projection function applied after each optimizer update.
The function takes the variable's value as input and returns the projected
value with the same shape.

```python
# Constrain weights to have unit norm
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=lambda x: tf.nn.l2_normalize(x, axis=0))

# Constrain to non-negative values
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=tf.nn.relu)

# Use a Keras constraint
from tf.keras.constraints import MaxNorm
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=MaxNorm(max_value=2.0))
```

#### `synchronization` and `aggregation`

These parameters control distributed training behavior:

**Synchronization** (`tf.VariableSynchronization`):
- `AUTO` (default): Determined by `DistributionStrategy`
- `NONE`: One copy, no synchronization
- `ON_WRITE`: Sync across devices on every write
- `ON_READ`: Sync across devices on every read

**Aggregation** (`tf.VariableAggregation`):
- `NONE` (default): Error if multiple replicas update
- `SUM`: Sum updates across replicas
- `MEAN`: Average updates across replicas
- `ONLY_FIRST_REPLICA`: Only update on the first replica

```python
# Example: moving average variable
moving_avg = tf.Variable(
    0.0,
    trainable=False,
    synchronization=tf.VariableSynchronization.ON_READ,
    aggregation=tf.VariableAggregation.MEAN
)
```

---

## Variable Methods

### Assignment Methods

#### `assign(value, use_locking=False, name=None, read_value=True)`

Assigns a new value to the variable.

```python
v = tf.Variable(1.0)
v.assign(2.0)          # Returns the variable with value 2.0
v.assign([1.0, 2.0])   # Assign a new shape (if shape is TensorShape(None))

# read_value=False returns the assign op (graph mode) or None (eager mode)
v.assign(3.0, read_value=False)
```

#### `assign_add(delta, use_locking=False, name=None, read_value=True)`

Adds a value to the current variable value.

```python
v = tf.Variable(1.0)
v.assign_add(0.5)      # v is now 1.5
v.assign_add(2.0)      # v is now 3.5
```

#### `assign_sub(delta, use_locking=False, name=None, read_value=True)`

Subtracts a value from the current variable value.

```python
v = tf.Variable(5.0)
v.assign_sub(1.0)      # v is now 4.0
```

### Reading Methods

#### `read_value()`

Returns the value of this variable as a `tf.Tensor`, read in the current
execution context. This may differ from `value()` when control dependencies
or device placement are involved.

```python
v = tf.Variable([1.0, 2.0, 3.0])
t = v.read_value()     # tf.Tensor([1.0, 2.0, 3.0])
```

#### `value()`

Returns the last snapshot of this variable. In eager mode, this is the
live value. In graph mode, this may be a cached copy.

```python
v = tf.Variable([1.0, 2.0, 3.0])
t = v.value()          # tf.Tensor with the variable's value
```

#### `numpy()`

Returns the variable's value as a NumPy array (eager mode only, through
tensor conversion).

```python
v = tf.Variable([1.0, 2.0, 3.0])
arr = v.numpy()        # array([1., 2., 3.], dtype=float32)
```

### Scatter Operations

#### `scatter_update(sparse_delta, use_locking=False, name=None)`

Assigns `tf.IndexedSlices` to the variable.

```python
v = tf.Variable(tf.zeros([8]))
indices = tf.constant([1, 3, 5, 7])
values = tf.constant([10.0, 20.0, 30.0, 40.0])
sparse_delta = tf.IndexedSlices(values, indices)
v.scatter_update(sparse_delta)
# v is now [0, 10, 0, 20, 0, 30, 0, 40]
```

#### `scatter_add(sparse_delta, use_locking=False, name=None)`

Adds `tf.IndexedSlices` to the variable.

```python
v = tf.Variable(tf.ones([8]))
sparse_delta = tf.IndexedSlices(tf.constant([1.0, 2.0, 3.0]),
                                 tf.constant([0, 2, 4]))
v.scatter_add(sparse_delta)
# v is now [2, 1, 3, 1, 4, 1, 1, 1]
```

#### `scatter_sub(sparse_delta, use_locking=False, name=None)`

Subtracts `tf.IndexedSlices` from the variable.

#### `scatter_max(sparse_delta, use_locking=False, name=None)`

Updates variable with element-wise maximum of itself and `tf.IndexedSlices`.

#### `scatter_min(sparse_delta, use_locking=False, name=None)`

Updates variable with element-wise minimum of itself and `tf.IndexedSlices`.

#### `scatter_mul(sparse_delta, use_locking=False, name=None)`

Multiplies the variable element-wise by `tf.IndexedSlices`.

#### `scatter_div(sparse_delta, use_locking=False, name=None)`

Divides the variable element-wise by `tf.IndexedSlices`.

### Scatter ND Operations

#### `scatter_nd_update(indices, updates, name=None)`

Applies sparse assignment to individual values or slices.

```python
v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
v.scatter_nd_update(indices, updates)
# v is now [1, 11, 3, 10, 9, 6, 7, 12]
```

#### `scatter_nd_add(indices, updates, name=None)`

Adds `updates` at the specified `indices`.

```python
v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
indices = tf.constant([[4], [3], [1], [7]])
updates = tf.constant([9, 10, 11, 12])
v.scatter_nd_add(indices, updates)
# v is now [1, 13, 3, 14, 14, 6, 7, 20]
```

#### `scatter_nd_sub(indices, updates, name=None)`

Subtracts `updates` at the specified `indices`.

### Gathering Methods

#### `sparse_read(indices, name=None)`

Gathers slices from the variable along axis 0 (equivalent to `tf.gather`
on the variable value).

```python
v = tf.Variable([10, 20, 30, 40, 50])
v.sparse_read([0, 2, 4])  # [10, 30, 50]
```

#### `gather_nd(indices, name=None)`

Gathers slices from the variable using multi-dimensional indices.

```python
v = tf.Variable([[1, 2], [3, 4], [5, 6]])
v.gather_nd([[0, 0], [2, 1]])  # [1, 6]
```

---

## Variable Properties

### Read-Only Properties

```python
v = tf.Variable(tf.random.normal([3, 4], dtype=tf.float32), name='weights')

# Name and device
v.name       # 'weights:0' (string)
v.device     # Device string (e.g., '/device:CPU:0')

# Shape and type
v.shape      # TensorShape([3, 4])
v.dtype      # tf.float32
v.ndim       # 2 (rank)

# Training properties
v.trainable  # True or False
v.synchronization  # VariableSynchronization enum
v.aggregation      # VariableAggregation enum
v.constraint       # Constraint function or None

# Graph-related (graph mode)
v.op         # The Operation producing this variable
v.graph      # The Graph containing this variable
v.initializer  # The initializer Operation

# Trackable
v.initial_value  # The initial value tensor
```

### Equality and Hashing

Variables use element-wise equality in eager mode and are unhashable:

```python
v1 = tf.Variable([1.0, 2.0])
v2 = tf.Variable([1.0, 2.0])

v1 == v2  # tf.Tensor([True, True])  (element-wise)

# Variables are unhashable in TF2:
# my_set = {v1, v2}  # TypeError
# Use v.ref() instead:
my_set = {v1.ref(), v2.ref()}
print(v1.ref() in my_set)  # True
print(v1.ref().deref())    # <tf.Variable ... numpy=array([1., 2.])>
```

### Operator Overloading

Variables support all the same operators as `tf.Tensor` (except `==` and `!=`
which use element-wise comparison):

```python
v = tf.Variable([1.0, 2.0, 3.0])

# Arithmetic
v + 1       # tf.Tensor([2.0, 3.0, 4.0])
v * 2       # tf.Tensor([2.0, 4.0, 6.0])
-v          # tf.Tensor([-1.0, -2.0, -3.0])

# Indexing
v[0]        # tf.Tensor(1.0)
v[1:]       # tf.Tensor([2.0, 3.0])

# Matrix operations
m = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
m @ tf.constant([[1.0], [2.0]])  # Matrix multiply
```

Note: These operations create new tensors; they do NOT modify the variable.
To modify the variable, use `assign`, `assign_add`, etc.

---

## ResourceVariable vs RefVariable

### ResourceVariable (TF2 Default)

In TensorFlow 2.x, all variables are `ResourceVariable` instances. This
implementation uses a resource handle (opaque token) to identify the
variable's storage.

**Advantages**:
- Atomic read-modify-write operations (thread-safe)
- Faster for small variables
- Supports copying between devices
- Better behavior in concurrent settings
- Required for TPU

```python
from tensorflow.python.ops import resource_variable_ops

# This is what tf.Variable() creates in TF2
v = resource_variable_ops.ResourceVariable(
    initial_value=1.0,
    trainable=True,
    name='my_var'
)
```

Key implementation details from `resource_variable_ops.py`:
- Uses `VarHandleOp` to create a resource handle
- Uses `ReadVariableOp` to read the value
- Uses `AssignVariableOp` / `AssignAddVariableOp` for updates
- The resource handle is a `tf.Tensor` of dtype `tf.resource`
- Handle data stores shape and dtype metadata

### RefVariable (TF1 Legacy)

RefVariable was the variable implementation in TF1. It uses a mutable
reference tensor that can be updated in-place.

**Characteristics**:
- Mutable tensor reference (value can change in-place)
- Non-atomic updates (potential race conditions)
- Used in TF1 graph mode
- Still available via `tf.compat.v1.Variable`

```python
# TF1 style (deprecated)
import tensorflow.compat.v1 as tf
v = tf.Variable(1.0, use_resource=False)  # Creates RefVariable
```

### Key Differences

| Feature | ResourceVariable | RefVariable |
|---------|-----------------|-------------|
| TF Version | TF2 default | TF1 legacy |
| Handle Type | Resource handle (opaque) | Mutable tensor reference |
| Thread Safety | Atomic operations | Non-atomic |
| Copy Between Devices | Supported | Limited |
| TPU Compatible | Yes | No |
| Graph Mode | Works in both | Graph only |
| Eager Mode | Works natively | Not supported |

---

## tf.Module

### Overview

`tf.Module` is the base class for stateful components in TensorFlow. It
provides automatic variable tracking, naming, and serialization support.

Defined in `tensorflow/python/module/module.py`:

```python
@tf_export("Module")
class Module(autotrackable.AutoTrackable):
```

### Basic Usage

```python
class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]),
            name='w'
        )
        self.b = tf.Variable(
            tf.zeros([out_features]),
            name='b'
        )

    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b

# Create and use
layer = Dense(3, 2)
output = layer(tf.constant([[1.0, 2.0, 3.0]]))
```

### Variable Tracking

`tf.Module` automatically tracks variables assigned as attributes:

```python
class MyModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.w1 = tf.Variable(tf.random.normal([10, 20]))  # Tracked
        self.b1 = tf.Variable(tf.zeros([20]))               # Tracked
        self.sub_module = Dense(20, 10)                     # Tracked (nested)
        self._private_var = tf.Variable(0.0)                # Tracked
        self.loss = 0.0                                     # NOT tracked (not a Variable)

model = MyModel()
model.variables          # (w1, b1, sub_module's w, sub_module's b)
model.trainable_variables  # (w1, b1, sub_module's w, sub_module's b)
model.submodules          # (Dense instance,)
```

### Properties

#### `name`

The name of this module as passed or determined in the constructor.
If no name is provided, it uses the snake_case version of the class name.

```python
m = tf.Module()
print(m.name)  # 'module'

class MyDenseLayer(tf.Module):
    pass

layer = MyDenseLayer()
print(layer.name)  # 'my_dense_layer'

layer2 = MyDenseLayer(name='custom_name')
print(layer2.name)  # 'custom_name'
```

#### `name_scope`

Returns a `tf.name_scope` for grouping operations:

```python
class Model(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(1.0)

    @tf.Module.with_name_scope
    def __call__(self, x):
        return x * self.w
```

#### `variables`

Returns a tuple of all variables owned by this module and its submodules,
sorted by attribute name, then recursively by submodule (breadth-first).

```python
model.variables
# (<tf.Variable 'model/w:0' ...>, <tf.Variable 'model/b:0' ...>, ...)
```

#### `trainable_variables`

Returns only the trainable variables:

```python
model.trainable_variables
# Only variables with trainable=True
```

#### `non_trainable_variables`

Returns only the non-trainable variables:

```python
model.non_trainable_variables
# Only variables with trainable=False
```

#### `submodules`

Returns a tuple of all sub-modules (modules that are properties of this
module, recursively):

```python
model.submodules
# (SubModule1, SubModule2, ...)
```

### Class Methods

#### `with_name_scope(method)`

A decorator that applies the module's name scope to the method:

```python
class MyModule(tf.Module):
    @tf.Module.with_name_scope
    def process(self, x):
        # Operations created here are under the module's name scope
        return tf.matmul(x, self.w)
```

### Tracking Mechanism

`tf.Module` inherits from `AutoTrackable`, which overrides `__setattr__`
to track tf.Variable and tf.Module instances:

```python
class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(1.0)     # Tracked automatically
        self.loss_val = 0.0            # Not tracked (not a Variable/Module)

    def add_layer(self):
        self.layer = Dense(3, 2)       # Tracked when assigned later
```

The `_flatten` method recursively finds all tracked attributes:

```python
# Internal method
module._flatten(predicate=_is_variable, expand_composites=True)
module._flatten(predicate=_is_trainable_variable, expand_composites=True)
module._flatten(predicate=_is_module)
```

---

## tf.TensorSpec

### Overview

`tf.TensorSpec` describes the type of a `tf.Tensor`. It is used for:
- Input signatures of `tf.function`
- Specifying expected tensor shapes and dtypes
- Type checking and casting

Defined in `tensorflow/python/framework/tensor.py`:

```python
@tf_export("TensorSpec")
class TensorSpec(DenseSpec, type_spec.BatchableTypeSpec,
                 trace_type.Serializable, internal.TensorSpec):
```

### Creation

```python
# Basic creation
spec = tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name='input')

# From a tensor
t = tf.constant([1.0, 2.0, 3.0])
spec = tf.TensorSpec.from_tensor(t)
# TensorSpec(shape=(3,), dtype=tf.float32, name=None)

# From another spec
spec2 = tf.TensorSpec.from_spec(spec, name='new_name')
```

### Properties

```python
spec = tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name='input')

spec.shape   # TensorShape([None, 3])
spec.dtype   # tf.float32
spec.name    # 'input'
```

### Methods

```python
# Compatibility check
spec.is_compatible_with(tf.constant([[1.0, 2.0, 3.0]]))  # True
spec.is_compatible_with(tf.constant([1.0, 2.0]))          # False

# Subtype check
spec2 = tf.TensorSpec([2, 3], tf.float32)
spec.is_subtype_of(spec2)  # Depends on shape compatibility

# Serialization
spec.experimental_as_proto()  # TensorSpecProto
spec2 = tf.TensorSpec.experimental_from_proto(proto)
```

### Usage in tf.function

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    tf.TensorSpec(shape=[None, 10], dtype=tf.float32),
])
def train_step(images, labels):
    # Function body
    pass
```

---

## Device Placement for Variables

### Default Placement

Variables are placed on the default device (typically CPU or the first GPU):

```python
v = tf.Variable(tf.zeros([100, 100]))
print(v.device)  # /device:CPU:0 or /device:GPU:0
```

### Explicit Device Placement

```python
# Place on specific GPU
with tf.device('/GPU:0'):
    v = tf.Variable(tf.zeros([100, 100]))
    print(v.device)  # /device:GPU:0

# Place on CPU
with tf.device('/CPU:0'):
    v = tf.Variable(tf.zeros([100, 100]))
    print(v.device)  # /device:CPU:0

# Place on remote device (distributed)
with tf.device('/job:worker/task:0/device:CPU:0'):
    v = tf.Variable(tf.zeros([100, 100]))
```

### Distribution Strategy Placement

When using `tf.distribute.Strategy`, variables are automatically placed
and replicated:

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    v = tf.Variable(1.0)
    # v is replicated across all GPUs
```

### Soft Placement

If a variable cannot be placed on the requested device, TensorFlow falls
back to another device:

```python
tf.config.set_soft_device_placement(True)  # Default in TF2
```

---

## Variable Initialization Strategies

### Eager Mode (TF2 Default)

In eager mode, variables are initialized immediately upon creation:

```python
v = tf.Variable(tf.random.normal([100, 100]))
# Variable is ready to use immediately
print(v.numpy())  # Works right away
```

### Deferred Initialization

For large variables or when the shape depends on runtime values:

```python
# Callable initializer
v = tf.Variable(lambda: tf.random.normal([10000, 10000]))

# Uninitialized variable with shape override
v = tf.Variable(tf.zeros([1]), shape=tf.TensorShape(None))
```

### Common Initializers

```python
# Zeros and Ones
v = tf.Variable(tf.zeros([100, 100]))
v = tf.Variable(tf.ones([100, 100]))

# Random normal
v = tf.Variable(tf.random.normal([100, 100], stddev=0.01))

# Glorot/Xavier uniform
v = tf.Variable(tf.initializers.glorot_uniform()([100, 100]))

# He normal
v = tf.Variable(tf.initializers.he_normal()([100, 100]))

# Orthogonal
v = tf.Variable(tf.initializers.orthogonal()([100, 100]))

# Constant
v = tf.Variable(tf.initializers.constant(0.1)([100, 100]))
```

---

## Partitioned Variables

### Overview

`PartitionedVariable` (defined in `variables.py`) wraps multiple `Variable`
objects that together form a single logical variable. This is used for
very large variables that need to be split across devices or parameter
servers.

```python
class PartitionedVariable:
    def __init__(self, name, shape, dtype, variable_list, partitions):
```

### Usage

```python
# Create partitioned variable (TF1/parameter server pattern)
partitioned = tf.compat.v1.variable_axis_size_partitioner(1024 * 1024)
with tf.compat.v1.variable_scope('', partitioner=partitioned):
    w = tf.compat.v1.get_variable('weights', [10000, 10000])
```

### Properties and Methods

```python
pv.name          # Overall name
pv.dtype         # Data type
pv.shape         # Overall shape
pv.get_shape()   # Same as shape

# Iteration over partitions
for var in pv:
    print(var.shape)

# Assignment (splits value across partitions)
pv.assign(tf.zeros([10000, 10000]))
pv.assign_add(delta)
pv.assign_sub(delta)

# Convert to single tensor
tensor = pv.as_tensor()
```

### Partition Axes

Partitioning is typically along one axis:

```python
# Partition axis 0 into 4 parts
# Variable [10000, 100] -> 4 x [2500, 100]
```

---

## Variable Aggregation in Distributed Settings

### VariableSynchronization Enum

```python
class VariableSynchronization(enum.Enum):
    AUTO = 0       # Determined by DistributionStrategy
    NONE = 1       # One copy, no sync
    ON_WRITE = 2   # Sync on every write
    ON_READ = 3    # Sync on every read
```

### VariableAggregation Enum

```python
class VariableAggregationV2(enum.Enum):
    NONE = 0               # Error if multiple replicas update
    SUM = 1                # Sum across replicas
    MEAN = 2               # Average across replicas
    ONLY_FIRST_REPLICA = 3 # Only first replica's value used
```

### Distribution Strategy Example

```python
strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1'])

with strategy.scope():
    # Synced variable (ON_WRITE by default)
    v = tf.Variable(1.0, aggregation=tf.VariableAggregation.MEAN)

    # Moving average variable (synced ON_READ)
    ema = tf.Variable(
        0.0,
        trainable=False,
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.MEAN
    )

@tf.function
def update():
    v.assign_add(1.0)
    return v

strategy.run(update)
# Both replicas see v = 2.0 (1.0 + 1.0 summed, but MEAN aggregation -> 2.0)
```

---

## Constraint Functions on Variables

### Purpose

Constraints are projection functions applied to the variable value after
each optimizer update. They enforce invariants like non-negativity,
bounded norms, or specific value ranges.

### Built-in Constraints

```python
from tensorflow.keras import constraints

# Non-negativity
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=constraints.NonNeg())

# Maximum norm per row/column
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=constraints.MaxNorm(max_value=2.0))

# Unit norm
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=constraints.UnitNorm(axis=0))

# Min-Max
v = tf.Variable(tf.random.normal([3, 3]),
                constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0))
```

### Custom Constraint Functions

```python
# Lambda function
v = tf.Variable(
    tf.random.normal([3, 3]),
    constraint=lambda x: tf.clip_by_norm(x, 1.0)
)

# Named function
def unit_norm_constraint(w):
    return w / tf.norm(w)

v = tf.Variable(
    tf.random.normal([3, 3]),
    constraint=unit_norm_constraint
)

# Callable class
class L2BallConstraint:
    def __init__(self, radius=1.0):
        self.radius = radius

    def __call__(self, w):
        norm = tf.norm(w)
        return tf.cond(
            norm > self.radius,
            lambda: w * (self.radius / norm),
            lambda: w
        )

v = tf.Variable(tf.random.normal([100]), constraint=L2BallConstraint(1.0))
```

### When Constraints Are Applied

Constraints are applied automatically by optimizers after the gradient
update step:

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Constraints are applied here for each variable that has one
```

---

## Custom Variable Classes

### Subclassing tf.Variable

```python
class ExponentialMovingAverageVariable(tf.Variable):
    """A variable that maintains an exponential moving average."""

    def __init__(self, initial_value, decay=0.99, **kwargs):
        self._decay = decay
        super().__init__(initial_value, **kwargs)

    def update(self, new_value):
        """Update the EMA with a new value."""
        self.assign(self._decay * self.read_value() +
                    (1 - self._decay) * new_value)
```

### VariableMetaclass

The `VariableMetaclass` allows overriding variable construction:

```python
class VariableMetaclass(abc.ABCMeta):
    @traceback_utils.filter_traceback
    def __call__(cls, *args, **kwargs):
        if hasattr(cls, '_variable_call') and callable(cls._variable_call):
            variable_call = cls._variable_call(*args, **kwargs)
            if variable_call is not None:
                return variable_call
        return super().__call__(*args, **kwargs)
```

This enables custom variable creation through `_variable_call`:

```python
class MyVariable(tf.Variable):
    @classmethod
    def _variable_call(cls, *args, **kwargs):
        # Custom creation logic
        return custom_variable_creator(*args, **kwargs)
```

---

## Resource Management and Cleanup

### Variable Lifecycle

In eager mode, variables are garbage-collected when there are no more
references:

```python
def create_temp_variable():
    v = tf.Variable(1.0)
    return v.ref()  # Return reference to prevent GC

ref = create_temp_variable()
# Variable still alive via ref
del ref  # Now variable can be garbage collected
```

### Resource Handle Management

ResourceVariable uses resource handles (opaque tokens) backed by C++
objects. The handle lifecycle is managed through:

1. **VarHandleOp**: Creates the resource handle
2. **Resource allocation**: Backed by a C++ resource manager
3. **Reference counting**: Python wrapper holds a reference
4. **Garbage collection**: When the Python object is collected, the C++
   resource is freed

```python
# Resource variable handles
v = tf.Variable(1.0)
handle = v.handle  # tf.Tensor of dtype tf.resource
print(handle.dtype)  # tf.resource
```

### Memory Management

```python
# Delete a large variable to free memory
v = tf.Variable(tf.zeros([10000, 10000]))
del v  # Frees the underlying buffer

# Clear GPU memory
tf.keras.backend.clear_session()
```

---

## Variable Saving and Restoration

### tf.train.Checkpoint

The primary mechanism for saving and restoring variables in TF2:

```python
# Create variables
model = MyModel()
optimizer = tf.keras.optimizers.Adam(0.001)

# Create checkpoint
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Save
save_path = checkpoint.save('/tmp/checkpoints/ckpt')
print(save_path)  # /tmp/checkpoints/ckpt-1

# Restore
checkpoint.restore('/tmp/checkpoints/ckpt-1')

# Restore with expect_partial (suppress warnings for partial restore)
checkpoint.restore('/tmp/checkpoints/ckpt-1').expect_partial()
```

### CheckpointManager

Manage multiple checkpoints:

```python
manager = tf.train.CheckpointManager(
    checkpoint,
    directory='/tmp/checkpoints',
    max_to_keep=3
)

# Save
save_path = manager.save()
print(f'Saved: {save_path}')

# List checkpoints
print(manager.checkpoints)  # ['/tmp/checkpoints/ckpt-1', ...]

# Restore latest
manager.restore_or_initialize()
```

### SavedModel Integration

Variables are saved as part of a SavedModel:

```python
# Save with variables
tf.saved_model.save(model, '/tmp/saved_model')

# Load with variables
loaded = tf.saved_model.load('/tmp/saved_model')
```

### Variable Collections (TF1)

In TF1, variables are managed through graph collections:

```python
# TF1 collections (deprecated)
tf.compat.v1.global_variables()      # All global variables
tf.compat.v1.local_variables()       # All local variables
tf.compat.v1.trainable_variables()   # All trainable variables
tf.compat.v1.model_variables()       # All model variables
```

### Variable Scopes (TF1 Compatibility)

```python
# TF1 variable scopes (deprecated in TF2)
import tensorflow.compat.v1 as tf

with tf.variable_scope('layer1'):
    w = tf.get_variable('weights', [10, 20])
    b = tf.get_variable('bias', [20])

# Variables created as 'layer1/weights:0', 'layer1/bias:0'
```

### SaveSliceInfo

Internal mechanism for saving partitioned variables:

```python
class Variable.SaveSliceInfo:
    # Properties
    full_name    # Name of the full variable
    full_shape   # Shape of the full variable
    var_offset   # Offset of this slice
    var_shape    # Shape of this slice

    # Methods
    spec         # Spec string for saving
    to_proto()   # Convert to SaveSliceInfoDef
```

---

## Summary

TensorFlow's variable system is built around:

- **tf.Variable**: The primary interface for mutable state, providing assign,
  scatter, and gather operations
- **ResourceVariable**: The TF2 implementation using resource handles for
  thread-safe, device-portable variable management
- **tf.Module**: Object-oriented variable tracking that replaces TF1's
  name-based collections and scopes
- **tf.TensorSpec**: Type specification for tensors used in function
  signatures and serialization
- **Distributed variable support**: Synchronization and aggregation options
  for multi-device training
- **Constraint functions**: Post-update projections for enforcing invariants
  on variable values
- **Checkpoint/SavedModel**: Variable persistence through `tf.train.Checkpoint`
  and `tf.saved_model`

The TF2 variable system emphasizes object-oriented design over TF1's
string-based name scopes, providing cleaner code organization, better
type safety, and easier debugging.
