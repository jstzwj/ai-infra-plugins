# TensorFlow Automatic Differentiation Reference

## Table of Contents

1. [tf.GradientTape](#tfgradienttape)
2. [Forward-Mode Autodiff](#forward-mode-autodiff)
3. [Custom Gradients](#custom-gradients)
4. [tf.function](#tffunction)
5. [AutoGraph](#autograph)
6. [tf.function Limitations and Gotchas](#tffunction-limitations-and-gotchas)
7. [Python Side Effects in tf.function](#python-side-effects-in-tffunction)
8. [Variable Creation in tf.function](#variable-creation-in-tffunction)
9. [Retracing and Performance](#retracing-and-performance)
10. [tf.TensorSpec and Function Signatures](#tftensorspec-and-function-signatures)
11. [Composite Tensors in Functions](#composite-tensors-in-functions)
12. [Function Serialization and SavedModel](#function-serialization-and-savedmodel)

---

## tf.GradientTape

### Overview

`tf.GradientTape` is TensorFlow 2.x's primary API for automatic
differentiation. It records operations on tensors inside its context
manager and computes gradients using reverse-mode automatic
differentiation (backpropagation).

The tape implementation is split between Python and C++:
- `tensorflow/python/eager/tape.py`: Python `Tape` wrapper
- `tensorflow/python/eager/backprop.py`: `GradientTape` class
- C++ tape implementation via `pywrap_tfe`

### Basic Usage

```python
import tensorflow as tf

# Simple gradient computation
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2

dy_dx = tape.gradient(y, x)
print(dy_dx)  # tf.Tensor(6.0, shape=(), dtype=float32)  (dy/dx = 2x = 6)
```

### Watching Variables

`GradientTape` automatically watches trainable `tf.Variable` objects:

```python
# Variables are watched automatically
w = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = w ** 2 + w * 3

dy_dw = tape.gradient(y, w)
print(dy_dw)  # 7.0  (dy/dw = 2w + 3 = 7)
```

Regular tensors are NOT watched automatically; you must explicitly call
`tape.watch()`:

```python
# Tensors must be watched explicitly
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)           # Must call this for non-Variable tensors
    y = x ** 3

dy_dx = tape.gradient(y, x)
print(dy_dx)  # 27.0  (dy/dx = 3x^2 = 27)
```

### Multiple Gradients

Compute gradients with respect to multiple variables:

```python
w1 = tf.Variable(2.0)
w2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = w1 ** 2 + w2 ** 3

grads = tape.gradient(y, [w1, w2])
print(grads[0])  # 4.0  (dy/dw1 = 2w1 = 4)
print(grads[1])  # 27.0 (dy/dw2 = 3w2^2 = 27)

# Dictionary form
grads = tape.gradient(y, {'w1': w1, 'w2': w2})
print(grads['w1'])  # 4.0
print(grads['w2'])  # 27.0
```

### Higher-Order Gradients

Stack gradient tapes for second-order derivatives:

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape1:
    tape1.watch(x)
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y = x ** 4  # y = x^4

    dy_dx = tape2.gradient(y, x)   # dy/dx = 4x^3
    print(dy_dx)  # 108.0

d2y_dx2 = tape1.gradient(dy_dx, x)  # d2y/dx2 = 12x^2
print(d2y_dx2)  # 108.0
```

### Persistent Tapes

By default, a `GradientTape` releases its resources after `gradient()`
is called once. For multiple gradient calls, use `persistent=True`:

```python
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x ** 2
    z = x ** 3

dy_dx = tape.gradient(y, x)  # 6.0
dz_dx = tape.gradient(z, x)  # 27.0

# Must manually delete persistent tapes to free resources
del tape
```

### Controlling Watched Variables

```python
# Disable automatic variable watching
with tf.GradientTape(watch_accessed_variables=False) as tape:
    w1 = tf.Variable(2.0)
    w2 = tf.Variable(3.0)

    tape.watch(w1)  # Only watch w1
    y = w1 * w2

dy_dw1 = tape.gradient(y, w1)  # 3.0
dy_dw2 = tape.gradient(y, w2)  # None (not watched)
```

### Gradient with Multiple Targets

```python
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    z = x ** 3

# Gradient of sum of targets
dy_dx = tape.gradient([y, z], x)  # Sum of gradients: 4 + 12 = 16
```

### Gradient with None Results

Gradients may be `None` for several reasons:

1. **Tensor not watched**: The tensor was not a `tf.Variable` and was not
   explicitly watched.
2. **Operation has no gradient**: Some operations (e.g., string operations,
   integer operations) do not have registered gradients.
3. **Disconnected graph**: The target does not depend on the source.

```python
x = tf.constant(2.0)
y = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    z = y * 2  # z does not depend on x

dz_dx = tape.gradient(z, x)  # None (disconnected)

# Control behavior with unconnected_gradients
dz_dx = tape.gradient(z, x,
    unconnected_gradients=tf.UnconnectedGradients.ZERO)  # 0.0
```

### Gradient with IndexedSlices

For large embedding lookups, gradients may be returned as `tf.IndexedSlices`
instead of dense tensors:

```python
embedding = tf.Variable(tf.random.normal([1000, 64]))
indices = tf.constant([0, 5, 10, 15])

with tf.GradientTape() as tape:
    selected = tf.gather(embedding, indices)
    loss = tf.reduce_sum(selected)

grad = tape.gradient(loss, embedding)
print(type(grad))  # <class 'tensorflow.python.framework.indexed_slices.IndexedSlices'>
print(grad.indices)  # [0, 5, 10, 15]
print(grad.values.shape)  # (4, 64)
```

### Jacobian Computation

```python
# tf.GradientTape.jacobian - computes jacobian for each source element
x = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2

jacobian = tape.jacobian(y, x)
print(jacobian)  # [[2., 0., 0.], [0., 4., 0.], [0., 0., 6.]]
# Diagonal because y[i] only depends on x[i]

# tf.GradientTape.batch_jacobian - efficient when outputs depend only on
# their corresponding inputs
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x

batch_jacobian = tape.batch_jacobian(y, x)
print(batch_jacobian)  # [[[2., 0.], [0., 4.]], [[6., 0.], [0., 8.]]]
```

### Gradient Tape Training Loop

```python
# Typical training loop with GradientTape
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        # Add regularization losses
        loss += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

---

## Forward-Mode Autodiff

### tf.autodiff.ForwardAccumulator

TensorFlow also supports forward-mode automatic differentiation via
`ForwardAccumulator`. This is useful when the number of outputs is much
larger than the number of inputs (opposite case from reverse-mode).

```python
# Forward-mode: compute directional derivative
x = tf.constant([1.0, 2.0, 3.0])

# Tangent vector (direction of differentiation)
tangent = tf.constant([1.0, 0.0, 0.0])  # Differentiate w.r.t. x[0]

with tf.autodiff.ForwardAccumulator(
    primals=x,
    tangents=tangent
) as acc:
    y = x ** 2

# Result is the directional derivative
print(acc.jvp(y))  # [2.0, 0.0, 0.0]  (dy/dx[0] * tangent)
```

### When to Use Forward-Mode

Forward-mode is more efficient when:
- Number of inputs << number of outputs (Jacobian has few columns)
- Computing directional derivatives
- Computing Jacobian-vector products (JVP) efficiently

Reverse-mode (GradientTape) is more efficient when:
- Number of outputs << number of inputs (typical for neural networks)
- Computing vector-Jacobian products (VJP) efficiently

---

## Custom Gradients

### tf.custom_gradient Decorator

The `tf.custom_gradient` decorator (defined in `tensorflow/python/ops/custom_gradient.py`)
allows defining custom gradient functions for operations:

```python
@tf.custom_gradient
def clip_gradient_if_nan(x):
    """Clips gradients to 1.0 when NaN values are detected."""
    def grad(upstream):
        return tf.where(tf.math.is_nan(upstream),
                        tf.zeros_like(upstream),
                        upstream)
    return x, grad

# Usage
x = tf.constant(1.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = clip_gradient_if_nan(x)

dy_dx = tape.gradient(y, x)
```

### Custom Gradient with Multiple Inputs

```python
@tf.custom_gradient
def multiply_with_custom_grad(x, y):
    """Custom multiplication with scaled gradient."""
    result = x * y

    def grad(upstream):
        # Return gradient for each input
        scale = 0.5  # Scale gradients
        return upstream * y * scale, upstream * x * scale

    return result, grad

# Usage
x = tf.constant(2.0)
y = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch([x, y])
    z = multiply_with_custom_grad(x, y)

grads = tape.gradient(z, [x, y])
print(grads[0])  # 1.5 (3.0 * 0.5)
print(grads[1])  # 1.0 (2.0 * 0.5)
```

### Custom Gradient for Numerical Stability

```python
@tf.custom_gradient
def log1p_exp(x):
    """Numerically stable log(1 + exp(x))."""
    result = tf.math.log1p(tf.exp(x))

    def grad(upstream):
        # Stable sigmoid gradient: 1 / (1 + exp(-x))
        return upstream * tf.sigmoid(x)

    return result, grad
```

### Custom Gradient with Variables

```python
@tf.custom_gradient
def scale_by_var(x, scale_var):
    result = x * scale_var

    def grad(upstream):
        # upstream: gradient from above
        # grad w.r.t. x: upstream * scale_var
        # grad w.r.t. scale_var: upstream * x (if trainable)
        return upstream * scale_var, upstream * x

    return result, grad
```

### Gradient Registry

TensorFlow maintains a gradient registry that maps operation types to
their gradient functions:

```python
# Register a gradient function for a custom op
@ops.RegisterGradient("MyCustomOp")
def _my_custom_op_grad(op, grad):
    # op: the forward operation
    # grad: gradient from downstream
    return grad * tf.constant(2.0)
```

### Gradient Checking

```python
# Verify custom gradients numerically
from tensorflow.python.ops import gradient_checker_v2

def test_gradient():
    def f(x):
        return x ** 3

    # Check analytical vs numerical gradient
    result = gradient_checker_v2.compute_gradient(f, [tf.constant(2.0)])
    print(result)  # (analytical_grad, numerical_grad)
```

---

## tf.function

### Overview

`tf.function` compiles a Python function into a TensorFlow graph for
optimized execution. It is the primary mechanism for achieving graph-mode
performance in TF2.

Defined in `tensorflow/python/eager/polymorphic_function/polymorphic_function.py`:

```python
@tf_export("function")
def function(
    func=None,
    input_signature=None,
    autograph=True,
    jit_compile=None,
    experimental_relax_shapes=False,
    experimental_follow_type_hints=True,
):
```

### Basic Usage

```python
# Decorator form
@tf.function
def add(a, b):
    return a + b

result = add(tf.constant(1.0), tf.constant(2.0))  # 3.0

# Callable form
def multiply(a, b):
    return a * b

tf_multiply = tf.function(multiply)
result = tf_multiply(tf.constant(3.0), tf.constant(4.0))  # 12.0
```

### Tracing Process

When a `tf.function` is called:

1. **Signature matching**: TF checks if a `ConcreteFunction` exists for the
   input types (dtype, shape, and object type).
2. **Tracing** (if no matching concrete function):
   a. A new `FuncGraph` is created
   b. Placeholder tensors are created for each input
   c. The Python function is called with these placeholders
   d. All TF operations are recorded in the graph
   e. The graph is compiled and optimized
   f. The `ConcreteFunction` is cached
3. **Execution**: The cached concrete function runs with actual input values.

```python
@tf.function
def my_function(x):
    print("Tracing!")  # Only prints during tracing
    return x * 2

# First call with int32 triggers tracing
my_function(tf.constant(1))   # Prints: "Tracing!"

# Second call with int32 uses cached graph
my_function(tf.constant(2))   # No print

# Different dtype triggers new trace
my_function(tf.constant(1.0))  # Prints: "Tracing!"
```

### Concrete Functions

A `ConcreteFunction` is a graph-compiled function bound to specific input
types:

```python
@tf.function
def process(x):
    return x + 1

# Get concrete function for a specific input signature
cf = process.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))

# Inspect the concrete function
print(cf.graph)            # The tf.Graph
print(cf.inputs)           # Input tensors
print(cf.outputs)          # Output tensors
print(cf.structured_input_signature)
print(cf.structured_outputs)

# Call directly
result = cf(tf.constant([[1.0, 2.0, 3.0]]))
```

### Polymorphic Functions

A `tf.function` creates a new `ConcreteFunction` for each unique input
signature. The collection of concrete functions forms a "polymorphic function":

```python
@tf.function
def process(x):
    return x * 2

# Different input signatures create different concrete functions
process(tf.constant([1, 2, 3]))        # int32, shape (3,)
process(tf.constant([1.0, 2.0, 3.0]))  # float32, shape (3,)
process(tf.constant([[1.0]]))          # float32, shape (1, 1)
```

### Input Signatures

Specify expected input types to control tracing behavior:

```python
# Fixed input signature
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32)
])
def train_step(images, step):
    return tf.reduce_mean(images) + tf.cast(step, tf.float32)

# This works
train_step(tf.zeros([32, 784]), tf.constant(0))

# This raises an error (incompatible shape)
# train_step(tf.zeros([32, 100]), tf.constant(0))

# This raises an error (incompatible dtype)
# train_step(tf.zeros([32, 784], tf.int32), tf.constant(0))
```

### jit_compile Option

```python
# Force XLA compilation
@tf.function(jit_compile=True)
def xla_matmul(a, b):
    return tf.matmul(a, b)

# XLA compilation provides:
# - Operator fusion
# - Memory optimization
# - Better TPU performance
# - Strict shape/dtype requirements
```

### experimental_relax_shapes

Allow retracing for different shapes within the same polymorphic function:

```python
# Without relax_shapes: different shapes trigger retracing
@tf.function
def process(x):
    return x + 1

process(tf.zeros([2]))    # Trace 1
process(tf.zeros([3]))    # Trace 2
process(tf.zeros([4]))    # Trace 3

# With relax_shapes: similar shapes share a trace
@tf.function(experimental_relax_shapes=True)
def process_relaxed(x):
    return x + 1

process_relaxed(tf.zeros([2]))  # Trace 1
process_relaxed(tf.zeros([3]))  # Reuses Trace 1 (same rank, different size)
process_relaxed(tf.zeros([4]))  # Reuses Trace 1
```

---

## AutoGraph

### Overview

AutoGraph converts Python code into TensorFlow graph operations. It is
automatically applied to functions decorated with `tf.function` (when
`autograph=True`, which is the default).

Defined in `tensorflow/python/autograph/`.

### Control Flow Conversion

#### if Statements

AutoGraph converts Python `if` statements to `tf.cond`:

```python
@tf.function
def conditional_multiply(x, flag):
    if flag:
        return x * 2
    else:
        return x * 3

# flag must be a tensor for AutoGraph conversion
result = conditional_multiply(tf.constant(5.0), tf.constant(True))
```

Rules:
- If the condition is a `tf.Tensor`, it is converted to `tf.cond`
- If the condition is a Python boolean, it is evaluated at trace time
- Both branches must return compatible types and shapes

#### while Loops

AutoGraph converts Python `while` loops to `tf.while_loop`:

```python
@tf.function
def fibonacci(n):
    a = tf.constant(0)
    b = tf.constant(1)
    i = tf.constant(0)

    while i < n:
        a, b = b, a + b
        i += 1

    return a

result = fibonacci(tf.constant(10))
```

Rules:
- Loop condition must depend on a tensor for dynamic unrolling
- Variables modified in the loop must have consistent shapes across iterations
- `shape_invariants` can be specified for varying shapes

#### for Loops

AutoGraph converts Python `for` loops:

```python
@tf.function
def sum_range(n):
    total = tf.constant(0)
    for i in tf.range(n):          # Tensor iteration -> tf.while_loop
        total += i
    return total

@tf.function
def process_list(items):
    results = []
    for item in items:              # Python list iteration (traced)
        results.append(item * 2)
    return results
```

Types of `for` loop conversion:
- Iterating over `tf.range()` or tensors: converted to `tf.while_loop`
- Iterating over Python lists: unrolled during tracing
- Iterating over `tf.data.Dataset`: uses dataset iteration

### AutoGraph Limitations

1. **Source code access**: AutoGraph needs access to the Python source code.
   Functions defined in the REPL or dynamically may not work.

2. **Closures**: Variables from outer scopes must be captured correctly.

3. **Global state**: Global variables are evaluated at trace time.

4. **Class methods**: `self` attribute access is traced.

5. **Recursive functions**: Limited support for recursion.

6. **Generator functions**: Not supported.

7. **Exception handling**: Limited support for try/except.

### AutoGraph Conversion Control

```python
# Disable AutoGraph for a specific function
@tf.function(autograph=False)
def no_autograph(x):
    # Must use TF control flow operations directly
    return tf.cond(x > 0, lambda: x * 2, lambda: x * 3)

# Convert a function without @tf.function
from tensorflow.python.autograph import to_graph

def my_python_function(x):
    if x > 0:
        return x * 2
    return x * 3

converted = to_graph(my_python_function)
# converted() is now a graph-compatible function
```

---

## tf.function Limitations and Gotchas

### 1. Python Side Effects Execute Only During Tracing

```python
@tf.function
def buggy_function(x):
    print("This prints during tracing, not every call!")
    tf.print("This prints every call!")
    return x + 1

buggy_function(tf.constant(1))  # Prints both messages
buggy_function(tf.constant(2))  # Only prints tf.print message
```

### 2. Python Collections Are Captured at Trace Time

```python
state = []

@tf.function
def append_to_state(x):
    state.append(x.numpy())  # WRONG: Only executes during tracing
    return x

append_to_state(tf.constant(1))
print(state)  # [1]

append_to_state(tf.constant(2))
print(state)  # Still [1] - the append was captured, not re-executed
```

### 3. Python Random Is Deterministic

```python
import random

@tf.function
def random_values():
    return random.randint(0, 100)  # Returns the SAME value each call!

# Use tf.random instead
@tf.function
def proper_random():
    return tf.random.uniform([], 0, 100, dtype=tf.int32)
```

### 4. Python `is` and `isinstance` Checks

```python
@tf.function
def check_type(x):
    if isinstance(x, tf.Tensor):  # Evaluated at trace time
        return x * 2
    return x
```

### 5. Mutable Default Arguments

```python
# WRONG: Default argument is captured once
@tf.function
def append_value(x, lst=[]):
    lst.append(x)
    return lst

# CORRECT: Use tf.Variable or pass explicitly
counter = tf.Variable(0)

@tf.function
def increment(x):
    counter.assign_add(x)
    return counter
```

### 6. Return Value Consistency

```python
# WRONG: Returns different types based on condition
@tf.function
def inconsistent(x):
    if x > 0:
        return x          # Returns a tensor
    return [x, x]         # Returns a list

# CORRECT: Always return the same structure
@tf.function
def consistent(x):
    if x > 0:
        return [x]
    return [x, x]
```

---

## Python Side Effects in tf.function

### Understanding Trace-Time vs Call-Time

```python
side_effects = []

@tf.function
def traced_function(x):
    # This executes only during tracing
    side_effects.append("traced")
    print(f"Python print: {x}")

    # This executes every call (TF operation)
    tf.print(f"TF print: {x}")

    return x * 2

traced_function(tf.constant(1.0))
# Python print executes, side_effects = ["traced"]
# TF print executes

traced_function(tf.constant(2.0))
# Python print does NOT execute
# TF print executes
# side_effects still ["traced"]
```

### Using tf.print for Debugging

```python
@tf.function
def debug_function(x):
    tf.print("x =", x)                     # Executes every call
    tf.print("shape =", tf.shape(x))
    tf.print("dtype =", x.dtype)
    return x
```

### Using tf.debugging for Assertions

```python
@tf.function
def safe_divide(x, y):
    tf.debugging.assert_none_equal(y, 0.0, message="Division by zero!")
    return x / y
```

### Using tf.Variable for State

```python
# Use tf.Variable for mutable state inside tf.function
step_counter = tf.Variable(0, dtype=tf.int32)

@tf.function
def training_step(x):
    step_counter.assign_add(1)
    tf.print("Step:", step_counter)
    return x * 2
```

---

## Variable Creation in tf.function

### Rules for Variable Creation

Variables must be created **exactly once** during the lifetime of a
`tf.function`. They cannot be created on every call:

```python
# WRONG: Creates new variable on every trace
@tf.function
def buggy_create(x):
    v = tf.Variable(0.0)  # Error if traced more than once
    return v + x

# CORRECT: Create variable outside
v = tf.Variable(0.0)

@tf.function
def correct_use(x):
    return v + x

# CORRECT: Use tf.Module or tf.keras.layer for lazy creation
class MyModule(tf.Module):
    def __init__(self):
        self.v = None

    @tf.function
    def __call__(self, x):
        if self.v is None:
            self.v = tf.Variable(0.0)
        return self.v + x
```

### Variable Ownership

Variables created inside `tf.function` must be owned by a containing
object (module, layer, or model):

```python
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    @tf.function
    def call(self, inputs):
        # build() creates variables; it is called once
        if not self.built:
            self.build(inputs.shape)
        return tf.matmul(inputs, self.kernel) + self.bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel', shape=[input_shape[-1], self.units])
        self.bias = self.add_weight('bias', shape=[self.units])
```

### experimental_enable_variable_lifting

When `True` (default), variables created inside `tf.function` are "lifted"
out of the function and must follow creation rules. When `False`, variables
behave like mutable tensors:

```python
@tf.function(experimental_enable_variable_lifting=False)
def mutable_tensor_fn(x):
    # Variable acts like a mutable tensor
    v = tf.Variable(x)
    v.assign_add(1.0)
    return v.read_value()
```

---

## Retracing and Performance

### When Retracing Occurs

A `tf.function` creates a new `ConcreteFunction` when the input signature
changes:

1. **Different dtypes**: `tf.int32` vs `tf.float32`
2. **Different ranks**: `[3]` vs `[2, 3]`
3. **Different object types**: `tf.Tensor` vs `tf.Variable`
4. **Different Python values** (non-tensor arguments)
5. **Different tensor shapes** (without `experimental_relax_shapes`)

```python
@tf.function
def add(a, b):
    return a + b

add(tf.constant(1), tf.constant(2))       # Trace 1: int32
add(tf.constant(1.0), tf.constant(2.0))   # Trace 2: float32
add(tf.constant([1]), tf.constant([2]))   # Trace 3: int32, rank 1
add(tf.constant(1), tf.constant(2))       # Reuses Trace 1
```

### Avoiding Excessive Retracing

```python
# BAD: Python int triggers retracing on every different value
@tf.function
def process(x, threshold):
    return tf.cast(x > threshold, tf.float32)

for i in range(100):
    process(tf.constant([1.0, 2.0, 3.0]), i)  # 100 traces!

# GOOD: Pass threshold as a tensor
@tf.function
def process(x, threshold):
    return tf.cast(x > threshold, tf.float32)

for i in range(100):
    process(tf.constant([1.0, 2.0, 3.0]), tf.constant(float(i)))  # 1 trace

# GOOD: Use input_signature to constrain
@tf.function(input_signature=[
    tf.TensorSpec([None], tf.float32),
    tf.TensorSpec([], tf.float32)
])
def process(x, threshold):
    return tf.cast(x > threshold, tf.float32)
```

### Monitoring Retracing

```python
# Use tf.debugging.set_log_device_placement to monitor tracing
import logging
tf.get_logger().setLevel(logging.DEBUG)

# Check the number of traces
@tf.function
def my_func(x):
    return x + 1

my_func.experimental_get_tracing_count()  # 0
my_func(tf.constant(1))
my_func.experimental_get_tracing_count()  # 1
my_func(tf.constant(2))
my_func.experimental_get_tracing_count()  # 1 (reused)
my_func(tf.constant(1.0))
my_func.experimental_get_tracing_count()  # 2 (new dtype)
```

---

## tf.TensorSpec and Function Signatures

### Input Signature Specifications

```python
# Single input
@tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
def process(x):
    return x * 2

# Multiple inputs
@tf.function(input_signature=[
    tf.TensorSpec([None, 28, 28, 1], tf.float32, name='images'),
    tf.TensorSpec([None], tf.int32, name='labels')
])
def train(images, labels):
    pass

# Nested structures
@tf.function(input_signature={
    'images': tf.TensorSpec([None, 28, 28, 1], tf.float32),
    'labels': tf.TensorSpec([None], tf.int32)
})
def train(data):
    return data['images'], data['labels']

# Optional inputs (None in the input signature)
@tf.function(input_signature=[
    tf.TensorSpec([None, 3], tf.float32),
    tf.TensorSpec([None, 3], tf.float32),  # Could be None
])
def process(x, mask=None):
    if mask is not None:
        return x * mask
    return x
```

### experimental_relax_shapes

```python
# Without relax: each unique shape creates a new trace
@tf.function
def dense(x):
    return tf.reduce_sum(x)

dense(tf.zeros([1]))     # Trace 1
dense(tf.zeros([10]))    # Trace 2
dense(tf.zeros([100]))   # Trace 3

# With relax: shapes of same rank share a trace
@tf.function(experimental_relax_shapes=True)
def dense_relaxed(x):
    return tf.reduce_sum(x)

dense_relaxed(tf.zeros([1]))     # Trace 1
dense_relaxed(tf.zeros([10]))    # Reuses Trace 1
dense_relaxed(tf.zeros([100]))   # Reuses Trace 1
```

### experimental_follow_type_hints

```python
# Use Python type hints for automatic input signatures
@tf.function(experimental_follow_type_hints=True)
def typed_function(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return x + y
```

---

## Composite Tensors in Functions

### Supported Composite Types

`tf.function` supports composite tensors (tensors with non-standard
representations):

```python
# SparseTensor
@tf.function
def process_sparse(st):
    return tf.sparse.to_dense(st) * 2

st = tf.sparse.SparseTensor(
    indices=[[0, 0], [1, 2]],
    values=[1.0, 2.0],
    dense_shape=[3, 3]
)
process_sparse(st)

# RaggedTensor
@tf.function
def process_ragged(rt):
    return rt * 2

rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
process_ragged(rt)

# TensorArray
@tf.function
def process_array():
    ta = tf.TensorArray(tf.float32, size=3)
    ta = ta.write(0, 1.0)
    ta = ta.write(1, 2.0)
    ta = ta.write(2, 3.0)
    return ta.stack()
```

### Custom Composite Tensors

```python
class MyCompositeTensor(tf.experimental.ExtensionType):
    values: tf.Tensor
    indices: tf.Tensor

    @property
    def shape(self):
        return self.values.shape

@tf.function
def process_composite(ct):
    return ct.values[ct.indices]
```

---

## Function Serialization and SavedModel

### Saving tf.function as Part of SavedModel

```python
class MyModule(tf.Module):
    def __init__(self):
        self.v = tf.Variable(1.0)

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def serve(self, x):
        return self.v * x

module = MyModule()
tf.saved_model.save(module, '/tmp/model',
    signatures=module.serve.get_concrete_function(
        tf.TensorSpec([], tf.float32)))
```

### Loading and Using Saved Functions

```python
loaded = tf.saved_model.load('/tmp/model')
result = loaded.serve(tf.constant(2.0))  # 2.0
```

### ConcreteFunction Serialization

```python
# Export specific concrete function
@tf.function
def my_func(x):
    return x * 2

# Get concrete function for serialization
cf = my_func.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))

# Save
tf.saved_model.save(tf.Module(), '/tmp/model',
    signatures=cf)

# The concrete function captures:
# - The computation graph
# - Variable captures (if any)
# - Input/output signatures
# - Captured assets
```

### Function Graph Transformation

```python
# Access the underlying graph
@tf.function
def my_func(x):
    return tf.matmul(x, x)

cf = my_func.get_concrete_function(tf.TensorSpec([3, 3], tf.float32))
graph = cf.graph

# Inspect operations
for op in graph.get_operations():
    print(op.name, op.type)

# Graph transformation (advanced)
from tensorflow.python.eager.polymorphic_function.transform import (
    FUNC_GRAPH_TRANSFORMS
)
```

---

## Summary

TensorFlow's automatic differentiation and function compilation system
provides:

- **tf.GradientTape**: Reverse-mode automatic differentiation with
  support for higher-order gradients, persistent tapes, and Jacobian
  computation
- **ForwardAccumulator**: Forward-mode differentiation for efficient
  Jacobian-vector products
- **tf.custom_gradient**: Custom gradient definitions for numerical
  stability or special gradient behavior
- **tf.function**: Python-to-graph compilation with polymorphic
  dispatch based on input signatures
- **AutoGraph**: Automatic conversion of Python control flow to graph
  operations (if, while, for)
- **ConcreteFunction**: Graph-compiled functions bound to specific
  input types, used for serialization and SavedModel
- **Tracing**: The process of recording operations to build a graph,
  with caching and retracing controlled by input signatures

Key best practices:
- Use `tf.function` for performance-critical code
- Avoid Python side effects in traced functions (use `tf.print`, `tf.Variable`)
- Use `input_signature` to control retracing
- Create variables outside `tf.function` or use `tf.Module`/`tf.keras.layers`
- Use `experimental_relax_shapes` when input shapes vary
- Monitor tracing count to detect excessive retracing
