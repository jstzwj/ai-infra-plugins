# TensorFlow Control Flow Reference

## Table of Contents

1. [Overview](#overview)
2. [tf.cond](#tfcond)
3. [tf.while_loop](#tfwhile_loop)
4. [tf.case](#tfcase)
5. [tf.switch_case](#tfswitch_case)
6. [tf.map_fn](#tfmap_fn)
7. [tf.foldl and tf.foldr](#tfoldl-and-tfoldr)
8. [tf.TensorArray](#tftensorarray)
9. [tf.identity](#tfidentity)
10. [tf.control_dependencies](#tfcontrol_dependencies)
11. [tf.group](#tfgroup)
12. [tf.no_op](#tfno_op)
13. [tf.print](#tfprint)
14. [tf.Assert](#tfassert)
15. [AutoGraph Control Flow](#autograph-control-flow)
16. [tf.function and Control Flow](#tffunction-and-control-flow)

---

## Overview

TensorFlow provides a rich set of control flow operations for building complex
computation graphs. In graph mode, control flow is represented as special ops
(Switch, Merge, Enter, Exit, NextIteration) that the TensorFlow runtime
executes. In eager mode, standard Python control flow is used directly.

Control flow in TensorFlow serves two primary purposes:
1. **Graph-mode control flow**: Enables conditional execution, loops, and other
   control structures within `tf.function`-decorated code, where the computation
   is represented as a static graph.
2. **Eager-mode compatibility**: In eager execution, standard Python control flow
   works naturally, but graph-compatible control flow ops ensure code can be
   seamlessly traced into graphs.

The key control flow primitives are:
- **Switch/Merge**: For conditional branching (`tf.cond`).
- **Enter/Exit/NextIteration**: For loop iteration (`tf.while_loop`).
- **TensorArray**: For accumulating results across loop iterations.

---

## tf.cond

### Signature

```python
tf.cond(
    pred,
    true_fn,
    false_fn,
    name=None
)
```

### Parameters

**pred** (`tf.Tensor` or callable):
A scalar boolean tensor, or a callable returning one. Determines which branch
to execute.

**true_fn** (callable):
A callable executed when `pred` is `True`. Must return a tensor or nested
structure of tensors. Should not have side effects that depend on whether it
is called (both branches may be traced).

**false_fn** (callable):
A callable executed when `pred` is `False`. Must return a tensor or nested
structure of tensors with the same structure and dtypes as `true_fn`.

**name** (`str`, optional):
Name for the operation.

### Return Value

Returns the result of `true_fn()` if `pred` is true, or `false_fn()` otherwise.
The return value has the same nested structure as the branch outputs.

### Behavior

When executing eagerly, only the relevant branch is executed. When tracing
(e.g., inside `tf.function`), both branches are traced into the graph, and the
runtime selects which to execute based on `pred`.

### Example: Basic Usage

```python
x = tf.constant(5)
result = tf.cond(
    x > 3,
    lambda: tf.multiply(x, 2),
    lambda: tf.multiply(x, 3)
)
# result = 10
```

### Example: Multiple Outputs

```python
def true_branch():
    return tf.constant(1), tf.constant('true')

def false_branch():
    return tf.constant(0), tf.constant('false')

a, b = tf.cond(tf.constant(True), true_branch, false_branch)
# a = 1, b = 'true'
```

### Example: With Variables

```python
v = tf.Variable(0.0)
result = tf.cond(
    v.read_value() > 0,
    lambda: v.assign_add(1.0),
    lambda: v.assign_sub(1.0)
)
```

### Important Notes

1. **Both branches are traced**: When used in `tf.function`, both `true_fn` and
   `false_fn` are traced. Side effects in one branch may affect the traced graph.

2. **Output structure must match**: The return values of both branches must have
   the same nested structure, dtypes, and compatible shapes.

3. **Tensor arguments**: Use closures or `functools.partial` to pass additional
   arguments to branch functions.

4. **Nested cond**: `tf.cond` can be nested for complex conditional logic.

### Internal Implementation

In graph mode, `tf.cond` creates:
- A `Switch` op that routes `pred` to control which branch is active.
- Two subgraphs, one for each branch.
- A `Merge` op that combines the outputs.

---

## tf.while_loop

### Signature

```python
tf.while_loop(
    cond,
    body,
    loop_vars,
    shape_invariants=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    name=None,
    maximum_iterations=None,
    return_same_structure=False
)
```

### Parameters

**cond** (callable):
A callable that takes `loop_vars` and returns a boolean scalar tensor. The loop
continues as long as `cond` returns `True`.

**body** (callable):
A callable that takes `loop_vars` and returns a (possibly updated) set of loop
variables. Must return the same number and types of tensors as it received.

**loop_vars** (tensor or nested structure):
Initial values for the loop variables. Can be a single tensor, a tuple of
tensors, or a nested structure.

**shape_invariants** (nested structure of `tf.TensorShape`):
Shape invariants for the loop variables. Allows shapes to change across
iterations (e.g., dynamic batch sizes). Only the specified dimensions must
remain constant.

```python
# Allow variable-length second dimension
shape_invariants = [tf.TensorShape([None, None])]
```

**parallel_iterations** (`int`):
Number of iterations allowed to run in parallel. Higher values use more memory
but can improve performance for small loops. Default 10.

**back_prop** (`bool`):
Whether to support backpropagation through the loop. Default `True`. Setting to
`False` disables gradient computation, which can improve performance.

**swap_memory** (`bool`):
Whether to enable GPU-CPU memory swapping for backpropagation. Useful for
loops with large intermediate values. Default `False`.

**name** (`str`, optional):
Name for the loop operation.

**maximum_iterations** (`int` or `tf.Tensor`, optional):
Maximum number of iterations. The loop stops when either `cond` returns `False`
or this limit is reached.

**return_same_structure** (`bool`):
If `True`, the return value has the same structure as `loop_vars`. If `False`
(default), returns a flat tuple of tensors.

### Return Value

The final values of the loop variables after the loop terminates. If
`return_same_structure=True`, the return value has the same nested structure
as `loop_vars`.

### Example: Basic Counter

```python
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
result = tf.while_loop(c, b, [i])
# result = [10]
```

### Example: Fibonacci

```python
def cond(i, a, b):
    return i < 10

def body(i, a, b):
    return i + 1, b, a + b

i, a, b = tf.while_loop(cond, body, [0, 0, 1])
# i=10, a=34, b=55
```

### Example: Dynamic Shape

```python
def cond(i, arr):
    return i < 5

def body(i, arr):
    new_element = tf.expand_dims(tf.constant([i * 10]), 0)
    arr = tf.concat([arr, new_element], axis=0)
    return i + 1, arr

# Must specify shape_invariants for dynamic shapes
i, result = tf.while_loop(
    cond, body,
    [0, tf.constant([[0]], dtype=tf.int32)],
    shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, 1])])
```

### Example: Maximum Iterations

```python
i = tf.constant(0)
c = lambda i: True  # Would loop forever
b = lambda i: i + 1
result = tf.while_loop(c, b, [i], maximum_iterations=100)
# result = [100]
```

### Gradient Flow

When `back_prop=True`, gradients flow through the while loop. Each iteration's
contribution to the gradient is computed:

```python
@tf.function
def cumulative_sum(x, n):
    i = tf.constant(0)
    total = tf.constant(0.0)
    def cond(i, total):
        return i < n
    def body(i, total):
        return i + 1, total + x
    _, result = tf.while_loop(cond, body, [i, total])
    return result

with tf.GradientTape() as tape:
    x = tf.Variable(2.0)
    y = cumulative_sum(x, 5)
# dy/dx = 5 (sum is x * n)
```

### Internal Implementation

In graph mode, `tf.while_loop` creates:
1. **Enter ops**: Move loop variables into the loop frame.
2. **Switch ops**: Route variables based on loop condition.
3. **Merge ops**: Combine variables from initial entry and loop back-edge.
4. **NextIteration ops**: Pass updated variables to the next iteration.
5. **Exit ops**: Move final variable values out of the loop frame.

The loop body is executed in a unique "frame" context, allowing the runtime
to manage iteration state.

---

## tf.case

### Signature

```python
tf.case(
    pred_fn_pairs,
    default=None,
    exclusive=False,
    name='case'
)
```

### Parameters

**pred_fn_pairs** (list or dict):
A list of `(pred, fn)` pairs or a dictionary mapping predicates to functions.
Each `pred` is a boolean scalar tensor, and `fn` is a callable.

**default** (callable, optional):
The default function to execute if no predicate is `True`. If `None` and no
predicate matches, an error is raised.

**exclusive** (`bool`):
If `True`, exactly one predicate must be `True`. If `False`, the first `True`
predicate's function is executed.

**name** (`str`):
Name for the operation.

### Return Value

The result of the first function whose predicate is `True`.

### Example

```python
x = tf.constant(5)
result = tf.case(
    [(tf.equal(x, 1), lambda: tf.constant('one')),
     (tf.equal(x, 5), lambda: tf.constant('five')),
     (tf.greater(x, 10), lambda: tf.constant('big'))],
    default=lambda: tf.constant('other'),
    exclusive=False
)
# result = 'five'
```

### Example: Exclusive Mode

```python
result = tf.case(
    {tf.constant(True): lambda: 1},
    exclusive=True,
    default=lambda: 0
)
```

---

## tf.switch_case

### Signature

```python
tf.switch_case(
    branch_index,
    branches,
    default=None,
    name='switch_case'
)
```

### Parameters

**branch_index** (`int` or `tf.Tensor`):
An integer tensor specifying which branch to execute.

**branches** (list or dict):
A list of callables or dict mapping integers to callables.

**default** (callable, optional):
Default function if `branch_index` is out of range.

**name** (`str`):
Name for the operation.

### Example

```python
branch_index = tf.constant(1)
result = tf.switch_case(
    branch_index,
    branches=[
        lambda: tf.constant('branch 0'),
        lambda: tf.constant('branch 1'),
        lambda: tf.constant('branch 2')
    ],
    default=lambda: tf.constant('default')
)
# result = 'branch 1'
```

---

## tf.map_fn

### Signature

```python
tf.map_fn(
    fn,
    elems,
    dtype=None,
    parallel_iterations=None,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    name=None
)
```

### Parameters

**fn** (callable):
A callable that takes a single element from `elems` and returns a tensor or
nested structure of tensors.

**elems** (`tf.Tensor` or nested structure):
The input to map over. The first dimension is the batch/map dimension.

**dtype** (`tf.DType` or nested structure, optional):
Output dtype(s). If `None`, inferred from the first call to `fn`.

**parallel_iterations** (`int`, optional):
Number of elements to process in parallel. Default depends on execution mode.

**back_prop** (`bool`):
Whether to support backpropagation. Default `True`.

**swap_memory** (`bool`):
Whether to enable memory swapping. Default `False`.

**infer_shape** (`bool`):
Whether to infer output shapes. Default `True`. Set to `False` if output
shapes vary.

**name** (`str`, optional):
Name for the operation.

### Return Value

A tensor or nested structure of tensors, where the first dimension matches the
first dimension of `elems`.

### Example: Element-wise Transformation

```python
elems = tf.constant([1, 2, 3, 4, 5])
result = tf.map_fn(lambda x: x * x, elems)
# result = [1, 4, 9, 16, 25]
```

### Example: Multiple Inputs

```python
elems = (tf.constant([1, 2, 3]), tf.constant([4, 5, 6]))
result = tf.map_fn(lambda x: x[0] + x[1], elems)
# result = [5, 7, 9]
```

### Example: Structured Output

```python
elems = tf.constant([1.0, 2.0, 3.0])
result = tf.map_fn(
    lambda x: {'square': x * x, 'cube': x * x * x},
    elems,
    dtype={'square': tf.float32, 'cube': tf.float32}
)
# result['square'] = [1.0, 4.0, 9.0]
# result['cube'] = [1.0, 8.0, 27.0]
```

### Performance Notes

- `tf.map_fn` is generally slower than vectorized operations (`tf.vectorized_map`).
- Use vectorized ops when possible (e.g., `tf.multiply(elems, elems)` instead of
  `tf.map_fn(lambda x: x * x, elems)`).
- `tf.vectorized_map` can be faster as it batches operations.
- `parallel_iterations` can be tuned for performance vs. memory trade-off.

---

## tf.foldl and tf.foldr

### tf.foldl

```python
tf.foldl(
    fn,
    elems,
    initializer=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    name=None
)
```

Left fold: applies `fn` cumulatively from left to right.

**fn** (callable): `fn(accumulator, element) -> new_accumulator`.

```python
elems = tf.constant([1, 2, 3, 4, 5])
result = tf.foldl(lambda a, x: a + x, elems)
# result = 15 (1+2+3+4+5)

# With initializer
result = tf.foldl(lambda a, x: a + x, elems, initializer=10)
# result = 25 (10+1+2+3+4+5)
```

### tf.foldr

```python
tf.foldr(
    fn,
    elems,
    initializer=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    name=None
)
```

Right fold: applies `fn` cumulatively from right to left.

```python
elems = tf.constant([1, 2, 3])
# foldr: 1 + (2 + (3 + 0))
result = tf.foldr(lambda a, x: a + x, elems)
# result = 6
```

---

## tf.TensorArray

### Overview

`tf.TensorArray` provides dynamic-size arrays that can be used inside
`tf.while_loop` to accumulate results across iterations. Unlike regular tensors,
TensorArrays support writing at variable indices and can grow dynamically.

### Constructor

```python
tf.TensorArray(
    dtype,
    size=None,
    dynamic_size=None,
    clear_after_read=None,
    element_shape=None,
    name=None,
    colocate_with_first_write_call=True,
    infer_shape=True
)
```

**dtype** (`tf.DType`):
Data type of elements.

**size** (`int` or `tf.Tensor`):
Initial size of the array. Required if `dynamic_size=False`.

**dynamic_size** (`bool`):
If `True`, the array can grow beyond its initial size. Default behavior depends
on usage context.

**clear_after_read** (`bool`):
If `True` (default), elements are cleared after reading to free memory. Set to
`False` for repeated reads.

**element_shape** (`tf.TensorShape`):
Shape of each element. Required for graph mode if writing after reading.

**name** (`str`):
Name for the TensorArray.

### Core Methods

**write(index, value, name=None)**
Write a value at the given index. Returns a new TensorArray with the value
written.

```python
ta = tf.TensorArray(dtype=tf.float32, size=3)
ta = ta.write(0, tf.constant([1.0, 2.0]))
ta = ta.write(1, tf.constant([3.0, 4.0]))
ta = ta.write(2, tf.constant([5.0, 6.0]))
```

**read(index, name=None)**
Read a value at the given index.

```python
value = ta.read(1)  # [3.0, 4.0]
```

**stack(name=None)**
Stack all elements into a single tensor along a new first dimension.

```python
stacked = ta.stack()
# shape: [3, 2], values: [[1,2],[3,4],[5,6]]
```

**unstack(value, name=None)**
Unstack a tensor into individual elements.

```python
ta = tf.TensorArray(dtype=tf.float32, size=3)
ta = ta.unstack(tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
```

**size(name=None)**
Return the current size of the TensorArray.

**scatter(indices, value, name=None)**
Scatter values from `value` into the specified indices.

**gather(indices, name=None)**
Gather elements at the specified indices.

**concat(name=None)**
Return the elements as a concatenated tensor.

**split(value, lengths, name=None)**
Split a value into elements based on lengths.

### Example: Accumulate in While Loop

```python
@tf.function
def accumulate_loop(n):
    ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def cond(i, ta):
        return i < n

    def body(i, ta):
        ta = ta.write(i, tf.cast(i, tf.float32) ** 2)
        return i + 1, ta

    _, ta = tf.while_loop(cond, body, [0, ta])
    return ta.stack()

result = accumulate_loop(5)
# [0.0, 1.0, 4.0, 9.0, 16.0]
```

### Example: Clear After Read

```python
ta = tf.TensorArray(dtype=tf.float32, size=3, clear_after_read=False)
ta = ta.unstack(tf.constant([1.0, 2.0, 3.0]))
a = ta.read(0)  # 1.0
b = ta.read(0)  # 1.0 (still available because clear_after_read=False)
```

### Example: Dynamic Size

```python
@tf.function
def dynamic_accumulate():
    ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                         element_shape=tf.TensorShape([]))
    for i in tf.range(5):
        ta = ta.write(i, tf.cast(i * 10, tf.float32))
    return ta.stack()

result = dynamic_accumulate()
# [0.0, 10.0, 20.0, 30.0, 40.0]
```

---

## tf.identity

### Signature

```python
tf.identity(input, name=None)
```

Creates a passthrough op that returns its input unchanged. Useful for:
1. Creating explicit control dependencies.
2. Ensuring a tensor is evaluated at a specific point in the graph.
3. Renaming tensors for debugging.

### Example: Control Dependency

```python
with tf.control_dependencies([update_op]):
    result = tf.identity(value_tensor)
# result is computed only after update_op completes
```

### Example: Evaluation Anchor

```python
loss = compute_loss(x)
# Create a named identity for fetching
total_loss = tf.identity(loss, name='total_loss')
```

---

## tf.control_dependencies

### Signature

```python
tf.control_dependencies(control_inputs)
```

Context manager that specifies operations that must complete before operations
within the context execute.

### Parameters

**control_inputs** (list):
List of operations or tensors to create dependencies on.

### Example

```python
with tf.control_dependencies([a, b]):
    c = tf.add(x, y)
    d = tf.multiply(x, y)
# c and d are computed only after a and b have completed
```

### Example: Variable Update Ordering

```python
v = tf.Variable(0.0)
update = v.assign_add(1.0)

with tf.control_dependencies([update]):
    # This read happens after the update
    value = tf.identity(v)
```

### Important Notes

1. Only operations within the scope are affected. Operations defined outside
   the scope are not affected.
2. The dependency applies to newly created operations, not to existing tensors.
3. Use `tf.identity` to anchor a read to the dependency.

---

## tf.group

### Signature

```python
tf.group(*inputs, **kwargs)
```

Creates an operation that groups multiple operations. The returned op completes
only when all input operations have completed.

### Example

```python
update_a = var_a.assign_add(1.0)
update_b = var_b.assign_sub(1.0)
update_all = tf.group(update_a, update_b)
# Running update_all runs both updates
```

### Use Case

Commonly used to combine multiple variable updates into a single operation:

```python
with tf.control_dependencies([loss_op]):
    gradients = tape.gradient(loss, variables)
    apply_grads = optimizer.apply_gradients(zip(gradients, variables))
    train_op = tf.group(apply_grads)
```

---

## tf.no_op

### Signature

```python
tf.no_op(name=None)
```

Creates an operation that does nothing. Useful as a placeholder in control flow:

```python
# Either update or do nothing
update_op = tf.cond(
    should_update,
    lambda: var.assign_add(1.0),
    lambda: tf.no_op()
)
```

---

## tf.print

### Signature

```python
tf.print(
    *inputs,
    output_stream=None,
    name=None,
    **kwargs
)
```

Prints tensors during graph execution. Unlike Python's `print`, this works
inside `tf.function` and during graph mode execution.

### Parameters

**inputs**: Tensors or strings to print.

**output_stream** (`str`): Where to print. Options:
- `'stdout'`, `'stderr'`, `'logging:info'`, `'logging:warning'`, etc.
- Default is `'stderr'`.

**sep** (`str`): Separator between inputs. Default `' '`.

**end** (`str`): End character. Default `'\n'`.

### Example

```python
@tf.function
def compute(x):
    tf.print("x =", x, "x^2 =", x * x, output_stream="stdout")
    return x * x

result = compute(tf.constant(3.0))
# Prints: x = 3 x^2 = 9
```

### Example: Debug in Graph

```python
x = tf.constant(5.0)
y = x * x
with tf.control_dependencies([tf.print("y =", y)]):
    z = y + 1
# When z is evaluated, it prints "y = 25" first
```

---

## tf.Assert

### Signature

```python
tf.Assert(
    condition,
    data,
    summarize=None,
    name=None
)
```

Asserts that `condition` is true during graph execution. If `condition` is
false, prints `data` and raises `tf.errors.InvalidArgumentError`.

### Parameters

**condition** (`tf.Tensor`):
Boolean scalar tensor.

**data** (list):
Tensors or strings to print if the assertion fails.

**summarize** (`int`):
Number of entries to print for large tensors.

### Example

```python
@tf.function
def safe_divide(a, b):
    assert_op = tf.Assert(
        tf.reduce_all(b != 0),
        ["Division by zero! b =", b]
    )
    with tf.control_dependencies([assert_op]):
        return a / b
```

---

## AutoGraph Control Flow

### Overview

AutoGraph (`tf.autograph`) automatically converts Python control flow statements
into their TensorFlow graph equivalents:

| Python | TensorFlow |
|--------|-----------|
| `if` statement | `tf.cond` |
| `while` statement | `tf.while_loop` |
| `for ... in range()` | `tf.while_loop` |
| `for ... in tf.data.Dataset` | Dataset iteration |
| `for ... in tensor` | `tf.while_loop` with `tf.TensorArray` |
| `break` statement | Loop exit condition |
| `continue` statement | Skip to next iteration |

### Conversion Rules

#### if/else

Python `if/else` with tensor-dependent conditions is converted to `tf.cond`:

```python
@tf.function
def my_func(x):
    if x > 0:  # Converted to tf.cond
        return x * 2
    else:
        return -x
```

Conditions must be tensor expressions. Python constants (non-tensor conditions)
are evaluated at trace time and not converted.

#### while Loops

Python `while` loops with tensor-dependent conditions become `tf.while_loop`:

```python
@tf.function
def my_func(n):
    i = tf.constant(0)
    total = tf.constant(0.0)
    while i < n:  # Converted to tf.while_loop
        total += tf.cast(i, tf.float32)
        i += 1
    return total
```

#### for Loops

Python `for` loops over `tf.range` become `tf.while_loop`:

```python
@tf.function
def my_func():
    total = tf.constant(0)
    for i in tf.range(10):  # Converted to tf.while_loop
        total += i
    return total
```

For loops over Python `range` are unrolled at trace time (not recommended
for large ranges).

#### for Loops Over Tensors

```python
@tf.function
def my_func(tensor):
    result = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for i, elem in enumerate(tensor):
        result = result.write(i, elem * 2)
    return result.stack()
```

### AutoGraph Options

```python
@tf.function(experimental_follow_type_hints=True)
def my_func(x: tf.Tensor) -> tf.Tensor:
    ...
```

### Limitations

1. **Variable creation**: Variables should not be created inside control flow
   branches (create them outside and assign inside).

2. **Python side effects**: `print()`, file I/O, etc. inside control flow
   only execute during tracing, not during execution (use `tf.print` instead).

3. **Shape changes**: Loops that change tensor shapes need `shape_invariants`
   specified via `tf.while_loop` directly.

4. **Break/continue**: Supported but may have performance implications.

---

## tf.function and Control Flow

### Tracing Implications

When a `tf.function` is traced, control flow operations are captured in the
graph. The tracing behavior depends on the type of arguments:

1. **Concrete input signature**: The function is traced once for the specific
   input shapes and dtypes.

2. **Polymorphic input signature**: The function is retraced for each unique
   input signature (set of shapes and dtypes).

### Control Flow in tf.function

```python
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    # This for loop over tf.range becomes a tf.while_loop in the graph
    gradients = tape.gradient(loss, model.trainable_variables)

    # tf.cond is used inside the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### ConcreteFunction vs Polymorphic Function

**ConcreteFunction**: A single graph with fixed input/output signatures.
```python
cf = my_func.get_concrete_function(tf.TensorSpec([None, 3], tf.float32))
```

**Polymorphic Function**: A collection of ConcreteFunctions, one per
input signature. The dispatcher selects the correct one at call time.

### Side Effects in Control Flow

Side effects (variable updates, `tf.print`) inside `tf.cond` or
`tf.while_loop` behave differently in graph vs. eager mode:

```python
@tf.function
def side_effect_example(x):
    # tf.print executes during graph execution
    tf.print("Processing:", x)
    if x > 0:
        tf.print("Positive!")
        return x
    else:
        tf.print("Non-positive!")
        return -x
```

### Retracing and Control Flow

Avoid Python-level control flow that depends on tensor values, as this causes
retracing:

```python
# BAD: Causes retracing for each new value
@tf.function
def bad_func(x):
    if x > 0:  # x is a tensor, not a Python value
        return x * 2

# GOOD: Use tf.cond for tensor-dependent conditions
@tf.function
def good_func(x):
    return tf.cond(x > 0, lambda: x * 2, lambda: -x)
```

However, AutoGraph handles this automatically in most cases.

### Nested tf.function

Control flow inside nested `tf.function` calls behaves correctly:

```python
@tf.function
def inner(x):
    return tf.cond(x > 0, lambda: x * 2, lambda: -x)

@tf.function
def outer(x):
    result = inner(x)
    return result + 1
```

### Gradient Tape and Control Flow

`tf.GradientTape` correctly handles gradients through control flow ops:

```python
@tf.function
def loop_fn(x):
    for _ in tf.range(5):
        x = x * 2  # Gradient: 2^5 = 32
    return x

with tf.GradientTape() as tape:
    v = tf.Variable(1.0)
    y = loop_fn(v)
grad = tape.gradient(y, v)
# grad = 32.0
```
