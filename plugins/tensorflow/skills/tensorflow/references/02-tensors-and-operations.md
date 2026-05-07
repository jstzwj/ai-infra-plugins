# TensorFlow Tensors and Operations Reference

## Table of Contents

1. [The tf.Tensor Class](#the-tftensor-class)
2. [Tensor Creation](#tensor-creation)
3. [Data Types](#data-types)
4. [Tensor Shapes](#tensor-shapes)
5. [Broadcasting Rules](#broadcasting-rules)
6. [Indexing and Slicing](#indexing-and-slicing)
7. [Math Operations](#math-operations)
8. [Matrix Operations](#matrix-operations)
9. [Tensor Manipulation](#tensor-manipulation)
10. [Type Casting and Numeric Conversion](#type-casting-and-numeric-conversion)
11. [String Tensor Operations](#string-tensor-operations)
12. [Complex Number Operations](#complex-number-operations)
13. [Quantized Tensor Operations](#quantized-tensor-operations)
14. [Ragged and Sparse Tensor Basics](#ragged-and-sparse-tensor-basics)

---

## The tf.Tensor Class

### Definition

`tf.Tensor` is the core data structure in TensorFlow, representing a
multidimensional array of elements. All elements share a single known data type.
The class is defined in `tensorflow/python/framework/tensor.py` and is exported
as `tf.Tensor`.

```python
@tf_export("Tensor", "experimental.numpy.ndarray", v1=["Tensor"])
class Tensor(internal.NativeObject, core_tf_types.Symbol):
```

### Properties

#### `dtype`

Returns the `tf.DType` of elements in this tensor.

```python
t = tf.constant([1, 2, 3])
print(t.dtype)  # tf.int32

t_float = tf.constant([1.0, 2.0, 3.0])
print(t_float.dtype)  # tf.float32
```

The dtype is set at creation time and cannot be changed. To convert between
types, use `tf.cast()`.

#### `shape`

Returns a `tf.TensorShape` representing the static shape of the tensor.

```python
t = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(t.shape)  # (2, 3)
print(type(t.shape))  # <class 'tensorflow.python.framework.tensor_shape.TensorShape'>
```

In eager execution, the shape is always fully known. In `tf.function` tracing,
the shape may be partially known with `None` for unknown dimensions.

Note: `tf.Tensor.shape` is equivalent to `tf.Tensor.get_shape()`. To get a
tensor containing the shape (for dynamic shapes), use `tf.shape(t)`.

#### `name`

Returns the name of the tensor. In eager mode, this is typically auto-generated.
In graph mode, it follows the format `op_name:output_index`.

```python
t = tf.constant(1.0, name="my_tensor")
print(t.name)  # "my_tensor:0" (in graph mode)
```

#### `device`

Returns the device string where the tensor is located.

```python
with tf.device('/GPU:0'):
    t = tf.constant([1.0, 2.0])
print(t.device)  # /job:localhost/replica:0/task:0/device:GPU:0
```

#### `ndim`

Returns the rank (number of dimensions) of the tensor as a Python integer,
or `None` if the rank is unknown.

```python
t = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t.ndim)  # 3
```

#### `graph`

Returns the `tf.Graph` that contains this tensor (graph mode only). In eager
mode, this attribute does not exist.

### Methods

#### `numpy()`

Converts the tensor to a NumPy `ndarray`. Only available in eager mode.

```python
t = tf.constant([1.0, 2.0, 3.0])
arr = t.numpy()
print(type(arr))  # <class 'numpy.ndarray'>
print(arr)  # [1. 2. 3.]
```

Note: For `EagerTensor` (the internal subclass used in eager mode), `numpy()`
is a method on the C++ object. For graph-mode `Tensor` objects, `numpy()` is
not available; use `eval()` with a session instead.

#### `get_shape()`

Alias for the `shape` property. Returns a `tf.TensorShape`.

```python
t = tf.constant([1, 2, 3, 4, 5])
t.get_shape()  # TensorShape([5])
```

#### `set_shape(shape)`

Updates the shape of this tensor by merging with the provided shape. Used to
provide additional shape information that cannot be inferred automatically.

```python
t = tf.keras.Input(shape=[None, None, 3])
print(t.shape)  # (None, None, None, 3)
t.set_shape([None, 224, 224, None])
print(t.shape)  # (None, 224, 224, 3)
```

Raises `ValueError` if the new shape is incompatible with the existing shape.

#### `eval(feed_dict=None, session=None)`

Evaluates this tensor in a session (TF1/graph mode only). In TF2 eager mode,
use `.numpy()` instead.

#### `ref()`

Returns a hashable reference object for use in sets or as dictionary keys.
Since TF2 tensors are unhashable (element-wise equality), `ref()` provides
a workaround:

```python
x = tf.constant(5)
y = tf.constant(10)
tensor_set = {x.ref(), y.ref()}
print(x.ref() in tensor_set)  # True
print(x.ref().deref())  # <tf.Tensor: shape=(), dtype=int32, numpy=5>
```

### Operator Overloading

`tf.Tensor` supports the following Python operators (defined in
`OVERLOADABLE_OPERATORS`):

**Binary operators**: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `@` (matmul),
`<`, `<=`, `>`, `>=`, `==`, `!=`, `&`, `|`, `^`

**Unary operators**: `-` (negate), `~` (invert), `abs()`

**Indexing**: `[]` (getitem)

**Right-hand variants**: `__radd__`, `__rsub__`, etc. for operations where
the tensor is on the right side.

```python
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# All of these work naturally
c = a + b        # tf.add
d = a * b        # tf.multiply
e = a @ tf.reshape(b, [3, 1])  # tf.matmul
f = a < b        # tf.less
g = a ** 2       # tf.pow
h = -a           # tf.negative
i = abs(a - 5)   # tf.abs
j = a[0]         # tf.gather
```

### Equality and Hashing

In TF2, tensors use element-wise equality and are **unhashable**:

```python
x = tf.constant([1, 2, 3])
y = tf.constant([1, 2, 3])
print(x == y)  # tf.Tensor([ True  True  True], shape=(3,), dtype=bool)

# This raises TypeError:
# my_set = {x, y}  # TypeError: Tensor is unhashable
# Use x.ref() instead
```

This behavior is controlled by `Tensor._USE_EQUALITY`, which defaults to
`True` in TF2.

### EagerTensor

In eager mode, tensors are actually instances of `EagerTensor`, a C++
subclass that adds:
- `numpy()` method for immediate conversion to ndarray
- Device placement tracking
- Direct buffer access for performance

`EagerTensor` is an internal detail; users should treat it as `tf.Tensor`.

---

## Tensor Creation

### tf.constant

Creates a tensor from a constant value (Python literal, list, or NumPy array).

```python
# From a scalar
t1 = tf.constant(42)                  # shape=(), dtype=int32
t2 = tf.constant(3.14)                # shape=(), dtype=float32

# From a list
t3 = tf.constant([1, 2, 3])           # shape=(3,), dtype=int32
t4 = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # shape=(2, 2), dtype=float32

# Specify dtype explicitly
t5 = tf.constant([1, 2, 3], dtype=tf.float64)  # shape=(3,), dtype=float64

# Specify shape
t6 = tf.constant(0, shape=[2, 3])     # [[0, 0, 0], [0, 0, 0]]

# From a string
t7 = tf.constant("hello")             # shape=(), dtype=string
t8 = tf.constant([b"hello", b"world"])  # shape=(2,), dtype=string
```

**Important**: When creating a tensor from a NumPy array, TensorFlow may share
the underlying buffer. Modifying the NumPy array after tensor creation can
affect the tensor value:

```python
import numpy as np
a = np.array([1, 2, 3])
b = tf.constant(a)
a[0] = 4
print(b)  # May print [4, 2, 3] due to buffer sharing
```

### tf.zeros

Creates a tensor of zeros with the specified shape and dtype.

```python
tf.zeros([2, 3])               # [[0, 0, 0], [0, 0, 0]], float32
tf.zeros([2, 3], tf.int32)    # [[0, 0, 0], [0, 0, 0]], int32
tf.zeros_like(tf.constant([[1, 2], [3, 4]]))  # [[0, 0], [0, 0]], int32
```

### tf.ones

Creates a tensor of ones with the specified shape and dtype.

```python
tf.ones([2, 3])                # [[1, 1, 1], [1, 1, 1]], float32
tf.ones([2, 3], tf.float64)   # [[1., 1., 1.], [1., 1., 1.]], float64
tf.ones_like(tf.constant([[1, 2], [3, 4]]))   # [[1, 1], [1, 1]], int32
```

### tf.fill

Creates a tensor filled with a scalar value.

```python
tf.fill([2, 3], 9)             # [[9, 9, 9], [9, 9, 9]]
tf.fill([3], 3.14)             # [3.14, 3.14, 3.14]
```

Unlike `tf.constant(value, shape=shape)`, `tf.fill` evaluates at runtime and
supports dynamic shapes.

### tf.range

Creates a sequence of numbers (1-D tensor).

```python
tf.range(5)                    # [0, 1, 2, 3, 4]
tf.range(2, 5)                 # [2, 3, 4]
tf.range(1, 10, 3)             # [1, 4, 7]
tf.range(5, 1, -1)             # [5, 4, 3, 2]
tf.range(5, dtype=tf.float32)  # [0., 1., 2., 3., 4.]
```

### tf.linspace

Creates a sequence of evenly spaced values.

```python
tf.linspace(0.0, 1.0, 5)      # [0.0, 0.25, 0.5, 0.75, 1.0]
tf.linspace(10.0, 12.0, 3)    # [10.0, 11.0, 12.0]
```

### Random Tensors

```python
# Uniform distribution
tf.random.uniform([2, 3], minval=0, maxval=10, dtype=tf.int32)
tf.random.uniform([2, 3], minval=-1.0, maxval=1.0)

# Normal distribution
tf.random.normal([2, 3], mean=0.0, stddev=1.0)

# Truncated normal (values beyond 2 stddev are re-drawn)
tf.random.truncated_normal([2, 3], mean=0.0, stddev=1.0)

# Shuffle a tensor along the first dimension
tf.random.shuffle(tf.range(10))

# Random categorical (multinomial)
tf.random.categorical(tf.math.log([[0.1, 0.5, 0.4]]), 5)

# Set global random seed for reproducibility
tf.random.set_seed(42)

# Stateless random (deterministic given seed)
tf.random.stateless_uniform([2, 3], seed=[1, 2])
tf.random.stateless_normal([2, 3], seed=[1, 2])
```

### tf.convert_to_tensor

Converts various Python objects to tensors. This is called implicitly by most
operations that accept tensor inputs.

```python
t1 = tf.convert_to_tensor([1, 2, 3])           # int32 tensor
t2 = tf.convert_to_tensor(np.array([1, 2, 3]))  # int64 tensor (NumPy default)
t3 = tf.convert_to_tensor(3.14)                  # float32 scalar
t4 = tf.convert_to_tensor([1, 2, 3], dtype=tf.float32)  # explicit dtype
```

### Special Value Tensors

```python
# Identity matrix
tf.eye(3)                       # 3x3 identity
tf.eye(3, num_columns=4)        # 3x4 with diagonal ones

# Diagonal
tf.linalg.diag([1, 2, 3])      # [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

# Meshgrid
x, y = tf.meshgrid(tf.range(3), tf.range(4))
```

---

## Data Types

### tf.DType Class

The `tf.dtypes.DType` class represents the type of elements in a tensor.
Defined in `tensorflow/python/framework/dtypes.py`.

```python
@tf_export("dtypes.DType", "DType")
class DType(_dtypes.DType, trace.TraceType, trace_type.Serializable):
```

### Available Data Types

#### Floating Point Types

| TF DType | Description | Precision | Range | NumPy Equivalent |
|----------|------------|-----------|-------|-----------------|
| `tf.float16` / `tf.half` | Half precision (16-bit) | 10-bit mantissa, 5-bit exponent | +/- 65504 | `np.float16` |
| `tf.float32` | Single precision (32-bit) | 23-bit mantissa, 8-bit exponent | +/- 3.4e38 | `np.float32` |
| `tf.float64` / `tf.double` | Double precision (64-bit) | 52-bit mantissa, 11-bit exponent | +/- 1.8e308 | `np.float64` |
| `tf.bfloat16` | Brain float (16-bit) | 7-bit mantissa, 8-bit exponent | +/- 3.4e38 | `ml_dtypes.bfloat16` |

#### Experimental Floating Point Types

| TF DType | Description |
|----------|------------|
| `tf.experimental.float8_e5m2` | 8-bit float, 5 exponent bits, 2 mantissa bits |
| `tf.experimental.float8_e4m3fn` | 8-bit float, 4 exponent bits, 3 mantissa bits, extended range |
| `tf.experimental.float8_e4m3fnuz` | 8-bit float, 4 exponent bits, 3 mantissa bits, no inf |
| `tf.experimental.float8_e5m2fnuz` | 8-bit float, 5 exponent bits, 2 mantissa bits, no inf |
| `tf.experimental.float8_e4m3b11fnuz` | 8-bit float, 4 exponent bits, 3 mantissa bits, 11-bit bias |
| `tf.experimental.float4_e2m1fn` | 4-bit float, 2 exponent bits, 1 mantissa bit, no inf/NaN |

#### Integer Types

| TF DType | Description | Range | NumPy Equivalent |
|----------|------------|-------|-----------------|
| `tf.int8` | Signed 8-bit | -128 to 127 | `np.int8` |
| `tf.int16` | Signed 16-bit | -32768 to 32767 | `np.int16` |
| `tf.int32` | Signed 32-bit | -2^31 to 2^31-1 | `np.int32` |
| `tf.int64` | Signed 64-bit | -2^63 to 2^63-1 | `np.int64` |
| `tf.uint8` | Unsigned 8-bit | 0 to 255 | `np.uint8` |
| `tf.uint16` | Unsigned 16-bit | 0 to 65535 | `np.uint16` |
| `tf.uint32` | Unsigned 32-bit | 0 to 2^32-1 | `np.uint32` |
| `tf.uint64` | Unsigned 64-bit | 0 to 2^64-1 | `np.uint64` |

#### Experimental Sub-byte Integer Types

| TF DType | Description |
|----------|------------|
| `tf.experimental.int4` | Signed 4-bit integer |
| `tf.experimental.uint4` | Unsigned 4-bit integer |
| `tf.experimental.int2` | Signed 2-bit integer |
| `tf.experimental.uint2` | Unsigned 2-bit integer |

#### Complex Types

| TF DType | Description | NumPy Equivalent |
|----------|------------|-----------------|
| `tf.complex64` | 64-bit complex (two float32) | `np.complex64` |
| `tf.complex128` | 128-bit complex (two float64) | `np.complex128` |

#### Other Types

| TF DType | Description |
|----------|------------|
| `tf.bool` | Boolean |
| `tf.string` | Variable-length byte string |
| `tf.resource` | Handle to a mutable resource |
| `tf.variant` | Data of arbitrary type |

#### Quantized Types

| TF DType | Description |
|----------|------------|
| `tf.qint8` | Signed quantized 8-bit |
| `tf.quint8` | Unsigned quantized 8-bit |
| `tf.qint16` | Signed quantized 16-bit |
| `tf.quint16` | Unsigned quantized 16-bit |
| `tf.qint32` | Signed quantized 32-bit |

### DType Properties

```python
dtype = tf.float32

dtype.name          # 'float32'
dtype.base_dtype    # float32 (non-reference base type)
dtype.real_dtype    # float32 (real part dtype)
dtype.is_integer    # False
dtype.is_floating   # True
dtype.is_bool       # False
dtype.is_complex    # False
dtype.is_quantized  # False
dtype.is_unsigned   # False
dtype.min           # -3.4028235e+38
dtype.max           # 3.4028235e+38
dtype.as_numpy_dtype  # <class 'numpy.float32'>
dtype.size          # 4 (bytes)
```

### Type Casting

```python
# tf.cast - convert tensor dtype
x = tf.constant([1, 2, 3], dtype=tf.int32)
y = tf.cast(x, tf.float32)   # [1.0, 2.0, 3.0]

# Casting to lower precision (truncation/rounding)
z = tf.cast(y, tf.int32)     # [1, 2, 3]

# Boolean casting
b = tf.cast(tf.constant([True, False, True]), tf.int32)  # [1, 0, 1]

# tf.dtypes.complex - create complex from real and imag
c = tf.complex(tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0]))

# tf.dtypes.complex128 and complex64 conversion
x = tf.constant([1.0 + 2.0j])
print(tf.math.real(x))  # [1.0]
print(tf.math.imag(x))  # [2.0]
```

### Type Promotion Rules

When operations receive inputs of different dtypes, TensorFlow applies
type promotion (similar to NumPy):

1. **Integer + Float** -> Float (wider of the float type)
2. **Integer + Integer** -> Wider integer type
3. **Float + Float** -> Wider float type
4. **Integer + Boolean** -> Integer type
5. **Same type** -> Same type

```python
# int32 + float32 -> float32
tf.constant(1, tf.int32) + tf.constant(1.0, tf.float32)  # float32

# int32 + int64 -> int64
tf.constant(1, tf.int32) + tf.constant(1, tf.int64)  # int64
```

### tf.as_dtype

Converts various Python types to `tf.DType`:

```python
tf.as_dtype('float')       # tf.float32
tf.as_dtype(np.int32)      # tf.int32
tf.as_dtype(1)             # tf.float64 (enum value for DT_DOUBLE)
tf.as_dtype(tf.int32)      # tf.int32
```

---

## Tensor Shapes

### tf.TensorShape Class

Defined in `tensorflow/python/framework/tensor_shape.py`. Represents the
static shape of a tensor.

```python
@tf_export("TensorShape")
class TensorShape(trace.TraceType, trace_type.Serializable):
```

### Shape Representations

A `TensorShape` can represent three levels of knowledge:

1. **Fully-known shape**: All dimensions are known
   ```python
   TensorShape([2, 3])       # 2x3 matrix
   TensorShape([])           # scalar
   ```

2. **Partially-known shape**: Some dimensions unknown (`None`)
   ```python
   TensorShape([None, 3])    # unknown batch, 3 features
   TensorShape([None, None]) # 2D, both dimensions unknown
   ```

3. **Unknown shape**: Even the rank is unknown
   ```python
   TensorShape(None)         # completely unknown
   ```

### Creating TensorShapes

```python
# From a list
shape1 = tf.TensorShape([2, 3])

# From a tuple
shape2 = tf.TensorShape((None, 3))

# From another TensorShape
shape3 = tf.TensorShape(shape1)

# Unknown shape
shape4 = tf.TensorShape(None)

# Scalar shape
shape5 = tf.TensorShape([])
```

### TensorShape Properties and Methods

```python
shape = tf.TensorShape([2, 3, 4])

shape.rank            # 3 (number of dimensions)
shape.ndims           # 3 (alias for rank)
shape.dims            # [Dimension(2), Dimension(3), Dimension(4)]
shape.as_list()       # [2, 3, 4]
shape.num_elements()  # 24
shape.is_fully_defined()  # True
shape[0]              # 2 (integer in TF2, Dimension in TF1)
shape[1:]             # TensorShape([3, 4])
```

### Shape Compatibility

```python
a = tf.TensorShape([2, 3])
b = tf.TensorShape([None, 3])
c = tf.TensorShape([2, None])
d = tf.TensorShape(None)

a.is_compatible_with(b)  # True
a.is_compatible_with(c)  # True
a.is_compatible_with(d)  # True
b.is_compatible_with(d)  # True

# Merge shapes (combines information)
b.merge_with(c)  # TensorShape([2, 3])

# Concatenation
tf.TensorShape([2]) + tf.TensorShape([3, 4])  # TensorShape([2, 3, 4])
```

### Shape Manipulation Functions

```python
# tf.shape - returns the dynamic shape as a tensor
t = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.shape(t)  # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3])>

# tf.reshape
tf.reshape(t, [3, 2])   # [[1, 2], [3, 4], [5, 6]]
tf.reshape(t, [-1])     # [1, 2, 3, 4, 5, 6]  (-1 infers dimension)
tf.reshape(t, [2, -1])  # [[1, 2, 3], [4, 5, 6]]

# tf.expand_dims - add a dimension
tf.expand_dims(t, 0)    # shape (1, 2, 3)
tf.expand_dims(t, 1)    # shape (2, 1, 3)
tf.expand_dims(t, -1)   # shape (2, 3, 1)

# tf.squeeze - remove dimensions of size 1
t2 = tf.constant([[[1], [2], [3]]])  # shape (1, 3, 1)
tf.squeeze(t2)         # shape (3,)
tf.squeeze(t2, axis=0) # shape (3, 1)
tf.squeeze(t2, axis=2) # shape (1, 3)

# tf.broadcast_to
tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3])
# [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

# tf.ensure_shape - runtime shape validation + static info update
@tf.function
def my_fn(x):
    x = tf.ensure_shape(x, [None, 3])
    return x
```

### tf.TensorSpec

A type specification for tensors, used in `tf.function` input signatures:

```python
spec = tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name='input')
spec.shape   # TensorShape([None, 3])
spec.dtype   # tf.float32
spec.name    # 'input'

# Use as function input signature
@tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
def process(x):
    return x * 2

# From an existing tensor
t = tf.constant([1.0, 2.0, 3.0])
spec = tf.TensorSpec.from_tensor(t)
# TensorSpec(shape=(3,), dtype=tf.float32, name=None)
```

### BoundedTensorSpec

A `TensorSpec` with minimum and maximum value constraints:

```python
spec = tf.BoundedTensorSpec(
    shape=(3, 5), dtype=tf.int32,
    minimum=0, maximum=2
)
spec.minimum  # array(0, dtype=int32)
spec.maximum  # array(2, dtype=int32)
```

---

## Broadcasting Rules

Broadcasting allows operations on tensors with different but compatible shapes.

### Rules

1. If tensors have different ranks, the smaller-rank tensor is padded with
   ones on the left.
2. If any dimension is 1, it is stretched to match the corresponding dimension
   in the other tensor.
3. If any dimension is `None` (unknown), it is treated as compatible with any
   size.
4. If the dimensions are different and neither is 1, an error is raised.

### Examples

```python
# Scalar + Vector
tf.constant(5) + tf.constant([1, 2, 3])
# [6, 7, 8]

# Vector + Matrix
tf.constant([1, 2, 3]) + tf.constant([[10, 20, 30], [40, 50, 60]])
# [[11, 22, 33], [41, 52, 63]]

# Column vector + Row vector
tf.constant([[1], [2], [3]]) + tf.constant([[10, 20, 30]])
# [[11, 21, 31], [12, 22, 32], [13, 23, 33]]

# Shape (1, 3) + Shape (3, 1)
a = tf.ones([1, 3])  # [[1, 1, 1]]
b = tf.ones([3, 1])  # [[1], [1], [1]]
a + b                 # [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

# Shape (8, 1, 6, 1) + Shape (7, 1, 5) -> Shape (8, 7, 6, 5)
```

### Broadcasting Helper Functions

```python
# Check if shapes are broadcastable
tf.broadcast_static_shape(
    tf.TensorShape([8, 1, 6, 1]),
    tf.TensorShape([7, 1, 5])
)  # TensorShape([8, 7, 6, 5])

# Get dynamic broadcast shape
tf.broadcast_dynamic_shape(
    tf.shape(a), tf.shape(b)
)
```

---

## Indexing and Slicing

### Basic Indexing

TensorFlow supports NumPy-style indexing:

```python
t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Single element
t[0, 1]       # 2
t[-1, -1]     # 9

# Row/column selection
t[0]          # [1, 2, 3]
t[:, 1]       # [2, 5, 8]

# Slicing
t[0:2]        # [[1, 2, 3], [4, 5, 6]]
t[:, 1:]      # [[2, 3], [5, 6], [8, 9]]
t[::2]        # [[1, 2, 3], [7, 8, 9]]
t[::-1]       # [[7, 8, 9], [4, 5, 6], [1, 2, 3]]

# Ellipsis
t[..., 0]     # [1, 4, 7]  (same as t[:, 0])
t[0, ...]     # [1, 2, 3]  (same as t[0])

# tf.newaxis / None
t[:, tf.newaxis, :]  # shape (3, 1, 3)
t[None, ...]         # shape (1, 3, 3)
```

### Advanced Indexing

```python
# Integer array indexing
indices = tf.constant([0, 2])
tf.gather(t, indices)          # [[1, 2, 3], [7, 8, 9]]
tf.gather(t, indices, axis=1)  # [[1, 3], [4, 6], [7, 9]]

# Multi-dimensional gathering
tf.gather_nd(t, [[0, 0], [1, 2], [2, 1]])  # [1, 6, 8]
tf.gather_nd(t, [[[0, 0], [0, 1]], [[2, 0], [2, 2]]])
```

### Boolean Mask

```python
t = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
mask = t > 3                    # [False, False, False, True, True, True, True, True]
tf.boolean_mask(t, mask)        # [4, 5, 6, 7, 8]

# Multi-dimensional boolean mask
t2 = tf.constant([[1, 2], [3, 4], [5, 6]])
mask2 = tf.constant([True, False, True])
tf.boolean_mask(t2, mask2)      # [[1, 2], [5, 6]]
```

### Scatter Operations

```python
# scatter_nd_update
ref = tf.Variable(tf.zeros([8]))
indices = tf.constant([[1], [3], [5], [7]])
updates = tf.constant([10, 20, 30, 40])
ref.scatter_nd_update(indices, updates)
# [0, 10, 0, 20, 0, 30, 0, 40]

# scatter_nd_add
ref = tf.Variable(tf.ones([8]))
ref.scatter_nd_add(indices, updates)
# [1, 11, 1, 21, 1, 31, 1, 41]
```

### tf.where

```python
# Conditional selection
condition = tf.constant([True, False, True, False])
x = tf.constant([1, 2, 3, 4])
y = tf.constant([10, 20, 30, 40])
tf.where(condition, x, y)  # [1, 20, 3, 40]

# Find indices of True elements
tf.where(tf.constant([[True, False], [False, True]]))
# [[0, 0], [1, 1]]
```

---

## Math Operations

### Element-wise Arithmetic

```python
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# Basic arithmetic
tf.add(a, b)          # [5.0, 7.0, 9.0]     (also a + b)
tf.subtract(a, b)     # [-3.0, -3.0, -3.0]  (also a - b)
tf.multiply(a, b)     # [4.0, 10.0, 18.0]   (also a * b)
tf.divide(a, b)       # [0.25, 0.4, 0.5]    (also a / b)
tf.math.floordiv(a, b)  # [0.0, 0.0, 0.0]  (also a // b)
tf.math.mod(a, b)     # [1.0, 2.0, 3.0]     (also a % b)
tf.pow(a, b)          # [1.0, 32.0, 729.0]  (also a ** b)

# Scalar operations
tf.add(a, 10)         # [11.0, 12.0, 13.0]
a * 2                 # [2.0, 4.0, 6.0]

# Negative
tf.negative(a)        # [-1.0, -2.0, -3.0]  (also -a)

# Absolute value
tf.abs(tf.constant([-1.5, 2.0, -3.5]))  # [1.5, 2.0, 3.5]

# Sign
tf.sign(tf.constant([-2.0, 0.0, 3.0]))  # [-1.0, 0.0, 1.0]

# Reciprocal
tf.math.reciprocal(a)  # [1.0, 0.5, 0.333]
```

### Exponential and Logarithmic

```python
x = tf.constant([1.0, 2.0, 3.0])

tf.exp(x)             # [2.718, 7.389, 20.086]
tf.math.exp2(x)       # [2.0, 4.0, 8.0]
tf.math.expm1(x)      # [1.718, 6.389, 19.086] (exp(x) - 1)

tf.math.log(x)        # [0.0, 0.693, 1.099]
tf.math.log2(x)       # [0.0, 1.0, 1.585]
tf.math.log10(x)      # [0.0, 0.301, 0.477]
tf.math.log1p(x)      # [0.693, 1.099, 1.386] (log(1 + x))

# Sigmoid
tf.math.sigmoid(x)    # [0.731, 0.881, 0.953]

# Log-sigmoid
tf.math.log_sigmoid(x)
```

### Power and Root

```python
x = tf.constant([1.0, 4.0, 9.0])

tf.sqrt(x)            # [1.0, 2.0, 3.0]
tf.math.rsqrt(x)      # [1.0, 0.5, 0.333] (1/sqrt)
tf.math.square(x)     # [1.0, 16.0, 81.0]
tf.pow(x, 3)          # [1.0, 64.0, 729.0]
tf.math.xlogy(x, x)   # x * log(y) where x != 0
```

### Trigonometric Functions

```python
x = tf.constant([0.0, 1.0, 3.14159])

tf.sin(x)       # [0.0, 0.841, 0.0]
tf.cos(x)       # [1.0, 0.540, -1.0]
tf.tan(x)       # [0.0, 1.557, -0.0]

# Inverse trigonometric
tf.asin(x / 2)  # arcsin
tf.acos(x / 2)  # arccos
tf.atan(x)      # arctan
tf.math.atan2(x, x)  # arctan2(y, x)

# Hyperbolic
tf.sinh(x)
tf.cosh(x)
tf.tanh(x)      # Common activation function
tf.math.asinh(x)
tf.math.acosh(x)
tf.math.atanh(x)
```

### Rounding and Clipping

```python
x = tf.constant([-1.7, -1.2, 0.0, 1.2, 1.7])

tf.round(x)       # [-2.0, -1.0, 0.0, 1.0, 2.0]
tf.math.ceil(x)   # [-1.0, -1.0, 0.0, 2.0, 2.0]
tf.math.floor(x)  # [-2.0, -2.0, 0.0, 1.0, 1.0]

# Clipping
tf.clip_by_value(x, -1.0, 1.0)  # [-1.0, -1.0, 0.0, 1.0, 1.0]
tf.clip_by_norm(x, 2.0)          # Clip by global norm
tf.clip_by_global_norm([x], 5.0)  # Clip multiple tensors by global norm
```

### Reduction Operations

```python
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Sum
tf.reduce_sum(x)           # 21.0 (all elements)
tf.reduce_sum(x, axis=0)   # [5.0, 7.0, 9.0] (along rows)
tf.reduce_sum(x, axis=1)   # [6.0, 15.0] (along columns)
tf.reduce_sum(x, keepdims=True)  # [[21.0]] (keep dimensions)

# Mean
tf.reduce_mean(x)          # 3.5
tf.reduce_mean(x, axis=0)  # [2.5, 3.5, 4.5]

# Product
tf.reduce_prod(x)          # 720.0

# Min and Max
tf.reduce_min(x)           # 1.0
tf.reduce_max(x)           # 6.0
tf.reduce_min(x, axis=0)   # [1.0, 2.0, 3.0]

# ArgMin and ArgMax
tf.argmax(x, axis=0)       # [1, 1, 1]
tf.argmin(x, axis=0)       # [0, 0, 0]

# LogSumExp (numerically stable)
tf.reduce_logsumexp(x)     # log(sum(exp(x)))

# Any and All (for boolean tensors)
b = tf.constant([[True, False], [True, True]])
tf.reduce_any(b)           # True
tf.reduce_all(b)           # False
tf.reduce_all(b, axis=1)   # [False, True]

# Variance and Standard Deviation
tf.math.reduce_variance(x)    # variance
tf.math.reduce_std(x)         # standard deviation
```

### Comparison Operations

```python
a = tf.constant([1, 2, 3])
b = tf.constant([1, 3, 2])

tf.equal(a, b)            # [True, False, False]
tf.not_equal(a, b)        # [False, True, True]
tf.greater(a, b)          # [False, False, True]
tf.greater_equal(a, b)    # [True, False, True]
tf.less(a, b)             # [False, True, False]
tf.less_equal(a, b)       # [True, True, False]

# Works across types
tf.equal(tf.constant(1), tf.constant(1.0))  # True
```

### Logical Operations

```python
a = tf.constant([True, True, False, False])
b = tf.constant([True, False, True, False])

tf.logical_and(a, b)  # [True, False, False, False]
tf.logical_or(a, b)   # [True, True, True, False]
tf.logical_not(a)      # [False, False, True, True]
tf.logical_xor(a, b)   # [False, True, True, False]
```

### Cumulative Operations

```python
x = tf.constant([1, 2, 3, 4, 5])

tf.cumsum(x)            # [1, 3, 6, 10, 15]
tf.cumsum(x, reverse=True)  # [15, 14, 12, 9, 5]
tf.math.cumprod(x)      # [1, 2, 6, 24, 120]
```

---

## Matrix Operations

### Basic Matrix Operations

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiplication
tf.matmul(a, b)          # [[19, 22], [43, 50]]
a @ b                    # Same (Python 3.5+)

# Batched matrix multiplication
a_batch = tf.random.normal([10, 3, 4])
b_batch = tf.random.normal([10, 4, 5])
tf.matmul(a_batch, b_batch)  # shape [10, 3, 5]

# Transpose
tf.transpose(a)          # [[1, 3], [2, 4]]
tf.transpose(a, perm=[1, 0])  # explicit permutation

# Conjugate transpose (for complex matrices)
c = tf.constant([[1+2j, 3+4j]])
tf.linalg.adjoint(c)     # [[1-2j], [3-4j]]

# Matrix determinant
tf.linalg.det(a)         # -2.0

# Matrix inverse
tf.linalg.inv(a)         # [[-2.0, 1.0], [1.5, -0.5]]

# Trace
tf.linalg.trace(a)       # 5.0
```

### Matrix Decompositions

```python
m = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# QR decomposition
q, r = tf.linalg.qr(m)

# SVD (Singular Value Decomposition)
s, u, v = tf.linalg.svd(m)

# Cholesky decomposition (for positive definite matrices)
pd = tf.constant([[4.0, 2.0], [2.0, 3.0]])
tf.linalg.cholesky(pd)   # Lower triangular

# Eigenvalue decomposition
tf.linalg.eigvalsh(pd)   # Eigenvalues of Hermitian matrix

# LU decomposition
lu, p = tf.linalg.lu(m)
```

### Solving Linear Systems

```python
# Solve Ax = b
A = tf.constant([[3.0, 1.0], [1.0, 2.0]])
b = tf.constant([9.0, 8.0])
tf.linalg.solve(A, tf.expand_dims(b, -1))  # [[2.0], [3.0]]

# Triangular solve
tf.linalg.triangular_solve(A, b, lower=True)

# Least squares
tf.linalg.lstsq(m, tf.constant([1.0, 2.0, 3.0]))
```

### Norms

```python
v = tf.constant([3.0, 4.0])
m = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# Vector norm
tf.norm(v)               # 5.0 (L2 norm)
tf.norm(v, ord=1)        # 7.0 (L1 norm)
tf.norm(v, ord=np.inf)   # 4.0 (infinity norm)

# Matrix norm (Frobenius)
tf.norm(m)               # sqrt(30) (Frobenius norm)

# Normalized
tf.linalg.l2_normalize(v)  # [0.6, 0.8]
```

### Einsum (Einstein Summation)

```python
# Matrix multiplication
tf.einsum('ij,jk->ik', a, b)  # Same as tf.matmul(a, b)

# Dot product
tf.einsum('i,i->', v1, v2)

# Outer product
tf.einsum('i,j->ij', v1, v2)

# Batch matrix multiplication
tf.einsum('bij,bjk->bik', A_batch, B_batch)

# Trace
tf.einsum('ii->', a)

# Diagonal extraction
tf.einsum('ii->i', a)
```

---

## Tensor Manipulation

### Reshaping

```python
t = tf.range(12)

# Reshape
tf.reshape(t, [3, 4])    # [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
tf.reshape(t, [2, -1])   # [[0,1,2,3,4,5],[6,7,8,9,10,11]]
tf.reshape(t, [-1, 3])   # [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]

# Transpose
t2 = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.transpose(t2)         # [[1, 4], [2, 5], [3, 6]]
tf.transpose(t2, perm=[1, 0])  # Same as above

# Move axes
tf.transpose(t2, perm=[1, 0])  # Swap axes 0 and 1
```

### Concatenation and Stacking

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Concatenate along existing axis
tf.concat([a, b], axis=0)  # [[1,2],[3,4],[5,6],[7,8]]  shape (4, 2)
tf.concat([a, b], axis=1)  # [[1,2,5,6],[3,4,7,8]]      shape (2, 4)

# Stack along a new axis
tf.stack([a, b], axis=0)   # shape (2, 2, 2)
tf.stack([a, b], axis=2)   # shape (2, 2, 2)

# Unstack
t = tf.constant([[1, 2], [3, 4], [5, 6]])
tf.unstack(t, axis=0)  # [tf.constant([1,2]), tf.constant([3,4]), tf.constant([5,6])]
tf.unstack(t, axis=1)  # [tf.constant([1,3,5]), tf.constant([2,4,6])]
```

### Splitting

```python
t = tf.range(10)

# Split into equal parts
tf.split(t, 2)       # [[0,1,2,3,4], [5,6,7,8,9]]
tf.split(t, 5)       # [[0,1], [2,3], [4,5], [6,7], [8,9]]

# Split with specific sizes
tf.split(t, [3, 3, 4])  # [[0,1,2], [3,4,5], [6,7,8,9]]
```

### Tiling and Repeating

```python
# Tile (repeat entire tensor)
t = tf.constant([[1, 2], [3, 4]])
tf.tile(t, [2, 3])
# [[1,2,1,2,1,2],[3,4,3,4,3,4],[1,2,1,2,1,2],[3,4,3,4,3,4]]

# Repeat (repeat each element)
tf.repeat(t, 3, axis=0)  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]]
tf.repeat(t, 3, axis=1)  # [[1,1,1,2,2,2],[3,3,3,4,4,4]]
```

### Reversing and Rolling

```python
t = tf.constant([[1, 2, 3], [4, 5, 6]])

# Reverse
tf.reverse(t, [0])        # [[4,5,6],[1,2,3]]
tf.reverse(t, [1])        # [[3,2,1],[6,5,4]]
tf.reverse(t, [0, 1])     # [[6,5,4],[3,2,1]]

# Roll
tf.roll(t, shift=1, axis=1)  # [[3,1,2],[6,4,5]]
```

### Padding

```python
t = tf.constant([[1, 2], [3, 4]])

# Pad with zeros
tf.pad(t, [[1, 1], [2, 2]])
# [[0,0,0,0,0,0],[0,0,1,2,0,0],[0,0,3,4,0,0],[0,0,0,0,0,0]]

# Constant padding
tf.pad(t, [[1, 1], [1, 1]], constant_values=9)
# [[9,9,9,9],[9,1,2,9],[9,3,4,9],[9,9,9,9]]

# Reflect/symmetric padding
tf.pad(t, [[1, 1], [1, 1]], mode='REFLECT')
# [[4,3,4,3],[2,1,2,1],[4,3,4,3]]
```

### Gather and Scatter

```python
params = tf.constant([10, 20, 30, 40, 50])

# gather
tf.gather(params, [0, 2, 4])   # [10, 30, 50]

# gather with axis
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])
tf.gather(matrix, [0, 2], axis=0)  # [[1,2],[5,6]]

# gather_nd (multi-dimensional indices)
tf.gather_nd(matrix, [[0, 0], [1, 1], [2, 0]])  # [1, 4, 5]

# scatter_nd
indices = tf.constant([[0], [2]])
updates = tf.constant([100, 200])
tf.scatter_nd(indices, updates, [4])  # [100, 0, 200, 0]
```

### Boolean Mask

```python
t = tf.constant([[1, 2], [3, 4], [5, 6]])
mask = tf.constant([True, False, True])

tf.boolean_mask(t, mask)  # [[1, 2], [5, 6]]
```

### Unique and Segment Operations

```python
# Unique values
x = tf.constant([1, 2, 3, 2, 1, 3])
y, idx = tf.unique(x)
# y = [1, 2, 3], idx = [0, 1, 2, 1, 0, 2]

# Segment sum
data = tf.constant([1, 2, 3, 4, 5])
segment_ids = tf.constant([0, 0, 1, 1, 2])
tf.math.segment_sum(data, segment_ids)  # [3, 7, 5]
tf.math.segment_mean(data, segment_ids) # [1.5, 3.5, 5.0]
tf.math.segment_max(data, segment_ids)  # [2, 4, 5]
tf.math.segment_min(data, segment_ids)  # [1, 3, 5]
```

---

## Type Casting and Numeric Conversion

### tf.cast

```python
# Basic casting
x_int = tf.constant([1, 2, 3], dtype=tf.int32)
x_float = tf.cast(x_int, tf.float32)  # [1.0, 2.0, 3.0]

# String to number
tf.strings.to_number(tf.constant(["1.5", "2.3", "3.7"]), tf.float32)

# Number to string
tf.as_string(tf.constant([1, 2, 3]))

# Saturating cast (clamp to target type's range)
tf.dtypes.saturate_cast(tf.constant(300), tf.uint8)  # 255

# Bitwise cast (reinterpret bits)
tf.bitcast(tf.constant(1, tf.int32), tf.float32)  # 1.4012985e-45
```

### Numeric Conversion Rules

When casting between types:
- **Int to Float**: Exact conversion (value preserved)
- **Float to Int**: Truncation (decimal part discarded)
- **Large Int to Small Int**: Wrapping (overflow behavior)
- **Float to Smaller Float**: May lose precision
- **Bool to Int**: True -> 1, False -> 0
- **Int to Bool**: 0 -> False, nonzero -> True

---

## String Tensor Operations

### Creation

```python
# String scalar
s1 = tf.constant("hello")

# String vector
s2 = tf.constant(["hello", "world", "tensorflow"])

# Byte strings
s3 = tf.constant([b"hello", b"world"])
```

### String Operations (in `tf.strings` module)

```python
s = tf.constant(["Hello World", "TensorFlow"])

# Case conversion
tf.strings.lower(s)       # ["hello world", "tensorflow"]
tf.strings.upper(s)       # ["HELLO WORLD", "TENSORFLOW"]

# Joining
tf.strings.join(["hello", "world"], separator=" ")  # "hello world"
tf.strings.join([["a", "b"], ["1", "2"]])  # ["a1", "b2"]

# Splitting
tf.strings.split("a,b,c", sep=",")  # ["a", "b", "c"]
tf.strings.split(["a,b,c", "x,y"], sep=",")  # RaggedTensor

# Length
tf.strings.length(s)      # [11, 10]

# Substring
tf.strings.substr(s, 0, 5)  # ["Hello", "Tensor"]

# Pattern matching
tf.strings.regex_full_match(s, "H.*")

# Pattern replacement
tf.strings.regex_replace(s, "o", "0")  # ["Hell0 W0rld", "TensorFl0w"]

# Format
tf.strings.format("value: {}", [42])

# Number conversion
tf.strings.to_number(tf.constant(["1.5", "2.5"]), tf.float32)  # [1.5, 2.5]
tf.strings.as_string(tf.constant([1, 2, 3]))  # ["1", "2", "3"]

# Encoding/decoding
tf.strings.unicode_encode(tf.constant([[72, 101, 108, 108, 111]]), "UTF-8")
tf.strings.unicode_decode(tf.constant("Hello"), "UTF-8")
```

---

## Complex Number Operations

### Creation

```python
# From real and imaginary parts
z = tf.complex(tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0]))
# [1+3j, 2+4j]

# From Python complex literals
z = tf.constant([1+2j, 3+4j])  # complex64

# Explicit dtype
z = tf.constant([1+2j], dtype=tf.complex128)
```

### Operations

```python
z = tf.constant([3+4j, 5+12j])

# Real and imaginary parts
tf.math.real(z)      # [3.0, 5.0]
tf.math.imag(z)      # [4.0, 12.0]

# Conjugate
tf.math.conj(z)      # [3-4j, 5-12j]

# Magnitude (absolute value)
tf.abs(z)            # [5.0, 13.0]

# Argument (angle)
tf.math.angle(z)     # [0.927, 1.176] (radians)

# Complex arithmetic
z1 = tf.constant([1+2j])
z2 = tf.constant([3+4j])
z1 + z2              # [4+6j]
z1 * z2              # [-5+10j]
z1 / z2              # [0.44+0.08j]
```

---

## Quantized Tensor Operations

### Quantized Types

TensorFlow provides special types for quantized inference:

| Type | Description | Use Case |
|------|------------|----------|
| `tf.qint8` | Signed quantized 8-bit | Quantized weights |
| `tf.quint8` | Unsigned quantized 8-bit | Quantized activations |
| `tf.qint16` | Signed quantized 16-bit | Higher precision quantized |
| `tf.quint16` | Unsigned quantized 16-bit | Higher precision quantized |
| `tf.qint32` | Signed quantized 32-bit | Quantized accumulators |

### Quantization Operations

```python
# Quantize a float tensor
x = tf.constant([0.5, 1.0, 1.5, 2.0])
q_x, min_val, max_val = tf.quantization.quantize(
    x, min_value=0.0, max_value=2.0, dtype=tf.quint8
)

# Dequantize back to float
dq_x = tf.quantization.dequantize(q_x, min_val, max_val, dtype=tf.float32)

# Fake quantization (for training-aware quantization)
fq_x = tf.quantization.fake_quant_with_min_max_args(
    x, min=-2.0, max=2.0, num_bits=8
)
```

---

## Ragged and Sparse Tensor Basics

### RaggedTensors

Ragged tensors represent tensors with variable-length dimensions:

```python
# From a list of variable-length lists
rt = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
print(rt.shape)        # (3, None)
print(rt.to_tensor())  # [[1,2,3],[4,0,0],[5,6,0]]

# From row_splits
rt = tf.RaggedTensor.from_row_splits(
    values=[1, 2, 3, 4, 5, 6],
    row_splits=[0, 3, 3, 5, 6]
)

# Operations
rt2 = tf.ragged.constant([[1, 2], [3]])
tf.concat([rt, rt2], axis=0)
tf.reduce_sum(rt, axis=1)  # [6, 4, 11]

# Conversion
rt.to_tensor()       # Ragged -> Dense (pad with default value)
rt.to_sparse()       # Ragged -> Sparse
```

### SparseTensors

Sparse tensors represent tensors where most values are zero:

```python
# Create a sparse tensor
st = tf.sparse.SparseTensor(
    indices=[[0, 0], [1, 2], [2, 1]],
    values=[1.0, 2.0, 3.0],
    dense_shape=[3, 3]
)
# Represents:
# [[1, 0, 0],
#  [0, 0, 2],
#  [0, 3, 0]]

# Convert to dense
dense = tf.sparse.to_dense(st)

# Sparse operations
tf.sparse.reduce_sum(st)        # 6.0
tf.sparse.transpose(st)         # Transpose
tf.sparse.reshape(st, [1, 9])   # Reshape

# Sparse-dense multiplication
tf.sparse.sparse_dense_matmul(st, tf.ones([3, 2]))
```

### SparseTensor Properties

```python
st.indices    # 2-D int64 tensor of index positions
st.values     # 1-D tensor of non-zero values
st.dense_shape  # 1-D int64 tensor of the dense shape
```

---

## Summary

TensorFlow provides a comprehensive tensor system with rich data type support
(including experimental sub-byte types), flexible shape handling, and extensive
mathematical and manipulation operations. The core abstractions are:

- **tf.Tensor**: Immutable multidimensional array with dtype, shape, and device
- **tf.TensorShape**: Static shape representation supporting fully-known,
  partially-known, and unknown shapes
- **tf.DType**: Type system supporting float, int, complex, string, quantized,
  and experimental sub-byte types
- **Operations**: Over 1000 registered operations covering math, array
  manipulation, linear algebra, signal processing, and more
- **Special tensor types**: Sparse tensors for sparse data, ragged tensors for
  variable-length sequences, quantized tensors for efficient inference

The design follows NumPy conventions where possible for familiarity, while
adding TensorFlow-specific features like symbolic shapes, device placement,
and graph-mode compatibility.
