# TensorFlow Special Tensors Reference

## Table of Contents

1. [SparseTensor](#sparsetensor)
2. [RaggedTensor](#raggedtensor)
3. [StringTensor](#stringtensor)
4. [CompositeTensor](#compositetensor)
5. [Extension Types](#extension-types)
6. [TensorSpec](#tensorspec)
7. [SparseTensorSpec and RaggedTensorSpec](#sparsetensorspec-and-raggedtensorspec)
8. [Registered Types](#registered-types)
9. [tf.TensorArray](#tftensorarray)
10. [Variant Tensors](#variant-tensors)
11. [Optional Values](#optional-values)
12. [Structured Tensors](#structured-tensors)

---

## SparseTensor

### Overview

A `SparseTensor` efficiently represents tensors where most values are zero.
Instead of storing all values, it stores only the non-zero values and their
indices. This is critical for natural language processing, recommendation
systems, and other domains with high-dimensional but sparse data.

### Constructor

```python
tf.sparse.SparseTensor(
    indices,
    values,
    dense_shape
)
```

**indices** (`tf.Tensor` of `int64`):
A 2-D tensor of shape `[N, rank]` where N is the number of non-zero values
and rank is the number of dimensions. Each row specifies the coordinates of
a non-zero value.

**values** (`tf.Tensor`):
A 1-D tensor of shape `[N]` containing the non-zero values.

**dense_shape** (`tf.Tensor` of `int64`):
A 1-D tensor of shape `[rank]` specifying the full shape of the dense tensor.

### Example: Creation

```python
# Create a 3x4 sparse matrix:
# [[0, 0, 3, 0],
#  [0, 0, 0, 0],
#  [1, 0, 0, 7]]
st = tf.sparse.SparseTensor(
    indices=tf.constant([[0, 2], [2, 0], [2, 3]], dtype=tf.int64),
    values=tf.constant([3.0, 1.0, 7.0]),
    dense_shape=tf.constant([3, 4], dtype=tf.int64)
)
```

### Properties

- **`.indices`**: The indices tensor.
- **`.values`**: The values tensor.
- **`.dense_shape`**: The dense shape tensor.
- **`.shape`**: The TensorShape representation.
- **`.dtype`**: Data type of values.
- **`.graph`**: The graph containing the tensors (graph mode only).
- **`.op`**: The operation producing this tensor (graph mode only).

### Conversion Operations

**tf.sparse.to_dense(sp_input, default_value=None, validate_indices=True, name=None)**
Convert a SparseTensor to a dense tensor.
```python
dense = tf.sparse.to_dense(st)
# [[0, 0, 3, 0],
#  [0, 0, 0, 0],
#  [1, 0, 0, 7]]
```

**tf.sparse.from_dense(tensor, name=None)**
Convert a dense tensor to a SparseTensor.
```python
dense = tf.constant([[0, 0, 3], [0, 0, 0]])
st = tf.sparse.from_dense(dense)
```

**tf.sparse.to_indicator(sp_input, vocab_size, name=None)**
Convert to indicator (one-hot) representation.

### Arithmetic Operations

**tf.sparse.add(a, b, threshold=None, name=None)**
Add two SparseTensors element-wise. Values at shared indices are summed.
```python
c = tf.sparse.add(st_a, st_b)
```

**tf.sparse.sparse_dense_matmul(sp_a, b, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)**
Multiply a SparseTensor by a dense tensor.
```python
# sp_a: [M, K] sparse, b: [K, N] dense
result = tf.sparse.sparse_dense_matmul(sp_a, b)  # [M, N] dense
```

### Reduction Operations

**tf.sparse.reduce_sum(sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None)**
Sum of elements across dimensions.
```python
tf.sparse.reduce_sum(st, axis=0)  # Sum along rows
tf.sparse.reduce_sum(st, axis=1)  # Sum along columns
```

**tf.sparse.reduce_max(sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None)**
Maximum across dimensions.

**tf.sparse.reduce_min(sp_input, axis=None, keepdims=None, output_is_sparse=False, name=None)**

### Reshaping Operations

**tf.sparse.reshape(sp_input, shape, name=None)**
Reshape a SparseTensor. Indices are recalculated for the new shape.
```python
reshaped = tf.sparse.reshape(st, [2, 6])
```

**tf.sparse.transpose(sp_input, perm=None, name=None)**
Transpose a SparseTensor.
```python
transposed = tf.sparse.transpose(st, perm=[1, 0])
```

**tf.sparse.expand_dims(sp_input, axis=None, name=None)**
Insert a dimension of size 1.

### Slicing and Splitting

**tf.sparse.slice(sp_input, start, size, name=None)**
Extract a slice from a SparseTensor.
```python
sliced = tf.sparse.slice(st, start=[0, 0], size=[2, 3])
```

**tf.sparse.split(sp_input, num_split, axis, name=None)**
Split a SparseTensor along axis.

**tf.sparse.sparse_slice(sp_input, start, size, name=None)**
Alias for `tf.sparse.slice`.

### Reordering and Validation

**tf.sparse.reorder(sp_input, name=None)**
Reorder a SparseTensor into canonical row-major order.

**tf.sparse.reset_shape(sp_input, new_shape=None, name=None)**
Reset the dense shape of a SparseTensor.

**tf.sparse.fill_empty_rows(sp_input, default_value, name=None)**
Fill empty rows with a default value.

### Segment Operations

**tf.sparse.segment_sum(data, segment_ids, num_segments=None, name=None)**
Sparse segment sum.

**tf.sparse.segment_mean(data, segment_ids, num_segments=None, name=None)**
Sparse segment mean.

**tf.sparse.segment_sqrt_n(data, segment_ids, num_segments=None, name=None)**
Sparse segment sum divided by sqrt(N).

**tf.sparse.segment_mean(data, segment_ids, num_segments=None, name=None)**
Sparse segment mean.

### Filtering and Masking

**tf.sparse.mask(a, mask_indices, name=None)**
Mask elements at specified indices.

**tf.sparse.retain(sp_input, retain_mask, name=None)**
Retain elements where mask is True.

### Comparison Operations

**tf.sparse.maximum(sp_a, sp_b, name=None)**
Element-wise maximum of two SparseTensors.

**tf.sparse.minimum(sp_a, sp_b, name=None)**
Element-wise minimum of two SparseTensors.

### Concatenation

**tf.sparse.concat(axis, sp_inputs, expand_nonconcat_dim=False, name=None)**
Concatenate SparseTensors along a dimension.
```python
combined = tf.sparse.concat(0, [st1, st2])  # Concatenate along axis 0
```

### Special Operations

**tf.sparse.softmax(logits, name=None)**
Apply softmax to a SparseTensor (treating missing values as 0).

---

## RaggedTensor

### Overview

A `RaggedTensor` represents tensors with variable-length dimensions (ragged
dimensions). Unlike dense tensors where each dimension has a fixed size,
RaggedTensors allow different sizes along one or more dimensions.

For example, a batch of sentences where each sentence has a different number
of words can be represented as a RaggedTensor:
```python
# [["Hello", "world"], ["Hi"], ["Greetings", "from", "TensorFlow"]]
```

### Internal Representation

A RaggedTensor is defined by:
- **values**: A flat (or nested) tensor containing all the values.
- **row_splits**: A 1-D integer tensor specifying how to split `values` into rows.
  Element `i` of the output is `values[row_splits[i]:row_splits[i+1]]`.

### Factory Methods

**tf.RaggedTensor.from_row_splits(values, row_splits, name=None)**
Create from values and row split indices.
```python
rt = tf.RaggedTensor.from_row_splits(
    values=tf.constant([1, 2, 3, 4, 5, 6]),
    row_splits=tf.constant([0, 3, 3, 5, 6])
)
# [[1, 2, 3], [], [4, 5], [6]]
```

**tf.RaggedTensor.from_row_lengths(values, row_lengths, name=None)**
Create from values and row lengths.
```python
rt = tf.RaggedTensor.from_row_lengths(
    values=tf.constant([1, 2, 3, 4, 5]),
    row_lengths=tf.constant([2, 0, 3])
)
# [[1, 2], [], [3, 4, 5]]
```

**tf.RaggedTensor.from_row_starts(values, row_starts, name=None)**
Create from values and row start indices.

**tf.RaggedTensor.from_row_limits(values, row_limits, name=None)**
Create from values and row limit indices.

**tf.RaggedTensor.from_uniform_row_length(values, uniform_row_length, nrows=None, name=None)**
Create from values with a fixed row length.

**tf.RaggedTensor.from_tensor(tensor, lengths=None, padding=None, ragged_rank=1, name=None)**
Convert a dense or padded tensor to a RaggedTensor.
```python
# From padded tensor, stripping padding
dense = tf.constant([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
rt = tf.RaggedTensor.from_tensor(dense, padding=0)
# [[1, 2], [3], [4, 5, 6]]
```

**tf.RaggedTensor.from_sparse(st_input, name=None)**
Convert a SparseTensor to a RaggedTensor.

**tf.RaggedTensor.from_nested_row_splits(values, nested_row_splits, name=None)**
Create a multi-level ragged tensor.

### Properties

- **`.values`**: The flat values tensor (or nested RaggedTensor for multi-rag).
- **`.row_splits`**: The row split indices.
- **`.flat_values`**: The innermost flat values tensor.
- **`.nested_row_splits`**: Tuple of row_splits tensors at each ragged level.
- **`.ragged_rank`**: Number of ragged dimensions.
- **`.shape`**: The TensorShape (with `None` for ragged dimensions).
- **`.dtype`**: Data type of the values.
- **`.bounding_shape`**: Maximum shape of all rows.

### Conversion Operations

**rt.to_tensor(default_value=None, name=None)**
Convert to a dense tensor, padding with `default_value`.
```python
rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
dense = rt.to_tensor(default_value=0)
# [[1, 2, 0], [3, 0, 0], [4, 5, 6]]
```

**rt.to_sparse(name=None)**
Convert to a SparseTensor.

### Manipulation Operations

**tf.concat(values, axis, name=None)**
Concatenate ragged tensors.
```python
combined = tf.concat([rt1, rt2], axis=0)
```

**tf.expand_dims(input, axis, name=None)**
Expand dimensions of a ragged tensor.

**rt.merge_dims(outer_axis, inner_axis, name=None)**
Merge two dimensions into one. Useful for flattening nested structures.

**rt.bounding_shape(axis=None, name=None)**
Get the bounding (maximum) shape.

**rt.flat_values**
Access the innermost flat values.

**rt.flatten()**
Flatten the ragged tensor.

### Example: Text Processing

```python
# Batch of sentences with different lengths
sentences = tf.ragged.constant([
    [1, 5, 3, 7],    # sentence with 4 words
    [2, 8],           # sentence with 2 words
    [4, 6, 9, 1, 3]  # sentence with 5 words
])

# Embed each word
embeddings = tf.gather(embedding_table, sentences)
# embeddings is a RaggedTensor of shape [3, None, embed_dim]

# Get sentence lengths
lengths = sentences.row_lengths()  # [4, 2, 5]
```

### Example: Nested RaggedTensors

```python
# Paragraphs containing sentences of different lengths
paragraphs = tf.ragged.constant([
    [[1, 2], [3]],           # paragraph 1: 2 sentences
    [[4, 5, 6], [7], [8, 9]] # paragraph 2: 3 sentences
])
# Shape: [2, None, None]
```

---

## StringTensor

### Overview

String tensors hold variable-length string data. They are represented as
1-D or higher-dimensional tensors with `tf.string` dtype.

### Creation

```python
# From Python strings
s = tf.constant(["hello", "world", "tensorflow"])

# Scalar string
s = tf.constant("hello world")

# Ragged string tensor
rs = tf.ragged.constant([["hello", "world"], ["hi"]])
```

### Operations

**tf.strings.join(inputs, separator='', name=None)**
```python
tf.strings.join([["a", "b"], ["c", "d"]], separator="-")
# ["a-c", "b-d"]
```

**tf.strings.lower(input, encoding='', name=None)**
Convert strings to lowercase.

**tf.strings.upper(input, encoding='', name=None)**
Convert strings to uppercase.

**tf.strings.split(input, sep=None, maxsplit=-1, name=None)**
Split strings into ragged tensors.
```python
result = tf.strings.split(["hello world", "foo bar"])
# RaggedTensor: [["hello", "world"], ["foo", "bar"]]
```

**tf.strings.substr(input, pos, len, unit='BYTE', name=None)**
Extract substrings.

**tf.strings.strip(input, name=None)**
Strip leading and trailing whitespace.

**tf.strings.regex_replace(input, pattern, rewrite, replace_global=True, name=None)**
Replace using regular expressions.

**tf.strings.to_number(input, out_type=tf.dtypes.float32, name=None)**
Parse strings to numbers.

**tf.strings.length(input, unit='BYTE', name=None)**
Get string lengths.

**tf.strings.format(template, inputs, placeholder='{}', summarize=3, name=None)**
Format strings.

---

## CompositeTensor

### Overview

`CompositeTensor` is the abstract base class for tensors that are composed of
multiple component tensors. `SparseTensor` and `RaggedTensor` are the primary
implementations.

The CompositeTensor protocol enables:
1. Uniform handling of dense and composite tensors in APIs.
2. Proper decomposition and recomposition during function tracing.
3. Nesting of composite tensors within structures.

### Interface

```python
class CompositeTensor:
    @property
    def _type_spec(self):
        """Returns the TypeSpec for this tensor."""
        ...

    def _shape_invariant_to_type_spec(self, shape):
        """Returns a TypeSpec with the given shape invariant."""
        ...

    def _component_tensor(self, component_index):
        """Returns the component tensor at the given index."""
        ...

    @property
    def _flat_components(self):
        """Returns a flat list of component tensors."""
        ...
```

### TypeSpec

Every CompositeTensor has an associated `TypeSpec` that describes its structure
without the actual data:

```python
class TypeSpec:
    @property
    def value_type(self):
        """The Python type of values described by this TypeSpec."""
        ...

    def is_compatible_with(self, spec_or_value):
        """Check if this spec is compatible with another."""
        ...

    def most_specific_compatible_type(self, other):
        """Return the most specific compatible TypeSpec."""
        ...

    def _serialize(self):
        """Serialize the TypeSpec for TensorSpec encoding."""
        ...

    def _to_components(self, value):
        """Decompose a value into its component tensors."""
        ...

    def _from_components(self, components):
        """Reconstruct a value from component tensors."""
        ...
```

### Decomposition and Composition

```python
# Decompose a SparseTensor into components
spec = st._type_spec
components = spec._to_components(st)

# Recompose from components
restored = spec._from_components(components)
```

### Usage with tf.function

CompositeTensors work seamlessly with `tf.function`:

```python
@tf.function(input_signature=[
    tf.SparseTensorSpec(shape=[None, 10], dtype=tf.float32)
])
def process_sparse(sp):
    return tf.sparse.to_dense(sp) * 2
```

---

## Extension Types

### Overview

`tf.ExtensionType` allows users to define custom composite tensor types that
behave like built-in TensorFlow types. Extension types are subclasses of
`CompositeTensor` and can be used with `tf.function`, `tf.GradientTape`,
and other TensorFlow APIs.

### Definition

```python
class MyExtension(tf.ExtensionType):
    """A custom tensor type."""

    # Define the fields (component tensors)
    values: tf.Tensor
    indices: tf.Tensor
    metadata: tf.Tensor  # Can be scalar (shared across batch)

    def validate(self):
        """Validate the fields."""
        tf.debugging.assert_shapes([
            (self.values, ('n', 'd')),
            (self.indices, ('n',)),
        ])
```

### Batchable Extension Types

For types that support batching:

```python
class BatchableExtension(tf.ExtensionType):
    values: tf.Tensor

    @classmethod
    def _batchable_shape_keys(cls):
        """Return keys that have a batch dimension."""
        return ['values']

    def _batch(self, batch_size):
        """Batch this instance."""
        ...

    def _unbatch(self):
        """Unbatch this instance."""
        ...
```

### Registration

Register an extension type for use with SavedModel:

```python
@tf.register_extension_type
class MyExtension(tf.ExtensionType):
    ...
```

### Encoding/Decoding

Extension types can define custom encoding for tensor operations:

```python
class MyExtension(tf.ExtensionType):
    values: tf.Tensor

    def _to_tensor_list(self):
        """Encode as a list of tensors for serialization."""
        return [self.values]

    @classmethod
    def _from_tensor_list(cls, tensors):
        """Decode from a list of tensors."""
        return cls(values=tensors[0])
```

---

## TensorSpec

### Overview

`TensorSpec` describes the shape, dtype, and optional name of a tensor without
containing actual data. It is used for:
- Input/output signatures of `tf.function`.
- Specifying expected tensor shapes in APIs.
- Type checking and compatibility testing.

### Constructor

```python
tf.TensorSpec(
    shape=None,
    dtype=tf.dtypes.float32,
    name=None
)
```

**shape** (`tf.TensorShape`, list, tuple, or `None`):
The expected shape. `None` or `Dimension(None)` indicates unknown dimension
size. A fully unknown shape is `TensorShape(None)`.

**dtype** (`tf.DType`):
The expected data type.

**name** (`str`, optional):
A name for the tensor.

### Properties

- **`.shape`**: The `TensorShape`.
- **`.dtype`**: The `tf.DType`.
- **`.name`**: The name string (or `None`).

### Methods

**is_compatible_with(spec_or_value)**
Check if this spec is compatible with another spec or a concrete tensor.
```python
spec = tf.TensorSpec([None, 3], tf.float32)
spec.is_compatible_with(tf.TensorSpec([5, 3], tf.float32))  # True
spec.is_compatible_with(tf.TensorSpec([5, 4], tf.float32))  # False
spec.is_compatible_with(tf.TensorSpec([5, 3], tf.int32))    # False
```

**most_specific_compatible_type(spec_or_value)**
Return the most specific TypeSpec compatible with both.

### Example: Function Signatures

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name="images"),
    tf.TensorSpec(shape=[None, 10], dtype=tf.float32, name="labels")
])
def train_step(images, labels):
    ...
```

### Example: Named Inputs

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[], dtype=tf.float32, name="learning_rate"),
    tf.TensorSpec(shape=[None, 3], dtype=tf.int32, name="features")
])
def process(learning_rate, features):
    ...
```

---

## SparseTensorSpec and RaggedTensorSpec

### SparseTensorSpec

```python
tf.SparseTensorSpec(
    shape=None,
    dtype=tf.dtypes.float32
)
```

Describes a SparseTensor:
- **shape**: The dense shape.
- **dtype**: The value data type.

```python
spec = tf.SparseTensorSpec(shape=[100, 50], dtype=tf.float32)

# Use in function signatures
@tf.function(input_signature=[spec])
def process_sparse(sp):
    return tf.sparse.reduce_sum(sp, axis=1)
```

### RaggedTensorSpec

```python
tf.RaggedTensorSpec(
    shape=None,
    dtype=tf.dtypes.float32,
    ragged_rank=None,
    row_splits_dtype=tf.dtypes.int64,
    flat_values_spec=None
)
```

Describes a RaggedTensor:
- **shape**: The shape (with `None` for ragged dimensions).
- **dtype**: The value data type.
- **ragged_rank**: Number of ragged dimensions.
- **row_splits_dtype**: Integer dtype for row split indices.
- **flat_values_spec**: TypeSpec for the flat values.

```python
spec = tf.RaggedTensorSpec(shape=[None, None, 3], dtype=tf.float32, ragged_rank=2)

@tf.function(input_signature=[spec])
def process_ragged(rt):
    return rt.to_tensor()
```

---

## Registered Types

### TypeSpec Registry

TensorFlow maintains a global registry of `TypeSpec` implementations. This
enables deserialization of composite tensors from SavedModels and other
serialization formats.

### Registration

Types are registered automatically when defined. Custom types can be registered:

```python
@tf.register_tensor_spec_type
class MyTypeSpec(tf.TypeSpec):
    ...
```

### Lookup

```python
# Check if a type is registered
spec = tf.TensorSpec.from_value(my_tensor)

# Get the TypeSpec for a value
spec = tf.type_spec_from_value(my_composite_tensor)
```

### Builtin Registered Types

- `tf.TensorSpec` (dense tensors)
- `tf.SparseTensorSpec` (sparse tensors)
- `tf.RaggedTensorSpec` (ragged tensors)
- `tf.TensorArraySpec` (tensor arrays)
- `tf.data.DatasetSpec` (datasets)
- `tf.DistributedValuesSpec` (distributed values)

---

## tf.TensorArray

### Overview (Detailed)

`tf.TensorArray` provides dynamic-size, dynamically-indexed arrays optimized
for use within `tf.while_loop`. They support writing at arbitrary indices,
dynamic growth, and conversion to/from regular tensors.

### Constructor (Full)

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

**dtype** (`tf.DType`): Required. Data type of elements.

**size** (`int` or `tf.Tensor`): Initial size. Required if `dynamic_size=False`.

**dynamic_size** (`bool`):
- `True`: Array grows automatically when writing beyond current size.
- `False` (default): Fixed size; writing beyond bounds raises an error.

**clear_after_read** (`bool`):
- `True` (default): Memory is freed after reading an element.
- `False`: Elements can be read multiple times.

**element_shape** (`tf.TensorShape`):
Expected shape of each element. Required for graph mode when the shape cannot
be inferred from the first write.

**name** (`str`): Optional name prefix.

**colocate_with_first_write_call** (`bool`):
If `True` (default), the TensorArray is colocated with the device of the
first write call.

**infer_shape** (`bool`):
If `True` (default), shapes are inferred from writes and enforced on subsequent
writes.

### Complete Method Reference

#### write(index, value, name=None)
Write `value` at `index`. Returns a new TensorArray with the update.
The index must be a scalar integer.
```python
ta = tf.TensorArray(dtype=tf.float32, size=3)
ta = ta.write(0, tf.constant([1.0, 2.0]))
ta = ta.write(1, tf.constant([3.0, 4.0]))
ta = ta.write(2, tf.constant([5.0, 6.0]))
```

#### read(index, name=None)
Read the value at `index`. Returns a tensor.
```python
val = ta.read(1)  # [3.0, 4.0]
```

#### stack(name=None)
Stack all elements along a new first dimension.
```python
stacked = ta.stack()
# shape: [3, 2], values: [[1,2],[3,4],[5,6]]
```

#### unstack(value, name=None)
Unstack a tensor into the array. Each element of the first dimension becomes
an array element.
```python
ta = tf.TensorArray(dtype=tf.float32, size=3)
ta = ta.unstack(tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
```

#### size(name=None)
Return the current size (number of elements).
```python
s = ta.size()  # 3
```

#### scatter(indices, value, name=None)
Scatter the values from `value` into the specified indices.
```python
ta = tf.TensorArray(dtype=tf.float32, size=5)
ta = ta.scatter(
    tf.constant([0, 2, 4]),
    tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
)
# ta.read(0) = [1.0, 2.0]
# ta.read(1) = 0 (uninitialized)
# ta.read(2) = [3.0, 4.0]
```

#### gather(indices, name=None)
Read elements at the specified indices and stack them.
```python
values = ta.gather(tf.constant([0, 2, 4]))
```

#### concat(name=None)
Return the elements as a concatenated tensor (along the first axis).
```python
flat = ta.concat()
# shape: [6], values: [1, 2, 3, 4, 5, 6]
```

#### split(value, lengths, name=None)
Split `value` into elements based on `lengths`.
```python
ta = tf.TensorArray(dtype=tf.float32, size=3)
ta = ta.split(
    tf.constant([1.0, 2.0, 3.0, 4.0, 5.0]),
    tf.constant([2, 1, 2])
)
# ta.read(0) = [1.0, 2.0]
# ta.read(1) = [3.0]
# ta.read(2) = [4.0, 5.0]
```

### Advanced Usage: Gradient Computation

TensorArrays support gradient computation through while loops:

```python
@tf.function
def cumulative_sum_grad(n):
    ta = tf.TensorArray(dtype=tf.float32, size=n, dynamic_size=False)
    def body(i, ta):
        ta = ta.write(i, tf.cast(i, tf.float32) ** 2)
        return i + 1, ta
    _, ta = tf.while_loop(lambda i, ta: i < n, body, [0, ta])
    return ta.stack()

with tf.GradientTape() as tape:
    n = tf.Variable(5.0)
    result = cumulative_sum_grad(tf.cast(n, tf.int32))
```

### Advanced Usage: Nested Structures

```python
@tf.function
def process_sequence(inputs):
    batch_size = tf.shape(inputs)[0]
    seq_len = tf.shape(inputs)[1]

    ta_output = tf.TensorArray(
        dtype=tf.float32,
        size=seq_len,
        dynamic_size=False,
        element_shape=tf.TensorShape([batch_size])
    )

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_len)
    inputs_ta = inputs_ta.unstack(inputs)

    def body(t, ta):
        x = inputs_ta.read(t)
        y = tf.math.sin(x)
        ta = ta.write(t, y)
        return t + 1, ta

    _, ta_output = tf.while_loop(
        lambda t, ta: t < seq_len,
        body,
        [0, ta_output]
    )
    return ta_output.stack()
```

---

## Variant Tensors

### Overview

`tf.variant` is a special dtype for opaque, user-defined data that can be
passed through the TensorFlow graph. Variant tensors hold arbitrary C++ objects
and are primarily used for:
- Custom data formats.
- Inter-op communication of non-standard data.
- Performance-critical data passing.

### Encoding

Variant tensors use a `VariantTensorDataProto` for serialization:

```protobuf
message VariantTensorDataProto {
    string type_name = 1;
    bytes metadata = 2;
    repeated TensorProto tensors = 3;
}
```

### Usage

```python
# Variant tensors are typically created by ops, not directly
# Example: TensorList uses variant tensors internally
tensor_list = tf.raw_ops.TensorListReserve(
    element_shape=tf.TensorShape([3]),
    num_elements=5,
    element_dtype=tf.float32
)
# tensor_list.dtype == tf.variant
```

### Decoding

```python
# Decode a variant tensor (op-specific)
result = tf.raw_ops.TensorListStack(
    input_handle=tensor_list,
    element_shape=tf.TensorShape([3]),
    element_dtype=tf.float32
)
```

---

## Optional Values

### Overview

`tf.experimental.Optional` represents a value that may or may not exist. It is
the TensorFlow equivalent of `Optional` in Java or `Option` in Rust.

### Factory Methods

**tf.experimental.Optional.from_value(value)**
Create an Optional containing the given value.
```python
opt = tf.experimental.Optional.from_value(tf.constant([1, 2, 3]))
opt.has_value()  # True
opt.get_value()  # [1, 2, 3]
```

**tf.experimental.Optional.none(spec)**
Create an empty Optional with the given TypeSpec.
```python
opt = tf.experimental.Optional.none(tf.TensorSpec([3], tf.float32))
opt.has_value()  # False
```

### Methods

**has_value()**
Returns a scalar boolean tensor indicating whether the Optional has a value.

**get_value()**
Returns the contained value. Raises an error if the Optional is empty.

### Use Case: tf.data

Optional values are used in `tf.data` for representing empty batches:

```python
dataset = tf.data.Dataset.range(5)
dataset = dataset.batch(3)
# The last batch has only 2 elements, not 3
# Internally, drop_remainder=True would skip the last batch
```

---

## Structured Tensors

### Overview

TensorFlow provides utilities for working with nested structures (trees) of
tensors, composite tensors, and other values. These are collectively called
"structured tensors" or "nested structures."

### tf.nest

The `tf.nest` module provides utilities for operating on nested structures.

#### tf.nest.flatten(structure, expand_composites=False)
Flatten a nested structure into a flat list.

```python
structure = {'a': [1, 2], 'b': {'c': 3, 'd': 4}}
flat = tf.nest.flatten(structure)
# [1, 2, 3, 4]
```

With `expand_composites=True`, composite tensors are expanded into their
components:
```python
st = tf.sparse.SparseTensor(...)
flat = tf.nest.flatten(st, expand_composites=True)
# [indices, values, dense_shape]
```

#### tf.nest.pack_sequence_as(structure, flat_sequence, expand_composites=False)
Pack a flat sequence into a nested structure matching `structure`.

```python
structure = {'a': [None, None], 'b': {'c': None, 'd': None}}
flat = [1, 2, 3, 4]
packed = tf.nest.pack_sequence_as(structure, flat)
# {'a': [1, 2], 'b': {'c': 3, 'd': 4}}
```

#### tf.nest.map_structure(func, *structures, expand_composites=False, **kwargs)
Apply `func` to each element in the nested structure(s).

```python
structure1 = {'a': [1, 2], 'b': 3}
structure2 = {'a': [10, 20], 'b': 30}
result = tf.nest.map_structure(lambda x, y: x + y, structure1, structure2)
# {'a': [11, 22], 'b': 33}
```

#### tf.nest.map_structure_up_to(shallow_structure, func, *structures, **kwargs)
Like `map_structure` but only traverses as deep as `shallow_structure`.

```python
shallow = {'a': None, 'b': None}
deep = {'a': [1, 2, 3], 'b': {'c': 4, 'd': 5}}
result = tf.nest.map_structure_up_to(
    shallow, lambda x: x * 2, deep)
# {'a': [2, 4, 6], 'b': {'c': 8, 'd': 10}}
```

#### tf.nest.assert_same_structure(a, b, expand_composites=False)
Assert that two structures have the same nested layout.

```python
tf.nest.assert_same_structure(
    {'a': [1, 2], 'b': 3},
    {'a': [4, 5], 'b': 6}
)  # OK

tf.nest.assert_same_structure(
    {'a': [1, 2], 'b': 3},
    {'a': 4, 'b': [5, 6]}
)  # Raises ValueError
```

#### tf.nest.is_nested(structure)
Check if a structure is nested (not a leaf).

```python
tf.nest.is_nested([1, 2])  # True
tf.nest.is_nested({'a': 1})  # True
tf.nest.is_nested(42)  # False
tf.nest.is_nested(tf.constant([1, 2]))  # False
```

### Supported Structure Types

Nested structures can contain:
- **Lists**: Regular Python lists.
- **Tuples**: Regular Python tuples.
- **Dicts**: Regular Python dicts.
- **NamedTuples**: `collections.namedtuple` instances.
- **OrderedDicts**: `collections.OrderedDict` instances.
- **CompositeTensors**: `SparseTensor`, `RaggedTensor`, etc. (with
  `expand_composites=True`).

### Leaf Types

The following are treated as leaves (not further traversed):
- `tf.Tensor`
- NumPy arrays
- Python scalars (int, float, str, bytes, bool)
- `None`
- Any object that is not a list, tuple, dict, or namedtuple

### Example: Processing Model Outputs

```python
# Model returns a structured output
output = {
    'classifications': tf.constant([[0.1, 0.9], [0.8, 0.2]]),
    'boxes': tf.constant([[10, 20, 30, 40], [50, 60, 70, 80]]),
    'masks': tf.sparse.SparseTensor(...)
}

# Apply a transformation to all tensors
processed = tf.nest.map_structure(
    lambda x: x * 2 if isinstance(x, tf.Tensor) else x,
    output,
    expand_composites=False
)
```

### Example: Dataset with Structured Elements

```python
# Dataset yields structured elements
dataset = tf.data.Dataset.from_tensor_slices({
    'image': images,
    'label': labels,
    'bbox': bboxes
})

# Map function receives and returns structured elements
def augment(element):
    return {
        'image': tf.image.random_flip_left_right(element['image']),
        'label': element['label'],
        'bbox': element['bbox']
    }

dataset = dataset.map(augment)
```

### Example: tf.function with Structured Signatures

```python
input_spec = {
    'features': tf.TensorSpec([None, 10], tf.float32),
    'mask': tf.SparseTensorSpec([None, 10], tf.float32),
    'metadata': {
        'ids': tf.TensorSpec([None], tf.int32),
        'timestamps': tf.TensorSpec([None], tf.float32)
    }
}

@tf.function(input_signature=[input_spec])
def process(data):
    features = data['features']
    mask = data['mask']
    ids = data['metadata']['ids']
    ...
```
