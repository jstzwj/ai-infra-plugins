# TensorFlow Operations Reference

## Table of Contents

1. [tf.math](#tfmath)
2. [tf.array](#tfarray)
3. [tf.nn](#tfnn)
4. [tf.linalg](#tflinalg)
5. [tf.image](#tfimage)
6. [tf.signal](#tfsignal)
7. [tf.strings](#tfstrings)
8. [tf.random](#tfrandom)
9. [tf.io](#tfio)

---

## tf.math

### Arithmetic Operations

**tf.math.add(x, y, name=None)**
Element-wise addition. Supports broadcasting.
```python
tf.math.add(tf.constant([1, 2]), tf.constant([3, 4]))  # [4, 6]
tf.math.add(tf.constant([1, 2]), 3)  # [4, 5] (broadcasting)
```

**tf.math.subtract(x, y, name=None)**
Element-wise subtraction.
```python
tf.math.subtract(tf.constant([5, 3]), tf.constant([1, 2]))  # [4, 1]
```

**tf.math.multiply(x, y, name=None)**
Element-wise multiplication.
```python
tf.math.multiply(tf.constant([2, 3]), tf.constant([4, 5]))  # [8, 15]
```

**tf.math.divide(x, y, name=None)**
Element-wise division (Python 3 semantics, returns float).
```python
tf.math.divide(tf.constant([6.0, 8.0]), tf.constant([2.0, 4.0]))  # [3.0, 2.0]
```

**tf.math.divide_no_nan(x, y, name=None)**
Division returning 0 when y is 0.
```python
tf.math.divide_no_nan(tf.constant([1.0, 2.0]), tf.constant([0.0, 2.0]))  # [0.0, 1.0]
```

**tf.math.truediv(x, y, name=None)**
Division that always returns a float result.

**tf.math.floordiv(x, y, name=None)**
Element-wise floor division.

**tf.math.real(x, name=None)**
Returns the real part of a complex tensor.

**tf.math.imag(x, name=None)**
Returns the imaginary part of a complex tensor.

**tf.math.complex(real, imag, name=None)**
Constructs complex numbers from real and imaginary parts.

**tf.math.negative(x, name=None)**
Element-wise numerical negation.
```python
tf.math.negative(tf.constant([-2, 3]))  # [2, -3]
```

**tf.math.reciprocal(x, name=None)**
Element-wise reciprocal (1/x).
```python
tf.math.reciprocal(tf.constant([2.0, 4.0]))  # [0.5, 0.25]
```

**tf.math.reciprocal_no_nan(x, name=None)**
Reciprocal returning 0 when x is 0.

**tf.math.sign(x, name=None)**
Element-wise sign: -1, 0, or 1.

**tf.math.square(x, name=None)**
Element-wise square.
```python
tf.math.square(tf.constant([2, 3]))  # [4, 9]
```

**tf.math.sqrt(x, name=None)**
Element-wise square root.
```python
tf.math.sqrt(tf.constant([4.0, 9.0]))  # [2.0, 3.0]
```

**tf.math.rsqrt(x, name=None)**
Element-wise reciprocal of square root.
```python
tf.math.rsqrt(tf.constant([4.0, 9.0]))  # [0.5, 0.333]
```

**tf.math.pow(x, y, name=None)**
Element-wise power.
```python
tf.math.pow(tf.constant([2, 3]), tf.constant([3, 2]))  # [8, 9]
```

**tf.math.abs(x, name=None)**
Element-wise absolute value.

**tf.math.floormod(x, y, name=None)**
Element-wise floor modulus.

### Exponential and Logarithmic

**tf.math.exp(x, name=None)**
Element-wise exponential (e^x).

**tf.math.expm1(x, name=None)**
Element-wise exp(x) - 1. More accurate for small x.

**tf.math.log(x, name=None)**
Element-wise natural logarithm.

**tf.math.log1p(x, name=None)**
Element-wise log(1 + x). More accurate for small x.

**tf.math.log_sigmoid(x, name=None)**
Element-wise log(sigmoid(x)). Numerically stable.

**tf.math.xlogy(x, y, name=None)**
Element-wise x * log(y). Returns 0 when x == 0.

**tf.math.xlog1py(x, y, name=None)**
Element-wise x * log1p(y). Returns 0 when x == 0.

**tf.math.xdivy(x, y, name=None)**
Element-wise x / y. Returns 0 when x == 0.

### Trigonometric

**tf.math.sin(x, name=None)**
Element-wise sine.

**tf.math.cos(x, name=None)**
Element-wise cosine.

**tf.math.tan(x, name=None)**
Element-wise tangent.

**tf.math.asin(x, name=None)**
Element-wise arcsine.

**tf.math.acos(x, name=None)**
Element-wise arccosine.

**tf.math.atan(x, name=None)**
Element-wise arctangent.

**tf.math.atan2(y, x, name=None)**
Element-wise arctangent of y/x with proper quadrant handling.

### Hyperbolic

**tf.math.sinh(x, name=None)**
Element-wise hyperbolic sine.

**tf.math.cosh(x, name=None)**
Element-wise hyperbolic cosine.

**tf.math.tanh(x, name=None)**
Element-wise hyperbolic tangent.

**tf.math.asinh(x, name=None)**
Element-wise inverse hyperbolic sine.

**tf.math.acosh(x, name=None)**
Element-wise inverse hyperbolic cosine.

**tf.math.atanh(x, name=None)**
Element-wise inverse hyperbolic tangent.

### Activation Functions

**tf.math.sigmoid(x, name=None)**
Element-wise sigmoid: 1 / (1 + exp(-x)).

**tf.math.softplus(x, name=None)**
Element-wise softplus: log(exp(x) + 1). Numerically stable.

**tf.math.softsign(x, name=None)**
Element-wise softsign: x / (abs(x) + 1).

### Rounding and Classification

**tf.math.ceil(x, name=None)**
Element-wise ceiling.

**tf.math.floor(x, name=None)**
Element-wise floor.

**tf.math.round(x, name=None)**
Element-wise rounding (round half to even).

**tf.math.rint(x, name=None)**
Element-wise rounding to nearest integer.

**tf.math.fix(x, name=None)**
Element-wise rounding towards zero.

**tf.math.is_finite(x, name=None)**
Element-wise check for finite values (not inf or nan).

**tf.math.is_inf(x, name=None)**
Element-wise check for infinity.

**tf.math.is_nan(x, name=None)**
Element-wise check for NaN.

**tf.math.is_nonnegative(x, name=None)**
Element-wise check for non-negative values.

**tf.math.nextafter(x1, x2, name=None)**
Element-wise next representable float value after x1 towards x2.

### Comparison

**tf.math.maximum(x, y, name=None)**
Element-wise maximum.

**tf.math.minimum(x, y, name=None)**
Element-wise minimum.

**tf.math.argmax(input, axis=None, output_type=tf.dtypes.int64, name=None)**
Returns index of maximum value along axis.

**tf.math.argmin(input, axis=None, output_type=tf.dtypes.int64, name=None)**
Returns index of minimum value along axis.

**tf.math.equal(x, y, name=None)**
Element-wise equality comparison.

**tf.math.not_equal(x, y, name=None)**
Element-wise inequality comparison.

**tf.math.greater(x, y, name=None)**
Element-wise greater than.

**tf.math.greater_equal(x, y, name=None)**
Element-wise greater than or equal.

**tf.math.less(x, y, name=None)**
Element-wise less than.

**tf.math.less_equal(x, y, name=None)**
Element-wise less than or equal.

### Reduction Operations

**tf.math.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)**
Sum of elements across dimensions.
```python
tf.math.reduce_sum(tf.constant([[1, 2], [3, 4]]))  # 10
tf.math.reduce_sum(tf.constant([[1, 2], [3, 4]]), axis=0)  # [4, 6]
tf.math.reduce_sum(tf.constant([[1, 2], [3, 4]]), axis=1)  # [3, 7]
```

**tf.math.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)**
Mean of elements across dimensions.

**tf.math.reduce_max(input_tensor, axis=None, keepdims=False, name=None)**
Maximum of elements across dimensions.

**tf.math.reduce_min(input_tensor, axis=None, keepdims=False, name=None)**
Minimum of elements across dimensions.

**tf.math.reduce_prod(input_tensor, axis=None, keepdims=False, name=None)**
Product of elements across dimensions.

**tf.math.reduce_all(input_tensor, axis=None, keepdims=False, name=None)**
Logical AND across dimensions (boolean tensors).

**tf.math.reduce_any(input_tensor, axis=None, keepdims=False, name=None)**
Logical OR across dimensions (boolean tensors).

**tf.math.reduce_logsumexp(input_tensor, axis=None, keepdims=False, name=None)**
Log(sum(exp(elements))) computed in a numerically stable way.

**tf.math.reduce_euclidean_norm(input_tensor, axis=None, keepdims=False, name=None)**
Euclidean norm across dimensions: sqrt(sum(x^2)).

**tf.math.reduce_variance(input_tensor, axis=None, keepdims=False, name=None)**
Variance across dimensions.

**tf.math.reduce_std(input_tensor, axis=None, keepdims=False, name=None)**
Standard deviation across dimensions.

### Cumulative Operations

**tf.math.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)**
Cumulative sum along axis.
```python
tf.math.cumsum(tf.constant([1, 2, 3, 4]))  # [1, 3, 6, 10]
tf.math.cumsum(tf.constant([1, 2, 3, 4]), reverse=True)  # [10, 9, 7, 4]
```

**tf.math.cumprod(x, axis=0, exclusive=False, reverse=False, name=None)**
Cumulative product along axis.

**tf.math.cumulative_logsumexp(x, axis=0, exclusive=False, reverse=False, name=None)**
Cumulative log-sum-exp along axis.

### Segment Operations

**tf.math.segment_sum(data, segment_ids, name=None)**
Sum along segments. `segment_ids` must be sorted.
```python
tf.math.segment_sum(
    tf.constant([1.0, 2.0, 3.0, 4.0, 5.0]),
    tf.constant([0, 0, 1, 1, 2])
)  # [3.0, 7.0, 5.0]
```

**tf.math.segment_mean(data, segment_ids, name=None)**
Mean along segments.

**tf.math.segment_max(data, segment_ids, name=None)**
Maximum along segments.

**tf.math.segment_min(data, segment_ids, name=None)**
Minimum along segments.

**tf.math.segment_prod(data, segment_ids, name=None)**
Product along segments.

**tf.math.unsorted_segment_sum(data, segment_ids, num_segments, name=None)**
Sum along segments without requiring sorted segment_ids.
```python
tf.math.unsorted_segment_sum(
    tf.constant([1.0, 2.0, 3.0]),
    tf.constant([2, 0, 1]),
    num_segments=4
)  # [2.0, 3.0, 1.0, 0.0]
```

**tf.math.unsorted_segment_max(data, segment_ids, num_segments, name=None)**

**tf.math.unsorted_segment_min(data, segment_ids, num_segments, name=None)**

**tf.math.unsorted_segment_mean(data, segment_ids, num_segments, name=None)**

**tf.math.unsorted_segment_prod(data, segment_ids, num_segments, name=None)**

**tf.math.unsorted_segment_sqrt_n(data, segment_ids, num_segments, name=None)**

### Special Functions

**tf.math.erf(x, name=None)**
Element-wise error function.

**tf.math.erfc(x, name=None)**
Element-wise complementary error function: 1 - erf(x).

**tf.math.lgamma(x, name=None)**
Element-wise log of the absolute value of the gamma function.

**tf.math.digamma(x, name=None)**
Element-wise digamma function (psi(x)), the derivative of lgamma.

**tf.math.igamma(a, x, name=None)**
Element-wise lower regularized incomplete gamma function.

**tf.math.igammac(a, x, name=None)**
Element-wise upper regularized incomplete gamma function.

**tf.math.betainc(a, b, x, name=None)**
Element-wise regularized incomplete beta function.

**tf.math.bincount(arr, weights=None, minlength=None, maxlength=None, dtype=tf.int32, name=None)**
Count occurrences of each value in integer array.

**tf.math.cross(a, b, name=None)**
Element-wise cross product of 3-element vectors.

**tf.math.einsum(equation, *inputs, **kwargs)**
Tensor contraction via Einstein summation convention.
```python
# Matrix multiplication
tf.math.einsum('ij,jk->ik', A, B)
# Batch matmul
tf.math.einsum('bij,bjk->bik', A, B)
# Element-wise multiply then sum
tf.math.einsum('i,i->', a, b)  # dot product
```

**tf.math.count_nonzero(input, axis=None, keepdims=False, dtype=tf.dtypes.int64, name=None)**
Count non-zero elements across dimensions.

**tf.math.diff(x, axis=0)**
Computes the n-th discrete difference along given axis.

**tf.math.unique(x, out_idx=tf.dtypes.int32, name=None)**
Find unique elements.
```python
y, idx = tf.math.unique(tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8]))
# y = [1, 2, 4, 7, 8]
# idx = [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

**tf.math.unique_with_counts(x, out_idx=tf.dtypes.int32, name=None)**
Find unique elements with counts.
```python
y, idx, count = tf.math.unique_with_counts(
    tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8]))
# y = [1, 2, 4, 7, 8]
# idx = [0, 0, 1, 2, 2, 2, 3, 4, 4]
# count = [2, 1, 3, 1, 2]
```

---

## tf.array

### Shape Manipulation

**tf.reshape(tensor, shape, name=None)**
Reshape a tensor.
```python
tf.reshape(tf.constant([[1, 2], [3, 4]]), [1, 4])  # [[1, 2, 3, 4]]
tf.reshape(tf.constant([1, 2, 3, 4, 5, 6]), [2, -1])  # [[1, 2, 3], [4, 5, 6]]
```

**tf.expand_dims(input, axis, name=None)**
Insert a dimension of length 1 at the given axis.
```python
tf.expand_dims(tf.constant([1, 2, 3]), 0)  # [[1, 2, 3]] shape (1, 3)
tf.expand_dims(tf.constant([1, 2, 3]), 1)  # [[1], [2], [3]] shape (3, 1)
```

**tf.squeeze(input, axis=None, name=None)**
Remove dimensions of length 1.
```python
tf.squeeze(tf.constant([[[1, 2, 3]]]))  # [1, 2, 3]
tf.squeeze(tf.constant([[[1, 2, 3]]]), axis=0)  # [[1, 2, 3]]
```

**tf.broadcast_to(input, shape, name=None)**
Broadcast input to given shape.
```python
tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3])  # [[1,2,3],[1,2,3],[1,2,3]]
```

**tf.shape(input, out_type=tf.dtypes.int32, name=None)**
Get the shape of a tensor as a 1-D integer tensor.

**tf.size(input, out_type=tf.dtypes.int32, name=None)**
Get the total number of elements in a tensor.

**tf.rank(input, name=None)**
Get the rank (number of dimensions) of a tensor.

### Transposition and Reordering

**tf.transpose(a, perm=None, conjugate=False, name=None)**
Transpose a tensor by permuting dimensions.
```python
tf.transpose(tf.constant([[1, 2, 3], [4, 5, 6]]))  # [[1,4],[2,5],[3,6]]
tf.transpose(x, perm=[0, 2, 1])  # For 3D: swap last two dims
```

**tf.reverse(tensor, axis, name=None)**
Reverse tensor along specified axis.
```python
tf.reverse(tf.constant([[1, 2], [3, 4]]), [1])  # [[2, 1], [4, 3]]
```

**tf.roll(input, shift, axis, name=None)**
Roll elements along given axis.
```python
tf.roll(tf.constant([1, 2, 3, 4, 5]), shift=2, axis=0)  # [4, 5, 1, 2, 3]
```

### Tiling and Repeating

**tf.tile(input, multiples, name=None)**
Repeat a tensor by tiling.
```python
tf.tile(tf.constant([[1, 2]]), [3, 2])  # [[1,2,1,2],[1,2,1,2],[1,2,1,2]]
```

**tf.repeat(input, repeats, axis=None, name=None)**
Repeat elements of a tensor.
```python
tf.repeat(tf.constant([1, 2, 3]), [2, 3, 1])  # [1, 1, 2, 2, 2, 3]
```

### Concatenation and Splitting

**tf.concat(values, axis, name=None)**
Concatenate tensors along axis.
```python
tf.concat([tf.constant([1, 2]), tf.constant([3, 4])], axis=0)  # [1, 2, 3, 4]
```

**tf.stack(values, axis=0, name=None)**
Stack tensors along a new axis.
```python
tf.stack([tf.constant([1, 2]), tf.constant([3, 4])])  # [[1, 2], [3, 4]]
```

**tf.unstack(value, num=None, axis=0, name=None)**
Unstack tensor along axis into a list.
```python
tf.unstack(tf.constant([[1, 2], [3, 4]]))  # [tf.constant([1, 2]), tf.constant([3, 4])]
```

**tf.split(value, num_or_size_splits, axis=0, num=None, name=None)**
Split tensor into sub-tensors.
```python
tf.split(tf.constant([1, 2, 3, 4, 5, 6]), 3)  # [[1,2],[3,4],[5,6]]
tf.split(tf.constant([1, 2, 3, 4, 5, 6]), [2, 4])  # [[1,2],[3,4,5,6]]
```

### Indexing and Gathering

**tf.gather(params, indices, axis=None, batch_dims=0, name=None)**
Gather slices from params according to indices.
```python
tf.gather(tf.constant([10, 20, 30, 40]), tf.constant([0, 2]))  # [10, 30]
```

**tf.gather_nd(params, indices, batch_dims=0, name=None)**
Gather slices from params into a tensor with shape specified by indices.
```python
tf.gather_nd(
    tf.constant([[1, 2], [3, 4]]),
    tf.constant([[0, 0], [1, 1]])
)  # [1, 4]
```

**tf.boolean_mask(tensor, mask, axis=None, name=None)**
Apply boolean mask to tensor.
```python
tf.boolean_mask(
    tf.constant([1, 2, 3, 4, 5]),
    tf.constant([True, False, True, False, True])
)  # [1, 3, 5]
```

### Scatter Operations

**tf.scatter_nd(indices, updates, shape, name=None)**
Scatter updates into a new tensor according to indices.
```python
tf.scatter_nd(
    tf.constant([[0], [2]]),
    tf.constant([10, 20]),
    tf.constant([4])
)  # [10, 0, 20, 0]
```

**tf.scatter_nd_update(ref, indices, updates, name=None)**
Scatter updates into an existing variable.

**tf.scatter_nd_add(ref, indices, updates, name=None)**
Add updates to existing variable at indices.

**tf.scatter_nd_sub(ref, indices, updates, name=None)**
Subtract updates from existing variable at indices.

**tf.scatter_nd_min(ref, indices, updates, name=None)**
Element-wise minimum at indices.

**tf.scatter_nd_max(ref, indices, updates, name=None)**
Element-wise maximum at indices.

**tf.tensor_scatter_nd_update(tensor, indices, updates, name=None)**
Return a copy of tensor with updates scattered at indices.

**tf.tensor_scatter_nd_add(tensor, indices, updates, name=None)**

**tf.tensor_scatter_nd_sub(tensor, indices, updates, name=None)**

**tf.tensor_scatter_nd_max(tensor, indices, updates, name=None)**

**tf.tensor_scatter_nd_min(tensor, indices, updates, name=None)**

### Creation and Utility

**tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)**
One-hot encoding.
```python
tf.one_hot(tf.constant([0, 1, 2]), depth=3)
# [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

**tf.where(condition, x=None, y=None, name=None)**
Select elements from x or y based on condition. If x and y are None, returns
coordinates of True elements.
```python
tf.where(tf.constant([True, False, True]), tf.constant([1, 2, 3]), tf.constant([4, 5, 6]))
# [1, 5, 3]
```

**tf.meshgrid(*args, **kwargs)**
Broadcasting version of numpy.meshgrid.

**tf.fill(dims, value, name=None)**
Create tensor filled with a scalar value.
```python
tf.fill([2, 3], 9)  # [[9, 9, 9], [9, 9, 9]]
```

**tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None)**
Pad a tensor.
```python
tf.pad(tf.constant([[1, 2], [3, 4]]), [[1, 1], [1, 1]])
# [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
```

**tf.extract_patches(images, sizes, strides, rates, padding, name=None)**
Extract patches from images.

**tf.extract_glimpse(input, size, offsets, centered=True, normalized=True, uniform_noise=True, noise='uniform', name=None)**
Extract a glimpse from input tensor.

---

## tf.nn

### Activation Functions

**tf.nn.relu(features, name=None)**
Rectified Linear Unit: max(0, x).

**tf.nn.relu6(features, name=None)**
ReLU6: min(max(0, x), 6).

**tf.nn.crelu(features, name=None)**
Concatenated ReLU: concat([relu(x), relu(-x)]).

**tf.nn.elu(features, name=None)**
Exponential Linear Unit: x if x > 0 else exp(x) - 1.

**tf.nn.selu(features, name=None)**
Scaled ELU: scale * elu(x, alpha) where alpha=1.67326..., scale=1.0507...

**tf.nn.softplus(features, name=None)**
log(exp(features) + 1).

**tf.nn.softsign(features, name=None)**
features / (abs(features) + 1).

**tf.nn.sigmoid(x, name=None)**
1 / (1 + exp(-x)).

**tf.nn.tanh(x, name=None)**
Hyperbolic tangent.

**tf.nn.softmax(logits, axis=None, name=None)**
Softmax: exp(logits) / sum(exp(logits)).
```python
tf.nn.softmax(tf.constant([[1.0, 2.0, 3.0]]))  # [[0.090, 0.245, 0.665]]
```

**tf.nn.log_softmax(logits, axis=None, name=None)**
Log-softmax: logits - log(sum(exp(logits))). Numerically stable.

### Convolution

**tf.nn.convolution(input, filters, strides=None, padding='VALID', data_format=None, dilations=None, name=None)**
General N-D convolution. Handles 1D, 2D, and 3D.

**tf.nn.conv1d(input, filters, stride, padding, data_format='NWC', dilations=None, name=None)**
1D convolution.

**tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)**
2D convolution.
```python
# input: [batch, height, width, in_channels]
# filters: [filter_height, filter_width, in_channels, out_channels]
output = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
```

**tf.nn.conv3d(input, filters, strides, padding, data_format='NDHWC', dilations=None, name=None)**
3D convolution.

**tf.nn.conv2d_transpose(value, filters, output_shape, strides, padding='SAME', data_format='NHWC', name=None)**
2D transposed convolution (deconvolution).

**tf.nn.conv3d_transpose(value, filters, output_shape, strides, padding='SAME', data_format='NDHWC', name=None)**
3D transposed convolution.

**tf.nn.conv1d_transpose(value, filters, output_shape, strides, padding='SAME', data_format='NWC', name=None)**
1D transposed convolution.

**tf.nn.depthwise_conv2d(input, filter, strides, padding, rate=None, name=None, data_format=None, dilations=None)**
Depthwise 2D convolution.

**tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, rate=None, name=None, data_format=None, dilations=None)**
Separable 2D convolution.

**tf.nn.atrous_conv2d(value, filters, rate, padding, name=None)**
Atrous (dilated) 2D convolution.

**tf.nn.dilation2d(input, filters, strides, rates, padding, data_format=None, name=None)**
Morphological dilation.

### Pooling

**tf.nn.avg_pool(input, ksize, strides, padding, data_format='NHWC', name=None)**
Average pooling.

**tf.nn.max_pool(input, ksize, strides, padding, data_format='NHWC', name=None)**
Max pooling.

**tf.nn.avg_pool2d(input, ksize, strides, padding, data_format='NHWC', name=None)**
2D average pooling.

**tf.nn.max_pool2d(input, ksize, strides, padding, data_format='NHWC', name=None)**
2D max pooling.

**tf.nn.avg_pool3d(input, ksize, strides, padding, data_format='NDHWC', name=None)**
3D average pooling.

**tf.nn.max_pool3d(input, ksize, strides, padding, data_format='NDHWC', name=None)**
3D max pooling.

**tf.nn.max_pool_with_argmax(input, ksize, strides, padding, data_format='NHWC', output_dtype=tf.dtypes.int64, include_batch_in_index=False, name=None)**
Max pooling with returned argmax indices.

**tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None)**
Fractional average pooling.

**tf.nn.fractional_max_pool(value, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None)**
Fractional max pooling.

### Normalization

**tf.nn.l2_normalize(x, axis=None, epsilon=1e-12, name=None)**
L2 normalization: x / sqrt(sum(x^2) + epsilon).

**tf.nn.l2_loss(t, name=None)**
L2 loss: sum(t^2) / 2.

**tf.nn.local_response_normalization(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)**
Local response normalization.

**tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)**
Batch normalization: scale * (x - mean) / sqrt(variance + epsilon) + offset.

**tf.nn.sufficient_statistics(x, axes, shift=None, keepdims=False, name=None)**
Compute sufficient statistics for batch normalization.

**tf.nn.normalization_moments(axes, shift=None, keep_dims=False, name='normalization_moments')**
Compute mean and variance from sufficient statistics.

**tf.nn.fused_batch_norm(x, scale, offset, mean=None, variance=None, epsilon=1e-4, data_format='NHWC', is_training=True, name=None)**
Fused batch normalization (single kernel, optimized for GPU).

**tf.nn.moments(x, axes, shift=None, keepdims=False, name=None)**
Compute mean and variance.

**tf.nn.weighted_moments(x, axes, frequency_weights, keepdims=False, name=None)**
Compute weighted mean and variance.

### Loss Functions

**tf.nn.sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None)**
Sigmoid cross-entropy loss. Computes max(x, 0) - x*z + log(1 + exp(-abs(x))).

**tf.nn.softmax_cross_entropy_with_logits(labels=None, logits=None, axis=-1, name=None)**
Softmax cross-entropy loss.

**tf.nn.sparse_softmax_cross_entropy_with_logits(labels=None, logits=None, name=None)**
Sparse softmax cross-entropy (labels are integers, not one-hot).

**tf.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, partition_strategy='mod', name='sampled_softmax_loss')**
Sampled softmax loss for large vocabulary.

**tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, partition_strategy='mod', name='nce_loss')**
Noise-contrastive estimation loss.

**tf.nn.ctc_loss(labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None)**
Connectionist Temporal Classification loss.

### Embedding Operations

**tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)**
Look up embeddings for given IDs.
```python
embeddings = tf.nn.embedding_lookup(embedding_matrix, tf.constant([0, 2, 4]))
```

**tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy='mod', name=None, combiner='mean', max_norm=None)**
Sparse embedding lookup with aggregation.

**tf.nn.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner='mean', default_id=None, name=None, partition_strategy='div', max_norm=None)**
Safe version that handles empty rows.

### Other

**tf.nn.bias_add(value, bias, data_format=None, name=None)**
Add bias to value. Supports NHWC and NCHW formats.

**tf.nn.top_k(input, k=1, sorted=True, name=None)**
Return top K values and their indices.

**tf.nn.in_top_k(predictions, targets, k, name=None)**
Check if targets are in top K predictions.

**tf.nn.xw_plus_b(x, weights, biases, name=None)**
Compute matmul(x, weights) + biases.

**tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True, name=None)**
CTC greedy decoder.

**tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True, name=None)**
CTC beam search decoder.

**tf.nn.ctc_unique_labels(labels, name=None)**
Get unique labels from CTC labels.

---

## tf.linalg

### Matrix Operations

**tf.linalg.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, output_type=None, name=None)**
Matrix multiplication.
```python
tf.linalg.matmul(A, B)                    # A @ B
tf.linalg.matmul(A, B, transpose_b=True)  # A @ B^T
```

**tf.linalg.einsum(equation, *inputs, **kwargs)**
Generalized Einstein summation.

**tf.linalg.tensordot(a, b, axes, name=None)**
Tensor contraction over specified axes.

**tf.linalg.trace(x, name=None)**
Compute the trace (sum of diagonal) of a matrix.

**tf.linalg.diag(diagonal, name=None)**
Create a diagonal tensor from a diagonal value.

**tf.linalg.diag_part(input, name=None)**
Extract the diagonal of a matrix.

**tf.linalg.set_diag(input, diagonal, name=None)**
Set the diagonal of a matrix.

**tf.linalg.tensor_diag(diagonal, name=None)**
Create a diagonal tensor (supports batch).

**tf.linalg.tensor_diag_part(input, name=None)**
Extract diagonal (supports batch).

### Matrix Properties

**tf.linalg.det(input, name=None)**
Determinant of a square matrix.

**tf.linalg.matrix_rank(a, tol=None, validate_args=False, name=None)**
Rank of a matrix.

**tf.linalg.norm(tensor, ord='euclidean', axis=None, keepdims=False, name=None)**
Matrix or vector norm.

**tf.linalg.normalize(tensor, ord='euclidean', axis=None, name=None)**
Normalize a tensor along an axis.

### Matrix Factorization

**tf.linalg.cholesky(input, name=None)**
Cholesky decomposition: input = L * L^T.

**tf.linalg.cholesky_solve(chol, rhs, name=None)**
Solve linear system using Cholesky decomposition.

**tf.linalg.lu(input, output_idx_type=tf.dtypes.int32, name=None)**
LU decomposition with partial pivoting.

**tf.linalg.qr(input, full_matrices=False, name=None)**
QR decomposition.

**tf.linalg.svd(tensor, full_matrices=False, compute_uv=True, name=None)**
Singular value decomposition.

**tf.linalg.eig(tensor, name=None)**
Eigenvalue decomposition (general).

**tf.linalg.eigh(tensor, name=None)**
Eigenvalue decomposition (Hermitian/symmetric).

**tf.linalg.slogdet(input, name=None)**
Sign and log of the absolute value of the determinant.

### Matrix Functions

**tf.linalg.inv(input, adjoint=False, name=None)**
Matrix inverse.

**tf.linalg.solve(matrix, rhs, adjoint=False, name=None)**
Solve linear system: matrix @ x = rhs.

**tf.linalg.triangular_solve(matrix, rhs, lower=True, adjoint=False, name=None)**
Solve triangular linear system.

**tf.linalg.tridiagonal_solve(diagonals, rhs, diagonals_format='compact', transpose_rhs=False, conjugate_rhs=False, name=None, partial_pivoting=True)**
Solve tridiagonal linear system.

**tf.linalg.matrix_exp(input, name=None)**
Matrix exponential.

**tf.linalg.matrix_power(input, n, name=None)**
Matrix power: A^n.

**tf.linalg.matrix_square_root(input, name=None)**
Matrix square root.

**tf.linalg.adjoint(matrix, name=None)**
Adjoint (conjugate transpose) of a matrix.

**tf.linalg.matrix_transpose(a, name='matrix_transpose')**
Transpose the last two dimensions.

**tf.linalg.band_part(input, num_lower, num_upper, name=None)**
Copy a tensor setting elements outside a central band to zero.

**tf.linalg.cross(a, b, name=None)**
Compute pairwise cross product.

**tf.linalg.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.dtypes.float32, name=None)**
Create an identity matrix.

**tf.linalg.pinv(a, rcond=None, validate_args=False, name=None)**
Moore-Penrose pseudo-inverse.

**tf.linalg.global_norm(t_list, name=None)**
Compute the global norm of multiple tensors.

---

## tf.image

### Decoding and Encoding

**tf.image.decode_bmp(contents, channels=0, name=None)**
Decode a BMP-encoded image.

**tf.image.decode_gif(contents, name=None)**
Decode a GIF-encoded image.

**tf.image.decode_jpeg(contents, channels=0, ratio=1, fancy_upscaling=True, try_recover_truncated=False, acceptable_fraction=1, dct_method='', name=None)**
Decode a JPEG-encoded image.

**tf.image.decode_png(contents, channels=0, dtype=tf.dtypes.uint8, name=None)**
Decode a PNG-encoded image.

**tf.image.decode_image(contents, channels=None, dtype=tf.dtypes.uint8, name=None, expand_animations=True)**
Auto-detect and decode any supported image format.

**tf.image.encode_jpeg(image, format='', quality=95, progressive=False, optimize_size=False, chroma_downsampling=True, density_unit='in', x_density=300, y_density=300, xmp_metadata='', name=None)**
Encode an image to JPEG.

**tf.image.encode_png(image, compression=-1, name=None)**
Encode an image to PNG.

### Resizing

**tf.image.resize(images, size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None)**
Resize images to the given size. Methods:
- `bilinear`: Bilinear interpolation.
- `nearest`: Nearest neighbor.
- `bicubic`: Bicubic interpolation.
- `area`: Area interpolation.
- `lanczos3`, `lanczos5`: Lanczos interpolation.
- `gaussian`: Gaussian interpolation.
- `mitchellcubic`: Mitchell-Cubic interpolation.

**tf.image.resize_with_pad(images, target_height, target_width, method=ResizeMethod.BILINEAR, antialias=False)**
Resize and pad to target dimensions, preserving aspect ratio.

**tf.image.resize_with_crop_or_pad(image, target_height, target_width)**
Crop and/or pad to target dimensions.

**tf.image.crop_and_resize(image, boxes, box_indices, crop_size, method='bilinear', extrapolation_value=0, name=None)**
Extract crops and resize.

### Color Space Conversion

**tf.image.convert_image_dtype(image, dtype, saturate=False, name=None)**
Convert image dtype, scaling values appropriately.

**tf.image.rgb_to_grayscale(images, name=None)**
Convert RGB to grayscale.

**tf.image.grayscale_to_rgb(images, name=None)**
Convert grayscale to RGB.

**tf.image.rgb_to_hsv(images, name=None)**
Convert RGB to HSV.

**tf.image.hsv_to_rgb(images, name=None)**
Convert HSV to RGB.

**tf.image.rgb_to_yiq(images, name=None)**
Convert RGB to YIQ.

**tf.image.yiq_to_rgb(images, name=None)**
Convert YIQ to RGB.

**tf.image.rgb_to_yuv(images, name=None)**
Convert RGB to YUV.

**tf.image.yuv_to_rgb(images, name=None)**
Convert YUV to RGB.

### Adjustments

**tf.image.adjust_brightness(image, delta, name=None)**
Adjust brightness by delta.

**tf.image.adjust_contrast(images, contrast_factor, name=None)**
Adjust contrast by contrast_factor.

**tf.image.adjust_gamma(image, gamma=1, gain=1, name=None)**
Apply gamma correction.

**tf.image.adjust_hue(image, delta, name=None)**
Adjust hue by delta (in [-0.5, 0.5]).

**tf.image.adjust_jpeg_quality(image, jpeg_quality, name=None)**
Adjust JPEG encoding quality.

**tf.image.adjust_saturation(image, saturation_factor, name=None)**
Adjust saturation by saturation_factor.

### Random Augmentation

**tf.image.random_brightness(image, max_delta, seed=None)**
Randomly adjust brightness.

**tf.image.random_contrast(image, lower, upper, seed=None)**
Randomly adjust contrast.

**tf.image.random_hue(image, max_delta, seed=None)**
Randomly adjust hue.

**tf.image.random_saturation(image, lower, upper, seed=None)**
Randomly adjust saturation.

**tf.image.random_crop(value, size, seed=None, name=None)**
Randomly crop a tensor.

**tf.image.random_flip_left_right(image, seed=None)**
Randomly flip horizontally.

**tf.image.random_flip_up_down(image, seed=None)**
Randomly flip vertically.

### Deterministic Transformations

**tf.image.flip_left_right(image, name=None)**
Flip horizontally.

**tf.image.flip_up_down(image, name=None)**
Flip vertically.

**tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)**
Crop to bounding box.

**tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)**
Pad to bounding box.

**tf.image.extract_glimpse(input, size, offsets, centered=True, normalized=True, uniform_noise=True, noise='uniform', name=None)**
Extract a rectangular glimpse from a tensor.

**tf.image.extract_image_patches(images, ksizes, strides, rates, padding, name=None)**
Extract patches from images.

**tf.image.extract_patches(images, sizes, strides, rates, padding, name=None)**
Extract patches from images.

### Detection Utilities

**tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), name=None)**
Greedily select a subset of bounding boxes in descending score order.

**tf.image.non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=float('-inf'), soft_nms_sigma=0.0, name=None)**
NMS with soft-NMS option.

**tf.image.combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold=0.5, score_threshold=float('-inf'), pad_per_class=False, clip_boxes=True, name=None)**
Batched NMS across multiple classes.

**tf.image.draw_bounding_boxes(images, boxes, colors=None, name=None)**
Draw bounding boxes on batch of images.

**tf.image.sample_distorted_bounding_box(image_size, bounding_boxes, seed=None, min_object_covered=0.1, aspect_ratio_range=None, area_range=None, max_attempts=None, use_image_if_no_bounding_boxes=None, name=None)**
Generate randomly distorted bounding box.

### Quality Metrics

**tf.image.total_variation(images, name=None)**
Compute total variation: sum of absolute differences of adjacent pixels.

**tf.image.ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, name=None)**
Compute Structural Similarity Index (SSIM).

**tf.image.psnr(a, b, max_val, name=None)**
Compute Peak Signal-to-Noise Ratio.

### Gradient

**tf.image.image_gradients(image, name=None)**
Compute image gradients (dy, dx) using finite differences.

**tf.image.sobel_edges(image, name=None)**
Compute Sobel edge maps.

---

## tf.signal

### Discrete Fourier Transforms

**tf.signal.fft(input, name=None)**
1D Fast Fourier Transform (complex input/output).

**tf.signal.ifft(input, name=None)**
1D Inverse FFT.

**tf.signal.fft2d(input, name=None)**
2D FFT.

**tf.signal.ifft2d(input, name=None)**
2D Inverse FFT.

**tf.signal.fft3d(input, name=None)**
3D FFT.

**tf.signal.ifft3d(input, name=None)**
3D Inverse FFT.

### Real-Valued FFTs

**tf.signal.rfft(input_tensor, fft_length=None, name=None)**
1D Real FFT. Real input, complex output.

**tf.signal.irfft(input, fft_length=None, name=None)**
1D Inverse Real FFT. Complex input, real output.

**tf.signal.rfft2d(input_tensor, fft_length=None, name=None)**
2D Real FFT.

**tf.signal.irfft2d(input, fft_length=None, name=None)**
2D Inverse Real FFT.

**tf.signal.rfft3d(input_tensor, fft_length=None, name=None)**
3D Real FFT.

**tf.signal.irfft3d(input, fft_length=None, name=None)**
3D Inverse Real FFT.

### Short-Time Fourier Transform

**tf.signal.stft(signals, frame_length, frame_step, fft_length=None, window_fn=tf.signal.hann_window, pad_end=False, name=None)**
Compute the Short-Time Fourier Transform.
```python
signal = tf.random.normal([1, 16000])
stft = tf.signal.stft(signal, frame_length=400, frame_step=160)
# shape: [1, 98, 257] (batch, frames, fft_bins)
```

**tf.signal.inverse_stft(stfts, frame_length, frame_step, fft_length=None, window_fn=tf.signal.hann_window, name=None)**
Compute the Inverse STFT.

### Window Functions

**tf.signal.hann_window(window_length, periodic=True, dtype=tf.dtypes.float32, name=None)**
Hann window function.

**tf.signal.hamming_window(window_length, periodic=True, dtype=tf.dtypes.float32, name=None)**
Hamming window function.

**tf.signal.kaiser_window(window_length, beta=12.0, dtype=tf.dtypes.float32, name=None)**
Kaiser window function.

**tf.signal.vorbis_window(window_length, dtype=tf.dtypes.float32, name=None)**
Vorbis window function.

### Framing

**tf.signal.frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1, name=None)**
Expand signal into frames (overlapping windows).
```python
signal = tf.range(10)
tf.signal.frame(signal, frame_length=3, frame_step=2)
# [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]]
```

---

## tf.strings

### Basic Operations

**tf.strings.join(inputs, separator='', name=None)**
Join strings.
```python
tf.strings.join(['hello', 'world'], separator=' ')  # 'hello world'
```

**tf.strings.lower(input, encoding='', name=None)**
Convert to lowercase.

**tf.strings.upper(input, encoding='', name=None)**
Convert to uppercase.

**tf.strings.split(input, sep=None, maxsplit=-1, name=None)**
Split strings.
```python
tf.strings.split('hello world', ' ')  # ['hello', 'world']
```

**tf.strings.substr(input, pos, len, unit='BYTE', name=None)**
Extract substring.

**tf.strings.strip(input, name=None)**
Remove leading and trailing whitespace.

**tf.strings.regex_replace(input, pattern, rewrite, replace_global=True, name=None)**
Replace using regular expression.
```python
tf.strings.regex_replace('Hello 123', '[0-9]', 'X')  # 'Hello XXX'
```

**tf.strings.regex_full_match(input, pattern, name=None)**
Check if strings fully match a regex pattern.

**tf.strings.to_number(input, out_type=tf.dtypes.float32, name=None)**
Convert strings to numbers.

**tf.strings.to_hash_bucket(input, num_buckets, name=None)**
Hash strings to bucket IDs.

**tf.strings.to_hash_bucket_fast(input, num_buckets, name=None)**
Faster hash bucket (non-deterministic across runs).

**tf.strings.length(input, unit='BYTE', name=None)**
Get string lengths.

**tf.strings.format(template, inputs, placeholder='{}', summarize=3, name=None)**
Format strings using a template.
```python
tf.strings.format('{} + {} = {}', [1, 2, 3])  # '1 + 2 = 3'
```

**tf.strings.printable(input, name=None)**
Get printable representation.

**tf.strings.bytes_split(input, name=None)**
Split strings into bytes.

**tf.strings.reduce_join(inputs, axis=None, keepdims=False, separator='', name=None)**
Join string tensors along an axis.

**tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments, separator='', name=None)**
Join strings by segment.

### Unicode Operations

**tf.strings.unicode_decode(input, input_encoding, errors='replace', replacement_char=65533, replace_control_characters=False, name=None)**
Decode Unicode strings to code points.

**tf.strings.unicode_encode(input, output_encoding, errors='replace', replacement_char=65533, name=None)**
Encode code points to Unicode strings.

**tf.strings.unicode_script(input, name=None)**
Get Unicode script of code points.

**tf.strings.unicode_transcode(input, input_encoding, output_encoding, errors='replace', replacement_char=65533, replace_control_characters=False, name=None)**
Transcode between Unicode encodings.

---

## tf.random

### Random Tensors

**tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)**
Random normal distribution.
```python
tf.random.normal([2, 3], mean=0.0, stddev=1.0)
```

**tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)**
Random uniform distribution.
```python
tf.random.uniform([2, 3], minval=0, maxval=10)
```

**tf.random.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)**
Truncated normal (values beyond 2 stddevs are discarded and re-drawn).

**tf.random.shuffle(value, seed=None, name=None)**
Randomly shuffle a tensor along its first dimension.
```python
tf.random.shuffle(tf.range(10))
```

**tf.random.categorical(logits, num_samples, dtype=None, seed=None, name=None)**
Draw samples from a categorical distribution.
```python
tf.random.categorical(tf.math.log([[0.1, 0.4, 0.5]]), 5)
# Draw 5 samples from the distribution
```

**tf.random.gamma(shape, alpha, beta=None, dtype=tf.dtypes.float32, seed=None, name=None)**
Draw samples from a Gamma distribution.

**tf.random.poisson(shape, lam, dtype=tf.dtypes.float32, seed=None, name=None)**
Draw samples from a Poisson distribution.

### Seed Control

**tf.random.set_seed(seed)**
Set the global random seed.
```python
tf.random.set_seed(42)
a = tf.random.normal([2])
tf.random.set_seed(42)
b = tf.random.normal([2])
# a == b
```

### Generator

**tf.random.Generator(copy_from=None, state=None, alg=None, name=None)**
Stateful random number generator.
```python
g = tf.random.Generator.from_seed(42)
a = g.normal([2, 3])
b = g.uniform([2, 3])
```

Methods:
- `from_seed(seed, alg=None)`: Create from seed.
- `from_state(state, alg)`: Create from state.
- `from_non_deterministic_state(alg=None)`: Create from non-deterministic state.
- `normal(shape, ...)`: Normal distribution.
- `uniform(shape, ...)`: Uniform distribution.
- `truncated_normal(shape, ...)`: Truncated normal.
- `shuffle(value, ...)`: Shuffle.
- `binomial(shape, counts, probs, ...)`: Binomial distribution.
- `state`: The current PRNG state.
- `algorithm`: The PRNG algorithm being used.
- `reset(state)`: Reset to a specific state.

### Stateful Distributions

**tf.random.StatefulUniform(shape, seed, minval=0, maxval=None, dtype=tf.dtypes.float32, name=None)**

**tf.random.StatefulNormal(shape, seed, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None)**

**tf.random.StatefulTruncatedNormal(shape, seed, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None)**

---

## tf.io

### File I/O

**tf.io.read_file(filename, name=None)**
Read entire file contents as a string tensor.

**tf.io.write_file(filename, contents, name=None)**
Write string contents to a file.

### Serialization

**tf.io.decode_raw(bytes, out_type, little_endian=True, name=None)**
Decode raw bytes into a tensor.

**tf.io.encode_base64(input, pad=False, name=None)**
Encode string tensor to base64.

**tf.io.decode_base64(input, name=None)**
Decode base64 string tensor.

**tf.io.serialize_tensor(tensor, name=None)**
Serialize a tensor to a string tensor.

**tf.io.deserialize_tensor(tensor, dtype=None, name=None)**
Deserialize a string tensor to a typed tensor.

**tf.io.parse_tensor(serialized, out_type, name=None)**
Parse a serialized tensor proto.

### Data Format Parsing

**tf.io.decode_csv(records, record_defaults, field_delim=',', use_quote_delim=True, na_value='', select_cols=None, name=None)**
Decode CSV records into tensors.
```python
tf.io.decode_csv(
    '1,2,3\n4,5,6',
    record_defaults=[tf.constant(0, dtype=tf.int32)] * 3
)
# [[1, 4], [2, 5], [3, 6]]
```

**tf.io.decode_json_example(json_examples, name=None)**
Decode JSON-encoded example records to binary string tensors.

**tf.io.parse_example(serialized, features, name=None)**
Parse Example protos into a dict of tensors.
```python
tf.io.parse_example(
    serialized_examples,
    features={
        'x': tf.io.FixedLenFeature([3], tf.float32),
        'y': tf.io.VarLenFeature(tf.int64)
    }
)
```

**tf.io.parse_single_example(serialized, features, name=None)**
Parse a single Example proto.

**tf.io.parse_sequence_example(serialized, context_features=None, sequence_features=None, example_names=None, name=None)**
Parse SequenceExample protos.

### Feature Specifications

**tf.io.FixedLenFeature(shape, dtype, default_value=None)**
Feature with fixed shape and optional default.

**tf.io.FixedLenSequenceFeature(shape, dtype, allow_missing=False, default_value=None)**
Variable-length feature with fixed-size elements.

**tf.io.VarLenFeature(dtype)**
Variable-length feature.

**tf.io.SparseFeature(index_key, value_key, dtype, size, already_sorted=False)**
Sparse feature specification.

### TFRecord

**tf.io.TFRecordWriter(path, options=None)**
Write records to a TFRecords file.
```python
writer = tf.io.TFRecordWriter('/tmp/data.tfrecord')
writer.write(example.SerializeToString())
writer.close()
```

**tf.io.tf_record_iterator(path, options=None)**
Iterator over records in a TFRecord file.

**tf.io.TFRecordOptions(compression_type=None, flush_mode=None, buffer_size=None, window_bits=None, compression_level=None, compression_method=None, mem_level=None, compression_strategy=None, input_buffer_size=None, output_buffer_size=None, zstd_compression_level=None)**
Options for TFRecord reading/writing.

### GFile (File System Operations)

**tf.io.gfile.copy(src, dst, overwrite=False)**
Copy a file.

**tf.io.gfile.exists(path)**
Check if path exists.

**tf.io.gfile.glob(pattern)**
Get files matching a glob pattern.

**tf.io.gfile.isdir(path)**
Check if path is a directory.

**tf.io.gfile.listdir(path)**
List directory contents.

**tf.io.gfile.makedirs(path)**
Create directory and parents.

**tf.io.gfile.remove(path)**
Delete a file.

**tf.io.gfile.rename(oldname, newname, overwrite=False)**
Rename a file.

**tf.io.gfile.rmtree(path)**
Delete a directory tree.

**tf.io.gfile.stat(path)**
Get file statistics.

**tf.io.gfile.walk(path)**
Recursively walk a directory tree (like os.walk).

**tf.io.gfile.GFile(name, mode='r')**
File object for reading/writing. Supports local and remote filesystems.
```python
with tf.io.gfile.GFile('/tmp/file.txt', 'w') as f:
    f.write('hello')
with tf.io.gfile.GFile('/tmp/file.txt', 'r') as f:
    content = f.read()
```

### Other

**tf.io.matching_files(pattern, name=None)**
Get files matching a glob pattern (returns tensor).

**tf.io.print(input_, output_stream=None, name=None, **kwargs)**
Print a tensor in graph mode.
