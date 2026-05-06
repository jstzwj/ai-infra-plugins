# 07 — Relax Operators — Complete Reference

## Neural Network Operators (`R.nn`)

### Matrix Operations
| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.nn.matmul` | `(a, b, out_dtype=None)` | Matrix multiplication |
| `R.nn.linear` | `(data, weight, bias=None, out_dtype=None)` | Linear (dense) layer |

### Convolution
| Operator | Description |
|----------|-------------|
| `R.nn.conv1d` | 1D convolution |
| `R.nn.conv2d` | 2D convolution |
| `R.nn.conv3d` | 3D convolution |
| `R.nn.conv1d_transpose` | Transposed 1D convolution |
| `R.nn.conv2d_transpose` | Transposed 2D convolution |
| `R.nn.conv3d_transpose` | Transposed 3D convolution |
| `R.nn.group_conv2d` | Grouped 2D convolution |
| `R.nn.depthwise_conv2d` | Depthwise 2D convolution |

### Normalization
| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.nn.batch_norm` | `(data, gamma, beta, moving_mean, moving_var, ...)` | Batch normalization |
| `R.nn.layer_norm` | `(data, gamma, beta, axes, epsilon)` | Layer normalization |
| `R.nn.group_norm` | `(data, gamma, beta, num_groups, ...)` | Group normalization |
| `R.nn.rms_norm` | `(data, weight, axes, epsilon)` | RMS normalization |

### Activation
| Operator | Description |
|----------|-------------|
| `R.nn.relu` | ReLU: max(0, x) |
| `R.nn.gelu` | GELU activation |
| `R.nn.silu` | SiLU/Swish: x * sigmoid(x) |
| `R.nn.sigmoid` | Sigmoid: 1 / (1 + exp(-x)) |
| `R.nn.tanh` | Hyperbolic tangent |
| `R.nn.leaky_relu` | Leaky ReLU: max(alpha*x, x) |
| `R.nn.prelu` | Parametric ReLU |
| `R.nn.softmax` | Softmax along axis |
| `R.nn.log_softmax` | Log-softmax along axis |

### Pooling
| Operator | Description |
|----------|-------------|
| `R.nn.max_pool2d` | Max pooling 2D |
| `R.nn.avg_pool2d` | Average pooling 2D |
| `R.nn.adaptive_avg_pool2d` | Adaptive average pooling 2D |
| `R.nn.adaptive_max_pool2d` | Adaptive max pooling 2D |
| `R.nn.global_avg_pool2d` | Global average pooling 2D |

### Other NN
| Operator | Description |
|----------|-------------|
| `R.nn.bias_add` | Add bias to data |
| `R.nn.dense` | Dense (fully connected) layer |
| `R.nn.sparse_dense` | Sparse dense layer |
| `R.nn.pad` | Pad tensor |
| `R.nn.resize2d` | 2D resize (bilinear, nearest, bicubic) |
| `R.nn.upsampling` | 2D upsampling |
| `R.nn.nll_loss` | Negative log-likelihood loss |
| `R.nn.cross_entropy_with_logits` | Cross-entropy loss |
| `R.nn.mse_loss` | Mean squared error loss |

---

## Math Operators (`R`)

### Arithmetic
| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.add` | `(x1, x2)` | Element-wise addition |
| `R.subtract` | `(x1, x2)` | Element-wise subtraction |
| `R.multiply` | `(x1, x2)` | Element-wise multiplication |
| `R.divide` | `(x1, x2)` | Element-wise division |
| `R.floor_divide` | `(x1, x2)` | Floor division |
| `R.power` | `(x1, x2)` | Element-wise power |
| `R.mod` | `(x1, x2)` | Element-wise modulo |
| `R.neg` | `(x)` | Element-wise negation |
| `R.abs` | `(x)` | Absolute value |
| `R.sign` | `(x)` | Sign function |

### Exponential/Logarithmic
| Operator | Description |
|----------|-------------|
| `R.exp` | Exponential: e^x |
| `R.log` | Natural logarithm |
| `R.log2` | Base-2 logarithm |
| `R.log10` | Base-10 logarithm |
| `R.sqrt` | Square root |
| `R.rsqrt` | Reciprocal square root: 1/sqrt(x) |

### Trigonometric
| Operator | Description |
|----------|-------------|
| `R.sin` | Sine |
| `R.cos` | Cosine |
| `R.tan` | Tangent |
| `R.asin` | Arc sine |
| `R.acos` | Arc cosine |
| `R.atan` | Arc tangent |
| `R.sinh` | Hyperbolic sine |
| `R.cosh` | Hyperbolic cosine |
| `R.tanh` | Hyperbolic tangent |

### Rounding
| Operator | Description |
|----------|-------------|
| `R.floor` | Floor |
| `R.ceil` | Ceiling |
| `R.round` | Round to nearest integer |
| `R.clip` | Clip values to [min, max] |

### Comparison
| Operator | Description |
|----------|-------------|
| `R.maximum` | Element-wise maximum |
| `R.minimum` | Element-wise minimum |

### Reduction
| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.sum` | `(data, axis=None, keepdims=False)` | Sum reduction |
| `R.prod` | `(data, axis=None, keepdims=False)` | Product reduction |
| `R.max` | `(data, axis=None, keepdims=False)` | Max reduction |
| `R.min` | `(data, axis=None, keepdims=False)` | Min reduction |
| `R.argmax` | `(data, axis=None)` | Argmax |
| `R.argmin` | `(data, axis=None)` | Argmin |
| `R.mean` | `(data, axis=None, keepdims=False)` | Mean |
| `R.variance` | `(data, axis=None, keepdims=False)` | Variance |
| `R.std` | `(data, axis=None, keepdims=False)` | Standard deviation |
| `R.cumsum` | `(data, axis=None)` | Cumulative sum |
| `R.all` | `(data, axis=None)` | Logical AND reduction |
| `R.any` | `(data, axis=None)` | Logical OR reduction |

---

## Tensor Manipulation

### Shape Operations
| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.reshape` | `(data, new_shape)` | Reshape tensor |
| `R.expand_dims` | `(data, axis)` | Add dimension |
| `R.squeeze` | `(data, axis=None)` | Remove dimension(s) |
| `R.flatten` | `(data, start_dim=0, end_dim=-1)` | Flatten dimensions |
| `R.permute_dims` | `(data, axes=None)` | Permute dimensions |
| `R.transpose` | `(data, axes=None)` | Transpose (alias for permute_dims) |
| `R.broadcast_to` | `(data, shape)` | Broadcast to shape |
| `R.concat` | `(data, axis=None)` | Concatenate tensors |
| `R.split` | `(data, indices_or_sections, axis=0)` | Split tensor |
| `R.stack` | `(data, axis=0)` | Stack tensors along new axis |

### Indexing
| Operator | Description |
|----------|-------------|
| `R.take` | Take elements by index |
| `R.gather` | Gather elements by index |
| `R.scatter` | Scatter elements to indices |
| `R.scatter_elements` | Scatter elements |
| `R.scatter_nd` | Scatter into N-D tensor |
| `R.gather_nd` | Gather from N-D tensor |
| `R.strided_slice` | Slice with strides |
| `R.dynamic_strided_slice` | Dynamic strided slice |

### Transformation
| Operator | Description |
|----------|-------------|
| `R.repeat` | Repeat elements |
| `R.tile` | Tile tensor |
| `R.flip` | Flip along axis |
| `R.roll` | Roll elements |
| `R.unstack` | Unstack along axis |
| `R.meshgrid` | Create mesh grid |
| `R.where` | Conditional selection |

### Type Conversion
| Operator | Description |
|----------|-------------|
| `R.astype` | Cast data type |
| `R.reinterpret` | Reinterpret bits |
| `R.bitcast` | Bitcast to new type |

---

## Comparison Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.equal` | `(x1, x2)` | Element-wise equality |
| `R.not_equal` | `(x1, x2)` | Element-wise inequality |
| `R.greater` | `(x1, x2)` | Element-wise greater than |
| `R.greater_equal` | `(x1, x2)` | Element-wise greater or equal |
| `R.less` | `(x1, x2)` | Element-wise less than |
| `R.less_equal` | `(x1, x2)` | Element-wise less or equal |
| `R.logical_and` | `(x1, x2)` | Logical AND |
| `R.logical_or` | `(x1, x2)` | Logical OR |
| `R.logical_not` | `(x)` | Logical NOT |
| `R.logical_xor` | `(x1, x2)` | Logical XOR |
| `R.isnan` | `(x)` | Check for NaN |
| `R.isinf` | `(x)` | Check for infinity |
| `R.isfinite` | `(x)` | Check for finite values |

---

## Image Operators (`R.image`)

| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.image.resize1d` | `(data, size, layout, method, ...)` | 1D resize |
| `R.image.resize2d` | `(data, size, layout, method, ...)` | 2D resize |
| `R.image.resize3d` | `(data, size, layout, method, ...)` | 3D resize |
| `R.image.crop_and_resize` | `(data, boxes, box_indices, ...)` | Crop and resize |
| `R.image.affine_grid` | `(data, target_shape)` | Generate affine grid |
| `R.image.grid_sample` | `(data, grid, method, ...)` | Sample using grid |

Resize methods: `"bilinear"`, `"nearest"`, `"bicubic"`

---

## Vision Operators (`R.vision`)

| Operator | Description |
|----------|-------------|
| `R.vision.nms` | Non-maximum suppression |
| `R.vision.all_class_nms` | All-class NMS |
| `R.vision.topk` | Top-K elements |
| `R.vision.non_max_suppression` | Non-max suppression (general) |
| `R.vision.roi_align` | Region of interest align |

---

## Memory / Calling Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.call_tir` | `(func, args, out_sinfo)` | Call TIR PrimFunc |
| `R.call_dps_packed` | `(func, args, out_sinfo)` | Call packed function (dest-passing) |
| `R.call_pure_packed` | `(func, args, sinfo_args)` | Call pure packed function |
| `R.call_tir_with_grad` | `(func, args, out_sinfo, ...)` | call_tir with gradient support |
| `R.call_dps_packed_with_grad` | Call DPS packed with gradient |

---

## Creation Operators

| Operator | Signature | Description |
|----------|-----------|-------------|
| `R.full` | `(shape, fill_value, dtype)` | Create filled tensor |
| `R.full_like` | `(data, fill_value)` | Create filled tensor like data |
| `R.zeros` | `(shape, dtype)` | Create zero tensor |
| `R.zeros_like` | `(data)` | Create zero tensor like data |
| `R.ones` | `(shape, dtype)` | Create ones tensor |
| `R.ones_like` | `(data)` | Create ones tensor like data |
| `R.arange` | `(start, stop, step, dtype)` | Create range tensor |
| `R.linspace` | `(start, stop, num, dtype)` | Create linearly spaced tensor |
| `R.tril` | `(data, k=0)` | Lower triangle |
| `R.triu` | `(data, k=0)` | Upper triangle |
| `R.eye` | `(n, m, dtype)` | Identity matrix |

---

## Tuple / Structural Operators

| Operator | Description |
|----------|-------------|
| `R.tuple` | Create tuple from fields |
| `R.tuple_getitem` | Get tuple element by index |
| `R.shape_of` | Get shape of tensor |
| `R.null_value` | Create null value |

---

## Code Examples

### MLP Forward Pass
```python
from tvm.script import relax as R

@R.function
def mlp(
    x: R.Tensor(("n", 784), "float32"),
    w0: R.Tensor((784, 256), "float32"),
    b0: R.Tensor((256,), "float32"),
    w1: R.Tensor((256, 10), "float32"),
    b1: R.Tensor((10,), "float32"),
) -> R.Tensor(("n", 10), "float32"):
    with R.dataflow():
        lv0 = R.matmul(x, w0)
        lv0 = R.add(lv0, b0)
        lv1 = R.nn.relu(lv0)
        lv2 = R.matmul(lv1, w1)
        lv2 = R.add(lv2, b1)
        R.output(lv2)
    return lv2
```

### Conv2d Block
```python
@R.function
def conv_block(
    x: R.Tensor((1, 3, 224, 224), "float32"),
    w: R.Tensor((64, 3, 7, 7), "float32"),
    b: R.Tensor((64,), "float32"),
) -> R.Tensor((1, 64, 112, 112), "float32"):
    with R.dataflow():
        lv = R.nn.conv2d(x, w, strides=[2, 2], padding=[3, 3, 3, 3])
        lv = R.add(lv, b)  # bias_add
        lv = R.nn.relu(lv)
        R.output(lv)
    return lv
```

### Reduction and Statistics
```python
@R.function
def stats(
    x: R.Tensor(("n", "d"), "float32"),
) -> R.Tuple([R.Tensor(("d",), "float32"), R.Tensor(("d",), "float32")]):
    with R.dataflow():
        mean = R.mean(x, axis=0)
        var = R.variance(x, axis=0)
        R.output(mean, var)
    return (mean, var)
```
