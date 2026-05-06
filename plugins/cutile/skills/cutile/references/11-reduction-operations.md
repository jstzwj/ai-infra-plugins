# Reduction Operations

This chapter covers reduction operations in cuTile, which reduce tensor dimensions by computing aggregate values. Reductions are fundamental operations for statistics, normalization, and aggregation in deep learning and scientific computing.

## Overview

Reduction operations collapse one or more dimensions of a tile by applying a binary reduction function across elements. cuTile supports:

- **Basic reductions**: sum, max, min, prod
- **Index reductions**: argmax, argmin
- **Custom reductions**: user-defined reduction functions

All reduction operations follow similar semantics:
- Reduce along specified axes
- Remove reduced dimensions from output
- Support single-axis and multi-axis reduction
- Handle various data types appropriately

## Basic Reduction Operations

### `ct.sum(tile, axis)`

Computes the sum of elements along specified axes.

**Signature:**
```python
ct.sum(tile: Tile, axis: int | tuple[int, ...] | None = None) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis or axes to reduce (None = reduce all axes)

**Returns:**
- Tile with reduced dimensions removed

**Examples:**

**Full reduction (scalar output):**
```python
import cutile as ct

# Create 2D tile
matrix = ct.arange(1024, dtype=ct.float32).reshape((32, 32))

# Sum all elements: (32, 32) → scalar
total = ct.sum(matrix, axis=None)
print(total.shape)  # () - scalar
# Value: sum of 0 through 1023 = 523776
```

**Row-wise reduction (reduce along columns):**
```python
matrix = ct.arange(1024, dtype=ct.float32).reshape((32, 32))

# Sum along axis 1 (columns): (32, 32) → (32,)
row_sums = ct.sum(matrix, axis=1)
print(row_sums.shape)  # (32,)

# Each row sum is the sum of 32 consecutive integers
# Row 0: sum of 0-31 = 496
# Row 1: sum of 32-63 = 1520, etc.
```

**Column-wise reduction (reduce along rows):**
```python
# Sum along axis 0 (rows): (32, 32) → (32,)
col_sums = ct.sum(matrix, axis=0)
print(col_sums.shape)  # (32,)

# Each column sum samples elements at stride 32
# Col 0: sum of 0, 32, 64, ..., 992
```

**Multi-axis reduction:**
```python
# Create 4D tile: (batch, channels, height, width)
tile = ct.randn((8, 16, 32, 32), dtype=ct.float32)

# Reduce over height and width: (8, 16, 32, 32) → (8, 16)
spatial_sum = ct.sum(tile, axis=(2, 3))
print(spatial_sum.shape)  # (8, 16)

# Reduce over channels and spatial: (8, 16, 32, 32) → (8,)
feature_sum = ct.sum(tile, axis=(1, 2, 3))
print(feature_sum.shape)  # (8,)
```

**Keep dimensions (using expand_dims):**
```python
matrix = ct.randn((32, 32), dtype=ct.float32)

# Sum along rows: (32, 32) → (32,)
row_sums = ct.sum(matrix, axis=1)

# Keep dimension: (32,) → (32, 1)
row_sums_expanded = ct.expand_dims(row_sums, axis=1)
print(row_sums_expanded.shape)  # (32, 1)

# Now can broadcast for normalization
normalized = matrix / row_sums_expanded
```

**Use case - batch normalization:**
```python
def batch_normalize(x, eps=1e-5):
    """
    Normalize batch along feature dimension.
    
    Args:
        x: (batch_size, features) input
        eps: small constant for numerical stability
    
    Returns:
        (batch_size, features) normalized output
    """
    # Compute mean: (batch_size, features) → (features,)
    mean = ct.sum(x, axis=0) / x.shape[0]
    
    # Compute variance
    centered = x - ct.expand_dims(mean, axis=0)
    variance = ct.sum(centered * centered, axis=0) / x.shape[0]
    
    # Normalize
    std = ct.sqrt(variance + eps)
    normalized = centered / ct.expand_dims(std, axis=0)
    
    return normalized

# Example
batch = ct.randn((64, 128), dtype=ct.float32)
normalized = batch_normalize(batch)
print(normalized.shape)  # (64, 128)
```

### `ct.max(tile, axis)`

Computes the maximum value along specified axes.

**Signature:**
```python
ct.max(tile: Tile, axis: int | tuple[int, ...] | None = None) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis or axes to reduce (None = reduce all axes)

**Returns:**
- Tile with maximum values

**Examples:**

**Full reduction:**
```python
matrix = ct.randn((32, 32), dtype=ct.float32)

# Global maximum: scalar
global_max = ct.max(matrix, axis=None)
print(global_max.shape)  # ()
```

**Row-wise maximum:**
```python
# Max along columns: (32, 32) → (32,)
row_max = ct.max(matrix, axis=1)
print(row_max.shape)  # (32,)

# Each element is the max of one row
```

**Column-wise maximum:**
```python
# Max along rows: (32, 32) → (32,)
col_max = ct.max(matrix, axis=0)
print(col_max.shape)  # (32,)
```

**Multi-axis maximum:**
```python
# Create 4D tile
tile = ct.randn((8, 16, 32, 32), dtype=ct.float32)

# Max over spatial dimensions: (8, 16, 32, 32) → (8, 16)
spatial_max = ct.max(tile, axis=(2, 3))
print(spatial_max.shape)  # (8, 16)
```

**Use case - ReLU activation:**
```python
def relu(x):
    """
    ReLU activation: max(x, 0).
    
    While typically done element-wise, this shows how max
    can be used with broadcasting.
    """
    zero = ct.full(x.shape, 0.0, dtype=x.dtype)
    return ct.maximum(x, zero)

# Example
x = ct.randn((32, 32), dtype=ct.float32)
activated = relu(x)
print(activated.shape)  # (32, 32)
```

**Use case - max pooling:**
```python
def max_pool2d(x, pool_size=2):
    """
    Simple 2D max pooling.
    
    Args:
        x: (batch, channels, height, width)
        pool_size: size of pooling window
    
    Returns:
        (batch, channels, height//pool_size, width//pool_size)
    """
    batch, channels, height, width = x.shape
    
    # Reshape for pooling
    # (batch, channels, height, width) → 
    # (batch, channels, height//pool_size, pool_size, width//pool_size, pool_size)
    pooled_h = height // pool_size
    pooled_w = width // pool_size
    
    reshaped = ct.reshape(x, (batch, channels, pooled_h, pool_size, pooled_w, pool_size))
    
    # Max over pooling dimensions
    pooled = ct.max(reshaped, axis=(3, 5))
    
    return pooled

# Example
x = ct.randn((4, 16, 32, 32), dtype=ct.float32)
pooled = max_pool2d(x, pool_size=2)
print(pooled.shape)  # (4, 16, 16, 16)
```

### `ct.min(tile, axis)`

Computes the minimum value along specified axes.

**Signature:**
```python
ct.min(tile: Tile, axis: int | tuple[int, ...] | None = None) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis or axes to reduce (None = reduce all axes)

**Returns:**
- Tile with minimum values

**Examples:**

**Full reduction:**
```python
matrix = ct.randn((32, 32), dtype=ct.float32)

# Global minimum: scalar
global_min = ct.min(matrix, axis=None)
print(global_min.shape)  # ()
```

**Row-wise minimum:**
```python
# Min along columns: (32, 32) → (32,)
row_min = ct.min(matrix, axis=1)
print(row_min.shape)  # (32,)
```

**Use case - min-max normalization:**
```python
def min_max_normalize(x):
    """
    Normalize x to [0, 1] range using min and max.
    
    Args:
        x: (batch_size, features) input
    
    Returns:
        (batch_size, features) normalized to [0, 1]
    """
    # Compute min and max: (batch_size, features) → (features,)
    min_val = ct.min(x, axis=0)
    max_val = ct.max(x, axis=0)
    
    # Normalize: (x - min) / (max - min)
    range_val = max_val - min_val
    normalized = (x - ct.expand_dims(min_val, axis=0)) / ct.expand_dims(range_val, axis=0)
    
    return normalized

# Example
batch = ct.randn((64, 128), dtype=ct.float32)
normalized = min_max_normalize(batch)
print(normalized.shape)  # (64, 128)

# Verify range
print(f"Min: {ct.min(normalized)}")  # Should be ~0.0
print(f"Max: {ct.max(normalized)}")  # Should be ~1.0
```

### `ct.prod(tile, axis)`

Computes the product of elements along specified axes.

**Signature:**
```python
ct.prod(tile: Tile, axis: int | tuple[int, ...] | None = None) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis or axes to reduce (None = reduce all axes)

**Returns:**
- Tile with product values

**Examples:**

**Full reduction:**
```python
matrix = ct.full((4, 4), 2.0, dtype=ct.float32)

# Product of all elements: 2^16 = 65536
total_prod = ct.prod(matrix, axis=None)
print(total_prod.shape)  # ()
# Value: 65536.0
```

**Row-wise product:**
```python
# Product along columns: (4, 4) → (4,)
row_prod = ct.prod(matrix, axis=1)
print(row_prod.shape)  # (4,)
# Each element is 2^4 = 16
```

**Use case - computing probability product:**
```python
def joint_probability(probs):
    """
    Compute joint probability from independent probabilities.
    
    Args:
        probs: (batch_size, num_events) probabilities
    
    Returns:
        (batch_size,) joint probabilities
    """
    return ct.prod(probs, axis=1)

# Example
probs = ct.full((10, 5), 0.5, dtype=ct.float32)
joint = joint_probability(probs)
print(joint.shape)  # (10,)
# Each value is 0.5^5 = 0.03125
```

**Use case - factorial computation:**
```python
def factorial(n):
    """Compute n! for scalar n using product reduction."""
    if n == 0:
        return 1.0
    
    # Create sequence [1, 2, ..., n]
    seq = ct.arange(1, n + 1, dtype=ct.float32)
    
    # Product of all elements
    return ct.prod(seq, axis=None)

# Example
# 5! = 120
result = factorial(5)
print(result)  # 120.0
```

## Index Reduction Operations

### `ct.argmax(tile, axis)`

Finds the index of the maximum value along specified axes.

**Signature:**
```python
ct.argmax(tile: Tile, axis: int | None = None) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis to reduce (None = flatten and find global argmax)

**Returns:**
- Tile with indices (int32 dtype)

**Examples:**

**1D argmax:**
```python
# Create 1D tile
vec = ct.from_list([1.0, 5.0, 3.0, 9.0, 2.0], dtype=ct.float32)

# Find index of maximum: argmax = 3 (value 9.0)
idx = ct.argmax(vec, axis=None)
print(idx.shape)  # ()
print(idx)  # 3
```

**Row-wise argmax:**
```python
# Create matrix where each row has a different max position
matrix = ct.from_list([
    [1.0, 5.0, 3.0],   # max at index 1
    [9.0, 2.0, 4.0],   # max at index 0
    [3.0, 7.0, 1.0],   # max at index 1
], dtype=ct.float32)

# Argmax along columns: (3, 3) → (3,)
row_argmax = ct.argmax(matrix, axis=1)
print(row_argmax.shape)  # (3,)
# Values: [1, 0, 1]
```

**Column-wise argmax:**
```python
# Argmax along rows: (3, 3) → (3,)
col_argmax = ct.argmax(matrix, axis=0)
print(col_argmax.shape)  # (3,)
# Column 0: max is 9.0 at index 1
# Column 1: max is 7.0 at index 2
# Column 2: max is 4.0 at index 1
# Values: [1, 2, 1]
```

**Multi-dimensional argmax:**
```python
# For multi-dimensional, first flatten then find argmax
tensor = ct.randn((4, 8, 16), dtype=ct.float32)

# Flatten and find global argmax
flat_argmax = ct.argmax(tensor, axis=None)
print(flat_argmax.shape)  # ()

# Convert to 3D indices
total_size = tensor.numel
depth_idx = flat_argmax // (8 * 16)
remaining = flat_argmax % (8 * 16)
height_idx = remaining // 16
width_idx = remaining % 16
```

**Use case - classification prediction:**
```python
def predict_class(logits):
    """
    Predict class from logits using argmax.
    
    Args:
        logits: (batch_size, num_classes) raw scores
    
    Returns:
        (batch_size,) predicted class indices
    """
    return ct.argmax(logits, axis=1)

# Example
batch_size = 32
num_classes = 10
logits = ct.randn((batch_size, num_classes), dtype=ct.float32)

predictions = predict_class(logits)
print(predictions.shape)  # (32,)
```

**Use case - top-k accuracy:**
```python
def top_k_accuracy(logits, labels, k=5):
    """
    Compute top-k accuracy.
    
    Args:
        logits: (batch_size, num_classes)
        labels: (batch_size,) ground truth
        k: consider top k predictions
    
    Returns:
        scalar accuracy
    """
    batch_size, num_classes = logits.shape
    
    # Get top k predictions
    # For simplicity, we'll use argmax (top-1)
    predictions = ct.argmax(logits, axis=1)
    
    # Compare with labels
    correct = ct.astype(predictions == labels, ct.float32)
    accuracy = ct.sum(correct) / batch_size
    
    return accuracy

# Example
batch_size = 100
num_classes = 10
logits = ct.randn((batch_size, num_classes), dtype=ct.float32)
labels = ct.randint(0, num_classes, (batch_size,), dtype=ct.int32)

accuracy = top_k_accuracy(logits, labels)
print(f"Top-1 accuracy: {accuracy}")
```

### `ct.argmin(tile, axis)`

Finds the index of the minimum value along specified axes.

**Signature:**
```python
ct.argmin(tile: Tile, axis: int | None = None) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis to reduce (None = flatten and find global argmin)

**Returns:**
- Tile with indices (int32 dtype)

**Examples:**

**1D argmin:**
```python
# Create 1D tile
vec = ct.from_list([5.0, 2.0, 8.0, 1.0, 7.0], dtype=ct.float32)

# Find index of minimum: argmin = 3 (value 1.0)
idx = ct.argmin(vec, axis=None)
print(idx.shape)  # ()
print(idx)  # 3
```

**Row-wise argmin:**
```python
# Create matrix
matrix = ct.from_list([
    [5.0, 2.0, 8.0],   # min at index 1
    [9.0, 1.0, 4.0],   # min at index 1
    [3.0, 7.0, 0.0],   # min at index 2
], dtype=ct.float32)

# Argmin along columns: (3, 3) → (3,)
row_argmin = ct.argmin(matrix, axis=1)
print(row_argmin.shape)  # (3,)
# Values: [1, 1, 2]
```

**Use case - finding nearest neighbor:**
```python
def find_nearest(query, candidates):
    """
    Find nearest candidate to query using Euclidean distance.
    
    Args:
        query: (features,) query vector
        candidates: (num_candidates, features) candidate vectors
    
    Returns:
        scalar index of nearest candidate
    """
    # Compute squared distances: (num_candidates,)
    distances = ct.sum((candidates - query) ** 2, axis=1)
    
    # Find minimum distance index
    nearest_idx = ct.argmin(distances, axis=None)
    
    return nearest_idx

# Example
query = ct.randn((128,), dtype=ct.float32)
candidates = ct.randn((1000, 128), dtype=ct.float32)

nearest = find_nearest(query, candidates)
print(f"Nearest candidate index: {nearest}")
```

## Custom Reduction Operations

### `ct.reduce(tile, axis, init, fn)`

Performs a custom reduction using a user-defined binary function.

**Signature:**
```python
ct.reduce(tile: Tile, axis: int | tuple[int, ...], init: float, fn: callable) -> Tile
```

**Parameters:**
- `tile`: Input tile
- `axis`: Axis or axes to reduce
- `init`: Initial value for reduction
- `fn`: Binary reduction function `(a, b) -> result`

**Requirements:**
- `fn` must be associative: `(a op b) op c == a op (b op c)`
- `fn` must handle the input dtype
- `init` should be the identity element for the operation

**Examples:**

**Custom sum (reimplementing ct.sum):**
```python
def custom_sum(tile, axis=None):
    """Implement sum using reduce."""
    init = 0.0
    return ct.reduce(tile, axis, init, lambda a, b: a + b)

# Test
matrix = ct.arange(16, dtype=ct.float32).reshape((4, 4))
result = custom_sum(matrix, axis=1)
print(result.shape)  # (4,)
```

**Custom product (reimplementing ct.prod):**
```python
def custom_prod(tile, axis=None):
    """Implement product using reduce."""
    init = 1.0
    return ct.reduce(tile, axis, init, lambda a, b: a * b)

# Test
matrix = ct.full((4, 4), 2.0, dtype=ct.float32)
result = custom_prod(matrix, axis=1)
print(result.shape)  # (4,)
# Each value is 16.0
```

**Finding second-largest value:**
```python
def second_largest(tile, axis):
    """
    Find the second largest value along axis.
    
    This uses a custom reduction that tracks both max and second_max.
    """
    init = (-float('inf'), -float('inf'))
    
    def update(acc, val):
        max_val, second_max = acc
        if val > max_val:
            return (val, max_val)
        elif val > second_max:
            return (max_val, val)
        else:
            return acc
    
    result = ct.reduce(tile, axis, init, update)
    return result[1]  # Return second_max

# Example
matrix = ct.from_list([
    [1.0, 5.0, 3.0],
    [9.0, 2.0, 4.0],
    [3.0, 7.0, 1.0],
], dtype=ct.float32)

second_largest_row = second_largest(matrix, axis=1)
print(second_largest_row.shape)  # (3,)
# Row 0: second largest of [1, 5, 3] is 3
# Row 1: second largest of [9, 2, 4] is 4
# Row 2: second largest of [3, 7, 1] is 3
```

**Logical operations:**
```python
def logical_all(tile, axis=None):
    """Check if all elements are non-zero (truthy)."""
    init = True
    return ct.reduce(tile, axis, init, lambda a, b: a and b)

def logical_any(tile, axis=None):
    """Check if any element is non-zero (truthy)."""
    init = False
    return ct.reduce(tile, axis, init, lambda a, b: a or b)

# Example
mask = ct.from_list([
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
], dtype=ct.int32)

all_nonzero = logical_all(mask, axis=1)
print(all_nonzero.shape)  # (3,)
# Values: [False, True, False]

any_nonzero = logical_any(mask, axis=1)
print(any_nonzero.shape)  # (3,)
# Values: [True, True, False]
```

**Custom mean with count:**
```python
def mean_with_count(tile, axis):
    """
    Compute mean and track count during reduction.
    Returns tuple of (mean, count).
    """
    init = (0.0, 0)
    
    def update(acc, val):
        sum_val, count = acc
        return (sum_val + val, count + 1)
    
    result = ct.reduce(tile, axis, init, update)
    mean_val = result[0] / result[1]
    return mean_val

# Example
matrix = ct.arange(16, dtype=ct.float32).reshape((4, 4))
row_means = mean_with_count(matrix, axis=1)
print(row_means.shape)  # (4,)
```

## Complete Examples

### Example 1: Dot Product

```python
def dot_product(a, b):
    """
    Compute dot product using sum of elementwise product.
    
    Args:
        a: (N,) vector
        b: (N,) vector
    
    Returns:
        scalar dot product
    """
    # Elementwise product then sum
    return ct.sum(a * b, axis=None)

# Example
a = ct.arange(32, dtype=ct.float32)
b = ct.arange(32, dtype=ct.float32) * 2

result = dot_product(a, b)
print(f"Dot product: {result}")
# sum of i * 2i for i in [0, 31] = 2 * sum of i^2
```

### Example 2: Matrix Vector Multiplication

```python
def matvec(matrix, vector):
    """
    Matrix-vector multiplication using reductions.
    
    Args:
        matrix: (M, N)
        vector: (N,)
    
    Returns:
        (M,) result
    """
    # Broadcast multiply then sum along columns
    broadcasted = matrix * ct.expand_dims(vector, axis=0)  # (M, N)
    return ct.sum(broadcasted, axis=1)  # (M,)

# Example
M, N = 64, 128
matrix = ct.randn((M, N), dtype=ct.float32)
vector = ct.randn((N,), dtype=ct.float32)

result = matvec(matrix, vector)
print(result.shape)  # (64,)
```

### Example 3: Softmax Function

```python
def softmax(logits, axis=-1):
    """
    Compute softmax probabilities.
    
    Args:
        logits: (batch_size, num_classes) raw scores
        axis: dimension to normalize over
    
    Returns:
        (batch_size, num_classes) probabilities
    """
    # Subtract max for numerical stability
    max_val = ct.expand_dims(ct.max(logits, axis=axis), axis=axis)
    shifted = logits - max_val
    
    # Exponentiate
    exp_vals = ct.exp(shifted)
    
    # Normalize
    sum_val = ct.expand_dims(ct.sum(exp_vals, axis=axis), axis=axis)
    probs = exp_vals / sum_val
    
    return probs

# Example
batch_size = 32
num_classes = 10
logits = ct.randn((batch_size, num_classes), dtype=ct.float32)

probs = softmax(logits)
print(probs.shape)  # (32, 10)

# Verify sum to 1
row_sums = ct.sum(probs, axis=1)
print(f"Row sums (should be ~1.0): {row_sums}")
```

### Example 4: Layer Normalization

```python
def layer_norm(x, eps=1e-5):
    """
    Layer normalization.
    
    Args:
        x: (batch_size, seq_len, features) input
        eps: small constant for numerical stability
    
    Returns:
        (batch_size, seq_len, features) normalized output
    """
    # Compute mean and variance along features
    mean = ct.expand_dims(ct.sum(x, axis=-1), axis=-1)  # (batch, seq, 1)
    var = ct.expand_dims(ct.sum((x - mean) ** 2, axis=-1), axis=-1)  # (batch, seq, 1)
    
    # Normalize
    normalized = (x - mean) / ct.sqrt(var + eps)
    
    return normalized

# Example
batch_size, seq_len, features = 8, 64, 128
x = ct.randn((batch_size, seq_len, features), dtype=ct.float32)

normalized = layer_norm(x)
print(normalized.shape)  # (8, 64, 128)

# Verify statistics
mean = ct.sum(normalized, axis=-1) / features
variance = ct.sum((normalized - ct.expand_dims(mean, axis=-1)) ** 2, axis=-1) / features
print(f"Mean (should be ~0): {mean}")
print(f"Variance (should be ~1): {variance}")
```

### Example 5: Attention Mechanism

```python
def scaled_dot_product_attention(Q, K, V):
    """
    Scaled dot-product attention.
    
    Args:
        Q: (batch_size, seq_len, d_k) query
        K: (batch_size, seq_len, d_k) key
        V: (batch_size, seq_len, d_v) value
    
    Returns:
        (batch_size, seq_len, d_v) attention output
    """
    # Compute attention scores
    # Q @ K^T: (batch, seq_len, d_k) @ (batch, d_k, seq_len) = (batch, seq_len, seq_len)
    K_T = ct.transpose(K, axis0=-2, axis1=-1)  # (batch, seq_len, d_k) → (batch, d_k, seq_len)
    scores = ct.matmul(Q, K_T) / ct.sqrt(Q.shape[-1])  # Scale by sqrt(d_k)
    
    # Apply softmax along last dimension
    attn_weights = softmax(scores, axis=-1)
    
    # Apply attention weights to values
    output = ct.matmul(attn_weights, V)  # (batch, seq_len, seq_len) @ (batch, seq_len, d_v)
    
    return output

# Example
batch_size = 8
seq_len = 64
d_k = 32
d_v = 32

Q = ct.randn((batch_size, seq_len, d_k), dtype=ct.float32)
K = ct.randn((batch_size, seq_len, d_k), dtype=ct.float32)
V = ct.randn((batch_size, seq_len, d_v), dtype=ct.float32)

output = scaled_dot_product_attention(Q, K, V)
print(output.shape)  # (8, 64, 32)
```

### Example 6: Cross-Entropy Loss

```python
def cross_entropy_loss(logits, labels):
    """
    Compute cross-entropy loss.
    
    Args:
        logits: (batch_size, num_classes) raw scores
        labels: (batch_size,) ground truth class indices
    
    Returns:
        scalar loss
    """
    # Compute softmax probabilities
    probs = softmax(logits, axis=1)
    
    # Get negative log likelihood of true classes
    batch_size = logits.shape[0]
    
    # Convert labels to one-hot
    num_classes = logits.shape[1]
    one_hot = ct.zeros((batch_size, num_classes), dtype=ct.float32)
    
    # For each sample, set the true class to 1
    # (In practice, this would be done more efficiently)
    for i in range(batch_size):
        label = labels[i]
        one_hot[i, label] = 1.0
    
    # Compute loss: -sum(true_class * log(pred_class))
    log_probs = ct.log(probs + 1e-10)  # Small epsilon for numerical stability
    loss = -ct.sum(one_hot * log_probs) / batch_size
    
    return loss

# Example
batch_size = 32
num_classes = 10
logits = ct.randn((batch_size, num_classes), dtype=ct.float32)
labels = ct.randint(0, num_classes, (batch_size,), dtype=ct.int32)

loss = cross_entropy_loss(logits, labels)
print(f"Cross-entropy loss: {loss}")
```

### Example 7: Variance and Standard Deviation

```python
def variance(tile, axis=None):
    """
    Compute variance along axis.
    
    Args:
        tile: input tile
        axis: axis to reduce over
    
    Returns:
        variance
    """
    # Compute mean
    mean_val = ct.sum(tile, axis=axis) / tile.shape[axis]
    
    # Compute mean of squares
    if axis is not None:
        mean_val_expanded = ct.expand_dims(mean_val, axis=axis)
    else:
        mean_val_expanded = mean_val
    
    squared_diff = (tile - mean_val_expanded) ** 2
    var = ct.sum(squared_diff, axis=axis) / tile.shape[axis]
    
    return var

def std_dev(tile, axis=None):
    """Compute standard deviation."""
    return ct.sqrt(variance(tile, axis))

# Example
matrix = ct.randn((32, 64), dtype=ct.float32)

row_var = variance(matrix, axis=1)
print(row_var.shape)  # (32,)

row_std = std_dev(matrix, axis=1)
print(row_std.shape)  # (32,)
```

### Example 8: L2 Normalization

```python
def l2_normalize(x, axis=-1, eps=1e-8):
    """
    L2 normalize along axis.
    
    Args:
        x: input tile
        axis: axis to normalize over
        eps: small constant for numerical stability
    
    Returns:
        normalized tile
    """
    # Compute L2 norm
    norm = ct.sqrt(ct.sum(x ** 2, axis=axis) + eps)
    
    # Normalize
    if axis is not None:
        norm_expanded = ct.expand_dims(norm, axis=axis)
    else:
        norm_expanded = norm
    
    return x / norm_expanded

# Example
x = ct.randn((16, 128), dtype=ct.float32)

normalized = l2_normalize(x, axis=1)
print(normalized.shape)  # (16, 128)

# Verify unit norm
norms = ct.sqrt(ct.sum(normalized ** 2, axis=1))
print(f"L2 norms (should be ~1.0): {norms}")
```

## Best Practices

1. **Numerical Stability**: When computing softmax or log-sum-exp, subtract the maximum before exponentiation to avoid overflow.

2. **Epsilon Values**: Add small epsilon values (1e-5 to 1e-10) when dividing or taking logarithms to avoid division by zero or log(0).

3. **Dimension Preservation**: Use `expand_dims` after reduction to keep dimensions for broadcasting operations.

4. **Multi-Axis Reduction**: Use tuple of axes for efficient multi-axis reduction instead of sequential reductions.

5. **Custom Reductions**: Ensure custom reduction functions are associative for correct results.

6. **Memory Efficiency**: Reduction operations generally don't require extra memory allocation beyond the output tile.

7. **Data Types**: Be aware of precision issues when reducing across different data types, especially with float16.

8. **Performance**: Reduction operations are typically bandwidth-bound; consider data layout for optimal memory access patterns.
