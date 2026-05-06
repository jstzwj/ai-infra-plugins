# XLA Reference - Chapter 3: Broadcasting Semantics

This reference provides comprehensive documentation about XLA's broadcasting semantics. Broadcasting is the mechanism by which operations on arrays of different shapes are resolved by virtually expanding one or both operands to a common shape without actually copying data. Understanding broadcasting is essential because it is deeply integrated into XLA's operation semantics, affects compilation behavior, and influences the correctness and performance of compiled programs.

---

## 3.1 What Is Broadcasting?

Broadcasting is a technique borrowed from NumPy and array programming languages that allows element-wise operations (binary operations like addition, multiplication, comparison) to be applied to arrays with different but compatible shapes. Instead of requiring the programmer to explicitly expand smaller arrays to match larger ones, the system implicitly "broadcasts" the smaller array across the missing dimensions.

For example, adding a scalar `5` to a 1D array `[1, 2, 3]`:

```
[1, 2, 3] + 5  ==>  [1+5, 2+5, 3+5]  ==>  [6, 7, 8]
```

The scalar `5` is conceptually "stretched" or "broadcast" to the shape `[3]`, with each element equal to 5. In reality, no memory is allocated for the expanded array -- the operation simply uses the scalar value for each output element.

In XLA, broadcasting is handled at the HLO level and is represented by explicit `broadcast` instructions. Binary operations in XLA also support an optional `broadcast_dimensions` attribute that specifies how to broadcast the operands.

---

## 3.2 Broadcasting Principles

XLA supports two forms of broadcasting:

### 3.2.1 Explicit Broadcasting

Explicit broadcasting uses the dedicated `broadcast` HLO instruction to expand an array to a larger shape. This is a standalone operation that creates a new instruction in the HLO graph.

```
// Explicit broadcast of a scalar to a 1D array
scalar = f32[] constant(5.0)
broadcast.0 = f32[3] broadcast(scalar), dimensions={}
```

```
// Explicit broadcast of a 1D array to a 2D array
vector = f32[3] parameter(0)
broadcast.0 = f32[2, 3] broadcast(vector), dimensions={1}
```

The `dimensions` attribute specifies which dimensions of the output correspond to dimensions of the input. This is explained in detail in Section 3.5.

### 3.2.2 Implicit Broadcasting

Implicit broadcasting is the automatic broadcasting that occurs when binary operations (e.g., `add`, `mul`) are applied to operands of different shapes. XLA's implicit broadcasting follows a restricted subset of NumPy's broadcasting rules:

- **Scalar broadcasting**: A scalar (rank-0) is automatically broadcast to any shape.
- **No automatic NumPy-style broadcasting**: Unlike NumPy, XLA does not automatically align dimensions from the right. For example, adding `f32[3]` and `f32[2, 3]` is not automatically valid in XLA without specifying `broadcast_dimensions`.

For non-scalar operands with different ranks, XLA requires the programmer (or the framework frontend) to explicitly specify the `broadcast_dimensions` attribute on the binary operation.

---

## 3.3 Broadcasting Lower-Dimensional to Higher-Dimensional Arrays

### 3.3.1 The General Case

When broadcasting a lower-dimensional array to a higher-dimensional shape, the fundamental question is: which dimensions of the output correspond to which dimensions of the input?

For an input array of rank `M` being broadcast to output rank `N` (where `M < N`):

1. The input array is "placed" into the output array by mapping `M` of the `N` output dimensions to input dimensions.
2. The remaining `N - M` output dimensions are "broadcast" dimensions -- the single input value is replicated along these dimensions.
3. The mapping is specified by the `broadcast_dimensions` attribute: a list of `M` integers, where the i-th integer specifies which output dimension corresponds to the i-th input dimension.

### 3.3.2 Rules

The following rules govern broadcast dimension specification:

1. **Size matching**: For each input dimension `i`, the input size `input_dim[i]` must equal the output size `output_dim[broadcast_dimensions[i]]`, OR the input size must be 1 (degenerate dimension broadcasting -- see Section 3.7).

2. **Strictly increasing**: The `broadcast_dimensions` values must be in strictly increasing order: `broadcast_dimensions[0] < broadcast_dimensions[1] < ... < broadcast_dimensions[M-1]`.

3. **Valid range**: Each `broadcast_dimensions[i]` must satisfy `0 <= broadcast_dimensions[i] < N`.

4. **Unique mapping**: Each output dimension can be mapped to at most one input dimension.

### 3.3.3 Examples

#### Example 1: Scalar to 2D

Input: `f32[]` (scalar value `5.0`)
Output shape: `f32[2, 3]`
`broadcast_dimensions`: `{}` (empty, since input has rank 0)

```
scalar = f32[] constant(5.0)
broadcast.0 = f32[2, 3] broadcast(scalar), dimensions={}
// Result: [[5.0, 5.0, 5.0],
//          [5.0, 5.0, 5.0]]
```

The scalar is replicated across all dimensions of the output.

#### Example 2: 1D to 2D -- Broadcasting Along Rows

Input: `f32[3]` with values `[1, 2, 3]`
Output shape: `f32[2, 3]`
`broadcast_dimensions`: `{1}` (input dimension 0 maps to output dimension 1)

```
vector = f32[3] parameter(0)  // [1, 2, 3]
broadcast.0 = f32[2, 3] broadcast(vector), dimensions={1}
// Result: [[1, 2, 3],
//          [1, 2, 3]]
```

The vector is broadcast along output dimension 0 (rows). Each row of the output is a copy of the input vector.

#### Example 3: 1D to 2D -- Broadcasting Along Columns

Input: `f32[2]` with values `[1, 2]`
Output shape: `f32[2, 3]`
`broadcast_dimensions`: `{0}` (input dimension 0 maps to output dimension 0)

```
vector = f32[2] parameter(0)  // [1, 2]
broadcast.0 = f32[2, 3] broadcast(vector), dimensions={0}
// Result: [[1, 1, 1],
//          [2, 2, 2]]
```

The vector is broadcast along output dimension 1 (columns). Each column of the output is a copy of the input vector.

#### Example 4: 2D to 4D

Input: `f32[3, 224]` (e.g., a 1D signal with 3 channels)
Output shape: `f32[32, 3, 224, 224]` (batch of 32, 3 channels, 224x224 spatial)
`broadcast_dimensions`: `{1, 2}` (input dim 0 -> output dim 1, input dim 1 -> output dim 2)

```
signal = f32[3, 224] parameter(0)
broadcast.0 = f32[32, 3, 224, 224] broadcast(signal), dimensions={1, 2}
// The [3, 224] signal is broadcast along batch (dim 0) and width (dim 3)
// Each of the 32 batch elements and each of the 224 columns is a copy of the signal
```

---

## 3.4 Formal Definition of Broadcasting Dimensions

### 3.4.1 Mathematical Definition

Let `input` be an array with shape `[d_0, d_1, ..., d_{M-1}]` (rank M).
Let `output` be an array with shape `[D_0, D_1, ..., D_{N-1}]` (rank N), where `M <= N`.
Let `broadcast_dimensions = [b_0, b_1, ..., b_{M-1}]` be a strictly increasing sequence where `0 <= b_i < N`.

The broadcast maps each input element `input[i_0, i_1, ..., i_{M-1}]` to output positions:

```
output[j_0, j_1, ..., j_{N-1}]
```

where `j_{b_k} = i_k` for all `k` in `0..M-1`, and `j_m` can be any value in `0..D_m-1` for dimensions `m` not in `broadcast_dimensions`.

For the mapping to be valid:
- For each `k`, `d_k == D_{b_k}` OR `d_k == 1` (degenerate broadcasting).

### 3.4.2 Inverse Mapping

Given an output index `(j_0, j_1, ..., j_{N-1})`, the corresponding input index is:

```
(i_0, i_1, ..., i_{M-1}) = (j_{b_0}, j_{b_1}, ..., j_{b_{M-1}})
```

If `d_k == 1` (degenerate dimension), then `i_k = 0` regardless of `j_{b_k}`.

### 3.4.3 Output Element Computation

For each output index `(j_0, ..., j_{N-1})`:

```
output[j_0, ..., j_{N-1}] = input[j_{b_0}, j_{b_1}, ..., j_{b_{M-1}}]
```

(with degenerate dimension handling as described above).

---

## 3.5 Broadcasting with Degenerate Dimensions

A **degenerate dimension** is a dimension of size 1. When an input dimension is degenerate (size 1) and it is mapped to an output dimension of size greater than 1, the single value along the degenerate dimension is replicated across the entire output dimension.

### 3.5.1 Examples

#### Example 1: Broadcasting a Column Vector

Input: `f32[3, 1]` with values `[[1], [2], [3]]`
Output shape: `f32[3, 4]`
`broadcast_dimensions`: `{0, 1}`

```
col_vector = f32[3, 1] parameter(0)  // [[1], [2], [3]]
broadcast.0 = f32[3, 4] broadcast(col_vector), dimensions={0, 1}
// Result: [[1, 1, 1, 1],
//          [2, 2, 2, 2],
//          [3, 3, 3, 3]]
```

Input dimension 1 has size 1 (degenerate). It maps to output dimension 1 which has size 4. The single value is replicated 4 times.

#### Example 2: Broadcasting a Row Vector

Input: `f32[1, 4]` with values `[[1, 2, 3, 4]]`
Output shape: `f32[3, 4]`
`broadcast_dimensions`: `{0, 1}`

```
row_vector = f32[1, 4] parameter(0)  // [[1, 2, 3, 4]]
broadcast.0 = f32[3, 4] broadcast(row_vector), dimensions={0, 1}
// Result: [[1, 2, 3, 4],
//          [1, 2, 3, 4],
//          [1, 2, 3, 4]]
```

Input dimension 0 has size 1 (degenerate). It maps to output dimension 0 which has size 3. The single value is replicated 3 times.

#### Example 3: Broadcasting with Multiple Degenerate Dimensions

Input: `f32[1, 1]` with value `[[42]]`
Output shape: `f32[3, 4]`
`broadcast_dimensions`: `{0, 1}`

```
scalar_like = f32[1, 1] parameter(0)  // [[42]]
broadcast.0 = f32[3, 4] broadcast(scalar_like), dimensions={0, 1}
// Result: [[42, 42, 42, 42],
//          [42, 42, 42, 42],
//          [42, 42, 42, 42]]
```

Both input dimensions are degenerate. Both are replicated to produce a 3x4 array of 42s.

### 3.5.2 Degenerate Dimensions vs. Rank Reduction

Note the difference between a degenerate dimension and a lower-rank array:

- `f32[1, 3]` has rank 2 with a degenerate dimension 0.
- `f32[3]` has rank 1.

These are different shapes. Broadcasting `f32[1, 3]` to `f32[2, 3]` uses `broadcast_dimensions={0, 1}`. Broadcasting `f32[3]` to `f32[2, 3]` uses `broadcast_dimensions={1}`. Both produce the same result, but the mechanism differs.

---

## 3.6 Compatibility Rules

Two array shapes are **broadcast compatible** if they can be broadcast to a common output shape. The rules for compatibility depend on whether implicit or explicit broadcasting is used.

### 3.6.1 Implicit Compatibility (Binary Operations Without broadcast_dimensions)

For binary operations without explicit `broadcast_dimensions`, XLA applies limited implicit broadcasting:

1. **Identical shapes**: Two arrays with the same shape are always compatible. No broadcasting occurs.

2. **Scalar operand**: If one operand is a scalar (rank 0), it is broadcast to the shape of the other operand.

3. **Otherwise**: The shapes are incompatible, and the operation is invalid without specifying `broadcast_dimensions`.

This is more restrictive than NumPy's broadcasting, which automatically right-aligns shapes and broadcasts matching/unmatched dimensions.

### 3.6.2 Explicit Compatibility (With broadcast_dimensions)

When `broadcast_dimensions` is specified, compatibility is determined by the formal rules in Section 3.4:

1. The input rank must be less than or equal to the output rank.
2. `broadcast_dimensions` must be strictly increasing and all values must be valid output dimension indices.
3. For each mapped dimension, the input size must equal the output size or be 1 (degenerate).

### 3.6.3 NumPy-Style Compatibility (In-Memory)

While XLA's HLO does not directly implement NumPy-style broadcasting, the framework frontends (JAX, TensorFlow, PyTorch) typically emulate NumPy broadcasting by inserting explicit broadcast instructions. The NumPy compatibility rules are:

1. **Right-align**: Shapes are aligned from the right (trailing dimensions first).
2. **Match or one**: Two dimensions are compatible if they are equal, or if one of them is 1.
3. **Missing dimensions**: If one array has fewer dimensions, it is padded with size-1 dimensions on the left.

Example NumPy-style broadcasting:

```
Shape A: (5, 4, 1)
Shape B: (    4, 3)
-------------------
Right-aligned:
A:     5 x 4 x 1
B:         4 x 3
Result: 5 x 4 x 3
```

In XLA, this would be expressed as:

```
a = f32[5, 4, 1] parameter(0)
b = f32[4, 3] parameter(1)
b_broadcast = f32[5, 4, 3] broadcast(b), dimensions={1, 2}
result = f32[5, 4, 3] add(a, b_broadcast)
```

Note: The input `a` also needs degenerate broadcasting from `f32[5, 4, 1]` to `f32[5, 4, 3]`, which would be handled by adding it as well or by the add operation's implicit degenerate handling.

---

## 3.7 Broadcast Composition

Broadcasts can be composed -- broadcasting a broadcast result produces the same result as a single broadcast to the final shape.

### 3.7.1 Composition Rule

If `A` is broadcast to `B` with dimensions `dims1`, and `B` is broadcast to `C` with dimensions `dims2`, then broadcasting `A` directly to `C` is equivalent if the resulting dimensions are computed correctly.

### 3.7.2 Example

```
// Step 1: Broadcast scalar to 1D
scalar = f32[] constant(1.0)
vec = f32[3] broadcast(scalar), dimensions={}

// Step 2: Broadcast 1D to 2D
matrix = f32[2, 3] broadcast(vec), dimensions={1}

// Equivalent single step:
matrix_direct = f32[2, 3] broadcast(scalar), dimensions={}
```

Both paths produce a `f32[2, 3]` array filled with `1.0`.

### 3.7.3 Optimizer Handling

The XLA optimizer may simplify composed broadcasts:
- A broadcast of a broadcast may be collapsed into a single broadcast.
- A broadcast followed by a binary operation may be fused into the binary operation's implicit broadcasting.
- Broadcasts of constants may be folded into constant arrays (if profitable).

---

## 3.8 InDim Broadcasting

**InDim broadcasting** refers to the mechanism by which degenerate (size-1) dimensions in an operand of a binary operation are implicitly broadcast without an explicit `broadcast` instruction. This is the XLA analog of NumPy's broadcasting within matching ranks.

### 3.8.1 How InDim Broadcasting Works

When a binary operation has two operands of the same rank but with different dimension sizes, InDim broadcasting applies if and only if for each dimension, the sizes either match exactly or one of them is 1.

For each dimension `d`:
- If `operand1.size[d] == operand2.size[d]`: no broadcasting needed.
- If `operand1.size[d] == 1` and `operand2.size[d] > 1`: `operand1` is broadcast along dimension `d`.
- If `operand1.size[d] > 1` and `operand2.size[d] == 1`: `operand2` is broadcast along dimension `d`.
- If `operand1.size[d] != operand2.size[d]` and neither is 1: the shapes are incompatible.

### 3.8.2 Example

```
operand1: f32[1, 4]  // shape [[a, b, c, d]]
operand2: f32[3, 1]  // shape [[x], [y], [z]]

add(operand1, operand2):
// operand1 is broadcast from [1, 4] to [3, 4] along dimension 0
// operand2 is broadcast from [3, 1] to [3, 4] along dimension 1
// Result: f32[3, 4]
// [[a+x, b+x, c+x, d+x],
//  [a+y, b+y, c+y, d+y],
//  [a+z, b+z, c+z, d+z]]
```

### 3.8.3 InDim vs. broadcast_dimensions

InDim broadcasting (degenerate dimensions within same-rank operands) and `broadcast_dimensions` (explicit dimension mapping for different-rank operands) are complementary mechanisms:

- **InDim**: Used when both operands have the same rank. Degenerate dimensions are implicitly broadcast.
- **broadcast_dimensions**: Used when operands have different ranks. Explicitly maps lower-rank operand dimensions to higher-rank dimensions.
- **Combined**: Both mechanisms can be active simultaneously -- `broadcast_dimensions` raises the rank, and then InDim broadcasting handles degenerate dimensions.

---

## 3.9 Comprehensive Examples

### Example 1: Simple Scalar Addition

```
HloModule scalar_add

ENTRY main {
  a = f32[3] parameter(0)       // [1.0, 2.0, 3.0]
  b = f32[] parameter(1)         // 10.0
  b_broadcast = f32[3] broadcast(b), dimensions={}
  ROOT result = f32[3] add(a, b_broadcast)
  // Result: [11.0, 12.0, 13.0]
}
```

### Example 2: Vector-Matrix Addition (Broadcast Along Rows)

```
HloModule vector_matrix_add

ENTRY main {
  matrix = f32[3, 4] parameter(0)
  // [[1, 2, 3, 4],
  //  [5, 6, 7, 8],
  //  [9, 10, 11, 12]]
  
  vector = f32[4] parameter(1)   // [100, 200, 300, 400]
  
  // Broadcast vector from rank 1 to rank 2
  // broadcast_dimensions={1} maps input dim 0 to output dim 1
  vector_broadcast = f32[3, 4] broadcast(vector), dimensions={1}
  // [[100, 200, 300, 400],
  //  [100, 200, 300, 400],
  //  [100, 200, 300, 400]]
  
  ROOT result = f32[3, 4] add(matrix, vector_broadcast)
  // [[101, 202, 303, 404],
  //  [105, 206, 307, 408],
  //  [109, 210, 311, 412]]
}
```

### Example 3: Column Broadcast for Bias Addition

```
HloModule bias_add

ENTRY main {
  activations = f32[128, 512] parameter(0)   // batch of 128, 512 features
  bias = f32[512] parameter(1)               // bias vector
  
  // Broadcast bias to match activations shape
  // broadcast_dimensions={1} maps bias dim 0 to activations dim 1
  bias_broadcast = f32[128, 512] broadcast(bias), dimensions={1}
  
  ROOT result = f32[128, 512] add(activations, bias_broadcast)
}
```

### Example 4: Degenerate Broadcasting in Binary Op

```
HloModule degenerate_broadcast

ENTRY main {
  // operand1: f32[2, 1, 3]
  a = f32[2, 1, 3] parameter(0)
  // [[[1, 2, 3]],
  //  [[4, 5, 6]]]
  
  // operand2: f32[1, 4, 3]
  b = f32[1, 4, 3] parameter(1)
  // [[[10, 20, 30],
  //   [40, 50, 60],
  //   [70, 80, 90],
  //   [100, 110, 120]]]
  
  // InDim broadcasting:
  // dim 0: a has 2, b has 1 -> b broadcast along dim 0
  // dim 1: a has 1, b has 4 -> a broadcast along dim 1
  // dim 2: both have 3 -> no broadcast
  ROOT result = f32[2, 4, 3] add(a, b)
  // For each i in 0..1, j in 0..3, k in 0..2:
  //   result[i,j,k] = a[i,0,k] + b[0,j,k]
  //
  // result[0,:,:] = [[1+10, 2+20, 3+30],    = [[11, 22, 33],
  //                   [1+40, 2+50, 3+60],       [41, 52, 63],
  //                   [1+70, 2+80, 3+90],       [71, 82, 93],
  //                   [1+100, 2+110, 3+120]]    [101, 112, 123]]
  // result[1,:,:] = [[4+10, 5+20, 6+30],    = [[14, 25, 36],
  //                   [4+40, 5+50, 6+60],       [44, 55, 66],
  //                   [4+70, 5+80, 6+90],       [74, 85, 96],
  //                   [4+100, 5+110, 6+120]]    [104, 115, 126]]
}
```

### Example 5: Multi-Step Broadcasting (Simulating NumPy)

```
HloModule numpy_style_broadcast

ENTRY main {
  // Simulate: a[5,4,1] + b[4,3] = result[5,4,3]
  a = f32[5, 4, 1] parameter(0)
  b = f32[4, 3] parameter(1)
  
  // Step 1: Broadcast b from rank 2 to rank 3
  // broadcast_dimensions={1, 2} maps b dim 0 -> output dim 1, b dim 1 -> output dim 2
  b_broadcast = f32[5, 4, 3] broadcast(b), dimensions={1, 2}
  
  // Step 2: InDim broadcast a from [5,4,1] to [5,4,3] (degenerate dim 2)
  ROOT result = f32[5, 4, 3] add(a, b_broadcast)
}
```

### Example 6: Complex Broadcasting Pattern

```
HloModule attention_bias_broadcast

ENTRY main {
  // Attention: query [batch, heads, seq_len, head_dim]
  query = f32[4, 8, 128, 64] parameter(0)
  
  // Bias: [heads, head_dim] (position-independent per-head bias)
  bias = f32[8, 64] parameter(1)
  
  // Broadcast bias to [batch, heads, seq_len, head_dim]
  // broadcast_dimensions={1, 3} maps bias dim 0 -> output dim 1, bias dim 1 -> output dim 3
  bias_broadcast = f32[4, 8, 128, 64] broadcast(bias), dimensions={1, 3}
  
  ROOT result = f32[4, 8, 128, 64] add(query, bias_broadcast)
}
```

### Example 7: Invalid Broadcasting (Error Cases)

```
// INVALID: broadcast_dimensions not strictly increasing
// b = f32[3, 4] parameter(1)
// broadcast.0 = f32[2, 3, 4] broadcast(b), dimensions={2, 1}  // ERROR: 2 > 1 is not < 1

// INVALID: dimension size mismatch
// b = f32[3, 5] parameter(1)
// broadcast.0 = f32[2, 3, 4] broadcast(b), dimensions={1, 2}  // ERROR: input dim 1 (size 5) != output dim 2 (size 4)

// VALID: degenerate dimension
// b = f32[3, 1] parameter(1)
// broadcast.0 = f32[2, 3, 4] broadcast(b), dimensions={1, 2}  // OK: input dim 1 is degenerate (size 1), maps to output dim 2 (size 4)
```

---

## 3.10 XlaBuilder Broadcasting API

The `XlaBuilder` C++ class provides methods for constructing broadcast operations programmatically. Framework frontends use these methods to generate HLO broadcast instructions.

### 3.10.1 Broadcast Method

```cpp
// Broadcast an array to a given output shape
XlaOp Broadcast(XlaOp operand,
                absl::Span<const int64_t> broadcast_dimensions);
```

Parameters:
- `operand`: The input array to broadcast.
- `broadcast_dimensions`: The dimension mapping (as described in Section 3.4).

Returns: An `XlaOp` representing the broadcast result.

### 3.10.2 BroadcastInDim Method

```cpp
// Broadcast an array to a given output shape with explicit dimension mapping
XlaOp BroadcastInDim(XlaOp operand,
                     const Shape& shape,
                     absl::Span<const int64_t> broadcast_dimensions);
```

Parameters:
- `operand`: The input array to broadcast.
- `shape`: The desired output shape.
- `broadcast_dimensions`: The dimension mapping.

Returns: An `XlaOp` representing the broadcast result.

### 3.10.3 Collapse and Broadcast Pattern

A common pattern is to collapse leading dimensions, broadcast, and then reshape:

```cpp
// Example: Broadcast f32[A, B, C, D] to f32[A, B, X, C, D]
// where X is a new dimension inserted between dim 1 and dim 2

XlaOp input = ...;  // f32[A, B, C, D]

// Step 1: Collapse dimensions 2 and 3 into a single dimension
XlaOp collapsed = Collapse(input, {2, 3});  // f32[A, B, C*D]

// Step 2: Broadcast to f32[A, B, X, C*D]
XlaOp broadcast = BroadcastInDim(collapsed,
                                  ShapeUtil::MakeShape(F32, {A, B, X, C*D}),
                                  {0, 1, 3});  // map input dims to output dims

// Step 3: Reshape back to f32[A, B, X, C, D]
XlaOp result = Reshape(broadcast, {A, B, X, C, D});
```

### 3.10.4 Implicit Broadcasting in Binary Operations

Binary operations in `XlaBuilder` can perform implicit broadcasting through the `broadcast_dimensions` parameter:

```cpp
// Add with implicit broadcasting
XlaOp Add(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});

// Multiply with implicit broadcasting
XlaOp Mul(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64_t> broadcast_dimensions = {});
```

When `broadcast_dimensions` is empty and both operands have the same shape, no broadcasting occurs. When `broadcast_dimensions` is empty and one operand is a scalar, scalar broadcasting occurs. Otherwise, `broadcast_dimensions` must specify the dimension mapping for the lower-rank operand.

### 3.10.5 Python API (JAX)

In JAX, broadcasting is handled implicitly by `jax.numpy` functions, following NumPy conventions:

```python
import jax.numpy as jnp

# Scalar broadcasting
a = jnp.array([1, 2, 3])
b = 5
result = a + b  # [6, 7, 8]

# Vector-matrix broadcasting
matrix = jnp.ones((3, 4))
vector = jnp.array([1, 2, 3, 4])
result = matrix + vector  # vector broadcast along dim 0

# Explicit broadcast
x = jnp.array([1, 2, 3])
y = jnp.broadcast_to(x, (4, 3))  # explicit broadcast to (4, 3)

# broadcast_to with reshape
z = jnp.broadcast_to(x.reshape(3, 1), (3, 4))
```

### 3.10.6 Python API (TensorFlow with XLA)

```python
import tensorflow as tf

@tf.function(jit_compile=True)
def broadcast_example():
    # TensorFlow handles broadcasting automatically
    a = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape [2, 3]
    b = tf.constant([10, 20, 30])             # shape [3]
    result = a + b  # b is broadcast to [2, 3]
    return result

# tf.broadcast_to for explicit broadcasting
c = tf.broadcast_to(tf.constant([1, 2, 3]), [4, 3])
```

---

## 3.11 Broadcasting and Performance

### 3.11.1 Memory Efficiency

Broadcasting is memory-efficient because it does not actually copy data. The broadcast "expansion" is virtual -- at code generation time, the compiler emits code that reads the original data element and uses it for all output positions that map to that element.

For example, broadcasting `f32[512]` to `f32[128, 512]` does not allocate 128x512 = 65,536 floats. It allocates only 512 floats and reuses each value 128 times.

### 3.11.2 Fusion Opportunities

Broadcasts are excellent candidates for fusion:

- **Broadcast + element-wise op**: A broadcast followed by an element-wise operation can be fused into a single kernel that reads the broadcast operand once per iteration and applies the element-wise operation.
- **Broadcast + reduction**: A broadcast operand in a reduction can be optimized by reading the broadcast value once and using it for all reduction iterations.
- **Broadcast + dot**: A broadcast bias addition after a dot operation is commonly fused with the dot operation's epilogue.

### 3.11.3 Cache Effects

Broadcasting can have positive cache effects because:
- The broadcast source data is small and fits in cache.
- Each element is read many times, so it stays in cache after the first read.

However, broadcasting can also cause negative cache effects if:
- The broadcast source data is large enough to cause cache evictions.
- The access pattern causes cache thrashing (e.g., broadcasting along a non-contiguous dimension).

The layout assignment pass considers these effects when choosing layouts for broadcast operations.

---

## 3.12 Summary

XLA's broadcasting semantics provide a powerful mechanism for performing operations on arrays of different shapes. Broadcasting comes in two forms: explicit (using the `broadcast` instruction with `broadcast_dimensions`) and implicit (scalar broadcasting and InDim degenerate broadcasting in binary operations). The `broadcast_dimensions` attribute provides fine-grained control over how lower-rank arrays are mapped to higher-rank shapes, following strict rules about dimension ordering and size compatibility. Broadcasting is memory-efficient (no data copying) and offers excellent fusion opportunities. Framework frontends translate their own broadcasting conventions (NumPy-style for JAX, TensorFlow-style for TF) into XLA's explicit broadcast instructions and binary operation attributes.
