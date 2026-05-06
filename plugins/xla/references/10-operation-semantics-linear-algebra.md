# XLA Operation Semantics: Linear Algebra Operations

This reference provides comprehensive documentation of all XLA linear algebra operations, including dot products, generalized matrix multiplication, scaled and ragged dot products, Cholesky decomposition, and batch normalization operations. These operations form the mathematical foundation for neural network training and inference.

---

## Table of Contents

1. [Dot](#dot)
2. [DotGeneral](#dotgeneral)
3. [ScaledDot](#scaleddot)
4. [RaggedDot](#raggeddot)
5. [Cholesky](#cholesky)
6. [BatchNormTraining](#batchnormtraining)
7. [BatchNormInference](#batchnorminference)
8. [BatchNormGrad](#batchnormgrad)
9. [StableHLO Cross-References](#stablehlo-cross-references)

---

## Dot

`Dot` computes the dot product of two tensors. The behavior depends on the ranks of the operands.

### Signature

```
Dot(lhs, rhs, precision_config)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | The left-hand side operand. |
| `rhs` | `XlaOp` | The right-hand side operand. |
| `precision_config` | `std::vector<PrecisionConfig::Precision>` | Optional precision specification for each operand. |

### Semantics by Operand Rank

The behavior of `Dot` depends on the ranks of `lhs` and `rhs`:

#### Case 1: Vector-Vector (Both Rank 1)

Computes the standard inner product (dot product):

```
result = sum(lhs[i] * rhs[i]) for i in [0, n)
```

- Input: `lhs` shape `[N]`, `rhs` shape `[N]`
- Output: Scalar shape `[]`

**Example**:
```
lhs = [1, 2, 3]  // f32[3]
rhs = [4, 5, 6]  // f32[3]
result = 1*4 + 2*5 + 3*6 = 32  // f32[]
```

```
%result = f32[] dot(f32[3] %lhs, f32[3] %rhs)
```

#### Case 2: Matrix-Vector (Rank 2 and Rank 1)

Computes a matrix-vector product:

```
result[i] = sum(lhs[i, j] * rhs[j]) for j in [0, K)
```

- Input: `lhs` shape `[M, K]`, `rhs` shape `[K]`
- Output: Shape `[M]`

**Example**:
```
lhs = [[1, 2], [3, 4], [5, 6]]  // f32[3, 2]
rhs = [10, 20]                    // f32[2]
result = [1*10+2*20, 3*10+4*20, 5*10+6*20] = [50, 110, 170]  // f32[3]
```

```
%result = f32[3] dot(f32[3, 2] %lhs, f32[2] %rhs)
```

#### Case 3: Vector-Matrix (Rank 1 and Rank 2)

Computes a vector-matrix product:

```
result[j] = sum(lhs[i] * rhs[i, j]) for i in [0, K)
```

- Input: `lhs` shape `[K]`, `rhs` shape `[K, N]`
- Output: Shape `[N]`

**Example**:
```
lhs = [1, 2, 3]               // f32[3]
rhs = [[1, 2], [3, 4], [5, 6]] // f32[3, 2]
result = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]  // f32[2]
```

```
%result = f32[2] dot(f32[3] %lhs, f32[3, 2] %rhs)
```

#### Case 4: Matrix-Matrix (Both Rank 2)

Computes standard matrix multiplication:

```
result[i, j] = sum(lhs[i, k] * rhs[k, j]) for k in [0, K)
```

- Input: `lhs` shape `[M, K]`, `rhs` shape `[K, N]`
- Output: Shape `[M, N]`

**Example**:
```
lhs = [[1, 2], [3, 4]]  // f32[2, 2]
rhs = [[5, 6], [7, 8]]  // f32[2, 2]
result = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
       = [[19, 22], [43, 50]]  // f32[2, 2]
```

```
%result = f32[2, 2] dot(f32[2, 2] %lhs, f32[2, 2] %rhs)
```

### HLO Text Format

```
%result = f32[] dot(f32[3]{0} %lhs, f32[3]{0} %rhs),
  precision_config={HIGH, HIGH}

%result = f32[2,2]{1,0} dot(f32[2,2]{1,0} %lhs, f32[2,2]{1,0} %rhs)
```

### Constraints

- `Dot` only supports operands of rank 1 or 2.
- The inner dimensions must match: `lhs.shape[-1] == rhs.shape[0]` (for rank 2) or `lhs.shape[0] == rhs.shape[0]` (for rank 1).
- For higher-rank tensors, use `DotGeneral`.

---

## DotGeneral

`DotGeneral` is the most general form of dot product in XLA. It supports batched and contracted dimensions with explicit dimension numbering, enabling arbitrary tensor contractions (einsum-like operations).

### Signature

```
DotGeneral(lhs, rhs, dimension_numbers, precision_config,
           preferred_element_type)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | The left-hand side operand. Can have any rank >= 1. |
| `rhs` | `XlaOp` | The right-hand side operand. Can have any rank >= 1. |
| `dimension_numbers` | `DotDimensionNumbers` | Configuration specifying contracting and batch dimensions. |
| `precision_config` | `std::vector<PrecisionConfig::Precision>` | Precision for each operand. |
| `preferred_element_type` | `std::optional<PrimitiveType>` | The preferred element type for accumulation and output. |

### DotDimensionNumbers

```cpp
struct DotDimensionNumbers {
  std::vector<int64> lhs_contracting_dimensions;
  std::vector<int64> rhs_contracting_dimensions;
  std::vector<int64> lhs_batch_dimensions;
  std::vector<int64> rhs_batch_dimensions;
};
```

| Field | Description |
|---|---|
| `lhs_contracting_dimensions` | Dimensions of `lhs` that are summed over (reduced). These must have matching sizes in `rhs`. |
| `rhs_contracting_dimensions` | Dimensions of `rhs` that are summed over. Must have the same size as corresponding `lhs` contracting dimensions. |
| `lhs_batch_dimensions` | Dimensions of `lhs` that are treated as batch (not contracted, not free). These pair with corresponding `rhs` batch dimensions. |
| `rhs_batch_dimensions` | Dimensions of `rhs` that are batch. Must have the same size as corresponding `lhs` batch dimensions. |

### Semantics

The `DotGeneral` operation computes:

```
result[b0, b1, ..., lhs_free0, lhs_free1, ..., rhs_free0, rhs_free1, ...]
    = sum over contracting dimensions:
        lhs[b0, b1, ..., lhs_free0, lhs_free1, ..., c0, c1, ...]
        * rhs[b0, b1, ..., rhs_free0, rhs_free1, ..., c0, c1, ...]
```

Where:
- `b*` are batch dimensions (shared between lhs and rhs)
- `lhs_free*` are non-contracting, non-batch dimensions of lhs
- `rhs_free*` are non-contracting, non-batch dimensions of rhs
- `c*` are contracting dimensions (summed over)

### Output Shape

```
output_shape = [lhs_batch_sizes...] ++ [lhs_free_sizes...] ++ [rhs_free_sizes...]
```

The batch dimensions come first, followed by the non-contracting non-batch dimensions of `lhs`, followed by the non-contracting non-batch dimensions of `rhs`.

### Example 1: Standard Matrix Multiplication

Equivalent to `Dot` for rank-2 operands:

```
lhs: f32[M, K]
rhs: f32[K, N]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {1},
  rhs_contracting_dimensions = {0},
  lhs_batch_dimensions = {},
  rhs_batch_dimensions = {}
)

result: f32[M, N]
result[i, j] = sum_k lhs[i, k] * rhs[k, j]
```

```
%result = f32[M, N] dot-general(f32[M, K] %lhs, f32[K, N] %rhs),
  lhs_contracting_dims={1}, rhs_contracting_dims={0},
  lhs_batch_dims={}, rhs_batch_dims={}
```

### Example 2: Batched Matrix Multiplication

```
lhs: f32[B, M, K]
rhs: f32[B, K, N]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {2},
  rhs_contracting_dimensions = {1},
  lhs_batch_dimensions = {0},
  rhs_batch_dimensions = {0}
)

result: f32[B, M, N]
result[b, i, j] = sum_k lhs[b, i, k] * rhs[b, k, j]
```

```
%result = f32[B, M, N] dot-general(f32[B, M, K] %lhs, f32[B, K, N] %rhs),
  lhs_contracting_dims={2}, rhs_contracting_dims={1},
  lhs_batch_dims={0}, rhs_batch_dims={0}
```

### Example 3: Einsum "ij,jk->ik" (Standard MatMul)

```
lhs: f32[I, J]
rhs: f32[J, K]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {1},  // J dimension
  rhs_contracting_dimensions = {0},  // J dimension
  lhs_batch_dimensions = {},
  rhs_batch_dimensions = {}
)

result: f32[I, K]
```

### Example 4: Einsum "bij,bjk->bik" (Batched MatMul)

```
lhs: f32[B, I, J]
rhs: f32[B, J, K]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {2},
  rhs_contracting_dimensions = {1},
  lhs_batch_dimensions = {0},
  rhs_batch_dimensions = {0}
)

result: f32[B, I, K]
```

### Example 5: Einsum "ijk,ilm->jklm" (Tensor Contraction)

```
lhs: f32[I, J, K]
rhs: f32[I, L, M]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {0},  // I dimension contracted
  rhs_contracting_dimensions = {0},  // I dimension contracted
  lhs_batch_dimensions = {},
  rhs_batch_dimensions = {}
)

result: f32[J, K, L, M]
result[j, k, l, m] = sum_i lhs[i, j, k] * rhs[i, l, m]
```

### Example 6: Einsum "bnqd,bnkd->bnqk" (Attention Q*K^T)

```
lhs (Q): f32[B, N, Q, D]
rhs (K): f32[B, N, K, D]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {3},  // D dimension
  rhs_contracting_dimensions = {3},  // D dimension
  lhs_batch_dimensions = {0, 1},     // B, N are batch
  rhs_batch_dimensions = {0, 1}      // B, N are batch
)

result: f32[B, N, Q, K]
result[b, n, q, k] = sum_d Q[b, n, q, d] * K[b, n, k, d]
```

### Example 7: Multiple Contracting Dimensions

```
lhs: f32[A, B, C, D]
rhs: f32[B, C, D, E]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {1, 2, 3},  // B, C, D
  rhs_contracting_dimensions = {0, 1, 2},  // B, C, D
  lhs_batch_dimensions = {},
  rhs_batch_dimensions = {}
)

result: f32[A, E]
result[a, e] = sum_{b,c,d} lhs[a, b, c, d] * rhs[b, c, d, e]
```

### HLO Text Format

```
%result = f32[B,M,N]{2,1,0} dot-general(
  f32[B,M,K]{2,1,0} %lhs, f32[B,K,N]{2,1,0} %rhs
), lhs_contracting_dims={2}, rhs_contracting_dims={1},
  lhs_batch_dims={0}, rhs_batch_dims={0},
  precision_config={HIGH, HIGH}

%result = f32[B,N,Q,K]{3,2,1,0} dot-general(
  f32[B,N,Q,D]{3,2,1,0} %query, f32[B,N,K,D]{3,2,1,0} %key
), lhs_contracting_dims={3}, rhs_contracting_dims={3},
  lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
```

### Precision Config

On GPU backends, `precision_config` controls the use of tensor cores:

| Config | GPU Behavior |
|---|---|
| `DEFAULT` | Backend decides (usually allows tensor cores) |
| `HIGH` | Allows mixed-precision tensor cores (FP16/BF16 inputs, FP32 accumulate) |
| `HIGHEST` | Forces full-precision computation (no tensor cores) |

### Preferred Element Type

The `preferred_element_type` parameter allows specifying the accumulation type:

```
// FP16 matmul with FP32 accumulation
%result = f16[M, N] dot-general(f16[M, K] %lhs, f16[K, N] %rhs),
  lhs_contracting_dims={1}, rhs_contracting_dims={0},
  preferred_element_type=f32
```

### Constraints

- Each contracting dimension of `lhs` must have the same size as the corresponding contracting dimension of `rhs`.
- Each batch dimension of `lhs` must have the same size as the corresponding batch dimension of `rhs`.
- Contracting and batch dimensions must not overlap within a single operand.
- The number of lhs contracting dims must equal the number of rhs contracting dims.
- The number of lhs batch dims must equal the number of rhs batch dims.

---

## ScaledDot

`ScaledDot` computes a scaled dot product, primarily used for attention mechanisms. It scales one or both operands before computing the dot product.

### Signature

```
ScaledDot(lhs, rhs, lhs_scaling, rhs_scaling, dimension_numbers,
          precision_config, preferred_element_type)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | Left-hand side operand (e.g., query in attention). |
| `rhs` | `XlaOp` | Right-hand side operand (e.g., key in attention). |
| `lhs_scaling` | `std::optional<XlaOp>` | Optional scalar scaling factor for `lhs`. The `lhs` is multiplied by this before the dot product. |
| `rhs_scaling` | `std::optional<XlaOp>` | Optional scalar scaling factor for `rhs`. The `rhs` is multiplied by this before the dot product. |
| `dimension_numbers` | `DotDimensionNumbers` | Same as `DotGeneral`. |
| `precision_config` | `std::vector<PrecisionConfig::Precision>` | Precision for each operand. |
| `preferred_element_type` | `std::optional<PrimitiveType>` | Preferred accumulation type. |

### Semantics

Computes:
```
result = DotGeneral(lhs * lhs_scaling, rhs * rhs_scaling, dimension_numbers)
```

Or equivalently, when only one scaling factor is used (e.g., in scaled dot-product attention):
```
result = (1 / sqrt(d_k)) * DotGeneral(Q, K, dimension_numbers)
```

where `d_k` is the dimension of the key vectors and `lhs_scaling = 1/sqrt(d_k)`.

### Example: Scaled Dot-Product Attention (Q * K^T / sqrt(d))

```
// Q: f32[B, N, Q, D]
// K: f32[B, N, K, D]
// scale = 1/sqrt(D)

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {3},
  rhs_contracting_dimensions = {3},
  lhs_batch_dimensions = {0, 1},
  rhs_batch_dimensions = {0, 1}
)

%result = f32[B, N, Q, K] scaled-dot(
  f32[B, N, Q, D] %query,
  f32[B, N, K, D] %key,
  lhs_scaling=%scale,
  rhs_scaling=none
), dimension_numbers=%dim_nums
```

### Use Cases

1. **Self-Attention**: Q = K = V (same input projected differently).
2. **Cross-Attention**: Q from decoder, K and V from encoder.
3. **Flash Attention**: Backends may fuse ScaledDot with softmax for memory efficiency.

---

## RaggedDot

`RaggedDot` extends `DotGeneral` to support ragged (variable-length) batch dimensions. This is useful for attention mechanisms with variable sequence lengths.

### Signature

```
RaggedDot(lhs, rhs, dimension_numbers, precision_config,
          preferred_element_type, lhs_group_sizes, rhs_group_sizes)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | Left-hand side operand. |
| `rhs` | `XlaOp` | Right-hand side operand. |
| `dimension_numbers` | `DotDimensionNumbers` | Same as `DotGeneral`. |
| `precision_config` | `std::vector<PrecisionConfig::Precision>` | Precision configuration. |
| `preferred_element_type` | `std::optional<PrimitiveType>` | Preferred output type. |
| `lhs_group_sizes` | `XlaOp` | 1D integer tensor specifying the size of each ragged group in `lhs`. Used when batch dimensions have variable sizes. |
| `rhs_group_sizes` | `XlaOp` | 1D integer tensor specifying the size of each ragged group in `rhs`. |

### Semantics

`RaggedDot` performs the same tensor contraction as `DotGeneral` but with ragged batch dimensions. The `lhs_group_sizes` and `rhs_group_sizes` tensors define how the batch dimensions are partitioned into groups of variable size.

Within each ragged group, the dot product is computed normally. Between groups (where one group ends and another begins in the batch dimension), the computation is independent.

This is particularly useful for:
1. **Padded batched attention**: Where different sequences in a batch have different lengths, and padding should not contribute to the dot product.
2. **Block-sparse attention**: Where the attention pattern has variable block sizes.

### Example

```
// Ragged attention with variable sequence lengths
// lhs (Q): f32[total_tokens, D]  (tokens from all sequences concatenated)
// rhs (K): f32[total_tokens, D]
// lhs_group_sizes: s32[num_sequences]  (e.g., [3, 5, 2] for 3 sequences)
// rhs_group_sizes: s32[num_sequences]

dim_nums = DotDimensionNumbers(
  lhs_contracting_dimensions = {1},
  rhs_contracting_dimensions = {1},
  lhs_batch_dimensions = {},
  rhs_batch_dimensions = {}
)

%result = ragged-dot(%Q, %K, %dim_nums, %lhs_group_sizes, %rhs_group_sizes)
```

---

## Cholesky

`Cholesky` computes the Cholesky decomposition of a batch of symmetric positive-definite matrices. Given a matrix `A`, it computes a lower (or upper) triangular matrix `L` such that `A = L * L^T` (or `A = U^T * U`).

### Signature

```
Cholesky(a, lower)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `a` | `XlaOp` | A tensor of shape `[..., N, N]`. The innermost two dimensions form square matrices. All leading dimensions are batch dimensions. Must be of floating point type (`F32`, `F64`, `BF16`, `F16`). |
| `lower` | `bool` | If `true`, compute the lower triangular Cholesky factor `L` where `A = L * L^T`. If `false`, compute the upper triangular factor `U` where `A = U^T * U`. Default `true`. |

### Semantics

For each batch element, computes the Cholesky decomposition:

**Lower triangular** (`lower = true`):
```
A = L * L^T
```
where `L` is lower triangular with positive diagonal entries.

**Upper triangular** (`lower = false`):
```
A = U^T * U
```
where `U` is upper triangular with positive diagonal entries.

The output has the same shape as the input `[..., N, N]`. The triangle not computed (upper when `lower=true`, lower when `lower=false`) is filled with zeros.

### Mathematical Definition

For a lower triangular decomposition, element `(i, j)` of `L` is:

```
L[i, j] = (1/L[j,j]) * (A[i,j] - sum_{k=0}^{j-1} L[i,k] * L[j,k])   for i > j
L[j, j] = sqrt(A[j,j] - sum_{k=0}^{j-1} L[j,k]^2)                    for i == j
L[i, j] = 0                                                             for i < j
```

### Example: 3x3 Matrix

```
A = [[4, 2, 0],
     [2, 5, 2],
     [0, 2, 6]]

// Lower Cholesky decomposition
L = Cholesky(A, lower=true)

// L[0,0] = sqrt(4) = 2
// L[1,0] = 2/2 = 1
// L[1,1] = sqrt(5 - 1*1) = sqrt(4) = 2
// L[2,0] = 0/2 = 0
// L[2,1] = (2 - 0*1) / 2 = 1
// L[2,2] = sqrt(6 - 0*0 - 1*1) = sqrt(5) ~ 2.236

L = [[2,     0,     0    ],
     [1,     2,     0    ],
     [0,     1,     2.236]]
```

```
%result = f32[3, 3] cholesky(f32[3, 3] %a), lower=true
```

### Example: Batched Cholesky

```
// a: f32[4, 3, 3]  (batch of 4 matrices)
%result = f32[4, 3, 3] cholesky(f32[4, 3, 3] %a), lower=true
// Each of the 4 matrices is independently decomposed
```

### HLO Text Format

```
%result = f32[3,3]{1,0} cholesky(f32[3,3]{1,0} %a), lower=true

%result = f32[4,3,3]{2,1,0} cholesky(f32[4,3,3]{2,1,0} %a), lower=false
```

### Failure Handling

If the input matrix is not positive definite, the Cholesky decomposition fails. XLA handles this by:
1. Setting the diagonal elements to `NaN` for failed matrices.
2. The remaining elements of failed matrices may contain undefined values.

To check for failure, inspect the diagonal: if any diagonal element is `NaN`, the decomposition failed.

### Use Cases

1. **Solving linear systems**: After `L = Cholesky(A)`, use `TriangularSolve(L, b, lower=true)` twice to solve `A * x = b`.
2. **Computing log-determinant**: `log(det(A)) = 2 * sum(log(diag(L)))`.
3. **Sampling from multivariate normal**: `x = L * z` where `z ~ N(0, I)`.
4. **Matrix inversion**: `A^(-1) = L^(-T) * L^(-1)`, computed via triangular solves.

---

## BatchNormTraining

`BatchNormTraining` computes the batch normalization of the input during training. It returns the normalized output, the batch mean, and the batch variance as a tuple.

### Signature

```
BatchNormTraining(operand, scale, offset, epsilon, feature_index)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor to be normalized. Shape: `[D0, D1, ..., Dn]`. |
| `scale` | `XlaOp` | The gamma (scaling) parameter. 1D tensor of size `operand.shape[feature_index]`. |
| `offset` | `XlaOp` | The beta (shifting) parameter. 1D tensor of size `operand.shape[feature_index]`. |
| `epsilon` | `float` | A small value added to the variance for numerical stability (to avoid division by zero). Typical value: `1e-5`. |
| `feature_index` | `int64` | The dimension index of the feature (channel) axis in `operand`. For NHWC format, this is typically the last dimension (e.g., 3 for a 4D tensor). |

### Semantics

For each feature channel `c`, computes the mean and variance across all non-feature dimensions:

```
mu_c = (1 / M) * sum_{non-feature indices} operand[..., c, ...]

sigma_c^2 = (1 / M) * sum_{non-feature indices} (operand[..., c, ...] - mu_c)^2
```

where `M` is the product of all non-feature dimension sizes (total number of elements per feature).

Then normalizes:

```
normalized[..., c, ...] = (operand[..., c, ...] - mu_c) / sqrt(sigma_c^2 + epsilon)

output[..., c, ...] = scale[c] * normalized[..., c, ...] + offset[c]
```

### Mathematical Formulas

Let `l` index the feature dimension and let `m` range over all non-feature indices. Let `M` be the count of elements per feature.

**Mean**:
```
mu_l = (1/M) * sum_m operand[m_0, ..., m_{l-1}, l, m_l, ..., m_{n-1}]
```

**Variance**:
```
sigma_l^2 = (1/M) * sum_m (operand[m_0, ..., l, ...] - mu_l)^2
```

**Normalization**:
```
hat{x}_{m,l} = (x_{m,l} - mu_l) / sqrt(sigma_l^2 + epsilon)
```

**Output**:
```
y_{m,l} = gamma_l * hat{x}_{m,l} + beta_l
```

**Return value**: A tuple `(output, mu, sigma^2)`:
- `output`: Same shape as `operand`. The normalized, scaled, and shifted tensor.
- `mu`: 1D tensor of shape `[num_features]`. The per-feature mean.
- `sigma^2`: 1D tensor of shape `[num_features]`. The per-feature variance.

### Example

```
// operand: f32[2, 3, 4, 5] (NHWC: batch=2, H=3, W=4, C=5)
// scale: f32[5] (gamma)
// offset: f32[5] (beta)
// epsilon: 1e-5
// feature_index: 3 (channel dim in NHWC)

%result = (f32[2,3,4,5], f32[5], f32[5]) batch-norm-training(
  f32[2,3,4,5] %operand,
  f32[5] %scale,
  f32[5] %offset
), epsilon=1e-5, feature_index=3

%output = get-tuple-element(%result, 0)   // f32[2,3,4,5] normalized output
%mean = get-tuple-element(%result, 1)    // f32[5] batch mean
%variance = get-tuple-element(%result, 2) // f32[5] batch variance
```

For channel `c`:
```
M = 2 * 3 * 4 = 24  (batch * H * W)
mu[c] = mean of all 24 values in channel c
var[c] = variance of all 24 values in channel c
output[..., c] = gamma[c] * (operand[..., c] - mu[c]) / sqrt(var[c] + 1e-5) + beta[c]
```

### HLO Text Format

```
%result = (f32[2,3,4,5]{3,2,1,0}, f32[5]{0}, f32[5]{0})
  batch-norm-training(f32[2,3,4,5]{3,2,1,0} %operand,
                      f32[5]{0} %scale,
                      f32[5]{0} %offset),
  epsilon=0.00001, feature_index=3
```

---

## BatchNormInference

`BatchNormInference` applies batch normalization during inference using pre-computed mean and variance (typically running averages from training).

### Signature

```
BatchNormInference(operand, scale, offset, mean, variance,
                   epsilon, feature_index)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. Shape: `[D0, D1, ..., Dn]`. |
| `scale` | `XlaOp` | Gamma (scaling). 1D tensor of size `num_features`. |
| `offset` | `XlaOp` | Beta (shifting). 1D tensor of size `num_features`. |
| `mean` | `XlaOp` | Pre-computed population mean. 1D tensor of size `num_features`. |
| `variance` | `XlaOp` | Pre-computed population variance. 1D tensor of size `num_features`. |
| `epsilon` | `float` | Small value for numerical stability. |
| `feature_index` | `int64` | Index of the feature dimension. |

### Semantics

Unlike `BatchNormTraining`, this does not compute mean and variance from the batch. Instead, it uses the provided `mean` and `variance` (typically exponential moving averages computed during training):

```
output[..., c, ...] = scale[c] * (operand[..., c, ...] - mean[c]) / sqrt(variance[c] + epsilon) + offset[c]
```

**Output**: A single tensor with the same shape as `operand` (not a tuple).

### Example

```
// operand: f32[1, 224, 224, 3] (single image, 3 channels)
// mean: f32[3] (running mean from training)
// variance: f32[3] (running variance from training)

%result = f32[1, 224, 224, 3] batch-norm-inference(
  f32[1, 224, 224, 3] %operand,
  f32[3] %scale,
  f32[3] %offset,
  f32[3] %mean,
  f32[3] %variance
), epsilon=1e-5, feature_index=3
```

### HLO Text Format

```
%result = f32[1,224,224,3]{3,2,1,0} batch-norm-inference(
  f32[1,224,224,3]{3,2,1,0} %operand,
  f32[3]{0} %scale,
  f32[3]{0} %offset,
  f32[3]{0} %mean,
  f32[3]{0} %variance
), epsilon=0.00001, feature_index=3
```

### Fusion Optimization

`BatchNormInference` is often fused with the preceding convolution operation on GPU backends. This fusion eliminates the need to write the intermediate convolution result to memory and then read it back for normalization.

---

## BatchNormGrad

`BatchNormGrad` computes the gradients of batch normalization with respect to the operand, scale, and offset. It is used during the backward pass of training.

### Signature

```
BatchNormGrad(operand, scale, mean, variance, grad_output,
              epsilon, feature_index)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The original input tensor (from the forward pass). Shape: `[D0, ..., Dn]`. |
| `scale` | `XlaOp` | The gamma parameter. 1D tensor. |
| `mean` | `XlaOp` | The batch mean (from `BatchNormTraining`). 1D tensor. |
| `variance` | `XlaOp` | The batch variance (from `BatchNormTraining`). 1D tensor. |
| `grad_output` | `XlaOp` | The gradient of the loss with respect to the batch norm output. Same shape as `operand`. |
| `epsilon` | `float` | Same epsilon used in the forward pass. |
| `feature_index` | `int64` | The feature dimension index. |

### Semantics

Computes three gradients simultaneously:

**Return value**: A tuple `(grad_operand, grad_scale, grad_offset)`:

#### Gradient with Respect to Operand (grad_operand)

The gradient of the loss `L` with respect to the input `x`:

```
grad_operand[m, l] = (gamma_l / (M * sqrt(sigma_l^2 + epsilon)))
    * (M * grad_output[m, l]
       - sum_m(grad_output[m, l])
       - hat{x}_{m,l} * sum_m(grad_output[m, l] * hat{x}_{m,l}))
```

where:
- `M` is the number of elements per feature (product of non-feature dimensions)
- `hat{x}_{m,l} = (x_{m,l} - mu_l) / sqrt(sigma_l^2 + epsilon)` is the normalized input

This can be decomposed as:

```
grad_operand = (1 / (M * sigma)) * gamma * (M * grad_output - sum(grad_output) - x_hat * sum(grad_output * x_hat))
```

#### Gradient with Respect to Scale (grad_scale)

```
grad_scale[l] = sum_m(grad_output[m, l] * hat{x}_{m, l})
```

The sum of `grad_output * normalized_input` over all non-feature dimensions for each feature.

#### Gradient with Respect to Offset (grad_offset)

```
grad_offset[l] = sum_m(grad_output[m, l])
```

The sum of `grad_output` over all non-feature dimensions for each feature.

### Complete Gradient Formulas

Let:
- `x` = operand (input)
- `y` = batch norm output
- `L` = loss
- `dy` = grad_output
- `mu` = mean
- `sigma^2` = variance
- `gamma` = scale
- `beta` = offset
- `eps` = epsilon
- `M` = number of elements per feature
- `sigma = sqrt(sigma^2 + eps)`
- `x_hat = (x - mu) / sigma`

Then:

```
grad_beta = reduce_sum(dy, non_feature_dims)
grad_gamma = reduce_sum(dy * x_hat, non_feature_dims)

dx_hat = dy * gamma
dsigma = reduce_sum(dx_hat * (x - mu) * (-0.5) * (sigma^2 + eps)^(-3/2), non_feature_dims)
dmu = reduce_sum(dx_hat * (-1/sigma), non_feature_dims) + dsigma * reduce_sum(-2*(x-mu), non_feature_dims) / M

grad_x = dx_hat / sigma + dsigma * 2*(x-mu)/M + dmu / M
```

Which simplifies to:

```
grad_x = (1/(M*sigma)) * gamma * (M*dy - sum(dy) - x_hat * sum(dy * x_hat))
```

### Example

```
// operand: f32[32, 28, 28, 64] (batch=32, H=28, W=28, C=64)
// grad_output: f32[32, 28, 28, 64] (upstream gradient)
// mean, variance: f32[64] (from forward pass)
// scale: f32[64]

%result = (f32[32,28,28,64], f32[64], f32[64]) batch-norm-grad(
  f32[32,28,28,64] %operand,
  f32[64] %scale,
  f32[64] %mean,
  f32[64] %variance,
  f32[32,28,28,64] %grad_output
), epsilon=1e-5, feature_index=3

%grad_operand = get-tuple-element(%result, 0)  // f32[32,28,28,64]
%grad_scale = get-tuple-element(%result, 1)    // f32[64]
%grad_offset = get-tuple-element(%result, 2)   // f32[64]
```

For channel `c`:
```
M = 32 * 28 * 28 = 25088

x_hat_c = (operand[..., c] - mean[c]) / sqrt(variance[c] + 1e-5)

grad_offset[c] = sum(grad_output[..., c])
grad_scale[c] = sum(grad_output[..., c] * x_hat_c)
grad_operand[..., c] = (1 / (M * sqrt(var[c] + 1e-5))) * scale[c]
                       * (M * grad_output[..., c]
                          - sum(grad_output[..., c])
                          - x_hat_c * sum(grad_output[..., c] * x_hat_c))
```

### HLO Text Format

```
%result = (f32[32,28,28,64]{3,2,1,0}, f32[64]{0}, f32[64]{0})
  batch-norm-grad(f32[32,28,28,64]{3,2,1,0} %operand,
                  f32[64]{0} %scale,
                  f32[64]{0} %mean,
                  f32[64]{0} %variance,
                  f32[32,28,28,64]{3,2,1,0} %grad_output),
  epsilon=0.00001, feature_index=3
```

---

## StableHLO Cross-References

| XLA Operation | StableHLO Operation | Notes |
|---|---|---|
| Dot | `stablehlo.dot` | Limited to rank 1-2 operands |
| DotGeneral | `stablehlo.dot_general` | Full generality with dimension numbers |
| Cholesky | `stablehlo.cholesky` | Same semantics |
| BatchNormTraining | `stablehlo.batch_norm_training` | Same tuple output |
| BatchNormInference | `stablehlo.batch_norm_inference` | Same semantics |
| BatchNormGrad | `stablehlo.batch_norm_grad` | Same tuple output |

### StableHLO Example: DotGeneral

```mlir
%result = stablehlo.dot_general(%lhs, %rhs) {
  dot_dimension_numbers = #stablehlo.dot<
    lhs_contracting_dimensions = [2],
    rhs_contracting_dimensions = [1],
    lhs_batch_dimensions = [0],
    rhs_batch_dimensions = [0]
  >,
  precision_config = [HIGH, HIGH]
} : (tensor<4x8x16xf32>, tensor<4x16x32xf32>) -> tensor<4x8x32xf32>
```

### StableHLO Example: Cholesky

```mlir
%result = stablehlo.cholesky(%a) {
  lower = true
} : (tensor<3x3xf32>) -> tensor<3x3xf32>
```

### StableHLO Example: BatchNormTraining

```mlir
%output, %mean, %variance = stablehlo.batch_norm_training(
  %operand, %scale, %offset
) {
  epsilon = 1.0e-5 : f32,
  feature_index = 3 : i64
} : (tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<5xf32>)
  -> (tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<5xf32>)
```

### StableHLO Example: BatchNormInference

```mlir
%result = stablehlo.batch_norm_inference(
  %operand, %scale, %offset, %mean, %variance
) {
  epsilon = 1.0e-5 : f32,
  feature_index = 3 : i64
} : (tensor<1x224x224x3xf32>, tensor<3xf32>, tensor<3xf32>,
     tensor<3xf32>, tensor<3xf32>) -> tensor<1x224x224x3xf32>
```

### StableHLO Example: BatchNormGrad

```mlir
%grad_operand, %grad_scale, %grad_offset = stablehlo.batch_norm_grad(
  %operand, %scale, %mean, %variance, %grad_output
) {
  epsilon = 1.0e-5 : f32,
  feature_index = 3 : i64
} : (tensor<32x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>,
     tensor<64xf32>, tensor<32x28x28x64xf32>)
  -> (tensor<32x28x28x64xf32>, tensor<64xf32>, tensor<64xf32>)
```

---

## Appendix: Linear Algebra Operations Summary

### Dot Operation Variants

| Operation | Rank Support | Batch | Contract | Key Use |
|---|---|---|---|---|
| Dot | 1-2 | No | Implicit last/first | Simple matmul, vec-dot |
| DotGeneral | Any | Yes | Explicit dims | Einsum, attention, tensor contractions |
| ScaledDot | Any | Yes | Explicit dims | Attention (Q*K^T / sqrt(d)) |
| RaggedDot | Any | Ragged | Explicit dims | Variable-length attention |

### Batch Normalization Operations

| Operation | Computes Mean/Var | Returns | Phase |
|---|---|---|---|
| BatchNormTraining | Yes (from batch) | (output, mean, variance) | Training forward |
| BatchNormInference | No (uses provided) | output only | Inference |
| BatchNormGrad | No (uses provided) | (grad_input, grad_scale, grad_offset) | Training backward |

### Dimension Numbering Quick Reference for DotGeneral

```
Given:
  lhs: [B1, B2, ..., M1, M2, ..., K1, K2, ...]
  rhs: [B1, B2, ..., K1, K2, ..., N1, N2, ...]

  lhs_batch_dims = {indices of B dimensions in lhs}
  rhs_batch_dims = {indices of B dimensions in rhs}
  lhs_contracting_dims = {indices of K dimensions in lhs}
  rhs_contracting_dims = {indices of K dimensions in rhs}

Output: [B1, B2, ..., M1, M2, ..., N1, N2, ...]
```

### Common Einsum to DotGeneral Mappings

| Einsum Notation | lhs shape | rhs shape | lhs_contract | rhs_contract | lhs_batch | rhs_batch | Output shape |
|---|---|---|---|---|---|---|---|
| `ij,jk->ik` | `[I,J]` | `[J,K]` | `{1}` | `{0}` | `{}` | `{}` | `[I,K]` |
| `bij,bjk->bik` | `[B,I,J]` | `[B,J,K]` | `{2}` | `{1}` | `{0}` | `{0}` | `[B,I,K]` |
| `bnqd,bnkd->bnqk` | `[B,N,Q,D]` | `[B,N,K,D]` | `{3}` | `{3}` | `{0,1}` | `{0,1}` | `[B,N,Q,K]` |
| `ij,j->i` | `[I,J]` | `[J]` | `{1}` | `{0}` | `{}` | `{}` | `[I]` |
| `i,i->` | `[I]` | `[I]` | `{0}` | `{0}` | `{}` | `{}` | `[]` |
| `abcd,cdef->abef` | `[A,B,C,D]` | `[C,D,E,F]` | `{2,3}` | `{0,1}` | `{}` | `{}` | `[A,B,E,F]` |
| `bsnh,bsnh->bs` | `[B,S,N,H]` | `[B,S,N,H]` | `{2,3}` | `{2,3}` | `{0,1}` | `{0,1}` | `[B,S]` |
