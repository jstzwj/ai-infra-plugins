# Symbolic Expressions in XLA

This document provides comprehensive documentation about symbolic expressions in XLA, covering symbolic dimension expressions, compile-time shape inference, indexing analysis, and their use in emitters and sparse operations.

## Table of Contents

- [SymbolicExpression Overview](#symbolicexpression-overview)
- [Indexing Analysis](#indexing-analysis)
- [Use in Emitters](#use-in-emitters)
- [SparseCore Support](#sparsecore-support)

## SymbolicExpression Overview

### Symbolic Dimension Expressions

XLA uses symbolic expressions to represent and reason about tensor dimensions at compile time. This is essential for supporting dynamic shapes, where the exact dimension sizes are not known until runtime, but relationships between dimensions can be determined statically.

A `SymbolicExpression` represents a dimension as a symbolic formula involving constants, parameters, and arithmetic operations:

```cpp
// Examples of symbolic dimension expressions:
//   1. Constant: 128
//   2. Parameter: dim_0  (the value of dimension 0 of the input)
//   3. Expression: dim_0 * 4 + 16
//   4. Complex: (dim_0 + dim_1) / 2

class SymbolicExpression {
 public:
  // Expression kinds
  enum class Kind {
    kConstant,    // A known constant value
    kParameter,   // A symbolic parameter (e.g., dimension of an input)
    kAdd,         // Addition of two expressions
    kMul,         // Multiplication of two expressions
    kDiv,         // Integer division
    kMod,         // Modulo
    kMax,         // Maximum of two expressions
    kMin,         // Minimum of two expressions
  };

  // Factory methods
  static SymbolicExpression Constant(int64_t value);
  static SymbolicExpression Parameter(int64_t parameter_id);
  static SymbolicExpression Add(const SymbolicExpression& a,
                                 const SymbolicExpression& b);
  static SymbolicExpression Mul(const SymbolicExpression& a,
                                 const SymbolicExpression& b);

  // Evaluation with concrete parameter values
  StatusOr<int64_t> Evaluate(
      absl::Span<const int64_t> parameter_values) const;

  // Simplification
  SymbolicExpression Simplify() const;

  // Comparison
  bool Equals(const SymbolicExpression& other) const;
};

// Free operators for building expressions
SymbolicExpression operator+(const SymbolicExpression& a,
                              const SymbolicExpression& b);
SymbolicExpression operator*(const SymbolicExpression& a,
                              const SymbolicExpression& b);
```

### Expression Representation

Symbolic expressions are represented as trees, where each node is either:

1. **A constant**: A known integer value (e.g., 128, 4, 16).
2. **A parameter**: A symbolic value that will be resolved at runtime (e.g., `dim_0` represents the 0th dimension of an input tensor).
3. **An operation**: A binary operation (add, multiply, divide, etc.) applied to sub-expressions.

For example, the expression `(dim_0 * 4) + 16` is represented as:

```
        (+)
       /   \
     (*)   16
    /   \
 dim_0   4
```

### Compile-Time Shape Inference

Symbolic expressions enable compile-time shape inference for operations with dynamic shapes. Instead of requiring exact dimension values, XLA can infer the output shape symbolically:

```cpp
// Example: Reshape with dynamic dimensions
// Input: tensor<batch, seq, hidden>
// Reshape to: tensor<batch * seq, hidden>
//
// The output's first dimension is: dim_0 * dim_1
// This is known symbolically at compile time,
// even though batch and seq may be dynamic.

Shape InferReshapeOutput(const Shape& input_shape,
                          const absl::Span<const int64_t>& output_dimensions) {
  // Build symbolic expressions for output dimensions
  std::vector<SymbolicExpression> output_exprs;
  for (int64_t dim : output_dimensions) {
    if (dim == Shape::kDynamicDimension) {
      // This dimension is a product of input dimensions
      output_exprs.push_back(SymbolicExpression::Parameter(dim));
    } else {
      output_exprs.push_back(SymbolicExpression::Constant(dim));
    }
  }
  return ShapeUtil::MakeShapeWithSymbolicDimensions(
      input_shape.element_type(), output_exprs);
}
```

### Dynamic Shapes Support

XLA supports dynamic shapes where dimension sizes are not known at compile time. Symbolic expressions are crucial for this:

1. **Shape checking**: Even with dynamic dimensions, XLA can verify that the output shape is consistent with the operation semantics.

2. **Buffer allocation**: XLA can compute buffer sizes using symbolic expressions, allocating the maximum possible size when the exact size is not known.

3. **Layout assignment**: XLA can assign memory layouts even for dynamic shapes, using symbolic constraints.

```cpp
// Example: Dynamic slice with symbolic bounds
// Input: tensor<?x1024xf32>  (first dimension is dynamic)
// Dynamic slice with start indices and limit indices

// The output shape is determined symbolically:
// output_dim_0 = limit_indices[0] - start_indices[0]
// output_dim_1 = 1024 (static)

SymbolicExpression output_dim_0 =
    SymbolicExpression::Parameter(/*limit_0*/) -
    SymbolicExpression::Parameter(/*start_0*/);

Shape output_shape = ShapeUtil::MakeShapeWithSymbolicDimensions(
    F32, {output_dim_0, SymbolicExpression::Constant(1024)});
```

### Expression Simplification

Symbolic expressions can be simplified to reduce runtime evaluation overhead:

```
// Before simplification:
(dim_0 + 0) * 1 + (dim_1 * 0)

// After simplification:
dim_0
```

Common simplification rules:

1. **Identity removal**: `x + 0 = x`, `x * 1 = x`
2. **Zero absorption**: `x * 0 = 0`
3. **Constant folding**: `3 + 5 = 8`
4. **Association**: `(x + a) + b = x + (a + b)` when `a` and `b` are constants
5. **Distribution**: `a * (b + c) = a*b + a*c` (when beneficial)

## Indexing Analysis

### IndexingMap

The `IndexingMap` is a key abstraction in XLA that maps from output element indices to input element indices. It is used extensively in code generation to determine which input elements to load for each output element.

```cpp
class IndexingMap {
 public:
  // Create an identity indexing map (output index = input index)
  static IndexingMap Identity(int64_t rank);

  // Create from symbolic expressions for each input dimension
  static IndexingMap FromOutputToInput(
      absl::Span<const SymbolicExpression> index_exprs);

  // Compose two indexing maps: f(g(x))
  IndexingMap Compose(const IndexingMap& other) const;

  // Get the expression for a specific input dimension
  const SymbolicExpression& GetInputIndex(int64_t dim) const;

  // Domain constraints (which output indices are valid)
  void AddConstraint(const SymbolicExpression& constraint);

  // Evaluate for concrete output indices
  StatusOr<std::vector<int64_t>> Evaluate(
      absl::Span<const int64_t> output_indices) const;
};
```

### Index Computation for Operations

Each HLO operation has an associated indexing map that describes how output indices map to input indices:

#### Elementwise Operations

For elementwise operations like `add`, `multiply`, etc., the indexing map is an identity map (each output element at position `(i, j)` reads from input elements at the same position):

```
Output index: (i, j)
Input 0 index: (i, j)
Input 1 index: (i, j)
```

```cpp
IndexingMap elementwise_map = IndexingMap::Identity(/*rank=*/2);
```

#### Broadcast

For broadcast, the output index maps to the input index with the broadcast dimensions dropped:

```
# Broadcast from [1024] to [4, 1024]
Output index: (i, j)
Input index: (j)
```

```cpp
IndexingMap broadcast_map = IndexingMap::FromOutputToInput({
    SymbolicExpression::Parameter(1)  // Only use dim 1 of output
});
```

#### Transpose

For transpose, the output index maps to the input index with dimensions permuted:

```
# Transpose from [M, N] to [N, M]
Output index: (i, j)
Input index: (j, i)
```

```cpp
IndexingMap transpose_map = IndexingMap::FromOutputToInput({
    SymbolicExpression::Parameter(1),  // Input dim 0 = output dim 1
    SymbolicExpression::Parameter(0),  // Input dim 1 = output dim 0
});
```

#### Dot (Matmul)

For matrix multiplication `C[M, N] = A[M, K] * B[K, N]`, the output element `(i, j)` requires reading all elements `A[i, k]` and `B[k, j]` for `k = 0..K-1`:

```
Output index: (i, j)
Input A index: (i, k)  for all k
Input B index: (k, j)  for all k
```

This requires a reduction dimension, which is represented as an implicit loop over `k`:

```cpp
IndexingMap dot_map_a = IndexingMap::FromOutputToInput({
    SymbolicExpression::Parameter(0),  // A's first dim = output's first dim
    SymbolicExpression::Parameter(2),  // A's second dim = reduction dim
});

IndexingMap dot_map_b = IndexingMap::FromOutputToInput({
    SymbolicExpression::Parameter(2),  // B's first dim = reduction dim
    SymbolicExpression::Parameter(1),  // B's second dim = output's second dim
});
```

#### Convolution

For convolution, the indexing map encodes the window striding, padding, and dilation:

```
# Convolution with window size 3x3, stride 2x2, padding 1x1
Output index: (b, oy, ox, co)
Input index: (b, oy * sy + ky - pad_y, ox * sx + kx - pad_x, ci)
Kernel index: (ky, kx, ci, co)
```

Where:
- `sy`, `sx` are the stride values
- `pad_y`, `pad_x` are the padding values
- `ky`, `kx` range over the kernel spatial dimensions

```cpp
IndexingMap conv_input_map = IndexingMap::FromOutputToInput({
    SymbolicExpression::Parameter(0),  // batch
    SymbolicExpression::Parameter(1) * SymbolicExpression::Constant(stride_y) +
        SymbolicExpression::Parameter(4) -
        SymbolicExpression::Constant(pad_y),  // input y
    SymbolicExpression::Parameter(2) * SymbolicExpression::Constant(stride_x) +
        SymbolicExpression::Parameter(5) -
        SymbolicExpression::Constant(pad_x),  // input x
    SymbolicExpression::Parameter(6),  // input channel
});
```

### Domain Constraints

Domain constraints specify the valid range of output indices for which the indexing map is defined:

```cpp
// For a convolution, the output spatial dimensions are constrained
IndexingMap conv_map = ...;
conv_map.AddConstraint(SymbolicExpression::Parameter(1) >= 0);
conv_map.AddConstraint(SymbolicExpression::Parameter(1) < output_height);
conv_map.AddConstraint(SymbolicExpression::Parameter(2) >= 0);
conv_map.AddConstraint(SymbolicExpression::Parameter(2) < output_width);
```

Constraints are used for:

1. **Bounds checking**: Verifying that index computations do not go out of bounds.
2. **Iteration space**: Determining the range of output indices to iterate over.
3. **Simplification**: Constraining the domain can enable simplifications.

## Use in Emitters

### Indexing Transformations

XLA's code generation (emission) uses indexing maps to generate efficient kernel code. For each output element, the emitter uses the indexing map to compute the input indices and generate load/store instructions.

#### GPU Kernel Emission

For GPU kernels, the indexing map is used to compute thread-to-element mappings:

```cpp
// Pseudo-code for GPU kernel emission using indexing maps
void EmitElementwiseKernel(GpuEmitter* emitter,
                            const HloInstruction* op,
                            const IndexingMap& indexing_map) {
  // Get the linear thread index
  auto thread_id = emitter->EmitGlobalThreadId();

  // Convert linear thread index to multi-dimensional output index
  auto output_index = emitter->EmitUnflattenIndex(thread_id, output_shape);

  // Apply the indexing map to get input indices
  auto input_indices = indexing_map.Evaluate(output_index);

  // Emit loads from input buffers
  for (int i = 0; i < op->operand_count(); ++i) {
    emitter->EmitLoad(input_buffers[i], input_indices[i]);
  }

  // Emit the operation
  emitter->EmitOperation(op, input_values);

  // Emit store to output buffer
  emitter->EmitStore(output_buffer, result_value, output_index);
}
```

#### Fusion Kernel Emission

For fused operations, the indexing map is composed across the fused operations:

```cpp
// For a fusion of: multiply(add(x, y), z)
// The indexing map for the fusion output is the composition of
// the individual operation indexing maps.

// add: output[i,j] = x[i,j] + y[i,j]
// multiply: output[i,j] = add_out[i,j] * z[i,j]
// Composed: output[i,j] = (x[i,j] + y[i,j]) * z[i,j]

IndexingMap add_map = IndexingMap::Identity(/*rank=*/2);
IndexingMap mul_map = IndexingMap::Identity(/*rank=*/2);

// For elementwise fusion, all maps are identity, so composition is trivial.
// For more complex fusions (e.g., involving broadcast, transpose, reshape),
// the maps are composed:

// Example: multiply(broadcast(x), transpose(y))
// broadcast map: input[i,j] = x[j]  (x is 1D, broadcast along dim 0)
// transpose map: input[i,j] = y[j,i]  (y is [N,M], transposed to [M,N])
// composed: input1[j], input2[j,i] from output (i,j)
```

### Elemental Emission

Elemental emission generates code for computing a single output element. The indexing map tells the emitter which input elements to read:

```cpp
// Elemental emission for dot product
Status EmitDotElement(GpuEmitter* emitter,
                       const HloInstruction* dot,
                       int64_t output_m, int64_t output_n) {
  // Accumulate over the contraction dimension
  auto acc = emitter->EmitZeroAccumulator(dot->shape().element_type());

  int64_t k_dim = dot->operand(0)->shape().dimensions(1);
  for (int64_t k = 0; k < k_dim; ++k) {
    // Load from lhs[m, k]
    auto lhs_val = emitter->EmitLoad(lhs_buffer, {output_m, k});

    // Load from rhs[k, n]
    auto rhs_val = emitter->EmitLoad(rhs_buffer, {k, output_n});

    // Multiply and accumulate
    auto product = emitter->EmitMul(lhs_val, rhs_val);
    acc = emitter->EmitAdd(acc, product);
  }

  // Store the result
  emitter->EmitStore(output_buffer, acc, {output_m, output_n});
  return OkStatus();
}
```

#### Vectorized Emission

When the indexing map allows, the emitter can vectorize the computation to process multiple elements simultaneously:

```cpp
// Vectorized emission for elementwise operations
Status EmitVectorizedElementwise(GpuEmitter* emitter,
                                  const HloInstruction* op,
                                  int64_t vector_size) {
  // Compute a vector of output indices
  auto base_index = emitter->EmitGlobalThreadId() * vector_size;
  auto indices = emitter->EmitVectorOfIndices(base_index, vector_size);

  // Vector load from inputs
  auto input_vecs = emitter->EmitVectorLoad(input_buffer, indices, vector_size);

  // Vector operation
  auto result_vec = emitter->EmitVectorOp(op, input_vecs);

  // Vector store to output
  emitter->EmitVectorStore(output_buffer, result_vec, indices);
  return OkStatus();
}
```

## SparseCore Support

### Sparse Operations

XLA's SparseCore provides support for sparse tensor operations, which are important for embeddings and other operations on sparse data.

Sparse operations supported by XLA include:

1. **Sparse dense matrix multiplication**: Multiplying a sparse matrix by a dense matrix.
2. **Sparse embedding lookups**: Looking up embeddings from a sparse table.
3. **Sparse updates**: Updating specific elements of a tensor using sparse indices.
4. **Sparse reductions**: Reducing values at specific indices.

### Sparse Tensor Representation

Sparse tensors in XLA are represented using coordinate format (COO) or compressed sparse row (CSR) format:

#### COO (Coordinate) Format

A sparse tensor is represented as three arrays:
- **Values**: The non-zero values.
- **Indices**: The coordinates of each non-zero value.
- **Shape**: The full tensor shape (including zero elements).

```
# Dense tensor (3x3):
[[1, 0, 2],
 [0, 0, 3],
 [4, 5, 0]]

# COO representation:
values = [1, 2, 3, 4, 5]
indices = [(0,0), (0,2), (1,2), (2,0), (2,1)]
shape = (3, 3)
```

#### CSR (Compressed Sparse Row) Format

A sparse matrix is represented as three arrays:
- **Values**: The non-zero values in row-major order.
- **Column indices**: The column index for each non-zero value.
- **Row pointers**: The index into values/column_indices where each row starts.

```
# Dense matrix (3x3):
[[1, 0, 2],
 [0, 0, 3],
 [4, 5, 0]]

# CSR representation:
values = [1, 2, 3, 4, 5]
column_indices = [0, 2, 2, 0, 1]
row_pointers = [0, 2, 3, 5]  # Row 0 starts at 0, row 1 at 2, row 2 at 3, end at 5
```

### Sparse Operations in HLO

Sparse operations are represented in HLO using custom calls with sparse-specific attributes:

```
HloModule sparse_matmul

ENTRY main {
  // Sparse matrix in COO format
  %values = f32[5] parameter(0)         // Non-zero values
  %indices = s32[5, 2] parameter(1)     // Row and column indices
  %dense = f32[3, 4] parameter(2)       // Dense matrix

  // Sparse x Dense matrix multiplication
  %result = f32[3, 4] custom-call(%values, %indices, %dense),
      custom_call_target="SparseDenseMatMul",
      sparsity={sparse_format=coo, lhs_shape=[3,3]}

  ROOT %root = %result
}
```

### Indexing for Sparse Operations

Symbolic expressions and indexing maps are used for sparse operations:

1. **Sparse access pattern**: The indexing map for a sparse operation includes indirect indexing through the sparse index arrays:

```
// For sparse dense matmul: C[i,j] = sum_k A[i,k] * B[k,j]
// Where A is sparse (COO format)
// Only non-zero elements of A contribute to the sum

// Indexing for sparse iteration:
// For each non-zero entry (values[n], indices[n]):
//   i = indices[n][0]
//   k = indices[n][1]
//   C[i,j] += values[n] * B[k,j]
```

2. **Symbolic dimension expressions for sparse shapes**: The number of non-zero elements is often a symbolic expression:

```cpp
// Number of non-zero elements may be dynamic
SymbolicExpression nnz = SymbolicExpression::Parameter(/*nnz_id*/);

// Shape of the sparse tensor is symbolic
Shape sparse_shape = ShapeUtil::MakeShapeWithSymbolicDimensions(
    F32, {nnz, 2});  // indices array: [nnz, 2]
```

### SparseCore Emitter

The SparseCore emitter generates code for sparse operations on hardware that supports sparse computation:

```cpp
class SparseCoreEmitter {
 public:
  // Emit a sparse dense matmul
  Status EmitSparseDenseMatMul(
      const HloInstruction* op,
      const SparseDenseMatMulConfig& config) {
    // Get sparse matrix components
    auto values = GetBuffer(op->operand(0));   // Non-zero values
    auto indices = GetBuffer(op->operand(1));   // COO indices
    auto dense = GetBuffer(op->operand(2));     // Dense matrix
    auto output = GetOutputBuffer(op);          // Output buffer

    // Initialize output to zero
    EmitMemset(output, 0.0f);

    // Iterate over non-zero elements
    // Each thread processes a chunk of non-zero elements
    auto nnz = values.element_count();
    EmitParallelFor(nnz, [&](int64_t n) {
      // Load sparse entry
      auto val = EmitLoad(values, {n});
      auto row = EmitLoad(indices, {n, 0});
      auto col = EmitLoad(indices, {n, 1});

      // For each output column
      for (int64_t j = 0; j < dense_dim_1; ++j) {
        auto dense_val = EmitLoad(dense, {col, j});
        auto contribution = EmitMul(val, dense_val);
        // Atomic add to output (multiple threads may write to same row)
        EmitAtomicAdd(output, {row, j}, contribution);
      }
    });

    return OkStatus();
  }
};
```

### Sparse Operations and Dynamic Shapes

Sparse operations interact with dynamic shapes because the number of non-zero elements is often not known at compile time:

1. **Dynamic NNZ**: The number of non-zero elements is a symbolic expression.
2. **Dynamic work estimation**: The amount of work depends on the sparsity pattern, which is data-dependent.
3. **Load balancing**: Work distribution across threads/devices must account for non-uniform sparsity patterns.

XLA handles these challenges by:

- Using symbolic expressions to represent dynamic NNZ in the compilation pipeline.
- Generating code that queries the actual NNZ at runtime for work distribution.
- Providing load-balancing strategies for sparse operations (e.g., work-stealing, binning).

### Sparse Optimizations

XLA applies several optimizations specific to sparse operations:

1. **Sparse fusion**: Fusing sparse operations with subsequent dense operations to avoid materializing intermediate sparse tensors.

2. **Sparse format conversion**: Converting between sparse formats (COO, CSR, CSC) to match the most efficient format for the downstream operation.

3. **Sparse pruning**: Removing explicit zeros from sparse representations to reduce storage and computation.

4. **Sparse tiling**: Tiling sparse computations to improve cache locality and parallelism.
