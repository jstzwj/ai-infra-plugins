# PyTorch Sparse Tensors - Comprehensive Reference

This chapter covers sparse tensor support in PyTorch, including all sparse formats (COO, CSR, CSC, BSR, BSC), creation, conversion, operations, semi-structured sparsity, and best practices for when to use sparse tensors.

---

## 1. Sparse Formats Overview

PyTorch supports multiple sparse formats, each optimized for different use cases:

| Format | Full Name | Best For | Storage |
|--------|-----------|----------|---------|
| COO | Coordinate | Construction, arbitrary sparsity | indices + values |
| CSR | Compressed Sparse Row | Row slicing, matrix-vector multiply | crow + col_indices + values |
| CSC | Compressed Sparse Column | Column slicing, transposed ops | ccol + row_indices + values |
| BSR | Block Compressed Sparse Row | Block-structured sparsity | crow + col_indices + values (blocks) |
| BSC | Block Compressed Sparse Column | Block-structured sparsity (col) | ccol + row_indices + values (blocks) |

---

## 2. COO (Coordinate) Format

The simplest sparse format. Stores a list of (row, col, value) tuples.

### torch.sparse_coo_tensor

```python
torch.sparse_coo_tensor(
    indices,        # (Tensor) 2D tensor of shape [ndim, nnz] with index coordinates
    values,         # (Tensor) 1D or multi-dim tensor of non-zero values
    size=None,      # (tuple) explicit size of the sparse tensor
    *, dtype=None,
    device=None,
    requires_grad=False,
    is_coalesced=False,
    check_invariants=None,
)
```

```python
import torch

# Basic COO sparse tensor
indices = torch.tensor([[0, 1, 2],
                        [0, 1, 2]])
values = torch.tensor([1.0, 2.0, 3.0])

sparse = torch.sparse_coo_tensor(indices, values, size=(3, 3))
print(sparse)
# tensor(indices=tensor([[0, 1, 2],
#                          [0, 1, 2]]),
#        values=tensor([1., 2., 3.]),
#        size=(3, 3), nnz=3, layout=torch.sparse_coo)

# Convert to dense
dense = sparse.to_dense()
print(dense)
# tensor([[1., 0., 0.],
#         [0., 2., 0.],
#         [0., 0., 3.]])

# With explicit size (larger than max index)
sparse = torch.sparse_coo_tensor(indices, values, size=(5, 5))
print(sparse.to_dense().shape)  # torch.Size([5, 5])

# Multi-valued sparse tensor (values have extra dimensions)
indices = torch.tensor([[0, 1], [2, 3]])
values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Each value is 1D
sparse = torch.sparse_coo_tensor(indices, values, size=(4, 4, 2))
```

### Coalescing

```python
# Coalesce duplicate indices (sum their values)
indices = torch.tensor([[0, 0, 1], [0, 0, 2]])
values = torch.tensor([1.0, 2.0, 3.0])  # indices [0,0] appears twice

sparse = torch.sparse_coo_tensor(indices, values, size=(3, 3))
coalesced = sparse.coalesce()
print(coalesced.values())   # tensor([3., 3.])  (1+2=3)
print(coalesced.indices())
# tensor([[0, 1],
#         [0, 2]])

# Check if already coalesced
print(sparse.is_coalesced())   # False
print(coalesced.is_coalesced())  # True
```

### COO Tensor Attributes

```python
sparse = torch.sparse_coo_tensor(indices, values, size=(3, 3))

print(sparse.indices())   # Index tensor
print(sparse.values())    # Value tensor
print(sparse.shape)       # torch.Size([3, 3])
print(sparse.layout)      # torch.sparse_coo
print(sparse.nnz())       # Number of specified values
print(sparse.is_coalesced())  # Whether indices are coalesced
```

### Creating COO Tensors from Dense

```python
# Convert dense to sparse
dense = torch.tensor([[1.0, 0.0, 2.0],
                       [0.0, 3.0, 0.0],
                       [4.0, 0.0, 5.0]])

sparse = dense.to_sparse()
print(sparse)
print(sparse.indices())
print(sparse.values())

# Convert back to dense
dense_back = sparse.to_dense()
print(torch.allclose(dense, dense_back))  # True
```

---

## 3. CSR (Compressed Sparse Row) Format

Efficient for row-based operations and matrix-vector multiplication.

### Creation

```python
# From COO
sparse_coo = torch.sparse_coo_tensor(indices, values, size=(4, 4))
sparse_csr = sparse_coo.to_sparse_csr()

# From dense
dense = torch.tensor([[1.0, 0.0, 2.0],
                       [0.0, 0.0, 3.0],
                       [4.0, 5.0, 0.0],
                       [0.0, 0.0, 0.0]])
sparse_csr = dense.to_sparse_csr()

# Direct construction
crow_indices = torch.tensor([0, 2, 3, 5, 5])   # Row pointers (cumulative nnz)
col_indices = torch.tensor([0, 2, 2, 0, 1])     # Column indices
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
sparse_csr = torch.sparse_csr_tensor(
    crow_indices, col_indices, values, size=(4, 3)
)
```

### CSR Tensor Attributes

```python
print(sparse_csr.crow_indices())   # Row pointer tensor
print(sparse_csr.col_indices())    # Column index tensor
print(sparse_csr.values())         # Value tensor
print(sparse_csr.shape)            # torch.Size([4, 3])
print(sparse_csr.layout)           # torch.sparse_csr
print(sparse_csr.nnz())            # Number of non-zeros
```

### Row Slicing

```python
# CSR format is efficient for row slicing
row_1 = sparse_csr[1]  # Access row 1 efficiently
print(row_1.to_dense())
```

---

## 4. CSC (Compressed Sparse Column) Format

Efficient for column-based operations.

### Creation

```python
# From COO
sparse_coo = torch.sparse_coo_tensor(indices, values, size=(4, 4))
sparse_csc = sparse_coo.to_sparse_csc()

# From dense
dense = torch.randn(4, 4)
sparse_csc = dense.to_sparse_csc()

# Direct construction
ccol_indices = torch.tensor([0, 2, 3, 5])  # Column pointers
row_indices = torch.tensor([0, 2, 1, 0, 1]) # Row indices
values = torch.tensor([1.0, 4.0, 3.0, 2.0, 5.0])
sparse_csc = torch.sparse_csc_tensor(
    ccol_indices, row_indices, values, size=(3, 3)
)
```

### CSC Tensor Attributes

```python
print(sparse_csc.ccol_indices())   # Column pointer tensor
print(sparse_csc.row_indices())    # Row index tensor
print(sparse_csc.values())         # Value tensor
print(sparse_csc.layout)           # torch.sparse_csc
```

---

## 5. BSR (Block Compressed Sparse Row) Format

Block-sparse format for 2D block-structured sparsity patterns.

### Creation

```python
# From dense with block size
dense = torch.tensor([
    [1.0, 2.0, 0.0, 0.0],
    [3.0, 4.0, 0.0, 0.0],
    [0.0, 0.0, 5.0, 6.0],
    [0.0, 0.0, 7.0, 8.0],
])
bsr = dense.to_sparse_bsr(blocksize=(2, 2))

# Direct construction
crow_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([0, 1])
values = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]],
])
bsr = torch.sparse_bsr_tensor(
    crow_indices, col_indices, values, size=(4, 4)
)
```

### BSR Tensor Attributes

```python
print(bsr.crow_indices())   # Block row pointers
print(bsr.col_indices())    # Block column indices
print(bsr.values())         # Block values tensor
print(bsr.shape)            # Overall tensor shape
```

---

## 6. BSC (Block Compressed Sparse Column) Format

Block-sparse format for column-oriented block structure.

### Creation

```python
# From dense with block size
dense = torch.randn(8, 8)
bsc = dense.to_sparse_bsc(blocksize=(2, 2))

# Direct construction
ccol_indices = torch.tensor([0, 1, 2])
row_indices = torch.tensor([0, 1])
values = torch.randn(2, 2, 2)  # 2 blocks, each 2x2
bsc = torch.sparse_bsc_tensor(
    ccol_indices, row_indices, values, size=(4, 4)
)
```

---

## 7. Format Conversion

```python
# Start with any format
dense = torch.randn(10, 10)
dense[dense < 0.5] = 0  # Make sparse

# Convert to all formats
sparse_coo = dense.to_sparse()           # COO (default)
sparse_csr = dense.to_sparse_csr()       # CSR
sparse_csc = dense.to_sparse_csc()       # CSC
sparse_bsr = dense.to_sparse_bsr(blocksize=(2, 2))  # BSR
sparse_bsc = dense.to_sparse_bsc(blocksize=(2, 2))  # BSC

# Convert back to dense
dense_from_coo = sparse_coo.to_dense()
dense_from_csr = sparse_csr.to_dense()
dense_from_csc = sparse_csc.to_dense()
dense_from_bsr = sparse_bsr.to_dense()
dense_from_bsc = sparse_bsc.to_dense()

# Convert between sparse formats
csr_from_coo = sparse_coo.to_sparse_csr()
csc_from_csr = sparse_csr.to_sparse_csc()
coo_from_csr = sparse_csr.to_sparse_coo()
```

---

## 8. Sparse Operations

### Sparse Matrix Multiplication (spmm)

```python
# Sparse @ Dense matrix multiplication
sparse_A = torch.randn(10, 20).to_sparse()
dense_B = torch.randn(20, 5)

result = torch.sparse.mm(sparse_A, dense_B)
print(result.shape)  # torch.Size([10, 5])
```

### torch.sparse.mm

```python
torch.sparse.mm(
    mat1,          # (SparseTensor) sparse matrix
    mat2,          # (Tensor) dense matrix
    *, reduce=None # (str) reduction mode: 'sum', 'amax', 'amin', 'mean'
)
```

```python
# Basic sparse-dense multiplication
A = torch.randn(100, 200).to_sparse()
B = torch.randn(200, 50)
C = torch.sparse.mm(A, B)
print(C.shape)  # torch.Size([100, 50])

# With reduction (for uncoalesced inputs)
A_coalesced = A.coalesce()
C = torch.sparse.mm(A_coalesced, B)
```

### torch.sparse.sum

```python
# Sum all non-zero values
sparse = torch.randn(3, 4).to_sparse()
total = torch.sparse.sum(sparse)
print(total)

# Sum along a dimension
row_sums = torch.sparse.sum(sparse, dim=1)
col_sums = torch.sparse.sum(sparse, dim=0)
```

### torch.sparse.addmm

```python
# beta * input + alpha * (mat1 @ mat2) where mat1 is sparse
mat1 = torch.randn(3, 4).to_sparse()
mat2 = torch.randn(4, 5)
input_dense = torch.randn(3, 5)

result = torch.sparse.addmm(input_dense, mat1, mat2, beta=1.0, alpha=1.0)
print(result.shape)  # torch.Size([3, 5])
```

### torch.sparse.softmax

```python
# Softmax on sparse tensor
sparse = torch.randn(3, 5).to_sparse()
soft = torch.sparse.softmax(sparse, dim=-1)
print(soft.to_dense())  # Each row sums to 1

# With specified dtype
soft = torch.sparse.softmax(sparse, dim=-1, dtype=torch.float64)
```

### Element-wise Operations

```python
# Scalar multiplication
sparse = torch.randn(3, 3).to_sparse()
scaled = sparse * 2.0

# Addition (sparse + sparse may densify)
sparse1 = torch.randn(3, 3).to_sparse()
sparse2 = torch.randn(3, 3).to_sparse()

# This returns dense
result = sparse1 + sparse2

# To keep sparse, convert back
result_sparse = result.to_sparse()
```

### Transpose

```python
sparse = torch.randn(3, 5).to_sparse()

# Transpose (efficient for COO)
transposed = sparse.t()
print(transposed.shape)  # torch.Size([5, 3])
```

---

## 9. Semi-Structured Sparsity

PyTorch supports 2:4 semi-structured sparsity (NVIDIA Ampere+), where 2 out of every 4 elements are zero in a structured pattern.

### Semi-Structured Sparsity Pattern

```
For every 4 consecutive elements:
[0, x, 0, x]  - valid (2 zeros, 2 non-zeros)
[x, 0, x, 0]  - valid
[0, 0, x, x]  - valid
[x, x, 0, 0]  - valid
[x, 0, 0, x]  - valid
[0, x, x, 0]  - valid

[0, 0, 0, x]  - INVALID (only 1 non-zero)
[x, x, x, x]  - INVALID (0 zeros)
```

### Benefits of Semi-Structured Sparsity

```python
# 2:4 sparsity enables hardware acceleration on NVIDIA Ampere+
# - 2x theoretical speedup for matrix multiplication
# - Maintains model accuracy (pruning typically loses <1% accuracy)
# - Supported in cuSPARSE for structured patterns

# Requires CUDA with compute capability >= 8.0
```

---

## 10. When to Use Sparse Tensors

### Guidelines

```python
# USE SPARSE when:
# 1. Sparsity ratio is high (>90% zeros)
# 2. The sparsity pattern is structured or predictable
# 3. You need memory savings for very large tensors
# 4. Doing sparse matrix-vector multiplication
# 5. Working with graph adjacency matrices

# DO NOT USE SPARSE when:
# 1. Sparsity ratio is low (<70% zeros)
# 2. Dense operations would be faster
# 3. You need many element-wise operations
# 4. The tensor is small
# 5. You need operations not supported on sparse tensors

# Rule of thumb: sparse is worthwhile when nnz/total < 0.1
dense = torch.randn(1000, 1000)
dense[dense.abs() < 2.0] = 0
sparsity = (dense == 0).float().mean()
print(f"Sparsity: {sparsity:.2%}")
if sparsity > 0.9:
    sparse = dense.to_sparse()
    # Sparse will likely be beneficial
else:
    # Stick with dense
    pass
```

### Performance Comparison

```python
import time

def benchmark_sparse_dense(n, sparsity, n_iter=100):
    """Compare sparse vs dense matrix-vector multiply."""
    # Create test data
    dense = torch.randn(n, n)
    mask = torch.rand(n, n) > sparsity
    dense = dense * mask

    vec = torch.randn(n)

    # Dense benchmark
    start = time.time()
    for _ in range(n_iter):
        _ = dense @ vec
    dense_time = time.time() - start

    # Sparse benchmark
    sparse = dense.to_sparse_csr()
    start = time.time()
    for _ in range(n_iter):
        _ = torch.sparse.mm(sparse.unsqueeze(0), vec.unsqueeze(0)).squeeze()
    sparse_time = time.time() - start

    print(f"n={n}, sparsity={sparsity:.0%}")
    print(f"  Dense:  {dense_time:.4f}s")
    print(f"  Sparse: {sparse_time:.4f}s")
    print(f"  Speedup: {dense_time/sparse_time:.2f}x")
```

---

## 11. Practical Examples

### Sparse Adjacency Matrix (Graph Neural Networks)

```python
def create_adjacency_sparse(num_nodes, edges):
    """Create a sparse adjacency matrix from edge list."""
    src, dst = edges[:, 0], edges[:, 1]

    # COO format is natural for edge lists
    indices = torch.stack([src, dst])
    values = torch.ones(edges.shape[0])

    adj = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_nodes)
    )

    # Coalesce to sum duplicate edges
    adj = adj.coalesce()

    return adj

# Example: 5-node graph
edges = torch.tensor([
    [0, 1], [1, 0],  # Edge 0-1 (bidirectional)
    [1, 2], [2, 1],  # Edge 1-2
    [2, 3], [3, 2],  # Edge 2-3
    [3, 4], [4, 3],  # Edge 3-4
])
adj = create_adjacency_sparse(5, edges)
print(adj.to_dense())

# Sparse matrix multiplication for message passing
node_features = torch.randn(5, 16)  # 5 nodes, 16 features
messages = torch.sparse.mm(adj, node_features)
print(messages.shape)  # torch.Size([5, 16])
```

### Sparse One-Hot Encoding

```python
def sparse_one_hot(indices, num_classes):
    """Create sparse one-hot encoding."""
    batch_size = indices.shape[0]
    row_indices = torch.arange(batch_size).unsqueeze(0)
    col_indices = indices.unsqueeze(0)

    sparse_indices = torch.cat([row_indices, col_indices], dim=0)
    values = torch.ones(batch_size)

    return torch.sparse_coo_tensor(
        sparse_indices, values, size=(batch_size, num_classes)
    )

# Usage
labels = torch.tensor([0, 2, 5, 1, 3])
one_hot = sparse_one_hot(labels, num_classes=10)
print(one_hot.to_dense())
```

### Sparse Attention Mask

```python
def create_sparse_attention_mask(seq_len, window_size):
    """Create a banded (local) attention mask as sparse tensor."""
    indices_list = []
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        for j in range(start, end):
            indices_list.append([i, j])

    indices_t = torch.tensor(indices_list).T
    values_t = torch.ones(indices_t.shape[1])
    mask = torch.sparse_coo_tensor(
        indices_t, values_t, size=(seq_len, seq_len)
    )
    return mask

# Usage: 512-length sequence with window of 64 tokens
mask = create_sparse_attention_mask(512, 64)
print(f"Dense size: {512 * 512 * 4} bytes")
print(f"Sparse nnz: {mask.nnz()} entries")
```

### Converting Between SciPy and PyTorch Sparse

```python
def scipy_csr_to_torch(scipy_sparse):
    """Convert SciPy CSR sparse matrix to PyTorch CSR tensor."""
    crow = torch.tensor(scipy_sparse.indptr, dtype=torch.int64)
    col = torch.tensor(scipy_sparse.indices, dtype=torch.int64)
    val = torch.tensor(scipy_sparse.data)
    return torch.sparse_csr_tensor(
        crow, col, val, size=scipy_sparse.shape
    )

def torch_csr_to_scipy(torch_sparse):
    """Convert PyTorch CSR sparse tensor to SciPy CSR matrix."""
    import scipy.sparse as sp
    return sp.csr_matrix(
        (torch_sparse.values().numpy(),
         torch_sparse.col_indices().numpy(),
         torch_sparse.crow_indices().numpy()),
        shape=torch_sparse.shape,
    )
```

---

## 12. Sparse Tensor Gradients

```python
# Sparse tensors support autograd
sparse = torch.randn(5, 5, requires_grad=True)
sparse.data[sparse.data.abs() < 0.5] = 0
sparse_coo = sparse.to_sparse()

# Forward pass
result = torch.sparse.mm(sparse_coo, torch.randn(5, 3))
loss = result.sum()

# Backward pass
loss.backward()
print(sparse.grad is not None)  # True (gradient is dense)
```

---

## 13. Limitations and Gotchas

```python
# Limitation 1: Not all operations support sparse tensors
sparse = torch.randn(5, 5).to_sparse()
# These may not work or will densify:
# - torch.norm(sparse)        # Densifies
# - torch.sigmoid(sparse)     # Densifies
# - Many element-wise ops      # Densify

# Limitation 2: Batched sparse tensors are limited
# COO supports batched tensors, CSR/CSC are 2D only

# Limitation 3: Gradient computation may densify
# Gradients of sparse operations are typically dense

# Limitation 4: Conversion overhead
# Converting between formats has a cost
# Best to choose one format and stick with it

# Gotcha: Empty sparse tensors
empty = torch.sparse_coo_tensor(
    torch.zeros(2, 0, dtype=torch.long),
    torch.zeros(0),
    size=(5, 5),
)
print(empty.nnz())  # 0
```
