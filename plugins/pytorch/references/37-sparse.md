# PyTorch - Chapter 37: Sparse Tensors

This reference covers sparse tensor formats and operations in PyTorch.

---

## 37.1 Sparse Formats

| Format | Method | Description |
|--------|--------|-------------|
| COO | `to_sparse()` | Coordinate format (default sparse) |
| CSR | `to_sparse_csr()` | Compressed Sparse Row |
| CSC | `to_sparse_csc()` | Compressed Sparse Column |
| BSR | `to_sparse_bsr()` | Block Sparse Row |
| BSC | `to_sparse_bsc()` | Block Sparse Column |

---

## 37.2 Creating Sparse Tensors

```python
# From dense
dense = torch.randn(3, 3).to_sparse()         # COO format
sparse_csr = dense.to_sparse_csr()             # CSR format

# COO directly
indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
sparse = torch.sparse_coo_tensor(indices, values, size=(3, 3))

# CSR directly
crow_indices = torch.tensor([0, 1, 2, 3])
col_indices = torch.tensor([0, 1, 2])
values = torch.tensor([1.0, 2.0, 3.0])
sparse_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))
```

---

## 37.3 Sparse Operations

```python
torch.sparse.mm(sparse, dense)               # Sparse × Dense matrix multiply
torch.sparse.sum(input, dim=None)            # Sum sparse tensor
torch.sparse.addmm(input, mat1, mat2)        # input + mat1 @ mat2
torch.sparse.softmax(input, dim)             # Softmax on sparse tensor
torch.sparse.spdiags(diagonals, offsets, shape)  # Diagonal sparse matrix
```

---

## 37.4 Properties and Conversion

```python
sparse.is_sparse       # True
sparse.is_coalesced()  # Check if COO is coalesced
sparse.coalesce()      # Coalesce COO (remove duplicates)
sparse.indices()       # COO indices
sparse.values()        # Non-zero values
sparse.to_dense()      # Convert back to dense
sparse._nnz()          # Number of non-zero elements
```

---

## 37.5 Semi-Structured Sparsity (2:4)

```python
# NVIDIA's 2:4 sparsity pattern (2 zeros per 4 elements)
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

dense = torch.randn(32, 32, device='cuda')
sparse = to_sparse_semi_structured(dense)
# Hardware-accelerated sparse matrix multiply on Ampere+
```
