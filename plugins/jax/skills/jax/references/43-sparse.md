# 43 - Sparse Matrices

## Overview

JAX provides experimental support for sparse matrix operations through `jax.experimental.sparse`. The primary format is BCOO (Batched Coordinate).

---

## 1. BCOO Format

### Structure

BCOO stores a sparse matrix using:
- `data`: Non-zero values, shape `(nse,) + batch_dims + block_dims`
- `indices`: Column indices, shape `(nse, n_sparse) + batch_dims`

```python
from jax.experimental import sparse
import jax.numpy as jnp

# Create from dense
dense = jnp.array([[1., 0., 2.],
                    [0., 0., 3.],
                    [4., 5., 0.]])
sp = sparse.BCOO.fromdense(dense)

print(sp.data)     # [1. 2. 3. 4. 5.]
print(sp.indices)  # [[0] [2] [2] [0] [1]] (column indices)
print(sp.shape)    # (3, 3)
print(sp.nse)      # 5 (number of stored elements)
```

### Creating BCOO directly

```python
data = jnp.array([1.0, 2.0, 3.0])
indices = jnp.array([[0, 0], [1, 2], [2, 1]])
sp = sparse.BCOO((data, indices), shape=(3, 3))
```

---

## 2. Sparse Operations

### Matrix multiplication

```python
# Sparse × Dense
result = sp @ jnp.ones(3)  # Returns dense

# Dense × Sparse (transpose)
result = jnp.ones(3) @ sp  # May convert to dense

# Sparse × Sparse (limited support)
```

### Element-wise operations

```python
# Scalar operations
sp2 = sp * 2.0
sp3 = sp + 5.0

# Sparse + Sparse (same sparsity pattern)
```

### Reductions

```python
sp.sum()           # Sum of all elements
sp.sum(axis=0)     # Column-wise sum
sp.sum(axis=1)     # Row-wise sum
sp.max()           # Max element
sp.min()           # Min element
```

---

## 3. Conversion Functions

```python
# Dense → Sparse
sp = sparse.BCOO.fromdense(dense_matrix, nse=5)  # Specify nse
sp = sparse.BCOO.fromdense(dense_matrix)           # Auto-detect

# Sparse → Dense
dense = sp.todense()

# Sparsify a function
@sparse.sparsify
def f(x):
    return x @ x.T
```

---

## 4. Sparsify Transform

```python
# Automatically use sparse operations where beneficial
@sparse.sparsify
def sparse_matmul(x, y):
    return x @ y

result = sparse_matmul(sp_matrix, dense_vector)
```

---

## 5. BCSR Format (Block CSR)

```python
from jax.experimental.sparse import BCSR

# Block Compressed Sparse Row format
# Better for structured sparsity patterns
bcsr = BCSR.fromdense(dense_matrix, block_size=(2, 2))
```

---

## 6. Performance Considerations

### When to use sparse

- Matrix has >90% zeros
- Sparse operations available for your use case
- Memory savings are significant

### When NOT to use sparse

- Dense matrix fits in memory
- Operations not supported in sparse
- Sparse representation overhead is too large
- Need maximum performance (sparse ops are slower per-element)

---

## 7. Supported Operations

| Operation | BCOO | BCSR |
|---|---|---|
| Creation from dense | Yes | Yes |
| Conversion to dense | Yes | Yes |
| Matrix multiply (SpMM) | Yes | Yes |
| Element-wise scalar | Yes | Yes |
| Reduction (sum) | Yes | Yes |
| Transpose | Yes | Yes |
| Slicing | Limited | Limited |
| JIT compatible | Yes | Yes |
| Grad compatible | Partial | Partial |
