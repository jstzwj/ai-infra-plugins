# PyTorch - Chapter 36: Linear Algebra (torch.linalg)

This reference covers all linear algebra operations in torch.linalg.

---

## 36.1 Matrix Decompositions

```python
torch.linalg.cholesky(A, *, upper=False, out=None)         # Cholesky: A = L @ L^T
torch.linalg.cholesky_ex(A, *, check_errors=False)          # With error detection
torch.linalg.qr(A, mode='reduced')                          # QR: A = Q @ R
torch.linalg.svd(A, full_matrices=True, *, driver=None)     # SVD: A = U @ S @ V^T
torch.linalg.svdvals(A, *, driver=None)                     # Singular values only
torch.linalg.lu_factor(A, *, pivot=True)                    # LU factorization
torch.linalg.lu_factor_ex(A, *, pivot=True, check_errors=False)
torch.linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False)
torch.linalg.eig(A)                                          # Eigenvalue decomposition
torch.linalg.eigh(A, UPLO='L')                               # Hermitian eigenvalues
torch.linalg.eigvals(A)                                      # Eigenvalues only
torch.linalg.eigvalsh(A, UPLO='L')                           # Hermitian eigenvalues only
```

---

## 36.2 Solvers

```python
torch.linalg.solve(A, B, *, left=True, adjoint=False)       # Solve Ax = B
torch.linalg.solve_ex(A, B, ..., check_errors=False)
torch.linalg.lstsq(A, B, *, rcond=None, driver=None)        # Least squares
torch.linalg.inv(A, *, out=None)                              # Matrix inverse
torch.linalg.inv_ex(A, *, check_errors=False)
torch.linalg.pinv(A, *, atol=None, rtol=None, hermitian=False) # Pseudo-inverse
```

---

## 36.3 Matrix Operations

```python
torch.linalg.det(A)                    # Determinant
torch.linalg.slogdet(A)                # Sign and log-det
torch.linalg.matrix_exp(A)             # Matrix exponential
torch.linalg.matrix_power(A, n)        # Matrix power
torch.linalg.matrix_norm(A, ord='fro', dim=(-2,-1))  # Matrix norm
torch.linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False)
torch.linalg.cond(A, p=None)           # Condition number
torch.linalg.cross(input, other, *, dim=-1)  # Cross product
torch.linalg.householder_product(A, tau)     # QR via Householder
```

---

## 36.4 Vector and Norm Operations

```python
torch.linalg.dot(x, y)                # Dot product (1D)
torch.linalg.vdot(x, y)               # Vector dot product (conjugate)
torch.linalg.outer(x, y)              # Outer product
torch.linalg.norm(input, ord=None, dim=None, keepdim=False)  # General norm
torch.linalg.vector_norm(input, ord=2, dim=None, keepdim=False)  # Vector norm
torch.linalg.multi_dot(tensors)       # Optimal chain matrix multiplication
```

---

## 36.5 Tensor Operations

```python
torch.linalg.tensorinv(input, ind=2)
torch.linalg.tensorsolve(A, B, dims=None)
torch.linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1)
torch.linalg.diag(input, *, offset=0)
```

---

## 36.6 Examples

```python
A = torch.randn(3, 3)
b = torch.randn(3)

# Solve Ax = b
x = torch.linalg.solve(A, b)

# SVD decomposition
U, S, Vh = torch.linalg.svd(A)
A_reconstructed = U @ torch.diag(S) @ Vh

# Cholesky (requires positive definite)
A_pd = A @ A.T + 3 * torch.eye(3)
L = torch.linalg.cholesky(A_pd)

# Condition number
c = torch.linalg.cond(A)
```
