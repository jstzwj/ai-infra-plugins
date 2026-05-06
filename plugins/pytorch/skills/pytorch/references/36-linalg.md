# PyTorch Linear Algebra - Comprehensive Reference

This chapter covers all linear algebra operations in `torch.linalg`, including decompositions, solvers, matrix operations, norms, and eigenvalue problems.

---

## 1. Decompositions

### cholesky

Computes the Cholesky decomposition of a symmetric positive-definite matrix.

```python
torch.linalg.cholesky(A, *, upper=False, out=None)
```

```python
import torch

# Create a positive-definite matrix
A = torch.randn(3, 3)
A = A @ A.T + torch.eye(3)  # Ensure positive definite

# Cholesky decomposition: A = L @ L^T
L = torch.linalg.cholesky(A)
print(torch.allclose(A, L @ L.T))  # True

# Upper triangular variant: A = U^T @ U
U = torch.linalg.cholesky(A, upper=True)
print(torch.allclose(A, U.T @ U))  # True

# Batched Cholesky
A_batch = torch.randn(10, 3, 3)
A_batch = A_batch @ A_batch.transpose(-1, -2) + 3 * torch.eye(3)
L_batch = torch.linalg.cholesky(A_batch)
print(L_batch.shape)  # torch.Size([10, 3, 3])
```

### cholesky_ex

Computes the Cholesky decomposition with error checking.

```python
torch.linalg.cholesky_ex(A, *, upper=False, check_errors=False, out=None)
```

```python
# Returns (L, info) tuple
A = torch.randn(3, 3)
A = A @ A.T + torch.eye(3)

L, info = torch.linalg.cholesky_ex(A)
print(info)  # tensor(0) = success; >0 means not positive definite

# With error checking (raises RuntimeError if not PD)
L, info = torch.linalg.cholesky_ex(A, check_errors=True)

# Safe PD check
def is_positive_definite(A):
    _, info = torch.linalg.cholesky_ex(A)
    return (info == 0).all()
```

### lu_factor

Computes a compact LU factorization with partial pivoting.

```python
torch.linalg.lu_factor(A, *, pivot=True, out=None)
```

```python
A = torch.randn(3, 3)
LU, pivots = torch.linalg.lu_factor(A)

# Use with lu_solve to solve linear systems
b = torch.randn(3)
x = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(-1)).squeeze(-1)
print(torch.allclose(A @ x, b, atol=1e-5))  # True
```

### lu_solve

Solves a linear system using the LU factorization from `lu_factor`.

```python
torch.linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None)
```

```python
A = torch.randn(5, 5)
B = torch.randn(5, 3)
LU, pivots = torch.linalg.lu_factor(A)
X = torch.linalg.lu_solve(LU, pivots, B)
print(torch.allclose(A @ X, B, atol=1e-5))  # True

# Solve A^T @ X = B
X = torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
print(torch.allclose(A.T @ X, B, atol=1e-5))  # True
```

### qr

Computes the QR decomposition.

```python
torch.linalg.qr(A, mode='reduced', *, out=None)
```

```python
A = torch.randn(5, 3)

# Reduced QR: Q is (m, k), R is (k, n) where k = min(m, n)
Q, R = torch.linalg.qr(A)
print(Q.shape)  # torch.Size([5, 3])
print(R.shape)  # torch.Size([3, 3])
print(torch.allclose(A, Q @ R, atol=1e-5))  # True

# Complete QR: Q is (m, m), R is (m, n)
Q_full, R_full = torch.linalg.qr(A, mode='complete')
print(Q_full.shape)  # torch.Size([5, 5])
print(R_full.shape)  # torch.Size([5, 3])

# Verify Q is orthogonal
print(torch.allclose(Q.T @ Q, torch.eye(3), atol=1e-5))  # True
```

### svd

Computes the singular value decomposition.

```python
torch.linalg.svd(A, full_matrices=True, *, driver=None, out=None)
```

```python
A = torch.randn(5, 3)

# Full SVD: A = U @ diag(S) @ Vh
U, S, Vh = torch.linalg.svd(A, full_matrices=True)
print(U.shape)   # torch.Size([5, 5])
print(S.shape)   # torch.Size([3])
print(Vh.shape)  # torch.Size([3, 3])

# Reconstruct
A_reconstructed = U[:, :3] @ torch.diag(S) @ Vh
print(torch.allclose(A, A_reconstructed, atol=1e-5))  # True

# Reduced SVD
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
print(U.shape)   # torch.Size([5, 3])

# Driver options (on CUDA): 'gesvd', 'gesvdj', 'gesvda', 'gesvdp'
U, S, Vh = torch.linalg.svd(A, driver='gesvdj')
```

### svdvals

Computes only the singular values (faster than full SVD).

```python
torch.linalg.svdvals(A, *, driver=None, out=None)
```

```python
A = torch.randn(5, 3)
S = torch.linalg.svdvals(A)
print(S.shape)  # torch.Size([3])

# Matrix rank estimation
rank = (S > 1e-5).sum().item()
print(f"Estimated rank: {rank}")
```

### eig

Computes the eigenvalue decomposition of a general square matrix.

```python
torch.linalg.eig(A, *, out=None)
```

```python
A = torch.randn(3, 3)
eigenvalues, eigenvectors = torch.linalg.eig(A)

print(eigenvalues)       # Complex eigenvalues
print(eigenvectors)      # Complex eigenvectors

# Verify: A @ v = lambda * v
for i in range(3):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(torch.allclose(A @ v, lam * v, atol=1e-4))  # True
```

### eigh

Computes the eigenvalue decomposition of a symmetric/Hermitian matrix.

```python
torch.linalg.eigh(A, UPLO='L', *, out=None)
```

```python
# Symmetric matrix
A = torch.randn(3, 3)
A = A + A.T  # Symmetric

eigenvalues, eigenvectors = torch.linalg.eigh(A)
print(eigenvalues)  # Real eigenvalues, sorted in ascending order

# Verify orthogonality
print(torch.allclose(eigenvectors.T @ eigenvectors, torch.eye(3), atol=1e-5))

# Upper triangle variant
eigenvalues, eigenvectors = torch.linalg.eigh(A, UPLO='U')

# Batched
A_batch = torch.randn(10, 3, 3)
A_batch = A_batch + A_batch.transpose(-1, -2)
evals, evecs = torch.linalg.eigh(A_batch)
```

### eigvals

Computes eigenvalues of a general matrix.

```python
torch.linalg.eigvals(A, *, out=None)
```

```python
A = torch.randn(3, 3)
eigenvalues = torch.linalg.eigvals(A)
print(eigenvalues)  # Complex tensor
```

### eigvalsh

Computes eigenvalues of a symmetric/Hermitian matrix.

```python
torch.linalg.eigvalsh(A, UPLO='L', *, out=None)
```

```python
A = torch.randn(3, 3)
A = A + A.T
eigenvalues = torch.linalg.eigvalsh(A)
print(eigenvalues)  # Real tensor, sorted ascending
```

---

## 2. Solvers

### solve

Solves a square system of linear equations.

```python
torch.linalg.solve(A, B, *, left=True, out=None)
```

```python
# Solve A @ x = b for single vector
A = torch.randn(3, 3)
b = torch.randn(3)
x = torch.linalg.solve(A, b)
print(torch.allclose(A @ x, b, atol=1e-5))  # True

# Solve for multiple right-hand sides
A = torch.randn(5, 5)
B = torch.randn(5, 3)
X = torch.linalg.solve(A, B)
print(torch.allclose(A @ X, B, atol=1e-5))  # True

# Solve X @ A = B (left=False)
X = torch.linalg.solve(A, B, left=False)
print(torch.allclose(X @ A, B, atol=1e-5))  # True

# Batched
A_batch = torch.randn(10, 5, 5)
B_batch = torch.randn(10, 5)
X_batch = torch.linalg.solve(A_batch, B_batch)
```

### solve_ex

Like `solve` but returns info instead of raising on singular matrices.

```python
torch.linalg.solve_ex(A, B, *, left=True, check_errors=False, out=None)
```

```python
A = torch.randn(3, 3)
B = torch.randn(3)
X, info = torch.linalg.solve_ex(A, B)
print(info)  # 0 = success
```

### lstsq

Computes the least squares solution to an overdetermined system.

```python
torch.linalg.lstsq(A, B, rcond=None, *, driver=None, out=None)
```

```python
# Least squares: minimize ||A @ x - B||^2
A = torch.randn(10, 3)  # Overdetermined (more rows than columns)
B = torch.randn(10)
result = torch.linalg.lstsq(A, B)
X = result.solution
print(X.shape)  # torch.Size([3])

# With residual information
print(result.solution)         # torch.Size([3])
print(result.residuals)        # torch.Size([]) - sum of squared residuals
print(result.rank)             # Effective rank
print(result.singular_values)  # Singular values of A

# Driver options: 'gels', 'gelsy', 'gelsd', 'gelss'
X = torch.linalg.lstsq(A, B, driver='gelsd').solution
```

### tensorsolve

Solves a tensor equation.

```python
torch.linalg.tensorsolve(A, B, dims=None, *, out=None)
```

```python
# Solve for X in: torch.tensordot(A, X, dims) = B
A = torch.randn(2, 3, 3, 2)
B = torch.randn(3, 3)
X = torch.linalg.tensorsolve(A, B)
print(X.shape)  # torch.Size([2, 2])
```

### tensorinv

Computes the tensor inverse.

```python
torch.linalg.tensorinv(A, ind=2, *, out=None)
```

```python
A = torch.randn(4, 6, 8, 3)
# ind=2: treat first 2 dims as product and remaining as product
Ainv = torch.linalg.tensorinv(A, ind=2)
```

---

## 3. Matrix Operations

### inv

Computes the inverse of a square matrix.

```python
torch.linalg.inv(A, *, out=None)
```

```python
A = torch.randn(3, 3)
A_inv = torch.linalg.inv(A)
print(torch.allclose(A @ A_inv, torch.eye(3), atol=1e-5))  # True

# Batched
A_batch = torch.randn(10, 3, 3)
A_inv_batch = torch.linalg.inv(A_batch)

# Note: inv() is numerically less stable than solve()
# Prefer solve(A, B) over inv(A) @ B when possible
```

### inv_ex

Like `inv` but returns info instead of raising on singular matrices.

```python
torch.linalg.inv_ex(A, *, check_errors=False, out=None)
```

```python
A = torch.randn(3, 3)
A_inv, info = torch.linalg.inv_ex(A)
print(info)  # 0 = success
```

### matmul

Matrix product of two tensors.

```python
torch.linalg.matmul(input, other, *, out=None)
```

```python
# 1D x 1D: dot product
a = torch.randn(3)
b = torch.randn(3)
result = torch.linalg.matmul(a, b)  # scalar

# 2D x 2D: matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.linalg.matmul(A, B)  # torch.Size([3, 5])

# 1D x 2D: broadcast vector
v = torch.randn(4)
M = torch.randn(4, 5)
result = torch.linalg.matmul(v, M)  # torch.Size([5])

# Batched
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
C = torch.linalg.matmul(A, B)  # torch.Size([10, 3, 5])
```

### matrix_exp

Computes the matrix exponential.

```python
torch.linalg.matrix_exp(A)
```

```python
A = torch.randn(3, 3)
expA = torch.linalg.matrix_exp(A)

# Verify: exp(0) = I
print(torch.allclose(
    torch.linalg.matrix_exp(torch.zeros(3, 3)),
    torch.eye(3),
    atol=1e-5
))  # True

# Batched
A_batch = torch.randn(5, 3, 3)
expA_batch = torch.linalg.matrix_exp(A_batch)
```

### matrix_norm

Computes a matrix norm.

```python
torch.linalg.matrix_norm(
    A,                     # Input tensor
    ord='fro',             # Norm order: 'fro', 'nuc', 1, 2, -1, -2, inf, -inf
    dim=(-2, -1),          # Dimensions to compute norm over
    keepdim=False,         # Keep dimensions
    *, dtype=None, out=None,
)
```

```python
A = torch.randn(3, 4)

# Frobenius norm (default)
norm_fro = torch.linalg.matrix_norm(A)

# Spectral norm (largest singular value)
norm_2 = torch.linalg.matrix_norm(A, ord=2)

# Nuclear norm (sum of singular values)
norm_nuc = torch.linalg.matrix_norm(A, ord='nuc')

# 1-norm (max column sum)
norm_1 = torch.linalg.matrix_norm(A, ord=1)

# Infinity norm (max row sum)
norm_inf = torch.linalg.matrix_norm(A, ord=float('inf'))
```

### matrix_power

Computes the n-th power of a square matrix.

```python
torch.linalg.matrix_power(A, n, *, out=None)
```

```python
A = torch.randn(3, 3)

# Positive power
A3 = torch.linalg.matrix_power(A, 3)
print(torch.allclose(A3, A @ A @ A, atol=1e-4))  # True

# Inverse power
A_neg2 = torch.linalg.matrix_power(A, -2)
print(torch.allclose(A_neg2, torch.linalg.inv(A @ A), atol=1e-4))

# Zero power = identity
A0 = torch.linalg.matrix_power(A, 0)
print(torch.allclose(A0, torch.eye(3)))  # True
```

### matrix_rank

Computes the numerical rank of a matrix.

```python
torch.linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None)
```

```python
A = torch.randn(5, 3)
rank = torch.linalg.matrix_rank(A)
print(rank)  # Usually 3 (full rank)

# Rank-deficient matrix
A = torch.tensor([[1.0, 2.0, 3.0],
                   [2.0, 4.0, 6.0],  # Row 2 = 2 * Row 1
                   [0.0, 1.0, 1.0]])
rank = torch.linalg.matrix_rank(A)
print(rank)  # tensor(2)

# With tolerance
rank = torch.linalg.matrix_rank(A, atol=1e-5)

# For Hermitian matrices
A = torch.randn(3, 3)
A = A + A.T
rank = torch.linalg.matrix_rank(A, hermitian=True)
```

### multi_dot

Efficiently multiplies two or more matrices.

```python
torch.linalg.multi_dot(tensors, *, out=None)
```

```python
A = torch.randn(5, 3)
B = torch.randn(3, 4)
C = torch.randn(4, 2)

# Efficiently computes A @ B @ C (chooses optimal order)
result = torch.linalg.multi_dot([A, B, C])
print(result.shape)  # torch.Size([5, 2])

# More matrices
D = torch.randn(2, 6)
result = torch.linalg.multi_dot([A, B, C, D])
print(result.shape)  # torch.Size([5, 6])
```

### pinv

Computes the pseudoinverse (Moore-Penrose inverse).

```python
torch.linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None)
```

```python
A = torch.randn(5, 3)
A_pinv = torch.linalg.pinv(A)

# Verify: A @ A_pinv @ A = A
print(torch.allclose(A @ A_pinv @ A, A, atol=1e-4))  # True

# With tolerance
A_pinv = torch.linalg.pinv(A, atol=1e-5)

# For Hermitian matrices
A_sq = torch.randn(3, 3)
A_sq = A_sq + A_sq.T
A_pinv = torch.linalg.pinv(A_sq, hermitian=True)
```

### slogdet

Computes the sign and log of the absolute value of the determinant.

```python
torch.linalg.slogdet(A, *, out=None)
```

```python
A = torch.randn(3, 3)
sign, logabsdet = torch.linalg.slogdet(A)
print(sign)           # +1 or -1
print(logabsdet)      # log(|det(A)|)

# det = sign * exp(logabsdet)
det = sign * logabsdet.exp()
print(torch.allclose(det, torch.linalg.det(A)))  # True

# More numerically stable than det() for large matrices
```

### det

Computes the determinant.

```python
torch.linalg.det(A, *, out=None)
```

```python
A = torch.randn(3, 3)
d = torch.linalg.det(A)
print(d)  # scalar

# Batched
A_batch = torch.randn(10, 3, 3)
d_batch = torch.linalg.det(A_batch)
print(d_batch.shape)  # torch.Size([10])
```

### cross

Computes the cross product of two 3D vectors.

```python
torch.linalg.cross(input, other, *, dim=-1, out=None)
```

```python
a = torch.tensor([1.0, 0.0, 0.0])
b = torch.tensor([0.0, 1.0, 0.0])
c = torch.linalg.cross(a, b)
print(c)  # tensor([0., 0., 1.])

# Batched
a_batch = torch.randn(10, 3)
b_batch = torch.randn(10, 3)
c_batch = torch.linalg.cross(a_batch, b_batch)
print(c_batch.shape)  # torch.Size([10, 3])
```

### diag

Extracts or constructs a diagonal matrix.

```python
torch.linalg.diag(A, *, offset=0, out=None)
```

```python
# Extract diagonal from matrix
A = torch.randn(3, 3)
d = torch.linalg.diag(A)
print(d)  # [A[0,0], A[1,1], A[2,2]]

# Extract off-diagonal
d_upper = torch.linalg.diag(A, offset=1)
d_lower = torch.linalg.diag(A, offset=-1)

# Construct diagonal matrix from 1D tensor
d = torch.tensor([1.0, 2.0, 3.0])
D = torch.linalg.diag(d)
print(D)
# tensor([[1., 0., 0.],
#         [0., 2., 0.],
#         [0., 0., 3.]])
```

### cond

Computes the condition number of a matrix.

```python
torch.linalg.cond(A, p=None)
```

```python
A = torch.randn(3, 3)
# Default: 2-norm condition number
c = torch.linalg.cond(A)

# 1-norm condition number
c1 = torch.linalg.cond(A, p=1)

# Infinity-norm condition number
cinf = torch.linalg.cond(A, p=float('inf'))

# Frobenius condition number
cfro = torch.linalg.cond(A, p='fro')

# High condition number = ill-conditioned (near-singular)
```

---

## 4. Norms

### norm

Computes a vector or matrix norm.

```python
torch.linalg.norm(
    A,                     # Input tensor
    ord=None,              # Norm order
    dim=None,              # Dimension to compute norm over
    keepdim=False,         # Keep dimensions
    *, dtype=None, out=None,
)
```

```python
# Vector norms
v = torch.randn(5)
torch.linalg.norm(v)           # L2 norm (default)
torch.linalg.norm(v, ord=1)    # L1 norm
torch.linalg.norm(v, ord=2)    # L2 norm
torch.linalg.norm(v, ord=float('inf'))  # Max norm
torch.linalg.norm(v, ord=float('-inf')) # Min abs value

# Matrix norms
A = torch.randn(3, 4)
torch.linalg.norm(A)           # Frobenius norm (default for matrices)
torch.linalg.norm(A, ord='fro')  # Frobenius norm
torch.linalg.norm(A, ord='nuc')  # Nuclear norm
```

### vector_norm

Computes a vector norm.

```python
torch.linalg.vector_norm(
    input,
    ord=2,                 # Norm order
    dim=None,              # Dimension
    keepdim=False,         # Keep dimensions
    *, dtype=None, out=None,
)
```

```python
v = torch.randn(5)

# L2 norm (default)
n2 = torch.linalg.vector_norm(v)

# L1 norm
n1 = torch.linalg.vector_norm(v, ord=1)

# L-inf norm
ninf = torch.linalg.vector_norm(v, ord=float('inf'))

# Over specific dimension
M = torch.randn(3, 4)
row_norms = torch.linalg.vector_norm(M, dim=-1)  # torch.Size([3])
col_norms = torch.linalg.vector_norm(M, dim=-2)  # torch.Size([4])
```

### vdot

Computes the dot product of two 1D tensors (conjugating the first).

```python
torch.linalg.vdot(input, other, *, out=None)
```

```python
a = torch.tensor([1.0 + 2j, 3.0 + 4j])
b = torch.tensor([5.0 + 6j, 7.0 + 8j])
result = torch.linalg.vdot(a, b)  # conj(a) . b
```

---

## 5. Complete Examples

### Principal Component Analysis (PCA)

```python
def pca(X, n_components=2):
    """Perform PCA using torch.linalg."""
    # Center the data
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Compute covariance matrix
    n = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]

    # Project data
    X_transformed = X_centered @ components

    return X_transformed, components, explained_variance

# Usage
X = torch.randn(100, 10)  # 100 samples, 10 features
X_pca, components, variance = pca(X, n_components=2)
print(X_pca.shape)  # torch.Size([100, 2])
```

### Linear Regression via Normal Equations

```python
def linear_regression(X, y):
    """Solve linear regression using torch.linalg.solve."""
    # Add bias column
    ones = torch.ones(X.shape[0], 1)
    X_aug = torch.cat([ones, X], dim=1)

    # Normal equations: (X^T X) beta = X^T y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    beta = torch.linalg.solve(XtX, Xty)

    return beta

# Usage
X = torch.randn(100, 3)
y = X @ torch.tensor([1.0, 2.0, 3.0]) + 0.5 + torch.randn(100) * 0.1
beta = linear_regression(X, y)
print(beta)  # [bias, w1, w2, w3] close to [0.5, 1.0, 2.0, 3.0]
```

### Low-Rank Approximation

```python
def low_rank_approximation(A, rank):
    """Compute best rank-k approximation using SVD."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Truncate to desired rank
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]

    # Reconstruct
    A_k = U_k @ torch.diag(S_k) @ Vh_k

    # Compute approximation error
    error = torch.linalg.matrix_norm(A - A_k, ord='fro')
    total = torch.linalg.matrix_norm(A, ord='fro')
    relative_error = error / total

    return A_k, relative_error

A = torch.randn(100, 100)
A_approx, err = low_rank_approximation(A, rank=10)
print(f"Relative error: {err:.4f}")
```

### Matrix Square Root (via Eigendecomposition)

```python
def matrix_sqrt(A):
    """Compute matrix square root for SPD matrices."""
    eigenvalues, eigenvectors = torch.linalg.eigh(A)

    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = eigenvalues.clamp(min=0)

    # sqrt(A) = V @ diag(sqrt(lambda)) @ V^T
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    A_sqrt = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

    return A_sqrt

A = torch.randn(5, 5)
A = A @ A.T + torch.eye(5)  # SPD matrix
A_sqrt = matrix_sqrt(A)
print(torch.allclose(A_sqrt @ A_sqrt, A, atol=1e-4))  # True
```

### Mahalanobis Distance

```python
def mahalanobis_distance(x, mean, cov):
    """Compute Mahalanobis distance."""
    L = torch.linalg.cholesky(cov)
    diff = x - mean
    # Solve L @ y = diff for y
    y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
    return (y ** 2).sum(dim=-1).sqrt()

# Usage
mean = torch.zeros(3)
cov = torch.eye(3) * 2
x = torch.randn(10, 3)
distances = mahalanobis_distance(x, mean, cov)
```
