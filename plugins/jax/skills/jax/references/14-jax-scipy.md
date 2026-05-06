# JAX SciPy Module (jax.scipy)

## Table of Contents

- [1. Overview](#1-overview)
- [2. jax.scipy.cluster](#2-jaxscipycluster)
- [3. jax.scipy.fft](#3-jaxscipyfft)
- [4. jax.scipy.linalg](#4-jaxscipylinalg)
- [5. jax.scipy.ndimage](#5-jaxscipyndimage)
- [6. jax.scipy.optimize](#6-jaxscipyoptimize)
- [7. jax.scipy.signal](#7-jaxscipysignal)
- [8. jax.scipy.sparse](#8-jaxscipysparse)
- [9. jax.scipy.spatial](#9-jaxscipyspatial)
- [10. jax.scipy.special](#10-jaxscipyspecial)
- [11. jax.scipy.stats](#11-jaxscipystats)

---

## 1. Overview

The `jax.scipy` module provides a JAX-compatible subset of SciPy's functionality. These implementations are designed to be:

- **JIT-compatible**: All functions work within `jax.jit` compiled code
- **Differentiable**: Most functions support automatic differentiation via `jax.grad`
- **Batchable**: Compatible with `jax.vmap` for vectorized operations
- **Accelerator-ready**: Execute on GPU and TPU backends

Not all SciPy functions are implemented; JAX provides the most commonly used functions from each submodule. Functions that are missing will raise an `AttributeError`.

```python
import jax
import jax.numpy as jnp
import jax.scipy as jsp

# All jax.scipy functions are JIT-compatible
@jax.jit
def compute_svd(x):
    return jsp.linalg.svd(x)

# They are differentiable
grad_fn = jax.grad(lambda x: jsp.linalg.det(x).sum())
x = jnp.eye(3) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (3, 3))
g = grad_fn(x)

# They work on GPU/TPU
with jax.devices('gpu')[0]:
    result = jsp.special.gammaln(jnp.array([1.0, 2.0, 5.0]))
```

---

## 2. jax.scipy.cluster

### 2.1 vq Module

The `jax.scipy.cluster.vq` module provides vector quantization and k-means clustering routines.

#### whiten

**Signature:** `jax.scipy.cluster.vq.whiten(obs, check_finite=True)`

Normalizes each feature (column) to have unit variance. This is a common preprocessing step before k-means.

```python
import jax
import jax.numpy as jnp
import jax.scipy.cluster.vq as vq

# Observed data: 5 samples, 3 features
obs = jnp.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0],
])

whitened = vq.whiten(obs)
# Each column now has unit variance
print(jnp.std(whitened, axis=0))  # ~[1.0, 1.0, 1.0]
```

#### kmeans2

**Signature:** `jax.scipy.cluster.vq.kmeans2(data, k, iter=10, thresh=1e-5, minit='points', missing='warn')`

Classifies data into k clusters using the k-means algorithm.

```python
import jax
import jax.numpy as jnp
import jax.scipy.cluster.vq as vq

key = jax.random.PRNGKey(0)

# Generate clustered data
key1, key2, key = jax.random.split(key, 3)
cluster1 = jax.random.normal(key1, (100, 2)) + jnp.array([5.0, 5.0])
cluster2 = jax.random.normal(key2, (100, 2)) + jnp.array([-5.0, -5.0])
data = jnp.concatenate([cluster1, cluster2])

# Run k-means
centroids, labels = vq.kmeans2(data, k=2, iter=20)
print(f"Centroids:\n{centroids}")
print(f"Labels shape: {labels.shape}")  # (200,)
```

#### kmeans

**Signature:** `jax.scipy.cluster.vq.kmeans(obs, k_or_guess, iter=20, thresh=1e-5)`

Performs k-means on a set of observation vectors and returns the centroids.

```python
key = jax.random.PRNGKey(42)
data = jax.random.normal(key, (500, 3))

centroids, distortion = vq.kmeans(data, k=5)
print(f"Centroids shape: {centroids.shape}")  # (5, 3)
print(f"Distortion: {distortion}")
```

---

## 3. jax.scipy.fft

The `jax.scipy.fft` module provides Fourier transform routines that extend `jnp.fft` with additional transforms.

### 3.1 Basic FFT Functions

#### fft / ifft

**Signature:**
```python
jax.scipy.fft.fft(x, n=None, axis=-1, norm=None)
jax.scipy.fft.ifft(x, n=None, axis=-1, norm=None)
```

Compute the 1-dimensional discrete Fourier Transform and its inverse.

```python
import jax
import jax.numpy as jnp
import jax.scipy.fft as jsp_fft

# Simple signal: combination of two frequencies
t = jnp.linspace(0, 1, 128, endpoint=False)
signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.5 * jnp.sin(2 * jnp.pi * 20 * t)

# Forward FFT
spectrum = jsp_fft.fft(signal)
frequencies = jnp.fft.fftfreq(128, d=1.0/128)

# Power spectrum
power = jnp.abs(spectrum) ** 2

# Inverse FFT (reconstruct signal)
reconstructed = jsp_fft.ifft(spectrum)
assert jnp.allclose(signal, reconstructed.real, atol=1e-5)

# Find dominant frequencies
top_freq_idx = jnp.argsort(power)[-4:]  # Top 4 (positive and negative)
print(f"Dominant frequencies: {frequencies[top_freq_idx]}")
```

#### fft2 / ifft2 / fftn / ifftn

**Signature:**
```python
jax.scipy.fft.fft2(x, s=None, axes=(-2, -1), norm=None)
jax.scipy.fft.ifft2(x, s=None, axes=(-2, -1), norm=None)
jax.scipy.fft.fftn(x, s=None, axes=None, norm=None)
jax.scipy.fft.ifftn(x, s=None, axes=None, norm=None)
```

2-D and N-dimensional FFTs.

```python
# 2D FFT for image processing
image = jax.random.normal(jax.random.PRNGKey(0), (64, 64))

# 2D FFT
freq_image = jsp_fft.fft2(image)

# Apply low-pass filter (keep only low frequencies)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = jnp.zeros((rows, cols))
radius = 10
Y, X = jnp.ogrid[:rows, :cols]
mask = ((Y - crow)**2 + (X - ccol)**2 <= radius**2).astype(jnp.float32)

# Shift, apply mask, shift back
freq_shifted = jnp.fft.fftshift(freq_image)
filtered = freq_shifted * mask
filtered_shifted = jnp.fft.ifftshift(filtered)

# Inverse FFT to get filtered image
filtered_image = jsp_fft.ifft2(filtered_shifted).real

# N-dimensional FFT
data_3d = jax.random.normal(jax.random.PRNGKey(1), (16, 16, 16))
freq_3d = jsp_fft.fftn(data_3d)
reconstructed_3d = jsp_fft.ifftn(freq_3d).real
```

### 3.2 Discrete Cosine and Sine Transforms

#### dct / idct

**Signature:**
```python
jax.scipy.fft.dct(x, type=2, n=None, axis=-1, norm=None)
jax.scipy.fft.idct(x, type=2, n=None, axis=-1, norm=None)
```

Discrete Cosine Transform and its inverse. Types 1, 2, 3, 4 are supported.

```python
# DCT for signal compression
signal = jnp.sin(jnp.linspace(0, 4 * jnp.pi, 128))

# DCT Type-II (most common)
dct_coeffs = jsp_fft.dct(signal, type=2, norm='ortho')

# Compression: keep only top-k coefficients
k = 20
top_k_idx = jnp.argsort(jnp.abs(dct_coeffs))[-k:]
compressed = jnp.zeros_like(dct_coeffs)
compressed = compressed.at[top_k_idx].set(dct_coeffs[top_k_idx])

# Reconstruct
reconstructed = jsp_fft.idct(compressed, type=2, norm='ortho')
```

#### dst / idst

**Signature:**
```python
jax.scipy.fft.dst(x, type=2, n=None, axis=-1, norm=None)
jax.scipy.fft.idst(x, type=2, n=None, axis=-1, norm=None)
```

Discrete Sine Transform and its inverse.

```python
signal = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 64))

# Forward DST
dst_coeffs = jsp_fft.dst(signal, type=2, norm='ortho')

# Inverse DST
reconstructed = jsp_fft.idst(dst_coeffs, type=2, norm='ortho')
assert jnp.allclose(signal, reconstructed, atol=1e-5)
```

### 3.3 Hartley and Hermitian FFT

#### hfft / ihfft

**Signature:**
```python
jax.scipy.fft.hfft(x, n=None, axis=-1, norm=None)
jax.scipy.fft.ihfft(x, n=None, axis=-1, norm=None)
```

Hartley FFT and its inverse.

```python
# Hartley transform of a real signal
signal = jnp.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0])
hartley = jsp_fft.hfft(signal)
reconstructed = jsp_fft.ihfft(hartley)
```

---

## 4. jax.scipy.linalg

The `jax.scipy.linalg` module provides linear algebra routines beyond `jnp.linalg`, including matrix decompositions, solvers, and matrix functions computations.

### 4.1 Linear System Solvers

#### solve

**Signature:** `jax.scipy.linalg.solve(a, b, lower=False, assume_a='gen', check_finite=True)`

Solves the linear equation `a @ x = b` for x.

```python
import jax
import jax.numpy as jnp
import jax.scipy.linalg as la

# Solve Ax = b
A = jnp.array([[3.0, 1.0], [1.0, 2.0]])
b = jnp.array([9.0, 8.0])

x = la.solve(A, b)
# Verify: A @ x should equal b
print(jnp.allclose(A @ x, b))  # True

# Solve multiple right-hand sides
B = jnp.array([[9.0, 1.0], [8.0, 2.0]])
X = la.solve(A, B)
print(X.shape)  # (2, 2)
```

#### solve_triangular

**Signature:** `jax.scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, check_finite=True)`

Solves a triangular linear system efficiently.

```python
# Create a triangular system
L = jnp.array([[2.0, 0.0, 0.0],
               [1.0, 3.0, 0.0],
               [4.0, 2.0, 1.0]])
b = jnp.array([4.0, 7.0, 12.0])

# Solve Lx = b
x = la.solve_triangular(L, b, lower=True)
print(jnp.allclose(L @ x, b))  # True

# Solve L^T x = b
x_t = la.solve_triangular(L, b, lower=True, trans=1)
print(jnp.allclose(L.T @ x_t, b))  # True
```

### 4.2 Matrix Decompositions

#### lu / lu_solve

**Signature:**
```python
jax.scipy.linalg.lu(a, permute_l=False, check_finite=True)
jax.scipy.linalg.lu_solve(lu_and_piv, b, trans=0, check_finite=True)
```

LU decomposition factors a matrix as `P @ L @ U`.

```python
A = jnp.array([[2.0, 1.0, 1.0],
               [4.0, 3.0, 3.0],
               [8.0, 7.0, 9.0]])

# LU decomposition
P, L, U = la.lu(A)
print(f"Permutation:\n{P}")
print(f"Lower:\n{L}")
print(f"Upper:\n{U}")
print(jnp.allclose(P @ L @ U, A))  # True

# Solve using LU
lu_piv = la.lu(A)
# lu_piv returns (P, L, U); use with lu_solve
b = jnp.array([1.0, 1.0, 1.0])
```

#### qr

**Signature:** `jax.scipy.linalg.qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, check_finite=True)`

QR decomposition factors a matrix into an orthogonal matrix Q and upper triangular matrix R.

```python
A = jnp.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 10.0]])

# Full QR decomposition
Q, R = la.qr(A)
print(jnp.allclose(Q @ R, A))     # True
print(jnp.allclose(Q.T @ Q, jnp.eye(3), atol=1e-6))  # True (orthogonal)

# Reduced QR
Q_r, R_r = la.qr(A, mode='economic')
print(Q_r.shape)  # (3, 3)
```

#### cholesky / cho_solve

**Signature:**
```python
jax.scipy.linalg.cholesky(a, lower=False, overwrite_a=False, check_finite=True)
jax.scipy.linalg.cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True)
```

Cholesky decomposition for positive definite matrices: `A = L @ L.T`.

```python
# Create a positive definite matrix
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (4, 4))
A = A @ A.T + jnp.eye(4) * 5  # Make positive definite

# Cholesky decomposition
L = la.cholesky(A, lower=True)
print(jnp.allclose(L @ L.T, A))  # True

# Solve A @ x = b using Cholesky
b = jnp.array([1.0, 2.0, 3.0, 4.0])
x = la.cho_solve((L, True), b)
print(jnp.allclose(A @ x, b))  # True
```

### 4.3 Eigenvalue Decompositions

#### eig / eigh / eigvals / eigvalsh

**Signature:**
```python
jax.scipy.linalg.eig(a, b=None, left=False, right=True)
jax.scipy.linalg.eigh(a, b=None, lower=True, eigvals_only=False)
jax.scipy.linalg.eigvals(a, b=None)
jax.scipy.linalg.eigvalsh(a, b=None, lower=True)
```

- `eig`: General eigenvalues for non-symmetric matrices (complex output)
- `eigh`: Eigenvalues for symmetric/Hermitian matrices (real output, ordered)
- `eigvals`: Eigenvalues only (non-symmetric)
- `eigvalsh`: Eigenvalues only (symmetric)

```python
# Symmetric matrix eigenvalue decomposition
A_sym = jnp.array([[4.0, 1.0, 0.0],
                    [1.0, 3.0, 1.0],
                    [0.0, 1.0, 2.0]])

eigenvalues, eigenvectors = la.eigh(A_sym)
print(f"Eigenvalues: {eigenvalues}")  # Sorted ascending
print(f"Eigenvectors shape: {eigenvectors.shape}")

# Verify: A @ v = lambda * v
for i in range(3):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    assert jnp.allclose(A_sym @ v, lam * v, atol=1e-5)

# Just eigenvalues
vals = la.eigvalsh(A_sym)
print(jnp.allclose(vals, eigenvalues))  # True

# Non-symmetric matrix
A_gen = jnp.array([[0.0, -1.0],
                    [1.0,  0.0]])
w, v = la.eig(A_gen)
print(f"Complex eigenvalues: {w}")  # [0+1j, 0-1j] (rotation matrix)
```

### 4.4 Singular Value Decomposition

#### svd

**Signature:** `jax.scipy.linalg.svd(a, full_matrices=True, compute_uv=True, check_finite=True)`

```python
A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Full SVD: A = U @ diag(s) @ Vh
U, s, Vh = la.svd(A, full_matrices=True)
print(f"U shape: {U.shape}")     # (3, 3)
print(f"s shape: {s.shape}")     # (2,)
print(f"Vh shape: {Vh.shape}")   # (2, 2)

# Reconstruct
A_reconstructed = U[:, :2] @ jnp.diag(s) @ Vh
print(jnp.allclose(A_reconstructed, A))  # True

# Reduced SVD
U_r, s_r, Vh_r = la.svd(A, full_matrices=False)
print(f"U_r shape: {U_r.shape}")  # (3, 2)

# Low-rank approximation
rank = 1
A_approx = U_r[:, :rank] @ jnp.diag(s_r[:rank]) @ Vh_r[:rank, :]
print(f"Rank-{rank} approximation error: {jnp.linalg.norm(A - A_approx):.4f}")

# Pseudoinverse via SVD
def pinv_svd(A, tol=1e-8):
    U, s, Vh = la.svd(A, full_matrices=False)
    s_inv = jnp.where(s > tol, 1.0 / s, 0.0)
    return Vh.T @ jnp.diag(s_inv) @ U.T
```

### 4.5 Matrix Functions Computations

#### expm / logm / sqrtm / signm

**Signature:**
```python
jax.scipy.linalg.expm(A)
jax.scipy.linalg.logm(A)
jax.scipy.linalg.sqrtm(A)
jax.scipy.linalg.signm(A)
```

Matrix functions computed via eigendecomposition or Padé approximation.

```python
# Matrix exponential (useful for solving ODEs, Lie groups)
A = jnp.array([[0.0, -1.0],
               [1.0,  0.0]])
exp_A = la.expm(A)
# expm of a skew-symmetric matrix is a rotation matrix
print(jnp.allclose(exp_A, jnp.array([[-1., 0.], [0., 1.]]) * jnp.cos(1) +
                    jnp.array([[0., -1.], [1., 0.]]) * jnp.sin(1)))

# Matrix logarithm (inverse of expm)
log_exp_A = la.logm(exp_A)
print(jnp.allclose(log_exp_A, A, atol=1e-5))

# Matrix square root
B = jnp.array([[4.0, 2.0], [2.0, 3.0]])
sqrt_B = la.sqrtm(B)
print(jnp.allclose(sqrt_B @ sqrt_B, B, atol=1e-5))

# Matrix sign function (used in eigenvalue algorithms)
C = jnp.array([[1.0, 2.0], [3.0, 4.0]])
sign_C = la.signm(C)
```

#### cosm / sinm / tanm

```python
# Matrix cosine and sine
A = jnp.array([[0.5, 0.2], [0.1, 0.3]])
cos_A = la.cosm(A)
sin_A = la.sinm(A)

# Verify Euler's formula for matrices: expm(iA) = cos(A) + i*sin(A)
# For real matrices, verify: expm(A) != cos(A) + sin(A) generally
print(f"cos(A) + sin(A):\n{cos_A + sin_A}")
print(f"expm(A):\n{la.expm(A)}")
```

#### funm / fractional_matrix_power / khatri_rao

```python
# funm: Apply arbitrary function to matrix eigenvalues
A = jnp.array([[1.0, 0.5], [0.0, 2.0]])
# Apply exp element-wise to eigenvalues
exp_eigenvalues_A = la.funm(A, jnp.exp)

# Fractional matrix power
A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
A_half = la.fractional_matrix_power(A, 0.5)  # Same as sqrtm
A_third = la.fractional_matrix_power(A, 1.0/3.0)

# Khatri-Rao product (column-wise Kronecker product)
A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
kr = la.khatri_rao(A, B)
print(f"Khatri-Rao shape: {kr.shape}")  # (4, 2)
```

### 4.6 Other Matrix Operations

#### inv / pinv / det / norm

```python
import jax.scipy.linalg as la

A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

# Matrix inverse
A_inv = la.inv(A)
print(jnp.allclose(A @ A_inv, jnp.eye(2)))  # True

# Pseudoinverse (works for non-square and singular matrices)
A_tall = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
A_pinv = la.pinv(A_tall)
print(A_pinv.shape)  # (2, 3)

# Determinant
det_A = la.det(A)
print(f"det(A) = {det_A}")  # -2.0

# Matrix norms
frobenius = la.norm(A, ord='fro')
nuclear = la.norm(A, ord='nuc')
spectral = la.norm(A, ord=2)
```

#### Special Matrix Construction

```python
import jax.scipy.linalg as la

# Block diagonal matrix
blocks = [jnp.ones((2, 2)), 2 * jnp.ones((3, 3)), 3 * jnp.ones((2, 2))]
bd = la.block_diag(*blocks)
print(f"Block diagonal shape: {bd.shape}")  # (7, 7)

# Circulant matrix
c = jnp.array([1.0, 2.0, 3.0, 4.0])
C = la.circulant(c)
# [[1 4 3 2], [2 1 4 3], [3 2 1 4], [4 3 2 1]]

# Toeplitz matrix
col = jnp.array([1.0, 2.0, 3.0, 4.0])
row = jnp.array([1.0, 5.0, 6.0])
T = la.toeplitz(col, row)

# Hankel matrix
H = la.hankel(jnp.array([1.0, 2.0, 3.0, 4.0]),
               jnp.array([4.0, 5.0, 6.0]))

# Hadamard matrix
n = 4
H = la.hadamard(n)
print(jnp.allclose(H @ H.T, n * jnp.eye(n)))  # True

# Companion matrix
companion = la.companion(jnp.array([1.0, -10.0, 31.0, -30.0]))
```

#### lstsq / schur / solve_sylvester

```python
# Least-squares solution
A = jnp.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
b = jnp.array([1.0, 2.0, 2.0])
x, residuals, rank, sv = la.lstsq(A, b)
print(f"Least-squares solution: {x}")

# Sylvester equation solver: A @ X + X @ B = Q
A = jnp.array([[1.0, 2.0], [0.0, 4.0]])
B = jnp.array([[5.0, -6.0], [0.0, 7.0]])
Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
X = la.solve_sylvester(A, B, Q)
# Verify: A @ X + X @ B ~= Q
print(jnp.allclose(A @ X + X @ B, Q, atol=1e-5))
```

---

## 5. jax.scipy.ndimage

The `jax.scipy.ndimage` module provides n-dimensional image processing functions.

### 5.1 Coordinate Mapping

#### map_coordinates

**Signature:** `jax.scipy.ndimage.map_coordinates(input, coordinates, order=1, mode='constant', cval=0.0)`

Maps input to new coordinates by interpolation at the specified coordinates.

```python
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as ndi

# Create a simple 2D array
data = jnp.arange(25.0).reshape(5, 5)

# Sample at fractional coordinates (bilinear interpolation)
coords = jnp.array([[1.5, 2.0, 3.5],  # row coordinates
                    [1.0, 2.5, 3.0]])  # col coordinates
result = ndi.map_coordinates(data, coords, order=1)
print(f"Interpolated values: {result}")

# 3D example
data_3d = jnp.arange(64.0).reshape(4, 4, 4)
coords_3d = jnp.array([[1.5, 2.0], [1.0, 2.5], [2.0, 1.0]])
result_3d = ndi.map_coordinates(data_3d, coords_3d, order=1)

# Boundary handling modes
for mode in ['constant', 'nearest', 'wrap', 'reflect', 'mirror']:
    r = ndi.map_coordinates(data, coords, order=1, mode=mode)
    print(f"{mode}: {r}")
```

### 5.2 Labeling and Measurements

#### label

**Signature:** `jax.scipy.ndimage.label(input, structure=None)`

Labels connected components in a binary array.

```python
# Binary image with connected components
binary = jnp.array([
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],
])

labeled, num_features = ndi.label(binary)
print(f"Number of features: {num_features}")  # 3
print(f"Labeled array:\n{labeled}")
```

#### sum / mean / variance / minimum / maximum

**Signature:** `jax.scipy.ndimage.sum(input, labels=None, index=None)` (similarly for others)

Compute statistics for labeled regions.

```python
data = jnp.array([
    [10.0, 20.0, 30.0, 40.0],
    [50.0, 60.0, 70.0, 80.0],
    [90.0, 100.0, 110.0, 120.0],
])
labels = jnp.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [2, 2, 2, 2],
])

# Statistics per label
sums = ndi.sum(data, labels, index=jnp.array([0, 1, 2]))
means = ndi.mean(data, labels, index=jnp.array([0, 1, 2]))
variances = ndi.variance(data, labels, index=jnp.array([0, 1, 2]))
mins = ndi.minimum(data, labels, index=jnp.array([0, 1, 2]))
maxs = ndi.maximum(data, labels, index=jnp.array([0, 1, 2]))

print(f"Region sums: {sums}")
print(f"Region means: {means}")
print(f"Region variances: {variances}")
print(f"Region mins: {mins}")
print(f"Region maxs: {maxs}")
```

### 5.3 Filtering

#### gaussian_filter

**Signature:** `jax.scipy.ndimage.gaussian_filter(input, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0)`

Multidimensional Gaussian filter.

```python
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as ndi

key = jax.random.PRNGKey(0)
image = jax.random.normal(key, (64, 64))

# Apply Gaussian blur with different sigma values
smoothed_1 = ndi.gaussian_filter(image, sigma=1.0)
smoothed_3 = ndi.gaussian_filter(image, sigma=3.0)
smoothed_5 = ndi.gaussian_filter(image, sigma=5.0)

# Anisotropic filtering (different sigma per axis)
smoothed_aniso = ndi.gaussian_filter(image, sigma=(1.0, 5.0))

# Gaussian derivative (order > 0)
# order=1: first derivative, order=2: second derivative (Laplacian of Gaussian)
grad_x = ndi.gaussian_filter(image, sigma=2.0, order=[1, 0])
grad_y = ndi.gaussian_filter(image, sigma=2.0, order=[0, 1])
laplacian = ndi.gaussian_filter(image, sigma=2.0, order=[2, 2])
```

#### uniform_filter / median_filter

```python
# Uniform (box) filter
box_filtered = ndi.uniform_filter(image, size=5)

# Median filter (good for salt-and-pepper noise removal)
# Note: JAX's median_filter may have limited window support
noisy = image + jax.random.normal(key, image.shape) * 0.5
median_filtered = ndi.median_filter(noisy, size=3)
```

#### sobel / prewitt

```python
# Edge detection with Sobel filter
edges_x = ndi.sobel(image, axis=0)
edges_y = ndi.sobel(image, axis=1)
edge_magnitude = jnp.sqrt(edges_x**2 + edges_y**2)

# Prewitt filter (alternative edge detector)
prewitt_x = ndi.prewitt(image, axis=0)
prewitt_y = ndi.prewitt(image, axis=1)
prewitt_magnitude = jnp.sqrt(prewitt_x**2 + prewitt_y**2)
```

### 5.4 Morphological Operations

#### distance_transform_edt

**Signature:** `jax.scipy.ndimage.distance_transform_edt(input, sampling=None, return_distances=True, return_indices=False)`

Exact Euclidean distance transform for binary inputs.

```python
# Binary mask
mask = jnp.zeros((10, 10), dtype=bool)
mask = mask.at[5, 5].set(True)

# Distance transform
distances = ndi.distance_transform_edt(mask)
print(f"Distance at corner: {distances[0, 0]:.2f}")  # ~7.07
print(f"Distance at center neighbor: {distances[4, 5]:.2f}")  # 1.0
```

#### binary_fill_holes / binary_erosion / binary_dilation

```python
# Create a hollow square
binary = jnp.zeros((10, 10), dtype=bool)
binary = binary.at[2:8, 2].set(True)
binary = binary.at[2:8, 7].set(True)
binary = binary.at[2, 2:8].set(True)
binary = binary.at[7, 2:8].set(True)

# Fill holes
filled = ndi.binary_fill_holes(binary)

# Erosion (shrink objects)
eroded = ndi.binary_erosion(binary)

# Dilation (grow objects)
dilated = ndi.binary_dilation(binary)

# Opening = erosion followed by dilation
opened = ndi.binary_dilation(ndi.binary_erosion(binary))

# Closing = dilation followed by erosion
closed = ndi.binary_erosion(ndi.binary_dilation(binary))
```

---

## 6. jax.scipy.optimize

The `jax.scipy.optimize` module provides optimization routines.

### 6.1 minimize

**Signature:**
```python
jax.scipy.optimize.minimize(
    fun, x0, args=(), method=None, jac=None, tol=None,
    callback=None, options=None
)
```

Minimize a scalar function using one of several methods.

```python
import jax
import jax.numpy as jnp
import jax.scipy.optimize as opt

# Quadratic function
def quadratic(x):
    return jnp.sum((x - 3.0) ** 2)

# Minimize starting from origin
result = opt.minimize(quadratic, jnp.array([0.0, 0.0]), method='BFGS')
print(f"Minimum at: {result.x}")      # ~[3.0, 3.0]
print(f"Function value: {result.fun}")  # ~0.0
print(f"Success: {result.success}")
print(f"Iterations: {result.nit}")

# With gradient provided
grad_quadratic = jax.grad(quadratic)
result = opt.minimize(
    quadratic,
    jnp.array([0.0, 0.0]),
    method='BFGS',
    jac=grad_quadratic
)

# Rosenbrock function (classic optimization benchmark)
def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

result = opt.minimize(
    rosenbrock,
    jnp.zeros(5),
    method='BFGS',
    jac=jax.grad(rosenbrock)
)
print(f"Rosenbrock minimum near: {result.x}")  # All ~1.0

# Non-linear least squares (curve fitting)
def residual(params, x_data, y_data):
    a, b, c = params
    return y_data - (a * x_data**2 + b * x_data + c)

key = jax.random.PRNGKey(0)
x_data = jnp.linspace(-2, 2, 50)
true_params = jnp.array([2.0, -1.0, 0.5])
y_data = true_params[0] * x_data**2 + true_params[1] * x_data + true_params[2]
y_data += 0.1 * jax.random.normal(key, x_data.shape)

def least_squares_loss(params):
    return jnp.sum(residual(params, x_data, y_data)**2)

result = opt.minimize(
    least_squares_loss,
    jnp.array([0.0, 0.0, 0.0]),
    method='BFGS',
    jac=jax.grad(least_squares_loss)
)
print(f"Fitted params: {result.x}")  # ~[2.0, -1.0, 0.5]
```

### 6.2 minimize_scalar

**Signature:** `jax.scipy.optimize.minimize_scalar(fun, bracket=None, bounds=None, args=(), method='brent', tol=None, options=None)`

Minimize a scalar function of one variable.

```python
# Minimize a 1D function
def f(x):
    return (x - 2.0) ** 2 + 1.0

result = opt.minimize_scalar(f, bracket=(0.0, 5.0))
print(f"Minimum at x={result.x}, f(x)={result.fun}")

# Bounded minimization
result_bounded = opt.minimize_scalar(f, bounds=(0.0, 1.5), method='bounded')
print(f"Bounded minimum at x={result_bounded.x}")
```

### 6.3 line_search / approx_fprime / check_grad

```python
# Line search: find optimal step size along a direction
def f(x):
    return jnp.sum(x ** 2)

x0 = jnp.array([5.0, 5.0])
direction = jnp.array([-1.0, -1.0])

result = opt.line_search(f, jax.grad(f), x0, direction)
print(f"Step size: {result.step_size}")
print(f"Function evaluations: {result.nfev}")

# Approximate gradient using finite differences
def f_scalar(x):
    return x[0]**2 + x[1]**3

grad_approx = opt.approx_fprime(jnp.array([1.0, 2.0]), f_scalar, epsilon=1e-6)
grad_exact = jax.grad(f_scalar)(jnp.array([1.0, 2.0]))
print(f"Approx gradient: {grad_approx}")
print(f"Exact gradient: {grad_exact}")

# Check gradient (compares analytical vs numerical)
def f_with_grad(x):
    return jnp.sum(x ** 2), 2.0 * x

err = opt.check_grad(
    lambda x: jnp.sum(x ** 2),
    jax.grad(lambda x: jnp.sum(x ** 2)),
    jnp.array([1.0, 2.0])
)
```

---

## 7. jax.scipy.signal

The `jax.scipy.signal` module provides signal processing functions.

### 7.1 Convolution and Correlation

#### fftconvolve / convolve / correlate

```python
import jax
import jax.numpy as jnp
import jax.scipy.signal as sig

# 1D convolution via FFT (efficient for large kernels)
signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
kernel = jnp.array([0.25, 0.5, 0.25])

# FFT-based convolution
result_fft = sig.fftconvolve(signal, kernel, mode='same')
print(f"FFT convolve: {result_fft}")

# Direct convolution
result_direct = sig.convolve(signal, kernel, mode='same')
print(jnp.allclose(result_fft, result_direct))  # True

# Cross-correlation
result_corr = sig.correlate(signal, kernel, mode='same')

# 2D convolution and correlation
image = jnp.ones((8, 8))
kernel_2d = jnp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Laplacian

result_2d = sig.convolve2d(image, kernel_2d, mode='same')
result_corr_2d = sig.correlate2d(image, kernel_2d, mode='same')

# Median filtering
noisy_1d = jnp.array([1.0, 2.0, 100.0, 3.0, 4.0, 5.0, 200.0, 6.0])
filtered_1d = sig.medfilt(noisy_1d, kernel_size=3)
print(f"Median filtered: {filtered_1d}")

# 2D median filter
noisy_2d = jax.random.normal(jax.random.PRNGKey(0), (16, 16))
filtered_2d = sig.medfilt2d(noisy_2d, kernel_size=3)
```

### 7.2 Spectral Analysis

#### stft / istft

**Signature:**
```python
jax.scipy.signal.stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, boundary='zeros', padded=True, axis=-1)
jax.scipy.signal.istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
```

Short-Time Fourier Transform and its inverse.

```python
key = jax.random.PRNGKey(0)
fs = 1000  # Sampling frequency
t = jnp.arange(0, 1.0, 1.0/fs)

# Chirp signal (frequency increases over time)
signal = jnp.sin(2 * jnp.pi * (50 + 200 * t) * t)

# Compute STFT
frequencies, times, Zxx = sig.stft(
    signal, fs=fs, nperseg=128, noverlap=64
)
print(f"STFT shape: {Zxx.shape}")  # (n_freqs, n_times)

# Magnitude spectrogram
magnitude = jnp.abs(Zxx)

# Reconstruct signal from STFT
_, reconstructed = sig.istft(Zxx, fs=fs, nperseg=128, noverlap=64)

# Spectrogram (convenience function)
f_spec, t_spec, Sxx = sig.spectrogram(
    signal, fs=fs, nperseg=128, noverlap=64
)
print(f"Spectrogram shape: {Sxx.shape}")
```

#### csd / welch

```python
# Cross spectral density
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (1000,))
y = jax.random.normal(key2, (1000,))

f_csd, Pxy = sig.csd(x, y, fs=1000, nperseg=256)
print(f"CSD shape: {Pxy.shape}")

# Power spectral density using Welch's method
f_welch, Pxx = sig.welch(x, fs=1000, nperseg=256)
print(f"Welch PSD shape: {Pxx.shape}")
```

---

## 8. jax.scipy.sparse

The `jax.scipy.sparse` module provides sparse matrix formats and operations.

### 8.1 Sparse Matrix Formats

```python
import jax
import jax.numpy as jnp
import jax.scipy.sparse as sp

# Create a sparse matrix from dense
dense = jnp.array([
    [1.0, 0.0, 0.0, 2.0],
    [0.0, 0.0, 3.0, 0.0],
    [0.0, 4.0, 0.0, 5.0],
])

# COO (Coordinate) format
coo = sp.coo_matrix(dense)
print(f"COO data: {coo.data}")
print(f"COO row: {coo.row}")
print(f"COO col: {coo.col}")

# CSR (Compressed Sparse Row) format
csr = sp.csr_matrix(dense)
print(f"CSR data: {csr.data}")
print(f"CSR indices: {csr.indices}")
print(f"CSR indptr: {csr.indptr}")

# CSC (Compressed Sparse Column) format
csc = sp.csc_matrix(dense)

# BSR (Block Sparse Row) format
bsr = sp.bsr_matrix(dense, blocksize=(1, 1))

# Check if sparse
print(sp.issparse(coo))   # True
print(sp.issparse(dense))  # False

# Convert back to dense
dense_back = coo.todense()
print(jnp.allclose(dense, dense_back))  # True
```

### 8.2 Save and Load

```python
# Save sparse matrix to disk
# sp.save_npz('sparse_matrix.npz', coo)

# Load sparse matrix from disk
# loaded = sp.load_npz('sparse_matrix.npz')

# Sparse matrix-vector multiplication
dense = jax.random.normal(jax.random.PRNGKey(0), (100, 50))
dense = jnp.where(jnp.abs(dense) > 1.5, dense, 0.0)
sparse = sp.csr_matrix(dense)
vec = jax.random.normal(jax.random.PRNGKey(1), (50,))

# Matrix-vector product
result_sparse = sparse @ vec
result_dense = dense @ vec
print(jnp.allclose(result_sparse, result_dense, atol=1e-5))  # True
```

---

## 9. jax.scipy.spatial

### 9.1 Distance Computations

#### cdist / pdist / squareform

**Signature:**
```python
jax.scipy.spatial.distance.cdist(XA, XB, metric='euclidean')
jax.scipy.spatial.distance.pdist(X, metric='euclidean')
jax.scipy.spatial.distance.squareform(X)
```

```python
import jax
import jax.numpy as jnp
import jax.scipy.spatial.distance as dist

# Pairwise distances between two sets of points
points_a = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
points_b = jnp.array([[1.0, 1.0], [2.0, 2.0]])

# All pairwise distances between A and B
d_ab = dist.cdist(points_a, points_b, metric='euclidean')
print(f"A-B distances shape: {d_ab.shape}")  # (3, 2)

# Pairwise distances within a single set
d_self = dist.pdist(points_a, metric='euclidean')
print(f"Self distances shape: {d_self.shape}")  # (3,) - condensed form

# Convert to square form
d_square = dist.squareform(d_self)
print(f"Square form shape: {d_square.shape}")  # (3, 3)

# Other metrics
d_cosine = dist.cdist(points_a, points_b, metric='cosine')
d_cityblock = dist.cdist(points_a, points_b, metric='cityblock')
d_minkowski = dist.cdist(points_a, points_b, metric='minkowski')
```

### 9.2 Spatial Transforms

#### Rotation

```python
from jax.scipy.spatial.transform import Rotation

# Create rotation from Euler angles
r = Rotation.from_euler('xyz', jnp.array([0.0, 0.0, jnp.pi / 4]))

# Apply rotation to vectors
vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
rotated = r.apply(vectors)
print(f"Rotated vectors:\n{rotated}")

# Get rotation matrix
rot_matrix = r.as_matrix()
print(f"Rotation matrix:\n{rot_matrix}")

# Create rotation from quaternion
quat = jnp.array([0.0, 0.0, jnp.sin(jnp.pi/8), jnp.cos(jnp.pi/8)])
r_quat = Rotation.from_quat(quat)

# Compose rotations
r1 = Rotation.from_euler('z', jnp.array(jnp.pi / 6))
r2 = Rotation.from_euler('x', jnp.array(jnp.pi / 4))
r_combined = r1 * r2
```

---

## 10. jax.scipy.special

The `jax.scipy.special` module provides special mathematical functions.

### 10.1 Gamma and Beta Functions

```python
import jax
import jax.numpy as jnp
import jax.scipy.special as special

x = jnp.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

# Log of gamma function (numerically stable)
lgamma = special.gammaln(x)
print(f"gammaln: {lgamma}")

# Log of beta function
a, b = jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 3.0, 4.0])
lbeta = special.betaln(a, b)
print(f"betaln: {lbeta}")

# Digamma (psi) function
psi = special.digamma(x)
print(f"digamma: {psi}")

# Polygamma function
pg = special.polygamma(1, x)  # n=1 (trigamma)
print(f"polygamma(1, x): {pg}")

# Regularized incomplete gamma functions
gammainc = special.gammainc(1.0, x)
gammaincc = special.gammaincc(1.0, x)
print(f"gammainc: {gammainc}")
print(f"gammaincc: {gammaincc}")

# Beta function
beta_val = special.beta(2.0, 3.0)
print(f"beta(2, 3) = {beta_val}")

# Regularized incomplete beta function
betainc = special.betainc(0.5, 0.5, jnp.array([0.1, 0.5, 0.9]))
print(f"betainc: {betainc}")
```

### 10.2 Error Functions

```python
x = jnp.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])

# Error function
erf_vals = special.erf(x)
print(f"erf: {erf_vals}")

# Complementary error function
erfc_vals = special.erfc(x)
print(f"erfc: {erfc_vals}")
print(jnp.allclose(erf_vals + erfc_vals, 1.0))  # True

# Inverse error function
erfinv_vals = special.erfinv(erf_vals)
print(jnp.allclose(erfinv_vals, x))  # True
```

### 10.3 Logistic and Softmax Functions

```python
x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])

# Expit (sigmoid): 1 / (1 + exp(-x))
expit_vals = special.expit(x)
print(f"expit: {expit_vals}")

# Logit: log(p / (1 - p))
logit_vals = special.logit(expit_vals)
print(jnp.allclose(logit_vals, x, atol=1e-5))  # True

# logsumexp (numerically stable)
x_2d = jnp.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
lse = special.logsumexp(x_2d, axis=-1)
print(f"logsumexp: {lse}")

# logsumexp with keepdims
lse_keep = special.logsumexp(x_2d, axis=-1, keepdims=True)
print(f"Shape with keepdims: {lse_keep.shape}")

# softmax and log_softmax (same as jax.nn versions)
sm = special.softmax(x)
lsm = special.log_softmax(x)
print(f"softmax: {sm}")
```

### 10.4 Bessel Functions

```python
x = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0])

# Modified Bessel functions of the first kind
i0_vals = special.i0(x)    # order 0
i0e_vals = special.i0e(x)  # exponentially scaled
i1_vals = special.i1(x)    # order 1
i1e_vals = special.i1e(x)  # exponentially scaled

print(f"i0: {i0_vals}")
print(f"i0e: {i0e_vals}")

# Bessel functions of the first kind
j0_vals = special.j0(x)  # order 0
j1_vals = special.j1(x)  # order 1
jv_vals = special.jv(2.5, x)  # fractional order

print(f"j0: {j0_vals}")
print(f"jv(2.5): {jv_vals}")

# Bessel functions of the second kind
y0_vals = special.y0(x)  # order 0
y1_vals = special.y1(x)  # order 1
yv_vals = special.yv(2.5, x)  # fractional order

print(f"y0: {y0_vals}")
print(f"yv(2.5): {yv_vals}")
```

### 10.5 Information Theory Functions

```python
# Entropy
p = jnp.array([0.3, 0.5, 0.2])
entropy = special.entr(p)
print(f"Entropy: {entropy}")

# Relative entropy (KL divergence component)
p = jnp.array([0.3, 0.5, 0.2])
q = jnp.array([0.25, 0.25, 0.5])
rel_entr_vals = special.rel_entr(p, q)
print(f"Relative entropy components: {rel_entr_vals}")
kl_divergence = jnp.sum(rel_entr_vals)
print(f"KL divergence: {kl_divergence}")

# KL divergence
kl_vals = special.kl_div(p, q)
print(f"KL divergence components: {kl_vals}")
```

### 10.6 Combinatorial Functions

```python
# Factorial (log-space)
n = jnp.array([0, 1, 5, 10, 20])
log_fact = special.gammaln(n + 1)  # log(n!)
print(f"log factorial: {log_fact}")

# Combinations: C(n, k)
n_val, k_val = 10, 3
comb_val = special.gammaln(n_val + 1) - special.gammaln(k_val + 1) - special.gammaln(n_val - k_val + 1)
print(f"C(10, 3) = {jnp.exp(comb_val)}")

# Using comb function if available
comb_direct = special.comb(10, 3, exact=False)
print(f"C(10, 3) direct: {comb_direct}")

# Permutations: P(n, k) = n! / (n-k)!
perm_val = special.gammaln(n_val + 1) - special.gammaln(n_val - k_val + 1)
print(f"P(10, 3) = {jnp.exp(perm_val)}")

# Using perm function
perm_direct = special.perm(10, 3, exact=False)
print(f"P(10, 3) direct: {perm_direct}")

# Factorial
fact_5 = special.factorial(5)
print(f"5! = {fact_5}")
```

---

## 11. jax.scipy.stats

The `jax.scipy.stats` module provides statistical distributions and hypothesis tests.

### 11.1 Continuous Distributions

Each distribution provides at minimum `logpdf`, `pdf`, `cdf`, `ppf`, and `sf` (survival function) methods.

#### Normal Distribution

```python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

# Normal distribution
x = jnp.linspace(-5, 5, 100)
mu, sigma = 0.0, 1.0

# Probability density function
pdf_vals = stats.norm.pdf(x, loc=mu, scale=sigma)

# Log probability density
logpdf_vals = stats.norm.logpdf(x, loc=mu, scale=sigma)

# Cumulative distribution function
cdf_vals = stats.norm.cdf(x, loc=mu, scale=sigma)

# Survival function: 1 - cdf
sf_vals = stats.norm.sf(x, loc=mu, scale=sigma)

# Percent point function (inverse CDF)
quantiles = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99])
ppf_vals = stats.norm.ppf(quantiles, loc=mu, scale=sigma)
print(f"Quantiles: {ppf_vals}")  # ~[-2.33, -0.67, 0, 0.67, 2.33]

# Log-likelihood of data under normal distribution
data = jax.random.normal(jax.random.PRNGKey(0), (1000,))
ll = jnp.sum(stats.norm.logpdf(data, loc=0.0, scale=1.0))
print(f"Log-likelihood: {ll}")
```

#### Uniform Distribution

```python
x = jnp.linspace(-1, 2, 100)

pdf = stats.uniform.pdf(x, loc=0.0, scale=1.0)  # U(0, 1)
logpdf = stats.uniform.logpdf(x, loc=0.0, scale=1.0)
cdf = stats.uniform.cdf(x, loc=0.0, scale=1.0)
ppf = stats.uniform.ppf(jnp.array([0.0, 0.5, 1.0]), loc=0.0, scale=1.0)
```

#### Exponential Distribution

```python
x = jnp.linspace(0, 5, 100)
lam = 2.0  # rate parameter

pdf = stats.expon.pdf(x, scale=1.0/lam)
logpdf = stats.expon.logpdf(x, scale=1.0/lam)
cdf = stats.expon.cdf(x, scale=1.0/lam)
```

#### Poisson Distribution

```python
k = jnp.arange(10)
mu = 3.0

pmf = stats.poisson.pmf(k, mu)
logpmf = stats.poisson.logpmf(k, mu)
cdf = stats.poisson.cdf(k, mu)
```

#### Beta Distribution

```python
x = jnp.linspace(0, 1, 100)
a, b = 2.0, 5.0

pdf = stats.beta.pdf(x, a, b)
logpdf = stats.beta.logpdf(x, a, b)
cdf = stats.beta.cdf(x, a, b)
```

#### Gamma Distribution

```python
x = jnp.linspace(0.01, 10, 100)
a, b = 2.0, 1.0  # shape and rate

pdf = stats.gamma.pdf(x, a, scale=1.0/b)
logpdf = stats.gamma.logpdf(x, a, scale=1.0/b)
cdf = stats.gamma.cdf(x, a, scale=1.0/b)
```

#### Student's t Distribution

```python
x = jnp.linspace(-5, 5, 100)
df = 10.0

pdf = stats.t.pdf(x, df)
logpdf = stats.t.logpdf(x, df)
cdf = stats.t.cdf(x, df)
```

#### Chi-squared Distribution

```python
x = jnp.linspace(0.01, 20, 100)
df = 5.0

pdf = stats.chi2.pdf(x, df)
logpdf = stats.chi2.logpdf(x, df)
cdf = stats.chi2.cdf(x, df)
```

#### F Distribution

```python
x = jnp.linspace(0.01, 5, 100)
dfn, dfd = 5.0, 20.0

pdf = stats.f.pdf(x, dfn, dfd)
cdf = stats.f.cdf(x, dfn, dfd)
```

### 11.2 Discrete Distributions

#### Binomial Distribution

```python
k = jnp.arange(20)
n, p = 20, 0.3

pmf = stats.binom.pmf(k, n, p)
logpmf = stats.binom.logpmf(k, n, p)
cdf = stats.binom.cdf(k, n, p)
```

#### Bernoulli Distribution

```python
k = jnp.array([0, 1])
p = 0.7

pmf = stats.bernoulli.pmf(k, p)
logpmf = stats.bernoulli.logpmf(k, p)
```

### 11.3 Multivariate Distributions

#### Multivariate Normal

```python
x = jnp.array([1.0, 2.0])
mean = jnp.array([0.0, 0.0])
cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])

pdf = stats.multivariate_normal.pdf(x, mean, cov)
logpdf = stats.multivariate_normal.logpdf(x, mean, cov)

# Batch evaluation
x_batch = jnp.array([[1.0, 2.0], [0.0, 0.0], [-1.0, 1.0]])
pdf_batch = stats.multivariate_normal.pdf(x_batch, mean, cov)
logpdf_batch = stats.multivariate_normal.logpdf(x_batch, mean, cov)

# Maximum likelihood estimation of mean and covariance
data = jax.random.multivariate_normal(
    jax.random.PRNGKey(0),
    mean=jnp.zeros(3),
    cov=jnp.eye(3),
    shape=(1000,)
)
mle_mean = jnp.mean(data, axis=0)
mle_cov = jnp.cov(data, rowvar=False)
print(f"MLE mean: {mle_mean}")
print(f"MLE cov diagonal: {jnp.diag(mle_cov)}")
```

#### Dirichlet Distribution

```python
alpha = jnp.array([2.0, 5.0, 3.0])

# Sample from Dirichlet
key = jax.random.PRNGKey(0)
samples = jax.random.dirichlet(key, alpha, shape=(1000,))
print(f"Sample mean: {jnp.mean(samples, axis=0)}")  # ~alpha/sum(alpha)

# Log PDF
x = jnp.array([0.2, 0.5, 0.3])
logpdf = stats.dirichlet.logpdf(x, alpha)
pdf = stats.dirichlet.pdf(x, alpha)
```

### 11.4 Hypothesis Tests

#### ks_1samp / ks_2samp

```python
import jax.scipy.stats as stats

# One-sample Kolmogorov-Smirnov test
key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (100,))

# Test if data comes from N(0, 1)
statistic, pvalue = stats.ks_1samp(data, stats.norm.cdf, args=(0.0, 1.0))
print(f"KS 1-sample statistic: {statistic}")
print(f"KS 1-sample p-value: {pvalue}")

# Two-sample Kolmogorov-Smirnov test
key1, key2 = jax.random.split(jax.random.PRNGKey(1))
sample1 = jax.random.normal(key1, (100,))
sample2 = jax.random.normal(key2, (100,)) + 0.5

statistic, pvalue = stats.ks_2samp(sample1, sample2)
print(f"KS 2-sample statistic: {statistic}")
print(f"KS 2-sample p-value: {pvalue}")
```

### 11.5 Practical Statistics Example

```python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

def fit_gaussian_mixture(data, n_components=2, n_iter=100):
    """Simple Gaussian Mixture Model using EM algorithm."""
    key = jax.random.PRNGKey(42)
    n_samples = data.shape[0]

    # Initialize parameters
    responsibilities = jax.random.uniform(key, (n_samples, n_components))
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

    for _ in range(n_iter):
        # M-step
        nk = responsibilities.sum(axis=0)
        means = (responsibilities.T @ data) / nk[:, None]
        variances = jnp.stack([
            jnp.sum(responsibilities[:, k:k+1] * (data - means[k])**2, axis=0) / nk[k]
            for k in range(n_components)
        ])
        weights = nk / n_samples

        # E-step
        log_resp = jnp.stack([
            jnp.log(weights[k]) + stats.norm.logpdf(data, means[k], jnp.sqrt(variances[k]))
            for k in range(n_components)
        ], axis=-1)
        log_resp = log_resp - jax.nn.log_softmax(log_resp, axis=-1)
        responsibilities = jnp.exp(log_resp)

    return weights, means, variances, responsibilities

# Generate mixture data
key = jax.random.PRNGKey(0)
k1, k2, key = jax.random.split(key, 3)
data1 = jax.random.normal(k1, (300, 2)) + jnp.array([3.0, 3.0])
data2 = jax.random.normal(k2, (200, 2)) + jnp.array([-2.0, -2.0])
data = jnp.concatenate([data1, data2])

weights, means, variances, resp = fit_gaussian_mixture(data, n_components=2)
print(f"Mixture weights: {weights}")
print(f"Component means: {means}")
```
