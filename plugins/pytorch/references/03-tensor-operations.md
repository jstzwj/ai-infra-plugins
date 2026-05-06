# PyTorch - Chapter 3: Tensor Operations

This reference covers all tensor mathematical, reduction, comparison, BLAS/LAPACK, and other operations.

---

## 3.1 Pointwise Operations

### Arithmetic

```python
torch.add(input, other, *, alpha=1, out=None)
# input + alpha * other

torch.sub(input, other, *, alpha=1, out=None)
torch.subtract(input, other, *, alpha=1, out=None)
# input - alpha * other

torch.mul(input, other, *, out=None)
torch.multiply(input, other, *, out=None)
# Element-wise multiplication

torch.div(input, other, *, rounding_mode=None, out=None)
torch.divide(input, other, *, rounding_mode=None, out=None)
# Element-wise division
# rounding_mode: None (true div), 'trunc' (round toward zero), 'floor'

torch.true_divide(input, other)  # Always true division
torch.floor_divide(input, other)  # Floor division

torch.remainder(input, other)  # Python-style modulo
torch.fmod(input, other)       # C-style modulo
torch.mod(input, other)        # Alias for remainder

torch.neg(input)               # Negation
torch.negative(input)          # Same as neg

torch.abs(input)               # Absolute value
torch.absolute(input)          # Same as abs

torch.sign(input)              # -1, 0, or 1
torch.sgn(input)               # Complex-aware sign

torch.pow(input, exponent)     # Power
torch.square(input)            # input ** 2
torch.sqrt(input)              # Square root
torch.rsqrt(input)             # 1/sqrt(input)
torch.cbrt(input)              # Cube root
torch.reciprocal(input)        # 1/input

torch.exp(input)               # e^input
torch.exp2(input)              # 2^input
torch.expm1(input)             # e^input - 1 (more precise for small values)
torch.log(input)               # Natural log
torch.log2(input)              # Base-2 log
torch.log10(input)             # Base-10 log
torch.log1p(input)             # log(1 + input) (more precise for small values)

torch.logaddexp(input, other)  # log(exp(input) + exp(other))
torch.logaddexp2(input, other) # log2(2^input + 2^other)
torch.xlogy(input, other)      # input * log(other), returns 0 when input=0
torch.xlog1py(input, other)    # input * log1p(other)
```

### Rounding

```python
torch.ceil(input)              # Round up
torch.floor(input)             # Round down
torch.round(input, decimals=0) # Round to nearest
torch.trunc(input)             # Truncate toward zero
torch.frac(input)              # Fractional part
torch.clamp(input, min=None, max=None)  # Clip values
torch.clip(input, min=None, max=None)   # Alias for clamp
```

### Trigonometric

```python
torch.sin(input)               # Sine
torch.cos(input)               # Cosine
torch.tan(input)               # Tangent

torch.asin(input)              # Arc sine
torch.acos(input)              # Arc cosine
torch.atan(input)              # Arc tangent
torch.atan2(input, other)      # Arc tangent of input/other

torch.sinh(input)              # Hyperbolic sine
torch.cosh(input)              # Hyperbolic cosine
torch.tanh(input)              # Hyperbolic tangent

torch.asinh(input)             # Inverse hyperbolic sine
torch.acosh(input)             # Inverse hyperbolic cosine
torch.atanh(input)             # Inverse hyperbolic tangent

torch.sinc(input)              # sin(pi*input) / (pi*input)

torch.deg2rad(input)           # Degrees to radians
torch.rad2deg(input)           # Radians to degrees

torch.hypot(input, other)      # sqrt(input^2 + other^2)
```

### Activation Functions

```python
torch.relu(input)              # max(0, input)
torch.relu6(input)             # min(max(0, input), 6)
torch.sigmoid(input)           # 1 / (1 + exp(-input))
torch.logsigmoid(input)        # log(sigmoid(input)) = -log(1+exp(-input))

torch.tanh(input)              # Hyperbolic tangent

torch.elu(input, alpha=1.0)    # max(0,x) + min(0, alpha*(exp(x)-1))
torch.selu(input)              # Scaled ELU (self-normalizing)
torch.celu(input, alpha=1.0)   # Continuously-differentiable ELU

torch.gelu(input, approximate='none')  # Gaussian Error Linear Unit
torch.silu(input)              # x * sigmoid(x) (Swish)
torch.mish(input)              # x * tanh(softplus(x))
torch.hardswish(input)         # x * relu6(x+3) / 6
torch.hardsigmoid(input)       # relu6(x+3) / 6
torch.hardtanh(input, min_val=-1.0, max_val=1.0)

torch.leaky_relu(input, negative_slope=0.01)
torch.prelu(input, weight)     # Parametric ReLU

torch.softplus(input, beta=1, threshold=20)  # 1/beta * log(1 + exp(beta*input))
torch.softsign(input)          # input / (1 + |input|)
torch.softmin(input, dim)
torch.softmax(input, dim)
torch.log_softmax(input, dim)
```

### Bitwise and Logical

```python
torch.bitwise_and(input, other)
torch.bitwise_or(input, other)
torch.bitwise_xor(input, other)
torch.bitwise_not(input)

torch.logical_and(input, other)
torch.logical_or(input, other)
torch.logical_not(input)
torch.logical_xor(input, other)

# Shift operations (integer tensors only)
torch.bitwise_left_shift(input, other)
torch.bitwise_right_shift(input, other)
```

### Comparison (Element-wise)

```python
torch.eq(input, other)     # Equal (returns bool tensor)
torch.ne(input, other)     # Not equal
torch.gt(input, other)     # Greater than
torch.ge(input, other)     # Greater than or equal
torch.lt(input, other)     # Less than
torch.le(input, other)     # Less than or equal

torch.isclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False)
torch.equal(input, other)  # Returns True if identical (single Python bool)
torch.allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False)

torch.isnan(input)         # NaN check
torch.isinf(input)         # Inf check
torch.isfinite(input)      # Neither NaN nor Inf
torch.isneginf(input)      # Negative infinity
torch.isposinf(input)      # Positive infinity
torch.isreal(input)        # Real check
```

### Special Math

```python
torch.erf(input)               # Error function
torch.erfc(input)              # Complementary error function
torch.erfinv(input)            # Inverse error function

torch.lgamma(input)            # Log of gamma function
torch.digamma(input)           # Digamma function
torch.mvlgamma(input, p)       # Multivariate log-gamma
torch.polygamma(n, input)      # Polygamma function

torch.igamma(input, other)     # Lower incomplete gamma
torch.igammac(input, other)    # Upper incomplete gamma

torch.i0(input)                # Modified Bessel function of order 0

torch.sigmoid(input)
torch.logit(input, eps=None)   # log(p / (1-p))
torch.expit(input)             # 1 / (1 + exp(-input)) = sigmoid

torch.angle(input)             # Phase angle of complex tensor
torch.conj(input)              # Complex conjugate
torch.conj_physical(input)     # Physical conjugate
torch.resolve_conj(input)      # Resolve conjugate
torch.resolve_neg(input)       # Resolve negation

torch.real(input)              # Real part of complex tensor
torch.imag(input)              # Imaginary part of complex tensor

torch.view_as_real(input)      # Complex → (..., 2) real
torch.view_as_complex(input)   # (..., 2) real → complex

torch.nan_to_num(input, nan=0.0, posinf=None, neginf=None)
```

### Lerp / Misc

```python
torch.lerp(input, end, weight)  # Linear interpolation
# result = input + weight * (end - input)

torch.addcdiv(input, tensor1, tensor2, value=1)
# input + value * (tensor1 / tensor2)

torch.addcmul(input, tensor1, tensor2, value=1)
# input + value * (tensor1 * tensor2)

torch.madd(input, other, alpha=1, beta=1)
# beta * input + alpha * other

torch.nextafter(input, other)   # Next floating-point value
torch.heaviside(input, values)  # Heaviside step function
torch.gradient(input, dim)      # Numerical gradient
torch.diff(input, n=1, dim=-1)  # n-th discrete difference
```

---

## 3.2 Reduction Operations

### sum / prod

```python
torch.sum(input, dim=None, keepdim=False, *, dtype=None)
torch.prod(input, dim=None, keepdim=False, *, dtype=None)

t = torch.randn(3, 4)
torch.sum(t)           # Scalar sum of all elements
torch.sum(t, dim=0)    # Sum along rows → (4,)
torch.sum(t, dim=1)    # Sum along columns → (3,)
torch.sum(t, dim=1, keepdim=True)  # (3, 1)
```

### mean / median / mode

```python
torch.mean(input, dim=None, keepdim=False, *, dtype=None)
torch.median(input, dim=None, keepdim=False)
torch.mode(input, dim=None, keepdim=False)

# median and mode return namedtuple (values, indices)
result = torch.median(t, dim=1)
result.values  # Median values
result.indices # Indices of median values
```

### var / std

```python
torch.var(input, dim=None, unbiased=True, keepdim=False, *, correction=1)
torch.std(input, dim=None, unbiased=True, keepdim=False, *, correction=1)
torch.var_mean(input, dim=None, unbiased=True, keepdim=False)
torch.std_mean(input, dim=None, unbiased=True, keepdim=False)

# correction=1 for unbiased (Bessel's), correction=0 for biased
t = torch.randn(10)
torch.var(t)              # Unbiased variance (ddof=1)
torch.var(t, correction=0) # Biased variance (ddof=0)
```

### norm

```python
torch.norm(input, p=2, dim=None, keepdim=False, dtype=None)
# p: order of norm. Default 2 (L2 norm)
# p='fro' for Frobenius norm, p='nuc' for nuclear norm
# p=1 for L1, p=2 for L2, p=float('inf') for max abs
```

### argmax / argmin

```python
torch.argmax(input, dim=None, keepdim=False)
torch.argmin(input, dim=None, keepdim=False)

t = torch.randn(3, 4)
torch.argmax(t)          # Flat index of max element
torch.argmax(t, dim=0)   # Column-wise argmax → (4,)
torch.argmax(t, dim=1)   # Row-wise argmax → (3,)
```

### max / min

```python
torch.max(input)                          # Scalar max
torch.max(input, dim)                     # Returns (values, indices)
torch.max(input, other)                   # Element-wise max of two tensors
torch.max(input, other, *, out=None)      # With output tensor

torch.min(input)                          # Scalar min
torch.min(input, dim)                     # Returns (values, indices)
torch.min(input, other)                   # Element-wise min

t = torch.randn(3, 4)
result = torch.max(t, dim=1)
result.values  # Max values per row
result.indices # Indices of max values per row
```

### amax / amin / aminmax

```python
torch.amax(input, dim=None, keepdim=False)  # Like max but no indices
torch.amin(input, dim=None, keepdim=False)  # Like min but no indices
torch.aminmax(input, dim=None, keepdim=False)  # Returns (min, max)
```

### topk / sort

```python
torch.topk(input, k, dim=None, largest=True, sorted=True)
# Returns (values, indices) of top-k elements

torch.sort(input, dim=-1, descending=False)
# Returns (values, indices)

torch.argsort(input, dim=-1, descending=False)
# Returns indices that would sort

torch.kthvalue(input, k, dim=None, keepdim=False)
# Returns (values, indices) of k-th smallest element

t = torch.tensor([3, 1, 4, 1, 5, 9])
values, indices = torch.topk(t, 3)    # tensor([9, 5, 4]), tensor([5, 4, 2])
values, indices = torch.sort(t)        # tensor([1, 1, 3, 4, 5, 9])
```

### all / any / count_nonzero

```python
torch.all(input)                   # True if all elements are nonzero
torch.all(input, dim, keepdim=False)
torch.any(input)                   # True if any element is nonzero
torch.any(input, dim, keepdim=False)
torch.count_nonzero(input, dim=None)
```

### Other Reductions

```python
torch.logsumexp(input, dim, keepdim=False)  # log(sum(exp(x))) - numerically stable
torch.dist(input, other, p=2)              # p-norm of (input - other)
torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)
torch.unique_consecutive(input, return_inverse=False, return_counts=False, dim=None)
torch.quantile(input, q, dim=None, keepdim=False)
torch.nanquantile(input, q, dim=None, keepdim=False)
torch.nansum(input, dim=None, keepdim=False)
torch.nanmean(input, dim=None, keepdim=False)
torch.cumsum(input, dim)
torch.cumprod(input, dim)
torch.cummax(input, dim)   # Returns (values, indices)
torch.cummin(input, dim)   # Returns (values, indices)
```

---

## 3.3 BLAS and LAPACK Operations

### Matrix Multiplication

```python
torch.mm(input, mat2)          # Matrix-matrix product (2D only)
torch.bmm(input, mat2)         # Batch matrix-matrix product (3D only)
torch.matmul(input, other)     # General matrix product (broadcasting)
torch.baddbmm(input, batch1, batch2, *, alpha=1, beta=1)
# beta*input + alpha*(batch1 @ batch2)

# Chain matrix multiplication (optimal order)
torch.linalg.multi_dot(tensors)
```

### Matrix-Vector

```python
torch.mv(input, vec)           # Matrix-vector product
torch.addmv(input, mat, vec, *, beta=1, alpha=1)
# beta*input + alpha*(mat @ vec)
```

### Outer Product

```python
torch.outer(input, vec2)       # Outer product of two 1D tensors
torch.ger(input, vec2)         # Alias for outer (deprecated)
torch.addr(input, vec1, vec2, *, beta=1, alpha=1)
# beta*input + alpha*(vec1 outer vec2)
```

### Inner Product

```python
torch.dot(input, tensor)       # Inner product of two 1D tensors
torch.inner(input, other)      # General inner product (contracts last dim)
```

### Matrix Operations (via torch.linalg)

```python
torch.linalg.det(A)            # Determinant
torch.linalg.slogdet(A)        # Sign and log of determinant
torch.linalg.inv(A)            # Matrix inverse
torch.linalg.pinv(A)           # Pseudo-inverse
torch.linalg.matrix_exp(A)     # Matrix exponential
torch.linalg.matrix_power(A, n) # Matrix power

torch.linalg.svd(A, full_matrices=True)     # SVD
torch.linalg.svdvals(A)                     # Singular values only
torch.linalg.eig(A)                         # Eigenvalue decomposition
torch.linalg.eigh(A, UPLO='L')              # Hermitian eigenvalue decomposition
torch.linalg.eigvals(A)                     # Eigenvalues only
torch.linalg.eigvalsh(A, UPLO='L')          # Hermitian eigenvalues

torch.linalg.cholesky(A, upper=False)       # Cholesky decomposition
torch.linalg.qr(A, mode='reduced')          # QR decomposition
torch.linalg.lu_factor(A)                   # LU factorization
torch.linalg.solve(A, B)                    # Solve linear system Ax=B
torch.linalg.lstsq(A, B, rcond=None)        # Least squares solution

torch.linalg.norm(A)                        # Matrix/vector norm
torch.linalg.matrix_norm(A)                 # Matrix norm
torch.linalg.vector_norm(A)                 # Vector norm
torch.linalg.cond(A)                        # Condition number
torch.linalg.matrix_rank(A)                 # Matrix rank

torch.linalg.cross(input, other, dim=-1)    # Cross product
torch.linalg.multi_dot(tensors)             # Optimal chain matmul
torch.linalg.householder_product(A, tau)    # Householder product

# Low-rank approximations
torch.svd_lowrank(A, q)                     # Randomized SVD
torch.pca_lowrank(A, q, center=True, niter=2) # PCA via randomized SVD
```

---

## 3.4 Other Useful Operations

### Comparison and Selection

```python
torch.sort(input, dim=-1, descending=False)
torch.topk(input, k, dim=-1, largest=True, sorted=True)
torch.argsort(input, dim=-1, descending=False)
torch.kthvalue(input, k, dim=-1, keepdim=False)
torch.mode(input, dim=-1, keepdim=False)
torch.bucketize(input, boundaries, out_int32=False, right=False)
torch.searchsorted(sorted_sequence, values, right=False)
torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)
```

### Scatter/Gather (Advanced)

```python
torch.scatter(input, dim, index, src)
torch.scatter_add(input, dim, index, src)
torch.scatter_reduce(input, dim, index, src, reduce)
torch.gather(input, dim, index)
torch.index_add(input, dim, index, source)
torch.index_copy(input, dim, index, source)
torch.index_reduce(input, dim, index, source, reduce)
torch.index_select(input, dim, index)
torch.masked_select(input, mask)
torch.take(input, index)
torch.take_along_dim(input, indices, dim)
```

### Cropping and Padding

```python
torch.nn.functional.pad(input, pad, mode='constant', value=0)
# pad: (left, right, top, bottom) for 2D
# mode: 'constant', 'reflect', 'replicate', 'circular'
```

### Histogram

```python
torch.histc(input, bins=100, min=0, max=0)
torch.histogram(input, bins, range=None, weight=None, density=False)
torch.bincount(input, weights=None, minlength=0)
```

### Miscellaneous

```python
torch.cdist(x1, x2, p=2.0)                    # Pairwise distance
torch.pdist(input, p=2.0)                      # Pairwise distance within set
torch.pairwise_distance(x1, x2, p=2.0, eps=1e-6)
torch.cosine_similarity(x1, x2, dim=1, eps=1e-8)
torch.triu(input, diagonal=0)                  # Upper triangular
torch.tril(input, diagonal=0)                  # Lower triangular
torch.trace(input)                             # Sum of diagonal
torch.diag(input, diagonal=0)                  # Diagonal or construct diagonal matrix
torch.diagflat(input, offset=0)                # Create diagonal matrix from flat input
torch.diagonal(input, offset=0, dim1=0, dim2=1) # Extract diagonal
torch.flip(input, dims)                        # Reverse along dimensions
torch.fliplr(input)                            # Flip left-right (2D)
torch.flipud(input)                            # Flip up-down (2D)
torch.roll(input, shifts, dims)                # Roll along dimensions
torch.rot90(input, k=1, dims=(0, 1))           # Rotate 90 degrees
torch.meshgrid(*tensors, indexing='ij')         # Create coordinate grids
torch.cartesian_prod(*tensors)                  # Cartesian product
torch.combinations(input, r=2, with_replacement=False)
torch.block_diag(*tensors)                      # Block diagonal matrix
torch.vander(x, N=None, increasing=False)       # Vandermonde matrix
```
