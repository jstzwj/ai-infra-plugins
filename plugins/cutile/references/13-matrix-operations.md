# Matrix Operations

## Overview

Matrix operations are at the heart of high-performance computing workloads, particularly in deep learning and scientific computing. cuTile provides comprehensive support for matrix operations through its `ct.mma()` and `ct.matmul()` functions, which map directly to GPU tensor core operations for maximum performance.

This chapter covers:
- **Matrix Multiply-Accumulate (MMA)**: The fundamental tensor core operation
- **Matrix Multiplication (MATMUL)**: A convenience wrapper for matrix multiplication
- **Performance considerations**: Tile sizing, data types, and hardware utilization
- **Complete examples**: From basic matmul to fused operations with activation functions

## ct.mma() — Matrix Multiply-Accumulate

### Syntax

```python
ct.mma(a, b, c) -> Tile
```

### Description

Performs matrix multiply-accumulate operation: `a × b + c`. This is the fundamental operation that maps directly to GPU tensor core hardware instructions on modern NVIDIA GPUs (Volta architecture and later).

### Parameters

| Parameter | Type | Description | Shape |
|-----------|------|-------------|-------|
| `a` | `Tile` | First matrix operand | (M, K) |
| `b` | `Tile` | Second matrix operand | (K, N) |
| `c` | `Tile` | Accumulator tile | (M, N) |

### Returns

- **Type**: `Tile`
- **Shape**: (M, N)
- **Description**: Result of `a × b + c`

### Data Type Support

The `ct.mma()` operation supports various data type combinations, with specific requirements for tensor core utilization:

| Input A/B Type | Accumulator Type | Tensor Core Support | Notes |
|----------------|------------------|---------------------|-------|
| `float16` | `float32` | Yes (FP16 MMA) | Most common configuration |
| `bfloat16` | `float32` | Yes (BF16 MMA) | Ampere+ architecture |
| `float32` | `float32` | Yes (TF32 MMA) | Ampere+ architecture |
| `int8` | `int32` | Yes (INT8 MMA) | Turing+ architecture |
| `int4` | `int32` | Yes (INT4 MMA) | Hopper+ architecture |

### Key Characteristics

1. **Accumulator Precision**: Even when using FP16 or BF16 inputs, the accumulator should typically be `float32` to maintain numerical stability and prevent overflow during accumulation.

2. **Tensor Core Mapping**: The operation directly maps to hardware MMA instructions:
   - `HMMA` (Half-precision Matrix Multiply-Accumulate) for FP16
   - `IMMA` (Integer Matrix Multiply-Accumulate) for INT8/INT4
   - `BF16 MMA` for BFloat16

3. **Alignment Requirements**: Optimal performance requires proper memory alignment. Tiles should be aligned to 128-byte boundaries when possible.

4. **Tile Size Constraints**: For best tensor core utilization, tile dimensions should be multiples of 16 (the fundamental tensor core operation size).

### Example: Basic MatMul Kernel

Here's a complete matrix multiplication kernel using `ct.mma()`:

```python
import cutile as ct

@ct.kernel
def matmul(
    X: ct.Buffer,
    Y: ct.Buffer,
    Out: ct.Buffer,
    TM: ct.Constant[int],
    TN: ct.Constant[int],
    TK: ct.Constant[int]
):
    """
    Matrix multiplication: Out = X @ Y
    
    Args:
        X: Input matrix of shape (M, K)
        Y: Input matrix of shape (K, N)
        Out: Output matrix of shape (M, N)
        TM: Tile size for M dimension
        TN: Tile size for N dimension
        TK: Tile size for K dimension
    """
    # Get block indices
    i = ct.bid(0)  # Block index along M
    j = ct.bid(1)  # Block index along N
    
    # Create tiled views with zero-padding for boundary handling
    x_view = X.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    y_view = Y.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    
    # Initialize accumulator to zero
    acc = ct.zeros((TM, TN), ct.float32)
    
    # Iterate over K dimension tiles
    for k in range(x_view.num_tiles(1)):
        # Load tiles
        tx = x_view.load((i, k))
        ty = y_view.load((k, j))
        
        # Perform matrix multiply-accumulate
        acc = ct.mma(tx, ty, acc)
    
    # Store result with type conversion
    ct.store(Out, (i, j), acc.astype(Out.dtype))
```

To launch this kernel:

```python
import torch

# Create input matrices
M, K, N = 1024, 1024, 1024
X = torch.randn(M, K, dtype=torch.float16, device='cuda')
Y = torch.randn(K, N, dtype=torch.float16, device='cuda')
Out = torch.empty(M, N, dtype=torch.float16, device='cuda')

# Configure tile sizes (should be multiples of 16 for tensor cores)
TM, TN, TK = 128, 128, 32

# Launch kernel
grid = (M // TM, N // TN)
matmul[grid](X, Y, Out, TM, TN, TK)
```

### Example: FP16 MatMul with FP32 Accumulation

This is the most common configuration for deep learning workloads:

```python
@ct.kernel
def matmul_fp16_fp32(
    A: ct.Buffer,  # (M, K) float16
    B: ct.Buffer,  # (K, N) float16
    C: ct.Buffer,  # (M, N) float16
):
    """
    FP16 matrix multiply with FP32 accumulation.
    
    This configuration provides:
    - Reduced memory bandwidth (FP16 inputs)
    - High tensor core throughput (FP16 MMA)
    - Numerical stability (FP32 accumulation)
    """
    TM, TN, TK = 128, 128, 32
    
    i = ct.bid(0)
    j = ct.bid(1)
    
    # Load FP16 tiles
    a_tile = A.load((i, ct.tid()), shape=(TM, TK))
    b_tile = B.load((ct.tid(), j), shape=(TK, TN))
    
    # FP32 accumulator
    c_tile = ct.zeros((TM, TN), ct.float32)
    
    # MMA: FP16 x FP16 -> FP32
    c_tile = ct.mma(a_tile, b_tile, c_tile)
    
    # Convert back to FP16 and store
    C.store((i, j), c_tile.astype(ct.float16))
```

### Example: Batched Matrix Multiplication

For batched operations (3D tensors):

```python
@ct.kernel
def batched_matmul(
    A: ct.Buffer,  # (Batch, M, K)
    B: ct.Buffer,  # (Batch, K, N)
    C: ct.Buffer,  # (Batch, M, N)
    TM: ct.Constant[int],
    TN: ct.Constant[int],
    TK: ct.Constant[int]
):
    """
    Batched matrix multiplication.
    Each batch element is processed independently.
    """
    batch = ct.bid(0)
    i = ct.bid(1)
    j = ct.bid(2)
    
    # Create views for the current batch
    a_view = A[batch].tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    b_view = B[batch].tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    
    acc = ct.zeros((TM, TN), ct.float32)
    
    for k in range(a_view.num_tiles(1)):
        tx = a_view.load((i, k))
        ty = b_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    
    C[batch].store((i, j), acc.astype(C.dtype))
```

## ct.matmul() — Matrix Multiplication

### Syntax

```python
ct.matmul(a, b) -> Tile
```

### Description

Performs matrix multiplication: `a × b`. This is a convenience wrapper around `ct.mma()` that doesn't require an explicit accumulator tile.

### Parameters

| Parameter | Type | Description | Shape |
|-----------|------|-------------|-------|
| `a` | `Tile` | First matrix operand | (M, K) |
| `b` | `Tile` | Second matrix operand | (K, N) |

### Returns

- **Type**: `Tile`
- **Shape**: (M, N)
- **Description**: Result of `a × b`

### Usage

```python
@ct.kernel
def simple_matmul(A: ct.Buffer, B: ct.Buffer, C: ct.Buffer):
    """Simple matrix multiplication without explicit accumulator."""
    i, j = ct.bid(0), ct.bid(1)
    
    # Load tiles
    a_tile = A.load((i, ct.tid()))
    b_tile = B.load((ct.tid(), j))
    
    # Direct matrix multiplication
    c_tile = ct.matmul(a_tile, b_tile)
    
    C.store((i, j), c_tile)
```

### When to Use `ct.matmul()` vs `ct.mma()`

**Use `ct.matmul()` when:**
- You need a simple, one-line matrix multiplication
- You don't need explicit control over the accumulator
- The operation is not part of a reduction loop

**Use `ct.mma()` when:**
- You need explicit control over the accumulator (e.g., for fused operations)
- You're implementing a reduction loop over K dimension
- You need to maintain accumulator state across multiple iterations
- You want to ensure FP32 accumulation for numerical stability

## Performance Considerations

### Tile Size Selection

Choosing the right tile size is critical for performance:

| Tile Size | Use Case | Pros | Cons |
|-----------|----------|------|------|
| 16×16 | Small matrices, minimal register pressure | Low register usage | Not enough work per thread |
| 32×32 | Medium matrices, balanced performance | Good tensor core utilization | Moderate register pressure |
| 64×64 | Large matrices, maximum throughput | Excellent tensor core utilization | High register pressure |
| 128×128 | Very large matrices, memory-bound | Hides memory latency well | May cause register spills |

**Guidelines:**
1. **Always use multiples of 16**: Tensor cores operate on 16×16 blocks
2. **Consider register limits**: Larger tiles use more registers
3. **Balance dimensions**: For square matrices, use square tiles
4. **Profile for your specific hardware**: Optimal sizes vary by GPU architecture

### Pipeline Depth and Latency Hiding

Modern GPUs can overlap computation with memory transfers. Here's a pattern for pipelined matmul:

```python
@ct.kernel
def pipelined_matmul(
    A: ct.Buffer,
    B: ct.Buffer,
    C: ct.Buffer,
    STAGES: ct.Constant[int] = 4
):
    """
    Pipelined matrix multiplication with software-managed cache.
    
    Uses async loads to overlap memory transfers with computation.
    """
    TM, TN, TK = 128, 128, 32
    
    i, j = ct.bid(0), ct.bid(1)
    
    # Shared memory tiles
    A_shared = ct.shared((TM, TK), A.dtype)
    B_shared = ct.shared((TK, TN), B.dtype)
    
    acc = ct.zeros((TM, TN), ct.float32)
    
    # Pipeline loop
    for k in range(A.shape[1] // TK):
        # Async load next tiles
        if k + STAGES < A.shape[1] // TK:
            A_shared.copy_from(A.load((i, k + STAGES), shape=(TM, TK)))
            B_shared.copy_from(B.load((k + STAGES, j), shape=(TK, TN)))
        
        # Wait for current tiles
        ct.sync_threads()
        
        # Compute with current tiles
        acc = ct.mma(A_shared, B_shared, acc)
    
    C.store((i, j), acc.astype(C.dtype))
```

### Data Type Combinations

Different data type combinations offer different trade-offs:

**FP16 + FP32 Accumulation** (Recommended for most DL workloads):
```python
# Best balance of speed and accuracy
A = ct.float16
B = ct.float16
Acc = ct.float32
```

**BF16 + FP32 Accumulation** (Better dynamic range):
```python
# Better for training gradients
A = ct.bfloat16
B = ct.bfloat16
Acc = ct.float32
```

**TF32 + FP32 Accumulation** (Ampere+ only):
```python
# Faster than FP32, similar accuracy
A = ct.float32  # Will use TF32 on hardware
B = ct.float32
Acc = ct.float32
```

**INT8 + INT32 Accumulation** (Inference):
```python
# Maximum throughput, reduced precision
A = ct.int8
B = ct.int8
Acc = ct.int32
```

### Memory Alignment

For optimal performance, ensure proper memory alignment:

```python
# Align buffer allocations to 128-byte boundaries
A = ct.empty((M, K), dtype=ct.float16, alignment=128)
B = ct.empty((K, N), dtype=ct.float16, alignment=128)
C = ct.empty((M, N), dtype=ct.float16, alignment=128)
```

## Complete Examples

### Fused MatMul + Bias + ReLU

```python
@ct.kernel
def matmul_bias_relu(
    X: ct.Buffer,      # (M, K)
    W: ct.Buffer,      # (K, N)
    B: ct.Buffer,      # (N,) bias
    Y: ct.Buffer,      # (M, N) output
    TM: ct.Constant[int],
    TN: ct.Constant[int],
    TK: ct.Constant[int]
):
    """
    Fused operation: Y = ReLU(X @ W + B)
    
    Fusion reduces memory traffic and improves performance.
    """
    i, j = ct.bid(0), ct.bid(1)
    
    # Tiled views
    x_view = X.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    w_view = W.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    
    # MatMul
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(x_view.num_tiles(1)):
        tx = x_view.load((i, k))
        tw = w_view.load((k, j))
        acc = ct.mma(tx, tw, acc)
    
    # Add bias (broadcast along M dimension)
    bias_tile = B.load((j,), shape=(TN,))
    acc = acc + bias_tile
    
    # ReLU activation
    acc = ct.maximum(acc, 0)
    
    # Store
    Y.store((i, j), acc.astype(Y.dtype))
```

### Fused MatMul + LayerNorm

```python
@ct.kernel
def matmul_layernorm(
    X: ct.Buffer,      # (M, K)
    W: ct.Buffer,      # (K, N)
    Gamma: ct.Buffer,  # (N,)
    Beta: ct.Buffer,   # (N,)
    Y: ct.Buffer,      # (M, N)
    TM: ct.Constant[int],
    TN: ct.Constant[int],
    TK: ct.Constant[int],
    EPS: ct.Constant[float] = 1e-5
):
    """
    Fused: Y = LayerNorm(X @ W)
    
    Computes: Y = (X @ W - mean) / sqrt(var + EPS) * Gamma + Beta
    """
    i, j = ct.bid(0), ct.bid(1)
    
    # MatMul phase
    x_view = X.tiled_view((TM, TK), padding_mode=ct.PaddingMode.ZERO)
    w_view = W.tiled_view((TK, TN), padding_mode=ct.PaddingMode.ZERO)
    
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(x_view.num_tiles(1)):
        tx = x_view.load((i, k))
        tw = w_view.load((k, j))
        acc = ct.mma(tx, tw, acc)
    
    # LayerNorm phase
    # Compute mean
    mean = ct.mean(acc, axis=1, keepdim=True)
    
    # Compute variance
    var = ct.mean((acc - mean) ** 2, axis=1, keepdim=True)
    
    # Normalize
    normalized = (acc - mean) / ct.sqrt(var + EPS)
    
    # Scale and shift
    gamma_tile = Gamma.load((j,), shape=(TN,))
    beta_tile = Beta.load((j,), shape=(TN,))
    y_tile = normalized * gamma_tile + beta_tile
    
    Y.store((i, j), y_tile.astype(Y.dtype))
```

### Strided Matrix Multiplication

```python
@ct.kernel
def strided_matmul(
    A: ct.Buffer,  # (M, K) with leading dimension
    B: ct.Buffer,  # (K, N)
    C: ct.Buffer,  # (M, N)
    lda: ct.Constant[int],  # Leading dimension of A
    ldb: ct.Constant[int],  # Leading dimension of B
    ldc: ct.Constant[int],  # Leading dimension of C
):
    """
    Matrix multiplication with custom leading dimensions.
    Useful for working with submatrices and views.
    """
    TM, TN, TK = 64, 64, 32
    
    i, j = ct.bid(0), ct.bid(1)
    
    # Load with custom strides
    a_tile = A.load((i, ct.tid()), shape=(TM, TK), stride=(lda, 1))
    b_tile = B.load((ct.tid(), j), shape=(TK, TN), stride=(ldb, 1))
    
    # Compute
    c_tile = ct.zeros((TM, TN), ct.float32)
    c_tile = ct.mma(a_tile, b_tile, c_tile)
    
    # Store with custom stride
    C.store((i, j), c_tile, stride=(ldc, 1))
```

### Split-K Matrix Multiplication

```python
@ct.kernel
def splitk_matmul(
    A: ct.Buffer,  # (M, K)
    B: ct.Buffer,  # (K, N)
    C: ct.Buffer,  # (M, N, SPLIT_K) - intermediate results
    TM: ct.Constant[int],
    TN: ct.Constant[int],
    TK: ct.Constant[int],
    SPLIT_K: ct.Constant[int] = 4
):
    """
    Split-K matrix multiplication for better parallelization.
    
    The K dimension is split across SPLIT_K parallel reductions.
    Final reduction is done in a separate kernel.
    """
    i, j = ct.bid(0), ct.bid(1)
    split_k = ct.bid(2)  # Third grid dimension
    
    # Adjust K range for this split
    k_start = split_k * (K // SPLIT_K)
    k_end = (split_k + 1) * (K // SPLIT_K)
    
    # Create views for partial K range
    a_view = A[:, k_start:k_end].tiled_view((TM, TK))
    b_view = B[k_start:k_end, :].tiled_view((TK, TN))
    
    # Partial reduction
    acc = ct.zeros((TM, TN), ct.float32)
    for k in range(a_view.num_tiles(1)):
        tx = a_view.load((i, k))
        ty = b_view.load((k, j))
        acc = ct.mma(tx, ty, acc)
    
    # Store partial result
    C.store((i, j, split_k), acc.astype(C.dtype))

@ct.kernel
def splitk_reduce(
    C_partial: ct.Buffer,  # (M, N, SPLIT_K)
    C_final: ct.Buffer,    # (M, N)
):
    """Final reduction for split-K matmul."""
    i, j = ct.bid(0), ct.bid(1)
    
    # Sum over split-K dimension
    acc = ct.sum(C_partial[i, j, :], axis=0)
    C_final.store((i, j), acc)
```

## Summary

Matrix operations in cuTile provide direct access to GPU tensor cores for maximum performance:

- **`ct.mma(a, b, c)`**: Matrix multiply-accumulate, maps to tensor core instructions
- **`ct.matmul(a, b)`**: Convenience wrapper for simple matrix multiplication

**Key takeaways:**
1. Use FP16/BF16 inputs with FP32 accumulation for best performance/accuracy trade-off
2. Choose tile sizes that are multiples of 16 for optimal tensor core utilization
3. Consider fusing operations (bias, activation, normalization) to reduce memory traffic
4. Profile different tile sizes and data type combinations for your specific hardware
5. Use pipelining and software-managed caching to hide memory latency

The next chapters cover element-wise operations, reductions, and memory operations that can be combined with matrix operations to build complete, high-performance kernels.
