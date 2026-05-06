# FlashAttention Triton Kernels Reference

This document provides comprehensive reference documentation for the Triton-based kernels in FlashAttention. These kernels provide highly optimized implementations using OpenAI's Triton language for GPU programming.

---

## Table of Contents

1. [Overview](#overview)
2. [Cross Entropy Loss (cross_entropy.py)](#cross-entropy-loss)
3. [Layer Norm (layer_norm.py)](#layer-norm-triton)
4. [Linear Operations (linear.py)](#linear-operations)
5. [MLP Operations (mlp.py)](#mlp-operations)
6. [Rotary Embedding (rotary.py)](#rotary-embedding)
7. [Activation Kernels (k_activations.py)](#activation-kernels)
8. [Tensor Parallel Utilities](#tensor-parallel-utilities)

---

## Overview

The Triton ops directory provides alternative implementations of common transformer operations using Triton, which offers:

1. **Portability**: Triton kernels work across GPU architectures without architecture-specific code
2. **Fusion**: Multiple operations are fused into single kernels, reducing memory bandwidth
3. **torch.compile compatibility**: Triton ops integrate with PyTorch's compilation infrastructure
4. **Autotuning**: Automatic selection of optimal kernel configurations

### Module Structure

```
flash_attn/ops/triton/
    __init__.py
    cross_entropy.py     # Cross-entropy loss with label smoothing
    layer_norm.py        # Layer norm / RMS norm with dropout + residual
    linear.py            # Fused linear + activation (forward and backward)
    mlp.py               # Fused Dense-SqReLU-Dense MLP
    rotary.py            # Rotary embedding kernel
    k_activations.py     # Triton activation function kernels
```

---

## Cross Entropy Loss

**File:** `flash_attn/ops/triton/cross_entropy.py`

Highly optimized cross-entropy loss implementation with support for label smoothing, logit scaling, z-loss regularization, and tensor parallelism.

### Functional API

#### `cross_entropy_loss`

```python
def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    precomputed_lse: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index=-100,
    inplace_backward: bool = False,
    process_group=None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Parameters:**
- `logits` (torch.Tensor): Raw logits, shape `(batch, vocab_size)`. Can be a shard of the full vocabulary for tensor parallel.
- `labels` (torch.Tensor): Target labels, shape `(batch,)`
- `precomputed_lse` (torch.Tensor, optional): Pre-computed log-sum-exp. Only used when `logit_scale == 1.0` and `label_smoothing == 0.0`
- `label_smoothing` (float): Label smoothing factor (default: 0.0). With smoothing `s`, the loss becomes:
  ```
  loss = -((1-s) * log(p_target) + s * sum(log(p_i)) / total_classes)
  ```
- `logit_scale` (float): Multiply logits by this scale before computing loss (default: 1.0). Must be > 0.
- `lse_square_scale` (float): Z-loss coefficient. If > 0, adds `lse_square_scale * lse(logits)^2` to the loss for training stability (default: 0.0)
- `ignore_index` (int): Labels matching this value contribute 0 to the loss (default: -100)
- `inplace_backward` (bool): Modify logits in-place during backward to save memory (default: False)
- `process_group` (ProcessGroup, optional): For tensor parallel, each rank handles a shard of the vocabulary

**Returns:**
- `losses` (torch.Tensor): Per-sample losses, shape `(batch,)`, dtype float32
- `z_losses` (torch.Tensor): Per-sample z-loss components, shape `(batch,)`, dtype float32

### Triton Kernels

#### `cross_entropy_fwd_kernel`

```python
@triton.heuristics({"HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0})
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr, lse_ptr, z_loss_ptr, logits_ptr, labels_ptr,
    smoothing, logit_scale, lse_square_scale, ignore_index,
    total_classes, class_start_idx, n_cols, logits_row_stride,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr, PRECOMPUTED_LSE: tl.constexpr,
)
```

**Forward kernel algorithm:**
1. One program per row (sample) in the batch
2. Computes online softmax statistics (max, sum-exp) for numerical stability
3. Computes log-sum-exp (LSE)
4. Calculates loss based on label index, smoothing, and z-loss scale
5. For tensor parallel (`SPLIT=True`): does not include LSE in loss (added after all-gather)

**Compile-time constants:**
- `BLOCK_SIZE`: Power-of-2 block size for vocabulary traversal
- `HAS_SMOOTHING`: Whether label smoothing is applied
- `SPLIT`: Whether in tensor parallel mode
- `PRECOMPUTED_LSE`: Whether LSE is provided externally

**Block size selection:**
- Up to 16K columns: Uses `min(next_power_of_2(n_cols), 16384)`
- Warps: 4 for < 2048, 8 for < 8192, 16 for < 128K, 32 otherwise

#### `cross_entropy_bwd_kernel`

```python
@triton.heuristics({"HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr, dloss_ptr, logits_ptr, lse_ptr, labels_ptr,
    smoothing, logit_scale, lse_square_scale, ignore_index,
    total_classes, class_start_idx, n_cols,
    logits_row_stride, dlogits_row_stride, dloss_row_stride,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr,
)
```

**Backward kernel algorithm:**
1. 2D grid: one program per (row, column_block) pair
2. Loads logits, computes probabilities via `exp(logit - lse)`
3. Applies z-loss gradient: `probs += 2 * lse_square_scale * lse * probs`
4. Applies label smoothing gradient
5. Subtracts 1 from the target probability
6. Scales by `dloss * logit_scale`

### Tensor Parallel Cross Entropy

When `process_group` is provided:

1. **Forward**:
   - Each rank computes local LSE for its vocabulary shard
   - All-gather LSEs across ranks
   - Compute global LSE via `logsumexp` of gathered LSEs
   - All-reduce local losses (partial sums)
   - Add global LSE to get final loss

2. **Backward**:
   - Each rank computes gradients only for its vocabulary shard
   - No cross-rank communication needed in backward (each rank has its logits)

### Usage Example

```python
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

# Basic cross entropy
logits = torch.randn(128, 50257, device="cuda", dtype=torch.float32)
labels = torch.randint(0, 50257, (128,), device="cuda")
losses, z_losses = cross_entropy_loss(logits, labels)

# With label smoothing and z-loss
losses, z_losses = cross_entropy_loss(
    logits, labels,
    label_smoothing=0.1,
    lse_square_scale=1e-4,
)

# Tensor parallel (each rank has vocab_size/world_size columns)
losses, z_losses = cross_entropy_loss(
    logits, labels,
    process_group=tp_group,
)

# In-place backward to save memory
losses, z_losses = cross_entropy_loss(
    logits, labels,
    inplace_backward=True,
)
```

---

## Layer Norm Triton

**File:** `flash_attn/ops/triton/layer_norm.py`

Triton implementation of dropout + residual addition + LayerNorm/RMSNorm. Supports both standard and parallel block architectures, with full `torch.compile` compatibility via `triton_op`.

### Functional API

#### `layer_norm_fn`

```python
def layer_norm_fn(
    x, weight, bias,
    residual=None, x1=None, weight1=None, bias1=None,
    eps=1e-6, dropout_p=0.0, rowscale=None,
    prenorm=False, residual_in_fp32=False,
    zero_centered_weight=False, is_rms_norm=False,
    return_dropout_mask=False, out_dtype=None,
    out=None, residual_out=None
)
```

**Parameters:**
- `x` (torch.Tensor): Main input, shape `(*, hidden_size)`
- `weight` (torch.Tensor): Gamma (gain) parameter, shape `(hidden_size,)`
- `bias` (torch.Tensor, optional): Beta (bias) parameter, shape `(hidden_size,)`
- `residual` (torch.Tensor, optional): Residual tensor to add before normalization
- `x1` (torch.Tensor, optional): Second input for parallel block (e.g., MLP output). Added to `x` before normalization.
- `weight1` (torch.Tensor, optional): Second set of gamma parameters for parallel norm
- `bias1` (torch.Tensor, optional): Second set of beta parameters
- `eps` (float): Epsilon for numerical stability (default: 1e-6)
- `dropout_p` (float): Dropout probability (default: 0.0)
- `rowscale` (torch.Tensor, optional): Per-row scaling, shape `(rows,)`
- `prenorm` (bool): Return pre-norm residual (default: False)
- `residual_in_fp32` (bool): Cast residual to fp32 (default: False)
- `zero_centered_weight` (bool): Treat weight as (weight - 1), initialize to zeros (default: False)
- `is_rms_norm` (bool): Use RMSNorm instead of LayerNorm (default: False)
- `return_dropout_mask` (bool): Return dropout masks (default: False)
- `out` (torch.Tensor, optional): Pre-allocated output tensor
- `residual_out` (torch.Tensor, optional): Pre-allocated residual output tensor

**Returns (varies by configuration):**
- Default: Normalized output `y`
- `prenorm=True`: `(y, residual_out)`
- `weight1 is not None`: `(y, y1)` or `(y, y1, residual_out)` with prenorm
- `return_dropout_mask=True`: Adds dropout masks to return tuple

#### `rms_norm_fn`

```python
def rms_norm_fn(
    x, weight, bias,
    residual=None, x1=None, weight1=None, bias1=None,
    eps=1e-6, dropout_p=0.0, rowscale=None,
    prenorm=False, residual_in_fp32=False,
    zero_centered_weight=False, return_dropout_mask=False,
    out_dtype=None, out=None, residual_out=None
)
```

Same as `layer_norm_fn` with `is_rms_norm=True`.

### Module Classes

#### `RMSNorm` (Triton version)

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0,
                 zero_centered_weight=False, device=None, dtype=None)
```

**Parameters:**
- `hidden_size` (int): Hidden dimension
- `eps` (float): Epsilon (default: 1e-5)
- `dropout_p` (float): Dropout probability
- `zero_centered_weight` (bool): Initialize weight to zeros, actual weight = weight + 1

**Methods:**

##### `forward`

```python
def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False)
```

### Fused Layer Norm + Linear

#### `layer_norm_linear_fn`

```python
def layer_norm_linear_fn(
    x, norm_weight, norm_bias, linear_weight, linear_bias,
    residual=None, eps=1e-6, prenorm=False,
    residual_in_fp32=False, is_rms_norm=False,
)
```

Fuses LayerNorm and subsequent linear layer. Recomputes the LayerNorm output in backward to save memory (instead of storing it).

**Parameters:**
- `x` (torch.Tensor): Input
- `norm_weight` (torch.Tensor): LayerNorm gamma
- `norm_bias` (torch.Tensor): LayerNorm beta
- `linear_weight` (torch.Tensor): Linear layer weight
- `linear_bias` (torch.Tensor, optional): Linear layer bias
- `residual` (torch.Tensor, optional): Residual connection

**Returns:**
- Default: Linear layer output
- `prenorm=True`: `(linear_output, residual_out)`

### Reference Implementations

#### `layer_norm_ref`

Reference implementation of all layer norm operations using standard PyTorch. Used for testing.

```python
def layer_norm_ref(x, weight, bias, residual=None, x1=None, weight1=None, bias1=None,
                   eps=1e-6, dropout_p=0.0, rowscale=None, prenorm=False,
                   zero_centered_weight=False, dropout_mask=None, dropout_mask1=None, upcast=False)
```

#### `rms_norm_ref`

Reference implementation of RMS norm operations.

### Triton Kernels

#### `_layer_norm_fwd_1pass_kernel`

Single-pass forward kernel that fuses all operations:
1. Load input row
2. Apply rowscale (if provided)
3. Apply dropout (using Triton's random number generator)
4. Add second input `x1` (for parallel blocks)
5. Add residual
6. Store residual output
7. Compute mean (for LayerNorm) or skip (for RMSNorm)
8. Compute variance and reciprocal standard deviation
9. Normalize and apply affine transformation (weight, bias)
10. Apply second affine transformation (weight1, bias1) if parallel norm

**Autotuning:** Automatically selects optimal number of warps from `[1, 2, 4, 8, 16, 32]` based on dimension and GPU.

**Block size:** `min(65536 // element_size, next_power_of_2(N))` -- supports hidden dimensions up to 64KB.

#### `_layer_norm_bwd_kernel`

Backward kernel that computes all gradients in a single pass:
1. Each program handles a block of rows (`rows_per_program = ceil(M / sm_count)`)
2. Accumulates weight and bias gradients in registers
3. Writes partial weight/bias gradients to global memory
4. Final reduction of weight/bias gradients happens on CPU

**Key optimizations:**
- Weight and bias gradients are accumulated in registers (fast for hidden_dim up to ~8K)
- Multiple rows per program to amortize launch overhead
- Support for recomputing output during backward to save memory

### torch.compile Integration

The Triton layer norm uses `triton_op` decorator for `torch.compile` compatibility:

```python
@triton_op("flash_attn::layer_norm_fwd_impl", mutates_args={"out", "residual_out"}, ...)
def _layer_norm_fwd_impl(...)

@triton_op("flash_attn::layer_norm_bwd_impl", mutates_args={}, allow_decomposition=False, ...)
def _layer_norm_bwd_impl(...)
```

This allows the kernels to work seamlessly with `torch.compile` while maintaining their optimized implementations.

---

## Linear Operations

**File:** `flash_attn/ops/triton/linear.py`

Triton-based fused linear + activation operations for both forward and backward passes.

### Functional API

#### `triton_linear_act`

```python
def triton_linear_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "id",
    save_act_input: bool = False,
) -> torch.Tensor
```

Computes `activation(x @ weight.T + bias)` using Triton.

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape `(*, K)`
- `weight` (torch.Tensor): Weight matrix, shape `(N, K)`
- `bias` (torch.Tensor, optional): Bias vector, shape `(N,)`
- `activation` (str): Activation function, one of `"id"`, `"gelu"`, `"gelu_approx"`, `"squared_relu"`
- `save_act_input` (bool): Save the pre-activation input for backward pass

**Returns:**
- If `save_act_input=False`: Output tensor, shape `(*, N)`
- If `save_act_input=True`: Tuple of (output, act_input), both shape `(*, N)`

#### `triton_dgrad_act`

```python
def triton_dgrad_act(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    activation: str = "id",
    act_input: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

Computes the backward pass through linear + activation: `grad_input = activation'(grad_output @ weight) * act_input_grad`

**Parameters:**
- `grad_output` (torch.Tensor): Gradient from next layer, shape `(*, N)`
- `weight` (torch.Tensor): Weight matrix, shape `(N, K)` -- note: NOT transposed
- `activation` (str): Activation function (must match forward)
- `act_input` (torch.Tensor): Pre-activation input saved during forward

**Returns:** `grad_input`, shape `(*, K)`

### Triton Kernels

#### `kernel_fwd`

Autotuned kernel for `output = activation(x @ weight.T + bias)`.

**Autotuning configurations:**
- 18 primary configs with varying `BLOCK_M`, `BLOCK_N`, `BLOCK_K` values
- Additional IO-bound configs for small matrices
- Uses `triton.ops.matmul_perf_model` for pruning and selection

**Key compile-time options:**
- `BIAS`: Whether to add bias
- `SAVE_ACT_INPUT`: Whether to save pre-activation values
- `ACTIVATION`: Which activation to apply (`"gelu"`, `"gelu_approx"`, `"squared_relu"`)
- `A_ROWMAJOR`: Whether input is row-major (stride trick for efficiency)
- `B_COLMAJOR`: Whether weight is column-major

**L2 optimization:** Groups program IDs for better L2 cache locality using `GROUP_M=8`.

#### `kernel_bwd`

Kernel for computing `grad_input = (grad_output @ weight) * activation'(act_input)`.

Similar autotuning to `kernel_fwd`. Applies activation gradient element-wise after the matmul.

---

## MLP Operations

**File:** `flash_attn/ops/triton/mlp.py`

Fused Dense-SqReLU-Dense MLP implementation using Triton for the first layer (fused matmul + squared ReLU) and CUDA for the second layer.

### Classes

#### `FusedDenseSqreluDense`

```python
class FusedDenseSqreluDense(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=True,
        bias2=True,
        checkpoint_lvl=0,
        device=None,
        dtype=None,
    )
```

Two-layer MLP with squared ReLU activation: `output = Linear2(SqReLU(Linear1(x)))`

**Parameters:**
- `in_features` (int): Input dimension
- `hidden_features` (int, optional): Hidden dimension (default: `4 * in_features`)
- `out_features` (int, optional): Output dimension (default: `in_features`)
- `bias1` (bool): Must be True (limitation)
- `bias2` (bool): Must be True (limitation)
- `checkpoint_lvl` (int): Gradient checkpointing level:
  - `0`: No recomputation
  - `1`: Recompute SqReLU output in backward
  - `2`: Recompute pre-activation and SqReLU output

**Methods:**

##### `forward`

```python
def forward(self, x)
```

**Behavior:**
- For **bf16**: Uses `fused_dense_cuda.linear_bias_forward` + `sqrelu_fwd` (separate kernels)
- For **fp16**: Uses `triton_linear_act` with `activation="squared_relu"` (fused kernel)

### Autograd Function

#### `FusedDenseSqreluDenseFunc`

```python
class FusedDenseSqreluDenseFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight1, bias1, weight2, bias2, checkpoint_lvl=0)

    @staticmethod
    def backward(ctx, grad_output)
```

**Forward optimizations:**
- bf16: Separate CUDA linear + JIT SqReLU
- fp16: Fused Triton linear + squared ReLU in single kernel

**Backward optimizations:**
- bf16: CUDA linear wgrad + SqReLU bwd + CUDA linear bwd
- fp16: CUDA linear wgrad + Triton fused dgrad + CUDA linear bwd

---

## Rotary Embedding

**File:** `flash_attn/ops/triton/rotary.py`

Triton kernel for applying rotary positional embeddings (RoPE).

### Functional API

#### `apply_rotary`

```python
def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor
```

Applies rotary embedding to input tensor.

**Parameters:**
- `x` (torch.Tensor): Input tensor.
  - If `cu_seqlens` is None: shape `(batch, seqlen, nheads, headdim)`
  - If `cu_seqlens` is provided: shape `(total_seqlen, nheads, headdim)` (varlen)
- `cos` (torch.Tensor): Cosine values, shape `(seqlen_ro, rotary_dim / 2)`
- `sin` (torch.Tensor): Sine values, shape `(seqlen_ro, rotary_dim / 2)`
- `seqlen_offsets` (int or torch.Tensor): Position offsets.
  - If int: Applied to all sequences
  - If tensor: Shape `(batch,)`, per-sequence offsets (for packed sequences)
- `cu_seqlens` (torch.Tensor, optional): Cumulative sequence lengths for varlen, shape `(batch + 1,)`
- `max_seqlen` (int, optional): Maximum sequence length (required with `cu_seqlens`)
- `interleaved` (bool): Whether rotary dimensions are interleaved (default: False)
  - Non-interleaved (LLaMA-style): First half is x0, second half is x1
  - Interleaved (GPT-J-style): x0 and x1 alternate
- `inplace` (bool): Modify `x` in place to save memory (default: False)
- `conjugate` (bool): Apply conjugate rotation (for backward pass) (default: False)

**Returns:** Rotated tensor, same shape as `x`

**Constraints:**
- `rotary_dim <= headdim`
- `headdim <= 256`
- `seqlen_ro >= seqlen`

### Triton Kernel

#### `rotary_kernel`

```python
@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, nheads, seqlen_ro,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    ROTARY_DIM: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr,
)
```

**3D grid:** `(pid_head, pid_m, pid_batch)`

**Algorithm for non-interleaved:**
1. Load first half of head dim (`x0`) and second half (`x1`)
2. Load cos and sin for current position
3. Compute: `o0 = x0 * cos - x1 * sin`, `o1 = x0 * sin + x1 * cos`
4. Store results

**Algorithm for interleaved:**
1. Load full head dim as interleaved `[x0, x1, x0, x1, ...]`
2. Split into x0 and x1 halves
3. Apply rotation
4. Join back and store

**Compile-time constants:**
- `ROTARY_DIM`: Must be constexpr for optimal LDG.128 vectorized loads
- `IS_SEQLEN_OFFSETS_TENSOR`: Whether offsets are per-batch
- `IS_VARLEN`: Whether using variable-length sequences
- `INTERLEAVED`: Interleaved vs. split rotary pattern
- `CONJUGATE`: Negate sin for backward pass

**Block sizes:**
- `BLOCK_M`: 8 if `rotary_dim <= 128`, else 4
- `BLOCK_H`: 2 (processes 2 heads per program)

### Usage Example

```python
from flash_attn.ops.triton.rotary import apply_rotary
import torch

batch, seqlen, nheads, headdim = 2, 512, 32, 128
rotary_dim = 64  # Apply rotary to first 64 dimensions

x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.float16)
cos = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=torch.float16)
sin = torch.randn(seqlen, rotary_dim // 2, device="cuda", dtype=torch.float16)

# Standard (non-interleaved) rotary
output = apply_rotary(x, cos, sin)

# Interleaved (GPT-J style)
output = apply_rotary(x, cos, sin, interleaved=True)

# With position offsets (e.g., for continuation)
offsets = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
output = apply_rotary(x, cos, sin, seqlen_offsets=offsets)

# Variable-length sequences
cu_seqlens = torch.tensor([0, 100, 512], device="cuda", dtype=torch.int32)
x_varlen = torch.randn(512, nheads, headdim, device="cuda", dtype=torch.float16)
output = apply_rotary(x_varlen, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=512)

# In-place for memory savings
apply_rotary(x, cos, sin, inplace=True)  # Modifies x directly
```

---

## Activation Kernels

**File:** `flash_attn/ops/triton/k_activations.py`

Triton JIT-compiled activation functions for use within Triton kernels. These are called from `@triton.jit` decorated kernels, not from Python.

### Available Activations

#### ReLU

```python
@triton.jit
def relu(x):
    """Standard ReLU: max(0, x)"""

@triton.jit
def relu_grad(x):
    """ReLU gradient: 1 if x >= 0, else 0"""
```

#### Squared ReLU

```python
@triton.jit
def squared_relu(x):
    """Squared ReLU: max(0, x)^2, from the Primer paper"""

@triton.jit
def squared_relu_grad(x):
    """Gradient: 2*x if x >= 0, else 0"""
```

#### Leaky ReLU

```python
@triton.jit
def leaky_relu(x):
    """Leaky ReLU with slope 0.01 for negative values"""

@triton.jit
def leaky_relu_grad(x):
    """Leaky ReLU gradient: 1 if x >= 0, else 0.01"""
```

#### GELU (Exact)

```python
@triton.jit
def gelu(x):
    """Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))"""

@triton.jit
def gelu_grad(x):
    """GELU gradient: CDF(x) + x * PDF(x)"""
```

#### GELU (Tanh Approximation)

```python
@triton.jit
def gelu_approx(x):
    """Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))"""

@triton.jit
def gelu_approx_grad(x):
    """Approximate GELU gradient"""
```

### Activation Enum

```python
class Activation(str, Enum):
    SquaredReLU = "squared_relu"
    GeLU = "gelu"
    GeLUApprox = "gelu_approx"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"
```

### Lookup Functions

#### `get_triton_activation_kernel`

```python
def get_triton_activation_kernel(activation: Optional[Activation])
```

Returns the Triton JIT function for the given activation. Returns `None` if `activation` is `None`.

#### `get_triton_activation_bwd_kernel`

```python
def get_triton_activation_bwd_kernel(activation: Optional[Activation])
```

Returns the Triton JIT backward function for the given activation.

### Helper Functions

```python
@triton.jit
def tanh(x):
    """Tanh via sigmoid: 2 * sigmoid(2x) - 1"""

@triton.jit
def cosh(x):
    """Hyperbolic cosine"""
```

---

## Tensor Parallel Utilities

The Triton ops leverage distributed communication primitives from `flash_attn.utils.distributed`:

### `all_gather_raw`

```python
def all_gather_raw(tensor, process_group, async_op=False)
```

Gathers tensor from all ranks. Returns (gathered_tensor, handle) when `async_op=True`.

### `all_reduce_raw`

```python
def all_reduce_raw(tensor, process_group, op=ReduceOp.SUM, async_op=False)
```

Reduces tensor across all ranks. Returns (reduced_tensor, handle) when `async_op=True`.

### `reduce_scatter_raw`

```python
def reduce_scatter_raw(tensor, process_group, async_op=False)
```

Reduces then scatters tensor across ranks. Returns (scattered_tensor, handle) when `async_op=True`.

### `all_gather`

```python
def all_gather(tensor, process_group)
```

Synchronous all-gather wrapper.

### `reduce_scatter`

```python
def reduce_scatter(tensor, process_group)
```

Synchronous reduce-scatter wrapper.

### `all_reduce`

```python
def all_reduce(tensor, process_group)
```

Synchronous all-reduce wrapper.

### Communication Overlap Pattern

The Triton ops use async communication to overlap computation and communication:

```python
# Example: FusedDense forward with tensor parallel
total_x, handle_x = all_gather_raw(x, process_group, async_op=True)  # Start all-gather
# ... weight dtype conversion happens while all-gather is in flight ...
handle_x.wait()  # Wait for all-gather to complete
output = F.linear(total_x, weight, bias)  # Use gathered input
```
