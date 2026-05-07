# Autograd Functions Reference

This document provides a comprehensive reference for all autograd functions in bitsandbytes, covering the 8-bit and 4-bit matrix multiplication autograd functions, the dispatch functions, and the `MatmulLtState` dataclass that manages quantization state.

---

## Table of Contents

1. [MatMul8bitLt](#matmul8bitlt)
2. [MatMul8bitFp](#matmul8bitfp)
3. [MatMul4Bit](#matmul4bit)
4. [matmul() Dispatch Function](#matmul-dispatch-function)
5. [matmul_4bit() Dispatch Function](#matmul_4bit-dispatch-function)
6. [MatmulLtState Dataclass](#matmulltstate-dataclass)
7. [GlobalOutlierPooler](#globaloutlierpooler)

---

## MatMul8bitLt

`MatMul8bitLt` is a `torch.autograd.Function` implementing the LLM.int8() algorithm for 8-bit matrix multiplication. It is the primary GPU path for 8-bit quantized linear layers and supports both forward and backward passes with mixed-precision outlier handling.

**Location:** `bitsandbytes/autograd/_functions.py`

### Forward Pass

```python
class MatMul8bitLt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=None):
```

#### Step 1: Handle Empty Inputs

If the input tensor A has zero elements, the forward pass returns an empty tensor with the appropriate shape without performing any computation:

```python
ctx.is_empty = False
if prod(A.shape) == 0:
    ctx.is_empty = True
    ctx.A = A
    ctx.B = B
    ctx.bias = bias
    if A.shape[-1] == B.shape[0]:
        return torch.empty(A.shape[:-1] + B.shape[1:], dtype=A.dtype, device=A.device)
    else:
        return torch.empty(A.shape[:-1] + B.shape[:1], dtype=A.dtype, device=A.device)
```

#### Step 2: Cast A to fp16

The input A is cast to float16 for quantization. A warning is logged if the input is not already float16 (unless `torch.compiler.is_compiling()` is active):

```python
if A.dtype != torch.float16 and not _is_compiling():
    logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", A.dtype)
```

If A has 3 dimensions (batched input), it is reshaped to 2D:

```python
if len(A.shape) == 3:
    A = A.reshape(-1, A.shape[-1])
```

#### Step 3: Quantize A

Two paths depending on whether gradients are needed for B:

**Fast path (no grad for B):**
Uses `int8_vectorwise_quant`, which only computes row-wise quantization:

```python
if ctx.needs_input_grad[1]:
    # Slower path: double quantization for grad_B computation
    CA, CAt, SCA, SCAt, outlier_cols = F.int8_double_quant(
        A.to(torch.float16), threshold=state.threshold
    )
else:
    # Fast path: single quantization
    CA, SCA, outlier_cols = F.int8_vectorwise_quant(
        A.to(torch.float16), threshold=state.threshold
    )
    CAt = SCAt = None
```

- `CA` / `CAt`: Int8 quantized versions of A (row-wise and column-wise)
- `SCA` / `SCAt`: Float32 scaling statistics (row-wise and column-wise)
- `outlier_cols`: Optional tensor of column indices exceeding the threshold

#### Step 4: Quantize B (if needed)

B is quantized if it has fp16 weights or if no cached quantized version exists:

```python
if state.has_fp16_weights or state.CB is None:
    has_grad = getattr(B, "grad", None) is not None
    is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
    if is_transposed:
        B = B.contiguous()

    if (state.is_training and not has_grad) or state.CB is None or state.SCB is None:
        state.reset_grads()
        state.CB, state.SCB, _ = F.int8_vectorwise_quant(B.to(torch.float16))
```

The `reset_grads()` call clears cached `CB`, `SB`, `SCB`, `SBt`, `CBt` to ensure fresh quantization. The quantized weight `CB` and its scaling factors `SCB` are cached in `state` for reuse across forward passes.

#### Step 5: Mixed-Precision Path (threshold > 0)

When the outlier threshold is positive, the LLM.int8() mixed-precision decomposition is activated:

```python
if state.threshold > 0.0:
    state.idx = outlier_cols
    output, subA = torch.ops.bitsandbytes.int8_mixed_scaled_mm(
        A, CA, state.CB, SCA, state.SCB, outlier_cols, bias,
    )
```

This path:
1. Performs int8 matmul on non-outlier columns
2. Extracts the outlier values from A into `subA`
3. The `subA` matrix is later used in the backward pass for gradient correction

#### Step 6: Standard Path (threshold == 0)

```python
else:
    output = torch.ops.bitsandbytes.int8_scaled_mm.default(
        CA, state.CB, SCA, state.SCB, bias=bias, dtype=A.dtype
    )
    subA = None
```

This performs a standard int8 matmul followed by dequantization using row and column statistics.

#### Step 7: Save State for Backward

```python
ctx.state = state
ctx.grad_shape = input_shape
ctx.dtype_A = A.dtype
ctx.dtype_bias = None if bias is None else bias.dtype

if any(ctx.needs_input_grad[:2]):
    ctx.tensors = (CAt, subA, A)
    ctx.tensor_states = (SCAt, state.idx)
else:
    ctx.tensors = [None, None, None]
    ctx.tensor_states = (None, None)
    ctx.save_for_backward(None, None)
```

What is saved depends on which inputs require gradients:
- If A or B needs gradients: saves `CAt` (column-wise quantized A), `subA` (outlier submatrix), original `A`, and `SCAt` with `idx`
- If neither needs gradients: saves nothing (saves memory)

#### Output Shape

```python
output_shape = (*input_shape[:-1], state.CB.shape[0])
if len(input_shape) == 3:
    return output.reshape(output_shape)
return output
```

### Backward Pass

```python
@staticmethod
def backward(ctx, grad_output):
```

#### Empty Tensor Handling

```python
if ctx.is_empty:
    bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
    return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None
```

#### grad_bias

Computed first before any dtype changes to grad_output:

```python
if req_gradBias:
    grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
```

The sum is performed over the batch dimension (dim=0), with the result cast to the original bias dtype.

#### grad_output Reshaping

If grad_output is 3D (batched), it is reshaped to 2D for matrix operations:

```python
if len(grad_output.shape) == 3:
    grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
```

#### grad_B

The gradient of B is computed using double quantization of grad_output and an int8 scaled matmul:

```python
if req_gradB:
    # Double quantize grad_output (both row-wise and column-wise)
    Cgrad, _, _, SCgradt, _ = F.int8_double_quant(grad_output.to(torch.float16))

    # Int8 matmul: Cgrad^T @ CAt^T with column-wise scaling
    grad_B = torch.ops.bitsandbytes.int8_scaled_mm.default(
        Cgrad.t().contiguous(),
        CAt.t(),
        SCgradt,
        SCAt,
        dtype=torch.float16,
    )

    # Outlier correction for mixed-precision path
    if state.threshold > 0.0 and subA is not None and subA.numel() > 0:
        grad_B[:, idx] += torch.matmul(grad_output.t(), subA)
```

The outlier correction adds the fp16 gradient contribution from outlier columns that were excluded from the int8 computation.

#### grad_A

The gradient of A is computed by dequantizing CB back to the original dtype and performing a standard fp16 matmul:

```python
if req_gradA:
    if state.CB is not None:
        # Dequantize CB: int8 -> float by multiplying by SCB/127
        CB = state.CB.to(ctx.dtype_A, copy=True).mul_(
            state.SCB.unsqueeze(1).mul(1.0 / 127.0)
        )
        grad_A = torch.matmul(grad_output.to(ctx.dtype_A), CB).view(ctx.grad_shape)
    else:
        raise Exception("State must contain CB matrix for backward")
```

The dequantization formula: `CB_float = CB_int8 * SCB / 127` where SCB is the row-wise absolute maximum.

#### Return Signature

```python
return grad_A, grad_B, None, grad_bias, None
```

Maps to `(A, B, out, bias, state)` from the forward signature.

---

## MatMul8bitFp

`MatMul8bitFp` is a faster alternative to `MatMul8bitLt` for CPU and XPU devices. Instead of performing int8 matmul with dequantization, it quantizes B to int8, dequantizes in-place (multiplying by SCB/127), and uses standard `F.linear` for the actual computation.

**Location:** `bitsandbytes/autograd/_functions.py`

### Why CPU/XPU Fast Path?

From the source comments:

> For Intel CPU and XPU, MatMul8bitFp is much faster (~3x) than MatMul8bitLt in finetune. Because MatMul8bitLt has more mechanisms in computing grad. We don't have fast kernels for quant/dequant 8-bit in CPU/XPU, so it's very slow. We'd like to use dequant + matmul to run finetune with good performance.

### Forward Pass

```python
@staticmethod
def forward(ctx, A, B, out=None, bias=None, state=MatmulLtState):
    if state.has_fp16_weights or state.CB is None:
        has_grad = getattr(B, "grad", None) is not None
        is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
        if is_transposed:
            B = B.contiguous()

        if (state.is_training and not has_grad) or state.CB is None or state.SCB is None:
            state.reset_grads()
            state.CB, state.SCB, _ = F.int8_vectorwise_quant(B.to(torch.float16))
            B = state.CB

    # Dequantize in-place: CB * (SCB / 127)
    CB = state.CB.data.to(A.dtype).mul_(state.SCB.unsqueeze(1).mul(1.0 / 127.0))

    # Standard linear operation
    output = torch.nn.functional.linear(A, CB, bias)

    ctx.state = state
    ctx.dtype_A = A.dtype
    ctx.grad_shape = A.shape
    ctx.A = A
    ctx.dtype_bias = None if bias is None else bias.dtype
    return output
```

Key differences from MatMul8bitLt:
1. No int8 matmul -- uses `F.linear` (standard PyTorch) after dequantization
2. Dequantization is in-place on the CB tensor: `CB.data.to(A.dtype).mul_(SCB/127)`
3. Saves the original A tensor directly (not quantized versions)
4. No mixed-precision/outlier handling

### Backward Pass

```python
@staticmethod
def backward(ctx, grad_output):
    req_gradA, req_gradB, _, req_gradBias, _ = ctx.needs_input_grad
    A = ctx.A
    state = ctx.state
    grad_A = grad_B = grad_bias = None

    if req_gradBias:
        grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

    if len(grad_output.shape) == 3:
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

    if req_gradB:
        # Standard matmul for gradient of B
        grad_B = torch.matmul(A.t(), grad_output).t()

    if req_gradA:
        if state.CB is not None:
            # Dequantize CB and compute grad_A
            CB = state.CB.to(ctx.dtype_A, copy=True).mul_(
                state.SCB.unsqueeze(1).mul(1.0 / 127.0)
            )
            grad_A = torch.matmul(grad_output.to(ctx.dtype_A), CB).view(ctx.grad_shape)
        else:
            raise Exception("State must contain CB matrix for backward")

    return grad_A, grad_B, None, grad_bias, None
```

Key differences from MatMul8bitLt backward:
1. `grad_B = matmul(A.t(), grad_output).t()` -- simple fp16 matmul, no double quantization
2. No outlier correction needed
3. Saves the original A tensor instead of quantized CAt/SCAt

---

## MatMul4Bit

`MatMul4Bit` is a `torch.autograd.Function` for 4-bit quantized matrix multiplication (QLoRA). It dequantizes the weight matrix and performs standard `F.linear`.

**Location:** `bitsandbytes/autograd/_functions.py`

### Forward Pass

```python
@staticmethod
def forward(ctx, A, B, out=None, bias=None, quant_state=None):
    # Handle empty inputs
    ctx.is_empty = False
    if prod(A.shape) == 0:
        ctx.is_empty = True
        ctx.A = A; ctx.B = B; ctx.bias = bias
        B_shape = quant_state.shape
        if A.shape[-1] == B_shape[0]:
            return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
        else:
            return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

    # 1. Dequantize B from 4-bit to float
    # 2. Standard linear: F.linear(A, dequantized_B.t(), bias)
    output = torch.nn.functional.linear(
        A,
        F.dequantize_4bit(B, quant_state).to(A.dtype).t(),
        bias,
    )

    if out is not None:
        out.copy_(output)
        output = out

    # 3. Save state for backward
    ctx.state = quant_state
    ctx.dtype_A = A.dtype
    ctx.dtype_B = B.dtype
    ctx.dtype_bias = None if bias is None else bias.dtype

    if any(ctx.needs_input_grad[:2]):
        ctx.tensors = (None, B)
    else:
        ctx.tensors = (None, None)

    return output
```

Key notes:
- `F.dequantize_4bit(B, quant_state)` reconstructs the float weight from 4-bit packed data
- The dequantized weight is transposed (`.t()`) for `F.linear` convention
- The optional `out` parameter allows in-place output (used by the `.out` op variant)
- Only saves B (the quantized weight) for backward, not the dequantized version

### Backward Pass

```python
@staticmethod
def backward(ctx, grad_output):
    if ctx.is_empty:
        bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
        return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

    req_gradA, _, _, req_gradBias, _ = ctx.needs_input_grad
    _, B = ctx.tensors

    grad_A = grad_B = grad_bias = None

    if req_gradBias:
        grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

    # NOTE: grad_B is NOT supported!
    # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
    # "not supported by PyTorch" -- the comment in the source

    if req_gradA:
        grad_A = torch.matmul(
            grad_output,
            F.dequantize_4bit(B, ctx.state).to(grad_output.dtype).t(),
        )

    return grad_A, grad_B, None, grad_bias, None
```

**Critical limitation:** `grad_B` (the gradient with respect to the 4-bit quantized weight) is **not computed**. It is always `None`. The source comment reads: "not supported by PyTorch. TODO: create work-around."

This means that 4-bit quantized weights cannot be trained directly -- their gradients are not propagated. This is by design in the QLoRA approach: the 4-bit base weights are frozen, and training happens on Low-Rank Adaptation (LoRA) matrices instead.

The backward pass does compute `grad_A` by re-dequantizing B and performing `grad_output @ B_dequant^T`.

---

## matmul() Dispatch Function

The `matmul()` function dispatches to the appropriate 8-bit matmul autograd function based on device type and training state.

**Location:** `bitsandbytes/autograd/_functions.py`

```python
def matmul(A, B, out=None, state=None, threshold=0.0, bias=None):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold

    if state.is_training:
        if A.device.type in ("cpu", "xpu"):
            return MatMul8bitFp.apply(A, B, out, bias, state)
    return MatMul8bitLt.apply(A, B, out, bias, state)
```

### Dispatch Logic

```
matmul(A, B, ...)
    |
    +-- threshold > 0? -> state.threshold = threshold
    |
    +-- state.is_training AND device in (cpu, xpu)?
    |       |
    |       YES -> MatMul8bitFp  (fast dequant+linear path)
    |       |
    |       NO
    |
    +-- MatMul8bitLt  (full int8 matmul with optional mixed-precision)
```

### When Each Path Is Used

| Condition | Function | Rationale |
|-----------|----------|-----------|
| Training + CPU/XPU | `MatMul8bitFp` | ~3x faster on CPU/XPU since int8 kernels are slow there |
| Training + GPU | `MatMul8bitLt` | Full int8 path with CUDA tensor core acceleration |
| Not training + GPU | `MatMul8bitLt` | Cached quantized weights for inference |
| Not training + CPU/XPU | `MatMul8bitLt` | Falls through to same path |

---

## matmul_4bit() Dispatch Function

The `matmul_4bit()` function dispatches to the appropriate 4-bit matmul implementation based on device, batch size, gradient requirements, and packing format.

**Location:** `bitsandbytes/autograd/_functions.py`

```python
def matmul_4bit(A, B, quant_state, out=None, bias=None):
    assert quant_state is not None

    # CPU with packing format: use fast CPU GEMV
    if A.device.type == "cpu":
        if getattr(quant_state, "packing_format_for_cpu", False):
            out = F.gemv_4bit(A, B, out, state=quant_state)
            if bias is not None:
                out += bias
            return out
        else:
            return MatMul4Bit.apply(A, B, out, bias, quant_state)

    # GPU: single-batch, no-grad, non-HPU fast path
    if (A.numel() == A.shape[-1] and
        A.requires_grad == False and
        A.device.type != "hpu"):
        if A.shape[-1] % quant_state.blocksize != 0:
            warn(f"Some matrices hidden dimension is not a multiple of "
                 f"{quant_state.blocksize} and efficient inference kernels "
                 f"are not supported for these (slow). Matrix input size found: {A.shape}")
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)
```

### Dispatch Logic

```
matmul_4bit(A, B, quant_state, ...)
    |
    +-- quant_state is None? -> AssertionError
    |
    +-- CPU device?
    |       |
    |       +-- packing_format_for_cpu? -> gemv_4bit (AVX-512 BF16 kernel)
    |       |
    |       +-- No packing -> MatMul4Bit (dequantize + F.linear)
    |
    +-- GPU device?
            |
            +-- Single-batch (A.numel() == A.shape[-1])
            |      AND no grad (A.requires_grad == False)
            |      AND not HPU?
            |       |
            |       +-- A.shape[-1] % blocksize != 0?
            |       |       -> Warning + MatMul4Bit (slow)
            |       |
            |       +-- Divisible by blocksize?
            |               -> gemv_4bit (fast 4-bit GEMV kernel)
            |
            +-- Batched or requires grad or HPU?
                    -> MatMul4Bit (dequantize + F.linear)
```

### When Each Path Is Used

| Condition | Function | Rationale |
|-----------|----------|-----------|
| CPU + packed format | `gemv_4bit` | AVX-512 BF16 accelerated CPU inference |
| CPU + no packing | `MatMul4Bit` | Fallback: dequantize then F.linear |
| GPU + single vector + no grad + divisible blocksize | `gemv_4bit` | Fast 4-bit GEMV CUDA kernel |
| GPU + batched or needs grad | `MatMul4Bit` | Full dequantize for correctness |
| HPU (Habana) | `MatMul4Bit` | No fast GEMV kernel available |

### Blocksize Divisibility Check

The fast `gemv_4bit` path requires that the hidden dimension (`A.shape[-1]`) be a multiple of `quant_state.blocksize` (default 64). If not, a warning is issued and the slower `MatMul4Bit` path is used:

```python
if A.shape[-1] % quant_state.blocksize != 0:
    warn(f"Some matrices hidden dimension is not a multiple of {quant_state.blocksize} ...")
    return MatMul4Bit.apply(A, B, out, bias, quant_state)
```

---

## MatmulLtState Dataclass

`MatmulLtState` is a Python dataclass that manages the quantization state for `MatMul8bitLt` and `MatMul8bitFp`. It caches quantized weight tensors and their statistics across forward passes.

**Location:** `bitsandbytes/autograd/_functions.py`

### All Fields

```python
@dataclass
class MatmulLtState:
    force_no_igemmlt: bool = False
        # If True, skip the igemmlt integer GEMM path and use standard matmul

    CB: Optional[torch.Tensor] = None
        # Quantized (int8) version of weight matrix B (row-major)
        # Shape: [out_features, in_features], dtype: int8

    SB: Optional[torch.Tensor] = None
        # (Unused in current implementation, kept for compatibility)

    SCB: Optional[torch.Tensor] = None
        # Row-wise absolute maximum statistics for CB
        # Shape: [out_features], dtype: float32
        # Used for dequantization: dequant = CB * SCB / 127

    SBt: Optional[torch.Tensor] = None
        # (Unused in current implementation)

    CBt: Optional[torch.Tensor] = None
        # (Unused in current implementation)

    subB: Optional[torch.Tensor] = None
        # (Unused in current implementation)

    outlier_pool: Optional[GlobalOutlierPooler] = None
        # Optional shared pooler for cross-layer outlier tracking

    has_accumulated_gradients: bool = False
        # (Flag for gradient accumulation, currently unused)

    threshold: float = 0.0
        # Outlier threshold for mixed-precision decomposition (LLM.int8())
        # Columns where any activation exceeds this value are computed in fp16
        # Set to 0.0 to disable mixed-precision (all columns quantized to int8)

    idx: Optional[torch.Tensor] = None
        # Indices of outlier columns detected during the current forward pass

    is_training: bool = True
        # Whether the parent module is in training mode
        # Affects caching behavior and quantization path selection

    has_fp16_weights: bool = True
        # If True, weights remain in fp16 and are quantized each forward pass
        # If False, weights are quantized once and cached in CB/SCB
        # Linear8bitLt sets this based on the has_fp16_weights constructor arg

    use_pool: bool = False
        # If True, use the GlobalOutlierPooler for cross-layer outlier sharing
        # Automatically set to True when threshold > 0 and has_fp16_weights is False
```

### Deprecated Fields

The following fields are no longer used but are intercepted via `__getattr__` for backward compatibility with downstream libraries (TGI, vLLM):

```python
_deprecated_fields = frozenset({"CxB", "CxBt", "formatB", "_tile_indices"})

def __getattr__(self, name):
    if name in MatmulLtState._deprecated_fields:
        warnings.warn(
            f"MatmulLtState.{name} is deprecated and will be removed "
            f"in the next bitsandbytes release.",
            FutureWarning,
            stacklevel=2,
        )
        return None
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

### reset_grads()

Clears all cached quantized tensors. Called before re-quantizing B in the forward pass:

```python
def reset_grads(self):
    self.CB = None
    self.SB = None
    self.SCB = None
    self.SBt = None
    self.CBt = None
```

### Usage in Linear8bitLt

The `MatmulLtState` is created and configured by `Linear8bitLt.__init__()`:

```python
class Linear8bitLt(nn.Linear):
    def __init__(self, input_features, output_features, bias=True,
                 has_fp16_weights=True, threshold=0.0, index=None, device=None):
        super().__init__(input_features, output_features, bias, device)
        self.state = bnb.MatmulLtState()
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights

        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
```

In the forward pass, `state.is_training` is updated to reflect the module's training mode:

```python
def forward(self, x):
    self.state.is_training = self.training
    # ...
    out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
```

---

## GlobalOutlierPooler

`GlobalOutlierPooler` is a singleton that pools outlier feature dimensions across layers. This is particularly important for small models where outlier features are less systematic and occur with low frequency.

**Location:** `bitsandbytes/autograd/_functions.py`

```python
class GlobalOutlierPooler:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.outliers = set()           # Set of outlier column indices (accumulated)
        self.model_dim = None           # Expected feature dimension (for validation)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def add_outliers(self, outlier_idx, feature_dim):
        if self.model_dim is None:
            self.model_dim = feature_dim
        if feature_dim != self.model_dim:
            return  # Do not encode outliers for different FFN layers
        self.outliers.update(outlier_idx.tolist())

    def get_current_outlier_idx(self):
        return torch.Tensor(list(self.outliers)).to(torch.int64)
```

### Outlier Pooling Strategy

The pooler accumulates outlier column indices across sequential linear layers. For small models, outlier features may not be consistent within a single layer but can be identified across layers. By pooling, the threshold for outlier detection becomes more robust.

The `model_dim` check ensures outliers are only pooled for layers with the same hidden dimension (typically the first FFN layer), avoiding contamination from layers with different dimensions (e.g., the second FFN projection).

---

## Summary: Function Selection Guide

### For 8-bit Quantization

| Scenario | Function | Reason |
|----------|----------|--------|
| GPU training (fp16 weights) | `MatMul8bitLt` | Full int8 path with outlier support |
| GPU training (int8 cached weights) | `MatMul8bitLt` | Cached CB/SCB for efficiency |
| CPU/XPU training | `MatMul8bitFp` | ~3x faster on CPU/XPU |
| GPU inference | `MatMul8bitLt` | Cached quantized weights |
| Mixed-precision (threshold > 0) | `MatMul8bitLt` (mixed path) | Outlier columns computed in fp16 |

### For 4-bit Quantization

| Scenario | Function | Reason |
|----------|----------|--------|
| CPU inference (AVX-512 BF16) | `gemv_4bit` | Fast CPU kernel |
| GPU single-batch inference | `gemv_4bit` | Fast CUDA 4-bit GEMV |
| GPU batched inference | `MatMul4Bit` | Dequantize + F.linear |
| GPU training (QLoRA) | `MatMul4Bit` | Dequantize for grad_A; grad_B not supported |
| Blocksize not divisible | `MatMul4Bit` | gemv_4bit requires divisible blocksize |
