# PyTorch - Chapter 5: Tensor Types and Device Management

This reference covers all data types (dtypes), type promotion rules, device management, layouts, memory formats, and symbolic shape types.

---

## 5.1 Data Types (dtypes)

### Complete dtype Listing

| dtype | Description | Size (bytes) | Alias |
|-------|-------------|-------------|-------|
| `torch.float16` | 16-bit floating point (half) | 2 | `torch.half` |
| `torch.float32` | 32-bit floating point (single) | 4 | `torch.float` |
| `torch.float64` | 64-bit floating point (double) | 8 | `torch.double` |
| `torch.bfloat16` | Brain floating point | 2 | - |
| `torch.float8_e4m3fn` | 8-bit float (E4M3) | 1 | - |
| `torch.float8_e5m2` | 8-bit float (E5M2) | 1 | - |
| `torch.float8_e4m3fnuz` | 8-bit float (E4M3, unsigned zero) | 1 | - |
| `torch.float8_e5m2fnuz` | 8-bit float (E5M2, unsigned zero) | 1 | - |
| `torch.int8` | 8-bit signed integer | 1 | - |
| `torch.int16` | 16-bit signed integer | 2 | `torch.short` |
| `torch.int32` | 32-bit signed integer | 4 | `torch.int` |
| `torch.int64` | 64-bit signed integer | 8 | `torch.long` |
| `torch.uint8` | 8-bit unsigned integer | 1 | `torch.byte` |
| `torch.uint16` | 16-bit unsigned integer | 2 | - |
| `torch.uint32` | 32-bit unsigned integer | 4 | - |
| `torch.uint64` | 64-bit unsigned integer | 8 | - |
| `torch.bool` | Boolean | 1 | - |
| `torch.complex32` | 32-bit complex (2x float16) | 4 | - |
| `torch.complex64` | 64-bit complex (2x float32) | 8 | `torch.cfloat` |
| `torch.complex128` | 128-bit complex (2x float64) | 16 | `torch.cdouble` |

### Quantized dtypes

| dtype | Description |
|-------|-------------|
| `torch.qint8` | 8-bit signed quantized integer |
| `torch.quint8` | 8-bit unsigned quantized integer |
| `torch.qint32` | 32-bit signed quantized integer |
| `torch.quint4x2` | 4-bit unsigned quantized (2 per byte) |
| `torch.quint2x4` | 2-bit quantized (4 per byte) |

### Type Properties

```python
torch.float32.is_floating_point    # True
torch.float32.is_complex           # False
torch.complex64.is_complex         # True
torch.int64.itemsize               # 8
torch.int64.dtype                  # torch.int64
```

---

## 5.2 Type Promotion Rules

PyTorch follows these rules for implicit type promotion when mixing dtypes in operations:

### Simplified Rules

1. If both operands have the same type, no promotion needed.
2. If one operand is complex, the result is complex (with the wider float type).
3. If one operand is floating and the other integral, result is the floating type.
4. Between integers, promote to the wider integer type.
5. `bool` promotes to any integer or float type.

### Promotion Table (Common Cases)

| A \ B | bool | int8 | int16 | int32 | int64 | float16 | float32 | float64 | bfloat16 | complex64 | complex128 |
|-------|------|------|-------|-------|-------|---------|---------|---------|----------|-----------|------------|
| bool | bool | int8 | int16 | int32 | int64 | float16 | float32 | float64 | bfloat16 | complex64 | complex128 |
| int8 | int8 | int8 | int16 | int32 | int64 | float16 | float32 | float64 | bfloat16 | complex64 | complex128 |
| int16 | int16 | int16 | int16 | int32 | int64 | float16 | float32 | float64 | bfloat16 | complex64 | complex128 |
| int32 | int32 | int32 | int32 | int32 | int64 | float16 | float32 | float64 | bfloat16 | complex64 | complex128 |
| int64 | int64 | int64 | int64 | int64 | int64 | float16 | float32 | float64 | bfloat16 | complex64 | complex128 |
| float16 | float16 | float16 | float16 | float16 | float16 | float16 | float32 | float64 | float32 | complex64 | complex128 |
| float32 | float32 | float32 | float32 | float32 | float32 | float32 | float32 | float64 | float32 | complex64 | complex128 |
| float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | complex128 | complex128 |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | float32 | float32 | float64 | bfloat16 | complex64 | complex128 |

### Explicit Casting

```python
# Via .to()
t = torch.tensor([1, 2, 3])
t = t.to(torch.float32)

# Via shorthand methods
t = t.float()        # → float32
t = t.double()       # → float64
t = t.half()         # → float16
t = t.bfloat16()     # → bfloat16
t = t.int()          # → int32
t = t.long()         # → int64
t = t.short()        # → int16
t = t.byte()         # → uint8
t = t.char()         # → int8
t = t.bool()         # → bool

# Via creation
t = torch.tensor([1.0, 2.0], dtype=torch.float16)
```

---

## 5.3 Device Management

### torch.device

```python
device = torch.device('cuda:0')     # CUDA device 0
device = torch.device('cuda')       # Current CUDA device
device = torch.device('cpu')        # CPU
device = torch.device('xpu')        # Intel XPU
device = torch.device('mps')        # Apple Metal

device.type    # 'cuda', 'cpu', 'xpu', 'mps'
device.index   # 0, 1, ... or None
```

### Default Device

```python
torch.set_default_device('cuda')    # New tensors default to CUDA
torch.get_default_device()          # Returns current default device

torch.set_default_tensor_type(torch.FloatTensor)  # Set default type and device (legacy)
torch.get_default_dtype()           # Returns torch.float32
torch.set_default_dtype(torch.float64)
```

### CUDA Device Management

```python
torch.cuda.is_available()           # True if CUDA is available
torch.cuda.is_initialized()         # True if CUDA is initialized
torch.cuda.device_count()           # Number of GPUs
torch.cuda.current_device()         # Current device index
torch.cuda.set_device(device)       # Set current device

# Device context manager
with torch.cuda.device(1):
    # All operations on device 1
    x = torch.randn(3, 3)  # On cuda:1

# Device properties
props = torch.cuda.get_device_properties(0)
props.name           # 'NVIDIA A100-SXM4-80GB'
props.total_memory   # 85167349760
props.major          # 8 (compute capability major)
props.minor          # 0

# Memory info
torch.cuda.memory_allocated()       # Current GPU memory used by tensors
torch.cuda.max_memory_allocated()   # Peak GPU memory
torch.cuda.memory_reserved()        # Memory held by caching allocator
torch.cuda.max_memory_reserved()    # Peak reserved memory
torch.cuda.empty_cache()            # Release cached memory back to CUDA

# Synchronize
torch.cuda.synchronize()            # Wait for all CUDA ops to finish
torch.cuda.synchronize(device=0)    # Wait for specific device
```

### XPU Device Management

```python
torch.xpu.is_available()            # True if Intel XPU available
torch.xpu.device_count()
torch.xpu.current_device()
torch.xpu.set_device(device)
torch.xpu.get_device_properties(device)
```

### MPS Device Management

```python
torch.backends.mps.is_available()   # True if Apple Metal available
torch.backends.mps.is_built()       # True if PyTorch was built with MPS
```

---

## 5.4 Layout Types

| Layout | Description | Use Case |
|--------|-------------|----------|
| `torch.strided` | Dense strided tensor (default) | Most operations |
| `torch.sparse_coo` | COOrdinate sparse format | Sparse matrices |
| `torch.sparse_csr` | Compressed Sparse Row | Sparse matrix operations |
| `torch.sparse_csc` | Compressed Sparse Column | Sparse matrix operations |
| `torch.sparse_bsr` | Block Sparse Row | Block sparse matrices |
| `torch.sparse_bsc` | Block Sparse Column | Block sparse matrices |

```python
t = torch.randn(3, 3, layout=torch.strided)  # Default
s = t.to_sparse()                              # Convert to sparse COO
s = t.to_sparse_csr()                          # Convert to sparse CSR
```

---

## 5.5 Memory Formats

```python
torch.contiguous_format    # Default: row-major (C-contiguous)
torch.channels_last        # (N, C, H, W) → strides based on NHWC
torch.preserve_format      # Keep the format of input tensor
```

```python
# Contiguous (default)
t = torch.randn(1, 3, 224, 224)
t.is_contiguous(memory_format=torch.contiguous_format)  # True
t.stride()  # (150528, 50176, 224, 1)

# Channels last
cl = t.contiguous(memory_format=torch.channels_last)
cl.is_contiguous(memory_format=torch.channels_last)  # True
cl.stride()  # (150528, 1, 224, 672)
```

**When to use channels_last**: Convolution operations can be faster with channels_last format on GPU because it matches the memory layout expected by cuDNN.

---

## 5.6 Symbolic Shape Types (SymInt, SymFloat, SymBool)

PyTorch uses symbolic integers and floats to represent dynamic shapes:

### SymInt

```python
# Created internally by torch.compile for dynamic shapes
# Can be used in comparisons and arithmetic
from torch import SymInt, SymFloat, SymBool

# User-facing APIs
torch.sym_int(x)           # Convert to SymInt if needed
torch.sym_float(x)         # Convert to SymFloat
torch.sym_max(a, b)        # Symbolic max (avoids guard on which is larger)
torch.sym_min(a, b)        # Symbolic min
torch.sym_ite(pred, a, b)  # Symbolic if-then-else
torch.sym_not(pred)        # Symbolic not
torch.sym_sum(xs)          # Symbolic sum of list

# SymInt behaves like int for most operations
s = torch SymInt(5)
s + 3       # SymInt(8)
s * 2       # SymInt(10)
```

### Usage in torch.compile

When using `torch.compile` with dynamic shapes, PyTorch uses SymInt to represent symbolic dimension sizes:

```python
@torch.compile(dynamic=True)
def f(x):
    # x.shape[0] is a SymInt here
    batch_size = x.shape[0]
    return x[:batch_size // 2]
```

---

## 5.7 Device-Agnostic Code Patterns

```python
# Pattern 1: Auto-detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# Pattern 2: Use default device
torch.set_default_device('cuda')

# Pattern 3: Module.to()
model = model.cuda() if torch.cuda.is_available() else model

# Pattern 4: Device-aware DataLoader
dataloader = DataLoader(dataset, pin_memory=torch.cuda.is_available())
```
