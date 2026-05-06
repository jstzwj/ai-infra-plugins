# Tensor Expression (TE) and TOPI

This reference covers TVM's Tensor Expression (TE) DSL and the Tensor Operator Inventory (TOPI). TE provides a domain-specific language for describing tensor computations at a high level, while TOPI provides a comprehensive library of pre-defined operators for common machine learning workloads. Together, they form TVM's legacy operator definition layer, which is still widely used for defining custom operators and for understanding how TVM generates code for standard operations.

---

## 21.1 Tensor Expression DSL (tvm.te)

### 21.1.1 Overview

The Tensor Expression (TE) DSL is TVM's original high-level language for describing tensor computations. TE allows users to define computations declaratively without specifying low-level implementation details like loop order, memory layout, or thread mapping. The TE description is then lowered to a TensorIR PrimFunc, which can be scheduled and compiled.

Key characteristics of TE:

- **Declarative**: Describe what to compute, not how
- **Composable**: Build complex computations from simple building blocks
- **Convertible to PrimFunc**: Use `te.create_prim_func` to bridge to TensorIR
- **Legacy but supported**: TE predates the direct TensorIR/TVMScript workflow but remains fully supported

**Important**: TE itself is NOT a self-contained executable function. A TE computation must be converted to a `tir.PrimFunc` via `te.create_prim_func` before it can be scheduled and compiled.

### 21.1.2 Core Concepts

TE computations are built from three fundamental building blocks:

1. **Placeholder**: Declares an input tensor with a shape and dtype
2. **Compute**: Declares an output tensor defined by a lambda function over indices
3. **Extern**: Declares an operation implemented by external code

These building blocks form a computation graph (a DAG of tensors), where each tensor depends on zero or more other tensors.

### 21.1.3 te.placeholder

`te.placeholder` declares an input tensor with a given shape, data type, and name.

```python
import tvm
from tvm import te

# Basic placeholder: 2D float32 tensor
A = te.placeholder((128, 128), name="A", dtype="float32")

# 4D tensor (e.g., image batch in NCHW format)
data = te.placeholder((1, 3, 224, 224), name="data", dtype="float32")

# 1D vector
bias = te.placeholder((128,), name="bias", dtype="float32")

# Scalar (0-dimensional)
scale = te.placeholder((), name="scale", dtype="float32")

# Mixed precision: float16 inputs
X = te.placeholder((4096, 4096), name="X", dtype="float16")
```

**Placeholder properties:**

| Property | Access | Description |
|----------|--------|-------------|
| `A.shape` | Read | Shape tuple of the tensor |
| `A.dtype` | Read | Data type string (e.g., "float32") |
| `A.name` | Read | Name of the tensor |
| `A.op` | Read | The operation that produces this tensor (a PlaceholderOp) |
| `A.value_index` | Read | Index into the operation's outputs |

Placeholders do not define any computation -- they represent external inputs whose values will be provided at runtime.

### 21.1.4 te.compute

`te.compute` declares an output tensor by specifying its shape and a lambda function that computes each element.

```python
import tvm
from tvm import te

# Input
A = te.placeholder((128, 128), name="A", dtype="float32")

# Element-wise multiply by 2
B = te.compute(
    (128, 128),
    lambda i, j: A[i, j] * 2.0,
    name="B"
)

# Element-wise add with scalar
C = te.compute(
    (128, 128),
    lambda i, j: B[i, j] + 1.0,
    name="C"
)

# Reduction: sum over axis 1
D = te.compute(
    (128,),
    lambda i: te.sum(A[i, j], axis=[te.axis(128, "j")]),
    name="D"
)
```

**Signature:**

```python
te.compute(
    shape,           # Output tensor shape (tuple of ints or tvm.tir.Var)
    fcompute,        # Lambda function: indices -> value
    name="compute",  # Name of the output tensor
    tag="",          # Optional tag for pattern matching
    attrs=None,      # Optional attributes dict
    varargs_names=None,  # Variable argument names
)
```

**Compute with reduction:**

```python
# Matrix multiplication: C[i, j] = sum_k A[i, k] * B[k, j]
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")

k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)
```

**Multiple reductions:**

```python
# Compute both sum and sum of squares simultaneously
rk = te.reduce_axis((0, K), name="k")
sum_val = te.compute(
    (M,),
    lambda i: te.sum(A[i, rk], axis=rk),
    name="sum_val"
)

sum_sq_val = te.compute(
    (M,),
    lambda i: te.sum(A[i, rk] * A[i, rk], axis=rk),
    name="sum_sq_val"
)
```

**Conditional compute:**

```python
# ReLU: max(0, x)
relu_out = te.compute(
    (128, 128),
    lambda i, j: tvm.tir.if_then_else(
        A[i, j] > 0, A[i, j], tvm.tir.const(0, "float32")
    ),
    name="relu"
)

# Clipped ReLU: min(max(0, x), 6.0)
clipped_relu = te.compute(
    (128, 128),
    lambda i, j: tvm.tir.max(
        tvm.tir.const(0, "float32"),
        tvm.tir.min(A[i, j], tvm.tir.const(6, "float32"))
    ),
    name="clipped_relu"
)
```

**Dynamic shape compute:**

```python
# Using symbolic dimensions
M = te.size_var("M")
N = te.size_var("N")
K = te.size_var("K")

A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")

rk = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, rk] * B[rk, j], axis=rk),
    name="C"
)
```

### 21.1.5 te.reduce_axis

`te.reduce_axis` declares a reduction axis that iterates over a range and is used inside `te.compute` with reduction functions like `te.sum`, `te.max`, `te.min`.

```python
# Basic reduction axis over range [0, 128)
rk = te.reduce_axis((0, 128), name="k")

# Reduction axis with symbolic bounds
K = te.size_var("K")
rk = te.reduce_axis((0, K), name="k")
```

**Reduction functions available:**

| Function | Operation | Identity element |
|----------|-----------|------------------|
| `te.sum(expr, axis)` | Summation | 0 |
| `te.max(expr, axis)` | Maximum | -inf |
| `te.min(expr, axis)` | Minimum | +inf |
| `te.prod(expr, axis)` | Product | 1 |

### 21.1.6 te.scan

`te.scan` defines recurrent/scan operations where each output element depends on previous output elements.

```python
# Cumulative sum: out[i] = out[i-1] + in[i]
n = te.size_var("n")
X = te.placeholder((n,), name="X", dtype="float32")

s_state = te.placeholder((n,), name="s_state", dtype="float32")
s_init = te.compute((1,), lambda _: tvm.tir.const(0, "float32"), name="s_init")
s_update = te.compute(
    (n,),
    lambda i: s_state[i - 1] + X[i],
    name="s_update"
)

scan_out = te.scan(
    s_init,     # Initial state (shape: [1])
    s_update,   # Update function
    s_state,    # State placeholder
    name="cumsum"
)
```

**Signature:**

```python
te.scan(
    init,        # Initial state tensors (list or single tensor)
    update,      # Update function tensors (list or single tensor)
    state_placeholder,  # State placeholder tensors
    inputs=[],   # Additional input tensors
    name="scan", # Name
    tag="",      # Tag
)
```

### 21.1.7 te.extern

`te.extern` defines operations that call external functions (e.g., cuBLAS, cuDNN, custom CUDA kernels).

```python
# Call an external function for matrix multiplication
C = te.extern(
    shape=(M, N),
    inputs=[A, B],
    fcompute=lambda ins, outs: tvm.tir.call_packed(
        "my_custom_matmul",
        ins[0], ins[1], outs[0], M, N, K
    ),
    name="C",
    dtype="float32",
)
```

**Signature:**

```python
te.extern(
    shape,           # Output shape
    inputs,          # List of input tensors
    fcompute,        # Function: (ins, outs) -> list of Stmt
    name="extern",   # Name
    dtype=None,      # Output dtype
    in_buffers=None, # Explicit input buffers
    out_buffers=None,# Explicit output buffers
    tag="",          # Tag
    attrs=None,      # Attributes
)
```

**External function with multiple outputs:**

```python
# Compute both sum and max in a single external call
results = te.extern(
    [(M,), (M,)],  # Two outputs: sum and max
    [A],
    fcompute=lambda ins, outs: tvm.tir.call_packed(
        "sum_and_max",
        ins[0], outs[0], outs[1], M, N
    ),
    name=["sum_out", "max_out"],
    dtype=["float32", "float32"],
)
sum_out, max_out = results
```

### 21.1.8 te.create_prim_func

`te.create_prim_func` converts a TE computation graph into a `tir.PrimFunc`, which can then be scheduled using TensorIR schedule primitives or included in an IRModule.

```python
import tvm
from tvm import te

A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.compute((128, 128), lambda i, j: A[i, j] * 2.0, name="B")
C = te.compute((128, 128), lambda i, j: B[i, j] + 1.0, name="C")

# Convert to PrimFunc
# The argument list includes all leaf inputs and the final output
prim_func = te.create_prim_func([A, C])

# Print the generated TIR
print(prim_func)
```

**Important details about `te.create_prim_func`:**

1. The argument list should include all **input placeholders** and the **final output tensor**. Intermediate tensors (like B above) are automatically inlined or allocated as local buffers.

2. The generated PrimFunc uses `T.block` and `T.axis.remap` notation, making it compatible with DLight, MetaSchedule, and manual TensorIR scheduling.

3. If the output tensor appears in the argument list, it is treated as an output parameter. If not, it is allocated internally.

**Example: Full matrix multiply to PrimFunc:**

```python
M, N, K = 1024, 1024, 1024

A = te.placeholder((M, K), name="A", dtype="float16")
B = te.placeholder((K, N), name="B", dtype="float16")

k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(
        A[i, k].astype("float32") * B[k, j].astype("float32"),
        axis=k
    ),
    name="C"
)

# Create PrimFunc with inputs and output
prim_func = te.create_prim_func([A, B, C])

# Print the TIR
print(prim_func.script())
```

This generates a PrimFunc equivalent to:

```python
@T.prim_func
def main(
    A: T.Buffer((1024, 1024), "float16"),
    B: T.Buffer((1024, 1024), "float16"),
    C: T.Buffer((1024, 1024), "float32"),
):
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            T.reads(A[vi, vk], B[vk, vj])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")
```

### 21.1.9 TE to IRModule Workflow

The complete workflow from TE definition to compiled executable:

```python
import tvm
from tvm import te

# Step 1: Define computation with TE
A = te.placeholder((1024, 1024), name="A", dtype="float32")
B = te.placeholder((1024, 1024), name="B", dtype="float32")
k = te.reduce_axis((0, 1024), name="k")
C = te.compute(
    (1024, 1024),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

# Step 2: Convert to PrimFunc
prim_func = te.create_prim_func([A, B, C])

# Step 3: Wrap in IRModule
mod = tvm.IRModule.from_expr(prim_func)

# Step 4: Apply scheduling (DLight, MetaSchedule, or manual)
from tvm import dlight as dl
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Fallback())(mod)

# Step 5: Build
target = tvm.target.Target("nvidia/nvidia-a100")
exec_mod = tvm.build(mod, target=target)

# Step 6: Run
import numpy as np
dev = tvm.cuda(0)
a_np = np.random.randn(1024, 1024).astype("float32")
b_np = np.random.randn(1024, 1024).astype("float32")
a_tvm = tvm.nd.array(a_np, dev)
b_tvm = tvm.nd.array(b_np, dev)
c_tvm = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

exec_mod(a_tvm, b_tvm, c_tvm)
```

### 21.1.10 TE Scheduling (Legacy)

TE has its own scheduling API (`te.schedule.Schedule`) which predates the TensorIR `tir.Schedule`. While still functional, the TensorIR schedule is recommended for new code.

**Legacy TE schedule operations:**

```python
# Create a schedule for the computation graph
s = te.create_schedule(C.op)

# Split a loop into two levels
i, j = s[C].op.axis
i_outer, i_inner = s[C].split(i, factor=32)
j_outer, j_inner = s[C].split(j, factor=32)

# Reorder loops
s[C].reorder(i_outer, j_outer, i_inner, j_inner)

# Fuse loops
fused = s[C].fuse(i_outer, j_outer)

# Bind to GPU threads
s[C].bind(fused, te.thread_axis("blockIdx.x"))
s[C].bind(i_inner, te.thread_axis("threadIdx.x"))
s[C].bind(j_inner, te.thread_axis("threadIdx.y"))

# Compute at (inline into a consumer)
s[B].compute_at(s[C], i_inner)

# Compute inline (fully inline a stage)
s[B].compute_inline()
```

**Migration from legacy TE schedule to TensorIR schedule:**

```python
# OLD: Legacy TE schedule
s = te.create_schedule(C.op)
# ... apply schedule primitives ...

# Build
func = tvm.build(s, [A, B, C], target="cuda")

# NEW: Convert to PrimFunc and use TensorIR schedule
prim_func = te.create_prim_func([A, B, C])
mod = tvm.IRModule.from_expr(prim_func)

# Apply TensorIR schedule (or DLight)
sch = tvm.tir.Schedule(mod)
block = sch.get_block("C")
i, j, k = sch.get_loops(block)
# ... apply TensorIR schedule primitives ...

# Or use DLight directly
from tvm import dlight as dl
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
```

### 21.1.11 Relationship to TensorIR

The relationship between TE and TensorIR is fundamental to understanding TVM's compilation pipeline:

```
+------------------+       te.create_prim_func       +------------------+
|  TE Computation  |  --------------------------------> |  tir.PrimFunc    |
|  (Declarative)   |                                  |  (Imperative)    |
|  WHAT to compute |                                  |  HOW to compute  |
+------------------+                                  +------------------+
                                                             |
                                                             v
                                                      +------------------+
                                                      | TensorIR Schedule|
                                                      | (DLight/Meta)    |
                                                      +------------------+
                                                             |
                                                             v
                                                      +------------------+
                                                      | Code Generation  |
                                                      | (LLVM, CUDA,     |
                                                      |  Metal, etc.)    |
                                                      +------------------+
```

- **TE describes WHAT to compute**: The lambda functions define the mathematical relationships between inputs and outputs.
- **TensorIR describes HOW to compute**: The PrimFunc includes loop structure, memory access patterns, and scheduling information.
- **`te.create_prim_func` bridges the gap**: It generates a default PrimFunc with a straightforward loop nest that can then be optimized by scheduling.

---

## 21.2 TOPI -- Tensor Operator Inventory (tvm.topi)

### 21.2.1 Overview

TOPI (Tensor Operator Inventory) is TVM's library of pre-defined operators for common machine learning workloads. It provides both generic (device-independent) and backend-specific implementations of standard operations. TOPI uses the TE DSL internally and can be used to quickly construct complex neural network operations.

**Key features:**

- **Generic implementations**: Work on any supported backend (CPU, GPU, etc.)
- **Backend-specific implementations**: Optimized for CUDA, ROCm, Metal, Vulkan, etc.
- **Consistent API**: Same operator name across all backends
- **Schedule integration**: Each operator comes with a suggested schedule

### 21.2.2 Neural Network Operators (topi.nn)

#### Convolution Operators

```python
import tvm
from tvm import te, topi

# conv2d: Standard 2D convolution
data = te.placeholder((1, 3, 224, 224), name="data", dtype="float32")
kernel = te.placeholder((32, 3, 3, 3), name="kernel", dtype="float32")
conv = topi.nn.conv2d(
    data,
    kernel,
    strides=1,          # Can be int or tuple
    padding=1,          # Can be int, tuple, or "VALID"/"SAME"
    dilation=1,
    data_layout="NCHW", # Default
    kernel_layout="OIHW", # Default
    out_dtype="float32",
)
```

**conv2d parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | Tensor | required | Input data tensor |
| `kernel` | Tensor | required | Weight tensor |
| `strides` | int or tuple | 1 | Spatial stride |
| `padding` | int or tuple or str | 0 | Spatial padding |
| `dilation` | int or tuple | 1 | Dilation rate |
| `data_layout` | str | "NCHW" | Data layout |
| `kernel_layout` | str | "OIHW" | Kernel layout |
| `out_dtype` | str | None | Output data type |

**Convolution variants:**

```python
# conv1d: 1D convolution (for text, time series)
conv1d_out = topi.nn.conv1d(data_1d, kernel_1d, strides=1, padding=0)

# conv3d: 3D convolution (for video, medical imaging)
conv3d_out = topi.nn.conv3d(data_3d, kernel_3d, strides=1, padding=0)

# conv2d_transpose: Transposed (deconvolution) 2D convolution
conv_t = topi.nn.conv2d_transpose(
    data, kernel,
    strides=2,
    padding=0,
    output_padding=(1, 1),
)

# conv1d_transpose, conv3d_transpose: Transposed 1D and 3D
conv1d_t = topi.nn.conv1d_transpose(data_1d, kernel_1d, strides=2)
conv3d_t = topi.nn.conv3d_transpose(data_3d, kernel_3d, strides=2)

# depthwise_conv2d: Depthwise separable convolution
dw_conv = topi.nn.depthwise_conv2d(
    data, depthwise_kernel,
    strides=1,
    padding=1,
    dilation=1,
)

# group_conv2d: Grouped convolution
grp_conv = topi.nn.group_conv2d(
    data, kernel,
    strides=1,
    padding=1,
    groups=4,
)

# deformable_conv2d: Deformable convolution
def_conv = topi.nn.deformable_conv2d(
    data, offset, kernel,
    strides=1,
    padding=1,
    dilation=1,
    deformable_groups=1,
)
```

#### Dense (Fully Connected) Operators

```python
# dense: Standard dense/fully connected layer
# data: [batch, in_dim], weight: [out_dim, in_dim], out: [batch, out_dim]
data = te.placeholder((128, 784), name="data", dtype="float32")
weight = te.placeholder((10, 784), name="weight", dtype="float32")
dense_out = topi.nn.dense(data, weight, out_dtype="float32")

# With bias
bias = te.placeholder((10,), name="bias", dtype="float32")
dense_biased = topi.nn.bias_add(dense_out, bias)

# sparse_dense: Sparse matrix multiplication
# sparse_data: sparse values, sparse_indices: column indices, sparse_ptr: row pointers
sparse_out = topi.nn.sparse_dense(
    sparse_data, weight, sparse_indices, sparse_ptr
)
```

#### Activation Functions

```python
# relu: Rectified linear unit
relu_out = topi.nn.relu(data)

# leaky_relu: Leaky ReLU with configurable slope
leaky_out = topi.nn.leaky_relu(data, alpha=0.01)

# prelu: Parametric ReLU (learnable slope per channel)
prelu_out = topi.nn.prelu(data, alpha, axis=1)

# sigmoid: 1 / (1 + exp(-x))
sigmoid_out = topi.sigmoid(data)

# tanh: Hyperbolic tangent
tanh_out = topi.tanh(data)

# softmax: Softmax along specified axis
softmax_out = topi.nn.softmax(data, axis=1)

# log_softmax: Logarithm of softmax (numerically stable)
log_softmax_out = topi.nn.log_softmax(data, axis=1)

# gelu: Gaussian Error Linear Unit (approximation)
# gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
# Note: gelu may be available via topi.nn or topi.math depending on version
```

#### Normalization Operators

```python
# batch_norm: Batch normalization
# gamma, beta: [channels], mean, var: [channels]
bn_out = topi.nn.batch_norm(
    data, gamma, beta, mean, var,
    axis=1,        # Channel axis
    epsilon=1e-5,
    center=True,
    scale=True,
)

# layer_norm: Layer normalization
# gamma, beta: [normalized_shape]
ln_out = topi.nn.layer_norm(
    data, gamma, beta,
    axis=-1,       # Normalization axis
    epsilon=1e-5,
)

# instance_norm: Instance normalization
in_out = topi.nn.instance_norm(
    data, gamma, beta,
    axis=1,
    epsilon=1e-5,
)

# group_norm: Group normalization
gn_out = topi.nn.group_norm(
    data, gamma, beta,
    num_groups=32,
    axis=1,
    epsilon=1e-5,
)

# lrn: Local Response Normalization
lrn_out = topi.nn.lrn(data, size=5, axis=1, bias=1.0, alpha=0.001, beta=0.75)

# l2_normalize: L2 normalization along an axis
l2_out = topi.nn.l2_normalize(data, eps=1e-10, axis=1)
```

#### Pooling Operators

```python
# pool2d: 2D max/avg pooling
max_pool = topi.nn.pool2d(
    data,
    kernel_size=(3, 3),
    stride=(2, 2),
    dilation=(1, 1),
    padding=(1, 1),
    pool_type="max",
)

avg_pool = topi.nn.pool2d(
    data,
    kernel_size=(3, 3),
    stride=(2, 2),
    dilation=(1, 1),
    padding=(1, 1),
    pool_type="avg",
)

# pool1d, pool3d: 1D and 3D pooling
max_pool1d = topi.nn.pool1d(data_1d, kernel_size=3, stride=2, pool_type="max")
max_pool3d = topi.nn.pool3d(data_3d, kernel_size=3, stride=2, pool_type="max")

# adaptive_pool: Adaptive pooling (output size specified)
adaptive_avg = topi.nn.adaptive_pool(data, output_size=(7, 7), pool_type="avg")

# global_pool: Global pooling (output size = 1x1)
global_avg = topi.nn.global_pool(data, pool_type="avg")
```

#### Spatial Operations

```python
# upsampling: Nearest or bilinear upsampling
up_nearest = topi.nn.upsampling(data, scale_h=2, scale_w=2, method="nearest_neighbor")
up_bilinear = topi.nn.upsampling(data, scale_h=2, scale_w=2, method="bilinear")

# resize: Resize images (bilinear, nearest, bicubic)
resized = topi.image.resize(
    data,
    size=(448, 448),
    layout="NCHW",
    method="bilinear",
    coordinate_transformation_mode="align_corners",
)

# dilate: Spatial dilation (insert zeros)
dilated = topi.nn.dilate(data, strides=(1, 1, 2, 2))
```

### 21.2.3 Math Operators (topi.math)

#### Element-wise Math Functions

```python
import tvm
from tvm import te, topi

# Use TE tensor as input
A = te.placeholder((128,), name="A", dtype="float32")

# Basic arithmetic
abs_out = topi.math.abs(A)
ceil_out = topi.math.ceil(A)
floor_out = topi.math.floor(A)
round_out = topi.math.round(A)
sign_out = topi.math.sign(A)

# Exponential and logarithmic
exp_out = topi.math.exp(A)
log_out = topi.math.log(A)
log2_out = topi.math.log2(A)
log10_out = topi.math.log10(A)

# Power and root
sqrt_out = topi.math.sqrt(A)
rsqrt_out = topi.math.rsqrt(A)   # 1 / sqrt(x)
pow_out = topi.math.power(A, B)  # A^B element-wise

# Trigonometric
sin_out = topi.math.sin(A)
cos_out = topi.math.cos(A)
tan_out = topi.math.tan(A)
asin_out = topi.math.asin(A)
acos_out = topi.math.acos(A)
atan_out = topi.math.atan(A)

# Hyperbolic
sinh_out = topi.math.sinh(A)
cosh_out = topi.math.cosh(A)
tanh_out = topi.math.tanh(A)

# Rounding
clip_out = topi.math.clip(A, a_min=0.0, a_max=6.0)  # Clipped ReLU range
cast_out = topi.math.cast(A, "float16")               # Type casting

# Comparison / logical
isnan_out = topi.math.isnan(A)
isinf_out = topi.math.isinf(A)
isfinite_out = topi.math.isfinite(A)
```

#### Binary Math Operations

```python
A = te.placeholder((128,), name="A", dtype="float32")
B = te.placeholder((128,), name="B", dtype="float32")

# Element-wise binary operations
add_out = topi.math.add(A, B)          # A + B
sub_out = topi.math.subtract(A, B)     # A - B
mul_out = topi.math.multiply(A, B)     # A * B
div_out = topi.math.divide(A, B)       # A / B
mod_out = topi.math.fmod(A, B)         # A % B

# Element-wise comparisons (result is boolean)
max_out = topi.math.maximum(A, B)      # Element-wise max
min_out = topi.math.minimum(A, B)      # Element-wise min

# Shift operations (integer types)
lshift_out = topi.math.left_shift(A_int, B_int)
rshift_out = topi.math.right_shift(A_int, B_int)
```

#### Tensor Creation Operations

```python
# Create tensors filled with specific values
full_out = topi.math.full((128, 128), fill_value=0.0, dtype="float32")
zeros_out = topi.math.zeros((128, 128), dtype="float32")
ones_out = topi.math.ones((128, 128), dtype="float32")

# Create tensors with the same shape as another tensor
full_like_out = topi.math.full_like(A, fill_value=1.0)
zeros_like_out = topi.math.zeros_like(A)
ones_like_out = topi.math.ones_like(A)

# Element-wise sum of multiple tensors
elemwise_sum_out = topi.math.elemwise_sum([A, B, C])
```

### 21.2.4 Reduction Operators (topi)

```python
# sum: Sum over specified axis
sum_all = topi.sum(A)                # Sum all elements
sum_axis0 = topi.sum(A, axis=0)      # Sum over axis 0
sum_axes = topi.sum(A, axis=(0, 2))  # Sum over axes 0 and 2
sum_keepdims = topi.sum(A, axis=1, keepdims=True)  # Keep reduced dimension

# max: Maximum over specified axis
max_all = topi.max(A)
max_axis0 = topi.max(A, axis=0)

# min: Minimum over specified axis
min_all = topi.min(A)
min_axis0 = topi.min(A, axis=0)

# argmax, argmin: Index of max/min element
argmax_out = topi.argmax(A, axis=1)
argmin_out = topi.argmin(A, axis=1)

# prod: Product over specified axis
prod_out = topi.prod(A, axis=0)

# any: Logical OR over specified axis (boolean)
any_out = topi.any(A_bool, axis=0)

# all: Logical AND over specified axis (boolean)
all_out = topi.all(A_bool, axis=0)
```

### 21.2.5 Transform Operators (topi)

```python
# reshape: Change tensor shape
reshaped = topi.reshape(A, (32, 4))

# expand_dims: Add a dimension of size 1
expanded = topi.expand_dims(A, axis=0)  # [128] -> [1, 128]

# squeeze: Remove dimensions of size 1
squeezed = topi.squeeze(A, axis=0)  # [1, 128] -> [128]

# flatten: Flatten to 2D (batch, features)
flattened = topi.nn.flatten(A)

# concatenate: Join tensors along an axis
concat = topi.concatenate([A, B, C], axis=0)

# split: Split tensor along an axis
split_parts = topi.split(A, 4, axis=0)  # Split into 4 equal parts
split_indices = topi.split(A, [32, 64], axis=0)  # Split at indices

# take: Gather elements by index along an axis
taken = topi.take(A, indices, axis=0)

# gather: Gather elements by index (advanced indexing)
gathered = topi.gather(A, axis=0, indices=indices)

# scatter: Scatter updates into tensor
scattered = topi.scatter(data, indices, updates, axis=0)

# scatter_nd: Scatter updates into tensor (N-dimensional indexing)
scattered_nd = topi.scatter_nd(data, indices, updates)

# gather_nd: Gather elements by N-dimensional indices
gathered_nd = topi.gather_nd(A, indices)

# transpose: Permute dimensions
transposed = topi.transpose(A, axes=(1, 0))  # 2D transpose

# flip: Reverse elements along an axis
flipped = topi.flip(A, axis=0)

# reverse: Reverse sequence along an axis
reversed_seq = topi.reverse_sequence(A, seq_lengths, seq_axis=0, batch_axis=1)

# strided_slice: Slice with strides
sliced = topi.strided_slice(A, begin=[0, 0], end=[64, 64], strides=[1, 2])

# broadcast_to: Broadcast to a target shape
broadcasted = topi.broadcast_to(A, (4, 128))

# tile: Repeat tensor along each dimension
tiled = topi.tile(A, reps=(4, 1))

# repeat: Repeat elements along an axis
repeated = topi.repeat(A, repeats=3, axis=0)

# stack: Stack tensors along a new axis
stacked = topi.stack([A, B, C], axis=0)

# meshgrid: Generate coordinate grids
grid_x, grid_y = topi.meshgrid([x_range], [y_range], indexing="ij")

# where: Conditional element selection
result = topi.where(condition, A, B)  # Choose A where condition is True, else B
```

### 21.2.6 Image Operators (topi.image)

```python
import tvm
from tvm import te, topi

# resize: Resize images with various interpolation methods
data = te.placeholder((1, 3, 224, 224), name="data", dtype="float32")

# Bilinear resize
resized_bilinear = topi.image.resize(
    data,
    size=(448, 448),
    layout="NCHW",
    method="bilinear",
    coordinate_transformation_mode="half_pixel",
)

# Nearest neighbor resize
resized_nearest = topi.image.resize(
    data,
    size=(448, 448),
    layout="NCHW",
    method="nearest_neighbor",
    coordinate_transformation_mode="asymmetric",
)

# Bicubic resize
resized_bicubic = topi.image.resize(
    data,
    size=(448, 448),
    layout="NCHW",
    method="bicubic",
    coordinate_transformation_mode="half_pixel",
)

# affine_grid: Generate affine transformation grid
grid = topi.image.affine_grid(theta, out_size)

# grid_sample: Sample from input using a grid
sampled = topi.image.grid_sample(
    data, grid,
    method="bilinear",
    padding_mode="zeros",
)
```

### 21.2.7 Vision Operators (topi.vision)

```python
# nms: Non-Maximum Suppression
kept_indices = topi.vision.nms(
    boxes,          # [N, 4] bounding boxes
    scores,         # [N] confidence scores
    iou_threshold=0.5,
    force_suppress=False,
    top_k=100,
)

# non_max_suppression: Full NMS (returns boxes and scores)
valid_boxes, valid_scores, valid_ids = topi.vision.non_max_suppression(
    boxes, scores, max_output_size=100,
    iou_threshold=0.5, score_threshold=0.05,
)

# roi_align: Region of interest alignment
roi_features = topi.vision.roi_align(
    data,           # [batch, channels, height, width]
    rois,           # [num_rois, 5] (batch_idx, x1, y1, x2, y2)
    pooled_size=(7, 7),
    spatial_scale=1.0 / 16.0,
    sample_ratio=-1,  # -1 = adaptive
)

# roi_pool: Region of interest pooling
roi_pooled = topi.vision.roi_pool(
    data, rois,
    pooled_size=(7, 7),
    spatial_scale=1.0 / 16.0,
)

# topk: Top-K elements
values, indices = topi.vision.topk(
    data,
    k=10,
    axis=-1,
    ret_type="both",  # "values", "indices", or "both"
    largest=True,      # True for top-K, False for bottom-K
)
```

### 21.2.8 Sorting and Searching Operators

```python
# topk: Get top-K elements along an axis
values, indices = topi.topk(data, k=5, axis=-1, largest=True)

# argsort: Sort indices along an axis
sorted_indices = topi.argsort(data, axis=-1, ascending=True)

# searchsorted: Find indices where elements should be inserted
indices = topi.searchsorted(sorted_sequence, values, right=False)
```

### 21.2.9 Backend-Specific Implementations

TOPI provides backend-specific implementations that are automatically selected based on the target. Each backend module contains optimized schedules for that hardware.

#### topi.cuda -- NVIDIA GPU

```python
import tvm
from tvm import te, topi

# CUDA-optimized conv2d
data = te.placeholder((1, 3, 224, 224), name="data", dtype="float32")
kernel = te.placeholder((32, 3, 3, 3), name="kernel", dtype="float32")

# The generic conv2d operator
conv = topi.nn.conv2d(data, kernel, strides=1, padding=1)

# Create CUDA-optimized schedule
with tvm.target.Target("cuda"):
    s = topi.cuda.schedule_conv2d_nchw([conv])
    # Build
    func = tvm.build(s, [data, kernel, conv], target="cuda")
```

**CUDA-specific schedules available:**

| Schedule function | Operator |
|-------------------|----------|
| `schedule_conv2d_nchw` | Conv2D in NCHW layout |
| `schedule_conv2d_nhwc` | Conv2D in NHWC layout |
| `schedule_conv2d_winograd` | Winograd convolution |
| `schedule_depthwise_conv2d_nchw` | Depthwise conv2D |
| `schedule_dense` | Dense/FC layer |
| `schedule_softmax` | Softmax |
| `schedule_reduce` | Reduction operations |
| `schedule_pool` | Pooling |
| `schedule_batch_norm` | Batch normalization |
| `schedule_injective` | Element-wise/injective operations |

#### topi.rocm -- AMD GPU

```python
# ROCm-optimized schedules for AMD GPUs
with tvm.target.Target("rocm"):
    s = topi.rocm.schedule_conv2d_nchw([conv])
```

#### topi.metal -- Apple Metal

```python
# Metal-optimized schedules for Apple GPUs
with tvm.target.Target("metal"):
    s = topi.metal.schedule_conv2d_nchw([conv])
```

#### topi.opencl -- OpenCL

```python
# OpenCL-optimized schedules
with tvm.target.Target("opencl"):
    s = topi.opencl.schedule_conv2d_nchw([conv])
```

#### topi.vulkan -- Vulkan

```python
# Vulkan-optimized schedules
with tvm.target.Target("vulkan"):
    s = topi.vulkan.schedule_conv2d_nchw([conv])
```

#### topi.hexagon -- Qualcomm DSP

```python
# Hexagon-optimized schedules for Qualcomm DSPs
with tvm.target.Target("hexagon"):
    s = topi.hexagon.schedule_conv2d_nchw([conv])
```

### 21.2.10 Contrib Operators

TOPI includes additional operators in the `contrib` subpackage:

```python
# Winograd convolution (faster for 3x3 filters)
from tvm.topi.nn import conv2d_winograd_weight_transform
winograd_weights = conv2d_winograd_weight_transform(kernel, tile_size=6)

# NCHW[x]c layout convolution (blocked layout for vectorization)
conv_blocked = topi.nn.conv2d_nchwc(data_nchwc, kernel_nchwc)

# GEMM-based convolution (im2col + matmul)
conv_gemm = topi.nn.conv2d_gemm_weight_preprocess(kernel)
```

**Available contrib modules:**

| Module | Description |
|--------|-------------|
| `topi.contrib.cublas` | cuBLAS integration for dense operations |
| `topi.contrib.cudnn` | cuDNN integration for conv/pool |
| `topi.contrib.miopen` | MIOpen integration for AMD GPUs |
| `topi.contrib.clml` | CLML integration for OpenCL ML |
| `topi.contrib.nnpack` | NNPACK integration for CPU |

### 21.2.11 Using TOPI with create_prim_func

The most common pattern is to use TOPI operators and then convert to PrimFunc for scheduling:

```python
import tvm
from tvm import te, topi
from tvm import dlight as dl

# Define inputs
data = te.placeholder((1, 3, 224, 224), name="data", dtype="float32")
kernel = te.placeholder((32, 3, 3, 3), name="kernel", dtype="float32")
bias = te.placeholder((32,), name="bias", dtype="float32")

# Use TOPI for conv2d + bias + relu
conv = topi.nn.conv2d(data, kernel, strides=1, padding=1)
biased = topi.nn.bias_add(conv, bias)
relu_out = topi.nn.relu(biased)

# Convert to PrimFunc
prim_func = te.create_prim_func([data, kernel, bias, relu_out])

# Wrap in IRModule
mod = tvm.IRModule.from_expr(prim_func)

# Schedule with DLight
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

# Build
exec_mod = tvm.build(mod, target="nvidia/nvidia-a100")
```

### 21.2.12 Building a Complete Model with TOPI

```python
import tvm
from tvm import te, topi
import numpy as np

def build_simple_cnn():
    """Build a simple CNN: Conv2D -> BN -> ReLU -> Pool -> Dense."""

    # Input
    data = te.placeholder((1, 3, 32, 32), name="data", dtype="float32")

    # Conv2D layer
    conv_weight = te.placeholder((16, 3, 3, 3), name="conv_w", dtype="float32")
    conv_bias = te.placeholder((16,), name="conv_b", dtype="float32")
    conv = topi.nn.conv2d(data, conv_weight, strides=1, padding=1)
    conv_biased = topi.nn.bias_add(conv, conv_bias)

    # Batch normalization (simplified: assume pre-computed mean/var/gamma/beta)
    bn_gamma = te.placeholder((16,), name="gamma", dtype="float32")
    bn_beta = te.placeholder((16,), name="beta", dtype="float32")
    bn_mean = te.placeholder((16,), name="mean", dtype="float32")
    bn_var = te.placeholder((16,), name="var", dtype="float32")
    bn_out = topi.nn.batch_norm(
        conv_biased, bn_gamma, bn_beta, bn_mean, bn_var, axis=1, epsilon=1e-5
    )

    # ReLU
    relu_out = topi.nn.relu(bn_out)

    # Max Pool 2x2
    pool_out = topi.nn.pool2d(
        relu_out, kernel_size=(2, 2), stride=(2, 2),
        padding=(0, 0), pool_type="max"
    )

    # Flatten
    flat_out = topi.nn.flatten(pool_out)

    # Dense layer
    dense_weight = te.placeholder((10, 16 * 16 * 16), name="dense_w", dtype="float32")
    dense_bias = te.placeholder((10,), name="dense_b", dtype="float32")
    dense_out = topi.nn.dense(flat_out, dense_weight)
    output = topi.nn.bias_add(dense_out, dense_bias)

    # Convert to PrimFunc
    prim_func = te.create_prim_func([
        data, conv_weight, conv_bias,
        bn_gamma, bn_beta, bn_mean, bn_var,
        dense_weight, dense_bias,
        output
    ])

    return tvm.IRModule.from_expr(prim_func)

# Build and run
mod = build_simple_cnn()

# Schedule with DLight
from tvm import dlight as dl
with tvm.target.Target("nvidia/nvidia-a100"):
    mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

# Compile
target = tvm.target.Target("nvidia/nvidia-a100")
exec_mod = tvm.build(mod, target=target)
```

---

## 21.3 TOPI Schedule Catalog

### 21.3.1 Generic Schedules

When no backend-specific schedule is needed, TOPI provides generic schedules:

```python
from tvm import topi

# Generic schedule for injective (element-wise) operations
s = topi.generic.schedule_injective([output_tensor])

# Generic schedule for reductions
s = topi.generic.schedule_reduce([output_tensor])

# Generic schedule for convolutions
s = topi.generic.schedule_conv2d_nchw([conv_out])

# Generic schedule for dense
s = topi.generic.schedule_dense([dense_out])
```

### 21.3.2 Schedule Selection

TOPI uses a fallback mechanism for schedule selection:

```
1. Check for target-specific schedule (cuda, rocm, metal, etc.)
2. If not found, use generic schedule
3. Generic schedules provide correct but not necessarily optimal code
```

For production use, always use the target-specific schedule or convert to PrimFunc and use DLight/MetaSchedule.

---

## 21.4 Advanced TE Patterns

### 21.4.1 Fused Operations

```python
# Fused bias-add + ReLU + quantize
A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.placeholder((128,), name="B", dtype="float32")
scale = te.placeholder((), name="scale", dtype="float32")

# Single fused kernel
fused = te.compute(
    (128, 128),
    lambda i, j: tvm.tir.min(
        tvm.tir.max(A[i, j] + B[j], tvm.tir.const(0, "float32")),
        tvm.tir.const(255, "float32")
    ) * scale,
    name="fused_bias_relu_quant",
)

prim_func = te.create_prim_func([A, B, scale, fused])
```

### 21.4.2 Tiled Computation

```python
# Tiled matrix multiply with intermediate buffers
M, N, K = 1024, 1024, 1024
TILE = 32

A = te.placeholder((M, K), name="A", dtype="float16")
B = te.placeholder((K, N), name="B", dtype="float16")

# Stage 1: Load tile of A
A_tile = te.compute(
    (TILE, TILE),
    lambda i, j: A[i, j],
    name="A_tile"
)

# Stage 2: Load tile of B
B_tile = te.compute(
    (TILE, TILE),
    lambda i, j: B[i, j],
    name="B_tile"
)

# Stage 3: Compute partial product
rk = te.reduce_axis((0, TILE), name="rk")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, rk] * B[rk, j], axis=rk),
    name="C"
)
```

### 21.4.3 Padding and Unpadding

```python
# Pad a 2D tensor
A = te.placeholder((224, 224), name="A", dtype="float32")
padded = topi.nn.pad(A, pad_before=((1, 1)), pad_after=((1, 1)))

# Unpad (slice)
unpadded = topi.strided_slice(padded, begin=[1, 1], end=[225, 225])
```

### 21.4.4 Winograd Convolution

```python
# Winograd-based convolution for 3x3 filters (F(2,3) or F(4,3))
data = te.placeholder((1, 32, 56, 56), name="data", dtype="float32")
kernel = te.placeholder((32, 32, 3, 3), name="kernel", dtype="float32")

# Use TOPI's Winograd implementation
winograd_conv = topi.nn.conv2d_winograd(
    data, kernel,
    strides=1,
    padding=1,
    tile_size=4,  # F(4,3) Winograd
)
```

---

## 21.5 Common Patterns and Recipes

### 21.5.1 Defining a Custom Operator with TE

```python
import tvm
from tvm import te

def my_gelu(x):
    """Approximate GELU activation: x * sigmoid(1.702 * x)."""
    return x * tvm.tir.sigmoid(tvm.tir.const(1.702, "float32") * x)

A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.compute((128, 128), lambda i, j: my_gelu(A[i, j]), name="B")
prim_func = te.create_prim_func([A, B])
```

### 21.5.2 Multi-Output Operator

```python
# Compute both max and argmax in one pass
A = te.placeholder((1024,), name="A", dtype="float32")

max_val = te.compute((), lambda: topi.max(A), name="max_val")
argmax_val = te.compute((), lambda: topi.argmax(A), name="argmax_val")

# Create PrimFunc with both outputs
prim_func = te.create_prim_func([A, max_val, argmax_val])
```

### 21.5.3 Conditional Computation

```python
# Swish activation: x * sigmoid(beta * x)
A = te.placeholder((128,), name="A", dtype="float32")
beta = te.placeholder((), name="beta", dtype="float32")

swish = te.compute(
    (128,),
    lambda i: A[i] * tvm.tir.sigmoid(beta * A[i]),
    name="swish"
)
```

### 21.5.4 Quantized Operations

```python
# Quantized matrix multiply: int8 inputs, int32 accumulation
A = te.placeholder((128, 128), name="A", dtype="int8")
B = te.placeholder((128, 128), name="B", dtype="int8")

k = te.reduce_axis((0, 128), name="k")
C = te.compute(
    (128, 128),
    lambda i, j: te.sum(
        A[i, k].astype("int32") * B[k, j].astype("int32"),
        axis=k
    ),
    name="C"
)

# Dequantize result
scale = te.placeholder((), name="scale", dtype="float32")
zero_point = te.placeholder((), name="zero_point", dtype="float32")
D = te.compute(
    (128, 128),
    lambda i, j: (C[i, j].astype("float32") - zero_point) * scale,
    name="D"
)

prim_func = te.create_prim_func([A, B, scale, zero_point, D])
```

### 21.5.5 Softmax with Numerical Stability

```python
import tvm
from tvm import te, topi

# TOPI already provides numerically stable softmax
A = te.placeholder((128, 1024), name="A", dtype="float32")
softmax_out = topi.nn.softmax(A, axis=-1)

# But here is how to implement it manually for learning purposes:
# Step 1: Max along axis for numerical stability
max_val = topi.max(A, axis=-1, keepdims=True)

# Step 2: Subtract max and exp
exp_out = te.compute(
    (128, 1024),
    lambda i, j: topi.math.exp(A[i, j] - max_val[i, 0]),
    name="exp_out"
)

# Step 3: Sum of exp
sum_exp = topi.sum(exp_out, axis=-1, keepdims=True)

# Step 4: Normalize
manual_softmax = te.compute(
    (128, 1024),
    lambda i, j: exp_out[i, j] / sum_exp[i, 0],
    name="manual_softmax"
)
```

---

## 21.6 Migration from TE to TVMScript/TensorIR

### 21.6.1 When to Use TE vs TVMScript

| Scenario | Recommended Approach |
|----------|---------------------|
| Standard NN operators | Use TOPI directly |
| Custom simple operator | TE + create_prim_func |
| Complex custom operator | Write TVMScript directly |
| Need fine-grained control | TVMScript |
| Quick prototyping | TE |
| Production code | TVMScript or TOPI |

### 21.6.2 TE to TVMScript Translation Examples

**TE version:**

```python
from tvm import te

A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.compute((128, 128), lambda i, j: A[i, j] * 2.0 + 1.0, name="B")
prim_func = te.create_prim_func([A, B])
```

**Equivalent TVMScript:**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
    ):
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * T.float32(2.0) + T.float32(1.0)
```

**TE matmul:**

```python
from tvm import te

M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
prim_func = te.create_prim_func([A, B, C])
```

**Equivalent TVMScript matmul:**

```python
from tvm.script import ir as I, tir as T

@I.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ):
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

---

## 21.7 Reference Tables

### 21.7.1 TE API Quick Reference

| API | Description | Example |
|-----|-------------|---------|
| `te.placeholder(shape, name, dtype)` | Declare input tensor | `A = te.placeholder((128, 128), name="A")` |
| `te.compute(shape, fcompute, name)` | Declare output tensor | `B = te.compute((128,), lambda i: A[i] * 2)` |
| `te.reduce_axis(range, name)` | Declare reduction axis | `k = te.reduce_axis((0, K), name="k")` |
| `te.sum(expr, axis)` | Summation reduction | `te.sum(A[i, k] * B[k, j], axis=k)` |
| `te.max(expr, axis)` | Maximum reduction | `te.max(A[i, k], axis=k)` |
| `te.min(expr, axis)` | Minimum reduction | `te.min(A[i, k], axis=k)` |
| `te.scan(init, update, state)` | Recurrent operation | `te.scan(init, update, state)` |
| `te.extern(shape, inputs, fcompute)` | External function | `te.extern((M,N), [A,B], f)` |
| `te.create_prim_func(tensors)` | Convert to PrimFunc | `te.create_prim_func([A, B, C])` |
| `te.create_schedule(op)` | Legacy schedule creation | `s = te.create_schedule(C.op)` |
| `te.size_var(name)` | Symbolic dimension | `M = te.size_var("M")` |

### 21.7.2 TOPI Operator Quick Reference

| Category | Operators |
|----------|-----------|
| **Convolution** | `conv1d`, `conv2d`, `conv3d`, `conv1d_transpose`, `conv2d_transpose`, `conv3d_transpose`, `depthwise_conv2d`, `group_conv2d`, `deformable_conv2d` |
| **Dense** | `dense`, `sparse_dense`, `bias_add` |
| **Activation** | `relu`, `leaky_relu`, `prelu`, `sigmoid`, `tanh`, `softmax`, `log_softmax` |
| **Normalization** | `batch_norm`, `layer_norm`, `instance_norm`, `group_norm`, `lrn`, `l2_normalize` |
| **Pooling** | `pool1d`, `pool2d`, `pool3d`, `adaptive_pool`, `global_pool` |
| **Spatial** | `upsampling`, `dilate`, `pad` |
| **Reduction** | `sum`, `max`, `min`, `argmax`, `argmin`, `prod`, `any`, `all` |
| **Transform** | `reshape`, `expand_dims`, `squeeze`, `flatten`, `concatenate`, `split`, `take`, `gather`, `scatter`, `transpose`, `flip`, `strided_slice`, `broadcast_to`, `tile`, `repeat`, `stack`, `where` |
| **Math** | `abs`, `ceil`, `floor`, `round`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `power`, `clip`, `cast`, `sign` |
| **Binary** | `add`, `subtract`, `multiply`, `divide`, `fmod`, `maximum`, `minimum`, `left_shift`, `right_shift` |
| **Image** | `resize` (bilinear, nearest, bicubic), `affine_grid`, `grid_sample` |
| **Vision** | `nms`, `non_max_suppression`, `roi_align`, `roi_pool`, `topk` |
| **Creation** | `full`, `full_like`, `zeros`, `zeros_like`, `ones`, `ones_like`, `elemwise_sum` |

### 21.7.3 Data Type Reference

| TE dtype string | C type | Size (bytes) |
|-----------------|--------|-------------|
| `"float32"` | `float` | 4 |
| `"float64"` | `double` | 8 |
| `"float16"` | `__half` | 2 |
| `"bfloat16"` | `__bfloat16` | 2 |
| `"int8"` | `int8_t` | 1 |
| `"int16"` | `int16_t` | 2 |
| `"int32"` | `int32_t` | 4 |
| `"int64"` | `int64_t` | 8 |
| `"uint8"` | `uint8_t` | 1 |
| `"uint16"` | `uint16_t` | 2 |
| `"uint32"` | `uint32_t` | 4 |
| `"uint64"` | `uint64_t` | 8 |
| `"bool"` | `bool` | 1 |

---

## 21.8 Summary

TE and TOPI form TVM's operator definition layer:

- **TE (Tensor Expression)** provides the DSL for defining custom tensor computations declaratively. It uses `placeholder`, `compute`, `reduce_axis`, `scan`, and `extern` to build computation graphs that are converted to `tir.PrimFunc` via `te.create_prim_func`.

- **TOPI (Tensor Operator Inventory)** provides a comprehensive library of pre-defined operators spanning convolutions, dense layers, activations, normalization, pooling, math functions, transforms, and vision-specific operations. TOPI operators are implemented using TE internally and provide both generic and backend-specific schedules.

- **The modern workflow** is to define operators using TE/TOPI, convert to PrimFunc, and then apply TensorIR scheduling (DLight or MetaSchedule) for optimization. This combines the ease of TE's declarative API with the power of TensorIR's scheduling primitives.

- **For new code**, consider writing operators directly in TVMScript/TensorIR when fine-grained control is needed, or use TOPI for standard operations. The TE DSL remains fully supported and is the primary way to define custom operators that will be converted to PrimFunc for scheduling.
