# Apache TVM — Chapter 24: BYOC (Bring Your Own Codegen) & External Library Dispatch

This reference covers TVM's BYOC framework, which enables integration of custom hardware backends and vendor libraries into the compilation pipeline. External library dispatch allows routing specific operator patterns to optimized library implementations, such as cuBLAS, cuDNN, CUTLASS, and TensorRT.

---

## 24.1 BYOC Overview

### Why BYOC Matters

Hardware vendors frequently provide highly optimized libraries for specific operations (e.g., NVIDIA's cuBLAS for matrix multiplication, Intel's oneDNN for convolution). BYOC allows TVM to leverage these libraries while still using TVM's own compilation for the remaining operations. This hybrid approach combines the best of both worlds:

- **Vendor-optimized performance**: Library implementations are hand-tuned for specific hardware and often outperform auto-generated code.
- **TVM's optimization pipeline**: The remaining operations still benefit from TVM's fusion, scheduling, and code generation.
- **Rapid hardware support**: New hardware backends can be integrated without modifying TVM's core compiler.
- **Extensibility**: Third-party developers can add support for custom accelerators and NPUs.

### BYOC in the Compilation Pipeline

```
Input Model (PyTorch/ONNX/etc.)
        |
        v
   Relax Frontend Import
        |
        v
   IRModule (relax::Function)
        |
        v
   [FuseOpsByPattern]  -- Pattern-based partitioning for external backends
        |
        v
   [Annotate Codegen]  -- Mark subgraphs with target codegen backend
        |
        v
   [RunCodegen]        -- Invoke external codegen for annotated subgraphs
        |
        v
   [LegalizeOps]       -- Lower remaining Relax ops to TIR
        |
        v
   [MetaSchedule / DLight]  -- Schedule TIR PrimFunc
        |
        v
   Compiled Executable (mix of external + TVM-generated kernels)
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| FusionPattern | A declarative pattern describing a subgraph to be offloaded |
| Codegen Annotation | Metadata marking a subgraph for a specific external backend |
| External Codegen | A registered function that compiles annotated subgraphs to runtime.Module |
| Runtime Module | The compiled artifact that wraps external library calls in PackedFunc |
| Pattern Table | A collection of patterns registered by a backend, ordered by specificity |

---

## 24.2 External Library Dispatch Pipeline

The external library dispatch pipeline consists of four stages: pattern registration, graph partitioning, external codegen, and runtime module generation.

### 24.2.1 Step 1: Register Patterns

Patterns are defined using the Dataflow Pattern Language (DPL), which provides a declarative way to describe computation subgraphs:

```python
from tvm.relax.dpl import wildcard, is_op, is_var, is_tuple_get_item
from tvm.relax.transform import FusionPattern

# Define individual pattern nodes
x = wildcard()
w = wildcard()
matmul = is_op("relax.matmul")(x, w)
bias = wildcard()
add = is_op("relax.add")(matmul, bias)
relu = is_op("relax.nn.relu")(add)

# Create a named fusion pattern
pattern = FusionPattern(
    name="cutlass.matmul_bias_relu",
    pattern=relu,  # The root of the pattern (final operation)
    annotation_patterns={
        "matmul": matmul,
        "bias": bias,
        "activation": relu,
    },
    check=lambda ctx: True,  # Optional validation function
)
```

### 24.2.2 Pattern Composition Rules

Patterns can be composed to match complex subgraphs:

```python
# Wildcard: matches any single node
x = wildcard()

# is_var: matches a specific variable
x = is_var("input")

# is_op: matches a specific operator
conv = is_op("relax.nn.conv2d")(x, w)

# is_tuple_get_item: matches tuple extraction
bn = is_op("relax.nn.batch_norm")(conv, gamma, beta, mean, var)
bn_out = is_tuple_get_item(bn, 0)

# Chaining: match multi-operator sequences
add = is_op("relax.add")(bn_out, residual)  # skip connection
relu = is_op("relax.nn.relu")(add)
```

### 24.2.3 Step 2: Pattern-based Graph Partitioning

`FuseOpsByPattern` partitions the computation graph based on registered patterns:

```python
from tvm.relax.transform import FuseOpsByPattern

# Define multiple patterns with priority ordering
patterns = [
    matmul_bias_relu_pattern,   # Most specific first
    matmul_bias_pattern,
    matmul_pattern,
    conv_bias_relu_pattern,
    conv_bias_pattern,
]

mod = FuseOpsByPattern(
    patterns=patterns,
    bind_constants=False,
    annotate_codegen=True,  # Mark subgraphs for external codegen
)(mod)
```

When `annotate_codegen=True`, each matched subgraph is wrapped with:
- A `Codegen` attribute specifying the target backend name
- A `global_symbol` attribute for runtime function lookup

### 24.2.4 Step 3: Run External Codegen

The `RunCodegen` pass invokes the registered external codegen functions for each annotated subgraph:

```python
from tvm.relax.transform import RunCodegen

# Run codegen for all annotated backends
mod = RunCodegen()(mod)

# Or configure specific backends
mod = RunCodegen(
    config={
        "cutlass": {"use_dp4a": False, "smemo": False},
        "cublas": {"use_cublaslt": True},
    }
)(mod)
```

### 24.2.5 Step 4: Generate Runtime Modules

Each partitioned subgraph is compiled to a `runtime.Module` containing:
- The external library function call wrapped in a PackedFunc
- Any necessary constant data (weights, bias values)
- Device-specific metadata

```python
# The generated module structure:
# runtime.Module
#   ├── host_module (LLVM)      -- orchestrates execution
#   ├── cutlass_module           -- CUTLASS kernel calls
#   └── cublas_module            -- cuBLAS library calls
```

---

## 24.3 FusionPattern API Reference

### 24.3.1 FusionPattern Constructor

```python
class FusionPattern:
    def __init__(
        self,
        name: str,
        pattern: DFPattern,
        annotation_patterns: Optional[Dict[str, DFPattern]] = None,
        check: Optional[Callable[[PatternContext], bool]] = None,
        attrs: Optional[Dict[str, Object]] = None,
    ):
        """
        Args:
            name: Unique identifier for the pattern (also used as codegen key)
            pattern: The root DFPattern to match
            annotation_patterns: Named sub-patterns for codegen extraction
            check: Optional validation function called after matching
            attrs: Additional attributes to attach to matched subgraphs
        """
```

### 24.3.2 Pattern Check Function

The `check` function provides additional validation beyond structural matching:

```python
def check_matmul_pattern(ctx: PatternContext) -> bool:
    """Validate that the matched matmul has supported dimensions."""
    # Access matched node attributes
    matmul_call = ctx.matched_nodes["matmul"]
    x_shape = matmul_call.args[0].struct_info.shape

    # Only offload if batch dimension is 1 or small
    if x_shape[0].value > 16:
        return False

    # Only offload FP16 or FP32
    dtype = matmul_call.args[0].struct_info.dtype
    return dtype in ("float16", "float32")

pattern = FusionPattern(
    name="custom_backend.matmul",
    pattern=matmul_root,
    annotation_patterns={"matmul": matmul_node},
    check=check_matmul_pattern,
)
```

### 24.3.3 Pattern Naming Convention

Pattern names follow a convention: `<backend>.<operation>[_<fused_ops>]`

```python
# Examples:
"cutlass.matmul_bias_relu"        # CUTLASS backend, matmul + bias + relu
"cutlass.matmul_bias_gelu"        # CUTLASS backend, matmul + bias + gelu
"cublas.matmul"                   # cuBLAS backend, plain matmul
"cudnn.conv2d_bias_relu"          # cuDNN backend, conv2d + bias + relu
"tensorrt.subgraph"               # TensorRT backend, generic subgraph
"my_npu.linear_relu"              # Custom NPU backend, linear + relu
```

---

## 24.4 CUTLASS Integration

### 24.4.1 Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines) provides high-performance GEMM and convolution kernels for NVIDIA GPUs. TVM's CUTLASS integration supports epilogue fusion, where activation functions are computed within the GEMM kernel.

### 24.4.2 CUTLASS Pattern Table

```python
from tvm.contrib.cutlass import get_patterns

# Get all registered CUTLASS patterns
cutlass_patterns = get_patterns()
# Returns patterns for:
#   - matmul
#   - matmul_bias
#   - matmul_bias_relu
#   - matmul_bias_gelu
#   - matmul_bias_sigmoid
#   - batch_matmul
#   - batch_matmul_bias
#   - conv2d_bias_relu
#   - conv2d_bias
#   - attention (fused Q*K*V)
```

### 24.4.3 Using CUTLASS in the Compilation Pipeline

```python
import tvm
from tvm import relax
from tvm.contrib.cutlass import get_patterns as get_cutlass_patterns
from tvm.relax.transform import FuseOpsByPattern, RunCodegen

# Step 1: Get CUTLASS patterns
patterns = get_cutlass_patterns()

# Step 2: Partition the graph
mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)

# Step 3: Run CUTLASS codegen
mod = RunCodegen(config={"cutlass": {}})(mod)
```

### 24.4.4 CUTLASS Codegen Configuration

```python
cutlass_config = {
    # Use DP4A instructions for INT8 operations
    "use_dp4a": True,

    # Use shared memory for epilogue
    "smemo": False,

    # Maximum number of CUTLASS instances to generate
    "max_smem": None,

    # Target SM architecture
    "sm": 80,  # SM80 for A100

    # Whether to use split-K for reduction
    "split_k": [],
}

mod = RunCodegen(config={"cutlass": cutlass_config})(mod)
```

### 24.4.5 CUTLASS Epilogue Fusion

CUTLASS supports epilogue fusion where activation functions are computed directly in the GEMM kernel's output stage, avoiding a separate kernel launch and global memory round-trip:

```python
# Without epilogue fusion (3 kernels):
#   Kernel 1: C = A @ B
#   Kernel 2: D = C + bias
#   Kernel 3: E = relu(D)

# With CUTLASS epilogue fusion (1 kernel):
#   Kernel 1: E = relu(A @ B + bias)
#   The activation is computed in registers as part of the GEMM output stage
```

### 24.4.6 CUTLASS Attention Pattern

```python
# Fused attention pattern (Flash Attention style)
from tvm.relax.dpl import wildcard, is_op

Q = wildcard()
K = wildcard()
V = wildcard()

# Q * K^T
matmul_qk = is_op("relax.matmul")(Q, K)
# Scale
scale = wildcard()
scaled = is_op("relax.multiply")(matmul_qk, scale)
# Softmax
softmax = is_op("relax.nn.softmax")(scaled)
# * V
attention = is_op("relax.matmul")(softmax, V)

# This pattern can be matched by CUTLASS to generate
# fused attention kernels
```

### 24.4.7 Build-Time Integration

```python
import tvm
from tvm import relax

# Method 1: Use CompileConfig
with tvm.transform.PassContext(config={
    "relax.backend.use_cutlass": True,
}):
    exec_mod = relax.build(mod, target="nvidia/nvidia-a100")

# Method 2: Manual pipeline with CUTLASS
from tvm.contrib.cutlass import (
    get_patterns,
    optimize_for_cutlass,
)

# Apply CUTLASS-specific optimizations
mod = optimize_for_cutlass(mod)

# Build
exec_mod = relax.build(mod, target="nvidia/nvidia-a100")
```

---

## 24.5 cuBLAS Integration

### 24.5.1 Overview

cuBLAS provides optimized BLAS operations for NVIDIA GPUs. TVM can dispatch matrix multiplications and batched matrix multiplications to cuBLAS.

### 24.5.2 cuBLAS Pattern Table

```python
cublas_patterns = [
    # Basic matmul
    FusionPattern("cublas.matmul", matmul_pattern),

    # Batched matmul
    FusionPattern("cublas.batch_matmul", batch_matmul_pattern),

    # Matmul with bias
    FusionPattern("cublas.matmul_bias", matmul_bias_pattern),
]
```

### 24.5.3 cuBLAS Matmul Pattern Definition

```python
from tvm.relax.dpl import wildcard, is_op

def make_cublas_matmul_pattern():
    x = wildcard()
    w = wildcard()
    matmul = is_op("relax.matmul")(x, w)
    return matmul

def make_cublas_matmul_bias_pattern():
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    matmul = is_op("relax.matmul")(x, w)
    add = is_op("relax.add")(matmul, bias)
    return add
```

### 24.5.4 Enabling cuBLAS

```python
import tvm
from tvm import relax

# Enable cuBLAS via CompileConfig
with tvm.transform.PassContext(config={
    "relax.backend.use_cublases": True,
}):
    exec_mod = relax.build(mod, target="nvidia/nvidia-a100")
```

### 24.5.5 cuBLAS Data Type Support

| Data Type | Support |
|-----------|---------|
| FP32 | Full support |
| FP16 | Full support (with TF32 tensor cores where available) |
| BF16 | Supported on Ampere+ |
| INT8 | Supported via cuBLASLt |
| INT4 | Limited support (via cuBLASLt weight-only) |
| FP64 | Full support |

### 24.5.6 cuBLASLt Integration

cuBLASLt is the next-generation cuBLAS API that provides more flexible matrix multiplication:

```python
# cuBLASLt supports:
# - Arbitrary matrix layouts (row-major, column-major)
# - Fused bias and activation in a single call
# - INT8 quantized matmul
# - Weight-only quantization (W4A16, W8A16)
```

---

## 24.6 cuDNN Integration

### 24.6.1 Overview

cuDNN provides optimized implementations for deep learning primitives, particularly convolutions, pooling, and normalization operations.

### 24.6.2 cuDNN Pattern Table

```python
cudnn_patterns = [
    # Basic convolution
    FusionPattern("cudnn.conv2d", conv2d_pattern),

    # Convolution with bias
    FusionPattern("cudnn.conv2d_bias", conv2d_bias_pattern),

    # Convolution with bias and activation
    FusionPattern("cudnn.conv2d_bias_relu", conv2d_bias_relu_pattern),
    FusionPattern("cudnn.conv2d_bias_sigmoid", conv2d_bias_sigmoid_pattern),

    # ResNet-style skip connection
    FusionPattern("cudnn.conv2d_bias_add_relu", conv2d_bias_add_relu_pattern),

    # Batch normalization
    FusionPattern("cudnn.conv2d_bias_batch_norm", conv2d_bias_bn_pattern),
]
```

### 24.6.3 cuDNN Convolution Pattern Definition

```python
from tvm.relax.dpl import wildcard, is_op, is_tuple_get_item

def make_cudnn_conv2d_bias_relu_pattern():
    x = wildcard()
    w = wildcard()
    bias = wildcard()

    conv = is_op("relax.nn.conv2d")(x, w)
    add = is_op("relax.add")(conv, bias)
    relu = is_op("relax.nn.relu")(add)

    return relu

def make_cudnn_conv2d_bias_add_relu_pattern():
    """Matches ResNet-style: conv(x,w) + bias + residual -> relu"""
    x = wildcard()
    w = wildcard()
    bias = wildcard()
    residual = wildcard()

    conv = is_op("relax.nn.conv2d")(x, w)
    add_bias = is_op("relax.add")(conv, bias)
    add_residual = is_op("relax.add")(add_bias, residual)
    relu = is_op("relax.nn.relu")(add_residual)

    return relu
```

### 24.6.4 Enabling cuDNN

```python
import tvm
from tvm import relax

# Enable cuDNN via CompileConfig
with tvm.transform.PassContext(config={
    "relax.backend.use_cudnn": True,
}):
    exec_mod = relax.build(mod, target="nvidia/nvidia-a100")
```

### 24.6.5 cuDNN Convolution Algorithms

cuDNN supports multiple convolution algorithms, and TVM can leverage cuDNN's auto-tuning to select the best one:

```python
# cuDNN algorithm selection modes:
# - IMPLICIT_GEMM: Good for small filters
# - IMPLICIT_PRECOMP_GEMM: Precomputed index for faster execution
# - GEMM: Explicit GEMM-based convolution
# - DIRECT: Direct convolution
# - FFT: FFT-based convolution (good for large filters)
# - WINOGRAD: Winograd-based convolution (good for 3x3 filters)
# - WINOGRAD_NONFUSED: Non-fused Winograd for batched processing
```

---

## 24.7 TensorRT Integration

### 24.7.1 Overview

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. TVM can offload entire subgraphs to TensorRT, leveraging its kernel auto-tuning, quantization, and graph optimization capabilities.

### 24.7.2 TensorRT Offloading Flow

```
TVM IRModule
      |
      v
[Pattern Matching]  -- Find TRT-compatible subgraphs
      |
      v
[Graph Partitioning]  -- Split into TRT and non-TRT regions
      |
      v
[TRT Engine Building]  -- Build TRT engines for matched subgraphs
      |
      v
[Runtime Module]  -- Combined module with TRT and TVM kernels
```

### 24.7.3 TensorRT Pattern Definition

TensorRT uses a different approach than CUTLASS/cuBLAS -- instead of matching specific operator patterns, it offloads entire supported subgraphs:

```python
from tvm.relax.dpl import wildcard, is_op

def is_trt_supported(op_name: str) -> bool:
    """Check if an operation is supported by TensorRT."""
    trt_supported_ops = {
        "relax.nn.conv2d",
        "relax.nn.max_pool2d",
        "relax.nn.avg_pool2d",
        "relax.nn.relu",
        "relax.nn.softmax",
        "relax.nn.batch_norm",
        "relax.add",
        "relax.multiply",
        "relax.matmul",
        "relax.reshape",
        "relax.transpose",
        "relax.nn.sigmoid",
        "relax.nn.clip",
    }
    return op_name in trt_supported_ops
```

### 24.7.4 TensorRT ByOC Configuration

```python
import tvm
from tvm import relax
from tvm.contrib.tensorrt import partition_for_tensorrt

# Partition the graph for TensorRT
mod = partition_for_tensorrt(mod)

# Build with TensorRT
exec_mod = relax.build(mod, target="nvidia/nvidia-a100")
```

### 24.7.5 TensorRT Engine Caching

```python
# TensorRT engine building can be slow (minutes for complex models)
# TVM supports caching built engines to avoid recompilation

from tvm.contrib.tensorrt import build_engine_cache

# Save engine cache after first compilation
cache = build_engine_cache(mod, target="nvidia/nvidia-a100")
cache.save("trt_engine_cache.bin")

# Load engine cache for subsequent compilations
cache = build_engine_cache.load("trt_engine_cache.bin")
exec_mod = relax.build(mod, target="nvidia/nvidia-a100", trt_cache=cache)
```

### 24.7.6 TensorRT Supported Operations

| Operation | TensorRT Support | Notes |
|-----------|-----------------|-------|
| Conv2D | Full | All formats, dilations, groups |
| Matmul | Full | FP16, FP32, INT8 |
| BatchNorm | Full | Training and inference mode |
| ReLU/LeakyReLU | Full | Including clipped variants |
| Pooling | Full | Max, Average, Global |
| Softmax | Full | All axes |
| Element-wise | Full | Add, Mul, Sub, Div, Min, Max |
| Concatenate | Full | Along any axis |
| Resize | Full | Nearest, Bilinear, Trilinear |
| Transpose | Full | Arbitrary permutation |
| LayerNorm | FP16, FP32 | Not INT8 |

### 24.7.7 TensorRT Limitations

```python
# Operations NOT supported by TensorRT (remain in TVM):
# - Dynamic control flow (if/else based on data)
# - Custom user-defined operations
# - Operations with dynamic output shapes
# - Scatter/gather operations with dynamic indices
# - Random number generation

# When an unsupported operation is encountered,
# the graph is split at that boundary:
#   [TRT subgraph 1] -> [TVM kernel] -> [TRT subgraph 2]
```

---

## 24.8 Custom NPU Backend Integration

### 24.8.1 Complete Example: Custom NPU Backend

This example shows how to integrate a custom NPU (Neural Processing Unit) backend into TVM's compilation pipeline.

```python
import tvm
from tvm import relax
from tvm.relax.dpl import wildcard, is_op, is_var
from tvm.relax.transform import FusionPattern, FuseOpsByPattern, RunCodegen

# ============================================
# Step 1: Define patterns for the NPU
# ============================================

# Pattern: matmul + bias + relu
x = wildcard()
w = wildcard()
bias = wildcard()
matmul = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(matmul, bias)
relu = is_op("relax.nn.relu")(add)

matmul_bias_relu_pattern = FusionPattern(
    name="my_npu.matmul_bias_relu",
    pattern=relu,
    annotation_patterns={
        "matmul": matmul,
        "bias": bias,
        "activation": relu,
    },
)

# Pattern: conv2d + bias
x = wildcard()
w = wildcard()
bias = wildcard()
conv = is_op("relax.nn.conv2d")(x, w)
add = is_op("relax.add")(conv, bias)

conv_bias_pattern = FusionPattern(
    name="my_npu.conv2d_bias",
    pattern=add,
    annotation_patterns={
        "conv": conv,
        "bias": bias,
    },
)

# Pattern: standalone matmul
x = wildcard()
w = wildcard()
matmul = is_op("relax.matmul")(x, w)

matmul_pattern = FusionPattern(
    name="my_npu.matmul",
    pattern=matmul,
    annotation_patterns={"matmul": matmul},
)

# ============================================
# Step 2: Register external codegen
# ============================================

@tvm.register_func("relax.ext.my_npu")
def my_npu_codegen(mod, target):
    """
    External codegen function for the NPU backend.

    Args:
        mod: IRModule containing annotated subgraphs
        target: The compilation target

    Returns:
        runtime.Module containing NPU kernel calls
    """
    from tvm import runtime

    # Process each function in the module
    for gv, func in mod.functions.items():
        if hasattr(func, 'attrs') and func.attrs.get('Codegen') == 'my_npu':
            # Extract the subgraph
            # Generate NPU-specific binary code
            # Create PackedFunc wrapper
            pass

    # Return a runtime module wrapping the NPU calls
    return runtime.Module()

# ============================================
# Step 3: Partition and compile
# ============================================

patterns = [
    matmul_bias_relu_pattern,  # Most specific patterns first
    conv_bias_pattern,
    matmul_pattern,
]

# Partition the graph
mod = FuseOpsByPattern(
    patterns,
    bind_constants=False,
    annotate_codegen=True,
)(mod)

# Run external codegen
mod = RunCodegen(
    config={"my_npu": {"npu_id": 0, "precision": "fp16"}}
)(mod)

# Build the final executable
exec_mod = relax.build(mod, target="llvm")
```

### 24.8.2 Codegen Registration API

```python
# Register an external codegen function
@tvm.register_func("relax.ext.<backend_name>")
def codegen_func(mod, target):
    """
    External codegen function.

    Args:
        mod (IRModule): Module containing functions annotated with
                        Codegen="<backend_name>"
        target (Target): The compilation target

    Returns:
        runtime.Module: Compiled module wrapping external library calls

    The returned runtime.Module must implement:
    - GetFunction(name) -> PackedFunc
    - The PackedFunc accepts TVM NDArray arguments and returns NDArray
    """
    pass
```

### 24.8.3 NPU Backend with Shape Constraints

```python
def make_npu_matmul_pattern_with_constraints():
    x = wildcard()
    w = wildcard()
    matmul = is_op("relax.matmul")(x, w)

    def check(ctx):
        # Only offload to NPU if M dimension is small
        # (NPU has limited on-chip memory)
        x_info = ctx.get_shape(x)
        w_info = ctx.get_shape(w)

        M = x_info[0]
        N = w_info[1]
        K = w_info[0]

        # NPU constraint: M * K <= 4096
        if M is not None and K is not None:
            return M.value * K.value <= 4096
        return True

    return FusionPattern(
        name="my_npu.matmul",
        pattern=matmul,
        annotation_patterns={"matmul": matmul},
        check=check,
    )
```

### 24.8.4 Custom Backend with Multiple Operations

```python
# Define a complete attention pattern for NPU offloading
def make_npu_attention_pattern():
    Q = wildcard()
    K = wildcard()
    V = wildcard()

    # Q * K^T
    matmul_qk = is_op("relax.matmul")(Q, K)

    # Scale by 1/sqrt(d)
    scale = wildcard()
    scaled = is_op("relax.multiply")(matmul_qk, scale)

    # Softmax
    softmax = is_op("relax.nn.softmax")(scaled)

    # Attention * V
    attention = is_op("relax.matmul")(softmax, V)

    return FusionPattern(
        name="my_npu.attention",
        pattern=attention,
        annotation_patterns={
            "q": Q,
            "k": K,
            "v": V,
            "matmul_qk": matmul_qk,
            "softmax": softmax,
            "matmul_av": attention,
        },
    )
```

---

## 24.9 Multi-Backend Integration

### 24.9.1 Combining Multiple External Backends

When multiple external backends are available, patterns are prioritized by specificity and order:

```python
from tvm.contrib.cutlass import get_patterns as get_cutlass_patterns
from tvm.contrib.cublas import get_patterns as get_cublas_patterns

# Build combined pattern list
# Order matters: first match wins
all_patterns = []

# 1. CUTLASS patterns (most fusion capability)
all_patterns.extend(get_cutlass_patterns())

# 2. cuBLAS patterns (fallback for basic matmul)
all_patterns.extend(get_cublas_patterns())

# 3. Custom NPU patterns
all_patterns.extend([
    my_npu_matmul_bias_relu_pattern,
    my_npu_conv2d_bias_pattern,
])

# Apply combined patterns
mod = FuseOpsByPattern(
    all_patterns,
    bind_constants=False,
    annotate_codegen=True,
)(mod)

# Run codegen for all backends
mod = RunCodegen()(mod)
```

### 24.9.2 Backend Priority Resolution

When two backends can handle the same pattern, the priority is determined by:

1. **Pattern order in the list**: Patterns earlier in the list take priority.
2. **Pattern specificity**: More specific patterns (more operators) are preferred over less specific ones.
3. **Explicit priority**: Some backends register patterns with explicit priority levels.

```python
# Priority example:
# cutlass.matmul_bias_relu matches matmul + bias + relu (3 ops)
# cublas.matmul matches just matmul (1 op)
#
# If cutlass.matmul_bias_relu is listed first, it will match
# the 3-op pattern before cublas.matmul can match the matmul alone.

# If only matmul is present (no bias, no relu), then
# cutlass.matmul_bias_relu won't match, and cublas.matmul will.
```

### 24.9.3 CompileConfig for External Backends

```python
import tvm
from tvm import relax

# Enable/disable specific backends
with tvm.transform.PassContext(config={
    "relax.backend.use_cublas": True,
    "relax.backend.use_cudnn": True,
    "relax.backend.use_cutlass": True,
    "relax.backend.use_tensorrt": False,  # Disable TensorRT
}):
    exec_mod = relax.build(mod, target="nvidia/nvidia-a100")
```

### 24.9.4 Target-Specific Backend Selection

```python
# Different targets may prefer different backends

# For A100 (Ampere): CUTLASS with FP16/BF16 tensor cores
target_a100 = tvm.target.Target("nvidia/nvidia-a100")

# For T4 (Turing): cuBLAS with FP16
target_t4 = tvm.target.Target("nvidia/nvidia-t4")

# For Jetson (Xavier): cuDNN for conv, cuBLAS for matmul
target_jetson = tvm.target.Target("nvidia/jetson-xavier")
```

---

## 24.10 Dataflow Pattern Language (DPL) Reference

### 24.10.1 Core Pattern Constructors

```python
from tvm.relax.dpl import *

# Wildcard: matches any single expression node
x = wildcard()

# is_var: matches a specific variable by name
x = is_var("input_tensor")

# is_const: matches a constant
c = is_const()

# is_op: matches a specific operator call
add = is_op("relax.add")(x, y)

# is_tuple_get_item: matches tuple extraction at specific index
out = is_tuple_get_item(tuple_expr, index=0)

# is_call_tir: matches a call_tir invocation
call = is_call_tir(func_name_pattern, args_pattern)
```

### 24.10.2 Pattern Composition

```python
# Sequential composition: match operator chains
x = wildcard()
w = wildcard()
matmul = is_op("relax.matmul")(x, w)
bias = wildcard()
add = is_op("relax.add")(matmul, bias)

# Alternative patterns: match either of two patterns
# (use separate FusionPattern entries)

# Optional patterns: match with or without an operation
# (define two patterns, one with and one without)
```

### 24.10.3 Pattern with Attribute Matching

```python
# Match conv2d with specific attributes
x = wildcard()
w = wildcard()

def match_conv2d_3x3(ctx):
    """Only match 3x3 convolutions."""
    conv_call = ctx.matched_nodes["conv"]
    # Check kernel size in attributes
    kernel_size = conv_call.attrs.kernel_size
    return kernel_size == (3, 3)

conv = is_op("relax.nn.conv2d")(x, w)
pattern = FusionPattern(
    name="my_backend.conv2d_3x3",
    pattern=conv,
    annotation_patterns={"conv": conv},
    check=match_conv2d_3x3,
)
```

### 24.10.4 Helper Pattern Builders

```python
from tvm.relax.dpl.pattern import make_fused_bias_activation_pattern

# Build a matmul + bias + activation pattern in one call
pattern = make_fused_bias_activation_pattern(
    op_name="relax.matmul",
    activation="relax.nn.relu",
)

# Equivalent to:
# x = wildcard()
# w = wildcard()
# bias = wildcard()
# matmul = is_op("relax.matmul")(x, w)
# add = is_op("relax.add")(matmul, bias)
# relu = is_op("relax.nn.relu")(add)
```

---

## 24.11 Runtime Integration

### 24.11.1 External Library Runtime Wrappers

Each external backend generates runtime wrappers that bridge between TVM's PackedFunc calling convention and the external library API:

```python
# CUTLASS runtime wrapper (conceptual):
# The generated PackedFunc:
# 1. Extracts NDArray pointers from TVM arguments
# 2. Sets up cuBLAS workspace
# 3. Calls the CUTLASS kernel
# 4. Returns output NDArray

# cuBLAS runtime wrapper:
# 1. Creates cuBLAS handle (or reuses from cache)
# 2. Sets cuBLAS math mode (tensor core enablement)
# 3. Calls cublasGemmEx / cublasLtMatmul
# 4. Returns output NDArray
```

### 24.11.2 Handle Caching

External library handles (cuBLAS, cuDNN) are cached per-device to avoid repeated creation:

```python
# cuBLAS handle lifecycle:
# - Created lazily on first use for each device
# - Cached in thread-local storage
# - Reused across calls to the same device
# - Destroyed when the device context is cleaned up

# cuDNN handle lifecycle:
# - Similar to cuBLAS
# - Bound to a specific CUDA stream
# - Cached per (device, stream) pair
```

### 24.11.3 Workspace Management

Some external library calls require temporary workspace memory:

```python
# cuDNN convolution requires workspace for intermediate results
# TVM manages workspace allocation:
# 1. Query workspace size: cudnnGetConvolutionForwardWorkspaceSize()
# 2. Allocate workspace: TVM device memory allocator
# 3. Execute convolution with workspace
# 4. Workspace is freed or returned to the pool

# CUTLASS kernels may require workspace for split-K reduction
# Workspace is allocated from the TVM memory pool
```

### 24.11.4 Stream Ordering

External library calls must respect TVM's stream ordering:

```python
# All external library calls are enqueued on the current CUDA stream
# This ensures correct ordering with TVM-generated kernels

import tvm

dev = tvm.cuda(0)
stream = tvm.cuda.stream(dev.device_id)

with stream:
    # TVM kernel
    result1 = vm["tvm_kernel"](input)

    # External library call (enqueued on same stream)
    result2 = vm["cublas_matmul"](result1, weight)

    # Another TVM kernel (waits for cuBLAS via stream ordering)
    result3 = vm["tvm_postprocess"](result2)
```

---

## 24.12 Debugging BYOC

### 24.12.1 Inspecting Pattern Matching Results

```python
import tvm
from tvm import relax

# Apply FuseOpsByPattern and inspect the result
mod_partitioned = relax.transform.FuseOpsByPattern(
    patterns,
    annotate_codegen=True,
)(mod)

# Print all functions and their annotations
for gv, func in mod_partitioned.functions.items():
    print(f"Function: {gv.name_hint}")
    if hasattr(func, 'attrs') and func.attrs:
        print(f"  Codegen: {func.attrs.get('Codegen', 'default')}")
        print(f"  global_symbol: {func.attrs.get('global_symbol', 'N/A')}")
    print()

# Check which operations were matched
print(mod_partitioned.script())
```

### 24.12.2 Verifying Graph Partitioning

```python
# Count how many subgraphs were created for each backend
from collections import Counter

backend_counts = Counter()
for gv, func in mod_partitioned.functions.items():
    if hasattr(func, 'attrs') and func.attrs:
        backend = func.attrs.get('Codegen', None)
        if backend:
            backend_counts[backend] += 1

print("Subgraphs per backend:")
for backend, count in backend_counts.items():
    print(f"  {backend}: {count}")
```

### 24.12.3 Testing External Codegen Output

```python
# Test the generated external codegen module independently
import tvm

# Build the module
exec_mod = relax.build(mod, target="nvidia/nvidia-a100")

# Check that external functions are accessible
vm = relax.VirtualMachine(exec_mod, tvm.cuda(0))

# List all available functions
for name in dir(vm):
    if not name.startswith("_"):
        print(f"Available function: {name}")

# Test individual external function
import numpy as np
x = tvm.nd.array(np.random.randn(1, 128).astype("float32"), device=tvm.cuda(0))
w = tvm.nd.array(np.random.randn(128, 64).astype("float32"), device=tvm.cuda(0))
result = vm["main"](x, w)
print(f"Output shape: {result.shape}")
```

### 24.12.4 Common Issues and Solutions

```python
# Issue 1: Pattern not matching
# Solution: Verify pattern structure matches the actual graph
print("Graph structure:")
print(mod.script())  # Check actual operator names and connections

# Issue 2: Pattern matching but not being offloaded
# Solution: Check pattern priority and check function
# Make sure the pattern is listed before less specific patterns

# Issue 3: External codegen crashes
# Solution: Check data types, shapes, and device compatibility
# Verify that the external library supports the operation's parameters

# Issue 4: Runtime errors from external library
# Solution: Check library version compatibility
# Verify CUDA/cuDNN/cuBLAS version matches the compiled version
```

### 24.12.5 Verbose Logging

```python
import os
import tvm

# Enable verbose logging for BYOC
os.environ["TVM_LOG_DEBUG"] = "1"

# Or use specific debug flags
# Print pattern matching details
os.environ["TVM_FUSEOPS_PATTERN_DEBUG"] = "1"

# Print codegen details
os.environ["TVM_BYOC_DEBUG"] = "1"
```

---

## 24.13 Advanced Topics

### 24.13.1 Dynamic Shape Handling

External backends may have limited support for dynamic shapes:

```python
def check_dynamic_shapes(ctx):
    """Only offload operations with static shapes."""
    for name, pattern in ctx.annotation_patterns.items():
        node = ctx.matched_nodes[name]
        shape = node.struct_info.shape
        # Check if all dimensions are static (not symbolic)
        for dim in shape:
            if not isinstance(dim, tvm.tir.IntImm):
                return False  # Dynamic dimension found
    return True

pattern = FusionPattern(
    name="my_npu.matmul",
    pattern=matmul,
    check=check_dynamic_shapes,
)
```

### 24.13.2 Quantization and BYOC

```python
# Quantized operations can be offloaded to backends that support them
# CUTLASS supports INT8 matmul via DP4A instructions
# cuBLASLt supports INT8 and weight-only INT4

# Pattern for quantized matmul (INT8)
x_int8 = wildcard()
w_int8 = wildcard()
matmul = is_op("relax.matmul")(x_int8, w_int8)

# The codegen function checks data types and generates
# appropriate quantized kernels
```

### 24.13.3 Multi-GPU and Distributed BYOC

```python
# External backends can be used in multi-GPU settings
# The same patterns and codegen work across multiple devices

from tvm.runtime import disco

# Create a distributed session
session = disco.Session(num_workers=4, device_type="cuda")

# Each worker uses the same compiled module
# External library calls are made independently on each device
```

### 24.13.4 BYOC with Relax VM

```python
# The Relax VM transparently handles external backend calls
# External functions are loaded as PackedFunc within the VM

import tvm
from tvm import relax

# Build with external backends
exec_mod = relax.build(mod, target="nvidia/nvidia-a100")

# Create VM -- external functions are available alongside TVM kernels
vm = relax.VirtualMachine(exec_mod, tvm.cuda(0))

# Execute -- external library calls are transparent
result = vm["main"](input_tensor)
```

### 24.13.5 Fallback Behavior

When an external backend cannot handle a specific operation, TVM falls back to its own code generation:

```python
# The graph is partitioned into:
# 1. External backend subgraphs (handled by cuBLAS, CUTLASS, etc.)
# 2. TVM subgraphs (handled by TVM's TIR code generation)
#
# Both types of subgraphs are combined into a single runtime.Module
# and executed seamlessly by the Relax VM
```

---

## 24.14 Source Code Locations

| Component | Path |
|-----------|------|
| FuseOpsByPattern pass | `src/relax/transform/fuse_ops_by_pattern.cc` |
| RunCodegen pass | `src/relax/transform/run_codegen.cc` |
| DPL pattern language | `python/tvm/relax/dpl/` |
| FusionPattern class | `python/tvm/relax/transform.py` |
| CUTLASS integration | `python/tvm/contrib/cutlass/` |
| cuBLAS integration | `python/tvm/contrib/cublas.py` |
| cuDNN integration | `python/tvm/contrib/cudnn.py` |
| TensorRT integration | `python/tvm/contrib/tensorrt.py` |
| BYOC runtime utils | `src/runtime/contrib/` |
| External codegen registry | `src/relax/backend/` |
| Pattern matching engine | `src/relax/analysis/` |

---

## 24.15 Summary

BYOC and external library dispatch enable TVM to leverage vendor-optimized libraries while maintaining a unified compilation framework. The key components are:

| Component | Purpose |
|-----------|---------|
| FusionPattern | Declaratively describes subgraphs to offload |
| FuseOpsByPattern | Partitions the graph based on registered patterns |
| RunCodegen | Invokes external codegen for annotated subgraphs |
| CompileConfig | Enables/disables specific backends at build time |

The system is designed to be extensible: new backends can be added by registering patterns and a codegen function, without modifying TVM's core compilation pipeline. Multiple backends can coexist, with priority determined by pattern order and specificity.
