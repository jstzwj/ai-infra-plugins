# CUTLASS - Chapter 29: Python Interface

This reference covers the Python bindings and interfaces for CUTLASS, including PyCUTLASS (the high-level Python API), the CuTe DSL (Python-native interface for CUDA kernels), the `cutlass_library` Python package for kernel generation, and the `cutlass_cppgen` C++ code generation module.

---

## 29.1 Overview

CUTLASS provides several Python interfaces that serve different purposes:

| Component | Purpose | Location |
|-----------|---------|----------|
| PyCUTLASS | High-level GEMM/Conv Python API | `python/cutlass/` |
| CuTe DSL | Python-native CuTe kernel writing | `python/cute/` |
| cutlass_library | Operation/kernel emission and library generation | `tools/library/scripts/` |
| cutlass_cppgen | C++ code generation from Python | `python/cutlass/` |

These Python interfaces allow users to:
- Define and launch GEMM and convolution operations from Python.
- Write custom CUDA kernels using the CuTe DSL with Python syntax.
- Generate CUTLASS kernel libraries without manual C++ coding.
- Perform rapid prototyping and experimentation with kernel configurations.

---

## 29.2 Installation and Setup

### 29.2.1 Prerequisites

```bash
# Python dependencies
pip install numpy cuda-python

# CUTLASS must be cloned and built
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Set CUTLASS_PATH environment variable
export CUTLASS_PATH=$(pwd)
```

### 29.2.2 Installing PyCUTLASS

```bash
# Install PyCUTLASS from the CUTLASS repository
cd python
pip install -e .

# Verify installation
python -c "import cutlass; print(cutlass.__version__)"
```

### 29.2.3 Installing CuTe DSL

```bash
# CuTe DSL is part of the Python package
cd python
pip install -e .

# Verify CuTe DSL
python -c "import cute; print('CuTe DSL available')"
```

### 29.2.4 Setting Up the Library Generator

```bash
# The cutlass_library scripts are in tools/library/
# Add to Python path
export PYTHONPATH=$CUTLASS_PATH/tools/library/scripts:$PYTHONPATH

# Verify
python -c "import cutlass_library; print('Library generator available')"
```

### 29.2.5 CUDA Toolkit Requirements

```bash
# PyCUTLASS requires CUDA 11.8+ (CUDA 12+ recommended)
# Ensure nvcc is in PATH
nvcc --version

# For CuTe DSL, CUDA 12.0+ is required
# For Blackwell features, CUDA 13.0+ may be needed
```

---

## 29.3 PyCUTLASS: High-Level GEMM/Conv API

### 29.3.1 Basic GEMM Example

PyCUTLASS provides a high-level interface for defining and running GEMM operations:

```python
import cutlass
import numpy as np

# Configure the GEMM operation
plan = cutlass.op.Gemm(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.RowMajor,
    layout_C=cutlass.LayoutType.RowMajor,
)

# Set math instruction (Tensor Core operation)
plan.math_instruction = cutlass.MathInstruction(
    math_operation=cutlass.MathOperation.multiply_add,
    element_accumulator=cutlass.float32,
    instruction_shape=(16, 8, 16),
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float32,
)

# Set the tile description
plan.tile_description = cutlass.TileDescription(
    threadblock_shape=(128, 128, 64),
    stages=3,
    warp_count=(2, 2, 1),
)

# Set architecture
plan.arch = 80  # SM80 (Ampere)

# Create input tensors
M, N, K = 1024, 1024, 1024
A = np.random.randn(M, K).astype(np.float16)
B = np.random.randn(K, N).astype(np.float16)
C = np.zeros((M, N), dtype=np.float16)
D = np.zeros((M, N), dtype=np.float16)

# Run the GEMM
plan.run(A, B, C, D, alpha=1.0, beta=0.0)

print(f"GEMM result shape: {D.shape}")
print(f"Result sample: {D[0, :5]}")
```

### 29.3.2 Using the CollectiveBuilder (SM90+)

For Hopper and later architectures, use the CollectiveBuilder-based API:

```python
import cutlass
from cutlass import Gemm, LayoutType, DataType
from cutlass.swizzle import ThreadblockSwizzle

# Define GEMM with CollectiveBuilder approach
gemm = cutlass.GemmUniversal(
    element_A=DataType.e4m3,        # FP8 E4M3
    element_B=DataType.e4m3,        # FP8 E4M3
    element_C=DataType.float16,     # FP16
    element_D=DataType.float16,     # FP16
    layout_A=LayoutType.RowMajor,
    layout_B=LayoutType.ColumnMajor,
    arch=90,                         # SM90 (Hopper)
    kernel_schedule="TmaWarpSpecialized",
    epilogue_schedule="TmaWarpSpecialized",
)

# Problem size
M, N, K = 4096, 4096, 4096

# Allocate tensors
A = np.random.randn(M, K).astype(np.float8_e4m3)
B = np.random.randn(K, N).astype(np.float8_e4m3)
C = np.zeros((M, N), dtype=np.float16)
D = np.zeros((M, N), dtype=np.float16)

# Run
gemm.run(A, B, C, D, alpha=1.0, beta=0.0, M=M, N=N, K=K)
```

### 29.3.3 Convolution Operations

```python
import cutlass
import numpy as np

# Define a Conv2D operation
conv = cutlass.op.Conv2d(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    layout_A=cutlass.LayoutType.TensorNHWC,
    layout_B=cutlass.LayoutType.TensorNHWC,
)

# Set problem size
N, H, W, C = 32, 56, 56, 64
K, R, S = 64, 3, 3

# Create tensors
activation = np.random.randn(N, H, W, C).astype(np.float16)
filter = np.random.randn(K, R, S, C).astype(np.float16)
output = np.zeros((N, H, W, K), dtype=np.float16)

# Run convolution
conv.run(
    activation, filter, output,
    padding=(1, 1),
    stride=(1, 1),
    dilation=(1, 1),
)
```

### 29.3.4 Epilogue Fusion from Python

PyCUTLASS supports epilogue fusion directly from Python:

```python
import cutlass
import numpy as np

# GEMM with fused bias + ReLU
plan = cutlass.op.Gemm(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.RowMajor,
)

# Create a bias tensor
M, N, K = 1024, 1024, 1024
bias = np.random.randn(1, N).astype(np.float16)

# Fuse bias addition
plan.activation = cutlass.epilogue.relu
plan.bias = bias

# Alternative: Use composed epilogue operations
from cutlass.epilogue import EpilogueFunctor

plan.epilogue_functor = EpilogueFunctor.compose(
    EpilogueFunctor.bias_add(),
    EpilogueFunctor.relu(),
)
```

### 29.3.5 Batched GEMM

```python
import cutlass
import numpy as np

# Batched GEMM
plan = cutlass.op.Gemm(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.RowMajor,
)

batch_size = 16
M, N, K = 512, 512, 512

A_batch = np.random.randn(batch_size, M, K).astype(np.float16)
B_batch = np.random.randn(batch_size, K, N).astype(np.float16)
C_batch = np.zeros((batch_size, M, N), dtype=np.float16)
D_batch = np.zeros((batch_size, M, N), dtype=np.float16)

plan.run(
    A_batch, B_batch, C_batch, D_batch,
    alpha=1.0, beta=0.0,
    batch_size=batch_size,
)
```

---

## 29.4 CuTe DSL: Python-Native Interface

### 29.4.1 Overview

The CuTe DSL provides a Python-native interface for writing CUDA kernels using CuTe abstractions. It translates Python code into CUDA C++ kernels that use the CuTe library internally.

```python
from cute import *

# Define a simple copy kernel using CuTe DSL
@cute_kernel
def copy_kernel(
    src: cute.Tensor,
    dst: cute.Tensor,
    src_layout: cute.Layout,
    dst_layout: cute.Layout,
):
    # Define the tiled copy
    tiled_copy = make_tiled_copy(
        Copy_Atom(SM80_U32Copy(), src.element_type),
        Layout((4, 8), (8, 1)),   # 32 threads in a 4x8 arrangement
        Layout((1, 1), (1, 1))    # Each thread copies 1 element
    )

    # Get the thread slice
    thr_copy = tiled_copy.get_slice(thread_idx())
    tSr = thr_copy.partition_S(src)
    tDd = thr_copy.partition_D(dst)

    # Copy
    copy(tiled_copy, tSr, tDd)
```

### 29.4.2 CuTe Layout and Tensor in Python

The CuTe DSL mirrors the C++ CuTe abstractions:

```python
from cute import *

# Create a CuTe layout
# Layout<Shape, Stride>
layout = make_layout(
    shape=(128,),
    stride=(1,)
)

# 2D layout (row-major)
layout_rm = make_layout(
    shape=(128, 64),
    stride=(1, 128)
)

# 2D layout (column-major)
layout_cm = make_layout(
    shape=(128, 64),
    stride=(64, 1)
)

# Compose layouts
composed = composition(layout_rm, layout_cm)

# Create a tensor with layout
tensor = make_tensor(
    ptr=device_ptr,
    layout=layout_rm,
    memory_space=cute.MemorySpace.Global
)

# Slice and partition tensors
slice = tensor[slice_coord]
partitioned = partition(tensor, thread_layout)
```

### 29.4.3 CuTe Algorithms in Python

```python
from cute import *

# Copy algorithm
@cute_kernel
def copy_example(src, dst):
    # Create a vectorized copy atom
    copy_atom = Copy_Atom(SM80_Copy_16B(), cutlass.float16)

    # Create tiled copy for the threadblock
    tiled = make_tiled_copy(
        copy_atom,
        make_layout((4, 8)),
        make_layout((1, 1))
    )

    # Partition and copy
    thr_slice = tiled.get_slice(thread_idx())
    copy(tiled, thr_slice.partition_S(src), thr_slice.partition_D(dst))

# GEMM algorithm using CuTe
@cute_kernel
def gemm_example(A, B, C, M, N, K):
    # Define MMA atom for Tensor Core
    mma_atom = MMA_Atom(
        SM80_16x8x16_F16F16F16(),
        a=make_layout((1, 1)),
        b=make_layout((1, 1)),
        c=make_layout((1, 1))
    )

    # Create tiled MMA
    tiled_mma = make_tiled_mma(
        mma_atom,
        make_layout((2, 2, 1))  # 2x2 warp tiling
    )

    # Get thread's partition of the MMA
    thr_mma = tiled_mma.get_slice(thread_idx())

    # Partition A, B, C
    tCrA = thr_mma.partition_A(A)
    tCrB = thr_mma.partition_B(B)
    tCrC = thr_mma.partition_C(C)

    # Initialize accumulator
    accumulator = zeros_like(tCrC)

    # GEMM mainloop
    for k in range(0, K, K_TILE):
        # Load A and B tiles
        copy(tCrA[:, :, k], tAsA)
        copy(tCrB[:, :, k], tBsB)

        # MMA
        gemm(tiled_mma, tAsA, tBsB, accumulator)

    # Store result
    copy(accumulator, tCrC)
```

### 29.4.4 Code Generation Pipeline

The CuTe DSL translates Python code through several stages:

1. **Python AST parsing**: The Python source is parsed into an abstract syntax tree.
2. **Type inference**: CuTe types (Layout, Tensor, etc.) are inferred from annotations and usage.
3. **C++ code generation**: The AST is translated to CUDA C++ code using CuTe C++ templates.
4. **Compilation**: The generated C++ is compiled with nvcc.
5. **Loading**: The compiled kernel is loaded as a Python-callable function.

```python
# The code generation pipeline is invoked automatically
# but can also be controlled manually:

from cute.compiler import compile_kernel

# Compile a kernel to PTX/CUBIN
kernel_binary = compile_kernel(
    copy_kernel,
    arch="sm_80",
    optimize=True,
)

# Inspect generated C++ code
print(kernel_binary.cpp_source)

# Inspect PTX
print(kernel_binary.ptx)
```

### 29.4.5 CuTe DSL Example: Softmax Kernel

```python
from cute import *

@cute_kernel
def softmax_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    M: int,
    N: int,
):
    """Flash-style softmax using CuTe DSL"""
    # Each threadblock handles one row
    block_id = block_idx()
    if block_id >= M:
        return

    # Get the row
    row_in = input[block_id, :]
    row_out = output[block_id, :]

    # Tile the row for the threadblock
    tile_size = 128
    tiled_copy = make_tiled_copy(
        Copy_Atom(SM80_Copy_16B(), cutlass.float16),
        make_layout((32,)),
        make_layout((4,))
    )

    # Phase 1: Find max
    max_val = -float('inf')
    thr_copy = tiled_copy.get_slice(thread_idx())

    for tile_start in range(0, N, tile_size):
        tile = row_in[tile_start:tile_start + tile_size]
        t_tile = thr_copy.partition_S(tile)
        # Reduce max within tile
        for i in range(t_tile.size()):
            max_val = max(max_val, float(t_tile[i]))

    # Warp-level max reduction
    max_val = warp_reduce(max_val, Maximum())

    # Phase 2: Compute exp and sum
    exp_sum = 0.0
    for tile_start in range(0, N, tile_size):
        tile = row_in[tile_start:tile_start + tile_size]
        t_tile = thr_copy.partition_S(tile)
        for i in range(t_tile.size()):
            exp_val = expf(float(t_tile[i]) - max_val)
            exp_sum += exp_val

    exp_sum = warp_reduce(exp_sum, Plus())

    # Phase 3: Write output
    inv_sum = 1.0 / exp_sum
    for tile_start in range(0, N, tile_size):
        tile_in = row_in[tile_start:tile_start + tile_size]
        tile_out = row_out[tile_start:tile_start + tile_size]
        t_in = thr_copy.partition_S(tile_in)
        t_out = thr_copy.partition_D(tile_out)
        for i in range(t_in.size()):
            t_out[i] = cutlass.float16(
                expf(float(t_in[i]) - max_val) * inv_sum
            )
```

---

## 29.5 Python cutlass_library: Operation Generation

### 29.5.1 Overview

The `cutlass_library` Python package generates CUTLASS kernel implementations by emitting C++ source code from Python descriptions. This is the same infrastructure used by the CUTLASS build system to generate the kernel library.

```python
import cutlass_library
from cutlass_library import *

# The library generator creates:
# 1. C++ header files with kernel type aliases
# 2. C++ source files with kernel instantiations
# 3. CMake files for building the generated library
```

### 29.5.2 Operation Generation

```python
from cutlass_library import *

# Define a GEMM operation
gemm_op = GemmOperation(
    arch=80,
    tile_description=TileDescription(
        threadblock_shape=[128, 128, 32],
        warp_count=[2, 2, 1],
        stages=3,
        instruction_shape=[16, 8, 16],
    ),
    A=TensorDescription(
        element=DataType.f16,
        layout=LayoutType.RowMajor,
    ),
    B=TensorDescription(
        element=DataType.f16,
        layout=LayoutType.ColumnMajor,
    ),
    C=TensorDescription(
        element=DataType.f32,
        layout=LayoutType.RowMajor,
    ),
    element_epilogue=DataType.f32,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    iterator_A=IteratorType.Optimized,
    iterator_B=IteratorType.Optimized,
)

# Generate C++ code for this operation
emitter = GemmEmitter()
cpp_code = emitter.emit(gemm_op)
print(cpp_code)
```

### 29.5.3 Library Generation

```python
from cutlass_library import *

# Generate a complete library of GEMM kernels
operations = []

# Generate operations for multiple architectures and data types
for arch in [80, 90]:
    for dtype_a, dtype_b, dtype_c in [
        (DataType.f16, DataType.f16, DataType.f32),
        (DataType.bf16, DataType.bf16, DataType.f32),
        (DataType.tf32, DataType.tf32, DataType.f32),
        (DataType.e4m3, DataType.e4m3, DataType.f32),
    ]:
        for tb_shape in [(128, 128, 64), (128, 64, 64), (64, 128, 64)]:
            for stages in [3, 4]:
                op = GemmOperation(
                    arch=arch,
                    tile_description=TileDescription(
                        threadblock_shape=list(tb_shape),
                        warp_count=[2, 2, 1],
                        stages=stages,
                    ),
                    A=TensorDescription(element=dtype_a, layout=LayoutType.RowMajor),
                    B=TensorDescription(element=dtype_b, layout=LayoutType.ColumnMajor),
                    C=TensorDescription(element=dtype_c, layout=LayoutType.RowMajor),
                    element_epilogue=DataType.f32,
                    epilogue_functor=EpilogueFunctor.LinearCombination,
                )
                operations.append(op)

# Generate the library
generate_library(
    operations=operations,
    output_dir="generated_library",
    library_name="my_cutlass_lib",
)
```

### 29.5.4 Emitters for Kernel Instantiation

The library uses several emitter classes:

| Emitter | Purpose |
|---------|---------|
| `GemmEmitter` | Generate GEMM kernel type aliases and instantiations |
| `ConvEmitter` | Generate Conv kernel type aliases and instantiations |
| `SparseGemmEmitter` | Generate sparse GEMM kernels |
| `LibraryEmitter` | Generate the top-level library header and CMake |

```python
from cutlass_library import GemmEmitter, ConvEmitter

# Use individual emitters for fine-grained control
gemm_emitter = GemmEmitter()

# Emit kernel type alias
type_alias = gemm_emitter.emit_type_alias(gemm_op)
print(type_alias)

# Emit kernel instantiation
instantiation = gemm_emitter.emit_instantiation(gemm_op)
print(instantiation)

# Emit the complete kernel source file
source = gemm_emitter.emit_source(gemm_op)
print(source)
```

---

## 29.6 Python cutlass_cppgen: C++ Code Generation

### 29.6.1 Overview

The `cutlass_cppgen` module generates C++ code that can be compiled and linked into applications. It provides a programmatic way to create CUTLASS kernel configurations without writing C++ templates directly.

```python
from cutlass.cppgen import *

# Create a GEMM kernel generator
generator = GemmGenerator()

# Configure the kernel
generator.set_arch(90)
generator.set_data_types(
    element_A="float_e4m3_t",
    element_B="float_e4m3_t",
    element_C="float16_t",
    element_D="float16_t",
    accumulator="float",
)
generator.set_layouts(
    layout_A="RowMajor",
    layout_B="ColumnMajor",
    layout_C="RowMajor",
)
generator.set_tile(
    threadblock_shape=(128, 128, 64),
    warp_count=(2, 2, 1),
    stages=0,  # auto
)
generator.set_schedule(
    kernel_schedule="KernelTmaWarpSpecialized",
    epilogue_schedule="EpilogueTmaWarpSpecialized",
)

# Generate C++ code
cpp_code = generator.generate()
print(cpp_code)
```

### 29.6.2 Generating Header Files

```python
from cutlass.cppgen import *

# Generate a header file with kernel type definitions
header_gen = HeaderGenerator()

header_gen.add_include("cutlass/cutlass.h")
header_gen.add_include("cutlass/gemm/device/gemm_universal_adapter.h")
header_gen.add_include("cutlass/gemm/collective/collective_builder.hpp")

# Define kernel type
header_gen.add_typedef(
    name="MyGemmKernel",
    definition="""cutlass::gemm::kernel::GemmUniversal<
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
            cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
            float,
            cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::collective::StageCountAutoCarveout<0>,
            cutlass::gemm::collective::KernelScheduleAuto
        >::CollectiveOp,
        cutlass::epilogue::collective::DefaultEpilogue<
            cutlass::layout::RowMajor, cutlass::layout::RowMajor,
            cutlass::epilogue::collective::EpilogueScheduleAuto
        >
    >"""
)

# Write header
header_gen.write("my_gemm_kernel.h")
```

### 29.6.3 Generating CMakeLists

```python
from cutlass.cppgen import CMakeGenerator

cmake_gen = CMakeGenerator(project_name="my_cutlass_kernels")
cmake_gen.set_minimum_version(3.18)
cmake_gen.set_cuda_architectures([80, 90])
cmake_gen.add_cutlass_dependency("/path/to/cutlass")
cmake_gen.add_source("my_gemm_kernel.cu")
cmake_gen.add_source("main.cpp")
cmake_gen.write("CMakeLists.txt")
```

---

## 29.7 Python Example Workflows

### 29.7.1 Rapid Kernel Prototyping

```python
import cutlass
import numpy as np

# Quick prototype: Test different tile sizes
tile_sizes = [
    (64, 64, 64),
    (128, 128, 32),
    (128, 128, 64),
    (256, 128, 64),
]

M, N, K = 4096, 4096, 4096
A = np.random.randn(M, K).astype(np.float16)
B = np.random.randn(K, N).astype(np.float16)
C = np.zeros((M, N), dtype=np.float16)
D = np.zeros((M, N), dtype=np.float16)

results = {}

for tb_shape in tile_sizes:
    plan = cutlass.op.Gemm(
        element_A=cutlass.float16,
        element_B=cutlass.float16,
        element_C=cutlass.float16,
        element_D=cutlass.float16,
        layout_A=cutlass.LayoutType.RowMajor,
        layout_B=cutlass.LayoutType.RowMajor,
    )
    plan.tile_description = cutlass.TileDescription(
        threadblock_shape=tb_shape,
        stages=3,
        warp_count=(2, 2, 1),
    )
    plan.arch = 80

    # Measure time
    import time
    start = time.time()
    plan.run(A, B, C, D, alpha=1.0, beta=0.0)
    elapsed = time.time() - start

    gflops = 2 * M * N * K / (elapsed * 1e9)
    results[tb_shape] = gflops
    print(f"Tile {tb_shape}: {gflops:.1f} GFLOPS")

# Find best configuration
best = max(results, key=results.get)
print(f"Best tile: {best} with {results[best]:.1f} GFLOPS")
```

### 29.7.2 Custom Epilogue Pipeline

```python
import cutlass
import numpy as np

# GEMM with custom epilogue: GELU activation
plan = cutlass.op.Gemm(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.RowMajor,
)

# Define GELU activation
# GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
from cutlass.epilogue import EpilogueOp

plan.epilogue_functor = EpilogueOp.compose(
    EpilogueOp.linear_combination(alpha=1.0, beta=0.0),
    EpilogueOp.gelu(),
)

M, N, K = 2048, 2048, 2048
A = np.random.randn(M, K).astype(np.float16)
B = np.random.randn(K, N).astype(np.float16)
C = np.zeros((M, N), dtype=np.float16)
D = np.zeros((M, N), dtype=np.float16)

plan.run(A, B, C, D, alpha=1.0, beta=0.0)
```

### 29.7.3 FP8 Mixed Precision Workflow

```python
import cutlass
import numpy as np

# FP8 GEMM on Hopper (SM90)
plan = cutlass.op.Gemm(
    element_A=cutlass.e4m3,     # FP8 input A
    element_B=cutlass.e4m3,     # FP8 input B
    element_C=cutlass.float16,  # FP16 source
    element_D=cutlass.float16,  # FP16 output
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.ColumnMajor,
    accumulator=cutlass.float32,  # FP32 accumulation
    arch=90,
)

M, N, K = 8192, 8192, 8192

# FP8 inputs (simulated with uint8 for storage)
A = np.random.randint(0, 256, (M, K), dtype=np.uint8)
B = np.random.randint(0, 256, (K, N), dtype=np.uint8)
C = np.zeros((M, N), dtype=np.float16)
D = np.zeros((M, N), dtype=np.float16)

plan.run(A, B, C, D, alpha=1.0, beta=0.0)
```

---

## 29.8 Migration from Deprecated Python API

### 29.8.1 API Changes

CUTLASS has evolved its Python interface over versions. Here are key migration points:

**Old (deprecated) API:**
```python
# Deprecated: Direct cutlass bindings
import cutlass_bindings as cutlass

gemm = cutlass.Gemm(
    A_type=cutlass.half,
    B_type=cutlass.half,
    C_type=cutlass.float,
    # ...
)
```

**New (current) API:**
```python
# Current: PyCUTLASS high-level API
import cutlass

plan = cutlass.op.Gemm(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    # ...
)
```

### 29.8.2 Migration Checklist

| Old API | New API | Notes |
|---------|---------|-------|
| `cutlass_bindings.Gemm` | `cutlass.op.Gemm` | Use the op module |
| `cutlass.half` | `cutlass.float16` | Use NumPy-style type names |
| `cutlass.Layout.RowMajor` | `cutlass.LayoutType.RowMajor` | Use LayoutType enum |
| Manual kernel compilation | Automatic compilation | No need to manually invoke nvcc |
| `cutlass_library.GemmOperation` | `cutlass.op.Gemm` + `cutlass.cppgen` | Combined API |

### 29.8.3 Backward Compatibility

```python
# The new API maintains backward compatibility for common use cases
# Old-style operation definition still works but emits deprecation warnings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                       module="cutlass")
```

---

## 29.9 Advanced Python Features

### 29.9.1 Custom Kernel Launch Configuration

```python
import cutlass

plan = cutlass.op.Gemm(
    element_A=cutlass.float16,
    element_B=cutlass.float16,
    element_C=cutlass.float16,
    element_D=cutlass.float16,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.RowMajor,
)

# Custom launch configuration
plan.launch_config = cutlass.LaunchConfig(
    grid=(32, 32, 1),
    block=(128, 1, 1),
    shared_memory=32768,
    stream=cuda_stream,
)

plan.run(A, B, C, D, alpha=1.0, beta=0.0)
```

### 29.9.2 Stream and Event Management

```python
import cutlass
import cupy as cp  # or use cuda-python

# Create CUDA streams
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

# Run GEMM on specific stream
plan = cutlass.op.Gemm(...)
plan.run(A, B, C, D, stream=stream1)

# Overlap with other operations
plan2 = cutlass.op.Gemm(...)
plan2.run(E, F, G, H, stream=stream2)
```

### 29.9.3 Profiling from Python

```python
import cutlass
import numpy as np

# Profile GEMM operations
plan = cutlass.op.Gemm(...)
plan.arch = 80

# Warmup
for _ in range(10):
    plan.run(A, B, C, D)

# Benchmark
num_iters = 100
start_event = cutlass.CUDAEvent()
end_event = cutlass.CUDAEvent()

start_event.record()
for _ in range(num_iters):
    plan.run(A, B, C, D)
end_event.record()
end_event.synchronize()

elapsed_ms = start_event.elapsed_time(end_event) / num_iters
gflops = 2 * M * N * K / (elapsed_ms * 1e6)
print(f"Average time: {elapsed_ms:.3f} ms")
print(f"Performance: {gflops:.1f} GFLOPS")
```

---

## 29.10 Summary

CUTLASS Python interfaces provide multiple levels of abstraction:

- **PyCUTLASS** (`cutlass.op.Gemm`, `cutlass.op.Conv2d`): High-level API for defining and running GEMM/Conv operations. Supports epilogue fusion, batched operations, and multiple data types.

- **CuTe DSL** (`cute`): Low-level Python-native interface for writing custom CUDA kernels using CuTe abstractions (Layout, Tensor, Copy_Atom, MMA_Atom). Translates Python to CUDA C++ via code generation.

- **cutlass_library**: Python package for generating CUTLASS kernel libraries. Used by the build system and for custom kernel library generation.

- **cutlass_cppgen**: C++ code generation from Python descriptions. Generates header files, source files, and CMake configurations.

The recommended workflow is:
1. Start with PyCUTLASS for standard GEMM/Conv operations.
2. Use the CuTe DSL for custom kernels that require fine-grained control.
3. Use cutlass_library/cppgen for generating production kernel libraries.
