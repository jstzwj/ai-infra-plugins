# Tensor Intrinsics - Hardware-Specific Acceleration

This reference covers tensor intrinsics in TVM, which map computation blocks to hardware-specific instructions for accelerated execution on specialized units such as NVIDIA Tensor Cores, Intel VNNI, ARM dot product units, and Hexagon DSP vector units.

---

## 17.1 Overview

Tensor intrinsics provide a mechanism to replace computation blocks in a TIR schedule with hardware-accelerated implementations. Rather than generating scalar or simple vector operations, tensor intrinsics allow TVM to target matrix multiply-accumulate units, dot product engines, and other specialized hardware found in modern processors.

The core idea is:
1. Define (or use a pre-defined) tensor intrinsic that describes a computation pattern and its hardware implementation.
2. Use `sch.tensorize()` to apply the intrinsic to a matching scheduling block.
3. The code generator emits the appropriate hardware instruction instead of the original loop nest.

This abstraction enables TVM to generate code that leverages:
- **NVIDIA Tensor Cores** (wmma, mma instructions)
- **Intel x86 VNNI** (vpdpbusd, vpdpbusds)
- **ARM dot product** (vdot, vmlaldavxq)
- **Qualcomm Hexagon DSP** (HVX vector operations)

---

## 17.2 Intrinsic Structure

Every tensor intrinsic consists of two parts:

### 17.2.1 Computation Description (TIR Pattern)

The computation description defines the mathematical operation the intrinsic implements. It is expressed as a TIR `PrimFunc` that describes the loop nest, buffer accesses, and arithmetic pattern.

```python
import tvm
from tvm import tir
from tvm.script import tirx as T

# Example: description of a 16x16x16 matmul accumulate
@T.prim_func
def matmul_16x16x16_desc(
    A: T.Buffer((16, 16), "float16"),
    B: T.Buffer((16, 16), "float16"),
    C: T.Buffer((16, 16), "float32"),
) -> None:
    with T.sblock("matmul_16x16x16"):
        # Reduction initialization
        with T.init():
            for i, j in T.grid(16, 16):
                C[i, j] = T.float32(0)
        # Computation body
        for i, j, k in T.grid(16, 16, 16):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")
```

### 17.2.2 Implementation (Hardware Instruction)

The implementation describes how to emit the actual hardware instruction. It is also a `PrimFunc`, but uses special TIR constructs that map directly to the target instruction.

```python
@T.prim_func
def matmul_16x16x16_impl(
    A: T.Buffer((16, 16), "float16"),
    B: T.Buffer((16, 16), "float16"),
    C: T.Buffer((16, 16), "float32"),
) -> None:
    with T.sblock("matmul_16x16x16"):
        # T.call_intrin maps to the hardware instruction
        T.call_intrin(
            "float32",
            "tir.nvidia_mma_sync_16x16x16_f16f16f32",
            A.data,
            A.elem_offset,
            B.data,
            B.elem_offset,
            C.data,
            C.elem_offset,
        )
```

### 17.2.3 Registering the Intrinsic

```python
from tvm.tir import TensorIntrin

TensorIntrin.register(
    "nvidia_mma_16x16x16_f16f16f32",
    matmul_16x16x16_desc,
    matmul_16x16x16_impl,
)
```

---

## 17.3 CUDA Tensor Core Intrinsics

NVIDIA Tensor Cores provide mixed-precision matrix multiply-accumulate operations. TVM supports multiple Tensor Core instruction shapes and data type combinations.

### 17.3.1 WMMA Intrinsics (Compute Capability 7.0+)

The WMMA (Warp Matrix Multiply-Accumulate) API operates on 16x16x16 and 32x8x16 matrix fragments.

```python
from tvm.tir.tensor_intrin.cuda import (
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_SYNC_16x16x16_s8s8s32_INTRIN,
    WMMA_LOAD_16x16x16_f16_A_GLOBAL_INTRIN,
    WMMA_LOAD_16x16x16_f16_B_GLOBAL_INTRIN,
    WMMA_STORE_16x16x16_f16_GLOBAL_INTRIN,
    WMMA_FILL_16x16x16_f32_INTRIN,
)
```

### 17.3.2 MMA Sync Intrinsics (Compute Capability 7.5+, 8.0+)

The MMA (Matrix Multiply-Accumulate) sync instructions provide more flexible shapes:

| Intrinsic Name | Shape (M-N-K) | A Type | B Type | C Type | Min CC |
|---|---|---|---|---|---|
| `nvidia_mma_sync_16x16x16_f16f16f16` | 16x16x16 | float16 | float16 | float16 | 7.5 |
| `nvidia_mma_sync_16x16x16_f16f16f32` | 16x16x16 | float16 | float16 | float32 | 7.5 |
| `nvidia_mma_sync_16x8x16_f16f16f32` | 16x8x16 | float16 | float16 | float32 | 7.5 |
| `nvidia_mma_sync_16x8x32_s8s8s32` | 16x8x32 | int8 | int8 | int32 | 7.5 |
| `nvidia_mma_sync_16x16x16_b1b1s32` | 16x16x16 | bit | bit | int32 | 7.5 |
| `nvidia_mma_sync_16x16x128_b1b1s32` | 16x16x128 | bit | bit | int32 | 8.0 |

### 17.3.3 Complete CUDA Tensor Core Example

```python
import tvm
from tvm import tir
from tvm.script import ir as I, tirx as T

# Define a matmul with tensor core intrinsic
@I.ir_module
class MatMulTensorCore:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float16"),
        B: T.Buffer((1024, 1024), "float16"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Local buffers for Tensor Core fragments
        A_shared = T.alloc_buffer((1024, 1024), "float16", scope="shared")
        B_shared = T.alloc_buffer((1024, 1024), "float16", scope="shared")
        C_local = T.alloc_buffer((1024, 1024), "float32", scope="local")
        for bx0, by0 in T.grid(T.ceildiv(1024, 128), T.ceildiv(1024, 128)):
            for bx1, by1 in T.grid(T.ceildiv(128, 16), T.ceildiv(128, 16)):
                with T.sblock("C_o"):
                    vi = T.axis.spatial(1024, bx0 * 128 + by1 * 16)
                    vj = T.axis.spatial(1024, bx1 * 16)
                    T.reads(A[vi, 0:1024], B[0:1024, vj])
                    T.writes(C[vi, vj])
                    for k_0 in range(T.ceildiv(1024, 16)):
                        with T.sblock("C"):
                            vi_i = T.axis.spatial(16, bx0 * 128 + by1 * 16)
                            vj_j = T.axis.spatial(16, bx1 * 16)
                            vk_k = T.axis.reduce(16, k_0 * 16)
                            T.reads(
                                A[vi_i, vk_k],
                                B[vk_k, vj_j],
                            )
                            T.writes(C[vi_i, vj_j])
                            with T.init():
                                for ii, jj in T.grid(16, 16):
                                    C_local[vi_i + ii, vj_j + jj] = T.float32(0.0)
                            # This inner block will be tensorized
                            for ii, jj, kk in T.grid(16, 16, 16):
                                with T.sblock("C_update"):
                                    iii = T.axis.spatial(16, ii)
                                    jjj = T.axis.spatial(16, jj)
                                    kkk = T.axis.reduce(16, kk)
                                    C_local[vi_i + iii, vj_j + jjj] += (
                                        T.cast(A[vi_i + iii, vk_k + kkk], "float32")
                                        * T.cast(B[vk_k + kkk, vj_j + jjj], "float32")
                                    )

# Schedule and tensorize
sch = tvm.s_tir.Schedule(MatMulTensorCore)
block = sch.get_block("C_update")
# Apply the tensor intrinsic
sch.tensorize(block, "nvidia_mma_sync_16x16x16_f16f16f32")
```

### 17.3.4 Tensor Core Load/Store Intrinsics

Tensor Core operations also require specialized load and store intrinsics for moving data between global memory, shared memory, and fragment registers.

```python
# Load matrix A from shared memory to fragments
@T.prim_func
def wmma_load_a_16x16x16_f16_shared_desc(
    A_shared: T.Buffer((16, 16), "float16"),
    A_frag: T.Buffer((16, 16), "float16"),
) -> None:
    with T.sblock("load_a"):
        for i, j in T.grid(16, 16):
            with T.sblock("load"):
                vi, vj = T.axis.remap("SS", [i, j])
                A_frag[vi, vj] = A_shared[vi, vj]

@T.prim_func
def wmma_load_a_16x16x16_f16_shared_impl(
    A_shared: T.Buffer((16, 16), "float16"),
    A_frag: T.Buffer((16, 16), "float16"),
) -> None:
    with T.sblock("load_a"):
        T.call_intrin(
            "float16",
            "tir.nvidia_wmma_load_a_16x16x16_shared_f16",
            A_shared.data,
            A_shared.elem_offset,
            A_frag.data,
            A_frag.elem_offset,
        )

# Store from fragment to global memory
@T.prim_func
def wmma_store_16x16x16_f16_global_desc(
    C_frag: T.Buffer((16, 16), "float16"),
    C_global: T.Buffer((16, 16), "float16"),
) -> None:
    with T.sblock("store"):
        for i, j in T.grid(16, 16):
            with T.sblock("store_inner"):
                vi, vj = T.axis.remap("SS", [i, j])
                C_global[vi, vj] = C_frag[vi, vj]

@T.prim_func
def wmma_store_16x16x16_f16_global_impl(
    C_frag: T.Buffer((16, 16), "float16"),
    C_global: T.Buffer((16, 16), "float16"),
) -> None:
    with T.sblock("store"):
        T.call_intrin(
            "void",
            "tir.nvidia_wmma_store_16x16x16_global_f16",
            C_frag.data,
            C_frag.elem_offset,
            C_global.data,
            C_global.elem_offset,
        )
```

### 17.3.5 PTX-Level MMA Intrinsics (Compute Capability 8.0+)

For Ampere and Hopper architectures, TVM supports PTX-level MMA instructions with additional shapes:

```python
from tvm.tir.tensor_intrin.cuda import (
    MMA_SYNC_16x8x16_f16f16f32_INTRIN,
    MMA_SYNC_16x8x32_s8s8s32_INTRIN,
    LDMATRIX_32x8_f16_INTRIN,
    LDMATRIX_32x8_b16_INTRIN,
)
```

The `ldmatrix` intrinsic loads matrices from shared memory in the layout expected by MMA operations:

```python
# ldmatrix loads 32x8 elements (per warp) from shared memory
@T.prim_func
def ldmatrix_32x8_f16_desc(
    smem: T.Buffer((32, 8), "float16"),
    frag: T.Buffer((32, 8), "float16"),
) -> None:
    with T.sblock("ldmatrix"):
        for i, j in T.grid(32, 8):
            with T.sblock("ld"):
                vi, vj = T.axis.remap("SS", [i, j])
                frag[vi, vj] = smem[vi, vj]
```

---

## 17.4 x86 AVX512 VNNI Intrinsics

Intel's Vector Neural Network Instructions (VNNI) accelerate INT8 and UINT8 matrix multiplication on x86 CPUs.

### 17.4.1 Available VNNI Intrinsics

| Intrinsic | Description | Operation |
|---|---|---|
| `dp4a` | 4-element dot product accumulate | `sum += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]` |
| `vpdpbusd` | VNNI dot product (u8 x i8 -> i32) | Vectorized 4-element dot products |
| `vpdpbusds` | VNNI dot product with saturation | Same as vpdpbusd with saturation |

### 17.4.2 VNNI Intrinsic Definition

```python
@T.prim_func
def dot_16x4x4_i8i8i32_desc(
    A: T.Buffer((16, 4), "int8"),
    B: T.Buffer((4, 4), "int8"),
    C: T.Buffer((16, 4), "int32"),
) -> None:
    with T.sblock("dot"):
        with T.init():
            for i, j in T.grid(16, 4):
                C[i, j] = 0
        for i, j, k in T.grid(16, 4, 4):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + T.cast(A[vi, vk], "int32") * T.cast(B[vk, vj], "int32")

@T.prim_func
def dot_16x4x4_i8i8i32_vnni(
    A: T.Buffer((16, 4), "int8"),
    B: T.Buffer((4, 4), "int8"),
    C: T.Buffer((16, 4), "int32"),
) -> None:
    with T.sblock("dot"):
        T.call_intrin(
            "int32",
            "tir.x86_vpdpbusd_16x4x4",
            A.data,
            A.elem_offset,
            B.data,
            B.elem_offset,
            C.data,
            C.elem_offset,
        )
```

### 17.4.3 VNNI Tensorize Example

```python
import tvm
from tvm import tir
from tvm.script import ir as I, tirx as T
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4x4_INTRIN

@I.ir_module
class Conv2DVNNI:
    @T.prim_func
    def main(
        data: T.Buffer((1, 56, 56, 64), "uint8"),
        weight: T.Buffer((64, 3, 3, 64), "int8"),
        out: T.Buffer((1, 54, 54, 64), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for n, oh, ow, oc, kh, kw, ic in T.grid(1, 54, 54, 64, 3, 3, 64):
            with T.sblock("conv2d"):
                vn = T.axis.spatial(1, n)
                voh = T.axis.spatial(54, oh)
                vow = T.axis.spatial(54, ow)
                voc = T.axis.spatial(64, oc)
                vkh = T.axis.reduce(3, kh)
                vkw = T.axis.reduce(3, kw)
                vic = T.axis.reduce(64, ic)
                with T.init():
                    out[vn, voh, vow, voc] = 0
                out[vn, voh, vow, voc] += (
                    T.cast(data[vn, voh + vkh, vow + vkw, vic], "int32")
                    * T.cast(weight[voc, vkh, vkw, vic], "int32")
                )

sch = tvm.s_tir.Schedule(Conv2DVNNI)
# Tile the inner IC dimension to create 4-element groups for VNNI
block = sch.get_block("conv2d")
# ... tiling and scheduling steps ...
# Apply VNNI intrinsic to the inner reduction block
sch.tensorize(inner_block, VNNI_DOT_16x4x4_INTRIN)
```

### 17.4.4 AMX (Advanced Matrix Extensions) Intrinsics

For Intel Sapphire Rapids and later processors with AMX:

```python
from tvm.tir.tensor_intrin.x86 import (
    AMX_TILE_CONFIG_INTRIN,
    AMX_TILE_DPBF16PS_INTRIN,  # BF16 dot product
    AMX_TILE_DPBSSD_INTRIN,    # INT8 dot product
)
```

AMX intrinsics operate on tile registers (up to 16 tiles, each up to 16x64 bytes):

```python
# AMX BF16 matmul: 16x16x2 (bf16 pairs) -> 16x16 int32
@T.prim_func
def amx_dpbf16ps_16x16x2_desc(
    A: T.Buffer((16, 32), "bf16"),  # 16 rows, 32 bf16 = 64 bytes
    B: T.Buffer((32, 16), "bf16"),  # 32 rows, 16 bf16
    C: T.Buffer((16, 16), "int32"),
) -> None:
    with T.sblock("amx_dot"):
        with T.init():
            for i, j in T.grid(16, 16):
                C[i, j] = 0
        for i, j, k in T.grid(16, 16, 2):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] += T.cast(A[vi, vk * 2], "int32") * T.cast(B[vk * 2, vj], "int32") + \
                             T.cast(A[vi, vk * 2 + 1], "int32") * T.cast(B[vk * 2 + 1, vj], "int32")
```

---

## 17.5 ARM Dot Product Intrinsics

ARM v8.4+ and v8.6+ introduce dot product and matrix multiplication instructions.

### 17.5.1 Available ARM Intrinsics

| Instruction | ISA | Operation | Data Types |
|---|---|---|---|
| `sdot` | ARM v8.4+ | Signed dot product | int8 x int8 -> int32 |
| `udot` | ARM v8.4+ | Unsigned dot product | uint8 x uint8 -> uint32 |
| `usdot` | ARM v8.6+ | Mixed sign dot product | uint8 x int8 -> int32 |
| `bf16mlal` | ARM v8.6+ | BF16 multiply-accumulate | bf16 x bf16 -> fp32 |
| `i8mm` | ARM v8.6+ | INT8 matrix multiply | int8 x uint8 -> int32 |

### 17.5.2 ARM Dot Product Intrinsic Definition

```python
from tvm.tir.tensor_intrin.arm import (
    DOT_4x4x16_I8I8I32_INTRIN,
    DOT_4x4x16_U8U8U32_INTRIN,
)

# ARM dot product: 4x4 outer, 16 inner reduction
@T.prim_func
def arm_dot_4x4x16_i8i8i32_desc(
    A: T.Buffer((4, 16), "int8"),
    B: T.Buffer((16, 4), "int8"),
    C: T.Buffer((4, 4), "int32"),
) -> None:
    with T.sblock("dot"):
        with T.init():
            for i, j in T.grid(4, 4):
                C[i, j] = 0
        for i, j, k in T.grid(4, 4, 16):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] += T.cast(A[vi, vk], "int32") * T.cast(B[vk, vj], "int32")

@T.prim_func
def arm_dot_4x4x16_i8i8i32_impl(
    A: T.Buffer((4, 16), "int8"),
    B: T.Buffer((16, 4), "int8"),
    C: T.Buffer((4, 4), "int32"),
) -> None:
    with T.sblock("dot"):
        T.call_intrin(
            "int32",
            "tir.arm_sdot_4x4x16",
            A.data,
            A.elem_offset,
            B.data,
            B.elem_offset,
            C.data,
            C.elem_offset,
        )
```

### 17.5.3 ARM Matrix Multiply (MMLA) Intrinsics

ARM v8.6+ introduces the outer product and matrix multiply instructions:

```python
from tvm.tir.tensor_intrin.arm import (
    AARCH64_MATMUL_4x4x16_I8I8I32_INTRIN,  # smmla
    AARCH64_MATMUL_4x4x16_BF16BF16F32_INTRIN,  # bfmmla
)

# BF16 matmul on ARM
@T.prim_func
def aarch64_bfmmla_4x4x16_desc(
    A: T.Buffer((4, 16), "bf16"),
    B: T.Buffer((16, 4), "bf16"),
    C: T.Buffer((4, 4), "float32"),
) -> None:
    with T.sblock("matmul"):
        with T.init():
            for i, j in T.grid(4, 4):
                C[i, j] = T.float32(0.0)
        for i, j, k in T.grid(4, 4, 16):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] += T.cast(A[vi, vk], "float32") * T.cast(B[vk, vj], "float32")
```

### 17.5.4 Complete ARM Example

```python
import tvm
from tvm import tir, target
from tvm.script import ir as I, tirx as T

@I.ir_module
class DenseARM:
    @T.prim_func
    def main(
        A: T.Buffer((128, 256), "int8"),
        W: T.Buffer((256, 512), "int8"),
        Out: T.Buffer((128, 512), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 512, 256):
            with T.sblock("dense"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(512, j)
                vk = T.axis.reduce(256, k)
                with T.init():
                    Out[vi, vj] = 0
                Out[vi, vj] += T.cast(A[vi, vk], "int32") * T.cast(W[vk, vj], "int32")

sch = tvm.s_tir.Schedule(DenseARM)
block = sch.get_block("dense")
i, j, k = sch.get_loops(block)

# Tile to match ARM dot product shape: 4x4 outer, 16 inner
i_o, i_i = sch.split(i, factors=[None, 4])
j_o, j_i = sch.split(j, factors=[None, 4])
k_o, k_i = sch.split(k, factors=[None, 16])
sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

block_inner = sch.get_block("dense")
# Tensorize the innermost block
sch.tensorize(block_inner, "arm_dot_4x4x16_i8i8i32")
```

---

## 17.6 Hexagon DSP Intrinsics

Qualcomm Hexagon DSP provides HVX (Hexagon Vector eXtensions) for SIMD processing. TVM supports intrinsics targeting HVX instructions.

### 17.6.1 Available Hexagon Intrinsics

| Intrinsic | HVX Width | Data Types | Operation |
|---|---|---|---|
| `dot_32x4x8_u8u8i32` | 128 bytes | uint8 x uint8 -> int32 | 4-element dot product |
| `dot_32x4x8_i8i8i32` | 128 bytes | int8 x int8 -> int32 | Signed dot product |
| `vrmpy` | 128 bytes | uint8 x uint8 -> int32 | Vector round multiply |
| `conv2d_nhwc` | Variable | uint8/int8 | 2D convolution |

### 17.6.2 Hexagon HVX Dot Product

```python
from tvm.tir.tensor_intrin.hexagon import (
    VRMPY_32x4x8_U8U8I32_INTRIN,
    VRMPY_32x4x8_I8I8I32_INTRIN,
    VRMPY_16x4x8_U8U8I32_INTRIN,
)

# Hexagon VRMPY: 32 rows x 4 cols outer, 8 inner reduction
@T.prim_func
def vrmpy_32x4x8_u8u8i32_desc(
    A: T.Buffer((32, 8), "uint8"),
    B: T.Buffer((8, 4), "uint8"),
    C: T.Buffer((32, 4), "int32"),
) -> None:
    with T.sblock("vrmpy"):
        with T.init():
            for i, j in T.grid(32, 4):
                C[i, j] = 0
        for i, j, k in T.grid(32, 4, 8):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] += T.cast(A[vi, vk], "int32") * T.cast(B[vk, vj], "int32")
```

### 17.6.3 Hexagon Conv2D Intrinsic

```python
from tvm.tir.tensor_intrin.hexagon import VRMPY_U8U8I32_INTRIN  # "dot_32x4x8_u8u8i32"

# Conv2D with HVX acceleration
@I.ir_module
class Conv2DHexagon:
    @T.prim_func
    def main(
        data: T.Buffer((1, 56, 56, 32), "uint8"),
        kernel: T.Buffer((32, 3, 3, 32), "int8"),
        out: T.Buffer((1, 54, 54, 32), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for n, oh, ow, oc, kh, kw, ic in T.grid(1, 54, 54, 32, 3, 3, 32):
            with T.sblock("conv2d"):
                vn = T.axis.spatial(1, n)
                voh = T.axis.spatial(54, oh)
                vow = T.axis.spatial(54, ow)
                voc = T.axis.spatial(32, oc)
                vkh = T.axis.reduce(3, kh)
                vkw = T.axis.reduce(3, kw)
                vic = T.axis.reduce(32, ic)
                with T.init():
                    out[vn, voh, vow, voc] = 0
                out[vn, voh, vow, voc] += (
                    T.cast(data[vn, voh + vkh, vow + vkw, vic], "int32")
                    * T.cast(kernel[voc, vkh, vkw, vic], "int32")
                )

sch = tvm.s_tir.Schedule(Conv2DHexagon)
# Apply HVX intrinsic after appropriate tiling
block = sch.get_block("conv2d")
# ... tiling steps to create 32x4x8 inner block ...
sch.tensorize(inner_block, VRMPY_32x4x8_U8U8I32_INTRIN)
```

---

## 17.7 Using sch.tensorize()

The `tensorize` schedule primitive replaces a scheduling block's computation with a hardware intrinsic.

### 17.7.1 Signature

```python
def tensorize(
    block: Union[BlockRV, str],  # The block or block name to tensorize
    intrin: Union[str, TensorIntrin],  # Intrinsic name or TensorIntrin object
) -> None:
    """Replace the computation of a block with a tensor intrinsic."""
```

### 17.7.2 Matching Rules

The tensorize operation succeeds only when the block's computation pattern matches the intrinsic's description:
- The buffer access patterns must match (same shapes, same data types).
- The loop nest structure must correspond.
- The arithmetic operations must be equivalent.

If the patterns do not match, TVM raises a `TVMError` with a diagnostic message.

### 17.7.3 Manual Tensorize Workflow

```python
import tvm
from tvm import tir

# Step 1: Create or load the module
mod = ...  # IRModule with a matmul or conv2d

# Step 2: Create schedule
sch = tvm.s_tir.Schedule(mod)

# Step 3: Find the block to tensorize
block = sch.get_block("matmul")  # or the name of your compute block

# Step 4: Tile to match intrinsic shape
i, j, k = sch.get_loops(block)
i_o, i_i = sch.split(i, factors=[None, 16])
j_o, j_i = sch.split(j, factors=[None, 16])
k_o, k_i = sch.split(k, factors=[None, 16])

# Reorder for outer loops and inner tensorize block
sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

# Get the innermost block after decomposition
inner_block = sch.get_block("matmul_inner")  # name varies after decomposition

# Step 5: Apply tensorize
sch.tensorize(inner_block, "nvidia_mma_sync_16x16x16_f16f16f32")
```

### 17.7.4 Checking Available Intrinsics

```python
from tvm.tir import TensorIntrin

# List all registered tensor intrinsics
all_intrins = TensorIntrin.list_intrins()
for name in all_intrins:
    print(name)

# Get a specific intrinsic
intrin = TensorIntrin.get("nvidia_mma_sync_16x16x16_f16f16f32")
print(intrin.desc)   # The computation description PrimFunc
print(intrin.impl)   # The implementation PrimFunc
```

---

## 17.8 Defining Custom Tensor Intrinsics

When pre-defined intrinsics do not cover your hardware target, you can define custom tensor intrinsics.

### 17.8.1 Custom Intrinsic Template

```python
from tvm.script import tirx as T
from tvm.tir import TensorIntrin

# Step 1: Define the computation description
@T.prim_func
def my_custom_dot_8x8x8_desc(
    A: T.Buffer((8, 8), "float32"),
    B: T.Buffer((8, 8), "float32"),
    C: T.Buffer((8, 8), "float32"),
) -> None:
    # Must use sblock for the outer block
    with T.sblock("root"):
        with T.init():
            for i, j in T.grid(8, 8):
                C[i, j] = T.float32(0.0)
        for i, j, k in T.grid(8, 8, 8):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] += A[vi, vk] * B[vk, vj]

# Step 2: Define the implementation
@T.prim_func
def my_custom_dot_8x8x8_impl(
    A: T.Buffer((8, 8), "float32"),
    B: T.Buffer((8, 8), "float32"),
    C: T.Buffer((8, 8), "float32"),
) -> None:
    with T.sblock("root"):
        # Use T.call_intrin with a unique name
        T.call_intrin(
            "float32",
            "tir.my_custom_dot_8x8x8",
            A.data,
            A.elem_offset,
            B.data,
            B.elem_offset,
            C.data,
            C.elem_offset,
        )

# Step 3: Register the intrinsic
TensorIntrin.register(
    "my_custom_dot_8x8x32_f32f32f32",
    my_custom_dot_8x8x8_desc,
    my_custom_dot_8x8x8_impl,
)
```

### 17.8.2 Implementing the Codegen Hook

For custom intrinsics that use `T.call_intrin`, you must implement the code generation in C++:

```cpp
// In src/target/source/literal/cuda.cc or similar backend file
void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::call_intrin())) {
        std::string name = op->args[0].as<StringImmNode>()->value;
        if (name == "tir.my_custom_dot_8x8x8") {
            // Emit custom assembly or intrinsic call
            this->PrintIndent();
            stream << "my_custom_asm_instruction(";
            // ... emit operands from op->args ...
            stream << ");\n";
            return;
        }
    }
    // Default handling
}
```

### 17.8.3 Custom Intrinsic with Inline Assembly

For targets that support inline assembly emission:

```python
@T.prim_func
def my_custom_dot_8x8x8_asm_impl(
    A: T.Buffer((8, 8), "float32"),
    B: T.Buffer((8, 8), "float32"),
    C: T.Buffer((8, 8), "float32"),
) -> None:
    with T.sblock("root"):
        # Use T.evaluate with a call to emit inline assembly
        T.evaluate(
            T.tvm_call_pure_extern(
                "int32",
                "my_custom_dot_8x8x8_kernel",
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                C.data,
                C.elem_offset,
            )
        )
```

---

## 17.9 Integration with MetaSchedule

MetaSchedule can automatically select and apply tensor intrinsics during the auto-tuning search process.

### 17.9.1 Tensorize in MetaSchedule Rules

```python
from tvm.tir.tensor_intrin.cuda import (
    MMA_SYNC_16x16x16_f16f16f32_INTRIN,
    MMA_SYNC_16x8x16_f16f16f32_INTRIN,
)
from tvm.meta_schedule.schedule_rule import AutoInline, AutoTensorize

# AutoTensorize rule tries to match blocks against registered intrinsics
auto_tensorize = AutoTensorize(
    intrins=[
        MMA_SYNC_16x16x16_f16f16f32_INTRIN,
        MMA_SYNC_16x8x16_f16f16f32_INTRIN,
    ],
)
```

### 17.9.2 MetaSchedule with Tensor Core Targeting

```python
import tvm
from tvm import relax, meta_schedule as ms
from tvm.target import Target

# Configure MetaSchedule task for Tensor Core
target = Target("nvidia/nvidia-a100")

# The database will include tensorize candidates
database = ms.tune_tir(
    mod=mod,
    target=target,
    work_dir="./tune_logs",
    max_trials_global=1000,
    # Tensorize rules are automatically included for CUDA targets
)

# Build with the best configuration (may include tensorize)
sch = ms.compile_tir(database, mod, target)
```

### 17.9.3 DLight Tensorize Rules

DLight provides pre-defined schedule rules that include tensorize for common operations:

```python
from tvm.dlight import gpu

# DLight rules automatically apply tensor intrinsics when beneficial
with tvm.target.Target("nvidia/nvidia-a100"):
    sch = tvm.dlight.base.normalize(sch)
    # The DLight GPU rule may apply tensor intrinsics internally
    rule = gpu.MatmulTensorCore()
    sch = rule.apply(sch, target)
```

### 17.9.4 Tensorize Strategy Selection

MetaSchedule selects tensorize strategies based on:
1. **Target hardware**: Determines available intrinsics.
2. **Data types**: Matches intrinsic data type requirements.
3. **Block shape**: Checks if the block shape is divisible by the intrinsic shape.
4. **Performance model**: Estimates whether tensorize will improve performance.

```python
# Inspect the available tensor intrinsics for a target
from tvm.tir.tensor_intrin.cuda import get_cuda_tensor_intrins
from tvm.tir.tensor_intrin.x86 import get_x86_tensor_intrins
from tvm.tir.tensor_intrin.arm import get_arm_tensor_intrins

# Get intrins matching a specific dtype
cuda_fp16_intrins = get_cuda_tensor_intrins(
    a_dtype="float16",
    b_dtype="float16",
    c_dtype="float32",
)
```

---

## 17.10 Pre-registered Intrinsics Reference

### 17.10.1 CUDA Intrinsics

| Name | Shape | A dtype | B dtype | C dtype | Min CC |
|---|---|---|---|---|---|
| `nvidia_mma_sync_16x16x16_f16f16f16` | 16x16x16 | float16 | float16 | float16 | 7.0 |
| `nvidia_mma_sync_16x16x16_f16f16f32` | 16x16x16 | float16 | float16 | float32 | 7.0 |
| `nvidia_mma_sync_16x8x16_f16f16f32` | 16x8x16 | float16 | float16 | float32 | 7.5 |
| `nvidia_mma_sync_16x8x32_s8s8s32` | 16x8x32 | int8 | int8 | int32 | 7.5 |
| `nvidia_mma_sync_16x16x128_b1b1s32` | 16x16x128 | bit | bit | int32 | 8.0 |
| `nvidia_wmma_load_a_16x16x16_shared_f16` | 16x16 | float16 | - | float16 | 7.0 |
| `nvidia_wmma_load_b_16x16x16_shared_f16` | 16x16 | - | float16 | float16 | 7.0 |
| `nvidia_wmma_store_16x16x16_global_f16` | 16x16 | - | - | float16 | 7.0 |
| `nvidia_wmma_fill_16x16x16_f32` | 16x16 | - | - | float32 | 7.0 |
| `nvidia_ldmatrix_32x8_f16` | 32x8 | float16 | - | float16 | 7.5 |
| `nvidia_ldmatrix_32x8_b16` | 32x8 | bit16 | - | bit16 | 7.5 |

### 17.10.2 x86 Intrinsics

| Name | Shape | A dtype | B dtype | C dtype | ISA |
|---|---|---|---|---|---|
| `x86_dot_16x4x4_i8i8i32` | 16x4x4 | int8 | int8 | int32 | VNNI |
| `x86_dot_16x4x4_u8u8i32` | 16x4x4 | uint8 | uint8 | int32 | VNNI |
| `x86_dp4a_i8i8i32` | 4x1x4 | int8 | int8 | int32 | AVX2 |
| `x86_amx_tile_dpbf16ps` | 16x16x2 | bf16 | bf16 | int32 | AMX |

### 17.10.3 ARM Intrinsics

| Name | Shape | A dtype | B dtype | C dtype | ISA |
|---|---|---|---|---|---|
| `arm_dot_4x4x16_i8i8i32` | 4x4x16 | int8 | int8 | int32 | v8.4+ |
| `arm_dot_4x4x16_u8u8i32` | 4x4x16 | uint8 | uint8 | int32 | v8.4+ |
| `aarch64_matmul_4x4x16_i8i8i32` | 4x4x16 | int8 | int8 | int32 | v8.6+ |
| `aarch64_matmul_4x4x16_bf16bf16f32` | 4x4x16 | bf16 | bf16 | float32 | v8.6+ |

### 17.10.4 Hexagon Intrinsics

| Name | Shape | A dtype | B dtype | C dtype | Unit |
|---|---|---|---|---|---|
| `hexagon_vrmpy_32x4x8_u8u8i32` | 32x4x8 | uint8 | uint8 | int32 | HVX |
| `hexagon_vrmpy_32x4x8_i8i8i32` | 32x4x8 | int8 | int8 | int32 | HVX |
| `hexagon_vrmpy_16x4x8_u8u8i32` | 16x4x8 | uint8 | uint8 | int32 | HVX |

---

## 17.11 Debugging Tensorize Failures

### 17.11.1 Common Issues

**Shape mismatch**: The block's loop extents must exactly match the intrinsic's description shape. Use `sch.split()` and `sch.reorder()` to create inner blocks of the correct size.

**Data type mismatch**: The intrinsic requires specific data types (e.g., float16 inputs for Tensor Core). Ensure buffers use the correct dtype before tensorizing.

**Buffer layout mismatch**: Some intrinsics expect specific memory layouts (row-major vs. column-major). The buffer access pattern in the block must match.

### 17.11.2 Diagnostic Tools

```python
# Print the block's access pattern
block = sch.get_block("my_block")
block_stmt = sch.get(block)
print(block_stmt)

# Check intrinsic requirements
intrin = TensorIntrin.get("nvidia_mma_sync_16x16x16_f16f16f32")
print("Description:")
print(intrin.desc.script())
print("Implementation:")
print(intrin.impl.script())

# Verify the schedule before tensorize
sch.show()  # Display the current TIR after scheduling
```

### 17.11.3 Verifying Tensorize Correctness

```python
import numpy as np

# Build both tensorized and non-tensorized versions
mod_tensorized = sch.tensorize(block, intrin)
mod_reference = sch_reference  # unscheduled reference

# Compare outputs
dev = tvm.cuda(0)
a_np = np.random.uniform(size=(M, K)).astype("float16")
b_np = np.random.uniform(size=(K, N)).astype("float16")

# Execute both and compare
rt_mod_tensorized = tvm.build(mod_tensorized, target="cuda")
rt_mod_reference = tvm.build(mod_reference, target="cuda")

# ... allocate, copy, run, compare ...
np.testing.assert_allclose(
    result_tensorized.numpy(),
    result_reference.numpy(),
    rtol=1e-3,
    atol=1e-3,
)
```

---

## 17.12 Source Code Locations

| Component | Path |
|---|---|
| TensorIntrin class | `python/tvm/tir/tensor_intrin.py` |
| CUDA intrinsics | `python/tvm/tir/tensor_intrin/cuda.py` |
| x86 intrinsics | `python/tvm/tir/tensor_intrin/x86.py` |
| ARM intrinsics | `python/tvm/tir/tensor_intrin/arm.py` |
| Hexagon intrinsics | `python/tvm/tir/tensor_intrin/hexagon.py` |
| Tensorize schedule primitive | `src/tir/schedule/primitive/tensorize.cc` |
| Tensorize C++ intrin matching | `src/tir/schedule/instruction.cc` |
| CUDA codegen intrin handling | `src/target/source/literal/cuda.cc` |
| LLVM codegen intrin handling | `src/target/llvm/llvm.cc` |
| MetaSchedule AutoTensorize | `python/tvm/meta_schedule/schedule_rule/auto_tensorize.py` |
