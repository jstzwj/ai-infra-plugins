# TileLang ROCm/AMD Backend Reference

## 1. Overview

The ROCm backend provides high-performance code generation targeting AMD GPUs via
the HIP programming model. It supports both CDNA (Compute DNA) and RDNA (Radeon DNA)
architectures, leveraging AMD's Matrix Core instructions (MFMA/WMMA) for accelerated
matrix operations. The backend is designed to be a near-peer to the CUDA backend,
offering equivalent functionality for AMD hardware.

### Architecture Diagram

```
TileLang IR (PrimFunc)
        |
        v
+-------------------+
|  Backend Selection |  ---> Target = "hip" / "rocm"
+-------------------+
        |
        v
+-------------------+
|  Transform Passes |  (lower_tile_op, lower_ptx_async_copy, ...)
+-------------------+
        |
        v
+-------------------+
| CodeGenTileLangHIP |  (codegen_hip.cc)
+-------------------+
        |
        v
+-------------------+
| HIPRTC/hipcc       |  (hiprtc.py, hipcc.py)
+-------------------+
        |
        v
HIP Kernel (.hip)
```

---

## 2. ROCm Backend Architecture

### 2.1 Code Generator: CodeGenTileLangHIP

The HIP code generator is defined in `src/backend/rocm/codegen/codegen_hip.h`:

```cpp
class CodeGenTileLangHIP final : public CodeGenC {
public:
  CodeGenTileLangHIP();
  std::string Finish();
  void SetTarget(Target target) { target_ = std::move(target); }
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f, std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope, std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final;
  void BindThreadIndex(const IterVar &iv) final;
  // ... more overrides
};
```

### 2.2 Key Differences from CUDA Backend

| Feature | CUDA | ROCm/HIP |
|---|---|---|
| Warp size | 32 threads | 64 threads (wavefront) |
| Tensor Core | MMA/WGMMA/TCGEN05 | MFMA/WMMA |
| Async copy | cp.async | buffer_load (CDNA) |
| Shared memory | `__shared__` | `__shared__` (LDS) |
| Tensor memory | TMEM (SM100+) | Not available |
| Synchronization | `__syncthreads()` | `__syncthreads()` |
| Shuffle | `__shfl_sync` | `__shfl` |

### 2.3 Header Management

```cpp
bool need_cooperative_groups_{false};
bool need_math_constants_h_{false};
bool need_wmma_h_{false};
bool enable_fp8_{false};
bool need_mma_h_{false};
bool need_cast_smem_ptr_to_int_{false};
```

### 2.4 Barrier Support

```cpp
const std::string barrier_name_ = "barrier";
const int barrier_alignment_bytes_ = 16;
int barrier_count_ = -1;
```

---

## 3. HIP Code Generation

### 3.1 Function Generation

HIP kernel functions follow a similar pattern to CUDA:

```hip
extern "C" __global__ void kernel_name(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C,
    ...)
{
    // kernel body
}
```

### 3.2 Thread Index Binding

```cpp
void BindThreadIndex(const IterVar &iv) final;
```

Maps TileLang thread variables to HIP built-in variables. The key difference
from CUDA is the wavefront size:

- HIP uses 64 threads per wavefront vs 32 threads per warp in CUDA
- `threadIdx.x` ranges 0..63 within a wavefront
- Group size for MFMA operations is based on wavefront (64 threads)

### 3.3 Type Printing

```cpp
void PrintType(DataType t, std::ostream &os) final;
```

Maps TileLang data types to HIP types:

| TileLang Type | HIP Type |
|---|---|
| float16 | `half` |
| bfloat16 | `hip_bfloat16` or `__hip_bfloat16` |
| float32 | `float` |
| float64 | `double` |
| int8 | `int8_t` |
| int32 | `int32_t` |
| float8_e4m3 | `__hip_fp8_e4m3` (if available) |
| float8_e5m2 | `__hip_fp8_e5m2` (if available) |

### 3.4 Storage Scope Mapping

| TileLang Scope | HIP Address Space |
|---|---|
| `local` | Registers (default) |
| `shared` | `__shared__` (LDS) |
| `shared.dyn` | `__shared__` (dynamic LDS) |
| `global` | Global memory |

### 3.5 Synchronization

```cpp
void PrintStorageSync(const CallNode *op) final;
```

Block-level synchronization maps to:

```hip
__syncthreads();
```

---

## 4. MFMA (Matrix Fused Multiply-Add) Instructions

### 4.1 Overview

MFMA is AMD's primary Matrix Core instruction for CDNA architectures. It performs
matrix multiply-accumulate operations at the wavefront level (64 threads).

**Available on:** gfx908 (MI100), gfx90a (MI250X), gfx942 (MI300X), gfx950

### 4.2 Instruction Selection

In `src/backend/rocm/op/gemm.cc`:

```cpp
static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (TargetIsCDNA(target)) {
        return kROCmMFMA;    // "rocm.mfma"
    }
    if (TargetIsRDNA(target)) {
        return kROCmWMMA;    // "rocm.wmma"
    }
    LOG(FATAL) << "Unsupported ROCm target for gemm: " << target->str();
}
```

### 4.3 Supported MFMA Instruction Shapes

MFMA supports several instruction shapes depending on data type:

#### Float16

```hip
// 16x16x16 (4 elements per thread)
__builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);

// 32x32x8 (8 elements per thread)
__builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
```

#### BFloat16

```hip
// 16x16x16 (4 elements per thread)
__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(b_vec, a_vec, c, 0, 0, 0);

// 32x32x8 (8 elements per thread)
__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(b_vec, a_vec, c, 0, 0, 0);
```

#### Int8

```hip
// 16x16x32 (4 elements per thread, 2 int8 packed into 1 int16)
__builtin_amdgcn_mfma_i32_16x16x32_i8(b_packed, a_packed, c, 0, 0, 0);

// 32x32x16 (8 elements per thread)
__builtin_amdgcn_mfma_i32_32x32x16_i8(b_packed, a_packed, c, 0, 0, 0);
```

#### FP8 (gfx940+)

```hip
// 16x16x32 (E4M3 x E4M3)
__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b_val, a_val, c, 0, 0, 0);
```

### 4.4 MFMA Layout

The MFMA emitter uses specific layouts to map between shared memory and the
register layout expected by MFMA instructions.

From `tilelang/intrinsics/mfma_layout.py`:

```python
# MFMA layout mappings for shared-to-register conversion
shared_16x4_to_local_64x1_layout_A     # 16x4 shared -> 64x1 register
shared_4x16_to_local_64x1_layout_B     # 4x16 shared -> 64x1 register
shared_16x16_to_local_64x4_layout_A    # 16x16 shared -> 64x4 register
shared_16x16_to_local_64x4_layout_B
shared_16x32_to_local_64x8_layout_A
shared_16x32_to_local_64x8_layout_B
shared_16x64_to_local_64x16_layout_A
shared_16x64_to_local_64x16_layout_B
shared_32x32_to_local_64x16_layout_A
shared_32x32_to_local_64x16_layout_B
```

And the corresponding thread-id access layouts:

```python
thread_id_shared_access_64x1_to_16x4_layout_A
thread_id_shared_access_64x1_to_4x16_layout_B
thread_id_shared_access_64x4_to_16x16_layout_A
thread_id_shared_access_64x4_to_16x16_layout_B
thread_id_shared_access_64x8_to_16x32_layout_A
thread_id_shared_access_64x8_to_16x32_layout_B
thread_id_shared_access_64x16_to_16x64_layout_A
thread_id_shared_access_64x16_to_16x64_layout_B
thread_id_shared_access_64x16_to_32x32_layout_A
thread_id_shared_access_64x16_to_32x32_layout_B
```

### 4.5 MFMA Intrinsic Emitter

From `tilelang/intrinsics/mfma_macro_generator.py`:

```python
class MatrixCoreIntrinEmitter:
    WARP_SIZE = 64  # Wavefront size for AMD

    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e4m3fn": "e4m3fn",
        "float8_e4m3fnuz": "e4m3fnuz",
        "float8_e5m2fnuz": "e5m2fnuz",
    }

    k_pack = 1  # Vectorization factor for MFMA instructions
```

**k_pack** represents the number of elements in a vectorized MFMA instruction.
This is analogous to the `kPack` concept in AMD's Triton backend.

### 4.6 MFMA Micro-Size Configuration

```python
# From mfma_macro_generator.py
micro_size_x = 16   # M dimension of MFMA atom
micro_size_y = 16   # N dimension of MFMA atom
micro_size_k = 32 / sizeof(A_type)  # K dimension depends on data type

# vec_size determines the vector load width
vec_size = 8 / sizeof(A_type)  # 8 bytes per vector load
```

For different data types:

| A_type | micro_size_k | vec_size |
|---|---|---|
| float16 | 16 | 4 |
| bfloat16 | 16 | 4 |
| int8 | 32 | 8 |
| float8_e4m3 | 32 | 8 |

### 4.7 C++ MFMA Template

From `src/tl_templates/hip/gemm.h`:

```cpp
template <int M, int N, int K, int num_warp_m, int num_warp_n,
          bool TransposeA, bool TransposeB, bool clear_accum,
          int kPack, typename A_type, typename B_type, typename C_type,
          typename AccDataType = float>
class GemmTensorOp {
public:
    static constexpr int micro_size_x = 16;
    static constexpr int micro_size_y = 16;
    static constexpr int micro_size_k = 32 / sizeof(A_type);
    static constexpr int vec_size = 8 / sizeof(A_type);
```

#### MFMA Traits

Specialized traits for each data type:

```cpp
template <> struct MfmaTraits<int8_t> {
    static TL_DEVICE void mfma_op(const int8_t *b, const int8_t *a, AccType *c) {
        int64_t *b_packed = reinterpret_cast<int64_t *>(const_cast<int8_t *>(b));
        int64_t *a_packed = reinterpret_cast<int64_t *>(const_cast<int8_t *>(a));
        *c = __builtin_amdgcn_mfma_i32_16x16x32_i8(*b_packed, *a_packed, *c, 0, 0, 0);
    }
};

template <> struct MfmaTraits<half> {
    static TL_DEVICE void mfma_op(const half *b, const half *a, AccType *c) {
        *c = __builtin_amdgcn_mfma_f32_16x16x16f16(
            *((float16x4 *)b), *((float16x4 *)a), *c, 0, 0, 0);
    }
};

template <> struct MfmaTraits<bfloat16_t> {
    static TL_DEVICE void mfma_op(const bfloat16_t *b, const bfloat16_t *a, AccType *c) {
        bfloat16x4_vec b_vec, a_vec;
        short *b_short = reinterpret_cast<short *>(const_cast<bfloat16_t *>(b));
        short *a_short = reinterpret_cast<short *>(const_cast<bfloat16_t *>(a));
        for (int i = 0; i < 4; ++i) {
            b_vec[i] = b_short[i];
            a_vec[i] = a_short[i];
        }
        *c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(b_vec, a_vec, *c, 0, 0, 0);
    }
};
```

---

## 5. WMMA (Warp Matrix Multiply-Accomulate) Instructions

### 5.1 Overview

WMMA is AMD's cooperative matrix multiply instruction for RDNA architectures
(consumer GPUs). It operates at a smaller granularity than MFMA.

**Available on:** gfx1100 (RDNA 3), gfx1200 (RDNA 4)

### 5.2 Instruction Selection

WMMA is selected for RDNA targets:

```cpp
if (TargetIsRDNA(target)) {
    return kROCmWMMA;    // "rocm.wmma"
}
```

### 5.3 WMMA vs MFMA Comparison

| Feature | MFMA (CDNA) | WMMA (RDNA) |
|---|---|---|
| Wavefront size | 64 threads | 32 threads |
| Primary use | Data center compute | Consumer graphics |
| Data types | fp16, bf16, int8, fp8 | fp16, bf16 |
| Instruction size | 16x16x16, 32x32x8 | 16x16x16 |
| Accumulator | fp32 | fp16, fp32 |

### 5.4 WMMA Intrinsic Emitter

From `tilelang/intrinsics/wmma_macro_generator.py`:

```python
class MatrixCoreIntrinEmitter:
    # WMMA uses smaller warp size
    WARP_SIZE = 32  # RDNA uses 32 threads per wavefront
```

### 5.5 WMMA Layout

From `tilelang/intrinsics/wmma_layout.py`:

```python
# WMMA-specific layouts for RDNA architectures
# These differ from MFMA layouts due to the different wavefront size
```

---

## 6. register_gemm_impl for ROCm

### 6.1 Python Registration

In `tilelang/backend/rocm/gemm.py`:

```python
from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_mfma import GEMM_INST_MFMA, GemmMFMA
from tilelang.tileop.gemm.gemm_wmma import GEMM_INST_WMMA, GemmWMMA
from tilelang.utils.target import target_is_hip

def _match_mfma(target) -> bool:
    return target_is_hip(target)

def _match_wmma(target) -> bool:
    return target_is_hip(target)

register_gemm_impl("rocm.mfma", GEMM_INST_MFMA, _match_mfma, GemmMFMA)
register_gemm_impl("rocm.wmma", GEMM_INST_WMMA, _match_wmma, GemmWMMA)
```

### 6.2 C++ Registration

In `src/backend/rocm/op/gemm.cc`:

```cpp
bool MatchROCmGemmTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmGemm() {
    RegisterGemmImpl(GemmImpl{
        "rocm.Gemm",
        MatchROCmGemmTarget,
        rocm::Gemm::SelectInst,
        rocm::Gemm::ComputeWarpPartition,
        rocm::Gemm::ReuseExistingSharedLayout,
        rocm::Gemm::InstructionKind,
    });
    return true;
}
```

### 6.3 Warp Partition

MFMA/WMMA use the same default warp partition logic:

```cpp
std::pair<int, int> ComputeDefaultWarpPartition(
    const GemmWarpPolicyNode &policy, int M, int N, int num_warps)
{
    constexpr int kMPerWarp = 16;
    constexpr int kNPerWarp = 16;
    // Supports FullRow, FullCol, and Square policies
    // ...
}
```

Note that for MFMA, `num_warps = block_size / 64` (wavefront size is 64),
and for WMMA, `num_warps = block_size / 32`.

### 6.4 ReuseExistingSharedLayout

```cpp
static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;  // ROCm always creates new shared layouts
}
```

Unlike the CUDA MMA backend which can reuse existing shared memory layouts,
the ROCm backend always creates fresh layouts. This is because MFMA/WMMA
register layouts differ significantly from shared memory layouts.

### 6.5 Instruction Kind

```cpp
static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kROCmMFMA) return "mfma";
    if (gemm_inst == kROCmWMMA) return "wmma";
    return "unknown";
}
```

---

## 7. Shared Memory Optimization for AMD GPUs

### 7.1 Local Data Share (LDS)

AMD GPUs use LDS (Local Data Share) as their shared memory, equivalent to
CUDA's shared memory. Key characteristics:

| Property | Value |
|---|---|
| Banks | 32 (CDNA) or 64 (some RDNA) |
| Bank width | 4 bytes |
| Total size | 64 KB (CDNA3) |
| Latency | ~30 cycles |
| Bandwidth | ~10 TB/s (CDNA3) |

### 7.2 LDS Layout for MFMA

MFMA requires specific shared memory layouts to ensure efficient loading:

```python
# For 16x16 MFMA with float16:
# A matrix layout: 16 rows x K columns
# B matrix layout: K rows x 16 columns
# K is padded to ensure no bank conflicts

# Layout selection is based on:
# - micro_size_x (M dimension of MFMA atom)
# - micro_size_y (N dimension of MFMA atom)
# - Data type size
# - Vector load width
```

### 7.3 Bank Conflict Avoidance

For MFMA, bank conflicts are avoided through:

1. **Padding:** Adding extra columns to ensure each row starts at a different bank
2. **Swizzling:** XOR-based patterns for power-of-2 dimensions
3. **Access pattern ordering:** Ensuring wavefront threads access different banks

### 7.4 LDS Access Patterns

```hip
// Vectorized LDS loads for MFMA
float16x4 a_vec = *((float16x4 *)(lds_a_ptr + offset));
float16x4 b_vec = *((float16x4 *)(lds_b_ptr + offset));

// Or using ds_read instructions for gfx950
// ds_read_tr16_b64, ds_read_tr8_b64 (see Section 10)
```

---

## 8. Wavefront Execution Model

### 8.1 Wavefront vs Warp

| Property | CUDA Warp | AMD Wavefront |
|---|---|---|
| Size | 32 threads | 64 threads |
| Execution model | SIMT | SIMT |
| Divergence | Branch divergence | Branch divergence |
| Shuffle | `__shfl_sync` | `__shfl` |

### 8.2 Implications for TileLang

The 64-thread wavefront size affects several aspects:

1. **Thread allocation:** `block_size / 64` wavefronts per block (vs `block_size / 32` warps)
2. **MFMA operation:** Each MFMA instruction spans all 64 threads in a wavefront
3. **Reduction:** Wavefront reductions use `__shfl` across 64 lanes
4. **Fragment size:** Register fragments are distributed across 64 threads

### 8.3 Thread Mapping in MFMA

For a 16x16 MFMA instruction with fp16 input:

```
Each of 64 threads holds:
  - A matrix: 4 fp16 values (from 16x16 tile, 16*16/64 = 4)
  - B matrix: 4 fp16 values
  - C matrix: 4 fp32 accumulator values
```

The thread-to-element mapping follows the layout functions in `mfma_layout.py`.

---

## 9. CDNA Architecture Specifics

### 9.1 CDNA Detection

```cpp
bool TargetIsCDNA(Target target) {
    if (!TargetIsRocm(target)) return false;
    if (target->attrs.count("mcpu")) {
        std::string mcpu = Downcast<tvm::ffi::String>(target->attrs.at("mcpu"));
        return mcpu.find("gfx9") == 0;  // gfx9xx = CDNA
    }
    return false;
}
```

### 9.2 CDNA Generations

| Generation | GPU | Target | MFMA | Features |
|---|---|---|---|---|
| CDNA 1 | MI100 | gfx908 | fp16, bf16, int8 | First CDNA |
| CDNA 2 | MI250X | gfx90a | fp16, bf16, int8, fp8 (limited) | MCM design |
| CDNA 3 | MI300X | gfx942 | fp16, bf16, int8, fp8 | APU, 192GB HBM3 |
| CDNA 3+ | MI350 | gfx950 | fp16, bf16, int8, fp8 | Enhanced LDS |

### 9.3 CDNA MFMA Instruction Summary

| Shape | fp16 | bf16 | int8 | fp8 |
|---|---|---|---|---|
| 16x16x16 | Yes | Yes | - | - |
| 16x16x32 | - | - | Yes | Yes |
| 32x32x4 | Yes | Yes | - | - |
| 32x32x8 | Yes | Yes | - | - |
| 32x32x16 | - | - | Yes | Yes |

---

## 10. MI300X Optimizations

### 10.1 Architecture Overview

The MI300X (gfx942) is a CDNA 3 APU with:

- 304 CUs (Compute Units)
- 192 GB HBM3 memory
- 5.3 TB/s memory bandwidth
- Matrix Core (MFMA) support for fp16, bf16, int8, fp8

### 10.2 FP8 Support

MI300X supports FP8 (E4M3 and E5M2) data types in MFMA:

```hip
#if defined(HIP_FP8_ENABLED)
template <> struct MfmaTraits<fp8_e4_t> {
    static TL_DEVICE void mfma_op(const fp8_e4_t *b, const fp8_e4_t *a, AccType *c) {
        int64_t a_val = *reinterpret_cast<const int64_t *>(a);
        int64_t b_val = *reinterpret_cast<const int64_t *>(b);
        *c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b_val, a_val, *c, 0, 0, 0);
    }
};
#endif
```

### 10.3 Target Configuration

```python
import tilelang as tl

# MI300X target
target = tl.target.Target("hip", mcpu="gfx942")

# This enables:
# - MFMA instructions
# - FP8 data types
# - 64-thread wavefront
# - Async copy support
```

### 10.4 Async Copy Check

```cpp
bool TargetHasAsyncCopy(Target target) {
    if (TargetIsCDNA(target)) {
        std::string mcpu = Downcast<tvm::ffi::String>(target->attrs.at("mcpu"));
        if (mcpu.rfind("gfx9", 0) == 0) {
            int gfx_version = std::stoi(mcpu.substr(3, 2));
            return gfx_version >= 94;  // gfx940+ supports async copy
        }
        return false;
    }
    return false;
}
```

---

## 11. gfx950 LDS Operations

### 11.1 Overview

The gfx950 target (CDNA 3+) introduces enhanced LDS operations for improved
shared memory access patterns, particularly transposed reads.

### 11.2 Detection

```cpp
bool TargetIsGfx950(Target target) {
    if (!TargetIsRocm(target)) return false;
    if (target->attrs.count("mcpu")) {
        std::string mcpu = Downcast<tvm::ffi::String>(target->attrs.at("mcpu"));
        return mcpu.find("gfx950") != std::string::npos;
    }
    return false;
}
```

### 11.3 Transposed LDS Read Instructions

gfx950 introduces hardware-level transposed LDS read operations:

#### ds_read_tr16_b64

Transposed 16-bit read from LDS. Reads 64 bytes (32 x 16-bit elements) with
automatic transposition, avoiding the need for software transpose.

```hip
// Load and transpose in one instruction
// Equivalent to loading a column and getting it as a row
__builtin_amdgcn_ds_read_tr16_b64(base_ptr, offset);
```

#### ds_read_tr8_b64

Transposed 8-bit read from LDS. Reads 64 bytes (64 x 8-bit elements) with
automatic transposition.

```hip
// Load and transpose int8 elements
__builtin_amdgcn_ds_read_tr8_b64(base_ptr, offset);
```

### 11.4 Benefits for GEMM

These transposed LDS operations are particularly beneficial for:

1. **Transposed B matrix loading:** When matrix B is stored in column-major
   but needs to be accessed row-major for MFMA
2. **Eliminating shared memory padding:** The hardware transpose avoids the
   need for software-managed transposition in shared memory
3. **Reducing register pressure:** Fewer intermediate registers needed for
   transpose operations

### 11.5 Usage in TileLang

```python
from tilelang.utils.target import target_is_gfx950

# In GEMM emitter:
if target_is_gfx950(target):
    # Use transposed LDS reads for B matrix
    # ds_read_tr16_b64 for fp16/bf16
    # ds_read_tr8_b64 for int8
```

---

## 12. Composable Kernel Integration

### 12.1 Overview

TileLang integrates with AMD's Composable Kernel (CK) library for certain
optimized operations. The CK library is included as a third-party dependency
in `3rdparty/composable_kernel/`.

### 12.2 Integration Points

CK is used for:

1. **Reference implementations:** Validating TileLang-generated kernels
2. **Optimized reduction patterns:** Leveraging CK's reduction algorithms
3. **Layout transformations:** Using CK's tensor coordinate systems

### 12.3 CK vs TileLang Native

| Feature | CK | TileLang Native |
|---|---|---|
| Programming model | C++ templates | Python DSL -> TIR |
| Flexibility | Template specialization | Programmatic |
| Performance | Highly optimized | Compiler-optimized |
| Ease of use | Complex | Simple API |

---

## 13. Target String Configuration

### 13.1 Target Creation

```python
import tilelang as tl

# Basic ROCm/HIP target
target = tl.target.Target("hip")

# With specific GPU architecture
target = tl.target.Target("hip", mcpu="gfx90a")    # MI250X
target = tl.target.Target("hip", mcpu="gfx942")    # MI300X
target = tl.target.Target("hip", mcpu="gfx950")    # gfx950

# Alternative target string
target = tl.target.Target("rocm")
```

### 13.2 Target Key

The ROCm backend matches targets with the "hip" device type:

```cpp
bool TargetIsRocm(Target target) {
    return target->GetTargetDeviceType() == kDLROCM;
}
```

### 13.3 Architecture Detection

```python
# From tilelang/utils/target.py
from tilelang import _ffi_api

target_is_hip(target)      # True for ROCm/HIP targets
target_is_cdna(target)     # True for CDNA (gfx9xx)
target_is_rdna(target)     # True for RDNA (gfx11xx, gfx12xx)
target_is_gfx950(target)   # True specifically for gfx950
```

### 13.4 Warp Size Configuration

```cpp
int TargetGetWarpSize(Target target) {
    int res = 32;
    if (TargetIsCDNA(target))
        res = 64;  // CDNA uses 64-thread wavefronts
    return res;
}
```

Note: RDNA uses 32 threads per wavefront (same as CUDA warps).

---

## 14. Building for AMD GPUs

### 14.1 Prerequisites

- ROCm 5.7+ (for gfx90a support)
- ROCm 6.0+ (for gfx942/MI300X support)
- HIP compiler (hipcc or hiprtc)
- AMD Composable Kernel (optional)

### 14.2 Build Configuration

```python
# In setup.py or CMakeLists.txt
# Set the target architecture
export HCC_AMDGPU_TARGET=gfx942

# Or specify via environment
export ROCM_PATH=/opt/rocm
```

### 14.3 HIPRTC Compilation

```python
# From tilelang/contrib/hiprtc.py
def compile_hip(code, target, arch):
    # Compile HIP code using HIPRTC
    # Similar to NVRTC but for HIP
```

### 14.4 hipcc Compilation

```python
# From tilelang/contrib/hipcc.py
def compile_hip(code, target, arch):
    # Compile using hipcc
    # For ahead-of-time compilation
```

### 14.5 Compilation Flags

```bash
# Common HIP compilation flags
-DSUPPORTED_GFX_ARCH=gfx942
-DHIP_FP8_ENABLED
--offload-arch=gfx942
-mllvm -amdgpu-early-inline-all
-mllvm -amdgpu-function-calls=false
```

---

## 15. RDNA Architecture Specifics

### 15.1 RDNA Detection

```cpp
bool TargetIsRDNA(Target target) {
    if (!TargetIsRocm(target)) return false;
    if (target->attrs.count("mcpu")) {
        std::string mcpu = Downcast<tvm::ffi::String>(target->attrs.at("mcpu"));
        return mcpu.find("gfx11") == 0 || mcpu.find("gfx12") == 0;
    }
    return false;
}
```

### 15.2 RDNA Generations

| Generation | GPU | Target | WMMA | Wavefront |
|---|---|---|---|---|
| RDNA 3 | RX 7900 XTX | gfx1100 | Yes | 32 |
| RDNA 4 | (Future) | gfx1200 | Yes | 32 |

### 15.3 RDNA Generation Number

```cpp
int TargetGetRDNAGeneration(Target target) {
    if (!TargetIsRDNA(target)) return 0;
    std::string mcpu = Downcast<tvm::ffi::String>(target->attrs.at("mcpu"));
    if (mcpu.rfind("gfx11", 0) == 0) return 11;
    if (mcpu.rfind("gfx12", 0) == 0) return 12;
    return 0;
}
```

---

## 16. HIP Intrinsics and Stubs

### 16.1 HIP Runtime Stubs

The ROCm backend includes stub definitions for HIP runtime functions:

Located in `src/backend/rocm/codegen/stubs/`:
- `hip.cc` / `hip.h` - HIP runtime API stubs
- `hiprtc.cc` - HIPRTC API stubs
- `vendor/hip_runtime.h` - HIP runtime header stubs

These stubs enable the TileLang compiler to generate HIP code without
requiring the full HIP SDK at compile time.

### 16.2 HIP-Specific Intrinsics

From `src/backend/rocm/codegen/intrin_rule_hip.cc`:

The HIP backend registers intrinsic rules that map TIR operations to HIP-specific
implementations:

- Math operations (sin, cos, sqrt, etc.) -> HIP math intrinsics
- Type conversions -> HIP type cast functions
- Warp-level operations -> `__shfl`, `__ballot`
- Atomic operations -> HIP atomics

### 16.3 HIP-Specific Codegen

From `src/backend/rocm/codegen/rt_mod_hip.cc`:

The HIP runtime module builder creates HIP kernels:

```cpp
ffi::Module BuildTileLangHIP(IRModule mod, Target target) {
    // Build HIP code from TIR
    // Compile using HIPRTC or hipcc
    // Return runtime module
}
```

---

## 17. HIP Templates (tl_templates/hip/)

| File | Purpose |
|---|---|
| `common.h` | Common HIP utilities, type traits |
| `gemm.h` | MFMA GEMM implementation |
| `copy.h` | Copy operations |
| `atomic.h` | Atomic operations |
| `reduce.h` | Wavefront/block reduction |
| `barrier.h` | Barrier operations |
| `ldsm.h` | LDS memory operations |
| `debug.h` | Debug utilities |
| `threadblock_swizzle.h` | Thread block mapping |
| `hip_fp8.h` | FP8 data type support |

---

## 18. Complete GEMM Example for ROCm

### 18.1 TileLang GEMM Kernel

```python
import tilelang as tl
from tilelang import language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### 18.2 Compiling for MI300X

```python
import tilelang as tl

# Create MI300X target
target = tl.target.Target("hip", mcpu="gfx942")

# Compile the kernel
kernel = tl.compile(matmul(1024, 1024, 1024, 128, 128, 32), target=target)

# Run the kernel
import torch
A = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")  # Uses HIP
B = torch.randn(1024, 1024, dtype=torch.float16, device="cuda")
C = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")

kernel(A, B, C)
```

---

## 19. Known Limitations

### 19.1 General Limitations

1. **No TMEM support:** AMD GPUs do not have an equivalent to NVIDIA's Tensor Memory
2. **No cluster support:** No equivalent to NVIDIA's thread block clusters
3. **No WGMMA equivalent:** No warp-group level matrix multiply instruction
4. **Limited TMA support:** No hardware tensor memory access; async copy uses buffer_load

### 19.2 FP8 Limitations

- FP8 MFMA is only available on gfx940+ (CDNA 3+)
- Requires `HIP_FP8_ENABLED` compile flag
- FP8 E4M3 and E5M2 only (no FP8 E8M0)

### 19.3 Sparse GEMM

- ROCm MFMA does not have native 2:4 sparse support like NVIDIA
- Sparse GEMM would require software-managed sparsity patterns

### 19.4 Warp Specialization

- AMD does not have hardware-level warp specialization like NVIDIA Hopper/Blackwell
- Producer-consumer patterns must be implemented in software

### 19.5 Pipeline Support

- Software pipelining is supported but less efficient than NVIDIA's cp.async pipeline
- No hardware-level pipeline management

### 19.6 Shared Memory

- LDS has a different bank structure than NVIDIA shared memory
- Some swizzle patterns may not translate directly

### 19.7 Compilation

- HIPRTC may have different compilation errors than NVRTC
- Some CUDA-specific headers need HIP equivalents
- FP8 type support requires separate HIP headers (`hip_fp8.h`)

### 19.8 Performance Considerations

- Wavefront size (64) means different thread utilization patterns
- MFMA instructions have different throughput characteristics compared to NVIDIA MMA
- Memory bandwidth and cache hierarchy differ significantly
- LDS throughput may differ from NVIDIA shared memory
