# TileLang CUDA Backend Reference

## 1. Overview

The CUDA backend is the primary compute backend for TileLang, providing high-performance
code generation targeting NVIDIA GPUs from Volta (SM70) through Blackwell (SM100+). It
leverages Tensor Core instructions at every architecture level, generates inline PTX
assembly for low-level operations, and integrates with CUTLASS/CuTe for advanced Hopper
and Blackwell features.

### Architecture Diagram

```
TileLang IR (PrimFunc)
        |
        v
+-------------------+
|  Backend Selection |  ---> Target = "cuda"
+-------------------+
        |
        v
+-------------------+
|  Transform Passes |  (lower_tile_op, lower_hopper_intrin, lower_ptx_async_copy, ...)
+-------------------+
        |
        v
+-------------------+
| CodeGenTileLangCUDA |  (codegen_cuda.cc)
+-------------------+
        |
        v
+-------------------+
| PTX/NVRTC Compile  |  (ptx.cc, nvrtc.py)
+-------------------+
        |
        v
CUDA Kernel (.cu)
```

---

## 2. Code Generation: codegen_cuda.cc / codegen_cuda.h

The CUDA code generator `CodeGenTileLangCUDA` is a subclass of `CodeGenC` that produces
valid CUDA C++ code from TileLang's TIR (Tensor Intermediate Representation).

### Class Structure

Defined in `src/backend/cuda/codegen/codegen_cuda.h`:

```cpp
class CodeGenTileLangCUDA final : public CodeGenC {
public:
  CodeGenTileLangCUDA();
  std::string Finish();
  void PrintFuncPrefix(std::ostream &os) final;
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintType(DataType t, std::ostream &os) final;
  void BindThreadIndex(const IterVar &iv) final;
  // ... many more overrides
};
```

### Key Code Generation Features

#### 2.1 Data Type Handling

The CUDA codegen supports a wide range of data types and tracks which headers are needed
through boolean flags:

```cpp
bool enable_fp16_{false};
bool enable_bf16_{false};
bool enable_fp8_{false};
bool enable_fp6_{false};
bool enable_fp4_{false};
bool enable_int8_{false};
bool enable_sparse_gemm_{false};
```

When a particular data type is encountered during code generation, the corresponding
flag is set, and the appropriate headers are included in the final output:

- `enable_fp16_` triggers `#include <cuda_fp16.h>`
- `enable_bf16_` triggers `#include <cuda_bf16.h>`
- `enable_fp8_` triggers `#include <cuda_fp8.h>` or the internal `cuda_fp8.h` template
- `enable_fp4_` triggers `#include "cuda_fp4.h"` from tl_templates

#### 2.2 MMA Instruction Header Management

The codegen tracks which Tensor Core instruction headers are needed:

```cpp
bool need_mma_h_{false};                     // Ampere MMA (m16n8k16 etc.)
bool need_mma_instruction_h_{false};         // Custom MMA instruction header
bool need_wgmma_instruction_h_{false};       // Hopper WGMMA instruction header
bool need_tcgen05mma_instruction_h_{false};  // Blackwell TCGEN05 header
bool need_mma_sm70_instruction_h_{false};    // Volta MMA (sm70) header
bool need_tcgen05_common_h_{false};          // TCGEN05 common utilities
```

These correspond to the template headers in `src/tl_templates/cuda/instruction/`:
- `mma.h` - Ampere MMA PTX wrappers
- `wgmma.h` - Hopper WGMMA PTX wrappers
- `tcgen05mma.h` - Blackwell TCGEN05 MMA PTX wrappers
- `mma_sm70.h` - Volta MMA PTX wrappers

#### 2.3 Function Generation

`AddFunction` handles `__grid_constant__` parameters for kernels targeting SM70+:

```cpp
void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
void PrintFunctionSignature(const ffi::String &function_name,
                            const PrimFunc &func, std::ostream &os);
```

Kernel functions are generated with CUDA-specific attributes:

```cuda
extern "C" __global__ void kernel_name(
    __grid_constant__ half* __restrict__ A,
    __grid_constant__ half* __restrict__ B,
    ...)
```

#### 2.4 Barrier Management

The CUDA backend manages both legacy barriers and mbarrier objects:

```cpp
const std::string barrier_name_ = "barrier";
const std::string mbarrier_name_ = "mbarrier";
const std::string mbarrier_dtype_ = "Barrier";
const int barrier_alignment_bytes_ = 16;
```

- `barrier` is used for `__syncthreads()` style synchronization
- `mbarrier` is used for async copy completion tracking on SM80+
- Alignment is set to 16 bytes for async bulk copy requirements

#### 2.5 Thread Index Binding

```cpp
void BindThreadIndex(const IterVar &iv) final;
```

Maps TileLang thread variables to CUDA built-in variables:
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
- `blockDim.x`, `blockDim.y`, `blockDim.z`

#### 2.6 Storage Scope Handling

```cpp
void PrintStorageScope(const std::string &scope, std::ostream &os) final;
```

Maps TileLang storage scopes to CUDA address spaces:

| TileLang Scope | CUDA Address Space |
|---|---|
| `local` | (default, registers) |
| `shared` | `__shared__` |
| `shared.dyn` | `__shared__` (dynamic) |
| `shared.tmem` | Tensor Memory (SM100+) |
| `global` | (default, global memory) |

---

## 3. GEMM Implementations

The CUDA backend provides four families of Tensor Core GEMM instructions, each targeting
different GPU architectures:

### 3.1 MMA (Ampere) - cuda.mma

**Target:** SM80, SM86, SM89 (Ampere, Ada Lovelace)

**Instruction sizes:** m16n8k16, m16n8k32, m16n8k8, m8n8k4

**Supported data types:**

| A dtype | B dtype | C dtype | Instruction |
|---|---|---|---|
| float16 | float16 | float16 | mma.sync.aligned.m16n8k16 |
| float16 | float16 | float32 | mma.sync.aligned.m16n8k16 |
| bfloat16 | bfloat16 | float32 | mma.sync.aligned.m16n8k16 |
| int8 | int8/uint8 | int32 | mma.sync.aligned.m16n8k32 |
| int8 | uint8 | int32 | mma.sync.aligned.m16n8k16 |
| float8_e4m3 | float8_e4m3 | float32 | mma.sync.aligned.m16n8k32 |
| float8_e5m2 | float8_e5m2 | float32 | mma.sync.aligned.m16n8k32 |
| float4_e2m1fn | float4_e2m1fn | float16 | mma.sync.aligned.m16n8k64 |
| int4 | int4 | int32 | mma.sync.aligned.m16n8k64 |

**Layout:** Each warp holds a 32xN fragment in registers. The ldmatrix instruction
is used to load from shared memory into the register layout expected by MMA.

**C++ Template:** `src/tl_templates/cuda/gemm_sm80.h` and `src/tl_templates/cuda/gemm_mma.h`

**Python Intrinsic:** `tilelang/intrinsics/mma_macro_generator.py` (`TensorCoreIntrinEmitter`)

**Key parameters:**
- `M_DIM = 16` (fixed)
- `n_dim = 16` (default, can be 8 for some instructions)
- `WARP_SIZE = 32`
- `k_dim` depends on data type (8, 16, 32, 64 bits of K)

**Register layout:**

MMA stores results in a specific register layout where each of the 32 threads in a warp
holds a portion of the output matrix. The layout for m16n8k16 is:

```
Thread layout (32 threads, 4 elements each for fp16):
  Thread i holds elements at positions determined by shared_16x*_to_mma_32x*_layout
```

**Warp partition for MMA:**

```python
# From gemm.cc ComputeDefaultWarpPartition
# For MMA (non-WGMMA, non-TCGEN05):
# k_n_per_warp = 8 (Ampere/Turing) or 16 (Volta)
# Uses GemmWarpPolicy: FullRow, FullCol, Square
```

### 3.2 MMA_sm70 (Volta) - cuda.mma_sm70

**Target:** SM70 (Volta - V100)

**Instruction:** mma.sync.m8n8k4 (HMMA.884)

**Supported data types:**

| A dtype | B dtype | C dtype |
|---|---|---|
| float16 | float16 | float16 |
| float16 | float16 | float32 |

**C++ Template:** `src/tl_templates/cuda/gemm_sm70.h`

**Python Intrinsic:** `tilelang/intrinsics/mma_sm70_macro_generator.py`

**Layout:** Uses special Volta-specific layouts:
- `shared_16x4_to_mma_a_32x4_layout` (for matrix A)
- `shared_4x16_to_mma_b_32x4_layout` (for matrix B, row-major)
- `shared_16x4_to_mma_b_32x4_layout_trans` (for matrix B, transposed)

The Volta Tensor Core has a different register layout compared to Ampere+:
- `HALF_WARP_SIZE = 16` (threads are split into two halves)
- Different ldmatrix patterns

### 3.3 WGMMA (Hopper) - cuda.wgmma

**Target:** SM90 (Hopper - H100, H200)

**WGMMA instructions** operate at the warp-group level (4 warps = 128 threads) rather
than single warp level, enabling much larger matrix operations per instruction.

**Supported data types and shapes:**

| A dtype | B dtype | C dtype | Constraints |
|---|---|---|---|
| float16 | float16 | float16 | k % 16 == 0 |
| float16 | float16 | float32 | k % 16 == 0 |
| bfloat16 | bfloat16 | float32 | k % 16 == 0 |
| float8_e4m3 | float8_e4m3 | float16 | !transA, transB, k % 32 == 0 |
| float8_e4m3 | float8_e4m3 | float32 | !transA, transB, k % 32 == 0 |
| float32 | float32 | float32 | !transA, transB, k % 8 == 0 |
| int8 | int8/uint8 | int32 | !transA, transB, k % 32 == 0 |

**Key constraints for WGMMA eligibility** (from `AllowWgmma`):

```cpp
bool AllowWgmma(const GemmNode &op, int block_size, Target target) {
  int warp_size = TargetGetWarpSize(target);   // 32 for CUDA
  int num_warps = block_size / warp_size;
  return !disable_wgmma &&      // not explicitly disabled
         TargetIsHopper(target) && // SM90+
         op.m_ >= 64 &&           // M dimension at least 64
         num_warps % 4 == 0 &&    // warp-group sized
         CheckWgmma(op);          // data type constraints
}
```

**C++ Template:** `src/tl_templates/cuda/gemm_sm90.h` (CuTe-based)

**Python Intrinsic:** `tilelang/intrinsics/wgmma_macro_generator.py`

**WGMMA warp partition:**

```cpp
// From ComputeWgmmaWarpPartition
constexpr int kMPerWarp = 16;
constexpr int kNPerWarp = 8;
constexpr int kGroup = 4;  // WGMMA requires 4-warps per warp-group

// Default: 4 warps along M, remaining along N
m_warp = kGroup;  // = 4
n_warp = num_warps / m_warp;
```

**Swizzle modes for WGMMA:**

```python
class SwizzleMode(IntEnum):
    NONE = 0
    SWIZZLE_128B = 1
    SWIZZLE_64B = 2
    SWIZZLE_32B = 3
```

The WGMMA emitter selects the appropriate swizzle mode based on the data layout
to match the hardware requirements for shared memory access patterns.

**WGMMA SS (Shared-Shared) operation:**

The Hopper backend uses CUTLASS CuTe library for WGMMA:

```cpp
// From gemm_sm90.h
template <int wg_wait = 0>
static CUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
  auto tiled_mma = make_tiled_mma(
      GMMA::ss_op_selector<A_type, B_type, C_type, Shape<...>,
                            GmmaMajorA, GmmaMajorB>(),
      Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>{});

  warpgroup_fence_operand(acc);
  warpgroup_arrive();
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), acc);
  }
  warpgroup_commit_batch();
  warpgroup_wait<wg_wait>();
  warpgroup_fence_operand(acc);
}
```

### 3.4 TCGEN05 (Blackwell) - cuda.tcgen05

**Target:** SM100+ (Blackwell - B200, B100)

**TCGEN05MMA instructions** are the next-generation matrix multiply instructions for
Blackwell GPUs. They support block-scaled GEMM and operate with Tensor Memory (TMEM).

**Supported variants:**

1. **SS (Shared-Shared):** Both A and B from shared memory, result to TMEM
2. **TS (TMEM-Shared):** A from TMEM, B from shared memory

**Key features:**

| Feature | Description |
|---|---|
| Block-scaled GEMM | Supports per-block scaling factors |
| TMEM output | Results written to Tensor Memory |
| 2-CTA mode | Optional two-CTA cluster operation |
| Warp specialization | Dedicated warps for MMA vs memory |

**C++ Template:** `src/tl_templates/cuda/gemm_sm100.h`

**Python Intrinsic:** `tilelang/intrinsics/tcgen05_macro_generator.py`

**TCGEN05 constraints** (from `AllowTcgen5Mma`):

```cpp
bool AllowTcgen5Mma(const GemmNode &op, Target target) {
  bool scope_ok = (IsSharedBuffer(op.a_) || op.a_.scope() == "shared.tmem") &&
                  IsSharedBuffer(op.b_) && op.c_.scope() == "shared.tmem";
  if (!TargetIsSm100(target) || !scope_ok)
    return false;
  return GetTCGEN5MMAMeta(op.m_, op.n_, op.k_, ab_dtype, op.c_->dtype).first;
}
```

The C buffer must have `shared.tmem` scope for TCGEN05 operations.

**TCGEN05 instruction descriptor:**

```python
# From op/tcgen5_meta.h
desc = GetTCGEN5InstrDesc(atom_m, atom_n, atom_k, ab_dtype, c_dtype,
                           a_is_k_major, b_is_k_major, scale_in_a, scale_in_b)
```

For block-scaled GEMM:

```python
desc = GetTCGEN5BlockScaledInstrDesc(atom_m, atom_n, ab_dtype,
                                      a_is_k_major, b_is_k_major,
                                      scale_in_a, scale_in_b,
                                      a_sf_id, b_sf_id)
```

**Swizzle modes for TCGEN05:**

```python
class SwizzleMode(IntEnum):
    NONE = 0
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6
```

Note the different encoding compared to WGMMA (2/4/6 instead of 1/2/3).

---

## 4. Backend Selection: register_gemm_impl and resolve_gemm_impl

### 4.1 Registration System

The GEMM backend selection uses a registration pattern defined in
`tilelang/backend/gemm.py`:

```python
@dataclass(frozen=True)
class GemmImplEntry:
    name: str           # e.g. "cuda.mma"
    inst_name: str      # e.g. "cuda.wgmma"
    predicate: GemmTargetPredicate
    impl_class: type

_GEMM_IMPLS: list[GemmImplEntry] = []

def register_gemm_impl(name, inst_name, predicate, impl_class):
    entry = GemmImplEntry(name, inst_name, predicate, impl_class)
    for idx, registered in enumerate(_GEMM_IMPLS):
        if registered.name == name:
            _GEMM_IMPLS[idx] = entry
            return
    _GEMM_IMPLS.append(entry)
```

### 4.2 CUDA GEMM Registration

In `tilelang/backend/cuda/gemm.py`:

```python
register_gemm_impl("cuda.mma",      GEMM_INST_MMA,    _match_mma,      GemmMMA)
register_gemm_impl("cuda.mma_sm70", GEMM_INST_MMA,    _match_mma_sm70, GemmMMASm70)
register_gemm_impl("cuda.wgmma",    GEMM_INST_WGMMA,  _match_wgmma,    GemmWGMMA)
register_gemm_impl("cuda.tcgen05",  GEMM_INST_TCGEN05, _match_tcgen05,  GemmTCGEN5)
```

### 4.3 C++ Side Instruction Selection

In `src/backend/cuda/op/gemm.cc`, the `Gemm::SelectInst` function determines which
hardware instruction to use:

```cpp
static String SelectInst(const GemmNode &op, int block_size, Target target) {
  if (op.isWgmma_) {         // explicitly requested WGMMA
    return kCudaWGMMA;
  }
  if (op.isTcgen05_) {       // explicitly requested TCGEN05
    return kCudaTCGEN05;
  }
  // Auto-select best available
  if (AllowTcgen5Mma(op, target)) return kCudaTCGEN05;
  if (AllowWgmma(op, block_size, target)) return kCudaWGMMA;
  return kCudaMMA;           // fallback to Ampere MMA
}
```

### 4.4 Resolution

```python
def resolve_gemm_impl(gemm_inst: str, target: Target) -> type:
    matches = [entry for entry in _GEMM_IMPLS
               if entry.inst_name == gemm_inst and entry.predicate(target)]
    if not matches:
        raise ValueError(f"No GEMM implementation for {gemm_inst} and {target}")
    if len(matches) > 1:
        raise ValueError(f"Multiple implementations for {gemm_inst}")
    return matches[0].impl_class
```

---

## 5. PTX Code Generation: ptx.cc / ptx.h

The PTX code generation module produces inline PTX assembly strings for low-level GPU
operations.

### 5.1 Data Types

The `ptx::DataType` enum covers all PTX-representable types:

```cpp
enum class DataType : int {
  kInt4, kUInt4,
  kInt8, kUInt8,
  kInt16, kUInt16,
  kInt32, kUInt32,
  kInt64, kUInt64,
  kFloat8_e4m3, kFloat8_e5m2,
  kFloat16, kBFloat16, kFloat16x2,
  kFloat32, kTensorFloat32, kFloat64,
  kBit1, kBit8, kBit16, kBit32, kBit64,
  kFloat6_e2m3fn, kFloat6_e3m2fn, kFloat4_e2m1fn
};
```

### 5.2 MMA Assembly Generation

```cpp
std::string PrintMMAAssembly(
    const std::string &shape,      // "m16n8k16" etc.
    const std::string &A_layout,   // "row" or "col"
    const std::string &B_layout,   // "row" or "col"
    const std::string &A_dtype,
    const std::string &B_dtype,
    const std::string &C_dtype,
    const std::string &a_ptr, const std::string &a_offset,
    const std::string &b_ptr, const std::string &b_offset,
    const std::string &c_ptr, const std::string &c_offset,
    const std::string &metadata, const std::string &metadata_offset,
    const std::string &sparsity_selector,
    const std::string &bit_op,     // "xor" or "and"
    bool sparse, bool saturate);
```

This generates PTX inline assembly of the form:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
    : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
    : "r"(a0), "r"(a1), "r"(b0),
      "f"(c0_in), "f"(c1_in), "f"(c2_in), "f"(c3_in)
);
```

### 5.3 WGMMA Assembly Generation

```cpp
std::string PrintWGMMAAssembly(
    const std::string &shape,
    const bool &a_is_k_major, const bool &b_is_k_major,
    const std::string &A_dtype, const std::string &B_dtype,
    const std::string &C_dtype,
    const std::string &a_desc, const std::string &A_offset,
    const std::string &b_desc, const std::string &B_offset,
    const std::string &c_ptr, const std::string &c_offset,
    const bool &scale_out, const bool &scale_in_a, const bool &scale_in_b,
    const bool &a_is_shared,
    const std::string &metadata, const std::string &metadata_offset,
    const std::string &sparsity_selector, bool sparse);
```

WGMMA uses tensor descriptor-based access rather than raw pointers:

```cuda
asm volatile(
    "wgmma.mma_async.sync.aligned.m64nNkK.row.col.f32.e4m3.e4m3.f32 "
    "{%0, ...}, %1, %2, %3, %4;"
    : ... : "l"(desc_a), "l"(desc_b), "r"(scale_a), "r"(scale_b)
);
```

### 5.4 Load Matrix (ldmatrix) Assembly

```cpp
std::string PrintLoadMatrixAssembly(
    bool trans, int num, const std::string &type,
    const std::string &local_ptr, const std::string &local_elem_offset,
    const std::string &smem_ptr, const std::string &smem_elem_offset);
```

Generates:

```cuda
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
    "{%0, %1, %2, %3}, [%4 + %5];"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(smem_ptr), "r"(offset)
);
```

### 5.5 Async Copy (cp.async) Assembly

#### Basic cp.async

```cpp
std::string PrintCpAsyncAssembly(
    const std::string &shared_ptr, const std::string &shared_elem_offset,
    const std::string &global_ptr, const std::string &global_elem_offset,
    const std::string &bytes);  // 4, 8, or 16
```

Generates:

```cuda
asm volatile(
    "cp.async.cg.shared.global [%0 + %1], [%2 + %3], %4;"
    :: "r"(smem_ptr), "r"(smem_offset),
       "r"(gmem_ptr), "r"(gmem_offset), "n"(bytes)
);
```

#### Predicated cp.async

```cpp
std::string PrintPredicatedCpAsyncAssembly(
    const std::string &shared_ptr, const std::string &shared_elem_offset,
    const std::string &global_ptr, const std::string &global_elem_offset,
    const std::string &bytes, const std::string &predicate_value);
```

Generates:

```cuda
@p cp.async.cg.shared.global [smem + off], [gmem + off], bytes;
```

#### Bulk Async Copy (cp.async.bulk)

```cpp
std::string PrintCpAsyncBulkAsm(
    const std::string &shared_ptr, const std::string &shared_elem_offset,
    const std::string &global_ptr, const std::string &global_elem_offset,
    const std::string &bytes, const std::string &barrier);
```

Generates TMA-like bulk copy:

```cuda
cp.async.bulk.shared.global.mbarrier::complete_tx::bytes
    [smem + off], [gmem + off], bytes, [barrier];
```

### 5.6 Barrier Operations

```cpp
std::string PrintInitBarrierThreadCountAsm(
    const std::string &barrier, const std::string &thread_count);

std::string PrintArriveBarrierAsm(const std::string &barrier);

std::string PrintArriveBarrierExpectTxAsm(
    const std::string &barrier, const std::string &byte_count);

std::string PrintWaitBarrierAsm(const std::string &barrier);
```

These generate the mbarrier PTX operations:

```cuda
// Init
mbarrier.init.shared.b64 [barrier], thread_count;

// Arrive
mbarrier.arrive.shared.b64 _, [barrier];

// Arrive with expect_tx
mbarrier.arrive.expect_tx.shared.b64 _, [barrier], byte_count;

// Wait
mbarrier.try_wait.parity.shared.b64 _, [barrier], parity;
```

### 5.7 Register Types

```cpp
std::string GetMMARegisterType(const ptx::DataType &dtype);
```

Maps MMA data types to their C++ register representations:

| PTX dtype | Register type |
|---|---|
| float16 | `uint32_t` (packed __half2) |
| bfloat16 | `uint32_t` (packed __nv_bfloat162) |
| float32 | `float` |
| int32 | `int32_t` |
| float8_e4m3 | `uint32_t` (packed) |
| float8_e5m2 | `uint32_t` (packed) |

---

## 6. CuTeDSL Integration: codegen_cutedsl.cc / codegen_cutedsl.h

### 6.1 Overview

The CuTeDSL code generator (`CodeGenTileLangCuTeDSL`) extends `CodeGenTileLangPY`
to produce CuTe DSL C++ code that can leverage CUTLASS's CuTe library directly.
This enables advanced features like TMA descriptors and WGMMA that require CuTe
abstractions.

### 6.2 Class Hierarchy

```
CodeGenC
  -> CodeGenTileLangPY
    -> CodeGenTileLangCuTeDSL
```

### 6.3 Key Differences from CUDA CodeGen

- Produces Python-like CuTe DSL syntax instead of raw CUDA C++
- Handles loop-break via guard variables (CuTeDSL doesn't support early exit)
- Supports fastmath configuration via PassContext
- Different buffer reference handling for CuTe tensors
- Manages mbarrier objects in shared memory

### 6.4 Loop Break Handling

Since CuTeDSL doesn't support the `break` keyword, the codegen transforms
break statements into guard variables:

```cpp
bool in_break_loop_ = false;
int loop_break_counter_ = 0;
int current_break_id_ = -1;
bool break_emitted_in_seq_ = false;
```

A loop containing `break` is transformed to:

```python
# Original:
for i in range(N):
    if condition:
        break
    body()

# Transformed:
break_flag_0 = False
for i in range(N):
    if break_flag_0:
        break_flag_0 = True  # propagate
    else:
        if condition:
            break_flag_0 = True
        else:
            body()
```

---

## 7. Tensor Core Utilization

### 7.1 MMA Instructions (SM75+, SM80+)

The Ampere MMA instructions (`mma.sync`) are the primary Tensor Core interface:

**Operation:** D = A * B + C (at warp level)

**Fragment layout per warp (m16n8k16, fp16):**
- A matrix: 32 threads, each holding 4 fp16 values (via ldmatrix)
- B matrix: 32 threads, each holding 2 fp16 values
- C/D matrix: 32 threads, each holding 4 fp16/float32 values

**Loading into fragments:**

```cuda
// Load matrix A (16xK tile) into registers
ldmatrix.sync.aligned.m8n8.x4.shared.b16
    {%0, %1, %2, %3}, [%4];

// Execute MMA
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%0,...}, {%4,...}, {%8,...}, {%12,...};
```

### 7.2 WGMMA Instructions (SM90)

WGMMA operates at the warp-group level (128 threads = 4 warps):

**Key differences from MMA:**
- Source A and B from shared memory (no explicit register loads needed)
- Uses shared memory descriptors
- Requires warp-group arrive/commit protocol
- Larger M dimension (64+) per instruction

**Warp-group protocol:**

```cuda
warpgroup_fence_operand(accumulator);
warpgroup_arrive();
// WGMMA instructions issued here
warpgroup_commit_batch();
warpgroup_wait<0>();
warpgroup_fence_operand(accumulator);
```

---

## 8. Shared Memory Optimization

### 8.1 Swizzle Layouts

The CUDA backend provides multiple swizzle patterns to avoid shared memory bank conflicts:

#### Generic Swizzle (for TMA/Copy operations)

```python
# From tilelang/layout/swizzle.py
layout = make_swizzled_layout(buffer, k_major=True, allow_pad=True)
```

#### Volta Swizzle

```python
layout = make_volta_swizzled_layout(buffer, is_a=True, k_inner=True)
```

#### WGMMA Swizzle

```python
layout = make_wgmma_swizzled_layout(buffer, continuity=None, k_major=True)
```

#### TCGEN05 Swizzle

```python
layout = make_tcgen05mma_swizzled_layout(buffer, continuity=None, k_major=True)
```

### 8.2 Bank Conflict Avoidance

Three levels of bank swizzling are provided:

```python
make_full_bank_swizzled_layout(buffer)     # 128-byte swizzle
make_half_bank_swizzled_layout(buffer)     # 64-byte swizzle
make_quarter_bank_swizzled_layout(buffer)  # 32-byte swizzle
```

These create XOR-based swizzle patterns that remap shared memory addresses to
distribute accesses across all 32 memory banks.

### 8.3 Dynamic Shared Memory

TileLang uses `shared.dyn` scope for dynamically-allocated shared memory:

```python
# In TileLang kernel
smem = T.alloc_shared((M, K), dtype, scope="shared.dyn")
```

The CUDA codegen handles dynamic shared memory allocation:

```cuda
extern __shared__ uint8_t dyn_shared[];
// Cast to appropriate type
half* smem = reinterpret_cast<half*>(dyn_shared + offset);
```

---

## 9. Register Allocation Strategies

### 9.1 Fragment Buffers

Fragment buffers in TileLang map directly to CUDA registers:

```python
# Allocate register-level fragment
frag = T.alloc_fragment((M, N), dtype)
```

The CUDA codegen maps these to register arrays:

```cuda
half frag[M * N];  // or appropriate register-packed representation
```

### 9.2 Register Pressure Management

For Hopper WGMMA, register allocation is critical because WGMMA uses many registers
(accumulators are held in registers across the entire warp-group).

The transform pass `annotate_warp_group_reg_alloc.cc` can set register allocation
hints:

```cuda
__launch_bounds__(blockSize, minBlocksPerMultiprocessor)
```

### 9.3 Warp Group Register Allocation

On SM90, the `nve` (number of registers per warp-group) can be controlled:

```cuda
// SM90 specific: control register allocation for warp groups
asm volatile("setmaxnreg.inc.sync.u32 %0;" : : "n"(num_regs));
```

---

## 10. Thread Synchronization Lowering

### 10.1 Block-Level Sync

```python
# TileLang
T.barrier_sync()
```

Lowers to:

```cuda
__syncthreads();
```

### 10.2 Named Barrier Sync

```python
# TileLang (SM90+)
T.barrier(name="mbarrier", count=thread_count)
```

Lowers to mbarrier PTX operations:

```cuda
// Init
mbarrier.init.shared.b64 [mbarrier], thread_count;
// Arrive
mbarrier.arrive.shared.b64 _, [mbarrier];
// Wait
mbarrier.try_wait.parity.shared.b64 _, [mbarrier], parity;
```

### 10.3 Warp-Level Sync

```python
# TileLang
T.warp_sync(mask=0xFFFFFFFF)
```

Lowers to:

```cuda
__syncwarp(0xFFFFFFFF);
```

---

## 11. Async Copy Lowering (cp.async)

### 11.1 Transform Pass: lower_ptx_async_copy

The transform pass `src/transform/lower_ptx_async_copy.cc` converts TileLang async
copy operations into PTX cp.async instructions.

**Available on:** SM80+ (Ampere and later)

**Check:** `TargetHasAsyncCopy(target)` returns true for SM80+

### 11.2 Async Copy Pipeline

A typical async copy pipeline:

```python
# TileLang pseudo-code
for ko in range(K_tiles):
    # Init barrier
    T.barrier_init(mbarrier, num_threads)

    # Issue async copies
    T.copy_async(src_global, dst_shared, mbarrier)

    # Wait for completion
    T.barrier_wait(mbarrier)

    # Compute
    T.gemm(A_smem, B_smem, C_frag)
```

This lowers to:

```cuda
// Init mbarrier
mbarrier.init.shared.b64 [mbarrier], thread_count;

// Issue async copy with expect_tx
cp.async.cg.shared.global [smem + off], [gmem + off], 16;
mbarrier.arrive.expect_tx.shared.b64 _, [mbarrier], 16;

// Wait for all copies
mbarrier.try_wait.parity.shared.b64 _, [mbarrier], 0;

// Compute
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {c0,...}, {a0,...}, {b0,...}, {c0_in,...};
```

### 11.3 Copy Analysis

The `src/backend/cuda/op/copy_analysis.cc` analyzes copy operations to determine:

1. Whether async copy is beneficial
2. Optimal copy granularity (4, 8, or 16 bytes)
3. Whether TMA can be used (SM90+)
4. Predication requirements for boundary handling

---

## 12. TMA (Tensor Memory Access) Operations

### 12.1 Overview

TMA is available on SM90+ (Hopper) and provides hardware-accelerated tensor copies
between global memory and shared memory with automatic swizzling and bounds handling.

**Check:** `TargetHasBulkCopy(target)` returns true for SM90+

### 12.2 TMA Descriptor Creation

TMA operations use tensor descriptors that encode the tensor shape, stride, and
swizzle pattern:

```cuda
// Create TMA descriptor (handled by CuTe)
TensorMap tma_desc;
cuTensorMapEncodeTiled(&tma_desc, ...);
```

### 12.3 TMA Copy Operations

```cuda
// Bulk copy using TMA
cp.async.bulk.tensor.2d.shared.global.mbarrier::complete_tx::bytes
    [smem], [tma_desc + offset], bytes, [mbarrier];
```

### 12.4 TMA in TileLang

In TileLang, TMA is typically invoked through the copy operation with appropriate
buffer scopes and target configuration:

```python
# TMA copy is automatically selected when:
# 1. Target is SM90+
# 2. Source is global memory
# 3. Destination is shared memory
# 4. Bulk copy is available
T.copy(A_global, A_shared)
```

---

## 13. Cluster / Block Cluster Support

### 13.1 Overview

Thread block clusters allow multiple CTAs (Cooperative Thread Arrays) to cooperate
through distributed shared memory. Available on SM90+.

### 13.2 Cluster Planning

The transform pass `src/transform/cluster_planning.cc` handles cluster configuration:

```python
# TileLang cluster configuration
with T.cluster(cluster_dims=(2, 1, 1)):
    # Kernel code that uses cluster operations
    T.copy(src_shared_remote, dst_shared_local)
```

### 13.3 Cluster Synchronization

```cuda
// Cluster-level barrier
cluster_barrier_wait();
cluster_barrier_arrive();

// Distributed shared memory access
extern __cluster_shared__ uint8_t dist_smem[];
```

### 13.4 Cluster Dimensions

The codegen stores cluster dimensions:

```cpp
std::optional<std::tuple<int64_t, int64_t, int64_t>> cluster_dims;
```

This affects kernel launch configuration:

```cuda
dim3 grid_dim(grid_x / cluster_x, grid_y, grid_z);
dim3 block_dim(block_size);
void* kernel_params[] = {...};
cudaLaunchClusterKernel(kernel, grid_dim, block_dim,
                        cluster_dim, kernel_params, 0);
```

---

## 14. Warp Specialization

### 14.1 Overview

Warp specialization divides warps within a CTA into producer and consumer roles.
Available on SM90+ and heavily used in SM100+.

### 14.2 Producer-Consumer Warp Specialization

The transform pass `src/transform/producer_consumer_ws.cc` handles this:

```python
# TileLang warp specialization
with T.warp_specialize(
    producer_fn=lambda: ...,
    consumer_fn=lambda: ...,
    num_producer_warps=2,
    num_consumer_warps=6,
):
    ...
```

### 14.3 Hopper Warp Specialization Pattern

```cuda
// Producer warps (load data)
if (warp_id < num_producer_warps) {
    // Issue async copies
    cp.async.cg.shared.global [...], [...], 16;
    mbarrier.arrive.expect_tx.shared.b64 _, [barrier], 16;
}
// Consumer warps (compute)
else {
    // Wait for data
    mbarrier.try_wait.parity.shared.b64 _, [barrier], 0;
    // Compute
    wgmma.mma_async.sync.aligned.m64nNkK ...;
}
```

### 14.4 Blackwell Warp Specialization

On SM100+, TCGEN05 has built-in warp specialization where certain warps are
dedicated to TMEM operations:

```cuda
// TCGEN05 MMA with warp specialization
tcgen05.mma.ws.cta_group::1.kind::f16
    [tmem_c], desc_a, desc_b, scaleC, pred, 0;
```

---

## 15. Pipeline Lowering

### 15.1 Software Pipeline

The transform pass `src/transform/inject_pipeline.cc` and
`src/transform/pipeline_planning.cc` convert TileLang pipeline annotations into
overlapped execution:

```python
# TileLang pipeline
with T.pipeline(num_stages=3):
    for ko in range(K_tiles):
        T.copy(A_global[ko], A_smem[stage])
        T.commit()
        T.wait()
        T.gemm(A_smem, B_smem, C_frag)
```

### 15.2 Multi-Stage Pipeline Lowering

The pipeline lowering creates software-pipelined code where:

1. **Prologue:** Load first few stages
2. **Steady state:** Overlap load of stage (i+N) with compute of stage i
3. **Epilogue:** Compute remaining stages

```cuda
// Prologue: load stages 0..N-1
cp.async.cg.shared.global [smem_0], [gmem_0], 16;
cp.async.cg.shared.global [smem_1], [gmem_1], 16;

// Steady state
for (int k = 0; k < K_tiles - N; k++) {
    cp.async.cg.shared.global [smem_next], [gmem_next], 16;
    mma.sync ... (using smem_current);
    // Rotate buffers
}

// Epilogue: compute last N stages
mma.sync ... (using smem_last);
```

### 15.3 Fuse mbarrier Arrive Expect TX

The transform `src/transform/fuse_mbarrier_arrive_expect_tx.cc` optimizes
pipeline operations by fusing the mbarrier arrive with the expected transaction
count, reducing synchronization overhead.

---

## 16. CUDA Kernel Launch Configuration

### 16.1 Grid and Block Dimensions

TileLang determines grid and block dimensions from the program structure:

```python
# TileLang kernel definition
@T.prim_func
def kernel(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    C: T.Buffer((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=THREADS) as (bx, by):
        ...
```

This generates:

```cuda
dim3 grid(ceil(M / BLOCK_M), ceil(N / BLOCK_N));
dim3 block(THREADS);
kernel<<<grid, block, shared_mem_size, stream>>>(A, B, C);
```

### 16.2 Shared Memory Size

Dynamic shared memory size is determined by the compiler and passed to the
kernel launch:

```cuda
size_t shared_mem_size = compute_dynamic_smem_size(kernel_ir);
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

### 16.3 Cluster Launch (SM90+)

For kernels using thread block clusters:

```cuda
cudaLaunchAttribute cluster_attr;
cluster_attr.id = cudaLaunchAttributeClusterDimension;
cluster_attr.val.clusterDim.x = cluster_x;
cluster_attr.val.clusterDim.y = 1;
cluster_attr.val.clusterDim.z = 1;

cudaLaunchConfig_t config;
config.gridDim = grid;
config.blockDim = block;
config.sharedMem = shared_mem_size;
config.attrs = &cluster_attr;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel, ...);
```

---

## 17. SM90 (Hopper) Specific Features

### 17.1 WGMMA (Warp Group Matrix Multiply-Accumulate)

As detailed in Section 3.3, WGMMA is the primary compute instruction for Hopper.

### 17.2 TMA (Tensor Memory Access)

As detailed in Section 12, TMA provides hardware-accelerated tensor copies.

### 17.3 Cluster Support

As detailed in Section 13, clusters enable distributed shared memory.

### 17.4 Warp Specialization

As detailed in Section 14, warp specialization enables producer-consumer patterns.

### 17.5 Hopper-Specific Transforms

#### lower_hopper_intrin.cc

Converts Hopper-specific intrinsics into PTX operations:

- `tl.wgmma_gemm` -> WGMMA PTX instructions
- `tl.tma_copy` -> TMA bulk copy PTX
- `tl.cluster_wait` -> cluster barrier PTX

#### lower_shared_barrier.cc

Manages shared memory barrier operations for async pipelines.

#### lower_ptx_async_copy.cc

Lowers async copy operations to cp.async PTX instructions.

---

## 18. SM100 (Blackwell) Specific Features

### 18.1 TCGEN05 Instructions

As detailed in Section 3.4, TCGEN05 is the primary compute instruction for Blackwell.

### 18.2 Block-Scaled GEMM

Blackwell introduces block-scaled GEMM where scaling factors are applied per-block
rather than per-element:

```python
# Block-scaled GEMM in TileLang
desc = GetTCGEN5BlockScaledInstrDesc(
    atom_m, atom_n, ab_dtype,
    a_is_k_major, b_is_k_major,
    scale_in_a, scale_in_b,
    a_sf_id, b_sf_id
)
```

This generates PTX like:

```cuda
tcgen05.mma.kind::f16.scale_a.scale_b
    [tmem_c], desc_a, desc_b, scaleC, pred, 0;
```

Where scale factors come from shared memory and are applied to each block of the
input matrices.

### 18.3 Tensor Memory (TMEM)

TMEM is a new memory hierarchy level on Blackwell GPUs:

- Located between registers and shared memory
- Addressable by all warps within a CTA
- Used as accumulator storage for TCGEN05 MMA results
- `shared.tmem` scope in TileLang

```python
# TMEM allocation in TileLang
C_tmem = T.alloc_shared((M, N), dtype, scope="shared.tmem")
```

**Check:** `TargetHasTmem(target)` returns true for SM100+

#### TMEM Operations

The transform `src/transform/lower_shared_tmem.cc` handles TMEM-specific operations:

```cuda
// TMEM load
tmem.load.global.shared.tile.{b128, b64} [...], [...];

// TMEM store
tmem.store.shared.tile.global.{b128, b64} [...], [...];
```

### 18.4 2-CTA Mode

TCGEN05 supports operating across 2 CTAs in a cluster for larger operations:

```python
meta = GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype, disable_2cta=False)
# meta.enable_2cta indicates if 2-CTA mode is used
```

### 18.5 Blackwell-Specific Transforms

#### lower_blackwell_2sm.cc

Handles the dual-CTA (2SM) mode for TCGEN05 operations.

#### inject_tcgen05_fence.cc

Inserts necessary fences around TCGEN05 operations:

```cuda
// TCGEN05 fence
fence.proxy.tensormap::generic.async.shared.shared;
tcgen05.mma ...;
fence.proxy.async.shared.cta;
```

### 18.6 Vectorized 256-bit Operations

SM100+ supports 256-bit vectorized operations:

```cpp
bool TargetSupportVectorize256(Target target) {
  return TargetIsCuda(target) && arch >= 100;
}
```

This enables wider memory transactions for improved bandwidth utilization.

---

## 19. SM80/SM86 (Ampere) Features

### 19.1 MMA Instructions

Ampere introduced the `mma.sync` instruction family, which is the primary Tensor
Core interface:

```cuda
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

### 19.2 Async Copy (cp.async)

Ampere introduced asynchronous memory copies from global to shared memory:

```cuda
cp.async.cg.shared.global [smem], [gmem], 16;
cp.async.commit_group;
cp.async.wait_group 0;
```

**Check:** `TargetHasAsyncCopy(target)` returns true for SM80+

### 19.3 ldmatrix

The ldmatrix instruction loads data from shared memory into the register layout
expected by MMA:

```cuda
ldmatrix.sync.aligned.m8n8.x4.shared.b16
    {%0, %1, %2, %3}, [%4];
```

**Check:** `TargetHasLdmatrix(target)` returns true for SM75+

### 19.4 Sparse GEMM Support

Ampere supports sparse MMA operations where one operand has 2:4 sparsity:

```python
# Sparse GEMM in TileLang
from tilelang.layout import make_cutlass_metadata_layout

metadata_layout = make_cutlass_metadata_layout(buffer, mma_dtype="float16")
```

The sparse MMA uses a metadata vector to indicate which elements are non-zero:

```cuda
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {c0,...}, {a0,...}, {b0,...}, {c0_in,...}, metadata;
```

### 19.5 Sparse GEMM on SM80

For SM80/SM86, sparse GEMM uses the `gemm_sp.h` templates:

```python
# Sparse GEMM registration
register_gemm_impl("cuda.sparse_mma", ...)
```

Sparse GEMM on Ampere uses `gemm_sp_sm80.h` which implements 2:4 structured
sparsity using the metadata layout.

---

## 20. CUDA Templates (tl_templates/cuda/)

The CUDA backend uses several C++ template headers for code generation:

| File | Purpose |
|---|---|
| `common.h` | Common CUDA utilities, type traits |
| `gemm.h` | GEMM operation dispatcher |
| `gemm_mma.h` | Ampere MMA GEMM implementation |
| `gemm_sm70.h` | Volta (SM70) GEMM |
| `gemm_sm80.h` | Ampere (SM80) GEMM |
| `gemm_sm89.h` | Ada Lovelace (SM89) GEMM |
| `gemm_sm90.h` | Hopper (SM90) WGMMA GEMM (CuTe-based) |
| `gemm_sm100.h` | Blackwell (SM100) TCGEN05 GEMM (CuTe-based) |
| `gemm_sm120.h` | SM120 GEMM |
| `gemm_sp.h` | Sparse GEMM dispatcher |
| `gemm_sp_sm80.h` | Sparse GEMM for SM80 |
| `gemm_sp_sm90.h` | Sparse GEMM for SM90 |
| `copy.h` | Copy operations |
| `copy_sm90.h` | TMA-based copy for SM90 |
| `copy_sm100.h` | Copy for SM100 |
| `barrier.h` | Barrier/mbarrier operations |
| `cluster.h` | Thread block cluster operations |
| `reduce.h` | Warp/block reduction |
| `atomic.h` | Atomic operations |
| `ldsm.h` | ldmatrix/stmatrix operations |
| `intrin.h` | CUDA intrinsic wrappers |
| `debug.h` | Debug utilities |
| `compress_sm90.cu` | SM90 compression operations |
| `cuda_fp4.h` | FP4 data type support |
| `cuda_fp8.h` | FP8 data type support |
| `cuda_bf16_wrapper.h` | BF16 wrapper utilities |
| `cuda_bf16_fallbacks.cuh` | BF16 fallback implementations |
| `threadblock_swizzle.h` | Thread block swizzle for CTA mapping |
| `tcgen_05.h` | TCGEN05 common definitions |
| `tcgen_05_ld.h` | TCGEN05 load operations |
| `tcgen_05_st.h` | TCGEN05 store operations |

### Instruction Headers

| File | Purpose |
|---|---|
| `instruction/mma.h` | MMA inline PTX wrappers |
| `instruction/wgmma.h` | WGMMA inline PTX wrappers |
| `instruction/tcgen05mma.h` | TCGEN05 MMA inline PTX wrappers |
| `instruction/mma_sm70.h` | Volta MMA inline PTX wrappers |

---

## 21. Target String Configuration

The CUDA backend is selected via the target string:

```python
import tilelang as tl

# Basic CUDA target
target = tl.target.Target("cuda")

# With specific architecture
target = tl.target.Target("cuda", arch="sm_90")

# CuTeDSL target (for CuTe codegen)
target = tl.target.Target("cutedsl")
```

### Target Detection Functions

```python
# From tilelang/utils/target.py
target_is_cuda(target)     # True for CUDA and CuTeDSL targets
target_is_volta(target)    # SM70
target_is_turing(target)   # SM75
target_is_ampere(target)   # SM80-SM89
target_is_hopper(target)   # SM90-SM99
target_is_sm100(target)    # SM100+
```

### Architecture Version Detection (C++)

```cpp
// From src/target/utils.cc
bool TargetIsVolta(target)   { return arch >= 70 && arch < 75; }
bool TargetIsTuring(target)  { return arch >= 75 && arch < 80; }
bool TargetIsAmpere(target)  { return arch >= 80 && arch < 90; }
bool TargetIsHopper(target)  { return arch >= 90 && arch < 100; }
bool TargetIsSm100(target)   { return arch >= 100 && arch <= 110; }
bool TargetIsSM120(target)   { return arch >= 120 && arch < 130; }
```

---

## 22. Building and Compilation

### 22.1 NVRTC Compilation

TileLang uses NVRTC (NVIDIA Runtime Compiler) for JIT compilation:

```python
# From tilelang/contrib/nvrtc.py
def compile_cuda(code, target, arch):
    # Compile CUDA code using NVRTC
    ptx = nvrtc_compile(code, arch)
    return ptx
```

### 22.2 NVCC Compilation

For ahead-of-time compilation:

```python
# From tilelang/contrib/nvcc.py
def compile_cuda(code, target, arch):
    # Compile using nvcc
    cubin = nvcc_compile(code, arch)
    return cubin
```

### 22.3 Kernel Caching

Compiled kernels are cached to avoid recompilation:

```python
# From tilelang/cache/kernel_cache.py
cache = KernelCache()
cache.save(key=kernel_hash, compiled_module=module)
module = cache.load(key=kernel_hash)
```
