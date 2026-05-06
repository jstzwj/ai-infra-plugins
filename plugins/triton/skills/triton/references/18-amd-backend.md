# Chapter 18: AMD Backend

## Architecture

The AMD backend compiles Triton kernels for ROCm/HIP GPUs (ROCm 6.2+).

### Supported Architectures

| Architecture | GFX | Target | Features |
|-------------|-----|--------|----------|
| CDNA2 | gfx90a | MI250X | Matrix cores |
| CDNA3 | gfx942 | MI300X/A | MFMA, FP8 |
| CDNA4 | gfx950 | MI350X | MFMA scaled, async copy |
| RDNA3 | gfx11xx | RX 7000 | WMMA |
| RDNA4 | gfx1200 | RX 9000 | WMMA |
| GFX1250 | gfx1250 | Future | TDM, cluster |

### HIPBackend

```python
class HIPBackend(BaseBackend):
    def add_stages(self, stages, options):
        stages["ttir"] = (lambda src: src, True)
        stages["ttgir"] = (ttir_to_ttgir, True)
        stages["llir"] = (ttgir_to_llir, True)
        stages["amdgcn"] = (llir_to_amdgcn, True)
        stages["hsaco"] = (amdgcn_to_hsaco, False)
```

**Pipeline:** AST → TTIR → TTGIR → LLVM IR → AMDGPU ISA → HSACO

### HIPOptions

```python
@dataclass
class HIPOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    maxnreg: int = 0
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    enable_persistent: bool = False
    extern_libs: dict = None
    ...
```

### Architecture-Specific Features

#### CDNA3 (gfx942)
- MFMA (Matrix Fused Multiply-Add) instructions
- FP8 data types
- Buffer load/store operations

#### CDNA4 (gfx950)
- MFMA scaled operations
- Async copy with swizzle
- In-thread transpose

#### GFX1250
- TDM (Tensor Data Movement)
- Cluster operations
- Async copy with mbarrier

### Buffer Operations

AMD-specific buffer operations for efficient memory access:

```python
# CDNA3 buffer load
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
data = buffer_load(ptr, indices, mask)

# CDNA3 buffer store
from triton.experimental.gluon.language.amd.cdna3 import buffer_store
buffer_store(ptr, indices, data, mask)
```

### MFMA Operations

Matrix operations for AMD GPUs:

```python
# CDNA3 MFMA
from triton.experimental.gluon.language.amd.cdna3 import mfma
result = mfma(a, b, acc)

# CDNA4 MFMA scaled
from triton.experimental.gluon.language.amd.cdna4 import mfma_scaled
result = mfma_scaled(a, a_scale, b, b_scale, acc)
```

## HIPDriver

### Kernel Launch

```python
class HIPLauncher:
    def __call__(self, *args, grid, stream=None):
        # Compiles C launcher code
        # Loads HSACO binary
        # Configures grid
        # Launches kernel via HIP
```

### Type Mapping

| Triton | C/HIP |
|--------|-------|
| `i1` | `bool` |
| `i8` | `int8_t` |
| `i32` | `int32_t` |
| `i64` | `int64_t` |
| `fp16` | `_Float16` |
| `bf16` | `hip_bfloat16` |
| `fp32` | `float` |
| `fp64` | `double` |

## Environment Variables (AMD-specific)

| Variable | Description |
|----------|-------------|
| `AMDGCN_USE_BUFFER_OPS` | Use buffer operations (default: true) |
| `AMDGCN_USE_BUFFER_ATOMICS` | Use buffer atomics (default: true) |
| `TRITON_LIBHIP_PATH` | Custom libhip path |
| `TRITON_HIP_USE_BLOCK_PINGPONG` | Enable ping-pong scheduling |
| `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` | Enable in-thread transpose |
| `TRITON_HIP_USE_ASYNC_COPY` | Enable async copy |
| `AMDGCN_ENABLE_DUMP` | Dump AMDGPU ISA |
| `AMDGCN_SCALARIZE_PACKED_FOPS` | Scalarize packed operations |
| `TRITON_DUMP_MIR` | Dump MIR for debugging |
| `TRITON_SWAP_MIR` | Swap MIR files |
| `TRITON_ENABLE_ASAN` | Enable address sanitizer |

## Address Sanitizer

AMD supports address sanitizer for memory debugging:

```bash
export TRITON_ENABLE_ASAN=1
python your_script.py
```

Requires ASAN libraries documented at [ROCm GPU Sanitizer](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).
