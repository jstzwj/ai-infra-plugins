# Chapter 17: NVIDIA Backend

## Architecture

The NVIDIA backend compiles Triton kernels for CUDA GPUs (Compute Capability 8.0+).

### Compilation Stages

```python
class CUDABackend(BaseBackend):
    def add_stages(self, stages, options):
        stages["ttir"] = (lambda src: src, True)
        stages["ttgir"] = (ttir_to_ttgir, True)
        stages["llir"] = (ttgir_to_llir, True)
        stages["ptx"] = (llir_to_ptx, True)
        stages["cubin"] = (ptx_to_cubin, False)
```

**Pipeline:** AST → TTIR → TTGIR → LLVM IR → PTX → CUBIN

### CUDAOptions

```python
@dataclass
class CUDAOptions:
    num_warps: int = 4        # Number of warps per block
    num_ctas: int = 1         # Number of CTAs (cooperative launch)
    num_stages: int = 3       # Pipeline stages
    maxnreg: int = 0          # Max registers per thread
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    enable_persistent: bool = False
    extern_libs: dict = None
    debug: bool = False
    sanitize_overflow: bool = True
    ...
```

### Compute Capabilities

| Architecture | CC | Features |
|-------------|-----|----------|
| Ampere | 8.0, 8.6, 8.9 | Tensor cores, async copy |
| Hopper | 9.0 | TMA, warp specialization, FP8 |
| Blackwell | 10.0, 10.3 | Tensor memory, FP4, cluster |

### Key Functions

```python
# Convert compute capability to SM architecture string
sm_arch = sm_arch_from_capability((9, 0))  # "sm_90"

# Get PTX assembler
ptxas = get_ptxas(arch)  # Returns NvidiaTool

# Get LLVM features
features = get_features(arch)
```

### Tensor Memory Access (TMA)

Available on Hopper (CC 9.0+) and Blackwell:

```python
# TMA tensor descriptor
desc = tl.make_tensor_descriptor(
    base=ptr,
    shape=(M, K),
    strides=(stride_m, stride_k),
    block_shape=(BLOCK_M, BLOCK_K),
)

# Load using descriptor
data = desc.load([offset_m, offset_k])

# Store using descriptor
desc.store([offset_m, offset_k], data)
```

### Warp Specialization

Available on Hopper+:

```python
# In Gluon mode
with tl.warp_specialize(num_warps_d=1, num_warps_lds=3):
    # Different warp groups do different work
```

### External Libraries

```python
# Link external CUDA libraries
@triton.jit
def kernel(..., extern_libs={"libdevice": "/path/to/libdevice.10.bc"}):
    pass
```

## CudaDriver

### Kernel Launch

```python
class CudaLauncher:
    def __call__(self, *args, grid, stream=None):
        # Compiles C launcher code
        # Loads kernel binary
        # Configures grid dimensions
        # Launches kernel
```

### Type Mapping

| Triton | C/CUDA |
|--------|--------|
| `i1` | `bool` |
| `i8` | `int8_t` |
| `i16` | `int16_t` |
| `i32` | `int32_t` |
| `i64` | `int64_t` |
| `fp16` | `half` |
| `bf16` | `__nv_bfloat16` |
| `fp32` | `float` |
| `fp64` | `double` |

### CUDA Graph Support

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    kernel[grid](args)
g.replay()  # Fast replay
```

## CUDA-Specific Language Extensions

Located in `triton.language.extra.cuda`:

```python
from triton.language.extra.cuda import (
    gdc_wait,               # Programmatic dependent wait
    gdc_launch_dependents,  # Launch dependent kernels
)
```

## Environment Variables (NVIDIA-specific)

| Variable | Description |
|----------|-------------|
| `TRITON_PTXAS_PATH` | Custom ptxas path |
| `TRITON_CUOBJDUMP_PATH` | Custom cuobjdump path |
| `TRITON_NVDISASM_PATH` | Custom nvdisasm path |
| `TRITON_LIBDEVICE_PATH` | Custom libdevice path |
| `PTXAS_OPTIONS` | Additional ptxas options |
| `DISABLE_PTXAS_OPT` | Disable ptxas optimizations |
| `NVPTX_ENABLE_DUMP` | Dump NVPTX IR |
| `TRITON_CUDACRT_PATH` | CUDA CRT path |
| `TRITON_CUDART_PATH` | CUDA runtime path |
