# Chapter 11: Runtime Driver Module (`triton.runtime.driver`)

## Driver Architecture

The driver module provides a unified interface to GPU operations across different hardware backends (CUDA, HIP).

### `DriverConfig` Class

```python
class DriverConfig:
    _default: Optional[DriverBase] = None
    _active: Optional[DriverBase] = None

    @property
    def default(self) -> DriverBase

    @property
    def active(self) -> DriverBase

    def set_active(self, name: str)
    def reset_active(self)
```

### Global Driver Instance

```python
from triton.runtime import driver

# Access active driver
drv = driver.driver.active

# Set default backend
driver.driver.set_active("cuda")  # or "hip"

# Reset to default
driver.driver.reset_active()
```

## DriverBase ABC

```python
class DriverBase(ABCMeta):
    @classmethod
    @abstractmethod
    def is_active(cls) -> bool:
        """Check if this driver is available on the system."""

    @abstractmethod
    def get_current_target(self) -> GPUTarget:
        """Get the current GPU target."""

    @abstractmethod
    def get_active_torch_device(self):
        """Get the active PyTorch device."""

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """Get the benchmarking function."""

    def map_python_to_cpp_type(self, ty: str) -> str:
        """Map Triton type to C++ type."""
```

## GPUDriver Class

Concrete implementation using PyTorch CUDA:

```python
class GPUDriver(DriverBase):
    def __init__(self):
        # Uses torch.cuda for device management

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        """Convert tensor maps to kernel arguments."""

    def allocate_default_profile_scratch(self, size, alignment, stream):
        """Allocate GPU memory for profiling."""
```

## Backend Discovery

### `_discover_backends() -> dict[str, Backend]`

Discovers available backends through:
1. **Entry points:** `triton.backends` entry point group
2. **In-tree backends:** `triton.backends.{name}` when `TRITON_BACKENDS_IN_TREE=1`

```python
# All discovered backends
from triton.backends import backends
# backends = {"nvidia": Backend(compiler=CUDABackend, driver=CUDADriver),
#             "amd": Backend(compiler=HIPBackend, driver=HIPDriver)}
```

## Benchmarker Protocol

```python
class Benchmarker(Protocol):
    def __call__(
        self,
        kernel_call: Callable,
        *,
        quantiles: List[float],
        **kwargs
    ) -> Sequence[float]:
        """Benchmark a kernel call and return timing statistics."""
```

## Memory Allocation

### Setting a Custom Allocator

```python
import triton

def my_allocator(size: int, alignment: int, stream=None):
    """Custom allocator for kernel workspace memory."""
    import torch
    buf = torch.empty(size, dtype=torch.int8, device='cuda')
    return buf

triton.runtime.set_allocator(my_allocator)
```

### Buffer Protocol

```python
class Buffer(Protocol):
    def data_ptr(self) -> int:
        """Return the device pointer."""
```

### Allocator Protocol

```python
class Allocator(Protocol):
    def __call__(
        self,
        size: int,
        alignment: int,
        stream: Optional[int]
    ) -> Buffer:
        """Allocate a buffer of given size with alignment."""
```

### Profile Allocator

```python
# Set a profile allocator called before kernel launch
triton.runtime.set_profile_allocator(my_allocator)

# Check if profile allocator is set
if triton.runtime.has_profile_allocator():
    print("Profile allocator active")
```

## CUDA-Specific Driver

### `CUDADriver`
Located in `third_party/nvidia/backend/driver.py`:

- `CudaUtils`: Singleton for CUDA utility functions
- `CudaLauncher`: Kernel launcher for CUDA
- Kernel compilation to C launcher code
- PTX assembly and CUBIN loading
- Tensor map assembly for TMA

### `ty_to_cpp` Type Mapping

| Triton Type | C Type |
|------------|--------|
| `i1` | `bool` |
| `i8` | `int8_t` |
| `i16` | `int16_t` |
| `i32` | `int32_t` |
| `i64` | `int64_t` |
| `u8` | `uint8_t` |
| `u16` | `uint16_t` |
| `u32` | `uint32_t` |
| `u64` | `uint64_t` |
| `fp16` | `half` |
| `bf16` | `__nv_bfloat16` |
| `fp32` | `float` |
| `fp64` | `double` |

## HIP-Specific Driver

### `HIPDriver`
Located in `third_party/amd/backend/driver.py`:

- `HIPUtils`: Singleton for HIP utility functions
- `HIPLauncher`: Kernel launcher for HIP
- Kernel compilation to C launcher code
- HSACO binary loading
- Architecture-specific support (gfx90a, gfx942, gfx950, gfx1250)

## CUDA Graph Support

```python
# Using CUDA graphs with Triton kernels
import torch

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    # Warmup
    for _ in range(3):
        kernel[grid](args)

with torch.cuda.graph(g):
    kernel[grid](args)

# Replay graph (fast, no Python overhead)
g.replay()
```
