# Paged Memory Management Reference

This document provides a comprehensive reference for bitsandbytes' paged memory management system, which uses CUDA Unified Memory to enable training of large models whose optimizer states would otherwise exceed GPU memory capacity.

---

## Table of Contents

1. [Overview](#overview)
2. [CUDA Unified Memory](#cuda-unified-memory)
3. [GlobalPageManager](#globalpagemanager)
4. [prefetch_tensor](#prefetch_tensor)
5. [Paged Optimizer States](#paged-optimizer-states)
6. [Paged Optimizer Step](#paged-optimizer-step)
7. [elementwise_func Infrastructure](#elementwise_func-infrastructure)
8. [XPU Paged Memory](#xpu-paged-memory)
9. [Use Cases](#use-cases)

---

## Overview

When training large models, optimizer states (e.g., Adam's momentum and variance) consume significantly more memory than the model parameters themselves. For a model with N parameters in float16:
- Parameters: 2N bytes
- Adam state1 (momentum): 4N bytes (float32)
- Adam state2 (variance): 4N bytes (float32)
- Total optimizer state: 8N bytes (4x the model)

For a 7B parameter model, the optimizer states alone require approximately 56GB. The paged memory system allows these states to be allocated in CUDA Unified Memory, which can overflow to CPU RAM when GPU memory is full, and be prefetched back to GPU when needed.

### Architecture

```
GPU Memory (VRAM)                  CPU Memory (RAM)
+-------------------+              +-------------------+
| Model Parameters  |              |                   |
| Activations       |              | Paged Optimizer   |
| Gradients         |    <--->     | States (overflow) |
| Active Optimizer  |   prefetch   |                   |
| States            |              |                   |
+-------------------+              +-------------------+
         ^                                  |
         |         CUDA Unified Memory       |
         +----------------------------------+
                    (Managed by driver)
```

### Key Concepts

- **CUDA Unified Memory**: A single address space accessible from both CPU and GPU. The CUDA driver automatically migrates pages between CPU and GPU memory.
- **Paged Tensor**: A tensor allocated via `cudaMallocManaged()` with the `is_paged=True` attribute set. It lives in unified memory and can be migrated between devices.
- **Prefetch**: Explicitly moving a paged tensor's data to a specific device (GPU or CPU) before it is needed, to avoid page faults during computation.
- **Threshold**: Only tensors with 100,000+ elements (approximately 400KB for float32) are allocated as paged tensors. Smaller tensors remain as regular GPU allocations.

---

## CUDA Unified Memory

### get_paged()

Allocates a tensor in CUDA Unified Memory via the native library.

**Location:** `bitsandbytes/functional.py`

```python
def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    num_bytes = dtype.itemsize * prod(shape)
    managed_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(managed_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out
```

#### Step-by-Step Breakdown

**1. Compute allocation size:**
```python
num_bytes = dtype.itemsize * prod(shape)
```
For a tensor of shape `(4096, 4096)` with dtype `float32`, this is `4 * 4096 * 4096 = 67,108,864` bytes (64MB).

**2. Allocate managed memory via C function:**
```python
managed_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
```

**3. Convert pointer to numpy array:**
```python
c_ptr = ct.cast(managed_ptr, ct.POINTER(ct.c_int))
new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
```
The raw pointer is cast to a pointer-to-int type, then numpy's `ctypeslib.as_array` wraps it as a numpy ndarray without copying data.

**4. Wrap numpy array as PyTorch tensor:**
```python
out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
```
`torch.frombuffer` creates a tensor that shares memory with the numpy array (and thus the CUDA managed memory).

**5. Set paged attributes:**
```python
out.is_paged = True
out.page_deviceid = device.index
```
These custom attributes mark the tensor as paged and record which GPU device it should be prefetched to.

### C Implementation (cget_managed_ptr)

#### CUDA Path

```cpp
void* cget_managed_ptr(size_t bytes) {
    void* ptr;
    CUDA_CHECK_RETURN(cudaMallocManaged(&ptr, bytes, cudaMemAttachHost));
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
    return ptr;
}
```

`cudaMallocManaged` allocates `bytes` bytes of unified memory. The `cudaMemAttachHost` flag indicates the memory is initially accessible from the host (CPU). The returned pointer is a `c_void_p` that wraps the raw address.

#### XPU Path

```cpp
void* cget_managed_ptr(size_t bytes) {
    try {
        auto& q = xpu_default_queue();
        void* ptr = sycl::malloc_shared(bytes, q);
        if (ptr == nullptr) {
            fprintf(stderr, "XPU Error: sycl::malloc_shared returned nullptr for %zu bytes\n", bytes);
        }
        return ptr;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "XPU SYCL Error in cget_managed_ptr: %s\n", e.what());
        return nullptr;
    }
}
```

On XPU, SYCL's `sycl::malloc_shared` provides the equivalent of CUDA managed memory. The memory is accessible from both host and device.

---

## GlobalPageManager

`GlobalPageManager` is a singleton that tracks all paged tensors and provides bulk prefetch operations.

**Location:** `bitsandbytes/functional.py`

```python
class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance
```

### paged_tensors

A list of all paged tensors created by the optimizer. Each entry is a tensor returned by `get_paged()` with `is_paged=True` and `page_deviceid` set. Tensors are appended in creation order.

### prefetch_all()

```python
def prefetch_all(self, to_cpu=False):
    # Assume the first added will be the ones that are used first,
    # so swap them in last in case they are evicted again.
    for t in self.paged_tensors[::-1]:
        prefetch_tensor(t, to_cpu)
```

Prefetches all tracked paged tensors in **reverse order** (LIFO -- Last In, First Out). The rationale is that tensors added later (which appear at the end of the list) are likely to be used sooner, so they are prefetched first and are more likely to remain in GPU memory.

The `to_cpu` parameter controls the direction:
- `to_cpu=False` (default): Prefetch to GPU (device indicated by `page_deviceid`)
- `to_cpu=True`: Prefetch to CPU (device ID set to -1)

---

## prefetch_tensor

Explicitly prefetches a paged tensor to a specific device.

**Location:** `bitsandbytes/functional.py`

```python
def prefetch_tensor(A: torch.Tensor, to_cpu=False):
    assert A.is_paged, "Only paged tensors can be prefetched!"

    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid

    lib.cprefetch(get_ptr(A), ct.c_size_t(A.nbytes), ct.c_int32(deviceid))
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `torch.Tensor` | The paged tensor to prefetch (must have `is_paged=True`) |
| `to_cpu` | `bool` | If True, prefetch to CPU (deviceid=-1). If False, prefetch to GPU. |

### C Implementation (CUDA)

```cpp
void cprefetch(void* ptr, size_t bytes, int device) {
    int hasPrefetch = 0;
    CUDA_CHECK_RETURN(
        cudaDeviceGetAttribute(&hasPrefetch, cudaDevAttrConcurrentManagedAccess, device)
    );  // ~40ns overhead
    if (hasPrefetch == 0)
        return;  // Device does not support prefetching

#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;
    CUDA_CHECK_RETURN(cudaMemPrefetchAsync(ptr, bytes, loc, 0u, 0));
#else
    CUDA_CHECK_RETURN(cudaMemPrefetchAsync(ptr, bytes, device, 0));
#endif
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}
```

Key details:
1. **Concurrency check**: `cudaDevAttrConcurrentManagedAccess` must be non-zero for prefetching to work. On devices without this capability, the function is a no-op.
2. **CUDA 13.0+ path**: Uses the newer `cudaMemLocation` struct API.
3. **Pre-13.0 path**: Uses the simpler `cudaMemPrefetchAsync(ptr, bytes, device, stream)` API.
4. **Stream 0**: The prefetch is submitted to the default stream (stream 0), meaning it will execute in order with other default-stream operations.

### C Implementation (XPU)

```cpp
void cprefetch(void* ptr, size_t bytes, int device) {
    if (device < 0)
        return;  // SYCL prefetch targets the device associated with the queue
    try {
        auto& q = xpu_default_queue();
        q.prefetch(ptr, bytes);
    } catch (const sycl::exception& e) {
        fprintf(stderr, "XPU Warning: sycl::queue::prefetch failed: %s\n", e.what());
    }
}
```

On XPU, SYCL's `queue::prefetch()` provides the equivalent functionality. The `device < 0` case (CPU prefetch) is skipped because SYCL prefetching targets the device associated with the queue, not the host.

---

## Paged Optimizer States

### get_state_buffer()

The `Optimizer8bit.get_state_buffer()` method decides whether to use regular or paged allocation for optimizer state tensors.

**Location:** `bitsandbytes/optim/optimizer.py`

```python
def get_state_buffer(self, p, dtype=torch.float32):
    if p.device.type == "cpu":
        if self.is_paged and not getattr(self, "_cpu_paged_warned", False):
            warnings.warn(
                "Paged optimizers are not supported on CPU. "
                "Falling back to non-paged optimizer behavior.",
                stacklevel=2,
            )
            self._cpu_paged_warned = True
        return torch.zeros_like(p, dtype=dtype, device=p.device)

    if not self.is_paged or p.numel() < 1e5:
        # Regular allocation: standard GPU tensor
        return torch.zeros_like(p, dtype=dtype, device=p.device)
    else:
        # Paged allocation: unified memory tensor
        # > 1 MB (approximately)
        buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
        F.fill(buff, 0)
        self.page_mng.paged_tensors.append(buff)
        return buff
```

### Decision Logic

```
get_state_buffer(p, dtype)
    |
    +-- CPU device?
    |       -> torch.zeros_like (regular CPU tensor)
    |          (warns once if is_paged, since CPU paging is not supported)
    |
    +-- Not paged OR parameter has < 100,000 elements?
    |       -> torch.zeros_like (regular GPU tensor)
    |
    +-- Paged AND parameter has >= 100,000 elements?
            -> F.get_paged (unified memory tensor)
               F.fill(buff, 0)
               Track in page_mng.paged_tensors
```

### Threshold: 100,000 Elements

The threshold of `1e5` (100,000) elements corresponds to approximately:
- float32: 100,000 * 4 = 400 KB
- float16: 100,000 * 2 = 200 KB
- uint8: 100,000 * 1 = 100 KB

Below this size, the overhead of unified memory management (page faults, prefetch latency) outweighs the memory savings. Small tensors remain as regular GPU allocations.

### Zero-Initialization

Paged tensors are filled with zeros using `F.fill(buff, 0)` rather than `torch.zeros()` because the paged tensor is allocated via raw memory and is not initialized:

```python
buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
F.fill(buff, 0)
```

The `fill` function uses the C kernel `cfill_fp32` (or `cfill_uint8`) to set all elements to the specified value, handling prefetch and synchronization for paged tensors.

### Paged Tensor Tracking

Each paged tensor is appended to `GlobalPageManager.paged_tensors`:

```python
self.page_mng.paged_tensors.append(buff)
```

This allows `prefetch_all()` to track and prefetch all optimizer state tensors at once.

### Usage in Optimizer Init

The `Optimizer2State.init_state()` method allocates two state buffers (for Adam's momentum and variance):

```python
if dtype == torch.float32:
    state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
    state["state2"] = self.get_state_buffer(p, dtype=torch.float32)
elif dtype == torch.uint8:
    state["state1"] = self.get_state_buffer(p, dtype=torch.uint8)
    # ...
    state["state2"] = self.get_state_buffer(p, dtype=torch.uint8)
    # ...
```

When `is_paged=True` and the parameter is large enough, both `state1` and `state2` will be paged tensors tracked by the `GlobalPageManager`.

### 8-bit Paged State Layout

For `Optimizer2State` (e.g., Adam) with 8-bit paged:

| Key | Type | Paged? | Reason |
|-----|------|--------|--------|
| `state1` | uint8 | Yes (if large enough) | Quantized optimizer state (e.g., Adam's m) |
| `state2` | uint8 | Yes (if large enough) | Quantized optimizer state (e.g., Adam's v) |
| `qmap1` | float32 (256,) | No | Shared quantization map (small, constant) |
| `qmap2` | float32 (256,) | No | Shared quantization map (small, constant) |
| `absmax1` | float32 | No | Block absmax (1/256 of state size) |
| `absmax2` | float32 | No | Block absmax (1/256 of state size) |
| `step` | int | No | Scalar counter |

Only `state1` and `state2` are paged because they are the large tensors (same number of elements as the parameter). The quantization maps and absmax vectors are small enough to remain in regular GPU memory.

---

## Paged Optimizer Step

### Overall Step Flow

```python
class Optimizer8bit:
    @torch.no_grad()
    def step(self, closure=None):
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        p = None
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)                  # 1. Prefetch paged states
                self.update_step(group, p, gindex, pindex)  # 2. Run optimizer kernel
                sync_gpu(p)                             # 3. Wait for async ops

        if self.is_paged and p is not None:
            # All paged operations are asynchronous. We need to sync
            # to make sure all tensors are in the correct state.
            sync_gpu(p)
```

### prefetch_state()

Before each optimizer update step, the optimizer prefetches the state tensors to GPU:

```python
def prefetch_state(self, p):
    if self.is_paged:
        state = self.state[p]
        s1 = state["state1"]
        is_paged = getattr(s1, "is_paged", False)
        if is_paged:
            F.prefetch_tensor(state["state1"])
            if "state2" in state:
                F.prefetch_tensor(state["state2"])
```

This ensures the optimizer state is resident in GPU memory before the update computation begins, avoiding page faults during the kernel execution. Only tensors with `is_paged=True` are prefetched; regular tensors are already on GPU.

### update_step()

The actual optimizer update is performed by `update_step()`, which calls the appropriate C kernel:

```python
@torch.no_grad()
def update_step(self, group, p, gindex, pindex):
    # Ensure contiguous memory layout (avoids errors from non-contiguous data)
    p.data = p.data.contiguous()
    p.grad = p.grad.contiguous()

    state = self.state[p]
    grad = p.grad
    config = self.get_config(gindex, pindex, group)
    state["step"] += 1

    if state["state1"].dtype == torch.float:
        F.optimizer_update_32bit(
            self.optimizer_name, grad, p,
            state["state1"], config["betas"][0], config["eps"],
            state["step"], config["lr"],
            state["state2"], config["betas"][1],
            ...
        )
    elif state["state1"].dtype == torch.uint8:
        F.optimizer_update_8bit_blockwise(
            self.optimizer_name, grad, p,
            state["state1"], state["state2"],
            ...
        )
```

### sync_gpu()

After each optimizer step, a GPU synchronization ensures all async operations complete:

```python
def sync_gpu(t: torch.Tensor):
    if t.device.type == "cuda":
        torch.cuda.synchronize()
    elif t.device.type == "xpu":
        torch.xpu.synchronize()
```

### Why Synchronization Is Critical

Paged operations are fully asynchronous. The CUDA kernel may start executing while the data is still being prefetched. Without synchronization, the optimizer could read stale data or the parameter update could be incomplete when the next operation begins. The explicit `torch.cuda.synchronize()` call acts as a barrier, ensuring:
1. All prefetch operations have completed
2. All optimizer update kernels have finished
3. The tensor data is in a consistent state before the next parameter is processed

---

## elementwise_func Infrastructure

The `elementwise_func` function provides a generic interface for elementwise operations on paged tensors, with automatic prefetch and synchronization.

**Location:** `bitsandbytes/functional.py`

```python
def elementwise_func(func_name, A, B, value, prefetch=True):
    # Select the appropriate C function based on dtype
    func = None
    if A.dtype == torch.float32:
        func = getattr(lib, f"c{func_name}_fp32", None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f"c{func_name}_uint8", None)
        cvalue = ct.c_uint8(value)

    if func is None:
        raise NotImplementedError(f"Function not implemented: {func_name}")

    # Prefetch paged tensors if needed
    is_managed = getattr(A, "is_managed", False)
    if is_managed and prefetch:
        prefetch_tensor(A)
        if B is not None:
            prefetch_tensor(B)

    # Execute the C kernel
    func(get_ptr(A), get_ptr(B), cvalue, ct.c_int64(A.numel()))

    # Synchronize for paged tensors
    if A.is_paged or B.is_paged:
        # Paged functions are fully asynchronous.
        # If we return from this function, we want the tensor
        # to be in the correct state, that is the final state after the
        # operation occurred. So we synchronize.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
```

### fill()

```python
def fill(A, value, device=None, prefetch=True):
    elementwise_func("fill", A, None, value)
```

Fills tensor `A` with the given scalar `value`. Uses the `cfill_fp32` or `cfill_uint8` C kernel.

### _mul()

```python
def _mul(A, B, device=None):
    elementwise_func("_mul", A, B, 0)
```

Elementwise multiplication: `A *= B`. The `value` parameter is unused (set to 0) for multiply operations.

### C Kernel Implementations

```cpp
// CUDA path
MAKE_ELEMENTWISE_FUNC(fill, fp32, float, FILL)
MAKE_ELEMENTWISE_FUNC(fill, uint8, unsigned char, FILL)
MAKE_ELEMENTWISE_FUNC(_mul, fp32, float, _MUL)
```

These expand to functions like `fill_fp32(float* A, float* B, float value, long n)` and `_mul_fp32(float* A, float* B, float value, long n)`, which launch CUDA kernels with one thread per element.

### XPU fill Implementations

```cpp
void cfill_fp32(float* A, float* B, float value, long n) {
    auto& q = xpu_default_queue();
    q.fill(A, value, static_cast<size_t>(n)).wait();
}

void cfill_uint8(unsigned char* A, unsigned char* B, unsigned char value, long n) {
    // Use host-side memset instead of sycl::queue::fill<unsigned char>
    // which segfaults on certain Intel GPU drivers (e.g. Max 1550).
    // USM shared memory is host-accessible, so memset works directly.
    memset(A, value, static_cast<size_t>(n));
}
```

### Prefetch and Synchronize Pattern

The `elementwise_func` follows the same pattern as the optimizer step:
1. **Prefetch**: If the tensor has `is_managed=True`, prefetch it to GPU before the operation.
2. **Execute**: Call the C kernel with raw pointers.
3. **Synchronize**: If either tensor is paged, synchronize to ensure the operation completes before returning.

This pattern ensures correctness: the operation does not return until the data is in its final state.

---

## XPU Paged Memory

On Intel XPU (GPU) devices, paged memory uses SYCL's Unified Shared Memory (USM) instead of CUDA's managed memory.

### API Equivalence

| Operation | CUDA | XPU (SYCL) |
|-----------|------|------------|
| Allocate managed memory | `cudaMallocManaged` | `sycl::malloc_shared` |
| Prefetch to device | `cudaMemPrefetchAsync` | `sycl::queue::prefetch` |
| Error checking | `cudaPeekAtLastError` | SYCL exceptions |
| Fill (fp32) | Custom CUDA kernel | `sycl::queue::fill` |
| Fill (uint8) | Custom CUDA kernel | `memset` (host-side) |

### XPU Default Queue

```cpp
static sycl::queue& xpu_default_queue() {
    static sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    return q;
}
```

The queue is created with:
- `sycl::gpu_selector_v`: Selects the first available GPU
- `sycl::property::queue::in_order{}`: Ensures operations execute in submission order (important for prefetch synchronization)

---

## Use Cases

### Large Model Training with Paged AdamW

```python
import bitsandbytes as bnb

# Create model
model = LargeModel(num_params=7_000_000_000)  # 7B parameters

# Use paged 8-bit AdamW optimizer
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=1e-4,
)

# Training loop -- paging is transparent
for batch in dataloader:
    loss = model(batch).loss
    loss.backward()
    optimizer.step()  # Automatically prefetches and synchronizes paged states
    optimizer.zero_grad()
```

### Memory Savings Calculation

For a 7B parameter model with Adam optimizer:

| Component | Standard (float32) | 8-bit Quantized | Paged 8-bit |
|-----------|-------------------|-----------------|-------------|
| Parameters | 14 GB (fp16) | 14 GB (fp16) | 14 GB (fp16) |
| State1 (m) | 28 GB (fp32) | 7 GB (uint8) | 7 GB (paged) |
| State2 (v) | 28 GB (fp32) | 7 GB (uint8) | 7 GB (paged) |
| Absmax1 | - | 0.11 GB (fp32) | 0.11 GB (paged) |
| Absmax2 | - | 0.11 GB (fp32) | 0.11 GB (paged) |
| Qmap1 | - | 0.001 GB (fp32) | 0.001 GB |
| Qmap2 | - | 0.001 GB (fp32) | 0.001 GB |
| **Total GPU** | **70 GB** | **28.2 GB** | **~14 GB + active pages** |

With paged 8-bit Adam, only the optimizer states for the currently-updating parameter group need to be resident in GPU memory. The rest can stay in CPU RAM, reducing peak GPU memory usage dramatically.

### 32-bit vs 8-bit Paged Optimizers

```python
# 32-bit paged optimizer (states in float32, but paged to CPU)
optimizer = bnb.optim.PagedAdamW32bit(
    model.parameters(),
    lr=1e-4,
)

# 8-bit paged optimizer (states quantized to uint8 AND paged)
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(),
    lr=1e-4,
)
```

The 8-bit paged optimizer provides the best memory savings because it combines two memory reduction techniques:
1. **Quantization**: Reduces state size by 4x (float32 to uint8)
2. **Paging**: Offloads inactive states to CPU RAM

### When to Use Paged Optimizers

| Scenario | Recommendation |
|----------|---------------|
| GPU memory > 2x model size | Regular (non-paged) optimizer |
| GPU memory < 2x model size | Paged optimizer (avoid OOM) |
| Very large model (>30B) | Paged 8-bit optimizer (maximum savings) |
| Multi-GPU with FSDP | Paged optimizer per rank |
| CPU training | Paged optimizers not supported (warning issued) |

### Available Paged Optimizers

| Regular Optimizer | Paged Equivalent |
|-------------------|-----------------|
| `bnb.optim.Adam` | `bnb.optim.PagedAdam` |
| `bnb.optim.AdamW` | `bnb.optim.PagedAdamW` |
| `bnb.optim.Adam8bit` | `bnb.optim.PagedAdam8bit` |
| `bnb.optim.AdamW8bit` | `bnb.optim.PagedAdamW8bit` |
| `bnb.optim.Lion8bit` | `bnb.optim.PagedLion8bit` |
| `bnb.optim.AdEMAMix8bit` | `bnb.optim.PagedAdEMAMix8bit` |

The paged variants set `is_paged=True` in the constructor and use `get_state_buffer()` for state allocation. All other optimizer logic (quantization, update formulas, configuration) is identical.

### Performance Considerations

1. **PCIe bandwidth**: Paging between GPU and CPU goes through PCIe. Expect approximately 12-25 GB/s for PCIe Gen4 x16. The prefetch latency for a 64MB state tensor is about 2.5-5.3ms.

2. **Prefetch timing**: States are prefetched just before the optimizer step. If prefetch has not completed, the kernel stalls waiting for the data, which shows up as increased step time.

3. **Small parameters**: Parameters under 100K elements use regular allocation to avoid paging overhead. This threshold is appropriate for most bias vectors, layer norm parameters, and small embedding tables.

4. **CPU fallback**: Paged optimizers on CPU produce a warning and fall back to non-paged behavior. CPU paging would require a different mechanism (e.g., mmap) that is not currently implemented.

5. **Single GPU optimization**: When `torch.cuda.device_count() == 1`, the `_cuda_device_of` context manager uses `contextlib.nullcontext()`, avoiding the overhead of `cudaGetDevice/cudaSetDevice` calls.

6. **Synchronization cost**: `torch.cuda.synchronize()` blocks the CPU until all GPU work completes. This can reduce training throughput for small models where the optimizer step is fast, because the CPU cannot overlap computation with kernel execution.
