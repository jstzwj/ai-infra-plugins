# TorchInductor Backend

TorchInductor is the default backend for `torch.compile`, generating optimized Triton GPU kernels and C++ CPU kernels from FX graphs.

---

## Architecture

```
Python Model
    |
    v
TorchDynamo (bytecode analysis, graph capture)
    |
    v
FX Graph (IR with ATen operators)
    |
    v
AOTAutograd (functionalize forward+backward, deduplicate)
    |
    v
Inductor Backend:
    1. Decomposition (lower to core ATen ops)
    2. Scheduling (fuse ops, determine kernel boundaries)
    3. Memory Planning (buffer allocation, reuse)
    4. Code Generation (Triton GPU / C++ CPU)
    |
    v
Optimized Executable
```

---

## Triton GPU Codegen

Inductor generates Triton kernels for GPU operations.

### Kernel Fusion

Inductor automatically fuses pointwise operations into single kernels:

```python
# These are fused into one Triton kernel
x = a + b          # elementwise add
y = x * c          # elementwise mul
z = torch.relu(y)  # elementwise relu

# Reductions create separate kernels
s = z.sum(dim=1)
```

Fusion rules:
- **Pointwise-pointwise**: Always fused
- **Reduction-pointwise**: Often fused (epilogue fusion)
- **Pointwise-reduction**: Separate kernels
- **Matmul-pointwise**: Epilogue fused into matmul kernel

### Example Generated Triton Kernel

```python
import triton
import triton.language as tl

@triton.jit
def fused_relu_add_kernel(X_ptr, Y_ptr, OUT_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.load(Y_ptr + offsets, mask=mask)
    out = tl.where(x > 0, x, 0) + y  # relu(x) + y
    tl.store(OUT_ptr + offsets, out, mask=mask)
```

### Autotuning

Inductor autotunes Triton kernel configurations (tile sizes, number of warps, pipeline stages).

```python
import torch._inductor.config as config

config.max_autotune = True          # enable extensive autotuning
config.max_autotune_gemm = True     # autotune GEMM kernels
config.max_autotune_pointwise = False  # skip pointwise autotuning
```

Tuning parameters include: block sizes (BLOCK_M, BLOCK_N, BLOCK_K), number of warps, number of stages (pipeline depth), unroll factors.

---

## C++ CPU Codegen

For CPU, Inductor generates C++ code compiled with GCC/Clang, using OpenMP and SIMD.

```python
import torch._inductor.config as config
config.cpp_wrapper = True     # use C++ codegen for CPU
```

Example generated CPU code:

```cpp
#pragma omp parallel for
for (long i0 = 0; i0 < N; i0++) {
    auto x = X[i0];
    auto y = Y[i0];
    OUT[i0] = (x > 0 ? x : 0) + y;  // relu(x) + y
}
```

---

## torch._inductor.config

```python
import torch._inductor.config as config

# General
config.triton.cudagraphs = True         # use CUDA graphs for kernel launch
config.triton.unique_kernel_names = True # unique names in profiler
config.size_asserts = True              # runtime shape assertions
config.fx_graph_cache = True            # cache FX graphs across runs

# Performance
config.triton.max_kernel_autotune = 64  # max autotune attempts
config.aggressive_fusion = False        # more aggressive fusion heuristics

# Memory
config.triton.enable_persistent = True  # persistent kernel optimizations
config.memory_planning = True           # enable memory planning

# Debug
config.debug = False                    # enable debug output
config.trace.graph_diagram = False      # generate graph SVG
config.save_args = False                # save args for replay

# Fallbacks
config.fallback_random = True           # use ATen for random ops
config.triton.enable_fallback = True    # allow fallback to ATen

# CPU
config.cpp_wrapper = False              # use C++ codegen for CPU ops
config.cpp.enable_kernel_profile = True
```

---

## Scheduling

The scheduler determines kernel boundaries and operation ordering.

### Scheduling Decisions

1. **Fusion groups**: Which ops go into a single kernel
2. **Computation order**: Which group runs first
3. **Buffer lifetime**: When to allocate and free buffers

### Fusion Types
- **Vertical fusion**: Sequential ops (matmul + relu into single kernel)
- **Horizontal fusion**: Independent ops sharing input (mean and variance together)

```bash
# View scheduling decisions
TORCH_LOGS="scheduler" python train.py
```

---

## Memory Planning

Inductor plans buffer allocation to minimize memory usage.

- **Buffer reuse**: Reuses memory for tensors with non-overlapping lifetimes
- **In-place operations**: Converts to in-place where safe
- **Alias analysis**: Tracks when tensors share underlying storage

---

## Code Cache

Compiled kernels are cached to avoid recompilation.

```bash
# Default cache location
/tmp/torchinductor_<username>/

# Custom cache directory
TORCHINDUCTOR_CACHE_DIR=/path/to/cache

# Disable cache for debugging
TORCHINDUCTOR_DISABLE_CACHE=1
```

---

## Debugging with TORCH_LOGS

```bash
TORCH_LOGS="+inductor"        # Inductor codegen logs
TORCH_LOGS="+inductor_detail" # detailed Inductor IR
TORCH_LOGS="+triton"          # generated Triton source code
TORCH_LOGS="+kernel"          # kernel launch info
TORCH_LOGS="+scheduler"       # scheduling decisions
TORCH_LOGS="+buffer"          # memory planning details
TORCH_LOGS="+fusion"          # fusion decisions
TORCH_LOGS="+aot"             # AOT autograd logs
TORCH_LOGS="+dynamo"          # Dynamo tracing
TORCH_LOGS="dynamic"          # dynamic shape tracing
TORCH_LOGS="perf_hardware"    # hardware performance counters

# Multiple logs combined
TORCH_LOGS="+inductor,+triton,+scheduler"

# Python API
import torch._inductor.config
torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True
```

---

## Decomposition

Complex operations are decomposed into simpler primitives before codegen.

```python
# Example: batch_norm decomposes into:
# mean = x.mean(dim)
# var = x.var(dim)
# x_norm = (x - mean) / sqrt(var + eps)
# output = weight * x_norm + bias
```

---

## Custom Lowering

Register custom Inductor lowering for specific ATen operations.

```python
from torch._inductor.lowering import register_lowering, make_boxed_func

@register_lowering(torch.ops.my_custom.op)
def my_custom_lowering(inputs, ...):
    # Return an Inductor IR expression
    return ...

# Or delegate to existing implementation
register_lowering(torch.ops.my_custom.op)(make_boxed_func(my_impl))
```

---

## Performance Tips

1. **Use `mode="max-autotune"`** for best throughput (longer compile)
2. **Use `mode="reduce-overhead"`** for latency-sensitive workloads (CUDA graphs)
3. **Enable CUDA graphs**: `config.triton.cudagraphs = True`
4. **Warm up** with representative inputs before timing
5. **Avoid graph breaks**: Use `torch.compiler.disable` / `allow_in_graph`
6. **Use `fullgraph=True`** for whole-program optimization
7. **Pin GPU memory**: Use `pin_memory=True` on DataLoader
8. **Use `dynamic=True`** for variable-size inputs (may reduce optimization)

---

## Compilation Example

```python
import torch
import torch._inductor.config

torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.max_autotune = True

model = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
).cuda()

model = torch.compile(model, mode="max-autotune")

# Warmup (triggers compilation)
with torch.no_grad():
    model(torch.randn(32, 784, device="cuda"))

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    model(torch.randn(32, 784, device="cuda"))
end.record()
torch.cuda.synchronize()
print(f"Avg: {start.elapsed_time(end) / 100:.2f} ms")
```

---

## Inductor + FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# use_orig_params=True required for FSDP + compile compatibility
model = FSDP(model, use_orig_params=True, ...)
model = torch.compile(model)
```
