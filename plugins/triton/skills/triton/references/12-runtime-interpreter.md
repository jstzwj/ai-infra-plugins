# Chapter 12: Runtime Interpreter (`TRITON_INTERPRET=1`)

The Triton interpreter allows running kernels on CPU without a GPU. Set `TRITON_INTERPRET=1` to enable.

## Usage

```bash
TRITON_INTERPRET=1 python your_script.py
```

Or programmatically:
```python
import os
os.environ['TRITON_INTERPRET'] = '1'
```

## Key Features

1. **CPU Execution**: Kernels run on CPU using NumPy
2. **Python Breakpoints**: You can insert `breakpoint()` in kernel code
3. **Debugging**: Full Python stack traces for kernel errors
4. **No GPU Required**: Run tests on machines without GPUs

## Architecture

### InterpretedFunction
When `TRITON_INTERPRET=1`, `@triton.jit` creates `InterpretedFunction` instead of `JITFunction`:

```python
class InterpretedFunction(KernelInterface):
    def run(self, *args, **kwargs):
        # Rewrites kernel function for interpreter
        # Executes on CPU using NumPy arrays
```

### GridExecutor
Executes the kernel grid:

```python
class GridExecutor:
    def __init__(self, fn, grid, args, kwargs, ...):
        # Sets up execution context

    def __call__(self):
        # Iterates over program grid
        # Calls kernel for each program ID
```

### InterpreterBuilder
Provides type constructors and operations for the interpreter:

```python
class InterpreterBuilder:
    def __init__(self, arch, options, codegen_fns):
        self.arch = arch
        self.options = options
```

### TensorHandle
CPU tensor representation:

```python
@dataclass
class TensorHandle:
    data: np.ndarray  # NumPy array
    dtype: tl.dtype    # Triton dtype
    attr: dict         # Attributes
```

## Supported Operations

The interpreter supports:
- All arithmetic operations (add, sub, mul, div, etc.)
- Memory operations (load, store)
- Reductions (sum, max, min, argmax, argmin)
- Scans (cumsum, cumprod, associative_scan)
- Sorting (sort, topk)
- Random number generation
- Atomic operations (simulated)
- Control flow (if/else, for, while)
- Type casting

## Limitations

1. **Performance**: Much slower than GPU execution
2. **Shared Memory**: Simulated, not real shared memory
3. **Warp-level Operations**: Not accurately modeled
4. **Memory Layout**: No actual GPU memory hierarchy
5. **Some GPU-specific features**: May not be supported

## Debugging Tips

```python
@triton.jit
def my_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # You can use Python breakpoint!
    # breakpoint()  # Uncomment to debug

    x = tl.load(x_ptr + offsets, mask=mask)

    # Print values for debugging
    # tl.device_print("x =", x)

    result = x * 2
    tl.store(x_ptr + offsets, result, mask=mask)

# Run with interpreter
# TRITON_INTERPRET=1 python script.py
```
