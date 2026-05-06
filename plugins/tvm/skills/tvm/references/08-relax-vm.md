# 08 — Relax Virtual Machine

## Overview

The Relax Virtual Machine (VM) is the runtime component that executes compiled Relax programs. Relax defines *what* to compute (the graph-level IR); the VM handles *how* to run it.

### Architecture

- **Register-based interpreter** with only **4 opcodes**: `Call`, `Ret`, `Goto`, `If`
- The VM performs **no mathematical computation** itself — it orchestrates control flow while dispatching actual work to compiled TIR kernels or external libraries
- Built on the `runtime.Module` and `PackedFunc` interface

---

## Instruction Set

### Call
Invokes a function with arguments and stores the result in a virtual register.
- Used for TIR kernel calls, external library dispatch (cuBLAS, cuDNN, etc.)
- Arguments can be registers, constants, or immediate values

### Ret
Returns a value from the current function.
- Pops the call stack and returns the register value to caller

### Goto
Unconditional jump to a target instruction.
- Used for straight-line control flow

### If
Conditional branch based on a register value.
- If register is true/non-zero, jump to one target; otherwise jump to another
- Used for control flow in models (e.g., conditional execution)

---

## Compilation Pipeline

```
IRModule → relax.build() → VMExecutable → runtime.Module
```

### Building

```python
from tvm import relax

# Build for CPU
exec_cpu = relax.build(mod, target="llvm")

# Build for GPU
exec_gpu = relax.build(mod, target="nvidia/nvidia-a100")

# Build with multiple targets
exec = relax.build(mod, target=tvm.target.Target("cuda"))
```

---

## Python Interface

### Creating a VM

```python
import tvm
from tvm import relax

# Method 1: Build and create
exec_mod = relax.build(mod, target="llvm")
vm = relax.VirtualMachine(exec_mod, tvm.cpu())

# Method 2: Load from disk
exec_mod = tvm.runtime.load_module("model.so")
vm = relax.VirtualMachine(exec_mod, tvm.cuda(0))
```

### Executing Functions

```python
# Call function by name
result = vm["main"](input_data)

# Get function info
func_names = vm.function_names()

# Multiple outputs
outputs = vm["main"](input1, input2)
# outputs is a tuple if function returns multiple values
```

### Save and Load

```python
# Save compiled executable
exec_mod.export_library("model.so")

# Load back
loaded = tvm.runtime.load_module("model.so")
vm = relax.VirtualMachine(loaded, tvm.cuda(0))
```

---

## VMExecutable Structure

### Function Table
Maps function names to instruction sequences. Each function has:
- Name (string)
- Register count (number of virtual registers needed)
- Instruction list (sequence of Call/Ret/Goto/If)
- Parameter indices

### Constant Pool
Stores constant tensors used by functions:
- Model weights
- Constant parameters (shape, dtype info)
- Immediate values

### Global Section
Contains function metadata:
- Function signatures
- Parameter information
- Return type descriptions

### Code Section
Serialized instructions for each function. The VM interpreter reads and executes these instructions.

---

## Memory Management

### Static Memory Planning
The `StaticPlanBlockMemory` pass plans memory allocation at compile time:
- Analyzes tensor lifetimes
- Plans memory reuse (in-place operations)
- Generates allocation instructions in the VM

### Device Memory
- All tensors allocated on the target device
- Memory management handled by device API
- Pool allocator for frequent small allocations

### Register Allocation
- Virtual registers assigned at compile time
- Each register holds one tensor value
- No runtime register allocation overhead

---

## Integration with runtime.Module

### Module Composition
The VM executable is wrapped in a `runtime.Module`:
```
VMExecutable (runtime.Module)
├── Host functions (parameter setup, kernel launch)
├── CUDA functions (kernel code)
├── External functions (cuBLAS, cuDNN, etc.)
└── VM bytecode
```

### Multi-module Support
```python
# The VM can load additional modules
vm = relax.VirtualMachine(exec, tvm.cuda(0))
# exec may contain: LLVM host code + CUDA device code + cuBLAS dispatch
```

---

## Advanced Features

### Profiling
```python
# Time evaluation
from tvm.runtime import profiling

# Profile function execution
result = profiling.profile_function(vm["main"], tvm.cuda(0), input_data)
print(result)
```

### Debug Mode
```python
# Enable debug mode for detailed logging
import os
os.environ["TVM_LOG_DEBUG"] = "1"
```

### Custom Function Registration
```python
# Register custom runtime function
@tvm.register_func("my_custom_func")
def my_func(x):
    return x * 2

# Can be called from compiled VM via call_packed
```

### VM with Distributed Execution
The VM integrates with Disco for multi-device execution:
```python
from tvm.runtime import disco

session = disco.ThreadedSession(num_workers=4)
# Load VM module into distributed session
dmod = session.import_module(exec)
result = dmod["main"](input_data)
```

---

## Build Configuration

### Target-Specific Build
```python
# CPU with specific architecture
exec = relax.build(mod, target=tvm.target.Target("llvm -mcpu=skylake-avx512"))

# GPU with specific architecture
exec = relax.build(mod, target=tvm.target.Target("cuda -arch=sm_80"))

# Multi-target
exec = relax.build(mod, target=tvm.target.Target("cuda"))
```

### PassContext Configuration
```python
with tvm.transform.PassContext(opt_level=3, config={
    "relax.backend.use_cublas": True,
}):
    exec = relax.build(mod, target="cuda")
```

---

## VM vs Graph Executor

| Feature | Relax VM | Graph Executor (Legacy) |
|---------|----------|------------------------|
| Control flow | Full support (if/else, loops) | Limited |
| Dynamic shapes | Native support | Limited |
| Instruction set | 4 opcodes (Call/Ret/Goto/If) | Graph-based execution |
| External backends | Full BYOC support | Limited |
| Training | Supported via Gradient pass | Not supported |
| Distributed | Disco integration | Not supported |
| Recommended | Yes (current) | Legacy only |

---

## Complete Example

```python
import tvm
from tvm import relax
from tvm.relax.frontend import nn
import numpy as np

# 1. Define model
class SimpleModel(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)

# 2. Export to IRModule
model = SimpleModel()
mod, params = model.export_tvm(
    spec={"x": {"shape": [1, 784], "dtype": "float32"}}
)

# 3. Optimize
mod = relax.get_pipeline("zero")(mod)

# 4. Build
exec = relax.build(mod, target="llvm")

# 5. Save
exec.export_library("simple_model.so")

# 6. Load and run
loaded = tvm.runtime.load_module("simple_model.so")
vm = relax.VirtualMachine(loaded, tvm.cpu())

# 7. Prepare input
data = tvm.nd.array(np.random.randn(1, 784).astype("float32"))

# 8. Execute
result = vm["main"](data)
print(f"Output shape: {result.shape}, dtype: {result.dtype}")
print(f"Output: {result.numpy()}")
```
