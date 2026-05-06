# 02 — Getting Started

## Installation

### Quick Install
```bash
# Clone repository
git clone https://github.com/apache/tvm.git
cd tvm
git submodule update --init --recursive

# Build
mkdir build && cd build
cp ../cmake/config.cmake .
cmake .. && cmake --build . --parallel $(nproc)

# Install Python package
cd ../python && pip install -e .
```

See [Installation Guide](30-installation-guide.md) for detailed instructions.

---

## Quick Start Tutorial

### Define a Model Using Relax NN Frontend

```python
import tvm
from tvm import relax
from tvm.relax.frontend import nn

class MLPModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: nn.Tensor):
        x = self.fc1(x)
        x = nn.relu(x)
        return self.fc2(x)

# Export to IRModule
model = MLPModel()
mod, params = model.export_tvm(
    spec={"x": {"shape": [1, 784], "dtype": "float32"}}
)
```

### Apply Optimization Pipeline

```python
# Apply the "zero" optimization pipeline
mod = relax.get_pipeline("zero")(mod)
```

### Build and Deploy

```python
# Build for CPU
exec_cpu = relax.build(mod, target="llvm")
vm_cpu = relax.VirtualMachine(exec_cpu, tvm.cpu())

# Build for GPU
exec_gpu = relax.build(mod, target="nvidia/nvidia-a100")
vm_gpu = relax.VirtualMachine(exec_gpu, tvm.cuda(0))

# Run inference
import numpy as np
data = tvm.nd.array(np.random.randn(1, 784).astype("float32"), tvm.cpu())
result = vm_cpu["main"](data)
print(result.numpy())
```

### Export and Load

```python
# Export compiled artifact
exec_cpu.export_library("mlp_cpu.so")

# Load and run
loaded = tvm.runtime.load_module("mlp_cpu.so")
vm = relax.VirtualMachine(loaded, tvm.cpu())
result = vm["main"](data)
```

---

## IRModule Creation Methods

### Method 1: Import from PyTorch

```python
import torch
import torch.nn as nn
from tvm.relax.frontend.torch import from_exported_program

# Define PyTorch model
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Export with torch.export
model = TorchMLP().eval()
example_input = (torch.randn(1, 784),)
exported_program = torch.export.export(model, example_input)

# Convert to TVM IRModule
mod = from_exported_program(exported_program)
```

### Method 2: Import from ONNX

```python
import onnx
from tvm.relax.frontend.onnx import from_onnx

onnx_model = onnx.load("model.onnx")
mod = from_onnx(onnx_model, shape_info={"input": (1, 3, 224, 224)})
```

### Method 3: Import from TFLite

```python
from tvm.relax.frontend.tflite import from_tflite

mod = from_tflite(
    tflite_model,
    shape_dict={"input": (1, 224, 224, 3)},
    dtype_dict={"input": "float32"},
)
```

### Method 4: Relax NNModule (Direct Construction)

```python
from tvm.relax.frontend import nn

class MyModel(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = nn.relu(self.conv(x))
        x = nn.flatten(x, start_dim=1)
        return self.fc(x)

mod, params = MyModel().export_tvm(
    spec={"x": {"shape": [1, 3, 28, 28], "dtype": "float32"}}
)
```

### Method 5: TVMScript (Direct IR Definition)

```python
from tvm.script import ir as I, tirx as T, relax as R

@I.ir_module
class MyModule:
    @T.prim_func
    def my_kernel(A: T.Buffer((128,), "float32"),
                  B: T.Buffer((128,), "float32")):
        for i in range(128):
            with T.sblock("B"):
                vi = T.axis.spatial(128, i)
                B[vi] = A[vi] * T.float32(2.0)

    @R.function
    def main(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
        cls = MyModule
        with R.dataflow():
            lv = R.call_tir(cls.my_kernel, (x,),
                           out_sinfo=R.Tensor((128,), "float32"))
            R.output(lv)
        return lv
```

---

## Basic Workflow

### Step 1: Create or Import Model → IRModule

```python
mod = from_exported_program(exported_program)
# or mod = from_onnx(onnx_model, shape_info)
# or mod, params = model.export_tvm(spec)
```

### Step 2: Apply Transformations

```python
# Option A: Use built-in pipeline
mod = relax.get_pipeline("zero")(mod)

# Option B: Apply individual passes
mod = relax.transform.LegalizeOps()(mod)
mod = relax.transform.FuseOps()(mod)
mod = relax.transform.DeadCodeElimination()(mod)

# Option C: Custom pipeline
from tvm import transform
pipeline = transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.FuseOpsByPattern([...]),
    relax.transform.FuseTIR(),
    relax.transform.DeadCodeElimination(),
])
mod = pipeline(mod)
```

### Step 3: Build

```python
# CPU
exec = relax.build(mod, target="llvm")

# NVIDIA GPU
exec = relax.build(mod, target="nvidia/nvidia-a100")

# AMD GPU
exec = relax.build(mod, target="rocm")
```

### Step 4: Deploy

```python
# Python deployment
vm = relax.VirtualMachine(exec, tvm.cuda(0))
result = vm["main"](data)

# Export for non-Python deployment
exec.export_library("model.so")
```

---

## Universal Deployment

TVM runtime can work in non-Python environments:

### Python
```python
import tvm
mod = tvm.runtime.load_module("model.so")
vm = tvm.relax.VirtualMachine(mod, tvm.cuda(0))
result = vm["main"](data)
```

### C++
```cpp
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/vm.h>

tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("model.so");
tvm::runtime::vm::VirtualMachine vm;
vm.LoadExecutable(mod);
// Execute...
```

### Java / Rust / Go / JavaScript
TVM provides C API as foundation, with language-specific bindings available.

### DLPack Integration
Zero-copy data exchange with other frameworks:
```python
import torch, tvm

# TVM → PyTorch (zero copy)
tvm_array = tvm.nd.empty((128, 128), device=tvm.cuda(0))
torch_tensor = torch.from_dlpack(tvm_array)

# PyTorch → TVM (zero copy)
torch_tensor = torch.randn(128, 128, device="cuda")
tvm_array = tvm.nd.from_dlpack(torch_tensor)
```

---

## Common Pipelines

### "zero" Pipeline (Default)
Fast compilation with good performance. Suitable for most use cases.
```python
mod = relax.get_pipeline("zero")(mod)
```

### "static_shape_tuning" Pipeline
Auto-tuning with MetaSchedule for best performance. Slower compilation.
```python
mod = relax.get_pipeline("static_shape_tuning")(mod)
```

### Custom Pipeline
```python
def my_pipeline():
    return tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.FuseOps(),
        relax.transform.FuseOpsByPattern(my_patterns),
        relax.transform.FuseTIR(),
        relax.transform.DeadCodeElimination(),
        relax.transform.SimplifyExpr(),
        relax.transform.StaticPlanBlockMemory(),
    ])

mod = my_pipeline()(mod)
```

---

## Verifying Correctness

After importing and optimizing, verify numerical correctness:

```python
import numpy as np

# Get TVM output
vm = relax.VirtualMachine(exec, tvm.cpu())
tvm_output = vm["main"](data).numpy()

# Get reference output (e.g., from PyTorch)
with torch.no_grad():
    ref_output = torch_model(torch_data).numpy()

# Compare
np.testing.assert_allclose(tvm_output, ref_output, rtol=1e-5, atol=1e-5)
print("Verification passed!")
```

---

## Next Steps

- [Relax IR](04-relax-ir.md) — understand the high-level graph IR
- [TensorIR Abstraction](10-tensor-ir-abstraction.md) — understand low-level tensor programs
- [TVMScript](18-tvmscript.md) — learn the TVM DSL syntax
- [Relax Transformations](05-relax-transformations.md) — explore optimization passes
- [Schedule Primitives](14-schedule-primitives.md) — learn how to optimize TIR programs
- [Target System](19-target-system.md) — configure compilation targets
