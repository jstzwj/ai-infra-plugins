# Chapter 30: Plugin System

Triton supports out-of-tree plugins for adding custom dialects and passes.

## Plugin Architecture

```
Triton Core
├── Built-in Dialects (Triton, TritonGPU, Gluon)
├── Built-in Backends (NVIDIA, AMD)
└── Plugin Interface
    ├── Custom Dialects
    ├── Custom Passes
    └── Custom Backends
```

## Out-of-Tree Plugin Backends

### Setup

```bash
export TRITON_PLUGIN_DIRS="/path/to/my_plugin"
pip install -e /path/to/triton
```

### Required Files

```
my_plugin/
├── backend/
│   ├── name.conf           # Contains backend name, e.g., "my_backend"
│   ├── compiler.py         # Must define a concrete BaseBackend subclass
│   └── driver.py           # Must define a concrete DriverBase subclass
├── language/               # Optional: language extensions
│   └── my_ext/
│       └── __init__.py
└── tools/                  # Optional: CLI tools
    └── my_tool/
        └── __init__.py
```

### Backend Implementation

```python
# backend/compiler.py
from triton.backends import BaseBackend, GPUTarget

class MyBackend(BaseBackend):
    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "my_hardware"

    def hash(self) -> str:
        return "my_backend_hash"

    def parse_options(self, options: dict):
        return MyOptions(**options)

    def add_stages(self, stages, options):
        stages["ttir"] = (lambda src: src, True)
        stages["ttgir"] = (self._to_ttgir, True)
        stages["binary"] = (self._to_binary, False)

    def load_dialects(self, context):
        pass

    def get_module_map(self):
        return {}
```

```python
# backend/driver.py
from triton.backends.compiler import DriverBase

class MyDriver(DriverBase):
    @classmethod
    def is_active(cls) -> bool:
        return True  # Check if hardware is available

    def get_current_target(self) -> GPUTarget:
        return GPUTarget("my_hardware", "v1", 32)

    def get_active_torch_device(self):
        return "my_device"

    def get_benchmarker(self):
        return my_benchmarker

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {"fp32": "float", "fp64": "double", ...}
        return mapping[ty]
```

## Custom MLIR Passes

### Using the Inspection Hook

```python
import triton

def my_custom_pass(module):
    # Apply custom MLIR transformations
    pass

def inspect_stages(self, stages, options, language, capability):
    # Add custom pass after TTGIR stage
    original_ttgir = stages["ttgir"]

    def custom_ttgir(src):
        result = original_ttgir[0](src)
        my_custom_pass(result)
        return result

    stages["ttgir"] = (custom_ttgir, original_ttgir[1])
    return stages

triton.knobs.runtime.add_stages_inspection_hook = inspect_stages
```

### Plugin Dialect Example

See `examples/plugins/` for a complete example of creating a custom MLIR dialect:

```
examples/plugins/
├── CMakeLists.txt
├── README.md
├── TritonPlugin.cpp
├── Passes.td
└── DialectPlugins/
    └── DialectPlugin/
        ├── include/DialectPlugin/
        │   ├── Dialect.h
        │   └── Ops.h
        └── lib/DialectPlugin/
            ├── Dialect.cpp
            └── Ops.cpp
```

### Plugin Registration

```cpp
// TritonPlugin.cpp
#include "mlir/IR/DialectRegistry.h"

// Register plugin dialect
void registerPluginDialects(mlir::DialectRegistry &registry) {
    registry.insert<plugin::PluginDialect>();
}
```

## Runtime Hooks

### Launch Hooks

```python
def my_launch_hook(metadata):
    """Called before kernel launch."""
    print(f"Launching kernel: {metadata}")

def my_exit_hook(metadata):
    """Called after kernel launch."""
    print(f"Kernel completed: {metadata}")

triton.knobs.runtime.launch_enter_hook.add(my_launch_hook)
triton.knobs.runtime.launch_exit_hook.add(my_exit_hook)
```

### Kernel Load Hooks

```python
def on_kernel_load(module, function, name, metadata, hash):
    """Called when kernel binary is loaded."""
    print(f"Loaded kernel: {name}")

triton.knobs.runtime.kernel_load_end_hook.add(on_kernel_load)
```

### JIT Hooks

```python
def on_jit_cache(*, key, repr, fn, compile, is_manual_warmup, already_compiled):
    """Called when JIT function is compiled or loaded from cache."""
    if not already_compiled:
        print(f"Compiling: {repr}")

triton.knobs.runtime.jit_cache_hook = on_jit_cache
```

## Autotune Listeners

```python
def on_autotune(*, fn, key, best_config, configs_timings, duration, cache_hit):
    """Called after autotuning completes."""
    print(f"Best config for {fn}: {best_config}")
    for config, times in configs_timings.items():
        print(f"  {config}: median={statistics.median(times):.3f}ms")

triton.knobs.autotuning.listener = on_autotune
```
