# TorchGen: Code Generation System

## Overview

TorchGen is PyTorch's code generation system that automatically produces C++ boilerplate, Python bindings, dispatch registrations, and other generated code from operator definitions. It reads `native_functions.yaml` (and related YAML files) and generates the hundreds of files that make up PyTorch's operator infrastructure.

**Source location**: `torchgen/`

Without TorchGen, adding a new operator would require manually editing dozens of files across C++ and Python. With TorchGen, adding a single entry to `native_functions.yaml` generates all necessary bindings automatically.

---

## native_functions.yaml

### Location

```
aten/src/ATen/native/native_functions.yaml
```

### Schema Format

Each entry defines an operator with its schema, dispatch behavior, and metadata.

```yaml
# Basic format
- func: op_name(Tensor self, Tensor other) -> Tensor
  dispatch:
    CPU: kernel_name_cpu
    CUDA: kernel_name_cuda

# Complete example with all fields
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  structured: True
  structured_delegate: add.out
  dispatch:
    CPU: add::kernel
    CUDA: add::kernel
    MPS: add::kernel
    SparseCPU: add::sparse_cpu
    SparseCUDA: add::sparse_cuda
    SparseCsrCPU: add::sparse_csr
    SparseCsrCUDA: add::sparse_csr
  tags: [pointwise, core]
  variants: function, method
  manual_kernel_registration: False
  manual_cpp_binding: False
  python_module: nn
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `func` | string | Function schema: name, arguments, return types |
| `dispatch` | map | Backend key to kernel function mapping |
| `structured` | bool | Use structured kernel pattern (separate meta/impl) |
| `structured_delegate` | string | Delegate to another structured op |
| `device_check` | string | NoCheck, ExactSame, etc. |
| `tags` | list | Operation category tags |
| `variants` | string | "function", "method", or both |
| `manual_kernel_registration` | bool | Skip auto kernel registration |
| `manual_cpp_binding` | bool | Skip auto C++ binding generation |
| `python_module` | string | Python module to place the binding |
| `supports_autograd` | bool | Whether autograd is supported |

### Argument Types in Schema

```yaml
# Tensor arguments
Tensor self                    # required tensor
Tensor? bias                   # optional tensor
Tensor[] tensors               # list of tensors
Tensor(a!) self                # mutable tensor (alias annotation)
Tensor(a) self                 # aliased tensor

# Scalar arguments
Scalar alpha                   # scalar (int or float)
ScalarType dtype               # data type
ScalarType? dtype              # optional data type

# Numeric arguments
int dim                        # integer
int[] dims                     # list of integers
int dim=0                      # with default
float momentum=0.1             # float with default
bool keepdim=False             # boolean

# Other types
str mode                       # string
Layout layout                  # tensor layout
Device device                  # device
Generator? generator           # RNG generator
MemoryFormat? memory_format    # memory format

# Keyword-only arguments (after *)
Tensor self, *, int dim=0
```

### Return Types

```yaml
# Single return
-> Tensor

# Multiple returns
-> (Tensor values, Tensor indices)

# No return
-> ()

# With alias annotations
-> Tensor(a!)
-> (Tensor(a!), Tensor)
```

### Dispatch Annotations

```yaml
dispatch:
  CPU: add_cpu                    # direct function name
  CUDA: add_cuda                  # direct function name
  CPU: add::kernel                # namespaced function
  CompositeExplicitAutograd: add_composite  # composite kernel
```

### Tags

```yaml
tags:
  - core           # core operation
  - pointwise      # element-wise operation
  - view           # view operation (no copy)
  - inplace        # in-place operation
  - reduction      # reduction operation
  - math           # mathematical operation
```

---

## Generated Files

TorchGen produces files across several categories:

### C++ API Generation

| File | Description |
|------|-------------|
| `build/aten/src/ATen/TensorBody.h` | Tensor method bodies (all at::Tensor methods) |
| `build/aten/src/ATen/Functions.h` | Namespace-level function declarations |
| `build/aten/src/ATen/NativeFunctions.h` | Native function declarations |
| `build/aten/src/ATen/RegistrationDeclarations.h` | Operator registration declarations |
| `build/aten/src/ATen/CompositeViewFunctions.h` | Composite view operation implementations |
| `build/aten/src/ATen/DispatchPointerFunction.h` | Dispatch function pointers |

### Python Bindings

| File | Description |
|------|-------------|
| `torch/_C/_VariableFunctions.pyi` | Python type stubs for torch functions |
| `torch/_C/_VariableFunctions.py` | Python wrapper implementations |
| `torch/_C/_torch_docs.py` | Documentation strings |
| `aten/src/ATen/python/` | Python binding source files |

### Dispatcher Registration

| File | Description |
|------|-------------|
| `build/aten/src/ATen/RegisterBackendSelect.cpp` | Backend selection registrations |
| `build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp` | Composite kernel registrations |
| `build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp` | Implicit autograd registrations |
| `build/aten/src/ATen/RegisterCPU.cpp` | CPU kernel registrations |
| `build/aten/src/ATen/RegisterCUDA.cpp` | CUDA kernel registrations |
| `build/aten/src/ATen/RegisterMPS.cpp` | MPS kernel registrations |
| `build/aten/src/ATen/RegisterMath.cpp` | Math kernel registrations |
| `build/aten/src/ATen/RegisterMeta.cpp` | Meta kernel registrations |
| `build/aten/src/ATen/RegisterAutograd.cpp` | Autograd registrations |
| `build/aten/src/ATen/RegisterFunctionalization.cpp` | Functionalization registrations |

### Shape Function Generation

| File | Description |
|------|-------------|
| `torch/_prims/__init__.py` | Prim operations |
| `aten/src/ATen/native/` | Shape function implementations |

---

## torchgen/gen.py: Main Generation Entry Point

The main entry point for code generation.

### Command Line Usage

```bash
# Generate all files
python torchgen/gen.py \
    --source-path aten/src/ATen \
    --install-dir build/aten/src/ATen \
    --output-dependencies build/aten/src/ATen/dependencies.txt

# Generate specific categories
python torchgen/gen.py \
    --source-path aten/src/ATen \
    --install-dir build/aten/src/ATen \
    --gen-per-dispatch-key-registrations \
    --gen-unboxing \
    --gen-composite-implicit-autograd
```

### Key Generation Functions

```python
# torchgen/gen.py

def gen(
    source_path: str,       # path to aten/src/ATen
    install_dir: str,       # output directory
    operator_gen_dir: str,  # operator generation directory
    # ... other options
) -> None:
    """Main generation entry point."""

    # 1. Parse native_functions.yaml
    native_functions = parse_native_functions_yaml()

    # 2. Parse derivatives.yaml (autograd definitions)
    derivatives = parse_derivatives_yaml()

    # 3. Generate C++ headers
    gen_tensor_body(native_functions)
    gen_functions_h(native_functions)
    gen_native_functions_h(native_functions)
    gen_registration_declarations(native_functions)

    # 4. Generate registration files
    for key in DISPATCH_KEYS:
        gen_register_backend(key, native_functions)

    # 5. Generate Python bindings
    gen_python_bindings(native_functions)

    # 6. Generate shape functions
    gen_shape_functions(native_functions)
```

---

## Generator Categories

### 1. Python Bindings Generation

Generates Python wrapper functions that call into the C++ dispatcher.

```python
# torchgen/dest/gen_python_functions.py

# For each operator in native_functions.yaml:
# 1. Generate Python function signature
# 2. Generate argument parsing code
# 3. Generate C++ dispatch call
# 4. Generate return value handling

# Example input (native_functions.yaml):
# func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

# Example output (Python binding):
def add(input, other, alpha=1, out=None):
    if out is not None:
        return torch._C._add_impl(input, other, alpha=alpha, out=out)
    return torch._C._add_impl(input, other, alpha=alpha)
```

### 2. C++ API Generation

Generates the C++ Tensor methods and namespace functions.

```python
# torchgen/dest/gen_tensor_method.py

# For each operator that has "method" variant:
# Generate a method on at::Tensor

# Example output (TensorBody.h):
Tensor Tensor::add(const Tensor& other, Scalar alpha) const {
    return at::add(*this, other, alpha);
}
```

### 3. Dispatcher Registration Generation

Generates registration code that maps dispatch keys to kernels.

```python
# torchgen/dest/gen_dispatcher_registrations.py

# For each dispatch key, generate:
# TORCH_LIBRARY_IMPL(aten, CPU, m) {
#     m.impl("add.Tensor", TORCH_FN(add_cpu_kernel));
#     m.impl("mm", TORCH_FN(mm_cpu_kernel));
#     // ... hundreds more
# }
```

### 4. ATen Native Function Generation

Generates declarations for native function implementations.

```python
# torchgen/dest/gen_aten_lib.py

# Generates:
# - Native function declarations
# - Structured kernel base classes
# - Meta function declarations
```

### 5. Decomposition Generation

Generates decomposition rules that express complex operators in terms of simpler ones.

```python
# torchgen/dest/gen_decompositions.py

# Used by torch.compile to decompose operators
# into primitives that the compiler can optimize
```

### 6. Shape Function Generation

Generates functions that compute output shapes without computing data.

```python
# torchgen/dest/gen_shape_functions.py

# For structured operators, the meta kernel computes output shape:
# Example for add.Tensor:
# Meta function returns tensor with same shape and dtype as input
```

---

## torchgen/api/

The `torchgen/api/` directory contains API definitions used by generators to translate between different representations.

### Key Modules

```
torchgen/api/
  types.py           # Type representations (BaseType, ListType, OptionalType)
  native.py          # Native function analysis
  dispatcher.py      # Dispatcher-specific API
  cpp.py             # C++ code generation utilities
  python.py          # Python code generation utilities
  meta.py            # Meta function analysis
  structured.py      # Structured kernel analysis
  autograd.py        # Autograd-specific generation
  translate.py       # Translate between type representations
  unwrapper.py       # Argument unwrapping for dispatch
```

### types.py: Type Representations

```python
# torchgen/api/types.py

class Type:
    """Base type representation."""
    pass

class BaseType(Type):
    """Primitive type: Tensor, int, float, bool, str, Scalar, etc."""
    def cpp_type(self) -> str: ...
    def python_type(self) -> str: ...

class OptionalType(Type):
    """Optional[T] type."""
    def __init__(self, elem: Type): ...

class ListType(Type):
    """List[T] type."""
    def __init__(self, elem: Type, size: Optional[int] = None): ...

class TupleType(Type):
    """Tuple[T1, T2, ...] type."""
    def __init__(self, elems: List[Type]): ...

class TensorType(BaseType):
    """Tensor type with optional alias annotation."""
    alias_info: Optional[AliasInfo]
```

---

## torchgen/dest/

The `torchgen/dest/` directory contains destination-specific generators.

### Generator Modules

```
torchgen/dest/
  gen_tensor_method.py          # Tensor method generation
  gen_aten_lib.py               # ATen library generation
  gen_dispatcher_registrations.py # Dispatch registration generation
  gen_python_functions.py       # Python binding generation
  gen_unboxing.py               # Unboxing function generation
  gen_composite_implicit_autograd.py  # Composite kernel generation
  gen_decompositions.py         # Decomposition generation
  gen_backend_stubs.py          # Backend stub generation
  gen_view_funcs.py             # View function generation
```

---

## torchgen/model.py

Core data model for representing operators, schemas, arguments, and returns.

### Key Classes

```python
# torchgen/model.py

class OperatorName:
    """Operator name with namespace and overload."""
    name: str           # e.g., "add"
    overload: str       # e.g., "Tensor"
    @property
    def full_name(self) -> str:  # "add.Tensor"

class FunctionSchema:
    """Complete operator schema."""
    name: OperatorName
    arguments: List[Argument]
    returns: List[Return]
    @property
    def func(self) -> str:  # full schema string

class Argument:
    """Function argument."""
    name: str
    type: Type
    default: Optional[str]
    alias_info: Optional[AliasInfo]
    is_kwarg_only: bool
    annotation: Optional[str]

    @property
    def is_tensor_like(self) -> bool: ...

class Return:
    """Function return value."""
    name: str
    type: Type
    alias_info: Optional[AliasInfo]

class Operator:
    """Complete operator definition."""
    func: FunctionSchema
    dispatch: Dict[str, str]  # dispatch key -> kernel name
    structured: bool
    structured_delegate: Optional[str]
    tags: Set[str]
    variants: Set[str]
    manual_kernel_registration: bool

class NativeFunction:
    """Native function with all metadata."""
    func: FunctionSchema
    # ... all other fields from YAML
```

### Example Usage

```python
from torchgen.model import FunctionSchema

# Parse a schema string
schema = FunctionSchema.parse(
    "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
)

print(schema.name.name)         # "add"
print(schema.name.overload)     # "Tensor"
print(schema.arguments[0].name) # "self"
print(schema.arguments[0].type) # TensorType
print(schema.returns[0].type)   # TensorType
```

---

## Adding a New Operator: Step by Step

### Step 1: Add to native_functions.yaml

```yaml
# aten/src/ATen/native/native_functions.yaml

- func: my_op(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
  structured: True
```

### Step 2: Implement the Meta Kernel

```cpp
// aten/src/ATen/native/Meta.cpp
TORCH_META_FUNC(my_op) {
    auto result = at::empty_like(self);
    return result;
}
```

### Step 3: Implement the CPU Kernel

```cpp
// aten/src/ATen/native/MyOp.cpp
TORCH_IMPL_FUNC(my_op_cpu)(
    const Tensor& self, const Tensor& other, Scalar alpha, const Tensor& result
) {
    auto self_acc = self.accessor<float, 1>();
    auto other_acc = other.accessor<float, 1>();
    auto result_acc = result.accessor<float, 1>();
    for (int i = 0; i < self_acc.size(0); ++i) {
        result_acc[i] = self_acc[i] + alpha.to<float>() * other_acc[i];
    }
}
```

### Step 4: Implement the CUDA Kernel

```cpp
// aten/src/ATen/native/cuda/MyOp.cu
TORCH_IMPL_FUNC(my_op_cuda)(
    const Tensor& self, const Tensor& other, Scalar alpha, const Tensor& result
) {
    // CUDA kernel implementation
    at::cuda::my_op_kernel(self, other, alpha, result);
}
```

### Step 5: Add Derivative (if needed)

```yaml
# tools/autograd/derivatives.yaml

- name: my_op(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: alpha * grad
  other: alpha * grad
```

### Step 6: Rebuild

```bash
# The build system will run torchgen automatically
python setup.py build
# or
python torchgen/gen.py --source-path aten/src/ATen --install-dir build/aten/src/ATen
```

### Step 7: Test

```python
import torch

x = torch.randn(5)
y = torch.randn(5)
result = torch.my_op(x, y, alpha=2.0)
```

---

## Build System Integration

### CMake Integration

```cmake
# cmake/Codegen.cmake (simplified)

# TorchGen runs during the build process
set(TORCHGEN_COMMAND
    ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/torchgen/gen.py
    --source-path ${CMAKE_CURRENT_SOURCE_DIR}/aten/src/ATen
    --install-dir ${CMAKE_BINARY_DIR}/aten/src/ATen
)

# Custom command to run generation
add_custom_command(
    OUTPUT ${GENERATED_FILES}
    COMMAND ${TORCHGEN_COMMAND}
    DEPENDS ${YAML_FILES} ${TORCHGEN_SOURCES}
    COMMENT "Running torchgen..."
)
```

### Dependencies

TorchGen depends on:
- `native_functions.yaml` - operator definitions
- `tools/autograd/derivatives.yaml` - gradient definitions
- `torchgen/` - the generator code itself

When any of these change, the generated files must be regenerated.

### Selective Build

For mobile/embedded deployment, TorchGen supports selective operator builds:

```yaml
# selected_ops.yaml
ops:
  - add.Tensor
  - mm
  - relu
  - conv2d
```

```bash
# Generate only selected operators
python torchgen/gen.py \
    --source-path aten/src/ATen \
    --install-dir build/aten/src/ATen \
    --op-selection-yaml selected_ops.yaml
```

---

## Summary

TorchGen is the bridge between operator definitions and actual code:

1. **native_functions.yaml**: Single source of truth for all operator schemas
2. **Generated C++ files**: Tensor methods, namespace functions, registration code
3. **Generated Python files**: Type stubs, wrapper functions, documentation
4. **Generator categories**: Python bindings, C++ API, dispatcher registration, decompositions, shape functions
5. **Model classes**: `Operator`, `FunctionSchema`, `Argument`, `Return` represent operators in code
6. **Adding new ops**: Edit YAML, implement kernels, rebuild -- TorchGen handles the rest
7. **Selective build**: Mobile deployments can include only needed operators
