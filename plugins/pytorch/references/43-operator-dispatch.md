# Operator Dispatch System

## Overview

PyTorch's dispatch system is the mechanism that routes operator calls (like `torch.add`) to the correct implementation based on the types and devices of the input tensors. It is a central piece of PyTorch's architecture that enables multiple backends, autograd, tracing, and other features to coexist without conflicting.

The dispatch system works as a multi-level table: each operator has a dispatch table mapping `DispatchKey` values to kernel functions. When an operator is called, the system determines the relevant dispatch keys from the inputs, selects the highest-priority registered kernel, and invokes it.

**Source location**: `c10/core/Dispatcher.h`, `aten/src/ATen/core/dispatch/`

---

## Dispatch Architecture

### High-Level Flow

```
Python: torch.add(a, b)
    |
    v
C++: at::add(a, b)           <-- ATen entry point
    |
    v
Dispatcher::call(op, args)   <-- Central dispatcher
    |
    v
Extract DispatchKeySet        <-- From tensor metadata
    |
    v
Lookup kernel in dispatch table
    |
    v
Call kernel(op, args)         <-- Backend-specific implementation
```

### Dispatch Table Structure

For each operator, the dispatcher maintains a table like:

```
Operator: add.Tensor
+---------------------------+---------------------------+
| DispatchKey               | Kernel                    |
+---------------------------+---------------------------+
| Tracer                    | (if registered)           |
| Functionalize             | (if registered)           |
| ADInplaceOrView           | add_adinplaceorview       |
| Autograd                  | add_autograd              |
| CPU                       | add_cpu_kernel            |
| CUDA                      | add_cuda_kernel           |
| MPS                       | add_mps_kernel            |
| XPU                       | add_xpu_kernel            |
| CompositeExplicitAutograd | add_composite             |
| SparseCPU                 | add_sparse_cpu            |
| SparseCUDA                | add_sparse_cuda           |
+---------------------------+---------------------------+
```

When `add.Tensor` is called with CPU tensors that require gradients, the dispatch key set is `{Autograd, CPU}`. The dispatcher walks the table in priority order, finds `Autograd` first, and calls the autograd kernel. The autograd kernel records the operation for backward, then re-dispatches with the `Autograd` key removed, landing on the `CPU` kernel for the actual computation.

---

## DispatchKey

### Complete Dispatch Key Listing

```cpp
#include <c10/core/DispatchKey.h>

namespace c10 {

enum class DispatchKey : uint16_t {
  // =========================================
  // FUNCTIONALITY KEYS (highest priority)
  // =========================================
  // These keys handle cross-cutting concerns like tracing,
  // functionalization, and autocast. They run before anything else.

  // Alias analysis
  AliasAnalysis = 0,

  // Tracing (JIT tracer wraps operations)
  Tracer = 1,

  // Backend dispatch key selection logic
  BackendSelect = 2,

  // Python implementation (for Python-defined ops)
  Python = 3,

  // Python TLS snapshot
  PythonTLSSnapshot = 4,

  // Automatic mixed precision (CPU)
  AutocastCPU = 5,

  // Automatic mixed precision (CUDA)
  AutocastCUDA = 6,

  // functorch: dynamic layer
  FuncTorchDynamicLayer = 7,

  // functorch: grad wrapper
  FuncTorchGradWrapper = 8,

  // Functionalization pass (converts in-place to functional)
  Functionalize = 9,

  // =========================================
  // AUTOGRAD KEYS
  // =========================================
  // These handle gradient computation. They wrap the forward
  // computation to record operations for backward.

  AutogradOther = 10,              // Catch-all autograd
  AutogradFunction = 11,           // For custom autograd functions
  AutogradCPU = 12,
  AutogradCUDA = 13,
  AutogradXPU = 14,
  AutogradIPU = 15,
  AutogradXLA = 16,
  AutogradHPU = 17,
  AutogradVE = 18,
  AutogradLazy = 19,
  AutogradMPS = 20,
  AutogradMTIA = 21,
  AutogradPrivateUse1 = 22,
  AutogradPrivateUse2 = 23,
  AutogradPrivateUse3 = 24,

  // ADInplaceOrView: handles in-place and view operations for autograd
  ADInplaceOrView = 25,

  // =========================================
  // BACKEND KEYS (lowest priority in their tier)
  // =========================================
  // These are the actual computation kernels.

  CPU = 26,
  CUDA = 27,
  XPU = 28,
  IPU = 29,
  XLA = 30,
  HPU = 31,
  VE = 32,
  Lazy = 33,
  MPS = 34,
  MTIA = 35,
  PrivateUse1 = 36,
  PrivateUse2 = 37,
  PrivateUse3 = 38,

  // Sparse variants
  SparseCPU = 39,
  SparseCUDA = 40,
  SparseXPU = 41,
  SparseVulkan = 42,
  SparseMeta = 43,

  // SparseCsr variants
  SparseCsrCPU = 44,
  SparseCsrCUDA = 45,

  // Nested tensor
  NestedTensorCPU = 46,
  NestedTensorCUDA = 47,

  // Backend fallback
  Dense = 48,

  // =========================================
  // COMPOSITE KEYS
  // =========================================
  // These provide fallback implementations using other ops.
  // They have the lowest priority.

  CompositeExplicitAutogradNonFunctional = 49,
  CompositeExplicitAutograd = 50,
  CompositeImplicitAutograd = 51,

  // =========================================
  // SPECIAL KEYS
  // =========================================
  // Zero tensor optimization
  ZeroTensor = 52,

  // Conjugate handling
  Conjugate = 53,

  // Negative handling
  Negative = 54,

  // Complex number handling
  Complex = 55,

  // Named tensor handling (deprecated)
  Named = 56,

  // =========================================
  // Meta
  // =========================================
  Meta = 57,

  // =========================================
  // Count
  // =========================================
  // Always last
  NumDispatchKeys = 58,

  // Alias for "no key"
  Undefined = 100,
};

} // namespace c10
```

---

## DispatchKeySet

`DispatchKeySet` represents a set of dispatch keys as a bitmask. It is used to determine which kernels to consider when dispatching an operator.

### Header

```cpp
#include <c10/core/DispatchKeySet.h>
```

### Key Operations

```cpp
namespace c10 {

class DispatchKeySet {
public:
  // Constructors
  DispatchKeySet() = default;
  explicit DispatchKeySet(DispatchKey key);
  DispatchKeySet(std::initializer_list<DispatchKey> keys);

  // Set operations
  bool has(DispatchKey key) const;
  DispatchKeySet add(DispatchKey key) const;
  DispatchKeySet add(DispatchKeySet other) const;
  DispatchKeySet remove(DispatchKey key) const;

  // Get highest-priority key
  DispatchKey highestPriorityTypeId() const;

  // Iterator for all keys in the set
  DispatchKeySetIterator begin() const;
  DispatchKeySetIterator end() const;

  // Check if empty
  bool empty() const;

private:
  uint64_t repr_ = 0;  // bitmask representation
};

// Extract dispatch key set from tensor inputs
DispatchKeySet dispatchKeySet(at::TensorList tensors);
DispatchKeySet dispatchKeySet(const at::Tensor& tensor);

} // namespace c10
```

### How DispatchKeySet is Determined

```cpp
// For a single tensor:
// 1. Get backend key from tensor's device (CPU, CUDA, etc.)
// 2. If requires_grad is true, add the corresponding Autograd key
// 3. If tensor is sparse, add Sparse key
// 4. If tensor is conjugated, add Conjugate key

// For multiple tensors:
// Union of all individual tensor key sets
// Plus any functionality keys from thread-local state

// Example:
torch::Tensor t = torch::randn({3, 4}, torch::requires_grad());
// DispatchKeySet = {AutogradCPU, CPU}

torch::Tensor t_cuda = t.to(torch::kCUDA);
// DispatchKeySet = {AutogradCUDA, CUDA}

torch::Tensor t_nograd = torch::randn({3, 4});
// DispatchKeySet = {CPU}
```

---

## Dispatch Table: Operator to Kernel Mapping

### Dispatcher Class

```cpp
#include <c10/core/Dispatcher.h>

namespace c10 {

class Dispatcher {
public:
  // Singleton access
  static Dispatcher& singleton();

  // Operator registration
  // (Typically done via TORCH_LIBRARY macros, not directly)

  // Dispatch an operator
  c10::OperatorHandle findSchema(const OperatorName& name);
  c10::OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);

  // Call with boxing (typed)
  template<typename... Args>
  auto call(const OperatorHandle& op, Args... args) const;

  // Call with unboxing (via IValue)
  void callBoxed(const OperatorHandle& op, std::vector<IValue>& args) const;

  // Registration
  RegistrationHandleRAII registerDef(OperatorSchema schema, std::string debug);
  RegistrationHandleRAII registerImpl(OperatorName name, DispatchKey key,
                                       KernelFunction kernel, std::string debug);

  // Debug
  void dumpOperator(const char* name) const;
  std::vector<OperatorName> getAllOpNames() const;

private:
  std::unordered_map<OperatorName, OperatorHandle> operators_;
};

} // namespace c10
```

---

## Operator Registration

### TORCH_LIBRARY: Defining Operator Schemas

```cpp
// Define operator schemas for a library namespace
TORCH_LIBRARY(my_ops, m) {
  // Define an operator schema
  m.def("add_custom(Tensor self, Tensor other) -> Tensor");

  // Define with default arguments
  m.def("mul_custom(Tensor self, Tensor other, Scalar alpha=1) -> Tensor");

  // Define with multiple overloads
  m.def("norm_custom(Tensor self, Scalar p=2) -> Tensor");
  m.def("norm_custom.dim(Tensor self, Scalar p=2, int[] dim={}, bool keepdim=False) -> Tensor");

  // Define with alias annotations
  m.def("add_inplace(Tensor(a!) self, Tensor other) -> Tensor(a!)");

  // Define with schema string (full control)
  m.def("my_op(Tensor self, int dim, *, bool keepdim=False) -> Tensor");
}
```

### TORCH_LIBRARY_IMPL: Registering Kernels

```cpp
// Register implementations for specific dispatch keys
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("add_custom", &add_custom_cpu);
  m.impl("mul_custom", &mul_custom_cpu);
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("add_custom", &add_custom_cuda);
  m.impl("mul_custom", &mul_custom_cuda);
}

TORCH_LIBRARY_IMPL(my_ops, Autograd, m) {
  m.impl("add_custom", &add_custom_autograd);
}

// Composite implementation (works on all backends)
TORCH_LIBRARY_IMPL(my_ops, CompositeExplicitAutograd, m) {
  m.impl("some_op", [](const Tensor& self, int dim) {
    // Implementation using other ATen ops
    // Works on any backend
    return self.sum(dim);
  });
}
```

### Registration Patterns

```cpp
// Pattern 1: Function pointer
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("add_custom", TORCH_FN(add_custom_cpu_kernel));
}

// Pattern 2: Lambda (must be stateless)
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("add_custom", [](const Tensor& self, const Tensor& other) {
    return self + other;
  });
}

// Pattern 3: Structured kernel (separate meta and impl)
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("add_custom", TORCH_FN(structured_add_cpu_kernel));
}

// Pattern 4: Fallthrough (skip this key, try next)
TORCH_LIBRARY_IMPL(my_ops, Autograd, m) {
  m.fallback(CppFunction::makeFallthrough());
}

// Pattern 5: Error kernel (this key is not supported)
TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.fallback(CppFunction::makeNamedNotFound());
}
```

---

## m.def(): Defining Operator Schemas

### Schema Syntax

```
namespace::name(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
```

Components:
1. **Namespace and name**: `my_ops::add_custom`
2. **Arguments**: positional arguments, then keyword-only (after `*`)
3. **Default values**: `=1`, `=False`, `=None`
4. **Return types**: `-> Tensor`, `-> (Tensor, Tensor)`
5. **Alias annotations**: `Tensor(a!)`, `Tensor(a) -> Tensor(a)`

### Argument Types

| Type | Description | Example |
|------|-------------|---------|
| `Tensor` | Tensor argument | `Tensor self` |
| `Tensor?` | Optional tensor | `Tensor? bias=None` |
| `Tensor[]` | List of tensors | `Tensor[] tensors` |
| `int` | Integer | `int dim=0` |
| `int[]` | List of integers | `int[] dims={}` |
| `float` | Float | `float momentum=0.1` |
| `Scalar` | Scalar (int or float) | `Scalar alpha=1` |
| `bool` | Boolean | `bool keepdim=False` |
| `str` | String | `str mode="exact"` |
| `ScalarType` | Data type | `ScalarType dtype=float32` |
| `Layout` | Tensor layout | `Layout layout=strided` |
| `Device` | Device | `Device device=cpu` |
| `Generator?` | RNG generator | `Generator? generator=None` |
| `MemoryFormat?` | Memory format | `MemoryFormat? memory_format=None` |

### Return Types

```
-> Tensor                    # single tensor
-> (Tensor, Tensor)          # tuple of tensors
-> (Tensor, Tensor, Tensor)  # triple
-> ()                        # no return (in-place ops)
-> Tensor(a!)                # in-place with alias annotation
```

### Alias Annotations

Alias annotations tell the dispatcher about memory aliasing between inputs and outputs:

```
Tensor(a) self               # self aliases with group 'a'
Tensor(a!) self              # self is mutated and aliases with 'a'
Tensor(a) -> Tensor(a)       # output aliases with input 'a'
Tensor(a!) -> Tensor(a!)     # in-place: output is same as input
```

Common patterns:
```
# In-place operation
add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)

# View operation
view(Tensor(a) self, int[] size) -> Tensor(a)

# Out variant
add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
```

---

## Fallthrough Mechanism

Some dispatch keys don't need to do anything special and should pass through to the next key. This is called "fallthrough".

```cpp
// Fallthrough: skip this dispatch key entirely
TORCH_LIBRARY_IMPL(aten, AutogradOther, m) {
  m.fallback(CppFunction::makeFallthrough());
}

// When a key falls through, the dispatcher removes it from the
// DispatchKeySet and tries the next highest-priority key.

// Example dispatch flow with fallthrough:
// Input: {Autograd, CPU}
// 1. Check Autograd -> has kernel -> call autograd kernel
// 2. Autograd kernel re-dispatches without Autograd: {CPU}
// 3. Check CPU -> has kernel -> call CPU kernel
// 4. CPU kernel computes and returns

// Example with fallthrough:
// Input: {Functionalize, Autograd, CPU}
// 1. Check Functionalize -> fallthrough -> skip
// 2. Check Autograd -> has kernel -> call autograd kernel
// 3. Re-dispatch: {CPU}
// 4. Check CPU -> has kernel -> call CPU kernel
```

---

## Dispatch Priority

Dispatch keys are checked in a specific priority order. More specific keys override general ones:

```
Priority (highest to lowest):

1. Functionality keys:
   Tracer > Functionalize > Autocast* > FuncTorch*

2. Autograd keys:
   ADInplaceOrView > Autograd* (backend-specific > generic)

3. Backend keys:
   Dense/Sparse variants > CPU/CUDA/etc.

4. Composite keys:
   CompositeExplicitAutograd > CompositeImplicitAutograd
```

### Priority Resolution Example

```cpp
// A CPU tensor requiring gradients has keys: {AutogradCPU, CPU}
// The dispatcher checks in this order:
// 1. Tracer (not in set, skip)
// 2. Functionalize (not in set, skip)
// 3. Autocast (not in set, skip)
// 4. ADInplaceOrView (not in set, skip)
// 5. AutogradCPU (in set! Use this kernel)

// The AutogradCPU kernel:
// - Records the operation for backward
// - Re-dispatches with AutogradCPU removed: {CPU}
// - Dispatcher now finds CPU kernel
// - CPU kernel performs the actual computation
```

---

## Autograd Dispatch

### How Autograd Wraps Operations

The autograd dispatch key intercepts operations on tensors that require gradients, records them in the computation graph, then delegates to the backend kernel.

```cpp
// Simplified autograd kernel for add:
Tensor add_autograd(const Tensor& self, const Tensor& other, Scalar alpha) {
  // 1. Record the operation for backward
  //    - Save input tensors
  //    - Create a Node for the backward graph

  // 2. Compute the forward result using the backend kernel
  //    - Remove Autograd from dispatch key set
  //    - Re-dispatch to CPU/CUDA kernel
  auto result = dispatcher->callWithKeySetRemoved(
    op, {AutogradCPU}, self, other, alpha);

  // 3. Attach backward graph to result
  //    - result.grad_fn() points to the backward node

  return result;
}
```

### ADInplaceOrView

This dispatch key handles special cases for autograd:
- **In-place operations**: Updates the version counter and handles backward correctly
- **View operations**: Tracks view relationships for gradient computation

```cpp
// In-place operation dispatch
// When you do: x.add_(y)
// ADInplaceOrView kernel:
// 1. Increments x's version counter
// 2. Records the in-place operation
// 3. Re-dispatches to backend

// View operation dispatch
// When you do: y = x.view({3, 4})
// ADInplaceOrView kernel:
// 1. Records that y is a view of x
// 2. Sets up the backward correctly
// 3. Returns the view tensor
```

---

## Functionalization

Functionalization converts in-place and view operations into their functional (out-of-place) equivalents. This is used by torch.compile and the new export system.

### How Functionalization Works

```python
# Original (with side effects):
x.add_(1)           # in-place
y = x.view({3, 4})  # view

# After functionalization:
x_new = torch.add(x, 1)     # functional version
y = torch.view(x_new, {3, 4})  # no aliasing

# The functionalization pass:
# 1. Replaces in-place ops with their out-of-place equivalents
# 2. Replaces view ops with non-aliasing copies
# 3. Tracks mutations to apply them at the end
```

### Functionalization Dispatch

```cpp
// Functionalization kernel for in-place add:
Tensor add__functionalize(Tensor& self, const Tensor& other, Scalar alpha) {
  // Instead of modifying self in-place:
  // 1. Compute new value: result = self + alpha * other
  // 2. Propagate the mutation
  // 3. Return result as if it were self
  auto result = at::add(self, other, alpha);
  // Propagate mutation to aliased tensors
  propagate_mutation(self, result);
  return self;
}
```

---

## Python Dispatcher

The Python dispatcher allows Python-level interception of operator calls. It is used for tracing, profiling, and debugging.

### Python Dispatch Key

```python
# Access dispatch key information
import torch

# Get dispatch key from tensor
t = torch.randn(3, 4)
key = torch._C._dispatch_key(t)  # returns the primary dispatch key

# Get all dispatch key sets
key_set = torch._C._dispatch_key_set(t)

# Python-level dispatch hooks
class MyDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"Dispatching: {func}")
        return func(*args, **kwargs)

with MyDispatchMode():
    result = torch.add(torch.randn(3), torch.randn(3))
    # Prints: "Dispatching: aten.add.Tensor"
```

### torch._dispatch Module

```python
# Low-level dispatch access
import torch._dispatch

# Get operator handle
op = torch._dispatch.get_op("aten::add")

# Check if kernel is registered
has_cpu = torch._dispatch.has_kernel("aten::add", "CPU")
has_cuda = torch._dispatch.has_kernel("aten::add", "CUDA")

# List all registered kernels
kernels = torch._dispatch.get_all_kernels("aten::add")
```

---

## Tracing Dispatch with TORCH_SHOW_DISPATCH_TRACE

```bash
# Enable dispatch tracing
export TORCH_SHOW_DISPATCH_TRACE=1

# Run Python script
python my_script.py

# Output shows dispatch decisions:
# DISPATCH_TRACE: aten::add.Tensor key=AutogradCPU
# DISPATCH_TRACE: aten::add.Tensor key=CPU
# DISPATCH_TRACE: aten::mm key=AutogradCPU
# DISPATCH_TRACE: aten::mm key=CPU
```

### Other Dispatch Debug Flags

```bash
# Show operator registration
TORCH_LOGS="+dispatch" python script.py

# Show dispatch key extraction
TORCH_SHOW_DISPATCH_TRACE=1 python script.py

# Show kernel registration details
TORCH_LOGS="+dispatch_registration" python script.py
```

---

## Dispatch Key Extraction from Tensors

### How Keys are Extracted

```cpp
// From a single tensor:
DispatchKeySet extractKeys(const Tensor& t) {
  DispatchKeySet keys;

  // 1. Backend key from device
  if (t.is_cpu()) keys.add(DispatchKey::CPU);
  if (t.is_cuda()) keys.add(DispatchKey::CUDA);
  // ... other backends

  // 2. Autograd key if requires_grad
  if (t.requires_grad()) {
    if (t.is_cpu()) keys.add(DispatchKey::AutogradCPU);
    if (t.is_cuda()) keys.add(DispatchKey::AutogradCUDA);
    // ... other backends
  }

  // 3. Sparse key if sparse layout
  if (t.is_sparse()) {
    keys.add(DispatchKey::SparseCPU);  // or SparseCUDA
  }

  // 4. Conjugate/Negative flags
  if (t.is_conj()) keys.add(DispatchKey::Conjugate);
  if (t.is_neg()) keys.add(DispatchKey::Negative);

  return keys;
}

// From multiple tensors:
DispatchKeySet extractKeys(TensorList tensors) {
  DispatchKeySet keys;
  for (const auto& t : tensors) {
    keys = keys | extractKeys(t);
  }
  return keys;
}
```

---

## Backend Dispatch Keys vs Autograd Keys

### Backend Keys

Backend keys represent actual computation hardware:
- `CPU`: Execute on CPU
- `CUDA`: Execute on NVIDIA GPU
- `XPU`: Execute on Intel GPU
- `MPS`: Execute on Apple Metal
- `XLA`: Execute on TPU via XLA
- etc.

Backend kernels perform the actual numerical computation.

### Autograd Keys

Autograd keys wrap backend kernels to record gradient information:
- `AutogradCPU`: Autograd for CPU tensors
- `AutogradCUDA`: Autograd for CUDA tensors
- etc.

Autograd kernels do NOT perform computation. They:
1. Save tensors needed for backward
2. Re-dispatch to the backend kernel
3. Record the operation in the computation graph

### Separation Benefits

```
The separation allows:

1. Backend developers only need to write the computation kernel
   (registered under CPU, CUDA, etc.)

2. Autograd is handled automatically by the autograd kernels
   (registered under AutogradCPU, etc.)

3. New functionality (tracing, functionalization) can be added
   by inserting new dispatch keys without modifying existing code

4. Custom backends can be added by registering kernels for
   PrivateUse1/2/3 keys
```

---

## Dispatch Key Resolution Flow (Detailed)

```
Operator call: torch.add(a, b) where a, b are CPU tensors with requires_grad=True

Step 1: Determine DispatchKeySet
  - a.is_cpu() -> add CPU
  - a.requires_grad() -> add AutogradCPU
  - b.is_cpu() -> add CPU (already present)
  - b.requires_grad() -> add AutogradCPU (already present)
  - Thread-local state -> add any functionality keys
  Result: {AutogradCPU, CPU}

Step 2: Look up operator handle
  - Find "aten::add.Tensor" in operator registry

Step 3: Walk dispatch table in priority order
  - Tracer: not in set, skip
  - Functionalize: not in set, skip
  - AutocastCPU: not in set, skip
  - ADInplaceOrView: not in set (not in-place/view), skip
  - AutogradCPU: IN SET! Found kernel.

Step 4: Call AutogradCPU kernel
  - Saves tensors for backward
  - Records add operation in computation graph
  - Re-dispatches with AutogradCPU removed: {CPU}

Step 5: Walk dispatch table again with {CPU}
  - Skip all autograd keys
  - CPU: IN SET! Found kernel.

Step 6: Call CPU kernel
  - Performs actual element-wise addition
  - Returns result tensor

Step 7: AutogradCPU kernel receives result
  - Attaches grad_fn to result
  - Returns result with gradient tracking
```

---

## Thread-Local Dispatch Key Set

In addition to keys extracted from tensors, the dispatcher also considers thread-local dispatch keys set by features like:

```python
# Autocast (automatic mixed precision)
with torch.autocast("cuda"):
    # Adds AutocastCUDA to thread-local dispatch key set
    result = torch.mm(a, b)  # may use FP16

# Custom dispatch mode
with torch.utils._python_dispatch.TorchDispatchMode():
    # Adds Python dispatch key
    pass

# Inference mode (disables autograd)
with torch.inference_mode():
    # Removes Autograd from dispatch key set
    result = torch.add(a, b)  # no autograd overhead
```

### C++ Equivalent

```cpp
// Inference mode (skip autograd)
{
    at::InferenceMode guard(true);
    // Autograd keys are excluded from dispatch
    auto result = at::add(a, b);
}

// No grad guard
{
    at::NoGradGuard guard;
    // Same as inference mode for autograd
}
```

---

## Registration Macros Summary

```cpp
// Define a new operator (creates dispatch table)
TORCH_LIBRARY(namespace, m) {
    m.def("op_name(type args) -> return_type");
}

// Register implementation for specific dispatch key
TORCH_LIBRARY_IMPL(namespace, DispatchKey, m) {
    m.impl("op_name", kernel_function);
}

// Catchall implementation (any dispatch key not yet handled)
TORCH_LIBRARY_IMPL(namespace, CompositeImplicitAutograd, m) {
    m.impl("op_name", generic_kernel);
}

// Register autograd implementation
TORCH_LIBRARY_IMPL(namespace, Autograd, m) {
    m.impl("op_name", autograd_kernel);
}

// Register meta implementation (for shape inference)
TORCH_LIBRARY_IMPL(namespace, Meta, m) {
    m.impl("op_name", meta_kernel);
}
```

---

## Debugging Dispatch Issues

### Common Problems

```python
# Problem: "add could not be resolved to an operator"
# Cause: Operator not registered
# Solution: Check that TORCH_LIBRARY is linked

# Problem: "No kernel registered for add.Tensor on MPS"
# Cause: No MPS kernel registered for this op
# Solution: Register a kernel or use CompositeExplicitAutograd

# Problem: "add_ is not supported for tensors that require grad"
# Cause: In-place op on a leaf tensor with requires_grad
# Solution: Use out-of-place version or detach first
```

### Debugging Commands

```python
# Check registered kernels for an operator
import torch
op = torch._C._get_op("aten::add")
print(op.schema())
print(op.dispatch_kernels())  # list registered kernels

# Check dispatch key set for a tensor
t = torch.randn(3, requires_grad=True)
print(torch._C._dispatch_key_set(t))

# Enable dispatch logging
import logging
torch._logging.set_logs(dispatch=True)
```

---

## Summary

PyTorch's dispatch system is a sophisticated multi-level dispatch mechanism:

1. **DispatchKey**: Identifies each dispatch concern (backend, autograd, tracing, etc.)
2. **DispatchKeySet**: Bitmask set of keys extracted from input tensors and thread-local state
3. **Priority ordering**: Functionality keys > Autograd keys > Backend keys > Composite keys
4. **Registration**: `TORCH_LIBRARY` defines schemas, `TORCH_LIBRARY_IMPL` registers kernels
5. **Autograd dispatch**: Intercepts gradient-requiring operations, records for backward, re-dispatches
6. **Functionalization**: Converts in-place/view ops to functional form for compilation
7. **Fallthrough**: Allows keys to pass through without processing
8. **Python dispatcher**: Enables Python-level hooks for tracing and debugging
