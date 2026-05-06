# Apache TVM - Error Handling and Debugging

This reference covers error handling and debugging in Apache TVM, including error types, diagnostic mechanisms, debugging strategies, and common troubleshooting scenarios.

---

## 33.1 Error Types in TVM

TVM provides a hierarchy of error types that map to different failure modes across the compilation pipeline. Understanding these error types is essential for efficient debugging.

### 33.1.1 tvm.error.TVMError

`TVMError` is the base exception class for all TVM-related errors. It wraps errors originating from both the C++ backend and the Python frontend. Most TVM operations raise `TVMError` (or one of its subclasses) when something goes wrong.

```python
import tvm
from tvm import relay

try:
    mod = tvm.ir.IRModule()
    # Attempting to run a pass on an empty module may raise TVMError
    result = tvm.relay.transform.InferType()(mod)
except tvm.error.TVMError as e:
    print(f"TVMError caught: {e}")
```

The error message typically includes:
- A high-level description of what failed.
- The C++ source file and line number where the error was raised.
- Any contextual information (e.g., variable names, shapes) that was available.

### 33.1.2 tvm.error.InternalError

`InternalError` indicates a bug within TVM itself -- an invariant violation or unreachable code path that was unexpectedly reached. These errors should be reported to the TVM project as issues.

```python
import tvm
from tvm import tir

# InternalError is raised when TVM's internal invariants are violated.
# For example, if a TIR pass encounters an IR node combination that
# should never occur after valid transformations:
try:
    # Hypothetical: malformed IR that violates internal invariants
    pass
except tvm.error.InternalError as e:
    print(f"Internal compiler error: {e}")
    print("Please file a bug report at https://github.com/apache/tvm/issues")
```

When you encounter an `InternalError`:
1. Note the exact TVM version and commit hash.
2. Minimize the reproducer as much as possible.
3. File an issue on the Apache TVM GitHub repository with the full stack trace.

### 33.1.3 tvm.error.RPCError

`RPCError` is raised when an operation fails during remote procedure call (RPC) execution. This commonly occurs when using TVM's RPC infrastructure for cross-compilation and remote device testing.

```python
import tvm
from tvm import rpc

# Connect to a remote RPC server
remote = rpc.connect("192.168.1.100", 9090)

try:
    # Attempting to create a remote device that does not exist
    dev = remote.device("cuda", 0)
    # Or running a remote module that encounters an error
except tvm.error.RPCError as e:
    print(f"RPC error: {e}")
    # Common causes:
    # - RPC server not running or unreachable
    # - Remote device not available
    # - Remote module loading failure
    # - Network timeout
```

Common RPC error scenarios:
- **Connection refused**: The RPC server is not running at the specified host/port.
- **Device not found**: The target device (e.g., CUDA GPU) is not available on the remote machine.
- **Module load error**: The compiled module cannot be loaded on the remote target, often due to architecture mismatch.
- **Timeout**: The remote operation took too long, possibly due to an infinite loop or extremely slow computation.

### 33.1.4 tvm.error.OpError

`OpError` is the base class for operator-related errors. It is raised when an operator cannot be executed due to invalid inputs, missing implementations, or other operator-specific issues.

```python
import tvm
from tvm import relay

try:
    # Calling an operator with invalid arguments
    x = relay.var("x", shape=(3, 4), dtype="float32")
    # Attempting an invalid reshape (total elements mismatch)
    y = relay.reshape(x, (5, 5))  # 3*4=12 != 5*5=25
    mod = tvm.IRModule.from_expr(y)
    mod = relay.transform.InferType()(mod)
except tvm.error.OpError as e:
    print(f"Operator error: {e}")
```

### 33.1.5 tvm.error.OpNotImplementedError

`OpNotImplementedError` is raised when an operator is not supported by the current target or compilation path. This commonly occurs when:
- A Relay operator has no registered implementation for the specified target.
- A TIR intrinsic is not supported by the target code generator.
- A strategy is not available for the given target and operator combination.

```python
import tvm
from tvm import relay

# Example: Using an operator that may not be implemented for all targets
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
# Some exotic operations may not be implemented for certain targets
try:
    y = relay.nn.batch_norm(x, relay.var("gamma"), relay.var("beta"),
                            relay.var("mean"), relay.var("var"),
                            axis=1, epsilon=1e-5)
    mod = tvm.IRModule.from_expr(y[0])
    with tvm.transform.PassContext(opt_level=3):
        # Compiling for a target that does not support this op
        lib = tvm.relay.build(mod, target="llvm")
except tvm.error.OpNotImplementedError as e:
    print(f"Operator not implemented: {e}")
```

To resolve `OpNotImplementedError`:
1. Check if the operator is registered for the target using `tvm.target.Target.current()`.
2. Consider implementing a custom operator or strategy for the target.
3. Use a different target that supports the operator.
4. Implement a fallback using `relay frontend` op conversion.

### 33.1.6 tvm.error.AttributeError

`AttributeError` in TVM is raised when an IR node has invalid or missing attributes. This often occurs when:
- A Relay operator is created with incorrect attribute types.
- A TIR builtin is called with wrong attribute values.
- An IR node is accessed with a missing field.

```python
import tvm
from tvm import relay, tir

# Example: Invalid attribute for an operator
try:
    x = relay.var("x", shape=(3, 4), dtype="float32")
    # Conv2D requires specific attribute format
    w = relay.var("w", shape=(16, 3, 3, 3), dtype="float32")
    # Forgetting required attributes or passing invalid values
    y = relay.nn.conv2d(x, w, strides=1, padding=0, channels=16, kernel_size=(3, 3))
except tvm.error.AttributeError as e:
    print(f"Attribute error: {e}")
```

### 33.1.7 tvm.error.DeviceAPIError

`DeviceAPIError` is raised when a device-specific API call fails. This includes GPU kernel launch failures, memory allocation errors on accelerators, and other hardware-level issues.

```python
import tvm
import numpy as np

try:
    # Attempting to allocate too much GPU memory
    dev = tvm.cuda(0)
    # Or launching a kernel with invalid parameters
    x = tvm.nd.array(np.zeros((10000, 10000), dtype="float32"), device=dev)
except tvm.error.DeviceAPIError as e:
    print(f"Device API error: {e}")
    # Common causes:
    # - Out of memory
    # - Invalid device
    # - CUDA driver error
    # - Hardware failure
```

Common `DeviceAPIError` scenarios and resolutions:
- **Out of memory**: Reduce batch size, use a smaller model, or free unused allocations.
- **Invalid device index**: Verify the device is available with `tvm.cuda(0).exist`.
- **Driver mismatch**: Ensure the CUDA driver version is compatible with the runtime.

---

## 33.2 Error Reporting Mechanisms

### 33.2.1 Diagnostic Context

TVM's diagnostic system provides structured error reporting with source location information. The `DiagnosticContext` captures errors and warnings during compilation, associating them with specific source locations.

```python
import tvm
from tvm import relay, diagnostics

# The diagnostic context is automatically active during compilation
# Errors are collected and reported with source information
@tvm.register_func("diag_callback")
def diag_callback(diag):
    print(f"Diagnostic: {diag.level} at {diag.span}")
    print(f"  Message: {diag.message}")

# Register a diagnostic engine
diag_ctx = diagnostics.DiagnosticContext(
    diagnostics.DefaultDiagnosticEngine,
    tvm.get_global_func("diag_callback")
)
```

The diagnostic system includes:
- **Diagnostic level**: `Error`, `Warning`, `Note`.
- **Source span**: File, line, and column information.
- **Context**: Surrounding source code for the error location.

### 33.2.2 Source Location Tracking

Every IR node in TVM can carry a `Span` that records the source location where it was created. This enables precise error reporting that points back to the original source.

```python
import tvm
from tvm import tir, ir

# Creating a TIR variable with source location
span = ir.Span(ir.SourceName("my_script.py"), line=10, column=5)
x = tir.Var("x", dtype="float32", span=span)

# The span is preserved through transformations
print(f"Variable {x.name_hint} defined at: {x.span}")
# Output: Variable x defined at: my_script.py:10:5
```

### 33.2.3 TVMScript Error Reporting with Line Numbers

TVMScript provides rich error messages that include the exact source line and column where the error occurred. This makes debugging TVMScript programs much easier.

```python
import tvm
from tvm.script import tir as T

@T.prim_func
def matmul(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    # TVMScript tracks line numbers for error reporting
    for i, j, k in T.grid(128, 128, 128):
        # If there's a type error here, TVM reports the exact line
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# If a transformation introduces an error, the diagnostic points to
# the original TVMScript line where the affected operation was defined.
```

When a TVMScript error occurs, the output looks like:

```
Traceback (most recent call last):
  File "example.py", line 15
    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
TVMError: The following error was encountered while parsing TVMScript:
  line 15: type mismatch: expected 'float32' but got 'int32'
```

### 33.2.4 Span Information for IR Nodes

Spans are propagated through the compilation pipeline, allowing later stages to reference back to the original source. When writing passes, you should preserve spans.

```python
import tvm
from tvm import tir, ir

def my_tir_pass(f: tir.PrimFunc, mod: tvm.IRModule) -> tvm.IRModule:
    """Example pass that preserves span information."""
    new_body = []
    for stmt in f.body:
        if isinstance(stmt, tir.For):
            # Preserve the span from the original statement
            new_for = tir.For(
                loop_var=stmt.loop_var,
                min_val=stmt.min_val,
                extent=stmt.extent * 2,  # Modified extent
                kind=stmt.kind,
                body=stmt.body,
                thread_binding=stmt.thread_binding,
                annotations=stmt.annotations,
                span=stmt.span,  # Preserve original span
            )
            new_body.append(new_for)
        else:
            new_body.append(stmt)
    # Return updated module
    return mod
```

---

## 33.3 Debugging Strategies

### 33.3.1 Print IR at Various Stages

One of the most powerful debugging techniques in TVM is printing the intermediate representation at various stages of compilation. This allows you to inspect the IR before and after each transformation.

```python
import tvm
from tvm import relay

# Create a simple Relay model
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
w = relay.var("w", shape=(32, 3, 3, 3), dtype="float32")
y = relay.nn.conv2d(x, w, padding=(1, 1), channels=32, kernel_size=(3, 3))
y = relay.nn.relu(y)
mod = tvm.IRModule.from_expr(y)

# Method 1: Use mod.show() to pretty-print the IR
print("=== Original Module ===")
mod.show()

# Method 2: Use mod.script() to get TVMScript representation
print("=== TVMScript ===")
print(mod.script())

# Method 3: Print after each pass in a sequence
seq = tvm.transform.Sequential([
    relay.transform.InferType(),
    relay.transform.FoldConstant(),
    relay.transform.SimplifyExpr(),
])

# Apply passes one at a time and print intermediate results
mod_infertype = relay.transform.InferType()(mod)
print("=== After InferType ===")
mod_infertype.show()

mod_foldconst = relay.transform.FoldConstant()(mod_infertype)
print("=== After FoldConstant ===")
mod_foldconst.show()

mod_simplexpr = relay.transform.SimplifyExpr()(mod_foldconst)
print("=== After SimplifyExpr ===")
mod_simplexpr.show()
```

### 33.3.2 PassContext Tracing

The `PassContext` supports tracing, which captures the IR module before and after each pass execution. This is invaluable for identifying which pass introduced a bug.

```python
import tvm
from tvm import relay

# Create a module
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
w1 = relay.var("w1", shape=(32, 3, 3, 3), dtype="float32")
y = relay.nn.conv2d(x, w1, padding=(1, 1), channels=32, kernel_size=(3, 3))
mod = tvm.IRModule.from_expr(y)

# Enable tracing via PassContext
with tvm.transform.PassContext(trace=tvm.transform.PassTrace()):
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FoldConstant()(mod)
    mod = relay.transform.AlterOpLayout()(mod)

# Access the trace to see IR at each step
# The trace records the IR before and after each pass
```

You can also use a callback-based approach:

```python
import tvm
from tvm import relay

def print_pass_info(pass_info, mod_before, mod_after):
    """Callback that prints pass information."""
    print(f"Pass: {pass_info.name} (opt_level={pass_info.opt_level})")
    print("  Before:")
    print(f"    Functions: {len(mod_before.functions)}")
    print("  After:")
    print(f"    Functions: {len(mod_after.functions)}")
    print()

# Register as an instrumentation callback
instrument = tvm.transform.PassInstrument([
    tvm.transform.PassTimingInstrument(),
])

with tvm.transform.PassContext(instruments=[instrument]):
    mod = tvm.relay.optimize(mod, target="llvm")
```

### 33.3.3 Individual Pass Application

Instead of running the entire compilation pipeline, apply passes individually to isolate issues.

```python
import tvm
from tvm import relay
import numpy as np

# Create a model
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
w = relay.const(np.random.randn(32, 3, 3, 3).astype("float32"))
y = relay.nn.conv2d(x, w, padding=(1, 1), channels=32, kernel_size=(3, 3))
y = relay.nn.batch_norm(y, relay.var("gamma"), relay.var("beta"),
                         relay.var("mean"), relay.var("var"))[0]
mod = tvm.IRModule.from_expr(y)

# Apply passes one at a time
passes = [
    relay.transform.InferType(),
    relay.transform.FoldConstant(),
    relay.transform.FuseOps(fuse_opt_level=2),
    relay.transform.AlterOpLayout(),
    relay.transform.FuseOpsByPattern([]),
]

current_mod = mod
for p in passes:
    print(f"Applying: {p.info.name}")
    try:
        current_mod = p(current_mod)
        print("  Success!")
        # Print IR if needed
        # current_mod.show()
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  IR before this pass:")
        current_mod.show()
        raise
```

### 33.3.4 Verbose Logging with TVM_LOG_DEBUG

TVM uses a logging system that supports different verbosity levels. The `TVM_LOG_DEBUG` environment variable controls which debug messages are emitted.

```bash
# Enable all debug logging
export TVM_LOG_DEBUG=1

# Enable debug logging for specific modules
export TVM_LOG_DEBUG="relay/transform/*=1,tir/transform/*=1"

# Enable verbose logging for a specific pass
export TVM_LOG_DEBUG="relay/transform/fuse_ops.cc=2"

# Disable all debug logging (default)
export TVM_LOG_DEBUG=0
```

In Python, you can also configure logging:

```python
import tvm
import logging

# Set TVM's logging level
tvm.set_log_level("DEBUG")  # or "INFO", "WARNING", "ERROR", "FATAL"

# Or use Python's logging module
logging.basicConfig(level=logging.DEBUG)
```

In C++ code, you can add debug logging:

```cpp
#include <tvm/support/logging.h>

void my_pass(tir::PrimFunc func) {
    // Use VLOG for verbose debug logging
    VLOG(1) << "Processing function: " << func->GetNameHint();

    // Use LOG for important messages
    LOG(INFO) << "Pass started";

    // Use DLOG for debug-only logging (only in debug builds)
    DLOG(INFO) << "Debug: visiting statements";

    // Use LOG(FATAL) for unrecoverable errors
    // LOG(FATAL) << "Unexpected node type: " << node->GetTypeKey();
}
```

### 33.3.5 Breakpoints in Custom Passes

You can add breakpoints in custom passes to inspect the IR at specific points during compilation.

```python
import tvm
from tvm import relay, ir
import pdb  # Python debugger

@relay.transform.function_pass(opt_level=0)
class DebugPass:
    """A pass that sets a breakpoint for debugging."""

    def transform_function(self, func, mod, ctx):
        print("=== Breakpoint in DebugPass ===")
        print(f"Function: {func}")

        # Set a breakpoint here
        # pdb.set_trace()

        # Or use a conditional breakpoint
        # if isinstance(func.body, relay.Call):
        #     pdb.set_trace()

        return func

# Use the debug pass in a pipeline
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
y = relay.nn.relu(x)
mod = tvm.IRModule.from_expr(y)

seq = tvm.transform.Sequential([
    relay.transform.InferType(),
    DebugPass(),  # Insert debug pass at any point
    relay.transform.FoldConstant(),
])

mod = seq(mod)
```

---

## 33.4 BasePyModule for Debugging

### 33.4.1 Hybrid Python + TIR Execution

`BasePyModule` provides a hybrid execution environment where TIR programs can run partially on the compiled backend and partially in Python. This is extremely useful for debugging and incremental development.

```python
import tvm
from tvm import tir
from tvm.script import tir as T
import numpy as np

@T.prim_func
def vector_add(
    A: T.Buffer((1024,), "float32"),
    B: T.Buffer((1024,), "float32"),
    C: T.Buffer((1024,), "float32"),
) -> None:
    for i in range(1024):
        with T.block("add"):
            vi = T.axis.spatial(1024, i)
            C[vi] = A[vi] + B[vi]

# Create a BasePyModule for hybrid execution
mod = tvm.ir.IRModule.from_expr(vector_add)
pymod = tvm.tir.BasePyModule(mod["main"], mod, tvm.ir.IRModule())

# Execute using the Python fallback
a = tvm.nd.array(np.ones(1024, dtype="float32"))
b = tvm.nd.array(np.ones(1024, dtype="float32") * 2)
c = tvm.nd.array(np.zeros(1024, dtype="float32"))

pymod["main"](a, b, c)
print(f"Result: {c.numpy()[:5]}")  # [3., 3., 3., 3., 3.]
```

### 33.4.2 Python Fallback for Unimplemented Ops

When a TIR operation is not yet implemented for a target, `BasePyModule` falls back to Python execution, allowing you to test the logic even before full compilation support is available.

```python
import tvm
from tvm.script import tir as T

@T.prim_func
def custom_op(
    X: T.Buffer((128,), "float32"),
    Y: T.Buffer((128,), "float32"),
) -> None:
    for i in range(128):
        with T.block("compute"):
            vi = T.axis.spatial(128, i)
            # Custom computation that may not be compiled yet
            Y[vi] = T.max(X[vi], T.float32(0.0)) * T.float32(2.0)

mod = tvm.ir.IRModule.from_expr(custom_op)

# Use BasePyModule to execute in Python without full compilation
pymod = tvm.tir.BasePyModule(mod["main"], mod, tvm.ir.IRModule())

# This works even if the target doesn't support this operation natively
x = tvm.nd.array(np.array([-1, 2, -3, 4], dtype="float32"))
y = tvm.nd.array(np.zeros(4, dtype="float32"))
pymod["main"](x, y)
print(y.numpy())  # [0., 4., 0., 8.]
```

### 33.4.3 Incremental Development Workflow

`BasePyModule` supports an incremental development workflow where you can:
1. Write TIR code in TVMScript.
2. Execute it immediately in Python.
3. Gradually add optimizations.
4. Verify correctness at each step.

```python
import tvm
from tvm.script import tir as T
import numpy as np

# Step 1: Write initial (unoptimized) version
@T.prim_func
def matrix_mul_naive(
    A: T.Buffer((64, 64), "float32"),
    B: T.Buffer((64, 64), "float32"),
    C: T.Buffer((64, 64), "float32"),
) -> None:
    for i, j, k in T.grid(64, 64, 64):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

mod = tvm.ir.IRModule.from_expr(matrix_mul_naive)

# Step 2: Verify correctness using Python execution
pymod = tvm.tir.BasePyModule(mod["main"], mod, tvm.ir.IRModule())
a = tvm.nd.array(np.random.randn(64, 64).astype("float32"))
b = tvm.nd.array(np.random.randn(64, 64).astype("float32"))
c = tvm.nd.array(np.zeros((64, 64), dtype="float32"))
pymod["main"](a, b, c)

# Step 3: Compare against NumPy reference
np_ref = a.numpy() @ b.numpy()
np.testing.assert_allclose(c.numpy(), np_ref, rtol=1e-5)
print("Correctness verified!")

# Step 4: Now apply optimizations and compile for target
# (Apply TIR passes, schedule, compile, etc.)
```

### 33.4.4 Zero-Copy Tensor Conversion via DLPack

`BasePyModule` uses DLPack for zero-copy tensor conversion between TVM and Python, avoiding unnecessary data copies during debugging.

```python
import tvm
import numpy as np
import torch

# Create a TVM NDArray
arr = tvm.nd.array(np.random.randn(128, 128).astype("float32"))

# Convert to DLPack (zero-copy)
dlpack = arr.to_dlpack()

# Import into PyTorch (zero-copy)
torch_tensor = torch.from_dlpack(dlpack)

# Modifications in PyTorch are reflected in TVM
torch_tensor[0, 0] = 42.0
assert arr.numpy()[0, 0] == 42.0  # Same memory

# Convert from PyTorch back to TVM (zero-copy)
dlpack_back = torch_tensor.to_dlpack()
arr_back = tvm.nd.from_dlpack(dlpack_back)
```

### 33.4.5 Cross-Level Calling Between Python and TVM

`BasePyModule` supports calling between Python functions and compiled TVM functions, enabling mixed execution for debugging complex pipelines.

```python
import tvm
from tvm.script import tir as T
import numpy as np

@T.prim_func
def elementwise_add(
    A: T.Buffer((256,), "float32"),
    B: T.Buffer((256,), "float32"),
    C: T.Buffer((256,), "float32"),
) -> None:
    for i in range(256):
        with T.block("add"):
            vi = T.axis.spatial(256, i)
            C[vi] = A[vi] + B[vi]

@T.prim_func
def elementwise_mul(
    A: T.Buffer((256,), "float32"),
    B: T.Buffer((256,), "float32"),
    C: T.Buffer((256,), "float32"),
) -> None:
    for i in range(256):
        with T.block("mul"):
            vi = T.axis.spatial(256, i)
            C[vi] = A[vi] * B[vi]

# Create module with multiple functions
mod = tvm.ir.IRModule()
mod["add"] = elementwise_add
mod["mul"] = elementwise_mul

# Use BasePyModule for debugging
pymod = tvm.tir.BasePyModule(mod["add"], mod, tvm.ir.IRModule())

# Execute and verify each function independently
a = tvm.nd.array(np.ones(256, dtype="float32"))
b = tvm.nd.array(np.ones(256, dtype="float32") * 3)
c = tvm.nd.array(np.zeros(256, dtype="float32"))

pymod["add"](a, b, c)
assert np.allclose(c.numpy(), 4.0)

pymod = tvm.tir.BasePyModule(mod["mul"], mod, tvm.ir.IRModule())
d = tvm.nd.array(np.zeros(256, dtype="float32"))
pymod["mul"](a, c, d)
assert np.allclose(d.numpy(), 4.0)
```

---

## 33.5 Common Debugging Scenarios

### 33.5.1 Model Import Failures

Model import failures typically occur when converting models from frameworks like PyTorch, TensorFlow, or ONNX into Relay.

```python
import tvm
from tvm import relay
import numpy as np

# Common failure: Unsupported operator in frontend
try:
    # Attempting to import a model with unsupported ops
    import onnx
    model = onnx.load("model.onnx")
    mod, params = relay.frontend.from_onnx(model)
except tvm.error.TVMError as e:
    if "not supported" in str(e):
        print(f"Unsupported operator: {e}")
        # Solution: Register a custom converter for the operator
    elif "shape inference" in str(e):
        print(f"Shape inference failed: {e}")
        # Solution: Provide static input shapes
    else:
        raise

# Common failure: Dynamic shapes not supported
# Solution: Specify static input shapes during import
mod, params = relay.frontend.from_onnx(
    model,
    shape={"input": [1, 3, 224, 224]},  # Force static shapes
    freeze_params=True,
)
```

Debugging steps for model import failures:
1. **Check operator support**: Verify the operator is supported by the TVM frontend.
2. **Provide static shapes**: Dynamic shapes are not always supported; provide concrete shapes.
3. **Simplify the model**: Try importing a smaller model or subgraph.
4. **Check version compatibility**: Ensure the model format version matches TVM's frontend.

### 33.5.2 Shape Mismatch Errors

Shape mismatch errors occur when tensor dimensions do not align correctly.

```python
import tvm
from tvm import relay

# Example: Shape mismatch in matrix multiplication
x = relay.var("x", shape=(1, 128), dtype="float32")
w = relay.var("w", shape=(256, 64), dtype="float32")  # Mismatch: 128 != 256
try:
    y = relay.nn.dense(x, w)  # Will fail during InferType
    mod = tvm.IRModule.from_expr(y)
    mod = relay.transform.InferType()(mod)
except tvm.error.TVMError as e:
    print(f"Shape mismatch: {e}")

# Fix: Correct the weight shape
w_fixed = relay.var("w", shape=(64, 128), dtype="float32")  # Correct for dense
y = relay.nn.dense(x, w_fixed)  # Now x: (1, 128) @ w_fixed.T: (128, 64) -> (1, 64)
mod = tvm.IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
print("Fixed! Output shape:", mod["main"].body.checked_type.shape)
```

### 33.5.3 Type Mismatch Errors

Type mismatch errors occur when operations receive tensors of incompatible data types.

```python
import tvm
from tvm import relay

# Example: Type mismatch
x = relay.var("x", shape=(1, 128), dtype="float32")
y = relay.var("y", shape=(1, 128), dtype="int32")  # Different dtype
try:
    z = relay.add(x, y)  # Type mismatch: float32 + int32
    mod = tvm.IRModule.from_expr(z)
    mod = relay.transform.InferType()(mod)
except tvm.error.TVMError as e:
    print(f"Type mismatch: {e}")

# Fix: Insert a cast
y_cast = relay.cast(y, "float32")
z = relay.add(x, y_cast)  # Now both float32
mod = tvm.IRModule.from_expr(z)
mod = relay.transform.InferType()(mod)
```

### 33.5.4 Target-Specific Codegen Errors

Target-specific errors occur during code generation for a particular hardware target.

```python
import tvm
from tvm import relay
import numpy as np

# Example: Unsupported operation for target
x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
y = relay.nn.softmax(x, axis=1)
mod = tvm.IRModule.from_expr(y)

try:
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target="cuda")
except tvm.error.OpNotImplementedError as e:
    print(f"Not implemented for CUDA: {e}")
    # Fallback to LLVM
    lib = tvm.relay.build(mod, target="llvm")
```

### 33.5.5 Runtime Execution Errors

Runtime errors occur during module execution on the target device.

```python
import tvm
import numpy as np

# Example: Device mismatch
dev = tvm.cuda(0)
lib = tvm.relay.build(mod, target="llvm")  # Compiled for CPU

try:
    # Attempting to run CPU module on GPU
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
except Exception as e:
    print(f"Runtime error: {e}")
    # Fix: Use correct device
    dev = tvm.cpu(0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

# Example: Input shape mismatch at runtime
try:
    runtime.set_input("x", np.random.randn(2, 3, 224, 224).astype("float32"))
except Exception as e:
    print(f"Input shape error: {e}")
    # Fix: Use correct input shape
    runtime.set_input("x", np.random.randn(1, 3, 224, 224).astype("float32"))
```

---

## 33.6 Logging and Diagnostics

### 33.6.1 TVM_LOG_DEBUG Environment Variable

The `TVM_LOG_DEBUG` environment variable controls verbose debug logging from TVM's C++ backend. The syntax supports module-level filtering.

```bash
# Syntax: TVM_LOG_DEBUG="module_path=level[;module_path=level...]"

# Enable all debug logging (level 1 = verbose, 2 = very verbose)
export TVM_LOG_DEBUG=1

# Enable debug for specific modules
export TVM_LOG_DEBUG="tir/transform/*=1"

# Multiple modules
export TVM_LOG_DEBUG="tir/transform/*=1;relay/transform/*=2"

# Specific source file
export TVM_LOG_DEBUG="tir/transform/storage_rewrite.cc=2"
```

### 33.6.2 Logging Levels

TVM supports the following logging levels:

| Level | Name | Description |
|-------|------|-------------|
| 0 | DISABLED | No logging |
| 1 | INFO | Informational messages |
| 2 | DEBUG | Debug messages |
| 3 | VERBOSE | Verbose debug messages |
| 4 | TRACE | Trace-level messages (very detailed) |

```python
import tvm

# Set logging level programmatically
tvm.set_log_level("INFO")
tvm.set_log_level("DEBUG")
tvm.set_log_level("WARNING")
tvm.set_log_level("ERROR")
tvm.set_log_level("FATAL")
```

### 33.6.3 Custom Logging

You can implement custom logging handlers in Python to capture and process TVM log messages.

```python
import tvm
import logging

class TVMLogHandler(logging.Handler):
    """Custom handler for TVM log messages."""

    def __init__(self, log_file="tvm_debug.log"):
        super().__init__()
        self.log_file = log_file
        self.messages = []

    def emit(self, record):
        msg = self.format(record)
        self.messages.append(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

# Set up custom logging
handler = TVMLogHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
))
logger = logging.getLogger("tvm")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
```

---

## 33.7 Testing Strategies

### 33.7.1 Roundtrip Testing

Roundtrip testing verifies that an IR module can be serialized to TVMScript, re-parsed, and produce an equivalent module.

```python
import tvm
from tvm.script import tir as T
import numpy as np

@T.prim_func
def vector_add(
    A: T.Buffer((256,), "float32"),
    B: T.Buffer((256,), "float32"),
    C: T.Buffer((256,), "float32"),
) -> None:
    for i in range(256):
        with T.block("add"):
            vi = T.axis.spatial(256, i)
            C[vi] = A[vi] + B[vi]

# Roundtrip test: IRModule -> script -> parse -> compare
mod = tvm.ir.IRModule.from_expr(vector_add)

# Step 1: Serialize to TVMScript
script = mod.script()

# Step 2: Parse back
mod_roundtrip = tvm.ir.IRModule.from_expr(
    tvm.script.from_source(script)["main"]
)

# Step 3: Verify structural equality
assert tvm.ir.structural_equal(mod["main"], mod_roundtrip["main"])
print("Roundtrip test passed!")
```

### 33.7.2 Numerical Verification

Numerical verification ensures that compiled TVM functions produce the same results as reference implementations.

```python
import tvm
import numpy as np
from tvm.script import tir as T

def verify_numerical_correctness(tvm_func, ref_func, input_shapes, dtypes, tol=1e-5):
    """Verify numerical correctness of a TVM function against a reference."""
    # Generate random inputs
    inputs = []
    for shape, dtype in zip(input_shapes, dtypes):
        if "int" in dtype:
            inputs.append(np.random.randint(0, 10, size=shape).astype(dtype))
        else:
            inputs.append(np.random.randn(*shape).astype(dtype))

    # Compute reference output
    ref_output = ref_func(*inputs)

    # Compute TVM output
    tvm_inputs = [tvm.nd.array(x) for x in inputs]
    if isinstance(ref_output, np.ndarray):
        tvm_output = tvm.nd.array(np.zeros_like(ref_output))
        tvm_inputs.append(tvm_output)
        tvm_func(*tvm_inputs)
        result = tvm_output.numpy()
    else:
        result = tvm_func(*tvm_inputs)

    # Compare
    if isinstance(ref_output, np.ndarray):
        np.testing.assert_allclose(result, ref_output, rtol=tol, atol=tol)
        print(f"Numerical verification passed (tol={tol})")
    else:
        assert abs(result - ref_output) < tol

# Example usage: verify vector add
@T.prim_func
def vec_add(A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
            C: T.Buffer((1024,), "float32")) -> None:
    for i in range(1024):
        with T.block("add"):
            vi = T.axis.spatial(1024, i)
            C[vi] = A[vi] + B[vi]

def numpy_add(a, b):
    return a + b

# verify_numerical_correctness(vec_add, numpy_add, [(1024,), (1024,)], ["float32", "float32"])
```

### 33.7.3 Gradient Checking

For differentiable operations, gradient checking verifies that the computed gradients match numerical gradients.

```python
import tvm
import numpy as np

def numerical_gradient(func, inputs, epsilon=1e-4):
    """Compute numerical gradient using finite differences."""
    grads = []
    for idx, x in enumerate(inputs):
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'])
        while not it.finished:
            mi = it.multi_index
            old_val = x[mi]

            x[mi] = old_val + epsilon
            loss_plus = func(*[inp for inp in inputs])

            x[mi] = old_val - epsilon
            loss_minus = func(*[inp for inp in inputs])

            grad[mi] = (loss_plus - loss_minus) / (2 * epsilon)
            x[mi] = old_val
            it.iternext()

        grads.append(grad)
    return grads

# Usage with Relay
def check_gradient(relay_expr, input_shapes, dtypes):
    """Check that Relay's gradients match numerical gradients."""
    # Compile forward pass
    mod = tvm.IRModule.from_expr(relay_expr)
    # Use relay.gradient to compute analytical gradients
    grads = relay.gradient(relay_expr)
    # Compare with numerical gradients
    # ... (implementation depends on specific use case)
```

---

## 33.8 Advanced Debugging Tools

### 33.8.1 IR Comparison

When a pass changes the IR unexpectedly, comparing the IR before and after can pinpoint the issue.

```python
import tvm
from tvm import relay

def compare_ir(mod_before, mod_after):
    """Compare two IR modules and report differences."""
    funcs_before = set(mod_before.functions.keys())
    funcs_after = set(mod_after.functions.keys())

    added = funcs_after - funcs_before
    removed = funcs_before - funcs_after
    common = funcs_before & funcs_after

    if added:
        print(f"Functions added: {added}")
    if removed:
        print(f"Functions removed: {removed}")

    for name in common:
        f_before = mod_before[name]
        f_after = mod_after[name]
        if not tvm.ir.structural_equal(f_before, f_after):
            print(f"Function {name} changed:")
            print(f"  Before: {f_before}")
            print(f"  After:  {f_after}")
```

### 33.8.2 Memory Debugging

Memory-related issues can be debugged using TVM's object system and reference counting.

```python
import tvm
import gc

# Force garbage collection to check for reference leaks
gc.collect()

# Check TVM object reference counts
x = tvm.runtime.NDArray(np.zeros(100))
print(f"Reference count: {x.__tvm_object__.use_count()}")

# Use weak references to detect leaks
import weakref
arr = tvm.nd.array(np.zeros(100))
ref = weakref.ref(arr)
del arr
gc.collect()
assert ref() is None, "NDArray was not freed - potential leak"
```

### 33.8.3 Debug Builds

Building TVM in debug mode enables additional checks and assertions.

```bash
# CMake debug build configuration
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DUSE_PROFILER=ON \
      -DUSE_GRAPH_EXECUTOR_DEBUG=ON \
      ..

# Or with address sanitizer
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
      ..
```

Debug builds enable:
- Additional assertions in the C++ code.
- Bounds checking for array access.
- Memory leak detection with sanitizers.
- Detailed stack traces on crashes.

---

## 33.9 Summary

Effective debugging in TVM requires familiarity with:
- The error type hierarchy and what each error means.
- The diagnostic system for source-level error reporting.
- IR inspection tools (`show()`, `script()`) for examining intermediate results.
- `BasePyModule` for incremental development and verification.
- Logging configuration for detailed execution traces.
- Testing strategies (roundtrip, numerical, gradient) for ensuring correctness.

By combining these tools and strategies, you can efficiently diagnose and fix issues across all stages of the TVM compilation pipeline, from model import through code generation to runtime execution.
