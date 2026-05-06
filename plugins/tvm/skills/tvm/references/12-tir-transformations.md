# 12 — TIR Transformations and Lowering Passes

## Overview

TIR transformation passes convert `tirx::PrimFunc` from high-level scheduling form to low-level target-specific code. These passes are the bridge between scheduling decisions and actual code generation.

Two modules provide TIR transforms:
- **`tirx.transform`** — Core IR lowering passes
- **`s_tir.transform`** — Scheduling-related transforms

---

## Core Lowering Passes (tirx.transform)

### FlattenBuffer
Converts multi-dimensional buffer access to 1D pointer access.
- Essential step before code generation
- Converts `A[i, j]` to `A[i * stride_j + j]`
- Computes buffer strides

```python
mod = tirx.transform.FlattenBuffer()(mod)
```

### LowerIntrin
Lowers high-level intrinsics to target-specific implementations.
- Platform-dependent lowering
- Lowers math functions (exp, log, sin, etc.)
- Lowers hardware intrinsics

```python
mod = tirx.transform.LowerIntrin()(mod)
```

### LowerCustomDatatypes
Lowers custom/user-defined data types to hardware-supported types.

```python
mod = tirx.transform.LowerCustomDatatypes()(mod)
```

### LowerDeviceKernelLaunch
Lowers device kernel launch calls for GPU targets.
- Generates host-side kernel launch code
- Sets up grid/block dimensions

```python
mod = tirx.transform.LowerDeviceKernelLaunch()(mod)
```

### VectorizeLoop
Converts loops marked as `Vectorized` to vector instructions.
- Respects vector length from target
- Generates SIMD operations

```python
mod = tirx.transform.VectorizeLoop()(mod)
```

### StorageRewrite
Optimizes storage allocation and memory planning:
- Storage reuse between non-overlapping lifetimes
- In-place operations
- Memory layout optimization

```python
mod = tirx.transform.StorageRewrite()(mod)
```

### UnrollLoop
Unrolls loops marked as `Unrolled`:
- Configurable: `auto_max_step`, `auto_max_depth`, `explicit_unroll`
- Controlled via PassContext:

```python
with tvm.transform.PassContext(config={
    "tirx.UnrollLoop": {"auto_max_step": 10}
}):
    mod = tirx.transform.UnrollLoop()(mod)
```

### Simplify
Simplifies TIR expressions and statements using arithmetic analyzer:
- Constant folding
- Algebraic simplification
- Dead code elimination

```python
mod = tirx.transform.Simplify()(mod)
```

### RemoveNoOp
Removes no-operation statements and dead code:
- Empty loops
- Trivial assignments
- Unused allocations

```python
mod = tirx.transform.RemoveNoOp()(mod)
```

### HoistIfThenElse
Moves conditional statements outside loops when condition is loop-invariant.

```python
mod = tirx.transform.HoistIfThenElse()(mod)
```

### PartitionLoop
Partitions loops based on conditions for better optimization.

```python
mod = tirx.transform.PartitionLoop()(mod)
```

### InjectCopyIntrin
Injects copy intrinsics for optimized memory transfers.

```python
mod = tirx.transform.InjectCopyIntrin()(mod)
```

### InjectDoubleBuffer
Adds double buffering for overlapping computation and data transfer.

```python
mod = tirx.transform.InjectDoubleBuffer()(mod)
```

### InjectVirtualThread
Virtual threading support — maps virtual threads to physical threads.

```python
mod = tirx.transform.InjectVirtualThread()(mod)
```

### LiftAttrScope
Lifts attribute scope to outer level when possible.

```python
mod = tirx.transform.LiftAttrScope()(mod)
```

### LoopPartition
Partitions loop for better optimization opportunities.

```python
mod = tirx.transform.LoopPartition()(mod)
```

### NarrowDataType
Narrows data types where possible to reduce memory usage.

```python
mod = tirx.transform.NarrowDataType()(mod)
```

### ConvertBlocksToOpaque
Converts SBlock to opaque blocks, removing scheduling annotations.
- Used when no further scheduling is needed
- Prepares for final lowering

```python
mod = tirx.transform.ConvertBlocksToOpaque()(mod)
```

### LiftThreadBinding
Lifts thread binding annotations to outer scope for GPU codegen.

```python
mod = tirx.transform.LiftThreadBinding()(mod)
```

### DecorateDeviceScope
Adds device scope decoration required for GPU kernel generation.

```python
mod = tirx.transform.DecorateDeviceScope()(mod)
```

### MergeDynamicSharedMemoryAllocations
Merges multiple shared memory allocations to reduce overhead on GPU.

```python
mod = tirx.transform.MergeDynamicSharedMemoryAllocations()(mod)
```

### BindTarget
Binds target information to PrimFunc.

```python
mod = tirx.transform.BindTarget(target)(mod)
```

### MakePackedAPI
Creates packed function API for runtime calling:
- Generates wrapper with PackedFunc calling convention
- Handles type-erased argument passing

```python
mod = tirx.transform.MakePackedAPI()(mod)
```

### MakeUnpackedAPI
Creates unpacked function API with typed arguments.

```python
mod = tirx.transform.MakeUnpackedAPI()(mod)
```

### VerifyMemory
Verifies memory access patterns — checks for invalid access.

```python
mod = tirx.transform.VerifyMemory()(mod)
```

### InstrumentBoundCheckers
Adds runtime bound checking instrumentation for debugging.

```python
mod = tirx.transform.InstrumentBoundCheckers()(mod)
```

---

## s_tir.transform Passes

These passes operate on scheduled TIR programs:

- **RenormalizeSplitPattern** — renormalize after split scheduling
- **FuseOpsByPattern** (TIR-level) — fuse TIR operations based on patterns
- **ApplyPass** — apply a pass conditionally

---

## Standard Lowering Pipeline

The typical lowering order from scheduled TIR to codegen-ready TIR:

```
1. Simplify            — Clean up after scheduling
2. VectorizeLoop        — Apply vectorization
3. InjectDoubleBuffer   — Add double buffering (optional)
4. StorageRewrite       — Optimize memory layout
5. FlattenBuffer        — Flatten multi-dim to 1D
6. LowerIntrin          — Lower intrinsics to target
7. LowerCustomDatatypes — Lower custom types
8. DecorateDeviceScope  — Add device scope (GPU)
9. MergeDynamicSharedMemoryAllocations  — Merge shared mem (GPU)
10. MakePackedAPI       — Create runtime API
```

### GPU Lowering Pipeline
```python
def gpu_lower_pipeline():
    return tvm.transform.Sequential([
        tirx.transform.Simplify(),
        tirx.transform.VectorizeLoop(),
        tirx.transform.StorageRewrite(),
        tirx.transform.FlattenBuffer(),
        tirx.transform.LowerIntrin(),
        tirx.transform.DecorateDeviceScope(),
        tirx.transform.MergeDynamicSharedMemoryAllocations(),
        tirx.transform.MakePackedAPI(),
    ])
```

### CPU Lowering Pipeline
```python
def cpu_lower_pipeline():
    return tvm.transform.Sequential([
        tirx.transform.Simplify(),
        tirx.transform.VectorizeLoop(),
        tirx.transform.StorageRewrite(),
        tirx.transform.FlattenBuffer(),
        tirx.transform.LowerIntrin(),
        tirx.transform.MakePackedAPI(),
    ])
```

---

## Pass Ordering Rules

### Dependencies
1. **FlattenBuffer** must run before code generation
2. **LowerIntrin** depends on target information
3. **VectorizeLoop** must run after scheduling (vectorized annotation)
4. **StorageRewrite** should run before FlattenBuffer
5. **DecorateDeviceScope** must run for GPU targets
6. **MakePackedAPI** should be one of the last passes

### Order Sensitivity
- Simplify can run at any point (beneficial early and late)
- StorageRewrite benefits from running after scheduling
- FlattenBuffer should run after all structural transforms

---

## Custom TIR Passes

### Function-level Pass
```python
@tvm.tirx.transform.prim_func_pass(opt_level=2, name="my_tir_pass")
def my_tir_pass(func, mod, ctx):
    # Transform func
    return new_func

# Apply
mod = my_tir_pass(mod)
```

### Module-level Pass
```python
@tvm.transform.module_pass(opt_level=2, name="my_module_pass")
def my_module_pass(mod, ctx):
    # Transform mod
    return new_mod
```

---

## Debugging TIR Passes

### Print IR at Each Stage
```python
# Print IR before and after pass
mod.show()  # Print current IR as TVMScript
mod = my_pass(mod)
mod.show()  # Print transformed IR
```

### Pass Tracing
```python
with tvm.transform.PassContext(traceback=True):
    mod = pipeline(mod)
```

### Individual Pass Application
Apply passes one at a time to isolate issues:
```python
mod = tirx.transform.Simplify()(mod)
mod = tirx.transform.FlattenBuffer()(mod)
mod = tirx.transform.LowerIntrin()(mod)
```
