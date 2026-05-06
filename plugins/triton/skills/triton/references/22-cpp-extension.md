# Chapter 22: C++ Extension (`_C/libtriton`)

The Python C++ extension provides high-performance bindings for MLIR operations.

## Module Structure

The `libtriton` module is defined in `python/src/main.cc`:

```
libtriton/
├── ir              # IR types, operations, builder
├── passes          # Compilation passes
├── llvm            # LLVM backend operations
├── interpreter     # Interpreter operations
├── linear_layout   # Linear layout utilities
├── specialize      # Native type specialization
├── gluon_ir        # Gluon IR extensions
├── env_vars        # Environment variable access
└── backends/       # Backend-specific modules
    ├── nvidia
    └── amd
```

## IR Module (`ir`)

### TritonOpBuilder

Custom MLIR OpBuilder that tracks source locations:

```cpp
class TritonOpBuilder {
    MLIRContext *context;
    OpBuilder builder;
    Location lastLoc;
    bool lineInfoEnabled;

    template<typename OpTy, typename... Args>
    OpTy create(Args&&... args);

    template<typename OpTy, typename... Args>
    Value createOrFold(Args&&... args);
};
```

### Type Bindings

All Triton types are exposed to Python:

```python
from triton._C.libtriton import ir

# Integer types
i1 = ir.type.INT1
i8 = ir.type.INT8
i32 = ir.type.INT32
i64 = ir.type.INT64

# Float types
f16 = ir.type.FP16
bf16 = ir.type.BF16
f32 = ir.type.FP32
f64 = ir.type.FP64

# Pointer type
ptr_f32 = ir.type.ptr(f32)

# Function type
func_type = ir.type.function([f32, f32], [f32])
```

### Operation Bindings

```python
# Create operations through the builder
builder = ir.builder(context)

# Create constants
val = builder.create_splat(f32, [128], 1.0)

# Create arithmetic
result = builder.create_add(a, b)

# Create load/store
data = builder.create_load(ptr, mask)
builder.create_store(ptr, data, mask)
```

## Passes Module (`passes`)

All compilation passes exposed to Python via macros:

```cpp
// Macro for creating pass wrappers
ADD_PASS_WRAPPER_0(name, builder)
ADD_PASS_WRAPPER_1(name, builder, ty0)
ADD_PASS_WRAPPER_2(name, builder, ty0, ty1)
```

### Pass Categories

```python
from triton._C.libtriton import passes

# Common passes
passes.sccp(module)
passes.symbol_dce(module)
passes.inliner(module)
passes.canonicalizer(module)
passes.cse(module)
passes.licm(module)

# TTIR passes
passes.ttir_combine_ops(module)
passes.ttir_loop_unroll(module)
passes.convert_triton_to_triton_gpu(module, num_warps, num_ctas)

# TTGIR passes
passes.ttgir_coalesce(module)
passes.ttgir_optimize_thread_locality(module)
passes.ttgir_accelerate_matmul(module, capability)
passes.ttgir_pipeline(module, num_stages)
passes.ttgir_allocate_shared_memory(module)

# Conversion passes
passes.convert_triton_gpu_to_llvm(module)
passes.convert_scf_to_cf(module)
passes.convert_cf_to_llvm(module)
```

## LLVM Module (`llvm`)

LLVM backend operations:

```python
from triton._C.libtriton import llvm

# Create target machine
target = llvm.create_target_machine(arch, features)

# Convert MLIR to LLVM IR
llvm_module = llvm.to_module(mlir_module, target)

# Optimize LLVM module
llvm.optimize_module(llvm_module, opt_level)

# Translate to assembly
asm = llvm.translate_to_asm(llvm_module, target)

# Translate to object code
obj = llvm.translate_to_obj(llvm_module, target)
```

## Interpreter Module (`interpreter`)

CPU interpreter for atomic and memory operations:

```python
from triton._C.libtriton import interpreter

# Atomic operations (CPU simulation)
result = interpreter.atomic_add(ptr, value)
result = interpreter.atomic_cas(ptr, compare, value)
```

### Supported Atomic Ops

| Operation | Integer | Float |
|-----------|---------|-------|
| ADD | Yes | FADD |
| AND | Yes | - |
| OR | Yes | - |
| XOR | Yes | - |
| MAX | Yes | - |
| MIN | Yes | - |
| XCHG | Yes | - |
| CAS | Yes | - |

## Specialize Module (`specialize`)

Native type specialization for kernel arguments:

```python
from triton._C.libtriton import specialize

# Specialize Python objects to C++ types
specialized = specialize.specialize_arg(arg, is_constexpr=False)
```

Handles:
- Integer types (int, bool)
- Float types
- Tensor arguments
- Tensor descriptors
- JIT callables
- Tuples

## LinearLayout Module (`linear_layout`)

Linear algebra for memory layout representation:

```python
from triton._C.libtriton import linear_layout

# Create layouts
identity = linear_layout.identity_1d(size, dim_name)
strided = linear_layout.strided_1d(size, stride, dim_name)

# Operations
composed = layout_a.compose(layout_b)
inverted = layout.invert()
pseudo = layout.pseudoinvert()

# Properties
layout.is_surjective()
layout.is_injective()
layout.is_invertible()
```

## Gluon IR Module (`gluon_ir`)

Extended IR for Gluon programming model:

- GluonOpBuilder with additional operations
- Layout types (AutoLayout, BlockedLayout, etc.)
- Memory descriptor operations
- Async copy operations
- Barrier operations
- Tensor memory operations
- Architecture-specific operations (NVIDIA, AMD)
