# MLIR Python Bindings

## Installation

```bash
# Install from pip (if available)
pip install mlir

# Or build from source
cmake -DLLVM_ENABLE_PROJECTS=mlir \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python3) \
      -B build llvm-project/llvm
cmake --build build --target mlir-python
```

## Core API

### Context

```python
from mlir.ir import Context, Location, Module, InsertionPoint

ctx = Context()
ctx.allow_unregistered_dialects = True

with Location.unknown(ctx):
    module = Module.create()
```

### Location

```python
from mlir.ir import Location

loc = Location.unknown(ctx)
loc = Location.file("test.mlir", 4, 12, ctx)
loc = Location.fused([loc1, loc2], ctx)
```

### Types

```python
from mlir.ir import Type, IntegerType, F32Type, MemRefType, TensorType, VectorType, FunctionType

# Integer types
i1 = IntegerType.get_signless(1, ctx)
i32 = IntegerType.get_signless(32, ctx)
si32 = IntegerType.get_signed(32, ctx)
ui32 = IntegerType.get_unsigned(32, ctx)

# Float types
f16 = F16Type.get(ctx)
f32 = F32Type.get(ctx)
f64 = F64Type.get(ctx)
bf16 = BF16Type.get(ctx)

# MemRef
memref = MemRefType.get([10, 20], f32, ctx=ctx)
memref_dyn = MemRefType.get([-1, 20], f32, ctx=ctx)

# Tensor
tensor = TensorType.get([10, 20], f32, ctx=ctx)
tensor_dyn = TensorType.get([-1, 20], f32, ctx=ctx)

# Vector
vec = VectorType.get([4], f32, ctx=ctx)

# Function
func_type = FunctionType.get([i32, f32], [i32], ctx)

# Index
from mlir.ir import IndexType
index = IndexType.get(ctx)
```

### Attributes

```python
from mlir.ir import Attribute, IntegerAttr, FloatAttr, StringAttr, ArrayAttr, DictionaryAttr, TypeAttr

# Integer attribute
int_attr = IntegerAttr.get(i32, 42)

# Float attribute
float_attr = FloatAttr.get(f32, 3.14)

# String attribute
str_attr = StringAttr.get("hello", ctx)

# Array attribute
arr_attr = ArrayAttr.get([int_attr, str_attr], ctx)

# Dictionary attribute
dict_attr = DictionaryAttr.get({"name": str_attr, "value": int_attr}, ctx)

# Type attribute
type_attr = TypeAttr.get(i32)
```

### Operations

```python
from mlir.ir import Operation, OpView, InsertionPoint, Block

with Location.unknown(ctx):
    module = Module.create()

    with InsertionPoint(module.body):
        # Create operation using generic API
        op = Operation.create("arith.constant", results=[i32],
                              attributes={"value": IntegerAttr.get(i32, 42)})

        # Using dialect-specific Python bindings
        from mlir.dialects import arith, func
        value = arith.ConstantOp(i32, 42)
```

### Block and Region

```python
from mlir.ir import Block, Region

# Create block
block = Block.create_at_start(region, [i32, f32])

# Block arguments
arg0 = block.arguments[0]

# Insert at block
with InsertionPoint(block):
    # create operations
    pass
```

### Values

```python
from mlir.ir import Value, OpResult, BlockArgument

# Check value type
value_type = value.type

# Replace uses
value.replace_all_uses_with(new_value)
```

## Dialect Modules

### Func Dialect

```python
from mlir.dialects import func

with InsertionPoint(module.body):
    # Define function
    f = func.FuncOp("my_func", FunctionType.get([i32, f32], [i32]))
    with InsertionPoint(f.add_entry_block()):
        func.ReturnOp([f.arguments[0]])
```

### Arith Dialect

```python
from mlir.dialects import arith

c1 = arith.ConstantOp(i32, 42)
c2 = arith.ConstantOp(i32, 10)
result = arith.AddIOp(c1.result, c2.result)
```

### MemRef Dialect

```python
from mlir.dialects import memref

buf = memref.AllocOp(memref_type, [], [])
val = memref.LoadOp(f32, buf, [idx])
memref.StoreOp(val, buf, [idx])
memref.DeallocOp(buf)
```

### SCF Dialect

```python
from mlir.dialects import scf

# If operation
result = scf.IfOp(cond.type, cond, hasElse=True)
with InsertionPoint(result.then_block):
    scf.YieldOp([true_val])
with InsertionPoint(result.else_block):
    scf.YieldOp([false_val])
```

### Linalg Dialect

```python
from mlir.dialects import linalg

# Fill
linalg.FillOp(value, output)

# Matmul
linalg.MatmulOp(A, B, C)
```

## Pass Management

```python
from mlir.passmanager import PassManager

pm = PassManager(ctx)
pm.add("canonicalize")
pm.add("cse")
pm.add("one-shot-bufferize")
pm.run(module)
```

### Custom Pass Pipeline

```python
pm = PassManager(ctx)
pm.add("func.func(scf-for-loop-tiling{tile-size=32})")
pm.add("canonicalize")
pm.add("cse")
pm.run(module)
```

## Pattern Rewriting

```python
from mlir.rewrite import PatternRewriteWalker, RewritePattern

class MyPattern(RewritePattern):
    def match(self, op):
        if op.name == "arith.addi":
            return True
        return None

    def rewrite(self, op, rewriter):
        # rewrite logic
        rewriter.replace_op(op, new_op)

walker = PatternRewriteWalker([MyPattern()])
walker.rewrite(module)
```

## Execution Engine

```python
from mlir.execution_engine import ExecutionEngine

# JIT compile and execute
engine = ExecutionEngine(module)
result = engine.invoke("main")
```

## Module I/O

```python
# Parse from string
module = Module.parse("""
  module {
    func.func @main() -> i32 {
      %c = arith.constant 42 : i32
      return %c : i32
    }
  }
""", ctx)

# Print to string
print(module)

# Write to file
with open("output.mlir", "w") as f:
    f.write(str(module))
```
