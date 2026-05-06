# 11 — TIR Expressions and Statements

## Overview

TensorIR uses a tree-structured IR with expressions (produce values) and statements (perform actions). All nodes are subclasses of `runtime.Object` and accessible from Python.

---

## Expressions

Expressions produce values. They are the building blocks of computation in TIR.

### Variables

| Type | Description | Example |
|------|-------------|---------|
| `tirx.Var(name, dtype)` | Named variable | `i = tirx.Var("i", "int32")` |
| `tirx.SizeVar(name, dtype)` | Symbolic size variable | `n = tirx.SizeVar("n", "int64")` |
| `tirx.BufferVar(buffer, index)` | Buffer data pointer variable | — |

### Constants

| Type | Description | TVMScript |
|------|-------------|-----------|
| `IntImm(dtype, value)` | Integer constant | `T.int32(5)`, `T.int64(128)` |
| `FloatImm(dtype, value)` | Float constant | `T.float32(3.14)`, `T.float64(0.0)` |
| `StringImm(value)` | String constant | — |

### Type Casting

| Type | Description | TVMScript |
|------|-------------|-----------|
| `Cast(dtype, value)` | Type conversion | `T.cast(x, "float16")` |

### Arithmetic

| Type | Description | TVMScript |
|------|-------------|-----------|
| `Add(a, b)` | Addition | `a + b` |
| `Sub(a, b)` | Subtraction | `a - b` |
| `Mul(a, b)` | Multiplication | `a * b` |
| `Div(a, b)` | Division | `a / b` |
| `Mod(a, b)` | Modulo | `a % b` |
| `FloorDiv(a, b)` | Floor division | `T.floordiv(a, b)` |
| `FloorMod(a, b)` | Floor modulo | `T.floormod(a, b)` |
| `CeilDiv(a, b)` | Ceiling division | `T.ceildiv(a, b)` |
| `Min(a, b)` | Minimum | `T.min(a, b)` |
| `Max(a, b)` | Maximum | `T.max(a, b)` |

### Comparisons

| Type | Description | TVMScript |
|------|-------------|-----------|
| `EQ(a, b)` | Equal | `a == b` |
| `NE(a, b)` | Not equal | `a != b` |
| `LT(a, b)` | Less than | `a < b` |
| `LE(a, b)` | Less or equal | `a <= b` |
| `GT(a, b)` | Greater than | `a > b` |
| `GE(a, b)` | Greater or equal | `a >= b` |

### Logical

| Type | Description | TVMScript |
|------|-------------|-----------|
| `And(a, b)` | Logical AND | `a and b` |
| `Or(a, b)` | Logical OR | `a or b` |
| `Not(a)` | Logical NOT | `not a` |
| `Select(cond, t, f)` | Conditional select | `T.if_then_else(cond, t, f)` |

### Memory Access

| Type | Description |
|------|-------------|
| `BufferLoad(buffer, indices)` | Read from buffer: `A[i, j]` |
| `ProducerLoad(producer, indices)` | Read from producer store |

### Vector Operations

| Type | Description |
|------|-------------|
| `Ramp(base, stride, lanes)` | Generate vector of values: `base, base+stride, ...` |
| `Broadcast(value, lanes)` | Broadcast scalar to vector |
| `Shuffle(vectors, indices)` | Shuffle vector elements |

### Function Calls

| Type | Description |
|------|-------------|
| `Call(dtype, op, args)` | Call an intrinsic or function |

### Let Binding

| Type | Description | TVMScript |
|------|-------------|-----------|
| `Let(var, value, body)` | Let binding | `T.let(var, value, body)` |

---

## Statements

Statements perform actions. They form the body of PrimFunc.

### Loops — `For`

```python
# Loop with kind
for i in T.serial(128):           # Serial — sequential
    ...
for i in T.parallel(128):         # Parallel — parallelizable
    ...
for i in T.vectorized(8):         # Vectorized — SIMD
    ...
for i in T.unroll(4):             # Unrolled — unroll factor
    ...
```

ForKind values:
| Kind | Description |
|------|-------------|
| `Serial` | Sequential execution |
| `Parallel` | Parallelizable loop |
| `Vectorized` | Vector/SIMD instructions |
| `Unrolled` | Loop unrolling |

### SBlock — `SBlock`

The fundamental scheduling block:
```python
with T.sblock("name"):
    vi = T.axis.spatial(range, loop_var)
    # computation
```

### SBlockRealize
Realizes a block with specific iter values. Created implicitly by TVMScript.

### Memory Operations

#### BufferStore
```python
A[i, j] = value  # Write to buffer
```

#### BufferRealize
Realizes a buffer region for computation. Created during lowering.

### Allocation

#### Allocate
```python
with T.allocate([128], "float32") as buf:
    # buf is a Var pointing to allocated memory
    ...
```

#### AllocateConst
```python
# Allocate constant data
with T.allocate_const([1.0, 2.0, 3.0], "float32", [3]) as buf:
    ...
```

### Control Flow

#### IfThenElse
```python
if T.likely(condition):
    # then case
else:
    # else case
```

#### While
```python
while condition:
    # body
```

### Other Statements

| Type | Description | TVMScript |
|------|-------------|-----------|
| `AssertStmt(cond, msg, body)` | Runtime assertion | `T.assert(condition, message)` |
| `Evaluate(value)` | Evaluate expression for side effects | `T.evaluate(expr)` |
| `SeqStmt(stmts)` | Sequence of statements | Implicit in TVMScript |
| `LetStmt(var, value, body)` | Let binding | `T.let(var, value)` |
| `AttrStmt(node, key, value, body)` | Attribute annotation | Various `T.attr` |

---

## Data Layout

The `tvm.tirx.data_layout` module handles layout descriptions:

### Layout String
```
"NCHW"     — Batch, Channel, Height, Width
"NHWC"     — Batch, Height, Width, Channel
"NCHW4c"   — NCHW with channel split by 4
"NC"       — 2D layout
```

### Layout Operations
```python
import tvm.tirx as T
from tvm.tirx import data_layout

layout = data_layout.Layout("NCHW")
# Access dimensions
n, c, h, w = layout["N"], layout["C"], layout["H"], layout["W"]
```

---

## Buffer Node

A Buffer node describes a multi-dimensional array:

| Field | Type | Description |
|-------|------|-------------|
| `data` | Var | Pointer to data |
| `shape` | Array<IntImm> | Shape of buffer |
| `dtype` | DataType | Element data type |
| `strides` | Array<IntImm> | Strides (can be None for compact) |
| `elem_offset` | PrimExpr | Offset of first element |
| `name` | String | Buffer name |
| `scope` | String | Memory scope ("global", "shared", etc.) |

### Buffer Creation in TVMScript
```python
# Parameter buffer
A: T.Buffer((128, 128), "float32")

# Allocated buffer
Y = T.alloc_buffer((128, 128), dtype="float32")

# Match buffer from handle
X = T.match_buffer(handle, (M, K), "float32")
```

---

## Expression Simplification

TVM provides powerful arithmetic simplification through the `arith` module:

```python
import tvm
from tvm import arith, tirx

analyzer = arith.Analyzer()

x = tirx.Var("x", "int32")
y = tirx.Var("y", "int32")

# Simplify expressions
result = analyzer.simplify((x + y) - y)  # → x
result = analyzer.simplify(x * 0 + y)    # → y
result = analyzer.simplify(x * 1 + 0)    # → x
```

---

## Common Patterns

### Reduction Pattern
```python
# Sum reduction
for k in range(K):
    with T.sblock("reduce"):
        vk = T.axis.reduce(K, k)
        with T.init():
            result[vi] = T.float32(0)
        result[vi] = result[vi] + data[vi, vk]
```

### Pipeline Pattern
```python
# Producer-consumer across two blocks
for i in range(N):
    with T.sblock("producer"):
        vi = T.axis.spatial(N, i)
        temp[vi] = A[vi] * T.float32(2.0)

for i in range(N):
    with T.sblock("consumer"):
        vi = T.axis.spatial(N, i)
        B[vi] = temp[vi] + C[vi]
```
