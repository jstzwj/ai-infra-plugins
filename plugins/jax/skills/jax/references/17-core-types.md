# 17 - Core Types and Abstract Values

## Overview

JAX's type system revolves around abstract values that represent array properties (shape, dtype) without concrete data. This chapter covers the core type hierarchy, abstract values, and how JAX uses them internally.

---

## 1. Abstract Value Hierarchy

```
AbstractValue
├── ShapedArray
│   ├── DShapedArray (dynamic shapes)
│   └── ConcreteArray (has concrete values)
├── AbstractToken
├── CoreUnit
└── UnshapedArray (legacy)
```

### AbstractValue

Base class for all JAX abstract types. Properties:
- `aval` — The abstract value itself
- Used during tracing to represent input/output types

### ShapedArray

The most common abstract value:

```python
from jax._src.core import ShapedArray

# Properties
sa = ShapedArray(shape=(3, 4), dtype=jnp.float32)
sa.shape      # (3, 4)
sa.dtype      # float32
sa.ndim       # 2
sa.size       # 12
sa.strides    # Computed from shape and dtype
```

### ConcreteArray

An abstract value with known concrete data:

```python
from jax._src.core import ConcreteArray

ca = ConcreteArray(jnp.array([1.0, 2.0, 3.0]))
ca.val  # The concrete array value
ca.shape  # (3,)
ca.dtype  # float32
```

---

## 2. Var and Literal

### Var (Variable)

Represents a named variable in a jaxpr:

```python
from jax._src.core import Var

# Properties
var = Var(count=0, suffix='', aval=ShapedArray((3,), jnp.float32))
var.count   # 0 (unique identifier)
var.suffix  # '' (for debugging)
var.aval    # ShapedArray((3,), float32)
str(var)    # 'a' (auto-generated name from count)
```

### Literal

Represents a constant value:

```python
from jax._src.core import Literal

lit = Literal(val=3.14)
lit.val     # 3.14
lit.aval    # ConcreteArray(3.14)
```

---

## 3. Primitive

A named operation in JAX's IR:

```python
from jax._src.core import Primitive

# Creating a primitive
my_add_p = Primitive('my_add')

# Registering implementation
@my_add_p.def_impl
def my_add_impl(x, y):
    return x + y

# Registering abstract evaluation
@my_add_p.def_abstract_eval
def my_add_abstract(x_aval, y_aval):
    return ShapedArray(x_aval.shape, x_aval.dtype)
```

### Primitive properties

```python
prim = Primitive('add')
prim.name           # 'add'
prim.multiple_results  # False (returns single value)
```

---

## 4. Jaxpr

JAX's intermediate representation:

```python
from jax._src.core import Jaxpr, JaxprEqn

# Structure
class Jaxpr:
    constvars: list[Var]     # Constants
    invars: list[Var]        # Inputs
    outvars: list[Var]       # Outputs
    eqns: list[JaxprEqn]     # Equations (operations)

class JaxprEqn:
    invars: list[Var | Literal]  # Inputs
    outvars: list[Var]           # Outputs
    primitive: Primitive         # Operation
    params: dict                 # Extra parameters
    source_info: SourceInfo      # Debug info
```

### Viewing jaxprs

```python
import jax, jax.numpy as jnp

def f(x):
    return jnp.sin(x) + jnp.cos(x)

jaxpr = jax.make_jaxpr(f)(jnp.ones(3))
jaxpr.jaxpr.invars    # Input variables
jaxpr.jaxpr.outvars   # Output variables
jaxpr.jaxpr.eqns      # List of equations
jaxpr.jaxpr.constvars # Constants
```

---

## 5. Trace and Tracer

### Trace

A context for tracing operations:

```python
from jax._src.core import Trace

# Key method: process_primitive
# Called for each primitive during tracing
```

### Tracer

A wrapper that records operations:

```python
from jax._src.core import Tracer

# Properties
tracer.aval    # AbstractValue (shape, dtype)
tracer.trace   # The Trace that created this Tracer
```

### Trace types

| Trace | Used by |
|---|---|
| `JaxprTrace` | `jax.make_jaxpr`, `jax.jit` |
| `JVPTrace` | `jax.jvp`, `jax.grad` (forward) |
| `BatchTrace` | `jax.vmap` |
| `EvalTrace` | Normal execution |

---

## 6. ShapeDtypeStruct

User-facing abstract value representation:

```python
from jax import ShapeDtypeStruct

# Create without data
x_aval = ShapeDtypeStruct(shape=(3, 4), dtype=jnp.float32)
x_aval.shape   # (3, 4)
x_aval.dtype   # float32
x_aval.ndim    # 2
x_aval.size    # 12

# Use for abstract tracing
jaxpr = jax.make_jaxpr(f)(x_aval)
```

---

## 7. DTypes

### Supported dtypes

| JAX dtype | NumPy equivalent | Size |
|---|---|---|
| `jnp.bool_` | `np.bool_` | 1 byte |
| `jnp.int8` | `np.int8` | 1 byte |
| `jnp.int16` | `np.int16` | 2 bytes |
| `jnp.int32` | `np.int32` | 4 bytes |
| `jnp.int64` | `np.int64` | 8 bytes |
| `jnp.uint8` | `np.uint8` | 1 byte |
| `jnp.uint16` | `np.uint16` | 2 bytes |
| `jnp.uint32` | `np.uint32` | 4 bytes |
| `jnp.uint64` | `np.uint64` | 8 bytes |
| `jnp.float16` | `np.float16` | 2 bytes |
| `jnp.float32` | `np.float32` | 4 bytes |
| `jnp.float64` | `np.float64` | 8 bytes |
| `jnp.bfloat16` | N/A | 2 bytes |
| `jnp.complex64` | `np.complex64` | 8 bytes |
| `jnp.complex128` | `np.complex128` | 16 bytes |

### bfloat16

```python
# JAX-specific brain float16
x_bf16 = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)
# Same exponent range as float32, lower mantissa precision
# Ideal for deep learning
```

### dtype promotion

```python
# JAX follows NumPy promotion rules (configurable)
jax.config.update("jax_numpy_dtype_promotion", "standard")

# Check promotion
jnp.result_type(jnp.float16, jnp.float32)  # float32
jnp.promote_types(jnp.int32, jnp.float64)  # float64
```

---

## 8. Array class

### `jax.Array`

The fundamental JAX array type:

```python
x = jnp.ones(3)
type(x)  # <class 'jax.Array'>

# Properties
x.shape       # (3,)
x.dtype       # float32
x.ndim        # 1
x.size        # 3
x.device()    # Device(id=0, process_index=0)
x.devices()   # {Device(id=0)}
x.is_fully_addressable  # True
x.is_fully_replicated    # True

# Methods
x.block_until_ready()  # Wait for async computation
x.to_py()              # Convert to Python/NumPy
x.addressable_shards   # List of Shard objects
x.global_shards        # List of Shard objects (all processes)
```

---

## 9. RaiseToShaped

Utility to convert abstract values:

```python
from jax._src.core import raise_to_shaped

# Strip concrete values, keep shape/dtype
concrete = ConcreteArray(jnp.array([1.0, 2.0]))
shaped = raise_to_shaped(concrete)
# shaped is ShapedArray((2,), float32)
```

---

## 10. Custom Abstract Values

```python
from jax._src.core import AbstractValue

class MyAbstractToken(AbstractValue):
    """Custom abstract value type."""
    def __init__(self):
        self._hash = id(self)

    @property
    def shape(self):
        return ()

    def __eq__(self, other):
        return isinstance(other, MyAbstractToken)

    def __hash__(self):
        return self._hash

    def _broadcast(self, other, **kwargs):
        return self
```

---

## 11. Type System Rules

### Shape compatibility

```python
# Same shape → compatible
ShapedArray((3, 4), float32) == ShapedArray((3, 4), float32)  # True

# Different shape → different abstract value
ShapedArray((3, 4), float32) == ShapedArray((4, 3), float32)  # False

# Different dtype → different
ShapedArray((3, 4), float32) == ShapedArray((3, 4), float16)  # False
```

### Weak types

JAX has "weak" types for Python scalars:

```python
# Python int → weakly typed
jnp.add(1, jnp.array([1, 2], dtype=jnp.int32))
# Result dtype: int32 (weak int promotes to int32, not int64)

# Enable 64-bit
jax.config.update("jax_enable_x64", True)
```
