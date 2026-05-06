# 18 - Tracing and Jaxpr Internals

## Overview

Tracing is JAX's mechanism for converting Python functions into a functional intermediate representation (jaxpr). Understanding tracing is essential for debugging and extending JAX.

---

## 1. The Tracing Process

### What happens during tracing

```
1. JAX wraps each input in a Tracer object
2. The Python function executes with Tracers as inputs
3. Each JAX operation on Tracers is recorded
4. The recorded operations form a Jaxpr
5. Tracer objects are unwrapped from outputs
```

### Example: Step by step

```python
import jax
import jax.numpy as jnp

def f(x):
    y = x * 2
    z = jnp.sin(y)
    return z + 1

# Trace the function
jaxpr = jax.make_jaxpr(f)(jnp.ones(3))
print(jaxpr)
# { lambda ; a:f32[3]. let
#     b:f32[3] = mul a 2.0
#     c:f32[3] = sin b
#     d:f32[3] = add c 1.0
#   in (d,) }
```

What happened:
1. `x` became `Traced<ShapedArray(float32[3])>`
2. `x * 2` recorded as `mul` primitive → new Tracer `b`
3. `jnp.sin(y)` recorded as `sin` primitive → new Tracer `c`
4. `z + 1` recorded as `add` primitive → new Tracer `d`
5. Tracer `d` is unwrapped and returned

---

## 2. Tracer Types

### MainTrace stack

JAX maintains a stack of traces:

```python
# Each transformation pushes a trace
with jax.jit:
    # JaxprTrace is active
    with jax.vmap:
        # BatchTrace is active (on top of JaxprTrace)
        pass
```

### Trace implementations

| Trace class | Purpose | Interpreter |
|---|---|---|
| `JaxprTrace` | Capture jaxpr | `partial_eval.py` |
| `JVPTrace` | Forward-mode AD | `ad.py` |
| `BatchTrace` | Vectorization | `batching.py` |
| `EvalTrace` | Direct execution | `core.py` |
| `ShardingTrace` | Distributed execution | `sharding_impls.py` |

---

## 3. Tracing with make_jaxpr

### Basic usage

```python
jaxpr = jax.make_jaxpr(f)(*args)
jaxpr = jax.make_jaxpr(f, static_argnums=...)(*args)
```

### Return value structure

```python
class ClosedJaxpr:
    jaxpr: Jaxpr           # The core jaxpr
    consts: list[Any]      # Closed-over constants

class Jaxpr:
    constvars: list[Var]   # Variable names for constants
    invars: list[Var]      # Input variables
    outvars: list[Var]     # Output variables
    eqns: list[JaxprEqn]   # Equations

class JaxprEqn:
    invars: list           # Input vars/literals
    outvars: list[Var]     # Output vars
    primitive: Primitive   # Operation name
    params: dict           # Extra parameters
    source_info: SourceInfo
```

### Inspecting a jaxpr

```python
jaxpr = jax.make_jaxpr(f)(jnp.ones(3))

# Access the underlying Jaxpr
j = jaxpr.jaxpr
print(f"Inputs: {j.invars}")      # [a:f32[3]]
print(f"Outputs: {j.outvars}")    # [d:f32[3]]
print(f"Equations: {len(j.eqns)}")  # 3

for eqn in j.eqns:
    print(f"  {eqn.primitive.name}: {eqn.invars} -> {eqn.outvars}")
    if eqn.params:
        print(f"    params: {eqn.params}")
```

---

## 4. Side Effects and Tracing

### What is NOT captured

```python
state = []

def f(x):
    state.append(x)         # NOT captured
    print(f"x = {x}")      # NOT captured (prints Tracer)
    y = x + 1
    return y

jaxpr = jax.make_jaxpr(f)(jnp.ones(3))
# No append or print in jaxpr!
```

### What IS captured

```python
def f(x):
    # Python math → JAX op
    y = x + 1           # Captured: add primitive
    z = jnp.sin(y)      # Captured: sin primitive
    w = z.shape[0]      # NOT captured (static attribute)
    return z

# Pure JAX ops are captured; Python control flow is traced through
```

### Python conditionals

```python
def f(x, flag):
    if flag:              # flag is static (Python bool)
        return x + 1
    else:
        return x - 1

# flag=True traces only the first branch
jaxpr_true = jax.make_jaxpr(f)(jnp.ones(3), True)

# flag=False traces only the second branch
jaxpr_false = jax.make_jaxpr(f)(jnp.ones(3), False)
```

### Traced conditionals

```python
def f(x):
    if x > 0:           # x > 0 is a Traced bool → error!
        return x + 1

# Fix: use jnp.where or lax.cond
def f(x):
    return jnp.where(x > 0, x + 1, x - 1)
```

---

## 5. Multi-level Tracing

### JIT + grad (two levels of tracing)

```python
@jax.jit
def f(x):
    return jax.grad(lambda y: y**3)(x)

# Trace 1: JIT traces the outer function
#   - Inside, grad does its own trace for the inner function
# Trace 2: grad traces lambda y: y**3
# Result: A single jaxpr with both transformations applied
```

### JIT + vmap + grad

```python
@jax.jit
def train_step(params, X, Y):
    def batch_loss(params, X, Y):
        return jnp.mean(jax.vmap(single_loss, in_axes=(None, 0, 0))(params, X, Y))
    return jax.grad(batch_loss)(params, X, Y)

# Three traces happen:
# 1. JIT traces the outer function
# 2. Inside, vmap traces single_loss
# 3. Inside that, grad traces through the vmapped function
```

---

## 6. Abstract Evaluation

Each primitive must define how to compute output shapes/dtypes from input shapes/dtypes:

```python
from jax._src.core import ShapedArray

# Abstract evaluation rule for a custom primitive
@my_prim.def_abstract_eval
def my_prim_abstract(x_aval, y_aval):
    # Compute output shape/dtype without data
    out_shape = x_aval.shape  # Example: output same shape as input
    out_dtype = jnp.result_type(x_aval.dtype, y_aval.dtype)
    return ShapedArray(out_shape, out_dtype)
```

### For multiple results

```python
@my_prim.def_abstract_eval
def my_prim_abstract(*avals, **params):
    return (ShapedArray(...), ShapedArray(...))
```

---

## 7. Jaxpr Evaluation

### Running a jaxpr directly

```python
from jax._src.core import eval_jaxpr

# Evaluate a jaxpr with concrete inputs
jaxpr = jax.make_jaxpr(f)(jnp.ones(3))
result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, jnp.array([1.0, 2.0, 3.0]))
```

### Jaxpr as data

```python
# Jaxprs are serializable data structures
import dataclasses

# All fields are plain Python objects (lists, dicts, strings)
# Can be serialized, transmitted, stored
```

---

## 8. Partial Evaluation

JAX can partially evaluate a function, separating traced and static parts:

```python
from jax._src.partial_eval import trace_to_jaxpr

# This is what happens internally during JIT
# The function is partially evaluated:
# - Static computations are executed in Python
# - Traced computations are recorded in jaxpr
```

### Known vs unknown

```python
# During partial evaluation, each variable is "known" or "unknown"
# Known variables have concrete values (evaluated in Python)
# Unknown variables are abstract (recorded in jaxpr)
```

---

## 9. Debugging Tracing

### Print during trace

```python
def f(x):
    # This only prints during tracing, not at runtime
    print(f"Tracing with x = {x}")
    return x + 1

f_jit = jax.jit(f)
f_jit(jnp.ones(3))  # Prints: "Tracing with x = Traced<ShapedArray(float32[3])>"
f_jit(jnp.ones(3))  # Does NOT print (uses cached compilation)
f_jit(jnp.ones(5))  # Prints (re-traces due to different shape)
```

### Runtime printing

```python
def f(x):
    jax.debug.print("x = {}", x)  # Prints at runtime with actual values
    return x + 1
```

### Checking trace count

```python
trace_count = 0

def f(x):
    global trace_count
    trace_count += 1  # Only increments during tracing
    return x + 1

f_jit = jax.jit(f)
f_jit(jnp.ones(3))  # trace_count = 1
f_jit(jnp.ones(3))  # trace_count = 1 (cached)
f_jit(jnp.ones(5))  # trace_count = 2 (re-trace)
```

### Tracer leak detection

```python
with jax.check_tracer_leaks():
    # Raises error if any tracer escapes
    @jax.jit
    def f(x):
        return x + 1
    result = f(jnp.ones(3))
```

---

## 10. Advanced: Custom Trace

```python
from jax._src.core import Trace, Tracer

class MyTrace(Trace):
    """Custom trace for logging all operations."""

    def process_primitive(self, primitive, tracers, params):
        print(f"Op: {primitive.name}, params: {params}")
        # Delegate to the underlying trace
        out = primitive.bind(*[t.val for t in tracers], **params)
        if primitive.multiple_results:
            return [MyTracer(self, o) for o in out]
        return MyTracer(self, out)

class MyTracer(Tracer):
    def __init__(self, trace, val):
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        return raise_to_shaped(get_aval(self.val))

    @property
    def trace(self):
        return self._trace
```

---

## 11. Jaxpr Optimization

### Dead code elimination

```python
from jax._src.core import optimize

# JAX automatically eliminates dead code in jaxprs
def f(x):
    y = jnp.sin(x)   # y is computed but not used
    return x + 1

jaxpr = jax.make_jaxpr(f)(jnp.ones(3))
# The sin operation may or may not appear
# (depends on optimization level)
```

### Constant folding

```python
def f(x):
    y = jnp.add(2.0, 3.0)  # Constant expression
    return x + y

# XLA will fold 2+3 into 5 during compilation
```

### Fusion

```python
def f(x):
    y = jnp.sin(x)
    z = jnp.cos(x)
    return y + z

# XLA fuses sin, cos, and add into a single kernel
```
