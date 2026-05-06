# Chapter 30: JAX Export and StableHLO Serialization

## 30.1 Overview

JAX provides the ability to export compiled programs to a portable, serialized
format based on StableHLO (Stable High-Level Operations). This enables:

- **Model deployment**: Export trained models for serving without JAX dependency
- **Cross-framework interoperability**: Share models between ML frameworks
- **Version stability**: Run models on different JAX/XLA versions
- **Ahead-of-time compilation**: Compile once, deploy many times

The primary API is `jax.export` (previously `jax.experimental.export`).

---

## 30.2 jax.export API

### 30.2.1 Basic Export

```python
import jax
import jax.numpy as jnp
from jax import export

# Define a function to export
@jax.jit
def predict(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Create example inputs for shape/dtype inference
key = jax.random.key(0)
params = (jnp.ones((784, 10)), jnp.ones((10,)))
x = jnp.ones((1, 784))

# Export the function
exported = export.export(predict)(
    params, x
)

print(type(exported))  # <class 'jax.export.Exported'>
```

### 30.2.2 The Exported Object

An `Exported` object contains:

```python
# The serialized StableHLO module (MLIR bytecode)
mlir_module = exported.mlir_module()
print(mlir_module)  # Prints the MLIR/StableHLO representation

# The module as serialized bytes
serialized = exported.serialize()
print(type(serialized))  # bytes

# Input/output shapes and dtypes
print(exported.in_tree)    # PyTree structure of inputs
print(exported.out_tree)   # PyTree structure of outputs

# Lowering and compilation details
print(exported.lowering)   # The lowered MLIR module
```

### 30.2.3 Serialization and Deserialization

```python
# Serialize to bytes
serialized_bytes = exported.serialize()

# Save to file
with open("model.jax_export", "wb") as f:
    f.write(serialized_bytes)

# Load from file
with open("model.jax_export", "rb") as f:
    loaded_bytes = f.read()

# Deserialize
loaded_exported = export.deserialize(loaded_bytes)

# Call the loaded function
result = jax.jit(loaded_exported.call)(params, x)
print(result.shape)  # (1, 10)
```

---

## 30.3 Shape Polymorphism

### 30.3.1 The Problem

By default, exported functions have fixed input shapes. A model compiled for
batch size 32 cannot process batch size 64 without recompilation. Shape
polymorphism allows exporting functions that work with multiple input shapes.

### 30.3.2 Symbolic Shapes

JAX export supports symbolic dimension variables that represent unknown
dimensions at export time:

```python
import jax
import jax.numpy as jnp
from jax import export
from jax.export import shape_poly

# Define a function that works for any batch size
def model(params, x):
    w, b = params
    return jax.nn.relu(jnp.dot(x, w) + b)

# Export with symbolic batch dimension 'b'
b = export.symbolic_shape("b")
params = (jnp.ones((784, 10)), jnp.ones((10,)))
x_example = jnp.ones((1, 784))  # Just needs to be compatible

# Export with polymorphic shapes
exported_poly = export.export(
    jax.jit(model),
    # Specify that first input's leading dimension is symbolic
)(params, x_example)

# Can also specify shapes more explicitly
exported_v2 = export.export(jax.jit(model))(
    params,
    jax.ShapeDtypeStruct((b, 784), jnp.float32),
)
```

### 30.3.3 Shape Constraints

When using symbolic shapes, you can add constraints:

```python
from jax.export import shape_poly

# Symbolic dimensions
b, s = export.symbolic_shape("b, s")

# Some operations require dimension relationships to be known
# e.g., reshape from (b, s) to (b * s,) requires knowing b * s
def reshape_fn(x):
    return x.reshape((-1,))

# Export with constraint that total elements = b * s
exported_reshape = export.export(
    jax.jit(reshape_fn),
)(jnp.ones((4, 8)))
```

### 30.3.4 Shape Polymorphism Rules

Rules for valid shape-polymorphic exports:

1. **Addition/subtraction of symbolic dims**: `b + 1`, `b - s` (if known >= 0)
2. **Multiplication by constants**: `b * 4`
3. **Division by constants**: `b // 4`
4. **Min/Max with constants**: `min(b, 128)`

```python
# Valid shape expressions
dim1 = export.symbolic_shape("b")
dim2 = export.symbolic_shape("s")

# These work in most contexts:
# b + s  (sum of two symbolic dims)
# b * 4  (symbolic times constant)
# b // 8 (floor division by constant)
```

### 30.3.5 Limitations of Shape Polymorphism

Not all operations are compatible with symbolic shapes:

```python
# PROBLEMATIC: reshape to exact shape with symbolic dims
def bad_reshape(x):
    # x has shape (b, 10) -- can't reshape to fixed (100,) without knowing b
    return x.reshape((100,))  # Error if b != 10

# WORKAROUND: use -1 for inferred dimension
def good_reshape(x):
    return x.reshape((-1,))  # Flattens, works for any b

# PROBLEMATIC: conditional logic based on symbolic shape
def bad_conditional(x):
    if x.shape[0] > 10:  # Can't evaluate at compile time with symbolic dim
        return x + 1
    return x

# WORKAROUND: use jax.lax.cond with runtime values
def good_conditional(x, threshold):
    return jax.lax.cond(
        x.shape[0] > threshold,
        lambda x: x + 1,
        lambda x: x,
        x,
    )
```

---

## 30.4 Forward and Backward Compatibility

### 30.4.1 StableHLO Compatibility Guarantees

StableHLO provides versioning guarantees:

| Compatibility | Description |
|---|---|
| Forward | A program exported with an older version can run on a newer runtime |
| Backward | A program exported with a newer version can run on an older runtime (limited) |

```python
# Check the StableHLO version of an exported module
exported = export.export(jax.jit(lambda x: x + 1))(jnp.ones(3))
print(f"StableHLO version: {exported.stablehlo_version()}")

# The target version can be specified during export
exported_v1 = export.export(
    jax.jit(lambda x: x + 1),
    # Optional: target a specific StableHLO version for compatibility
)(jnp.ones(3))
```

### 30.4.2 Version Selection

```python
# Export targeting a specific compatibility window
exported_for_serving = export.export(
    jax.jit(model),
    # The serialized module will be compatible with runtimes
    # supporting this StableHLO version or later
)(example_inputs)
```

### 30.4.3 Checking Compatibility

```python
# Check if an exported module is compatible with the current runtime
import jax

try:
    # Attempt to load and run
    loaded = export.deserialize(serialized_bytes)
    # Check compatibility
    print(f"Module is compatible with current JAX version")
except Exception as e:
    print(f"Compatibility error: {e}")
```

### 30.4.4 Serialization Format Details

```python
# The serialization format includes:
# 1. StableHLO MLIR module (the computation graph)
# 2. Shape information (input/output shapes and dtypes)
# 3. Module metadata (name, version, etc.)

# Access individual components
exported = export.export(jax.jit(lambda x: x * 2))(jnp.ones(5))

# MLIR module as text
print(exported.mlir_module_text())

# MLIR module as bytecode (compact binary format)
bytecode = exported.mlir_module_bytecode()

# Full serialization (includes metadata)
full_serialized = exported.serialize()
```

---

## 30.5 Exporting with Side Effects and State

### 30.5.1 Pure Functions

Exported functions must be pure (no side effects). State must be passed
explicitly:

```python
# WRONG: Using global state (not exportable)
counter = jnp.array(0)

def bad_increment(x):
    global counter
    counter = counter + 1  # Side effect!
    return x + counter

# RIGHT: Pass state explicitly
def good_increment(state, x):
    new_state = state + 1
    return new_state, x + new_state

state_example = jnp.array(0)
x_example = jnp.ones(5)

exported = export.export(jax.jit(good_increment))(state_example, x_example)
```

### 30.5.2 Random State

Random keys must be passed as explicit arguments:

```python
def stochastic_model(key, params, x):
    # key must be an argument, not captured from closure
    noise = jax.random.normal(key, x.shape)
    return jax.nn.relu(jnp.dot(x + noise, params))

key = jax.random.key(42)
params = jnp.ones((784, 10))
x = jnp.ones((1, 784))

exported = export.export(jax.jit(stochastic_model))(key, params, x)
```

---

## 30.6 Exporting Transformed Functions

### 30.6.1 JIT-Compiled Functions

```python
# Export works with jax.jit
@jax.jit
def model(x):
    return jnp.dot(x, x.T)

exported = export.export(model)(jnp.ones((10, 10)))
```

### 30.6.2 Grad (Differentiated Functions)

```python
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

# Export the gradient function
grad_fn = jax.grad(loss_fn)

params = jnp.ones((5,))
x = jnp.ones((3, 5))
y = jnp.ones((3,))

exported_grad = export.export(jax.jit(grad_fn))(params, x, y)
```

### 30.6.3 Vmap (Vectorized Functions)

```python
def single_predict(params, x):
    return jnp.dot(params, x)

batched_predict = jax.vmap(single_predict, in_axes=(None, 0))

params = jnp.ones((10, 5))
x_batch = jnp.ones((32, 5))  # Batch of 32

exported_vmap = export.export(jax.jit(batched_predict))(params, x_batch)
```

### 30.6.4 Combined Transformations

```python
def loss(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

# Gradient of batched loss
batched_loss = lambda params, x, y: jnp.mean(
    jax.vmap(loss, in_axes=(None, 0, 0))(params, x, y)
)
grad_fn = jax.jit(jax.grad(batched_loss))

params = jnp.ones((5,))
x = jnp.ones((8, 3, 5))
y = jnp.ones((8, 3,))

exported = export.export(grad_fn)(params, x, y)
```

---

## 30.7 Exporting for Specific Backends

### 30.7.1 GPU Export

```python
# Export for GPU execution
with jax.default_device(jax.devices("gpu")[0]):
    exported_gpu = export.export(
        jax.jit(model, backend="gpu")
    )(example_inputs)
```

### 30.7.2 TPU Export

```python
# Export for TPU execution
with jax.default_device(jax.devices("tpu")[0]):
    exported_tpu = export.export(
        jax.jit(model, backend="tpu")
    )(example_inputs)
```

### 30.7.3 CPU Export (Portable)

```python
# Export for CPU (most portable)
exported_cpu = export.export(
    jax.jit(model)
)(example_inputs)
```

---

## 30.8 Integration with Serving Systems

### 30.8.1 Exporting for TensorFlow Serving

```python
# The StableHLO module can be wrapped in a SavedModel for TF Serving
# (requires tensorflow or tensorflow-serving integration)

# Step 1: Export the JAX function
exported = export.export(jax.jit(model))(example_inputs)

# Step 2: Get the StableHLO module
stablehlo_module = exported.mlir_module()

# Step 3: Wrap in a TensorFlow SavedModel (using tfx/tensorflow bridge)
# The exact integration depends on your serving infrastructure
```

### 30.8.2 Exporting for ONNX

```python
# JAX can export to StableHLO, which can be converted to ONNX
# through the jax2onnx bridge or similar tools

# Note: Direct ONNX export from JAX is available through community tools
# The primary path is JAX -> StableHLO -> Target runtime
```

### 30.8.3 Direct C++ Execution

```python
# Exported StableHLO modules can be loaded and executed by the
# PJRT C API or XLA runtime directly:

# 1. Serialize the module
serialized = exported.serialize()

# 2. In C++ (pseudocode):
# auto module = StableHLO::Deserialize(serialized);
# auto executable = client->compile(module);
# auto result = executable->run(inputs);
```

---

## 30.9 Advanced Topics

### 30.9.1 Custom Objects in Export

```python
# Custom PyTree classes need proper serialization support
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class ModelParams:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def tree_flatten(self):
        return ((self.weights, self.biases), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# Now ModelParams can be used as input/output of exported functions
def model(params, x):
    return jnp.dot(x, params.weights) + params.biases

params = ModelParams(jnp.ones((5, 3)), jnp.ones((3,)))
x = jnp.ones((1, 5))
exported = export.export(jax.jit(model))(params, x)
```

### 30.9.2 Platform-Dependent Lowering

```python
# You can control which platform optimizations are included
exported_portable = export.export(
    jax.jit(model),
    # Platform-specific lowering options
)(example_inputs)

# For maximum portability, avoid platform-specific primitives
# jax.lax platform-independent ops are preferred
```

### 30.9.3 Multiple Outputs

```python
def multi_output_model(params, x):
    hidden = jnp.dot(x, params["w1"]) + params["b1"]
    hidden = jax.nn.relu(hidden)
    output = jnp.dot(hidden, params["w2"]) + params["b2"]
    return output, hidden  # Return both final and intermediate

params = {
    "w1": jnp.ones((784, 256)),
    "b1": jnp.ones((256,)),
    "w2": jnp.ones((256, 10)),
    "b2": jnp.ones((10,)),
}
x = jnp.ones((1, 784))

exported = export.export(jax.jit(multi_output_model))(params, x)
# exported.out_tree shows both outputs
```

### 30.9.4 Exporting with Donated Arguments

```python
# Some arguments can be "donated" (their buffer can be reused)
# This is useful for in-place updates during training

def update_fn(params, grads):
    return jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)

params = {"w": jnp.ones((5, 3)), "b": jnp.ones((3,))}
grads = {"w": jnp.ones((5, 3)) * 0.1, "b": jnp.ones((3,)) * 0.1}

# Export with donation of params buffer
exported = export.export(
    jax.jit(update_fn, donate_argnums=0)
)(params, grads)
```

---

## 30.10 Error Handling and Debugging

### 30.10.1 Common Export Errors

```python
# Error: Non-hashable inputs
# Solution: Use only arrays and valid PyTree leaves as inputs

# Error: Dynamic shapes not supported
# Solution: Use shape polymorphism with symbolic shapes

# Error: Side effects detected
# Solution: Make function pure, pass state explicitly

# Error: Unsupported operation
# Solution: Rewrite using supported ops, check jax.lax alternatives
```

### 30.10.2 Debugging Export Issues

```python
# Step 1: Ensure the function works with jax.jit first
@jax.jit
def model(x):
    return jnp.dot(x, x.T)

result = model(jnp.ones((3, 3)))  # Should work

# Step 2: Try export
try:
    exported = export.export(model)(jnp.ones((3, 3)))
except Exception as e:
    print(f"Export error: {e}")

    # Lower to see the HLO
    lowered = model.lower(jnp.ones((3, 3)))
    print(lowered.as_text())  # Show the HLO text

# Step 3: Inspect the exported module
if exported:
    print(exported.mlir_module_text())  # Show StableHLO MLIR
```

### 30.10.3 Validating Exported Functions

```python
def validate_export(exported, test_inputs):
    """Validate that an exported function produces the same results."""
    # Run the original function (via exported.call)
    result_exported = exported.call(*test_inputs)

    # Compare with expected output
    # (assuming you have a reference implementation)
    print(f"Output shape: {jax.tree.map(lambda x: x.shape, result_exported)}")
    print(f"Output dtype: {jax.tree.map(lambda x: x.dtype, result_exported)}")

    return result_exported

# Validate
result = validate_export(exported, (jnp.ones((3, 3)),))
```

---

## 30.11 Complete Example: Exporting a Neural Network

```python
import jax
import jax.numpy as jnp
from jax import export

# Define a simple neural network
def init_params(key, layers):
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(layers[:-1], layers[1:])):
        key, k1, k2 = jax.random.split(key, 3)
        params[f"w{i}"] = jax.random.normal(k1, (fan_in, fan_out)) * 0.01
        params[f"b{i}"] = jnp.zeros((fan_out,))
    return params

def predict(params, x):
    """Multi-layer neural network with ReLU activations."""
    num_layers = len(params) // 2
    h = x
    for i in range(num_layers):
        h = jnp.dot(h, params[f"w{i}"]) + params[f"b{i}"]
        if i < num_layers - 1:
            h = jax.nn.relu(h)
    return h

def predict_with_softmax(params, x):
    """Network output with softmax for classification."""
    logits = predict(params, x)
    return jax.nn.softmax(logits, axis=-1)

# Initialize and export
key = jax.random.key(0)
params = init_params(key, [784, 256, 128, 10])
x_example = jnp.ones((1, 784))

# Export 1: Basic prediction (fixed batch size)
exported_fixed = export.export(jax.jit(predict_with_softmax))(params, x_example)
print(f"Fixed export output shapes: {exported_fixed.out_tree}")

# Serialize and save
with open("model_fixed.jax_export", "wb") as f:
    f.write(exported_fixed.serialize())

# Load and verify
with open("model_fixed.jax_export", "rb") as f:
    loaded = export.deserialize(f.read())

# Run with different batch size (requires shape polymorphism)
# For now, verify with same shape
result = loaded.call(params, jnp.ones((1, 784)))
print(f"Predictions shape: {result.shape}")
print(f"Predictions sum to 1: {jnp.sum(result):.4f}")

# Export 2: Gradient function for inference-time adaptation
grad_fn = jax.grad(lambda p, x, y: jnp.mean((predict_with_softmax(p, x) - y) ** 2))

y_example = jnp.zeros((1, 10)).at[0, 3].set(1.0)
exported_grad = export.export(jax.jit(grad_fn))(params, x_example, y_example)

with open("model_grad.jax_export", "wb") as f:
    f.write(exported_grad.serialize())

print("Export complete: model_fixed.jax_export, model_grad.jax_export")
print(f"Module size: {len(exported_fixed.serialize())} bytes")
```

---

## 30.12 Best Practices

### 30.12.1 When to Export

| Scenario | Export? | Alternative |
|---|---|---|
| Production serving | Yes | N/A |
| Cross-version deployment | Yes | N/A |
| Cross-framework sharing | Yes | ONNX, SavedModel |
| Research/development | No | Use jax.jit directly |
| Interactive notebooks | No | Use eager mode |
| Performance profiling | No | Use jax.jit + profiler |

### 30.12.2 Tips for Successful Exports

1. **Test with jax.jit first**: If it doesn't JIT-compile, it won't export
2. **Avoid Python control flow**: Use `jax.lax.cond`, `jax.lax.scan`, etc.
3. **Use explicit state**: Pass all state as arguments, not closures
4. **Keep it pure**: No side effects, no global mutation
5. **Specify shapes early**: Use `jax.ShapeDtypeStruct` for abstract inputs
6. **Version your exports**: Track which JAX/StableHLO version produced them
7. **Validate after loading**: Always run sanity checks on deserialized modules
8. **Use shape polymorphism for variable batch sizes**: Avoid re-exporting for each size

---

## 30.13 API Reference Summary

```python
# Core API
jax.export.export(fn)                  # Returns an exporter callable
jax.export.deserialize(bytes)          # Load from serialized bytes
jax.export.symbolic_shape(spec)        # Create symbolic dimension variables

# Exported object methods
exported.serialize()                   # Serialize to bytes
exported.mlir_module()                 # Get MLIR module
exported.mlir_module_text()            # Get MLIR as text
exported.mlir_module_bytecode()        # Get MLIR as bytecode
exported.call(*args)                   # Call the exported function
exported.in_tree                       # Input PyTree structure
exported.out_tree                      # Output PyTree structure
```
