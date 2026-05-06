# XLA Aliasing

This document provides comprehensive documentation about XLA's aliasing mechanism, which enables in-place updates and memory optimization by allowing input and output buffers to share the same device memory.

## Table of Contents

- [Overview](#overview)
- [Compile-Time Aliasing](#compile-time-aliasing)
- [Runtime Aliasing](#runtime-aliasing)
- [Use Cases](#use-cases)

## Overview

Aliasing in XLA refers to the mechanism by which output buffers can share the same device memory as input buffers. This is a critical optimization that enables:

1. **In-place updates**: Modifying a buffer in place without allocating a new buffer, reducing memory usage and avoiding unnecessary copies.

2. **Memory optimization**: Reducing peak memory usage by reusing memory that would otherwise be held by both the input and output.

3. **Zero-copy semantics**: Passing data through the computation graph without any physical memory copies.

XLA's aliasing system has two components:

1. **Compile-time aliasing**: Declared during compilation via `XlaBuilder::SetUpAlias`, informing the compiler that specific inputs and outputs should share memory.

2. **Runtime aliasing**: Managed during execution via buffer donation, where input buffers are "donated" to the computation and potentially reused for outputs.

## Compile-Time Aliasing

### XlaBuilder::SetUpAlias API

The compile-time aliasing is set up through the `XlaBuilder` API, which is used to construct XLA computations:

```cpp
class XlaBuilder {
 public:
  // Set up an alias between a parameter and the output.
  //
  // Args:
  //   param_number: The parameter number (0-indexed) that should be aliased.
  //   param_index: The index within the parameter tuple (for tuple parameters).
  //   output_index: The index within the output tuple where the alias applies.
  //
  // After calling this, the compiler knows that the specified output buffer
  // can reuse the memory of the specified input buffer.
  Status SetUpAlias(
      const ShapeIndex& param_index,
      int64_t param_number,
      const ShapeIndex& output_index);
};
```

#### Usage Example

```cpp
// Create an XLA builder
XlaBuilder builder("in_place_add");

// Create parameters
auto x = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {1024}), "x");
auto y = Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {1024}), "y");

// Compute: x = x + y (in-place update of x)
auto result = Add(x, y);

// Set up the alias: output[0] shares memory with parameter 0
// This tells the compiler that the output can be written into
// the same memory as parameter 0.
TF_CHECK_OK(builder.SetUpAlias(
    /*param_index=*/{},         // Root of parameter 0
    /*param_number=*/0,         // Parameter 0 (x)
    /*output_index=*/{}));      // Root of output
```

### input_output_alias Format

The aliasing information is encoded in the HLO module as `input_output_alias` entries. These appear in the HLO module text format:

```
HloModule in_place_add, input_output_alias={{0}: (0, {})}

ENTRY main {
  %p0 = f32[1024] parameter(0)
  %p1 = f32[1024] parameter(1)
  ROOT %add = f32[1024] add(%p0, %p1)
}
```

The `input_output_alias` format is:

```
input_output_alias={{output_index}: (param_number, param_index), ...}
```

Where:
- `output_index`: The shape index within the output where the alias applies. Use `{}` for the root, `{0}` for the first tuple element, etc.
- `param_number`: The parameter number (0-indexed) whose buffer is being aliased.
- `param_index`: The shape index within the parameter. Use `{}` for the root, `{0}` for the first tuple element, etc.

#### Multiple Aliases

For outputs that are tuples, multiple aliases can be specified:

```
HloModule multi_alias, input_output_alias={{}: (0, {}), {1}: (1, {})}

ENTRY main {
  %p0 = f32[1024] parameter(0)
  %p1 = f32[1024] parameter(1)
  %add = f32[1024] add(%p0, %p1)
  ROOT %tuple = (f32[1024], f32[1024]) tuple(%p0, %add)
}
```

In this example:
- Output `{}` (the entire output tuple) aliases with parameter 0. Wait, this is incorrect for a tuple output. Let me correct:

```
HloModule multi_alias, input_output_alias={{0}: (0, {}), {1}: (1, {})}

ENTRY main {
  %p0 = f32[1024] parameter(0)
  %p1 = f32[1024] parameter(1)
  %add = f32[1024] add(%p0, %p1)
  ROOT %tuple = (f32[1024], f32[1024]) tuple(%p0, %add)
}
```

In this example:
- Output `{0}` (the first tuple element) aliases with parameter 0.
- Output `{1}` (the second tuple element) aliases with parameter 1.

#### Nested Tuple Aliases

For deeply nested tuples:

```
input_output_alias={{0, 1}: (2, {3, 0})}
```

This means:
- The element at output shape index `{0, 1}` shares memory with the element at parameter 2's shape index `{3, 0}`.

### Example HLO Module with Aliasing

#### Simple In-Place Update

```
HloModule inplace_update, input_output_alias={{}: (0, {})}

ENTRY main {
  %x = f32[1024] parameter(0)
  %y = f32[1024] parameter(1)
  ROOT %result = f32[1024] add(%x, %y)
}
```

The `input_output_alias={{}: (0, {})}` declares that the output (root of the output) can reuse the buffer of parameter 0 (root of parameter 0).

After compilation:
- XLA's buffer assignment will allocate only one buffer for both `%x` and `%result`.
- The `add` operation writes directly into the buffer that was passed as `%x`.
- At runtime, if the caller donates the buffer for `%x`, no copy is needed.

#### Tuple Output with Partial Aliasing

```
HloModule partial_alias,
    input_output_alias={{0}: (0, {}), {2}: (1, {})}

ENTRY main {
  %x = f32[1024] parameter(0)
  %y = f32[1024] parameter(1)
  %z = f32[512] parameter(2)
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%add, %add)
  %reduced = f32[512] reduce(%mul, %z), dimensions={0}, to_apply=add
  ROOT %output = (f32[1024], f32[1024], f32[512]) tuple(%add, %mul, %reduced)
}
```

Here:
- Output `{0}` (`%add`) aliases with parameter 0 (`%x`).
- Output `{2}` (`%reduced`) aliases with parameter 1 (`%y`).
- Output `{1}` (`%mul`) has no alias and gets its own buffer.

Note: Output `{2}` aliases with parameter 1, but `%reduced` depends on `%mul` which depends on `%add` which may alias with `%x`. The compiler must verify that the aliasing is safe (i.e., the aliased buffer is not read after it is written).

## Runtime Aliasing

### LocalClient::RunAsync API

At runtime, aliasing is managed through the `LocalClient` API:

```cpp
class LocalClient {
 public:
  // Execute a compiled executable with optional buffer donation.
  //
  // ExecutionInput supports buffer donation: when an input buffer is donated,
  // XLA can reuse its memory for outputs that are aliased to that input.
  StatusOr<std::unique_ptr<ScopedShapedBuffer>> RunAsync(
      const Executable& executable,
      absl::Span<ExecutionInput> arguments,
      HloExecutionProfile* profile = nullptr);
};
```

### ExecutionInput and MaybeOwningDeviceMemory

`ExecutionInput` wraps the input buffers and tracks whether each buffer can be donated:

```cpp
class ExecutionInput {
 public:
  // Create an ExecutionInput with a non-donatable buffer.
  ExecutionInput(ShapedBuffer shaped_buffer);

  // Create an ExecutionInput with a potentially donatable buffer.
  ExecutionInput(Shape shape,
                 absl::Span<MaybeOwningDeviceMemory> buffers);

  // Set whether a specific buffer can be donated.
  void SetBuffer(const ShapeIndex& index,
                 MaybeOwningDeviceMemory buffer);

  // Get the shaped buffer for execution.
  const ShapedBuffer& ShapedBuffer() const;

  // Mark all buffers as donatable.
  void MarkAllDonatable();
};
```

`MaybeOwningDeviceMemory` represents a device memory buffer that may or may not be owned:

```cpp
class MaybeOwningDeviceMemory {
 public:
  // Create a non-owning reference (buffer cannot be donated)
  static MaybeOwningDeviceMemory NonOwning(se::DeviceMemoryBase mem);

  // Create an owning reference (buffer can be donated)
  static MaybeOwningDeviceMemory Owning(se::OwningDeviceMemory mem);

  // Check if the buffer is owned (donatable)
  bool HasOwnership() const;
};
```

### Buffer Donation Mechanism

The buffer donation process works as follows:

1. **Compile time**: The compiler identifies aliasing opportunities via `SetUpAlias`. It records which output buffers can reuse which input buffers.

2. **Execution time**:
   a. The caller creates `ExecutionInput` objects, potentially marking input buffers as donatable.
   b. XLA checks whether the donated buffers match the declared aliasing.
   c. If a donated buffer matches an aliased output, XLA writes the output directly into the donated buffer.
   d. If a donated buffer does not match an aliased output (or the caller did not donate), XLA allocates a new buffer for the output and copies the input as needed.

3. **Post-execution**:
   a. The caller receives the output buffers.
   b. Donated buffers are no longer valid (their memory has been reused).
   c. Non-donated input buffers remain valid.

```
Before execution:
  Input buffer 0: [1, 2, 3, 4]  (donatable)
  Input buffer 1: [5, 6, 7, 8]  (not donatable)

During execution (with alias: output -> input 0):
  Input buffer 0: [1, 2, 3, 4]  -> overwritten in place
  Input buffer 1: [5, 6, 7, 8]  -> read only

After execution:
  Output buffer: [6, 8, 10, 12]  (same memory as former input buffer 0)
  Input buffer 0: INVALID (memory was donated/reused)
  Input buffer 1: [5, 6, 7, 8]  (still valid)
```

### Copy-Protection When Buffer Not Donated

If the compiler expects an alias but the caller does not donate the buffer, XLA must ensure correctness by copying the input:

1. **Buffer is donatable**: XLA writes the output directly into the input buffer. No copy needed. This is the fast path.

2. **Buffer is not donatable**: XLA must:
   a. Allocate a new buffer for the output.
   b. Copy the input data to the new buffer.
   c. Perform the computation using the copied data.
   d. This ensures the original input buffer is not modified.

This copy-protection mechanism ensures correctness but adds overhead. To get optimal performance, callers should donate buffers whenever possible.

```cpp
// Example: Donating a buffer in JAX
import jax
import jax.numpy as jnp

@jax.jit(donate_argnums=(0,))
def update_in_place(x, y):
    # The buffer for x is donated, and the output reuses x's memory.
    # After this function returns, the buffer for x is invalid.
    return x + y

x = jnp.ones((1024,))
y = jnp.ones((1024,)) * 2

result = update_in_place(x, y)
# x is now invalid! Do not use x after donation.
# result is [3, 3, 3, ...] stored in what was x's memory.
```

## Use Cases

### In-Place Updates (increment p++)

The primary use case for aliasing is in-place updates. Consider a parameter update in a training loop:

```python
# Training loop without aliasing (allocates new memory each step)
@jax.jit
def train_step(params, grads, lr):
    new_params = jax.tree.map(
        lambda p, g: p - lr * g,
        params, grads
    )
    return new_params

# Training loop with aliasing (reuses params memory)
@jax.jit(donate_argnums=(0,))
def train_step_inplace(params, grads, lr):
    new_params = jax.tree.map(
        lambda p, g: p - lr * g,
        params, grads
    )
    return new_params
```

With the `donate_argnums=(0,)` annotation:
- The memory for `params` is donated to the computation.
- The output `new_params` reuses the same memory.
- No additional memory allocation is needed for the parameter update.
- Peak memory usage is reduced by the size of the parameters.

### Memory Optimization

Aliasing reduces peak memory usage in several scenarios:

#### Weight Updates in Training

In a typical training loop, each step computes:
```
new_weights = weights - learning_rate * gradients
```

Without aliasing:
```
Memory = |weights| + |gradients| + |new_weights| = 2 * |weights| + |gradients|
```

With aliasing:
```
Memory = |weights| + |gradients| = |weights| + |gradients|
```

The output `new_weights` reuses the memory of `weights`, saving `|weights|` bytes.

#### Scan/Loop Accumulator Pattern

When using `jax.lax.scan` or `jax.lax.fori_loop`, the loop carry variable can be updated in place:

```python
@jax.jit(donate_argnums=(0,))
def scan_loop(carry, xs):
    # The carry is donated and reused across iterations
    # This reduces memory from O(iterations * carry_size) to O(carry_size)
    new_carry = carry + xs
    return new_carry, new_carry

carry_init = jnp.zeros((1024,))
xs = jnp.ones((100, 1024))

final_carry, intermediates = jax.lax.scan(scan_loop, carry_init, xs)
```

#### Buffer Swapping

Aliasing enables efficient buffer swapping in alternating computations:

```python
@jax.jit(donate_argnums=(0, 1))
def swap_and_compute(a, b):
    # Compute new values that reuse a and b's memory
    new_a = b * 2
    new_b = a + 1
    return new_a, new_b
```

### Limitations and Considerations

1. **Shape compatibility**: The aliased output must have the same shape and element type as the input parameter. You cannot alias an f32[1024] output with an f16[1024] input.

2. **Data dependency safety**: The compiler verifies that the aliasing is safe. If an aliased input is read after its buffer would be overwritten by the output, the compiler rejects the alias.

3. **No partial donation**: For tuple parameters, you cannot donate individual elements; the entire tuple must be donated or not.

4. **Donation is permanent**: Once a buffer is donated, it cannot be used again. The caller must ensure that donated buffers are not accessed after the computation.

5. **Cross-device aliasing**: Aliasing only works when the input and output are on the same device. Cross-device computations cannot use aliasing.
