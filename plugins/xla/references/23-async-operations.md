# Async HLO Operations

This document provides comprehensive documentation about asynchronous HLO operations in XLA, covering the generic async opcode approach, the `kAsyncStart`, `kAsyncUpdate`, and `kAsyncDone` instructions, and their usage patterns.

## Table of Contents

- [Motivation](#motivation)
- [kAsyncStart, kAsyncUpdate, kAsyncDone](#kasyncstart-kasyncupdate-kasyncdone)
- [Example Representations](#example-representations)
- [Syntax Sugar](#syntax-sugar)
- [Constraints](#constraints)

## Motivation

### The Problem with Explicit Start/Done Splits

Prior to the generic async opcode approach, XLA used operation-specific start/done instruction pairs for asynchronous operations. For example:

- `copy-start` / `copy-done` for asynchronous copies
- `collective-permute-start` / `collective-permute-done` for asynchronous collectives
- `send` / `recv` with `send-done` / `recv-done` for inter-device communication

This approach had several problems:

1. **Proliferation of opcodes**: Every asynchronous operation needed two or more dedicated opcodes, bloating the instruction set.

2. **Duplicated infrastructure**: Each start/done pair required its own handling in the compiler (optimization passes, scheduling, buffer assignment, etc.), leading to significant code duplication.

3. **Inconsistent patterns**: Different start/done operations had slightly different semantics and behavior patterns, making it harder to write generic infrastructure.

4. **Difficulty extending**: Adding a new asynchronous operation required creating new opcodes and implementing all the associated infrastructure.

### The Generic Async Opcode Approach

To address these issues, XLA introduced a generic async opcode mechanism. Instead of creating operation-specific start/done opcodes, any HLO operation can be wrapped in a generic async pattern using three opcodes:

- `kAsyncStart`: Begins asynchronous execution of the wrapped operation.
- `kAsyncUpdate`: Updates the state of an in-progress async operation (optional, for multi-step async operations).
- `kAsyncDone`: Completes the asynchronous operation and produces the output.

This approach provides:

1. **Uniform handling**: All asynchronous operations use the same three opcodes, enabling generic infrastructure for scheduling, buffer assignment, and optimization.

2. **Reduced opcode count**: No need for operation-specific start/done opcodes (for new operations).

3. **Consistent semantics**: All async operations follow the same pattern and have consistent semantics.

4. **Easy extension**: Making any operation asynchronous requires only wrapping it in the async pattern, with no new opcodes needed.

## kAsyncStart, kAsyncUpdate, kAsyncDone

### Overview

The three async opcodes form a lifecycle for asynchronous operations:

```
kAsyncStart -> [kAsyncUpdate]* -> kAsyncDone
```

- `kAsyncStart` is always required (begins the operation).
- `kAsyncUpdate` is optional (zero or more updates for multi-step operations).
- `kAsyncDone` is always required (completes the operation and produces output).

### Output Shape: Tuple of (Inputs, Outputs, Context State)

Each async instruction produces a tuple that carries the state of the asynchronous operation:

#### kAsyncStart Output Shape

```
async_start_output = (input_0, input_1, ..., output_0, output_1, ..., context_state)
```

The tuple contains:

1. **Inputs**: Copies of (or references to) the input buffers. These are passed through so that subsequent operations can continue to use the inputs while the async operation is in progress.

2. **Outputs**: Pre-allocated buffers for the operation's outputs. These will be filled in by the async operation.

3. **Context state**: A backend-specific buffer that tracks the state of the asynchronous operation (e.g., a CUDA event, a command queue ID, or other tracking data).

#### kAsyncUpdate Output Shape

```
async_update_output = (input_0, input_1, ..., output_0, output_1, ..., updated_context_state)
```

The tuple has the same structure as the start output, but with an updated context state that reflects the progress of the operation.

#### kAsyncDone Output Shape

```
async_done_output = (output_0, output_1, ...)
```

The done instruction extracts just the output buffers from the tuple, producing the actual computation results.

### Buffer Aliasing Between Start and Done

A key optimization in the async pattern is buffer aliasing:

1. **Input aliasing**: The input buffers in the async tuple are aliased (shared) with the original input buffers. This means no copy is needed for the inputs.

2. **Output aliasing**: The output buffers in the async tuple are the same buffers that the async_done instruction produces. This means the output is written directly into the buffer that the consumer expects.

3. **Context aliasing**: The context state buffer is reused across start, update, and done operations.

This aliasing is critical for performance because it eliminates unnecessary buffer copies and ensures that the async operation's outputs are available to consumers without additional data movement.

```
                ┌─────────────────────┐
  input_0 ──────►│                     │──────► input_0 (aliased)
                │                     │
  input_1 ──────►│   async tuple       │──────► input_1 (aliased)
                │                     │
                │                     │──────► output_0 (written by op)
                │                     │
                │                     │──────► output_1 (written by op)
                │                     │
                │                     │──────► context_state
                └─────────────────────┘
                         │
                         ▼
               (async_start output tuple)
```

### Detailed Instruction Semantics

#### kAsyncStart

```cpp
// HLO instruction representation
%async_tuple = async-start(%input0, %input1, ...), async_execution={
  %result = some-operation(%input0, %input1, ...)
  ROOT %root = %result
}
```

Semantics:

1. Allocates output buffers and context state.
2. Starts executing the wrapped operation asynchronously.
3. Returns a tuple containing inputs (aliased), outputs (to be filled), and context state.
4. The caller can continue executing other operations while the async operation runs.

#### kAsyncUpdate

```cpp
// HLO instruction representation
%updated_tuple = async-update(%async_tuple), async_execution={
  %result = some-operation(%input0, %input1, ...)
  ROOT %root = %result
}
```

Semantics:

1. Takes the tuple from a previous async-start or async-update.
2. Performs the next step of the asynchronous operation (if multi-step).
3. Updates the context state.
4. Returns an updated tuple with the same structure but updated context state.

#### kAsyncDone

```cpp
// HLO instruction representation
%result0, %result1, ... = async-done(%async_tuple)
```

Semantics:

1. Takes the tuple from a previous async-start or async-update.
2. Waits for the asynchronous operation to complete (synchronization point).
3. Extracts the output buffers from the tuple.
4. Returns the output buffers as individual results (or as a tuple for multi-output operations).

## Example Representations

### Basic Async Operation

A simple asynchronous copy operation using the generic async pattern:

```
HloModule async_copy_example

ENTRY main {
  %input = f32[1024] parameter(0)

  // Start the async copy
  %async_tuple = (f32[1024], f32[1024], token[]) async-start(%input), async_execution={
    %copy = f32[1024] copy(%input)
    ROOT %root = %copy
  }

  // ... other operations can execute here while the copy runs ...

  // Complete the async copy
  %result = f32[1024] async-done(%async_tuple)

  ROOT %root = f32[1024] add(%result, %result)
}
```

In this example:

1. `async-start` begins copying `%input` into a new buffer asynchronously.
2. Other operations (like the `add` placeholder) can execute concurrently.
3. `async-done` waits for the copy to complete and returns the result.

### Multiple Tensor Input/Output

An asynchronous operation with multiple inputs and outputs:

```
HloModule async_multi_io_example

ENTRY main {
  %a = f32[4, 8] parameter(0)
  %b = f32[8, 4] parameter(1)
  %bias = f32[4] parameter(2)

  // Start an async matmul + bias add
  %async_tuple = async-start(%a, %b, %bias), async_execution={
    %matmul = f32[4, 4] dot(%a, %b),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    %broadcast_bias = f32[4, 4] broadcast(%bias), dimensions={1}
    %result = f32[4, 4] add(%matmul, %broadcast_bias)
    ROOT %root = %result
  }

  // ... concurrent computation ...

  // Complete the async operation
  %result = f32[4, 4] async-done(%async_tuple)

  ROOT %root = %result
}
```

The async tuple in this case contains:
```
(f32[4, 8], f32[8, 4], f32[4], f32[4, 4], context_state)
 │         │         │       │          │
 a         b        bias   output   async context
```

### With Async-Update Steps

For multi-step asynchronous operations (e.g., pipelined collectives):

```
HloModule async_update_example

ENTRY main {
  %input = f32[1024] parameter(0)

  // Step 1: Start the async operation
  %tuple_0 = async-start(%input), async_execution={
    %result = f32[1024] collective-permute(%input),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
    ROOT %root = %result
  }

  // ... some computation ...

  // Step 2: Update the async operation (progress the pipeline)
  %tuple_1 = async-update(%tuple_0), async_execution={
    %result = f32[1024] collective-permute(%input),
        source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
    ROOT %root = %result
  }

  // ... more computation ...

  // Step 3: Complete the async operation
  %result = f32[1024] async-done(%tuple_1)

  ROOT %root = %result
}
```

### Async Operation with Tuple Output

When the wrapped computation produces multiple outputs:

```
HloModule async_tuple_output_example

ENTRY main {
  %input = f32[1024] parameter(0)

  // Async operation that returns two outputs
  %async_tuple = async-start(%input), async_execution={
    %double = f32[1024] multiply(%input, %input)
    %half = f32[1024] multiply(%input, constant(0.5))
    ROOT %root = (f32[1024], f32[1024]) tuple(%double, %half)
  }

  %result_tuple = (f32[1024], f32[1024]) async-done(%async_tuple)
  %double = f32[1024] get-tuple-element(%result_tuple), index=0
  %half = f32[1024] get-tuple-element(%result_tuple), index=1

  ROOT %root = (f32[1024], f32[1024]) tuple(%double, %half)
}
```

## Syntax Sugar

### Automatic -start, -update, -done Suffixes

In HLO text representation, XLA provides syntax sugar that automatically expands an `async` wrapper into the three instructions:

```
// Sugar form
%result = async[%async_context], async_execution={
  %op_result = some-op(%input)
  ROOT %root = %op_result
}(%input)

// Expanded form (what XLA actually creates)
%async_context_start = async-start(%input), async_execution={
  %op_result = some-op(%input)
  ROOT %root = %op_result
}
%async_context_done = async-done(%async_context_start)
%result = get-tuple-element(%async_context_done), index=0
```

### Pretty-Printing

When dumping HLO, the pretty-printer can show async operations in their sugar form for readability:

```
// Pretty-printed (with --xla_dump_hlo_as_short_text)
async %result = copy(%input)

// Full representation
%1 = async-start(%input), async_execution={ copy(%input) }
%2 = async-done(%1)
%result = get-tuple-element(%2), index=0
```

### HLO Text Format

The HLO module text format supports both the expanded and sugar forms:

```
HloModule example

ENTRY main {
  // Explicit form:
  %p0 = f32[1024] parameter(0)
  %async_start = (f32[1024], f32[1024], s32[]) async-start(%p0), async_execution={
    %copy = f32[1024] copy(%p0)
    ROOT %root = %copy
  }
  %async_done = f32[1024] async-done(%async_start)
  ROOT %result = %async_done
}
```

## Constraints

### No Wrapping Operations That Have Explicit Start/Done Opcodes

The generic async pattern should not be used to wrap operations that already have their own dedicated start/done opcodes. These legacy operations continue to use their specific opcodes:

- `copy-start` / `copy-done`
- `collective-permute-start` / `collective-permute-done`
- `send` / `send-done`
- `recv` / `recv-done`
- `infeed` / `infeed-done`
- `outfeed` / `outfeed-done`

This constraint exists because:

1. **Backward compatibility**: Existing code and optimizations rely on these specific opcodes.
2. **Special semantics**: Some of these operations have unique semantics (e.g., `send`/`recv` have specific channel semantics) that are better expressed with dedicated opcodes.
3. **Transition period**: The generic async pattern and legacy opcodes coexist during a gradual transition.

### Legacy Operations

#### Copy Start/Done

The `copy-start` / `copy-done` pair is used for asynchronous buffer copies:

```
%p0 = f32[1024] parameter(0)
%copy_start = (f32[1024], f32[1024], token[]) copy-start(%p0)
// ... other operations ...
%copy_done = f32[1024] copy-done(%copy_start)
```

#### Collective Permute Start/Done

The `collective-permute-start` / `collective-permute-done` pair is used for asynchronous collective permutation operations:

```
%p0 = f32[1024] parameter(0)
%cp_start = (f32[1024], f32[1024], token[]) collective-permute-start(%p0),
    source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
// ... other operations ...
%cp_done = f32[1024] collective-permute-done(%cp_start)
```

### Nesting Constraints

Async operations cannot be arbitrarily nested:

1. **No nested async**: An async operation's body cannot itself contain async-start/async-done pairs for the same operation. This prevents infinite recursion and ambiguous scheduling.

2. **Sequential async-done**: An `async-done` instruction must correspond to a preceding `async-start` (or `async-update`). The compiler verifies that the chain is well-formed.

3. **Single consumer of context**: The async context (tuple) produced by `async-start` should have a single consumer chain leading to `async-done`. This ensures clear ownership of the async operation's lifecycle.

### Shape Constraints

1. **Matching shapes**: The shapes of the wrapped operation's inputs and outputs must match the shapes declared in the async-start/async-done instructions.

2. **Tuple structure**: The async tuple must follow the (inputs, outputs, context_state) structure. The context state is typically a scalar or small tensor that can hold the backend-specific tracking information.

3. **Element types**: All element types used in the async operation must be supported by the target backend.

### Scheduling Constraints

1. **Dependency tracking**: The `async-done` instruction implicitly depends on the completion of the `async-start` operation. The scheduler must respect this dependency.

2. **Resource constraints**: Each backend may limit the number of concurrent async operations (e.g., limited CUDA streams, limited DMA engines). The scheduler must account for these constraints.

3. **Memory pressure**: Async operations hold their output buffers until `async-done` is reached. This means that many concurrent async operations can increase peak memory usage. The scheduler should balance concurrency with memory pressure.

### Backend Implementation Requirements

Each XLA backend must implement support for async operations:

1. **Stream allocation**: The backend must be able to allocate a device stream or queue for the async operation.

2. **Event mechanism**: The backend needs an event or notification mechanism to signal completion of the async operation.

3. **Buffer management**: The backend must support the aliasing pattern used by async operations.

4. **Error handling**: The backend must properly propagate errors that occur during async execution to the `async-done` instruction.
