# Operations: Control Flow

Tile IR contains a standard set of control flow operations that enable conditionals and loops. The operations are designed in the style of the MLIR Control Flow dialect. A notable difference is that we allow the nesting of control flow operations; for example, a `cuda_tile.if` may appear inside a `cuda_tile.loop` or `cuda_tile.for`.

The main control structures are:

- `cuda_tile.if` -- conditional branching
- `cuda_tile.loop` -- a loop with arbitrary exit conditions
- `cuda_tile.for` -- a range-based loop with a fixed number of iterations

---

## `cuda_tile.assert`

Terminate kernel execution with an error message if condition is false-y.

```
cuda_tile.assert %condition %message
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| condition | `tile<i1>` | The condition tile to check |
| message | `String` | The error message to display if assertion fails |

**Results:** None.

**Description:**

Takes a tile of i1 values. For each value that is 0, it prints the given error message, along with the index of the value within the tile. If at least one value is 0, an error is signalled to the host side. The kernel, including the tile block that failed the assertion, may keep running.

Assertions are for debugging purposes. They can affect performance and it is therefore recommended to remove them in production code.

**Examples:**

```cuda_tile
assert %arg0, "assertion failed" : tile<i1>
```

---

## `cuda_tile.break`

Break from loop.

```
cuda_tile.break %operands
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| operands | `Variadic<Any>` | The operands to yield to the parent loop upon termination |

**Results:** None.

**Description:**

A terminator operation of `cuda_tile.loop`. It may yield any number of operands to the parent loop upon termination. The break operation always returns control to the innermost enclosing loop operation, even when it is nested within other control constructs such as if or additional loops.

**Examples:**

```cuda_tile
// Break from the body of a loop.
loop {
    break
}

// Break from an if nested within the loop.
loop  {
    %condition = constant <i1: 1> : tile<i1>
    if %condition  {
        break
    }
    // ...
}

%initValue0 = constant <f32: 0.0> : tile<f32>
// Break from an if nested within the loop, while yielding values.
%results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
    %condition = constant <i1: 1> : tile<i1>
    if %condition  {
        yield
    } else {
        %loopValue0 = constant <f32: 1.0> : tile<f32>
        break %loopValue0 : tile<f32>
    }
    %loopValue1 = constant <f32: 1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

---

## `cuda_tile.continue`

Continue to next loop iteration.

```
cuda_tile.continue %operands
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| operands | `Variadic<Any>` | The values to yield to the parent loop |

**Results:** None.

**Description:**

Represents a block terminator that returns control to a loop operation, such as `cuda_tile.for` and `cuda_tile.loop`. The operation may yield any number of operands to the parent loop. The continue operation always returns control to the innermost enclosing loop operation.

**Examples:**

```cuda_tile
%lowerBound = constant <i32: 0> : tile<i32>
%upperBound = constant <i32: 10> : tile<i32>
%step = constant <i32: 1> : tile<i32>
%condition = constant <i1: 1> : tile<i1>

// Continue from the body of a loop.
for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
    continue
}

// Continue from an if nested within the loop.
for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
    if %condition  {
        continue
    }
    // ...
}

// Continue from an if nested within the loop, while yielding values.
%initVar0 = constant <f32: 0.0> : tile<f32>
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
          iter_values(%var0 = %initVar0) -> (tile<f32>)
  {
    if %condition {
        yield
    } else {
        %loopValue0 = constant <f32: 1.0> : tile<f32>
        continue %loopValue0 : tile<f32>
    }
    %loopValue1 = constant <f32: 1.0> : tile<f32>
    continue %loopValue1 : tile<f32>
}
```

---

## `cuda_tile.for`

For loop over integer range.

```
cuda_tile.for %lowerBound %upperBound %step %initValues %unsignedCmp
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| lowerBound | `tile<any>` | The lower bound of the loop |
| upperBound | `tile<any>` | The upper bound of the loop |
| step | `tile<any>` | The step of the loop |
| initValues | `Variadic<Any>` | The initial values of the loop-carried values |
| unsignedCmp | `Flag` | If present, use unsigned comparison for loop termination |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| resultValues | `Variadic<Any>` | The values of loop-carried variables after termination |

**Description:**

A structured range-based sequential loop consisting of: (1) a range formed by lowerBound, upperBound, and step, (2) a set of loop-carried values initialized by initValues, and (3) a region representing the loop body.

The iteration space is defined by the interval [lowerBound, upperBound) with each value separated by step. lowerBound and upperBound specify a half-open range. step must be positive but the bounds may be negative or zero. The bounds are interpreted as signed integers by default; the optional `unsignedCmp` flag enables unsigned comparison.

The body must be terminated by `cuda_tile.continue` that yields the next iteration's value for each loop carried variable.

> **Warning:** Loop carried variables cannot be a tensor_view or view type. `for` operations cannot terminate early.

**Examples:**

```cuda_tile
%lowerBound = constant <i32: 0> : tile<i32>
%upperBound = constant <i32: 10> : tile<i32>
%step = constant <i32: 1> : tile<i32>

// A simple loop iterating over an i32 range.
for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
    continue
}

%initVal0 = constant <f32: 0.0> : tile<f32>
// A loop with a loop carried value.
%results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                    iter_values(%val00 = %initVal0) -> (tile<f32>) {
  %loopVal0 = constant <f32: 1.0> : tile<f32>
  continue %loopVal0 : tile<f32>
}
```

---

## `cuda_tile.if`

Conditional execution.

```
cuda_tile.if %condition
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| condition | `tile<i1>` | The condition of the if operation |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| results | `Variadic<Any>` | The results of the if operation |

**Description:**

Represents an if-then-else construct consisting of: (1) a control operand which is a `tile<i1>` value, (2) a true branch thenRegion, and (3) an optional false branch elseRegion.

May produce results by yielding values in each branch using `cuda_tile.yield`. If yielding values, both branches must yield values and the types must match.

> **Warning:** Results of if must not be a tensor_view or view type.

**Examples:**

```cuda_tile
%condition = constant <i1: 1> : tile<i1>

// A simple if operation that conditionally executes a region.
if %condition  {
  // ...
}

// An if operation with an "else" branch.
if %condition  {
  // ...
} else {
  // ...
}

// An if operation that returns mixed types (f32,i32)
%x, %y = if %condition -> (tile<f32>, tile<i32>) {
  %x_then = constant <f32: 1.0> : tile<f32>
  %y_then = constant <i32: 2> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
} else {
  %x_then = constant <f32: 1.0> : tile<f32>
  %y_then = constant <i32: 42> : tile<i32>
  yield %x_then, %y_then : tile<f32>, tile<i32>
}
```

---

## `cuda_tile.loop`

Loop until a break operation.

```
cuda_tile.loop %initValues
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| initValues | `Variadic<Any>` | The initial values of the loop |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| resultValues | `Variadic<Any>` | The result values of the loop |

**Description:**

Represents an unstructured infinite loop that executes until a `cuda_tile.break` is reached. Each control path must be terminated by either a `cuda_tile.continue` or a `cuda_tile.break`.

> **Warning:** Early returns from inside loops are not supported. Loop carried variables cannot be a tensor_view or view type.

**Examples:**

```cuda_tile
// A simple "while-do" loop.
loop {
    %cond = constant <i1: 1> : tile<i1>
    if %cond {
        continue
    }
    break
}

// A loop with carried values.
%initValue0 = constant <f32: 0.0> : tile<f32>
%results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
    %cond = constant <i1: 1> : tile<i1>
    if %cond {
        %loopValue0 = constant <f32: 0.0> : tile<f32>
        continue %loopValue0 : tile<f32>
    }
    break %value0 : tile<f32>
}
```

---

## `cuda_tile.return`

Return value(s) from a function.

```
cuda_tile.return %operands
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| operands | `Variadic<Any>` | The values to return |

**Results:** None.

**Description:**

Returns control to the caller of a function. `cuda_tile.entry` operations do not produce return values and thus return may be used with no operands to terminate the kernel.

> **Warning:** Cannot be directly used inside loop bodies to terminate kernel execution.

**Examples:**

```cuda_tile
entry @foo() {
  %0 = constant <i32: 0> : tile<i32>
  %1 = constant <f16: 0.0> : tile<f16>
  // ...
  return
}
```

---

## `cuda_tile.yield`

Yield a value from the block.

```
cuda_tile.yield %operands
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| operands | `Variadic<Any>` | The operands to yield to the parent operation |

**Results:** None.

**Description:**

Terminates a block that must yield control back to the parent operation such as if, scan, reduce. The number of values yielded and the execution semantics are determined by the parent operation.

> **Note:** Unlike standard MLIR control flow dialects, yield is not used for loop control flow; see `cuda_tile.break` and `cuda_tile.continue` for loop control flow.

**Examples:**

```cuda_tile
%condition = constant <i1: true> : tile<i1>
// Yield from the body of an if conditional.
if %condition  {
    yield
}

// Yield values from within an if conditional.
%x, %y = if %condition -> (tile<f32>, tile<f32>) {
    %x_then = constant <f32: 0.0> : tile<f32>
    %y_then = constant <f32: 1.0> : tile<f32>
    yield %x_then, %y_then : tile<f32>, tile<f32>
} else {
    %x_else = constant <f32: 2.0> : tile<f32>
    %y_else = constant <f32: 3.0> : tile<f32>
    yield %x_else, %y_else : tile<f32>, tile<f32>
}
```
