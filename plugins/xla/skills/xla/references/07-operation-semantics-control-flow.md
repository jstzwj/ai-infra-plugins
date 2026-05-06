# XLA Operation Semantics: Control Flow, Reduction, and Sorting Operations

This reference provides comprehensive documentation of XLA control flow operations, reduction operations (including windowed and scattered variants), sorting, and top-k. These operations form the backbone of expressing complex computation graphs in XLA, enabling conditional execution, loops, reductions, and data rearrangement.

---

## Table of Contents

1. [Conditional](#conditional)
2. [While](#while)
3. [Call](#call)
4. [SelectAndScatter](#selectandscatter)
5. [Reduce](#reduce)
6. [ReduceWindow](#reducewindow)
7. [Scatter](#scatter)
8. [Sort](#sort)
9. [TopK](#topk)
10. [StableHLO Cross-References](#stablehlo-cross-references)

---

## Conditional

`Conditional` provides branching control flow in XLA. It comes in two variants: a **predicate-based** ternary form (analogous to `cond ? true_val : false_val`) and a **branch-index-based** switch form (analogous to a switch/case statement).

### Variant 1: Predicate-Based (Ternary)

```
Conditional(predicate, true_operand, true_computation,
            false_operand, false_computation)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `predicate` | `XlaOp` | A scalar of type `PRED` (boolean). If `true`, `true_computation` is executed; otherwise `false_computation` is executed. |
| `true_operand` | `XlaOp` | The argument passed to `true_computation` if the predicate is true. Can be a tuple of tensors. |
| `true_computation` | `XlaComputation` | The computation to execute when `predicate` is true. Must accept a single argument of the same type (shape and layout) as `true_operand`. |
| `false_operand` | `XlaOp` | The argument passed to `false_computation` if the predicate is false. Can be a tuple of tensors. |
| `false_computation` | `XlaComputation` | The computation to execute when `predicate` is false. Must accept a single argument of the same type as `false_operand`. |

#### Semantics

If `predicate` evaluates to `true`:
```
result = true_computation(true_operand)
```
If `predicate` evaluates to `false`:
```
result = false_computation(false_operand)
```

Only one branch is executed. The output type of both branches must match (same shape and element type).

#### Example

```
// Conditional: compute abs(x) if x < 0, else return x
%neg = pred[] compare(f32[] %x, f32{0.0}), direction=LT

// True branch: negate x
%true_comp = (f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %negate = f32[] negate(f32[] %p)
}

// False branch: identity
%false_comp = (f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %identity = f32[] parameter(0)
}

%result = f32[] conditional(pred[] %neg, f32[] %x, %true_comp,
                             f32[] %x, %false_comp)
```

### Variant 2: Branch-Index-Based (Switch/Case)

```
Conditional(branch_index, branch_operands, branch_computations)
```

#### Arguments

| Argument | Type | Description |
|---|---|---|
| `branch_index` | `XlaOp` | A scalar of type `S32` (signed 32-bit integer). Selects which branch to execute. If the value is outside `[0, num_branches)`, the last branch is executed as a default. |
| `branch_operands` | `std::vector<XlaOp>` | A list of operands, one per branch. `branch_operands[i]` is passed to `branch_computations[i]`. |
| `branch_computations` | `std::vector<XlaComputation>` | A list of computations, one per branch. `branch_computations[i]` is executed when `branch_index == i`. |

#### Semantics

Let `n = branch_index`:
- If `n` is in `[0, N)` where `N` is the number of branches: `result = branch_computations[n](branch_operands[n])`
- If `n < 0` or `n >= N`: `result = branch_computations[N-1](branch_operands[N-1])` (last branch is the default)

All branch computations must return the same output shape and type.

#### Example

```
// Switch on operation type:
// 0 -> add, 1 -> subtract, 2 -> multiply (default)
%index = s32[] constant(1)  // will select subtract

%add_comp = ((f32[], f32[])) -> f32[] computation(...) { ... add ... }
%sub_comp = ((f32[], f32[])) -> f32[] computation(...) { ... subtract ... }
%mul_comp = ((f32[], f32[])) -> f32[] computation(...) { ... multiply ... }

%result = f32[] conditional(
  s32[] %index,
  {f32[] %a, f32[] %b}, %add_comp,
  {f32[] %a, f32[] %b}, %sub_comp,
  {f32[] %a, f32[] %b}, %mul_comp
)
```

### HLO Text Format

```
// Predicate variant
%result = f32[] conditional(pred[] %pred,
  f32[] %true_val, %true_computation,
  f32[] %false_val, %false_computation)

// Branch index variant
%result = f32[] conditional(s32[] %index,
  {f32[] %op0}, %comp0,
  {f32[] %op1}, %comp1,
  {f32[] %op2}, %comp2)
```

### Constraints

- Both `true_computation` and `false_computation` (or all branch computations) must return values of the same shape and element type.
- Each computation's parameter type must match the corresponding operand type.
- The predicate must be a scalar `PRED` value; the branch index must be a scalar `S32` value.

---

## While

`While` implements a loop construct in XLA. It repeatedly applies a body computation until a condition computation returns false.

### Signature

```
While(condition, body, init)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `condition` | `XlaComputation` | A computation that takes the current loop state (a tuple or single tensor) and returns a scalar `PRED`. The loop continues as long as this returns `true`. |
| `body` | `XlaComputation` | A computation that takes the current loop state and returns the updated state. Its input and output types must match. |
| `init` | `XlaOp` | The initial value of the loop state. Must match the parameter type of both `condition` and `body`. |

### Semantics

The `While` operation executes the following pseudocode:

```
state = init
while condition(state):
    state = body(state)
return state
```

More precisely:

1. The `condition` computation is called with `init` as its argument. It returns a scalar boolean.
2. If the result is `true`, the `body` computation is called with the current state, producing a new state.
3. The `condition` is called again with the new state. Steps 2-3 repeat until `condition` returns `false`.
4. The final state is returned as the result.

If `condition(init)` is initially `false`, the loop body never executes and `init` is returned directly.

### Example: Sum 1 to N

Compute `sum = 1 + 2 + ... + N`:

```
// State: (i, sum) where i is the counter, sum is the accumulator
// init: (1, 0)
// condition: i <= N
// body: (i+1, sum+i)

// Condition computation
%cond_comp = ((s32[], s32[], s32[])) -> pred[] {
  %i = s32[] get-tuple-element(parameter(0), 0)
  %N = s32[] get-tuple-element(parameter(0), 2)
  ROOT %cmp = pred[] compare(s32[] %i, s32[] %N), direction=LE
}

// Body computation
%body_comp = ((s32[], s32[], s32[])) -> (s32[], s32[], s32[]) {
  %state = parameter(0)
  %i = s32[] get-tuple-element(%state, 0)
  %sum = s32[] get-tuple-element(%state, 1)
  %N = s32[] get-tuple-element(%state, 2)
  %new_sum = s32[] add(s32[] %sum, s32[] %i)
  %new_i = s32[] add(s32[] %i, s32[] constant(1))
  ROOT %new_state = (s32[], s32[], s32[]) tuple(%new_i, %new_sum, %N)
}

// Initial state: (i=1, sum=0, N=100)
%init = (s32[], s32[], s32[]) tuple(
  s32[] constant(1),
  s32[] constant(0),
  s32[] constant(100)
)

%result = (s32[], s32[], s32[]) while(%cond_comp, %body_comp, %init)
```

#### Extracting Results

```
%final_state = while(...)
%final_sum = s32[] get-tuple-element(%result, 1)  // sum = 5050
```

### HLO Text Format

```
%result = (s32[], s32[], s32[]) while((s32[], s32[], s32[]) %init),
  condition=%cond_comp, body=%body_comp
```

### Important Considerations

1. **Static trip count**: When possible, XLA attempts to unroll or analyze while loops. If the trip count is statically known, the compiler may fully unroll the loop for optimization.

2. **Unbounded loops**: XLA may impose limits on maximum iteration counts. Frameworks like JAX use `jax.lax.while_loop` which maps to this operation, with a configurable maximum trip count.

3. **Loop-carried state**: The state must have the same shape and type across all iterations. Dynamic shapes are not supported in the loop state.

4. **Collective operations in loops**: As noted in the collective operations reference, placing collective operations inside `While` loops with `Infeed` can cause deadlocks if different replicas take different paths.

---

## Call

`Call` invokes a sub-computation with given arguments, analogous to a function call.

### Signature

```
Call(computation, arguments)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `computation` | `XlaComputation` | The computation to invoke. Must accept parameters matching the types of `arguments`. |
| `arguments` | `std::vector<XlaOp>` | The arguments to pass to the computation. Each argument's shape and layout must match the corresponding parameter of `computation`. |

### Semantics

`Call` is a pure function invocation:

```
result = computation(arguments[0], arguments[1], ..., arguments[N-1])
```

The computation is executed synchronously. The result type is determined by the root instruction of `computation`.

#### Example

```
%add_square_comp = (f32[], f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  %sum = f32[] add(f32[] %a, f32[] %b)
  ROOT %result = f32[] multiply(f32[] %sum, f32[] %sum)
}

%x = f32[] constant(3.0)
%y = f32[] constant(4.0)
%result = f32[] call(%add_square_comp, %x, %y)
// result = (3.0 + 4.0)^2 = 49.0
```

### HLO Text Format

```
%result = f32[] call(f32[] %x, f32[] %y), to_apply=%add_square_comp
```

### CompositeCall

`CompositeCall` is a specialized form that includes metadata for decomposition and versioning. It represents a composite operation that can be decomposed into simpler HLO operations.

#### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `std::string` | The name of the composite operation (e.g., `"fused_attention"`, `"rms_norm"`). |
| `decomposition` | `XlaComputation` | The HLO computation that implements the composite operation. Used by backends that cannot natively handle the composite. |
| `version` | `int64` | A version number for the composite. Incremented when the semantics or decomposition changes. |
| `frontend_attributes` | `std::map<std::string, std::string>` | Additional key-value metadata from the frontend framework. |

#### Semantics

A `CompositeCall` behaves semantically identically to inlining the `decomposition` computation. However, backends may recognize specific composite names and replace them with optimized implementations:

- If a backend has a native implementation for the named composite, it uses that.
- Otherwise, the decomposition is inlined and compiled normally.

This mechanism allows frameworks to express high-level operations (e.g., fused attention, RMS normalization) while maintaining compatibility with all XLA backends.

#### Example

```python
# Conceptual: a framework defines a fused attention composite
composite = CompositeCall(
    name="fused_attention",
    decomposition=attention_decomposition_hlo,
    version=2,
    frontend_attributes={"framework": "jax", "backend_hint": "cudnn"}
)
```

---

## SelectAndScatter

`SelectAndScatter` is a specialized operation that combines element selection and scattering. It is primarily used to implement pooling layers in neural networks (specifically, max pooling with gradient support).

### Signature

```
SelectAndScatter(operand, select, window_dimensions, window_strides,
                 padding, source, init_value, scatter)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input array from which values are selected. |
| `select` | `XlaComputation` | A binary computation that takes two scalar values and returns a boolean. Returns `true` if the first argument should be selected over the second (e.g., greater-than for max pooling). |
| `window_dimensions` | `std::vector<int64>` | The size of the sliding window in each dimension. |
| `window_strides` | `std::vector<int64>` | The stride of the sliding window in each dimension. |
| `padding` | `std::vector<std::pair<int64, int64>>` | Padding for each dimension: `{(low_0, high_0), (low_1, high_1), ...}`. |
| `source` | `XlaOp` | The values to scatter. Its shape matches the output shape of the select operation. |
| `init_value` | `XlaOp` | A scalar representing the initial value for the scatter accumulation. |
| `scatter` | `XlaComputation` | A binary computation used to accumulate scattered values. Takes two scalars and returns one (typically addition). |

### Semantics

`SelectAndScatter` operates in two phases:

**Phase 1 -- Select**: For each window position in `operand`, the `select` computation is applied iteratively (in undefined order) to find the "selected" element -- the one that wins all pairwise comparisons. This produces an output array where each element corresponds to a window position and holds the selected value.

**Phase 2 -- Scatter**: Each value from `source` is placed (scattered) into the position of the selected element in the operand. If multiple source values scatter to the same position, the `scatter` computation combines them (typically by summation).

### Mathematical Description

Let `S` be the output of the select phase. For window position `w`:
```
S[w] = operand[argmax_window(operand, w)]  // for max pooling
```

For the scatter phase, the output array `O` (same shape as `operand`) is initialized to `init_value`. Then:
```
for each window position w:
    selected_position = position of S[w] in operand
    O[selected_position] = scatter(O[selected_position], source[w])
```

### Example: Max Pooling Backward Pass

Given a 1D input `[1, 2, 3, 4]` with window size 2, stride 2:

**Select phase** (max): `[max(1,2), max(3,4)]` = `[2, 4]`

**Scatter phase**: Given `source = [10, 20]`:
- Position of max(1,2) = index 1 -> scatter 10 to position 1
- Position of max(3,4) = index 3 -> scatter 20 to position 3
- Output: `[0, 10, 0, 20]`

#### HLO Text Format

```
%result = f32[4]{0} select-and-scatter(
  f32[4]{0} %operand,
  f32[2]{0} %source,
  f32[] %init_value
), select=%gt_comp, scatter=%add_comp,
  window_dimensions={2}, window_strides={2}, padding={{0,0}}
```

---

## Reduce

`Reduce` applies a reduction computation along specified dimensions of the input operand(s), producing an output with fewer dimensions.

### Signature

```
Reduce(operands, init_values, computation, dimensions)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operands` | `std::vector<XlaOp>` | One or more input arrays to be reduced. All must have the same shape. |
| `init_values` | `std::vector<XlaOp>` | Scalar initial values for the reduction, one per operand. The reduction starts with these values. |
| `computation` | `XlaComputation` | A reduction function. For a single-operand reduce, it takes two scalars and returns one. For a variadic (multi-operand) reduce, it takes `2 * N` scalars (N per operand) and returns `N` scalars. |
| `dimensions` | `std::vector<int64>` | The dimensions along which to reduce. These dimensions are removed from the output shape. |

### Semantics

For each unique combination of indices in the non-reduced dimensions, the reduction computation is applied to all elements along the reduced dimensions, accumulating from `init_value`.

The output shape is the input shape with `dimensions` removed:

```
output_rank = input_rank - len(dimensions)
output_shape[i] = input_shape[j] for j not in dimensions
```

The computation must be **associative** and **commutative** for the result to be deterministic (since the order of application is not specified).

### Single-Operand Reduce

#### Example 1: 1D Array Sum

Input: `f32[6]` = `[1, 2, 3, 4, 5, 6]`

```
%add_comp = (f32[], f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %sum = f32[] add(%x, %y)
}

%result = f32[] reduce(f32[6] %operand, f32[] constant(0.0)),
  to_apply=%add_comp, dimensions={0}
```

Result: `21.0`

#### Example 2: 2D Array Row Sum

Input: `f32[3, 4]` =
```
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 10, 11, 12]]
```

Reduce along dimension 1 (`dimensions={1}`) with addition:

```
%result = f32[3] reduce(f32[3,4] %operand, f32[] constant(0.0)),
  to_apply=%add_comp, dimensions={1}
```

Result: `f32[3]` = `[10, 26, 42]`

#### Example 3: 2D Array Column Sum

Same input, reduce along dimension 0 (`dimensions={0}`):

```
%result = f32[4] reduce(f32[3,4] %operand, f32[] constant(0.0)),
  to_apply=%add_comp, dimensions={0}
```

Result: `f32[4]` = `[15, 18, 21, 24]`

#### Example 4: Reduce All Dimensions

```
%result = f32[] reduce(f32[3,4] %operand, f32[] constant(0.0)),
  to_apply=%add_comp, dimensions={0, 1}
```

Result: `78.0` (scalar)

### Variadic Reduce (Multi-Operand)

Variadic reduce reduces multiple operands simultaneously with a single computation. This is useful when the reduction of one operand depends on another (e.g., computing both sum and count, or finding both argmax and max).

#### Signature

```
Reduce({operand_0, operand_1, ..., operand_N},
       {init_value_0, init_value_1, ..., init_value_N},
       computation, dimensions)
```

The computation takes `2 * N` parameters: `(acc_0, acc_1, ..., acc_N, val_0, val_1, ..., val_N)` and returns `N` values: `(new_acc_0, new_acc_1, ..., new_acc_N)`.

#### Example: Compute Sum and Count of Positive Elements

```
// Operands:
//   operand_0: the data array (f32[M, N])
//   operand_1: indicator array, 1.0 where positive, 0.0 otherwise (f32[M, N])

// Init values: 0.0, 0.0

// Computation:
%var_comp = (f32[], f32[], f32[], f32[]) -> (f32[], f32[]) {
  %sum_acc = f32[] parameter(0)    // accumulator for sum
  %count_acc = f32[] parameter(1)  // accumulator for count
  %val = f32[] parameter(2)        // current data value
  %indicator = f32[] parameter(3)  // current indicator
  %new_sum = f32[] add(%sum_acc, %val)
  %new_count = f32[] add(%count_acc, %indicator)
  ROOT %result = (f32[], f32[]) tuple(%new_sum, %new_count)
}

%results = (f32[], f32[]) reduce(
  {f32[M,N] %data, f32[M,N] %indicator},
  {f32[] constant(0.0), f32[] constant(0.0)},
  %var_comp, dimensions={0, 1}
)
```

### HLO Text Format

```
// Single operand
%result = f32[3] reduce(f32[3,4] %operand, f32[] %init),
  to_apply=%add_comp, dimensions={1}

// Variadic
%results = (f32[], f32[]) reduce(
  f32[M,N] %data, f32[M,N] %indicator,
  f32[] %init0, f32[] %init1
), to_apply=%var_comp, dimensions={0, 1}
```

### Common Reduction Patterns

| Pattern | Init Value | Computation | Description |
|---|---|---|---|
| Sum | `0` | `Add(x, y)` | Sum of elements |
| Product | `1` | `Mul(x, y)` | Product of elements |
| Max | `-inf` (or type min) | `Max(x, y)` | Maximum element |
| Min | `+inf` (or type max) | `Min(x, y)` | Minimum element |
| Logical AND | `true` | `And(x, y)` | All elements true? |
| Logical OR | `false` | `Or(x, y)` | Any element true? |
| ArgMax | `(min_val, -1)` | Variadic | Index of maximum |

---

## ReduceWindow

`ReduceWindow` applies a reduction computation over a sliding window across the input operand. It is the foundation for operations like max pooling, average pooling, and local response normalization.

### Signature

```
ReduceWindow(operand, init_value, computation, window_dimensions,
             window_strides, padding, window_dilations, base_dilations)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input array. |
| `init_value` | `XlaOp` | A scalar representing the initial value for the reduction. |
| `computation` | `XlaComputation` | A binary reduction function (two scalars -> one scalar). Must be associative and commutative. |
| `window_dimensions` | `std::vector<int64>` | The size of the window in each dimension. |
| `window_strides` | `std::vector<int64>` | The stride between consecutive window positions in each dimension. |
| `padding` | `std::vector<std::pair<int64, int64>>` | Padding for each dimension: `{(low_0, high_0), (low_1, high_1), ...}`. Elements outside the original array are treated as `init_value`. |
| `window_dilations` | `std::vector<int64>` | Dilation factor applied to the window in each dimension. A dilation of `d` means window elements are spaced `d` apart. |
| `base_dilations` | `std::vector<int64>` | Dilation factor applied to the base operand before applying the window. |

### Window Semantics

For each dimension `i`:
- A window of size `window_dimensions[i]` slides across the operand.
- The window moves by `window_strides[i]` positions at a time.
- Padding of `padding[i].first` on the low side and `padding[i].second` on the high side is applied.
- If `window_dilations[i] > 1`, the window is dilated: elements within the window are sampled at intervals of `window_dilations[i]`, effectively expanding the receptive field without increasing the window parameter count.
- If `base_dilations[i] > 1`, the operand is dilated: zeros (init values) are inserted between elements, expanding the spatial extent.

### Output Shape Calculation

For each dimension `i`:

```
padded_size = padding[i].low + operand_size[i] + padding[i].high
dilated_window = (window_dimensions[i] - 1) * window_dilations[i] + 1
output_size[i] = (padded_size - dilated_window) / window_strides[i] + 1
```

### Padding Options

Padding can be specified explicitly as pairs of low/high padding for each dimension. Common padding modes:

| Padding | Description |
|---|---|
| `{(0, 0), (0, 0), ...}` | No padding (valid convolution) |
| `{(k/2, k/2), ...}` | Same padding (where `k` is window size) |
| `{(k-1, k-1), ...}` | Full padding |

### Example 1: 1D Max Pooling

Input: `f32[4]` = `[3, 1, 4, 2]`
Window: `{2}`, Stride: `{2}`, Padding: `{(0, 0)}`

```
%result = f32[2] reduce-window(f32[4] %operand, f32[] -inf),
  to_apply=%max_comp, window_dimensions={2}, window_strides={2},
  padding={{0, 0}}
```

Window positions:
- Position 0: `[3, 1]` -> max = `3`
- Position 1: `[4, 2]` -> max = `4`

Result: `f32[2]` = `[3, 4]`

### Example 2: 2D Max Pooling with Padding

Input: `f32[4, 4]`:
```
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 10, 11, 12],
 [13, 14, 15, 16]]
```

Window: `{2, 2}`, Stride: `{2, 2}`, Padding: `{(0, 0), (0, 0)}`

```
%result = f32[2, 2] reduce-window(f32[4, 4] %operand, f32[] -inf),
  to_apply=%max_comp, window_dimensions={2, 2}, window_strides={2, 2},
  padding={{0, 0}, {0, 0}}
```

Window positions:
- `(0,0)`: `[[1,2],[5,6]]` -> max = `6`
- `(0,1)`: `[[3,4],[7,8]]` -> max = `8`
- `(1,0)`: `[[9,10],[13,14]]` -> max = `14`
- `(1,1)`: `[[11,12],[15,16]]` -> max = `16`

Result: `f32[2, 2]` = `[[6, 8], [14, 16]]`

### Example 3: Average Pooling

Use `Add` as the computation with `init_value = 0`, then divide by window size:

```
%sum = f32[H, W] reduce-window(f32[H, W] %input, f32[] 0.0),
  to_apply=%add_comp, window_dimensions={kH, kW},
  window_strides={sH, sW}, padding=...
%count = f32[] constant(kH * kW)
%avg = f32[H, W] divide(%sum, %count)
```

### Example 4: Window Dilation

Input: `f32[6]` = `[1, 2, 3, 4, 5, 6]`
Window: `{2}`, Stride: `{1}`, Padding: `{(0, 0)}`, Window Dilation: `{2}`

With window dilation of 2, the window samples at indices `[0, 2]`, `[1, 3]`, `[2, 4]`, etc. (dilated window of effective size 3 with 2 actual elements).

```
%result = f32[4] reduce-window(f32[6] %operand, f32[] -inf),
  to_apply=%max_comp, window_dimensions={2}, window_strides={1},
  padding={{0, 0}}, window_dilations={2}
```

Positions:
- `[0, 2]` = `[1, 3]` -> `3`
- `[1, 3]` = `[2, 4]` -> `4`
- `[2, 4]` = `[3, 5]` -> `5`
- `[3, 5]` = `[4, 6]` -> `6`

Result: `f32[4]` = `[3, 4, 5, 6]`

### HLO Text Format

```
%result = f32[2,2]{1,0} reduce-window(f32[4,4]{1,0} %operand, f32[] %init),
  to_apply=%max_comp, window_dimensions={2,2}, window_strides={2,2},
  padding={{0,0},{0,0}}
```

---

## Scatter

`Scatter` scatters values from a `updates` tensor into a copy of `operand` at positions specified by `scatter_indices`. It is the inverse of `Gather` and is used for operations like scatter-add, scatter-max, and assignment.

### Signature

```
Scatter(operand, scatter_indices, updates, update_computation,
        scatter_dimension_numbers, indices_are_sorted, unique_indices)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The tensor into which values are scattered. The starting tensor that is copied and then modified. |
| `scatter_indices` | `XlaOp` | A tensor specifying where to scatter. Shape: `[index_vector_dim_size, ...]`. Contains indices into `operand`. |
| `updates` | `XlaOp` | The values to scatter into `operand`. Shape depends on `scatter_dimension_numbers`. |
| `update_computation` | `XlaComputation` | A binary function that combines the existing value at a scatter position with the new update value. Takes two scalars (existing, update) and returns one scalar. |
| `scatter_dimension_numbers` | `ScatterDimensionNumbers` | Configuration specifying how dimensions map between indices, operand, and updates. |
| `indices_are_sorted` | `bool` | Hint that `scatter_indices` are sorted. Enables optimization. Default `false`. |
| `unique_indices` | `bool` | Hint that each index in `scatter_indices` is unique (no duplicate scatter targets). Default `false`. |

### ScatterDimensionNumbers

The `ScatterDimensionNumbers` struct configures the index mapping:

| Field | Type | Description |
|---|---|---|
| `update_window_dims` | `std::vector<int64>` | Dimensions of `updates` that correspond to the window in `operand`. These dimensions of `updates` map to a slice of `operand` starting at the scattered index. |
| `inserted_window_dims` | `std::vector<int64>` | Dimensions of `operand` that are not present in `updates`. These dimensions of the operand slice are fully indexed by the scatter indices (not windowed). |
| `scatter_dims_to_operand_dims` | `std::vector<int64>` | Maps each dimension of the index vector (last dimension of `scatter_indices`) to a dimension of `operand`. |
| `index_vector_dim` | `int64` | The dimension in `scatter_indices` that contains the index vector. All other dimensions are "batch" dimensions that iterate over scatter operations. |

### Update Computation Semantics

When scattering to a position, the `update_computation` is called with:
```
new_value = update_computation(existing_value_at_position, update_value)
```

Common update computations:

| Operation | Computation | Result |
|---|---|---|
| Scatter-assign | Return `update` (ignore existing) | Overwrite |
| Scatter-add | `Add(existing, update)` | Accumulate |
| Scatter-multiply | `Mul(existing, update)` | Scale |
| Scatter-max | `Max(existing, update)` | Track maximum |
| Scatter-min | `Min(existing, update)` | Track minimum |

### Example 1: Simple Scatter-Assign

Operand: `f32[6]` = `[0, 0, 0, 0, 0, 0]`
Scatter indices: `s32[3, 1]` = `[[1], [3], [5]]`
Updates: `f32[3]` = `[10, 20, 30]`

```
%scatter_dim_nums = ScatterDimensionNumbers(
  update_window_dims = {},
  inserted_window_dims = {0},
  scatter_dims_to_operand_dims = {0},
  index_vector_dim = 1
)

%result = f32[6] scatter(f32[6] %operand, s32[3,1] %indices,
  f32[3] %updates, %assign_comp, %scatter_dim_nums)
```

Result: `[0, 10, 0, 20, 0, 30]`

The update computation for scatter-assign simply returns the update value:
```
%assign_comp = (f32[], f32[]) -> f32[] {
  %existing = f32[] parameter(0)
  %update = f32[] parameter(1)
  ROOT %result = f32[] parameter(1)  // return update
}
```

### Example 2: Scatter-Add (2D)

Operand: `f32[3, 3]` = all zeros
Scatter indices: `s32[2, 2]` = `[[0, 0], [1, 1]]`
Updates: `f32[2]` = `[5, 7]`

```
%scatter_dim_nums = ScatterDimensionNumbers(
  update_window_dims = {},
  inserted_window_dims = {0, 1},
  scatter_dims_to_operand_dims = {0, 1},
  index_vector_dim = 1
)

%result = f32[3, 3] scatter(f32[3, 3] %operand, s32[2, 2] %indices,
  f32[2] %updates, %add_comp, %scatter_dim_nums)
```

Result:
```
[[5, 0, 0],
 [0, 7, 0],
 [0, 0, 0]]
```

### Example 3: Scatter with Window (Slice Update)

Operand: `f32[4, 4]` (all zeros)
Scatter indices: `s32[2, 2]` = `[[0, 0], [2, 2]]`
Updates: `f32[2, 2, 2]` = two 2x2 update slices

```
%scatter_dim_nums = ScatterDimensionNumbers(
  update_window_dims = {1, 2},      // last 2 dims of updates are window
  inserted_window_dims = {},         // no collapsed dims
  scatter_dims_to_operand_dims = {0, 1},
  index_vector_dim = 1
)

%result = f32[4, 4] scatter(f32[4,4] %operand, s32[2,2] %indices,
  f32[2,2,2] %updates, %assign_comp, %scatter_dim_nums)
```

Result (with update slices `[[1,2],[3,4]]` and `[[5,6],[7,8]]`):
```
[[1, 2, 0, 0],
 [3, 4, 0, 0],
 [0, 0, 5, 6],
 [0, 0, 7, 8]]
```

### HLO Text Format

```
%result = f32[6]{0} scatter(f32[6]{0} %operand, s32[3,1]{1,0} %indices,
  f32[3]{0} %updates), update_computation=%assign_comp,
  scatter_dimension_numbers={
    update_window_dims=[], inserted_window_dims=[0],
    scatter_dims_to_operand_dims=[0], index_vector_dim=1
  },
  indices_are_sorted=false, unique_indices=false
```

---

## Sort

`Sort` sorts a tensor along a specified dimension. It supports sorting multiple tensors together (key-value sort) using a custom comparator.

### Signature

```
Sort(operands, comparator, dimension, is_stable)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operands` | `std::vector<XlaOp>` | One or more input arrays. All must have the same shape. The first operand determines the sort order; subsequent operands are rearranged to maintain correspondence. |
| `comparator` | `XlaComputation` | A binary comparison function that takes two scalars (for single-operand sort) or `2 * N` scalars (for N-operand sort) and returns a `PRED` (boolean). Returns `true` if the first set of values should come before the second. |
| `dimension` | `int64` | The dimension along which to sort. Default is the last dimension (`rank - 1`). |
| `is_stable` | `bool` | Whether to use a stable sort. If `true`, equal elements maintain their original relative order. Default `false`. |

### Semantics

The sort reorders elements along `dimension` according to the `comparator`. For multi-operand sort, all operands are rearranged identically based on the sort order determined by the comparison of the first operand (or according to the comparator's multi-key logic).

**Output shape**: Same as the input shape for each operand.

### Single-Operand Sort

#### Example: Sort a 1D Array

Input: `f32[5]` = `[3.0, 1.0, 4.0, 1.5, 2.0]`

```
%comp = (f32[], f32[]) -> pred[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %lt = pred[] compare(%a, %b), direction=LT
}

%result = f32[5] sort(f32[5] %operand), to_apply=%comp, dimension=0
```

Result: `f32[5]` = `[1.0, 1.5, 2.0, 3.0, 4.0]`

### Multi-Operand Sort (Key-Value Sort)

Sort keys and rearrange values to match:

```
// Keys: [3, 1, 4, 1, 2]
// Values: ['c', 'a', 'd', 'b', 'e']

%comp = (s32[], s32[], s32[], s32[]) -> pred[] {
  %key_a = s32[] parameter(0)
  %val_a = s32[] parameter(1)  // ignored for comparison
  %key_b = s32[] parameter(2)
  %val_b = s32[] parameter(3)  // ignored for comparison
  ROOT %lt = pred[] compare(%key_a, %key_b), direction=LT
}

%results = (s32[5], s32[5]) sort(s32[5] %keys, s32[5] %values),
  to_apply=%comp, dimension=0
```

Result:
- Sorted keys: `[1, 1, 2, 3, 4]`
- Corresponding values rearranged: `['a', 'b', 'e', 'c', 'd']`

### Stable Sort

When `is_stable = true`, elements that compare equal preserve their input order. This is important for deterministic behavior in multi-key sorting:

```
%result = f32[5] sort(f32[5] %operand), to_apply=%comp,
  dimension=0, is_stable=true
```

For input `[3.0, 1.0, 4.0, 1.5, 2.0]`, stable sort guarantees that if `1.0` appeared before `1.5` in the input (which it does), and we use a custom comparator that considers them equal, `1.0` would still come first.

### HLO Text Format

```
%result = f32[5]{0} sort(f32[5]{0} %operand), to_apply=%comp,
  dimension=0, is_stable=false

// Multi-operand
%results = (s32[5]{0}, s32[5]{0}) sort(s32[5]{0} %keys, s32[5]{0} %values),
  to_apply=%comp, dimension=0, is_stable=false
```

---

## TopK

`TopK` returns the top `k` elements from the last dimension of the input, along with their indices. It is equivalent to a partial sort.

### Signature

```
TopK(operand, k)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. Must be of a sortable type (floating point or integer). |
| `k` | `int64` | The number of top elements to return. Must be in `[1, operand.shape.dimensions[-1]]`. |

### Semantics

`TopK` sorts the last dimension of the operand in descending order and returns the top `k` elements along with their original indices.

**Output**: A tuple `(values, indices)`:
- `values`: Same shape as operand except the last dimension is `k`. Contains the top `k` values in descending order.
- `indices`: Same shape as `values` but with element type `S32`. Contains the original indices of the top elements.

### Example

Input: `f32[6]` = `[3.0, 1.0, 4.0, 1.5, 2.0, 5.0]`
`k = 3`

```
%result = (f32[3], s32[3]) top-k(f32[6] %operand), k=3
```

Result:
- `values`: `f32[3]` = `[5.0, 4.0, 3.0]`
- `indices`: `s32[3]` = `[5, 2, 0]`

### Multi-Dimensional Example

Input: `f32[2, 4]` = `[[3, 1, 4, 2], [7, 5, 8, 6]]`
`k = 2`

```
%result = (f32[2, 2], s32[2, 2]) top-k(f32[2, 4] %operand), k=2
```

Result:
- `values`: `f32[2, 2]` = `[[4, 3], [8, 7]]`
- `indices`: `s32[2, 2]` = `[[2, 0], [2, 0]]`

### HLO Text Format

```
%result = (f32[3]{0}, s32[3]{0}) top-k(f32[6]{0} %operand), k=3
```

### Implementation Notes

`TopK` may be implemented as:
1. A full sort followed by slicing the top `k` elements.
2. A partial selection algorithm (e.g., quickselect) for better efficiency when `k` is much smaller than the input size.
3. A specialized hardware kernel on certain backends.

The choice of implementation depends on the backend and the relationship between `k` and the input size.

---

## StableHLO Cross-References

| XLA Operation | StableHLO Operation | Notes |
|---|---|---|
| Conditional (pred) | `stablehlo.if` | Uses a predicate region |
| Conditional (index) | `stablehlo.case` | Branch index variant |
| While | `stablehlo.while` | Same semantics with condition/body regions |
| Call | `stablehlo.call` | Direct function call |
| Reduce | `stablehlo.reduce` | Same semantics; uses region for computation |
| ReduceWindow | `stablehlo.reduce_window` | Same window semantics |
| Scatter | `stablehlo.scatter` | Same scatter dimension mapping |
| Sort | `stablehlo.sort` | Same multi-operand sort |
| TopK | `stablehlo.top_k` | Same semantics |

### StableHLO Example: Reduce

```mlir
%result = stablehlo.reduce(%operand init: %init_value) applies %reduce_computation
  across dimensions = [0, 1] : (tensor<3x4xf32>, tensor<f32>) -> tensor<f32>
```

Where `%reduce_computation` is a region:

```mlir
stablehlo.reduce applies {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    stablehlo.return %0 : tensor<f32>
}
```

### StableHLO Example: Scatter

```mlir
%result = stablehlo.scatter(%operand, %indices, %updates) ({
  ^bb0(%existing: tensor<f32>, %update: tensor<f32>):
    %0 = stablehlo.add %existing, %update : tensor<f32>
    stablehlo.return %0 : tensor<f32>
}) {
  scatter_dimension_numbers = #stablehlo.scatter<
    update_window_dims = [],
    inserted_window_dims = [0],
    scatter_dims_to_operand_dims = [0],
    index_vector_dim = 1
  >,
  indices_are_sorted = false,
  unique_indices = false
} : (tensor<6xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<6xf32>
```

### StableHLO Example: Sort

```mlir
%result:2 = stablehlo.sort(%keys, %values) ({
  ^bb0(%a_key: tensor<i32>, %a_val: tensor<i32>,
       %b_key: tensor<i32>, %b_val: tensor<i32>):
    %cmp = stablehlo.compare %a_key, %b_key, LT : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
}) {
  dimension = 0 : i64,
  is_stable = false
} : (tensor<5xi32>, tensor<5xi32>) -> (tensor<5xi32>, tensor<5xi32>)
```

---

## Appendix: Operation Comparison

### Reduction Operations Summary

| Operation | Reduces Over | Input | Output | Key Use |
|---|---|---|---|---|
| Reduce | Entire dimensions | Tensor | Smaller tensor | Global aggregation |
| ReduceWindow | Sliding window | Tensor | Tensor (spatially smaller) | Pooling, normalization |
| ReduceScatter | Replicas + dim | Tensor (per replica) | Shard of reduced tensor | Distributed reduction |
| SelectAndScatter | Window (select) + scatter | Tensor + source | Tensor | Pooling gradients |
| Scatter | Indexed positions | Tensor + indices + updates | Tensor | Indexed updates |

### Control Flow Summary

| Operation | Semantics | Key Constraint |
|---|---|---|
| Conditional (pred) | If/else | Branches return same type |
| Conditional (index) | Switch/case | Default is last branch |
| While | Loop with condition | State shape must be invariant |
| Call | Synchronous function call | Parameter types must match |
| CompositeCall | Named composite with decomposition | Backend may override |
