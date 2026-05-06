# XLA HLO IR (High-Level Optimizer IR) Reference

This reference provides comprehensive documentation of XLA's High-Level Optimizer IR (HLO), the intermediate representation that sits between frontend frameworks (JAX, TensorFlow, PyTorch) and backend code generation. HLO is the central data structure for XLA's optimization passes, analysis, and lowering to target-specific code.

---

## 12.1 HLO Module Structure

An XLA program is represented as an `HloModule`, which contains one or more `HloComputation` objects, each composed of `HloInstruction` nodes forming a dataflow graph.

### 12.1.1 HloModule

`HloModule` is the top-level container for an XLA program. It corresponds to a single compilation unit.

Key properties:

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Module name, typically derived from the computation name |
| `entry_computation` | `HloComputation*` | The main computation that produces the program's output |
| `computations` | `std::vector<HloComputation*>` | All computations in the module (entry + called) |
| `config` | `HloModuleConfig` | Compilation configuration (debug options, etc.) |
| `program_shape` | `ProgramShape` | The parameter and result types of the entry computation |
| `input_output_info` | `InputOutputInfo` | Alias analysis for parameters and outputs |

```
HloModule "my_module"
  ENTRY %main (x: f32[128], y: f32[128]) -> f32[128] {
    %x = f32[128] parameter(0)
    %y = f32[128] parameter(1)
    ROOT %result = f32[128] add(%x, %y)
  }
```

#### Module-level operations

- **Cloning**: `module->Clone()` creates a deep copy of the entire module.
- **Unique computation names**: Each computation has a unique name within the module.
- **UniqueId**: Each instruction has a unique ID within the module for identification.
- **Proto serialization**: `module->ToProto()` / `HloModule::CreateFromProto()` for serialization.

### 12.1.2 HloComputation

An `HloComputation` is a named dataflow graph of `HloInstruction` nodes. It corresponds roughly to a function: it takes parameters and produces a single root result.

Key properties:

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Computation name |
| `parameters` | `std::vector<HloInstruction*>` | Parameter instructions |
| `root_instruction` | `HloInstruction*` | The instruction whose output is the computation result |
| `instructions` | `std::vector<HloInstruction*>` | All instructions in the computation |
| `parent` | `HloModule*` | Containing module |

#### Entry Computation

The entry computation is the top-level computation invoked when the compiled program runs. It is identified by the `ENTRY` keyword in the text format:

```
HloModule "example"
  ENTRY %main (p0: f32[64]) -> f32[64] {
    %p0 = f32[64] parameter(0)
    ROOT %result = f32[64] negate(%p0)
  }
```

There is exactly one entry computation per module. Its parameters correspond to the program's inputs, and its root instruction's output is the program's result.

#### Called Computations

Computations other than the entry computation are "called" by instructions that need sub-computations. These include:

| Calling Instruction | Purpose |
|---------------------|---------|
| `while` | Loop body and condition computations |
| `conditional` | Branch computations |
| `call` | Directly invoke a sub-computation |
| `map` | Apply a computation element-wise |
| `reduce` | Reduction computation |
| `scatter` | Scatter update computation |
| `select-and-scatter` | Select and scatter computations |
| `sort` | Comparator computation |
| `fusion` | Fused computation (represents a subgraph) |

Example with called computations:

```
HloModule "while_loop"
  %body (param: (f32[128], u32[])) -> (f32[128], u32[]) {
    %param = (f32[128], u32[]) parameter(0)
    %data = f32[128] get-tuple-element(%param), index=0
    %counter = u32[] get-tuple-element(%param), index=1
    %incremented = f32[128] add(%data, %data)
    %new_counter = u32[] add(%counter, u32[] constant(1))
    ROOT %result = (f32[128], u32[]) tuple(%incremented, %new_counter)
  }

  %condition (param: (f32[128], u32[])) -> pred[] {
    %param = (f32[128], u32[]) parameter(0)
    %counter = u32[] get-tuple-element(%param), index=1
    ROOT %cmp = pred[] compare(%counter, u32[] constant(10)), direction=LT
  }

  ENTRY %main (x: f32[128]) -> f32[128] {
    %x = f32[128] parameter(0)
    %init_counter = u32[] constant(0)
    %init = (f32[128], u32[]) tuple(%x, %init_counter)
    %loop_result = (f32[128], u32[]) while(%init), condition=%condition, body=%body
    ROOT %result = f32[128] get-tuple-element(%loop_result), index=0
  }
```

### 12.1.3 HloInstruction

`HloInstruction` is a node in the computation graph. Each instruction has:

| Property | Type | Description |
|----------|------|-------------|
| `opcode` | `HloOpcode` | The operation type (kAdd, kMultiply, etc.) |
| `shape` | `Shape` | Output shape (type + dimensions + layout) |
| `operands` | `std::vector<HloInstruction*>` | Input operands |
| `name` | `string` | Human-readable name |
| `unique_id` | `int64` | Unique ID within the module |
| `parent` | `HloComputation*` | Containing computation |
| `metadata` | `OpMetadata` | Source location and framework metadata |
| `frontend_attributes` | `FrontendAttributes` | Framework-specific key-value attributes |

#### Instruction Construction

Instructions are typically created via `HloInstruction::Create*` factory methods:

```cpp
// Create an add instruction
auto add = HloInstruction::CreateBinary(
    output_shape, HloOpcode::kAdd, operand0, operand1);

// Create a parameter
auto param = HloInstruction::CreateParameter(
    0, shape, "x");

// Create a constant
auto constant = HloInstruction::CreateConstant(
    LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f}));

// Create a reshape
auto reshape = HloInstruction::CreateReshape(
    new_shape, operand);

// Create a while loop
auto while_op = HloInstruction::CreateWhile(
    shape, condition_computation, body_computation, init);
```

---

## 12.2 HLO Instruction Types (Complete List)

### 12.2.1 Elementwise Unary Operations

Unary operations take a single array operand and apply the operation element-wise.

| Opcode | Name | Description | Supported Types |
|--------|------|-------------|-----------------|
| `kAbs` | `abs` | Absolute value | F16, F32, F64, S8, S16, S32, S64 |
| `kCeil` | `ceil` | Ceiling (round toward +inf) | F16, F32, F64 |
| `kClz` | `clz` | Count leading zeros | U8, U16, U32, U64, S8, S16, S32, S64 |
| `kConvert` | `convert` | Type conversion | All numeric types |
| `kBitcastConvert` | `bitcast-convert` | Bit reinterpretation conversion | All types with same bit width |
| `kCos` | `cosine` | Cosine | F16, F32, F64 |
| `kExp` | `exponential` | e^x | F16, F32, F64, C64, C128 |
| `kExpm1` | `exponential-minus-one` | e^x - 1 | F16, F32, F64 |
| `kFloor` | `floor` | Floor (round toward -inf) | F16, F32, F64 |
| `kImag` | `imag` | Imaginary part of complex | C64, C128 |
| `kIsFinite` | `is-finite` | Test if finite (not NaN/Inf) | F16, F32, F64 |
| `kLog` | `log` | Natural logarithm | F16, F32, F64, C64, C128 |
| `kLog1p` | `log-plus-one` | log(1 + x) | F16, F32, F64 |
| `kNegate` | `negate` | Negation | F16, F32, F64, S8, S16, S32, S64 |
| `kNot` | `not` | Bitwise NOT | U8, U16, U32, U64, S8, S16, S32, S64, PRED |
| `kPopulationCount` | `popcount` | Population count (bit count) | U8, U16, U32, U64, S8, S16, S32, S64 |
| `kReal` | `real` | Real part of complex | C64, C128 |
| `kRoundNearestEven` | `round-nearest-even` | Round to nearest even integer | F16, F32, F64 |
| `kRsqrt` | `rsqrt` | 1 / sqrt(x) | F16, F32, F64 |
| `kSign` | `sign` | Sign function (-1, 0, or 1) | F16, F32, F64, S8, S16, S32, S64 |
| `kSin` | `sine` | Sine | F16, F32, F64 |
| `kSqrt` | `sqrt` | Square root | F16, F32, F64, C64, C128 |
| `kCbrt` | `cbrt` | Cube root | F16, F32, F64 |
| `kTanh` | `tanh` | Hyperbolic tangent | F16, F32, F64, C64, C128 |
| `kLogistic` | `logistic` | Sigmoid: 1 / (1 + e^(-x)) | F16, F32, F64 |

#### Example (Unary)

```
%x = f32[128] parameter(0)
%abs_x = f32[128] abs(%x)
%sqrt_x = f32[128] sqrt(%abs_x)
%neg = f32[128] negate(%x)
%sigmoid = f32[128] logistic(%x)
```

### 12.2.2 Elementwise Binary Operations

Binary operations take two array operands of compatible (broadcastable) shapes.

| Opcode | Name | Description | Supported Types |
|--------|------|-------------|-----------------|
| `kAdd` | `add` | Addition | All numeric |
| `kSubtract` | `subtract` | Subtraction | All numeric |
| `kMultiply` | `multiply` | Multiplication | All numeric |
| `kDivide` | `divide` | Division | All numeric |
| `kRemainder` | `remainder` | Remainder (modulo) | All numeric |
| `kPower` | `power` | Exponentiation (x^y) | F16, F32, F64, C64, C128 |
| `kMaximum` | `maximum` | Element-wise maximum | All numeric |
| `kMinimum` | `minimum` | Element-wise minimum | All numeric |
| `kAnd` | `and` | Bitwise AND | U8, U16, U32, U64, S8, S16, S32, S64, PRED |
| `kOr` | `or` | Bitwise OR | U8, U16, U32, U64, S8, S16, S32, S64, PRED |
| `kXor` | `xor` | Bitwise XOR | U8, U16, U32, U64, S8, S16, S32, S64, PRED |
| `kShiftLeft` | `shift-left` | Left shift | U8, U16, U32, U64, S8, S16, S32, S64 |
| `kShiftRightArithmetic` | `shift-right-arithmetic` | Arithmetic right shift (sign-extending) | U8, U16, U32, U64, S8, S16, S32, S64 |
| `kShiftRightLogical` | `shift-right-logical` | Logical right shift (zero-extending) | U8, U16, U32, U64, S8, S16, S32, S64 |
| `kAtan2` | `atan2` | Two-argument arctangent | F16, F32, F64 |
| `kCompare` | `compare` | Comparison | All numeric, PRED |

#### Compare Operation

The `compare` operation uses a `direction` attribute:

| Direction | Meaning |
|-----------|---------|
| `EQ` | Equal |
| `NE` | Not equal |
| `GE` | Greater or equal |
| `GT` | Greater than |
| `LE` | Less or equal |
| `LT` | Less than |

```
%cmp = pred[128] compare(%x, %y), direction=LT
```

#### Broadcasting

Binary operations implicitly support broadcasting. When operands have different ranks or dimension sizes, XLA applies NumPy-style broadcasting rules:

```
// Scalar + array
%s = f32[] constant(1.0)
%a = f32[128] parameter(0)
%r = f32[128] add(%s, %a)   // scalar broadcast to f32[128]

// Array[128,1] + Array[1,64] -> Array[128,64]
%m1 = f32[128, 1] parameter(0)
%m2 = f32[1, 64] parameter(1)
%r = f32[128, 64] add(%m1, %m2)
```

### 12.2.3 Elementwise Ternary Operations

| Opcode | Name | Description |
|--------|------|-------------|
| `kClamp` | `clamp` | Clamp value to [min, max] range |
| `kSelect` | `select` | Conditional selection (ternary) |

#### Clamp

```
%result = T clamp(%min, %operand, %max)
```

All three operands must have the same shape (or broadcastable shapes). Each element of the result is `max(min_val, min(max_val, operand_val))`.

```
%val = f32[128] parameter(0)
%lo = f32[] constant(0.0)
%hi = f32[] constant(1.0)
%clamped = f32[128] clamp(%lo, %val, %hi)  // clamp to [0.0, 1.0]
```

#### Select

```
%result = T select(%pred, %on_true, %on_false)
```

- `pred` must be of type PRED (boolean).
- `on_true` and `on_false` must have the same shape.
- Element-wise: for each position, selects `on_true` if `pred` is true, else `on_false`.

```
%mask = pred[128] compare(%x, %y), direction=GT
%result = f32[128] select(%mask, %x, %y)  // max without calling maximum
```

### 12.2.4 Data Movement Operations

These operations rearrange or reshape data without performing arithmetic.

#### Broadcast (kBroadcast)

Adds new dimensions by replicating data along existing dimensions.

```
%result = T[D0..Dn] broadcast(%operand), dimensions={d0,d1,...}
```

- `dimensions` specifies which dimensions of the result correspond to the operand's dimensions.
- Dimensions not in the set are broadcast (size must be > 1 in the result).

```
%scalar = f32[] constant(5.0)
%vec = f32[128] broadcast(%scalar), dimensions={}       // broadcast scalar to vector

%mat = f32[128, 64] parameter(0)
%cube = f32[128, 64, 32] broadcast(%mat), dimensions={0,1}  // broadcast along new dim 2
```

#### Reshape (kReshape)

Reinterprets the data with a new shape. The total number of elements must remain the same.

```
%result = T[new_shape] reshape(%operand)
```

```
%flat = f32[1024] parameter(0)
%mat = f32[32, 32] reshape(%flat)
%cube = f32[8, 8, 16] reshape(%mat)
```

Reshape may rearrange elements if the layout changes. For a layout-preserving reshape (dimensions are collapsed or expanded in order), no data movement occurs.

#### Transpose (kTranspose)

Permutes the dimensions of the operand.

```
%result = T[D0, D1, ...] transpose(%operand), permutation={p0, p1, ...}
```

- `permutation` is a permutation of [0, 1, ..., rank-1].
- The result's dimension `i` has size `operand.dimensions()[permutation[i]]`.

```
%mat = f32[128, 64] parameter(0)
%transposed = f32[64, 128] transpose(%mat), permutation={1, 0}
```

#### Slice (kSlice)

Extracts a sub-array by specifying start indices, limit indices, and strides for each dimension.

```
%result = T[...] slice(%operand), slice={[start:limit:stride], ...}
```

```
%mat = f32[128, 64] parameter(0)
%sub = f32[32, 16] slice(%mat), slice={[0:128:4], [8:24:1]}
// Takes every 4th row, columns 8 through 23
```

#### DynamicSlice (kDynamicSlice)

Like `Slice`, but start indices are runtime values.

```
%result = T[...] dynamic-slice(%operand, %start_index0, %start_index1, ...),
    dynamic_slice_sizes={s0, s1, ...}
```

```
%mat = f32[128, 64] parameter(0)
%i = s32[] parameter(1)
%j = s32[] parameter(2)
%sub = f32[8, 8] dynamic-slice(%mat, %i, %j),
    dynamic_slice_sizes={8, 8}
```

#### DynamicUpdateSlice (kDynamicUpdateSlice)

Replaces a slice of the operand with update data at a dynamic position.

```
%result = T[D0...] dynamic-update-slice(%operand, %update, %start0, %start1, ...)
```

```
%mat = f32[128, 64] parameter(0)
%patch = f32[8, 8] parameter(1)
%i = s32[] parameter(2)
%j = s32[] parameter(3)
%updated = f32[128, 64] dynamic-update-slice(%mat, %patch, %i, %j)
```

#### Concatenate (kConcatenate)

Joins arrays along a specified dimension.

```
%result = T[...] concatenate(%op0, %op1, ...), dimensions=d
```

All operands must have the same shape except along dimension `d`.

```
%a = f32[32, 64] parameter(0)
%b = f32[32, 64] parameter(1)
%joined = f32[64, 64] concatenate(%a, %b), dimensions=0
```

#### Pad (kPad)

Pads an array with a value according to padding configuration.

```
%result = T[...] pad(%operand, %padding_value), padding={[lo, hi, interior], ...}
```

```
%img = f32[224, 224, 3] parameter(0)
%zero = f32[] constant(0.0)
%padded = f32[230, 230, 3] pad(%img, %zero), padding={[3,3,0], [3,3,0], [0,0,0]}
```

#### Reverse (kReverse)

Reverses elements along specified dimensions.

```
%result = T[...] reverse(%operand), dimensions={d0, d1, ...}
```

```
%vec = f32[128] parameter(0)
%rev = f32[128] reverse(%vec), dimensions={0}
```

#### Gather (kGather)

Gathers elements from the operand according to an index array. This is a generalization of indexing.

```
%result = T[...] gather(%operand, %indices),
    gather_dims_to_operand_dims={d0, d1, ...},
    index_vector_dim=d,
    offset_dims={...},
    slice_sizes={s0, s1, ...},
    collapsed_slice_dims={...},
    start_index_map={...}
```

Gather parameters:

| Parameter | Description |
|-----------|-------------|
| `offset_dims` | Which dimensions of the result are offset into the gathered slice |
| `collapsed_slice_dims` | Which slice dimensions are collapsed (must have size 1) |
| `start_index_map` | Maps index vector dimensions to operand dimensions |
| `index_vector_dim` | Which dimension of indices is the index vector |
| `slice_sizes` | Size of the slice gathered at each index |

#### Scatter (kScatter)

Scatter updates into the operand according to indices.

```
%result = T[...] scatter(%operand, %indices, %updates),
    to_apply=%update_computation,
    scatter_dims_to_operand_dims={...},
    index_vector_dim=d,
    update_window_dims={...},
    inserted_window_dims={...}
```

#### Copy (kCopy)

Creates a copy of the operand. Used for layout changes or to explicitly materialize a value.

```
%result = T copy(%operand)
```

### 12.2.5 Control Flow Operations

#### While (kWhile)

Implements a while loop with a condition and body computation.

```
%result = T while(%init), condition=%cond, body=%body
```

| Component | Type | Description |
|-----------|------|-------------|
| `init` | `T` | Initial loop carry value |
| `condition` | `HloComputation` | Takes `T`, returns `pred[]` |
| `body` | `HloComputation` | Takes `T`, returns `T` |
| Result | `T` | Final carry value |

Semantics: `while(cond(init)) { init = body(init) }`. The condition computation is evaluated before each iteration. The loop terminates when the condition returns `false`.

#### Conditional (kConditional)

Implements a conditional branch (if/else or switch).

```
// If/else form
%result = T conditional(%pred, %true_val, %false_val),
    true_computation=%true_branch, false_computation=%false_branch

// Switch form (indexed conditional)
%result = T conditional(%index, %branch_val0, %branch_val1, ...),
    branches={%branch0, %branch1, ...}
```

| Form | Description |
|------|-------------|
| Binary | `pred` selects between two branches |
| Indexed | `index` (s32) selects among N branches; if out of range, last branch is taken |

#### Call (kCall)

Directly invokes a sub-computation.

```
%result = T call(%arg0, %arg1, ...), to_apply=%subcomputation
```

The sub-computation's parameters map to the call's operands.

#### Tuple (kTuple) and GetTupleElement (kGetTupleElement)

```
// Create a tuple
%tuple = (T0, T1, ...) tuple(%val0, %val1, ...)

// Extract an element
%elem = Tk get-tuple-element(%tuple), index=k
```

### 12.2.6 Collective Operations

Collective operations implement inter-device communication patterns.

#### AllReduce (kAllReduce)

Reduces values across all replicas/participants using a reduction computation.

```
%result = T all-reduce(%operand), to_apply=%reduction,
    channel_id=c, replica_groups={{0,1},{2,3}}, use_global_device_ids=false
```

| Parameter | Description |
|-----------|-------------|
| `to_apply` | Reduction computation (e.g., add) |
| `channel_id` | Optional channel for cross-computation communication |
| `replica_groups` | Groups of replicas that reduce together |
| `use_global_device_ids` | Whether replica IDs are global device IDs |

#### AllGather (kCollectivePermute / kAllGather)

Concatenates values from all participants along a specified dimension.

```
%result = T all-gather(%operand), all_gather_dim=d,
    channel_id=c, replica_groups={{0,1},{2,3}}
```

#### AllToAll (kAllToAll)

Splits each participant's operand along a split dimension and sends each shard to a different participant, then concatenates received shards along a concat dimension.

```
%result = T all-to-all(%operand), split_dimension=d0, concat_dimension=d1,
    split_count=n, replica_groups={{...}}
```

#### CollectivePermute (kCollectivePermute)

Sends data between specific pairs of replicas.

```
%result = T collective-permute(%operand),
    source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
```

Each pair `{src, tgt}` means replica `src` sends to replica `tgt`.

#### ReduceScatter (kReduceScatter)

Performs an all-reduce followed by a scatter (each participant gets a different shard of the result).

```
%result = T reduce-scatter(%operand), to_apply=%reduction,
    scatter_dimension=d, channel_id=c, replica_groups={{...}}
```

### 12.2.7 Convolution and Dot

#### Convolution (kConvolution)

General n-dimensional convolution.

```
%result = T[...] convolution(%lhs, %rhs),
    window={size=[kh, kw], stride=[sh, sw], pad=[[ph_lo, ph_hi], [pw_lo, pw_hi]],
            lhs_dilate=[ldh, ldw], rhs_dilate=[rdh, rdw],
            window_reversal=[0, 0]},
    dim_labels=b_i_o_f01_x_i_o->b_o_f01,
    batch_group_count=1,
    feature_group_count=1
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `window` | Kernel size, stride, padding, dilation |
| `dim_labels` | Dimension labeling for input, kernel, and output |
| `batch_group_count` | Number of batch groups (for grouped convolution) |
| `feature_group_count` | Number of feature groups (for depthwise/grouped convolution) |
| `precision_config` | Precision hints for backend |

The `dim_labels` string encodes the dimension layout:
- `b` = batch, `f` = feature, `i` = input feature, `o` = output feature
- `0..9` = spatial dimensions
- Format: `lhs_labels_rhs_labels->output_labels`

#### Dot (kDot)

Matrix multiplication and general dot product.

```
%result = T[...] dot(%lhs, %rhs),
    dot_dimension_numbers={lhs_batching={0}, rhs_batching={0},
                          lhs_contracting={2}, rhs_contracting={1}},
    precision_config={HIGH, HIGH}
```

| Parameter | Description |
|-----------|-------------|
| `lhs_batching_dims` | Batch dimensions in LHS (contracted with RHS batch dims) |
| `rhs_batching_dims` | Batch dimensions in RHS |
| `lhs_contracting_dims` | Contracting dimensions in LHS (summed over) |
| `rhs_contracting_dims` | Contracting dimensions in RHS |
| `precision_config` | Precision hints (DEFAULT, HIGH, HIGHEST) |

Simple matrix multiply:

```
%a = f32[128, 64] parameter(0)  // M x K
%b = f32[64, 256] parameter(1)  // K x N
%c = f32[128, 256] dot(%a, %b),
    dot_dimension_numbers={lhs_contracting={1}, rhs_contracting={0}}
```

### 12.2.8 Reduce and ReduceWindow

#### Reduce (kReduce)

Reduces an array along specified dimensions using a reduction computation.

```
%result = T[...] reduce(%operand, %init_value), dimensions={d0, d1, ...},
    to_apply=%reduction_computation
```

```
%mat = f32[128, 64] parameter(0)
%zero = f32[] constant(0.0)
%row_sums = f32[128] reduce(%mat, %zero), dimensions={1},
    to_apply=%add_computation
```

The reduction computation takes two scalars of type T and returns one scalar of type T.

#### ReduceWindow (kReduceWindow)

Reduces within a sliding window over the operand.

```
%result = T[...] reduce-window(%operand, %init_value), window={size=[wh, ww],
    stride=[sh, sw], pad=[[ph_lo, ph_hi], [pw_lo, pw_hi]], window_dilation=[dh, dw]},
    to_apply=%reduction_computation
```

Commonly used for pooling operations:

```
%img = f32[1, 224, 224, 3] parameter(0)
%neg_inf = f32[] constant(-inf)
%max_pooled = f32[1, 112, 112, 3] reduce-window(%img, %neg_inf),
    window={size=[1, 2, 2, 1], stride=[1, 2, 2, 1]},
    to_apply=%max_computation
```

### 12.2.9 Custom Calls (kCustomCall)

See File 11 for comprehensive documentation. Summary:

```
%result = T custom-call(%op0, %op1, ...),
    call_target_name="name",
    has_side_effect=false,
    backend_config="{...}",
    api_version=2
```

### 12.2.10 Fusion (kFusion)

Fusion represents a subgraph of instructions that will be emitted as a single kernel. Fusion is created by XLA's fusion optimization pass.

```
%result = T fusion(%op0, %op1, ...), kind=kLoop,
    fused_computation=%fused_comp
```

Fusion kinds:

| Kind | Description |
|------|-------------|
| `kLoop` | Loop fusion: all fused ops are elementwise or broadcast |
| `kInput` | Input fusion: a reduce or reduce-window fused with its producers |
| `kOutput` | Output fusion: multi-output fusion producing several results |
| `kCustom` | Backend-specific fusion pattern |
| `kTransposeDot` | Dot fused with transposes of its operands |
| `kDot` | Dot operation fused with surrounding elementwise ops |

Example of a fused computation:

```
%fused_computation (param_0: f32[128], param_1: f32[128]) -> f32[128] {
  %p0 = f32[128] parameter(0)
  %p1 = f32[128] parameter(1)
  %add = f32[128] add(%p0, %p1)
  ROOT %relu = f32[128] maximum(%add, f32[128] broadcast(f32[] constant(0)))
}

ENTRY %main {
  %x = f32[128] parameter(0)
  %y = f32[128] parameter(1)
  ROOT %result = f32[128] fusion(%x, %y), kind=kLoop,
      calls=%fused_computation
}
```

### 12.2.11 Sort (kSort)

Sorts an array or multiple arrays together using a comparator computation.

```
%result = (T0, T1, ...) sort(%op0, %op1, ...), dimension=d,
    is_stable=true, to_apply=%comparator
```

- `dimension`: The dimension along which to sort.
- `is_stable`: Whether to use a stable sort.
- `comparator`: Takes two pairs `(T0, T1)` and returns `pred[]`.

```
%keys = s32[100] parameter(0)
%values = f32[100] parameter(1)
%sorted = (s32[100], f32[100]) sort(%keys, %values), dimension=0,
    is_stable=true, to_apply=%compare_keys
```

### 12.2.12 TopK (kTopK)

Returns the top-k elements and their indices.

```
%result = (T[k], s32[k]) topk(%operand), k=k
```

### 12.2.12 Async Operations

Async operations decompose a long-running operation into three phases: start (launch), update (optional intermediate step), and done (wait for completion).

#### AsyncStart

```
%async_token = async-start(%operand0, %operand1, ...), async_execution_group=N
```

Launches the async sub-computation. Returns a tuple containing the operands and an async token.

#### AsyncUpdate

```
%updated_token = async-update(%async_token)
```

Optional intermediate step for operations that support incremental computation.

#### AsyncDone

```
%result = T async-done(%async_token)
```

Waits for the async operation to complete and returns the result.

Example:

```
ENTRY %main {
  %a = f32[128, 128] parameter(0)
  %b = f32[128, 128] parameter(1)

  // Start async dot product
  %async = (f32[128,128], f32[128,128], token[]) async-start(%a, %b),
      async_execution_group=0, calls=%dot_computation

  // Do other work here...
  %other = f32[128] parameter(2)

  // Wait for dot result
  ROOT %result = f32[128, 128] async-done(%async), calls=%dot_computation
}
```

### 12.2.13 Token Operations

| Opcode | Name | Description |
|--------|------|-------------|
| `kAfterAll` | `after-all` | Creates a token ordered after inputs |
| `kInfeed` | `infeed` | Reads data from host |
| `kOutfeed` | `outfeed` | Writes data to host |
| `kSend` | `send` | Sends data to another device |
| `kSendDone` | `send-done` | Waits for send to complete |
| `kRecv` | `recv` | Receives data from another device |
| `kRecvDone` | `recv-done` | Waits for recv to complete |

See File 11 for detailed documentation.

### 12.2.14 Other Operations

| Opcode | Name | Description |
|--------|------|-------------|
| `kParameter` | `parameter` | Input parameter |
| `kConstant` | `constant` | Constant literal |
| `kGetTupleElement` | `get-tuple-element` | Extract tuple element |
| `kRngBitGenerator` | `rng-bit-generator` | Generate random bits |
| `kRngGetAndUpdateState` | `rng-get-and-update-state` | Access global RNG state |
| `kBitcast` | `bitcast` | Reinterpret bits |
| `kAddDependency` | `add-dependency` | Add execution dependency |
| `kDomain` | `domain` | Scheduling domain boundary |
| `kGetDimensionSize` | `get-dimension-size` | Query dynamic dimension size |
| `kSetDimensionSize` | `set-dimension-size` | Set dynamic dimension size |
| `kOptimizationBarrier` | `optimization-barrier` | Prevent optimization reordering |
| `kReplicaId` | `replica-id` | Get replica ID |
| `kPartitionId` | `partition-id` | Get partition ID |
| `kTrace` | `trace` | Debug trace |
| `kMap` | `map` | Apply computation element-wise |
| `kBatchGroupGrad` | `batch-group-grad` | Gradient batch grouping |

---

## 12.3 HLO Text Format

The HLO text format is a human-readable representation used for debugging, logging, and testing. Understanding how to read HLO dumps is essential for performance analysis and debugging.

### 12.3.1 Module Header

```
HloModule module_name, entry_computation_layout={(f32[128]{0}, f32[128]{0})->f32[128]{0}}
```

The header includes:
- Module name
- Optional entry computation layout specifying parameter and result shapes with layouts

### 12.3.2 Computation Format

```
%computation_name (param0_name: type0, param1_name: type1) -> result_type {
  %name0 = type instruction(...)
  %name1 = type instruction(...)
  ROOT %result_name = type instruction(...)
}
```

- `ROOT` marks the instruction whose output is the computation result.
- Each instruction is assigned a name (`%name`).
- Instructions are typically in topological order.

### 12.3.3 Shape and Layout Notation

Shapes in HLO text format:

```
element_type[dim0, dim1, ...]{layout_dim0, layout_dim1, ...}
```

- `element_type`: f16, f32, f64, bf16, s8, s16, s32, s64, u8, u16, u32, u64, pred, c64, c128, token, opaque
- `[dim0, dim1, ...]`: Dimension sizes
- `{layout_dim0, layout_dim1, ...}`: Layout (minor-to-major dimension order)

Examples:

```
f32[128]{0}                    // 1-D array, 128 elements, dim 0 is minor
f32[128, 64]{1, 0}            // 2-D array (128x64), row-major (dim 1 major, dim 0 minor)
f32[128, 64]{0, 1}            // 2-D array (128x64), column-major (dim 0 major, dim 1 minor)
(f32[128]{0}, s32[]{})        // Tuple of f32[128] and scalar s32
token                          // Token type
f32[128, 64, 3]{2, 1, 0}     // 3-D array, dim 2 is major, dim 0 is minor
```

**Layout interpretation**: The layout numbers in `{}` list dimensions from minor (fastest-varying, innermost) to major (slowest-varying, outermost). For a 2-D matrix:
- `{1, 0}` = row-major (C convention): elements in the same row are contiguous. The rightmost index varies fastest.
- `{0, 1}` = column-major (Fortran convention): elements in the same column are contiguous. The leftmost index varies fastest.

**Tiled layouts** (GPU-specific):
```
f32[128, 64]{1, 0:T(128,64)}
f32[256, 256]{1, 0, 2, 3:T(2,128,4,64)(2,1)}
```

### 12.3.4 Example Modules

#### Simple Elementwise Module

```
HloModule simple_add

ENTRY %main (x: f32[128], y: f32[128]) -> f32[128] {
  %x = f32[128]{0} parameter(0)
  %y = f32[128]{0} parameter(1)
  ROOT %result = f32[128]{0} add(%x, %y)
}
```

#### Module with Called Computations

```
HloModule reduce_sum

%add_scalar (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %sum = f32[] add(%x, %y)
}

ENTRY %main (x: f32[128, 64]) -> f32[128] {
  %x = f32[128, 64]{1, 0} parameter(0)
  %zero = f32[] constant(0.0)
  ROOT %reduced = f32[128]{0} reduce(%x, %zero), dimensions={1},
      to_apply=%add_scalar
}
```

#### Module with While Loop

```
HloModule while_example

%body (param: (f32[128], s32[])) -> (f32[128], s32[]) {
  %param = (f32[128]{0}, s32[]{}) parameter(0)
  %data = f32[128]{0} get-tuple-element(%param), index=0
  %i = s32[] get-tuple-element(%param), index=1
  %doubled = f32[128]{0} add(%data, %data)
  %next_i = s32[] add(%i, s32[] constant(1))
  ROOT %out = (f32[128]{0}, s32[]{}) tuple(%doubled, %next_i)
}

%cond (param: (f32[128], s32[])) -> pred[] {
  %param = (f32[128]{0}, s32[]{}) parameter(0)
  %i = s32[] get-tuple-element(%param), index=1
  ROOT %test = pred[] compare(%i, s32[] constant(10)), direction=LT
}

ENTRY %main (x: f32[128]) -> f32[128] {
  %x = f32[128]{0} parameter(0)
  %init_i = s32[] constant(0)
  %init = (f32[128]{0}, s32[]{}) tuple(%x, %init_i)
  %loop = (f32[128]{0}, s32[]{}) while(%init), condition=%cond, body=%body
  ROOT %result = f32[128]{0} get-tuple-element(%loop), index=0
}
```

### 12.3.5 Reading HLO Dumps

XLA generates HLO dumps during compilation. Key dump types:

| Dump Stage | Description |
|------------|-------------|
| `before_optimizations` | HLO before any optimization passes |
| `after_optimizations` | HLO after target-independent optimizations |
| `before_backend_optimizations` | HLO before backend-specific passes |
| `after_backend_optimizations` | HLO after backend passes |
| `after_fusion` | HLO after fusion optimization |
| `after_layout` | HLO after layout assignment |
| `after_scheduling` | HLO after instruction scheduling |

To enable HLO dumps:

```python
# JAX
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_pass_re=.*'

# Or more selectively:
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_pass_re=after_optimizations'
```

HLO dump files:
- `module_name.before_optimizations.txt` -- Human-readable text
- `module_name.before_optimizations.pb` -- Binary protobuf
- `module_name.before_optimizations.html` -- HTML visualization (graph)

---

## 12.4 HLO Verification

The `HloVerifier` checks invariants of the HLO module after each optimization pass. Understanding these invariants is important for anyone writing HLO passes or debugging verification failures.

### 12.4.1 Structural Invariants

| Invariant | Description |
|-----------|-------------|
| **Unique instruction IDs** | Every instruction has a unique ID within the module |
| **Parent consistency** | Each instruction's parent matches the computation it belongs to |
| **Operand uniqueness** | No instruction appears as an operand of itself |
| **Root instruction** | Every computation has exactly one root instruction |
| **Entry computation** | The module has exactly one entry computation |
| **No cycles** | The computation graph is a DAG (no cycles) |
| **Called computation ownership** | Called computations belong to the same module |

### 12.4.2 Shape Invariants

| Invariant | Description |
|-----------|-------------|
| **Shape compatibility** | Each instruction's result shape matches what its opcode and operands imply |
| **Operand shape matching** | Operand shapes are compatible with the instruction's requirements |
| **Layout consistency** | Layouts are valid for the shape dimensions |
| **Tuple shape correctness** | Tuple shapes have the correct number of elements |

### 12.4.3 Opcode-Specific Invariants

| Opcode | Invariant |
|--------|-----------|
| `kReduce` | `init_value` shape is a scalar matching the element type; reduction computation takes two scalars and returns one scalar |
| `kWhile` | Condition returns `pred[]`; body input/output types match; init type matches |
| `kConditional` | Branch computations have matching input/output types |
| `kDot` | Contracting dimensions have matching sizes; result shape is correct |
| `kConvolution` | Window and dimension labels are valid; feature group count divides correctly |
| `kBroadcast` | Broadcast dimensions are valid and match the operand rank |
| `kReshape` | Total element count is preserved |
| `kTranspose` | Permutation is a valid permutation of [0, rank) |
| `kSlice` | Start <= limit <= dimension size; stride >= 1 |
| `kGather` | All gather parameters are consistent |
| `kScatter` | All scatter parameters are consistent |
| `kConcatenate` | All operands have the same shape except on the concatenation dimension |
| `kFusion` | Fused computation's parameters match the fusion's operands; fused root matches fusion output |
| `kCustomCall` | Call target name is non-empty |

### 12.4.4 Verification Modes

```cpp
enum class HloVerifierOptimizationMode {
  kFull,            // Verify all invariants
  kPostSimplify,    // Skip some checks after simplification
};
```

The verifier can be configured to check different levels of strictness depending on the compilation stage.

---

## 12.5 HLO Instruction Properties

### 12.5.1 HasSideEffect

An instruction has a side effect if its execution changes state beyond producing its output value. Side-effecting instructions cannot be eliminated as dead code.

Side-effecting opcodes:

| Opcode | Side Effect |
|--------|-------------|
| `kInfeed` | Reads from host queue |
| `kOutfeed` | Writes to host queue |
| `kSend` | Sends data to another device |
| `kRecv` | Receives data from another device |
| `kCustomCall` | Has side effect if `has_side_effect=true` |
| `kRngGetAndUpdateState` | Mutates global PRNG state |
| `kTrace` | Logs to debug output |
| `kSendDone` | Waits for send completion |
| `kRecvDone` | Waits for recv completion |

Non-side-effecting instructions are pure and can be eliminated if their results are unused.

### 12.5.2 IsCustomCall

```cpp
bool HloInstruction::IsCustomCall() const {
  return opcode() == HloOpcode::kCustomCall;
}
```

Custom calls are identified by their opcode and further characterized by:
- `call_target_name()`: The name of the target function
- `has_side_effect()`: Whether the custom call has side effects
- `api_version()`: The FFI API version
- `backend_config()`: Backend-specific configuration string or proto

### 12.5.3 Opcode Enum

The `HloOpcode` enum (defined in `xla/service/hlo_opcode.h`) contains all valid HLO instruction types:

```cpp
enum class HloOpcode {
  kAbs = 0,
  kAdd,
  kAfterAll,
  kAllGather,
  kAllReduce,
  kAllToAll,
  kAnd,
  kAtan2,
  kBatchGroupGrad,
  kBitcast,
  kBitcastConvert,
  kBroadcast,
  kCall,
  kCeil,
  kClamp,
  kClz,
  kCollectivePermute,
  kCompare,
  kComplex,
  kConcatenate,
  kConditional,
  kConstant,
  kConvert,
  kConvolution,
  kCopy,
  kCopyDone,
  kCopyStart,
  kCosine,
  kCbrt,
  kCustomCall,
  kDivide,
  kDomain,
  kDot,
  kDynamicSlice,
  kDynamicUpdateSlice,
  kExp,
  kExpm1,
  kFloor,
  kFusion,
  kGather,
  kGetDimensionSize,
  kGetTupleElement,
  kImag,
  kInfeed,
  kIota,
  kIsFinite,
  kLog,
  kLog1p,
  kLogistic,
  kMap,
  kMaximum,
  kMinimum,
  kMultiply,
  kNegate,
  kNot,
  kOptimizationBarrier,
  kOr,
  kOutfeed,
  kPad,
  kPartitionId,
  kPopulationCount,
  kPower,
  kReal,
  kReduce,
  kReduceScatter,
  kReduceWindow,
  kRemainder,
  kReplicaId,
  kReshape,
  kReverse,
  kRngBitGenerator,
  kRngGetAndUpdateState,
  kRoundNearestEven,
  kRsqrt,
  kScatter,
  kSelect,
  kSelectAndScatter,
  kSend,
  kSendDone,
  kSetDimensionSize,
  kShiftLeft,
  kShiftRightArithmetic,
  kShiftRightLogical,
  kSign,
  kSine,
  kSlice,
  kSort,
  kSqrt,
  kSubtract,
  kTanh,
  kTopK,
  kTranspose,
  kTuple,
  kWhile,
  kXor,
};
```

Each opcode has a string representation (e.g., `HloOpcodeToString(kAdd)` returns `"add"`).

### 12.5.4 Shape Inference

Each HLO instruction has its output shape computed from its opcode, operand shapes, and attributes. The `ShapeInference` class provides static methods for computing output shapes:

```cpp
class ShapeInference {
 public:
  static StatusOr<Shape> InferUnaryOpShape(HloOpcode opcode,
                                           const Shape& operand);
  static StatusOr<Shape> InferBinaryOpShape(HloOpcode opcode,
                                            const Shape& lhs,
                                            const Shape& rhs);
  static StatusOr<Shape> InferDotShape(const Shape& lhs, const Shape& rhs,
                                       const DotDimensionNumbers& dnums);
  static StatusOr<Shape> InferConvolveShape(const Shape& lhs, const Shape& rhs,
                                             const ConvolutionDimensionNumbers& dnums,
                                             const Window& window);
  static StatusOr<Shape> InferReduceShape(const Shape& operand,
                                           const Shape& init_value,
                                           absl::Span<const int64_t> dimensions);
  static StatusOr<Shape> InferReshapeShape(const Shape& operand,
                                            const Shape& new_shape);
  static StatusOr<Shape> InferTransposeShape(const Shape& operand,
                                              const std::vector<int64_t>& permutation);
  // ... many more
};
```

Shape inference is used by:
1. **Instruction construction** -- Verifies that the specified output shape matches the inferred shape.
2. **HloVerifier** -- Checks that instruction shapes are consistent.
3. **Optimization passes** -- Computes shapes for newly created instructions.

### 12.5.5 Instruction Metadata

Each instruction carries metadata that traces back to the source program:

```cpp
struct OpMetadata {
  std::string op_type;        // e.g., "addmm", "convolution"
  std::string op_name;        // e.g., "layers.0.self_attention"
  std::string source_file;
  int32_t source_line = 0;
  // Framework-specific metadata
};
```

This metadata is propagated through optimization passes and is invaluable for debugging:

```
%x = f32[128]{0} parameter(0), metadata={op_type="input" op_name="x"}
%w = f32[128, 64]{1, 0} parameter(1), metadata={op_type="weight" op_name="linear.weight"}
%dot = f32[64]{0} dot(%x, %w), metadata={op_type="addmm" op_name="linear" source_file="model.py" source_line=42}
```

### 12.5.6 Frontend Attributes

Frontend attributes are key-value string pairs that frameworks attach to instructions to convey additional semantics:

```
%result = f32[128] add(%x, %y),
    frontend_attributes={_xla_old_op_type="bias_add"}
```

Common frontend attributes:

| Key | Description |
|-----|-------------|
| `_xla_old_op_type` | Original operation type from the framework |
| `_xla_sharding` | Sharding annotation |
| `_xla_backend_config` | Backend-specific configuration |
| `_xla_old_op_name` | Original operation name |

Frontend attributes are generally preserved through optimization passes unless the instruction is eliminated.

---

## 12.6 HLO Module Config

The `HloModuleConfig` contains compilation parameters:

| Field | Type | Description |
|-------|------|-------------|
| `entry_computation_layout` | `ComputationLayout` | Parameter and result layouts for the entry computation |
| `replica_count` | `int64` | Number of replicas |
| `num_partitions` | `int64` | Number of SPMD partitions |
| `debug_options` | `DebugOptions` | Compilation flags and options |
| `seed` | `uint64` | Seed for deterministic compilation |
| `alias_config` | `HloInputOutputAliasConfig` | Parameter-output alias information |

### DebugOptions (Key Options)

| Option | Type | Description |
|--------|------|-------------|
| `xla_dump_to` | string | Directory for HLO dumps |
| `xla_dump_hlo_pass_re` | string | Regex for which passes to dump |
| `xla_disable_all_hlo_passes` | bool | Disable all optimization passes |
| `xla_gpu_enable_triton_gemm` | bool | Enable Triton GEMM codegen |
| `xla_gpu_autotune_level` | int | Autotuning aggressiveness |
| `xla_force_all_intermediate_buffer_occupancy` | bool | Force all buffers to be in-memory |

---

## 12.7 HLO Pass Infrastructure

XLA optimizations are implemented as `HloPassInterface` implementations that transform the HLO module.

### Pass Interface

```cpp
class HloPassInterface {
 public:
  virtual ~HloPassInterface() = default;
  virtual absl::string_view name() const = 0;
  virtual StatusOr<bool> Run(HloModule* module,
                             const RunOptions& options) = 0;
};
```

- Returns `true` if the module was modified, `false` if it was unchanged.
- Passes are composed into `HloPassPipeline` for ordered execution.

### Key Optimization Passes

| Pass | Description |
|------|-------------|
| `HloCSE` | Common subexpression elimination |
| `HloDCE` | Dead code elimination |
| `AlgebraicSimplifier` | Algebraic simplifications (identity ops, constant folding) |
| `DotMerger` | Merge consecutive dot operations |
| `DotDecomposer` | Decompose dots for better parallelism |
| `FusionMerger` | Merge small fusions |
| `InstructionFusion` | Create fusion instructions |
| `LayoutAssignment` | Assign memory layouts |
| `Scheduling` | Schedule instruction execution order |
| `SpmdPartitioner` | Partition HLO for SPMD execution |
| `AllGatherDegenerateDimRemover` | Remove degenerate dimensions in all-gather |
| `ConvolutionPredExpander` | Expand predicates in convolutions |
| `SortSimplifier` | Simplify sort operations |
| `TupleSimplifier` | Simplify tuple/get-tuple-element patterns |
| `WhileLoopSimplifier` | Simplify while loop structures |
| `CallInliner` | Inline call instructions |
| `ZeroSizedHloElimination` | Remove zero-sized operations |
| `Defuser` | Break apart fusion instructions |
| `RearLayoutAssignment` | Insert copy instructions for layout changes |
| `HloRematerialization` | Rematerialize instructions to reduce memory |
| `BufferAssignment` | Assign buffers to instructions |

### Pass Pipeline

On the GPU backend, the optimization pipeline roughly follows:

```
StableHLO -> HLO conversion
  |
  v
Target-independent HLO optimizations:
  - AlgebraicSimplifier
  - CSE
  - DCE
  - DotMerger
  - ConvolutionRewriter
  - Fusion
  - WhileLoopSimplifier
  |
  v
Layout Assignment
  |
  v
Target-specific HLO optimizations:
  - GPU-specific convolution rewrites (cuDNN, cuBLAS)
  - GEMM algorithm selection
  - Fusion (second pass)
  - Rematerialization
  - Scheduling
  |
  v
HLO -> LLVM IR lowering
  |
  v
LLVM optimization and native code generation
```
