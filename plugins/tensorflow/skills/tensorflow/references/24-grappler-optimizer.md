# TensorFlow Grappler Optimizer Reference

This document provides a comprehensive reference for TensorFlow's Grappler
graph optimization framework. Grappler optimizes TensorFlow computation graphs
to improve execution speed, reduce memory usage, and simplify graph structure.

## Table of Contents

1. [Grappler Architecture](#grappler-architecture)
2. [GraphOptimizer Interface](#graphoptimizer-interface)
3. [GrapplerItem](#grappleritem)
4. [MetaOptimizer](#metaoptimizer)
5. [ConstantFolding](#constantfolding)
6. [ArithmeticOptimizer](#arithmeticoptimizer)
7. [Remapper](#remapper)
8. [LayoutOptimizer](#layoutoptimizer)
9. [MemoryOptimizer](#memoryoptimizer)
10. [AutoMixedPrecision](#automixedprecision)
11. [ModelPruner](#modelpruner)
12. [LoopOptimizer](#loopoptimizer)
13. [ShapeOptimizer](#shapeoptimizer)
14. [FunctionOptimizer](#functionoptimizer)
15. [DebugStripper](#debugstripper)
16. [ScopedAllocatorOptimizer](#scopedallocatoroptimizer)
17. [PinToHostOptimizer](#pintohostoptimizer)
18. [ImplementationSelector](#implementationselector)
19. [DependencyOptimizer](#dependencyoptimizer)
20. [CommonSubgraphElimination](#commonsubgraphelimination)
21. [AutoParallel](#autoparallel)
22. [RewriterConfig Reference](#rewriterconfig-reference)

---

## Grappler Architecture

Grappler is TensorFlow's graph optimization system. It takes a computation graph
(GraphDef) and produces an optimized version. Optimizations are implemented as
individual `GraphOptimizer` instances that are orchestrated by the `MetaOptimizer`.

### Architecture Diagram

```
                +------------------+
                |   User GraphDef  |
                +--------+---------+
                         |
                +--------+---------+
                |   MetaOptimizer   |
                +--------+---------+
                         |
         +-------+-------+-------+-------+
         |       |       |       |       |
    ModelPruner ConstFold ArithOpt Remapper ...
         |       |       |       |       |
         +-------+-------+-------+-------+
                         |
                +--------+---------+
                | Optimized GraphDef |
                +------------------+
```

### Key Source Directories

| Directory | Purpose |
|-----------|---------|
| `tensorflow/core/grappler/` | Core framework |
| `tensorflow/core/grappler/optimizers/` | Individual optimizers |
| `tensorflow/core/grappler/clusters/` | Hardware cluster abstractions |
| `tensorflow/core/grappler/costs/` | Cost estimation |
| `tensorflow/core/grappler/utils/` | Graph analysis utilities |
| `tensorflow/core/grappler/verifiers/` | Graph verification |

---

## GraphOptimizer Interface

**File**: `tensorflow/core/grappler/optimizers/graph_optimizer.h`

The base class for all Grappler optimizers.

```cpp
class GraphOptimizer {
 public:
  GraphOptimizer() : deadline_usec_(0) {}
  virtual ~GraphOptimizer() {}

  // Returns the name of this optimizer
  virtual std::string name() const = 0;

  // Whether this optimizer requires a valid function library
  virtual bool UsesFunctionLibrary() const = 0;

  // Optimize the graph
  virtual absl::Status Optimize(Cluster* cluster,
                                const GrapplerItem& item,
                                GraphDef* optimized_graph) = 0;

  // Variant that can consume the item
  virtual absl::Status Optimize(Cluster* cluster,
                                GrapplerItem&& item,
                                GraphDef* optimized_graph);

  // Deadline management
  void set_deadline_usec(uint64_t deadline_usec);
  uint64_t deadline_usec() const;
  bool DeadlineExceeded() const;
};
```

### Key Methods

- **name()**: Returns a unique string identifier for the optimizer
- **UsesFunctionLibrary()**: If false, the function library is replaced with
  stubs (valid signatures, empty bodies) to save memory
- **Optimize()**: The main optimization method. Takes the original graph and
  produces an optimized version
- **DeadlineExceeded()**: Check if the optimizer has exceeded its time budget

### Deadline Macro

```cpp
#define GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED()                \
  do {                                                        \
    if (this->DeadlineExceeded()) {                           \
      return absl::DeadlineExceededError(                     \
          absl::StrCat(this->name(), " exceeded deadline.")); \
    }                                                         \
  } while (0)
```

### CustomGraphOptimizer

```cpp
class CustomGraphOptimizer : public GraphOptimizer {
 public:
  virtual absl::Status Init(
      const RewriterConfig::CustomGraphOptimizer* config) = 0;
};
```

Custom optimizers can be registered via `CustomGraphOptimizerRegistry`.

---

## GrapplerItem

**File**: `tensorflow/core/grappler/grappler_item.h`

Represents a TensorFlow model to optimize. Contains the graph, feeds, fetches,
and configuration.

```cpp
struct GrapplerItem {
  std::string id;                     // Unique identifier
  GraphDef graph;                     // The computation graph
  std::vector<std::pair<std::string, Tensor>> feed;  // Input feeds
  std::vector<std::string> fetch;     // Output fetches
  std::vector<std::string> init_ops;  // Initialization operations
  int64_t expected_init_time = 0;     // Expected init time in seconds
  std::string save_op;                // Save operation name
  std::string restore_op;             // Restore operation name
  std::string save_restore_loc_tensor;// Save/restore location tensor
  std::vector<QueueRunnerDef> queue_runners;  // Queue runners
  std::vector<std::string> keep_ops;  // Ops to preserve

  // Analysis methods
  std::vector<const NodeDef*> MainOpsFanin() const;
  std::vector<const NodeDef*> EnqueueOpsFanin() const;
  std::vector<const NodeDef*> InitOpsFanin() const;
  std::vector<const NodeDef*> MainVariables() const;
  std::unordered_set<std::string> NodesToPreserve() const;
};
```

### OptimizationOptions

```cpp
struct OptimizationOptions {
  bool allow_non_differentiable_rewrites = true;
  bool allow_pruning_stateful_and_dataset_ops = true;
  bool optimize_function_library = true;
  bool is_eager_mode = false;
  int intra_op_parallelism_threads = port::MaxParallelism();
};
```

| Option | Description |
|--------|-------------|
| `allow_non_differentiable_rewrites` | Allow adding nodes without gradient functions |
| `allow_pruning_stateful_and_dataset_ops` | Allow pruning stateful and dataset ops in function graphs |
| `optimize_function_library` | Optimize functions in the function library |
| `is_eager_mode` | Whether running in eager mode |
| `intra_op_parallelism_threads` | Number of intra-op threads |

---

## MetaOptimizer

**File**: `tensorflow/core/grappler/optimizers/meta_optimizer.h`

The main entry point for Grappler optimization. Orchestrates all other
optimizers based on the `RewriterConfig` configuration.

```cpp
class MetaOptimizer : public GraphOptimizer {
 public:
  MetaOptimizer(DeviceBase* cpu_device, const ConfigProto& cfg);

  std::string name() const override { return "meta_optimizer"; }
  bool UsesFunctionLibrary() const override { return true; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override;
  absl::Status OptimizeConsumeItem(Cluster* cluster, GrapplerItem&& item,
                                   GraphDef* optimized_graph);
};
```

### MetaOptimizer Behavior

1. **Initialize Optimizers**: Based on `RewriterConfig`, creates the list of
   active optimizers
2. **Run Optimization Passes**: Iterates through optimizers, running each one
3. **Verify Graph**: Optionally verifies the graph between optimizer passes
4. **Multiple Passes**: May run multiple passes (main graph + function library)

### Optimizer Execution Order

The default execution order when all optimizers are enabled:

1. `model_pruner` -- Remove dead nodes
2. `debug_stripper` -- Remove debug ops
3. `constant_folding` -- Fold constants
4. `shape_optimizer` -- Optimize shape operations
5. `function_optimizer` -- Inline and optimize functions
6. `arithmetic_optimizer` -- Simplify arithmetic
7. `layout` -- Optimize data layout
8. `remapper` -- Fuse operation sequences
9. `loop_optimizer` -- Optimize loops
10. `dependency_optimizer` -- Remove unnecessary dependencies
11. `auto_mixed_precision` -- Convert to FP16/BF16
12. `memory_optimizer` -- Memory planning
13. `pin_to_host_optimizer` -- Move small ops to CPU
14. `scoped_allocator_optimizer` -- Optimize allocation patterns
15. `implementation_selector` -- Select best implementation

### Checking if MetaOptimizer is Enabled

```cpp
bool MetaOptimizerEnabled(const ConfigProto& cfg);
```

---

## ConstantFolding

**File**: `tensorflow/core/grappler/optimizers/constant_folding.h`

Folds constant expressions in the graph, evaluating them at graph construction
time rather than at runtime.

```cpp
class ConstantFolding : public GraphOptimizer {
 public:
  explicit ConstantFolding(DeviceBase* cpu_device,
                           bool disable_compressed_tensor_optimization = false,
                           bool fold_quantization_emulation = true);
  ConstantFolding(RewriterConfig::Toggle opt_level, DeviceBase* cpu_device, ...);

  std::string name() const override { return "constant_folding"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Key Optimization Techniques

1. **Constant Propagation**: Evaluates operations whose inputs are all constants
2. **Shape Materialization**: Materializes shape values from `Shape`, `Size`,
   `Rank` ops when shapes are statically known
3. **Constant Push-Down**: Pushes constants through arithmetic operations
4. **Strength Reduction**: Replaces expensive operations with cheaper ones
5. **Identity Elimination**: Removes unnecessary Identity and Reshape ops

### Specific Optimizations

#### Shape Materialization
- `Shape(tensor)` -> Const if shape is fully known
- `Size(tensor)` -> Const if shape is fully known
- `Rank(tensor)` -> Const if rank is known
- `BroadcastGradientArgs` -> Const if shapes are known

#### Constant Push-Down
```
Before:                   After:
  Const a                  Const a
  Const b                  Const (a OP c)
    \                        |
     OP ----> Const c  =>    OP  (with b propagated)
    /                        |
  node                      node
```

#### Arithmetic Simplifications

- `Mul(x, 1)` -> `Identity(x)`
- `Mul(x, 0)` -> `BroadcastTo(0, shape(x))`
- `Add(x, 0)` -> `Identity(x)`
- `Div(x, 1)` -> `Identity(x)`
- `Sub(x, x)` -> `BroadcastTo(0, shape(x))`
- `Pow(x, 1)` -> `Identity(x)`
- `Pow(x, 2)` -> `Square(x)`

#### Reshape Simplification
- `Reshape(x, shape)` -> `Identity(x)` if shape is identical
- `Reshape(x, [-1])` -> `Flatten(x)` equivalent

#### Reduction Simplification
- Reduce with empty indices -> `Identity` or `Reshape`
- Reduce single-element dimensions -> `Identity`

#### Partial Concat Folding
For non-commutative Concat: fold only the constant prefix/suffix portions.

#### Variable Update Elimination
Replaces no-op variable updates (e.g., `AssignAdd(x, 0)`) with `NoOp`.

### Size Limits

```cpp
extern const int64_t kMaxConstantSize;
```

Constant folding will not produce constants larger than this limit to avoid
excessive memory usage.

---

## ArithmeticOptimizer

**File**: `tensorflow/core/grappler/optimizers/arithmetic_optimizer.h`

Reduces arithmetic complexity through algebraic simplifications and
transformations.

```cpp
class ArithmeticOptimizer : public GraphOptimizer {
 public:
  ArithmeticOptimizer();
  explicit ArithmeticOptimizer(RewriterConfig::Toggle opt_level);

  std::string name() const override { return "arithmetic_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### ArithmeticOptimizerOptions

Granular control for individual optimization stages:

```cpp
struct ArithmeticOptimizerOptions {
  bool combine_add_to_addn = true;
  bool convert_sqrt_div_to_rsqrt_mul = true;
  bool dedup_computations = true;
  bool fold_conjugate_into_transpose = true;
  bool fold_multiply_into_conv = true;
  bool fold_transpose_into_matmul = true;
  bool fuse_squared_diff = true;
  bool hoist_common_factor_out_of_aggregation = true;
  bool hoist_cwise_unary_chains = true;
  bool minimize_broadcasts = true;
  bool optimize_max_or_min_of_monotonic = true;
  bool remove_idempotent = true;
  bool remove_identity_transpose = true;
  bool remove_involution = true;
  bool remove_logical_not = true;
  bool remove_negation = true;
  bool remove_redundant_bitcast = true;
  bool remove_redundant_cast = true;
  bool remove_redundant_reshape = true;
  bool reduce_upsampling_dims = true;
  bool reorder_cast_like_and_value_preserving = true;
  bool replace_mul_with_tile = true;
  bool replace_mul_with_square = true;
  bool replace_pack_with_tile_reshape = true;
  bool convert_pow = true;
  bool convert_log1p = true;
  bool convert_log_softmax = true;
  bool convert_expm1 = true;
  bool unary_ops_composition = true;
  bool remove_stack_slice_same_axis = true;
  bool simplify_aggregation = true;
  bool simplify_embedding_lookup = true;
  bool remove_cast_into_segment_reduction = true;
};
```

### Key Optimizations

#### HoistCommonUnaryOp (hoist_cwise_unary_chains)

Hoists common unary operations out of aggregation:

```
Before:                    After:
  a --> Log --+             a --+
  b --> Log --+-- Add  =>   b --+-- Add --> Log
```

#### RemoveIdentity (remove_idempotent)

Removes identity operations:
- `Transpose(x, [0, 1, 2, ...])` -> remove
- `Conjugate(Conjugate(x))` -> remove
- `Bitcast(Bitcast(x, T), original_type)` -> remove

#### RemoveNegation (remove_negation)

Simplifies negation patterns:
- `Neg(Neg(x))` -> `x`
- `Sub(0, x)` -> `Neg(x)`
- `Add(x, Neg(y))` -> `Sub(x, y)`

#### SimplifyAggregation (simplify_aggregation)

Combines aggregation operations:
- `Add(Add(x, y), z)` -> `AddN([x, y, z])`
- Combines multiple Add/AddN operations

#### ConvertPow (convert_pow)

Converts Pow to simpler operations:
- `Pow(x, 2)` -> `Square(x)` (via `replace_mul_with_square`)
- `Pow(x, -0.5)` -> `Rsqrt(x)` (via `convert_sqrt_div_to_rsqrt_mul`)
- `Pow(x, n)` where n is integer -> repeated multiplication

#### ConvertLog1p / ConvertExpm1

Converts compound operations to specialized ops:
- `Log(1 + x)` -> `Log1p(x)` (if available)
- `Exp(x) - 1` -> `Expm1(x)` (if available)

#### FoldMultiplyIntoConv (fold_multiply_into_conv)

Absorbs scalar multiplication into convolution:
```
Mul(Conv(input, filter), scalar) => Conv(input, Mul(filter, scalar))
```

#### MinimizeBroadcasts (minimize_broadcasts)

Reorders inputs to minimize broadcast overhead:
```
Before: Add(small_tensor, large_tensor)
After:  Add(large_tensor, small_tensor)
```

#### UnaryOpsComposition (unary_ops_composition)

Chains consecutive unary element-wise operations into a single composite op:
```
Sigmoid(Tanh(x)) => composed_unary_op(x)
```

#### FoldTransposeIntoMatMul (fold_transpose_into_matmul)

Absorbs Transpose into MatMul when possible:
```
MatMul(Transpose(a, [1,0]), b) => MatMul(a, b, transpose_a=true)
```

---

## Remapper

**File**: `tensorflow/core/grappler/optimizers/remapper.h`

Fuses sequences of operations into single, more efficient operations.

```cpp
class Remapper : public GraphOptimizer {
 public:
  explicit Remapper(RewriterConfig::Toggle opt_level,
                    RewriterConfig::CpuLayout cpu_layout_conversion = ...,
                    bool xla_auto_clustering_on = false);

  std::string name() const override { return "remapper"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Supported Fusion Patterns

#### GPU Fusion Patterns

| Pattern | Fused Op | Benefit |
|---------|----------|---------|
| `Conv2D + BiasAdd` | `_FusedConv2D` | Single kernel launch |
| `Conv2D + BiasAdd + Relu` | `_FusedConv2D` | Avoids intermediate tensor |
| `Conv2D + BiasAdd + Relu6` | `_FusedConv2D` | Avoids intermediate tensor |
| `Conv2D + BiasAdd + Elu` | `_FusedConv2D` | Avoids intermediate tensor |
| `Conv2D + BiasAdd + LeakyRelu` | `_FusedConv2D` | Avoids intermediate tensor |
| `Conv2D + BiasAdd + Sigmoid` | `_FusedConv2D` | Avoids intermediate tensor |
| `Conv2D + BiasAdd + Tanh` | `_FusedConv2D` | Avoids intermediate tensor |
| `MatMul + BiasAdd` | `_FusedMatMul` | Single kernel launch |
| `MatMul + BiasAdd + Relu` | `_FusedMatMul` | Avoids intermediate tensor |
| `MatMul + BiasAdd + Relu6` | `_FusedMatMul` | Avoids intermediate tensor |
| `MatMul + BiasAdd + Elu` | `_FusedMatMul` | Avoids intermediate tensor |
| `MatMul + BiasAdd + Sigmoid` | `_FusedMatMul` | Avoids intermediate tensor |
| `MatMul + BiasAdd + Tanh` | `_FusedMatMul` | Avoids intermediate tensor |

#### CPU Fusion Patterns (with oneDNN)

| Pattern | Fused Op |
|---------|----------|
| `Conv2D + BiasAdd + Relu` | `_FusedConv2D` |
| `Conv2D + BiasAdd + Elu` | `_FusedConv2D` |
| `Conv2D + BiasAdd + Gelu` | `_FusedConv2D` |
| `MatMul + BiasAdd + Relu` | `_FusedMatMul` |
| `MatMul + BiasAdd + Elu` | `_FusedMatMul` |
| `MatMul + BiasAdd + Gelu` | `_FusedMatMul` |
| `MatMul + BiasAdd + Sigmoid` | `_FusedMatMul` |

#### Other Fusion Patterns

- `Mul + Sum` -> `WeightedSum` (for attention mechanisms)
- `Softmax + CrossEntropy` -> `SoftmaxCrossEntropyWithLogits`
- `BiasAdd + Add` -> `BiasAdd` (merge two bias additions)
- `Contraction + Add` -> Fused contraction with bias

---

## LayoutOptimizer

**File**: `tensorflow/core/grappler/optimizers/generic_layout_optimizer.h`

Optimizes data layout for convolutional models, primarily converting between
NHWC and NCHW formats on GPU.

```cpp
class GenericLayoutOptimizer : public GraphOptimizer {
 public:
  explicit GenericLayoutOptimizer(std::string enforced_layout = "");
  explicit GenericLayoutOptimizer(RewriterConfig::Toggle opt_level,
                                  std::string enforced_layout = "");
  explicit GenericLayoutOptimizer(RewriterConfig::Toggle opt_level,
                                  RewriterConfig::CpuLayout layout_conversion,
                                  std::string enforced_layout = "");

  std::string name() const override { return "layout"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Layout Conversion

- **GPU**: Converts NHWC -> NCHW for better memory coalescing and cuDNN
  compatibility
- **CPU**: Optionally converts NHWC -> NCHW for oneDNN compatibility

### Transposer System

Each operation type has a transposer that knows how to:
1. Transpose its inputs from source to target layout
2. Update its attributes (e.g., data_format, strides)
3. Transpose its outputs back if needed

Supported operations:
- Conv2D, Conv3D, DepthwiseConv2dNative
- AvgPool, MaxPool
- BiasAdd
- FusedBatchNorm
- ResizeNearestNeighbor, ResizeBilinear
- Concat, Split
- Pad, MirrorPad

---

## MemoryOptimizer

**File**: `tensorflow/core/grappler/optimizers/memory_optimizer.h`

Optimizes memory usage through tensor swapping and recomputation.

```cpp
class MemoryOptimizer : public GraphOptimizer {
 public:
  explicit MemoryOptimizer(
      RewriterConfig::MemOptType optimization_level,
      const std::string& recomputation_targets_name_scope = "gradients/");

  std::string name() const override { return "memory_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Optimization Levels

| Level | Description |
|-------|-------------|
| `NO_MEM_OPT` | No memory optimization |
| `SWAP_HEURISTICS` | Use heuristics to swap tensors between GPU and CPU |
| `RECOMPUTATION_HEURISTICS` | Use heuristics to recompute tensors instead of storing |
| `MANUAL` | Use user-specified annotations for swapping/recomputation |

### Swapping

Moves tensors between GPU and CPU memory:
1. Identifies large tensors that are needed later
2. Inserts `SwapOut` (GPU->CPU) and `SwapIn` (CPU->GPU) nodes
3. Schedules swaps to overlap with computation

### Recomputation

Recomputes tensors instead of storing them:
1. Identifies nodes whose outputs are needed for backpropagation
2. Inserts recomputation nodes that recompute the tensor on-the-fly
3. Particularly effective for activation functions in training

**Target name scope**: By default targets nodes in `gradients/` namespace.

---

## AutoMixedPrecision

**File**: `tensorflow/core/grappler/optimizers/auto_mixed_precision.h`

Automatically converts data types to float16 or bfloat16 for improved
performance on supported hardware.

```cpp
enum class AutoMixedPrecisionMode { CUDA, BF16, CPU, FP16_CPU };

class AutoMixedPrecision : public GraphOptimizer {
 public:
  explicit AutoMixedPrecision(
      AutoMixedPrecisionMode mode = AutoMixedPrecisionMode::CUDA);

  std::string name() const override;  // varies by mode
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Modes

| Mode | Target | Conversion | Optimizer Name |
|------|--------|------------|----------------|
| `CUDA` | NVIDIA GPU | float32 -> float16 | `auto_mixed_precision` |
| `BF16` | CPU (oneDNN) | float32 -> bfloat16 | `auto_mixed_precision_onednn_bfloat16` |
| `CPU` | CPU | Emulated float16 | `auto_mixed_precision_cpu` |
| `FP16_CPU` | CPU (oneDNN) | float32 -> float16 | `auto_mixed_precision_onednn_float16` |

### Conversion Process

1. **Safety Analysis**: Categorize nodes as:
   - **Clear** (safe to convert): Conv, MatMul, arithmetic ops
   - **Deny** (must not convert): Loss, Softmax, operations requiring FP32 precision
   - **Infer** (propagate): Operations that follow their inputs' types
2. **Type Propagation**: Starting from clear nodes, propagate FP16 types through
   the graph
3. **Cast Insertion**: Insert `Cast` nodes at boundaries between FP16 and FP32
4. **Loss Scaling**: Optionally insert loss scaling for gradient stability

### Safe-to-Convert Operations

- `MatMul`, `BatchMatMul`, `BatchMatMulV2`
- `Conv2D`, `Conv3D`, `DepthwiseConv2dNative`
- `BiasAdd`
- Arithmetic: `Add`, `Sub`, `Mul`, `Div`
- Activation: `Relu`, `Relu6`, `Sigmoid`, `Tanh`, `Elu`

### Must-Not-Convert Operations

- `Softmax`, `LogSoftmax`
- Loss functions: `SoftmaxCrossEntropyWithLogits`
- `Sum` (over reduction dimensions)
- `Norm` operations
- Operations producing final outputs

---

## ModelPruner

**File**: `tensorflow/core/grappler/optimizers/model_pruner.h`

Removes dead and unreachable nodes from the graph.

```cpp
class ModelPruner : public GraphOptimizer {
 public:
  ModelPruner() {}
  std::string name() const override { return "model_pruner"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Pruning Strategy

1. **Forward Traversal**: Starting from fetch nodes, traverse the graph
   to find all reachable nodes
2. **Identify Dead Nodes**: Any node not reachable from fetches is dead
3. **Preserve Special Nodes**: Keep nodes that are in `keep_ops`, `init_ops`,
   or are referenced in collections
4. **Remove Dead Nodes**: Delete all dead nodes from the graph

### Additional Pruning

- **NoOp Removal**: Removes `NoOp` nodes with no control outputs
- **Identity Chain**: Simplifies chains of Identity nodes
- **Gradient Pruning**: Optimizes gradient computations by removing
   unnecessary gradient calculations

---

## LoopOptimizer

**File**: `tensorflow/core/grappler/optimizers/loop_optimizer.h`

Optimizes while loop constructs in the graph.

```cpp
class LoopOptimizer : public GraphOptimizer {
 public:
  LoopOptimizer();
  explicit LoopOptimizer(RewriterConfig::Toggle opt_level,
                         DeviceBase* cpu_device);

  std::string name() const override { return "loop_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### LoopOptimizerOptions

```cpp
struct LoopOptimizerOptions {
  bool enable_loop_invariant_node_motion = false;
  bool enable_stack_push_removal = true;
  bool enable_dead_branch_removal = true;
};
```

### Optimizations

#### Loop Invariant Node Motion

Moves loop-invariant computations out of while loops:

```
Before:                        After:
while_loop {                   x = compute_invariant(input)
  x = compute_invariant(input)  while_loop {
  use(x)                          use(x)  // x from outside
}                                }
```

#### Stack Push Removal

Removes unnecessary Stack push operations in loops when the stack
is never consumed.

#### Dead Branch Removal

Simplifies Switch/Merge patterns when the branch condition is statically known:

```
Before:                     After:
Switch(x, true) ->          Identity(x)
  true_branch: use(x)
  false_branch: dead
```

---

## ShapeOptimizer

**File**: `tensorflow/core/grappler/optimizers/shape_optimizer.h`

Optimizes subgraphs that operate on shape and shape-related information.

```cpp
class ShapeOptimizer : public GraphOptimizer {
 public:
  ShapeOptimizer() {}
  explicit ShapeOptimizer(RewriterConfig::Toggle opt_level) {}

  std::string name() const override { return "shape_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Optimizations

- **Shape Propagation**: Replaces `Shape`, `Size`, `Rank` ops with constants
  when shapes are statically known
- **Shape Composition**: Simplifies chains of shape operations:
  `Shape(Reshape(x, s))` -> `s`
- **StridedSlice on Shape**: Simplifies slicing of shape tensors
- **ConcatV2 of Shapes**: Simplifies concatenation of known shape values

---

## FunctionOptimizer

**File**: `tensorflow/core/grappler/optimizers/function_optimizer.h`

Optimizes TensorFlow function definitions, including inlining and specialization.

```cpp
class FunctionOptimizer : public GraphOptimizer {
 public:
  explicit FunctionOptimizer(RewriterConfig::Toggle opt_level,
                             bool lower_control_flow);

  std::string name() const override { return "function_optimizer"; }
  bool UsesFunctionLibrary() const override { return true; }
};
```

### Optimization Strategies

#### Function Inlining

Replaces function call nodes with the function body:
```
Before: y = my_func(x)
After:  x --> [function body ops] --> y
```

Inlining is performed when:
- The function is small (fewer than N nodes)
- The function is called only once
- Inlining would enable further optimizations

#### Function Specialization

Creates specialized versions of functions based on constant attributes:
```
Before: my_func[T=DT_FLOAT, shape=unknown](x)
After:  my_func_specialized_T_float[x_shape_known](x)
```

#### Control Flow Lowering

When `lower_control_flow` is true, converts functional control flow
(Switch/Merge) to V1 style nodes.

---

## DebugStripper

**File**: `tensorflow/core/grappler/optimizers/debug_stripper.h`

Removes debug-related nodes from the graph to improve production performance.

```cpp
class DebugStripper : public GraphOptimizer {
 public:
  DebugStripper() = default;
  std::string name() const override { return "debug_stripper"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Removed Operations

- `Assert` -> Replaced with `NoOp`
- `CheckNumerics` -> Replaced with `Identity`
- `Print` -> Replaced with `Identity`
- `GuardedPhaseRandomUniform` -> Replaced with `RandomUniform`

---

## ScopedAllocatorOptimizer

**File**: `tensorflow/core/grappler/optimizers/scoped_allocator_optimizer.h`

Optimizes memory allocation patterns by grouping allocations.

```cpp
class ScopedAllocatorOptimizer : public GraphOptimizer {
 public:
  ScopedAllocatorOptimizer(RewriterConfig::Toggle opt_level,
                           const ScopedAllocatorOptions& opts);

  std::string name() const override { return "scoped_allocator_optimizer"; }
  bool UsesFunctionLibrary() const override { return true; }
};
```

### How It Works

1. **Identify Groups**: Finds groups of operations that can share allocations
2. **Create ScopedAllocator**: Inserts a `ScopedAllocator` node that allocates
   a single large buffer
3. **Rewrite Consumers**: Rewrites consuming operations to use slices of the
   shared buffer
4. **Benefits**: Reduces allocation overhead and memory fragmentation

### Rewriter Interface

```cpp
class Rewriter {
 public:
  virtual ~Rewriter() {}
  virtual absl::Status Rewrite(ScopedAllocatorOptimizer* paopti,
                               int64_t invocation_count,
                               GraphDef* graph,
                               const std::string& op_name,
                               const std::vector<NodeDef*>& nodes,
                               bool* applied) = 0;
};
```

Custom rewriters can be registered for specific operation patterns.

---

## PinToHostOptimizer

**File**: `tensorflow/core/grappler/optimizers/pin_to_host_optimizer.h`

Moves small operations from GPU to CPU to avoid CPU-GPU transfer overhead.

```cpp
class PinToHostOptimizer : public GraphOptimizer {
 public:
  PinToHostOptimizer() {}
  explicit PinToHostOptimizer(RewriterConfig::Toggle opt_level) {}

  std::string name() const override { return "pin_to_host_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Heuristics

- Small integer constants are moved to CPU
- Shape-related operations are moved to CPU
- Operations producing tensors only consumed by CPU ops are moved to CPU

---

## ImplementationSelector

**File**: `tensorflow/core/grappler/optimizers/implementation_selector.h`

Selects the best function implementation based on runtime properties.

```cpp
class ImplementationSelector : public CustomGraphOptimizer {
 public:
  ImplementationSelector() = default;
  std::string name() const override { return "implementation_selector"; }
  bool UsesFunctionLibrary() const override { return false; }
};
```

### Selection Approaches

#### Approach 1: DeviceIndex + Case

Rewrites `DeviceIndex` ops with constants and uses `Case` to select
device-specific implementations:

```python
# Before
device_idx = DeviceIndex(device_names=["CPU", "GPU"])
result = Case(device_idx, [cpu_fn, gpu_fn])

# After (when running on GPU:2)
device_idx = Const(2)  # index of "GPU" in device_names
result = Case(device_idx, [cpu_fn, gpu_fn])
```

#### Approach 2: Function Attribute Swapping

Uses `api_implements` and `api_preferred_device` attributes to select
the best implementation:

```python
@Defun(tf.float32, api_implements='plus_one', api_preferred_device='GPU')
def plus_one_gpu(x): return x + 1.0

@Defun(tf.float32, api_implements='plus_one')
def plus_one_cpu(x): return x + 1.0
```

At runtime, the GPU implementation is preferred when a GPU is available.

---

## DependencyOptimizer

**File**: `tensorflow/core/grappler/optimizers/dependency_optimizer.h`

Removes unnecessary control dependencies from the graph.

### Optimizations

- **Identity Removal**: Removes Identity nodes used only for control flow
- **NoOp Simplification**: Simplifies chains of NoOp control dependencies
- **Redundant Dependencies**: Removes transitive control dependencies
- **Direct Dependencies**: Makes control dependencies direct rather than
  going through intermediate nodes

---

## CommonSubgraphElimination

**File**: `tensorflow/core/grappler/optimizers/common_subgraph_elimination.h`

Identifies and eliminates common subexpressions in the graph.

### Process

1. **Hash Computation**: Compute content-based hashes for all nodes
2. **Matching**: Find nodes with identical operation, attributes, and inputs
3. **Deduplication**: Replace duplicates with a single node
4. **Rewiring**: Update all consumers to use the deduplicated node

---

## AutoParallel

**File**: `tensorflow/core/grappler/optimizers/auto_parallel.h`

Automatically parallelizes graph operations by replicating them across devices.

### Process

1. **Identify Parallelizable Ops**: Find ops that can be split across devices
2. **Insert Split/Concat**: Add Split and Concat nodes around parallelized ops
3. **Assign Devices**: Assign replicated ops to different devices

---

## RewriterConfig Reference

The `RewriterConfig` proto controls Grappler optimization behavior:

### Toggle Values

```protobuf
enum Toggle {
  DEFAULT = 0;       // Use default behavior
  ON = 1;            // Enable
  OFF = 2;           // Disable
  FORCED = 3;        // Force enable (even if disabled by default)
  EXPERIMENTAL = 4;  // Enable experimental features
}
```

### MemOptType Values

```protobuf
enum MemOptType {
  NO_MEM_OPT = 0;
  SWAP_HEURISTICS = 1;           // Use swapping heuristics
  RECOMPUTATION_HEURISTICS = 2;  // Use recomputation heuristics
  MANUAL = 3;                    // Use manual annotations
}
```

### Key Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `layout_optimizer` | `Toggle` | `DEFAULT` | Enable layout optimization |
| `constant_folding` | `Toggle` | `ON` | Enable constant folding |
| `arithmetic_optimization` | `Toggle` | `DEFAULT` | Enable arithmetic optimization |
| `remapping` | `Toggle` | `DEFAULT` | Enable operation remapping/fusion |
| `memory_optimization` | `MemOptType` | `NO_MEM_OPT` | Memory optimization level |
| `auto_mixed_precision` | `Toggle` | `OFF` | Enable auto mixed precision |
| `debug_stripper` | `Toggle` | `OFF` | Strip debug operations |
| `implementation_selector` | `Toggle` | `OFF` | Enable implementation selection |
| `function_optimization` | `Toggle` | `DEFAULT` | Enable function optimization |
| `scoped_allocator_optimization` | `Toggle` | `OFF` | Enable scoped allocation |
| `pin_to_host_optimization` | `Toggle` | `DEFAULT` | Enable pin-to-host optimization |
| `memory_optimizer_target_node_name_scope` | `string` | `""` | Target nodes for memory optimization |
| `min_graph_nodes` | `int32` | `0` | Minimum nodes to run optimizers |
| `optimizers` | `repeated string` | `[]` | Explicit list of optimizers to run |
| `custom_optimizers` | `repeated CustomGraphOptimizer` | `[]` | Custom optimizer configurations |
| `meta_optimizer_iterations` | `int32` | `1` | Number of meta optimizer passes |
| `meta_optimizer_timeout_ms` | `int64` | `0` | Timeout per optimizer in ms |
| `fail_on_optimizer_errors` | `bool` | `false` | Fail on optimizer errors |

### Python Configuration

```python
config = tf.ConfigProto()
config.graph_options.rewrite_options.layout_optimizer = tf.RewriterConfig.ON
config.graph_options.rewrite_options.constant_folding = tf.RewriterConfig.ON
config.graph_options.rewrite_options.arithmetic_optimization = tf.RewriterConfig.ON
config.graph_options.rewrite_options.remapping = tf.RewriterConfig.ON

# Disable all optimizers
config.graph_options.optimization_options.global_meta_optimizer = False

# Custom optimizer order
config.graph_options.rewrite_options.optimizers.extend([
    "model_pruner",
    "constant_folding",
    "arithmetic_optimizer"
])
```

---

## Graph Verification

Between optimization passes and after all optimizations, Grappler can verify
the graph structure:

### Verifier Types

1. **StructureVerifier**: Checks graph structure integrity (valid node names,
   no cycles, valid input references)
2. **GraphPropertiesVerifier**: Verifies that graph properties (shapes, types)
   are consistent

### Configuration

```protobuf
message VerifierConfig {
  enum Toggle {
    DEFAULT = 0;
    ON = 1;
    OFF = 2;
  }
  Toggle structure_verifier = 1;  // Verify graph structure
}
```

---

## Cost Estimation

### GraphProperties

Infers static properties of tensors in the graph:

```cpp
class GraphProperties {
 public:
  static absl::StatusOr<GraphProperties> Infer(
      const GrapplerItem& item, bool assume_valid_feeds = false);
  // Returns properties for a specific node output
  absl::StatusOr<std::vector<OpInfo::TensorProperties>>
  GetOutputProperties(const string& node_name) const;
};
```

Properties include:
- Data types
- Static shapes (when inferrable)
- Value ranges (for constant folding decisions)

---

## Cluster Interface

The `Cluster` interface provides hardware information for optimization decisions:

```cpp
class Cluster {
 public:
  virtual const std::vector<string>& GetDeviceTypes() const = 0;
  virtual int NumDevices() const = 0;
  // ...
};
```

Implementations:
- **VirtualCluster**: Simulated cluster for testing
- **GpuCluster**: Real GPU hardware information
- **SingleMachine**: Single machine with local devices
