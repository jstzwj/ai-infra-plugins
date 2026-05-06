# XLA Reference - Chapter 14: HLO Optimization and Transformation Passes

This reference provides comprehensive documentation on HLO (High-Level Optimizer) passes in XLA, covering the base class architecture, hardware-independent optimization passes, hardware-specific passes, analysis utilities, and tooling for pass development.

---

## 14.1 Overview

HLO passes are the fundamental transformation and optimization units in XLA's compilation pipeline. Each pass examines, transforms, or analyzes an HLO module (or computation) according to a well-defined contract. The pass infrastructure enables:

- **Composable transformations**: Passes can be chained into pipelines.
- **Iterative optimization**: Passes can be run repeatedly until a fixpoint is reached.
- **Hardware targeting**: Passes can be generic or specific to a backend (GPU, CPU, TPU).
- **Verification and debugging**: Analysis passes verify invariants without mutation.

The HLO pass framework is defined primarily in `xla/service/` and follows a design pattern similar to LLVM's pass manager. Each pass implements a specific interface and is invoked by the compiler's pass pipeline driver.

---

## 14.2 Base Classes

### 14.2.1 HloPassInterface

`HloPassInterface` is the abstract base class for all HLO passes. It defines the fundamental contract that every pass must satisfy:

```cpp
class HloPassInterface {
 public:
  virtual ~HloPassInterface() = default;

  // The name of this pass, used for logging and debugging.
  virtual absl::string_view name() const = 0;

  // Run the pass on the given HLO module. Returns whether the module was
  // modified. The `execution_threads` parameter controls which threads
  // (in a multi-threaded execution context) the pass should consider.
  virtual StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const DebugOptions& debug_options) = 0;

  // Run the pass on a single computation. Not all passes support this;
  // the default implementation returns an error.
  virtual StatusOr<bool> RunOnComputation(
      HloComputation* computation,
      const DebugOptions& debug_options);
};
```

Key design decisions:

- **Return value**: `Run()` returns `StatusOr<bool>` where the boolean indicates whether the module was actually modified. This enables pipeline-level bookkeeping and determines whether subsequent fixpoint iteration is necessary.
- **Execution threads**: The `execution_threads` parameter enables fine-grained control over which computations are affected. In XLA's execution model, different HLO computations may be associated with different host-side threads. This parameter allows passes to target only computations belonging to specific threads.
- **Debug options**: Passes receive debug options that control logging verbosity, verification behavior, and other debugging aids.

### 14.2.2 HloModulePass

`HloModulePass` extends `HloPassInterface` and is the most commonly used base class. It operates at the granularity of an entire `HloModule`:

```cpp
class HloModulePass : public HloPassInterface {
 public:
  // Run the pass across all computations in the module.
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const DebugOptions& debug_options) override;

 protected:
  // Subclasses implement this method to perform the actual transformation.
  virtual StatusOr<bool> RunOnModule(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) = 0;
};
```

The `HloModulePass` base class handles the plumbing of iterating over computations and dispatching to the subclass's `RunOnModule` implementation. Most optimization passes inherit from `HloModulePass` because they need to reason about cross-computation properties (e.g., call graph structure, aliasing between parameters and outputs of different computations).

Typical implementation pattern:

```cpp
class AlgebraicSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override { return "algebraic-simplifier"; }

 protected:
  StatusOr<bool> RunOnModule(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override {
    bool changed = false;
    for (HloComputation* computation : module->computations()) {
      if (!execution_threads.empty() &&
          !execution_threads.contains(computation->execution_thread())) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(computation));
      changed |= comp_changed;
    }
    return changed;
  }
};
```

### 14.2.3 HloPassPipeline

`HloPassPipeline` is a composite pass that runs a sequence of passes in order. It is the mechanism by which XLA organizes its compilation phases:

```cpp
class HloPassPipeline : public HloPassInterface {
 public:
  explicit HloPassPipeline(absl::string_view name) : name_(name) {}

  // Add a pass to the pipeline. The pipeline takes ownership.
  template <typename T, typename... Args>
  T* AddPass(Args&&... args) {
    auto pass = std::make_unique<T>(std::forward<Args>(args)...);
    T* ptr = pass.get();
    passes_.push_back(std::move(pass));
    return ptr;
  }

  // Run all passes in order. Returns true if any pass modified the module.
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const DebugOptions& debug_options) override;

 private:
  std::string name_;
  std::vector<std::unique_ptr<HloPassInterface>> passes_;
};
```

Pipeline features:

- **Fixpoint iteration**: Pipelines can be configured to run repeatedly until no pass reports a modification, achieving a global fixpoint. This is controlled via the `set_run_before_and_after_verifier()` and related configuration methods.
- **Verification**: Between passes, the pipeline can run `HloVerifier` to catch invariant violations early. This is invaluable during development.
- **Logging**: The pipeline logs pass execution order, execution time, and whether each pass made changes. This is controlled via `--xla_hlo_passes_dump_to` and related flags.
- **Module printing**: After each pass, the pipeline can dump the HLO module to a file for offline analysis. This is the primary debugging mechanism for pass development.

XLA defines several major pipelines, each containing dozens of passes:

1. **Optimization pipeline** (`BuildOptimizationPipeline`): The main optimization pipeline containing all hardware-independent and backend-specific optimizations.
2. **Layout assignment pipeline**: Assigns physical layouts to all instructions.
3. **Scheduling pipeline**: Orders instructions and performs buffer assignment.

Example of pipeline construction:

```cpp
std::unique_ptr<HloPassPipeline> BuildOptimizationPipeline(
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const DebugOptions& debug_options) {
  auto pipeline = std::make_unique<HloPassPipeline>("optimization");
  pipeline->AddPass<CallInliner>();
  pipeline->AddPass<FlattenCallGraph>();
  pipeline->AddPass<AlgebraicSimplifier>(...);
  pipeline->AddPass<HloConstantFolding>();
  pipeline->AddPass<HloDCE>();
  pipeline->AddPass<ReshapeMover>();
  pipeline->AddPass<TransposeFolding>();
  pipeline->AddPass<HloRematerialization>(...);
  pipeline->AddPass<Fusion>(...);
  // ... many more passes
  return pipeline;
}
```

---

## 14.3 Hardware-Independent Passes

These passes implement optimizations that are valid regardless of the target hardware. They operate purely on the algebraic and structural properties of HLO operations.

### 14.3.1 AlgebraicSimplifier

The `AlgebraicSimplifier` is one of the most important and far-reaching optimization passes. It applies algebraic identities and arithmetic simplifications to reduce the computational and memory footprint of the HLO graph.

**Core simplification rules:**

| Pattern | Simplification | Notes |
|---------|---------------|-------|
| `add(x, 0)` | `x` | Additive identity |
| `mul(x, 1)` | `x` | Multiplicative identity |
| `mul(x, 0)` | `broadcast(0)` | Zero propagation |
| `sub(x, x)` | `broadcast(0)` | Self-subtraction |
| `div(x, 1)` | `x` | Division by unity |
| `div(x, x)` | `broadcast(1)` | Self-division (non-zero) |
| `pow(x, 0)` | `broadcast(1)` | Zero exponent |
| `pow(x, 1)` | `x` | Unity exponent |
| `exp(log(x))` | `x` | Inverse function cancellation |
| `log(exp(x))` | `x` | Inverse function cancellation |
| `max(x, x)` | `x` | Idempotent operation |
| `min(x, x)` | `x` | Idempotent operation |
| `clamp(x, x, y)` | `max(x, y)` | Redundant clamp bound |
| `concatenate([x], 0)` | `x` | Trivial concatenation |
| `reshape(x, same_shape)` | `x` | Identity reshape |
| `transpose(x, {0,1,...,n})` | `x` | Identity transpose |
| `slice(x, full_dims)` | `x` | Full-dimension slice |
| `broadcast(x, no_new_dims)` | `x` | Trivial broadcast |

**Advanced simplifications:**

**Division to multiplication conversion**: When the divisor is a constant, the pass converts `div(x, const)` to `mul(x, reciprocal_of_const)`. This is profitable because multiplication is significantly faster than division on all hardware:

```
// Before:
%div = div(%x, constant(3.0))

// After:
%reciprocal = constant(0.333333...)
%mul = mul(%x, %reciprocal)
```

**Broadcast collapsing**: When a broadcast feeds into an elementwise operation, the broadcast dimensions can sometimes be simplified or eliminated:

```
// Before:
%bcast = broadcast(%scalar, dimensions={})
%add = add(%tensor, %bcast)

// After: (broadcast folded into the add)
%add = add(%tensor, %scalar)  // scalar broadcast is implicit
```

**Concatenate simplification**: Adjacent concatenations on the same dimension can be merged; single-element concatenations are eliminated:

```
// Before:
%c1 = concatenate(%a, %b), dimension=0
%c2 = concatenate(%c1, %c), dimension=0

// After:
%c = concatenate(%a, %b, %c), dimension=0
```

**Reshape-transpose collapse**: A reshape followed by a transpose (or vice versa) can sometimes be collapsed into a single reshape:

```
// Before:
%r = reshape(%x, [N, H, W, C])
%t = transpose(%r, {0, 3, 1, 2})

// After:
%r = reshape(%x, [N, C, H, W])
```

**Comparison simplification**: Comparisons involving constants can be folded or eliminated:

```
// Before:
%cmp = compare(%x, constant(MAX_FLOAT), direction=GT)

// After:
%result = broadcast(false)  // No float exceeds MAX_FLOAT
```

**Configuration**: The `AlgebraicSimplifier` accepts an `AlgebraicSimplifierOptions` struct that controls which simplifications are enabled. This allows backends to disable simplifications that would be unprofitable on their hardware. For example, the GPU backend may disable certain layout-changing simplifications that would conflict with optimal layout assignment.

```cpp
struct AlgebraicSimplifierOptions {
  bool enable_div_to_mul_conversion = true;
  bool enable_dot_reshape = true;
  bool is_layout_sensitive = false;
  ShapeSizeFunction shape_size_function;
  // ...
};
```

### 14.3.2 HloRematerialization

`HloRematerialization` (also called "remat") is a critical memory optimization pass that trades computation for memory by recomputing values instead of storing them. This is especially important on accelerators with limited high-bandwidth memory (HBM).

**Problem statement**: After optimization, the HLO graph may contain operations whose results need to be kept alive for a long time because they are used by many downstream consumers. This increases peak memory usage. Rematerialization identifies cases where it is cheaper to recompute a value than to keep it in memory.

**Algorithm overview:**

1. **Live range analysis**: For each instruction, compute the live range -- the span from its definition to its last use. The live range determines how long the instruction's output buffer must remain allocated.

2. **Memory pressure identification**: Identify points in the program where the total size of live buffers exceeds the available memory budget.

3. **Rematerialization candidate selection**: Find instructions whose recomputation would reduce peak memory. Good candidates are:
   - Cheap to compute (elementwise ops, small constants).
   - Have long live ranges.
   - Have large output buffers.
   - Have few operands (recomputation overhead is proportional to the number of operands that must be live).

4. **Cost-benefit analysis**: For each candidate, compute:
   - **Benefit**: The reduction in peak memory usage.
   - **Cost**: The additional compute required to re-evaluate the instruction.
   - The pass rematerializes only when the benefit exceeds the cost.

**Example:**

```
// Before rematerialization:
%0 = exp(%input)           // Large tensor, expensive to keep in memory
%1 = ... long sequence of operations ...
%10 = add(%1, %0)          // %0 must be live from definition to here

// After rematerialization:
%0 = exp(%input)
%1 = ... long sequence of operations ...  // %0 can be freed
%0_remat = exp(%input)     // Recompute %0 just before it's needed
%10 = add(%1, %0_remat)
```

**Implementation details:**

- The pass uses a greedy algorithm that repeatedly finds the best rematerialization opportunity and applies it until the memory budget is satisfied or no more profitable rematerializations exist.
- The `RematerializationCostAnalysis` computes the cost of rematerializing each instruction, considering:
  - The number of FLOPs required.
  - The memory bandwidth consumed.
  - Whether the instruction is already in the working set of nearby operations.
- The pass can also perform **composable rematerialization**: if instruction A feeds into instruction B, and both are rematerialized, the recomputation of B automatically includes the recomputation of A.
- The pass maintains a **memory scheduler** that tracks the set of live buffers at each point in the program execution order.

**Configuration flags:**

```
--xla_memory_budget_bytes=N      // Set memory budget
--xla_rematerialize_all_dot_operands  // Force rematerialization of dot operands
--xla_enable_remateralization_heuristic=auto  // Enable/disable heuristics
```

### 14.3.3 HloConstantFolding

`HloConstantFolding` evaluates operations on constants at compile time, replacing them with precomputed constant values. This eliminates runtime computation entirely for expressions whose inputs are known at compile time.

**Scope of folding:**

- **Elementwise operations**: Any elementwise operation (add, mul, exp, log, sin, cos, etc.) applied to constant inputs is folded.
- **Broadcast of constants**: `broadcast(constant)` is folded by materializing the full broadcast result.
- **Reshape of constants**: `reshape(constant)` is folded by reinterpreting the constant's shape.
- **Slice of constants**: `slice(constant, ...)` is folded by extracting the relevant sub-array.
- **Concatenate of constants**: `concatenate(constant1, constant2, ...)` is folded by combining the arrays.
- **Dot/Convolution of constants**: When both operands are constants, the entire dot product or convolution is computed at compile time.
- **Compare of constants**: Comparison operations with constant inputs are folded to boolean constants.
- **Select with constant condition**: `select(true, x, y)` simplifies to `x`.

**Implementation constraints:**

- **Size limit**: To prevent excessive compile-time memory usage and compilation time, the pass has configurable limits on the size of constants it will materialize:
  ```cpp
  int64_t max_constant_size_in_bytes = 64 * 1024 * 1024;  // 64 MB default
  ```
  Constants larger than this limit are not folded to avoid blowing up the HLO module size.

- **Numerical consistency**: Constant folding uses the same floating-point semantics as the target device. For example, if the target uses BF16 arithmetic, the constant folding evaluation should use BF16 precision. This is achieved by using the same `EvalHelper` routines that the interpreter backend uses.

- **Layout handling**: When the pass is layout-sensitive (i.e., operating after layout assignment), it must respect the assigned layouts when materializing folded constants.

**Example transformation:**

```
// Before:
%c1 = constant([1.0, 2.0, 3.0])
%c2 = constant([4.0, 5.0, 6.0])
%sum = add(%c1, %c2)

// After:
%sum = constant([5.0, 7.0, 9.0])
```

### 14.3.4 HloDCE (Dead Code Elimination)

`HloDCE` removes instructions whose results are never used. This is essential for keeping the HLO graph clean, especially after other passes have simplified or replaced operations.

**Definition of "dead"**: An instruction is dead if:

1. It has no users (i.e., no other instruction reads its output).
2. It is not the root instruction of an entry computation (the root's output is the module's result).
3. It is not a side-effecting instruction (e.g., `Recv`, `Send`, `Outfeed`, `RngGetAndUpdateState`). Side-effecting instructions are never removed even if their results appear unused, because they may affect global state.

**Algorithm:**

1. Mark all live instructions starting from computation roots.
2. Traverse use-def chains transitively.
3. Remove all unmarked instructions.

**Special considerations:**

- **Control dependencies**: XLA supports explicit control dependencies between instructions (via `AddControlDependencyTo`). DCE must preserve these dependencies when removing dead instructions. If instruction A has a control dependency on instruction B, and B is dead but A is live, the control dependency must be transferred to one of B's operands or removed.
- **Conditional and call operations**: Instructions inside conditional branches or called computations must be analyzed recursively. An instruction in a branch computation may appear dead within that computation but is actually used by the conditional's result.
- **While loops**: The condition and body computations of while loops require special handling. Instructions in the body that are not used by the loop's output are dead, but they may still be needed if the loop condition depends on them.

**Example:**

```
// Before DCE:
%x = parameter(0)
%y = parameter(1)
%unused = exp(%x)        // No users
%result = add(%x, %y)

// After DCE:
%x = parameter(0)
%y = parameter(1)
%result = add(%x, %y)
```

### 14.3.5 FlattenCallGraph

`FlattenCallGraph` transforms the HLO call graph from a directed acyclic graph (DAG) into a tree by duplicating computations that are called from multiple sites. This simplification is necessary for several downstream passes that assume a tree-structured call graph.

**Motivation**: In HLO, a computation can be called from multiple sites (e.g., a fusion computation used in two different fusions, or a while body called from two different while loops). This creates a DAG structure where a single computation has multiple call sites. Some passes (particularly layout assignment and buffer assignment) need to specialize the computation for each call site, which is only possible with a tree structure.

**Algorithm:**

1. For each computation that is called from more than one site:
   a. Duplicate the computation.
   b. Redirect each call site to use its own copy.
2. Repeat until all computations have at most one call site.

**Example:**

```
// Before:
%fused_computation {
  %p = parameter(0)
  %result = exp(%p)
}
%call1 = fusion(%a), calls=%fused_computation
%call2 = fusion(%b), calls=%fused_computation

// After:
%fused_computation.1 {
  %p = parameter(0)
  %result = exp(%p)
}
%fused_computation.2 {
  %p = parameter(0)
  %result = exp(%p)
}
%call1 = fusion(%a), calls=%fused_computation.1
%call2 = fusion(%b), calls=%fused_computation.2
```

This duplication may increase total code size, but it enables per-call-site optimization. In practice, most computations already have a tree structure by the time `FlattenCallGraph` runs, so the duplication is minimal.

### 14.3.6 ReshapeMover

`ReshapeMover` improves optimization opportunities by moving reshape operations across elementwise operations. When a reshape appears before an elementwise operation, it can often be moved to after the elementwise operation without changing semantics. This enables:

- Better fusion opportunities (the elementwise operation can be fused with surrounding operations).
- Simpler intermediate shapes.
- More effective constant folding and algebraic simplification.

**Rule**: For an elementwise operation `f` and a reshape `r`:

```
reshape(f(x, y, ...)) == f(reshape(x), reshape(y), ...)
```

This is valid when the reshape merely reorders elements without changing the data and the elementwise operation is applied independently to each element.

**Example:**

```
// Before:
%a = parameter(0)    // shape: [2, 3]
%b = parameter(1)    // shape: [2, 3]
%add = add(%a, %b)   // shape: [2, 3]
%reshape = reshape(%add, [6])  // shape: [6]

// After:
%a_reshaped = reshape(%a, [6])  // shape: [6]
%b_reshaped = reshape(%b, [6])  // shape: [6]
%add = add(%a_reshaped, %b_reshaped)  // shape: [6]
```

**Limitations:**

- The rule only applies to elementwise operations. Reshapes cannot be moved across reductions, dot products, convolutions, or other non-elementwise operations.
- The reshape must be a "valid" reshape -- one that does not change the total number of elements.
- Moving the reshape must not increase the total number of reshape operations. If moving a reshape past a binary operation would require inserting two reshapes (one for each operand), the transformation is only applied if at least one of the operand reshapes already exists.

### 14.3.7 ZeroSizedHloElimination

`ZeroSizedHloElimination` removes operations that produce or consume zero-sized tensors. Zero-sized tensors arise naturally in programs with dynamic shapes (e.g., batch dimensions that may be zero) and can create unnecessary computation and memory overhead.

**What it does:**

1. Identifies instructions that produce tensors with zero in at least one dimension (e.g., `[0, 5]`, `[3, 0, 7]`).
2. Replaces such instructions with simpler equivalents (typically a `broadcast` of the zero-sized constant).
3. Propagates the zero-size information through the graph to eliminate downstream dead code.

**Example:**

```
// Before:
%dynamic = parameter(0)  // shape: [0, 5]
%result = exp(%dynamic)  // shape: [0, 5]

// After:
%result = broadcast(constant([]), [0, 5])  // No computation needed
```

This pass is important for dynamic shape workloads where zero-sized inputs are common (e.g., variable batch sizes, sparse operations with empty slices).

### 14.3.8 Fusion

Fusion is arguably the most important optimization pass in XLA. It combines multiple HLO operations into a single "fusion" computation that is emitted as a single kernel on the target hardware. Fusion dramatically reduces memory bandwidth consumption by keeping intermediate results in registers or shared memory rather than writing them to global memory.

**Why fusion matters:**

Consider a chain of elementwise operations:

```
%0 = exp(%input)    // Kernel 1: read input, write exp result
%1 = add(%0, %bias) // Kernel 2: read exp result, read bias, write add result
%2 = tanh(%1)       // Kernel 3: read add result, write tanh result
```

Without fusion, each operation requires a separate kernel launch with its own memory reads and writes. The total memory traffic is:

- 3 reads of large tensors (input, exp result, add result)
- 3 writes of large tensors (exp result, add result, tanh result)
- Total: 6 tensor-sized memory operations

With fusion, all three operations are combined into a single kernel:

```
%fused = fusion(%input, %bias) {
  %p0 = parameter(0)  // input
  %p1 = parameter(1)  // bias
  %e = exp(%p0)
  %a = add(%e, %p1)
  ROOT %t = tanh(%a)
}
```

Now the total memory traffic is:
- 2 reads (input, bias)
- 1 write (tanh result)
- Intermediate values (`exp` and `add` results) stay in registers.
- Total: 3 tensor-sized memory operations (a 2x reduction).

**Fusion strategies:**

#### Loop Fusion

Loop fusion is the simplest and most widely applicable strategy. It applies to chains of elementwise operations. The resulting kernel iterates over all elements in a single loop nest, computing the entire fused chain for each element:

```
// Fused kernel pseudocode:
for (int i = 0; i < N; ++i) {
  result[i] = tanh(exp(input[i]) + bias[i]);
}
```

Loop fusion applies when all fused operations are elementwise and the shapes are compatible (no complex broadcasting needed). This is the default fusion strategy and handles the vast majority of fusion opportunities.

#### Input Fusion

Input fusion (also called "producer-consumer fusion") merges a producer operation into a consumer fusion. The producer is typically a non-elementwise operation (like a reduction or transpose) whose result feeds into an elementwise chain:

```
// Before:
%0 = reduce(%input, axes={1})  // [N, M] -> [N]
%1 = exp(%0)                    // [N]
%2 = add(%1, %bias)            // [N]

// After (input fusion):
%fused = fusion(%input, %bias) {
  %p0 = parameter(0)  // input [N, M]
  %p1 = parameter(1)  // bias [N]
  %r = reduce(%p0, axes={1})  // [N]
  %e = exp(%r)                  // [N]
  ROOT %a = add(%e, %p1)       // [N]
}
```

The "input" in the name refers to the fact that the fusion is anchored at the non-elementwise operation (the reduce), and elementwise operations are fused into it as additional computation after the reduce.

#### Output Fusion

Output fusion (also called "multi-output fusion") merges multiple consumers of the same producer into a single fusion. This is profitable when multiple operations consume the same intermediate result:

```
// Before:
%0 = exp(%input)
%1 = tanh(%0)    // Consumer 1
%2 = sigmoid(%0) // Consumer 2

// After (output fusion):
%fused = fusion(%input) {
  %p0 = parameter(0)
  %e = exp(%p0)
  %t = tanh(%e)
  %s = sigmoid(%e)
  ROOT %result = tuple(%t, %s)
}
```

Output fusion is particularly valuable for saving memory bandwidth when the same expensive intermediate result (the `exp` in this example) is used by multiple downstream operations.

#### Custom Fusion

The GPU backend supports custom fusion strategies that go beyond the generic approaches:

- **Triton fusion**: The Triton fusion rewriter identifies fusion patterns that can be efficiently emitted as Triton kernels, particularly those involving matrix multiplications, softmax, layer normalization, and other common deep learning primitives.
- **Gemm fusion**: Fuses elementwise operations around matrix multiplications to eliminate intermediate memory traffic.
- **Convolution fusion**: Fuses operations around convolutions (e.g., conv + bias + activation).

These custom fusion strategies are typically implemented as separate passes that run after the generic fusion pass and rewrite specific patterns into specialized fusion computations.

**Fusion invariants:**

XLA maintains several invariants about fusion operations:

1. **No cycles**: The fusion computation's dataflow graph must be a DAG.
2. **Single root**: Each fusion has exactly one root instruction (though multi-output fusion produces a tuple).
3. **Parameter operands**: The fusion's operands are exactly the parameters of the fused computation.
4. **Shape compatibility**: All instructions in a loop fusion must have compatible iteration domains.
5. **No nested fusion**: A fusion cannot contain another fusion (fusion computations are always flat).

### 14.3.9 LayoutAssignment

`LayoutAssignment` assigns a physical memory layout to each tensor in the HLO module. The layout determines how multi-dimensional arrays are stored in linear memory, which has profound performance implications on all hardware backends.

**Logical shape vs. physical layout:**

In HLO, every tensor has a *logical shape* (e.g., `[batch, height, width, channels]`) that describes its mathematical structure. The *physical layout* determines the ordering of these dimensions in linear memory:

```
Logical shape: [2, 3, 4]
Layout {0, 1, 2}: element [i, j, k] is at offset i*12 + j*4 + k      (row-major / "C" order)
Layout {2, 1, 0}: element [i, j, k] is at offset k*6 + j*2 + i      (column-major / "Fortran" order)
Layout {2, 0, 1}: element [i, j, k] is at offset k*6 + i*3 + j      (custom order)
```

**Layout selection strategy:**

The pass uses a cost model to select the optimal layout for each instruction:

1. **Backend-specific constraints**: Different backends have different preferences:
   - **GPU**: cuDNN prefers NHWC (batch, height, width, channels) for convolutions. cuBLAS prefers column-major for matrix operations.
   - **CPU**: Eigen prefers NHWC for convolutions.
   - **TPU**: The TPU has fixed layout requirements determined by the hardware.

2. **Propagation**: The pass propagates layout preferences from "anchor" instructions (like convolutions and dot products) through elementwise operations. If a convolution prefers NHWC, the elementwise operations around it should also use NHWC to avoid copy operations.

3. **Copy insertion**: When a layout conflict cannot be resolved (i.e., an instruction's output needs different layouts for different consumers), the pass inserts a `copy` instruction that transposes the data:

```
// If convolution output is NHWC but dot product needs NCHW:
%conv = convolution(...), layout={0,1,2,3}  // NHWC
%copy = copy(%conv), layout={0,3,1,2}       // NCHW
%dot = dot(%copy, ...)
```

4. **Cost minimization**: The pass minimizes the total number of copy instructions while satisfying all layout constraints. This is formulated as an optimization problem and solved using a greedy algorithm.

**Implementation:**

The `LayoutAssignment` pass is actually a pipeline of several sub-passes:

```
LayoutAssignmentPipeline:
  1. LayoutAssignment           -- Assign layouts to all instructions
  2. CopyInsertion              -- Insert copies for layout conflicts
  3. MemorySpaceAssignment      -- Assign memory spaces (for TPU)
  4. LayoutSimplification       -- Simplify redundant copies
  5. HloDCE                     -- Remove dead copies
```

### 14.3.10 ShardingPropagation

`ShardingPropagation` implements SPMD (Single Program, Multiple Data) sharding by propagating sharding annotations from explicitly annotated instructions throughout the HLO graph. This is based on the GSPMD paper and is the mechanism by which XLA implements tensor parallelism, data parallelism, and pipeline parallelism.

**How it works:**

1. Users annotate key operations with sharding specifications (e.g., "this matrix is sharded along dimension 1 across 4 devices").
2. The propagation pass propagates these shardings through the graph:
   - For elementwise operations, the sharding is propagated directly (the output has the same sharding as the input).
   - For reshape, the sharding is adjusted to match the new shape.
   - For dot products, the sharding is determined by the operand shardings and may require communication (all-reduce, all-gather, etc.).
   - For reduce operations, dimensions that are sharded are reduced locally and then combined via communication.

3. Where sharding conflicts arise (an instruction would need different shardings from different operands or uses), the pass inserts explicit communication operations:

```
// If operand A is sharded on dim 1 but operand B expects sharding on dim 0:
%resharded = collective-permute(%A), source_target_pairs={{0,1}, {1,0}}
```

**Sharding types:**

```protobuf
message OpSharding {
  enum Type {
    REPLICATED = 0;     // Full copy on each device
    MAXIMAL = 1;        // Entire tensor on one device
    TUPLE = 2;          // Tuple of shardings
    OTHER = 3;          // Tiled sharding
  }
  Type type = 1;
  repeated int64 tile_assignment_dimensions = 2;
  repeated int64 tile_assignment_devices = 3;
  repeated int64 replicate_group_ids = 4;
}
```

**Communication insertion**: When operations require data from multiple devices, the pass inserts collective operations:

- `all-reduce`: Combine partial reductions across devices.
- `all-gather`: Gather sharded dimensions.
- `collective-permute`: Exchange data between specific device pairs.
- `reduce-scatter`: Reduce and redistribute.

### 14.3.11 BFloat16Normalization / BFloat16Propagation

`BFloat16Propagation` (and its companion `BFloat16Normalization`) manages precision reduction from FP32 to BF16 where it is safe to do so. This optimization reduces memory bandwidth and increases throughput on hardware that supports BF16 (TPU, NVIDIA Ampere+ GPUs).

**Algorithm:**

1. **Identify supported operations**: Not all operations can safely use BF16. The pass maintains a set of operations that support BF16 inputs/outputs.
2. **Propagation from roots**: Starting from operations that benefit from BF16 (e.g., dot products, convolutions), propagate BF16 precision backward through elementwise chains.
3. **Boundary handling**: At the boundary between BF16 and FP32 regions, insert `convert` instructions to change precision:
   ```
   // Mixed precision:
   %bf16_input = convert(%fp32_input)  // FP32 -> BF16
   %bf16_result = dot(%bf16_input, %bf16_weights)
   %fp32_result = convert(%bf16_result)  // BF16 -> FP32
   ```
4. **Lossy operation protection**: Operations that are sensitive to precision loss (e.g., reductions of small values, softmax denominators) may be kept in FP32 to maintain numerical stability.

---

## 14.4 GPU-Specific Passes

These passes are only run when targeting the GPU backend. They rewrite HLO operations into forms that can be efficiently emitted as GPU kernels or library calls.

### 14.4.1 CudnnFusedConvRewriter

`CudnnFusedConvRewriter` identifies patterns of the form `convolution + bias + activation` and rewrites them into a single `custom-call` instruction that invokes cuDNN's fused convolution API. This allows cuDNN to optimize the entire operation as a unit, which is significantly more efficient than executing each operation separately.

**Pattern matching:**

```
// Matches:
%conv = convolution(%input, %weights)
%bias = broadcast(%bias_param)
%add = add(%conv, %bias)
%act = relu(%add)  // or sigmoid, tanh, etc.

// Rewrites to:
%fused = custom-call(%input, %weights, %bias_param),
         custom_call_target="__cudnn$convBiasActivation",
         backend_config={activation: "relu", ...}
```

**Supported activations**: ReLU, sigmoid, tanh, and identity (bias-only fusion).

**Benefits**: Fusing bias and activation into the convolution:
- Eliminates two kernel launches (add, activation).
- Eliminates two global memory reads/writes for intermediate results.
- Allows cuDNN to use optimized fused kernels that keep intermediate results in registers or shared memory.

### 14.4.2 CudnnNormRewriter

`CudnnNormRewriter` identifies batch normalization and layer normalization patterns and rewrites them into cuDNN normalization calls:

```
// Matches (batch norm):
%mean = reduce(%input, axes={0,2,3})
%var = reduce(square(sub(%input, %mean)), axes={0,2,3})
%norm = div(sub(%input, %mean), sqrt(add(%var, epsilon)))
%output = add(mul(%norm, %gamma), %beta)

// Rewrites to:
%bn = custom-call(%input, %gamma, %beta),
      custom_call_target="__cudnn$batchNorm",
      ...
```

### 14.4.3 Triton Fusion Rewriter

The Triton fusion rewriter is a GPU-specific pass that identifies fusion patterns amenable to Triton kernel generation. It targets:

- **Matmul + elementwise**: Fusing bias addition, activation, and dropout around matrix multiplications.
- **Softmax**: Complete softmax computation including exp, sum, and div.
- **Layer normalization**: Mean, variance, normalize, scale, and shift.
- **Flash attention**: The complete scaled dot-product attention pattern (Q * K^T / sqrt(d) + mask + softmax * V).

The rewriter creates `custom-call` instructions with the `__triton` prefix that are later lowered to Triton IR during code generation.

```
// Matches:
%dot = dot(%query, %transpose(%key))
%scale = multiply(%dot, %scale_factor)
%softmax = softmax(%scale)
%output = dot(%softmax, %value)

// Rewrites to:
%attention = custom-call(%query, %key, %value),
             custom_call_target="__triton$flash_attention",
             ...
```

### 14.4.4 GemmRewriter

`GemmRewriter` rewrites dot-product operations into a form suitable for calling optimized GEMM libraries (cuBLAS). It handles:

1. **Batched matmul**: Recognizing dot operations with batch dimensions.
2. **Transpose folding**: Incorporating transposes of operands into the GEMM call (using cuBLAS transpose flags instead of explicit transpose operations).
3. **Contraction detection**: Identifying general contraction patterns that can be expressed as batched matrix multiplications.

```
// Before:
%0 = transpose(%weights, {1, 0})  // Transpose weights
%1 = dot(%input, %0)              // Input is [M, K], Weights are [K, N]

// After:
%1 = custom-call(%input, %weights),
     custom_call_target="__cublas$gemm",
     backend_config={
       lhs_stride: K,
       rhs_stride: N,
       transpose_rhs: true,  // Tell cuBLAS to transpose internally
       ...
     }
```

---

## 14.5 CPU-Specific Passes

### 14.5.1 ConvCanonicalization

`ConvCanonicalization` rewrites convolution operations into a canonical form that the CPU backend (using Eigen) can efficiently process. This involves:

1. **Dimension reordering**: Ensuring the convolution dimensions are in the order expected by Eigen's convolution implementation (NHWC format).
2. **Padding normalization**: Converting padding specifications to Eigen's format.
3. **Dilation handling**: Expressing dilated convolutions in terms of non-dilated operations where Eigen lacks direct dilation support.
4. **Grouped convolution decomposition**: Decomposing grouped convolutions into multiple smaller convolutions if Eigen does not natively support the grouping configuration.

### 14.5.2 ParallelTaskAssigner

`ParallelTaskAssigner` analyzes the HLO graph and assigns instructions to parallel tasks for multi-threaded execution on the CPU. It determines:

1. **Parallelism boundaries**: Which instructions can execute in parallel vs. which must be sequential.
2. **Cost estimation**: The computational cost of each instruction, used to balance work across threads.
3. **Data dependency analysis**: Ensures that parallel tasks respect all data dependencies.
4. **Thread assignment**: Maps tasks to threads in a way that minimizes synchronization overhead and maximizes cache locality.

The pass uses a cost model based on `HloCostAnalysis` to estimate the FLOP count and memory traffic of each instruction, then uses a greedy scheduling algorithm to assign instructions to threads.

---

## 14.6 TPU-Specific Passes

### 14.6.1 Spatial Partitioning

TPU spatial partitioning breaks large tensors into tiles that fit within a single TPU core's spatial dimensions. When a tensor is too large to fit in a single TPU's systolic array, spatial partitioning:

1. **Tiles the computation**: Breaks large dot products or convolutions into smaller tiles.
2. **Inserts gather/scatter**: Combines partial results from each tile.
3. **Manages cross-tile dependencies**: Ensures correct data flow between partitioned operations.

### 14.6.2 BFloat16 Conversion

The TPU backend aggressively converts operations to BF16 because TPU hardware operates natively in BF16. This pass:

1. Inserts explicit `convert` instructions at FP32/BF16 boundaries.
2. Ensures that accumulation in dot products uses BF16 where supported.
3. Maintains FP32 precision for operations that require it (e.g., loss functions, certain normalization steps).

---

## 14.7 Analysis Passes

Analysis passes examine the HLO module without modifying it. They are used by transformation passes to guide optimization decisions and by the verifier to check invariants.

### 14.7.1 HloDataflowAnalysis

`HloDataflowAnalysis` performs a dataflow analysis of the HLO module, computing the set of values that flow through the module's computations and their relationships. It tracks:

- **Value definitions**: Where each value is created (parameters, constants, operation outputs).
- **Value uses**: Where each value is consumed (operand of an instruction).
- **Aliases**: Values that share the same underlying buffer (e.g., the output of a `tuple` instruction aliases its inputs).
- **Transitive uses**: The complete set of uses reachable from a definition.

This analysis is essential for:

- **Buffer assignment**: Determining when buffers can be reused.
- **Alias analysis**: Identifying when two instructions may share memory.
- **Rematerialization**: Computing live ranges.
- **Copy insertion**: Determining when copies are needed to preserve semantics.

Key API:

```cpp
class HloDataflowAnalysis {
  // Get the set of values that instruction `instruction` produces at index
  // `index` in its output shape.
  const ValueSet& GetValueSet(const HloInstruction* instruction,
                               const ShapeIndex& index) const;

  // Get the unique value that defines the value at the given position,
  // or nullptr if the position has multiple defining values.
  const HloValue* GetUniqueValueAt(const HloInstruction* instruction,
                                     const ShapeIndex& index) const;

  // Get all values defined in the module.
  const std::vector<HloValue>& values() const;
};
```

### 14.7.2 HloAliasAnalysis

`HloAliasAnalysis` determines which instruction outputs may share (alias) the same underlying buffer. This is critical for:

- **Memory optimization**: Buffers that never overlap in time can share the same memory.
- **Copy insertion**: If an instruction's output aliases its input, modifications to the output would corrupt the input. A copy must be inserted.
- **In-place operations**: Identifying operations that can safely operate in-place (modifying their input buffer directly).

The analysis considers:

- **Buffer sharing constraints**: `tuple` instructions always alias their operands. `while` loop outputs alias their initial values.
- **Interference**: Two values interfere if they are both live at the same point in the execution order. Non-interfering values can share buffers.
- **User-specified aliasing**: Some operations (e.g., in-feed/out-feed) have specific aliasing constraints defined by the frontend.

```cpp
class HloAliasAnalysis {
  // Get the alias analysis result for the entire module.
  const HloAliasInfo& alias_info() const;

  // Check if two instruction outputs may alias.
  bool MayAlias(const HloInstruction* a, const ShapeIndex& a_index,
                const HloInstruction* b, const ShapeIndex& b_index) const;
};
```

### 14.7.3 HloCostAnalysis

`HloCostAnalysis` estimates the computational cost (in FLOPs) and memory access cost (in bytes) of each instruction in the HLO module. This information is used by:

- **Fusion**: To decide whether fusion is profitable (the memory savings must outweigh the potential register pressure increase).
- **Rematerialization**: To compute the cost of recomputing an instruction.
- **Scheduling**: To prioritize the scheduling of expensive operations.
- **Parallel task assignment**: To balance work across threads.

The cost model provides:

```cpp
struct HloCostAnalysis {
  // Computational cost metrics
  int64_t flop_count() const;          // Total FLOPs
  int64_t transcendental_count() const; // transcendental ops (exp, log, etc.)

  // Memory access metrics
  int64_t bytes_accessed() const;      // Total bytes read/written
  int64_t operand_bytes_accessed(int64_t operand_num) const;

  // Utilization metrics
  float optimal_seconds() const;       // Estimated runtime on ideal hardware

  // Per-instruction costs
  int64_t GetInstructionFlopCount(const HloInstruction* instr) const;
};
```

The cost model is parameterized by the target hardware's peak FLOP rate and memory bandwidth. Different backends provide different parameterizations.

### 14.7.4 HloVerifier

`HloVerifier` checks the structural and semantic invariants of the HLO module. It is typically run before and after each optimization pass to catch bugs early.

**Invariants checked:**

1. **Shape consistency**: Every instruction's output shape matches its computation. Operand shapes match parameter declarations.
2. **Computation structure**: Every computation has exactly one root instruction. All instructions are reachable from the root.
3. **Parameter numbering**: Parameters in each computation are numbered starting from 0 without gaps.
4. **Fusion invariants**: Fusion computations have the correct parameter count. No nested fusions.
5. **Layout consistency**: If layout-sensitive verification is enabled, all layouts are consistent with the module's layout assignment.
6. **Control flow**: Control dependencies form a valid DAG (no cycles).
7. **Custom call targets**: Custom call instructions reference valid targets.
8. **Sharding consistency**: Sharding annotations are compatible with operation semantics.

**Usage:**

```cpp
// The verifier is typically added to a pipeline:
pipeline->AddPass<HloVerifier>(/*layout_sensitive=*/false,
                                /*mix_type=*/false);
```

When the verifier detects an invariant violation, it produces a detailed error message including the offending instruction, the expected invariant, and a dump of the HLO module for debugging.

---

## 14.8 Tooling

### 14.8.1 hlo-opt

`hlo-opt` is a command-line tool for testing and developing HLO passes. It is analogous to LLVM's `opt` tool and allows running individual passes or pipelines on HLO modules read from files.

**Usage:**

```bash
# Run a single pass on an HLO module
hlo-opt --pass=algebraic-simplifier input.hlo

# Run multiple passes in sequence
hlo-opt --pass=algebraic-simplifier --pass=constant-folding --pass=dce input.hlo

# Run the full optimization pipeline
hlo-opt --optimize input.hlo

# Dump the HLO module after each pass
hlo-opt --optimize --dump-passes-to=/tmp/hlo-dumps input.hlo

# Compare before and after
hlo-opt --pass=reshape-mover --print-before --print-after input.hlo

# Run with specific backend configuration
hlo-opt --backend=gpu --pass=fusion input.hlo
```

**Input format**: `hlo-opt` accepts HLO text proto format (`.hlo` or `.hlo.txt`):

```
HloModule my_module

ENTRY %entry (param0: f32[128,512], param1: f32[512,256]) -> f32[128,256] {
  %param0 = parameter(0), f32[128,512]
  %param1 = parameter(1), f32[512,256]
  ROOT %dot = dot(%param0, %param1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

### 14.8.2 Testing HLO Passes

XLA uses Google Test (gtest) for testing HLO passes. Each pass has a corresponding test file in `xla/service/` that uses the `HloTestBase` class:

```cpp
class AlgebraicSimplifierTest : public HloTestBase {
 protected:
  void SetUp() override {
    HloTestBase::SetUp();
    options_ = AlgebraicSimplifierOptions(/*layout_sensitive=*/false);
  }

  AlgebraicSimplifierOptions options_;
};

TEST_F(AlgebraicSimplifierTest, AddZero) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder("test");

  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {4}), "p"));
  auto zero = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({0,0,0,0})));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {4}),
                                    HloOpcode::kAdd, param, zero));
  builder.AddInstruction(HloInstruction::CreateTuple({add}));

  module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(RunPass(module.get()));
  EXPECT_EQ(module->entry_computation()->root_instruction(), param);
}
```

**Test patterns:**

- **Positive tests**: Verify that the pass transforms known patterns correctly.
- **Negative tests**: Verify that the pass does not transform patterns it should not.
- **Fixpoint tests**: Verify that running the pass again produces no further changes.
- **Roundtrip tests**: Verify that the transformation is semantics-preserving by comparing results before and after.

### 14.8.3 Pass Registration

HLO passes are registered with the pass pipeline through the backend-specific pipeline builder. Each backend (GPU, CPU, TPU) defines its own pipeline:

```cpp
// In xla/service/gpu/gpu_compiler.cc:
StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, ...) {
  auto pipeline = BuildGpuCompilerPipeline(module->config(), ...);
  TF_RETURN_IF_ERROR(pipeline->Run(module.get()));
  return module;
}
```

Passes can also be registered dynamically via flags:

```bash
# Disable specific passes
--xla_disable_hlo_passes=algebraic-simplifier,dce

# Enable optional passes
--xla_enable_hlo_passes=triton-fusion-rewriter

# Run passes in a custom order
--xla_hlo_passes_only=algebraic-simplifier,constant-folding
```

### 14.8.4 Custom Pipelines

Developers can create custom optimization pipelines for experimentation or specialized workloads:

```cpp
std::unique_ptr<HloPassPipeline> BuildCustomPipeline(
    const HloModuleConfig& config) {
  auto pipeline = std::make_unique<HloPassPipeline>("custom");

  // Phase 1: Simplification
  pipeline->AddPass<CallInliner>();
  pipeline->AddPass<FlattenCallGraph>();
  pipeline->AddPass<AlgebraicSimplifier>(AlgebraicSimplifierOptions());
  pipeline->AddPass<HloConstantFolding>();
  pipeline->AddPass<HloDCE>();

  // Phase 2: Fusion
  pipeline->AddPass<Fusion>(FusionConfig());

  // Phase 3: Layout and memory
  pipeline->AddPass<LayoutAssignment>(...);
  pipeline->AddPass<HloRematerialization>(...);

  // Phase 4: Backend-specific
  pipeline->AddPass<CudnnFusedConvRewriter>();
  pipeline->AddPass<GemmRewriter>();

  return pipeline;
}
```

Custom pipelines are activated via the `--xla_backend_extra_passes` flag or by modifying the backend's pipeline builder function.

---

## 14.9 Pass Execution Order

The typical pass execution order in XLA's GPU compilation pipeline is:

```
Phase 1: Simplification
  1. CallInliner
  2. FlattenCallGraph
  3. AlgebraicSimplifier
  4. HloConstantFolding
  5. HloDCE
  6. ReshapeMover
  7. TransposeFolding
  8. ZeroSizedHloElimination

Phase 2: Sharding and Layout
  9. ShardingPropagation
  10. BFloat16Normalization
  11. LayoutAssignment
  12. CopyInsertion

Phase 3: Memory Optimization
  13. HloRematerialization
  14. HloDCE

Phase 4: Fusion
  15. Fusion (generic)
  16. CudnnFusedConvRewriter (GPU)
  17. CudnnNormRewriter (GPU)
  18. TritonFusionRewriter (GPU)
  19. GemmRewriter (GPU)

Phase 5: Scheduling and Buffer Assignment
  20. InstructionScheduling
  21. BufferAssignment

Phase 6: Verification
  22. HloVerifier (final)
```

The exact order may vary between backends and XLA versions. The pipeline is designed so that each phase produces output suitable for the next phase, and the phases are ordered to maximize the effectiveness of downstream optimizations.

---

## 14.10 Debugging HLO Passes

**Pass dump files**: When `--xla_dump_hlo_as_proto` or `--xla_dump_hlo_as_text` is set, XLA dumps the HLO module before and after each pass. The files are named:

```
module_name.before_passes.hlo
module_name.after_PASS_NAME.hlo
module_name.after_passes.hlo
```

**Interactive debugging**: Use `--xla_hlo_dump_dir` to specify a directory for dumps, then use `hlo-opt` or the XLA HLO visualizer to examine the transformations:

```bash
# Enable HLO dumping during JAX execution
XLA_FLAGS="--xla_dump_hlo_to=/tmp/hlo_dumps --xla_dump_hlo_pass_re=.*" \
  python my_program.py

# Then inspect the dumps
hlo-opt /tmp/hlo_dumps/my_module.after_algebraic_simplifier.hlo
```

**Pass profiling**: Use `--xla_hlo_profile` to enable pass-level profiling, which reports the wall-clock time spent in each pass:

```
Pass                         Time (ms)  Changed?
algebraic-simplifier         12.3       yes
constant-folding              2.1       yes
dce                           0.8       yes
reshape-mover                 3.4       no
fusion                      145.2       yes
```

This information is invaluable for identifying passes that are slow or that iterate without making progress.
