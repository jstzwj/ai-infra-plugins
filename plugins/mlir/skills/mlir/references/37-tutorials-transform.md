# MLIR Transform Dialect Tutorial

## Overview

This tutorial covers the Transform dialect, which allows precise targeting of transformations at specific operations in the IR through a declarative transformation scripting approach.

### Key Concepts

- **Transform IR**: Operations that specify what transformations to apply
- **Payload IR**: The IR being transformed
- **Handles**: Transform values associated with payload operations/values
- **Parameters**: Transform values associated with constant attributes

## Chapter 0: Structured Linalg Operations Primer

### Uniform Elementwise Extension

```mlir
// Scalar
%2 = arith.addf %0, %1 : f32

// 1D vector
%2 = arith.addf %0, %1 : vector<8xf32>

// Multi-dimensional vector
%2 = arith.addf %0, %1 : vector<8x4xf32>
```

### Reduction

```mlir
%1 = vector.reduction <add>, %0 : vector<8xf32> into f32
```

Loop equivalent:
```mlir
%result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
  %element = vector.extract %0[%i] : f32 into vector<8xf32>
  %updated = arith.addf %partial, %element : f32
  scf.yield %updated : f32
}
```

### Contraction

```mlir
// Matrix multiplication as contraction
%result = vector.contract {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
```

Loop form:
```
for i in 0 to 8:
  for j in 0 to 16:
    for k in 0 to 10:
      init[i, j] += lhs[i, k] * rhs[k, j]
```

### Generic Operation on Memory

```mlir
linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : memref<8x10xf32>, memref<10x16xf32>)
  outs(%init : memref<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
}
```

### Loop Fusion in Region

```mlir
// ReLU = max(0, x) expressed as fused operation
linalg.generic {
  indexing_maps [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
  iterator_types = ["parallel"]
} ins(%in : memref<?xf32>) outs(%out : memref<?xf32>) {
^bb0(%in_one : f32, %out_one : f32):
  %c0 = arith.constant 0.0 : f32
  %0 = arith.cmpf ogt %in_one, %c0 : f32
  %1 = arith.select %0, %in_one, %c0 : f32
  linalg.yield %1 : f32
}
```

### Generic Operation on Tensors

```mlir
%result = linalg.generic {
  indexing_maps = [...],
  iterator_types = [...]
} ins(%lhs, %rhs : tensor<8x10xf32>, tensor<10x16xf32>)
  outs(%init : tensor<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
} -> tensor<8x16xf32>
```

### Tiling

Tiling partitions iteration space, materializing loops:

```mlir
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %init) -> (tensor<8x16xf32>) {
  %3 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %4 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)
  %lhs_slice = tensor.extract_slice %lhs[%3, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  // ... same linalg.generic on smaller tiles ...
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial into %shared[%3, %4] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
```

### Named Operations

```mlir
%matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
                        outs(%init: tensor<8x16xf32>) -> tensor<8x16xf32>
```

## Chapter 1: Combining Existing Transformations

### Transform IR Basics

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elementwise">):
    transform.yield
  }
}
```

### Handle Types

| Type | Description |
|------|-------------|
| `!transform.any_op` | Handle to any operation |
| `!transform.op<"X">` | Handle to operations of kind X |
| `!transform.any_value` | Handle to any value |

### Failure Propagation

```mlir
transform.sequence failures(propagate) {
  // Fails entire sequence if any nested transform fails
}

transform.sequence failures(suppress) {
  // Continues despite failures (skips remaining transforms after failure)
}
```

### Debugging

```mlir
transform.debug.emit_remark_at %arg1, "matmul"
    : !transform.op<"linalg.matmul">
```

### Running the Interpreter

```bash
mlir-opt sequence.mlir --pass-pipeline="
    builtin.module(transform-interpreter{
        debug-bind-trailing-args=linalg.matmul,linalg.elementwise})"
```

### Tiling

```mlir
%loop, %tiled = transform.structured.tile_using_forall %arg1
                tile_sizes [4, 32]
  : (!transform.op<"linalg.matmul">)
 -> (!transform.any_op, !transform.any_op)
```

Returns:
- Handle to tiled `linalg.matmul` operating on subset
- Handle to the `scf.forall` loop

### Handle Invalidation

Transforms that erase/recreate payload operations **consume** their operand handles, invalidating them. Using an invalidated handle is undefined behavior.

```mlir
// This consumes %arg1, invalidating it
%loop, %tiled = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]

// ERROR: %arg1 is now invalid!
transform.debug.emit_remark_at %arg1, "remark"
```

### Expensive Checks Mode

The interpreter defaults to expensive checks that detect invalid handle usage:

```sh
error: op uses a handle invalidated by a previously executed transform op
```

Disable with `disable-expensive-checks` option when performance matters.

### Aliasing Handles

```mlir
%casted = transform.cast %arg1 : !transform.op<"linalg.matmul">
    to !transform.any_op
// Both %arg1 and %casted reference the same payload
// Consuming either invalidates both
```

### Splitting Handles

```mlir
%add, %max = transform.split_handle %arg2
    : (!transform.op<"linalg.elementwise">)
    -> (!transform.any_op, !transform.any_op)
```

### Chaining: Tile and Fuse

```mlir
// Tile the last operation
%tiled_max, %loop =
    transform.structured.tile_using_forall %max tile_sizes [8, 32]

// Fuse producers into the loop
%add_fused, %loop_0 =
    transform.structured.fuse_into_containing_op %add into %loop
%matmul_fused, %loop_1 =
    transform.structured.fuse_into_containing_op %arg1 into %loop_0

// Tile again for microkernel size
%tiled_2, %loop_2 =
    transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]
%matmul_fused_2, %loop_3 =
    transform.structured.fuse_into_containing_op %matmul_fused into %loop_2
```

### Outlining

```mlir
// Materialize outer loop for outlining
%_, %outline_target =
    transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
transform.structured.fuse_into_containing_op %matmul_fused_2 into %outline_target

// Outline the loop into a function
%func, %call = transform.loop.outline %outline_target
               {func_name = "outlined"}
    : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)
```

### IR Modification Tracking

The Transform dialect automatically tracks all IR changes:
- Erased payload ops are removed from all handles
- Replaced ops are updated in handles if replacement is unambiguous

## Chapter 2: Adding Custom Transform Operations

### Dialect Extension

```c++
class MyExtension : public ::mlir::transform::TransformDialectExtension<MyExtension> {
public:
  using Base::Base;
  void init();
};

void MyExtension::init() {
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();
  registerTransformOps<ChangeCallTargetOp>();
}
```

### ODS Definition

```tablegen
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Changes the callee of a call operation";
  let arguments = (ins
    TransformHandleTypeInterface:$call,
    StrAttr:$new_target);
  let results = (outs);
  let assemblyFormat = "$call `,` $new_target attr-dict `:` type($call)";
}
```

### Implementing TransformOpInterface

```c++
::mlir::DiagnosedSilenceableFailure ChangeCallTargetOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {

  auto payload = state.getPayloadOps(getCall());
  for (Operation *payloadOp : payload) {
    auto call = dyn_cast<::mlir::func::CallOp>(payloadOp);
    if (!call) {
      return emitSilenceableError() << "only applies to func.call payloads";
    }
    updateCallee(call, getNewTarget());
  }
  return DiagnosedSilenceableFailure::success();
}
```

### Memory Effects

```c++
void ChangeCallTargetOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getCall(), effects);  // Handle is not consumed
  modifiesPayload(effects);             // Payload IR is modified
}
```

### Registration

```c++
void registerMyExtension(::mlir::DialectRegistry &registry) {
  registry.addExtensions<MyExtension>();
}
```

### CMake Integration

```cmake
set(LLVM_TARGET_DEFINITIONS MyExtension.td)
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyExtensionIncGen)

add_mlir_library(MyExtension
  MyExtension.cpp
  DEPENDS MyExtensionIncGen
  LINK_LIBS PUBLIC MLIRTransformDialect MLIRFuncDialect MLIRSCFDialect
)
```

## Chapter 3: Advanced Transform Operations

### TransformEachOpTrait

Simplifies per-payload-operation transforms:

```tablegen
def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
    [TransformOpInterface, TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins
    Transform_ConcreteOpType<"func.call">:$call,
    StrAttr:$new_target);
  let results = (outs);
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::func::CallOp call,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}
```

Implementation:

```c++
::mlir::DiagnosedSilenceableFailure ChangeCallTargetOp::applyToOne(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::func::CallOp call,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  updateCallee(call, getNewTarget());
  return DiagnosedSilenceableFailure::success();
}
```

### Custom Transform Type

```tablegen
def CallOpInterfaceHandle
  : TypeDef<Transform_Dialect, "CallOpInterfaceHandle",
      [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let summary = "handle to payload operations implementing CallOpInterface";
  let mnemonic = "my.call_op_interface";
  let assemblyFormat = "";
}
```

Implementation:

```c++
mlir::DiagnosedSilenceableFailure
CallOpInterfaceHandleType::checkPayload(
    mlir::Location loc,
    llvm::ArrayRef<mlir::Operation *> payload) const {
  for (Operation *op : payload) {
    if (llvm::isa<mlir::CallOpInterface>(op))
      continue;
    return emitSilenceableError(loc)
        << "expected payload to implement CallOpInterface";
  }
  return DiagnosedSilenceableFailure::success();
}
```

### Consuming Operands and Producing Results

```tablegen
def CallToOp : Op<Transform_Dialect, "my.call_to_op",
    [TransformOpInterface, TransformEachOpTrait,
     MemoryEffectsOpInterface, FunctionalStyleTransformOpTrait]> {
  let arguments = (ins CallOpInterfaceHandle:$call);
  let results = (outs TransformHandleTypeInterface:$transformed);
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(...);
  }];
}
```

```c++
::mlir::DiagnosedSilenceableFailure CallToOp::applyToOne(
    TransformRewriter &rewriter, CallOpInterface call,
    ApplyToEachResultList &results, TransformState &state) {
  Operation *rewritten = rewriteToOp(call);
  if (!rewritten) return emitDefiniteError() << "failed to rewrite";
  results.push_back(rewritten);
  return DiagnosedSilenceableFailure::success();
}
```

### Memory Effects Traits

| Trait | Meaning |
|-------|---------|
| `FunctionalStyleTransformOpTrait` | All operands consumed, all results produced, payload modified |
| `NavigationTransformOpTrait` | All operands read-only, all results produced, payload read-only |

## Chapter 4: Matching Payload with Transform Operations

### Simple Match

```mlir
transform.named_sequence @match_elemwise(
    %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.operation_name %entry ["linalg.elementwise"]
    : !transform.any_op
  transform.yield %entry : !transform.any_op
}
```

### Collecting Matches

```mlir
%elemwise = transform.collect_matching @match_elemwise in %root
  : (!transform.any_op) -> !transform.any_op
%matmul = transform.collect_matching @match_matmul in %root
  : (!transform.any_op) -> !transform.any_op
```

### Matching Chains of Operations

```mlir
transform.named_sequence @match_matmul_elemwise(
    %last: !transform.any_op {transform.readonly})
    -> (!transform.any_op, !transform.any_op, !transform.any_op) {
  transform.match.operation_name %last ["linalg.elementwise"] : !transform.any_op
  %middle = transform.get_producer_of_operand %last[0]
    : (!transform.any_op) -> !transform.any_op
  transform.match.operation_name %middle ["linalg.elementwise"] : !transform.any_op
  %matmul = transform.get_producer_of_operand %middle[0]
    : (!transform.any_op) -> !transform.any_op
  transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
  transform.yield %matmul, %middle, %last
    : !transform.any_op, !transform.any_op, !transform.any_op
}
```

### Defining Custom Match Operations

```tablegen
def HasOperandSatisfyingOp : TransformDialectOp<"match.my.has_operand_satisfying",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     MatchOpInterface,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let arguments = (ins TransformHandleTypeInterface:$op);
  let results = (outs TransformParamTypeInterface:$position,
                      Variadic<Transform_AnyHandleOrParamType>:$results);
  let regions = (region SizedRegion<1>:$body);
}
```

Match operations:
- Must implement `MatchOpInterface` (tag interface, no extra methods)
- Must not modify payload IR
- Return silenceable failure to indicate no match

### Using Custom Matchers

```mlir
%pos, %middle = transform.match.my.has_operand_satisfying %last
    : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op) {
^bb0(%operand: !transform.any_value):
  %def = transform.get_defining_op %operand
    : (!transform.any_value) -> !transform.any_op
  transform.match.operation_name %def ["linalg.elementwise"] : !transform.any_op
  transform.yield %def : !transform.any_op
}
```

### foreach_match

```mlir
transform.foreach_match in %root
  @match_matmul_elemwise -> @action_matmul_elemwise
  : (!transform.any_op) -> !transform.any_op
```

### Matching Inferred Features

Match `linalg.generic` that is actually a matrix multiplication:

```mlir
transform.named_sequence @match_generic_matmul(
    %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
  transform.match.structured %candidate : !transform.any_op {
  ^bb0(%c: !transform.any_op):
    // Rank == 3
    %rank = transform.match.structured.rank %c : (!transform.any_op) -> !transform.param<i64>
    %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>

    // 2 inputs, 1 output
    %n_ins = transform.match.structured.num_inputs %c : (!transform.any_op) -> !transform.param<i64>
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %n_ins, %c2 : !transform.param<i64>

    // All inputs/outputs accessed with projected permutation
    transform.match.structured.input %c[all] {projected_permutation} : !transform.any_op
    transform.match.structured.init %c[0] {projected_permutation} : !transform.any_op

    // Body is mulf/addf contraction
    transform.match.structured.body %c
      { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op

    // Exactly 1 LHS, 1 RHS, 1 reduction, 0 batch dimensions
    %batch, %lhs, %rhs, %reduction =
    transform.match.structured.classify_contraction_dims %c : (!transform.any_op)
      -> (!transform.param<i64>, !transform.param<i64>,
          !transform.param<i64>, !transform.param<i64>)
    %c0 = transform.param.constant 0 : i64 -> !transform.param<i64>
    %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
    %n_batch = transform.num_associations %batch : (!transform.param<i64>) -> !transform.param<i64>
    transform.match.param.cmpi eq %n_batch, %c0 : !transform.param<i64>
    transform.match.param.cmpi eq %n_lhs, %c1 : !transform.param<i64>
  }
  transform.yield %candidate : !transform.any_op
}
```

## Chapter H: Reproducing Halide Schedule

### Problem: 2D Channeled Convolution

```cpp
// Halide computation
conv(c, x, y, n) = bias(c);
conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y, y + r.z, n);
relu(c, x, y, n) = max(0, conv(c, x, y, n));
```

MLIR equivalent:
```mlir
%convolved = linalg.generic {
  iterator_types = ["parallel", "parallel", "parallel", "parallel",
                    "reduction", "reduction", "reduction"],
  indexing_maps = [...]
} ins(%filter, %input: ...) outs(%biased : ...) { ... }

%relued = linalg.generic {
  iterator_types = ["parallel", "parallel", "parallel", "parallel"],
  ...
} ins(%c0, %convolved : ...) outs(%output : ...) { ... }
```

### Mapping Halide Primitives

| Halide | Transform Dialect |
|--------|-------------------|
| `split` | `tile_using_forall` |
| `reorder` | `tile_using_forall` with size 1, or `interchange` |
| `vectorize` | `vectorize_children_and_apply_patterns` |
| `unroll` | `transform.loop.unroll` |
| `compute_at` | `fuse_into_containing_op` |

### Recreating Loop Structure

```mlir
// Split c dimension
%co, %relu2 = transform.structured.tile_using_forall %relu
    tile_sizes [0, 0, 0, 64]

// Split n, y, x dimensions
%n_y_xo, %relu3 = transform.structured.tile_using_forall %relu2
    tile_sizes [1, 1, 5, 0]

// Fuse conv into relu loops
%conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
%conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2 into %n_y_xo

// Fuse bias into conv+relu loops
%bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
%bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2 into %n_y_xo2

// Materialize reduction loops
%rz_ry_rx, %red_fill, %conv4, %comb =
  transform.structured.tile_reduction_using_for %conv3
  by tile_sizes=[0, 0, 0, 0, 1, 1, 1]
```

### Unrolling

```mlir
// Unroll inner loops (inner first to avoid invalidating handles)
transform.loop.unroll %bias_ci {factor = 4}
transform.loop.unroll %bias_xi {factor = 5}
transform.loop.unroll %conv_ci {factor = 4}
transform.loop.unroll %conv_xi {factor = 5}
transform.loop.unroll %relu_ci {factor = 4}
transform.loop.unroll %relu_xi {factor = 5}
```

### Vectorization

```mlir
// Simplify before vectorization
transform.apply_patterns to %f00 {
  transform.apply_patterns.canonicalization
  transform.apply_patterns.linalg.tiling_canonicalization
}
transform.apply_cse to %f00

// Fold unit dimensions
transform.apply_patterns to %f00 {
  transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
}

// Vectorize
%fv = transform.structured.vectorize_children_and_apply_patterns %f00
```

### Bufferization and Lowering

```mlir
// One-shot bufferize
%arg1 = transform.bufferization.one_shot_bufferize %arg0 {
  bufferize_function_boundaries = true,
  function_boundary_type_conversion = 1 : i32
}

// Buffer deallocation
%f = transform.structured.match ops{["func.func"]} in %arg1
transform.apply_registered_pass "buffer-deallocation-pipeline" to %f

// Lower vector operations
transform.apply_patterns to %fb {
  transform.apply_patterns.vector.lower_contraction lowering_strategy = parallelarith
  transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
  transform.apply_patterns.vector.lower_transpose lowering_strategy = eltwise
  transform.apply_patterns.vector.lower_shape_cast
}

// Transfer to SCF
transform.apply_patterns to %fb {
  transform.apply_patterns.vector.transfer_to_scf
  transform.apply_patterns.memref.alloc_to_alloca
}
transform.bufferization.buffer_loop_hoisting %fb
```

### Performance Results

| Approach | Time | GFlops | Peak % |
|----------|------|--------|--------|
| MLIR (loop unroll before bufferize) | ~420ms | ~14 | 22% |
| MLIR (multi-dim vectors) | ~110ms | ~54 | 84% |
| Halide | ~120ms | ~49 | 77% |

### Key Insight: Multi-Dimensional Vectors

Instead of materializing and unrolling loops, use multi-dimensional vector types directly:

```mlir
// Instead of tiling + unrolling + vectorizing a 5x4 loop nest,
// just vectorize the 5x64 structure directly:
// vector<5x64xf32> operations are lowered to target vectors automatically
```

This produces cleaner IR and better register allocation in the final assembly.

### Summary

| Chapter | Key Concept |
|---------|-------------|
| Ch0 | Structured operations, tiling, fusion in Linalg |
| Ch1 | Transform IR, handles, tiling, fusion, invalidation |
| Ch2 | Custom transform ops via dialect extensions |
| Ch3 | TransformEachOpTrait, custom types, memory effects traits |
| Ch4 | Match operations, payload matching, inferred features |
| ChH | Complete Halide schedule reproduction, vectorization strategies |
