# MLIR Dialects in TensorFlow

This document provides a comprehensive reference for the MLIR (Multi-Level Intermediate
Representation) dialects used in TensorFlow. MLIR provides a flexible infrastructure
for compiler development, and TensorFlow uses multiple dialects to represent computations
at various levels of abstraction.

## Table of Contents

1. [MLIR Overview](#mlir-overview)
2. [TF Dialect](#tf-dialect)
3. [tf_executor Dialect](#tf_executor-dialect)
4. [tf_saved_model Dialect](#tf_saved_model-dialect)
5. [MHLO Dialect](#mhlo-dialect)
6. [StableHLO Dialect](#stablehlo-dialect)
7. [TOSA Dialect](#tosa-dialect)
8. [TFLite Dialect](#tflite-dialect)
9. [Legalization Passes](#legalization-passes)
10. [Lowering Pipeline](#lowering-pipeline)
11. [MLIR Optimization Passes](#mlir-optimization-passes)
12. [Op Definition](#op-definition)
13. [Pattern Rewriting](#pattern-rewriting)
14. [Conversion Framework](#conversion-framework)
15. [Dialect Registration](#dialect-registration)

---

## MLIR Overview

MLIR is a reusable and extensible compiler infrastructure that provides a common
framework for implementing domain-specific compilers. It is designed for representability,
reliability, and composability.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Dialect** | A namespace for a set of operations, types, and attributes |
| **Operation** | A unit of computation (e.g., `tf.Add`, `mhlo.Dot`) |
| **Region** | A ordered list of blocks, representing a scope for operations |
| **Block** | An ordered list of operations, with arguments (SSA values) |
| **Value** | An SSA value produced by an operation or block argument |
| **Type** | The type of a value (e.g., `tensor<10xf32>`, `f32`) |
| **Attribute** | Compile-time constant data attached to operations |

### MLIR Hierarchy

```
Module
  |-- Region (function body)
  |     |-- Block (basic block)
  |     |     |-- Operation (tf.Add, mhlo.Dot, etc.)
  |     |     |-- BlockArgument
  |     |     +-- Value
  |     +-- Block
  +-- Region
```

### MLIR Textual Form

```mlir
module {
  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tf.Add(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    func.return %0 : tensor<10xf32>
  }
}
```

### Key MLIR Components in TensorFlow

```
tensorflow/compiler/mlir/
  |-- tensorflow/        # TF dialect, tf_executor, tf_saved_model
  |-- lite/              # TFLite conversion, flatbuffer support
  |-- tosa/              # TOSA conversion passes
  |-- tf2xla/            # TF to XLA/HLO conversion
  |-- tfr/               # TF Rewriter dialect
  |-- tfrt/              # TF Runtime dialects
  |-- tools/             # MLIR tools (tf-mlir-translate, etc.)
  |-- utils/             # Utility functions
  +-- python/            # Python bindings
```

---

## TF Dialect

The TF dialect (`tf`) represents TensorFlow operations directly in MLIR. Each
TensorFlow operation maps to a corresponding MLIR operation in the TF dialect.

### Dialect Registration

```cpp
// From: tensorflow/compiler/mlir/tensorflow/dialect_registration.h

inline void RegisterAllTensorFlowDialectsImpl(DialectRegistry &registry,
                                              bool include_extensions = true) {
  registry
      .insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
              mlir::ml_program::MLProgramDialect, mlir::TF::TensorFlowDialect,
              mlir::tf_type::TFTypeDialect, mlir::cf::ControlFlowDialect,
              mlir::tf_device::TensorFlowDeviceDialect,
              mlir::tf_executor::TensorFlowExecutorDialect,
              mlir::tf_saved_model::TensorFlowSavedModelDialect,
              mlir::tfg::TFGraphDialect>();
}
```

### Common TF Operations

| Operation | TF Op | Description |
|-----------|-------|-------------|
| `tf.Const` | `tf.constant` | Constant tensor |
| `tf.Cast` | `tf.cast` | Type casting |
| `tf.Add` | `tf.add` | Element-wise addition |
| `tf.Sub` | `tf.subtract` | Element-wise subtraction |
| `tf.Mul` | `tf.multiply` | Element-wise multiplication |
| `tf.MatMul` | `tf.matmul` | Matrix multiplication |
| `tf.Softmax` | `tf.nn.softmax` | Softmax activation |
| `tf.Relu` | `tf.nn.relu` | ReLU activation |
| `tf.Conv2D` | `tf.nn.conv2d` | 2D convolution |
| `tf.MaxPool` | `tf.nn.max_pool2d` | Max pooling |
| `tf.Reshape` | `tf.reshape` | Reshape tensor |
| `tf.Transpose` | `tf.transpose` | Transpose tensor |
| `tf.ConcatV2` | `tf.concat` | Concatenation |
| `tf.Split` | `tf.split` | Split tensor |
| `tf.ReduceSum` | `tf.reduce_sum` | Sum reduction |
| `tf.Shape` | `tf.shape` | Get tensor shape |
| `tf.ReadVariableOp` | `read_var` | Read resource variable |
| `tf.AssignVariableOp` | `assign_var` | Write resource variable |

### TF Type System

The TF dialect uses standard MLIR types plus TF-specific types:

| Type | Description |
|------|-------------|
| `tensor<NxMxf32>` | Static-shape tensor |
| `tensor<?xMxf32>` | Dynamic-shape tensor |
| `tf.string` | String type |
| `tf.resource<tensor<NxMxf32>>` | Resource variable type |
| `tf.variant` | Variant type |
| `tf.quint8` | Quantized unsigned 8-bit |

### TF Operation Attributes

TF operations carry TensorFlow-specific attributes:

```mlir
%0 = "tf.Conv2D"(%input, %filter) {
    strides = [1, 1, 1, 1],
    padding = "SAME",
    data_format = "NHWC",
    dilations = [1, 1, 1, 1]
} : (tensor<1x28x28x3xf32>, tensor<3x3x3x32xf32>) -> tensor<1x28x28x32xf32>
```

---

## tf_executor Dialect

The `tf_executor` dialect represents the TensorFlow graph execution model, including
control flow and synchronization.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `tf_executor.graph` | Top-level graph container |
| `tf_executor.island` | Wraps a set of operations that execute together |
| `tf_executor.fetch` | Collects output values |
| `tf_executor.sink` | Terminator for the graph |
| `tf_executor.switch` | Switch based on predicate |
| `tf_executor.merge` | Merge from multiple branches |
| `tf_executor.next_iteration` | Loop iteration |
| `tf_executor.loop_cond` | Loop condition |
| `tf_executor.control_trigger` | Control dependency trigger |
| `tf_executor.enter` | Enter a frame |
| `tf_executor.exit` | Exit a frame |

### Execution Model

The tf_executor dialect models TensorFlow's send/recv-based execution:

```mlir
tf_executor.graph {
  tf_executor.island {
    %0 = tf.Add(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    tf_executor.yield %0 : tensor<10xf32>
  }
  tf_executor.fetch %0 : tensor<10xf32>
  tf_executor.sink
}
```

### Island Execution

An `island` groups operations that execute as a single unit. Operations within
an island are executed sequentially, and their side effects are visible in order.

---

## tf_saved_model Dialect

The `tf_saved_model` dialect represents TensorFlow SavedModel concepts in MLIR.

### Key Operations and Attributes

| Element | Description |
|---------|-------------|
| `tf_saved_model.global_tensor` | Global tensor (variable) |
| `tf_saved_model.asset` | Asset file path |
| `session_initizer` | Session initialization op |
| `exported_names` | Attribute listing exported names |
| `initializers` | Attribute listing initializer functions |

### Saved Model Dialect Example

```mlir
module attributes {tf_saved_model.exported_names = ["serving_default"]} {
  tf_saved_model.global_tensor @var {
    value = dense<0.0> : tensor<f32>,
    type = tensor<f32>
  }

  func.func @serving_default(%arg0: tensor<10xf32> {tf_saved_model.index_path = ["input"]})
      -> (tensor<10xf32> {tf_saved_model.index_path = ["output"]}) {
    %0 = tf.ReadVariableOp(@var) : () -> tensor<f32>
    %1 = tf.Add(%arg0, %0) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    func.return %1 : tensor<10xf32>
  }
}
```

---

## MHLO Dialect

The MHLO (Meta HLO) dialect represents XLA HLO operations in MLIR. It is a direct
mapping of the XLA HLO instruction set to MLIR operations.

### Common MHLO Operations

| Operation | HLO Equivalent | Description |
|-----------|---------------|-------------|
| `mhlo.add` | `kAdd` | Element-wise addition |
| `mhlo.multiply` | `kMultiply` | Element-wise multiplication |
| `mhlo.dot` | `kDot` | Dot product / matrix multiplication |
| `mhlo.convolution` | `kConvolution` | Convolution |
| `mhlo.reduce` | `kReduce` | Reduction |
| `mhlo.broadcast` | `kBroadcast` | Broadcast |
| `mhlo.reshape` | `kReshape` | Reshape |
| `mhlo.transpose` | `kTranspose` | Transpose |
| `mhlo.concatenate` | `kConcatenate` | Concatenation |
| `mhlo.slice` | `kSlice` | Slicing |
| `mhlo.dynamic_slice` | `kDynamicSlice` | Dynamic slicing |
| `mhlo.scatter` | `kScatter` | Scatter update |
| `mhlo.gather` | `kGather` | Gather |
| `mhlo.sort` | `kSort` | Sort |
| `mhlo.while` | `kWhile` | While loop |
| `mhlo.if` / `mhlo.case` | `kConditional` | Conditional |
| `mhlo.fusion` | `kFusion` | Fusion group |
| `mhlo.custom_call` | `kCustomCall` | Custom call |
| `mhlo.all_reduce` | `kAllReduce` | All-reduce collective |
| `mhlo.all_gather` | `kAllGather` | All-gather collective |
| `mhlo.reduce_scatter` | `kReduceScatter` | Reduce-scatter |
| `mhlo.real_dynamic_slice` | dynamic | Dynamic slice with computed bounds |

### MHLO Example

```mlir
func.func @matmul_add(%arg0: tensor<10x20xf32>, %arg1: tensor<20x30xf32>, %arg2: tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = mhlo.dot(%arg0, %arg1) : (tensor<10x20xf32>, tensor<20x30xf32>) -> tensor<10x30xf32>
  %1 = mhlo.add(%0, %arg2) : (tensor<10x30xf32>, tensor<10x30xf32>) -> tensor<10x30xf32>
  func.return %1 : tensor<10x30xf32>
}
```

### MHLO Types

MHLO uses standard MLIR types:
- `tensor<NxMxDtype>` for tensor types
- `tuple<...>` for tuple types
- `token` for token types (side effect ordering)

---

## StableHLO Dialect

StableHLO is a stable, portable operation set for representing machine learning
workloads. It is derived from MHLO but guarantees backward compatibility.

### Key Differences from MHLO

| Aspect | MHLO | StableHLO |
|--------|------|-----------|
| Stability | Unstable (changes with XLA) | Stable (backward compatible) |
| Scope | XLA-specific | Cross-framework |
| Versioning | None | Semantic versioning |
| Portability | XLA backends | Any MLIR-compatible backend |
| Compatibility | No guarantees | Guaranteed backward compatibility |

### StableHLO Operations

StableHLO operations mirror MHLO with some differences:
- Stricter type constraints
- Well-defined semantics
- Portable across frameworks (JAX, PyTorch, etc.)

### TF to StableHLO Conversion

```cpp
// From: tensorflow/compiler/mlir/tensorflow_to_stablehlo/tf_to_stablehlo.h
// Converts TF dialect to StableHLO dialect
```

---

## TOSA Dialect

The TOSA (Tensor Operator Set Architecture) dialect provides a standardized set
of tensor operations suitable for targeting various hardware accelerators.

### TOSA Operations

| Category | Operations |
|----------|-----------|
| **Arithmetic** | `tosa.add`, `tosa.sub`, `tosa.mul`, `tosa.div` |
| **Activation** | `tosa.relu`, `tosa.sigmoid`, `tosa.tanh` |
| **Convolution** | `tosa.conv2d`, `tosa.depthwise_conv2d`, `tosa.conv3d` |
| **Pooling** | `tosa.max_pool2d`, `tosa.avg_pool2d` |
| **Reduction** | `tosa.reduce_sum`, `tosa.reduce_max`, `tosa.reduce_min` |
| **Comparison** | `tosa.equal`, `tosa.greater`, `tosa.greater_equal` |
| **Data Layout** | `tosa.reshape`, `tosa.transpose`, `tosa.slice`, `tosa.concat` |
| **Element-wise** | `tosa.abs`, `tosa.ceil`, `tosa.floor`, `tosa.exp`, `tosa.log` |
| **NN** | `tosa.fully_connected`, `tosa.matmul` |

### TOSA Benefits

- Hardware-agnostic representation
- Precise specification of each operation's semantics
- Suitable for NPUs (Neural Processing Units), DSPs, and accelerators
- Standardized by the MLIR community

### TF to TOSA Conversion

```cpp
// From: tensorflow/compiler/mlir/tosa/tf_passes.h
// Conversion passes from TF to TOSA
```

---

## TFLite Dialect

The TFLite dialect represents TensorFlow Lite operations in MLIR for conversion
and optimization.

### Common TFLite Operations

| Operation | Description |
|-----------|-------------|
| `tfl.add` | Element-wise addition |
| `tfl.sub` | Element-wise subtraction |
| `tfl.mul` | Element-wise multiplication |
| `tfl.fully_connected` | Fully connected layer |
| `tfl.conv_2d` | 2D convolution |
| `tfl.depthwise_conv_2d` | Depthwise convolution |
| `tfl.reshape` | Reshape |
| `tfl.softmax` | Softmax |
| `tfl.relu` | ReLU activation |
| `tfl.quantize` | Quantize to lower precision |
| `tfl.dequantize` | Dequantize to higher precision |

### TFLite Conversion in MLIR

```cpp
// From: tensorflow/compiler/mlir/lite/tf_tfl_passes.h
// Conversion passes from TF to TFLite
```

---

## Legalization Passes

Legalization passes convert operations from one dialect to another.

### TF to MHLO Legalization

Converts TF operations to MHLO (HLO) operations:

| TF Operation | MHLO Operation |
|-------------|----------------|
| `tf.Add` | `mhlo.add` |
| `tf.MatMul` | `mhlo.dot` |
| `tf.Conv2D` | `mhlo.convolution` |
| `tf.ReduceSum` | `mhlo.reduce` |
| `tf.Softmax` | `mhlo.softmax` (or exp+reduce+div) |
| `tf.Relu` | `mhlo.max(broadcast(0), x)` |
| `tf.Reshape` | `mhlo.reshape` |
| `tf.Transpose` | `mhlo.transpose` |

### TF to TFLite Legalization

| TF Operation | TFLite Operation |
|-------------|-----------------|
| `tf.Add` | `tfl.add` |
| `tf.MatMul` | `tfl.fully_connected` |
| `tf.Conv2D` | `tfl.conv_2d` |
| `tf.Relu` | `tfl.relu` |
| `tf.Reshape` | `tfl.reshape` |
| `tf.Softmax` | `tfl.softmax` |
| `tf.ConcatV2` | `tfl.concatenation` |

### TF to TOSA Legalization

```cpp
// From: tensorflow/compiler/mlir/tosa/tf_passes.h
// TF -> TOSA conversion passes
```

### MHLO to GPU Legalization

Converts MHLO operations to GPU-specific dialects:

| MHLO Operation | Target Dialect |
|---------------|----------------|
| `mhlo.fusion` | GPU kernel launch |
| `mhlo.dot` | GPU matmul kernel |
| `mhlo.convolution` | GPU conv kernel |
| `mhlo.reduce` | GPU reduce kernel |

---

## Lowering Pipeline

The lowering pipeline progressively converts from high-level to low-level IR.

### Standard Lowering Pipeline

```
TF Graph (GraphDef)
      |
      v
TF Executor Dialect (tf_executor)
      |
      v
TF Dialect (tf)
      |
      v
  +---+---+
  |       |
  v       v
MHLO     TFLite     (depending on target)
  |       |
  v       v
GPU/    FlatBuffer
NVVM/   (.tflite)
ROCDL/
LLVM
```

### TF to MHLO Pipeline

```
1. Import GraphDef to TF Executor dialect
2. Remove tf_executor wrapper (graph -> island -> ops)
3. Shape inference on TF dialect
4. Legalize TF -> MHLO (per-op conversion)
5. MHLO optimization (canonicalization, CSE)
6. Lower to backend (GPU/NVVM, CPU/LLVM, TPU)
```

### TF to TFLite Pipeline

```
1. Import GraphDef or SavedModel to TF dialect
2. Shape inference
3. Legalize TF -> TFLite
4. Quantization (optional)
5. Export to FlatBuffer (.tflite)
```

### Pipeline Configuration

```cpp
// From: tensorflow/compiler/mlir/mlir_graph_optimization_pass.h
// MLIR-based graph optimization pass
```

---

## MLIR Optimization Passes

MLIR provides built-in optimization passes that work across dialects.

### Canonicalization

Simplifies operations using their canonicalization patterns:
- Fold constant operations
- Simplify identity operations
- Merge equivalent operations

```mlir
// Before canonicalization:
%0 = "tf.Shape"(%input) : (tensor<10x20xf32>) -> tensor<2xi32>
%1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
%2 = "tf.Gather"(%0, %1) : (tensor<2xi32>, tensor<i32>) -> tensor<i32>

// After canonicalization (shape is known):
%2 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
```

### CSE (Common Subexpression Elimination)

Removes duplicate computations:

```mlir
// Before CSE:
%0 = tf.Add(%x, %y) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%1 = tf.Add(%x, %y) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%2 = tf.Mul(%0, %1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

// After CSE:
%0 = tf.Add(%x, %y) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
%2 = tf.Mul(%0, %0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
```

### Inliner

Inlines function calls to enable cross-function optimization:

```mlir
// Before inlining:
func.func @helper(%x: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tf.Relu(%x) : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
func.func @main(%x: tensor<10xf32>) -> tensor<10xf32> {
  %0 = func.call @helper(%x) : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// After inlining:
func.func @main(%x: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tf.Relu(%x) : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
```

### Shape Inference

Propagates shape information through the computation:

```cpp
// Shape inference pass propagates tensor shapes through the graph
// This enables downstream passes to make optimization decisions
```

### Pass Manager

MLIR provides a `PassManager` for organizing and running passes:

```cpp
mlir::PassManager pm(&context);
pm.addPass(mlir::createCanonicalizerPass());
pm.addPass(mlir::createCSEPass());
pm.addPass(mlir::TF::createTFShapeInferencePass());
pm.addPass(mlir::createConvertTFToMHLOPass());
```

---

## Op Definition

MLIR operations are defined using TableGen (`.td`) files that generate C++ code.

### TableGen Definition Example

```tablegen
// Example TF op definition (.td file)
def TF_AddOp : TF_Op<"Add", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Element-wise addition";
  let description = [{
    Returns x + y element-wise.
  }];
  let arguments = (ins
    TF_Tensor:$x,
    TF_Tensor:$y
  );
  let results = (outs
    TF_Tensor:$z
  );
  let hasCanonicalizer = 1;
}
```

### TFR Ops Definition

The TF Rewriter (TFR) dialect uses TableGen for defining decomposition patterns:

```tablegen
// From: tensorflow/compiler/mlir/tfr/ir/tfr_ops.td
// TFR operation definitions for pattern-based op decomposition
```

### Generated Code

From a `.td` definition, the TableGen tool generates:
- C++ operation class
- Parser/printer methods
- Verifier methods
- Canonicalization patterns

---

## Pattern Rewriting

MLIR provides powerful pattern rewriting infrastructure for transforming IR.

### DRR (Declarative Rewrite Rules)

DRR patterns are defined in TableGen:

```tablegen
// Pattern: tf.Add(x, 0) -> x
def AddZeroPattern : Pat<
  (TF_AddOp $x, (TF_ConstOp $zero_value)),
  (replaceWithValue $x)
>;
```

### C++ Rewrite Patterns

For more complex patterns, C++ rewrite patterns are used:

```cpp
class ConvertTFAddToMHLO : public OpConversionPattern<TF::AddOp> {
 public:
  using OpConversionPattern<TF::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(
        op, op.getType(), adaptor.getX(), adaptor.getY());
    return success();
  }
};
```

### Pattern Application

Patterns are collected and applied through a `RewritePatternSet`:

```cpp
RewritePatternSet patterns(&context);
patterns.add<ConvertTFAddToMHLO>(&context);
patterns.add<ConvertTFMatMulToMHLODot>(&context);
// ... more patterns

if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
  return failure();
}
```

---

## Conversion Framework

The MLIR conversion framework provides infrastructure for converting between dialects.

### TypeConverter

Converts types from the source to the target dialect:

```cpp
class TFToMHLOTypeConverter : public TypeConverter {
 public:
  TFToMHLOTypeConverter() {
    // Convert TF tensor types to MHLO tensor types
    addConversion([](TensorType type) { return type; });
    // Convert TF resource types
    addConversion([](TF::ResourceType type) {
      return type.getSubtypes().front();
    });
  }
};
```

### ConversionPattern

Base class for patterns that convert operations from one dialect to another:

```cpp
class ConversionPattern : public RewritePattern {
 public:
  // Override matchAndRewrite to perform conversion
  virtual LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const = 0;
};
```

### RewritePatternSet

Collects patterns for batch application:

```cpp
RewritePatternSet patterns(&context);
// Add all TF->MHLO conversion patterns
populateTFToMHLOConversionPatterns(patterns, &context);
```

### Conversion Target

Specifies which operations are legal after conversion:

```cpp
ConversionTarget target(getContext());
target.addLegalDialect<mhlo::MhloDialect>();
target.addLegalDialect<func::FuncDialect>();
target.addIllegalDialect<TF::TensorFlowDialect>();
// Some TF ops may be partially legal
target.addDynamicallyLegalOp<TF::WhileOp>([](TF::WhileOp op) {
  return isLegalWhileOp(op);
});
```

### Full Conversion Example

```cpp
LogicalResult convertTFToMHLO(ModuleOp module) {
  MLIRContext *context = module.getContext();
  TFToMHLOTypeConverter type_converter;

  ConversionTarget target(*context);
  target.addLegalDialect<mhlo::MhloDialect>();
  target.addIllegalDialect<TF::TensorFlowDialect>();

  RewritePatternSet patterns(context);
  populateTFToMHLOConversionPatterns(patterns, type_converter, context);

  return applyFullConversion(module, target, std::move(patterns));
}
```

---

## Dialect Registration

Dialects must be registered with the MLIR context before they can be used.

### Dialect Registry

```cpp
// From: tensorflow/compiler/mlir/register_common_dialects.h

// Inserts common Tensorflow dialects used for offline tools
void RegisterCommonToolingDialects(mlir::DialectRegistry& registry);
```

### All TensorFlow Dialects

```cpp
// From: tensorflow/compiler/mlir/tensorflow/dialect_registration.h

inline void RegisterAllTensorFlowDialects(DialectRegistry &registry) {
  registry
      .insert<mlir::arith::ArithDialect,
              mlir::func::FuncDialect,
              mlir::ml_program::MLProgramDialect,
              mlir::TF::TensorFlowDialect,
              mlir::tf_type::TFTypeDialect,
              mlir::cf::ControlFlowDialect,
              mlir::tf_device::TensorFlowDeviceDialect,
              mlir::tf_executor::TensorFlowExecutorDialect,
              mlir::tf_saved_model::TensorFlowSavedModelDialect,
              mlir::tfg::TFGraphDialect>();
}
```

### Dialect Insertion

The `insert` method registers a dialect:

```cpp
DialectRegistry registry;

// Register individual dialects
registry.insert<TF::TensorFlowDialect>();
registry.insert<mhlo::MhloDialect>();

// Or register all TF dialects at once
RegisterAllTensorFlowDialects(registry);

// Create context with registered dialects
MLIRContext context(registry);
```

### Dialect Initialization

Each dialect has an `initialize()` method that loads operations and types:

```cpp
class TensorFlowDialect : public Dialect {
 public:
  explicit TensorFlowDialect(MLIRContext *context);
  void initialize() override;
  static StringRef getDialectNamespace() { return "tf"; }
};
```

---

## TFR (TensorFlow Rewriter) Dialect

The TFR dialect provides a way to define TensorFlow operation decompositions
using MLIR patterns.

### TFR Operations

```cpp
// From: tensorflow/compiler/mlir/tfr/ir/tfr_ops.h
// TFR operation definitions
```

### Decomposition Patterns

```tablegen
// From: tensorflow/compiler/mlir/tfr/passes/decompose_patterns.td
// Declarative decomposition patterns
```

### TFR Integration

```cpp
// From: tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h
// Integration context for TFR decomposition
```

---

## TFRT (TensorFlow Runtime) Dialects

TFRT provides additional dialects for the TensorFlow Runtime.

### Runtime Fallback Dialects

```cpp
// From: tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h
// Fallback dialect for TFRT runtime
```

### GPU Operations

```cpp
// From: tensorflow/compiler/mlir/tfrt/ir/gpu_ops.h
// GPU-specific operations in TFRT
```

### Saved Model Support

```cpp
// From: tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h
// Saved model loading in TFRT
```

---

## MLIR Tools

### tf-mlir-translate

Translates between GraphDef and MLIR:

```bash
# Import GraphDef to MLIR
tf-mlir-translate -graphdef-to-mlir input.pb -o output.mlir

# Export MLIR to GraphDef
tf-mlir-translate -mlir-to-graphdef output.mlir -o output.pb
```

### TFLite Translation

```cpp
// From: tensorflow/compiler/mlir/lite/tf_tfl_translate_cl.h
// TFLite MLIR translation command-line flags
```

### FlatBuffer Translation

```cpp
// From: tensorflow/compiler/mlir/lite/flatbuffer_translate.h
// Translates between MLIR and TFLite FlatBuffer
```

### FlatBuffer Import/Export

```cpp
// From: tensorflow/compiler/mlir/lite/flatbuffer_import.h
// Imports FlatBuffer to MLIR
// From: tensorflow/compiler/mlir/lite/flatbuffer_export.h
// Exports MLIR to FlatBuffer
```

---

## MLIR Bridge

The MLIR bridge provides an alternative implementation of some TF optimization
passes using MLIR.

### Rollout Policy

```cpp
// From: tensorflow/compiler/mlir/tf2xla/mlir_bridge_rollout_policy.h
// Controls when the MLIR bridge is used instead of the legacy TF->XLA path
```

### Graph Optimization Pass

```cpp
// From: tensorflow/compiler/mlir/mlir_graph_optimization_pass.h
// MLIR-based graph optimization pass for TF
```

### MLIR Initialization

```cpp
// From: tensorflow/compiler/mlir/init_mlir.h
// Initializes MLIR infrastructure for TensorFlow
```

---

## Key Source Files Reference

| File Path | Description |
|-----------|-------------|
| `compiler/mlir/tensorflow/dialect_registration.h` | Dialect registration |
| `compiler/mlir/register_common_dialects.h` | Common dialect registration |
| `compiler/mlir/mlir_graph_optimization_pass.h` | MLIR graph optimization |
| `compiler/mlir/init_mlir.h` | MLIR initialization |
| `compiler/mlir/lite/tf_tfl_passes.h` | TF to TFLite passes |
| `compiler/mlir/lite/flatbuffer_translate.h` | FlatBuffer translation |
| `compiler/mlir/lite/flatbuffer_import.h` | FlatBuffer import |
| `compiler/mlir/lite/flatbuffer_export.h` | FlatBuffer export |
| `compiler/mlir/tosa/tf_passes.h` | TF to TOSA passes |
| `compiler/mlir/tosa/tf_tfl_passes.h` | TF/TFLite TOSA passes |
| `compiler/mlir/tfr/ir/tfr_ops.h` | TFR operation definitions |
| `compiler/mlir/tfr/passes/passes.h` | TFR passes |
| `compiler/mlir/tfrt/ir/tfrt_fallback.h` | TFRT fallback dialect |
| `compiler/mlir/tfrt/saved_model/saved_model.h` | TFRT saved model |
| `compiler/mlir/python/mlir.h` | Python MLIR bindings |
| `compiler/mlir/tools/tf_mlir_translate_cl.h` | Translation tool flags |
| `compiler/mlir/tensorflow_to_stablehlo/tf_to_stablehlo.h` | TF to StableHLO |

---

## Advanced MLIR Patterns

### Multi-Dialect Modules

MLIR modules can contain operations from multiple dialects simultaneously:

```mlir
module {
  func.func @mixed_computation(%arg0: tensor<10x20xf32>) -> tensor<10x20xf32> {
    // TF dialect operation
    %0 = "tf.Relu"(%arg0) : (tensor<10x20xf32>) -> tensor<10x20xf32>

    // MHLO dialect operation
    %1 = mhlo.dot(%0, %0) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>

    // Arith dialect operation
    %cst = arith.constant 0.0 : f32
    %2 = tensor.insert %cst into %1[%{{.*}}] : tensor<10x20xf32>

    func.return %1 : tensor<10x20xf32>
  }
}
```

### Op Verification

Each MLIR operation defines verification rules that are checked during IR construction:

```cpp
// Custom verification for a TF operation
LogicalResult verifyTFConv2D(Operation *op) {
  auto strides = op->getAttrOfType<DictionaryAttr>("strides");
  auto padding = op->getAttrOfType<StringAttr>("padding");
  // Verify attribute constraints
  if (strides.size() != 4) return failure();
  // ...
  return success();
}
```

### Region-Based Control Flow

MLIR uses regions to represent nested computation:

```mlir
// While loop as region-based control flow
%result = "mhlo.while"(%init) ({
  ^bb0(%val: tensor<10xf32>):
    %cond = "mhlo.compare"(%val, %threshold) {comparison_direction = #mhlo<LT>}
      : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
    mhlo.return %cond : tensor<10xi1>
  }, {
  ^bb0(%val: tensor<10xf32>):
    %next = mhlo.add(%val, %step) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    mhlo.return %next : tensor<10xf32>
  }) : (tensor<10xf32>) -> tensor<10xf32>
```

### Attribute Types in MLIR

MLIR supports various attribute types for operation metadata:

| Attribute Type | Example | Description |
|---------------|---------|-------------|
| `IntegerAttr` | `42 : i32` | Integer constant |
| `FloatAttr` | `3.14 : f32` | Floating-point constant |
| `StringAttr` | `"NHWC"` | String value |
| `ArrayAttr` | `[1, 2, 3]` | Array of attributes |
| `DictionaryAttr` | `{key = "value"}` | Dictionary of named attributes |
| `TypeAttr` | `tensor<10xf32>` | Type as attribute |
| `DenseElementsAttr` | `dense<1.0> : tensor<10xf32>` | Dense tensor constant |
| `SymbolRefAttr` | `@function_name` | Symbol reference |

### Type System

MLIR types used in TensorFlow dialects:

```mlir
// Standard MLIR types
tensor<10x20xf32>           // Static shape tensor
tensor<?x20xf32>            // Dynamic first dimension
tensor<*xf32>               // Unranked tensor
f32, f64, f16, bf16         // Floating-point types
i1, i8, i16, i32, i64      // Integer types

// TF-specific types
!tf.string                   // String type
!tf.resource<tensor<10xf32>> // Resource variable
!tf.variant                   // Variant type
!tf.quint8                    // Quantized uint8

// Ranked tensor type with encoding
tensor<10x20xf32, #encoding>  // With custom encoding attribute
```

### Dialect Extension

New dialects can be added to extend TensorFlow's MLIR infrastructure:

```cpp
// Define a new dialect
class MyDialect : public Dialect {
 public:
  explicit MyDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "my_dialect"; }
};

// Register the dialect
static DialectRegistration<MyDialect> registration;
```

### MLIR Pass Pipeline Construction

Complex pass pipelines can be constructed programmatically:

```cpp
void buildTFLiteConversionPipeline(OpPassManager &pm) {
  // TF Executor -> TF dialect
  pm.addPass(createTFExecutorToTFDialectPass());

  // Shape inference
  pm.addPass(createTFShapeInferencePass());

  // TF -> TFLite legalization
  pm.addNestedPass<func::FuncOp>(createConvertTFToTFLitePass());

  // Post-legalization optimization
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Quantization (if configured)
  pm.addPass(createTFLiteQuantizationPass());
}
```

### MLIR Debug Actions

MLIR provides a debug action framework for tracing compilation:

```cpp
// Debug action for tracing pass execution
mlir::DebugAction<Pass *> pass_action("mlir-pass", "Pass execution");
```

This enables runtime tracing of which passes are executed and their effects.

### Cross-Platform Considerations

MLIR dialects enable cross-platform compilation by abstracting platform differences:

```
TF Dialect (platform-independent)
      |
      +---> MHLO -> GPU (NVIDIA/AMD)
      |
      +---> MHLO -> CPU (x86/ARM)
      |
      +---> TOSA -> NPU/DSP
      |
      +---> TFLite -> Mobile/IoT
      |
      +---> StableHLO -> Portable across frameworks
```

Each target uses a different lowering pipeline from the same source TF dialect IR.
