# XLA's MLIR Integration

This document provides comprehensive documentation about XLA's integration with MLIR (Multi-Level Intermediate Representation), covering the dialects used, the infrastructure, and the optimization pipeline.

## Table of Contents

- [Overview](#overview)
- [MLIR-HLO Dialects](#mlir-hlo-dialects)
- [MLIR Infrastructure in XLA](#mlir-infrastructure-in-xla)
- [TableGen Definitions](#tablegen-definitions)
- [MLIR-Based Optimization Passes](#mlir-based-optimization-passes)

## Overview

MLIR is a framework for building extensible compilers. XLA uses MLIR extensively for:

1. **Representing computation graphs**: HLO operations are modeled as MLIR dialects.
2. **Code generation**: MLIR's lowering infrastructure is used to generate target-specific code.
3. **Optimization**: MLIR's transformation framework provides powerful optimization capabilities.
4. **Interoperability**: MLIR provides a common interchange format between different tools and frameworks.

The integration of MLIR into XLA has evolved over time, with MLIR increasingly becoming the primary representation for XLA's internal compilation pipeline, especially for GPU code generation.

## MLIR-HLO Dialects

XLA uses several MLIR dialects to represent different levels of abstraction in the compilation stack.

### StableHLO Dialect

StableHLO is the stable, publicly-facing MLIR dialect that represents HLO operations. It is designed for portability and stability across different versions of XLA.

#### Purpose

- **Portable representation**: StableHLO provides a stable serialization format that can be consumed by different XLA versions and implementations.
- **Framework integration**: Frameworks like JAX can emit StableHLO directly, which is then compiled by XLA or any StableHLO-compatible compiler.
- **Interoperability**: StableHLO serves as a common interchange format between ML frameworks and compilers.

#### Key Properties

- **Versioned**: Each StableHLO operation has a version number, and the dialect follows a formal versioning and deprecation policy.
- **Backward compatible**: Programs compiled with older StableHLO versions should work with newer versions.
- **Well-specified**: The semantics of each operation are precisely defined in the StableHLO specification.

#### Example Operations

```mlir
// StableHLO addition
%result = stablehlo.add %lhs, %rhs : tensor<4xf32>

// StableHLO matmul (dot general)
%result = stablehlo.dot_general %lhs, %rhs,
    contracting_dims = [1] x [0],
    batching_dims = [] x [] : tensor<4x8xf32> * tensor<8x4xf32> -> tensor<4x4xf32>

// StableHLO convolution
%result = stablehlo.convolution(%input, %filter)
    dim_sizes = [4, 4, 1, 8]
    window_strides = [1, 1]
    padding = [[0, 0], [0, 0]]
    lhs_dilation = [1, 1]
    rhs_dilation = [1, 1]
    window_reversal = [false, false]
    dimension_numbers = {
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    }
    : tensor<1x4x4x1xf32> * tensor<3x3x1x8xf32> -> tensor<1x4x4x8xf32>

// StableHLO reduce
%result = stablehlo.reduce(%input, %init_value) applies stablehlo.add
    dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
```

#### StableHLO Location

The StableHLO dialect is maintained in the `stablehlo` repository and integrated into XLA. Key source locations:

```
stablehlo/
  stablehlo/dialect/
    StableHLOOps.td        # Operation definitions
    StableHLOTypes.td      # Type definitions
    StableHLOAttrs.td      # Attribute definitions
  stablehlo/transforms/
    stablehlo_legalize_to_hlo.cc  # Conversion to MHLO
```

### MHLO Dialect (Internal)

MHLO (Multi-dialect HLO) is XLA's internal MLIR dialect. It represents HLO operations in a form that is optimized for XLA's internal compilation pipeline.

#### Purpose

- **Internal representation**: MHLO is the working representation inside XLA's compiler.
- **Optimization target**: MHLO operations are the subject of XLA's MLIR-based optimization passes.
- **Lowering source**: MHLO is lowered to target-specific dialects (e.g., LLVM IR for CPU, GPU dialects for GPU).

#### Relationship to StableHLO

The relationship between StableHLO and MHLO is:

```
Framework (JAX, etc.)
        |
        v
    StableHLO    (stable, versioned, portable)
        |
        | stablehlo-legalize-to-mhlo
        v
      MHLO        (internal, evolving, optimized)
        |
        | lowering passes
        v
   Target-specific dialects
```

StableHLO is converted to MHLO early in the compilation pipeline. This conversion may involve:
- Adding internal metadata
- Legalizing operations that have different internal representations
- Applying default attributes

#### MHLO-Specific Operations

MHLO includes some operations that are not present in StableHLO:

- **Internal fusion operations**: Representing fused computation blocks.
- **Custom call markers**: Operations that map to XLA custom calls.
- **Backend-specific hints**: Attributes that guide backend-specific optimizations.

### CHLO Dialect (Client HLO)

CHLO (Client HLO) is a higher-level dialect that represents operations closer to the user-facing API level. It includes operations that are not directly expressible in HLO but can be lowered to HLO operations.

#### Purpose

- **High-level representation**: CHLO captures the user's intent at a higher level than HLO.
- **Client-side operations**: Operations that are typically performed on the client side before lowering to HLO.

#### Example CHLO Operations

```mlir
// CHLO bessel function (lowered to HLO polynomial approximation)
%result = chlo.bessel_j0 %arg : tensor<4xf32> -> tensor<4xf32>

// CHLO next-after (IEEE 754 next representable value)
%result = chlo.next_after %x, %y : tensor<f32> -> tensor<f32>

// CHLO digamma function
%result = chlo.digamma %arg : tensor<4xf32> -> tensor<4xf32>

// CHLO dynamic slice with computed indices
%result = chlo.dynamic_slice %operand, %start_indices
    slice_sizes = [2, 3] : tensor<4x6xf32>, tensor<2xi64> -> tensor<2x3xf32>
```

#### Lowering Path

```
CHLO
  |
  | chlo-legalize-to-stablehlo
  v
StableHLO
  |
  | stablehlo-legalize-to-mhlo
  v
MHLO
```

### Conversion Between Dialects

The conversion between dialects follows a well-defined path:

1. **CHLO to StableHLO**: High-level operations (like bessel functions) are lowered to sequences of standard HLO operations (polynomial approximations, series expansions).

2. **StableHLO to MHLO**: The stable representation is converted to the internal representation. This is mostly a one-to-one mapping but may include:
   - Adding internal attributes
   - Renaming operations for consistency
   - Inserting bookkeeping operations

3. **MHLO to Target**: The internal representation is lowered to target-specific dialects:
   - **CPU**: MHLO -> Linalg -> Standard -> LLVM IR
   - **GPU**: MHLO -> GPU dialect -> LLVM IR (NVVM/ROCL) -> PTX/AMDGCN

## MLIR Infrastructure in XLA

### MLIR Code Generation Pipeline

XLA uses MLIR for code generation, particularly for the GPU backend. The MLIR-based code generation pipeline is:

```
HLO (XLA internal representation)
  |
  | ConvertHloToMlir
  v
MHLO MLIR Module
  |
  | MHLO optimization passes
  v
Optimized MHLO
  |
  | Lower to Linalg (via MHLO -> Linalg lowering)
  v
Linalg on Tensors
  |
  | Linalg bufferization
  v
Linalg on Buffers (MemRef)
  |
  | Lower to target-specific dialects
  v
Target Code
```

#### GPU Code Generation Pipeline

For the GPU backend, the pipeline is more specific:

```
HLO Module
  |
  | HLO -> MHLO conversion
  v
MHLO Module
  |
  | GpuFusionPipeline
  |   - Fusion clustering
  |   - Fusion region formation
  v
Fused MHLO Module
  |
  | MHLO -> GPU lowering
  v
GPU Module (gpu.func with tensor operations)
  |
  | Tensor bufferization
  v
GPU Module (gpu.func with memref operations)
  |
  | Math -> LLVM approximation (for functions like exp, log)
  | Arith -> LLVM conversion
  | SCF -> LLVM conversion (loops to branches)
  | GPU -> LLVM conversion (kernel metadata)
  v
LLVM IR Module
  |
  | LLVM optimization passes
  v
Optimized LLVM IR
  |
  | Backend code generation (NVPTX or AMDGPU target)
  v
PTX or AMDGCN
```

### xla_gpu MLIR Dialect

XLA includes a custom MLIR dialect for GPU-specific operations and abstractions. This dialect sits between the generic HLO operations and the target-specific GPU code.

#### Key Components

- **Fusion operations**: Representing fused computation blocks that map to single GPU kernels.
- **Tiling and distribution**: Operations for tiling computations across GPU thread blocks.
- **Shared memory management**: Operations for managing shared memory in GPU kernels.

### Standard MLIR Dialects Used

XLA leverages several standard MLIR dialects:

#### Tensor Dialect

The `tensor` dialect provides operations for tensor-level computations:

```mlir
// Tensor extract (read a scalar from a tensor)
%val = tensor.extract %tensor[%idx] : tensor<10xf32>

// Tensor insert (write a scalar into a tensor)
%new_tensor = tensor.insert %val into %tensor[%idx] : tensor<10xf32>

// Tensor empty (allocate an uninitialized tensor)
%tensor = tensor.empty() : tensor<4x4xf32>
```

#### Arith Dialect

The `arith` dialect provides standard arithmetic operations:

```mlir
// Integer arithmetic
%sum = arith.addi %a, %b : i32
%diff = arith.subi %a, %b : i32
%prod = arith.muli %a, %b : i32

// Floating-point arithmetic
%fsum = arith.addf %a, %b : f32
%fprod = arith.mulf %a, %b : f32

// Comparisons
%cmp = arith.cmpi eq, %a, %b : i32
%fcmp = arith.cmpf oeq, %a, %b : f32

// Type conversions
%float_val = arith.sitofp %int_val : i32 to f32
%int_val = arith.fptosi %float_val : f32 to i32
```

#### Math Dialect

The `math` dialect provides mathematical functions:

```mlir
// Transcendental functions
%exp = math.exp %arg : tensor<4xf32>
%log = math.log %arg : tensor<4xf32>
%sqrt = math.sqrt %arg : tensor<4xf32>
%rsqrt = math.rsqrt %arg : tensor<4xf32>
%sin = math.sin %arg : tensor<4xf32>
%cos = math.cos %arg : tensor<4xf32>

// Other functions
%pow = math.pow %base, %exp : tensor<4xf32>
%erf = math.erf %arg : tensor<4xf32>
```

#### SCF (Structured Control Flow) Dialect

The `scf` dialect provides structured control flow operations:

```mlir
// For loop (parallelizable)
%result = scf.for %iv = %lb to %ub step %step
    init(%init) -> (tensor<4xf32>) {
  %val = ... compute ...
  scf.yield %val : tensor<4xf32>
}

// If-then-else
%result = scf.if %cond -> (tensor<4xf32>) {
  %val = ... then branch ...
  scf.yield %val : tensor<4xf32>
} else {
  %val = ... else branch ...
  scf.yield %val : tensor<4xf32>
}

// While loop
%result = scf.while(%arg = %init) : (tensor<4xf32>) -> (tensor<4xf32>) {
  %cond = ... compute condition ...
  scf.condition(%cond) %arg : tensor<4xf32>
} do {
^bb0(%arg: tensor<4xf32>):
  %next = ... compute next value ...
  scf.yield %next : tensor<4xf32>
}
```

#### GPU Dialect

The `gpu` dialect provides GPU-specific operations and abstractions:

```mlir
// GPU kernel launch
gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gx, %grid_y = %gy, %grid_z = %gz)
           threads(%tx, %ty, %tz) in (%block_x = %bx, %block_y = %by, %block_z = %bz)
           args(%arg0 = %input, %arg1 = %output) {
  // Kernel body
  %idx = arith.addi %bx, %tx : index
  %val = memref.load %arg0[%idx] : memref<1024xf32>
  memref.store %val, %arg1[%idx] : memref<1024xf32>
  gpu.terminator
}

// GPU barrier synchronization
gpu.barrier

// Shared memory operations
%shared = gpu.dynamic_shared_memory() : !gpu.dynamic_shared_memory<f32>
```

#### Vector Dialect

The `vector` dialect provides SIMD/SIMT vector operations:

```mlir
// Vector load/store
%vec = vector.load %mem[%idx] : memref<1024xf32>, vector<4xf32>
vector.store %vec, %mem[%idx] : memref<1024xf32>, vector<4xf32>

// Vector arithmetic
%sum = arith.addf %a, %b : vector<4xf32>

// Vector reduction
%reduced = vector.reduction <add>, %vec : vector<4xf32> into f32

// Vector shuffle
%shuffled = vector.shuffle %a, %b [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
```

## TableGen Definitions

MLIR uses TableGen to define operations, types, and attributes. XLA follows this convention for its dialects.

### Operation Definitions (.td files)

Operations are defined in `.td` (TableGen) files, which generate C++ code at build time.

#### StableHLO Operation Example

```tablegen
// StableHLO_Add operation definition
def StableHLO_AddOp : StableHLO_Op<"add", [Pure, Elementwise, UniformTypeBroadcastable]> {
  let summary = "Addition operation";
  let description = [{
    Performs element-wise addition of two tensors.
  }];

  let arguments = (ins
    StableHLO_Tensor:$lhs,
    StableHLO_Tensor:$rhs
  );

  let results = (outs
    StableHLO_Tensor:$result
  );

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

#### MHLO Operation Example

```tablegen
// MHLO_FusionOp definition
def MHLO_FusionOp : MHLO_Op<"fusion"> {
  let summary = "Fusion operation";
  let description = [{
    Represents a group of operations that should be compiled into a single
    kernel. Contains a region with the fused operations.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "kLoop">:$fusion_kind
  );

  let regions = (region AnyRegion:$region);

  let results = (outs
    Variadic<HLO_Tensor>:$results
  );

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}
```

#### Type Definitions

```tablegen
// StableHLO token type
def StableHLO_Token : HLO_TokenType<"token"> {
  let summary = "Token type for modeling side effects";
}

// StableHLO bounded dynamism
def StableHLO_BoundedTensorType : HLO_Type<"BoundedTensor", "Type"> {
  let summary = "Tensor type with bounded dynamic dimensions";
  let parameters = (ins
    "ArrayRef<int64_t>":$shape,
    "Type":$elementType,
    "ArrayRef<int64_t>":$bounds
  );
}
```

### Type Definitions

XLA defines several MLIR types for representing tensors, layouts, and other concepts:

```tablegen
// Ranked tensor type extension
def HLO_Tensor : TypeDef<HLO_Dialect, "Tensor"> {
  let summary = "Ranked tensor type";
  let parameters = (ins
    "ArrayRef<int64_t>":$shape,
    "Type":$element_type
  );
}
```

### Attribute Definitions

Attributes carry compile-time information about operations:

```tablegen
// Convolution dimension numbers attribute
def StableHLO_ConvDimensionNumbersAttr :
    HLO_Attr<"ConvDimensionNumbers", "stablehlo.conv_dimension_numbers"> {
  let summary = "Attribute encoding convolution dimension numbering";
  let parameters = (ins
    "int64_t":$input_batch_dimension,
    "int64_t":$input_feature_dimension,
    "ArrayRef<int64_t>":$input_spatial_dimensions,
    "int64_t":$kernel_input_feature_dimension,
    "int64_t":$kernel_output_feature_dimension,
    "ArrayRef<int64_t>":$kernel_spatial_dimensions,
    "int64_t":$output_batch_dimension,
    "int64_t":$output_feature_dimension,
    "ArrayRef<int64_t>":$output_spatial_dimensions
  );
}

// Padding attribute
def StableHLO_PaddingAttr : HLO_Attr<"Padding", "stablehlo.padding"> {
  let summary = "Padding type attribute";
  let enumDecodeMethod = [{
    return ::mlir::symbolizeEnum<$0>();
  }];
}
```

### Build-Time Generation

The TableGen definitions are processed during the build to generate:

1. **C++ operation classes**: Each operation becomes a C++ class with builder methods, accessors, and verification logic.
2. **Operation documentation**: Auto-generated documentation from the TableGen descriptions.
3. **Serialization/deserialization code**: Code for reading and writing operations in MLIR bytecode format.
4. **Verification logic**: Auto-generated verifiers that check operation invariants.

The build targets for TableGen processing are typically:

```python
# Bazel build rule for generating C++ from TableGen
gentbl_cc_library(
    name = "StableHLOOpsIncGen",
    tbl_outs = [
        ("-gen-op-decls", "StableHLOOps.h.inc"),
        ("-gen-op-defs", "StableHLOOps.cpp.inc"),
        ("-gen-dialect-decls", "StableHLOOpsDialect.h.inc"),
        ("-gen-dialect-defs", "StableHLOOpsDialect.cpp.inc"),
        ("-gen-typedef-decls", "StableHLOOpsTypes.h.inc"),
        ("-gen-typedef-defs", "StableHLOOpsTypes.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StableHLOOps.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TensorOpsTdFiles",
    ],
)
```

## MLIR-Based Optimization Passes

### MLIR-to-MLIR Rewrites

XLA implements many optimization passes as MLIR pattern rewrites. These follow MLIR's rewrite framework.

#### Pattern Definition

```cpp
// Example: Fold transpose into dot general
struct FoldTransposeIntoDotGeneral : public OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp dot,
                                 PatternRewriter& rewriter) const override {
    // Check if one of the operands is a transpose
    auto lhs_transpose = dot.getLhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!lhs_transpose) return failure();

    // Modify the dot_general's dimension numbers instead of materializing the transpose
    auto dim_numbers = dot.getDotDimensionNumbersAttr();
    // ... modify dimension numbers to account for transpose ...

    rewriter.replaceOpWithNewOp<mlir::stablehlo::DotGeneralOp>(
        dot, dot.getType(), lhs_transpose.getOperand(), dot.getRhs(),
        /* new dimension numbers */);
    return success();
  }
};
```

#### Pass Registration

```cpp
// Define a pass using the generated patterns
void populateStableHLOFoldingPatterns(RewritePatternSet& patterns) {
  patterns.add<FoldTransposeIntoDotGeneral,
               FoldReshapeIntoDotGeneral,
               FoldBroadcastIntoElementwise>(patterns.getContext());
}

// Register as a pass
LogicalResult StableHLOFoldingPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateStableHLOFoldingPatterns(patterns);
  return applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
```

### Key MLIR-Based Passes in XLA

#### CSE (Common Subexpression Elimination)

Eliminates redundant computations by identifying and merging identical operations:

```mlir
// Before CSE:
%a = stablehlo.add %x, %y : tensor<4xf32>
%b = stablehlo.add %x, %y : tensor<4xf32>  // identical to %a
%c = stablehlo.multiply %a, %z : tensor<4xf32>
%d = stablehlo.multiply %b, %z : tensor<4xf32>

// After CSE:
%a = stablehlo.add %x, %y : tensor<4xf32>
%c = stablehlo.multiply %a, %z : tensor<4xf32>
```

#### Canonicalization

Simplifies operations to canonical forms:

```mlir
// Before: multiply by 1
%one = stablehlo.constant dense<1.0> : tensor<f32>
%result = stablehlo.multiply %x, %one : tensor<4xf32>

// After: identity
%result = %x : tensor<4xf32>
```

#### Fusion

Combines multiple operations into fused regions that execute as a single kernel:

```mlir
// Before fusion:
%a = stablehlo.exp %x : tensor<4xf32>
%b = stablehlo.add %a, %y : tensor<4xf32>
%c = stablehlo.multiply %b, %z : tensor<4xf32>

// After fusion (loop fusion):
%c = stablehlo.fusion() ({
  %x_ = stablehlo.argument %x : tensor<4xf32>
  %y_ = stablehlo.argument %y : tensor<4xf32>
  %z_ = stablehlo.argument %z : tensor<4xf32>
  %a = stablehlo.exp %x_ : tensor<4xf32>
  %b = stablehlo.add %a, %y_ : tensor<4xf32>
  %c = stablehlo.multiply %b, %z_ : tensor<4xf32>
  stablehlo.result %c : tensor<4xf32>
}) {fusion_kind = "kLoop"} : tensor<4xf32>
```

#### Shape Refinement

Replaces dynamic shapes with static shapes when possible:

```mlir
// Before:
%x : tensor<?xf32>  // dynamic dimension
%y = stablehlo.add %x, %z : tensor<?xf32>

// After (if we know the dimension is 4 at compile time):
%x : tensor<4xf32>
%y = stablehlo.add %x, %z : tensor<4xf32>
```

### Lowering Pipeline

The lowering pipeline converts operations from higher-level dialects to lower-level ones:

#### MHLO to Linalg Lowering

This pass converts MHLO operations to Linalg operations, which are a more general-purpose MLIR dialect for linear algebra:

```mlir
// MHLO:
%result = stablehlo.add %lhs, %rhs : tensor<4x8xf32>

// After lowering to Linalg:
%result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%init : tensor<4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  } -> tensor<4x8xf32>
```

#### Linalg to GPU Lowering

Converts Linalg operations to GPU kernel launches:

```mlir
// Linalg generic op -> GPU parallel loop
%result = gpu.launch blocks(%bx, %by, %bz) in (%gx, %gy, %gz)
           threads(%tx, %ty, %tz) in (%bxx, %byy, %bzz) {
  // Compute global index
  %idx = arith.addi %bx, %tx : index
  // Bounds check
  %in_bounds = arith.cmpi slt, %idx, %size : index
  scf.if %in_bounds {
    // Load, compute, store
    %a = memref.load %lhs[%idx] : memref<1024xf32>
    %b = memref.load %rhs[%idx] : memref<1024xf32>
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %result[%idx] : memref<1024xf32>
  }
  gpu.terminator
}
```

#### Bufferization

Converts tensor-level operations to buffer-level (memref) operations:

```mlir
// Before bufferization (tensor level):
%result = stablehlo.add %lhs, %rhs : tensor<4x8xf32>

// After bufferization (buffer level):
%result = bufferization.alloc_tensor() : memref<4x8xf32>
linalg.generic ... ins(%lhs, %rhs : memref<4x8xf32>, memref<4x8xf32>)
                   outs(%result : memref<4x8xf32>) {
  // Same body, but operates on memref
}
```

### Debugging MLIR in XLA

#### Dumping MLIR

Use the following flags to dump MLIR at various stages:

```bash
# Dump MLIR after each pass
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python my_program.py

# The dump directory will contain files like:
# module_0000_before_stablehlo_legalize.mlir
# module_0001_after_stablehlo_legalize.mlir
# module_0002_before_mhlo_optimizations.mlir
# ...
```

#### Using mlir-opt

The `mlir-opt` tool can be used to test individual passes:

```bash
# Run a specific pass on an MLIR file
mlir-opt --stablehlo-canonicalize input.mlir

# Run multiple passes
mlir-opt --stablehlo-canonicalize --stablehlo-cse input.mlir

# Dump the IR after each pass
mlir-opt --debug --stablehlo-canonicalize input.mlir
```

#### MLIR Diagnostic Hooks

```cpp
// Register diagnostic handlers to intercept MLIR errors
context.getDiagEngine().registerHandler([&](Diagnostic& diag) {
  llvm::errs() << "MLIR Diagnostic: " << diag << "\n";
  return Diagnostic::Success();
});
```
