# MLIR Toy Tutorial

## Overview

The Toy tutorial is a comprehensive guide that walks through building a compiler for a tensor-based "Toy" language using MLIR. It covers all stages from language design through MLIR dialect creation, optimization, lowering, and code generation.

### Toy Language

Toy is a tensor-based language with the following characteristics:
- Tensors of rank <= 2
- Only 64-bit floating point type (double)
- Immutable values (every operation returns a newly allocated value)
- Automatic deallocation
- Generic functions with type inference
- Builtins: `transpose()` and `print()`

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
}
```

## Chapter 1: Toy Language and AST

### Language Features

```toy
# Variable with shape inference
var a = [[1, 2, 3], [4, 5, 6]];

# Explicit shape declaration
var b<2, 3> = [1, 2, 3, 4, 5, 6];

# Generic functions (unranked parameters)
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

# Functions specialize at call sites
var c = multiply_transpose(a, b);  # Specializes with <2, 3>
```

### AST Structure

The AST has these node types:
- `Module` - Top-level container
- `Function` - Function definition with proto, params, and body
- `Block` - Sequence of statements
- `VarDecl` - Variable declaration
- `Return` - Return statement
- `BinOp` - Binary operation
- `Call` - Function call (including builtins)
- `Literal` - Tensor literal
- `Var` - Variable reference

### Implementation

- Lexer: `examples/toy/Ch1/include/toy/Lexer.h`
- Parser: `examples/toy/Ch1/include/toy/Parser.h` (recursive descent)

### Running

```bash
toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast
```

## Chapter 2: Emitting Basic MLIR

### MLIR Operation Anatomy

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true}
    : (tensor<2x3xf64>) -> tensor<3x2xf64>
    loc("example/file/path":12:1)
```

Components:
- `%t_tensor` - Result name (SSA value)
- `"toy.transpose"` - Operation name (dialect.operation)
- `(%tensor)` - Operands (SSA values)
- `{inplace = true}` - Attributes (constant metadata)
- `(tensor<2x3xf64>) -> tensor<3x2xf64>` - Type signature
- `loc(...)` - Source location (mandatory in MLIR)

### Defining the Toy Dialect

```c++
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "toy"; }
  void initialize();
};
```

ODS definition:

```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A high-level dialect for analyzing and optimizing the Toy language";
  let cppNamespace = "toy";
}
```

Loading the dialect:

```c++
context.loadDialect<ToyDialect>();
```

### Defining Operations with ODS

Base class:

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

Constant operation:

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value.
  }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build(builder, result, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];
}
```

### Custom Assembly Format

```tablegen
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

C++ printer/parser:

```c++
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();
  return mlir::success();
}
```

### Op vs Operation

- `Operation` - Generic base class for all operations (opaque)
- `Op<ConcreteType>` - Typed wrapper with specific accessors (smart pointer around `Operation*`)

```c++
// Cast from generic to specific
ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);
```

### Generated IR

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

## Chapter 3: High-Level Language-Specific Optimization

### Pattern Rewriting with C++

Eliminate redundant transposes: `transpose(transpose(X)) -> X`

```c++
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  llvm::LogicalResult matchAndRewrite(
      TransposeOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();
    if (!transposeInputOp)
      return failure();
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

Registering patterns:

```c++
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

Adding the `Pure` trait enables dead code elimination:

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {...}
```

Running optimizations:

```c++
mlir::PassManager pm(module->getName());
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

### DRR (Declarative Rewrite Rules)

Reshape(Reshape(x)) = Reshape(x):

```tablegen
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

Redundant reshape (identical types):

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

Fold constant reshape:

```tablegen
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

### Result

Before optimization:
```mlir
%0 = toy.constant dense<[1.0, 2.0]> : tensor<2xf64>
%1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
%2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
%3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
toy.print %3 : tensor<2x1xf64>
```

After optimization:
```mlir
%0 = toy.constant dense<[[1.0], [2.0]]> : tensor<2x1xf64>
toy.print %0 : tensor<2x1xf64>
```

## Chapter 4: Generic Transformation with Interfaces

### Inlining via Dialect Interface

```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final { return true; }
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final { return true; }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final { return true; }
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return CastOp::create(builder, conversionLoc, resultType, input);
  }
};
```

### Call Interface

```tablegen
def FuncOp : Toy_Op<"func",
    [FunctionOpInterface, IsolatedFromAbove]> { ... }

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<F64Tensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
}
```

### Cast Operation for Type Mismatches

```tablegen
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape]
  > {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

### Shape Inference via Operation Interface

Define the interface:

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types.
  }];
  let methods = [
    InterfaceMethod<"Infer and set the output shape.",
                    "void", "inferShapes">
  ];
}
```

Add to operations:

```tablegen
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> { ... }
```

Implement:

```c++
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
```

### Shape Inference Pass

```c++
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp function = getOperation();
    // 1. Build worklist of ops with dynamic shapes
    // 2. Iterate: find ready op, infer shapes
    // 3. Success if worklist empty
  }
};
```

### Pipeline

```c++
mlir::PassManager pm(module->getName());
pm.addPass(mlir::createInlinerPass());
pm.addPass(mlir::createShapeInferencePass());
pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

## Chapter 5: Partial Lowering to Affine

### Dialect Conversion Framework

Three components:
1. **ConversionTarget** - Specifies legal/illegal operations
2. **Rewrite Patterns** - Convert illegal to legal operations
3. **Type Converter** (optional) - Maps types during conversion

### Conversion Target

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), llvm::IsaPred<TensorType>);
  });
}
```

### Conversion Patterns

```c++
struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter,
                   [&](OpBuilder &builder, ValueRange loopIvs) {
                     Value input = adaptor.getInput();
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
                   });
    return success();
  }
};
```

### Partial Lowering

```c++
if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
  signalPassFailure();
```

### Tensor to MemRef

Update PrintOp to accept both tensor and memref:

```tablegen
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

### Affine Optimization

```c++
pm.addPass(mlir::createLoopFusionPass());
pm.addPass(mlir::createAffineScalarReplacementPass());
```

Before (redundant loads):
```mlir
affine.for %arg0 = 0 to 3 {
  affine.for %arg1 = 0 to 2 {
    %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
    %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
    %5 = arith.mulf %3, %4 : f64
    affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
  }
}
```

After (fused, simplified):
```mlir
affine.for %arg0 = 0 to 3 {
  affine.for %arg1 = 0 to 2 {
    %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>
    %3 = arith.mulf %2, %2 : f64
    affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
  }
}
```

## Chapter 6: Lowering to LLVM and Code Generation

### Lowering toy.print

Lower to a loop nest calling `printf`:

```c++
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(module.getContext(), "printf");
  auto llvmI32Ty = IntegerType::get(module.getContext(), 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(module.getContext(), 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(module.getContext(), "printf");
}
```

### Conversion Setup

```c++
// Target: LLVM dialect
mlir::ConversionTarget target(getContext());
target.addLegalDialect<mlir::LLVMDialect>();
target.addLegalOp<mlir::ModuleOp>();

// Type converter
LLVMTypeConverter typeConverter(&getContext());

// Patterns (transitive lowering)
mlir::RewritePatternSet patterns(&getContext());
mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());
patterns.add<PrintOpLowering>(&getContext());

// Full conversion
if (mlir::failed(mlir::applyFullConversion(module, target, patterns)))
  signalPassFailure();
```

### Emitting LLVM IR

```c++
llvm::LLVMContext llvmContext;
auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
```

### JIT Execution

```c++
int runJit(mlir::ModuleOp module) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(
      EnableOpt ? 3 : 0, 0, nullptr);
  auto maybeEngine = mlir::ExecutionEngine::create(module,
      nullptr, optPipeline);
  auto &engine = maybeEngine.get();
  auto invocationResult = engine->invoke("main");
  return 0;
}
```

### Running

```bash
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
# Output:
# 1.000000 2.000000
# 3.000000 4.000000
```

## Chapter 7: Adding a Composite Type

### Struct Type in Toy

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
  return transpose(value.a) * transpose(value.b);
}

def main() {
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};
  var c = multiply_transpose(value);
  print(c);
}
```

### Custom Type Storage

```c++
struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<mlir::Type>;
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}
  bool operator==(const KeyTy &key) const { return key == elementTypes; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

### Type Class

```c++
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  using Base::Base;
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    return getImpl()->elementTypes;
  }
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

Register:

```c++
void ToyDialect::initialize() {
  addTypes<StructType>();
}
```

### ODS Exposure

```tablegen
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"isa<StructType>($_self)">,
                "Toy struct type">;
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

### Parsing and Printing

```c++
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    mlir::Type elementType;
    if (parser.parseType(elementType)) return nullptr;
    elementTypes.push_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseGreater()) return Type();
  return StructType::get(elementTypes);
}

void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  StructType structType = type.cast<StructType>();
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

Syntax: `!toy.struct<tensor<*xf64>, tensor<*xf64>>`

### New Operations

```mlir
// Struct constant
%0 = toy.struct_constant [
  dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
] : !toy.struct<tensor<*xf64>>

// Struct access
%1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
```

### Constant Folding

```c++
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
  return value();
}

OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr = dyn_cast_or_null<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr) return nullptr;
  size_t elementIndex = index().getZExtValue();
  return structAttr[elementIndex];
}
```

### Materialize Constant

```c++
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (isa<StructType>(type))
    return StructConstantOp::create(builder, loc, type,
                                            cast<mlir::ArrayAttr>(value));
  return ConstantOp::create(builder, loc, type,
                                    cast<mlir::DenseElementsAttr>(value));
}
```

### Complete Pipeline Summary

```
Toy Source → AST → Toy MLIR → Inline → Shape Inference → Canonicalize
    → Lower to Affine (Tensor→MemRef) → Affine Optimization
    → Lower to LLVM → LLVM IR → JIT Execution
```

| Chapter | Key Concept |
|---------|-------------|
| Ch1 | Language design, AST, lexer/parser |
| Ch2 | MLIR dialect, ODS, operations, custom assembly format |
| Ch3 | Pattern rewriting (C++ and DRR), canonicalization |
| Ch4 | Interfaces (inliner, call, shape inference) |
| Ch5 | Dialect conversion, partial lowering to Affine |
| Ch6 | Full lowering to LLVM, code generation, JIT |
| Ch7 | Custom types, type storage, constant folding |
