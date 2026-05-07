# MLIR Dialect Conversion

## Overview

The dialect conversion framework provides a systematic approach for converting operations between dialects while ensuring type compatibility.

## Core Components

### ConversionTarget

Specifies which operations are legal/illegal:

```c++
ConversionTarget target(getContext());

// Mark specific ops as illegal
target.addIllegalOp<arith::AddIOp, arith::MulIOp>();

// Mark dialect as illegal
target.addIllegalDialect<arith::ArithDialect>();

// Mark specific ops as legal
target.addLegalOp<func::FuncOp, func::ReturnOp>();

// Mark dialect as legal
target.addLegalDialect<LLVM::LLVMDialect>();

// Mark ops as dynamically legal
target.addDynamicallyLegalOp<MyOp>([](MyOp op) {
  return op.getType().isSignlessInteger();
});

// Mark ops as dynamically legal based on type
target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
  return TypeConverter::isLegal(op.getFunctionType());
});
```

### TypeConverter

Handles type conversion between dialects:

```c++
TypeConverter typeConverter;

// Add basic type conversion
typeConverter.addConversion([](IntegerType type) -> Type {
  return IntegerType::get(type.getContext(), type.getWidth());
});

// Add conversion for specific types
typeConverter.addConversion([](MemRefType type) -> std::optional<Type> {
  auto elementType = type.getElementType();
  auto convertedElement = typeConverter.convertType(elementType);
  if (!convertedElement)
    return std::nullopt;
  return MemRefType::get(type.getShape(), *convertedElement,
                          type.getLayout(), type.getMemorySpace());
});

// Add argument materialization
typeConverter.addArgumentMaterialization(
  [](OpBuilder &builder, Type resultType, ValueRange inputs,
     Location loc) -> std::optional<Value> {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

// Add source materialization
typeConverter.addSourceMaterialization(
  [](OpBuilder &builder, Type resultType, ValueRange inputs,
     Location loc) -> std::optional<Value> {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

// Add target materialization
typeConverter.addTargetMaterialization(
  [](OpBuilder &builder, Type resultType, ValueRange inputs,
     Location loc) -> std::optional<Value> {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
```

### TypeConverter Methods

```c++
// Convert a single type
std::optional<Type> convertType(Type type);

// Convert multiple types
LogicalResult convertTypes(TypeRange types, SmallVectorImpl<Type> &result);

// Check if type is legal
bool isLegal(Type type);
bool isLegal(TypeRange types);

// Check if operation is legal
bool isLegal(Operation *op);

// Compute type mapping
LogicalResult computeTypeMapping(Type original, Type converted);
```

### ConversionPattern

```c++
struct MyConversionPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::AddIOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // adaptor contains converted operands
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};
```

### ConversionPatternRewriter

Extended PatternRewriter with additional methods:

```c++
// Apply signature conversion to a block
LogicalResult convertRegionTypes(Region *region, TypeConverter &converter);

// Convert function signature
LogicalResult convertFunctionSignature(FunctionOpInterface func);

// Replace argument uses
void replaceUsesOfBlockArgument(BlockArgument oldArg, Value newVal);

// Start/commit/rollback op conversion
void startOpConversion(Operation *op);
void commitOpConversion();
void cancelOpConversion();
```

## Conversion Drivers

### Full Conversion

```c++
// All operations must be converted (no unconverted ops allowed)
if (failed(applyFullConversion(module, target, std::move(patterns))))
  return failure();
```

### Partial Conversion

```c++
// Unconverted ops are wrapped in unrealized_conversion_cast
if (failed(applyPartialConversion(module, target, std::move(patterns))))
  return failure();
```

### Analysis Conversion

```c++
// Only analyze convertibility (no actual conversion)
if (failed(applyAnalysisConversion(module, target, patterns, callback)))
  return failure();
```

## Block and Region Conversion

### Signature Conversion

```c++
// Convert block arguments
TypeConverter::SignatureConversion sigConversion(block->getNumArguments());
for (auto [idx, arg] : llvm::enumerate(block->getArguments())) {
  Type convertedType = typeConverter.convertType(arg.getType());
  sigConversion.addInputs(idx, convertedType);
}

rewriter.applySignatureConversion(block, sigConversion, &typeConverter);
```

### Region Conversion

```c++
// Convert all regions of an operation
for (Region &region : op->getRegions()) {
  if (failed(rewriter.convertRegionTypes(&region, typeConverter)))
    return failure();
}
```

### Function Signature Conversion

```c++
// Convert function arguments and results
auto funcType = dyn_cast<FunctionOpInterface>(op);
TypeConverter::SignatureConversion sigConversion(
    funcType.getNumArguments());

for (unsigned i = 0; i < funcType.getNumArguments(); ++i) {
  Type argType = funcType.getArgumentTypes()[i];
  Type convertedType = typeConverter.convertType(argType);
  sigConversion.addInputs(i, convertedType);
}

SmallVector<Type, 4> resultTypes;
for (Type resultType : funcType.getResultTypes()) {
  resultTypes.push_back(typeConverter.convertType(resultType));
}
sigConversion.addResults(resultTypes);

rewriter.applySignatureConversion(&funcType.getBody(), sigConversion);
```

## Complete Conversion Example

```c++
struct ArithToLLVMConverter : public TypeConverter {
  ArithToLLVMConverter() {
    // Integer type conversion
    addConversion([](IntegerType type) {
      return IntegerType::get(type.getContext(), type.getWidth());
    });

    // Float type conversion
    addConversion([](FloatType type) { return type; });

    // Index type conversion
    addConversion([](IndexType type) {
      return IntegerType::get(type.getContext(), 64);
    });

    // MemRef type conversion
    addConversion([](MemRefType type) -> std::optional<Type> {
      Type element = type.getElementType();
      return LLVM::LLVMPointerType::get(element);
    });

    // Function type conversion
    addConversion([this](FunctionType type) {
      SmallVector<Type> inputs, results;
      for (auto t : type.getInputs())
        if (auto converted = convertType(t))
          inputs.push_back(*converted);
      for (auto t : type.getResults())
        if (auto converted = convertType(t))
          results.push_back(*converted);
      return LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(type.getContext()), inputs);
    });

    // Materialization
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }

  static std::optional<Value> materializeCast(
      OpBuilder &builder, Type resultType, ValueRange inputs,
      Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  }
};

void populateArithToLLVMPatterns(RewritePatternSet &patterns,
                                  TypeConverter &converter) {
  patterns.add<AddIConverter, MulIConverter, CmpIConverter>(
      converter, patterns.getContext());
}
```

## Best Practices

1. **Use TypeConverter** for all type conversions to maintain consistency
2. **Define materialization hooks** to handle type mismatches during conversion
3. **Use applyFullConversion** when all operations must be converted
4. **Use applyPartialConversion** when some operations can remain unconverted
5. **Test conversion** with the `-convert-*` pass options in mlir-opt
6. **Handle regions carefully** - convert types in nested regions
7. **Consider operation ordering** - some conversions may depend on others
