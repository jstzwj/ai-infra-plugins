# MLIR C API

## Overview

MLIR provides a C API for interfacing with MLIR from non-C++ languages.

## Core Types

| C Type | Description |
|--------|-------------|
| `MlirContext` | MLIR context |
| `MlirModule` | Module operation |
| `MlirOperation` | Operation |
| `MlirValue` | SSA value |
| `MlirBlock` | Basic block |
| `MlirRegion` | Region |
| `MlirType` | Type |
| `MlirAttribute` | Attribute |
| `MlirLocation` | Source location |
| `MlirOpPrintingFlags` | Print options |

## Context API

```c
MlirContext mlirContextCreate();
void mlirContextDestroy(MlirContext context);
MlirDialectRegistry mlirDialectRegistryCreate();
void mlirDialectRegistryDestroy(MlirDialectRegistry registry);
void mlirContextAppendDialectRegistry(MlirContext context, MlirDialectRegistry registry);
int mlirContextGetNumRegisteredDialects(MlirContext context);
MlirDialect mlirContextGetOrLoadDialect(MlirContext context, MlirStringRef name);
```

## Location API

```c
MlirLocation mlirLocationUnknownGet(MlirContext context);
MlirLocation mlirLocationFileLineColGet(MlirContext context, MlirStringRef filename,
                                        unsigned line, unsigned col);
MlirLocation mlirLocationFusedGet(MlirContext context, MlirAttribute metadata,
                                   MlirLocation const *locations, size_t n);
void mlirLocationPrint(MlirLocation location, MlirStringCallback callback, void *userData);
```

## Type API

```c
MlirType mlirTypeParseGet(MlirContext context, MlirStringRef type);
MlirContext mlirTypeGetContext(MlirType type);
bool mlirTypeEqual(MlirType t1, MlirType t2);
void mlirTypePrint(MlirType type, MlirStringCallback callback, void *userData);

// Integer types
MlirType mlirIntegerTypeGet(MlirContext context, unsigned bitwidth);
MlirType mlirIntegerTypeSignedGet(MlirContext context, unsigned bitwidth);
MlirType mlirIntegerTypeUnsignedGet(MlirContext context, unsigned bitwidth);

// Float types
MlirType mlirF16TypeGet(MlirContext context);
MlirType mlirF32TypeGet(MlirContext context);
MlirType mlirF64TypeGet(MlirContext context);
MlirType mlirBF16TypeGet(MlirContext context);

// Index type
MlirType mlirIndexTypeGet(MlirContext context);

// Function type
MlirType mlirFunctionTypeGet(MlirContext context, intptr_t nInputs,
                              MlirType const *inputs, intptr_t nResults,
                              MlirType const *results);

// MemRef type
MlirType mlirMemRefTypeGet(MlirType elementType, intptr_t rank,
                            int64_t const *shape, MlirAttribute layout,
                            MlirAttribute memorySpace);

// Tensor type
MlirType mlirRankedTensorTypeGet(intptr_t rank, int64_t const *shape,
                                 MlirType elementType, MlirAttribute encoding);
MlirType mlirUnrankedTensorTypeGet(MlirType elementType);
```

## Attribute API

```c
MlirAttribute mlirAttributeParseGet(MlirContext context, MlirStringRef attr);
void mlirAttributePrint(MlirAttribute attr, MlirStringCallback callback, void *userData);
bool mlirAttributeEqual(MlirAttribute a1, MlirAttribute a2);

// Integer attribute
MlirAttribute mlirIntegerAttrGet(MlirType type, int64_t value);

// Float attribute
MlirAttribute mlirFloatAttrDoubleGet(MlirContext context, MlirType type, double value);

// String attribute
MlirAttribute mlirStringAttrGet(MlirContext context, MlirStringRef str);

// Array attribute
MlirAttribute mlirArrayAttrGet(MlirContext context, intptr_t n, MlirAttribute const *elements);

// Dictionary attribute
MlirAttribute mlirDictionaryAttrGet(MlirContext context, intptr_t n,
                                    MlirNamedAttribute const *elements);
```

## Operation API

```c
// Create operation state
MlirOperationState mlirOperationStateGet(MlirStringRef name, MlirLocation location);
void mlirOperationStateAddResults(MlirOperationState *state, intptr_t n, MlirType const *results);
void mlirOperationStateAddOperands(MlirOperationState *state, intptr_t n, MlirValue const *operands);
void mlirOperationStateAddOwnedRegions(MlirOperationState *state, intptr_t n, MlirRegion const *regions);
void mlirOperationStateAddAttributes(MlirOperationState *state, intptr_t n, MlirNamedAttribute const *attributes);

// Create operation
MlirOperation mlirOperationCreate(MlirOperationState *state);
void mlirOperationDestroy(MlirOperation op);

// Operation properties
MlirContext mlirOperationGetContext(MlirOperation op);
MlirLocation mlirOperationGetLocation(MlirOperation op);
MlirBlock mlirOperationGetBlock(MlirOperation op);
MlirStringRef mlirOperationGetName(MlirOperation op);

// Operands and results
intptr_t mlirOperationGetNumOperands(MlirOperation op);
MlirValue mlirOperationGetOperand(MlirOperation op, intptr_t pos);
intptr_t mlirOperationGetNumResults(MlirOperation op);
MlirValue mlirOperationGetResult(MlirOperation op, intptr_t pos);

// Regions
intptr_t mlirOperationGetNumRegions(MlirOperation op);
MlirRegion mlirOperationGetRegion(MlirOperation op, intptr_t pos);

// Attributes
MlirAttribute mlirOperationGetAttribute(MlirOperation op, MlirStringRef name);
void mlirOperationSetAttribute(MlirOperation op, MlirStringRef name, MlirAttribute attr);

// Print
void mlirOperationPrint(MlirOperation op, MlirStringCallback callback, void *userData);
void mlirOperationPrintWithFlags(MlirOperation op, MlirOpPrintingFlags flags,
                                  MlirStringCallback callback, void *userData);
```

## Block API

```c
MlirBlock mlirBlockCreate(intptr_t nArgs, MlirType const *argTypes,
                           MlirLocation const *argLocations);
void mlirBlockDestroy(MlirBlock block);
intptr_t mlirBlockGetNumArguments(MlirBlock block);
MlirValue mlirBlockGetArgument(MlirBlock block, intptr_t pos);
void mlirBlockInsertOwnedOperation(MlirBlock block, intptr_t pos, MlirOperation operation);
void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation operation);
```

## Region API

```c
MlirRegion mlirRegionCreate();
void mlirRegionDestroy(MlirRegion region);
intptr_t mlirRegionGetNumBlocks(MlirRegion region);
void mlirRegionInsertOwnedBlock(MlirRegion region, intptr_t pos, MlirBlock block);
void mlirRegionAppendOwnedBlock(MlirRegion region, MlirBlock block);
```

## Module API

```c
MlirModule mlirModuleCreateParse(MlirContext context, MlirStringRef module);
MlirModule mlirModuleCreateEmpty(MlirLocation location);
void mlirModuleDestroy(MlirModule module);
MlirOperation mlirModuleGetOperation(MlirModule module);
MlirContext mlirModuleGetContext(MlirModule module);
```

## Pass Management API

```c
MlirPassManager mlirPassManagerCreate(MlirContext context);
void mlirPassManagerDestroy(MlirPassManager pm);
MlirLogicalResult mlirPassManagerRun(MlirPassManager pm, MlirModule module);
void mlirPassManagerAddOwnedPass(MlirPassManager pm, MlirPass pass);

MlirOpPassManager mlirPassManagerGetNestedUnder(MlirPassManager pm, MlirStringRef operationName);
void mlirOpPassManagerAddOwnedPass(MlirOpPassManager opm, MlirPass pass);
```

## Registration API

```c
// Register all dialects
void mlirRegisterAllDialects(MlirDialectRegistry registry);

// Register all passes
void mlirRegisterAllPasses();

// Register specific passes
void mlirRegisterTransformsCanonicalizer();
void mlirRegisterTransformsCSE();
void mlirRegisterTransformsInliner();

// Register conversion passes
void mlirRegisterConversionArithToLLVM();
void mlirRegisterConversionFuncToLLVM();
void mlirRegisterConversionSCFToControlFlow();
```
