# MLIR Interfaces

## Overview

Interfaces provide abstract APIs for operating on operations, types, and attributes generically without knowing their concrete types. They are the key mechanism enabling MLIR's extensibility while maintaining generic transformation support.

## Operation Interfaces

### CallInterfaces

```c++
// CallOpInterface - operations that call functions
struct CallOpInterface {
  CallInterfaceCallable getCallable();
  void setCallable(CallInterfaceCallable);
  Operation *getCallableForCallee();
  void setCalleeFromCallable(CallInterfaceCallable);
};

// CallableOpInterface - operations that define callable regions
struct CallableOpInterface {
  Region *getCallableRegion();
  ArrayRef<Type> getArgumentTypes();
  ArrayRef<Type> getResultTypes();
};
```

### ControlFlowInterfaces

```c++
// BranchOpInterface - branch operations
struct BranchOpInterface {
  Block *getSuccessor(unsigned index);
  void setSuccessor(Block *block, unsigned index);
  MutableOperandRange getSuccessorOperands(unsigned index);
};

// RegionBranchOpInterface - operations with regions
struct RegionBranchOpInterface {
  void getSuccessorRegions(Optional<unsigned> index,
                            SmallVectorImpl<RegionSuccessor> &regions);
  OperandRange getSuccessorEntryOperands(Optional<unsigned> index);
};
```

### FunctionInterfaces

```c++
struct FunctionOpInterface {
  StringRef getName();
  void setName(StringRef name);
  FunctionType getFunctionType();
  void setFunctionType(FunctionType type);
  unsigned getNumArguments();
  BlockArgument getArgument(unsigned idx);
  unsigned getNumResults();
  Region &getFunctionBody();
  bool isExternal();
};
```

### InferTypeOpInterface

```c++
struct InferTypeOpInterface {
  static LogicalResult inferReturnTypes(
      MLIRContext *context, Optional<Location> location,
      ValueRange operands, DictionaryAttr attributes,
      RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes);
};
```

### SideEffectInterfaces

```c++
struct MemoryEffectOpInterface {
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

// Memory effects
struct MemoryEffects {
  struct Alloc : public Effect {};  // Allocates memory
  struct Free : public Effect {};   // Frees memory
  struct Read : public Effect {};   // Reads from memory
  struct Write : public Effect {};  // Writes to memory
};
```

### LoopLikeOpInterface

```c++
struct LoopLikeOpInterface {
  Value getInductionVar();
  Block *getBody();
  OpFoldResult getLowerBound();
  OpFoldResult getUpperBound();
  OpFoldResult getStep();
};
```

### TilingInterface

```c++
struct TilingInterface {
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder &builder);
  FailureOr<TilingResult> getTiledImplementation(OpBuilder &builder,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
};
```

### ViewLikeOpInterface

```c++
struct ViewLikeOpInterface {
  Value getViewSource();
  OpFoldResult getMixedOffset(OpBuilder &builder);
  SmallVector<OpFoldResult> getMixedSizes(OpBuilder &builder);
  SmallVector<OpFoldResult> getMixedStrides(OpBuilder &builder);
};
```

### VectorUnrollOpInterface

```c++
struct VectorUnrollOpInterface {
  std::optional<SmallVector<int64_t>> getShapeForUnroll();
};
```

### DestinationStyleOpInterface

```c++
struct DestinationStyleOpInterface {
  bool hasPureBufferSemantics();
  bool hasPureTensorSemantics();
  SmallVector<OpOperand *> getDpsInits();
  SmallVector<OpOperand *> getDpsInputs();
};
```

### DataLayoutInterfaces

```c++
struct DataLayoutOpInterface {
  DataLayoutSpecInterface getDataLayoutSpec();
  TargetSystemSpecInterface getTargetSystemSpec();
};

struct DataLayoutTypeInterface {
  unsigned getTypeSize(Type type, const DataLayout &dataLayout);
  unsigned getTypeSizeInBits(Type type, const DataLayout &dataLayout);
  unsigned getTypeAlignment(Type type, const DataLayout &dataLayout);
  std::optional<uint64_t> getIndexBitwidth(Type type, const DataLayout &dataLayout);
};
```

## Complete Built-in Interfaces Reference

| Interface | Category | Description |
|-----------|----------|-------------|
| `CallOpInterface` | Control Flow | Callable operations |
| `CallableOpInterface` | Control Flow | Function definitions |
| `BranchOpInterface` | Control Flow | Branch operations |
| `RegionBranchOpInterface` | Control Flow | Region-based control flow |
| `CopyOpInterface` | Memory | Memory copy |
| `DataLayoutOpInterface` | Data Layout | Data layout specification |
| `DataLayoutTypeInterface` | Data Layout | Type size/alignment queries |
| `DestinationStyleOpInterface` | Structured | Destination-passing style |
| `FunctionOpInterface` | Functions | Function operations |
| `InferTypeOpInterface` | Types | Type inference |
| `InferIntRangeInterface` | Types | Integer range inference |
| `LoopLikeOpInterface` | Loops | Loop operations |
| `MemoryEffectOpInterface` | Side Effects | Memory effects |
| `ParallelCombiningOpInterface` | Parallel | Parallel combining |
| `ShapedOpInterface` | Types | Shaped type operations |
| `TilingInterface` | Transformations | Tiling support |
| `ValueBoundsOpInterface` | Analysis | Value bounds |
| `VectorUnrollOpInterface` | Vector | Vector unrolling |
| `ViewLikeOpInterface` | Memory | View/subview operations |
| `RegionKindInterface` | Regions | Region kind (CFG/Graph) |
| `SymbolOpInterface` | Symbols | Symbol operations |

## Defining Custom Interfaces

### TableGen Definition

```tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let description = [{ Description of the interface }];

  let methods = [
    InterfaceMethod<
      "Get the custom value",
      "unsigned", "getCustomValue"
    >,
    InterfaceMethod<
      "Check if operation has property",
      "bool", "hasProperty", (ins "unsigned":$idx)
    >,
    StaticInterfaceMethod<
      "Static method",
      "LogicalResult", "checkStatic", (ins "Operation *":$op)
    >
  ];
}
```

### C++ Usage

```c++
// Query interface
if (auto iface = dyn_cast<MyInterface>(op)) {
  unsigned val = iface.getCustomValue();
  bool prop = iface.hasProperty(42);
}

// Add to operation in ODS
def MyOp : MyDialect<"my_op"> {
  let interfaces = [MyInterface];
}
```
