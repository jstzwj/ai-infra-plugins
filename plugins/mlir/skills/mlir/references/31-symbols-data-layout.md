# MLIR Symbols & Data Layout

## Symbol System

### Symbol Operations

Operations implementing `SymbolOpInterface` define named symbols:

```mlir
// Function as symbol
func.func @my_function(%arg: i32) -> i32 { ... }

// GPU module as symbol
gpu.module @my_gpu_module { ... }

// Global memref as symbol
memref.global "private" @my_global : memref<10xf32> = dense<0.0>
```

### Symbol Visibility

| Visibility | Prefix | Description |
|-----------|--------|-------------|
| Public | (default) | Visible to all |
| Private | `private` | Only within containing symbol table |
| Nested | `nested` | Only within parent symbol table |

```mlir
func.func @public_func() -> () { ... }           // Public
func.func private @private_func() -> () { ... }   // Private
func.func nested @nested_func() -> () { ... }     // Nested
```

### SymbolRefAttr

```mlir
// Simple reference
@function_name

// Nested reference
@module::@nested_function

// Deeply nested
@parent::@child::@grandchild
```

### SymbolTable API

```c++
class SymbolTable {
  // Lookup
  Operation *lookup(StringRef name);
  template <typename T> T lookup(StringRef name);

  // Insert
  void insert(Operation *op);
  StringRef insert(Operation *op, StringRef name);

  // Erase
  void erase(Operation *op);

  // Replace all symbol uses
  LogicalResult replaceAllUsesWith(StringRef oldName, StringRef newName);

  // Collection
  template <typename T> iterator_range<...> getOps();

  // Static utilities
  static StringRef getSymbolName(Operation *op);
  static void setSymbolName(Operation *op, StringRef name);
  static Visibility getSymbolVisibility(Operation *op);
  static void setSymbolVisibility(Operation *op, Visibility vis);
  static bool isSymbol(Operation *op);
  static SymbolTable::UseRange getSymbolUses(Operation *from);
  static bool symbolIsUsedSomewhere(Operation *symbol, Operation *from);
};
```

### Symbol Use Analysis

```c++
// Find all uses of a symbol
auto uses = SymbolTable::getSymbolUses(funcOp);

// Check if symbol is used
bool used = SymbolTable::symbolIsUsedSomewhere(@my_func, moduleOp);

// Walk uses
for (SymbolTable::SymbolUse use : uses) {
  Operation *user = use.getUser();
  SymbolRefAttr symbol = use.getSymbolRef();
}
```

## Data Layout

### Data Layout Specification

```mlir
// Module with data layout
module attributes { dl_spec = #dl } {
  // ...
}

#dl = #dlti.dl_spec<
  #dlti.dl_entry<i64, dense<[64, 8]> : vector<2xi32>>,
  #dlti.dl_entry<f64, dense<[64, 8]> : vector<2xi32>>,
  #dlti.dl_entry<!llvm.ptr, dense<[64, 64]> : vector<2xi32>>,
  #dlti.dl_entry<i32, dense<[32, 4]> : vector<2xi32>>,
  #dlti.dl_entry<i1, dense<[8, 1]> : vector<2xi32>>>
```

### Data Layout Queries

```c++
// Get data layout for current context
DataLayout layout(moduleOp);

// Query type properties
unsigned size = layout.getTypeSize(i64Type);       // Size in bits
unsigned sizeBytes = layout.getTypeSizeInBits(i64Type) / 8;
unsigned align = layout.getTypeAlignment(i64Type);  // Alignment in bytes
uint64_t idxWidth = layout.getIndexBitwidth(i64Type);

// For aggregate types
unsigned structSize = layout.getTypeSize(structType);
unsigned arrayAlign = layout.getTypeAlignment(arrayType);
```

### Data Layout on Operations

```c++
// Module-level data layout
moduleOp->getAttrOfType<DataLayoutSpecAttr>("dl_spec");

// Function-level overrides
funcOp->getAttrOfType<DataLayoutSpecAttr>("dl_spec");
```

### Custom Data Layout

```c++
// Define custom type layout
TypeSize MyType::getTypeSize(Type type, const DataLayout &dataLayout) {
  return TypeSize(/*sizeInBits=*/128);
}

unsigned MyType::getTypeAlignment(Type type, const DataLayout &dataLayout) {
  return 16;
}
```

### Data Layout Interfaces

```c++
// DataLayoutOpInterface - on modules/ops with data layout
struct DataLayoutOpInterface {
  DataLayoutSpecInterface getDataLayoutSpec();
  TargetSystemSpecInterface getTargetSystemSpec();
};

// DataLayoutTypeInterface - on types
struct DataLayoutTypeInterface {
  unsigned getTypeSize(Type type, const DataLayout &dataLayout);
  unsigned getTypeSizeInBits(Type type, const DataLayout &dataLayout);
  unsigned getTypeAlignment(Type type, const DataLayout &dataLayout);
  std::optional<uint64_t> getIndexBitwidth(Type type, const DataLayout &dataLayout);
};
```

### Data Layout in Lowering

```c++
// During LLVM lowering, data layout maps to LLVM data layout
auto llvmDataLayout = translateDataLayout(dataLayoutSpec);
```
