# MLIR Builtin & Func Dialects

## Builtin Dialect

The builtin dialect provides core operations and types that are always available.

### Module Operation

```mlir
module {
  // top-level operations
}

module @named_module attributes {sym_name = "name"} {
  // named module
}
```

#### ModuleOp API

```c++
class ModuleOp {
  Region &getBodyRegion();
  Block *getBody();
  iterator_range<Block::iterator> getOps();
  template <typename T> iterator_range<typename OpIterator<T>::type> getOps();
  void push_back(Operation *op);
  SymOpInterface lookupSymbol(StringRef name);
};
```

### UnrealizedConversionCastOp

Used during dialect conversion for type materialization:

```mlir
%result = builtin.unrealized_conversion_cast %input : i32 to i64
```

## Func Dialect

### func.func Operation

Defines a function:

```mlir
func.func @name(%arg0: i32, %arg1: f32) -> i64 {
  // function body
  return %result : i64
}

// Function with attributes
func.func @main(%arg: i32) -> i32 attributes {sym_name = "main"} {
  return %arg : i32
}
```

#### FuncOp API

```c++
class FuncOp {
  // Type
  FunctionType getFunctionType();
  void setFunctionType(FunctionType type);

  // Arguments
  unsigned getNumArguments();
  BlockArgument getArgument(unsigned idx);
  Type getArgumentType(unsigned idx);
  ArrayRef<Type> getArgumentTypes();
  void setType(FunctionType type);

  // Results
  unsigned getNumResults();
  Type getResultType(unsigned idx);
  ArrayRef<Type> getResultTypes();

  // Body
  Region &getBody();
  bool isExternal();  // No body (declaration)
  void eraseBody();

  // Symbol
  StringAttr getSymNameAttr();
  StringRef getSymName();
  void setSymName(StringRef name);

  // Visibility
  Visibility getVisibility();
  bool isPublic();
  bool isPrivate();
  bool isDeclaration();
};
```

### func.call Operation

Call a function:

```mlir
%result = func.call @my_func(%arg0, %arg1) : (i32, f32) -> i64

// Call with indirect function pointer
%result = func.call_indirect %func_ptr(%arg0) : (i32) -> i64
```

#### CallOp API

```c++
class CallOp {
  StringRef getCallee();
  void setCallee(StringRef name);
  OperandRange getArguments();
  ResultRange getResults();
  FunctionType getCalleeType();
};
```

### func.return Operation

Return from a function:

```mlir
func.return %val1, %val2 : i32, f32
```

### func.constant Operation

Create a function reference constant:

```mlir
%func_ptr = func.constant @my_func : (i32) -> i64
```

### func.call_indirect Operation

```mlir
%result = func.call_indirect %func_ptr(%arg) : (i32) -> i64
```

## Symbol System

### Symbol Visibility

```mlir
// Public (default) - visible to all
func.func @public_func() -> () { ... }

// Private - only visible within module
func.func private @private_func() -> () { ... }

// Nested - visible within parent symbol table
func.func nested @nested_func() -> () { ... }
```

### SymbolRefAttr

```mlir
// Simple symbol reference
@function_name

// Nested symbol reference
@module::@nested_function

// Deeply nested
@parent::@child::@grandchild
```

### SymbolTable

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
  void erase(StringRef name);

  // Replace all uses
  void replaceAllUsesWith(StringRef oldName, StringRef newName);

  // Collection
  iterator_range<iterator> getOps();
  template <typename T> iterator_range<typename OpIterator<T>::type> getOps();
  unsigned getNumOps();

  // Static utilities
  static StringRef getSymbolName(Operation *op);
  static void setSymbolName(Operation *op, StringRef name);
  static Visibility getSymbolVisibility(Operation *op);
  static void setSymbolVisibility(Operation *op, Visibility vis);
  static bool isSymbol(Operation *op);
};
```

### Symbol Use Analysis

```c++
// Find all uses of a symbol
auto uses = SymbolTable::getSymbolUses(@my_func, moduleOp);

// Check if symbol is used
bool used = SymbolTable::symbolIsUsedSomewhere(@my_func, moduleOp);

// Walk symbol uses
moduleOp->walk([&](SymbolOpInterface symbolOp) {
  auto name = symbolOp.getName();
  auto uses = SymbolTable::getSymbolUses(name, moduleOp);
});
```

## Builtin Operations Reference

### builtin.module

```
module ::= `module` symbol-ref-id? attributes? region
```

### builtin.unrealized_conversion_cast

```
%result = builtin.unrealized_conversion_cast %inputs : input_types to result_types
```

## Func Operations Reference

| Operation | Syntax | Description |
|-----------|--------|-------------|
| `func.func` | `func.func @name(args) -> results { body }` | Function definition |
| `func.call` | `%r = func.call @name(args) : (types) -> types` | Direct call |
| `func.call_indirect` | `%r = func.call_indirect %ptr(args) : (types) -> types` | Indirect call |
| `func.return` | `func.return values : types` | Return from function |
| `func.constant` | `%ptr = func.constant @name : func_type` | Function reference |
