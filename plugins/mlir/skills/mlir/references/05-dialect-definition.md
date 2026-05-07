# MLIR Dialect Definition

## Overview

Dialects are the primary extension mechanism in MLIR. Each dialect groups related operations, types, and attributes under a unique namespace.

## Defining a Dialect

### TableGen Definition

```tablegen
include "mlir/IR/DialectBase.td"

def MyDialect : Dialect {
  let summary = "A short description of my dialect.";

  let description = [{
    Detailed description of the dialect's purpose, operations,
    and design philosophy.
  }];

  // Namespace for operations, types, and attributes
  let name = "my_dialect";

  // C++ namespace for generated classes
  let cppNamespace = "::my_dialect";

  // Dependencies - dialects that must be loaded alongside
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}
```

### C++ Implementation

```c++
// MyDialect.h
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

class MyDialect : public Dialect {
public:
  explicit MyDialect(MLIRContext *ctx);

  static constexpr StringLiteral getDialectNamespace() { return "my_dialect"; }

  // Initialization - register types, attributes, operations, interfaces
  void initialize() override;

  // Parse/print custom types and attributes
  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &printer) const override;
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
  void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
};

// MyDialect.cpp
MyDialect::MyDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<MyDialect>()) {
  initialize();
}

void MyDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "MyDialectOps.cpp.inc"
  >();
  addTypes<MyType1, MyType2>();
  addAttributes<MyAttr1, MyAttr2>();
}
```

### Registration

```c++
// Register dialect with context
mlir::DialectRegistry registry;
registry.insert<MyDialect>();
context.appendDialectRegistry(registry);
context.loadDialect<MyDialect>();
```

## Dialect Fields

### Documentation Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary` | String | One-line description |
| `description` | CodeBlock | Extended description |

### Configuration Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Dialect namespace |
| `cppNamespace` | String | C++ namespace |
| `dependentDialects` | List | Required dialect dependencies |

### Feature Flags

| Field | Default | Description |
|-------|---------|-------------|
| `hasConstantMaterializer` | 0 | Enable constant materialization from attributes |
| `hasNonDefaultDestructor` | 0 | Custom destructor |
| `hasOperationAttrVerify` | 0 | Verify dialect-prefixed operation attributes |
| `hasRegionArgAttrVerify` | 0 | Verify dialect-prefixed region argument attributes |
| `hasRegionResultAttrVerify` | 0 | Verify dialect-prefixed region result attributes |
| `hasOperationInterfaceFallback` | 0 | Fallback for unregistered op interfaces |
| `hasCanonicalizer` | 0 | Dialect-level canonicalization patterns |
| `useDefaultAttributePrinterParser` | 1 | Auto-generate attribute parser/printer |
| `useDefaultTypePrinterParser` | 1 | Auto-generate type parser/printer |
| `isExtensible` | 0 | Allow runtime addition of operations/types/attributes |

## Constant Materialization

When `hasConstantMaterializer = 1`:

```c++
Operation *MyDialect::materializeConstant(OpBuilder &builder,
                                           Attribute value, Type type,
                                           Location loc) {
  // Return constant-like operation for the given attribute value
  if (auto intAttr = dyn_cast<IntegerAttr>(value))
    return builder.create<arith::ConstantOp>(loc, type, intAttr);
  return nullptr;
}
```

## Attribute Verification

### Operation Attribute Verification

```c++
LogicalResult MyDialect::verifyOperationAttribute(
    Operation *op, NamedAttribute attr) {
  StringRef attrName = attr.getName().getValue();
  if (attrName == "my_dialect.my_attr") {
    if (!isa<IntegerAttr>(attr.getValue()))
      return op->emitError("my_attr must be an integer");
  }
  return success();
}
```

### Region Argument Attribute Verification

```c++
LogicalResult MyDialect::verifyRegionArgAttribute(
    Operation *op, unsigned regionIndex, unsigned argIndex,
    NamedAttribute attr) {
  // Verify dialect-prefixed attributes on region arguments
  return success();
}
```

### Region Result Attribute Verification

```c++
LogicalResult MyDialect::verifyRegionResultAttribute(
    Operation *op, unsigned regionIndex, unsigned argIndex,
    NamedAttribute attr) {
  // Verify dialect-prefixed attributes on region results
  return success();
}
```

## Operation Interface Fallback

```c++
void *MyDialect::getRegisteredInterfaceForOp(TypeID typeID,
                                               StringAttr opName) {
  // Return interface model for unregistered operations
  if (auto id = typeID.lookupInterfaceID<MyInterface>())
    return getInterfaceForMyOp(opName);
  return nullptr;
}
```

## Dialect-Level Canonicalization

```c++
void MyDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  // Add dialect-wide canonicalization patterns
  results.add<MyCanonicalizationPattern>(results.getContext());
}
```

## Extensible Dialects

When `isExtensible = 1`, operations/types/attributes can be added at runtime:

```c++
// Register dynamic operation
dialect.addDynamicOp(std::make_unique<MyDynamicOp>("my_op"));

// Register dynamic type
dialect.addDynamicType(std::make_unique<MyDynamicType>("my_type"));
```

## CMake Integration

```cmake
# CMakeLists.txt for a dialect library
add_mlir_dialect_library(MyDialectDialect
  MyDialect.cpp
  MyDialectOps.cpp
  MyDialectTypes.cpp

  DEPENDS
  MLIRMyDialectOpsIncGen
  MLIRMyDialectTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
)

# TableGen rules
mlir_tablegen(MyDialectOps.h.inc -gen-op-decls)
mlir_tablegen(MyDialectOps.cpp.inc -gen-op-defs)
mlir_tablegen(MyDialectTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MyDialectTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(MyDialectDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MyDialectDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRMyDialectOpsIncGen)
```

## Dialect Extension

Dialects can define extensions for modular loading:

```c++
class MyDialectExtension : public DialectExtension<MyDialectExtension> {
public:
  void apply(MLIRContext *ctx, RewriterConfig &config) const override {
    // Load dependent dialects and configure extensions
    ctx->loadDialect<MyDialect>();
    config.addPattern<MyRewritePattern>();
  }
};
```

## Best Practices

1. **Define dialect class** in a separate `.td` file from operations/types
2. **Use full namespaces** for `cppNamespace` to avoid collisions
3. **Declare dependencies** explicitly using `dependentDialects`
4. **Use `useDefaultAttributePrinterParser`** unless you need custom parsing
5. **Implement constant materializer** if your dialect has foldable operations
6. **Verify attributes** using the `hasOperationAttrVerify` hook
7. **Follow naming conventions**: `dialect_namespace.op_name`
8. **Add canonicalization patterns** at both op and dialect levels
