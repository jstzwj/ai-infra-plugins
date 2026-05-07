# MLIR Attributes & Types - ODS Definition

## Overview

MLIR allows defining custom types and attributes using TableGen, with automatic generation of C++ classes, parsers, printers, and storage.

## Type Definitions (TypeDef)

### Basic Type Definition

```tablegen
def MyType : TypeDef<MyDialect, "MyType"> {
  let summary = "My custom type";
  let description = [{ Extended description }];

  let parameters = (ins
    "int32_t":$width,
    "mlir::Type":$elementType
  );

  let mnemonic = "my_type";
}
```

### Type Parameters

```tablegen
let parameters = (ins
  // Simple parameters
  "int32_t":$width,
  "unsigned":$rank,

  // MLIR types
  "mlir::Type":$elementType,

  // MLIR attributes
  "mlir::Attribute":$encoding,

  // Complex types
  "llvm::ArrayRef<int64_t>":$shape,
  "llvm::StringRef":$name,

  // Optional parameters
  "std::optional<int32_t>":$optionalParam
);
```

### Custom Storage

For types requiring custom uniquing or complex parameters:

```tablegen
let genStorageClass = 1;  // Generate storage class (default)
```

Or define custom storage:

```c++
struct MyTypeStorage : public TypeStorage {
  // Key type for uniquing
  using KeyTy = std::tuple<int32_t, Type>;

  // Constructor
  MyTypeStorage(int32_t width, Type elementType)
      : width(width), elementType(elementType) {}

  // Equality comparison
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == width && std::get<1>(key) == elementType;
  }

  // Hashing
  static llvm::hash_code hashValue(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  // Construction
  static MyTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<MyTypeStorage>())
        MyTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  int32_t width;
  Type elementType;
};
```

### Assembly Format for Types

```tablegen
let mnemonic = "my_type";

// Simple format using parameters
let assemblyFormat = "`<` $width `,` $elementType `>`";

// With custom directives
let assemblyFormat = [{
  `<` $width custom<PrintType>($elementType) `>`
}];
```

### Type with Builder

```tablegen
let builders = [
  TypeBuilderWithInferredContext<(ins "int32_t":$width, "Type":$elementType)>,
  TypeBuilder<(ins "int32_t":$width), [{
    return get($_ctxt, width, NoneType::get($_ctxt));
  }]>
];
```

### Type Verification

```tablegen
let verifier = [{
  if (getWidth() <= 0)
    return emitError() << "width must be positive";
  if (!getElementType())
    return emitError() << "element type required";
  return success();
}];
```

### Type with Traits and Interfaces

```tablegen
def MyShapedType : TypeDef<MyDialect, "MyShaped"> {
  let parameters = (ins "llvm::ArrayRef<int64_t>":$shape);
  let mnemonic = "my_shaped";

  let traits = [
    ShapedType::Trait
  ];

  let interfaces = [
    ShapedTypeInterface
  ];
}
```

### Type Extra Declarations

```tablegen
let extraClassDeclaration = [{
  // Custom methods
  unsigned getRank() const { return getShape().size(); }
  bool hasStaticShape() const;
  int64_t getNumElements() const;
}];
```

## Attribute Definitions (AttrDef)

### Basic Attribute Definition

```tablegen
def MyAttr : AttrDef<MyDialect, "MyAttr"> {
  let summary = "My custom attribute";
  let description = [{ Extended description }];

  let parameters = (ins
    "int32_t":$value,
    "llvm::StringRef":$name
  );

  let mnemonic = "my_attr";
}
```

### Attribute Parameters

Same as Type parameters. Common patterns:

```tablegen
let parameters = (ins
  "int32_t":$intValue,
  "llvm::ArrayRef<int32_t>":$intArray,
  "mlir::Type":$type,
  "llvm::StringRef":$str,
  "std::optional<int32_t>":$optValue
);
```

### Assembly Format for Attributes

```tablegen
let mnemonic = "my_attr";
let assemblyFormat = "`<` $value `,` $name `>`";
```

### Attribute with Type Builder

```tablegen
def TypedAttr : AttrDef<MyDialect, "TypedAttr"> {
  let parameters = (ins "mlir::Type":$type, "int32_t":$value);
  let mnemonic = "typed_attr";

  // This attribute has an associated type
  let hasCustomTypeBuilder = 1;
}
```

### Attribute Value Accessor

For attributes wrapping values:

```tablegen
def MyEnumAttr : AttrDef<MyDialect, "MyEnum"> {
  let parameters = (ins "MyEnum":$value);
  let mnemonic = "my_enum";
  let enumName = "MyEnum";

  let extraClassDeclaration = [{
    MyEnum getValue() const { return getEnumValue(); }
  }];
}
```

## Constraints

Constraints validate parameter types in ODS.

### Built-in Type Constraints

| Constraint | Description |
|------------|-------------|
| `I1`, `I32`, `I64` | Integer type of specific width |
| `F16`, `F32`, `F64` | Float type |
| `Index` | Index type |
| `AnyInteger` | Any integer type |
| `AnyFloat` | Any float type |
| `AnyType` | Any type |
| `TensorOf<[...]>` | Tensor with given element types |
| `MemRefOf<[...]>` | Memref with given element types |
| `VectorOf<[...]>` | Vector with given element types |
| `ShapedType` | Any shaped type |
| `FunctionType` | Function type |

### Custom Constraints

```tablegen
def MyTypeConstraint : TypeConstraint<CPred<"isMyType($_self)">,
                                       "my type">;

// Complex predicate
def PositiveIntConstraint : AttrConstraint<
  CPred<"::llvm::cast<IntegerAttr>($_self).getInt() > 0">,
  "positive integer">;
```

### Combined Constraints

```tablegen
// OR constraint
def IntOrFloat : TypeConstraint<Or<[
  CPred<"::llvm::isa<IntegerType>($_self)">,
  CPred<"::llvm::isa<FloatType>($_self)">
]>, "integer or float">;

// AND constraint
def RankedTensorOfInt : TypeConstraint<And<[
  CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">,
  CPred<"::llvm::isa<IntegerType>(
    ::llvm::cast<::mlir::ShapedType>($_self).getElementType())">
]>, "ranked tensor of integers">;
```

## Assembly Format Reference

### Type Assembly Format

```
type-assembly ::= mnemonic `<` parameter-list `>`
parameter-list ::= parameter (`,` parameter)*
parameter ::= directive | literal | variable
```

### Attribute Assembly Format

```
attr-assembly ::= mnemonic `<` parameter-list `>`
```

### Format Elements

| Element | Description |
|---------|-------------|
| `$param` | Parameter variable |
| `literal` | Fixed string |
| `,` `:` `->` | Punctuation |
| `(` `)` `[` `]` `{` `}` | Grouping |
| `qualified($param)` | Qualified (dialect-prefixed) |
| `struct($param1, $param2)` | Key-value struct |
| `custom<Name>($param)` | Custom directive |

### Optional Groups

```tablegen
let assemblyFormat = "`<` $required (`,` $optional^)? `>`";
```

### Variadic Parameters

```tablegen
let parameters = (ins
  "llvm::ArrayRef<int64_t>":$dims
);

let assemblyFormat = "`<` $dims `>`";
// Prints as: !dialect.my_type<1, 2, 3>
```

## Generated C++ API

For a type `MyType`:

```c++
class MyType : public Type {
public:
  // Construction
  static MyType get(MLIRContext *ctx, int32_t width, Type elementType);
  static MyType getChecked(Location loc, int32_t width, Type elementType);

  // Accessors
  int32_t getWidth();
  Type getElementType();

  // Mutators (returns new instance)
  MyType withWidth(int32_t newWidth);
  MyType withElementType(Type newType);

  // Verification
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                               int32_t width, Type elementType);

  // Parser/Printer
  static Type parse(AsmParser &parser);
  void print(AsmPrinter &printer) const;

  // Type checking
  static bool classof(Type type);
};
```

For an attribute `MyAttr`:

```c++
class MyAttr : public Attribute {
public:
  // Construction
  static MyAttr get(MLIRContext *ctx, int32_t value, StringRef name);
  static MyAttr getChecked(Location loc, int32_t value, StringRef name);

  // Accessors
  int32_t getValue();
  StringRef getName();

  // Verification
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                               int32_t value, StringRef name);

  // Parser/Printer
  static Attribute parse(AsmParser &parser, Type type);
  void print(AsmPrinter &printer) const;

  // Type checking
  static bool classof(Attribute attr);
};
```

## Complete Example

```tablegen
// TensorMapType - a custom tensor with named dimensions
def TensorMapType : TypeDef<MyDialect, "TensorMap"> {
  let summary = "Tensor with named dimensions";
  let description = [{
    A tensor type where each dimension has an associated name.
  }];

  let parameters = (ins
    "llvm::ArrayRef<int64_t>":$shape,
    "llvm::ArrayRef<llvm::StringRef>":$dimNames,
    "mlir::Type":$elementType
  );

  let mnemonic = "tensor_map";

  let assemblyFormat = [{
    `<` $shape `,` $dimNames `,` qualified($elementType) `>`
  }];

  let genStorageClass = 1;

  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
    bool hasStaticShape() const;
    StringRef getDimName(unsigned i) const { return getDimNames()[i]; }
  }];
}
```
