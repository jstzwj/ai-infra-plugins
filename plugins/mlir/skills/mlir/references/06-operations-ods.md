# MLIR Operations - ODS (Operation Definition Specification)

## Overview

ODS (Operation Definition Specification) is MLIR's declarative system for defining operations using TableGen. It automatically generates C++ classes, builders, verifiers, parsers, and printers.

## Basic Operation Definition

```tablegen
def MyOp : MyDialect<"op"> {
  let summary = "One line description";
  let description = [{
    Extended description of the operation.
  }];

  let arguments = (ins
    I32:$input,
    F32:$scale
  );

  let results = (outs
    I32:$result
  );
}
```

## Operation Fields

### Documentation

| Field | Description |
|-------|-------------|
| `summary` | One-line description |
| `description` | Extended documentation |
| `mnemonic` | Operation name within dialect |

### Arguments

Arguments combine operands and attributes:

```tablegen
let arguments = (ins
  // Operands
  I32:$input,                           // Single value
  Variadic<I32>:$inputs,                // Variable number of values
  Optional<I32>:$opt_input,             // Optional value
  Variadic<I32>:$inputs = [],           // With default

  // Attributes
  I32Attr:$value,                        // Required attribute
  DefaultValuedAttr<I32Attr, "0">:$flag, // With default
  OptionalAttr<I32Attr>:$opt_attr,       // Optional attribute
  UnitAttr:$is_flag                      // Boolean flag
);
```

### Results

```tablegen
let results = (outs
  I32:$result,                           // Single result
  Variadic<I32>:$results,                // Variable results
  Optional<I32>:$opt_result              // Optional result
);
```

### Multi-Result Operations

```tablegen
let results = (outs
  I32:$result1,
  F32:$result2
);

// Or using sized variadic:
let results = (outs
  Variadic<I32>:$results:2  // Exactly 2 results
);
```

## Type Constraints

### Primitive Types

| Constraint | Description |
|------------|-------------|
| `I1`, `I8`, `I16`, `I32`, `I64` | Signless integer of specific width |
| `SI1`, `SI8`, `SI16`, `SI32`, `SI64` | Signed integer |
| `UI1`, `UI8`, `UI16`, `UI32`, `UI64` | Unsigned integer |
| `F16`, `F32`, `F64`, `BF16` | Float types |
| `Index` | Index type |
| `NoneType` | None type |

### Aggregate Type Constraints

| Constraint | Description |
|------------|-------------|
| `TensorOf<[I32, F32]>` | Tensor with given element types |
| `RankedTensorOf<[I32]>` | Ranked tensor |
| `UnrankedTensorOf<[I32]>` | Unranked tensor |
| `MemRefOf<[I32]>` | Memref with given element types |
| `VectorOf<[I32]>` | Vector with given element types |
| `VectorOfAnyRankOf<[I32]>` | Vector of any rank |

### Composite Constraints

| Constraint | Description |
|------------|-------------|
| `AnyInteger` | Any integer type |
| `AnyFloat` | Any float type |
| `AnyType` | Any type |
| `FloatLike` | Float or vector/tensor of floats |
| `IntLike` | Integer or vector/tensor of integers |
| `SignlessIntegerLike` | Signless integer or container of |
| `ShapedType` | Any shaped type |

### Custom Type Constraints

```tablegen
def MyTypeConstraint : TypeConstraint<CPred<"::llvm::isa<MyType>($_self)">,
                                       "my type">;
```

## Traits

Traits describe properties of operations:

### Common Traits

| Trait | Description |
|-------|-------------|
| `NoMemoryEffect` | Operation has no memory side effects |
| `Pure` | No side effects + always produces same result |
| `SameOperandsAndResultType` | All operands and results have same type |
| `SameTypeOperands` | All operands have same type |
| `HasParent<ParentOp>` | Must be contained within ParentOp |
| `SingleBlock` | Region has exactly one block |
| `SingleBlockImplicitTerminator<TermOp>` | Single block with auto-terminator |
| `IsolatedFromAbove` | Cannot reference values from parent regions |
| `RecursiveSideEffects` | Side effects propagate through regions |
| `Terminator` | Operation is a block terminator |
| `MemRefsNormalizable` | MemRef subview operands are normalizable |

### Variadic Operand/Result Traits

| Trait | Description |
|-------|-------------|
| `SameVariadicOperandSize` | Variadic operand counts match |
| `SameVariadicResultSize` | Variadic result counts match |
| `AttrSizedOperandSegments` | Operands sized by attribute |
| `AttrSizedResultSegments` | Results sized by attribute |

### Example with Traits

```tablegen
def MyOp : MyDialect<"my_op"> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$result);
  let traits = [Pure, SameOperandsAndResultType];
}
```

## Builders

### Default Builders

ODS auto-generates builders from arguments:

```c++
// Auto-generated
static void build(OpBuilder &builder, OperationState &result,
                  Value input, IntegerAttr value);
```

### Custom Builders

```tablegen
let builders = [
  OpBuilder<(ins "Value":$input, "int32_t":$value), [{
    build($_builder, $_state, input,
          $_builder.getI32IntegerAttr(value));
  }]>,
  // Builder with default argument
  OpBuilder<(ins "Value":$input,
             CArg<"int32_t", "0">:$value), [{
    build($_builder, $_state, input,
          $_builder.getI32IntegerAttr(value));
  }]>
];
```

## Verifier

```tablegen
let verifier = [{
  if (getNumOperands() == 0)
    return emitOpError("requires at least one operand");
  return success();
}];
```

Or implement in C++:

```c++
LogicalResult MyOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  return success();
}
```

## Assembly Format

The `assemblyFormat` field defines custom printing/parsing:

```tablegen
let assemblyFormat = "$input `,` $scale attr-dict `:` type($result)";
```

### Assembly Format Directives

| Directive | Description |
|-----------|-------------|
| `$operand` | Operand variable |
| `$attr` | Attribute variable |
| `attr-dict` | Print all attributes |
| `attr-dict-with-keyword` | Print with `{` keyword |
| `type($var)` | Type of operand/result |
| `functional-type(operands, results)` | Function type |
| `(` `)` | Grouping |
| `,` | Literal comma |
| `:` | Literal colon |
| `->` | Arrow |
| `qualified($attr)` | Qualified attribute/type |
| `custom<CustomParser>($args)` | Custom parser/printer |
| `oilist` | Optional keyword groups |
| `regions` | Print regions |
| `successors` | Print successors |
| `type` | Print result types |

### Optional Groups

```tablegen
let assemblyFormat = [{
  $input (`fastmath` $fastmath^)? attr-dict `:` type($result)
}];
```

The `^` marker indicates the "anchor" of the optional group.

### Custom Directive

```tablegen
let assemblyFormat = [{
  $input custom<MyCustomFormat>($attr) attr-dict `:` type($result)
}];
```

Implement parser/printer:

```c++
// Parser
static ParseResult parseMyCustomFormat(OpAsmParser &parser, Attribute &attr) { ... }
// Printer
static void printMyCustomFormat(OpAsmPrinter &printer, MyOp op, Attribute attr) { ... }
```

## Canonicalization

```tablegen
let hasCanonicalizer = 1;
// Or use declarative patterns:
let hasCanonicalizeMethod = 1;
```

Implement:

```c++
void MyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MyCanonicalizationPattern>(context);
}
```

## Folding

```tablegen
let hasFolder = 1;
```

```c++
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast<IntegerAttr>(adaptor.getInput()))
    return IntegerAttr::get(getType(), input.getInt() * 2);
  return {};
}
```

## Return Type Inference

```tablegen
let traits = [InferTypeOpInterface];

let extraClassDeclaration = [{
  static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs);
}];
```

Or use `InferOpTypeInterface`:

```tablegen
let results = (outs InferType<0>:$result);
```

## Side Effects

```tablegen
// No side effects
let traits = [NoMemoryEffect];

// Memory effects
let arguments = (ins
  Arg<MemRefType, "", [MemRead]>:$readMem,
  Arg<MemRefType, "", [MemWrite]>:$writeMem
);
```

Memory effects:
- `MemRead` - Reads from memory
- `MemWrite` - Writes to memory
- `MemAlloc` - Allocates memory
- `MemFree` - Frees memory
- `Alloc` - Allocates resource
- `Free` - Frees resource
- `Read` - Reads resource
- `Write` - Writes resource

## Complete Operation Example

```tablegen
def MatMulOp : MyDialect<"matmul"> {
  let summary = "Matrix multiplication";
  let description = [{
    Computes matrix multiplication C = A * B.
    A is MxK, B is KxN, C is MxN.
  }];

  let arguments = (ins
    TensorOf<[F32]>:$a,
    TensorOf<[F32]>:$b,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs
    TensorOf<[F32]>:$c
  );

  let traits = [
    Pure,
    SameOperandsAndResultElementType
  ];

  let assemblyFormat = [{
    $a `,` $b (`transpose_b` $transpose_b^)?
    attr-dict `:` type($a) `,` type($b) `->` type($c)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$a, "Value":$b, "bool":$transpose_b = false)>
  ];

  let extraClassDeclaration = [{
    // Custom methods
    bool isTransposeB() { return getTransposeB(); }
  }];
}
```

## Generated C++ API

For an operation `MyOp` defined in ODS, MLIR generates:

```c++
class MyOp : public Op<MyOp> {
  // Constructors
  static void build(OpBuilder &, OperationState &, ...);

  // Accessors (get/set for each operand, result, attribute)
  Value getInput();
  void setInput(Value value);
  IntegerAttr getValue();
  void setValue(IntegerAttr value);
  Value getResult();

  // Verification
  static LogicalResult verifyInvariants();
  LogicalResult verify();

  // Parser/Printer
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &printer);

  // Folding/Canonicalization
  OpFoldResult fold(FoldAdaptor adaptor);
  static void getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *);
};
```
