# MLIR Traits

## Overview

Traits are mixin-like properties that can be attached to operations to provide common functionality and verification.

## Built-in Traits

### Memory Traits

| Trait | Description |
|-------|-------------|
| `NoMemoryEffect` | Operation has no memory side effects |
| `Pure` | No side effects + always produces same result for same inputs |
| `MemRefsNormalizable` | MemRef subview operands are normalizable |

### Type Traits

| Trait | Description |
|-------|-------------|
| `SameOperandsAndResultType` | All operands and results have same type |
| `SameOperandsAndResultShape` | Same shape (allows different element types) |
| `SameTypeOperands` | All operands have same type |
| `SameOperandsAndResultElementType` | Same element type |
| `TypesMatchWith` | Custom type matching rule |

### Structural Traits

| Trait | Description |
|-------|-------------|
| `HasParent<ParentOp>` | Must be directly contained within ParentOp |
| `HasOnlyGraphRegion` | Region is graph-like |
| `IsolatedFromAbove` | Cannot reference values from parent regions |
| `RecursiveSideEffects` | Side effects propagate through regions |
| `SingleBlock` | Region has exactly one block |
| `SingleBlockImplicitTerminator<TermOp>` | Single block with implicit terminator |
| `Terminator` | Operation is a block terminator |
| `TriviallyTerminator` | Simple terminator with no operands |

### Operand/Result Traits

| Trait | Description |
|-------|-------------|
| `SameVariadicOperandSize` | Variadic operand sizes match |
| `SameVariadicResultSize` | Variadic result sizes match |
| `AttrSizedOperandSegments` | Operand segment sizes in attribute |
| `AttrSizedResultSegments` | Result segment sizes in attribute |
| `ElementwiseMappable` | Operation can be applied element-wise |

### Utilitary Traits

| Trait | Description |
|-------|-------------|
| `Commutive` | Operation is commutative |
| `Idempotent` | f(f(x)) = f(x) |
| `Involution` | f(f(x)) = x |

## Using Traits in ODS

```tablegen
def MyOp : MyDialect<"my_op"> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$result);
  let traits = [Pure, SameOperandsAndResultType, ElementwiseMappable];
}
```

## ElementwiseMappable Trait

Operations with this trait can be applied element-wise to tensors and vectors:

```mlir
// Works on scalars, vectors, and tensors
%r = arith.addi %a, %b : i32
%r = arith.addi %a, %b : vector<4xi32>
%r = arith.addi %a, %b : tensor<4xi32>
```

## Custom Traits

### Defining a Trait

```c++
// Define a trait class
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  // Verify trait invariants
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumOperands() != 2)
      return op->emitOpError("requires exactly 2 operands");
    return success();
  }

  // Add methods
  bool hasMyProperty() {
    return this->getOperation()->getNumResults() > 0;
  }
};
```

### Using Custom Trait in ODS

```tablegen
def MyTrait : NativeOpTrait<"::my_namespace::MyTrait">;

def MyOp : MyDialect<"my_op"> {
  let traits = [MyTrait, Pure];
}
```

### PredOpTrait (Predicate-based)

```tablegen
def HasTwoResults : PredOpTrait<"has two results",
  CPred<"$_op->getNumResults() == 2">>;

def MyOp : MyDialect<"my_op"> {
  let traits = [HasTwoResults];
}
```

## Trait Verification

Traits automatically add verification:

```c++
// Trait verification is called during op verification
LogicalResult MyOp::verify() {
  // All trait verifiers run first
  if (failed(OpTrait::SameOperandsAndResultType<MyOp>::verifyTrait(*this)))
    return failure();
  // Then operation-specific verification
  return success();
}
```

## Broadcastable Trait

Ensures operands are broadcast-compatible (for tensor/vector ops):

```tablegen
def MyBroadcastOp : MyDialect<"broadcast_op"> {
  let arguments = (ins
    TensorOf<[F32]>:$lhs,
    TensorOf<[F32]>:$rhs
  );
  let results = (outs TensorOf<[F32]>:$result);
  let traits = [Broadcastable, ElementwiseMappable];
}
```

## Trait vs Interface Comparison

| Aspect | Trait | Interface |
|--------|-------|-----------|
| Purpose | Properties/verification | Behavior/API |
| Definition | C++ class or TableGen | TableGen + C++ model |
| Methods | Optional | Required |
| Runtime query | Limited | Full |
| Extensibility | Compile-time | Runtime |
| Multiple inheritance | Yes | Yes |
| Virtual dispatch | No | Yes |
