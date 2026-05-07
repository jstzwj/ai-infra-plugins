# MLIR Declarative Rewrite Rules (DRR)

## Overview

DRR (Declarative Rewrite Rules) is a TableGen-based system for defining pattern rewrite rules declaratively.

## Basic Syntax

```tablegen
def : Pat<(SourceOp $args), (ResultOp $args)>;
```

### Simple Replacement

```tablegen
// Replace MyAdd with arith.addi
def : Pat<(MyDialect.AddOp $lhs, $rhs),
          (arith.AddIOp $lhs, $rhs)>;
```

### Erase Pattern

```tablegen
def : Pat<(MyDialect.NoOp $input), (replaceWithValue $input)>;
```

## Source Pattern

The source pattern matches operations:

```tablegen
// Match with constant operand
def : Pat<(arith.AddIOp $lhs, (arith.ConstantOp $val)),
          (replaceWithValue $lhs)>;

// Match nested operations
def : Pat<(MyDialect.FusedOp
            (MyDialect.OpA $a),
            (MyDialect.OpB $b)),
          (MyDialect.OpC $a, $b)>;
```

## Result Pattern

The result pattern generates new operations:

```tablegen
// Single result operation
def : Pat<(MyDialect.DoubleOp $input),
          (arith.MulIOp $input, (arith.ConstantOp 2))>;

// Multiple result operations
def : Pat<(MyDialect.UnpackOp $input),
          (MyDialect.First $input),
          [(MyDialect.Second $input)]>;
```

## Constraints

Constraints restrict when patterns apply:

### PredOpAttr Constraint

```tablegen
def IsZero : PredOpTrait<"is zero",
  CPred<"::llvm::cast<IntegerAttr>($_op.getAttr(\"value\"))"
        ".getInt() == 0">>;

def : Pat<(arith.AddIOp $lhs, (arith.ConstantOp $val:IsZero)),
          (replaceWithValue $lhs)>;
```

### SameOperandsAndResultType

```tablegen
def : Pat<(MyDialect.BinaryOp $a, $b),
          (MyDialect.ResultOp $a, $b)>,
      [(SameOperandsAndResultType $a, $b)];
```

### Custom Constraints

```tablegen
def IsPositiveInt : PredOpTrait<"is positive",
  CPred<"::llvm::cast<IntegerAttr>($_op.getAttr(\"value\"))"
        ".getInt() > 0">>;
```

### Constraint Combinators

```tablegen
// Multiple constraints
def : Pat<(MyDialect.Op $input),
          (MyDialect.TransformedOp $input)>,
      [(Constraint1 $input), (Constraint2 $input)];
```

## Native Code Calls

### Generate Result via Native Code

```tablegen
def : Pat<(MyDialect.ComplexOp $a, $b),
          (NativeCodeCall<"createComplexResult($0, $1)"> $a, $b)>;
```

Implement the native function:

```c++
Value createComplexResult(Value a, Value b) {
  OpBuilder builder(a.getContext());
  builder.setInsertionPointAfterValue(a);
  return builder.create<MyDialect::ResultOp>(a.getLoc(), a, b);
}
```

### Helper Functions

```tablegen
def CreateAddOp : NativeCodeCall<"createAdd($0, $1)">;

def : Pat<(MyDialect.Add $a, $b),
          (CreateAddOp $a, $b)>;
```

## Pattern Benefits

```tablegen
// Higher benefit = higher priority
def : Pat<(MyDialect.Op $input), (MyDialect.FastPath $input)>,
      [], /*benefit=*/10>;

def : Pat<(MyDialect.Op $input), (MyDialect.SlowPath $input)>,
      [], /*benefit=*/1>;
```

## Supplementary Parameters

Attributes from source can be forwarded to result:

```tablegen
def : Pat<(MyDialect.TaggedOp $input, $attr:$tag),
          (MyDialect.NewOp $input, $tag)>;
```

## MultiResult Patterns

```tablegen
def : Pat<(MyDialect.SplitOp $input),
          (MyDialect.LowOp $input),
          [(MyDialect.HighOp $input)]>;
```

## Common DRR Patterns

### Constant Folding

```tablegen
def : Pat<(arith.AddIOp $lhs, (arith.ConstantOp $val:IsZero)),
          (replaceWithValue $lhs)>;
```

### Operation Simplification

```tablegen
def : Pat<(MyDialect.IdentityOp $input),
          (replaceWithValue $input)>;
```

### Operation Composition

```tablegen
def : Pat<(MyDialect.MulAddOp $a, $b, $c),
          (arith.AddIOp (arith.MulIOp $a, $b), $c)>;
```

### Type Conversion

```tablegen
def : Pat<(MyDialect.CastOp $input),
          (arith.IndexCastOp $input)>;
```
