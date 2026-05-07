# MLIR PDLL (Pattern Descriptor Language)

## Overview

PDLL is a domain-specific language for defining MLIR rewrite patterns. It provides a more expressive alternative to DRR with better type checking and composability.

## Basic Pattern

```pdll
// Simple pattern
Pattern {
  let op = MyDialect.AddOp(input: Value, rhs: Value);
  erase op;
}

// Pattern with rewrite
Pattern {
  let op = MyDialect.DoubleOp(input: Value);
  rewrite op with {
    let constant = arith.ConstantOp(2: i32);
    replace op with arith.MulIOp(input, constant.result);
  };
}
```

## Pattern Matching

### Operation Matching

```pdll
// Match by operation name
let op = arith.AddIOp;

// Match with operands
let op = arith.AddIOp(lhs: Value, rhs: Value);

// Match with specific attributes
let op = MyDialect.Op<{attr = "value"}>;

// Match with results
let op = arith.AddIOp -> (result: Value);

// Match with all components
let op = arith.AddIOp(lhs, rhs) -> (result) {attr = "value"};
```

### Value Binding

```pdll
// Bind operands
let op = arith.AddIOp(lhs: Value, rhs: Value);

// Bind results
let op = arith.AddIOp -> (result: Value);

// Bind specific result
let op = arith.AddIOp -> (result: Value);
```

### Type Constraints

```pdll
// Constrain operand types
let op = arith.AddIOp(lhs: Value<i32>, rhs: Value<i32>);

// Constrain result type
let op = arith.AddIOp -> (result: Value<f32>);
```

### Attribute Constraints

```pdll
// Match specific attribute value
let op = MyDialect.TaggedOp<{tag = "important"}>;

// Match attribute existence
let op = MyDialect.Op<{attr = _}>;
```

## Rewrite Actions

### Replace Operation

```pdll
Pattern {
  let op = MyDialect.AddOp(lhs: Value, rhs: Value);
  replace op with arith.AddIOp(lhs, rhs);
}
```

### Erase Operation

```pdll
Pattern {
  let op = MyDialect.NopOp;
  erase op;
}
```

### Create New Operations

```pdll
Pattern {
  let op = MyDialect.DoubleOp(input: Value);
  rewrite op with {
    let c = arith.ConstantOp(2: i32);
    replace op with arith.MulIOp(input, c);
  };
}
```

## Constraints

### Inline Constraints

```pdll
Constraint IsZero(value: Value) [{
  return ::llvm::isa<arith::ConstantOp>(value.getDefiningOp()) &&
         ::llvm::cast<arith::ConstantOp>(value.getDefiningOp()).getValue()
             .isZero();
}];

Pattern {
  let op = arith.AddIOp(lhs: Value, rhs: Value)[IsZero(rhs)];
  replace op with lhs;
}
```

### Native Constraints

```pdll
// Using C++ constraint
Constraint myConstraint(arg: Value) -> Attr;

// With body
Constraint IsPositive(attr: Attr) [{
  if (auto intAttr = ::llvm::dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt() > 0 ? success() : failure();
  return failure();
}];
```

### Rewriter Constraints

```pdll
Constraint CheckType(value: Value) [{
  auto type = value.getType();
  if (::llvm::isa<IntegerType>(type))
    return success();
  return failure();
}];
```

## Variables and Expressions

### Variable Declarations

```pdll
// Value variables
let input: Value;
let input: Value<i32>;

// Operation variables
let op: arith.AddIOp;

// Type variables
let ty: Type;
let ty: Type<i32>;

// Attribute variables
let attr: Attr;
let attr: Attr<i32>;
```

### Operation Results

```pdll
// Access results
let addOp = arith.AddIOp(lhs, rhs) -> (result: Value);
let value = addOp.result;

// Multiple results
let op = MyDialect.SplitOp(input) -> (low: Value, high: Value);
```

## Native Code Calls

```pdll
// Call C++ function
let result = NativeCodeCall<"myFunction($0)">(input);

// With multiple arguments
let result = NativeCodeCall<"combine($0, $1)">(a, b);
```

## Pattern Rewriting

### Multi-Step Rewrites

```pdll
Pattern {
  let op = MyDialect.ComplexOp(input: Value);
  rewrite op with {
    // Step 1: Create intermediate value
    let cast = arith.ExtSIOp(input);
    // Step 2: Create final result
    let result = arith.MulIOp(cast, cast);
    replace op with result;
  };
}
```

### Conditional Rewrites

```pdll
Pattern {
  let op = MyDialect.CondOp(cond: Value, input: Value);
  rewrite op with {
    if (cond) {
      replace op with arith.AddIOp(input, input);
    } else {
      replace op with input;
    }
  };
}
```

## Modules and Includes

```pdll
// Import other PDLL files
#include "MyPatterns.pdll"

// Define reusable patterns
Pattern SimplifyAddZero {
  let op = arith.AddIOp(lhs: Value, rhs: Value)[IsZero(rhs)];
  replace op with lhs;
}
```

## PDLL Compilation

```bash
# Compile PDLL to C++ patterns
mlir-pdll my_patterns.pdll -o my_patterns.cpp.inc

# Compile to PDL IR
mlir-pdll my_patterns.pdll -o my_patterns.mlir --emit-pdl
```

## Integration with Pass Pipeline

```c++
// Register PDLL-generated patterns
void populateMyPatterns(RewritePatternSet &patterns) {
  // Include generated patterns
  #include "my_patterns.cpp.inc"
  patterns.add<GeneratedPattern0, GeneratedPattern1>(patterns.getContext());
}
```

## PDLL vs DRR

| Feature | PDLL | DRR |
|---------|------|-----|
| Language | Dedicated DSL | TableGen |
| Type checking | Full | Limited |
| Composability | High | Low |
| Complex patterns | Supported | Limited |
| Debugging | Better | Harder |
| Native code | Easy | Possible |
| Multi-step rewrites | Yes | Limited |
| Constraints | Rich | Basic |

## Complete Example

```pdll
// File: simplify_math.pdll

// Constraint: check if value is a power of 2
Constraint IsPowerOfTwo(value: Value) [{
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    return intAttr && (intAttr.getInt() & (intAttr.getInt() - 1)) == 0;
  }
  return false;
}];

// Pattern: Replace multiply by power of 2 with shift left
Pattern ReplaceMulByShift {
  let mul = arith.MulIOp(lhs: Value, rhs: Value)[IsPowerOfTwo(rhs)];
  rewrite mul with {
    let shift_amount = arith.ConstantOp(42: i32); // simplified
    replace mul with arith.ShLIOp(lhs, shift_amount);
  };
}

// Pattern: Simplify add with zero
Pattern SimplifyAddZero {
  let op = arith.AddIOp(lhs: Value, rhs: Value);
  let zero = arith.ConstantOp(0: i32);
  rewrite op with {
    if (rhs == zero.result) {
      replace op with lhs;
    }
  };
}
```
