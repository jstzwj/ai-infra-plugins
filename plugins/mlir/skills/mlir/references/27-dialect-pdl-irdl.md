# MLIR PDL & IRDL Dialects

## PDL (Pattern Descriptor Language) Dialect

PDL provides operations for defining patterns in the IR itself, enabling runtime pattern matching.

### Pattern Definition

```mlir
pdl.pattern @simplify_add_zero : benefit(1) {
  %input = pdl.operand : i32
  %zero = pdl.attribute = 0 : i32
  %const = pdl.operation "arith.constant" {"value" = %zero} -> (%zero : !pdl.attribute)
  %add = pdl.operation "arith.addi"(%input, %const) -> (%result : !pdl.value)
  pdl.rewrite %add with "SimplifyAddZeroRewriter"
}
```

### PDL Operations

#### Value Types
```mlir
%val = pdl.operand : i32              // Match operand of type i32
%vals = pdl.operands : !pdl.range<value>  // Match range of operands
%result = pdl.result 0 of %op         // Match result of operation
%type = pdl.type : i32                // Match type i32
%types = pdl.types : !pdl.range<type> // Match range of types
%attr = pdl.attribute = 42 : i32      // Match attribute with value
%attrs = pdl.attributes               // Match range of attributes
```

#### Operation Matching
```mlir
// Match operation by name
%op = pdl.operation "arith.addi"(%input1, %input2) -> (%result : !pdl.value)

// Match with attributes
%op = pdl.operation "arith.cmpi" {"predicate" = %pred}(%lhs, %rhs) -> (%result : !pdl.value)

// Match with operand types
%op = pdl.operation "arith.addi"(%a : i32, %b : i32) -> (%r : !pdl.value)
```

#### Rewrites
```mlir
// Replace with native rewriter
pdl.rewrite %op with "MyRewriter"

// Replace with new operation
pdl.rewrite %op {
  %new_op = pdl.operation "arith.mul"(%a, %b) -> (%new_result : !pdl.value)
  pdl.replace %op with %new_op
}

// Erase operation
pdl.rewrite %op {
  pdl.erase %op
}
```

### PDL Type System

| Type | Description |
|------|-------------|
| `!pdl.attribute` | Attribute value |
| `!pdl.operation` | Operation |
| `!pdl.type` | Type |
| `!pdl.value` | SSA value |
| `!pdl.range<attribute>` | Range of attributes |
| `!pdl.range<operation>` | Range of operations |
| `!pdl.range<type>` | Range of types |
| `!pdl.range<value>` | Range of values |

## PDL Interpreter Dialect

Implements the PDL matching engine:

```mlir
// These are generated from PDL patterns by the pdl-to-pdl-interp lowering

// Check operation name
pdl_interp.check_operation_name "arith.addi" of %val -> ^match, ^nomatch

// Check operand count
pdl_interp.check_operand_count 2 of %val -> ^match, ^nomatch

// Check result count
pdl_interp.check_result_count 1 of %val -> ^match, ^nomatch

// Get operand
%operand = pdl_interp.get_operand 0 of %val : !pdl.value

// Get result
%result = pdl_interp.get_result 0 of %val : !pdl.value

// Is null check
pdl_interp.is_not_null %val -> ^not_null, ^is_null

// Match attribute
pdl_interp.match_attribute @pred of %val -> ^match, ^nomatch

// Create operation
%new_op = pdl_interp.create_operation "arith.mul"(%a, %b) -> (%r : !pdl.value)

// Replace operation
pdl_interp.replace %op with %new_op

// Erase operation
pdl_interp.erase %op

// Record match
pdl_interp.finalize %matched

// Switch on type
pdl_interp.switch_type %type [i32: ^int, f32: ^float] ^default

// Switch on attribute
pdl_interp.switch_attribute %attr [...]

// Branch
pdl_interp.branch ^target
```

## IRDL (IR Dialect Definition) Dialect

IRDL allows defining dialects dynamically in MLIR:

### Dialect Definition

```mlir
irdl.dialect @my_dialect {
  // Define operations
  irdl.operation @add {
    // Operands
    %lhs = irdl.is i32
    %rhs = irdl.is i32
    irdl.operands(%lhs, %rhs)

    // Results
    %res = irdl.is i32
    irdl.results(%res)

    // Attributes
    irdl.attributes {"mode" = irdl.is "default"}
  }

  // Define types
  irdl.type @my_type {
    irdl.parameters(%width : irdl.is i32)
  }
}
```

### IRDL Operations

| Operation | Description |
|-----------|-------------|
| `irdl.dialect` | Define a dialect |
| `irdl.operation` | Define an operation |
| `irdl.type` | Define a type |
| `irdl.operands` | Declare operands |
| `irdl.results` | Declare results |
| `irdl.attributes` | Declare attributes |
| `irdl.is` | Equality constraint |
| `irdl.any_of` | Union constraint |
| `irdl.all_of` | Intersection constraint |
| `irdl.any` | Any type constraint |
| `irdl.parametric` | Parametric type constraint |

### Constraint Combinators

```mlir
// Any of (union)
%any_float = irdl.any_of(irdl.is f32, irdl.is f64)

// All of (intersection)
%valid = irdl.all_of(irdl.is i32, irdl.is_signless)

// Any type
%any = irdl.any

// Parametric type
%my_vec = irdl.parametric(@vector, irdl.is i32)
```

## Comparison: PDL vs IRDL vs ODS

| Feature | PDL | IRDL | ODS |
|---------|-----|------|-----|
| Purpose | Pattern matching | Dialect definition | Dialect definition |
| Runtime | Yes | Yes | No (compile-time) |
| Language | MLIR IR | MLIR IR | TableGen |
| Flexibility | High | High | Medium |
| Performance | Lower | Lower | Higher |
| Tooling | Rich | Basic | Rich |
| Use case | Dynamic patterns | Dynamic dialects | Static dialects |
