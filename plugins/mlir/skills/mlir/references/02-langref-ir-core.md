# MLIR Language Reference

This document provides the complete MLIR language reference based on the MLIR specification.

## High-Level Structure

MLIR is fundamentally based on a graph-like data structure of nodes (Operations) and edges (Values). Each Value is the result of exactly one Operation or Block Argument, and has a Value Type defined by the type system. Operations are contained in Blocks, Blocks in Regions. Operations may contain Regions, enabling hierarchical structures.

## Notation

MLIR uses Extended Backus-Naur Form (EBNF):

```
alternation ::= expr0 | expr1 | expr2  // Either expr0 or expr1 or expr2
sequence    ::= expr0 expr1 expr2      // Sequence
repetition0 ::= expr*                  // 0 or more occurrences
repetition1 ::= expr+                  // 1 or more occurrences
optionality ::= expr?                  // 0 or 1 occurrence
grouping    ::= (expr)                 // Grouping
literal     ::= `abcd`                // Matches literal
```

## Common Syntax

```
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

integer-literal   ::= decimal-literal | hexadecimal-literal
decimal-literal   ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal     ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
string-literal    ::= `"` [^"\n\f\v\r]* `"`
```

Comments use BCPL syntax: `//` to end of line.

## Top-Level Productions

```
toplevel := (operation | attribute-alias-def | type-alias-def)*
```

## Identifiers and Keywords

```
// Identifiers
bare-id       ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list  ::= bare-id (`,` bare-id)*
value-id      ::= `%` suffix-id
alias-name    ::= bare-id
suffix-id     ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*

// Uses of value
value-use      ::= value-id (`#` decimal-literal)?
value-use-list ::= value-use (`,` value-use)*
```

Identifiers are prefixed with sigils (`%`, `#`, `@`, `^`, `!`) to prevent collision with keywords.

Value identifiers (`%`) are only in scope within the region where defined.

## Dialects

Dialects are the extension mechanism for MLIR. Each dialect has a unique namespace prefixed to operations/types/attributes.

```
dialect-namespace ::= bare-id
```

Multiple dialects coexist within one module. Dialects can define:
- Operations (e.g., `arith.addi`)
- Types (e.g., `!llvm.ptr`)
- Attributes (e.g., `#map`)

### Target-Specific Operations

Dialects expose target-specific operations:

```mlir
// LLVM intrinsic
%x:2 = "llvm.sadd.with.overflow.i16"(%a, %b) : (i16, i16) -> (i16, i1)
```

## Operations

### Syntax

```
operation             ::= op-result-list? (generic-operation | custom-operation)
                          trailing-location?
generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                          dictionary-properties? region-list? dictionary-attribute?
                          `:` function-type
custom-operation      ::= bare-id custom-operation-format
op-result-list        ::= op-result (`,` op-result)* `=`
op-result             ::= value-id (`:` integer-literal)?
successor-list        ::= `[` successor (`,` successor)* `]`
successor             ::= caret-id (`:` block-arg-list)?
dictionary-properties ::= `<` dictionary-attribute `>`
region-list           ::= `(` region (`,` region)* `)`
dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location     ::= `loc` `(` location `)`
```

### Generic vs Custom Assembly

Every operation has a **generic form**:

```mlir
%result = "arith.addi"(%a, %b) : (i32, i32) -> i32
```

Registered operations may have a **custom form**:

```mlir
%result = arith.addi %a, %b : i32
```

### Operation Components

1. **Results**: SSA values produced (zero or more)
2. **Operands**: SSA values consumed (zero or more)
3. **Properties**: Inherent data stored on the operation
4. **Attributes**: Dictionary of named constant values
5. **Successors**: Target blocks for control flow
6. **Regions**: Nested blocks
7. **Location**: Source tracking information

### Multi-Result Operations

```mlir
// Generic form
%result:2 = "foo_div"() : () -> (f32, i32)

// Named results
%foo, %bar = "foo_div"() : () -> (f32, i32)

// Access individual results
%first = %result#0 : f32
```

### Operations with Attributes

```mlir
// Inherent attributes in properties
%2 = "tf.scramble"(%result#0, %bar) <{fruit = "banana"}> : (f32, i32) -> f32

// Discardable attributes
%foo, %bar = "foo_div"() {some_attr = "value", other_attr = 42 : i64} : () -> (f32, i32)
```

### Operations with Regions

```mlir
"scf.if"(%cond) ({
  // then region
  scf.yield %a : i32
}, {
  // else region
  scf.yield %b : i32
}) : (i1) -> i32
```

### Operations with Successors

```mlir
"cf.cond_br"(%cond)[^bb1, ^bb2(%a : i32)] : (i1) -> ()
```

## Blocks

### Syntax

```
block           ::= block-label operation+
block-label     ::= block-id block-arg-list? `:`
block-id        ::= caret-id
caret-id        ::= `^` suffix-id
value-id-and-type ::= value-id `:` type
value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*
block-arg-list  ::= `(` value-id-and-type-list? `)`
```

A Block is a list of operations. In SSACFG regions, blocks represent basic blocks:
- Operations execute sequentially
- Last operation must be a terminator
- Blocks take arguments (not PHI nodes)

### Example

```mlir
func.func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1):
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  cf.br ^bb3(%a: i64)

^bb2:
  %b = arith.addi %a, %a : i64
  cf.br ^bb3(%b: i64)

^bb3(%c: i64):
  cf.br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = arith.addi %d, %e : i64
  return %0 : i64
}
```

### Block Arguments

Block arguments replace PHI nodes in traditional SSA:
- Entry block arguments = region/function arguments
- Non-entry block arguments = branch target values
- Values defined outside the region can be referenced directly

## Regions

### Definition

A region is an ordered list of blocks contained within an operation. Regions have no name, type, or attributes.

```
region      ::= `{` entry-block? block* `}`
entry-block ::= operation+
```

### Region Kinds

1. **SSACFG Regions** (`RegionKind::SSACFG`): Operations execute sequentially, blocks form CFG
2. **Graph Regions** (`RegionKind::Graph`): No control flow, operations represent graph nodes

### Value Scoping

- Values defined in a region don't escape the enclosing region
- Operations inside a region can reference values from enclosing regions (unless restricted by `IsolatedFromAbove` trait)
- Hierarchical dominance defines value visibility

```mlir
"any_op"(%a) ({
  // %a is in-scope here
  %new_value = "another_op"(%a) : (i64) -> (i64)
}) : (i64) -> (i64)
```

### Control Flow in SSACFG Regions

- Control flow enters through the entry block (first block)
- Terminator operations specify successor blocks
- Control flow exits via terminators without successors
- Single-Entry-Multiple-Exit (SEME) pattern common

### Graph Regions

- Only contain a single block
- Operations can reference results of other operations in any order
- Order is not semantically meaningful
- Useful for representing dataflow graphs

```mlir
"test.graph_region"() ({
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: cycles allowed
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
```

## Type System

### Syntax

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::= type (`,` type)*
type-list-parens ::= `(` `)` | `(` type-list-no-parens `)`

ssa-use-and-type ::= ssa-use `:` type
function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
```

### Type Aliases

```
type-alias-def ::= `!` alias-name `=` type
type-alias     ::= `!` alias-name
```

```mlir
!avx_m128 = vector<4 x f32>
"foo"(%x) : !avx_m128 -> ()
```

### Dialect Types

```
dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
opaque-dialect-type ::= dialect-namespace dialect-type-body
pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident dialect-type-body?
dialect-type-body ::= `<` dialect-type-contents+ `>`
```

```mlir
!tf<string>       // Opaque form
!tf.string         // Pretty form
!llvm<"i32*">      // LLVM pointer type
```

### Builtin Types

See [03 - Types & Attributes](03-types-and-attributes.md) for complete reference.

Core builtin types:
- **Integer**: `i1`, `i8`, `i16`, `i32`, `i64`, `si8`, `ui8`, etc.
- **Float**: `f16`, `bf16`, `f32`, `f64`, `f80`, `f128`
- **Index**: `index` (target-specific size type)
- **None**: `none`
- **Vector**: `vector<NxMxELEMENT_TYPE>`
- **Tensor**: `tensor<MxNxELEMENT_TYPE>` or `tensor<?xELEMENT_TYPE>`
- **MemRef**: `memref<MxNxELEMENT_TYPE>` or `memref<?xELEMENT_TYPE, LAYOUT_MAP>`
- **Tuple**: `tuple<T1, T2, ...>`
- **Function**: `(T1, T2) -> T3`

## Properties

Properties store inherent attributes directly on Operation classes. They can always be serialized to Attribute for generic printing.

## Attributes

### Syntax

```
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

### Attribute Classification

1. **Inherent attributes**: Required by operation semantics, no dialect prefix
2. **Discardable attributes**: External semantics, must have dialect prefix

### Attribute Aliases

```
attribute-alias-def ::= `#` alias-name `=` attribute-value
attribute-alias     ::= `#` alias-name
```

```mlir
#map = affine_map<(d0) -> (d0 + 10)>
%b = affine.apply #map(%a)
```

### Dialect Attributes

```
dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
```

```mlir
#foo<string<"">>    // Opaque form
#foo.string<"">      // Pretty form
```

## Location Information

Every operation carries location information for diagnostics and debugging:

```mlir
%0 = arith.addi %a, %b : i32 loc("example.mlir":4:12)
```

Location types:
- `loc("file":line:col)` - File line/column
- `loc fused<...>` - Fused locations
- `loc(callsite(...))` - Call site locations
- `loc(unknown)` - Unknown location

## Module Structure

The top-level MLIR unit is a module:

```mlir
module {
  func.func @main() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}
```

Module operations can be named:
```mlir
module @my_module attributes {sym_name = "my_module"} {
  // ...
}
```

## IR Versioning

Dialects can handle versioning through `BytecodeDialectInterface`:
- Version encoded in bytecode files
- Lazy loading of version information
- IR upgrades post-parsing via `upgradeFromVersion`
- Custom Attribute/Type encoding upgrades supported

## Common Patterns

### Function Definition

```mlir
func.func @name(%arg0: type0, %arg1: type1) -> result_type {
  // body
  return %value : result_type
}
```

### Function Call

```mlir
%result = func.call @name(%arg0, %arg1) : (type0, type1) -> result_type
```

### Conditional Branch

```mlir
cf.cond_br %cond, ^true_bb, ^false_bb(%val : i32)
```

### Loop

```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%arg = %init) -> (i32) {
  %next = arith.addi %arg, %iv : i32
  scf.yield %next : i32
}
```

### Memory Access

```mlir
%buf = memref.alloc() : memref<10xf32>
%val = memref.load %buf[%idx] : memref<10xf32>
memref.store %val, %buf[%idx] : memref<10xf32>
memref.dealloc %buf : memref<10xf32>
```
