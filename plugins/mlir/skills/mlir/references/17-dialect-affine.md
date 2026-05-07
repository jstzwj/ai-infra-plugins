# MLIR Affine Dialect

## Overview

The Affine dialect provides operations for polyhedral compilation with affine loop nests, affine maps, and integer sets.

## Affine Maps

### Syntax

```
affine-map ::= `affine_map` `<` dim-and-symbol-use-list `->` multi-dim-affine-expr `>`
```

### Examples

```mlir
// Identity
#identity = affine_map<(d0, d1) -> (d0, d1)>

// Transpose
#transpose = affine_map<(d0, d1) -> (d1, d0)>

// Linear access
#linear = affine_map<(d0, d1) -> (d0 * 10 + d1)>

// With symbols
#strided = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

// Shifted access
#shifted = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>

// Tiled access
#tiled = affine_map<(d0) -> (d0 floordiv 32, d0 mod 32)>
```

### Affine Expressions

```
affine-expr ::= dim-id | symbol-id | constant | binary-op | unary-op
binary-op ::= affine-expr `+` affine-expr
            | affine-expr `-` affine-expr
            | affine-expr `*` affine-expr  (if one is constant)
            | affine-expr `floordiv` constant
            | affine-expr `ceildiv` constant
            | affine-expr `mod` constant
unary-op ::= `-` affine-expr
```

## Integer Sets

### Syntax

```
integer-set ::= `affine_set` `<` dim-and-symbol-use-list `:` affine-constraint-conjunction `>`
```

### Examples

```mlir
// Simple bounds
#set0 = affine_set<(d0)[s0] : (d0 >= 0, d0 - s0 < 0)>

// Non-negative
#nonneg = affine_set<(d0) : (d0 >= 0)>

// Range constraint
#range = affine_set<(d0)[s0, s1] : (d0 >= s0, s0 + s1 - d0 >= 0)>
```

## Affine Operations

### affine.for

```mlir
// Basic affine for loop
affine.for %i = 0 to 10 {
  // body
}

// With step
affine.for %i = 0 to 100 step 5 {
  // body
}

// With symbolic bounds
affine.for %i = 0 to %n {
  // %n is a symbol (index type)
}

// With iter_args
%sum = affine.for %i = 0 to 10
    iter_args(%acc = %init) -> (f32) {
  %val = affine.load %buf[%i] : memref<10xf32>
  %new = arith.addf %acc, %val : f32
  affine.yield %new : f32
}
```

### affine.if

```mlir
// Basic conditional
affine.if #set(%i)[%n] {
  // then region
}

// With else
affine.if #set(%i)[%n] {
  // then
} else {
  // else
}

// With results
%val = affine.if #set(%i)[%n] -> (f32) {
  affine.yield %a : f32
} else {
  affine.yield %b : f32
}
```

### affine.load / affine.store

```mlir
// Load with affine map
%val = affine.load %buf[%i, %j] : memref<10x20xf32>

// Load with explicit map
%val = affine.load %buf[%i + 1, %j] : memref<10x20xf32>

// Store
affine.store %val, %buf[%i, %j] : memref<10x20xf32>
```

### affine.apply

Apply an affine map to compute indices:

```mlir
// Compute index using affine map
%idx = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%i)

// With symbols
%idx = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%i)[%stride]
```

### affine.prefetch

```mlir
affine.prefetch %buf[%i, %j], read, locality<3> : memref<10x20xf32>
```

### affine.parallel

```mlir
%sum = affine.parallel (%i) = (0) to (100) reduce ("addf") -> (f32) {
  %val = affine.load %buf[%i] : memref<100xf32>
  affine.yield %val : f32
}
```

### affine.min / affine.max

```mlir
%min = affine.min affine_map<(d0)[s0] -> (d0, s0)>(%i)[%n]
%max = affine.max affine_map<(d0)[s0] -> (d0, s0)>(%i)[%n]
```

## Polyhedral Transformations

### Loop Tiling

```c++
// Tile a loop nest by factors
affine::tileLoops(loop, tileSizes);
```

### Loop Unrolling

```c++
// Unroll loop completely
affine::loopUnrollFull(forOp);

// Unroll by factor
affine::loopUnrollByFactor(forOp, 4);
```

### Loop Fusion

```c++
// Fuse loop nests
affine::fuseLoops(loopA, loopB, fusedLoop);
```

### Loop Interchange

```c++
// Interchange loop nest dimensions
affine::permuteLoopNest(loops, permutation);
```

### Loop Skewing

```c++
// Skew loop for parallelism
affine::loopSkew(loop, skewFactor);
```

## Affine Analysis

### Dependence Analysis

```c++
// Check for dependences between accesses
affine::DependenceResult result =
    affine::checkMemrefAccessDependence(accessA, accessB, loopDepth);
```

### Affine Access Analysis

```c++
// Get affine value bounds
affine::BoundResult bounds = affine::getBoundForAffineAccess(access);
```

## Restrictions on Affine Operations

1. **Loop bounds** must be affine expressions of outer loop variables and symbols
2. **Array subscripts** must be affine expressions
3. **Conditionals** must use integer sets with affine constraints
4. **No data-dependent control flow** inside affine loops

### Valid Affine Access

```mlir
affine.for %i = 0 to %n {
  affine.for %j = 0 to %m {
    // Valid: affine subscripts
    %val = affine.load %A[%i, %j + 1] : memref<?x?xf32>
    affine.store %val, %B[%i * 2, %j] : memref<?x?xf32>
  }
}
```

### Invalid (Non-Affine) Access

```mlir
affine.for %i = 0 to %n {
  // Invalid: %idx is not an affine expression
  %idx = arith.muli %i, %non_const : index
  %val = affine.load %A[%idx] : memref<?xf32>  // Error!
}
```
