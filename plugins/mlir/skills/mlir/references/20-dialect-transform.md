# MLIR Transform Dialect

## Overview

The Transform dialect provides a composable, programmable way to apply transformations to IR. It separates the "transform IR" (what to do) from the "payload IR" (what to transform).

## Core Concepts

- **Transform IR**: Operations that describe transformations
- **Payload IR**: The actual MLIR being transformed
- **Handle**: Reference from transform IR to payload IR operations

## Transform Operations

### transform.sequence

Execute a sequence of transformations:

```mlir
transform.sequence failures(propagate) {
^bb0(%arg: !transform.any_op):
  // transformations
}
```

Failure propagation modes: `propagate` (abort on failure), `suppress` (continue on failure)

### transform.alternatives

Try alternatives until one succeeds:

```mlir
transform.alternatives {
  transform.sequence {
    // first attempt
  },
  transform.sequence {
    // second attempt
  }
}
```

### Named Sequence

```mlir
transform.named_sequence @main(%arg: !transform.any_op) -> () {
  // transformations
  transform.yield
}
```

## Structural Transformations

### Loop Tiling

```mlir
%tiled = transform.structured.tile_using_forall %target
    tile_sizes [32, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### Loop Tiling with scf.for

```mlir
%loop, %tiled = transform.structured.tile %target
    sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### Vectorize

```mlir
transform.structured.vectorize %target : !transform.any_op
```

### Lower to Loops

```mlir
%loops = transform.structured.lower_to_loops %target : (!transform.any_op) -> (!transform.any_op)
```

### Bufferize

```mlir
transform.bufferization.one_shot_bufferize %target : !transform.any_op
```

### Lower Packing

```mlir
transform.structured.lower_pack %target : !transform.any_op
```

### Lower Unpacking

```mlir
transform.structured.lower_unpack %target : !transform.any_op
```

## Match Operations

### transform.match.operation_name

```mlir
%matched = transform.match.operation_name %handle ["linalg.matmul"] : !transform.any_op
```

### transform.match.operation_empty

```mlir
%empty = transform.match.operation_empty %handle : !transform.any_op
```

## Handle Manipulation

### transform.get_parent_op

```mlir
%parent = transform.get_parent_op %handle {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
```

### transform.get_producer_of_operand

```mlir
%producer = transform.get_producer_of_operand %handle {operand_number = 0} : (!transform.any_op) -> !transform.any_op
```

### transform.get_consumers_of_operand

```mlir
%consumers = transform.get_consumers_of_operand %handle {operand_number = 0} : (!transform.any_op) -> !transform.any_op
```

### transform.get_result

```mlir
%result = transform.get_result %handle {position = 0} : (!transform.any_op) -> !transform.any_value
```

## Fusion

```mlir
// Fuse producer into consumer
transform.structured.fuse_into_containment_region %producer into %consumer : !transform.any_op
```

## Loop Transformations

### Loop Unroll

```mlir
transform.loop.unroll %loop {factor = 4} : !transform.any_op
```

### Loop Peeling

```mlir
%main, %remainder = transform.loop.peel %loop : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

### Loop Outline

```mlir
%func, %call = transform.loop.outline %loop {func_name = "outlined"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

## Interleaved Patterns

```mlir
transform.apply_patterns to %target {
  transform.apply_patterns.canonicalization
  transform.apply_patterns.linalg.tiling_canonicalization
}
```

## Type System

| Type | Description |
|------|-------------|
| `!transform.any_op` | Handle to any payload operation |
| `!transform.any_value` | Handle to any payload value |
| `!transform.op<name>` | Handle to specific operation type |
| `!transform.param<type>` | Transform parameter |

## Extension Mechanism

The Transform dialect supports extensions for dialect-specific transformations:

```c++
// Register transform extension
transform::registerTransformDialectExtension<MyTransformExtension>();
```

## Complete Operations Reference

| Operation | Description |
|-----------|-------------|
| `transform.sequence` | Execute sequence of transforms |
| `transform.named_sequence` | Define named transform sequence |
| `transform.alternatives` | Try alternatives |
| `transform.yield` | Yield from transform block |
| `transform.structured.tile` | Tile operation |
| `transform.structured.tile_using_forall` | Tile with forall |
| `transform.structured.vectorize` | Vectorize operation |
| `transform.structured.lower_to_loops` | Lower to loops |
| `transform.structured.fuse_into_containment_region` | Fuse operations |
| `transform.structured.lower_pack` | Lower pack |
| `transform.structured.lower_unpack` | Lower unpack |
| `transform.structured.match_operation_name` | Match op name |
| `transform.bufferization.one_shot_bufferize` | Bufferize |
| `transform.loop.unroll` | Unroll loop |
| `transform.loop.peel` | Peel loop |
| `transform.loop.outline` | Outline loop |
| `transform.get_parent_op` | Get parent operation |
| `transform.get_producer_of_operand` | Get operand producer |
| `transform.get_consumers_of_operand` | Get operand consumers |
| `transform.apply_patterns` | Apply rewrite patterns |
| `transform.match.operation_name` | Match by name |
| `transform.match.operation_empty` | Match empty op |
