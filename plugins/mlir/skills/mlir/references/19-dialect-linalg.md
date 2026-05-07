# MLIR Linalg Dialect

## Overview

The Linalg (Linear Algebra) dialect provides structured operations on tensors and memrefs, serving as a key intermediate representation for ML compiler pipelines.

## Generic Operation

The fundamental operation is `linalg.generic`:

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,    // input A
    affine_map<(i, j) -> (j, k)>,    // input B
    affine_map<(i, j) -> (i, k)>     // output C
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : tensor<10x20xf32>, tensor<20x30xf32>)
  outs(%C : tensor<10x30xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %prod = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %prod : f32
      linalg.yield %sum : f32
}
```

### Iterator Types

| Type | Description |
|------|-------------|
| `parallel` | Parallel loop dimension |
| `reduction` | Reduction dimension |

### Indexing Maps

Affine maps define how iteration indices map to tensor/memref dimensions.

## Named Operations

### Elementwise Operations

```mlir
// Fill
linalg.fill ins(%value : f32) outs(%output : tensor<10xf32>)

// Map (elementwise)
linalg.map ins(%input : tensor<10xf32>) outs(%output : tensor<10xf32>) {
  ^bb0(%in: f32):
    %result = math.absf %in : f32
    linalg.yield %result : f32
}
```

### Matmul Operations

```mlir
// General matrix multiply
linalg.matmul ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
  outs(%C : tensor<MxNxf32>)

// Batched matrix multiply
linalg.batch_matmul ins(%A, %B : tensor<BxMxKxf32>, tensor<BxKxNxf32>)
  outs(%C : tensor<BxMxNxf32>)

// Matmul with transpose
linalg.matmul_transpose_b ins(%A, %B : tensor<MxKxf32>, tensor<NxKxf32>)
  outs(%C : tensor<MxNxf32>)
```

### Convolution Operations

```mlir
// 1D convolution
linalg.conv_1d ins(%input, %filter : tensor<NxWxf32>, tensor<Sxf32>)
  outs(%output : tensor<NxOf32>)

// 2D convolution
linalg.conv_2d ins(%input, %filter : tensor<NxHxWxf32>, tensor<RxSxf32>)
  outs(%output : tensor<NxOxPxf32>)

// 3D convolution
linalg.conv_3d ins(%input, %filter : tensor<NxDxHxWxf32>, tensor<TxRxSxf32>)
  outs(%output : tensor<NxOxPxQxf32>)

// Depthwise convolution
linalg.depthwise_conv_2d ins(%input, %filter : tensor<NxHxWxCxf32>, tensor<RxSxCxf32>)
  outs(%output : tensor<NxOxPxCxf32>)
```

### Pooling Operations

```mlir
// Max pooling
linalg.pooling_nhwc_max ins(%input, %window : tensor<NxHxWxCxf32>, tensor<RxSxf32>)
  outs(%output : tensor<NxOxPxCxf32>)

// Min pooling
linalg.pooling_nhwc_min ins(%input, %window : tensor<NxHxWxCxf32>, tensor<RxSxf32>)
  outs(%output : tensor<NxOxPxCxf32>)

// Sum pooling (average)
linalg.pooling_nhwc_sum ins(%input, %window : tensor<NxHxWxCxf32>, tensor<RxSxf32>)
  outs(%output : tensor<NxOxPxCxf32>)
```

### Reduction Operations

```mlir
// Dot product
linalg.dot ins(%A, %B : tensor<Nxf32>, tensor<Nxf32>)
  outs(%C : tensor<f32>)

// Sum reduction
%sum = linalg.reduce ins(%input : tensor<10xf32>)
  init(%zero : f32)
  (%in: f32, %acc: f32) {
    %new = arith.addf %in, %acc : f32
    linalg.yield %new : f32
  }
```

### Copy and Broadcast

```mlir
// Copy
linalg.copy ins(%src : memref<10xf32>) outs(%dst : memref<10xf32>)

// Broadcast
linalg.broadcast ins(%input : tensor<10xf32>)
  outs(%output : tensor<10x20xf32>) dimensions = [1]
```

### Reverse and Transpose

```mlir
// Reverse
linalg.reverse ins(%input : tensor<10xf32>)
  outs(%output : tensor<10xf32>) dimensions = [0]

// Transpose
linalg.transpose ins(%input : tensor<MxNxf32>)
  outs(%output : tensor<NxMxf32>) permutation = [1, 0]
```

## OpDSL (Operation DSL)

OpDSL defines named linalg operations in Python:

```python
# Example: matmul operation definition
@linalg_structured_op
def matmul(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n, D.k)
  C[D.m, D.n] += TypeFn.cast_unsigned(
      T, A[D.m, D.k]) * TypeFn.cast_unsigned(T, B[D.k, D.n])
```

### OpDSL Concepts

| Concept | Description |
|---------|-------------|
| `TensorDef` | Define tensor operand |
| `IndexAttr` | Define index attribute |
| `S.M`, `S.N` | Size variables |
| `D.m`, `D.n` | Dimension variables |
| `domain()` | Define iteration domain |
| `TypeFn` | Type conversion functions |

## Linalg Transformations

### Tiling

```c++
// Tile by size
linalg::tileLinalgOp(op, tileSizes);

// Tile to iterators
linalg::tileToLinalgOps(op, tileSizes);
```

### Fusion

```c++
// Fuse producer-consumer
linalg::fuseProducerOfTensor(op, producer);
```

### Vectorization

```c++
// Vectorize linalg operations
linalg::vectorize(op);
```

### Lowering to Loops

```c++
// Lower to affine loops
linalg::lowerToAffineLoops(op);

// Lower to parallel loops
linalg::lowerToParallelLoops(op);

// Lower to scf loops
linalg::lowerToLoops(op);
```

### Decomposition

```c++
// Decompose complex operations
linalg::decomposeOperation(op);
```

## Complete Named Operations Reference

| Operation | Inputs | Output | Iterator Types |
|-----------|--------|--------|---------------|
| `linalg.fill` | scalar | tensor | parallel |
| `linalg.copy` | tensor | tensor | parallel |
| `linalg.matmul` | A(M,K), B(K,N) | C(M,N) | parallel, parallel, reduction |
| `linalg.batch_matmul` | A(B,M,K), B(B,K,N) | C(B,M,N) | parallel*3, reduction |
| `linalg.matmul_transpose_b` | A(M,K), B(N,K) | C(M,N) | parallel, parallel, reduction |
| `linalg.conv_1d` | input, filter | output | parallel, parallel, reduction |
| `linalg.conv_2d` | input, filter | output | parallel*3, reduction*2 |
| `linalg.conv_3d` | input, filter | output | parallel*4, reduction*3 |
| `linalg.depthwise_conv_2d` | input, filter | output | parallel*3, reduction*2 |
| `linalg.pooling_nhwc_max` | input, window | output | parallel*3, reduction*2 |
| `linalg.pooling_nhwc_min` | input, window | output | parallel*3, reduction*2 |
| `linalg.pooling_nhwc_sum` | input, window | output | parallel*3, reduction*2 |
| `linalg.dot` | A(N), B(N) | C() | reduction |
| `linalg.broadcast` | input | output | parallel |
| `linalg.reverse` | input | output | parallel |
| `linalg.transpose` | input | output | parallel |
| `linalg.map` | input | output | parallel |
| `linalg.reduce` | input | scalar | reduction |
