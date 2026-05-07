# MLIR Vector Dialect

## Overview

The Vector dialect provides operations for SIMD (Single Instruction Multiple Data) vector processing.

## Vector Type

```mlir
vector<4xf32>              // 1D: 4 floats
vector<2x4xf32>            // 2D: 2x4 floats
vector<4x4x8xf32>          // 3D: 4x4x8 floats
vector<16xi32>             // 1D: 16 integers
vector<[4]xf32>            // Scalable vector
```

## Creation Operations

```mlir
// Splat (broadcast scalar to all lanes)
%v = vector.splat %scalar : vector<4xf32>

// Constant
%c = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>

// Create from elements
%v = vector.from_elements %a, %b, %c, %d : vector<4xf32>

// Create mask
%mask = vector.create_mask %c0, %c1, %c2 : vector<3xi1>

// Iota (sequential index)
%iota = vector.iota : vector<4xindex>

// Step
%step = vector.step %start, %step_val : vector<4xf32>
```

## Element Access Operations

```mlir
// Extract single element
%elem = vector.extractelement %v[%idx : i32] : vector<4xf32>

// Insert single element
%new_v = vector.insertelement %val, %v[%idx : i32] : f32 into vector<4xf32>

// Extract from position (static)
%sub = vector.extract %v[1] : vector<4xf32> to f32
%sub2 = vector.extract %v[0, 1] : vector<2x4xf32> to f32
%subvec = vector.extract %v[0] : vector<2x4xf32> to vector<4xf32>

// Insert at position (static)
%new = vector.insert %scalar, %v[1] : f32 into vector<4xf32>
%new2 = vector.insert %subvec, %v[0] : vector<4xf32> into vector<2x4xf32>
```

## Shuffle and Shape Operations

```mlir
// Shuffle (permute elements)
%shuf = vector.shuffle %v1, %v2 [0, 4, 1, 5]
    : vector<4xf32>, vector<4xf32> -> vector<4xf32>

// Interleave
%inter = vector.interleave %v1, %v2 : vector<4xf32> -> vector<8xf32>

// Deinterleave
%deinter1, %deinter2 = vector.deinterleave %v : vector<8xf32> -> vector<4xf32>

// Shape cast (reinterpret shape)
%cast = vector.shape_cast %v : vector<4xf32> to vector<2x2xf32>

// Type cast (reinterpret element type)
%cast = vector.type_cast %m : memref<4xf32> to memref<vector<4xf32>>

// Broadcast
%bc = vector.broadcast %scalar : f32 to vector<4xf32>
%bc2 = vector.broadcast %v1 : vector<2xf32> to vector<4xf32>

// Transpose
%trans = vector.transpose %v, [1, 0] : vector<2x4xf32> -> vector<4x2xf32>

// Flat transpose
%ft = vector.flat_transpose %v {rows = 4 : i32, columns = 4 : i32}
    : vector<16xf32> -> vector<16xf32>
```

## Arithmetic Operations

```mlir
// Fused multiply-add
%fma = vector.fma %a, %b, %c : vector<4xf32>

// Multiply-add
%mac = vector.mac %a, %b, %c : vector<4xf32>

// Contraction (general matrix multiply)
%result = vector.contract {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} %a, %b, %c : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>

// Multi-dimensional reduction
%reduced = vector.multi_reduction <add>, %v [1] : vector<2x4xf32> to vector<2xf32>

// Reduction
%sum = vector.reduction <add>, %v : vector<4xf32> into f32
%min = vector.reduction <minf>, %v : vector<4xf32> into f32
```

### Reduction Operations

Supported reduction kinds: `add`, `mul`, `minf`, `maxf`, `minsi`, `maxsi`, `minui`, `maxui`, `and`, `or`, `xor`

## Memory Operations

```mlir
// Vector load
%v = vector.load %base[%i, %j] : memref<10x10xf32>, vector<4xf32>

// Vector store
vector.store %v, %base[%i, %j] : memref<10x10xf32>, vector<4xf32>

// Masked load
%v = vector.maskedload %base[%idx], %mask, %passthrough
    : memref<10xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>

// Masked store
vector.maskedstore %base[%idx], %mask, %value
    : memref<10xf32>, vector<4xi1>, vector<4xf32>

// Gather (scatter load)
%v = vector.gather %base[%idxs], %mask, %passthrough
    : memref<?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>

// Scatter (scatter store)
vector.scatter %base[%idxs], %mask, %value
    : memref<?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>

// Expand load (load with mask, packing)
%v = vector.expandload %base[%idx], %mask, %passthrough
    : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>

// Compress store (store with mask, packing)
vector.compressstore %base[%idx], %mask, %value
    : memref<?xf32>, vector<4xi1>, vector<4xf32>

// Matrix multiply
%r = vector.matrix_multiply %a, %b
    {lhs_columns = 4 : i32, rhs_columns = 4 : i32}
    : vector<4x4xf32> -> vector<4xf32>
```

## Masking

```mlir
// Create mask from conditions
%mask = vector.create_mask %c0, %c1, %c2 : vector<3xi1>

// Masked operation
%result = vector.mask %mask {
  vector.yield %computed : vector<3xf32>
} : vector<3xi1> -> vector<3xf32>

// Constant mask
%true_mask = arith.constant dense<true> : vector<4xi1>
%false_mask = arith.constant dense<false> : vector<4xi1>
```

## Complete Vector Operations Reference

| Operation | Description |
|-----------|-------------|
| `vector.splat` | Broadcast scalar to vector |
| `vector.from_elements` | Create from elements |
| `vector.extractelement` | Extract single element |
| `vector.insertelement` | Insert single element |
| `vector.extract` | Extract subvector/element |
| `vector.insert` | Insert subvector/element |
| `vector.extractmap` | Extract with affine map |
| `vector.insertmap` | Insert with affine map |
| `vector.shuffle` | Permute elements |
| `vector.interleave` | Interleave elements |
| `vector.deinterleave` | Deinterleave elements |
| `vector.shape_cast` | Reshape |
| `vector.type_cast` | Reinterpret type |
| `vector.broadcast` | Broadcast |
| `vector.transpose` | Transpose |
| `vector.flat_transpose` | Flat transpose |
| `vector.fma` | Fused multiply-add |
| `vector.mac` | Multiply-accumulate |
| `vector.contract` | General contraction |
| `vector.multi_reduction` | Multi-dim reduction |
| `vector.reduction` | Reduction |
| `vector.load` | Vector load |
| `vector.store` | Vector store |
| `vector.maskedload` | Masked load |
| `vector.maskedstore` | Masked store |
| `vector.gather` | Gather load |
| `vector.scatter` | Scatter store |
| `vector.expandload` | Expand load |
| `vector.compressstore` | Compress store |
| `vector.matrix_multiply` | Matrix multiply |
| `vector.create_mask` | Create mask |
| `vector.mask` | Masked operation |
| `vector.iota` | Sequential index |
| `vector.step` | Step vector |
| `vector.yield` | Yield from mask region |
