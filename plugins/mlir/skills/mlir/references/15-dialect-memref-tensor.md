# MLIR MemRef & Tensor Dialects

## MemRef Dialect

The MemRef dialect provides operations on memory references (buffers).

### Memory Management

```mlir
// Heap allocation
%buf = memref.alloc() : memref<10xf32>
%dyn = memref.alloc(%n) : memref<?xf32>
%2d = memref.alloc(%m, %n) : memref<?x?xf32>

// Stack allocation
%local = memref.alloca() : memref<4xi32>
%dyn_local = memref.alloca(%n) : memref<?xi32>

// Deallocation
memref.dealloc %buf : memref<10xf32>
```

### Memory Access

```mlir
// Load
%val = memref.load %buf[%idx] : memref<10xf32>
%val2 = memref.load %2d[%i, %j] : memref<4x4xf32>

// Store
memref.store %val, %buf[%idx] : memref<10xf32>
memref.store %val, %2d[%i, %j] : memref<4x4xf32>
```

### Memory Copy

```mlir
memref.copy %src, %dst : memref<10xf32> to memref<10xf32>
```

### Subview Operations

```mlir
// Static subview
%sub = memref.subview %buf[0][4][1] : memref<10xf32> to memref<4xf32>

// Dynamic subview
%sub = memref.subview %2d[%off0, %off1][%sz0, %sz1][%st0, %st1]
    : memref<10x20xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
```

### Cast Operations

```mlir
// View (byte-level reinterpretation)
%view = memref.view %byte_buf[%offset][%size] : memref<i8> to memref<?xf32>

// SubView to contiguous
%c = memref.subview %buf[0][4][1] : memref<10xf32> to memref<4xf32, strided<[1]>>

// Reinterpret cast
%rc = memref.reinterpret_cast %buf to offset: [0], sizes: [10], strides: [1]
    : memref<?xi8> to memref<10xf32>

// Cast
%cast = memref.cast %buf : memref<10xf32> to memref<?xf32>

// Bitcast (reinterpret bit pattern)
%bc = memref.bitcast %a : memref<10xf32> to memref<10xi32>
```

### Dimension Operations

```mlir
// Get dimension
%dim = memref.dim %buf, 0 : memref<?x?xf32>
%rank = memref.rank %buf : memref<4x4xf32>  // returns 2
```

### Prefetch

```mlir
memref.prefetch %buf[%idx], read, locality<3>, data : memref<10xf32>
memref.prefetch %buf[%idx], write, locality<0>, instruction : memref<10xf32>
```

### Atomic Operations

```mlir
%old = memref.atomic_rmw "addf", %value, %buf[%idx] : (f32, memref<10xf32>) -> f32
```

Supported RMW operations: `addf`, `addi`, `assign`, `maximumf`, `minimumf`, `maxf`, `minf`, `mulf`, `ori`, `andi`, `xori`, `maxs`, `maxu`, `mins`, `minu`

### Memory Space

```mlir
// With integer memory space
%global = memref.alloc() : memref<10xf32, 1>

// With attribute memory space
%shared = memref.alloc() : memref<10xf32, "shared">

// GPU shared memory
%smem = memref.alloc() : memref<10xf32, #gpu.address_space<workgroup>>
```

### Global Memory

```mlir
// Define global
memref.global "private" @my_global : memref<10xf32> = dense<0.0>

// Reference global
%ptr = memref.get_global @my_global : memref<10xf32>
```

### Complete MemRef Operations Reference

| Operation | Description |
|-----------|-------------|
| `memref.alloc` | Heap allocation |
| `memref.alloca` | Stack allocation |
| `memref.dealloc` | Free memory |
| `memref.load` | Read element |
| `memref.store` | Write element |
| `memref.copy` | Copy between memrefs |
| `memref.subview` | Create subview |
| `memref.view` | Byte-level view |
| `memref.reinterpret_cast` | Reinterpret layout |
| `memref.cast` | Type cast |
| `memref.bitcast` | Bit-level cast |
| `memref.dim` | Get dimension size |
| `memref.rank` | Get rank |
| `memref.prefetch` | Prefetch memory |
| `memref.atomic_rmw` | Atomic read-modify-write |
| `memref.generic_atomic_rmw` | Generic atomic RMW |
| `memref.get_global` | Get global memref |
| `memref.global` | Define global memref |
| `memref.assume_alignment` | Assume alignment |
| `memref.extract_strided_metadata` | Extract strided metadata |
| `memref.extract_aligned_pointer_as_index` | Get pointer as index |
| `memref.collapse_shape` | Collapse dimensions |
| `memref.expand_shape` | Expand dimensions |
| `memref.reshape` | Reshape with dynamic shape |
| `memref.realloc` | Reallocate |
| `memref.memory_space_cast` | Cast memory space |

## Tensor Dialect

The Tensor dialect provides operations on tensor values (immutable, value semantics).

### Tensor Creation

```mlir
// Empty tensor (uninitialized)
%t = tensor.empty() : tensor<10xf32>
%t = tensor.empty(%n) : tensor<?xf32>

// From elements
%t = tensor.from_elements %a, %b, %c : tensor<3xi32>

// Generate
%t = tensor.generate %m, %n {
  ^bb0(%i: index, %j: index):
    %val = arith.addi %i, %j : index
    tensor.yield %val : index
} : tensor<?x?xindex>

// Splat constant
%splat = arith.constant dense<1.0> : tensor<4xf32>
```

### Tensor Access

```mlir
// Extract element
%val = tensor.extract %t[%idx] : tensor<10xf32>
%val = tensor.extract %t[%i, %j] : tensor<4x4xf32>

// Insert element (returns new tensor)
%new_t = tensor.insert %val into %t[%idx] : tensor<10xf32>
```

### Slice Operations

```mlir
// Extract slice (read-only view)
%slice = tensor.extract_slice %t[0][4][1] : tensor<10xf32> to tensor<4xf32>
%dyn_slice = tensor.extract_slice %t[%off][%sz][%st]
    : tensor<10xf32> to tensor<?xf32>

// Insert slice
%new_t = tensor.insert_slice %slice into %t[0][4][1]
    : tensor<4xf32> into tensor<10xf32>
```

### Shape Operations

```mlir
// Get dimension
%dim = tensor.dim %t, 0 : tensor<?x?xf32>

// Rank (from type)
%rank = tensor.rank %t : tensor<4x4xf32>   // not an op, type-level

// Reshape
%reshaped = tensor.collapse_shape %t [[0, 1], [2]]
    : tensor<?x?x?xf32> into tensor<?x?xf32>
%expanded = tensor.expand_shape %t [[0, 1], [2]]
    : tensor<?x?xf32> into tensor<?x?x?xf32>

// Reshape (dynamic)
%reshaped = tensor.reshape %t %shape : (tensor<10xf32>, tensor<2xi32>) -> tensor<?x?xf32>

// Pad
%padded = tensor.pad %t low[0, 1] high[2, 0] {
  ^bb0(%i: index, %j: index):
    %zero = arith.constant 0.0 : f32
    tensor.yield %zero : f32
} : tensor<3x4xf32> to tensor<5x5xf32>
```

### Type Conversion

```mlir
// Cast
%cast = tensor.cast %t : tensor<?xf32> to tensor<10xf32>

// Tensor from memref (materialize)
%t = bufferization.to_tensor %buf : memref<10xf32>

// Tensor to memref
%buf = bufferization.to_memref %t : memref<10xf32>
```

### Parallel Insert Slice

```mlir
%result = tensor.empty() : tensor<100xf32>
%result = scf.forall (%i) in (%n) shared_outs(%o = %result) -> (tensor<100xf32>) {
  %slice = tensor.extract_slice %o[%i][10][1] : tensor<100xf32> to tensor<10xf32>
  %filled = fill_slice(%slice)
  tensor.parallel_insert_slice %filled into %o[%i][10][1]
    : tensor<10xf32> into tensor<100xf32>
}
```

### Complete Tensor Operations Reference

| Operation | Description |
|-----------|-------------|
| `tensor.empty` | Create uninitialized tensor |
| `tensor.extract` | Read element |
| `tensor.insert` | Insert element |
| `tensor.extract_slice` | Extract slice |
| `tensor.insert_slice` | Insert slice |
| `tensor.parallel_insert_slice` | Parallel insert (forall) |
| `tensor.from_elements` | Create from values |
| `tensor.generate` | Generate with closure |
| `tensor.dim` | Get dimension size |
| `tensor.rank` | Get rank |
| `tensor.collapse_shape` | Collapse dimensions |
| `tensor.expand_shape` | Expand dimensions |
| `tensor.reshape` | Dynamic reshape |
| `tensor.pad` | Pad tensor |
| `tensor.cast` | Type cast |
| `tensor.splat` | Broadcast scalar |

## Layout Maps

### Affine Maps for MemRef Layout

```mlir
// Row-major (identity)
#identity = affine_map<(d0, d1) -> (d0, d1)>

// Column-major
#col_major = affine_map<(d0, d1) -> (d1, d0)>

// Strided
#strided = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// With offset
#offset = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1 + 10)>

// Usage
memref<4x4xf32, #identity>
memref<4x4xf32, #col_major>
```

### Strided Layout Attribute

```mlir
memref<4x4xf32, strided<[4, 1]>>
memref<4x4xf32, strided<[4, 1], offset: 0>>
memref<?x?xf32, strided<[?, ?], offset: ?>>
```
