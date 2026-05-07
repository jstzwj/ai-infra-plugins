# MLIR Specialized Dialects

## EmitC Dialect

Generates C/C++ code from MLIR:

```mlir
// Include header
emitc.include "stdio.h"

// Call C function
%result = emitc.call "printf"(%format, %value) : (emitc.ptr<i8>, i32) -> i32

// Get opaque value
%opaque = emitc.call_opaque "my_func"(%arg) : (i32) -> f32

// Literal
%lit = emitc.literal "42" : i32

// Apply (function application)
%result = emitc.apply "abs"(%input) : (i32) -> i32

// Cast (C-style)
%cast = emitc.cast %val : i32 -> f32

// C expressions
%expr = emitc.expression : i32 {
  emitc.yield %val : i32
}
```

## SparseTensor Dialect

Support for sparse tensor computations:

```mlir
// Convert between dense and sparse
%sparse = sparse_tensor.convert %dense : tensor<10xf32> to tensor<10xf32, #SV>

// Sparse tensor encoding
#SV = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

// Load
%val = sparse_tensor.load %tensor : tensor<10xf32, #SV>

// Concatenate
%result = sparse_tensor.concatenate %a, %b {dimension = 0} : tensor<10xf32, #SV>, tensor<10xf32, #SV> to tensor<20xf32, #SV>

// Insert
sparse_tensor.insert %val, %tensor[%idx] : f32 into tensor<10xf32, #SV>

// Expand
%values, %filled, %added, %count = sparse_tensor.expand %tensor : tensor<10xf32, #SV> to (memref<?xf32>, memref<?xi1>, memref<?xindex>, index)

// Compress
sparse_tensor.compress %values, %filled, %added, %count into %tensor[%idx] : memref<?xf32>, memref<?xi1>, memref<?xindex>, index into tensor<10xf32, #SV>
```

## ARM Dialects

### ArmSVE Dialect (Scalable Vector Extension)

```mlir
// SVE vector type (scalable)
// vector<[16]xi8> = svint8_t
// vector<[4]xf32> = svfloat32_t

// SVE intrinsics
%result = arm_sve.sfmatmul %a, %b : vector<[4]x[4]xf16>, vector<[4]x[4]xf16> -> vector<[4]x[4]xf32>

// SVE load/store
%vec = arm_sve.ld1 %ptr {svetype = vector<[4]xf32>} : !llvm.ptr<f32> -> vector<[4]xf32>
arm_sve.st1 %vec, %ptr {svetype = vector<[4]xf32>} : vector<[4]xf32>, !llvm.ptr<f32>

// SVE add
%sum = arm_sve.add %a, %b : vector<[4]xi32>

// SVE convert
%ext = arm_sve.sxtw %vec : vector<[4]xi32> to vector<[4]xi64>

// SVDOT
%dot = arm_sve.sdot %a, %b, %c : vector<[4]xi32>, vector<[4]xi8>, vector<[4]xi8> -> vector<[4]xi32>

// SMMLA (matrix multiply)
%mma = arm_sve.smmla %acc, %a, %b : vector<[2]x[2]xi32>, vector<[2]x[4]xi8>, vector<[2]x[4]xi8> -> vector<[2]x[2]xi32>
```

### ArmSME Dialect (Scalable Matrix Extension)

```mlir
// SME outer product
%result = arm_sme.fmopa %a, %b, %acc : vector<[4]xf32>, vector<[4]xf32>, vector<[4]x[4]xf32> -> vector<[4]x[4]xf32>

// SME tile load/store
%tile = arm_sme.load_tile %src : memref<?x?xf32> -> vector<[4]x[4]xf32>
arm_sme.store_tile %tile, %dst : vector<[4]x[4]xf32>, memref<?x?xf32>

// SME zero
%zero = arm_sme.zero : vector<[4]x[4]xf32>

// SME move tile to vector
%vec = arm_sme.move_tile_to_vec %tile, %tile_slice, %tile_idx : vector<[4]x[4]xf32> -> vector<[4]xf32>
```

### ArmNeon Dialect

```mlir
// NEON intrinsics
%dot = arm_neon.dot %a, %b : vector<16xi8>, vector<16xi8> -> vector<4xi32>
```

## X86 Dialect

```mlir
// MMX/SSE/AVX intrinsics
%result = x86.mmxbuiltin_padds %a, %b : vector<8xi8>, vector<8xi8> -> vector<8xi8>
```

## XeGPU Dialect (Intel GPU)

```mlir
// XeGPU load/store 2D
%tile = xegpu.load_nd %src [%x, %y] {tile_height = 8, tile_width = 16, element_type = f16} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
xegpu.store_nd %tile, %dst [%x, %y] : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>

// XeGPU create tensor descriptor
%desc = xegpu.create_nd_tdesc %base, %x, %y : memref<?x?xf16>, index, index -> !xegpu.tensor_desc<8x16xf16>

// XeGPU Dpas (dot product accumulate)
%result = xegpu.dpas %a, %b, %c : vector<8x8xf16>, vector<8x8xf16>, vector<8x8xf32> -> vector<8x8xf32>
```

## DLTI Dialect (Data Layout & Target Info)

```mlir
// Data layout specification
#dl = #dlti.dl_spec<
  #dlti.dl_entry<i64, dense<64> : vector<2xi32>>,
  #dlti.dl_entry<f64, dense<64> : vector<2xi32>>,
  #dlti.dl_entry<!llvm.ptr, dense<64> : vector<2xi32>>>
```

## Ptr Dialect

```mlir
// Pointer type
%ptr = ptr.from_memref %memref : memref<10xf32> -> !ptr.ptr<f32>
%val = ptr.load %ptr : !ptr.ptr<f32> -> f32
ptr.store %val, %ptr : f32, !ptr.ptr<f32>
```

## UB (Undefined Behavior) Dialect

```mlir
// Poison value
%poison = ub.poison : i32

// Explicit UB marker
ub.unreachable
```

## MPI Dialect

```mlir
// MPI operations
mpi.send %buf, %count, %dest, %tag, %comm : memref<10xf32>, i32, i32, i32, !mpi.comm
mpi.recv %buf, %count, %source, %tag, %comm : memref<10xf32>, i32, i32, i32, !mpi.comm
```

## WasmSSA Dialect

```mlir
// WebAssembly SSA operations
%result = wasm_ssa.global_get @global : i32
wasm_ssa.global_set @global, %val : i32
```

## SMT Dialect

```mlir
// SMT solver operations
%result = smt.check_sat %constraints : !smt.bool
```
