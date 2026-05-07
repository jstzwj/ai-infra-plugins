# MLIR SPIR-V Dialect

## Overview

The SPIR-V dialect maps directly to the SPIR-V specification, enabling compilation to Vulkan, OpenCL, and other SPIR-V targets.

## SPIR-V Types

```mlir
!spv.array<10 x f32>                      // Array
!spv.rtarray<f32>                         // Runtime array
!spv.image<f32, dim1d, no_depth, ...>     // Image
!spv.sampler                               // Sampler
!spv.sampled_image<!spv.image<...>>       // Sampled image
!spv.ptr<!spv.struct<f32, i32>, Storage>  // Pointer
!spv.struct<f32, i32>                      // Struct
!spv.matrix<4 x vector<4xf32>>            // Matrix
!spv.bool                                  // Boolean
!spv.void                                  // Void
```

### Scalar Types
```
!spv.bool
i32 (signless), si32 (signed), ui32 (unsigned)
f16, f32, f64
```

### Composite Types
```
!spv.vec4<f32>                             // Vector (4 floats)
!spv.array<10 x f32>                      // Fixed array
!spv.rtarray<f32>                         // Runtime array
!spv.struct<f32, !spv.vec4<i32>>          // Struct
!spv.matrix<4 x vector<4xf32>>            // Matrix
```

## Module Structure

```mlir
spv.module Logical GLSL450 requires #spv.v1.0 {
  spv.GlobalVariable @__builtin_var_NumWorkgroups builtIn("NumWorkgroups") : !spv.ptr<!spv.vec3<i32>, Input>
  spv.func @kernel(%arg0: !spv.ptr<!spv.rtarray<f32>, StorageBuffer>) "None" attributes {
    spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}
  } {
    // kernel body
    spv.Return
  }
}
```

## Arithmetic Operations

```mlir
// Integer arithmetic
%sum = spv.IAdd %a, %b : i32
%dif = spv.ISub %a, %b : i32
%prod = spv.IMul %a, %b : i32
%sdiv = spv.SDiv %a, %b : i32
%udiv = spv.UDiv %a, %b : i32
%srem = spv.SRem %a, %b : i32
%umod = spv.UMod %a, %b : i32

// Float arithmetic
%fsum = spv.FAdd %a, %b : f32
%fdif = spv.FSub %a, %b : f32
%fprod = spv.FMul %a, %b : f32
%fdiv = spv.FDiv %a, %b : f32
%frem = spv.FRem %a, %b : f32
%fmod = spv.FMod %a, %b : f32
%fneg = spv.FNegate %a : f32
```

## Bitwise Operations

```mlir
%and = spv.BitwiseAnd %a, %b : i32
%or = spv.BitwiseOr %a, %b : i32
%xor = spv.BitwiseXor %a, %b : i32
%not = spv.Not %a : i32
%shl = spv.ShiftLeftLogical %a, %b : i32, i32
%shr = spv.ShiftRightArithmetic %a, %b : i32, i32
%shr = spv.ShiftRightLogical %a, %b : i32, i32
```

## Comparison Operations

```mlir
// Integer comparison
%eq = spv.IEqual %a, %b : i32
%ne = spv.INotEqual %a, %b : i32
%slt = spv.SLessThan %a, %b : i32
%sle = spv.SLessThanEqual %a, %b : i32
%sgt = spv.SGreaterThan %a, %b : i32
%sge = spv.SGreaterThanEqual %a, %b : i32
%ult = spv.ULessThan %a, %b : i32
%ule = spv.ULessThanEqual %a, %b : i32
%ugt = spv.UGreaterThan %a, %b : i32
%uge = spv.UGreaterThanEqual %a, %b : i32

// Float comparison
%foeq = spv.FOrdEqual %a, %b : f32
%fone = spv.FOrdNotEqual %a, %b : f32
%folt = spv.FOrdLessThan %a, %b : f32
%fole = spv.FOrdLessThanEqual %a, %b : f32
%fogt = spv.FOrdGreaterThan %a, %b : f32
%foge = spv.FOrdGreaterThanEqual %a, %b : f32
%fueq = spv.FUnordEqual %a, %b : f32
%fune = spv.FUnordNotEqual %a, %b : f32
%fult = spv.FUnordLessThan %a, %b : f32
%fule = spv.FUnordLessThanEqual %a, %b : f32
%fugt = spv.FUnordGreaterThan %a, %b : f32
%fuge = spv.FUnordGreaterThanEqual %a, %b : f32
```

## Control Flow

```mlir
// Branch
spv.Branch ^bb1

// Conditional branch
spv.BranchConditional %cond, ^bb1(%val : i32), ^bb2

// Selection (if/else)
%result = spv.Select %cond, %true_val, %false_val : i32

// Loop
spv.LoopMerge ^merge, ^continue, None
spv.Loop

// Switch
spv.Switch %val : i32, ^default, [
  0: ^bb0,
  1: ^bb1(%val : i32)
]
```

## Memory Operations

```mlir
// Variable
%var = spv.Variable init(%val) : !spv.ptr<i32, Function>

// Load
%val = spv.Load "StorageBuffer" %ptr : i32

// Store
spv.Store "StorageBuffer" %ptr, %val : i32

// Access chain (GEP equivalent)
%field = spv.AccessChain %struct[%idx] : !spv.ptr<!spv.struct<i32, f32>, StorageBuffer>, i32

// Copy memory
spv.CopyMemory %dst, %src : !spv.ptr<i32, Function>, !spv.ptr<i32, Function>
```

## Conversion Functions

```mlir
// Integer to integer
%trunc = spv.SConvert %a : i32 to i16
%ext = spv.SConvert %a : i16 to i32
%trunc = spv.UConvert %a : i32 to i16
%ext = spv.UConvert %a : i16 to i32

// Float to float
%trunc = spv.FConvert %a : f64 to f32
%ext = spv.FConvert %a : f32 to f64

// Integer to float
%ftosi = spv.ConvertSToF %a : i32 to f32
%ftoui = spv.ConvertUToF %a : i32 to f32

// Float to integer
%sitof = spv.ConvertFToS %a : f32 to i32
%uitof = spv.ConvertFToU %a : f32 to i32

// Bitcast
%cast = spv.Bitcast %a : i32 to f32
```

## Composite Operations

```mlir
// Construct
%vec = spv.CompositeConstruct %a, %b, %c, %d : !spv.vec4<f32>

// Extract
%elem = spv.CompositeExtract %vec [2 : i32] : !spv.vec4<f32>

// Insert
%new_vec = spv.CompositeInsert %val, %vec [2 : i32] : f32, !spv.vec4<f32>

// Vector shuffle
%shuf = spv.VectorShuffle [0, 4, 1, 5] %v1, %v2 : !spv.vec4<f32>, !spv.vec4<f32> -> !spv.vec4<f32>

// Vector extract dynamic
%elem = spv.VectorExtractDynamic %vec, %idx : !spv.vec4<f32>, i32

// Vector insert dynamic
%new = spv.VectorInsertDynamic %val, %vec, %idx : f32, !spv.vec4<f32>, i32
```

## SPIR-V Capabilities

```mlir
spv.module Logical GLSL450 requires #spv.v1.0 [
  #spv.capability<Shader>,
  #spv.capability<Float16>,
  #spv.capability<Int64>
] {
  // ...
}
```

## SPIR-V Serialization

```c++
// Serialize to binary
SmallVector<uint32_t> binary;
if (failed(spv::serialize(module, binary)))
  return failure();
```
