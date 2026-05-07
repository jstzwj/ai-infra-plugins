# MLIR LLVM Dialect & Target

## LLVM Dialect

The LLVM dialect provides a 1:1 mapping to LLVM IR, serving as the final target for CPU compilation.

## LLVM Types

```mlir
!llvm.void
!llvm.half         // f16
!llvm.bfloat       // bf16
!llvm.float        // f32
!llvm.double       // f64
!llvm.fp128        // f128
!llvm.x86_fp80     // f80
!llvm.ppc_fp128    // ppc128
!llvm.i1, !llvm.i8, !llvm.i16, !llvm.i32, !llvm.i64, !llvm.i128
!llvm.ptr<i32>     // pointer to i32
!llvm.ptr          // opaque pointer
!llvm.array<10 x i32>
!llvm.vec<4 x i32>
!llvm.struct<(i32, f32)>
!llvm.struct<"name", (i32, f32)>   // named struct
!llvm.func<i32 (i32, f32)>
!llvm.token
!llvm.metadata
!llvm.x86_mmx
```

## Core Operations

### Integer Arithmetic

```mlir
%add = llvm.add %a, %b : i32
%sub = llvm.sub %a, %b : i32
%mul = llvm.mul %a, %b : i32
%udiv = llvm.udiv %a, %b : i32
%sdiv = llvm.sdiv %a, %b : i32
%urem = llvm.urem %a, %b : i32
%srem = llvm.srem %a, %b : i32
```

### Floating-Point Arithmetic

```mlir
%fadd = llvm.fadd %a, %b : f32
%fsub = llvm.fsub %a, %b : f32
%fmul = llvm.fmul %a, %b : f32
%fdiv = llvm.fdiv %a, %b : f32
%frem = llvm.frem %a, %b : f32
%fneg = llvm.fneg %a : f32
```

### Bitwise Operations

```mlir
%and = llvm.and %a, %b : i32
%or = llvm.or %a, %b : i32
%xor = llvm.xor %a, %b : i32
%shl = llvm.shl %a, %b : i32
%lshr = llvm.lshr %a, %b : i32
%ashr = llvm.ashr %a, %b : i32
```

### Comparison

```mlir
// Integer comparison
%eq = llvm.icmp "eq", %a, %b : i32
%ne = llvm.icmp "ne", %a, %b : i32
%slt = llvm.icmp "slt", %a, %b : i32
%sle = llvm.icmp "sle", %a, %b : i32
%sgt = llvm.icmp "sgt", %a, %b : i32
%sge = llvm.icmp "sge", %a, %b : i32
%ult = llvm.icmp "ult", %a, %b : i32
%ule = llvm.icmp "ule", %a, %b : i32
%ugt = llvm.icmp "ugt", %a, %b : i32
%uge = llvm.icmp "uge", %a, %b : i32

// Float comparison
%foeq = llvm.fcmp "oeq", %a, %b : f32
%fone = llvm.fcmp "one", %a, %b : f32
%folt = llvm.fcmp "olt", %a, %b : f32
%fule = llvm.fcmp "ule", %a, %b : f32
```

### Memory Operations

```mlir
// Alloca
%ptr = llvm.alloca %size x i32 : (i32) -> !llvm.ptr<i32>

// Load
%val = llvm.load %ptr : !llvm.ptr<i32> -> i32

// Store
llvm.store %val, %ptr : i32, !llvm.ptr<i32>

// GetElementPtr (GEP)
%field = llvm.getelementptr %ptr[%idx] : (!llvm.ptr<!llvm.struct<(i32, f32)>>, i32) -> !llvm.ptr<i32>
%elem = llvm.getelementptr %ptr[0, 1] : (!llvm.ptr<!llvm.array<10 x i32>>, i32, i32) -> !llvm.ptr<i32>
```

### Function Operations

```mlir
// Function definition
llvm.func @main(%arg0: i32) -> i32 {
  llvm.return %arg0 : i32
}

// Function call
%result = llvm.call @func(%arg) : (i32) -> i32

// Indirect call
%result = llvm.call %func_ptr(%arg) : !llvm.ptr<func<i32 (i32)>>, (i32) -> i32

// Invoke (with landing pad)
%result = llvm.invoke @func(%arg) to ^normal(%val : i32) unwind ^cleanup(%exn : !llvm.ptr)
```

### Control Flow

```mlir
// Branch
llvm.br ^bb1(%val : i32)

// Conditional branch
llvm.cond_br %cond, ^bb1(%val : i32), ^bb2

// Switch
llvm.switch %val : i32, ^default, [
  0: ^bb0,
  1: ^bb1(%val : i32)
]

// Return
llvm.return %val : i32
```

### Type Conversion

```mlir
// Integer truncation/extension
%trunc = llvm.trunc %a : i64 to i32
%zext = llvm.zext %a : i32 to i64
%sext = llvm.sext %a : i32 to i64

// Float truncation/extension
%truncf = llvm.fptrunc %a : f64 to f32
%extf = llvm.fpext %a : f32 to f64

// Int to float
%sitofp = llvm.sitofp %a : i32 to f32
%uitofp = llvm.uitofp %a : i32 to f32

// Float to int
%fptosi = llvm.fptosi %a : f32 to i32
%fptoui = llvm.fptoui %a : f32 to i32

// Bitcast
%cast = llvm.bitcast %a : !llvm.ptr<i32> to !llvm.ptr<f32>

// Inttoptr / Ptrtoint
%ptr = llvm.inttoptr %a : i64 to !llvm.ptr<i32>
%int = llvm.ptrtoint %a : !llvm.ptr<i32> to i64

// Address space cast
%cast = llvm.addrspacecast %a : !llvm.ptr<i32, 0> to !llvm.ptr<i32, 1>
```

### Aggregate Operations

```mlir
// Extract value
%field = llvm.extractvalue %struct[0] : !llvm.struct<(i32, f32)> -> i32

// Insert value
%new = llvm.insertvalue %val, %struct[1] : f32 into !llvm.struct<(i32, f32)>

// Extract element
%elem = llvm.extractelement %vec[%idx] : vector<4xf32>, i32 -> f32

// Insert element
%new = llvm.insertelement %val, %vec[%idx] : f32 into vector<4xf32>

// Shuffle vector
%shuf = llvm.shufflevector %v1, %v2 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32> -> vector<4xf32>
```

### Special Values

```mlir
// Undef value
%undef = llvm.mlir.undef : i32

// Poison value
%poison = llvm.mlir.poison : i32

// Constant
%c = llvm.mlir.constant(42 : i32) : i32
%fc = llvm.mlir.constant(3.14 : f32) : f32
%null = llvm.mlir.null : !llvm.ptr<i32>
%zero = llvm.mlir.zero : i32
```

### LLVM Intrinsics

```mlir
// Saturating add
%result = "llvm.intr.sadd.with.overflow"(%a, %b) : (i32, i32) -> !llvm.struct<(i32, i1)>

// Saturating sub
%result = "llvm.intr.ssub.with.overflow"(%a, %b) : (i32, i32) -> !llvm.struct<(i32, i1)>

// memcpy
"llvm.intr.memcpy"(%dst, %src, %len, %volatile) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i32, i1) -> ()

// memset
"llvm.intr.memset"(%ptr, %val, %len, %volatile) : (!llvm.ptr<i8>, i8, i32, i1) -> ()

// Vector reduction
%sum = "llvm.intr.vector.reduce.add"(%v) : (vector<4xi32>) -> i32

// FMA
%result = "llvm.intr.fma"(%a, %b, %c) : (f32, f32, f32) -> f32

// Debug trap
"llvm.intr.debugtrap"() : () -> ()
```

## Target LLVM IR Translation

```c++
// Translate MLIR LLVM dialect to LLVM IR
llvm::LLVMContext llvmCtx;
auto llvmModule = translateModuleToLLVMIR(module, llvmCtx);
```

### Lowering Pipeline

```c++
void buildCPULoweringPipeline(OpPassManager &pm) {
  // Lower high-level dialects
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertSCFToControlFlowPass());

  // Bufferize
  bufferization::OneShotBufferizationOptions opts;
  opts.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(opts));

  // Lower to LLVM dialect
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertArithToLLVMPass());
  pm.addPass(createConvertMemRefToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}
```
