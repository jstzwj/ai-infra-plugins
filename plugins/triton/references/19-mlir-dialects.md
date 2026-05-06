# Chapter 19: MLIR Dialects

Triton defines several MLIR dialects for progressive lowering.

## Dialects Overview

| Dialect | Prefix | Purpose | Location |
|---------|--------|---------|----------|
| Triton | `tt.` | Core tensor operations | `include/triton/Dialect/Triton/IR/` |
| TritonGPU | `ttg.` | GPU-specific ops & layouts | `include/triton/Dialect/TritonGPU/IR/` |
| Gluon | `gluon.` | Low-level GPU programming | `include/triton/Dialect/Gluon/IR/` |
| TritonInstrument | `tti.` | Instrumentation | `include/triton/Dialect/TritonInstrument/IR/` |
| TritonNvidiaGPU | `ttnvgpu.` | NVIDIA-specific GPU ops | `include/triton/Dialect/TritonNvidiaGPU/IR/` |
| Proton | `proton.` | Profiling | `third_party/proton/Dialect/` |

## Triton Dialect (`tt`)

Core operations for tensor programming.

### Operations

| Operation | Description |
|-----------|-------------|
| `tt.func` | Function definition |
| `tt.return` | Return from function |
| `tt.load` | Load from memory |
| `tt.store` | Store to memory |
| `tt.dot` | Matrix multiplication |
| `tt.dot_scaled` | Scaled dot product (FP8) |
| `tt.add`, `tt.sub`, `tt.mul` | Arithmetic |
| `tt.fdiv`, `tt.fma` | Floating-point operations |
| `tt.make_range` | Create range (arange) |
| `tt.splat` | Broadcast scalar to tensor |
| `tt.expand_dims` | Add dimension |
| `tt.broadcast` | Broadcast tensor |
| `tt.reshape` | Change shape |
| `tt.permute` | Permute dimensions |
| `tt.trans` | Transpose |
| `tt.cat` | Concatenate |
| `tt.reduce` | Reduction |
| `tt.scan` | Associative scan |
| `tt.make_block_ptr` | Create block pointer |
| `tt.advance` | Advance block pointer |
| `tt.assert` | Runtime assertion |
| `tt.print` | Device print |
| `tt.atomic_add` | Atomic addition |
| `tt.atomic_max` | Atomic maximum |
| `tt.atomic_min` | Atomic minimum |
| `tt.atomic_cas` | Compare-and-swap |
| `tt.gather` | Gather operation |
| `tt.histogram` | Histogram |
| `tt.inline_asm` | Inline assembly |
| `tt.make_tensor_desc` | Create tensor descriptor |
| `tt.load_tensor_desc` | Load via descriptor |
| `tt.store_tensor_desc` | Store via descriptor |

### Types

| Type | Description |
|------|-------------|
| `tt.ptr<type>` | Pointer type |
| `tt.tensor<shape, type>` | Tensor type |
| `tt.tensordesc<shape, type>` | Tensor descriptor |
| `tt.blocked<shape>` | Blocked encoding |

## TritonGPU Dialect (`ttg`)

GPU-specific operations with layout encoding.

### Key Operations

| Operation | Description |
|-----------|-------------|
| `ttg.convert_layout` | Convert between layouts |
| `ttg.alloc_tensor` | Allocate shared memory |
| `ttg.dealloc_tensor` | Deallocate shared memory |
| `ttg.memdesc` | Memory descriptor |
| `ttg.async_wait` | Wait for async operations |
| `ttg.async_commit_group` | Commit async group |
| `ttg.global_scratch_alloc` | Allocate global scratch |

### Layout Encodings

| Encoding | Description |
|----------|-------------|
| `#ttg.blocked` | Standard blocked layout |
| `#ttg.slice` | Slice of another layout |
| `#ttg.dot_op` | Layout for dot operands |
| `#ttg.nv_mma` | NVIDIA MMA layout |
| `#ttg.shared` | Shared memory layout |
| `#ttg.swizzled_shared` | Swizzled shared memory |
| `#ttg.linear` | Generic linear layout |
| `#ttg.amd_mfma` | AMD MFMA layout |
| `#ttg.amd_wmma` | AMD WMMA layout |

## Gluon Dialect (`gluon`)

Low-level GPU programming with explicit layout control.

### Operations

- Memory allocation and management
- Layout conversion
- Warp specialization
- Barrier operations
- Async memory operations
- Architecture-specific intrinsics

## TritonInstrument Dialect (`tti`)

Instrumentation for profiling and debugging:

- FP sanitizer (FpSan)
- GSAN instrumentation
- Performance counters

## Conversion Pipeline

```
tt.func  ──→  ttg.func  ──→  llvm.func
tt.load  ──→  ttg.local_load  ──→  llvm.load
tt.dot   ──→  ttg.mma  ──→  llvm.inline_asm(nvvm.mma)
```

Each conversion adds GPU-specific information:
1. **Triton → TritonGPU:** Adds layout encodings, memory space annotations
2. **TritonGPU → LLVM:** Lowers to hardware-specific intrinsics (NVVM, ROCDL)
