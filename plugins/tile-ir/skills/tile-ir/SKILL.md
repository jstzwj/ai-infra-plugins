# Tile IR - Complete Reference Manual

> Tile IR: A Portable, Low-Level Tile Virtual Machine and Instruction Set for NVIDIA GPUs
> Specification Version: 13.2 (2026-03-11)
> Part of the CUDA Platform

## Table of Contents

1. [Introduction and Overview](#1-introduction-and-overview)
2. [Programming Model](#2-programming-model)
3. [Syntax Reference](#3-syntax-reference)
4. [Binary Format and Bytecode](#4-binary-format-and-bytecode)
5. [Type System](#5-type-system)
6. [Semantics](#6-semantics)
7. [Memory Model](#7-memory-model)
8. [Operations Reference](#8-operations-reference)
9. [Debug Information](#9-debug-information)
10. [Stability and Compatibility](#10-stability-and-compatibility)
11. [Appendix - Example Programs](#11-appendix---example-programs)

---

## Detailed Reference Files

The following reference files contain in-depth documentation for each chapter:

| File | Description |
|------|-------------|
| [references/01-introduction.md](references/01-introduction.md) | Introduction, goals, scope, and document structure |
| [references/02-programming-model.md](references/02-programming-model.md) | Tile kernels, tile grids, structured pointers, views, tiling, GEMM examples |
| [references/03-syntax.md](references/03-syntax.md) | Module structure, items, globals, kernels, types syntax |
| [references/04-binary-format.md](references/04-binary-format.md) | Bytecode primitives, file structure, sections, type encodings, operation opcodes |
| [references/05-type-system.md](references/05-type-system.md) | Element types, pointers, tensor types, tiles, views, subviews, type equivalence |
| [references/06-semantics.md](references/06-semantics.md) | Abstract machine, modules, values, tile grid, register file, execution semantics |
| [references/07-memory-model.md](references/07-memory-model.md) | Memory operations, scopes, ordering, tokens, coherence, data races, PTX interop |
| [references/08-operations-core.md](references/08-operations-core.md) | Core operations: broadcast, cat, constant, entry, extract, get_global, iota, offset, permute, reduce, reshape, scan, select |
| [references/09-operations-conversions.md](references/09-operations-conversions.md) | Type conversion operations: bitcast, exti, ftof, ftoi, itof, int_to_ptr, ptr_to_int, ptr_to_ptr, trunci |
| [references/10-operations-control-flow.md](references/10-operations-control-flow.md) | Control flow: assert, break, continue, for, if, loop, return, yield |
| [references/11-operations-memory.md](references/11-operations-memory.md) | Memory operations: join_tokens, load_ptr_tko, make_token, store_ptr_tko |
| [references/12-operations-float.md](references/12-operations-float.md) | Floating-point operations: absf, addf, atan2, ceil, cmpf, cos/cosh, divf, exp/exp2, floor, fma, log/log2, maxf/minf, mmaf, mulf, negf, pow, remf, rsqrt, sin/sinh, sqrt, subf, tan/tanh |
| [references/13-operations-integer.md](references/13-operations-integer.md) | Integer operations: absi, addi, cmpi, divi, maxi/mini, mmai, muli, mulhii, negi, ori, remi, shli, shri, subi, xori, andi |
| [references/14-operations-atomics.md](references/14-operations-atomics.md) | Atomic operations: atomic_cas_tko, atomic_rmw_tko |
| [references/15-operations-views.md](references/15-operations-views.md) | View operations: get_index_space_shape, get_tensor_shape, load_view_tko, make_partition_view, make_tensor_view, store_view_tko, assume |
| [references/16-debug-info.md](references/16-debug-info.md) | Debug info: location information, scope metadata, compile units, files, lexical blocks, subprograms |
| [references/17-stability.md](references/17-stability.md) | Platform guarantees, supported architectures, feature availability, emulation, execution guarantees |
| [references/18-appendix.md](references/18-appendix.md) | Complete example programs, operation examples, release notes |

---

## 1. Introduction and Overview

### What is Tile IR?

Tile IR is a **portable, low-level tile virtual machine and instruction set** designed for NVIDIA GPUs. Unlike PTX which models the GPU as a data-parallel SIMT (Single Instruction Multiple Thread) processor, Tile IR models the GPU as a **tile-based processor** where each logical thread (tile block) computes over partial fragments (tiles) of multi-dimensional arrays (tensors).

### Why Tile IR?

The rapid evolution of hardware features like tensor cores has increased GPU programming complexity. Since Volta, each new GPU generation introduces new hardware features requiring greater expertise. Tile IR addresses this by:

- Providing a **virtual instruction set** for native tile-based programming
- Abstracting tensor cores and their programming model for hardware innovation
- Abstracting low-level architecture details (CUDA threads, memory hierarchy)
- Enabling higher-level DSLs and compilers as code generation targets
- Maintaining **seamless interoperability** with CUDA C++ and PTX

### Key Components (in order of importance)

1. **Versioned specification** of the Tile IR abstract machine with portable bytecode
2. **Optimizing compiler** available as part of the CUDA driver and standalone toolkit
3. **MLIR dialect** for existing compilers to target Tile IR as a backend

### Design Principles

- **Explicit broadcast** - No implicit broadcast; all shape changes must be explicit
- **Distinct float/int operations** - Separate opsets for floating-point and integer
- **Explicit overflow annotations** - Overflow behavior is explicit, not implicit
- **Token-ordered memory** - Memory operations use tokens for ordering control

---

## 2. Programming Model

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Tile Kernel** | Entry point function running as N parallel copies |
| **Tile Block** | Single logical tile thread operating over tiles of data |
| **Tile Grid** | 1D/2D/3D grid of tile blocks launched in parallel |
| **Tile** | N-dimensional array of scalars (rank-0 = scalar) |
| **Tensor View** | Structured pointer with shape and stride information |
| **Partition View** | Tiled partitioning of a tensor view |
| **Token** | Abstract value for memory ordering between operations |

### Key Operations Quick Reference

```
# Get tile block coordinates
%bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

# Get grid dimensions
%dimx, %dimy, %dimz = get_num_tile_blocks : tile<i32>

# Create constants
%zero = constant <i32: 0> : tile<i32>
%f32_zero = constant <f32: 0.0> : tile<64x64xf32>

# Create range tensor
%range = iota : tile<128xi32>

# Tensor operations
%reshaped = reshape %src : tile<i32> -> tile<1xi32>
%broadcast = broadcast %src : tile<1xi32> -> tile<128xi32>
%offsets = offset %ptrs, %range : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

# Memory operations
%val, %token = load_ptr_tko weak %ptrs : tile<128xptr<f32>> -> tile<128xf32>, token
store_ptr_tko weak %ptrs, %val : tile<128xptr<f32>>, tile<128xf32> -> token

# Tensor views
%view = make_tensor_view %ptr, shape=[%M, %N], strides=[%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
%partition = make_partition_view %view : partition_view<tile=(128x64), tensor_view<?x?xf32, strides=[?,1]>>
%tile, %tok = load_view_tko weak %partition[%x, %y] : partition_view<...>, tile<i32> -> tile<128x64xf32>, token

# Compute
%result = mmaf %A, %B, %acc : tile<MxKxf16>, tile<KxNxf16>, tile<MxNxf32>
%sum = addf %a, %b rounding<nearest_even> : tile<128xf32>

# Control flow
%result = for %i in (%lo to %hi, step %step) : tile<i32>
    iter_values(%acc = %init) -> (tile<MxNxf32>) {
  %new_acc = mmaf %A, %B, %acc : ...
  continue %new_acc : tile<MxNxf32>
}
```

---

## 3. Syntax Reference

### Module Structure

```
cuda_tile.module @module_name {
    <items>*
}
```

### Items

```
<items> ::= <kernel_definition> | <global_variable_definition>
```

### Globals

```
global @name : <type> = <value>
```

### Kernels

```
entry @kernel_name(%param0: type0, %param1: type1) {
    <operation>*
}
```

### Type Syntax

```
element_type ::= f32 | f64 | i8 | i16 | i32 | i64 | bf16 | tf32 | e4m3 | e5m2
type ::= tile<shape x element_type>
shape ::= [integer_literal (x integer_literal)*]
ptr_type ::= ptr<element_type>
tensor_view ::= tensor_view<shape x element_type, strides=[stride_list]>
partition_view ::= partition_view<tile=shape, tensor_view_type>
```

---

## 4. Binary Format and Bytecode

### File Structure

```
bytecode {
  magic: "\x7FTileIR\x00"    // 8 bytes
  version: { major, minor, tag }  // 4 bytes
  sections: section[]
}
```

### Section Types

| Section | ID | Required |
|---------|-----|----------|
| String Section | 0x01 | Yes |
| Function Table | 0x02 | Yes |
| Debug Section | 0x03 | Optional |
| Constant Data | 0x04 | Optional |
| Type Section | 0x05 | Yes |
| Global Section | 0x06 | Optional |

---

## 5. Type System

### Element Types

| Type | Size | Description |
|------|------|-------------|
| `i1`, `i8`, `i16`, `i32`, `i64` | 1-64 bits | Signless integers |
| `f16`, `f32`, `f64` | 16-64 bits | IEEE floating-point |
| `bf16` | 16 bits | Brain floating-point (8 exp, 7 mantissa) |
| `tf32` | 32 bits | TensorFloat-32 (8 exp, 10 mantissa) |
| `e4m3` | 8 bits | FP8 E4M3FN format |
| `e5m2` | 8 bits | FP8 E5M2 format |

### Tensor Types

- **Tile**: `tile<MxNxKxE>` - Static shape tensor (all dimensions power of 2)
- **Tensor View**: `tensor_view<Sx...xE, strides=[...]>` - Strided view of memory
- **Partition View**: `partition_view<tile=shape, view=tv_type>` - Tiled view

---

## 6. Semantics

### Abstract Machine State

```
S = (M, B, R, G, P)
```

- **M**: Well-formed module
- **B**: Grid of tile blocks
- **R**: Per-tile-block infinite register file
- **G**: Global memory store
- **P**: Pending memory accesses

### Execution Model

1. **Initialization**: Launch creates grid of tile blocks with unique coordinates
2. **Forward Progress**: Unspecified scheduling; all blocks eventually execute
3. **Tile Block Execution**: Isolated; communicate only via global memory
4. **Termination**: Block terminates with `cuda_tile.return`

---

## 7. Memory Model

### Scopes

| Scope | Description |
|-------|-------------|
| `tile_block` | Communication within a single tile block |
| `device` | Communication within the same GPU |
| `sys` | System-wide communication |

### Memory Orderings

| Ordering | Description |
|----------|-------------|
| `weak` | No concurrent accesses assumed |
| `relaxed` | Concurrent but no happens-before |
| `release` | Establishes happens-before with acquire |
| `acquire` | Observes release ordering |
| `acq_rel` | Both release and acquire |

### Token Ordering

Tokens are abstract values for building dependencies between memory operations. Program dependencies (control flow, data dependency) do NOT provide ordering - tokens must be used explicitly.

---

## 8. Operations Reference

### Operation Categories

| Category | Operations |
|----------|------------|
| **Core** | broadcast, cat, constant, entry, extract, get_global, get_num_tile_blocks, get_tile_block_id, global, iota, module, offset, permute, reduce, reshape, scan, select |
| **Conversions** | bitcast, exti, ftof, ftoi, itof, int_to_ptr, ptr_to_int, ptr_to_ptr, trunci |
| **Control Flow** | assert, break, continue, for, if, loop, return, yield |
| **Memory** | join_tokens, load_ptr_tko, make_token, store_ptr_tko |
| **Floating-Point** | absf, addf, atan2, ceil, cmpf, cos, cosh, divf, exp, exp2, floor, fma, log, log2, maxf, minf, mmaf, mulf, negf, pow, remf, rsqrt, sin, sinh, sqrt, subf, tan, tanh |
| **Integer** | absi, addi, andi, cmpi, divi, maxi, mini, mmai, muli, mulhii, negi, ori, remi, shli, shri, subi, xori |
| **Atomics** | atomic_cas_tko, atomic_rmw_tko |
| **Views** | get_index_space_shape, get_tensor_shape, load_view_tko, make_partition_view, make_tensor_view, store_view_tko, assume |

---

## 9. Debug Information

### Location Types

- `#cuda_tile.di_loc` - File/line/column with scope metadata
- `CallSiteLoc` - Function call location

### Scope Metadata

| Type | Fields |
|------|--------|
| Compile Unit | file, is_optimized, emission_kind |
| File | name, directory |
| Lexical Block | scope, file, line, column |
| Subprogram | file, line, name, linkage_name, compile_unit |

---

## 10. Stability and Compatibility

### Supported Architectures

| Family | CC | GPUs | Since |
|--------|-----|------|-------|
| Ampere | sm_80 | A100, A30 | 13.2 |
| Ampere | sm_86 | A40, RTX 3090 | 13.2 |
| Ada | sm_89 | L40, RTX 4090 | 13.2 |
| Blackwell | sm_100 | B200 | 13.1 |
| Blackwell | sm_120 | RTX 5090 | 13.1 |

### Key Guarantees

- **Bytecode Stability**: Programs can be loaded by all conforming drivers
- **Program Portability**: Conforming programs are syntactically portable
- **CUDA Compatibility**: Respects CUDA forward/backward compatibility
- **Execution Determinism**: Deterministic within fixed toolchain/config/hardware

---

## 11. Appendix - Example Programs

### Hello World

```cuda_tile
cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}
```

### Vector Addition (128 elements)

```cuda_tile
cuda_tile.module @vector_block_add_128x1 {
    entry @vector_block_add_128x1_kernel(
        %a_ptr: tile<ptr<f32>>,
        %b_ptr: tile<ptr<f32>>,
        %c_ptr: tile<ptr<f32>>
    ) {
        %offset = iota : tile<128xi32>
        %a_base = reshape %a_ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
        %a_ptrs = broadcast %a_base : tile<1xptr<f32>> -> tile<128xptr<f32>>
        %a_tensor = offset %a_ptrs, %offset : tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>
        // ... same for b, c ...
        %a_val, %t1 = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
        %b_val, %t2 = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
        %c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
        store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token
    }
}
```

### Dynamic GEMM with Views

```cuda_tile
cuda_tile.module @gemm_kloop_module {
    entry @gemm_kloop_kernel(
        %A_ptr: tile<ptr<f16>>, %B_ptr: tile<ptr<f16>>, %C_ptr: tile<ptr<f32>>,
        %M: tile<i32>, %N: tile<i32>, %K: tile<i32>,
        %stride_ak: tile<i32>, %stride_bn: tile<i32>, %stride_cm: tile<i32>
    ) {
        // Create tensor views
        %A = make_tensor_view %A_ptr, shape=[%K, %M], strides=[%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        %B = make_tensor_view %B_ptr, shape=[%N, %K], strides=[%stride_bn, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        %C = make_tensor_view %C_ptr, shape=[%M, %N], strides=[%stride_cm, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

        // Create partition views
        %A_block = make_partition_view %A : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        %B_block = make_partition_view %B : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        %C_block = make_partition_view %C : partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>

        %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>
        %mk_len = get_index_space_shape %A_block : partition_view<...> -> tile<i32>

        // Reduction loop over K dimension
        %result = for %k in (%i0 to %mk_len#1, step %i1) : tile<i32>
            iter_values(%acc = %cst) -> (tile<128x128xf32>) {
            %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : ... -> tile<128x64xf16>, token
            %B_frag, %t2 = load_view_tko weak %B_block[%k, %bidy] : ... -> tile<64x128xf16>, token
            %acc = mmaf %A_frag, %B_frag, %acc_prev : tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
            continue %acc : tile<128x128xf32>
        }

        %t3 = store_view_tko weak %result, %C_block[%bidx, %bidy] : tile<128x128xf32>, partition_view<...>, tile<i32> -> token
    }
}
```
