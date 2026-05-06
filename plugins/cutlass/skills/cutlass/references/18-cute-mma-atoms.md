# CuTe MMA Atoms and Copy Atoms

## Table of Contents

- [1. Overview](#1-overview)
- [2. MMA_Atom: Hardware MMA Operations](#2-mma_atom-hardware-mma-operations)
- [3. TiledMMA: Tiling MMA Atoms](#3-tiledmma-tiling-mma-atoms)
- [4. Fragment Generation](#4-fragment-generation)
- [5. Copy_Atom: Hardware Copy Operations](#5-copy_atom-hardware-copy-operations)
- [6. TiledCopy: Tiling Copy Atoms](#6-tiledcopy-tiling-copy-atoms)
- [7. Architecture-Specific MMA Atoms](#7-architecture-specific-mma-atoms)
- [8. Architecture-Specific Copy Atoms](#8-architecture-specific-copy-atoms)
- [9. Atom Traits and Concepts](#9-atom-traits-and-concepts)
- [10. Code Examples](#10-code-examples)

---

## 1. Overview

CuTe's atom system provides a uniform interface to hardware-accelerated operations on NVIDIA GPUs. An **atom** represents the smallest hardware operation that can be performed -- a single matrix multiply-accumulate (MMA) instruction, or a single load/store instruction. Atoms are then **tiled** across threads and data dimensions to build larger cooperative operations.

The atom system has two primary hierarchies:
- **MMA atoms**: Wrap hardware Tensor Core instructions (MMA, WMMA, GMMA, UMMA)
- **Copy atoms**: Wrap hardware memory copy instructions (cp.async, TMA, LDS/STS)

Key headers:
- `include/cute/arch/mma.hpp` -- MMA atom base
- `include/cute/arch/copy.hpp` -- Copy atom base
- `include/cute/arch/mma_sm75.hpp`, `mma_sm80.hpp`, `mma_sm90.hpp`, `mma_sm100.hpp`
- `include/cute/arch/copy_sm50.hpp`, `copy_sm80.hpp`, `copy_sm90.hpp`

---

## 2. MMA_Atom: Hardware MMA Operations

`MMA_Atom` wraps a single hardware MMA instruction and provides CuTe with the necessary type information to tile it across threads and data.

### MMA_Atom Structure

```cpp
template <class MMA_Op>
struct MMA_Atom {
    // The underlying hardware operation
    using Operation = MMA_Op;

    // The shape of the MMA operation: (M, N, K)
    using Shape_MNK = typename MMA_Op::Shape_MNK;

    // Per-thread fragment layouts for A, B, C operands
    // These describe how data is arranged in registers for the MMA instruction
    using AtomLayoutA_TV = typename MMA_Op::AtomLayoutA_TV;  // Thread x Value layout for A
    using AtomLayoutB_TV = typename MMA_Op::AtomLayoutB_TV;  // Thread x Value layout for B
    using AtomLayoutC_TV = typename MMA_Op::AtomLayoutC_TV;  // Thread x Value layout for C

    // Element types
    using ValTypeA = typename MMA_Op::ValTypeA;
    using ValTypeB = typename MMA_Op::ValTypeB;
    using ValTypeC = typename MMA_Op::ValTypeC;
    using ValTypeD = typename MMA_Op::ValTypeD;
};
```

### MMA Operation Traits

Each hardware MMA operation provides:

```cpp
struct SM80_16x8x16_F16F16F16F16_TN {
    // Shape of the MMA instruction: M=16, N=8, K=16
    using Shape_MNK = Shape<_16, _8, _16>;

    // Operand types
    using ValTypeA = half_t;    // Type of A elements
    using ValTypeB = half_t;    // Type of B elements
    using ValTypeC = half_t;    // Type of C elements (input)
    using ValTypeD = half_t;    // Type of D elements (output)

    // Thread layouts: how threads within a warp map to the MMA
    // AtomLayoutA_TV: (T, V) layout where T=threads, V=values per thread
    using AtomLayoutA_TV = Layout<Shape<_4, _8>, Stride<_1, _4>>;
    using AtomLayoutB_TV = Layout<Shape<_4, _4>, Stride<_1, _4>>;
    using AtomLayoutC_TV = Layout<Shape<_4, _4>, Stride<_1, _4>>;

    // Execute the MMA
    template <class FrgA, class FrgB, class FrgC>
    CUTE_HOST_DEVICE constexpr void
    operator()(FrgA& dA, FrgB const& dB, FrgC const& dC) const;
};
```

### MMA Atom Shape

The shape of an MMA atom `(M, N, K)` defines:
- **M**: Number of rows in the output (A rows)
- **N**: Number of columns in the output (B columns)
- **K**: Inner dimension (A columns / B rows) per MMA instruction

Common shapes:

| Atom | Shape (M, N, K) | Notes |
|---|---|---|
| SM75 MMA | 16x8x4 to 16x8x16 | Varies by data type |
| SM80 MMA | 16x8x4 to 16x8x32 | Extended shapes |
| SM90 GMMA | 16x8xK, 32x8xK, 64x8xK | K varies, async |
| SM100 UMMA | Variable | Block-scaled support |

### MMA Atom Execution

```cpp
// Direct execution of an MMA atom
auto atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{};
atom(frgA, frgB, frgC);
// frgC += frgA * frgB (matrix multiply-accumulate)
```

The fragments `frgA`, `frgB`, `frgC` must have the correct layout as specified by `AtomLayoutA_TV`, `AtomLayoutB_TV`, `AtomLayoutC_TV`.

---

## 3. TiledMMA: Tiling MMA Atoms

`TiledMMA` tiles an MMA atom across multiple threads and data dimensions to perform a larger matrix multiply.

### TiledMMA Structure

```cpp
template <class MMA_Atom, class AtomLayoutMNK, class PermutationMNK>
struct TiledMMA {
    using Atom = MMA_Atom;

    // The tile shape (total M, N, K covered by all atoms)
    using TiledShape_MNK = ...;  // Computed from atom shape and tiling

    // Thread layout: how atoms are distributed across threads
    using ThrLayoutVMNK = ...;

    // Number of threads used
    static constexpr int NumThreads = ...;
};
```

### make_tiled_mma()

Factory function to create a TiledMMA:

```cpp
template <class MMA_Atom, class AtomLayoutMNK = Layout<Shape<_1,_1,_1>>, class PermutationMNK = Tile<_1,_1,_1>>
CUTE_HOST_DEVICE constexpr auto
make_tiled_mma(MMA_Atom const& mma_atom,
               AtomLayoutMNK const& atom_layout = {},
               PermutationMNK const& permutation = {});
```

Parameters:
- `mma_atom`: The base MMA atom (e.g., `MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{}`)
- `atom_layout`: How to tile the atom across M, N, K dimensions. E.g., `Layout<Shape<_2, _2, _1>>{}` tiles 2x in M and 2x in N.
- `permutation`: Optional permutation of the M, N, K tiling.

```cpp
// Create a tiled MMA: 16x8x16 atom tiled 2x2 in M,N
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
    Layout<Shape<_2, _2, _1>>{}  // 2x in M, 2x in N, 1x in K
);
// Resulting tile shape: (32, 16, 16) = (2*16, 2*8, 1*16)
// Uses 2*2 = 4x more threads than a single atom
```

### TiledMMA Partitioning

A TiledMMA provides methods to partition input/output tensors among threads:

```cpp
auto thr_mma = tiled_mma.get_slice(thread_id);

// Partition A matrix (M x K)
auto tCrA = thr_mma.partition_A(gA);
// Returns a tensor with the thread's fragment of A

// Partition B matrix (K x N)
auto tCrB = thr_mma.partition_B(gB);
// Returns a tensor with the thread's fragment of B

// Partition C matrix (M x N) -- accumulator
auto tCrC = thr_mma.partition_C(gC);
// Returns a tensor with the thread's accumulator fragment
```

### make_tiled_mma_A/B/C()

Helper functions that create partitioned fragment tensors:

```cpp
// Create fragment for A operand
template <class TiledMMA, class TensorA>
auto make_tiled_mma_A(TiledMMA const& tiled_mma, TensorA const& tensor_A);

// Create fragment for B operand
template <class TiledMMA, class TensorB>
auto make_tiled_mma_B(TiledMMA const& tiled_mma, TensorB const& tensor_B);

// Create fragment for C operand
template <class TiledMMA, class TensorC>
auto make_tiled_mma_C(TiledMMA const& tiled_mma, TensorC const& tensor_C);
```

### Thread Layout Management

The thread layout determines which threads compute which elements:

```cpp
// AtomLayoutA_TV for SM80 16x8x16 F16:
// Layout<Shape<_4, _8>, Stride<_1, _4>>
// Means: 4 threads per row, 8 values per thread
// Thread t handles elements at offsets: t, t+4, t+8, t+12, t+16, t+20, t+24, t+28

// When tiling with Layout<Shape<_2, _2, _1>>:
// Total threads = 4 * 2 * 2 = 16 for M dimension
// Actually: atom_threads * tiles_M * tiles_N
```

### Permutation Support

The permutation parameter allows reordering the tiling dimensions:

```cpp
// Standard tiling: replicate along M then N
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
    Layout<Shape<_2, _4, _1>>{}
);

// With permutation: tile N first, then M
auto tiled_mma_perm = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
    Layout<Shape<_2, _4, _1>>{},
    Tile<_2, _1, _1>{}  // Permute: swap M and N tiling order
);
```

---

## 4. Fragment Generation

Fragments are register tensors that hold the data for MMA operands. Their layout is determined by the MMA atom's traits.

### Partition Fragments for A

```cpp
// Get thread's partition of A
auto thr_mma = tiled_mma.get_slice(thread_id);
auto tCrA = thr_mma.partition_A(gA);

// Allocate register tensor matching the partition layout
auto rA = make_tensor<half_t>(tCrA.layout());
```

The A fragment layout depends on:
- The MMA atom's `AtomLayoutA_TV` (thread-value layout within one atom)
- The tiling layout (how atoms are replicated across threads)
- The input tensor's layout

### Partition Fragments for B

```cpp
auto tCrB = thr_mma.partition_B(gB);
auto rB = make_tensor<half_t>(tCrB.layout());
```

### Partition Fragments for C (Accumulator)

```cpp
auto tCrC = thr_mma.partition_C(gC);
auto rC = make_tensor<float>(tCrC.layout());
```

The C fragment (accumulator) typically uses a wider type (float) even when inputs are half precision.

### Fragment Storage and Access Patterns

```cpp
// After partitioning, the fragment tensor has a 4D shape:
// (V0, V1, V2, tile_K) or similar depending on the tiling
// V0, V1, V2 represent the values per thread within one MMA tile
// tile_K represents iterations over the K dimension

// Access pattern during GEMM:
for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    // Load this tile's A and B data
    copy(tiled_copy, tAgA(_, _, _, k_tile), tAsA);
    copy(tiled_copy, tBgB(_, _, _, k_tile), tBsB);

    // Wait for loads to complete
    cp_async_fence();
    cp_async_wait<0>();

    // Execute MMA for this K tile
    gemm(tiled_mma, tCrA, tCrB, tCrC);
}
```

---

## 5. Copy_Atom: Hardware Copy Operations

`Copy_Atom` wraps a single hardware copy instruction (load or store) and provides the interface for tiling across threads.

### Copy_Atom Structure

```cpp
template <class CopyOp, class T>
struct Copy_Atom {
    using Operation = CopyOp;
    using ValType = T;

    // The number of elements copied per thread per instruction
    using CopyShape = ...;

    // Thread layout for the copy
    using ThrLayout = ...;
};
```

### Copy Operation Examples

```cpp
// Universal copy (simple assignment)
struct UniversalCopy {
    template <class T>
    CUTE_HOST_DEVICE constexpr void
    operator()(T& dst, T const& src) const {
        dst = src;
    }
};

// SM80 async copy (cp.async)
template <class T>
struct SM80_CP_ASYNC_COPY {
    using ValType = T;
    static constexpr int CopySize = sizeof(T);

    template <class S, class D>
    CUTE_HOST_DEVICE void operator()(S const& src, D& dst) const;
};

// SM80 cache-global copy
template <class T>
struct SM80_CP_ASYNC_CACHEGLOBAL {
    // Copy to shared memory with cache-global hint
};

// SM80 cache-always copy
template <class T>
struct SM80_CP_ASYNC_CACHEALWAYS {
    // Copy to shared memory with cache-always hint
};
```

---

## 6. TiledCopy: Tiling Copy Atoms

`TiledCopy` tiles a copy atom across threads to perform bulk data movement.

### TiledCopy Structure

```cpp
template <class Copy_Atom, class ThrLayout, class ValLayout>
struct TiledCopy {
    using Atom = Copy_Atom;

    // Thread layout: how threads map to data elements
    using ThrLayout_TV = ...;

    // Value layout: data elements per thread
    using ValLayout_TV = ...;

    // Total copy shape
    using TiledShape = ...;
};
```

### make_tiled_copy()

Factory function to create a TiledCopy:

```cpp
template <class Copy_Atom, class ThrLayout, class ValLayout>
CUTE_HOST_DEVICE constexpr auto
make_tiled_copy(Copy_Atom const& copy_atom,
                ThrLayout const& thr_layout,
                ValLayout const& val_layout);
```

```cpp
// Create a tiled copy for loading A matrix from global to shared memory
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<half_t>, half_t>{},
    Layout<Shape<_128, _1>, Stride<_1, _0>>{},   // 128 threads, 1 row
    Layout<Shape<_1, _8>, Stride<_1, _0>>{}       // 1 element per thread in M, 8 in N
);

// This tiled copy loads 128x8 elements using 128 threads
// Each thread copies 8 consecutive elements along N
```

### TiledCopy Partitioning

```cpp
// Partition source tensor for reading
auto thr_copy = tiled_copy.get_slice(thread_id);
auto tAsA = thr_copy.partition_S(gA);  // Source partition

// Partition destination tensor for writing
auto tAdA = thr_copy.partition_D(sA);  // Destination partition

// Execute the copy
copy(tiled_copy, tAsA, tAdA);
```

### Retile Functions

When the copy atom and MMA atom use different thread layouts, retile functions convert between them:

```cpp
// Retile from copy layout to MMA layout
auto tCrA_copy = tiled_copy.retile(tCrA);
// Now tCrA_copy has the layout expected by the tiled_copy
```

---

## 7. Architecture-Specific MMA Atoms

### SM75 MMA Atoms (Turing)

SM75 introduces the MMA (Matrix Multiply-Accumulate) inline assembly instructions, replacing the WMMA API with more flexible per-warp operations.

```cpp
// SM75 MMA atoms available in CuTe:

// FP16: 16x8x8 (M=16, N=8, K=8)
struct SM75_16x8x8_F16F16F16F16_TN {
    using Shape_MNK = Shape<_16, _8, _8>;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = half_t;
    using ValTypeD = half_t;
    // ...
};

// FP16: 16x8x16 (M=16, N=8, K=16) -- uses packed uint32_t loads
struct SM75_16x8x16_F16F16F16F16_TN {
    using Shape_MNK = Shape<_16, _8, _16>;
    // ...
};

// Mixed precision: FP16 inputs, FP32 accumulate
struct SM75_16x8x8_F16F16F32F32_TN {
    using Shape_MNK = Shape<_16, _8, _8>;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = float;
    using ValTypeD = float;
    // ...
};

// INT8: 16x8x16
struct SM75_16x8x16_S32S32S32S32_TN {
    using Shape_MNK = Shape<_16, _8, _16>;
    using ValTypeA = int8_t;
    using ValTypeB = int8_t;
    using ValTypeC = int32_t;
    using ValTypeD = int32_t;
    // ...
};

// INT4: 16x8x32
struct SM75_16x8x32_S32S32S32S32_TN {
    using Shape_MNK = Shape<_16, _8, _32>;
    using ValTypeA = int4b_t;
    using ValTypeB = int4b_t;
    using ValTypeC = int32_t;
    using ValTypeD = int32_t;
    // ...
};

// B1 (binary): 16x8x128
struct SM75_16x8x128_S32S32S32S32_TN {
    using Shape_MNK = Shape<_16, _8, _128>;
    using ValTypeA = uint1b_t;
    using ValTypeB = uint1b_t;
    using ValTypeC = int32_t;
    using ValTypeD = int32_t;
    // ...
};
```

SM75 MMA fragment layouts:
- A operand: 4 threads x N values per thread (along M dimension)
- B operand: 4 threads x N values per thread (along N dimension)
- C operand: 4 threads x 4 values per thread (accumulators)

### SM80 MMA Atoms (Ampere)

SM80 extends SM75 with additional data types and larger K dimensions:

```cpp
// TF32: 16x8x4 (TF32 x TF32 -> FP32)
struct SM80_16x8x4_TF32TF32F32F32_TN {
    using Shape_MNK = Shape<_16, _8, _4>;
    using ValTypeA = tfloat32_t;
    using ValTypeB = tfloat32_t;
    using ValTypeC = float;
    using ValTypeD = float;
    // ...
};

// TF32: 16x8x8
struct SM80_16x8x8_TF32TF32F32F32_TN {
    using Shape_MNK = Shape<_16, _8, _8>;
    // ...
};

// BF16: 16x8x16 (BF16 x BF16 -> FP32)
struct SM80_16x8x16_BF16BF16F32F32_TN {
    using Shape_MNK = Shape<_16, _8, _16>;
    using ValTypeA = bfloat16_t;
    using ValTypeB = bfloat16_t;
    using ValTypeC = float;
    using ValTypeD = float;
    // ...
};

// FP16: 16x8x32 (double K width)
struct SM80_16x8x32_F16F16F16F16_TN {
    using Shape_MNK = Shape<_16, _8, _32>;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = half_t;
    using ValTypeD = half_t;
    // ...
};

// FP64: 8x8x4 (FP64 x FP64 -> FP64)
struct SM80_8x8x4_F64F64F64F64_TN {
    using Shape_MNK = Shape<_8, _8, _4>;
    using ValTypeA = double;
    using ValTypeB = double;
    using ValTypeC = double;
    using ValTypeD = double;
    // ...
};

// INT8: 16x8x32
struct SM80_16x8x32_S32S32S32S32_TN {
    using Shape_MNK = Shape<_16, _8, _32>;
    using ValTypeA = int8_t;
    using ValTypeB = int8_t;
    using ValTypeC = int32_t;
    using ValTypeD = int32_t;
    // ...
};

// FP16 sparse: 16x8x16 with 2:4 sparsity on A
struct SM80_16x8x16_F16F16F16F16_TN_SP {
    using Shape_MNK = Shape<_16, _8, _16>;
    // Sparse variant: A has 2:4 structured sparsity
    // Only 8 of 16 K values are non-zero per 4-element group
};
```

SM80 introduces `OpMultiplyAddFastF32` and `OpMultiplyAddFastBF16` for reduced-precision accumulation:

```cpp
// Fast F32: TF32 with reduced mantissa for higher throughput
// Fast BF16: BF16 with reduced precision for higher throughput
```

### SM90 GMMA Atoms (Hopper)

SM90 introduces GMMA (Group MMA) instructions, which are asynchronous matrix multiply operations:

```cpp
// GMMA atoms have different characteristics:
// - Asynchronous: issued with a start instruction, result read later
// - Larger M dimensions: 16, 32, 64 rows
// - Source from register or shared memory

// FP16 GMMA: 64x8xK from register
struct SM90_64x8x16_F16F16F32F32_TN_GMMA {
    using Shape_MNK = Shape<_64, _8, _16>;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = float;
    using ValTypeD = float;
    // GMMA specific: async operation with fence
};

// FP16 GMMA: 16x8xK
struct SM90_16x8x16_F16F16F32F32_TN_GMMA {
    using Shape_MNK = Shape<_16, _8, _16>;
    // ...
};

// TF32 GMMA: 64x8x8
struct SM90_64x8x8_TF32TF32F32F32_TN_GMMA {
    using Shape_MNK = Shape<_64, _8, _8>;
    using ValTypeA = tfloat32_t;
    using ValTypeB = tfloat32_t;
    using ValTypeC = float;
    using ValTypeD = float;
};

// BF16 GMMA: 64x8x16
struct SM90_64x8x16_BF16BF16F32F32_TN_GMMA {
    using Shape_MNK = Shape<_64, _8, _16>;
    using ValTypeA = bfloat16_t;
    using ValTypeB = bfloat16_t;
    using ValTypeC = float;
    using ValTypeD = float;
};

// FP8 GMMA: 64x8x32 (FP8 E4M3 x FP8 E5M2 -> FP32)
struct SM90_64x8x32_FE4M3FE5M2F32F32_TN_GMMA {
    using Shape_MNK = Shape<_64, _8, _32>;
    using ValTypeA = float_e4m3_t;
    using ValTypeB = float_e5m2_t;
    using ValTypeC = float;
    using ValTypeD = float;
};

// FP8 GMMA: 64x8x32 (FP8 E5M2 x FP8 E4M3 -> FP32)
struct SM90_64x8x32_FE5M2FE4M3F32F32_TN_GMMA {
    using Shape_MNK = Shape<_64, _8, _32>;
    using ValTypeA = float_e5m2_t;
    using ValTypeB = float_e4m3_t;
    // ...
};

// FP64 GMMA: 8x8x4
struct SM90_8x8x4_F64F64F64F64_TN_GMMA {
    using Shape_MNK = Shape<_8, _8, _4>;
    using ValTypeA = double;
    using ValTypeB = double;
    using ValTypeC = double;
    using ValTypeD = double;
};

// INT8 GMMA: 64x8x32
struct SM90_64x8x32_S32S32S32S32_TN_GMMA {
    using Shape_MNK = Shape<_64, _8, _32>;
    using ValTypeA = int8_t;
    using ValTypeB = int8_t;
    using ValTypeC = int32_t;
    using ValTypeD = int32_t;
};
```

GMMA atoms feature:
- **Asynchronous execution**: The MMA is started with a fence, and results are available after the fence completes.
- **Warp group execution**: GMMA operates across an entire warp group (4 warps = 128 threads).
- **Larger M dimensions**: Supports 16, 32, and 64 rows in a single instruction.

### SM100 UMMA Atoms (Blackwell)

SM100 introduces UMMA (Unified MMA) with block-scaled operation support:

```cpp
// UMMA atoms support block-scaled types
// NVFP4, MXFP4, MXFP6, MXFP8 block-scaled operations

// FP16 UMMA
struct SM100_64x8x16_F16F16F32F32_TN_UMMA {
    using Shape_MNK = Shape<_64, _8, _16>;
    // UMMA variant with improved scheduling
};

// Block-scaled FP8 UMMA
struct SM100_64x8x32_FE4M3FE4M3F32F32_TN_UMMA {
    using Shape_MNK = Shape<_64, _8, _32>;
    using ValTypeA = float_e4m3_t;
    using ValTypeB = float_e4m3_t;
    // Block-scaled variant with scale factors
};

// NVFP4 block-scaled UMMA
struct SM100_64x8x256_NVFP4NVFP4F32F32_TN_UMMA {
    using Shape_MNK = Shape<_64, _8, _256>;
    // NVFP4: 4-bit floating point with block scaling
    // Scale factors applied per 32-element block
};

// MXFP8 block-scaled UMMA
struct SM100_64x8x64_MXFP8MXFP8F32F32_TN_UMMA {
    using Shape_MNK = Shape<_64, _8, _64>;
    // MX (Microscaling) FP8 with block scaling
};
```

UMMA features:
- **Block scaling**: Scale factors applied per block of elements
- **Unified interface**: Single API across all data types
- **Green context support**: Can operate in green contexts for improved resource sharing

---

## 8. Architecture-Specific Copy Atoms

### SM50 Copy Atoms (Maxwell/Pascal)

Basic copy operations using standard load/store instructions:

```cpp
// Universal copy: simple assignment, works on all architectures
struct UniversalCopy {
    template <class T>
    CUTE_HOST_DEVICE constexpr void operator()(T& dst, T const& src) const {
        dst = src;
    }
};

// Vectorized copy for SM50+
template <int N, class T>
struct VectorizedCopy {
    static_assert(N == 4 || N == 8 || N == 16, "Unsupported vector width");
    // Copies N bytes at a time using vectorized load/store
};
```

### SM75/SM80 Copy Atoms (Turing/Ampere)

SM80 introduces asynchronous copy from global to shared memory:

```cpp
// cp.async: async copy from global to shared memory
template <class T>
struct SM80_CP_ASYNC_COPY {
    using ValType = T;
    static constexpr int NumValSrc = 1;
    static constexpr int NumValDst = 1;

    template <class S, class D>
    CUTE_HOST_DEVICE void operator()(S const& src, D& dst) const {
        // Emits: cp.async.ca.shared.global [dst], [src], size;
        __cp_async(dst, src);
    }
};

// Cache-always variant: cache the source in L2
template <class T>
struct SM80_CP_ASYNC_CACHEALWAYS {
    using ValType = T;
    template <class S, class D>
    CUTE_HOST_DEVICE void operator()(S const& src, D& dst) const {
        // Emits: cp.async.ca.shared.global [dst], [src], size;
        // Hint: cache the source line in L2
    }
};

// Cache-global variant: evict from L2 after copy
template <class T>
struct SM80_CP_ASYNC_CACHEGLOBAL {
    using ValType = T;
    template <class S, class D>
    CUTE_HOST_DEVICE void operator()(S const& src, D& dst) const {
        // Emits: cp.async.cg.shared.global [dst], [src], size;
        // Hint: evict source line from L2 after copy
    }
};

// Cache-shared variant: cache in shared memory level
template <class T>
struct SM80_CP_ASYNC_CACHESHARED {
    using ValType = T;
    // ...
};
```

Async copy fence and wait:

```cpp
// Fence: commit all pending cp.async operations
CUTE_HOST_DEVICE void cp_async_fence();

// Wait: wait for at least N pending operations to complete
template <int N>
CUTE_HOST_DEVICE void cp_async_wait();

// Wait for all pending operations
// cp_async_wait<0>();
```

### SM90 TMA Copy Atoms (Hopper)

SM90 introduces TMA (Tensor Memory Accelerator) for bulk asynchronous tensor copies:

```cpp
// TMA load: copy from global to shared memory using TMA
template <class T, int Rank>
struct SM90_TMA_LOAD {
    using ValType = T;
    // TMA hardware handles the entire tile copy in one operation
    // No per-thread work; initiated by a single thread
};

// TMA store: copy from shared to global memory using TMA
template <class T, int Rank>
struct SM90_TMA_STORE {
    using ValType = T;
};

// TMA load multicast: copy to multiple shared memories (cluster)
template <class T, int Rank>
struct SM90_TMA_LOAD_MULTICAST {
    using ValType = T;
    // Multicast: sends data to multiple thread blocks in a cluster
};

// TMA prefetch: prefetch data into L2 cache
template <class T, int Rank>
struct SM90_TMA_PREFETCH {
    using ValType = T;
};

// TMA reduce: atomic reduction via TMA
template <class T, int Rank, class ReduceOp>
struct SM90_TMA_REDUCE {
    using ValType = T;
};
```

TMA copy characteristics:
- **Bulk transfer**: Copies entire tiles (up to 256 bytes per dimension) in one hardware operation.
- **Hardware-addressed**: The TMA unit handles strided access patterns without thread participation.
- **Multicast**: Can send the same data to multiple thread blocks in a cluster simultaneously.
- **Descriptor-based**: Uses a TMA descriptor (`.tma_desc`) that encodes the tensor's layout in global memory.

TMA descriptor creation:

```cpp
// Create TMA descriptor for a tensor
auto tma_desc = make_tma_tensor(
    make_gmem_ptr(d_data),
    make_layout(make_shape(M, N, K), make_stride(N*K, K, 1)),
    smem_tile_shape
);

// Use TMA descriptor for copy
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM90_TMA_LOAD<half_t, 2>, half_t>{},
    Layout<Shape<_1>, Stride<_0>>{},   // Single thread initiates
    Layout<Shape<_1>, Stride<_0>>{}    // Single value (bulk transfer)
);
```

### SM100 Copy Atoms (Blackwell)

SM100 extends TMA with additional capabilities:

```cpp
// SM100 TMA with enhanced features
template <class T, int Rank>
struct SM100_TMA_LOAD {
    using ValType = T;
    // Enhanced TMA with larger tile support
};

// SM100 TMA with accumulate
template <class T, int Rank>
struct SM100_TMA_LOAD_ACCUMULATE {
    using ValType = T;
    // Loads and accumulates into existing shared memory data
};

// SM100 TMA store with reduction
template <class T, int Rank, class ReduceOp>
struct SM100_TMA_STORE_REDUCE {
    using ValType = T;
};
```

---

## 9. Atom Traits and Concepts

### is_mma_atom

```cpp
template <class T>
struct is_mma_atom : false_type {};

template <class Op>
struct is_mma_atom<MMA_Atom<Op>> : true_type {};

template <class T>
constexpr bool is_mma_atom_v = is_mma_atom<T>::value;
```

### is_tiled_mma

```cpp
template <class T>
struct is_tiled_mma : false_type {};

template <class Atom, class Layout, class Perm>
struct is_tiled_mma<TiledMMA<Atom, Layout, Perm>> : true_type {};

template <class T>
constexpr bool is_tiled_mma_v = is_tiled_mma<T>::value;
```

### is_copy_atom

```cpp
template <class T>
struct is_copy_atom : false_type {};

template <class Op, class T>
struct is_copy_atom<Copy_Atom<Op, T>> : true_type {};

template <class T>
constexpr bool is_copy_atom_v = is_copy_atom<T>::value;
```

### is_tiled_copy

```cpp
template <class T>
struct is_tiled_copy : false_type {};

template <class Atom, class Thr, class Val>
struct is_tiled_copy<TiledCopy<Atom, Thr, Val>> : true_type {};

template <class T>
constexpr bool is_tiled_copy_v = is_tiled_copy<T>::value;
```

### Atom Shape Traits

```cpp
// Get the shape of an MMA atom
template <class MMA>
using mma_shape_mnk_t = typename MMA::Shape_MNK;

// Get the M, N, K dimensions separately
template <class MMA>
constexpr auto mma_m_v = get<0>(typename MMA::Shape_MNK{});

template <class MMA>
constexpr auto mma_n_v = get<1>(typename MMA::Shape_MNK{});

template <class MMA>
constexpr auto mma_k_v = get<2>(typename MMA::Shape_MNK{});
```

---

## 10. Code Examples

### Example 1: Basic TiledMMA Construction

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
using namespace cute;

__global__ void gemm_basic(half_t const* A, half_t const* B, float* C,
                            int M, int N, int K) {
    // Create the tiled MMA
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
        Layout<Shape<_2, _2, _1>>{}  // 2x tiling in M and N
    );
    // Tile shape: (32, 16, 16)
    // Threads: 32 (warp-level)

    // Create global tensors
    auto gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(K, N), make_stride(N, 1)));
    auto gC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(M, N), make_stride(N, 1)));

    // Partition
    auto thr = tiled_mma.get_slice(threadIdx.x);
    auto tCrA = thr.partition_A(gA);
    auto tCrB = thr.partition_B(gB);
    auto tCrC = thr.partition_C(gC);

    // Allocate fragments
    auto rA = make_tensor<half_t>(tCrA.layout());
    auto rB = make_tensor<half_t>(tCrB.layout());
    auto rC = make_tensor<float>(tCrC.layout());
    clear(rC);

    // GEMM loop
    constexpr int k_tile = decltype(mma_k_v<decltype(tiled_mma)>)::value;
    for (int k = 0; k < K; k += k_tile) {
        copy(tCrA(_, _, k), rA);
        copy(tCrB(_, _, k), rB);
        gemm(tiled_mma, rA, rB, rC);
    }

    // Store result
    copy(rC, tCrC);
}
```

### Example 2: SM90 TMA Copy with GMMA

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
using namespace cute;

__global__ void gemm_sm90(half_t const* A, half_t const* B, float* C,
                           int M, int N, int K) {
    // Tiled MMA with SM90 GMMA atom
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM90_64x8x16_F16F16F32F32_TN_GMMA>{},
        Layout<Shape<_1, _8, _1>>{}  // 8x tiling in N
    );
    // Tile shape: (64, 64, 16)

    // Tiled copy using TMA for SM90
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<SM90_TMA_LOAD<half_t, 2>, half_t>{},
        Layout<Shape<_1>, Stride<_0>>{},
        Layout<Shape<_1>, Stride<_0>>{}
    );

    // ... (full kernel would include shared memory allocation,
    //      TMA descriptor setup, pipeline, and MMA loop)
}
```

### Example 3: SM80 Async Copy with Pipeline

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/copy_sm80.hpp"
using namespace cute;

__global__ void gemm_async_copy(half_t const* d_A, half_t const* d_B,
                                 float* d_C, int M, int N, int K) {
    extern __shared__ char smem_buf[];
    half_t* sA = reinterpret_cast<half_t*>(smem_buf);
    half_t* sB = reinterpret_cast<half_t*>(smem_buf + M * K * sizeof(half_t));

    // Create tiled copy for async global-to-shared copy
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<half_t>, half_t>{},
        Layout<Shape<_128, _1>, Stride<_1, _0>>{},  // 128 threads
        Layout<Shape<_1, _8>, Stride<_1, _0>>{}     // 8 elements per thread
    );

    // Global tensor
    auto gA = make_tensor(make_gmem_ptr(d_A), make_layout(make_shape(M, K)));

    // Shared memory tensor
    auto sA_tensor = make_tensor(make_smem_ptr(sA), make_layout(make_shape(M, K)));

    // Partition
    auto thr = tiled_copy.get_slice(threadIdx.x);
    auto tAgA = thr.partition_S(gA);
    auto tAsA = thr.partition_D(sA_tensor);

    // Async copy with fence
    copy(tiled_copy, tAgA, tAsA);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Now sA_tensor is populated and can be used for MMA
}
```

### Example 4: TiledCopy Partitioning and Execution

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void copy_example(float const* src, float* dst, int M, int N) {
    // Source tensor (global memory)
    auto gSrc = make_tensor(make_gmem_ptr(src), make_layout(make_shape(M, N)));

    // Destination tensor (global memory)
    auto gDst = make_tensor(make_gmem_ptr(dst), make_layout(make_shape(M, N)));

    // Create tiled copy
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy, float>{},
        Layout<Shape<_32, _4>, Stride<_4, _1>>{},   // 128 threads in 32x4 layout
        Layout<Shape<_1, _1>, Stride<_1, _1>>{}     // 1 element per thread per dim
    );

    // Partition source and destination
    auto thr = tiled_copy.get_slice(threadIdx.x);
    auto tSrc = thr.partition_S(gSrc);
    auto tDst = thr.partition_D(gDst);

    // Execute copy
    copy(tiled_copy, tSrc, tDst);
}
```

### Example 5: Multiple Data Types with TiledMMA

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
using namespace cute;

// BF16 GEMM with FP32 accumulation
__global__ void gemm_bf16(bfloat16_t const* A, bfloat16_t const* B, float* C,
                           int M, int N, int K) {
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_BF16BF16F32F32_TN>{},
        Layout<Shape<_2, _2, _1>>{}
    );

    auto gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(K, N), make_stride(N, 1)));
    auto gC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(M, N), make_stride(N, 1)));

    auto thr = tiled_mma.get_slice(threadIdx.x);
    auto tCrC = thr.partition_C(gC);
    auto rC = make_tensor<float>(tCrC.layout());
    clear(rC);

    // ... (GEMM loop)

    copy(rC, tCrC);
}

// TF32 GEMM with FP32 accumulation
__global__ void gemm_tf32(tfloat32_t const* A, tfloat32_t const* B, float* C,
                           int M, int N, int K) {
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x8_TF32TF32F32F32_TN>{},
        Layout<Shape<_2, _2, _1>>{}
    );

    // ... (similar pattern as above)
}
```

### Example 6: Sparse MMA Atom

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
using namespace cute;

__global__ void sparse_gemm(half_t const* A, half_t const* B, float* C,
                             int M, int N, int K) {
    // Sparse MMA atom: A has 2:4 structured sparsity
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN_SP>{},
        Layout<Shape<_2, _2, _1>>{}
    );

    // A is stored in sparse format: K/2 values per row
    // Metadata indicates which 2 of every 4 elements are non-zero
    auto gA = make_tensor(make_gmem_ptr(A),
        make_layout(make_shape(M, K / 2), make_stride(K / 2, 1)));

    // B is stored in dense format
    auto gB = make_tensor(make_gmem_ptr(B),
        make_layout(make_shape(K, N), make_stride(N, 1)));

    // C accumulator
    auto gC = make_tensor(make_gmem_ptr(C),
        make_layout(make_shape(M, N), make_stride(N, 1)));

    // ... (GEMM loop with sparse A)
}
```

---

## Summary

The CuTe atom system provides a layered abstraction for GPU operations:

1. **MMA_Atom** wraps hardware Tensor Core instructions with per-thread fragment layouts.
2. **TiledMMA** tiles MMA atoms across threads and data dimensions for cooperative matrix multiplication.
3. **Copy_Atom** wraps hardware memory copy instructions (synchronous, async, TMA).
4. **TiledCopy** tiles copy atoms for bulk data movement across threads.
5. **Architecture-specific atoms** (SM75/SM80/SM90/SM100) map directly to hardware instructions.
6. **Fragment generation** creates register tensors with layouts matching hardware expectations.
7. **Partitioning** distributes tensor elements across threads using the tiled atom's thread layout.
