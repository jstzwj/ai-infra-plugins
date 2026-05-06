# CuTe Algorithms

## Table of Contents

- [1. Overview](#1-overview)
- [2. Copy Algorithms](#2-copy-algorithms)
- [3. Fill Algorithms](#3-fill-algorithms)
- [4. Prefetch Algorithms](#4-prefetch-algorithms)
- [5. GEMM Algorithm](#5-gemm-algorithm)
- [6. Element-wise Operations](#6-element-wise-operations)
- [7. Cooperative Versions](#7-cooperative-versions)
- [8. Algorithm Integration with Atoms and Tiled Operations](#8-algorithm-integration-with-atoms-and-tiled-operations)
- [9. Autovectorizing Copies](#9-autovectorizing-copies)
- [10. Async Copy with TMA](#10-async-copy-with-tma)
- [11. Code Examples](#11-code-examples)

---

## 1. Overview

CuTe algorithms are high-level operations that work on CuTe tensors. They provide clean, composable interfaces for common GPU operations including data movement (copy), computation (GEMM, element-wise), and memory management (fill, clear, prefetch). These algorithms are built on top of the atom system (covered in chapter 18) and automatically handle thread coordination, vectorization, and synchronization.

Key design principles of CuTe algorithms:
- **Tensor-based**: All algorithms operate on CuTe tensors, leveraging the layout system for correct indexing.
- **Atom-driven**: Algorithms use tiled atoms (TiledCopy, TiledMMA) to map operations to hardware.
- **Composable**: Algorithms can be combined to build complex kernels (copy -> GEMM -> copy).
- **Architecture-portable**: The same algorithm code works across GPU architectures by swapping the underlying atom.

Primary headers:
- `include/cute/algorithm/copy.hpp` -- Copy algorithms
- `include/cute/algorithm/fill.hpp` -- Fill and clear algorithms
- `include/cute/algorithm/gemm.hpp` -- GEMM algorithms
- `include/cute/algorithm/prefetch.hpp` -- Prefetch algorithms
- `include/cute/algorithm/axpby.hpp` -- Element-wise operations

---

## 2. Copy Algorithms

### copy(): Basic Tensor-to-Tensor Copy

The fundamental copy operation transfers data from a source tensor to a destination tensor using a tiled copy atom.

```cpp
// Basic copy: element-by-element
template <class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);

// Tiled copy: using a TiledCopy atom
template <class TiledCopy, class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy(TiledCopy const& tiled_copy,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

The basic copy iterates over all elements and assigns:

```cpp
// Simple copy between two tensors with the same shape
auto src = make_tensor(make_gmem_ptr(d_src), make_layout(_128{}));
auto dst = make_tensor(make_smem_ptr(smem), make_layout(_128{}));
copy(src, dst);
// Copies all 128 elements from global to shared memory
```

The tiled copy uses the TiledCopy's thread layout to distribute work:

```cpp
auto tiled_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy, float>{},
    Layout<Shape<_32, _1>, Stride<_1, _0>>{},
    Layout<Shape<_1, _4>, Stride<_1, _0>>{}
);

auto thr = tiled_copy.get_slice(threadIdx.x);
auto tSrc = thr.partition_S(src);
auto tDst = thr.partition_D(dst);
copy(tiled_copy, tSrc, tDst);
```

**Copy semantics**: The source and destination must have compatible shapes. The copy is performed element-by-element (or vector-by-vector when using vectorized atoms).

### copy_if(): Predicated Copy with Mask

Conditional copy that only copies elements where the predicate is true:

```cpp
template <class PredEngine, class PredLayout,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy_if(Tensor<PredEngine, PredLayout> const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst);

// Tiled predicated copy
template <class TiledCopy,
          class PredEngine, class PredLayout,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy_if(TiledCopy const& tiled_copy,
        Tensor<PredEngine, PredLayout> const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst);
```

The predicate tensor has the same shape as src/dst and contains boolean values. Elements are only copied when `pred(coord)` is true (non-zero).

```cpp
// Create a predicate for boundary handling
auto pred = make_tensor<bool>(make_layout(make_shape(remaining_rows, remaining_cols)));
for (int i = 0; i < size(pred); ++i) {
    pred(i) = (i < valid_count);
}

// Conditional copy
copy_if(pred, src_tensor, dst_tensor);
// Only elements where pred is true are copied
```

Predicated copy is essential for handling boundary conditions in GEMM kernels where the problem dimensions are not exact multiples of the tile size.

### copy_vec(): Vectorized Copy

Performs a vectorized copy by treating elements as larger vector types:

```cpp
template <class VecType, class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
         Tensor<DstEngine, DstLayout>      & dst);
```

```cpp
// Copy using uint4 vector (16 bytes = 4 floats)
copy_vec<uint4_t>(src_tensor, dst_tensor);
// Copies 4 floats at a time using vector load/store

// Copy using uint128_t vector (16 bytes = 8 halfs)
copy_vec<uint128_t>(half_src, half_dst);
// Copies 8 halfs at a time
```

The vectorized copy requires:
- Source and destination are contiguous in the vectorized dimension
- The vector type size evenly divides the tensor size
- Proper alignment of source and destination addresses

### copy_filter(): Filtered Copy

Copies only specific elements based on a filter layout:

```cpp
template <class FilterLayout, class TiledCopy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy_filter(FilterLayout const& filter,
            TiledCopy const& tiled_copy,
            Tensor<SrcEngine, SrcLayout> const& src,
            Tensor<DstEngine, DstLayout>      & dst);
```

### copy_async(): Async Copy (SM80+)

Asynchronous copy from global memory to shared memory using hardware async copy instructions:

```cpp
// Async copy using SM80 cp.async
template <class TiledCopy, class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
copy_async(TiledCopy const& tiled_copy,
           Tensor<SrcEngine, SrcLayout> const& src,
           Tensor<DstEngine, DstLayout>      & dst);
```

```cpp
// Create tiled copy with async atom
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<half_t>, half_t>{},
    Layout<Shape<_128, _1>, Stride<_1, _0>>{},
    Layout<Shape<_1, _8>, Stride<_1, _0>>{}
);

// Issue async copy
copy_async(tiled_copy, tAgA, tAsA);

// Commit and wait
cp_async_fence();
cp_async_wait<0>();
__syncthreads();

// Data is now available in shared memory
```

The async copy pipeline for multi-stage GEMM:

```cpp
// Stage 0: Issue first copy
copy_async(tiled_copy, tAgA(_, _, _, 0), tAsA(_, _, _, 0));
cp_async_fence();

for (int k = 1; k < num_stages; ++k) {
    // Wait for previous stage
    cp_async_wait<1>();

    // Issue next copy
    copy_async(tiled_copy, tAgA(_, _, _, k), tAsA(_, _, _, k));
    cp_async_fence();
}

// Wait for last stage
cp_async_wait<0>();
__syncthreads();
```

---

## 3. Fill Algorithms

### fill(): Fill Tensor with Value

Sets all elements of a tensor to a specified value:

```cpp
template <class Engine, class Layout, class Scalar>
CUTE_HOST_DEVICE constexpr void
fill(Tensor<Engine, Layout>& tensor, Scalar const& value);
```

```cpp
auto tensor = make_tensor<float>(make_layout(_128{}));
fill(tensor, 0.0f);  // All 128 elements set to 0.0f

auto tensor_2d = make_tensor<float>(make_layout(make_shape(_16{}, _8{})));
fill(tensor_2d, 1.0f);  // All 128 elements set to 1.0f
```

The fill algorithm iterates over all elements and assigns the value. For register tensors, this typically compiles to a series of register initialization instructions.

### clear(): Zero-Fill Tensor

Special case of fill that sets all elements to zero:

```cpp
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr void
clear(Tensor<Engine, Layout>& tensor);
```

```cpp
auto accumulator = make_tensor<float>(make_layout(make_shape(_16{}, _8{})));
clear(accumulator);  // Set all accumulator elements to 0.0f
```

`clear()` is semantically equivalent to `fill(tensor, 0)` but may be optimized differently (e.g., using `memset` for shared memory or zero-initialization instructions for registers).

```cpp
// Common usage: clear accumulator before GEMM
auto rC = make_tensor<float>(thr_mma.partition_C(gC).layout());
clear(rC);  // Initialize accumulator to zero

// GEMM loop: rC accumulates A * B
for (int k = 0; k < K_tiles; ++k) {
    gemm(tiled_mma, rA, rB, rC);  // rC += rA * rB
}
```

---

## 4. Prefetch Algorithms

### prefetch(): Prefetch Data to Cache Hierarchy

Prefetches data from global memory into the cache hierarchy without performing an actual copy:

```cpp
template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr void
prefetch(Tensor<Engine, Layout> const& tensor);

// Tiled prefetch
template <class TiledCopy, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr void
prefetch(TiledCopy const& tiled_copy,
         Tensor<Engine, Layout> const& tensor);
```

```cpp
// Prefetch a global memory tensor to L2 cache
auto gmem_tensor = make_tensor(make_gmem_ptr(d_data), make_layout(_128{}));
prefetch(gmem_tensor);

// Prefetch with tiled copy for thread-coordinated prefetching
auto tiled_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy, float>{},
    Layout<Shape<_32, _1>, Stride<_1, _0>>{},
    Layout<Shape<_1, _4>, Stride<_1, _0>>{}
);
prefetch(tiled_copy, gmem_tensor);
```

The prefetch operation emits `__prefetch_global()` instructions that hint the hardware to load the data into L1 or L2 cache. This is useful for overlapping data movement with computation:

```cpp
// Prefetch next tile while computing current tile
for (int k = 0; k < K_tiles; ++k) {
    // Prefetch next tile
    if (k + 1 < K_tiles) {
        prefetch(tiled_copy, gA(_, _, k + 1));
    }

    // Compute current tile
    gemm(tiled_mma, rA, rB, rC);
}
```

---

## 5. GEMM Algorithm

### gemm(): Matrix Multiply Using Tiled MMA

The primary GEMM operation performs a matrix multiply-accumulate using a tiled MMA atom:

```cpp
template <class TiledMMA,
          class AEngine, class ALayout,
          class BEngine, class BLayout,
          class CEngine, class CLayout>
CUTE_HOST_DEVICE constexpr void
gemm(TiledMMA const& tiled_mma,
     Tensor<AEngine, ALayout> const& A,
     Tensor<BEngine, BLayout> const& B,
     Tensor<CEngine, CLayout>      & C);
```

The GEMM performs: `C += A * B` (accumulate into C).

```cpp
// Standard GEMM usage
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
    Layout<Shape<_2, _2, _1>>{}
);

// Allocate and clear accumulator
auto rC = make_tensor<float>(thr_mma.partition_C(gC).layout());
clear(rC);

// GEMM loop over K dimension
for (int k_tile = 0; k_tile < K / K_TILE; ++k_tile) {
    // Load A and B fragments for this K tile
    copy(tiled_copy, tAgA(_, _, _, k_tile), rA);
    copy(tiled_copy, tBgB(_, _, _, k_tile), rB);

    // Perform MMA
    gemm(tiled_mma, rA, rB, rC);
    // rC += rA * rB (accumulates over k_tiles)
}

// Store result
copy(rC, tCrC);
```

### gemm() with Untiled Tensors

For simple cases, gemm can be called without explicit tiling:

```cpp
// Simple GEMM on small matrices (single thread)
auto A = make_tensor(make_gmem_ptr(ptr_A), make_layout(make_shape(_4{}, _4{})));
auto B = make_tensor(make_gmem_ptr(ptr_B), make_layout(make_shape(_4{}, _4{})));
auto C = make_tensor<float>(make_layout(make_shape(_4{}, _4{})));
clear(C);

gemm(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{}, A, B, C);
```

### gemm_rs(): Register-to-Shared GEMM

A specialized GEMM variant where one operand is in registers and another is in shared memory:

```cpp
template <class TiledMMA,
          class AEngine, class ALayout,
          class BEngine, class BLayout,
          class CEngine, class CLayout>
CUTE_HOST_DEVICE constexpr void
gemm_rs(TiledMMA const& tiled_mma,
        Tensor<AEngine, ALayout> const& A,
        Tensor<BEngine, BLayout> const& B,
        Tensor<CEngine, CLayout>      & C);
```

This variant is used when:
- The A operand has already been loaded into registers
- The B operand is in shared memory (or vice versa)
- The MMA instruction supports direct shared memory input (SM90 GMMA)

```cpp
// SM90 GMMA: B operand can come directly from shared memory
gemm_rs(tiled_mma, rA, sB, rC);
// rA: register tensor, sB: shared memory tensor, rC: register accumulator
```

### GEMM with Multiple K Tiles

The complete GEMM pattern with tiling:

```cpp
// Full GEMM kernel pattern
template <class TA, class TB, class TC>
__global__ void gemm_kernel(TA const* ptr_A, TB const* ptr_B, TC* ptr_C,
                             int M, int N, int K) {
    // Tiled MMA
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
        Layout<Shape<_2, _2, _1>>{}
    );

    // Global tensors
    auto gA = make_tensor(make_gmem_ptr(ptr_A), make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(ptr_B), make_layout(make_shape(K, N), make_stride(N, 1)));
    auto gC = make_tensor(make_gmem_ptr(ptr_C), make_layout(make_shape(M, N), make_stride(N, 1)));

    // Partition
    auto thr = tiled_mma.get_slice(threadIdx.x);
    auto tCrA = thr.partition_A(gA);
    auto tCrB = thr.partition_B(gB);
    auto tCrC = thr.partition_C(gC);

    // Allocate register fragments
    auto rA = make_tensor<TA>(tCrA.layout());
    auto rB = make_tensor<TB>(tCrB.layout());
    auto rC = make_tensor<float>(tCrC.layout());
    clear(rC);

    // K tiling
    constexpr int K_TILE = 16;
    for (int k = 0; k < K; k += K_TILE) {
        // Load fragments
        copy(tCrA(_, _, k), rA);
        copy(tCrB(_, _, k), rB);

        // MMA
        gemm(tiled_mma, rA, rB, rC);
    }

    // Store result
    copy(rC, tCrC);
}
```

---

## 6. Element-wise Operations

### axpby(): Y = alpha*X + beta*Y

Scaled addition of two tensors:

```cpp
template <class ScalarA, class AEngine, class ALayout,
          class ScalarB, class BEngine, class BLayout>
CUTE_HOST_DEVICE constexpr void
axpby(ScalarA const& alpha,
      Tensor<AEngine, ALayout> const& X,
      ScalarB const& beta,
      Tensor<BEngine, BLayout>      & Y);
```

```cpp
auto X = make_tensor<float>(make_layout(_128{}));
auto Y = make_tensor<float>(make_layout(_128{}));
fill(X, 2.0f);
fill(Y, 3.0f);

axpby(1.5f, X, 0.5f, Y);
// Y[i] = 1.5 * X[i] + 0.5 * Y[i] = 1.5 * 2.0 + 0.5 * 3.0 = 4.5
```

This is the standard GEMM epilogue operation for combining the GEMM result with existing data:
- `alpha` scales the new GEMM result (X)
- `beta` scales the existing data (Y)
- Common in fused GEMM + bias/activation patterns

### axmy(): Y = alpha*X*Y

Element-wise scaled multiplication:

```cpp
template <class ScalarA, class AEngine, class ALayout,
          class BEngine, class BLayout>
CUTE_HOST_DEVICE constexpr void
axmy(ScalarA const& alpha,
     Tensor<AEngine, ALayout> const& X,
     Tensor<BEngine, BLayout>      & Y);
```

```cpp
auto X = make_tensor<float>(make_layout(_128{}));
auto Y = make_tensor<float>(make_layout(_128{}));
fill(X, 2.0f);
fill(Y, 3.0f);

axmy(0.5f, X, Y);
// Y[i] = 0.5 * X[i] * Y[i] = 0.5 * 2.0 * 3.0 = 3.0
```

### Additional Element-wise Operations

CuTe also provides:

```cpp
// Element-wise addition: Y = X + Y
template <class AEngine, class ALayout, class BEngine, class BLayout>
CUTE_HOST_DEVICE constexpr void
add(Tensor<AEngine, ALayout> const& X,
    Tensor<BEngine, BLayout>      & Y);

// Element-wise subtraction: Y = X - Y
template <class AEngine, class ALayout, class BEngine, class BLayout>
CUTE_HOST_DEVICE constexpr void
sub(Tensor<AEngine, ALayout> const& X,
    Tensor<BEngine, BLayout>      & Y);

// Element-wise multiplication: Y = X * Y
template <class AEngine, class ALayout, class BEngine, class BLayout>
CUTE_HOST_DEVICE constexpr void
mul(Tensor<AEngine, ALayout> const& X,
    Tensor<BEngine, BLayout>      & Y);

// Scale: X = alpha * X
template <class Scalar, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr void
scale(Scalar const& alpha, Tensor<Engine, Layout>& X);
```

---

## 7. Cooperative Versions

### cooperative_copy()

A cooperative copy that distributes work across a group of threads (typically a warp or warp group):

```cpp
template <class ThrLayout, class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr void
cooperative_copy(ThrLayout const& thr_layout,
                 Tensor<SrcEngine, SrcLayout> const& src,
                 Tensor<DstEngine, DstLayout>      & dst);
```

Unlike regular copy where each thread works on its own partition, cooperative_copy explicitly coordinates threads to work together on the data transfer. This is useful for:

- Shared memory to register transfers that benefit from coalesced access
- Register to shared memory transfers that need bank-conflict-free patterns
- Cross-warp data exchanges

```cpp
// Cooperative copy: 128 threads copy 128x32 elements from shared to register
auto thr_layout = Layout<Shape<_128, _1>, Stride<_1, _0>>{};
cooperative_copy(thr_layout, smem_tensor, reg_tensor);
```

### cooperative_gemm()

A cooperative GEMM that distributes the matrix multiply across a group of threads:

```cpp
template <class ThrLayout, class TiledMMA,
          class AEngine, class ALayout,
          class BEngine, class BLayout,
          class CEngine, class CLayout>
CUTE_HOST_DEVICE constexpr void
cooperative_gemm(ThrLayout const& thr_layout,
                 TiledMMA const& tiled_mma,
                 Tensor<AEngine, ALayout> const& A,
                 Tensor<BEngine, BLayout> const& B,
                 Tensor<CEngine, CLayout>      & C);
```

Cooperative GEMM is used when:
- The MMA tile is larger than what a single warp can handle
- Multiple warps need to cooperate on a single tile
- Thread block clusters share data (SM90+)

```cpp
// Cooperative GEMM across a warp group (128 threads)
auto thr_layout = Layout<Shape<_128, _1>, Stride<_1, _0>>{};
cooperative_gemm(thr_layout, tiled_mma, rA, rB, rC);
```

### Cooperative GEMM with Pipeline

For SM90+ with pipeline support:

```cpp
// SM90 warp-specialized cooperative GEMM
// Producer warp group: load data via TMA
// Consumer warp group: compute GEMM

__global__ void ws_gemm(/* ... */) {
    // Producer: issue TMA copies
    if (producer_warp_group()) {
        for (int k = 0; k < K_tiles; ++k) {
            copy_async(tma_copy, gA_tile, sA_tile);
            copy_async(tma_copy, gB_tile, sB_tile);
            cp_async_fence();
        }
    }

    // Consumer: perform GEMM
    if (consumer_warp_group()) {
        for (int k = 0; k < K_tiles; ++k) {
            wait_pipeline(k);
            gemm(tiled_mma, rA, rB, rC);
        }
    }
}
```

---

## 8. Algorithm Integration with Atoms and Tiled Operations

CuTe algorithms are designed to work seamlessly with the atom system. The integration pattern is:

### Step 1: Define Atoms

```cpp
// MMA atom for the hardware
auto mma_atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{};

// Copy atom for data movement
auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<half_t>, half_t>{};
```

### Step 2: Tile Atoms

```cpp
// Tile MMA across threads
auto tiled_mma = make_tiled_mma(mma_atom, Layout<Shape<_2, _2, _1>>{});

// Tile copy across threads
auto tiled_copy = make_tiled_copy(
    copy_atom,
    Layout<Shape<_128, _1>, Stride<_1, _0>>{},
    Layout<Shape<_1, _8>, Stride<_1, _0>>{}
);
```

### Step 3: Partition Tensors

```cpp
auto thr_mma = tiled_mma.get_slice(thread_idx);
auto thr_copy = tiled_copy.get_slice(thread_idx);

// MMA partitions
auto tCrA = thr_mma.partition_A(gA);
auto tCrB = thr_mma.partition_B(gB);
auto tCrC = thr_mma.partition_C(gC);

// Copy partitions
auto tAsA = thr_copy.partition_D(sA);
auto tAgA = thr_copy.partition_S(gA);
```

### Step 4: Execute Algorithms

```cpp
// Clear accumulator
clear(rC);

// Main loop
for (int k = 0; k < K_tiles; ++k) {
    // Copy data
    copy(tiled_copy, tAgA(_, _, _, k), tAsA);
    copy(tiled_copy, tBgB(_, _, _, k), tBsB);

    // Compute
    gemm(tiled_mma, rA, rB, rC);
}

// Store result
copy(rC, tCrC);
```

### Atom Selection by Architecture

```cpp
// Architecture-conditional atom selection
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // SM90: use GMMA and TMA
    using MMAAtom = MMA_Atom<SM90_64x8x16_F16F16F32F32_TN_GMMA>;
    using CopyAtom = Copy_Atom<SM90_TMA_LOAD<half_t, 2>, half_t>;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // SM80: use MMA and cp.async
    using MMAAtom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<half_t>, half_t>;
#else
    // SM75: use basic MMA and synchronous copy
    using MMAAtom = MMA_Atom<SM75_16x8x16_F16F16F16F16_TN>;
    using CopyAtom = Copy_Atom<UniversalCopy, half_t>;
#endif
```

---

## 9. Autovectorizing Copies

CuTe can automatically vectorize copy operations when the source and destination tensors have compatible layouts.

### Auto-vectorization Conditions

A copy is auto-vectorizable when:
1. Both source and destination have stride 1 along the innermost dimension
2. The copy size is a multiple of the vector width
3. The addresses are properly aligned

### Vectorization Detection

```cpp
// CuTe checks at compile time if a copy can be vectorized
// The copy algorithm inspects the source and destination layouts
// and chooses the largest vector width that satisfies:
// - src stride == 1 in the innermost dimension
// - dst stride == 1 in the innermost dimension
// - contiguous range of elements
// - proper alignment
```

### Manual Vectorization Control

```cpp
// Force vector width using copy_vec
copy_vec<uint4_t>(src, dst);   // 16-byte vector copy
copy_vec<uint64_t>(src, dst);  // 8-byte vector copy

// Or use a vectorized copy atom
auto vec_copy = make_tiled_copy(
    Copy_Atom<UniversalCopy, uint4_t>{},  // Copy uint4_t at a time
    Layout<Shape<_32, _1>, Stride<_1, _0>>{},
    Layout<Shape<_1, _1>, Stride<_1, _0>>{}
);
```

### Vectorization for Different Types

```cpp
// FP16: 128-bit vectors = 8 elements
// FP32: 128-bit vectors = 4 elements
// INT8: 128-bit vectors = 16 elements

// The vector width is typically 128 bits (16 bytes) for global memory access
// on modern GPUs, providing maximum memory throughput.
```

---

## 10. Async Copy with TMA

### TMA Copy Overview

SM90 introduces the Tensor Memory Accelerator (TMA), a hardware unit that performs bulk tensor copies from global to shared memory without thread involvement.

### TMA Load

```cpp
// Create TMA descriptor for source tensor
auto tma_desc_A = make_tma_tensor(
    make_gmem_ptr(ptr_A),
    make_layout(make_shape(M, K), make_stride(K, 1)),
    smem_tile_shape_A
);

// TMA load from global to shared
auto tma_copy = make_tiled_copy(
    Copy_Atom<SM90_TMA_LOAD<half_t, 2>, half_t>{},
    Layout<Shape<_1>, Stride<_0>>{},  // Single thread initiates TMA
    Layout<Shape<_1>, Stride<_0>>{}
);

// Issue TMA load
copy(tma_copy, tma_gmem_tensor, smem_tensor);
```

### TMA Store

```cpp
// TMA store from shared to global
auto tma_store = make_tiled_copy(
    Copy_Atom<SM90_TMA_STORE<float, 2>, float>{},
    Layout<Shape<_1>, Stride<_0>>{},
    Layout<Shape<_1>, Stride<_0>>{}
);

copy(tma_store, smem_tensor, tma_gmem_tensor);
```

### TMA Multicast (Cluster)

For thread block clusters, TMA can multicast data to multiple shared memories:

```cpp
// Multicast TMA load: send data to all blocks in the cluster
auto tma_multicast = make_tiled_copy(
    Copy_Atom<SM90_TMA_LOAD_MULTICAST<half_t, 2>, half_t>{},
    Layout<Shape<_1>, Stride<_0>>{},
    Layout<Shape<_1>, Stride<_0>>{}
);

// The multicast mask determines which blocks receive the data
uint16_t multicast_mask = ((1 << cluster_size) - 1);
copy(tma_multicast.with(multicast_mask), tma_gmem_tensor, smem_tensor);
```

### TMA with Warp Specialization

```cpp
// SM90 warp-specialized GEMM with TMA
__global__ void ws_gemm_tma(/* ... */) {
    // Elect a leader thread to initiate TMA
    int lane_predicate = cute::elect_one();

    if (lane_predicate) {
        // Leader thread issues TMA copies
        for (int k = 0; k < K_tiles; ++k) {
            copy(tma_load, gA_tile_k, sA_tile);
            copy(tma_load, gB_tile_k, sB_tile);
        }
    }

    // All threads participate in MMA
    for (int k = 0; k < K_tiles; ++k) {
        // Wait for TMA to deliver data
        wait_barrier(k);
        gemm(tiled_mma, rA, rB, rC);
    }
}
```

---

## 11. Code Examples

### Example 1: Complete SM80 GEMM with Async Copy

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/arch/copy_sm80.hpp"
using namespace cute;

template <int BM, int BN, int BK>
__global__ void gemm_sm80(half_t const* __restrict__ ptr_A,
                           half_t const* __restrict__ ptr_B,
                           float* __restrict__ ptr_C,
                           int M, int N, int K) {
    extern __shared__ char smem_buf[];

    // Tiled MMA
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
        Layout<Shape<_2, _2, _1>>{}
    );

    // Tiled async copy
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<half_t>, half_t>{},
        Layout<Shape<_128, _1>, Stride<_1, _0>>{},
        Layout<Shape<_1, _8>, Stride<_1, _0>>{}
    );

    // Shared memory tensors
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(smem_buf)),
                          make_layout(make_shape(Int<BM>{}, Int<BK>{})));
    auto sB = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(smem_buf + BM * BK * sizeof(half_t))),
                          make_layout(make_shape(Int<BK>{}, Int<BN>{})));

    // Global tensors
    auto gA = make_tensor(make_gmem_ptr(ptr_A), make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(ptr_B), make_layout(make_shape(K, N), make_stride(N, 1)));
    auto gC = make_tensor(make_gmem_ptr(ptr_C), make_layout(make_shape(M, N), make_stride(N, 1)));

    // Partition for copy
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);
    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto tAsA = thr_copy.partition_D(sA);
    auto tBsB = thr_copy.partition_D(sB);

    // Partition for MMA
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrC = thr_mma.partition_C(gC);
    auto rC = make_tensor<float>(tCrC.layout());
    clear(rC);

    // Main GEMM loop
    for (int k = 0; k < K; k += BK) {
        // Async copy A and B
        copy_async(tiled_copy, tAgA(_, _, k), tAsA);
        copy_async(tiled_copy, tBgB(_, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Load to registers and compute
        auto rA = make_tensor<half_t>(thr_mma.partition_A(sA).layout());
        auto rB = make_tensor<half_t>(thr_mma.partition_B(sB).layout());
        copy(thr_mma.partition_A(sA), rA);
        copy(thr_mma.partition_B(sB), rB);
        gemm(tiled_mma, rA, rB, rC);
    }

    // Store result
    copy(rC, tCrC);
}
```

### Example 2: Predicated Copy for Boundary Handling

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void copy_with_bounds(float const* src, float* dst, int M, int N) {
    auto gSrc = make_tensor(make_gmem_ptr(src), make_layout(make_shape(M, N), make_stride(N, 1)));
    auto gDst = make_tensor(make_gmem_ptr(dst), make_layout(make_shape(M, N), make_stride(N, 1)));

    // Determine tile bounds
    int tile_m = blockIdx.y * 128;
    int tile_n = blockIdx.x * 64;
    int valid_m = min(128, M - tile_m);
    int valid_n = min(64, N - tile_n);

    // Create predicate
    auto pred = make_tensor<bool>(make_layout(make_shape(_128{}, _64{})));
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 64; ++j) {
            pred(make_coord(i, j)) = (i < valid_m && j < valid_n);
        }
    }

    // Predicated copy
    auto tile_src = gSrc(make_coord(tile_m + _, tile_n + _), make_coord(_, _));
    auto tile_dst = gDst(make_coord(tile_m + _, tile_n + _), make_coord(_, _));
    copy_if(pred, tile_src, tile_dst);
}
```

### Example 3: Element-wise Operations on GEMM Output

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void gemm_with_epilogue(half_t const* A, half_t const* B,
                                    float const* bias, half_t* C,
                                    float alpha, float beta,
                                    int M, int N, int K) {
    // ... (GEMM computation produces rC accumulator)

    auto rC = make_tensor<float>(make_layout(make_shape(_16{}, _8{})));
    // Assume rC is filled with GEMM results

    // Bias tensor
    auto gBias = make_tensor(make_gmem_ptr(bias), make_layout(_16{}));

    // Epilogue: C = alpha * GEMM_result + beta * bias
    auto rBias = make_tensor<float>(make_layout(_16{}));
    copy(gBias, rBias);

    // Scale GEMM result
    scale(alpha, rC);

    // Add bias (broadcast over N dimension)
    for (int n = 0; n < 8; ++n) {
        for (int m = 0; m < 16; ++m) {
            rC(make_coord(m, n)) += beta * rBias(m);
        }
    }

    // ReLU activation
    for (int i = 0; i < size(rC); ++i) {
        rC(i) = max(rC(i), 0.0f);
    }

    // Cast and store
    auto rC_half = recast<half_t>(rC);
    // ... (store to global memory)
}
```

### Example 4: SM90 TMA GEMM Pattern

```cpp
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
using namespace cute;

__global__ void gemm_sm90_tma(half_t const* A, half_t const* B, float* C,
                               int M, int N, int K) {
    // SM90 GMMA atom
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM90_64x8x16_F16F16F32F32_TN_GMMA>{},
        Layout<Shape<_1, _8, _1>>{}
    );

    // TMA copy for loading
    // (TMA descriptor creation omitted for brevity)
    auto tma_load = make_tiled_copy(
        Copy_Atom<SM90_TMA_LOAD<half_t, 2>, half_t>{},
        Layout<Shape<_1>, Stride<_0>>{},
        Layout<Shape<_1>, Stride<_0>>{}
    );

    // Warp group cooperative GEMM
    auto rC = make_tensor<float>(/* MMA partition C layout */);
    clear(rC);

    // Pipeline with TMA
    for (int k = 0; k < K / 16; ++k) {
        // TMA load (issued by leader thread)
        if (cute::elect_one()) {
            copy(tma_load, gA_tile_k, sA);
            copy(tma_load, gB_tile_k, sB);
        }

        // Wait for TMA
        // ... (pipeline wait)

        // GMMA: B directly from shared memory
        gemm_rs(tiled_mma, rA, sB, rC);
    }

    // Store result via TMA store
    copy(tma_store, rC, gC_tile);
}
```

### Example 5: Fill and Clear with Different Memory Spaces

```cpp
#include "cute/tensor.hpp"
using namespace cute;

__global__ void fill_example(float* d_data, int N) {
    extern __shared__ char smem_buf[];

    // Register tensor: clear (uses register initialization)
    auto reg = make_tensor<float>(make_layout(_128{}));
    clear(reg);
    // Compiles to register zero-initialization

    // Shared memory tensor: fill with value
    auto smem = make_tensor(
        make_smem_ptr(reinterpret_cast<float*>(smem_buf)),
        make_layout(_256{})
    );
    fill(smem, 1.0f);
    // Compiles to shared memory store instructions

    // Global memory tensor: fill with value
    auto gmem = make_tensor(
        make_gmem_ptr(d_data),
        make_layout(N)
    );
    fill(gmem, 0.0f);
    // Compiles to global memory store instructions
}
```

---

## Summary

CuTe algorithms provide a high-level interface for GPU operations:

1. **copy/copy_if/copy_async** handle data movement with predication and async support.
2. **fill/clear** initialize tensor data.
3. **prefetch** hints data into the cache hierarchy for latency hiding.
4. **gemm/gemm_rs** perform matrix multiply-accumulate using tiled MMA atoms.
5. **axpby/axmy** provide element-wise scaled arithmetic.
6. **cooperative_copy/cooperative_gemm** distribute work across thread groups.
7. **TMA copy** leverages SM90 hardware for bulk tensor transfers.
8. All algorithms integrate with the atom and tensor systems for type-safe, efficient operations.
