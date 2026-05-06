# 14 - GPU Architecture Reference

This document provides an exhaustive reference for GPU architectures supported by FlashAttention, covering Ampere (SM80), Ada Lovelace (SM89), Hopper (SM90), Blackwell (SM100/SM110), and SM120 architectures.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Ampere SM80 Architecture (A100)](#ampere-sm80-architecture-a100)
3. [Ampere SM86/SM89 (A6000, L40, RTX 4090)](#ampere-sm86sm89)
4. [Hopper SM90 Architecture (H100)](#hopper-sm90-architecture-h100)
5. [Blackwell SM100/SM110 Architecture (B200)](#blackwell-sm100sm110-architecture-b200)
6. [SM120 Architecture Support](#sm120-architecture-support)
7. [Memory Hierarchy](#memory-hierarchy)
8. [Tensor Core Capabilities](#tensor-core-capabilities)
9. [Block Size Heuristics per Architecture](#block-size-heuristics-per-architecture)
10. [Performance Characteristics](#performance-characteristics)

---

## Architecture Overview

| Architecture | Compute Capability | GPU Examples | Key Features |
|-------------|-------------------|-------------|-------------|
| Ampere | SM80 | A100 | Async copy, 3rd-gen tensor cores |
| Ampere | SM86 | A6000, A40 | Similar to SM80, different smem limits |
| Ada Lovelace | SM89 | L40, RTX 4090 | 4th-gen tensor cores, DP4A |
| Hopper | SM90 | H100, H200 | TMA, WGMMA, clusters, dynamic smem |
| Blackwell | SM100 | B200 | UMMA, 2CTA, persistent kernels, FP8 native |
| Blackwell | SM110 | B100 | Reduced Blackwell, subset of SM100 |
| Future | SM120 | TBD | Next-generation support |

### FlashAttention Generation by Architecture

| Generation | SM80 | SM86/89 | SM90 | SM100/110 |
|-----------|------|---------|------|-----------|
| FA2 | Full support | Full support | Compatible | Compatible |
| FA3 | -- | -- | Optimized | Compatible |
| FA4 | -- | -- | Supported | Optimized |

---

## Ampere SM80 Architecture (A100)

### Streaming Multiprocessor (SM) Specifications

| Specification | Value |
|--------------|-------|
| CUDA Cores per SM | 64 FP32, 32 FP64 |
| Tensor Cores per SM | 4 (3rd generation) |
| Max Threads per SM | 2048 |
| Max Warps per SM | 64 |
| Max Thread Blocks per SM | 32 |
| Shared Memory per SM | Up to 164 KB (configurable) |
| Shared Memory per Block | Up to 163 KB |
| Registers per SM | 65536 |
| Registers per Thread Block | Max 65536 |
| L1 Cache per SM | 192 KB (unified with shared memory) |
| L2 Cache | 40 MB (A100 80GB) |
| HBM2 Bandwidth | 2039 GB/s (A100 80GB) |
| HBM2 Capacity | 80 GB (A100 80GB) |

### Third-Generation Tensor Cores

SM80 tensor cores support:

| Instruction | Shape | Input Types | Output Type |
|------------|-------|------------|-------------|
| `mma.sync` | 16x8x16 | fp16 x fp16 | fp32 |
| `mma.sync` | 16x8x16 | bf16 x bf16 | fp32 |
| `mma.sync` | 16x8x8 | tf32 x tf32 | fp32 |
| `mma.sync` | 8x8x4 | fp64 x fp64 | fp64 |
| `mma.sync` | 16x8x32 | int8 x int8 | int32 |
| `mma.sync` | 8x8x16 | int4 x int4 | int32 |

For FlashAttention, the relevant instruction is `mma.sync.aligned.m16n8k16` which multiplies a 16x16 (MxK) matrix A by a 16x8 (KxN) matrix B to produce a 16x8 (MxN) result, accumulated in fp32.

### Asynchronous Copy (cp.async)

SM80 introduced `cp.async` for asynchronous global-to-shared memory copies:

```cuda
// Commit a group of async copies
cp.async.commit_group;

// Wait for the Nth most recent commit group to complete
cp.async.wait_group <N>;
```

FlashAttention uses `cp.async` to pipeline K/V loading with computation:
- Stage 0: Issue K/V load for next block
- Stage 1: Compute QK^T with current K/V
- The `cp_async_fence()` ensures all pending async copies are committed
- `cp_async_wait<0>()` waits for all committed copies to complete

### Shared Memory Configuration

A100 shared memory can be configured with up to 164 KB per SM. FlashAttention requests expanded shared memory per block via:

```cpp
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
```

Typical smem usage for FA2 forward with hdim=128:
- 128 x 128 x 2 bytes (Q) + 2 x 64 x 128 x 2 bytes (KV) = 32 KB + 32 KB = 64 KB
- This allows 2 CTAs per SM on A100

### Memory Hierarchy on A100

```
Registers (per thread, ~255 max)
    |  <1 cycle access
    v
Shared Memory / L1 (up to 164 KB/SM)
    |  ~30 cycles
    v
L2 Cache (40 MB)
    |  ~200-300 cycles
    v
HBM2 (80 GB, 2039 GB/s)
```

### Occupancy Considerations

For FA2 with `kBlockM=128, kBlockN=64, kHeadDim=128, 4 warps`:
- Threads per block: 128
- Shared memory: ~64 KB
- Registers per thread: ~40-60
- Typical occupancy: 2 CTAs per SM = 256 threads/SM
- A100 has 108 SMs = 216 concurrent thread blocks

---

## Ampere SM86/SM89

### SM86 (A6000, A40)

| Specification | Value |
|--------------|-------|
| Shared Memory per SM | Up to 100 KB (configurable) |
| Shared Memory per Block | Up to 99 KB |
| L2 Cache | 6 MB (A6000) |
| HBM2 Bandwidth | GDDR6X, ~960 GB/s |
| SMs (A6000) | 84 |

### SM89 (L40, RTX 4090)

| Specification | Value |
|--------------|-------|
| 4th-Gen Tensor Cores | Yes |
| Shared Memory per SM | Up to 100 KB |
| L2 Cache | 72 MB (RTX 4090) |
| Memory | GDDR6X |

### Block Size Differences

SM86/SM89 have less shared memory per SM (100 KB vs 164 KB), so FlashAttention adjusts block sizes:

For hdim=128:
- **SM80**: `128 x 64` with 64 KB smem, 2 CTAs/SM
- **SM86**: `128 x 32` with 48 KB smem for non-causal, or `64 x 64` for causal
- The smaller block size on SM86 allows better occupancy despite less smem

For hdim=96:
- **SM80**: `128 x 64`
- **SM86/89**: `64 x 64` for causal (square tiles are faster), `128 x 64` for non-causal

---

## Hopper SM90 Architecture (H100)

### Streaming Multiprocessor (SM) Specifications

| Specification | Value |
|--------------|-------|
| CUDA Cores per SM | 128 FP32 |
| Tensor Cores per SM | 4 (4th generation) |
| Max Threads per SM | 2048 |
| Shared Memory per SM | Up to 228 KB |
| Shared Memory per Block | Up to 227 KB |
| Registers per SM | 65536 |
| L2 Cache | 50 MB |
| HBM3 Bandwidth | 3352 GB/s (H100 SXM) |
| HBM3 Capacity | 80 GB |
| SMs (H100 SXM) | 132 |

### Tensor Memory Accelerator (TMA)

TMA is the defining feature of SM90, enabling hardware-managed bulk memory transfers:

**Descriptor-Based Addressing**: Instead of computing memory addresses in threads, TMA uses pre-built tensor descriptors that encode the full multidimensional layout.

```cuda
// Create TMA descriptor (host side)
cudaErrort = cudaGetTmaTensorDescriptor(&desc, ...);

// Issue TMA load (device side, single thread)
cp.async.bulk.tensor.{1d|2d|3d|4d|5d}.shared::cta.global.tile
    [smem_ptr], [desc, {coordinates}], [mbarrier];
```

**Key TMA Benefits for FlashAttention**:
1. Single-thread issue: Only one thread (via `elect_one()`) needs to issue the TMA load, freeing other threads for computation
2. Multi-dimensional addressing: No manual stride computation for 2D tiles
3. Hardware swizzle: TMA can apply swizzle patterns during the transfer
4. Cluster scope: TMA can target shared memory in any CTA within a cluster

### Warpgroup MMA (WGMMA)

SM90 introduces warpgroup-level matrix multiply-accumulate:

```cuda
wgmma.mma_async.sync.aligned.m64n{N}k{k}.f32.{a_type}.{b_type}
    {    reg_or_mem}, {mem}, {reg_or_mem};
```

- A warpgroup is 4 consecutive warps (128 threads)
- WGMMA operates on 64-row tiles of the M dimension
- N dimension is configurable: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256
- K dimension: 16 for fp16/bf16, 8 for fp8
- Async execution: WGMMA results appear in registers after a fence/wait

**FA3 Usage**: The SM90 forward kernel uses WGMMA for:
- Q @ K^T computation (score matrix)
- P @ V computation (output matrix)
- Multiple warpgroups can overlap for higher throughput

### Cluster Support

SM90 allows CTAs to form clusters that share distributed shared memory:

```cuda
// Cluster launch
__cluster_dims__(cluster_dim_x, 1, 1)
kernel_name<<<grid, block, smem, stream>>>(...);
```

- CTAs in a cluster can access each other's shared memory
- `map_shared_memory` converts local smem addresses to remote CTA addresses
- Named barriers synchronize across CTAs in a cluster

FlashAttention FA3 uses clusters for:
- 2CTA cooperative attention (two CTAs cooperate on the same attention tile)
- Shared KV loading (both CTAs load the same K/V, different Q blocks)

### Dynamic Shared Memory

SM90 allows up to 227 KB of shared memory per block. This enables larger tile sizes:

For hdim=128 forward:
- `kBlockM=128, kBlockN=176`: ~224 KB smem (near the limit)
- Or `kBlockM=128, kBlockN=128`: ~160 KB smem, allowing 1 CTA per SM

### mbarrier Synchronization

SM90 introduces mbarrier objects for producer-consumer synchronization:

```cuda
// Initialize mbarrier with expected transaction count
mbarrier.init(mbar_ptr, transaction_count);

// Producer: signal expected bytes
mbarrier.arrive.expect_tx(mbar_ptr, byte_count);

// Consumer: wait for completion
mbarrier.try_wait.parity(mbar_ptr, phase);

// Producer: signal completion
mbarrier.arrive(mbar_ptr);
```

FlashAttention uses mbarriers extensively for TMA pipeline management:
- Each pipeline stage has an associated mbarrier
- The producer warp issues TMA loads with `arrive.expect_tx`
- Consumer warps (MMA) `try_wait` before reading from smem
- Phase tracking ensures correct pipeline behavior

### Hopper Forward Pipeline (SM90)

```
Time ─────────────────────────────────────────────────────────>

Producer:
  [TMA load Q] [TMA load K] [TMA load V] [TMA load K] [TMA load V] ...
                    |              |             |              |
                    v              v             v              v
Consumer:                                                             
  ..............[WGMMA QK] ..[WGMMA PV] ..[WGMMA QK] ..[WGMMA PV] ...
                ^                                    ^
                |                                    |
           cp.async_wait                        cp.async_wait
```

The pipeline double-buffers K and V (2 stages each) while Q is loaded once per tile.

---

## Blackwell SM100/SM110 Architecture (B200)

### Streaming Multiprocessor (SM) Specifications

| Specification | SM100 (B200) | SM110 (B100) |
|--------------|-------------|-------------|
| CUDA Cores per SM | 128+ FP32 | 128+ FP32 |
| Tensor Cores | 5th generation | 5th generation |
| Shared Memory per SM | ~228 KB+ | ~228 KB+ |
| HBM3e Bandwidth | ~8000 GB/s | ~4900 GB/s |
| HBM3e Capacity | 192 GB | ~96 GB |
| SMs | ~170 | ~120 |

### UMMA (Unified Matrix Multiply-Accumulate)

Blackwell introduces UMMA, which extends WGMMA with:
- FP8 native support (both E4M3 and E5M2)
- FP4 and INT4 support for maximum throughput
- Larger tile sizes
- 2CTA (cooperative) MMA instructions

### 2CTA Cooperative Attention

SM100 supports 2-CTA cluster MMA where two CTAs cooperate on a single large MMA operation:

- Both CTAs load different portions of the Q tile
- Both CTAs load the same K/V tiles
- The MMA operation spans both CTAs' shared memory
- Result is split between the two CTAs

This doubles the effective M-dimension throughput for large head dimensions.

### Persistent Kernels

SM100 supports persistent kernel mode where CTAs stay resident on SMs and pull work from a shared queue:

- A tile scheduler assigns work tiles to CTAs as they become available
- Eliminates kernel launch overhead between attention operations
- Better load balancing for variable-length sequences
- CLC (Cooperative Load-balance Cluster) scheduling mode

### FP8 Support

Native FP8 (E4M3 and E5M2) support:
- Descale factors per tensor: `Q_descale`, `K_descale`, `V_descale`
- Scaling applied during GEMM: `result = (Q * Q_descale) @ (K * K_descale)^T`
- Output can be FP8, FP16, BF16, or FP32

---

## SM120 Architecture Support

SM120 represents the next generation beyond Blackwell. FlashAttention is designed to support future architectures through:

1. **CuTeDSL (FA4)**: Kernel specification in Python using CUTLASS DSL, compiled to PTX/CUBIN at runtime
2. **Architecture-agnostic abstractions**: Tile sizes, pipeline stages, and scheduling are parameterized
3. **JIT compilation**: Kernels are compiled for the specific GPU at runtime

The CuTeDSL approach means adding SM120 support primarily requires updating the compilation backend rather than rewriting kernels.

---

## Memory Hierarchy

### Complete Memory Hierarchy

```
Level          | Latency (cycles) | Bandwidth        | Capacity
---------------|------------------|------------------|----------
Registers      | <1               | N/A              | 256/thread
Shared Memory  | ~30              | ~19 TB/s         | 228 KB/SM
L1 Cache       | ~30              | ~19 TB/s         | (unified w/ smem)
L2 Cache       | ~200-300         | ~6 TB/s          | 50 MB
HBM            | ~500-800         | 3.3 TB/s (H100)  | 80 GB
```

### FlashAttention Memory Usage

For a forward pass with batch=B, seqlen=S, heads=H, headdim=D:

**HBM Traffic (Standard Attention)**: O(B * H * S^2 * D) for reading/writing the attention matrix

**HBM Traffic (FlashAttention)**: O(B * H * S * D) by keeping attention in SRAM

**Shared Memory Usage (FA2 Forward)**:
```
smem = kBlockM * D * sizeof(elem) + 2 * kBlockN * D * sizeof(elem)
```

| Config | kBlockM | kBlockN | D | smem (KB) |
|--------|---------|---------|---|-----------|
| hdim32 | 128 | 128 | 32 | 20 |
| hdim64 | 128 | 128 | 64 | 40 |
| hdim96 | 128 | 64 | 96 | 36 |
| hdim128 | 128 | 64 | 128 | 64 |
| hdim192 | 128 | 64 | 192 | 96 |
| hdim256 | 64 | 64 | 256 | 96 |

### Register Pressure

Register usage is a key constraint. Each thread typically uses:
- **MMA accumulators**: 4 * MMA_M * MMA_N values in fp32
- **Source fragments**: Elements of Q, K, V in registers
- **Predicates and indices**: For boundary handling
- **Softmax state**: row_max, row_sum values

Typical register usage per thread:
| Phase | Registers | Notes |
|-------|-----------|-------|
| Forward peak | ~64 | acc_s + acc_o + Q fragment |
| Backward peak | ~128 | acc_s + acc_dp + acc_dk + acc_dv |

---

## Tensor Core Capabilities

### Matrix Multiply-Accumulate Instructions by Architecture

| Architecture | Instruction | MxNxK (per warp) | Types |
|-------------|------------|-------------------|-------|
| SM80 | `mma.sync` | 16x8x16 | fp16/bf16 -> fp32 |
| SM90 | `wgmma.mma_async` | 64xNxK (warpgroup) | fp16/bf16/fp8 -> fp32 |
| SM100 | UMMA | 64xNxK (2CTA) | fp16/bf16/fp8/fp4 -> fp32/fp16 |

### Effective Throughput

For fp16 matrix multiply:

| Architecture | Peak Tensor TFLOPS (SP equivalent) | Notes |
|-------------|-----------------------------------|-------|
| A100 (SM80) | 312 | mma.sync |
| H100 (SM90) | 989 | wgmma, warpgroup |
| B200 (SM100) | ~2250 | UMMA, 2CTA |

### FlashAttention Tensor Core Utilization

The tiling strategy is designed to keep tensor cores busy:

1. **Forward**: Two MMA operations per K/V block
   - QK^T: `(kBlockM, kBlockN) = (M, K) x (N, K)^T`
   - PV: `(kBlockM, kHeadDim) = (M, N) x (D, N)^T`

2. **Backward**: Five MMA operations per Q block x K/V block
   - S = Q @ K^T, dP = dO @ V, dV = P^T @ dO, dK = dS^T @ Q, dQ = dS @ K

3. **Pipeline overlap**: While MMA computes on current tiles, async copy loads the next tiles into shared memory.

---

## Block Size Heuristics per Architecture

### FA2 (SM80+) Block Sizes

| Head Dim | SM80 Block MxN | SM86/89 Block MxN | Warps | smem |
|----------|---------------|-------------------|-------|------|
| 32 | 128x128 | 128x128 | 4 | 20 KB |
| 64 | 128x128 | 128x128 | 4 | 40 KB |
| 96 | 128x64 | 64x64 (causal), 128x64 | 4 | 36 KB |
| 128 | 128x64 | 128x32, 64x64 (causal) | 4 | 48-64 KB |
| 192 | 128x64 | 128x64 | 8 | 96 KB |
| 256 | 128x64 or 64x64 | 64x64 | 4-8 | 96-128 KB |

### FA3 (SM90) Forward Block Sizes

| Head Dim | Block MxN | MmaPV_is_RS | IntraWGOverlap | smem |
|----------|-----------|-------------|----------------|------|
| <= 64 | 192x128-192 | varies | true | ~120 KB |
| <= 96 | 192x128-144 | false | true | ~150 KB |
| <= 128 | 128x128-176 | true | true | ~160 KB |
| <= 192 | 128x96-128 | true | true | ~200 KB |
| > 192 | 128x64-80 | true | true | ~210 KB |

### FA3 (SM90) Backward Warp Group Configurations

| Head Dim | num_wg | Regs/Thread | tile_m | Best tile_n |
|----------|--------|-------------|--------|-------------|
| <= 128 | 2 | 216 | 128 | 128 |
| 129-192 | 3 | 128 | 192 | 96 |

---

## Performance Characteristics

### Forward Pass Performance (H100, hdim=128, bf16)

| Sequence Length | FA2 (ms) | FA3 (ms) | Speedup vs PyTorch |
|----------------|----------|----------|-------------------|
| 512 | 0.12 | 0.09 | ~6x |
| 1024 | 0.28 | 0.21 | ~7x |
| 2048 | 0.68 | 0.49 | ~8x |
| 4096 | 1.84 | 1.32 | ~9x |
| 8192 | 5.12 | 3.68 | ~10x |
| 16384 | 17.3 | 12.4 | ~11x |

### Backward Pass Performance (H100, hdim=128, bf16)

Backward is approximately 3-4x slower than forward due to the 5-GEMM structure.

### Memory Bandwidth Utilization

FlashAttention achieves:
- **Forward**: ~80-90% of theoretical HBM bandwidth for the arithmetic intensity
- **Backward**: ~70-80% due to more complex memory access patterns

### A100 vs H100 Scaling

FA3 on H100 is approximately 2-3x faster than FA2 on A100 for the same problem, due to:
- ~1.6x higher HBM bandwidth (3352 vs 2039 GB/s)
- ~3x higher tensor core throughput (989 vs 312 TFLOPS)
- TMA reducing the overhead of memory management
- Larger effective tile sizes due to more shared memory

### Occupancy Analysis

For optimal performance, FlashAttention targets:
- **A100**: 2 CTAs per SM (limited by shared memory)
- **H100**: 1-2 CTAs per SM depending on tile size
- **B200**: 1 CTA per SM with larger tiles

The key bottleneck is shared memory: larger tiles reduce the number of CTAs per SM, but the increased tile size improves arithmetic intensity and reduces loop overhead.

### FP8 Performance

On SM100 with native FP8:
- 2x throughput compared to fp16 for tensor core operations
- Requires descale factors to maintain numerical accuracy
- Best for inference workloads where pre-quantized KV caches are available
- Forward-only support in FA3/FA4; backward not yet optimized for FP8
