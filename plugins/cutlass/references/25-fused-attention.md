# CUTLASS Reference - Chapter 25: Fused Attention (FMHA)

This reference covers Fused Multi-Head Attention (FMHA) in CUTLASS, which implements the attention mechanism as a single fused kernel for maximum performance. FMHA avoids materializing the large attention matrix (Q * K^T) in global memory by fusing the GEMM, softmax, and output GEMM into a single kernel.

---

## 25.1 Fused Multi-Head Attention Concept

### 25.1.1 The Attention Mechanism

The standard scaled dot-product attention computes:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```

Where:
- **Q (Query)**: Shape (batch, num_heads, seq_len_q, head_dim)
- **K (Key)**: Shape (batch, num_heads, seq_len_k, head_dim)
- **V (Value)**: Shape (batch, num_heads, seq_len_k, head_dim)
- **Output**: Shape (batch, num_heads, seq_len_q, head_dim)

The intermediate attention matrix `P = softmax(Q * K^T / sqrt(d))` has shape (seq_len_q, seq_len_k), which can be very large for long sequences.

### 25.1.2 Why Fused Attention?

In a naive implementation, the attention computation requires:
1. **GEMM 1**: Compute `S = Q * K^T` (materialize full attention matrix).
2. **Softmax**: Row-wise softmax on S (read and write full matrix).
3. **GEMM 2**: Compute `O = P * V` (read full attention matrix).

The problems with this approach:
- **Memory**: The S matrix requires `seq_len_q * seq_len_k * sizeof(Element)` bytes, which can be hundreds of MB for long sequences.
- **Bandwidth**: Writing and reading S to/from global memory dominates runtime.
- **Latency**: Multiple kernel launches add overhead.

Fused attention solves these by:
- **Keeping S in registers/Shared Memory**: Never materialize S in global memory.
- **Online softmax**: Compute softmax incrementally as tiles of Q*K^T arrive.
- **Single kernel launch**: All operations fused into one kernel.

### 25.1.3 Memory Savings

For a sequence length of 16K with FP16:
- **Naive**: S matrix = 16K * 16K * 2 bytes = 512 MB per attention head
- **Fused**: No S matrix in global memory = 0 MB extra

---

## 25.2 Forward Pass: GEMM-Online Softmax-GEMM Fusion

### 25.2.1 Algorithm Overview

The fused attention forward pass computes the attention output in a tiled fashion:

```
For each output tile (query tile):
  Initialize: O_tile = 0, m_i = -inf, l_i = 0

  For each K,V tile (key/value tile):
    // Step 1: Compute attention scores for this tile
    S_tile = Q_tile * K_tile^T          // GEMM

    // Step 2: Update running max and softmax normalization
    m_new = max(m_i, rowmax(S_tile))     // Track max for numerical stability
    correction = exp(m_i - m_new)         // Correction factor for old accumulations
    P_tile = exp(S_tile - m_new)          // Apply softmax numerator

    // Step 3: Update running sum and output
    l_new = correction * l_i + rowsum(P_tile)   // Update normalization sum
    O_tile = correction * O_tile + P_tile * V_tile  // GEMM + accumulation

    m_i = m_new
    l_i = l_new

  // Final normalization
  O_tile = O_tile / l_i
```

This is the **FlashAttention** algorithm, implemented in CUTLASS using CuTe for tensor operations.

### 25.2.2 Online Softmax

The online softmax is the key innovation that enables fused attention. Instead of computing the full softmax in one pass (which requires the full attention matrix), it maintains running statistics:

- **m_i**: Running maximum of the attention scores seen so far.
- **l_i**: Running sum of exp(S - m_i) seen so far.

When a new tile of scores arrives:
1. Compute the new maximum `m_new = max(m_i, max(S_tile))`.
2. Apply correction: previous contributions are rescaled by `exp(m_i - m_new)`.
3. Add new contributions from the current tile.
4. After all tiles, normalize by `l_i`.

### 25.2.3 Tiling Strategy

CUTLASS uses a two-level tiling strategy:

1. **Thread block tile**: The output tile assigned to one thread block (covers a range of query tokens and all heads in a group).
2. **Warp tile**: The sub-tile processed by one warp within the thread block.
3. **Instruction tile**: The MMA instruction shape (e.g., 16x8x16 for FP16).

The tile sizes are chosen to:
- Keep Q, K, V tiles in shared memory (fits within shared memory budget).
- Keep the attention score tile `S_tile` in registers (avoids shared memory bank conflicts).
- Maximize Tensor Core utilization for the Q*K^T and P*V GEMMs.

---

## 25.3 Backward Pass

### 25.3.1 Backward Pass Overview

The attention backward pass computes gradients with respect to Q, K, V. The computation involves:

```
Given: dO (gradient of output), Q, K, V, O (output from forward)

Step 1: Recompute S = Q * K^T / sqrt(d)       -- GEMM
Step 2: Recompute P = softmax(S)                -- Softmax
Step 3: dP = dO * V^T                           -- GEMM
Step 4: dS = P * (dP - sum(dO * O, axis=-1))    -- Element-wise
Step 5: dQ = dS * K                              -- GEMM
Step 6: dK = dS^T * Q                            -- GEMM
Step 7: dV = P^T * dO                            -- GEMM
```

This requires 5 GEMMs plus element-wise operations. CUTLASS fuses these into 2-3 kernel launches:

1. **Kernel 1**: Recompute S and P, compute dV = P^T * dO and dP = dO * V^T.
2. **Kernel 2**: Compute dS, then dQ and dK.

### 25.3.2 Gradient Checkpointing

The backward pass recomputes the attention scores `S = Q * K^T` rather than storing them from the forward pass. This trades compute for memory:

- **Memory saved**: No need to store the full attention matrix (seq_len_q * seq_len_k per head).
- **Extra compute**: One additional GEMM (Q * K^T) in the backward pass.
- **Net benefit**: Memory savings far outweigh the compute cost for long sequences.

---

## 25.4 Hopper FMHA (SM90)

### 25.4.1 Head Dimension Support

Hopper FMHA supports the following head dimensions:

| Data Type | Supported Head Dimensions | Notes |
|---|---|---|
| FP16 (half_t) | 32, 64, 128, 256 | All power-of-2 sizes |
| BF16 (bfloat16_t) | 32, 64, 128, 256 | All power-of-2 sizes |
| FP8 (e4m3) | 128, 256 | Larger heads for efficient TMA |
| FP8 (e5m2) | 128, 256 | Same as e4m3 |

### 25.4.2 TMA-Based Data Movement

Hopper FMHA uses TMA (Tensor Memory Accelerator) for loading Q, K, V tiles from global memory to shared memory:

```cpp
// TMA descriptors for Q, K, V tensors
// The TMA hardware handles:
// - Multi-dimensional tensor traversal
// - Out-of-bounds handling (padding)
// - Swizzling for shared memory bank conflict avoidance
// - Asynchronous load with completion signals
```

TMA advantages for FMHA:
- **Zero-register loads**: TMA loads data directly to shared memory without using registers.
- **Multidimensional traversal**: TMA natively supports the 4D tensor layout (batch, heads, seq, dim).
- **Overlap**: Producer warps can issue TMA loads while consumer warps compute.

### 25.4.3 Warp-Specialized Cooperative Scheduling

Hopper FMHA uses a warp-specialized cooperative scheduling model:

- **Producer warp group**: Issues TMA loads for Q, K, V tiles.
- **Consumer warp group**: Executes MMA (Q*K^T and P*V) on Tensor Cores.

The cooperative variant splits the output tile across multiple consumer warp groups:

```cpp
// Warp-specialized cooperative FMHA on Hopper
// One thread block processes multiple heads or a large query range
// Multiple warp groups cooperate on the same output tile
```

### 25.4.4 Context and Generation Modes

FMHA supports two execution modes:

**Context (Pre-fill) Mode**:
- Both seq_len_q and seq_len_k are large (e.g., processing the full prompt).
- Both Q*K^T and P*V GEMMs are compute-intensive.
- Uses full TMA + warp specialization.

**Generation (Decode) Mode**:
- seq_len_q = 1 (generating one token at a time).
- seq_len_k is large (the full KV cache).
- The Q*K^T GEMM is memory-bound (single query row against many key rows).
- Uses a different scheduling strategy optimized for single-row attention.

```cpp
// Context mode: large Q and K sequences
auto context_args = FMHA::Arguments{
    mode: cutlass::fmha::Mode::Context,
    ...
};

// Generation mode: single query token
auto generation_args = FMHA::Arguments{
    mode: cutlass::fmha::Mode::Generation,
    ...
};
```

---

## 25.5 Blackwell FMHA (SM100)

### 25.5.1 Extended Head Dimension Support

Blackwell extends FMHA with support for additional head dimensions and data types:

- **Head dimensions**: All power-of-2 sizes from 32 to 512.
- **FP8 support**: Improved FP8 attention with native UMMA instructions.
- **NVFP4 attention**: Experimental support for ultra-low precision attention.

### 25.5.2 MLA (Multi-head Latent Attention) Inference

Blackwell FMHA introduces support for MLA, a variant of attention used in DeepSeek models:

In MLA, the KV cache is compressed into a lower-dimensional latent representation:
```
Standard Attention: Q * K^T * V
MLA:               Q * (c_kv * W_k)^T * (c_kv * W_v)
     = Q * W_k^T * c_kv^T * c_kv * W_v     (weight absorption)
```

The "weight absorption" technique absorbs the KV projection matrices into the Q*K^T GEMM, enabling:
- Smaller KV cache (latent dimension << head dimension).
- Fewer memory accesses (compressed KV cache).
- Modified attention computation that is optimized for Blackwell UMMA.

### 25.5.3 Two Tiles Per Threadblock (Ping-Pong)

Blackwell FMHA uses a ping-pong strategy where each thread block processes two output tiles simultaneously:

```
Threadblock 0:
  Tile A: Load Q_0, K_0 --> Compute S_0, P_0 --> Compute O_0
  Tile B: Load Q_1, K_1 --> Compute S_1, P_1 --> Compute O_1
  (Overlapping: Load for tile B happens during compute for tile A)
```

This doubles the effective compute density per thread block and improves SM utilization.

```cpp
// Blackwell FMHA with ping-pong scheduling
using FMHA_Blackwell = cutlass::fmha::kernel::FMHA<
    cutlass::arch::Sm100,
    cutlass::fmha::KernelSchedule::PingPong,  // Two tiles per threadblock
    ElementQ, ElementK, ElementV, ElementO,
    /* HeadDim */ 128,
    /* ... other template parameters ... */
>;
```

---

## 25.6 Attention Scale and Masking

### 25.6.1 Attention Scale Factor

The attention scale factor is `1 / sqrt(d)` where `d` is the head dimension. This prevents the dot products from growing too large for large head dimensions:

```cpp
float attention_scale = 1.0f / std::sqrt((float)head_dim);

// In CUTLASS FMHA, the scale is applied during the Q*K^T GEMM:
// S = scale * Q * K^T
// This is fused into the MMA operation for efficiency.
```

### 25.6.2 Masking Types

CUTLASS FMHA supports several attention mask types:

| Mask Type | Description | Use Case |
|---|---|---|
| No mask | All positions attend to all positions | Encoder attention |
| Causal mask | Position i can only attend to positions <= i | Decoder (autoregressive) attention |
| Causal from bottom-right | Causal mask with a shifted bottom-right corner | Document masking |
| Local mask | Position i attends to positions [i - window, i + window] | Longformer-style local attention |
| Custom mask | User-provided mask tensor | Arbitrary attention patterns |

### 25.6.3 Causal Mask Implementation

The causal mask is implemented efficiently by adjusting the K tile range for each Q tile:

```
Q tile 0: attends to K tiles [0]           (only current and past positions)
Q tile 1: attends to K tiles [0, 1]
Q tile 2: attends to K tiles [0, 1, 2]
...
Q tile i: attends to K tiles [0, 1, ..., i]
```

For tiles where the mask is partially applied (the boundary tile), the mask is applied element-wise using predication:

```cpp
// Inside the FMHA kernel, for the boundary tile:
// For each element S[q, k]:
if (k > q) {
    S[q, k] = -infinity;  // Mask out future positions
}
```

### 25.6.4 Local Attention Mask

Local attention restricts each query to attend to only nearby keys:

```
For query at position q:
  Attend to keys at positions max(0, q - window_size) to min(seq_len, q + window_size)
```

This reduces the attention complexity from O(seq_len^2) to O(seq_len * window_size).

```cpp
// Local attention mask configuration
struct LocalMaskConfig {
    int window_size_left;   // How many positions to the left of q to attend
    int window_size_right;  // How many positions to the right of q to attend
};

// In FMHA arguments:
auto args = FMHA::Arguments{
    ...
    mask_type = cutlass::fmha::MaskType::Local,
    mask_config = LocalMaskConfig{.window_size_left = 256, .window_size_right = 0}
};
```

---

## 25.7 Complete FMHA Examples

### 25.7.1 Hopper FP16 FMHA Forward

```cpp
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"

// FMHA forward pass on Hopper
// Note: The exact API may vary by CUTLASS version.
// This example uses the CUTLASS 3.x FMHA interface.

template <typename ElementQ, typename ElementK, typename ElementV, typename ElementO>
struct FMHAForward {
    // Configuration constants
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockM = 128;   // Query tile size
    static constexpr int kBlockN = 64;    // Key tile size
    static constexpr int kNumWarps = 8;   // Warps per thread block

    using ElementAccumulator = float;

    // Kernel configuration using CuTe
    using TileShape_QK = cute::Shape<
        cute::Int<kBlockM>, cute::Int<kBlockN>, cute::Int<kHeadDim>
    >;

    using TileShape_PV = cute::Shape<
        cute::Int<kBlockM>, cute::Int<kHeadDim>, cute::Int<kBlockN>
    >;
};

// Launch FMHA forward
void run_fmha_forward_sm90(
    int batch_size, int num_heads,
    int seq_len_q, int seq_len_k, int head_dim,
    const cutlass::half_t* Q, int64_t stride_qb, int64_t stride_qh, int64_t stride_qm,
    const cutlass::half_t* K, int64_t stride_kb, int64_t stride_kh, int64_t stride_kn,
    const cutlass::half_t* V, int64_t stride_vb, int64_t stride_vh, int64_t stride_vn,
    cutlass::half_t* O, int64_t stride_ob, int64_t stride_oh, int64_t stride_om,
    float scale,
    cutlass::fmha::MaskType mask_type,
    cudaStream_t stream = 0
) {
    // Configure FMHA kernel arguments
    // The exact API varies by CUTLASS version
    // See examples/40_cutlass_fmha for complete working examples

    // Problem dimensions
    auto problem_shape = cute::make_shape(batch_size, num_heads, seq_len_q, seq_len_k, head_dim);

    // Stride configurations
    auto stride_Q = cute::make_stride(stride_qb, stride_qh, stride_qm, cute::Int<1>{});
    auto stride_K = cute::make_stride(stride_kb, stride_kh, stride_kn, cute::Int<1>{});
    auto stride_V = cute::make_stride(stride_vb, stride_vh, stride_vn, cute::Int<1>{});
    auto stride_O = cute::make_stride(stride_ob, stride_oh, stride_om, cute::Int<1>{});

    // Launch the FMHA kernel
    // The kernel handles: Q*K^T GEMM, online softmax, P*V GEMM, all fused
}
```

### 25.7.2 FP8 FMHA on Hopper

```cpp
// FP8 FMHA forward pass
using ElementQ = cutlass::float_e4m3_t;
using ElementK = cutlass::float_e4m3_t;
using ElementV = cutlass::float_e4m3_t;
using ElementO = cutlass::float_e4m3_t;
using ElementAccumulator = float;

// FP8 FMHA configuration
// Head dimensions: 128 or 256
// Scale factors for FP8 quantization
struct FP8FMHAConfig {
    static constexpr int kHeadDim = 128;
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;

    // Scale factors for Q, K, V (per-tensor or per-head)
    float scale_Q;
    float scale_K;
    float scale_V;

    // Attention scale
    float attention_scale;
};
```

### 25.7.3 Variable-Length Attention (Paged KV Cache)

```cpp
// Variable-length FMHA with paged KV cache
// Used for inference with batched variable-length sequences

struct PagedKVCache {
    const cutlass::half_t* k_cache;   // Paged key cache: [num_pages, page_size, num_heads, head_dim]
    const cutlass::half_t* v_cache;   // Paged value cache: [num_pages, page_size, num_heads, head_dim]
    const int* page_table;            // Page table mapping: [batch, max_num_pages_per_seq]
    const int* seq_lengths;           // Sequence lengths: [batch]
    int page_size;                    // Number of tokens per page (e.g., 16)
    int num_pages;                    // Total number of pages in the cache
};

void run_paged_attention(
    int batch_size, int num_heads, int head_dim,
    int seq_len_q, int max_seq_len_k,
    const cutlass::half_t* Q,
    const PagedKVCache& kv_cache,
    cutlass::half_t* O,
    float scale,
    cudaStream_t stream = 0
) {
    // Paged attention uses a page table to access non-contiguous KV cache
    // CUTLASS FMHA supports this through the paged KV cache interface
    // Each sequence can have a different length, mapped through the page table
}
```

---

## 25.8 Performance Optimization Tips

### 25.8.1 Tile Size Selection

The tile sizes (kBlockM, kBlockN) significantly impact FMHA performance:

| kBlockM | kBlockN | Shared Memory | Best For |
|---|---|---|---|
| 64 | 64 | Small | Short sequences, many heads |
| 128 | 64 | Medium | Balanced workloads |
| 128 | 128 | Large | Long sequences, compute-bound |
| 256 | 64 | Medium | Very long query sequences |

Guidelines:
- **Compute-bound** (long sequences): Use larger kBlockM (128-256) for better Tensor Core utilization.
- **Memory-bound** (short sequences, decode): Use smaller kBlockM (64) to reduce wasted computation.
- **Shared memory budget**: The Q, K, V tiles must fit in shared memory. With FP16 and head_dim=128:
  - kBlockM=128, kBlockN=64: Q tile = 128*128*2 = 32KB, K tile = 64*128*2 = 16KB, V tile = 64*128*2 = 16KB = 64KB total.

### 25.8.2 Kernel Selection

For Hopper (SM90):
- **Context mode**: Use `KernelTmaWarpSpecialized` or `KernelTmaWarpSpecializedCooperative`.
- **Generation mode (decode)**: Use a dedicated decode kernel optimized for single-row attention.

### 25.8.3 Quantization for FMHA

For FP8 FMHA:
- Quantize Q, K, V to FP8 before the kernel.
- Apply per-tensor or per-head scale factors.
- Keep accumulation in FP32 for accuracy.
- The softmax computation is always in FP32 (cannot quantize softmax).

```cpp
// Recommended: Per-head scaling for FP8 FMHA
// Q_scale[h] and K_scale[h] are per-head scale factors
// The effective attention scale is: (1/sqrt(d)) * Q_scale[h] * K_scale[h]
```

### 25.8.4 Avoiding Common Pitfalls

1. **Head dimension alignment**: Head dimensions should be multiples of 8 (FP16) or 16 (FP8) for optimal Tensor Core utilization.
2. **Sequence length padding**: Pad sequence lengths to multiples of the tile size to avoid wasted tail tiles.
3. **Batch size**: Small batch sizes may not fully utilize the GPU. Consider grouping multiple heads or sequences.
4. **Attention dropout**: Dropout can be fused into FMHA but requires a random number generator state per thread block.

---

## 25.9 Attention Variants

### 25.9.1 Grouped Query Attention (GQA)

In GQA, the number of KV heads is less than the number of Q heads. Multiple Q heads share the same K and V:

```
Q heads:   [0, 1, 2, 3, 4, 5, 6, 7]   (num_q_heads = 8)
K/V heads: [0, 0, 1, 1, 2, 2, 3, 3]   (num_kv_heads = 4, group_size = 2)
```

CUTLASS FMHA handles GQA by mapping multiple Q heads to the same KV head within the kernel:

```cpp
// GQA configuration
int num_q_heads = 8;
int num_kv_heads = 4;
int group_size = num_q_heads / num_kv_heads;  // 2

// In the FMHA kernel, each thread block processes:
// - One or more Q heads
// - The corresponding K/V head (shared by all Q heads in the group)
```

### 25.9.2 Multi-Query Attention (MQA)

MQA is a special case of GQA where there is only one KV head:

```
Q heads:   [0, 1, 2, ..., 31]   (num_q_heads = 32)
K/V heads: [0, 0, 0, ...,  0]   (num_kv_heads = 1)
```

This dramatically reduces the KV cache size and memory bandwidth for inference.

### 25.9.3 Sliding Window Attention

For very long sequences, sliding window attention restricts each query to attend to a fixed window of keys:

```cpp
// Sliding window attention configuration
struct SlidingWindowConfig {
    int window_size;  // Number of key positions to attend (left of current query)
};

// In the FMHA kernel:
// For query at position q, iterate over K tiles in range [max(0, q - window_size), q]
// This reduces computation from O(seq_len^2) to O(seq_len * window_size)
```

---

## 25.10 Summary

Fused Multi-Head Attention is one of the most important fused kernels in modern deep learning:

1. **Memory efficiency**: Avoids materializing the full attention matrix in global memory, reducing memory from O(seq_len^2) to O(seq_len).
2. **Forward pass**: Fuses Q*K^T GEMM, online softmax, and P*V GEMM into a single kernel.
3. **Backward pass**: Fuses gradient computation with recomputation of attention scores for memory efficiency.
4. **Hopper optimizations**: TMA-based data movement, warp-specialized scheduling, context and generation modes.
5. **Blackwell enhancements**: MLA support, ping-pong scheduling, extended head dimensions.
6. **Attention variants**: Causal masking, local attention, GQA, MQA, and sliding window attention.
7. **FP8 support**: Quantized attention with FP8 inputs and FP32 accumulation/softmax.
8. **Paged KV cache**: Variable-length sequence support for efficient inference serving.
