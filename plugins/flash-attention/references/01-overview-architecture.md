# FlashAttention: Overview and Architecture Reference

## Table of Contents

1. [Introduction](#introduction)
2. [The Attention Problem](#the-attention-problem)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Standard Attention Implementation](#standard-attention-implementation)
5. [FlashAttention Algorithm Deep Dive](#flashattention-algorithm-deep-dive)
6. [IO Complexity Analysis](#io-complexity-analysis)
7. [Memory Complexity Analysis](#memory-complexity-analysis)
8. [Forward Pass Algorithm Step-by-Step](#forward-pass-algorithm-step-by-step)
9. [Backward Pass Algorithm Step-by-Step](#backward-pass-algorithm-step-by-step)
10. [Online Softmax](#online-softmax)
11. [Tiling Strategy](#tiling-strategy)
12. [Comparison with Standard Attention](#comparison-with-standard-attention)
13. [Repository Structure](#repository-structure)
14. [Version History](#version-history)
15. [Key Innovations Per Version](#key-innovations-per-version)
16. [Performance Characteristics on Different GPUs](#performance-characteristics-on-different-gpus)
17. [Supported Features Matrix](#supported-features-matrix)

---

## Introduction

FlashAttention is a family of IO-aware exact attention algorithms that provide
significant speedup and memory savings over standard PyTorch attention. The key
insight is that attention computation is bottlenecked by memory reads/writes
(HBM access), not by arithmetic throughput (FLOPs). By minimizing the number of
memory reads and writes to High Bandwidth Memory (HBM), FlashAttention achieves
both faster execution and lower memory footprint.

The FlashAttention project has evolved through four major versions:

- **FlashAttention-1 (FA1)**: Introduced the tiling and online softmax approach
- **FlashAttention-2 (FA2)**: Improved parallelism and work partitioning, 2x faster
- **FlashAttention-3 (FA3)**: Optimized for Hopper GPUs (H100) with async, FP8
- **FlashAttention-4 (FA4)**: Written in CuTeDSL for Hopper and Blackwell GPUs

### Key Publications

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
   - NeurIPS 2022
   - Paper: https://arxiv.org/abs/2205.14135

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - Tri Dao
   - ICLR 2024
   - Paper: https://tridao.me/publications/flash2/flash2.pdf

3. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**
   - Tri Dao, et al.
   - Paper: https://tridao.me/publications/flash3/flash3.pdf

---

## The Attention Problem

The scaled dot-product attention (SDPA) computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Where:
- `Q` (Query): shape `(batch, seqlen_q, num_heads, head_dim)`
- `K` (Key): shape `(batch, seqlen_k, num_heads_kv, head_dim)`
- `V` (Value): shape `(batch, seqlen_k, num_heads_kv, head_dim_v)`

### Dimensions and Notation

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| `N` | Sequence length | 512 - 128k |
| `d` | Head dimension | 64, 96, 128, 256 |
| `h` | Number of query heads | 8 - 128 |
| `h_kv` | Number of KV heads (GQA) | 1 - 128 |
| `B` | Batch size | 1 - 128 |

### Memory Bottleneck

The naive attention implementation materializes the full `N x N` attention
matrix `S = QK^T` in HBM. This requires `O(N^2)` memory, which:

- Limits maximum sequence length
- Causes excessive HBM reads/writes
- Results in poor arithmetic intensity (ratio of FLOPs to memory bytes transferred)

For a sequence length of 8192 with head dimension 128 and batch size 16:
- Attention matrix size: `16 * 32 * 8192 * 8192 * 2 bytes = 64 GB`
- This exceeds the HBM capacity of most GPUs

---

## Mathematical Formulation

### Standard Attention

Given query matrix `Q`, key matrix `K`, and value matrix `V`, all of head
dimension `d`:

```
S = Q @ K^T                    # (N, d) x (d, N) -> (N, N)
P = softmax(S / sqrt(d))       # (N, N) row-wise softmax
O = P @ V                      # (N, N) x (N, d) -> (N, d)
```

### FlashAttention Formulation

FlashAttention reformulates this to avoid materializing the full `N x N`
attention matrix. Instead, it processes the attention in blocks (tiles),
computing the output incrementally using an online softmax technique.

For each block of rows of `Q` (indexed by `i`), and each block of columns of
`K` and rows of `V` (indexed by `j`):

```
S_ij = Q_i @ K_j^T             # Block of attention scores
m_ij = rowmax(S_ij)            # Block-wise maximum
P_ij = exp(S_ij - m_ij)        # Numerically stable softmax numerator
l_ij = rowsum(P_ij)            # Block-wise sum for normalization
```

The running statistics are updated across all `j` blocks:

```
m_new = max(m_old, m_ij)                   # Running maximum
l_new = exp(m_old - m_new) * l_old + exp(m_ij - m_new) * l_ij  # Running sum
O_i = (exp(m_old - m_new) * l_old * O_i_old + exp(m_ij - m_new) * P_ij @ V_j) / l_new
```

After processing all `j` blocks, the final output for row block `i` is:

```
O_i = softmax(S_i / sqrt(d)) @ V
```

which is mathematically identical to the standard formulation.

### Extension to Causal Attention

For causal (autoregressive) attention, the mask ensures that position `i`
can only attend to positions `j <= i`. In FlashAttention, this is implemented
by:

1. Skipping blocks where all entries are masked (no computation needed)
2. For partially masked blocks, applying the causal mask within the kernel

The causal mask alignment in FA2+ is to the **bottom-right corner** of the
attention matrix. For `seqlen_q < seqlen_k`, the last `seqlen_q` query
positions attend causally to all key positions. For `seqlen_q > seqlen_k`,
the first `seqlen_q - seqlen_k` query positions produce zero output.

### Extension to Sliding Window Attention

Local/sliding window attention restricts each query at position `i` to attend
only to keys in the range:

```
[i + seqlen_k - seqlen_q - window_size_left, i + seqlen_k - seqlen_q + window_size_right]
```

This is controlled by the `window_size` parameter: `window_size=(-1, -1)` means
unrestricted (infinite context window).

### Extension to Softcapping

Some models (e.g., Gemma-2, Grok) use attention softcapping:

```
S = tanh(S / softcap) * softcap
```

This bounds the attention logits to `[-softcap, softcap]`, preventing
numerical instability from extremely large logits.

### Extension to ALiBi (Attention with Linear Bias)

ALiBi adds a position-dependent bias to attention scores:

```
S_ij = S_ij - alibi_slope * |i - j|
```

Where `alibi_slopes` is a per-head (or per-batch-per-head) FP32 tensor.

---

## Standard Attention Implementation

The baseline PyTorch implementation:

```python
def standard_attention(q, k, v, softmax_scale=None, causal=False):
    """
    q: (batch, seqlen_q, num_heads, head_dim)
    k: (batch, seqlen_k, num_heads_kv, head_dim)
    v: (batch, seqlen_k, num_heads_kv, head_dim)
    """
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Materialize full N x N attention matrix in HBM
    scores = torch.einsum('bshd,bthd->bhst', q, k) * softmax_scale
    # scores shape: (batch, num_heads, seqlen_q, seqlen_k)

    if causal:
        # Create causal mask
        mask = torch.ones_like(scores, dtype=torch.bool).tril(diagonal=0)
        scores = scores.masked_fill(~mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)  # Another HBM write
    output = torch.einsum('bhst,bthd->bshd', attn_weights, v)

    return output
```

**Problems with this approach:**

1. `scores` requires `O(B * h * N^2)` memory (quadratic in sequence length)
2. `attn_weights` also requires `O(B * h * N^2)` memory
3. Multiple reads/writes to HBM for intermediate results
4. Poor arithmetic intensity: memory-bound rather than compute-bound

**IO analysis of standard attention:**

| Operation | HBM Reads | HBM Writes | Total IO |
|-----------|-----------|------------|----------|
| `Q @ K^T` | `O(B*h*N*d)` read Q,K | `O(B*h*N^2)` write S | `O(B*h*N*(d+N))` |
| `softmax` | `O(B*h*N^2)` read S | `O(B*h*N^2)` write P | `O(B*h*N^2)` |
| `P @ V` | `O(B*h*N^2)` read P, `O(B*h*N*d)` read V | `O(B*h*N*d)` write O | `O(B*h*N*(N+d))` |
| **Total** | | | `O(B*h*N^2)` |

---

## FlashAttention Algorithm Deep Dive

### Core Idea: Tiling + Online Softmax

FlashAttention combines two key techniques:

1. **Tiling**: Break the Q, K, V matrices into blocks that fit in SRAM (shared
   memory/on-chip memory). Process one block of Q against all blocks of K/V.

2. **Online Softmax** (Milakov & Gimelshein, 2018; earlier by Pichai et al.):
   Compute the softmax normalization incrementally, without needing to
   materialize the full attention matrix. This requires tracking running
   maximum (`m`) and running sum (`l`) for each row.

### SRAM vs HBM

| Memory Type | Size | Bandwidth | Latency |
|-------------|------|-----------|---------|
| HBM (GPU DRAM) | 40-192 GB | 1.5-4.8 TB/s | ~100-300 ns |
| SRAM (Shared Memory) | 48-228 KB/SM | ~19 TB/s | ~1-5 ns |
| Registers | 256 KB/SM | ~100+ TB/s | <1 ns |

The key insight: SRAM is ~10x faster than HBM per byte, but ~1000x smaller.
FlashAttention's goal is to minimize HBM accesses by keeping intermediate
results in SRAM.

### Block Size Selection

The block sizes (tile dimensions) are chosen to balance:

1. **SRAM capacity**: The Q block + K block + V block + output block + running
   statistics must fit in shared memory.
2. **Computation efficiency**: Larger blocks amortize memory access overhead.
3. **GPU architecture**: Different architectures have different shared memory
   sizes and warp-level instruction requirements.

For FlashAttention-2, the block size N heuristic (`_get_block_size_n`):

| Head Dimension | SM80 (A100) | SM8x (Consumer Ampere) | SM90 (Hopper) |
|----------------|-------------|------------------------|---------------|
| <= 32 | 128 | 128 | 128 |
| <= 64 | 128 (64 w/ dropout) | 128 (64 w/ dropout) | 128 (64 w/ dropout) |
| <= 96 | 64 | 64 | 64 |
| <= 128 | 64 (32 w/ dropout) | 64/32 causal/dropout | 64 (32 w/ dropout) |
| <= 192 | 64 | 64 | 64 |
| <= 256 | 64 | 64 | 64 |

For FlashAttention-4 CuTeDSL, forward tile sizes vary by architecture:

**SM90 Forward Configs:**

| Head Dimension | Tile (M x N) | RS | Overlap | Notes |
|----------------|--------------|-----|---------|-------|
| <= 64 | 192 x 128 | Yes | Yes | Best across seqlens |
| <= 96 | 192 x 144 (non-causal) | No | Yes | RS catastrophic at 192 |
| <= 96 | 192 x 128 (causal/local) | No | Yes | Slightly better short seqlen |
| <= 128 | 128 x 128 | Yes | Yes | |
| <= 192 | 128 x 128/112 | Yes | Yes | 96 if local |
| <= 256 | 128 x 64/80 | Yes | Yes | 64 if local |

**SM100 Forward Configs:**

| Head Dimension | Tile (M x N) | Features |
|----------------|--------------|----------|
| 64 | 128 x 128 | 2CTA possible |
| 128 | 128 x 128 | 2CTA, UMMA |
| 256 | 128 x 64 | Dedicated 2CTA kernel |

---

## IO Complexity Analysis

### FlashAttention-1/2 IO Complexity

Let:
- `N` = sequence length
- `d` = head dimension
- `M` = SRAM size
- `B_r` = block size for rows (Q), `B_r = ceil(M / (4 * d))`
- `B_c` = block size for columns (K/V), `B_c = ceil(M / (4 * d))`

**Forward pass HBM accesses:**

1. Load Q in blocks of `B_r` rows: `O(N * d)` per Q block, `N/B_r` blocks
2. Load K, V in blocks of `B_c` rows: `O(N * d)` per K/V block, `N/B_c` blocks
3. Total: `O(N^2 * d^2 / M)` HBM accesses

This is `O(N^2 * d^2 / M)` compared to standard attention's `O(N^2 + N*d)`.

When `d^2 < M` (which is typical, since `M` is ~192 KB and `d` is typically
64-256), FlashAttention's IO is **less** than standard attention.

**Detailed forward pass IO:**

| Step | HBM Reads | HBM Writes |
|------|-----------|------------|
| Load Q_i (per i-block) | `B_r * d` | 0 |
| Load K_j, V_j (per j-block) | `2 * B_c * d` | 0 |
| Write O_i (per i-block) | 0 | `B_r * d` |
| Write L_i (per i-block) | 0 | `B_r` |
| **Total** | `N/B_r * (N/B_c * 2 * B_c * d + B_r * d)` | `N/B_r * (B_r * d + B_r)` |
| | `= O(N^2 * d / B_r + N * d)` | `= O(N * d + N)` |

With optimal block sizes: **Total IO = O(N^2 * d^2 / M + N * d)**

### FlashAttention-2 Improvements over FA1

FA2 reduces the number of HBM accesses by:

1. **Non-atomic accumulation**: FA1 used atomics for the output, FA2 uses
   thread-local accumulators and writes once.
2. **Better work partitioning**: FA2 assigns thread blocks to (batch, head)
   pairs and parallelizes within, avoiding load balancing issues.
3. **Reduced non-matmul FLOPs**: FA2 restructures the computation to minimize
   non-GEMM operations.

### Backward Pass IO Complexity

The backward pass has the same IO complexity as the forward pass:
`O(N^2 * d^2 / M + N * d)`, but with a constant factor of approximately 4-5x
more memory accesses (need to read Q, K, V, O, dO, L, and write dQ, dK, dV).

---

## Memory Complexity Analysis

### Standard Attention Memory

| Tensor | Size | Notes |
|--------|------|-------|
| Q | `B * N * h * d` | Input |
| K | `B * N * h_kv * d` | Input |
| V | `B * N * h_kv * d_v` | Input |
| S (attention scores) | `B * h * N^2` | Must materialize |
| P (attention weights) | `B * h * N^2` | Must materialize |
| O | `B * N * h * d_v` | Output |
| **Total** | `O(B * h * N^2)` | Quadratic in N |

### FlashAttention Memory

| Tensor | Size | Notes |
|--------|------|-------|
| Q | `B * N * h * d` | Input |
| K | `B * N * h_kv * d` | Input |
| V | `B * N * h_kv * d_v` | Input |
| O | `B * N * h * d_v` | Output |
| L (log-sum-exp) | `B * h * N` | Running statistics |
| **Total** | `O(B * N * h * d)` | Linear in N |

### Memory Savings

At `N = 2048`, `d = 128`, `h = 32`, `B = 16`:
- Standard: `16 * 32 * 2048^2 * 2 = 4 GB` for the attention matrix alone
- FlashAttention: `16 * 32 * 2048 * 128 * 2 = 256 MB` total

Memory savings scale linearly with sequence length: approximately `N / d`
times less memory.

### Memory Savings by Sequence Length

| Sequence Length | Standard Memory (h=32, d=128, B=16) | FlashAttention Memory | Savings |
|-----------------|--------------------------------------|-----------------------|---------|
| 512 | 256 MB | 64 MB | 4x |
| 1024 | 1 GB | 128 MB | 8x |
| 2048 | 4 GB | 256 MB | 16x |
| 4096 | 16 GB | 512 MB | 32x |
| 8192 | 64 GB | 1 GB | 64x |
| 16384 | 256 GB | 2 GB | 128x |

---

## Forward Pass Algorithm Step-by-Step

### Algorithm (FlashAttention-2)

```
Input: Q, K, V in HBM; softmax_scale, causal, window_size
Output: O in HBM, LSE (log-sum-exp) in HBM

1. Initialize:
   - Set block sizes B_r (query block), B_c (key/value block)
   - Allocate O = zeros(B, N, h, d) in HBM
   - Allocate LSE = zeros(B, h, N) in HBM (log-sum-exp)
   - Allocate m = -inf(B, h, N) in HBM (running row max)
   - Allocate l = zeros(B, h, N) in HBM (running row sum)

2. For each (batch, head) pair, launch one or more thread blocks:
   For i = 0 to N/B_r - 1:                          # Outer loop: Q blocks
     Load Q_i from HBM to SRAM                      # (B_r, d) read
     Initialize m_i = -inf, l_i = 0, O_i = 0       # In registers/SRAM

     For j = 0 to N/B_c - 1:                        # Inner loop: K/V blocks
       [Optional: Skip if causal/local and block is fully masked]

       Load K_j, V_j from HBM to SRAM               # 2 * (B_c, d) read
       S_ij = Q_i @ K_j^T * softmax_scale           # (B_r, B_c) GEMM

       [Apply ALiBi bias if provided]
       [Apply softcapping if enabled: tanh(S / cap) * cap]

       m_ij = rowmax(S_ij)                           # (B_r,) reduction
       P_ij = exp(S_ij - m_ij)                       # (B_r, B_c) element-wise

       [Apply causal/sliding window mask if needed]
       [Apply dropout if enabled]

       l_ij = rowsum(P_ij)                           # (B_r,) reduction

       # Online softmax update
       m_new = max(m_i, m_ij)                        # New running max
       alpha = exp(m_i - m_new)                      # Rescale factor for old
       beta = exp(m_ij - m_new)                      # Rescale factor for new

       O_i = alpha * O_i + beta * (P_ij @ V_j)      # (B_r, d) GEMM + accumulate
       l_i = alpha * l_i + beta * l_ij               # Update running sum
       m_i = m_new                                   # Update running max

     # End inner loop (all K/V blocks processed)

     # Final normalization
     O_i = O_i / l_i                                # Normalize by row sum

     Store O_i from SRAM to HBM                      # (B_r, d) write
     Store LSE_i = m_i + log(l_i) from SRAM to HBM   # (B_r,) write

3. Return O, LSE
```

### Key Observations

1. The inner loop processes K/V blocks sequentially for each Q block
2. Only `Q_i`, `K_j`, `V_j`, and `S_ij` are in SRAM at any time
3. The full `N x N` attention matrix is **never** materialized
4. The output `O_i` is accumulated incrementally across all K/V blocks
5. The `m` and `l` statistics track the running max and sum for numerical stability

---

## Backward Pass Algorithm Step-by-Step

The backward pass computes gradients with respect to Q, K, V given `dO`
(the gradient of the loss with respect to the output O).

### Algorithm

```
Input: Q, K, V, O, LSE, dO in HBM
Output: dQ, dK, dV in HBM

1. Preprocess:
   Compute D = rowsum(dO * O)                       # (B, h, N) in HBM
   # D[i] = sum_j dO[i,j] * O[i,j] is the softmax derivative constant

2. For each (batch, head) pair, launch thread blocks:

   For j = 0 to N/B_c - 1:                          # Outer loop: K/V blocks
     Load K_j, V_j from HBM to SRAM
     Initialize dK_j = 0, dV_j = 0 in SRAM

     For i = 0 to N/B_r - 1:                        # Inner loop: Q blocks
       [Optional: Skip if causal/local and block is fully masked]

       Load Q_i, O_i, dO_i from HBM to SRAM
       Load LSE_i, D_i from HBM

       # Recompute attention (no need to store P during forward)
       S_ij = Q_i @ K_j^T * softmax_scale           # (B_r, B_c) GEMM
       P_ij = softmax(S_ij, LSE_i)                  # Recomputed P

       # Compute dV
       dV_j += P_ij^T @ dO_i                        # (B_c, d) GEMM

       # Compute dP
       dP_ij = dO_i @ V_j^T                         # (B_r, B_c) GEMM

       # Compute dS (softmax backward)
       dS_ij = P_ij * (dP_ij - D_i)                 # (B_r, B_c) element-wise

       # Compute dQ
       dQ_i += dS_ij @ K_j * softmax_scale          # (B_r, d) GEMM

       # Compute dK
       dK_j += dS_ij^T @ Q_i * softmax_scale        # (B_c, d) GEMM

     Store dK_j, dV_j to HBM

   Store all dQ_i blocks to HBM
```

### Key Observations

1. **Recomputation**: The attention weights `P` are recomputed from Q, K, LSE
   during the backward pass, instead of storing them from the forward pass.
   This trades computation for memory: `O(N^2)` memory savings at the cost of
   `O(N^2 * d)` additional FLOPs for recomputation.

2. **Memory-efficient backward**: The backward pass uses `O(N * d)` memory
   instead of `O(N^2)` memory.

3. **Deterministic vs non-deterministic**:
   - Non-deterministic (default): Uses atomic adds for dQ, dK, dV accumulation.
     Faster but may have non-deterministic results due to floating-point
     addition order.
   - Deterministic (`deterministic=True`): Uses a different accumulation
     strategy that produces deterministic results. Slightly slower and uses
     more memory.

---

## Online Softmax

The online softmax algorithm is the key enabler of FlashAttention. It allows
computing the softmax without knowing all values in advance.

### Standard Softmax

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

This requires knowing the sum over all `j` before producing any output.

### Online Softmax (Milakov & Gimelshein, 2018)

Process elements in chunks. Maintain running statistics:
- `m`: running maximum (for numerical stability)
- `l`: running sum of exp values

```
For each new chunk x_k:
  m_new = max(m_old, max(x_k))
  l_new = exp(m_old - m_new) * l_old + exp(max(x_k) - m_new) * sum(exp(x_k - max(x_k)))
```

The key insight: when the maximum changes, we can rescale the old sum by
`exp(m_old - m_new)` to maintain correctness.

### FlashAttention's Use of Online Softmax

FlashAttention applies online softmax at the block level:

1. For each Q block `i`, process K/V blocks `j = 0, 1, 2, ...`
2. For each K/V block `j`, compute `S_ij = Q_i @ K_j^T * scale`
3. Update running max `m_i` and running sum `l_i`
4. Update running output `O_i` by rescaling and accumulating `P_ij @ V_j`

After all blocks are processed, normalize: `O_i /= l_i`

---

## Tiling Strategy

### FlashAttention-1 Tiling

FA1 uses a 2D tiling approach:
- Q is divided into blocks of `B_r` rows
- K, V are divided into blocks of `B_c` rows
- For each (Q_block, KV_block) pair, compute partial attention

The thread block assignment is: one thread block per (Q_block, batch, head).

### FlashAttention-2 Tiling

FA2 improves the tiling strategy:
- One thread block per (batch, head) pair
- Within a thread block, iterate over Q blocks and K/V blocks
- This provides better L2 cache locality for K/V blocks

### FlashAttention-3 Tiling (Hopper SM90)

FA3 leverages Hopper-specific hardware:
- Uses TMA (Tensor Memory Accelerator) for async data loading
- Uses WGMMA (Warp Group Matrix Multiply-Accumulate) for async GEMM
- Overlaps data loading with computation
- Supports larger tile sizes due to async capabilities

Tile sizes for FA3 forward on SM90:

| Head Dimension | Block M | Block N | Notes |
|----------------|---------|---------|-------|
| <= 64 | 128/192 | 128 | Causal may use different M |
| <= 96 | 64 | 128 | |
| <= 128 | 64/80 | 128 | 64 for causal/local |
| > 128 | 64 | 64/80 | |

### FlashAttention-4 Tiling (CuTeDSL)

FA4 introduces architecture-specific tile selection:

**SM90 Forward:**
- Head dim <= 64: `192 x 128` (RS + overlap)
- Head dim <= 96: `192 x 144` (noRS + overlap)
- Head dim <= 128: `128 x 128` (RS + overlap)
- Head dim <= 192: `128 x 128/112` (RS + overlap)
- Head dim <= 256: `128 x 64/80` (RS + overlap)

**SM100 Forward:**
- Supports 2CTA instructions for head_dim=128
- Dedicated 2CTA kernel for head_dim=256
- Persistent kernel scheduling for non-causal attention
- SplitKV for long sequences

**SM120 Forward:**
- Uses SM80 MMA instructions
- Reduced SMEM (99 KB): 128x128 for d<=64, 128x64 for d>64

---

## Comparison with Standard Attention

### Runtime Comparison

| Feature | Standard Attention | FlashAttention |
|---------|-------------------|----------------|
| Memory | O(N^2) | O(N) |
| HBM Reads | O(N^2 * d + N * d) | O(N^2 * d^2 / M + N * d) |
| HBM Writes | O(N^2 + N * d) | O(N * d) |
| Forward FLOPs | 2 * B * h * N^2 * d | Same (exact attention) |
| Backward FLOPs | ~5 * B * h * N^2 * d | ~5.5 * B * h * N^2 * d (recomputation) |
| Numerics | Standard | Slightly better (online softmax) |
| Dropout | Supported | Supported |
| Causal Mask | External | Built-in |
| Sliding Window | External | Built-in |
| ALiBi | External | Built-in |
| Softcapping | External | Built-in |

### When FlashAttention Is Slower

For very short sequences (`N < 128`), FlashAttention may be slower than
standard attention because:
1. The tiling overhead doesn't amortize
2. The kernel launch overhead dominates
3. The online softmax tracking adds overhead

For such cases, standard `torch.nn.functional.scaled_dot_product_attention`
(which may use FlashAttention, memory-efficient attention, or math fallback)
handles the dispatch automatically.

---

## Repository Structure

```
flash-attention/
|
|-- flash_attn/                      # FA2 and FA4 Python package
|   |-- __init__.py                  # Package init, version 2.8.4
|   |-- flash_attn_interface.py      # FA2 public API
|   |-- flash_blocksparse_attn_interface.py  # Block-sparse attention
|   |-- flash_blocksparse_attention.py
|   |-- flash_attn_triton.py         # Triton fallback implementation
|   |-- flash_attn_triton_og.py      # Original Triton implementation
|   |-- bert_padding.py              # Padding utilities
|   |
|   |-- cute/                        # FA4 (CuTeDSL) implementation
|   |   |-- __init__.py              # Exports flash_attn_func, flash_attn_varlen_func
|   |   |-- interface.py             # FA4 public API + autograd functions
|   |   |-- flash_fwd.py             # SM80 forward kernel
|   |   |-- flash_fwd_sm90.py        # SM90 forward kernel
|   |   |-- flash_fwd_sm100.py       # SM100 forward kernel (2CTA, SplitKV, paged KV)
|   |   |-- flash_fwd_sm120.py       # SM120 forward kernel
|   |   |-- flash_fwd_combine.py     # SplitKV partial result combine
|   |   |-- flash_fwd_mla_sm100.py   # MLA absorbed forward (SM100)
|   |   |-- flash_bwd.py             # SM80 backward kernel
|   |   |-- flash_bwd_sm90.py        # SM90 backward kernel
|   |   |-- flash_bwd_sm100.py       # SM100 backward kernel (2CTA, block sparse)
|   |   |-- flash_bwd_sm120.py       # SM120 backward kernel
|   |   |-- flash_bwd_preprocess.py  # Backward preprocessing kernel
|   |   |-- flash_bwd_postprocess.py # Backward postprocessing kernel
|   |   |-- softmax.py               # Online softmax with score modifiers
|   |   |-- mask.py                  # Attention masks (causal, local, block sparse)
|   |   |-- block_info.py            # Tile dimension and block range computation
|   |   |-- seqlen_info.py           # Sequence length tracking for varlen
|   |   |-- pipeline.py              # Pipeline state for async data loading
|   |   |-- tile_scheduler.py        # Tile scheduling (varlen, persistent)
|   |   |-- pack_gqa.py              # GQA head packing
|   |   |-- paged_kv.py              # Paged KV cache with TMA
|   |   |-- block_sparsity.py        # Block sparse attention support
|   |   |-- cache_utils.py           # JIT compilation cache
|   |   |-- utils.py                 # Hash functions, warp reductions
|   |   |-- fast_math.py             # exp2 polynomial, softcap creation
|   |   |-- copy_utils.py            # Type-converting copies
|   |   |-- cute_dsl_utils.py        # Patched compile, tensor conversion
|   |   |-- hopper_helpers.py        # SM90 WGMMA, shared memory
|   |   |-- blackwell_helpers.py     # SM100 UMMA, 2CTA support
|   |   |-- mma_sm100_desc.py        # SM100 MMA descriptor enums
|   |   |-- named_barrier.py         # Warp synchronization barriers
|   |   |-- testing.py               # FakeTensorMode utilities
|   |   |-- fa_logging.py            # Logging configuration
|   |   |-- benchmark.py             # FA4 benchmarks
|   |   |-- testing.py               # Test utilities
|   |
|   |-- modules/                     # Higher-level modules
|       |-- mha.py                    # Multi-head attention layer
|       |-- mlp.py                    # MLP layer
|       |-- block.py                  # Transformer block
|       |-- gpt.py                    # GPT model
|       |-- bert.py                   # BERT model
|       |-- falcon.py                 # Falcon model
|       |-- baichuan.py              # Baichuan model
|       |-- rotary.py                # Rotary embedding
|
|-- hopper/                          # FA3 (Hopper-optimized) implementation
|   |-- __init__.py                  # FA3 package init
|   |-- flash_attn_interface.py      # FA3 public API
|   |-- setup.py                     # FA3 build system
|   |-- flash_api.cpp                # C++/CUDA binding
|   |-- flash_api_stable.cpp         # Stable ABI binding (torch >= 2.9)
|   |-- flash_fwd_kernel_sm90.h      # SM90 forward kernel header
|   |-- flash_bwd_kernel_sm90.h      # SM90 backward kernel header
|   |-- flash_fwd_kernel_sm80.h      # SM80 fallback kernel header
|   |-- flash_bwd_kernel_sm80.h      # SM80 fallback backward
|   |-- mainloop_fwd_sm90_tma_gmma_ws.hpp  # SM90 TMA+WGMM forward mainloop
|   |-- mainloop_bwd_sm90_tma_gmma_ws.hpp  # SM90 TMA+WGMM backward mainloop
|   |-- mainloop_fwd_sm80.hpp        # SM80 forward mainloop
|   |-- mainloop_bwd_sm80.hpp        # SM80 backward mainloop
|   |-- epilogue_bwd.hpp             # Backward epilogue
|   |-- tile_scheduler.hpp           # Tile scheduling
|   |-- softmax.h                    # Online softmax
|   |-- mask.h                       # Attention masking
|   |-- block.h                      # Block utilities
|   |-- seqlen.h                     # Sequence length handling
|   |-- rotary.h                     # Rotary embedding
|   |-- pack_gqa.h                   # GQA packing
|   |-- paged_kv.h                   # Paged KV cache
|   |-- named_barrier.hpp            # Named barrier utilities
|   |-- static_switch.h              # Static kernel dispatch
|   |-- utils.h                      # General utilities
|   |-- tile_size.h                  # Tile size heuristics
|   |-- generate_kernels.py          # Kernel instantiation generator
|   |-- instantiations/              # Generated kernel instantiations
|
|-- csrc/                            # FA2 CUDA source code
|   |-- flash_attn/                  # FA2 CUDA kernels
|   |   |-- flash_api.cpp            # C++/CUDA binding
|   |   |-- src/                     # Per-head-dim CUDA kernel files
|   |       |-- flash_fwd_hdim*_*.cu # Forward kernels
|   |       |-- flash_bwd_hdim*_*.cu # Backward kernels
|   |       |-- flash_fwd_split_*.cu # Split-KV forward kernels
|   |       |-- alibi.h              # ALiBi implementation
|   |       |-- block_info.h         # Block info header
|   |       |-- dropout.h            # Dropout implementation
|   |-- cutlass/                     # CUTLASS submodule
|   |-- composable_kernel/           # AMD CK submodule (ROCm)
|   |-- flash_attn_ck/              # AMD CK integration
|   |-- fused_dense_lib/            # Fused dense layer kernels
|   |-- layer_norm/                  # Layer norm kernels
|
|-- setup.py                         # FA2 build system
|-- tests/                           # Test suite
|-- benchmarks/                      # Benchmarks
|-- examples/                        # Usage examples
|-- training/                        # Training scripts
|-- tools/                           # Development tools
```

---

## Version History

### FlashAttention-1 (2022)

Initial release introducing the tiling + online softmax approach.

**Key features:**
- Exact attention (no approximation)
- IO-aware tiling to minimize HBM access
- Online softmax for memory-efficient computation
- Memory: O(N) instead of O(N^2)
- Support for causal masks, dropout
- CUDA kernel implementation using CUTLASS

**Limitations:**
- Thread block assignment was suboptimal
- Used atomic operations for accumulation
- Single-threaded inner loop over K/V blocks
- Supported head dimensions: 16, 32, 64, 128
- Required A100 (SM80) or later

### FlashAttention-2 (2023)

Complete rewrite with 2x speedup over FA1.

**Key changes (v2.0):**
- Better parallelism: one thread block per (batch, head) pair
- Better work partitioning: sequential reduction within thread block
- Reduced non-matmul FLOPs
- Support for all head dimensions up to 256
- Backward pass on consumer GPUs for hdim <= 256 (no dropout)

**Additional features by version:**

| Version | Feature |
|---------|---------|
| 2.1 | Causal mask aligned to bottom-right corner for unequal seqlen_q/seqlen_k |
| 2.2 | Inference optimization (iterative decoding), `flash_attn_with_kvcache`, rotary embedding |
| 2.3 | Sliding window (local) attention (collaboration with Mistral AI) |
| 2.4 | ALiBi (attention with linear bias), deterministic backward pass |
| 2.5 | Paged KV cache (PagedAttention) |
| 2.6 | Softcapping attention (Gemma-2, Grok) |
| 2.7 | `torch.compile` compatibility |
| 2.8 | Current version (2.8.4); ROCm support (CK + Triton backends) |

### FlashAttention-3 (2024)

Optimized for Hopper GPUs (H100/H800).

**Key features:**
- Asynchronous execution with WGMMA and TMA
- FP16/BF16 forward and backward on SM90
- FP8 (E4M3) forward pass
- Leverages Hopper TMA for async memory loading
- Leverages Hopper WGMMA for async matrix multiply
- Overlaps softmax with data movement
- In-place reduction in GEMM accumulator
- Package name: `flash_attn_3`
- Requires CUDA >= 12.3, H100/H800 GPU

### FlashAttention-4 (2025)

Written in CuTeDSL (Python-based CUDA DSL) for Hopper and Blackwell.

**Key features:**
- CuTeDSL kernel implementation (no CUDA C++ for kernels)
- Supports SM80, SM90, SM100 (Blackwell B200), SM110 (Thor), SM120 (DGX Spark)
- User-defined score modifiers (`score_mod`) and mask modifiers (`mask_mod`)
- Block sparse attention
- 2CTA instructions on SM100 for head_dim=128/256
- SplitKV for long sequence decoding
- Paged KV cache
- MLA weight-absorbed attention
- JIT compilation with caching
- Package name: `flash-attn-4`
- Requires `nvidia-cutlass-dsl>=4.4.1`

---

## Key Innovations Per Version

### FlashAttention-1 Innovations

1. **IO-aware tiling**: First to formulate attention as an IO-aware problem
2. **Online softmax for attention**: Applied online softmax to attention computation
3. **Recomputation in backward**: Trade compute for memory by recomputing attention
4. **SRAM-aware block sizes**: Automatically choose block sizes based on SRAM

### FlashAttention-2 Innovations

1. **Better parallelism**: One thread block per (batch, head), inner loop sequential
2. **Reduced non-matmul operations**: Minimized non-GEMM FLOPs
3. **Improved softmax**: Fewer rescaling operations
4. **Wider head dimension support**: Up to 256 without hardware restrictions

### FlashAttention-3 Innovations

1. **Hopper TMA (Tensor Memory Accelerator)**: Async data loading from HBM
2. **Hopper WGMMA**: Async warp-group matrix multiply-accumulate
3. **FP8 forward**: E4M3 floating point for forward pass only
4. **Producer-consumer overlap**: Softmax and GEMM pipelining

### FlashAttention-4 Innovations

1. **CuTeDSL kernels**: Kernels written in Python, compiled to PTX/CUBIN
2. **User-defined score/mask modifiers**: `@cute.jit` callables for custom behavior
3. **Block sparse attention**: Structured sparsity at block level
4. **2CTA instructions**: Two-CTA clusters on Blackwell for head_dim=128/256
5. **Multi-architecture support**: SM80, SM90, SM100, SM110, SM120
6. **MLA absorption**: DeepSeek-style MLA with `qv` tensor support
7. **Persistent kernels**: CLC-style persistent scheduling on SM100
8. **JIT compilation with disk caching**: Compile once, reuse across runs

---

## Performance Characteristics on Different GPUs

### A100 80GB SXM (SM80)

**FlashAttention-2 benchmarks** (head_dim=128, hidden_dim=2048, FP16/BF16):

| Sequence Length | Batch Size | FA2 Forward+Backward (TFLOPS) | Speedup vs PyTorch |
|-----------------|------------|-------------------------------|---------------------|
| 512 | 32 | ~120 | 2-3x |
| 1024 | 16 | ~155 | 3-4x |
| 2048 | 8 | ~185 | 4-5x |
| 4096 | 4 | ~200 | 5-6x |
| 8192 | 2 | ~205 | 6-8x |

### H100 80GB SXM (SM90)

**FlashAttention-2 benchmarks** (head_dim=128, hidden_dim=2048, FP16/BF16):

| Sequence Length | Batch Size | FA2 Forward+Backward (TFLOPS) | Speedup vs PyTorch |
|-----------------|------------|-------------------------------|---------------------|
| 512 | 64 | ~200 | 2-3x |
| 1024 | 32 | ~320 | 3-4x |
| 2048 | 16 | ~420 | 4-5x |
| 4096 | 8 | ~480 | 5-7x |
| 8192 | 4 | ~500 | 6-8x |

**FlashAttention-3 benchmarks** (head_dim=128, FP16 on H100):

| Sequence Length | FA3 Forward (TFLOPS) | FA3 Backward (TFLOPS) | vs FA2 |
|-----------------|---------------------|-----------------------|--------|
| 1024 | ~500-600 | ~350-450 | 1.5-2x forward |
| 2048 | ~550-620 | ~400-500 | 1.5-2x forward |
| 4096 | ~580-640 | ~420-520 | 1.3-1.8x forward |
| 8192 | ~590-640 | ~440-530 | 1.2-1.6x forward |

**FlashAttention-3 FP8 forward** (head_dim=128, FP8 E4M3 on H100):
- Up to ~1.2-1.5x faster than FP16 FA3 forward

### B200 (SM100)

**FlashAttention-4 CuTeDSL benchmarks** (estimated, head_dim=128):
- FP16/BF16 forward: ~800-1200 TFLOPS
- FP8 forward: ~1500-2000 TFLOPS
- 2CTA instructions provide ~20-30% improvement for non-causal
- SplitKV improves long-sequence decoding by 2-4x

### Consumer GPUs

**RTX 3090 (SM86), RTX 4090 (SM89)**:
- FA2 works with all head dimensions up to 256
- Performance scales with memory bandwidth (slower than data center GPUs)
- hdim=256 backward supported without dropout since v2.5.5

**T4 (SM75 - Turing)**:
- Not officially supported; see [flash-attention-turing](https://github.com/ssiu/flash-attention-turing) fork

### AMD GPUs (ROCm)

**Supported GPUs: MI200, MI250, MI300, MI355, RDNA 3/4**

Two backends:
1. **Composable Kernel (CK)**: Default, fp16/bf16, forward+backward, hdim up to 256
2. **Triton**: FP16/BF16/FP32, forward+backward, causal, varlen, MQA/GQA, dropout, rotary, ALiBi, paged attention, FP8 (via FA3 interface)

---

## Supported Features Matrix

### Feature Support by Version

| Feature | FA2 | FA3 | FA4 |
|---------|-----|-----|-----|
| FP16/BF16 Forward | Yes | Yes | Yes |
| FP16/BF16 Backward | Yes | Yes | Yes |
| FP8 (E4M3) Forward | No | Yes | Yes (SM100) |
| FP8 Backward | No | No | No |
| Causal Mask | Yes | Yes | Yes |
| Sliding Window | Yes | Yes | Yes |
| ALiBi | Yes | No | No |
| Softcapping | Yes | Yes | Yes |
| Dropout | Yes | No | No |
| MQA/GQA | Yes | Yes | Yes |
| Paged KV Cache | Yes | Yes | Yes |
| Variable Length (varlen) | Yes | Yes | Yes |
| KV Cache (inference) | Yes | Yes | No (use FA3) |
| Rotary Embedding | Yes (KV cache) | Yes | No |
| Block Sparse | Yes | No | Yes |
| Custom Score Mod | No | No | Yes |
| Custom Mask Mod | No | No | Yes |
| 2CTA Instructions | No | No | Yes (SM100) |
| SplitKV | Yes | Yes | Yes (SM100) |
| MLA Absorption | No | No | Yes (SM100) |
| torch.compile | Yes | Yes | N/A |
| ROCm Support | Yes | No | No |
| SM80 (A100) | Yes | Yes (backward) | Yes |
| SM90 (H100) | Yes | Yes | Yes |
| SM100 (B200) | No | No | Yes |
| SM120 (DGX Spark) | No | No | Yes |

### Head Dimension Support

| Head Dimension | FA2 | FA3 | FA4 (SM90) | FA4 (SM100) |
|----------------|-----|-----|------------|-------------|
| 32 | Yes | No | Yes | No |
| 64 | Yes | Yes | Yes | Yes |
| 96 | Yes | Yes | Yes | Yes |
| 128 | Yes | Yes | Yes | Yes |
| 192 | Yes | Yes | Yes | No (DeepSeek: 192,128) |
| 256 | Yes | Yes | Yes | Yes (2CTA kernel) |

### GPU Architecture Support

| Architecture | GPU Examples | FA2 | FA3 | FA4 |
|-------------|--------------|-----|-----|-----|
| SM75 | T4 | No* | No | No |
| SM80 | A100 | Yes | Fallback | Yes |
| SM86 | RTX 3090 | Yes | Fallback | Yes |
| SM89 | RTX 4090 | Yes | Fallback | Yes |
| SM90 | H100 | Yes | Primary | Yes |
| SM100 | B200 | No | No | Primary |
| SM110 | Thor | No | No | Yes |
| SM120 | DGX Spark | No | No | Yes |

*T4 supported via community fork only.

---

## References

1. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
2. Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.
3. Dao, T., et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024.
4. Milakov, M., & Gimelshein, N. "Online normalizer calculation for softmax." arXiv:1805.02867, 2018.
