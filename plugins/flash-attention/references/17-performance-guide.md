# 17 - Performance Guide

This document provides comprehensive guidance for optimizing FlashAttention performance, covering block size selection, memory optimization, sequence length tuning, batch size optimization, and architecture-specific recommendations.

---

## Table of Contents

1. [Block Size Selection and Tuning](#block-size-selection-and-tuning)
2. [Memory Optimization Strategies](#memory-optimization-strategies)
3. [Sequence Length Optimization](#sequence-length-optimization)
4. [Batch Size Tuning](#batch-size-tuning)
5. [GQA/MQA Performance Impact](#gqamqa-performance-impact)
6. [Sliding Window Performance](#sliding-window-performance)
7. [FP8 Performance](#fp8-performance)
8. [Split-KV Tuning](#split-kv-tuning)
9. [Multi-GPU Considerations](#multi-gpu-considerations)
10. [Benchmarking](#benchmarking)
11. [Architecture-Specific Tuning](#architecture-specific-tuning)
12. [Profiling](#profiling)

---

## Block Size Selection and Tuning

### Overview

Block sizes (`kBlockM` for Q rows, `kBlockN` for K/V rows) are the most important performance parameters. They determine:
- Shared memory usage
- Arithmetic intensity
- Register pressure
- Occupancy (CTAs per SM)

### Selection Criteria

#### kBlockM (Q tile rows)

- **Larger kBlockM**: Better M-direction parallelism, less loop overhead, more register pressure
- **Smaller kBlockM**: Lower register pressure, better for short sequences
- **Constraints**: Must be a multiple of the MMA tile M dimension (16 for SM80, 64 for SM90)

#### kBlockN (K/V tile rows)

- **Larger kBlockN**: Less K/V loading overhead, more shared memory
- **Smaller kBlockN**: Less shared memory, more CTAs per SM, better for causal masking
- **Constraints**: Must be a multiple of the MMA tile N dimension (8 or 16)

### Default Block Sizes

#### SM80/SM86 (FA2)

| Head Dim | Default (M x N) | Causal (M x N) | Warps | smem (KB) |
|----------|----------------|----------------|-------|-----------|
| 32 | 128 x 128 | 128 x 128 | 4 | 20 |
| 64 | 128 x 128 | 128 x 128 | 4 | 40 |
| 96 | 128 x 64 | 64 x 64 (SM86) | 4 | 36 |
| 128 | 128 x 64 | 64 x 64 (SM86) | 4 | 48 |
| 192 | 128 x 64 | 128 x 64 | 8 | 96 |
| 256 | 64 x 64 | 64 x 64 | 4 | 96 |

#### SM90 (FA3)

| Head Dim | Default (M x N) | RS | smem (KB) |
|----------|----------------|-----|-----------|
| 64 | 192 x 128-192 | Yes | ~120 |
| 96 | 192 x 128-144 | No | ~150 |
| 128 | 128 x 128-176 | Yes | ~160 |
| 192 | 128 x 96-128 | Yes | ~200 |
| 256 | 128 x 64-80 | Yes | ~210 |

### Custom Block Sizes (FA4)

FA4 allows custom block sizes through the API:

```python
output = flash_attn_func(
    q, k, v,
    m_block_size=128,  # Custom M block size
    n_block_size=128,  # Custom N block size
    num_threads=256,   # Custom thread count
)
```

### Block Size Tuning Tool

```bash
# SM90 configuration search
python flash_attn/cute/sm90_config_search.py --headdim 128 --mode fwd

# Custom tile ranges
python flash_attn/cute/sm90_config_search.py --headdim 128 --tile-m 64,128 --tile-n 64,128,192
```

---

## Memory Optimization Strategies

### HBM Traffic Minimization

FlashAttention's key advantage is reducing HBM traffic from O(S^2) to O(S) by keeping the attention matrix in SRAM. Further optimizations:

1. **Q in registers**: Setting `Is_Q_in_regs=true` keeps Q tiles in registers, freeing shared memory for larger K/V blocks. This is beneficial when register pressure allows.

2. **Share Q/K shared memory**: Setting `Share_Q_K_smem=true` uses the same shared memory for Q and K/V (Q is loaded first, then copied to registers, and the space is reused for K/V).

3. **Register-source P*V**: When `MmaPV_is_RS=true`, the P matrix stays in registers after softmax and is fed directly to the PV GEMM. This eliminates:
   - P write to shared memory
   - P read from shared memory
   - ~kBlockM * kBlockN * 2 bytes of shared memory

### Shared Memory Budget

Total shared memory budget per SM:
- A100 (SM80): 164 KB
- H100 (SM90): 228 KB
- B200 (SM100): ~228+ KB

Shared memory allocation per CTA determines how many CTAs can run per SM:

| smem per CTA | CTAs per SM (A100) | CTAs per SM (H100) |
|-------------|-------------------|-------------------|
| 32 KB | 5 | 7 |
| 48 KB | 3 | 4 |
| 64 KB | 2 | 3 |
| 96 KB | 1 | 2 |
| 128 KB | 1 | 1 |
| 160 KB | 1 | 1 |

### Memory Pooling

For inference workloads, pre-allocate KV cache to avoid repeated allocation:

```python
from flash_attn.utils.generation import allocate_inference_cache

cache = allocate_inference_cache(
    max_batch_size=32,
    max_seqlen=4096,
    nheads=32,
    headdim=128,
    layers=32,
    device='cuda',
    dtype=torch.bfloat16
)
```

---

## Sequence Length Optimization

### Tile Quantization

When `seqlen` is not a multiple of `kBlockM` or `kBlockN`, the last tile is partially filled. This "tile quantization waste" reduces utilization:

```
Efficiency = seqlen / ceil(seqlen, kBlockM)

For seqlen=65, kBlockM=128: efficiency = 65/128 = 50.8%
For seqlen=129, kBlockM=128: efficiency = 129/256 = 50.4%
For seqlen=128, kBlockM=128: efficiency = 128/128 = 100%
```

**Mitigation**: Choose sequence lengths that are multiples of the block size (128 is common).

### Long Sequence Optimization

For sequences > 4096:

1. **Split-KV parallelism**: Splits the K/V dimension across thread blocks
2. **PackGQA**: For GQA models, packs multiple Q heads to increase M-dimension parallelism
3. **L2 cache considerations**: If one head of KV exceeds L2 size (~50MB), use SplitKV to ensure each split fits in L2

### Short Sequence Optimization

For sequences < 256:

1. **PackGQA**: Packs Q heads across the KV dimension
2. **Larger batch sizes**: Batch multiple sequences to increase total work
3. **Persistent kernels**: On SM100, persistent kernels reduce launch overhead

### Inference (seqlen_q=1) Optimization

For single-token generation:

```python
# Use dedicated KV cache function
flash_attn_with_kvcache(q, cache_k, cache_v)
```

This uses a specialized kernel that:
- Loads all K/V in a single pass
- No Q tiling loop (only 1 row)
- Optimized for throughput on the KV dimension

---

## Batch Size Tuning

### Grid Sizing

The CUDA grid is `(num_m_blocks, batch_size, num_heads)`. For full GPU utilization:

```
total_blocks = num_m_blocks * batch_size * num_heads
target: total_blocks >= num_SMs * 2  # At least 2 waves for good utilization
```

For A100 (108 SMs) with hdim=128, seqlen=512, 32 heads:
- `num_m_blocks = 512/128 = 4`
- `total_blocks = 4 * batch_size * 32 = 128 * batch_size`
- For 2 waves on 108 SMs: `batch_size >= 2`

### Batch Size vs Sequence Length Trade-off

For fixed total tokens `B * S`:

```
Short sequences (B large, S small): More grid parallelism but tile quantization waste
Long sequences (B small, S large): Less grid parallelism but better per-CTA utilization
```

Optimal: `S` is a multiple of `kBlockM` and `B * ceil(S/kBlockM) * H >= num_SMs * 2`.

### Varlen Batching

Use `flash_attn_varlen_func` to pack variable-length sequences:

```python
# Instead of padding to max_seqlen (wasting compute on padding):
q_padded = pad_input(q, ...)  # (batch, max_seqlen, H, D)
output = flash_attn_func(q_padded, k_padded, v_padded)

# Use varlen (compute only on valid tokens):
output = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k)
```

Memory savings: Up to `(max_seqlen - avg_seqlen) / max_seqlen` fraction of compute is eliminated.

---

## GQA/MQA Performance Impact

### MQA (Multi-Query Attention)

With MQA (`num_kv_heads=1`), all Q heads share the same K/V:

- **K/V loading**: 1x instead of Hx (major bandwidth savings)
- **Grid**: `(num_m_blocks, B, H)` still launches H thread blocks per batch
- **Redundant K/V loads**: Each Q head loads K/V independently

**Optimization**: PackGQA packs Q heads into fewer CTAs, sharing K/V loads.

### GQA (Grouped-Query Attention)

With GQA (`num_kv_heads=H/G`):

- **K/V loading**: H/G heads of K/V instead of H
- **Compression ratio**: G:1 reduction in K/V storage and bandwidth

### Performance Scaling

| GQA Group Size | K/V Bandwidth | Attention Compute | PackGQA Benefit |
|---------------|--------------|-------------------|-----------------|
| 1 (MHA) | 1x | 1x | None |
| 2 | 0.5x | Same | Small |
| 4 | 0.25x | Same | Moderate |
| 8 | 0.125x | Same | Large |
| H (MQA) | 1/H x | Same | Very large |

PackGQA benefit increases with group size because more Q heads share each KV head, increasing the opportunity for shared loading.

---

## Sliding Window Performance

### Block Skipping

Sliding window attention skips K/V blocks that are entirely outside the window:

```
For window_size_left=256, window_size_right=0, seqlen=4096, kBlockN=128:
Full attention: 32 K/V blocks per Q block
Sliding window: 2 K/V blocks per Q block (only 2 within the window)
Speedup: ~16x less computation
```

### Partial Blocks

At window boundaries, some blocks are partially inside the window. These still require masking but not full computation. The kernel handles this with the masking loop optimization.

### Memory Access Pattern

Sliding window improves cache behavior:
- Working set per Q tile: `window_size * D` instead of `seqlen * D`
- For window_size=512, D=128: 128 KB per Q tile (fits in L2)
- For full attention, seqlen=8192: 2 MB per Q tile (exceeds L1, benefits from L2)

---

## FP8 Performance

### Throughput Benefits

FP8 tensor cores provide 2x throughput compared to fp16:

| Type | Tensor Core Throughput | Memory Bandwidth |
|------|----------------------|------------------|
| FP16 | 1x | 2 bytes/element |
| BF16 | 1x | 2 bytes/element |
| FP8 (E4M3) | 2x | 1 byte/element |
| INT8 | 2x | 1 byte/element |

### Total Speedup

FP8 attention benefits from both compute and bandwidth improvements:
- **Compute**: 2x throughput on tensor cores
- **Memory**: 2x reduction in Q/K/V traffic (1 byte vs 2 bytes per element)
- **Combined**: Typically 2-3x faster than fp16 for forward pass

### Limitations

- FP8 is only supported for forward pass (backward uses fp16/bf16)
- Requires descale factors for numerical accuracy
- E4M3 format has limited dynamic range (max ~448)
- Best suited for inference with pre-quantized KV caches

---

## Split-KV Tuning

### When to Split

```python
# Automatic split selection (default)
output = flash_attn_func(q, k, v)  # num_splits chosen by heuristic

# Manual split count
output = flash_attn_func(q, k, v, num_splits=4)
```

### Split Count Selection

The heuristic considers:
1. **Occupancy**: More splits = more thread blocks = better SM utilization
2. **L2 cache**: If one KV head exceeds L2 (~50MB), split to fit
3. **Combine overhead**: More splits = more partial results to combine

Guidelines:

| Scenario | Recommended Splits |
|----------|-------------------|
| Short sequences (S < 1024) | 1 (no split) |
| Medium sequences (1024-4096) | 1-2 |
| Long sequences (4096-16384) | 2-4 |
| Very long sequences (> 16384) | 4-8 |

### Memory Overhead

SplitKV allocates:
- Oaccum: `num_splits * B * H * S_q * D * 4` bytes
- LSEaccum: `num_splits * B * H * S_q * 4` bytes

For `num_splits=4, B=4, H=32, S=8192, D=128`:
- Oaccum: 16 GB
- LSEaccum: 64 MB

**Mitigation**: Use smaller `num_splits` when memory is constrained.

---

## Multi-GPU Considerations

### Tensor Parallelism

FlashAttention supports sequence-parallel tensor parallelism through the distributed utilities:

```python
from flash_attn.utils.distributed import all_gather, reduce_scatter

# Sequence parallel: each GPU handles a portion of the sequence
# Forward: all_gather inputs, compute attention, reduce_scatter outputs
# Backward: all_gather gradients, compute backward, reduce_scatter gradients
```

### Communication Overhead

For tensor parallel attention:

```
Communication volume per layer:
Forward: all_gather(Q), all_gather(K), all_gather(V) = 3 * B * S * H * D
         reduce_scatter(O) = B * S * H * D
Backward: all_gather(dO) = B * S * H * D
          reduce_scatter(dQ), reduce_scatter(dK), reduce_scatter(dV) = 3 * B * S * H * D
```

### Gradient Synchronization

```python
from flash_attn.utils.distributed import allreduce_sequence_parallel_grad

# Synchronize gradients for sequence-parallel parameters
allreduce_sequence_parallel_grad(model, process_group)
```

### Shared Parameter Synchronization

```python
from flash_attn.utils.distributed import sync_shared_params

# Broadcast shared parameters from rank 0 to all ranks
sync_shared_params(model, process_group)
```

---

## Benchmarking

### Built-in Benchmark Utilities

File: `flash_attn/utils/benchmark.py`

```python
from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward, benchmark_combined

# Forward only
t, m = benchmark_forward(flash_attn_func, q, k, v, causal=True, repeats=100)

# Forward + Backward
t_fwd, t_bwd = benchmark_fwd_bwd(flash_attn_func, q, k, v, repeats=100)

# Combined (fwd+bwd in single measurement)
t, m = benchmark_combined(flash_attn_func, q, k, v, repeats=100)
```

### PyTorch Profiler Integration

```python
from flash_attn.utils.benchmark import pytorch_profiler

pytorch_profiler(
    flash_attn_func, q, k, v, causal=True,
    backward=True,
    trace_filename="trace.json"
)
```

### Memory Benchmarking

```python
from flash_attn.utils.benchmark import benchmark_memory

mem = benchmark_memory(flash_attn_func, q, k, v, desc="flash_attn")
# Prints: "flash_attn max memory: X.XX GB"
```

### Custom Benchmarks

```python
import torch.utils.benchmark as benchmark

t = benchmark.Timer(
    stmt='flash_attn_func(q, k, v, causal=True)',
    globals={'flash_attn_func': flash_attn_func, 'q': q, 'k': k, 'v': v},
)

print(t.timeit(100))
```

---

## Architecture-Specific Tuning

### A100 (SM80)

- **Optimal kBlockN**: 64 for most head dimensions
- **Occupancy**: Target 2 CTAs per SM
- **smem budget**: 164 KB per SM
- **Best practices**: Use `kBlockM=128, kBlockN=64` for hdim=128

### H100 (SM90)

- **TMA benefits**: Single-thread issue frees warps for computation
- **WGMMA**: 64-row M dimension means `kBlockM` should be a multiple of 64
- **IntraWGOverlap**: Overlapping producer/consumer within a warp group
- **Best practices**: Use `kBlockM=128, kBlockN=128` for hdim=128 with RS

### B200 (SM100)

- **2CTA**: Cooperative attention doubles effective M dimension
- **Persistent kernels**: Better load balancing for varlen
- **FP8**: Native FP8 for 2x throughput
- **Best practices**: Use 2CTA mode for hdim=128, persistent kernels for varlen

---

## Profiling

### Nsight Systems

```bash
nsys profile -o flash_attn_profile python my_benchmark.py
nsys-ui flash_attn_profile.qdrep
```

Look for:
- Kernel execution time
- Memory transfer overlaps
- SM utilization percentage
- Launch overhead

### Nsight Compute

```bash
ncu --set full -o flash_attn_report python my_benchmark.py
ncu-ui flash_attn_report.ncu-rep
```

Key metrics:
- `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum`: FP64 instructions
- `smsp__sass_thread_inst_executed_op_hadd_pred_on.sum`: FP16 instructions
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`: Memory bandwidth utilization
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`: Shared memory loads
- `lts__t_sectors_op_read.sum`: L2 cache reads

### Roofline Analysis

FlashAttention's arithmetic intensity:

```
AI = FLOPs / Bytes_transferred

Forward FLOPs ≈ 2 * B * H * S_q * S_k * D
Forward HBM reads ≈ B * H * S_q * D + B * H_kv * S_k * D * 2
Forward HBM writes ≈ B * H * S_q * D + B * H * S_q * sizeof(float)

For S_q = S_k = S:
AI ≈ 2 * H * S * D / (H * D + H_kv * D * 2 + H * D + H * 4)
```

For typical parameters (H=32, S=2048, D=128), AI is compute-bound on H100 (peak 989 TFLOPS, peak bandwidth 3.3 TB/s, ridge point ~300).

### Performance Checklist

1. **Correct head dimension**: Must be one of {16, 32, 64, 96, 128, 192, 256}
2. **Correct data type**: FP16 or BF16 for FA2/FA3
3. **Contiguous last dimension**: `q.is_contiguous()` should be True for the head_dim
4. **Optimal sequence length**: Multiple of kBlockM (typically 128)
5. **Sufficient batch size**: `B * ceil(S/128) * H >= num_SMs * 2`
6. **GQA packing**: Use PackGQA for GQA models, especially with short sequences
7. **SplitKV for long sequences**: Enable for seqlen > 4096
8. **Avoid unnecessary dropout**: Dropout adds RNG overhead
9. **Use bf16 over fp16**: BF16 has better numerical range
10. **Warm up**: Run a few iterations before benchmarking to warm up CUDA caches
