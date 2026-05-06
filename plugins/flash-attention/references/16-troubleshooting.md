# 16 - Troubleshooting Guide

This document provides comprehensive guidance for debugging and troubleshooting FlashAttention GPU kernels, covering common issues, debugging techniques, race conditions, and tool usage.

---

## Table of Contents

1. [Common Error Messages](#common-error-messages)
2. [GPU Kernel Debugging Techniques](#gpu-kernel-debugging-techniques)
3. [Kernel Hang/Deadlock Debugging](#kernel-hangdeadlock-debugging)
4. [2CTA Debugging](#2cta-debugging)
5. [TMA Race Conditions](#tma-race-conditions)
6. [compute-sanitizer Usage](#compute-sanitizer-usage)
7. [PTX Inspection](#ptx-inspection)
8. [Numerical Issues](#numerical-issues)
9. [Varlen Preprocess Tile Bug](#varlen-preprocess-tile-bug)
10. [SM90 Block Size Tuning](#sm90-block-size-tuning)
11. [R2P Masking SASS Analysis](#r2p-masking-sass-analysis)
12. [CLC Trace Debugging](#clc-trace-debugging)
13. [Compilation Issues](#compilation-issues)
14. [Memory Issues](#memory-issues)

---

## Common Error Messages

### "FlashAttention requires building with sm version sm80-sm90"

**Cause**: The CUDA kernel was compiled for an architecture below SM80.

**Solution**: Rebuild with `TORCH_CUDA_ARCH_LIST="8.0"` or higher:
```bash
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install flash-attn --no-build-isolation
```

### "CUDA error: an illegal memory access was encountered"

**Cause**: Out-of-bounds memory access, often from:
- Incorrect `cu_seqlens` values (not sorted, not cumulative, or negative)
- Mismatched batch dimensions between Q, K, V
- Head dimension not matching the compiled kernel's expectation
- Using fp16 tensors with a bf16-compiled kernel or vice versa

**Solution**: 
1. Verify tensor shapes: Q should be `(batch, seqlen_q, num_heads, head_dim)`
2. Check that `cu_seqlens` is properly formatted: non-decreasing, starting at 0
3. Ensure `head_dim` is one of {16, 32, 64, 96, 128, 192, 256}
4. Run with `compute-sanitizer --tool memcheck`

### "CUDA error: a sync was issued in a divergent thread"

**Cause**: A `__syncthreads()` call inside a conditional that not all threads take.

**Solution**: This is an internal kernel bug. Check if:
- You are using varlen with zero-length sequences
- The sequence length is exactly 0 for some batch elements
- The problem dimensions trigger an unexpected code path

### "smem_size >= 48 * 1024" assertion

**Cause**: The kernel's shared memory requirement exceeds the default 48 KB limit.

**Solution**: This should be handled automatically by `cudaFuncSetAttribute`. If it fails:
- Check that your GPU supports enough shared memory
- A100: 164 KB/SM, H100: 228 KB/SM
- For SM86/SM89: Maximum ~100 KB/SM

### NaN in output

**Cause**: Several possible causes:
1. Zero-length sequences (LSE becomes infinity)
2. All-masked rows (softmax of all -inf produces NaN)
3. Incorrect scale factors
4. Accumulated numerical errors in fp16 for very long sequences

**Solution**:
1. Check `cu_seqlens` values
2. Verify `softmax_scale` is correct (typically `1/sqrt(head_dim)`)
3. Use bf16 instead of fp16 for better numerical range
4. Check if `softcap` is set appropriately

---

## GPU Kernel Debugging Techniques

### Printf-Based Debugging

GPU `printf` (via `cute.printf` in CuTeDSL or direct `printf` in CUDA) is the primary debugging tool.

#### Printf Guards

Avoid print storms by limiting output:

```python
# One thread per warp
if cute.arch.thread_idx()[0] % 32 == 0:
    cute.printf("...")

# One thread per CTA
with cute.arch.elect_one():
    cute.printf("...")

# One specific thread
if tidx == 0:
    cute.printf("...")

# One specific CTA
if cute.arch.block_idx()[0] == 0 and tidx == 0:
    cute.printf("...")
```

#### Binary Search Strategy

1. **Coarse**: Print at entry/exit of each warp's main function to identify which warp is stuck
2. **Medium**: Print before/after each pipeline wait to identify which barrier is stuck
3. **Fine**: Print barrier index, phase, stage, and iteration count

### What to Print

- CTA index (`blockIdx.x`)
- Pipeline stage index and phase
- Loop iteration count
- Whether `try_wait` succeeds or fails
- Tensor values at key points (small tiles only)

### Checking Intermediate Results

For numerical debugging, compare against the reference implementation:

```python
from flash_attn.utils.testing import attention_ref

output_ref, attn_ref = attention_ref(q, k, v, causal=True, upcast=True)
output_fa = flash_attn_func(q, k, v, causal=True)

# Check max absolute error
print((output_fa - output_ref).abs().max().item())
```

---

## Kernel Hang/Deadlock Debugging

### General Approach

#### Step 1: Build a Minimal Reproduction

- batch=1, nheads=1, smallest seqlen that hangs
- Single config, no loops, no benchmarking
- Add a timeout or run with `compute-sanitizer`

#### Step 2: Add Printf to Locate the Hang

Use binary search to narrow down which operation is blocked:

```cuda
// At the start of the main loop
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Enter loop, n_block=%d\n", n_block);
}

// Before each barrier wait
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Before wait, stage=%d, phase=%d\n", stage, phase);
}
```

#### Step 3: Identify the Deadlock Chain

A hang is always a cycle. Typical chain:

```
MMA waiting for K from load (pipeline_kv full barrier)
  -> Load finished but stuck in producer_tail
    -> MMA can't release because it's waiting for K
```

Trace backwards: who should signal the barrier, and why haven't they?

#### Step 4: Vary Problem Size Systematically

Test with different sequence lengths to find the pattern:

| seqlen | n_blocks | Result |
|--------|----------|--------|
| 128 | 1 | ? |
| 256 | 2 | ? |
| 384 | 3 | ? |
| 512 | 4 | ? |

If the hang correlates with pipeline stages wrapping around, the problem is likely in barrier tx_count or phase tracking.

#### Step 5: Check Barrier Byte Counts (tx_count)

For TMA-based pipelines, `arrive_and_expect_tx` sets the expected transaction byte count:

```cuda
// Correct: tx_count must match actual bytes arriving
mbarrier.arrive.expect_tx(barrier, expected_bytes);
```

If expected < actual: Barrier fires too early -> data race
If expected > actual: Barrier never fires -> hang

#### Step 6: Check Phase/Parity Tracking

`mbarrier_try_wait_parity` uses a single parity bit (0 or 1):

```cuda
// Correct: use phase % 2
mbarrier.try_wait.parity(barrier, phase % 2);
// Wrong: use raw phase counter
mbarrier.try_wait.parity(barrier, phase);  // Bug: phase=2 looks like phase=0
```

#### Step 7: Beware Compiler-as-Bug-Source

If the kernel works WITH printf but hangs WITHOUT it:

- The printf acts as a **compiler barrier**
- The MLIR/LLVM backend reorders instructions incorrectly
- PTX fences (`fence_view_async_shared`, etc.) may NOT fix it (they affect hardware, not compiler scheduling)

Signs this is happening:
- A single `cute.printf("\n")` in the right function fixes the hang
- The fix is location-sensitive (printf in one function fixes it, in another doesn't)

Workarounds:
- `@dsl_user_op` decorator on pipeline methods
- Compare generated PTX/SASS with and without printf
- File a compiler bug

---

## 2CTA Debugging

### Specific Pitfalls in 2CTA Mode

#### 1. tcgen05.commit with Empty Commit Groups

`tcgen05.commit(mbar, mask, cta_group::2)` signals an mbarrier after all pending MMA operations complete. But if there are NO pending operations (empty commit group), the signal only reaches the local CTA's barrier, not the remote CTA's.

**Fix**: Use explicit `mbarrier_arrive(barrier, dst_cta_rank)` to both CTAs.

#### 2. producer_tail Deadlock

The default `producer_tail` (inherited from SM90 pipelines) drains the pipeline by calling `producer_acquire` in a loop. In 2CTA mode this deadlocks because the consumer (MMA warp) may have already exited without releasing all stages.

**Fix**: Make `producer_tail` a no-op for 2CTA.

#### 3. Tile Scheduler Must Account for Cluster Shape

Both CTAs in a cluster must get the **same** tile coordinate:

```cuda
// Wrong: blockIdx.x assigns consecutive values to CTAs in same cluster
int m_block = blockIdx.x;

// Correct: divide by cluster_shape_m
int m_block = blockIdx.x / cluster_shape_m;
```

#### 4. Cross-CTA vs Per-CTA Pipelines

Pipelines where CTA 1's threads remotely arrive on CTA 0's barriers need cluster-sized cooperative group counts. Pipelines that are purely local to each CTA keep per-CTA counts.

#### 5. Softmax Masking Offset

Causal mask row positions must account for the CTA's position within the cluster:

```cuda
// Correct: multiply m_block by cta_group_size for mask coordinates
int row_idx = m_block * cta_group_size * kBlockM + ...;
```

### Debugging 2CTA Hangs

1. First verify the kernel works in non-2CTA mode
2. Add printf at each pipeline stage to identify which stage hangs
3. Check that tx_count is multiplied by `cta_group_size` for all TMA barriers
4. Verify the tile scheduler produces identical tile coordinates for paired CTAs
5. Check that softmax masking uses the correct row offset accounting for cluster position

---

## TMA Race Conditions

### compute-sanitizer False Positives with cp.async.bulk

`compute-sanitizer --tool=racecheck` reports false-positive shared-memory race hazards when `cp.async.bulk` (raw-address TMA) is used in a cross-warp producer/consumer pipeline inside a dynamic loop.

The same pattern with `cp.async.bulk.tensor` (descriptor-based TMA) reports **zero hazards**.

### Root Cause

racecheck instruments every shared memory access and checks for conflicting accesses lacking a recognized happens-before relationship:

- **`cp.async.bulk` (raw address)**: The sanitizer attributes the smem write to the issuing thread (thread 0 of warp 0 via `elect_one`). When warp 1 issues `ld.shared.b32`, the sanitizer cannot find a happens-before edge across warps in a dynamic loop.

- **`cp.async.bulk.tensor` (TMA descriptor)**: The TMA engine is a separate hardware unit. The sanitizer does not attribute the write to any thread, so no hazard pair is reported.

### Proof of False Positive

1. **Data correctness**: All variants produce bit-identical results
2. **Single-warp test**: One warp doing both TMA write and thread read reports zero hazards
3. **Unrolled loop**: Fully unrolling (`unroll_full=True`) reports zero hazards
4. **Named barrier**: Adding `bar.sync` per iteration eliminates the hazard
5. **Descriptor TMA**: Switching to `cp.async.bulk.tensor` eliminates the hazard

### Fix

Switch from raw-address `cp.async.bulk` to descriptor-based `cp.async.bulk.tensor`:

```python
# Before (triggers false positive):
copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)

# After (clean):
copy_atom_stats = cpasync.make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), ...)
```

### Affected Code

Only buffers consumed by thread-level shared memory reads (`lds`) are affected:
- LSE and dPsum buffers (consumed by autovec_copy from smem)
- Q/K/V/dO are NOT affected (consumed by UMMA hardware, no thread `lds`)

---

## compute-sanitizer Usage

### Basic Usage

```bash
# Memory access checking
compute-sanitizer --tool memcheck python my_test.py

# Race condition checking
compute-sanitizer --tool racecheck python my_test.py

# With source line info (requires -lineinfo compilation)
CUTE_DSL_LINEINFO=1 compute-sanitizer --tool racecheck python my_test.py

# With launch synchronization checking
compute-sanitizer --tool synccheck python my_test.py
```

### Racecheck with TMA

```bash
# Note: --racecheck-memcpy-async=no does NOT suppress cp.async.bulk hazards
# It only controls the older cp.async (SM80) family
compute-sanitizer --tool racecheck python my_test.py
```

### Common Racecheck Findings

| Finding | Cause | Action |
|---------|-------|--------|
| Hazard on smem after cp.async.bulk | False positive (see TMA Race Conditions) | Switch to descriptor-based TMA or ignore |
| Hazard on sK/sV during dQ compute | Missing `__syncthreads()` between sK read and sdQ write | Add synchronization |
| Hazard on sP between dV and dK | sP and sdS share memory, need barrier | Verify pipeline stage management |

### Memory Leak Detection

```bash
compute-sanitizer --tool memcheck --leak-check full python my_test.py
```

---

## PTX Inspection

### Dumping PTX

```bash
# FA4 (CuTeDSL)
CUTE_DSL_KEEP_PTX=1 python my_test.py
# PTX files saved to /tmp/${USER}/cutlass_dsl_*/

# FA2/FA3 (C++/CUDA)
# PTX is embedded in the .so file; use cuobjdump:
cuobjdump -ptx flash_attn_cuda.cpython-*.so > output.ptx
```

### Inspecting SASS (Assembly)

```bash
# Dump SASS for specific kernel
cuobjdump -sass flash_attn_cuda.cpython-*.so > output.sass

# FA4: Dump CUBIN/SASS
CUTE_CUBIN_PATH=/tmp/cubin_dump python my_test.py
```

### Key PTX Instructions to Look For

| Instruction | Purpose |
|------------|---------|
| `cp.async.ca.shared.global` | SM80 async copy (FA2) |
| `cp.async.bulk.tensor.2d.shared::cta.global.tile` | SM90 TMA load (FA3) |
| `wgmma.mma_async.sync.aligned` | SM90 warpgroup MMA |
| `mbarrier.init`, `arrive`, `try_wait` | SM90 barrier operations |
| `tcgen05.commit` | SM100 MMA commit |
| `cp.async.bulk.shared::cta.global` | SM100 bulk copy |

### Lineinfo for Sanitizer Source Mapping

```bash
CUTE_DSL_LINEINFO=1 compute-sanitizer --tool racecheck python my_test.py
```

This adds source line information to sanitizer error reports, mapping PTX addresses back to Python source lines.

---

## Numerical Issues

### Common Causes

1. **FP16 overflow**: Scores exceed fp16 range (max ~65504). Use bf16 or ensure scores are scaled properly.

2. **Softmax underflow**: Very long sequences can cause softmax values to underflow to zero. FlashAttention's online softmax minimizes this but doesn't eliminate it.

3. **LSE infinity**: When all scores in a row are -inf (fully masked), the LSE becomes infinity. The kernel handles this by setting the output to zero.

4. **Softcap edge cases**: Very large softcap values approach standard attention; very small values approach uniform attention.

### Validation Against Reference

```python
from flash_attn.utils.testing import attention_ref

output_fa = flash_attn_func(q.float().cuda(), k.float().cuda(), v.float().cuda())
output_ref, _ = attention_ref(q.float(), k.float(), v.float(), upcast=True)

max_err = (output_fa - output_ref).abs().max().item()
rel_err = (output_fa - output_ref).abs().max() / output_ref.abs().max()

print(f"Max absolute error: {max_err}")
print(f"Max relative error: {rel_err.item()}")
```

Expected errors:
- FP32: < 1e-6 (near machine epsilon)
- BF16: < 1e-2 (bf16 has ~3 decimal digits of precision)
- FP16: < 5e-3 (fp16 has ~3.3 decimal digits)

---

## Varlen Preprocess Tile Bug

### Summary

`SeqlenInfo.create` in `flash_bwd_preprocess.py` defaulted `tile=128`, but the backward kernel uses `tile_m=m_block_size` (e.g., 64 for causal SM90). This caused the preprocess to zero `dq_accum` and write `lse_log2/dpsum` at wrong padded offsets for batches after batch 0.

### How Padded Offsets Work

For varlen, buffers like `dq_accum` use tile-aligned gaps between sequences:

```python
padded_offset_q = ((offset_q + batch_idx * tile_m) // tile_m) * tile_m
```

With `tile_m=64` vs `tile_m=128`:
- tile=64: `padded_offset = ((128 + 64) // 64) * 64 = 192`
- tile=128: `padded_offset = ((128 + 128) // 128) * 128 = 256`

The preprocess was zeroing at 256, the backward was writing at 192.

### Symptoms

- Tests pass in isolation (torch.empty gets clean memory)
- Tests fail when run in sequence (CUDA memory caching reuses NaN-polluted memory)
- `dq_accum` contains NaN after backward kernel
- `torch.zeros` for `dq_accum` masks the bug (zeroes everywhere)

### Fix

```python
# Before:
seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ)
# After:
seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ,
                            tile=self.tile_m)
```

### Lesson

Any code computing `padded_offset` for varlen buffers must use the same tile size as the kernel that accesses those buffers.

---

## SM90 Block Size Tuning

### Configuration Search Tool

```bash
# Enumerate feasible configs for SM90
python flash_attn/cute/sm90_config_search.py --headdim 128
python flash_attn/cute/sm90_config_search.py --mode fwd --headdim 128
python flash_attn/cute/sm90_config_search.py --mode bwd --headdim 192 --tile-m 64,80 --tile-n 64,96
```

### Hardware Constraints (H100)

- **SMEM**: 228 KB total. Reserve ~3 KB for LSE, dPsum, mbarriers -> 224 KB for tensors
- **Registers**: 2 WG -> 240 regs/thread, 3 WG -> 160 regs/thread
- **GMMA atom**: Always M=64. Effective M must be divisible by 64 (after swap). N must be divisible by `atom_layout_n * 8`

### Key Configuration Decisions

1. **Number of Warp Groups**: 2 (hdim <= 128) or 3 (hdim 129-192)
2. **swap_AB**: Transposes output tile if natural M isn't divisible by 64
3. **AtomLayout**: Distributes WGs across M and N dimensions
4. **mma_dkv_is_rs**: Register-source optimization for P/dS (saves smem)
5. **Pipeline Staging**: Q=1 stage, K/V=2 stages (forward); Q=2 stages (backward)

### Register Accounting

Forward peak registers: `regs_S + regs_P + regs_O` (with WG overlap)
Backward peak registers: `max(2 * regs_SdP, regs_dQ) + regs_dK + regs_dV`

### SMEM Accounting

Forward: `max(sQ, sO) + sK*2 + sV*2 + sP`
Backward: `sQ*2 + sK + sV + sdO*stages + sP + sdS + sdQaccum`

---

## R2P Masking SASS Analysis

### Instruction Count Comparison

For hdim=128, tile_n=128 (32 accumulator elements per row):

| Case | Old (no R2P) | New (R2P) | Savings |
|------|-------------|-----------|---------|
| Non-causal | 3104 instructions | 3072 | 32 (-1%) |
| Causal | 5008 | 4857 | 151 (-3%) |
| Local (wl=64, wr=0) | 7296 | 6217 | 1079 (-15%) |

### How R2P Works

Each `R2P` instruction converts 7 bits of a register byte into 7 predicate registers (1 instruction instead of 7 `ISETP` instructions):

```assembly
R2P PR, R9, 0x7f          ; bits 0-6 -> P0-P6
  14x FSEL using P0-P6     ; apply to elements
LOP3.LUT P0, RZ, R9, 0x80 ; bit 7 (leftover)
  2x FSEL using P0
```

32 elements: 4 R2P (28 elements) + 4 LOP3/ISETP (4 elements) = 32 total.

### Performance Impact

| Case | Old (ms) | New (ms) | Speedup |
|------|----------|----------|---------|
| Causal hdim=64 s=8192 | 2.463 | 2.473 | ~0% |
| Local hdim=64 s=8192 | 0.394 | 0.346 | +14% |
| Local hdim=128 s=8192 | 0.237 | 0.222 | +7% |

Causal sees no gain because masking is a tiny fraction of total work. Local sees significant gains because the sliding window has many partially-masked blocks.

---

## CLC Trace Debugging

### When to Use

When you suspect the CLC (Cooperative Load-balance Cluster) work scheduler is making surprising tile assignment decisions.

### Capturing a Trace

```bash
FA_LOG_LEVEL=3 FA_CLC=1 CUDA_VISIBLE_DEVICES=0 python - <<'PY' > trace.log 2>&1
import torch
from flash_attn.cute.interface import flash_attn_func
torch.manual_seed(0)
q = torch.randn(1, 512, 16, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 512, 1, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 512, 1, 128, device='cuda', dtype=torch.bfloat16)
flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
PY
```

### Trace Format

```text
[CLC] query sm=<smid> cta=<blockIdx.x> (m_blk=<m>,h=<h>,b=<b>,s=<s>) valid=<0|1>
```

### Parsing

```bash
python AI/parse_clc_log.py trace.log           # Text summary
python AI/parse_clc_log.py trace.log --html -o trace.html  # HTML view
```

### What to Look For

- `scheduling_mode=CLC` in host logs confirms CLC was used
- `valid=1` = valid work tile, `valid=0` = scheduler exhausted
- Multiple CTAs should get different tiles

---

## Compilation Issues

### JIT Compilation Failures

**Symptom**: `RuntimeError: Failed to compile kernel`

**Common causes**:
1. Missing CUTLASS dependencies
2. Incompatible CUDA toolkit version (need >= 11.8 for SM90)
3. Out of disk space for PTX/CUBIN cache
4. Incompatible PyTorch version

**Solutions**:
```bash
# Check CUDA toolkit version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Clear compilation cache
rm -rf /tmp/${USER}/flash_attention_cute_dsl_cache/

# Enable verbose compilation
CUTE_DSL_KEEP_PTX=1 python my_test.py
```

### Slow Compilation

FA4 kernels are JIT-compiled, which can be slow. Use the two-pass testing approach:

```bash
# Pass 1: Compile all kernels in parallel (no GPU needed)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: Run tests with cached kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -x tests/cute/test_flash_attn.py
```

---

## Memory Issues

### Out of Memory (OOM)

**Forward pass memory**: `B * H * S_q * D * sizeof(element) * 4` (Q, K, V, O) plus LSE buffer `B * H * S_q * sizeof(float)`.

**Backward pass memory**: Additional `B * H * S_q * D * sizeof(float)` for `dQ_accum` plus `B * H * S_k * D * sizeof(element) * 2` for dK and dV.

**Solutions**:
1. Use `flash_attn_varlen_func` to pack sequences and avoid padding waste
2. Use SplitKV parallelism for very long sequences (trades memory for compute)
3. Reduce batch size or sequence length
4. Use fp16 instead of fp32 for inputs
5. Use gradient checkpointing

### Workspace Memory

SplitKV forward allocates workspace:
```python
# Oaccum: (num_splits * B * H * S_q * D * sizeof(float))
# LSEaccum: (num_splits * B * H * S_q * sizeof(float))
```

For `num_splits=4, B=4, H=32, S=8192, D=128`:
- Oaccum: `4 * 4 * 32 * 8192 * 128 * 4 = 16 GB`
- LSEaccum: `4 * 4 * 32 * 8192 * 4 = 64 MB`

### Deterministic Backward Memory

When `deterministic=True`, each thread block writes to a separate `dQ_accum` buffer to avoid non-deterministic atomic operations:
```python
# Extra memory: gridDim.x * B * H * S_q * D * sizeof(float)
```

This can significantly increase memory usage for large grids.
