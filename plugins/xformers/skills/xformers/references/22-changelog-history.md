# 22 - Changelog & Version History

## Recent Releases

### [0.0.36] - 2026 (upcoming)

### [0.0.35] - 2026-02-20

Pre-built binary wheels available for PyTorch 2.10.0+.

**Improved:**
- Supported free-threading Python

**Removed:**
- Stopped bundling pre-built Flash-Attention 3; relies on PyTorch indices instead

### [0.0.34] - 2026-01-23

Pre-built binary wheels available for PyTorch 2.10.0+.

**Improved:**
- Migrated to PyTorch stable API/ABI - binary builds targeting PyTorch 2.10+ are compatible with any later version

**Removed:**
- Removed optimized fast-path of SwiGLU (was only available for A100)
- Removed most legacy components

### [0.0.33] - 2025-11-12

Pre-built binary wheels available for PyTorch 2.9.0.

**Added:**
- CUTLASS FMHA Op for Blackwell GPUs
- Support Flash-Attention up to 2.8.3
- Expose FA3 deterministic mode
- FW+BW pass overlap for DeepSeek-like comms/compute overlap

**Improved:**
- merge_attentions support for irregular head dimension

### [0.0.32] - 2025-08-13

**Added:**
- Support Flash-Attention up to 2.8.2
- Speed improvements to `python -m xformers.profiler.find_slowest`

**Removed:**
- Removed autograd backward pass for merge_attentions
- Attention biases are no longer `torch.Tensor` subclasses

### [0.0.31] - 2025-06-25

**Added:**
- Wheels are now Python-version agnostic (3.9-3.13)
- Added support for Flash-Attention 3 on Ampere GPUs

**Removed:**
- No longer support V100 or older GPUs (following PyTorch)
- Deprecated Flash-Attention 2 build; uses FA3 on Ampere

### [0.0.30] - 2025-04-28

**Added:**
- FMHA: Local attention on Flash3 backend (H100)
- FMHA: New paged gappy attention bias

**Improved:**
- FMHA: FlashAttention3 ships with more head dimensions for MLA
- Fused operators for sequence parallelism migrated to SymmetricMemory
- Profiler prepends traces' filenames with rank

### [0.0.29] - 2024-12-27

**Improved:**
- `LowerTriangularMask` no longer creates a CUDA tensor
- Updated Flash-Attention to v2.7.2.post1
- Flash-Attention v3 used by default when available (~10% faster training on H100)
- Fixed CUTLASS backward pass performance regression
- Fixed SwiGLU compatibility with torch-compile

**Removed:**
- No longer builds binaries for conda (pip only)
- Removed unmaintained/deprecated components

### [0.0.28] - 2024-09-12

**Added:**
- Wheels for CUDA 12.4
- Conda builds for Python 3.11
- Wheels for ROCm 6.1

**Improved:**
- Profiler: MFU/HFU calculation fixes and performance
- FMHA/splitK: Fixed `nan` with consecutive masked keys

**Removed:**
- FMHA: Removed `decoder` and `small_k` backends
- Profiler: Removed `DetectSlowOpsProfiler`
- Removed compatibility with PyTorch < 2.4

### [0.0.27] - 2024-07-10

**Added:**
- FMHA: `PagedBlockDiagonalGappyKeysMask`
- FMHA: Heterogeneous queries in `triton_splitk`
- FMHA: Paged attention in Flash
- FMHA: Backwards pass for `merge_attentions`
- FMHA: `torch.compile` support for biases
- 2:4 Sparsity: `sparsify24_ste` for Straight-Through Estimator

**Improved:**
- Fixed out-of-bounds reading for Split-K Triton
- Profiler: Manual trigger support

### [0.0.26] - 2024-04-29

**Added:**
- 2:4 Sparsity: STE gradient for `sparsify24`
- 2:4 Sparsity: `sparsify24_like` supports cuSparseLt backend
- Basic `torch.compile` support for `memory_efficient_attention`

**Improved:**
- `merge_attentions` no longer needs stacked inputs
- FMHA: Triton splitk supports additive bias

### [0.0.25] - 2024-03-14

**Added:**
- New `merge_attentions` function
- FMHA: Gappy attention biases
- FMHA: Triton splitk with LSE amalgamation, autotune, paged attention support

**Improved:**
- Updated Flash-Attention to v2.5.6 (multiquery improvement)
- Fixed `rope_padded` CUDA error with >65k queries
- Fixed `rmsnorm` CUDA error with large inputs

**Removed:**
- FMHA: Removed Triton operator (correctness issues)

### [0.0.24] - 2024-01-31

**Added:**
- Model/sequence parallelism components (Column&RowParallelLinear)
- 2:4 Sparsity training kernels with `sparsify24` API
- torch-compile compatible 2:4 sparsity

**Improved:**
- Selective activation checkpointing compatible with torch.compile

**Removed:**
- Triton kernels require compute capability >= 8.0 (A100+)
- Removed support for PyTorch < 2.1.0

### [0.0.23] - 2023-12-05

**Fixed:**
- FMHA: Fixed cutlass logsumexp bug with MQA
- Updated Flash-Attention to v2.3.6

## Key Historical Milestones

| Version | Date | Key Changes |
|---------|------|-------------|
| 0.0.34 | 2026-01 | PyTorch stable ABI migration |
| 0.0.33 | 2025-11 | Blackwell GPU support, FW+BW overlap |
| 0.0.31 | 2025-06 | Dropped V100, Python-agnostic wheels |
| 0.0.29 | 2024-12 | FA3 default on H100, removed legacy components |
| 0.0.27 | 2024-07 | Paged attention, torch.compile support |
| 0.0.24 | 2024-01 | Model/sequence parallelism, 2:4 sparsity |
| 0.0.23 | 2023-12 | Flash-Attention v2.3.6 |
| 0.0.22 | 2023-10 | Flash-Attention v2 integration |
| 0.0.20 | 2023-07 | Initial block-diagonal attention |
| 0.0.16 | 2023-01 | First Triton kernels |
| 0.0.14 | 2022-10 | Memory-efficient attention (CUTLASS) |
| 0.0.1 | 2022-01 | Initial release |

## Deprecation Timeline

| Feature | Removed | Version | Replacement |
|---------|---------|---------|-------------|
| V100 support | 0.0.31 | 2025-06 | Use A100+ |
| Flash-Attention 2 bundled build | 0.0.31 | 2025-06 | FA3 or PyTorch FA2 |
| SwiGLU optimized fast-path | 0.0.34 | 2026-01 | Eager PyTorch |
| Triton FMHA operator | 0.0.25 | 2024-03 | Flash/CUTLASS |
| `decoder` and `small_k` FMHA backends | 0.0.28 | 2024-09 | Flash/CUTLASS |
| Attention biases as Tensor subclasses | 0.0.32 | 2025-08 | Non-Tensor biases |
| Conda builds | 0.0.29 | 2024-12 | pip only |
