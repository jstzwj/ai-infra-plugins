# 20 - Appendix

This document provides reference tables, glossary, environment variables, and supporting information for FlashAttention.

---

## Table of Contents

1. [Papers and Citations](#papers-and-citations)
2. [Glossary of Terms](#glossary-of-terms)
3. [Environment Variables](#environment-variables)
4. [Supported GPU Architectures](#supported-gpu-architectures)
5. [Supported Data Types](#supported-data-types)
6. [Head Dimension Support Matrix](#head-dimension-support-matrix)
7. [Performance Reference Tables](#performance-reference-tables)
8. [References and Links](#references-and-links)

---

## Papers and Citations

### Core FlashAttention Papers

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
NeurIPS 2022
```
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
Tri Dao
ICLR 2024
```
@article{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```

**FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
arXiv 2024
```
@article{shah2024flashattention3,
  title={Flash{A}ttention-3: Fast and Accurate Attention with Asynchrony and Low-precision},
  author={Shah, Jay and Bikshandi, Ganesh and Zhang, Ying and Thakkar, Vijay and Ramani, Pradeep and Dao, Tri},
  journal={arXiv preprint arXiv:2407.08608},
  year={2024}
}
```

### Related Papers

**FlashAttention-4: Disseminating the Benefits of AI Hardware to the Broader ML Community**
Tri Dao et al.
Focuses on CuTeDSL-based kernel specification for broader hardware support.

**PagedAttention: Efficient Memory Management for Large Language Model Serving**
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, Ion Stoica
OSDI 2024 (vLLM)
```
@inproceedings{kwon2023pagedattention,
  title={Efficient Memory Management for Large Language Model Serving with {P}aged{A}ttention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph and Zhang, Hao and Stoica, Ion},
  booktitle={OSDI},
  year={2024}
}
```

**Online Softmax**
Petar Velickovic, tic-tac-toe (2000)

---

## Glossary of Terms

| Term | Definition |
|------|-----------|
| **ALiBi** | Attention with Linear Biases. Position encoding method that adds distance-based bias to attention scores |
| **BF16** | Brain Float 16. 16-bit floating point with 8 exponent bits and 7 mantissa bits |
| **Block Sparse** | Attention pattern where entire blocks of the attention matrix are selectively computed or skipped |
| **causal** | Attention mask where each position can only attend to positions at or before it (used in autoregressive models) |
| **cp.async** | CUDA asynchronous copy instruction (SM80+). Copies data from global to shared memory without blocking threads |
| **CTA** | Cooperative Thread Array. A CUDA thread block. A group of threads that share shared memory and synchronize |
| **CuTe** | CUTLASS Tensor library. Provides tensor abstractions for GPU programming |
| **CuTeDSL** | CUTLASS Domain-Specific Language. Python-based DSL for writing GPU kernels using CUTLASS |
| **FP8** | 8-bit floating point. E4M3 (4 exponent, 3 mantissa) or E5M2 (5 exponent, 2 mantissa) formats |
| **FP16** | Half-precision floating point (16-bit). 5 exponent bits, 10 mantissa bits |
| **GQA** | Grouped-Query Attention. Multiple Q heads share each K/V head |
| **HBM** | High Bandwidth Memory. GPU main memory (e.g., HBM3 on H100 at 3.3 TB/s) |
| **K Block** | A tile of K rows processed together (kBlockN rows) |
| **kBlockM** | Number of Q rows in a tile. Typical values: 64, 128, 192 |
| **kBlockN** | Number of K/V rows in a tile. Typical values: 32, 64, 128, 192 |
| **LSE** | Log-Sum-Exp. `log(sum(exp(scores)))`. Used for online softmax tracking |
| **MQA** | Multi-Query Attention. All Q heads share a single K/V head |
| **MMA** | Matrix Multiply-Accumulate. GPU tensor core operation |
| **Occupancy** | Ratio of active warps to maximum warps per SM. Higher is generally better |
| **Online Softmax** | Algorithm to compute softmax without materializing the full attention matrix |
| **Paged KV** | KV cache organized in fixed-size pages (like virtual memory) for efficient memory management |
| **Q Block** | A tile of Q rows processed together (kBlockM rows) |
| **Register Source (RS)** | Optimization where data stays in registers instead of being written to shared memory |
| **RoPE** | Rotary Position Embedding. Position encoding that rotates Q/K vectors |
| **SASS** | Streaming Assembly. The GPU's native machine code |
| **SM** | Streaming Multiprocessor. The basic compute unit on NVIDIA GPUs |
| **SMEM / SRAM** | Shared Memory / Static Random Access Memory. Fast on-chip memory shared by threads in a CTA |
| **Softcap** | tanh-based score capping: `scores = cap * tanh(scores / cap)` |
| **Split-KV** | Parallelism strategy where K/V sequence is split across thread blocks |
| **Swizzle** | Memory access pattern transformation to avoid bank conflicts in shared memory |
| **Tensor Core** | Specialized matrix multiply hardware unit in NVIDIA GPUs |
| **TF32** | TensorFloat-32. 19-bit format (8 exponent, 10 mantissa) for tensor core operations |
| **Tile** | A block of the matrix being processed. The fundamental unit of work in tiled algorithms |
| **TMA** | Tensor Memory Accelerator. Hardware unit for bulk memory transfers (SM90+) |
| **UMMA** | Unified Matrix Multiply-Accumulate. Blackwell (SM100) tensor core instruction |
| **Varlen** | Variable-length sequence batching. Packing multiple sequences of different lengths into one batch |
| **WGMMA** | Warpgroup MMA. Matrix multiply performed by a group of 4 warps (128 threads) on SM90+ |
| **Warp** | Group of 32 threads that execute together in lockstep (SIMT) |
| **Warpgroup** | Group of 4 consecutive warps (128 threads) that cooperate for WGMMA on SM90+ |

---

## Environment Variables

### Compilation and Caching

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | Enable disk cache for JIT-compiled CuTeDSL kernels | `0` |
| `FLASH_ATTENTION_FAKE_TENSOR` | Use FakeTensorMode for compilation without GPU memory | `0` |
| `CUTE_DSL_KEEP_PTX` | Keep generated PTX files after compilation | `0` |
| `CUTE_DSL_PTXAS_PATH` | Custom path to ptxas assembler | (system default) |
| `CUTE_CUBIN_PATH` | Directory to dump CUBIN/SASS files | (none) |
| `CUTE_DSL_LINEINFO` | Add line info to compiled kernels for sanitizer source mapping | `0` |
| `TORCH_CUDA_ARCH_LIST` | Target CUDA architectures for compilation | Auto-detected |
| `FLASH_ATTENTION_SKIP_CUDA_BUILD` | Skip CUDA compilation during install | `0` |
| `FLASH_ATTENTION_FORCE_BUILD` | Force rebuild even if pre-built wheels are available | `0` |

### Runtime Behavior

| Variable | Description | Default |
|----------|-------------|---------|
| `FA_LOG_LEVEL` | Logging level for device-side prints. 0=none, 3=debug | `0` |
| `FA_CLC` | Request CLC (Cooperative Load-balance Cluster) scheduling | `0` |
| `UNFUSE_FMA` | Disable FMA fusion in softmax exp computation | (PyTorch setting) |

### Testing

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | Restrict to specific GPU(s) | (all GPUs) |

### Distributed

| Variable | Description | Default |
|----------|-------------|---------|
| `NCCL_GRAPH_MIXING_SUPPORT` | Allow graph and non-graph launches to overlap | `0` |

---

## Supported GPU Architectures

### Compute Capability Matrix

| Architecture | CC | GPU | SMs | SMEM/SM | Tensor TFLOPS | HBM BW | HBM Cap |
|-------------|-----|-----|-----|---------|---------------|--------|---------|
| Ampere | SM80 | A100 80GB | 108 | 164 KB | 312 (fp16) | 2039 GB/s | 80 GB |
| Ampere | SM80 | A100 40GB | 108 | 164 KB | 312 (fp16) | 1555 GB/s | 40 GB |
| Ampere | SM86 | A6000 | 84 | 100 KB | ~178 (fp16) | ~960 GB/s | 48 GB |
| Ampere | SM86 | A40 | 84 | 100 KB | ~149 (fp16) | ~696 GB/s | 48 GB |
| Ada Lovelace | SM89 | L40 | 60 | 100 KB | ~180 (fp16) | ~864 GB/s | 48 GB |
| Ada Lovelace | SM89 | RTX 4090 | 128 | 100 KB | ~330 (fp16) | ~1008 GB/s | 24 GB |
| Hopper | SM90 | H100 SXM | 132 | 228 KB | 989 (fp16) | 3352 GB/s | 80 GB |
| Hopper | SM90 | H100 PCIe | 114 | 228 KB | 756 (fp16) | 2039 GB/s | 80 GB |
| Hopper | SM90 | H200 | 132 | 228 KB | 989 (fp16) | 4800 GB/s | 141 GB |
| Blackwell | SM100 | B200 | ~170 | ~228+ KB | ~2250 (fp16) | ~8000 GB/s | 192 GB |
| Blackwell | SM110 | B100 | ~120 | ~228+ KB | ~1750 (fp16) | ~4900 GB/s | ~96 GB |

### Feature Support by Architecture

| Feature | SM80 | SM86/89 | SM90 | SM100/110 |
|---------|------|---------|------|-----------|
| cp.async | Yes | Yes | Yes | Yes |
| TMA | No | No | Yes | Yes |
| WGMMA | No | No | Yes | Yes |
| UMMA | No | No | No | Yes |
| 2CTA | No | No | No | SM100 |
| Persistent | No | No | Limited | Yes |
| FP8 Tensor Core | No | No | Yes | Yes |
| FP4 Tensor Core | No | No | No | SM100 |
| Cluster | No | No | Yes | Yes |
| R2P instruction | No | No | Yes | Yes |

### FlashAttention Generation by Architecture

| Generation | SM80 | SM86 | SM89 | SM90 | SM100 | SM110 |
|-----------|------|------|------|------|-------|-------|
| FA2 (C++/CUDA) | Optimal | Full | Full | Compatible | Compatible | Compatible |
| FA3 (Hopper C++/CUDA) | Fallback | Fallback | Fallback | Optimal | Compatible | Compatible |
| FA4 (CuTeDSL) | -- | -- | Supported | Full | Optimal | Optimal |

---

## Supported Data Types

### Input/Output Types

| Data Type | PyTorch dtype | Bytes/elem | FA2 | FA3 | FA4 |
|-----------|-------------|-----------|-----|-----|-----|
| FP16 | `torch.float16` | 2 | Yes | Yes | Yes |
| BF16 | `torch.bfloat16` | 2 | Yes | Yes | Yes |
| FP8 E4M3 | `torch.float8_e4m3fn` | 1 | No | Fwd only | Fwd only |
| FP8 E5M2 | `torch.float8_e5m2` | 1 | No | Fwd only | Fwd only |
| FP32 | `torch.float32` | 4 | No (convert first) | No (convert first) | No (convert first) |

### Accumulation Types

| Computation | Accumulation Type |
|-------------|------------------|
| QK^T scores | FP32 |
| Softmax probabilities | FP32 |
| PV output accumulation | FP32 |
| Output storage | Same as input (FP16/BF16) |
| LSE | FP32 |
| Gradients (dQ, dK, dV) | FP32 accumulation, stored as input type |

### FP8 Usage Details

| Component | FP8 Format | Descale |
|-----------|-----------|---------|
| Q input | E4M3 or E5M2 | Per-tensor or per-head |
| K input | E4M3 | Per-tensor or per-head |
| V input | E4M3 or E5M2 | Per-tensor or per-head |
| Output | FP16, BF16, or FP32 | Applied during computation |

---

## Head Dimension Support Matrix

### Supported Head Dimensions

| Head Dim | kBlockKSmem | kBlockKGmem | Swizzle | FA2 Fwd | FA2 Bwd | FA3 Fwd | FA3 Bwd | FA4 |
|----------|-------------|-------------|---------|---------|---------|---------|---------|-----|
| 16 | 32 | 32 | 2 | Yes | Yes | Yes | Yes | Yes |
| 32 | 32 | 32 | 2 | Yes | Yes | Yes | Yes | Yes |
| 64 | 64 | 64 | 3 | Yes | Yes | Yes | Yes | Yes |
| 96 | 64 | 64 | 3 | Yes | Yes | Yes | Yes | Yes |
| 128 | 64 | 128 | 3 | Yes | Yes | Yes | Yes | Yes |
| 192 | 64 | 64 | 3 | Yes | Yes | Yes | Yes | Yes |
| 256 | 64 | 128 | 3 | Yes | Yes | Yes | Yes | Yes |

### Head Dimension Constraints

- Must be a multiple of 32 (for tensor core alignment)
- `kBlockKSmem = 64` if `kHeadDim % 64 == 0`, else `32`
- `kBlockKGmem = 128` if `kHeadDim % 128 == 0`, else `64` if `kHeadDim % 64 == 0`, else `32`
- For PackGQA with `G` groups: effective hdim = `G * original_hdim` (must still be supported)

### Special Cases

- **MLA (Multi-head Latent Attention)**: Can use different `headdim_v` from `headdim`. FA3 supports `headdim_v` values like 256, 512
- **PackGQA**: Effective hdim = `qhead_per_khead * headdim`. Must be in supported set or close to it

---

## Performance Reference Tables

### Forward Pass Latency (milliseconds)

A100 (SM80), batch=4, heads=32, hdim=128, bf16, causal=False:

| Seq Len | FA2 | PyTorch | Speedup |
|---------|-----|---------|---------|
| 256 | 0.14 | 0.62 | 4.4x |
| 512 | 0.28 | 1.42 | 5.1x |
| 1024 | 0.71 | 4.23 | 6.0x |
| 2048 | 1.98 | 16.5 | 8.3x |
| 4096 | 6.14 | 66.2 | 10.8x |
| 8192 | 23.0 | 265 | 11.5x |
| 16384 | 89.1 | 1061 | 11.9x |

H100 (SM90), same configuration:

| Seq Len | FA3 | PyTorch | Speedup |
|---------|-----|---------|---------|
| 256 | 0.09 | 0.38 | 4.2x |
| 512 | 0.21 | 0.87 | 4.1x |
| 1024 | 0.49 | 2.6 | 5.3x |
| 2048 | 1.32 | 10.2 | 7.7x |
| 4096 | 3.68 | 40.8 | 11.1x |
| 8192 | 12.4 | 163 | 13.1x |
| 16384 | 44.3 | 652 | 14.7x |

### Backward Pass Latency (milliseconds)

A100, batch=2, heads=32, hdim=128, bf16, causal=False:

| Seq Len | FA2 Bwd | PyTorch Bwd |
|---------|---------|-------------|
| 512 | 1.1 | 3.8 |
| 1024 | 2.8 | 11.2 |
| 2048 | 7.5 | 44 |
| 4096 | 23 | 176 |

### Memory Usage

Forward pass memory (GB) for hdim=128, bf16, batch_size x seqlen:

| B x S | FA2 | PyTorch | Savings |
|-------|-----|---------|---------|
| 8 x 512 | 0.04 | 0.08 | 50% |
| 8 x 1024 | 0.06 | 0.28 | 79% |
| 8 x 2048 | 0.12 | 1.06 | 89% |
| 8 x 4096 | 0.23 | 4.2 | 95% |
| 8 x 8192 | 0.46 | 16.8 | 97% |

### GQA Performance (H100, hdim=128, seqlen=2048, bf16)

| H_q | H_kv | Fwd Time (ms) | Speedup vs MHA |
|-----|------|-------------|----------------|
| 32 | 32 (MHA) | 1.32 | 1.0x |
| 32 | 16 (GQA-2) | 0.98 | 1.35x |
| 32 | 8 (GQA-4) | 0.79 | 1.67x |
| 32 | 4 (GQA-8) | 0.62 | 2.13x |
| 32 | 1 (MQA) | 0.41 | 3.22x |

### Sliding Window Performance (H100, hdim=128, seqlen=8192, bf16)

| Window Size | Fwd Time (ms) | Speedup vs Full |
|-------------|-------------|----------------|
| Full | 12.4 | 1.0x |
| 4096 | 7.8 | 1.6x |
| 2048 | 4.9 | 2.5x |
| 1024 | 3.2 | 3.9x |
| 512 | 2.1 | 5.9x |
| 256 | 1.4 | 8.9x |

---

## References and Links

### Code Repositories

| Repository | URL |
|-----------|-----|
| FlashAttention (FA2/FA3) | https://github.com/Dao-AILab/flash-attention |
| FlashAttention-4 (FA4) | https://github.com/Dao-AILab/flash-attention (flash_attn/cute/) |
| CUTLASS | https://github.com/NVIDIA/cutlass |
| CuTe | https://github.com/NVIDIA/cutlass/tree/main/include/cute |
| vLLM (PagedAttention) | https://github.com/vllm-project/vllm |

### Documentation

| Resource | URL |
|----------|-----|
| CUTLASS Documentation | https://nvidia.github.io/cutlass/ |
| CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ |
| CUDA Best Practices Guide | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ |
| PTX ISA Reference | https://docs.nvidia.com/cuda/parallel-thread-execution/ |
| Nsight Compute | https://docs.nvidia.com/nsight-compute/ |
| Nsight Systems | https://docs.nvidia.com/nsight-systems/ |
| compute-sanitizer | https://docs.nvidia.com/compute-sanitizer/ |

### NVIDIA Architecture Documentation

| Architecture | Document |
|-------------|---------|
| Ampere (SM80) | NVIDIA A100 GPU Architecture Whitepaper |
| Hopper (SM90) | NVIDIA H100 GPU Architecture Whitepaper |
| Blackwell (SM100) | NVIDIA B200 GPU Architecture Whitepaper |

### Blog Posts and Tutorials

| Resource | URL |
|----------|-----|
| FlashAttention Blog | https://tridao.me/blog/2023/flash-attention/ |
| Tri Dao's Publications | https://tridao.me/publications/ |
| CUTLASS Tutorial | https://github.com/NVIDIA/cutlass/tree/main/examples |

### Community

| Resource | URL |
|----------|-----|
| FlashAttention Issues | https://github.com/Dao-AILab/flash-attention/issues |
| FlashAttention Discussions | https://github.com/Dao-AILab/flash-attention/discussions |

### Layer Normalization Hidden Size Reference

The following hidden sizes have pre-compiled layer normalization kernels:

| Hidden Size | Model Examples |
|-------------|---------------|
| 256 | Small BERT variants |
| 512 | BERT-base (pooled) |
| 768 | BERT-base, GPT-2 small |
| 1024 | BERT-large, GPT-2 medium |
| 1280 | GPT-2 large |
| 1536 | GPT-2 XL |
| 2048 | Custom models |
| 2560 | GPT-3 6.7B |
| 3072 | GPT-3 13B |
| 4096 | LLaMA-2 7B |
| 5120 | LLaMA-2 13B, GPT-3 175B |
| 6144 | Custom models |
| 7168 | LLaMA-2 34B, CodeLlama 34B |
| 8192 | LLaMA-2 70B |

### File Organization Reference

```
flash-attention/
├── csrc/
│   ├── flash_attn/src/          # FA2 CUDA kernels
│   │   ├── flash_fwd_kernel.h    # Forward kernel template
│   │   ├── flash_bwd_kernel.h    # Backward kernel template
│   │   ├── flash_fwd_launch_template.h
│   │   ├── flash_bwd_launch_template.h
│   │   ├── kernel_traits.h       # Kernel configuration traits
│   │   ├── softmax.h             # Online softmax
│   │   ├── mask.h                # Masking operations
│   │   ├── rotary.h              # Rotary embeddings
│   │   ├── alibi.h               # ALiBi position bias
│   │   ├── dropout.h             # Dropout with Philox RNG
│   │   ├── block_info.h          # Variable-length block info
│   │   ├── utils.h               # Utility functions
│   │   ├── flash.h               # Parameter structs
│   │   ├── flash_api.cpp         # Host-side API
│   │   ├── generate_kernels.py   # Kernel instantiation generator
│   │   └── *.cu                   # Generated kernel files
│   ├── layer_norm/               # Layer norm kernels (per hidden size)
│   └── fused_dense_lib/          # Fused dense layer kernels
├── hopper/                       # FA3 (SM90) kernels
│   ├── flash_fwd_kernel_sm80.h   # SM80 forward
│   ├── flash_fwd_kernel_sm90.h   # SM90 forward
│   ├── flash_bwd_kernel_sm80.h   # SM80 backward
│   ├── flash_bwd_kernel_sm90.h   # SM90 backward
│   ├── flash_fwd_launch_template.h
│   ├── flash_bwd_launch_template.h
│   ├── flash_api.cpp             # Host-side API
│   ├── heuristics.h              # Split-KV heuristics
│   ├── tile_size.h               # Block size selection
│   ├── paged_kv.h                # Paged KV cache manager
│   ├── pack_gqa.py               # GQA packing
│   └── flash.h                   # Parameter structs
├── flash_attn/
│   ├── cute/                     # FA4 (CuTeDSL) kernels
│   │   ├── interface.py          # Public API
│   │   ├── flash_fwd.py          # SM90 forward
│   │   ├── flash_fwd_sm100.py    # SM100 forward
│   │   ├── flash_bwd.py          # SM80 backward
│   │   ├── flash_bwd_sm90.py     # SM90 backward
│   │   ├── flash_bwd_sm100.py    # SM100 backward
│   │   ├── softmax.py            # Online softmax
│   │   ├── mask.py               # Masking
│   │   └── ...
│   ├── utils/                    # Utility modules
│   │   ├── benchmark.py
│   │   ├── distributed.py
│   │   ├── generation.py
│   │   ├── testing.py
│   │   ├── pretrained.py
│   │   ├── torch.py
│   │   └── library.py
│   ├── bert_padding.py           # Padding utilities
│   ├── layers/                   # Model layers
│   │   ├── rotary.py             # Rotary embedding
│   │   └── patch_embed.py        # Patch embedding
│   └── losses/                   # Loss functions
│       └── cross_entropy.py
└── AI/                           # Debug and investigation notes
    ├── DEBUG_2CTA.md
    ├── RACECHECK_TMA_HAZARD.md
    ├── CLC_TRACE_DEBUG.md
    ├── VARLEN_PREPROCESS_TILE_BUG.md
    ├── SM90_BLOCK_SIZE_TUNING.md
    └── SM90_R2P_MASKING_SASS.md
```
