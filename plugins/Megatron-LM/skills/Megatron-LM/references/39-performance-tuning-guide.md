# Chapter 39: Performance Tuning Guide

## Overview

This guide covers strategies for maximizing training throughput (MFU) and minimizing memory usage when training large transformer models with Megatron-LM.

## MFU (Model FLOP Utilization) Optimization

### What is MFU?
```
MFU = Actual TFLOPS / Peak Hardware TFLOPS

Actual TFLOPS = 6 × num_params × tokens_per_second / (num_gpus × 1e12)

Peak Hardware TFLOPS:
  H100 SXM: 989 TFLOPS (FP8), 495 TFLOPS (BF16)
  A100 SXM: 312 TFLOPS (FP16/BF16)
  H200 SXM: 989 TFLOPS (FP8), 495 TFLOPS (BF16)
```

### Target MFU Benchmarks

| Model Size | GPUs | Target MFU (BF16) | Target MFU (FP8) |
|---|---|---|---|
| 7B | 8 | 45-50% | 55-60% |
| 13B | 16 | 45-50% | 55-60% |
| 70B | 64 | 45-50% | 55-60% |
| 175B | 256 | 45-48% | 55-58% |
| 405B | 1024 | 42-46% | 50-55% |

## Parallelism Configuration Optimization

### Choosing TP Degree
- **Rule of thumb**: TP fits within NVLink domain (8 GPUs on H100)
- Each TP rank adds 2 AllReduce operations per layer (forward + backward)
- Larger TP = more communication overhead but less memory per GPU

```
Recommended TP sizes:
  hidden_size <= 4096:  TP=1-2
  hidden_size <= 8192:  TP=2-4
  hidden_size <= 16384: TP=4-8
  hidden_size > 16384:  TP=8
```

### Choosing PP Degree
- Use PP when model is too deep for TP alone
- Interleaved PP (virtual stages) reduces pipeline bubbles

```
Pipeline bubble fraction ≈ (PP - 1) / (PP × num_microbatches)

Recommended:
  num_layers <= 24:  PP=1-2
  num_layers <= 48:  PP=2-4
  num_layers <= 96:  PP=4-8
  num_layers > 96:   PP=8-16
```

### Choosing DP Degree
- DP adds no computation overhead (only gradient communication)
- Use the largest DP that fits memory after TP/PP allocation

```
DP = total_gpus / (TP × PP × CP × EP)
```

### Context Parallelism for Long Sequences
```
Recommended CP sizes:
  seq_length <= 4096:  CP=1
  seq_length <= 8192:  CP=1-2
  seq_length <= 32768: CP=2-4
  seq_length <= 131072: CP=4-8
```

## Communication Optimization

### Enable Communication Overlap
```bash
# TP communication overlap
--tp-comm-overlap

# Gradient reduction overlap
--overlap-grad-reduce

# Parameter gathering overlap
--overlap-param-gather
```

### NCCL Environment Variables
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1   # Critical for TP performance
export NCCL_ALGO=Ring                   # Force Ring algorithm
export NCCL_PROTO=Simple                # Simple protocol for latency
export NCCL_MIN_NCHANNELS=16            # Minimum channels
export NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA
export NCCL_IB_TC=106                   # InfiniBand traffic class
```

### Distributed Optimizer
```bash
--use-distributed-optimizer    # Saves ~30% optimizer memory
```

## Memory Optimization

### Activation Recomputation
```bash
# Selective: recomputes only attention (recommended for most cases)
--recompute-granularity selective

# Full: recomputes entire layer (maximum memory savings)
--recompute-granularity full --recompute-method uniform --recompute-num-layers 8
```

### Memory Budget Estimation
```
Per-GPU Memory (GB) for GPT models:

Parameters:  2 × P × bytes / TP
Gradients:   2 × P × bytes / DP
Optimizer:   2 × 2 × P × bytes / DP  (Adam: m + v, if not using dist optim)
Activations: ≈ 34 × L × s × b × h × bytes / (TP × PP)

Where:
  P = total parameters
  L = num_layers
  s = seq_length
  b = micro_batch_size
  h = hidden_size
  bytes = 2 for BF16, 1 for FP8
```

### CPU Offloading
```bash
--cpu-offloading                    # Enable CPU offloading
--cpu-offloading-num-layers 16      # Offload first N layers
--cpu-offloading-activations        # Offload activations
--cpu-offloading-weights            # Offload weights
```

### Fine-Grained Activation Offloading
```bash
--fine-grained-activation-offloading
--offload-modules core_attn,mlp     # Choose which modules to offload
```

## FP8 Training Optimization

### FP8 Configuration (Hopper/Blackwell)
```bash
--fp8-format e4m3             # Uniform FP8
--fp8-recipe delayed          # Delayed scaling (most common)
--fp8-param-gather            # FP8 parameters (saves memory)
--fp8-amax-history-len 256    # Longer history for stability
```

### FP8 Performance Impact
| Precision | Relative Throughput | Memory Savings |
|---|---|---|
| BF16 | 1.0x (baseline) | 0% |
| FP8 (delayed) | ~1.3-1.5x | ~20-30% |
| FP8 (mxfp8) | ~1.4-1.6x | ~25-35% |
| FP8 param + mxfp8 | ~1.4-1.6x | ~40-50% |

## CUDA Graphs Optimization

```bash
# For maximum throughput
--cuda-graph-impl local
--cuda-graph-scope full_iteration
--cuda-graph-warmup-steps 3
```

CUDA graphs eliminate kernel launch overhead, providing up to 15-20% speedup for smaller models where launch overhead is significant.

## MoE Performance Optimization

```bash
# Grouped GEMM for efficient expert computation
--moe-grouped-gemm

# EP communication overlap
--overlap-moe-expert-parallel-comm

# Shared expert overlap
--moe-shared-expert-overlap

# Token dispatcher optimization
--moe-token-dispatcher-type alltoall  # Better for EP > 1
--moe-permute-fusion                  # Fused permute operations
```

## Scaling Strategies

### Single Node (8 GPUs)
```
Best config: TP=8 or TP=4 + PP=2
Focus on: sequence parallelism, TP comm overlap
```

### Multi-Node (64 GPUs)
```
Best config: TP=8, PP=4, DP=2
Focus on: pipeline parallelism, NCCL tuning, gradient overlap
```

### Large Scale (512+ GPUs)
```
Best config: TP=8, PP=8, DP=8+
Focus on: distributed optimizer, communication overlap, FSDP
```

## Checklist: Performance Optimization Order

1. **Enable BF16/FP16**: `--bf16` (first step, 2x memory savings)
2. **Enable sequence parallelism**: `--sequence-parallel`
3. **Enable distributed optimizer**: `--use-distributed-optimizer`
4. **Enable FP8 (if Hopper+)**: `--fp8-format e4m3 --fp8-recipe delayed`
5. **Enable gradient overlap**: `--overlap-grad-reduce`
6. **Enable parameter overlap**: `--overlap-param-gather`
7. **Enable TP comm overlap**: `--tp-comm-overlap`
8. **Tune micro-batch-size**: Maximize without OOM
9. **Enable CUDA graphs**: `--cuda-graph-impl local`
10. **Tune NCCL**: `CUDA_DEVICE_MAX_CONNECTIONS=1`
11. **Enable FP8 params**: `--fp8-param-gather` (additional memory savings)
12. **Consider FSDP**: For very large models (100B+)
