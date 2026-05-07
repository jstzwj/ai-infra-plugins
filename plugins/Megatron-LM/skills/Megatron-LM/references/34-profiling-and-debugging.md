# Chapter 34: Profiling and Debugging

## Source Files
- `sources/Megatron-LM/megatron/core/timers.py` - Timing utilities
- `sources/Megatron-LM/tools/` - Profiling tools

## Overview

Megatron-LM provides built-in profiling and debugging tools for analyzing training performance, identifying bottlenecks, and diagnosing issues.

## Built-in Timers

### Timer Usage
```python
from megatron.core.timers import Timers

timers = Timers()

# Start a timer
timers("forward").start()

# Stop and log
timers("forward").stop()
timers.log(["forward"])
```

### Training Timer Categories

| Timer | Description |
|---|---|
| `forward` | Forward pass computation |
| `backward` | Backward pass computation |
| `optimizer` | Optimizer step |
| `data-loading` | Data loading and preprocessing |
| `batch-generator` | Batch generation |
| `all-reduce` | Gradient all-reduce communication |
| `embedding` | Embedding computation |
| `mlp` | MLP layer computation |
| `attention` | Attention computation |
| `pipeline-send` | Pipeline parallel send |
| `pipeline-recv` | Pipeline parallel receive |

### Timing Configuration
```bash
--timing-log-level 2          # Log level (0=disabled, 1=summary, 2=detailed)
--timing-log-option max       # How to aggregate (max, min, avg)
--barrier-with-level-1-timing # Add barrier before L1 timers
--no-barrier-with-level-1-timing # Disable barrier
```

## PyTorch Profiler Integration

```bash
python pretrain_gpt.py \
    --profile \
    --profile-step-start 10 \
    --profile-step-end 15 \
    --profile-dir /path/to/profiles \
    [training args...]
```

## Nsight Systems Integration

```bash
# Profile with Nsight Systems
nsys profile -t cuda,nvtx,osrt \
    -s none \
    --output megatron-profile \
    --force-overwrite true \
    torchrun --nproc_per_node=8 pretrain_gpt.py [args...]
```

### NVTX Annotations
Megatron-LM uses NVTX ranges for marking key operations:
- Forward pass layers
- Backward pass layers
- Communication operations
- Data loading
- Optimizer step

## Memory Profiling

### GPU Memory Analysis
```python
import torch

# After model creation
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Peak memory tracking
torch.cuda.reset_peak_memory_stats()
# ... run training step ...
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Memory Breakdown
```
GPU Memory Usage Breakdown:
├── Model Parameters:     ~2× params × bytes_per_param / TP
├── Gradients:            ~1× params × bytes_per_param / DP
├── Optimizer States:     ~2× params × bytes_per_param (Adam: m + v)
├── Activations:          Variable, depends on batch size and sequence length
├── Temporary Buffers:    Communication buffers, workspace
└── Framework Overhead:   PyTorch CUDA allocator fragmentation
```

### Memory Estimation Formula
```
Memory per GPU (GB) ≈ (
    2 × num_params × bytes_per_param / TP       # Parameters
  + 1 × num_params × bytes_per_param / DP       # Gradients
  + 2 × num_params × bytes_per_param / DP       # Optimizer (m + v)
  + activation_memory                           # Activations
) / (1024³)

activation_memory ≈ (
    seq_len × batch_size × hidden_size × num_layers × bytes_per_element
  × (2 + num_attention_heads / num_query_groups)  # Q, K, V + attention
  / TP / PP / CP                                   # Parallelism savings
)
```

## Performance Metrics

### MFU (Model FLOP Utilization)
```python
# MFU = actual_flops / peak_flops
# actual_flops = 6 × num_params × tokens_per_second
# peak_flops = gpu_peak_tflops × num_gpus × 1e12

# GPT-3 175B on 512 H100 GPUs:
# actual_flops = 6 × 175e9 × 370000 / 1e12 = 388,500 TFLOPS
# peak_flops = 989 × 512 = 506,368 TFLOPS (H100 FP8)
# MFU = 388,500 / 506,368 ≈ 0.77 (with FP8)
# MFU = 388,500 / (495 × 512) ≈ 0.47 (with BF16)
```

### Key Performance Metrics
| Metric | Description | Target |
|---|---|---|
| MFU | Model FLOP Utilization | >40% for LLMs |
| TFLOPS/GPU | Per-GPU throughput | >300 (H100 BF16) |
| tokens/s | Token throughput | Depends on model size |
| Iteration time | Time per training step | Minimize |
| Memory utilization | GPU memory used/total | >80% |

## Common Debugging Scenarios

### Training Hang Diagnosis
1. Check NCCL logs: `NCCL_DEBUG=INFO`
2. Verify all ranks running: Check process status on all nodes
3. Check network connectivity: `nccl-net-scan`
4. Look for unbalanced pipeline stages

### OOM (Out of Memory) Diagnosis
1. Reduce `--micro-batch-size`
2. Enable `--recompute-granularity selective`
3. Enable `--use-distributed-optimizer`
4. Enable `--cpu-offloading`
5. Reduce `--seq-length`
6. Increase `--tensor-model-parallel-size`

### Numerical Issues
1. Check loss for NaN/Inf: `--check-for-nan-in-loss`
2. Reduce learning rate
3. Enable `--fp32-residual-connection`
4. Check `--attention-softmax-in-fp32`
5. Use `--deterministic-mode` for debugging
