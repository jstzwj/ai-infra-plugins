# Chapter 40: Troubleshooting FAQ

## Common Errors

### 1. CUDA Out of Memory (OOM)

**Error**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solutions** (in order of effectiveness):
1. Reduce `--micro-batch-size`
2. Enable activation recomputation: `--recompute-granularity selective`
3. Enable distributed optimizer: `--use-distributed-optimizer`
4. Enable sequence parallelism: `--sequence-parallel`
5. Increase TP: `--tensor-model-parallel-size 4→8`
6. Enable CPU offloading: `--cpu-offloading`
7. Reduce `--seq-length`
8. Enable FP8: `--fp8-format e4m3 --fp8-param-gather`

**Memory Estimation**:
```python
# Quick OOM check
import torch
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"Currently allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
```

### 2. Training Hang

**Symptoms**: All GPUs stuck, no progress, no error output

**Diagnosis**:
```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check process status on all nodes
pdsh -w node[01-08] "ps aux | grep python"

# Check GPU utilization
pdsh -w node[01-08] "nvidia-smi"
```

**Common Causes**:
1. **Network issue**: InfiniBand not configured, firewall blocking
2. **Uneven pipeline stages**: Wrong `--num-layers` for PP config
3. **Deadlock in custom code**: Ensure all ranks execute same communication ops
4. **RNG state mismatch**: Set `--deterministic-mode` for debugging

**Solutions**:
1. Check NCCL connectivity: `nccl-net-scan`
2. Verify `num_layers % (PP × VP) == 0`
3. Check all ranks reach same communication points
4. Enable timeout: `export NCCL_COMM_BLOCKING=0`

### 3. NaN Loss

**Error**: Loss becomes NaN during training

**Diagnosis**:
```bash
--check-for-nan-in-loss    # Enable NaN detection
--deterministic-mode        # Reproducible for debugging
```

**Common Causes**:
1. **Learning rate too high**: Reduce `--lr` by 10x
2. **Loss overflow in FP16**: Enable loss scaling `--loss-scale`
3. **Attention overflow**: Enable `--attention-softmax-in-fp32`
4. **Bad initialization**: Check `--init-method-std`
5. **Data corruption**: Verify data preprocessing

**Solutions**:
1. Reduce learning rate: `--lr 1e-5`
2. Enable FP32 residual: `--fp32-residual-connection`
3. Use BF16 instead of FP16: `--bf16` (wider dynamic range)
4. Check for data issues: inspect training samples

### 4. Model Parallelism Errors

**Error**: `ValueError: num_attention_heads must be a multiple of tensor_model_parallel_size`

**Solution**: Ensure `num_attention_heads % TP == 0`

| TP Size | Valid num_attention_heads |
|---|---|
| 1 | Any |
| 2 | 2, 4, 6, 8, 10, 12, 16, 24, 32, ... |
| 4 | 4, 8, 12, 16, 20, 24, 32, ... |
| 8 | 8, 16, 24, 32, 40, 48, 64, ... |

**Error**: `ValueError: Cannot use sequence parallelism without tensor parallelism`

**Solution**: `--sequence-parallel` requires `--tensor-model-parallel-size > 1`

### 5. Checkpoint Loading Errors

**Error**: `FileNotFoundError: checkpoint not found`

**Solutions**:
1. Check `--load` path is accessible from all nodes
2. Use absolute paths
3. Verify checkpoint format matches training config

**Error**: `RuntimeError: shape mismatch in parameter loading`

**Solutions**:
1. Use checkpoint rescaling: `--target-tensor-parallel-size`
2. Verify model config matches checkpoint
3. Check vocabulary size: `--make-vocab-size-divisible-by`

### 6. NCCL Communication Errors

**Error**: `NCCL error: unhandled system error`

**Solutions**:
```bash
# Force NCCL to use specific network
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

# Increase NCCL timeout
export NCCL_COMM_BLOCKING=0
export NCCL_MIN_NCHANNELS=4

# Disable P2P if NVLink issues
export NCCL_P2P_DISABLE=1
```

### 7. FP8 Training Issues

**Error**: `ValueError: fp8_param must be used together with fp8 mode`

**Solution**: `--fp8-param-gather` requires `--fp8-format e4m3` or `--fp8-format hybrid`

**Error**: FP8 accuracy degradation

**Solutions**:
1. Use delayed scaling: `--fp8-recipe delayed`
2. Increase AMAX history: `--fp8-amax-history-len 256`
3. Keep first/last layers in BF16: `--first-last-n-layers-in-bf16`
4. Reduce learning rate by 10-20%

### 8. MoE Training Issues

**Error**: Load imbalance (some experts get many more tokens)

**Solutions**:
1. Enable load balancing: `--moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.01`
2. Increase aux loss coefficient
3. Use router replay: `--moe-enable-routing-replay`
4. Try different score functions: `--moe-router-score-function sigmoid`

**Error**: MoE memory issues

**Solutions**:
1. Enable MoE recomputation: `--recompute-granularity selective --recompute-modules moe`
2. Use grouped GEMM: `--moe-grouped-gemm`
3. Increase EP: `--expert-model-parallel-size`

## Frequently Asked Questions

### Q: How do I choose between BF16 and FP16?
**A**: Use BF16 (recommended). It has the same dynamic range as FP32, avoiding the overflow issues common with FP16. FP16 requires loss scaling and is generally more finicky.

### Q: How do I enable Flash Attention?
**A**: Flash Attention is used automatically when TransformerEngine is installed. Ensure `--transformer-impl transformer_engine` (default).

### Q: What is sequence parallelism and should I enable it?
**A**: Sequence parallelism shards LayerNorm and dropout across TP ranks, reducing memory. Enable it whenever using TP: `--sequence-parallel`.

### Q: How many microbatches should I use?
**A**: `num_microbatches = global_batch_size / (micro_batch_size × DP)`. For pipeline parallelism, aim for `num_microbatches >= 4 × PP` to minimize pipeline bubbles.

### Q: How do I convert a HuggingFace model to Megatron format?
**A**: Use Megatron Bridge tools. See [Chapter 32: Megatron Bridge](32-megatron-bridge.md).

### Q: How do I resume training from a checkpoint?
**A**: Use `--load /path/to/checkpoint`. By default, optimizer state and RNG state are loaded. Use `--no-load-optim` or `--no-load-rng` to skip.

### Q: What's the minimum GPU memory required?
**A**:
- 7B model (BF16): ~28GB with TP=1, ~14GB with TP=2
- 70B model (BF16): ~140GB, requires TP=4+ on 80GB GPUs
- 175B model (BF16): ~350GB, requires TP=8 + PP=4+

### Q: How do I debug slow training?
**A**:
1. Enable timing: `--timing-log-level 2`
2. Check GPU utilization: `nvidia-smi dmon -s u`
3. Profile with PyTorch: `--profile --profile-step-start 10 --profile-step-end 12`
4. Check MFU and compare against targets in [Chapter 39](39-performance-tuning-guide.md)
