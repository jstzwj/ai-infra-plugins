# Chapter 21: Optimizer and Training Loop

## Source Files
- `megatron/core/optimizer/optimizer.py` - Base optimizer classes
- `megatron/core/optimizer/optimizer_config.py` - Optimizer configuration
- `megatron/core/optimizer/distrib_optimizer.py` - Distributed optimizer
- `megatron/core/optimizer/grad_scaler.py` - Loss scaling
- `megatron/core/optimizer/clip_grads.py` - Gradient clipping
- `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` - Hybrid CPU/GPU optimizer
- `megatron/core/optimizer/emerging_optimizers.py` - Muon, SOAP, AdaptiveMuon
- `megatron/core/optimizer/layer_wise_optimizer.py` - Layer-wise distributed optimizer
- `megatron/core/optimizer/optimizer_cuda_graph.py` - CUDA graph for optimizer step
- `megatron/training/training.py` - Training loop
- `megatron/core/optimizer_param_scheduler.py` - LR scheduling

## Optimizer Architecture

### Class Hierarchy

```
MegatronOptimizer (ABC)
├── FP32Optimizer                    -- Full FP32 training
├── MixedPrecisionOptimizer
│   ├── Float16OptimizerWithFloat16Params  -- FP16/BF16 mixed precision
│   └── DistributedOptimizer               -- Sharded optimizer state
└── ChainedOptimizer                 -- Chains multiple optimizers
    └── LayerWiseDistributedOptimizer -- Layer-wise sharding for emerging optimizers
```

### OptimizerConfig (`OptimizerConfig`)

The `OptimizerConfig` dataclass controls all optimizer behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | None | Initial learning rate |
| `min_lr` | None | Minimum learning rate for schedulers |
| `decoupled_lr` | None | Separate LR for input/output layers |
| `decoupled_min_lr` | None | Minimum LR for decoupled layers |
| `weight_decay` | 0.01 | L2 regularization coefficient |
| `optimizer` | 'adam' | Optimizer name ('adam', 'sgd', 'muon') |
| `fp16` | False | FP16 mixed precision |
| `bf16` | False | BF16 mixed precision |
| `adam_beta1` | 0.9 | Adam beta1 |
| `adam_beta2` | 0.999 | Adam beta2 |
| `adam_eps` | 1e-08 | Adam epsilon |
| `decoupled_weight_decay` | True | AdamW-style weight decay |
| `clip_grad` | 1.0 | Global gradient L2 norm clipping |
| `use_distributed_optimizer` | False | Shard optimizer state across DP |
| `optimizer_cpu_offload` | False | Offload optimizer state to CPU |
| `optimizer_offload_fraction` | 0.0 | Fraction of state to offload |
| `use_precision_aware_optimizer` | False | Lower precision optimizer tensors |
| `loss_scale` | None | Static loss scale (None=dynamic) |
| `initial_loss_scale` | 2^32 | Initial dynamic loss scale |
| `min_loss_scale` | 1.0 | Minimum loss scale |
| `loss_scale_window` | 1000 | Window for scale adjustment |
| `hysteresis` | 2 | Consecutive NaNs before scale-down |

## Supported Optimizers

### Adam / AdamW

The default optimizer. Uses Transformer Engine's `FusedAdam` when available, falls back to Apex `FusedAdam`, then PyTorch `Adam`.

```bash
--optimizer adam
--lr 1e-4
--adam-beta1 0.9
--adam-beta2 0.999
--adam-eps 1e-8
--weight-decay 0.01
--decoupled-weight-decency   # AdamW (default: True)
```

With `decoupled_weight_decay=True` (default), this is equivalent to AdamW. When False, the original Adam weight decay formulation is used.

### SGD

Standard stochastic gradient descent with optional momentum:

```bash
--optimizer sgd
--lr 0.1
--sgd-momentum 0.9
```

### Muon (Momentum via Orthogonalization)

Muon uses Newton-Schulz iteration to orthogonalize the momentum matrix, providing a spectral preconditoner. Non-linear/embedding parameters are routed to Adam by default.

```bash
--optimizer muon
--muon-momentum 0.95
--muon-nesterov False
--muon-scale-mode spectral
--muon-coefficient-type quintic
--muon-num-ns-steps 5
--muon-tp-mode blockwise
--muon-split-qkv True
--muon-scalar-optimizer adam
```

**TensorParallelMuon** handles tensor parallel weights with three modes:
- `blockwise`: Each TP shard is orthogonalized independently
- `duplicated`: NS iteration runs on duplicated weights, result is sharded
- `distributed`: NS iteration is aware of the TP partition dimension

### Adaptive Muon

Extends Muon with AdamW-style or NorMuon-style second moment accumulation:

```bash
--optimizer adaptive_muon
--adaptive-muon-moment2-method adamuon   # or normuon
--adaptive-muon-beta2 0.95
--adaptive-muon-eps 1e-8
```

### SOAP

Shampoo-based optimizer from `emerging_optimizers` package:

```bash
--optimizer soap
--soap-shampoo-beta 0.95
--soap-precondition-frequency 1
--soap-use-kl-shampoo True
```

## Distributed Optimizer

The `DistributedOptimizer` shards optimizer state across data-parallel ranks to reduce per-GPU memory. Instead of each DP rank storing the full optimizer state (momentum, variance, main parameters), each rank owns only a slice.

### Memory Savings

For a model with N parameters and DP world size D:

| Component | Standard | Distributed |
|-----------|----------|-------------|
| Model params | 2N bytes (BF16) | 2N bytes (BF16) |
| Main params (FP32) | 4N bytes | 4N/D bytes |
| Momentum (FP32) | 4N bytes | 4N/D bytes |
| Variance (FP32) | 4N bytes | 4N/D bytes |
| **Total optimizer** | **12N** | **12N/D** |

```bash
--use-distributed-optimizer
```

### How It Works

1. Parameters are grouped into contiguous buckets based on data type and parallelism type
2. Each bucket is divided into D contiguous shards (D = DP world size)
3. Each rank owns one shard per bucket and maintains FP32 main params only for its shard
4. During forward: all-gather gathers the full BF16 params from all shards
5. During backward: reduce-scatter reduces gradients and distributes to shard owners
6. During optimizer step: each rank updates only its owned shard
7. FP8 param gather: when using `--fp8-param-gather`, the gathered parameters are in FP8 format, reducing all-gather communication volume

### Precision-Aware Optimizer

With `--use-precision-aware-optimizer`, optimizer tensors can use lower precision:

```bash
--use-precision-aware-optimizer
--main-params-dtype bfloat16   # BF16 master weights
--exp-avg-dtype bfloat16       # BF16 momentum
--exp-avg-sq-dtype bfloat16    # BF16 variance
--main-grads-dtype bfloat16    # BF16 gradients
```

This requires TE >= 2.1.0 with FusedAdam and the distributed optimizer. When `store_param_remainders=True`, only the difference between FP32 and BF16 is stored, saving memory.

## CPU Offloading (HybridDeviceOptimizer)

The `HybridDeviceOptimizer` splits parameters between GPU and CPU for memory-constrained training:

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 0.5          # Fraction to offload
--overlap-cpu-optimizer-d2h-h2d           # Overlap data transfers
--use-torch-optimizer-for-cpu-offload     # Use torch.optim for CPU part
--pin-cpu-grads                           # Pin CPU grad memory
--pin-cpu-params                          # Pin CPU param memory
```

### How HybridDeviceOptimizer Works

1. Parameters are split between GPU and CPU based on `offload_fraction`
2. GPU optimizer (e.g., TE FusedAdam) handles GPU params
3. CPU optimizer (e.g., torch.optim.AdamW) handles offloaded params
4. When `overlap_cpu_optimizer_d2h_h2d=True`:
   - Each CPU parameter gets its own optimizer instance
   - Gradient D2H transfer runs on a separate CUDA stream
   - Parameter H2D transfer runs on another stream
   - GPU optimizer step overlaps with CPU data transfers
5. When `param_update_in_fp32=True`, main parameters are maintained in FP32 even for BF16 training

## Loss Scaling

### Static Loss Scaling

```bash
--loss-scale 4294967296   # 2^32
```

Uses `ConstantGradScaler` which never adjusts the scale.

### Dynamic Loss Scaling

Default when `--loss-scale` is not provided for FP16 training:

```bash
--initial-loss-scale 4294967296
--min-loss-scale 1.0
--loss-scale-window 1000
--hysteresis 2
```

`DynamicGradScaler` adjusts the scale:
- **Scale up**: Multiply by growth_factor (2.0) after `growth_interval` consecutive steps without overflow
- **Scale down**: Multiply by backoff_factor (0.5) after `hysteresis` consecutive steps with overflow
- The hysteresis counter prevents rapid scale oscillation

### BF16 Training

BF16 does not require loss scaling due to the wider exponent range. The grad scaler is None for BF16.

## Learning Rate Scheduling

The `OptimizerParamScheduler` supports multiple LR schedules:

### Cosine Annealing (default)
```bash
--lr-decay-style cosine
--lr 1e-4
--min-lr 1e-5
--lr-warmup-fraction 0.01
```

LR follows a cosine curve from `lr` to `min_lr` after the warmup phase.

### Polynomial Decay
```bash
--lr-decay-style polynomial
--lr-decay-power 2.0
```

### Constant LR
```bash
--lr-decay-style constant
```

### Inverse Square Root
```bash
--lr-decay-style inverse-square-root
```

### Warmup Strategies

```bash
--lr-warmup-fraction 0.01       # Fraction of total iterations
--lr-warmup-iters 2000          # Absolute number of iterations
--lr-warmup-init 1e-7           # Initial warmup LR
--override-opt_param-scheduler  # Override the scheduler
```

### Decoupled Learning Rate for Embeddings

Input embeddings and output layer can use a separate LR:

```bash
--decoupled-lr 1e-5
--decoupled-min-lr 1e-6
```

## Gradient Handling

### Gradient Clipping

Global L2 norm clipping is applied after gradient reduction:

```bash
--clip-grad 1.0
```

The clipping is done via `clip_grad_by_total_norm_fp32`:
1. Compute total gradient norm across all model-parallel and data-parallel ranks
2. Calculate clip coefficient: `clip_coeff = max_norm / (total_norm + 1e-6)`
3. Scale all gradients by min(clip_coeff, 1.0)

For efficiency, the L2 norm uses multi-tensor L2 norm kernels from Transformer Engine or Apex.

### Gradient Accumulation

Gradient accumulation is implicit via micro-batching:
1. The global batch is divided into micro-batches
2. Each micro-batch runs forward and backward, accumulating gradients
3. The optimizer step runs once per global batch

```bash
--global-batch-size 1024
--micro-batch-size 4       # Effective accumulation = 1024 / (4 * DP_size)
```

### Gradient Fusion

When using DDP with contiguous gradient buffers, gradients are stored in pre-allocated contiguous buffers. This avoids memory fragmentation and enables efficient all-reduce operations. The `overlap_grad_reduce` option overlaps gradient reduction with backward computation.

## Training Loop

The training loop in `megatron/training/training.py` follows this structure:

```
pretrain()
  ├── initialize_megatron()         # Setup parallel groups, random seeds
  ├── setup_model_and_optimizer()   # Build model, optimizer, LR scheduler
  │   ├── get_model()               # Build model provider
  │   ├── get_megatron_optimizer()  # Wrap optimizer with Megatron classes
  │   └── OptimizerParamScheduler() # LR schedule
  ├── train()
  │   └── for iteration in range(max_iters):
  │       ├── forward_backward_func()
  │       │   ├── forward_step() per micro-batch
  │       │   └── backward_step() per micro-batch
  │       ├── optimizer.prepare_grads()
  │       │   ├── copy model grads to main grads
  │       │   └── unscale and check for NaN/Inf
  │       ├── optimizer.get_grad_norm()
  │       ├── clip_grad_norm()
  │       ├── optimizer.step_with_ready_grads()
  │       │   ├── optimizer.step()
  │       │   └── copy main params back to model params
  │       ├── report_memory()
  │       ├── save_checkpoint()
  │       └── log training metrics
  └── destroy_global_state()
```

### Step Function Detail

The optimizer `step()` method:
1. `prepare_grads()`: Copy model gradients to main parameters; if using a grad scaler, unscale and check for overflow
2. If overflow detected, skip the update and return `(False, None, None)`
3. Compute gradient norm via `get_grad_norm()`
4. Clip gradients via `clip_grad_norm()` if `clip_grad > 0`
5. Count gradient zeros if `log_num_zeros_in_grad` is enabled
6. `step_with_ready_grads()`: Execute the base optimizer step and copy updated main params back to model params

### Mixed Precision Step

For FP16/BF16 training:
1. Model parameters are in FP16/BF16
2. A FP32 copy (main params) is maintained for the optimizer
3. Forward/backward use FP16/BF16 model params
4. After backward, gradients are copied from model params to main params
5. The optimizer updates the FP32 main params
6. Updated main params are copied back to FP16/BF16 model params

### ChainedOptimizer

When using multiple optimizers (e.g., Muon for linear layers + Adam for embeddings), `ChainedOptimizer` wraps them:
- Steps each optimizer sequentially
- Aggregates gradient norms across all sub-optimizers
- Supports RL offload/restore of all sub-optimizer states
- Manages checkpoint save/load across all sub-optimizers

### RL Optimizer Offloading

For RL training, the optimizer can be offloaded to CPU during inference to free GPU memory:

```python
optimizer.offload_to_cpu()    # Move all optimizer tensors to CPU
optimizer.restore_from_cpu()  # Restore back to GPU for training
```

## CUDA Graph for Optimizer Step

The `OptimizerCudaGraphWrapper` captures the optimizer step as a CUDA graph for reduced kernel launch overhead:

```bash
--optimizer-cuda-graph
```

This is especially beneficial when using the distributed optimizer with many small all-gather/reduce-scatter operations.

## Per-Parameter Optimizer Overrides

The `ParamKey` system allows different optimizer settings for different parameter groups:

```python
ParamKey(
    name="*weight*",                    # Glob pattern on parameter name
    attr="is_embedding_or_output_parameter",  # Match on parameter attribute
    predicate=ParamPredicate(...),       # Custom matching function
)
```

This enables configurations like:
- Using Adam for embeddings and Muon for linear layers
- Different weight decay for attention vs. MLP weights
- Different learning rates for biases vs. weights
