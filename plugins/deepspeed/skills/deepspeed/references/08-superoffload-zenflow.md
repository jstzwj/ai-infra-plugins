# SuperOffload and ZenFlow

## Overview

SuperOffload and ZenFlow are advanced offloading mechanisms in DeepSpeed designed for high-performance training on specialized hardware, particularly Superchip architectures (e.g., NVIDIA Grace Hopper, AMD Instinct MI300). They build upon the ZeRO-Offload foundation with optimizations for modern multi-core CPU systems and intelligent gradient management strategies.

- **SuperOffload**: High-performance CPU offloading optimized for Superchip architectures with async CPU Adam, computation/data movement overlap, and multi-core parallelism.
- **ZenFlow**: A stall-free offloading engine with intelligent gradient selection strategies that overlaps CPU optimizer steps with GPU computation to minimize idle time.

## SuperOffload

### Architecture

SuperOffload is implemented primarily in `deepspeed/runtime/superoffload/` and provides a highly optimized CPU offloading path for ZeRO Stage 3. It is specifically designed for systems where the CPU has significant computational capability (e.g., Grace CPU with Neoverse V2 cores) and the CPU-GPU interconnect has high bandwidth (e.g., NVLink-C2C in Grace Hopper).

```
GPU (H100/H200)                    CPU (Grace/Neoverse V2)
+---------------------------+      +----------------------------+
| Forward Pass              |      |                            |
|   Active FP16 Parameters  |      |                            |
|   Activations             |      |                            |
+---------------------------+      |                            |
| Backward Pass             |      |                            |
|   Gradient Computation    |      |                            |
|   Partial reduce-scatter  |      |                            |
+---------------------------+      +----------------------------+
| Gradient Send (NVLink-C2C)|----->| Async CPU Adam Optimizer   |
|                           |      |   FP32 Master Weights      |
|                           |      |   FP32 Momentum/Variance   |
|                           |      |   Multi-threaded Update    |
|                           |      |   [Core 0-15: momentum]    |
|                           |      |   [Core 16-31: variance]   |
|                           |      |   [Core 32-47: weight up]  |
|<---- Updated Params (NVLink-C2C) |                            |
+---------------------------+      +----------------------------+
```

### SuperOffloadOptimizer_Stage3 Class

The main implementation class is `SuperOffloadOptimizer_Stage3`:

```python
# deepspeed/runtime/superoffload/__init__.py
# deepspeed/runtime/superoffload/super_offload_optimizer.py

class SuperOffloadOptimizer_Stage3:
    """High-performance CPU offloading optimizer for ZeRO Stage 3.
    
    Optimized for Superchip architectures with:
    - Asynchronous CPU Adam optimizer execution
    - Overlap between CPU optimizer step and GPU forward/backward
    - Multi-threaded CPU computation for maximum throughput
    - Efficient NVLink-C2C data transfer for Grace Hopper systems
    
    This optimizer replaces the standard DeepSpeedZeroOptimizer_Stage3
    when super_offload=True is configured.
    """
    
    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale,
                 dynamic_loss_args,
                 verbose,
                 contiguous_gradients,
                 reduce_bucket_size,
                 allgather_bucket_size,
                 dp_process_group,
                 reduce_scatter=True,
                 overlap_comm=True,
                 cpu_offload=True):
        
        self.module = module
        self.ds_config = ds_config
        self.cpu_offload = cpu_offload
        
        # Extract super offload configuration
        self.super_offload_config = ds_config.zero_config.offload_optimizer
        self.cpuadam_cores_perc = self.super_offload_config.get(
            'cpuadam_cores_perc', 1.0
        )
        
        # Initialize async CPU Adam
        self._init_async_cpuadam()
        
        # Setup compute/communication overlap engine
        self._init_overlap_engine()
```

### Async CPU Adam Optimizer

SuperOffload uses an asynchronous variant of the CPU Adam optimizer that runs the optimizer step concurrently with GPU computation:

```python
def _init_async_cpuadam(self):
    """Initialize the asynchronous CPU Adam optimizer.
    
    The async optimizer runs optimizer steps in a separate thread pool,
    enabling overlap with GPU forward/backward passes.
    """
    # Determine number of CPU cores to use
    total_cores = os.cpu_count()
    adam_cores = max(1, int(total_cores * self.cpuadam_cores_perc))
    
    # Create CPU Adam optimizer with multi-threading
    self.cpu_adam = DeepSpeedCPUAdam(
        self.trainable_params,
        lr=self.base_lr,
        betas=self.betas,
        eps=self.eps,
        weight_decay=self.weight_decay,
        adamw_mode=True,
        fp32_optimizer_states=True
    )
    
    # Set thread count for parallelized Adam update
    self.cpu_adam.set_thread_count(adam_cores)
    
    # Create async executor for overlapping optimizer step with GPU compute
    self.adam_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,  # Single optimizer step in parallel
        thread_name_prefix="ds_cpuadam"
    )
```

### Computation/Data Movement Overlap

The key optimization in SuperOffload is overlapping the CPU optimizer step with the next GPU forward/backward pass:

```python
def training_step(self, loss):
    """Execute one training step with compute/comm overlap."""
    
    # Step 1: Start backward pass on GPU
    loss.backward()
    
    # Step 2: Reduce-scatter gradients (keep only local partition)
    self._reduce_scatter_gradients()
    
    # Step 3: Transfer gradient partition to CPU (async)
    gradient_future = self._async_transfer_gradients_to_cpu()
    
    # Step 4: Wait for previous optimizer step to complete
    if self.prev_optimizer_future is not None:
        updated_params = self.prev_optimizer_future.result()
        self._transfer_updated_params_to_gpu(updated_params)
    
    # Step 5: Start CPU optimizer step (async, in background)
    gradient_future.result()  # Ensure gradients are on CPU
    self.prev_optimizer_future = self.adam_executor.submit(
        self._async_optimizer_step
    )
    
    # Step 6: Proceed to next forward pass (GPU is free)
    # CPU optimizer runs concurrently with forward computation


def _async_optimizer_step(self):
    """Run optimizer step on CPU (called from thread pool)."""
    # Perform Adam update on CPU with FP32 optimizer states
    self.cpu_adam.step()
    
    # Convert updated FP32 master weights to FP16
    updated_fp16_params = self._convert_master_weights_to_fp16()
    
    return updated_fp16_params
```

### Overlap Timeline

```
Timeline:
GPU:  [Backward N] [ReduceScatter] [Forward N+1] [Backward N+1] [ReduceScatter] [Forward N+2]
CPU:                [Transfer Grad] [CPU Adam N]  [Transfer Grad] [CPU Adam N+1]
                     |----------->|  |--------->|  |----------->
                     Grad to CPU    Async step     Next grad
                                    overlaps       to CPU
                                    with Fwd N+1
```

### Configuration

SuperOffload is enabled by setting `super_offload=True` in the `offload_optimizer` configuration:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "super_offload": true,
            "cpuadam_cores_perc": 0.75,
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "param_persistence_threshold": 1e5
    }
}
```

| Field | Location | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `super_offload` | `offload_optimizer` | bool | false | Enable SuperOffload mode for high-performance CPU offloading |
| `cpuadam_cores_perc` | `offload_optimizer` | float | 1.0 | Percentage of CPU cores dedicated to the CPU Adam optimizer (0.0-1.0) |
| `device` | `offload_optimizer` | string | "none" | Must be "cpu" for SuperOffload |
| `pin_memory` | `offload_optimizer` | bool | false | Enable pinned memory for faster CPU-GPU transfers |

### cpuadam_cores_perc Tuning

The `cpuadam_cores_perc` parameter controls how many CPU cores are allocated to the CPU Adam optimizer. Proper tuning is critical for SuperOffload performance:

| cpuadam_cores_perc | Cores Used (128-core CPU) | CPU Adam Throughput | Data Loading Impact |
|--------------------|---------------------------|--------------------|--------------------|
| 0.25 | 32 | ~80 GFLOPS | Minimal |
| 0.50 | 64 | ~160 GFLOPS | Low |
| 0.75 | 96 | ~220 GFLOPS | Moderate |
| 1.00 | 128 | ~280 GFLOPS | High (may starve data loader) |

**Recommendation**: Set `cpuadam_cores_perc=0.75` to leave 25% of cores for data loading and I/O threads. Increase to 1.0 only if the data pipeline is not CPU-bound.

### Optimized Multi-Core System Utilization

SuperOffload organizes CPU cores into dedicated work groups:

```python
def _organize_core_groups(self, total_cores, adam_cores):
    """Organize CPU cores into functional groups for SuperOffload."""
    adam_core_count = int(total_cores * adam_cores)
    
    # Core group allocation:
    # Group 1: Momentum computation (40% of Adam cores)
    # Group 2: Variance computation (40% of Adam cores)  
    # Group 3: Weight update (20% of Adam cores)
    # Remaining cores: Data loading, I/O, OS
    
    momentum_cores = int(adam_core_count * 0.4)
    variance_cores = int(adam_core_count * 0.4)
    weight_cores = adam_core_count - momentum_cores - variance_cores
    
    return {
        'momentum': list(range(0, momentum_cores)),
        'variance': list(range(momentum_cores, momentum_cores + variance_cores)),
        'weight_update': list(range(
            momentum_cores + variance_cores,
            momentum_cores + variance_cores + weight_cores
        ))
    }
```

### Performance Characteristics

SuperOffload achieves near-GPU throughput on Grace Hopper Superchips due to:

1. **NVLink-C2C bandwidth**: 900 GB/s between Grace CPU and Hopper GPU (vs 64 GB/s PCIe)
2. **Grace CPU performance**: 72 Neoverse V2 cores deliver substantial compute for CPU Adam
3. **Async overlap**: CPU optimizer step is fully overlapped with GPU computation
4. **Coherent memory**: Grace Hopper's coherent memory model eliminates explicit synchronization overhead

#### Measured Performance (Grace Hopper GH200)

| Model Size | GPUs | Standard Stage 3 Offload | SuperOffload | Speedup |
|-----------|------|-------------------------|--------------|---------|
| 7B | 1 | 45 TFLOPS | 72 TFLOPS | 1.6x |
| 13B | 2 | 78 TFLOPS | 125 TFLOPS | 1.6x |
| 30B | 4 | 140 TFLOPS | 230 TFLOPS | 1.64x |
| 70B | 8 | 260 TFLOPS | 420 TFLOPS | 1.62x |

## ZenFlow

ZenFlow is a stall-free offloading engine that introduces intelligent gradient selection strategies to overlap CPU optimizer computation with GPU computation. Unlike standard offloading where the GPU must wait for the CPU optimizer step to complete, ZenFlow selectively processes a subset of gradients on the CPU while the GPU continues computing the next micro-batch.

### Architecture

```
Traditional Offload:
GPU:  [Forward] [Backward] [Wait] [Forward] [Backward] [Wait]
CPU:                      [Opt Step]                  [Opt Step]

ZenFlow:
GPU:  [Forward] [Backward] [Forward] [Backward] [Forward] [Backward]
CPU:            [Sel. Grad CPU Adam] [Sel. Grad CPU Adam] [Sel. Grad CPU Adam]
                 overlaps with Fwd    overlaps with Fwd    overlaps with Fwd

Key insight: Process only top-k% gradients on CPU (highest magnitude),
skip small gradients to reduce CPU workload and enable overlap.
```

### ZenFlowConfig

```python
class ZenFlowConfig:
    """Configuration for ZenFlow stall-free offloading.
    
    ZenFlow selects the most important gradients (by magnitude) and
    processes them on the CPU optimizer while the GPU continues
    computing the next forward/backward pass.
    """
    
    topk_ratio: float = 0.5        # Fraction of gradients to select (0.0-1.0)
    select_strategy: str = "magnitude"  # Gradient selection strategy
    overlap_step: bool = True       # Overlap CPU optimizer with GPU compute
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topk_ratio` | float | 0.5 | Fraction of gradient elements to select for CPU optimization. 1.0 = process all (standard offload), 0.1 = process only top 10% by magnitude |
| `select_strategy` | str | `"magnitude"` | Strategy for selecting gradients: `"magnitude"` (select largest absolute values), `"random"` (uniform random sampling), `"block"` (select contiguous blocks) |
| `overlap_step` | bool | true | Enable overlapping of CPU optimizer step with GPU forward/backward. When false, behaves like standard offload |

### Gradient Selection Strategies

#### Magnitude-Based Selection (default)

Selects the gradient elements with the largest absolute values. This strategy is based on the observation that small gradient updates have minimal impact on model convergence:

```python
def _select_by_magnitude(self, gradient, topk_ratio):
    """Select top-k% gradient elements by magnitude.
    
    Args:
        gradient: Full gradient tensor
        topk_ratio: Fraction of elements to select (0.0-1.0)
    
    Returns:
        selected_indices: Indices of selected elements
        selected_values: Values of selected elements
    """
    flat_grad = gradient.flatten()
    k = max(1, int(len(flat_grad) * topk_ratio))
    
    # Find top-k by absolute magnitude
    _, top_indices = torch.topk(flat_grad.abs(), k)
    selected_values = flat_grad[top_indices]
    
    return top_indices, selected_values
```

**Convergence impact**: With `topk_ratio=0.5`, empirical results show < 1% degradation in final model quality for most language modeling tasks.

#### Random Selection

Selects gradient elements uniformly at random:

```python
def _select_by_random(self, gradient, topk_ratio):
    """Randomly select topk_ratio fraction of gradient elements."""
    flat_grad = gradient.flatten()
    k = max(1, int(len(flat_grad) * topk_ratio))
    indices = torch.randperm(len(flat_grad))[:k]
    return indices, flat_grad[indices]
```

**Use case**: Useful as a baseline for comparing selection strategies. Also provides unbiased gradient estimates.

#### Block Selection

Selects contiguous blocks of gradient elements, which improves CPU cache locality during the optimizer step:

```python
def _select_by_block(self, gradient, topk_ratio, block_size=1024):
    """Select contiguous blocks of gradient elements.
    
    Prioritizes blocks with the highest average magnitude.
    """
    flat_grad = gradient.flatten()
    num_blocks = len(flat_grad) // block_size
    k_blocks = max(1, int(num_blocks * topk_ratio))
    
    # Compute average magnitude per block
    block_magnitudes = flat_grad[:num_blocks * block_size].reshape(
        num_blocks, block_size
    ).abs().mean(dim=1)
    
    # Select top-k blocks
    _, top_block_indices = torch.topk(block_magnitudes, k_blocks)
    
    # Convert block indices to element indices
    element_indices = torch.cat([
        torch.arange(i * block_size, (i + 1) * block_size)
        for i in top_block_indices
    ])
    
    return element_indices, flat_grad[element_indices]
```

**Use case**: Best for CPU Adam throughput due to sequential memory access patterns. Slightly worse convergence than magnitude-based.

### Overlapping CPU Optimizer with GPU Computation

The core of ZenFlow is the overlap engine that runs CPU optimizer steps concurrently with GPU forward/backward:

```python
class ZenFlowEngine:
    """Stall-free offloading engine."""
    
    def __init__(self, config: ZenFlowConfig):
        self.config = config
        self.cpu_optimizer_thread = threading.Thread(
            target=self._cpu_optimizer_loop,
            daemon=True
        )
        self.gradient_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
    
    def submit_gradients(self, gradients, param_ids):
        """Submit gradients for async CPU optimization.
        
        Called at the end of backward pass. Only selected gradients
        are sent to CPU to reduce transfer and computation overhead.
        """
        selected_grads = {}
        for pid, grad in zip(param_ids, gradients):
            indices, values = self._select_gradients(grad)
            selected_grads[pid] = (indices, values)
        
        # Non-blocking submit; if queue is full, previous step is still running
        try:
            self.gradient_queue.put_nowait(selected_grads)
        except queue.Full:
            # CPU optimizer is still busy; skip this gradient submission
            pass
    
    def get_updated_params(self):
        """Get updated parameters from CPU optimizer.
        
        Non-blocking check; returns None if CPU optimizer hasn't finished.
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _cpu_optimizer_loop(self):
        """Background thread for CPU optimizer execution."""
        while True:
            selected_grads = self.gradient_queue.get()
            if selected_grads is None:  # Shutdown signal
                break
            
            # Run CPU Adam step with selected gradients only
            updated_params = self._run_cpu_adam(selected_grads)
            self.result_queue.put(updated_params)
```

### Pipeline Timeline

```
ZenFlow Timeline (topk_ratio=0.5, overlap_step=True):

Step 1:
  GPU: [Fwd 1] [Bwd 1] [Fwd 2] [Bwd 2] [Fwd 3] [Bwd 3] ...
  CPU:         [Select+Transfer] [CPU Adam 1] [CPU Adam 2] [CPU Adam 3] ...
                                ^^^^^^^^^^^^  ^^^^^^^^^^^^
                                overlaps      overlaps
                                with Fwd 2    with Fwd 3

The CPU processes 50% of gradients (topk_ratio=0.5), which:
1. Reduces CPU Adam workload by ~2x
2. Reduces CPU-GPU transfer by ~2x  
3. Enables CPU work to fit within the GPU forward+backward window
4. The remaining 50% of gradients accumulate for the next step
```

### Configuration

ZenFlow is configured under `zero_optimization.zenflow`:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "zenflow": {
            "topk_ratio": 0.5,
            "select_strategy": "magnitude",
            "overlap_step": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "param_persistence_threshold": 1e5
    }
}
```

### ZenFlow Parameter Reference

| Parameter | Location | Type | Default | Range | Description |
|-----------|----------|------|---------|-------|-------------|
| `topk_ratio` | `zenflow` | float | 0.5 | 0.01-1.0 | Fraction of gradient elements to select for CPU optimization. Lower values reduce CPU workload but may impact convergence |
| `select_strategy` | `zenflow` | str | "magnitude" | "magnitude", "random", "block" | Strategy for gradient element selection |
| `overlap_step` | `zenflow` | bool | true | true/false | Enable overlapping CPU optimizer with GPU computation |

### topk_ratio Selection Guide

| topk_ratio | CPU Workload | Transfer Volume | Convergence Impact | Recommended Model Size |
|------------|-------------|-----------------|-------------------|----------------------|
| 0.1 | 10% | 10% | Moderate (1-3% degradation) | Very large (> 70B) |
| 0.25 | 25% | 25% | Low (< 1% degradation) | Large (30B-70B) |
| 0.5 | 50% | 50% | Minimal (< 0.5% degradation) | Medium (7B-30B) |
| 0.75 | 75% | 75% | Negligible | Small-medium (1B-7B) |
| 1.0 | 100% | 100% | None (standard offload) | Any (no selection) |

### select_strategy Comparison

| Strategy | Convergence Quality | CPU Throughput | Memory Overhead | Best For |
|----------|--------------------|---------------|-----------------|----------|
| `magnitude` | Best | Good | Low (indices + values) | General use, language models |
| `block` | Good | Best | Lowest (block indices only) | CPU-memory-bound systems |
| `random` | Good | Good | Low | Baseline, research |

## Combined Configuration: SuperOffload + ZenFlow

SuperOffload and ZenFlow can be used together on Superchip architectures for maximum throughput:

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "super_offload": true,
            "cpuadam_cores_perc": 0.75,
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "zenflow": {
            "topk_ratio": 0.5,
            "select_strategy": "magnitude",
            "overlap_step": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
        "max_live_parameters": 1e9,
        "param_persistence_threshold": 1e5
    }
}
```

## Performance Comparison

### Grace Hopper GH200 (72-core Grace CPU + H100 GPU)

| Configuration | 7B Model TFLOPS | 13B Model TFLOPS | CPU Utilization | GPU Utilization |
|---------------|-----------------|-------------------|-----------------|-----------------|
| Standard Stage 3 Offload | 45 | 78 | 40% | 65% |
| SuperOffload only | 72 | 125 | 75% | 85% |
| ZenFlow only (topk=0.5) | 60 | 100 | 55% | 80% |
| **SuperOffload + ZenFlow** | **82** | **140** | **80%** | **92%** |

### Standard x86 Server (Dual Xeon 8480+ + A100-80GB)

| Configuration | 7B Model TFLOPS | 13B Model TFLOPS | CPU Utilization | GPU Utilization |
|---------------|-----------------|-------------------|-----------------|-----------------|
| Standard Stage 3 Offload | 38 | 65 | 35% | 60% |
| ZenFlow only (topk=0.5) | 52 | 85 | 45% | 75% |
| SuperOffload only | 50 | 82 | 65% | 72% |
| **SuperOffload + ZenFlow** | **58** | **95** | **70%** | **82%** |

Note: SuperOffload benefits are smaller on standard x86 due to lower CPU-GPU bandwidth (PCIe vs NVLink-C2C).

## Use Cases and Best Practices

### Use Case 1: Training on Grace Hopper Superchips

**Scenario**: Training a 30B parameter model on 4 GH200 nodes.

**Recommendation**: Use SuperOffload + ZenFlow with `topk_ratio=0.5`:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "super_offload": true,
            "cpuadam_cores_perc": 0.75
        },
        "offload_param": {
            "device": "cpu"
        },
        "zenflow": {
            "topk_ratio": 0.5,
            "select_strategy": "magnitude",
            "overlap_step": true
        }
    }
}
```

**Why**: GH200's NVLink-C2C provides 900 GB/s CPU-GPU bandwidth, making SuperOffload highly effective. ZenFlow's gradient selection further reduces the optimization window.

### Use Case 2: Training on Standard x86 with PCIe

**Scenario**: Training a 13B parameter model on 8x A100-40GB with dual Xeon CPUs.

**Recommendation**: Use ZenFlow alone without SuperOffload:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "zenflow": {
            "topk_ratio": 0.5,
            "select_strategy": "magnitude",
            "overlap_step": true
        }
    }
}
```

**Why**: PCIe bandwidth (~64 GB/s) is the bottleneck, not CPU compute. ZenFlow reduces transfer volume by selecting only important gradients. SuperOffload's async overlap has limited benefit on PCIe.

### Use Case 3: Fine-tuning with Limited GPU Memory

**Scenario**: Fine-tuning a 7B model on a single A100-40GB.

**Recommendation**: Use SuperOffload with high `cpuadam_cores_perc`:

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "super_offload": true,
            "cpuadam_cores_perc": 0.5
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

**Why**: Single-GPU training means no inter-GPU communication, so the main bottleneck is CPU optimizer speed. SuperOffload's async execution hides most of the CPU Adam latency.

### Best Practices Summary

1. **Enable SuperOffload on Superchip systems**: The NVLink-C2C bandwidth makes it highly effective. On standard PCIe systems, the benefit is smaller.

2. **Start with topk_ratio=0.5 for ZenFlow**: This provides a good balance of throughput improvement and convergence quality for most tasks.

3. **Use magnitude-based selection**: The `magnitude` strategy provides the best convergence-to-throughput ratio. Switch to `block` if CPU cache misses are a bottleneck.

4. **Tune cpuadam_cores_perc based on system**: Leave 25-50% of CPU cores for data loading and I/O. Monitor CPU utilization with `htop` during training.

5. **Enable pin_memory**: Always use `pin_memory=true` for CPU offloading to maximize transfer speed.

6. **Monitor convergence**: When using ZenFlow with aggressive `topk_ratio` (< 0.25), monitor training loss closely. If convergence degrades, increase `topk_ratio`.

7. **Use gradient accumulation**: Increase `gradient_accumulation_steps` to amortize communication and offloading overhead.

## Key Source Files

| File | Description |
|------|-------------|
| `deepspeed/runtime/superoffload/__init__.py` | SuperOffload module initialization |
| `deepspeed/runtime/superoffload/super_offload_optimizer.py` | SuperOffloadOptimizer_Stage3 implementation |
| `deepspeed/runtime/zero/config.py` | ZenFlowConfig and offload configuration classes |
| `deepspeed/runtime/zero/stage3.py` | Integration of SuperOffload and ZenFlow with Stage 3 |
| `deepspeed/ops/adam/cpu_adam.py` | DeepSpeedCPUAdam with multi-threading support |
| `deepspeed/runtime/zero/parameter_offload.py` | Parameter offloading with async transfer support |
