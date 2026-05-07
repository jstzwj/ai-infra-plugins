# Chapter 33: Elastic Training (Elastification)

## Source Files
- `sources/Megatron-LM/megatron/elastification/` - Elastic training modules
- `sources/Megatron-LM/megatron/elastification/flextron_config.py` - Flexible training config
- `sources/Megatron-LM/megatron/elastification/memory_config.py` - Memory management
- `sources/Megatron-LM/megatron/elastification/router/` - Elastic routing

## Overview

Elastic Training (Elastification) enables dynamic resource allocation during training, allowing the model to adapt to available GPU resources without restarting. This is useful for cloud environments with variable GPU availability and for efficient resource utilization.

## Key Concepts

### Flextron Configuration
Flextron provides flexible training configuration that can adapt model parallelism and batch size during training.

```python
from megatron.elastification.flextron_config import FlextronConfig

config = FlextronConfig(
    min_tensor_parallel=1,
    max_tensor_parallel=8,
    min_pipeline_parallel=1,
    max_pipeline_parallel=4,
    min_micro_batch_size=1,
    max_micro_batch_size=8,
)
```

### Memory Management
Dynamic memory management monitors GPU memory usage and adjusts parallelism configuration to fit within available memory.

```python
from megatron.elastification.memory_config import MemoryConfig

mem_config = MemoryConfig(
    memory_margin_fraction=0.1,  # Keep 10% memory buffer
    enable_memory_monitoring=True,
)
```

### Router
The elastic router distributes work across available resources dynamically.

## Features

- **Dynamic Parallelism Adjustment**: Change TP/PP during training
- **Memory-Aware Scheduling**: Automatically adjust to fit GPU memory
- **Fault Tolerance**: Handle GPU failures gracefully
- **Resource Optimization**: Maximize GPU utilization

## Configuration

```bash
python pretrain_gpt.py \
    --enable-elastic-training \
    --min-tensor-parallel-size 1 \
    --max-tensor-parallel-size 8 \
    --min-pipeline-parallel-size 1 \
    --max-pipeline-parallel-size 4 \
    [other training args...]
```
