# DeepSpeed Reference - Chapter 1: Overview and Architecture

This chapter provides a comprehensive overview of DeepSpeed's design philosophy, system architecture, package structure, main entry points, key classes, hardware support, and the research publications that underpin its innovations.

---

## 1.1 Design Philosophy

### 1.1.1 Origins and Motivation

DeepSpeed was developed by Microsoft Research and released as an open-source project in May 2020. It was designed to address the fundamental challenge of training ever-larger deep learning models that exceed the memory and compute capacity of single devices. The project emerged from the recognition that model sizes were growing at a rate far exceeding hardware improvements, and that existing distributed training approaches (data parallelism alone) were insufficient.

The core insight behind DeepSpeed is that the traditional separation between model parallelism and data parallelism is too rigid. DeepSpeed introduced **ZeRO (Zero Redundancy Optimizer)**, which progressively eliminates memory redundancy across data-parallel processes by partitioning optimizer states, gradients, and parameters across devices -- achieving the memory efficiency of model parallelism while maintaining the simplicity and communication efficiency of data parallelism.

### 1.1.2 Core Design Principles

1. **Transparent Scaling**: Users should be able to scale their training from a single GPU to thousands of GPUs with minimal code changes. DeepSpeed achieves this through a lightweight integration API (`deepspeed.initialize()`) that wraps existing PyTorch models.

2. **Memory Efficiency First**: DeepSpeed prioritizes reducing memory footprint as the primary enabler for large model training. ZeRO stages 1, 2, and 3 progressively partition optimizer states, gradients, and parameters to eliminate redundant memory usage.

3. **Compute Efficiency**: Beyond memory savings, DeepSpeed optimizes compute through techniques like overlapped communication, gradient accumulation, mixed-precision training, and custom CUDA kernels.

4. **Flexibility and Composability**: Different training scenarios require different optimization strategies. DeepSpeed provides a modular, configuration-driven approach where users compose features through `ds_config.json` without modifying training code.

5. **Full-Stack Optimization**: DeepSpeed does not rely on a single technique. It combines system-level optimizations (memory management, communication scheduling, I/O optimization) with algorithmic innovations (sparse attention, 1-bit compression) to push training efficiency.

6. **Ease of Use**: The DeepSpeed launcher, automatic micro-batching, and configuration-driven approach minimize the engineering effort required to use advanced distributed training techniques.

### 1.1.3 Version History

| Version | Date | Key Features |
|---------|------|-------------|
| 0.1 | May 2020 | Initial release, ZeRO Stage 1 & 2 |
| 0.2 | Jul 2020 | ZeRO Stage 3 (parameter partitioning), ZeRO-Offload |
| 0.3 | Sep 2020 | DeepSpeed Sparse Attention, 1-bit Adam |
| 0.4 | Dec 2020 | Pipeline parallelism (DeepSpeed-MoE), 0/1 Adam |
| 0.5 | Apr 2021 | DeepSpeed-MoE, Mixture of Experts support |
| 0.6 | Aug 2021 | DeepSpeed Training with NVMe offload, ZeRO-Infinity |
| 0.7 | Feb 2022 | DeepSpeed Inference, kernel injections, model parallelism |
| 0.8 | Aug 2022 | DeepSpeed Compression, ZeroQuant |
| 0.9 | Nov 2022 | ZeRO++ (quantized communication), DeepSpeed-Chat |
| 0.10 | Apr 2023 | DeepSpeed ZeRO-Infinity improvements, Autotuning |
| 0.11 | Aug 2023 | DeepSpeed-FastGen, Hybrid Engine |
| 0.12 | Dec 2023 | FP8 support, improved inference engine (v2) |
| 0.13 | Mar 2024 | AMD ROCm support, HPU (Habana) support |
| 0.14 | Jun 2024 | DeepCompile, activation checkpointing improvements |
| 0.15 | Oct 2024 | ZenFlow, DataStates, enhanced elastic training |
| 0.16 | Feb 2025 | Expanded accelerator support (NPU, XPU, MLU, SDAA) |
| 0.17 | Jul 2025 | DeepSpeed v2 inference engine stable, tensor parallelism overhaul |

---

## 1.2 System Architecture

### 1.2.1 High-Level Architecture

DeepSpeed is organized as a layered system that sits between the user's PyTorch training code and the underlying hardware/communication libraries:

```
+-------------------------------------------------------------------+
|                        User Training Code                          |
|  (model definition, data loading, training loop, evaluation)       |
+-------------------------------------------------------------------+
                                |
                    deepspeed.initialize() / init_inference()
                                |
+-------------------------------------------------------------------+
|                     DeepSpeed Engine Layer                         |
|  +------------------+  +------------------+  +------------------+ |
|  | DeepSpeedEngine  |  | PipelineEngine   |  | HybridEngine     | |
|  | (ZeRO training)  |  | (Pipeline parallel| | (Train+Inference)| |
|  +------------------+  +------------------+  +------------------+ |
|  +------------------+  +------------------+                        |
|  | InferenceEngine  |  | InferenceEngineV2|                        |
|  | (Kernel inject)  |  | (Next-gen inf)   |                        |
|  +------------------+  +------------------+                        |
+-------------------------------------------------------------------+
                                |
+-------------------------------------------------------------------+
|                     DeepSpeed Runtime Layer                        |
|  +-----------+ +----------+ +-----------+ +----------+ +--------+ |
|  | ZeRO      | | Pipe     | | Checkpoint| | Launcher | | Monitor| |
|  | Runtime   | | Runtime  | | Manager   | | Engine   | | System | |
|  +-----------+ +----------+ +-----------+ +----------+ +--------+ |
|  +-----------+ +----------+ +-----------+ +----------+ +--------+ |
|  | Elastic   | | Auto-    | | Profiling | | Compile  | | Activ. | |
|  | Training  | | Tuning   | | Tools     | | Runtime  | | Checkpt| |
|  +-----------+ +----------+ +-----------+ +----------+ +--------+ |
+-------------------------------------------------------------------+
                                |
+-------------------------------------------------------------------+
|                     DeepSpeed Operations Layer                     |
|  +------------+ +------------+ +-----------+ +------------------+ |
|  | CUDA Ops   | | Comm Ops   | | SPARSE    | | Transformer Kernels|
|  | (Adam,     | | (Allgather,| | Attention | | (GPT, BERT,      |
|  |  Lion,     | |  Reduce,   | | Kernels   | |  LLaMA, etc.)    |
|  |  Quantize) | |  AlltoAll) | |           | |                  |
|  +------------+ +------------+ +-----------+ +------------------+ |
+-------------------------------------------------------------------+
                                |
+-------------------------------------------------------------------+
|                     DeepSpeed Backend/Accelerator Layer            |
|  +--------+ +------+ +-----+ +-----+ +-----+ +-----+ +------+   |
|  | CUDA   | | ROCm | | HPU | | NPU | | XPU | | MLU | | SDAA |   |
|  | (NVIDIA| |(AMD) | |(Hab)| |(Asc)| |(Int)| |(Cam)| |(MT)  |   |
|  +--------+ +------+ +-----+ +-----+ +-----+ +-----+ +------+   |
+-------------------------------------------------------------------+
                                |
+-------------------------------------------------------------------+
|                     Hardware Layer                                 |
|  NVIDIA GPUs  |  AMD GPUs  |  Intel GPUs/HPUs  |  Ascend NPUs    |
|  Cambricon MLUs  |  Moore Threads GPUs  |  CPUs (fallback)      |
+-------------------------------------------------------------------+
```

### 1.2.2 Architecture Diagram - Training Data Flow

```
                          Training Loop
                               |
                    +----------v----------+
                    |   User calls        |
                    |   engine.step()     |
                    +----------+----------+
                               |
              +----------------+----------------+
              |                                 |
    +---------v---------+            +----------v----------+
    | Forward Pass      |            | Loss Computation    |
    | engine.forward()  |            |                     |
    +---------+---------+            +----------+----------+
              |                                 |
    +---------v---------+            +----------v----------+
    | Micro-batch Loop  |            | Backward Pass       |
    | (gradient accum.) |            | engine.backward()   |
    +---------+---------+            +----------+----------+
              |                                 |
    +---------v---------+            +----------v----------+
    | Mixed Precision   |            | Gradient            |
    | (FP16/BF16 cast)  |            | Reduction/AllReduce |
    +---------+---------+            +----------+----------+
              |                                 |
    +---------v---------+            +----------v----------+
    | ZeRO Stage 1:     |            | ZeRO Stage 2:       |
    | Optimizer State   |            | Gradient            |
    | Partitioning      |            | Partitioning        |
    +---------+---------+            +----------+----------+
              |                                 |
    +---------v---------+            +----------v----------+
    | ZeRO Stage 3:     |            | Offload (optional)  |
    | Parameter         |            | CPU/NVME offload    |
    | Partitioning      |            | of params/grads     |
    +---------+---------+            +----------+----------+
              |                                 |
              +----------------+----------------+
                               |
                    +----------v----------+
                    | Optimizer Step      |
                    | (AdamW/Lamb/Lion)   |
                    +----------+----------+
                               |
                    +----------v----------+
                    | Learning Rate      |
                    | Scheduler Step     |
                    +----------+----------+
                               |
                    +----------v----------+
                    | Next Training      |
                    | Iteration          |
                    +---------------------+
```

### 1.2.3 Architecture Diagram - ZeRO Memory Partitioning

```
                    Without ZeRO (Standard DP)
    GPU 0: [Parameters | Gradients | Optimizer States]
    GPU 1: [Parameters | Gradients | Optimizer States]
    GPU 2: [Parameters | Gradients | Optimizer States]
    GPU 3: [Parameters | Gradients | Optimizer States]
    ==> 4x total memory (full replication)

                    ZeRO Stage 1
    GPU 0: [Parameters | Gradients | Optim_States_0]
    GPU 1: [Parameters | Gradients | Optim_States_1]
    GPU 2: [Parameters | Gradients | Optim_States_2]
    GPU 3: [Parameters | Gradients | Optim_States_3]
    ==> ~2x memory savings (optimizer states partitioned)

                    ZeRO Stage 2
    GPU 0: [Parameters | Gradients_0 | Optim_States_0]
    GPU 1: [Parameters | Gradients_1 | Optim_States_1]
    GPU 2: [Parameters | Gradients_2 | Optim_States_2]
    GPU 3: [Parameters | Gradients_3 | Optim_States_3]
    ==> ~4x memory savings (+ gradient partitioning)

                    ZeRO Stage 3
    GPU 0: [Params_0 | Gradients_0 | Optim_States_0]
    GPU 1: [Params_1 | Gradients_1 | Optim_States_1]
    GPU 2: [Params_2 | Gradients_2 | Optim_States_2]
    GPU 3: [Params_3 | Gradients_3 | Optim_States_3]
    ==> ~N_gpu x memory savings (+ parameter partitioning)
```

---

## 1.3 Package Structure

The DeepSpeed package is organized into the following subdirectories, each responsible for a specific subsystem:

### 1.3.1 Top-Level Package: `deepspeed/`

```
deepspeed/
|-- __init__.py              # Public API exports (initialize, init_inference, etc.)
|-- constants.py             # Shared constants and enums
|-- env_report.py            # ds_report environment diagnostics
|-- git_version_info.py      # Auto-generated version from git tags
|-- version.py               # Version string management
|-- eager_op_utils.py        # Utilities for eager CUDA op compilation
|
|-- runtime/                 # Core training runtime
|-- inference/               # Inference engine and optimizations
|-- ops/                     # Custom CUDA/C++ operations
|-- comm/                    # Communication primitives
|-- accelerator/             # Hardware accelerator abstraction
|-- compile/                 # DeepCompile module
|-- compression/             # Model compression utilities
|-- moe/                     # Mixture of Experts layers
|-- pipe/                    # Pipeline parallelism
|-- sequence/                # Sequence parallelism
|-- checkpoint/              # Checkpoint management
|-- launcher/                # Multi-process launcher
|-- monitor/                 # Training monitoring integrations
|-- autotuning/              # Automatic hyperparameter tuning
|-- elasticity/              # Elastic training support
|-- profiling/               # Profiling and diagnostics
|-- module_inject/           # Model replacement/injection for inference
|-- model_implementations/   # Pre-built model implementations
|-- linear/                  # Optimized linear layer implementations
|-- datastates/              # DataStates checkpointing
|-- nebula/                  # Nebula async checkpointing
|-- nvme/                    # NVMe offload management
|-- io/                      # I/O utilities
|-- utils/                   # Shared utility functions
```

### 1.3.2 `deepspeed/runtime/`

The runtime module is the heart of DeepSpeed's training engine. It contains the engine classes, ZeRO implementations, optimizer wrappers, and all configuration-driven training logic.

```
deepspeed/runtime/
|-- __init__.py
|-- engine.py                # DeepSpeedEngine (the main training engine)
|-- pipe/engine.py           # PipelineEngine (pipeline parallel engine)
|-- hybrid_engine.py         # DeepSpeedHybridEngine (train+inference)
|-- zero/                    # ZeRO optimization implementations
|   |-- __init__.py
|   |-- stage_1_and_2.py     # ZeRO Stage 1 & 2 (optimizer/gradient partitioning)
|   |-- stage3.py            # ZeRO Stage 3 (parameter partitioning)
|   |-- partition_parameters.py  # Parameter partitioning logic
|   |-- config.py            # ZeRO configuration parsing
|   |-- contiguous_memory_allocator.py
|   |-- offload_config.py    # CPU/NVMe offload configuration
|   |-- parameter_offload.py # Parameter offloading to CPU/NVMe
|   |-- offload_manager.py   # Unified offload management
|   |-- utils.py             # ZeRO utility functions
|   |-- mics/mics_utils.py   # MiCS (Minimal-communication Scaling) utilities
|   |-- tiling.py            # Tiling for large parameter management
|-- fp16/                    # FP16 mixed precision training
|   |-- unfused_optimizer.py # Unfused FP16 optimizer
|   |-- fused_optimizer.py   # Fused FP16 optimizer
|-- bf16_optimizer.py        # BF16 optimizer implementation
|-- config.py                # DeepSpeedConfig class (parses ds_config.json)
|-- config_utils.py          # Configuration validation and utilities
|-- dllogger.py              # Deep Learning Logger integration
|-- activation_checkpointing/
|   |-- checkpointing.py     # Activation checkpointing (gradient checkpointing)
|   |-- config.py            # Activation checkpointing configuration
|-- sparse_attention/
|   |-- sparse_attention.py  # Sparse attention implementation
|   |-- bert_sparse_attention.py
|   |-- longformer_attention.py
|   |-- bigbird_attention.py
|   |-- bsda.py              # Block-sparse attention
|-- swap_tensor/
|   |-- aio_handler.py       # Async I/O for tensor swapping
|   |-- partitioned_param_swapper.py  # Parameter swap to NVMe
|   |-- optimizer_swapper.py # Optimizer state swap to NVMe
|-- comm/coalesced_collectives.py  # Coalesced communication primitives
|-- data/                    # Data pipeline utilities
|-- lr_schedules.py          # Learning rate schedules
|-- progressive_layer_drop.py # Progressive layer dropping
|-- curriculum_learning.py   # Curriculum learning
|-- quantize.py              # Quantization utilities
|-- weight_quantizer.py      # Weight quantization (Int8/Int4)
|-- communicator.py          # Distributed communicator wrapper
|-- state_dict_factory.py    # State dict creation utilities
|-- utils.py                 # Runtime utility functions
|-- torch_autocast.py        # Integration with torch.autocast
```

### 1.3.3 `deepspeed/inference/`

The inference module provides optimized inference capabilities with kernel injection, tensor parallelism, and model-parallel inference support.

```
deepspeed/inference/
|-- __init__.py
|-- engine.py                # InferenceEngine (v1)
|-- v2/                      # InferenceEngineV2 (next generation)
|   |-- __init__.py
|   |-- engine.py            # InferenceEngineV2 implementation
|   |-- model_params.py      # Model parameter management
|   |-- policy_manager.py    # Policy-based kernel selection
|-- config.py                # Inference configuration
|-- module_inject/           # Module replacement/injection
|   |-- replace_policy.py    # Replacement policies for different models
|   |-- inject.py            # Module injection logic
|   |-- policy.py            # Base policy class
|-- fp_quantizer.py          # FP quantization for inference
```

### 1.3.4 `deepspeed/ops/`

The ops module contains custom CUDA/C++ operations that provide high-performance implementations of key training operations. Operations are compiled just-in-time (JIT) or pre-built using the op_builder system.

```
deepspeed/ops/
|-- __init__.py
|-- op_builder/              # Build system for custom ops
|   |-- __init__.py
|   |-- builder.py           # OpBuilder base class
|   |-- all_ops.py           # Registry of all available ops
|   |-- cpu_adagrad.py       # CPU Adagrad builder
|   |-- cpu_adam.py          # CPU Adam builder
|   |-- fused_adam.py        # Fused Adam builder
|   |-- fused_lamb.py        # Fused Lamb builder
|   |-- fused_lion.py        # Fused Lion builder
|   |-- utils.py             # Build utilities
|-- adagrad/                 # CPU Adagrad implementation
|-- adam/                    # CPU/CUDA Adam implementations
|-- lion/                    # Lion optimizer kernels
|-- transformer/             # Transformer kernel implementations
|   |-- infer/
|   |   |-- ops/             # Inference transformer ops
|-- sparse_attention/        # Sparse attention CUDA kernels
|-- quantizer/               # Quantization kernels
|-- adam_lamb/               # Adam/Lamb fused kernels
|-- adlr/                    # ADLR custom operations
|-- comm/                    # Communication-optimized operations
|-- euler/                   # Euler-based operations
|-- op_context/              # Context management for ops
|-- transformer_inference/   # Transformer inference kernels
|-- megartron/               # Megatron-style operations
```

### 1.3.5 `deepspeed/comm/`

The communication module provides optimized collective communication primitives that can overlap with computation, support gradient compression, and work across different hardware backends.

```
deepspeed/comm/
|-- __init__.py
|-- comm.py                  # Communication primitives (allgather, reduce_scatter, etc.)
|-- reduce_scatter.py        # Optimized reduce-scatter
|-- alltoall.py              # All-to-all communication
|-- torch.py                 # torch.distributed backend wrapper
|-- utils.py                 # Communication utilities
|-- backend.py               # Backend abstraction layer
|-- ccl/                     # Intel oneCCL backend
|-- mccl/                    # Moore Threads communication backend
|-- msccl/                   # MSDI communication backend
|-- nccl_backend.py          # NCCL backend wrapper
|-- torch_backend.py         # PyTorch distributed backend wrapper
```

### 1.3.6 `deepspeed/accelerator/`

The accelerator module provides a hardware abstraction layer that allows DeepSpeed to run on different accelerator backends (CUDA, ROCm, HPU, NPU, etc.) through a unified interface.

```
deepspeed/accelerator/
|-- __init__.py
|-- accelerator.py           # Accelerator abstract base class
|-- cuda_accelerator.py      # NVIDIA CUDA accelerator
|-- rocm_accelerator.py      # AMD ROCm accelerator
|-- hpu_accelerator.py       # Intel Habana HPU accelerator
|-- npu_accelerator.py       # Huawei Ascend NPU accelerator
|-- xpu_accelerator.py       # Intel XPU accelerator
|-- mlu_accelerator.py       # Cambricon MLU accelerator
|-- sdaa_accelerator.py      # Moore Threads SDAA accelerator
|-- cpu_accelerator.py       # CPU fallback accelerator
|-- real_accelerator.py      # Auto-detect and create accelerator
```

### 1.3.7 `deepspeed/compile/`

The compile module (DeepCompile) provides graph-capture-based optimization, activation offloading, and compilation passes for training acceleration.

```
deepspeed/compile/
|-- __init__.py
|-- ds_compile.py            # Main compile API
|-- engine.py                # Compile engine
|-- config.py                # Compile configuration
|-- utils.py                 # Compile utilities
|-- pass_manager.py          # Optimization pass manager
|-- passes/                  # Individual optimization passes
|   |-- __init__.py
|   |-- free_activation.py   # Free activation pass
|   |-- offload_activation.py # Activation offload pass
|   |-- offload_opt_states.py # Optimizer state offload pass
|   |-- double_buffer.py     # Double buffering pass
|   |-- symmetric_memory.py  # Symmetric memory optimization pass
```

### 1.3.8 `deepspeed/compression/`

The compression module provides model compression techniques including quantization, pruning, and knowledge distillation.

```
deepspeed/compression/
|-- __init__.py
|-- basic_layer.py           # Basic compression layer
|-- cnv.py                   # Compression with normalization and quantization
|-- compression.py           # Main compression API
|-- constant.py              # Compression constants
|-- lu_decomposition.py      # LU decomposition-based compression
|-- quantization.py          # Quantization utilities
|-- stochastic_quantization.py # Stochastic quantization
|-- tas.py                   # Template-based Architecture Search
|-- tsp.py                   # Template-based Structure Pruning
```

### 1.3.9 `deepspeed/moe/`

The MoE module provides Mixture of Experts layer implementations for training sparse MoE models.

```
deepspeed/moe/
|-- __init__.py
|-- layer.py                 # MoE layer implementation
|-- experts.py               # Expert module management
|-- gating.py                # Gating/routing functions
|-- shard.py                 # Expert sharding utilities
|-- utils.py                 # MoE utilities
|-- topkgate.py              # Top-K gating
|-- linear.py                # MoE linear layers
|-- sharded_moe.py           # Sharded MoE implementation
```

### 1.3.10 `deepspeed/pipe/`

The pipe module implements pipeline parallelism, splitting models across stages and scheduling micro-batch execution.

```
deepspeed/pipe/
|-- __init__.py
|-- p2p.py                   # Point-to-point communication for pipeline
|-- pipeline.py              # Pipeline scheduling and execution
|-- schedule.py              # Pipeline schedule types (GPipe, 1F1B, Interleaved)
|-- module.py                # PipelineModule base class
|-- topology.py              # Pipeline topology management
|-- sync.py                  # Synchronization utilities
```

### 1.3.11 `deepspeed/sequence/`

The sequence module implements sequence parallelism, which partitions sequence-length dimension computations across devices.

```
deepspeed/sequence/
|-- __init__.py
|-- sequence_parallel.py     # Sequence parallelism implementation
|-- utils.py                 # Sequence parallelism utilities
```

### 1.3.12 `deepspeed/checkpoint/`

The checkpoint module handles saving and loading training checkpoints with support for ZeRO partitioned states, parallel I/O, and universal checkpoint format.

```
deepspeed/checkpoint/
|-- __init__.py
|-- utils.py                 # Checkpoint utilities
|-- deepspeed_checkpoint.py  # DeepSpeed-specific checkpoint handling
|-- universal_checkpoint.py  # Universal checkpoint format
|-- torch_checkpoint.py      # Standard PyTorch checkpoint compatibility
|-- fp16_utils.py            # FP16 checkpoint utilities
```

### 1.3.13 `deepspeed/launcher/`

The launcher module provides multi-process launching capabilities for distributed training.

```
deepspeed/launcher/
|-- __init__.py
|-- runner.py                # Main launch runner (deepspeed command)
|-- multinode_runner.py      # Multi-node launching (ssh, pdsh, etc.)
|-- misc.py                  # Launch utilities
```

### 1.3.14 `deepspeed/monitor/`

The monitor module provides integrations with popular monitoring and logging systems.

```
deepspeed/monitor/
|-- __init__.py
|-- monitor.py               # Monitor base class
|-- tensorboard.py           # TensorBoard integration
|-- wandb.py                 # Weights & Biases integration
|-- comet.py                 # Comet ML integration
|-- csv_monitor.py           # CSV file logging
```

### 1.3.15 `deepspeed/autotuning/`

The autotuning module automatically searches for optimal training configurations (batch size, ZeRO stage, offloading, etc.).

```
deepspeed/autotuning/
|-- __init__.py
|-- autotuner.py             # Main autotuning driver
|-- config.py                # Autotuning configuration
|-- runtime.py               # Autotuning trial runtime
|-- utils.py                 # Autotuning utilities
```

### 1.3.16 `deepspeed/elasticity/`

The elasticity module supports dynamic resource allocation, allowing training to adapt to changing GPU counts.

```
deepspeed/elasticity/
|-- __init__.py
|-- elasticity.py            # Elastic training manager
|-- config.py                # Elasticity configuration
|-- utils.py                 # Elastic training utilities
```

### 1.3.17 `deepspeed/profiling/`

The profiling module provides profiling and diagnostic capabilities including FLOPS estimation and memory tracking.

```
deepspeed/profiling/
|-- __init__.py
|-- flops_profiler.py        # FLOPS profiler
|-- memory_profiler.py       # Memory usage profiler
|-- timer.py                 # High-precision timer
```

### 1.3.18 `deepspeed/module_inject/`

The module inject module enables automatic replacement of standard PyTorch modules with DeepSpeed-optimized equivalents for inference.

```
deepspeed/module_inject/
|-- __init__.py
|-- inject.py                # Module injection logic
|-- replace_policy.py        # Replacement policies
|-- utils.py                 # Injection utilities
|-- auto_tp.py               # Automatic tensor parallelism
```

### 1.3.19 `deepspeed/model_implementations/`

Pre-built model implementations optimized for DeepSpeed inference.

```
deepspeed/model_implementations/
|-- __init__.py
|-- dlatch_directional_bias.py
|-- dataloader.py
|-- transformer/             # Transformer model implementations
|   |-- transformer_layer.py # Transformer layer implementation
```

### 1.3.20 `deepspeed/linear/`

Optimized linear layer implementations that support ZeRO-3 parameter partitioning natively.

```
deepspeed/linear/
|-- __init__.py
|-- linear.py                # Optimized Linear layer
|-- fp_quantized_linear.py   # FP-quantized linear layer
```

### 1.3.21 `deepspeed/datastates/`

DataStates provides an asynchronous, distributed checkpointing system.

```
deepspeed/datastates/
|-- __init__.py
|-- datastates.py            # DataStates implementation
|-- config.py                # DataStates configuration
|-- utils.py                 # DataStates utilities
```

### 1.3.22 `deepspeed/nebula/`

Nebula provides an asynchronous checkpointing system that reduces checkpoint I/O overhead by performing writes in the background.

```
deepspeed/nebula/
|-- __init__.py
|-- nebula.py                # Nebula checkpoint manager
|-- config.py                # Nebula configuration
```

### 1.3.23 `deepspeed/nvme/`

NVMe offload management for swapping tensors and optimizer states to NVMe storage.

```
deepspeed/nvme/
|-- __init__.py
|-- nvme.py                  # NVMe device management
|-- aio.py                   # Async I/O operations for NVMe
```

### 1.3.24 `deepspeed/io/`

General I/O utilities for data loading and file operations.

```
deepspeed/io/
|-- __init__.py
|-- io.py                    # I/O utility functions
|-- data_loader.py           # Data loading utilities
```

### 1.3.25 `deepspeed/utils/`

Shared utility functions used across the DeepSpeed codebase.

```
deepspeed/utils/
|-- __init__.py
|-- logging.py               # Logging configuration and utilities
|-- timer.py                 # Synchronized wall-clock timer
|-- groups.py                # Process group management
|-- tensor_fragment.py       # Tensor fragmentation utilities
|-- init_on_device.py        # On-device initialization utilities
|-- nvtx.py                  # NVTX range instrumentation
|-- types.py                 # Type definitions
|-- debug.py                 # Debug utilities
```

---

## 1.4 Main Entry Points

### 1.4.1 `deepspeed.initialize()`

The primary entry point for training. It wraps a PyTorch model and optimizer with DeepSpeed's distributed training capabilities.

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    config_params=ds_config,  # or path to ds_config.json
    dist_init_backend='nccl',
    model_parameters=None,
    training_data=None,
    lr_scheduler=None,
    mpu=None,
    dist_init_required=None,
    config=None,
    collate_fn=None,
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `torch.nn.Module` | Yes | The PyTorch model to wrap |
| `optimizer` | `torch.optim.Optimizer` or `None` | No | Existing optimizer or None (DeepSpeed can create one) |
| `args` | `argparse.Namespace` or `None` | No | Command-line arguments (passed to launcher) |
| `config_params` | `dict` or `str` | No | DeepSpeed configuration (dict or path to JSON file) |
| `dist_init_backend` | `str` | No | Distributed backend ('nccl', 'mpi', 'gloo', 'ccl') |
| `model_parameters` | `iterable` or `None` | No | Model parameters for optimizer creation |
| `training_data` | `Dataset` or `DataLoader` | No | Training dataset (creates a DataLoader) |
| `lr_scheduler` | `object` or `None` | No | Learning rate scheduler |
| `mpu` | `object` or `None` | No | Model parallelism unit (for Megatron-style parallelism) |
| `dist_init_required` | `bool` or `None` | No | Whether to initialize distributed process group |
| `config` | `str` | No | Path to ds_config.json (alternative to config_params) |
| `collate_fn` | `callable` or `None` | No | Collate function for DataLoader |

**Returns:** A tuple of `(engine, optimizer, training_dataloader, lr_scheduler)`.

**Engine type selection logic:**

1. If pipeline parallelism is configured (`pipeline.enabled: true` in config), a `PipelineEngine` is created.
2. If DeepCompile is enabled (`deepcompile.enabled: true`), a compiled engine is created.
3. If hybrid engine mode is enabled, a `DeepSpeedHybridEngine` is created.
4. Otherwise, a `DeepSpeedEngine` is created.

**Example - Basic Usage:**

```python
import deepspeed
import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = torch.nn.Linear(hidden_size * 4, hidden_size)
        self.ln = torch.nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.ln(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2},
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config_params=ds_config,
)

# Training loop
for batch in dataloader:
    outputs = model_engine(batch)
    loss = outputs.mean()
    model_engine.backward(loss)
    model_engine.step()
```

**Example - With HuggingFace Trainer:**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",  # Path to DeepSpeed config
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 1.4.2 `deepspeed.init_inference()`

The entry point for inference mode. It wraps a PyTorch model with DeepSpeed's inference optimizations including kernel injection, tensor parallelism, and quantization.

```python
model = deepspeed.init_inference(
    model=model,
    mp_size=1,                           # Tensor parallel size
    dtype=torch.float16,                 # Target dtype
    checkpoint=None,                     # Checkpoint path or dict
    replace_with_kernel_inject=True,     # Enable kernel injection
    policy=None,                         # Custom replacement policy
    config=None,                         # Inference config
    injection_policy=None,               # Custom injection policy
    return_tuple=True,                   # Return tuple from forward
    meta_device=None,                    # Meta device for initialization
    pin_memory=False,                    # Pin memory for faster transfers
    tensor_parallel={'tp_size': 1},      # Tensor parallel config
    max_out_tokens=1024,                 # Max output tokens
    trust_remote_code=False,             # Trust remote code
    save_mp_checkpoint_path=None,        # Path to save mp checkpoint
    base_dir=None,                       # Base directory for model
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `torch.nn.Module` | Yes | The model to optimize for inference |
| `mp_size` | `int` | No | Model parallel size (deprecated, use tensor_parallel) |
| `dtype` | `torch.dtype` | No | Target data type (torch.float16, torch.bfloat16, torch.int8) |
| `checkpoint` | `str` or `dict` | No | Checkpoint path or state dict |
| `replace_with_kernel_inject` | `bool` | No | Replace modules with optimized kernels |
| `policy` | `object` | No | Custom module replacement policy |
| `config` | `dict` | No | Inference configuration dictionary |
| `injection_policy` | `dict` | No | Custom injection policy mapping |
| `return_tuple` | `bool` | No | Whether forward returns a tuple |
| `meta_device` | `str` | No | Meta device for initialization |
| `pin_memory` | `bool` | No | Pin memory for CPU-GPU transfers |
| `tensor_parallel` | `dict` | No | Tensor parallel configuration `{'tp_size': N}` |
| `max_out_tokens` | `int` | No | Maximum output token count |
| `trust_remote_code` | `bool` | No | Allow remote code execution |
| `save_mp_checkpoint_path` | `str` | No | Path to save model-parallel checkpoint |
| `base_dir` | `str` | No | Base directory for model files |

**Returns:** The wrapped inference model.

**Example - Basic Inference:**

```python
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)

# Initialize DeepSpeed inference
ds_model = deepspeed.init_inference(
    model=model,
    mp_size=1,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    tensor_parallel={'tp_size': 1},
)

inputs = tokenizer("DeepSpeed is", return_tensors="pt").to("cuda")
outputs = ds_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**Example - Multi-GPU Tensor Parallel Inference:**

```python
import deepspeed
import torch

# Launch with: deepspeed --num_gpus 4 inference_script.py

model = deepspeed.init_inference(
    model=model,
    tensor_parallel={'tp_size': 4},
    dtype=torch.bfloat16,
    replace_with_kernel_inject=True,
    max_out_tokens=2048,
)
```

---

## 1.5 Key Classes

### 1.5.1 `DeepSpeedEngine`

**Module:** `deepspeed.runtime.engine`

The central class in DeepSpeed. It wraps a PyTorch model and provides ZeRO optimization, mixed precision training, gradient accumulation, checkpoint management, and distributed training coordination.

**Class Hierarchy:**

```
DeepSpeedEngine
    |-- PipelineEngine (extends DeepSpeedEngine)
    |       |-- DeepSpeedHybridEngine (extends PipelineEngine)
    |-- InferenceEngine
    |-- InferenceEngineV2
```

**Key Responsibilities:**

- Configuration parsing and validation
- ZeRO stage initialization (1, 2, or 3)
- Mixed precision setup (FP16, BF16, AMP)
- Optimizer creation or wrapping
- Distributed process group management
- Gradient accumulation and micro-batching
- Checkpoint save/load
- Learning rate scheduling
- Activation checkpointing management
- Memory tracking and reporting

**Key Methods:**

```python
class DeepSpeedEngine:
    # Core training loop methods
    def forward(self, *inputs, **kwargs)       # Forward pass
    def backward(self, loss)                     # Backward pass
    def step(self)                               # Optimizer step + LR schedule step
    def train(self)                              # Set training mode
    def eval(self)                               # Set evaluation mode

    # Checkpoint methods
    def save_checkpoint(self, save_dir, tag=None, ...)    # Save checkpoint
    def load_checkpoint(self, load_dir, tag=None, ...)    # Load checkpoint

    # Properties
    @property train_batch_size -> int
    @property train_micro_batch_size_per_gpu -> int
    @property gradient_accumulation_steps -> int
    @property zero_optimization_stage -> int
    @property fp16_enabled -> bool
    @property bf16_enabled -> bool
```

### 1.5.2 `PipelineEngine`

**Module:** `deepspeed.runtime.pipe.engine`

Extends `DeepSpeedEngine` with pipeline parallelism support. The model is split into pipeline stages, and micro-batches are processed using schedules like GPipe (fill-drain) or 1F1B (one-forward-one-backward).

**Key Additions:**

```python
class PipelineEngine(DeepSpeedEngine):
    def __init__(self, ...)
    def forward(self, ...)         # Pipeline-aware forward
    def backward(self, ...)        # Pipeline-aware backward
    def step(self, ...)            # Pipeline-aware optimizer step
    def set_has_optimizer_model(self, ...)  # Configure optimizer for pipeline
    def is_first_stage(self) -> bool       # Check if first pipeline stage
    def is_last_stage(self) -> bool        # Check if last pipeline stage
```

**Pipeline Schedule Types:**

| Schedule | Description | Memory Usage | Throughput |
|----------|-------------|-------------|------------|
| GPipe | All forwards, then all backwards | High (all micro-batch activations stored) | Moderate |
| 1F1B | Interleave forward and backward | Low (steady-state activation count) | High |
| Interleaved 1F1B | Multiple stages per device | Moderate | Highest |

### 1.5.3 `DeepSpeedHybridEngine`

**Module:** `deepspeed.runtime.hybrid_engine`

Combines training and inference capabilities in a single engine. Used for RLHF (Reinforcement Learning from Human Feedback) workloads where the model alternates between training and inference (e.g., PPO training for LLMs).

**Key Features:**

- Seamless switching between training and inference modes
- Reuses ZeRO partitioned weights for inference
- Supports tensor parallelism during inference
- Minimizes weight re-sharding overhead during mode switches

```python
class DeepSpeedHybridEngine(PipelineEngine):
    def __init__(self, ...)
    def inference_forward(self, ...)  # Inference-optimized forward pass
    def inference(self, ...)          # Full inference pipeline
    def generate(self, ...)           # Text generation
```

### 1.5.4 `InferenceEngine`

**Module:** `deepspeed.inference.engine`

The inference engine (v1) for optimized model inference. It replaces standard PyTorch modules with DeepSpeed-optimized CUDA kernels, supports tensor parallelism, and provides quantization.

**Key Features:**

- Kernel injection: Replaces transformer layers with optimized kernels
- Tensor parallelism: Splits model across GPUs for large models
- Quantization support: INT8/INT4 weight quantization
- Continuous batching support
- KV-cache management

```python
class InferenceEngine:
    def __init__(self, model, ...)
    def forward(self, ...)       # Optimized forward pass
    def generate(self, ...)      # Text generation
```

### 1.5.5 `InferenceEngineV2`

**Module:** `deepspeed.inference.v2.engine`

The next-generation inference engine with improved architecture, better kernel selection, and more flexible configuration.

**Key Improvements over V1:**

- Policy-based kernel selection (instead of hardcoded replacement)
- Better support for heterogeneous hardware
- Improved memory management
- Support for newer model architectures
- FP8 quantization support

```python
class InferenceEngineV2:
    def __init__(self, model, ...)
    def forward(self, ...)       # Optimized forward pass
    def generate(self, ...)      # Text generation
```

---

## 1.6 Hardware Support

### 1.6.1 Supported Accelerators

DeepSpeed supports the following hardware accelerators through its accelerator abstraction layer:

| Accelerator | Hardware | Backend | Communication | Status |
|-------------|----------|---------|---------------|--------|
| **CUDA** | NVIDIA GPUs (A100, H100, B200, etc.) | CUDA | NCCL | Full support |
| **ROCm** | AMD GPUs (MI250, MI300, etc.) | ROCm | RCCL | Full support |
| **HPU** | Intel Habana Gaudi (Gaudi2, Gaudi3) | Habana SDK | HCCL | Full support |
| **NPU** | Huawei Ascend (910A, 910B) | CANN | HCCL | Full support |
| **XPU** | Intel GPU (Max, Flex) | Level Zero / SYCL | oneCCL | Full support |
| **MLU** | Cambricon (MLU370, MLU590) | CNToolkit | CNCL | Full support |
| **SDAA** | Moore Threads (MTT S4000) | MUSA | MCCL | Experimental |
| **CPU** | x86/ARM CPUs | None | Gloo/MPI | Limited (inference, offloading) |

### 1.6.2 Accelerator Detection

DeepSpeed auto-detects the available accelerator at runtime. The detection priority is:

1. CUDA (if `torch.cuda.is_available()`)
2. ROCm (if AMD GPU detected via CUDA compatibility layer)
3. HPU (if `habana_frameworks` is importable)
4. NPU (if `torch_npu` is importable)
5. XPU (if `torch.xpu.is_available()`)
6. MLU (if `torch_mlu` is importable)
7. SDAA (if `torch_musa` is importable)
8. CPU (fallback)

```python
# Manual accelerator override via environment variable
import os
os.environ["DS_ACCELERATOR"] = "cuda"  # Force specific accelerator
```

### 1.6.3 Communication Backend Support

| Backend | Hardware | When to Use |
|---------|----------|-------------|
| NCCL | NVIDIA GPUs | Default for CUDA, highest performance |
| RCCL | AMD GPUs | Default for ROCm |
| HCCL | Huawei Ascend NPUs | Default for NPU |
| HCCL | Intel Habana HPUs | Default for HPU |
| oneCCL | Intel XPUs | Default for XPU |
| CNCL | Cambricon MLUs | Default for MLU |
| MCCL | Moore Threads GPUs | Default for SDAA |
| Gloo | CPU / Any | Fallback for CPU, testing |
| MPI | CPU / Any | For HPC environments with MPI |

---

## 1.7 Publications

DeepSpeed is built on a foundation of peer-reviewed research publications. The following papers describe the core algorithms and techniques:

### 1.7.1 ZeRO Optimization

1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**
   - R. Rajbhandari, J. Rasley, O. Ruwase, Y. He
   - SC 2020 (International Conference for High Performance Computing, Networking, Storage and Analysis)
   - Introduces ZeRO Stages 1, 2, and 3

2. **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**
   - J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, Y. He
   - SC 2021
   - Extends ZeRO to offload to CPU and NVMe, enabling training models with trillions of parameters

3. **ZeRO++: Extremely Efficient Collective Communication for Giant Model Training**
   - H. Lu, K. Huang, T. A. Feng, C. Li, A. Ponomarev, A. Cheng, W. Xiao, Y. He
   - SC 2023
   - Introduces quantized communication (qwZ, qgZ, qiZ) for reducing communication volume

### 1.7.2 1-bit Compression

4. **1-bit Adam: Communication Efficient Adam for Large-Scale Deep Learning**
   - H. Tang, S. Lian, M. Yan, C. Zhang, T. Gu, C. Yu, W. Wu, Y. He
   - MLSys 2021
   - 1-bit compressed Adam optimizer

5. **1-bit LAMB: Communication Efficient LAMB for Large-Scale Deep Learning**
   - S. Lian, C. Li, X. Wang, Y. He
   - 2021
   - 1-bit compressed LAMB optimizer

6. **0/1 Adam: An Interleaved 0/1 Compression for Distributed Adam**
   - C. Li, S. Lian, H. Tang, M. Yan, C. Zhang, T. Gu, W. Wu, Y. He
   - 2022
   - Hybrid 0/1 compression for Adam

### 1.7.3 Pipeline Parallelism

7. **PipeDream: Fast and Efficient Pipeline Parallel DNN Training**
   - D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur, G. R. Ganger, P. B. Gibbons, M. Zaharia
   - SOSP 2019
   - Pipeline parallelism with 1F1B schedule

8. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**
   - D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. A. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, M. Zaharia
   - SC 2021
   - Combined 3D parallelism (data + tensor + pipeline)

### 1.7.4 Mixture of Experts

9. **DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale**
   - S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, Y. He
   - OSDI 2022
   - MoE training with expert parallelism

### 1.7.5 Sparse Attention

10. **DeepSpeed Sparse Attention: Linear-Time Attention for Very Long Sequences**
    - S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, Y. He
    - 2020
    - Block-sparse attention patterns for long sequences

### 1.7.6 Inference

11. **DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale**
    - R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, J. Rasley, Y. He
    - SC 2022
    - Inference engine with kernel injection and tensor parallelism

12. **DeepSpeed-FastGen: High-Throughput Text Generation for LLMs**
    - 2023
    - Continuous batching and dynamic batch scheduling for inference

### 1.7.7 Compression

13. **ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers**
    - Z. Yao, R. Y. Aminabadi, M. Zhang, X. Wu, C. Li, Y. He
    - NeurIPS 2022
    - Post-training quantization for inference

14. **ZeroQuant-V2: Exploring Post-training Quantization for LLMs from the Perspective of Optimal Balance among Multiple Constraints**
    - 2023
    - Improved quantization with multiple constraint optimization

### 1.7.8 Autotuning

15. **Autotuning: Automating the Search for Optimal DeepSpeed Configuration**
    - 2022
    - Automatic hyperparameter tuning for DeepSpeed

### 1.7.9 System-Level

16. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
    - M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro
    - SC 2019
    - Tensor parallelism (Megatron-LM, integrated into DeepSpeed)

17. **Making PyTorch Go Bolder: Enabling Large-Scale Training with DeepSpeed**
    - J. Rasley, S. Rajbhandari, O. Ruwase, Y. He
    - MLSys 2020
    - Original DeepSpeed system paper

### 1.7.10 DeepCompile

18. **DeepCompile: Compiler-Based Optimizations for Deep Learning Training**
    - 2024
    - Graph-based compilation for training optimization

### 1.7.11 Elastic Training

19. **Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large Language Models**
    - H. Zhang, J. Li, K. Kara, D. Alistarh, G. Gu, C. Li
    - 2023
    - Redundancy-based resilience for elastic training

---

## 1.8 Integration Points

### 1.8.1 PyTorch Integration

DeepSpeed is built on top of PyTorch and uses the following PyTorch subsystems:

- **`torch.distributed`**: Process group management, collective communication
- **`torch.cuda`**: CUDA memory management, device management
- **`torch.nn`**: Module system, parameter management
- **`torch.autograd`**: Automatic differentiation
- **`torch.amp`**: Automatic mixed precision
- **`torch.utils.data`**: DataLoader integration

### 1.8.2 HuggingFace Integration

DeepSpeed integrates tightly with HuggingFace Transformers through the `Trainer` class:

```python
# Launch with: deepspeed --num_gpus 4 train.py --deepspeed ds_config.json
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    deepspeed="ds_config.json",
    # ... other args
)
```

The HuggingFace integration handles:
- Automatic DeepSpeed engine creation via `deepspeed.initialize()`
- Configuration merging between HuggingFace args and ds_config.json
- Gradient accumulation coordination
- Checkpoint save/load with ZeRO partitioning

### 1.8.3 Megatron-LM Integration

DeepSpeed integrates with Megatron-LM for tensor and pipeline parallelism:

```python
# Megatron + DeepSpeed
from megatron import get_args
from deepspeed import initialize as ds_initialize

model_engine, optimizer, _, _ = ds_initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    mpu=model_parallel_group_utils,  # Megatron's model parallel utility
)
```

### 1.8.4 Microsoft Azure Integration

DeepSpeed provides first-class Azure support:

- **Azure ML**: Direct integration with Azure Machine Learning workspaces
- **Azure ND Series**: Optimized for NDm A100 v4 and ND H100 v5 VMs
- **Azure Blob Storage**: For checkpoint storage
- **Azure CycleCloud**: For cluster management

### 1.8.5 Weights & Biases Integration

DeepSpeed integrates with W&B for experiment tracking:

```json
{
    "wandb": {
        "enabled": true,
        "project": "my-project",
        "team": "my-team",
        "group": "my-group",
        "name": "my-run-name"
    }
}
```

---

## 1.9 Configuration Overview

DeepSpeed is configured through a JSON configuration file (`ds_config.json`) or a Python dictionary passed to `initialize()`. The configuration drives all aspects of training behavior without requiring code changes.

**Configuration Categories:**

| Category | Key | Description |
|----------|-----|-------------|
| Batch Size | `train_batch_size`, `train_micro_batch_size_per_gpu`, `gradient_accumulation_steps` | Control global and per-GPU batch sizes |
| Optimizer | `optimizer` | Optimizer type and parameters |
| Scheduler | `scheduler` | Learning rate schedule |
| FP16 | `fp16` | FP16 mixed precision training |
| BF16 | `bf16` | BF16 mixed precision training |
| ZeRO | `zero_optimization` | ZeRO stage and parameters |
| Offload | `offload_param`, `offload_optimizer` | CPU/NVMe offloading |
| Pipeline | `pipeline` | Pipeline parallelism configuration |
| Checkpoint | `checkpoint` | Checkpoint behavior |
| Logging | `steps_per_print`, `wall_clock_breakdown` | Logging and profiling |
| Communication | `communication_data_type`, `prescale_gradients` | Communication optimization |
| Tensor Parallel | `tensor_parallel` | Tensor parallelism configuration |
| Autotuning | `autotuning` | Automatic tuning |
| Compression | `compression` | Model compression |
| Elasticity | `elasticity` | Elastic training |
| Monitor | `tensorboard`, `wandb`, `comet`, `csv_monitor` | Monitoring integrations |

> See Chapter 3: Configuration Reference for the complete configuration specification.

---

## 1.10 Glossary

| Term | Definition |
|------|-----------|
| **ZeRO** | Zero Redundancy Optimizer - eliminates memory redundancy across data-parallel processes |
| **ZeRO Stage 1** | Optimizer state partitioning |
| **ZeRO Stage 2** | Optimizer state + gradient partitioning |
| **ZeRO Stage 3** | Optimizer state + gradient + parameter partitioning |
| **ZeRO-Infinity** | Extends ZeRO with CPU/NVMe offloading |
| **ZeRO++** | Communication-optimized ZeRO with quantization |
| **Offload** | Moving tensors/states from GPU to CPU or NVMe |
| **NVMe Offload** | Swapping tensors to NVMe SSD storage |
| **Pipeline Parallelism** | Splitting model across devices by layer depth |
| **Tensor Parallelism** | Splitting individual layers across devices |
| **3D Parallelism** | Combining data + pipeline + tensor parallelism |
| **Gradient Accumulation** | Accumulating gradients across multiple micro-batches before optimizer step |
| **Micro-batch** | A small batch processed in one forward+backward pass |
| **Kernel Injection** | Replacing PyTorch modules with optimized CUDA kernels |
| **Mixed Precision** | Training with reduced precision (FP16/BF16) while maintaining accuracy |
| **Communication Overlap** | Overlapping communication with computation |
| **Activation Checkpointing** | Recomputing activations during backward instead of storing them |
| **MoE** | Mixture of Experts - sparse model architecture |
| **1-bit Compression** | Compressing gradient communication to 1-bit per value |
| **DeepCompile** | Compiler-based training optimization |
| **Hybrid Engine** | Engine supporting both training and inference |
| **FP16** | 16-bit floating point (IEEE 754 half precision) |
| **BF16** | 16-bit brain floating point (wider exponent range) |
| **AMP** | Automatic Mixed Precision (PyTorch native) |
| **NCCL** | NVIDIA Collective Communications Library |
| **RCCL** | AMD ROCm Communications Library |
| **HCCL** | Huawei Collective Communications Library |
| **oneCCL** | Intel oneAPI Collective Communications Library |
