---
name: deepspeed
description: >
  Comprehensive reference documentation and skill for DeepSpeed - the distributed deep learning training
  and inference optimization library. Covers ZeRO optimization (Stages 0-3), ZeRO-Offload, ZeRO-Infinity,
  SuperOffload, ZenFlow, pipeline parallelism, tensor parallelism (AutoTP), sequence parallelism (Ulysses/ALST),
  MoE (Mixture of Experts), inference engines (v1/v2), quantization and compression, custom optimizers
  (Adam, LAMB, LION, Muon, 1-bit Adam), mixed precision training (FP16/BF16/AMP), activation checkpointing,
  model checkpointing, communication primitives, launcher, autotuning, elasticity, monitoring, profiling,
  DeepCompile, DeepNVMe, accelerator abstraction, module injection, and supported model implementations.
  Based on DeepSpeed source code analysis.
version: 0.16.x
---

# DeepSpeed - Distributed Deep Learning Training & Inference

## Overview

DeepSpeed is a deep learning optimization library that provides distributed training and inference capabilities for extreme-scale models. It powers some of the world's largest language models including MT-530B, BLOOM-176B, and GLM-130B.

**Core Capabilities:**
1. **Memory Optimization** - ZeRO (Zero Redundancy Optimizer) stages 0-3, ZeRO-Offload, ZeRO-Infinity, SuperOffload
2. **Parallelism** - 3D parallelism (data, pipeline, tensor), sequence parallelism (Ulysses/ALST)
3. **Inference** - High-performance inference with kernel injection, ragged batching, quantization
4. **Communication** - Compressed communication (1-bit Adam, ZeRO++), efficient collectives
5. **System** - DeepCompile, DeepNVMe, accelerator abstraction across 8+ hardware platforms

**Supported Hardware:** NVIDIA (CUDA), AMD (ROCm), Intel Gaudi (HPU), Intel XPU, Huawei Ascend (NPU), Cambricon (MLU), Tecorigin (SDAA), CPU

**Key Publications:** ZeRO (SC'20), ZeRO-Offload (ATC'21), ZeRO-Infinity (SC'21), DeepSpeed-MoE (ICML'22), DeepSpeed Inference (SC'22), ZenFlow (2025), SuperOffload (ASPLOS'26)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Application                            │
│         model_engine.forward() / backward() / step()            │
├─────────────────────────────────────────────────────────────────┤
│                    DeepSpeed Python API                          │
│  initialize() │ init_inference() │ PipelineEngine │ HybridEngine │
├────────────────────────┬────────────────────────────────────────┤
│    Training Runtime    │          Inference Runtime              │
│  ┌──────────────────┐  │  ┌──────────────────────────────────┐  │
│  │ DeepSpeedEngine  │  │  │ InferenceEngine (v1)             │  │
│  │  - ZeRO 0/1/2/3  │  │  │  - Kernel Injection             │  │
│  │  - Pipeline      │  │  │  - Tensor Parallel               │  │
│  │  - Tensor Para   │  │  │ InferenceEngineV2                │  │
│  │  - Sequence Para │  │  │  - Ragged Batching               │  │
│  │  - Mixed Prec    │  │  │  - Blocked KV Cache              │  │
│  │  - Checkpointing │  │  │  - MoE Support                   │  │
│  └──────────────────┘  │  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Communication Layer                         │
│           NCCL │ CCL │ MPI │ HCCL │ Custom Backends             │
├─────────────────────────────────────────────────────────────────┤
│                     Accelerator Abstraction                      │
│    CUDA │ CPU │ HPU │ NPU │ XPU │ MLU │ SDAA │ MPS             │
├─────────────────────────────────────────────────────────────────┤
│                   Custom Ops & CUDA Kernels                      │
│  FusedAdam │ FusedLamb │ Transformer │ SparseAttn │ Quantizer   │
│  AsyncIO │ GDS │ CPU Adam │ Evoformer │ Spatial                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Reference

### Basic Training
```python
import deepspeed

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=args,
    config_params=ds_config,  # or path to ds_config.json
)

# Training loop
for step, batch in enumerate(dataloader):
    inputs, labels = batch
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)
    model_engine.backward(loss)
    model_engine.step()
```

### Basic Inference
```python
import deepspeed

# Initialize inference engine
model = deepspeed.init_inference(
    model=model,
    mp_size=2,              # tensor parallel size
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)

# Run inference
outputs = model(inputs)
```

### Launch with DeepSpeed
```bash
# Single-node
deepspeed --num_gpus=4 train.py --deepspeed ds_config.json

# Multi-node
deepspeed --num_nodes=2 --hostfile=myhostfile train.py --deepspeed ds_config.json
```

### Minimal Configuration (ZeRO-2)
```json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "bf16": {
        "enabled": true
    }
}
```

## Reference Chapters

| # | Chapter | Description |
|---|---------|-------------|
| 01 | [Overview & Architecture](references/01-overview-and-architecture.md) | DeepSpeed architecture, design philosophy, component overview |
| 02 | [Installation & Setup](references/02-installation-and-setup.md) | Installation methods, requirements, environment setup |
| 03 | [Configuration Reference](references/03-configuration-reference.md) | Complete ds_config.json schema with all fields |
| 04 | [Getting Started](references/04-getting-started.md) | Quick-start guide, basic training and inference |
| 05 | [DeepSpeed Engine](references/05-deepspeed-engine.md) | DeepSpeedEngine, PipelineEngine, HybridEngine classes |
| 06 | [ZeRO Optimization](references/06-zero-optimization.md) | ZeRO stages 0-3, parameter/gradient partitioning |
| 07 | [ZeRO Offload & Infinity](references/07-zero-offload-infinity.md) | ZeRO-Offload, ZeRO-Infinity, NVMe offloading |
| 08 | [SuperOffload & ZenFlow](references/08-superoffload-zenflow.md) | SuperOffload, ZenFlow stall-free offloading |
| 09 | [Mixed Precision Training](references/09-mixed-precision-training.md) | FP16, BF16, AMP, loss scaling |
| 10 | [Pipeline Parallelism](references/10-pipeline-parallelism.md) | Pipeline stages, topology, scheduling |
| 11 | [Tensor Parallelism](references/11-tensor-parallelism.md) | AutoTP, manual TP, partition configs |
| 12 | [Sequence Parallelism](references/12-sequence-parallelism.md) | Ulysses, ALST, long sequence training |
| 13 | [MoE (Mixture of Experts)](references/13-moe-mixture-of-experts.md) | Expert parallelism, sharded MoE, routing |
| 14 | [Inference Engine V1](references/14-inference-engine-v1.md) | Kernel injection, model replacement, quantization |
| 15 | [Inference Engine V2](references/15-inference-engine-v2.md) | Ragged batching, blocked KV cache, model implementations |
| 16 | [Quantization & Compression](references/16-quantization-and-compression.md) | ZeroQuant, MoQ, weight/activation quantization, pruning |
| 17 | [Optimizers](references/17-optimizers.md) | FusedAdam, FusedLamb, FusedLion, 1-bit Adam, Muon |
| 18 | [Schedulers](references/18-schedulers.md) | WarmupLR, OneCycle, LRRangeTest, custom schedulers |
| 19 | [Activation Checkpointing](references/19-activation-checkpointing.md) | Memory optimization through activation recomputation |
| 20 | [Model Checkpointing](references/20-model-checkpointing.md) | Save/load, universal checkpoint, ZeRO checkpoint |
| 21 | [Communication Primitives](references/21-communication-primitives.md) | Comm module, NCCL, CCL, coalesced collectives |
| 22 | [Launcher](references/22-launcher.md) | Multi-node launch, elastic training, resource management |
| 23 | [Autotuning](references/23-autotuning.md) | Automatic hyperparameter optimization |
| 24 | [Elasticity](references/24-elasticity.md) | Dynamic resource scaling, fault tolerance |
| 25 | [Monitoring & Profiling](references/25-monitoring-and-profiling.md) | TensorBoard, WandB, FLOPS profiler |
| 26 | [Accelerator Abstraction](references/26-accelerator-abstraction.md) | Multi-hardware support, custom accelerators |
| 27 | [Custom Ops & Kernels](references/27-custom-ops-and-kernels.md) | CUDA kernels, op builder, JIT compilation |
| 28 | [Module Inject & AutoTP](references/28-module-inject-and-auto-tp.md) | Model replacement, automatic tensor parallelism |
| 29 | [Model Implementations](references/29-model-implementations.md) | Supported models: LLaMA, Mistral, Falcon, Qwen, etc. |
| 30 | [DeepCompile](references/30-deepcompile.md) | Compiler optimizations, FX passes, Inductor integration |
| 31 | [DeepSpeed4Science](references/31-deepspeed4science.md) | Scientific computing, Evoformer attention |
| 32 | [Data Pipeline](references/32-data-pipeline.md) | Data efficiency, curriculum learning, data routing |
| 33 | [IO & DeepNVMe](references/33-io-and-nvme.md) | Async IO, GPU Direct Storage, NVMe offloading |
| 34 | [Communication Compression](references/34-communication-compression.md) | 1-bit Adam, 0/1 Adam, ZeRO++ quantization |
| 35 | [Nebula & DataStates](references/35-nebula-and-datastates.md) | Asynchronous checkpointing, data state management |
| 36 | [Muon Optimizer](references/36-muon-optimizer.md) | MomentUm Orthogonalized by Newton-Schulz |
| 37 | [Constants & Utilities](references/37-constants-and-utils.md) | All constants, utility functions, helpers |
| 38 | [Debugging & Troubleshooting](references/38-debugging-and-troubleshooting.md) | Common issues, debugging tools, environment report |
| 39 | [Integrations](references/39-integrations.md) | HuggingFace, Accelerate, Lightning, MosaicML |
| 40 | [API Reference](references/40-api-reference.md) | Complete public API reference |

## Key Configuration Sections

### ZeRO Stages
```json
{"zero_optimization": {"stage": 0}}  // Disabled (standard DDP)
{"zero_optimization": {"stage": 1}}  // Optimizer state partitioning
{"zero_optimization": {"stage": 2}}  // + Gradient partitioning
{"zero_optimization": {"stage": 3}}  // + Parameter partitioning
```

### Offloading Options
```json
{"offload_optimizer": {"device": "cpu"}}     // CPU optimizer offload
{"offload_optimizer": {"device": "nvme"}}    // NVMe optimizer offload
{"offload_param": {"device": "cpu"}}         // CPU param offload (stage 3)
{"offload_param": {"device": "nvme"}}        // NVMe param offload (stage 3)
```

### Supported Optimizers
`Adam`, `AdamW`, `Lamb`, `OneBitAdam`, `OneBitLamb`, `ZeroOneAdam`, `Lion`, `Muon`, `MuAdam`, `MuAdamW`, `MuSGD`, `Adagrad`

### Supported Models for Inference
LLaMA 2, Mistral, Mixtral, Qwen, Qwen v2, Falcon, Phi, Phi-3, OPT, ExaOne4, and custom models via injection policies.
