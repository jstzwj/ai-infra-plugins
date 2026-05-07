# Chapter 38: CLI Arguments Reference

## Source Files
- `sources/Megatron-LM/megatron/training/arguments.py` - Argument definitions
- `sources/Megatron-LM/megatron/core/transformer/transformer_config.py` - Config-driven args

## Overview

Megatron-LM uses a comprehensive command-line argument system with 200+ flags organized into functional groups. Arguments are defined both in `arguments.py` and automatically generated from dataclass fields in `TransformerConfig`.

## Model Architecture Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num-layers` | int | 0 | Number of transformer layers |
| `--hidden-size` | int | 0 | Transformer hidden size |
| `--num-attention-heads` | int | 0 | Number of attention heads |
| `--num-query-groups` | int | None | Number of query groups for GQA |
| `--ffn-hidden-size` | int | None | FFN hidden size (default: 4×hidden) |
| `--kv-channels` | int | None | KV projection dimension |
| `--max-position-embeddings` | int | 0 | Maximum position embeddings |
| `--seq-length` | int | 0 | Input sequence length |
| `--normalization` | str | LayerNorm | LayerNorm or RMSNorm |
| `--swiglu` | flag | False | Use SwiGLU activation |
| `--position-embedding-type` | str | learned_absolute | Position embedding: learned_absolute, rope, none |
| `--rotary-base` | float | 10000 | RoPE base frequency |
| `--rotary-percent` | float | 1.0 | Fraction of dimensions with RoPE |
| `--rotary-interleaved` | flag | False | Interleaved RoPE (RoFormer style) |
| `--no-position-embedding` | flag | False | Disable position embeddings |
| `--make-vocab-size-divisible-by` | int | 128 | Pad vocab to multiple of this |
| `--disable-bias-linear` | flag | False | Remove bias from linear layers |
| `--add-qkv-bias` | flag | False | Add bias to QKV projections |
| `--layernorm-epsilon` | float | 1e-5 | LayerNorm epsilon |
| `--apply-layernorm-1p` | flag | False | Zero-centered gamma for LayerNorm |

## Parallelism Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--tensor-model-parallel-size` | int | 1 | Tensor parallelism degree |
| `--pipeline-model-parallel-size` | int | 1 | Pipeline parallelism degree |
| `--context-parallel-size` | int | 1 | Context parallelism degree |
| `--expert-model-parallel-size` | int | 1 | Expert parallelism degree |
| `--sequence-parallel` | flag | False | Enable sequence parallelism |
| `--num-layers-per-virtual-pipeline-stage` | int | None | Virtual pipeline stages |
| `--overlap-grad-reduce` | flag | False | Overlap gradient all-reduce |
| `--overlap-param-gather` | flag | False | Overlap parameter all-gather |
| `--tp-comm-overlap` | flag | False | Overlap TP communication |
| `--overlap-p2p-communication` | flag | False | Overlap P2P in pipeline |

## Training Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--micro-batch-size` | int | None | Micro batch size per GPU |
| `--global-batch-size` | int | None | Global batch size across GPUs |
| `--train-iters` | int | None | Number of training iterations |
| `--lr` | float | None | Learning rate |
| `--min-lr` | float | None | Minimum learning rate for decay |
| `--lr-decay-style` | str | linear | lr schedule: constant, cosine, polynomial, exponential |
| `--lr-decay-iters` | int | None | Number of LR decay iterations |
| `--lr-warmup-fraction` | float | None | Fraction of warmup steps |
| `--lr-warmup-iters` | int | 0 | Number of warmup iterations |
| `--weight-decay` | float | 0.01 | Weight decay coefficient |
| `--adam-beta1` | float | 0.9 | Adam beta1 |
| `--adam-beta2` | float | 0.999 | Adam beta2 |
| `--adam-eps` | float | 1e-8 | Adam epsilon |
| `--clip-grad` | float | 1.0 | Gradient clipping norm |
| `--bf16` | flag | False | BF16 mixed precision |
| `--fp16` | flag | False | FP16 mixed precision |
| `--loss-scale` | float | None | Static loss scale for FP16 |
| `--initial-loss-scale` | float | 2**32 | Initial dynamic loss scale |
| `--min-loss-scale` | float | 1.0 | Minimum dynamic loss scale |
| `--loss-scale-window` | float | 1000 | Window for dynamic loss scale |
| `--hysteresis` | int | 2 | Hysteresis for dynamic loss scaling |

## Optimizer Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--optimizer` | str | adam | Optimizer: adam, sgd |
| `--use-distributed-optimizer` | flag | False | Enable distributed optimizer |
| `--use-megatron-fsdp` | flag | False | Enable Megatron-FSDP |
| `--data-parallel-sharding-strategy` | str | no_shard | DP sharding: optim, optim_grads, optim_grads_params |
| `--cpu-optimizer` | flag | False | Run optimizer on CPU |
| `--cpu-offloading` | flag | False | Offload activations to CPU |

## FP8/Quantization Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fp8-format` | str | None | FP8 format: e4m3, hybrid |
| `--fp8-recipe` | str | delayed | FP8 recipe: delayed, tensorwise, mxfp8, blockwise |
| `--fp8-param-gather` | flag | False | Keep params in FP8 |
| `--fp4-format` | str | None | FP4 format: e2m1 |
| `--fp4-recipe` | str | nvfp4 | FP4 recipe |
| `--fp4-param-gather` | flag | False | Keep params in FP4 |
| `--fp8-amax-history-len` | int | 1 | AMAX history length |
| `--fp8-amax-compute-algo` | str | most_recent | AMAX algorithm: most_recent, max |
| `--first-last-n-layers-in-bf16` | flag | False | Keep first/last layers in BF16 |

## MoE Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--num-experts` | int | None | Number of MoE experts |
| `--expert-tensor-parallel-size` | int | None | TP for experts |
| `--moe-router-topk` | int | 2 | Top-K routing |
| `--moe-router-load-balancing-type` | str | aux_loss | Load balancing: aux_loss, seq_aux_loss, global_aux_loss, sinkhorn, none |
| `--moe-aux-loss-coeff` | float | 0.0 | Auxiliary loss coefficient |
| `--moe-z-loss-coeff` | float | None | Z-loss coefficient |
| `--moe-grouped-gemm` | flag | False | Use grouped GEMM for experts |
| `--moe-token-dispatcher-type` | str | allgather | Token dispatcher: allgather, alltoall, flex |
| `--moe-router-score-function` | str | softmax | Score function: softmax, sigmoid, sqrtsoftplus |
| `--moe-router-pre-softmax` | flag | False | Pre-softmax routing |
| `--moe-router-dtype` | str | None | Router dtype: fp32, fp64 |
| `--moe-per-layer-logging` | flag | False | Per-layer MoE logging |
| `--moe-shared-expert-intermediate-size` | int | None | Shared expert FFN size |
| `--moe-layer-freq` | int/list | 1 | MoE layer frequency |
| `--moe-router-num-groups` | int | None | Number of routing groups |
| `--moe-router-group-topk` | int | None | Top-K groups in group routing |
| `--moe-enable-deepep` | flag | False | Enable DeepEP dispatcher |

## Checkpoint Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--load` | str | None | Checkpoint load directory |
| `--save` | str | None | Checkpoint save directory |
| `--save-interval` | int | None | Steps between saves |
| `--no-save-optim` | flag | False | Don't save optimizer state |
| `--no-save-rng` | flag | False | Don't save RNG state |
| `--no-load-optim` | flag | False | Don't load optimizer state |
| `--no-load-rng` | flag | False | Don't load RNG state |
| `--finetune` | flag | False | Load checkpoint but don't load optimizer/RNG |
| `--ckpt-format` | str | torch | Checkpoint format: torch, fsdp_dtensor |

## Data Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-path` | str | None | Training data path(s) |
| `--split` | str | 969,30,1 | Train/valid/test split percentages |
| `--tokenizer-type` | str | None | Tokenizer type |
| `--tokenizer-model` | str | None | Tokenizer model path |
| `--vocab-file` | str | None | Vocabulary file |
| `--merge-file` | str | None | BPE merge file |
| `--data-impl` | str | infer | Data implementation: infer, mmap, lazy |
| `--dataloader-type` | str | single | Dataloader: single, cyclic |
| `--num-workers` | int | 2 | Dataloader workers |

## Activation Recompute Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--recompute-granularity` | str | None | Recompute level: full, selective |
| `--recompute-method` | str | None | Recompute method: uniform, block |
| `--recompute-num-layers` | int | None | Layers to recompute |
| `--recompute-modules` | list | core_attn | Modules to recompute: core_attn, mlp, moe, layernorm, moe_act, shared_experts |

## CUDA Graph Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cuda-graph-impl` | str | none | Implementation: none, local, transformer_engine |
| `--cuda-graph-scope` | str | full | Scope: full, attn, mlp, moe, moe_router |
| `--cuda-graph-warmup-steps` | int | 3 | Warmup steps before capture |

## Logging Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--log-interval` | int | 100 | Steps between logging |
| `--log-throughput` | flag | False | Log throughput metrics |
| `--tensorboard-dir` | str | None | TensorBoard directory |
| `--wandb-project` | str | None | Weights & Biases project |
| `--wandb-name` | str | None | W&B run name |
| `--timing-log-level` | int | 0 | Timing log level (0-2) |
| `--profile` | flag | False | Enable PyTorch profiler |
| `--profile-step-start` | int | 10 | Step to start profiling |
| `--profile-step-end` | int | 12 | Step to stop profiling |

## Miscellaneous Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--seed` | int | 1234 | Random seed |
| `--deterministic-mode` | flag | False | Deterministic execution |
| `--exit-interval` | int | None | Exit after N steps |
| `--exit-duration-in-mins` | int | None | Exit after N minutes |
| `--exit-signal-handler` | flag | False | Handle SIGTERM gracefully |
| `--no-initialization` | flag | False | Skip weight initialization |
| `--use-mcore-models` | flag | False | Use Megatron Core model implementations |
| `--transformer-impl` | str | transformer_engine | Implementation: local, transformer_engine, inference_optimized |
