# SGLang Speculative Decoding Reference

This document provides a comprehensive reference for all speculative decoding methods, configurations, and optimization techniques supported by SGLang. Speculative decoding accelerates LLM inference by generating draft tokens (from a smaller model or auxiliary mechanism) and verifying them in parallel with the target model.

## Table of Contents

- [Overview](#overview)
- [Method Comparison](#method-comparison)
- [EAGLE-2 Decoding](#eagle-2-decoding)
- [EAGLE-3 Decoding](#eagle-3-decoding)
- [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
- [DFLASH Decoding](#dflash-decoding)
- [Standalone Draft Model](#standalone-draft-model)
- [NGRAM Speculative Decoding](#ngram-speculative-decoding)
- [Adaptive Speculative Decoding](#adaptive-speculative-decoding)
- [Speculative Decoding V2 (Overlap Scheduler)](#speculative-decoding-v2-overlap-scheduler)
- [EAGLE with torch.compile](#eagle-with-torchcompile)
- [Frequency-Ranked Speculative Sampling (FR-Spec)](#frequency-ranked-speculative-sampling-fr-spec)
- [Full Parameter Reference](#full-parameter-reference)
- [Performance Benchmarks](#performance-benchmarks)
- [OOM Troubleshooting](#oom-troubleshooting)
- [Source Code Structure](#source-code-structure)

---

## Overview

SGLang provides several speculative decoding options, including EAGLE-2/EAGLE-3, MTP, DFLASH, classic draft-model decoding, and an NGRAM-based variant. The implementation is considered among the fastest in open-source LLM engines.

### Quick Guidance

| Goal | Method | Configuration |
|------|--------|---------------|
| Best speed/quality (recommended) | EAGLE-3 | `--speculative-algorithm EAGLE3` |
| Strong default, broad compatibility | EAGLE-2 | `--speculative-algorithm EAGLE` |
| Variable acceptance over time | EAGLE + Adaptive | `--speculative-algorithm EAGLE --speculative-eagle-topk 1 --speculative-adaptive` |
| Lower lm_head overhead for EAGLE-2 | FR-Spec | Add `--speculative-token-map` |
| Model has MTP heads built-in | MTP | Small `speculative_num_steps/topk/num_draft_tokens` |
| DFlash draft checkpoint available | DFLASH | `--speculative-algorithm DFLASH` |
| Smaller draft LLM available | STANDALONE | `--speculative-algorithm STANDALONE` |
| No extra model available | NGRAM | `--speculative-algorithm NGRAM` (CUDA-only) |
| Overlap scheduler (experimental) | SpecV2 | `SGLANG_ENABLE_SPEC_V2=True` |

---

## Method Comparison

| Method | Draft Source | Separate Draft Model? | How to Enable | Constraints |
|--------|-------------|----------------------|---------------|-------------|
| EAGLE-2 | EAGLE draft model (feature drafting + tree) | Typically yes | `--speculative-algorithm EAGLE` + `--speculative-draft-model-path` | Tune `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` |
| EAGLE-2 + torch.compile | Same as EAGLE-2 | Typically yes | Add `--enable-torch-compile` | Benefit varies by hardware; benchmark to verify |
| EAGLE-2 + FR-Spec | Same + token subset | Typically yes | Add `--speculative-token-map` | Reduces lm_head overhead |
| EAGLE-3 | EAGLE3 draft model | Yes | `--speculative-algorithm EAGLE3` + `--speculative-draft-model-path` | Best throughput in benchmarks |
| MTP | Built-in multi-token heads | Often no | See MTP section | Draft path may be auto-handled |
| DFLASH | DFlash draft model (linear block verification) | Yes | `--speculative-algorithm DFLASH` + `--speculative-draft-model-path` | No `--enable-dp-attention`; `pp_size == 1`; disables overlap scheduler and mixed chunked prefill |
| STANDALONE | Smaller draft LLM (token-level) | Yes | `--speculative-algorithm STANDALONE` + `--speculative-draft-model-path` | Does not support `--enable-dp-attention` |
| SpecV2 (experimental) | V2 workers + overlap scheduler | N/A | `SGLANG_ENABLE_SPEC_V2=True` | Only supports `--speculative-eagle-topk 1` |
| NGRAM | Ngram cache from previous tokens | No | `--speculative-algorithm NGRAM` | CUDA-only; no `--enable-dp-attention`; disables overlap scheduler and mixed chunked prefill |

---

## EAGLE-2 Decoding

EAGLE-2 uses a lightweight draft model that predicts feature vectors (hidden states) and constructs a tree of candidate tokens for parallel verification by the target model.

### How EAGLE Works

1. The draft model predicts the next feature vector using the feature sequence and token sequence
2. The next token is sampled from the predicted feature via LMHead
3. Sequences are extended in a tree style with branching factor controlled by `--speculative-eagle-topk`
4. The draft tree is expanded for configured steps, then reranked to select the top `speculative_num_draft_tokens` nodes
5. The target model verifies these draft tokens in a single forward pass

### Usage

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

### Key Parameters for EAGLE

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-draft-model-path` | Draft model path. Typically required for EAGLE. | `None` |
| `--speculative-num-steps` | Depth of autoregressive drafting. Increases speculation range but risks rejection cascades. | Auto (5 for Llama/Grok; 3 for others) |
| `--speculative-eagle-topk` | Branching factor per step. Higher values improve candidate diversity but increase memory/compute. | Auto (4 for Llama/Grok; 1 for others) |
| `--speculative-num-draft-tokens` | Maximum parallel verification capacity. Allows deeper tree evaluation but increases GPU memory. | Auto (8 for Llama/Grok; 4 for others) |
| `--speculative-accept-threshold-single` | Acceptance threshold for single-token verification. Lower values accept more aggressively. | `1.0` |
| `--speculative-accept-threshold-acc` | Accumulated acceptance threshold across steps. | `1.0` |
| `--speculative-attention-mode` | Attention mode for speculative operations: `prefill` or `decode`. | `"prefill"` |
| `--speculative-draft-attention-backend` | Override attention backend for the draft model. | `None` (same as target) |
| `--speculative-draft-model-quantization` | Quantization method for the draft model. Use `"unquant"` to force no quantization. | Same as target |
| `--speculative-draft-model-revision` | Specific revision/commit of the draft model. | Auto-set to `"main"` when draft path is set |
| `--speculative-draft-load-format` | Load format for the draft model weights. | `None` |

**Tuning tips**:
- Leave all three (`num_steps`, `topk`, `num_draft_tokens`) unset for auto-tuning, or set all three explicitly
- If `topk=1`, `num_draft_tokens` is automatically adjusted to `num_steps + 1`
- Use `bench_speculative.py` to find the best parameter combination

---

## EAGLE-3 Decoding

EAGLE-3 removes the feature prediction objective, incorporates low and mid-layer features, and is trained in an on-policy manner. This results in better drafting accuracy and higher throughput compared to EAGLE-2.

### Usage

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**Notes**: `--speculative-token-map` (FR-Spec) is ignored for EAGLE-3 models.

### Training EAGLE Models

For guidance on training your own EAGLE model, see the [EAGLE repo](https://github.com/SafeAILab/EAGLE). For EAGLE-3 training specifically, check out [SpecForge](https://github.com/sgl-project/SpecForge), the SGLang team's training framework.

---

## Multi-Token Prediction (MTP)

MTP leverages built-in multi-token prediction heads in models like DeepSeek V3 and Xiaomi MiMo. It uses the speculative decoding infrastructure but does not require a separate draft model in many cases.

### Usage with MiMo

```bash
python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --host 0.0.0.0 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

### Usage with DeepSeek V3

For DeepSeek V3 MTP usage, refer to the DeepSeek V3.2 documentation. MTP with DeepSeek uses small `speculative_num_steps`, `topk`, and `num_draft_tokens` values.

---

## DFLASH Decoding

DFLASH uses a dedicated draft model checkpoint that verifies a linear draft block (rather than tree verification like EAGLE). It is configured around a block size / draft window.

### DFLASH-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-draft-model-path` | Required DFlash draft model path. | `None` |
| `--speculative-num-draft-tokens` | DFlash verify block size. | Inferred from draft config, or `16` |
| `--speculative-dflash-block-size` | Alias of `--speculative-num-draft-tokens` for DFlash. | `None` |
| `--speculative-dflash-draft-window-size` | Draft KV sliding-window size. Must be >= `speculative-num-draft-tokens`. | `None` |

### Usage

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
```

**Constraints**:
- Does NOT support `--enable-dp-attention`
- Requires `pp_size == 1`
- Disables overlap scheduler and mixed chunked prefill

---

## Standalone Draft Model

Uses a smaller LLM as the draft model for token-level speculative decoding. The draft model generates candidate tokens autoregressively, which the target model verifies in a single pass.

### Parameters for STANDALONE

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-draft-model-path` | Draft model weights (smaller than target). | `None` |
| `--speculative-num-steps` | Draft depth (autoregressive steps). | `3` (auto for STANDALONE) |
| `--speculative-eagle-topk` | Branching factor (token candidates per step). | `1` (auto for STANDALONE) |
| `--speculative-num-draft-tokens` | Verification capacity. | `4` (auto for STANDALONE) |
| `--speculative-draft-model-quantization` | Quantization for draft model. Use `"unquant"` to disable. | Same as target |

### Usage

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 7 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Constraint**: Does NOT support `--enable-dp-attention`.

---

## NGRAM Speculative Decoding

NGRAM-based speculative decoding does not require a separate draft model. It retrieves draft tokens from an ngram cache built from previously generated tokens and verifies them with the target model.

### NGRAM-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-num-draft-tokens` | Number of draft tokens verified per step. If omitted, defaults to `min(--speculative-ngram-max-trie-depth, 12)`. | `12` (with default settings) |
| `--speculative-ngram-min-bfs-breadth` | Minimum BFS breadth. | `1` |
| `--speculative-ngram-max-bfs-breadth` | Maximum BFS breadth. | `10` |
| `--speculative-ngram-match-type` | Ngram tree-building mode: `"BFS"` (recency-based) or `"PROB"` (frequency-based). | `"BFS"` |
| `--speculative-ngram-max-trie-depth` | Maximum suffix length stored and matched by the ngram trie. | `18` |
| `--speculative-ngram-capacity` | Cache capacity (number of entries). | `10,000,000` |

### Usage

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-bfs-breadth 10 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Constraints**:
- CUDA-only
- Does NOT support `--enable-dp-attention`
- Disables overlap scheduler and mixed chunked prefill
- If `--speculative-ngram-max-bfs-breadth > 1` and `page_size > 1`, must use `--attention-backend flashinfer`
- Optional: Set `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True` to force greedy verification

---

## Adaptive Speculative Decoding

Adaptive speculative decoding adjusts `speculative_num_steps` and `speculative_num_draft_tokens` at runtime based on acceptance behavior. This is designed for workloads where the optimal step count varies over time.

### Current Support

- Only `--speculative-algorithm EAGLE`
- Only `--speculative-eagle-topk 1`
- If either condition is not met, SGLang falls back to static settings

### Why Adaptive Steps Help

- If `num_steps` is too small: the draft model could have produced more accepted tokens, but the round stops too early
- If `num_steps` is too large: the draft model produces many rejected tokens, wasting compute
- Real traffic often moves between high-acceptance and low-acceptance phases

### Design

The adaptive mechanism has three components:
1. **AdaptiveSpeculativeParams**: EMA-based policy
2. **SpecRuntimeState**: Per-tier runtime state bundle (CUDA graphs, attention backends)
3. **AdaptiveController**: Coordinator that chooses a tier and activates matching runtime state

At startup, SGLang pre-builds one runtime state per candidate tier. Default tiers: `[1, 3, 7]`.

Tier switching happens after the current round completes. Backends and CUDA graphs are never swapped mid-round.

### Usage

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-eagle-topk 1 \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-adaptive
```

### Configuration File

Override defaults with `--speculative-adaptive-config /path/to/adaptive_spec.json`:

```json
{
  "candidate_steps": [1, 3, 7],
  "ema_alpha": 0.2,
  "warmup_batches": 10,
  "update_interval": 5,
  "down_hysteresis": -0.25,
  "up_hysteresis": 0.0
}
```

| Config Key | Default | Meaning |
|------------|---------|---------|
| `candidate_steps` | `[1, 3, 7]` | Discrete `speculative_num_steps` tiers to switch between |
| `ema_alpha` | `0.2` | EMA smoothing factor for accepted draft length |
| `update_interval` | `5` | Recompute interval in verify batches after warmup |
| `warmup_batches` | `10` | Number of verify batches to observe before switching |
| `down_hysteresis` | `-0.25` | Extra margin before moving to a smaller step |
| `up_hysteresis` | `0.0` | Extra margin before moving to a larger step |

### Monitoring

Inspect the active tier and acceptance metrics via `/server_info`:

```bash
curl -s http://127.0.0.1:30000/server_info | \
    jq '.internal_states[0] | {speculative_num_steps, avg_spec_accept_length}'
```

### Tuning Tips

- Start with default candidate tiers `[1, 3, 7]`
- Use fewer tiers for lower startup and graph-memory overhead
- Increase `ema_alpha` to react faster, or lower it for more stability
- Increase `warmup_batches` or `update_interval` if tier switching is too noisy
- If workload is stable, adaptive mode may not help much

---

## Speculative Decoding V2 (Overlap Scheduler)

SpecV2 is an experimental implementation that enables an overlap scheduler using V2 speculative workers (`StandaloneWorkerV2`, `EAGLEWorkerV2`).

### Enabling SpecV2

Set the environment variable:
```bash
SGLANG_ENABLE_SPEC_V2=True
```

**Critical**: SpecV2 only supports `--speculative-eagle-topk 1`. You MUST set this explicitly:
```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Important notes**:
- If you explicitly set `--speculative-eagle-topk > 1`, the server will error
- If you omit `--speculative-eagle-topk`, auto-tuning may pick topk > 1 for some models (e.g., Llama), which is incompatible
- Applies to `EAGLE`, `EAGLE3`, and `STANDALONE` algorithms

---

## EAGLE with torch.compile

Optionally enable `torch.compile` to apply kernel-level optimizations (operator fusion, autotune) to the draft model.

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --enable-torch-compile \
    --torch-compile-max-bs 8 \
    --log-level warning
```

**Note**: The actual speedup depends on hardware, model architecture, and batch size. In some configurations (e.g., small draft models on H100 where cuBLAS is already optimal and CUDA graphs are enabled), the benefit may be negligible. Benchmark with and without this flag.

---

## Frequency-Ranked Speculative Sampling (FR-Spec)

FR-Spec reduces `lm_head` computational overhead by using a truncated high-frequency token vocabulary in the draft model. This accelerates the pipeline without quality degradation.

### Usage

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

High-frequency tokens for FR-Spec are available from the [FR-Spec model](https://huggingface.co/thunlp/LLaMA3-Instruct-8B-FR-Spec) or the [FR-Spec repo](https://github.com/thunlp/FR-Spec).

**Note**: `--speculative-token-map` is ignored for EAGLE-3 models.

---

## Full Parameter Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--speculative-algorithm` | `str` | `None` | Algorithm: `DFLASH`, `EAGLE`, `EAGLE3`, `STANDALONE`, `NGRAM`, `NEXTN` (alias of `EAGLE`) |
| `--speculative-draft-model-path` | `str` | `None` | Path to the draft model weights |
| `--speculative-draft-model-revision` | `str` | `None` | Specific revision/commit of the draft model (`"main"` auto-used when draft path set) |
| `--speculative-draft-load-format` | `str` | `None` | Load format for draft model weights |
| `--speculative-num-steps` | `int` | `None` (auto) | Autoregressive drafting depth |
| `--speculative-eagle-topk` | `int` | `None` (auto) | Branching factor per drafting step |
| `--speculative-num-draft-tokens` | `int` | `None` (auto) | Maximum number of draft tokens for verification |
| `--speculative-dflash-block-size` | `int` | `None` | DFlash-only alias of `--speculative-num-draft-tokens` |
| `--speculative-dflash-draft-window-size` | `int` | `None` | DFlash-only draft KV sliding-window size |
| `--speculative-accept-threshold-single` | `float` | `1.0` | Single-token acceptance threshold |
| `--speculative-accept-threshold-acc` | `float` | `1.0` | Accumulated acceptance threshold |
| `--speculative-token-map` | `str` | `None` | Path to FR-Spec high-frequency token map |
| `--speculative-attention-mode` | `str` | `"prefill"` | Attention mode for speculative operations (`"prefill"` or `"decode"`) |
| `--speculative-draft-attention-backend` | `str` | `None` | Override attention backend for the draft model |
| `--speculative-moe-runner-backend` | `str` | `None` | MoE runner backend for the draft model |
| `--speculative-moe-a2a-backend` | `str` | `None` | MoE all-to-all backend for the draft model |
| `--speculative-draft-model-quantization` | `str` | Same as target | Quantization for the draft model (`"unquant"` to disable) |
| `--speculative-adaptive` | flag | `False` | Enable adaptive speculative decoding |
| `--speculative-adaptive-config` | `str` | `None` | Path to adaptive speculative config JSON |

### NGRAM-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--speculative-ngram-min-bfs-breadth` | `int` | `1` | Minimum BFS breadth |
| `--speculative-ngram-max-bfs-breadth` | `int` | `10` | Maximum BFS breadth |
| `--speculative-ngram-match-type` | `str` | `"BFS"` | Ngram tree-building mode: `"BFS"` or `"PROB"` |
| `--speculative-ngram-max-trie-depth` | `int` | `18` | Maximum suffix length stored and matched |
| `--speculative-ngram-capacity` | `int` | `10,000,000` | Cache capacity (number of entries) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_ENABLE_SPEC_V2` | `False` | Enable Speculative Decoding V2 (overlap scheduler) |
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `False` | Force greedy verification for ngram decoding |

### Related Flags

| Parameter | Description |
|-----------|-------------|
| `--enable-multi-layer-eagle` | Enable multi-layer EAGLE (auto-enabled for MiMoV2 and Step3p5 models) |
| `--enable-torch-compile` | Enable `torch.compile` for kernel-level optimizations |
| `--torch-compile-max-bs` | Maximum batch size for `torch.compile` |
| `--mem-fraction-static` | Controls memory budget for model weights + KV cache pool (lower to reduce OOM) |
| `--cuda-graph-max-bs` | Maximum batch size for CUDA graph capture (fewer captures = less memory) |
| `--max-running-requests` | Limit concurrent requests to reduce in-flight load |

---

## Performance Benchmarks

### LLaMA-3.1 8B Instruct on MT-Bench (1x H100)

| Method | Throughput (tokens/s) | Speedup |
|--------|-----------------------|---------|
| SGLang (without speculative) | 158.34 | 1.0x |
| SGLang + EAGLE-2 | 244.10 | 1.54x |
| SGLang + EAGLE-3 | 373.25 | 2.36x |

EAGLE-3 provides approximately 2.36x throughput improvement over the non-speculative baseline.

---

## OOM Troubleshooting

Speculative decoding increases GPU memory usage because the draft tree, CUDA graphs, and verification-related buffers consume additional VRAM.

### Step 1: Lower Static Memory Fraction (Most Effective)

```bash
--mem-fraction-static 0.5   # when omitted, auto-computed
```

This controls the memory budget for model weights + KV cache pool. Lowering it increases dynamic headroom for activations and CUDA graph buffers.

### Step 2: Reduce CUDA Graph Batch Size

```bash
--cuda-graph-max-bs 4   # or even 2 for tight memory
```

Fewer CUDA graph captures means less memory reserved.

### Step 3: Reduce Draft Tree Size

```bash
# Before (aggressive, high memory)
--speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64

# After (conservative, lower memory)
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### Step 4: Limit Concurrent Requests

```bash
--max-running-requests 4
```

### Quick OOM Recovery Recipe

Start with this minimal configuration and scale up:

```bash
python3 -m sglang.launch_server \
    --model <your-model> \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <your-draft-model> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-max-bs 2 \
    --mem-fraction-static 0.5 \
    --max-running-requests 4 \
    --log-level warning
```

Then gradually increase `--speculative-num-draft-tokens`, `--speculative-eagle-topk`, and `--cuda-graph-max-bs`. Increase `--mem-fraction-static` last, only after the run is stable.

---

## Source Code Structure

The speculative decoding implementation is located in `python/sglang/srt/speculative/`:

| File | Description |
|------|-------------|
| `base_spec_worker.py` | Base class for all speculative workers |
| `eagle_worker.py` | EAGLE-2 worker implementation |
| `eagle_worker_v2.py` | EAGLE-2 worker with V2 overlap scheduler |
| `multi_layer_eagle_worker.py` | Multi-layer EAGLE worker (MiMoV2, Step3p5) |
| `multi_layer_eagle_worker_v2.py` | Multi-layer EAGLE V2 worker |
| `standalone_worker.py` | Standalone draft model worker |
| `standalone_worker_v2.py` | Standalone worker with V2 overlap scheduler |
| `dflash_worker.py` | DFLASH worker implementation |
| `ngram_worker.py` | NGRAM speculative decoding worker |
| `eagle_draft_cuda_graph_runner.py` | CUDA graph runner for EAGLE draft model |
| `eagle_draft_extend_cuda_graph_runner.py` | CUDA graph runner for EAGLE draft extend |
| `multi_layer_eagle_draft_extend_cuda_graph_runner.py` | CUDA graph for multi-layer EAGLE |
| `eagle_utils.py` | EAGLE utility functions |
| `eagle_info.py` | EAGLE speculative info dataclass |
| `eagle_info_v2.py` | EAGLE V2 speculative info |
| `multi_layer_eagle_utils.py` | Multi-layer EAGLE utilities |
| `dflash_info.py` | DFLASH speculative info |
| `dflash_utils.py` | DFLASH utility functions |
| `ngram_info.py` | NGRAM speculative info |
| `spec_info.py` | Base speculative info |
| `spec_utils.py` | General speculative utilities |
| `draft_utils.py` | Draft model utilities |
| `adaptive_spec_params.py` | Adaptive speculative EMA policy |
| `adaptive_runtime_state.py` | Per-tier runtime state |
| `lora_drainer.py` | LoRA adapter drain management |
| `external_corpus_manager.py` | External corpus for ngram |
| `cpp_ngram/` | C++ ngram implementation |

---

## References

- [EAGLE-2 Paper](https://arxiv.org/abs/2406.16858) - Feature-level speculative decoding
- [EAGLE-3 Paper](https://arxiv.org/abs/2503.01840) - On-policy training, low/mid-layer features
- [MTP Paper](https://arxiv.org/pdf/2404.19737) - Multi-Token Prediction
- [FR-Spec Paper](https://arxiv.org/pdf/2502.14856) - Frequency-Ranked Speculative Sampling
- [S-LoRA Paper](https://arxiv.org/pdf/2311.03285) - Serving thousands of LoRA adapters
- [SpecForge](https://github.com/sgl-project/SpecForge) - EAGLE-3 training framework
- [EAGLE Repo](https://github.com/SafeAILab/EAGLE) - EAGLE model training
- [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) - Benchmarking script
