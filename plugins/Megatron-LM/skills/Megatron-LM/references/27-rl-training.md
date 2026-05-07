# Chapter 27: RL Training (Reinforcement Learning)

## Source Files
- `megatron/rl/rl_utils.py` - Core RL utilities, GRPO loss, rollout management
- `megatron/rl/sequence_packing_utils.py` - Sequence packing for RL
- `megatron/rl/parallel_utils.py` - Parallel execution utilities
- `megatron/rl/logging.py` - RL-specific logging
- `megatron/rl/agent/api.py` - Agent API (Rollout, RolloutGroup, EvaluationRequest)
- `megatron/rl/agent/weighted_multi_task.py` - Weighted multi-task agent
- `megatron/rl/agent/remote_agent.py` - Remote agent for distributed evaluation
- `megatron/rl/agent/reward_only_agent.py` - Reward-only evaluation agent
- `megatron/rl/agent/huggingface_dataset_agent.py` - Agent using HF datasets
- `megatron/rl/agent/pass_at_evaluation_agent.py` - Pass@k evaluation agent
- `megatron/rl/inference/inference_interface.py` - Inference interface
- `megatron/rl/inference/megatron.py` - MegatronLocal inference backend
- `megatron/rl/inference/api.py` - Inference request/response types
- `megatron/rl/server/api.py` - RL server API
- `megatron/rl/server/inference/inference_interface_server.py` - Inference server
- `megatron/rl/server/agent/fastapi_env_server.py` - Agent server
- `train_rl.py` - RL training entry point

## Overview

Megatron-LM provides a built-in reinforcement learning training framework for aligning large language models. The primary algorithm supported is GRPO (Group Relative Policy Optimization), with infrastructure supporting PPO-like training patterns.

The RL framework integrates tightly with Megatron's distributed training:
- Leverages tensor, pipeline, expert, and data parallelism
- Supports CUDA graphs for both training and inference
- Handles optimizer offloading during inference to maximize GPU memory for generation
- Supports sequence packing for efficient RL training

## GRPO Algorithm

Group Relative Policy Optimization is a variant of reinforcement learning from human feedback (RLHF) that uses group-wise advantage normalization instead of a separate value function.

### Core GRPO Loss

The GRPO loss is computed in `calculate_grpo_loss`:

```
L_GRPO = -E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]

Where:
- ratio = pi_theta(a|s) / pi_old(a|s)  (new policy / old policy probability ratio)
- A = (R - mean(R_group)) / std(R_group)  (normalized advantage within group)
- clip(ratio, 1-eps, 1+eps) = clipped ratio for PPO-style trust region
```

### Key Differences from PPO

1. **No value function:** GRPO uses group-relative advantages instead of a learned value function
2. **Group normalization:** Rewards within a group of responses to the same prompt are normalized
3. **Simpler architecture:** Only one model (policy) is trained, no critic network needed

### Grouped Rollouts

GRPO generates multiple responses (rollouts) per prompt and normalizes rewards within each group:

```
Prompt: "Solve 2+2"
  ├── Rollout 1: "4" (reward: 1.0)
  ├── Rollout 2: "three" (reward: 0.0)
  └── Rollout 3: "4.0" (reward: 0.8)

Group advantage for Rollout 1: (1.0 - 0.6) / 0.5 = 0.8  (high advantage)
Group advantage for Rollout 2: (0.0 - 0.6) / 0.5 = -1.2  (low advantage)
```

## PPO Support

While GRPO is the primary algorithm, the framework also supports PPO-style training:

- Policy model: Generates responses
- Value/critic model: Estimates state values for advantage computation
- Reference model: Provides KL divergence penalty

## Reward Models

### Reward Computation

Rewards are computed by the environment agent, which can be:

1. **Remote API agent:** Calls an external reward service
2. **HuggingFace dataset agent:** Uses pre-computed rewards from datasets
3. **Reward-only agent:** Computes rewards from a reward model
4. **Custom agent:** User-defined reward computation

### Reward Types

- **Binary:** 1.0 for correct, 0.0 for incorrect
- **Continuous:** Real-valued reward (e.g., from a reward model)
- **Rule-based:** Deterministic reward from code execution, math verification, etc.

## Sequence Packing for RL

RL training benefits significantly from sequence packing since generated responses have variable lengths:

```bash
--rl-use-sequence-packing
```

### Packing Process

1. Collect rollouts with their token sequences
2. Pack multiple short sequences into bins of `seq_length` tokens
3. Create `PackedSeqParams` with:
   - `cu_seqlens`: Cumulative sequence lengths
   - `max_seqlen`: Maximum sequence length in the packed batch
   - `packing_params`: Per-token mapping to original sequences

### Packing Efficiency Metrics

```python
packing_efficiency = actual_tokens / (total_bin_capacity)
compute_tokens = sum_of_packed_sequence_lengths
actual_tokens = sum_of_original_sequence_lengths
```

Logged metrics:
- `compute_toks/s`: Total tokens in packed bins / time
- `actual_toks/s`: Real non-padding tokens / time
- `packing_eff`: Fraction of bin capacity filled with real tokens

## Inference Model for Rollouts

### Single-Model Setup

By default, the same model is used for both training and inference:

```bash
# No separate inference model
python train_rl.py ...
```

The training model is temporarily switched to inference mode for rollout generation.

### Separate Inference Model

For better performance, a separate inference model can be used:

```bash
--rl-separate-inference-model
--rl-inference-model-unified-memory-level 1   # Use UVM for offloading
```

With a separate inference model:
1. Training model parameters are copied to the inference model via `swap_model_weights`
2. Inference model runs generation on GPU
3. Training model continues training with collected rollouts
4. Weight transfer methods: `refit_method` (e.g., "refit", "copy")

### Weight Offloading

When the inference model is idle, its weights can be offloaded to CPU:

```bash
--rl-offload-inference-model-weights-when-idle
```

Two offloading mechanisms:
1. **UVM (Unified Virtual Memory):** When `--rl-inference-model-unified-memory-level=1`
   - Uses CUDA managed memory for seamless CPU/GPU transfer
   - Automatic page migration based on access patterns

2. **torch_memory_saver:** When UVM is not available
   - Explicit pause/resume of model weights
   - Requires `pip install torch_memory_saver`

### Optimizer Offloading During Inference

During rollout generation, the training optimizer state is offloaded to free GPU memory:

```bash
--rl-offload-optimizer-during-inference
```

This calls:
```python
optimizer.offload_to_cpu()   # Before inference: move state to CPU
optimizer.restore_from_cpu()  # After inference: restore to GPU
```

## KL Divergence Regularization

KL divergence between the policy and reference model prevents the policy from diverging too far:

```bash
--rl-kl-coefficient 0.01   # KL penalty coefficient
```

The KL penalty is added to the loss:
```
L_total = L_GRPO + beta * KL(pi_theta || pi_ref)
```

Where `beta` is the KL coefficient.

## Entropy Regularization

Entropy regularization encourages exploration:

```bash
--rl-entropy-coefficient 0.01
```

Higher entropy coefficient encourages the policy to maintain a more uniform distribution over actions, preventing premature convergence.

## Agent System

### Agent Architecture

```
AgentBaseModel
├── RolloutRequest              # Request N rollouts
├── GroupedRolloutRequest       # Request grouped rollouts (for GRPO)
├── EvaluationRequest           # Request evaluation of N prompts
├── Rollout                     # Single rollout data
├── TokenRollout                # Tokenized rollout
├── RolloutGroup                # Group of rollouts for one prompt
├── ContrastiveRollout          # Preference data (chosen/rejected)
└── Head2HeadRolloutRequest     # Head-to-head comparison

WeightedMultiTask
├── from_config(config)         # Load from YAML config
├── get_grouped_rollouts(req)   # Generate grouped rollouts
└── Multiple environments with weighted sampling
```

### Environment Configuration

Environments are configured via YAML:

```yaml
# langrl_env_config.yaml
environments:
  - name: math_env
    weight: 0.5
    config:
      type: huggingface
      dataset: math_dataset
      reward_type: rule_based
  - name: code_env
    weight: 0.5
    config:
      type: remote
      url: http://reward-server:8080
      reward_type: execution
```

```bash
--langrl-env-config langrl_env_config.yaml
```

### Rollout Data Structure

```python
class Rollout:
    trajectory: list[str]             # The generated text turns
    prompt_length: list[int]          # Length of each prompt
    reward: float                     # Computed reward
    env_id: str                       # Environment identifier
    problem_id: str                   # Problem identifier
    policy_epoch: list[list[tuple]]   # KV cache epoch tracking
    kv_cache_epoch: list[list[tuple]] # KV cache management
    num_evictions: list[int]          # KV cache eviction counts

class TokenRollout:
    trajectory: list[list[int]]       # Tokenized trajectory
    reward: list[float] | float       # Per-token or sequence reward
    generation_mask: list[list[bool]] # Which tokens were generated
    logprobs: list[list[float]]       # Per-token log probabilities
```

### Streaming / Partial Rollouts

For long-running generation tasks, partial rollouts can be streamed:

```bash
--rl-partial-rollouts
```

This enables:
- Incremental rollout collection
- Better overlap between generation and training
- Reduced peak memory for long sequences

## Inference Interface

### MegatronLocal

The primary inference backend uses Megatron's own inference engine:

```python
class MegatronLocal(InferenceInterface):
    @classmethod
    async def launch(cls, model, host, port, verbose=False):
        """Launch a local inference server."""
```

### Inference Request/Response

```python
class InferenceRequest:
    prompt: str | list[LLMChatMessage]
    generation_args: GenericGenerationArgs

class InferenceResponse:
    # Contains generated text, tokens, logprobs
```

### Generation Arguments

```bash
--rl-default-temperature 0.7
--rl-default-top-p 0.9
--rl-default-top-k 50
--inference-max-seq-length 4096
```

## Training Configuration

### Key RL Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rl-generation-batch-size` | 32 | Number of prompts per rollout batch |
| `--rl-default-temperature` | 0.7 | Sampling temperature |
| `--rl-default-top-p` | 0.9 | Nucleus sampling threshold |
| `--rl-default-top-k` | 50 | Top-k sampling threshold |
| `--rl-kl-coefficient` | 0.01 | KL divergence penalty |
| `--rl-entropy-coefficient` | 0.01 | Entropy bonus |
| `--rl-use-sequence-packing` | False | Enable sequence packing |
| `--rl-offload-optimizer-during-inference` | False | Offload optimizer during generation |
| `--rl-offload-inference-model-weights-when-idle` | False | Offload inference model when idle |
| `--rl-separate-inference-model` | False | Use separate model for inference |
| `--rl-training-cuda-graphs` | False | CUDA graphs for training step |
| `--rl-partial-rollouts` | False | Enable streaming rollouts |
| `--rl-parallel-generation-tasks` | None | Parallel generation tasks |
| `--rl-enforce-generation-order` | False | Deterministic generation order |
| `--grpo-filter-groups-with-same-reward` | False | Filter groups where all rewards equal |
| `--rl-verify-model-weights-swap` | False | Verify weight swap correctness |

## RL Training Loop

```
train_rl.py
  ├── pretrain()
  │   ├── model_provider()         # Build training + optional inference model
  │   ├── forward_step()           # GRPO loss computation
  │   └── train()
  │       └── for iteration:
  │           ├── get_environment_rollouts()
  │           │   ├── offload optimizer (optional)
  │           │   ├── swap model weights to inference model
  │           │   ├── collect rollouts via agent
  │           │   │   └── get_rollout_generator()
  │           │   │       └── agent.get_grouped_rollouts()
  │           │   │           └── inference_interface.generate()
  │           │   ├── broadcast rollouts to all ranks
  │           │   └── restore optimizer (optional)
  │           ├── compute logprobs
  │           │   ├── get_logprobs() on collected rollouts
  │           │   └── align inference logprobs with training logprobs
  │           ├── compute advantages
  │           │   └── normalize within groups
  │           ├── pack sequences (optional)
  │           ├── forward_backward_func()
  │           │   └── calculate_grpo_loss()
  │           │       ├── ratio = exp(new_logprobs - old_logprobs)
  │           │       ├── clipped_ratio = clip(ratio, 1-eps, 1+eps)
  │           │       └── loss = -min(ratio * A, clipped_ratio * A)
  │           ├── optimizer step
  │           └── log metrics
  └── destroy
```

## Rollout Statistics

The `RolloutStats` dataclass tracks:

```python
@dataclass
class RolloutStats:
    rewards: list[list[float]]         # Per-group rewards
    env_ids: list[str]                 # Environment identifiers
    turn_lens: list[list[int]]         # Token lengths per turn
    traj_lens: list[list[int]]         # Token lengths per trajectory
    advantages: list[list[float]]      # Computed advantages
    min/max/mean_piold_to_inf_prob     # Policy divergence metrics
    min/max/mean_inf_train_prob_abs_diff  # Inference vs train agreement
    min/max/mean_inf_prob              # Inference probability statistics
    policy_epoch: list[list[int]]      # KV cache tracking
    kv_cache_epoch: list[list[int]]    # KV cache epoch
    num_evictions: list[list[int]]     # KV cache evictions
```

## Configuration Examples

### Basic GRPO Training
```bash
python train_rl.py \
    --langrl-env-config env.yaml \
    --rl-generation-batch-size 64 \
    --rl-default-temperature 0.7 \
    --rl-kl-coefficient 0.01 \
    --num-layers 32 --hidden-size 4096 \
    --num-attention-heads 32 --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 --global-batch-size 128 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer \
    --lr 1e-6 --min-lr 1e-7
```

### GRPO with Sequence Packing and CUDA Graphs
```bash
python train_rl.py \
    --langrl-env-config env.yaml \
    --rl-use-sequence-packing \
    --cuda-graph-impl local \
    --rl-training-cuda-graphs \
    --rl-offload-optimizer-during-inference \
    --rl-generation-batch-size 128 \
    ...
```

### GRPO with Separate Inference Model
```bash
python train_rl.py \
    --langrl-env-config env.yaml \
    --rl-separate-inference-model \
    --rl-inference-model-unified-memory-level 1 \
    --rl-offload-inference-model-weights-when-idle \
    --rl-verify-model-weights-swap \
    --refit-method refit \
    ...
```
