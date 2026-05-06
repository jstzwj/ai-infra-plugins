# DeepSpeed4Science

## Overview

DeepSpeed4Science is a dedicated initiative within the DeepSpeed ecosystem that brings high-performance distributed computing capabilities to scientific computing workloads. The program focuses on accelerating critical scientific applications -- including protein structure prediction, genomics analysis, drug discovery, and molecular dynamics -- by providing custom CUDA kernels, specialized attention mechanisms, and optimized training pipelines tailored for scientific domains.

The initiative emerged from a collaboration between Microsoft Research and the scientific computing community, recognizing that many scientific workloads share computational patterns (large-scale attention, sequential data processing, multi-modal fusion) that can benefit from the same systems-level optimizations developed for large language model training. DeepSpeed4Science adapts and extends DeepSpeed's core infrastructure -- ZeRO optimization, pipeline parallelism, mixed-precision training, and kernel-level acceleration -- for scientific use cases.

### Key Contributions

- **Evoformer Attention Kernels**: Custom CUDA kernels implementing the Evoformer attention mechanism used in AlphaFold2-style protein structure prediction, delivering significant speedups over na\"ive PyTorch implementations.
- **Scientific Model Support**: Infrastructure for training and inference on models with unique data patterns (3D coordinate tensors, multi-sequence alignments, distance matrices) that differ from standard NLP workloads.
- **Domain-Specific Optimizations**: Memory layout transformations, fused kernels, and communication patterns optimized for scientific computing data flows.

---

## Source Code Organization

```
csrc/deepspeed4science/
    evoformer_attn/
        evoformer_attn_cuda.cpp          # C++/CUDA bridge for Evoformer attention
        evoformer_attn_backward.cu       # Backward-pass CUDA kernels
        evoformer_attn_forward.cu        # Forward-pass CUDA kernels
        evoformer_attn_binding.cpp       # Pybind11 bindings
        README.md                        # Kernel documentation

op_builder/
    evoformer_attn.py                    # Op builder for compiling Evoformer attention
```

---

## Evoformer Attention

### Background

The Evoformer attention mechanism is the core innovation in AlphaFold2's architecture. Unlike standard multi-head self-attention used in NLP transformers, Evoformer attention operates on a pair of representations:

1. **MSA Representation** (N_seq x N_res x d_msa): Encodes the multiple sequence alignment of the protein, where N_seq is the number of sequences in the MSA and N_res is the number of residues (amino acids).
2. **Pair Representation** (N_res x N_res x d_pair): Encodes pairwise relationships between residues, including geometric and co-evolutionary information.

The attention mechanism includes several specialized variants:
- **MSA row-wise attention with pair bias**: Attends across positions within each MSA sequence, with pair representation providing a bias term.
- **MSA column-wise attention**: Attends across sequences at each position, aggregating evolutionary information.
- **Triangular attention**: Operates on the pair representation using triangular multiplicative updates and attention over edges in the graph.

These attention patterns have unique computational properties:
- Variable sequence lengths in MSA inputs
- Asymmetric attention patterns (row-wise vs column-wise)
- Pair bias injection requiring custom softmax modifications
- Triangular masking for graph-structured data

### Custom CUDA Kernels

DeepSpeed4Science provides highly optimized CUDA kernels for Evoformer attention that deliver substantial speedups over standard PyTorch attention implementations.

#### Forward Pass Kernels

The forward pass kernels implement the following operations:

```
# MSA Row-wise Gated Self-Attention with Pair Bias
# For each sequence s in [0, N_seq):
#   Q = W_q * msa_repr[s]              # (N_res, d_head)
#   K = W_k * msa_repr[s]              # (N_res, d_head)
#   V = W_v * msa_repr[s]              # (N_res, d_head)
#   B = W_b * pair_repr                # (N_res, N_res)  -- pair bias
#   attn_logits = Q @ K^T / sqrt(d_head) + B
#   attn_weights = softmax(attn_logits)
#   output = gate * (attn_weights @ V)

# MSA Column-wise Attention
# For each position i in [0, N_res):
#   Q = W_q * msa_repr[:, i, :]        # (N_seq, d_head)
#   K = W_k * msa_repr[:, i, :]        # (N_seq, d_head)
#   V = W_v * msa_repr[:, i, :]        # (N_seq, d_head)
#   attn_logits = Q @ K^T / sqrt(d_head)
#   attn_weights = softmax(attn_logits)
#   output = attn_weights @ V

# Triangular Multiplicative Update (outgoing)
# For pair representation:
#   a = W_a * pair_repr (left side)
#   b = W_b * pair_repr (right side)
#   output = a @ b^T (with gating)
```

#### Backward Pass Kernels

The backward pass kernels compute gradients through all the specialized attention variants, including:
- Gradient computation through pair-bias injection
- Gradient computation through gating mechanisms
- Efficient gradient accumulation for triangular operations
- Fused backward passes reducing memory round-trips

#### Kernel Features

| Feature | Description |
|---------|-------------|
| **Fused QKV projection** | Combines query, key, value linear projections into a single kernel launch |
| **Pair bias fusion** | Fuses pair representation bias computation into the attention score kernel |
| **Gating fusion** | Fuses the gating operation into the attention output projection |
| **Memory-efficient attention** | Uses tiling strategies to avoid materializing full N_res x N_res attention matrices |
| **Mixed precision support** | Supports FP16 and BF16 computation with FP32 accumulation |
| **Custom masking** | Handles triangular masks, row/column masks, and attention bias patterns |

### Op Builder (evoformer_attn.py)

The `evoformer_attn.py` op builder compiles the Evoformer attention CUDA kernels and makes them available as a PyTorch extension.

```python
# op_builder/evoformer_attn.py

from deepspeed.ops.op_builder import OpBuilder

class EvoformerAttnBuilder(OpBuilder):
    """Builder for the Evoformer attention CUDA kernels."""

    BUILD_VAR = "DS_BUILD_EVOFORMER_ATTN"
    NAME = "deepspeed_evoformer_attn"

    def __init__(self, name="deepspeed_evoformer_attn"):
        super().__init__(name=name)

    def absolute_name(self):
        return "deepspeed.ops.evoformer_attn"

    def sources(self):
        return [
            "csrc/deepspeed4science/evoformer_attn/evoformer_attn_cuda.cpp",
            "csrc/deepspeed4science/evoformer_attn/evoformer_attn_forward.cu",
            "csrc/deepspeed4science/evoformer_attn/evoformer_attn_backward.cu",
            "csrc/deepspeed4science/evoformer_attn/evoformer_attn_binding.cpp",
        ]

    def extra_ldflags(self):
        return ['-lcurand']

    def cxx_args(self):
        return ['-O3', '-std=c++17', '-DVERSION_GE_1_1']

    def nvcc_args(self):
        args = [
            '-O3',
            '-use_fast_math',
            '-std=c++17',
            '-DVERSION_GE_1_1',
            '--generate-line-info',
        ]
        return args
```

#### Building the Kernels

```bash
# Build at install time
DS_BUILD_EVOFORMER_ATTN=1 pip install deepspeed --global-option="build_ext"

# Build at runtime (JIT compilation)
import deepspeed
from deepspeed.ops.op_builder import EvoformerAttnBuilder

builder = EvoformerAttnBuilder()
evoformer_module = builder.load()
```

---

## Using Evoformer Attention

### Basic Usage

```python
import torch
import deepspeed
from deepspeed.ops.op_builder import EvoformerAttnBuilder

# Build and load the kernel
builder = EvoformerAttnBuilder()
evoformer = builder.load()

# Input tensors
# msa_repr: (batch, N_seq, N_res, d_msa)
# pair_repr: (batch, N_res, N_res, d_pair)
msa_repr = torch.randn(2, 256, 384, 128, device='cuda', dtype=torch.float16)
pair_repr = torch.randn(2, 384, 384, 64, device='cuda', dtype=torch.float16)

# MSA row-wise attention with pair bias
# q_proj, k_proj, v_proj: weight matrices for Q, K, V projections
# pair_bias_proj: weight matrix for pair bias
# gating: whether to apply gating
output = evoformer.evoformer_attn_forward(
    msa_repr,
    pair_repr,
    q_weight,
    k_weight,
    v_weight,
    pair_bias_weight,
    output_weight,
    gating_weight,
    num_heads=8,
    head_dim=16,
    is_msa_row=True,  # True for row-wise, False for column-wise
)
```

### Integration with AlphaFold-like Models

```python
import torch
import torch.nn as nn
import deepspeed
from deepspeed.ops.op_builder import EvoformerAttnBuilder

class MSARowAttentionWithPairBias(nn.Module):
    """MSA row-wise gated self-attention with pair bias.

    This is one of the core attention blocks in the Evoformer stack.
    It attends across residues within each MSA sequence, using the
    pair representation to bias the attention scores.
    """

    def __init__(self, d_msa, d_pair, num_heads, dropout=0.0):
        super().__init__()
        self.d_msa = d_msa
        self.d_pair = d_pair
        self.num_heads = num_heads
        self.head_dim = d_msa // num_heads

        # Q, K, V projections for MSA representation
        self.q_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.k_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.v_proj = nn.Linear(d_msa, d_msa, bias=False)

        # Pair bias projection
        self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

        # Output projection with gating
        self.output_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.gating_proj = nn.Linear(d_msa, d_msa, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Load optimized kernel
        builder = EvoformerAttnBuilder()
        self.evoformer_kernel = builder.load()

    def forward(self, msa_repr, pair_repr):
        """
        Args:
            msa_repr: (batch, N_seq, N_res, d_msa)
            pair_repr: (batch, N_res, N_res, d_pair)

        Returns:
            Updated msa_repr: (batch, N_seq, N_res, d_msa)
        """
        batch_size, n_seq, n_res, _ = msa_repr.shape

        # Use optimized kernel
        output = self.evoformer_kernel.evoformer_attn_forward(
            msa_repr,
            pair_repr,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.pair_bias_proj.weight,
            self.output_proj.weight,
            self.gating_proj.weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            is_msa_row=True,
        )

        return msa_repr + self.dropout(output)


class MSAColumnAttention(nn.Module):
    """MSA column-wise attention.

    Attends across sequences at each position, aggregating
    evolutionary information from the multiple sequence alignment.
    """

    def __init__(self, d_msa, num_heads, dropout=0.0):
        super().__init__()
        self.d_msa = d_msa
        self.num_heads = num_heads
        self.head_dim = d_msa // num_heads

        self.q_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.k_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.v_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.output_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.gating_proj = nn.Linear(d_msa, d_msa, bias=False)
        self.dropout = nn.Dropout(dropout)

        builder = EvoformerAttnBuilder()
        self.evoformer_kernel = builder.load()

    def forward(self, msa_repr):
        """
        Args:
            msa_repr: (batch, N_seq, N_res, d_msa)

        Returns:
            Updated msa_repr: (batch, N_seq, N_res, d_msa)
        """
        output = self.evoformer_kernel.evoformer_attn_forward(
            msa_repr,
            None,  # No pair representation for column attention
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            None,  # No pair bias weight
            self.output_proj.weight,
            self.gating_proj.weight,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            is_msa_row=False,  # Column-wise attention
        )

        return msa_repr + self.dropout(output)
```

### Complete Evoformer Stack

```python
class EvoformerBlock(nn.Module):
    """A single Evoformer block containing MSA and pair stack updates."""

    def __init__(self, d_msa, d_pair, num_heads_msa=8, num_heads_pair=4):
        super().__init__()
        # MSA stack
        self.msa_row_attn = MSARowAttentionWithPairBias(d_msa, d_pair, num_heads_msa)
        self.msa_col_attn = MSAColumnAttention(d_msa, num_heads_msa)
        self.msa_transition = nn.Sequential(
            nn.LayerNorm(d_msa),
            nn.Linear(d_msa, d_msa * 4),
            nn.GELU(),
            nn.Linear(d_msa * 4, d_msa),
        )

        # Pair stack
        self.outer_product_mean = OuterProductMean(d_msa, d_pair)
        self.tri_mul_out = TriangleMultiplicationOutgoing(d_pair)
        self.tri_mul_in = TriangleMultiplicationIncoming(d_pair)
        self.tri_attn_start = TriangleAttentionStartingNode(d_pair, num_heads_pair)
        self.tri_attn_end = TriangleAttentionEndingNode(d_pair, num_heads_pair)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair * 4),
            nn.GELU(),
            nn.Linear(d_pair * 4, d_pair),
        )

    def forward(self, msa_repr, pair_repr):
        # MSA updates
        msa_repr = self.msa_row_attn(msa_repr, pair_repr)
        msa_repr = self.msa_col_attn(msa_repr)
        msa_repr = msa_repr + self.msa_transition(msa_repr)

        # Pair updates
        pair_repr = pair_repr + self.outer_product_mean(msa_repr)
        pair_repr = pair_repr + self.tri_mul_out(pair_repr)
        pair_repr = pair_repr + self.tri_mul_in(pair_repr)
        pair_repr = pair_repr + self.tri_attn_start(pair_repr)
        pair_repr = pair_repr + self.tri_attn_end(pair_repr)
        pair_repr = pair_repr + self.pair_transition(pair_repr)

        return msa_repr, pair_repr


class EvoformerStack(nn.Module):
    """Full Evoformer stack: N blocks of MSA + pair updates."""

    def __init__(self, d_msa, d_pair, num_blocks=48, num_heads_msa=8, num_heads_pair=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            EvoformerBlock(d_msa, d_pair, num_heads_msa, num_heads_pair)
            for _ in range(num_blocks)
        ])

    def forward(self, msa_repr, pair_repr):
        for block in self.blocks:
            msa_repr, pair_repr = block(msa_repr, pair_repr)
        return msa_repr, pair_repr
```

---

## Training Scientific Models with DeepSpeed

### DeepSpeed Configuration for Protein Structure Prediction

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 12,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "allgather_partitions": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "contiguous_gradients": true
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}
```

### Launch Script

```bash
#!/bin/bash

# Train an AlphaFold-like model with DeepSpeed4Science kernels
NUM_GPUS=8
deepspeed --num_gpus=$NUM_GPUS \
    train_evoformer.py \
    --deepspeed ds_config.json \
    --model-config config_evoformer.json \
    --data-dir /path/to/pdb_data \
    --num-blocks 48 \
    --d-msa 256 \
    --d-pair 128 \
    --num-heads-msa 8 \
    --num-heads-pair 4 \
    --max-seqs 256 \
    --max-residues 512 \
    --learning-rate 1e-3 \
    --epochs 100 \
    --use-evoformer-kernels
```

### Training Script Skeleton

```python
import argparse
import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import AlphaFoldLikeModel  # Your model definition
from dataset import ProteinStructureDataset  # Your data pipeline

def add_args():
    parser = argparse.ArgumentParser(description='DeepSpeed4Science Training')
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--model-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--num-blocks', type=int, default=48)
    parser.add_argument('--d-msa', type=int, default=256)
    parser.add_argument('--d-pair', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use-evoformer-kernels', action='store_true')
    return parser.parse_args()

def main():
    args = add_args()

    # Initialize dataset
    train_dataset = ProteinStructureDataset(
        data_dir=args.data_dir,
        max_seqs=256,
        max_residues=512,
    )

    # Build model
    model = AlphaFoldLikeModel(
        num_blocks=args.num_blocks,
        d_msa=args.d_msa,
        d_pair=args.d_pair,
        use_optimized_kernels=args.use_evoformer_kernels,
    )

    # Initialize DeepSpeed engine
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
    )

    # Training loop
    for epoch in range(args.epochs):
        for batch in model_engine:
            # batch contains MSA, pair features, and ground truth structure
            msa_repr = batch['msa'].cuda()
            pair_repr = batch['pair'].cuda()
            gt_coords = batch['coordinates'].cuda()

            # Forward pass
            predicted_coords, auxiliary_losses = model_engine(msa_repr, pair_repr)

            # Loss computation
            loss = compute_structure_loss(predicted_coords, gt_coords)
            loss = loss + sum(auxiliary_losses)

            # Backward pass
            model_engine.backward(loss)

            # Optimizer step
            model_engine.step()

        # Save checkpoint
        model_engine.save_checkpoint(f'checkpoints/epoch_{epoch}')

if __name__ == '__main__':
    main()
```

---

## Scientific Use Cases

### Protein Structure Prediction

The primary use case for DeepSpeed4Science is protein structure prediction, inspired by AlphaFold2 and related models. The Evoformer attention kernels accelerate the core attention computations that dominate runtime in these models.

**Typical problem sizes:**
- MSA depth (N_seq): 128 -- 2048 sequences
- Protein length (N_res): 100 -- 1000+ residues
- Model dimension (d_msa): 64 -- 256
- Pair dimension (d_pair): 32 -- 128

**Performance characteristics:**
- MSA row attention: O(N_seq * N_res^2 * d) per block
- MSA column attention: O(N_res * N_seq^2 * d) per block
- Triangular attention: O(N_res^2 * d) per block
- 48 blocks with multiple attention ops per block leads to massive compute

### Genomics

DeepSpeed4Science applies to genomic sequence modeling tasks:
- **DNA sequence modeling**: Transformer models over DNA sequences (e.g., Enformer, Nucleotide Transformer) with long context lengths
- **Variant effect prediction**: Predicting the impact of genetic variants using attention-based models
- **Gene expression prediction**: Modeling gene regulation from sequence

**Relevant DeepSpeed features:**
- Sequence parallelism for handling long DNA sequences (up to 200K+ base pairs)
- ZeRO Stage 3 for large genomic models
- Activation checkpointing for memory-efficient training

### Drug Discovery

Molecular property prediction and drug design:
- **Molecular graph transformers**: Attention over molecular graphs
- **Protein-ligand binding prediction**: Joint modeling of protein and ligand representations
- **Generative drug design**: Generating novel molecular structures with desired properties

### Molecular Dynamics

Accelerating molecular dynamics simulations with learned potentials:
- **Neural network potentials**: Training ML models to predict interatomic forces
- **Coarse-grained simulations**: Learning reduced representations of molecular systems
- **Enhanced sampling**: Using ML to accelerate rare event sampling

---

## DeepSpeed4Science Publications

The DeepSpeed4Science initiative is supported by several key publications:

1. **"DeepSpeed4Science: Enabling Large-Scale Scientific Discovery through DeepSpeed Innovations"** -- The foundational paper describing the initiative's goals, architecture, and initial results across multiple scientific domains.

2. **"Evoformer Attention Optimization for Protein Structure Prediction"** -- Detailed description of the custom CUDA kernels for Evoformer attention, including performance analysis and comparison with baseline implementations.

3. **"Scaling Laws for Scientific Foundation Models"** -- Investigation of scaling behavior in scientific models, including protein structure prediction and genomic sequence models.

4. **"Efficient Training of AlphaFold-like Models with ZeRO and Pipeline Parallelism"** -- Techniques for distributing AlphaFold-like training across hundreds of GPUs using DeepSpeed's parallelism strategies.

---

## Performance Benchmarks

### Evoformer Attention Kernel Performance

Comparison of DeepSpeed4Science Evoformer kernels vs. PyTorch baseline:

| Operation | Input Size | PyTorch (ms) | DS4Science (ms) | Speedup |
|-----------|-----------|--------------|-----------------|---------|
| MSA Row Attn | B=2, S=256, R=384, D=128 | 45.2 | 12.8 | 3.5x |
| MSA Col Attn | B=2, S=256, R=384, D=128 | 38.7 | 14.1 | 2.7x |
| Tri. Attn (start) | B=2, R=384, D=64 | 52.1 | 18.3 | 2.8x |
| Tri. Attn (end) | B=2, R=384, D=64 | 53.8 | 19.1 | 2.8x |
| Full Evoformer Block | B=2, S=256, R=384 | 380 | 145 | 2.6x |

*Benchmarks run on NVIDIA A100 80GB GPUs with FP16 precision.*

### End-to-End Training Performance

Training an AlphaFold-like model (48 Evoformer blocks, 256-dim MSA, 128-dim pair):

| Configuration | GPUs | Throughput (samples/hr) | Time per Epoch |
|---------------|------|------------------------|----------------|
| Baseline PyTorch | 8x A100 | 24.5 | 8.2 hrs |
| DeepSpeed ZeRO-2 | 8x A100 | 38.1 | 5.3 hrs |
| DS ZeRO-2 + Kernels | 8x A100 | 62.4 | 3.2 hrs |
| DS ZeRO-3 + Kernels | 32x A100 | 198.7 | 1.0 hrs |

---

## Advanced Configuration

### Multi-Node Training for Large Proteins

For proteins with >1000 residues, use ZeRO Stage 3 with activation checkpointing:

```json
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 48,
        "synchronize_checkpoint_boundary": true
    },
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": true
}
```

### Pipeline Parallelism for Evoformer

When each Evoformer block is large enough, pipeline parallelism can be applied across blocks:

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999]
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 1
    },
    "pipeline": {
        "enabled": true,
        "parallel_size": 4,
        "micro_batches": 8,
        "activation_checkpoint_interval": 6
    }
}
```

---

## Troubleshooting

### Common Issues

1. **Kernel compilation failure**: Ensure CUDA toolkit version is compatible (11.6+). Set `DS_BUILD_EVOFORMER_ATTN=1` during installation.

2. **Out of memory with large MSA**: Reduce `max_seqs` or enable ZeRO Stage 3 with activation checkpointing. Column attention scales as O(N_seq^2), so reducing MSA depth is effective.

3. **NaN losses**: Evoformer attention uses FP16 with large attention matrices. Increase `initial_scale_power` in fp16 config, or switch to BF16 if hardware supports it.

4. **Slow data loading**: Protein structure data (MSA, templates, distances) can be large. Use `num_workers > 0` in DataLoader and consider pre-processing data into memory-mapped formats.

### Environment Variables

```bash
# Enable Evoformer kernel compilation
export DS_BUILD_EVOFORMER_ATTN=1

# Debug kernel compilation
export DS_DEBUG=1

# Disable kernel (fall back to PyTorch)
export DS_DISABLE_EVOFORMER=1
```

---

## API Reference

### evoformer_attn_forward

```python
def evoformer_attn_forward(
    msa_repr: torch.Tensor,          # (B, N_seq, N_res, d_msa)
    pair_repr: Optional[torch.Tensor], # (B, N_res, N_res, d_pair) or None
    q_weight: torch.Tensor,           # (d_msa, d_msa)
    k_weight: torch.Tensor,           # (d_msa, d_msa)
    v_weight: torch.Tensor,           # (d_msa, d_msa)
    pair_bias_weight: Optional[torch.Tensor], # (num_heads, d_pair) or None
    output_weight: torch.Tensor,      # (d_msa, d_msa)
    gating_weight: torch.Tensor,      # (d_msa, d_msa)
    num_heads: int,
    head_dim: int,
    is_msa_row: bool,                 # True: row-wise, False: column-wise
) -> torch.Tensor:                    # (B, N_seq, N_res, d_msa)
    """Forward pass for Evoformer attention."""
    ...
```

### evoformer_attn_backward

```python
def evoformer_attn_backward(
    grad_output: torch.Tensor,        # (B, N_seq, N_res, d_msa)
    saved_tensors: Tuple[torch.Tensor, ...],
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    pair_bias_weight: Optional[torch.Tensor],
    output_weight: torch.Tensor,
    gating_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    is_msa_row: bool,
) -> Tuple[torch.Tensor, ...]:
    """Backward pass for Evoformer attention.

    Returns gradients for all input tensors and weight matrices.
    """
    ...
```

---

## Summary

DeepSpeed4Science extends DeepSpeed's high-performance training infrastructure to scientific computing workloads. The centerpiece is the Evoformer attention kernel suite, which provides optimized CUDA implementations of the specialized attention mechanisms used in AlphaFold2-style protein structure prediction. By combining these domain-specific kernels with DeepSpeed's general-purpose optimizations (ZeRO, mixed precision, pipeline parallelism), researchers can train large scientific models efficiently across multiple GPUs and nodes, enabling advances in protein structure prediction, genomics, drug discovery, and molecular dynamics.
