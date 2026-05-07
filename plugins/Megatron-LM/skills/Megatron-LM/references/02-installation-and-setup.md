# Chapter 02: Installation and Setup

## Source Files
- `sources/Megatron-LM/docs/get-started/install.md`
- `sources/Megatron-LM/docs/get-started/quickstart.md`
- `sources/Megatron-LM/docker/`
- `sources/Megatron-LM/setup.py`
- `sources/Megatron-LM/pyproject.toml`

## System Requirements

### Hardware
- **Minimum**: NVIDIA Volta (V100) or later
- **Recommended**: NVIDIA Turing (A100) or later
- **FP8 Support**: NVIDIA Hopper (H100), Ada, or Blackwell GPUs
- **NVLink**: Recommended for tensor/pipeline parallelism across GPUs

### Software
| Dependency | Minimum Version | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| PyTorch | 2.6.0 | Latest stable |
| CUDA Toolkit | 11.8 | Latest stable |
| cuDNN | 8.9 | Latest stable |
| NCCL | 2.18 | Latest stable |

## Installation Options

### Option A: PyPI Install (Recommended)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Basic install
uv pip install megatron-core

# With training dependencies (W&B, SentencePiece, HF Transformers)
uv pip install "megatron-core[training]"

# Full install with all extras (includes TransformerEngine)
uv pip install --group build
uv pip install --no-build-isolation "megatron-core[training,dev]"

# Lighter development extras (no TransformerEngine, no ModelOpt)
uv pip install --no-build-isolation "megatron-core[training,lts]"
```

**Build memory warning**: Building from source compiles CUDA kernels. Set `MAX_JOBS` to limit parallel compilation jobs:
```bash
MAX_JOBS=4 uv pip install --no-build-isolation "megatron-core[training,dev]"
```

**Build time**: Expect 20+ minutes for full build with all CUDA extensions.

### Option B: Install from Source

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Development install
uv pip install -e .

# With all dev dependencies
uv pip install --group build
uv pip install --no-build-isolation -e ".[training,dev]"
```

### Option C: NGC Container (Pre-built)

```bash
docker run --gpus all -it --rm \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:26.01-py3

# Inside container, install Megatron Core
pip install uv
uv pip install --no-build-isolation "megatron-core[training,dev]"
```

**Note**: Use the previous month's NGC container for best compatibility with current Megatron Core release.

## Optional Dependencies

| Package | Purpose | Install |
|---|---|---|
| TransformerEngine | FP8, fused attention, fused layers | `pip install transformer-engine[pytorch]` |
| FlashAttention | Faster attention kernels | `pip install flash-attn` |
| FlashInfer | MoE inference backend | `pip install flashinfer` |
| grouped_gemm | MoE grouped GEMM | `pip install grouped_gemm` |
| NVIDIA Apex | Fused kernels | Build from source with `--cpp_ext --cuda_ext` |
| TensorRT-LLM | Model export | `pip install tensorrt-llm` |
| Megatron Energon | Multimodal dataloader | `pip install megatron-energon` |
| Weights & Biases | Experiment tracking | `pip install wandb` |
| SentencePiece | Tokenizer | `pip install sentencepiece` |
| mamba-ssm | Mamba SSM layers | `pip install mamba-ssm` |

## First Training Run

### Minimal Example (2 GPUs)
```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### LLaMA-3 8B Training (8 GPUs)
```bash
./examples/llama/train_llama3_8b_h100_fp8.sh
```

### Custom Data Training

**Step 1: Prepare JSONL data**
```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
```

**Step 2: Preprocess data**
```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

**Step 3: Launch training**
```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 256 \
    --train-iters 100000 \
    --lr 1e-4 \
    --bf16 \
    --data-path processed_data_text_document \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --split 98,1,1
```

## Multi-Node Setup

### Using torchrun
```bash
# Node 0 (master)
torchrun --nproc_per_node=8 --nnodes=4 \
    --master_addr=MASTER_IP --master_port=6000 \
    pretrain_gpt.py [ARGS...]

# Nodes 1-3 (workers)
torchrun --nproc_per_node=8 --nnodes=4 \
    --master_addr=MASTER_IP --master_port=6000 \
    --node_rank=NODE_RANK \
    pretrain_gpt.py [ARGS...]
```

### Using SLURM
```bash
#!/bin/bash
#SBATCH --job-name=megatron-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=6000

srun torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    pretrain_gpt.py [ARGS...]
```

### Environment Variables
```bash
# NCCL configuration for multi-node
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_IB_GID_INDEX=3            # IB GID index
export NCCL_SOCKET_IFNAME=eth0        # Network interface
export NCCL_DEBUG=INFO                # Debug output
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For tensor parallelism performance
```

## Verification

```python
# Verify Megatron Core installation
import megatron.core
print(megatron.core.__version__)

# Verify TransformerEngine
import transformer_engine as te
print(te.__version__)

# Verify GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```
