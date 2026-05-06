# Integrations

## Overview

DeepSpeed integrates with a wide ecosystem of machine learning frameworks, libraries, and tools. This reference covers all major integration points, including training frameworks (HuggingFace Transformers/Accelerate, PyTorch Lightning, MosaicML/MosaicML Composer), model parallelism libraries (Megatron-LM), orchestration systems (Ray, Azure ML, torchrun), compilation (torch.compile / DeepCompile), and monitoring (Weights & Biases, TensorBoard).

---

## HuggingFace Transformers Integration

### Overview

DeepSpeed is a first-class backend in HuggingFace Transformers' `Trainer` class. The integration allows training any HuggingFace model with DeepSpeed's ZeRO optimization, offloading, mixed precision, and other features using only a configuration file.

### Basic Usage with Trainer

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",       # Path to DeepSpeed config
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

### Configuration File

Create a DeepSpeed config JSON and pass its path to `TrainingArguments(deepspeed=...)`:

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### Auto-Configuration Values

HuggingFace Transformers automatically fills in certain DeepSpeed config values when set to `"auto"`:

| Config Key | Auto-Resolved From |
|---|---|---|
| `train_batch_size` | `per_device_train_batch_size * num_processes * gradient_accumulation_steps` |
| `train_micro_batch_size_per_gpu` | `per_device_train_batch_size` |
| `gradient_accumulation_steps` | `TrainingArguments.gradient_accumulation_steps` |
| `gradient_clipping` | `TrainingArguments.max_grad_norm` |
| `bf16.enabled` | `TrainingArguments.bf16` |
| `fp16.enabled` | `TrainingArguments.fp16` |

### Launching

```bash
# Using HuggingFace Accelerate launcher
accelerate launch --use_deepspeed train.py --deepspeed ds_config.json

# Using DeepSpeed launcher
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json

# Using torchrun
torchrun --nproc_per_node=8 train.py --deepspeed ds_config.json
```

### DeepSpeed with HuggingFace Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Or use Accelerate config file
# accelerate config  # Interactive setup, select DeepSpeed
# accelerate launch train.py
```

The Accelerate integration handles:
- Automatic DeepSpeed engine wrapping
- Config file parsing
- Device placement and data loading
- Gradient synchronization
- Mixed precision

### Checkpoint Integration

```python
# Saving
trainer.save_model(output_dir="./checkpoint")
# DeepSpeed saves both the model weights and optimizer state

# Loading
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./checkpoint")
```

### ZeRO-3 with HuggingFace

For ZeRO-3 with HuggingFace models, some additional handling is needed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model in meta device for ZeRO-3
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
```

---

## PyTorch Lightning Integration

### Overview

DeepSpeed is integrated as a native strategy in PyTorch Lightning. The `DeepSpeedStrategy` class provides seamless access to all DeepSpeed features.

### Basic Usage

```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy

trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(
        stage=2,
        offload_optimizer=True,
        precision="bf16-mixed"
    ),
    accelerator="gpu",
    devices=8,
    max_epochs=3,
)

trainer.fit(model)
```

### DeepSpeedStrategy Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stage` | int | `2` | ZeRO optimization stage (0, 1, 2, or 3) |
| `config` | str or dict | `None` | Path to DeepSpeed config JSON or dict |
| `offload_optimizer` | bool | `False` | Enable CPU offloading for optimizer states |
| `offload_parameters` | bool | `False` | Enable CPU offloading for parameters |
| `remote_device` | str | `None` | Remote device for offloading (`"cpu"`, `"nvme"`) |
| `precision` | str | `"32-true"` | Training precision (`"16-mixed"`, `"bf16-mixed"`, `"32-true"`) |
| `logging_batch_size_per_gpu` | int | `None` | Batch size for logging |
| `parallel_devices` | list | `None` | List of GPU devices |
| `cluster_environment` | object | `None` | Cluster environment plugin |
| `loss_scale` | float | `0` | Loss scale for FP16 (0 = dynamic) |
| `initial_scale_power` | int | `16` | Initial scale power for dynamic scaling |
| `loss_scale_window` | int | `1000` | Window size for loss scale adjustment |
| `hysteresis` | int | `2` | Hysteresis factor for loss scaling |
| `min_loss_scale` | float | `1` | Minimum loss scale |
| `accumulate_grad_batches` | int | `1` | Gradient accumulation steps |
| `gradient_clipping` | float | `0` | Gradient clipping value |

### Custom Configuration

```python
strategy = DeepSpeedStrategy(
    config={
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            }
        },
        "bf16": {
            "enabled": true
        },
        "gradient_accumulation_steps": 4
    }
)
```

### LightningModule for DeepSpeed

```python
class MyModel(pl.LightningModule):
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=32)
```

DeepSpeed handles the optimizer wrapping, gradient scaling, and loss accumulation automatically.

---

## MosaicML / Composer Integration

### Overview

MosaicML Composer integrates DeepSpeed as a backend for distributed training, providing access to ZeRO optimization alongside Composer's own speedup methods.

### Basic Usage

```python
from composer import Trainer
from composer.optim import DecoupledAdamW

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    optimizers=DecoupledAdamW(model.parameters(), lr=1e-4),
    max_duration="1ep",
    device_train_microbatch_size=4,
    deepspeed_config={  # or path to JSON file
        "zero_optimization": {
            "stage": 2
        },
        "bf16": {
            "enabled": True
        }
    },
)

trainer.fit()
```

### Configuration

Composer accepts DeepSpeed config as a dict or JSON file path via the `deepspeed_config` parameter:

```python
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True,
        "initial_scale_power": 16
    }
}
```

### Launching

```bash
# Using Composer launcher
composer --world_size 8 --nnodes 1 train.py

# Or using DeepSpeed launcher
deepspeed --num_gpus=8 train.py
```

---

## Megatron-LM Integration

### Overview

DeepSpeed integrates with NVIDIA's Megatron-LM for training large transformer models with 3D parallelism (data, tensor, and pipeline parallelism). The combination provides DeepSpeed's memory optimization (ZeRO) with Megatron's efficient transformer implementations.

### Architecture

```
Megatron-LM (Model Architecture + Tensor/Pipeline Parallelism)
    |
DeepSpeed (ZeRO Memory Optimization + Offloading + Checkpointing)
    |
PyTorch (CUDA Kernels + Distributed Communication)
```

### Setup

1. Install Megatron-LM:
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

2. Install DeepSpeed with Megatron support:
```bash
pip install deepspeed[megatron]
```

### Configuration

DeepSpeed works alongside Megatron's own parallelism configuration:

```json
{
    "zero_optimization": {
        "stage": 1
    },
    "gradient_clipping": 1.0,
    "bf16": {
        "enabled": true
    }
}
```

Note: ZeRO Stage 2 and 3 have limited compatibility with Megatron's tensor parallelism. Stage 1 is the recommended default.

### Launching

```bash
deepspeed --num_gpus=8 --num_nodes=2 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 4 \
    --global-batch-size 256 \
    --deepspeed ds_config.json
```

### Key Considerations

- ZeRO Stage 1 works with all Megatron parallelism modes
- ZeRO Stage 2 requires careful configuration with pipeline parallelism
- ZeRO Stage 3 has limited support with Megatron-LM
- Use Megatron's `--deepspeed` flag to enable the integration
- Checkpoint format differs between standalone Megatron and Megatron+DeepSpeed

---

## Ray Integration

### Overview

DeepSpeed integrates with Ray for elastic training and hyperparameter tuning at scale. The integration uses Ray's actor-based distributed computing model.

### Ray Train Integration

```python
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    import deepspeed
    # Initialize DeepSpeed within the Ray worker
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config["ds_config"]
    )

    for epoch in range(config["epochs"]):
        for batch in dataloader:
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={
        "ds_config": {
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True}
        },
        "epochs": 10
    },
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
)

result = trainer.fit()
```

### Ray with DeepSpeed Launcher

```bash
# Using ray job submit
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"pip": ["deepspeed"]}' \
    -- python train.py
```

---

## Azure ML Integration

### Overview

DeepSpeed runs natively on Azure ML using the Azure ML PyTorch estimator or the Azure ML CLI v2.

### Using Azure ML CLI v2

```yaml
# azureml-job.yml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: >-
  deepspeed --num_gpus=$AZUREML_GPU_COUNT train.py
  --deepspeed ds_config.json
  --output_dir ${{ outputs.model }}
code: .
environment:
  image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04
  conda_file: environment.yml
distribution:
  type: pytorch
  process_count_per_node: 8
resources:
  cluster: gpu-cluster
  instance_type: Standard_NC24ads_A100_v4
  instance_count: 2
outputs:
  model:
    type: mlflow_model
```

```bash
az ml job create -f azureml-job.yml
```

### Using Azure ML SDK

```python
from azure.ai.ml import command
from azure.ai.ml import MLClient

job = command(
    command="deepspeed --num_gpus=$AZUREML_GPU_COUNT train.py --deepspeed ds_config.json",
    environment="azureml:deepspeed-env:1",
    distribution={"type": "pytorch", "process_count_per_node": 8},
    compute="gpu-cluster",
    instance_count=2,
)

ml_client.jobs.create_or_update(job)
```

### Docker Image

Microsoft provides pre-built DeepSpeed Docker images:
```bash
docker pull mcr.microsoft.com/azureml/deepspeed:latest
```

---

## torchrun / torch.distributed.run Integration

### Overview

DeepSpeed works with PyTorch's native launcher `torchrun` (also known as `torch.distributed.run`) as an alternative to the DeepSpeed launcher.

### Basic Usage

```bash
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    train.py --deepspeed ds_config.json
```

### Differences from DeepSpeed Launcher

| Feature | DeepSpeed Launcher | torchrun |
|---|---|---|
| Elastic training | No (unless using DeepSpeed Elastic) | Yes |
| RDMA support | Built-in | Manual NCCL config |
| Multi-node setup | `--hostfile` | `--rdzv_endpoint` |
| Affinity control | `--bind_cores_to_rank` | Manual `taskset` |
| Environment injection | DeepSpeed env vars | Standard torchrun env vars |

### torchrun Environment Variables

When using torchrun, DeepSpeed reads rank information from the standard PyTorch environment variables:

| Variable | Description |
|---|---|
| `RANK` | Global rank |
| `LOCAL_RANK` | Local rank within node |
| `WORLD_SIZE` | Total number of processes |
| `MASTER_ADDR` | Address of the master node |
| `MASTER_PORT` | Port of the master node |

### Multi-Node with torchrun

```bash
# On master node (rank 0)
torchrun --nproc_per_node=8 --nnodes=2 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py --deepspeed ds_config.json

# On worker node
torchrun --nproc_per_node=8 --nnodes=2 \
    --rdzv_id=job1 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train.py --deepspeed ds_config.json
```

### Code Requirements for torchrun

Your training script must call `deepspeed.initialize()` without relying on DeepSpeed's launcher to set up the distributed environment:

```python
import deepspeed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

# DeepSpeed automatically uses torch.distributed.init_process_group()
# if not already initialized
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
```

---

## torch.compile / DeepCompile Integration

### Overview

DeepSpeed supports PyTorch 2.x's `torch.compile()` for just-in-time compilation of model computation graphs. This can provide significant speedups by fusing operations and optimizing memory access patterns.

### Basic Usage

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Compile the model
model_engine.compile()
```

### Configuration

```json
{
    "compile": {
        "enabled": true,
        "backend": "inductor",
        "mode": "default"
    }
}
```

### `engine.compile()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | str | `"inductor"` | Compilation backend. Options: `"inductor"`, `"eager"`, `"aot_eager"`, `"cudagraphs"` |
| `mode` | str | `"default"` | Compilation mode. Options: `"default"`, `"reduce-overhead"`, `"max-autotune"` |
| `dynamic` | bool | `False` | Enable dynamic shape support |
| `fullgraph` | bool | `False` | Compile the entire model as a single graph |

### DeepCompile (Advanced)

DeepSpeed provides `DeepCompile` for tighter integration with ZeRO:

```python
from deepspeed.compile import DeepCompile

# Compile with ZeRO-aware optimizations
dc = DeepCompile(model, backend="inductor", mode="max-autotune")
compiled_model = dc.compile()
```

### Compatibility Notes

| Feature | torch.compile Compatible | Notes |
|---|---|---|
| ZeRO Stage 1 | Yes | Fully compatible |
| ZeRO Stage 2 | Partial | May require `mode="reduce-overhead"` |
| ZeRO Stage 3 | Partial | Graph breaks on parameter gathering |
| Pipeline Parallelism | No | Not compatible |
| Tensor Parallelism | Partial | Requires `fullgraph=False` |
| Mixed Precision | Yes | Works with FP16/BF16 |
| Gradient Checkpointing | Yes | Fully compatible |

### Troubleshooting torch.compile

```python
# Debug compilation
import torch
torch._logging.set_logs(dynamo=True)

# See compilation graphs
model_engine.compile(backend="eager")  # No compilation, for debugging

# Common fix: disable graph breaks
model_engine.compile(fullgraph=False, dynamic=True)
```

---

## Weights & Biases (W&B) Integration

### Overview

DeepSpeed supports Weights & Biases for experiment tracking, logging training metrics, and visualizing training progress.

### Setup

```bash
pip install wandb
wandb login
```

### Configuration

```python
import wandb

wandb.init(
    project="deepspeed-training",
    config={
        "model": "gpt2-large",
        "deepspeed_config": ds_config,
    }
)
```

### Logging from DeepSpeed

```python
# Log after each step
model_engine, optimizer, _, _ = deepspeed.initialize(...)

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

    if model_engine.is_gradient_accumulation_boundary():
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/step": model_engine.global_steps,
            "perf/throughput": model_engine.get_batch_size() / model_engine.tictoc.step_time_ms() * 1000,
        })
```

### DeepSpeed Auto-Logging

DeepSpeed automatically logs to W&B when the `WANDB_PROJECT` environment variable is set:

```bash
WANDB_PROJECT=my-project deepspeed --num_gpus=8 train.py
```

### ZeRO Metrics

```python
# Log ZeRO-specific metrics
if hasattr(model_engine, 'memory_status'):
    mem = model_engine.memory_status()
    wandb.log({
        "memory/gpu_allocated": mem.get("gpu_allocated", 0),
        "memory/gpu_reserved": mem.get("gpu_reserved", 0),
    })
```

---

## TensorBoard Integration

### Overview

DeepSpeed supports TensorBoard for training visualization. Metrics are logged using PyTorch's `SummaryWriter`.

### Setup

```bash
pip install tensorboard
```

### Basic Usage

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./runs/deepspeed_experiment")

for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

    if model_engine.is_gradient_accumulation_boundary():
        writer.add_scalar("train/loss", loss.item(), model_engine.global_steps)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], model_engine.global_steps)

writer.close()
```

### Launching TensorBoard

```bash
tensorboard --logdir=./runs --port 6006
```

### DeepSpeed TensorBoard Logging

Enable automatic TensorBoard logging in the config:

```json
{
    "tensorboard": {
        "enabled": true,
        "output_path": "./logs",
        "job_name": "deepspeed_train"
    }
}
```

### Monitoring Loss Scale

```python
# Log loss scale for FP16 training
if hasattr(model_engine, 'get_loss_scale'):
    writer.add_scalar("fp16/loss_scale", model_engine.get_loss_scale(), step)
```

---

## Integration Compatibility Matrix

| Integration | ZeRO-1 | ZeRO-2 | ZeRO-3 | Pipeline | Tensor | Offload | torch.compile |
|---|---|---|---|---|---|---|---|
| HuggingFace Transformers | Yes | Yes | Yes | No | No | Yes | Partial |
| HuggingFace Accelerate | Yes | Yes | Yes | No | No | Yes | Partial |
| PyTorch Lightning | Yes | Yes | Yes | Yes | No | Yes | Partial |
| MosaicML Composer | Yes | Yes | Partial | No | No | Yes | No |
| Megatron-LM | Yes | Partial | Limited | Yes | Yes | No | No |
| Ray | Yes | Yes | Yes | No | No | Yes | No |
| Azure ML | Yes | Yes | Yes | No | No | Yes | No |
| torchrun | Yes | Yes | Yes | Yes | Yes | Yes | Partial |
| W&B | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| TensorBoard | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

---

## Common Integration Patterns

### Pattern 1: HuggingFace + ZeRO-3 + BF16

```python
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

# Use meta device initialization with ZeRO-3
import deepspeed
with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
    )

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed=ds_config,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

### Pattern 2: PyTorch Lightning + ZeRO-2 + CPU Offload

```python
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy

strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=True,
    precision="bf16-mixed",
)

trainer = Trainer(
    strategy=strategy,
    accelerator="gpu",
    devices=8,
    max_epochs=3,
)
```

### Pattern 3: torchrun + ZeRO-3 + NVMe Offload

```bash
torchrun --nproc_per_node=8 train.py \
    --deepspeed ds_config_zero3_nvme.json
```

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme/deepspeed_offload"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme/deepspeed_offload"
        }
    }
}
```
