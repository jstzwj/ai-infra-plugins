# Chapter 37: Testing Framework

## Source Files
- `sources/Megatron-LM/tests/` - Test suite
- `sources/Megatron-LM/.github/workflows/` - CI/CD configuration
- `sources/Megatron-LM/.gitlab-ci.yml` - GitLab CI

## Overview

Megatron-LM includes a comprehensive test suite covering unit tests, integration tests, and end-to-end training validation. Tests ensure correctness of parallelism strategies, model implementations, and training pipelines.

## Test Structure

```
tests/
├── unit_tests/
│   ├── transformer/
│   │   ├── test_attention.py
│   │   ├── test_mlp.py
│   │   ├── test_transformer_layer.py
│   │   ├── test_transformer_block.py
│   │   └── test_transformer_config.py
│   ├── tensor_parallel/
│   │   ├── test_column_parallel.py
│   │   ├── test_row_parallel.py
│   │   └── test_sequence_parallel.py
│   ├── pipeline_parallel/
│   │   └── test_pipeline_schedule.py
│   ├── models/
│   │   ├── test_gpt_model.py
│   │   ├── test_bert_model.py
│   │   └── test_t5_model.py
│   ├── moe/
│   │   ├── test_router.py
│   │   ├── test_token_dispatcher.py
│   │   └── test_experts.py
│   ├── distributed/
│   │   ├── test_ddp.py
│   │   └── test_fsdp.py
│   ├── optimizer/
│   │   └── test_optimizer.py
│   ├── quantization/
│   │   ├── test_fp8.py
│   │   └── test_fp4.py
│   └── tokenizers/
│       └── test_tokenizers.py
├── integration_tests/
│   ├── test_training_gpt.py
│   ├── test_training_bert.py
│   ├── test_checkpointing.py
│   └── test_inference.py
└── functional_tests/
    ├── test_parity.py
    └── test_convergence.py
```

## Running Tests

### Run All Unit Tests
```bash
# Using pytest
pytest tests/unit_tests/ -v

# Using torchrun for distributed tests
torchrun --nproc_per_node=4 pytest tests/unit_tests/tensor_parallel/ -v
```

### Run Specific Test Module
```bash
pytest tests/unit_tests/transformer/test_attention.py -v
```

### Run with Coverage
```bash
pytest tests/unit_tests/ --cov=megatron.core --cov-report=html
```

### Run Distributed Tests
```bash
# TP tests require multiple GPUs
torchrun --nproc_per_node=2 pytest tests/unit_tests/tensor_parallel/ -v

# PP tests
torchrun --nproc_per_node=4 pytest tests/unit_tests/pipeline_parallel/ -v
```

## Writing Tests

### Basic Unit Test
```python
import pytest
import torch
from megatron.core.transformer import TransformerConfig, TransformerLayer

class TestTransformerLayer:
    @pytest.fixture
    def config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            seq_length=128,
            bf16=True,
        )

    def test_forward(self, config):
        layer = TransformerLayer(config)
        x = torch.randn(2, 128, 256).cuda()
        output = layer(x)
        assert output.shape == (2, 128, 256)
```

### Distributed Test
```python
import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

class TestTensorParallel:
    @pytest.fixture(autouse=True)
    def init_parallel(self):
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        yield
        parallel_state.destroy_model_parallel()

    def test_column_parallel(self):
        from megatron.core.tensor_parallel import ColumnParallelLinear
        # Test implementation...
```

## CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  unit-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -e ".[training,dev]"
      - name: Run unit tests
        run: pytest tests/unit_tests/ -v
      - name: Run distributed tests
        run: torchrun --nproc_per_node=2 pytest tests/unit_tests/tensor_parallel/ -v
```

### Test Categories

| Category | Description | GPU Required |
|---|---|---|
| Unit Tests | Individual component testing | No (CPU) or Yes |
| Integration Tests | Multi-component testing | Yes (1+ GPUs) |
| Distributed Tests | Parallelism strategy testing | Yes (2+ GPUs) |
| Functional Tests | End-to-end training validation | Yes (4+ GPUs) |
| Convergence Tests | Training convergence verification | Yes (8+ GPUs) |

## Test Configuration

```bash
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "distributed: marks tests requiring multiple GPUs",
    "fp8: marks tests requiring FP8 hardware",
]
```

### Running Marked Tests
```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only distributed tests
pytest tests/ -v -m "distributed"

# Run only FP8 tests (requires Hopper+)
pytest tests/ -v -m "fp8"
```
