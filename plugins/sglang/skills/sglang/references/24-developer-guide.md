# SGLang Developer Guide

This document provides a comprehensive guide for developers who want to contribute to SGLang,
understand its internals, extend its capabilities, and participate in its development workflow.
It covers environment setup, project structure, coding conventions, testing, benchmarking,
debugging, kernel development, model integration, release processes, and the contribution workflow.

---

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Code Style and Conventions](#code-style-and-conventions)
4. [Adding New Models](#adding-new-models)
5. [Custom Kernel Development](#custom-kernel-development)
6. [Testing](#testing)
7. [Benchmarking](#benchmarking)
8. [Debugging](#debugging)
9. [Release Process](#release-process)
10. [Contributing Guide](#contributing-guide)
11. [Architecture Deep Dive](#architecture-deep-dive)

---

## Development Environment Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04+ | Ubuntu 22.04+ |
| Python | 3.10 | 3.11+ |
| CUDA | 11.8 | 12.1+ |
| GPU | NVIDIA A10 (sm80) | NVIDIA H100/B200 (sm90+) |
| RAM | 32 GB | 128 GB+ |
| Disk | 50 GB free | 200 GB+ (model weights) |
| Docker | 20.10+ | 24.0+ (with NVIDIA Container Toolkit) |

### Python Environment Setup

#### Fork and Clone the Repository

New contributors do not have write permission to the official SGLang repository. Fork the
repository under your GitHub account first, then clone your fork locally:

```bash
git clone https://github.com/<your_user_name>/sglang.git
cd sglang
```

#### Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or using conda:

```bash
conda create -n sglang-dev python=3.11 -y
conda activate sglang-dev
```

#### Install SGLang from Source

Install the package in editable (development) mode so that code changes take effect immediately:

```bash
# Basic installation
pip install -e "python[dev]"

# With all extras (CUDA, multimodal, etc.)
pip install -e "python[all]"
```

The editable install means that modifications to the source code under `python/sglang/` are
reflected immediately without reinstalling.

#### Install Development Dependencies

```bash
pip install pre-commit build twine pytest pytest-cov
pip install -r test/requirements.txt  # if available
```

### Docker Development Environment

SGLang provides a `.devcontainer` folder in the repository root for automated containerized
development. This is the recommended approach for reproducible builds.

#### Option 1: VS Code Dev Container (Recommended)

1. Install the [VS Code Dev Container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Press `F1`, type and choose "Dev Container: Open Folder in Container".
3. Input the `sglang` local repo path and press Enter.

The first open will take longer due to Docker pull and build. Once successful, the VS Code
status bar shows you are in a dev container. Running `sglang.launch_server` in the VS Code
terminal will start the server with your local changes applied automatically.

#### Option 2: Manual Docker Container

```bash
# Basic dev container
docker run -itd --shm-size 32g --gpus all \
  --ipc=host --network=host --privileged \
  --name sglang_dev lmsysorg/sglang:dev /bin/zsh

# With HuggingFace cache mount (avoids re-downloading models)
docker run -itd --shm-size 32g --gpus all \
  -v ~/.cache/huggingface/:/root/.cache/huggingface \
  --ipc=host --network=host --privileged \
  --name sglang_dev lmsysorg/sglang:dev /bin/zsh

# With both HuggingFace cache and local SGLang repo
# Local code changes are automatically synced (editable install)
docker run -itd --shm-size 32g --gpus all \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface \
  -v $HOME/src/sglang:/sgl-workspace/sglang \
  --ipc=host --network=host --privileged \
  --name sglang_dev lmsysorg/sglang:dev /bin/zsh
```

**Note on RDMA**: `--network host` and `--privileged` are required for RDMA. If you do not
need RDMA, you can remove them, but keeping them does no harm. You may need to set
`NCCL_IB_GID_INDEX=3` when using RoCE.

### IDE Configuration

#### VS Code

1. **Remote Tunnel Setup** (for developing on a remote GPU host):

   ```bash
   # On the remote host
   wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/fabdb6a30b49f79a7aba0f2ad9df9b399473380f/vscode_cli_alpine_x64_cli.tar.gz
   tar xf vscode_cli_alpine_x64_cli.tar.gz
   ./code tunnel
   ```

   On your local machine, press `F1` in VS Code and choose "Remote Tunnels: Connect to Tunnel".

2. **Debug Configuration** (`launch.json`):

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python Debugger: launch_server",
         "type": "debugpy",
         "request": "launch",
         "module": "sglang.launch_server",
         "console": "integratedTerminal",
         "args": [
           "--model-path", "meta-llama/Llama-3.2-1B",
           "--host", "0.0.0.0",
           "--port", "30000",
           "--trust-remote-code"
         ],
         "justMyCode": false
       }
     ]
   }
   ```

   Press `F5` to start debugging. Breakpoints work even over remote SSH/tunnel + dev container.

3. **Recommended Extensions**:
   - Python (ms-python.python)
   - Pylance (ms-python.vscode-pylance)
   - Dev Containers (ms-vscode-remote.remote-containers)
   - clangd (for JIT kernel C++ development)

#### PyCharm

1. Open the `sglang` repository as a project.
2. Configure the Python interpreter to point to your virtual environment or conda environment.
3. Set the source roots to include `python/` and `sgl-kernel/python/`.
4. Create a run/debug configuration for `sglang.launch_server` with appropriate arguments.

---

## Project Structure

### Top-Level Directory Tree

```
sglang/
+-- python/                  # Main Python package
|   +-- sglang/              # The sglang package
|   |   +-- srt/             # SRT (SGLang Runtime) - core inference engine
|   |   +-- lang/            # Frontend language (SGLang programming language)
|   |   +-- jit_kernel/      # Just-in-time compiled CUDA/C++ kernels
|   |   +-- test/            # Test utilities and few-shot eval scripts
|   |   +-- benchmark/       # Benchmark utility modules
|   |   +-- multimodal_gen/  # Multimodal generation
|   |   +-- eval/            # Evaluation utilities
|   |   +-- cli/             # CLI entry points
|   |   +-- launch_server.py # Server launch entry point
|   |   +-- bench_serving.py # Online serving benchmark
|   |   +-- bench_one_batch.py       # Single-batch kernel-level benchmark
|   |   +-- bench_one_batch_server.py # Single-batch HTTP benchmark
|   |   +-- bench_offline_throughput.py # Offline throughput benchmark
|   |   +-- profiler.py      # Profiling utilities
|   |   +-- global_config.py # Global configuration
|   |   +-- utils.py         # Shared utilities
|   |   +-- version.py       # Version information
|   |   +-- __init__.py      # Package init (exports version)
|   +-- pyproject.toml       # Package metadata and dependencies
|   +-- upload_pypi.sh       # PyPI upload script
+-- sgl-kernel/              # Ahead-of-time compiled CUDA/C++ kernels
|   +-- csrc/                # CUDA/C++ kernel source files
|   +-- include/             # Kernel header files
|   +-- python/              # Python bindings for kernels
|   +-- cmake/               # CMake build configuration
|   +-- CMakeLists.txt       # CMake build entry point
|   +-- pyproject.toml       # Kernel package metadata
+-- sgl-model-gateway/       # Model gateway / router (Rust)
|   +-- src/                 # Rust source code
|   +-- bindings/            # Python bindings
|   +-- tests/               # Gateway tests
|   +-- Cargo.toml           # Rust package manifest
+-- test/                    # Integration and end-to-end tests
|   +-- registered/          # CI-registered tests
|   |   +-- unit/            # Unit tests (no server required)
|   |   +-- core/            # Core functionality tests
|   |   +-- eval/            # Accuracy evaluation tests
|   |   +-- ...              # Other test categories
|   +-- srt/                 # SRT-specific tests
|   +-- lm_eval_configs/     # LM evaluation configurations
|   +-- pytest.ini           # Pytest configuration
+-- benchmark/               # Benchmark scripts and datasets
|   +-- gsm8k/               # GSM8K benchmark
|   +-- hellaswag/           # HellaSwag benchmark
|   +-- deepseek_v3/         # DeepSeek V3 benchmarks
|   +-- ...                  # Other benchmark suites
+-- docs/                    # Documentation source
|   +-- get_started/         # Getting started guides
|   +-- developer_guide/     # Developer documentation
|   +-- advanced_features/   # Advanced feature docs
|   +-- references/          # Reference documentation
+-- docker/                  # Docker build files
+-- examples/                # Example scripts and notebooks
+-- scripts/                 # Build and CI helper scripts
+-- 3rdparty/                # Third-party dependencies
+-- proto/                   # Protocol buffer definitions
+-- rust/                    # Additional Rust components
+-- assets/                  # Static assets
+-- .devcontainer/           # Dev container configuration
+-- .pre-commit-config.yaml  # Pre-commit hook definitions
+-- .codespellrc             # Spell-check configuration
```

### python/sglang/srt/ -- Runtime Components

The `srt/` directory is the heart of SGLang. It contains the complete inference runtime:

```
srt/
+-- server_args.py           # Server argument definitions (all CLI flags)
+-- server_args_config_parser.py  # Config file parser
+-- environ.py               # Environment variable handling
+-- constants.py             # Global constants
+-- models/                  # Model implementations (one file per model architecture)
+-- managers/                # Process managers (Tokenizer, Scheduler, Detokenizer)
|   +-- tokenizer_manager.py      # Tokenizer process manager
|   +-- scheduler.py              # Core scheduler (batching, eviction, scheduling)
|   +-- detokenizer_manager.py    # Detokenizer process manager
|   +-- tp_worker.py             # Tensor parallel worker
|   +-- schedule_batch.py        # Batch data structures
|   +-- schedule_policy.py       # Scheduling policies (fcfs, lpm, etc.)
|   +-- io_struct.py             # ZMQ IPC message definitions
|   +-- communicator.py          # Inter-process communication
+-- layers/                  # Neural network layer implementations
|   +-- attention/                # Attention backends and kernels
|   +-- moe/                      # Mixture-of-experts implementations
|   +-- linear.py                 # Linear layer with TP
|   +-- activation.py             # Activation functions
|   +-- layernorm.py              # Layer normalization
|   +-- logits_processor.py       # Logits processing
|   +-- model_parallel.py         # Model parallelism utilities
+-- mem_cache/               # Memory management for KV cache
|   +-- radix_cache.py            # Radix tree for prefix caching
|   +-- memory_pool.py            # GPU memory pool
|   +-- allocator.py              # Memory allocator
|   +-- chunk_cache.py            # Chunked cache
|   +-- hicache_storage.py        # Hierarchical cache storage
+-- sampling/                # Token sampling logic
+-- tokenizer/               # Tokenizer implementations
+-- entrypoints/             # Server entry points (HTTP, gRPC)
+-- model_loader/            # Weight loading logic
+-- model_executor/          # Model execution coordination
+-- lora/                    # LoRA adapter management
+-- speculative/             # Speculative decoding implementations
+-- constrained/             # Constrained/structured output
+-- distributed/             # Distributed communication
+-- configs/                 # Model configuration handling
+-- platforms/               # Hardware platform abstractions
+-- multimodal/              # Multimodal input processing
+-- layers/flashinfer_comm_fusion/  # FlashInfer communication fusion
+-- compilation/             # torch.compile integration
+-- checkpoint_engine/       # Checkpoint management
+-- disaggregation/          # Prefill-decode disaggregation
+-- observability/           # Metrics and tracing
+-- parser/                  # Output parsing (tool calls, reasoning)
+-- session/                 # Session management
+-- utils/                   # Internal utilities
+-- debug_utils/             # Debug utilities
+-- hardware_backend/        # Hardware abstraction layer
```

### python/sglang/lang/ -- Frontend Language

The `lang/` directory implements the SGLang frontend programming language for structured generation:

```
lang/
+-- api.py          # Public API for SGLang language
+-- backend/        # Backend implementations
+-- interpreter.py  # SGLang program interpreter
+-- tracer.py       # Program tracing
+-- ir.py           # Intermediate representation
+-- choices.py      # Choice point handling
+-- chat_template.py # Chat template processing
```

### sgl-kernel/ -- CUDA Kernels

The `sgl-kernel/` directory contains ahead-of-time (AOT) compiled kernels distributed as a
separate Python package (`sglang-kernel` on PyPI):

```
sgl-kernel/
+-- csrc/              # CUDA/C++ source files
+-- include/           # Public header files
+-- python/            # Python bindings
+-- cmake/             # CMake modules
+-- CMakeLists.txt     # Build configuration
+-- pyproject.toml     # Package definition
+-- Makefile           # Build targets
+-- build.sh           # Build script
+-- tests/             # Kernel tests
+-- benchmark/         # Kernel benchmarks
```

### sgl-model-gateway/ -- Gateway/Router

The model gateway is a high-performance router written in Rust that handles load balancing,
data parallelism routing, and request distribution:

```
sgl-model-gateway/
+-- src/               # Rust source
+-- bindings/          # Python bindings
+-- tests/             # Tests
+-- benches/           # Rust benchmarks
+-- examples/          # Usage examples
+-- Cargo.toml         # Rust package manifest
+-- Makefile           # Build targets
```

### test/ -- Tests

```
test/
+-- registered/           # CI-registered tests
|   +-- unit/             # Unit tests (mirror srt/ structure)
|   |   +-- mem_cache/    # Memory cache tests
|   |   +-- sampling/     # Sampling tests
|   |   +-- ...           # Other module tests
|   +-- core/             # Core feature tests
|   +-- eval/             # Evaluation tests
|   +-- 1-gpu-models/     # Single-GPU model tests
|   +-- 4-gpu-models/     # Multi-GPU model tests
|   +-- attention/        # Attention backend tests
|   +-- backends/         # Backend tests
|   +-- hicache/          # Hierarchical cache tests
|   +-- ...               # Many more categories
+-- srt/                  # Legacy SRT tests
+-- lm_eval_configs/      # LM eval configurations
+-- manual/               # Manual test scripts
+-- pytest.ini            # Pytest configuration
+-- README.md             # Test documentation
```

### benchmark/ -- Benchmarks

```
benchmark/
+-- gsm8k/               # GSM8K math benchmark
+-- hellaswag/           # HellaSwag benchmark
+-- deepseek_v3/         # DeepSeek V3 specific
+-- bench_linear_attention/  # Attention benchmarks
+-- bench_rope/          # RoPE benchmarks
+-- benchmark_batch/     # Batch benchmarks
+-- blog_v0_2/           # Blog reproduction benchmarks
+-- ...                  # Other benchmark suites
```

---

## Code Style and Conventions

### Python Style Guide

SGLang enforces consistent code style through automated tooling. The key conventions are:

- **Formatter**: [Black](https://github.com/psf/black) (version 26.1.0) for Python, including
  Jupyter notebooks (`black-jupyter` hook).
- **Import Sorting**: [isort](https://pycqa.github.io/isort/) (version 7.0.0) with standard
  configuration.
- **Line Length**: Black default (88 characters).
- **Type Hints**: Encouraged but not strictly enforced. Use `TYPE_CHECKING` for forward
  references.

### Pre-commit Hooks

SGLang uses [pre-commit](https://pre-commit.com/) to enforce code quality. Install and run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

If pre-commit fails the first time, re-run it to ensure lint errors are fully resolved.
All checks must pass before creating a Pull Request.

The configured hooks (from `.pre-commit-config.yaml`) are:

| Hook | Purpose | Scope |
|------|---------|-------|
| `check-symlinks` | Detect broken symlinks | All files |
| `destroyed-symlinks` | Detect destroyed symlinks | All files |
| `trailing-whitespace` | Remove trailing whitespace | All files |
| `end-of-file-fixer` | Ensure newline at EOF | All files |
| `check-yaml` | Validate YAML syntax | YAML files |
| `check-toml` | Validate TOML syntax | TOML files |
| `check-ast` | Validate Python syntax | Python files |
| `check-added-large-files` | Prevent large file commits | All files |
| `check-merge-conflict` | Detect unresolved conflicts | All files |
| `detect-private-key` | Prevent private key commits | All files |
| `debug-statements` | Catch `pdb`/`breakpoint` left in code | Python files |
| `no-commit-to-branch` | Prevent direct commits to main | All files |
| `isort` | Sort Python imports | Python files |
| `ruff` (F401, F821) | Remove unused imports, undefined names | Python files |
| `black-jupyter` | Format Python and notebooks | Python/Jupyter files |
| `codespell` | Spell check | All files |
| `clang-format` | Format C++/CUDA code | C++/CUDA files |
| `nbstripout` | Strip notebook outputs (keep output) | Jupyter notebooks |
| `lychee` | Check documentation links | Markdown/RST (manual stage only) |

Custom local hooks:

| Hook | Purpose |
|------|---------|
| `check-chinese-characters` | Detect Chinese characters in multimodal_gen |
| `sort-ci-permissions` | Sort CI_PERMISSIONS.json |
| `check-workflow-job-names` | Validate CI workflow job names |
| `check-registered-tests` | Ensure registered tests have CI registry |
| `check-no-docs-changes` | Reject changes under legacy docs/ |

### Linting

**Ruff** is configured to check for unused imports (F401) and undefined names (F821):

```bash
# Via pre-commit
pre-commit run ruff --all-files

# Manually
ruff check --select=F401,F821 --fix python/sglang/
```

Excluded from ruff: `__init__.py` files, Jupyter notebooks, and gRPC-generated files.

**Codespell** catches common misspellings:

```bash
pre-commit run codespell --all-files
```

### Type Checking

While not enforced by CI, type checking with mypy or pyright is recommended for new code.
Use `TYPE_CHECKING` blocks for imports that would create circular dependencies:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Batch
```

### Import Conventions

- Use `isort` for consistent import ordering: stdlib, third-party, local.
- Avoid wildcard imports (`from module import *`).
- For JIT kernel imports, use:

  ```python
  from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
  ```

- gRPC-generated files (`*_pb2.py`, `*_pb2_grpc.py`) are excluded from formatting and import
  sorting.

### C++/CUDA Code Style

C++ and CUDA files are formatted with `clang-format` using a project-level `.clang-format`
configuration. The hook runs `clang-format --style=file --verbose` on all C++ and CUDA files.

For JIT kernel development, install `clangd` for IDE integration and run
`python -m sglang.jit_kernel` to generate a `.clangd` configuration file for code completion.

---

## Adding New Models

### Model Registration

SGLang uses a model registry pattern. Models are registered by creating a Python file in
`python/sglang/srt/models/`. The file name should match the model architecture (e.g.,
`llama.py` for LLaMA models, `deepseek_v2.py` for DeepSeek V2).

The model auto-detection works by mapping HuggingFace model configuration class names to
the corresponding SGLang model implementation. The mapping is defined in the model loader.

### Model Class Implementation

A model implementation must follow this structure:

1. **Create the model file** at `python/sglang/srt/models/<model_name>.py`.

2. **Define the model class** that implements the forward pass:

   ```python
   import torch
   import torch.nn as nn
   from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
   from sglang.srt.layers.layernorm import RMSNorm
   from sglang.srt.layers.activation import SiluAndMul

   class MyNewModel(nn.Module):
       def __init__(self, config, quant_config=None):
           super().__init__()
           self.config = config
           # Initialize layers: embeddings, transformer blocks, lm_head
           self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
           self.layers = nn.ModuleList([
               MyDecoderLayer(config, quant_config) for _ in range(config.num_hidden_layers)
           ])
           self.norm = RMSNorm(config.hidden_size)
           self.lm_head = ColumnParallelLinear(
               config.hidden_size, config.vocab_size, bias=False
           )

       def forward(self, input_ids, positions, forward_batch):
           hidden = self.embed_tokens(input_ids)
           for layer in self.layers:
               hidden = layer(hidden, positions, forward_batch)
           hidden = self.norm(hidden)
           return self.lm_head(hidden)
   ```

3. **Implement the decoder layer**:

   ```python
   class MyDecoderLayer(nn.Module):
       def __init__(self, config, quant_config=None):
           super().__init__()
           self.self_attn = MyAttention(config, quant_config)
           self.mlp = MyMLP(config, quant_config)
           self.input_layernorm = RMSNorm(config.hidden_size)
           self.post_attention_layernorm = RMSNorm(config.hidden_size)

       def forward(self, hidden, positions, forward_batch):
           hidden = hidden + self.self_attn(
               self.input_layernorm(hidden), positions, forward_batch
           )
           hidden = hidden + self.mlp(
               self.post_attention_layernorm(hidden)
           )
           return hidden
   ```

4. **Register the entry point class** at the bottom of the file:

   ```python
   # This class is what the model registry looks up
   EntryClass = MyNewModel
   ```

5. **Handle model configuration** by mapping HuggingFace config fields to your model's
   constructor parameters.

### Weight Loading

SGLang supports multiple weight loading formats via the `--load-format` flag. The default
(`auto`) tries safetensors first, then falls back to PyTorch bin format.

For a new model, weight loading typically works automatically if:
- The model uses standard HuggingFace weight naming conventions.
- The `nn.Module` parameter names match the checkpoint names.

If custom weight loading is needed:

1. Create a weight loader in `python/sglang/srt/model_loader/`.
2. Map checkpoint weight names to model parameter names.
3. Handle quantization-specific weight formats (AWQ, GPTQ, FP8, etc.).

### Testing New Models

#### Quick Smoke Test

```bash
# Launch with the new model
python3 -m sglang.launch_server --model-path <your-model> --trust-remote-code

# Quick accuracy check (GSM8K)
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

#### Comprehensive Accuracy Evaluation

Run the full suite of accuracy benchmarks:

**MMLU**:

```bash
python -m sglang.test.run_eval \
  --eval-name mmlu \
  --port 30000 \
  --num-examples 1000 \
  --max-tokens 8192
```

**GSM8K**:

```bash
python -m sglang.test.few_shot_gsm8k \
  --host 127.0.0.1 --port 30000 \
  --num-questions 200 --num-shots 5
```

**HellaSwag**:

```bash
python benchmark/hellaswag/bench_sglang.py \
  --host 127.0.0.1 --port 30000 \
  --num-questions 200 --num-shots 20
```

**GPQA** (for advanced models):

```bash
python -m sglang.test.run_eval \
  --eval-name gpqa --port 30000 \
  --num-examples 198 --max-tokens 120000 --repeat 8
```

For reasoning models, add `--thinking-mode <mode>` (e.g., `qwen3`, `deepseek-v3`).

### Model Evaluation Guide

#### Reporting Results

When evaluating a new model, report:

1. **Metric Score**: Accuracy % (LLMs and VLMs); Latency (ms) and Throughput (tok/s).
2. **Environment Settings**: GPU type/count, SGLang commit hash.
3. **Launch Configuration**: Model path, TP size, special flags.
4. **Evaluation Parameters**: Number of shots, examples, max tokens.

#### Performance Benchmarking

**Latency-Sensitive** (single user):

```bash
python -m sglang.bench_serving \
  --backend sglang --host 0.0.0.0 --port 30000 \
  --dataset-name random --num-prompts 10 --max-concurrency 1
```

**Throughput-Sensitive** (high traffic):

```bash
python -m sglang.bench_serving \
  --backend sglang --host 0.0.0.0 --port 30000 \
  --dataset-name random --num-prompts 1000 --max-concurrency 100
```

#### VLM Evaluation

**MMMU**:

```bash
python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 64
```

For video-capable models, extend evaluation to include VideoMME, MVBench, and other
video benchmarks.

---

## Custom Kernel Development

SGLang provides two paths for kernel development:
- **JIT kernels** (`python/sglang/jit_kernel/`): Compiled at runtime.
- **AOT kernels** (`sgl-kernel/`): Compiled ahead of time, distributed as `sglang-kernel`.

### JIT Kernel Guide

JIT kernels are compiled at runtime using the [tvm-ffi](https://github.com/apache/tvm-ffi)
framework. They are located in `python/sglang/jit_kernel/`.

#### Environment Setup

Use `clangd` as the language server. For VS Code, install the clangd extension. Run:

```bash
python -m sglang.jit_kernel
```

This generates a `.clangd` configuration file in the current directory. Restart clangd to
enable code completion for all JIT kernel files.

#### Code Structure

```
jit_kernel/
+-- csrc/              # C++/CUDA source files
+-- include/           # Reusable C++ header files
+-- <kernel>.py        # Python interface for each kernel
+-- utils.py           # load_jit, cache_once, make_cpp_args utilities
+-- tests/             # Kernel tests
+-- __main__.py        # Generates .clangd config
```

#### Adding a New JIT Kernel (Step by Step)

This walks through adding an `add_constant` kernel that adds a constant to every tensor element.

**Step 1: Write the C++ Kernel**

Create `python/sglang/jit_kernel/csrc/add_constant.cuh`:

```cpp
#include <sgl_kernel/tensor.h>   // TensorMatcher, SymbolicSize
#include <sgl_kernel/utils.cuh>  // LaunchKernel
#include <sgl_kernel/utils.h>    // div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, size_t length) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    dst[idx] = src[idx] + kConstant;
  }
}

constexpr size_t kBlockSize = 256;

template <int32_t kConstant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  // 1. Validate input tensors
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  TensorMatcher({N})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(dst)
      .verify(src);

  // 2. Extract parameters
  const size_t num_elements = N.unwrap();
  const size_t grid_size = div_ceil(num_elements, kBlockSize);
  const DLDevice device = device_.unwrap();
  RuntimeCheck(num_elements > 0, "Only non-empty tensors supported, got ",
               num_elements);

  // 3. Launch kernel
  LaunchKernel(grid_size, kBlockSize, device)(
      add_constant_kernel<kConstant>,
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      num_elements);
}

}  // namespace
```

**Step 2: Create Python Interface**

Create `python/sglang/jit_kernel/add_constant.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)
    return load_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    if not src.is_cuda:
        raise RuntimeError("src must be a CUDA tensor")
    if src.dtype != torch.int32:
        raise RuntimeError(f"Unsupported dtype {src.dtype}. Supported: int32")
    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst
```

**Step 3: Use the Kernel**

```python
from sglang.jit_kernel.add_constant import add_constant

result = add_constant(torch.tensor([1, 2, 3], device="cuda"), 10)
```

#### C++ Utility Reference

JIT kernels provide a rich C++ header library in `python/sglang/jit_kernel/include/sgl_kernel/`:

**Core Utilities**:

| Header | Namespace | Purpose |
|--------|-----------|---------|
| `utils.h` | `host` | `RuntimeCheck`, `Panic`, `div_ceil`, `irange` |
| `utils.cuh` | `device`/`host` | Type aliases (`fp16_t`, `bf16_t`), `SGL_DEVICE` macro, PDL helpers, `LaunchKernel`, `RuntimeDeviceCheck` |
| `source_location.h` | global | Portable `std::source_location` wrapper |
| `runtime.cuh` | `host::runtime` | CUDA runtime queries: `get_blocks_per_sm`, `get_sm_count`, `get_cc_major` |

**Tensor Validation**:

| Header | Namespace | Purpose |
|--------|-----------|---------|
| `tensor.h` | `host` | `TensorMatcher`, `SymbolicSize`, `SymbolicDType`, `SymbolicDevice` |

**Math and Type System**:

| Header | Namespace | Purpose |
|--------|-----------|---------|
| `math.cuh` | `device::math` | `max`, `min`, `abs`, `sqrt`, `rsqrt`, `exp`, `sin`, `cos` |
| `type.cuh` | global/device | `dtype_trait<T>`, `packed_t<T>`, `device::cast<To>(from)` |

**Memory Access**:

| Header | Namespace | Purpose |
|--------|-----------|---------|
| `vec.cuh` | `device` | `AlignedVector<T, N>` - vectorized load/store up to 128-bit |
| `tile.cuh` | `device::tile` | `Memory<T>` - cooperative tiled memory I/O |

**Parallel Primitives**:

| Header | Namespace | Purpose |
|--------|-----------|---------|
| `warp.cuh` | `device::warp` | `reduce_sum`, `reduce_max` via `__shfl_xor_sync` |
| `cta.cuh` | `device::cta` | `reduce_max` across warps via shared memory |
| `atomic.cuh` | `device::atomic` | `max` - atomic float max (CUDA + ROCm fallback) |

**Kernel Templates**:

| Header | Namespace | Purpose |
|--------|-----------|---------|
| `impl/norm.cuh` | `host::norm`/`device::norm` | RMSNorm building blocks (warp & CTA paths) |

#### Key Python Utilities

- `load_jit(name, *args, cuda_files, cuda_wrappers)`: Loads and returns a compiled module.
- `cache_once`: Alternative to `functools.lru_cache` that is compatible with `torch.compile`.
- `make_cpp_args(*values)`: Formats template arguments for C++ kernel instantiation.

### sgl-kernel Development (AOT Kernels)

Since `sglang` and `sglang-kernel` are separate Python packages, updating a kernel requires
a multi-PR workflow:

**Step 1**: Submit a PR to update the sgl-kernel source code without using it in the sglang
Python package.

**Step 2**: Bump the version of the kernel package. Once merged, this triggers an automatic
release of the `sglang-kernel` wheel to PyPI.

**Step 3**: Update the `sglang-kernel` version in `sglang/python/pyproject.toml` and update
the caller code to use the new kernel.

If not urgent, you can wait for a new kernel version to be released (typically within one week).

### Triton Kernels

SGLang uses Triton kernels for certain operations (e.g., attention, sampling). Triton kernels
are Python-based and can be developed and tested without a separate compilation step.

Triton kernel files are typically located in `python/sglang/srt/layers/attention/` or
`python/sglang/jit_kernel/`.

### CUDA Kernel Development

For AOT CUDA kernels in `sgl-kernel/`:

1. Write kernel code in `sgl-kernel/csrc/`.
2. Add headers to `sgl-kernel/include/`.
3. Create Python bindings in `sgl-kernel/python/`.
4. Build with CMake: `cd sgl-kernel && bash build.sh`.
5. Test with the kernel test suite.

### Testing Kernels

JIT kernels have tests in `python/sglang/jit_kernel/tests/`. AOT kernel tests are in
`sgl-kernel/tests/`. Write unit tests that:

- Verify correctness against reference PyTorch implementations.
- Test edge cases (empty tensors, large tensors, different dtypes).
- Benchmark performance against alternative implementations.

---

## Testing

### Running Tests

#### Unit Tests (No Server Required)

Unit tests live under `test/registered/unit/`, organized to mirror `python/sglang/srt/`:

```bash
# All unit tests
pytest test/registered/unit/ -v

# One module
pytest test/registered/unit/mem_cache/ -v

# With coverage
pytest test/registered/unit/ --cov --cov-config=.coveragerc -v
```

The mapping between source and test files follows this convention:

```
srt/mem_cache/radix_cache.py   ->  unit/mem_cache/test_radix_cache.py
srt/sampling/sampling_params.py ->  unit/sampling/test_sampling_params.py
```

#### E2E Tests (Server Required)

E2E tests are in `test/registered/` under various subdirectories. These tests launch a server
and validate end-to-end behavior.

```bash
# Run specific test categories
pytest test/registered/core/ -v
pytest test/registered/eval/ -v
```

### Writing Tests

#### Unit Test Conventions

- Use Python's built-in `unittest` framework with `pytest` as the test runner.
- Place tests in the appropriate subdirectory under `test/registered/unit/`.
- Keep tests fast. If a single test file runs longer than 500 seconds, split it into
  multiple files (e.g., `test_eagle_infer_a.py`, `test_eagle_infer_b.py`).
- Reuse server launches across tests to avoid repeated startup overhead.

Example unit test:

```python
import unittest
from sglang.srt.mem_cache.radix_cache import TreeNode, RadixCache


class TestRadixCache(unittest.TestCase):
    def setUp(self):
        self.cache = RadixCache()

    def test_insert_and_match(self):
        # Test basic prefix matching
        tokens = [1, 2, 3, 4, 5]
        self.cache.insert(tokens)
        matched = self.cache.match_prefix(tokens[:3])
        self.assertEqual(len(matched), 3)
```

#### E2E Test Conventions

- Tests that require a server should be placed in `test/registered/` under the appropriate
  category.
- Document the GPU requirements and model used.
- Follow the test registration process for CI (see
  `test/registered/unit/README.md`).

### CI/CD Pipeline

#### Triggering CI

CI runs are gated by the "run-ci" label on PRs. Only authorized users listed in
`CI_PERMISSIONS.json` can add this label. Authorized users can use these commands:

- `/tag-run-ci-label`: Adds the "run-ci" label. Every future commit triggers CI.
- `/rerun-failed-ci`: Reruns failed or flaky tests from the most recent commit.
- `/tag-and-rerun-ci`: Performs both `/tag-run-ci-label` and `/rerun-failed-ci`.
- `/rerun-stage <stage-name>`: Reruns a specific test stage without waiting for dependencies.

PR authors can always use `/rerun-failed-ci` on their own PRs.

#### CI Rate Limits

Each CI workflow has a default cooldown period (120 minutes). Higher-priority PRs may preempt
running jobs. Users in `CI_PERMISSIONS.json` may have per-user cooldown intervals.

#### CI Test Stages

CI runs are organized into stages that run sequentially. Example stages include:
- Unit tests (backend)
- Single-GPU model tests
- Multi-GPU model tests (4-GPU, 8-GPU)
- Evaluation tests
- Specialized hardware tests (AMD, Ascend)

### Test Categories

| Category | Directory | Requires Server | GPU Required |
|----------|-----------|-----------------|--------------|
| Unit | `test/registered/unit/` | No | No (some tests may need GPU) |
| Core | `test/registered/core/` | Yes | Yes |
| Eval | `test/registered/eval/` | Yes | Yes |
| Model (1-GPU) | `test/registered/1-gpu-models/` | Yes | 1 GPU |
| Model (4-GPU) | `test/registered/4-gpu-models/` | Yes | 4 GPUs |
| Model (8-GPU) | `test/registered/8-gpu-models/` | Yes | 8 GPUs |
| Attention | `test/registered/attention/` | Yes | Yes |
| HiCache | `test/registered/hicache/` | Yes | Yes |
| AMD | `test/registered/amd/` | Yes | AMD GPU |
| Ascend | `test/registered/ascend/` | Yes | Ascend NPU |

---

## Benchmarking

### Overview of Benchmark Tools

SGLang provides four benchmark tools operating at different levels:

| Tool | HTTP Server | Scheduler | Use Case |
|------|-------------|-----------|----------|
| `bench_serving` | Yes (async HTTP client) | Yes (via server) | Realistic online serving benchmarks |
| `bench_one_batch_server` | Yes (HTTP requests) | Yes (via server) | Single-batch latency with HTTP overhead |
| `bench_offline_throughput` | No | Yes (in-process Engine) | Max throughput without HTTP |
| `bench_one_batch` | No | No (direct ModelRunner) | Kernel-level latency profiling |

Use `bench_serving` by default unless you have specific needs.

### bench_serving Usage

`bench_serving` is the primary tool for online serving benchmarks. It measures throughput,
TTFT, TPOT, ITL, and end-to-end latency.

#### Quick Start

```bash
# Terminal 1: Launch server
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct

# Terminal 2: Run benchmark
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

#### Supported Backends

| Backend | Endpoint |
|---------|----------|
| `sglang` / `sglang-native` | `POST /generate` |
| `sglang-oai`, `vllm`, `lmdeploy` | `POST /v1/completions` |
| `sglang-oai-chat`, `vllm-chat`, `lmdeploy-chat` | `POST /v1/chat/completions` |
| `trt` (TensorRT-LLM) | `POST /v2/models/ensemble/generate_stream` |

#### Dataset Options

- `sharegpt` (default): ShareGPT-style conversation pairs.
- `random`: Random text lengths; use `--random-input-len` and `--random-output-len`.
- `random-ids`: Random token IDs.
- `image`: Image data for VLMs; supports custom resolution and format.
- `generated-shared-prefix`: Synthetic shared prefix + short questions.
- `mmmu`: MMMU benchmark with images.

#### Key Parameters

- `--num-prompts N`: Number of requests.
- `--request-rate R`: Requests per second (`inf` for burst).
- `--max-concurrency C`: Cap concurrent in-flight requests.
- `--random-input-len`, `--random-output-len`: Token lengths for random datasets.
- `--profile`: Enable profiling (requires `SGLANG_TORCH_PROFILER_DIR`).
- `--flush-cache`: Flush server cache before benchmark.
- `--output-file FILE.jsonl`: Save results to JSONL.

#### Steady-State Benchmarking

Use `num-prompts >= 5 * max-concurrency` to measure steady-state performance.

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --max-concurrency 16 \
  --num-prompts 80 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

### Performance Metrics

The following metrics are reported:

| Metric | Description |
|--------|-------------|
| Request throughput | Requests per second |
| Input token throughput | Input tokens per second (text + vision) |
| Output token throughput | Output tokens per second |
| Total token throughput | Total tokens per second |
| End-to-End Latency | Per-request total latency (mean/median/std/p99) |
| TTFT | Time to First Token (streaming only) |
| ITL | Inter-Token Latency (mean/median/std/p95/p99/max) |
| TPOT | Time per Output Token: `(latency - ttft) / (tokens - 1)` |
| Accept length | Speculative decoding accept length (sglang-only) |

### Benchmark Methodology

#### Latency-Sensitive Benchmark

Simulates a single-user scenario:

```bash
python -m sglang.bench_serving \
  --backend sglang --host 0.0.0.0 --port 30000 \
  --dataset-name random --num-prompts 10 --max-concurrency 1
```

#### Throughput-Sensitive Benchmark

Simulates high-traffic:

```bash
python -m sglang.bench_serving \
  --backend sglang --host 0.0.0.0 --port 30000 \
  --dataset-name random --num-prompts 1000 --max-concurrency 100
```

#### Granular Concurrency Levels

- **Low**: `--num-prompts 10 --max-concurrency 1`
- **Medium**: `--num-prompts 80 --max-concurrency 16`
- **High**: `--num-prompts 500 --max-concurrency 100`

#### Single Batch Performance

```bash
python -m sglang.bench_one_batch_server \
  --model <model-path> \
  --batch-size 8 --input-len 1024 --output-len 1024
```

### Interpreting Results

1. **Check token usage**: `token usage > 0.9` means good KV cache utilization.
2. **Check queue depth**: A healthy `#queue-req` is 100-2000. Zero means the client is too slow.
3. **Compare TTFT vs TPOT**: High TTFT with low TPOT indicates prefill bottleneck.
4. **Check ITL distribution**: Tight ITL (low std) indicates consistent decode performance.
5. **Verify output quality**: Use `--output-details` to inspect generated text.

### Hyperparameter Tuning for Benchmarks

Key tuning parameters:

- `--mem-fraction-static`: Maximize KV cache (increase until OOM, then back off).
- `--schedule-conservativeness`: Decrease if token usage is low with queued requests.
- `--chunked-prefill-size`: Reduce to `4096` or `2048` if OOM during prefill.
- `--cuda-graph-max-bs`: Increase for large TP sizes (e.g., 512 or 768).
- `--dp-size` and `--tp-size`: Prefer DP for throughput when memory allows.

---

## Debugging

### Debug Tools

#### Forward Hooks

SGLang supports attaching PyTorch forward hooks to submodules for inspecting intermediate
activations. Configure hooks via `--forward-hooks`:

```json
{
  "forward_hooks": [
    {
      "name": "inspect_attention",
      "target_modules": ["model.layers.0.self_attn", "model.layers.*.mlp"],
      "hook_factory": "my_project.hooks:debug_hook_factory",
      "config": {"tag": "attention_debug"}
    }
  ]
}
```

**Hook spec fields**:

- `name`: Human-readable name for logging.
- `target_modules`: Glob patterns matched against `model.named_modules()`.
- `hook_factory`: Python import path (`module:factory` or `module.factory`).
- `config`: Arbitrary JSON passed to the factory.

**Writing a hook factory**:

```python
HOOK_CALLS = []

def debug_hook_factory(config):
    tag = config.get("tag", "default")
    def hook(module, inputs, output):
        HOOK_CALLS.append({
            "module_type": type(module).__name__,
            "tag": tag,
            "shape": tuple(output.shape),
        })
        return output
    return hook
```

#### msprobe Debugging

MSProbe diagnoses accuracy anomalies and numerical errors by capturing intermediate data
(activations, weights, feature maps) and supporting visual analysis.

**Install**:

```bash
pip install mindstudio-probe --pre
```

**Configuration**:

Create `msprobe-config.json`:

```json
{
  "task": "statistics",
  "dump_path": "./dump_path",
  "rank": [],
  "step": [],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

**Dump levels**:

| Level | Target | Use Case |
|-------|--------|----------|
| L0 | `nn.Module` outputs | Module-level debugging |
| L1 | `torch` API calls | Fine-grained numerical checking |
| mix | Both L0 + L1 | Graph reconstruction + numerical comparison |

**Launch with msprobe**:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --msprobe-dump-config /path/to/msprobe-config.json
```

**Analysis workflow**: Enable -> Collect Data -> Visualize (TensorBoard) -> Analyze Root Cause.

Key outputs:
- `dump.json`: Tensor metadata (dtype, shape, min, max, mean, L2 norm).
- `stack.json`: Call stack information.
- `construct.json`: Hierarchical structure (for visualization).
- `dump_tensor_data/`: Raw tensor data.

**Note**: When msprobe is enabled, CUDA graph is disabled and warmup is skipped.

### Common Debugging Patterns

#### NaN Detection

Enable NaN detection for debugging:

```bash
python -m sglang.launch_server --model-path <model> --enable-nan-detection
```

#### Debug Tensor Dumps

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --debug-tensor-dump-output-folder /tmp/dumps \
  --debug-tensor-dump-layers "[0,1,2]"
```

#### Verbose Logging

```bash
# General verbose logging
python -m sglang.launch_server --model-path <model> --log-level debug

# Request-level logging
python -m sglang.launch_server --model-path <model> \
  --log-requests --log-requests-level 3

# Show time costs
python -m sglang.launch_server --model-path <model> --show-time-cost
```

Request logging levels:
- `0`: Metadata only.
- `1`: Metadata + sampling parameters.
- `2`: Metadata + sampling parameters + partial I/O.
- `3`: Full I/O.

#### Crash Dumping

```bash
python -m sglang.launch_server --model-path <model> \
  --crash-dump-folder /tmp/crash_dumps
```

### Log Analysis

Key log patterns to look for:

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `token usage: 0.82` | KV cache utilization | Should be > 0.9 for good throughput |
| `#queue-req: 0` | No queued requests | Client may be too slow |
| `KV cache pool is full. Retract requests.` | Memory pressure | Increase `--schedule-conservativeness` or reduce `--mem-fraction-static` |
| `Decode batch. #running-req: N` | Active requests in decode | Check batch size vs expected concurrency |
| `available_gpu_mem: X GB` | Memory left for activations | Should be 5-8 GB |

### Profiling

#### PyTorch Profiler

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log

# Profile with bench_serving
python -m sglang.bench_serving \
  --backend sglang --num-prompts 10 --profile

# Profile with bench_one_batch
python -m sglang.bench_one_batch \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --batch 32 --input-len 1024 --output-len 10 --profile
```

View traces at https://ui.perfetto.dev/ or `chrome://tracing`.

#### Nsight Systems

```bash
# Profile a single batch
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  python3 -m sglang.bench_one_batch \
  --model meta-llama/Meta-Llama-3-8B --batch-size 64 --input-len 512

# Profile a server
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  -o sglang.out --delay 60 --duration 70 \
  python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct --disable-radix-cache
```

#### Layerwise NVTX Profiling

```bash
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o layerwise_profile \
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-layerwise-nvtx-marker \
    --disable-cuda-graph
```

---

## Release Process

### Version Management

Version numbers are defined in two locations:
- `python/pyproject.toml`: Package version for pip.
- `python/sglang/__init__.py`: Runtime version string.

Follow semantic versioning (MAJOR.MINOR.PATCH).

### Release Checklist

1. **Update version** in `python/pyproject.toml` and `python/sglang/__init__.py`.
2. **Run full test suite** to ensure no regressions.
3. **Update CHANGELOG** if applicable.
4. **Create a release PR** with version bump.
5. **Get approval** from maintainers.
6. **Merge the PR**.
7. **Build and publish** packages.
8. **Create GitHub release**.

### PyPI Publishing

```bash
pip install build twine
cd python
bash upload_pypi.sh
```

### Docker Image Publishing

Docker images are built and published to Docker Hub as `lmsysorg/sglang`:

```bash
# Development image
docker build -t lmsysorg/sglang:dev -f docker/Dockerfile .

# Release image
docker build -t lmsysorg/sglang:v<VERSION> -f docker/Dockerfile .
docker push lmsysorg/sglang:v<VERSION>
```

### GitHub Release

Create a new release at https://github.com/sgl-project/sglang/releases/new with:
- Tag: `v<VERSION>`
- Title: `v<VERSION>`
- Release notes summarizing changes.

### sgl-kernel Release

The sgl-kernel package follows a separate release cycle:
1. Update kernel source in a PR.
2. Bump kernel version in `sgl-kernel/pyproject.toml`.
3. Merged PR triggers automatic PyPI release of `sglang-kernel`.
4. Update `sglang-kernel` version in `sglang/python/pyproject.toml`.

---

## Contributing Guide

### PR Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make changes** following the code style conventions.

3. **Run pre-commit**:
   ```bash
   pre-commit run --all-files
   ```

4. **Add tests** for new features or bug fixes.

5. **Push and create a PR** against the `main` branch.

6. **Get CI triggered**: Authorized users add the "run-ci" label or comment `/tag-run-ci-label`.

7. **Address review feedback** and iterate.

8. **Get approval** from codeowners and merge oncall.

### Code Review Guidelines

#### For Reviewers

- Check for code style and convention compliance.
- Verify test coverage for new code.
- Assess performance impact (especially in model forward paths).
- Check for proper error handling and logging.
- Verify documentation updates for public API changes.

#### For Authors

- Keep PRs focused and reasonably sized.
- Write clear commit messages.
- Link related issues.
- Provide testing instructions in the PR description.
- Run `pre-commit run --all-files` before requesting review.

### Code Style Guidance

- **Avoid duplication**: Extract shared code into functions if the same snippet (5+ lines)
  appears multiple times.
- **Minimize synchronization**: Reduce `tensor.item()`, `tensor.cpu()`, and other CPU-GPU
  sync points. Use vectorized code.
- **Optimize the critical path**: SGLang is a runtime; most code runs on every request's
  critical path. Cache runtime checks as booleans in `__init__` when possible.
- **Pure functions**: Avoid in-place modification of arguments.
- **Keep files concise**: Split files exceeding 2,000 lines (e.g., `scheduler.py` was split
  into `scheduler.py` + `scheduler_output_processor_mixin.py`).
- **File organization**: Put core data structures at the top, utility functions at the bottom.
- **Fast tests**: Keep test files under 500 seconds. Reuse server launches.
- **Security**: Never use `pickle.loads()`, `pickle.load()`, or `recv_pyobj()` for
  untrusted data. Use msgpack or JSON instead.
- **Hardware support**: When adding new hardware:
  - Do not drastically change existing code.
  - Prefer new files for hardware-specific components (e.g., `allocator_ascend.py`).
  - Put the common path (NVIDIA/existing code) as the first branch in if/else blocks.

### GitHub Runner Setup

To add a self-hosted runner for CI:

**Step 1: Start a Docker container**:

```bash
docker pull nvidia/cuda:12.9.1-devel-ubuntu22.04
docker run --shm-size 128g -it -v /tmp/huggingface:/hf_home \
  --gpus all nvidia/cuda:12.9.1-devel-ubuntu22.04 /bin/bash
```

**Step 2: Configure the runner**:

```bash
apt update && apt install -y curl python3-pip git
pip install --upgrade pip
export RUNNER_ALLOW_RUNASROOT=1
# Then follow https://docs.github.com/en/actions/hosting-your-own-runners to run config.sh
```

Give the runner a descriptive name (e.g., `test-sgl-gpu-0`) and labels (e.g., `1-gpu-h100`).

**Step 3: Run the runner**:

```bash
export HF_HOME=/hf_home
export SGLANG_IS_IN_CI=true
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
export CUDA_VISIBLE_DEVICES=0

while true; do ./run.sh; echo "Restarting..."; sleep 2; done
```

### Community Guidelines

- **Good first issues**: Look for issues labeled "good first issue" or "help wanted".
- **Startup resources**:
  - [Mini-SGLang](https://github.com/sgl-project/mini-sglang) for a quick overview.
  - [Code Walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) for a deeper look.
  - [GTC-2026 Training Lab](https://drive.google.com/file/d/1mwOZEtipNLJzrflCTodj34KhuOZEoEw5/view) for hands-on practices.
- **Communication**: Ask questions in the [Slack channel](https://slack.sglang.io).
- **Documentation contributions**: Writing documentation is an excellent way to learn the
  codebase. New contributors are encouraged to start with docs.

---

## Architecture Deep Dive

### Process Model

SGLang uses a multi-process architecture with three main process types:

```
                    +------------------+
                    |   HTTP Server    |
                    |  (FastAPI/uvicorn)|
                    +--------+---------+
                             |
                    +--------v---------+
                    | Tokenizer Manager|
                    |   (Process 1)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Scheduler     |
                    |   (Process 2)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Detokenizer Mgr  |
                    |   (Process 3)    |
                    +------------------+
```

#### Tokenizer Manager

The Tokenizer Manager process handles:
- Receiving requests from the HTTP server.
- Tokenizing input text into token IDs.
- Pre-processing multimodal inputs (images, video, audio).
- Routing requests to the appropriate Scheduler.
- Maintaining the tokenizer and model configuration.

#### Scheduler

The Scheduler is the core orchestrator:
- Manages request batching and scheduling.
- Handles the KV cache (RadixAttention for prefix caching).
- Implements the scheduling policy (FCFS, LPM, DFS, priority).
- Coordinates with the ModelRunner for forward passes.
- Manages request lifecycle (queuing, running, completion, eviction).

Key scheduler components:
- `schedule_batch.py`: Batch data structures for prefill and decode.
- `schedule_policy.py`: Scheduling algorithms (FCFS, LPM, random, priority).
- `scheduler_dp_attn_mixin.py`: Data parallelism attention support.
- `scheduler_pp_mixin.py`: Pipeline parallelism support.
- `scheduler_output_processor_mixin.py`: Output processing logic.

#### Detokenizer Manager

The Detokenizer Manager handles:
- Converting output token IDs back to text.
- Streaming partial outputs to the client.
- Managing stop conditions and special tokens.

### ZMQ IPC Communication

Processes communicate via ZeroMQ (ZMQ) for high-performance inter-process communication:

- **Message Format**: Defined in `managers/io_struct.py` using serialized dataclasses.
- **Serialization**: Uses msgpack (via msgspec) for fast serialization, not pickle.
- **Communication Pattern**: Request-response between Tokenizer Manager <-> Scheduler and
  Scheduler <-> Detokenizer Manager.

The communication flow for a request:

```
HTTP Request -> Tokenizer Manager -> ZMQ -> Scheduler -> ZMQ -> Detokenizer Manager -> HTTP Response
```

Key ZMQ endpoints:
- Tokenizer Manager sends tokenized requests to the Scheduler.
- Scheduler sends decode results to the Detokenizer Manager.
- Scheduler receives processed results back for cache management.

### Memory Management Internals

#### KV Cache Architecture

SGLang uses a paged KV cache with a Radix tree for prefix caching:

```
memory_pool.py       -> GPU memory allocation and page management
radix_cache.py       -> Radix tree for prefix-aware KV cache
allocator.py         -> Token-level memory allocator
chunk_cache.py       -> Chunked cache for long sequences
```

The memory pool is pre-allocated based on `--mem-fraction-static`:

```
Total GPU Memory = Model Weights + KV Cache Pool + CUDA Graph Buffers + Activations
                     |<--- mem-fraction-static --->|
```

#### RadixAttention

The RadixAttention system provides automatic prefix caching by organizing KV cache pages
in a radix tree. When multiple requests share a common prefix (e.g., system prompt), the
KV cache for that prefix is computed once and reused.

Key eviction policies:
- `lru` (default): Least Recently Used.
- `lfu`: Least Frequently Used.

#### Hierarchical Cache (HiCache)

HiCache extends KV cache to host (CPU) memory and even storage:

```
GPU KV Cache (fast) <-> Host KV Cache (medium) <-> Storage Backend (slow)
```

Configuration:
- `--enable-hierarchical-cache`: Enable HiCache.
- `--hicache-ratio`: Host pool size ratio relative to device pool.
- `--hicache-write-policy`: `write_through` (default), `write_back`, or `write_through_selective`.
- `--hicache-storage-backend`: `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`, or `dynamic`.

#### Memory Pool Management

The `TokenToKVPool` class manages the actual KV cache tensors:

- Pre-allocates a large tensor for all KV cache pages.
- Uses a free list to track available page indices.
- Supports layer-first or page-first memory layouts.
- Integrates with the radix tree for reference-counted page management.

### CUDA Graph Internals

CUDA graphs capture a sequence of GPU operations into a single graph that can be replayed
with minimal CPU overhead. This is critical for small batch decode performance.

#### How CUDA Graphs Work in SGLang

1. **Capture Phase**: During server warmup, the model forward pass is captured for various
   batch sizes into CUDA graphs.

2. **Batch Size Buckets**: Pre-captured at specific batch sizes (1, 2, 4, 8, ..., up to
   `--cuda-graph-max-bs`).

3. **Replay Phase**: During serving, the captured graph is replayed with the actual batch
   data. Inputs are padded to the nearest captured batch size.

4. **Piecewise CUDA Graph**: An advanced mode that captures parts of the model separately,
   allowing dynamic control flow between captured segments.

#### Key Configuration

- `--disable-cuda-graph`: Disable CUDA graphs entirely.
- `--cuda-graph-max-bs`: Maximum batch size for graph capture (default varies by model).
- `--cuda-graph-bs`: Explicit list of batch sizes to capture.
- `--disable-piecewise-cuda-graph`: Disable piecewise graph for prefill.
- `--enable-profile-cuda-graph`: Profile graph capture process.

#### When CUDA Graphs Help

CUDA graphs provide the most benefit for:
- Small batch decode (reducing kernel launch overhead).
- Models with many small operations per layer.
- Scenarios where CPU overhead dominates.

CUDA graphs may not help for:
- Large batch sizes (kernel compute dominates).
- Prefill with variable sequence lengths.
- Models with dynamic control flow.

#### Graph Capture Process

During `ModelRunner.initialize()`:
1. Determine batch size buckets based on `--cuda-graph-max-bs`.
2. For each batch size, run a warmup forward pass.
3. Capture the forward pass into a CUDA graph.
4. Store the graph and its input/output buffers.

During serving:
1. Check if the batch size fits within a captured graph.
2. If yes, copy inputs to graph buffers and replay.
3. If no, fall back to eager execution.

---

## Appendix: Quick Reference

### Common Development Commands

```bash
# Install from source (editable)
pip install -e "python[all]"

# Run pre-commit
pre-commit run --all-files

# Run unit tests
pytest test/registered/unit/ -v

# Launch debug server
python -m sglang.launch_server --model-path <model> --log-level debug

# Profile with PyTorch Profiler
SGLANG_TORCH_PROFILER_DIR=/tmp/profiles python -m sglang.bench_serving --profile

# Profile with Nsight
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  python -m sglang.bench_one_batch --model-path <model> --batch-size 32

# Run accuracy check
python -m sglang.test.few_shot_gsm8k --num-questions 200

# Run benchmark
python -m sglang.bench_serving --backend sglang --num-prompts 1000
```

### Key File Locations

| Purpose | Path |
|---------|------|
| Server arguments | `python/sglang/srt/server_args.py` |
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| Model implementations | `python/sglang/srt/models/` |
| KV cache | `python/sglang/srt/mem_cache/radix_cache.py` |
| Memory pool | `python/sglang/srt/mem_cache/memory_pool.py` |
| Attention layers | `python/sglang/srt/layers/attention/` |
| Sampling | `python/sglang/srt/sampling/` |
| JIT kernels | `python/sglang/jit_kernel/` |
| AOT kernels | `sgl-kernel/` |
| Gateway/Router | `sgl-model-gateway/` |
| Tests | `test/registered/` |
| Benchmarks | `benchmark/` |
| Pre-commit config | `.pre-commit-config.yaml` |

### Server Arguments Quick Reference

| Category | Key Arguments |
|----------|---------------|
| Model | `--model-path`, `--tokenizer-path`, `--load-format`, `--trust-remote-code` |
| Server | `--host`, `--port`, `--grpc-mode` |
| Parallelism | `--tp-size`, `--dp-size`, `--pp-size`, `--ep-size` |
| Memory | `--mem-fraction-static`, `--max-running-requests`, `--chunked-prefill-size` |
| Quantization | `--quantization`, `--kv-cache-dtype`, `--dtype` |
| Optimization | `--disable-cuda-graph`, `--enable-torch-compile`, `--disable-radix-cache` |
| Debug | `--log-level`, `--log-requests`, `--enable-nan-detection` |
| LoRA | `--enable-lora`, `--lora-paths`, `--max-lora-rank` |
| Speculative | `--speculative-algorithm`, `--speculative-draft-model-path` |
| Disaggregation | `--disaggregation-mode`, `--disaggregation-transfer-backend` |
