# bitsandbytes: Development and Contributing

This document covers the development setup, testing, CI/CD, code standards, PR workflow, and project structure for bitsandbytes contributors.

---

## Development Setup

### Clone and Install

The recommended development installation uses editable mode with test dependencies:

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
pip install -e ".[test]"
```

For full development tools (linting, pre-commit):

```bash
pip install -e ".[dev]"
```

This installs:
- The package in editable mode (`-e`).
- Test dependencies: `pytest`, `scipy`, `transformers`, `einops`, `lion-pytorch`.
- Dev dependencies: `ruff`, `pre-commit`, `build`, `wheel`.

### CUDA Compilation

bitsandbytes uses CMake for building native CUDA/C++ libraries. The build system is configured in:

- **`CMakeLists.txt`**: Defines CUDA kernel compilation targets, source files, and platform detection.
- **`pyproject.toml`**: Build system configuration using `scikit-build-core` and `setuptools`.

```toml
[build-system]
requires = ["scikit-build-core", "setuptools >= 77.0.3", "trove-classifiers>=2025.8.6.13"]
build-backend = "scikit_build_core.setuptools.build_meta"
```

To compile from source:

```bash
# Standard build (auto-detects CUDA)
pip install -e .

# CPU-only build
cmake -DCOMPUTE_BACKEND=cpu -S . && make

# Force specific CUDA version
CUDA_VERSION=12.1 pip install -e .
```

The compiled libraries are placed in the `bitsandbytes/` package directory as:
- `libbitsandbytes_cpu.so` -- CPU-only library.
- `libbitsandbytes_cudaXXX.so` -- CUDA-specific library (e.g., `libbitsandbytes_cuda121.so`).
- `libbitsandbytes_xpu.so` -- Intel XPU library.
- `libbitsandbytes_rocmXX.so` -- AMD ROCm library.

### Pre-commit Hooks

Install the pre-commit hooks:

```bash
pre-commit install
```

The pre-commit configuration runs the following hooks:

| Hook | Purpose |
|---|---|
| `ruff` | Python linting (bugbear, pycodestyle, pyflakes, isort, etc.) |
| `ruff-format` | Python code formatting |
| `typos` | Spell checking |
| `trailing-whitespace` | Remove trailing whitespace |
| `clang-format` | C++/CUDA code formatting |

To run all hooks manually:

```bash
pre-commit run --all-files
```

**IMPORTANT**: Always run the full pre-commit suite before pushing. CI will reject PRs that fail any check. Do NOT run only `ruff check` and `ruff format` -- those are just 2 of 10+ hooks.

---

## Testing

### Test Files

The test suite is in the `tests/` directory:

| File | Tests |
|---|---|
| `test_functional.py` | Quantization/dequantization primitives, blockwise ops, 4-bit ops, int8 ops |
| `test_linear4bit.py` | `Linear4bit`, `LinearFP4`, `LinearNF4` modules |
| `test_linear8bitlt.py` | `Linear8bitLt` module |
| `test_modules.py` | `StableEmbedding`, `Embedding8bit`, `Embedding4bit`, `Params4bit`, `Int8Params` |
| `test_autograd.py` | `MatMul8bitLt`, `MatMul4Bit`, `MatMul8bitFp` autograd functions |
| `test_ops.py` | PyTorch custom op definitions and fake implementations |
| `test_optim.py` | All optimizer variants (8-bit, 32-bit, paged) |
| `test_parametrize.py` | `Bnb4bitParametrization`, `replace_parameter_4bit` |
| `test_generation.py` | End-to-end text generation with quantized models |
| `test_cuda_setup_evaluator.py` | CUDA setup and device detection |
| `conftest.py` | Shared pytest fixtures |
| `helpers.py` | Test utility functions |

### Running Specific Tests

Only run the tests that cover the code you changed. Do NOT run the full test suite (10+ minutes):

```bash
# Run a single test file
pytest tests/test_linear4bit.py -v --tb=short

# Run specific tests by name
pytest tests/test_optim.py -v --tb=short -k "test_adam"

# Run with markers (exclude slow/benchmark tests)
pytest tests/test_functional.py -v --tb=short -m "not slow and not benchmark"
```

Pytest configuration in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "-rP -m 'not slow and not benchmark and not deprecated'"
log_cli = true
log_cli_level = "INFO"
log_file = "logs/pytest.log"
markers = [
    "benchmark: mark test as a benchmark",
    "deprecated: mark test as covering a deprecated feature",
    "slow: mark test as slow",
]
```

### Platform-Specific Tests

Tests that require specific hardware are guarded by availability checks:

```python
# Tests requiring CUDA
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_something_cuda():
    ...

# Tests requiring specific compute capability
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="Requires SM 7.5+"
)
def test_tensor_core_ops():
    ...
```

### Integration Tests with PyTorch

Some tests verify integration with PyTorch features:
- `torch.compile` compatibility via custom op fake implementations.
- FSDP state_dict save/load with quantized parameters.
- Device movement (`.to()`, `.cuda()`, `.cpu()`) triggering quantization.
- Gradient computation through quantized layers.

---

## CI/CD

### GitHub Actions Workflows

The CI/CD pipeline runs on GitHub Actions with workflows for building, testing, and releasing.

### Multi-Platform Builds

The project builds native libraries for multiple platforms:

| Platform | Backend | Build Target |
|---|---|---|
| NVIDIA CUDA | `cuda` | Multiple CUDA versions (11.8, 12.x, 13.x) |
| AMD ROCm | `rocm` | Multiple ROCm versions |
| Intel XPU | `xpu` | SYCL/DPC++ |
| CPU | `cpu` | CPU-only (subset of operations) |

Build scripts are in `.github/scripts/`:
- `set_platform_tag.py` -- Sets platform tags for wheel distribution.
- `auditwheel_show.py` -- Audits wheel for library dependencies.

### Nightly Test Runs

Nightly tests run the full test suite on all supported platforms to catch regressions.

### PR-Specific Test Runs

Each PR triggers:
1. Linting checks (ruff, ruff-format, typos, clang-format).
2. Platform-specific builds.
3. Subset of tests relevant to the changes.

---

## Code Standards

### Python: Ruff Linting

Ruff configuration in `pyproject.toml`:

```toml
[tool.ruff]
src = ["bitsandbytes", "tests", "benchmarking"]
target-version = "py310"
line-length = 119

[tool.ruff.lint]
select = [
    "B",    # bugbear: security warnings
    "E",    # pycodestyle (error)
    "W",    # pycodestyle (warning)
    "F",    # pyflakes
    "I",    # isort
    "ISC",  # implicit string concatenation
    "UP",   # alert you when better syntax is available
    "RUF",  # ruff developer's own rules
]
```

### Python: Ruff Formatting

Code formatting uses `ruff format` (replaces `yapf` and `black`). The line length is 119 characters.

### C++/CUDA: clang-format

C++ and CUDA source files in `csrc/` are formatted with `clang-format` via the pre-commit hook.

### Type Hints

Type hints are used throughout the codebase. Key patterns:

```python
from typing import Optional, Sequence

def quantize_4bit(
    A: torch.Tensor,
    absmax: Optional[torch.Tensor] = None,
    blocksize: Optional[int] = None,
    quant_type: str = "fp4",
) -> tuple[torch.Tensor, QuantState]:
    ...
```

### Docstring Conventions (Google-style)

Docstrings follow Google-style formatting:

```python
def quantize_blockwise(
    A: torch.Tensor,
    code: Optional[torch.Tensor] = None,
    blocksize: int = 4096,
) -> tuple[torch.Tensor, QuantState]:
    """Quantize a tensor in blocks of values.

    The input tensor is quantized by dividing it into blocks of `blocksize` values.
    The absolute maximum value within these blocks is calculated for scaling.

    Args:
        A (`torch.Tensor`): The input tensor. Supports `float16`, `bfloat16`, or `float32`.
        code (`torch.Tensor`, *optional*): A mapping describing the low-bit data type.
        blocksize (`int`, *optional*): The size of the blocks. Defaults to 4096.

    Returns:
        `Tuple[torch.Tensor, QuantState]`: The quantized tensor and state object.
    """
```

---

## PR Workflow

### Mandatory Worktree Usage

All branch work must be done in git worktrees, not in the main checkout:

```bash
cd ~/git/bitsandbytes
git worktree add ~/git/bnb-fix-123 -b fix/issue-123
cd ~/git/bnb-fix-123
```

This keeps the main checkout clean and allows parallel sessions. See `agents/worktree_guide.md` for details.

### Check for Existing PRs

Before starting work on any issue, check for existing PRs:

```bash
gh pr list --search "issue-number OR keyword" --state open
```

If a PR exists, review and build on it instead of starting from scratch.

### Full Pre-commit Before Push

Before pushing a PR branch, run:

```bash
pre-commit run --all-files
```

Review and commit any changes it makes. CI will reject PRs that fail any check.

### PR Review Checklist

When reviewing PRs, consult the agent guides:

1. `agents/pr_review_guide.md` -- Complete review workflow.
2. `agents/architecture_guide.md` -- Codebase architecture and patterns.
3. `agents/code_standards.md` -- Code quality expectations.
4. `agents/api_surface.md` -- Public API catalog (for detecting breaking changes).
5. `agents/downstream_integrations.md` -- How Transformers, PEFT, Accelerate, TGI, and vLLM depend on bitsandbytes.
6. `agents/security_guide.md` -- Trust model and security checklist.

### Agent Guides

The `agents/` directory contains comprehensive guides for automation and code review:

| File | Purpose |
|---|---|
| `architecture_guide.md` | Codebase architecture, module organization, data flow |
| `api_surface.md` | Public API catalog for breaking-change detection |
| `code_standards.md` | Code quality expectations and patterns |
| `pr_review_guide.md` | Complete PR review workflow |
| `security_guide.md` | Security trust model and checklist |
| `testing_guide.md` | Testing best practices and known issues |
| `linting_guide.md` | Linting configuration and troubleshooting |
| `dispatch_guide.md` | Issue triage and agent dispatch |
| `worktree_guide.md` | Git worktree management |
| `downstream_integrations.md` | Detailed downstream integration catalog |
| `issue_maintenance_guide.md` | Stale/duplicate issue management |
| `issue_triage_workflow.md` | Issue triage process |
| `issue_patterns.md` | Common closeable issue patterns |
| `github_tools_guide.md` | GitHub CLI tools reference |

---

## Build System

### pyproject.toml

The project metadata and build configuration:

```toml
[project]
name = "bitsandbytes"
dynamic = ["version"]
description = "k-bit optimizers and matrix multiplication routines."
requires-python = ">=3.10"
dependencies = [
    "torch>=2.3,<3",
    "numpy>=1.17",
    "packaging>=20.9",
]

[project.optional-dependencies]
test = ["einops~=0.8.0", "lion-pytorch==0.2.3", "pytest~=8.3", "scipy>=1.11.4,<2", "transformers>=4.30.1,<5"]
dev = ["bitsandbytes[test]", "build>=1.0.0,<2", "ruff~=0.14.3", "pre-commit>=3.5.0,<4", "wheel>=0.42,<1"]
```

### CMakeLists.txt

The CMake build configuration handles:
- CUDA kernel compilation with architecture-specific optimizations.
- Platform detection (CUDA, ROCm, CPU).
- Multi-CUDA version support.
- Output library naming (e.g., `libbitsandbytes_cuda121.so`).

### Multi-CUDA Version Support

The build system produces separate binaries for each CUDA version:

```
bitsandbytes/
  libbitsandbytes_cpu.so
  libbitsandbytes_cuda118.so
  libbitsandbytes_cuda120.so
  libbitsandbytes_cuda121.so
  ...
```

At runtime, `cextension.py` selects the correct binary based on `get_cuda_specs()`:

```python
def get_cuda_bnb_library_path(cuda_specs: CUDASpecs) -> Path:
    prefix = "rocm" if torch.version.hip else "cuda"
    library_name = f"libbitsandbytes_{prefix}{cuda_specs.cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"
    return PACKAGE_DIR / library_name
```

---

## Project Structure

```
bitsandbytes/                          # Repository root
|
+-- bitsandbytes/                      # Main Python package
|   +-- __init__.py                    # Package init, backend loading, version (0.50.0.dev0)
|   +-- functional.py                  # Quantization/dequantization primitives, QuantState, optim updates
|   +-- _ops.py                        # PyTorch custom op definitions and fake impls
|   +-- utils.py                       # Utility functions (replace_linear, pack_dict_to_tensor, OutlierTracer, sync_gpu)
|   +-- cextension.py                  # C library loading, BNBNativeLibrary, error handling
|   +-- cuda_specs.py                  # CUDA specification detection (CUDASpecs dataclass)
|   +-- consts.py                      # Constants (PACKAGE_DIR, DYNAMIC_LIBRARY_SUFFIX)
|   +-- __main__.py                    # CLI entry point (python -m bitsandbytes)
|   |
|   +-- nn/                            # Neural network modules
|   |   +-- __init__.py                # Module exports
|   |   +-- modules.py                 # Linear8bitLt, Linear4bit, LinearFP4, LinearNF4,
|   |   |                              #   Int8Params, Params4bit, StableEmbedding, Embedding*
|   |   +-- parametrize.py             # Bnb4bitParametrization, replace_parameter_4bit
|   |
|   +-- autograd/                      # Autograd functions
|   |   +-- __init__.py                # Exports
|   |   +-- _functions.py              # MatMul8bitLt, MatMul4Bit, MatMul8bitFp,
|   |                                  #   MatmulLtState, matmul, matmul_4bit
|   |
|   +-- optim/                         # Optimizers
|   |   +-- __init__.py                # Exports
|   |   +-- optimizer.py               # Optimizer8bit, Optimizer2State, Optimizer1State,
|   |   |                              #   GlobalOptimManager, MockArgs
|   |   +-- adam.py                    # Adam, Adam8bit, Adam32bit, PagedAdam variants
|   |   +-- adamw.py                   # AdamW, AdamW8bit, AdamW32bit, PagedAdamW variants
|   |   +-- sgd.py                     # SGD, SGD8bit, SGD32bit
|   |   +-- lion.py                    # Lion, Lion8bit, Lion32bit, PagedLion variants
|   |   +-- lamb.py                    # LAMB, LAMB8bit, LAMB32bit
|   |   +-- lars.py                    # LARS, LARS8bit, LARS32bit, PytorchLARS
|   |   +-- adagrad.py                 # Adagrad, Adagrad8bit, Adagrad32bit
|   |   +-- rmsprop.py                 # RMSprop, RMSprop8bit, RMSprop32bit
|   |   +-- ademamix.py                # AdEMAMix, AdEMAMix8bit, AdEMAMix32bit, Paged variants
|   |
|   +-- backends/                      # Hardware backend implementations
|   |   +-- __init__.py                # Backend exports
|   |   +-- utils.py                   # Shared backend utilities
|   |   +-- default/                   # Fallback implementations
|   |   |   +-- __init__.py
|   |   |   +-- ops.py                 # Default (CPU-compatible) op implementations
|   |   +-- cuda/                      # NVIDIA CUDA implementations
|   |   |   +-- __init__.py
|   |   |   +-- ops.py                 # CUDA-specific op implementations
|   |   +-- cpu/                       # CPU implementations
|   |   |   +-- __init__.py
|   |   |   +-- ops.py                 # CPU-specific op implementations
|   |   +-- triton/                    # Triton kernel implementations
|   |   |   +-- __init__.py
|   |   |   +-- ops.py                 # Triton op implementations
|   |   |   +-- kernels_4bit.py        # 4-bit dequantization Triton kernels
|   |   |   +-- kernels_8bit_quant.py  # 8-bit quantization Triton kernels
|   |   |   +-- kernels_optim.py       # Optimizer Triton kernels
|   |   +-- mps/                       # Metal Performance Shaders (Apple Silicon)
|   |   |   +-- __init__.py
|   |   |   +-- ops.py                 # MPS op implementations
|   |   +-- xpu/                       # Intel XPU (GPU)
|   |   |   +-- __init__.py
|   |   |   +-- ops.py                 # XPU op implementations
|   |   +-- hpu/                       # Intel Gaudi
|   |       +-- __init__.py
|   |       +-- ops.py                 # HPU op implementations
|   |
|   +-- diagnostics/                   # Diagnostics module
|       +-- __init__.py
|       +-- main.py                    # Main diagnostics (python -m bitsandbytes)
|       +-- cuda.py                    # CUDA diagnostics, library path detection
|       +-- utils.py                   # Diagnostic utility functions (print_header, print_dedented)
|
+-- csrc/                              # CUDA/C++ source
|   +-- pythonInterface.cpp            # Pybind11/C interface to Python
|   +-- ops.cu                         # CUDA operation kernels
|   +-- ops.cuh                        # CUDA operation headers
|   +-- kernels.cu                     # Lower-level CUDA kernels
|   +-- kernels.cuh                    # Kernel headers
|   +-- common.h / common.cuh         # Common C++/CUDA definitions
|   +-- compat.cuh / compat_device.cuh # Compatibility headers
|   +-- cpu_ops.cpp / cpu_ops.h        # CPU operation implementations
|   +-- xpu_ops.cpp / xpu_ops.h       # Intel XPU operation implementations
|   +-- xpu_kernels.cpp / xpu_kernels.h # XPU kernel implementations
|   +-- mps_kernels.metal / mps_ops.mm # Apple Metal kernel implementations
|
+-- tests/                             # Test suite
|   +-- conftest.py                    # Pytest configuration and fixtures
|   +-- helpers.py                     # Test helper functions
|   +-- __init__.py                    # Test package init
|   +-- test_functional.py            # Functional API tests
|   +-- test_linear4bit.py            # 4-bit linear layer tests
|   +-- test_linear8bitlt.py          # 8-bit linear layer tests
|   +-- test_modules.py               # Module tests (embeddings, params)
|   +-- test_autograd.py              # Autograd function tests
|   +-- test_ops.py                   # Custom op tests
|   +-- test_optim.py                 # Optimizer tests
|   +-- test_parametrize.py           # Parametrization tests
|   +-- test_generation.py            # End-to-end generation tests
|   +-- test_cuda_setup_evaluator.py  # CUDA setup tests
|   +-- fsdp_state_dict_save.py       # FSDP state dict save test script
|
+-- examples/                          # Usage examples
|   +-- int8_inference_huggingface.py  # 8-bit inference with HuggingFace
|   +-- compile_inference.py           # torch.compile inference example
|   +-- cpu/                           # CPU-specific examples
|   |   +-- cpu_training.py            # CPU training example
|   +-- xpu/                           # XPU-specific examples
|       +-- paged_xpu_training.py      # Paged XPU training example
|       +-- benchmark_paged_memory.py  # Paged memory benchmark
|
+-- benchmarking/                      # Performance benchmarks
|   +-- inference_benchmark.py         # Inference speed benchmarks
|   +-- matmul_benchmark.py            # Matmul performance benchmarks
|   +-- optimizer_benchmark.py         # Optimizer performance benchmarks
|   +-- int8/                          # INT8-specific benchmarks
|   |   +-- training_benchmark.py
|   |   +-- int8_benchmark.py
|   +-- xpu/                           # XPU-specific benchmarks
|       +-- inference_benchmark.py
|
+-- docs/                              # Documentation source
|   +-- source/                        # MDX documentation files
|   |   +-- index.mdx                  # Documentation index
|   |   +-- quickstart.mdx             # Quick start guide
|   |   +-- installation.mdx           # Installation guide
|   |   +-- optimizers.mdx             # Optimizer documentation
|   |   +-- integrations.mdx           # Integration guides
|   |   +-- contributing.mdx           # Contribution guide
|   |   +-- errors.mdx                 # Error documentation
|   |   +-- faqs.mdx                   # FAQ
|   |   +-- fsdp_qlora.md             # FSDP+QLoRA guide
|   |   +-- reference/                 # API reference
|   |   |   +-- functional.mdx         # Functional API reference
|   |   |   +-- nn/                    # Neural network module references
|   |   |   |   +-- linear4bit.mdx
|   |   |   |   +-- linear8bit.mdx
|   |   |   |   +-- embeddings.mdx
|   |   |   +-- optim/                 # Optimizer references
|   |   |       +-- optim_overview.mdx
|   |   |       +-- adam.mdx / adamw.mdx / sgd.mdx / lion.mdx
|   |   |       +-- lamb.mdx / lars.mdx / adagrad.mdx / rmsprop.mdx
|   |   |       +-- ademamix.mdx
|   |   +-- explanations/              # In-depth explanations
|   |       +-- optimizers.mdx         # How 8-bit optimizers work
|   |       +-- resources.mdx          # Resource usage guide
|   +-- _toctree.yml                   # Documentation table of contents
|
+-- agents/                            # Agent guides for automation
|   +-- architecture_guide.md          # Codebase architecture
|   +-- api_surface.md                 # Public API catalog
|   +-- code_standards.md              # Code quality standards
|   +-- pr_review_guide.md             # PR review workflow
|   +-- security_guide.md              # Security checklist
|   +-- testing_guide.md               # Testing practices
|   +-- linting_guide.md               # Linting configuration
|   +-- dispatch_guide.md              # Issue dispatch workflow
|   +-- worktree_guide.md              # Git worktree management
|   +-- downstream_integrations.md     # Downstream integration catalog
|   +-- issue_maintenance_guide.md     # Issue maintenance
|   +-- issue_triage_workflow.md       # Issue triage
|   +-- issue_patterns.md              # Common issue patterns
|   +-- github_tools_guide.md          # GitHub tools reference
|   +-- fetch_issues.py                # GitHub issue fetching script
|   +-- query_issues.py                # GitHub issue query script
|
+-- scripts/                           # Utility scripts
|   +-- stale.py                       # Stale issue management
|
+-- .github/                           # GitHub configuration
|   +-- scripts/                       # CI scripts
|       +-- set_platform_tag.py        # Platform tag setter
|       +-- auditwheel_show.py         # Wheel audit script
|
+-- pyproject.toml                     # Project configuration
+-- CMakeLists.txt                     # CMake build configuration
+-- setup.py                           # Legacy setup script
+-- install_cuda.py                    # CUDA installation helper
+-- install_cuda.sh                    # CUDA installation shell script
+-- check_bnb_install.py              # Installation verification script
+-- _typos.toml                        # Typos configuration
+-- MANIFEST.in                        # Package manifest
+-- CLAUDE.md                          # Agent instructions
```
