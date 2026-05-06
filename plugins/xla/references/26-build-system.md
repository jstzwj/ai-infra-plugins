# Building XLA

This document provides comprehensive documentation about building XLA from source, including prerequisites, configuration, build commands, and contribution guidelines.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting the Code](#getting-the-code)
- [Build Configuration](#build-configuration)
- [Build Commands](#build-commands)
- [Static Analysis](#static-analysis)
- [Creating PRs](#creating-prs)

## Prerequisites

### Bazel (Bazelisk)

XLA uses [Bazel](https://bazel.build/) as its build system. The recommended way to install Bazel is through [Bazelisk](https://github.com/bazelbuild/bazelisk), which automatically manages Bazel versions:

```bash
# Install Bazelisk via npm
npm install -g @bazel/bazelisk

# Or via direct download
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
```

Bazelisk reads the `.bazelversion` file in the repository to determine the correct Bazel version to use.

XLA requires a specific version of Bazel, which is specified in the `.bazelversion` file. Common versions used include Bazel 6.x and Bazel 7.x.

### Docker (ml-build container)

Google provides Docker containers with all build dependencies pre-installed. Using Docker is the recommended approach for reproducible builds:

```bash
# Pull the ml-build container
docker pull us-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest

# Run the container
docker run -it --rm \
    -v $(pwd):/workspace \
    us-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest \
    /bin/bash
```

The ml-build container includes:
- Pre-configured Bazel
- CUDA and cuDNN libraries
- Python development headers
- All required system libraries

#### Using Docker with GPU support

For GPU builds, use the GPU-enabled container:

```bash
# Pull the GPU ml-build container
docker pull us-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-gpu:latest

# Run with GPU access
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace \
    us-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build-gpu:latest \
    /bin/bash
```

### GitHub Account

To contribute to XLA, you need:

1. A GitHub account.
2. A signed Contributor License Agreement (CLA).

### CLA (Contributor License Agreement)

Before your first contribution can be merged, you must sign Google's CLA:

1. Visit https://cla.developers.google.com/
2. Sign in with your GitHub account.
3. Follow the prompts to sign the CLA.
4. The CLA status is checked automatically on pull requests.

## Getting the Code

### Fork and Clone

1. Fork the [openxla/xla](https://github.com/openxla/xla) repository on GitHub.

2. Clone your fork:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/xla.git
cd xla

# Or using SSH
git clone git@github.com:YOUR_USERNAME/xla.git
cd xla
```

3. Initialize the repository (XLA uses git submodules or sparse checkout depending on the setup):

```bash
# Initialize submodules if present
git submodule update --init --recursive
```

### Remote Upstream Configuration

Set up the upstream remote to keep your fork in sync:

```bash
# Add the upstream remote
git remote add upstream https://github.com/openxla/xla.git

# Fetch upstream branches
git fetch upstream

# Keep your main branch in sync
git checkout main
git merge upstream/main
```

For day-to-day development:

```bash
# Create a feature branch from upstream main
git fetch upstream
git checkout -b my-feature upstream/main

# Push to your fork
git push -u origin my-feature
```

## Build Configuration

### CPU Backend

To build XLA with only CPU backend support (no GPU dependencies):

```bash
bazel build --backend=CPU //xla/...
```

Or equivalently:

```bash
bazel build --config=cpu //xla/...
```

This configuration:
- Does not require CUDA or cuDNN installed.
- Compiles the CPU backend and all CPU-specific optimizations.
- Runs CPU backend tests.

### GPU Backend

To build XLA with GPU (CUDA) backend support:

```bash
bazel build --backend=CUDA //xla/...
```

Or equivalently:

```bash
bazel build --config=cuda //xla/...
```

This configuration:
- Requires CUDA Toolkit and cuDNN to be installed.
- Compiles the GPU backend including kernel generation and PTX compilation.
- Includes GPU-specific optimization passes.

### CUDA Compute Capabilities

By default, XLA auto-detects the GPU compute capability. To specify manually:

```bash
# Target specific GPU architectures
bazel build --config=cuda \
    --action_env=CUDA_COMPUTE_CAPABILITIES="sm_80,sm_90" \
    //xla/...

# Or for a single architecture
bazel build --config=cuda \
    --action_env=CUDA_COMPUTE_CAPABILITIES="sm_80" \
    //xla/...
```

Common compute capabilities:

| GPU | Compute Capability |
|-----|-------------------|
| NVIDIA V100 | sm_70 |
| NVIDIA T4 | sm_75 |
| NVIDIA A100 | sm_80 |
| NVIDIA A10G | sm_86 |
| NVIDIA L40S | sm_89 |
| NVIDIA H100 | sm_90 |
| NVIDIA B200 | sm_100 |

#### Auto-detect vs Manual

- **Auto-detect** (default): XLA queries the installed GPU at build time. This is convenient for development but requires a GPU to be present.
- **Manual**: Specify `CUDA_COMPUTE_CAPABILITIES` explicitly. This is useful for:
  - Cross-compilation (building on a machine with a different GPU).
  - CI/CD pipelines.
  - Building for multiple architectures.

### Hermetic CUDA Rules

XLA supports hermetic CUDA builds that bundle CUDA libraries with the build:

```bash
# Enable hermetic CUDA
bazel build --config=cuda \
    --repo_env=HERMETIC_CUDA_VERSION=12.3.2 \
    --repo_env=HERMETIC_CUDNN_VERSION=9.1.0 \
    //xla/...
```

Hermetic builds:
- Download CUDA and cuDNN from the network during the build.
- Ensure reproducible builds across different environments.
- Do not require system-installed CUDA/cuDNN.
- Are the recommended approach for CI/CD.

Additional hermetic CUDA configuration options:

```bash
bazel build --config=cuda \
    --repo_env=HERMETIC_CUDA_VERSION=12.3.2 \
    --repo_env=HERMETIC_CUDNN_VERSION=9.1.0 \
    --repo_env=HERMETIC_CUDA_NCCL_VERSION=2.19.3 \
    --repo_env=HERMETIC_CUDA_TENSORRT_VERSION=8.6.1 \
    //xla/...
```

### .bazelrc Configuration

XLA uses `.bazelrc` files for build configuration. Key configurations include:

```
# .bazelrc (simplified)

# CPU build
build:cpu --define=with_cuda_support=false

# CUDA build
build:cuda --define=with_cuda_support=true
build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain

# Common settings
build --enable_bzlmod
build --lockfile_mode=update

# Optimization
build --compilation_mode=opt
build --copt=-O2
```

## Build Commands

### Building All of XLA

```bash
# Build everything
bazel build //xla/...

# Build with all optimizations
bazel build -c opt //xla/...

# Build with debugging symbols
bazel build -c dbg //xla/...

# Build with address sanitizer
bazel build --config=asan //xla/...
```

### Building Specific Targets

```bash
# Build the XLA service library
bazel build //xla/service:xla_service

# Build the GPU compiler
bazel build //xla/service/gpu:gpu_compiler

# Build the PJRT plugin
bazel build //xla/pjrt:plugin

# Build a specific tool
bazel build //xla/tools:run_hlo_module
bazel build //xla/tools:hlo_opt
```

### --spawn_strategy=sandboxed

The `--spawn_strategy=sandboxed` flag runs build actions in a sandboxed environment, which:

- Prevents build actions from accessing undeclared dependencies.
- Ensures hermetic builds.
- Catches accidental dependencies on system files.

```bash
bazel build --spawn_strategy=sandboxed //xla/...
```

### Running Tests

```bash
# Run all XLA tests
bazel test //xla/...

# Run tests with full output
bazel test --test_output=all //xla/...

# Run a specific test
bazel test //xla/service:algebraic_simplifier_test

# Run GPU tests
bazel test --config=cuda //xla/service/gpu:gpu_compiler_test

# Run tests in parallel
bazel test --jobs=8 //xla/...

# Run only tests that match a pattern
bazel test //xla/... --test_tag_filters=gpu

# Run with test filtering
bazel test //xla/service:compiler_test --test_filter=*CompileTest*
```

### Build Performance Tips

```bash
# Use remote caching (if available)
bazel build --remote_cache=grpc://cache.example.com:9092 //xla/...

# Use multiple jobs
bazel build --jobs=$(nproc) //xla/...

# Enable build event streaming
bazel build --bes_backend=grpc://bes.example.com:9092 //xla/...

# Disable Nagle's algorithm for faster network transfers
bazel build --grpc_keepalive_time=10s //xla/...

# Increase Java heap for Bazel
export BAZEL_JAVAC_OPTS="-J-Xmx8g"
```

## Static Analysis

### clang-tidy

XLA uses `clang-tidy` for static analysis to catch common programming errors and enforce coding standards.

#### Running clang-tidy

```bash
# Run clang-tidy on all changed files
bazel build //xla/... \
    --aspects=@llvm-project//clang-tools-extra/clang-tidy:clang_tidy_aspect \
    --output_groups=clang_tidy_checks

# Run on specific targets
bazel build //xla/service:algebraic_simplifier \
    --aspects=@llvm-project//clang-tools-extra/clang-tidy:clang_tidy_aspect \
    --output_groups=clang_tidy_checks
```

#### Run Modes

clang-tidy can be run in different modes:

1. **Full analysis**: Checks all enabled diagnostics.
   ```bash
   bazel build --config=clang-tidy //xla/...
   ```

2. **Specific checks**: Run only specific checks.
   ```bash
   bazel build //xla/... \
       --aspects=@llvm-project//clang-tools-extra/clang-tidy:clang_tidy_aspect \
       --output_groups=clang_tidy_checks \
       --@llvm-project//clang-tools-extra/clang-tidy:checks=-*,bugprone-*,modernize-*
   ```

3. **Fix mode**: Automatically fix issues.
   ```bash
   bazel build //xla/... \
       --aspects=@llvm-project//clang-tools-extra/clang-tidy:clang_tidy_aspect \
       --output_groups=clang_tidy_fixes
   ```

#### Common clang-tidy Checks for XLA

- `bugprone-*`: Catch common programming mistakes.
- `modernize-*`: Enforce modern C++ patterns.
- `performance-*`: Catch performance issues.
- `readability-*`: Enforce readability standards.
- `google-*`: Enforce Google C++ style guide.

## Creating PRs

### Branch Naming Convention

Use descriptive branch names:

```
feature/add-new-fusion-pass
fix/gpu-compiler-crash-on-reshape
docs/update-pjrt-guide
test/add-coverage-for-dot-merger
```

### Commit Message Format

Follow the conventional commit format:

```
[category] Brief description of the change

Detailed explanation of what the change does and why it is needed.

Fixes #12345
```

Categories:
- `[compiler]`: Compiler-related changes.
- `[gpu]`: GPU backend changes.
- `[cpu]`: CPU backend changes.
- `[pjrt]`: PJRT API changes.
- `[mlir]`: MLIR integration changes.
- `[tools]`: Tool changes.
- `[build]`: Build system changes.
- `[test]`: Test additions or fixes.
- `[docs]`: Documentation changes.

### PR Checklist

Before creating a PR:

1. **Build succeeds**:
   ```bash
   bazel build //xla/...
   ```

2. **Tests pass**:
   ```bash
   bazel test //xla/...
   ```

3. **Static analysis passes**:
   ```bash
   bazel build --config=clang-tidy //xla/...
   ```

4. **Code is formatted**:
   ```bash
   # Format C++ code
   clang-format -i $(git diff --name-only HEAD~1 | grep '\.cc$\|\.h$')
   ```

5. **PR description is clear**: Include:
   - What the change does.
   - Why the change is needed.
   - How the change was tested.
   - Any performance impact.

### Creating the PR

```bash
# Push your branch
git push -u origin my-feature

# Create a PR using the GitHub CLI
gh pr create \
    --title "[compiler] Add new fusion pass for elementwise operations" \
    --body "$(cat <<'EOF'
## Summary
Adds a new fusion pass that combines consecutive elementwise operations
into a single fusion region, reducing memory bandwidth requirements.

## Test Plan
- Added unit tests for the new fusion pass.
- Ran existing fusion tests to verify no regressions.
- Benchmarked on A100 showing 15% improvement on transformer models.

Fixes #12345
EOF
)"
```

### CI Checks

PRs are automatically checked by:

1. **Build**: Code must build successfully on CPU and GPU configurations.
2. **Tests**: All tests must pass.
3. **clang-tidy**: Static analysis must pass.
4. **CLA**: Contributor License Agreement must be signed.
5. **Code review**: At least one reviewer must approve.

### Iterating on PRs

```bash
# Make changes
git add -A
git commit -m "[compiler] Address review feedback"
git push

# The CI will automatically re-run on the push
```
