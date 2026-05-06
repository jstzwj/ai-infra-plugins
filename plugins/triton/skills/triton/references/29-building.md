# Chapter 29: Building from Source

## Prerequisites

- Python 3.10-3.14
- CMake >= 3.20
- Ninja
- C++ compiler (gcc/clang)
- pybind11

## Quick Build

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -r python/requirements.txt
pip install -e .
```

## Build System Architecture

### CMakeLists.txt (Top-level)

The build system:
1. Downloads/locates LLVM
2. Builds MLIR and LLVM libraries
3. Compiles Triton C++ sources
4. Links Python extension module (`libtriton`)
5. Builds backends (NVIDIA, AMD)

### Key CMake Variables

| Variable | Description |
|----------|-------------|
| `TRITON_BUILD_PYTHON_MODULE` | Build Python module |
| `LLVM_ENABLE_WERROR` | Treat warnings as errors |
| `TRITON_CODEGEN_BACKENDS` | Backend names (nvidia;amd) |
| `TRITON_PLUGIN_DIRS` | External plugin directories |
| `TRITON_BUILD_PROTON` | Build Proton profiler |
| `TRITON_OFFLINE_BUILD` | No network downloads |
| `TRITON_BUILD_WITH_CCACHE` | Enable ccache |
| `TRITON_EXT_ENABLED` | Enable extensions |
| `TRITON_BUILD_UT` | Build unit tests |
| `TRITON_WHEEL_DIR` | Wheel output directory |
| `TRITON_CACHE_PATH` | Triton cache path |
| `TRITON_VERSION` | Version string |

### LLVM Version

Check `cmake/llvm-hash.txt` for the required LLVM commit.

## Build Targets

```bash
# Build everything
cmake --build . --config Release

# Build only the optimizer tool
ninja triton-opt

# Build MLIR documentation
cmake --build . --target mlir-doc

# Build unit tests
cmake --build . --target triton-unittest
```

## Build Tools

### `triton-opt`

MLIR optimizer tool for running passes:

```bash
# Run specific passes
triton-opt input.mlir --triton-to-triton-gpu --tritongpu-pipeline

# Run reproducer
triton-opt reproducer.mlir --run-reproducer

# Dump pass pipeline
triton-opt input.mlir --dump-pass-pipeline
```

### `triton-lsp`

Language Server Protocol server for Triton MLIR:

```bash
triton-lsp  # Starts LSP server
```

### `triton-reduce`

MLIR reducer tool for bug minimization:

```bash
triton-reduce input.mlir --test-script=test.sh
```

### `triton-tensor-layout`

Tensor layout visualization tool:

```bash
triton-tensor-layout input.mlir
```

## Makefile Targets

```makefile
make dev-install     # Full development setup
make dev-install-llvm # Build with custom LLVM
make test            # Run all tests
make test-nogpu      # Tests without GPU
make clean           # Clean build
```

## Cross-Compilation

For building on a machine without GPU:

```bash
# Set target architecture
export TRITON_OVERRIDE_ARCH=90

# Build without GPU
pip install -e .
```

## Custom LLVM Build

```bash
# Quick method
make dev-install-llvm

# Manual method
export LLVM_BUILD_DIR=$HOME/llvm-project/build
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
pip install -e .
```

## Build Optimization

```bash
# Use clang+lld for faster builds
export TRITON_BUILD_WITH_CLANG_LLD=true

# Use ccache
export TRITON_BUILD_WITH_CCACHE=true

# Limit parallel jobs
export MAX_JOBS=8

# Faster nop builds
pip install -e . --no-build-isolation

# Build type
export DEBUG=1          # Debug build
export REL_WITH_DEB_INFO=1  # Release with debug info
```

## Plugin Build

```bash
# Build with external plugins
export TRITON_PLUGIN_DIRS="/path/to/plugin1;/path/to/plugin2"
pip install -e .
```

### Plugin Directory Structure

```
my_plugin/
├── backend/
│   ├── name.conf        # Backend name
│   ├── compiler.py      # Compiler backend
│   └── driver.py        # Driver backend
├── language/            # Optional language extensions
│   └── my_ext/
└── tools/               # Optional tools
    └── my_tool/
```
