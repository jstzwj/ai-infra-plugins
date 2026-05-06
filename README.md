# AI Infra Plugins Marketplace

**[中文文档](README.zh-CN.md)**

A curated marketplace of Claude Code plugins for AI infrastructure engineering, covering GPU kernels, compilers, runtimes, benchmarking, profiling, and model-serving infrastructure.

> Make sure you trust a plugin before installing, updating, or using it. Plugins may include instructions, MCP servers, hooks, or other software that affects your development environment.

## Available Plugins

| Plugin | Version | Description |
|--------|---------|-------------|
| **cuda** | 13.2 | NVIDIA CUDA C++ Programming Guide and Best Practices Guide |
| **cutile** | - | NVIDIA cuTile tile-based GPU programming model |
| **cutlass** | 3.8 | NVIDIA CUTLASS/CuTe GPU kernel development (GEMM, convolution, tensor operations) |
| **flash-attention** | 2.8.4/4.x | FlashAttention fast and memory-efficient exact attention |
| **pytorch** | 2.7 | PyTorch tensor computation and deep learning framework |
| **tilelang** | 0.1.0 | TileLang high-performance GPU/CPU kernel DSL on Apache TVM |
| **triton** | 3.7.0 | OpenAI Triton GPU programming language and compiler |
| **tvm** | - | Apache TVM machine learning compilation framework |
| **xla** | 2.0 | XLA (Accelerated Linear Algebra) compiler for GPUs, CPUs, and TPUs |

## Installation

### 1. Add the marketplace

```text
/marketplace add https://github.com/jstzwj/ai-infra-plugins.git
```

### 2. Install plugins

Once the marketplace is added, install any plugin by name:

```text
/plugin install triton@ai-infra-plugins
```

## License

See each referenced project and plugin directory for license details.
