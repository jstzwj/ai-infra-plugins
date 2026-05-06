# AI Infra Plugins Marketplace

**[中文文档](README.zh-CN.md)**

A curated marketplace of Claude Code plugins for AI infrastructure engineering.

This repository focuses on workflows around GPU kernels, compilers, runtimes, benchmarking, profiling, and model-serving infrastructure.

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

## Structure

- `/.claude-plugin/marketplace.json` - marketplace metadata and plugin index
- `/plugins` - local plugins maintained in this marketplace
- `/sources` - reference source repositories used by plugin guidance
- `/.github/scripts` - marketplace validation utilities
- `/.github/workflows` - CI checks for marketplace metadata

## Installation

Plugins can be installed from this marketplace with Claude Code's plugin system:

```text
/plugin install {plugin-name}@ai-infra-plugins
```

Or install directly from the repository URL:

```text
/plugin install https://github.com/jstzwj/ai-infra-plugins/tree/main/plugins/{plugin-name}
```

## Plugin Structure

Each plugin follows the Claude Code plugin layout:

```text
plugin-name/
├── SKILL.md              # Model-invoked guidance
├── references/           # Organized reference documentation
└── README.md             # Plugin documentation
```

## Validation

If Bun is available, run:

```bash
bun .github/scripts/validate-marketplace.ts .claude-plugin/marketplace.json
bun .github/scripts/check-marketplace-sorted.ts
```

## License

See each referenced project and plugin directory for license details.
