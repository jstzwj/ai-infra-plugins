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
| **deepspeed** | 0.16 | DeepSpeed distributed deep learning training and inference optimization |
| **flash-attention** | 2.8.4/4.x | FlashAttention fast and memory-efficient exact attention |
| **jax** | 0.6 | JAX high-performance numerical computing and ML research |
| **mlir** | 19.0 | MLIR extensible compiler infrastructure from the LLVM project |
| **nccl** | 2.30 | NVIDIA NCCL GPU collective communications library |
| **nsight** | 2025.x | NVIDIA Nsight Systems performance analysis and profiling |
| **onnxruntime** | 1.22 | ONNX Runtime cross-platform inference and training engine |
| **pytorch** | 2.7 | PyTorch tensor computation and deep learning framework |
| **ray** | 2.47 | Ray unified framework for scaling AI and Python applications |
| **sglang** | - | SGLang high-performance LLM serving framework |
| **tensorflow** | 2.22 | TensorFlow end-to-end machine learning platform |
| **tile-ir** | 13.2 | Tile IR low-level tile virtual machine for NVIDIA GPUs |
| **tilelang** | 0.1.0 | TileLang high-performance GPU/CPU kernel DSL on Apache TVM |
| **triton** | 3.7.0 | OpenAI Triton GPU programming language and compiler |
| **tvm** | - | Apache TVM machine learning compilation framework |
| **vllm** | 0.9 | vLLM high-throughput LLM inference and serving engine |
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
