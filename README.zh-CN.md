# AI Infra Plugins

**[English](README.md)**

给 Claude Code 用的 AI 基础设施插件集合，主要覆盖 GPU 算子开发、编译器、推理部署等场景。

> 安装插件前请确认来源可信。插件可能带有 MCP 服务器、hooks 等会影响本地环境的组件。

## 插件列表

| 插件 | 版本 | 简介 |
|------|------|------|
| **cuda** | 13.2 | CUDA C++ 编程指南与最佳实践 |
| **cutile** | - | cuTile 基于 Tile 的 GPU 编程 |
| **cutlass** | 3.8 | CUTLASS/CuTe GPU 算子开发（GEMM、卷积等） |
| **deepspeed** | 0.16 | DeepSpeed 分布式训练与推理优化 |
| **flash-attention** | 2.8.4/4.x | FlashAttention 高效注意力算子 |
| **jax** | 0.6 | JAX 高性能数值计算与 ML 研究库 |
| **mlir** | 19.0 | MLIR 可扩展编译器基础设施（LLVM 项目） |
| **nccl** | 2.30 | NVIDIA NCCL GPU 集合通信库 |
| **nsight** | 2025.x | NVIDIA Nsight Systems 性能分析与 Profiling |
| **onnxruntime** | 1.22 | ONNX Runtime 跨平台推理与训练引擎 |
| **pytorch** | 2.7 | PyTorch 框架全量参考 |
| **ray** | 2.47 | Ray 统一分布式 AI 与 Python 应用框架 |
| **sglang** | - | SGLang 高性能大模型服务框架 |
| **tensorflow** | 2.22 | TensorFlow 端到端机器学习平台 |
| **tile-ir** | 13.2 | Tile IR——NVIDIA GPU 底层 Tile 虚拟机指令集 |
| **tilelang** | 0.1.0 | TileLang——基于 TVM 的高性能算子 DSL |
| **triton** | 3.7.0 | Triton GPU 编程语言与编译器 |
| **tvm** | - | Apache TVM 模型编译与部署 |
| **vllm** | 0.9 | vLLM 高吞吐大模型推理与服务引擎 |
| **xla** | 2.0 | XLA 编译器（GPU/CPU/TPU） |

## 使用方法

### 第一步：添加插件源

```text
/marketplace add https://github.com/jstzwj/ai-infra-plugins.git
```

### 第二步：安装插件

```text
/plugin install triton@ai-infra-plugins
```

把 `triton` 换成你想装的插件名就行。

## 许可证

各插件和引用项目的许可证请查看对应目录。
