# AI Infra Plugins 插件市场

**[English](README.md)**

为 AI 基础设施工程精心策划的 Claude Code 插件集合。

本仓库专注于 GPU 内核、编译器、运行时、基准测试、性能分析以及模型服务基础设施相关的工作流。

> 在安装、更新或使用插件之前，请确保您信任该插件。插件可能包含影响开发环境的指令、MCP 服务器、钩子或其他软件。

## 可用插件

| 插件 | 版本 | 描述 |
|------|------|------|
| **cuda** | 13.2 | NVIDIA CUDA C++ 编程指南与最佳实践指南 |
| **cutile** | - | NVIDIA cuTile 基于 Tile 的 GPU 编程模型 |
| **cutlass** | 3.8 | NVIDIA CUTLASS/CuTe GPU 内核开发（GEMM、卷积、张量操作） |
| **flash-attention** | 2.8.4/4.x | FlashAttention 快速且内存高效的精确注意力机制 |
| **pytorch** | 2.7 | PyTorch 张量计算与深度学习框架 |
| **tilelang** | 0.1.0 | TileLang 基于 Apache TVM 的高性能 GPU/CPU 内核 DSL |
| **triton** | 3.7.0 | OpenAI Triton GPU 编程语言与编译器 |
| **tvm** | - | Apache TVM 机器学习编译框架 |
| **xla** | 2.0 | XLA（加速线性代数）编译器，支持 GPU、CPU 和 TPU |

## 目录结构

- `/.claude-plugin/marketplace.json` - 市场元数据和插件索引
- `/plugins` - 本地维护的插件
- `/sources` - 插件指导所用的参考源码仓库
- `/.github/scripts` - 市场验证工具
- `/.github/workflows` - 市场元数据的 CI 检查

## 安装方式

通过 Claude Code 的插件系统从本市场安装：

```text
/plugin install {plugin-name}@ai-infra-plugins
```

或直接通过仓库 URL 安装：

```text
/plugin install https://github.com/jstzwj/ai-infra-plugins/tree/main/plugins/{plugin-name}
```

## 插件结构

每个插件遵循 Claude Code 插件布局：

```text
plugin-name/
├── SKILL.md              # 模型调用的指导文档
├── references/           # 有序组织的参考文档
└── README.md             # 插件文档
```

## 验证

如果已安装 Bun，可运行：

```bash
bun .github/scripts/validate-marketplace.ts .claude-plugin/marketplace.json
bun .github/scripts/check-marketplace-sorted.ts
```

## 许可证

请参阅各参考项目和插件目录的许可证详情。
