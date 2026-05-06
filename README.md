# AI Infra Plugins Marketplace

A curated marketplace of Claude Code plugins for AI infrastructure engineering.

This repository focuses on workflows around GPU kernels, compilers, runtimes, benchmarking, profiling, and model-serving infrastructure. The local `sources/` directory contains reference projects used by the starter plugins.

> Make sure you trust a plugin before installing, updating, or using it. Plugins may include instructions, MCP servers, hooks, or other software that affects your development environment.

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

Available starter plugins:

- `cutlass-dev` - guidance for CUTLASS/CuTe GPU kernel development
- `tilelang-dev` - guidance for TileLang kernel and scheduling workflows
- `triton-dev` - guidance for Triton compiler and kernel development

## Plugin Structure

Each plugin follows the Claude Code plugin layout:

```text
plugin-name/
├── .claude-plugin/
│   └── plugin.json      # Plugin metadata
├── skills/
│   └── plugin-name/
│       └── SKILL.md     # Model-invoked guidance
└── README.md            # Plugin documentation
```

## Validation

If Bun is available, run:

```bash
bun .github/scripts/validate-marketplace.ts .claude-plugin/marketplace.json
bun .github/scripts/check-marketplace-sorted.ts
```

## License

See each referenced project and plugin directory for license details.
