# MLIR Tools

## Overview

MLIR provides a set of command-line tools for processing, transforming, and analyzing MLIR code. This chapter covers the main tools and their usage.

## mlir-opt

### Overview

`mlir-opt` is the primary command-line entry point for running passes and lowerings on MLIR code. It loads textual IR or bytecode into an in-memory structure, optionally executes a sequence of passes, and serializes back the IR.

### Basic Usage

```bash
# Parse and verify MLIR (no passes)
mlir-opt input.mlir

# Read from stdin
cat input.mlir | mlir-opt

# Run a single pass
mlir-opt --pass-pipeline="builtin.module(convert-math-to-llvm)" input.mlir

# Show all available flags
mlir-opt --help
mlir-opt --help-hidden
```

### Pass Pipeline Syntax

```bash
# Single pass anchored on builtin.module
mlir-opt --pass-pipeline="builtin.module(canonicalize)" input.mlir

# Multiple passes chained
mlir-opt --pass-pipeline="builtin.module(pass1,pass2,pass3)" input.mlir

# Nested pipeline with anchoring
mlir-opt --pass-pipeline='
    builtin.module(
        builtin.module(
            func.func(cse,canonicalize),
            convert-to-llvm
        )
    )' input.mlir

# Pass with options (space-separated key=value pairs)
mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion{compute-tolerance=0})" input.mlir
```

### Useful CLI Flags

| Flag | Description |
|------|-------------|
| `--debug` | Print all debug output from `LLVM_DEBUG` calls |
| `--debug-only="tag"` | Print only debug output for the given tag |
| `--dump-pass-pipeline` | Dump the pass pipeline to stdout |
| `--emit-bytecode` | Emit MLIR in bytecode format |
| `--mlir-pass-statistics` | Print pass statistics |
| `--mlir-print-ir-after-all` | Print IR after each pass |
| `--mlir-print-ir-after-change` | Print IR after each pass that changed it |
| `--mlir-print-ir-after-failure` | Print IR after each failed pass |
| `--mlir-print-ir-before-all` | Print IR before each pass |
| `--mlir-print-ir-tree-dir` | Write IR dumps to files in a directory tree |
| `--mlir-timing` | Display execution times of each pass |
| `--view-op-graph` | Generate Graphviz DOT file of the module |
| `-mlir-print-debuginfo` | Include source locations in output |

### Debug Tags

Common debug tags for `--debug-only`:

| Tag | Description |
|-----|-------------|
| `greedy-rewriter` | Debug greedy pattern rewriter |
| `dialect-conversion` | Debug dialect conversion framework |

### Example: Complete Lowering Pipeline

```bash
# Lower from high-level dialects to LLVM
mlir-opt input.mlir \
  --pass-pipeline="builtin.module(\
    func.func(scf-for-loop-tiling{tile-size=32}),\
    canonicalize,\
    cse,\
    one-shot-bufferize,\
    convert-linalg-to-loops,\
    convert-scf-to-cf,\
    convert-cf-to-llvm,\
    convert-func-to-llvm,\
    convert-arith-to-llvm,\
    reconcile-unrealized-casts\
  )"
```

## mlir-translate

### Overview

`mlir-translate` converts between MLIR and other formats (e.g., LLVM IR, SPIR-V binary).

### Usage

```bash
# Translate MLIR to LLVM IR
mlir-translate --mlir-to-llvmir input.mlir

# Translate LLVM IR to MLIR
mlir-translate --llvmir-to-mlir input.ll

# SPIR-V serialization
mlir-translate --serialize-spirv input.mlir -o output.spv

# SPIR-V deserialization
mlir-translate --deserialize-spirv input.spv
```

## mlir-cpu-runner

### Overview

`mlir-cpu-runner` JIT-compiles and executes MLIR modules on the CPU.

### Usage

```bash
# Run a module
mlir-cpu-runner input.mlir

# With optimization
mlir-cpu-runner -O3 input.mlir

# With shared libraries
mlir-cpu-runner --shared-libs=/path/to/lib.mlir_runtime.shlib input.mlir

# Entry point function (default: main)
mlir-cpu-runner -e my_func input.mlir
```

### Example Module

```mlir
module {
  func.func @main() -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }
}
```

## mlir-lsp-server

### Overview

Language Server Protocol server for `.mlir` files, providing IDE features like code completion, cross-references, and diagnostics.

### Features

| Feature | Description |
|---------|-------------|
| Diagnostics | Live verification as you type |
| Code completion | Suggestions for dialect constructs, block names, keywords |
| Find definition | Navigate to SSA value/symbol/block definitions |
| Find references | Show all uses of an entity |
| Hover | Show operation info, generic format |
| Navigation | Symbol table outline and navigation |
| Bytecode editing | Transparently view/edit bytecode files |

### Custom Dialect Support

```c++
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerMyDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
```

### VSCode Setup

Install the [MLIR extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir) and configure the server path:

```json
{
  "mlir.server_path": "/path/to/mlir-lsp-server"
}
```

## mlir-pdll-lsp-server

### Overview

Language server for `.pdll` (PDLL pattern) files.

### Compilation Database

Create `pdll_compile_commands.yml`:

```yaml
--- !FileInfo:
  filepath: "/path/to/file.pdll"
  includes: "/path/to/include1;/path/to/include2"
```

### Features

- Diagnostics
- Code completion and signature help
- Cross-references (definition, references)
- Hover with ODS information
- Navigation
- View intermediate output (AST, MLIR, C++)
- Inlay hints (types, operand/result names)

## tblgen-lsp-server

### Overview

Language server for `.td` (TableGen) files.

### Compilation Database

Create `tablegen_compile_commands.yml`:

```yaml
--- !FileInfo:
  filepath: "/path/to/file.td"
  includes: "/path/to/include1;/path/to/include2"
```

### Features

- Diagnostics
- Cross-references
- Hover (type, documentation, overridden field info)

## mlir-reduce

### Overview

Test case reducer for MLIR, similar to LLVM's `bugpoint`. Reduces MLIR input to a minimal reproducer.

### Usage

```bash
# Write an interestingness test script
cat > test.sh << 'EOF'
mlir-opt -convert-vector-to-spirv $1 | grep "failed to materialize"
if [[ $? -eq 1 ]]; then
  exit 1
else
  exit 0
fi
EOF

# Run the reducer
mlir-reduce input.mlir -reduction-tree='traversal-mode=0 test=./test.sh'
```

### Reduction Strategies

1. **Operation elimination**: Directly removes operations
2. **Rewrite to simpler forms**: Rewrites types/operations to simpler variants
3. **Built-in optimization passes**: Runs passes like Symbol-DCE

### Custom Reduction Patterns

```c++
#include "mlir/Reducer/ReductionPatternInterface.h"

struct MyReductionPatternInterface : public DialectReductionPatternInterface {
  MyReductionPatternInterface(Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {};

  void populateReductionPatterns(RewritePatternSet &patterns) const final {
    populateMyReductionPatterns(patterns);
  }
};
```

### Building Custom mlir-reduce

```c++
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerMyDialects(registry);
  MLIRContext context(registry);
  return failed(mlirReduceMain(argc, argv, context));
}
```

## mlir-rewrite

### Overview

Tool to simplify rewriting `.mlir` files. Still in early development.

### simple-rename

Rename substrings in operations:

```bash
mlir-rewrite input.mlir -o output.mlir --simple-rename \
   --simple-rename-op-name="test.concat" \
   --simple-rename-match="axis" \
   --simple-rename-replace="bxis"
```

## mlir-tblgen

### Overview

TableGen driver for MLIR, generating C++ code from ODS definitions.

### Usage

```bash
# Generate operation declarations
mlir-tblgen -gen-op-decls Ops.td -I /path/to/include

# Generate operation definitions
mlir-tblgen -gen-op-defs Ops.td -I /path/to/include

# Generate dialect declarations
mlir-tblgen -gen-dialect-decls Dialect.td -I /path/to/include

# Generate documentation
mlir-tblgen -gen-op-doc Ops.td -I /path/to/include

# Generate enum declarations
mlir-tblgen -gen-enum-decls Enums.td -I /path/to/include

# Generate type definitions
mlir-tblgen -gen-typedef-defs Types.td -I /path/to/include

# Generate attribute definitions
mlir-tblgen -gen-attrdef-defs Attrs.td -I /path/to/include

# Generate pass declarations
mlir-tblgen -gen-pass-decls Passes.td -I /path/to/include

# Generate pass documentation
mlir-tblgen -gen-pass-doc Passes.td -I /path/to/include
```

### Common Generation Actions

| Action | Description |
|--------|-------------|
| `-gen-op-decls` | Operation class declarations |
| `-gen-op-defs` | Operation class definitions |
| `-gen-dialect-decls` | Dialect class declarations |
| `-gen-op-doc` | Operation documentation (Markdown) |
| `-gen-enum-decls` | Enum declarations |
| `-gen-enum-defs` | Enum definitions |
| `-gen-typedef-decls` | Type declarations |
| `-gen-typedef-defs` | Type definitions |
| `-gen-attrdef-decls` | Attribute declarations |
| `-gen-attrdef-defs` | Attribute definitions |
| `-gen-pass-decls` | Pass declarations |
| `-gen-pass-doc` | Pass documentation |
| `-gen-rewriters` | DRR pattern definitions |

## mlir-pdll

### Overview

PDLL compiler, translates PDLL patterns to PDL MLIR or C++ code.

### Usage

```bash
# Compile PDLL to PDL MLIR
mlir-pdll patterns.pdll -o output.mlir

# Compile PDLL to C++
mlir-pdll patterns.pdll -o output.cpp

# With include paths
mlir-pdll -I /path/to/include patterns.pdll -o output.mlir
```

## mlir-query

### Overview

Interactive query tool for MLIR operations.

### Usage

```bash
# Start interactive query
mlir-query input.mlir

# Query commands
mlir-query> match "arith.addi"
mlir-query> match "func.func"
mlir-query> help
mlir-query> quit
```

## mlir-irdl-to-cpp

### Overview

Converts IRDL dialect definitions to C++ code.

### Usage

```bash
mlir-irdl-to-cpp dialect.mlir -o dialect.cpp
```

## LSP Server Design

All MLIR LSP servers share the same architecture:

```
┌─────────────────┐    JSON-RPC    ┌──────────────────┐
│  Language Client │ ◄──────────► │  Language Server  │
│  (IDE/Editor)   │   stdin/out   │                   │
└─────────────────┘               │  ┌──────────────┐ │
                                  │  │ JSONTransport│ │
                                  │  └──────┬───────┘ │
                                  │         │         │
                                  │  ┌──────▼───────┐ │
                                  │  │MessageHandler│ │
                                  │  └──────┬───────┘ │
                                  │         │         │
                                  │  ┌──────▼───────┐ │
                                  │  │  LSPServer   │ │
                                  │  └──────┬───────┘ │
                                  │         │         │
                                  │  ┌──────▼───────┐ │
                                  │  │MLIRServer    │ │
                                  │  │PDLLServer    │ │
                                  │  │TblgenServer  │ │
                                  │  └──────────────┘ │
                                  └──────────────────┘
```

### Components

1. **Communication and Transport**: `JSONTransport` handles JSON-RPC over stdin/stdout
2. **Language Server Protocol**: `LSPServer` interprets LSP messages and forwards to language-specific server
3. **Language-Specific Server**: Implements actual queries (parsing, definitions, references, etc.)
