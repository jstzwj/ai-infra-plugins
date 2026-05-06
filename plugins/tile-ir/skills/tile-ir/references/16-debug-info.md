# Chapter 9: Debug Information

## Table of Contents

1. [Overview](#1-overview)
2. [Purpose and Design Philosophy](#2-purpose-and-design-philosophy)
3. [Supported Debugging Features](#3-supported-debugging-features)
4. [Location Information](#4-location-information)
5. [Location Attribute: #cuda_tile.di_loc](#5-location-attribute-cuda_tiledi_loc)
6. [CallSiteLoc](#6-callsiteloc)
7. [Location Type Support Table](#7-location-type-support-table)
8. [Options for Generating Scope Metadata](#8-options-for-generating-scope-metadata)
9. [Trade-offs Between Approaches](#9-trade-offs-between-approaches)
10. [SynthesizeDebugInfoScopes Pass](#10-synthesizedebuginfoscopes-pass)
11. [Scope Metadata Types](#11-scope-metadata-types)
12. [Complete Example with All Debug Info Attributes](#12-complete-example-with-all-debug-info-attributes)
13. [Best Practices](#13-best-practices)
14. [Limitations and Known Issues](#14-limitations-and-known-issues)

---

## 1. Overview

Debug information in Tile IR provides the mechanism to map generated GPU code back to
the original source program. This mapping is essential for developers who need to set
breakpoints, step through control flow, and inspect the structure of their tile-based
programs during development. The debug information system is designed around the MLIR
location infrastructure and extends it with Tile IR-specific scope metadata.

The debug information subsystem operates at two levels:

1. **Location tracking** -- records file, line, and column positions for every operation
   in the Tile IR module, enabling source-level correlation.

2. **Scope metadata** -- describes the lexical structure of the program (compile units,
   files, subprograms, lexical blocks) so that debuggers can present the correct call
   stack and variable scoping information.

---

## 2. Purpose and Design Philosophy

The primary purpose of Tile IR debug information is to bridge the semantic gap between
the high-level tile-based programming model and the low-level GPU machine code that
executes on hardware. Without debug information, a developer confronted with a crash
or incorrect result would see only raw PTX or SASS instructions with no way to
correlate them back to the source.

### Design Goals

- **Minimal overhead**: Debug metadata should not significantly increase bytecode size
  or compilation time. All debug sections are optional in the binary format.

- **MLIR compatibility**: Tile IR debug info builds on MLIR's `Location` infrastructure,
  ensuring interoperability with MLIR-based toolchains and compilers.

- **Control-flow focus**: Debug information supports breakpoints, stepping, and stack
  frame inspection. It does NOT support value inspection of tile contents at runtime,
  since tiles are virtual constructs that may not map to a single physical storage
  location.

- **Incremental adoption**: Compilers can emit location information without scope
  metadata, or emit both. The `SynthesizeDebugInfoScopes` pass can automatically
  generate scope metadata from raw location data.

---

## 3. Supported Debugging Features

The following table summarizes the debugging features supported by Tile IR debug
information:

| Feature | Supported | Notes |
|---------|-----------|-------|
| Breakpoints | Yes | Set at source lines mapped via `#cuda_tile.di_loc` |
| Source-level stepping | Yes | Step into, over, out of tile block functions |
| Stack frame inspection | Yes | Control flow frames only |
| Variable value inspection | No | Tile contents are virtual; no runtime mapping |
| Memory inspection | Partial | Global memory addresses can be inspected |
| Conditional breakpoints | Yes | Based on tile block coordinates or loop indices |
| Parallel thread selection | Yes | Select tile block by (x, y, z) coordinate |

### What is NOT Supported

- **Value inspection**: You cannot inspect the values held in tile registers at
  runtime. Tiles are abstract multi-dimensional fragments that may be distributed
  across physical registers and shared memory in complex ways.

- **Data race detection**: Debug info does not provide tools for detecting memory
  races between tile blocks. Use the memory model's token-based ordering and the
  formal happens-before rules for reasoning about races.

- **Performance profiling**: Debug information is separate from profiling
  infrastructure. Use CUDA profiling tools (Nsight Compute, Nsight Systems) for
  performance analysis.

---

## 4. Location Information

Location information associates each Tile IR operation with a position in the original
source file. The location is encoded as an attribute on the operation and is stored in
the optional Debug Section (section ID `0x03`) of the Tile IR bytecode.

### Location Structure

Each location record contains:

| Field | Type | Description |
|-------|------|-------------|
| `file` | string reference | Index into the string section for the source file path |
| `line` | u32 | 1-based line number in the source file |
| `column` | u32 | 1-based column number in the source line |
| `scope` | metadata reference (optional) | Reference to the enclosing scope metadata node |

### Location in Bytecode

In the binary format, locations are stored as part of the Debug Section:

```
debug_section {
  location_table: location_record[]
  scope_metadata: metadata_node[]
}

location_record {
  file_id:    varuint    // index into string section
  line:       varuint    // 1-based line number
  column:     varuint    // 1-based column number
  scope_id:   varuint    // 0 = no scope, else index into scope_metadata
}
```

Each operation in the Function Table can optionally reference a location record by
index. When the Debug Section is absent, all operations have unknown locations.

---

## 5. Location Attribute: #cuda_tile.di_loc

The `#cuda_tile.di_loc` attribute is the primary mechanism for attaching location
information to Tile IR operations in the textual (MLIR-based) representation. It is
defined as a dialect attribute under the `cuda_tile` namespace.

### Syntax

```
#cuda_tile.di_loc<file: "path/to/source.tile", line: N, column: M>
#cuda_tile.di_loc<file: "path/to/source.tile", line: N, column: M, scope: @metadata_id>
```

### Usage Example

```
// An addf operation with location information
%result = addf %a, %b rounding<nearest_even> : tile<128xf32>
    loc(#cuda_tile.di_loc<file: "gemm.tile", line: 42, column: 15>)

// A for loop with location and scope metadata
%sum = for %i in (%lo to %hi, step %step) : tile<i32>
    iter_values(%acc = %init) -> (tile<128xf32>) {
    %new = addf %acc, %elem rounding<nearest_even> : tile<128xf32>
        loc(#cuda_tile.di_loc<file: "gemm.tile", line: 44, column: 20,
            scope: @sp_gemm_kernel>)
    continue %new : tile<128xf32>
        loc(#cuda_tile.di_loc<file: "gemm.tile", line: 45, column: 5,
            scope: @sp_gemm_kernel>)
} loc(#cuda_tile.di_loc<file: "gemm.tile", line: 43, column: 12,
      scope: @sp_gemm_kernel>)
```

### Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `file` | Yes | string | Source file path, relative to a compilation root or absolute |
| `line` | Yes | integer | 1-based line number |
| `column` | Yes | integer | 1-based column number (byte offset, not character) |
| `scope` | No | metadata ref | Reference to a scope metadata node |

When the `scope` field is omitted, the operation is assumed to belong to the enclosing
lexical scope of the kernel body, but no formal scope chain is established. The
`SynthesizeDebugInfoScopes` pass can infer scopes in this case.

---

## 6. CallSiteLoc

`CallSiteLoc` is an MLIR-standard location type used to represent the site where a
function or kernel is invoked. In Tile IR, it is used when a tile kernel is launched
from a host program and the debugger needs to show the call site in the host code.

### Syntax

```
loc(callsite<"callee_location", "caller_location">)
```

### Usage Example

```
// The kernel entry with a CallSiteLoc indicating where it was launched
cuda_tile.module @my_module {
    entry @my_kernel(%a: tile<ptr<f32>>) {
        // kernel body
    } loc(loc(callsite<
        #cuda_tile.di_loc<file: "my_kernel.tile", line: 10, column: 1>,
        #cuda_tile.di_loc<file: "host.cu", line: 200, column: 5>
    >))
}
```

### Semantics

- The **callee location** (first argument) identifies the definition of the callee --
  in Tile IR, this is typically the `entry` declaration of the kernel.

- The **caller location** (second argument) identifies the call site in the host code
  where the kernel launch was initiated.

- When a debugger presents a stack frame for a tile block, the `CallSiteLoc` allows
  it to show both the kernel source position and the host code that launched it.

---

## 7. Location Type Support Table

Not all Tile IR operations are equally amenable to source-level debugging. The
following table classifies each operation category by the level of debug information
it supports:

| Operation Category | Location Tracking | Scope Metadata | Breakpoints | Stepping |
|--------------------|-------------------|----------------|-------------|----------|
| **Core** (broadcast, cat, constant, extract, iota, offset, permute, reduce, reshape, scan, select) | Full | Full | No | No |
| **Conversions** (bitcast, exti, ftof, ftoi, itof, trunci, int_to_ptr, ptr_to_int, ptr_to_ptr) | Full | Full | No | No |
| **Control Flow** (for, if, loop, break, continue, return, yield, assert) | Full | Full | Yes | Yes |
| **Memory** (load_ptr_tko, store_ptr_tko, load_view_tko, store_view_tko, atomic_cas_tko, atomic_rmw_tko) | Full | Full | No | No |
| **Floating-Point** (addf, mulf, subf, divf, mmaf, etc.) | Full | Full | No | No |
| **Integer** (addi, muli, subi, mmai, etc.) | Full | Full | No | No |
| **Views** (make_tensor_view, make_partition_view, get_tensor_shape, get_index_space_shape, assume) | Full | Full | No | No |
| **Module/Entry** (module, entry, global, get_global, get_num_tile_blocks, get_tile_block_id) | Full | Full | Yes | Yes |

### Key

- **Location Tracking**: The operation can carry `#cuda_tile.di_loc` attributes.
- **Scope Metadata**: The operation participates in scope chains.
- **Breakpoints**: A debugger can stop execution at this operation.
- **Stepping**: The debugger can step over, into, or past this operation.

### Notes

- Core, conversion, memory, arithmetic, and view operations are dataflow operations
  that execute atomically from the debugger's perspective. They support location
  tracking for error reporting but are not individually steppable.

- Control flow operations (`for`, `if`, `loop`, `break`, `continue`, `return`) are
  the primary targets for interactive debugging. The debugger can set breakpoints on
  the source lines containing these operations and step through them.

- Module-level items (`module`, `entry`, `global`) support breakpoints at the kernel
  entry point, allowing the debugger to stop when a specific tile block begins
  execution.

---

## 8. Options for Generating Scope Metadata

There are three strategies for generating debug scope metadata in a Tile IR compilation
pipeline. Each has different trade-offs in terms of compiler complexity, debug
information quality, and maintenance burden.

### Strategy 1: Frontend-Generated Scopes

The frontend compiler (e.g., a CuTeDSL-to-Tile IR compiler) generates complete scope
metadata as it emits Tile IR operations. Each operation receives a `#cuda_tile.di_loc`
with a `scope` field referencing the appropriate metadata node.

**Advantages**:
- Most accurate scope information
- Preserves original source structure faithfully
- No additional compiler passes needed

**Disadvantages**:
- Increases frontend complexity
- Frontend must track scope state during code generation
- Metadata can become stale if Tile IR is transformed

### Strategy 2: SynthesizeDebugInfoScopes Pass

A dedicated compiler pass (`SynthesizeDebugInfoScopes`) analyzes the flat location
information on operations and synthesizes the scope metadata tree. This pass is
invoked after all Tile IR transformations are complete.

**Advantages**:
- Frontend only needs to emit location attributes (no scope references)
- Scope metadata is always consistent with final IR structure
- Simple to implement in the frontend

**Disadvantages**:
- Synthesized scopes may not perfectly match source structure
- Adds a compilation pass
- Scope boundaries are inferred, not authoritatively stated

### Strategy 3: Hybrid Approach

The frontend generates scope metadata for top-level constructs (compile units, files,
subprograms) but omits fine-grained lexical block scopes. The `SynthesizeDebugInfoScopes`
pass then fills in the lexical blocks.

**Advantages**:
- Good balance of accuracy and simplicity
- Frontend controls important scope boundaries
- Pass handles routine nesting

**Disadvantages**:
- More complex than either pure approach
- Requires coordination between frontend and pass

---

## 9. Trade-offs Between Approaches

The following table summarizes the trade-offs between the three strategies:

| Criterion | Frontend-Generated | SynthesizeDebugInfoScopes | Hybrid |
|-----------|-------------------|---------------------------|--------|
| **Scope accuracy** | Highest | Medium | High |
| **Frontend complexity** | High | Low | Medium |
| **Pass complexity** | None | Medium | Medium |
| **Robustness to IR transforms** | Low (metadata can become stale) | High (always fresh) | Medium |
| **Bytecode size** | Same | Same | Same |
| **Compilation time** | Fastest (no pass) | Slightly slower | Slightly slower |
| **Maintenance burden** | Frontend team | Pass team | Shared |
| **Recommended for** | Production compilers with stable IR | Rapid prototyping, DSL compilers | Most production use |

### Recommendation

For most Tile IR producers, the **Hybrid approach** (Strategy 3) is recommended:
- Emit compile unit, file, and subprogram metadata from the frontend.
- Let `SynthesizeDebugInfoScopes` synthesize lexical block scopes.
- This gives the best trade-off between accuracy, simplicity, and robustness.

---

## 10. SynthesizeDebugInfoScopes Pass

The `SynthesizeDebugInfoScopes` pass is a Tile IR-to-Tile IR transformation pass that
takes a module with location attributes but incomplete or absent scope metadata, and
produces a module with a fully populated scope metadata tree.

### Pass Signature

```
SynthesizeDebugInfoScopes {
    input:  Module with #cuda_tile.di_loc (no scope or partial scope)
    output: Module with #cuda_tile.di_loc (full scope chains)
}
```

### Algorithm

1. **Collect all locations**: Scan all operations and collect unique (file, line,
   column) tuples from their `#cuda_tile.di_loc` attributes.

2. **Create Compile Unit and File nodes**: For each unique source file, create a
   `DICompileUnit` and `DIFile` metadata node.

3. **Create Subprogram nodes**: For each `entry` kernel declaration, create a
   `DISubprogram` node referencing the appropriate file and compile unit.

4. **Create Lexical Block nodes**: For nested regions (loop bodies, if-then/else
   branches), create `DILexicalBlock` nodes based on the line numbers of the
   containing operations. A new block is created when the line number decreases
   (indicating return from a nested region) or when nesting depth changes.

5. **Attach scope references**: Update all `#cuda_tile.di_loc` attributes to include
   the `scope` field referencing the appropriate metadata node.

### Python Example

The following Python-like pseudocode illustrates how a frontend compiler might emit
location information and how the `SynthesizeDebugInfoScopes` pass processes it:

```python
# === Frontend: Emit Tile IR with location info (no scope metadata) ===

class TileIRCodegen:
    """Frontend code generator that emits Tile IR with location attributes."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.operations = []

    def emit_operation(self, op: str, line: int, col: int) -> str:
        """Emit an operation with a location attribute (no scope)."""
        loc = f'#cuda_tile.di_loc<file: "{self.source_file}", ' \
              f'line: {line}, column: {col}>'
        return f"{op} loc({loc})"

    def emit_kernel(self, name: str, params: list, body_lines: list) -> str:
        """Emit a complete kernel with location tracking."""
        lines = [f'entry @{name}({", ".join(params)}) {{']

        for op_str, line, col in body_lines:
            lines.append(f'    {self.emit_operation(op_str, line, col)}')

        lines.append('}')
        return '\n'.join(lines)


# Example: Generate a simple vector addition kernel
codegen = TileIRCodegen("vector_add.tile")

kernel_body = [
    ("%offset = iota : tile<128xi32>",                          3,  5),
    ("%a_ptrs = offset %a_base, %offset : "
     "tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>", 4,  5),
    ("%a_val, %t1 = load_ptr_tko weak %a_ptrs : "
     "tile<128xptr<f32>> -> tile<128xf32>, token",              5,  5),
    ("%b_val, %t2 = load_ptr_tko weak %b_ptrs : "
     "tile<128xptr<f32>> -> tile<128xf32>, token",              6,  5),
    ("%result = addf %a_val, %b_val rounding<nearest_even> : "
     "tile<128xf32>",                                           7,  5),
    ("store_ptr_tko weak %c_ptrs, %result : "
     "tile<128xptr<f32>>, tile<128xf32> -> token",              8,  5),
    ("cuda_tile.return",                                         9,  1),
]

kernel_ir = codegen.emit_kernel(
    "vector_add_128",
    ["%a_ptr: tile<ptr<f32>>", "%b_ptr: tile<ptr<f32>>",
     "%c_ptr: tile<ptr<f32>>"],
    kernel_body
)
print(kernel_ir)


# === SynthesizeDebugInfoScopes Pass ===

class ScopeInfo:
    """Represents a scope node in the debug metadata tree."""
    def __init__(self, kind: str, fields: dict):
        self.kind = kind
        self.fields = fields
        self.children = []

    def __repr__(self):
        return f"{self.kind}({self.fields})"


class SynthesizeDebugInfoScopes:
    """Pass that synthesizes scope metadata from location information."""

    def __init__(self):
        self.metadata_nodes = {}  # id -> ScopeInfo
        self.next_id = 0

    def _alloc_id(self) -> int:
        """Allocate a new metadata node ID."""
        mid = self.next_id
        self.next_id += 1
        return mid

    def create_compile_unit(self, file_path: str) -> int:
        """Create a DICompileUnit metadata node for a source file."""
        cu_id = self._alloc_id()
        file_id = self.create_file(file_path)
        self.metadata_nodes[cu_id] = ScopeInfo("DICompileUnit", {
            "file": file_id,
            "is_optimized": True,
            "emission_kind": "full",
        })
        return cu_id

    def create_file(self, file_path: str) -> int:
        """Create a DIFile metadata node."""
        import os
        file_id = self._alloc_id()
        self.metadata_nodes[file_id] = ScopeInfo("DIFile", {
            "name": os.path.basename(file_path),
            "directory": os.path.dirname(file_path),
        })
        return file_id

    def create_subprogram(self, name: str, linkage_name: str,
                          file_id: int, cu_id: int, line: int) -> int:
        """Create a DISubprogram metadata node for a kernel."""
        sp_id = self._alloc_id()
        self.metadata_nodes[sp_id] = ScopeInfo("DISubprogram", {
            "name": name,
            "linkage_name": linkage_name,
            "file": file_id,
            "line": line,
            "scope_line": line,
            "compile_unit": cu_id,
        })
        return sp_id

    def create_lexical_block(self, scope_id: int, file_id: int,
                             line: int, column: int) -> int:
        """Create a DILexicalBlock metadata node."""
        lb_id = self._alloc_id()
        self.metadata_nodes[lb_id] = ScopeInfo("DILexicalBlock", {
            "scope": scope_id,
            "file": file_id,
            "line": line,
            "column": column,
        })
        # Register as child of enclosing scope
        self.metadata_nodes[scope_id].children.append(lb_id)
        return lb_id

    def process_module(self, operations: list) -> list:
        """
        Process a list of operations with location info, synthesizing scope
        metadata and returning updated operations with scope references.
        """
        # Step 1: Collect unique files
        files = {}
        for op, line, col in operations:
            # Extract file from location (simplified)
            files.setdefault("vector_add.tile", None)

        # Step 2: Create compile units and files
        compile_units = {}
        for fpath in files:
            cu_id = self.create_compile_unit(fpath)
            compile_units[fpath] = cu_id

        # Step 3: Create subprogram for the kernel
        file_id = self.metadata_nodes[compile_units["vector_add.tile"]].fields["file"]
        sp_id = self.create_subprogram(
            name="vector_add_128",
            linkage_name="vector_add_128",
            file_id=file_id,
            cu_id=compile_units["vector_add.tile"],
            line=2,  # kernel entry line
        )

        # Step 4: Create lexical blocks for nested regions
        # (In practice, this requires analyzing the nesting structure)
        current_scope = sp_id
        result = []
        for op, line, col in operations:
            scope = current_scope
            # Simple heuristic: create new lexical block for loop bodies
            if "for" in op:
                lb_id = self.create_lexical_block(
                    scope_id=sp_id,
                    file_id=file_id,
                    line=line,
                    column=col,
                )
                scope = lb_id
            result.append((op, line, col, scope))

        return result

    def emit_metadata(self) -> str:
        """Emit all metadata nodes as Tile IR attributes."""
        lines = []
        for mid, info in self.metadata_nodes.items():
            fields_str = ", ".join(
                f"{k} = {v}" for k, v in info.fields.items()
            )
            lines.append(
                f"cuda_tile.metadata @{info.kind.lower()}_{mid} "
                f"= #{info.kind}<{fields_str}>"
            )
        return "\n".join(lines)


# Run the pass
pass_instance = SynthesizeDebugInfoScopes()
scoped_ops = pass_instance.process_module(kernel_body)
metadata_ir = pass_instance.emit_metadata()
print("\n--- Generated Scope Metadata ---")
print(metadata_ir)
```

---

## 11. Scope Metadata Types

Tile IR defines four types of scope metadata nodes, modeled after the DWARF debugging
standard. Each metadata node is identified by a unique ID and stored in the Debug
Section of the bytecode.

### 11.1 Compile Unit (DICompileUnit)

A compile unit represents a single compilation (a single source file or translation
unit). It is the root of the scope metadata tree.

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | metadata ref | Yes | Reference to the `DIFile` node for this compilation |
| `is_optimized` | bool | Yes | Whether the compilation was optimized |
| `emission_kind` | enum | Yes | One of: `full`, `line_tables_only`, `none` |

#### Syntax

```
cuda_tile.metadata @cu_0 = #DICompileUnit<
    file = @file_0,
    is_optimized = true,
    emission_kind = "full"
>
```

#### Emission Kinds

| Kind | Description |
|------|-------------|
| `full` | Complete debug information: locations, scopes, variables |
| `line_tables_only` | Only line number tables; no scope or variable information |
| `none` | No debug information (equivalent to omitting the compile unit) |

The `emission_kind` controls how much debug information is preserved during
compilation. For production builds, `line_tables_only` is common; for development,
`full` is recommended.

### 11.2 File (DIFile)

A file metadata node describes a source file. Each compile unit references exactly one
file, but a file may be referenced by multiple compile units (e.g., in a system with
include files).

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Base name of the file (e.g., `"kernel.tile"`) |
| `directory` | string | Yes | Directory path (may be empty for current directory) |

#### Syntax

```
cuda_tile.metadata @file_0 = #DIFile<
    name = "gemm_kernel.tile",
    directory = "/home/user/projects"
>
```

#### Path Resolution

The full path to the source file is constructed as `directory + "/" + name`. If
`directory` is empty, only `name` is used. The path may be absolute or relative to a
compilation root specified externally (e.g., via a compiler flag).

### 11.3 Lexical Block (DILexicalBlock)

A lexical block represents a nested scope within a function, such as a loop body,
conditional branch, or an explicit block in the source language.

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scope` | metadata ref | Yes | The enclosing scope (another lexical block or a subprogram) |
| `file` | metadata ref | Yes | The file containing this lexical block |
| `line` | u32 | Yes | The source line where the lexical block begins |
| `column` | u32 | Yes | The source column where the lexical block begins |

#### Syntax

```
cuda_tile.metadata @block_0 = #DILexicalBlock<
    scope = @sp_gemm,
    file = @file_0,
    line = 25,
    column = 8
>
```

#### Nesting Rules

- Lexical blocks may be nested to arbitrary depth.
- The root scope of a kernel body is always a `DISubprogram`.
- Each control flow operation (`for`, `if`, `loop`) typically introduces a new
  lexical block for its body region.
- The `SynthesizeDebugInfoScopes` pass creates lexical blocks based on changes in
  source line numbers and operation nesting.

### 11.4 Subprogram (DISubprogram)

A subprogram represents a function or kernel in the source program. In Tile IR, each
`entry` kernel declaration corresponds to a `DISubprogram` metadata node.

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | metadata ref | Yes | The file containing this subprogram |
| `line` | u32 | Yes | The source line where the subprogram is declared |
| `name` | string | Yes | Human-readable name of the subprogram |
| `linkage_name` | string | Yes | Mangled/linker-visible name (in Tile IR, same as `name`) |
| `compile_unit` | metadata ref | Yes | The compile unit containing this subprogram |
| `scope_line` | u32 | Yes | The line where the subprogram's scope begins |

#### Syntax

```
cuda_tile.metadata @sp_gemm = #DISubprogram<
    file = @file_0,
    line = 10,
    name = "gemm_kernel",
    linkage_name = "gemm_kernel",
    compile_unit = @cu_0,
    scope_line = 10
>
```

#### Notes

- In Tile IR, there is no separate linkage name mangling scheme. The `linkage_name`
  is always identical to `name`.

- The `scope_line` field indicates the line at which the subprogram's formal scope
  begins, which may differ from the `line` field if the function signature spans
  multiple source lines.

- All operations within a kernel body that are not inside a nested lexical block
  belong to the subprogram's scope.

---

## 12. Complete Example with All Debug Info Attributes

The following example shows a complete Tile IR module with full debug information,
including compile unit, file, subprogram, lexical blocks, and location attributes on
every operation.

```
// ===================================================================
// Complete Tile IR Module with Debug Information
// Source file: /home/user/projects/gemm.tile
// ===================================================================

cuda_tile.module @gemm_debug_module {

    // --- Scope Metadata ---

    // File metadata: describes the source file
    cuda_tile.metadata @file_0 = #DIFile<
        name = "gemm.tile",
        directory = "/home/user/projects"
    >

    // Compile unit metadata: one per source file
    cuda_tile.metadata @cu_0 = #DICompileUnit<
        file = @file_0,
        is_optimized = true,
        emission_kind = "full"
    >

    // Subprogram metadata: one per kernel entry
    cuda_tile.metadata @sp_gemm = #DISubprogram<
        file = @file_0,
        line = 5,
        name = "gemm_kernel",
        linkage_name = "gemm_kernel",
        compile_unit = @cu_0,
        scope_line = 5
    >

    // Lexical block: the for loop body at line 15
    cuda_tile.metadata @block_loop = #DILexicalBlock<
        scope = @sp_gemm,
        file = @file_0,
        line = 15,
        column = 4
    >

    // Lexical block: the if-then body at line 18
    cuda_tile.metadata @block_if_then = #DILexicalBlock<
        scope = @block_loop,
        file = @file_0,
        line = 18,
        column = 8
    >

    // Lexical block: the if-else body at line 22
    cuda_tile.metadata @block_if_else = #DILexicalBlock<
        scope = @block_loop,
        file = @file_0,
        line = 22,
        column = 8
    >

    // --- Kernel Definition ---

    entry @gemm_kernel(
        %A_ptr: tile<ptr<f16>>,
        %B_ptr: tile<ptr<f16>>,
        %C_ptr: tile<ptr<f32>>,
        %M: tile<i32>,
        %N: tile<i32>,
        %K: tile<i32>,
        %stride_a: tile<i32>,
        %stride_b: tile<i32>,
        %stride_c: tile<i32>
    ) {
        // Line 6: Create tensor views
        %A_tv = make_tensor_view %A_ptr, shape=[%K, %M],
            strides=[%stride_a, 1]
            : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 6, column: 5,
                scope: @sp_gemm>)

        %B_tv = make_tensor_view %B_ptr, shape=[%N, %K],
            strides=[%stride_b, 1]
            : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 7, column: 5,
                scope: @sp_gemm>)

        %C_tv = make_tensor_view %C_ptr, shape=[%M, %N],
            strides=[%stride_c, 1]
            : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 8, column: 5,
                scope: @sp_gemm>)

        // Line 10: Create partition views
        %A_pv = make_partition_view %A_tv
            : partition_view<tile=(128x64),
              tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 10, column: 5,
                scope: @sp_gemm>)

        %B_pv = make_partition_view %B_tv
            : partition_view<tile=(64x128),
              tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 11, column: 5,
                scope: @sp_gemm>)

        %C_pv = make_partition_view %C_tv
            : partition_view<tile=(128x128),
              tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 12, column: 5,
                scope: @sp_gemm>)

        // Line 13: Get tile block coordinates
        %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 13, column: 5,
                scope: @sp_gemm>)

        // Line 14: Initialize accumulator
        %cst = constant <f32: 0.0> : tile<128x128xf32>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 14, column: 5,
                scope: @sp_gemm>)

        %i0 = constant <i32: 0> : tile<i32>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 14, column: 20,
                scope: @sp_gemm>)
        %i1 = constant <i32: 1> : tile<i32>
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 14, column: 30,
                scope: @sp_gemm>)

        // Line 15: K-dimension reduction loop
        %result = for %k in (%i0 to %K, step %i64) : tile<i32>
            iter_values(%acc = %cst) -> (tile<128x128xf32>) {

            // Line 16: Load A tile (inside loop block)
            %A_frag, %t1 = load_view_tko weak %A_pv[%bidx, %k]
                : partition_view<tile=(128x64),
                  tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>,
                  tile<i32> -> tile<128x64xf16>, token
                loc(#cuda_tile.di_loc<file: "gemm.tile", line: 16, column: 8,
                    scope: @block_loop>)

            // Line 17: Load B tile (inside loop block)
            %B_frag, %t2 = load_view_tko weak %B_pv[%k, %bidy]
                : partition_view<tile=(64x128),
                  tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>,
                  tile<i32> -> tile<64x128xf16>, token
                loc(#cuda_tile.di_loc<file: "gemm.tile", line: 17, column: 8,
                    scope: @block_loop>)

            // Line 18-19: Conditional computation (if-then)
            %cond = cmpi %k, %i0 cmp<gt> : tile<i32>
                loc(#cuda_tile.di_loc<file: "gemm.tile", line: 18, column: 8,
                    scope: @block_loop>)

            %new_acc = if %cond : tile<i1> {
                // Then branch: matrix multiply-accumulate
                %mma_result = mmaf %A_frag, %B_frag, %acc
                    : tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
                    loc(#cuda_tile.di_loc<file: "gemm.tile", line: 19, column: 12,
                        scope: @block_if_then>)
                continue %mma_result : tile<128x128xf32>
                    loc(#cuda_tile.di_loc<file: "gemm.tile", line: 20, column: 12,
                        scope: @block_if_then>)
            } else {
                // Else branch: just use loaded tiles (first iteration)
                %mma_init = mmaf %A_frag, %B_frag, %cst
                    : tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
                    loc(#cuda_tile.di_loc<file: "gemm.tile", line: 23, column: 12,
                        scope: @block_if_else>)
                continue %mma_init : tile<128x128xf32>
                    loc(#cuda_tile.di_loc<file: "gemm.tile", line: 24, column: 12,
                        scope: @block_if_else>)
            } -> tile<128x128xf32>
                loc(#cuda_tile.di_loc<file: "gemm.tile", line: 18, column: 8,
                    scope: @block_loop>)

        } loc(#cuda_tile.di_loc<file: "gemm.tile", line: 15, column: 5,
              scope: @sp_gemm>)

        // Line 28: Store result
        %t3 = store_view_tko weak %result, %C_pv[%bidx, %bidy]
            : tile<128x128xf32>,
              partition_view<tile=(128x128),
                tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>,
              tile<i32> -> token
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 28, column: 5,
                scope: @sp_gemm>)

        // Line 29: Return
        cuda_tile.return
            loc(#cuda_tile.di_loc<file: "gemm.tile", line: 29, column: 1,
                scope: @sp_gemm>)

    } loc(loc(callsite<
        #cuda_tile.di_loc<file: "gemm.tile", line: 5, column: 1>,
        #cuda_tile.di_loc<file: "host.cu", line: 150, column: 3>
    >))
}
```

### Metadata Tree Visualization

The scope metadata for the above example forms the following tree:

```
DICompileUnit(@cu_0)
  +-- DIFile(@file_0)
  +-- DISubprogram(@sp_gemm)
        +-- DILexicalBlock(@block_loop)        [for loop, line 15]
        |     +-- DILexicalBlock(@block_if_then)  [if-then, line 18]
        |     +-- DILexicalBlock(@block_if_else)  [if-else, line 22]
```

Each operation's `scope` field references the innermost containing node in this tree.

---

## 13. Best Practices

### When Emitting Debug Information

1. **Always emit locations**: Even without scope metadata, `#cuda_tile.di_loc` with
   file, line, and column is invaluable for error messages and crash diagnosis.

2. **Use consistent file paths**: Ensure all operations from the same source file use
   the same path string. The compiler may intern strings, but inconsistent paths will
   create duplicate `DIFile` nodes.

3. **Set emission_kind appropriately**: Use `full` during development and
   `line_tables_only` for production builds to reduce bytecode size.

4. **Prefer the hybrid approach**: Emit compile units, files, and subprograms from
   the frontend; let `SynthesizeDebugInfoScopes` handle lexical blocks.

### When Consuming Debug Information

1. **Handle missing debug sections gracefully**: The Debug Section is optional. A
   well-behaved debugger or tool should work correctly (with degraded functionality)
   when it is absent.

2. **Do not rely on value inspection**: Tile IR debug info does not support inspecting
   tile register values. Use print operations (`print_tko`) for outputting values.

3. **Use tile block coordinates for parallel selection**: When debugging, select a
   specific tile block by its (x, y, z) coordinates rather than trying to inspect
   all blocks simultaneously.

4. **Respect scope boundaries**: When presenting stack frames, use the scope metadata
   tree to determine the correct lexical nesting. Do not rely solely on line numbers,
   as transformations may reorder operations.

---

## 14. Limitations and Known Issues

### Current Limitations

1. **No variable metadata**: Tile IR does not currently define `DIVariable` or
   `DILocalVariable` metadata types. All tile values are anonymous registers and
   cannot be mapped back to named source variables.

2. **No type visualization**: There is no mechanism for describing how tile types
   should be displayed in a debugger. A `tile<128xf32>` may appear as an opaque value.

3. **No cross-tile-block debugging**: Debuggers can inspect individual tile blocks
   but cannot simultaneously present a coherent view of multiple tile blocks' state.

4. **Limited breakpoint granularity**: Breakpoints can be set at the operation level,
   but individual iterations of a `for` loop cannot be distinguished unless the loop
   variable is used in a condition.

5. **Scope metadata may be approximate**: When synthesized by the
   `SynthesizeDebugInfoScopes` pass, lexical block boundaries are inferred from line
   numbers and may not perfectly match the original source structure.

### Known Issues in Version 13.2

- Debug section strings are not deduplicated across modules, potentially increasing
  bytecode size for multi-module programs.

- The `SynthesizeDebugInfoScopes` pass may create unnecessary lexical blocks when
  source lines are non-monotonically ordered due to compiler transformations.

- `CallSiteLoc` is not preserved through certain optimization passes, causing the
  host-side call site information to be lost in optimized builds.
