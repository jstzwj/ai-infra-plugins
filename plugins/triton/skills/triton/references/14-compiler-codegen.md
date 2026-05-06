# Chapter 14: Compiler Code Generation (`triton.compiler.code_generator`)

The code generator converts Python AST (Abstract Syntax Tree) to Triton MLIR IR.

## AST-to-MLIR Pipeline

```
Python Function → ast.parse() → CodeGenerator.visit() → MLIR Module (TTIR)
```

## CodeGenerator Class

The main visitor that walks Python AST and emits MLIR operations.

### Construction
```python
class CodeGenerator(ast.NodeVisitor):
    def __init__(self, context, is_gluon, options, codegen_fns, module_map,
                 file_name=None, begin_line=0, begin_col=0):
        self.context = context        # MLIR context
        self.is_gluon = is_gluon      # Gluon vs Triton mode
        self.builder = ...            # MLIR OpBuilder
        self.semantic = ...           # Semantic implementation
        self.gscope = {}              # Global scope
        self.lscope = {}              # Local scope
        self.options = options        # Compilation options
```

### Key Visitor Methods

| Python AST Node | MLIR Operation | Description |
|----------------|----------------|-------------|
| `FunctionDef` | `tt.func` | Define Triton function |
| `Assign` | SSA binding | Variable assignment |
| `BinOp(Add)` | `tt.add` | Addition |
| `BinOp(Sub)` | `tt.sub` | Subtraction |
| `BinOp(Mult)` | `tt.mul` | Multiplication |
| `BinOp(Div)` | `tt.fdiv` | Division |
| `BinOp(FloorDiv)` | `tt.fdiv` | Floor division |
| `BinOp(Mod)` | `tt.mod` | Modulo |
| `BoolOp(And)` | `tt.logical_and` | Logical AND |
| `BoolOp(Or)` | `tt.logical_or` | Logical OR |
| `Compare(Eq)` | `tt.compare_eq` | Equality |
| `Compare(NotEq)` | `tt.compare_ne` | Inequality |
| `Compare(Lt)` | `tt.compare_lt` | Less than |
| `If` | `scf.if` | Conditional |
| `For` | `scf.for` | Loop (range-based) |
| `While` | `scf.while` | While loop |
| `Call(tl.load)` | `tt.load` | Memory load |
| `Call(tl.store)` | `tt.store` | Memory store |
| `Call(tl.dot)` | `tt.dot` | Matrix multiply |
| `Call(tl.arange)` | `tt.make_range` | Range tensor |
| `Return` | `tt.return` | Return from function |
| `Subscript` | Index/slice | Tensor indexing |
| `UnaryOp(USub)` | `tt.sub(0, x)` | Negation |
| `UnaryOp(Not)` | `tt.logical_not` | Logical NOT |

### Function Mangling

Functions are mangled based on argument types for specialization:

```python
def mangle_fn(name, arg_tys, caller_context):
    # Generates unique names for different type specializations
    # e.g., add_kernel_int32, add_kernel_float32
```

### ASTFunction

Represents a function's type information:

```python
class ASTFunction:
    ret_types: list          # Return types
    arg_types: list          # Argument types
    attrs: dict              # Function attributes

    def serialize(self, builder):
        # Create MLIR function type

    @staticmethod
    def deserialize(fn):
        # Create template from MLIR function
```

## Main Entry Point

### `ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=None)`

Converts a Triton function to TTIR MLIR module.

```python
from triton.compiler.code_generator import ast_to_ttir

module = ast_to_ttir(
    fn=kernel_function,
    src=ASTSource(...),
    context=mlir_context,
    options=compilation_options,
    codegen_fns=backend_codegen_fns,
    module_map=backend_module_map,
)
```

**Process:**
1. Parse function source to AST
2. Extract function signature and parameters
3. Create MLIR builder
4. Create `tt.func` operation
5. Visit each AST node and emit MLIR operations
6. Handle control flow (loops, conditionals)
7. Return MLIR module

## Identifier Validation

```python
def check_identifier_legality(name, type):
    """Validates that identifiers don't conflict with Python builtins."""
    # Reserved names: 'range', 'static_range', 'constexpr', etc.
```

## Error Handling

The code generator provides detailed error messages with source location:

```python
class CompilationError(Exception):
    src: ASTSource     # Source code
    node: ast.AST      # AST node where error occurred

    def _format_message(self):
        # Formats error with line number and context
```

## CompileTimer

Tracks timing for compilation stages:

```python
timer = CompileTimer()
timer.finished_ir_initialization()
timer.stage_finished("ttir")
timer.stage_finished("ttgir")
timer.stage_finished("llir")
timer.end()  # Returns CompileTimes
```
