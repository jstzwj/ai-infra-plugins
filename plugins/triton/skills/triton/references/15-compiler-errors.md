# Chapter 15: Compiler Errors

## Error Hierarchy

```
Exception
├── TritonError                          # Base Triton error
│   ├── CompilationError                 # Compile-time errors
│   │   ├── CompileTimeAssertionFailure  # static_assert failure
│   │   └── UnsupportedLanguageConstruct # Unsupported feature
│   ├── InterpreterError                 # Interpreter errors
│   ├── OutOfResources                   # GPU resource exhaustion
│   ├── PTXASError                       # PTX assembler errors
│   └── AutotunerError                   # Autotuning errors
```

## CompilationError

Raised during kernel compilation:

```python
class CompilationError(Exception):
    src: ASTSource       # Source code (may be None)
    node: ast.AST        # AST node where error occurred
    error_message: str   # Error description

    def _format_message(self):
        # Includes file name, line number, and source context
```

**Common causes:**
- Type mismatch in operations
- Invalid tensor shapes
- Unsupported operations
- Missing required arguments

## CompileTimeAssertionFailure

Raised when `tl.static_assert(condition)` fails:

```python
@triton.jit
def kernel(BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive")
```

## InterpreterError

Raised during interpreter execution (`TRITON_INTERPRET=1`):

```python
class InterpreterError(TritonError):
    error_message: str
```

## OutOfResources

Raised when GPU resources are exceeded:

```python
class OutOfResources(TritonError):
    required: int  # Required amount
    limit: int     # Hardware limit
    name: str      # Resource name

    def __str__(self):
        return (f"out of resource: {self.name}, "
                f"Required: {self.required}, "
                f"Hardware limit: {self.limit}. "
                f"Reducing block sizes or `num_stages` may help.")
```

**Common resources:**
- Shared memory
- Registers
- Threads per block

## PTXASError

Raised when PTX assembly fails:

```python
class PTXASError(TritonError):
    error_message: str
```

## AutotunerError

Raised during autotuning:

```python
class AutotunerError(TritonError):
    error_message: str
```

## Error Handling Best Practices

1. **Use static_assert for compile-time checks:**
```python
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    tl.static_assert(BLOCK_SIZE >= 16, "BLOCK_SIZE must be >= 16")
    tl.static_assert(n > 0, "n must be positive")
```

2. **Use device_assert for runtime checks:**
```python
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.device_assert(mask.all(), "Some indices out of bounds")
```

3. **Handle resource errors by reducing block sizes:**
```python
try:
    kernel[grid](args, BLOCK_SIZE=2048)
except triton.OutOfResources:
    kernel[grid](args, BLOCK_SIZE=1024)  # Retry with smaller block
```

4. **Filter tracebacks for cleaner error messages:**
```python
# The compiler automatically filters internal frames
# Set TRITON_FRONT_END_DEBUGGING=1 to see full traceback
```
