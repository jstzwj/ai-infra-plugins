# MLIR Bytecode Format, Diagnostics & Tracing

## Bytecode Format

### Overview

MLIR supports a compact binary serialization format (bytecode) as an alternative to the textual assembly format.

### Writing Bytecode

```c++
// Write module to bytecode file
if (failed(writeBytecodeToFile(module, outputFile)))
  return failure();

// With options
BytecodeWriterConfig config;
config.setDesiredChecksum(hash);
writeBytecodeToFile(module, outputFile, config);
```

### Reading Bytecode

```c++
// Parse bytecode from file
OwningOpRef<ModuleOp> module = parseBytecodeSource(fileSource, context);
```

### Bytecode Versioning

Dialects can handle versioning via `BytecodeDialectInterface`:

```c++
struct MyDialectBytecodeInterface : public BytecodeDialectInterface {
  // Write version
  void writeVersion(DialectBytecodeWriter &writer) const override;

  // Read version
  FailureOr<std::unique_ptr<DialectVersion>>
  readVersion(DialectBytecodeReader &reader) const override;

  // Upgrade from old version
  LogicalResult upgradeFromVersion(DialectVersion &version,
                                    Attribute &attr) const override;

  // Custom type/attribute encoding
  LogicalResult readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type, DialectBytecodeWriter &writer) const override;
  LogicalResult readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr, DialectBytecodeWriter &writer) const override;
};
```

## Diagnostics

### Diagnostic Engine

```c++
MLIRContext ctx;
DiagnosticEngine &engine = ctx.getDiagEngine();

// Register handler
engine.registerHandler([](Diagnostic &diag) {
  llvm::errs() << diag.getLocation() << ": ";
  switch (diag.getSeverity()) {
    case DiagnosticSeverity::Error: llvm::errs() << "error: "; break;
    case DiagnosticSeverity::Warning: llvm::errs() << "warning: "; break;
    case DiagnosticSeverity::Remark: llvm::errs() << "remark: "; break;
    case DiagnosticSeverity::Note: llvm::errs() << "note: "; break;
  }
  for (auto &note : diag.getNotes())
    llvm::errs() << note << "\n";
});
```

### Emitting Diagnostics

```c++
// From operation
op->emitError("something went wrong");
op->emitWarning("suspicious pattern");
op->emitRemark("optimization applied");
op->emitOpError("invalid operand type");

// With location
auto loc = op->getLoc();
emitError(loc, "message");
emitWarning(loc, "message");
emitRemark(loc, "message");

// In-flight diagnostics (chain notes)
op->emitError() << "main error message"
                << attachNote(otherLoc) << "additional context";
```

### Location Types

```mlir
loc("file.mlir":4:12)           // File/line/column
loc(fused<"name">[loc1, loc2])  // Fused
loc(callsite(loc0 at loc1))     // Call site
loc(unknown)                     // Unknown
loc(name<"name">)               // Named
```

### Diagnostic Severity

| Severity | Description |
|----------|-------------|
| Error | Compilation error (must fix) |
| Warning | Potential issue (should fix) |
| Remark | Informational (optimization note) |
| Note | Attached note to another diagnostic |

## Remarks

```c++
// Enable remarks
context.printOpOnDiagnostic(true);
context.printStackTraceOnDiagnostic(true);

// Emit remarks from passes
op->emitRemark() << "Applied transformation X";
```

## Action Tracing

### Overview

Action tracing allows recording and replaying compiler actions for debugging.

### Enable Tracing

```c++
// Enable action tracing
context.executeAction<PrintIRAction>("action-trace");
```

### Action Handler

```c++
// Register action handler
ctx.registerActionHandler([](function_ref<void()> action,
                              ArrayRef<IRUnit> irUnits,
                              Action::Type type) {
  llvm::errs() << "Action: " << type.getStringRef() << "\n";
  action();  // Execute the action
});
```

### Debugger Integration

MLIR provides GDB/LLDB scripts for debugging:

```bash
# Load MLIR LLDB scripts
command script import /path/to/mlir/utils/lldb-scripts/mlirDataFormatters.py
```
