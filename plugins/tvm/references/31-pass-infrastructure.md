# 31 — Pass Infrastructure

## Overview

TVM's pass infrastructure provides a composable framework for transforming IRModules and functions. It follows the standard compiler pass design with module-level and function-level passes.

---

## Pass Base Classes

### tvm.transform.Pass
Abstract base class for all passes. Every pass:
- Takes an IRModule as input
- Returns a transformed IRModule
- Can be configured via PassContext

### tvm.transform.ModulePass
Transforms the entire IRModule:

```python
@tvm.transform.module_pass(opt_level=2, name="my_module_pass")
def my_module_pass(mod, ctx):
    # Transform mod
    return new_mod

# Apply
mod = my_module_pass(mod)
```

### tvm.transform.FunctionPass
Applied to each function in the module:

```python
@tvm.transform.function_pass(opt_level=2, name="my_func_pass")
def my_func_pass(func, mod, ctx):
    # Transform individual function
    return new_func

# Apply — automatically iterates over all functions
mod = my_func_pass(mod)
```

### tvm.transform.Sequential
Executes passes in sequence:

```python
pipeline = tvm.transform.Sequential([
    pass1,
    pass2,
    pass3,
])
mod = pipeline(mod)
```

---

## PassContext

PassContext configures pass behavior and provides runtime settings.

### Basic Usage

```python
with tvm.transform.PassContext(opt_level=3):
    mod = pipeline(mod)
```

### Configuration Options

```python
with tvm.transform.PassContext(
    opt_level=3,              # Optimization level (0-3)
    config={
        "tirx.UnrollLoop": {
            "auto_max_step": 10,
        },
        "relax.backend.use_cublas": True,
        "relax.backend.use_cutlass": True,
    },
    required_pass=["FuseOps", "LegalizeOps"],  # Only run these passes
    disabled_pass=["UnrollLoop"],               # Skip these passes
    traceback=True,                              # Enable pass traceback
    make_traceable=["FuseOps"],                  # Make specific passes traceable
):
    mod = pipeline(mod)
```

### PassContext.current()
Get the current PassContext from within a pass:

```python
ctx = tvm.transform.PassContext.current()
opt_level = ctx.opt_level
config = ctx.config
```

### Per-Pass Configuration
Each pass reads its configuration from `PassContext.config` using the pass name as key:

```python
# Configure tirx.UnrollLoop
with tvm.transform.PassContext(config={
    "tirx.UnrollLoop": {"auto_max_step": 10},
}):
    mod = tirx.transform.UnrollLoop()(mod)
```

---

## Pass Registration

### Module-level Pass
```python
@tvm.transform.module_pass(opt_level=2, name="my_pass")
def my_pass(mod, ctx):
    """Module-level pass documentation."""
    # Implementation
    return new_mod

# Apply
mod = my_pass(mod)
```

### Function-level Pass
```python
@tvm.transform.function_pass(opt_level=2, name="my_func_pass")
def my_func_pass(func, mod, ctx):
    """Function-level pass documentation."""
    # Implementation
    return new_func

# Apply
mod = my_func_pass(mod)
```

### TIR Function Pass
```python
@tvm.tirx.transform.prim_func_pass(opt_level=2, name="my_tir_pass")
def my_tir_pass(func, mod, ctx):
    """TIR PrimFunc pass documentation."""
    return new_func
```

---

## Pipeline Composition

### Built-in Pipelines

```python
from tvm import relax

# Zero pipeline (default, fast compilation)
mod = relax.get_pipeline("zero")(mod)

# Static shape tuning (MetaSchedule)
mod = relax.get_pipeline("static_shape_tuning")(mod)
```

### Custom Pipelines

```python
from tvm import transform

def my_pipeline():
    return transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
        relax.transform.DeadCodeElimination(),
        relax.transform.SimplifyExpr(),
    ])

mod = my_pipeline()(mod)
```

### Conditional Pipeline
```python
def adaptive_pipeline(target):
    passes = [
        relax.transform.LegalizeOps(),
        relax.transform.FuseOps(),
    ]
    if target.kind == "cuda":
        passes.append(relax.transform.FuseOpsByPattern(cutlass_patterns))
    passes.extend([
        relax.transform.FuseTIR(),
        relax.transform.DeadCodeElimination(),
    ])
    return transform.Sequential(passes)
```

---

## Pass Instrument

### Overview
PassInstrument allows hooking into pass execution for monitoring, profiling, and debugging.

### Custom Instrument

```python
class MyInstrument(tvm.transform.PassInstrument):
    def run_before_pass(self, mod, info):
        print(f"Before: {info.name}")

    def run_after_pass(self, mod, info):
        print(f"After: {info.name}")

with tvm.transform.PassContext(instruments=[MyInstrument()]):
    mod = pipeline(mod)
```

### Pass Timing
```python
# Enable pass timing
with tvm.transform.PassContext(traceback=True):
    mod = pipeline(mod)
```

### Debug Instrument
```python
class DebugInstrument(tvm.transform.PassInstrument):
    def __init__(self, target_pass_name):
        self.target = target_pass_name

    def run_before_pass(self, mod, info):
        if info.name == self.target:
            print(f"=== Before {info.name} ===")
            mod.show()

    def run_after_pass(self, mod, info):
        if info.name == self.target:
            print(f"=== After {info.name} ===")
            mod.show()

with tvm.transform.PassContext(instruments=[DebugInstrument("FuseOps")]):
    mod = pipeline(mod)
```

---

## Debug Passes

### PrintIR
Print IR at a specific point in the pipeline:

```python
pipeline = tvm.transform.Sequential([
    pass1,
    tvm.transform.PrintIR(name="after_pass1"),
    pass2,
])
```

### VerifySSA
Verify SSA (Single Static Assignment) form:

```python
mod = tvm.transform.VerifySSA()(mod)
```

---

## Pass Categories

### Relax Passes
All in `tvm.relax.transform`:
- LegalizeOps, FuseOps, FuseTIR, FuseOpsByPattern
- FoldConstant, SimplifyExpr, DeadCodeElimination
- DecomposeOpsForInference, DecomposeOpsForTraining
- CanonicalizeBindings, SimplifyReshape
- ToNonDataflow, VMBuiltinLower
- StaticPlanBlockMemory, RunCodegen
- And many more (see [Chapter 05](05-relax-transformations.md))

### TIR Passes
All in `tvm.tirx.transform`:
- FlattenBuffer, LowerIntrin, VectorizeLoop
- StorageRewrite, UnrollLoop, Simplify
- MakePackedAPI, DecorateDeviceScope
- And many more (see [Chapter 12](12-tir-transformations.md))

### Schedulable TIR Passes
In `tvm.s_tir.transform`:
- RenormalizeSplitPattern
- ApplyPass

---

## Best Practices

1. **Use PassContext** for configuration rather than modifying pass code
2. **Keep passes small** and composable
3. **Name passes** clearly for debugging
4. **Set opt_level** appropriately:
   - 0: No optimization (debug)
   - 1: Basic optimization
   - 2: Standard optimization
   - 3: Aggressive optimization
5. **Use Sequential** for predictable ordering
6. **Debug with PrintIR** and PassInstrument
