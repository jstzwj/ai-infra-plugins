# Apache TVM - Contributing Guide

This reference provides a comprehensive guide for contributing to Apache TVM, covering code organization, development workflow, code style, code review, CI/CD, documentation, and community resources.

---

## 35.1 Code Organization

TVM's codebase is organized into several top-level directories, each serving a distinct purpose in the compilation stack.

### 35.1.1 C++ Code in src/

The `src/` directory contains the core C++ implementation of TVM. It is organized by subsystem:

```
src/
├── arith/               # Arithmetic analysis and simplification
│   ├── analyzer.cc
│   ├── canonical_simplify.cc
│   ├── const_int_bound.cc
│   ├── int_set.cc
│   ├── modular_set.cc
│   └── rewrite_simplify.cc
├── auto_scheduler/      # Auto-scheduling (Ansor)
│   ├── search_policy/
│   ├── cost_model/
│   └── feature.h
├── ir/                  # Common IR infrastructure
│   ├── attrFunctor.h
│   ├── diagnostic.cc
│   ├── env_func.cc
│   ├── module.cc
│   ├── source_map.cc
│   └── transform.cc
├── meta_schedule/       # MetaSchedule auto-tuning
│   ├── builder.cc
│   ├── cost_model.cc
│   ├── database.cc
│   ├── runner.cc
│   ├── search_strategy.cc
│   ├── space_generator.cc
│   └── task_scheduler.cc
├── node/                # TVM object system
│   ├── attr_registry_map.h
│   ├── reflection.cc
│   ├── repr_printer.cc
│   └── structural_equal.cc
├── printer/             # IR text printer
│   ├── doc.cc
│   ├── text_printer.cc
│   └── tvmscript_printer.cc
├── relay/               # Relay IR and compiler
│   ├── analysis/
│   ├── backend/
│   ├── collage/
│   ├── dataflow_pattern/
│   ├── op/
│   ├── qnn/
│   ├── transforms/
│   └── type/
├── runtime/             # Runtime system
│   ├── cuda/
│   ├── graph_executor/
│   ├── metal/
│   ├── micro/
│   ├── opencl/
│   ├── rocm/
│   ├── rpc/
│   ├── vulkan/
│   └── vm/
├── support/             # Utility libraries
│   ├── arena.h
│   ├── generic_arena.h
│   ├── pipe.h
│   ├── ring_buffer.h
│   ├── socket.h
│   └── str_escape.h
├── target/              # Target descriptions and codegen
│   ├── source/
│   ├── spirv/
│   ├── stackvm/
│   ├── target.cc
│   └── target_info.cc
├── te/                  # Tensor Expression language
│   ├── operation/
│   ├── schedule/
│   ├── schedule_pass/
│   └── tensor.cc
├── tir/                 # Tensor Intermediate Representation
│   ├── analysis/
│   ├── schedule/
│   ├── schedule_pass/
│   ├── transforms/
│   └── usmp/
└── TVMRegisterAllTVMGlobalFuncs() in api_registry.cc
```

### 35.1.2 Python Code in python/tvm/

The Python frontend is organized to mirror the C++ structure:

```
python/tvm/
├── __init__.py
├── arith/               # Arithmetic analysis (Python bindings)
├── auto_scheduler/      # Auto-scheduling Python API
├── autotvm/             # AutoTVM tuning
├── contrib/             # External integrations
│   ├── cuda_graph.py
│   ├── cublas.py
│   ├── cufft.py
│   ├── miopen.py
│   ├── onnx.py
│   ├── pickle_memoize.py
│   ├── tflite_runtime.py
│   └── torch.py
├── driver/              # Build driver API
├── exec/                # Executable scripts (rpc_server, etc.)
├── ir/                  # IR module and base types
│   ├── diagnostic.py
│   ├── function.py
│   ├── module.py
│   └── base.py
├── meta_schedule/       # MetaSchedule Python API
├── micro/               # MicroTVM (embedded devices)
├── ndarray.py           # NDArray implementation
├── parser/              # IR parser
├── printing/            # Printer utilities
├── relax/               # Relax IR
│   ├── analysis/
│   ├── backend/
│   ├── dpl/
│   ├── frontend/
│   ├── op/
│   ├── transform/
│   └── vm.py
├── relay/               # Relay compiler
│   ├── analysis/
│   ├── backend/
│   ├── dataflow_pattern/
│   ├── frontend/
│   ├── op/
│   ├── qnn/
│   ├── testing/
│   └── transform/
├── rpc/                 # Remote procedure call
├── runtime/             # Runtime Python bindings
├── script/              # TVMScript
│   ├── parser/
│   └── printer/
├── target/              # Target descriptions
├── te/                  # Tensor Expression
├── testing/             # Testing utilities
├── tir/                 # TIR Python bindings
│   ├── analysis/
│   ├── schedule/
│   ├── transform/
│   └── usmp/
└── topi/                # Tensor Operator Inventory
    ├── cuda/
    ├── generic/
    ├── image/
    ├── nn/
    ├── rocm/
    ├── testing/
    └── x86/
```

### 35.1.3 Headers in include/tvm/

Public C++ headers define the TVM API. The header structure mirrors the source organization:

```
include/tvm/
├── arith/               # Arithmetic analysis headers
│   ├── analyzer.h
│   ├── bound.h
│   ├── int_set.h
│   └── solver.h
├── auto_scheduler/
├── base.h               # Base definitions
├ data/                   # Data structure headers
│   └── array.h
├── ir/                  # IR headers
│   ├── attrs.h
│   ├── diagnostic.h
│   ├── env_func.h
│   ├── function.h
│   ├── module.h
│   ├── source_map.h
│   └── transform.h
├── meta_schedule/
├── node/                # Object system headers
│   ├── attr_registry_map.h
│   ├── node.h
│   ├── reflection.h
│   └── structural_equal.h
├── relay/               # Relay headers
│   ├── adt.h
│   ├── attrs/
│   ├── analysis.h
│   ├── dataflow_pattern.h
│   ├── expr.h
│   ├── feature.h
│   ├── function.h
│   ├── op.h
│   ├── op_attr_types.h
│   ├── pattern_functor.h
│   ├── transform.h
│   └── type.h
├── runtime/             # Runtime headers
│   ├── container/
│   ├── cuda/
│   ├── micro/
│   ├── vm/
│   ├── c_runtime_api.h
│   ├── data_type.h
│   ├── device_api.h
│   ├── logging.h
│   ├── memory.h
│   ├── module.h
│   ├── ndarray.h
│   ├── object.h
│   ├── packed_func.h
│   ├── registry.h
│   └── threading_backend.h
├── target/              # Target headers
│   ├── target.h
│   ├── target_info.h
│   └── virtual_device.h
├── te/                  # Tensor Expression headers
├── tir/                 # TIR headers
│   ├── analysis.h
│   ├── builtin.h
│   ├── data_layout.h
│   ├── expr.h
│   ├── function.h
│   ├── op.h
│   ├── op_attr_types.h
│   ├── schedule.h
│   ├── stmt.h
│   └── transform.h
└── topi/                # TOPI headers
    ├── contrib/
    ├── cuda/
    ├── generic/
    ├── nn/
    ├── detail/
    ├── broadcast.h
    ├── elemwise.h
    ├── nn.h
    ├── reduction.h
    └── tags.h
```

### 35.1.4 Tests in tests/

The test directory is described in detail in the Testing and Benchmarking reference (Chapter 34). The main structure:

```
tests/
├── python/
│   ├── unittest/        # Core unit tests
│   ├── relay/           # Relay tests
│   ├── contrib/         # Integration tests
│   ├── driver/          # Driver tests
│   ├── frontend/        # Frontend tests
│   └── topi/            # TOPI tests
├── scripts/             # Test helper scripts
├── lint/                # Linting configuration
└── CI/                  # CI configuration
```

### 35.1.5 Documentation in docs/

Documentation is written in reStructuredText (RST) and built using Sphinx:

```
docs/
├── conf.py              # Sphinx configuration
├── Makefile             # Build automation
├── _static/             # Static assets (CSS, JS, images)
├── _templates/          # Custom Sphinx templates
├── README.md            # Documentation build instructions
├── index.rst            # Documentation root
├── install/             # Installation guides
│   ├── index.rst
│   └── from_source.rst
├── tutorial/            # Interactive tutorials
│   ├── introduction/
│   ├── cross_compilation/
│   └── autotvm/
├── how_to/              # How-to guides
│   ├── compile_models/
│   ├── deploy_models/
│   └── work_with_relay/
├── reference/           # API reference
│   ├── api/
│   ├── langref/
│   └── contrib/
└── archived/            # Archived documentation
```

---

## 35.2 Code Style

### 35.2.1 C++ Style Guide

TVM follows Google C++ Style Guide with several modifications:

**Naming Conventions:**

| Element | Convention | Example |
|---------|-----------|---------|
| Classes/Structs | PascalCase | `PrimFunc`, `IRModule`, `Buffer` |
| Functions/Methods | PascalCase | `GetBlock`, `Transform`, `VisitExpr` |
| Variables | snake_case | `loop_var`, `buffer_region` |
| Class members | snake_case with trailing `_` | `name_hint_`, `dtype_` |
| Constants | kCamelCase | `kMaxThreads`, `kDefaultTimeout` |
| Enum values | kCamelCase | `kFloat32`, `kInt32` |
| Macros | UPPER_SNAKE_CASE | `TVM_REGISTER_GLOBAL`, `TVM_DLL` |
| Namespaces | snake_case | `tvm::tir`, `tvm::relay` |
| Template params | CamelCase or short names | `typename T`, `typename ObjectType` |

**Include Order:**

```cpp
// 1. Corresponding header (if .cc file)
#include "tvm/tir/transform/storage_rewrite.h"

// 2. Blank line

// 3. C system headers
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// 4. Blank line

// 5. Other TVM headers (alphabetical)
#include "tvm/arith/analyzer.h"
#include "tvm/runtime/container.h"
#include "tvm/runtime/registry.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"
```

**Key Rules:**

```cpp
// Use `using` instead of `typedef`
using PrimExpr = Expr;  // Good
typedef Expr PrimExpr;   // Avoid

// Use smart pointers instead of raw pointers
std::shared_ptr<PassInfo> info;  // Good for shared ownership
std::unique_ptr<PassInfo> info;  // Good for unique ownership

// Use TVM's object system for reference-counted objects
class MyNode : public Object {
 public:
  String name;
  int value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "MyNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(MyNode, Object);
};

// Use TVM_DECLARE_BASE_OBJECT_INFO for base classes
// Use TVM_DECLARE_FINAL_OBJECT_INFO for final (non-derivable) classes

// Register functions and passes
TVM_REGISTER_GLOBAL("tir.transform.MyPass")
.set_body_typed(MyPassFunction);

TVM_REGISTER_PASS_CONFIG_OPTION("my_pass.enabled", Bool);
```

### 35.2.2 Python Style Guide

TVM follows PEP 8 with modifications:

**Naming Conventions:**

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | snake_case | `transform.py`, `analysis.py` |
| Classes | PascalCase | `IRModule`, `PrimFunc`, `BuildConfig` |
| Functions | snake_case | `build`, `lower`, `get_global_func` |
| Variables | snake_case | `opt_level`, `target_host` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_OPT_LEVEL`, `MAX_THREADS` |
| Private members | _leading_underscore | `_internal_state`, `_check_validity` |

**Key Rules:**

```python
# Use type hints for function signatures
from typing import Optional, List, Dict
from tvm import ir, tir

def my_transform(
    mod: ir.IRModule,
    opt_level: int = 3,
    targets: Optional[List[str]] = None,
) -> ir.IRModule:
    """Transform the IR module.

    Parameters
    ----------
    mod : IRModule
        The input IR module.
    opt_level : int
        Optimization level (0-3).
    targets : List[str], optional
        Target devices.

    Returns
    -------
    IRModule
        The transformed module.
    """
    ...

# Use TVM's FFI (Foreign Function Interface) for C++ bindings
import tvm._ffi

# Register a Python function to be callable from C++
@tvm.register_func("my.python.function")
def my_python_function(x):
    return x + 1

# Use tvm.ffi for C++ -> Python conversions
from tvm._ffi import register_object, funcify
```

**Docstring Convention (NumPy style):**

```python
def build(
    ir_mod: tvm.ir.IRModule,
    target: Optional[Union[str, Target]] = None,
    target_host: Optional[Union[str, Target]] = None,
    name: str = "main",
    params: Optional[Dict[str, tvm.runtime.NDArray]] = None,
    mod_name: Optional[str] = None,
) -> tvm.runtime.Module:
    """Build a function with a target, or multiple functions with
    multiple targets.

    Parameters
    ----------
    ir_mod : IRModule
        The IR module to build.
    target : Union[str, Target], optional
        The compilation target, e.g., "llvm", "cuda".
    target_host : Union[str, Target], optional
        Host compilation target. If None, derived from target.
    name : str
        The name of the main function.
    params : Dict[str, NDArray], optional
        Parameters to bind.
    mod_name : str, optional
        The module name.

    Returns
    -------
    runtime.Module
        The compiled runtime module.

    Examples
    --------
    >>> mod = tvm.ir.IRModule.from_expr(func)
    >>> rt_mod = tvm.build(mod, target="llvm")

    See Also
    --------
    tvm.relay.build : Build a Relay module.
    tvm.lower : Lower a schedule to TIR.
    """
    ...
```

---

## 35.3 Development Workflow

### 35.3.1 Fork and Clone

```bash
# 1. Fork the repository on GitHub (https://github.com/apache/tvm)

# 2. Clone your fork
git clone https://github.com/<your-username>/tvm.git
cd tvm

# 3. Add the upstream remote
git remote add upstream https://github.com/apache/tvm.git

# 4. Initialize submodules
git submodule update --init --recursive

# 5. Build TVM
mkdir build
cp cmake/config.cmake build/
cd build

# Edit config.cmake to enable features
# For example:
#   set(USE_CUDA ON)
#   set(USE_LLVM ON)
#   set(USE_GRAPH_EXECUTOR ON)
#   set(USE_PROFILER ON)

cmake ..
make -j$(nproc)

# 6. Set up Python environment
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Verify installation
python -c "import tvm; print(tvm.__version__)"
```

### 35.3.2 Create Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/my-new-feature

# Or for a bugfix
git checkout -b fix/issue-12345
```

### 35.3.3 Make Changes

Follow these principles when making changes:

1. **Keep changes focused**: Each PR should address one concern.
2. **Write tests first**: Consider test-driven development.
3. **Update documentation**: Add or update docs for new features.
4. **Follow code style**: Run linters before committing.

```bash
# After making changes, run relevant tests locally
pytest tests/python/unittest/test_my_feature.py -v

# Run linters
python tests/lint/git-black.sh
python tests/lint/cppcheck.sh
python tests/lint/pylint.sh
python tests/lint/check_file_type.py
```

### 35.3.4 Run Tests Locally

```bash
# Run specific test file
pytest tests/python/unittest/test_ir_module.py -v

# Run specific test function
pytest tests/python/unittest/test_ir_module.py::test_ir_module_get_attr -v

# Run all unit tests
pytest tests/python/unittest/ -v --timeout=300

# Run with specific markers
pytest tests/python/unittest/ -m "not slow" -v

# Run C++ tests
cd build
ctest --output-on-failure

# Run linting
bash tests/lint/task_lint.sh

# Run CI scripts locally (if available)
bash tests/scripts/task_python_unittest.sh
bash tests/scripts/task_python_relay.sh
```

### 35.3.5 Submit Pull Request

```bash
# Push your changes to your fork
git push origin feature/my-new-feature

# Then create a PR on GitHub from your fork to apache/tvm:main
```

PR title format: `[Area] Description`

Examples:
- `[Relay] Add new fusion rule for attention operators`
- `[TIR] Fix buffer access out of bounds in storage rewrite`
- `[Runtime] Add RPC support for Vulkan backend`
- `[Docs] Update installation guide for Ubuntu 22.04`
- `[CI] Pin pytest version for stability`

### 35.3.6 Code Review Process

1. **Automated checks**: CI must pass before review.
2. **Reviewer assignment**: At least one committer must approve.
3. **Review cycle**: Address comments, push fixes, request re-review.
4. **Merge**: A committer merges the PR after approval.

---

## 35.4 PR Requirements

### 35.4.1 Title Format

PR titles must follow the `[Area] Description` format. Common areas:

| Area | Description |
|------|-------------|
| `[Relay]` | Relay compiler changes |
| `[TIR]` | TIR changes |
| `[Runtime]` | Runtime system changes |
| `[TOPI]` | Tensor operator inventory changes |
| `[AutoTVM]` | AutoTVM tuning changes |
| `[MetaSchedule]` | MetaSchedule changes |
| `[TVMScript]` | TVMScript parser/printer changes |
| `[Target]` | Target description or codegen changes |
| `[Frontend]` | Model import changes |
| `[Docs]` | Documentation changes |
| `[CI]` | CI/CD changes |
| `[Build]` | Build system changes |
| `[MicroTVM]` | MicroTVM (embedded) changes |

### 35.4.2 Description Template

Every PR should include a description following this template:

```markdown
## Description
<!-- Brief description of the change -->

## Motivation
<!-- Why is this change needed? Link to relevant issues. -->

## Changes
<!-- List of specific changes -->
-

## Testing
<!-- How was this tested? -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] PR title follows `[Area] Description` format
- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] No breaking changes (or breaking changes documented)
```

### 35.4.3 Test Coverage

All PRs must include appropriate test coverage:

- **New features**: New test cases that verify the feature works correctly.
- **Bug fixes**: Test cases that reproduce the bug and verify the fix.
- **Refactoring**: Existing tests should continue to pass.
- **Performance improvements**: Benchmarks showing the improvement.

```python
# Example: Test for a new feature
import tvm
import numpy as np
import pytest

def test_my_new_pass():
    """Test that MyNewPass correctly transforms the IR."""
    # Create input IR
    x = relay.var("x", shape=(10,), dtype="float32")
    y = relay.nn.relu(x)
    mod = tvm.IRModule.from_expr(y)

    # Apply the pass
    result = MyNewPass()(mod)

    # Verify the transformation
    assert tvm.ir.structural_equal(result["main"], expected_func)

    # Verify numerical correctness
    # ... (compile and execute, compare with reference)

def test_my_new_pass_edge_case():
    """Test edge cases for MyNewPass."""
    # Empty module
    # Single-function module
    # Module with recursive calls
    # ...

# Parametrize for multiple targets
@tvm.testing.parametrize_targets
def test_my_new_pass_multi_target(target, dev):
    """Test MyNewPass on all available targets."""
    # ...
```

### 35.4.4 Documentation Updates

Documentation must be updated for:
- New public APIs.
- Changed API behavior.
- New features or passes.
- New command-line options or configuration parameters.

---

## 35.5 Code Review Guidelines

### 35.5.1 Review Criteria

Reviewers evaluate PRs based on:

1. **Correctness**: Does the code do what it claims?
2. **Performance**: Are there any performance regressions?
3. **Maintainability**: Is the code readable and well-documented?
4. **API design**: Is the API consistent with TVM conventions?
5. **Testing**: Are there sufficient tests?
6. **Documentation**: Is the documentation updated?

### 35.5.2 Reviewer Responsibilities

- Review code within a reasonable timeframe (typically 1-2 weeks).
- Provide constructive, specific feedback.
- Approve only when satisfied with all aspects.
- Request changes when issues are found.
- Consider the broader impact on the codebase.

### 35.5.3 Author Responsibilities

- Respond to review comments promptly.
- Make requested changes or provide rationale for alternatives.
- Keep the PR focused and reasonably sized.
- Rebase on the latest main when requested.
- Ensure CI passes before requesting re-review.

### 35.5.4 Resolving Comments

- Address every comment, even if just acknowledging.
- If you disagree, explain your reasoning clearly.
- Mark comments as resolved only after both parties agree.
- Use `nit:`, `suggestion:`, `question:`, `blocking:` prefixes for clarity.

---

## 35.6 CI System

### 35.6.1 GitHub Actions Workflows

TVM uses GitHub Actions for CI. The main workflows are:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main | Full CI pipeline |
| `lint.yml` | Push/PR to main | Code style checks |
| `docs.yml` | Push to main | Documentation build |

The CI pipeline runs these stages:

1. **Lint**: Check code style (black, cpplint, mypy, etc.)
2. **Build**: Compile TVM with various configurations
3. **Test**: Run the test suite
4. **Report**: Collect and report results

### 35.6.2 CI Checks

```bash
# Run CI checks locally before pushing

# Python lint (black formatter)
bash tests/lint/git-black.sh
black --check python/tvm/

# C++ lint (cpplint)
python tests/lint/cppcheck.sh

# Type checking (mypy)
mypy python/tvm/ --ignore-missing-imports

# File type checks
python tests/lint/check_file_type.py

# Full lint script
bash tests/scripts/task_lint.sh

# Run the full CI test suite
bash tests/scripts/task_python_unittest.sh
bash tests/scripts/task_python_relay.sh
bash tests/scripts/task_python_topi.sh
bash tests/scripts/task_python_frontend.sh
```

### 35.6.3 Handling CI Failures

When CI fails:

1. **Read the error log**: Click the failed check in GitHub to see the log.
2. **Reproduce locally**: Run the same test locally.
3. **Fix the issue**: Make the necessary changes.
4. **Push the fix**: Push to the same branch; CI will re-run automatically.

Common CI failure causes:
- **Test timeout**: Increase timeout or optimize the test.
- **Flaky test**: Add retries or fix the non-determinism.
- **Platform-specific failure**: Test on the specific platform (macOS, Windows).
- **Dependency issue**: Check if a dependency version changed.

```bash
# Debug CI failures locally

# Run the exact failing test
pytest tests/python/unittest/test_specific.py::test_failing -v -s

# Run with more debug output
export TVM_LOG_DEBUG=1
pytest tests/python/unittest/test_specific.py::test_failing -v -s

# Run on specific Python version
python3.9 -m pytest tests/python/unittest/test_specific.py -v
```

---

## 35.7 Documentation

### 35.7.1 Writing RST Docs

TVM documentation uses reStructuredText (RST) with Sphinx extensions.

```rst
.. _chapter-my-feature:

My Feature
==========

This section describes my feature and how to use it.

Basic Usage
-----------

.. code-block:: python

    import tvm
    from tvm import relay

    # Create a simple model
    x = relay.var("x", shape=(10,), dtype="float32")
    y = relay.nn.relu(x)

API Reference
-------------

.. py:function:: tvm.relay.nn.relu(data)

   Compute element-wise ReLU (Rectified Linear Unit).

   .. math::
      \text{ReLU}(x) = \max(0, x)

   :param data: Input tensor.
   :type data: tvm.relay.Expr
   :return: The ReLU of the input.
   :rtype: tvm.relay.Expr

   Example:

   .. code-block:: python

       x = relay.var("x", shape=(3,), dtype="float32")
       y = relay.nn.relu(x)
       # Equivalent to: relay.clip(x, a_min=0, a_max=None)

.. note::
   This is a note box.

.. warning::
   This is a warning box.

.. seealso::
   :py:func:`tvm.relay.nn.leaky_relu`
```

### 35.7.2 Writing Tutorials

Tutorials are interactive Jupyter notebooks converted to RST. They should include:
- Clear objectives.
- Step-by-step instructions with code.
- Explanations of each step.
- Visualizations where appropriate.
- Links to related documentation.

### 35.7.3 API Documentation

API documentation is generated from docstrings. Use the NumPy docstring convention:

```python
def my_function(x: tvm.tir.PrimExpr, axis: int = 0) -> tvm.tir.PrimExpr:
    """Sum elements along an axis.

    Parameters
    ----------
    x : PrimExpr
        The input expression.
    axis : int
        The axis along which to sum.

    Returns
    -------
    PrimExpr
        The sum expression.

    Examples
    --------
    .. code-block:: python

        x = tir.Var("x", "int32")
        result = my_function(x, axis=0)

    See Also
    --------
    tvm.tir.sum : TIR sum builtin
    """
    ...
```

### 35.7.4 Building Docs Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build the documentation
cd docs
make html

# Or with live reload (requires sphinx-autobuild)
sphinx-autobuild . _build/html

# View the built documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux

# Build a specific page only (faster iteration)
sphinx-build -b html . _build/html -D exclude_patterns="*" \
    -D include_patterns="tutorial/my_tutorial.rst"
```

---

## 35.8 Error Handling Guidelines

### 35.8.1 Using TVM Error Types

Choose the appropriate error type for each situation:

```cpp
// C++ error handling
#include <tvm/support/logging.h>

// Use LOG(FATAL) for unrecoverable errors
void process_buffer(const Buffer& buffer) {
    if (buffer->shape.size() == 0) {
        LOG(FATAL) << "ValueError: buffer must have non-zero rank, "
                   << "but got shape " << buffer->shape;
    }
}

// Use ICHECK for internal invariants
void transform_function(const PrimFunc& func) {
    ICHECK(func.defined()) << "InternalError: func must not be null";
    ICHECK(func->body.defined()) << "InternalError: func body must be defined";
}

// Provide helpful context in error messages
void check_shape_match(const Array<PrimExpr>& shape1,
                       const Array<PrimExpr>& shape2) {
    ICHECK_EQ(shape1.size(), shape2.size())
        << "ValueError: shape rank mismatch: "
        << shape1 << " vs " << shape2;
    for (size_t i = 0; i < shape1.size(); ++i) {
        ICHECK(is_equal(shape1[i], shape2[i]))
            << "ValueError: shape dimension mismatch at axis " << i
            << ": " << shape1[i] << " vs " << shape2[i];
    }
}
```

```python
# Python error handling
import tvm

def my_function(mod, target):
    if not isinstance(mod, tvm.ir.IRModule):
        raise ValueError(
            f"Expected IRModule, got {type(mod).__name__}"
        )

    if mod.functions is None or len(mod.functions) == 0:
        raise tvm.error.TVMError(
            "IRModule must contain at least one function"
        )

    # Provide actionable error messages
    try:
        compiled = tvm.relay.build(mod, target=target)
    except tvm.error.OpNotImplementedError as e:
        raise tvm.error.OpNotImplementedError(
            f"Operator not supported for target '{target}'. "
            f"Consider implementing a custom strategy or using a "
            f"different target. Original error: {e}"
        ) from e
```

### 35.8.2 Diagnostic Messages

Use TVM's diagnostic system for source-level error reporting:

```cpp
#include <tvm/ir/diagnostics.h>

void report_error(const PrimExpr& expr, const std::string& message) {
    auto diag_ctx = DiagnosticContext::Default();
    diag_ctx.Emit(Diagnostic::Error(expr->span)
        << message);
    diag_ctx.Throw();
}

void report_warning(const PrimExpr& expr, const std::string& message) {
    auto diag_ctx = DiagnosticContext::Default();
    diag_ctx.Emit(Diagnostic::Warning(expr->span)
        << message);
}
```

### 35.8.3 Error Propagation

When propagating errors across the FFI boundary:

```python
# Python -> C++ error propagation
# Errors raised in C++ are automatically converted to TVMError in Python

# C++ -> Python error propagation
# Use tvm.error types for consistency
@tvm.register_func("my.registered.function")
def my_registered_function(x):
    if x < 0:
        raise tvm.error.TVMError(
            f"Expected non-negative value, got {x}"
        )
    return x * 2
```

---

## 35.9 Release Process

### 35.9.1 Version Numbering

TVM follows semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes.
- **MINOR**: New features, backwards compatible.
- **PATCH**: Bug fixes, backwards compatible.

The version is defined in:
- `version.py` (Python package version).
- `CMakeLists.txt` (C++ library version).
- `include/tvm/runtime/c_runtime_api.h` (ABI version).

### 35.9.2 Release Candidates

Before each release:
1. A release branch is created (e.g., `v0.15.0`).
2. Release candidates are tagged (e.g., `v0.15.0.rc0`).
3. Community testing period (typically 1-2 weeks).
4. Bug fixes are cherry-picked to the release branch.

### 35.9.3 Release Notes

Release notes are generated from merged PRs and include:
- New features with PR links.
- Bug fixes with PR links.
- Breaking changes with migration instructions.
- Deprecation notices.
- Contributors list.

### 35.9.4 Binary Distribution

TVM provides pre-built binaries for:
- Python packages (PyPI: `apache-tvm`).
- Conda packages (`conda-forge`).
- Docker images (Docker Hub).

```bash
# Build a Python wheel locally
python setup.py bdist_wheel

# Upload to PyPI (for release managers)
twine upload dist/apache_tvm-*.whl
```

---

## 35.10 Community Resources

### 35.10.1 Mailing Lists

- **dev@tvm.apache.org**: Development discussions, RFCs, announcements.
  - Subscribe: dev-subscribe@tvm.apache.org
- **user@tvm.apache.org**: User questions, usage discussions.
  - Subscribe: user-subscribe@tvm.apache.org
- **commits@tvm.apache.org**: Automated commit notifications.
  - Subscribe: commits-subscribe@tvm.apache.org

### 35.10.2 Discord

The TVM community uses Discord for real-time communication:
- **#general**: General discussion.
- **#development**: Development questions.
- **#beginner**: Beginner-friendly help.
- **#auto-scheduler**: AutoTVM and MetaSchedule discussions.
- **#relay**: Relay compiler discussions.
- **#tir**: TIR and scheduling discussions.
- **#hardware**: Target-specific discussions.

### 35.10.3 Monthly Meetings

TVM holds regular community meetings:
- **Community Meeting**: Monthly, open to all. Discusses roadmap, features, and community topics.
- **Developer Sync**: Bi-weekly, for active contributors. Discusses technical details and PRs.

Meeting notes and recordings are posted on the TVM wiki: https://github.com/apache/tvm/wiki

### 35.10.4 RFC Process

For significant changes, use the Request for Comments (RFC) process:

1. **Write the RFC**: Use the RFC template in `rfcs/` directory.
2. **Submit as PR**: Create a PR with the `[RFC]` prefix.
3. **Community discussion**: Discuss on the PR and mailing list.
4. **Revision**: Address feedback and update the RFC.
5. **Approval**: After consensus, the RFC is approved.
6. **Implementation**: Implement the approved RFC.

RFC template structure:

```markdown
# RFC: [Feature Name]

## Summary
<!-- One-paragraph description -->

## Motivation
<!-- Why is this change needed? -->

## Design
### Overview
<!-- High-level design -->

### API Changes
<!-- New/modified APIs -->

### Implementation Plan
<!-- Step-by-step implementation plan -->

## Alternatives Considered
<!-- Other approaches that were considered -->

## Compatibility
<!-- Impact on existing code -->

## Testing
<!-- Testing strategy -->
```

### 35.10.5 Issue Reporting

When filing issues, use the issue template:

```markdown
## Environment
- TVM version (commit hash or tag):
- OS:
- Python version:
- CUDA version (if applicable):

## Problem Description
<!-- Clear description of the problem -->

## Steps to Reproduce
1. ...
2. ...

## Expected Behavior
<!-- What should happen -->

## Actual Behavior
<!-- What actually happens -->

## Triage
- [ ] Is this a regression?
- [ ] Does this affect multiple targets?
- [ ] Is there a workaround?

## Related Issues/PRs
-
```

### 35.10.6 Becoming a Committer

Apache TVM follows the Apache model for contributor progression:
1. **Contributor**: Anyone who submits patches.
2. **Committer**: Contributors who have shown sustained, high-quality contributions. Granted write access.
3. **PMC Member**: Committers who take on additional responsibilities (release management, community guidance).

The path to committer typically involves:
- Consistently submitting high-quality PRs.
- Participating in code reviews.
- Contributing to documentation and tests.
- Engaging in community discussions.
- Helping other users and contributors.

---

## 35.11 Summary

Contributing to Apache TVM involves understanding:
- The codebase organization across C++ (`src/`), Python (`python/tvm/`), headers (`include/tvm/`), tests (`tests/`), and docs (`docs/`).
- Code style guidelines for both C++ (Google style with modifications) and Python (PEP 8 with modifications).
- The development workflow: fork, branch, develop, test, submit PR, iterate on review.
- PR requirements: title format, description template, test coverage, documentation.
- CI system: GitHub Actions workflows, lint checks, test runs.
- Error handling guidelines using TVM's error types and diagnostic system.
- The RFC process for significant changes.
- Community resources: mailing lists, Discord, monthly meetings.

By following these guidelines, you can make effective contributions to the Apache TVM project and collaborate productively with the community.
