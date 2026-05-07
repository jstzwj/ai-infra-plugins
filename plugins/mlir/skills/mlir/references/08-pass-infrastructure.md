# MLIR Pass Infrastructure

## Overview

MLIR provides a comprehensive pass infrastructure for transforming IR. Passes are organized into a pass manager that handles execution, scheduling, instrumentation, and multithreading.

## Pass Base Classes

### OperationPass

```c++
// Pass that operates on a specific operation type
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // Transform func
  }

  // Optional: statistics
  Statistic opCount{this, "op-count", "Number of operations processed"};
};
```

### Generic OperationPass

```c++
// Pass that operates on any operation
struct MyGenericPass : public PassWrapper<MyGenericPass, OperationPass<>> {
  void runOnOperation() override {
    Operation *op = getOperation();
    // Transform op
  }
};
```

## Pass Registration

### Static Registration

```c++
// In a .cpp file
namespace mlir {
namespace my_namespace {
#define GEN_PASS_REGISTRATION
#include "MyPasses.h.inc"
} // namespace my_namespace
} // namespace mlir

// Register all passes
void registerMyPasses() {
  mlir::my_namespace::registerMyPass();
}
```

### Command-Line Registration

```c++
// For mlir-opt tool integration
void registerMyPasses() {
  PassRegistration<MyPass>();
}
```

## Pass Manager

### Basic Usage

```c++
MLIRContext ctx;
PassManager pm(&ctx);

// Add passes
pm.addPass(std::make_unique<MyPass>());
pm.addPass(createCanonicalizerPass());
pm.addPass(createCSEPass());

// Run
if (failed(pm.run(module)))
  llvm::errs() << "Pass pipeline failed\n";
```

### Nested Pass Managers

```c++
PassManager pm(&ctx);

// Module-level pass
pm.addPass(std::make_unique<MyModulePass>());

// Function-level pass (nested)
pm.nest<func::FuncOp>().addPass(std::make_unique<MyFuncPass>());

// Or nest under any op
pm.nestAny().addPass(std::make_unique<MyGenericPass>());
```

### Pass Pipelines

```c++
// Define a pipeline
void buildMyPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.nest<func::FuncOp>().addPass(std::make_unique<MyFuncPass>());
}

// Register pipeline
PassPipelineRegistration<>("my-pipeline", "My pipeline",
    [](OpPassManager &pm) { buildMyPipeline(pm); });
```

### Pass Options

```c++
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  // String option
  Option<std::string> outputFilename{this, "output",
    llvm::cl::desc("Output filename"),
    llvm::cl::init("-")};

  // Integer option
  Option<int> optimizationLevel{this, "opt-level",
    llvm::cl::desc("Optimization level"),
    llvm::cl::init(0)};

  // Boolean option
  Option<bool> verbose{this, "verbose",
    llvm::cl::desc("Enable verbose output"),
    llvm::cl::init(false)};

  // List option
  ListOption<std::string> passList{this, "passes",
    llvm::cl::desc("Passes to run")};

  void runOnOperation() override {
    StringRef filename = outputFilename;
    int level = optimizationLevel;
    bool isVerbose = verbose;
  }
};
```

## Pass Instrumentation

```c++
struct MyInstrumentation : public PassInstrumentation {
  void runBeforePass(Pass *pass, Operation *op) override {
    llvm::errs() << "Before: " << pass->getName() << "\n";
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    llvm::errs() << "After: " << pass->getName() << "\n";
  }

  void runAfterPassFailed(Pass *pass, Operation *op) override {
    llvm::errs() << "Failed: " << pass->getName() << "\n";
  }

  void runBeforeAnalysis(StringRef name, TypeID id, Operation *op) override {}
  void runAfterAnalysis(StringRef name, TypeID id, Operation *op) override {}
};

// Register instrumentation
pm.addInstrumentation(std::make_unique<MyInstrumentation>());
```

## Pass Statistics

```c++
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  Statistic opCount{this, "op-count", "Operations processed"};
  Statistic funcCount{this, "func-count", "Functions processed"};
  Statistic erasedCount{this, "erased", "Operations erased"};

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      ++opCount;
      if (isa<arith::AddIOp>(op)) {
        op->erase();
        ++erasedCount;
      }
    });
  }
};
```

## Pass Crashing Reproducer

```c++
// Enable reproducer on crash
pm.enableCrashReproducerGeneration("reproducer.mlir");

// Or with local reproducer
pm.enableCrashReproducerGeneration("reproducer.mlir",
                                    /*localReproducer=*/true);
```

## Multithreaded Pass Execution

```c++
MLIRContext ctx;
ctx.enableMultithreading();

PassManager pm(&ctx);
// Add passes...
pm.run(module);
```

Thread safety requirements:
- Passes must not share mutable state
- Each thread gets its own copy of the pass
- Use `MLIRContext::isMultithreadingEnabled()` to check

## Built-in Passes

### Canonicalizer

```c++
pm.addPass(createCanonicalizerPass());

// With specific patterns
RewritePatternSet patterns(ctx);
patterns.add<MyPattern1, MyPattern2>(ctx);
pm.addPass(createCanonicalizerPass(std::move(patterns)));
```

### Common Subexpression Elimination (CSE)

```c++
pm.addPass(createCSEPass());
```

### Inliner

```c++
pm.addPass(createInlinerPass());

// With custom pipeline
pm.addPass(createInlinerPass(
  [](OpPassManager &pm) { pm.addPass(createCanonicalizerPass()); }
));
```

### Symbol DCE

```c++
pm.addPass(createSymbolDCEPass());
```

### Strip Debug Info

```c++
pm.addPass(createStripDebugInfoPass());
```

### SROA (Scalar Replacement of Aggregates)

```c++
pm.addPass(createSROAStatisticsPass());
```

### Print IR Pass

```c++
pm.addPass(createPrintIRPass());
```

### SCCP (Sparse Conditional Constant Propagation)

```c++
pm.addPass(createSCCPPass());
```

### Loop Invariant Code Motion

```c++
pm.addPass(createLoopInvariantCodeMotionPass());
```

### Control Flow Sink

```c++
pm.addPass(createControlFlowSinkPass());
```

### Conversion Passes

```c++
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/FuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"

// Lower SCF to CF
pm.addPass(createSCFToControlFlowPass());

// Lower Arith to LLVM
pm.nest<func::FuncOp>().addPass(createArithToLLVMConversionPass());

// Lower MemRef to LLVM
pm.nest<func::FuncOp>().addPass(createMemRefToLLVMPass());

// Lower Func to LLVM
pm.addPass(createFuncToLLVMPass());

// Lower to LLVM dialect
pm.addPass(createConvertToLLVMPass());
```

## Complete Pass Example

```c++
struct SimplifyAddZero
    : public PassWrapper<SimplifyAddZero, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyAddZeroPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const override { return "simplify-add-zero"; }
  StringRef getDescription() const override {
    return "Simplify arith.addi with zero";
  }
};

struct SimplifyAddZeroPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                 PatternRewriter &rewriter) const override {
    if (auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>()) {
      if (rhs.getValue().cast<IntegerAttr>().getInt() == 0) {
        rewriter.replaceOp(op, op.getLhs());
        return success();
      }
    }
    return failure();
  }
};
```
