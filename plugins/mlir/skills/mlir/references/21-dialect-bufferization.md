# MLIR Bufferization

## Overview

Bufferization is the process of converting tensor-based operations to memref-based operations. MLIR provides a one-shot bufferization pass that performs this conversion efficiently.

## One-Shot Bufferization

### Usage

```c++
// Enable one-shot bufferization
pm.addPass(bufferization::createOneShotBufferizePass());

// With options
bufferization::OneShotBufferizationOptions options;
options.bufferizeFunctionBoundaries = true;
options.allowUnknownOps = true;
pm.addPass(bufferization::createOneShotBufferizePass(options));
```

### Bufferization Options

| Option | Default | Description |
|--------|---------|-------------|
| `bufferizeFunctionBoundaries` | false | Bufferize function boundaries |
| `allowUnknownOps` | false | Allow ops without BufferizableOpInterface |
| `allowReturnAllocs` | false | Allow returning allocations |
| `createDeallocs` | true | Create deallocation ops |
| `copyBeforeWrite` | true | Insert copies before writes |
| `unknownTypeConverter` | layout | How to convert unknown types |

### mlir-opt Usage

```bash
mlir-opt --one-shot-bufferize input.mlir
mlir-opt --one-shot-bufferize="bufferize-function-boundaries" input.mlir
```

## Bufferization Dialect Operations

### bufferization.to_memref

Convert tensor to memref (tensor → memref):

```mlir
%memref = bufferization.to_memref %tensor : memref<10xf32>
%memref = bufferization.to_memref %tensor : memref<?xf32, strided<[1]>>
```

### bufferization.to_tensor

Convert memref to tensor (memref → tensor):

```mlir
%tensor = bufferization.to_tensor %memref : tensor<10xf32>
%tensor = bufferization.to_tensor %memref restrict : tensor<?xf32>
```

### bufferization.materialize_in_destination

Materialize a tensor value into a destination buffer:

```mlir
bufferization.materialize_in_destination %source in writable %dest
    : (tensor<10xf32>, memref<10xf32>) -> ()
```

### bufferization.alloc_tensor

Allocate a tensor buffer:

```mlir
%t = bufferization.alloc_tensor() : tensor<10xf32>
%t = bufferization.alloc_tensor(%n) : tensor<?xf32>
```

### bufferization.clone

Clone a buffer:

```mlir
%clone = bufferization.clone %src : memref<10xf32>
```

### bufferization.dealloc

Deallocate a buffer:

```mlir
bufferization.dealloc %memref : memref<10xf32>
```

## Buffer Deallocation

### Ownership-Based Buffer Deallocation

MLIR provides automatic buffer deallocation:

```c++
// Add deallocation pass
pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
```

### Deallocation Process

1. Analyze buffer lifetimes
2. Insert bufferization.dealloc operations
3. Ensure no memory leaks

```mlir
// Before deallocation
%buf = memref.alloc() : memref<10xf32>
%result = some_op(%buf)
// missing dealloc!

// After deallocation pass
%buf = memref.alloc() : memref<10xf32>
%result = some_op(%buf)
bufferization.dealloc %buf : memref<10xf32>
```

## Bufferization Interfaces

### BufferizableOpInterface

Operations must implement this interface to participate in bufferization:

```c++
struct BufferizableOpInterface {
  // Check if operation buffers in-place
  bool bufferizesToMemoryRead(Operation *op, OpOperand &operand);
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &operand);
  bool bufferizesToElementwiseAccess(Operation *op);

  // Get buffer for operand
  FailureOr<Value> getBuffer(RewriterBase &rewriter, Value value);

  // Bufferize the operation
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter);
};
```

### Analysis

The bufferization analysis determines:
- Which tensors can be bufferized in-place
- Where copies are needed
- Buffer aliases and conflicts

## Bufferization Pipeline

```c++
void buildBufferizationPipeline(OpPassManager &pm) {
  // Step 1: One-shot bufferization
  bufferization::OneShotBufferizationOptions opts;
  opts.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(opts));

  // Step 2: Buffer deallocation
  pm.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());

  // Step 3: Buffer optimization (optional)
  pm.addPass(bufferization::createBufferOptimizationPass());

  // Step 4: Drop equivalent bufferization.to_tensor/to_memref pairs
  pm.addPass(bufferization::createDropEquivalentBufferResultsPass());

  // Step 5: Finalize
  pm.addPass(bufferization::createFinalizingBufferizePass());
}
```

## Common Patterns

### Tensor to MemRef Conversion

```mlir
// Before bufferization
func.func @add(%a: tensor<10xf32>, %b: tensor<10xf32>) -> tensor<10xf32> {
  %result = arith.addf %a, %b : tensor<10xf32>
  return %result : tensor<10xf32>
}

// After bufferization
func.func @add(%a: memref<10xf32>, %b: memref<10xf32>) -> memref<10xf32> {
  %result = memref.alloc() : memref<10xf32>
  linalg.generic {indexing_maps = [...], iterator_types = ["parallel"]}
    ins(%a, %b : memref<10xf32>, memref<10xf32>)
    outs(%result : memref<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %sum = arith.addf %arg0, %arg1 : f32
      linalg.yield %sum : f32
  }
  return %result : memref<10xf32>
}
```

### In-Place Bufferization

When a tensor operation writes to a buffer that can be reused:

```mlir
// Before: %output is a new tensor
%result = linalg.matmul ins(%A, %B) outs(%output)

// After: %output_buf is reused
%output_buf = bufferization.to_memref %output : memref<10x10xf32>
linalg.matmul ins(%A_buf, %B_buf) outs(%output_buf)
```
