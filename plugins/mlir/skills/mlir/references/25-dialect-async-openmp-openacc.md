# MLIR Async, OpenMP & OpenACC Dialects

## Async Dialect

Asynchronous execution operations:

```mlir
// Execute asynchronously
%token, %result = async.execute [%dep_token] (%arg as %captured: !async.value<f32>) -> !async.value<f32> {
  %computed = compute(%captured) : f32
  async.yield %computed : f32
}

// Await async value
%val = async.await %result : !async.value<f32>

// Await token
async.await %token : !async.token

// Create group
%group = async.create_group %size : !async.group

// Add to group
%new_group = async.add_to_group %token, %group : !async.token, !async.group

// Await all in group
async.await_all %group : !async.group

// Return from async.func
async.return %val : f32
```

### Async Types

| Type | Description |
|------|-------------|
| `!async.token` | Completion token |
| `!async.value<T>` | Async value of type T |
| `!async.group` | Group of tokens |

## OpenMP Dialect

### Parallel Region

```mlir
omp.parallel {
  // parallel body
  omp.terminator
}

// With optional clauses
omp.parallel num_threads(%n : i32) proc_bind(close) {
  omp.terminator
}
```

### Loop Constructs

```mlir
// Worksharing loop
omp.wsloop for %i = %lb to %ub step %step {
  // loop body
  omp.yield
}

// Simd loop
omp.simdloop for %i = %lb to %ub step %step {
  // simd body
  omp.yield
}

// Combined parallel worksharing
omp.parallel {
  omp.wsloop for %i = %lb to %ub step %step {
    // body
    omp.yield
  }
  omp.terminator
}
```

### Synchronization

```mlir
// Barrier
omp.barrier

// Taskwait
omp.taskwait

// Flush
omp.flush
```

### Task Constructs

```mlir
// Task
omp.task {
  // task body
  omp.terminator
}

// Task with dependencies
omp.task depend(depout: %a, %b) {
  // body
  omp.terminator
}
```

### Critical Section

```mlir
omp.critical {
  // critical section body
  omp.terminator
}
```

### Master

```mlir
omp.master {
  // executed by master thread only
  omp.terminator
}
```

### Reduction

```mlir
%sum = omp.wsloop for %i = %lb to %ub step %step
    reduction(@add_f32 -> %sum : f32) -> f32 {
  %val = load(%i) : f32
  omp.yield(%val : f32)
}

omp.reduction.declare @add_f32 : f32
init(%init: f32) {
  %zero = arith.constant 0.0 : f32
  omp.yield(%zero : f32)
}
combiner(%x: f32, %y: f32) {
  %sum = arith.addf %x, %y : f32
  omp.yield(%sum : f32)
}
```

### Atomic Operations

```mlir
// Atomic read
%val = omp.atomic.read %ptr : memref<i32>, i32

// Atomic write
omp.atomic.write %ptr, %val : memref<i32>, i32

// Atomic update
omp.atomic.update %ptr : memref<i32> {
  ^bb0(%old: i32):
    %new = arith.addi %old, %c1 : i32
    omp.yield(%new : i32)
}

// Atomic capture
%old = omp.atomic.capture %ptr : memref<i32> {
  omp.atomic.update %ptr : memref<i32> {
    ^bb0(%old: i32):
      %new = arith.addi %old, %c1 : i32
      omp.yield(%new : i32)
  }
  omp.atomic.read %ptr -> i32
}
```

### Sections

```mlir
omp.sections {
  omp.section {
    // section 1
    omp.terminator
  }
  omp.section {
    // section 2
    omp.terminator
  }
  omp.terminator
}
```

## OpenACC Dialect

### Parallel Region

```mlir
acc.parallel {
  // parallel body
  acc.yield
}

// With clauses
acc.parallel num_workers(%n : i32) vector_length(%v : i32) {
  acc.yield
}
```

### Loop

```mlir
acc.loop {
  scf.for %i = %lb to %ub step %step {
    acc.yield
  }
  acc.yield
}

// Gang, worker, vector
acc.loop gang num_workers(%n : i32) vector(%v : i32) {
  // loop
  acc.yield
}
```

### Data Directives

```mlir
// Data region
acc.data copyin(%a : memref<10xf32>) copyout(%b : memref<10xf32>) {
  // data region body
  acc.yield
}

// Enter/exit data
acc.enter_data copyin(%a : memref<10xf32>)
acc.exit_data copyout(%b : memref<10xf32>)

// Host data
acc.host_data use_device(%a : memref<10xf32>) {
  // use device pointer
  acc.yield
}
```

### Kernels

```mlir
acc.kernels {
  // kernel region
  acc.yield
}

// With num_gangs
acc.kernels num_gangs(%n : i32) {
  acc.yield
}
```

### Routine

```mlir
acc.routine @func gang worker vector seq
```

### Init/Shutdown

```mlir
acc.init
acc.shutdown
```

### Wait

```mlir
acc.wait %async_token : !acc.async.token
acc.wait_all
```

### Cache

```mlir
acc.cache %varptr[%lb:%ub] : memref<10xf32>
```
