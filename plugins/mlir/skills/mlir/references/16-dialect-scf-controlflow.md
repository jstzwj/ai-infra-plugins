# MLIR SCF & ControlFlow Dialects

## SCF (Structured Control Flow) Dialect

### scf.for

```mlir
// Basic loop
scf.for %iv = %lb to %ub step %step {
  // body (no carried values)
}

// Loop with iter_args
%sum = scf.for %iv = %lb to %ub step %step
    iter_args(%acc = %init) -> (i32) {
  %next = arith.addi %acc, %iv : i32
  scf.yield %next : i32
}

// Multiple iter_args
%sum, %prod = scf.for %iv = %lb to %ub step %step
    iter_args(%s = %s0, %p = %p0) -> (i32, i32) {
  %new_s = arith.addi %s, %iv : i32
  %new_p = arith.muli %p, %iv : i32
  scf.yield %new_s, %new_p : i32, i32
}
```

### scf.forall

Parallel loop with potential reduction:

```mlir
// Basic forall
scf.forall (%i, %j) in (%m, %n) {
  // parallel body
}

// With shared outputs
%result = scf.forall (%i) in (%n) shared_outs(%o = %init) -> (tensor<100xf32>) {
  %slice = tensor.extract_slice %o[%i][10][1] : tensor<100xf32> to tensor<10xf32>
  %filled = some_op(%slice)
  tensor.parallel_insert_slice %filled into %o[%i][10][1]
    : tensor<10xf32> into tensor<100xf32>
}

// With mapping
scf.forall (%i, %j) in (%m, %n)
    processor_mapping = [#gpu.block<x>, #gpu.block<y>] {
  // body mapped to GPU blocks
}
```

### scf.while

General while loop:

```mlir
// While loop
%final = scf.while (%arg = %init) : (i32) -> (i32) {
  // Condition
  %cond = arith.cmpi "slt", %arg, %limit : i32
  scf.condition(%cond) %arg : i32
} do {
^bb0(%arg: i32):
  // Body
  %next = arith.addi %arg, %step : i32
  scf.yield %next : i32
}
```

### scf.if

```mlir
// Without results
scf.if %cond {
  // then region
} else {
  // else region
}

// With results
%result = scf.if %cond -> (i32) {
  scf.yield %true_val : i32
} else {
  scf.yield %false_val : i32
}

// Multiple results
%a, %b = scf.if %cond -> (i32, f32) {
  scf.yield %va, %vb : i32, f32
} else {
  scf.yield %vc, %vd : i32, f32
}
```

### scf.parallel

Parallel loop with reductions:

```mlir
// Parallel loop
%sum = scf.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1)
        step (%step0, %step1)
        init(%init_val)
        reduce((%acc, %val) {
          %new_acc = arith.addf %acc, %val : f32
          scf.reduce.return %new_acc : f32
        }) : f32 {
  %val = compute(%i, %j)
  scf.reduce %val : f32
}
```

### scf.execute_region

Execute a region as a single statement:

```mlir
%result = scf.execute_region -> (i32) {
  scf.yield %value : i32
}
```

### scf.yield / scf.condition

```mlir
// Yield values from scf.for, scf.if, scf.parallel body
scf.yield %value : i32
scf.yield %a, %b : i32, f32

// Condition for scf.while
scf.condition(%cond) %value : i32
```

## ControlFlow (CF) Dialect

Low-level explicit control flow operations.

### cf.br

Unconditional branch:

```mlir
cf.br ^bb1(%value : i32)
cf.br ^bb1
```

### cf.cond_br

Conditional branch:

```mlir
cf.cond_br %cond, ^bb1(%a : i32), ^bb2(%b, %c : i32, f32)
cf.cond_br %cond, ^bb1, ^bb2
```

### cf.switch

Multi-way branch:

```mlir
cf.switch %value : i32, [
  default: ^bb1,
  0: ^bb2(%a : i32),
  1, 2: ^bb3,
  3: ^bb4(%b : i32)
]
```

### cf.assert

```mlir
cf.assert %cond, "expected positive value"
```

## SCF vs CF Comparison

| Feature | SCF | CF |
|---------|-----|-----|
| Style | Structured | Goto-like |
| Regions | Implicit | Explicit blocks |
| Yield | Required | Not used |
| Analysis | Easier | Harder |
| Lowering | SCF -> CF | CF -> LLVM |
| Loops | First-class | Manual CFG |
| Composition | Natural | Complex |
| Readability | Higher | Lower |

## Lowering Patterns

### SCF to CF

```c++
// Lower scf.for to cf.br/cf.cond_br
pm.addPass(createSCFToControlFlowPass());
```

### SCF to OpenMP

```c++
// Lower scf.parallel to omp.parallel
pm.addPass(createSCFToOpenMPPass());
```

### SCF to GPU

```c++
// Lower scf.forall to gpu.launch
pm.addPass(createSCFToGPUPass());
```
