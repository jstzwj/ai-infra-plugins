# MLIR Pattern Rewriting

## Overview

MLIR's pattern rewriting framework provides a declarative and imperative approach to transforming operations.

## RewritePattern

### OpRewritePattern

```c++
struct MyPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                 PatternRewriter &rewriter) const override {
    // Match condition
    if (!isa<arith::ConstantOp>(op.getRhs().getDefiningOp()))
      return failure();

    // Rewrite
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, op.getLhs(), op.getRhs());
    return success();
  }
};
```

### Generic RewritePattern

```c++
struct MyGenericPattern : public RewritePattern {
  MyGenericPattern(MLIRContext *ctx)
      : RewritePattern("my_dialect.op", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                 PatternRewriter &rewriter) const override {
    // Match and rewrite any operation
    return success();
  }
};
```

### Pattern Benefit

Higher benefit patterns are applied first:

```c++
// Explicit benefit
MyPattern(MLIRContext *ctx)
    : OpRewritePattern<arith::AddIOp>(ctx, /*benefit=*/10) {}
```

## PatternRewriter

The `PatternRewriter` provides IR modification methods:

### Operation Creation

```c++
// Create new operation
rewriter.create<arith::AddIOp>(loc, lhs, rhs);

// Create at specific location
OpBuilder::InsertionGuard guard(rewriter);
rewriter.setInsertionPointAfter(someOp);
rewriter.create<arith::ConstantOp>(loc, value);

// Create block arguments
rewriter.createBlock(region);
rewriter.createBlock(region, region->end(), argTypes, argLocs);
```

### Operation Replacement

```c++
// Replace with new values
rewriter.replaceOp(op, newValues);

// Replace with new operation
rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);

// Replace value
rewriter.replaceAllUsesWith(oldValue, newValue);
rewriter.replaceOpWithIf(op, newValues, [](OpOperand &operand) {
  return operand.getOwner()->getBlock() != excludedBlock;
});
```

### Operation Erasure

```c++
// Erase operation (must have no uses)
rewriter.eraseOp(op);

// Erase block
rewriter.eraseBlock(block);

// Inline block contents
rewriter.inlineBlockBefore(block, op);
```

### Block Operations

```c++
// Split block
Block *newBlock = rewriter.splitBlock(block, iterator);

// Merge blocks
rewriter.mergeBlocks(source, dest, argValues);

// Move operations
rewriter.moveOpBefore(op, beforeOp);
rewriter.moveOpAfter(op, afterOp);
```

### Region Operations

```c++
// Clone operation
Operation *clone = rewriter.clone(op);
Operation *clone = rewriter.cloneWithoutRegions(op);
```

## RewritePatternSet

```c++
RewritePatternSet patterns(ctx);

// Add individual patterns
patterns.add<Pattern1, Pattern2, Pattern3>(ctx);

// Add patterns from dialect
arith::ArithDialect::getCanonicalizationPatterns(patterns, ctx);

// Add patterns with functors
patterns.add([](MLIRContext *ctx) {
  return std::make_unique<MyPattern>(ctx);
});
```

## Pattern Application

### GreedyPatternRewriteDriver

```c++
// Apply patterns greedily until fixed point
if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
  return failure();

// With configuration
GreedyRewriteConfig config;
config.maxIterations = 10;
config.useTopDownTraversal = true;
config.regionScope = GreedyRewriteConfig::RegionScopeKind:: enclosing;
if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
  return failure();
```

### GreedyRewriteConfig

| Option | Default | Description |
|--------|---------|-------------|
| `maxIterations` | 10 | Max rewrite iterations |
| `useTopDownTraversal` | false | Top-down vs bottom-up |
| `regionScope` | enclosing | Region scope for rewriting |
| `fold` | true | Enable folding |
| `cse` | false | Enable common subexpression elimination |
| `strictMode` | true | Strict pattern application |

## Canonicalization

### Canonicalization Patterns

```c++
// Register canonicalization patterns for an operation
void MyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<MyCanonicalPattern1, MyCanonicalPattern2>(context);
}
```

### Folding

```c++
// Implement fold method
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  // Fold to constant
  if (auto input = dyn_cast_or_null<IntegerAttr>(adaptor.getInput()))
    return IntegerAttr::get(getType(), input.getInt() * 2);

  // Fold to existing value
  if (someCondition())
    return getInput();

  // Cannot fold
  return {};
}

// For operations with multiple results
LogicalResult MyMultiResultOp::fold(FoldAdaptor adaptor,
                                     SmallVectorImpl<OpFoldResult> &results) {
  results.resize(getNumResults());
  results[0] = IntegerAttr::get(getResultTypes()[0], 42);
  return success();
}
```

### Foldable Types

An `OpFoldResult` can be:
- `Attribute` - represents a compile-time constant
- `Value` - represents an existing SSA value

## Declarative Rewrite Rules (DRR)

DRR allows defining patterns in TableGen:

```tablegen
def : Pat<(MyDialect.AddOp $lhs, $rhs),
          (MyDialect.MulOp $lhs, $rhs)>;
```

### Pattern with Constraints

```tablegen
def : Pat<(arith.AddIOp $lhs, (arith.ConstantOp $val)),
          (replaceWithValue $lhs)>,
      [(Constraint<CPred<"isZero($0)">, "zero"> $val)]>;
```

### Pattern with Native Code Call

```tablegen
def : Pat<(MyDialect.ComplexOp $a, $b),
          (MyDialect.SimplifiedOp (NativeCodeCall<"transform($0, $1)"> $a, $b))>;
```

### Pattern with Multiple Results

```tablegen
def : Pat<(MyDialect.UnpackOp $input),
          (MyDialect.ExtractFirst $input),
          [(MyDialect.ExtractSecond $input)]>;
```

### Supplementary Parameters

```tablegen
def : Pat<(MyDialect.OpWithAttr $input, $attr:$value),
          (MyDialect.SimplifiedOp $input),
          [/* constraints */]>;
```

### DRR Syntax Reference

```
pattern ::= def `:` `Pat` `<` source-pattern `,` result-pattern `,` constraints? `>`
source-pattern ::= op-name `(` operands `)`
result-pattern ::= op-name `(` operands `)` | `replaceWithValue` `$var`
constraints ::= `[` constraint (`,` constraint)* `]`
constraint ::= `(` Constraint<...> `$var` `)`
```

## Common Rewriting Patterns

### Erasing Operations

```c++
struct ErasePattern : public OpRewritePattern<MyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MyOp op,
                                 PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
```

### Replacing with Different Op

```c++
struct LowerPattern : public OpRewritePattern<MyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MyOp op,
                                 PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = rewriter.create<arith::AddIOp>(loc, op.getA(), op.getB());
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### Creating New Blocks

```c++
struct SplitBlockPattern : public OpRewritePattern<MyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MyOp op,
                                 PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    auto nextOp = ++Block::iterator(op);
    Block *newBlock = rewriter.splitBlock(block, nextOp);

    // Add branch between blocks
    rewriter.setInsertionPointToEnd(block);
    rewriter.create<cf::BranchOp>(op.getLoc(), newBlock);
    return success();
  }
};
```

### Traversing and Rewriting

```c++
struct RewriteNested : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp func,
                                 PatternRewriter &rewriter) const override {
    bool changed = false;
    func.walk([&](MyOp op) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, op.getA(), op.getB());
      changed = true;
    });
    return success(changed);
  }
};
```
