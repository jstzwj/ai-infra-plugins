# MLIR Operations, Blocks, and Regions

## Operation

### Overview

`Operation` is the central class in MLIR. It represents a single instruction/statement in the IR.

### C++ API

```c++
class Operation {
  // === Identity ===
  StringAttr getName();
  Dialect *getDialect();
  Location getLoc();

  // === Operands ===
  Value getOperand(unsigned idx);
  void setOperand(unsigned idx, Value value);
  OpOperand &getOpOperand(unsigned idx);
  unsigned getNumOperands();
  operand_range getOperands();
  MutableArrayRef<OpOperand> getOpOperands();

  // === Results ===
  OpResult getResult(unsigned idx);
  unsigned getNumResults();
  result_range getResults();
  Type getResultType(unsigned idx);
  result_type_range getResultTypes();

  // === Attributes ===
  DictionaryAttr getAttrDictionary();
  Attribute getAttr(StringAttr name);
  Attribute getAttr(StringRef name);
  void setAttr(StringAttr name, Attribute value);
  void setAttr(StringRef name, Attribute value);
  void removeAttr(StringAttr name);
  bool hasAttr(StringAttr name);
  bool hasAttr(StringRef name);

  // === Properties ===
  Properties properties;
  template <typename T> T getProperties();

  // === Regions ===
  Region &getRegion(unsigned idx);
  unsigned getNumRegions();
  mutable_region_range getRegions();
  bool hasSingleBlock();
  Region &getRegionIterator(unsigned idx);

  // === Successors ===
  Block *getSuccessor(unsigned idx);
  unsigned getNumSuccessors();
  SuccessorRange getSuccessors();

  // === Hierarchy ===
  Block *getBlock();
  Region *getParentRegion();
  Operation *getParentOp();
  ModuleOp getParentOfType<ModuleOp>();

  // === Traversal ===
  template <typename Fn> void walk(Fn &&callback);
  template <typename Fn> WalkResult walk(Fn &&callback);

  // === Mutation ===
  void erase();
  void moveBefore(Operation *otherOp);
  void moveAfter(Operation *otherOp);
  void remove();
  void replaceWith(Operation *newOp);

  // === Diagnostics ===
  InFlightDiagnostic emitError();
  InFlightDiagnostic emitWarning();
  InFlightDiagnostic emitRemark();
  InFlightDiagnostic emitOpError();

  // === Utilities ===
  Operation *clone(IRMapping &mapper);
  Operation *cloneWithoutRegions(IRMapping &mapper);
  bool isProperAncestor(Operation *other);
  bool isAncestor(Operation *other);
  static bool classof(Operation *op);
};
```

### Op<T> Template

Generated C++ class for specific operations:

```c++
// For a dialect op "arith.addi":
class AddIOp : public Op<AddIOp> {
  // Generated accessors
  Value getLhs();
  Value getRhs();
  Value getResult();
  // ... generated builders, verifiers, etc.
};
```

### Operation Creation

```c++
// Using OpBuilder
OpBuilder builder(context);
auto loc = builder.getUnknownLoc();

// Create with generic API
OperationState state(loc, "arith.addi");
state.addOperands({lhs, rhs});
state.addTypes({builder.getI32Type()});
Operation *op = builder.create(state);

// Create using specific Op class
auto addOp = builder.create<arith::AddIOp>(loc, lhs, rhs);
```

### Operation Walking

```c++
// Walk all operations
op->walk([](Operation *op) {
  llvm::errs() << op->getName() << "\n";
});

// Walk specific op types
op->walk([](arith::AddIOp addOp) {
  // Process each addi operation
});

// Walk with early termination
op->walk<WalkOrder::PreOrder>([](Operation *op) -> WalkResult {
  if (auto addOp = dyn_cast<arith::AddIOp>(op))
    return WalkResult::interrupt();
  return WalkResult::advance();
});
```

## Block

### Overview

A `Block` represents a sequence of operations. In SSACFG regions, blocks are basic blocks with arguments.

### C++ API

```c++
class Block {
  // === Operations ===
  OpListType &getOperations();
  iterator begin();
  iterator end();
  iterator_range<iterator> getOps();
  template <typename T> iterator_range<typename OpIterator<T>::type> getOps();
  Operation &front();
  Operation &back();
  bool empty();
  void push_back(Operation *op);
  void push_front(Operation *op);
  Operation *remove(Operation *op);
  iterator insert(iterator insertPt, Operation *op);
  void erase();

  // === Arguments ===
  BlockArgument addArgument(Type type, Location loc);
  BlockArgument insertArgument(iterator iter, Type type, Location loc);
  void eraseArgument(unsigned index);
  BlockArgument getArgument(unsigned index);
  unsigned getNumArguments();
  args_range getArguments();
  iterator_range<args_iterator> args();

  // === Hierarchy ===
  Region *getParent();
  Operation *getParentOp();
  ModuleOp getParentOfType<ModuleOp>();

  // === Predecessors/Successors ===
  pred_iterator pred_begin();
  pred_iterator pred_end();
  iterator_range<pred_iterator> getPredecessors();
  bool hasNoPredecessors();
  bool hasNPredecessors(unsigned N);
  bool hasNPredecessorsOrMore(unsigned N);
  unsigned getNumPredecessors();
  SuccessorRange getSuccessors();

  // === Utilities ===
  bool isEntryBlock();
  Block *splitBlock(iterator splitBefore);
  void print(raw_ostream &os);
  void dump();
};
```

### Block Arguments

Block arguments replace PHI nodes in traditional SSA:

```mlir
^bb0(%arg0: i32, %arg1: f32):
  // %arg0 and %arg1 are block arguments
```

```c++
// Access block arguments
for (BlockArgument arg : block.getArguments()) {
  Type type = arg.getType();
  unsigned argNumber = arg.getArgNumber();
  Block *owner = arg.getOwner();
  Region *region = arg.getParentRegion();
}
```

### Block Splitting

```c++
// Split block at iterator position
Block *newBlock = block.splitBlock(block.begin()->getNextNode());
```

## Region

### Overview

A `Region` is an ordered list of blocks contained within an operation.

### C++ API

```c++
class Region {
  // === Blocks ===
  BlockListType &getBlocks();
  iterator begin();
  iterator end();
  Block &front();
  Block &back();
  bool empty();
  void push_back(Block *block);
  void push_front(Block *block);
  Block *remove(Block *block);
  iterator insert(iterator insertPt, Block *block);

  // === Entry Block ===
  Block &emplaceBlock();
  Block *empty() ? nullptr : &front();

  // === Hierarchy ===
  Operation *getParentOp();
  Region *getParentRegion();
  Block *getParentBlock();

  // === Traversal ===
  template <typename Fn> void walk(Fn &&callback);
  template <typename Fn> WalkResult walk(Fn &&callback);

  // === Queries ===
  bool isProperAncestor(Region *other);
  bool isAncestor(Region *other);
  unsigned getNumArguments();
  BlockArgument getArgument(unsigned index);
  RegionArgumentRange getArguments();
  unsigned getNumBlocks();

  // === Dominance ===
  DominanceInfo &getDominanceInfo();
  bool isDominates(Block *a, Block *b);
  bool properlyDominates(Operation *a, Operation *b);
  bool properlyDominates(Value a, Operation *b);

  // === Mutation ===
  Region *clone();
  void dropAllReferences();
  void clear();
};
```

### Region Kinds

```c++
enum class RegionKind {
  SSACFG,    // Sequential execution, blocks form CFG
  Graph      // No control flow, single block
};
```

Determined by `RegionKindInterface` on the parent operation.

### Value Visibility in Regions

```
┌─ Operation (parent) ──────────────────────┐
│  Values defined here visible in children  │
│  ┌─ Region 1 ───────────────────────────┐ │
│  │  ┌─ Block ─────────────────────────┐ │ │
│  │  │  Values visible within block    │ │ │
│  │  │  and children                   │ │ │
│  │  │  ┌─ Operation ────────────────┐ │ │ │
│  │  │  │  ┌─ Region 2 ──────────┐  │ │ │ │
│  │  │  │  │  Can see all above  │  │ │ │ │
│  │  │  │  └─────────────────────┘  │ │ │ │
│  │  │  └───────────────────────────┘ │ │ │
│  │  └────────────────────────────────┘ │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

## IR Traversal Patterns

### Pre-order vs Post-order

```c++
// Pre-order: parent before children
op->walk<WalkOrder::PreOrder>([](Operation *op) { ... });

// Post-order: children before parent (default)
op->walk<WalkOrder::PostOrder>([](Operation *op) { ... });
```

### Filtered Traversal

```c++
// Only walk specific op types
module.walk([](func::FuncOp func) {
  func.walk([](arith::AddIOp add) {
    // Process add operations within functions
  });
});

// Walk with type filter
for (auto op : module.getOps<func::FuncOp>()) {
  // ...
}
```

### Dominance Queries

```c++
DominanceInfo domInfo(moduleOp);

// Check if block A dominates block B
bool dom = domInfo.dominates(blockA, blockB);

// Check if operation A dominates operation B
bool dom = domInfo.dominates(opA, opB);

// Check if value dominates operation
bool dom = domInfo.properlyDominates(value, op);

// Check SSA dominance
bool dom = domInfo.properlyDominates(opA->getBlock(), opB->getBlock());
```

## Common IR Manipulation Patterns

### Creating Operations

```c++
OpBuilder builder(context);
auto loc = builder.getUnknownLoc();

// Using builder methods
auto constant = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(42));
auto add = builder.create<arith::AddIOp>(loc, lhs, rhs);

// Insertion point management
builder.setInsertionPointToEnd(block);
builder.setInsertionPointToStart(block);
builder.setInsertionPointAfter(op);
builder.setInsertionPointBefore(op);
```

### Replacing Values

```c++
// Replace all uses
value.replaceAllUsesWith(newValue);

// Replace uses within specific block
value.replaceUsesWithIf(newValue, [](OpOperand &operand) {
  return operand.getOwner()->getBlock() == targetBlock;
});

// Using Rewriter
rewriter.replaceOp(op, newValues);
rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
```

### Erasing Operations

```c++
// Erase operation (must have no uses)
op->erase();

// Using Rewriter (safer, handles uses)
rewriter.eraseOp(op);

// Erase block contents
block->erase();

// Clear region
region.clear();
```

### Cloning

```c++
// Clone operation
IRMapping mapping;
Operation *clone = op->clone(mapping);

// Clone without regions
Operation *clone = op->cloneWithoutRegions(mapping);

// Clone into different context
OpBuilder builder(destContext);
Operation *clone = builder.clone(*op);
```

## OpBuilder API

```c++
class OpBuilder {
  // === Insertion Point ===
  void setInsertionPoint(Operation *op);
  void setInsertionPointAfter(Operation *op);
  void setInsertionPointBefore(Operation *op);
  void setInsertionPointToStart(Block *block);
  void setInsertionPointToEnd(Block *block);
  InsertPoint saveInsertionPoint();
  void restoreInsertionPoint(InsertPoint ip);

  // === Type Creation ===
  IntegerType getI1Type();
  IntegerType getI8Type();
  IntegerType getI32Type();
  IntegerType getI64Type();
  IntegerType getIntegerType(unsigned width);
  FloatType getF16Type();
  FloatType getF32Type();
  FloatType getF64Type();
  IndexType getIndexType();

  // === Attribute Creation ===
  IntegerAttr getI8IntegerAttr(int8_t value);
  IntegerAttr getI32IntegerAttr(int32_t value);
  IntegerAttr getI64IntegerAttr(int64_t value);
  IntegerAttr getIntegerAttr(Type type, int64_t value);
  FloatAttr getF32FloatAttr(float value);
  FloatAttr getF64FloatAttr(double value);
  StringAttr getStringAttr(StringRef bytes);
  ArrayAttr getArrayAttr(ArrayRef<Attribute> values);
  DictionaryAttr getDictionaryAttr(ArrayRef<NamedAttribute> values);
  DenseElementsAttr getZeroAttr(Type type);

  // === Operation Creation ===
  template <typename OpTy, typename... Args>
  OpTy create(Location loc, Args &&...args);
  Operation *create(const OperationState &state);
  Operation *create(Operation *op);
};
```
