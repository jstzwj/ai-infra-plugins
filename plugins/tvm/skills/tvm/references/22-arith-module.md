# Apache TVM Reference - Chapter 22: Arithmetic Analysis Module (tvm.arith)

This reference covers the arithmetic analysis module in Apache TVM, which provides mathematical analysis utilities for low-level code generation and optimization. The `tvm.arith` module analyzes index arithmetic, value bounds, integer sets, and modular arithmetic properties. It is closely tied to TensorIR passes and scheduling decisions, and serves as the mathematical backbone for proving correctness of transformations.

---

## 22.1 Overview of the Arithmetic Module

The `tvm.arith` module provides a collection of tools for reasoning about integer arithmetic expressions that appear in TIR programs. These expressions typically represent array indices, loop bounds, memory offsets, and tensor shape computations. The ability to simplify, bound, and analyze these expressions is essential for:

- **Loop optimization**: Determining trip counts, proving loop independence, and computing iteration spaces.
- **Memory access analysis**: Computing the range of indices accessed by a statement, checking for out-of-bounds access, and analyzing access patterns.
- **Scheduling decisions**: Proving that transformations like loop splitting, tiling, or reordering preserve program semantics.
- **Code generation**: Generating efficient index arithmetic for tiled and fused loops.

The module is organized around a central `Analyzer` class that combines multiple specialized analysis components, along with standalone data structures and utilities for integer set operations, constraint solving, and pattern detection.

### Module Structure

| Component | Purpose |
|-----------|---------|
| `Analyzer` | Central hub combining all analysis capabilities |
| `IntSet` | Representations of sets of integer values |
| `IntSolver` | Solving linear integer equations and inequalities |
| `ConstIntBound` | Constant integer bound information |
| `ModularSet` | Modular arithmetic property analysis |
| `IterAffineMap` | Affine map detection on iterators |
| Pattern utilities | Arithmetic pattern matching and rewriting |

---

## 22.2 Analyzer (tvm.arith.Analyzer)

The `Analyzer` class is the primary entry point for arithmetic analysis in TVM. It combines multiple sub-analyzers (canonical simplifier, const int bound analyzer, int set analyzer, and modular set analyzer) into a single interface. It also manages a scope stack of variable bindings that allow incremental analysis.

### 22.2.1 Creating an Analyzer

```python
import tvm
from tvm import arith, tir

# Create a fresh analyzer instance
analyzer = arith.Analyzer()
```

The analyzer maintains internal state including variable bindings and cached analysis results. It is designed to be reused across multiple analysis queries within the same scope.

### 22.2.2 Core Methods

#### simplify(expr)

Simplifies an arithmetic expression using canonical simplification. This is the most commonly used method.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()

x = tir.Var("x", "int32")
y = tir.Var("y", "int32")

# Basic algebraic simplification
expr1 = (x + y) - y
print(analyzer.simplify(expr1))  # x

# Constant folding
expr2 = tir.IntImm("int32", 3) * tir.IntImm("int32", 4) + tir.IntImm("int32", 1)
print(analyzer.simplify(expr2))  # 13

# Polynomial simplification
expr3 = x * 2 + x * 3
print(analyzer.simplify(expr3))  # x * 5

# Modular arithmetic simplification
expr4 = (x + 16) % 16
print(analyzer.simplify(expr4))  # x % 16
```

**Signature:**
```python
def simplify(self, expr: tir.PrimExpr) -> tir.PrimExpr
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `tir.PrimExpr` | The expression to simplify |

**Return value:** A simplified `tir.PrimExpr`.

**Details:**

The simplifier works in several stages:
1. Convert the expression to a canonical form (sum of products with common factors collected).
2. Apply algebraic identities (e.g., `x + 0 = x`, `x * 1 = x`, `x * 0 = 0`).
3. Fold constant sub-expressions.
4. Apply modular arithmetic rules when modular set information is available.
5. Use known variable bindings to substitute concrete values or tighter bounds.

The simplifier handles:
- Integer addition, subtraction, multiplication, division, and modulo
- Comparison operators
- Logical AND, OR, NOT
- Min and max operations
- Floor division and ceiling division

#### can_prove(expr)

Attempts to prove that a boolean expression is always true. Returns `True` if the condition can be proven, `False` otherwise. Note that `False` does not mean the condition is false -- it means the analyzer cannot prove it is true.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()

n = tir.Var("n", "int32")

# Cannot prove without binding
print(analyzer.can_prove(n > 0))  # False

# Bind n to a known value
analyzer.bind(n, tir.IntImm("int32", 128))
print(analyzer.can_prove(n > 0))  # True

# Prove algebraic conditions
x = tir.Var("x", "int32")
analyzer.bind(x, tir.IntImm("int32", 10))
print(analyzer.can_prove(x * x >= 0))  # True
```

**Signature:**
```python
def can_prove(self, expr: tir.PrimExpr) -> bool
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `tir.PrimExpr` | A boolean expression to prove |

**Return value:** `True` if the expression can be proven to always hold, `False` otherwise.

**Details:**

The prover uses multiple strategies:
1. **Constant evaluation**: If the expression evaluates to a known constant after simplification.
2. **Bound analysis**: If const int bounds show the expression must be true (e.g., the lower bound of `x` is greater than zero, then `x > 0` is provable).
3. **Simplification**: If the simplified form is a known true value (e.g., `0 == 0`).
4. **Constraint propagation**: Uses known bindings and deduced constraints.

This method is heavily used in optimization passes to guard transformations. For example, before unrolling a loop, a pass will check `can_prove(loop_extent <= threshold)`.

#### const_int_bound(value)

Returns the constant integer bounds for an expression. The result is a `ConstIntBound` object with `min_value` and `max_value` fields.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()

x = tir.Var("x", "int32")

# Without binding, get the type range
bounds = analyzer.const_int_bound(x)
print(f"Lower: {bounds.min_value}")  # -2147483648 (int32 min)
print(f"Upper: {bounds.max_value}")  # 2147483647 (int32 max)

# With a binding
analyzer.bind(x, tir.IntImm("int32", 42))
bounds = analyzer.const_int_bound(x)
print(f"Lower: {bounds.min_value}")  # 42
print(f"Upper: {bounds.max_value}")  # 42

# Expression bounds
n = tir.Var("n", "int32")
bounds = analyzer.const_int_bound(n * 2 + 1)
print(f"Lower: {bounds.min_value}")  # depends on n's range
```

**Signature:**
```python
def const_int_bound(self, value: tir.PrimExpr) -> arith.ConstIntBound
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `tir.PrimExpr` | The expression to bound |

**Return value:** A `ConstIntBound` object with `min_value` and `max_value` attributes.

**Details:**

The bound analyzer tracks ranges through arithmetic operations:
- Addition: ranges are added
- Multiplication: ranges are multiplied (considering sign)
- Division: ranges are divided
- Min/Max: ranges are intersected or unioned as appropriate
- Modular: ranges are reduced modulo the divisor

When no information is available about a variable, its bounds default to the full range of its data type (e.g., `[-2^31, 2^31 - 1]` for `int32`).

#### int_set(expr, dom_map)

Computes the set of possible integer values an expression can take, given a domain map that specifies the range of each variable.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()

x = tir.Var("x", "int32")
y = tir.Var("y", "int32")

# Define domain: x in [0, 127], y in [0, 63]
dom_map = {
    x: arith.IntSet.interval(tir.IntImm("int32", 0), tir.IntImm("int32", 127)),
    y: arith.IntSet.interval(tir.IntImm("int32", 0), tir.IntImm("int32", 63)),
}

# Compute the integer set for x * 2 + y
result = analyzer.int_set(x * 2 + y, dom_map)
print(f"Min: {result.min_value}")  # 0
print(f"Max: {result.max_value}")  # 127 * 2 + 63 = 317
```

**Signature:**
```python
def int_set(self, expr: tir.PrimExpr, dom_map: dict) -> arith.IntSet
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `tir.PrimExpr` | The expression to analyze |
| `dom_map` | `dict[tir.Var, IntSet]` | Map from variables to their integer set domains |

**Return value:** An `IntSet` representing the possible values.

#### bind(var, value)

Binds a variable to a known value or expression. All subsequent analysis calls on this analyzer will use this binding.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()

n = tir.Var("n", "int32")

# Bind n to 256
analyzer.bind(n, tir.IntImm("int32", 256))

# Now analysis uses this binding
print(analyzer.simplify(n // 2))  # 128
print(analyzer.can_prove(n > 100))  # True
```

**Signature:**
```python
def bind(self, var: tir.Var, value: Union[tir.PrimExpr, Range]) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `var` | `tir.Var` | The variable to bind |
| `value` | `PrimExpr` or `Range` | The value or range to bind the variable to |

**Details:**

Binding can be done with a concrete value or with a `Range` object. When binding to a range, the analyzer will use the range bounds for `const_int_bound` queries but will not substitute the variable with a concrete value during simplification.

```python
# Bind to a range
analyzer.bind(i, tvm.ir.Range(0, 128))
# Now const_int_bound(i) returns [0, 127]
```

#### enter() / exit()

Manage the analyzer's scope for variable bindings. Entering a scope pushes a new binding context; exiting pops it and discards all bindings made within that scope.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()
x = tir.Var("x", "int32")

analyzer.bind(x, tir.IntImm("int32", 10))
print(analyzer.simplify(x + 1))  # 11

# Enter a new scope
analyzer.enter()
analyzer.bind(x, tir.IntImm("int32", 20))
print(analyzer.simplify(x + 1))  # 21

# Exit scope -- reverts to x = 10
analyzer.exit()
print(analyzer.simplify(x + 1))  # 11
```

**Usage pattern in TIR passes:**

```python
def analyze_in_scope(analyzer, func, loop_var, extent):
    """Analyze expressions within a loop's scope."""
    analyzer.enter()
    try:
        analyzer.bind(loop_var, tvm.ir.Range(0, extent))
        # Perform analysis with loop variable bound
        result = analyzer.simplify(some_expression)
        return result
    finally:
        analyzer.exit()
```

#### deduce_bound(var, condition, hint_range, dom_map)

Deduces the possible range of a variable given that a condition holds. This is particularly useful for analyzing loop bodies where the condition represents a loop guard or if-condition.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()
i = tir.Var("i", "int32")
n = tir.Var("n", "int32")

# Given that i < n holds, deduce bounds on i
condition = i < n
hint_range = tvm.ir.Range(0, tir.IntImm("int32", 1024))
dom_map = {n: arith.IntSet.interval(tir.IntImm("int32", 1), tir.IntImm("int32", 1024))}

bounds = analyzer.deduce_bound(i, condition, hint_range, dom_map)
# bounds tells us the range i can take when i < n
```

**Signature:**
```python
def deduce_bound(self, var, condition, hint_range=None, dom_map=None) -> IntSet
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `var` | `tir.Var` | The variable to deduce bounds for |
| `condition` | `tir.PrimExpr` | The condition that must hold |
| `hint_range` | `Range` | Optional range hint for the variable |
| `dom_map` | `dict` | Optional domain map for other variables |

**Return value:** An `IntSet` representing the deduced bounds.

---

## 22.3 ConstIntBound

The `ConstIntBound` class represents constant integer bounds for an expression. It stores a minimum and maximum value that the expression is guaranteed to lie within.

### 22.3.1 Structure

```python
class ConstIntBound:
    min_value: int  # Minimum possible value (inclusive)
    max_value: int  # Maximum possible value (inclusive)
```

### 22.3.2 Special Constants

```python
from tvm.arith import ConstIntBound

# Positive and negative infinity (represented as int64 limits)
ConstIntBound.POS_INF  # 2^63 - 1
ConstIntBound.NEG_INF  # -2^63

# Create explicit bounds
bound = ConstIntBound(min_value=0, max_value=127)
print(bound.min_value)  # 0
print(bound.max_value)  # 127
```

### 22.3.3 Bound Propagation Rules

The analyzer propagates bounds through arithmetic operations using the following rules:

| Operation | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| `a + b` | `a.min + b.min` | `a.max + b.max` |
| `a - b` | `a.min - b.max` | `a.max - b.min` |
| `a * b` (both positive) | `a.min * b.min` | `a.max * b.max` |
| `a * b` (mixed sign) | min of cross products | max of cross products |
| `a / b` | depends on signs | depends on signs |
| `a % b` | `0` | `b - 1` (when b positive) |
| `min(a, b)` | `min(a.min, b.min)` | `min(a.max, b.max)` |
| `max(a, b)` | `max(a.min, b.min)` | `max(a.max, b.max)` |
| `-a` | `-a.max` | `-a.min` |

For multiplication with mixed signs, the analyzer computes all four cross products and takes the minimum and maximum:

```python
# Given a in [a_lo, a_hi] and b in [b_lo, b_hi]:
products = [a_lo*b_lo, a_lo*b_hi, a_hi*b_lo, a_hi*b_hi]
lower = min(products)
upper = max(products)
```

### 22.3.4 Usage in Passes

```python
import tvm
from tvm import arith, tir

analyzer = arith.Analyzer()

def check_index_in_bounds(buffer_size, index_expr, analyzer):
    """Verify that an index expression is within buffer bounds."""
    bounds = analyzer.const_int_bound(index_expr)
    if bounds.min_value < 0:
        print(f"Warning: index may be negative (min={bounds.min_value})")
    if bounds.max_value >= buffer_size:
        print(f"Warning: index may overflow (max={bounds.max_value}, size={buffer_size})")
    return bounds.min_value >= 0 and bounds.max_value < buffer_size
```

---

## 22.4 IntSet (tvm.arith.IntSet)

The `IntSet` class represents sets of possible integer values. Unlike `ConstIntBound` which provides a single interval, `IntSet` supports more complex set representations including intervals and sets derived from conditions.

### 22.4.1 Creating IntSets

```python
from tvm import arith, tir

# Single-point set
point = arith.IntSet.single_point(tir.IntImm("int32", 42))

# Interval set [lower, upper]
interval = arith.IntSet.interval(
    tir.IntImm("int32", 0),
    tir.IntImm("int32", 127)
)

# Empty set
empty = arith.IntSet.nothing()

# Universe set (all possible integers)
universe = arith.IntSet.everything()
```

### 22.4.2 IntSet Methods

```python
# Access bounds
iset = arith.IntSet.interval(tir.IntImm("int32", 10), tir.IntImm("int32", 20))
print(iset.min_value)   # 10
print(iset.max_value)   # 20
print(iset.is_nothing)  # False
print(iset.is_everything)  # False
```

### 22.4.3 IntervalSet

`IntervalSet` is the primary concrete implementation of `IntSet`. It represents the set of all integers in a closed interval `[min_value, max_value]`.

```python
from tvm import arith, tir

# Create via the interval factory method
iset = arith.IntSet.interval(
    tir.IntImm("int32", 0),
    tir.IntImm("int32", 255)
)

# Check containment
analyzer = arith.Analyzer()
val = tir.IntImm("int32", 100)
# The IntSet can be used in range analysis
```

### 22.4.4 IntSet from Condition

The `int_set_from_cond` utility constructs an `IntSet` from a boolean condition and a domain:

```python
from tvm.arith import int_set_from_cond

# IntSet of i where 0 <= i < n
i = tir.Var("i", "int32")
n = tir.Var("n", "int32")
condition = i < n
dom = {i: arith.IntSet.interval(tir.IntImm("int32", 0), tir.IntImm("int32", 1023))}

result = int_set_from_cond(condition, i, dom)
```

### 22.4.5 Applications of IntSet

#### Array Index Bounds Checking

```python
def analyze_array_access(analyzer, buffer_size, index_expr, loop_var_ranges):
    """Analyze whether array accesses are always in bounds."""
    # Build domain map from loop variable ranges
    dom_map = {}
    for var, (lo, hi) in loop_var_ranges.items():
        dom_map[var] = arith.IntSet.interval(
            tir.IntImm("int32", lo),
            tir.IntImm("int32", hi)
        )

    # Compute the set of possible index values
    index_set = analyzer.int_set(index_expr, dom_map)

    # Check if all possible values are within bounds
    lower_bound = analyzer.simplify(index_set.min_value)
    upper_bound = analyzer.simplify(index_set.max_value)

    return lower_bound >= 0 and upper_bound < buffer_size
```

#### Loop Iteration Space Analysis

```python
def compute_iteration_space(analyzer, loop_vars, lower_bounds, upper_bounds):
    """Compute the total iteration space of a nested loop."""
    dom_map = {}
    for var, lo, hi in zip(loop_vars, lower_bounds, upper_bounds):
        dom_map[var] = arith.IntSet.interval(lo, hi)

    # The iteration space is the Cartesian product
    total_iterations = tir.IntImm("int32", 1)
    for var, lo, hi in zip(loop_vars, lower_bounds, upper_bounds):
        extent = analyzer.simplify(hi - lo + 1)
        total_iterations = total_iterations * extent

    return analyzer.simplify(total_iterations), dom_map
```

#### Dependence Analysis

```python
def check_independence(analyzer, var, expr1, expr2, dom_map):
    """Check if two accesses with the same pattern are independent.

    Returns True if for all values of var in dom_map,
    expr1 and expr2 access different locations.
    """
    diff = analyzer.simplify(expr1 - expr2)
    diff_set = analyzer.int_set(diff, dom_map)

    # If the minimum is positive or maximum is negative, no overlap
    if analyzer.can_prove(diff_set.min_value > 0):
        return True
    if analyzer.can_prove(diff_set.max_value < 0):
        return True
    return False
```

---

## 22.5 IntSolver (tvm.arith.IntSolver)

The `IntSolver` module provides utilities for solving systems of linear integer equations and inequalities. This is used for dependence analysis, loop transformation validation, and affine analysis.

### 22.5.1 solve_linear_inequality

Solves a system of linear inequalities over integer variables.

```python
from tvm.arith import IntSolver
from tvm import tir

# Define variables
i = tir.Var("i", "int32")
j = tir.Var("j", "int32")

# Define constraints: 0 <= i < 128 and 0 <= j < 64 and i + j < 128
constraints = [
    i >= 0,
    i < 128,
    j >= 0,
    j < 64,
    i + j < 128,
]

# Solve
result = IntSolver.solve_linear_inequality(constraints, [i, j])
```

### 225.2 solve_linear_equations

Solves a system of linear equations over integer variables.

```python
from tvm.arith import IntSolver
from tvm import tir

# Variables
x = tir.Var("x", "int32")
y = tir.Var("y", "int32")

# Equations: x + 2*y = 10, 3*x - y = 5
equations = [
    x + y * 2 - 10,
    x * 3 - y - 5,
]

result = IntSolver.solve_linear_equations(equations, [x, y])
# Returns the solution space (may be parametric)
```

### 22.5.3 Internal Representation

The solver works by converting constraints to a normal form and applying integer linear programming techniques:

1. **Normalization**: All inequalities are converted to the form `a1*x1 + a2*x2 + ... >= 0`.
2. **Fourier-Motzkin elimination**: Eliminates variables one at a time to check feasibility.
3. **Parametric solution**: When the system is under-determined, returns a parametric solution.

### 22.5.4 Applications

```python
def check_loop_carried_dependence(analyzer, i, j, access_i, access_j):
    """Check if there is a loop-carried dependence between two iterations.

    access_i: index at iteration i
    access_j: index at iteration j

    Returns True if there exists (i, j) with i < j such that access_i == access_j.
    """
    # We need to solve: access_i == access_j and i < j
    # within the loop bounds
    diff = access_i - access_j
    # If we can prove diff != 0 for all valid i < j, no dependence
    condition = tir.And(diff == 0, i < j)

    # Use the solver to check if the system is feasible
    constraints = [diff == 0, i < j, i >= 0, j >= 0]
    result = IntSolver.solve_linear_inequality(constraints, [i, j])
    return result is not None
```

---

## 22.6 Modular Set Analysis

### 22.6.1 ModularSet

The `ModularSet` class represents properties of expressions involving modular arithmetic. It tracks the relationship `value = base + coeff * var` where the expression is analyzed modulo a number.

```python
from tvm import arith, tir

analyzer = arith.Analyzer()

# The modular set of (x * 4) is {base=0, coeff=4} modulo the type range
x = tir.Var("x", "int32")
mod_info = analyzer.modular_set(x * 4)
print(f"Base: {mod_info.base}")    # 0
print(f"Coeff: {mod_info.coeff}")  # 4
```

### 22.6.2 Structure

```python
class ModularSet:
    base: int    # The constant offset
    coeff: int   # The multiplicative coefficient
```

A `ModularSet(base, coeff)` represents the set of values `{base + coeff * k | k in Z}`.

### 22.6.3 Applications of Modular Set Analysis

Modular set analysis is particularly important for:

1. **Index alignment**: Determining if array indices are aligned to certain boundaries (e.g., 16-byte alignment for vector loads).

2. **Simplifying modular expressions**: Reducing `(x * 4 + 2) % 4` to `2` when the modular set of `x * 4 + 2` is known.

3. **Vectorization checks**: Verifying that loop iterations access aligned memory addresses.

```python
def check_alignment(analyzer, index_expr, alignment):
    """Check if an index expression is always aligned to the given boundary."""
    mod_info = analyzer.modular_set(index_expr)
    # The expression is aligned if base % alignment == 0 and coeff % alignment == 0
    return mod_info.base % alignment == 0 and mod_info.coeff % alignment == 0
```

### 22.6.4 Propagation Rules

| Expression | Modular Set |
|------------|-------------|
| `a + b` | `(base_a + base_b, coeff_a + coeff_b)` |
| `a - b` | `(base_a - base_b, coeff_a - coeff_b)` |
| `a * c` (c constant) | `(base_a * c, coeff_a * c)` |
| `a % m` | `(base_a % m, gcd(coeff_a, m))` |

---

## 22.7 IterAffineMap

The `IterAffineMap` represents affine mappings on iterators. It is used to detect and represent patterns in loop iterations that can be expressed as affine transformations.

### 22.7.1 detect_iter_affine_map

Detects whether a set of iterator definitions can be expressed as an affine map over a set of source iterators.

```python
from tvm import arith, tir

# Simple case: detect that j = i * 4 + r is affine over i and r
i = tir.Var("i", "int32")
j = tir.Var("j", "int32")
r = tir.Var("r", "int32")

# j = i * 4 + r
iter_map = arith.detect_iter_affine_map(
    j,          # target iterator
    [i, r],     # source iterators
    {i: tvm.ir.Range(0, 32), r: tvm.ir.Range(0, 4)},
)
```

### 22.7.2 SubspaceDivide

Decomposes an index expression into a subspace defined by a set of iterators and the remaining components.

```python
from tvm.arith import subspace_divide

# Given expression e and iterators subspace_iters,
# decompose into (subspace_component, other_component)
i = tir.Var("i", "int32")
j = tir.Var("j", "int32")
k = tir.Var("k", "int32")

expr = i * 64 + j * 8 + k
subspace = [i, j]

result = subspace_divide(expr, subspace, dom_map)
# Returns the components of the expression that depend on i, j
# and the component that depends only on k
```

### 22.7.3 normalize_to_iter_sum

Normalizes an expression to a sum of scaled iterators plus a constant offset.

```python
from tvm.arith import normalize_to_iter_sum

i = tir.Var("i", "int32")
j = tir.Var("j", "int32")

expr = i * 128 + j * 4 + 7
result = normalize_to_iter_sum(expr, {i: tvm.ir.Range(0, 16), j: tvm.ir.Range(0, 32)})
# Represents the expression as a sum of iterator terms
```

### 22.7.4 Applications in Loop Transformations

```python
def analyze_tiling_pattern(analyzer, outer_idx, inner_idx, tile_size):
    """Analyze whether an index pattern corresponds to a tiled loop.

    Given outer loop variable i and inner loop variable j,
    check if some index is an affine function of i*tile_size + j.
    """
    combined = outer_idx * tile_size + inner_idx

    # Check if this is affine over the original loop variable
    src_var = tir.Var("original", "int32")
    result = arith.detect_iter_affine_map(
        combined,
        [src_var],
        {src_var: tvm.ir.Range(0, outer_extent * tile_size)},
    )
    return result is not None
```

---

## 22.8 Canonical Simplification

The canonical simplification pass converts expressions into a canonical form that enables algebraic simplification. This is the engine behind `Analyzer.simplify()`.

### 22.8.1 canonical_simplify

Applies canonical simplification rules to an expression.

```python
from tvm.arith import canonical_simplify
from tvm import tir

x = tir.Var("x", "int32")

# Direct usage (Analyzer.simplify calls this internally)
expr = (x * 2 + x * 3)
result = canonical_simplify(expr)
# Result: x * 5
```

### 22.8.2 Canonical Form Rules

The canonicalizer applies the following transformations:

**Polynomial Normalization:**

| Input | Output | Rule |
|-------|--------|------|
| `x + x` | `x * 2` | Combine like terms |
| `x * 2 + x * 3` | `x * 5` | Collect coefficients |
| `(x + 1) * 2` | `x * 2 + 2` | Distribute |
| `(x + 1) * (x - 1)` | `x * x - 1` | Expand and simplify |

**Division Simplification:**

| Input | Output | Rule |
|-------|--------|------|
| `(x * 4) / 2` | `x * 2` | Cancel factors |
| `(x * 4 + 2) / 2` | `x * 2 + 1` | Distribute division |
| `(x + 1) / 2 * 2` | depends on x | Requires bound info |

**Modular Simplification:**

| Input | Output | Rule |
|-------|--------|------|
| `(x * 4) % 4` | `0` | Factor divides modulus |
| `(x * 4 + 1) % 4` | `1` | Separate coefficient |
| `x % 4 % 2` | `x % 2` | Nested modulo reduction |

**Comparison Simplification:**

| Input | Output | Rule |
|-------|--------|------|
| `x + 1 > x` | `True` | Always true |
| `x * 2 == x * 2` | `True` | Identity |
| `x < x + 1` | `True` | Offset comparison |

### 22.8.3 Handling Corner Cases

```python
# Division by zero protection
from tvm import arith, tir

analyzer = arith.Analyzer()
x = tir.Var("x", "int32")

# FloorDiv rounds toward negative infinity
expr = tir.FloorDiv(x, tir.IntImm("int32", 4))
result = analyzer.simplify(expr)
# Preserves the FloorDiv if x's range is unknown

# With known bounds
analyzer.bind(x, tir.IntImm("int32", 7))
result = analyzer.simplify(tir.FloorDiv(x, tir.IntImm("int32", 4)))
# Result: 1
```

---

## 22.9 Pattern Matching Utilities

The `tvm.arith` module includes pattern matching utilities for detecting common arithmetic patterns.

### 22.9.1 detect_linear_equation

Detects if an expression can be written as a linear equation in a given variable.

```python
from tvm.arith import detect_linear_equation
from tvm import tir

x = tir.Var("x", "int32")
y = tir.Var("y", "int32")

# Detect that 2*x + 3*y + 1 is linear in x
result = detect_linear_equation(2 * x + 3 * y + 1, x)
# Returns information about the coefficients
```

### 22.9.2 DomainTouched

Computes the region of a buffer that is touched (read or written) by a statement.

```python
from tvm.tir.analysis import identify_region
from tvm import arith

# Given a buffer and a statement, compute the touched region
# This uses IntSet analysis internally
```

### 22.9.3 Common Subexpression Patterns

```python
from tvm import arith, tir

analyzer = arith.Analyzer()
x = tir.Var("x", "int32")

# Detect if two expressions are equivalent
expr1 = x * 4 + x * 3
expr2 = x * 7

# After simplification, they should be equal
s1 = analyzer.simplify(expr1)
s2 = analyzer.simplify(expr2)
# Can compare simplified forms
```

---

## 22.10 Integration with TIR Passes

The arithmetic module is used extensively throughout TVM's compilation pipeline. This section describes how TIR passes interact with the arithmetic analyzer.

### 22.10.1 Loop Bound Analysis

```python
import tvm
from tvm import tir, arith
from tvm.script import ir as I, tir as T

@I.ir_module
class LoopExample:
    @T.prim_func
    def tiled_matmul(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        for i_outer, j_outer, k_outer in T.grid(4, 4, 4):
            for i_inner, j_inner, k_inner in T.grid(32, 32, 32):
                with T.block("C"):
                    vi = i_outer * 32 + i_inner
                    vj = j_outer * 32 + j_inner
                    vk = k_outer * 32 + k_inner
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# The arithmetic analyzer can prove that vi, vj, vk are in [0, 127]
analyzer = arith.Analyzer()
i_outer = tir.Var("i_outer", "int32")
i_inner = tir.Var("i_inner", "int32")

analyzer.bind(i_outer, tvm.ir.Range(0, 4))
analyzer.bind(i_inner, tvm.ir.Range(0, 32))

vi = i_outer * 32 + i_inner
bounds = analyzer.const_int_bound(vi)
print(f"vi range: [{bounds.min_value}, {bounds.max_value}]")
# vi range: [0, 127]
```

### 22.10.2 Index Simplification in Scheduling

The MetaSchedule and DLight use the arithmetic analyzer extensively to simplify index expressions after scheduling transformations:

```python
from tvm import arith, tir

def simplify_scheduled_indices(scheduled_func):
    """Simplify all index expressions in a scheduled PrimFunc."""
    analyzer = arith.Analyzer()

    def simplify_expr(expr):
        return analyzer.simplify(expr)

    # Apply simplification to all buffer access indices
    # This is what SimplifyIndex pass does internally
    # ...
```

### 22.10.3 Memory Access Pattern Analysis

```python
from tvm import arith, tir

def analyze_strided_access(analyzer, base_expr, stride, element_size):
    """Analyze whether memory accesses are contiguous or strided.

    Returns 'contiguous' if consecutive iterations access consecutive
    addresses, 'strided' otherwise.
    """
    # Check if base_expr + stride is a simple increment
    diff = analyzer.simplify(stride)
    if analyzer.can_prove(diff == element_size):
        return "contiguous"
    else:
        return "strided"
```

### 22.10.4 Vectorization Safety Checks

```python
from tvm import arith, tir

def check_vectorizable(analyzer, loop_var, body, vector_width):
    """Check if a loop body can be safely vectorized.

    Conditions:
    1. All memory accesses are aligned to vector_width * sizeof(element)
    2. No loop-carried dependencies
    3. Index increment is 1 per iteration
    """
    # Bind loop variable to check properties
    analyzer.enter()
    try:
        analyzer.bind(loop_var, tvm.ir.Range(0, vector_width))

        # Check alignment of memory accesses
        # (would traverse body and check each access)
        # ...

        return True  # or False based on analysis
    finally:
        analyzer.exit()
```

### 22.10.5 SimplifyExpr Pass

The `SimplifyExpr` TIR pass applies arithmetic simplification to all expressions in a PrimFunc:

```python
import tvm
from tvm import tir

# The pass is typically applied after scheduling
mod = tir.transform.SimplifyExpr()(mod)

# Internally, it creates an Analyzer and calls simplify on every
# PrimExpr in the IR:
# - Loop bounds
# - Buffer access indices
# - If conditions
# - Store values
# - Let bindings
```

---

## 22.11 Advanced Examples

### 22.11.1 Proving Loop Independence

```python
from tvm import arith, tir

def prove_loop_independence(analyzer, i, j, expr_i, expr_j, loop_range):
    """Prove that iterations i and j access different memory locations.

    This is the foundation of parallelization safety.
    """
    analyzer.enter()
    try:
        # Bind both loop variables
        analyzer.bind(i, loop_range)
        analyzer.bind(j, loop_range)

        # Compute the difference in access indices
        diff = analyzer.simplify(expr_i - expr_j)

        # Try to prove diff != 0 when i != j
        if analyzer.can_prove(diff != 0):
            return True  # Independent

        # For more complex cases, check bounds
        diff_bounds = analyzer.const_int_bound(diff)
        if diff_bounds.min_value > 0 or diff_bounds.max_value < 0:
            return True  # Independent

        return False  # Cannot prove independence
    finally:
        analyzer.exit()
```

### 22.11.2 Workspace Size Calculation

```python
from tvm import arith, tir

def compute_workspace_bounds(analyzer, buffer_exprs, loop_vars, loop_ranges):
    """Compute the total workspace needed by analyzing buffer index ranges."""
    total_size = 0

    for buf_name, index_expr in buffer_exprs.items():
        # Build domain map from loop ranges
        dom_map = {}
        for var, rng in zip(loop_vars, loop_ranges):
            dom_map[var] = arith.IntSet.interval(rng.min, rng.min + rng.extent - 1)

        # Compute the range of indices accessed
        index_set = analyzer.int_set(index_expr, dom_map)
        min_idx = analyzer.simplify(index_set.min_value)
        max_idx = analyzer.simplify(index_set.max_value)

        # Size needed is max - min + 1
        size = analyzer.simplify(max_idx - min_idx + 1)
        total_size = total_size + size

    return total_size
```

### 22.11.3 Tiled Index Simplification

```python
from tvm import arith, tir

def simplify_tiled_index(analyzer, tile_i, tile_j, inner_i, inner_j, tile_size_i, tile_size_j):
    """Simplify the reconstructed index after tiling.

    Original: index = i * width + j
    After tiling: i = tile_i * tile_size + inner_i
                  j = tile_j * tile_size + inner_j
    """
    # Bind tile and inner indices
    analyzer.enter()
    try:
        analyzer.bind(tile_i, tvm.ir.Range(0, 128 // tile_size_i))
        analyzer.bind(inner_i, tvm.ir.Range(0, tile_size_i))
        analyzer.bind(tile_j, tvm.ir.Range(0, 64 // tile_size_j))
        analyzer.bind(inner_j, tvm.ir.Range(0, tile_size_j))

        # Reconstructed index
        i = tile_i * tile_size_i + inner_i
        j = tile_j * tile_size_j + inner_j
        index = i * 64 + j

        # Simplify
        simplified = analyzer.simplify(index)
        return simplified
    finally:
        analyzer.exit()
```

### 22.11.4 Proving Optimization Safety

```python
from tvm import arith, tir

def can_safely_reorder(analyzer, var_i, var_j, reads_i, reads_j, writes_i, writes_j):
    """Determine if two loops can be safely reordered.

    Loops can be reordered if there is no loop-carried true or anti-dependence
    between the two iteration spaces.
    """
    analyzer.enter()
    try:
        # Check true dependence: write at (i1, j1) and read at (i2, j2)
        # where i1 != i2 or j1 != j2, but the same memory location
        for write_expr in writes_i:
            for read_expr in reads_j:
                diff = analyzer.simplify(write_expr - read_expr)
                if not analyzer.can_prove(diff != 0):
                    return False  # Possible dependence

        # Check anti-dependence: read at (i1, j1) and write at (i2, j2)
        for read_expr in reads_i:
            for write_expr in writes_j:
                diff = analyzer.simplify(read_expr - write_expr)
                if not analyzer.can_prove(diff != 0):
                    return False  # Possible dependence

        return True  # Safe to reorder
    finally:
        analyzer.exit()
```

### 22.11.5 Shape Computation in Relax

The arithmetic analyzer is also used in Relax for shape computation and simplification:

```python
import tvm
from tvm import arith, tir, relax

def simplify_shape_expr(shape_expr):
    """Simplify a Relax shape expression using arithmetic analysis."""
    analyzer = arith.Analyzer()

    simplified_values = []
    for dim in shape_expr:
        if isinstance(dim, tir.PrimExpr):
            simplified_values.append(analyzer.simplify(dim))
        else:
            simplified_values.append(dim)

    return simplified_values
```

---

## 22.12 Relationship to Other TVM Components

### 22.12.1 Relationship to TIR Analysis

The `tvm.arith` module provides the mathematical foundation that `tvm.tir.analysis` builds upon:

| TIR Analysis | Uses arith Module For |
|--------------|----------------------|
| `calculate_workspace` | IntSet to compute buffer access ranges |
| `estimate_tir_flops` | Simplification of arithmetic expressions |
| `verify_gpu_code` | ConstIntBound for index bounds checking |
| `OOBChecker` | IntSet for access region analysis |

### 22.12.2 Relationship to MetaSchedule

MetaSchedule uses the arithmetic analyzer extensively during auto-tuning:

- **Post-condition checking**: After applying a scheduling primitive, uses `can_prove` to verify the transformation preserved correctness.
- **Index simplification**: After tiling, splitting, and fusing, simplifies the resulting index expressions.
- **Bound computation**: Computes the iteration space of scheduled loops for cost modeling.

### 22.12.3 Relationship to DLight

DLight rule implementations rely on the arithmetic analyzer for:

- **Pattern detection**: Using `detect_iter_affine_map` to identify loop patterns that match optimization rules.
- **Safety validation**: Proving that transformations like vectorization are safe for the detected pattern.
- **Code generation**: Simplifying index expressions in the generated optimized code.

### 22.12.4 Relationship to Relax

The Relax frontend uses arithmetic analysis for:

- **Shape inference**: Simplifying symbolic shape expressions during type checking.
- **Operator lowering**: Computing the sizes of intermediate tensors during operator decomposition.
- **Memory planning**: Analyzing memory access patterns for layout optimization.

---

## 22.13 Performance Considerations

### 22.13.1 Analyzer Reuse

Creating a new `Analyzer` is relatively expensive due to internal data structure initialization. When performing many analysis queries, reuse a single analyzer instance:

```python
# Good: reuse analyzer
analyzer = arith.Analyzer()
for expr in expressions:
    result = analyzer.simplify(expr)

# Bad: create new analyzer each time (unnecessary overhead)
for expr in expressions:
    analyzer = arith.Analyzer()
    result = analyzer.simplify(expr)
```

### 22.13.2 Scope Management

Always use `enter()`/`exit()` in a try/finally block to ensure scopes are properly cleaned up:

```python
analyzer = arith.Analyzer()

analyzer.enter()
try:
    analyzer.bind(var, value)
    result = analyzer.simplify(expr)
finally:
    analyzer.exit()
```

### 22.13.3 Binding vs. Substitution

For one-time substitution, it may be faster to use `tir.stmt_functor.substitute` directly. Binding is more efficient when the same binding is used for multiple analysis queries:

```python
# Efficient for single query
from tvm.tir import stmt_functor
result = stmt_functor.substitute(expr, {var: value})

# Efficient for multiple queries with same binding
analyzer.bind(var, value)
result1 = analyzer.simplify(expr1)
result2 = analyzer.simplify(expr2)
result3 = analyzer.can_prove(condition)
```

### 22.13.4 Caching

The analyzer caches intermediate results internally. This means that repeated queries with the same or similar expressions benefit from caching. However, the cache is invalidated when bindings change, so it is best to batch queries that share the same binding context.

---

## 22.14 API Reference Summary

### Analyzer Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `simplify` | `(expr) -> PrimExpr` | Simplify arithmetic expression |
| `can_prove` | `(expr) -> bool` | Prove a boolean condition |
| `const_int_bound` | `(expr) -> ConstIntBound` | Get constant bounds |
| `int_set` | `(expr, dom_map) -> IntSet` | Compute integer set |
| `bind` | `(var, value) -> None` | Bind variable to value |
| `enter` | `() -> None` | Push scope |
| `exit` | `() -> None` | Pop scope |
| `modular_set` | `(expr) -> ModularSet` | Get modular set properties |
| `deduce_bound` | `(var, cond, hint, dom) -> IntSet` | Deduce bounds from condition |

### IntSet Static Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `interval` | `(min, max) -> IntSet` | Create interval set |
| `single_point` | `(value) -> IntSet` | Create single-point set |
| `nothing` | `() -> IntSet` | Create empty set |
| `everything` | `() -> IntSet` | Create universal set |

### Standalone Functions

| Function | Description |
|----------|-------------|
| `canonical_simplify(expr)` | Apply canonical simplification |
| `detect_iter_affine_map(...)` | Detect affine iterator patterns |
| `subspace_divide(...)` | Decompose into subspace components |
| `normalize_to_iter_sum(...)` | Normalize to sum of iterators |
| `detect_linear_equation(...)` | Detect linear equation pattern |

---

## 22.15 Common Patterns and Recipes

### 22.15.1 Proving a Loop Count is a Power of 2

```python
from tvm import arith, tir
import math

def is_power_of_2(analyzer, expr):
    """Check if the value of expr is always a power of 2."""
    bounds = analyzer.const_int_bound(expr)
    if bounds.min_value == bounds.max_value:
        val = bounds.min_value
        return val > 0 and (val & (val - 1)) == 0
    return False
```

### 22.15.2 Computing Access Stride

```python
def compute_stride(analyzer, index_expr, loop_var, step=1):
    """Compute the stride of index_expr with respect to loop_var.

    Returns the difference in index when loop_var changes by step.
    """
    from tvm import tir
    delta = tir.Substitute(index_expr, {loop_var: loop_var + step}) - index_expr
    return analyzer.simplify(delta)
```

### 22.15.3 Checking Divisibility

```python
def is_divisible(analyzer, expr, divisor):
    """Check if expr is always divisible by divisor."""
    remainder = analyzer.simplify(expr % divisor)
    return analyzer.can_prove(remainder == 0)
```

### 22.15.4 Range Intersection

```python
def intersect_ranges(analyzer, range1, range2):
    """Compute the intersection of two integer ranges."""
    min_val = analyzer.simplify(tir.max(range1.min_value, range2.min_value))
    max_val = analyzer.simplify(tir.min(range1.max_value, range2.max_value))

    if analyzer.can_prove(min_val <= max_val):
        return arith.IntSet.interval(min_val, max_val)
    else:
        return arith.IntSet.nothing()
```

### 22.15.5 Estimating Trip Count

```python
def estimate_trip_count(analyzer, lower, upper, step):
    """Estimate the number of iterations for a loop with given bounds and step."""
    from tvm import tir
    if step > 0:
        range_expr = upper - lower
    else:
        range_expr = lower - upper

    abs_step = analyzer.simplify(tir.abs(step))
    trip_count = analyzer.simplify(tir.floordiv(range_expr, abs_step))
    return trip_count
```

---

## 22.16 Troubleshooting

### 22.16.1 Simplification Does Not Simplify

If `analyzer.simplify(expr)` returns the same expression unchanged:

1. **Missing bindings**: The expression contains variables that the analyzer has no information about. Bind them using `analyzer.bind(var, value_or_range)`.
2. **Non-linear expression**: The simplifier primarily handles linear and affine expressions. Non-linear expressions (e.g., `x * y` where both are variables) may not simplify fully.
3. **Complex control flow**: Expressions involving conditional operations (`tir.Select`, `tir.IfThenElse`) may not simplify without knowledge of the condition.

### 22.16.2 can_prove Returns False for True Conditions

This is expected behavior. `can_prove` is conservative -- it only returns `True` when it can construct a proof. Many true conditions cannot be proven by the analyzer because:

1. The condition involves non-linear arithmetic.
2. The required variable bindings are missing.
3. The proof would require multi-variable induction.

### 22.16.3 Performance Issues

If analysis is slow:

1. Reduce the number of variables in scope by using `enter()`/`exit()` to limit binding context.
2. Simplify expressions before passing them to the analyzer.
3. Avoid creating new `Analyzer` instances in tight loops.

---

## 22.17 Historical Notes and Design Rationale

The `tvm.arith` module was designed to be a self-contained mathematical analysis toolkit that could be used independently of the TIR IR. This separation allows:

1. **Reuse**: Other TVM components (Relax, MetaSchedule, DLight) can use arithmetic analysis without depending on TIR.
2. **Testing**: Arithmetic analysis can be tested in isolation without constructing full TIR programs.
3. **Extensibility**: New analysis capabilities can be added without modifying the IR or pass infrastructure.

The analyzer's scope-based binding model was inspired by SMT solver contexts, where assertions and bindings are pushed and popped to manage incremental analysis. This pattern is particularly well-suited to the hierarchical nature of loop nests, where inner loops inherit bindings from outer loops but may add their own.
